//! Trajectory data types for the photom crate.
//!
//! A *trajectory* is an ordered sequence of [`Observation`]s that all
//! originate from the same moving object (asteroid, comet, near-Earth object,
//! etc.).  This module defines the three core types used to represent and
//! query trajectories:
//!
//! - [`TrajId`] — a typed identifier for a trajectory, either an integer or a
//!   string label.
//! - [`Trajectory`] — a single trajectory: its identifier and the ordered list
//!   of [`ObsId`]s that belong to it.
//! - [`TrajDataset`] — a dataset that owns both a complete [`ObsDataset`] (the
//!   source of truth for all observations and observer metadata) and the
//!   grouping of those observations into trajectories.
//!
//! ## Key design notes
//!
//! ### Observations are not duplicated
//!
//! A [`Trajectory`] stores only [`ObsId`]s — lightweight `u64` keys — rather
//! than cloned [`Observation`] values.  All observation data lives in the
//! embedded [`ObsDataset`], which is the single source of truth.  Resolving an
//! [`ObsId`] to a full [`Observation`] always goes through
//! [`TrajDataset::get_observation`], which forwards to the LRU cache inside
//! [`ObsDataset`].
//!
//! ### LRU cache on trajectories
//!
//! [`TrajDataset`] maintains its own LRU cache over [`Trajectory`] values,
//! mirroring the cache that [`ObsDataset`] keeps over individual
//! [`Observation`]s.  The same `lru_cache_size` parameter controls **both**
//! caches so that callers have a single knob.
//!
//! ### Nullable `traj_id` column
//!
//! When loading from a Polars `DataFrame`, the `traj_id` column is optional
//! and nullable.  Rows whose `traj_id` cell is `null` are ingested into the
//! [`ObsDataset`] normally but are not assigned to any trajectory; they remain
//! accessible via [`TrajDataset::get_observation`] and
//! [`TrajDataset::obs_dataset`].
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`TrajId`] | enum | Typed trajectory identifier (integer or string) |
//! | [`Trajectory`] | struct | A single trajectory and its observation IDs |
//! | [`TrajDataset`] | struct | Full dataset with observations grouped into trajectories |

use std::num::NonZeroUsize;

use lru::LruCache;

#[cfg(feature = "polars")]
use crate::io::polars::{error::PolarsError, load_traj_from_polars};
#[cfg(feature = "polars")]
use polars::{frame::DataFrame, lazy::frame::LazyFrame};

use crate::observation::{ObsDataset, ObsId, Observation};
#[cfg(feature = "polars")]
use crate::observer::error_model::ObsErrorModel;

// ── TrajId ────────────────────────────────────────────────────────────────────

/// Typed identifier for a single trajectory.
///
/// A trajectory can be identified either by a **64-bit unsigned integer** (e.g.
/// a running index or a catalogue number) or by a **string label** (e.g. a
/// Minor Planet Center provisional designation such as `"2020 AV2"`, or a
/// proper name such as `"Ceres"`).
///
/// The column type of `traj_id` in the source `DataFrame` determines which
/// variant is used: a `UInt64` column produces [`TrajId::Int`] keys and a
/// `String` column produces [`TrajId::Str`] keys.  Mixing both types in a
/// single dataset is not supported; the column must be uniformly one type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TrajId {
    /// A 64-bit unsigned integer identifier (e.g. a catalogue number).
    Int(u64),
    /// A string label (e.g. a MPC provisional designation or a proper name).
    Str(String),
}

// ── Trajectory ────────────────────────────────────────────────────────────────

/// A single trajectory: an ordered list of observation IDs associated with one
/// moving object.
///
/// `Trajectory` stores only the [`ObsId`] keys; the actual [`Observation`]
/// data lives in the parent [`TrajDataset`]'s embedded [`ObsDataset`].
/// Resolve an ID to a full observation via [`TrajDataset::get_observation`].
///
/// The observation IDs are stored in the order they were encountered while
/// reading the source `DataFrame` (i.e. row order), which typically
/// corresponds to ascending epoch order if the frame was pre-sorted.
#[derive(Clone, Debug)]
pub struct Trajectory {
    /// Unique identifier for this trajectory within its dataset.
    pub id: TrajId,

    /// Ordered list of observation IDs that belong to this trajectory.
    ///
    /// Each value is a key into the parent [`TrajDataset`]'s [`ObsDataset`].
    /// The order matches the source `DataFrame` row order.
    pub obs_ids: Vec<ObsId>,
}

// ── TrajDataset ───────────────────────────────────────────────────────────────

/// A dataset that groups observations into trajectories.
///
/// `TrajDataset` owns a complete [`ObsDataset`] (all observations, custom
/// geodetic observers, MPC lazy table, and the per-observation LRU cache) and
/// adds a layer of trajectory grouping on top of it.
///
/// Observations whose `traj_id` was `null` in the source `DataFrame` are
/// present in the embedded [`ObsDataset`] but do not belong to any trajectory.
/// They can still be accessed via [`TrajDataset::get_observation`] or by
/// iterating [`TrajDataset::obs_dataset`].
pub struct TrajDataset {
    /// The underlying observation dataset.
    ///
    /// This is the single source of truth for all [`Observation`] data,
    /// custom geodetic [`crate::observer::Observer`]s, and the MPC lookup
    /// table.  It also owns the per-observation LRU cache.
    obs_dataset: ObsDataset,

    /// All trajectories in insertion order (i.e. order of first appearance of
    /// each `traj_id` value in the source `DataFrame`).
    trajectories: Vec<Trajectory>,

    /// LRU cache keyed by [`TrajId`].
    ///
    /// Mirrors the per-observation LRU cache inside [`ObsDataset`]: a
    /// [`Trajectory`] is cloned into the cache on first access via
    /// [`TrajDataset::get_trajectory`] and evicted in least-recently-used
    /// order when the cache is full.
    lru_cache_traj: LruCache<TrajId, Trajectory>,
}

impl TrajDataset {
    /// Construct a [`TrajDataset`] from a Polars [`DataFrame`].
    ///
    /// The frame must satisfy the same base-column schema as
    /// [`ObsDataset::from_polars`] and may additionally contain a `traj_id`
    /// column.  See [`crate::io::polars`] for the full schema.
    ///
    /// ## `traj_id` column rules
    ///
    /// | Situation | Outcome |
    /// |-----------|---------|
    /// | Column absent | All observations are loaded; no trajectories are created. |
    /// | Column present, type `UInt64` | Non-null cells produce [`TrajId::Int`] keys. |
    /// | Column present, type `String` | Non-null cells produce [`TrajId::Str`] keys. |
    /// | Column present, other type | [`PolarsError::TrajIdColumnTypeError`] is returned. |
    /// | Cell is `null` | The observation belongs to no trajectory. |
    ///
    /// # Arguments
    ///
    /// - `df`             — source Polars [`DataFrame`].
    /// - `error_model`    — astrometric error model forwarded to [`ObsDataset`].
    /// - `lru_cache_size` — shared capacity for **both** the observation LRU
    ///   cache (inside [`ObsDataset`]) and the trajectory LRU cache.  Defaults
    ///   to 1 000 when `None`.
    ///
    /// # Errors
    ///
    /// Returns a [`PolarsError`] if the base-column schema is violated, if any
    /// observer column rule is broken, or if the `traj_id` column has an
    /// unsupported type.
    #[cfg(feature = "polars")]
    pub fn from_polars(
        df: &DataFrame,
        error_model: ObsErrorModel,
        lru_cache_size: Option<usize>,
    ) -> Result<Self, PolarsError> {
        load_traj_from_polars(df, error_model, lru_cache_size)
    }

    /// Construct a [`TrajDataset`] from a Polars [`LazyFrame`].
    ///
    /// The lazy computation plan is executed (via [`LazyFrame::collect`]) before
    /// ingestion begins.  Once collected, the same validation and assembly
    /// pipeline as [`TrajDataset::from_polars`] is applied.
    ///
    /// # Arguments
    ///
    /// - `lf`             — source Polars [`LazyFrame`].
    /// - `error_model`    — astrometric error model forwarded to [`ObsDataset`].
    /// - `lru_cache_size` — shared capacity for **both** the observation LRU
    ///   cache and the trajectory LRU cache.  Defaults to 1 000 when `None`.
    ///
    /// # Errors
    ///
    /// Returns [`PolarsError::Polars`] if the lazy plan fails to execute, plus
    /// all errors documented on [`TrajDataset::from_polars`].
    #[cfg(feature = "polars")]
    pub fn from_lazy(
        lf: LazyFrame,
        error_model: ObsErrorModel,
        lru_cache_size: Option<usize>,
    ) -> Result<Self, PolarsError> {
        load_traj_from_polars(lf, error_model, lru_cache_size)
    }

    /// Build a [`TrajDataset`] from pre-parsed components.
    ///
    /// Used internally by [`load_traj_from_polars`].  The trajectory LRU cache
    /// is initialised with the same capacity as the observation LRU cache
    /// inside `obs_dataset`.
    pub(crate) fn new(
        obs_dataset: ObsDataset,
        trajectories: Vec<Trajectory>,
        lru_cache_size: Option<usize>,
    ) -> Self {
        let capacity = NonZeroUsize::new(lru_cache_size.unwrap_or(1000)).unwrap();
        Self {
            obs_dataset,
            trajectories,
            lru_cache_traj: LruCache::new(capacity),
        }
    }

    // ── observation dataset access ────────────────────────────────────────────

    /// Return a shared reference to the underlying [`ObsDataset`].
    ///
    /// Use this to iterate all observations (including those not assigned to
    /// any trajectory) or to call observer-resolution methods.
    pub fn obs_dataset(&self) -> &ObsDataset {
        &self.obs_dataset
    }

    /// Return a mutable reference to the underlying [`ObsDataset`].
    ///
    /// Required to call [`ObsDataset::get_observation`] or
    /// [`ObsDataset::get_observer`], which take `&mut self` because they
    /// update the internal LRU cache.
    pub fn obs_dataset_mut(&mut self) -> &mut ObsDataset {
        &mut self.obs_dataset
    }

    // ── observation look-up (delegates to ObsDataset LRU) ────────────────────

    /// Look up a single observation by its [`ObsId`].
    ///
    /// Delegates directly to [`ObsDataset::get_observation`], which probes the
    /// LRU cache before falling back to a linear scan.  Returns `None` if no
    /// observation with the given `id` exists in the dataset.
    pub fn get_observation(&mut self, id: ObsId) -> Option<&Observation> {
        self.obs_dataset.get_observation(id)
    }

    // ── trajectory look-up (LRU cache + linear scan) ─────────────────────────

    /// Look up a single trajectory by its [`TrajId`].
    ///
    /// Returns a shared reference to the matching [`Trajectory`], or `None`
    /// if no trajectory with the given `id` exists in this dataset.
    ///
    /// ## Caching strategy
    ///
    /// Mirrors [`ObsDataset::get_observation`] exactly:
    ///
    /// 1. **Cache probe** — [`LruCache::contains`] is called first; if the
    ///    entry is present, [`LruCache::get`] is called in a separate statement
    ///    to satisfy the borrow checker (a single `get` would hold a mutable
    ///    borrow that conflicts with returning a reference into `self`).
    /// 2. **Linear scan** — on a cache miss, the `trajectories` list is
    ///    searched with [`Iterator::find`].  The found value is cloned into the
    ///    cache before a reference is returned.
    pub fn get_trajectory(&mut self, id: &TrajId) -> Option<&Trajectory> {
        if self.lru_cache_traj.contains(id) {
            return self.lru_cache_traj.get(id);
        }
        let traj = self.trajectories.iter().find(|t| &t.id == id)?.clone();
        self.lru_cache_traj.put(id.clone(), traj);
        self.lru_cache_traj.get(id)
    }

    /// Return an iterator over all trajectories in insertion order.
    ///
    /// The iterator yields shared references and does not clone any data.
    /// Trajectories are returned in the order their `traj_id` first appeared
    /// in the source `DataFrame`.
    pub fn iter_trajectories(&self) -> impl Iterator<Item = &Trajectory> {
        self.trajectories.iter()
    }

    /// Return the total number of trajectories in this dataset.
    pub fn trajectory_count(&self) -> usize {
        self.trajectories.len()
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(feature = "polars")]
#[cfg(test)]
mod trajectory_tests {
    use super::*;
    use crate::io::polars::error::PolarsError;
    use polars::frame::DataFrame;
    use polars::prelude::Column;

    // ── shared test helpers ───────────────────────────────────────────────────

    /// Build the nine mandatory base columns for `n` rows.
    ///
    /// Each row `i` (0-indexed) gets:
    ///   - `id`        = `i + 1` (1-based, so ids are 1, 2, 3, …)
    ///   - `ra`        = 10.0 + i as f64
    ///   - `ra_err`    = 0.001
    ///   - `dec`       = -5.0 + i as f64
    ///   - `dec_err`   = 0.001
    ///   - `magnitude` = 15.0
    ///   - `mag_err`   = 0.05
    ///   - `filter`    = "G"
    ///   - `mjd_tt`    = 60000.0 + i as f64
    fn base_columns(n: usize) -> Vec<Column> {
        let ids: Vec<u64> = (1..=(n as u64)).collect();
        let ra: Vec<f64> = (0..n).map(|i| 10.0 + i as f64).collect();
        let ra_err: Vec<f64> = vec![0.001f64; n];
        let dec: Vec<f64> = (0..n).map(|i| -5.0 + i as f64).collect();
        let dec_err: Vec<f64> = vec![0.001f64; n];
        let magnitude: Vec<f64> = vec![15.0f64; n];
        let mag_err: Vec<f64> = vec![0.05f64; n];
        let filter: Vec<&str> = vec!["G"; n];
        let mjd_tt: Vec<f64> = (0..n).map(|i| 60000.0 + i as f64).collect();

        vec![
            Column::new("id".into(), ids.as_slice()),
            Column::new("ra".into(), ra.as_slice()),
            Column::new("ra_err".into(), ra_err.as_slice()),
            Column::new("dec".into(), dec.as_slice()),
            Column::new("dec_err".into(), dec_err.as_slice()),
            Column::new("magnitude".into(), magnitude.as_slice()),
            Column::new("mag_err".into(), mag_err.as_slice()),
            Column::new("filter".into(), filter.as_slice()),
            Column::new("mjd_tt".into(), mjd_tt.as_slice()),
        ]
    }

    // ── test 1 ────────────────────────────────────────────────────────────────

    /// A DataFrame without a `traj_id` column must be accepted (`Ok`), produce
    /// zero trajectories, and still expose all observations via `obs_dataset()`.
    #[test]
    fn test_no_traj_id_column() {
        let cols = base_columns(3);
        let df = DataFrame::new_infer_height(cols)
            .expect("DataFrame construction must succeed for valid base columns");

        let result = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10));
        assert!(
            result.is_ok(),
            "Expected Ok when traj_id column is absent, got: {:?}",
            result.err()
        );
        // unwrap safe: is_ok() asserted above
        let dataset = result.unwrap();

        assert_eq!(
            dataset.trajectory_count(),
            0,
            "Expected 0 trajectories when no traj_id column is present"
        );
        assert_eq!(
            dataset.obs_dataset().iter_observations().count(),
            3,
            "All 3 observations must still be accessible via obs_dataset()"
        );
    }

    // ── test 2 ────────────────────────────────────────────────────────────────

    /// A String `traj_id` column with values ["foo", "bar", "foo"] must produce
    /// exactly 2 trajectories: TrajId::Str("foo") with obs_ids [1, 3] and
    /// TrajId::Str("bar") with obs_id [2].
    #[test]
    fn test_string_traj_id_groups_correctly() {
        let mut cols = base_columns(3);
        // Row 0 → id=1 → "foo", Row 1 → id=2 → "bar", Row 2 → id=3 → "foo"
        cols.push(Column::new("traj_id".into(), &["foo", "bar", "foo"]));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10));
        assert!(
            result.is_ok(),
            "Expected Ok for String traj_id column, got: {:?}",
            result.err()
        );
        // unwrap safe: is_ok() asserted above
        let dataset = result.unwrap();

        assert_eq!(
            dataset.trajectory_count(),
            2,
            "Expected exactly 2 trajectories for traj_ids [foo, bar, foo]"
        );

        // Collect all trajectories and locate "foo" and "bar".
        let trajs: Vec<&Trajectory> = dataset.iter_trajectories().collect();

        let foo_traj = trajs
            .iter()
            .find(|t| t.id == TrajId::Str("foo".to_owned()))
            .expect("Trajectory with id TrajId::Str(\"foo\") must exist");

        let bar_traj = trajs
            .iter()
            .find(|t| t.id == TrajId::Str("bar".to_owned()))
            .expect("Trajectory with id TrajId::Str(\"bar\") must exist");

        // "foo" must own the obs_ids from row 0 (id=1) and row 2 (id=3).
        assert_eq!(
            foo_traj.obs_ids.len(),
            2,
            "TrajId::Str(\"foo\") must have 2 observation ids"
        );
        assert!(
            foo_traj.obs_ids.contains(&1u64),
            "TrajId::Str(\"foo\") must contain obs_id 1 (row 0)"
        );
        assert!(
            foo_traj.obs_ids.contains(&3u64),
            "TrajId::Str(\"foo\") must contain obs_id 3 (row 2)"
        );

        // "bar" must own only the obs_id from row 1 (id=2).
        assert_eq!(
            bar_traj.obs_ids.len(),
            1,
            "TrajId::Str(\"bar\") must have exactly 1 observation id"
        );
        assert!(
            bar_traj.obs_ids.contains(&2u64),
            "TrajId::Str(\"bar\") must contain obs_id 2 (row 1)"
        );
    }

    // ── test 3 ────────────────────────────────────────────────────────────────

    /// A UInt64-nullable `traj_id` column with values [Some(10), Some(20), Some(10)]
    /// must produce 2 trajectories: TrajId::Int(10) with 2 obs_ids and
    /// TrajId::Int(20) with 1 obs_id.
    #[test]
    fn test_int_traj_id_groups_correctly() {
        let mut cols = base_columns(3);
        let traj_ids: Vec<Option<u64>> = vec![Some(10u64), Some(20u64), Some(10u64)];
        cols.push(Column::new("traj_id".into(), traj_ids));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10));
        assert!(
            result.is_ok(),
            "Expected Ok for UInt64 nullable traj_id column, got: {:?}",
            result.err()
        );
        // unwrap safe: is_ok() asserted above
        let dataset = result.unwrap();

        assert_eq!(
            dataset.trajectory_count(),
            2,
            "Expected exactly 2 trajectories for traj_ids [10, 20, 10]"
        );

        let trajs: Vec<&Trajectory> = dataset.iter_trajectories().collect();

        let traj10 = trajs
            .iter()
            .find(|t| t.id == TrajId::Int(10))
            .expect("Trajectory with id TrajId::Int(10) must exist");

        let traj20 = trajs
            .iter()
            .find(|t| t.id == TrajId::Int(20))
            .expect("Trajectory with id TrajId::Int(20) must exist");

        // TrajId::Int(10) must own obs_ids from row 0 (id=1) and row 2 (id=3).
        assert_eq!(
            traj10.obs_ids.len(),
            2,
            "TrajId::Int(10) must have 2 observation ids"
        );
        assert!(
            traj10.obs_ids.contains(&1u64),
            "TrajId::Int(10) must contain obs_id 1 (row 0)"
        );
        assert!(
            traj10.obs_ids.contains(&3u64),
            "TrajId::Int(10) must contain obs_id 3 (row 2)"
        );

        // TrajId::Int(20) must own only obs_id from row 1 (id=2).
        assert_eq!(
            traj20.obs_ids.len(),
            1,
            "TrajId::Int(20) must have exactly 1 observation id"
        );
        assert!(
            traj20.obs_ids.contains(&2u64),
            "TrajId::Int(20) must contain obs_id 2 (row 1)"
        );
    }

    // ── test 4 ────────────────────────────────────────────────────────────────

    /// A nullable String `traj_id` column with values [Some("A"), None, Some("A")]
    /// must produce exactly 1 trajectory with 2 obs_ids. The row with null
    /// `traj_id` must NOT be part of any trajectory but must still exist in the
    /// ObsDataset (3 total observations).
    #[test]
    fn test_nullable_traj_id_skips_null_rows() {
        let mut cols = base_columns(3);
        let traj_ids: Vec<Option<&str>> = vec![Some("A"), None, Some("A")];
        cols.push(Column::new("traj_id".into(), traj_ids));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10));
        assert!(
            result.is_ok(),
            "Expected Ok for nullable String traj_id column, got: {:?}",
            result.err()
        );
        // unwrap safe: is_ok() asserted above
        let dataset = result.unwrap();

        assert_eq!(
            dataset.trajectory_count(),
            1,
            "Expected exactly 1 trajectory (the null row must be skipped)"
        );

        let traj_a = dataset
            .iter_trajectories()
            .next()
            .expect("At least one trajectory must exist");

        assert_eq!(
            traj_a.id,
            TrajId::Str("A".to_owned()),
            "The single trajectory must have id TrajId::Str(\"A\")"
        );
        assert_eq!(
            traj_a.obs_ids.len(),
            2,
            "TrajId::Str(\"A\") must have 2 observation ids (rows 0 and 2)"
        );
        assert!(
            traj_a.obs_ids.contains(&1u64),
            "TrajId::Str(\"A\") must contain obs_id 1 (row 0)"
        );
        assert!(
            traj_a.obs_ids.contains(&3u64),
            "TrajId::Str(\"A\") must contain obs_id 3 (row 2)"
        );

        // The null row (row 1, id=2) must still be in the ObsDataset.
        assert_eq!(
            dataset.obs_dataset().iter_observations().count(),
            3,
            "All 3 observations must be present in obs_dataset() even when one traj_id is null"
        );
    }

    // ── test 5 ────────────────────────────────────────────────────────────────

    /// When the `traj_id` column is present but every cell is null, the result
    /// must be `Ok` with 0 trajectories. All observations must still appear in
    /// the ObsDataset.
    #[test]
    fn test_all_traj_id_null() {
        let mut cols = base_columns(3);
        let traj_ids: Vec<Option<&str>> = vec![None, None, None];
        cols.push(Column::new("traj_id".into(), traj_ids));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10));
        assert!(
            result.is_ok(),
            "Expected Ok when all traj_id cells are null, got: {:?}",
            result.err()
        );
        // unwrap safe: is_ok() asserted above
        let dataset = result.unwrap();

        assert_eq!(
            dataset.trajectory_count(),
            0,
            "Expected 0 trajectories when all traj_id cells are null"
        );
        assert_eq!(
            dataset.obs_dataset().iter_observations().count(),
            3,
            "All 3 observations must still be in obs_dataset() even when all traj_ids are null"
        );
    }

    // ── test 6 ────────────────────────────────────────────────────────────────

    /// Calling `get_trajectory` twice for the same TrajId must return `Some` on
    /// both calls and produce the same obs_ids, verifying that the LRU cache hit
    /// path does not panic or produce stale data.
    #[test]
    fn test_get_trajectory_cache() {
        let mut cols = base_columns(2);
        // Both rows belong to the same trajectory.
        cols.push(Column::new("traj_id".into(), &["orbit1", "orbit1"]));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let mut dataset = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10))
            .expect("Expected Ok for valid DataFrame with String traj_id");

        let id = TrajId::Str("orbit1".to_owned());

        // First call — cache miss path.
        let first = dataset
            .get_trajectory(&id)
            .expect("Expected Some for TrajId::Str(\"orbit1\") on first call");
        let first_obs_ids = first.obs_ids.clone();

        assert_eq!(
            first_obs_ids.len(),
            2,
            "Trajectory must contain 2 obs_ids (one per row)"
        );
        assert!(
            first_obs_ids.contains(&1u64),
            "Trajectory must include obs_id 1 (row 0)"
        );
        assert!(
            first_obs_ids.contains(&2u64),
            "Trajectory must include obs_id 2 (row 1)"
        );

        // Second call — cache hit path (must not panic or return a different result).
        let second = dataset
            .get_trajectory(&id)
            .expect("Expected Some for TrajId::Str(\"orbit1\") on second (cache hit) call");

        assert_eq!(
            second.obs_ids.len(),
            2,
            "Cache hit must return a trajectory with the same obs_ids count"
        );
        assert!(
            second.obs_ids.contains(&1u64),
            "Cache hit must still contain obs_id 1"
        );
        assert!(
            second.obs_ids.contains(&2u64),
            "Cache hit must still contain obs_id 2"
        );
    }

    // ── test 7 ────────────────────────────────────────────────────────────────

    /// `get_observation` must delegate to the underlying ObsDataset and return
    /// a matching Observation for a known obs id.
    #[test]
    fn test_get_observation_delegates_to_obs_dataset() {
        let cols = base_columns(2);
        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let mut dataset = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10))
            .expect("Expected Ok for valid base-only DataFrame");

        // Row 0 has id=1, row 1 has id=2 (from base_columns helper).
        let obs = dataset
            .get_observation(1u64)
            .expect("Expected Some for obs_id 1");

        assert_eq!(obs.id, 1u64, "Returned Observation must have id == 1");

        // Also verify that a non-existent id returns None.
        assert!(
            dataset.get_observation(999u64).is_none(),
            "Expected None for an obs_id (999) that does not exist in the dataset"
        );
    }

    // ── test 8 ────────────────────────────────────────────────────────────────

    /// A `traj_id` column of type Float64 must be rejected with
    /// `PolarsError::TrajIdColumnTypeError`.
    #[test]
    fn test_wrong_traj_id_type() {
        let mut cols = base_columns(2);
        // Float64 is not a supported traj_id type.
        cols.push(Column::new("traj_id".into(), &[1.0f64, 2.0f64]));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10));

        // TrajDataset does not implement Debug, so use match instead of unwrap_err().
        match result {
            Err(PolarsError::TrajIdColumnTypeError(_)) => { /* expected */ }
            Err(other) => panic!(
                "Expected PolarsError::TrajIdColumnTypeError for a Float64 traj_id column, \
                 got: {other:?}"
            ),
            Ok(_) => panic!("Expected Err for a Float64 traj_id column, got Ok"),
        }
    }

    // ── test 9 ────────────────────────────────────────────────────────────────

    /// When `traj_id` values are ["B", "A", "B", "C", "A"], `iter_trajectories`
    /// must yield trajectories in first-appearance order: B first, then A, then C.
    #[test]
    fn test_insertion_order_preserved() {
        let mut cols = base_columns(5);
        // Row 0→"B", 1→"A", 2→"B", 3→"C", 4→"A"
        cols.push(Column::new("traj_id".into(), &["B", "A", "B", "C", "A"]));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(10));
        assert!(
            result.is_ok(),
            "Expected Ok for String traj_id with 3 distinct values, got: {:?}",
            result.err()
        );
        // unwrap safe: is_ok() asserted above
        let dataset = result.unwrap();

        assert_eq!(
            dataset.trajectory_count(),
            3,
            "Expected exactly 3 trajectories for traj_ids [B, A, B, C, A]"
        );

        let trajs: Vec<&Trajectory> = dataset.iter_trajectories().collect();

        // First-appearance order must be: B (row 0), A (row 1), C (row 3).
        assert_eq!(
            trajs[0].id,
            TrajId::Str("B".to_owned()),
            "First trajectory must be TrajId::Str(\"B\") (first appeared at row 0)"
        );
        assert_eq!(
            trajs[1].id,
            TrajId::Str("A".to_owned()),
            "Second trajectory must be TrajId::Str(\"A\") (first appeared at row 1)"
        );
        assert_eq!(
            trajs[2].id,
            TrajId::Str("C".to_owned()),
            "Third trajectory must be TrajId::Str(\"C\") (first appeared at row 3)"
        );

        // Also verify the obs_ids for each trajectory are correct.
        // B: rows 0 (id=1) and 2 (id=3).
        assert_eq!(trajs[0].obs_ids.len(), 2, "\"B\" must have 2 obs_ids");
        assert!(
            trajs[0].obs_ids.contains(&1u64),
            "\"B\" must contain obs_id 1"
        );
        assert!(
            trajs[0].obs_ids.contains(&3u64),
            "\"B\" must contain obs_id 3"
        );

        // A: rows 1 (id=2) and 4 (id=5).
        assert_eq!(trajs[1].obs_ids.len(), 2, "\"A\" must have 2 obs_ids");
        assert!(
            trajs[1].obs_ids.contains(&2u64),
            "\"A\" must contain obs_id 2"
        );
        assert!(
            trajs[1].obs_ids.contains(&5u64),
            "\"A\" must contain obs_id 5"
        );

        // C: row 3 (id=4).
        assert_eq!(
            trajs[2].obs_ids.len(),
            1,
            "\"C\" must have exactly 1 obs_id"
        );
        assert!(
            trajs[2].obs_ids.contains(&4u64),
            "\"C\" must contain obs_id 4"
        );
    }
}
