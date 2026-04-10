//! Night-of-observation grouping types for the photom pipeline.
//!
//! This module defines the types used to group [`Observation`]s by the
//! calendar night on which they were recorded:
//!
//! - [`NightId`] — a lightweight, copyable identifier for a single night,
//!   wrapping an integer MJD day number.
//! - [`NightObs`] — a group of [`ObsId`]s that all belong to the same night.
//! - [`NightDataset`] — a dataset that wraps an [`ObsDataset`] and adds a
//!   nightly-grouping layer on top of it.
//!
//! ## Relationship to `ObsDataset`
//!
//! [`NightDataset`] owns an [`ObsDataset`] as its primary store of observation
//! data.  The embedded [`ObsDataset`] is filtered to contain only those
//! observations that are assigned to at least one night (i.e. rows whose
//! `night_id` cell in the source `DataFrame` was not `null`).  All
//! [`Observation`] resolution — including lazy MPC observatory initialisation
//! — goes through the embedded [`ObsDataset`] via
//! [`NightDataset::obs_dataset_mut`] or the convenience wrapper
//! [`NightDataset::get_observation`].
//!
//! ## Night LRU cache
//!
//! [`NightDataset`] maintains its own LRU cache over [`NightObs`] values,
//! mirroring the per-observation LRU cache that [`ObsDataset`] keeps
//! internally.  On first access via [`NightDataset::get_night`], the
//! [`NightObs`] for a given [`NightId`] is cloned into the cache and evicted
//! in least-recently-used order when the cache is full.  The cache capacity is
//! set by the `lru_cache_size` parameter of `NightDataset::new` and defaults
//! to **1 000** entries.
//!
//! ## `polars` feature
//!
//! The constructors [`NightDataset::from_polars`] and
//! [`NightDataset::from_lazy`] are only available when the optional `polars`
//! feature is enabled.  They ingest a Polars `DataFrame` or `LazyFrame`
//! (respectively) and return a fully-assembled [`NightDataset`].
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`NightId`] | struct | Logical identifier for a night of observation |
//! | [`NightObs`] | struct | Group of observation IDs sharing the same night |
//! | [`NightDataset`] | struct | Dataset with observations grouped by night |

use std::num::NonZeroUsize;

use lru::LruCache;
#[cfg(feature = "polars")]
use polars::{frame::DataFrame, prelude::LazyFrame};

use crate::observation::{ObsDataset, ObsId, Observation};
#[cfg(feature = "polars")]
use crate::{io::polars::error::PolarsError, observer::error_model::ObsErrorModel};

/// Logical identifier for a night of observation.
///
/// Wraps a `u32` that typically represents an integer MJD day number
/// (e.g. `60312`).  The value must be stable across runs because it is used
/// as a directory name in on-disk outputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NightId(pub u32);

impl From<u32> for NightId {
    /// Convert a raw `u32` MJD day number into a [`NightId`].
    fn from(value: u32) -> Self {
        Self(value)
    }
}

/// A group of observation identifiers sharing the same night.
///
/// `NightObs` stores only the [`ObsId`] keys; the actual [`Observation`]
/// data lives in the parent [`NightDataset`]'s embedded [`ObsDataset`].
/// Resolve an ID to a full observation via [`NightDataset::get_observation`].
///
/// The observation IDs are stored in the order they were encountered while
/// reading the source `DataFrame` (i.e. row order).
#[derive(Clone, Debug)]
pub struct NightObs {
    /// The logical night identifier (wraps a MJD day number).
    pub night_id: NightId,

    /// Observation identifiers (in source-row order) that belong to this night.
    ///
    /// Each value is a key into the parent [`NightDataset`]'s [`ObsDataset`].
    pub obs_ids: Vec<ObsId>,
}

/// A dataset that groups observations by night of observation.
///
/// `NightDataset` owns a complete [`ObsDataset`] (all observations, custom
/// geodetic observers, lazy MPC observatory table, and the per-observation LRU
/// cache) and adds a layer of nightly grouping on top of it.
///
/// Observations whose `night_id` was `null` in the source `DataFrame` are
/// **not** present in the embedded [`ObsDataset`]; only observations with a
/// non-null `night_id` are ingested.  This distinguishes `NightDataset` from
/// `TrajDataset`, where null-keyed rows are still stored in the underlying
/// [`ObsDataset`].
///
/// Repeated look-ups by [`NightId`] are accelerated by a dedicated LRU cache
/// (see [`NightDataset::get_night`] for the caching strategy).  The same
/// `lru_cache_size` parameter passed to `NightDataset::new` controls the
/// capacity of this cache; it defaults to **1 000** entries.
#[derive(Debug)]
pub struct NightDataset {
    /// All observations in the dataset, indexed by [`ObsId`].
    ///
    /// Contains only those observations that are assigned to at least one
    /// night (i.e. rows with a non-null `night_id` in the source `DataFrame`).
    /// This is the single source of truth for all [`Observation`] data,
    /// custom geodetic observers, and the MPC lookup table; it also owns the
    /// per-observation LRU cache.  Use [`NightDataset::obs_dataset`] for
    /// shared access or [`NightDataset::obs_dataset_mut`] when mutation is
    /// required (e.g. to call cache-updating look-up methods).
    pub obs_dataset: ObsDataset,

    /// All nights in the dataset, in first-appearance order.
    ///
    /// The order matches the order in which each distinct `night_id` value
    /// first appeared in the source `DataFrame`.
    pub nights: Vec<NightObs>,

    /// LRU cache keyed by [`NightId`].
    ///
    /// A [`NightObs`] is cloned into the cache on the first successful call
    /// to [`NightDataset::get_night`] and evicted in least-recently-used order
    /// when the cache is full.  The capacity is set at construction time by
    /// `NightDataset::new` and defaults to **1 000** entries.
    pub lru_cache_night: LruCache<NightId, NightObs>,
}

impl NightDataset {
    /// Build a [`NightDataset`] from pre-parsed components.
    ///
    /// The night LRU cache is initialised with the given capacity.  If
    /// `lru_cache_size` is `None`, the cache is created with a default
    /// capacity of **1 000** entries.
    ///
    /// # Arguments
    ///
    /// - `obs_dataset`     — the fully-assembled [`ObsDataset`] containing
    ///   only night-assigned observations.
    /// - `nights`          — the list of [`NightObs`] groups in first-appearance
    ///   order.
    /// - `lru_cache_size`  — optional capacity for the night LRU cache;
    ///   `None` defaults to 1 000.
    pub(crate) fn new(
        obs_dataset: ObsDataset,
        nights: Vec<NightObs>,
        lru_cache_size: Option<usize>,
    ) -> Self {
        let capacity = NonZeroUsize::new(lru_cache_size.unwrap_or(1000)).unwrap();
        Self {
            obs_dataset,
            nights,
            lru_cache_night: LruCache::new(capacity),
        }
    }

    /// Construct a [`NightDataset`] from a Polars [`DataFrame`].
    ///
    /// The frame must satisfy the same base-column schema as
    /// [`ObsDataset::from_polars`] and may additionally contain a `night_id`
    /// column.  See [`crate::io::polars`] for the full schema specification.
    ///
    /// Rows whose `night_id` cell is `null` are **not** ingested; only
    /// observations with a non-null `night_id` are included in the resulting
    /// [`NightDataset`].
    ///
    /// # Arguments
    ///
    /// - `df`             — source Polars [`DataFrame`].
    /// - `error_model`    — astrometric error model forwarded to the embedded
    ///   [`ObsDataset`] for MPC observatory initialisation.
    /// - `lru_cache_size` — optional capacity for the night LRU cache;
    ///   `None` defaults to 1 000.
    ///
    /// # Errors
    ///
    /// Returns a [`PolarsError`] if the base-column schema is violated, if any
    /// observer column rule is broken, or if the `night_id` column has an
    /// unsupported type.
    #[cfg(feature = "polars")]
    pub fn from_polars(
        df: &DataFrame,
        error_model: ObsErrorModel,
        lru_cache_size: Option<usize>,
    ) -> Result<Self, PolarsError> {
        use crate::io::polars::load_night_from_polars;
        load_night_from_polars(df, error_model, lru_cache_size)
    }

    /// Construct a [`NightDataset`] from a Polars [`LazyFrame`].
    ///
    /// The lazy computation plan is executed (via [`LazyFrame::collect`]) before
    /// ingestion begins.  Once collected, the same validation and assembly
    /// pipeline as [`NightDataset::from_polars`] is applied.
    ///
    /// # Arguments
    ///
    /// - `lf`             — source Polars [`LazyFrame`].
    /// - `error_model`    — astrometric error model forwarded to the embedded
    ///   [`ObsDataset`] for MPC observatory initialisation.
    /// - `lru_cache_size` — optional capacity for the night LRU cache;
    ///   `None` defaults to 1 000.
    ///
    /// # Errors
    ///
    /// Returns [`PolarsError::Polars`] if the lazy plan fails to execute, plus
    /// all errors documented on [`NightDataset::from_polars`].
    #[cfg(feature = "polars")]
    pub fn from_lazy(
        lf: LazyFrame,
        error_model: ObsErrorModel,
        lru_cache_size: Option<usize>,
    ) -> Result<Self, PolarsError> {
        use crate::io::polars::load_night_from_polars;
        load_night_from_polars(lf, error_model, lru_cache_size)
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

    // ── night look-up (LRU cache + linear scan) ─────────────────────────

    /// Look up a single night by its [`NightId`].
    ///
    /// Returns a shared reference to the matching [`NightObs`], or `None` if
    /// no night with the given `id` exists in this dataset.
    ///
    /// ## Caching strategy
    ///
    /// Mirrors [`ObsDataset::get_observation`] exactly:
    ///
    /// 1. **Cache probe** — [`LruCache::contains`] is called first; if the
    ///    entry is present, [`LruCache::get`] is called in a separate statement
    ///    to satisfy the borrow checker (a single `get` call would hold a
    ///    mutable borrow that conflicts with returning a reference into `self`).
    /// 2. **Linear scan** — on a cache miss, the `nights` list is searched with
    ///    [`Iterator::find`].  The found value is cloned into the cache before a
    ///    reference is returned, so subsequent look-ups for the same `id` hit
    ///    the cache.
    pub fn get_night(&mut self, id: &NightId) -> Option<&NightObs> {
        if self.lru_cache_night.contains(id) {
            return self.lru_cache_night.get(id);
        }
        let night = self.nights.iter().find(|n| &n.night_id == id)?.clone();
        self.lru_cache_night.put(*id, night);
        self.lru_cache_night.get(id)
    }

    /// Return an iterator over all nights in insertion order.
    ///
    /// The iterator yields shared references and does not clone any data.
    /// Nights are returned in the order their `night_id` first appeared
    /// in the source `DataFrame`.
    pub fn iter_nights(&self) -> impl Iterator<Item = &NightObs> {
        self.nights.iter()
    }

    /// Return the total number of nights in this dataset.
    pub fn night_count(&self) -> usize {
        self.nights.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod night_tests {
    use crate::astrometry::EquCoord;
    use crate::nightly::{NightDataset, NightId, NightObs};
    use crate::observation::{ObsDataset, ObsId, Observation};
    use crate::observer::error_model::ObsErrorModel;
    use crate::photometry::{Filter, Photometry};

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_obs_dataset_empty() -> ObsDataset {
        ObsDataset::new(vec![], vec![], ObsErrorModel::FCCT14, Some(100))
    }

    fn make_observation(id: u64) -> Observation {
        Observation {
            id,
            equ_coord: EquCoord::new(0.5, 1e-5, 0.2, 1e-5),
            photometry: Photometry {
                magnitude: 15.0,
                error: 0.1,
                filter: Filter::String("G".to_string()),
            },
            mjd_tt: 60000.5,
            observer: None,
        }
    }

    fn make_obs_dataset(obs: Vec<Observation>) -> ObsDataset {
        ObsDataset::new(obs, vec![], ObsErrorModel::FCCT14, Some(100))
    }

    fn make_night_obs(night: u32, obs_ids: Vec<u64>) -> NightObs {
        NightObs {
            night_id: NightId(night),
            obs_ids,
        }
    }

    fn make_dataset(obs: Vec<Observation>, nights: Vec<NightObs>) -> NightDataset {
        NightDataset::new(make_obs_dataset(obs), nights, Some(100))
    }

    // -----------------------------------------------------------------------
    // NightId — ordering, Copy, Debug, Hash
    // -----------------------------------------------------------------------

    mod night_id {
        use std::collections::HashSet;

        use crate::nightly::NightId;

        /// Verifies that two NightIds with the same value compare as equal.
        #[test]
        fn night_id_equal_values_are_eq() {
            let a = NightId(60000);
            let b = NightId(60000);
            assert_eq!(a, b);
        }

        /// Verifies that a smaller NightId is less than a larger one.
        #[test]
        fn night_id_ordering_less_than() {
            let a = NightId(59999);
            let b = NightId(60000);
            assert!(a < b);
        }

        /// Verifies that a larger NightId is greater than a smaller one.
        #[test]
        fn night_id_ordering_greater_than() {
            let a = NightId(60001);
            let b = NightId(60000);
            assert!(a > b);
        }

        /// Verifies that NightId is Copy: the original is still usable after a copy.
        #[test]
        fn night_id_is_copy() {
            let original = NightId(60000);
            let copy = original; // Copy, not move
            assert_eq!(original, copy);
        }

        /// Verifies that the Debug output contains the inner value.
        #[test]
        fn night_id_debug_contains_inner_value() {
            let id = NightId(60312);
            let debug_str = format!("{id:?}");
            assert!(
                debug_str.contains("60312"),
                "Debug output should contain '60312', got: {debug_str}"
            );
        }

        /// Verifies that NightId can be inserted into a HashSet (requires Hash + Eq).
        #[test]
        fn night_id_can_be_inserted_into_hash_set() {
            let mut set: HashSet<NightId> = HashSet::new();
            set.insert(NightId(60000));
            set.insert(NightId(60001));
            set.insert(NightId(60000)); // duplicate
            assert_eq!(set.len(), 2);
        }

        /// Verifies that HashSet membership lookup works correctly for NightId.
        #[test]
        fn night_id_hash_set_contains() {
            let mut set: HashSet<NightId> = HashSet::new();
            set.insert(NightId(60000));
            assert!(set.contains(&NightId(60000)));
            assert!(!set.contains(&NightId(99999)));
        }
    }

    // -----------------------------------------------------------------------
    // NightId — From<u32> conversions
    // -----------------------------------------------------------------------

    mod night_id_from_u32 {
        use crate::nightly::NightId;

        /// Verifies that NightId::from(u32) produces the correct NightId value.
        #[test]
        fn from_u32_produces_correct_night_id() {
            let id = NightId::from(60312u32);
            assert_eq!(id, NightId(60312));
        }

        /// Verifies that the Into<NightId> blanket impl for u32 works correctly.
        #[test]
        fn into_night_id_from_u32() {
            let id: NightId = 60312u32.into();
            assert_eq!(id, NightId(60312));
        }
    }

    // -----------------------------------------------------------------------
    // NightObs — field access, ordering, Clone, Debug, empty obs_ids
    // -----------------------------------------------------------------------

    mod night_obs {
        use super::*;

        /// Verifies that NightObs fields are accessible and hold the values supplied.
        #[test]
        fn night_obs_fields_are_accessible() {
            let night_obs = make_night_obs(60000, vec![1, 2, 3]);
            assert_eq!(night_obs.night_id, NightId(60000));
            assert_eq!(night_obs.obs_ids, vec![1u64, 2u64, 3u64]);
        }

        /// Verifies that obs_ids preserves insertion order.
        #[test]
        fn night_obs_obs_ids_is_in_insertion_order() {
            let ids: Vec<ObsId> = vec![10, 20, 5, 99];
            let night_obs = make_night_obs(60001, ids.clone());
            assert_eq!(night_obs.obs_ids, ids);
        }

        /// Verifies that NightObs implements Clone and that the cloned value has
        /// equal fields to the original.
        #[test]
        fn night_obs_is_clone() {
            let original = make_night_obs(60002, vec![42, 43]);
            let cloned = original.clone();
            assert_eq!(cloned.night_id, original.night_id);
            assert_eq!(cloned.obs_ids, original.obs_ids);
        }

        /// Verifies that the Debug output of NightObs is non-empty.
        #[test]
        fn night_obs_debug_is_non_empty() {
            let night_obs = make_night_obs(60003, vec![1]);
            let debug_str = format!("{night_obs:?}");
            assert!(!debug_str.is_empty(), "Debug output must not be empty");
        }

        /// Verifies that a NightObs with an empty obs_ids vec is valid and accessible.
        #[test]
        fn night_obs_empty_obs_ids() {
            let night_obs = make_night_obs(60004, vec![]);
            assert_eq!(night_obs.night_id, NightId(60004));
            assert!(night_obs.obs_ids.is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // NightDataset::new — construction
    // -----------------------------------------------------------------------

    mod night_dataset_new {
        use super::*;

        /// Verifies that constructing an empty NightDataset with the default LRU
        /// size (None) does not panic.
        #[test]
        fn new_empty_does_not_panic() {
            let _ds = NightDataset::new(make_obs_dataset_empty(), vec![], None);
        }

        /// Verifies that constructing a NightDataset with a custom LRU cache size
        /// of Some(5) does not panic.
        #[test]
        fn new_with_custom_lru_size_does_not_panic() {
            let _ds = NightDataset::new(make_obs_dataset_empty(), vec![], Some(5));
        }

        /// Verifies that an empty NightDataset reports a night count of zero.
        #[test]
        fn new_empty_night_count_is_zero() {
            let ds = NightDataset::new(make_obs_dataset_empty(), vec![], Some(100));
            assert_eq!(ds.night_count(), 0);
        }

        /// Verifies that night_count() matches the number of NightObs supplied at
        /// construction time.
        #[test]
        fn new_with_nights_has_correct_count() {
            let nights = vec![
                make_night_obs(60000, vec![1]),
                make_night_obs(60001, vec![2]),
                make_night_obs(60002, vec![3]),
            ];
            let ds = NightDataset::new(make_obs_dataset_empty(), nights, Some(100));
            assert_eq!(ds.night_count(), 3);
        }
    }

    // -----------------------------------------------------------------------
    // NightDataset::iter_nights
    // -----------------------------------------------------------------------

    mod iter_nights {
        use super::*;

        /// Verifies that iter_nights on an empty dataset yields no items.
        #[test]
        fn iter_nights_empty_yields_nothing() {
            let ds = make_dataset(vec![], vec![]);
            assert_eq!(ds.iter_nights().count(), 0);
        }

        /// Verifies that iter_nights yields NightObs in the same order as they
        /// were provided at construction (insertion order).
        #[test]
        fn iter_nights_yields_in_insertion_order() {
            let nights = vec![
                make_night_obs(60000, vec![1]),
                make_night_obs(60001, vec![2]),
                make_night_obs(60002, vec![3]),
            ];
            let ds = make_dataset(vec![], nights);
            let ids: Vec<NightId> = ds.iter_nights().map(|n| n.night_id).collect();
            assert_eq!(ids, vec![NightId(60000), NightId(60001), NightId(60002)]);
        }

        /// Verifies that iter_nights on a single-night dataset yields exactly one item.
        #[test]
        fn iter_nights_single_item() {
            let nights = vec![make_night_obs(60000, vec![1])];
            let ds = make_dataset(vec![], nights);
            assert_eq!(ds.iter_nights().count(), 1);
        }
    }

    // -----------------------------------------------------------------------
    // NightDataset::get_night
    // -----------------------------------------------------------------------

    mod get_night {
        use super::*;

        /// Verifies that get_night returns Some for an existing NightId.
        #[test]
        fn get_night_returns_some_for_existing_id() {
            let nights = vec![make_night_obs(60000, vec![1])];
            let mut ds = make_dataset(vec![], nights);
            assert!(ds.get_night(&NightId(60000)).is_some());
        }

        /// Verifies that get_night returns None when the NightId is not in the dataset.
        #[test]
        fn get_night_returns_none_for_missing_id() {
            let nights = vec![make_night_obs(60000, vec![1])];
            let mut ds = make_dataset(vec![], nights);
            assert!(ds.get_night(&NightId(99999)).is_none());
        }

        /// Verifies that repeated calls to get_night for the same id return the
        /// same night_id (exercises the LRU cache hit path).
        #[test]
        fn get_night_repeated_calls_return_same_id() {
            let nights = vec![make_night_obs(60000, vec![1, 2])];
            let mut ds = make_dataset(vec![], nights);
            let first = ds.get_night(&NightId(60000)).map(|n| n.night_id);
            let second = ds.get_night(&NightId(60000)).map(|n| n.night_id);
            assert_eq!(first, second);
        }

        /// Verifies that get_night returns the correct obs_ids for the requested night.
        #[test]
        fn get_night_correct_obs_ids_returned() {
            let nights = vec![
                make_night_obs(60000, vec![10, 20]),
                make_night_obs(60001, vec![30, 40, 50]),
            ];
            let mut ds = make_dataset(vec![], nights);
            let night = ds.get_night(&NightId(60001));
            assert!(night.is_some(), "Expected Some for NightId(60001)");
            // unwrap() is safe: we just asserted Some above
            assert_eq!(night.unwrap().obs_ids, vec![30u64, 40u64, 50u64]);
        }

        /// Verifies the LRU eviction behaviour: when the cache capacity is 1 and a
        /// second night is looked up, the first is evicted from the cache.  The
        /// evicted night must still be findable via the linear-scan fallback.
        #[test]
        fn get_night_lru_eviction_still_findable() {
            let nights = vec![
                make_night_obs(60000, vec![1]),
                make_night_obs(60001, vec![2]),
            ];
            // LRU capacity of 1: looking up 60001 evicts 60000 from the cache.
            let mut ds = NightDataset::new(make_obs_dataset_empty(), nights, Some(1));

            // Populate the cache with 60000.
            assert!(ds.get_night(&NightId(60000)).is_some());
            // Looking up 60001 evicts 60000 from the cache.
            assert!(ds.get_night(&NightId(60001)).is_some());
            // 60000 must still be findable via the linear scan even after eviction.
            assert!(
                ds.get_night(&NightId(60000)).is_some(),
                "NightId(60000) should still be findable after LRU eviction"
            );
        }
    }

    // -----------------------------------------------------------------------
    // NightDataset — obs_dataset access and get_observation delegation
    // -----------------------------------------------------------------------

    mod obs_dataset_access {
        use super::*;

        /// Verifies that obs_dataset() returns a reference to the embedded ObsDataset,
        /// confirmed by checking that iter_observations yields the expected count.
        #[test]
        fn obs_dataset_returns_ref() {
            let obs = vec![make_observation(1), make_observation(2)];
            let ds = make_dataset(obs, vec![]);
            assert_eq!(ds.obs_dataset().iter_observations().count(), 2);
        }

        /// Verifies that obs_dataset_mut() allows calling get_observation via the
        /// mutable reference.
        #[test]
        fn obs_dataset_mut_allows_get_observation() {
            let obs = vec![make_observation(42)];
            let mut ds = make_dataset(obs, vec![]);
            let found = ds.obs_dataset_mut().get_observation(42);
            assert!(
                found.is_some(),
                "Expected Some for obs id=42 via obs_dataset_mut()"
            );
        }

        /// Verifies that NightDataset::get_observation delegates correctly to the
        /// embedded ObsDataset and returns the right observation.
        #[test]
        fn get_observation_delegates_to_obs_dataset() {
            let obs = vec![make_observation(7), make_observation(8)];
            let mut ds = make_dataset(obs, vec![]);
            let found = ds.get_observation(8);
            assert!(found.is_some(), "Expected Some for obs id=8");
            // unwrap() is safe: we just asserted Some above
            assert_eq!(found.unwrap().id, 8);
        }
    }
}
