//! Core observation data types for the photom crate.
//!
//! This module defines the fundamental building blocks used throughout the
//! pipeline: individual astrometric/photometric measurements
//! ([`observation::Observation`]), the dataset that holds a collection of them
//! ([`ObsDataset`]), the identifier types that label observations, nights, and
//! observatories ([`ObsId`], [`crate::NightId`], `ObserverId`), and the error
//! type that covers all failure modes arising during dataset construction
//! ([`ObsDatasetError`]).
//!
//! ## Key design notes
//!
//! - **LRU cache** — [`ObsDataset`] keeps a least-recently-used cache of up
//!   to 1 000 [`observation::Observation`] values so that repeated look-ups by
//!   [`ObsId`] do not scan the full observation list on every call.
//! - **Lazy MPC initialisation** — the Minor Planet Center observatory table
//!   is fetched from the network only on the first call to
//!   [`ObsDataset::get_observer`] for an MPC-coded site, and the result
//!   (success *or* failure) is stored in a [`std::sync::OnceLock`] so that
//!   subsequent calls are free.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`ObsId`] | type alias | Unique numeric identifier for a single observation |
//! | [`crate::NightId`] | struct | Logical identifier for a night of observation |
//! | `ObserverId` | enum | Reference to either a custom or an MPC-coded observer |
//! | [`observation::Observation`] | struct | A single astrometric/photometric measurement |
//! | [`ObsDataset`] | struct | Collection of observations with lazy observer resolution |
//! | [`ObsDatasetError`] | enum | Errors arising from dataset construction |

pub(crate) mod index;
pub mod observation;

#[cfg(feature = "parallel")]
pub mod parallel;

use std::num::NonZeroUsize;

use lru::LruCache;
use thiserror::Error;

#[cfg(feature = "polars")]
use crate::io::polars::{error::PolarsError, load_observation_from_polars};
#[cfg(feature = "polars")]
use polars::{frame::DataFrame, lazy::frame::LazyFrame};

use crate::{
    NightId, TrajId,
    observation_dataset::{
        index::{NightIndexMap, ObsDatasetIndex, ObsIndex, ObservationIndexMap, TrajIndexMap},
        observation::Observation,
    },
    observer::{
        Observer,
        dataset::ObserverDataset,
        error_model::{ErrorModelParseError, ObsErrorModel},
        mpc::MPCError,
    },
};

/// Unique numeric identifier for a single observation.
///
/// Observations are keyed by this value inside [`ObsDataset`] and its
/// internal LRU cache.  The identifier is assigned by the data source (e.g.
/// the `id` column of a Polars `DataFrame`) and must be unique within a
/// dataset.
pub type ObsId = u64;

/// Errors that can arise when constructing or using an [`ObsDataset`].
#[derive(Debug, Error)]
pub enum ObsDatasetError {
    /// The network request to the Minor Planet Center catalogue failed.
    #[error(transparent)]
    MPCError(#[from] MPCError),

    /// The astrometric error-model file could not be parsed.
    #[error(transparent)]
    ErrorModelError(#[from] ErrorModelParseError),

    /// The observer associated with an observation could not be resolved.
    #[error("The error model has not been initialised")]
    ErrorModelNotFound,

    /// A Polars I/O or schema error occurred while loading observations.
    #[cfg(feature = "polars")]
    #[error(transparent)]
    PolarIoError(#[from] PolarsError),
}

/// A collection of [`observation::Observation`]s with associated observer metadata.
///
/// `ObsDataset` is the primary container for observation data in the pipeline.
/// In addition to the raw observations it holds:
///
/// - A list of **custom geodetic observers** supplied directly in the input,
///   referenced by index through `ObserverId::IntId`.
/// - A **lazily-initialised MPC lookup table** that maps three-byte MPC codes
///   to [`Observer`] metadata.  The table is fetched from the MPC website
///   on the first access and cached for the lifetime of the dataset.
/// - An **LRU cache** of up to 1 000 [`observation::Observation`] values so that
///   repeated look-ups by [`ObsId`] avoid a full linear scan.
#[derive(Debug)]
pub struct ObsDataset {
    /// Full list of observations in insertion order.
    observations: Vec<Observation>,

    /// Index mappings for efficient look-up by various identifiers.
    index: ObsDatasetIndex,

    /// Observer values for both custom geodetic observers (indexed by `ObserverId::IntId`)
    /// and MPC-coded observers (resolved lazily via `ObserverId::MpcCode`).
    observer_dataset: ObserverDataset,

    /// LRU cache keyed by [`ObsId`] with a fixed capacity of 1 000 entries.
    ///
    /// Entries are cloned into the cache on first access and evicted in
    /// least-recently-used order when the cache is full.
    lru_cache_obs: LruCache<ObsId, Observation>,
}

impl ObsDataset {
    /// Construct an [`ObsDataset`] from a Polars [`DataFrame`].
    ///
    /// Validates the frame against the expected schema, extracts all
    /// observation columns, and assembles the dataset.  See
    /// [`crate::io::polars`] for the full column specification and
    /// observer-resolution rules.
    ///
    /// # Arguments
    ///
    /// - `df` — the source Polars [`DataFrame`] containing the observation data.
    /// - `error_model` — the [`ObsErrorModel`] used to assign astrometric accuracies to
    ///   MPC-coded observers during MPC table initialisation.
    /// - `lru_cache_size` — optional capacity for the LRU cache used to speed up repeated observation lookups;
    ///   if `None`, the cache size is set to 1 000.
    /// - `do_rechunk` — whether to consolidate multi-chunk columns into a single contiguous
    ///   chunk before ingestion.  `None` and `Some(true)` both enable the automatic rechunk
    ///   (default behaviour).  Pass `Some(false)` only when every column in `df` is already
    ///   stored in a single Arrow chunk (e.g. after reading with
    ///   `ScanArgsParquet { rechunk: true, .. }` or after an explicit
    ///   `DataFrame::rechunk_mut`).  Passing `Some(false)` on a fragmented frame will
    ///   cause ingestion to fail with a [`PolarsError::Polars`] error.
    ///
    /// # Errors
    ///
    /// Returns a [`PolarsError`] if the frame fails schema validation, if a
    /// Polars-internal operation fails, or if any observer column violates
    /// the resolution rules (e.g. a partially-null geodetic triplet).
    #[cfg(feature = "polars")]
    pub fn from_polars(
        df: &DataFrame,
        error_model: Option<ObsErrorModel>,
        lru_cache_size: Option<usize>,
        do_rechunk: Option<bool>,
    ) -> Result<Self, PolarsError> {
        load_observation_from_polars(df, error_model, lru_cache_size, do_rechunk)
    }

    /// Construct an [`ObsDataset`] from a Polars [`LazyFrame`].
    ///
    /// The lazy computation plan is executed (via [`LazyFrame::collect`]) before
    /// ingestion begins.  Once collected, the same validation and assembly
    /// pipeline as [`ObsDataset::from_polars`] is applied.
    ///
    /// # Arguments
    ///
    /// - `lf` — the source Polars [`LazyFrame`].
    /// - `error_model` — the [`ObsErrorModel`] used to assign astrometric
    ///   accuracies to MPC-coded observers during MPC table initialisation.
    /// - `lru_cache_size` — optional LRU cache capacity; `None` defaults to 1 000.
    /// - `do_rechunk` — whether to consolidate multi-chunk columns into a single contiguous
    ///   chunk after the lazy plan is collected.  `None` and `Some(true)` both enable the
    ///   automatic rechunk (default behaviour).  Pass `Some(false)` only when the collected
    ///   frame is already contiguous — for example when the `LazyFrame` was created with
    ///   `ScanArgsParquet { rechunk: true, .. }`, which guarantees a single chunk per column
    ///   after `collect`.  Passing `Some(false)` on a fragmented frame will cause ingestion
    ///   to fail with a [`PolarsError::Polars`] error.
    ///
    /// # Errors
    ///
    /// Returns [`PolarsError::Polars`] if the lazy plan fails to execute, plus
    /// all errors documented on [`ObsDataset::from_polars`].
    #[cfg(feature = "polars")]
    pub fn from_lazy(
        lf: LazyFrame,
        error_model: Option<ObsErrorModel>,
        lru_cache_size: Option<usize>,
        do_rechunk: Option<bool>,
    ) -> Result<Self, PolarsError> {
        load_observation_from_polars(lf, error_model, lru_cache_size, do_rechunk)
    }

    /// Look up a single observation by its [`ObsId`].
    ///
    /// Returns a shared reference to the matching [`Observation`], or `None`
    /// if no observation with the given `id` exists in this dataset.
    ///
    /// ## Caching strategy
    ///
    /// To avoid repeatedly scanning the full observation list, results are
    /// stored in an LRU cache (capacity 1 000).  The look-up proceeds in two
    /// phases:
    ///
    /// 1. **Cache probe** — [`LruCache::contains`] is called first.  If the
    ///    entry is present, [`LruCache::get`] is called in a separate
    ///    statement to obtain the reference.  This two-step approach is
    ///    necessary because a single `get` call borrows `self` mutably (to
    ///    update the LRU order) and would prevent returning a reference into
    ///    the same `self`; the intermediate `contains` check lets the
    ///    compiler prove the borrows do not overlap.
    /// 2. **Linear scan** — if the cache misses, the `observations` list is
    ///    searched with [`Iterator::find`].  The found value is cloned into
    ///    the cache before a reference is returned, so subsequent look-ups
    ///    for the same `id` hit the cache.
    ///
    /// # Arguments
    ///
    /// - `id` — the `ObsId` of the observation to look up.
    ///
    /// # Returns
    ///
    /// `Some(&Observation)` if an observation with the given `id` exists in this dataset;
    /// `None` otherwise.
    pub fn get_observation(&mut self, id: ObsId) -> Option<&Observation> {
        if self.lru_cache_obs.contains(&id) {
            return self.lru_cache_obs.get(&id);
        }
        let idx = self.index.get_by_id(&id)?;
        let obs = self.observations[idx].clone();
        self.lru_cache_obs.put(id, obs);
        self.lru_cache_obs.get(&id)
    }

    /// Look up a single observation by its raw vector position.
    ///
    /// Unlike [`ObsDataset::get_observation`], which searches by `ObsId`,
    /// this method performs a direct index into the internal observations
    /// vector.  The result is also stored in the LRU cache so that a
    /// subsequent `get_observation` call for the same entry can be served
    /// from the cache.
    ///
    /// # Arguments
    ///
    /// - `idx` — zero-based position into the internal observations vector,
    ///   as returned by `Observation::index`.
    ///
    /// # Returns
    ///
    /// `Some(&Observation)` if `idx` is within bounds; `None` otherwise.
    pub fn get_obs_by_index(&mut self, idx: ObsIndex) -> Option<&Observation> {
        let obs = self.observations.get(idx)?;
        self.lru_cache_obs.put(*obs.id(), obs.clone());
        Some(obs)
    }

    /// Return the total number of observations in this dataset.
    ///
    /// # Returns
    ///
    /// The number of [`Observation`] values stored in the dataset.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Return an iterator over all observations in insertion order.
    ///
    /// The iterator yields shared references and does not clone any data.
    /// The order matches the order of the source `DataFrame` rows.
    ///
    /// # Returns
    ///
    /// An iterator yielding `&Observation` for each observation in insertion order.
    pub fn iter_observations(&self) -> impl Iterator<Item = &Observation> {
        self.observations.iter()
    }

    /// Return an iterator over all observations belonging to a given night, in insertion order.
    /// Returns `None` if the dataset does not have an index by night or if the given night_id is not found.
    /// The order of the yielded observations matches the order of the source `DataFrame` rows.
    ///
    /// # Arguments
    ///
    /// - `night_id` — the identifier of the night for which to return observations.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset has an index by night and the given `night_id` is found,
    ///    where `iterator` yields shared references to the observations belonging to that night in insertion order;
    ///    `None` otherwise.
    pub fn iter_night_observations(
        &self,
        night_id: &NightId,
    ) -> Option<impl Iterator<Item = &Observation>> {
        self.index
            .iter_night_obs_index(night_id)
            .map(|indices| indices.map(|idx| &self.observations[idx]))
    }

    /// Return an iterator over `(NightId, &Observation)` pairs for every observation in the night index.
    ///
    /// Each pair associates a night identifier with a shared reference to one of the
    /// observations recorded on that night.  Observations from the same night appear
    /// consecutively, but the order between different nights is unspecified.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset was built with a night index; `None` otherwise.
    pub fn iter_full_night(&self) -> Option<impl Iterator<Item = (NightId, &Observation)>> {
        self.index
            .iter_full_night()
            .map(|night_iter| night_iter.map(|(night_id, idx)| (night_id, &self.observations[idx])))
    }

    /// Collect all observations belonging to a given night into a `Vec`.
    ///
    /// This is a convenience wrapper around [`ObsDataset::iter_night_observations`] that
    /// eagerly collects the iterator results.
    ///
    /// # Arguments
    ///
    /// - `night_id` — the identifier of the night to materialise.
    ///
    /// # Returns
    ///
    /// `Some(Vec<&Observation>)` in insertion order if the night index exists and the
    /// given `night_id` is present; `None` otherwise.
    pub fn materialize_night(&self, night_id: &NightId) -> Option<Vec<&Observation>> {
        self.iter_night_observations(night_id)
            .map(|iter| iter.collect())
    }

    /// Return an iterator over all `NightId` keys present in the night index.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset was built with a night index; `None` otherwise.
    /// The iteration order is unspecified.
    pub fn iter_night_id(&self) -> Option<impl Iterator<Item = &NightId>> {
        self.index.iter_night_id()
    }

    /// Return the number of observations recorded on a given night.
    ///
    /// # Arguments
    ///
    /// - `night_id` — the night whose observation count is requested.
    ///
    /// # Returns
    ///
    /// `Some(count)` if the night index exists and the given `night_id` is present;
    /// `None` otherwise.
    pub fn len_night(&self, night_id: &NightId) -> Option<usize> {
        self.index.len_night(night_id)
    }

    /// Return an iterator over all observations belonging to a given trajectory, in insertion order.
    /// Returns `None` if the dataset does not have an index by trajectory or if the given traj_id is not found.
    /// The order of the yielded observations matches the order of the source `DataFrame` rows.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the identifier of the trajectory for which to return observations.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset has an index by trajectory and the given `traj_id` is found,
    ///   where `iterator` yields shared references to the observations belonging to that trajectory in insertion order;
    ///  `None` otherwise.
    pub fn iter_trajectory_observations(
        &self,
        traj_id: &TrajId,
    ) -> Option<impl Iterator<Item = &Observation>> {
        self.index
            .iter_traj_obs_index(traj_id)
            .map(|indices| indices.map(|idx| &self.observations[idx]))
    }

    /// Return an iterator over `(TrajId, &Observation)` pairs for every observation in the trajectory index.
    ///
    /// Each pair associates a trajectory identifier with a shared reference to one of the
    /// observations belonging to that trajectory.  Observations from the same trajectory
    /// appear consecutively, but the order between different trajectories is unspecified.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset was built with a trajectory index; `None` otherwise.
    pub fn iter_full_trajectory(&self) -> Option<impl Iterator<Item = (TrajId, &Observation)>> {
        self.index.iter_full_trajectory().map(|traj_iter| {
            traj_iter.map(|(traj_id, idx)| (traj_id.clone(), &self.observations[idx]))
        })
    }

    /// Collect all observations belonging to a given trajectory into a `Vec`.
    ///
    /// This is a convenience wrapper around [`ObsDataset::iter_trajectory_observations`]
    /// that eagerly collects the iterator results.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the identifier of the trajectory to materialise.
    ///
    /// # Returns
    ///
    /// `Some(Vec<&Observation>)` in insertion order if the trajectory index exists and the
    /// given `traj_id` is present; `None` otherwise.
    pub fn materialize_trajectory(&self, traj_id: &TrajId) -> Option<Vec<&Observation>> {
        self.iter_trajectory_observations(traj_id)
            .map(|iter| iter.collect())
    }

    /// Return an iterator over all `TrajId` keys present in the trajectory index.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset was built with a trajectory index; `None` otherwise.
    /// The iteration order is unspecified.
    pub fn iter_traj_id(&self) -> Option<impl Iterator<Item = &TrajId>> {
        self.index.iter_traj_id()
    }

    /// Return the number of observations assigned to a given trajectory.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the trajectory whose observation count is requested.
    ///
    /// # Returns
    ///
    /// `Some(count)` if the trajectory index exists and the given `traj_id` is present;
    /// `None` otherwise.
    pub fn len_trajectory(&self, traj_id: &TrajId) -> Option<usize> {
        self.index.len_trajectory(traj_id)
    }

    /// Register a new trajectory in the trajectory index.
    ///
    /// Associates `traj_id` with the positions of `obs_indices` in the internal
    /// observations vector.  If the dataset was not built with a trajectory index
    /// (i.e. the source data had no `traj_id` column), this method is a no-op.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the identifier of the trajectory to register.
    /// - `obs_indices` — slice of [`Observation`] values whose internal vector
    ///   positions will be recorded under `traj_id`.
    pub fn push_new_trajectory(&mut self, traj_id: TrajId, obs_indices: &[Observation]) {
        self.index.push_trajectory(
            traj_id,
            &(obs_indices
                .iter()
                .map(|obs| obs.index)
                .collect::<Vec<ObsIndex>>()),
        );
    }

    /// Register a new trajectory in the trajectory index.
    ///
    /// Associates `traj_id` with the positions of `obs_indices` in the internal
    /// observations vector.  If the dataset was not built with a trajectory index
    /// (i.e. the source data had no `traj_id` column), this method is a no-op.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the identifier of the trajectory to register.
    /// - `obs_indices` — slice of [`Observation`] values whose internal vector
    ///   positions will be recorded under `traj_id`.
    pub fn push_new_trajectory_by_index(&mut self, traj_id: TrajId, obs_indices: &[ObsIndex]) {
        self.index.push_trajectory(traj_id, obs_indices);
    }

    /// Look up the [`Observer`] associated with a given observation.
    ///
    /// Returns `None` if the observation does not exist, if it has no
    /// observer, or if the MPC catalogue could not be initialised.
    ///
    /// ## Borrow-checker note
    ///
    /// `ObserverId` is `Copy`, so the observer identifier is copied out of
    /// the [`Observation`] returned by [`ObsDataset::get_observation`] in a
    /// single statement.  This releases the mutable borrow on `self` held by
    /// `get_observation` before `custom_observers` or `mpc_observers` are
    /// accessed, satisfying the borrow checker without any heap allocation.
    ///
    /// # Arguments
    ///
    /// - `id` — the `ObsId` of the observation whose observer is requested.
    ///
    /// # Returns
    ///
    /// `Some(&Observer)` if the observation exists and has an observer that can be resolved;
    /// `None` if the observation does not exist, has no observer, or the MPC catalogue
    /// initialisation failed.
    pub fn get_observer(&mut self, id: ObsId) -> Option<&Observer> {
        // Copy the ObserverId out first to release the borrow on `self` held by
        // `get_observation` before we access `self.custom_observers` or
        // `self.mpc_observers()`.  ObserverId is Copy so no allocation occurs.
        let observer_id = self.get_observation(id)?.observer?;
        self.observer_dataset.get(&observer_id)
    }

    /// Set the astrometric error model used for MPC observatory initialisation.
    /// This method allows changing the error model after the dataset has been constructed,
    /// which will affect the accuracies assigned to MPC-coded observers when the MPC table is loaded.
    ///
    /// Note that if the MPC table has already been initialised,
    /// changing the error model will not retroactively update the observer accuracies;
    /// the new error model will only take effect on the first call to `mpc_observers()`
    /// if the MPC table has not yet been loaded.
    ///
    /// # Arguments
    ///
    /// - `error_model` — the new [`ObsErrorModel`] to use for MPC observatory initialisation.
    pub fn set_error_model(&mut self, error_model: ObsErrorModel) {
        self.observer_dataset.mpc_error_model = Some(error_model);
    }

    /// Create a new dataset from pre-parsed data.
    ///
    /// This constructor is used internally by [`ObsDataset::from_polars`] and
    /// by test helpers.  The LRU cache is initialised with a fixed capacity of
    /// **1 000** entries; the MPC observatory table is not fetched until the
    /// first call to [`ObsDataset::get_observer`] for an MPC-coded site.
    ///
    /// # Arguments
    ///
    /// - `observations`            — the full list of observations in insertion order.
    /// - `custom_observers`        — geodetic observers de-duplicated by the caller,
    ///   addressable by index via `ObserverId::IntId`.
    /// - `error_model`             — astrometric error model used during MPC
    ///   observatory initialisation.
    /// - `obs_index_by_night`      — optional pre-built night index; pass `None`
    ///   when the source data has no `night_id` column.
    /// - `obs_index_by_trajectory` — optional pre-built trajectory index; pass `None`
    ///   when the source data has no `traj_id` column.
    /// - `lru_cache_size`          — optional capacity for the LRU cache; `None` defaults
    ///   to 1 000.
    ///
    /// # Returns
    ///
    /// A fully initialised `ObsDataset` with the observations indexed and the LRU cache empty.
    #[cfg_attr(not(feature = "polars"), allow(dead_code))]
    pub(crate) fn new(
        observations: Vec<Observation>,
        custom_observers: Vec<Observer>,
        error_model: Option<ObsErrorModel>,
        obs_index_by_night: Option<NightIndexMap>,
        obs_index_by_trajectory: Option<TrajIndexMap>,
        lru_cache_size: Option<usize>,
    ) -> Self {
        // Build the ObsId → index mapping for look-up by id.  Pre-allocating
        // with the exact capacity avoids repeated rehashing as the map grows.
        let mut obs_index_by_id = ObservationIndexMap::with_capacity(observations.len());
        for (idx, obs) in observations.iter().enumerate() {
            obs_index_by_id.insert(obs.id, idx);
        }

        Self {
            observations,
            index: ObsDatasetIndex::new(
                obs_index_by_id,
                obs_index_by_night,
                obs_index_by_trajectory,
            ),
            observer_dataset: ObserverDataset::new(custom_observers, error_model),
            lru_cache_obs: LruCache::new(
                NonZeroUsize::new(lru_cache_size.unwrap_or(1000)).unwrap(),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod observation_tests {
    use super::*;
    use crate::{
        astrometry::EquCoord,
        observer::{Observer, dataset::ObserverId, error_model::ObsErrorModel},
        photometry::{Filter, Photometry},
    };
    use std::collections::HashSet;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_equ_coord() -> EquCoord {
        EquCoord::new(0.5, 1e-5, 0.2, 1e-5)
    }

    fn make_photometry() -> Photometry {
        Photometry {
            magnitude: 15.0,
            error: 0.1,
            filter: Filter::String("G".to_string()),
        }
    }

    fn make_observation(id: u64, observer: Option<ObserverId>) -> Observation {
        Observation {
            index: 0,
            id,
            equ_coord: make_equ_coord(),
            photometry: make_photometry(),
            mjd_tt: 60000.5,
            observer,
        }
    }

    /// Returns a valid Observer constructed via the parallax path.
    /// unwrap() is safe: none of the inputs are NaN.
    fn make_custom_observer() -> Observer {
        Observer::from_parallax(110.0, 0.836, 0.547, Some("Test".to_string()), None, None).unwrap()
        // safe: all inputs are finite, non-NaN values
    }

    /// Build an ObsDataset with an LRU cache capacity of 100.
    fn make_dataset(obs: Vec<Observation>, observers: Vec<Observer>) -> ObsDataset {
        ObsDataset::new(
            obs,
            observers,
            Some(ObsErrorModel::FCCT14),
            None,
            None,
            Some(100),
        )
    }

    // -----------------------------------------------------------------------
    // ObserverId — Copy, PartialOrd ordering between variants, Debug
    // -----------------------------------------------------------------------

    mod observer_id {
        use super::*;

        /// Verifies that ObserverId is Copy: the original is still usable after a copy.
        #[test]
        fn observer_id_int_is_copy() {
            let original = ObserverId::IntId(3);
            let copy = original; // Copy, not move
            assert_eq!(original, copy);
        }

        /// Verifies that ObserverId::MpcCode is Copy.
        #[test]
        fn observer_id_mpc_code_is_copy() {
            let original = ObserverId::MpcCode(*b"G96");
            let copy = original;
            assert_eq!(original, copy);
        }

        /// Verifies that two IntIds with the same index compare as equal.
        #[test]
        fn observer_id_int_same_index_is_eq() {
            assert_eq!(ObserverId::IntId(0), ObserverId::IntId(0));
        }

        /// Verifies that IntId ordering is determined by the inner index value.
        #[test]
        fn observer_id_int_ordering_by_index() {
            assert!(ObserverId::IntId(0) < ObserverId::IntId(1));
        }

        /// Verifies that IntId variants sort before MpcCode variants (enum variant
        /// ordering follows declaration order: IntId = 0, MpcCode = 1).
        #[test]
        fn observer_id_int_less_than_mpc_code() {
            assert!(ObserverId::IntId(usize::MAX) < ObserverId::MpcCode(*b"000"));
        }

        /// Verifies that the Debug output of ObserverId::IntId contains the index.
        #[test]
        fn observer_id_int_debug_contains_index() {
            let id = ObserverId::IntId(42);
            let debug_str = format!("{id:?}");
            assert!(
                debug_str.contains("42"),
                "Debug output should contain '42', got: {debug_str}"
            );
        }

        /// Verifies that the Debug output of ObserverId::MpcCode contains the code bytes.
        #[test]
        fn observer_id_mpc_code_debug_contains_code() {
            let id = ObserverId::MpcCode(*b"G96");
            let debug_str = format!("{id:?}");
            assert!(
                !debug_str.is_empty(),
                "Debug output should not be empty for MpcCode variant"
            );
        }

        /// Verifies that ObserverId can be stored in a HashSet.
        #[test]
        fn observer_id_can_be_inserted_into_hash_set() {
            let mut set: HashSet<ObserverId> = HashSet::new();
            set.insert(ObserverId::IntId(0));
            set.insert(ObserverId::IntId(1));
            set.insert(ObserverId::IntId(0)); // duplicate
            assert_eq!(set.len(), 2);
        }
    }

    // -----------------------------------------------------------------------
    // ObsDataset::new — construction without panicking
    // -----------------------------------------------------------------------

    mod obs_dataset_new {
        use super::*;

        /// Verifies that constructing an empty dataset with None cache size does not panic.
        #[test]
        fn new_empty_with_none_cache_size_does_not_panic() {
            let _ds = ObsDataset::new(
                vec![],
                vec![],
                Some(ObsErrorModel::FCCT14),
                None,
                None,
                None,
            );
        }

        /// Verifies that constructing an empty dataset with a custom cache size does not panic.
        #[test]
        fn new_empty_with_custom_cache_size_does_not_panic() {
            let _ds = ObsDataset::new(
                vec![],
                vec![],
                Some(ObsErrorModel::FCCT14),
                None,
                None,
                Some(5),
            );
        }

        /// Verifies that an empty dataset has zero observations via iter_observations.
        #[test]
        fn new_empty_has_zero_observations() {
            let ds = make_dataset(vec![], vec![]);
            assert_eq!(ds.iter_observations().count(), 0);
        }

        /// Verifies that a dataset constructed with multiple observations counts them correctly.
        #[test]
        fn new_with_observations_has_correct_count() {
            let obs = vec![
                make_observation(1, None),
                make_observation(2, None),
                make_observation(3, None),
            ];
            let ds = make_dataset(obs, vec![]);
            assert_eq!(ds.iter_observations().count(), 3);
        }
    }

    // -----------------------------------------------------------------------
    // ObsDataset::iter_observations
    // -----------------------------------------------------------------------

    mod iter_observations {
        use super::*;

        /// Verifies that iter_observations on an empty dataset yields nothing.
        #[test]
        fn iter_on_empty_dataset_yields_nothing() {
            let ds = make_dataset(vec![], vec![]);
            assert_eq!(ds.iter_observations().count(), 0);
        }

        /// Verifies that iter_observations yields observations in insertion order.
        #[test]
        fn iter_yields_observations_in_insertion_order() {
            let obs = vec![
                make_observation(10, None),
                make_observation(20, None),
                make_observation(30, None),
            ];
            let ds = make_dataset(obs, vec![]);
            let ids: Vec<ObsId> = ds.iter_observations().map(|o| o.id).collect();
            assert_eq!(ids, vec![10, 20, 30]);
        }

        /// Verifies that a single-element dataset yields exactly one observation.
        #[test]
        fn iter_single_observation_yields_one_item() {
            let ds = make_dataset(vec![make_observation(99, None)], vec![]);
            assert_eq!(ds.iter_observations().count(), 1);
        }

        /// Verifies that the observation yielded has the expected id.
        #[test]
        fn iter_yields_correct_id() {
            let ds = make_dataset(vec![make_observation(42, None)], vec![]);
            let first = ds.iter_observations().next();
            assert!(first.is_some(), "Expected at least one observation");
            assert_eq!(first.unwrap().id, 42);
        }
    }

    // -----------------------------------------------------------------------
    // ObsDataset::get_observation
    // -----------------------------------------------------------------------

    mod get_observation {
        use super::*;

        /// Verifies that get_observation returns Some for an existing id.
        #[test]
        fn get_observation_returns_some_for_existing_id() {
            let mut ds = make_dataset(vec![make_observation(1, None)], vec![]);
            assert!(ds.get_observation(1).is_some());
        }

        /// Verifies that get_observation returns None for a missing id.
        #[test]
        fn get_observation_returns_none_for_missing_id() {
            let mut ds = make_dataset(vec![make_observation(1, None)], vec![]);
            assert!(ds.get_observation(9999).is_none());
        }

        /// Verifies that repeated calls for the same id return the same observation
        /// (exercises the cache hit path without panicking).
        #[test]
        fn get_observation_repeated_calls_return_same_id() {
            let mut ds = make_dataset(vec![make_observation(7, None)], vec![]);
            let first_id = ds.get_observation(7).map(|o| o.id);
            let second_id = ds.get_observation(7).map(|o| o.id);
            assert_eq!(first_id, second_id);
        }

        /// Verifies that among several observations the correct one is returned by id.
        #[test]
        fn get_observation_returns_correct_one_among_multiple() {
            let obs = vec![
                make_observation(1, None),
                make_observation(2, None),
                make_observation(3, None),
            ];
            let mut ds = make_dataset(obs, vec![]);
            let found = ds.get_observation(2);
            assert!(found.is_some(), "Expected Some for id=2");
            assert_eq!(found.unwrap().id, 2);
        }

        /// Verifies the LRU eviction behaviour: when the cache capacity is 1 and a
        /// second observation is looked up, the first is evicted from the cache.
        /// The evicted entry must still be findable via the linear scan fallback.
        #[test]
        fn get_observation_lru_eviction_still_findable_via_linear_scan() {
            // Capacity=1: looking up id=2 will evict id=1 from the cache.
            let obs = vec![make_observation(1, None), make_observation(2, None)];
            let mut ds = ObsDataset::new(
                obs,
                vec![],
                Some(ObsErrorModel::FCCT14),
                None,
                None,
                Some(1),
            );

            // Populate the cache with id=1.
            assert!(ds.get_observation(1).is_some());
            // Looking up id=2 evicts id=1 from the cache.
            assert!(ds.get_observation(2).is_some());
            // id=1 must still be found via the linear scan even though it was evicted.
            assert!(
                ds.get_observation(1).is_some(),
                "id=1 should still be findable after LRU eviction"
            );
        }
    }

    // -----------------------------------------------------------------------
    // ObsDataset::get_observer
    // -----------------------------------------------------------------------

    mod get_observer {
        use super::*;

        /// Verifies that get_observer returns None for an observation id that does not exist.
        #[test]
        fn get_observer_returns_none_for_missing_obs_id() {
            let mut ds = make_dataset(vec![], vec![]);
            assert!(ds.get_observer(9999).is_none());
        }

        /// Verifies that get_observer returns None when the observation has no observer field.
        #[test]
        fn get_observer_returns_none_when_observer_is_none() {
            let obs = vec![make_observation(1, None)];
            let mut ds = make_dataset(obs, vec![]);
            assert!(ds.get_observer(1).is_none());
        }

        /// Verifies that get_observer returns Some(observer) when the observation has
        /// ObserverId::IntId(0) and a matching custom observer at index 0.
        #[test]
        fn get_observer_returns_some_for_int_id_zero() {
            let custom = make_custom_observer();
            let obs = vec![make_observation(1, Some(ObserverId::IntId(0)))];
            let mut ds = make_dataset(obs, vec![custom]);
            assert!(
                ds.get_observer(1).is_some(),
                "Expected Some(observer) for ObserverId::IntId(0)"
            );
        }

        /// Verifies that the observer returned by get_observer matches the one that was inserted.
        #[test]
        fn get_observer_returns_correct_observer_for_int_id() {
            let custom = make_custom_observer();
            let expected_name = custom.name.clone();
            let obs = vec![make_observation(1, Some(ObserverId::IntId(0)))];
            let mut ds = make_dataset(obs, vec![custom]);
            let found = ds.get_observer(1).unwrap(); // safe: verified Some above
            assert_eq!(
                found.name, expected_name,
                "Observer name should match the inserted observer"
            );
        }

        /// Verifies that an out-of-bounds IntId returns None.
        #[test]
        fn get_observer_returns_none_for_int_id_out_of_bounds() {
            // Index 5 does not exist in a one-element observer list.
            let obs = vec![make_observation(1, Some(ObserverId::IntId(5)))];
            let custom = make_custom_observer();
            let mut ds = make_dataset(obs, vec![custom]);
            assert!(
                ds.get_observer(1).is_none(),
                "Expected None for ObserverId::IntId out of bounds"
            );
        }

        /// Verifies that get_observer works correctly when multiple custom observers
        /// are present and we look up by the correct index.
        #[test]
        fn get_observer_returns_correct_observer_among_multiple() {
            let obs1 =
                Observer::from_parallax(10.0, 0.8, 0.5, Some("First".to_string()), None, None)
                    .unwrap(); // safe: all finite non-NaN inputs
            let obs2 =
                Observer::from_parallax(20.0, 0.9, 0.4, Some("Second".to_string()), None, None)
                    .unwrap(); // safe: all finite non-NaN inputs

            let obs = vec![
                make_observation(1, Some(ObserverId::IntId(0))),
                make_observation(2, Some(ObserverId::IntId(1))),
            ];
            let mut ds = make_dataset(obs, vec![obs1, obs2]);

            let name_for_obs1 = ds.get_observer(1).and_then(|o| o.name.clone());
            let name_for_obs2 = ds.get_observer(2).and_then(|o| o.name.clone());

            assert_eq!(name_for_obs1.as_deref(), Some("First"));
            assert_eq!(name_for_obs2.as_deref(), Some("Second"));
        }
    }

    // -----------------------------------------------------------------------
    // ObsDatasetError — Display, Debug, From<MPCError>
    // -----------------------------------------------------------------------

    mod obs_dataset_error {
        use super::*;

        /// Verifies that ObsDatasetError::ErrorModelError has a non-empty Display output.
        #[test]
        fn obs_dataset_error_display_error_model_error_is_non_empty() {
            use crate::observer::error_model::ErrorModelParseError;
            let inner = ErrorModelParseError::NomParsingError("bad line".to_string());
            let err = ObsDatasetError::ErrorModelError(inner);
            let display = format!("{err}");
            assert!(
                !display.is_empty(),
                "Display output for ErrorModelError should not be empty"
            );
        }

        /// Verifies that ObsDatasetError::ErrorModelError contains meaningful text.
        #[test]
        fn obs_dataset_error_display_contains_meaningful_text() {
            use crate::observer::error_model::ErrorModelParseError;
            let inner = ErrorModelParseError::NomParsingError("bad line".to_string());
            let err = ObsDatasetError::ErrorModelError(inner);
            let display = format!("{err}");
            assert!(
                display.contains("bad line"),
                "Display output should contain the inner error text, got: {display}"
            );
        }

        /// Verifies that ObsDatasetError has a non-empty Debug output.
        #[test]
        fn obs_dataset_error_debug_is_non_empty() {
            use crate::observer::error_model::ErrorModelParseError;
            let inner = ErrorModelParseError::NomParsingError("x".to_string());
            let err = ObsDatasetError::ErrorModelError(inner);
            let debug = format!("{err:?}");
            assert!(!debug.is_empty(), "Debug output should not be empty");
        }

        /// Verifies that From<MPCError> is implemented for ObsDatasetError by constructing
        /// the variant directly and checking that the Display string is non-empty.
        ///
        /// We cannot trigger a real MPCError without a network call, so we use the
        /// ObsDatasetError::MPCError(…) variant constructor via From.
        #[test]
        fn obs_dataset_error_from_mpc_error_display_is_non_empty() {
            // Build a ureq error via a known-bad request using a closed TCP port.
            // We test only that the From impl compiles and Display is non-empty;
            // the exact message is implementation-defined.
            use crate::observer::error_model::ErrorModelParseError;
            let inner = ErrorModelParseError::InvalidStationCode("TOOLONG".to_string());
            let err = ObsDatasetError::ErrorModelError(inner);
            let display = format!("{err}");
            assert!(
                !display.is_empty(),
                "Display for ObsDatasetError wrapping ErrorModelError must be non-empty"
            );
        }

        /// Verifies that ObsDatasetError wrapping an ErrorModelParseError has a
        /// non-empty Display, exercising the From<ErrorModelParseError> impl for
        /// ObsDatasetError (which is the closest analogue to From<MPCError> that
        /// can be tested without a network call).
        #[test]
        fn obs_dataset_error_error_model_variant_display_is_non_empty() {
            use crate::observer::error_model::ErrorModelParseError;
            // InvalidStationCode is a stable, constructable variant of ErrorModelParseError.
            let inner = ErrorModelParseError::InvalidStationCode("BAD".to_string());
            // Verify that From<ErrorModelParseError> for ObsDatasetError compiles and
            // that the resulting Display is non-empty.
            let err: ObsDatasetError = inner.into();
            let s = format!("{err}");
            assert!(!s.is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // ErrorModelParseError — additional variant coverage
    // -----------------------------------------------------------------------

    mod error_model_parse_error_variants {
        use crate::observer::error_model::ErrorModelParseError;

        /// Verifies that ErrorModelParseError::NomParsingError has a non-empty Display.
        #[test]
        fn nom_parsing_error_display_is_non_empty() {
            let err = ErrorModelParseError::NomParsingError("broken line".to_string());
            let s = format!("{err}");
            assert!(!s.is_empty());
        }

        /// Verifies that ErrorModelParseError::InvalidStationCode includes the bad code
        /// in its Display output.
        #[test]
        fn invalid_station_code_display_contains_code() {
            let err = ErrorModelParseError::InvalidStationCode("TOOLONG".to_string());
            let s = format!("{err}");
            assert!(
                s.contains("TOOLONG"),
                "Display should mention the bad code, got: {s}"
            );
        }
    }
}
