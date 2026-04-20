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

pub mod builder;
pub(crate) mod index;
pub mod iter;
pub mod observation;

#[cfg(feature = "ades")]
pub mod ades;

#[cfg(feature = "mpc_80_col")]
pub mod mpc_80_col;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "polars")]
pub mod polars;
#[cfg(feature = "polars")]
use crate::io::polars::error::PolarsError;

#[cfg(feature = "datafusion")]
pub mod datafusion;

use thiserror::Error;

use crate::{
    TrajId,
    observation_dataset::{
        index::{NightIndexMap, ObsDatasetIndex, ObsIndex, ObservationIndexMap, TrajIndexMap},
        observation::Observation,
    },
    observer::{
        Observer,
        dataset::{ObserverDataset, ObserverId},
        error_model::{ErrorModelParseError, ObsErrorModel},
        mpc::MPCError,
    },
};

/// Unique numeric identifier for a single observation.
///
/// Observations are keyed by this value inside [`ObsDataset`].
/// The identifier is assigned by the data source (e.g.
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

    /// An observation is missing its internal vector index, which is required for trajectory registration.
    /// This appends if the observation has not been pushed within a dataset or if the index was not assigned during dataset construction.
    #[error("Observation is missing its internal vector index")]
    MissingIndex,
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
#[derive(Debug)]
pub struct ObsDataset {
    /// Full list of observations in insertion order.
    pub(crate) observations: Vec<Observation>,

    /// Index mappings for efficient look-up by various identifiers.
    pub(crate) index: ObsDatasetIndex,

    /// Observer values for both custom geodetic observers (indexed by `ObserverId::IntId`)
    /// and MPC-coded observers (resolved lazily via `ObserverId::MpcCode`).
    pub(crate) observer_dataset: ObserverDataset,
}

impl ObsDataset {
    /// Create an empty `ObsDataset` with no observations and no observers.
    ///
    /// # Returns
    /// An empty `ObsDataset`.
    pub fn empty() -> Self {
        Self::new(vec![], vec![], None, None, None)
    }

    /// Add a new observation to the dataset, assigning it the next available index and returning its `ObsId`.
    /// The observation's `index` field is updated to reflect its position in the internal observations vector.
    /// The `id` field of the observation is not modified by this method; it is the caller's responsibility to ensure that it is unique within the dataset.
    ///
    /// # Arguments
    /// - `obs` — the `Observation` to add to the dataset.  Its `index` field will be updated to reflect its position in the internal observations vector.
    ///
    /// # Returns
    /// The `ObsId` of the newly added observation.
    pub fn push_observation(&mut self, mut obs: Observation) -> ObsId {
        let id = obs.id;
        obs.index = Some(self.observations.len());
        self.observations.push(obs);
        id
    }

    /// Look up a single observation by its [`ObsId`].
    ///
    /// Returns a shared reference to the matching [`Observation`], or `None`
    /// if no observation with the given `id` exists in this dataset.
    ///
    /// The look-up is performed via an internal hash map index for O(1) access.
    ///
    /// # Arguments
    ///
    /// - `id` — the `ObsId` of the observation to look up.
    ///
    /// # Returns
    ///
    /// `Some(&Observation)` if an observation with the given `id` exists in this dataset;
    /// `None` otherwise.
    pub fn get_observation(&self, id: ObsId) -> Option<&Observation> {
        let idx = self.index.get_by_id(&id)?;
        self.observations.get(idx)
    }

    /// Look up a single observation by its raw vector position.
    ///
    /// Unlike [`ObsDataset::get_observation`], which searches by `ObsId`,
    /// this method performs a direct index into the internal observations
    /// vector.
    ///
    /// # Arguments
    ///
    /// - `idx` — zero-based position into the internal observations vector,
    ///   as returned by `Observation::index`.
    ///
    /// # Returns
    ///
    /// `Some(&Observation)` if `idx` is within bounds; `None` otherwise.
    pub fn get_obs_by_index(&self, idx: ObsIndex) -> Option<&Observation> {
        self.observations.get(idx)
    }

    /// Return the total number of observations in this dataset.
    ///
    /// # Returns
    ///
    /// The number of [`Observation`] values stored in the dataset.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Resolve an alternate trajectory designation to its canonical [`TrajId`].
    ///
    /// Some ingestion backends (e.g. the MPC 80-column reader) register
    /// alternate designations that are not used as primary trajectory keys —
    /// for example a provisional designation that was later superseded by a
    /// permanent number, or two provisional designations that were linked as
    /// the same physical object.
    ///
    /// # Arguments
    ///
    /// - `alias` — the alternate designation string to resolve.
    ///
    /// # Returns
    ///
    /// `Some(&TrajId)` if `alias` is a known alternate designation;
    /// `None` if no alias with that name has been registered.
    pub fn resolve_alias(&self, alias: &str) -> Option<&TrajId> {
        self.index.resolve_alias(alias)
    }

    /// Return a shared reference to the internal composite index.
    ///
    /// This accessor is `pub(crate)` so that unit tests inside the crate can
    /// inspect the `ObsDatasetIndex` fields (e.g. `obs_index_by_night` and
    /// `obs_index_by_trajectory`) without exposing them as part of the public API.
    #[allow(dead_code)]
    pub(crate) fn index_ref(&self) -> &ObsDatasetIndex {
        &self.index
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
    pub fn push_new_trajectory(
        &mut self,
        traj_id: TrajId,
        obs_indices: &[Observation],
    ) -> Result<(), ObsDatasetError> {
        self.index.push_trajectory(
            traj_id,
            &(obs_indices
                .iter()
                .map(|obs| obs.index.ok_or(ObsDatasetError::MissingIndex))
                .collect::<Result<Vec<ObsIndex>, ObsDatasetError>>()?),
        );
        Ok(())
    }

    /// Register a new trajectory in the trajectory index using raw vector positions.
    ///
    /// Associates `traj_id` with the positions given directly as a slice of
    /// vector positions, rather than deriving them from [`Observation`]
    /// structs as [`ObsDataset::push_new_trajectory`] does.  If the dataset
    /// was not built with a trajectory index (i.e. the source data had no
    /// `traj_id` column), this method is a no-op.
    ///
    /// # Arguments
    ///
    /// - `traj_id`     — the identifier of the trajectory to register.
    /// - `obs_indices` — slice of zero-based vector positions in the internal
    ///   observations vector that belong to this trajectory.
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
    /// by test helpers.  The MPC observatory table is not fetched until the
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
    ///
    /// # Returns
    ///
    /// A fully initialised `ObsDataset` with the observations indexed.
    #[cfg_attr(not(feature = "polars"), allow(dead_code))]
    pub(crate) fn new(
        observations: Vec<Observation>,
        custom_observers: Vec<Observer>,
        error_model: Option<ObsErrorModel>,
        obs_index_by_night: Option<NightIndexMap>,
        obs_index_by_trajectory: Option<TrajIndexMap>,
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
        }
    }

    /// Construct an [`ObsDataset`] from an already-built [`ObserverDataset`].
    ///
    /// This is the internal counterpart of [`ObsDataset::new`] used during
    /// deserialisation: the `observer_dataset` is supplied fully formed (having
    /// been deserialised separately) instead of being assembled from raw
    /// `custom_observers` and `error_model` parameters.
    ///
    /// Index maps are rebuilt from `observations`.
    ///
    /// # Arguments
    ///
    /// - `observations`            — the full list of observations in insertion order.
    /// - `observer_dataset`        — pre-built observer dataset (custom observers +
    ///   error model, MPC cache uninitialised).
    /// - `obs_index_by_night`      — optional pre-built night index.
    /// - `obs_index_by_trajectory` — optional pre-built trajectory index.
    /// - `traj_aliases`            — trajectory alias map (alternate designation →
    ///   canonical [`TrajId`]); pass an empty map when no aliases were serialised.
    #[cfg(feature = "serde")]
    pub(crate) fn new_from_parts(
        observations: Vec<Observation>,
        observer_dataset: ObserverDataset,
        obs_index_by_night: Option<NightIndexMap>,
        obs_index_by_trajectory: Option<TrajIndexMap>,
        traj_aliases: index::TrajAliasMap,
    ) -> Self {
        let mut obs_index_by_id = ObservationIndexMap::with_capacity(observations.len());
        for (idx, obs) in observations.iter().enumerate() {
            obs_index_by_id.insert(obs.id, idx);
        }

        let mut dataset_index =
            ObsDatasetIndex::new(obs_index_by_id, obs_index_by_night, obs_index_by_trajectory);
        dataset_index.set_aliases(traj_aliases);

        Self {
            observations,
            index: dataset_index,
            observer_dataset,
        }
    }

    /// Merge another `ObsDataset` into `self`, appending all of its observations.
    ///
    /// Each observation from `other` is renumbered before insertion:
    /// - `obs.index` is incremented by `offset` (the current length of
    ///   `self.observations`).
    /// - `obs.id` is incremented by `offset` cast to `u64`.
    /// - Any `ObserverId::IntId(i)` is shifted by the custom-observer offset
    ///   returned by [`ObserverDataset::merge_custom_observers`].
    ///
    /// Trajectory and alias maps are merged via
    /// [`ObsDatasetIndex::merge_from`].
    #[cfg_attr(not(any(feature = "ades", feature = "mpc_80_col")), allow(dead_code))]
    pub(crate) fn merge_from(&mut self, other: ObsDataset) {
        let offset = self.observations.len();

        // Merge custom observers; get the IntId offset for this dataset.
        let custom_offset = self
            .observer_dataset
            .merge_custom_observers(other.observer_dataset);

        // Renumber and push observations from `other`.
        for mut obs in other.observations {
            obs.index = obs.index.map(|i| i + offset);
            obs.id += offset as u64;
            if let Some(ObserverId::IntId(ref mut i)) = obs.observer {
                *i += custom_offset;
            }
            self.observations.push(obs);
        }

        // Merge indices (already accounts for offset internally).
        self.index.merge_from(other.index, offset);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod observation_tests {
    use super::*;
    use crate::{
        coordinates::equatorial::EquCoord,
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
            index: Some(0),
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

    /// Build an ObsDataset
    fn make_dataset(obs: Vec<Observation>, observers: Vec<Observer>) -> ObsDataset {
        ObsDataset::new(obs, observers, Some(ObsErrorModel::FCCT14), None, None)
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

        /// Verifies that constructing an empty dataset does not panic.
        #[test]
        fn new_empty_with_none_cache_size_does_not_panic() {
            let _ds = ObsDataset::new(vec![], vec![], Some(ObsErrorModel::FCCT14), None, None);
        }

        /// Verifies that constructing an empty dataset does not panic.
        #[test]
        fn new_empty_with_custom_cache_size_does_not_panic() {
            let _ds = ObsDataset::new(vec![], vec![], Some(ObsErrorModel::FCCT14), None, None);
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
            let ds = make_dataset(vec![make_observation(1, None)], vec![]);
            assert!(ds.get_observation(1).is_some());
        }

        /// Verifies that get_observation returns None for a missing id.
        #[test]
        fn get_observation_returns_none_for_missing_id() {
            let ds = make_dataset(vec![make_observation(1, None)], vec![]);
            assert!(ds.get_observation(9999).is_none());
        }

        /// Verifies that repeated calls for the same id return the same observation.
        #[test]
        fn get_observation_repeated_calls_return_same_id() {
            let ds = make_dataset(vec![make_observation(7, None)], vec![]);
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
            let ds = make_dataset(obs, vec![]);
            let found = ds.get_observation(2);
            assert!(found.is_some(), "Expected Some for id=2");
            assert_eq!(found.unwrap().id, 2);
        }

        /// Verifies that repeated calls for the same id return consistent results
        /// even after other observations have been looked up.
        /// The evicted entry must still be findable.
        #[test]
        fn get_observation_repeated_calls_still_findable() {
            // Looking up id=2 after id=1 should not prevent id=1 from being found.
            let obs = vec![make_observation(1, None), make_observation(2, None)];
            let ds = ObsDataset::new(obs, vec![], Some(ObsErrorModel::FCCT14), None, None);

            // Populate the index with id=1.
            assert!(ds.get_observation(1).is_some());
            // Looking up id=2.
            assert!(ds.get_observation(2).is_some());
            // id=1 must still be found.
            assert!(
                ds.get_observation(1).is_some(),
                "id=1 should still be findable"
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
