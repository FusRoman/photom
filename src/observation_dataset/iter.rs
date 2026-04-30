//! Sequential iterators over [`ObsDataset`] observations, nights, and trajectories.
//!
//! This module extends [`ObsDataset`] with iterator methods for all three
//! grouping axes: the flat observation list, individual nights, and individual
//! trajectories.  The parallel equivalents of these methods live in
//! [`super::parallel`] and are only compiled when the `parallel` feature is
//! enabled.
//!
//! ## Key types
//!
//! - [`MemLayoutObservations`] — a collection of observation references that is
//!   either a contiguous borrowed slice or a non-contiguous `Vec` of references,
//!   depending on how the observations are stored in the index.
//!
//! [`ObsDataset`]: crate::observation_dataset::ObsDataset
use itertools::Either;

use crate::{
    NightId, TrajId,
    observation_dataset::{
        ObsDataset, ObsDatasetError, index::ObsMapIndex, observation::Observation,
    },
    observer::{Observer, dataset::ObserverId},
};

// Observation iterator implementation for ObsDataset.
impl ObsDataset {
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

    /// Return an iterator over all observers in the dataset, including both custom geodetic observers and MPC-coded observers.
    /// The order of the yielded observers is unspecified.
    ///
    /// # Returns
    ///
    /// `Ok(iterator)` where `iterator` yields `(ObserverId, &Observer)` pairs for each observer in the dataset, if the MPC observatory catalogue was successfully loaded;
    /// `Err(ObsDatasetError::MpcCatalogueLoadError)` if the MPC observatory catalogue could not be loaded, which prevents access to MPC-coded observers.
    pub fn iter_observer(
        &self,
    ) -> Result<impl Iterator<Item = (ObserverId, &Observer)>, ObsDatasetError> {
        // MPC-coded observer iterator
        let mpc_iter = self
            .observer_dataset
            .mpc_observers()?
            .iter()
            .map(|(code, obs)| (ObserverId::MpcCode(*code), obs));

        let custom_observer_iter = self
            .observer_dataset
            .custom_observers
            .iter()
            .enumerate()
            .map(|(idx, obs)| (ObserverId::IntId(idx), obs));

        Ok(mpc_iter.chain(custom_observer_iter))
    }
}

/// A borrowed collection of observation references that may be either
/// contiguous in memory or scattered across multiple non-adjacent positions.
///
/// This type is returned by [`ObsDataset::materialize_night`] and
/// [`ObsDataset::materialize_trajectory`] to avoid unnecessary heap
/// allocation when the observations for a night or trajectory were stored
/// as a contiguous block in the internal `Vec`.
///
/// The two variants correspond to the two storage strategies used by the
/// internal observation index:
///
/// - [`Contiguous`](MemLayoutObservations::Contiguous) — a borrowed slice of
///   observations that lie in a consecutive block.
/// - [`Split`](MemLayoutObservations::Split) — a `Vec` of borrowed references
///   collected from non-adjacent positions.
pub enum MemLayoutObservations<'a> {
    /// Observations occupy a single contiguous block in the parent vector.
    Contiguous(&'a [Observation]),
    /// Observations are scattered at non-adjacent positions and have been
    /// collected into a `Vec` of shared references.
    Split(Vec<&'a Observation>),
}

impl<'a> MemLayoutObservations<'a> {
    /// Return the number of observations in this collection.
    ///
    /// # Returns
    ///
    /// The count of [`Observation`] references held by this collection.
    pub fn len(&self) -> usize {
        match self {
            MemLayoutObservations::Contiguous(slice) => slice.len(),
            MemLayoutObservations::Split(vec) => vec.len(),
        }
    }

    /// Return `true` if this collection contains no observations.
    ///
    /// # Returns
    ///
    /// `true` when [`len`](Self::len) is zero; `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a borrowing iterator over the contained observation references.
    ///
    /// Unlike [`into_iter`](IntoIterator::into_iter) this method does **not**
    /// consume `self`, so the collection can be inspected multiple times.
    ///
    /// # Returns
    ///
    /// An iterator yielding `&Observation` for each observation in this collection.
    pub fn iter(&self) -> impl Iterator<Item = &'a Observation> + '_ {
        match self {
            MemLayoutObservations::Contiguous(slice) => Either::Left(slice.iter()),
            MemLayoutObservations::Split(vec) => Either::Right(vec.iter().copied()),
        }
    }

    /// Eagerly collect the contained observation references into a `Vec`.
    /// This is a convenience method that simply collects the results of [`MemLayoutObservations::iter`] into a `Vec`.
    ///
    /// # Returns
    ///
    /// A `Vec` containing shared references to all observations in this collection,
    /// in the same order as [`MemLayoutObservations::iter`] would yield them.
    pub fn collect_into_vec(&self) -> Vec<&'a Observation> {
        self.iter().collect()
    }
}

impl<'a> IntoIterator for MemLayoutObservations<'a> {
    type Item = &'a Observation;
    type IntoIter = Either<std::slice::Iter<'a, Observation>, std::vec::IntoIter<&'a Observation>>;

    /// Consume this collection and return a sequential iterator over the
    /// contained observation references.
    ///
    /// The iterator is backed by [`itertools::Either`] so that both the
    /// `Contiguous` (slice iterator) and `Split` (vec iterator) branches share
    /// a single concrete return type with no virtual dispatch.
    fn into_iter(self) -> Self::IntoIter {
        match self {
            MemLayoutObservations::Contiguous(slice) => Either::Left(slice.iter()),
            MemLayoutObservations::Split(vec) => Either::Right(vec.into_iter()),
        }
    }
}

impl ObsDataset {
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
    /// `Some(MemLayoutObservations)` in insertion order if the night index exists and the
    /// given `night_id` is present; `None` otherwise.
    pub fn materialize_night(&self, night_id: &NightId) -> Option<MemLayoutObservations<'_>> {
        let night_index = self.index.obs_index_by_night.as_ref()?.get(night_id)?;
        match night_index {
            ObsMapIndex::Split(indices) => Some(MemLayoutObservations::Split(
                indices.iter().map(|idx| &self.observations[*idx]).collect(),
            )),
            ObsMapIndex::Contiguous { start, end } => Some(MemLayoutObservations::Contiguous(
                &self.observations[*start..*end],
            )),
        }
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

    /// Return the total number of nights present in the dataset, as determined by the night index.
    ///
    /// # Returns
    ///
    /// `Some(count)` if the night index exists; `None` otherwise.
    pub fn nb_night(&self) -> Option<usize> {
        self.iter_night_id().map(|iter| iter.count())
    }

    /// Return whether the index entry for a given night is stored as a
    /// contiguous block (`true`) or as a scattered list (`false`).
    ///
    /// A contiguous entry means all observations for that night occupy a
    /// single uninterrupted span in the internal observations vector, which
    /// allows cheaper slicing operations.  The representation is an
    /// implementation detail of the ingestion and merge logic; callers
    /// should not rely on it for correctness, but it is useful for
    /// verifying that index preservation guarantees hold in tests.
    ///
    /// # Returns
    ///
    /// `Some(true)` if the night index exists, the night is present, and its
    /// entry is `Contiguous`; `Some(false)` if the entry is
    /// `Split`; `None` if the night index is absent or the
    /// given `night_id` is not found.
    pub fn is_night_contiguous(&self, night_id: &NightId) -> Option<bool> {
        let entry = self.index.obs_index_by_night.as_ref()?.get(night_id)?;
        Some(matches!(entry, ObsMapIndex::Contiguous { .. }))
    }
}

// Trajectory iterator implementation for ObsDataset.
impl ObsDataset {
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
        traj_id: impl Into<TrajId>,
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
    /// `Some(MemLayoutObservations)` in insertion order if the trajectory index exists and the
    /// given `traj_id` is present; `None` otherwise.
    pub fn materialize_trajectory(
        &self,
        traj_id: impl Into<TrajId>,
    ) -> Option<MemLayoutObservations<'_>> {
        let traj_id = traj_id.into();
        let traj_index = self.index.obs_index_by_trajectory.as_ref()?.get(&traj_id)?;
        match traj_index {
            ObsMapIndex::Split(indices) => Some(MemLayoutObservations::Split(
                indices.iter().map(|idx| &self.observations[*idx]).collect(),
            )),
            ObsMapIndex::Contiguous { start, end } => Some(MemLayoutObservations::Contiguous(
                &self.observations[*start..*end],
            )),
        }
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
    pub fn len_trajectory(&self, traj_id: impl Into<TrajId>) -> Option<usize> {
        self.index.len_trajectory(traj_id)
    }

    /// Return whether the index entry for a given trajectory is stored as a
    /// contiguous block (`true`) or as a scattered list (`false`).
    ///
    /// Same semantics as [`ObsDataset::is_night_contiguous`] but for the
    /// trajectory index.
    ///
    /// # Returns
    ///
    /// `Some(true)` if the trajectory index exists, the trajectory is present,
    /// and its entry is `Contiguous`; `Some(false)` if the
    /// entry is `Split`; `None` if the trajectory index is
    /// absent or the given `traj_id` is not found.
    pub fn is_traj_contiguous(&self, traj_id: impl Into<TrajId>) -> Option<bool> {
        let traj_id = traj_id.into();
        let entry = self.index.obs_index_by_trajectory.as_ref()?.get(&traj_id)?;
        Some(matches!(entry, ObsMapIndex::Contiguous { .. }))
    }
}

#[cfg(test)]
mod iter_tests {
    use ahash::AHashMap;

    use crate::{
        NightId, TrajId,
        coordinates::equatorial::EquCoord,
        observation_dataset::{
            ObsDataset,
            index::{NightIndexMap, ObsMapIndex, TrajIndexMap},
            observation::ObservationInput,
        },
        observer::error_model::ObsErrorModel,
        photometry::{Filter, Photometry},
    };

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_obs(id: u64, _index: usize) -> ObservationInput {
        ObservationInput {
            id,
            equ_coord: EquCoord::new(0.5, 1e-5, 0.2, 1e-5),
            photometry: Photometry {
                magnitude: 15.0,
                error: 0.1,
                filter: Filter::String("G".to_string()),
            },
            mjd_tt: 60000.0 + id as f64,
            observer: None,
        }
    }

    /// Dataset with 4 observations, night index (2×2) and trajectory index (2×2).
    ///
    /// Night 1 → positions [0, 1]; Night 2 → positions [2, 3] (Contiguous)
    /// Traj 10 → positions [0, 2]; Traj 20 → positions [1, 3] (Split)
    fn make_dataset_with_index() -> ObsDataset {
        let obs = vec![
            make_obs(1, 0),
            make_obs(2, 1),
            make_obs(3, 2),
            make_obs(4, 3),
        ];

        let mut night_map: NightIndexMap = AHashMap::new();
        night_map.insert(NightId(1), ObsMapIndex::Contiguous { start: 0, end: 2 });
        night_map.insert(NightId(2), ObsMapIndex::Contiguous { start: 2, end: 4 });

        let mut traj_map: TrajIndexMap = AHashMap::new();
        traj_map.insert(TrajId::Int(10), ObsMapIndex::Split(vec![0, 2]));
        traj_map.insert(TrajId::Int(20), ObsMapIndex::Split(vec![1, 3]));

        ObsDataset::new(
            obs,
            vec![],
            Some(ObsErrorModel::FCCT14),
            Some(night_map),
            Some(traj_map),
        )
    }

    fn make_dataset_no_index() -> ObsDataset {
        let obs = vec![make_obs(1, 0), make_obs(2, 1)];
        ObsDataset::new(obs, vec![], Some(ObsErrorModel::FCCT14), None, None)
    }

    // ------------------------------------------------------------------
    // iter_observations
    // ------------------------------------------------------------------

    #[test]
    fn iter_observations_count() {
        let ds = make_dataset_with_index();
        assert_eq!(ds.iter_observations().count(), 4);
    }

    #[test]
    fn iter_observations_ids_in_order() {
        let ds = make_dataset_with_index();
        let ids: Vec<u64> = ds.iter_observations().map(|o| *o.id()).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    // ------------------------------------------------------------------
    // MemLayoutObservations
    // ------------------------------------------------------------------

    #[test]
    fn mem_layout_contiguous_len_and_iter() {
        let ds = make_dataset_with_index();
        let night = ds.materialize_night(&NightId(1)).unwrap();
        assert_eq!(night.len(), 2);
        assert!(!night.is_empty());
        let ids: Vec<u64> = night.iter().map(|o| *o.id()).collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn mem_layout_split_len_and_iter() {
        let ds = make_dataset_with_index();
        let traj = ds.materialize_trajectory(TrajId::Int(10)).unwrap();
        assert_eq!(traj.len(), 2);
        let ids: Vec<u64> = traj.iter().map(|o| *o.id()).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn mem_layout_is_empty_false_for_non_empty() {
        let ds = make_dataset_with_index();
        assert!(!ds.materialize_night(&NightId(1)).unwrap().is_empty());
    }

    #[test]
    fn mem_layout_into_iter_contiguous() {
        let ds = make_dataset_with_index();
        let night = ds.materialize_night(&NightId(2)).unwrap();
        let ids: Vec<u64> = night.into_iter().map(|o| *o.id()).collect();
        assert_eq!(ids, vec![3, 4]);
    }

    #[test]
    fn mem_layout_into_iter_split() {
        let ds = make_dataset_with_index();
        let traj = ds.materialize_trajectory(TrajId::Int(20)).unwrap();
        let ids: Vec<u64> = traj.into_iter().map(|o| *o.id()).collect();
        assert_eq!(ids, vec![2, 4]);
    }

    // ------------------------------------------------------------------
    // Night iterators
    // ------------------------------------------------------------------

    #[test]
    fn iter_night_observations_some_for_existing() {
        let ds = make_dataset_with_index();
        assert!(ds.iter_night_observations(&NightId(1)).is_some());
    }

    #[test]
    fn iter_night_observations_none_for_missing() {
        let ds = make_dataset_with_index();
        assert!(ds.iter_night_observations(&NightId(99)).is_none());
    }

    #[test]
    fn iter_night_observations_none_without_index() {
        let ds = make_dataset_no_index();
        assert!(ds.iter_night_observations(&NightId(1)).is_none());
    }

    #[test]
    fn iter_night_observations_count() {
        let ds = make_dataset_with_index();
        let count = ds.iter_night_observations(&NightId(1)).unwrap().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn iter_full_night_some_with_index() {
        let ds = make_dataset_with_index();
        assert!(ds.iter_full_night().is_some());
    }

    #[test]
    fn iter_full_night_none_without_index() {
        let ds = make_dataset_no_index();
        assert!(ds.iter_full_night().is_none());
    }

    #[test]
    fn iter_full_night_total_count() {
        let ds = make_dataset_with_index();
        assert_eq!(ds.iter_full_night().unwrap().count(), 4);
    }

    #[test]
    fn iter_night_id_count() {
        let ds = make_dataset_with_index();
        assert_eq!(ds.iter_night_id().unwrap().count(), 2);
    }

    #[test]
    fn len_night_correct() {
        let ds = make_dataset_with_index();
        assert_eq!(ds.len_night(&NightId(1)), Some(2));
        assert_eq!(ds.len_night(&NightId(99)), None);
    }

    // ------------------------------------------------------------------
    // Trajectory iterators
    // ------------------------------------------------------------------

    #[test]
    fn iter_trajectory_observations_some_for_existing() {
        let ds = make_dataset_with_index();
        assert!(ds.iter_trajectory_observations(&TrajId::Int(10)).is_some());
    }

    #[test]
    fn iter_trajectory_observations_none_for_missing() {
        let ds = make_dataset_with_index();
        assert!(ds.iter_trajectory_observations(&TrajId::Int(99)).is_none());
    }

    #[test]
    fn iter_trajectory_observations_none_without_index() {
        let ds = make_dataset_no_index();
        assert!(ds.iter_trajectory_observations(&TrajId::Int(10)).is_none());
    }

    #[test]
    fn iter_trajectory_observations_count() {
        let ds = make_dataset_with_index();
        let count = ds
            .iter_trajectory_observations(&TrajId::Int(10))
            .unwrap()
            .count();
        assert_eq!(count, 2);
    }

    #[test]
    fn iter_full_trajectory_some_with_index() {
        let ds = make_dataset_with_index();
        assert!(ds.iter_full_trajectory().is_some());
    }

    #[test]
    fn iter_full_trajectory_none_without_index() {
        let ds = make_dataset_no_index();
        assert!(ds.iter_full_trajectory().is_none());
    }

    #[test]
    fn iter_full_trajectory_total_count() {
        let ds = make_dataset_with_index();
        assert_eq!(ds.iter_full_trajectory().unwrap().count(), 4);
    }

    #[test]
    fn iter_traj_id_count() {
        let ds = make_dataset_with_index();
        assert_eq!(ds.iter_traj_id().unwrap().count(), 2);
    }

    #[test]
    fn len_trajectory_correct() {
        let ds = make_dataset_with_index();
        assert_eq!(ds.len_trajectory(TrajId::Int(10)), Some(2));
        assert_eq!(ds.len_trajectory(TrajId::Int(99)), None);
    }
}
