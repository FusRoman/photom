#![cfg(feature = "parallel")]
//! Parallel iterators over [`ObsDataset`] and its internal index, powered by
//! [rayon](https://docs.rs/rayon).
//!
//! This module is controlled by the `parallel` feature flag.  It is compiled
//! only when `parallel` is enabled in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! photom = { version = "0.1", features = ["parallel"] }
//! ```
//!
//! Every method in this module is the parallel counterpart of a sequential
//! iterator defined on [`ObsDataset`] in `observation_dataset/mod.rs`.  The
//! function names, arguments, and return types mirror those sequential
//! variants exactly, differing only in that they return a
//! [`rayon::iter::ParallelIterator`] instead of a standard
//! [`std::iter::Iterator`].
//!
//! ## Public methods
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`ObsDataset::par_iter_observations`] | Parallel iterator over all observations in insertion order |
//! | [`ObsDataset::par_iter_full_night`] | Parallel iterator over `(NightId, &Observation)` pairs for every indexed night |
//! | [`ObsDataset::par_iter_night_observations`] | Parallel iterator over observations for a single night |
//! | [`ObsDataset::materialize_night_par`] | Collect observations for a single night into a `Vec` using parallel iteration |
//! | [`ObsDataset::par_iter_trajectory_observations`] | Parallel iterator over observations for a single trajectory |
//! | [`ObsDataset::par_iter_full_trajectory`] | Parallel iterator over `(TrajId, &Observation)` pairs for every indexed trajectory |
//! | [`ObsDataset::materialize_trajectory_par`] | Collect observations for a single trajectory into a `Vec` using parallel iteration |
//!
//! ## Ordering guarantees
//!
//! - **Across nights / trajectories**: the order is unspecified because the
//!   underlying index is an `AHashMap` with non-deterministic key order.
//! - **Within a single night or trajectory**: observations appear in
//!   *insertion order* (the order of the source `DataFrame` rows).
//! - **Materialised `Vec`s**: the order of elements collected by
//!   [`ObsDataset::materialize_night_par`] and
//!   [`ObsDataset::materialize_trajectory_par`] is **not** guaranteed,
//!   because parallel collection does not preserve iterator order.
//!
//! [`ObsDataset`]: crate::observation_dataset::ObsDataset

use itertools::Either;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    NightId, TrajId,
    observation_dataset::{
        ObsDataset,
        index::{ObsDatasetIndex, ObsIndex, ObsMapIndex},
        iter::MemLayoutObservations,
        observation::Observation,
    },
};

impl ObsDatasetIndex {
    /// Return a parallel iterator over the vector positions of all observations on a given night.
    ///
    /// This is the parallel counterpart of [`ObsDatasetIndex::iter_night_obs_index`](crate::observation_dataset::index::ObsDatasetIndex).
    ///
    /// # Arguments
    ///
    /// - `night_id` — the night identifier whose observation positions are requested.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` yielding each `ObsIndex` in insertion order if the night index
    /// exists and the night is present; `None` otherwise.
    pub(crate) fn par_iter_night_obs_index(
        &self,
        night_id: &NightId,
    ) -> Option<impl ParallelIterator<Item = ObsIndex> + '_> {
        self.get_by_night(night_id).map(|indices| match indices {
            ObsMapIndex::Contiguous { start, end } => Either::Left((*start..*end).into_par_iter()),
            ObsMapIndex::Split(vec) => Either::Right(vec.par_iter().copied()),
        })
    }

    /// Return a parallel iterator over `(NightId, ObsIndex)` pairs for every observation
    /// in the night index.
    ///
    /// Each pair associates a night identifier with the vector position of one of the
    /// observations recorded on that night.  The order between nights is unspecified
    /// (hash map key order).
    ///
    /// This is the parallel counterpart of [`ObsDatasetIndex::iter_full_night`](crate::observation_dataset::index::ObsDatasetIndex).
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if a night index was built; `None` otherwise.
    pub(crate) fn par_iter_full_night(
        &self,
    ) -> Option<impl ParallelIterator<Item = (NightId, ObsIndex)> + '_> {
        self.obs_index_by_night.as_ref().map(|night_map| {
            night_map
                .par_iter()
                .flat_map(|(night_id, indices)| match indices {
                    ObsMapIndex::Contiguous { start, end } => Either::Left(
                        (*start..*end)
                            .into_par_iter()
                            .map(move |idx| (*night_id, idx)),
                    ),
                    ObsMapIndex::Split(vec) => {
                        Either::Right(vec.par_iter().map(move |&idx| (*night_id, idx)))
                    }
                })
        })
    }

    /// Return a parallel iterator over the vector positions of all observations in a given
    /// trajectory.
    ///
    /// This is the parallel counterpart of [`ObsDatasetIndex::iter_traj_obs_index`](crate::observation_dataset::index::ObsDatasetIndex).
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the trajectory identifier whose observation positions are requested.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` yielding each `ObsIndex` in insertion order if the trajectory index
    /// exists and the trajectory is present; `None` otherwise.
    pub(crate) fn par_iter_traj_obs_index(
        &self,
        traj_id: &TrajId,
    ) -> Option<impl ParallelIterator<Item = ObsIndex> + '_> {
        self.get_by_trajectory(traj_id)
            .map(|indices| match indices {
                ObsMapIndex::Contiguous { start, end } => {
                    Either::Left((*start..*end).into_par_iter())
                }
                ObsMapIndex::Split(vec) => Either::Right(vec.par_iter().copied()),
            })
    }

    /// Return a parallel iterator over `(TrajId, ObsIndex)` pairs for every observation in
    /// the trajectory index.
    ///
    /// Each pair associates a trajectory identifier with the vector position of one of the
    /// observations belonging to that trajectory.  The order between trajectories is
    /// unspecified (hash map key order).
    ///
    /// This is the parallel counterpart of [`ObsDatasetIndex::iter_full_trajectory`](crate::observation_dataset::index::ObsDatasetIndex).
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if a trajectory index was built; `None` otherwise.
    pub(crate) fn par_iter_full_trajectory(
        &self,
    ) -> Option<impl ParallelIterator<Item = (TrajId, ObsIndex)> + '_> {
        self.obs_index_by_trajectory.as_ref().map(|traj_map| {
            traj_map
                .par_iter()
                .flat_map(|(traj_id, indices)| match indices {
                    ObsMapIndex::Contiguous { start, end } => Either::Left(
                        (*start..*end)
                            .into_par_iter()
                            .map(move |idx| (traj_id.clone(), idx)),
                    ),
                    ObsMapIndex::Split(vec) => {
                        Either::Right(vec.par_iter().map(move |&idx| (traj_id.clone(), idx)))
                    }
                })
        })
    }
}

impl ObsDataset {
    /// Return a parallel iterator over all observations in insertion order.
    ///
    /// The iterator yields shared references and does not clone any data.
    /// The order matches the order of the source `DataFrame` rows.
    ///
    /// Unlike [`ObsDataset::get_observation`], this method takes `&self` and
    /// can therefore be called while other shared borrows of the dataset are live.
    ///
    /// This is the parallel counterpart of [`ObsDataset::iter_observations`].
    ///
    /// # Returns
    ///
    /// An iterator yielding `&Observation` for each observation in insertion order.
    pub fn par_iter_observations(&self) -> impl ParallelIterator<Item = &Observation> {
        self.observations.par_iter()
    }

    /// Return a parallel iterator over `(NightId, &Observation)` pairs for every observation
    /// in the night index.
    ///
    /// Each pair associates a night identifier with a shared reference to one of the
    /// observations recorded on that night.  The order between different nights is
    /// unspecified (hash map key order).
    ///
    /// This is the parallel counterpart of [`ObsDataset::iter_full_night`].
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset was built with a night index (`night_id` column
    /// present in the source data); `None` otherwise.
    pub fn par_iter_full_night(
        &self,
    ) -> Option<impl ParallelIterator<Item = (NightId, &Observation)>> {
        self.index
            .par_iter_full_night()
            .map(|night_iter| night_iter.map(|(night_id, idx)| (night_id, &self.observations[idx])))
    }

    /// Return a parallel iterator over all observations belonging to a given night, in
    /// insertion order.
    ///
    /// The order of the yielded observations matches the order of the source `DataFrame`
    /// rows.
    ///
    /// This is the parallel counterpart of [`ObsDataset::iter_night_observations`].
    ///
    /// # Arguments
    ///
    /// - `night_id` — the identifier of the night for which to return observations.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset has a night index and the given `night_id` is found,
    /// where `iterator` yields shared references to the observations belonging to that night
    /// in insertion order; `None` otherwise.
    pub fn par_iter_night_observations(
        &self,
        night_id: &NightId,
    ) -> Option<impl ParallelIterator<Item = &Observation> + '_> {
        self.index
            .par_iter_night_obs_index(night_id)
            .map(|indices| indices.map(|idx| &self.observations[idx]))
    }

    /// Collect all observations belonging to a given night into a `Vec` using parallel
    /// iteration.
    ///
    /// This is a convenience wrapper around [`ObsDataset::par_iter_night_observations`] that
    /// eagerly collects the parallel iterator into a `Vec`.  Because collection is parallel,
    /// the order of elements in the returned `Vec` is **not** guaranteed.
    ///
    /// This is the parallel counterpart of [`ObsDataset::materialize_night`].
    ///
    /// # Arguments
    ///
    /// - `night_id` — the identifier of the night to materialise.
    ///
    /// # Returns
    ///
    /// `Some(Vec<&Observation>)` if the night index exists and the given `night_id` is
    /// present; `None` otherwise.  The order of elements in the `Vec` is unspecified.
    pub fn materialize_night_par(&self, night_id: &NightId) -> Option<MemLayoutObservations<'_>> {
        let night_index = self.index.obs_index_by_night.as_ref()?.get(night_id)?;
        match night_index {
            ObsMapIndex::Split(indices) => Some(MemLayoutObservations::Split(
                indices
                    .par_iter()
                    .map(|idx| &self.observations[*idx])
                    .collect(),
            )),
            ObsMapIndex::Contiguous { start, end } => Some(MemLayoutObservations::Contiguous(
                &self.observations[*start..*end],
            )),
        }
    }

    /// Return a parallel iterator over all observations belonging to a given trajectory, in
    /// insertion order.
    ///
    /// The order of the yielded observations matches the order of the source `DataFrame`
    /// rows.
    ///
    /// This is the parallel counterpart of [`ObsDataset::iter_trajectory_observations`].
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the identifier of the trajectory for which to return observations.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset has a trajectory index and the given `traj_id` is
    /// found, where `iterator` yields shared references to the observations belonging to
    /// that trajectory in insertion order; `None` otherwise.
    pub fn par_iter_trajectory_observations(
        &self,
        traj_id: &TrajId,
    ) -> Option<impl ParallelIterator<Item = &Observation>> {
        self.index
            .par_iter_traj_obs_index(traj_id)
            .map(|indices| indices.map(|idx| &self.observations[idx]))
    }

    /// Return a parallel iterator over `(TrajId, &Observation)` pairs for every observation
    /// in the trajectory index.
    ///
    /// Each pair associates a trajectory identifier with a shared reference to one of the
    /// observations belonging to that trajectory.  The order between different trajectories
    /// is unspecified (hash map key order).
    ///
    /// This is the parallel counterpart of [`ObsDataset::iter_full_trajectory`].
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if the dataset was built with a trajectory index (`traj_id` column
    /// present in the source data); `None` otherwise.
    pub fn par_iter_full_trajectory(
        &self,
    ) -> Option<impl ParallelIterator<Item = (TrajId, &Observation)>> {
        self.index.par_iter_full_trajectory().map(|traj_iter| {
            traj_iter.map(|(traj_id, idx)| (traj_id.clone(), &self.observations[idx]))
        })
    }

    /// Collect all observations belonging to a given trajectory into a `Vec` using parallel
    /// iteration.
    ///
    /// This is a convenience wrapper around [`ObsDataset::par_iter_trajectory_observations`]
    /// that eagerly collects the parallel iterator into a `Vec`.  Because collection is
    /// parallel, the order of elements in the returned `Vec` is **not** guaranteed.
    ///
    /// This is the parallel counterpart of [`ObsDataset::materialize_trajectory`].
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the identifier of the trajectory to materialise.
    ///
    /// # Returns
    ///
    /// `Some(Vec<&Observation>)` if the trajectory index exists and the given `traj_id` is
    /// present; `None` otherwise.  The order of elements in the `Vec` is unspecified.
    pub fn materialize_trajectory_par(
        &self,
        traj_id: &TrajId,
    ) -> Option<MemLayoutObservations<'_>> {
        let traj_index = self.index.obs_index_by_trajectory.as_ref()?.get(traj_id)?;
        match traj_index {
            ObsMapIndex::Split(indices) => Some(MemLayoutObservations::Split(
                indices
                    .par_iter()
                    .map(|idx| &self.observations[*idx])
                    .collect(),
            )),
            ObsMapIndex::Contiguous { start, end } => Some(MemLayoutObservations::Contiguous(
                &self.observations[*start..*end],
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod obsdataset_parallel_tests {
    use super::*;
    use ahash::AHashMap;
    use rayon::iter::ParallelIterator;

    use crate::{
        NightId, TrajId,
        astrometry::EquCoord,
        observation_dataset::{
            index::{NightIndexMap, TrajIndexMap},
            observation::Observation,
        },
        observer::error_model::ObsErrorModel,
        photometry::{Filter, Photometry},
    };

    // -----------------------------------------------------------------------
    // Test data helpers
    // -----------------------------------------------------------------------

    /// Build a minimal [`Observation`] with a given id and its position in the Vec.
    ///
    /// No observer is attached; all coordinate/photometry values are fixed constants
    /// so tests remain purely structural and do not depend on astrometric values.
    fn make_obs(id: u64, index: usize) -> Observation {
        Observation {
            index,
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

    /// Build a dataset with 4 observations, a night index (2 nights × 2 obs) and a
    /// trajectory index (2 trajectories × 2 obs).
    ///
    /// Layout:
    /// - Night 1  → obs at positions 0, 1  (ids 1, 2)
    /// - Night 2  → obs at positions 2, 3  (ids 3, 4)
    /// - Traj 10  → obs at positions 0, 2  (ids 1, 3)
    /// - Traj 20  → obs at positions 1, 3  (ids 2, 4)
    fn make_dataset_with_index() -> ObsDataset {
        let obs = vec![
            make_obs(1, 0),
            make_obs(2, 1),
            make_obs(3, 2),
            make_obs(4, 3),
        ];

        let mut night_map: NightIndexMap = AHashMap::new();
        night_map.insert(NightId(1), ObsMapIndex::Split(vec![0, 1]));
        night_map.insert(NightId(2), ObsMapIndex::Split(vec![2, 3]));

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

    /// Build a dataset with 2 observations and **no** night or trajectory index.
    fn make_dataset_no_index() -> ObsDataset {
        let obs = vec![make_obs(1, 0), make_obs(2, 1)];
        ObsDataset::new(obs, vec![], Some(ObsErrorModel::FCCT14), None, None)
    }

    // -----------------------------------------------------------------------
    // Helper: extract the inner u32 from TrajId::Int for sorting.
    // Panics on TrajId::Str — only used in tests that exclusively use Int ids.
    // -----------------------------------------------------------------------
    /// Extract the inner `u32` from a [`TrajId::Int`] for use as a sort key.
    ///
    /// # Panics
    ///
    /// Panics if `id` is `TrajId::Str` — this helper is only used in tests
    /// that exclusively build `Int` trajectory identifiers.
    fn traj_id_key(id: &TrajId) -> u32 {
        match id {
            TrajId::Int(n) => *n,
            TrajId::Str(_) => panic!("unexpected TrajId::Str in sort key"),
        }
    }

    // =======================================================================
    // mod parallel_obs_iter — par_iter_observations (tests 1–3)
    // =======================================================================

    mod parallel_obs_iter {
        use super::*;

        /// Test 1 — parallel observation iterator over a 4-element dataset yields 4 items.
        #[test]
        fn par_iter_observations_count() {
            let dataset = make_dataset_with_index();
            assert_eq!(dataset.par_iter_observations().count(), 4);
        }

        /// Test 2 — collect observation ids from the parallel iterator; sorted result
        /// must equal [1, 2, 3, 4] regardless of thread scheduling order.
        #[test]
        fn par_iter_observations_ids() {
            let dataset = make_dataset_with_index();
            let mut ids: Vec<u64> = dataset.par_iter_observations().map(|o| o.id).collect();
            ids.sort_unstable();
            assert_eq!(ids, vec![1u64, 2, 3, 4]);
        }

        /// Test 3 — parallel observation iterator works even when the dataset was built
        /// without a night or trajectory index (no index is required for this method).
        #[test]
        fn par_iter_observations_no_index_count() {
            let dataset = make_dataset_no_index();
            assert_eq!(dataset.par_iter_observations().count(), 2);
        }
    }

    // =======================================================================
    // mod parallel_night — night-related parallel methods (tests 4–16)
    // =======================================================================

    mod parallel_night {
        use super::*;

        // -------------------------------------------------------------------
        // par_iter_full_night on ObsDataset
        // -------------------------------------------------------------------

        /// Test 4 — par_iter_full_night returns Some when the dataset was built with a
        /// night index.
        #[test]
        fn par_iter_full_night_some_when_night_index_present() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset.par_iter_full_night().is_some(),
                "Expected Some when night index is present"
            );
        }

        /// Test 5 — par_iter_full_night returns None when the dataset has no night index.
        #[test]
        fn par_iter_full_night_none_when_no_night_index() {
            let dataset = make_dataset_no_index();
            assert!(
                dataset.par_iter_full_night().is_none(),
                "Expected None when no night index"
            );
        }

        /// Test 6 — the full-night parallel iterator yields exactly 4 (NightId, &Observation)
        /// pairs across 2 nights of 2 observations each.
        #[test]
        fn par_iter_full_night_total_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 4 confirmed Some
            let count = dataset.par_iter_full_night().unwrap().count();
            assert_eq!(count, 4);
        }

        /// Test 7 — collect the NightId components, sort and dedup; the unique night ids
        /// must be exactly [NightId(1), NightId(2)].
        #[test]
        fn par_iter_full_night_night_ids() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 4 confirmed Some
            let mut night_ids: Vec<NightId> = dataset
                .par_iter_full_night()
                .unwrap()
                .map(|(nid, _)| nid)
                .collect();
            night_ids.sort_unstable();
            night_ids.dedup();
            assert_eq!(night_ids, vec![NightId(1), NightId(2)]);
        }

        // -------------------------------------------------------------------
        // par_iter_night_observations on ObsDataset
        // -------------------------------------------------------------------

        /// Test 8 — par_iter_night_observations returns Some for a night that exists in
        /// the index.
        #[test]
        fn par_iter_night_observations_some_for_existing_night() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset.par_iter_night_observations(&NightId(1)).is_some(),
                "Expected Some for NightId(1) which is present in the index"
            );
        }

        /// Test 9 — par_iter_night_observations returns None for a night that is not in
        /// the index.
        #[test]
        fn par_iter_night_observations_none_for_missing_night() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset.par_iter_night_observations(&NightId(99)).is_none(),
                "Expected None for NightId(99) which is absent from the index"
            );
        }

        /// Test 10 — par_iter_night_observations returns None when the dataset has no night
        /// index at all (None was passed for obs_index_by_night).
        #[test]
        fn par_iter_night_observations_none_without_index() {
            let dataset = make_dataset_no_index();
            assert!(
                dataset.par_iter_night_observations(&NightId(1)).is_none(),
                "Expected None when the dataset has no night index"
            );
        }

        /// Test 11 — the per-night parallel iterator for NightId(1) yields exactly 2 items.
        #[test]
        fn par_iter_night_observations_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 8 confirmed Some
            let count = dataset
                .par_iter_night_observations(&NightId(1))
                .unwrap()
                .count();
            assert_eq!(count, 2);
        }

        /// Test 12 — collect ids from night 1's parallel iterator, sort, and assert they
        /// equal [1, 2] (the two observations assigned to that night).
        #[test]
        fn par_iter_night_observations_ids() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 8 confirmed Some
            let mut ids: Vec<u64> = dataset
                .par_iter_night_observations(&NightId(1))
                .unwrap()
                .map(|o| o.id)
                .collect();
            ids.sort_unstable();
            assert_eq!(ids, vec![1u64, 2]);
        }

        // -------------------------------------------------------------------
        // materialize_night_par on ObsDataset
        // -------------------------------------------------------------------

        /// Test 13 — materialize_night_par returns Some(vec) for a night that exists in
        /// the index.
        #[test]
        fn materialize_night_par_some_for_existing_night() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset.materialize_night_par(&NightId(1)).is_some(),
                "Expected Some(vec) for NightId(1)"
            );
        }

        /// Test 14 — materialize_night_par returns None for a night that is not in the index.
        #[test]
        fn materialize_night_par_none_for_missing_night() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset.materialize_night_par(&NightId(99)).is_none(),
                "Expected None for NightId(99)"
            );
        }

        /// Test 15 — the materialised Vec for NightId(1) has exactly 2 elements.
        #[test]
        fn materialize_night_par_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 13 confirmed Some
            let vec = dataset.materialize_night_par(&NightId(1)).unwrap();
            assert_eq!(vec.len(), 2);
        }

        /// Test 16 — ids in the materialised Vec for NightId(1), sorted, equal [1, 2].
        #[test]
        fn materialize_night_par_ids() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 13 confirmed Some
            let vec = dataset.materialize_night_par(&NightId(1)).unwrap();
            let mut ids: Vec<u64> = vec.iter().map(|o| o.id).collect();
            ids.sort_unstable();
            assert_eq!(ids, vec![1u64, 2]);
        }
    }

    // =======================================================================
    // mod parallel_trajectory — trajectory-related parallel methods (tests 17–29)
    // =======================================================================

    mod parallel_trajectory {
        use super::*;

        // -------------------------------------------------------------------
        // par_iter_trajectory_observations on ObsDataset
        // -------------------------------------------------------------------

        /// Test 17 — par_iter_trajectory_observations returns Some for a trajectory that
        /// exists in the index.
        #[test]
        fn par_iter_trajectory_observations_some_for_existing_traj() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset
                    .par_iter_trajectory_observations(&TrajId::Int(10))
                    .is_some(),
                "Expected Some for TrajId::Int(10) which is present in the index"
            );
        }

        /// Test 18 — par_iter_trajectory_observations returns None for a trajectory that is
        /// not in the index.
        #[test]
        fn par_iter_trajectory_observations_none_for_missing_traj() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset
                    .par_iter_trajectory_observations(&TrajId::Int(99))
                    .is_none(),
                "Expected None for TrajId::Int(99) which is absent from the index"
            );
        }

        /// Test 19 — par_iter_trajectory_observations returns None when the dataset has no
        /// trajectory index at all.
        #[test]
        fn par_iter_trajectory_observations_none_without_index() {
            let dataset = make_dataset_no_index();
            assert!(
                dataset
                    .par_iter_trajectory_observations(&TrajId::Int(10))
                    .is_none(),
                "Expected None when the dataset has no trajectory index"
            );
        }

        /// Test 20 — the per-trajectory parallel iterator for TrajId::Int(10) yields exactly
        /// 2 items (positions 0 and 2 in the observations Vec).
        #[test]
        fn par_iter_trajectory_observations_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 17 confirmed Some
            let count = dataset
                .par_iter_trajectory_observations(&TrajId::Int(10))
                .unwrap()
                .count();
            assert_eq!(count, 2);
        }

        /// Test 21 — collect ids from TrajId::Int(10)'s parallel iterator, sort, and assert
        /// they equal [1, 3] (observations at positions 0 and 2).
        #[test]
        fn par_iter_trajectory_observations_ids() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 17 confirmed Some
            let mut ids: Vec<u64> = dataset
                .par_iter_trajectory_observations(&TrajId::Int(10))
                .unwrap()
                .map(|o| o.id)
                .collect();
            ids.sort_unstable();
            assert_eq!(ids, vec![1u64, 3]);
        }

        // -------------------------------------------------------------------
        // par_iter_full_trajectory on ObsDataset
        // -------------------------------------------------------------------

        /// Test 22 — par_iter_full_trajectory returns Some when the dataset was built with a
        /// trajectory index.
        #[test]
        fn par_iter_full_trajectory_some_when_traj_index_present() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset.par_iter_full_trajectory().is_some(),
                "Expected Some when trajectory index is present"
            );
        }

        /// Test 23 — par_iter_full_trajectory returns None when the dataset has no trajectory
        /// index.
        #[test]
        fn par_iter_full_trajectory_none_when_no_traj_index() {
            let dataset = make_dataset_no_index();
            assert!(
                dataset.par_iter_full_trajectory().is_none(),
                "Expected None when no trajectory index"
            );
        }

        /// Test 24 — the full-trajectory parallel iterator yields exactly 4 pairs across
        /// 2 trajectories of 2 observations each.
        #[test]
        fn par_iter_full_trajectory_total_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 22 confirmed Some
            let count = dataset.par_iter_full_trajectory().unwrap().count();
            assert_eq!(count, 4);
        }

        /// Test 25 — collect the TrajId components from the full-trajectory iterator, sort
        /// and dedup by inner u32 value; the unique ids must be exactly
        /// [TrajId::Int(10), TrajId::Int(20)].
        #[test]
        fn par_iter_full_trajectory_traj_ids() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 22 confirmed Some
            let mut traj_ids: Vec<TrajId> = dataset
                .par_iter_full_trajectory()
                .unwrap()
                .map(|(tid, _)| tid)
                .collect();
            traj_ids.sort_unstable_by_key(traj_id_key);
            traj_ids.dedup_by_key(|id| traj_id_key(id));
            assert_eq!(traj_ids, vec![TrajId::Int(10), TrajId::Int(20)]);
        }

        // -------------------------------------------------------------------
        // materialize_trajectory_par on ObsDataset
        // -------------------------------------------------------------------

        /// Test 26 — materialize_trajectory_par returns Some(vec) for a trajectory that
        /// exists in the index.
        #[test]
        fn materialize_trajectory_par_some_for_existing_traj() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset
                    .materialize_trajectory_par(&TrajId::Int(10))
                    .is_some(),
                "Expected Some(vec) for TrajId::Int(10)"
            );
        }

        /// Test 27 — materialize_trajectory_par returns None for a trajectory that is not in
        /// the index.
        #[test]
        fn materialize_trajectory_par_none_for_missing_traj() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset
                    .materialize_trajectory_par(&TrajId::Int(99))
                    .is_none(),
                "Expected None for TrajId::Int(99)"
            );
        }

        /// Test 28 — the materialised Vec for TrajId::Int(10) has exactly 2 elements.
        #[test]
        fn materialize_trajectory_par_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 26 confirmed Some
            let vec = dataset
                .materialize_trajectory_par(&TrajId::Int(10))
                .unwrap();
            assert_eq!(vec.len(), 2);
        }

        /// Test 29 — ids in the materialised Vec for TrajId::Int(10), sorted, equal [1, 3].
        #[test]
        fn materialize_trajectory_par_ids() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 26 confirmed Some
            let vec = dataset
                .materialize_trajectory_par(&TrajId::Int(10))
                .unwrap();
            let mut ids: Vec<u64> = vec.iter().map(|o| o.id).collect();
            ids.sort_unstable();
            assert_eq!(ids, vec![1u64, 3]);
        }
    }

    // =======================================================================
    // mod parallel_index — crate-private ObsDatasetIndex parallel methods (tests 30–39)
    // =======================================================================
    //
    // ObsDatasetIndex is crate-private; we access it through the `index` field
    // of ObsDataset (also pub(crate)), which is visible here because this test
    // module lives inside the same crate.

    mod parallel_index {
        use super::*;

        // -------------------------------------------------------------------
        // par_iter_night_obs_index (tests 30–32)
        // -------------------------------------------------------------------

        /// Test 30 — par_iter_night_obs_index returns None on an index that has no night map.
        #[test]
        fn index_par_iter_night_obs_index_none_without_index() {
            let dataset = make_dataset_no_index();
            assert!(
                dataset
                    .index
                    .par_iter_night_obs_index(&NightId(1))
                    .is_none(),
                "Expected None when the dataset index has no night map"
            );
        }

        /// Test 31 — par_iter_night_obs_index returns Some for a night that is present in
        /// the night map.
        #[test]
        fn index_par_iter_night_obs_index_some_for_existing() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset
                    .index
                    .par_iter_night_obs_index(&NightId(1))
                    .is_some(),
                "Expected Some for NightId(1) which is in the night map"
            );
        }

        /// Test 32 — par_iter_night_obs_index for NightId(1) yields exactly 2 raw indices
        /// (positions 0 and 1 in the observations Vec).
        #[test]
        fn index_par_iter_night_obs_index_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 31 confirmed Some
            let count = dataset
                .index
                .par_iter_night_obs_index(&NightId(1))
                .unwrap()
                .count();
            assert_eq!(count, 2);
        }

        // -------------------------------------------------------------------
        // par_iter_full_night on ObsDatasetIndex (tests 33–34)
        // -------------------------------------------------------------------

        /// Test 33 — ObsDatasetIndex::par_iter_full_night returns None when no night map
        /// exists.
        #[test]
        fn index_par_iter_full_night_none_without_index() {
            let dataset = make_dataset_no_index();
            assert!(
                dataset.index.par_iter_full_night().is_none(),
                "Expected None from index.par_iter_full_night() when no night map"
            );
        }

        /// Test 34 — ObsDatasetIndex::par_iter_full_night returns Some and yields exactly 4
        /// (NightId, ObsIndex) pairs (2 nights × 2 observations each).
        #[test]
        fn index_par_iter_full_night_some_and_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: the night map is present
            let count = dataset.index.par_iter_full_night().unwrap().count();
            assert_eq!(count, 4);
        }

        // -------------------------------------------------------------------
        // par_iter_traj_obs_index (tests 35–37)
        // -------------------------------------------------------------------

        /// Test 35 — par_iter_traj_obs_index returns None on an index that has no trajectory
        /// map.
        #[test]
        fn index_par_iter_traj_obs_index_none_without_index() {
            let dataset = make_dataset_no_index();
            assert!(
                dataset
                    .index
                    .par_iter_traj_obs_index(&TrajId::Int(10))
                    .is_none(),
                "Expected None when the dataset index has no trajectory map"
            );
        }

        /// Test 36 — par_iter_traj_obs_index returns Some for a trajectory that is present
        /// in the trajectory map.
        #[test]
        fn index_par_iter_traj_obs_index_some_for_existing() {
            let dataset = make_dataset_with_index();
            assert!(
                dataset
                    .index
                    .par_iter_traj_obs_index(&TrajId::Int(10))
                    .is_some(),
                "Expected Some for TrajId::Int(10) which is in the trajectory map"
            );
        }

        /// Test 37 — par_iter_traj_obs_index for TrajId::Int(10) yields exactly 2 raw
        /// indices (positions 0 and 2 in the observations Vec).
        #[test]
        fn index_par_iter_traj_obs_index_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: test 36 confirmed Some
            let count = dataset
                .index
                .par_iter_traj_obs_index(&TrajId::Int(10))
                .unwrap()
                .count();
            assert_eq!(count, 2);
        }

        // -------------------------------------------------------------------
        // par_iter_full_trajectory on ObsDatasetIndex (tests 38–39)
        // -------------------------------------------------------------------

        /// Test 38 — ObsDatasetIndex::par_iter_full_trajectory returns None when no
        /// trajectory map exists.
        #[test]
        fn index_par_iter_full_trajectory_none_without_index() {
            let dataset = make_dataset_no_index();
            assert!(
                dataset.index.par_iter_full_trajectory().is_none(),
                "Expected None from index.par_iter_full_trajectory() when no trajectory map"
            );
        }

        /// Test 39 — ObsDatasetIndex::par_iter_full_trajectory returns Some and yields
        /// exactly 4 (TrajId, ObsIndex) pairs (2 trajectories × 2 observations each).
        #[test]
        fn index_par_iter_full_trajectory_some_and_count() {
            let dataset = make_dataset_with_index();
            // unwrap is safe: the trajectory map is present
            let count = dataset.index.par_iter_full_trajectory().unwrap().count();
            assert_eq!(count, 4);
        }
    }
}
