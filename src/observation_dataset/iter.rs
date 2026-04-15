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
    observation_dataset::{ObsDataset, index::ObsMapIndex, observation::Observation},
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
    /// `Some(MemLayoutObservations)` in insertion order if the trajectory index exists and the
    /// given `traj_id` is present; `None` otherwise.
    pub fn materialize_trajectory(&self, traj_id: &TrajId) -> Option<MemLayoutObservations<'_>> {
        let traj_index = self.index.obs_index_by_trajectory.as_ref()?.get(traj_id)?;
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
    pub fn len_trajectory(&self, traj_id: &TrajId) -> Option<usize> {
        self.index.len_trajectory(traj_id)
    }
}
