//! Internal index structures for efficient observation look-up.
//!
//! This module is `pub(crate)` and is not part of the public API.  It
//! provides the auxiliary index maps that back the three look-up strategies
//! available on `ObsDataset`:
//!
//! | Type | Look-up key | Description |
//! |------|-------------|-------------|
//! | `ObservationIndexMap` | `ObsId` | Maps each observation identifier to its position in the observations `Vec` |
//! | `NightIndexMap` | `NightId` | Maps each night identifier to the positions of all observations recorded that night |
//! | `TrajIndexMap` | `TrajId` | Maps each trajectory identifier to the positions of all observations belonging to that trajectory |
//!
//! All maps are backed by `AHashMap` for fast, non-cryptographic hashing.
//!
//! ## Design note
//!
//! Indices stored in these maps are zero-based positions into the
//! `observations: Vec<Observation>` field of `ObsDataset`.  Storing
//! `usize` positions rather than cloned values keeps memory usage low and
//! allows the maps to be updated or queried without holding references into
//! the main observations vector.

use ahash::AHashMap;

use crate::{NightId, TrajId, observation_dataset::ObsId};

/// Zero-based position of an observation inside the `observations` vector of `ObsDataset`.
pub type ObsIndex = usize;

/// Hash map from observation identifier to its position in the observations vector.
pub type ObservationIndexMap = AHashMap<ObsId, ObsIndex>;

/// Ordered list of observation positions belonging to a single night.
pub type NightIndex = Vec<ObsIndex>;

/// Hash map from night identifier to the list of observation positions recorded on that night.
pub type NightIndexMap = AHashMap<NightId, NightIndex>;

/// Ordered list of observation positions belonging to a single trajectory.
pub type TrajIndex = Vec<ObsIndex>;

/// Hash map from trajectory identifier to the list of observation positions assigned to that trajectory.
pub type TrajIndexMap = AHashMap<TrajId, TrajIndex>;

/// Composite index for an `ObsDataset`.
///
/// Bundles three independent look-up maps:
///
/// - A mandatory map from `ObsId` to the observation's vector position,
///   present for every dataset.
/// - An optional map from `NightId` to the vector positions of all
///   observations on that night; absent when the source data contains no
///   `night_id` column.
/// - An optional map from `TrajId` to the vector positions of all
///   observations in that trajectory; absent when the source data contains
///   no `traj_id` column.
#[derive(Debug)]
pub struct ObsDatasetIndex {
    /// Mapping from `ObsId` to the index in the `observations` vector, used for look-up by observation identifier.
    obs_index_by_id: ObservationIndexMap,

    /// Mapping from `NightId` to the list of observation indices belonging to that night.
    ///
    /// `None` when the source data contained no `night_id` column.
    pub(crate) obs_index_by_night: Option<NightIndexMap>,

    /// Mapping from `TrajId` to the list of observation indices belonging to that trajectory.
    ///
    /// `None` when the source data contained no `traj_id` column.
    pub(crate) obs_index_by_trajectory: Option<TrajIndexMap>,
}

impl ObsDatasetIndex {
    /// Construct a new `ObsDatasetIndex` from the three component maps.
    ///
    /// # Arguments
    ///
    /// - `obs_index_by_id` — mandatory map from `ObsId` to vector position.
    /// - `obs_index_by_night` — optional map from `NightId` to vector positions; pass `None`
    ///   when the source data has no `night_id` column.
    /// - `obs_index_by_trajectory` — optional map from `TrajId` to vector positions; pass
    ///   `None` when the source data has no `traj_id` column.
    ///
    /// # Returns
    ///
    /// A fully initialised `ObsDatasetIndex`.
    #[cfg_attr(not(feature = "polars"), allow(dead_code))]
    pub(crate) fn new(
        obs_index_by_id: ObservationIndexMap,
        obs_index_by_night: Option<NightIndexMap>,
        obs_index_by_trajectory: Option<TrajIndexMap>,
    ) -> Self {
        Self {
            obs_index_by_id,
            obs_index_by_night,
            obs_index_by_trajectory,
        }
    }

    /// Return the number of observations recorded on the given night.
    ///
    /// # Arguments
    ///
    /// - `night_id` — the night whose observation count is requested.
    ///
    /// # Returns
    ///
    /// `Some(count)` if the night index exists and the night is present in it;
    /// `None` if no night index was built or the night identifier is unknown.
    pub(crate) fn len_night(&self, night_id: &NightId) -> Option<usize> {
        self.obs_index_by_night
            .as_ref()?
            .get(night_id)
            .map(|indices| indices.len())
    }

    /// Return the number of observations assigned to the given trajectory.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the trajectory whose observation count is requested.
    ///
    /// # Returns
    ///
    /// `Some(count)` if the trajectory index exists and the trajectory is present in it;
    /// `None` if no trajectory index was built or the trajectory identifier is unknown.
    pub(crate) fn len_trajectory(&self, traj_id: &TrajId) -> Option<usize> {
        self.obs_index_by_trajectory
            .as_ref()?
            .get(traj_id)
            .map(|indices| indices.len())
    }

    /// Return an iterator over all `NightId` keys present in the night index.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if a night index was built; `None` otherwise.
    /// The iteration order is unspecified (hash map key order).
    pub(crate) fn iter_night_id(&self) -> Option<impl Iterator<Item = &NightId>> {
        self.obs_index_by_night
            .as_ref()
            .map(|night_map| night_map.keys())
    }

    /// Return an iterator over all `TrajId` keys present in the trajectory index.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if a trajectory index was built; `None` otherwise.
    /// The iteration order is unspecified (hash map key order).
    pub(crate) fn iter_traj_id(&self) -> Option<impl Iterator<Item = &TrajId>> {
        self.obs_index_by_trajectory
            .as_ref()
            .map(|traj_map| traj_map.keys())
    }

    /// Look up the vector position of an observation by its identifier.
    ///
    /// # Arguments
    ///
    /// - `obs_id` — the observation identifier to look up.
    ///
    /// # Returns
    ///
    /// `Some(index)` if the observation exists; `None` otherwise.
    pub(crate) fn get_by_id(&self, obs_id: &ObsId) -> Option<ObsIndex> {
        self.obs_index_by_id.get(obs_id).copied()
    }

    /// Return the list of vector positions for all observations on a given night.
    ///
    /// # Arguments
    ///
    /// - `night_id` — the night identifier to look up.
    ///
    /// # Returns
    ///
    /// `Some(&NightIndex)` if a night index exists and the night is present in it;
    /// `None` otherwise.
    pub(crate) fn get_by_night(&self, night_id: &NightId) -> Option<&NightIndex> {
        self.obs_index_by_night.as_ref()?.get(night_id)
    }

    /// Return an iterator over the vector positions of all observations on a given night.
    ///
    /// # Arguments
    ///
    /// - `night_id` — the night identifier whose observation positions are requested.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` yielding each `ObsIndex` in insertion order if the night index
    /// exists and the night is present; `None` otherwise.
    pub(crate) fn iter_night_obs_index(
        &self,
        night_id: &NightId,
    ) -> Option<impl Iterator<Item = ObsIndex> + '_> {
        self.get_by_night(night_id)
            .map(|indices| indices.iter().copied())
    }

    /// Return an iterator over `(NightId, ObsIndex)` pairs for every observation in the night index.
    ///
    /// Each pair associates a night identifier with the vector position of one of the
    /// observations recorded on that night.  Observations from the same night appear
    /// as consecutive pairs in the iteration, but the order between nights is
    /// unspecified (hash map key order).
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if a night index was built; `None` otherwise.
    pub(crate) fn iter_full_night(&self) -> Option<impl Iterator<Item = (NightId, ObsIndex)> + '_> {
        self.obs_index_by_night.as_ref().map(|night_map| {
            night_map
                .iter()
                .flat_map(|(night_id, indices)| indices.iter().map(move |&idx| (*night_id, idx)))
        })
    }

    /// Return the list of vector positions for all observations in a given trajectory.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the trajectory identifier to look up.
    ///
    /// # Returns
    ///
    /// `Some(&TrajIndex)` if a trajectory index exists and the trajectory is present in it;
    /// `None` otherwise.
    pub(crate) fn get_by_trajectory(&self, traj_id: &TrajId) -> Option<&TrajIndex> {
        self.obs_index_by_trajectory.as_ref()?.get(traj_id)
    }

    /// Return an iterator over the vector positions of all observations in a given trajectory.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the trajectory identifier whose observation positions are requested.
    ///
    /// # Returns
    ///
    /// `Some(iterator)` yielding each `ObsIndex` in insertion order if the trajectory index
    /// exists and the trajectory is present; `None` otherwise.
    pub(crate) fn iter_traj_obs_index(
        &self,
        traj_id: &TrajId,
    ) -> Option<impl Iterator<Item = ObsIndex> + '_> {
        self.get_by_trajectory(traj_id)
            .map(|indices| indices.iter().copied())
    }

    /// Return an iterator over `(TrajId, ObsIndex)` pairs for every observation in the trajectory index.
    ///
    /// Each pair associates a trajectory identifier with the vector position of one of the
    /// observations belonging to that trajectory.  Observations from the same trajectory
    /// appear as consecutive pairs in the iteration, but the order between trajectories is
    /// unspecified (hash map key order).
    ///
    /// # Returns
    ///
    /// `Some(iterator)` if a trajectory index was built; `None` otherwise.
    pub(crate) fn iter_full_trajectory(
        &self,
    ) -> Option<impl Iterator<Item = (TrajId, ObsIndex)> + '_> {
        self.obs_index_by_trajectory.as_ref().map(|traj_map| {
            traj_map.iter().flat_map(|(traj_id, indices)| {
                indices.iter().map(move |&idx| (traj_id.clone(), idx))
            })
        })
    }

    /// Insert or replace the trajectory entry for a given `TrajId`.
    ///
    /// If the trajectory index was not built (i.e. `obs_index_by_trajectory` is `None`),
    /// this method is a no-op.  Otherwise the provided slice of observation positions is
    /// stored (as a new `Vec`) under `traj_id`, replacing any pre-existing entry for that key.
    ///
    /// # Arguments
    ///
    /// - `traj_id` — the trajectory identifier to insert or replace.
    /// - `obs_index` — slice of vector positions for the observations belonging to this trajectory.
    pub(crate) fn push_trajectory(&mut self, traj_id: TrajId, obs_index: &[ObsIndex]) {
        if let Some(traj_map) = self.obs_index_by_trajectory.as_mut() {
            traj_map.insert(traj_id, obs_index.to_vec());
        }
    }
}
