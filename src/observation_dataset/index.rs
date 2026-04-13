use ahash::AHashMap;

use crate::{NightId, TrajId, observation_dataset::ObsId};

pub type ObsIndex = usize;
pub type ObservationIndexMap = AHashMap<ObsId, ObsIndex>;

pub type NightIndex = Vec<ObsIndex>;
pub type NightIndexMap = AHashMap<NightId, NightIndex>;

pub type TrajIndex = Vec<ObsIndex>;
pub type TrajIndexMap = AHashMap<TrajId, TrajIndex>;

#[derive(Debug)]
pub struct ObsDatasetIndex {
    /// Mapping from ObsId to index in the `observations` list, used for look-up by id.
    obs_index_by_id: ObservationIndexMap,

    /// Mapping from NightId to the list of observation indices in the `observations` list that belong to that night,
    /// used for look-up by night.
    obs_index_by_night: Option<NightIndexMap>,

    /// Mapping from TrajId to the list of observation indices in the `observations` list that belong to that trajectory,
    obs_index_by_trajectory: Option<TrajIndexMap>,
}

impl ObsDatasetIndex {
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

    pub(crate) fn len_night(&self, night_id: &NightId) -> Option<usize> {
        self.obs_index_by_night
            .as_ref()?
            .get(night_id)
            .map(|indices| indices.len())
    }

    pub(crate) fn len_trajectory(&self, traj_id: &TrajId) -> Option<usize> {
        self.obs_index_by_trajectory
            .as_ref()?
            .get(traj_id)
            .map(|indices| indices.len())
    }

    pub(crate) fn iter_night_id(&self) -> Option<impl Iterator<Item = &NightId>> {
        self.obs_index_by_night
            .as_ref()
            .map(|night_map| night_map.keys())
    }

    pub(crate) fn iter_traj_id(&self) -> Option<impl Iterator<Item = &TrajId>> {
        self.obs_index_by_trajectory
            .as_ref()
            .map(|traj_map| traj_map.keys())
    }

    pub(crate) fn get_by_id(&self, obs_id: &ObsId) -> Option<ObsIndex> {
        self.obs_index_by_id.get(obs_id).copied()
    }

    pub(crate) fn get_by_night(&self, night_id: &NightId) -> Option<&NightIndex> {
        self.obs_index_by_night.as_ref()?.get(night_id)
    }

    pub(crate) fn iter_night_obs_index(
        &self,
        night_id: &NightId,
    ) -> Option<impl Iterator<Item = ObsIndex> + '_> {
        self.get_by_night(night_id)
            .map(|indices| indices.iter().copied())
    }

    pub(crate) fn iter_full_night(&self) -> Option<impl Iterator<Item = (NightId, ObsIndex)> + '_> {
        self.obs_index_by_night.as_ref().map(|night_map| {
            night_map
                .iter()
                .flat_map(|(night_id, indices)| indices.iter().map(move |&idx| (*night_id, idx)))
        })
    }

    pub(crate) fn get_by_trajectory(&self, traj_id: &TrajId) -> Option<&TrajIndex> {
        self.obs_index_by_trajectory.as_ref()?.get(traj_id)
    }

    pub(crate) fn iter_traj_obs_index(
        &self,
        traj_id: &TrajId,
    ) -> Option<impl Iterator<Item = ObsIndex> + '_> {
        self.get_by_trajectory(traj_id)
            .map(|indices| indices.iter().copied())
    }

    pub(crate) fn iter_full_trajectory(
        &self,
    ) -> Option<impl Iterator<Item = (TrajId, ObsIndex)> + '_> {
        self.obs_index_by_trajectory.as_ref().map(|traj_map| {
            traj_map.iter().flat_map(|(traj_id, indices)| {
                indices.iter().map(move |&idx| (traj_id.clone(), idx))
            })
        })
    }

    pub(crate) fn push_trajectory(&mut self, traj_id: TrajId, obs_index: &[ObsIndex]) {
        if let Some(traj_map) = self.obs_index_by_trajectory.as_mut() {
            traj_map.insert(traj_id, obs_index.to_vec());
        }
    }
}
