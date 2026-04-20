// ---------------------------------------------------------------------------
// Serde support for ObsDataset
// ---------------------------------------------------------------------------
//
// `ObsDataset` cannot use a derived `Serialize`/`Deserialize` because several
// of its fields are runtime-only state that must not be persisted:
//
// - `index`         — rebuilt at deserialisation from the per-observation
//                     `night_id` / `traj_ids` fields stored in each proxy.
// - `observer_dataset` — serialised via its own custom impl (`observer` sub-
//                     module), which skips the lazy MPC network cache.
//
// ## Proxy design
//
// Serialisation uses two proxy structs:
//
// - `ObservationProxy` — wraps every field of `Observation` and adds:
//     - `night_id: Option<NightId>` — the night this observation belongs to.
//     - `traj_ids: Vec<TrajId>`     — trajectories this observation belongs to
//                                    (an observation may belong to several).
//
// - `ObsDatasetProxy` — wraps `Vec<ObservationProxy>`, `ObserverDataset`,
//   and `traj_aliases` (alternate designation → canonical
//   `TrajId` map, used by the MPC 80-column reader).
//
// ## Index layout at deserialisation
//
// The standard `Deserialize for ObsDataset` impl rebuilds indices using
// `Split` entries (safe for any observation ordering).
//
// Users who need `Contiguous` entries (e.g. after an ingestion that sorted
// observations contiguously by night/trajectory) can use `ObsDatasetSeed`
// with `IndexLayout::TryContiguous` via the `DeserializeSeed` trait.  This
// is format-agnostic: the same seed works with JSON, bincode, YAML, etc.

pub mod observer;

use std::collections::HashMap;

use serde::{Deserialize, Serialize, de::DeserializeSeed, ser::SerializeStruct};

use crate::{
    MJDTT, NightId, TrajId,
    astrometry::EquCoord,
    observation_dataset::ObsId,
    observation_dataset::{
        ObsDataset,
        index::{NightIndexMap, ObsIndex, ObsMapIndex, TrajAliasMap, TrajIndexMap},
        observation::Observation,
    },
    observer::dataset::{ObserverDataset, ObserverId},
    photometry::Photometry,
};

// ---------------------------------------------------------------------------
// Format version
// ---------------------------------------------------------------------------

/// Current serialisation format version.
///
/// Increment this constant when the on-disk layout changes in a
/// backwards-incompatible way so that deserialisation code can reject (or
/// migrate) stale payloads.
///
/// History:
/// - `1` — initial release (no night/traj index, no aliases).
/// - `2` — adds `night_id`, `traj_ids` per observation and `traj_aliases`.
const FORMAT_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// ObservationProxy
// ---------------------------------------------------------------------------

/// Serialisable proxy for a single [`Observation`] **in the context of an
/// [`ObsDataset`]**.
///
/// In addition to all fields of [`Observation`], this proxy carries:
///
/// - `night_id`  — the [`NightId`] of the night this observation belongs to,
///   or `None` if no night index was built or this observation has no night.
/// - `traj_ids`  — the list of [`TrajId`]s of the trajectories this
///   observation belongs to.  An observation may belong to zero, one, or
///   several trajectories.
///
/// [`Observation`] itself keeps its own standalone `Serialize`/`Deserialize`
/// derives (useful for individual round-trips); this proxy is only used
/// inside the `ObsDataset` serialisation path.
#[derive(Serialize, Deserialize)]
struct ObservationProxy {
    // ── Observation fields ───────────────────────────────────────────────
    index: ObsIndex,
    id: ObsId,
    equ_coord: EquCoord,
    photometry: Photometry,
    mjd_tt: MJDTT,
    observer: Option<ObserverId>,
    // ── Index membership ────────────────────────────────────────────────
    /// Night this observation belongs to (`None` when no night index exists
    /// or this observation has not been assigned to a night).
    night_id: Option<NightId>,
    /// Trajectories this observation belongs to (empty when no trajectory
    /// index exists or this observation has not been assigned to any
    /// trajectory).
    traj_ids: Vec<TrajId>,
}

impl ObservationProxy {
    /// Extract the core [`Observation`] fields, discarding the index hints.
    fn into_observation(self) -> Observation {
        Observation {
            index: self.index,
            id: self.id,
            equ_coord: self.equ_coord,
            photometry: self.photometry,
            mjd_tt: self.mjd_tt,
            observer: self.observer,
        }
    }
}

// ---------------------------------------------------------------------------
// ObsDatasetProxy
// ---------------------------------------------------------------------------

/// Serialisable proxy for [`ObsDataset`].
///
/// Fields:
/// - `format_version`   — schema version guard; bumped on breaking changes.
/// - `observations`     — full list of [`ObservationProxy`] values (one per
///   observation, in insertion order).
/// - `observer_dataset` — custom observer sites and error-model variant; the
///   lazy MPC network cache is **not** included.
/// - `traj_aliases`     — alternate trajectory designations (e.g. provisional
///   MPC designations) and their canonical [`TrajId`].
#[derive(Deserialize)]
struct ObsDatasetProxy {
    format_version: u32,
    observations: Vec<ObservationProxy>,
    observer_dataset: ObserverDataset,
    traj_aliases: Vec<(String, TrajId)>,
}

// ---------------------------------------------------------------------------
// IndexLayout
// ---------------------------------------------------------------------------

/// Controls how night/trajectory index entries are represented after
/// deserialisation of an [`ObsDataset`].
///
/// Pass this to [`ObsDatasetSeed`] when using the [`DeserializeSeed`] API.
/// The standard [`Deserialize`] impl always uses [`IndexLayout::Split`].
pub enum IndexLayout {
    /// Always build `Split` index entries (a `Vec` of scattered positions).
    ///
    /// Safe regardless of observation ordering.  **This is the default used
    /// by `Deserialize for ObsDataset`.**
    Split,

    /// Attempt to build `Contiguous` index entries (a `start..end` range)
    /// when the observations belonging to a group occupy a contiguous block
    /// in the observations vector.  Falls back to `Split` for any group
    /// whose positions contain gaps.
    ///
    /// Choose this when you know the dataset was ingested with a contiguous
    /// sort order (e.g. `ContiguousNight` or `ContiguousTraj`).
    TryContiguous,
}

// ---------------------------------------------------------------------------
// ObsDatasetSeed
// ---------------------------------------------------------------------------

/// Seed for deserialising an [`ObsDataset`] with a custom [`IndexLayout`].
///
/// Implements [`DeserializeSeed`], making it usable with **any serde-
/// compatible format** — JSON, bincode, YAML, MessagePack, etc.
///
/// # Example
///
/// ```rust,ignore
/// use photom::io::serde::{IndexLayout, ObsDatasetSeed};
/// use serde::de::DeserializeSeed;
///
/// // JSON
/// let mut de = serde_json::Deserializer::from_str(json);
/// let dataset = ObsDatasetSeed { layout: IndexLayout::TryContiguous }
///     .deserialize(&mut de)?;
///
/// // Bincode (or any other format)
/// let dataset = ObsDatasetSeed { layout: IndexLayout::TryContiguous }
///     .deserialize(&mut bincode_deserializer)?;
/// ```
pub struct ObsDatasetSeed {
    pub layout: IndexLayout,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert an accumulated `Vec<ObsIndex>` into the appropriate [`ObsMapIndex`]
/// variant given the requested layout.
///
/// For `TryContiguous`, the indices are sorted and tested for contiguity
/// (no gaps).  A contiguous run is stored as `Contiguous { start, end }`
/// — otherwise `Split` is used as fallback.
fn to_obs_map_index(mut indices: Vec<ObsIndex>, layout: &IndexLayout) -> ObsMapIndex {
    match layout {
        IndexLayout::Split => ObsMapIndex::Split(indices),
        IndexLayout::TryContiguous => {
            indices.sort_unstable();
            let n = indices.len();
            let start = indices[0];
            let end = indices[n - 1] + 1;
            if end - start == n {
                // Contiguous: no gaps between start and end - 1.
                ObsMapIndex::Contiguous { start, end }
            } else {
                ObsMapIndex::Split(indices)
            }
        }
    }
}

/// Rebuild `NightIndexMap` and `TrajIndexMap` from the per-observation proxy
/// data, applying the requested [`IndexLayout`].
///
/// Returns `(None, None)` for an index that has no entries at all, mirroring
/// the convention used during ingestion (absent column → `None` index).
fn build_index_maps(
    proxies: &[ObservationProxy],
    layout: &IndexLayout,
) -> (Option<NightIndexMap>, Option<TrajIndexMap>) {
    // Accumulate positions into plain Vecs first, then convert to ObsMapIndex.
    let mut night_acc: HashMap<NightId, Vec<ObsIndex>> = HashMap::new();
    let mut traj_acc: HashMap<TrajId, Vec<ObsIndex>> = HashMap::new();

    for (obs_idx, proxy) in proxies.iter().enumerate() {
        if let Some(nid) = proxy.night_id {
            night_acc.entry(nid).or_default().push(obs_idx);
        }
        for tid in &proxy.traj_ids {
            traj_acc.entry(tid.clone()).or_default().push(obs_idx);
        }
    }

    let night_map = if night_acc.is_empty() {
        None
    } else {
        Some(
            night_acc
                .into_iter()
                .map(|(nid, indices)| (nid, to_obs_map_index(indices, layout)))
                .collect(),
        )
    };

    let traj_map = if traj_acc.is_empty() {
        None
    } else {
        Some(
            traj_acc
                .into_iter()
                .map(|(tid, indices)| (tid, to_obs_map_index(indices, layout)))
                .collect(),
        )
    };

    (night_map, traj_map)
}

/// Build an [`ObsDataset`] from a fully-deserialised proxy and a layout hint.
fn dataset_from_proxy<E: serde::de::Error>(
    proxy: ObsDatasetProxy,
    layout: &IndexLayout,
) -> Result<ObsDataset, E> {
    if proxy.format_version != FORMAT_VERSION {
        return Err(serde::de::Error::custom(format!(
            "unsupported ObsDataset format version {} (expected {})",
            proxy.format_version, FORMAT_VERSION,
        )));
    }

    let (night_map, traj_map) = build_index_maps(&proxy.observations, layout);

    let observations: Vec<Observation> = proxy
        .observations
        .into_iter()
        .map(ObservationProxy::into_observation)
        .collect();

    let traj_aliases: TrajAliasMap = proxy.traj_aliases.into_iter().collect();

    Ok(ObsDataset::new_from_parts(
        observations,
        proxy.observer_dataset,
        night_map,
        traj_map,
        traj_aliases,
    ))
}

// ---------------------------------------------------------------------------
// Serialize for ObsDataset
// ---------------------------------------------------------------------------

impl Serialize for ObsDataset {
    /// Serialise the persistent state of the dataset.
    ///
    /// Five top-level fields are written:
    ///
    /// - `format_version`   — schema version; currently `FORMAT_VERSION`.
    /// - `observations`     — list of `ObservationProxy` values, each
    ///   carrying the core observation data plus the `night_id` and
    ///   `traj_ids` membership hints needed to rebuild the index maps.
    /// - `observer_dataset` — custom observer sites and error-model variant;
    ///   the lazy MPC network cache is **not** included.
    /// - `traj_aliases`     — alternate trajectory designation → canonical
    ///   [`TrajId`] pairs.
    ///
    /// Runtime-only state (MPC network cache, index maps) is
    /// not written; it is either rebuilt or re-initialised on deserialisation.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // ── Build reverse look-up tables from the index maps ────────────
        //
        // night_by_pos: ObsIndex → NightId
        // (an observation belongs to at most one night)
        let night_by_pos: HashMap<ObsIndex, NightId> = match self.index.iter_full_night() {
            Some(iter) => iter.map(|(nid, idx)| (idx, nid)).collect(),
            None => HashMap::new(),
        };

        // traj_by_pos: ObsIndex → Vec<TrajId>
        // (an observation may belong to several trajectories)
        let mut traj_by_pos: HashMap<ObsIndex, Vec<TrajId>> = HashMap::new();
        if let Some(iter) = self.index.iter_full_trajectory() {
            for (tid, idx) in iter {
                traj_by_pos.entry(idx).or_default().push(tid);
            }
        }

        // ── Build ObservationProxy list ─────────────────────────────────
        let proxies: Vec<ObservationProxy> = self
            .observations
            .iter()
            .map(|obs| ObservationProxy {
                index: obs.index,
                id: obs.id,
                equ_coord: obs.equ_coord,
                photometry: obs.photometry.clone(),
                mjd_tt: obs.mjd_tt,
                observer: obs.observer,
                night_id: night_by_pos.get(&obs.index).copied(),
                traj_ids: traj_by_pos.get(&obs.index).cloned().unwrap_or_default(),
            })
            .collect();

        // ── Collect traj aliases ────────────────────────────────────────
        let traj_aliases: Vec<(String, TrajId)> = self
            .index
            .iter_aliases()
            .map(|(alias, tid)| (alias.to_owned(), tid.clone()))
            .collect();

        // ── Serialise ───────────────────────────────────────────────────
        let mut s = serializer.serialize_struct("ObsDataset", 4)?;
        s.serialize_field("format_version", &FORMAT_VERSION)?;
        s.serialize_field("observations", &proxies)?;
        s.serialize_field("observer_dataset", &self.observer_dataset)?;
        s.serialize_field("traj_aliases", &traj_aliases)?;
        s.end()
    }
}

// ---------------------------------------------------------------------------
// Deserialize for ObsDataset  (default: Split layout)
// ---------------------------------------------------------------------------

impl<'de> Deserialize<'de> for ObsDataset {
    /// Deserialise an [`ObsDataset`] using the [`IndexLayout::Split`] layout.
    ///
    /// Night and trajectory index entries are rebuilt as `Split` (scattered
    /// position lists), which is safe regardless of observation ordering.
    ///
    /// To choose `TryContiguous` instead, use [`ObsDatasetSeed`] with the
    /// [`DeserializeSeed`] API — it works with any serde-compatible format.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ObsDatasetSeed {
            layout: IndexLayout::Split,
        }
        .deserialize(deserializer)
    }
}

// ---------------------------------------------------------------------------
// DeserializeSeed for ObsDatasetSeed  (configurable layout)
// ---------------------------------------------------------------------------

impl<'de> DeserializeSeed<'de> for ObsDatasetSeed {
    type Value = ObsDataset;

    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        let proxy = ObsDatasetProxy::deserialize(deserializer)?;
        dataset_from_proxy(proxy, &self.layout)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod obsdataset_serde_tests {
    use serde::de::DeserializeSeed;

    use crate::{
        NightId, TrajId,
        astrometry::EquCoord,
        observation_dataset::{
            ObsDataset,
            index::{NightIndexMap, ObsIndex, ObsMapIndex, TrajIndexMap},
            observation::Observation,
        },
        observer::{Observer, dataset::ObserverId},
        photometry::{Filter, Photometry},
    };

    use super::{IndexLayout, ObsDatasetSeed};

    // ── helpers ─────────────────────────────────────────────────────────

    fn make_obs(id: u64, idx: ObsIndex, mjd_tt: f64) -> Observation {
        Observation {
            index: idx,
            id,
            equ_coord: EquCoord::new(0.1 + id as f64 * 0.01, 0.001, 0.2, 0.001),
            photometry: Photometry {
                magnitude: 18.0 + id as f64 * 0.5,
                error: 0.05,
                filter: Filter::String("V".to_string()),
            },
            mjd_tt,
            observer: None,
        }
    }

    fn make_observer() -> Observer {
        Observer::new(0.0, 0.7, 100.0, Some("Test site".to_string()), None, None)
            .expect("valid observer")
    }

    /// Build a dataset without night/traj indices.
    fn build_basic_dataset() -> ObsDataset {
        let obs0 = Observation {
            index: 0,
            id: 0,
            equ_coord: EquCoord::new(0.1, 0.001, 0.2, 0.001),
            photometry: Photometry {
                magnitude: 18.0,
                error: 0.05,
                filter: Filter::String("V".to_string()),
            },
            mjd_tt: 59_000.0,
            observer: Some(ObserverId::IntId(0)),
        };
        let obs1 = Observation {
            index: 1,
            id: 1,
            equ_coord: EquCoord::new(0.15, 0.001, 0.25, 0.001),
            photometry: Photometry {
                magnitude: 19.0,
                error: 0.05,
                filter: Filter::String("r".to_string()),
            },
            mjd_tt: 59_001.0,
            observer: None,
        };
        ObsDataset::new(vec![obs0, obs1], vec![make_observer()], None, None, None)
    }

    /// Build a dataset with a night index (two nights, two obs each).
    fn build_dataset_with_nights() -> ObsDataset {
        let obs = vec![
            make_obs(0, 0, 59_000.5),
            make_obs(1, 1, 59_000.6),
            make_obs(2, 2, 59_001.5),
            make_obs(3, 3, 59_001.6),
        ];
        let mut night_map: NightIndexMap = ahash::AHashMap::new();
        night_map.insert(NightId(59_000), ObsMapIndex::Split(vec![0, 1]));
        night_map.insert(NightId(59_001), ObsMapIndex::Split(vec![2, 3]));

        ObsDataset::new(obs, vec![], None, Some(night_map), None)
    }

    /// Build a dataset with a trajectory index.
    fn build_dataset_with_trajs() -> ObsDataset {
        let obs = vec![
            make_obs(0, 0, 59_000.5),
            make_obs(1, 1, 59_001.5),
            make_obs(2, 2, 59_002.5),
        ];
        let mut traj_map: TrajIndexMap = ahash::AHashMap::new();
        traj_map.insert(
            TrajId::Str("2020 AV2".to_string()),
            ObsMapIndex::Split(vec![0, 1]),
        );
        traj_map.insert(
            TrajId::Str("Ceres".to_string()),
            ObsMapIndex::Split(vec![2]),
        );

        ObsDataset::new(obs, vec![], None, None, Some(traj_map))
    }

    /// Build a dataset where one observation belongs to two trajectories.
    fn build_dataset_obs_in_multiple_trajs() -> ObsDataset {
        let obs = vec![
            make_obs(0, 0, 59_000.5),
            make_obs(1, 1, 59_001.5),
            make_obs(2, 2, 59_002.5),
        ];
        // obs index 1 belongs to both trajectories.
        let mut traj_map: TrajIndexMap = ahash::AHashMap::new();
        traj_map.insert(TrajId::Int(1), ObsMapIndex::Split(vec![0, 1]));
        traj_map.insert(TrajId::Int(2), ObsMapIndex::Split(vec![1, 2]));

        ObsDataset::new(obs, vec![], None, None, Some(traj_map))
    }

    // ── Serialise helper ────────────────────────────────────────────────
    fn roundtrip(ds: &ObsDataset) -> ObsDataset {
        let json = serde_json::to_string(ds).expect("serialise");
        serde_json::from_str(&json).expect("deserialise")
    }

    fn roundtrip_contiguous(ds: &ObsDataset) -> ObsDataset {
        let json = serde_json::to_string(ds).expect("serialise");
        let mut de = serde_json::Deserializer::from_str(&json);
        ObsDatasetSeed {
            layout: IndexLayout::TryContiguous,
        }
        .deserialize(&mut de)
        .expect("deserialise with TryContiguous")
    }

    // ── Basic round-trip (no indices) ────────────────────────────────────

    #[test]
    fn round_trip_observation_count() {
        let ds = build_basic_dataset();
        let restored = roundtrip(&ds);
        assert_eq!(ds.observation_count(), restored.observation_count());
    }

    #[test]
    fn round_trip_get_observation_by_id() {
        let ds = build_basic_dataset();
        let mut restored = roundtrip(&ds);
        assert!(restored.get_observation(0).is_some());
        assert!(restored.get_observation(1).is_some());
        assert!(restored.get_observation(99).is_none());
    }

    #[test]
    fn round_trip_custom_observer() {
        let ds = build_basic_dataset();
        let mut restored = roundtrip(&ds);
        let observer = restored.get_observer(0).expect("observer must resolve");
        assert_eq!(observer.name.as_deref(), Some("Test site"));
    }

    // ── Standalone Observation round-trip (unchanged) ────────────────────

    #[test]
    fn observation_standalone_round_trip_fields() {
        let obs = Observation {
            index: 3,
            id: 42,
            equ_coord: EquCoord::new(1.23, 0.002, -0.45, 0.003),
            photometry: Photometry {
                magnitude: 17.3,
                error: 0.04,
                filter: Filter::Int(5),
            },
            mjd_tt: 59_100.5,
            observer: Some(ObserverId::IntId(0)),
        };
        let json = serde_json::to_string(&obs).expect("serialise");
        let restored: Observation = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(obs, restored);
        assert_eq!(obs.id(), restored.id());
        assert_eq!(obs.mjd_tt(), restored.mjd_tt());
        assert_eq!(obs.equ_coord(), restored.equ_coord());
        assert_eq!(obs.photometry(), restored.photometry());
        assert_eq!(obs.index(), restored.index());
    }

    #[test]
    fn observation_ordering_preserved_after_round_trip() {
        let earlier = make_obs(1, 0, 59_000.0);
        let later = make_obs(2, 1, 59_001.0);
        let r_e: Observation =
            serde_json::from_str(&serde_json::to_string(&earlier).unwrap()).unwrap();
        let r_l: Observation =
            serde_json::from_str(&serde_json::to_string(&later).unwrap()).unwrap();
        assert!(
            r_e < r_l,
            "earlier mjd_tt must sort before later after round-trip"
        );
    }

    // ── Night index round-trip ───────────────────────────────────────────

    #[test]
    fn round_trip_night_index_len() {
        let ds = build_dataset_with_nights();
        let restored = roundtrip(&ds);

        assert_eq!(
            ds.index.len_night(&NightId(59_000)),
            restored.index.len_night(&NightId(59_000)),
        );
        assert_eq!(
            ds.index.len_night(&NightId(59_001)),
            restored.index.len_night(&NightId(59_001)),
        );
    }

    #[test]
    fn round_trip_night_index_iter_full_night() {
        let ds = build_dataset_with_nights();
        let restored = roundtrip(&ds);

        let mut orig_pairs: Vec<(NightId, ObsIndex)> =
            ds.index.iter_full_night().unwrap().collect();
        let mut rest_pairs: Vec<(NightId, ObsIndex)> =
            restored.index.iter_full_night().unwrap().collect();

        orig_pairs.sort();
        rest_pairs.sort();
        assert_eq!(orig_pairs, rest_pairs);
    }

    #[test]
    fn round_trip_no_night_index_stays_none() {
        let ds = build_basic_dataset();
        let restored = roundtrip(&ds);
        assert!(restored.index.iter_full_night().is_none());
    }

    // ── Trajectory index round-trip ──────────────────────────────────────

    #[test]
    fn round_trip_traj_index_len() {
        let ds = build_dataset_with_trajs();
        let restored = roundtrip(&ds);

        assert_eq!(
            ds.index
                .len_trajectory(&TrajId::Str("2020 AV2".to_string())),
            restored
                .index
                .len_trajectory(&TrajId::Str("2020 AV2".to_string())),
        );
        assert_eq!(
            ds.index.len_trajectory(&TrajId::Str("Ceres".to_string())),
            restored
                .index
                .len_trajectory(&TrajId::Str("Ceres".to_string())),
        );
    }

    #[test]
    fn round_trip_traj_index_iter_full_trajectory() {
        let ds = build_dataset_with_trajs();
        let restored = roundtrip(&ds);

        let mut orig: Vec<(TrajId, ObsIndex)> = ds.index.iter_full_trajectory().unwrap().collect();
        let mut rest: Vec<(TrajId, ObsIndex)> =
            restored.index.iter_full_trajectory().unwrap().collect();

        orig.sort_by_key(|(_, i)| *i);
        rest.sort_by_key(|(_, i)| *i);
        assert_eq!(orig, rest);
    }

    #[test]
    fn round_trip_no_traj_index_stays_none() {
        let ds = build_basic_dataset();
        let restored = roundtrip(&ds);
        assert!(restored.index.iter_full_trajectory().is_none());
    }

    // ── Observation in multiple trajectories ─────────────────────────────

    #[test]
    fn round_trip_obs_in_multiple_trajectories() {
        let ds = build_dataset_obs_in_multiple_trajs();
        let restored = roundtrip(&ds);

        // obs at index 1 must appear in both TrajId::Int(1) and TrajId::Int(2).
        let traj1_indices: Vec<ObsIndex> = restored
            .index
            .iter_traj_obs_index(&TrajId::Int(1))
            .expect("traj 1 must exist")
            .collect();
        let traj2_indices: Vec<ObsIndex> = restored
            .index
            .iter_traj_obs_index(&TrajId::Int(2))
            .expect("traj 2 must exist")
            .collect();

        assert!(traj1_indices.contains(&1), "obs 1 must be in traj 1");
        assert!(traj2_indices.contains(&1), "obs 1 must be in traj 2");
    }

    // ── Trajectory aliases ───────────────────────────────────────────────

    #[test]
    fn round_trip_traj_aliases() {
        let obs = vec![make_obs(0, 0, 59_000.0)];
        let mut traj_map: TrajIndexMap = ahash::AHashMap::new();
        traj_map.insert(
            TrajId::Str("2003 QQ47".to_string()),
            ObsMapIndex::Split(vec![0]),
        );
        let mut ds = ObsDataset::new(obs, vec![], None, None, Some(traj_map));
        // Register an alias: "QQ47" → TrajId::Str("2003 QQ47")
        ds.index.set_aliases(
            [("QQ47".to_string(), TrajId::Str("2003 QQ47".to_string()))]
                .into_iter()
                .collect(),
        );

        let restored = roundtrip(&ds);

        assert_eq!(
            restored.resolve_alias("QQ47"),
            Some(&TrajId::Str("2003 QQ47".to_string())),
            "alias must survive round-trip"
        );
        assert!(
            restored.resolve_alias("unknown").is_none(),
            "unregistered alias must return None"
        );
    }

    // ── IndexLayout::TryContiguous ───────────────────────────────────────

    #[test]
    fn seed_try_contiguous_produces_contiguous_when_sorted() {
        // obs 0,1 are night 59_000; obs 2,3 are night 59_001 — both contiguous.
        let ds = build_dataset_with_nights();
        let restored = roundtrip_contiguous(&ds);

        // Both nights must be represented as Contiguous in the restored index.
        let night_map = restored
            .index
            .obs_index_by_night
            .as_ref()
            .expect("night index must exist");

        for nid in [NightId(59_000), NightId(59_001)] {
            match night_map.get(&nid).expect("night must be present") {
                ObsMapIndex::Contiguous { .. } => {} // correct
                ObsMapIndex::Split(_) => panic!("expected Contiguous for {nid:?}"),
            }
        }
    }

    #[test]
    fn seed_try_contiguous_falls_back_to_split_when_not_contiguous() {
        // Build a dataset where the observations of a night are NOT contiguous
        // in the vector (positions 0 and 2, skipping 1).
        let obs = vec![
            make_obs(0, 0, 59_000.5),
            make_obs(1, 1, 59_001.5), // belongs to a different night
            make_obs(2, 2, 59_000.6),
        ];
        let mut night_map: NightIndexMap = ahash::AHashMap::new();
        night_map.insert(NightId(59_000), ObsMapIndex::Split(vec![0, 2])); // non-contiguous
        night_map.insert(NightId(59_001), ObsMapIndex::Split(vec![1]));

        let ds = ObsDataset::new(obs, vec![], None, Some(night_map), None);
        let restored = roundtrip_contiguous(&ds);

        let night_map = restored.index.obs_index_by_night.as_ref().unwrap();
        match night_map.get(&NightId(59_000)).unwrap() {
            ObsMapIndex::Split(_) => {} // correct fallback
            ObsMapIndex::Contiguous { .. } => {
                panic!("non-contiguous night must not produce Contiguous entry")
            }
        }
        // Single-element group can always be Contiguous.
        match night_map.get(&NightId(59_001)).unwrap() {
            ObsMapIndex::Contiguous { .. } | ObsMapIndex::Split(_) => {}
        }
    }

    // ── Format version guard ─────────────────────────────────────────────

    #[test]
    fn format_version_mismatch_is_rejected() {
        // Craft a payload with format_version = 999 (unknown).
        let bad = r#"{
            "format_version": 999,
            "observations": [],
            "observer_dataset": {"custom_observers": [], "mpc_error_model": null},
            "traj_aliases": []
        }"#;
        let result = serde_json::from_str::<ObsDataset>(bad);
        assert!(result.is_err(), "unknown format_version must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("format_version") || msg.contains("unsupported"),
            "error must mention format_version, got: {msg}"
        );
    }
}
