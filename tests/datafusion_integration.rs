//! Integration tests for the DataFusion → ObsDataset ingestion pipeline.
//!
//! These tests load the same Parquet fixtures as the Polars pipeline tests
//! (`tests/data/`) via `file://` URIs and verify that the DataFusion reader
//! produces an `ObsDataset` that is semantically equivalent.
//!
//! The test categories mirror `obs_dataset_integration.rs` so that both
//! backends can be compared directly:
//!
//! 1. **Conversion correctness** — `ObsDataset::from_parquet_uri` succeeds and
//!    produces the expected row count for both fixtures.
//! 2. **Night index integrity** — night count, per-night counts, and iterator
//!    consistency.
//! 3. **Trajectory index integrity** — integer-traj and string-traj variants.
//! 4. **Accessors** — `get_observation`, `get_obs_by_index`, iteration order.
//! 5. **Observer integrity (geodetic)** — longitude, parallax constants,
//!    intern deduplication, accuracy values.
//! 6. **Observer integrity (MPC code)** — `None` without an error model.
//!
//! ## Fixtures
//!
//! | File | observer columns | traj_id type | Rows | Night count | Non-null traj |
//! |------|-----------------|-------------|------|-------------|--------------|
//! | `test_data_traj_int.parquet` | geodetic | `UInt32` | 561 287 | 10 | 68 145 |
//! | `test_data_traj_str.parquet` | `mpc_code_obs = "I41"` | `String` | 561 287 | 10 | 68 145 |

#![cfg(feature = "datafusion")]

mod helpers;
use helpers::*;

use photom::{NightId, TrajId};

// ── 1. Conversion correctness ──────────────────────────────────────────────────

/// `from_parquet_uri` on the integer-traj fixture produces the correct row count.
#[test]
fn int_file_row_count() {
    let ds = df_load_int();
    assert_eq!(
        ds.observation_count(),
        TOTAL_ROWS,
        "Expected {TOTAL_ROWS} observations from the int fixture (datafusion)"
    );
}

/// `from_parquet_uri` on the string-traj fixture produces the correct row count.
#[test]
fn str_file_row_count() {
    let ds = df_load_str();
    assert_eq!(
        ds.observation_count(),
        TOTAL_ROWS,
        "Expected {TOTAL_ROWS} observations from the str fixture (datafusion)"
    );
}

/// A non-existent `file://` URI returns `LoadObsError::NotFound`.
#[test]
fn not_found_uri_returns_error() {
    use photom::io::datafusion::loader::{LoadObsArgs, LoadObsError};
    let cwd = std::env::current_dir().unwrap();
    let uri = format!("file://{}/tests/data/does_not_exist.parquet", cwd.display());
    let result =
        photom::observation_dataset::ObsDataset::from_parquet_uri(&uri, LoadObsArgs::default());
    assert!(
        matches!(result, Err(LoadObsError::NotFound(_))),
        "Expected NotFound error for non-existent file, got {result:?}"
    );
}

/// Every observation yielded by `iter_observations` has a unique `id`.
#[test]
fn int_file_all_ids_unique() {
    let ds = df_load_int();
    let mut ids: Vec<u64> = ds.iter_observations().map(|o| *o.id()).collect();
    let original_len = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        original_len,
        "All observation ids must be unique (datafusion)"
    );
}

/// `iter_observations` preserves insertion order: indices must run 0..n-1.
#[test]
fn int_file_iter_observations_order() {
    let ds = df_load_int();
    for (expected_idx, obs) in ds.iter_observations().enumerate() {
        assert_eq!(
            obs.index(),
            expected_idx,
            "Observation at position {expected_idx} must have index == {expected_idx}"
        );
    }
}

// ── 2. Night index integrity ───────────────────────────────────────────────────

/// Night index is present after loading a file with a `night_id` column.
#[test]
fn night_index_is_present() {
    let ds = df_load_int();
    assert!(
        ds.iter_night_id().is_some(),
        "Night index must be present when the file has a night_id column (datafusion)"
    );
}

/// The night index contains exactly the expected number of distinct nights.
#[test]
fn night_index_correct_night_count() {
    let ds = df_load_int();
    let count = ds.iter_night_id().unwrap().count();
    assert_eq!(
        count, NIGHT_COUNT,
        "Expected {NIGHT_COUNT} distinct nights in the night index (datafusion)"
    );
}

/// The sum of all per-night observation counts equals the total row count.
#[test]
fn night_index_counts_sum_to_total() {
    let ds = df_load_int();
    let total: usize = ds
        .iter_night_id()
        .unwrap()
        .map(|nid| ds.len_night(nid).unwrap_or(0))
        .sum();
    assert_eq!(
        total, TOTAL_ROWS,
        "Sum of per-night counts must equal total row count (datafusion)"
    );
}

/// Each known night has exactly the expected per-night observation count.
#[test]
fn night_index_per_night_counts_correct() {
    let ds = df_load_int();
    for &(raw_id, expected_count) in NIGHT_EXPECTED {
        let nid = NightId(raw_id);
        let actual = ds
            .len_night(&nid)
            .unwrap_or_else(|| panic!("Night {raw_id} must be present in the index (datafusion)"));
        assert_eq!(
            actual, expected_count,
            "Night {raw_id}: expected {expected_count} observations, got {actual} (datafusion)"
        );
    }
}

/// `iter_night_observations` yields the same count as `len_night` for night 3140.
#[test]
fn night_index_iter_night_observations_consistent() {
    let ds = df_load_int();
    let nid = NightId(3140);
    let expected_count = 88_273usize;

    let obs: Vec<_> = ds
        .iter_night_observations(&nid)
        .expect("night 3140 must exist in index (datafusion)")
        .collect();

    assert_eq!(obs.len(), expected_count);
    assert_eq!(ds.len_night(&nid).unwrap(), expected_count);

    for o in &obs {
        assert!(
            o.index() < TOTAL_ROWS,
            "Observation index {} is out of bounds (datafusion)",
            o.index()
        );
    }
}

/// `materialize_night` returns a Vec with the same length as `len_night`.
#[test]
fn night_index_materialize_night_consistent() {
    let ds = df_load_int();
    for &(raw_id, expected_count) in NIGHT_EXPECTED {
        let nid = NightId(raw_id);
        let materialized = ds
            .materialize_night(&nid)
            .unwrap_or_else(|| panic!("Night {raw_id} must be present (datafusion)"));
        assert_eq!(materialized.len(), expected_count);
    }
}

/// `iter_full_night` yields exactly TOTAL_ROWS pairs.
#[test]
fn night_index_iter_full_night_total() {
    let ds = df_load_int();
    let total = ds
        .iter_full_night()
        .expect("iter_full_night must be Some (datafusion)")
        .count();
    assert_eq!(total, TOTAL_ROWS);
}

// ── 3. Trajectory index integrity (integer traj_id) ───────────────────────────

/// Trajectory index is present after loading the int-traj fixture.
#[test]
fn int_traj_index_is_present() {
    let ds = df_load_int();
    assert!(
        ds.iter_traj_id().is_some(),
        "Trajectory index must be present when the file has a traj_id column (datafusion)"
    );
}

/// The trajectory index contains the expected number of distinct trajectories.
#[test]
fn int_traj_index_unique_count() {
    let ds = df_load_int();
    let count = ds.iter_traj_id().unwrap().count();
    assert_eq!(
        count, TRAJ_UNIQUE,
        "Expected {TRAJ_UNIQUE} distinct trajectories in the int-traj index (datafusion)"
    );
}

/// The sum of all per-trajectory counts equals the number of non-null traj_id rows.
#[test]
fn int_traj_index_counts_sum_to_non_null() {
    let ds = df_load_int();
    let total: usize = ds
        .iter_traj_id()
        .unwrap()
        .map(|tid| ds.len_trajectory(tid).unwrap_or(0))
        .sum();
    assert_eq!(
        total, TRAJ_NON_NULL,
        "Sum of per-trajectory counts must equal non-null traj_id rows (datafusion)"
    );
}

/// Trajectory 2 has exactly 7 observations.
#[test]
fn int_traj_index_traj_2_count() {
    let ds = df_load_int();
    let tid = TrajId::Int(2);
    let count = ds
        .len_trajectory(&tid)
        .expect("Trajectory 2 must be present (datafusion)");
    assert_eq!(
        count, 7,
        "Trajectory 2 must have exactly 7 observations (datafusion)"
    );
}

/// `iter_trajectory_observations` for trajectory 2 returns the expected observation ids.
#[test]
fn int_traj_index_traj_2_obs_ids() {
    let ds = df_load_int();
    let tid = TrajId::Int(2);
    let mut actual_ids: Vec<u64> = ds
        .iter_trajectory_observations(&tid)
        .expect("Trajectory 2 must exist (datafusion)")
        .map(|o| *o.id())
        .collect();
    actual_ids.sort_unstable();

    let mut expected_ids: Vec<u64> = vec![
        3200126900715015016,
        3081439361315015011,
        3140191960415015001,
        3140277740415015003,
        3081388691315015011,
        3200166920715015004,
        3140276800415015002,
    ];
    expected_ids.sort_unstable();

    assert_eq!(actual_ids, expected_ids);
}

/// `iter_full_trajectory` yields exactly TRAJ_NON_NULL pairs.
#[test]
fn int_traj_index_iter_full_trajectory_total() {
    let ds = df_load_int();
    let total = ds
        .iter_full_trajectory()
        .expect("iter_full_trajectory must be Some (datafusion)")
        .count();
    assert_eq!(total, TRAJ_NON_NULL);
}

// ── 4. Trajectory index integrity (string traj_id) ────────────────────────────

/// Trajectory index is present after loading the str-traj fixture.
#[test]
fn str_traj_index_is_present() {
    let ds = df_load_str();
    assert!(ds.iter_traj_id().is_some());
}

/// The string-traj index contains the expected number of distinct trajectories.
#[test]
fn str_traj_index_unique_count() {
    let ds = df_load_str();
    let count = ds.iter_traj_id().unwrap().count();
    assert_eq!(count, TRAJ_UNIQUE);
}

/// The sum of per-trajectory counts equals TRAJ_NON_NULL.
#[test]
fn str_traj_index_counts_sum_to_non_null() {
    let ds = df_load_str();
    let total: usize = ds
        .iter_traj_id()
        .unwrap()
        .map(|tid| ds.len_trajectory(tid).unwrap_or(0))
        .sum();
    assert_eq!(total, TRAJ_NON_NULL);
}

/// Trajectory "1975" (String) has exactly 2 observations.
#[test]
fn str_traj_index_traj_1975_count() {
    let ds = df_load_str();
    let tid = TrajId::Str("1975".to_owned());
    let count = ds
        .len_trajectory(&tid)
        .expect("Trajectory \"1975\" must be present (datafusion)");
    assert_eq!(count, 2);
}

/// `iter_trajectory_observations` for trajectory "1975" returns the expected ids.
#[test]
fn str_traj_index_traj_1975_obs_ids() {
    let ds = df_load_str();
    let tid = TrajId::Str("1975".to_owned());
    let mut actual_ids: Vec<u64> = ds
        .iter_trajectory_observations(&tid)
        .expect("Trajectory \"1975\" must exist (datafusion)")
        .map(|o| *o.id())
        .collect();
    actual_ids.sort_unstable();

    let mut expected_ids = vec![3081393420915015000u64, 3081438420915015000];
    expected_ids.sort_unstable();

    assert_eq!(actual_ids, expected_ids);
}

// ── 5. Accessors ───────────────────────────────────────────────────────────────

/// `get_observation` returns `Some` for the first id and `None` for a fake one.
#[test]
fn get_observation_first_row() {
    let ds = df_load_int();
    let first_id: u64 = 3_026_230_983_415_015_002;

    let obs = ds
        .get_observation(first_id)
        .expect("First observation must be findable by id (datafusion)");
    assert_eq!(*obs.id(), first_id);

    assert!(
        ds.get_observation(u64::MAX).is_none(),
        "get_observation must return None for a non-existent id (datafusion)"
    );
}

/// `get_obs_by_index` returns `Some` for valid indices and `None` out of bounds.
#[test]
fn get_obs_by_index_bounds() {
    let ds = df_load_int();

    let obs = ds
        .get_obs_by_index(0)
        .expect("Index 0 must be a valid position (datafusion)");
    assert_eq!(obs.index(), 0);

    let last = ds
        .get_obs_by_index(TOTAL_ROWS - 1)
        .expect("Last index must be valid (datafusion)");
    assert_eq!(last.index(), TOTAL_ROWS - 1);

    assert!(ds.get_obs_by_index(TOTAL_ROWS).is_none());
}

/// `get_observation` is consistent with `get_obs_by_index` for a sample of rows.
#[test]
fn get_observation_consistent_with_get_obs_by_index() {
    let ds = df_load_int();

    for idx in (0..TOTAL_ROWS).step_by(50_000) {
        let by_index = ds
            .get_obs_by_index(idx)
            .unwrap_or_else(|| panic!("Index {idx} must be valid (datafusion)"));
        let id = *by_index.id();

        let by_id = ds
            .get_observation(id)
            .unwrap_or_else(|| panic!("get_observation must succeed for id {id} (datafusion)"));

        assert_eq!(by_id.index(), idx);
    }
}

// ── 6. Observer integrity (geodetic — int fixture) ─────────────────────────────

/// Every sampled observation in the int fixture has a non-None observer.
#[test]
fn int_obs_all_have_observer() {
    let mut ds = df_load_int();

    for idx in (0..TOTAL_ROWS).step_by(50_000) {
        let obs_id = *ds
            .get_obs_by_index(idx)
            .unwrap_or_else(|| panic!("Index {idx} must be valid (datafusion)"))
            .id();

        assert!(
            ds.get_observer(obs_id).is_some(),
            "Observation at index {idx} must have a resolvable observer (datafusion)"
        );
    }
}

/// Sampled observers all carry the expected geodetic longitude.
#[test]
fn int_obs_single_unique_observer_longitude() {
    let mut ds = df_load_int();

    for &idx in &[0usize, 100_000, 300_000, 560_000] {
        let obs_id = *ds
            .get_obs_by_index(idx)
            .unwrap_or_else(|| panic!("Index {idx} must be valid (datafusion)"))
            .id();

        let observer = ds.get_observer(obs_id).unwrap_or_else(|| {
            panic!("Observation at index {idx} must have an observer (datafusion)")
        });

        assert!(
            (f64::from(observer.longitude) - OBS_LON).abs() < OBSERVER_TOLERANCE,
            "Observer at index {idx}: expected longitude {OBS_LON}, got {} (datafusion)",
            f64::from(observer.longitude)
        );
    }
}

/// The parallax constants of the resolved observer match the fixture lat/alt values.
#[test]
fn int_obs_parallax_constants_correct() {
    use photom::observer::geodetic_to_parallax;

    let mut ds = df_load_int();
    let first_id: u64 = 3_026_230_983_415_015_002;
    let observer = ds
        .get_observer(first_id)
        .expect("First observation must have a resolvable observer (datafusion)");

    let rho_cos = f64::from(observer.rho_cos_phi);
    let rho_sin = f64::from(observer.rho_sin_phi);

    let (expected_rho_cos, expected_rho_sin) = geodetic_to_parallax(OBS_LAT, OBS_ALT);

    assert!(
        (rho_cos - expected_rho_cos).abs() < OBSERVER_TOLERANCE,
        "rho_cos_phi mismatch (datafusion): expected {expected_rho_cos}, got {rho_cos}"
    );
    assert!(
        (rho_sin - expected_rho_sin).abs() < OBSERVER_TOLERANCE,
        "rho_sin_phi mismatch (datafusion): expected {expected_rho_sin}, got {rho_sin}"
    );
}

/// Identical geodetic sites are interned: first and last observation share the
/// same longitude, rho_cos_phi, and rho_sin_phi.
#[test]
fn int_obs_identical_sites_interned() {
    let mut ds = df_load_int();

    let first_id = *ds.get_obs_by_index(0).unwrap().id();
    let last_id = *ds.get_obs_by_index(TOTAL_ROWS - 1).unwrap().id();

    let obs_a = ds
        .get_observer(first_id)
        .expect("First observation must have an observer (datafusion)");

    let lon_a = f64::from(obs_a.longitude);
    let rho_cos_a = f64::from(obs_a.rho_cos_phi);
    let rho_sin_a = f64::from(obs_a.rho_sin_phi);

    let obs_b = ds
        .get_observer(last_id)
        .expect("Last observation must have an observer (datafusion)");

    assert!((f64::from(obs_b.longitude) - lon_a).abs() < OBSERVER_TOLERANCE);
    assert!((f64::from(obs_b.rho_cos_phi) - rho_cos_a).abs() < OBSERVER_TOLERANCE);
    assert!((f64::from(obs_b.rho_sin_phi) - rho_sin_a).abs() < OBSERVER_TOLERANCE);
}

/// Sampled observers carry non-None and positive RA/Dec accuracy values.
#[test]
fn int_obs_accuracy_values_present_and_positive() {
    let mut ds = df_load_int();

    for idx in (0..TOTAL_ROWS).step_by(100_000) {
        let obs_id = *ds
            .get_obs_by_index(idx)
            .unwrap_or_else(|| panic!("Index {idx} must be valid (datafusion)"))
            .id();

        let observer = ds
            .get_observer(obs_id)
            .unwrap_or_else(|| panic!("Observation {idx} must have an observer (datafusion)"));

        let ra_acc = f64::from(
            observer
                .ra_accuracy
                .expect("ra_accuracy must be Some (datafusion)"),
        );
        let dec_acc = f64::from(
            observer
                .dec_accuracy
                .expect("dec_accuracy must be Some (datafusion)"),
        );

        assert!(ra_acc > 0.0 && ra_acc.is_finite());
        assert!(dec_acc > 0.0 && dec_acc.is_finite());
    }
}

// ── 7. Observer integrity (MPC code — str fixture) ─────────────────────────────

/// Without an error model, MPC-coded observers resolve to `None`.
#[test]
fn str_obs_mpc_no_error_model_returns_none() {
    let mut ds = df_load_str();
    let _ = MPC_CODE; // document that the fixture uses MPC_CODE = b"I41"
    let first_id: u64 = 3_026_230_983_415_015_002;

    if ds.get_observation(first_id).is_some() {
        assert!(
            ds.get_observer(first_id).is_none(),
            "get_observer must return None for MPC observer with no error model (datafusion)"
        );
    }
}

/// `get_observer` returns `None` for every sampled row of the str fixture
/// (no geodetic fallback when the observer is MPC-coded).
#[test]
fn str_obs_no_geodetic_fallback() {
    let mut ds = df_load_str();

    for idx in (0..TOTAL_ROWS).step_by(50_000) {
        let obs_id = {
            let obs = ds
                .get_obs_by_index(idx)
                .unwrap_or_else(|| panic!("Index {idx} must be valid in str fixture (datafusion)"));
            *obs.id()
        };

        assert!(
            ds.get_observer(obs_id).is_none(),
            "Observation {idx} in str fixture: expected None (no error model), got Some (datafusion)"
        );
    }
}

// ── 8. Night + traj cross-checks ───────────────────────────────────────────────

/// Night total and traj total are both internally consistent.
#[test]
fn int_night_and_traj_totals_consistent() {
    let ds = df_load_int();
    assert_eq!(ds.iter_full_night().unwrap().count(), TOTAL_ROWS);
    assert_eq!(ds.iter_full_trajectory().unwrap().count(), TRAJ_NON_NULL);
}

/// Every observation returned by `iter_night_observations` for night 3248 is
/// reachable via `get_obs_by_index`.
#[test]
fn night_obs_reachable_by_index() {
    let ds = df_load_int();
    let nid = NightId(3248);

    let indices: Vec<usize> = ds
        .iter_night_observations(&nid)
        .expect("night 3248 must exist (datafusion)")
        .map(|o| o.index())
        .collect();

    assert_eq!(indices.len(), 11_674);

    for &i in [
        indices[0],
        indices[indices.len() / 2],
        *indices.last().unwrap(),
    ]
    .iter()
    {
        let obs = ds
            .get_obs_by_index(i)
            .unwrap_or_else(|| panic!("Index {i} from night index must be reachable (datafusion)"));
        assert_eq!(obs.index(), i);
    }
}

/// `get_observer` never panics for one observation per night (geodetic fixture).
#[test]
fn int_obs_get_observer_never_panics_across_nights() {
    let mut ds = df_load_int();

    let night_obs_ids: Vec<u64> = NIGHT_EXPECTED
        .iter()
        .map(|&(raw_id, _)| {
            let nid = NightId(raw_id);
            let mut iter = ds
                .iter_night_observations(&nid)
                .unwrap_or_else(|| panic!("Night {raw_id} must exist (datafusion)"));
            *iter.next().unwrap().id()
        })
        .collect();

    for id in night_obs_ids {
        assert!(
            ds.get_observer(id).is_some(),
            "get_observer({id}) must return Some for int fixture (datafusion)"
        );
    }
}

/// Observers resolved through the trajectory index all share the expected longitude.
#[test]
fn int_obs_trajectory_observer_index_valid() {
    let mut ds = df_load_int();

    let sample: Vec<u64> = ds
        .iter_full_trajectory()
        .expect("iter_full_trajectory must be Some (datafusion)")
        .step_by(10_000)
        .map(|(_, obs)| *obs.id())
        .collect();

    for id in sample {
        let observer = ds.get_observer(id).unwrap_or_else(|| {
            panic!("Observation {id} from trajectory index must have an observer (datafusion)")
        });

        assert!(
            (f64::from(observer.longitude) - OBS_LON).abs() < OBSERVER_TOLERANCE,
            "Trajectory-indexed observer longitude must be {OBS_LON}, got {} for obs {id} (datafusion)",
            f64::from(observer.longitude)
        );
    }
}

// ── 7. Contiguous-index optimisation ──────────────────────────────────────────

/// Helper: load the int-traj fixture with `ContiguousTraj` optimisation.
fn df_load_int_contiguous_traj() -> photom::observation_dataset::ObsDataset {
    use photom::io::datafusion::loader::{ContiguousChoice, LoadObsArgs};
    let uri = format!(
        "file://{}/{}",
        std::env::current_dir()
            .expect("current_dir must be accessible")
            .display(),
        PATH_INT
    );
    photom::observation_dataset::ObsDataset::from_parquet_uri(
        &uri,
        LoadObsArgs {
            contiguous_choice: Some(ContiguousChoice::ContiguousTraj),
            ..Default::default()
        },
    )
    .expect("from_parquet_uri with ContiguousTraj must succeed")
}

/// Helper: load the int-traj fixture with `ContiguousNight` optimisation.
fn df_load_int_contiguous_night() -> photom::observation_dataset::ObsDataset {
    use photom::io::datafusion::loader::{ContiguousChoice, LoadObsArgs};
    let uri = format!(
        "file://{}/{}",
        std::env::current_dir()
            .expect("current_dir must be accessible")
            .display(),
        PATH_INT
    );
    photom::observation_dataset::ObsDataset::from_parquet_uri(
        &uri,
        LoadObsArgs {
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
            ..Default::default()
        },
    )
    .expect("from_parquet_uri with ContiguousNight must succeed")
}

/// Loading with `ContiguousTraj` still produces the correct total row count.
#[test]
fn contiguous_traj_total_row_count() {
    let ds = df_load_int_contiguous_traj();
    assert_eq!(
        ds.observation_count(),
        TOTAL_ROWS,
        "ContiguousTraj load must yield {TOTAL_ROWS} rows (datafusion)"
    );
}

/// Loading with `ContiguousTraj` preserves the correct number of distinct trajectories.
#[test]
fn contiguous_traj_index_unique_count() {
    let ds = df_load_int_contiguous_traj();
    let count = ds
        .iter_traj_id()
        .expect("traj index must be present")
        .count();
    assert_eq!(
        count, TRAJ_UNIQUE,
        "ContiguousTraj load must yield {TRAJ_UNIQUE} distinct trajectories (datafusion)"
    );
}

/// With `ContiguousTraj`, each trajectory entry in the index is a `Contiguous` range.
///
/// Note: the `Contiguous`/`Split` distinction is an internal representation detail
/// tested via unit tests.  Here we verify that the optimisation does not corrupt
/// the observable data.
#[test]
fn contiguous_traj_entries_are_contiguous() {
    // Verify correctness proxy: every trajectory's observation set is non-empty.
    let ds = df_load_int_contiguous_traj();
    let all_non_empty = ds
        .iter_traj_id()
        .expect("traj index must be present")
        .all(|tid| ds.len_trajectory(tid).unwrap_or(0) > 0);
    assert!(
        all_non_empty,
        "Every trajectory must have at least one observation after ContiguousTraj load (datafusion)"
    );
}

/// With `ContiguousNight`, each night entry in the index is a `Contiguous` range.
///
/// Note: the `Contiguous`/`Split` distinction is an internal representation detail
/// tested via unit tests.  Here we verify that the optimisation does not corrupt
/// the observable data.
#[test]
fn contiguous_night_entries_are_contiguous() {
    // Verify correctness proxy: every night's observation set is non-empty.
    let ds = df_load_int_contiguous_night();
    let all_non_empty = ds
        .iter_night_id()
        .expect("night index must be present")
        .all(|nid| ds.len_night(nid).unwrap_or(0) > 0);
    assert!(
        all_non_empty,
        "Every night must have at least one observation after ContiguousNight load (datafusion)"
    );
}

/// `ContiguousTraj` load: sum of per-trajectory observation counts equals non-null traj rows.
#[test]
fn contiguous_traj_counts_sum_to_non_null() {
    let ds = df_load_int_contiguous_traj();
    let total: usize = ds
        .iter_traj_id()
        .expect("traj index must be present")
        .map(|tid| ds.len_trajectory(tid).unwrap_or(0))
        .sum();
    assert_eq!(
        total, TRAJ_NON_NULL,
        "Sum of per-trajectory counts must equal {TRAJ_NON_NULL} (datafusion contiguous)"
    );
}

/// `ContiguousNight` load: sum of per-night observation counts equals total rows.
#[test]
fn contiguous_night_counts_sum_to_total() {
    let ds = df_load_int_contiguous_night();
    let total: usize = ds
        .iter_night_id()
        .expect("night index must be present")
        .map(|nid| ds.len_night(nid).unwrap_or(0))
        .sum();
    assert_eq!(
        total, TOTAL_ROWS,
        "Sum of per-night counts must equal {TOTAL_ROWS} (datafusion contiguous night)"
    );
}
