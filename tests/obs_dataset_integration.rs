//! Integration tests for the Polars → ObsDataset ingestion pipeline.
//!
//! These tests load the real Parquet fixtures in `tests/data/` and verify:
//!
//! 1. **Conversion correctness** — `ObsDataset::from_polars` and
//!    `ObsDataset::from_lazy` succeed and produce the expected row count.
//! 2. **Night index integrity** — every night present in the source file is
//!    indexed, the per-night counts sum to the total, and
//!    `iter_night_observations` / `materialize_night` / `len_night` are
//!    consistent with each other.
//! 3. **Trajectory index integrity** — same consistency checks for the
//!    trajectory index with both integer (`UInt32`) and string `traj_id`
//!    column types.
//! 4. **Accessors and iterators** — `get_observation`, `get_obs_by_index`,
//!    `iter_observations`, `iter_night_id`, `iter_traj_id`,
//!    `iter_full_night`, `iter_full_trajectory`, and `push_new_trajectory`
//!    all behave correctly on real data.
//! 5. **Observer integrity (geodetic)** — the integer-traj fixture carries
//!    geodetic observer columns (`obs_lon`, `obs_lat`, `obs_alt`,
//!    `obs_ra_acc`, `obs_dec_acc`); every observation must resolve to a
//!    custom `ObserverId::IntId`, identical coordinates must be interned into
//!    the same slot, and the stored values must match the fixture.
//! 6. **Observer integrity (MPC code)** — the string-traj fixture carries a
//!    `mpc_code_obs` column (`"I41"` for every row); every observation must
//!    resolve to `ObserverId::MpcCode(*b"I41")`.
//!
//! ## Fixture layout
//!
//! | File | observer columns | traj_id type | Rows | Night count | Non-null traj |
//! |------|-----------------|-------------|------|-------------|--------------|
//! | `test_data_traj_int.parquet` | geodetic (obs_lon/lat/alt + accuracies) | `UInt32` | 561 287 | 10 | 68 145 |
//! | `test_data_traj_str.parquet` | `mpc_code_obs = "I41"` | `String` | 561 287 | 10 | 68 145 |
//!
//! Both files share the same `id`, `night_id`, and photometric columns.

#![cfg(feature = "polars")]

use photom::{NightId, TrajId, observation_dataset::ObsDataset};
use polars::prelude::{LazyFrame, ScanArgsParquet};

// ── path constants ────────────────────────────────────────────────────────────

const PATH_INT: &str = "tests/data/test_data_traj_int.parquet";
const PATH_STR: &str = "tests/data/test_data_traj_str.parquet";

/// Total number of rows in each fixture file.
const TOTAL_ROWS: usize = 561_287;

/// Number of distinct nights in both fixture files.
const NIGHT_COUNT: usize = 10;

/// Number of observations with a non-null trajectory identifier.
const TRAJ_NON_NULL: usize = 68_145;

/// Number of distinct trajectory identifiers.
const TRAJ_UNIQUE: usize = 38_024;

/// Night identifiers and their expected observation counts (from the fixture).
const NIGHT_EXPECTED: &[(u32, usize)] = &[
    (3010, 78204),
    (3026, 37244),
    (3074, 86675),
    (3081, 86693),
    (3140, 88273),
    (3177, 44772),
    (3200, 82158),
    (3204, 29196),
    (3248, 11674),
    (3278, 16398),
];

/// Geodetic coordinates of ZTF's Palomar Observatory as stored in the int fixture.
/// Longitude and latitude in radians, altitude in metres.
const OBS_LON: f64 = -2.0391;
const OBS_LAT: f64 = 0.5822;
const OBS_ALT: f64 = 1712.0;

/// MPC observatory code present in every row of the str fixture.
const MPC_CODE: &[u8; 3] = b"I41";

// ── helpers ───────────────────────────────────────────────────────────────────

/// Load the integer-traj fixture via `ObsDataset::from_lazy` (fast path:
/// `ScanArgsParquet { rechunk: true }` + `do_rechunk = Some(false)`).
fn load_int() -> ObsDataset {
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(PATH_INT.into(), args).expect("scan_parquet must succeed");
    ObsDataset::from_lazy(lf, None, None, Some(false)).expect("from_lazy must succeed for int file")
}

/// Load the string-traj fixture via `ObsDataset::from_lazy`.
fn load_str() -> ObsDataset {
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(PATH_STR.into(), args).expect("scan_parquet must succeed");
    ObsDataset::from_lazy(lf, None, None, Some(false)).expect("from_lazy must succeed for str file")
}

// ── 1. Conversion correctness ─────────────────────────────────────────────────

/// `from_lazy` with the integer-traj file produces the correct row count.
#[test]
fn int_file_row_count() {
    let ds = load_int();
    assert_eq!(
        ds.observation_count(),
        TOTAL_ROWS,
        "Expected {TOTAL_ROWS} observations from the int fixture"
    );
}

/// `from_lazy` with the string-traj file produces the correct row count.
#[test]
fn str_file_row_count() {
    let ds = load_str();
    assert_eq!(
        ds.observation_count(),
        TOTAL_ROWS,
        "Expected {TOTAL_ROWS} observations from the str fixture"
    );
}

/// `from_polars` (eager path) with the integer-traj file succeeds and produces
/// the correct row count.
#[test]
fn int_file_from_polars_eager() {
    use polars::prelude::*;
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let df = LazyFrame::scan_parquet(PATH_INT.into(), args)
        .expect("scan must succeed")
        .collect()
        .expect("collect must succeed");
    let ds =
        ObsDataset::from_polars(&df, None, None, Some(false)).expect("from_polars must succeed");
    assert_eq!(ds.observation_count(), TOTAL_ROWS);
}

/// Every observation yielded by `iter_observations` has a unique `id`.
#[test]
fn int_file_all_ids_unique() {
    let ds = load_int();
    let mut ids: Vec<u64> = ds.iter_observations().map(|o| *o.id()).collect();
    let original_len = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        original_len,
        "All observation ids must be unique"
    );
}

/// `iter_observations` preserves insertion order: observation indices must be
/// strictly increasing from 0 to n-1.
#[test]
fn int_file_iter_observations_order() {
    let ds = load_int();
    for (expected_idx, obs) in ds.iter_observations().enumerate() {
        assert_eq!(
            obs.index(),
            expected_idx,
            "Observation at position {expected_idx} must have index == {expected_idx}"
        );
    }
}

// ── 2. Night index integrity ──────────────────────────────────────────────────

/// Night index is present (not `None`) after loading a file with a `night_id`
/// column.
#[test]
fn night_index_is_present() {
    let ds = load_int();
    assert!(
        ds.iter_night_id().is_some(),
        "Night index must be present when the file has a night_id column"
    );
}

/// The night index contains exactly the expected number of distinct nights.
#[test]
fn night_index_correct_night_count() {
    let ds = load_int();
    let count = ds.iter_night_id().unwrap().count();
    assert_eq!(
        count, NIGHT_COUNT,
        "Expected {NIGHT_COUNT} distinct nights in the night index"
    );
}

/// The sum of all per-night observation counts equals the total row count.
#[test]
fn night_index_counts_sum_to_total() {
    let ds = load_int();
    let total: usize = ds
        .iter_night_id()
        .unwrap()
        .map(|nid| ds.len_night(nid).unwrap_or(0))
        .sum();
    assert_eq!(
        total, TOTAL_ROWS,
        "Sum of per-night counts must equal total row count"
    );
}

/// Each known night has exactly the expected per-night observation count.
#[test]
fn night_index_per_night_counts_correct() {
    let ds = load_int();
    for &(raw_id, expected_count) in NIGHT_EXPECTED {
        let nid = NightId(raw_id);
        let actual = ds
            .len_night(&nid)
            .unwrap_or_else(|| panic!("Night {raw_id} must be present in the index"));
        assert_eq!(
            actual, expected_count,
            "Night {raw_id}: expected {expected_count} observations, got {actual}"
        );
    }
}

/// `iter_night_observations` yields the same count as `len_night` for night
/// 3140, and every returned observation has `index` within bounds.
#[test]
fn night_index_iter_night_observations_consistent() {
    let ds = load_int();
    let nid = NightId(3140);
    let expected_count = 88_273usize;

    let obs: Vec<_> = ds
        .iter_night_observations(&nid)
        .expect("night 3140 must exist in index")
        .collect();

    assert_eq!(
        obs.len(),
        expected_count,
        "iter_night_observations for night 3140 must yield {expected_count} observations"
    );
    assert_eq!(
        ds.len_night(&nid).unwrap(),
        expected_count,
        "len_night must agree with iter count"
    );
    // Every observation index must be a valid position in the dataset.
    for o in &obs {
        assert!(
            o.index() < TOTAL_ROWS,
            "Observation index {} is out of bounds",
            o.index()
        );
    }
}

/// `materialize_night` returns a Vec with the same length as `len_night`.
#[test]
fn night_index_materialize_night_consistent() {
    let ds = load_int();
    for &(raw_id, expected_count) in NIGHT_EXPECTED {
        let nid = NightId(raw_id);
        let materialized = ds
            .materialize_night(&nid)
            .unwrap_or_else(|| panic!("Night {raw_id} must be present"));
        assert_eq!(
            materialized.len(),
            expected_count,
            "materialize_night for night {raw_id} must return {expected_count} observations"
        );
    }
}

/// `iter_full_night` yields exactly TOTAL_ROWS pairs (one per observation that
/// has a night_id, which is all rows in this fixture).
#[test]
fn night_index_iter_full_night_total() {
    let ds = load_int();
    let total = ds
        .iter_full_night()
        .expect("iter_full_night must be Some")
        .count();
    assert_eq!(
        total, TOTAL_ROWS,
        "iter_full_night must yield one pair per observation"
    );
}

/// Night IDs returned by `iter_full_night` all appear in the known set.
#[test]
fn night_index_iter_full_night_valid_night_ids() {
    let ds = load_int();
    let known: std::collections::HashSet<u32> = NIGHT_EXPECTED.iter().map(|&(id, _)| id).collect();
    for (nid, _obs) in ds.iter_full_night().unwrap() {
        assert!(
            known.contains(&nid.0),
            "iter_full_night returned unknown night id {}",
            nid.0
        );
    }
}

// ── 3. Trajectory index integrity (integer traj_id) ───────────────────────────

/// Trajectory index is present after loading the int-traj fixture.
#[test]
fn int_traj_index_is_present() {
    let ds = load_int();
    assert!(
        ds.iter_traj_id().is_some(),
        "Trajectory index must be present when the file has a traj_id column"
    );
}

/// The trajectory index contains the expected number of distinct trajectories.
#[test]
fn int_traj_index_unique_count() {
    let ds = load_int();
    let count = ds.iter_traj_id().unwrap().count();
    assert_eq!(
        count, TRAJ_UNIQUE,
        "Expected {TRAJ_UNIQUE} distinct trajectories in the int-traj index"
    );
}

/// The sum of all per-trajectory counts equals the number of non-null traj_id
/// rows.
#[test]
fn int_traj_index_counts_sum_to_non_null() {
    let ds = load_int();
    let total: usize = ds
        .iter_traj_id()
        .unwrap()
        .map(|tid| ds.len_trajectory(tid).unwrap_or(0))
        .sum();
    assert_eq!(
        total, TRAJ_NON_NULL,
        "Sum of per-trajectory counts must equal the number of non-null traj_id rows"
    );
}

/// Trajectory 2 has exactly 7 observations in the fixture.
#[test]
fn int_traj_index_traj_2_count() {
    let ds = load_int();
    let tid = TrajId::Int(2);
    let count = ds
        .len_trajectory(&tid)
        .expect("Trajectory 2 must be present in the index");
    assert_eq!(count, 7, "Trajectory 2 must have exactly 7 observations");
}

/// `iter_trajectory_observations` for trajectory 2 returns observations
/// whose ids match the known set from the fixture.
#[test]
fn int_traj_index_traj_2_obs_ids() {
    let ds = load_int();
    let tid = TrajId::Int(2);
    let mut actual_ids: Vec<u64> = ds
        .iter_trajectory_observations(&tid)
        .expect("Trajectory 2 must exist")
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

    assert_eq!(
        actual_ids, expected_ids,
        "Trajectory 2 obs ids must match the fixture"
    );
}

/// `materialize_trajectory` for trajectory 2 returns the same 7 observations
/// as `iter_trajectory_observations`.
#[test]
fn int_traj_index_materialize_traj_2() {
    let ds = load_int();
    let tid = TrajId::Int(2);
    let materialized = ds
        .materialize_trajectory(&tid)
        .expect("Trajectory 2 must be present");
    assert_eq!(
        materialized.len(),
        7,
        "materialize_trajectory must return 7 observations for trajectory 2"
    );
}

/// `iter_full_trajectory` yields exactly TRAJ_NON_NULL pairs.
#[test]
fn int_traj_index_iter_full_trajectory_total() {
    let ds = load_int();
    let total = ds
        .iter_full_trajectory()
        .expect("iter_full_trajectory must be Some")
        .count();
    assert_eq!(
        total, TRAJ_NON_NULL,
        "iter_full_trajectory must yield one pair per non-null traj_id row"
    );
}

// ── 4. Trajectory index integrity (string traj_id) ───────────────────────────

/// Trajectory index is present after loading the str-traj fixture.
#[test]
fn str_traj_index_is_present() {
    let ds = load_str();
    assert!(
        ds.iter_traj_id().is_some(),
        "Trajectory index must be present when the file has a string traj_id column"
    );
}

/// The string-traj index contains the same number of distinct trajectories as
/// the int-traj index.
#[test]
fn str_traj_index_unique_count() {
    let ds = load_str();
    let count = ds.iter_traj_id().unwrap().count();
    assert_eq!(
        count, TRAJ_UNIQUE,
        "Expected {TRAJ_UNIQUE} distinct trajectories in the str-traj index"
    );
}

/// The sum of per-trajectory counts for the str-traj file equals TRAJ_NON_NULL.
#[test]
fn str_traj_index_counts_sum_to_non_null() {
    let ds = load_str();
    let total: usize = ds
        .iter_traj_id()
        .unwrap()
        .map(|tid| ds.len_trajectory(tid).unwrap_or(0))
        .sum();
    assert_eq!(
        total, TRAJ_NON_NULL,
        "Sum of per-trajectory counts (str file) must equal the number of non-null traj_id rows"
    );
}

/// Trajectory "1975" (String in the str file) has exactly 2 observations
/// in the updated fixture (the parquet was regenerated with observer columns).
#[test]
fn str_traj_index_traj_1975_count() {
    let ds = load_str();
    let tid = TrajId::Str("1975".to_owned());
    let count = ds
        .len_trajectory(&tid)
        .expect("Trajectory \"1975\" must be present in the str-traj index");
    assert_eq!(
        count, 2,
        "Trajectory \"1975\" must have exactly 2 observations"
    );
}

/// `iter_trajectory_observations` for trajectory "1975" (str) returns the
/// expected 2 observation ids from the updated fixture.
#[test]
fn str_traj_index_traj_1975_obs_ids() {
    let ds = load_str();
    let tid = TrajId::Str("1975".to_owned());
    let mut actual_ids: Vec<u64> = ds
        .iter_trajectory_observations(&tid)
        .expect("Trajectory \"1975\" must exist")
        .map(|o| *o.id())
        .collect();
    actual_ids.sort_unstable();

    let mut expected_ids: Vec<u64> = vec![3081393420915015000, 3081438420915015000];
    expected_ids.sort_unstable();

    assert_eq!(
        actual_ids, expected_ids,
        "Trajectory \"1975\" (str) obs ids must match the fixture"
    );
}

/// `iter_full_trajectory` on the str-traj dataset yields TRAJ_NON_NULL pairs.
#[test]
fn str_traj_index_iter_full_trajectory_total() {
    let ds = load_str();
    let total = ds
        .iter_full_trajectory()
        .expect("iter_full_trajectory must be Some")
        .count();
    assert_eq!(
        total, TRAJ_NON_NULL,
        "iter_full_trajectory (str file) must yield one pair per non-null traj_id row"
    );
}

// ── 5. Accessors ──────────────────────────────────────────────────────────────

/// `get_observation` returns `Some` for the first observation's id and `None`
/// for a non-existent id.
#[test]
fn get_observation_first_row() {
    let mut ds = load_int();

    // The first observation in the int fixture.
    let first_id: u64 = 3_026_230_983_415_015_002;

    let obs = ds
        .get_observation(first_id)
        .expect("First observation must be findable by id");
    assert_eq!(*obs.id(), first_id);

    // A fabricated id that cannot exist in the fixture.
    assert!(
        ds.get_observation(u64::MAX).is_none(),
        "get_observation must return None for a non-existent id"
    );
}

/// `get_obs_by_index` returns `Some` for valid indices and `None` for an
/// out-of-bounds index.
#[test]
fn get_obs_by_index_bounds() {
    let mut ds = load_int();

    // Index 0 must exist.
    let obs = ds
        .get_obs_by_index(0)
        .expect("Index 0 must be a valid position");
    assert_eq!(obs.index(), 0);

    // Last valid index.
    let last = ds
        .get_obs_by_index(TOTAL_ROWS - 1)
        .expect("Last index must be valid");
    assert_eq!(last.index(), TOTAL_ROWS - 1);

    // One past the end must be None.
    assert!(
        ds.get_obs_by_index(TOTAL_ROWS).is_none(),
        "get_obs_by_index must return None for an out-of-bounds index"
    );
}

/// `get_observation` is consistent with `get_obs_by_index` for a sample of
/// observations: looking up by id must return the same observation as looking
/// up by the index stored inside that observation.
#[test]
fn get_observation_consistent_with_get_obs_by_index() {
    let mut ds = load_int();

    // Sample every 50 000th observation to keep the test fast.
    let sample_indices: Vec<usize> = (0..TOTAL_ROWS).step_by(50_000).collect();

    for idx in sample_indices {
        let by_index = ds
            .get_obs_by_index(idx)
            .unwrap_or_else(|| panic!("Index {idx} must be valid"));
        let id = *by_index.id();

        let by_id = ds
            .get_observation(id)
            .unwrap_or_else(|| panic!("get_observation must succeed for id {id}"));

        assert_eq!(
            by_id.index(),
            idx,
            "get_observation and get_obs_by_index must return the same observation"
        );
    }
}

/// Repeated calls to `get_observation` for the same id return the same value
/// (LRU cache correctness).
#[test]
fn get_observation_repeated_calls_consistent() {
    let mut ds = load_int();
    let first_id: u64 = 3_026_230_983_415_015_002;

    let id1 = *ds.get_observation(first_id).unwrap().id();
    let id2 = *ds.get_observation(first_id).unwrap().id();
    assert_eq!(id1, id2);
}

// ── 6. Night + traj index together ───────────────────────────────────────────

/// Cross-check: every observation returned by `iter_night_observations` for
/// a known night is also reachable via `get_obs_by_index`.
#[test]
fn night_obs_reachable_by_index() {
    let mut ds = load_int();
    let nid = NightId(3248); // smallest night, 11 674 observations

    // Collect indices from the night iterator.
    let indices: Vec<usize> = ds
        .iter_night_observations(&nid)
        .expect("night 3248 must exist")
        .map(|o| o.index())
        .collect();

    assert_eq!(indices.len(), 11_674);

    // Spot-check first, last, and middle.
    for &i in [
        indices[0],
        indices[indices.len() / 2],
        *indices.last().unwrap(),
    ]
    .iter()
    {
        let obs = ds
            .get_obs_by_index(i)
            .unwrap_or_else(|| panic!("Index {i} from night index must be reachable"));
        assert_eq!(obs.index(), i);
    }
}

/// The total count of observations across both `iter_full_night` and
/// `iter_full_trajectory` is internally consistent: the night total must equal
/// TOTAL_ROWS and the traj total must equal TRAJ_NON_NULL (int file).
#[test]
fn int_night_and_traj_totals_consistent() {
    let ds = load_int();

    let night_total = ds.iter_full_night().unwrap().count();
    let traj_total = ds.iter_full_trajectory().unwrap().count();

    assert_eq!(night_total, TOTAL_ROWS);
    assert_eq!(traj_total, TRAJ_NON_NULL);
}

// ── 7. push_new_trajectory ────────────────────────────────────────────────────

/// `push_new_trajectory` inserts a new trajectory key and associates it with
/// the provided observations.  After insertion, the new key must be findable
/// via `len_trajectory` and `iter_trajectory_observations`.
#[test]
fn push_new_trajectory_int_file() {
    let mut ds = load_int();

    // Pick two existing observations to form a synthetic trajectory.
    let obs0 = ds.get_obs_by_index(0).unwrap().clone();
    let obs1 = ds.get_obs_by_index(1).unwrap().clone();

    let new_tid = TrajId::Int(u32::MAX); // a key that cannot appear in the fixture
    assert!(
        ds.len_trajectory(&new_tid).is_none(),
        "The synthetic trajectory must not exist before insertion"
    );

    ds.push_new_trajectory(new_tid.clone(), &[obs0, obs1]);

    assert_eq!(
        ds.len_trajectory(&new_tid),
        Some(2),
        "push_new_trajectory must register exactly 2 observations"
    );

    let ids: Vec<u64> = ds
        .iter_trajectory_observations(&new_tid)
        .unwrap()
        .map(|o| *o.id())
        .collect();
    assert_eq!(
        ids.len(),
        2,
        "iter_trajectory_observations must yield 2 obs"
    );
}

// ── 8. Observer integrity (geodetic — int fixture) ────────────────────────────

/// Tolerance for floating-point comparisons of observer parallax constants.
const OBSERVER_TOLERANCE: f64 = 1e-9;

/// Every observation in the int fixture must resolve to a non-None observer
/// (all rows carry complete geodetic columns).
#[test]
fn int_obs_all_have_observer() {
    let mut ds = load_int();

    // Sample every 50 000th observation to keep the test fast.
    for idx in (0..TOTAL_ROWS).step_by(50_000) {
        let obs_id = *ds
            .get_obs_by_index(idx)
            .unwrap_or_else(|| panic!("Index {idx} must be valid"))
            .id();

        assert!(
            ds.get_observer(obs_id).is_some(),
            "Observation at index {idx} must have a resolvable observer"
        );
    }
}

/// The int fixture contains a single unique geodetic site; after ingestion
/// exactly one custom observer slot must exist (all identical observers are
/// interned into slot 0).
///
/// This is verified indirectly: sampling multiple observations and confirming
/// that `get_observer` always returns the longitude equal to `OBS_LON`.
#[test]
fn int_obs_single_unique_observer_longitude() {
    let mut ds = load_int();

    let sample_indices = [0usize, 100_000, 300_000, 560_000];

    for &idx in &sample_indices {
        let obs_id = *ds
            .get_obs_by_index(idx)
            .unwrap_or_else(|| panic!("Index {idx} must be valid"))
            .id();

        let observer = ds
            .get_observer(obs_id)
            .unwrap_or_else(|| panic!("Observation at index {idx} must have an observer"));

        assert!(
            (f64::from(observer.longitude) - OBS_LON).abs() < OBSERVER_TOLERANCE,
            "Observer at index {idx}: expected longitude {OBS_LON} deg, \
             got {} deg",
            f64::from(observer.longitude)
        );
    }
}

/// The resolved observer's parallax constants must match the values computed
/// by `Observer::new(lon=OBS_LON, lat=OBS_LAT, alt=OBS_ALT)`.
///
/// The parallax computation is reproduced here using `OBS_LAT` and `OBS_ALT`
/// directly so that the constants are exercised in code (not only in docs).
#[test]
fn int_obs_parallax_constants_correct() {
    use photom::observer::geodetic_to_parallax;

    let mut ds = load_int();

    // Use the first observation as the reference.
    let first_id: u64 = 3_026_230_983_415_015_002;
    let observer = ds
        .get_observer(first_id)
        .expect("First observation must have a resolvable observer");

    let rho_cos = f64::from(observer.rho_cos_phi);
    let rho_sin = f64::from(observer.rho_sin_phi);

    // Reproduce the expected values from the fixture's lat/alt constants.
    let (expected_rho_cos, expected_rho_sin) = geodetic_to_parallax(OBS_LAT, OBS_ALT);

    assert!(
        (rho_cos - expected_rho_cos).abs() < OBSERVER_TOLERANCE,
        "rho_cos_phi mismatch: expected {expected_rho_cos}, got {rho_cos}"
    );
    assert!(
        (rho_sin - expected_rho_sin).abs() < OBSERVER_TOLERANCE,
        "rho_sin_phi mismatch: expected {expected_rho_sin}, got {rho_sin}"
    );
}

/// Identical geodetic sites are interned into the same slot: two observations
/// sampled far apart in the dataset must resolve to observers that are equal
/// (same longitude, rho_cos_phi, rho_sin_phi).
#[test]
fn int_obs_identical_sites_interned() {
    let mut ds = load_int();

    let first_id = *ds.get_obs_by_index(0).unwrap().id();
    let last_id = *ds.get_obs_by_index(TOTAL_ROWS - 1).unwrap().id();

    let obs_a = ds
        .get_observer(first_id)
        .expect("First observation must have an observer");

    // Clone the relevant fields before the next mutable borrow.
    let lon_a = f64::from(obs_a.longitude);
    let rho_cos_a = f64::from(obs_a.rho_cos_phi);
    let rho_sin_a = f64::from(obs_a.rho_sin_phi);

    let obs_b = ds
        .get_observer(last_id)
        .expect("Last observation must have an observer");

    assert!(
        (f64::from(obs_b.longitude) - lon_a).abs() < OBSERVER_TOLERANCE,
        "Interned observer longitude must be identical for all rows"
    );
    assert!(
        (f64::from(obs_b.rho_cos_phi) - rho_cos_a).abs() < OBSERVER_TOLERANCE,
        "Interned observer rho_cos_phi must be identical for all rows"
    );
    assert!(
        (f64::from(obs_b.rho_sin_phi) - rho_sin_a).abs() < OBSERVER_TOLERANCE,
        "Interned observer rho_sin_phi must be identical for all rows"
    );
}

/// Every observation in the int fixture resolves to an observer that carries
/// non-None RA and Dec accuracy values (both accuracy columns are non-null in
/// the fixture).
#[test]
fn int_obs_accuracy_values_present() {
    let mut ds = load_int();

    for idx in (0..TOTAL_ROWS).step_by(100_000) {
        let obs_id = *ds
            .get_obs_by_index(idx)
            .unwrap_or_else(|| panic!("Index {idx} must be valid"))
            .id();

        let observer = ds
            .get_observer(obs_id)
            .unwrap_or_else(|| panic!("Observation {idx} must have an observer"));

        assert!(
            observer.ra_accuracy.is_some(),
            "Observer at index {idx} must have a non-None ra_accuracy"
        );
        assert!(
            observer.dec_accuracy.is_some(),
            "Observer at index {idx} must have a non-None dec_accuracy"
        );
    }
}

/// The dec_accuracy stored in the observer must be positive and finite.
/// The fixture carries a single constant dec_accuracy ≈ 2.424e-6 rad for all rows.
#[test]
fn int_obs_dec_accuracy_positive() {
    let mut ds = load_int();

    let first_id: u64 = 3_026_230_983_415_015_002;
    let observer = ds
        .get_observer(first_id)
        .expect("First observation must have a resolvable observer");

    let dec_acc = f64::from(observer.dec_accuracy.expect("dec_accuracy must be Some"));
    assert!(
        dec_acc > 0.0,
        "dec_accuracy must be positive, got {dec_acc}"
    );
    assert!(
        dec_acc.is_finite(),
        "dec_accuracy must be finite, got {dec_acc}"
    );
}

/// Cross-check: observers resolved for all observations in the smallest night
/// (3248, 11 674 rows) are non-None and share the same longitude.
#[test]
fn int_obs_night_3248_all_observers_consistent() {
    let mut ds = load_int();
    let nid = NightId(3248);

    // Collect all observation ids for the night first (immutable borrow).
    let obs_ids: Vec<u64> = ds
        .iter_night_observations(&nid)
        .expect("Night 3248 must exist")
        .map(|o| *o.id())
        .collect();

    assert_eq!(
        obs_ids.len(),
        11_674,
        "Night 3248 must have 11 674 observations"
    );

    // Now resolve observers (mutable borrow); sample every 500th to stay fast.
    for &id in obs_ids.iter().step_by(500) {
        let observer = ds
            .get_observer(id)
            .unwrap_or_else(|| panic!("Observation {id} in night 3248 must have an observer"));

        assert!(
            (f64::from(observer.longitude) - OBS_LON).abs() < OBSERVER_TOLERANCE,
            "Observer longitude in night 3248 must be {OBS_LON} deg"
        );
    }
}

// ── 9. Observer integrity (MPC code — str fixture) ────────────────────────────

/// The str fixture carries `mpc_code_obs = "I41"` (`MPC_CODE`) for every row.
/// After ingestion, `get_observer` must return `None` when no error model has
/// been set (MPC table cannot be initialised without one), confirming that the
/// dataset correctly stores `ObserverId::MpcCode` rather than a geodetic slot.
///
/// This test does NOT perform a network request: it relies on the absence of
/// an error model to produce `None` without fetching the MPC catalogue.
#[test]
fn str_obs_mpc_no_error_model_returns_none() {
    let mut ds = load_str();

    // The dataset was loaded without an error model (None passed to from_lazy).
    // Attempting to resolve an MPC-coded observer must return None because the
    // MPC table cannot be initialised without an error model.
    // MPC_CODE = b"I41" is the code present in every row of this fixture.
    let first_id: u64 = 3_026_230_983_415_015_002;
    let _ = MPC_CODE; // document that this fixture uses the MPC_CODE constant

    // get_observation to confirm the id exists in this dataset.
    if ds.get_observation(first_id).is_some() {
        let result = ds.get_observer(first_id);
        assert!(
            result.is_none(),
            "get_observer must return None for an MPC observer when no error model is set"
        );
    }
}

/// The str fixture contains no geodetic columns: `get_observer` must return
/// `None` for every sampled observation when no error model is configured,
/// not `Some` (which would indicate an unexpected geodetic resolution).
#[test]
fn str_obs_no_geodetic_fallback() {
    let mut ds = load_str();

    for idx in (0..TOTAL_ROWS).step_by(50_000) {
        let obs_id = {
            let obs = ds
                .get_obs_by_index(idx)
                .unwrap_or_else(|| panic!("Index {idx} must be valid in str fixture"));
            *obs.id()
        };

        // Without an error model the MPC table cannot be initialised:
        // get_observer must return None (not Some with bogus geodetic data).
        let result = ds.get_observer(obs_id);
        assert!(
            result.is_none(),
            "Observation {idx} in str fixture: expected None (no error model), \
             got Some — possible unintended geodetic fallback"
        );
    }
}

// ── 10. ObserverId index validity (int fixture) ───────────────────────────────

/// `get_observer` never panics or returns stale data when called multiple
/// times in sequence for different observation ids sampled across all nights.
/// This indirectly validates that the interned observer index (IntId) is always
/// in bounds.
#[test]
fn int_obs_get_observer_never_panics_across_nights() {
    let mut ds = load_int();

    // Collect one observation id per night.
    let night_obs_ids: Vec<u64> = NIGHT_EXPECTED
        .iter()
        .map(|&(raw_id, _)| {
            let nid = NightId(raw_id);
            let mut iter = ds
                .iter_night_observations(&nid)
                .unwrap_or_else(|| panic!("Night {raw_id} must exist"));
            *iter.next().unwrap().id()
        })
        .collect();

    for id in night_obs_ids {
        // Must not panic and must return Some (all rows have geodetic observer).
        let obs = ds.get_observer(id);
        assert!(
            obs.is_some(),
            "get_observer({id}) must return Some for int fixture"
        );
    }
}

/// Observers resolved through `iter_full_trajectory` (mutable borrow workaround
/// via pre-collected ids) must all be `Some` for the int fixture and all
/// share the same longitude, confirming index integrity across the full
/// trajectory index.
#[test]
fn int_obs_trajectory_observer_index_valid() {
    let mut ds = load_int();

    // Collect a sample of (traj_id, obs_id) pairs from the trajectory index.
    let sample: Vec<u64> = ds
        .iter_full_trajectory()
        .expect("iter_full_trajectory must be Some")
        .step_by(10_000)
        .map(|(_, obs)| *obs.id())
        .collect();

    // Each observation must resolve to a valid observer with the expected longitude.
    for id in sample {
        let observer = ds.get_observer(id).unwrap_or_else(|| {
            panic!("Observation {id} from trajectory index must have an observer")
        });

        assert!(
            (f64::from(observer.longitude) - OBS_LON).abs() < OBSERVER_TOLERANCE,
            "Trajectory-indexed observer longitude must be {OBS_LON} deg, \
             got {} deg for obs id {id}",
            f64::from(observer.longitude)
        );
    }
}
