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
//!    trajectory index with both integer (`Int64`) and string `traj_id`
//!    column types.
//! 4. **Accessors and iterators** — `get_observation`, `get_obs_by_index`,
//!    `iter_observations`, `iter_night_id`, `iter_traj_id`,
//!    `iter_full_night`, `iter_full_trajectory`, and `push_new_trajectory`
//!    all behave correctly on real data.
//!
//! ## Fixture layout
//!
//! | File | traj_id type | Rows | Night count | Non-null traj |
//! |------|-------------|------|-------------|--------------|
//! | `test_data_traj_int.parquet` | Arrow `Int64` | 655 215 | 10 | 86 096 |
//! | `test_data_traj_str.parquet` | Arrow `String` | 655 215 | 10 | 86 096 |
//!
//! Both files share the same `id`, `night_id`, and photometric columns; only
//! the `traj_id` column differs in type.

#![cfg(feature = "polars")]

use photom::{NightId, TrajId, observation_dataset::ObsDataset};
use polars::prelude::{LazyFrame, ScanArgsParquet};

// ── path constants ────────────────────────────────────────────────────────────

const PATH_INT: &str = "tests/data/test_data_traj_int.parquet";
const PATH_STR: &str = "tests/data/test_data_traj_str.parquet";

/// Total number of rows in each fixture file.
const TOTAL_ROWS: usize = 655_215;

/// Number of distinct nights in both fixture files.
const NIGHT_COUNT: usize = 10;

/// Number of observations with a non-null trajectory identifier.
const TRAJ_NON_NULL: usize = 86_096;

/// Number of distinct trajectory identifiers.
const TRAJ_UNIQUE: usize = 56_559;

/// Night identifiers and their expected observation counts (from the fixture).
const NIGHT_EXPECTED: &[(u32, usize)] = &[
    (2935, 45214),
    (2944, 79526),
    (3010, 78204),
    (3044, 71298),
    (3074, 86675),
    (3111, 95451),
    (3142, 81677),
    (3161, 15442),
    (3249, 87729),
    (3285, 13999),
];

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
/// 3142, and every returned observation has `index` within bounds.
#[test]
fn night_index_iter_night_observations_consistent() {
    let ds = load_int();
    let nid = NightId(3142);
    let expected_count = 81_677usize;

    let obs: Vec<_> = ds
        .iter_night_observations(&nid)
        .expect("night 3142 must exist in index")
        .collect();

    assert_eq!(
        obs.len(),
        expected_count,
        "iter_night_observations for night 3142 must yield {expected_count} observations"
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

/// Trajectory 1072 (Int64 in the file) has exactly 5 observations.
#[test]
fn int_traj_index_traj_1072_count() {
    let ds = load_int();
    let tid = TrajId::Int(1072);
    let count = ds
        .len_trajectory(&tid)
        .expect("Trajectory 1072 must be present in the index");
    assert_eq!(count, 5, "Trajectory 1072 must have exactly 5 observations");
}

/// `iter_trajectory_observations` for trajectory 1072 returns observations
/// whose ids match the known set from the fixture.
#[test]
fn int_traj_index_traj_1072_obs_ids() {
    let ds = load_int();
    let tid = TrajId::Int(1072);
    let mut actual_ids: Vec<u64> = ds
        .iter_trajectory_observations(&tid)
        .expect("Trajectory 1072 must exist")
        .map(|o| *o.id())
        .collect();
    actual_ids.sort_unstable();

    let mut expected_ids: Vec<u64> = vec![
        3142170405915015003,
        3142270925915015002,
        3111262960415015015,
        3111315860415015023,
        3142302485915015001,
    ];
    expected_ids.sort_unstable();

    assert_eq!(
        actual_ids, expected_ids,
        "Trajectory 1072 obs ids must match the fixture"
    );
}

/// `materialize_trajectory` for trajectory 1072 returns the same 5
/// observations as `iter_trajectory_observations`.
#[test]
fn int_traj_index_materialize_traj_1072() {
    let ds = load_int();
    let tid = TrajId::Int(1072);
    let materialized = ds
        .materialize_trajectory(&tid)
        .expect("Trajectory 1072 must be present");
    assert_eq!(
        materialized.len(),
        5,
        "materialize_trajectory must return 5 observations for trajectory 1072"
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

/// Trajectory "1975" (String in the str file) has exactly 5 observations.
///
/// Note: the str fixture uses independent string identifiers — the traj whose
/// 5 observations share ids with integer trajectory 1072 carries the string
/// key "1975" in the string fixture.
#[test]
fn str_traj_index_traj_1975_count() {
    let ds = load_str();
    let tid = TrajId::Str("1975".to_owned());
    let count = ds
        .len_trajectory(&tid)
        .expect("Trajectory \"1975\" must be present in the str-traj index");
    assert_eq!(
        count, 5,
        "Trajectory \"1975\" must have exactly 5 observations"
    );
}

/// `iter_trajectory_observations` for trajectory "1975" (str) returns the
/// expected 5 observation ids.
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

    let mut expected_ids: Vec<u64> = vec![
        3142170405915015003,
        3142270925915015002,
        3111262960415015015,
        3111315860415015023,
        3142302485915015001,
    ];
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

    // The first observation in the int fixture has this id (from Python inspection).
    let first_id: u64 = 3_142_170_400_315_010_015;

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
    let first_id: u64 = 3_142_170_400_315_010_015;

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
    let nid = NightId(2935); // smallest night, 45 214 observations

    // Collect indices from the night iterator.
    let indices: Vec<usize> = ds
        .iter_night_observations(&nid)
        .expect("night 2935 must exist")
        .map(|o| o.index())
        .collect();

    assert_eq!(indices.len(), 45_214);

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
