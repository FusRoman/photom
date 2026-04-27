//! Integration tests for multi-file (per-night) merge of Polars-loaded datasets.
//!
//! These tests split the Parquet fixture by `night_id`, load each night as an
//! independent [`ObsDataset`], merge them sequentially via [`merge_from`], and
//! verify:
//!
//! 1. The total observation count after the merge equals [`TOTAL_ROWS`].
//! 2. Each night's per-night count in the merged dataset matches [`NIGHT_EXPECTED`].
//! 3. Each night's index entry is `Contiguous` after the merge — because every
//!    per-night dataset contributes exactly one contiguous block of observations
//!    for that night, and no two per-night datasets share the same [`NightId`].

#![cfg(feature = "polars")]

mod helpers;
use helpers::*;

use photom::io::polars::FromPolarsArgs;
use photom::{NightId, observation_dataset::ObsDataset};
use polars::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Load a set of nights from the int-traj Parquet fixture into a single [`ObsDataset`].
///
/// Scans the full file, filters to rows whose `night_id` is in `raw_ids`, and
/// ingests the result in one shot.
fn load_int_nights(raw_ids: &[u32]) -> ObsDataset {
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let mask = raw_ids
        .iter()
        .map(|&id| col("night_id").eq(lit(id)))
        .reduce(|a, b| a.or(b))
        .expect("raw_ids must be non-empty");
    let lf = LazyFrame::scan_parquet(PATH_INT.into(), args)
        .expect("scan_parquet must succeed")
        .filter(mask);

    ObsDataset::from_lazy(
        lf,
        FromPolarsArgs {
            do_rechunk: Some(false),
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("from_lazy failed for nights {raw_ids:?}: {e}"))
}

/// Load a single night from the int-traj Parquet fixture.
///
/// Scans the full file, filters to `night_id == raw_id`, and ingests the
/// result into an [`ObsDataset`].
fn load_int_night(raw_id: u32) -> ObsDataset {
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(PATH_INT.into(), args)
        .expect("scan_parquet must succeed")
        .filter(col("night_id").eq(lit(raw_id)));

    ObsDataset::from_lazy(
        lf,
        FromPolarsArgs {
            do_rechunk: Some(false),
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("from_lazy failed for night {raw_id}: {e}"))
}

/// Build a merged [`ObsDataset`] from all nights in [`NIGHT_EXPECTED`],
/// loading each night separately from the int-traj fixture and merging
/// with [`ObsDataset::merge_from`].
fn build_per_night_merged() -> ObsDataset {
    let mut nights = NIGHT_EXPECTED.iter();

    // Seed with the first night.
    let &(first_id, _) = nights.next().expect("NIGHT_EXPECTED must be non-empty");
    let mut merged = load_int_night(first_id);

    for &(raw_id, _) in nights {
        let night_ds = load_int_night(raw_id);
        merged = merged
            .merge_from(night_ds)
            .unwrap_or_else(|e| panic!("merge_from failed for night {raw_id}: {e}"));
    }

    merged
}

// ── 1. Total observation count ────────────────────────────────────────────────

/// The merged dataset must contain exactly [`TOTAL_ROWS`] observations.
#[test]
fn polars_merge_total_row_count() {
    let ds = build_per_night_merged();
    assert_eq!(
        ds.observation_count(),
        TOTAL_ROWS,
        "merged per-night datasets must total {TOTAL_ROWS} rows"
    );
}

// ── 2. Per-night counts ───────────────────────────────────────────────────────

/// Each night's per-night count in the merged dataset must match the fixture.
#[test]
fn polars_merge_per_night_counts_correct() {
    let ds = build_per_night_merged();
    for &(raw_id, expected_count) in NIGHT_EXPECTED {
        let nid = NightId(raw_id);
        let actual = ds
            .len_night(&nid)
            .unwrap_or_else(|| panic!("Night {raw_id} must be present after merge"));
        assert_eq!(
            actual, expected_count,
            "Night {raw_id}: expected {expected_count} obs after merge, got {actual}"
        );
    }
}

// ── 3. Night index contiguity ─────────────────────────────────────────────────

/// Every night's index entry must be `Contiguous` after the per-night merge.
///
/// Because each per-night dataset contributes a single contiguous block for
/// its own night, and no two per-night datasets share a [`NightId`], the merge
/// must preserve the `Contiguous` representation for every night.
#[test]
fn polars_merge_night_index_is_contiguous() {
    let ds = build_per_night_merged();
    for &(raw_id, _) in NIGHT_EXPECTED {
        let nid = NightId(raw_id);
        assert_eq!(
            ds.is_night_contiguous(&nid),
            Some(true),
            "Night {raw_id}: expected Contiguous index after per-night merge"
        );
    }
}

// ── 4. ObsId uniqueness ───────────────────────────────────────────────────────

/// All observation ids in the merged dataset must be unique.
#[test]
fn polars_merge_all_obs_ids_unique() {
    let ds = build_per_night_merged();
    let ids: std::collections::HashSet<_> = ds.iter_observations().map(|o| o.id()).collect();
    assert_eq!(
        ids.len(),
        ds.observation_count(),
        "duplicate ObsIds found after per-night merge"
    );
}

// ── 6. Multi-night datasets merge correctly ───────────────────────────────────

/// Merging two datasets that each already contain several nights produces the
/// same counts and index integrity as the per-night merge.
///
/// The 10 fixture nights are split into two groups of 5.  Each group is loaded
/// as a single [`ObsDataset`] (multi-night in one shot) and the two datasets
/// are merged via [`ObsDataset::merge_from`].  The result must be identical to
/// loading all 10 nights from scratch: correct total, correct per-night counts,
/// and every night's index entry is `Contiguous`.
#[test]
fn polars_merge_multi_night_datasets() {
    let (first_half, second_half) = NIGHT_EXPECTED.split_at(NIGHT_EXPECTED.len() / 2);

    let first_ids: Vec<u32> = first_half.iter().map(|&(id, _)| id).collect();
    let second_ids: Vec<u32> = second_half.iter().map(|&(id, _)| id).collect();

    let mut ds = load_int_nights(&first_ids);
    let other = load_int_nights(&second_ids);
    ds = ds
        .merge_from(other)
        .expect("merge_from must succeed for multi-night datasets");

    // Total count.
    assert_eq!(
        ds.observation_count(),
        TOTAL_ROWS,
        "merged multi-night datasets must total {TOTAL_ROWS} rows"
    );

    // Per-night counts and contiguity.
    for &(raw_id, expected_count) in NIGHT_EXPECTED {
        let nid = NightId(raw_id);

        let actual = ds
            .len_night(&nid)
            .unwrap_or_else(|| panic!("Night {raw_id} must be present after multi-night merge"));
        assert_eq!(
            actual, expected_count,
            "Night {raw_id}: expected {expected_count} obs, got {actual}"
        );

        assert_eq!(
            ds.is_night_contiguous(&nid),
            Some(true),
            "Night {raw_id}: expected Contiguous index after multi-night merge"
        );
    }

    // ObsId uniqueness.
    let ids: std::collections::HashSet<_> = ds.iter_observations().map(|o| o.id()).collect();
    assert_eq!(
        ids.len(),
        ds.observation_count(),
        "duplicate ObsIds found after multi-night merge"
    );
}

/// The sum of all per-night counts (via the night iterator) must equal
/// [`TOTAL_ROWS`] after the per-night merge.
#[test]
fn polars_merge_night_count_sum_equals_total() {
    let ds = build_per_night_merged();
    let sum: usize = ds
        .iter_night_id()
        .expect("night index must be present")
        .map(|nid| ds.len_night(nid).unwrap_or(0))
        .sum();
    assert_eq!(
        sum, TOTAL_ROWS,
        "sum of per-night counts must equal total row count after merge"
    );
}
