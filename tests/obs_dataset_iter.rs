//! Integration tests for ObsDataset iterator methods.
//!
//! This file exercises the sequential and parallel iterator APIs of
//! [`ObsDataset`] against the real Parquet fixtures in `tests/data/`.
//!
//! ## Structure
//!
//! - [`single_thread`] — sequential iterators (`iter_*`, `materialize_*`,
//!   `len_*`), always compiled.
//! - [`parallel`] — parallel counterparts (`par_iter_*`,
//!   `materialize_*_par`), compiled only when the `parallel` feature is
//!   enabled.
//!
//! Both modules share fixture constants and loaders from the sibling
//! `helpers` module.

#![cfg(feature = "polars")]

mod helpers;
use helpers::*;

use photom::{NightId, TrajId};

// ═══════════════════════════════════════════════════════════════════════════════
// Sequential iterator tests
// ═══════════════════════════════════════════════════════════════════════════════

mod single_thread {
    use super::*;

    // ── iter_observations ─────────────────────────────────────────────────────

    /// `iter_observations` over the int fixture yields exactly TOTAL_ROWS items.
    #[test]
    fn iter_observations_count() {
        let ds = load_int();
        assert_eq!(
            ds.iter_observations().count(),
            TOTAL_ROWS,
            "iter_observations must yield one item per observation"
        );
    }

    /// Observation indices returned by `iter_observations` are strictly
    /// increasing from 0 to TOTAL_ROWS − 1.
    #[test]
    fn iter_observations_indices_sequential() {
        let ds = load_int();
        for (expected, obs) in ds.iter_observations().enumerate() {
            assert_eq!(
                obs.index(),
                expected,
                "Observation at position {expected} must carry index {expected}"
            );
        }
    }

    /// All ids returned by `iter_observations` are unique.
    #[test]
    fn iter_observations_ids_unique() {
        let ds = load_int();
        let mut ids: Vec<u64> = ds.iter_observations().map(|o| *o.id()).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "All observation ids must be unique");
    }

    // ── night iterators ───────────────────────────────────────────────────────

    /// `iter_night_observations` returns `None` for a night not present in the
    /// fixture.
    #[test]
    fn iter_night_observations_none_for_missing_night() {
        let ds = load_int();
        assert!(
            ds.iter_night_observations(&NightId(u32::MAX)).is_none(),
            "iter_night_observations must return None for an absent night"
        );
    }

    /// `iter_night_observations` for night 3010 yields the expected count.
    #[test]
    fn iter_night_observations_count_night_3010() {
        let ds = load_int();
        let nid = NightId(3010);
        let expected = 78_204usize;
        let count = ds
            .iter_night_observations(&nid)
            .expect("night 3010 must be present")
            .count();
        assert_eq!(
            count, expected,
            "iter_night_observations for night 3010 must yield {expected} observations"
        );
    }

    /// Every observation returned by `iter_night_observations` for night 3074
    /// has an index within the valid range.
    #[test]
    fn iter_night_observations_indices_in_bounds() {
        let ds = load_int();
        let nid = NightId(3074);
        for obs in ds
            .iter_night_observations(&nid)
            .expect("night 3074 must be present")
        {
            assert!(
                obs.index() < TOTAL_ROWS,
                "Observation index {} is out of bounds for night 3074",
                obs.index()
            );
        }
    }

    /// `len_night` and `iter_night_observations` agree for every known night.
    #[test]
    fn len_night_consistent_with_iter_for_all_nights() {
        let ds = load_int();
        for &(raw_id, expected_count) in NIGHT_EXPECTED {
            let nid = NightId(raw_id);
            let iter_count = ds
                .iter_night_observations(&nid)
                .unwrap_or_else(|| panic!("Night {raw_id} must be present"))
                .count();
            let len = ds
                .len_night(&nid)
                .unwrap_or_else(|| panic!("len_night({raw_id}) must be Some"));
            assert_eq!(
                iter_count, expected_count,
                "iter count mismatch for night {raw_id}"
            );
            assert_eq!(len, expected_count, "len_night mismatch for night {raw_id}");
        }
    }

    /// `materialize_night` returns a collection whose length equals `len_night`
    /// for every known night.
    #[test]
    fn materialize_night_length_matches_len_night() {
        let ds = load_int();
        for &(raw_id, expected_count) in NIGHT_EXPECTED {
            let nid = NightId(raw_id);
            let mat = ds
                .materialize_night(&nid)
                .unwrap_or_else(|| panic!("Night {raw_id} must be present"));
            assert_eq!(
                mat.len(),
                expected_count,
                "materialize_night length mismatch for night {raw_id}"
            );
        }
    }

    /// `materialize_night` returns `None` for an absent night.
    #[test]
    fn materialize_night_none_for_missing_night() {
        let ds = load_int();
        assert!(
            ds.materialize_night(&NightId(u32::MAX)).is_none(),
            "materialize_night must return None for an absent night"
        );
    }

    /// `iter_full_night` yields TOTAL_ROWS pairs and every returned NightId is
    /// in the known set.
    #[test]
    fn iter_full_night_count_and_valid_ids() {
        let ds = load_int();
        let known: std::collections::HashSet<u32> =
            NIGHT_EXPECTED.iter().map(|&(id, _)| id).collect();
        let mut count = 0usize;
        for (nid, obs) in ds.iter_full_night().expect("iter_full_night must be Some") {
            assert!(
                known.contains(&nid.0),
                "iter_full_night returned unknown NightId {}",
                nid.0
            );
            assert!(
                obs.index() < TOTAL_ROWS,
                "Observation index {} from iter_full_night is out of bounds",
                obs.index()
            );
            count += 1;
        }
        assert_eq!(
            count, TOTAL_ROWS,
            "iter_full_night must yield exactly TOTAL_ROWS pairs"
        );
    }

    /// `iter_full_night` returns `None` for the str fixture if all nights would
    /// require a night index — but actually the str fixture also has nights, so
    /// this tests the night total matches.
    #[test]
    fn iter_full_night_str_fixture_total() {
        let ds = load_str();
        let total = ds
            .iter_full_night()
            .expect("iter_full_night must be Some for str fixture")
            .count();
        assert_eq!(
            total, TOTAL_ROWS,
            "iter_full_night on str fixture must yield TOTAL_ROWS pairs"
        );
    }

    // ── trajectory iterators ──────────────────────────────────────────────────

    /// `iter_trajectory_observations` returns `None` for a trajectory not in
    /// the index.
    #[test]
    fn iter_trajectory_observations_none_for_missing_traj() {
        let ds = load_int();
        assert!(
            ds.iter_trajectory_observations(&TrajId::Int(u32::MAX))
                .is_none(),
            "iter_trajectory_observations must return None for an absent trajectory"
        );
    }

    /// `iter_trajectory_observations` for trajectory 2 yields exactly 7 items.
    #[test]
    fn iter_trajectory_observations_count_traj_2() {
        let ds = load_int();
        let tid = TrajId::Int(2);
        let count = ds
            .iter_trajectory_observations(&tid)
            .expect("Trajectory 2 must be present")
            .count();
        assert_eq!(count, 7, "Trajectory 2 must have exactly 7 observations");
    }

    /// `len_trajectory` and `iter_trajectory_observations` agree for trajectory 2.
    #[test]
    fn len_trajectory_consistent_with_iter_traj_2() {
        let ds = load_int();
        let tid = TrajId::Int(2);
        let iter_count = ds
            .iter_trajectory_observations(&tid)
            .expect("Trajectory 2 must be present")
            .count();
        let len = ds
            .len_trajectory(&tid)
            .expect("len_trajectory for traj 2 must be Some");
        assert_eq!(iter_count, len, "iter count and len_trajectory must agree");
    }

    /// The sum of `len_trajectory` across all trajectories equals TRAJ_NON_NULL.
    #[test]
    fn len_trajectory_sums_to_traj_non_null() {
        let ds = load_int();
        let total: usize = ds
            .iter_traj_id()
            .expect("trajectory index must be present")
            .map(|tid| ds.len_trajectory(tid).unwrap_or(0))
            .sum();
        assert_eq!(
            total, TRAJ_NON_NULL,
            "sum of len_trajectory must equal TRAJ_NON_NULL"
        );
    }

    /// `materialize_trajectory` for trajectory 2 returns 7 observations.
    #[test]
    fn materialize_trajectory_count_traj_2() {
        let ds = load_int();
        let tid = TrajId::Int(2);
        let mat = ds
            .materialize_trajectory(&tid)
            .expect("Trajectory 2 must be present");
        assert_eq!(mat.len(), 7, "materialize_trajectory must return 7 items");
    }

    /// `materialize_trajectory` and `iter_trajectory_observations` yield the
    /// same ids (sorted) for trajectory 2.
    #[test]
    fn materialize_trajectory_ids_match_iter_traj_2() {
        let ds = load_int();
        let tid = TrajId::Int(2);

        let mut iter_ids: Vec<u64> = ds
            .iter_trajectory_observations(&tid)
            .expect("Trajectory 2 must be present")
            .map(|o| *o.id())
            .collect();
        iter_ids.sort_unstable();

        let mut mat_ids: Vec<u64> = ds
            .materialize_trajectory(&tid)
            .expect("Trajectory 2 must be present")
            .into_iter()
            .map(|o| *o.id())
            .collect();
        mat_ids.sort_unstable();

        assert_eq!(
            iter_ids, mat_ids,
            "materialize_trajectory and iter_trajectory_observations must yield the same ids"
        );
    }

    /// `materialize_trajectory` returns `None` for an absent trajectory.
    #[test]
    fn materialize_trajectory_none_for_missing_traj() {
        let ds = load_int();
        assert!(
            ds.materialize_trajectory(&TrajId::Int(u32::MAX)).is_none(),
            "materialize_trajectory must return None for an absent trajectory"
        );
    }

    /// `iter_full_trajectory` yields exactly TRAJ_NON_NULL pairs.
    #[test]
    fn iter_full_trajectory_total() {
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

    /// Every observation yielded by `iter_full_trajectory` has a valid index.
    #[test]
    fn iter_full_trajectory_indices_in_bounds() {
        let ds = load_int();
        for (_tid, obs) in ds
            .iter_full_trajectory()
            .expect("iter_full_trajectory must be Some")
        {
            assert!(
                obs.index() < TOTAL_ROWS,
                "Observation index {} from iter_full_trajectory is out of bounds",
                obs.index()
            );
        }
    }

    /// `iter_full_trajectory` on the str fixture yields TRAJ_NON_NULL pairs.
    #[test]
    fn iter_full_trajectory_str_fixture_total() {
        let ds = load_str();
        let total = ds
            .iter_full_trajectory()
            .expect("iter_full_trajectory must be Some for str fixture")
            .count();
        assert_eq!(
            total, TRAJ_NON_NULL,
            "iter_full_trajectory on str fixture must yield TRAJ_NON_NULL pairs"
        );
    }

    /// `iter_traj_id` returns the expected number of unique trajectory ids.
    #[test]
    fn iter_traj_id_unique_count() {
        let ds = load_int();
        let count = ds
            .iter_traj_id()
            .expect("trajectory index must be present")
            .count();
        assert_eq!(
            count, TRAJ_UNIQUE,
            "iter_traj_id must yield {TRAJ_UNIQUE} distinct trajectory ids"
        );
    }

    /// `iter_night_id` returns the expected number of unique night ids.
    #[test]
    fn iter_night_id_unique_count() {
        let ds = load_int();
        let count = ds
            .iter_night_id()
            .expect("night index must be present")
            .count();
        assert_eq!(
            count, NIGHT_COUNT,
            "iter_night_id must yield {NIGHT_COUNT} distinct night ids"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parallel iterator tests — compiled only with the "parallel" feature
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use rayon::iter::ParallelIterator;

    // ── par_iter_observations ─────────────────────────────────────────────────

    /// `par_iter_observations` on the int fixture yields TOTAL_ROWS items.
    #[test]
    fn par_iter_observations_count() {
        let ds = load_int();
        assert_eq!(
            ds.par_iter_observations().count(),
            TOTAL_ROWS,
            "par_iter_observations must yield TOTAL_ROWS items"
        );
    }

    /// Ids collected from `par_iter_observations`, once sorted, match those
    /// from the sequential `iter_observations` — confirming no items are lost
    /// or duplicated.
    #[test]
    fn par_iter_observations_ids_match_sequential() {
        let ds = load_int();

        let mut seq_ids: Vec<u64> = ds.iter_observations().map(|o| *o.id()).collect();
        seq_ids.sort_unstable();

        let mut par_ids: Vec<u64> = ds.par_iter_observations().map(|o| *o.id()).collect();
        par_ids.sort_unstable();

        assert_eq!(
            seq_ids, par_ids,
            "par_iter_observations and iter_observations must yield the same ids"
        );
    }

    // ── par_iter_full_night ───────────────────────────────────────────────────

    /// `par_iter_full_night` is `Some` for the int fixture (which has a night
    /// index) and yields TOTAL_ROWS pairs.
    #[test]
    fn par_iter_full_night_count() {
        let ds = load_int();
        let count = ds
            .par_iter_full_night()
            .expect("par_iter_full_night must be Some for int fixture")
            .count();
        assert_eq!(
            count, TOTAL_ROWS,
            "par_iter_full_night must yield TOTAL_ROWS pairs"
        );
    }

    /// Every NightId returned by `par_iter_full_night` belongs to the known set.
    #[test]
    fn par_iter_full_night_valid_night_ids() {
        let ds = load_int();
        let known: std::collections::HashSet<u32> =
            NIGHT_EXPECTED.iter().map(|&(id, _)| id).collect();
        let invalid_count = ds
            .par_iter_full_night()
            .expect("par_iter_full_night must be Some")
            .filter(|(nid, _)| !known.contains(&nid.0))
            .count();
        assert_eq!(
            invalid_count, 0,
            "par_iter_full_night must not return unknown NightIds"
        );
    }

    /// Night counts collected in parallel match the sequential `len_night` values.
    #[test]
    fn par_iter_full_night_per_night_counts_match_sequential() {
        let ds = load_int();
        use std::collections::HashMap;

        // Collect (NightId, count) in parallel.
        let par_counts: HashMap<u32, usize> = {
            // Group counts by collecting all night ids then counting per group.
            let night_ids: Vec<NightId> = ds
                .par_iter_full_night()
                .expect("must be Some")
                .map(|(nid, _)| nid)
                .collect();
            let mut map: HashMap<u32, usize> = HashMap::new();
            for nid in night_ids {
                *map.entry(nid.0).or_insert(0) += 1;
            }
            map
        };

        for &(raw_id, expected_count) in NIGHT_EXPECTED {
            let par_count = par_counts.get(&raw_id).copied().unwrap_or(0);
            assert_eq!(
                par_count, expected_count,
                "Parallel count for night {raw_id} must equal {expected_count}"
            );
        }
    }

    // ── par_iter_night_observations ───────────────────────────────────────────

    /// `par_iter_night_observations` returns `None` for an absent night.
    #[test]
    fn par_iter_night_observations_none_for_missing_night() {
        let ds = load_int();
        assert!(
            ds.par_iter_night_observations(&NightId(u32::MAX)).is_none(),
            "par_iter_night_observations must return None for an absent night"
        );
    }

    /// `par_iter_night_observations` for night 3140 yields the expected count.
    #[test]
    fn par_iter_night_observations_count_night_3140() {
        let ds = load_int();
        let nid = NightId(3140);
        let expected = 88_273usize;
        let count = ds
            .par_iter_night_observations(&nid)
            .expect("night 3140 must be present")
            .count();
        assert_eq!(
            count, expected,
            "par_iter_night_observations for night 3140 must yield {expected} items"
        );
    }

    /// Ids from `par_iter_night_observations` match those from
    /// `iter_night_observations` for a known night (order-independent).
    #[test]
    fn par_iter_night_observations_ids_match_sequential() {
        let ds = load_int();
        let nid = NightId(3248); // smallest night — fast to process

        let mut seq_ids: Vec<u64> = ds
            .iter_night_observations(&nid)
            .expect("night 3248 must be present")
            .map(|o| *o.id())
            .collect();
        seq_ids.sort_unstable();

        let mut par_ids: Vec<u64> = ds
            .par_iter_night_observations(&nid)
            .expect("night 3248 must be present")
            .map(|o| *o.id())
            .collect();
        par_ids.sort_unstable();

        assert_eq!(
            seq_ids, par_ids,
            "par and seq night iterators must yield the same ids for night 3248"
        );
    }

    // ── materialize_night_par ─────────────────────────────────────────────────

    /// `materialize_night_par` returns `None` for an absent night.
    #[test]
    fn materialize_night_par_none_for_missing_night() {
        let ds = load_int();
        assert!(
            ds.materialize_night_par(&NightId(u32::MAX)).is_none(),
            "materialize_night_par must return None for an absent night"
        );
    }

    /// `materialize_night_par` for night 3010 returns the expected count.
    #[test]
    fn materialize_night_par_count_night_3010() {
        let ds = load_int();
        let nid = NightId(3010);
        let expected = 78_204usize;
        let vec = ds
            .materialize_night_par(&nid)
            .expect("night 3010 must be present");
        assert_eq!(
            vec.len(),
            expected,
            "materialize_night_par for night 3010 must return {expected} items"
        );
    }

    /// The sorted ids from `materialize_night_par` match those from
    /// `materialize_night` (sequential) for a known night.
    #[test]
    fn materialize_night_par_ids_match_sequential() {
        let ds = load_int();
        let nid = NightId(3248); // smallest night

        let mut seq_ids: Vec<u64> = ds
            .materialize_night(&nid)
            .expect("night 3248 must be present")
            .into_iter()
            .map(|o| *o.id())
            .collect();
        seq_ids.sort_unstable();

        let mut par_ids: Vec<u64> = ds
            .materialize_night_par(&nid)
            .expect("night 3248 must be present")
            .iter()
            .map(|o| *o.id())
            .collect();
        par_ids.sort_unstable();

        assert_eq!(
            seq_ids, par_ids,
            "materialize_night_par and materialize_night must yield the same ids"
        );
    }

    // ── par_iter_trajectory_observations ─────────────────────────────────────

    /// `par_iter_trajectory_observations` returns `None` for an absent
    /// trajectory.
    #[test]
    fn par_iter_trajectory_observations_none_for_missing_traj() {
        let ds = load_int();
        assert!(
            ds.par_iter_trajectory_observations(&TrajId::Int(u32::MAX))
                .is_none(),
            "par_iter_trajectory_observations must return None for an absent trajectory"
        );
    }

    /// `par_iter_trajectory_observations` for trajectory 2 yields exactly 7
    /// items.
    #[test]
    fn par_iter_trajectory_observations_count_traj_2() {
        let ds = load_int();
        let tid = TrajId::Int(2);
        let count = ds
            .par_iter_trajectory_observations(&tid)
            .expect("Trajectory 2 must be present")
            .count();
        assert_eq!(
            count, 7,
            "par_iter_trajectory_observations for traj 2 must yield 7 items"
        );
    }

    /// Ids from `par_iter_trajectory_observations` match those from
    /// `iter_trajectory_observations` for trajectory 2.
    #[test]
    fn par_iter_trajectory_observations_ids_match_sequential() {
        let ds = load_int();
        let tid = TrajId::Int(2);

        let mut seq_ids: Vec<u64> = ds
            .iter_trajectory_observations(&tid)
            .expect("Trajectory 2 must be present")
            .map(|o| *o.id())
            .collect();
        seq_ids.sort_unstable();

        let mut par_ids: Vec<u64> = ds
            .par_iter_trajectory_observations(&tid)
            .expect("Trajectory 2 must be present")
            .map(|o| *o.id())
            .collect();
        par_ids.sort_unstable();

        assert_eq!(
            seq_ids, par_ids,
            "par and seq trajectory iterators must yield the same ids for traj 2"
        );
    }

    // ── par_iter_full_trajectory ──────────────────────────────────────────────

    /// `par_iter_full_trajectory` is `Some` for the int fixture and yields
    /// TRAJ_NON_NULL pairs.
    #[test]
    fn par_iter_full_trajectory_count() {
        let ds = load_int();
        let count = ds
            .par_iter_full_trajectory()
            .expect("par_iter_full_trajectory must be Some for int fixture")
            .count();
        assert_eq!(
            count, TRAJ_NON_NULL,
            "par_iter_full_trajectory must yield TRAJ_NON_NULL pairs"
        );
    }

    /// Every observation index in `par_iter_full_trajectory` is within bounds.
    #[test]
    fn par_iter_full_trajectory_indices_in_bounds() {
        let ds = load_int();
        let out_of_bounds = ds
            .par_iter_full_trajectory()
            .expect("must be Some")
            .filter(|(_, obs)| obs.index() >= TOTAL_ROWS)
            .count();
        assert_eq!(
            out_of_bounds, 0,
            "All observation indices from par_iter_full_trajectory must be in bounds"
        );
    }

    /// Sorted ids from `par_iter_full_trajectory` match those from the
    /// sequential `iter_full_trajectory`.
    #[test]
    fn par_iter_full_trajectory_ids_match_sequential() {
        let ds = load_int();

        let mut seq_ids: Vec<u64> = ds
            .iter_full_trajectory()
            .expect("iter_full_trajectory must be Some")
            .map(|(_, obs)| *obs.id())
            .collect();
        seq_ids.sort_unstable();

        let mut par_ids: Vec<u64> = ds
            .par_iter_full_trajectory()
            .expect("par_iter_full_trajectory must be Some")
            .map(|(_, obs)| *obs.id())
            .collect();
        par_ids.sort_unstable();

        assert_eq!(
            seq_ids, par_ids,
            "par and seq full-trajectory iterators must yield the same observation ids"
        );
    }

    // ── materialize_trajectory_par ────────────────────────────────────────────

    /// `materialize_trajectory_par` returns `None` for an absent trajectory.
    #[test]
    fn materialize_trajectory_par_none_for_missing_traj() {
        let ds = load_int();
        assert!(
            ds.materialize_trajectory_par(&TrajId::Int(u32::MAX))
                .is_none(),
            "materialize_trajectory_par must return None for an absent trajectory"
        );
    }

    /// `materialize_trajectory_par` for trajectory 2 returns 7 items.
    #[test]
    fn materialize_trajectory_par_count_traj_2() {
        let ds = load_int();
        let tid = TrajId::Int(2);
        let vec = ds
            .materialize_trajectory_par(&tid)
            .expect("Trajectory 2 must be present");
        assert_eq!(
            vec.len(),
            7,
            "materialize_trajectory_par for traj 2 must return 7 items"
        );
    }

    /// Sorted ids from `materialize_trajectory_par` match those from the
    /// sequential `materialize_trajectory` for trajectory 2.
    #[test]
    fn materialize_trajectory_par_ids_match_sequential() {
        let ds = load_int();
        let tid = TrajId::Int(2);

        let mut seq_ids: Vec<u64> = ds
            .materialize_trajectory(&tid)
            .expect("Trajectory 2 must be present")
            .into_iter()
            .map(|o| *o.id())
            .collect();
        seq_ids.sort_unstable();

        let mut par_ids: Vec<u64> = ds
            .materialize_trajectory_par(&tid)
            .expect("Trajectory 2 must be present")
            .iter()
            .map(|o| *o.id())
            .collect();
        par_ids.sort_unstable();

        assert_eq!(
            seq_ids, par_ids,
            "materialize_trajectory_par and materialize_trajectory must yield the same ids"
        );
    }

    // ── str fixture (parallel) ────────────────────────────────────────────────

    /// `par_iter_full_trajectory` on the str fixture also yields TRAJ_NON_NULL
    /// pairs, confirming that both `TrajId::Str` and `TrajId::Int` indices work
    /// with the parallel API.
    #[test]
    fn par_iter_full_trajectory_str_fixture_total() {
        let ds = load_str();
        let count = ds
            .par_iter_full_trajectory()
            .expect("par_iter_full_trajectory must be Some for str fixture")
            .count();
        assert_eq!(
            count, TRAJ_NON_NULL,
            "par_iter_full_trajectory on str fixture must yield TRAJ_NON_NULL pairs"
        );
    }

    /// `par_iter_full_night` on the str fixture yields TOTAL_ROWS pairs.
    #[test]
    fn par_iter_full_night_str_fixture_total() {
        let ds = load_str();
        let count = ds
            .par_iter_full_night()
            .expect("par_iter_full_night must be Some for str fixture")
            .count();
        assert_eq!(
            count, TOTAL_ROWS,
            "par_iter_full_night on str fixture must yield TOTAL_ROWS pairs"
        );
    }
}
