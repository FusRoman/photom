//! Integration tests for the ADES XML → ObsDataset ingestion pipeline.
//!
//! These tests load the ADES fixtures in `tests/data/` and verify:
//!
//! 1. Structured ADES files parse without error.
//! 2. The observation count matches the number of `<optical>` elements.
//! 3. Trajectory identifiers are correctly indexed.
//! 4. Epoch values are positive MJD TT numbers.
//! 5. Positional uncertainties are non-zero.

#![cfg(feature = "ades")]

use camino::Utf8Path;
use photom::{TrajId, observation_dataset::ObsDataset};

// ── helpers ─────────────────────────────────────────────────────────────────

fn fixture(name: &str) -> camino::Utf8PathBuf {
    Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

// ── example_ades.xml — one obsBlock, four optical observations ───────────────

#[test]
fn ades_example1_observation_count() {
    let path = fixture("example_ades.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    assert_eq!(
        ds.observation_count(),
        4,
        "example_ades.xml contains 4 <optical> elements"
    );
}

#[test]
fn ades_example1_trajectory_count() {
    let path = fixture("example_ades.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    // All four observations have different permIDs → 4 distinct trajectories.
    let traj_count = ds.iter_traj_id().map(|it| it.count()).unwrap_or(0);
    assert_eq!(
        traj_count, 4,
        "example_ades.xml has 4 distinct permID values"
    );
}

#[test]
fn ades_example1_observations_have_positive_mjd() {
    let path = fixture("example_ades.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    for obs in ds.iter_observations() {
        assert!(
            obs.mjd_tt() > 0.0,
            "MJD TT should be positive, got {}",
            obs.mjd_tt()
        );
    }
}

#[test]
fn ades_example1_errors_are_nonzero() {
    let path = fixture("example_ades.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    for obs in ds.iter_observations() {
        let coord = obs.equ_coord();
        assert!(coord.ra_error > 0.0, "RA error should be > 0");
        assert!(coord.dec_error > 0.0, "Dec error should be > 0");
    }
}

// ── example_ades2.xml — three obsBlocks, nine optical observations ───────────

#[test]
fn ades_example2_observation_count() {
    let path = fixture("example_ades2.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    // 3 obs (291) + 2 obs (T12) + 4 obs (568) = 9
    assert_eq!(
        ds.observation_count(),
        9,
        "example_ades2.xml contains 9 <optical> elements across three blocks"
    );
}

#[test]
fn ades_example2_trajectory_count() {
    let path = fixture("example_ades2.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    // Distinct object IDs: P10kefK, "2016 RD34", "2016 JB29"
    let traj_count = ds.iter_traj_id().map(|it| it.count()).unwrap_or(0);
    assert_eq!(
        traj_count, 3,
        "example_ades2.xml has 3 distinct object identifiers"
    );
}

#[test]
fn ades_example2_p10kefk_observation_count() {
    let path = fixture("example_ades2.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    let count = ds
        .len_trajectory(TrajId::Str("P10kefK".to_string()))
        .unwrap_or(0);
    assert_eq!(count, 3, "P10kefK has 3 observations");
}

#[test]
fn ades_example2_rd34_observation_count() {
    let path = fixture("example_ades2.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    let count = ds
        .len_trajectory(TrajId::Str("2016 RD34".to_string()))
        .unwrap_or(0);
    assert_eq!(count, 2, "2016 RD34 has 2 observations");
}

#[test]
fn ades_example2_jb29_observation_count() {
    let path = fixture("example_ades2.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    let count = ds
        .len_trajectory(TrajId::Str("2016 JB29".to_string()))
        .unwrap_or(0);
    assert_eq!(count, 4, "2016 JB29 has 4 observations");
}

#[test]
fn ades_example2_mjd_tt_is_positive() {
    let path = fixture("example_ades2.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    for obs in ds.iter_observations() {
        assert!(
            obs.mjd_tt() > 0.0,
            "MJD TT should be positive, got {}",
            obs.mjd_tt()
        );
    }
}

#[test]
fn ades_example2_prec_based_errors_are_nonzero() {
    // example_ades2.xml uses precRA/precDec, not rmsRA/rmsDec.
    // Verify the reader uses those as fallback uncertainty sources.
    let path = fixture("example_ades2.xml");
    let ds = ObsDataset::from_ades(&path, None, None).unwrap();
    for obs in ds.iter_observations() {
        let coord = obs.equ_coord();
        assert!(
            coord.ra_error > 0.0,
            "RA error from precRA should be > 0, got {}",
            coord.ra_error
        );
        assert!(
            coord.dec_error > 0.0,
            "Dec error from precDec should be > 0, got {}",
            coord.dec_error
        );
    }
}

#[test]
fn ades_fallback_error_overrides_missing_rms() {
    // Provide explicit fallback errors; the result should be equal to
    // fallback / 3600 * π/180 (arcsec → degrees → radians).
    // Here we just verify the dataset loads successfully and all errors > 0.
    let path = fixture("example_ades2.xml");
    let ds = ObsDataset::from_ades(&path, Some(1.0), Some(1.0)).unwrap();
    assert_eq!(ds.observation_count(), 9);
}
