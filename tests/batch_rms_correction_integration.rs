//! Integration tests for `ModelCorrection::apply_batch_rms_correction` using real MPC data.
#![cfg(feature = "mpc_80_col")]

use approx::assert_ulps_eq;
use photom::{
    observation_dataset::ObsDataset,
    observer::error_model::{ModelCorrection, ObsErrorModel},
};

fn data(name: &str) -> camino::Utf8PathBuf {
    camino::Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

/// Verify batch RMS correction on the 37-observation K09R05F trajectory from 2015AB.obs.
///
/// The pipeline mirrors the original `outfit` approach:
/// 1. Load the MPC 80-column file.
/// 2. Assign per-observer FCCT14 model errors via `apply_model_errors` (replacing
///    the raw format-precision uncertainties with `max(format, model_rms / cos(dec))`).
/// 3. Apply the temporal batch correction via `apply_batch_rms_correction`.
///
/// The expected values are oracle values derived from a known-good run of the full
/// pipeline.  They must not be changed without a justified re-derivation.
#[test]
fn test_batch_real_data() {
    let corrected = ObsDataset::from_mpc_80_col(data("2015AB.obs").as_path())
        .expect("Failed to load 2015AB.obs")
        .with_error_model(ObsErrorModel::FCCT14)
        .apply_model_errors()
        .apply_batch_rms_correction(8.0 / 24.0);

    let obs: Vec<_> = corrected.iter_observations().collect();

    assert_ulps_eq!(
        obs[0].equ_coord().ra_error,
        2.507075226057322e-6,
        max_ulps = 2
    );
    assert_ulps_eq!(
        obs[0].equ_coord().dec_error,
        2.036217397086327e-6,
        max_ulps = 2
    );

    assert_ulps_eq!(
        obs[1].equ_coord().ra_error,
        2.5070681687218917e-6,
        max_ulps = 2
    );
    assert_ulps_eq!(
        obs[1].equ_coord().dec_error,
        2.036217397086327e-6,
        max_ulps = 2
    );

    assert_ulps_eq!(
        obs[2].equ_coord().ra_error,
        2.507_059_507_890_695_2E-6,
        max_ulps = 2
    );
    assert_ulps_eq!(
        obs[2].equ_coord().dec_error,
        2.036217397086327e-6,
        max_ulps = 2
    );
}
