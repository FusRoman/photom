//! Photometric measurements, and the two astrometric error-correction
//! mechanisms `photom` provides for MPC-coded observations.
//!
//! Requires the `mpc_80_col` feature (to load the fixture file):
//!
//! ```sh
//! cargo run --example 10_photometry_error_models --features mpc_80_col
//! ```

use camino::Utf8PathBuf;
use photom::{
    observation_dataset::ObsDataset,
    observer::error_model::{ModelCorrection, ObsErrorModel},
};

fn data(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2015AB.obs: 37 observations of a single trajectory ("K09R05F"), taken
    // from several MPC-coded sites in short bursts.
    let dataset = ObsDataset::from_mpc_80_col(data("2015AB.obs"))?;

    // 1. `Photometry` on a single observation: magnitude, its 1-sigma
    //    uncertainty, and the filter it was measured through. We copy out
    //    what we need before `dataset` is consumed by the builder-style
    //    methods below (each one takes `self` by value and returns `Self`).
    let first = dataset
        .iter_observations()
        .next()
        .expect("file is non-empty");
    println!("first observation's photometry: {}", first.photometry());
    let first_id = *first.id();
    let (ra_err_before, dec_err_before) = (first.equ_coord().ra_error, first.equ_coord().dec_error);

    // 2. Attach an astrometric error model. VFCC17 (Veres et al. 2017) is one
    //    of the three published models bundled with the crate.
    let dataset = dataset.with_error_model(ObsErrorModel::VFCC17);

    // 3. `apply_model_errors`: for each MPC-coded observation, replace the
    //    raw format-precision uncertainty with the site's nominal RMS from
    //    the model table (whichever is larger). This uses only the 3-byte
    //    MPC code already stored on the observation -- no network access.
    let dataset = dataset.apply_model_errors();
    let after_model = dataset
        .get_observation(first_id)
        .expect("id 0 still exists");
    println!(
        "\nafter apply_model_errors: ra_error {:.3e} -> {:.3e} rad, dec_error {:.3e} -> {:.3e} rad",
        ra_err_before,
        after_model.equ_coord().ra_error,
        dec_err_before,
        after_model.equ_coord().dec_error
    );

    // 4. `apply_batch_rms_correction`: observations of the *same trajectory*,
    //    from the *same site*, taken within `gap_max` days of each other are
    //    correlated (same plate solution, same reference stars) and get
    //    their uncertainty inflated by a factor derived from the batch size.
    //    8 hours is a typical `gap_max` for a single-night tracklet.
    let gap_max = 8.0 / 24.0;
    let corrected = dataset.apply_batch_rms_correction(gap_max);

    println!("\nfirst 4 observations after batch RMS correction (gap_max = 8h):");
    for obs in corrected.iter_observations().take(4) {
        println!(
            "  id={} ra_error={:.6e} rad dec_error={:.6e} rad",
            obs.id(),
            obs.equ_coord().ra_error,
            obs.equ_coord().dec_error
        );
    }

    Ok(())
}
