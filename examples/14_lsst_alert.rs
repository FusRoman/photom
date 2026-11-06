//! Convert Rubin/LSST `DiaSource` alert fields into photom observations.
//!
//! The `lsst_alert` module is pure: the caller decodes the alert (Avro,
//! parquet, …) into a `DiaSourceRecord`, optionally applies a quality-cut
//! policy, converts it, and pushes the result into a live `ObsDataset`.
//! No network access is needed: the `X05` observer is resolved lazily and
//! this example never asks for it.
//!
//! Requires the `lsst_alert` feature:
//!
//! ```sh
//! cargo run --example 14_lsst_alert --features lsst_alert
//! ```

use photom::{
    io::lsst_alert::{
        DiaSourceRecord, LsstConversionParams, LsstQualityCuts, dia_source_to_observation,
    },
    observation_dataset::ObsDataset,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Alert fields, in Rubin units (deg, nJy, MJD TAI). `visit` encodes
    //    the night: dayObs 2026-10-30, sequence number 508.
    let base = DiaSourceRecord {
        dia_source_id: 170_000_000_000_001,
        visit: 2_026_103_000_508,
        midpoint_mjd_tai: 61_344.125, // 2026-10-31 03:00 TAI
        ra_deg: 45.0,
        dec_deg: -30.0,
        ra_err_deg: 2.8e-5, // ~0.1" on the sky (σ_α·cos δ)
        dec_err_deg: 2.8e-5,
        psf_flux_njy: 2_500.0,
        psf_flux_err_njy: 120.0,
        band: "g",
    };
    let records = [
        base,
        // Too faint for the cut below (SNR = 3).
        DiaSourceRecord {
            dia_source_id: 170_000_000_000_002,
            psf_flux_njy: 360.0,
            ..base
        },
        // Unconvertible: the epoch belongs to the next night, not to the
        // night encoded in `visit`.
        DiaSourceRecord {
            dia_source_id: 170_000_000_000_003,
            midpoint_mjd_tai: 61_345.125,
            ..base
        },
    ];

    // 2. Policy (application-defined) and conversion parameters.
    let cuts = LsstQualityCuts {
        min_snr: Some(5.0),
        ..Default::default()
    };
    let params = LsstConversionParams {
        astrometric_floor_arcsec: 0.01,
    };

    // 3. Cut, convert, keep the night/visit keys alongside each observation.
    let mut accepted = Vec::new();
    for rec in &records {
        if let Err(reason) = cuts.check(rec) {
            println!("diaSourceId {} cut: {reason}", rec.dia_source_id);
            continue;
        }
        match dia_source_to_observation(rec, &params) {
            Ok(obs) => {
                println!(
                    "diaSourceId {} → night {} visit {} mag {:.3}±{:.3} σα {:.3e} rad",
                    rec.dia_source_id,
                    obs.night_id,
                    obs.visit,
                    obs.input.photometry.magnitude,
                    obs.input.photometry.error,
                    obs.input.equ_coord.ra_error,
                );
                accepted.push(obs.input);
            }
            Err(err) => println!("diaSourceId {} rejected: {err}", rec.dia_source_id),
        }
    }

    // 4. Append to a live dataset.
    let (dataset, indices) = ObsDataset::empty().push_observation(accepted)?;
    println!("\ninserted at {indices:?}\n{dataset}");
    Ok(())
}
