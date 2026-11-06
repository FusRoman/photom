#![cfg(feature = "lsst_alert")]
//! Conversion of Rubin/LSST `DiaSource` alerts into photom observations.
//!
//! This module is **pure**: it neither decodes Avro nor talks to Kafka.  The
//! caller fills a [`DiaSourceRecord`] from whatever transport it uses (Fink
//! Kafka stream, parquet replay, tests) and calls
//! [`dia_source_to_observation`], which yields an [`ObservationInput`] ready
//! for [`ObsDataset::push_observation`](crate::observation_dataset::ObsDataset::push_observation),
//! together with its [`NightId`] and Rubin `visit`.
//!
//! ## Conventions
//!
//! Inputs follow the Rubin APDB `DiaSource` schema
//! (<https://sdm-schemas.lsst.io/apdb.html>); outputs follow photom.
//!
//! | Quantity | Rubin field (unit) | photom value |
//! |----------|--------------------|--------------|
//! | id | `diaSourceId` (long) | [`ObsId`](crate::observation_dataset::ObsId), same value |
//! | position | `ra`, `dec` (deg) | radians |
//! | RA error | `raErr` (deg, **on sky**: σ_α·cos δ) | √(raErr² + floor²) / cos δ, radians — error on the α coordinate |
//! | Dec error | `decErr` (deg) | √(decErr² + floor²), radians |
//! | magnitude | `psfFlux` (nJy) | 31.4 − 2.5·log₁₀(psfFlux), AB |
//! | magnitude error | `psfFluxErr` (nJy) | 2.5/ln 10 · psfFluxErr / psfFlux |
//! | filter | `band` (`u`…`y`) | [`Filter::Int`](crate::photometry::Filter::Int) `0..=5` |
//! | epoch | `midpointMjdTai` (MJD TAI) | MJD TT (+32.184 s) |
//! | observer | — | [`ObserverId::MpcCode`]`(X05)` |
//! | night | `visit` = dayObs·100000 + seqNum | [`NightId`] = MJD of dayObs |
//!
//! The astrometric floor ([`LsstConversionParams::astrometric_floor_arcsec`])
//! is an on-sky systematic, so it is added in quadrature **before** the
//! `1 / cos δ` conversion.  The `ra_dec_Cov` term is ignored:
//! [`EquCoord`] carries marginal errors only.
//!
//! ## Conversion errors vs quality cuts
//!
//! [`dia_source_to_observation`] only rejects what **cannot** be converted
//! (non-positive flux, non-finite or out-of-range values, unknown band,
//! malformed visit, visit/epoch night disagreement); see [`LsstAlertError`].
//! Science-quality selection (SNR, astrometric ceilings) is a separate,
//! optional policy: [`LsstQualityCuts`].
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`DiaSourceRecord`] | struct | DiaSource fields needed for tracking, in Rubin units |
//! | [`LsstConversionParams`] | struct | Conversion parameters (astrometric floor) |
//! | [`LsstObservation`] | struct | Converted observation + its night and visit |
//! | [`dia_source_to_observation`] | fn | The conversion |
//! | [`RUBIN_MPC_CODE`] | const | MPC code of the Rubin Observatory (`X05`) |
//! | [`LsstAlertError`] | enum | Conversion errors |
//! | [`LsstQualityCuts`] / [`CutReason`] | struct / enum | Optional quality-cut policy |
//! | [`units`] | module | Pure unit conversions (flux, time, band, errors) |
//! | [`night`] | module | `visit` → `dayObs` → [`NightId`] |

pub mod cuts;
pub mod error;
pub mod night;
pub mod units;

pub use cuts::{CutReason, LsstQualityCuts};
pub use error::LsstAlertError;

use crate::{
    Arcseconds, NightId,
    constants::ARCSEC_TO_RAD,
    coordinates::equatorial::EquCoord,
    observation_dataset::observation::ObservationInput,
    observer::{dataset::ObserverId, mpc::MpcCode},
    photometry::Photometry,
};

use night::{day_obs_from_visit, day_obs_to_night_id, night_from_mjd};
use units::{
    band_to_filter, flux_err_to_mag_err, flux_njy_to_ab_mag, mjd_tai_to_tt, ra_coordinate_error,
    with_floor,
};

/// MPC observatory code of the Vera C. Rubin Observatory.
pub const RUBIN_MPC_CODE: MpcCode = *b"X05";

/// The fields of a Rubin `DiaSource` needed for tracking, in Rubin units.
///
/// Field types mirror the APDB schema (`float` columns are `f32`), so the
/// struct can be filled directly from a decoded alert without conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiaSourceRecord<'a> {
    /// `diaSourceId` — unique source identifier; must be non-negative.
    pub dia_source_id: i64,
    /// `visit` — `dayObs × 100000 + seqNum`.
    pub visit: i64,
    /// `midpointMjdTai` — exposure mid-point, MJD in TAI (days).
    pub midpoint_mjd_tai: f64,
    /// `ra` — right ascension, degrees, in `[0, 360)`.
    pub ra_deg: f64,
    /// `dec` — declination, degrees, in `(−90, 90)` (the poles are rejected).
    pub dec_deg: f64,
    /// `raErr` — on-sky RA uncertainty (σ_α·cos δ), degrees.
    pub ra_err_deg: f32,
    /// `decErr` — declination uncertainty, degrees.
    pub dec_err_deg: f32,
    /// `psfFlux` — PSF difference-image flux, nJy.
    pub psf_flux_njy: f32,
    /// `psfFluxErr` — uncertainty on `psfFlux`, nJy.
    pub psf_flux_err_njy: f32,
    /// `band` — filter name, one of `u g r i z y`.
    pub band: &'a str,
}

/// Parameters of [`dia_source_to_observation`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LsstConversionParams {
    /// On-sky systematic astrometric floor, in arcseconds, added in
    /// quadrature to `raErr` and `decErr`.  Must be finite and non-negative;
    /// `0` (the default) keeps the published errors unchanged.
    pub astrometric_floor_arcsec: Arcseconds,
}

/// A converted DiaSource: the observation to insert plus its tracking keys.
///
/// `night_id` and `visit` are not part of
/// [`Observation`](crate::observation_dataset::observation::Observation):
/// the night is a dataset index, and the visit groups the sources of one
/// exposure for the association step.
#[derive(Debug, Clone)]
pub struct LsstObservation {
    /// The observation, ready for
    /// [`ObsDataset::push_observation`](crate::observation_dataset::ObsDataset::push_observation).
    pub input: ObservationInput,
    /// Observing night: MJD of the `dayObs` date encoded in `visit`.
    pub night_id: NightId,
    /// Rubin `visit` identifier (one exposure).
    pub visit: i64,
}

/// Convert a Rubin `DiaSource` into a photom observation.
///
/// See the [module documentation](self) for the unit conventions.
///
/// # Arguments
/// - `rec` — the DiaSource fields, in Rubin units.
/// - `params` — conversion parameters.
///
/// # Returns
/// The [`LsstObservation`] carrying the [`ObservationInput`], its
/// [`NightId`] and `visit`.
///
/// # Errors
/// An [`LsstAlertError`] when the source cannot be converted; checks run in
/// the order: finiteness, id, visit and night consistency, coordinate range,
/// uncertainties, flux, band.
pub fn dia_source_to_observation(
    rec: &DiaSourceRecord<'_>,
    params: &LsstConversionParams,
) -> Result<LsstObservation, LsstAlertError> {
    check_finite(rec, params)?;

    let id = u64::try_from(rec.dia_source_id)
        .map_err(|_| LsstAlertError::NegativeId(rec.dia_source_id))?;

    let night_id = day_obs_to_night_id(day_obs_from_visit(rec.visit)?)?;
    let mjd_night = night_from_mjd(rec.midpoint_mjd_tai);
    if mjd_night != i64::from(night_id.0) {
        return Err(LsstAlertError::NightMismatch {
            visit_night: night_id,
            mjd_night,
        });
    }

    if !(0.0..360.0).contains(&rec.ra_deg) {
        return Err(LsstAlertError::CoordinateOutOfRange {
            field: "ra",
            value: rec.ra_deg,
        });
    }
    // Open interval: at the poles RA is undefined and σ_α = σ_sky / cos δ diverges.
    if rec.dec_deg.abs() >= 90.0 {
        return Err(LsstAlertError::CoordinateOutOfRange {
            field: "dec",
            value: rec.dec_deg,
        });
    }

    for (field, value) in [
        ("raErr", f64::from(rec.ra_err_deg)),
        ("decErr", f64::from(rec.dec_err_deg)),
        ("psfFluxErr", f64::from(rec.psf_flux_err_njy)),
        ("astrometric_floor_arcsec", params.astrometric_floor_arcsec),
    ] {
        if value < 0.0 {
            return Err(LsstAlertError::NegativeErr { field, value });
        }
    }

    if rec.psf_flux_njy <= 0.0 {
        return Err(LsstAlertError::NonPositiveFlux(rec.psf_flux_njy));
    }

    let filter = band_to_filter(rec.band)?;

    let floor_rad = params.astrometric_floor_arcsec * ARCSEC_TO_RAD;
    let dec = rec.dec_deg.to_radians();
    let ra_error = ra_coordinate_error(
        with_floor(f64::from(rec.ra_err_deg).to_radians(), floor_rad),
        dec,
    );
    let dec_error = with_floor(f64::from(rec.dec_err_deg).to_radians(), floor_rad);

    let flux = f64::from(rec.psf_flux_njy);
    let photometry = Photometry {
        magnitude: flux_njy_to_ab_mag(flux),
        error: flux_err_to_mag_err(flux, f64::from(rec.psf_flux_err_njy)),
        filter,
    };

    let input = ObservationInput::new(
        id,
        EquCoord::new(rec.ra_deg.to_radians(), ra_error, dec, dec_error),
        photometry,
        mjd_tai_to_tt(rec.midpoint_mjd_tai),
        Some(ObserverId::MpcCode(RUBIN_MPC_CODE)),
    );

    Ok(LsstObservation {
        input,
        night_id,
        visit: rec.visit,
    })
}

/// Reject any NaN or infinite floating-point input.
fn check_finite(
    rec: &DiaSourceRecord<'_>,
    params: &LsstConversionParams,
) -> Result<(), LsstAlertError> {
    [
        ("midpointMjdTai", rec.midpoint_mjd_tai),
        ("ra", rec.ra_deg),
        ("dec", rec.dec_deg),
        ("raErr", f64::from(rec.ra_err_deg)),
        ("decErr", f64::from(rec.dec_err_deg)),
        ("psfFlux", f64::from(rec.psf_flux_njy)),
        ("psfFluxErr", f64::from(rec.psf_flux_err_njy)),
        ("astrometric_floor_arcsec", params.astrometric_floor_arcsec),
    ]
    .into_iter()
    .find(|(_, v)| !v.is_finite())
    .map_or(Ok(()), |(field, _)| {
        Err(LsstAlertError::NonFinite { field })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{observation_dataset::ObsDataset, photometry::Filter};
    use approx::assert_relative_eq;

    /// A valid DiaSource observed on 2026-10-31 at 03:00 UTC (night 61343).
    pub(super) fn sample_record() -> DiaSourceRecord<'static> {
        DiaSourceRecord {
            dia_source_id: 170_000_000_000_042,
            visit: 2_026_103_000_508,
            midpoint_mjd_tai: 61_344.125,
            ra_deg: 30.0,
            dec_deg: -60.0,
            ra_err_deg: (0.1 / 3600.0) as f32,
            dec_err_deg: (0.1 / 3600.0) as f32,
            psf_flux_njy: 1000.0,
            psf_flux_err_njy: 50.0,
            band: "r",
        }
    }

    fn convert(rec: &DiaSourceRecord<'_>) -> Result<LsstObservation, LsstAlertError> {
        dia_source_to_observation(rec, &LsstConversionParams::default())
    }

    #[test]
    fn full_conversion() {
        let rec = sample_record();
        let out = convert(&rec).unwrap();

        assert_eq!(out.night_id, NightId(61_343));
        assert_eq!(out.visit, rec.visit);

        let obs = &out.input;
        assert_eq!(obs.id, 170_000_000_000_042);
        assert_eq!(obs.observer, Some(ObserverId::MpcCode(*b"X05")));
        assert_relative_eq!(obs.equ_coord.ra, 30f64.to_radians());
        assert_relative_eq!(obs.equ_coord.dec, (-60f64).to_radians());
        let sky_err = f64::from(rec.ra_err_deg).to_radians();
        assert_relative_eq!(obs.equ_coord.ra_error, 2.0 * sky_err, max_relative = 1e-12);
        assert_relative_eq!(obs.equ_coord.dec_error, sky_err);
        assert_relative_eq!(obs.photometry.magnitude, 31.4 - 7.5);
        assert_relative_eq!(obs.photometry.error, 2.5 / std::f64::consts::LN_10 * 0.05);
        assert_eq!(obs.photometry.filter, Filter::Int(2));
        assert_relative_eq!(
            (obs.mjd_tt - rec.midpoint_mjd_tai) * 86_400.0,
            32.184,
            epsilon = 1e-5
        );
    }

    #[test]
    fn floor_is_added_on_sky_before_cos_dec() {
        let mut rec = sample_record();
        rec.ra_err_deg = (0.3 / 3600.0) as f32;
        rec.dec_err_deg = (0.3 / 3600.0) as f32;
        let params = LsstConversionParams {
            astrometric_floor_arcsec: 0.4,
        };
        let obs = dia_source_to_observation(&rec, &params).unwrap().input;
        let half_arcsec = 0.5 * ARCSEC_TO_RAD;
        assert_relative_eq!(obs.equ_coord.dec_error, half_arcsec, max_relative = 1e-6);
        assert_relative_eq!(
            obs.equ_coord.ra_error,
            2.0 * half_arcsec,
            max_relative = 1e-6
        );
    }

    #[test]
    fn night_mismatch_is_rejected() {
        let mut rec = sample_record();
        rec.midpoint_mjd_tai = 61_345.125; // one night later than the visit
        assert_eq!(
            convert(&rec).unwrap_err(),
            LsstAlertError::NightMismatch {
                visit_night: NightId(61_343),
                mjd_night: 61_344,
            }
        );
    }

    #[test]
    fn each_conversion_error() {
        let base = sample_record();
        let cases: Vec<(DiaSourceRecord<'_>, LsstAlertError)> = vec![
            (
                DiaSourceRecord {
                    psf_flux_njy: 0.0,
                    ..base
                },
                LsstAlertError::NonPositiveFlux(0.0),
            ),
            (
                DiaSourceRecord {
                    psf_flux_njy: -3.0,
                    ..base
                },
                LsstAlertError::NonPositiveFlux(-3.0),
            ),
            (
                DiaSourceRecord {
                    ra_err_deg: f32::NAN,
                    ..base
                },
                LsstAlertError::NonFinite { field: "raErr" },
            ),
            (
                DiaSourceRecord {
                    midpoint_mjd_tai: f64::INFINITY,
                    ..base
                },
                LsstAlertError::NonFinite {
                    field: "midpointMjdTai",
                },
            ),
            (
                DiaSourceRecord {
                    dec_err_deg: -1e-5,
                    ..base
                },
                LsstAlertError::NegativeErr {
                    field: "decErr",
                    value: f64::from(-1e-5f32),
                },
            ),
            (
                DiaSourceRecord { band: "w", ..base },
                LsstAlertError::UnknownBand("w".to_string()),
            ),
            (
                DiaSourceRecord {
                    dia_source_id: -1,
                    ..base
                },
                LsstAlertError::NegativeId(-1),
            ),
            (
                DiaSourceRecord { visit: 508, ..base },
                LsstAlertError::InvalidVisit(508),
            ),
            (
                DiaSourceRecord {
                    visit: 2_026_133_000_508,
                    ..base
                },
                LsstAlertError::InvalidDayObs(20_261_330),
            ),
            (
                DiaSourceRecord {
                    ra_deg: 360.0,
                    ..base
                },
                LsstAlertError::CoordinateOutOfRange {
                    field: "ra",
                    value: 360.0,
                },
            ),
            (
                DiaSourceRecord {
                    dec_deg: -90.5,
                    ..base
                },
                LsstAlertError::CoordinateOutOfRange {
                    field: "dec",
                    value: -90.5,
                },
            ),
            (
                DiaSourceRecord {
                    dec_deg: 90.0,
                    ..base
                },
                LsstAlertError::CoordinateOutOfRange {
                    field: "dec",
                    value: 90.0,
                },
            ),
        ];
        for (rec, expected) in cases {
            assert_eq!(convert(&rec).unwrap_err(), expected, "record: {rec:?}");
        }

        let bad_floor = LsstConversionParams {
            astrometric_floor_arcsec: -0.1,
        };
        assert_eq!(
            dia_source_to_observation(&base, &bad_floor).unwrap_err(),
            LsstAlertError::NegativeErr {
                field: "astrometric_floor_arcsec",
                value: -0.1,
            }
        );
    }

    #[test]
    fn converted_observation_round_trips_through_dataset() {
        let rec = sample_record();
        let out = convert(&rec).unwrap();
        let expected = out.input.clone();

        let (ds, idx) = ObsDataset::empty()
            .push_observation(vec![out.input])
            .unwrap();
        assert_eq!(idx, vec![0]);

        let obs = ds.get_observation(expected.id).unwrap();
        assert_eq!(*obs.id(), expected.id);
        assert_eq!(obs.equ_coord(), &expected.equ_coord);
        assert_eq!(obs.photometry(), &expected.photometry);
        assert_eq!(obs.mjd_tt(), expected.mjd_tt);
    }
}
