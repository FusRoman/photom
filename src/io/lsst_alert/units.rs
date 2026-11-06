//! Pure unit conversions from Rubin conventions to photom conventions.
//!
//! Every function here is a total or fallible **pure** function with no I/O:
//! flux → AB magnitude, TAI → TT, LSST band → [`Filter`], on-sky RA error →
//! RA-coordinate error, and quadrature floor.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`AB_ZERO_POINT_NJY`] | const | AB zero point for fluxes in nJy (31.4) |
//! | [`TT_MINUS_TAI_DAYS`] | const | TT − TAI = 32.184 s, in days |
//! | [`LSST_BANDS`] | const | The six LSST bands in filter-code order |
//! | [`flux_njy_to_ab_mag`] | fn | nJy flux → AB magnitude |
//! | [`flux_err_to_mag_err`] | fn | flux uncertainty → magnitude uncertainty |
//! | [`mjd_tai_to_tt`] | fn | MJD (TAI) → MJD (TT) |
//! | [`band_to_filter`] | fn | `"u".."y"` → [`Filter::Int`] `0..=5` |
//! | [`ra_coordinate_error`] | fn | on-sky RA error σ·cos δ → error on α |
//! | [`with_floor`] | fn | quadrature sum √(σ² + floor²) |

use std::f64::consts::LN_10;

use crate::{Radians, photometry::Filter};

use super::error::LsstAlertError;

/// AB-magnitude zero point for fluxes expressed in nanojansky.
///
/// $m_{AB} = -2.5 \log_{10}(F / 3631\,\mathrm{Jy})$, so with $F$ in nJy the
/// zero point is $2.5 \log_{10}(3631 \times 10^{9}) \approx 31.4$.  The
/// rounded value `31.4` is the Rubin convention and the one used by the
/// fink-fat batch pipeline.
pub const AB_ZERO_POINT_NJY: f64 = 31.4;

/// Difference TT − TAI, in days (exactly 32.184 s).
pub const TT_MINUS_TAI_DAYS: f64 = 32.184 / 86_400.0;

/// The six LSST bands, in the order of their numeric filter code
/// (`u` → 0, …, `y` → 5).
pub const LSST_BANDS: [&str; 6] = ["u", "g", "r", "i", "z", "y"];

/// Convert a flux in nanojansky into an AB magnitude.
///
/// $m = 31.4 - 2.5 \log_{10}(F)$.
///
/// # Arguments
/// - `flux_njy` — PSF flux in nJy; must be strictly positive for a finite
///   result (zero gives `+∞`, a negative flux gives NaN).
///
/// # Returns
/// The AB magnitude.
#[inline]
pub fn flux_njy_to_ab_mag(flux_njy: f64) -> f64 {
    AB_ZERO_POINT_NJY - 2.5 * flux_njy.log10()
}

/// Propagate a flux uncertainty to a magnitude uncertainty (first order).
///
/// $\sigma_m = \frac{2.5}{\ln 10} \frac{\sigma_F}{F}$, i.e. $1.0857 / \mathrm{SNR}$.
///
/// # Arguments
/// - `flux` — flux, any unit; must be strictly positive.
/// - `flux_err` — 1-σ flux uncertainty, same unit as `flux`.
///
/// # Returns
/// The 1-σ magnitude uncertainty.
#[inline]
pub fn flux_err_to_mag_err(flux: f64, flux_err: f64) -> f64 {
    2.5 / LN_10 * (flux_err / flux)
}

/// Convert an MJD expressed in TAI into an MJD expressed in TT.
///
/// # Arguments
/// - `mjd_tai` — Modified Julian Date in the TAI time scale, in days.
///
/// # Returns
/// The same instant as an MJD in Terrestrial Time (`mjd_tai + 32.184 s`).
#[inline]
pub fn mjd_tai_to_tt(mjd_tai: f64) -> f64 {
    mjd_tai + TT_MINUS_TAI_DAYS
}

/// Map an LSST band name to its numeric [`Filter`] code.
///
/// The mapping follows [`LSST_BANDS`]: `u g r i z y` → `Filter::Int(0..=5)`,
/// the same encoding as the `filter` column of the fink-fat batch parquet.
///
/// # Arguments
/// - `band` — band name as found in the alert (exact, lower-case).
///
/// # Errors
/// [`LsstAlertError::UnknownBand`] for any other string.
pub fn band_to_filter(band: &str) -> Result<Filter, LsstAlertError> {
    LSST_BANDS
        .iter()
        .position(|b| *b == band)
        .map(|code| Filter::Int(code as u32))
        .ok_or_else(|| LsstAlertError::UnknownBand(band.to_string()))
}

/// Convert an on-sky RA uncertainty into an uncertainty on the α coordinate.
///
/// Rubin's `raErr` is the angular error along the RA direction on the sky
/// ($\sigma_\alpha \cos\delta$), whereas
/// [`EquCoord::ra_error`](crate::coordinates::equatorial::EquCoord::ra_error)
/// is the error on the coordinate α itself.  Hence
/// $\sigma_\alpha = \sigma_{\mathrm{sky}} / \cos\delta$.
///
/// # Arguments
/// - `ra_sky_err` — on-sky RA uncertainty, in radians.
/// - `dec` — declination, in radians.
///
/// # Returns
/// The α-coordinate uncertainty in radians.  It diverges at the poles
/// (`cos δ → 0`); callers must check finiteness.
#[inline]
pub fn ra_coordinate_error(ra_sky_err: Radians, dec: Radians) -> Radians {
    ra_sky_err / dec.cos()
}

/// Add a systematic floor in quadrature to a statistical uncertainty.
///
/// # Arguments
/// - `sigma` — statistical 1-σ uncertainty.
/// - `floor` — systematic floor, same unit as `sigma`.
///
/// # Returns
/// $\sqrt{\sigma^2 + \mathrm{floor}^2}$; equals `|sigma|` when `floor` is 0.
#[inline]
pub fn with_floor(sigma: f64, floor: f64) -> f64 {
    sigma.hypot(floor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    #[test]
    fn magnitude_zero_point() {
        assert_relative_eq!(flux_njy_to_ab_mag(1.0), 31.4);
        // 3631 Jy = 3631e9 nJy is magnitude ≈ 0 (31.4 is a rounded zero point).
        assert!(flux_njy_to_ab_mag(3631e9).abs() < 1e-2);
        assert_relative_eq!(flux_njy_to_ab_mag(100.0), 26.4);
    }

    #[test]
    fn magnitude_error_is_1_0857_over_snr() {
        assert_relative_eq!(flux_err_to_mag_err(100.0, 10.0), 0.108_573_620_475_812_9);
    }

    #[test]
    fn tt_minus_tai_is_32_184_s() {
        let dt = mjd_tai_to_tt(61_343.25) - 61_343.25;
        assert_relative_eq!(dt * 86_400.0, 32.184, epsilon = 1e-6);
    }

    #[test]
    fn six_bands_map_to_0_to_5() {
        for (code, band) in LSST_BANDS.iter().enumerate() {
            assert_eq!(band_to_filter(band), Ok(Filter::Int(code as u32)));
        }
    }

    #[test]
    fn unknown_band_is_rejected() {
        for bad in ["", "G", "U", "Y", "ug", "w"] {
            assert_eq!(
                band_to_filter(bad),
                Err(LsstAlertError::UnknownBand(bad.to_string()))
            );
        }
    }

    #[test]
    fn ra_error_doubles_at_dec_60() {
        let sigma = 1e-6;
        assert_relative_eq!(
            ra_coordinate_error(sigma, 60f64.to_radians()),
            2.0 * sigma,
            max_relative = 1e-12
        );
        assert_relative_eq!(ra_coordinate_error(sigma, 0.0), sigma);
    }

    #[test]
    fn floor_in_quadrature() {
        assert_relative_eq!(with_floor(3.0, 4.0), 5.0);
        assert_relative_eq!(with_floor(3.0, 0.0), 3.0);
    }

    proptest! {
        /// σ_α ≥ σ_sky for any declination strictly inside (−90°, 90°).
        #[test]
        fn prop_ra_coordinate_error_ge_sky_error(
            sigma in 0.0f64..1e-3,
            dec_deg in -89.9f64..89.9,
        ) {
            prop_assert!(ra_coordinate_error(sigma, dec_deg.to_radians()) >= sigma);
        }

        /// Brighter flux → smaller magnitude.
        #[test]
        fn prop_magnitude_strictly_decreasing(
            f in 1e-3f64..1e12,
            factor in 1.001f64..1e3,
        ) {
            prop_assert!(flux_njy_to_ab_mag(f * factor) < flux_njy_to_ab_mag(f));
        }

        /// The floor never decreases the uncertainty.
        #[test]
        fn prop_floor_never_decreases(sigma in 0.0f64..1.0, floor in 0.0f64..1.0) {
            prop_assert!(with_floor(sigma, floor) >= sigma);
        }
    }
}
