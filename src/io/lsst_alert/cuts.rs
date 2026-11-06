//! Optional science-quality cuts on Rubin DiaSources.
//!
//! These cuts are a **policy**, kept apart from the conversion: a DiaSource
//! that fails them is perfectly convertible, the caller simply chooses not to
//! track it.  Thresholds are therefore supplied by the application (e.g. the
//! fink-fat configuration), and [`LsstQualityCuts::default`] applies none.
//!
//! The SNR is recomputed as `psfFlux / psfFluxErr` (the `snr` column of the
//! alert is not used, so the cut behaves identically on every data source).
//! Astrometric ceilings apply to the **on-sky** errors `raErr` / `decErr` as
//! published by Rubin, before any `1 / cos δ` conversion or floor.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`LsstQualityCuts`] | struct | Optional SNR and astrometric-error thresholds |
//! | [`CutReason`] | enum | Why a DiaSource was cut |

use thiserror::Error;

use crate::{Arcseconds, constants::ARCSEC_TO_DEG};

use super::DiaSourceRecord;

/// Optional quality thresholds applied to a [`DiaSourceRecord`].
///
/// Every threshold is `None` by default, i.e. the default value cuts nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LsstQualityCuts {
    /// Keep only sources with `psfFlux / psfFluxErr` **strictly** above this value.
    pub min_snr: Option<f64>,
    /// Keep only sources with on-sky `raErr` at most this value, in arcseconds.
    pub max_ra_err_arcsec: Option<Arcseconds>,
    /// Keep only sources with `decErr` at most this value, in arcseconds.
    pub max_dec_err_arcsec: Option<Arcseconds>,
}

/// The reason a DiaSource was rejected by [`LsstQualityCuts::check`].
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum CutReason {
    /// `psfFlux / psfFluxErr` is not strictly above the threshold (NaN included).
    #[error("SNR {snr} not above {min}")]
    LowSnr {
        /// Recomputed SNR of the source.
        snr: f64,
        /// Configured minimum.
        min: f64,
    },
    /// On-sky `raErr` exceeds the ceiling (or is NaN).
    #[error("raErr {value_arcsec}\" above {max_arcsec}\"")]
    RaErrTooLarge {
        /// `raErr` of the source, in arcseconds.
        value_arcsec: Arcseconds,
        /// Configured ceiling, in arcseconds.
        max_arcsec: Arcseconds,
    },
    /// `decErr` exceeds the ceiling (or is NaN).
    #[error("decErr {value_arcsec}\" above {max_arcsec}\"")]
    DecErrTooLarge {
        /// `decErr` of the source, in arcseconds.
        value_arcsec: Arcseconds,
        /// Configured ceiling, in arcseconds.
        max_arcsec: Arcseconds,
    },
}

impl LsstQualityCuts {
    /// Check a DiaSource against the configured thresholds.
    ///
    /// Thresholds are evaluated in the order SNR, `raErr`, `decErr`; the
    /// first failing one is reported.  Comparisons are written so that a NaN
    /// value never passes an active cut.
    ///
    /// # Arguments
    /// - `rec` — the DiaSource to check.
    ///
    /// # Returns
    /// `Ok(())` when every active threshold passes.
    ///
    /// # Errors
    /// The [`CutReason`] of the first failing threshold.
    pub fn check(&self, rec: &DiaSourceRecord<'_>) -> Result<(), CutReason> {
        if let Some(min) = self.min_snr {
            let snr = f64::from(rec.psf_flux_njy) / f64::from(rec.psf_flux_err_njy);
            if snr.is_nan() || snr <= min {
                return Err(CutReason::LowSnr { snr, min });
            }
        }
        if let Some(max_arcsec) = self.max_ra_err_arcsec {
            let value_arcsec = deg_to_arcsec(rec.ra_err_deg);
            if value_arcsec.is_nan() || value_arcsec > max_arcsec {
                return Err(CutReason::RaErrTooLarge {
                    value_arcsec,
                    max_arcsec,
                });
            }
        }
        if let Some(max_arcsec) = self.max_dec_err_arcsec {
            let value_arcsec = deg_to_arcsec(rec.dec_err_deg);
            if value_arcsec.is_nan() || value_arcsec > max_arcsec {
                return Err(CutReason::DecErrTooLarge {
                    value_arcsec,
                    max_arcsec,
                });
            }
        }
        Ok(())
    }
}

/// Degrees (as stored in the alert, `f32`) → arcseconds.
#[inline]
fn deg_to_arcsec(deg: f32) -> Arcseconds {
    f64::from(deg) / ARCSEC_TO_DEG
}

#[cfg(test)]
mod tests {
    use super::super::tests::sample_record;
    use super::*;

    #[test]
    fn default_cuts_nothing() {
        let mut rec = sample_record();
        rec.psf_flux_njy = 1.0;
        rec.psf_flux_err_njy = 100.0;
        rec.ra_err_deg = 1.0;
        assert_eq!(LsstQualityCuts::default().check(&rec), Ok(()));
    }

    #[test]
    fn snr_cut_is_strict() {
        let cuts = LsstQualityCuts {
            min_snr: Some(5.0),
            ..Default::default()
        };
        let mut rec = sample_record();
        rec.psf_flux_njy = 50.0;
        rec.psf_flux_err_njy = 10.0;
        assert_eq!(
            cuts.check(&rec),
            Err(CutReason::LowSnr { snr: 5.0, min: 5.0 })
        );
        rec.psf_flux_njy = 50.5;
        assert_eq!(cuts.check(&rec), Ok(()));
    }

    #[test]
    fn nan_snr_is_cut() {
        let cuts = LsstQualityCuts {
            min_snr: Some(5.0),
            ..Default::default()
        };
        let mut rec = sample_record();
        rec.psf_flux_err_njy = f32::NAN;
        assert!(matches!(cuts.check(&rec), Err(CutReason::LowSnr { .. })));
    }

    #[test]
    fn astrometric_ceilings() {
        let cuts = LsstQualityCuts {
            max_ra_err_arcsec: Some(0.5),
            max_dec_err_arcsec: Some(0.5),
            ..Default::default()
        };
        let mut rec = sample_record();
        rec.ra_err_deg = (0.4 * ARCSEC_TO_DEG) as f32;
        rec.dec_err_deg = (0.4 * ARCSEC_TO_DEG) as f32;
        assert_eq!(cuts.check(&rec), Ok(()));

        rec.ra_err_deg = (0.6 * ARCSEC_TO_DEG) as f32;
        assert!(matches!(
            cuts.check(&rec),
            Err(CutReason::RaErrTooLarge { .. })
        ));

        rec.ra_err_deg = (0.4 * ARCSEC_TO_DEG) as f32;
        rec.dec_err_deg = (0.6 * ARCSEC_TO_DEG) as f32;
        assert!(matches!(
            cuts.check(&rec),
            Err(CutReason::DecErrTooLarge { .. })
        ));
    }
}
