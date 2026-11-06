//! Error type for the LSST alert conversion.
//!
//! Every variant describes a DiaSource that **cannot be converted** into an
//! [`ObservationInput`](crate::observation_dataset::observation::ObservationInput)
//! without inventing data.  Science-quality rejections (SNR, error ceilings)
//! are not errors: they belong to the
//! [`LsstQualityCuts`](super::cuts::LsstQualityCuts) policy.

use thiserror::Error;

use crate::NightId;

/// Errors raised while converting a Rubin `DiaSource` into a photom observation.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum LsstAlertError {
    /// `psfFlux` is zero or negative: the AB magnitude is undefined.
    #[error("non-positive psfFlux ({0} nJy): AB magnitude is undefined")]
    NonPositiveFlux(f32),

    /// A floating-point field (or a derived quantity) is NaN or infinite.
    #[error("non-finite value for `{field}`")]
    NonFinite {
        /// Name of the offending field, in Rubin APDB spelling when it is an
        /// input field (e.g. `"raErr"`).
        field: &'static str,
    },

    /// An uncertainty (or the astrometric floor) is negative.
    #[error("negative uncertainty for `{field}`: {value}")]
    NegativeErr {
        /// Name of the offending field.
        field: &'static str,
        /// The negative value as supplied.
        value: f64,
    },

    /// A sky coordinate lies outside its valid range
    /// (`ra` ∈ [0°, 360°), `dec` ∈ (−90°, 90°): RA is undefined at the poles).
    #[error("`{field}` out of range: {value}°")]
    CoordinateOutOfRange {
        /// `"ra"` or `"dec"`.
        field: &'static str,
        /// The value as supplied, in degrees.
        value: f64,
    },

    /// The `band` string is not one of the six LSST filters `u g r i z y`.
    #[error("unknown LSST band {0:?} (expected one of u, g, r, i, z, y)")]
    UnknownBand(String),

    /// `diaSourceId` is negative and cannot be used as an
    /// [`ObsId`](crate::observation_dataset::ObsId).
    #[error("negative diaSourceId {0}")]
    NegativeId(i64),

    /// `visit` does not follow the `dayObs × 100000 + seqNum` encoding.
    #[error("visit {0} does not encode a dayObs (expected dayObs × 100000 + seqNum)")]
    InvalidVisit(i64),

    /// The `YYYYMMDD` dayObs is not a valid calendar date on or after
    /// 1858-11-17 (MJD 0).
    #[error("dayObs {0} is not a valid YYYYMMDD date on or after 1858-11-17")]
    InvalidDayObs(u32),

    /// The night encoded in `visit` disagrees with the night derived from
    /// `midpointMjdTai` (`floor(mjd − 0.5)`).
    #[error("night mismatch: visit encodes night {visit_night}, midpointMjdTai gives {mjd_night}")]
    NightMismatch {
        /// Night derived from the `visit` field (MJD of `dayObs`).
        visit_night: NightId,
        /// Night derived from `midpointMjdTai`.
        mjd_night: i64,
    },
}
