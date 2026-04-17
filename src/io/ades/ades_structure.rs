use std::str::FromStr;

use hifitime::Epoch;
use serde::{Deserialize, Deserializer};

use crate::TrajId;

// ---------------------------------------------------------------------------
// XML deserialization structures (ADES format)
// ---------------------------------------------------------------------------

/// Top-level ADES document in "structured" form: one or more `<obsBlock>` elements,
/// each pairing observation context (observatory) with observation data.
#[derive(Debug, Deserialize)]
pub(crate) struct StructuredAdes {
    #[serde(rename = "obsBlock")]
    pub(crate) obs_blocks: Vec<ObsBlock>,
}

/// Top-level ADES document in "flat" form: all `<optical>` elements at the top
/// level, each carrying its own `<stn>` field.
#[derive(Debug, Deserialize)]
pub(crate) struct FlatAdes {
    #[serde(rename = "optical")]
    pub(crate) opticals: Vec<OpticalObs>,
}

/// A logical block grouping context metadata with the associated observations.
#[derive(Debug, Deserialize)]
pub(crate) struct ObsBlock {
    #[serde(rename = "obsContext")]
    pub(crate) obs_context: ObsContext,

    #[serde(rename = "obsData")]
    pub(crate) obs_data: ObsData,
}

/// Context metadata for a block of observations.
#[derive(Debug, Deserialize)]
pub(crate) struct ObsContext {
    pub(crate) observatory: Observatory,
}

/// Observatory identification inside an observation context.
#[derive(Debug, Deserialize)]
pub(crate) struct Observatory {
    #[serde(rename = "mpcCode")]
    pub(crate) mpc_code: String,
}

/// Container for the optical observations inside a block.
#[derive(Debug, Deserialize)]
pub(crate) struct ObsData {
    #[serde(rename = "optical")]
    pub(crate) opticals: Vec<OpticalObs>,
}

/// A single optical astrometric observation from an ADES file.
///
/// Fields follow the ADES XML element names:
/// - Object identity: `permID`, `provID`, or `trkSub` (first non-null wins).
/// - Epoch: `obsTime` (ISO-8601 / UTC string), deserialized as MJD TT.
/// - Position: `ra`, `dec` in degrees.
/// - Position uncertainties: `rmsRA`/`rmsDec` (arcsec, 1-σ, highest priority)
///   or `precRA`/`precDec` (arcsec, precision-based fallback).
/// - Observatory: `stn` (3-char MPC code).
/// - Photometry: `mag`, `rmsMag`, `band` (all optional).
#[derive(Debug, Deserialize)]
pub(crate) struct OpticalObs {
    /// Permanent object number (highest identity priority).
    #[serde(rename = "permID")]
    pub(crate) perm_id: Option<String>,
    /// Provisional designation (second priority).
    #[serde(rename = "provID")]
    pub(crate) prov_id: Option<String>,
    /// Tracklet sub-identifier (fallback identity).
    #[serde(rename = "trkSub")]
    pub(crate) trk_sub: Option<String>,

    /// Observation epoch in MJD TT, deserialized from an ISO-8601 UTC string.
    #[serde(rename = "obsTime", deserialize_with = "deserialize_mjd_tt")]
    pub(crate) obs_time: f64,

    /// Right ascension in degrees.
    pub(crate) ra: f64,
    /// Declination in degrees.
    pub(crate) dec: f64,

    /// Precision-based RA uncertainty in arcseconds (fallback when `rmsRA` is absent).
    #[serde(rename = "precRA")]
    pub(crate) prec_ra: Option<f64>,
    /// Precision-based Dec uncertainty in arcseconds (fallback when `rmsDec` is absent).
    #[serde(rename = "precDec")]
    pub(crate) prec_dec: Option<f64>,

    /// 1-σ RA uncertainty in arcseconds (preferred over `precRA`).
    #[serde(rename = "rmsRA")]
    pub(crate) rms_ra: Option<f64>,
    /// 1-σ Dec uncertainty in arcseconds (preferred over `precDec`).
    #[serde(rename = "rmsDec")]
    pub(crate) rms_dec: Option<f64>,

    /// MPC observatory code (3-character ASCII).
    pub(crate) stn: String,

    /// Apparent magnitude (optional).
    pub(crate) mag: Option<f64>,
    /// 1-σ magnitude uncertainty (optional).
    #[serde(rename = "rmsMag")]
    pub(crate) rms_mag: Option<f64>,
    /// Photometric filter label (optional).
    pub(crate) band: Option<String>,
}

impl OpticalObs {
    /// Return the trajectory identifier for this observation, or `None` if
    /// none of `permID`, `provID`, or `trkSub` is set.
    ///
    /// Precedence: `permID` > `provID` > `trkSub`.
    ///
    /// If the resolved identifier parses as a `u32` it is returned as
    /// `TrajId::Int`; otherwise as `TrajId::Str`.
    pub(crate) fn traj_id(&self) -> Option<TrajId> {
        let id = self
            .perm_id
            .clone()
            .or_else(|| self.prov_id.clone())
            .or_else(|| self.trk_sub.clone())?;

        if let Ok(n) = id.parse::<u32>() {
            Some(TrajId::Int(n))
        } else {
            Some(TrajId::Str(id))
        }
    }

    /// Resolve the RA uncertainty in arcseconds, or `None` if none of
    /// `rmsRA`, `precRA`, or the supplied `fallback` is present.
    ///
    /// Priority: `rmsRA` > `precRA` > `fallback`.
    pub(crate) fn ra_error_arcsec(&self, fallback: Option<f64>) -> Option<f64> {
        self.rms_ra.or(self.prec_ra).or(fallback)
    }

    /// Resolve the Dec uncertainty in arcseconds, or `None` if none of
    /// `rmsDec`, `precDec`, or the supplied `fallback` is present.
    ///
    /// Priority: `rmsDec` > `precDec` > `fallback`.
    pub(crate) fn dec_error_arcsec(&self, fallback: Option<f64>) -> Option<f64> {
        self.rms_dec.or(self.prec_dec).or(fallback)
    }

    /// Convert the `stn` field to a 3-byte MPC code array.
    ///
    /// If the station code is shorter than three bytes it is left-padded with
    /// spaces; bytes beyond the third are silently discarded.
    pub(crate) fn mpc_code_bytes(&self) -> [u8; 3] {
        let bytes = self.stn.as_bytes();
        let mut code = [b' '; 3];
        let len = bytes.len().min(3);
        code[..len].copy_from_slice(&bytes[..len]);
        code
    }
}

// ---------------------------------------------------------------------------
// Custom deserializer: ISO-8601 UTC string → MJD (TT)
// ---------------------------------------------------------------------------

/// Deserialize an ISO-8601 date/time string (UTC) into a Modified Julian Date
/// expressed in Terrestrial Time (TT).
fn deserialize_mjd_tt<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let date_str = String::deserialize(deserializer)?;
    let epoch = Epoch::from_str(&date_str).map_err(serde::de::Error::custom)?;
    Ok(epoch.to_mjd_tt_days())
}
