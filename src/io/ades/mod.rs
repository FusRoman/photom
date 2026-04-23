#![cfg(feature = "ades")]

pub mod ades_structure;

use ahash::AHashMap;
use camino::Utf8Path;
use quick_xml::de::from_str;
use thiserror::Error;

use crate::{
    Arcseconds, Degrees, TrajId,
    constants::ARCSEC_TO_DEG,
    coordinates::equatorial::EquCoord,
    observation_dataset::ObsId,
    observation_dataset::{
        ObsDataset,
        index::{ObsMapIndex, TrajIndexMap},
        observation::Observation,
    },
    observer::dataset::ObserverId,
    photometry::{Filter, Photometry},
};

use ades_structure::{FlatAdes, OpticalObs, StructuredAdes};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while reading an ADES XML observation file.
#[derive(Debug, Error)]
pub enum AdesError {
    /// The file could not be opened or read.
    #[error("I/O error reading ADES file: {0}")]
    Io(#[from] std::io::Error),

    /// The XML could not be parsed as either the flat or structured ADES variant.
    #[error(
        "failed to parse ADES document:\n  flat error: {flat_err}\n  structured error: {structured_err}"
    )]
    ParseXml {
        flat_err: String,
        structured_err: String,
    },

    /// An observation has no `permID`, `provID`, or `trkSub` field.
    #[error("ADES observation #{index} has no permID, provID, or trkSub")]
    MissingTrajId { index: usize },

    /// An observation has no RA uncertainty and no fallback was provided.
    #[error(
        "ADES observation #{index} has no RA uncertainty (rmsRA / precRA) and no fallback was provided"
    )]
    MissingRaError { index: usize },

    /// An observation has no Dec uncertainty and no fallback was provided.
    #[error(
        "ADES observation #{index} has no Dec uncertainty (rmsDec / precDec) and no fallback was provided"
    )]
    MissingDecError { index: usize },
}

// ---------------------------------------------------------------------------
// Public (crate-level) entry point
// ---------------------------------------------------------------------------

/// Parse an ADES XML file and return a fully built [`ObsDataset`].
///
/// Both the *structured* variant (observations inside `<obsBlock>` elements
/// each with an `<obsContext>`) and the *flat* variant (all `<optical>`
/// elements at document root with per-observation `<stn>` fields) are
/// supported.  The function tries the flat variant first.
///
/// # Arguments
///
/// - `ades_path` — path to the ADES XML file.
/// - `error_ra`  — optional fallback 1-σ RA uncertainty in **arcseconds**,
///   used for any observation that has neither `rmsRA` nor `precRA`.
/// - `error_dec` — optional fallback 1-σ Dec uncertainty in **arcseconds**,
///   used for any observation that has neither `rmsDec` nor `precDec`.
/// - `start_id`  — starting value for observation IDs; the first observation will have ID `start_id`, the second `start_id + 1`, and so on.
///
/// # Errors
///
/// Returns [`AdesError::Io`] if the file cannot be read,
/// [`AdesError::ParseXml`] if neither ADES variant parses successfully, or
/// [`AdesError::MissingTrajId`] / [`AdesError::MissingRaError`] /
/// [`AdesError::MissingDecError`] if a required field is absent.
pub(crate) fn parse_ades_file(
    ades_path: &Utf8Path,
    error_ra: Option<f64>,
    error_dec: Option<f64>,
    start_id: ObsId,
) -> Result<ObsDataset, AdesError> {
    let xml = std::fs::read_to_string(ades_path)?;
    parse_ades_xml(&xml, error_ra, error_dec, start_id)
}

/// Build an [`ObsDataset`] from an in-memory ADES XML string.
///
/// Same semantics as [`parse_ades_file`] but accepts an already-loaded string.
pub(crate) fn parse_ades_xml(
    xml: &str,
    error_ra: Option<f64>,
    error_dec: Option<f64>,
    start_id: ObsId,
) -> Result<ObsDataset, AdesError> {
    match from_str::<FlatAdes>(xml) {
        Ok(flat) => build_from_flat(&flat.opticals, error_ra, error_dec, start_id),
        Err(flat_err) => match from_str::<StructuredAdes>(xml) {
            Ok(structured) => build_from_structured(&structured, error_ra, error_dec, start_id),
            Err(structured_err) => Err(AdesError::ParseXml {
                flat_err: flat_err.to_string(),
                structured_err: structured_err.to_string(),
            }),
        },
    }
}

// ---------------------------------------------------------------------------
// Builders for each ADES variant
// ---------------------------------------------------------------------------

/// Build from a flat list of optical observations (no `obsContext`; each
/// observation carries its own `stn` field).
fn build_from_flat(
    opticals: &[OpticalObs],
    error_ra: Option<f64>,
    error_dec: Option<f64>,
    start_id: ObsId,
) -> Result<ObsDataset, AdesError> {
    let records: Vec<(&OpticalObs, Option<[u8; 3]>)> = opticals.iter().map(|o| (o, None)).collect();
    build_dataset(records, error_ra, error_dec, start_id)
}

/// Build from a structured ADES document, propagating the `obsContext` MPC
/// code to every observation within the same block.
fn build_from_structured(
    structured: &StructuredAdes,
    error_ra: Option<f64>,
    error_dec: Option<f64>,
    start_id: ObsId,
) -> Result<ObsDataset, AdesError> {
    let mut records: Vec<(&OpticalObs, Option<[u8; 3]>)> = Vec::new();
    for block in &structured.obs_blocks {
        let ctx_bytes = mpc_str_to_bytes(&block.obs_context.observatory.mpc_code);
        for optical in &block.obs_data.opticals {
            records.push((optical, Some(ctx_bytes)));
        }
    }
    build_dataset(records, error_ra, error_dec, start_id)
}

// ---------------------------------------------------------------------------
// Core conversion: optical records → ObsDataset
// ---------------------------------------------------------------------------

/// Convert a slice of `(optical, optional_context_mpc_code)` pairs into an
/// [`ObsDataset`].
///
/// Coordinate conventions:
/// - RA/Dec positions are stored in **radians** (via [`EquCoord::from_degrees`]).
/// - ADES uncertainties (arcseconds) are divided by 3 600 to obtain degrees
///   before being forwarded to [`EquCoord::from_degrees`].
/// - Epochs are already in MJD TT (the `obsTime` deserializer handles the
///   UTC → TT conversion).
/// - Observers are stored as [`ObserverId::MpcCode`] (`[u8; 3]`).
/// - Trajectory identifiers are indexed in the returned dataset so that
///   trajectory-based look-ups work immediately.
fn build_dataset(
    records: Vec<(&OpticalObs, Option<[u8; 3]>)>,
    error_ra: Option<Arcseconds>,
    error_dec: Option<Arcseconds>,
    start_id: ObsId,
) -> Result<ObsDataset, AdesError> {
    let mut observations: Vec<Observation> = Vec::with_capacity(records.len());
    let mut traj_index: AHashMap<TrajId, Vec<usize>> =
        AHashMap::with_capacity(records.len() / 4 + 1);

    for (idx, (optical, ctx_mpc)) in records.iter().enumerate() {
        // MPC code: context wins over observation-level stn.
        let mpc_code = ctx_mpc.unwrap_or_else(|| optical.mpc_code_bytes());

        // Trajectory identifier.
        let traj_id = optical
            .traj_id()
            .ok_or(AdesError::MissingTrajId { index: idx })?;

        // Positional uncertainties: arcsec → degrees.
        let ra_err_arcsec = optical
            .ra_error_arcsec(error_ra)
            .ok_or(AdesError::MissingRaError { index: idx })?;
        let dec_err_arcsec = optical
            .dec_error_arcsec(error_dec)
            .ok_or(AdesError::MissingDecError { index: idx })?;

        let ra_err_deg: Degrees = ra_err_arcsec * ARCSEC_TO_DEG;
        let dec_err_deg: Degrees = dec_err_arcsec * ARCSEC_TO_DEG;

        // Equatorial coordinates in radians.
        let equ_coord = EquCoord::from_degrees(optical.ra, ra_err_deg, optical.dec, dec_err_deg);

        let photometry = Photometry {
            magnitude: optical.mag.unwrap_or(0.0),
            error: optical.rms_mag.unwrap_or(0.0),
            filter: optical
                .band
                .as_deref()
                .map(|b| Filter::String(b.to_string()))
                .unwrap_or(Filter::String("unknown".to_string())),
        };

        observations.push(Observation {
            index: Some(idx),
            id: start_id + idx as u64,
            equ_coord,
            photometry,
            mjd_tt: optical.obs_time,
            observer: Some(ObserverId::MpcCode(mpc_code)),
        });

        traj_index.entry(traj_id).or_default().push(idx);
    }

    let traj_index_map: TrajIndexMap = traj_index
        .into_iter()
        .map(|(traj_id, indices)| (traj_id, ObsMapIndex::Split(indices)))
        .collect();

    Ok(ObsDataset::new(
        observations,
        vec![], // no custom geodetic observers
        None,   // error model resolved lazily on first MPC look-up
        None,   // no night index
        Some(traj_index_map),
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `&str` MPC code into a fixed-size 3-byte array.
///
/// Strings shorter than 3 bytes are right-padded with ASCII spaces; extra
/// bytes beyond the third are silently discarded.
fn mpc_str_to_bytes(code: &str) -> [u8; 3] {
    let bytes = code.as_bytes();
    let mut arr = [b' '; 3];
    let len = bytes.len().min(3);
    arr[..len].copy_from_slice(&bytes[..len]);
    arr
}
