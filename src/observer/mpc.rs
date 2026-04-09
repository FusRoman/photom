//! MPC observatory code types and the network-fetch routine.
//!
//! This module defines the type aliases used to identify observatories by
//! their Minor Planet Center (MPC) three-character code, provides the lookup
//! table type that maps each code to an [`Observer`], and exposes
//! [`init_observatories`], which fetches and parses the official MPC
//! observatory list from the network.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`MpcCode`] | type alias | Three-byte ASCII MPC observatory code |
//! | [`MpcCodeObs`] | type alias | Hash map from [`MpcCode`] to [`Observer`] |
//! | [`MPCError`] | enum | Errors arising from the MPC network request |
//! | [`init_observatories`] | fn | Fetch and parse the MPC observatory list |

use ahash::AHashMap;
use thiserror::Error;
use ureq::Agent;

use crate::observer::{
    error_model::{get_bias_rms, ErrorModelData},
    Observer,
};

/// Three-byte ASCII MPC observatory code (e.g. `b"I41"`, `b"500"`, `b"G96"`).
///
/// All observatory codes in the MPC catalogue are exactly three ASCII
/// characters.  The byte-array representation avoids heap allocation and
/// enables efficient use as a hash-map key.
pub type MpcCode = [u8; 3];

/// Hash map from [`MpcCode`] to [`Observer`] metadata.
///
/// Built by [`init_observatories`] from the official MPC observatory list.
/// Uses [`ahash`] for fast, non-cryptographic hashing.
pub type MpcCodeObs = AHashMap<MpcCode, Observer>;

/// Errors that can arise when fetching or processing the MPC observatory list.
#[derive(Error, Debug)]
pub enum MPCError {
    /// The HTTP request to the MPC website failed.
    #[error(transparent)]
    UreqError(#[from] ureq::Error),
}

/// Parse a fixed-width `f32` field from a slice of an MPC observatory line.
///
/// # Arguments
///
/// - `s` — fixed-width tail of the MPC line (the part after the three-char code).
/// - `slice` — byte range selecting the numeric field within `s`.
/// - `code` — MPC code string used in the panic message for diagnostics.
///
/// # Returns
///
/// `Ok(f32)` with the parsed value, or a [`std::num::ParseFloatError`] if
/// the trimmed slice is not a valid floating-point literal.
///
/// # Panics
///
/// Panics if `slice` is out of bounds for `s` (i.e. the line is shorter than
/// expected for the given field).
fn parse_f32(
    s: &str,
    slice: std::ops::Range<usize>,
    code: &str,
) -> Result<f32, std::num::ParseFloatError> {
    s.get(slice)
        .unwrap_or_else(|| panic!("Failed to parse float for observer code: {code}"))
        .trim()
        .parse()
}

/// Extract longitude, $\rho\cos\phi'$, $\rho\sin\phi'$, and name from a
/// fixed-width MPC observatory row.
///
/// The MPC fixed-width format places the fields at the following byte offsets
/// within the tail that follows the three-character code:
///
/// | Field | Bytes |
/// |-------|-------|
/// | longitude (deg) | 1–9 |
/// | $\rho\cos\phi'$ | 10–17 |
/// | $\rho\sin\phi'$ | 18–26 |
/// | site name | 27– |
///
/// Numeric fields that fail to parse fall back to `0.0`.
///
/// # Arguments
///
/// - `remain` — fixed-width tail of the line (after the three-char MPC code),
///   trailing whitespace already stripped.
/// - `code` — MPC code string used in diagnostic messages.
///
/// # Returns
///
/// `Some((longitude_deg, rho_cos_phi, rho_sin_phi, name))` on success, or
/// `None` if the line is too short to contain a site name (byte offset 27).
fn parse_remain(remain: &str, code: &str) -> Option<(f32, f32, f32, String)> {
    let name = remain.get(27..)?.to_string();

    let longitude = parse_f32(remain, 1..10, code).unwrap_or(0.0);
    let cos = parse_f32(remain, 10..18, code).unwrap_or(0.0);
    let sin = parse_f32(remain, 18..27, code).unwrap_or(0.0);

    Some((longitude, cos, sin, name))
}

/// Fetch the MPC observatory list from the network and build the lookup table.
///
/// Issues an HTTP GET to `https://minorplanetcenter.net/iau/lists/ObsCodes.html`,
/// skips the two-line HTML header, then parses each fixed-width data line into
/// an [`Observer`] via [`Observer::from_parallax`].  Astrometric uncertainties
/// are populated from `error_model` using [`get_bias_rms`] with catalog code
/// `"c"` as the default.  Lines that are malformed or whose three-character
/// code cannot be converted to a [`MpcCode`] are silently skipped.
///
/// The returned map is pre-allocated with capacity for 2 048 entries to avoid
/// rehashing on a typical MPC list (currently ~2 000 observatories).
///
/// # Arguments
///
/// - `ureq_agent` — a configured [`ureq::Agent`] (e.g. with a global timeout)
///   used to perform the HTTP request.
/// - `error_model` — pre-loaded [`ErrorModelData`] table used to assign
///   astrometric uncertainties to each observatory.
///
/// # Returns
///
/// An [`MpcCodeObs`] map on success.
///
/// # Errors
///
/// Returns [`MPCError::UreqError`] if the HTTP request fails or the response
/// body cannot be read.
pub fn init_observatories(
    ureq_agent: Agent,
    error_model: &ErrorModelData,
) -> Result<MpcCodeObs, MPCError> {
    let mpc_document = ureq_agent
        .get("https://minorplanetcenter.net/iau/lists/ObsCodes.html")
        .call()?
        .body_mut()
        .read_to_string()?;

    // The MPC list currently has ~2 000 entries; pre-allocate to avoid rehashing.
    let mut observatories = MpcCodeObs::with_capacity(2048);

    for line in mpc_document.lines().skip(2) {
        let line = line.trim();

        let Some((code, remain)) = line.split_at_checked(3) else {
            continue;
        };

        // Convert the 3-char ASCII code to MpcCode ([u8; 3]); skip malformed lines.
        let Ok(mpc_code) = code.as_bytes().try_into() else {
            continue;
        };

        let Some((longitude, cos, sin, name)) = parse_remain(remain.trim_end(), code) else {
            continue;
        };

        // TODO: support per-site catalog codes (not always "c")
        let bias_rms = get_bias_rms(error_model, mpc_code, "c");
        let (ra_acc, dec_acc) = match bias_rms {
            Some((ra, dec)) => (Some(ra as f64), Some(dec as f64)),
            None => (None, None),
        };

        if let Ok(observer) = Observer::from_parallax(
            longitude as f64,
            cos as f64,
            sin as f64,
            Some(name),
            ra_acc,
            dec_acc,
        ) {
            observatories.insert(mpc_code, observer);
        }
    }

    Ok(observatories)
}
