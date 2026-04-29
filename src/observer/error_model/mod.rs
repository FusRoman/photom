//! Astrometric error models for observatory weighting.
//!
//! This module provides tools to load and query astrometric bias/RMS tables
//! used in orbit determination.  Each table associates an observatory (by
//! [`MpcCode`]) and a star catalog code (by [`CatalogCode`]) with a pair of
//! RMS values `(rms_ra, rms_dec)` in arcseconds, as recommended in the
//! literature.
//!
//! Three models are supported, selectable via [`ObsErrorModel`]:
//!
//! - `ObsErrorModel::FCCT14` — Farnocchia, Chesley, Chamberlin & Tholen (2014)
//! - `ObsErrorModel::CBM10` — Chesley, Baer & Monet (2010)
//! - `ObsErrorModel::VFCC17` — Vereš, Farnocchia, Chesley & Chamberlin (2017)
//!
//! ## Typical usage
//!
//! 1. Choose a model variant (e.g. `ObsErrorModel::FCCT14`).
//! 2. Call [`ObsErrorModel::read_error_model_file`] to parse the bundled
//!    reference file into an [`ErrorModelData`] map.
//! 3. Query the map with [`get_bias_rms`] to obtain astrometric uncertainties
//!    for weighting residuals.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`CatalogCode`] | type alias | Single-character star catalog identifier |
//! | [`ErrorModelData`] | type alias | Map from `(MpcCode, CatalogCode)` to `(rms_ra, rms_dec)` |
//! | [`ObsErrorModel`] | enum | Supported astrometric error model variants |
//! | [`ErrorModelParseError`] | enum | Errors arising when parsing a model file |
//! | [`get_bias_rms`] | fn | Query an [`ErrorModelData`] map with cascade fallbacks |
//!
//! ## References
//!
//! - Farnocchia, D., Chesley, S. R., Chamberlin, A. B., & Tholen, D. J. (2014)
//! - Chesley, S. R., Baer, J., & Monet, D. G. (2010)
//! - Vereš, P., Farnocchia, D., Chesley, S. R., & Chamberlin, A. B. (2017)
pub mod model_correction;
mod vfcc17;
pub use model_correction::ModelCorrection;

use std::{collections::HashMap, str::FromStr};

use nom::{
    IResult, Parser,
    bytes::complete::{tag, take_while1},
    character::complete::{char, multispace0},
    combinator::{map, opt},
    number::complete::float,
    sequence::{preceded, separated_pair, terminated},
};

use thiserror::Error;

use vfcc17::parse_vfcc17_line;

use crate::observer::mpc::MpcCode;

/// A single-character star catalog identifier (e.g. `"c"`, `"U"`, `"*"`).
///
/// Used as the second key component in [`ErrorModelData`].
pub type CatalogCode = String;

/// Map from `(MpcCode, CatalogCode)` to `(rms_ra, rms_dec)` in arcseconds.
///
/// Built by [`ObsErrorModel::read_error_model_file`] and queried with
/// [`get_bias_rms`].
pub type ErrorModelData = HashMap<(MpcCode, CatalogCode), (f32, f32)>;

/// Supported astrometric error model variants.
///
/// Each variant corresponds to a published astrometric bias/RMS model
/// distributed as a bundled reference file.  Call
/// [`ObsErrorModel::read_error_model_file`] to parse the file for the
/// selected variant into an [`ErrorModelData`] map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ObsErrorModel {
    /// Farnocchia, Chesley, Chamberlin & Tholen (2014).
    FCCT14,
    /// Chesley, Baer & Monet (2010).
    CBM10,
    /// Vereš, Farnocchia, Chesley & Chamberlin (2017).
    VFCC17,
}

/// Bundled FCCT14 rules file, included at compile time.
static FCCT14_RULES: &str = include_str!("data_models/fcct14.rules");
/// Bundled CBM10 rules file, included at compile time.
static CBM10_RULES: &str = include_str!("data_models/cbm10.rules");
/// Bundled VFCC17 rules file, included at compile time.
static VFCC17_RULES: &str = include_str!("data_models/vfcc17.rules");

/// Internal parse result: a list of `((station_str, catalog_code), (rms_ra, rms_dec))`.
/// Station codes are kept as `String` here and converted to `MpcCode` ([u8; 3]) in
/// [`parse_full_file`] so individual parsers stay simple.
pub(in crate::observer::error_model) type ParseResult<'a> =
    IResult<&'a str, Vec<((String, CatalogCode), (f32, f32))>>;

// ---------------------------------------------------------------------------
// Low-level parsers — FCCT14 / CBM10
// ---------------------------------------------------------------------------

/// Parse an alphanumeric station code followed by `:` (e.g. `"699:"`, `"ALL:"`).
///
/// # Arguments
///
/// - `input` — the parser input slice starting at a station code.
///
/// # Returns
///
/// The station code string slice (without the trailing `:`), together with
/// the unconsumed input tail.
fn parse_station(input: &str) -> IResult<&str, &str> {
    terminated(take_while1(|c: char| c.is_alphanumeric()), char(':')).parse(input)
}

/// Parse the optional `c=` prefix followed by one or more catalog letters or `*`.
///
/// Handles both bare catalog chars (e.g. `699:c  @ …`) and the explicit
/// `c=` prefix (e.g. `ALL: c=cd @ …`).  Each ASCII character in the matched
/// run is expanded into its own [`CatalogCode`] string.
///
/// # Arguments
///
/// - `input` — the parser input slice positioned at optional whitespace before
///   the optional `c=` prefix and catalog characters.
///
/// # Returns
///
/// A `Vec<String>` where each element is a single-character catalog code,
/// together with the unconsumed input tail.
fn parse_catalog_codes(input: &str) -> IResult<&str, Vec<String>> {
    preceded(
        (multispace0, opt(tag("c="))),
        map(
            take_while1(|c: char| c.is_alphabetic() || c == '*'),
            // Each ASCII char becomes its own catalog code string.
            // We do a single slice walk instead of allocating one String per char
            // by using byte offsets — all catalog chars are guaranteed ASCII.
            |s: &str| {
                s.as_bytes()
                    .iter()
                    .map(|&b| String::from(b as char))
                    .collect()
            },
        ),
    )
    .parse(input)
}

/// Parse `@ rms_ra, rms_dec`, consuming any leading whitespace before the `@`.
///
/// # Arguments
///
/// - `input` — the parser input slice positioned at optional whitespace before
///   the `@` marker.
///
/// # Returns
///
/// A tuple `(rms_ra, rms_dec)` as `f32` values, together with the unconsumed
/// input tail.
fn parse_rms_values(input: &str) -> IResult<&str, (f32, f32)> {
    preceded(
        (multispace0, char('@')),
        separated_pair(
            preceded(multispace0, float),
            char(','),
            preceded(multispace0, float),
        ),
    )
    .parse(input)
}

/// Parse one FCCT14/CBM10 rule line, stripping inline comments (`! …`).
///
/// Expects the format `<station>: [c=]<catalogs> @ <rms_ra>, <rms_dec>`.
/// The portion from `!` to end-of-line is discarded before parsing.
///
/// # Arguments
///
/// - `input` — a single rule line from the FCCT14 or CBM10 rules file.
///
/// # Returns
///
/// A flat list of `((station, catalog), (rms_ra, rms_dec))` — one entry per
/// catalog character in the rule — together with the unconsumed input tail.
fn parse_full_line(input: &str) -> ParseResult<'_> {
    // Strip everything from `!` onward (inline comment), then trim whitespace.
    let line = match input.find('!') {
        Some(pos) => input[..pos].trim(),
        None => input.trim(),
    };

    map(
        (parse_station, parse_catalog_codes, parse_rms_values),
        |(station, catalogs, (rmsa, rmsd))| {
            catalogs
                .into_iter()
                .map(|cat| ((station.to_string(), cat), (rmsa, rmsd)))
                .collect()
        },
    )
    .parse(line)
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise when parsing an astrometric error model file.
#[derive(Debug, Error, Clone)]
pub enum ErrorModelParseError {
    /// A `nom` parser failed on the given line content.
    #[error("Nom parsing error on: {0}")]
    NomParsingError(String),

    /// A station code in the file is not exactly three ASCII bytes.
    #[error("Station code is not exactly 3 ASCII bytes: {0:?}")]
    InvalidStationCode(String),
}

// ---------------------------------------------------------------------------
// File-level parser
// ---------------------------------------------------------------------------

/// Convert a three-character ASCII station string (e.g. `"ALL"`, `"699"`) into
/// a [`MpcCode`] (`[u8; 3]`).
///
/// # Errors
///
/// Returns [`ErrorModelParseError::InvalidStationCode`] if `s` is not exactly
/// three bytes.
fn str_to_mpc_code(s: &str) -> Result<MpcCode, ErrorModelParseError> {
    s.as_bytes()
        .try_into()
        .map_err(|_| ErrorModelParseError::InvalidStationCode(s.to_string()))
}

/// Parse an entire rules file into an [`ErrorModelData`] map.
///
/// Blank lines and comment-only lines (starting with `!`) are skipped.
/// Each remaining line is parsed with `parse_line`; on error the offending
/// line text is reported in the returned [`ErrorModelParseError`].
///
/// # Arguments
///
/// - `file` — the full text content of the rules file.
/// - `parse_line` — line-level parser function (e.g. [`parse_full_line`] or
///   [`parse_vfcc17_line`]).
///
/// # Returns
///
/// An [`ErrorModelData`] map on success.
///
/// # Errors
///
/// Returns [`ErrorModelParseError::NomParsingError`] if any non-blank,
/// non-comment line fails to parse, or
/// [`ErrorModelParseError::InvalidStationCode`] if a station code is not
/// exactly three bytes.
fn parse_full_file<F>(file: &str, parse_line: F) -> Result<ErrorModelData, ErrorModelParseError>
where
    F: Fn(&str) -> ParseResult,
{
    // Single-pass: parse → flatten entries → convert station → insert.
    // Avoids the intermediate Vec<Vec<_>> allocation of the collect+flatten pattern.
    file.lines()
        .filter(|line| {
            let t = line.trim();
            !t.is_empty() && !t.starts_with('!')
        })
        .try_fold(ErrorModelData::default(), |mut map, line| {
            let (_, pairs) = parse_line(line)
                .map_err(|_| ErrorModelParseError::NomParsingError(line.to_string()))?;
            for ((station, cat), rms) in pairs {
                let code = str_to_mpc_code(&station)?;
                map.insert((code, cat), rms);
            }
            Ok(map)
        })
}

// ---------------------------------------------------------------------------
// ObsErrorModel impl
// ---------------------------------------------------------------------------

impl ObsErrorModel {
    /// Load the internal RMS/bias table for this astrometric error model.
    ///
    /// Parses the reference file for the selected variant and returns an
    /// [`ErrorModelData`] map ready to be queried with [`get_bias_rms`].
    ///
    /// # Returns
    ///
    /// An [`ErrorModelData`] map containing all `(MpcCode, CatalogCode)` →
    /// `(rms_ra, rms_dec)` entries from the bundled reference file.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorModelParseError`] if any line in the reference file
    /// cannot be parsed (this should never happen with the bundled files).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let data = ObsErrorModel::FCCT14.read_error_model_file().unwrap();
    /// println!("{} entries", data.len());
    /// ```
    pub fn read_error_model_file(&self) -> Result<ErrorModelData, ErrorModelParseError> {
        match self {
            ObsErrorModel::FCCT14 => parse_full_file(FCCT14_RULES, parse_full_line),
            ObsErrorModel::CBM10 => parse_full_file(CBM10_RULES, parse_full_line),
            ObsErrorModel::VFCC17 => parse_full_file(VFCC17_RULES, parse_vfcc17_line),
        }
    }
}

/// Parse an [`ObsErrorModel`] from its canonical string name.
///
/// Recognised names (case-sensitive): `"FCCT14"`, `"CBM10"`, `"VFCC17"`.
///
/// # Arguments
///
/// - `s` — the string to parse (e.g. `"FCCT14"`).
///
/// # Returns
///
/// `Ok(ObsErrorModel)` for a recognised name.
///
/// # Errors
///
/// Returns [`ErrorModelParseError::NomParsingError`] if `s` does not match
/// any known model name.
impl FromStr for ObsErrorModel {
    type Err = ErrorModelParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "FCCT14" => Ok(ObsErrorModel::FCCT14),
            "CBM10" => Ok(ObsErrorModel::CBM10),
            "VFCC17" => Ok(ObsErrorModel::VFCC17),
            _ => Err(ErrorModelParseError::NomParsingError(format!(
                "Unknown error model: {s}"
            ))),
        }
    }
}

/// Infallible conversion from `&str` to [`ObsErrorModel`] by delegating to [`FromStr`].
///
/// Recognised names (case-sensitive): `"FCCT14"`, `"CBM10"`, `"VFCC17"`.
///
/// # Arguments
///
/// - `value` — the string slice to convert (e.g. `"VFCC17"`).
///
/// # Returns
///
/// `Ok(ObsErrorModel)` for a recognised name.
///
/// # Errors
///
/// Returns [`ErrorModelParseError::NomParsingError`] if `value` is not a
/// recognised model name.
impl TryFrom<&str> for ObsErrorModel {
    type Error = ErrorModelParseError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

// ---------------------------------------------------------------------------
// Public query helper
// ---------------------------------------------------------------------------

/// Retrieve the astrometric `(rms_ra, rms_dec)` for a given observatory and
/// star catalog code, with a cascade of fallbacks.
///
/// Lookup priority:
///
/// 1. Exact match: `(mpc_code, catalog_code)`
/// 2. `(mpc_code, "e")` — generic entry for that observatory
/// 3. `(mpc_code, "c")` — catalog default for that observatory
/// 4. `(*b"ALL", catalog_code)` — global entry for this catalog
/// 5. `(*b"ALL", "e")` — global `"e"` fallback
/// 6. `(*b"ALL", "c")` — global catalog fallback
///
/// # Arguments
///
/// - `error_model` — preloaded [`ErrorModelData`] table.
/// - `mpc_code` — MPC observatory code as `[u8; 3]` (e.g. `*b"699"`).
/// - `catalog_code` — star catalog identifier (e.g. `"c"`, `"U"`).
///
/// # Returns
///
/// `Some((rms_ra, rms_dec))` on a match, `None` if no entry is found at any
/// fallback level.
pub fn get_bias_rms(
    error_model: &ErrorModelData,
    mpc_code: MpcCode,
    catalog_code: &str,
) -> Option<(f32, f32)> {
    // Build lookup keys inline to avoid heap-allocating the catalog fallbacks
    // on every call; the compiler will inline these small closures.
    let lookup = |code: MpcCode, cat: &str| -> Option<(f32, f32)> {
        error_model.get(&(code, cat.to_string())).copied()
    };

    lookup(mpc_code, catalog_code)
        .or_else(|| lookup(mpc_code, "e"))
        .or_else(|| lookup(mpc_code, "c"))
        .or_else(|| lookup(*b"ALL", catalog_code))
        .or_else(|| lookup(*b"ALL", "e"))
        .or_else(|| lookup(*b"ALL", "c"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test_error_model {
    use super::*;

    /// Verifies that `parse_full_line` correctly parses representative FCCT14/CBM10 rule lines.
    ///
    /// Covers three patterns:
    /// - multi-catalog `c=` prefix with inline `!` comment (`ALL: c=eqru @ …`),
    /// - multi-catalog `c=` prefix with inline comment and space padding
    ///   (`ALL: c=cd @ … ! …`),
    /// - bare single-catalog code without `c=` prefix (`699:c @ …`).
    #[test]
    fn test_parse_fcct14_line() {
        let line = "ALL:  c=eqru @ 0.33, 0.30";
        let result = parse_full_line(line);
        assert!(result.is_ok());
        let ((mpc_code, catalog_code), (rmsa, rmsd)) = &result.unwrap().1[0];
        assert_eq!(mpc_code, "ALL");
        assert_eq!(catalog_code, "e");
        assert_eq!(*rmsa, 0.33);
        assert_eq!(*rmsd, 0.3);

        let line = "ALL:  c=cd   @ 0.51, 0.40 ! CBM Generic Catalog weights";
        let result = parse_full_line(line);
        assert!(result.is_ok());
        let ((mpc_code, catalog_code), (rmsa, rmsd)) = &result.unwrap().1[0];
        assert_eq!(mpc_code, "ALL");
        assert_eq!(catalog_code, "c");
        assert_eq!(*rmsa, 0.51);
        assert_eq!(*rmsd, 0.4);

        let line = "699:c  @ 0.93, 0.78";
        let result = parse_full_line(line);
        assert!(result.is_ok());
        let ((mpc_code, catalog_code), (rmsa, rmsd)) = &result.unwrap().1[0];
        assert_eq!(mpc_code, "699");
        assert_eq!(catalog_code, "c");
        assert_eq!(*rmsa, 0.93);
        assert_eq!(*rmsd, 0.78);
    }

    /// Verifies that `read_error_model_file` succeeds and returns a non-empty map
    /// for all three bundled model variants (FCCT14, CBM10, and VFCC17).
    #[test]
    fn test_read_error_model_file() {
        let result = ObsErrorModel::FCCT14.read_error_model_file();
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());

        let result = ObsErrorModel::CBM10.read_error_model_file();
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());

        let result = ObsErrorModel::VFCC17.read_error_model_file();
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    /// Verifies that `get_bias_rms` returns the expected RMS values for known entries
    /// in all three bundled model files (FCCT14, CBM10, and VFCC17), covering both
    /// the `*b"ALL"` wildcard key and a specific observatory code (`*b"699"`).
    #[test]
    fn test_get_bias_rms() {
        let model = ObsErrorModel::FCCT14.read_error_model_file().unwrap();
        let (rmsa, rmsd) = get_bias_rms(&model, *b"ALL", "c").unwrap();
        assert_eq!(rmsa, 0.51);
        assert_eq!(rmsd, 0.4);

        let (rmsa, rmsd) = get_bias_rms(&model, *b"699", "c").unwrap();
        assert_eq!(rmsa, 0.47);
        assert_eq!(rmsd, 0.39);

        let model = ObsErrorModel::CBM10.read_error_model_file().unwrap();
        let (rmsa, rmsd) = get_bias_rms(&model, *b"ALL", "c").unwrap();
        assert_eq!(rmsa, 0.5);
        assert_eq!(rmsd, 0.5);

        let (rmsa, rmsd) = get_bias_rms(&model, *b"699", "c").unwrap();
        assert_eq!(rmsa, 0.84);
        assert_eq!(rmsd, 0.81);

        let model = ObsErrorModel::VFCC17.read_error_model_file().unwrap();
        let (rmsa, rmsd) = get_bias_rms(&model, *b"ALL", "U").unwrap();
        assert_eq!(rmsa, 0.6);
        assert_eq!(rmsd, 0.6);

        let (rmsa, rmsd) = get_bias_rms(&model, *b"699", "*").unwrap();
        assert_eq!(rmsa, 0.8);
        assert_eq!(rmsd, 0.8);
    }
}
