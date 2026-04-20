#![cfg(feature = "mpc_80_col")]

//! Parser for **MPC 80-column** astrometric observation files.
//!
//! This module is the backend for
//! [`ObsDataset::from_mpc_80_col`](crate::observation_dataset::ObsDataset::from_mpc_80_col).
//! It uses [`nom`] parsers for the RA, Dec, and date fields and produces a
//! fully-indexed [`ObsDataset`].

use ahash::AHashMap;
use camino::Utf8Path;
use hifitime::{Epoch, TimeScale};
use nom::{
    IResult, Parser,
    bytes::complete::take_while1,
    character::complete::{digit1, space1},
    combinator::{map_res, opt},
};

use thiserror::Error;

use crate::{
    Degrees, TrajId,
    constants::ARCSEC_TO_DEG,
    coordinates::equatorial::EquCoord,
    observation_dataset::{
        ObsDataset,
        index::{ObsMapIndex, TrajIndexMap},
        observation::Observation,
    },
    observer::dataset::ObserverId,
    photometry::{Filter, Photometry},
};

/// Errors that can occur while reading an MPC 80-column observation file.
#[derive(Debug, Error)]
pub enum Mpc80ColError {
    /// The file could not be opened or read.
    #[error("I/O error reading MPC 80-col file: {0}")]
    Io(#[from] std::io::Error),

    /// A line that should be 80+ columns was malformed and could not be parsed.
    #[error("failed to parse MPC 80-col line {line_no}: {reason}\n  line: {line}")]
    InvalidLine {
        line_no: usize,
        line: String,
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse an MPC 80-column file and return a fully built [`ObsDataset`].
///
/// Lines that carry satellite secondary-line data (column 15, 1-indexed,
/// equals `'s'`) are silently skipped.  Lines shorter than 80 characters
/// are also silently skipped.  All other parse errors are returned as
/// [`Mpc80ColError`].
///
/// Because a single MPC 80-column file always describes **one physical
/// object**, all observations are grouped under a single canonical
/// [`TrajId`], regardless of how many designation strings appear on
/// individual lines.  The canonical key is chosen as follows:
///
/// 1. If any line carries a numbered identifier (columns 1–5 non-blank after
///    stripping leading zeros), the corresponding [`TrajId::Int`] is used.
/// 2. Otherwise the first provisional designation encountered is used as a
///    [`TrajId::Str`].
///
/// Every *other* designation string found in the file is registered as an
/// alias via [`ObsDataset::resolve_alias`], pointing to the canonical key.
pub(crate) fn parse_mpc_80_col_file(path: &Utf8Path) -> Result<ObsDataset, Mpc80ColError> {
    let content = std::fs::read_to_string(path)?;
    parse_mpc_80_col_str(&content)
}

/// Build an [`ObsDataset`] from an in-memory MPC 80-column string.
///
/// Same semantics as [`parse_mpc_80_col_file`] but accepts an
/// already-loaded string, which is useful for unit tests.
pub(crate) fn parse_mpc_80_col_str(content: &str) -> Result<ObsDataset, Mpc80ColError> {
    // ── Pass 1: parse all valid lines ──────────────────────────────────────
    let mut records: Vec<(TrajId, LineRecord)> = Vec::new();

    for (line_no, line) in content.lines().enumerate() {
        if line.len() < 80 {
            continue;
        }
        if line.as_bytes()[14] == b's' {
            continue;
        }
        let record = parse_line(line).map_err(|reason| Mpc80ColError::InvalidLine {
            line_no: line_no + 1,
            line: line.to_string(),
            reason,
        })?;
        let traj_id = record.traj_id.clone();
        records.push((traj_id, record));
    }

    // ── Select canonical TrajId ────────────────────────────────────────────
    // Prefer a TrajId::Int (numbered asteroid) over any TrajId::Str.
    // Fall back to the first Str if no Int is present.
    let primary: TrajId = records
        .iter()
        .find_map(|(id, _)| match id {
            TrajId::Int(_) => Some(id.clone()),
            TrajId::Str(_) => None,
        })
        .or_else(|| records.first().map(|(id, _)| id.clone()))
        .unwrap_or(TrajId::Str(String::new()));

    // ── Pass 2: build observations and collect aliases ─────────────────────
    let mut observations: Vec<Observation> = Vec::with_capacity(records.len());
    let mut seen_aliases: AHashMap<String, ()> = AHashMap::new();

    for (traj_id, record) in records {
        let idx = observations.len();
        observations.push(record.into_observation(idx));

        // Register every designation that differs from the primary as an alias.
        if let TrajId::Str(s) = &traj_id
            && traj_id != primary
        {
            seen_aliases.entry(s.clone()).or_default();
        }
    }

    // ── Build trajectory index under the single canonical key ──────────────
    let mut traj_index: TrajIndexMap = AHashMap::new();
    let all_indices: Vec<usize> = (0..observations.len()).collect();
    traj_index.insert(primary.clone(), ObsMapIndex::Split(all_indices));

    let mut dataset = ObsDataset::new(observations, vec![], None, None, Some(traj_index));

    // ── Register aliases ───────────────────────────────────────────────────
    for alias in seen_aliases.into_keys() {
        dataset.register_alias(alias, primary.clone());
    }

    Ok(dataset)
}

// ---------------------------------------------------------------------------
// Internal record type
// ---------------------------------------------------------------------------

/// Decoded fields from a single 80-column observation line.
struct LineRecord {
    traj_id: TrajId,
    mjd_tt: f64,
    ra_deg: Degrees,
    ra_acc_arcsec: f64,
    dec_deg: Degrees,
    dec_acc_arcsec: f64,
    mag: Option<f32>,
    band: Option<String>,
    obs_code: [u8; 3],
}

impl LineRecord {
    fn into_observation(self, idx: usize) -> Observation {
        let ra_err_deg = self.ra_acc_arcsec * ARCSEC_TO_DEG;
        let dec_err_deg = self.dec_acc_arcsec * ARCSEC_TO_DEG;
        let equ_coord = EquCoord::from_degrees(self.ra_deg, ra_err_deg, self.dec_deg, dec_err_deg);

        let photometry = Photometry {
            magnitude: self.mag.unwrap_or(0.0) as f64,
            error: 0.0,
            filter: self
                .band
                .map(Filter::String)
                .unwrap_or(Filter::String("unknown".to_string())),
        };

        Observation {
            index: idx,
            id: idx as u64,
            equ_coord,
            photometry,
            mjd_tt: self.mjd_tt,
            observer: Some(ObserverId::MpcCode(self.obs_code)),
        }
    }
}

// ---------------------------------------------------------------------------
// Line parser
// ---------------------------------------------------------------------------

/// Parse a single 80-column observation line into a [`LineRecord`].
///
/// Field layout (0-indexed byte positions):
///
/// | Cols   | Content                       |
/// |--------|-------------------------------|
/// | 0–4    | Minor planet number (packed)  |
/// | 5–11   | Provisional designation       |
/// | 12     | Discovery flag (`*` or space) |
/// | 13     | Note 1                        |
/// | 14     | Note 2 / observation type     |
/// | 15–31  | Date `YYYY MM DD.ddddd`       |
/// | 32–43  | RA `HH MM SS.sss`            |
/// | 44–55  | Dec `±DD MM SS.ss`           |
/// | 56–64  | Blank                         |
/// | 65–70  | Magnitude                     |
/// | 71     | Band                          |
/// | 72–76  | Catalog code                  |
/// | 77–79  | MPC observatory code          |
fn parse_line(line: &str) -> Result<LineRecord, String> {
    let traj_id = extract_traj_id(line);

    let date_str = line[15..32].trim();
    let ra_str = line[32..44].trim();
    let dec_str = line[44..56].trim();
    let mag_str = line[65..70].trim();
    let band_str = line[70..71].trim();
    let obs_code_str = line[77..80].trim();

    let mjd_tt = parse_frac_date(date_str)
        .map_err(|_| format!("invalid date field: '{date_str}'"))?
        .1;

    let (ra_deg, ra_acc_arcsec) = parse_ra(ra_str)
        .map_err(|_| format!("invalid RA field: '{ra_str}'"))?
        .1;

    let (dec_deg, dec_acc_arcsec) = parse_dec(dec_str)
        .map_err(|_| format!("invalid Dec field: '{dec_str}'"))?
        .1;

    let mag = mag_str
        .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
        .parse::<f32>()
        .ok();

    let band = if band_str.is_empty() {
        None
    } else {
        Some(band_str.to_string())
    };

    let obs_code = mpc_str_to_bytes(obs_code_str);

    Ok(LineRecord {
        traj_id,
        mjd_tt,
        ra_deg,
        ra_acc_arcsec,
        dec_deg,
        dec_acc_arcsec,
        mag,
        band,
        obs_code,
    })
}

/// Extract the trajectory identifier from columns 0–11.
///
/// - Columns 0–4 (minor planet number): trimmed and stripped of leading zeros.
///   If non-empty after stripping, used as the trajectory ID.
/// - Columns 5–11 (provisional designation): fallback when the number field
///   is blank.
///
/// If the trimmed string parses as a [`u64`] → [`TrajId::Int`];
/// otherwise → [`TrajId::Str`].
fn extract_traj_id(line: &str) -> TrajId {
    let num_part = line[0..5].trim_start_matches('0').trim();
    let raw = if num_part.is_empty() {
        line[5..12].trim()
    } else {
        num_part
    };

    if let Ok(n) = raw.parse::<u32>() {
        TrajId::Int(n)
    } else {
        TrajId::Str(raw.to_string())
    }
}

// ---------------------------------------------------------------------------
// Nom parsers
// ---------------------------------------------------------------------------

/// Parse a fractional date `"YYYY MM DD.ddddd"` (UTC) and return MJD (TT).
fn parse_frac_date(input: &str) -> IResult<&str, f64> {
    let (input, year) = map_res(digit1, |s: &str| s.parse::<i32>()).parse(input)?;
    let (input, _) = space1(input)?;
    let (input, month) = map_res(digit1, |s: &str| s.parse::<u8>()).parse(input)?;
    let (input, _) = space1(input)?;
    let (input, day_str) = take_while1(|c: char| c.is_ascii_digit() || c == '.').parse(input)?;

    let day_frac: f64 = day_str.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Float))
    })?;

    let day = day_frac.trunc() as u8;
    let frac = day_frac - day as f64;

    let total_sec = frac * 86_400.0;
    let hour = (total_sec / 3_600.0).trunc() as u8;
    let rem_sec = (total_sec - hour as f64 * 3_600.0).max(0.0);
    let minute = (rem_sec / 60.0).trunc() as u8;
    let rem_sec2 = (rem_sec - minute as f64 * 60.0).max(0.0);
    let second = rem_sec2.trunc() as u8;
    let nano = ((rem_sec2 - second as f64) * 1e9).round() as u32;

    let epoch = Epoch::from_gregorian(year, month, day, hour, minute, second, nano, TimeScale::UTC);
    Ok((input, epoch.to_mjd_tt_days()))
}

/// Parse a right ascension string `"HH MM SS.sss"` into `(ra_deg, accuracy_arcsec)`.
///
/// The accuracy is derived from the number of decimal places in the seconds
/// field: `10⁻ⁿ × 15` arcseconds (1 time-second of RA = 15 arcseconds on sky).
fn parse_ra(input: &str) -> IResult<&str, (f64, f64)> {
    let (input, (h, _, m, _, s_str)) = (
        map_res(digit1, |s: &str| s.parse::<f64>()),
        space1,
        map_res(digit1, |s: &str| s.parse::<f64>()),
        space1,
        take_while1(|c: char| c.is_ascii_digit() || c == '.'),
    )
        .parse(input)?;

    let s: f64 = s_str.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Float))
    })?;

    let ra_deg = (h + m / 60.0 + s / 3_600.0) * 15.0;
    let acc_arcsec = decimal_precision_arcsec(s_str) * 15.0;

    Ok((input, (ra_deg, acc_arcsec)))
}

/// Parse a declination string `"±DD MM SS.ss"` into `(dec_deg, accuracy_arcsec)`.
///
/// The accuracy is derived from the number of decimal places in the seconds
/// field: `10⁻ⁿ` arcseconds.
fn parse_dec(input: &str) -> IResult<&str, (f64, f64)> {
    let (input, sign_char) = opt(nom::character::complete::one_of("+-")).parse(input)?;
    let sign = if sign_char == Some('-') { -1.0 } else { 1.0 };

    let (input, (d, _, m, _, s_str)) = (
        map_res(digit1, |s: &str| s.parse::<f64>()),
        space1,
        map_res(digit1, |s: &str| s.parse::<f64>()),
        space1,
        take_while1(|c: char| c.is_ascii_digit() || c == '.'),
    )
        .parse(input)?;

    let s: f64 = s_str.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Float))
    })?;

    let dec_deg = sign * (d + m / 60.0 + s / 3_600.0);
    let acc_arcsec = decimal_precision_arcsec(s_str);

    Ok((input, (dec_deg, acc_arcsec)))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Derive angular precision in arcseconds from the number of decimal digits
/// in a seconds field (e.g. `"23.37"` → 2 digits → `0.01` arcsec).
///
/// This is the raw arcsecond precision before any RA × 15 scaling.
fn decimal_precision_arcsec(seconds_str: &str) -> f64 {
    let digits = seconds_str
        .find('.')
        .map(|dot| seconds_str.trim_end().len() - dot - 1)
        .unwrap_or(0);
    10f64.powi(-(digits as i32))
}

/// Convert a `&str` MPC observatory code into a fixed 3-byte array,
/// right-padding with ASCII spaces if shorter than 3 characters.
fn mpc_str_to_bytes(code: &str) -> [u8; 3] {
    let bytes = code.as_bytes();
    let mut arr = [b' '; 3];
    let len = bytes.len().min(3);
    arr[..len].copy_from_slice(&bytes[..len]);
    arr
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod mpc_80_col_tests {
    use super::*;

    #[test]
    fn test_parse_ra_basic() {
        let (_, (ra, acc)) = parse_ra("22 52 23.37").unwrap();
        // (22 + 52/60 + 23.37/3600) * 15
        let expected = (22.0 + 52.0 / 60.0 + 23.37 / 3600.0) * 15.0;
        assert!((ra - expected).abs() < 1e-9);
        // 2 decimal places → 0.01 sec × 15 = 0.15 arcsec
        assert!((acc - 0.15).abs() < 1e-9);
    }

    #[test]
    fn test_parse_dec_negative() {
        let (_, (dec, acc)) = parse_dec("-14 47 05.4").unwrap();
        let expected = -(14.0 + 47.0 / 60.0 + 5.4 / 3600.0);
        assert!((dec - expected).abs() < 1e-9);
        // 1 decimal place → 0.1 arcsec
        assert!((acc - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_parse_dec_positive_explicit_sign() {
        let (_, (dec, _)) = parse_dec("+08 01 18.05").unwrap();
        let expected = 8.0 + 1.0 / 60.0 + 18.05 / 3600.0;
        assert!((dec - expected).abs() < 1e-9);
    }

    #[test]
    fn test_parse_frac_date() {
        // 2009 09 15.22735 UTC → should give a positive MJD (TT)
        let (_, mjd) = parse_frac_date("2009 09 15.22735").unwrap();
        assert!(mjd > 50_000.0, "MJD should be positive: {mjd}");
    }

    #[test]
    fn test_extract_traj_id_numbered() {
        let line =
            "08467         C2024 12 03.05243000 23 45.348+08 01 18.05         18.93cV~8TCpW68";
        assert_eq!(extract_traj_id(line), TrajId::Int(8467));
    }

    #[test]
    fn test_extract_traj_id_provisional() {
        let line =
            "     K09R05F* C2009 09 15.22735 22 52 23.37 -14 47 05.4          20.7 Vr~097wG96";
        assert_eq!(extract_traj_id(line), TrajId::Str("K09R05F".to_string()));
    }

    #[test]
    fn test_decimal_precision_arcsec() {
        assert_eq!(decimal_precision_arcsec("23.37"), 0.01);
        assert_eq!(decimal_precision_arcsec("05.4"), 0.1);
        assert_eq!(decimal_precision_arcsec("18.05"), 0.01);
        assert_eq!(decimal_precision_arcsec("45.348"), 0.001);
    }
}
