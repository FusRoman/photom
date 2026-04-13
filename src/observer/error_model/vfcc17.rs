//! VFCC17-specific parser for astrometric error model rule lines.
//!
//! This sub-module implements the parser for the Vereš, Farnocchia, Chesley &
//! Chamberlin (2017) rule file format.  Each non-comment line encodes an
//! observatory code, observation type, catalog codes, parallax flag, date
//! constraints, and RMS values in a fixed positional format:
//!
//! ```text
//! <station> t=<obs_type> c=<catalogs> p=<parallax> > <date> < <date> @ rms_ra, rms_dec
//! ```
//!
//! Only the station code, catalog codes (`c=…`), and RMS values (`@ …`) are
//! extracted; all other fields are consumed and discarded.  The public entry
//! point is [`parse_vfcc17_line`].

use nom::{
    IResult, Parser,
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{char, multispace0},
    combinator::{map, opt},
    number::complete::float,
    sequence::{preceded, separated_pair},
};

use crate::observer::error_model::ParseResult;

/// Returns `true` for alphanumeric characters and `*`.
///
/// Used by [`parse_word`] and [`parse_catalog_codes`] to delimit tokens
/// in the VFCC17 rule format.
///
/// # Arguments
///
/// - `c` — the character to test.
///
/// # Returns
///
/// `true` if `c` is alphanumeric or `'*'`, `false` otherwise.
#[inline]
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '*'
}

/// Parse a contiguous word token consisting of alphanumeric characters and `*`.
///
/// # Arguments
///
/// - `input` — the parser input slice positioned at the start of a word token.
///
/// # Returns
///
/// The matched slice on success together with the unconsumed input tail, or a
/// `nom` error if the input does not start with at least one word character.
fn parse_word(input: &str) -> IResult<&str, &str> {
    take_while1(is_word_char)(input)
}

/// Parse the `c=<chars>` catalog-code field and expand each character into
/// its own [`CatalogCode`](crate::observer::error_model::CatalogCode) string.
///
/// The `c=` prefix is required; this parser will fail if it is absent.
/// Each ASCII character in the run following `c=` becomes a separate element
/// in the returned vector (e.g. `c=tU` yields `["t", "U"]`).
///
/// # Arguments
///
/// - `input` — the parser input slice positioned at the `c=` prefix.
///
/// # Returns
///
/// A `Vec<String>` where each element is a single-character catalog code,
/// together with the unconsumed input tail.
fn parse_catalog_codes(input: &str) -> IResult<&str, Vec<String>> {
    preceded(
        tag("c="),
        map(take_while1(is_word_char), |s: &str| {
            s.as_bytes()
                .iter()
                .map(|&b| String::from(b as char))
                .collect()
        }),
    )
    .parse(input)
}

/// Parse `@ rms_ra, rms_dec`, skipping everything before the `@`.
///
/// All text between the current position and the `@` marker is consumed and
/// discarded (date range and parallax fields).
///
/// # Arguments
///
/// - `input` — the parser input slice positioned at the `t=…` field or any
///   text that precedes the `@` marker.
///
/// # Returns
///
/// A tuple `(rms_ra, rms_dec)` as `f32` values, together with the unconsumed
/// input tail.
fn parse_rms_values(input: &str) -> IResult<&str, (f32, f32)> {
    preceded(
        (take_until("@"), char('@')),
        separated_pair(
            preceded(multispace0, float),
            preceded(multispace0, char(',')),
            preceded(multispace0, float),
        ),
    )
    .parse(input)
}

/// Parse one VFCC17 rule line, stripping the inline `! comment` if present.
///
/// The expected line format is:
///
/// ```text
/// <station> t=<obs_type> c=<catalogs> p=<parallax> > <date> < <date> @ rms_ra, rms_dec
/// ```
///
/// Date and parallax fields are consumed and discarded; only the station code,
/// catalog codes, and RMS values are extracted.
///
/// # Arguments
///
/// - `input` — a single rule line, with or without a trailing `! comment`.
///
/// # Returns
///
/// On success, returns a residual input slice (typically empty) and a
/// `Vec<((station, catalog), (rms_ra, rms_dec))>` with one entry per catalog
/// character in the `c=…` field.
///
/// # Errors
///
/// Returns a `nom` error if the line does not match the expected format (e.g.
/// missing `c=` field or `@` marker).
pub fn parse_vfcc17_line(input: &str) -> ParseResult<'_> {
    // Strip inline comment (`! …`) then trim whitespace.
    let (input, before_comment) = opt(take_until("!")).parse(input)?;
    let line = before_comment.unwrap_or(input).trim();

    map(
        (
            parse_word,
            take_until("c="),
            parse_catalog_codes,
            parse_rms_values,
        ),
        |(station, _, catalogs, (rmsa, rmsd))| {
            catalogs
                .into_iter()
                .map(|cat| ((station.to_string(), cat), (rmsa, rmsd)))
                .collect()
        },
    )
    .parse(line)
}

#[cfg(test)]
mod test_vfcc17_parser {
    use super::*;

    /// Verifies that `parse_vfcc17_line` correctly parses representative VFCC17 rule lines.
    ///
    /// Covers two patterns:
    /// - the `c=*` wildcard catalog code with a trailing `! comment`
    ///   (`ALL t=… c=* … @ 1.00, 1.00 ! Unknown catalog`),
    /// - a specific single-catalog code with a trailing comment
    ///   (`568 t=… c=t … @ 0.20, 0.20 ! Micheli updated`).
    #[test]
    fn test_vfcc17_parser() {
        let input = "ALL t=cBCVn c=*          p=    >            <            @  1.00,  1.00 ! Unknown catalog";
        let (_, parsed) = parse_vfcc17_line(input).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].0.0, "ALL");
        assert_eq!(parsed[0].0.1, "*");
        assert_eq!(parsed[0].1.0, 1.0);
        assert_eq!(parsed[0].1.1, 1.0);

        let input = "568 t=cC    c=t          p=_   >            <            @  0.20,  0.20  ! Micheli updated ";
        let (_, parsed) = parse_vfcc17_line(input).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].0.0, "568");
        assert_eq!(parsed[0].0.1, "t");
        assert_eq!(parsed[0].1.0, 0.2);
        assert_eq!(parsed[0].1.1, 0.2);
    }
}
