//! MPC observatory code types and the network-fetch routine.
//!
//! This module defines the type aliases used to identify observatories by
//! their Minor Planet Center (MPC) three-character code, provides the lookup
//! table type that maps each code to an [`Observer`], and exposes
//! `init_observatories`, which fetches and parses the official MPC
//! observatory list from the network.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`MpcCode`] | type alias | Three-byte ASCII MPC observatory code |
//! | [`MpcCodeObs`] | type alias | Hash map from [`MpcCode`] to [`Observer`] |
//! | [`MPCError`] | enum | Errors arising from the MPC network request |
//! | `init_observatories` | fn | Fetch and parse the MPC observatory list |

use std::{path::PathBuf, time::Duration};

use ahash::AHashMap;
use thiserror::Error;
use ureq::Agent;

use crate::observer::{
    Observer,
    error_model::{ErrorModelData, get_bias_rms},
};

/// Three-byte ASCII MPC observatory code (e.g. `b"I41"`, `b"500"`, `b"G96"`).
///
/// All observatory codes in the MPC catalogue are exactly three ASCII
/// characters.  The byte-array representation avoids heap allocation and
/// enables efficient use as a hash-map key.
pub type MpcCode = [u8; 3];

/// Hash map from [`MpcCode`] to [`Observer`] metadata.
///
/// Built by `init_observatories` from the official MPC observatory list.
/// Uses [`ahash`] for fast, non-cryptographic hashing.
pub type MpcCodeObs = AHashMap<MpcCode, Observer>;

/// Errors that can arise when fetching or processing the MPC observatory list.
#[derive(Error, Debug)]
pub enum MPCError {
    /// The HTTP request to the MPC website failed.
    #[error(transparent)]
    UreqError(#[from] ureq::Error),

    /// A filesystem I/O error occurred while reading or writing the cache.
    #[error("cache I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The platform cache directory could not be determined.
    #[error("could not determine platform cache directory")]
    CacheDirUnavailable,
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

/// Return the path to the MPC observatory cache file.
///
/// Resolution order:
/// 1. `cache_dir_override` — explicit path supplied by the caller (used in tests).
/// 2. `PHOTOM_MPC_CACHE_DIR` environment variable (used for CI overrides).
/// 3. The platform cache directory reported by [`directories::BaseDirs`],
///    under the sub-path `photom/mpc_obs.html`.
///
/// Returns [`MPCError::CacheDirUnavailable`] when no source is usable.
fn cache_file_path(cache_dir_override: Option<&std::path::Path>) -> Result<PathBuf, MPCError> {
    if let Some(dir) = cache_dir_override {
        return Ok(dir.join("mpc_obs.html"));
    }

    if let Ok(dir) = std::env::var("PHOTOM_MPC_CACHE_DIR") {
        return Ok(PathBuf::from(dir).join("mpc_obs.html"));
    }

    let base = directories::BaseDirs::new().ok_or(MPCError::CacheDirUnavailable)?;
    Ok(base.cache_dir().join("photom").join("mpc_obs.html"))
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
/// ## Caching
///
/// The raw MPC document is cached on disk so subsequent calls avoid a network
/// round-trip.  On the first call (or when the cache is missing / unreadable)
/// the document is fetched via `ureq_agent`, then written atomically to:
///
/// * `<platform-cache-dir>/photom/mpc_obs.html` (via [`directories::BaseDirs`]), or
/// * the directory pointed to by the `PHOTOM_MPC_CACHE_DIR` environment
///   variable when set (useful for CI overrides).
///
/// The cache stores the raw HTML/text body so that a different `error_model`
/// can still be applied at construction time.  If the cache file cannot be
/// written the error is silently ignored and the in-memory document is still
/// returned.
///
/// The returned map is pre-allocated with capacity for 2 048 entries to avoid
/// rehashing on a typical MPC list (currently ~2 000 observatories).
///
/// # Arguments
///
/// - `error_model` — pre-loaded [`ErrorModelData`] table used to assign
///   astrometric uncertainties to each observatory.
///
/// # Returns
///
/// An [`MpcCodeObs`] map on success.
///
/// # Errors
///
/// | Variant | Cause |
/// |---------|-------|
/// | [`MPCError::UreqError`] | HTTP request failed or body unreadable |
/// | [`MPCError::CacheDirUnavailable`] | platform cache dir cannot be determined |
pub fn init_observatories(error_model: &ErrorModelData) -> Result<MpcCodeObs, MPCError> {
    // configure a ureq Agent with a global timeout to avoid hanging indefinitely on network issues
    let config = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(10)))
        .build();
    let agent: Agent = config.into();

    init_observatories_impl(agent, error_model, None)
}

/// Internal implementation; `cache_dir_override` is used by tests to avoid
/// touching the real system cache directory.
fn init_observatories_impl(
    ureq_agent: Agent,
    error_model: &ErrorModelData,
    cache_dir_override: Option<&std::path::Path>,
) -> Result<MpcCodeObs, MPCError> {
    let cache_path = cache_file_path(cache_dir_override)?;

    // --- Try reading from cache -----------------------------------------------
    let mpc_document = match std::fs::read_to_string(&cache_path) {
        Ok(content) => content,
        Err(_) => mpc_obs_request(ureq_agent, &cache_path)?,
    };

    // --- Parse ----------------------------------------------------------------
    parse_mpc_obs_result(&mpc_document, error_model)
}

/// Fetch the raw MPC observatory document from the network and write it to
/// `cache_path` atomically.
///
/// Issues an HTTP GET to `https://minorplanetcenter.net/iau/lists/ObsCodes.html`
/// and reads the full response body into a `String`.  The body is then written
/// to a sibling `.html.tmp` file in the same directory as `cache_path` and
/// atomically renamed into place so readers never observe a partial file.
/// Parent directories are created with [`std::fs::create_dir_all`] if they do
/// not yet exist.  Any filesystem error during the write is printed to `stderr`
/// and silently ignored — the in-memory document is still returned.
///
/// # Arguments
///
/// - `ureq_agent` — a configured [`ureq::Agent`] used to perform the HTTP GET.
/// - `cache_path` — destination path for the cached document; the parent
///   directory and a sibling `.html.tmp` file are derived from this path.
///
/// # Returns
///
/// The raw HTTP response body as a `String` on success.
///
/// # Errors
///
/// | Variant | Cause |
/// |---------|-------|
/// | [`MPCError::UreqError`] | HTTP request failed or response body could not be read |
fn mpc_obs_request(ureq_agent: Agent, cache_path: &std::path::Path) -> Result<String, MPCError> {
    // Cache miss (or unreadable): fetch from network.
    let fetched = ureq_agent
        .get("https://minorplanetcenter.net/iau/lists/ObsCodes.html")
        .call()?
        .body_mut()
        .read_to_string()?;

    // --- Write cache atomically ----------------------------------------
    // Write to a sibling .tmp file first, then rename so readers never
    // see a partial file.
    if let Some(parent) = cache_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            // Non-fatal: proceed without caching.
            eprintln!(
                "photom: could not create MPC cache directory {}: {e}",
                parent.display()
            );
        } else {
            let tmp_path = cache_path.with_extension("html.tmp");
            match std::fs::write(&tmp_path, fetched.as_bytes()) {
                Ok(()) => {
                    if let Err(e) = std::fs::rename(&tmp_path, cache_path) {
                        eprintln!("photom: could not rename MPC cache file: {e}");
                        // Best-effort cleanup; ignore secondary errors.
                        let _ = std::fs::remove_file(&tmp_path);
                    }
                }
                Err(e) => {
                    eprintln!("photom: could not write MPC cache: {e}");
                }
            }
        }
    }

    Ok(fetched)
}

/// Parse a raw MPC observatory document into an [`MpcCodeObs`] lookup table.
///
/// Iterates over `mpc_document` line by line, skipping the first two header
/// lines (the HTML/text preamble of the MPC list).  Each remaining line is
/// expected to follow the MPC fixed-width format: a three-character observatory
/// code followed by longitude, $\rho\cos\phi'$, $\rho\sin\phi'$, and site name
/// fields.  Lines that are too short to split at byte 3, whose three-character
/// prefix cannot be converted to an [`MpcCode`], or whose numeric fields cannot
/// be parsed into a valid [`Observer`] are silently skipped.
///
/// Astrometric uncertainties are looked up in `error_model` via [`get_bias_rms`]
/// using catalog code `"c"` as the default for every observatory.  The returned
/// map is pre-allocated with a capacity of 2 048 entries to avoid rehashing on
/// a typical MPC list (currently ~2 000 observatories).
///
/// # Arguments
///
/// - `mpc_document` — raw text body of the MPC observatory list (e.g. as
///   fetched by [`mpc_obs_request`] or read from the on-disk cache).
/// - `error_model` — pre-loaded [`ErrorModelData`] table used to assign
///   astrometric uncertainties to each observatory entry.
///
/// # Returns
///
/// An [`MpcCodeObs`] map from [`MpcCode`] to [`Observer`] on success.
fn parse_mpc_obs_result(
    mpc_document: &str,
    error_model: &ErrorModelData,
) -> Result<MpcCodeObs, MPCError> {
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod mpc_obs_tests {
    use super::*;
    use crate::observer::Observer;
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // parse_f32 tests
    // -----------------------------------------------------------------------

    /// Verifies that a padded integer string parses to the expected f32.
    #[test]
    fn parse_f32_valid_integer() {
        // "  42  " at range 0..6 should trim to "42" and parse as 42.0
        let result = parse_f32("  42  ", 0..6, "TST");
        assert!(result.is_ok(), "Expected Ok but got: {:?}", result);
        assert_relative_eq!(result.unwrap(), 42.0_f32);
    }

    /// Verifies that a padded decimal string parses to the expected f32.
    #[test]
    fn parse_f32_valid_float() {
        // " 1.5  " at range 0..6 should trim to "1.5" and parse as 1.5
        let result = parse_f32(" 1.5  ", 0..6, "TST");
        assert!(result.is_ok(), "Expected Ok but got: {:?}", result);
        assert_relative_eq!(result.unwrap(), 1.5_f32);
    }

    /// Verifies that a negative decimal string parses correctly.
    ///
    /// Float comparison uses relative epsilon because -1.23 is not exactly
    /// representable in IEEE 754 single precision.
    #[test]
    fn parse_f32_negative() {
        // "-1.23 " at range 0..6 should trim to "-1.23" and parse as -1.23
        let result = parse_f32("-1.23 ", 0..6, "TST");
        assert!(result.is_ok(), "Expected Ok but got: {:?}", result);
        assert_relative_eq!(result.unwrap(), -1.23_f32, epsilon = 1e-6_f32);
    }

    /// Verifies that an alphabetic string returns a ParseFloatError.
    #[test]
    fn parse_f32_invalid_returns_error() {
        // "  abc " cannot be parsed as a float; must return Err
        let result = parse_f32("  abc ", 0..6, "TST");
        assert!(
            result.is_err(),
            "Expected Err for non-numeric input, but got: {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // parse_remain tests
    // -----------------------------------------------------------------------

    /// Verifies that a well-formed MPC observatory tail parses to correct values.
    ///
    /// The MPC fixed-width layout within `remain` (the part after the 3-char code):
    ///   byte 0        : leading space (separator)
    ///   bytes  1..10  : longitude in degrees   (9 chars)
    ///   bytes 10..18  : rho_cos_phi             (8 chars)
    ///   bytes 18..27  : rho_sin_phi             (9 chars)
    ///   bytes 27..    : site name
    ///
    /// Float comparisons use a tight relative epsilon because the source
    /// values have at most 6 significant digits.
    #[test]
    fn parse_remain_valid_line() {
        // Construct remain from known fields so the test is self-documenting.
        //           [0][ 1..10 ][10..18][18..27][27..]
        let remain = " 289.265770.864977-0.500219Observatoire de Haute-Provence";

        let result = parse_remain(remain, "I41");
        assert!(result.is_some(), "Expected Some but got None");

        let (longitude, cos, sin, name) = result.unwrap();

        assert_relative_eq!(longitude, 289.26577_f32, epsilon = 1e-4_f32);
        assert_relative_eq!(cos, 0.864977_f32, epsilon = 1e-5_f32);
        assert_relative_eq!(sin, -0.500219_f32, epsilon = 1e-5_f32);
        assert!(
            name.contains("Observatoire"),
            "Expected name to contain 'Observatoire', got: {name:?}"
        );
    }

    /// Verifies that a `remain` string shorter than 28 bytes returns `None`.
    ///
    /// `remain.get(27..)` is `None` when the string is too short, so the
    /// function must propagate that with the `?` operator and return `None`.
    #[test]
    fn parse_remain_too_short_returns_none() {
        // Only 20 bytes — cannot possibly contain the name field at offset 27.
        let remain = " 289.265770.864977";
        let result = parse_remain(remain, "TST");
        assert!(
            result.is_none(),
            "Expected None for too-short input, got: {:?}",
            result
        );
    }

    /// Verifies that the site name is exactly the substring starting at byte 27.
    ///
    /// The MPC format places the observatory name at a fixed offset; this test
    /// uses a string with a recognisable sentinel starting precisely at byte 27.
    #[test]
    fn parse_remain_name_is_trimmed_correctly() {
        // Layout: 1 + 9 + 8 + 9 = 27 bytes before the name.
        //           [0][  1..10 ][10..18][18..27][27..]
        let remain = " 000.000000.0000000.000000 TestSiteName";
        //                                        ^-- byte 27

        let result = parse_remain(remain, "TST");
        assert!(result.is_some(), "Expected Some but got None");

        let (_, _, _, name) = result.unwrap();
        assert_eq!(
            name, "TestSiteName",
            "Name should be exactly the substring from byte 27; got: {name:?}"
        );
    }

    /// Verifies that unparseable numeric fields fall back to 0.0 while still
    /// returning `Some` as long as the string is at least 28 bytes long.
    ///
    /// The MPC parser calls `parse_f32(…).unwrap_or(0.0)`, so invalid numeric
    /// content (pure whitespace here) should silently yield 0.0.
    #[test]
    fn parse_remain_zero_fallback() {
        // All numeric slots are spaces (unparseable); only the name field is present.
        //           [0][  1..10 ][10..18][ 18..27 ][27..]
        let remain = "                           Spacewatch";
        //                                          ^-- byte 27

        let result = parse_remain(remain, "TST");
        assert!(result.is_some(), "Expected Some but got None");

        let (longitude, cos, sin, name) = result.unwrap();

        assert_relative_eq!(longitude, 0.0_f32);
        assert_relative_eq!(cos, 0.0_f32);
        assert_relative_eq!(sin, 0.0_f32);
        assert_eq!(
            name, "Spacewatch",
            "Name should be the text at byte 27; got: {name:?}"
        );
    }

    // -----------------------------------------------------------------------
    // MpcCode type alias and MpcCodeObs map
    // -----------------------------------------------------------------------

    /// Verifies that an `MpcCodeObs` map correctly stores and retrieves an
    /// `Observer` keyed by a `MpcCode` (`[u8; 3]`).
    ///
    /// This exercises the type aliases end-to-end: insertion uses the
    /// byte-array key, and lookup must return the exact same `Observer`.
    #[test]
    fn mpc_code_key_lookup() {
        let key: MpcCode = *b"G96";

        // Construct a valid Observer using known parallax constants for G96
        // (Catalina Sky Survey). unwrap() is safe: none of the inputs are NaN.
        let observer = Observer::from_parallax(
            110.789_f64, // longitude (degrees)
            0.836_f64,   // rho_cos_phi
            0.547_f64,   // rho_sin_phi
            Some("Catalina Sky Survey".to_string()),
            None,
            None,
        )
        .unwrap(); // safe: all inputs are finite, non-NaN values

        let mut map = MpcCodeObs::new();
        map.insert(key, observer.clone());

        let found = map.get(&key);
        assert!(
            found.is_some(),
            "Expected to find observer under key b\"G96\", but got None"
        );
        assert_eq!(
            found.unwrap(),
            &observer,
            "Retrieved observer does not match the inserted one"
        );
    }

    // -----------------------------------------------------------------------
    // init_observatories — integration test with a ureq Middleware mock
    // -----------------------------------------------------------------------

    /// Synthetic MPC observatory list used by the mock middleware.
    ///
    /// The format follows the real MPC fixed-width layout:
    ///   - First two lines are skipped (HTML header rows).
    ///   - Each data line: 3-char code, then fixed-width fields.
    ///
    /// Lines taken from the real MPC list provided in the task description.
    const MOCK_MPC_DOCUMENT: &str = "\
Code  Long.   cos      sin    Name\n\
\n\
000   0.0000 0.62411 +0.77873 Greenwich\n\
001   0.1542 0.62992 +0.77411 Crowborough\n\
002   0.62   0.622   +0.781   Rayleigh\n\
005   2.231000.659891+0.748875Meudon\n\
006   2.124170.751042+0.658129Fabra Observatory, Barcelona\n\
";

    /// A ureq [`Middleware`] that short-circuits every HTTP request and returns
    /// the fixed `MOCK_MPC_DOCUMENT` string as a 200 OK response, without
    /// touching the network.
    struct MockMpcMiddleware;

    impl ureq::middleware::Middleware for MockMpcMiddleware {
        fn handle(
            &self,
            _request: ureq::http::Request<ureq::SendBody>,
            _next: ureq::middleware::MiddlewareNext,
        ) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
            let body = ureq::Body::builder()
                .mime_type("text/plain")
                .data(MOCK_MPC_DOCUMENT.as_bytes().to_vec());

            Ok(ureq::http::Response::builder()
                .status(200)
                .body(body)
                .expect("valid response"))
        }
    }

    /// Build a [`ureq::Agent`] that intercepts every request with
    /// [`MockMpcMiddleware`] so no real network call is made.
    fn mock_agent() -> ureq::Agent {
        ureq::config::Config::builder()
            .middleware(MockMpcMiddleware)
            .build()
            .new_agent()
    }

    /// Verifies that [`init_observatories`] correctly parses the mock MPC
    /// document and returns a map with the expected entries.
    ///
    /// The mock skips the first two header lines, then parses five data lines.
    /// Lines with parseable parallax constants should produce `Observer`
    /// entries; lines whose numeric fields parse as `0.0` are still inserted
    /// as long as `Observer::from_parallax` succeeds.
    #[test]
    fn init_observatories_parses_mock_document() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let agent = mock_agent();
        let error_model: crate::observer::error_model::ErrorModelData =
            std::collections::HashMap::new(); // empty → all accuracies None

        let result = init_observatories_impl(agent, &error_model, Some(tmp.path()));
        assert!(
            result.is_ok(),
            "Expected Ok from init_observatories, got: {:?}",
            result
        );

        let map = result.unwrap();

        // All five data lines have codes long enough for split_at_checked(3)
        // and remain strings long enough for parse_remain to return Some.
        assert!(
            map.contains_key(b"000"),
            "Expected code '000' (Greenwich) to be in the map"
        );
        assert!(
            map.contains_key(b"001"),
            "Expected code '001' (Crowborough) to be in the map"
        );
        assert!(
            map.contains_key(b"005"),
            "Expected code '005' (Meudon) to be in the map"
        );
        assert!(
            map.contains_key(b"006"),
            "Expected code '006' (Fabra Observatory) to be in the map"
        );
    }

    /// Verifies that the `Observer` for Greenwich (code `000`) has the correct
    /// site name parsed from the mock document.
    #[test]
    fn init_observatories_observer_name_is_correct() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let agent = mock_agent();
        let error_model: crate::observer::error_model::ErrorModelData =
            std::collections::HashMap::new();

        let map = init_observatories_impl(agent, &error_model, Some(tmp.path())).unwrap();

        let greenwich = map.get(b"000").expect("Greenwich must be present");
        assert_eq!(
            greenwich.name.as_deref(),
            Some("Greenwich"),
            "Observer name for code '000' should be 'Greenwich'"
        );
    }

    /// Verifies that a document containing only the two header lines (no data)
    /// produces an empty map rather than an error.
    #[test]
    fn init_observatories_empty_document_produces_empty_map() {
        struct EmptyMpcMiddleware;

        impl ureq::middleware::Middleware for EmptyMpcMiddleware {
            fn handle(
                &self,
                _request: ureq::http::Request<ureq::SendBody>,
                _next: ureq::middleware::MiddlewareNext,
            ) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
                let body = ureq::Body::builder()
                    .mime_type("text/plain")
                    .data("header line 1\nheader line 2\n".as_bytes().to_vec());

                Ok(ureq::http::Response::builder()
                    .status(200)
                    .body(body)
                    .expect("valid response"))
            }
        }

        let agent = ureq::config::Config::builder()
            .middleware(EmptyMpcMiddleware)
            .build()
            .new_agent();

        let tmp = tempfile::tempdir().expect("tempdir");
        let map =
            init_observatories_impl(agent, &std::collections::HashMap::new(), Some(tmp.path()))
                .unwrap();
        assert!(
            map.is_empty(),
            "Expected empty map for a document with only header lines, got {} entries",
            map.len()
        );
    }

    /// Verifies that malformed lines (too short to split at byte 3) are
    /// silently skipped and do not cause a panic or error.
    #[test]
    fn init_observatories_skips_malformed_lines() {
        struct MalformedMpcMiddleware;

        impl ureq::middleware::Middleware for MalformedMpcMiddleware {
            fn handle(
                &self,
                _request: ureq::http::Request<ureq::SendBody>,
                _next: ureq::middleware::MiddlewareNext,
            ) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
                // Mix of: 2 header lines, a 1-char line (too short), a blank,
                // and one valid line.
                let content = "hdr\nhdr\nX\n\n000   0.0000 0.62411 +0.77873 Greenwich\n";
                let body = ureq::Body::builder()
                    .mime_type("text/plain")
                    .data(content.as_bytes().to_vec());

                Ok(ureq::http::Response::builder()
                    .status(200)
                    .body(body)
                    .expect("valid response"))
            }
        }

        let agent = ureq::config::Config::builder()
            .middleware(MalformedMpcMiddleware)
            .build()
            .new_agent();

        let tmp = tempfile::tempdir().expect("tempdir");
        let map =
            init_observatories_impl(agent, &std::collections::HashMap::new(), Some(tmp.path()))
                .unwrap();
        // Only '000' should survive; the short line must be silently skipped.
        assert_eq!(map.len(), 1, "Expected exactly 1 entry, got {}", map.len());
        assert!(
            map.contains_key(b"000"),
            "Expected code '000' to be present"
        );
    }

    // -----------------------------------------------------------------------
    // Cache hit / miss tests
    // -----------------------------------------------------------------------

    /// A ureq middleware that panics if called — used to assert no network
    /// request is made when the cache is warm.
    struct PanicMiddleware;

    impl ureq::middleware::Middleware for PanicMiddleware {
        fn handle(
            &self,
            _request: ureq::http::Request<ureq::SendBody>,
            _next: ureq::middleware::MiddlewareNext,
        ) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
            panic!("network request was made despite a warm cache");
        }
    }

    fn panic_agent() -> ureq::Agent {
        ureq::config::Config::builder()
            .middleware(PanicMiddleware)
            .build()
            .new_agent()
    }

    /// Verifies that when a valid cache file exists, `init_observatories`
    /// reads from it and never calls the network (the panic agent would
    /// explode if the network were touched).
    #[test]
    fn init_observatories_uses_cache_when_warm() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cache_file = tmp.path().join("mpc_obs.html");

        // Pre-populate the cache with the mock document.
        std::fs::write(&cache_file, MOCK_MPC_DOCUMENT).expect("write cache");

        let map = init_observatories_impl(
            panic_agent(),
            &std::collections::HashMap::new(),
            Some(tmp.path()),
        )
        .expect("init_observatories must succeed with warm cache");

        assert!(
            map.contains_key(b"000"),
            "Expected '000' (Greenwich) from cached document"
        );
        assert!(
            map.contains_key(b"005"),
            "Expected '005' (Meudon) from cached document"
        );
    }

    /// Verifies that when the cache is absent, `init_observatories` calls the
    /// network (via mock agent) and then writes the cache file to disk.
    #[test]
    fn init_observatories_writes_cache_on_miss() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cache_file = tmp.path().join("mpc_obs.html");

        // Ensure no pre-existing cache.
        assert!(!cache_file.exists(), "cache must not exist before the test");

        let map = init_observatories_impl(
            mock_agent(),
            &std::collections::HashMap::new(),
            Some(tmp.path()),
        )
        .expect("init_observatories must succeed on cache miss");

        // The network was called (mock returned MOCK_MPC_DOCUMENT).
        assert!(
            map.contains_key(b"000"),
            "Expected '000' from network response"
        );

        // Cache file must now exist and contain the document.
        assert!(
            cache_file.exists(),
            "cache file must be written after a miss"
        );
        let cached = std::fs::read_to_string(&cache_file).expect("read cache");
        assert!(
            cached.contains("Greenwich"),
            "Cache file should contain the network response body"
        );
    }
}
