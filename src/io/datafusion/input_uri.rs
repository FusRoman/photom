//! URI wrapper for DataFusion-based Parquet ingestion.
//!
//! This module defines [`InputUri`], the entry-point type accepted by both
//! [`load_obs_sync`](crate::io::datafusion::loader::load_obs_sync) and
//! [`load_obs_from_parquet_uri`](crate::io::datafusion::loader::load_obs_from_parquet_uri).
//! Callers construct an [`InputUri`] from any string that identifies a Parquet
//! resource, and the loader pipeline resolves the correct object-store backend
//! from the URI scheme.
//!
//! # Transparent newtype
//!
//! [`InputUri`] is a transparent newtype around a plain [`String`].  The
//! `#[serde(transparent)]` attribute means it serialises and deserialises as a
//! bare JSON string — no wrapper object is inserted:
//!
//! ```text
//! "file:///data/observations.parquet"   // ← JSON representation
//! ```
//!
//! # Supported schemes
//!
//! The storage resolver ([`crate::io::datafusion::storage`]) maps URI schemes
//! to object-store backends as follows:
//!
//! | Scheme | Object-store backend |
//! |--------|---------------------|
//! | `file://` | Local filesystem |
//! | `http://` / `https://` | HTTP(S) remote object store |
//! | `hdfs://` | HDFS via `hdfs-native-object-store` |
//!
//! # Construction
//!
//! Two equivalent ways to create an [`InputUri`]:
//!
//! ```rust,ignore
//! use photom::io::datafusion::InputUri;
//!
//! // Direct construction from a String literal.
//! let uri = InputUri("file:///data/observations.parquet".to_string());
//!
//! // Via the `FromStr` impl — useful when parsing from user input.
//! let uri: InputUri = "https://example.com/obs.parquet".parse().unwrap();
//! ```

use std::{fmt::Display, str::FromStr};

use serde::{Deserialize, Serialize};
use url::Url;

/// A raw URI string that points to a Parquet resource.
///
/// `InputUri` is a transparent newtype over [`String`] annotated with
/// `#[serde(transparent)]` so that it serialises as a bare JSON string rather
/// than a single-field object.  This makes it ergonomic to embed inside larger
/// configuration structs that are read from JSON or TOML.
///
/// The inner string is not validated at construction time; call [`parse`] to
/// obtain a fully validated [`Url`] when scheme or host information is needed.
/// The [`Display`] implementation simply forwards to the inner string, so
/// [`InputUri`] can be used wherever a URI string is expected without
/// unwrapping.
///
/// Supported schemes (resolved by [`crate::io::datafusion::storage`]):
///
/// - `file://` — local filesystem
/// - `http://` / `https://` — HTTP(S) remote object store
/// - `hdfs://` — HDFS via `hdfs-native-object-store`
///
/// [`parse`]: InputUri::parse
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InputUri(pub String);

impl InputUri {
    /// Parse the inner string as a [`Url`].
    ///
    /// Delegates to [`Url::parse`] from the [`url`] crate.  The resulting
    /// [`Url`] gives structured access to the scheme, host, port, and path
    /// without re-parsing the string on every access.
    ///
    /// # Errors
    ///
    /// Returns a [`url::ParseError`] if the inner string is not a valid
    /// absolute URL (e.g. missing scheme, illegal characters, or a relative
    /// reference without a base).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use photom::io::datafusion::InputUri;
    ///
    /// let uri = InputUri("https://example.com/data.parquet".to_string());
    /// let url = uri.parse().expect("URI must be valid");
    /// assert_eq!(url.scheme(), "https");
    /// assert_eq!(url.host_str(), Some("example.com"));
    /// assert_eq!(url.path(), "/data.parquet");
    /// ```
    pub fn parse(&self) -> Result<Url, url::ParseError> {
        Url::parse(&self.0)
    }
}

/// Constructs an [`InputUri`] by wrapping the given string slice without
/// validation.
///
/// The conversion is infallible — any string is accepted and stored as-is.
/// Call [`InputUri::parse`] afterwards when a valid URL is required.
impl FromStr for InputUri {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(InputUri(s.to_string()))
    }
}

/// Formats the [`InputUri`] as its raw URI string.
///
/// The output is identical to the inner [`String`], making it suitable for
/// logging, error messages, and anywhere a plain URI string is expected.
impl Display for InputUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod input_uri_tests {
    use super::*;

    #[test]
    fn parse_valid_http_uri() {
        let uri = InputUri("https://example.com/data.parquet".to_string());
        let parsed = uri.parse().expect("URI should parse");
        assert_eq!(parsed.scheme(), "https");
        assert_eq!(parsed.host_str(), Some("example.com"));
        assert_eq!(parsed.path(), "/data.parquet");
    }

    #[test]
    fn parse_valid_file_uri() {
        let uri = InputUri("file:///tmp/test.parquet".to_string());
        let parsed = uri.parse().expect("file URI should parse");
        assert_eq!(parsed.scheme(), "file");
        assert_eq!(parsed.path(), "/tmp/test.parquet");
    }

    #[test]
    fn parse_invalid_uri() {
        let uri = InputUri("not a valid uri".to_string());
        assert!(uri.parse().is_err(), "invalid URI should return error");
    }

    #[test]
    fn serde_roundtrip() {
        let uri = InputUri("https://example.com/foo".to_string());
        let json = serde_json::to_string(&uri).expect("serialization should work");
        let de: InputUri = serde_json::from_str(&json).expect("deserialization should work");
        assert_eq!(uri.0, de.0);
    }

    #[test]
    fn serde_transparent_representation() {
        let uri = InputUri("https://example.com/bar".to_string());
        let json = serde_json::to_string(&uri).unwrap();
        assert_eq!(json, "\"https://example.com/bar\"");
    }

    #[test]
    fn display_round_trips_inner_string() {
        let uri = InputUri("file:///data/obs.parquet".to_string());
        assert_eq!(uri.to_string(), "file:///data/obs.parquet");
    }
}
