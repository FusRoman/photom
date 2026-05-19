//! Object-store resolution for [`InputUri`].
//!
//! This module is the bridge between a raw URI string and the
//! [`ObjectStore`] / [`ObjPath`] pair that the DataFusion loader needs to open
//! a Parquet resource.  Its two public entry-points —
//! [`resolve_input_uri`] and [`resolve_url`] — accept either an [`InputUri`]
//! newtype or an already-parsed [`url::Url`] and return a [`ResolvedObject`]
//! that bundles:
//!
//! - an `Arc<dyn ObjectStore>` — the concrete I/O backend configured for the
//!   given scheme and authority, and
//! - an [`ObjPath`](object_store::path::Path) — the key within that store
//!   that identifies the resource.
//!
//! The caller (typically the DataFusion loader in
//! [`crate::io::datafusion::loader`]) registers the store with a
//! [`datafusion::prelude::SessionContext`] and opens the resource via the
//! returned path.
//!
//! # Scheme dispatch
//!
//! | Scheme | Backend | Path normalisation |
//! |--------|---------|--------------------|
//! | `file` | [`LocalFileSystem`] anchored at `/` | Leading `/` stripped; `file:///tmp/a.parquet` → `tmp/a.parquet` |
//! | `http` / `https` | [`HttpBuilder`] | Full URL string used as the object key |
//! | `hdfs` | [`HdfsObjectStoreBuilder`] | Leading `/` stripped; store rooted at `hdfs://<host>[:<port>]` |
//!
//! ## `file://` path normalisation
//!
//! [`LocalFileSystem`] is constructed with a prefix of `"/"`, which anchors it
//! at the filesystem root.  The leading `/` of the URL path component is then
//! stripped before constructing the [`ObjPath`], so that the path is relative
//! to the store root as required by the `object_store` API.  For example,
//! `file:///tmp/test.parquet` yields the in-store key `tmp/test.parquet`.

use std::sync::Arc;

use hdfs_native_object_store::HdfsObjectStoreBuilder;
use object_store::Error as ObjStoreError;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, path::Path as ObjPath};
use url::Url;

use crate::io::datafusion::input_uri::InputUri;

// ── public types ─────────────────────────────────────────────────────────────

/// A resolved object-store backend paired with its in-store resource key.
///
/// [`ResolvedObject`] is the output of [`resolve_input_uri`] and
/// [`resolve_url`].  It bundles two pieces of information that downstream
/// consumers — primarily the DataFusion loader — need to open a Parquet
/// resource:
///
/// - [`store`](ResolvedObject::store) — an `Arc<dyn `[`ObjectStore`]`>`
///   configured for the scheme and authority of the original URI.  The arc
///   allows cheap cloning and enables registration with a
///   [`datafusion::prelude::SessionContext`].
/// - [`path`](ResolvedObject::path) — the [`ObjPath`] key *within* the store
///   that identifies the resource.  The path is always relative to the store
///   root (leading `/` stripped where required by the `object_store` API).
///
/// ## `Debug` representation
///
/// The [`std::fmt::Debug`] implementation prints only the `path` field and
/// omits the `store`.  This avoids verbose or potentially sensitive backend
/// details in debug output; the store is marked `..` via
/// [`finish_non_exhaustive`](std::fmt::DebugStruct::finish_non_exhaustive).
#[derive(Clone)]
pub struct ResolvedObject {
    /// The object-store backend to use for I/O.
    pub store: Arc<dyn ObjectStore>,
    /// The in-store path (relative to the store root).
    pub path: ObjPath,
}

/// Formats `ResolvedObject` for debugging, showing only the `path` field.
///
/// The `store` field is omitted (replaced by `..`) because `dyn ObjectStore`
/// implementations do not guarantee a meaningful `Debug` representation and
/// may contain connection strings or credentials.
impl std::fmt::Debug for ResolvedObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedObject")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// Errors that can occur during URI → object-store resolution.
///
/// Each variant corresponds to a distinct failure mode in the resolution
/// pipeline:
///
/// - **[`InvalidUri`](UriStoreError::InvalidUri)** — the raw string supplied
///   to [`resolve_input_uri`] could not be parsed as a URL by the [`url`]
///   crate.  The inner `String` is the original, unparsed URI.
///
/// - **[`UnsupportedScheme`](UriStoreError::UnsupportedScheme)** — the URL
///   was parsed successfully but its scheme is not handled by this resolver.
///   Only `file`, `http`, `https`, and `hdfs` are supported.  The inner
///   `String` is the unrecognised scheme name.
///
/// - **[`MissingAuthority`](UriStoreError::MissingAuthority)** — an
///   `hdfs://` URI was supplied that has no host (authority) component, e.g.
///   `hdfs:///path/file`.  HDFS requires a namenode host to build the store.
///   The inner `String` is the full URI.
///
/// - **[`ObjectStore`](UriStoreError::ObjectStore)** — an
///   [`object_store::Error`] was returned during store construction (e.g. an
///   HTTP store built without a base URL).  The source error is accessible via
///   [`std::error::Error::source`].
///
/// - **[`HdfsBuild`](UriStoreError::HdfsBuild)** — the
///   [`HdfsObjectStoreBuilder`] returned an error that is not an
///   [`object_store::Error`].  The inner `String` is the `Display`
///   representation of the underlying error.
///
/// # Errors
///
/// The variants are produced in the following situations:
///
/// | Variant | Produced by |
/// |---------|-------------|
/// | [`InvalidUri`](UriStoreError::InvalidUri) | [`resolve_input_uri`] when `url::Url::parse` fails |
/// | [`UnsupportedScheme`](UriStoreError::UnsupportedScheme) | [`resolve_url`] for any scheme outside `file`, `http`, `https`, `hdfs` |
/// | [`MissingAuthority`](UriStoreError::MissingAuthority) | [`resolve_url`] for `hdfs://` URIs without a host |
/// | [`ObjectStore`](UriStoreError::ObjectStore) | [`resolve_url`] when [`LocalFileSystem`] or [`HttpBuilder`] construction fails |
/// | [`HdfsBuild`](UriStoreError::HdfsBuild) | [`resolve_url`] when [`HdfsObjectStoreBuilder::build`] returns an error |
#[derive(Debug)]
pub enum UriStoreError {
    /// The URI string could not be parsed as a URL.
    InvalidUri(String),
    /// The URI scheme is not supported by this resolver.
    UnsupportedScheme(String),
    /// An `hdfs://` URI was supplied without a host (authority) component.
    MissingAuthority(String),
    /// An `object_store` builder or I/O error.
    ObjectStore(ObjStoreError),
    /// The HDFS object-store builder returned an error.
    HdfsBuild(String),
}

/// Formats each [`UriStoreError`] variant as a human-readable error message.
impl std::fmt::Display for UriStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UriStoreError::InvalidUri(s) => write!(f, "invalid URI: {s}"),
            UriStoreError::UnsupportedScheme(s) => write!(f, "unsupported URI scheme: {s}"),
            UriStoreError::MissingAuthority(s) => {
                write!(f, "HDFS URI missing authority (host): {s}")
            }
            UriStoreError::ObjectStore(e) => write!(f, "object store error: {e}"),
            UriStoreError::HdfsBuild(s) => write!(f, "HDFS store build error: {s}"),
        }
    }
}

/// Implements [`std::error::Error`] for [`UriStoreError`].
///
/// [`source`](std::error::Error::source) returns the underlying
/// [`object_store::Error`] for the [`ObjectStore`](UriStoreError::ObjectStore)
/// variant, enabling error-chain traversal.  All other variants return `None`.
impl std::error::Error for UriStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UriStoreError::ObjectStore(e) => Some(e),
            _ => None,
        }
    }
}

/// Converts an [`object_store::Error`] into a
/// [`UriStoreError::ObjectStore`] variant.
///
/// This allows the `?` operator to be used in scheme handlers that call
/// `object_store` builder methods.
impl From<ObjStoreError> for UriStoreError {
    fn from(e: ObjStoreError) -> Self {
        UriStoreError::ObjectStore(e)
    }
}

// ── resolution entry-points ───────────────────────────────────────────────────

/// Resolve an [`InputUri`] into an object-store backend and an in-store path.
///
/// This is the primary entry-point for callers that hold an [`InputUri`]
/// newtype.  The function first parses the inner string as a [`url::Url`] and
/// then delegates to [`resolve_url`] for scheme dispatch and store
/// construction.
///
/// # Arguments
///
/// - `uri` — the [`InputUri`] to resolve.  Its inner string must be a valid
///   URL with a scheme supported by this resolver (`file`, `http`, `https`,
///   or `hdfs`).
///
/// # Returns
///
/// A [`ResolvedObject`] containing:
/// - `store` — the concrete [`ObjectStore`] backend for the URI's scheme and
///   authority.
/// - `path` — the in-store key (leading `/` stripped for `file://` and
///   `hdfs://` URIs).
///
/// # Errors
///
/// - [`UriStoreError::InvalidUri`] — the inner string of `uri` is not a valid
///   URL.
/// - All variants produced by [`resolve_url`] for scheme-specific failures.
///
/// # Example
///
/// ```rust,ignore
/// use photom::io::datafusion::input_uri::InputUri;
/// use photom::io::datafusion::storage::resolve_input_uri;
///
/// let uri = InputUri("file:///data/observations.parquet".to_string());
/// let resolved = resolve_input_uri(&uri).expect("resolution should succeed");
/// assert_eq!(resolved.path.as_ref(), "data/observations.parquet");
/// ```
pub fn resolve_input_uri(uri: &InputUri) -> Result<ResolvedObject, UriStoreError> {
    let url = uri
        .parse()
        .map_err(|_| UriStoreError::InvalidUri(uri.0.clone()))?;
    resolve_url(&url)
}

/// Resolve a parsed [`Url`] into an object-store backend and an in-store path.
///
/// Dispatches to a scheme-specific handler based on [`Url::scheme`]:
///
/// | Scheme | Handler | Backend |
/// |--------|---------|---------|
/// | `file` | internal | [`LocalFileSystem`] anchored at `/` |
/// | `http` | internal | [`HttpBuilder`] |
/// | `https` | internal | [`HttpBuilder`] |
/// | `hdfs` | internal | [`HdfsObjectStoreBuilder`] |
///
/// # Arguments
///
/// - `url` — a fully parsed [`url::Url`] whose scheme is one of `file`,
///   `http`, `https`, or `hdfs`.
///
/// # Returns
///
/// A [`ResolvedObject`] containing:
/// - `store` — the [`ObjectStore`] backend constructed for the URL's scheme
///   and authority.
/// - `path` — the in-store resource key.  For `file://` and `hdfs://` URIs
///   the leading `/` of the URL path is stripped so that the key is relative
///   to the store root.  For `http://` / `https://` URIs the full URL string
///   is used as the object key, as required by the HTTP object-store backend.
///
/// # Errors
///
/// - [`UriStoreError::UnsupportedScheme`] — the URL scheme is not one of
///   `file`, `http`, `https`, or `hdfs`.
/// - [`UriStoreError::ObjectStore`] — [`LocalFileSystem`] or [`HttpBuilder`]
///   construction failed.
/// - [`UriStoreError::MissingAuthority`] — the `hdfs` URL has no host
///   component.
/// - [`UriStoreError::HdfsBuild`] — [`HdfsObjectStoreBuilder::build`]
///   returned an error.
pub fn resolve_url(url: &Url) -> Result<ResolvedObject, UriStoreError> {
    match url.scheme() {
        "file" => resolve_file(url),
        "http" | "https" => resolve_http(url),
        "hdfs" => resolve_hdfs(url),
        other => Err(UriStoreError::UnsupportedScheme(other.to_string())),
    }
}

// ── scheme handlers ───────────────────────────────────────────────────────────

fn resolve_file(url: &Url) -> Result<ResolvedObject, UriStoreError> {
    // Anchor at "/" so absolute filesystem paths become relative object-store
    // paths by stripping the leading '/'.
    let store = Arc::new(LocalFileSystem::new_with_prefix("/")?);
    let rel = url.path().trim_start_matches('/');
    Ok(ResolvedObject {
        store,
        path: ObjPath::from(rel),
    })
}

fn resolve_http(url: &Url) -> Result<ResolvedObject, UriStoreError> {
    let store = Arc::new(HttpBuilder::new().build()?);
    // The HTTP store expects the full URL encoded as the object path.
    Ok(ResolvedObject {
        store,
        path: ObjPath::from(url.as_str()),
    })
}

fn resolve_hdfs(url: &Url) -> Result<ResolvedObject, UriStoreError> {
    let host = url
        .host_str()
        .ok_or_else(|| UriStoreError::MissingAuthority(url.as_str().to_string()))?;

    let base = match url.port() {
        Some(port) => format!("hdfs://{host}:{port}"),
        None => format!("hdfs://{host}"),
    };

    let rel = url.path().trim_start_matches('/');

    let hdfs_store = HdfsObjectStoreBuilder::new()
        .with_url(base)
        .build()
        .map_err(|e| UriStoreError::HdfsBuild(e.to_string()))?;

    Ok(ResolvedObject {
        store: Arc::new(hdfs_store),
        path: ObjPath::from(rel),
    })
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod datafusion_storage_tests {
    use super::*;

    fn url(s: &str) -> Url {
        Url::parse(s).expect("test URL must parse")
    }

    // ── InputUri -> Url parsing plumbing ─────────────────────────────────────

    #[test]
    fn resolve_input_uri_invalid_uri_maps_to_invalid_uri() {
        let uri = InputUri("not a uri".to_string());
        match resolve_input_uri(&uri).unwrap_err() {
            UriStoreError::InvalidUri(s) => assert_eq!(s, "not a uri"),
            other => panic!("expected InvalidUri, got: {other:?}"),
        }
    }

    // ── scheme dispatch ───────────────────────────────────────────────────────

    #[test]
    fn resolve_url_unsupported_scheme() {
        let u = url("s3://bucket/key");
        match resolve_url(&u).unwrap_err() {
            UriStoreError::UnsupportedScheme(s) => assert_eq!(s, "s3"),
            other => panic!("expected UnsupportedScheme, got: {other:?}"),
        }
    }

    // ── file:// ───────────────────────────────────────────────────────────────

    #[test]
    fn resolve_file_makes_absolute_fs_path_relative() {
        let u = url("file:///tmp/test.parquet");
        let resolved = resolve_url(&u).expect("file:// should resolve");
        assert_eq!(resolved.path.as_ref(), "tmp/test.parquet");
    }

    #[test]
    fn resolve_file_root_path_is_empty_relative_path() {
        let u = url("file:///");
        let resolved = resolve_url(&u).expect("file:/// should resolve");
        assert_eq!(resolved.path.as_ref(), "");
    }

    // ── hdfs:// ───────────────────────────────────────────────────────────────

    #[test]
    fn resolve_hdfs_missing_authority_is_error() {
        let u = url("hdfs:///data/file.parquet");
        match resolve_url(&u).unwrap_err() {
            UriStoreError::MissingAuthority(s) => {
                assert_eq!(s, "hdfs:///data/file.parquet");
            }
            other => panic!("expected MissingAuthority, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_hdfs_returns_relative_path_without_leading_slash() {
        let u = url("hdfs://localhost:9870/some/path.parquet");
        let resolved = resolve_url(&u).expect("builder may succeed without contacting HDFS");
        assert_eq!(resolved.path.as_ref(), "some/path.parquet");
    }

    // ── http(s):// ────────────────────────────────────────────────────────────

    #[test]
    fn resolve_http_without_base_url_returns_object_store_error() {
        let u = url("http://example.com/data.parquet");
        match resolve_url(&u).unwrap_err() {
            UriStoreError::ObjectStore(e) => {
                let s = format!("{e:?}");
                assert!(s.contains("HTTP"), "expected HTTP store error, got: {s}");
                assert!(s.contains("MissingUrl"), "expected MissingUrl, got: {s}");
            }
            other => panic!("expected ObjectStore error, got: {other:?}"),
        }
    }

    #[test]
    fn resolve_https_without_base_url_returns_object_store_error() {
        let u = url("https://example.com/a/b/c");
        match resolve_url(&u).unwrap_err() {
            UriStoreError::ObjectStore(e) => {
                let s = format!("{e:?}");
                assert!(s.contains("HTTP"), "expected HTTP store error, got: {s}");
                assert!(s.contains("MissingUrl"), "expected MissingUrl, got: {s}");
            }
            other => panic!("expected ObjectStore error, got: {other:?}"),
        }
    }
}
