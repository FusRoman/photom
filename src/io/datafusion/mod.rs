//! DataFusion-based Parquet ingestion for [`ObsDataset`](crate::observation_dataset::ObsDataset).
//!
//! This module is the `datafusion` feature-gate entry point of the `photom`
//! I/O layer.  It provides a complete, end-to-end pipeline for loading an
//! [`ObsDataset`](crate::observation_dataset::ObsDataset) from a Parquet file
//! located at any URI supported by the [`storage`] resolver.  It sits alongside the `polars`, `ades`, and
//! `mpc_80_col` backends inside [`crate::io`] and shares the same column
//! schema convention documented in the crate root.
//!
//! # Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`input_uri`] | [`InputUri`] newtype — a validated URI string forwarded to the storage resolver |
//! | [`storage`] | URI → object-store resolution: maps a scheme to an [`object_store`](https://docs.rs/object_store) backend and returns a [`ResolvedObject`] |
//! | [`loader`] | DataFusion query engine — scans the Parquet file, collects Arrow [`RecordBatch`](arrow_array::RecordBatch)es, and converts them into an [`ObsDataset`](crate::observation_dataset::ObsDataset) |
//!
//! # Supported URI Schemes
//!
//! | Scheme | Object-store backend | Notes |
//! |--------|---------------------|-------|
//! | `file://` | [`LocalFileSystem`](object_store::local::LocalFileSystem) anchored at `/` | Absolute paths: `file:///data/obs.parquet` |
//! | `http://` | [`HttpBuilder`](object_store::http::HttpBuilder) | Plain HTTP remote store |
//! | `https://` | [`HttpBuilder`](object_store::http::HttpBuilder) | TLS-encrypted HTTP remote store |
//! | `hdfs://` | `HdfsObjectStoreBuilder` | Requires a `host` authority: `hdfs://namenode:9870/path` |
//!
//! Any other scheme causes [`UriStoreError::UnsupportedScheme`] to be returned.
//!
//! # Ingestion Pipeline
//!
//! The pipeline proceeds through the following stages in order:
//!
//! 1. **[`InputUri`]** — the caller supplies a raw URI string wrapped in
//!    [`InputUri`].  The newtype delegates URL parsing to the [`url`] crate so
//!    that scheme and path information can be extracted without re-parsing on
//!    every access.
//!
//! 2. **[`resolve_url`]** — the parsed [`url::Url`] is dispatched to the
//!    scheme-specific handler inside [`storage`].  The handler constructs the
//!    appropriate [`object_store`](https://docs.rs/object_store) backend and
//!    returns a [`ResolvedObject`] containing both the backend instance and the
//!    relative in-store path.
//!
//! 3. **Object-store existence check** — for `file://` and `hdfs://` URIs the
//!    loader calls [`ObjectStore::head`](object_store::ObjectStore::head) to
//!    verify the file exists before opening DataFusion, returning
//!    [`LoadObsError::NotFound`] early if it does not.
//!
//! 4. **DataFusion session** — a [`SessionContext`](datafusion::prelude::SessionContext)
//!    is created and the resolved object store is registered under the original
//!    URI authority so that DataFusion's I/O layer can address it.
//!
//! 5. **Parquet scan** — a [`ListingTable`](datafusion::datasource::listing::ListingTable)
//!    is opened over the URI with schema inference.  If a
//!    [`ContiguousChoice`](loader::ContiguousChoice) is set in [`LoadObsArgs`],
//!    an `ORDER BY` clause is appended before collection so that all rows of the
//!    chosen group form a contiguous block in the output vector.
//!
//! 6. **Arrow [`RecordBatch`](arrow_array::RecordBatch) collection** — DataFusion
//!    executes the plan and materialises the result as a sequence of Arrow record
//!    batches.
//!
//! 7. **[`ObsDataset`](crate::observation_dataset::ObsDataset) construction** — the batches are iterated row-by-row.
//!    Mandatory columns are validated for presence and type; observer columns are
//!    resolved according to the rules documented in [`loader`]; optional index
//!    columns (`night_id`, `traj_id`) are used to build compact
//!    contiguous-range or split-vector index entries.
//!
//! # Async vs Sync Entry-points
//!
//! Two public entry-points are provided so the pipeline is usable from both
//! async and synchronous contexts:
//!
//! - **[`load_obs_from_parquet_uri`]** (async) — must be driven by an existing
//!   Tokio runtime.  Prefer this form when the caller already runs inside an
//!   `async` context to avoid blocking a thread-pool worker.
//! - **[`load_obs_sync`]** (sync wrapper) — creates an internal single-threaded
//!   Tokio runtime via [`tokio::runtime::Runtime::new`] and blocks until the
//!   async pipeline completes.  Use this from ordinary `fn` callsites that
//!   cannot `.await`.
//!
//! Both functions accept the same [`LoadObsArgs`] configuration struct and
//! return the same [`Result<ObsDataset, LoadObsError>`].
//!
//! # Re-exports
//!
//! The most commonly used items are re-exported directly from this module so
//! that callers can import them without navigating into sub-modules:
//!
//! ```text
//! use photom::io::datafusion::{
//!     InputUri,
//!     LoadObsArgs, LoadObsError,
//!     load_obs_sync, load_obs_from_parquet_uri,
//!     ResolvedObject, UriStoreError,
//!     resolve_input_uri, resolve_url,
//! };
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use photom::io::datafusion::{
//!     InputUri,
//!     LoadObsArgs,
//!     load_obs_sync,
//! };
//!
//! // Point at a local Parquet file.
//! let uri = InputUri("file:///data/observations.parquet".to_string());
//!
//! // Load with default arguments (night-contiguous index, no error model).
//! let dataset = load_obs_sync(&uri, LoadObsArgs::default())
//!     .expect("failed to load observations");
//!
//! println!("{} observations loaded", dataset.observation_count());
//!
//! // Iterate over every observation.
//! for obs in dataset.iter_observations() {
//!     println!(
//!         "id={} ra={:.6} dec={:.6} mag={:.2}",
//!         obs.id(),
//!         obs.equ_coord.ra,
//!         obs.equ_coord.dec,
//!         obs.photometry.magnitude,
//!     );
//! }
//! ```
//!
//! See [`loader`] for the full column schema, observer resolution rules, and
//! the [`LoadObsArgs`] configuration reference.

pub mod input_uri;
pub mod loader;
pub mod storage;

pub use input_uri::InputUri;
pub use loader::{LoadObsArgs, LoadObsError, load_obs_from_parquet_uri, load_obs_sync};
pub use storage::{ResolvedObject, UriStoreError, resolve_input_uri, resolve_url};
