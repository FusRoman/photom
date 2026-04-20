//! DataFusion-based Parquet loader for [`ObsDataset`].
//!
//! This module bridges the Arrow columnar world and the domain model: it reads
//! one or more Parquet files from any URI supported by the storage resolver,
//! streams the resulting Arrow [`RecordBatch`]es through a single validation
//! and assembly pass, and returns a fully populated [`ObsDataset`] with
//! optional night/trajectory index maps and an optional astrometric error
//! model.
//!
//! # Entry-points
//!
//! | Function | Style | Description |
//! |----------|-------|-------------|
//! | [`load_obs_sync`] | blocking | Wraps the async function in a single-threaded Tokio runtime |
//! | [`load_obs_from_parquet_uri`] | `async` | Full pipeline: URI → object-store → DataFusion → [`ObsDataset`] |
//!
//! # URI schemes
//!
//! | Scheme | Backend |
//! |--------|---------|
//! | `file://` | Local filesystem |
//! | `http://` / `https://` | HTTP(S) remote store |
//! | `hdfs://` | HDFS via `hdfs-native-object-store` |
//!
//! # Mandatory column schema
//!
//! The Parquet file **must** contain the following columns.  No unit
//! conversion is performed — values must already be in the listed units before
//! writing the file.
//!
//! | Column | Arrow type | Unit | Description |
//! |--------|------------|------|-------------|
//! | `id` | `UInt64` | — | Unique observation identifier |
//! | `ra` | `Float64` | **rad** | Right ascension |
//! | `ra_err` | `Float64` | **rad** | 1-σ right ascension uncertainty |
//! | `dec` | `Float64` | **rad** | Declination |
//! | `dec_err` | `Float64` | **rad** | 1-σ declination uncertainty |
//! | `magnitude` | `Float64` | AB mag | Apparent magnitude |
//! | `mag_err` | `Float64` | AB mag | 1-σ magnitude uncertainty |
//! | `filter` | `Utf8` / `Utf8View` or `UInt8` / `UInt16` / `UInt32` | — | Filter label or integer code |
//! | `mjd_tt` | `Float64` | MJD (TT) | Epoch in Modified Julian Date, Terrestrial Time |
//!
//! ## Optional observer columns
//!
//! | Column | Arrow type | Unit | Description |
//! |--------|------------|------|-------------|
//! | `obs_lon` | `Float64` | rad | Geodetic longitude east of Greenwich |
//! | `obs_lat` | `Float64` | rad | Geodetic latitude |
//! | `obs_alt` | `Float64` | m | Altitude above the reference ellipsoid |
//! | `obs_ra_acc` | `Float64` | rad | 1-σ RA measurement accuracy |
//! | `obs_dec_acc` | `Float64` | rad | 1-σ Dec measurement accuracy |
//! | `mpc_code_obs` | `Utf8` / `Utf8View` | — | Three-byte ASCII MPC observatory code |
//!
//! ## Optional index columns
//!
//! | Column | Arrow type | Description |
//! |--------|------------|-------------|
//! | `night_id` | `UInt32` | Night identifier — enables the night index map |
//! | `traj_id` | `UInt32` or `Utf8` / `Utf8View` | Trajectory identifier — enables the trajectory index map |
//!
//! # Observer resolution rules
//!
//! The following precedence applies **per row**:
//!
//! 1. `mpc_code_obs` is non-null → [`ObserverId::MpcCode`](crate::observer::dataset::ObserverId)
//!    is used.  MPC takes priority over any geodetic columns present in the same
//!    row.
//! 2. `obs_lon`, `obs_lat`, and `obs_alt` are all non-null → a custom geodetic
//!    [`Observer`] is constructed.  `obs_ra_acc` and `obs_dec_acc` must also be
//!    non-null; if either is null, [`LoadObsError::Arrow`] is returned.
//! 3. All observer columns are null or absent → the observation carries
//!    `observer: None`.
//! 4. The geodetic triplet is only partially non-null (exactly one or two of
//!    `obs_lon` / `obs_lat` / `obs_alt` are set) → [`LoadObsError::Arrow`].
//!
//! # `LoadObsArgs` configuration
//!
//! | Field | Type | Default | Effect |
//! |-------|------|---------|--------|
//! | `error_model` | `Option<ObsErrorModel>` | `None` | Astrometric error model for MPC accuracy look-up |
//! | `contiguous_choice` | `Option<ContiguousChoice>` | `Some(ContiguousNight)` | Column to sort by for the contiguous-block index optimisation |
//!
//! # Contiguous-block optimisation
//!
//! When [`LoadObsArgs::contiguous_choice`] is set, the DataFusion query is
//! ordered by the chosen column (`night_id` or `traj_id`) before collection.
//! All rows that share the same group key then form an unbroken run in the
//! output `observations` vector, and the corresponding index entry is stored
//! as a compact `ObsMapIndex::Contiguous` half-open range `[start, end)`
//! instead of a heap-allocated `ObsMapIndex::Split` index list.  For large
//! datasets this can cut the memory footprint of the index substantially.
//!
//! Only one column can be contiguous at a time; the other index (if present)
//! falls back to the `Split` representation.
//!
//! # Example
//!
//! ```rust,ignore
//! use photom::io::datafusion::{input_uri::InputUri, loader::{load_obs_sync, LoadObsArgs}};
//!
//! let uri = InputUri("file:///data/observations.parquet".to_owned());
//! let dataset = load_obs_sync(&uri, LoadObsArgs::default())?;
//! println!("loaded {} observations", dataset.observation_count());
//! ```

use std::sync::Arc;

use ahash::AHashMap;
use arrow_array::{
    Array, RecordBatch, StringArray, StringViewArray, UInt32Array,
    cast::AsArray,
    types::{Float64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type},
};
use datafusion::{
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    error::DataFusionError,
    prelude::*,
};
use object_store::{Error as ObjStoreError, ObjectStore};
use tokio::runtime::Runtime;
use url::Url;

use crate::{
    NightId, TrajId,
    coordinates::equatorial::EquCoord,
    io::datafusion::{
        input_uri::InputUri,
        storage::{UriStoreError, resolve_input_uri},
    },
    observation_dataset::{
        ObsDataset,
        index::{NightIndexMap, ObsMapIndex, TrajIndexMap},
        observation::Observation,
    },
    observer::error_model::ObsErrorModel,
    observer::{Observer, dataset::ObserverId, mpc::MpcCode},
    photometry::{Filter, Photometry},
};

// ── error type ────────────────────────────────────────────────────────────────

/// Errors that can occur when loading observations from a Parquet URI.
///
/// This enum is returned by both [`load_obs_sync`] and
/// [`load_obs_from_parquet_uri`].  Each variant corresponds to a distinct
/// failure stage in the loading pipeline: URI resolution, file existence
/// checks, DataFusion query execution, and Arrow column validation.  Inspect
/// the inner message or source error to obtain details suitable for logging or
/// user-facing diagnostics.
#[derive(Debug)]
pub enum LoadObsError {
    /// The resource pointed to by the URI does not exist in the backing store.
    ///
    /// Returned during the existence pre-check for `file://` and `hdfs://`
    /// URIs when the object store reports [`object_store::Error::NotFound`].
    /// The inner `String` is the original URI string that was not found.
    NotFound(String),
    /// URI parsing or object-store resolution failed.
    ///
    /// Returned when the URI cannot be parsed as a valid URL, when the URI
    /// scheme is not supported by the storage resolver, or when the object
    /// store's `head()` call fails for a reason other than file absence.
    /// The inner `String` contains a human-readable explanation.
    Resolve(String),
    /// A DataFusion error occurred during Parquet scan or plan execution.
    ///
    /// Wraps a [`DataFusionError`] that arose while registering the object
    /// store, building the listing table, sorting the logical plan, or
    /// collecting the result batches.  The underlying source is accessible via
    /// [`std::error::Error::source`].
    DataFusion(DataFusionError),
    /// An Arrow column was missing, had an unexpected type, or contained null
    /// values in a non-nullable position.
    ///
    /// Returned during the per-batch validation pass for any of the following
    /// conditions: a mandatory column is absent from the schema; a column's
    /// Arrow data type does not match the expected type; a required column
    /// contains a null value at the indicated global row; the geodetic triplet
    /// (`obs_lon` / `obs_lat` / `obs_alt`) is only partially set; or an MPC
    /// observatory code is not exactly three ASCII bytes.  The inner `String`
    /// includes the column name and the zero-based global row offset.
    Arrow(String),
}

impl std::fmt::Display for LoadObsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadObsError::NotFound(s) => write!(f, "resource not found: {s}"),
            LoadObsError::Resolve(s) => write!(f, "URI resolution error: {s}"),
            LoadObsError::DataFusion(e) => write!(f, "DataFusion error: {e}"),
            LoadObsError::Arrow(s) => write!(f, "Arrow conversion error: {s}"),
        }
    }
}

impl std::error::Error for LoadObsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadObsError::DataFusion(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DataFusionError> for LoadObsError {
    fn from(e: DataFusionError) -> Self {
        LoadObsError::DataFusion(e)
    }
}

impl From<UriStoreError> for LoadObsError {
    fn from(e: UriStoreError) -> Self {
        LoadObsError::Resolve(e.to_string())
    }
}

// ── load arguments ────────────────────────────────────────────────────────────

/// Selects which grouping column drives the contiguous-sort optimisation.
///
/// When set, the DataFusion logical plan is extended with an `ORDER BY` clause
/// on the chosen column (nulls last) before the batches are collected.  This
/// guarantees that all rows sharing the same group key form a contiguous block
/// in the assembled `observations` vector, enabling each index entry to be
/// stored as a compact `ObsMapIndex::Contiguous` half-open range instead of
/// a heap-allocated `ObsMapIndex::Split` index list.
///
/// **Memory trade-off** — the sort adds a DataFusion merge step but can
/// significantly reduce the memory footprint of the resulting index for large
/// datasets with many groups.
///
/// **Choosing between variants:**
/// - Prefer [`ContiguousNight`](ContiguousChoice::ContiguousNight) when
///   downstream queries iterate observation data night by night (the common
///   case for survey pipelines).
/// - Prefer [`ContiguousTraj`](ContiguousChoice::ContiguousTraj) when the
///   primary access pattern is trajectory-based (e.g. orbit-fitting loops).
///
/// Only one column can be contiguous at a time; the other index (if present)
/// retains the `Split` representation.
pub enum ContiguousChoice {
    /// Sort by `night_id` so that each night's observations form a contiguous block.
    ContiguousNight,
    /// Sort by `traj_id` so that each trajectory's observations form a contiguous block.
    ContiguousTraj,
}

/// Configuration for [`load_obs_sync`] / [`load_obs_from_parquet_uri`].
///
/// Controls two orthogonal aspects of the loading pipeline: the astrometric
/// error model used to attach per-site measurement accuracies to MPC-coded
/// observers, and the contiguous-sort optimisation that controls the in-memory
/// layout of the night and trajectory index maps.
///
/// Use [`Default::default`] to obtain sensible defaults (no error model,
/// contiguous sort by `night_id`).
///
/// | Field | Type | Default | Effect |
/// |-------|------|---------|--------|
/// | `error_model` | `Option<ObsErrorModel>` | `None` | Astrometric error model for MPC accuracy look-up |
/// | `contiguous_choice` | `Option<ContiguousChoice>` | `Some(ContiguousNight)` | Column to sort by for the contiguous-block index optimisation |
///
/// # Example
///
/// ```rust,ignore
/// use photom::io::datafusion::loader::{ContiguousChoice, LoadObsArgs};
/// use photom::observer::error_model::ObsErrorModel;
///
/// let args = LoadObsArgs {
///     error_model: Some(ObsErrorModel::default()),
///     contiguous_choice: Some(ContiguousChoice::ContiguousTraj),
/// };
/// ```
pub struct LoadObsArgs {
    /// Astrometric error model used for MPC observatory accuracy look-up.
    ///
    /// When set, the model is stored inside the resulting [`ObsDataset`] and
    /// consulted whenever [`ObsDataset::get_observer`] resolves an
    /// [`ObserverId::MpcCode`](crate::observer::dataset::ObserverId) for the
    /// first time.  `None` disables accuracy look-up; each MPC-coded observer
    /// will carry no accuracy until a model is attached later via
    /// [`ObsDataset::set_error_model`].
    pub error_model: Option<ObsErrorModel>,
    /// Which grouping column (if any) to sort by for the contiguous-block
    /// optimisation.
    ///
    /// Defaults to [`ContiguousChoice::ContiguousNight`], which sorts the
    /// DataFusion result by `night_id` before collection.  Set to `None` to
    /// disable sorting entirely; all index entries will then use the
    /// heap-allocated `ObsMapIndex::Split` representation regardless of the
    /// actual row order in the file.
    pub contiguous_choice: Option<ContiguousChoice>,
}

impl Default for LoadObsArgs {
    fn default() -> Self {
        Self {
            error_model: None,
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        }
    }
}

// ── public entry-points ────────────────────────────────────────────────────────

/// Load an [`ObsDataset`] from a Parquet URI synchronously.
///
/// Builds a single-threaded Tokio runtime internally and blocks on
/// [`load_obs_from_parquet_uri`].  This is the sync entry-point for callers
/// that cannot use `async` (e.g. CLI tools, Python bindings, or tests that
/// run outside a Tokio context).
///
/// # Arguments
///
/// - `input` — the Parquet URI to load from.  Any scheme supported by the
///   storage resolver is accepted (see [module-level docs](self) for the full
///   list).
/// - `args` — loading configuration; use [`LoadObsArgs::default`] for
///   sensible defaults.
///
/// # Returns
///
/// A fully populated [`ObsDataset`] on success.
///
/// # Errors
///
/// Propagates all errors from [`load_obs_from_parquet_uri`].  See that
/// function's documentation for the full list of failure conditions.
///
/// # Panics
///
/// Panics if the internal Tokio runtime cannot be created (e.g. when called
/// from inside an existing Tokio runtime context without nesting support).
///
/// # Example
///
/// ```rust,ignore
/// use photom::io::datafusion::{input_uri::InputUri, loader::{load_obs_sync, LoadObsArgs}};
///
/// let uri = InputUri("file:///data/survey.parquet".to_owned());
/// let dataset = load_obs_sync(&uri, LoadObsArgs::default())?;
/// println!("{} observations loaded", dataset.observation_count());
/// ```
pub fn load_obs_sync(input: &InputUri, args: LoadObsArgs) -> Result<ObsDataset, LoadObsError> {
    let rt = Runtime::new().expect("failed to build tokio runtime");
    rt.block_on(load_obs_from_parquet_uri(input, args))
}

/// Load an [`ObsDataset`] from a Parquet URI asynchronously.
///
/// The pipeline proceeds in the following steps:
///
/// 1. Parse and validate the URI string.
/// 2. Resolve the URI to an object-store backend and an object path.
/// 3. For `file://` and `hdfs://` URIs, perform an existence pre-check via
///    `ObjectStore::head()`.
/// 4. Register the object store in a fresh DataFusion [`SessionContext`].
/// 5. Open the Parquet dataset through a [`ListingTable`] with automatic
///    schema inference.
/// 6. If [`LoadObsArgs::contiguous_choice`] is set and the corresponding
///    column exists in the schema, extend the logical plan with an `ORDER BY`
///    clause (nulls last).
/// 7. Execute the plan and collect all Arrow [`RecordBatch`]es.
/// 8. Pass the batches to `build_obs_dataset_from_batches` for validation
///    and assembly into an [`ObsDataset`].
///
/// # Arguments
///
/// - `input` — the Parquet URI to load from.  Any scheme supported by the
///   storage resolver is accepted.
/// - `args` — loading configuration; use [`LoadObsArgs::default`] for
///   sensible defaults.
///
/// # Returns
///
/// A fully populated [`ObsDataset`] on success.
///
/// # Errors
///
/// - [`LoadObsError::Resolve`] — the URI cannot be parsed as a valid URL, the
///   scheme is unsupported, or the object store's `head()` call fails for a
///   reason other than file absence.
/// - [`LoadObsError::NotFound`] — the Parquet file does not exist (only
///   checked for `file://` and `hdfs://` URIs).
/// - [`LoadObsError::DataFusion`] — a DataFusion error occurs during schema
///   inference, plan construction, or batch collection.
/// - [`LoadObsError::Arrow`] — a required column is absent, has an
///   incompatible Arrow type, or contains null values in a non-nullable
///   position.
///
/// # Example
///
/// ```rust,ignore
/// use photom::io::datafusion::{input_uri::InputUri, loader::{load_obs_from_parquet_uri, LoadObsArgs}};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let uri = InputUri("file:///data/survey.parquet".to_owned());
/// let dataset = load_obs_from_parquet_uri(&uri, LoadObsArgs::default()).await?;
/// println!("{} observations loaded", dataset.observation_count());
/// # Ok(())
/// # }
/// ```
pub async fn load_obs_from_parquet_uri(
    input: &InputUri,
    args: LoadObsArgs,
) -> Result<ObsDataset, LoadObsError> {
    let url = input
        .parse()
        .map_err(|e| LoadObsError::Resolve(format!("invalid URI: {e}")))?;

    // 1) Resolve URI → object-store backend + path.
    let resolved = resolve_input_uri(input)?;

    // 2) Existence check for file:// and hdfs://.
    let scheme = url.scheme();
    if scheme == "file" || scheme == "hdfs" {
        match resolved.store.head(&resolved.path).await {
            Ok(_) => {}
            Err(ObjStoreError::NotFound { .. }) => {
                return Err(LoadObsError::NotFound(input.0.clone()));
            }
            Err(e) => {
                return Err(LoadObsError::Resolve(format!(
                    "store head() error for '{}': {e:?}",
                    input.0
                )));
            }
        }
    }

    // 3) Build a DataFusion context with the store registered.
    let ctx = build_session_context_with_store(&url, resolved.store)?;

    // 4) Open and project the Parquet dataset via ListingTable.
    let listing_url = ListingTableUrl::parse(input.0.as_str())
        .map_err(|e| LoadObsError::Resolve(e.to_string()))?;
    let listing_opts =
        ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
    let listing_cfg = ListingTableConfig::new(listing_url)
        .with_listing_options(listing_opts)
        .infer_schema(&ctx.state())
        .await
        .map_err(LoadObsError::DataFusion)?;
    let table = ListingTable::try_new(listing_cfg).map_err(LoadObsError::DataFusion)?;
    let df: DataFrame = ctx.read_table(Arc::new(table))?;

    // 5) Optionally sort by the contiguous-choice column (nulls last).
    let sort_col: Option<&str> = match &args.contiguous_choice {
        Some(ContiguousChoice::ContiguousNight) => Some("night_id"),
        Some(ContiguousChoice::ContiguousTraj) => Some("traj_id"),
        None => None,
    };
    let df = if let Some(col_name) = sort_col {
        // Only sort when the column actually exists in the schema.
        if df.schema().field_with_name(None, col_name).is_ok() {
            df.sort(vec![col(col_name).sort(true, true)])?
        } else {
            df
        }
    } else {
        df
    };

    // 6) Execute and collect batches.
    let batches: Vec<RecordBatch> = df.collect().await?;

    // 7) Convert Arrow RecordBatches → ObsDataset.
    build_obs_dataset_from_batches(&batches, args)
}

// ── DataFusion session context ────────────────────────────────────────────────

fn build_session_context_with_store(
    url: &Url,
    store: Arc<dyn ObjectStore>,
) -> Result<SessionContext, LoadObsError> {
    let ctx = SessionContext::new();
    ctx.runtime_env().register_object_store(url, store);
    Ok(ctx)
}

// ── batch → ObsDataset conversion ────────────────────────────────────────────

/// Tracks one "contiguous group" column during the single-pass row loop.
///
/// ## Invariant
///
/// The [`RecordBatch`] stream **must** have been pre-sorted by the group key
/// so that all rows for the same key form a contiguous block.  If the data is
/// not sorted, key transitions will be detected prematurely and the resulting
/// index will be incorrect (each run between transitions will be stored as a
/// separate [`ObsMapIndex::Contiguous`] entry).
///
/// ## Fields
///
/// - `current` — the group key and the start row index of the currently open
///   block, or `None` if no block is open yet.
/// - `make_entry` — a function pointer that converts a `(start, end)` pair
///   into the index entry type `I`.  Typically this constructs an
///   [`ObsMapIndex::Contiguous`] value.
///
/// ## Usage
///
/// Create one tracker per grouping column with [`ContiguousGroupTracker::new`],
/// call [`on_row`](ContiguousGroupTracker::on_row) for every row in order, and
/// then call [`finalize`](ContiguousGroupTracker::finalize) after iteration to
/// flush the last open block.
struct ContiguousGroupTracker<K, I> {
    current: Option<(K, usize)>,
    make_entry: fn(usize, usize) -> I,
}

impl<K: Clone + Eq, I> ContiguousGroupTracker<K, I> {
    /// Create a new tracker with no open block.
    ///
    /// # Arguments
    ///
    /// - `make_entry` — a function that converts a half-open `[start, end)`
    ///   row range into an index entry of type `I`.
    fn new(make_entry: fn(usize, usize) -> I) -> Self {
        Self {
            current: None,
            make_entry,
        }
    }

    /// Process one row.
    ///
    /// Compares `key` against the currently open group key and, if a
    /// transition is detected, finalises the previous group's index entry and
    /// opens a new block starting at `row_idx`.
    ///
    /// # Arguments
    ///
    /// - `row_idx` — the zero-based position of this row in the assembled
    ///   `observations` vector (i.e. `observations.len()` before this row is
    ///   pushed).
    /// - `key` — the group key value for this row, or `None` if the cell is
    ///   null.  A null key closes the current group (if any) without opening a
    ///   new one.
    ///
    /// # Returns
    ///
    /// - `Some((key, entry))` — the just-finalised group key and its index
    ///   entry, ready to be inserted into the index map.
    /// - `None` — the current group continues; nothing needs to be inserted
    ///   yet.
    fn on_row(&mut self, row_idx: usize, key: Option<K>) -> Option<(K, I)> {
        match key {
            Some(k) => match &self.current {
                Some((ck, _)) if *ck == k => None,
                Some((prev_key, start)) => {
                    let entry = (self.make_entry)(*start, row_idx);
                    let finished = (prev_key.clone(), entry);
                    self.current = Some((k, row_idx));
                    Some(finished)
                }
                None => {
                    self.current = Some((k, row_idx));
                    None
                }
            },
            None => self.current.take().map(|(key, start)| {
                let entry = (self.make_entry)(start, row_idx);
                (key, entry)
            }),
        }
    }

    /// Finalise the last open group after iteration completes.
    ///
    /// Must be called once after all rows have been processed through
    /// [`on_row`](Self::on_row).  The total number of observations `n` is used
    /// as the exclusive end of the last group's range.
    ///
    /// # Arguments
    ///
    /// - `n` — the total number of observations assembled so far (i.e.
    ///   `observations.len()` after the last row has been pushed).
    ///
    /// # Returns
    ///
    /// `Some((key, entry))` if a group was still open, `None` if the tracker
    /// was already empty.
    fn finalize(mut self, n: usize) -> Option<(K, I)> {
        self.current.take().map(|(key, start)| {
            let entry = (self.make_entry)(start, n);
            (key, entry)
        })
    }
}

/// Convert Arrow [`RecordBatch`]es into an [`ObsDataset`].
///
/// Iterates over every batch in `batches`, delegates per-batch row assembly to
/// [`process_batch`], and then finalises the last open contiguous group for
/// each tracker.  All validation (null checks, dtype checks, MPC code format,
/// partial geodetic triplet) is performed inside [`process_batch`].
///
/// # Arguments
///
/// - `batches` — a slice of Arrow [`RecordBatch`]es produced by executing the
///   DataFusion plan.  All batches must share the same schema.
/// - `args` — loading configuration that controls the error model and contiguous-sort strategy.
///
/// # Returns
///
/// A fully populated [`ObsDataset`] on success.
///
/// # Errors
///
/// Returns [`LoadObsError::Arrow`] for any schema validation or null-check
/// failure encountered while processing the batches.
fn build_obs_dataset_from_batches(
    batches: &[RecordBatch],
    args: LoadObsArgs,
) -> Result<ObsDataset, LoadObsError> {
    let mut observations: Vec<Observation> = Vec::new();
    let mut custom_observers: Vec<Observer> = Vec::with_capacity(16);
    let mut observer_lookup: AHashMap<Observer, usize> = AHashMap::with_capacity(16);

    // Optional index maps — created on first non-null key encountered.
    let mut night_map: Option<NightIndexMap> = None;
    let mut traj_map: Option<TrajIndexMap> = None;
    let mut schema_checked = false;

    let night_is_contiguous = matches!(
        args.contiguous_choice,
        Some(ContiguousChoice::ContiguousNight)
    );
    let traj_is_contiguous = matches!(
        args.contiguous_choice,
        Some(ContiguousChoice::ContiguousTraj)
    );

    let mut night_tracker: ContiguousGroupTracker<NightId, ObsMapIndex> =
        ContiguousGroupTracker::new(|start, end| ObsMapIndex::Contiguous { start, end });
    let mut traj_tracker: ContiguousGroupTracker<TrajId, ObsMapIndex> =
        ContiguousGroupTracker::new(|start, end| ObsMapIndex::Contiguous { start, end });

    let mut global_row = 0usize;

    for batch in batches {
        // Detect index column presence from the schema (once).
        if !schema_checked {
            schema_checked = true;
            if batch.schema().index_of("night_id").is_ok() {
                night_map = Some(NightIndexMap::new());
            }
            if batch.schema().index_of("traj_id").is_ok() {
                traj_map = Some(TrajIndexMap::new());
            }
        }

        process_batch(
            batch,
            &mut observations,
            &mut custom_observers,
            &mut observer_lookup,
            &mut night_map,
            &mut traj_map,
            &mut night_tracker,
            &mut traj_tracker,
            night_is_contiguous,
            traj_is_contiguous,
            &mut global_row,
        )?;
    }

    let total = observations.len();

    // Finalise the last open contiguous group for each tracker.
    if night_is_contiguous
        && let (Some(map), Some((key, entry))) = (&mut night_map, night_tracker.finalize(total))
    {
        map.insert(key, entry);
    }
    if traj_is_contiguous
        && let (Some(map), Some((key, entry))) = (&mut traj_map, traj_tracker.finalize(total))
    {
        map.insert(key, entry);
    }

    Ok(ObsDataset::new(
        observations,
        custom_observers,
        args.error_model,
        night_map,
        traj_map,
    ))
}

/// Process a single Arrow [`RecordBatch`] and append the resulting
/// [`Observation`]s to the shared accumulator vectors.
///
/// Extracts all mandatory and optional columns from the batch, then iterates
/// over each row to:
///
/// 1. Assert that every required column cell is non-null.
/// 2. Resolve the observer for the row via [`resolve_and_intern_observer`].
/// 3. Update the night and trajectory index trackers.
/// 4. Construct an [`Observation`] and push it to `observations`.
///
/// The `global_row` counter is advanced by one for every row processed and is
/// used in error messages to identify the failing row across batch boundaries.
///
/// # Arguments
///
/// - `batch` — the Arrow [`RecordBatch`] to process.
/// - `observations` — the accumulator vector to which new [`Observation`]s are
///   appended.
/// - `custom_observers` — the list of interned custom (geodetic) observers;
///   grown lazily as new unique observers are encountered.
/// - `observer_lookup` — reverse map from [`Observer`] to its index in
///   `custom_observers`, used for deduplication.
/// - `night_map` — optional night index map, present when the schema contains
///   a `night_id` column.
/// - `traj_map` — optional trajectory index map, present when the schema
///   contains a `traj_id` column.
/// - `night_tracker` — contiguous-group tracker for the `night_id` column.
/// - `traj_tracker` — contiguous-group tracker for the `traj_id` column.
/// - `night_is_contiguous` — whether the contiguous-sort optimisation is
///   active for `night_id`.
/// - `traj_is_contiguous` — whether the contiguous-sort optimisation is
///   active for `traj_id`.
/// - `global_row` — mutable counter tracking the current row's zero-based
///   position across all batches; incremented once per row.
///
/// # Errors
///
/// Returns [`LoadObsError::Arrow`] if any mandatory column is missing, has an
/// incompatible type, or contains a null value, or if [`resolve_and_intern_observer`]
/// returns an error.
#[allow(clippy::too_many_arguments)]
fn process_batch(
    batch: &RecordBatch,
    observations: &mut Vec<Observation>,
    custom_observers: &mut Vec<Observer>,
    observer_lookup: &mut AHashMap<Observer, usize>,
    night_map: &mut Option<NightIndexMap>,
    traj_map: &mut Option<TrajIndexMap>,
    night_tracker: &mut ContiguousGroupTracker<NightId, ObsMapIndex>,
    traj_tracker: &mut ContiguousGroupTracker<TrajId, ObsMapIndex>,
    night_is_contiguous: bool,
    traj_is_contiguous: bool,
    global_row: &mut usize,
) -> Result<(), LoadObsError> {
    let n = batch.num_rows();

    // ── mandatory base columns ────────────────────────────────────────────────
    let ids = col_u64(batch, "id")?;
    let ra = col_f64(batch, "ra")?;
    let ra_err = col_f64(batch, "ra_err")?;
    let dec = col_f64(batch, "dec")?;
    let dec_err = col_f64(batch, "dec_err")?;
    let magnitude = col_f64(batch, "magnitude")?;
    let mag_err = col_f64(batch, "mag_err")?;
    let mjd_tt = col_f64(batch, "mjd_tt")?;
    let filter_col = col_filter(batch, "filter")?;

    // ── optional observer columns ─────────────────────────────────────────────
    let obs_lon = opt_col_f64(batch, "obs_lon");
    let obs_lat = opt_col_f64(batch, "obs_lat");
    let obs_alt = opt_col_f64(batch, "obs_alt");
    let obs_ra_acc = opt_col_f64(batch, "obs_ra_acc");
    let obs_dec_acc = opt_col_f64(batch, "obs_dec_acc");
    let mpc_code_col = opt_col_string(batch, "mpc_code_obs");

    // ── optional index columns ─────────────────────────────────────────────────
    let night_id_col = opt_col_u32(batch, "night_id");
    let traj_id_col = TrajIdCol::from_batch(batch, "traj_id");

    // ── per-row assembly ──────────────────────────────────────────────────────
    for i in 0..n {
        let row_idx = observations.len();

        // Mandatory nullability checks.
        macro_rules! require_non_null {
            ($arr:expr, $name:literal) => {
                if $arr.is_null(i) {
                    return Err(LoadObsError::Arrow(format!(
                        "null in required column '{}' at global row {}",
                        $name, *global_row
                    )));
                }
            };
        }
        require_non_null!(ids, "id");
        require_non_null!(ra, "ra");
        require_non_null!(ra_err, "ra_err");
        require_non_null!(dec, "dec");
        require_non_null!(dec_err, "dec_err");
        require_non_null!(magnitude, "magnitude");
        require_non_null!(mag_err, "mag_err");
        require_non_null!(mjd_tt, "mjd_tt");
        filter_col.require_non_null(i, *global_row)?;

        // Observer resolution.
        let observer_id = resolve_and_intern_observer(
            i,
            *global_row,
            obs_lon.as_ref(),
            obs_lat.as_ref(),
            obs_alt.as_ref(),
            obs_ra_acc.as_ref(),
            obs_dec_acc.as_ref(),
            mpc_code_col.as_ref(),
            custom_observers,
            observer_lookup,
        )?;

        // Night index.
        if let Some(map) = night_map.as_mut() {
            let night_id = night_id_col
                .as_ref()
                .and_then(|c| c.value_at(i))
                .map(NightId);
            if night_is_contiguous {
                if let Some((key, entry)) = night_tracker.on_row(row_idx, night_id) {
                    map.insert(key, entry);
                }
            } else if let Some(nid) = night_id {
                map.entry(nid)
                    .or_insert_with(|| ObsMapIndex::Split(Vec::new()))
                    .push_split(row_idx);
            }
        }

        // Trajectory index.
        if let Some(map) = traj_map.as_mut() {
            let traj_id = traj_id_col.value_at(i);
            if traj_is_contiguous {
                if let Some((key, entry)) = traj_tracker.on_row(row_idx, traj_id) {
                    map.insert(key, entry);
                }
            } else if let Some(tid) = traj_id {
                map.entry(tid)
                    .or_insert_with(|| ObsMapIndex::Split(Vec::new()))
                    .push_split(row_idx);
            }
        }

        // Build Observation.
        observations.push(Observation {
            index: Some(row_idx),
            id: ids.value(i),
            equ_coord: EquCoord::new(ra.value(i), ra_err.value(i), dec.value(i), dec_err.value(i)),
            photometry: Photometry {
                magnitude: magnitude.value(i),
                error: mag_err.value(i),
                filter: filter_col.value_at(i),
            },
            mjd_tt: mjd_tt.value(i),
            observer: observer_id,
        });

        *global_row += 1;
    }
    Ok(())
}

// ── observer resolution ────────────────────────────────────────────────────────

/// Resolve the observer for a single row and intern it in the custom-observer
/// registry.
///
/// Applies the following precedence rules (see also the
/// [module-level observer resolution section](self)):
///
/// 1. **MPC code** — if `mpc_code` is present and the cell at row `i` is
///    non-null, parse the string as a three-byte ASCII [`MpcCode`] and return
///    [`ObserverId::MpcCode`](crate::observer::dataset::ObserverId).  No
///    geodetic columns are consulted.
/// 2. **Geodetic triplet** — if all three of `obs_lon`, `obs_lat`, and
///    `obs_alt` are non-null, construct a custom [`Observer`].  If the
///    resulting observer has already been seen, return its existing index;
///    otherwise push it onto `custom_observers` and return the new index.
/// 3. **No observer** — if all observer columns are absent or null, return
///    `None`.
/// 4. **Partial triplet** — if exactly one or two of `obs_lon`, `obs_lat`,
///    `obs_alt` are non-null, return [`LoadObsError::Arrow`].
///
/// # Arguments
///
/// - `i` — zero-based row index within the current batch.
/// - `global_row` — zero-based row index across all batches, used in error
///   messages.
/// - `obs_lon` / `obs_lat` / `obs_alt` — optional wrappers around the
///   geodetic coordinate columns; `None` when the column is absent from the
///   schema.
/// - `obs_ra_acc` / `obs_dec_acc` — optional wrappers around the astrometric
///   accuracy columns; must be non-null when a full geodetic triplet is
///   present.
/// - `mpc_code` — optional wrapper around the MPC observatory code column.
/// - `custom_observers` — the list of interned custom observers; grown lazily.
/// - `observer_lookup` — reverse map from [`Observer`] to its position in
///   `custom_observers`.
///
/// # Returns
///
/// `Some(ObserverId)` when an observer is resolved, or `None` when all
/// observer columns are absent or null.
///
/// # Errors
///
/// - [`LoadObsError::Arrow`] — the MPC code string is not exactly three ASCII
///   bytes, a required accuracy column is null while the geodetic triplet is
///   fully set, the [`Observer`] constructor rejects the coordinate values, or
///   the geodetic triplet is only partially non-null.
#[allow(clippy::too_many_arguments)]
fn resolve_and_intern_observer(
    i: usize,
    global_row: usize,
    obs_lon: Option<&OptF64Col<'_>>,
    obs_lat: Option<&OptF64Col<'_>>,
    obs_alt: Option<&OptF64Col<'_>>,
    obs_ra_acc: Option<&OptF64Col<'_>>,
    obs_dec_acc: Option<&OptF64Col<'_>>,
    mpc_code: Option<&StringCol<'_>>,
    custom_observers: &mut Vec<Observer>,
    observer_lookup: &mut AHashMap<Observer, usize>,
) -> Result<Option<ObserverId>, LoadObsError> {
    // MPC code takes precedence.
    if let Some(col) = mpc_code
        && let Some(code_str) = col.value_at(i)
    {
        let bytes: MpcCode = code_str.as_bytes().try_into().map_err(|_| {
            LoadObsError::Arrow(format!(
                "invalid MPC code '{code_str}' at global row {global_row}: must be exactly 3 ASCII bytes"
            ))
        })?;
        return Ok(Some(ObserverId::MpcCode(bytes)));
    }

    // Geodetic triplet.
    let lon = obs_lon.and_then(|c| c.value_at(i));
    let lat = obs_lat.and_then(|c| c.value_at(i));
    let alt = obs_alt.and_then(|c| c.value_at(i));

    match (lon, lat, alt) {
        (Some(lon), Some(lat), Some(alt)) => {
            let ra_acc = obs_ra_acc
                .and_then(|c| c.value_at(i))
                .ok_or_else(|| {
                    LoadObsError::Arrow(format!(
                        "obs_ra_acc is null at global row {global_row} but geodetic triplet is fully set"
                    ))
                })?;
            let dec_acc = obs_dec_acc
                .and_then(|c| c.value_at(i))
                .ok_or_else(|| {
                    LoadObsError::Arrow(format!(
                        "obs_dec_acc is null at global row {global_row} but geodetic triplet is fully set"
                    ))
                })?;

            let observer = Observer::new(lon, lat, alt, None, Some(ra_acc), Some(dec_acc))
                .map_err(|e| {
                    LoadObsError::Arrow(format!("invalid observer at global row {global_row}: {e}"))
                })?;

            let idx = match observer_lookup.get(&observer) {
                Some(&idx) => idx,
                None => {
                    let idx = custom_observers.len();
                    custom_observers.push(observer.clone());
                    observer_lookup.insert(observer, idx);
                    idx
                }
            };
            Ok(Some(ObserverId::IntId(idx)))
        }
        (None, None, None) => Ok(None),
        _ => Err(LoadObsError::Arrow(format!(
            "partial geodetic triplet (obs_lon/obs_lat/obs_alt) at global row {global_row}: \
             all three must be either all non-null or all null"
        ))),
    }
}

// ── column helpers ─────────────────────────────────────────────────────────────

/// Look up the index of a named column in a [`RecordBatch`] schema.
///
/// # Arguments
///
/// - `batch` — the [`RecordBatch`] whose schema is searched.
/// - `name` — the column name to look up.
///
/// # Errors
///
/// Returns [`LoadObsError::Arrow`] if `name` is not present in the schema.
fn col_index(batch: &RecordBatch, name: &str) -> Result<usize, LoadObsError> {
    batch
        .schema()
        .index_of(name)
        .map_err(|_| LoadObsError::Arrow(format!("missing required column '{name}'")))
}

/// Extract a mandatory `UInt64` column from a [`RecordBatch`].
///
/// # Arguments
///
/// - `batch` — the source [`RecordBatch`].
/// - `name` — the column name.
///
/// # Errors
///
/// Returns [`LoadObsError::Arrow`] if the column is absent or is not of type
/// `UInt64`.
fn col_u64<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::PrimitiveArray<UInt64Type>, LoadObsError> {
    let idx = col_index(batch, name)?;
    batch
        .column(idx)
        .as_primitive_opt::<UInt64Type>()
        .ok_or_else(|| LoadObsError::Arrow(format!("column '{name}' is not UInt64")))
}

/// Extract a mandatory `Float64` column from a [`RecordBatch`].
///
/// # Arguments
///
/// - `batch` — the source [`RecordBatch`].
/// - `name` — the column name.
///
/// # Errors
///
/// Returns [`LoadObsError::Arrow`] if the column is absent or is not of type
/// `Float64`.
fn col_f64<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::PrimitiveArray<Float64Type>, LoadObsError> {
    let idx = col_index(batch, name)?;
    batch
        .column(idx)
        .as_primitive_opt::<Float64Type>()
        .ok_or_else(|| LoadObsError::Arrow(format!("column '{name}' is not Float64")))
}

// ── optional column wrappers ──────────────────────────────────────────────────

/// Thin wrapper around an optional `Float64` column.
///
/// Absent when the column is not present in the [`RecordBatch`] schema.
/// When present, [`value_at`](OptF64Col::value_at) returns `None` for null
/// cells and `Some(f64)` for non-null cells.
struct OptF64Col<'a>(&'a arrow_array::PrimitiveArray<Float64Type>);

impl OptF64Col<'_> {
    /// Return the value at row `i`, or `None` if the cell is null.
    fn value_at(&self, i: usize) -> Option<f64> {
        if self.0.is_null(i) {
            None
        } else {
            Some(self.0.value(i))
        }
    }
}

/// Extract an optional `Float64` column from a [`RecordBatch`].
///
/// Returns `None` if the column is absent from the schema or has a type other
/// than `Float64`.
///
/// # Arguments
///
/// - `batch` — the source [`RecordBatch`].
/// - `name` — the column name to look up.
fn opt_col_f64<'a>(batch: &'a RecordBatch, name: &str) -> Option<OptF64Col<'a>> {
    let idx = batch.schema().index_of(name).ok()?;
    let arr = batch.column(idx).as_primitive_opt::<Float64Type>()?;
    Some(OptF64Col(arr))
}

/// Thin wrapper around an optional `UInt32` column.
///
/// Absent when the column is not present in the [`RecordBatch`] schema.
/// When present, [`value_at`](OptU32Col::value_at) returns `None` for null
/// cells and `Some(u32)` for non-null cells.
struct OptU32Col<'a>(&'a arrow_array::PrimitiveArray<UInt32Type>);

impl OptU32Col<'_> {
    /// Return the value at row `i`, or `None` if the cell is null.
    fn value_at(&self, i: usize) -> Option<u32> {
        if self.0.is_null(i) {
            None
        } else {
            Some(self.0.value(i))
        }
    }
}

/// Extract an optional `UInt32` column from a [`RecordBatch`].
///
/// Returns `None` if the column is absent from the schema or has a type other
/// than `UInt32`.
///
/// # Arguments
///
/// - `batch` — the source [`RecordBatch`].
/// - `name` — the column name to look up.
fn opt_col_u32<'a>(batch: &'a RecordBatch, name: &str) -> Option<OptU32Col<'a>> {
    let idx = batch.schema().index_of(name).ok()?;
    let arr = batch.column(idx).as_primitive_opt::<UInt32Type>()?;
    Some(OptU32Col(arr))
}

// ── string column (Utf8 or Utf8View) ─────────────────────────────────────────

/// String column that may be stored as `Utf8` or `Utf8View`.
///
/// Arrow allows string data to be encoded in two different physical layouts;
/// this enum abstracts over both so that callers can use
/// [`value_at`](StringCol::value_at) without inspecting the underlying array
/// type.
enum StringCol<'a> {
    /// Column stored in the classic offset-based `Utf8` (i.e. `LargeUtf8`)
    /// encoding.
    Utf8(&'a StringArray),
    /// Column stored in the newer variable-length view `Utf8View` encoding.
    View(&'a StringViewArray),
}

impl StringCol<'_> {
    /// Return the string value at row `i`, or `None` if the cell is null.
    fn value_at(&self, i: usize) -> Option<&str> {
        match self {
            StringCol::Utf8(a) => {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i))
                }
            }
            StringCol::View(a) => {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i))
                }
            }
        }
    }
}

/// Extract an optional string column (`Utf8` or `Utf8View`) from a
/// [`RecordBatch`].
///
/// Returns `None` if the column is absent from the schema or if its Arrow type
/// is neither `Utf8` nor `Utf8View`.
///
/// # Arguments
///
/// - `batch` — the source [`RecordBatch`].
/// - `name` — the column name to look up.
fn opt_col_string<'a>(batch: &'a RecordBatch, name: &str) -> Option<StringCol<'a>> {
    let idx = batch.schema().index_of(name).ok()?;
    let col = batch.column(idx);
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Some(StringCol::Utf8(arr));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
        return Some(StringCol::View(arr));
    }
    None
}

// ── filter column (Utf8, Utf8View, UInt8, UInt16, or UInt32) ─────────────────

/// Filter column that may be stored as `Utf8`, `Utf8View`, `UInt8`, `UInt16`,
/// or `UInt32`.
///
/// Survey pipelines encode the photometric filter either as a human-readable
/// string (e.g. `"g"`, `"r"`, `"VR"`) or as an integer code.  This enum
/// abstracts over all five supported Arrow physical types so that callers
/// obtain a [`Filter`] value via [`value_at`](FilterCol::value_at) without
/// inspecting the underlying array type.
enum FilterCol<'a> {
    /// String-encoded filter (`Utf8` or `Utf8View`).
    Str(StringCol<'a>),
    /// Integer-encoded filter stored as `UInt8`.
    U8(&'a arrow_array::PrimitiveArray<UInt8Type>),
    /// Integer-encoded filter stored as `UInt16`.
    U16(&'a arrow_array::PrimitiveArray<UInt16Type>),
    /// Integer-encoded filter stored as `UInt32`.
    U32(&'a arrow_array::PrimitiveArray<UInt32Type>),
}

impl FilterCol<'_> {
    /// Return the [`Filter`] value at row `i`.
    ///
    /// For string variants a null cell is treated as an empty string filter
    /// label.  For integer variants the raw value is widened to `u32`.
    fn value_at(&self, i: usize) -> Filter {
        match self {
            FilterCol::Str(sc) => Filter::String(sc.value_at(i).unwrap_or("").to_owned()),
            FilterCol::U8(arr) => Filter::Int(u32::from(arr.value(i))),
            FilterCol::U16(arr) => Filter::Int(u32::from(arr.value(i))),
            FilterCol::U32(arr) => Filter::Int(arr.value(i)),
        }
    }

    /// Assert that the cell at row `i` is non-null.
    ///
    /// # Arguments
    ///
    /// - `i` — zero-based row index within the current batch.
    /// - `global_row` — zero-based row index across all batches, used in the
    ///   error message.
    ///
    /// # Errors
    ///
    /// Returns [`LoadObsError::Arrow`] if the cell is null.
    fn require_non_null(&self, i: usize, global_row: usize) -> Result<(), LoadObsError> {
        let is_null = match self {
            FilterCol::Str(StringCol::Utf8(a)) => a.is_null(i),
            FilterCol::Str(StringCol::View(a)) => a.is_null(i),
            FilterCol::U8(a) => a.is_null(i),
            FilterCol::U16(a) => a.is_null(i),
            FilterCol::U32(a) => a.is_null(i),
        };
        if is_null {
            Err(LoadObsError::Arrow(format!(
                "null in required column 'filter' at global row {global_row}"
            )))
        } else {
            Ok(())
        }
    }
}

/// Extract the mandatory `filter` column from a [`RecordBatch`].
///
/// Tries each supported Arrow type in turn: `Utf8`, `Utf8View`, `UInt8`,
/// `UInt16`, `UInt32`.
///
/// # Arguments
///
/// - `batch` — the source [`RecordBatch`].
/// - `name` — the column name (typically `"filter"`).
///
/// # Errors
///
/// - [`LoadObsError::Arrow`] if the column is absent from the schema.
/// - [`LoadObsError::Arrow`] if the column has a type other than `Utf8`,
///   `Utf8View`, `UInt8`, `UInt16`, or `UInt32`.
fn col_filter<'a>(batch: &'a RecordBatch, name: &str) -> Result<FilterCol<'a>, LoadObsError> {
    let idx = col_index(batch, name)?;
    let col = batch.column(idx);

    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(FilterCol::Str(StringCol::Utf8(arr)));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
        return Ok(FilterCol::Str(StringCol::View(arr)));
    }
    if let Some(arr) = col.as_primitive_opt::<UInt8Type>() {
        return Ok(FilterCol::U8(arr));
    }
    if let Some(arr) = col.as_primitive_opt::<UInt16Type>() {
        return Ok(FilterCol::U16(arr));
    }
    if let Some(arr) = col.as_primitive_opt::<UInt32Type>() {
        return Ok(FilterCol::U32(arr));
    }

    Err(LoadObsError::Arrow(format!(
        "column '{name}' has an unsupported type for filter \
         (expected Utf8, Utf8View, UInt8, UInt16, or UInt32)"
    )))
}

// ── trajectory id column (UInt32 or Utf8/Utf8View) ────────────────────────────

/// Trajectory identifier column that may be stored as `UInt32`, `Utf8`,
/// `Utf8View`, or be entirely absent from the schema.
///
/// When the column is present, [`value_at`](TrajIdCol::value_at) maps each
/// non-null cell to the appropriate [`TrajId`] variant.  Null cells and the
/// `Absent` variant both return `None`.
enum TrajIdCol<'a> {
    /// Integer trajectory identifiers stored as `UInt32`.
    Int(&'a UInt32Array),
    /// String trajectory identifiers stored as `Utf8` or `Utf8View`.
    Str(StringCol<'a>),
    /// The `traj_id` column is not present in the schema.
    Absent,
}

impl TrajIdCol<'_> {
    /// Construct a [`TrajIdCol`] by inspecting the schema and column data of
    /// `batch`.
    ///
    /// Attempts to downcast to `UInt32Array` first, then `StringArray`, then
    /// `StringViewArray`.  Returns [`TrajIdCol::Absent`] if the column is
    /// missing from the schema or has an unrecognised type.
    ///
    /// # Arguments
    ///
    /// - `batch` — the source [`RecordBatch`].
    /// - `name` — the column name to look up (typically `"traj_id"`).
    fn from_batch<'a>(batch: &'a RecordBatch, name: &str) -> TrajIdCol<'a> {
        let Some(idx) = batch.schema().index_of(name).ok() else {
            return TrajIdCol::Absent;
        };
        let col = batch.column(idx);
        if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
            return TrajIdCol::Int(arr);
        }
        if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            return TrajIdCol::Str(StringCol::Utf8(arr));
        }
        if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
            return TrajIdCol::Str(StringCol::View(arr));
        }
        TrajIdCol::Absent
    }

    /// Return the [`TrajId`] at row `i`, or `None` if the cell is null or the
    /// column is absent.
    fn value_at(&self, i: usize) -> Option<TrajId> {
        match self {
            TrajIdCol::Int(arr) => {
                if arr.is_null(i) {
                    None
                } else {
                    Some(TrajId::Int(arr.value(i)))
                }
            }
            TrajIdCol::Str(sc) => sc.value_at(i).map(|s| TrajId::Str(s.to_owned())),
            TrajIdCol::Absent => None,
        }
    }
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod datafusion_loader_tests {
    use super::*;
    use arrow_array::{ArrayRef, Float64Array, StringArray, UInt32Array, UInt64Array};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    // ── schema helpers ────────────────────────────────────────────────────────

    fn base_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("ra", DataType::Float64, false),
            Field::new("ra_err", DataType::Float64, false),
            Field::new("dec", DataType::Float64, false),
            Field::new("dec_err", DataType::Float64, false),
            Field::new("magnitude", DataType::Float64, false),
            Field::new("mag_err", DataType::Float64, false),
            Field::new("filter", DataType::Utf8, false),
            Field::new("mjd_tt", DataType::Float64, false),
        ]))
    }

    fn make_base_batch(n_rows: usize) -> RecordBatch {
        let schema = base_schema();
        let ids: Vec<u64> = (0..n_rows as u64).collect();
        let vals: Vec<f64> = (0..n_rows).map(|i| i as f64).collect();
        let strs: Vec<&str> = (0..n_rows).map(|_| "G").collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(ids)) as ArrayRef,
                Arc::new(Float64Array::from(vals.clone())) as ArrayRef,
                Arc::new(Float64Array::from(
                    vals.iter().map(|_| 0.001).collect::<Vec<f64>>(),
                )) as ArrayRef,
                Arc::new(Float64Array::from(vals.clone())) as ArrayRef,
                Arc::new(Float64Array::from(
                    vals.iter().map(|_| 0.001).collect::<Vec<f64>>(),
                )) as ArrayRef,
                Arc::new(Float64Array::from(
                    vals.iter().map(|_| 15.0).collect::<Vec<f64>>(),
                )) as ArrayRef,
                Arc::new(Float64Array::from(
                    vals.iter().map(|_| 0.05).collect::<Vec<f64>>(),
                )) as ArrayRef,
                Arc::new(StringArray::from(strs)) as ArrayRef,
                Arc::new(Float64Array::from(
                    vals.iter().map(|_| 60000.0).collect::<Vec<f64>>(),
                )) as ArrayRef,
            ],
        )
        .unwrap()
    }

    // ── happy path ────────────────────────────────────────────────────────────

    #[test]
    fn base_columns_only_builds_dataset_with_no_observer() {
        let batch = make_base_batch(3);
        let ds = build_obs_dataset_from_batches(&[batch], LoadObsArgs::default()).unwrap();

        assert_eq!(ds.observation_count(), 3);
        // All observers should be None.
        for obs in ds.iter_observations() {
            assert!(obs.observer.is_none());
        }
    }

    #[test]
    fn mpc_code_obs_column_sets_mpc_observer() {
        let mut schema_fields = base_schema().fields().to_vec();
        schema_fields.push(Arc::new(Field::new("mpc_code_obs", DataType::Utf8, true)));
        let schema = Arc::new(Schema::new(schema_fields));

        let base = make_base_batch(1);
        let mpc: ArrayRef = Arc::new(StringArray::from(vec![Some("I41")]));
        let mut cols = base.columns().to_vec();
        cols.push(mpc);

        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let ds = build_obs_dataset_from_batches(&[batch], LoadObsArgs::default()).unwrap();

        let obs: Vec<_> = ds.iter_observations().collect();
        assert_eq!(obs.len(), 1);
        assert!(
            obs[0].observer == Some(ObserverId::MpcCode(*b"I41")),
            "expected MpcCode(b\"I41\"), got {:?}",
            obs[0].observer
        );
    }

    #[test]
    fn missing_required_column_returns_arrow_error() {
        // Build a batch without 'magnitude'.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("ra", DataType::Float64, false),
            Field::new("ra_err", DataType::Float64, false),
            Field::new("dec", DataType::Float64, false),
            Field::new("dec_err", DataType::Float64, false),
            // magnitude missing
            Field::new("mag_err", DataType::Float64, false),
            Field::new("filter", DataType::Utf8, false),
            Field::new("mjd_tt", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1u64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![1.0f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.001f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![1.0f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.001f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.05f64])) as ArrayRef,
                Arc::new(StringArray::from(vec!["G"])) as ArrayRef,
                Arc::new(Float64Array::from(vec![60000.0f64])) as ArrayRef,
            ],
        )
        .unwrap();

        let err = build_obs_dataset_from_batches(&[batch], LoadObsArgs::default()).unwrap_err();
        match err {
            LoadObsError::Arrow(msg) => {
                assert!(msg.contains("magnitude"), "msg={msg}");
            }
            other => panic!("expected Arrow error, got: {other:?}"),
        }
    }

    #[test]
    fn partial_geodetic_triplet_returns_arrow_error() {
        let mut schema_fields = base_schema().fields().to_vec();
        schema_fields.push(Arc::new(Field::new("obs_lon", DataType::Float64, true)));
        schema_fields.push(Arc::new(Field::new("obs_lat", DataType::Float64, true)));
        // obs_alt absent
        let schema = Arc::new(Schema::new(schema_fields));

        let base = make_base_batch(1);
        let lon: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.1f64)]));
        let lat: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.2f64)]));
        let mut cols = base.columns().to_vec();
        cols.push(lon);
        cols.push(lat);

        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let err = build_obs_dataset_from_batches(&[batch], LoadObsArgs::default()).unwrap_err();
        match err {
            LoadObsError::Arrow(msg) => {
                assert!(msg.contains("partial geodetic"), "msg={msg}");
            }
            other => panic!("expected Arrow error, got: {other:?}"),
        }
    }

    #[test]
    fn night_id_column_builds_night_index() {
        let mut schema_fields = base_schema().fields().to_vec();
        schema_fields.push(Arc::new(Field::new("night_id", DataType::UInt32, true)));
        let schema = Arc::new(Schema::new(schema_fields));

        let base = make_base_batch(3);
        let nights: ArrayRef = Arc::new(UInt32Array::from(vec![1u32, 1u32, 2u32]));
        let mut cols = base.columns().to_vec();
        cols.push(nights);

        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let ds = build_obs_dataset_from_batches(&[batch], LoadObsArgs::default()).unwrap();
        assert_eq!(ds.observation_count(), 3);

        let index = ds.index_ref();
        assert!(index.obs_index_by_night.is_some());
        let night_map = index.obs_index_by_night.as_ref().unwrap();
        assert!(night_map.contains_key(&NightId(1)));
        assert!(night_map.contains_key(&NightId(2)));
    }

    #[test]
    fn filter_column_uint8_is_accepted() {
        use arrow_array::UInt8Array;
        use datafusion::arrow::datatypes::DataType;

        let mut fields = base_schema().fields().to_vec();
        // Replace the Utf8 filter field with UInt8.
        let filter_pos = fields.iter().position(|f| f.name() == "filter").unwrap();
        fields[filter_pos] = Arc::new(Field::new("filter", DataType::UInt8, false));
        let schema = Arc::new(Schema::new(fields));

        let base = make_base_batch(2);
        // Replace the filter column (index 7 in base_batch) with UInt8 values.
        let mut cols = base.columns().to_vec();
        cols[7] = Arc::new(UInt8Array::from(vec![1u8, 2u8])) as ArrayRef;

        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let ds = build_obs_dataset_from_batches(&[batch], LoadObsArgs::default()).unwrap();

        let obs: Vec<_> = ds.iter_observations().collect();
        assert_eq!(obs.len(), 2);
        assert!(matches!(obs[0].photometry.filter, Filter::Int(1)));
        assert!(matches!(obs[1].photometry.filter, Filter::Int(2)));
    }

    #[test]
    fn filter_column_uint16_is_accepted() {
        use arrow_array::UInt16Array;
        use datafusion::arrow::datatypes::DataType;

        let mut fields = base_schema().fields().to_vec();
        let filter_pos = fields.iter().position(|f| f.name() == "filter").unwrap();
        fields[filter_pos] = Arc::new(Field::new("filter", DataType::UInt16, false));
        let schema = Arc::new(Schema::new(fields));

        let base = make_base_batch(2);
        let mut cols = base.columns().to_vec();
        cols[7] = Arc::new(UInt16Array::from(vec![10u16, 20u16])) as ArrayRef;

        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let ds = build_obs_dataset_from_batches(&[batch], LoadObsArgs::default()).unwrap();

        let obs: Vec<_> = ds.iter_observations().collect();
        assert_eq!(obs.len(), 2);
        assert!(matches!(obs[0].photometry.filter, Filter::Int(10)));
        assert!(matches!(obs[1].photometry.filter, Filter::Int(20)));
    }

    #[test]
    fn traj_id_contiguous_choice_builds_contiguous_index() {
        let mut schema_fields = base_schema().fields().to_vec();
        schema_fields.push(Arc::new(Field::new("traj_id", DataType::UInt32, true)));
        let schema = Arc::new(Schema::new(schema_fields));

        let base = make_base_batch(4);
        // traj 7 appears twice, traj 9 appears twice — contiguous when sorted.
        let trajs: ArrayRef = Arc::new(UInt32Array::from(vec![
            Some(7u32),
            Some(7u32),
            Some(9u32),
            Some(9u32),
        ]));
        let mut cols = base.columns().to_vec();
        cols.push(trajs);

        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let ds = build_obs_dataset_from_batches(
            &[batch],
            LoadObsArgs {
                contiguous_choice: Some(ContiguousChoice::ContiguousTraj),
                ..Default::default()
            },
        )
        .unwrap();

        let index = ds.index_ref();
        let traj_map = index.obs_index_by_trajectory.as_ref().unwrap();

        match traj_map.get(&TrajId::Int(7)).unwrap() {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 0);
                assert_eq!(*end, 2);
            }
            ObsMapIndex::Split(_) => panic!("expected Contiguous for traj 7"),
        }
        match traj_map.get(&TrajId::Int(9)).unwrap() {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 2);
                assert_eq!(*end, 4);
            }
            ObsMapIndex::Split(_) => panic!("expected Contiguous for traj 9"),
        }
    }

    #[test]
    fn night_id_contiguous_choice_builds_contiguous_index() {
        // Rows are already ordered night 1, 1, 2 — contiguous by night.
        let mut schema_fields = base_schema().fields().to_vec();
        schema_fields.push(Arc::new(Field::new("night_id", DataType::UInt32, true)));
        let schema = Arc::new(Schema::new(schema_fields));

        let base = make_base_batch(3);
        let nights: ArrayRef = Arc::new(UInt32Array::from(vec![1u32, 1u32, 2u32]));
        let mut cols = base.columns().to_vec();
        cols.push(nights);

        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let ds = build_obs_dataset_from_batches(
            &[batch],
            LoadObsArgs {
                contiguous_choice: Some(ContiguousChoice::ContiguousNight),
                ..Default::default()
            },
        )
        .unwrap();

        let index = ds.index_ref();
        let night_map = index.obs_index_by_night.as_ref().unwrap();

        // Night 1 should be a Contiguous entry covering rows 0..2.
        match night_map.get(&NightId(1)).unwrap() {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 0);
                assert_eq!(*end, 2);
            }
            ObsMapIndex::Split(_) => panic!("expected Contiguous for night 1"),
        }

        // Night 2 should be a Contiguous entry covering rows 2..3.
        match night_map.get(&NightId(2)).unwrap() {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 2);
                assert_eq!(*end, 3);
            }
            ObsMapIndex::Split(_) => panic!("expected Contiguous for night 2"),
        }
    }

    #[test]
    fn night_id_no_contiguous_choice_builds_split_index() {
        let mut schema_fields = base_schema().fields().to_vec();
        schema_fields.push(Arc::new(Field::new("night_id", DataType::UInt32, true)));
        let schema = Arc::new(Schema::new(schema_fields));

        let base = make_base_batch(3);
        let nights: ArrayRef = Arc::new(UInt32Array::from(vec![1u32, 1u32, 2u32]));
        let mut cols = base.columns().to_vec();
        cols.push(nights);

        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let ds = build_obs_dataset_from_batches(
            &[batch],
            LoadObsArgs {
                contiguous_choice: None,
                ..Default::default()
            },
        )
        .unwrap();

        let index = ds.index_ref();
        let night_map = index.obs_index_by_night.as_ref().unwrap();

        // Without sorting, split entries are expected.
        match night_map.get(&NightId(1)).unwrap() {
            ObsMapIndex::Split(v) => assert_eq!(v, &[0, 1]),
            ObsMapIndex::Contiguous { .. } => panic!("expected Split without contiguous choice"),
        }
        match night_map.get(&NightId(2)).unwrap() {
            ObsMapIndex::Split(v) => assert_eq!(v, &[2]),
            ObsMapIndex::Contiguous { .. } => panic!("expected Split without contiguous choice"),
        }
    }

    #[test]
    fn load_obs_sync_reads_local_parquet() {
        use crate::io::datafusion::loader::load_obs_sync;
        use arrow_array::RecordBatch;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::fs::File;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("obs.parquet");

        // Write a minimal Parquet file.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("ra", DataType::Float64, false),
            Field::new("ra_err", DataType::Float64, false),
            Field::new("dec", DataType::Float64, false),
            Field::new("dec_err", DataType::Float64, false),
            Field::new("magnitude", DataType::Float64, false),
            Field::new("mag_err", DataType::Float64, false),
            Field::new("filter", DataType::Utf8, false),
            Field::new("mjd_tt", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![42u64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![1.0f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.001f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.5f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.001f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![15.5f64])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.02f64])) as ArrayRef,
                Arc::new(StringArray::from(vec!["G"])) as ArrayRef,
                Arc::new(Float64Array::from(vec![60000.0f64])) as ArrayRef,
            ],
        )
        .unwrap();

        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let uri = InputUri(format!("file://{}", path.display()));
        let ds =
            load_obs_sync(&uri, LoadObsArgs::default()).expect("should load from local parquet");

        assert_eq!(ds.observation_count(), 1);
        let obs: Vec<_> = ds.iter_observations().collect();
        assert_eq!(*obs[0].id(), 42u64);
    }
}
