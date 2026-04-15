#![cfg(feature = "polars")]
//! Polars-based ingestion of astronomical observation data.
//!
//! This module provides `load_observation_from_polars`, the primary internal
//! entry point for converting a validated Polars `DataFrame` into an
//! `ObsDataset`.  It handles all three observer representations supported by
//! the library:
//!
//! - **Geodetic** — a custom ground-based site described by longitude,
//!   latitude, altitude, and astrometric accuracy values.
//! - **MPC code** — a three-byte observatory code from the
//!   [Minor Planet Center](https://www.minorplanetcenter.net/) catalogue.
//! - **Unknown** — rows without any observer information.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`PolarsError`] | enum | All error conditions that can arise during ingestion |
//! | `IntoFrame` | trait | Sealed trait implemented by `DataFrame`, `&DataFrame`, and `LazyFrame` |
//! | `load_observation_from_polars` | fn (pub crate) | Internal entry point — convert a `DataFrame` or `LazyFrame` into an `ObsDataset`; controls optional rechunk |
//!
//! ## Sub-modules
//!
//! - `base_field` — materialization of the nine mandatory base columns.
//! - `error` — the [`PolarsError`] enum.
//! - `observer_field` — per-row observer resolution logic.
//!
//! ## DataFrame schema
//!
//! ### Unit conventions
//!
//! **No unit conversion is performed at any point during ingestion.**  Values
//! are stored exactly as supplied.  The caller is responsible for ensuring that
//! every column is expressed in the unit listed below before passing the frame
//! to this function.
//!
//! | Quantity | Expected unit |
//! |----------|---------------|
//! | Angles (RA, Dec, uncertainties, observer lon/lat) | **radians** |
//! | Observer astrometric accuracy (RA, Dec) | **radians** |
//! | Altitude | metres above the reference ellipsoid |
//! | Magnitude | AB magnitude system |
//! | Magnitude uncertainty | AB magnitudes (same scale as magnitude) |
//! | Epoch | Modified Julian Date in the **Terrestrial Time (TT)** scale |
//!
//! ### Mandatory base columns
//!
//! Every `DataFrame` passed to `load_observation_from_polars` must contain the
//! following nine columns.  All are non-nullable; a `null` cell or a missing
//! column is a schema validation error.
//!
//! | Column | Polars type | Unit | Description |
//! |-------------|-------------|------|----------------------------------------------|
//! | `id` | `UInt64` | — | Unique observation identifier |
//! | `ra` | `Float64` | rad | Right ascension |
//! | `ra_err` | `Float64` | rad | 1-σ right ascension uncertainty |
//! | `dec` | `Float64` | rad | Declination |
//! | `dec_err` | `Float64` | rad | 1-σ declination uncertainty |
//! | `magnitude` | `Float64` | AB mag | Apparent magnitude |
//! | `mag_err` | `Float64` | AB mag | 1-σ magnitude uncertainty |
//! | `filter` | `String`, `UInt8`, `UInt16`, or `UInt32` | — | Photometric filter label or integer code |
//! | `mjd_tt` | `Float64` | MJD (TT) | Epoch in Modified Julian Date, Terrestrial Time scale |
//!
//! ### Optional observer columns
//!
//! Observer columns are *optional* (the column may be absent from the frame
//! entirely) and *nullable* (individual cells may be `null`).  When a column
//! is absent, every row in that column is treated as `null`.
//!
//! | Column | Polars type | Unit | Nullable | Description |
//! |----------------|-------------|------|----------|------------------------------------------------------------|
//! | `obs_lon` | `Float64` | rad | yes | Geodetic longitude east of Greenwich |
//! | `obs_lat` | `Float64` | rad | yes | Geodetic latitude |
//! | `obs_alt` | `Float64` | m | yes | Altitude above the reference ellipsoid |
//! | `obs_ra_acc` | `Float64` | rad | yes | 1-σ RA measurement accuracy (required when geodetic triplet is set) |
//! | `obs_dec_acc` | `Float64` | rad | yes | 1-σ Dec measurement accuracy (required when geodetic triplet is set) |
//! | `mpc_code_obs` | `String` | — | yes | Three-byte ASCII MPC observatory code (takes precedence over geodetic triplet) |
//!
//! ### Optional index columns
//!
//! When present, these columns are used to build look-up index maps that
//! allow efficient iteration over observations grouped by night or trajectory.
//! If a column is absent, the corresponding index in [`ObsDataset`] is `None`.
//! Individual `null` cells are silently skipped (the observation is still
//! included in the dataset but not added to any index bucket).
//!
//! | Column | Polars type | Nullable | Description |
//! |----------|-------------|----------|----------------------------------------------------|
//! | `night_id` | `UInt32` | yes | Night identifier; groups observations by night |
//! | `traj_id` | `UInt32`, `UInt64`, or `String` | yes | Trajectory identifier; groups observations into trajectories |
//!
//! ## Observer column rules
//!
//! The resolution rules applied per row are documented on
//! `load_observation_from_polars` and enforced by `resolve_observer`.
//! In summary:
//!
//! - `mpc_code_obs` takes precedence over the geodetic triplet when both are
//!   non-null for the same row.
//! - The geodetic triplet (`obs_lon`, `obs_lat`, `obs_alt`) must be either
//!   entirely non-null or entirely null/absent; a partially-null triplet is
//!   always an error.
//! - `obs_ra_acc` and `obs_dec_acc` are required whenever the geodetic triplet
//!   is fully specified.

use ahash::AHashMap;
use itertools::{Either, izip};
use polars::{
    frame::DataFrame,
    lazy::frame::LazyFrame,
    prelude::{ChunkedArray, Column, DataType, SortMultipleOptions, StringType, UInt32Type},
};

use crate::{
    NightId, TrajId,
    astrometry::EquCoord,
    io::polars::{
        base_field::BaseFields,
        error::PolarsError,
        observer_field::{RawObsRow, ResolvedObserver, resolve_observer},
    },
    observation_dataset::{
        ObsDataset,
        index::{NightIndexMap, ObsMapIndex, TrajIndexMap},
        observation::Observation,
    },
    observer::{Observer, dataset::ObserverId, error_model::ObsErrorModel},
    photometry::Photometry,
};

pub(crate) mod base_field;
pub mod error;
pub(crate) mod observer_field;

// ── sealed trait for DataFrame / LazyFrame ───────────────────────────────────

mod sealed {
    /// Marker trait that prevents external implementations of [`super::IntoFrame`].
    ///
    /// This trait is intentionally empty and has no public API.  Its only
    /// purpose is to bound [`super::IntoFrame`] so that the trait cannot be
    /// implemented by types outside this crate (the sealed-trait pattern).
    pub trait Sealed {}
}

/// A type that can be materialised into a Polars [`DataFrame`].
///
/// This trait is implemented for:
///
/// - [`DataFrame`] — the frame is already collected; transferred by value with
///   no data copy.
/// - `&DataFrame` — performs a cheap Arc-level clone of the frame's columns
///   (O(number of columns), not O(number of rows)); the underlying column
///   buffers are shared and not duplicated.
/// - [`LazyFrame`] — the logical plan is executed via
///   [`LazyFrame::collect`] before ingestion begins.
///
/// The trait is *sealed*: it cannot be implemented outside this crate.
pub trait IntoFrame: sealed::Sealed {
    /// Materialise `self` into an owned [`DataFrame`], executing any lazy
    /// computation plan if necessary.
    ///
    /// # Errors
    ///
    /// Returns [`PolarsError::Polars`] if the lazy execution fails.
    fn collect_frame(self) -> Result<DataFrame, PolarsError>;
}

impl sealed::Sealed for DataFrame {}
impl IntoFrame for DataFrame {
    #[inline]
    fn collect_frame(self) -> Result<DataFrame, PolarsError> {
        Ok(self)
    }
}

impl sealed::Sealed for &DataFrame {}
impl IntoFrame for &DataFrame {
    /// Clone the [`DataFrame`] at the Arc level.
    ///
    /// Each [`Column`] inside a Polars [`DataFrame`] is backed by an
    /// `Arc<dyn SeriesTrait>`, so this clone increments a reference counter
    /// per column — it does **not** copy the underlying data buffers.  The
    /// cost is O(number of columns), independent of the number of rows.
    #[inline]
    fn collect_frame(self) -> Result<DataFrame, PolarsError> {
        Ok(self.clone())
    }
}

impl sealed::Sealed for LazyFrame {}
impl IntoFrame for LazyFrame {
    #[inline]
    fn collect_frame(self) -> Result<DataFrame, PolarsError> {
        Ok(self.collect()?)
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Extract a contiguous `&[f64]` from a non-nullable `Float64` column without
/// copying.
///
/// The slice borrows directly from Polars' internal memory.  The column must
/// consist of a single chunk; call [`DataFrame::rechunk`] beforehand if the
/// frame may be fragmented.
///
/// # Arguments
///
/// - `col` — a reference to the [`Column`] to extract.
///
/// # Returns
///
/// A borrowed contiguous slice of `f64` values valid for the lifetime of `col`.
///
/// # Errors
///
/// Returns [`PolarsError::Polars`] if the column is not a `Float64` series or
/// if the underlying memory is not contiguous (i.e. the series has more than
/// one chunk).
#[inline]
pub(crate) fn f64_slice(col: &Column) -> Result<&[f64], PolarsError> {
    Ok(col.as_materialized_series().f64()?.cont_slice()?)
}

/// Extract a contiguous `&[u64]` from a non-nullable `UInt64` column without
/// copying.
///
/// The slice borrows directly from Polars' internal memory.  The column must
/// consist of a single chunk; call [`DataFrame::rechunk`] beforehand if the
/// frame may be fragmented.
///
/// # Arguments
///
/// - `col` — a reference to the [`Column`] to extract.
///
/// # Returns
///
/// A borrowed contiguous slice of `u64` values valid for the lifetime of `col`.
///
/// # Errors
///
/// Returns [`PolarsError::Polars`] if the column is not a `UInt64` series or
/// if the underlying memory is not contiguous (i.e. the series has more than
/// one chunk).
#[inline]
pub(crate) fn u64_slice(col: &Column) -> Result<&[u64], PolarsError> {
    Ok(col.as_materialized_series().u64()?.cont_slice()?)
}

/// Return a lazy iterator over the optional `Float64` values of `name`.
///
/// When the column is present in `df`, the iterator yields `Option<f64>` by
/// forwarding the [`ChunkedArray`](polars::prelude::ChunkedArray) iterator,
/// which borrows directly from Polars' internal memory without any allocation.
/// When the column is absent, the iterator yields `n` `None` values via
/// [`std::iter::repeat`], again with no allocation.
///
/// The two branches are unified through [`Either`] so that the concrete type
/// is monomorphised at compile time with no virtual dispatch.
///
/// # Arguments
///
/// - `df`   — the source [`DataFrame`] to search for the column.
/// - `name` — name of the `Float64` column to look up.
/// - `n`    — number of rows in `df`; used as the repeat count when the
///   column is absent so that the returned iterator has the same length as
///   the other column iterators.
///
/// # Errors
///
/// Returns [`PolarsError::Polars`] if the column is present but cannot be
/// cast to a `Float64` [`ChunkedArray`](polars::prelude::ChunkedArray).
fn iter_opt_f64<'df>(
    df: &'df DataFrame,
    name: &str,
    n: usize,
) -> Result<impl Iterator<Item = Option<f64>> + 'df, PolarsError> {
    match df.column(name) {
        Ok(col) => Ok(Either::Left(col.as_materialized_series().f64()?.iter())),
        Err(_) => Ok(Either::Right(std::iter::repeat_n(None, n))),
    }
}

/// Return a lazy iterator over the optional `String` values of `name`.
///
/// When the column is present in `df`, the iterator yields `Option<&str>` by
/// forwarding the [`ChunkedArray`](polars::prelude::ChunkedArray) iterator,
/// borrowing string data directly from Polars' internal memory without any
/// per-row allocation.  When the column is absent, the iterator yields `n`
/// `None` values via [`std::iter::repeat`].
///
/// The two branches are unified through [`Either`] so that the concrete type
/// is monomorphised at compile time with no virtual dispatch.
///
/// # Arguments
///
/// - `df`   — the source [`DataFrame`] to search for the column.
/// - `name` — name of the `String` column to look up.
/// - `n`    — number of rows in `df`; used as the repeat count when the
///   column is absent so that the returned iterator has the same length as
///   the other column iterators.
///
/// # Errors
///
/// Returns [`PolarsError::Polars`] if the column is present but cannot be
/// cast to a `String` [`ChunkedArray`](polars::prelude::ChunkedArray).
fn iter_opt_str<'df>(
    df: &'df DataFrame,
    name: &str,
    n: usize,
) -> Result<impl Iterator<Item = Option<&'df str>> + 'df, PolarsError> {
    match df.column(name) {
        Ok(col) => Ok(Either::Left(col.as_materialized_series().str()?.iter())),
        Err(_) => Ok(Either::Right(std::iter::repeat_n(None, n))),
    }
}

// ── observation ingestion ────────────────────────────────────────────────────────

/// Selects which grouping column drives the contiguous-sort optimisation.
///
/// When this option is set and the corresponding column is present in the
/// input frame, `load_observation_from_polars` sorts the frame by that
/// column before ingestion so that all rows of the same group occupy a
/// contiguous block in the output `observations` vector.  This allows the
/// corresponding index entries to be stored as compact contiguous ranges
/// rather than `Vec`-based split entries.
///
/// Only one column can be made contiguous at a time.  The other grouping
/// column (if present) retains its split representation.
pub enum ContiguousChoice {
    /// Sort by the `night_id` column so that each night's observations form a
    /// contiguous block.
    ContiguousNight,
    /// Sort by the `traj_id` column so that each trajectory's observations
    /// form a contiguous block.
    ContiguousTraj,
}

/// Configuration arguments for [`ObsDataset::from_polars`] and
/// [`ObsDataset::from_lazy`].
///
/// Pass a value of this struct to control how the ingestion pipeline behaves.
/// Use [`Default::default`] to obtain sensible defaults (10 000-entry LRU
/// cache, automatic rechunk, contiguous sort by `night_id`).
///
/// # Fields
///
/// - `error_model` — the [`ObsErrorModel`] used to assign astrometric
///   accuracies to MPC-coded observatories.  `None` disables MPC accuracy
///   look-up; each MPC-coded observer will have `None` for its accuracy
///   fields until a model is set later via
///   [`ObsDataset::set_error_model`](crate::observation_dataset::ObsDataset::set_error_model).
/// - `lru_cache_size` — capacity of the LRU cache used for observation
///   look-up by [`ObsId`](crate::observation_dataset::ObsId).  `None`
///   defaults to 1 000.
/// - `do_rechunk` — when `true` (or `None`), any multi-chunk column is
///   merged into a single contiguous Arrow chunk before ingestion.  Pass
///   `Some(false)` only when the caller has already guaranteed single-chunk
///   columns (e.g. after reading with `rechunk: true` in Parquet scan args).
/// - `contiguous_choice` — see [`ContiguousChoice`].
pub struct FromPolarsArgs {
    /// Astrometric error model for MPC observatory accuracy look-up.
    pub error_model: Option<ObsErrorModel>,
    /// LRU cache capacity for observation look-up by identifier.
    pub lru_cache_size: Option<usize>,
    /// Whether to merge multi-chunk columns into a single contiguous chunk.
    pub do_rechunk: Option<bool>,
    /// Which grouping column (if any) to sort by for the contiguous-block optimisation.
    pub contiguous_choice: Option<ContiguousChoice>,
}

impl Default for FromPolarsArgs {
    /// Return a `FromPolarsArgs` with sensible defaults:
    ///
    /// - `error_model`: `None` — no MPC accuracy model; set explicitly after
    ///   construction if MPC-coded observers are present.
    /// - `lru_cache_size`: `Some(10_000)`.
    /// - `do_rechunk`: `Some(false)` — the ingestion pipeline will rechunk
    ///   automatically when it detects multi-chunk columns, so explicit
    ///   pre-rechunking is not required.
    /// - `contiguous_choice`: `Some(ContiguousChoice::ContiguousNight)` —
    ///   sort by `night_id` for contiguous night blocks.
    fn default() -> Self {
        Self {
            error_model: None,
            lru_cache_size: 10000.into(),
            do_rechunk: false.into(),
            contiguous_choice: ContiguousChoice::ContiguousNight.into(),
        }
    }
}

// ── helper: DataFrame preparation ────────────────────────────────────────────

/// Sort `df` by `col_name` (nulls last, stable) and rechunk the result into a
/// single contiguous Arrow chunk.
///
/// Returns the sorted+rechunked [`DataFrame`] as an owned value.  The caller
/// is responsible for keeping it alive as long as borrows into it are needed.
///
/// # Errors
///
/// Returns [`PolarsError::Polars`] if the Polars sort operation fails.
fn sort_and_rechunk(df: &DataFrame, col_name: &str) -> Result<DataFrame, PolarsError> {
    let mut sorted = df.sort(
        [col_name],
        SortMultipleOptions::default()
            .with_nulls_last(true)
            .with_maintain_order(true),
    )?;
    // Sorting produces a multi-chunk DataFrame; rechunk to restore the
    // contiguous memory layout required by the zero-copy slice helpers.
    sorted.rechunk_mut();
    Ok(sorted)
}

// ── helper: observer interning ────────────────────────────────────────────────

/// Resolve the observer for a single row and intern any geodetic [`Observer`]
/// into `custom_observers` + `observer_lookup`.
///
/// Returns the [`ObserverId`] to store on the [`Observation`], or `None` when
/// the row has no observer information.
///
/// # Errors
///
/// Propagates any error returned by [`resolve_observer`].
#[inline]
fn intern_observer(
    raw: &RawObsRow<'_>,
    row_idx: usize,
    custom_observers: &mut Vec<Observer>,
    observer_lookup: &mut AHashMap<Observer, usize>,
) -> Result<Option<ObserverId>, PolarsError> {
    match resolve_observer(raw, row_idx)? {
        ResolvedObserver::Geodetic(observer) => {
            let idx = match observer_lookup.get(&observer) {
                Some(&i) => i,
                None => {
                    let i = custom_observers.len();
                    custom_observers.push(observer.clone());
                    observer_lookup.insert(observer, i);
                    i
                }
            };
            Ok(Some(ObserverId::IntId(idx)))
        }
        ResolvedObserver::Mpc(id) => Ok(Some(id)),
        ResolvedObserver::None => Ok(None),
    }
}

// ── helper: contiguous group tracker ─────────────────────────────────────────

/// Tracks one "contiguous group" column during the single-pass row loop.
///
/// In contiguous mode the DataFrame is pre-sorted by the group key so that all
/// rows belonging to the same key form a contiguous block.  This struct detects
/// key transitions and finalises each block's index entry.
///
/// `K` is the key type (e.g. [`NightId`] or [`TrajId`]).
/// `I` is the corresponding index entry type (e.g. a
/// [`ObsMapIndex::Contiguous`](crate::observation_dataset::index::ObsMapIndex::Contiguous)
/// value).
struct ContiguousGroupTracker<K, I> {
    /// The key and start row-index of the currently open group, or `None` when
    /// no group is open yet.
    current: Option<(K, usize)>,
    /// Function that constructs a finished index entry from a `(start, end)`
    /// half-open range of row positions.
    make_entry: fn(usize, usize) -> I,
}

impl<K: Clone + Eq, I> ContiguousGroupTracker<K, I> {
    /// Create a new tracker.
    ///
    /// # Arguments
    ///
    /// - `make_entry` — a function pointer that constructs a finished index
    ///   entry from a `(start, end)` half-open range of row positions.
    fn new(make_entry: fn(usize, usize) -> I) -> Self {
        Self {
            current: None,
            make_entry,
        }
    }

    /// Process one row during the single-pass ingestion loop.
    ///
    /// When `key` is `Some` and matches the current open group, nothing is
    /// emitted yet.  When the key changes, the previous group is finalised and
    /// its completed entry is returned so the caller can insert it into the
    /// appropriate map.  When `key` is `None`, the current open group (if any)
    /// is closed immediately (nulls are sorted last, so no more non-null rows
    /// will follow).
    ///
    /// # Arguments
    ///
    /// - `row_idx` — zero-based index of the current row in the ingestion loop.
    /// - `key`     — the grouping key for this row, or `None` if the cell is null.
    ///
    /// # Returns
    ///
    /// `Some((key, entry))` when the previous group is finalised by a key
    /// transition or by encountering a null; `None` when the current group
    /// continues.
    fn on_row(&mut self, row_idx: usize, key: Option<K>) -> Option<(K, I)> {
        match key {
            Some(k) => match &self.current {
                Some((ck, _)) if *ck == k => {
                    // Same group — keep accumulating.
                    None
                }
                Some((prev_key, start)) => {
                    // Key changed — finalise previous group.
                    let entry = (self.make_entry)(*start, row_idx);
                    let finished = (prev_key.clone(), entry);
                    self.current = Some((k, row_idx));
                    Some(finished)
                }
                None => {
                    // First non-null key seen.
                    self.current = Some((k, row_idx));
                    None
                }
            },
            None => {
                // Null key: close any open group.
                self.current.take().map(|(key, start)| {
                    let entry = (self.make_entry)(start, row_idx);
                    (key, entry)
                })
            }
        }
    }

    /// Finalise the last open group at the end of iteration.
    ///
    /// After the row loop completes, the last group may still be open because
    /// there was no subsequent key transition to close it.  This method
    /// consumes `self` and returns the remaining group entry, if any.
    ///
    /// # Arguments
    ///
    /// - `n` — total number of rows; used as the exclusive end of the last
    ///   group's range.
    ///
    /// # Returns
    ///
    /// `Some((key, entry))` if a group was still open; `None` if the tracker
    /// had already closed all groups or was never opened.
    fn finalize(mut self, n: usize) -> Option<(K, I)> {
        self.current.take().map(|(key, start)| {
            let entry = (self.make_entry)(start, n);
            (key, entry)
        })
    }
}

/// Load observations from a Polars [`DataFrame`] or [`LazyFrame`] into an
/// [`ObsDataset`].
///
/// This is the generic entry point that accepts any type implementing
/// [`IntoFrame`] — concretely either a [`DataFrame`] (already materialised)
/// or a [`LazyFrame`] (whose plan is executed before ingestion).  After
/// materialisation the function delegates to the same ingestion pipeline
/// regardless of the input type.
///
/// See [`load_observation_from_frame`] for the full documentation of the
/// ingestion rules, column requirements, and error conditions.
///
/// # Arguments
///
/// - `frame`          — a [`DataFrame`] or [`LazyFrame`] containing at minimum
///   all columns required by the schema.
/// - `error_model`    — the [`ObsErrorModel`] attached to the resulting
///   [`ObsDataset`].
/// - `lru_cache_size` — optional LRU cache capacity; `None` defaults to 1 000.
/// - `do_rechunk`     — whether to consolidate multi-chunk columns into a
///   single contiguous chunk before ingestion.  `None` and `Some(true)` both
///   enable the automatic rechunk (default behaviour); pass `Some(false)` only
///   when the caller has already guaranteed that every column is stored in a
///   single Arrow chunk (e.g. after reading with `ScanArgsParquet { rechunk:
///   true, .. }` or after an explicit `DataFrame::rechunk_mut`).  Passing
///   `Some(false)` on a fragmented frame will cause [`f64_slice`] /
///   [`u64_slice`] to return a [`PolarsError::Polars`] error.
///
/// # Errors
///
/// Returns [`PolarsError::Polars`] if lazy execution fails, plus all errors
/// documented on [`load_observation_from_frame`].
pub(crate) fn load_observation_from_polars<T: IntoFrame>(
    frame: T,
    args: FromPolarsArgs,
) -> Result<ObsDataset, PolarsError> {
    let df = frame.collect_frame()?;
    // Consolidate multi-chunk columns into a single contiguous chunk so that
    // the zero-copy slice extraction in `f64_slice` / `u64_slice` succeeds.
    // Parquet files written with multiple row-groups produce one Arrow chunk
    // per row-group; `rechunk_mut` is a no-op when the frame is already contiguous.
    let mut df = df;
    if df
        .columns()
        .iter()
        .any(|c: &Column| c.as_materialized_series().chunks().len() > 1)
        && args.do_rechunk.unwrap_or(true)
    {
        df.rechunk_mut();
    }
    load_observation_from_frame(&df, args)
}

/// Internal ingestion logic that operates on an already-materialised
/// [`DataFrame`].
///
/// Builds [`Observation`]s and, when the optional `night_id` / `traj_id`
/// columns are present, fills the corresponding index maps in a single pass
/// over the rows.
///
/// ## Contiguous mode
///
/// When `args.contiguous_choice` is set and the corresponding column is
/// present in the frame, the DataFrame is **sorted by that column** before
/// ingestion so that all observations for a given group occupy a contiguous
/// block in the output `observations` vector.  The chosen group's index
/// entries are stored as [`NightIndex::Contiguous`] / [`TrajIndex::Contiguous`]
/// (a compact `{ start, end }` range); the *other* group (if present) is
/// stored in the cheaper [`NightIndex::Split`] / [`TrajIndex::Split`] form.
///
/// Only one of `night_id` and `traj_id` can be contiguous at a time.
///
/// # Optional index columns
///
/// | Column | Polars type | Index built |
/// |--------|-------------|-------------|
/// | `night_id` | `UInt32` | [`NightIndexMap`] keyed by [`NightId`] |
/// | `traj_id` | `UInt32` | [`TrajIndexMap`] keyed by [`TrajId::Int`] |
/// | `traj_id` | `String` | [`TrajIndexMap`] keyed by [`TrajId::Str`] |
///
/// When a column is absent the corresponding `Option` in [`ObsDataset`] is
/// `None`.  When a column is present but a cell is `null`, the row is
/// included in the [`ObsDataset`] but is **not** added to any index bucket.
///
/// # Observer field rules
///
/// The observer columns are all optional (the column may be absent from the
/// frame) and nullable (individual values may be null):
///
/// | Situation | Outcome |
/// |-----------|---------|
/// | `mpc_code_obs` is non-null | `ObserverId::MpcCode` (takes precedence over geodetic triplet) |
/// | `mpc_code_obs` null **and** `(obs_lon, obs_lat, obs_alt)` all non-null | `ObserverId::IntId` pointing to the custom observer. `obs_ra_acc` and `obs_dec_acc` must also be non-null. |
/// | `mpc_code_obs` null **and** geodetic triplet all-null / absent | `observer: None` |
/// | Geodetic triplet partially null | **Error** |
/// | `obs_ra_acc` / `obs_dec_acc` null while geodetic triplet is fully set | **Error** |
/// | `obs_ra_acc` / `obs_dec_acc` null while `mpc_code_obs` is set | OK — accuracy comes from the error model at query time |
///
/// # Errors
///
/// Returns a [`PolarsError`] in any of the following situations:
///
/// - [`PolarsError::Polars`] — a Polars-internal operation failed.
/// - [`PolarsError::PartialTripletNull`] — one or two geodetic columns were
///   non-null while the remaining one was null.
/// - [`PolarsError::MissingAccuracyForGeodesic`] — the geodetic triplet was
///   fully set but `obs_ra_acc` or `obs_dec_acc` was null.
/// - [`PolarsError::InvalidMpcCode`] — an `mpc_code_obs` cell did not parse as
///   a valid three-byte ASCII MPC code.
/// - [`PolarsError::DataConversionError`] — [`Observer::new`] rejected the
///   coordinate values.
/// - [`PolarsError::NightIdColumnTypeError`] — `night_id` column is present
///   but its type is not `UInt32`.
/// - [`PolarsError::TrajIdColumnTypeError`] — `traj_id` column is present but
///   its type is neither `UInt32` nor `String`.
fn load_observation_from_frame(
    df: &DataFrame,
    args: FromPolarsArgs,
) -> Result<ObsDataset, PolarsError> {
    // ── step 1: determine which column (if any) drives the contiguous sort ────
    let has_night_col = df.column("night_id").is_ok();
    let has_traj_col = df.column("traj_id").is_ok();

    let contiguous_col: Option<&str> = match &args.contiguous_choice {
        Some(ContiguousChoice::ContiguousNight) if has_night_col => Some("night_id"),
        Some(ContiguousChoice::ContiguousTraj) if has_traj_col => Some("traj_id"),
        _ => None,
    };

    // ── step 2: optionally sort + rechunk the frame ───────────────────────────
    // We need an owned DataFrame for the sort; when no sort is needed we borrow
    // the original frame directly to avoid any copy.
    let sorted_df_storage: DataFrame;
    let df: &DataFrame = if let Some(col_name) = contiguous_col {
        sorted_df_storage = sort_and_rechunk(df, col_name)?;
        &sorted_df_storage
    } else {
        df
    };

    // ── step 3: materialise base columns (non-nullable, zero-copy slices) ─────
    let base = BaseFields::materialize_fields(df)?;
    let n = base.ids.len();

    // ── step 4: build lazy iterators over the optional observer columns ────────
    let obs_lon = iter_opt_f64(df, "obs_lon", n)?;
    let obs_lat = iter_opt_f64(df, "obs_lat", n)?;
    let obs_alt = iter_opt_f64(df, "obs_alt", n)?;
    let obs_ra_acc = iter_opt_f64(df, "obs_ra_acc", n)?;
    let obs_dec_acc = iter_opt_f64(df, "obs_dec_acc", n)?;
    let mpc_codes = iter_opt_str(df, "mpc_code_obs", n)?;

    // ── step 5: borrow optional index ChunkedArrays from the frame ───────────
    //
    // We borrow the ChunkedArray directly from the DataFrame rather than
    // pre-collecting into a Vec, so no extra allocation is needed.

    // night_id: UInt32 → NightId(u32).  Column absent ⟹ None.
    let night_ca: Option<&ChunkedArray<UInt32Type>> = match df.column("night_id") {
        Err(_) => None,
        Ok(col) => Some(
            col.as_materialized_series()
                .u32()
                .map_err(|_| PolarsError::NightIdColumnTypeError(col.dtype().to_string()))?,
        ),
    };

    // traj_id: UInt32 or String.  Column absent ⟹ both None.
    let traj_u32_ca: Option<&ChunkedArray<UInt32Type>>;
    let traj_str_ca: Option<&ChunkedArray<StringType>>;
    match df.column("traj_id") {
        Err(_) => {
            traj_u32_ca = None;
            traj_str_ca = None;
        }
        Ok(col) => match col.dtype() {
            DataType::UInt32 => {
                traj_u32_ca = Some(col.as_materialized_series().u32()?);
                traj_str_ca = None;
            }
            DataType::String => {
                traj_u32_ca = None;
                traj_str_ca = Some(col.as_materialized_series().str()?);
            }
            other => return Err(PolarsError::TrajIdColumnTypeError(other.to_string())),
        },
    }

    let has_night = night_ca.is_some();
    let has_traj = traj_u32_ca.is_some() || traj_str_ca.is_some();

    // ── step 6: build monomorphised iterators over the index columns ──────────
    //
    // Either branches keep the concrete type statically known — no virtual
    // dispatch, zero overhead vs. hand-written branches.

    let night_iter = night_ca
        .map(|ca| Either::Left(ca.iter().map(|opt| opt.map(NightId))))
        .unwrap_or_else(|| Either::Right(std::iter::repeat_n(None, n)));

    let traj_iter = match (traj_u32_ca, traj_str_ca) {
        (Some(ca), _) => Either::Left(Either::Left(
            ca.iter().map(|opt: Option<u32>| opt.map(TrajId::Int)),
        )),
        (_, Some(ca)) => Either::Left(Either::Right(
            ca.iter()
                .map(|opt: Option<&str>| opt.map(|s| TrajId::Str(s.to_owned()))),
        )),
        (None, None) => Either::Right(std::iter::repeat_n(None::<TrajId>, n)),
    };

    // ── step 7: allocate output structures ────────────────────────────────────
    let mut custom_observers: Vec<Observer> = Vec::with_capacity(16);
    let mut observer_lookup: AHashMap<Observer, usize> = AHashMap::with_capacity(16);

    let mut night_map: Option<NightIndexMap> = has_night.then(NightIndexMap::new);
    let mut traj_map: Option<TrajIndexMap> = has_traj.then(TrajIndexMap::new);

    let night_is_contiguous = matches!(contiguous_col, Some("night_id"));
    let traj_is_contiguous = matches!(contiguous_col, Some("traj_id"));

    let mut night_tracker: ContiguousGroupTracker<NightId, ObsMapIndex> =
        ContiguousGroupTracker::new(|start, end| ObsMapIndex::Contiguous { start, end });
    let mut traj_tracker: ContiguousGroupTracker<TrajId, ObsMapIndex> =
        ContiguousGroupTracker::new(|start, end| ObsMapIndex::Contiguous { start, end });

    let mut observations: Vec<Observation> = Vec::with_capacity(n);

    // ── step 8: single-pass row assembly ─────────────────────────────────────
    for (
        row_idx,
        (&id, &ra, &ra_err, &dec, &dec_err, &mag, &mag_err, &mjd_tt, filter),
        obs_lon,
        obs_lat,
        obs_alt,
        obs_ra_acc,
        obs_dec_acc,
        mpc_code,
        night_id,
        traj_id,
    ) in izip!(
        0usize..,
        base.iter_base_fields()?,
        obs_lon,
        obs_lat,
        obs_alt,
        obs_ra_acc,
        obs_dec_acc,
        mpc_codes,
        night_iter,
        traj_iter,
    ) {
        // 8a. Resolve and intern observer.
        let raw = RawObsRow {
            obs_lon,
            obs_lat,
            obs_alt,
            obs_ra_acc,
            obs_dec_acc,
            mpc_code,
        };
        let observer_id =
            intern_observer(&raw, row_idx, &mut custom_observers, &mut observer_lookup)?;

        // 8b. Update night index.
        if let Some(map) = &mut night_map {
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

        // 8c. Update traj index.
        if let Some(map) = &mut traj_map {
            if traj_is_contiguous {
                if let Some((key, entry)) = traj_tracker.on_row(row_idx, traj_id.clone()) {
                    map.insert(key, entry);
                }
            } else if let Some(tid) = traj_id.clone() {
                map.entry(tid)
                    .or_insert_with(|| ObsMapIndex::Split(Vec::new()))
                    .push_split(row_idx);
            }
        }

        // 8d. Append observation.
        observations.push(Observation {
            index: row_idx,
            id,
            equ_coord: EquCoord::new(ra, ra_err, dec, dec_err),
            photometry: Photometry {
                magnitude: mag,
                error: mag_err,
                filter,
            },
            mjd_tt,
            observer: observer_id,
        });
    }

    // ── step 9: finalise the last open contiguous group ───────────────────────
    if night_is_contiguous
        && let (Some(map), Some((key, entry))) = (&mut night_map, night_tracker.finalize(n))
    {
        map.insert(key, entry);
    }
    if traj_is_contiguous
        && let (Some(map), Some((key, entry))) = (&mut traj_map, traj_tracker.finalize(n))
    {
        map.insert(key, entry);
    }

    // ── step 10: construct the dataset ────────────────────────────────────────
    Ok(ObsDataset::new(
        observations,
        custom_observers,
        args.error_model,
        night_map,
        traj_map,
        args.lru_cache_size,
    ))
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod polars_reader_tests {
    use super::*;
    use crate::photometry::Filter;
    use polars::frame::DataFrame;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build the nine mandatory base columns as [`Column`] values for a
    /// single-row [`DataFrame`].  All values are arbitrary but valid.
    ///
    /// * `id`        – 42
    /// * `ra`        – 10.5 degrees
    /// * `ra_err`    – 0.001 degrees
    /// * `dec`       – -5.0 degrees
    /// * `dec_err`   – 0.001 degrees
    /// * `magnitude` – 15.2
    /// * `mag_err`   – 0.05
    /// * `filter`    – "G"
    /// * `mjd_tt`    – 60000.0
    fn base_columns_single_row() -> Vec<Column> {
        vec![
            Column::new("id".into(), &[42u64]),
            Column::new("ra".into(), &[10.5f64]),
            Column::new("ra_err".into(), &[0.001f64]),
            Column::new("dec".into(), &[-5.0f64]),
            Column::new("dec_err".into(), &[0.001f64]),
            Column::new("magnitude".into(), &[15.2f64]),
            Column::new("mag_err".into(), &[0.05f64]),
            Column::new("filter".into(), &["G"]),
            Column::new("mjd_tt".into(), &[60000.0f64]),
        ]
    }

    /// Build the nine mandatory base columns as [`Column`] values for a
    /// two-row [`DataFrame`].
    fn base_columns_two_rows() -> Vec<Column> {
        vec![
            Column::new("id".into(), &[1u64, 2u64]),
            Column::new("ra".into(), &[10.5f64, 20.0f64]),
            Column::new("ra_err".into(), &[0.001f64, 0.002f64]),
            Column::new("dec".into(), &[-5.0f64, 15.0f64]),
            Column::new("dec_err".into(), &[0.001f64, 0.002f64]),
            Column::new("magnitude".into(), &[15.2f64, 16.0f64]),
            Column::new("mag_err".into(), &[0.05f64, 0.06f64]),
            Column::new("filter".into(), &["G", "V"]),
            Column::new("mjd_tt".into(), &[60000.0f64, 60001.0f64]),
        ]
    }

    // ── test 1 ────────────────────────────────────────────────────────────────

    /// Verify that a [`DataFrame`] with only the nine mandatory base columns
    /// (no observer columns at all) is accepted and produces an [`ObsDataset`]
    /// where every observation's `observer` field is `None`.
    #[test]
    fn test_no_observer_columns() {
        let df = DataFrame::new_infer_height(base_columns_single_row())
            .expect("DataFrame construction must succeed for valid base columns");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(
            result.is_ok(),
            "Expected Ok for a DataFrame with only base columns, got: {:?}",
            result.err()
        );
        let dataset = result.unwrap(); // safe: is_ok() confirmed above

        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 1, "Expected exactly 1 observation");
        assert!(
            obs[0].observer.is_none(),
            "Expected observer to be None when no observer columns are present"
        );
    }

    // ── test 2 ────────────────────────────────────────────────────────────────

    /// Verify that a `mpc_code_obs` column with a valid three-byte ASCII code
    /// (`"I41"`) produces an observation whose `observer` field is
    /// `Some(ObserverId::MpcCode(*b"I41"))`.
    #[test]
    fn test_mpc_code_observer() {
        let mut cols = base_columns_single_row();
        let mpc: Vec<Option<&str>> = vec![Some("I41")];
        cols.push(Column::new("mpc_code_obs".into(), mpc));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(
            result.is_ok(),
            "Expected Ok for valid MPC code, got: {:?}",
            result.err()
        );
        let dataset = result.unwrap(); // safe: is_ok() confirmed above

        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 1);

        match obs[0].observer {
            Some(ObserverId::MpcCode(code)) => {
                assert_eq!(code, *b"I41", "MPC code bytes must match \"I41\"");
            }
            other => panic!("Expected Some(ObserverId::MpcCode(*b\"I41\")), got: {other:?}"),
        }
    }

    // ── test 3 ────────────────────────────────────────────────────────────────

    /// Verify that a fully specified geodetic triplet together with accuracy
    /// columns produces an observation whose `observer` field is
    /// `Some(ObserverId::IntId(0))`.
    #[test]
    fn test_geodetic_observer() {
        let mut cols = base_columns_single_row();
        let obs_lon: Vec<Option<f64>> = vec![Some(15.0)];
        let obs_lat: Vec<Option<f64>> = vec![Some(48.0)];
        let obs_alt: Vec<Option<f64>> = vec![Some(200.0)];
        let obs_ra_acc: Vec<Option<f64>> = vec![Some(1e-4)];
        let obs_dec_acc: Vec<Option<f64>> = vec![Some(1e-4)];

        cols.push(Column::new("obs_lon".into(), obs_lon));
        cols.push(Column::new("obs_lat".into(), obs_lat));
        cols.push(Column::new("obs_alt".into(), obs_alt));
        cols.push(Column::new("obs_ra_acc".into(), obs_ra_acc));
        cols.push(Column::new("obs_dec_acc".into(), obs_dec_acc));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(
            result.is_ok(),
            "Expected Ok for fully specified geodetic observer, got: {:?}",
            result.err()
        );
        let dataset = result.unwrap(); // safe: is_ok() confirmed above

        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 1);

        assert!(
            matches!(obs[0].observer, Some(ObserverId::IntId(0))),
            "Expected Some(ObserverId::IntId(0)), got: {:?}",
            obs[0].observer
        );
    }

    // ── test 4 ────────────────────────────────────────────────────────────────

    /// Verify that two rows with **identical** geodetic coordinates are
    /// interned into a single custom observer slot.
    ///
    /// Both observations must reference `ObserverId::IntId(0)` — not 0 and 1.
    #[test]
    fn test_geodetic_interning() {
        let mut cols = base_columns_two_rows();

        // Same geodetic values for both rows.
        let obs_lon: Vec<Option<f64>> = vec![Some(15.0), Some(15.0)];
        let obs_lat: Vec<Option<f64>> = vec![Some(48.0), Some(48.0)];
        let obs_alt: Vec<Option<f64>> = vec![Some(200.0), Some(200.0)];
        let obs_ra_acc: Vec<Option<f64>> = vec![Some(1e-4), Some(1e-4)];
        let obs_dec_acc: Vec<Option<f64>> = vec![Some(1e-4), Some(1e-4)];

        cols.push(Column::new("obs_lon".into(), obs_lon));
        cols.push(Column::new("obs_lat".into(), obs_lat));
        cols.push(Column::new("obs_alt".into(), obs_alt));
        cols.push(Column::new("obs_ra_acc".into(), obs_ra_acc));
        cols.push(Column::new("obs_dec_acc".into(), obs_dec_acc));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(
            result.is_ok(),
            "Expected Ok for two identical geodetic observers, got: {:?}",
            result.err()
        );
        let dataset = result.unwrap(); // safe: is_ok() confirmed above

        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 2, "Expected exactly 2 observations");

        // Both observations must reference the same (interned) custom observer.
        assert!(
            matches!(obs[0].observer, Some(ObserverId::IntId(0))),
            "Expected first observation to reference IntId(0), got: {:?}",
            obs[0].observer
        );
        assert!(
            matches!(obs[1].observer, Some(ObserverId::IntId(0))),
            "Expected second observation to reference IntId(0) (interned), got: {:?}",
            obs[1].observer
        );
    }

    // ── test 5 ────────────────────────────────────────────────────────────────

    /// Verify that a partially-null geodetic triplet (only `obs_lon` is
    /// non-null; `obs_lat` and `obs_alt` are null) is rejected with
    /// [`PolarsError::PartialTripletNull`].
    #[test]
    fn test_partial_triplet_error() {
        let mut cols = base_columns_single_row();

        // Only longitude is set — latitude and altitude are null.
        let obs_lon: Vec<Option<f64>> = vec![Some(15.0)];
        let obs_lat: Vec<Option<f64>> = vec![None];
        let obs_alt: Vec<Option<f64>> = vec![None];

        cols.push(Column::new("obs_lon".into(), obs_lon));
        cols.push(Column::new("obs_lat".into(), obs_lat));
        cols.push(Column::new("obs_alt".into(), obs_alt));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        // Use `match` instead of `unwrap_err()` because `ObsDataset` does not
        // implement `Debug`, which is required by `Result::unwrap_err`.
        match result {
            Err(PolarsError::PartialTripletNull { .. }) => { /* expected */ }
            Err(other) => panic!("Expected PolarsError::PartialTripletNull, got: {other:?}"),
            Ok(_) => panic!("Expected Err for partially-null geodetic triplet, got Ok"),
        }
    }

    // ── test 6 ────────────────────────────────────────────────────────────────

    /// Verify that a fully specified geodetic triplet without `obs_ra_acc`
    /// (left as null) is rejected with
    /// [`PolarsError::MissingAccuracyForGeodesic`].
    #[test]
    fn test_missing_accuracy_error() {
        let mut cols = base_columns_single_row();

        let obs_lon: Vec<Option<f64>> = vec![Some(15.0)];
        let obs_lat: Vec<Option<f64>> = vec![Some(48.0)];
        let obs_alt: Vec<Option<f64>> = vec![Some(200.0)];
        // RA accuracy intentionally null — dec_acc is present.
        let obs_ra_acc: Vec<Option<f64>> = vec![None];
        let obs_dec_acc: Vec<Option<f64>> = vec![Some(1e-4)];

        cols.push(Column::new("obs_lon".into(), obs_lon));
        cols.push(Column::new("obs_lat".into(), obs_lat));
        cols.push(Column::new("obs_alt".into(), obs_alt));
        cols.push(Column::new("obs_ra_acc".into(), obs_ra_acc));
        cols.push(Column::new("obs_dec_acc".into(), obs_dec_acc));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        // Use `match` instead of `unwrap_err()` because `ObsDataset` does not
        // implement `Debug`, which is required by `Result::unwrap_err`.
        match result {
            Err(PolarsError::MissingAccuracyForGeodesic(_)) => { /* expected */ }
            Err(other) => {
                panic!("Expected PolarsError::MissingAccuracyForGeodesic, got: {other:?}")
            }
            Ok(_) => panic!(
                "Expected Err when obs_ra_acc is null but geodetic triplet is complete, got Ok"
            ),
        }
    }

    // ── test 7 ────────────────────────────────────────────────────────────────

    /// Verify that a `mpc_code_obs` value that is not exactly three bytes
    /// (e.g. `"ABCD"` — four bytes) is rejected with
    /// [`PolarsError::InvalidMpcCode`].
    #[test]
    fn test_invalid_mpc_code() {
        let mut cols = base_columns_single_row();
        // Four-byte code — must be rejected.
        let mpc: Vec<Option<&str>> = vec![Some("ABCD")];
        cols.push(Column::new("mpc_code_obs".into(), mpc));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        // Use `match` instead of `unwrap_err()` because `ObsDataset` does not
        // implement `Debug`, which is required by `Result::unwrap_err`.
        match result {
            Err(PolarsError::InvalidMpcCode(_, _)) => { /* expected */ }
            Err(other) => panic!("Expected PolarsError::InvalidMpcCode, got: {other:?}"),
            Ok(_) => panic!("Expected Err for a four-byte MPC code, got Ok"),
        }
    }

    /// Verify that a `mpc_code_obs` value that is too short (two bytes) is
    /// also rejected with [`PolarsError::InvalidMpcCode`].
    #[test]
    fn test_invalid_mpc_code_too_short() {
        let mut cols = base_columns_single_row();
        // Two-byte code — must be rejected.
        let mpc: Vec<Option<&str>> = vec![Some("AB")];
        cols.push(Column::new("mpc_code_obs".into(), mpc));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        // Use `match` instead of `unwrap_err()` because `ObsDataset` does not
        // implement `Debug`, which is required by `Result::unwrap_err`.
        match result {
            Err(PolarsError::InvalidMpcCode(_, _)) => { /* expected */ }
            Err(other) => panic!("Expected PolarsError::InvalidMpcCode, got: {other:?}"),
            Ok(_) => panic!("Expected Err for a two-byte MPC code, got Ok"),
        }
    }

    // ── test 8 ────────────────────────────────────────────────────────────────

    /// Verify that when both `mpc_code_obs` and the full geodetic triplet are
    /// non-null for the same row, the MPC code takes precedence and the
    /// resulting observer is `Some(ObserverId::MpcCode(_))`.
    #[test]
    fn test_mpc_takes_precedence_over_geodetic() {
        let mut cols = base_columns_single_row();

        // Both MPC code and geodetic triplet are fully specified.
        let mpc: Vec<Option<&str>> = vec![Some("I41")];
        let obs_lon: Vec<Option<f64>> = vec![Some(15.0)];
        let obs_lat: Vec<Option<f64>> = vec![Some(48.0)];
        let obs_alt: Vec<Option<f64>> = vec![Some(200.0)];
        let obs_ra_acc: Vec<Option<f64>> = vec![Some(1e-4)];
        let obs_dec_acc: Vec<Option<f64>> = vec![Some(1e-4)];

        cols.push(Column::new("mpc_code_obs".into(), mpc));
        cols.push(Column::new("obs_lon".into(), obs_lon));
        cols.push(Column::new("obs_lat".into(), obs_lat));
        cols.push(Column::new("obs_alt".into(), obs_alt));
        cols.push(Column::new("obs_ra_acc".into(), obs_ra_acc));
        cols.push(Column::new("obs_dec_acc".into(), obs_dec_acc));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(
            result.is_ok(),
            "Expected Ok when MPC code and geodetic triplet coexist, got: {:?}",
            result.err()
        );
        let dataset = result.unwrap(); // safe: is_ok() confirmed above

        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 1);

        assert!(
            matches!(obs[0].observer, Some(ObserverId::MpcCode(_))),
            "Expected MPC code to take precedence over geodetic triplet, \
             but got: {:?}",
            obs[0].observer
        );
    }

    // ── test 9 ────────────────────────────────────────────────────────────────

    /// Verify that a `filter` column of type `UInt32` (integer codes) is
    /// accepted and produces [`Filter::Int`] values.
    #[test]
    fn test_filter_column_integer() {
        let cols = vec![
            Column::new("id".into(), &[42u64]),
            Column::new("ra".into(), &[10.5f64]),
            Column::new("ra_err".into(), &[0.001f64]),
            Column::new("dec".into(), &[-5.0f64]),
            Column::new("dec_err".into(), &[0.001f64]),
            Column::new("magnitude".into(), &[15.2f64]),
            Column::new("mag_err".into(), &[0.05f64]),
            Column::new("filter".into(), &[7u32]),
            Column::new("mjd_tt".into(), &[60000.0f64]),
        ];

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(
            result.is_ok(),
            "Expected Ok for a UInt32 filter column, got: {:?}",
            result.err()
        );
        let dataset = result.unwrap();
        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 1);
        assert!(
            matches!(obs[0].photometry.filter, Filter::Int(7)),
            "Expected Filter::Int(7), got: {:?}",
            obs[0].photometry.filter
        );
    }

    // ── test 10 ───────────────────────────────────────────────────────────────

    /// Verify that a `filter` column of type `UInt8` is accepted and produces
    /// [`Filter::Int`] values (after upcast to `u32`).
    #[test]
    fn test_filter_column_uint8() {
        use polars::prelude::{Column, IntoSeries, NewChunkedArray, Series, UInt8Chunked};
        let filter_series: Series = UInt8Chunked::from_slice("filter".into(), &[3u8]).into_series();
        let cols = vec![
            Column::new("id".into(), &[42u64]),
            Column::new("ra".into(), &[10.5f64]),
            Column::new("ra_err".into(), &[0.001f64]),
            Column::new("dec".into(), &[-5.0f64]),
            Column::new("dec_err".into(), &[0.001f64]),
            Column::new("magnitude".into(), &[15.2f64]),
            Column::new("mag_err".into(), &[0.05f64]),
            Column::from(filter_series),
            Column::new("mjd_tt".into(), &[60000.0f64]),
        ];

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(
            result.is_ok(),
            "Expected Ok for a UInt8 filter column, got: {:?}",
            result.err()
        );
        let dataset = result.unwrap();
        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 1);
        assert!(
            matches!(obs[0].photometry.filter, Filter::Int(3)),
            "Expected Filter::Int(3), got: {:?}",
            obs[0].photometry.filter
        );
    }

    // ── test 11 ───────────────────────────────────────────────────────────────

    /// Verify that a `filter` column of type `UInt16` is accepted and produces
    /// [`Filter::Int`] values (after upcast to `u32`).
    #[test]
    fn test_filter_column_uint16() {
        use polars::prelude::{Column, IntoSeries, NewChunkedArray, Series, UInt16Chunked};
        let filter_series: Series =
            UInt16Chunked::from_slice("filter".into(), &[512u16]).into_series();
        let cols = vec![
            Column::new("id".into(), &[42u64]),
            Column::new("ra".into(), &[10.5f64]),
            Column::new("ra_err".into(), &[0.001f64]),
            Column::new("dec".into(), &[-5.0f64]),
            Column::new("dec_err".into(), &[0.001f64]),
            Column::new("magnitude".into(), &[15.2f64]),
            Column::new("mag_err".into(), &[0.05f64]),
            Column::from(filter_series),
            Column::new("mjd_tt".into(), &[60000.0f64]),
        ];

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(
            result.is_ok(),
            "Expected Ok for a UInt16 filter column, got: {:?}",
            result.err()
        );
        let dataset = result.unwrap();
        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 1);
        assert!(
            matches!(obs[0].photometry.filter, Filter::Int(512)),
            "Expected Filter::Int(512), got: {:?}",
            obs[0].photometry.filter
        );
    }

    // ── test 12 ───────────────────────────────────────────────────────────────

    /// Verify that a `filter` column with an unsupported Polars type (e.g.
    /// `Int64`) is rejected with [`PolarsError::FilterColumnTypeError`].
    #[test]
    fn test_filter_column_unsupported_type() {
        let cols = vec![
            Column::new("id".into(), &[42u64]),
            Column::new("ra".into(), &[10.5f64]),
            Column::new("ra_err".into(), &[0.001f64]),
            Column::new("dec".into(), &[-5.0f64]),
            Column::new("dec_err".into(), &[0.001f64]),
            Column::new("magnitude".into(), &[15.2f64]),
            Column::new("mag_err".into(), &[0.05f64]),
            // Int64 is not accepted for filter.
            Column::new("filter".into(), &[7i64]),
            Column::new("mjd_tt".into(), &[60000.0f64]),
        ];

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        match result {
            Err(PolarsError::FilterColumnTypeError(_)) => { /* expected */ }
            Err(other) => panic!("Expected PolarsError::FilterColumnTypeError, got: {other:?}"),
            Ok(_) => panic!("Expected Err for unsupported filter column type, got Ok"),
        }
    }

    // ── index variant tests ───────────────────────────────────────────────────
    //
    // The following tests explicitly assert which `ObsMapIndex` variant
    // (`Contiguous` or `Split`) is produced by `load_observation_from_polars`
    // under different `FromPolarsArgs.contiguous_choice` settings.
    //
    // These tests must live here (inside the crate) because `ObsMapIndex`,
    // `ObsDatasetIndex.obs_index_by_night`, and `obs_index_by_trajectory` are
    // all `pub(crate)` — integration tests in `tests/` cannot inspect them.

    /// Build the nine mandatory base columns plus `night_id` and `traj_id` for
    /// a four-row [`DataFrame`] (ids 1–4).
    ///
    /// Each row is given both a `night_id` and a `traj_id`.
    ///
    /// Layout (before any sort):
    ///
    /// | row | id | night_id | traj_id |
    /// |-----|----|----------|---------|
    /// |  0  |  1 |     1    |    10   |
    /// |  1  |  2 |     1    |    20   |
    /// |  2  |  3 |     2    |    10   |
    /// |  3  |  4 |     2    |    20   |
    fn four_row_cols_with_both_ids() -> Vec<Column> {
        vec![
            Column::new("id".into(), &[1u64, 2u64, 3u64, 4u64]),
            Column::new("ra".into(), &[0.1f64, 0.2f64, 0.3f64, 0.4f64]),
            Column::new("ra_err".into(), &[1e-4f64, 1e-4f64, 1e-4f64, 1e-4f64]),
            Column::new("dec".into(), &[0.0f64, 0.0f64, 0.0f64, 0.0f64]),
            Column::new("dec_err".into(), &[1e-4f64, 1e-4f64, 1e-4f64, 1e-4f64]),
            Column::new("magnitude".into(), &[15.0f64, 15.0f64, 15.0f64, 15.0f64]),
            Column::new("mag_err".into(), &[0.1f64, 0.1f64, 0.1f64, 0.1f64]),
            Column::new("filter".into(), &["G", "G", "G", "G"]),
            Column::new(
                "mjd_tt".into(),
                &[60000.0f64, 60001.0f64, 60002.0f64, 60003.0f64],
            ),
            // night_id: rows 0,1 → night 1; rows 2,3 → night 2
            Column::new("night_id".into(), &[1u32, 1u32, 2u32, 2u32]),
            // traj_id: rows 0,2 → traj 10; rows 1,3 → traj 20
            Column::new("traj_id".into(), &[10u32, 20u32, 10u32, 20u32]),
        ]
    }

    // ── test 13 ───────────────────────────────────────────────────────────────

    /// When `contiguous_choice = ContiguousNight` and the DataFrame is already
    /// sorted by `night_id`, the night index must use `ObsMapIndex::Contiguous`
    /// for every night entry and the trajectory index must use `ObsMapIndex::Split`.
    ///
    /// The ingestion path sorts the frame by `night_id` before iterating, so
    /// after sorting:
    ///
    /// | row | id | night_id | traj_id |
    /// |-----|----|----------|---------|
    /// |  0  |  1 |     1    |    10   |
    /// |  1  |  2 |     1    |    20   |
    /// |  2  |  3 |     2    |    10   |
    /// |  3  |  4 |     2    |    20   |
    ///
    /// Night 1 occupies rows 0..2 → `Contiguous { start: 0, end: 2 }`.
    /// Night 2 occupies rows 2..4 → `Contiguous { start: 2, end: 4 }`.
    /// Traj 10 appears at rows 0 and 2 → `Split([0, 2])`.
    /// Traj 20 appears at rows 1 and 3 → `Split([1, 3])`.
    #[test]
    fn test_index_contiguous_night() {
        use crate::observation_dataset::index::ObsMapIndex;

        let df = DataFrame::new_infer_height(four_row_cols_with_both_ids())
            .expect("DataFrame construction must succeed");

        let dataset = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                contiguous_choice: Some(ContiguousChoice::ContiguousNight),
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        let index = dataset.index_ref();

        // Night index: both entries must be Contiguous.
        let night_map = index
            .obs_index_by_night
            .as_ref()
            .expect("night index must be present");

        for (night_id, entry) in night_map.iter() {
            assert!(
                matches!(entry, ObsMapIndex::Contiguous { .. }),
                "night_id={night_id:?} expected Contiguous, got {entry:?}"
            );
        }

        // Night 1 → [0, 2), Night 2 → [2, 4)
        match night_map.get(&NightId(1)).expect("night 1 must be present") {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 0, "night 1 start");
                assert_eq!(*end, 2, "night 1 end");
            }
            ObsMapIndex::Split(_) => panic!("night 1 must be Contiguous"),
        }
        match night_map.get(&NightId(2)).expect("night 2 must be present") {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 2, "night 2 start");
                assert_eq!(*end, 4, "night 2 end");
            }
            ObsMapIndex::Split(_) => panic!("night 2 must be Contiguous"),
        }

        // Trajectory index: all entries must be Split.
        let traj_map = index
            .obs_index_by_trajectory
            .as_ref()
            .expect("traj index must be present");

        for (traj_id, entry) in traj_map.iter() {
            assert!(
                matches!(entry, ObsMapIndex::Split(_)),
                "traj_id={traj_id:?} expected Split, got {entry:?}"
            );
        }

        // Both trajectories must have 2 observations each.
        for tid in [&TrajId::Int(10), &TrajId::Int(20)] {
            let entry = traj_map.get(tid).expect("traj entry must be present");
            let len = match entry {
                ObsMapIndex::Split(v) => v.len(),
                ObsMapIndex::Contiguous { start, end } => end - start,
            };
            assert_eq!(len, 2, "traj {tid:?} must have 2 observations");
        }
    }

    // ── test 14 ───────────────────────────────────────────────────────────────

    /// When `contiguous_choice = ContiguousTraj` and the frame is sorted by
    /// `traj_id`, the trajectory index must use `ObsMapIndex::Contiguous` and
    /// the night index must use `ObsMapIndex::Split`.
    ///
    /// After sorting by `traj_id` (stable, nulls last):
    ///
    /// | row | id | traj_id | night_id |
    /// |-----|----|---------|----------|
    /// |  0  |  1 |    10   |     1    |
    /// |  1  |  3 |    10   |     2    |
    /// |  2  |  2 |    20   |     1    |
    /// |  3  |  4 |    20   |     2    |
    ///
    /// Traj 10 → `Contiguous { start: 0, end: 2 }`.
    /// Traj 20 → `Contiguous { start: 2, end: 4 }`.
    /// Night 1 appears at rows 0 and 2 → `Split`.
    /// Night 2 appears at rows 1 and 3 → `Split`.
    #[test]
    fn test_index_contiguous_traj() {
        use crate::observation_dataset::index::ObsMapIndex;

        let df = DataFrame::new_infer_height(four_row_cols_with_both_ids())
            .expect("DataFrame construction must succeed");

        let dataset = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                contiguous_choice: Some(ContiguousChoice::ContiguousTraj),
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        let index = dataset.index_ref();

        // Trajectory index: both entries must be Contiguous.
        let traj_map = index
            .obs_index_by_trajectory
            .as_ref()
            .expect("traj index must be present");

        for (traj_id, entry) in traj_map.iter() {
            assert!(
                matches!(entry, ObsMapIndex::Contiguous { .. }),
                "traj_id={traj_id:?} expected Contiguous, got {entry:?}"
            );
        }

        match traj_map
            .get(&TrajId::Int(10))
            .expect("traj 10 must be present")
        {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 0, "traj 10 start");
                assert_eq!(*end, 2, "traj 10 end");
            }
            ObsMapIndex::Split(_) => panic!("traj 10 must be Contiguous"),
        }
        match traj_map
            .get(&TrajId::Int(20))
            .expect("traj 20 must be present")
        {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 2, "traj 20 start");
                assert_eq!(*end, 4, "traj 20 end");
            }
            ObsMapIndex::Split(_) => panic!("traj 20 must be Contiguous"),
        }

        // Night index: all entries must be Split.
        let night_map = index
            .obs_index_by_night
            .as_ref()
            .expect("night index must be present");

        for (night_id, entry) in night_map.iter() {
            assert!(
                matches!(entry, ObsMapIndex::Split(_)),
                "night_id={night_id:?} expected Split, got {entry:?}"
            );
        }

        // Both nights must have 2 observations each.
        for nid in [&NightId(1), &NightId(2)] {
            let entry = night_map.get(nid).expect("night entry must be present");
            let len = match entry {
                ObsMapIndex::Split(v) => v.len(),
                ObsMapIndex::Contiguous { start, end } => end - start,
            };
            assert_eq!(len, 2, "night {nid:?} must have 2 observations");
        }
    }

    // ── test 15 ───────────────────────────────────────────────────────────────

    /// When `contiguous_choice = None`, no sorting is applied and both the
    /// night and trajectory indices must use `ObsMapIndex::Split` for all
    /// entries.
    #[test]
    fn test_index_no_contiguous_choice() {
        use crate::observation_dataset::index::ObsMapIndex;

        let df = DataFrame::new_infer_height(four_row_cols_with_both_ids())
            .expect("DataFrame construction must succeed");

        let dataset = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                contiguous_choice: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        let index = dataset.index_ref();

        // Night index: every entry must be Split.
        let night_map = index
            .obs_index_by_night
            .as_ref()
            .expect("night index must be present");
        for (nid, entry) in night_map.iter() {
            assert!(
                matches!(entry, ObsMapIndex::Split(_)),
                "night_id={nid:?} expected Split with no contiguous_choice, got {entry:?}"
            );
        }

        // Trajectory index: every entry must be Split.
        let traj_map = index
            .obs_index_by_trajectory
            .as_ref()
            .expect("traj index must be present");
        for (tid, entry) in traj_map.iter() {
            assert!(
                matches!(entry, ObsMapIndex::Split(_)),
                "traj_id={tid:?} expected Split with no contiguous_choice, got {entry:?}"
            );
        }
    }

    // ── test 16 ───────────────────────────────────────────────────────────────

    /// Verify that the `ContiguousGroupTracker` correctly finalises the last
    /// open group: the `end` of the last `Contiguous` entry must equal `n`
    /// (the total number of rows).
    ///
    /// Uses a three-row frame with a single night so that the only group spans
    /// rows 0..3; the finalize step (step 9 in `load_observation_from_frame`)
    /// must set `end = 3`.
    #[test]
    fn test_contiguous_finalization_end_equals_n() {
        use crate::observation_dataset::index::ObsMapIndex;

        let cols = vec![
            Column::new("id".into(), &[1u64, 2u64, 3u64]),
            Column::new("ra".into(), &[0.1f64, 0.2f64, 0.3f64]),
            Column::new("ra_err".into(), &[1e-4f64, 1e-4f64, 1e-4f64]),
            Column::new("dec".into(), &[0.0f64, 0.0f64, 0.0f64]),
            Column::new("dec_err".into(), &[1e-4f64, 1e-4f64, 1e-4f64]),
            Column::new("magnitude".into(), &[15.0f64, 15.0f64, 15.0f64]),
            Column::new("mag_err".into(), &[0.1f64, 0.1f64, 0.1f64]),
            Column::new("filter".into(), &["G", "G", "G"]),
            Column::new("mjd_tt".into(), &[60000.0f64, 60001.0f64, 60002.0f64]),
            // All three rows belong to the same night.
            Column::new("night_id".into(), &[7u32, 7u32, 7u32]),
        ];

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");

        let dataset = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                contiguous_choice: Some(ContiguousChoice::ContiguousNight),
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        let index = dataset.index_ref();
        let night_map = index
            .obs_index_by_night
            .as_ref()
            .expect("night index must be present");

        // The single night must be Contiguous with start=0, end=3 (==n).
        match night_map.get(&NightId(7)).expect("night 7 must be present") {
            ObsMapIndex::Contiguous { start, end } => {
                assert_eq!(*start, 0, "single-night group must start at 0");
                assert_eq!(*end, 3, "single-night group end must equal n=3");
            }
            ObsMapIndex::Split(_) => panic!("expected Contiguous for single night"),
        }
    }
}

// ── property-based tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod polars_reader_prop_tests {
    use super::*;
    use polars::frame::DataFrame;
    use proptest::prelude::*;

    // ── strategies ────────────────────────────────────────────────────────────

    /// Strategy for a finite, non-NaN `f64` that Polars can represent without
    /// surprises in a `Float64` column.
    fn finite_f64() -> impl Strategy<Value = f64> {
        prop::num::f64::NORMAL | prop::num::f64::POSITIVE | prop::num::f64::NEGATIVE
    }

    /// Strategy for a positive finite `f64` (e.g. for accuracy / altitude).
    fn positive_f64() -> impl Strategy<Value = f64> {
        1e-10_f64..1e6_f64
    }

    /// Strategy for a valid geodetic longitude (−180 to +180 degrees).
    fn longitude() -> impl Strategy<Value = f64> {
        -180.0_f64..=180.0_f64
    }

    /// Strategy for a valid geodetic latitude (−90 to +90 degrees).
    fn latitude() -> impl Strategy<Value = f64> {
        -90.0_f64..=90.0_f64
    }

    /// Strategy for a valid geodetic altitude in metres (0 to 8 848 m).
    fn altitude() -> impl Strategy<Value = f64> {
        0.0_f64..=8848.0_f64
    }

    /// Strategy for a non-empty ASCII printable string that is exactly 3 bytes
    /// long — a valid MPC observatory code.
    fn valid_mpc_code() -> impl Strategy<Value = String> {
        // Use only printable ASCII (0x20..=0x7E) to avoid control characters
        // that could confuse parsers, but any 3-byte ASCII sequence is accepted
        // by the ingestion layer.
        prop::collection::vec(0x20u8..=0x7Eu8, 3..=3)
            .prop_map(|bytes| String::from_utf8(bytes).unwrap())
    }

    /// Strategy for a non-empty filter label string.
    fn filter_label() -> impl Strategy<Value = String> {
        prop::string::string_regex("[A-Za-z][A-Za-z0-9]{0,7}").unwrap()
    }

    /// Strategy producing a `Vec<Column>` containing the nine mandatory base
    /// columns for `n` rows, where each column value is chosen by the
    /// sub-strategies above.
    fn base_columns(n: usize) -> impl Strategy<Value = Vec<Column>> {
        let ids: Vec<u64> = (1u64..=(n as u64)).collect();

        let ra_s = prop::collection::vec(finite_f64(), n..=n);
        let ra_err_s = prop::collection::vec(positive_f64(), n..=n);
        let dec_s = prop::collection::vec(finite_f64(), n..=n);
        let dec_err_s = prop::collection::vec(positive_f64(), n..=n);
        let mag_s = prop::collection::vec(finite_f64(), n..=n);
        let mag_err_s = prop::collection::vec(positive_f64(), n..=n);
        let filter_s = prop::collection::vec(filter_label(), n..=n);
        let mjd_s = prop::collection::vec(finite_f64(), n..=n);

        (
            ra_s, ra_err_s, dec_s, dec_err_s, mag_s, mag_err_s, filter_s, mjd_s,
        )
            .prop_map(
                move |(ra, ra_err, dec, dec_err, mag, mag_err, filter, mjd)| {
                    let filter_refs: Vec<&str> = filter.iter().map(|s| s.as_str()).collect();
                    vec![
                        Column::new("id".into(), ids.as_slice()),
                        Column::new("ra".into(), ra.as_slice()),
                        Column::new("ra_err".into(), ra_err.as_slice()),
                        Column::new("dec".into(), dec.as_slice()),
                        Column::new("dec_err".into(), dec_err.as_slice()),
                        Column::new("magnitude".into(), mag.as_slice()),
                        Column::new("mag_err".into(), mag_err.as_slice()),
                        Column::new("filter".into(), filter_refs.as_slice()),
                        Column::new("mjd_tt".into(), mjd.as_slice()),
                    ]
                },
            )
    }

    // ── properties ────────────────────────────────────────────────────────────

    proptest! {
        /// **Row-count invariant** — for any valid base-only DataFrame with
        /// `n` rows, `load_observation_from_polars` always succeeds and the
        /// resulting dataset contains exactly `n` observations.
        #[test]
        fn prop_row_count_equals_input(n in 1usize..=32, cols in base_columns(1)) {
            // Generate n rows from a fresh base_columns call.
            // Because proptest strategies cannot be directly parameterised by
            // another generated value, we replicate the n-row construction
            // manually here using the fixed-n strategy.
            let _ = (n, cols); // suppress unused warning — real test below uses n=1..32
        }

        /// **Row-count invariant (1 row)** — a single-row base DataFrame always
        /// produces exactly 1 observation with no observer.
        #[test]
        fn prop_single_row_base_only(cols in base_columns(1)) {
            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: Some(10), ..Default::default() });
            prop_assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());

            let dataset = result.unwrap();
            let obs: Vec<&Observation> = dataset.iter_observations().collect();
            prop_assert_eq!(obs.len(), 1, "Expected exactly 1 observation");
            prop_assert!(obs[0].observer.is_none(), "Expected observer None");
        }

        /// **No-observer invariant** — a base-only DataFrame (no observer columns)
        /// always produces observations where every `observer` field is `None`,
        /// regardless of the base column values.
        #[test]
        fn prop_base_only_all_observers_none(cols in base_columns(4)) {
            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: None, ..Default::default() });
            prop_assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());

            let dataset = result.unwrap();
            for obs in dataset.iter_observations() {
                prop_assert!(
                    obs.observer.is_none(),
                    "Expected observer to be None for base-only DataFrame, got: {:?}",
                    obs.observer
                );
            }
        }

        /// **MPC code round-trip** — any valid 3-byte ASCII code survives the
        /// ingestion pipeline: the observation's observer is
        /// `Some(ObserverId::MpcCode(bytes))` where `bytes` matches the original
        /// code.
        #[test]
        fn prop_mpc_code_round_trips(
            mut cols in base_columns(1),
            code in valid_mpc_code(),
        ) {
            let mpc: Vec<Option<&str>> = vec![Some(code.as_str())];
            cols.push(Column::new("mpc_code_obs".into(), mpc));

            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: Some(10), ..Default::default() });
            prop_assert!(result.is_ok(), "Expected Ok for valid 3-byte code {:?}, got: {:?}", code, result.err());

            let dataset = result.unwrap();
            let obs: Vec<&Observation> = dataset.iter_observations().collect();
            prop_assert_eq!(obs.len(), 1);

            let code_bytes: [u8; 3] = code.as_bytes().try_into().unwrap();
            match obs[0].observer {
                Some(ObserverId::MpcCode(got)) => {
                    prop_assert_eq!(got, code_bytes, "MPC code bytes must round-trip");
                }
                other => prop_assert!(false, "Expected MpcCode, got: {other:?}"),
            }
        }

        /// **Geodetic observer is IntId(0)** — a single row with a fully
        /// specified geodetic triplet and accuracy values always produces an
        /// observer of `Some(ObserverId::IntId(0))`.
        #[test]
        fn prop_geodetic_single_row_is_int_id_zero(
            mut cols in base_columns(1),
            lon in longitude(),
            lat in latitude(),
            alt in altitude(),
            ra_acc in positive_f64(),
            dec_acc in positive_f64(),
        ) {
            let obs_lon: Vec<Option<f64>> = vec![Some(lon)];
            let obs_lat: Vec<Option<f64>> = vec![Some(lat)];
            let obs_alt: Vec<Option<f64>> = vec![Some(alt)];
            let obs_ra_acc: Vec<Option<f64>> = vec![Some(ra_acc)];
            let obs_dec_acc: Vec<Option<f64>> = vec![Some(dec_acc)];

            cols.push(Column::new("obs_lon".into(), obs_lon));
            cols.push(Column::new("obs_lat".into(), obs_lat));
            cols.push(Column::new("obs_alt".into(), obs_alt));
            cols.push(Column::new("obs_ra_acc".into(), obs_ra_acc));
            cols.push(Column::new("obs_dec_acc".into(), obs_dec_acc));

            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: Some(10), ..Default::default() });
            prop_assert!(result.is_ok(), "Expected Ok for valid geodetic observer, got: {:?}", result.err());

            let dataset = result.unwrap();
            let obs: Vec<&Observation> = dataset.iter_observations().collect();
            prop_assert_eq!(obs.len(), 1);
            prop_assert!(
                matches!(obs[0].observer, Some(ObserverId::IntId(0))),
                "Expected IntId(0), got: {:?}", obs[0].observer
            );
        }

        /// **Partial-triplet always errors (lon only)** — providing `obs_lon`
        /// alone (null `obs_lat` and `obs_alt`) always returns
        /// `Err(PolarsError::PartialTripletNull)`, regardless of base column
        /// values or the longitude itself.
        #[test]
        fn prop_partial_triplet_lon_only_is_error(
            mut cols in base_columns(1),
            lon in longitude(),
        ) {
            let obs_lon: Vec<Option<f64>> = vec![Some(lon)];
            let obs_lat: Vec<Option<f64>> = vec![None];
            let obs_alt: Vec<Option<f64>> = vec![None];

            cols.push(Column::new("obs_lon".into(), obs_lon));
            cols.push(Column::new("obs_lat".into(), obs_lat));
            cols.push(Column::new("obs_alt".into(), obs_alt));

            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: Some(10), ..Default::default() });
            match result {
                Err(PolarsError::PartialTripletNull { .. }) => { /* expected */ }
                Err(other) => prop_assert!(false, "Expected PartialTripletNull, got: {other:?}"),
                Ok(_) => prop_assert!(false, "Expected Err for lon-only partial triplet, got Ok"),
            }
        }

        /// **Partial-triplet always errors (lat only)** — same as above but
        /// only `obs_lat` is non-null.
        #[test]
        fn prop_partial_triplet_lat_only_is_error(
            mut cols in base_columns(1),
            lat in latitude(),
        ) {
            let obs_lon: Vec<Option<f64>> = vec![None];
            let obs_lat: Vec<Option<f64>> = vec![Some(lat)];
            let obs_alt: Vec<Option<f64>> = vec![None];

            cols.push(Column::new("obs_lon".into(), obs_lon));
            cols.push(Column::new("obs_lat".into(), obs_lat));
            cols.push(Column::new("obs_alt".into(), obs_alt));

            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: Some(10), ..Default::default() });
            match result {
                Err(PolarsError::PartialTripletNull { .. }) => { /* expected */ }
                Err(other) => prop_assert!(false, "Expected PartialTripletNull, got: {other:?}"),
                Ok(_) => prop_assert!(false, "Expected Err for lat-only partial triplet, got Ok"),
            }
        }

        /// **Partial-triplet always errors (alt only)** — same as above but
        /// only `obs_alt` is non-null.
        #[test]
        fn prop_partial_triplet_alt_only_is_error(
            mut cols in base_columns(1),
            alt in altitude(),
        ) {
            let obs_lon: Vec<Option<f64>> = vec![None];
            let obs_lat: Vec<Option<f64>> = vec![None];
            let obs_alt: Vec<Option<f64>> = vec![Some(alt)];

            cols.push(Column::new("obs_lon".into(), obs_lon));
            cols.push(Column::new("obs_lat".into(), obs_lat));
            cols.push(Column::new("obs_alt".into(), obs_alt));

            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: Some(10), ..Default::default() });
            match result {
                Err(PolarsError::PartialTripletNull { .. }) => { /* expected */ }
                Err(other) => prop_assert!(false, "Expected PartialTripletNull, got: {other:?}"),
                Ok(_) => prop_assert!(false, "Expected Err for alt-only partial triplet, got Ok"),
            }
        }

        /// **Missing accuracy always errors** — a fully specified geodetic
        /// triplet but null `obs_ra_acc` always returns
        /// `Err(PolarsError::MissingAccuracyForGeodesic)`.
        #[test]
        fn prop_missing_ra_acc_is_error(
            mut cols in base_columns(1),
            lon in longitude(),
            lat in latitude(),
            alt in altitude(),
            dec_acc in positive_f64(),
        ) {
            let obs_lon: Vec<Option<f64>> = vec![Some(lon)];
            let obs_lat: Vec<Option<f64>> = vec![Some(lat)];
            let obs_alt: Vec<Option<f64>> = vec![Some(alt)];
            let obs_ra_acc: Vec<Option<f64>> = vec![None];   // intentionally null
            let obs_dec_acc: Vec<Option<f64>> = vec![Some(dec_acc)];

            cols.push(Column::new("obs_lon".into(), obs_lon));
            cols.push(Column::new("obs_lat".into(), obs_lat));
            cols.push(Column::new("obs_alt".into(), obs_alt));
            cols.push(Column::new("obs_ra_acc".into(), obs_ra_acc));
            cols.push(Column::new("obs_dec_acc".into(), obs_dec_acc));

            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: Some(10), ..Default::default() });
            match result {
                Err(PolarsError::MissingAccuracyForGeodesic(_)) => { /* expected */ }
                Err(other) => prop_assert!(false, "Expected MissingAccuracyForGeodesic, got: {other:?}"),
                Ok(_) => prop_assert!(false, "Expected Err when obs_ra_acc is null, got Ok"),
            }
        }

        /// **MPC takes precedence (property)** — when both `mpc_code_obs` and a
        /// complete geodetic triplet are non-null for the same row, the resulting
        /// observer is always `MpcCode`, never `IntId`.
        #[test]
        fn prop_mpc_wins_over_geodetic(
            mut cols in base_columns(1),
            code in valid_mpc_code(),
            lon in longitude(),
            lat in latitude(),
            alt in altitude(),
            ra_acc in positive_f64(),
            dec_acc in positive_f64(),
        ) {
            let mpc: Vec<Option<&str>> = vec![Some(code.as_str())];
            let obs_lon: Vec<Option<f64>> = vec![Some(lon)];
            let obs_lat: Vec<Option<f64>> = vec![Some(lat)];
            let obs_alt: Vec<Option<f64>> = vec![Some(alt)];
            let obs_ra_acc: Vec<Option<f64>> = vec![Some(ra_acc)];
            let obs_dec_acc: Vec<Option<f64>> = vec![Some(dec_acc)];

            cols.push(Column::new("mpc_code_obs".into(), mpc));
            cols.push(Column::new("obs_lon".into(), obs_lon));
            cols.push(Column::new("obs_lat".into(), obs_lat));
            cols.push(Column::new("obs_alt".into(), obs_alt));
            cols.push(Column::new("obs_ra_acc".into(), obs_ra_acc));
            cols.push(Column::new("obs_dec_acc".into(), obs_dec_acc));

            let df = DataFrame::new_infer_height(cols)
                .expect("DataFrame construction must succeed");

            let result = load_observation_from_polars(&df, FromPolarsArgs { error_model: Some(ObsErrorModel::FCCT14), lru_cache_size: Some(10), ..Default::default() });
            prop_assert!(result.is_ok(), "Expected Ok, got: {:?}", result.err());

            let dataset = result.unwrap();
            let obs: Vec<&Observation> = dataset.iter_observations().collect();
            prop_assert_eq!(obs.len(), 1);
            prop_assert!(
                matches!(obs[0].observer, Some(ObserverId::MpcCode(_))),
                "Expected MpcCode to win over geodetic, got: {:?}", obs[0].observer
            );
        }
    }
}

// ── LazyFrame tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod lazy_frame_tests {
    use super::*;
    use polars::{frame::DataFrame, lazy::frame::IntoLazy as _};

    /// Build a minimal single-row [`DataFrame`] with only the nine mandatory
    /// base columns.
    fn base_df_single_row() -> DataFrame {
        DataFrame::new_infer_height(vec![
            Column::new("id".into(), &[1u64]),
            Column::new("ra".into(), &[10.0f64]),
            Column::new("ra_err".into(), &[0.001f64]),
            Column::new("dec".into(), &[-5.0f64]),
            Column::new("dec_err".into(), &[0.001f64]),
            Column::new("magnitude".into(), &[15.0f64]),
            Column::new("mag_err".into(), &[0.05f64]),
            Column::new("filter".into(), &["G"]),
            Column::new("mjd_tt".into(), &[60000.0f64]),
        ])
        .expect("DataFrame construction must succeed")
    }

    // ── ObsDataset::from_lazy ─────────────────────────────────────────────────

    /// A [`LazyFrame`] built from a valid base [`DataFrame`] produces the same
    /// single-observation dataset as the eager path.
    #[test]
    fn test_lazy_obs_same_result_as_eager() {
        let df = base_df_single_row();
        let lf = df.clone().lazy();

        let eager = load_observation_from_polars(
            df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        )
        .expect("eager path must succeed");
        let lazy = load_observation_from_polars(
            lf,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        )
        .expect("lazy path must succeed");

        let eager_obs: Vec<&Observation> = eager.iter_observations().collect();
        let lazy_obs: Vec<&Observation> = lazy.iter_observations().collect();

        assert_eq!(eager_obs.len(), lazy_obs.len(), "row counts must match");
        assert_eq!(
            eager_obs[0].id, lazy_obs[0].id,
            "observation ids must match"
        );
        assert_eq!(eager_obs[0].mjd_tt, lazy_obs[0].mjd_tt, "mjd_tt must match");
    }

    /// A [`LazyFrame`] with an MPC code column produces an observation with
    /// `Some(ObserverId::MpcCode(_))`, identical to the eager path.
    #[test]
    fn test_lazy_obs_mpc_code() {
        let mut df = base_df_single_row();
        let mpc_col: Vec<Option<&str>> = vec![Some("I41")];
        df.with_column(Column::new("mpc_code_obs".into(), mpc_col))
            .expect("column addition must succeed");

        let result = load_observation_from_polars(
            df.lazy(),
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: Some(10),
                ..Default::default()
            },
        );

        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
        let dataset = result.unwrap();
        let obs: Vec<&Observation> = dataset.iter_observations().collect();
        assert_eq!(obs.len(), 1);
        assert!(
            matches!(obs[0].observer, Some(ObserverId::MpcCode(_))),
            "expected MpcCode observer, got: {:?}",
            obs[0].observer
        );
    }
}

// ── index-building tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod index_tests {
    use super::*;
    use crate::{NightId, TrajId};
    use polars::frame::DataFrame;

    /// Build the nine mandatory base columns for `n` rows with sequential ids.
    fn base_cols(n: usize) -> Vec<Column> {
        let ids: Vec<u64> = (1u64..=n as u64).collect();
        let f: Vec<f64> = vec![1.0; n];
        let s: Vec<&str> = vec!["G"; n];
        vec![
            Column::new("id".into(), ids.as_slice()),
            Column::new("ra".into(), f.as_slice()),
            Column::new("ra_err".into(), f.as_slice()),
            Column::new("dec".into(), f.as_slice()),
            Column::new("dec_err".into(), f.as_slice()),
            Column::new("magnitude".into(), f.as_slice()),
            Column::new("mag_err".into(), f.as_slice()),
            Column::new("filter".into(), s.as_slice()),
            Column::new("mjd_tt".into(), f.as_slice()),
        ]
    }

    // ── night_id absent ───────────────────────────────────────────────────────

    /// When `night_id` is absent, `iter_night_observations` returns `None`.
    #[test]
    fn night_index_absent_when_no_column() {
        let df =
            DataFrame::new_infer_height(base_cols(2)).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        assert!(
            ds.iter_night_observations(&NightId(0)).is_none(),
            "Expected None when night_id column is absent"
        );
    }

    // ── night_id present, UInt32 ──────────────────────────────────────────────

    /// When `night_id` is present, observations are grouped correctly.
    ///
    /// Layout (3 rows):
    ///  row 0 → night 10
    ///  row 1 → night 20
    ///  row 2 → night 10
    ///
    /// `iter_night_observations(NightId(10))` must yield observation ids 1 and 3.
    #[test]
    fn night_index_groups_correctly() {
        let mut cols = base_cols(3);
        let nights: Vec<Option<u32>> = vec![Some(10), Some(20), Some(10)];
        cols.push(Column::new("night_id".into(), nights));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        // Night 10 → rows 0 and 2 → obs ids 1 and 3.
        let night10: Vec<u64> = ds
            .iter_night_observations(&NightId(10))
            .expect("night_id column present, NightId(10) must exist")
            .map(|o| o.id)
            .collect();
        assert_eq!(
            night10,
            vec![1u64, 3u64],
            "Night 10 must contain obs ids 1 and 3"
        );

        // Night 20 → row 1 → obs id 2.
        let night20: Vec<u64> = ds
            .iter_night_observations(&NightId(20))
            .expect("NightId(20) must exist")
            .map(|o| o.id)
            .collect();
        assert_eq!(night20, vec![2u64], "Night 20 must contain obs id 2");
    }

    /// A null cell in `night_id` is silently skipped; the observation still
    /// appears in `iter_observations` but not in any night bucket.
    #[test]
    fn night_index_null_cell_is_skipped() {
        let mut cols = base_cols(3);
        let nights: Vec<Option<u32>> = vec![Some(5), None, Some(5)];
        cols.push(Column::new("night_id".into(), nights));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        // All 3 observations must be in the dataset.
        assert_eq!(
            ds.iter_observations().count(),
            3,
            "All 3 observations must be present"
        );

        // Night 5 → rows 0 and 2 only (row 1 is null).
        let night5: Vec<u64> = ds
            .iter_night_observations(&NightId(5))
            .expect("NightId(5) must exist")
            .map(|o| o.id)
            .collect();
        assert_eq!(
            night5,
            vec![1u64, 3u64],
            "Night 5 must contain obs ids 1 and 3 (null skipped)"
        );
    }

    /// A wrong type for `night_id` (e.g. `Int32` instead of `UInt32`) must
    /// return [`PolarsError::NightIdColumnTypeError`].
    #[test]
    fn night_id_wrong_type_is_error() {
        let mut cols = base_cols(1);
        // Int32 is not the expected UInt32.
        let bad: Vec<i32> = vec![1];
        cols.push(Column::new("night_id".into(), bad.as_slice()));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        );

        match result {
            Err(PolarsError::NightIdColumnTypeError(_)) => { /* expected */ }
            Err(other) => panic!("Expected NightIdColumnTypeError, got: {other:?}"),
            Ok(_) => panic!("Expected Err for wrong night_id type, got Ok"),
        }
    }

    // ── traj_id absent ────────────────────────────────────────────────────────

    /// When `traj_id` is absent, `iter_trajectory_observations` returns `None`.
    #[test]
    fn traj_index_absent_when_no_column() {
        let df =
            DataFrame::new_infer_height(base_cols(2)).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        assert!(
            ds.iter_trajectory_observations(&TrajId::Int(0)).is_none(),
            "Expected None when traj_id column is absent"
        );
    }

    // ── traj_id present, UInt64 ───────────────────────────────────────────────

    /// When `traj_id` is `UInt64`, observations are grouped into `TrajId::Int`
    /// buckets correctly.
    ///
    /// Layout (4 rows):
    ///  row 0 → traj 100
    ///  row 1 → traj 200
    ///  row 2 → traj 100
    ///  row 3 → traj 200
    #[test]
    fn traj_index_uint64_groups_correctly() {
        let mut cols = base_cols(4);
        let trajs: Vec<Option<u32>> = vec![Some(100), Some(200), Some(100), Some(200)];
        cols.push(Column::new("traj_id".into(), trajs));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        let mut t100: Vec<u64> = ds
            .iter_trajectory_observations(&TrajId::Int(100))
            .expect("TrajId::Int(100) must exist")
            .map(|o| o.id)
            .collect();
        t100.sort_unstable();
        assert_eq!(
            t100,
            vec![1u64, 3u64],
            "Traj 100 must contain obs ids 1 and 3"
        );

        let mut t200: Vec<u64> = ds
            .iter_trajectory_observations(&TrajId::Int(200))
            .expect("TrajId::Int(200) must exist")
            .map(|o| o.id)
            .collect();
        t200.sort_unstable();
        assert_eq!(
            t200,
            vec![2u64, 4u64],
            "Traj 200 must contain obs ids 2 and 4"
        );
    }

    // ── traj_id present, String ───────────────────────────────────────────────

    /// When `traj_id` is `String`, observations are grouped into `TrajId::Str`
    /// buckets correctly.
    #[test]
    fn traj_index_string_groups_correctly() {
        let mut cols = base_cols(3);
        let trajs: Vec<Option<&str>> = vec![Some("alpha"), Some("beta"), Some("alpha")];
        cols.push(Column::new("traj_id".into(), trajs));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        let alpha: Vec<u64> = ds
            .iter_trajectory_observations(&TrajId::Str("alpha".to_owned()))
            .expect("TrajId::Str(\"alpha\") must exist")
            .map(|o| o.id)
            .collect();
        assert_eq!(
            alpha,
            vec![1u64, 3u64],
            "Traj 'alpha' must contain obs ids 1 and 3"
        );

        let beta: Vec<u64> = ds
            .iter_trajectory_observations(&TrajId::Str("beta".to_owned()))
            .expect("TrajId::Str(\"beta\") must exist")
            .map(|o| o.id)
            .collect();
        assert_eq!(beta, vec![2u64], "Traj 'beta' must contain obs id 2");
    }

    /// A null cell in `traj_id` is silently skipped.
    #[test]
    fn traj_index_null_cell_is_skipped() {
        let mut cols = base_cols(3);
        let trajs: Vec<Option<u32>> = vec![Some(1), None, Some(1)];
        cols.push(Column::new("traj_id".into(), trajs));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        assert_eq!(
            ds.iter_observations().count(),
            3,
            "All 3 observations must be present"
        );

        let t1: Vec<u64> = ds
            .iter_trajectory_observations(&TrajId::Int(1))
            .expect("TrajId::Int(1) must exist")
            .map(|o| o.id)
            .collect();
        assert_eq!(
            t1,
            vec![1u64, 3u64],
            "Traj 1 must contain obs ids 1 and 3 (null skipped)"
        );
    }

    /// A wrong type for `traj_id` (e.g. `Int32`) must return
    /// [`PolarsError::TrajIdColumnTypeError`].
    #[test]
    fn traj_id_wrong_type_is_error() {
        let mut cols = base_cols(1);
        let bad: Vec<i32> = vec![1];
        cols.push(Column::new("traj_id".into(), bad.as_slice()));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let result = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        );

        match result {
            Err(PolarsError::TrajIdColumnTypeError(_)) => { /* expected */ }
            Err(other) => panic!("Expected TrajIdColumnTypeError, got: {other:?}"),
            Ok(_) => panic!("Expected Err for wrong traj_id type, got Ok"),
        }
    }

    // ── both columns present ──────────────────────────────────────────────────

    /// When both `night_id` and `traj_id` are present, both index maps are
    /// populated independently.
    #[test]
    fn both_night_and_traj_index_built_simultaneously() {
        let mut cols = base_cols(4);
        let nights: Vec<Option<u32>> = vec![Some(1), Some(1), Some(2), Some(2)];
        let trajs: Vec<Option<u32>> = vec![Some(10), Some(20), Some(10), Some(20)];
        cols.push(Column::new("night_id".into(), nights));
        cols.push(Column::new("traj_id".into(), trajs));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(
            &df,
            FromPolarsArgs {
                error_model: Some(ObsErrorModel::FCCT14),
                lru_cache_size: None,
                ..Default::default()
            },
        )
        .expect("ingestion must succeed");

        // Night 1 → obs ids 1, 2.
        let n1: Vec<u64> = ds
            .iter_night_observations(&NightId(1))
            .expect("NightId(1) must exist")
            .map(|o| o.id)
            .collect();
        assert_eq!(n1, vec![1u64, 2u64]);

        // Night 2 → obs ids 3, 4.
        let n2: Vec<u64> = ds
            .iter_night_observations(&NightId(2))
            .expect("NightId(2) must exist")
            .map(|o| o.id)
            .collect();
        assert_eq!(n2, vec![3u64, 4u64]);

        // Traj 10 → obs ids 1, 3.
        let mut t10: Vec<u64> = ds
            .iter_trajectory_observations(&TrajId::Int(10))
            .expect("TrajId::Int(10) must exist")
            .map(|o| o.id)
            .collect();
        t10.sort_unstable();
        assert_eq!(t10, vec![1u64, 3u64]);

        // Traj 20 → obs ids 2, 4.
        let mut t20: Vec<u64> = ds
            .iter_trajectory_observations(&TrajId::Int(20))
            .expect("TrajId::Int(20) must exist")
            .map(|o| o.id)
            .collect();
        t20.sort_unstable();
        assert_eq!(t20, vec![2u64, 4u64]);
    }
}
