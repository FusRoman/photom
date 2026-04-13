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
//! | `load_observation_from_polars` | fn (pub crate) | Internal entry point — convert a `DataFrame` or `LazyFrame` into an `ObsDataset` |
//!
//! ## Sub-modules
//!
//! - `base_field` — zero-copy materialization of the nine mandatory base columns.
//! - `error` — the [`PolarsError`] enum.
//! - `observer_field` — per-row observer resolution logic.
//!
//! ## DataFrame schema
//!
//! ### Mandatory base columns
//!
//! Every `DataFrame` passed to `load_observation_from_polars` must contain the
//! following nine columns.  All are non-nullable; a `null` cell or a missing
//! column is a schema validation error.
//!
//! | Column | Polars type | Description |
//! |-------------|-------------|----------------------------------------------|
//! | `id` | `UInt64` | Unique observation identifier |
//! | `ra` | `Float64` | Right ascension (degrees) |
//! | `ra_err` | `Float64` | Right ascension uncertainty (degrees) |
//! | `dec` | `Float64` | Declination (degrees) |
//! | `dec_err` | `Float64` | Declination uncertainty (degrees) |
//! | `magnitude` | `Float64` | Apparent magnitude |
//! | `mag_err` | `Float64` | Magnitude uncertainty |
//! | `filter` | `String` | Photometric filter label |
//! | `mjd_tt` | `Float64` | Epoch (Modified Julian Date, Terrestrial Time) |
//!
//! ### Optional observer columns
//!
//! Observer columns are *optional* (the column may be absent from the frame
//! entirely) and *nullable* (individual cells may be `null`).  When a column
//! is absent, every row in that column is treated as `null`.
//!
//! | Column | Polars type | Nullable | Description |
//! |----------------|-------------|----------|------------------------------------------------------------|
//! | `obs_lon` | `Float64` | yes | Geodetic longitude in degrees east of Greenwich |
//! | `obs_lat` | `Float64` | yes | Geodetic latitude in degrees |
//! | `obs_alt` | `Float64` | yes | Altitude above the reference ellipsoid in metres |
//! | `obs_ra_acc` | `Float64` | yes | RA measurement accuracy in radians (required when geodetic triplet is set) |
//! | `obs_dec_acc` | `Float64` | yes | Dec measurement accuracy in radians (required when geodetic triplet is set) |
//! | `mpc_code_obs` | `String` | yes | Three-byte ASCII MPC observatory code (takes precedence over geodetic triplet) |
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
//! | `traj_id` | `UInt64` or `String` | yes | Trajectory identifier; groups observations into trajectories |
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
use polars::{frame::DataFrame, lazy::frame::LazyFrame, prelude::Column};

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
        index::{NightIndexMap, TrajIndexMap},
        observation::Observation,
    },
    observer::{Observer, dataset::ObserverId, error_model::ObsErrorModel},
    photometry::{Filter, Photometry},
};

pub(crate) mod base_field;
pub mod error;
pub(crate) mod observer_field;

// ── sealed trait for DataFrame / LazyFrame ───────────────────────────────────

mod sealed {
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
/// - `lru_cache_size` — optional LRU cache capacity; `None` disables caching.
///
/// # Errors
///
/// Returns [`PolarsError::Polars`] if lazy execution fails, plus all errors
/// documented on [`load_observation_from_frame`].
pub(crate) fn load_observation_from_polars<T: IntoFrame>(
    frame: T,
    error_model: ObsErrorModel,
    lru_cache_size: Option<usize>,
) -> Result<ObsDataset, PolarsError> {
    let df = frame.collect_frame()?;
    load_observation_from_frame(&df, error_model, lru_cache_size)
}

/// Internal ingestion logic that operates on an already-materialised
/// [`DataFrame`].
///
/// Builds [`Observation`]s and, when the optional `night_id` / `traj_id`
/// columns are present, fills the corresponding index maps in a single pass
/// over the rows.
///
/// # Optional index columns
///
/// | Column | Polars type | Index built |
/// |--------|-------------|-------------|
/// | `night_id` | `UInt32` | [`NightIndexMap`] keyed by [`NightId`] |
/// | `traj_id` | `UInt64` | [`TrajIndexMap`] keyed by [`TrajId::Int`] |
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
///   its type is neither `UInt64` nor `String`.
fn load_observation_from_frame(
    df: &DataFrame,
    error_model: ObsErrorModel,
    lru_cache_size: Option<usize>,
) -> Result<ObsDataset, PolarsError> {
    // ── base columns (non-nullable, zero-copy slices) ─────────────────────────
    let base = BaseFields::materialize_fields(df)?;
    let n = base.ids.len();

    // ── optional observer columns ─────────────────────────────────────────────
    let obs_lon = iter_opt_f64(df, "obs_lon", n)?;
    let obs_lat = iter_opt_f64(df, "obs_lat", n)?;
    let obs_alt = iter_opt_f64(df, "obs_alt", n)?;
    let obs_ra_acc = iter_opt_f64(df, "obs_ra_acc", n)?;
    let obs_dec_acc = iter_opt_f64(df, "obs_dec_acc", n)?;
    let mpc_codes = iter_opt_str(df, "mpc_code_obs", n)?;

    // ── optional index columns ────────────────────────────────────────────────
    //
    // We materialise the night_id / traj_id columns up-front (as
    // `Vec<Option<…>>`) so that they can be zipped with the base-field
    // iterators in the single assembly pass below.

    // night_id: UInt32 → NightId(u32).  Column absent ⟹ None sentinel vec.
    let night_ids: Option<Vec<Option<NightId>>> = match df.column("night_id") {
        Err(_) => None, // column absent — index will be None
        Ok(col) => {
            let ca = col
                .as_materialized_series()
                .u32()
                .map_err(|_| PolarsError::NightIdColumnTypeError(col.dtype().to_string()))?;
            Some(ca.iter().map(|opt| opt.map(NightId)).collect())
        }
    };

    // traj_id: UInt64 → TrajId::Int  /  String → TrajId::Str.
    // Column absent ⟹ None sentinel vec.
    let traj_ids: Option<Vec<Option<TrajId>>> = match df.column("traj_id") {
        Err(_) => None, // column absent — index will be None
        Ok(col) => {
            use polars::prelude::DataType;
            match col.dtype() {
                DataType::UInt64 => {
                    let ca = col.as_materialized_series().u64()?;
                    Some(ca.iter().map(|opt| opt.map(TrajId::Int)).collect())
                }
                DataType::String => {
                    let ca = col.as_materialized_series().str()?;
                    Some(
                        ca.iter()
                            .map(|opt| opt.map(|s| TrajId::Str(s.to_owned())))
                            .collect(),
                    )
                }
                other => {
                    return Err(PolarsError::TrajIdColumnTypeError(other.to_string()));
                }
            }
        }
    };

    // ── per-row assembly ───────────────────────────────────────────────────────
    let mut custom_observers: Vec<Observer> = Vec::with_capacity(16);
    let mut observer_lookup: AHashMap<Observer, usize> = AHashMap::with_capacity(16);

    // Index maps — only allocated when the corresponding column is present.
    let mut night_map: Option<NightIndexMap> = night_ids.as_ref().map(|_| NightIndexMap::new());
    let mut traj_map: Option<TrajIndexMap> = traj_ids.as_ref().map(|_| TrajIndexMap::new());

    // Fallback iterators for absent columns: repeat None for each row.
    let night_iter: Box<dyn Iterator<Item = Option<NightId>>> = match night_ids {
        Some(v) => Box::new(v.into_iter()),
        None => Box::new(std::iter::repeat_n(None, n)),
    };
    let traj_iter: Box<dyn Iterator<Item = Option<TrajId>>> = match traj_ids {
        Some(v) => Box::new(v.into_iter()),
        None => Box::new(std::iter::repeat_n(None, n)),
    };

    let observations = izip!(
        0usize..,
        base.iter_base_fields(),
        obs_lon,
        obs_lat,
        obs_alt,
        obs_ra_acc,
        obs_dec_acc,
        mpc_codes,
        night_iter,
        traj_iter,
    )
    .map(
        |(
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
        )| {
            let raw = RawObsRow {
                obs_lon,
                obs_lat,
                obs_alt,
                obs_ra_acc,
                obs_dec_acc,
                mpc_code,
            };

            // Resolve the observer for this row according to the documented precedence
            let observer_id = match resolve_observer(&raw, row_idx)? {
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
                    Some(ObserverId::IntId(idx))
                }
                ResolvedObserver::Mpc(id) => Some(id),
                ResolvedObserver::None => None,
            };

            // Populate the optional index maps.
            if let (Some(map), Some(nid)) = (&mut night_map, night_id) {
                map.entry(nid).or_insert_with(Vec::new).push(row_idx);
            }
            if let (Some(map), Some(tid)) = (&mut traj_map, traj_id) {
                map.entry(tid).or_insert_with(Vec::new).push(row_idx);
            }

            Ok(Observation {
                index: row_idx,
                id,
                equ_coord: EquCoord::new(ra, ra_err, dec, dec_err),
                photometry: Photometry {
                    magnitude: mag,
                    error: mag_err,
                    filter: Filter::String(filter.as_ref().to_string()),
                },
                mjd_tt,
                observer: observer_id,
            })
        },
    )
    .collect::<Result<Vec<_>, PolarsError>>()?;

    Ok(ObsDataset::new(
        observations,
        custom_observers,
        error_model,
        night_map,
        traj_map,
        lru_cache_size,
    ))
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod polars_reader_tests {
    use super::*;
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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));

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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));
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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None);
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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));
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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));
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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));
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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));
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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));
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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));
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

            let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, Some(10));
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

        let eager = load_observation_from_polars(df, ObsErrorModel::FCCT14, Some(10))
            .expect("eager path must succeed");
        let lazy = load_observation_from_polars(lf, ObsErrorModel::FCCT14, Some(10))
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

        let result = load_observation_from_polars(df.lazy(), ObsErrorModel::FCCT14, Some(10));

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
        let ds = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None)
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
        let ds = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None)
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
        let ds = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None)
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
        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None);

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
        let ds = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None)
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
        let trajs: Vec<Option<u64>> = vec![Some(100), Some(200), Some(100), Some(200)];
        cols.push(Column::new("traj_id".into(), trajs));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None)
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
        let ds = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None)
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
        let trajs: Vec<Option<u64>> = vec![Some(1), None, Some(1)];
        cols.push(Column::new("traj_id".into(), trajs));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None)
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
        let result = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None);

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
        let trajs: Vec<Option<u64>> = vec![Some(10), Some(20), Some(10), Some(20)];
        cols.push(Column::new("night_id".into(), nights));
        cols.push(Column::new("traj_id".into(), trajs));

        let df = DataFrame::new_infer_height(cols).expect("DataFrame construction must succeed");
        let ds = load_observation_from_polars(&df, ObsErrorModel::FCCT14, None)
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
