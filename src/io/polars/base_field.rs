//! Materialization of the base observation columns from a Polars [`DataFrame`].
//!
//! Every observation schema (default geodetic, MPC, MPC-with-accuracy) shares a
//! common set of nine columns defined by [`base_fields()`]:
//!
//! | Column | Type | Description |
//! |-------------|------------------------|----------------------------------------------|
//! | `id` | `UInt64` | Unique observation identifier |
//! | `ra` | `Float64` | Right ascension (degrees) |
//! | `ra_err` | `Float64` | Right ascension uncertainty (degrees) |
//! | `dec` | `Float64` | Declination (degrees) |
//! | `dec_err` | `Float64` | Declination uncertainty (degrees) |
//! | `magnitude` | `Float64` | Apparent magnitude |
//! | `mag_err` | `Float64` | Magnitude uncertainty |
//! | `filter` | `String` **or** `UInt32` | Photometric filter label or integer code |
//! | `mjd_tt` | `Float64` | Epoch (Modified Julian Date, Terrestrial Time) |
//!
//! ## Design decisions
//!
//! ### Zero-copy numeric extraction
//!
//! Numeric columns (`UInt64` and `Float64`) are extracted as contiguous slices
//! borrowed directly from the underlying Polars memory without any copying.
//! This is only possible when the column has a single chunk; the caller must
//! ensure the `DataFrame` is not fragmented (e.g. by calling
//! `DataFrame::rechunk` beforehand if necessary).
//!
//! ### Filter column
//!
//! The `filter` column may be either a `String` series or a `UInt32` series —
//! but never a mixture of both within the same `DataFrame`.  The column type is
//! detected once during [`BaseFields::materialize_fields`] and stored as a
//! [`FilterData`] variant.  Each call to
//! [`BaseFields::iter_base_fields`] produces a [`Filter`] value directly from
//! the underlying series without any interning or shared-pointer overhead.

use itertools::izip;
use polars::{
    frame::DataFrame,
    prelude::{self as pl, DataType},
    series::Series,
};

use crate::{
    io::polars::{PolarsError, f64_slice, u64_slice},
    photometry::Filter,
};

/// One row of base observation data yielded by [`BaseFields::iter_base_fields`].
///
/// The tuple order mirrors the declaration order of [`base_fields()`]:
/// `(id, ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt, filter)`.
pub(crate) type BaseRow<'a> = (
    &'a u64,
    &'a f64,
    &'a f64,
    &'a f64,
    &'a f64,
    &'a f64,
    &'a f64,
    &'a f64,
    Filter,
);

/// Returns an iterator over the name–type pairs that form the base observation schema.
///
/// The iterator yields the nine columns shared by every observation schema variant,
/// in declaration order:
/// `id` (`UInt64`), `ra`, `ra_err`, `dec`, `dec_err`, `magnitude`, `mag_err`
/// (all `Float64`), `filter` (`String` or `UInt32`), and `mjd_tt` (`Float64`).
///
/// This function is the single source of truth for both schema construction
/// (see [`crate::schema`]) and column materialization
/// (see [`BaseFields::materialize_fields`]). Schema-level field additions must
/// be made here.
///
/// > **Note:** the `filter` column is listed here with type `String` for schema
/// > documentation purposes only. The actual materialization logic in
/// > [`BaseFields::materialize_fields`] accepts both `String` and `UInt32`.
pub(crate) fn base_fields() -> impl Iterator<Item = (pl::PlSmallStr, pl::DataType)> {
    [
        ("id".into(), pl::DataType::UInt64),
        ("ra".into(), pl::DataType::Float64),
        ("ra_err".into(), pl::DataType::Float64),
        ("dec".into(), pl::DataType::Float64),
        ("dec_err".into(), pl::DataType::Float64),
        ("magnitude".into(), pl::DataType::Float64),
        ("mag_err".into(), pl::DataType::Float64),
        ("filter".into(), pl::DataType::String),
        ("mjd_tt".into(), pl::DataType::Float64),
    ]
    .into_iter()
}

/// Owned storage for the `filter` column, discriminated by Polars type.
///
/// The two variants correspond to the two accepted column types:
///
/// - [`FilterData::Str`] — the column is a `String` series; each row produces a
///   [`Filter::String`].
/// - [`FilterData::Int`] — the column is a `UInt32` series; each row produces a
///   [`Filter::Int`].
enum FilterData {
    Str(Series),
    Int(Series),
}

/// Holds zero-copy slices and the owned `filter` series for all base observation columns.
///
/// Numeric columns are represented as contiguous borrowed slices (`&'a [u64]` or
/// `&'a [f64]`), giving bound-check-free row iteration at zero allocation cost.
/// The `filter` column cannot be exposed as a slice because Polars stores string
/// data separately; instead, [`filter`](BaseFields::filter) owns the materialized
/// [`Series`] (in either its `String` or `UInt32` form) so that
/// [`ChunkedArray`](polars::prelude::ChunkedArray) borrows made during iteration
/// remain valid.
///
/// Construct via [`BaseFields::materialize_fields`], then iterate with
/// [`BaseFields::iter_base_fields`].
pub(crate) struct BaseFields<'a> {
    /// Unique observation identifiers (`id` column).
    pub(crate) ids: &'a [u64],
    /// Right ascension values in degrees (`ra` column).
    pub(crate) ra: &'a [f64],
    /// Right ascension uncertainties in degrees (`ra_err` column).
    pub(crate) ra_err: &'a [f64],
    /// Declination values in degrees (`dec` column).
    pub(crate) dec: &'a [f64],
    /// Declination uncertainties in degrees (`dec_err` column).
    pub(crate) dec_err: &'a [f64],
    /// Apparent magnitudes (`magnitude` column).
    pub(crate) magnitude: &'a [f64],
    /// Magnitude uncertainties (`mag_err` column).
    pub(crate) mag_err: &'a [f64],
    /// Observation epochs in Modified Julian Date (Terrestrial Time) (`mjd_tt` column).
    pub(crate) mjd_tt: &'a [f64],
    /// Owned series for the `filter` column, in either its `String` or `UInt32` form.
    filter: FilterData,
}

impl<'a> BaseFields<'a> {
    /// Extracts and materializes all base columns from `df` into a [`BaseFields`] struct.
    ///
    /// Column names and types are driven by [`base_fields()`] for all columns
    /// except `filter`, which accepts either `String` or `UInt32`:
    ///
    /// - `UInt64`  → borrowed contiguous `&[u64]` slice (`ids`)
    /// - `Float64` → borrowed contiguous `&[f64]` slice, collected in declaration order:
    ///   `ra`, `ra_err`, `dec`, `dec_err`, `magnitude`, `mag_err`, `mjd_tt`
    /// - `String`  → owned [`Series`] stored as [`FilterData::Str`]
    /// - `UInt32`  → owned [`Series`] stored as [`FilterData::Int`]
    ///
    /// The `filter` column type is detected at construction time.  A column
    /// that is neither `String` nor `UInt32` is rejected with a
    /// [`PolarsError::FilterColumnTypeError`].
    ///
    /// # Errors
    ///
    /// Returns a [`PolarsError`] if any required column is absent from `df`,
    /// if a numeric column cannot be cast to the expected type, or if the
    /// `filter` column is present but has an unsupported type.
    ///
    /// # Panics
    ///
    /// Panics if the layout of [`base_fields()`] has been modified in an inconsistent
    /// way — specifically if it no longer contains exactly one `UInt64` column or
    /// exactly seven `Float64` columns. These invariants are checked with `expect`
    /// at construction time.
    pub(crate) fn materialize_fields(df: &'a DataFrame) -> Result<Self, PolarsError> {
        let mut ids_slot: Option<&'a [u64]> = None;
        let mut f64_slices: Vec<&'a [f64]> = Vec::new();
        let mut filter_slot: Option<FilterData> = None;

        // Process all columns declared by base_fields(), but handle `filter`
        // separately because it accepts two possible Polars types.
        for (name, dtype) in base_fields() {
            let col = df.column(&name)?;
            if name == "filter" {
                // Accept String or UInt32; reject everything else.
                let series = col.as_materialized_series().clone();
                let filter_data = match col.dtype() {
                    DataType::String => FilterData::Str(series),
                    DataType::UInt32 => FilterData::Int(series),
                    other => {
                        return Err(PolarsError::FilterColumnTypeError(other.to_string()));
                    }
                };
                filter_slot = Some(filter_data);
            } else {
                match dtype {
                    DataType::UInt64 => ids_slot = Some(u64_slice(col)?),
                    DataType::Float64 => f64_slices.push(f64_slice(col)?),
                    _ => {}
                }
            }
        }

        let ids = ids_slot.expect("base_fields() must contain exactly one UInt64 column (id)");
        let filter =
            filter_slot.expect("base_fields() must contain exactly one String column (filter)");

        // Float64 columns arrive in declaration order from base_fields():
        // ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt  (7 total)
        let [ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt]: [&'a [f64]; 7] = f64_slices
            .try_into()
            .expect("base_fields() must contain exactly 7 Float64 columns");

        Ok(BaseFields {
            ids,
            ra,
            ra_err,
            dec,
            dec_err,
            magnitude,
            mag_err,
            mjd_tt,
            filter,
        })
    }

    /// Returns an iterator yielding one tuple per row over all nine base columns,
    /// or a [`PolarsError`] if the internal filter series cannot be cast to the
    /// expected type.
    ///
    /// Each tuple is `(id, ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt, filter)`,
    /// where all numeric elements are shared references into the borrowed slices and
    /// `filter` is a freshly constructed [`Filter`] value for each row.
    ///
    /// Depending on the column type detected at construction:
    ///
    /// - `String` column → yields [`Filter::String`] with an owned [`String`].
    /// - `UInt32` column → yields [`Filter::Int`] with a `u32` value.
    ///
    /// The iterator zips all nine columns with [`izip!`](itertools::izip), which
    /// eliminates per-element bounds checks and stops as soon as the shortest slice
    /// is exhausted. Because all slices are derived from the same validated
    /// `DataFrame`, they are guaranteed to have equal length.
    ///
    /// # Errors
    ///
    /// Returns [`PolarsError::Polars`] if the internal [`FilterData`] series
    /// cannot be cast to its declared type.  In practice this cannot occur when
    /// the struct was constructed via [`BaseFields::materialize_fields`], but the
    /// error is propagated rather than panicked to keep the API honest.
    pub(crate) fn iter_base_fields(
        &self,
    ) -> Result<impl Iterator<Item = BaseRow<'_>> + '_, PolarsError> {
        let numeric = izip!(
            self.ids.iter(),
            self.ra.iter(),
            self.ra_err.iter(),
            self.dec.iter(),
            self.dec_err.iter(),
            self.magnitude.iter(),
            self.mag_err.iter(),
            self.mjd_tt.iter(),
        );

        match &self.filter {
            FilterData::Str(series) => {
                let ca = series.str()?;
                let filter_iter = ca
                    .iter()
                    .map(|opt| Filter::String(opt.unwrap_or_default().to_owned()));
                Ok(itertools::Either::Left(izip!(numeric, filter_iter).map(
                    |((id, ra, ra_err, dec, dec_err, mag, mag_err, mjd), f)| {
                        (id, ra, ra_err, dec, dec_err, mag, mag_err, mjd, f)
                    },
                )))
            }
            FilterData::Int(series) => {
                let ca = series.u32()?;
                let filter_iter = ca.iter().map(|opt| Filter::Int(opt.unwrap_or(0)));
                Ok(itertools::Either::Right(izip!(numeric, filter_iter).map(
                    |((id, ra, ra_err, dec, dec_err, mag, mag_err, mjd), f)| {
                        (id, ra, ra_err, dec, dec_err, mag, mag_err, mjd, f)
                    },
                )))
            }
        }
    }
}
