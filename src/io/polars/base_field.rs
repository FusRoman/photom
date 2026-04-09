//! Materialization of the base observation columns from a Polars [`DataFrame`].
//!
//! Every observation schema (default geodetic, MPC, MPC-with-accuracy) shares a
//! common set of nine columns defined by [`base_fields()`]:
//!
//! | Column | Type | Description |
//! |-------------|-----------|----------------------------------------------|
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
//! ### Filter label interning
//!
//! The `filter` column holds string data, which Polars cannot expose as a
//! borrowed slice. Instead, [`BaseFields`] owns the materialized [`Series`] and
//! builds a [`AHashMap`](ahash::AHashMap) intern pool at construction time: each
//! unique filter label is allocated into an [`Arc<str>`] exactly once. Every
//! subsequent access via [`BaseFields::iter_base_fields`] clones the matching
//! `Arc` handle, paying only an atomic reference-count increment rather than a
//! heap allocation.
//!
//! ### Relationship between `base_fields()` and `BaseFields`
//!
//! [`base_fields()`] is the single source of truth for both the schema
//! definition (used by [`crate::schema`]) and the column-extraction logic in
//! [`BaseFields::materialize_fields`]. The materialization loop iterates
//! [`base_fields()`] and dispatches each column solely on its declared
//! [`DataType`], so adding or reordering columns only requires editing
//! [`base_fields()`] — no name string is hard-coded anywhere else in this
//! module.

use std::sync::Arc;

use ahash::AHashMap;
use itertools::izip;
use polars::{
    frame::DataFrame,
    prelude::{self as pl, DataType},
    series::Series,
};

use crate::io::polars::{f64_slice, u64_slice, PolarsError};

/// Returns an iterator over the name–type pairs that form the base observation schema.
///
/// The iterator yields the nine columns shared by every observation schema variant,
/// in declaration order:
/// `id` (`UInt64`), `ra`, `ra_err`, `dec`, `dec_err`, `magnitude`, `mag_err`
/// (all `Float64`), `filter` (`String`), and `mjd_tt` (`Float64`).
///
/// This function is the single source of truth for both schema construction
/// (see [`crate::schema`]) and column materialization
/// (see [`BaseFields::materialize_fields`]). Schema-level field additions must
/// be made here.
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

/// Holds zero-copy slices and the owned `filter` series for all base observation columns.
///
/// Numeric columns are represented as contiguous borrowed slices (`&'a [u64]` or
/// `&'a [f64]`), giving bound-check-free row iteration at zero allocation cost.
/// The `filter` column cannot be exposed as a slice because Polars stores string
/// data separately; instead, [`filter_series`](BaseFields::filter_series) owns
/// the materialized [`Series`] so that [`ChunkedArray`](polars::prelude::ChunkedArray)
/// borrows made during iteration remain valid.
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
    /// Owned `Series` for the `filter` column.
    ///
    /// Kept alive so that `&ChunkedArray` borrows produced inside
    /// [`iter_base_fields`](BaseFields::iter_base_fields) remain valid for the
    /// lifetime of the iterator.
    pub(crate) filter_series: Series,
    /// Intern pool mapping each unique filter label to a shared [`Arc<str>`].
    ///
    /// Built once in [`materialize_fields`](BaseFields::materialize_fields); every
    /// subsequent lookup in [`iter_base_fields`](BaseFields::iter_base_fields)
    /// clones an `Arc` handle instead of allocating a new string.
    pub(crate) filter_pool: AHashMap<String, Arc<str>>,
}

impl<'a> BaseFields<'a> {
    /// Extracts and materializes all base columns from `df` into a [`BaseFields`] struct.
    ///
    /// Column names and types are driven entirely by [`base_fields()`] — no column
    /// name is hard-coded here. Each column is dispatched by its declared [`DataType`]:
    ///
    /// - `UInt64`  → borrowed contiguous `&[u64]` slice (`ids`)
    /// - `Float64` → borrowed contiguous `&[f64]` slice, collected in declaration order:
    ///   `ra`, `ra_err`, `dec`, `dec_err`, `magnitude`, `mag_err`, `mjd_tt`
    /// - `String`  → owned [`Series`] stored in [`BaseFields::filter_series`], plus an
    ///   intern pool of `Arc<str>` stored in [`BaseFields::filter_pool`]
    ///
    /// The filter intern pool is built once so that each call to
    /// [`BaseFields::iter_base_fields`] pays only an atomic reference-count increment
    /// per row rather than a new heap allocation.
    ///
    /// # Errors
    ///
    /// Returns a [`PolarsResult`] error if any column named by [`base_fields()`] is
    /// absent from `df`, or if a column's data cannot be cast to the expected type
    /// (e.g. the `filter` column is not a `String` series).
    ///
    /// # Panics
    ///
    /// Panics if the layout of [`base_fields()`] has been modified in an inconsistent
    /// way — specifically if it no longer contains exactly one `UInt64` column, exactly
    /// one `String` column, or exactly seven `Float64` columns. These invariants are
    /// checked with `expect` at construction time.
    pub(crate) fn materialize_fields(df: &'a DataFrame) -> Result<Self, PolarsError> {
        let mut ids_slot: Option<&'a [u64]> = None;
        let mut f64_slices: Vec<&'a [f64]> = Vec::new();
        let mut filter_series_slot: Option<Series> = None;

        for (name, dtype) in base_fields() {
            let col = df.column(&name)?;
            match dtype {
                DataType::UInt64 => ids_slot = Some(u64_slice(col)?),
                DataType::Float64 => f64_slices.push(f64_slice(col)?),
                DataType::String => {
                    filter_series_slot = Some(col.as_materialized_series().clone());
                }
                _ => {}
            }
        }

        let ids = ids_slot.expect("base_fields() must contain exactly one UInt64 column (id)");
        let filter_series = filter_series_slot
            .expect("base_fields() must contain exactly one String column (filter)");

        // Float64 columns arrive in declaration order from base_fields():
        // ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt  (7 total)
        let [ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt]: [&'a [f64]; 7] = f64_slices
            .try_into()
            .expect("base_fields() must contain exactly 7 Float64 columns");

        let filter_pool: AHashMap<String, Arc<str>> = {
            let ca = filter_series.str()?;
            let mut pool = AHashMap::new();
            for opt in ca.iter() {
                if let Some(s) = opt {
                    if !pool.contains_key(s) {
                        pool.insert(s.to_string(), Arc::from(s));
                    }
                }
            }
            pool
        };

        Ok(BaseFields {
            ids,
            ra,
            ra_err,
            dec,
            dec_err,
            magnitude,
            mag_err,
            mjd_tt,
            filter_series,
            filter_pool,
        })
    }

    /// Returns an iterator yielding one tuple per row over all nine base columns.
    ///
    /// Each tuple is `(id, ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt, filter)`,
    /// where all numeric elements are shared references into the borrowed slices and
    /// `filter` is an [`Arc<str>`] cloned from the intern pool built at construction
    /// time. All rows that share the same filter label clone the same `Arc`, so the
    /// per-row cost is a single atomic reference-count increment rather than a heap
    /// allocation.
    ///
    /// The iterator zips all nine columns with [`izip!`](itertools::izip), which
    /// eliminates per-element bounds checks and stops as soon as the shortest slice
    /// is exhausted. Because all slices are derived from the same validated
    /// `DataFrame`, they are guaranteed to have equal length.
    ///
    /// # Panics
    ///
    /// Panics if [`BaseFields::filter_series`] is not a `String` series (which
    /// cannot occur when the struct was constructed via
    /// [`BaseFields::materialize_fields`]), or if a `filter` value encountered
    /// during iteration is absent from [`BaseFields::filter_pool`] (which cannot
    /// occur because the pool is built from the same series).
    pub(crate) fn iter_base_fields(
        &self,
    ) -> impl Iterator<Item = (&u64, &f64, &f64, &f64, &f64, &f64, &f64, &f64, Arc<str>)> + '_ {
        let filter_ca = self
            .filter_series
            .str()
            .expect("filter column is not String");
        izip!(
            self.ids.iter(),
            self.ra.iter(),
            self.ra_err.iter(),
            self.dec.iter(),
            self.dec_err.iter(),
            self.magnitude.iter(),
            self.mag_err.iter(),
            self.mjd_tt.iter(),
            filter_ca.iter(),
        )
        .map(
            |(id, ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt, filter)| {
                let filter = self
                    .filter_pool
                    .get(filter.unwrap_or_default())
                    .expect("filter value not in pool — pool was built from the same series")
                    .clone();
                (
                    id, ra, ra_err, dec, dec_err, magnitude, mag_err, mjd_tt, filter,
                )
            },
        )
    }
}
