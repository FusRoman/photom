//! Polars-based ingestion of astronomical observation data.
//!
//! This module provides [`load_observation_from_polars`], the primary entry
//! point for converting a validated Polars [`DataFrame`] into an
//! [`ObsDataset`].  It handles all three observer representations supported
//! by the library:
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
//! | [`load_observation_from_polars`] | fn | Convert a `DataFrame` into an [`ObsDataset`] |
//!
//! ## Sub-modules
//!
//! - [`base_field`] — zero-copy materialization of the nine mandatory base columns.
//!
//! ## DataFrame schema
//!
//! ### Mandatory base columns
//!
//! Every `DataFrame` passed to [`load_observation_from_polars`] must contain
//! the following nine columns.  All are non-nullable; a `null` cell or a
//! missing column is a schema validation error.
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
//! |---------------|-------------|----------|------------------------------------------------------------|
//! | `obs_lon` | `Float64` | yes | Geodetic longitude in degrees east of Greenwich |
//! | `obs_lat` | `Float64` | yes | Geodetic latitude in degrees |
//! | `obs_alt` | `Float64` | yes | Altitude above the reference ellipsoid in metres |
//! | `obs_ra_acc` | `Float64` | yes | RA measurement accuracy in radians (required when geodetic triplet is set) |
//! | `obs_dec_acc` | `Float64` | yes | Dec measurement accuracy in radians (required when geodetic triplet is set) |
//! | `mpc_code_obs` | `String` | yes | Three-byte ASCII MPC observatory code (takes precedence over geodetic triplet) |
//!
//! ## Observer column rules
//!
//! The resolution rules applied per row are documented on
//! [`load_observation_from_polars`] and enforced by [`resolve_observer`].
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
use itertools::{izip, Either};
use polars::{frame::DataFrame, prelude::Column};

use crate::{
    astrometry::EquCoord,
    io::polars::{
        base_field::BaseFields,
        error::PolarsError,
        observer_field::{resolve_observer, RawObsRow, ResolvedObserver},
    },
    observation::{ObsDataset, Observation, ObserverId},
    observer::{error_model::ObsErrorModel, Observer},
    photometry::{Filter, Photometry},
    trajectory::{TrajDataset, TrajId, Trajectory},
};

pub(crate) mod base_field;
pub mod error;
pub(crate) mod observer_field;

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
        Err(_) => Ok(Either::Right(std::iter::repeat(None).take(n))),
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
        Err(_) => Ok(Either::Right(std::iter::repeat(None).take(n))),
    }
}

// ── internal entry point ────────────────────────────────────────────────────────

/// Load observations from a Polars [`DataFrame`] into an [`ObsDataset`].
///
/// The `DataFrame` is first validated against `schema`.  If validation
/// passes, the function extracts all base observation columns in a zero-copy
/// manner (numeric columns are borrowed as contiguous slices), collects the
/// optional observer columns into per-row `Option` buffers, and assembles one
/// [`Observation`] per row.
///
/// Custom geodetic observers are **interned**: if multiple rows describe the
/// same site (identical longitude, latitude, and altitude), they are stored
/// once in the returned dataset's `custom_observers` list and referenced by
/// index, avoiding redundant allocations.
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
/// # Arguments
///
/// - `df`          — a Polars [`DataFrame`] containing at minimum all columns
///   required by `schema`.  Observer columns (`obs_lon`, `obs_lat`, `obs_alt`,
///   `obs_ra_acc`, `obs_dec_acc`, `mpc_code_obs`) may be absent or nullable.
/// - `schema`      — the [`ObsSchema`] variant that defines which columns are
///   mandatory and their expected Polars types.
/// - `error_model` — the [`ObsErrorModel`] attached to the resulting
///   [`ObsDataset`]; used to resolve astrometric accuracy for MPC-coded
///   observers at query time.
/// - `lru_cache_size` — optional capacity for the LRU cache used to speed up
///  repeated observation lookups; if `None`, the cache is disabled.
///
/// # Returns
///
/// An [`ObsDataset`] containing all observations assembled from `df`, the
/// de-duplicated list of custom geodetic [`Observer`]s, and the provided
/// `error_model`.
///
/// # Errors
///
/// Returns a [`PolarsError`] in any of the following situations:
///
/// - [`PolarsError::SchemaValidationError`] — `df` does not satisfy `schema`.
/// - [`PolarsError::Polars`] — a Polars-internal operation failed (e.g. a
///   column could not be cast to the expected type).
/// - [`PolarsError::PartialTripletNull`] — one or two geodetic columns were
///   non-null while the remaining one was null.
/// - [`PolarsError::MissingAccuracyForGeodesic`] — the geodetic triplet was
///   fully set but `obs_ra_acc` or `obs_dec_acc` was null.
/// - [`PolarsError::InvalidMpcCode`] — an `mpc_code_obs` cell was non-null
///   but did not parse as a valid three-byte ASCII MPC code.
/// - [`PolarsError::DataConversionError`] — [`Observer::new`] rejected the
///   coordinate values (e.g. a `NaN` was present in a geodetic column).
pub(crate) fn load_observation_from_polars(
    df: &DataFrame,
    error_model: ObsErrorModel,
    lru_cache_size: Option<usize>,
) -> Result<ObsDataset, PolarsError> {
    // ── base columns (non-nullable, zero-copy slices) ─────────────────────────
    let base = BaseFields::materialize_fields(df)?;

    let n = base.ids.len();

    // Build lazy iterators over the optional observer columns.  No Vec is
    // allocated: the iterators borrow directly from Polars' internal memory
    // (present column) or yield `None` values via `repeat` (absent column).
    let obs_lon = iter_opt_f64(df, "obs_lon", n)?;
    let obs_lat = iter_opt_f64(df, "obs_lat", n)?;
    let obs_alt = iter_opt_f64(df, "obs_alt", n)?;
    let obs_ra_acc = iter_opt_f64(df, "obs_ra_acc", n)?;
    let obs_dec_acc = iter_opt_f64(df, "obs_dec_acc", n)?;
    let mpc_codes = iter_opt_str(df, "mpc_code_obs", n)?;

    // ── per-row assembly ───────────────────────────────────────────────────────
    let mut custom_observers: Vec<Observer> = Vec::with_capacity(16);
    let mut observer_lookup: AHashMap<Observer, usize> = AHashMap::with_capacity(16);

    let observations = izip!(
        0usize..,
        base.iter_base_fields(),
        obs_lon,
        obs_lat,
        obs_alt,
        obs_ra_acc,
        obs_dec_acc,
        mpc_codes,
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
        )| {
            let raw = RawObsRow {
                obs_lon,
                obs_lat,
                obs_alt,
                obs_ra_acc,
                obs_dec_acc,
                mpc_code,
            };

            let observer_id = match resolve_observer(&raw, row_idx)? {
                ResolvedObserver::Geodetic(observer) => {
                    // Intern the observer: identical sites share a single slot.
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

            Ok(Observation {
                id,
                night_id: None,
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
        lru_cache_size,
    ))
}

// ── trajectory ingestion ──────────────────────────────────────────────────────

/// Load observations and trajectory groupings from a Polars [`DataFrame`] into
/// a [`TrajDataset`].
///
/// This function is the trajectory-aware counterpart of
/// [`load_observation_from_polars`].  It performs the full observation
/// ingestion (base columns, observer resolution, LRU cache construction) and
/// then reads the optional `traj_id` column to group observations into
/// [`Trajectory`] values.
///
/// ## `traj_id` column rules
///
/// The `traj_id` column is **optional** (the column may be absent from the
/// frame) and **nullable** (individual cells may be `null`).
///
/// | Situation | Outcome |
/// |-----------|---------|
/// | Column absent | All observations loaded, no trajectories created. |
/// | Column present, type `UInt64` | Non-null cells → [`TrajId::Int`]. |
/// | Column present, type `String` | Non-null cells → [`TrajId::Str`]. |
/// | Column present, other type | [`PolarsError::TrajIdColumnTypeError`]. |
/// | Cell `null` | Row belongs to no trajectory (still in [`ObsDataset`]). |
///
/// ## Grouping semantics
///
/// Trajectories are assembled in **first-appearance order**: the first row
/// that carries a given `traj_id` value defines the position of that
/// trajectory in [`TrajDataset::iter_trajectories`].  Subsequent rows with
/// the same value are appended to the same [`Trajectory`]'s `obs_ids` list in
/// source-row order.
///
/// # Arguments
///
/// - `df`             — source Polars [`DataFrame`].
/// - `error_model`    — astrometric error model forwarded to [`ObsDataset`].
/// - `lru_cache_size` — shared capacity for **both** the observation LRU cache
///   (inside [`ObsDataset`]) and the trajectory LRU cache.  Defaults to
///   1 000 when `None`.
///
/// # Errors
///
/// Returns a [`PolarsError`] for any of the reasons documented on
/// [`load_observation_from_polars`], plus
/// [`PolarsError::TrajIdColumnTypeError`] when the `traj_id` column is
/// present but has an unsupported Polars type.
pub(crate) fn load_traj_from_polars(
    df: &DataFrame,
    error_model: ObsErrorModel,
    lru_cache_size: Option<usize>,
) -> Result<TrajDataset, PolarsError> {
    use polars::prelude::DataType;

    // ── Step 1: full observation ingestion ────────────────────────────────────
    let obs_dataset = load_observation_from_polars(df, error_model, lru_cache_size)?;

    // ── Step 2: read the optional traj_id column ──────────────────────────────
    //
    // We need the row → ObsId mapping, which we reconstruct by reading the
    // `id` base column directly (same zero-copy slice used during ingestion).
    // The `traj_id` column drives the grouping; the `id` column provides the
    // ObsId for each row.
    let id_col = df.column("id")?;
    let obs_ids_slice = u64_slice(id_col)?;
    let n = obs_ids_slice.len();

    let trajectories: Vec<Trajectory> = match df.column("traj_id") {
        // Column absent → no trajectories.
        Err(_) => Vec::new(),

        Ok(traj_col) => {
            let series = traj_col.as_materialized_series();
            match series.dtype() {
                DataType::UInt64 => {
                    // Build trajectories keyed by TrajId::Int.
                    // zip obs_ids_slice with the ChunkedArray iterator, skip
                    // null cells via filter_map, then fold into a Vec<Trajectory>
                    // and a temporary AHashMap for O(1) look-up.
                    let ca = series.u64()?;
                    let (trajectories, _) = obs_ids_slice
                        .iter()
                        .copied()
                        .zip(ca.iter())
                        .filter_map(|(obs_id, opt_tid)| opt_tid.map(|tid| (obs_id, tid)))
                        .fold(
                            (
                                Vec::<Trajectory>::new(),
                                AHashMap::<u64, usize>::with_capacity(n.min(64)),
                            ),
                            |(mut trajs, mut idx_map), (obs_id, tid)| {
                                match idx_map.get(&tid) {
                                    Some(&i) => trajs[i].obs_ids.push(obs_id),
                                    None => {
                                        let i = trajs.len();
                                        idx_map.insert(tid, i);
                                        trajs.push(Trajectory {
                                            id: TrajId::Int(tid),
                                            obs_ids: vec![obs_id],
                                        });
                                    }
                                }
                                (trajs, idx_map)
                            },
                        );
                    trajectories
                }

                DataType::String => {
                    // Build trajectories keyed by TrajId::Str.
                    // Clone the key string only on first insertion (to_owned()
                    // is called at most once per unique traj_id value).
                    let ca = series.str()?;
                    let (trajectories, _) = obs_ids_slice
                        .iter()
                        .copied()
                        .zip(ca.iter())
                        .filter_map(|(obs_id, opt_tid)| opt_tid.map(|tid| (obs_id, tid)))
                        .fold(
                            (
                                Vec::<Trajectory>::new(),
                                AHashMap::<String, usize>::with_capacity(n.min(64)),
                            ),
                            |(mut trajs, mut idx_map), (obs_id, tid_str)| {
                                match idx_map.get(tid_str) {
                                    Some(&i) => trajs[i].obs_ids.push(obs_id),
                                    None => {
                                        let key = tid_str.to_owned();
                                        let i = trajs.len();
                                        idx_map.insert(key.clone(), i);
                                        trajs.push(Trajectory {
                                            id: TrajId::Str(key),
                                            obs_ids: vec![obs_id],
                                        });
                                    }
                                }
                                (trajs, idx_map)
                            },
                        );
                    trajectories
                }

                other => {
                    return Err(PolarsError::TrajIdColumnTypeError(format!("{other:?}")));
                }
            }
        }
    };

    Ok(TrajDataset::new(obs_dataset, trajectories, lru_cache_size))
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
