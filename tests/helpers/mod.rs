//! Shared helpers and constants for integration tests.
//!
//! Include this module in each integration test file with:
//!
//! ```rust,ignore
//! mod helpers;
//! use helpers::*;
//! ```
//!
//! The fixture-loading helpers (`load_int`, `load_str`) require the `polars`
//! feature.  Include the file-level gate `#![cfg(feature = "polars")]` in
//! each test file that uses them, or guard individual items with
//! `#[cfg(feature = "polars")]`.

// ── path constants ─────────────────────────────────────────────────────────────

/// Path to the integer-trajectory Parquet fixture.
pub const PATH_INT: &str = "tests/data/test_data_traj_int.parquet";

/// Path to the string-trajectory Parquet fixture.
pub const PATH_STR: &str = "tests/data/test_data_traj_str.parquet";

// ── expectation constants ──────────────────────────────────────────────────────

/// Total number of rows in each fixture file.
pub const TOTAL_ROWS: usize = 561_287;

/// Number of distinct nights in both fixture files.
#[allow(dead_code)]
pub const NIGHT_COUNT: usize = 10;

/// Number of observations with a non-null trajectory identifier.
#[allow(dead_code)]
pub const TRAJ_NON_NULL: usize = 68_145;

/// Number of distinct trajectory identifiers.
#[allow(dead_code)]
pub const TRAJ_UNIQUE: usize = 38_024;

/// Night identifiers and their expected observation counts (from the fixture).
pub const NIGHT_EXPECTED: &[(u32, usize)] = &[
    (3010, 78204),
    (3026, 37244),
    (3074, 86675),
    (3081, 86693),
    (3140, 88273),
    (3177, 44772),
    (3200, 82158),
    (3204, 29196),
    (3248, 11674),
    (3278, 16398),
];

/// Geodetic longitude (radians) of the observatory in the int fixture.
#[allow(dead_code)]
pub const OBS_LON: f64 = -2.0391;

/// Geodetic latitude (radians) of the observatory in the int fixture.
#[allow(dead_code)]
pub const OBS_LAT: f64 = 0.5822;

/// Altitude (metres) of the observatory in the int fixture.
#[allow(dead_code)]
pub const OBS_ALT: f64 = 1712.0;

/// MPC observatory code present in every row of the str fixture.
#[allow(dead_code)]
pub const MPC_CODE: &[u8; 3] = b"I41";

/// Tolerance for floating-point comparisons of observer parallax constants.
#[allow(dead_code)]
pub const OBSERVER_TOLERANCE: f64 = 1e-9;

// ── fixture loaders (datafusion) ───────────────────────────────────────────────

#[cfg(feature = "datafusion")]
use photom::io::datafusion::loader::LoadObsArgs;
#[cfg(feature = "datafusion")]
use photom::observation_dataset::ObsDataset as DfObsDataset;

/// Load the integer-traj fixture via `ObsDataset::from_parquet_uri` (`file://`).
#[cfg(feature = "datafusion")]
#[allow(dead_code)]
pub fn df_load_int() -> DfObsDataset {
    let uri = format!(
        "file://{}/{}",
        std::env::current_dir()
            .expect("current_dir must be accessible")
            .display(),
        PATH_INT
    );
    DfObsDataset::from_parquet_uri(&uri, LoadObsArgs::default())
        .expect("from_parquet_uri must succeed for int file")
}

/// Load the string-traj fixture via `ObsDataset::from_parquet_uri` (`file://`).
#[cfg(feature = "datafusion")]
#[allow(dead_code)]
pub fn df_load_str() -> DfObsDataset {
    let uri = format!(
        "file://{}/{}",
        std::env::current_dir()
            .expect("current_dir must be accessible")
            .display(),
        PATH_STR
    );
    DfObsDataset::from_parquet_uri(&uri, LoadObsArgs::default())
        .expect("from_parquet_uri must succeed for str file")
}

#[cfg(feature = "polars")]
use photom::{io::polars::FromPolarsArgs, observation_dataset::ObsDataset};
#[cfg(feature = "polars")]
use polars::prelude::{LazyFrame, ScanArgsParquet};

/// Load the integer-traj fixture via `ObsDataset::from_lazy`.
///
/// Uses `ScanArgsParquet { rechunk: true }` and `do_rechunk: Some(false)` for
/// the fast path (Polars rechunks during scan; the ingestion pipeline skips
/// its own rechunk step).
#[cfg(feature = "polars")]
#[allow(dead_code)]
pub fn load_int() -> ObsDataset {
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(PATH_INT.into(), args).expect("scan_parquet must succeed");
    ObsDataset::from_lazy(
        lf,
        FromPolarsArgs {
            do_rechunk: Some(false),
            ..Default::default()
        },
    )
    .expect("from_lazy must succeed for int file")
}

/// Load the string-traj fixture via `ObsDataset::from_lazy`.
#[cfg(feature = "polars")]
#[allow(dead_code)]
pub fn load_str() -> ObsDataset {
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(PATH_STR.into(), args).expect("scan_parquet must succeed");
    ObsDataset::from_lazy(
        lf,
        FromPolarsArgs {
            do_rechunk: Some(false),
            ..Default::default()
        },
    )
    .expect("from_lazy must succeed for str file")
}
