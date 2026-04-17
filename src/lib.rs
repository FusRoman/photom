//! Rust library for loading, structuring, and querying astronomical observation datasets —
//! with trajectory grouping, multi-observer support, and efficient lookups.
//!
//! `photom` provides a type-safe pipeline for ingesting astrometric and photometric
//! measurements, associating them with ground-based observatories, and grouping them into
//! trajectories of moving objects.  The library is designed around two primary dataset
//! types — [`observation_dataset::ObsDataset`] for flat observation collections — with LRU
//! caches providing fast repeated lookups.
//!
//! # Features
//!
//! - **Polars ingestion** (`polars` feature) — load observations from a `DataFrame` or
//!   `LazyFrame` with full schema validation.
//! - **Parallel iteration** (`parallel` feature) — process observations, nights, and
//!   trajectories in parallel via [rayon](https://docs.rs/rayon) with zero data copying.
//! - **ADES ingestion** (`ades` feature) — load observations directly from MPC ADES XML
//!   files ([`observation_dataset::ObsDataset::from_ades`]), supporting both structured
//!   (`obsBlock`/`obsContext`) and flat formats, with automatic MPC observer resolution.
//! - **MPC 80-column ingestion** (`mpc_80_col` feature) — load observations from the
//!   classic MPC fixed-width 80-column ASCII format
//!   ([`observation_dataset::ObsDataset::from_mpc_80_col`]), with automatic trajectory
//!   grouping and nom-based field parsing.
//! - **Parquet ingestion via DataFusion** (`datafusion` feature) — load observations from
//!   any Parquet file reachable by URI (`file://`, `http://`, `https://`, `hdfs://`) using
//!   Apache Arrow / DataFusion ([`observation_dataset::ObsDataset::from_parquet_uri`] and
//!   its async counterpart), with automatic contiguous index optimisation.
//! - **Serialisation / deserialisation** (`serde` feature) — persist and restore an
//!   [`observation_dataset::ObsDataset`] (and all constituent types) via
//!   [serde](https://docs.rs/serde).  Runtime-only state — the LRU cache contents, the
//!   lazy MPC observatory cache, and all derived index maps — is excluded from the
//!   serialised form and rebuilt transparently on deserialisation.  Only the LRU cache
//!   *capacity* is preserved so the restored dataset has identical eviction behaviour.
//! - **Multi-observer support** — MPC observatory codes (resolved lazily from the MPC
//!   website), custom geodetic sites (interned and deduplicated), or unknown observer.
//! - **Trajectory grouping** — group observations by a `traj_id` column; supports both
//!   integer (`UInt64`) and string (`String`) identifiers.
//! - **Three astrometric error models** — FCCT14, CBM10, and VFCC17, used to assign
//!   measurement accuracies to MPC-coded observatories.
//! - **LRU caches** — configurable cache capacity for both observation and trajectory
//!   lookups, avoiding repeated linear scans.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`astrometry`] | Equatorial sky coordinates ([`astrometry::EquCoord`]) with uncertainties and the Vincenty angular-separation formula |
//! | [`photometry`] | Photometric measurement types: apparent magnitude, uncertainty, and bandpass filter ([`photometry::Photometry`], [`photometry::Filter`]) |
//! | [`observation_dataset`] | Core observation types ([`observation_dataset::observation::Observation`], [`observation_dataset::ObsDataset`]) |
//! | [`observer`] | Ground-based observatory representation ([`observer::Observer`]) and geodetic utilities |
//! | [`observer::error_model`] | Astrometric error model variants ([`observer::error_model::ObsErrorModel`]: FCCT14, CBM10, VFCC17) |
//! | [`constants`] | Physical and geodetic constants (Earth axes, AU, etc.) |
#![cfg_attr(
    feature = "polars",
    doc = "| [`io`] | Internal ingestion backends (Polars adapter, schema validation) |"
)]
#![cfg_attr(
    feature = "ades",
    doc = "| [`io::ades`] | ADES XML ingestion backend ([`io::ades`]) |"
)]
#![cfg_attr(
    feature = "mpc_80_col",
    doc = "| [`io::mpc_80_col`] | MPC 80-column ingestion backend ([`io::mpc_80_col`]) |"
)]
#![cfg_attr(
    feature = "datafusion",
    doc = "| [`io::datafusion`] | DataFusion/Arrow Parquet ingestion backend ([`io::datafusion`]) |"
)]
//!
//! # Type Aliases
//!
//! The crate exports four primitive type aliases used throughout the API to make units
//! explicit in function signatures:
//!
//! | Alias | Underlying type | Unit |
//! |-------|-----------------|------|
//! | [`Arcseconds`] | `f64` | Angle in arcseconds |
//! | [`Radians`] | `f64` | Angle in radians |
//! | [`Degrees`] | `f64` | Angle in degrees |
//! | [`MJDTT`] | `f64` | Modified Julian Date (Terrestrial Time) in days |
//! | [`Meters`] | `f64` | Distance in metres |
//!
//! # DataFrame Schema
//!
//! *Requires the `polars` feature.*
//!
//! When loading data via `ObsDataset::from_polars`, `ObsDataset::from_lazy`,
//! `TrajDataset::from_polars`, or `TrajDataset::from_lazy`, the input frame must conform
//! to the following column layout.
//!
//! ## Mandatory base columns (non-nullable)
//!
//! | Column | Polars type | Description |
//! |--------|-------------|-------------|
//! | `id` | `UInt64` | Unique observation identifier |
//! | `ra` | `Float64` | Right ascension (radians) |
//! | `ra_err` | `Float64` | Right ascension uncertainty (radians) |
//! | `dec` | `Float64` | Declination (radians) |
//! | `dec_err` | `Float64` | Declination uncertainty (radians) |
//! | `magnitude` | `Float64` | Apparent magnitude |
//! | `mag_err` | `Float64` | Magnitude uncertainty |
//! | `filter` | `String` | Photometric filter label |
//! | `mjd_tt` | `Float64` | Epoch (MJD, Terrestrial Time) |
//!
//! ## Optional observer columns (nullable; column may be absent)
//!
//! | Column | Polars type | Description |
//! |--------|-------------|-------------|
//! | `obs_lon` | `Float64` | Geodetic longitude (radians, east positive) |
//! | `obs_lat` | `Float64` | Geodetic latitude (radians) |
//! | `obs_alt` | `Float64` | Altitude above ellipsoid (metres) |
//! | `obs_ra_acc` | `Float64` | RA accuracy (radians) — required when the geodetic triplet is set |
//! | `obs_dec_acc` | `Float64` | Dec accuracy (radians) — required when the geodetic triplet is set |
//! | `mpc_code_obs` | `String` | Three-byte ASCII MPC code (takes precedence over geodetic columns) |
//!
//! ## Optional grouping columns
//!
//! | Column | Polars type | Description |
//! |--------|-------------|-------------|
//! | `traj_id` | `UInt32` or `String` | Trajectory identifier; nullable — null rows are loaded into the `ObsDataset` but are not assigned to any trajectory |
//! | `night_id` | `UInt32` | Night identifier; nullable — null rows are included in the `ObsDataset` but are not assigned to any night |
//!
//! ## Observer resolution (per row, in precedence order)
//!
//! 1. `mpc_code_obs` non-null → [`observer::dataset::ObserverId::MpcCode`] (MPC site, resolved lazily).
//! 2. `obs_lon`, `obs_lat`, and `obs_alt` all non-null → [`observer::dataset::ObserverId::IntId`] (geodetic
//!    site; `obs_ra_acc` and `obs_dec_acc` must also be non-null).
//! 3. Otherwise → no observer (`None`).
//!
//! A partially-null geodetic triplet or a complete triplet without accuracy values causes
//! the ingestion to return an error.
//!
//! ## Ingestion arguments (`FromPolarsArgs`)
//!
//! *Requires the `polars` feature.*
//!
//! Both [`observation_dataset::ObsDataset::from_polars`] and
//! [`observation_dataset::ObsDataset::from_lazy`] accept a
//! `FromPolarsArgs` value that controls how the ingestion pipeline behaves.
//! Use `FromPolarsArgs::default()` to get sensible out-of-the-box settings,
//! or construct the struct explicitly to override individual fields.
//!
//! | Field | Type | Default | Description |
//! |-------|------|---------|-------------|
//! | `error_model` | `Option<ObsErrorModel>` | `None` | Astrometric error model used to assign accuracies to MPC-coded observatories; `None` leaves MPC observer accuracies unset until [`ObsDataset::set_error_model`](observation_dataset::ObsDataset::set_error_model) is called |
//! | `lru_cache_size` | `Option<usize>` | `Some(10_000)` | Capacity of the per-dataset LRU cache for observation look-up by `ObsId`; `None` falls back to an internal default |
//! | `do_rechunk` | `Option<bool>` | `Some(false)` | When `true`, forces all multi-chunk columns to be merged into a single contiguous Arrow chunk before ingestion; set to `Some(false)` when the caller has already guaranteed single-chunk layout (e.g. after reading a Parquet file with `rechunk: true`) |
//! | `contiguous_choice` | `Option<ContiguousChoice>` | `Some(ContiguousNight)` | Which grouping column (if any) to sort the frame by before iteration; sorting allows the corresponding index to use compact contiguous ranges instead of per-row index vectors (see below) |
//!
//! ### Contiguous index optimisation (`ContiguousChoice`)
//!
//! By default the ingestion pipeline sorts the input frame by `night_id`
//! (`ContiguousChoice::ContiguousNight`) so that all observations belonging to
//! the same night occupy a single contiguous block in the output `observations`
//! vector.  This lets the night index store a compact `(start, end)` range for
//! each night instead of a `Vec` of scattered positions, which saves memory and
//! improves cache locality during sequential and parallel night iteration.
//!
//! Setting `contiguous_choice` to `ContiguousChoice::ContiguousTraj` applies the
//! same optimisation to trajectories instead.  Setting it to `None` disables the
//! sort entirely; both indices will use the `Vec`-based split representation.
//!
//! Only one grouping column can be made contiguous at a time.  The other
//! column (if present in the frame) is always built as a split index.
//!
//! ```rust,ignore
//! use photom::io::polars::{ContiguousChoice, FromPolarsArgs};
//! use photom::observer::error_model::ObsErrorModel;
//! use photom::observation_dataset::ObsDataset;
//!
//! // Sort by traj_id so trajectory iteration is cache-friendly.
//! let dataset = ObsDataset::from_polars(
//!     &df,
//!     FromPolarsArgs {
//!         error_model: Some(ObsErrorModel::FCCT14),
//!         lru_cache_size: Some(5_000),
//!         contiguous_choice: Some(ContiguousChoice::ContiguousTraj),
//!         ..Default::default()
//!     },
//! )?;
//! ```
//!
//! # Usage Examples
//!
//! ## Build a minimal `DataFrame` and load observations
//!
//! ```rust,ignore
//! use polars::prelude::*;
//! use photom::observation::ObsDataset;
//! use photom::observer::error_model::ObsErrorModel;
//!
//! // Construct a two-row DataFrame matching the required schema.
//! // RA and Dec are in radians; errors are in radians.
//! // Observer accuracy columns (obs_ra_acc, obs_dec_acc) are also in radians.
//! let df = df! {
//!     "id"        => &[1_u64, 2_u64],
//!     "ra"        => &[1.4633_f64, 1.4682_f64],   // radians
//!     "ra_err"    => &[1.745e-5_f64, 1.745e-5_f64], // radians (~1 arcsec)
//!     "dec"       => &[0.3840_f64, 0.3847_f64],   // radians
//!     "dec_err"   => &[1.745e-5_f64, 1.745e-5_f64], // radians (~1 arcsec)
//!     "magnitude" => &[19.3_f64, 19.5_f64],
//!     "mag_err"   => &[0.05_f64, 0.05_f64],
//!     "filter"    => &["r", "r"],
//!     "mjd_tt"    => &[60000.0_f64, 60000.03_f64],
//! }?;
//!
//! let dataset = ObsDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(1000))?;
//! for obs in dataset.iter_observations() {
//!     println!("{} {:?}", obs.id, obs.equ_coord);
//! }
//! ```
//!
//! ## Use an MPC observatory code
//!
//! Add an optional `mpc_code_obs` column (`String`, nullable) to associate each
//! observation with an MPC-registered observatory.  The accuracy values for MPC
//! sites are derived from the chosen `ObsErrorModel`.
//!
//! ```rust,ignore
//! use polars::prelude::*;
//! use photom::observation::ObsDataset;
//! use photom::observer::error_model::ObsErrorModel;
//!
//! let df = df! {
//!     "id"           => &[1_u64],
//!     "ra"           => &[1.4633_f64],          // radians
//!     "ra_err"       => &[1.745e-5_f64],        // radians
//!     "dec"          => &[0.3840_f64],           // radians
//!     "dec_err"      => &[1.745e-5_f64],        // radians
//!     "magnitude"    => &[19.3_f64],
//!     "mag_err"      => &[0.05_f64],
//!     "filter"       => &["r"],
//!     "mjd_tt"       => &[60000.0_f64],
//!     "mpc_code_obs" => &[Some("F51")],   // Haleakalā Pan-STARRS 1
//! }?;
//!
//! let dataset = ObsDataset::from_polars(&df, ObsErrorModel::FCCT14, None)?;
//! ```
//!
//! ## Group observations by trajectory
//!
//! ```rust,ignore
//! use polars::prelude::*;
//! use photom::trajectory::{TrajDataset, TrajId};
//! use photom::observer::error_model::ObsErrorModel;
//!
//! // traj_id can be UInt32 or String; null rows are loaded but not grouped.
//! let df = df! {
//!     "id"        => &[1_u64, 2_u64, 3_u64],
//!     "ra"        => &[1.4633_f64, 1.4682_f64, 0.1745_f64],  // radians
//!     "ra_err"    => &[1.745e-5_f64; 3],                      // radians
//!     "dec"       => &[0.3840_f64, 0.3847_f64, 0.0873_f64],  // radians
//!     "dec_err"   => &[1.745e-5_f64; 3],                      // radians
//!     "magnitude" => &[19.3_f64, 19.5_f64, 18.0_f64],
//!     "mag_err"   => &[0.05_f64; 3],
//!     "filter"    => &["r", "r", "g"],
//!     "mjd_tt"    => &[60000.0_f64, 60000.03_f64, 60001.0_f64],
//!     "traj_id"   => &[Some("2020 AV2"), Some("2020 AV2"), None],
//! }?;
//!
//! let mut dataset = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(1000))?;
//! if let Some(traj) = dataset.get_trajectory(&TrajId::Str("2020 AV2".to_string())) {
//!     println!("{} observations in trajectory", traj.obs_ids.len());
//! }
//! ```
//!
//! ## Load observations from a `LazyFrame`
//!
//! ```rust,ignore
//! use photom::observation::ObsDataset;
//! use photom::observer::error_model::ObsErrorModel;
//!
//! // Any DataFrame can be turned into a LazyFrame with .lazy().
//! let dataset = ObsDataset::from_lazy(df.lazy(), ObsErrorModel::VFCC17, None)?;
//! ```
//!
//! ## Compute angular separation between two sky positions
//!
//! ```rust
//! use photom::astrometry::EquCoord;
//!
//! let a = EquCoord::from_degrees(10.0, 0.001, 20.0, 0.001);
//! let b = EquCoord::from_degrees(10.5, 0.001, 20.5, 0.001);
//! let sep = a.angular_separation(&b); // result in radians
//! ```
//!
//! ## Parallel iteration
//!
//! *Requires the `parallel` feature.*
//!
//! When the `parallel` feature is enabled, [`observation_dataset::ObsDataset`] gains a
//! family of `par_iter_*` methods that return
//! [`rayon::iter::ParallelIterator`](https://docs.rs/rayon/latest/rayon/iter/trait.ParallelIterator.html)
//! values instead of standard iterators.  These methods take `&self` and can be called
//! while other shared borrows of the dataset are live.
//!
//! ```rust,ignore
//! use photom::observation_dataset::ObsDataset;
//! use rayon::iter::ParallelIterator;
//!
//! // Iterate over every observation in parallel.
//! let count = dataset.par_iter_observations().count();
//!
//! // Iterate over every (NightId, &Observation) pair in parallel.
//! // Returns None if the dataset was built without a night_id column.
//! if let Some(par_iter) = dataset.par_iter_full_night() {
//!     par_iter.for_each(|(night_id, obs)| {
//!         println!("night {:?}: obs id {}", night_id, obs.id());
//!     });
//! }
//! ```
//!
//! # The `polars` Feature
//!
//! Polars-based ingestion is gated behind the optional `polars` feature.  To enable it,
//! add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! photom = { version = "0.1", features = ["polars"] }
//! ```
//!
//! Without this feature the crate is still fully usable: all types, constants, and
//! astrometric utilities are available; only the `from_polars` and `from_lazy`
//! constructors on [`observation_dataset::ObsDataset`] are absent.
//!
//! # The `parallel` Feature
//!
//! Parallel iteration is gated behind the optional `parallel` feature, which brings in
//! [rayon](https://docs.rs/rayon) as a dependency.  To enable it, add the following to
//! your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! photom = { version = "0.1", features = ["parallel"] }
//! ```
//!
//! The `parallel` feature can be combined freely with the `polars` feature:
//!
//! ```toml
//! photom = { version = "0.1", features = ["polars", "parallel"] }
//! ```
//!
//! When enabled, every method documented in
//! [`observation_dataset::parallel`](observation_dataset) becomes available on
//! [`observation_dataset::ObsDataset`].  All parallel methods take `&self`, so they do
//! not conflict with outstanding shared borrows of the dataset.
//!
//! # The `ades` Feature
//!
//! ADES ingestion is gated behind the optional `ades` feature.  To enable it, add the
//! following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! photom = { version = "0.1", features = ["ades"] }
//! ```
//!
//! Without this feature the crate is still fully usable: all types, constants, and
//! astrometric utilities are available; only the `from_ades` constructor on
//! [`observation_dataset::ObsDataset`] is absent.
//!
//! The `ades` feature can be combined freely with the `polars` and `parallel` features:
//!
//! ```toml
//! photom = { version = "0.1", features = ["polars", "parallel", "ades"] }
//! ```
//!
//! ## Loading an ADES file
//!
//! ```rust,ignore
//! use photom::observation_dataset::ObsDataset;
//!
//! // error_ra and error_dec are optional fallback uncertainties in arcseconds,
//! // used when the XML record does not supply rmsRA/rmsDec or precRA/precDec.
//! let dataset = ObsDataset::from_ades("observations.xml", Some(0.5), Some(0.5))?;
//! ```
//!
//! ## Uncertainty resolution (per observation, in precedence order)
//!
//! 1. `rmsRA` / `rmsDec` present in the XML record → used directly (arcseconds).
//! 2. `precRA` / `precDec` present → used as the uncertainty (arcseconds).
//! 3. Fallback `error_ra` / `error_dec` arguments → applied uniformly when neither
//!    of the above fields is available.
//!
//! ## Observer representation
//!
//! Each observation's `stn` field is stored as an
//! [`observer::dataset::ObserverId::MpcCode`] and resolved lazily from the MPC
//! observatory list the first time accuracy values are requested.
//!
//! # The `datafusion` Feature
//!
//! Parquet ingestion via Apache Arrow and DataFusion is gated behind the optional
//! `datafusion` feature.  When enabled it brings in the `datafusion` and `object_store`
//! crates and exposes both an async entry-point
//! (`ObsDataset::from_parquet_uri`) and a synchronous blocking wrapper on the
//! same function, so callers without an async runtime can still use the loader.
//! To enable it, add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! photom = { version = "0.1", features = ["datafusion"] }
//! ```
//!
//! Without this feature the crate is still fully usable: all types, constants, and
//! astrometric utilities remain available; only the `from_parquet_uri` constructor on
//! [`observation_dataset::ObsDataset`] is absent.
//!
//! The `datafusion` feature can be combined freely with the other optional features:
//!
//! ```toml
//! photom = { version = "0.1", features = ["polars", "parallel", "ades", "datafusion"] }
//! ```
//!
//! ## Parquet URI scheme
//!
//! The loader accepts any URI that `object_store` can resolve.  The following schemes are
//! supported out of the box:
//!
//! | Scheme | Backend |
//! |--------|---------|
//! | `file://` | Local filesystem |
//! | `http://` | Plain HTTP object store |
//! | `https://` | TLS-encrypted HTTP object store |
//! | `hdfs://` | Hadoop Distributed File System (requires the `hdfs` Cargo feature on `object_store`) |
//!
//! ## Parquet column schema
//!
//! The Parquet file must contain the following Arrow-typed columns.  Column names and
//! Arrow types must match exactly; the loader returns an error for any schema mismatch.
//!
//! ### Mandatory base columns (non-nullable)
//!
//! | Column | Arrow type | Description |
//! |--------|------------|-------------|
//! | `id` | `UInt64` | Unique observation identifier |
//! | `ra` | `Float64` | Right ascension (radians) |
//! | `ra_err` | `Float64` | Right ascension uncertainty (radians) |
//! | `dec` | `Float64` | Declination (radians) |
//! | `dec_err` | `Float64` | Declination uncertainty (radians) |
//! | `magnitude` | `Float64` | Apparent magnitude |
//! | `mag_err` | `Float64` | Magnitude uncertainty |
//! | `filter` | `Utf8`, `UInt8`, `UInt16`, or `UInt32` | Photometric filter label or code |
//! | `mjd_tt` | `Float64` | Epoch (MJD, Terrestrial Time) |
//!
//! ### Optional observer columns (nullable; column may be absent)
//!
//! | Column | Arrow type | Description |
//! |--------|------------|-------------|
//! | `obs_lon` | `Float64` | Geodetic longitude (radians, east positive) |
//! | `obs_lat` | `Float64` | Geodetic latitude (radians) |
//! | `obs_alt` | `Float64` | Altitude above ellipsoid (metres) |
//! | `obs_ra_acc` | `Float64` | RA accuracy (radians) — required when the geodetic triplet is set |
//! | `obs_dec_acc` | `Float64` | Dec accuracy (radians) — required when the geodetic triplet is set |
//! | `mpc_code_obs` | `Utf8` | Three-byte ASCII MPC code (takes precedence over geodetic columns) |
//!
//! ### Optional index columns
//!
//! | Column | Arrow type | Description |
//! |--------|------------|-------------|
//! | `night_id` | `UInt32` | Night identifier; nullable — null rows are included but not assigned to any night |
//! | `traj_id` | `UInt32` or `Utf8` | Trajectory identifier; nullable — null rows are loaded but not assigned to any trajectory |
//!
//! ## Loading a Parquet file
//!
//! ```rust,ignore
//! use photom::observation_dataset::ObsDataset;
//! use photom::io::datafusion::LoadObsArgs;
//!
//! let dataset = ObsDataset::from_parquet_uri(
//!     "file:///data/observations.parquet",
//!     LoadObsArgs::default(),
//! )?;
//! ```
//!
//! ## Ingestion arguments (`LoadObsArgs`)
//!
//! Both the async and the blocking variants of `from_parquet_uri` accept a `LoadObsArgs`
//! value that controls how the ingestion pipeline behaves.  Use `LoadObsArgs::default()`
//! to get sensible out-of-the-box settings, or construct the struct explicitly to override
//! individual fields.
//!
//! | Field | Type | Default | Description |
//! |-------|------|---------|-------------|
//! | `error_model` | `Option<ObsErrorModel>` | `None` | Astrometric error model used to assign accuracies to MPC-coded observatories; `None` leaves MPC observer accuracies unset until [`ObsDataset::set_error_model`](observation_dataset::ObsDataset::set_error_model) is called |
//! | `lru_cache_size` | `Option<usize>` | `Some(10_000)` | Capacity of the per-dataset LRU cache for observation look-up by `ObsId`; `None` falls back to an internal default |
//! | `contiguous_choice` | `Option<ContiguousChoice>` | `Some(ContiguousNight)` | Which grouping column (if any) to sort the query by before collecting; sorting allows the corresponding index to use compact contiguous ranges instead of per-row index vectors (see below) |
//!
//! The `contiguous_choice` field (defaulting to `ContiguousNight`) causes DataFusion to
//! append an `ORDER BY` clause to the internal SQL query before collecting the record
//! batches.  As a result, all observations belonging to the same night occupy a contiguous
//! block in the output `observations` vector, enabling the night index to store a compact
//! `(start, end)` range instead of a `Vec` of scattered positions.  This is the same
//! contiguous index optimisation applied by the Polars loader via `FromPolarsArgs`.
//!
//! # Minimum Supported Rust Version
//!
//! `photom` requires **Rust 1.94.0** or later.

pub mod astrometry;
pub mod constants;
pub mod io;
pub mod observation_dataset;
pub mod observer;
pub mod photometry;

#[cfg(feature = "mpc_80_col")]
pub use io::mpc_80_col::Mpc80ColError;

#[cfg(feature = "ades")]
pub use io::ades::AdesError;

#[cfg(feature = "serde")]
pub use io::serde::{IndexLayout, ObsDatasetSeed};

pub use observation_dataset::builder::{LoadWarning, ObsDatasetBuilder};

/// Arcseconds.
pub type Arcseconds = f64;
/// Radians.
pub type Radians = f64;
/// Degrees.
pub type Degrees = f64;
/// Modified Julian Date (Terrestrial Time).
pub type MJDTT = f64;
/// Meters.
pub type Meters = f64;

/// Logical identifier for a night of observation.
///
/// Wraps a `u32` that typically represents an integer MJD day number
/// (e.g. `60312`).  The value must be stable across runs because it is used
/// as a directory name in on-disk outputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NightId(pub u32);

// ── TrajId ────────────────────────────────────────────────────────────────────

/// Typed identifier for a single trajectory.
///
/// A trajectory can be identified either by a **32-bit unsigned integer** (e.g.
/// a running index or a catalogue number) or by a **string label** (e.g. a
/// Minor Planet Center provisional designation such as `"2020 AV2"`, or a
/// proper name such as `"Ceres"`).
///
/// The column type of `traj_id` in the source `DataFrame` determines which
/// variant is used: a `UInt32` column produces [`TrajId::Int`] keys and a
/// `String` column produces [`TrajId::Str`] keys.  Mixing both types in a
/// single dataset is not supported; the column must be uniformly one type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrajId {
    /// A 32-bit unsigned integer identifier (e.g. a catalogue number).
    Int(u32),
    /// A string label (e.g. a MPC provisional designation or a proper name).
    Str(String),
}
