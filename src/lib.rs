//! Rust library for loading, structuring, and querying astronomical observation datasets —
//! with trajectory grouping, multi-observer support, and efficient lookups.
//!
//! `photom` provides a type-safe pipeline for ingesting astrometric and photometric
//! measurements, associating them with ground-based observatories, and grouping them into
//! trajectories of moving objects.  The library is designed around two primary dataset
//! types — [`observation::ObsDataset`] for flat observation collections and
//! [`trajectory::TrajDataset`] for trajectory-grouped datasets — with LRU caches providing
//! fast repeated lookups in both cases.
//!
//! # Features
//!
//! - **Polars ingestion** (`polars` feature) — load observations from a `DataFrame` or
//!   `LazyFrame` with full schema validation.
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
//! | [`observation`] | Core observation types ([`observation::Observation`], [`observation::ObsDataset`], [`observation::ObserverId`]) |
//! | [`trajectory`] | Trajectory grouping types ([`trajectory::Trajectory`], [`trajectory::TrajDataset`], [`trajectory::TrajId`]) |
//! | [`observer`] | Ground-based observatory representation ([`observer::Observer`]) and geodetic utilities |
//! | [`observer::error_model`] | Astrometric error model variants ([`observer::error_model::ObsErrorModel`]: FCCT14, CBM10, VFCC17) |
//! | [`constants`] | Physical and geodetic constants (Earth axes, AU, etc.) |
//! | [`io`] | Internal ingestion backends (Polars adapter, schema validation) |
//!
//! # Type Aliases
//!
//! The crate exports four primitive type aliases used throughout the API to make units
//! explicit in function signatures:
//!
//! | Alias | Underlying type | Unit |
//! |-------|-----------------|------|
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
//! | `ra` | `Float64` | Right ascension (degrees) |
//! | `ra_err` | `Float64` | Right ascension uncertainty (degrees) |
//! | `dec` | `Float64` | Declination (degrees) |
//! | `dec_err` | `Float64` | Declination uncertainty (degrees) |
//! | `magnitude` | `Float64` | Apparent magnitude |
//! | `mag_err` | `Float64` | Magnitude uncertainty |
//! | `filter` | `String` | Photometric filter label |
//! | `mjd_tt` | `Float64` | Epoch (MJD, Terrestrial Time) |
//!
//! ## Optional observer columns (nullable; column may be absent)
//!
//! | Column | Polars type | Description |
//! |--------|-------------|-------------|
//! | `obs_lon` | `Float64` | Geodetic longitude (degrees east) |
//! | `obs_lat` | `Float64` | Geodetic latitude (degrees) |
//! | `obs_alt` | `Float64` | Altitude above ellipsoid (metres) |
//! | `obs_ra_acc` | `Float64` | RA accuracy (radians) — required when the geodetic triplet is set |
//! | `obs_dec_acc` | `Float64` | Dec accuracy (radians) — required when the geodetic triplet is set |
//! | `mpc_code_obs` | `String` | Three-byte ASCII MPC code (takes precedence over geodetic columns) |
//!
//! ## Optional trajectory column
//!
//! | Column | Polars type | Description |
//! |--------|-------------|-------------|
//! | `traj_id` | `UInt64` or `String` | Trajectory identifier; nullable — null rows are loaded into the [`observation::ObsDataset`] but are not assigned to any trajectory |
//!
//! ## Observer resolution (per row, in precedence order)
//!
//! 1. `mpc_code_obs` non-null → [`observation::ObserverId::MpcCode`] (MPC site, resolved lazily).
//! 2. `obs_lon`, `obs_lat`, and `obs_alt` all non-null → [`observation::ObserverId::IntId`] (geodetic
//!    site; `obs_ra_acc` and `obs_dec_acc` must also be non-null).
//! 3. Otherwise → no observer (`None`).
//!
//! A partially-null geodetic triplet or a complete triplet without accuracy values causes
//! the ingestion to return an error.
//!
//! # Usage Examples
//!
//! ## Load observations from a `DataFrame`
//!
//! ```rust,ignore
//! use photom::observation::ObsDataset;
//! use photom::observer::error_model::ObsErrorModel;
//!
//! let dataset = ObsDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(1000))?;
//! for obs in dataset.iter_observations() {
//!     println!("{} {:?}", obs.id, obs.equ_coord);
//! }
//! ```
//!
//! ## Load observations from a `LazyFrame`
//!
//! ```rust,ignore
//! use photom::observation::ObsDataset;
//! use photom::observer::error_model::ObsErrorModel;
//!
//! let dataset = ObsDataset::from_lazy(df.lazy(), ObsErrorModel::VFCC17, None)?;
//! ```
//!
//! ## Load and query trajectories
//!
//! ```rust,ignore
//! use photom::trajectory::{TrajDataset, TrajId};
//! use photom::observer::error_model::ObsErrorModel;
//!
//! let mut dataset = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(1000))?;
//! if let Some(traj) = dataset.get_trajectory(&TrajId::Str("2020 AV2".to_string())) {
//!     println!("{} observations in trajectory", traj.obs_ids.len());
//! }
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
//! constructors on [`observation::ObsDataset`] and [`trajectory::TrajDataset`] are absent.
//!
//! # Minimum Supported Rust Version
//!
//! `photom` requires **Rust 1.94.0** or later.

pub mod astrometry;
pub mod constants;
pub mod io;
pub mod observation;
pub mod observer;
pub mod photometry;
pub mod trajectory;

/// Radians.
pub type Radians = f64;
/// Degrees.
pub type Degrees = f64;
/// Modified Julian Date (Terrestrial Time).
pub type MJDTT = f64;
/// Meters.
pub type Meters = f64;
