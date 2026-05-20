# photom

<p align="center">
  <a href="https://crates.io/crates/photom">
    <img src="https://img.shields.io/crates/v/photom.svg" alt="Crates.io"/>
  </a>
  <a href="https://docs.rs/photom">
    <img src="https://docs.rs/photom/badge.svg" alt="Docs.rs"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-CeCILL--C-blue.svg" alt="License: CeCILL-C"/>
  </a>
  <a href="https://github.com/FusRoman/photom/actions">
    <img src="https://github.com/FusRoman/photom/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
  <a href="https://codecov.io/gh/FusRoman/photom">
    <img src="https://codecov.io/gh/FusRoman/photom/branch/main/graph/badge.svg" alt="codecov"/>
  </a>
  <a href="https://www.rust-lang.org/">
    <img src="https://img.shields.io/badge/MSRV-1.82%2B-orange" alt="MSRV"/>
  </a>
</p>

Rust library for loading, structuring, and querying astronomical observation datasets — with trajectory grouping, multi-observer support, and efficient lookups.

## Features

- **Serialisation / deserialisation** (`serde` feature) — persist an [`ObsDataset`] to JSON (or any other `serde`-compatible format) and restore it without losing observations or custom observers. Runtime-only state (MPC network cache) is automatically re-initialised on deserialisation.
- **Polars ingestion** (`polars` feature) — load observations from a `DataFrame` or `LazyFrame` with full schema validation.
- **Parallel iteration** (`parallel` feature) — iterate over observations, nights, and trajectories in parallel via [rayon](https://docs.rs/rayon), with zero data copying.
- **ADES ingestion** (`ades` feature) — load observations directly from MPC ADES XML files, with automatic MPC observer resolution.
- **MPC 80-column ingestion** (`mpc_80_col` feature) — load observations from the classic MPC fixed-width 80-column ASCII format.
- **Parquet ingestion via DataFusion** (`datafusion` feature) — load observations from any Parquet file reachable by URI (`file://`, `http://`, `https://`, `hdfs://`) using Apache Arrow / DataFusion.
- **Multi-observer support** — MPC observatory codes (resolved lazily from the MPC website), custom geodetic sites (interned and deduplicated), or unknown observer.
- **Trajectory grouping** — group observations by a `traj_id` column; supports both integer (`UInt32`) and string (`String`) identifiers.
- **Three astrometric error models** — FCCT14, CBM10, and VFCC17, used to assign measurement accuracies to MPC-coded observatories.

## Installation

Add `photom` to your `Cargo.toml`. Without any optional features:

```toml
[dependencies]
photom = "0.1"
```

Enable individual features as needed:

```toml
[dependencies]
photom = { version = "0.1", features = ["polars", "parallel", "ades", "mpc_80_col", "datafusion", "serde"] }
```

All features are independent and can be combined freely.

## Quick Start

### Serialise and deserialise a dataset (serde feature)

`ObsDataset` implements the standard `serde::Serialize` / `serde::Deserialize`
traits and works with any serde-compatible format (JSON, MessagePack, …).

```rust
use photom::observation_dataset::ObsDataset;

// Serialise — format-agnostic (use any serde serializer).
let json = serde_json::to_string(&dataset)?;
std::fs::write("dataset.json", &json)?;

// Deserialise with the default index layout (Split — always safe).
let json = std::fs::read_to_string("dataset.json")?;
let restored: ObsDataset = serde_json::from_str(&json)?;

// Binary format (rmp-serde / MessagePack).
let bytes: Vec<u8> = rmp_serde::to_vec(&dataset)?;
let restored: ObsDataset = rmp_serde::from_slice(&bytes)?;
```

#### Choosing the index layout at deserialisation

For potentially faster look-ups you can request a contiguous index layout via
[`ObsDatasetSeed`] (a [`serde::de::DeserializeSeed`] implementation).
Any format that exposes its `Deserializer` struct publicly works — both
`serde_json` and `rmp-serde` do:

```rust
use photom::{IndexLayout, ObsDatasetSeed};
use serde::de::DeserializeSeed as _;

// JSON
let mut de = serde_json::Deserializer::from_str(&json);
let restored = ObsDatasetSeed { layout: IndexLayout::TryContiguous }
    .deserialize(&mut de)?;

// MessagePack (rmp-serde — compact binary)
let mut de = rmp_serde::Deserializer::new(bytes.as_slice());
let restored = ObsDatasetSeed { layout: IndexLayout::TryContiguous }
    .deserialize(&mut de)?;
```

`TryContiguous` falls back to `Split` automatically for any index group whose
observations are not stored contiguously.

**What is persisted**

| State | Persisted? | Notes |
|---|---|---|
| Observations | Yes | Full list in insertion order |
| Custom geodetic observers | Yes | All sites and their coordinates |
| Astrometric error model | Yes | `FCCT14`, `CBM10`, `VFCC17`, or `None` |
| MPC network cache | No | Fetched lazily on first use |
| MPC network cache | No | Fetched lazily on first use |
| Trajectory aliases | Yes | Fully round-tripped |
| Night / trajectory indices | Yes | Membership stored per-observation; rebuilt on load |

### Load observations from a Polars DataFrame

```rust
use photom::observation_dataset::ObsDataset;
use photom::io::polars::{FromPolarsArgs};

let dataset = ObsDataset::from_polars(&df, FromPolarsArgs::default())?;

for obs in dataset.iter_observations() {
    println!("{:?}", obs);
}
```

### Load from a LazyFrame

```rust
use photom::observation_dataset::ObsDataset;
use photom::io::polars::FromPolarsArgs;

let dataset = ObsDataset::from_lazy(df.lazy(), FromPolarsArgs::default())?;
```

### Load from a Parquet file (DataFusion)

```rust
use photom::observation_dataset::ObsDataset;
use photom::io::datafusion::LoadObsArgs;

let dataset = ObsDataset::from_parquet_uri(
    "file:///data/observations.parquet",
    LoadObsArgs::default(),
)?;

println!("{} observations loaded", dataset.observation_count());
```

### Load from an ADES XML file

```rust
use photom::observation_dataset::ObsDataset;

// error_ra and error_dec are optional fallback uncertainties in arcseconds.
let dataset = ObsDataset::from_ades("observations.xml", Some(0.5), Some(0.5))?;
```

### Load from an MPC 80-column file

```rust
use photom::observation_dataset::ObsDataset;

let dataset = ObsDataset::from_mpc_80_col("observations.txt")?;
```

### Parallel iteration

```rust
use photom::observation_dataset::ObsDataset;
use rayon::iter::ParallelIterator;

let count = dataset.par_iter_observations().count();

if let Some(par_iter) = dataset.par_iter_full_night() {
    par_iter.for_each(|(night_id, obs)| {
        println!("night {:?}: obs id {}", night_id, obs.id());
    });
}
```

### Coordinate and astrometric utilities

`EquCoord` bundles a sky position (RA, Dec) with its 1-σ uncertainties.
All values are stored internally in **radians**; use `from_degrees` to supply
degrees.

```rust
use photom::coordinates::equatorial::EquCoord;
use photom::coordinates::cartesian::CartesianCoord;

// Construct from degrees — converted to radians internally.
let a = EquCoord::from_degrees(10.0, 0.001, 20.0, 0.001);
let b = EquCoord::from_degrees(10.5, 0.001, 20.5, 0.001);

// Great-circle separation via the Vincenty formula (result in radians).
let sep = a.angular_separation(&b);

// Vector-averaging midpoint on the sphere.
let mid = a.spherical_midpoint(&b);

// Lossless projection onto the unit sphere (uncertainties discarded).
let cart = CartesianCoord::from(a);
// Recover equatorial angles (errors set to zero).
let back: EquCoord = cart.into();

// Propagate astrometric covariance through the spherical → Cartesian mapping.
// Returns CartesianCoordCov with the full 3×3 covariance matrix.
let cov = a.to_cartesian_cov();
// Inverse: propagate back to equatorial marginal 1-σ errors.
let recovered = cov.to_equatorial();
```

### 2-D covariance on the tangent plane

`Cov2` is a compact symmetric 2×2 covariance matrix for astrometric error
ellipses expressed in a local tangent-plane frame.

```rust
use photom::coordinates::cov2::Cov2;
use photom::coordinates::equatorial::EquCoord;

// Build a diagonal covariance from the marginal errors of an EquCoord.
let coord = EquCoord::from_degrees(45.0, 0.001, 20.0, 0.002);
let cov = Cov2::from_equ(&coord);

// Semi-axes of the 1-σ confidence ellipse.
let sigma_major = cov.lambda_max().max(0.0).sqrt();
let sigma_minor = cov.lambda_min().max(0.0).sqrt();

// Mahalanobis distance for an offset vector (radians).
let offset = [1e-4_f64, 0.0_f64];
if let Some(d2) = cov.mahalanobis_sq(offset) {
    let _ = d2.sqrt(); // normalised distance
}

// Add isotropic process noise q·I (Kalman-style inflation).
let inflated = cov.inflate_isotropic(1e-8);
```

### Gnomonic (tangent-plane) projection

`TangentPlane` projects sky positions near a chosen tangent point onto a local
2-D Cartesian frame. Great circles project to straight lines, making this ideal
for short-arc astrometry and kinematic linking.

```rust
use photom::coordinates::equatorial::EquCoord;
use photom::coordinates::gnomonic_projection::{TangentPlane, TangentVec};

// Define the tangent point (degrees, converted internally to radians).
let ref_coord = EquCoord::from_degrees(45.0, 0.0, 20.0, 0.0);
let plane = TangentPlane::new(ref_coord);

// Forward projection: sky → tangent plane.
let target = EquCoord::from_degrees(45.5, 0.0, 20.5, 0.0);
let tp = plane.project(&target);

// Inverse projection: tangent plane → sky.
let sky = tp.unproject();

// Squared Euclidean distance between two projected points (radians²).
let other = plane.project(&EquCoord::from_degrees(45.1, 0.0, 20.1, 0.0));
let d2 = tp.dist2(&other);

// Translate a projected point by a displacement vector.
let v = TangentVec { dx: 1e-3, dy: -1e-3 };
let shifted = tp + v;
```

## DataFrame / Parquet Schema

All column values for `ra`, `ra_err`, `dec`, `dec_err`, `obs_lon`, `obs_lat`, `obs_ra_acc`, and `obs_dec_acc` must be supplied in **radians**. No unit conversion is performed during ingestion.

### Mandatory base columns (non-nullable)

| Column      | Polars type | Arrow type | Unit      | Description                           |
|-------------|-------------|------------|-----------|---------------------------------------|
| `id`        | `UInt64`    | `UInt64`   | —         | Unique observation identifier         |
| `ra`        | `Float64`   | `Float64`  | rad       | Right ascension                       |
| `ra_err`    | `Float64`   | `Float64`  | rad       | 1-σ right ascension uncertainty       |
| `dec`       | `Float64`   | `Float64`  | rad       | Declination                           |
| `dec_err`   | `Float64`   | `Float64`  | rad       | 1-σ declination uncertainty           |
| `magnitude` | `Float64`   | `Float64`  | mag       | Apparent magnitude                    |
| `mag_err`   | `Float64`   | `Float64`  | mag       | 1-σ magnitude uncertainty             |
| `filter`    | `String`    | `Utf8` / `UInt8` / `UInt16` / `UInt32` | — | Photometric filter label or code |
| `mjd_tt`    | `Float64`   | `Float64`  | MJD (TT)  | Epoch (Modified Julian Date, Terrestrial Time) |

### Optional observer columns (nullable; column may be absent)

| Column         | Polars type | Arrow type | Unit | Description                                                        |
|----------------|-------------|------------|------|--------------------------------------------------------------------|
| `obs_lon`      | `Float64`   | `Float64`  | rad  | Geodetic longitude, east of Greenwich                              |
| `obs_lat`      | `Float64`   | `Float64`  | rad  | Geodetic latitude                                                  |
| `obs_alt`      | `Float64`   | `Float64`  | m    | Altitude above the reference ellipsoid                             |
| `obs_ra_acc`   | `Float64`   | `Float64`  | rad  | 1-σ RA measurement accuracy — required when geodetic triplet is set |
| `obs_dec_acc`  | `Float64`   | `Float64`  | rad  | 1-σ Dec measurement accuracy — required when geodetic triplet is set |
| `mpc_code_obs` | `String`    | `Utf8`     | —    | Three-byte ASCII MPC code (takes precedence over geodetic columns)  |

### Optional grouping / index columns

| Column     | Polars type              | Arrow type          | Description                                                                     |
|------------|--------------------------|---------------------|---------------------------------------------------------------------------------|
| `traj_id`  | `UInt32` or `String`     | `UInt32` or `Utf8`  | Trajectory identifier; nullable — null rows are loaded but not assigned to any trajectory |
| `night_id` | `UInt32`                 | `UInt32`            | Night identifier; nullable — null rows are included but not assigned to any night |

## Observer Resolution

Each row's observer is resolved in the following order of precedence:

1. `mpc_code_obs` non-null → `ObserverId::MpcCode` (MPC site, resolved lazily from the MPC website).
2. `obs_lon`, `obs_lat`, and `obs_alt` all non-null → `ObserverId::IntId` (custom geodetic site). `obs_ra_acc` and `obs_dec_acc` must also be non-null.
3. Otherwise → no observer (`None`).

A partially-null geodetic triplet (one or two of the three columns non-null) is always an ingestion error. A complete triplet without accuracy values is also an error.

## Ingestion Arguments

### `FromPolarsArgs` (Polars feature)

| Field               | Type                      | Default              | Description                                                        |
|---------------------|---------------------------|----------------------|--------------------------------------------------------------------|
| `error_model`       | `Option<ObsErrorModel>`   | `None`               | Astrometric error model for MPC-coded observatories                |
| `do_rechunk`        | `Option<bool>`            | `Some(false)`        | Force single-chunk layout before ingestion                         |
| `contiguous_choice` | `Option<ContiguousChoice>`| `Some(ContiguousNight)` | Sort by night or trajectory for compact index ranges            |

### `LoadObsArgs` (DataFusion feature)

| Field               | Type                      | Default              | Description                                                        |
|---------------------|---------------------------|----------------------|--------------------------------------------------------------------|
| `error_model`       | `Option<ObsErrorModel>`   | `None`               | Astrometric error model for MPC-coded observatories                |
| `contiguous_choice` | `Option<ContiguousChoice>`| `Some(ContiguousNight)` | Sort by night or trajectory for compact index ranges            |

## Type Aliases

| Alias        | Underlying type | Unit                              |
|--------------|-----------------|-----------------------------------|
| `Arcseconds` | `f64`           | Angle in arcseconds               |
| `Radians`    | `f64`           | Angle in radians                  |
| `Degrees`    | `f64`           | Angle in degrees                  |
| `MJDTT`      | `f64`           | Modified Julian Date (Terrestrial Time) |
| `Meters`     | `f64`           | Distance in metres                |

## Error Types

| Error type      | Feature     | Description                                                                 |
|-----------------|-------------|-----------------------------------------------------------------------------|
| `PolarsError`   | `polars`    | Schema validation, type mismatch, null in required column, partial geodetic triplet, missing accuracy, invalid MPC code |
| `LoadObsError`  | `datafusion`| URI resolution failure, resource not found, DataFusion I/O error, Arrow column error |
| `AdesError`     | `ades`      | XML parse error, missing mandatory field, unresolvable observatory          |
| `Mpc80ColError` | `mpc_80_col`| Parse error in the fixed-width 80-column format                             |
| `ObserverError` | —           | Invalid float value, MPC code not found or malformed                        |

## Documentation

To compile the documentation locally, run the following command in the terminal:
```bash
RUSTDOCFLAGS="--html-in-header $(pwd)/katex-header.html" cargo doc --no-deps --all-features
```

## Testing Notes

The DataFusion tests require the large-test-fixtures feature to run. The large Parquet fixtures have been excluded from the crates.io package and are gated behind this feature.

To run the full test suite including DataFusion:

```bash
cargo test --features "datafusion,large-test-fixtures"
```

All other tests are gated behind their associated features and do not require this additional flag.

## Minimum Supported Rust Version

`photom` requires **Rust 1.94.0** or later.

## License

This project is licensed under the [CeCILL-C Free Software License Agreement](LICENSE).
