# photom

Rust library for loading, structuring and querying astronomical observation datasets — with trajectory grouping, multi-observer support, and efficient lookups.

## Features

- Ingestion from a Polars `DataFrame` or `LazyFrame` (via the optional `polars` feature)
- Multi-observer support: MPC observatory codes (lazy-fetched from the MPC website), custom geodetic sites (interned), or unknown
- Trajectory grouping by a `traj_id` column (integer or string)
- Astrometric error models: FCCT14, CBM10, VFCC17
- LRU caches for fast repeated lookups of observations and trajectories

## Installation

Add `photom` to your `Cargo.toml`. Without any optional features:

```toml
[dependencies]
photom = "0.1"
```

To enable Polars-based ingestion (`from_polars`, `from_lazy`):

```toml
[dependencies]
photom = { version = "0.1", features = ["polars"] }
```

The `polars` feature depends on Polars with the `lazy` feature enabled.

## Quick Start

### Load observations from a Polars DataFrame

```rust
use photom::observation::ObsDataset;
use photom::observer::error_model::ObsErrorModel;

let dataset = ObsDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(1000))?;

for obs in dataset.iter_observations() {
    println!("{:?}", obs);
}
```

### Load from a LazyFrame

```rust
let dataset = ObsDataset::from_lazy(df.lazy(), ObsErrorModel::VFCC17, None)?;
```

### Load and query trajectories

```rust
use photom::trajectory::{TrajDataset, TrajId};

let mut dataset = TrajDataset::from_polars(&df, ObsErrorModel::FCCT14, Some(1000))?;

if let Some(traj) = dataset.get_trajectory(&TrajId::Str("2020 AV2".to_string())) {
    println!("Trajectory {} has {} observations", traj.id, traj.obs_ids.len());
}

for traj in dataset.iter_trajectories() {
    println!("{:?}", traj);
}
```

### Astrometric utilities

```rust
use photom::astrometry::EquCoord;

let a = EquCoord::from_degrees(10.0, 0.001, 20.0, 0.001);
let b = EquCoord::from_degrees(10.5, 0.001, 20.5, 0.001);
let sep = a.angular_separation(&b); // result in radians
```

## DataFrame Schema

The following describes the expected column layout when using the `polars` feature.

### Mandatory base columns (non-nullable)

| Column      | Type      | Description                              |
|-------------|-----------|------------------------------------------|
| `id`        | `UInt64`  | Unique observation identifier            |
| `ra`        | `Float64` | Right ascension (degrees)                |
| `ra_err`    | `Float64` | Right ascension uncertainty (degrees)    |
| `dec`       | `Float64` | Declination (degrees)                    |
| `dec_err`   | `Float64` | Declination uncertainty (degrees)        |
| `magnitude` | `Float64` | Apparent magnitude                       |
| `mag_err`   | `Float64` | Magnitude uncertainty                    |
| `filter`    | `String`  | Photometric filter label                 |
| `mjd_tt`    | `Float64` | Epoch (MJD, Terrestrial Time)            |

### Optional observer columns (nullable; column may be absent)

| Column          | Type      | Description                                                         |
|-----------------|-----------|---------------------------------------------------------------------|
| `obs_lon`       | `Float64` | Geodetic longitude (degrees east)                                   |
| `obs_lat`       | `Float64` | Geodetic latitude (degrees)                                         |
| `obs_alt`       | `Float64` | Altitude above ellipsoid (metres)                                   |
| `obs_ra_acc`    | `Float64` | RA accuracy (radians) — required when geodetic triplet is set       |
| `obs_dec_acc`   | `Float64` | Dec accuracy (radians) — required when geodetic triplet is set      |
| `mpc_code_obs`  | `String`  | Three-byte ASCII MPC code (takes precedence over geodetic columns)  |

### Optional trajectory column

| Column    | Type                  | Description                                                              |
|-----------|-----------------------|--------------------------------------------------------------------------|
| `traj_id` | `UInt64` or `String`  | Trajectory identifier; nullable (null rows are loaded but have no trajectory) |

## Observer Resolution

Each row's observer is resolved in the following order of precedence:

1. If `mpc_code_obs` is non-null, the observer is identified as `ObserverId::MpcCode`. This takes precedence over any geodetic columns. The MPC site list is fetched lazily from the MPC website on the first lookup.
2. Otherwise, if `obs_lon`, `obs_lat`, and `obs_alt` are all non-null, the observer is identified as `ObserverId::IntId` (a geodetic site). The columns `obs_ra_acc` and `obs_dec_acc` must also be non-null in this case.
3. Otherwise, no observer is associated with the row (`None`).

**Error conditions:**

- A partial geodetic triplet (one or two of `obs_lon`, `obs_lat`, `obs_alt` non-null) produces a `PolarsError::PartialTripletNull`.
- A complete geodetic triplet without accuracy values produces a `PolarsError::MissingAccuracyForGeodesic`.

## Error Handling

The crate defines three main error types:

- **`PolarsError`** — errors arising during ingestion from a DataFrame or LazyFrame: `SchemaValidationError`, `ColumnTypeError`, `MissingColumnError`, `DataConversionError`, `PartialTripletNull`, `MissingAccuracyForGeodesic`, `InvalidMpcCode`, `TrajIdColumnTypeError`, and a wrapper around `polars::error::PolarsError`.
- **`ObsDatasetError`** — errors arising during dataset operations: `MPCError`, `ErrorModelError`, `PolarIoError`.
- **`ObserverError`** — errors arising during observer resolution: `InvalidFloatValue`, `MpcCodeNotFound`, `InvalidMpcCode`, `MissingMpcCode`.

## Minimum Supported Rust Version

`photom` requires **Rust 1.94.0** or later.

## License

This project is licensed under the [CeCILL-C Free Software License Agreement](LICENSE).
