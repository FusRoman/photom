# photom

Rust library for loading, structuring, and querying astronomical observation datasets — with trajectory grouping, multi-observer support, and efficient lookups.

## Features

- **Serialisation / deserialisation** (`serde` feature) — persist an [`ObsDataset`] to JSON (or any other `serde`-compatible format) and restore it without losing observations, custom observers, or LRU cache capacity. Runtime-only state (LRU contents, MPC network cache) is automatically re-initialised on deserialisation.
- **Polars ingestion** (`polars` feature) — load observations from a `DataFrame` or `LazyFrame` with full schema validation.
- **Parallel iteration** (`parallel` feature) — iterate over observations, nights, and trajectories in parallel via [rayon](https://docs.rs/rayon), with zero data copying.
- **ADES ingestion** (`ades` feature) — load observations directly from MPC ADES XML files, with automatic MPC observer resolution.
- **MPC 80-column ingestion** (`mpc_80_col` feature) — load observations from the classic MPC fixed-width 80-column ASCII format.
- **Parquet ingestion via DataFusion** (`datafusion` feature) — load observations from any Parquet file reachable by URI (`file://`, `http://`, `https://`, `hdfs://`) using Apache Arrow / DataFusion.
- **Multi-observer support** — MPC observatory codes (resolved lazily from the MPC website), custom geodetic sites (interned and deduplicated), or unknown observer.
- **Trajectory grouping** — group observations by a `traj_id` column; supports both integer (`UInt32`) and string (`String`) identifiers.
- **Three astrometric error models** — FCCT14, CBM10, and VFCC17, used to assign measurement accuracies to MPC-coded observatories.
- **LRU caches** — configurable cache capacity for fast repeated lookups of observations and trajectories.

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
| LRU cache capacity | Yes | Preserves eviction behaviour |
| LRU cache contents | No | Repopulated on access |
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

### Astrometric utilities

```rust
use photom::astrometry::EquCoord;

// from_degrees accepts values in degrees and converts internally to radians.
let a = EquCoord::from_degrees(10.0, 0.001, 20.0, 0.001);
let b = EquCoord::from_degrees(10.5, 0.001, 20.5, 0.001);
let sep = a.angular_separation(&b); // result in radians
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
| `lru_cache_size`    | `Option<usize>`           | `Some(10_000)`       | LRU cache capacity for observation lookup by `ObsId`               |
| `do_rechunk`        | `Option<bool>`            | `Some(false)`        | Force single-chunk layout before ingestion                         |
| `contiguous_choice` | `Option<ContiguousChoice>`| `Some(ContiguousNight)` | Sort by night or trajectory for compact index ranges            |

### `LoadObsArgs` (DataFusion feature)

| Field               | Type                      | Default              | Description                                                        |
|---------------------|---------------------------|----------------------|--------------------------------------------------------------------|
| `error_model`       | `Option<ObsErrorModel>`   | `None`               | Astrometric error model for MPC-coded observatories                |
| `lru_cache_size`    | `Option<usize>`           | `Some(10_000)`       | LRU cache capacity for observation lookup by `ObsId`               |
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

## Minimum Supported Rust Version

`photom` requires **Rust 1.94.0** or later.

## License

This project is licensed under the [CeCILL-C Free Software License Agreement](LICENSE).
