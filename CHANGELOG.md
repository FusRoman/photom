# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-06

### Added
- `lsst_alert` feature (no extra dependency): module `io::lsst_alert`, a pure conversion of Rubin/LSST `DiaSource` alert fields into an `ObservationInput` ready for `ObsDataset::push_observation`.
  - `DiaSourceRecord` (APDB field names and units: deg, nJy, MJD TAI), `LsstConversionParams` (on-sky astrometric floor added in quadrature, `0` by default) and `LsstObservation` (the input plus its `NightId` and Rubin `visit`).
  - `dia_source_to_observation`: `psfFlux` → AB magnitude (zero point 31.4), first-order magnitude error, TAI → TT (+32.184 s), band `u g r i z y` → `Filter::Int(0..=5)`, observer `X05`, and on-sky `raErr` (σ_α·cos δ) → error on the α coordinate (`/ cos δ`), consistent with `EquCoord::ra_error`.
  - Night key `NightId` = MJD of the `dayObs` date encoded in `visit` (`dayObs × 100000 + seqNum`), cross-checked against `⌊midpointMjdTai − 0.5⌋`.
  - `LsstAlertError` for unconvertible sources only; science selection is a separate optional policy, `LsstQualityCuts` (strict SNR = `psfFlux / psfFluxErr`, `raErr` / `decErr` ceilings), which cuts nothing by default.
  - Reusable pure building blocks in `io::lsst_alert::units` and `io::lsst_alert::night` (integer civil-date → MJD conversion).
  - Example `14_lsst_alert`.

### Fixed
- `ObsDataset::push_observation` now rejects ids repeated **within** the pushed batch with `ObsDatasetError::DuplicateObsIds`. Previously both copies were stored while the id index silently kept only the last one.
- `ObsDataset::push_observation` no longer copies the `IntId` observers referenced by the batch and no longer shifts their ids. An `IntId` designates an observer already registered in the dataset (e.g. through `push_observer`). The previous behaviour duplicated the observer and could produce an out-of-bounds `IntId`. `merge_from` is unchanged.
- Documentation of the Polars base columns (`io::polars::base_field`): `ra`, `ra_err`, `dec` and `dec_err` are expected in **radians** (they were documented in degrees; the ingestion performs no conversion, as stated in the crate-level schema).
- The 1.0.0 changelog entry wrongly listed an `Observation::new` constructor; `Observation` has no public constructor and observations are built through `ObservationInput::new`.

## [1.0.0] - 2026-09-16

First major release of `photom`, a Rust library for loading, structuring, and querying astronomical observation datasets, with trajectory grouping, multi-observer support, and efficient lookups. This release consolidates all development since the project's inception.

### Added

#### Core data model
- `ObsDataset`, the central structure for storing and indexing astronomical observations, with a `new` constructor, an empty constructor, and a `push` method to append individual observations.
- `Observation` type (built through `ObservationInput::new` and placed by `ObsDataset`), with ordering (`PartialEq`, `PartialOrd`, `Ord`), and display support.
- Night grouping: observations can be grouped by observation night.
- Trajectory grouping: observations can be grouped by a `traj_id` column, supporting both `UInt32` and `String` identifiers. `TrajId` supports conversion from common integer/string types and a stable hashing method.
- Multi-observer support via the `Observer` type and `ObserverId`, covering three kinds of observer: MPC observatory codes (resolved lazily and cached from the MPC website), custom geodetic sites (interned and deduplicated), and an "unknown observer" fallback.
- `ObserverDataset`, an indexed collection of observers with an `empty` constructor and `get` lookup by `ObserverId`.
- Geodetic/parallax constructors and conversions for observers: `new`, `from_parallax`, `geodetic_lat_height_wgs84`, `geocentric_lat_deg`, `geocentric_distance_earth_radii`, and a standalone `geodetic_to_parallax` helper (WGS84-based), with dedicated parallax tests.
- Display formatting for `Observer` and `ObservId`, and a parallel iterator over the observers contained in an `ObsDataset` (`parallel` feature).
- Observer-related tests migrated from the `outfit` crate into `photom` and expanded with dataset-level integration tests.
- Configurable in-memory index layouts for nights and trajectories, with memory-usage optimizations and consistency tests. Index construction was hardened for robustness.
- `merge_from`, a public API to merge two `ObsDataset` instances, with unique-`ObsId` guarantees across merges and dedicated integration tests.
- Iterators over observations, observers, nights, and trajectories, with a `parallel` feature (via [rayon](https://docs.rs/rayon)) providing zero-copy parallel iteration, later upgraded to indexed parallel iterators.
- A multi-file append API and a file builder for constructing an `ObsDataset` from several input files.

#### Coordinates and linear algebra
- Cartesian and spherical coordinate types, with conversions between them and a spherical mid-point calculation.
- Gnomonic projection support, including covariance propagation.
- Ecliptic coordinates.
- Supporting linear algebra structures and methods for covariance computation (including a Cholesky decomposition with property-based tests).

#### Astrometric error models
- Three astrometric error models — FCCT14, CBM10, and VFCC17 — used to assign measurement accuracies to MPC-coded observatories.
- A `ModelCorrection` trait to apply astrometric error corrections to a dataset (primarily for MPC 80-column files, but usable on datasets loaded from any supported format).
- Batch RMS correction, later fixed to correct the RMS per observer per time rather than only per time, and to correctly handle duplicate observations in the input data.
- Dedicated error-model rules for the LSST survey.

#### File format ingestion
- MPC 80-column fixed-width ASCII reader (`mpc_80_col` feature).
- ADES XML reader (`ades` feature), with automatic MPC observer resolution.
- Polars `DataFrame` / `LazyFrame` ingestion (`polars` feature), with full schema validation, support for `u8`/`u16`/`u32` filter columns, and reader performance optimizations (including optional rechunking).
- Parquet ingestion via Apache Arrow / DataFusion (`datafusion` feature), supporting any Parquet file reachable by URI (`file://`, `http://`, `https://`, `hdfs://`).
- Local caching of the MPC observer list fetched from the MPC website.

#### Serialization
- `serde` support (`serde` feature) to serialize and deserialize an `ObsDataset` to/from any `serde`-compatible format (e.g. JSON), preserving observations and custom observers. Runtime-only state (the MPC network cache) is automatically re-initialized on deserialization.

#### Examples
- Thirteen runnable, self-contained examples under `examples/`, each focused on a single use case: building an `ObsDataset` by hand, MPC 80-column and ADES ingestion, night/trajectory grouping and alias resolution, dataset merging, custom `Observer` construction, lazy MPC observer resolution, equatorial/Cartesian and ecliptic/gnomonic coordinate handling, the astrometric error-correction pipeline (`apply_model_errors`, `apply_batch_rms_correction`), loading observations from an in-memory Polars `DataFrame`/`LazyFrame` (`from_polars`/`from_lazy`, mixed MPC-coded/custom observer resolution, automatic night indexing), parallel iteration over observations/trajectories via the `par_iter_*` accessors, and `ObsDataset` (de)serialization with `serde_json` including the `ObsDatasetSeed` contiguous-layout option. Indexed in the README under a new "Examples" section.

#### Project infrastructure
- Continuous integration covering the full matrix of optional feature combinations, plus dedicated jobs for documentation builds and semver checks.
- A dedicated CI job builds every example and runs each of them, verifying they compile and execute successfully against their fixture data (the MPC-network-dependent example is allowed to fail without breaking the build).
- Code coverage reporting via Codecov, running on `main` after every merge.
- Pre-commit hooks (via husky), running across all feature combinations.
- Project license (CeCILL-C) and crate metadata (keywords, categories, repository/homepage/documentation links).
- README with badges (crates.io, docs.rs, license, CI, Codecov, MSRV) and usage documentation.
- Unit and integration tests across observations, MPC files, Polars readers, ADES/MPC ingestion, dataset merging, observers, and index consistency.

### Changed
- Error handling switched from panics to proper `Result`-based error propagation throughout the crate.
- Several method signatures updated to a consistent Rust ownership pattern, taking `self` by value and returning `Self`.
- Methods accepting a trajectory identifier now accept `impl Into<TrajId>` for a more ergonomic API; `ObsDataset` was made `Clone`.
- Optional dependencies gated behind their respective Cargo features, keeping the default build lean.

### Fixed
- Batch RMS correction now correctly accounts for duplicate observations present in the input data.
- Astrometric precision units corrected during MPC 80-column file parsing.
- Longitude type for custom observers corrected.
- Various documentation and `clippy`/`rustfmt` issues across feature combinations.
- `ethnum` dependency bumped from 1.5.2 to 1.5.3 to fix a compilation error (E0512) on recent `rustc` versions.
