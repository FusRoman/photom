# photom — agent instructions

Rust library (edition 2024, MSRV 1.94.0) for loading and querying astronomical observation datasets.
All features are **opt-in** — there are no default features.

## Git

- **Never create a git commit without an explicit user request.**

## Feature flags

| Flag | Key deps added |
|------|----------------|
| `polars` | polars (Parquet/lazy) |
| `parallel` | rayon |
| `ades` | hifitime, quick-xml, serde, camino |
| `mpc_80_col` | hifitime, camino |
| `datafusion` | datafusion, tokio, arrow-array, serde |
| `serde` | serde, serde_json |

Always pass `--no-default-features` and an explicit `--features` flag; omitting it builds nothing optional.

## Pre-commit / verification

Run before every commit — mirrors CI exactly:

```sh
bash .husky/pre-commit
```

What it does (in order):
1. `cargo fmt --all -- --check`
2. `cargo clippy --all-targets` + `cargo check --all-targets` for each feature set (no-features, each flag individually, all combined)
3. `cargo doc --no-deps --all-features` with `RUSTDOCFLAGS="-D warnings --html-in-header $(pwd)/katex-header.html"`

`katex-header.html` is required for doc builds — it enables KaTeX rendering in rustdoc. Math in doc-comments uses `$$…$$` / `$…$` syntax.

## Running tests

```sh
# default (no features)
cargo test --no-default-features

# single feature
cargo test --no-default-features --features polars

# all features
cargo test --no-default-features --features polars,parallel,ades,mpc_80_col,datafusion,serde

# single test
cargo test --no-default-features --features serde my_test_name
```

`proptest` is used for property-based tests; regression files live in `proptest-regressions/`.

## Doc-comment conventions

- Broken intra-doc links cause `cargo doc` to fail (`-D warnings`).
- LaTeX in doc-comments: use `$$…$$` blocks. Raw `[4pt]` or similar LaTeX bracket syntax inside doc-comments is misinterpreted as an intra-doc link — escape with `\\\\` or restructure.
- `astrometry` module was removed; the public coordinate types now live under `coordinates::equatorial` and `coordinates::cartesian`.

## Source layout

```
src/
  coordinates/      # EquCoord, CartesianCoord, spherical_midpoint, covariance propagation
  observation_dataset/  # ObsDataset, Observation, indexing
  observer/         # Observer, ObsErrorModel (FCCT14, CBM10, VFCC17)
  io/
    polars/         # FromPolarsArgs, Polars ingestion
    datafusion/     # LoadObsArgs, async DataFusion loader
    serde/          # Custom Serialize/Deserialize for ObsDataset
    ades/           # ADES XML parser
    mpc_80_col/     # MPC 80-column format parser
  photometry.rs     # Photometry, Filter
  constants.rs      # Earth axes, AU, geodetic constants
```

## CI

CI runs on PRs only (no push trigger). Jobs: `fmt`, `ci` (clippy+check matrix), `doc`, `semver` (cargo-semver-checks), `coverage` (cargo-llvm-cov → Codecov).
