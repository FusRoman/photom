//! I/O backends for loading astronomical observation data into `photom` types.
//!
//! Each sub-module is feature-gated:
//!
//! | Feature | Module | Description |
//! |---------|--------|-------------|
//! | `polars` | [`polars`] | Ingest from a Polars `DataFrame` / `LazyFrame` |
//! | `ades` | [`ades`] | Parse ADES (Astrometry Data Exchange Standard) files |
//! | `mpc_80_col` | [`mpc_80_col`] | Parse MPC 80-column observation records |
//! | `datafusion` | [`datafusion`] | Load from a Parquet URI via DataFusion + object_store |
//!
//! Without any feature enabled this module is empty and invisible to the compiler.

#[cfg(feature = "polars")]
pub mod polars;

#[cfg(feature = "ades")]
pub mod ades;

#[cfg(feature = "mpc_80_col")]
pub mod mpc_80_col;

#[cfg(feature = "datafusion")]
pub mod datafusion;

#[cfg(feature = "serde")]
pub mod serde;
