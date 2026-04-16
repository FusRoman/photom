//! I/O backends for loading astronomical observation data into `photom` types.
//!
//! This module is gated behind the `polars` feature flag and exposes only the
//! [`polars`] sub-module, which converts a Polars `DataFrame`
//! or `LazyFrame` into an
//! [`crate::observation_dataset::ObsDataset`].
//!
//! Without the `polars` feature this module is empty and invisible to the
//! compiler.
#[cfg(feature = "polars")]
pub mod polars;

#[cfg(feature = "ades")]
pub mod ades;

#[cfg(feature = "mpc_80_col")]
pub mod mpc_80_col;
