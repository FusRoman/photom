#![cfg(feature = "polars")]

use crate::io::polars::FromPolarsArgs;
use crate::io::polars::error::PolarsError;
use crate::observation_dataset::ObsDataset;
use polars::{frame::DataFrame, prelude::LazyFrame};

use crate::io::polars::load_observation_from_polars;

impl ObsDataset {
    /// Construct an [`ObsDataset`] from a Polars [`DataFrame`].
    ///
    /// Validates the frame against the expected schema, extracts all
    /// observation columns, and assembles the dataset.  See
    /// [`crate::io::polars`] for the full column specification and
    /// observer-resolution rules.
    ///
    /// # Arguments
    ///
    /// - `df` — the source Polars [`DataFrame`] containing the observation data.
    /// - `args` — the [`FromPolarsArgs`] used to configure the ingestion process,
    ///   including error model specification and observer resolution rules.
    ///
    /// # Returns
    ///
    /// A fully initialised [`ObsDataset`] with all observations indexed and
    /// the LRU cache empty.
    ///
    /// # Errors
    ///
    /// Returns a [`PolarsError`] if the frame fails schema validation, if a
    /// Polars-internal operation fails, or if any observer column violates
    /// the resolution rules (e.g. a partially-null geodetic triplet).
    pub fn from_polars(df: &DataFrame, args: FromPolarsArgs) -> Result<Self, PolarsError> {
        load_observation_from_polars(df, args)
    }

    /// Construct an [`ObsDataset`] from a Polars [`LazyFrame`].
    ///
    /// The lazy computation plan is executed (via [`LazyFrame::collect`]) before
    /// ingestion begins.  Once collected, the same validation and assembly
    /// pipeline as [`ObsDataset::from_polars`] is applied.
    ///
    /// # Arguments
    ///
    /// - `lf` — the source Polars [`LazyFrame`].
    /// - `args` — the [`FromPolarsArgs`] used to configure the ingestion process,
    ///   including error model specification and observer resolution rules.
    ///
    /// # Returns
    ///
    /// A fully initialised [`ObsDataset`] with all observations indexed and
    /// the LRU cache empty.
    ///
    /// # Errors
    ///
    /// Returns a [`PolarsError`] if the lazy computation fails, if the resulting
    /// frame fails schema validation, if a Polars-internal operation fails, or if
    /// any observer column violates the resolution rules (e.g. a partially-null
    /// geodetic triplet).  The error variants are the same as for [`ObsDataset::from_polars`], but
    /// may also include errors related to lazy execution (e.g. plan optimization or collection failures).
    pub fn from_lazy(lf: LazyFrame, args: FromPolarsArgs) -> Result<Self, PolarsError> {
        load_observation_from_polars(lf, args)
    }
}
