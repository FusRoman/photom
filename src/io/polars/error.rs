use thiserror::Error;

// ── error type ───────────────────────────────────────────────────────────────

/// Errors that can arise while loading observations from a Polars [`DataFrame`].
///
/// Every variant carries enough context to identify the offending row or
/// column so that callers can produce actionable diagnostics.
#[derive(Debug, Error)]
pub enum PolarsError {
    /// The [`DataFrame`] did not satisfy the required [`ObsSchema`].
    ///
    /// The inner `String` contains a human-readable description of which
    /// schema constraint was violated (e.g. a missing column or a type
    /// mismatch).
    #[error("Schema validation error: {0}")]
    SchemaValidationError(String),

    /// A column existed but held values of an unexpected Polars type.
    ///
    /// The inner `String` names the column and describes the type conflict.
    #[error("Column type error: {0}")]
    ColumnTypeError(String),

    /// A column that was required by the schema was absent from the frame.
    ///
    /// The inner `String` is the column name.
    #[error("Missing column error: {0}")]
    MissingColumnError(String),

    /// A value could not be converted to its target Rust type.
    ///
    /// The inner `String` describes the conversion that failed.
    #[error("Data conversion error: {0}")]
    DataConversionError(String),

    /// Exactly one or two of the three geodetic observer columns
    /// (`obs_lon`, `obs_lat`, `obs_alt`) were `null` at the given row.
    ///
    /// The geodetic triplet must be either entirely present or entirely
    /// absent; a partial specification is always rejected.  The boolean
    /// fields indicate which columns were *non-null* (i.e. `true` means
    /// the value was present).
    #[error(
        "Observer fields are partially null at row {row}: obs_lon={obs_lon}, obs_lat={obs_lat}, obs_alt={obs_alt} — the triplet must be all non-null or all null"
    )]
    PartialTripletNull {
        /// Zero-based index of the offending row.
        row: usize,
        /// Whether `obs_lon` was non-null at this row.
        obs_lon: bool,
        /// Whether `obs_lat` was non-null at this row.
        obs_lat: bool,
        /// Whether `obs_alt` was non-null at this row.
        obs_alt: bool,
    },

    /// The geodetic triplet was fully non-null but `obs_ra_acc` or
    /// `obs_dec_acc` (or both) were `null` at the given row.
    ///
    /// Astrometric accuracy values are mandatory whenever a custom geodetic
    /// observer is specified.  The inner value is the zero-based row index.
    #[error(
        "obs_ra_acc / obs_dec_acc are required when the geodetic triplet is fully specified (row {0})"
    )]
    MissingAccuracyForGeodesic(usize),

    /// The `mpc_code_obs` cell at the given row did not parse as a valid
    /// three-byte ASCII MPC observatory code.
    ///
    /// The first field is the offending raw string; the second is the
    /// zero-based row index.
    #[error("Invalid MPC code '{0}' at row {1}: must be exactly 3 ASCII bytes")]
    InvalidMpcCode(String, usize),

    /// The `traj_id` column is present in the [`DataFrame`] but its Polars
    /// type is neither `UInt64` nor `String`.
    ///
    /// Only these two column types are accepted: `UInt64` cells are mapped to
    /// [`crate::trajectory::TrajId::Int`] and `String` cells are mapped to
    /// [`crate::trajectory::TrajId::Str`].  Any other type (e.g. `Int32`,
    /// `Float64`) must be cast by the caller before ingestion.
    ///
    /// The inner `String` is a human-readable description of the actual type
    /// that was found.
    #[error("traj_id column has unsupported type: {0} (expected UInt64 or String)")]
    TrajIdColumnTypeError(String),

    /// A Polars-internal error propagated transparently from the underlying
    /// [`polars`] crate.
    #[error(transparent)]
    Polars(#[from] polars::error::PolarsError),
}
