//! Celestial coordinate types and coordinate-system conversions.
//!
//! Two sub-modules are provided:
//!
//! - [`equatorial`] — [`equatorial::EquCoord`]: equatorial sky position (RA, Dec) with
//!   1-σ astrometric uncertainties, Vincenty angular separation, spherical midpoint,
//!   and covariance propagation to Cartesian coordinates.
//! - [`cartesian`] — [`cartesian::CartesianCoord`]: Cartesian unit-sphere position;
//!   [`cartesian::CartesianCoordCov`]: position bundled with a full 3×3 covariance matrix
//!   and inverse propagation back to equatorial coordinates.
//!
//! ## Conversion paths
//!
//! Two conversion paths are available depending on whether astrometric uncertainties
//! need to be preserved:
//!
//! - **Lossless (position only):** [`From`] impls convert between
//!   [`equatorial::EquCoord`] and [`cartesian::CartesianCoord`] in either direction.
//!   Uncertainties are discarded on the forward path and set to zero on the inverse
//!   path.
//! - **Covariance-propagating:** [`equatorial::EquCoord::to_cartesian_cov`] maps
//!   equatorial coordinates to [`cartesian::CartesianCoordCov`] via a first-order
//!   Jacobian propagation. The reverse direction is provided by
//!   [`cartesian::CartesianCoordCov::to_equatorial`], which extracts marginal 1-σ
//!   errors from the diagonal of the back-propagated covariance.

pub mod cartesian;
pub mod equatorial;

/// Lower bound used to avoid division by a nearly-zero vector norm when
/// averaging or normalizing spherical vectors.
///
/// Context
/// -------
/// - Used in `spherical_midpoint()`: we add two unit vectors and normalize them.
/// - When the directions are **nearly opposite**, the sum vector can approach
///   the zero vector, making its length extremely small.
///
/// Consequences without guard
/// --------------------------
/// A norm `r ≈ 0` produces catastrophic amplification of noise when
/// normalizing `(x/r, y/r, z/r)` → `NaN`, `Inf`, or huge garbage values.
///
/// Why this constant?
/// ------------------
/// We clamp:
/// ```text
/// r = max(r, NORM_MIN)
/// ```
/// before normalizing.
///
/// Choice of value
/// ---------------
/// - `1e-16` is slightly above the smallest meaningful double-precision values
///   for normalized vectors (~1e-308 is too small, causes underflow well before).
/// - It preserves stability without biasing typical use.
/// - It only activates in extreme geometries (nearly antipodal sources) that we
///   *never* use for seed construction anyway.
///
/// This constant is purely a **numerical robustness guard**.
const NORM_MIN: f64 = 1e-16;
