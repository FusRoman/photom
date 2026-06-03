//! Celestial coordinate types and coordinate-system conversions.
//!
//! Five sub-modules cover the full chain from raw sky positions through
//! covariance propagation to tangent-plane projection:
//!
//! - [`equatorial`] — [`equatorial::EquCoord`]: equatorial sky position (RA, Dec) with
//!   1-σ astrometric uncertainties, Vincenty angular separation, spherical midpoint,
//!   and covariance propagation to Cartesian coordinates.
//! - [`ecliptic`] — [`ecliptic::EclipticCoord`]: ecliptic sky position (λ, β) with
//!   1-σ astrometric uncertainties; [`ecliptic::EclipticCoordCov`]: position bundled
//!   with a 2×2 covariance matrix and full covariance-propagating conversions to and
//!   from equatorial coordinates via $R_x(\varepsilon)$.
//! - [`cartesian`] — [`cartesian::CartesianCoord`]: Cartesian unit-sphere position;
//!   [`cartesian::CartesianCoordCov`]: position bundled with a full 3×3 covariance matrix
//!   and inverse propagation back to equatorial coordinates.
//! - [`cov2`] — [`cov2::Cov2`]: symmetric 2×2 covariance matrix for tangent-plane
//!   error ellipses; eigenvalues, Mahalanobis distance, and isotropic inflation.
//! - [`cov3`] — [`cov3::Cov3`]: symmetric 3×3 covariance matrix for Cartesian
//!   position uncertainty; bilinear forms and Jacobian propagation to 2D.
//! - [`gnomonic_projection`] — [`gnomonic_projection::TangentPlane`],
//!   [`gnomonic_projection::TangentPoint`], [`gnomonic_projection::TangentVec`]:
//!   gnomonic (tangent-plane) projection between equatorial sky coordinates and
//!   a local 2-D Cartesian frame.
//!
//! ## Conversion paths
//!
//! Two conversion paths are available depending on whether astrometric uncertainties
//! need to be preserved:
//!
//! - **Lossless (position only):** [`From`] impls convert between
//!   [`equatorial::EquCoord`] and [`cartesian::CartesianCoord`] in either direction,
//!   and between [`equatorial::EquCoord`] and [`ecliptic::EclipticCoord`].
//!   Uncertainties are discarded on the forward path and set to zero on the inverse
//!   path.
//! - **Covariance-propagating:** [`From`] impls between
//!   [`equatorial::EquCoordCov`] and [`ecliptic::EclipticCoordCov`] propagate the
//!   full 2×2 covariance via a first-order Jacobian through the obliquity rotation
//!   $R_x(\varepsilon)$. [`equatorial::EquCoordCov::to_cartesian_cov`] maps equatorial
//!   coordinates to [`cartesian::CartesianCoordCov`]; the reverse direction is provided
//!   by [`cartesian::CartesianCoordCov::to_equatorial_cov`].
//!
//! ## Typical workflow
//!
//! ```text
//! EquCoord  ──(From)──►  CartesianCoord
//!    │                        │
//!    │  EquCoordCov           │  to_equatorial_cov()
//!    │  ::to_cartesian_cov()  │
//!    ▼                        ▼
//! CartesianCoordCov  ◄──(From)── EquCoordCov
//!
//! EquCoord      ◄──(From)──►  EclipticCoord
//!    │                              │
//! EquCoordCov  ◄──(From)──►  EclipticCoordCov
//!
//! EquCoord  ──(TangentPlane::project)──►  TangentPoint
//!    ▲                                         │
//!    └──────────(TangentPoint::unproject)──────┘
//! ```

pub mod cartesian;
pub mod cov2;
pub mod cov3;
pub mod ecliptic;
pub mod equatorial;
pub mod gnomonic_projection;

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

/// Smallest allowed value for the denominator `cosc` in gnomonic projection.
///
/// Context
/// -------
/// In the gnomonic projection:
/// ```text
/// cosc = sin(dec0) sin(dec) + cos(dec0) cos(dec) cos(ra - ra0)
/// x    = cos(dec) sin(ra - ra0) / cosc
/// y    = [cos(dec0) sin(dec) - sin(dec0) cos(dec) cos(ra - ra0)] / cosc
/// ```
/// When the target direction lies close to **90° from the tangent point**,
/// `cosc → 0` and the exact projection diverges to infinity.
///
/// Why this constant?
/// ------------------
/// Mathematically, divergence is expected and correct. But numerically, when
/// `cosc` reaches values smaller than ~1e-15, floating-point rounding can produce:
/// - divisions by zero,
/// - NaNs,
/// - overflow into `Inf` even for non-pathological inputs.
///
/// We therefore enforce:
/// ```text
/// inv = 1.0 / max(cosc, INV_COSC_MIN)
/// ```
///
/// Choice of value
/// ----------------
/// - `1e-12` keeps enough dynamic range for small-angle work (few degrees),
/// - avoids Inf/NaN for borderline cases,
/// - does *not* distort the projection in the regime where we actually use it:
///   all fink-fat seeds remain well within the validity domain.
///
/// This value is not physically meaningful; it's a **numerical safety floor**.
const INV_COSC_MIN: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// J2000.0 obliquity of the ecliptic in radians.
///
/// $\varepsilon = 23.439\,291\,1° \times \pi / 180$
pub const OBLIQUITY_J2000: f64 = 23.439_291_1_f64 * std::f64::consts::PI / 180.0;

/// Precomputed $\sin\varepsilon$ for J2000.0.
const SIN_OBL: f64 = 0.397_777_155_931_913_7;

/// Precomputed $\cos\varepsilon$ for J2000.0.
const COS_OBL: f64 = 0.917_482_137_086_962_1;

/// Numerical floor on $r_{xy} = \sqrt{x_e^2 + y_e^2}$ applied when computing
/// the Jacobian near the ecliptic poles.
const RXY_MIN: f64 = 1e-12;
