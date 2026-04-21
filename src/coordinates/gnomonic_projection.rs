//! Gnomonic (tangent-plane) projection between sky and local 2-D coordinates.
//!
//! ## Overview
//!
//! The gnomonic projection maps neighbouring sky positions onto a flat
//! tangent plane touching the celestial sphere at a chosen *tangent point*
//! $(\alpha_0, \delta_0)$. Great circles project to straight lines, making
//! it well-suited for short-arc astrometry and kinematic linking.
//!
//! The forward projection formulas are:
//!
//! $$x = \frac{\cos\delta\,\sin(\alpha - \alpha_0)}{c}, \qquad
//!   y = \frac{\cos\delta_0\sin\delta - \sin\delta_0\cos\delta\cos(\alpha - \alpha_0)}{c}$$
//!
//! with the denominator
//!
//! $$c = \sin\delta_0\sin\delta + \cos\delta_0\cos\delta\cos(\alpha - \alpha_0)$$
//!
//! Coordinates $(x, y)$ are expressed in **radians**; $x$ grows towards
//! increasing right ascension (east) and $y$ towards increasing declination
//! (north).
//!
//! ## Types
//!
//! | Type | Role |
//! |------|------|
//! | [`TangentPlane`] | Reference frame defined by a tangent point $(\alpha_0, \delta_0)$; caches $\sin\delta_0$ and $\cos\delta_0$ |
//! | [`TangentPoint`] | A projected position $(x, y)$ bound to its [`TangentPlane`] |
//! | [`TangentVec`] | A displacement vector in the tangent-plane frame (plane-agnostic) |
//!
//! ## Key operations
//!
//! | Operation | Description |
//! |-----------|-------------|
//! | [`TangentPlane::new`] | Create a plane from an [`EquCoord`] tangent point |
//! | [`TangentPlane::project`] | Forward projection: sky → tangent plane |
//! | [`TangentPoint::unproject`] | Inverse projection: tangent plane → sky |
//! | [`TangentPoint::dist2`] | Squared Euclidean distance between two points on the same plane |
//! | [`TangentPoint::midpoint`] | Midpoint of two points on the same plane |
//! | `TangentPoint + TangentVec` | Translate a projected point by a vector |
//! | `TangentPoint - TangentPoint` | Difference vector between two projected points |
//! | [`TangentVec::norm`] / [`TangentVec::norm_sq`] | Euclidean length in the tangent plane |
//!
//! ## Singularity and numerical stability
//!
//! The projection is geometrically singular at the antipode of the tangent
//! point, where $c \to 0$. A floor constant `INV_COSC_MIN` (defined in the
//! parent module) is applied to the denominator to prevent division by zero
//! or overflow; in practice all inputs should remain well within a few
//! degrees of the tangent point.

use std::{
    f64::consts::TAU,
    fmt::{self, Display, Formatter},
    ops::{Add, Mul, Neg, Sub},
};

use crate::coordinates::{INV_COSC_MIN, equatorial::EquCoord};

/// Reference frame for a gnomonic (tangent-plane) projection.
///
/// A `TangentPlane` is defined by a single sky direction
/// $(\alpha_0, \delta_0)$ — the *tangent point* — onto which neighbouring
/// sky positions are projected via the gnomonic formulas:
///
/// $$x = \frac{\cos\delta\,\sin(\alpha - \alpha_0)}{c},\qquad
///   y = \frac{\cos\delta_0\sin\delta - \sin\delta_0\cos\delta\cos(\alpha - \alpha_0)}{c}$$
///
/// with $c = \sin\delta_0\sin\delta + \cos\delta_0\cos\delta\cos(\alpha - \alpha_0)$.
///
/// The struct caches $\sin\delta_0$ and $\cos\delta_0$ so that repeated
/// projections onto the same plane avoid redundant trigonometric calls.
///
/// Notes
/// -----
/// - The projection is singular at the antipode of $(\alpha_0, \delta_0)$;
///   a floor `INV_COSC_MIN` is applied to $c$ to guarantee numerical
///   stability.
/// - **Uncertainty propagation is intentionally out of scope.** This
///   module provides only the geometric mapping
///   $(\alpha, \delta) \leftrightarrow (x, y)$. Propagation of a
///   celestial covariance $\Sigma_{\alpha\delta}$ to the tangent plane
///   via the first-order relation
///   $\Sigma_{xy} = J\,\Sigma_{\alpha\delta}\,J^\top$ is the
///   responsibility of downstream consumers (e.g. the kinematic seed
///   model in `fink-fat::seeding`), which own the definition of the
///   input covariance format and the policy for accumulating
///   uncertainties across epochs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TangentPlane {
    /// Reference equatorial coordinates of the tangent point, with
    /// associated uncertainties (radians).
    ///
    /// Right ascension of the tangent point, $\alpha_0$ (radians).
    ///
    /// This is the sky direction onto which the plane is tangent: the
    /// origin $(x, y) = (0, 0)$ of the projected frame corresponds
    /// exactly to $(\alpha_0, \delta_0)$. Normalised to $[0, 2\pi)$ at
    /// construction.
    ///
    /// Declination of the tangent point, $\delta_0$ (radians).
    ///
    /// Defines, together with the right ascension stored in [`Self::equ_ref`],
    /// the point of tangency on the celestial sphere. Expected to lie in $[-\pi/2, \pi/2]$.
    pub equ_ref: EquCoord,

    /// Precomputed $\sin\delta_0$.
    ///
    /// Cached at construction because $\delta_0$ appears in every
    /// forward and inverse projection through the quantities
    /// $\sin\delta_0$ and $\cos\delta_0$. Storing them avoids a
    /// `sin_cos` call per projected point, which matters when a single
    /// plane is reused to project thousands of alerts.
    sin_dec0: f64,

    /// Precomputed $\cos\delta_0$. See [`Self::sin_dec0`] for rationale.
    cos_dec0: f64,
}

impl TangentPlane {
    /// Build a new tangent plane centred on the given sky position.
    ///
    /// Arguments
    /// ---------
    /// * `equ_ref` – Equatorial coordinates $(\alpha_0, \delta_0)$ of the
    ///   tangent point (radians). The `ra_error` and `dec_error` fields are
    ///   stored but not used by the projection itself.
    ///
    /// Returns
    /// -------
    /// A [`TangentPlane`] with `sin_dec0` and `cos_dec0` precomputed from
    /// `equ_ref.dec`.
    #[inline]
    pub fn new(equ_ref: EquCoord) -> Self {
        let (sin_dec0, cos_dec0) = equ_ref.dec.sin_cos();
        Self {
            equ_ref,
            sin_dec0,
            cos_dec0,
        }
    }

    /// Project a sky position onto this tangent plane.
    ///
    /// Arguments
    /// ---------
    /// * `c` – Target equatorial coordinates $(\alpha, \delta)$ (radians).
    ///   The error fields of `c` are ignored.
    ///
    /// Returns
    /// -------
    /// A [`TangentPoint`] bound to `self` with coordinates $(x, y)$ in radians:
    /// - $x > 0$ east (increasing RA),
    /// - $y > 0$ north (increasing Dec).
    #[inline]
    pub fn project(&self, c: &EquCoord) -> TangentPoint {
        let (sdec, cdec) = c.dec.sin_cos();
        let (sdra, cdra) = (c.ra - self.equ_ref.ra).sin_cos();

        let cosc = self.sin_dec0 * sdec + self.cos_dec0 * cdec * cdra;
        let inv = 1.0 / cosc.max(INV_COSC_MIN);

        let x = cdec * sdra * inv;
        let y = (self.cos_dec0 * sdec - self.sin_dec0 * cdec * cdra) * inv;
        TangentPoint { plane: *self, x, y }
    }

    /// Return the precomputed $\sin\delta_0$.
    ///
    /// Returns
    /// -------
    /// `f64` — $\sin$ of the tangent-point declination, cached at construction.
    #[inline]
    pub fn sin_dec0(&self) -> f64 {
        self.sin_dec0
    }

    /// Return the precomputed $\cos\delta_0$.
    ///
    /// Returns
    /// -------
    /// `f64` — $\cos$ of the tangent-point declination, cached at construction.
    #[inline]
    pub fn cos_dec0(&self) -> f64 {
        self.cos_dec0
    }
}

impl Display for TangentPlane {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "TangentPlane {{")?;
        writeln!(f, "  ra0  : {:.6e} rad", self.equ_ref.ra)?;
        writeln!(f, "  dec0 : {:.6e} rad", self.equ_ref.dec)?;
        write!(f, "}}")
    }
}

/// A point on a specific tangent plane.
///
/// This struct binds a tangent-plane coordinate `(x, y)` to the
/// [`TangentPlane`] in which it was projected, making it impossible to
/// accidentally mix coordinates from incompatible frames.
///
/// Coordinates are expressed in radians, with:
/// - `x` growing towards increasing right ascension (east),
/// - `y` growing towards increasing declination (north).
///
/// See [`TangentPlane::project`] for construction and
/// [`TangentPoint::unproject`] for the inverse operation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TangentPoint {
    /// Tangent plane this coordinate lives in.
    pub plane: TangentPlane,
    /// East offset on the tangent plane (radians).
    pub x: f64,
    /// North offset on the tangent plane (radians).
    pub y: f64,
}

impl TangentPoint {
    /// Construct a point directly from raw offsets on a given plane.
    ///
    /// Arguments
    /// ---------
    /// * `plane` – The [`TangentPlane`] in whose frame `(x, y)` are expressed.
    /// * `x` – East offset from the tangent point (radians).
    /// * `y` – North offset from the tangent point (radians).
    ///
    /// Returns
    /// -------
    /// A [`TangentPoint`] bound to `plane` with the given offsets.
    ///
    /// Notes
    /// -----
    /// Use this only when `(x, y)` are already known to belong to `plane`
    /// (e.g. produced by a kinematic prediction performed in the plane's
    /// local frame). For projecting a sky position, prefer
    /// [`TangentPlane::project`].
    #[inline]
    pub fn new(plane: TangentPlane, x: f64, y: f64) -> Self {
        Self { plane, x, y }
    }

    /// Inverse gnomonic projection: map this tangent-plane point back to the sky.
    ///
    /// Returns
    /// -------
    /// An [`EquCoord`] with `ra ∈ [0, 2π)` and `dec ∈ [-π/2, π/2]`. The
    /// `ra_error` and `dec_error` fields of the result are set to zero.
    ///
    /// Notes
    /// -----
    /// When `(x, y) ≈ (0, 0)` (within a threshold of $\rho^2 < 10^{-24}$)
    /// the tangent point itself is returned directly to avoid a degenerate
    /// `atan2` call.
    #[inline]
    pub fn unproject(&self) -> EquCoord {
        let rho2 = self.x * self.x + self.y * self.y;
        if rho2 < 1e-24 {
            return EquCoord::new(self.plane.equ_ref.ra, 0.0, self.plane.equ_ref.dec, 0.0);
        }
        let rho = rho2.sqrt();
        let c = rho.atan();
        let (sc, cc) = c.sin_cos();
        let s0 = self.plane.sin_dec0;
        let c0 = self.plane.cos_dec0;

        let dec = (cc * s0 + (self.y * sc * c0) / rho).asin();
        let denom = rho * c0 * cc - self.y * s0 * sc;
        let ra = (self.plane.equ_ref.ra + (self.x * sc).atan2(denom)).rem_euclid(TAU);
        EquCoord::new(ra, 0.0, dec, 0.0)
    }

    /// Squared Euclidean distance to another point on the **same** plane.
    ///
    /// $$d^2 = (x_1 - x_2)^2 + (y_1 - y_2)^2$$
    ///
    /// Arguments
    /// ---------
    /// * `other` – The second point; must share the same [`TangentPlane`].
    ///
    /// Returns
    /// -------
    /// `f64` — Non-negative squared distance in radians².
    ///
    /// Panics
    /// ------
    /// In debug builds, panics if `self.plane != other.plane`.
    #[inline]
    pub fn dist2(&self, other: &Self) -> f64 {
        debug_assert_eq!(
            self.plane, other.plane,
            "TangentPoint::dist2 requires both points to share the same plane"
        );
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Midpoint of two points lying on the **same** tangent plane.
    ///
    /// Arguments
    /// ---------
    /// * `a` – First [`TangentPoint`].
    /// * `b` – Second [`TangentPoint`]; must share the same [`TangentPlane`] as `a`.
    ///
    /// Returns
    /// -------
    /// A [`TangentPoint`] on the same plane at coordinates
    /// $\bigl(\tfrac{x_a+x_b}{2},\; \tfrac{y_a+y_b}{2}\bigr)$.
    ///
    /// Panics
    /// ------
    /// In debug builds, panics if `a.plane != b.plane`.
    pub fn midpoint(a: TangentPoint, b: TangentPoint) -> TangentPoint {
        debug_assert_eq!(a.plane, b.plane);
        a + (b - a) * 0.5
    }
}

impl Display for TangentPoint {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "TangentPoint {{")?;
        writeln!(f, "  plane: {}", self.plane)?;
        writeln!(f, "  x    : {:.6e} rad", self.x)?;
        writeln!(f, "  y    : {:.6e} rad", self.y)?;
        write!(f, "}}")
    }
}

/// A displacement vector in the tangent plane (plane-agnostic).
///
/// Represents a difference between two [`TangentPoint`]s or a scaled
/// direction. Carries no plane reference on purpose: it is an element
/// of the local tangent vector space, not a location.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TangentVec {
    /// East component of the displacement (radians).
    pub dx: f64,
    /// North component of the displacement (radians).
    pub dy: f64,
}

impl Display for TangentVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "TangentVec {{")?;
        writeln!(f, "  dx : {:.6e} rad", self.dx)?;
        writeln!(f, "  dy : {:.6e} rad", self.dy)?;
        write!(f, "}}")
    }
}

// point + vector = point (stays on the same plane)
impl Add<TangentVec> for TangentPoint {
    type Output = TangentPoint;
    fn add(self, v: TangentVec) -> TangentPoint {
        TangentPoint {
            plane: self.plane,
            x: self.x + v.dx,
            y: self.y + v.dy,
        }
    }
}

// point - point = vector (only meaningful on the same plane)
impl Sub for TangentPoint {
    type Output = TangentVec;
    fn sub(self, rhs: TangentPoint) -> TangentVec {
        debug_assert_eq!(
            self.plane, rhs.plane,
            "TangentPoint subtraction across different planes"
        );
        TangentVec {
            dx: self.x - rhs.x,
            dy: self.y - rhs.y,
        }
    }
}

impl TangentVec {
    /// Squared Euclidean norm $\|\mathbf{v}\|^2 = dx^2 + dy^2$ (radians²).
    ///
    /// Returns
    /// -------
    /// `f64` — Non-negative squared length of the vector in the tangent plane.
    #[inline]
    pub fn norm_sq(&self) -> f64 {
        self.dx * self.dx + self.dy * self.dy
    }

    /// Euclidean norm $\|\mathbf{v}\| = \sqrt{dx^2 + dy^2}$ (radians).
    ///
    /// Returns
    /// -------
    /// `f64` — Non-negative length of the vector in the tangent plane.
    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }
}

impl Add for TangentVec {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            dx: self.dx + rhs.dx,
            dy: self.dy + rhs.dy,
        }
    }
}

impl Sub for TangentVec {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            dx: self.dx - rhs.dx,
            dy: self.dy - rhs.dy,
        }
    }
}

// scalar * vector
impl Mul<f64> for TangentVec {
    type Output = TangentVec;
    fn mul(self, k: f64) -> TangentVec {
        TangentVec {
            dx: k * self.dx,
            dy: k * self.dy,
        }
    }
}

impl Neg for TangentVec {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            dx: -self.dx,
            dy: -self.dy,
        }
    }
}

#[cfg(test)]
mod gnomonic_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    // ------------------------------------------------------------------ //
    // Helpers                                                              //
    // ------------------------------------------------------------------ //

    fn make_plane(ra0_deg: f64, dec0_deg: f64) -> TangentPlane {
        let equ = EquCoord::new(ra0_deg.to_radians(), 0.0, dec0_deg.to_radians(), 0.0);
        TangentPlane::new(equ)
    }

    fn make_coord(ra_deg: f64, dec_deg: f64) -> EquCoord {
        EquCoord::new(ra_deg.to_radians(), 0.0, dec_deg.to_radians(), 0.0)
    }

    // ------------------------------------------------------------------ //
    // TangentPlane construction                                            //
    // ------------------------------------------------------------------ //

    #[test]
    fn new_caches_sin_cos() {
        let plane = make_plane(10.0, 30.0);
        let dec0 = 30.0_f64.to_radians();
        assert_abs_diff_eq!(plane.sin_dec0(), dec0.sin(), epsilon = 1e-15);
        assert_abs_diff_eq!(plane.cos_dec0(), dec0.cos(), epsilon = 1e-15);
    }

    // ------------------------------------------------------------------ //
    // project / unproject round-trip                                       //
    // ------------------------------------------------------------------ //

    #[test]
    fn project_tangent_point_is_origin() {
        let plane = make_plane(45.0, 20.0);
        let tp = plane.project(&plane.equ_ref);
        assert_abs_diff_eq!(tp.x, 0.0, epsilon = 1e-13);
        assert_abs_diff_eq!(tp.y, 0.0, epsilon = 1e-13);
    }

    #[test]
    fn unproject_origin_returns_tangent_point() {
        let plane = make_plane(45.0, 20.0);
        let tp = TangentPoint::new(plane, 0.0, 0.0);
        let sky = tp.unproject();
        assert_abs_diff_eq!(sky.ra, plane.equ_ref.ra, epsilon = 1e-13);
        assert_abs_diff_eq!(sky.dec, plane.equ_ref.dec, epsilon = 1e-13);
    }

    #[test]
    fn project_unproject_round_trip_small_offset() {
        let plane = make_plane(30.0, 10.0);
        // 0.5 deg offset in ra and dec
        let target = make_coord(30.5, 10.5);
        let tp = plane.project(&target);
        let sky = tp.unproject();
        assert_abs_diff_eq!(sky.ra, target.ra, epsilon = 1e-10);
        assert_abs_diff_eq!(sky.dec, target.dec, epsilon = 1e-10);
    }

    #[test]
    fn project_east_gives_positive_x() {
        let plane = make_plane(30.0, 0.0);
        let east = make_coord(31.0, 0.0); // higher RA = east
        let tp = plane.project(&east);
        assert!(tp.x > 0.0, "x should be positive for points to the east");
    }

    #[test]
    fn project_north_gives_positive_y() {
        let plane = make_plane(30.0, 0.0);
        let north = make_coord(30.0, 1.0); // higher Dec = north
        let tp = plane.project(&north);
        assert!(tp.y > 0.0, "y should be positive for points to the north");
    }

    // ------------------------------------------------------------------ //
    // dist2 / midpoint                                                     //
    // ------------------------------------------------------------------ //

    #[test]
    fn dist2_self_is_zero() {
        let plane = make_plane(45.0, 20.0);
        let p = plane.project(&make_coord(45.5, 20.5));
        assert_abs_diff_eq!(p.dist2(&p), 0.0, epsilon = 1e-20);
    }

    #[test]
    fn dist2_symmetric() {
        let plane = make_plane(45.0, 20.0);
        let p = plane.project(&make_coord(45.5, 20.5));
        let q = plane.project(&make_coord(46.0, 21.0));
        assert_abs_diff_eq!(p.dist2(&q), q.dist2(&p), epsilon = 1e-20);
    }

    #[test]
    fn dist2_pythagoras() {
        let plane = make_plane(0.0, 0.0);
        let a = TangentPoint::new(plane, 3.0e-3, 0.0);
        let b = TangentPoint::new(plane, 0.0, 4.0e-3);
        // expected: (3e-3)^2 + (4e-3)^2 = 25e-6
        assert_abs_diff_eq!(a.dist2(&b), 25.0e-6, epsilon = 1e-20);
    }

    #[test]
    fn midpoint_is_between_endpoints() {
        let plane = make_plane(0.0, 0.0);
        let a = TangentPoint::new(plane, 0.0, 0.0);
        let b = TangentPoint::new(plane, 2.0, 4.0);
        let m = TangentPoint::midpoint(a, b);
        assert_abs_diff_eq!(m.x, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(m.y, 2.0, epsilon = 1e-15);
    }

    // ------------------------------------------------------------------ //
    // TangentVec arithmetic                                                //
    // ------------------------------------------------------------------ //

    #[test]
    fn vec_norm_sq_pythagoras() {
        let v = TangentVec { dx: 3.0, dy: 4.0 };
        assert_abs_diff_eq!(v.norm_sq(), 25.0, epsilon = 1e-15);
        assert_abs_diff_eq!(v.norm(), 5.0, epsilon = 1e-15);
    }

    #[test]
    fn vec_add_sub_roundtrip() {
        let a = TangentVec { dx: 1.0, dy: 2.0 };
        let b = TangentVec { dx: 3.0, dy: -1.0 };
        let s = a + b;
        let d = s - b;
        assert_abs_diff_eq!(d.dx, a.dx, epsilon = 1e-15);
        assert_abs_diff_eq!(d.dy, a.dy, epsilon = 1e-15);
    }

    #[test]
    fn vec_neg() {
        let v = TangentVec { dx: 1.5, dy: -2.5 };
        let n = -v;
        assert_abs_diff_eq!(n.dx, -1.5, epsilon = 1e-15);
        assert_abs_diff_eq!(n.dy, 2.5, epsilon = 1e-15);
    }

    #[test]
    fn vec_mul_scales() {
        let v = TangentVec { dx: 2.0, dy: 3.0 };
        let s = v * 2.5;
        assert_abs_diff_eq!(s.dx, 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(s.dy, 7.5, epsilon = 1e-15);
    }

    #[test]
    fn point_plus_vec_gives_point_on_same_plane() {
        let plane = make_plane(0.0, 0.0);
        let p = TangentPoint::new(plane, 1.0, 2.0);
        let v = TangentVec { dx: 0.5, dy: -0.5 };
        let q = p + v;
        assert_eq!(q.plane, plane);
        assert_abs_diff_eq!(q.x, 1.5, epsilon = 1e-15);
        assert_abs_diff_eq!(q.y, 1.5, epsilon = 1e-15);
    }

    #[test]
    fn point_sub_gives_tangent_vec() {
        let plane = make_plane(0.0, 0.0);
        let a = TangentPoint::new(plane, 3.0, 4.0);
        let b = TangentPoint::new(plane, 1.0, 1.0);
        let v = a - b;
        assert_abs_diff_eq!(v.dx, 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(v.dy, 3.0, epsilon = 1e-15);
    }

    // ------------------------------------------------------------------ //
    // Display                                                              //
    // ------------------------------------------------------------------ //

    #[test]
    fn display_tangent_plane_contains_coords() {
        let plane = make_plane(45.0, 20.0);
        let s = format!("{}", plane);
        assert!(s.contains("TangentPlane"));
        assert!(s.contains("ra0"));
        assert!(s.contains("dec0"));
    }

    #[test]
    fn display_tangent_point_contains_xy() {
        let plane = make_plane(0.0, 0.0);
        let tp = TangentPoint::new(plane, 1.0e-3, 2.0e-3);
        let s = format!("{}", tp);
        assert!(s.contains("x"));
        assert!(s.contains("y"));
    }

    // ------------------------------------------------------------------ //
    // Property-based tests                                                 //
    // ------------------------------------------------------------------ //

    prop_compose! {
        fn valid_ra_deg()(ra in 0.0_f64..360.0) -> f64 { ra }
    }
    prop_compose! {
        fn valid_dec_deg()(dec in -89.0_f64..89.0) -> f64 { dec }
    }
    prop_compose! {
        /// Small offset in degrees so we stay well within the projection's
        /// valid region (avoids antipodal singularity).
        fn small_offset_deg()(off in -3.0_f64..3.0) -> f64 { off }
    }

    proptest! {
        /// project then unproject round-trips for modest offsets.
        #[test]
        fn project_unproject_roundtrip(
            ra0 in valid_ra_deg(),
            dec0 in valid_dec_deg(),
            dra in small_offset_deg(),
            ddec in small_offset_deg(),
        ) {
            let plane = make_plane(ra0, dec0);
            // Keep target dec within valid range
            let target_dec = (dec0 + ddec).clamp(-89.0, 89.0);
            let target = make_coord(ra0 + dra, target_dec);
            let tp = plane.project(&target);
            let sky = tp.unproject();
            // Allow some tolerance: ~1 arcsec = 5e-6 rad is generous
            prop_assert!((sky.ra - target.ra).abs() < 1e-7
                || (sky.ra - target.ra + TAU).abs() < 1e-7
                || (sky.ra - target.ra - TAU).abs() < 1e-7,
                "RA roundtrip failed: got {}, expected {}", sky.ra, target.ra);
            prop_assert!((sky.dec - target.dec).abs() < 1e-7,
                "Dec roundtrip failed: got {}, expected {}", sky.dec, target.dec);
        }

        /// dist2 is symmetric.
        #[test]
        fn dist2_symmetry(
            x1 in -1.0_f64..1.0, y1 in -1.0_f64..1.0,
            x2 in -1.0_f64..1.0, y2 in -1.0_f64..1.0,
        ) {
            let plane = make_plane(0.0, 0.0);
            let p = TangentPoint::new(plane, x1, y1);
            let q = TangentPoint::new(plane, x2, y2);
            prop_assert!((p.dist2(&q) - q.dist2(&p)).abs() < 1e-20);
        }

        /// dist2 ≥ 0.
        #[test]
        fn dist2_nonneg(
            x1 in -1.0_f64..1.0, y1 in -1.0_f64..1.0,
            x2 in -1.0_f64..1.0, y2 in -1.0_f64..1.0,
        ) {
            let plane = make_plane(0.0, 0.0);
            let p = TangentPoint::new(plane, x1, y1);
            let q = TangentPoint::new(plane, x2, y2);
            prop_assert!(p.dist2(&q) >= 0.0);
        }

        /// norm_sq of a scaled vector scales by k^2.
        #[test]
        fn vec_norm_scales(dx in -10.0_f64..10.0, dy in -10.0_f64..10.0, k in -5.0_f64..5.0) {
            let v = TangentVec { dx, dy };
            let scaled = v * k;
            prop_assert!((scaled.norm_sq() - v.norm_sq() * k * k).abs() < 1e-10);
        }

        /// tangent point at origin always unprojects to the tangent point.
        #[test]
        fn unproject_origin_is_ref(ra0 in valid_ra_deg(), dec0 in valid_dec_deg()) {
            let plane = make_plane(ra0, dec0);
            let tp = TangentPoint::new(plane, 0.0, 0.0);
            let sky = tp.unproject();
            prop_assert!((sky.dec - plane.equ_ref.dec).abs() < 1e-12);
        }
    }

    // keep PI in scope for the proptest! macro expansion
    #[allow(unused)]
    const _PI: f64 = PI;
}
