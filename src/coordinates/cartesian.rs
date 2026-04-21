//! Cartesian unit-sphere coordinate types with optional covariance.
//!
//! ## Types
//!
//! - [`CartesianCoord`] — a point on the unit celestial sphere represented by its
//!   `x`, `y`, `z` components (`f64`). No uncertainty information is stored.
//! - [`CartesianCoordCov`] — a [`CartesianCoord`] together with the full symmetric
//!   3×3 covariance matrix stored as a [`Cov3`].
//!
//! ## Why a full covariance matrix?
//!
//! The spherical-to-Cartesian map
//! $$(\alpha, \delta) \mapsto (\cos\delta\cos\alpha,\; \cos\delta\sin\alpha,\; \sin\delta)$$
//! is nonlinear, so even when $\alpha$ and $\delta$ are uncorrelated the Cartesian
//! components $(x, y, z)$ are in general correlated. Storing only the diagonal
//! would discard that information and bias any subsequent back-conversion to
//! equatorial coordinates. [`CartesianCoordCov`] therefore carries the full
//! symmetric matrix.
//!
//! ## Conversion paths
//!
//! - `From<`[`EquCoord`]`> for CartesianCoord` — lossless projection; uncertainties
//!   carried by the [`EquCoord`] are discarded.
//! - [`EquCoordCov::to_cartesian_cov`] (in the [`equatorial`][crate::coordinates::equatorial]
//!   module) — propagates the diagonal input covariance
//!   $(\sigma_\alpha^2, \sigma_\delta^2)$ to a full 3×3 output covariance via the
//!   Jacobian $J$ of $(\alpha,\delta)\to(x,y,z)$.
//! - [`CartesianCoordCov::to_equatorial_cov`] — inverse propagation via the Jacobian $K$
//!   of $(x,y,z)\to(\alpha,\delta)$; marginal 1-σ errors are extracted from the
//!   diagonal of the back-propagated covariance.
//! - `From<`[`CartesianCoord`]`> for EquCoord` — lossless; output errors set to zero.
//!
//! ## Pole singularity
//!
//! Near the celestial poles ($\rho = \sqrt{x^2+y^2} \to 0$) the RA partial
//! derivatives in $K$ diverge, so the propagated RA error grows without bound.
//! This reflects a genuine geometric singularity of the equatorial coordinate system,
//! not a numerical artefact.

use std::{f64::consts::TAU, ops::Add};

use crate::coordinates::{
    cov3::Cov3,
    equatorial::{EquCoord, EquCoordCov},
};

/// A point on the unit celestial sphere in Cartesian coordinates.
///
/// The three components satisfy  $ x^2 + y^2 + z^2 = 1 $  when derived from an
/// [`EquCoord`] via [`From`]. No uncertainty information is attached to this
/// type; use [`CartesianCoordCov`] when astrometric errors must be propagated.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CartesianCoord {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl CartesianCoord {
    /// Dot product of two Cartesian vectors.
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl Add for CartesianCoord {
    type Output = CartesianCoord;

    /// Component-wise addition of two Cartesian vectors.
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

/// A Cartesian sky position together with its full 3×3 covariance matrix.
///
/// Rationale
/// ---------
/// The spherical-to-Cartesian mapping
/// $$(\alpha, \delta) \mapsto (\cos\delta\cos\alpha,\; \cos\delta\sin\alpha,\; \sin\delta)$$
/// is nonlinear. Even when $\alpha$ and $\delta$ are uncorrelated (as assumed
/// in [`EquCoord`], which stores only marginal 1-σ errors), the resulting
/// Cartesian components $(x, y, z)$ are in general **correlated**. Storing
/// only the diagonal $(\sigma_x, \sigma_y, \sigma_z)$ would discard that
/// information and bias any subsequent conversion back to equatorial
/// coordinates.
///
/// This type therefore carries the full symmetric 3×3 covariance matrix as a
/// [`Cov3`] value. The covariance obtained from an [`EquCoord`] is rank-deficient
/// (rank 2): the residual direction is radial, which is consistent with a
/// position constrained to the unit sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CartesianCoordCov {
    pub coord: CartesianCoord,
    /// Full symmetric 3×3 covariance matrix in Cartesian $(x, y, z)$ coordinates.
    pub cov: Cov3,
}

impl CartesianCoordCov {
    /// Marginal 1-σ uncertainty on `x`.
    #[inline]
    pub fn x_error(&self) -> f64 {
        self.cov.xx.sqrt()
    }

    /// Marginal 1-σ uncertainty on `y`.
    #[inline]
    pub fn y_error(&self) -> f64 {
        self.cov.yy.sqrt()
    }

    /// Marginal 1-σ uncertainty on `z`.
    #[inline]
    pub fn z_error(&self) -> f64 {
        self.cov.zz.sqrt()
    }

    /// Marginal 1-σ uncertainties as a triple `(σ_x, σ_y, σ_z)`.
    #[inline]
    pub fn errors(&self) -> (f64, f64, f64) {
        (self.x_error(), self.y_error(), self.z_error())
    }
}

// ---------------------------------------------------------------------------
// Lossless conversion: positions only.
// ---------------------------------------------------------------------------

impl From<&EquCoord> for CartesianCoord {
    /// Project an equatorial direction onto the unit sphere.
    ///
    /// Uncertainties carried by `coord` are **discarded**. Use
    /// [`EquCoordCov::to_cartesian_cov`] to propagate them.
    fn from(coord: &EquCoord) -> Self {
        let (sdec, cdec) = coord.dec.sin_cos();
        let (sra, cra) = coord.ra.sin_cos();
        Self {
            x: cdec * cra,
            y: cdec * sra,
            z: sdec,
        }
    }
}

impl From<EquCoord> for CartesianCoord {
    /// Convenience wrapper for `From<&EquCoord>` to allow direct conversion from owned `EquCoord`.
    #[inline]
    fn from(equ: EquCoord) -> Self {
        Self::from(&equ)
    }
}

impl From<EquCoordCov> for CartesianCoordCov {
    /// Propagate a full 2×2 equatorial covariance to Cartesian 3×3 covariance.
    ///
    /// Delegates to [`EquCoordCov::to_cartesian_cov`].
    #[inline]
    fn from(ec: EquCoordCov) -> Self {
        ec.to_cartesian_cov()
    }
}

impl From<CartesianCoordCov> for EquCoordCov {
    /// Convert a Cartesian covariance back to equatorial, preserving the full
    /// 2×2 RA–Dec covariance.
    ///
    /// Delegates to [`CartesianCoordCov::to_equatorial_cov`].
    #[inline]
    fn from(cc: CartesianCoordCov) -> Self {
        cc.to_equatorial_cov()
    }
}

impl CartesianCoordCov {
    /// Convert back to equatorial coordinates, returning a full [`EquCoordCov`]
    /// that retains the RA–Dec off-diagonal covariance term.
    ///
    /// # Formulation
    ///
    /// With $\rho = \sqrt{x^2 + y^2}$, the Jacobian of
    /// $(x,y,z) \to (\alpha,\delta)$ is
    /// $$
    /// K = \begin{pmatrix}
    /// -\dfrac{y}{\rho^2} & \dfrac{x}{\rho^2} & 0 \\\\
    /// -\dfrac{xz}{\rho}  & -\dfrac{yz}{\rho} & \rho
    /// \end{pmatrix}.
    /// $$
    /// The output covariance is $\Sigma_{\alpha\delta} = K\,\Sigma_{xyz}\,K^\top$,
    /// computed via [`Cov3::transform_j2`].  The marginal 1-σ errors stored in
    /// the returned [`EquCoord`] are the square roots of the diagonal of
    /// $\Sigma_{\alpha\delta}$.
    ///
    /// # Notes
    ///
    /// - Near the poles ($\rho \to 0$) the RA error diverges; this is a
    ///   genuine geometric singularity, not a numerical artefact.
    /// - Unlike [`CartesianCoordCov::to_equatorial_cov`], this method preserves
    ///   the full 2×2 output covariance including the RA–Dec cross-term.
    pub fn to_equatorial_cov(self) -> EquCoordCov {
        let CartesianCoord { x, y, z } = self.coord;

        let rho2 = x * x + y * y;
        let rho = rho2.sqrt();

        let dec = z.atan2(rho);
        let ra = y.atan2(x).rem_euclid(TAU);

        let k = [
            [-y / rho2, x / rho2, 0.0],
            [-x * z / rho, -y * z / rho, rho],
        ];

        let cov2 = self.cov.transform_j2(k);
        let coord = EquCoord::new(ra, cov2.xx.sqrt(), dec, cov2.yy.sqrt());
        EquCoordCov::new(coord, cov2)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod cartesian_tests {
    use super::*;
    use crate::coordinates::equatorial::EquCoord;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    // ------------------------------------------------------------------ //
    // Helper                                                               //
    // ------------------------------------------------------------------ //

    fn norm(c: &CartesianCoord) -> f64 {
        (c.x * c.x + c.y * c.y + c.z * c.z).sqrt()
    }

    // ------------------------------------------------------------------ //
    // Proptest strategies                                                  //
    // ------------------------------------------------------------------ //

    prop_compose! {
        fn valid_ra()(ra in 0.0_f64..=(2.0 * PI)) -> f64 { ra }
    }

    prop_compose! {
        fn valid_dec()(dec in (-PI / 2.0)..=(PI / 2.0)) -> f64 { dec }
    }

    prop_compose! {
        fn valid_error()(e in 0.0_f64..=0.1_f64) -> f64 { e }
    }

    prop_compose! {
        fn valid_coord()(ra in valid_ra(), dec in valid_dec()) -> EquCoord {
            EquCoord::new(ra, 0.0, dec, 0.0)
        }
    }

    prop_compose! {
        fn valid_coord_with_errors()(
            ra  in valid_ra(),
            dec in (-1.3_f64)..=1.3_f64,         // stay away from poles
            ra_err  in 0.0_f64..=1e-4_f64,
            dec_err in 0.0_f64..=1e-4_f64,
        ) -> EquCoord {
            EquCoord::new(ra, ra_err, dec, dec_err)
        }
    }

    // ------------------------------------------------------------------ //
    // CartesianCoord: From<EquCoord> for known directions                 //
    // ------------------------------------------------------------------ //

    mod from_equ_coord {
        use super::*;

        /// RA=0, Dec=0 maps to the unit vector (1, 0, 0).
        #[test]
        fn ra0_dec0_maps_to_x_axis() {
            let equ = EquCoord::new(0.0, 0.0, 0.0, 0.0);
            let cart = CartesianCoord::from(equ);
            assert_abs_diff_eq!(cart.x, 1.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.y, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.z, 0.0, epsilon = 1e-15);
        }

        /// RA=90°, Dec=0 maps to the unit vector (0, 1, 0).
        #[test]
        fn ra90_dec0_maps_to_y_axis() {
            let equ = EquCoord::from_degrees(90.0, 0.0, 0.0, 0.0);
            let cart = CartesianCoord::from(equ);
            assert_abs_diff_eq!(cart.x, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.y, 1.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.z, 0.0, epsilon = 1e-15);
        }

        /// RA=0, Dec=90° (north pole) maps to the unit vector (0, 0, 1).
        #[test]
        fn ra0_dec90_maps_to_north_pole() {
            let equ = EquCoord::from_degrees(0.0, 0.0, 90.0, 0.0);
            let cart = CartesianCoord::from(equ);
            assert_abs_diff_eq!(cart.x, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.y, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.z, 1.0, epsilon = 1e-15);
        }

        /// RA=0, Dec=-90° (south pole) maps to the unit vector (0, 0, -1).
        #[test]
        fn ra0_dec_neg90_maps_to_south_pole() {
            let equ = EquCoord::from_degrees(0.0, 0.0, -90.0, 0.0);
            let cart = CartesianCoord::from(equ);
            assert_abs_diff_eq!(cart.x, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.y, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.z, -1.0, epsilon = 1e-15);
        }

        /// RA=180°, Dec=0 maps to the unit vector (-1, 0, 0).
        #[test]
        fn ra180_dec0_maps_to_neg_x_axis() {
            let equ = EquCoord::from_degrees(180.0, 0.0, 0.0, 0.0);
            let cart = CartesianCoord::from(equ);
            assert_abs_diff_eq!(cart.x, -1.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.y, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(cart.z, 0.0, epsilon = 1e-15);
        }
    }

    // ------------------------------------------------------------------ //
    // CartesianCoord: Add                                                  //
    // ------------------------------------------------------------------ //

    mod add {
        use super::*;

        /// Adding two known Cartesian vectors yields the component-wise sum.
        #[test]
        fn add_two_known_vectors() {
            let a = CartesianCoord {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            };
            let b = CartesianCoord {
                x: 0.5,
                y: -1.0,
                z: 0.25,
            };
            let sum = a + b;
            assert_abs_diff_eq!(sum.x, 1.5, epsilon = 1e-15);
            assert_abs_diff_eq!(sum.y, 1.0, epsilon = 1e-15);
            assert_abs_diff_eq!(sum.z, 3.25, epsilon = 1e-15);
        }

        /// Adding a vector to a zero vector returns the original vector.
        #[test]
        fn add_zero_vector_is_identity() {
            let a = CartesianCoord {
                x: 0.6,
                y: -0.8,
                z: 0.0,
            };
            let zero = CartesianCoord {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            let sum = a + zero;
            assert_abs_diff_eq!(sum.x, a.x, epsilon = 1e-15);
            assert_abs_diff_eq!(sum.y, a.y, epsilon = 1e-15);
            assert_abs_diff_eq!(sum.z, a.z, epsilon = 1e-15);
        }
    }

    // ------------------------------------------------------------------ //
    // From<CartesianCoord> for EquCoord                                   //
    // ------------------------------------------------------------------ //

    mod cart_to_equ {
        use super::*;

        /// (1, 0, 0) converts to RA=0, Dec=0.
        #[test]
        fn x_axis_gives_ra0_dec0() {
            let cart = CartesianCoord {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            };
            let equ = EquCoord::from(cart);
            assert_abs_diff_eq!(equ.ra, 0.0, epsilon = 1e-15);
            assert_abs_diff_eq!(equ.dec, 0.0, epsilon = 1e-15);
        }

        /// (0, 1, 0) converts to RA=π/2, Dec=0.
        #[test]
        fn y_axis_gives_ra_half_pi_dec0() {
            let cart = CartesianCoord {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            };
            let equ = EquCoord::from(cart);
            assert_abs_diff_eq!(equ.ra, PI / 2.0, epsilon = 1e-15);
            assert_abs_diff_eq!(equ.dec, 0.0, epsilon = 1e-15);
        }

        /// (0, 0, 1) (north pole) converts to Dec=π/2.
        #[test]
        fn north_pole_gives_dec_half_pi() {
            let cart = CartesianCoord {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            };
            let equ = EquCoord::from(cart);
            assert_abs_diff_eq!(equ.dec, PI / 2.0, epsilon = 1e-15);
        }

        /// (0, 0, -1) (south pole) converts to Dec=-π/2.
        #[test]
        fn south_pole_gives_dec_neg_half_pi() {
            let cart = CartesianCoord {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            };
            let equ = EquCoord::from(cart);
            assert_abs_diff_eq!(equ.dec, -PI / 2.0, epsilon = 1e-15);
        }

        /// `From<CartesianCoord>` always sets errors to zero.
        #[test]
        fn errors_are_zero_in_conversion() {
            let cart = CartesianCoord {
                x: 0.6,
                y: 0.8,
                z: 0.0,
            };
            let equ = EquCoord::from(cart);
            assert_abs_diff_eq!(equ.ra_error, 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(equ.dec_error, 0.0, epsilon = 0.0);
        }
    }

    // ------------------------------------------------------------------ //
    // Round-trip EquCoord → CartesianCoord → EquCoord                    //
    // ------------------------------------------------------------------ //

    mod roundtrip_equ_cart_equ {
        use super::*;

        /// RA is recovered modulo 2π within 1e-12.
        #[test]
        fn ra_recovered_after_roundtrip() {
            let equ = EquCoord::from_degrees(123.456, 0.0, 34.567, 0.0);
            let recovered = EquCoord::from(CartesianCoord::from(equ));
            // Normalise difference to [−π, π]
            let diff = (recovered.ra - equ.ra).rem_euclid(2.0 * PI);
            let diff = if diff > PI { diff - 2.0 * PI } else { diff };
            assert_abs_diff_eq!(diff, 0.0, epsilon = 1e-12);
        }

        /// Dec is recovered within 1e-12.
        #[test]
        fn dec_recovered_after_roundtrip() {
            let equ = EquCoord::from_degrees(55.0, 0.0, -27.3, 0.0);
            let recovered = EquCoord::from(CartesianCoord::from(equ));
            assert_abs_diff_eq!(recovered.dec, equ.dec, epsilon = 1e-12);
        }

        /// Errors are always zero after a lossless round-trip (no covariance stored).
        #[test]
        fn errors_are_dropped_to_zero_in_roundtrip() {
            let equ = EquCoord::new(1.0, 0.05, 0.3, 0.02);
            let recovered = EquCoord::from(CartesianCoord::from(equ));
            assert_abs_diff_eq!(recovered.ra_error, 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(recovered.dec_error, 0.0, epsilon = 0.0);
        }
    }

    // ------------------------------------------------------------------ //
    // EquCoord::spherical_midpoint — deterministic tests                  //
    // ------------------------------------------------------------------ //

    mod spherical_midpoint {
        use super::*;

        /// The midpoint of a point with itself equals the original point.
        #[test]
        fn midpoint_with_self_is_same_point() {
            let c = EquCoord::from_degrees(123.0, 0.0, 45.0, 0.0);
            let mid = c.spherical_midpoint(&c);
            assert_abs_diff_eq!(mid.ra, c.ra, epsilon = 1e-12);
            assert_abs_diff_eq!(mid.dec, c.dec, epsilon = 1e-12);
        }

        /// Two equatorial points (Dec=0) have a midpoint with Dec=0 and averaged RA.
        #[test]
        fn equatorial_midpoint_has_averaged_ra_and_zero_dec() {
            let c1 = EquCoord::from_degrees(40.0, 0.0, 0.0, 0.0);
            let c2 = EquCoord::from_degrees(100.0, 0.0, 0.0, 0.0);
            let mid = c1.spherical_midpoint(&c2);
            assert_abs_diff_eq!(mid.dec, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(mid.ra, 70.0_f64.to_radians(), epsilon = 1e-12);
        }

        /// Spherical midpoint is symmetric: midpoint(a, b) ≈ midpoint(b, a).
        #[test]
        fn midpoint_is_symmetric() {
            let a = EquCoord::from_degrees(30.0, 0.0, 20.0, 0.0);
            let b = EquCoord::from_degrees(80.0, 0.0, 50.0, 0.0);
            let mid_ab = a.spherical_midpoint(&b);
            let mid_ba = b.spherical_midpoint(&a);
            assert_abs_diff_eq!(mid_ab.ra, mid_ba.ra, epsilon = 1e-12);
            assert_abs_diff_eq!(mid_ab.dec, mid_ba.dec, epsilon = 1e-12);
        }

        /// Midpoint of north and south poles must not produce NaN (robustness guard).
        #[test]
        fn midpoint_of_antipodal_poles_is_not_nan() {
            let north = EquCoord::from_degrees(0.0, 0.0, 90.0, 0.0);
            let south = EquCoord::from_degrees(0.0, 0.0, -90.0, 0.0);
            let mid = north.spherical_midpoint(&south);
            assert!(!mid.ra.is_nan(), "midpoint RA is NaN for antipodal poles");
            assert!(!mid.dec.is_nan(), "midpoint Dec is NaN for antipodal poles");
        }

        /// Midpoint of two equatorial points 90° apart has Dec=0 and intermediate RA.
        #[test]
        fn equatorial_midpoint_90_degrees_apart() {
            let c1 = EquCoord::from_degrees(0.0, 0.0, 0.0, 0.0);
            let c2 = EquCoord::from_degrees(90.0, 0.0, 0.0, 0.0);
            let mid = c1.spherical_midpoint(&c2);
            assert_abs_diff_eq!(mid.dec, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(mid.ra, 45.0_f64.to_radians(), epsilon = 1e-12);
        }
    }

    // ------------------------------------------------------------------ //
    // Property-based tests                                                 //
    // ------------------------------------------------------------------ //

    proptest! {
     /// Unit vectors projected from any EquCoord must have norm ≈ 1.
     #[test]
     fn prop_equ_to_cart_unit_norm(coord in valid_coord()) {
         let cart = CartesianCoord::from(coord);
         let n = norm(&cart);
         prop_assert!(
             (n - 1.0).abs() < 1e-12,
             "norm={n} for coord ra={} dec={}", coord.ra, coord.dec
         );
     }

     /// Round-trip RA is recovered within 1e-10 (away from poles).
     #[test]
     fn prop_cart_to_equ_roundtrip_ra(
         ra  in valid_ra(),
         dec in (-1.4_f64)..=1.4_f64,  // exclude poles where RA is degenerate
     ) {
         let equ = EquCoord::new(ra, 0.0, dec, 0.0);
         let recovered = EquCoord::from(CartesianCoord::from(equ));
         let diff = (recovered.ra - ra).rem_euclid(2.0 * PI);
         let diff = if diff > PI { diff - 2.0 * PI } else { diff };
         prop_assert!(
             diff.abs() < 1e-10,
             "RA round-trip error: {diff} (ra={ra}, dec={dec})"
         );
     }

     /// Round-trip Dec is recovered within 1e-10.
     #[test]
     fn prop_cart_to_equ_roundtrip_dec(coord in valid_coord()) {
         let recovered = EquCoord::from(CartesianCoord::from(coord));
         prop_assert!(
             (recovered.dec - coord.dec).abs() < 1e-10,
             "Dec round-trip error: {} vs {}", recovered.dec, coord.dec
         );
     }

     /// spherical_midpoint is symmetric: |mid(a,b).ra − mid(b,a).ra| < 1e-10.
     #[test]
     fn prop_spherical_midpoint_symmetric(a in valid_coord(), b in valid_coord()) {
         let mid_ab = a.spherical_midpoint(&b);
         let mid_ba = b.spherical_midpoint(&a);
         prop_assert!(
             (mid_ab.ra - mid_ba.ra).abs() < 1e-10,
             "midpoint RA asymmetry: {} vs {}", mid_ab.ra, mid_ba.ra
         );
         prop_assert!(
             (mid_ab.dec - mid_ba.dec).abs() < 1e-10,
             "midpoint Dec asymmetry: {} vs {}", mid_ab.dec, mid_ba.dec
         );
     }

     /// Midpoint of a coord with itself returns the same RA/Dec within 1e-10.
     #[test]
     fn prop_spherical_midpoint_self(coord in valid_coord()) {
         let mid = coord.spherical_midpoint(&coord);
         let dra = (mid.ra - coord.ra).rem_euclid(2.0 * PI);
         let dra = if dra > PI { dra - 2.0 * PI } else { dra };
         prop_assert!(
             dra.abs() < 1e-10,
             "midpoint(self) RA mismatch: {} vs {}", mid.ra, coord.ra
         );
         prop_assert!(
             (mid.dec - coord.dec).abs() < 1e-10,
             "midpoint(self) Dec mismatch: {} vs {}", mid.dec, coord.dec
         );
     }
    }

    // ------------------------------------------------------------------ //
    // CartesianCoordCov::to_equatorial_cov tests                          //
    // ------------------------------------------------------------------ //

    mod to_equatorial_cov {
        use super::*;
        use crate::coordinates::equatorial::EquCoordCov;

        /// With zero covariance the output errors are zero.
        #[test]
        fn zero_covariance_gives_zero_errors() {
            let equ = EquCoord::from_degrees(10.0, 0.0, 20.0, 0.0);
            let cc = CartesianCoordCov {
                coord: CartesianCoord::from(equ),
                cov: Cov3::zero(),
            };
            let ec = cc.to_equatorial_cov();
            assert_abs_diff_eq!(ec.cov.xx, 0.0, epsilon = 1e-30);
            assert_abs_diff_eq!(ec.cov.yy, 0.0, epsilon = 1e-30);
            assert_abs_diff_eq!(ec.cov.xy, 0.0, epsilon = 1e-30);
        }

        /// Round-trip via EquCoordCov::to_cartesian_cov preserves covariance.
        /// EquCoordCov → CartesianCoordCov → EquCoordCov should recover the cov2.
        #[test]
        fn round_trip_equ_coord_cov() {
            use crate::coordinates::cov2::Cov2;
            let coord = EquCoord::from_degrees(50.0, 0.0, 15.0, 0.0);
            // Start from a diagonal cov2
            let cov2_in = Cov2::diag(4e-12, 9e-12);
            let ec_in = EquCoordCov::new(
                EquCoord::new(coord.ra, cov2_in.xx.sqrt(), coord.dec, cov2_in.yy.sqrt()),
                cov2_in,
            );
            // Propagate to 3D and back
            let ec_out = ec_in.to_cartesian_cov().to_equatorial_cov();
            // Covariance must be recovered to first order
            assert_abs_diff_eq!(ec_out.cov.xx, cov2_in.xx, epsilon = cov2_in.xx * 1e-6);
            assert_abs_diff_eq!(ec_out.cov.yy, cov2_in.yy, epsilon = cov2_in.yy * 1e-6);
            assert_abs_diff_eq!(ec_out.cov.xy, 0.0, epsilon = 1e-24);
        }
    }
}
