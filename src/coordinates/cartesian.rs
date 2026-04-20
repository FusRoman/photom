//! Cartesian unit-sphere coordinate types with optional covariance.
//!
//! ## Types
//!
//! - [`CartesianCoord`] — a point on the unit celestial sphere represented by its
//!   `x`, `y`, `z` components (`f64`). No uncertainty information is stored.
//! - [`CartesianCoordCov`] — a [`CartesianCoord`] together with the full symmetric
//!   3×3 covariance matrix, packed as its upper triangle in row-major order:
//!   `[xx, xy, xz, yy, yz, zz]`.
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
//! - [`EquCoord::to_cartesian_cov`] (in the [`equatorial`][crate::coordinates::equatorial]
//!   module) — propagates the diagonal input covariance
//!   $(\sigma_\alpha^2, \sigma_\delta^2)$ to a full 3×3 output covariance via the
//!   Jacobian $J$ of $(\alpha,\delta)\to(x,y,z)$.
//! - [`CartesianCoordCov::to_equatorial`] — inverse propagation via the Jacobian $K$
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

use crate::coordinates::equatorial::EquCoord;

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
/// This type therefore carries the full symmetric 3×3 covariance matrix,
/// packed as its upper triangle in row-major order:
///
/// | index | entry        |
/// |-------|--------------|
/// | 0     | $\sigma_{xx}$ |
/// | 1     | $\sigma_{xy}$ |
/// | 2     | $\sigma_{xz}$ |
/// | 3     | $\sigma_{yy}$ |
/// | 4     | $\sigma_{yz}$ |
/// | 5     | $\sigma_{zz}$ |
///
/// The covariance obtained from an [`EquCoord`] is rank-deficient (rank 2):
/// the residual direction is radial, which is consistent with a position
/// constrained to the unit sphere.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CartesianCoordCov {
    pub coord: CartesianCoord,
    /// Upper-triangular packed covariance: `[xx, xy, xz, yy, yz, zz]`.
    pub cov: [f64; 6],
}

impl CartesianCoordCov {
    /// Marginal 1-σ uncertainty on `x`.
    #[inline]
    pub fn x_error(&self) -> f64 {
        self.cov[0].sqrt()
    }

    /// Marginal 1-σ uncertainty on `y`.
    #[inline]
    pub fn y_error(&self) -> f64 {
        self.cov[3].sqrt()
    }

    /// Marginal 1-σ uncertainty on `z`.
    #[inline]
    pub fn z_error(&self) -> f64 {
        self.cov[5].sqrt()
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
    /// [`EquCoord::to_cartesian_cov`] to propagate them.
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

impl CartesianCoordCov {
    /// Convert back to equatorial coordinates, propagating the full 3×3
    /// covariance via first-order linearisation.
    ///
    /// Formulation
    /// -----------
    /// With $\rho = \sqrt{x^2 + y^2}$, the Jacobian of
    /// $(x,y,z) \to (\alpha,\delta)$ is
    /// $$
    /// K = \begin{pmatrix}
    /// -\dfrac{y}{\rho^2} & \dfrac{x}{\rho^2} & 0 \\\\
    /// -\dfrac{xz}{\rho}  & -\dfrac{yz}{\rho} & \rho
    /// \end{pmatrix}.
    /// $$
    /// The output covariance is
    /// $\Sigma_{\alpha\delta} = K \, \Sigma_{xyz} \, K^\top,$
    /// from which marginal 1-σ errors are recovered as the square roots of
    /// the diagonal. Any induced correlation between $\alpha$ and $\delta$
    /// is **discarded** when packing the result into an [`EquCoord`].
    ///
    /// Numerical notes
    /// ---------------
    /// - Near the poles ($\rho \to 0$) the RA error diverges; this reflects
    ///   a genuine geometric singularity, not a numerical artefact.
    /// - The input vector is not required to be unit-normalised, but the
    ///   linearisation is only meaningful when it is close to the sphere.
    pub fn to_equatorial(self) -> EquCoord {
        let CartesianCoord { x, y, z } = self.coord;
        let [cxx, cxy, cxz, cyy, cyz, czz] = self.cov;

        let rho2 = x * x + y * y;
        let rho = rho2.sqrt();

        let dec = z.atan2(rho);
        let ra = y.atan2(x).rem_euclid(TAU);

        // Jacobian rows: K[0,:] = ∂α/∂(x,y,z), K[1,:] = ∂δ/∂(x,y,z).
        let k00 = -y / rho2;
        let k01 = x / rho2;
        let k02 = 0.0;

        let k10 = -x * z / rho;
        let k11 = -y * z / rho;
        let k12 = rho;

        // Multiply Σ_xyz by a row vector k = (k0, k1, k2): returns K Σ kᵀ
        // components needed to build the 2×2 output covariance.
        #[inline]
        fn quad_form(k0: f64, k1: f64, k2: f64, c: [f64; 6]) -> f64 {
            let [cxx, cxy, cxz, cyy, cyz, czz] = c;
            k0 * k0 * cxx
                + k1 * k1 * cyy
                + k2 * k2 * czz
                + 2.0 * (k0 * k1 * cxy + k0 * k2 * cxz + k1 * k2 * cyz)
        }

        let cov_pack = [cxx, cxy, cxz, cyy, cyz, czz];
        let var_ra = quad_form(k00, k01, k02, cov_pack);
        let var_dec = quad_form(k10, k11, k12, cov_pack);

        EquCoord::new(ra, var_ra.sqrt(), dec, var_dec.sqrt())
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
    // to_cartesian_cov — deterministic tests                              //
    // ------------------------------------------------------------------ //

    mod to_cartesian_cov {
        use super::*;

        /// Position components in `to_cartesian_cov` match `From<EquCoord>` conversion.
        #[test]
        fn position_matches_direct_conversion() {
            let equ = EquCoord::from_degrees(37.0, 1e-5, -15.0, 1e-5);
            let cov_result = equ.to_cartesian_cov();
            let direct = CartesianCoord::from(equ);
            assert_abs_diff_eq!(cov_result.coord.x, direct.x, epsilon = 1e-15);
            assert_abs_diff_eq!(cov_result.coord.y, direct.y, epsilon = 1e-15);
            assert_abs_diff_eq!(cov_result.coord.z, direct.z, epsilon = 1e-15);
        }

        /// Zero input errors produce an all-zero covariance matrix.
        #[test]
        fn zero_errors_give_zero_covariance() {
            let equ = EquCoord::new(0.5, 0.0, 0.3, 0.0);
            let cov_result = equ.to_cartesian_cov();
            for (i, &v) in cov_result.cov.iter().enumerate() {
                assert!(
                    v.abs() < 1e-30,
                    "covariance[{i}] should be zero but was {v}"
                );
            }
        }

        /// Known numerical value: RA=0, Dec=0, ra_error=dec_error=1e-5.
        ///
        /// Jacobian at (RA=0, Dec=0): j00=0, j10=1, j20=0, j01=0, j11=0, j21=1.
        /// With var_ra=var_dec=1e-10 the covariance is [0, 0, 0, 1e-10, 0, 1e-10].
        #[test]
        fn known_numerical_covariance_ra0_dec0() {
            let sigma = 1e-5_f64;
            let equ = EquCoord::new(0.0, sigma, 0.0, sigma);
            let cov_result = equ.to_cartesian_cov();
            let var = sigma * sigma; // 1e-10
            let [cxx, cxy, cxz, cyy, cyz, czz] = cov_result.cov;
            assert_abs_diff_eq!(cxx, 0.0, epsilon = 1e-30);
            assert_abs_diff_eq!(cxy, 0.0, epsilon = 1e-30);
            assert_abs_diff_eq!(cxz, 0.0, epsilon = 1e-30);
            assert_abs_diff_eq!(cyy, var, epsilon = 1e-25);
            assert_abs_diff_eq!(cyz, 0.0, epsilon = 1e-30);
            assert_abs_diff_eq!(czz, var, epsilon = 1e-25);
        }

        /// The packed upper-triangular covariance is symmetric by construction:
        /// the off-diagonal elements are the cross-covariances (xy, xz, yz).
        /// Verify they remain finite and that diagonal entries are non-negative.
        #[test]
        fn covariance_diagonal_is_non_negative() {
            let equ = EquCoord::from_degrees(60.0, 5e-6, 30.0, 3e-6);
            let [cxx, _cxy, _cxz, cyy, _cyz, czz] = equ.to_cartesian_cov().cov;
            assert!(cxx >= 0.0, "cxx={cxx} should be non-negative");
            assert!(cyy >= 0.0, "cyy={cyy} should be non-negative");
            assert!(czz >= 0.0, "czz={czz} should be non-negative");
        }
    }

    // ------------------------------------------------------------------ //
    // CartesianCoordCov::to_equatorial — deterministic tests              //
    // ------------------------------------------------------------------ //

    mod to_equatorial {
        use super::*;

        /// With all-zero covariance the recovered RA and Dec match the lossless conversion.
        #[test]
        fn zero_covariance_recovers_ra_and_dec() {
            let equ = EquCoord::from_degrees(200.0, 0.0, -40.0, 0.0);
            let cov_struct = CartesianCoordCov {
                coord: CartesianCoord::from(equ),
                cov: [0.0; 6],
            };
            let recovered = cov_struct.to_equatorial();
            // RA difference normalised to (−π, π]
            let dra = (recovered.ra - equ.ra).rem_euclid(2.0 * PI);
            let dra = if dra > PI { dra - 2.0 * PI } else { dra };
            assert_abs_diff_eq!(dra, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(recovered.dec, equ.dec, epsilon = 1e-12);
        }

        /// With all-zero covariance, the output errors are zero.
        #[test]
        fn zero_covariance_gives_zero_errors() {
            let equ = EquCoord::from_degrees(10.0, 0.0, 20.0, 0.0);
            let cov_struct = CartesianCoordCov {
                coord: CartesianCoord::from(equ),
                cov: [0.0; 6],
            };
            let recovered = cov_struct.to_equatorial();
            assert_abs_diff_eq!(recovered.ra_error, 0.0, epsilon = 1e-30);
            assert_abs_diff_eq!(recovered.dec_error, 0.0, epsilon = 1e-30);
        }

        /// Full round-trip `to_cartesian_cov` → `to_equatorial` recovers RA/Dec within 1e-12.
        #[test]
        fn roundtrip_cov_recovers_ra_and_dec() {
            let equ = EquCoord::from_degrees(75.0, 1e-6, -20.0, 5e-7);
            let recovered = equ.to_cartesian_cov().to_equatorial();
            let dra = (recovered.ra - equ.ra).rem_euclid(2.0 * PI);
            let dra = if dra > PI { dra - 2.0 * PI } else { dra };
            assert_abs_diff_eq!(dra, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(recovered.dec, equ.dec, epsilon = 1e-12);
        }

        /// Full round-trip recovers ra_error and dec_error within 1 % (first-order approx.).
        #[test]
        fn roundtrip_cov_recovers_errors_within_one_percent() {
            let ra_err = 1e-5_f64;
            let dec_err = 8e-6_f64;
            let equ =
                EquCoord::from_degrees(120.0, ra_err.to_degrees(), 25.0, dec_err.to_degrees());
            let recovered = equ.to_cartesian_cov().to_equatorial();
            let tol = 0.01; // 1 %
            assert!(
                (recovered.ra_error - equ.ra_error).abs() <= tol * equ.ra_error.abs().max(1e-30),
                "ra_error round-trip mismatch: {} vs {}",
                recovered.ra_error,
                equ.ra_error
            );
            assert!(
                (recovered.dec_error - equ.dec_error).abs() <= tol * equ.dec_error.abs().max(1e-30),
                "dec_error round-trip mismatch: {} vs {}",
                recovered.dec_error,
                equ.dec_error
            );
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

        /// `to_cartesian_cov().coord` matches `CartesianCoord::from(equ)` within 1e-12.
        #[test]
        fn prop_to_cartesian_cov_pos_matches(coord in valid_coord()) {
            let direct   = CartesianCoord::from(coord);
            let via_cov  = coord.to_cartesian_cov().coord;
            prop_assert!((via_cov.x - direct.x).abs() < 1e-12, "x mismatch");
            prop_assert!((via_cov.y - direct.y).abs() < 1e-12, "y mismatch");
            prop_assert!((via_cov.z - direct.z).abs() < 1e-12, "z mismatch");
        }

        /// Full round-trip through covariance recovers ra/dec within 1e-10 and
        /// errors within 10 % (away from poles, small errors).
        #[test]
        fn prop_roundtrip_cov(coord in valid_coord_with_errors()) {
            let recovered = coord.to_cartesian_cov().to_equatorial();

            // Position recovery
            let dra = (recovered.ra - coord.ra).rem_euclid(2.0 * PI);
            let dra = if dra > PI { dra - 2.0 * PI } else { dra };
            prop_assert!(dra.abs() < 1e-10, "RA round-trip error: {dra}");
            prop_assert!((recovered.dec - coord.dec).abs() < 1e-10, "Dec round-trip error");

            // Error recovery (10 % relative, with small absolute floor to guard near-zero)
            let ra_err_ref  = coord.ra_error.max(1e-15);
            let dec_err_ref = coord.dec_error.max(1e-15);
            prop_assert!(
                (recovered.ra_error  - coord.ra_error ).abs() <= 0.10 * ra_err_ref,
                "ra_error round-trip: {} vs {}", recovered.ra_error, coord.ra_error
            );
            prop_assert!(
                (recovered.dec_error - coord.dec_error).abs() <= 0.10 * dec_err_ref,
                "dec_error round-trip: {} vs {}", recovered.dec_error, coord.dec_error
            );
        }
    }
}
