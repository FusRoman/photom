//! Symmetric 3×3 covariance matrix for Cartesian position uncertainty.
//!
//! ## Purpose
//!
//! [`Cov3`] is the natural container for the full positional covariance of a
//! unit-sphere direction once it has been expressed in Cartesian coordinates
//! $(x, y, z)$.  Because the spherical-to-Cartesian map is nonlinear, even
//! uncorrelated equatorial uncertainties $(\sigma_\alpha, \sigma_\delta)$
//! produce correlated Cartesian components.  [`Cov3`] retains all six
//! independent entries of that symmetric matrix.
//!
//! ## Storage layout
//!
//! Only the six independent entries of the symmetric 3×3 matrix are kept:
//!
//! $$\Sigma = \begin{pmatrix}
//!   \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
//!   \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
//!   \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
//! \end{pmatrix}$$
//!
//! Storing only the upper triangle eliminates the risk of accidentally
//! constructing a non-symmetric matrix.
//!
//! ## Jacobian propagation
//!
//! The central operation provided by [`Cov3`] is [`Cov3::transform_j2`],
//! which propagates a 3×3 covariance through a 2×3 Jacobian $K$:
//!
//! $$\Sigma_{2\times 2} = K\,\Sigma_{3\times 3}\,K^\top$$
//!
//! This is the key step in converting a Cartesian positional covariance back
//! to an equatorial covariance (see
//! [`CartesianCoordCov::to_equatorial_cov`][crate::coordinates::cartesian::CartesianCoordCov::to_equatorial_cov]).
//!
//! The symmetric analogue for 2D → 3D propagation lives on [`Cov2`]:
//! [`Cov2::transform_j3`][crate::coordinates::cov2::Cov2::transform_j3].
//!
//! ## Key operations
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`Cov3::zero`] | All-zero (singular) covariance |
//! | [`Cov3::diag`] | Diagonal covariance from three independent variances |
//! | [`Cov3::trace`] | Sum of diagonal entries |
//! | [`Cov3::bilinear`] | Symmetric bilinear form $a^\top \Sigma b$ |
//! | [`Cov3::transform_j2`] | Propagate through 2×3 Jacobian $K$: $K\Sigma K^\top$ |
//! | `+`, `* f64` | Component-wise addition and scalar scaling |

use std::{
    fmt::{Display, Formatter},
    ops::{Add, Mul},
};

use crate::coordinates::cov2::Cov2;

/// Symmetric 3×3 covariance matrix.
///
/// Stores only the six independent entries of a symmetric positive
/// semi-definite matrix:
///
/// $$\Sigma = \begin{pmatrix}
///   \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
///   \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
///   \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
/// \end{pmatrix}$$
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cov3 {
    /// Variance along $x$: $\sigma_{xx}$.
    pub xx: f64,
    /// Variance along $y$: $\sigma_{yy}$.
    pub yy: f64,
    /// Variance along $z$: $\sigma_{zz}$.
    pub zz: f64,
    /// Covariance between $x$ and $y$: $\sigma_{xy} = \sigma_{yx}$.
    pub xy: f64,
    /// Covariance between $x$ and $z$: $\sigma_{xz} = \sigma_{zx}$.
    pub xz: f64,
    /// Covariance between $y$ and $z$: $\sigma_{yz} = \sigma_{zy}$.
    pub yz: f64,
}

impl Cov3 {
    /// All-zero covariance — all six entries set to zero.
    ///
    /// A zero covariance is singular; it represents a distribution with no
    /// uncertainty (or unknown uncertainty). Primarily useful as an additive
    /// identity or a placeholder.
    ///
    /// # Returns
    ///
    /// A [`Cov3`] with all entries equal to `0.0`.
    #[inline]
    pub fn zero() -> Self {
        Self {
            xx: 0.0,
            yy: 0.0,
            zz: 0.0,
            xy: 0.0,
            xz: 0.0,
            yz: 0.0,
        }
    }

    /// Diagonal covariance from three independent variances.
    ///
    /// $$\Sigma = \begin{pmatrix}
    ///   v_x & 0   & 0   \\
    ///   0   & v_y & 0   \\
    ///   0   & 0   & v_z
    /// \end{pmatrix}$$
    ///
    /// # Arguments
    ///
    /// - `var_x` — Variance along the first axis ($\sigma_{xx} \geq 0$).
    /// - `var_y` — Variance along the second axis ($\sigma_{yy} \geq 0$).
    /// - `var_z` — Variance along the third axis ($\sigma_{zz} \geq 0$).
    ///
    /// # Returns
    ///
    /// A [`Cov3`] with `xx = var_x`, `yy = var_y`, `zz = var_z`, all
    /// off-diagonal entries set to `0`.
    #[inline]
    pub fn diag(var_x: f64, var_y: f64, var_z: f64) -> Self {
        Self {
            xx: var_x,
            yy: var_y,
            zz: var_z,
            xy: 0.0,
            xz: 0.0,
            yz: 0.0,
        }
    }

    /// Trace $\mathrm{tr}(\Sigma) = \sigma_{xx} + \sigma_{yy} + \sigma_{zz}$.
    ///
    /// # Returns
    ///
    /// `f64` — The sum of the three diagonal entries.
    #[inline]
    pub fn trace(&self) -> f64 {
        self.xx + self.yy + self.zz
    }

    /// Symmetric bilinear form $a^\top \Sigma\, b$.
    ///
    /// Computes
    ///
    /// $$a^\top \Sigma\, b = \sum_{i,j} a_i\,\Sigma_{ij}\,b_j$$
    ///
    /// using only the six independent entries stored in `self`.
    ///
    /// # Arguments
    ///
    /// - `a` — Left vector $[a_x, a_y, a_z]$.
    /// - `b` — Right vector $[b_x, b_y, b_z]$.
    ///
    /// # Returns
    ///
    /// The scalar $a^\top \Sigma\, b$.
    ///
    /// # Notes
    ///
    /// When `a == b` this reduces to the quadratic form $a^\top \Sigma\, a$
    /// (the Mahalanobis-style variance in direction $a$).
    #[inline]
    pub fn bilinear(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * (self.xx * b[0] + self.xy * b[1] + self.xz * b[2])
            + a[1] * (self.xy * b[0] + self.yy * b[1] + self.yz * b[2])
            + a[2] * (self.xz * b[0] + self.yz * b[1] + self.zz * b[2])
    }

    /// Propagate this 3×3 covariance through a 2×3 Jacobian $K$.
    ///
    /// Computes the 2×2 output covariance
    ///
    /// $$\Sigma_{2\times2} = K\,\Sigma_{3\times3}\,K^\top$$
    ///
    /// which is the first-order propagation formula for a linear map
    /// $\mathbb{R}^3 \to \mathbb{R}^2$.
    ///
    /// # Arguments
    ///
    /// - `k` — The 2×3 Jacobian matrix stored in row-major order:
    ///   `k[i]` is the $i$-th row, `k[i][j]` is the entry $K_{ij}$.
    ///
    /// # Returns
    ///
    /// A [`Cov2`] equal to $K \Sigma K^\top$.
    ///
    /// # Notes
    ///
    /// This is the key step in converting a Cartesian positional covariance
    /// back to an equatorial covariance.  The Jacobian $K$ of
    /// $(x,y,z) \to (\alpha,\delta)$ is 2×3; passing it here gives the full
    /// 2×2 equatorial covariance including the RA–Dec correlation term.
    #[inline]
    pub fn transform_j2(&self, k: [[f64; 3]; 2]) -> Cov2 {
        Cov2 {
            xx: self.bilinear(k[0], k[0]),
            yy: self.bilinear(k[1], k[1]),
            xy: self.bilinear(k[0], k[1]),
        }
    }
}

impl Add for Cov3 {
    type Output = Self;

    /// Component-wise addition of two [`Cov3`] matrices.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            xx: self.xx + rhs.xx,
            yy: self.yy + rhs.yy,
            zz: self.zz + rhs.zz,
            xy: self.xy + rhs.xy,
            xz: self.xz + rhs.xz,
            yz: self.yz + rhs.yz,
        }
    }
}

impl Add for &Cov3 {
    type Output = Cov3;

    /// Component-wise addition of two [`Cov3`] matrices by reference.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl Mul<f64> for Cov3 {
    type Output = Self;

    /// Scalar multiplication: scale all six entries by `rhs`.
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            xx: self.xx * rhs,
            yy: self.yy * rhs,
            zz: self.zz * rhs,
            xy: self.xy * rhs,
            xz: self.xz * rhs,
            yz: self.yz * rhs,
        }
    }
}

impl Mul<f64> for &Cov3 {
    type Output = Cov3;

    /// Scalar multiplication by reference: scale all six entries by `rhs`.
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        *self * rhs
    }
}

impl Display for Cov3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cov3 {{")?;
        writeln!(f, "  xx : {:.6e}", self.xx)?;
        writeln!(f, "  yy : {:.6e}", self.yy)?;
        writeln!(f, "  zz : {:.6e}", self.zz)?;
        writeln!(f, "  xy : {:.6e}", self.xy)?;
        writeln!(f, "  xz : {:.6e}", self.xz)?;
        writeln!(f, "  yz : {:.6e}", self.yz)?;
        write!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod cov3_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    const EPS: f64 = 1e-12;

    // ------------------------------------------------------------------ //
    // Constructors                                                         //
    // ------------------------------------------------------------------ //

    #[test]
    fn zero_all_entries_zero() {
        let c = Cov3::zero();
        assert_eq!(c.xx, 0.0);
        assert_eq!(c.yy, 0.0);
        assert_eq!(c.zz, 0.0);
        assert_eq!(c.xy, 0.0);
        assert_eq!(c.xz, 0.0);
        assert_eq!(c.yz, 0.0);
    }

    #[test]
    fn diag_sets_off_diagonal_to_zero() {
        let c = Cov3::diag(1.0, 4.0, 9.0);
        assert_abs_diff_eq!(c.xx, 1.0, epsilon = EPS);
        assert_abs_diff_eq!(c.yy, 4.0, epsilon = EPS);
        assert_abs_diff_eq!(c.zz, 9.0, epsilon = EPS);
        assert_abs_diff_eq!(c.xy, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(c.xz, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(c.yz, 0.0, epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // trace                                                                //
    // ------------------------------------------------------------------ //

    #[test]
    fn trace_sums_diagonal() {
        let c = Cov3 {
            xx: 1.0,
            yy: 2.0,
            zz: 3.0,
            xy: 0.5,
            xz: 0.1,
            yz: 0.2,
        };
        assert_abs_diff_eq!(c.trace(), 6.0, epsilon = EPS);
    }

    #[test]
    fn trace_diag_is_sum_of_variances() {
        let c = Cov3::diag(2.0, 5.0, 7.0);
        assert_abs_diff_eq!(c.trace(), 14.0, epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // bilinear                                                             //
    // ------------------------------------------------------------------ //

    #[test]
    fn bilinear_with_diagonal_cov() {
        // Σ = diag(2, 3, 5); a = b = [1, 1, 1] → aᵀΣa = 2+3+5 = 10
        let c = Cov3::diag(2.0, 3.0, 5.0);
        let v = [1.0, 1.0, 1.0];
        assert_abs_diff_eq!(c.bilinear(v, v), 10.0, epsilon = EPS);
    }

    #[test]
    fn bilinear_with_unit_vectors() {
        // bilinear(e_x, e_y, Σ) should equal Σ.xy
        let c = Cov3 {
            xx: 1.0,
            yy: 2.0,
            zz: 3.0,
            xy: 4.0,
            xz: 5.0,
            yz: 6.0,
        };
        let ex = [1.0, 0.0, 0.0];
        let ey = [0.0, 1.0, 0.0];
        let ez = [0.0, 0.0, 1.0];
        assert_abs_diff_eq!(c.bilinear(ex, ey), c.xy, epsilon = EPS);
        assert_abs_diff_eq!(c.bilinear(ex, ez), c.xz, epsilon = EPS);
        assert_abs_diff_eq!(c.bilinear(ey, ez), c.yz, epsilon = EPS);
        // Diagonal entries via quadratic form
        assert_abs_diff_eq!(c.bilinear(ex, ex), c.xx, epsilon = EPS);
        assert_abs_diff_eq!(c.bilinear(ey, ey), c.yy, epsilon = EPS);
        assert_abs_diff_eq!(c.bilinear(ez, ez), c.zz, epsilon = EPS);
    }

    #[test]
    fn bilinear_is_symmetric_in_a_b() {
        // aᵀΣb == bᵀΣa for a symmetric matrix
        let c = Cov3 {
            xx: 2.0,
            yy: 3.0,
            zz: 4.0,
            xy: 0.5,
            xz: -0.2,
            yz: 0.7,
        };
        let a = [1.0, 2.0, -1.0];
        let b = [-0.5, 0.3, 2.0];
        assert_abs_diff_eq!(c.bilinear(a, b), c.bilinear(b, a), epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // transform_j2                                                         //
    // ------------------------------------------------------------------ //

    #[test]
    fn transform_j2_identity_rows_gives_marginals() {
        // K = [[1,0,0],[0,1,0]]  →  K Σ Kᵀ = [[Σ.xx, Σ.xy],[Σ.xy, Σ.yy]]
        let c = Cov3 {
            xx: 2.0,
            yy: 3.0,
            zz: 5.0,
            xy: 0.5,
            xz: 0.1,
            yz: 0.2,
        };
        let k = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out = c.transform_j2(k);
        assert_abs_diff_eq!(out.xx, c.xx, epsilon = EPS);
        assert_abs_diff_eq!(out.yy, c.yy, epsilon = EPS);
        assert_abs_diff_eq!(out.xy, c.xy, epsilon = EPS);
    }

    #[test]
    fn transform_j2_diagonal_input_matches_manual() {
        // Σ = diag(4, 9, 16); K = [[1,1,0],[0,1,1]]
        // K Σ Kᵀ: xx = 4+9=13, yy = 9+16=25, xy = 9
        let c = Cov3::diag(4.0, 9.0, 16.0);
        let k = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]];
        let out = c.transform_j2(k);
        assert_abs_diff_eq!(out.xx, 13.0, epsilon = EPS);
        assert_abs_diff_eq!(out.yy, 25.0, epsilon = EPS);
        assert_abs_diff_eq!(out.xy, 9.0, epsilon = EPS);
    }

    #[test]
    fn transform_j2_zero_cov_gives_zero_output() {
        let c = Cov3::zero();
        let k = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = c.transform_j2(k);
        assert_eq!(out.xx, 0.0);
        assert_eq!(out.yy, 0.0);
        assert_eq!(out.xy, 0.0);
    }

    #[test]
    fn transform_j2_output_is_symmetric() {
        // xy of output must equal bilinear(k[0], k[1]) == bilinear(k[1], k[0])
        let c = Cov3 {
            xx: 1.0,
            yy: 2.0,
            zz: 3.0,
            xy: 0.3,
            xz: -0.1,
            yz: 0.4,
        };
        let k = [[0.5, -1.0, 0.2], [0.1, 0.8, -0.3]];
        let out = c.transform_j2(k);
        let xy_check = c.bilinear(k[1], k[0]);
        assert_abs_diff_eq!(out.xy, xy_check, epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // Arithmetic                                                           //
    // ------------------------------------------------------------------ //

    #[test]
    fn add_combines_entries() {
        let a = Cov3 {
            xx: 1.0,
            yy: 2.0,
            zz: 3.0,
            xy: 4.0,
            xz: 5.0,
            yz: 6.0,
        };
        let b = Cov3 {
            xx: 0.5,
            yy: 0.5,
            zz: 0.5,
            xy: 0.5,
            xz: 0.5,
            yz: 0.5,
        };
        let s = a + b;
        assert_abs_diff_eq!(s.xx, 1.5, epsilon = EPS);
        assert_abs_diff_eq!(s.yy, 2.5, epsilon = EPS);
        assert_abs_diff_eq!(s.zz, 3.5, epsilon = EPS);
        assert_abs_diff_eq!(s.xy, 4.5, epsilon = EPS);
        assert_abs_diff_eq!(s.xz, 5.5, epsilon = EPS);
        assert_abs_diff_eq!(s.yz, 6.5, epsilon = EPS);
    }

    #[test]
    fn mul_scales_all_entries() {
        let c = Cov3::diag(1.0, 2.0, 3.0);
        let s = c * 4.0;
        assert_abs_diff_eq!(s.xx, 4.0, epsilon = EPS);
        assert_abs_diff_eq!(s.yy, 8.0, epsilon = EPS);
        assert_abs_diff_eq!(s.zz, 12.0, epsilon = EPS);
        assert_abs_diff_eq!(s.xy, 0.0, epsilon = EPS);
    }

    #[test]
    fn display_contains_all_field_names() {
        let c = Cov3::diag(1.0, 2.0, 3.0);
        let s = format!("{c}");
        for name in ["xx", "yy", "zz", "xy", "xz", "yz"] {
            assert!(s.contains(name), "Display missing field {name}");
        }
    }

    // ------------------------------------------------------------------ //
    // Property-based tests                                                 //
    // ------------------------------------------------------------------ //

    prop_compose! {
        /// Generate a PSD Cov3 via Cholesky: Σ = L Lᵀ with L lower triangular.
        ///
        /// L = [[a, 0, 0], [b, c, 0], [d, e, f]]
        fn psd_cov3()(
            a in 0.01_f64..5.0,
            b in -2.0_f64..2.0,
            c in 0.01_f64..5.0,
            d in -2.0_f64..2.0,
            e in -2.0_f64..2.0,
            f in 0.01_f64..5.0,
        ) -> Cov3 {
            // Σ = L Lᵀ
            Cov3 {
                xx: a * a,
                yy: b * b + c * c,
                zz: d * d + e * e + f * f,
                xy: a * b,
                xz: a * d,
                yz: b * d + c * e,
            }
        }
    }

    proptest! {
        /// trace ≥ 0 for PSD matrices.
        #[test]
        fn trace_nonneg_for_psd(c in psd_cov3()) {
            prop_assert!(c.trace() >= 0.0);
        }

        /// bilinear(v, v) ≥ 0 for any v and PSD Σ.
        #[test]
        fn bilinear_quad_form_nonneg(
            c in psd_cov3(),
            vx in -5.0_f64..5.0,
            vy in -5.0_f64..5.0,
            vz in -5.0_f64..5.0,
        ) {
            let q = c.bilinear([vx, vy, vz], [vx, vy, vz]);
            prop_assert!(q >= -1e-10, "quadratic form negative: {q}");
        }

        /// bilinear(a, b) == bilinear(b, a) for symmetric Σ.
        #[test]
        fn bilinear_is_symmetric(
            c in psd_cov3(),
            ax in -3.0_f64..3.0, ay in -3.0_f64..3.0, az in -3.0_f64..3.0,
            bx in -3.0_f64..3.0, by in -3.0_f64..3.0, bz in -3.0_f64..3.0,
        ) {
            let ab = c.bilinear([ax, ay, az], [bx, by, bz]);
            let ba = c.bilinear([bx, by, bz], [ax, ay, az]);
            prop_assert!((ab - ba).abs() < 1e-10, "bilinear asymmetry: {ab} vs {ba}");
        }

        /// transform_j2 output has non-negative diagonal for PSD input.
        #[test]
        fn transform_j2_output_diagonal_nonneg(
            c in psd_cov3(),
            k00 in -3.0_f64..3.0, k01 in -3.0_f64..3.0, k02 in -3.0_f64..3.0,
            k10 in -3.0_f64..3.0, k11 in -3.0_f64..3.0, k12 in -3.0_f64..3.0,
        ) {
            let k = [[k00, k01, k02], [k10, k11, k12]];
            let out = c.transform_j2(k);
            prop_assert!(out.xx >= -1e-10, "out.xx negative: {}", out.xx);
            prop_assert!(out.yy >= -1e-10, "out.yy negative: {}", out.yy);
        }

        /// Add is commutative.
        #[test]
        fn add_commutative(a in psd_cov3(), b in psd_cov3()) {
            let ab = a + b;
            let ba = b + a;
            prop_assert!((ab.xx - ba.xx).abs() < 1e-14);
            prop_assert!((ab.yy - ba.yy).abs() < 1e-14);
            prop_assert!((ab.zz - ba.zz).abs() < 1e-14);
        }

        /// Scaling by s increases trace by factor s.
        #[test]
        fn mul_scales_trace(c in psd_cov3(), s in 0.01_f64..10.0) {
            let scaled = c * s;
            prop_assert!((scaled.trace() - c.trace() * s).abs() < 1e-10);
        }

        /// Scaling bilinear form: (s·Σ).bilinear(v,v) = s · Σ.bilinear(v,v).
        #[test]
        fn scaled_bilinear_homogeneous(
            c in psd_cov3(),
            s in 0.01_f64..10.0,
            vx in -3.0_f64..3.0, vy in -3.0_f64..3.0, vz in -3.0_f64..3.0,
        ) {
            let v = [vx, vy, vz];
            let scaled = c * s;
            let lhs = scaled.bilinear(v, v);
            let rhs = s * c.bilinear(v, v);
            prop_assert!((lhs - rhs).abs() < 1e-8, "lhs={lhs} rhs={rhs}");
        }

        /// Add is associative.
        #[test]
        fn add_associative(a in psd_cov3(), b in psd_cov3(), c in psd_cov3()) {
            let ab_c = (a + b) + c;
            let a_bc = a + (b + c);
            prop_assert!((ab_c.xx - a_bc.xx).abs() < 1e-12, "xx: {} vs {}", ab_c.xx, a_bc.xx);
            prop_assert!((ab_c.yy - a_bc.yy).abs() < 1e-12, "yy: {} vs {}", ab_c.yy, a_bc.yy);
            prop_assert!((ab_c.zz - a_bc.zz).abs() < 1e-12, "zz: {} vs {}", ab_c.zz, a_bc.zz);
        }
    }
}
