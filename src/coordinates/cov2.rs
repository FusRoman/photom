//! Symmetric 2×2 covariance matrix for tangent-plane error ellipses.
//!
//! ## Purpose
//!
//! This module provides [`Cov2`], a compact representation of a symmetric
//! positive semi-definite 2×2 matrix. It is the natural container for
//! astrometric uncertainties once they have been projected onto a local
//! tangent plane — i.e., after the sky covariance
//! $\Sigma_{\alpha\delta}$ has been transformed by the gnomonic Jacobian
//! into a planar covariance $\Sigma_{xy}$.
//!
//! ## Storage layout
//!
//! Only the three independent entries of the symmetric matrix are kept:
//!
//! $$\Sigma = \begin{pmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{xy} & \sigma_{yy} \end{pmatrix}
//! \quad\longleftrightarrow\quad (\sigma_{xx},\; \sigma_{yy},\; \sigma_{xy})$$
//!
//! This eliminates the possibility of accidentally constructing a
//! non-symmetric matrix and halves the memory footprint relative to a
//! full `[[f64; 2]; 2]`.
//!
//! ## Key operations
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`Cov2::diag`] | Diagonal covariance from independent variances |
//! | [`Cov2::isotropic`] | Isotropic $q \cdot I$ covariance |
//! | [`Cov2::from_equ`] | Build from the marginal errors of an [`EquCoord`] |
//! | [`Cov2::det`] / [`Cov2::trace`] | Scalar invariants |
//! | [`Cov2::inverse`] | Matrix inverse, or `None` if singular |
//! | [`Cov2::mahalanobis_sq`] | Quadratic form $v^\top \Sigma^{-1} v$ |
//! | [`Cov2::inflate_isotropic`] | Add isotropic process noise $q \cdot I$ |
//! | [`Cov2::lambda_max`] / [`Cov2::lambda_min`] | Eigenvalues (semi-axes of the error ellipse) |
//! | `+`, `* f64` | Component-wise addition and scalar scaling |

use std::{
    fmt::{Display, Formatter},
    ops::{Add, Mul},
};

use crate::coordinates::{cov3::Cov3, equatorial::EquCoord, gnomonic_projection::TangentVec};

/// Symmetric 2×2 covariance matrix.
///
/// Stores only the three independent entries of a symmetric positive
/// semi-definite matrix:
///
/// $$\Sigma = \begin{pmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{xy} & \sigma_{yy} \end{pmatrix}$$
///
/// Storing only the upper triangle eliminates the risk of
/// accidentally building a non-symmetric matrix, and halves the memory
/// footprint compared to a full `[[f64; 2]; 2]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cov2 {
    /// Variance along the first axis, $\sigma_{xx}$.
    pub xx: f64,
    /// Variance along the second axis, $\sigma_{yy}$.
    pub yy: f64,
    /// Covariance between the two axes, $\sigma_{xy} = \sigma_{yx}$.
    pub xy: f64,
}

impl Cov2 {
    /// Build a diagonal covariance from the marginal errors of an [`EquCoord`].
    ///
    /// $$\Sigma = \begin{pmatrix}
    ///     \sigma_\alpha^2 & 0 \\
    ///     0               & \sigma_\delta^2
    /// \end{pmatrix}$$
    ///
    /// # Arguments
    ///
    /// - `c` — Source equatorial coordinate whose `ra_error` and `dec_error`
    ///   fields (in radians) are squared to form the diagonal variances.
    ///
    /// # Returns
    ///
    /// A diagonal [`Cov2`] with `xx = ra_error²`, `yy = dec_error²`, `xy = 0`.
    ///
    /// # Notes
    ///
    /// - [`EquCoord`] carries only per-axis standard deviations; the
    ///   off-diagonal term is therefore set to zero.
    /// - No $\cos\delta$ rescaling is applied: the first axis is raw
    ///   right ascension.
    #[inline]
    pub fn from_equ(c: &EquCoord) -> Self {
        Self::diag(c.ra_error * c.ra_error, c.dec_error * c.dec_error)
    }

    /// Diagonal covariance from two independent variances.
    ///
    /// # Arguments
    ///
    /// - `var_x` — Variance along the first axis ($\sigma_{xx} \geq 0$).
    /// - `var_y` — Variance along the second axis ($\sigma_{yy} \geq 0$).
    ///
    /// # Returns
    ///
    /// A [`Cov2`] with `xx = var_x`, `yy = var_y`, `xy = 0`.
    #[inline]
    pub fn diag(var_x: f64, var_y: f64) -> Self {
        Self {
            xx: var_x,
            yy: var_y,
            xy: 0.0,
        }
    }

    /// Determinant $\det(\Sigma) = \sigma_{xx}\sigma_{yy} - \sigma_{xy}^2$.
    ///
    /// # Returns
    ///
    /// `f64` — The determinant. Non-negative for a positive semi-definite matrix.
    #[inline]
    pub fn det(&self) -> f64 {
        self.xx * self.yy - self.xy * self.xy
    }

    /// Trace $\mathrm{tr}(\Sigma) = \sigma_{xx} + \sigma_{yy}$.
    ///
    /// # Returns
    ///
    /// `f64` — The sum of the diagonal entries, equal to $\lambda_\max + \lambda_\min$.
    #[inline]
    pub fn trace(&self) -> f64 {
        self.xx + self.yy
    }

    /// Matrix inverse $\Sigma^{-1}$, or `None` if the matrix is singular.
    ///
    /// Singularity is detected when $|\det(\Sigma)| < \varepsilon_\text{machine}$
    /// (`f64::EPSILON`).
    ///
    /// # Returns
    ///
    /// `Option<Cov2>` — `Some(inv)` where `inv * self ≈ I`, or `None` if singular.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let d = self.det();
        if d.abs() < f64::EPSILON {
            return None;
        }
        let inv = 1.0 / d;
        Some(Self {
            xx: self.yy * inv,
            yy: self.xx * inv,
            xy: -self.xy * inv,
        })
    }

    /// Mahalanobis-style quadratic form $\mathbf{v}^\top \Sigma^{-1} \mathbf{v}$.
    ///
    /// # Arguments
    ///
    /// - `v` — A tangent-plane displacement [`TangentVec`] expressed in the same
    ///   frame and units as the covariance axes.
    ///
    /// # Returns
    ///
    /// `Option<f64>` — The non-negative scalar
    /// $\mathbf{v}^\top \Sigma^{-1} \mathbf{v}$, or `None` if the matrix is
    /// singular (see [`Cov2::inverse`]).
    #[inline]
    pub fn mahalanobis_sq(&self, v: TangentVec) -> Option<f64> {
        let inv = self.inverse()?;
        Some(inv.xx * v.dx * v.dx + 2.0 * inv.xy * v.dx * v.dy + inv.yy * v.dy * v.dy)
    }

    /// Zero covariance — all three entries set to zero.
    ///
    /// # Returns
    ///
    /// A [`Cov2`] with `xx = yy = xy = 0`.
    ///
    /// # Notes
    ///
    /// A zero covariance is singular; [`Cov2::inverse`] and
    /// [`Cov2::mahalanobis_sq`] will return `None` for such a matrix.
    #[inline]
    pub fn zero() -> Self {
        Self {
            xx: 0.0,
            yy: 0.0,
            xy: 0.0,
        }
    }

    /// Isotropic covariance $q \cdot I$.
    ///
    /// $$\Sigma = \begin{pmatrix} q & 0 \\ 0 & q \end{pmatrix}$$
    ///
    /// # Arguments
    ///
    /// - `q` — The common variance placed on both diagonal entries.
    ///
    /// # Returns
    ///
    /// A [`Cov2`] with `xx = yy = q`, `xy = 0`.
    ///
    /// # Notes
    ///
    /// Equivalent to `Cov2::diag(q, q)`, but the dedicated constructor
    /// documents the intent (e.g. isotropic process noise).
    #[inline]
    pub fn isotropic(q: f64) -> Self {
        Self::diag(q, q)
    }

    /// Generic quadratic form $\mathbf{v}^\top M \mathbf{v}$ where `self = M`.
    ///
    /// Computes the scalar:
    ///
    /// $$
    /// \mathbf{v}^\top M \mathbf{v} = M_{xx}\, v_x^2 + 2 M_{xy}\, v_x v_y + M_{yy}\, v_y^2
    /// $$
    ///
    /// Interpretation depends on what `self` represents:
    ///
    /// - If `self` is a **covariance** $\Sigma$, this returns the variance of
    ///   `v` projected along itself (useful to propagate a covariance along a
    ///   given direction).
    /// - If `self` is an **information matrix** $\Sigma^{-1}$, this returns the
    ///   **squared Mahalanobis distance** of `v` from the origin under the
    ///   underlying covariance $\Sigma$.
    ///
    /// # Arguments
    ///
    /// - `v` — A tangent vector expressed in the same basis as `self`.
    ///
    /// # Returns
    ///
    /// `f64` — Scalar value of the quadratic form. Non-negative when `self` is
    /// positive semi-definite (up to floating-point noise).
    #[inline]
    pub fn quad_form(&self, v: TangentVec) -> f64 {
        v.dot(*self * v)
    }

    /// Add an isotropic term $q \cdot I$ to the covariance.
    ///
    /// $$\Sigma' = \Sigma + q \cdot I$$
    ///
    /// Only the diagonal entries are affected; the off-diagonal
    /// correlation $\sigma_{xy}$ is preserved.
    ///
    /// # Arguments
    ///
    /// - `q` — Isotropic variance increment added to `xx` and `yy`.
    ///
    /// # Returns
    ///
    /// A new [`Cov2`] with `xx' = xx + q`, `yy' = yy + q`, `xy' = xy`.
    ///
    /// # Notes
    ///
    /// - Typically used to inject **isotropic process noise** when
    ///   propagating a covariance forward in time (Kalman-style
    ///   $P' = F P F^\top + Q$ with $Q = q I$).
    /// - The isotropy assumption is a modelling simplification: the
    ///   same variance is added to both axes, with no cross term.
    #[inline]
    pub fn inflate_isotropic(self, q: f64) -> Self {
        Self {
            xx: self.xx + q,
            yy: self.yy + q,
            xy: self.xy,
        }
    }

    /// Largest eigenvalue $\lambda_\max$ of the covariance.
    ///
    /// For a symmetric 2×2 matrix the eigenvalues are the roots of the
    /// characteristic polynomial:
    ///
    /// $$\lambda_{\pm} = \frac{\mathrm{tr}(\Sigma)}{2}
    ///   \pm \sqrt{\left(\frac{\sigma_{xx} - \sigma_{yy}}{2}\right)^2
    ///             + \sigma_{xy}^2}$$
    ///
    /// # Returns
    ///
    /// `f64` — The largest eigenvalue $\lambda_\max = \lambda_+$.
    /// $\sqrt{\lambda_\max}$ is the semi-major axis of the 1σ confidence
    /// ellipse.
    ///
    /// # Notes
    ///
    /// - For a positive semi-definite matrix the result is always
    ///   non-negative in exact arithmetic. Floating-point round-off
    ///   may produce a tiny negative value for near-singular inputs;
    ///   callers that need a radius should clamp with `.max(0.0)`.
    #[inline]
    pub fn lambda_max(&self) -> f64 {
        let half_tr = 0.5 * self.trace();
        let half_diff = 0.5 * (self.xx - self.yy);
        let disc = (half_diff * half_diff + self.xy * self.xy).sqrt();
        half_tr + disc
    }

    /// Smallest eigenvalue $\lambda_\min$ of the covariance.
    ///
    /// Uses the same characteristic-polynomial formula as [`Cov2::lambda_max`];
    /// see that method for the full derivation.
    ///
    /// # Returns
    ///
    /// `f64` — The smallest eigenvalue $\lambda_\min = \lambda_-$.
    /// $\sqrt{\lambda_\min}$ is the semi-minor axis of the 1σ confidence
    /// ellipse. Non-negative for a positive semi-definite matrix (up to
    /// floating-point round-off).
    #[inline]
    pub fn lambda_min(&self) -> f64 {
        let half_tr = 0.5 * self.trace();
        let half_diff = 0.5 * (self.xx - self.yy);
        let disc = (half_diff * half_diff + self.xy * self.xy).sqrt();
        half_tr - disc
    }

    /// Propagate this 2×2 covariance through a 3×2 Jacobian $J$.
    ///
    /// Computes the 3×3 output covariance
    ///
    /// $$\Sigma_{3\times3} = J\,\Sigma_{2\times2}\,J^\top$$
    ///
    /// which is the first-order propagation formula for a linear map
    /// $\mathbb{R}^2 \to \mathbb{R}^3$.
    ///
    /// # Arguments
    ///
    /// - `j` — The 3×2 Jacobian matrix stored in row-major order:
    ///   `j[i]` is the $i$-th row of $J$, `j[i][k]` is the entry $J_{ik}$.
    ///   Row 0 corresponds to the first output dimension ($x$), row 1 to $y$,
    ///   row 2 to $z$.
    ///
    /// # Returns
    ///
    /// A [`Cov3`] equal to $J \Sigma J^\top$.
    ///
    /// # Notes
    ///
    /// This is the algebraic dual of [`Cov3::transform_j2`]: for a pair of
    /// matching Jacobians $J$ (3×2) and $K = J^\top$-derived (2×3), the two
    /// operations compose as expected:
    ///
    /// $$K\bigl(J\,\Sigma_2\,J^\top\bigr)K^\top = (KJ)\,\Sigma_2\,(KJ)^\top.$$
    ///
    /// The key use-case is converting an equatorial covariance
    /// $\Sigma_{\alpha\delta}$ to a full Cartesian covariance $\Sigma_{xyz}$
    /// via the Jacobian $J$ of $(\alpha,\delta)\to(x,y,z)$.
    #[inline]
    pub fn transform_j3(&self, j: [[f64; 2]; 3]) -> Cov3 {
        // (J Σ Jᵀ)_{ij} = Σ_k Σ_l J_{ik} Σ_{kl} J_{jl}
        // For a 2×2 Σ with entries xx, yy, xy this simplifies to:
        //   row_quad(a, b) = a[0]*b[0]*xx + (a[0]*b[1]+a[1]*b[0])*xy + a[1]*b[1]*yy
        #[inline]
        fn row_quad(a: [f64; 2], b: [f64; 2], xx: f64, yy: f64, xy: f64) -> f64 {
            a[0] * b[0] * xx + (a[0] * b[1] + a[1] * b[0]) * xy + a[1] * b[1] * yy
        }
        Cov3 {
            xx: row_quad(j[0], j[0], self.xx, self.yy, self.xy),
            yy: row_quad(j[1], j[1], self.xx, self.yy, self.xy),
            zz: row_quad(j[2], j[2], self.xx, self.yy, self.xy),
            xy: row_quad(j[0], j[1], self.xx, self.yy, self.xy),
            xz: row_quad(j[0], j[2], self.xx, self.yy, self.xy),
            yz: row_quad(j[1], j[2], self.xx, self.yy, self.xy),
        }
    }

    /// Compute the lower-triangular Cholesky factor `L` such that
    /// $\Sigma = L L^\top$.
    ///
    /// This routine is specialized for symmetric 2×2 covariance matrices and
    /// is designed for **hot paths** (innovation whitening, covariance
    /// normalization, gating).
    ///
    /// # Numerical robustness
    ///
    /// The following safeguards are applied before factorization:
    /// - non-finite entries cause an immediate `None`,
    /// - `floor` must be finite and strictly positive,
    /// - diagonal terms are floored to at least `floor`,
    /// - positive-definiteness is checked via the Schur complement
    ///   $t = \sigma_{yy} - \sigma_{xy}^2 / \sigma_{xx}$, with the
    ///   requirement $t \geq \text{floor}$.
    ///
    /// If `self` is not numerically positive definite after flooring,
    /// `None` is returned.
    ///
    /// # Arguments
    ///
    /// - `floor` – Strictly positive minimum for diagonal terms and for the
    ///   Schur complement. Typical values: `1e-20` for radians² covariances,
    ///   or `1e-12` in more conservative settings.
    ///
    /// # Return
    ///
    /// - `Some(CholeskyLower2)` – Factor with $l_{00}, l_{11} > 0$ and all
    ///   entries finite.
    /// - `None` – If `self` contains non-finite entries, if `floor` is
    ///   non-finite or non-positive, or if `self` is not SPD after flooring.
    ///
    /// # Notes
    ///
    /// - For a symmetric 2×2 matrix, positive-definiteness is equivalent to
    ///   $\sigma_{xx} > 0$ and $\det(\Sigma) > 0$. The Schur-complement check
    ///   used here is algebraically equivalent.
    /// - If a *fallback* behavior (e.g. diagonal-only whitening) is desired,
    ///   it must be implemented at the call site when `None` is returned.
    ///
    /// # See also
    ///
    /// - [`Cov2::whiten_cholesky`] – Whiten a vector and obtain its
    ///   Mahalanobis squared distance in one pass.
    /// - [`Cov2::inverse`] – Exact inverse (no flooring), or `None` if singular.
    /// - [`Cov2::det`] – Determinant of the covariance.
    #[inline]
    pub fn cholesky_lower(&self, floor: f64) -> Option<CholeskyLower2> {
        if !self.xx.is_finite() || !self.yy.is_finite() || !self.xy.is_finite() {
            return None;
        }
        if !floor.is_finite() || floor <= 0.0 {
            return None;
        }

        let a = self.xx.max(floor);
        let d = self.yy.max(floor);
        let b = self.xy;

        // 2×2 Cholesky:
        //   l00 = sqrt(a)
        //   l10 = b / l00
        //   l11 = sqrt(d - l10²)
        let l00 = a.sqrt();
        if !l00.is_finite() || l00 <= 0.0 {
            return None;
        }

        let l10 = b / l00;
        let t = d - l10 * l10;
        if !t.is_finite() || t < floor {
            return None;
        }

        let l11 = t.sqrt();
        if !l11.is_finite() || l11 <= 0.0 {
            return None;
        }

        Some(CholeskyLower2 { l00, l10, l11 })
    }

    /// Whiten a tangent-plane vector using the Cholesky factor of `self`.
    ///
    /// Given $\Sigma = L L^\top$, this computes
    ///
    /// $$\mathbf{z} = L^{-1} \mathbf{v}$$
    ///
    /// by forward substitution, together with its squared norm
    ///
    /// $$\|\mathbf{z}\|^2 = \mathbf{v}^\top \Sigma^{-1} \mathbf{v},$$
    ///
    /// which is the squared Mahalanobis distance of $\mathbf{v}$ under
    /// $\Sigma$.
    ///
    /// # Numerical safeguards
    ///
    /// - Delegates SPD checks and diagonal flooring to
    ///   [`Cov2::cholesky_lower`].
    /// - Returns `None` when the factorization fails or produces non-finite
    ///   whitened components.
    /// - When `Some` is returned, both the whitened vector and its squared
    ///   norm are finite; the squared norm is clamped to be non-negative.
    ///
    /// # Arguments
    ///
    /// - `v` – Vector to whiten, expressed in the same tangent-plane frame
    ///   as `self`.
    /// - `floor` – Minimum diagonal value used to regularize the factorization
    ///   (forwarded to [`Cov2::cholesky_lower`]).
    ///
    /// # Return
    ///
    /// - `Some((z, z_sq))` – Whitened vector $\mathbf{z} = L^{-1}\mathbf{v}$
    ///   and its squared norm $\|\mathbf{z}\|^2$.
    /// - `None` – If `self` is not SPD (even after flooring) or the
    ///   factorization is numerically invalid.
    ///
    /// # Notes
    ///
    /// - This is the numerically preferred way to obtain both a whitened
    ///   residual and a Mahalanobis $\chi^2$ in a single pass: it avoids
    ///   forming $\Sigma^{-1}$ explicitly and keeps the two quantities
    ///   algebraically consistent.
    #[inline]
    pub fn whiten_cholesky(&self, v: TangentVec, floor: f64) -> Option<(TangentVec, f64)> {
        let l = self.cholesky_lower(floor)?;
        let z = l.solve_forward(v);

        if !z.dx.is_finite() || !z.dy.is_finite() {
            return None;
        }

        let z_sq = (z.dx * z.dx + z.dy * z.dy).max(0.0);
        Some((z, z_sq))
    }

    /// Diagonal whitening: $\mathrm{diag}(\Sigma)^{-1/2}\, \mathbf{v}$.
    ///
    /// Ignores off-diagonal correlation. Cheap and robust; well-defined even
    /// when $\Sigma$ is near-singular off-diagonally.
    ///
    /// # Arguments
    ///
    /// - `v` – Vector to whiten, expressed in the same tangent-plane frame as `self`.
    /// - `floor` – Minimum diagonal value used to regularize the whitening.
    ///   Typical values: `1e-20` for radians² covariances, or `1e-12` in more conservative settings.
    ///
    /// # Returns
    ///
    /// - `TangentVec` – The diagonally whitened vector. Finite for all finite inputs
    ///
    /// # Notes
    ///
    /// - This is a fallback whitening method that can be used when the full Cholesky factorization fails
    ///   (e.g. due to near-singularity or non-SPD inputs).
    ///   It is computationally cheaper and more robust than the full Cholesky whitening,
    ///   but it ignores off-diagonal correlations and may produce a poor approximation to
    ///   the true Mahalanobis whitening when those correlations are large.
    #[inline]
    pub fn whiten_diag(&self, v: TangentVec, floor: f64) -> TangentVec {
        let sx = self.xx.max(floor).sqrt();
        let sy = self.yy.max(floor).sqrt();

        let finite_or_zero = |x: f64| if x.is_finite() { x } else { 0.0 };

        TangentVec {
            dx: finite_or_zero(v.dx / sx),
            dy: finite_or_zero(v.dy / sy),
        }
    }
}

impl Add for Cov2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            xx: self.xx + rhs.xx,
            yy: self.yy + rhs.yy,
            xy: self.xy + rhs.xy,
        }
    }
}

impl Add for &Cov2 {
    type Output = Cov2;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl Mul<f64> for Cov2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            xx: self.xx * rhs,
            yy: self.yy * rhs,
            xy: self.xy * rhs,
        }
    }
}

impl Mul<f64> for &Cov2 {
    type Output = Cov2;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        *self * rhs
    }
}

impl Display for Cov2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cov2 {{")?;
        writeln!(f, "  xx : {:.6e}", self.xx)?;
        writeln!(f, "  yy : {:.6e}", self.yy)?;
        writeln!(f, "  xy : {:.6e}", self.xy)?;
        write!(f, "}}")
    }
}

/// Lower-triangular Cholesky factor of a symmetric 2×2 SPD matrix.
///
/// Stored as:
/// ```text
/// L = [ l00  0  ]
///     [ l10  l11]
/// ```
/// with the convention $\Sigma = L L^\top$.
#[derive(Debug, Clone, Copy)]
pub struct CholeskyLower2 {
    pub l00: f64,
    pub l10: f64,
    pub l11: f64,
}

impl CholeskyLower2 {
    /// Solve $L \mathbf{z} = \mathbf{v}$ by forward substitution.
    #[inline]
    pub fn solve_forward(&self, v: TangentVec) -> TangentVec {
        let vx = v.dx;
        let vy = v.dy;
        let z1 = vx / self.l00;
        let z2 = (vy - self.l10 * z1) / self.l11;
        TangentVec { dx: z1, dy: z2 }
    }
}

#[cfg(test)]
mod cov2_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    const EPS: f64 = 1e-12;

    // ------------------------------------------------------------------ //
    // Constructor / accessors                                              //
    // ------------------------------------------------------------------ //

    #[test]
    fn diag_sets_xy_to_zero() {
        let c = Cov2::diag(4.0, 9.0);
        assert_abs_diff_eq!(c.xx, 4.0, epsilon = EPS);
        assert_abs_diff_eq!(c.yy, 9.0, epsilon = EPS);
        assert_abs_diff_eq!(c.xy, 0.0, epsilon = EPS);
    }

    #[test]
    fn zero_all_entries_zero() {
        let c = Cov2::zero();
        assert_eq!(c.xx, 0.0);
        assert_eq!(c.yy, 0.0);
        assert_eq!(c.xy, 0.0);
    }

    #[test]
    fn isotropic_equals_diag_q_q() {
        let q = 3.7_f64;
        let iso = Cov2::isotropic(q);
        let diag = Cov2::diag(q, q);
        assert_eq!(iso, diag);
    }

    #[test]
    fn from_equ_uses_squared_errors() {
        use crate::coordinates::equatorial::EquCoord;
        let c = EquCoord::new(0.0, 0.01, 0.0, 0.02);
        let cov = Cov2::from_equ(&c);
        assert_abs_diff_eq!(cov.xx, 0.01_f64 * 0.01, epsilon = EPS);
        assert_abs_diff_eq!(cov.yy, 0.02_f64 * 0.02, epsilon = EPS);
        assert_abs_diff_eq!(cov.xy, 0.0, epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // det / trace                                                          //
    // ------------------------------------------------------------------ //

    #[test]
    fn det_diagonal() {
        let c = Cov2::diag(3.0, 5.0);
        assert_abs_diff_eq!(c.det(), 15.0, epsilon = EPS);
    }

    #[test]
    fn det_full() {
        let c = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        // 4*9 - 2^2 = 32
        assert_abs_diff_eq!(c.det(), 32.0, epsilon = EPS);
    }

    #[test]
    fn trace_sum_of_diagonal() {
        let c = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        assert_abs_diff_eq!(c.trace(), 13.0, epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // inverse                                                              //
    // ------------------------------------------------------------------ //

    #[test]
    fn inverse_diagonal() {
        let c = Cov2::diag(4.0, 9.0);
        let inv = c.inverse().expect("invertible");
        assert_abs_diff_eq!(inv.xx, 1.0 / 4.0, epsilon = EPS);
        assert_abs_diff_eq!(inv.yy, 1.0 / 9.0, epsilon = EPS);
        assert_abs_diff_eq!(inv.xy, 0.0, epsilon = EPS);
    }

    #[test]
    fn inverse_full() {
        let c = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        let inv = c.inverse().expect("invertible");
        // verify C * C^-1 ≈ I via quadratic form check
        // (c*inv).xx = c.xx*inv.xx + c.xy*inv.xy (row-col product)
        let a00 = c.xx * inv.xx + c.xy * inv.xy;
        let a01 = c.xx * inv.xy + c.xy * inv.yy;
        let a10 = c.xy * inv.xx + c.yy * inv.xy;
        let a11 = c.xy * inv.xy + c.yy * inv.yy;
        assert_abs_diff_eq!(a00, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a01, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a10, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a11, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn inverse_singular_returns_none() {
        // xx*yy - xy^2 = 1*1 - 1^2 = 0
        let c = Cov2 {
            xx: 1.0,
            yy: 1.0,
            xy: 1.0,
        };
        assert!(c.inverse().is_none());
    }

    // ------------------------------------------------------------------ //
    // mahalanobis_sq                                                       //
    // ------------------------------------------------------------------ //

    #[test]
    fn mahalanobis_sq_isotropic() {
        // Σ = q*I  →  v^T Σ^-1 v = (vx^2+vy^2)/q
        let q = 4.0_f64;
        let c = Cov2::isotropic(q);
        let v = TangentVec { dx: 3.0, dy: 4.0 };
        let got = c.mahalanobis_sq(v).unwrap();
        let expected = (3.0_f64 * 3.0 + 4.0 * 4.0) / q;
        assert_abs_diff_eq!(got, expected, epsilon = 1e-10);
    }

    #[test]
    fn mahalanobis_sq_singular_returns_none() {
        let c = Cov2 {
            xx: 1.0,
            yy: 1.0,
            xy: 1.0,
        };
        assert!(c.mahalanobis_sq(TangentVec { dx: 1.0, dy: 0.0 }).is_none());
    }

    // ------------------------------------------------------------------ //
    // inflate_isotropic                                                    //
    // ------------------------------------------------------------------ //

    #[test]
    fn inflate_isotropic_adds_to_diagonal_only() {
        let c = Cov2 {
            xx: 1.0,
            yy: 2.0,
            xy: 0.5,
        };
        let inflated = c.inflate_isotropic(3.0);
        assert_abs_diff_eq!(inflated.xx, 4.0, epsilon = EPS);
        assert_abs_diff_eq!(inflated.yy, 5.0, epsilon = EPS);
        assert_abs_diff_eq!(inflated.xy, 0.5, epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // eigenvalues                                                          //
    // ------------------------------------------------------------------ //

    #[test]
    fn lambda_max_min_diagonal() {
        let c = Cov2::diag(3.0, 7.0);
        assert_abs_diff_eq!(c.lambda_max(), 7.0, epsilon = EPS);
        assert_abs_diff_eq!(c.lambda_min(), 3.0, epsilon = EPS);
    }

    #[test]
    fn lambda_max_min_isotropic() {
        let c = Cov2::isotropic(5.0);
        assert_abs_diff_eq!(c.lambda_max(), 5.0, epsilon = EPS);
        assert_abs_diff_eq!(c.lambda_min(), 5.0, epsilon = EPS);
    }

    #[test]
    fn lambda_max_gte_lambda_min() {
        let c = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        assert!(c.lambda_max() >= c.lambda_min());
    }

    #[test]
    fn eigenvalues_sum_equals_trace() {
        let c = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        assert_abs_diff_eq!(c.lambda_max() + c.lambda_min(), c.trace(), epsilon = EPS);
    }

    #[test]
    fn eigenvalues_product_equals_det() {
        let c = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        assert_abs_diff_eq!(c.lambda_max() * c.lambda_min(), c.det(), epsilon = 1e-10);
    }

    // ------------------------------------------------------------------ //
    // Arithmetic ops                                                       //
    // ------------------------------------------------------------------ //

    #[test]
    fn add_combines_entries() {
        let a = Cov2 {
            xx: 1.0,
            yy: 2.0,
            xy: 3.0,
        };
        let b = Cov2 {
            xx: 4.0,
            yy: 5.0,
            xy: 6.0,
        };
        let s = a + b;
        assert_abs_diff_eq!(s.xx, 5.0, epsilon = EPS);
        assert_abs_diff_eq!(s.yy, 7.0, epsilon = EPS);
        assert_abs_diff_eq!(s.xy, 9.0, epsilon = EPS);
    }

    #[test]
    fn mul_scales_all_entries() {
        let c = Cov2 {
            xx: 1.0,
            yy: 2.0,
            xy: 3.0,
        };
        let s = c * 2.0;
        assert_abs_diff_eq!(s.xx, 2.0, epsilon = EPS);
        assert_abs_diff_eq!(s.yy, 4.0, epsilon = EPS);
        assert_abs_diff_eq!(s.xy, 6.0, epsilon = EPS);
    }

    #[test]
    fn display_contains_entries() {
        let c = Cov2::diag(1.0, 2.0);
        let s = format!("{}", c);
        assert!(s.contains("xx"));
        assert!(s.contains("yy"));
        assert!(s.contains("xy"));
    }

    // ------------------------------------------------------------------ //
    // transform_j3                                                         //
    // ------------------------------------------------------------------ //

    #[test]
    fn transform_j3_identity_gives_embedded_cov2() {
        // J = [[1,0],[0,1],[0,0]]  →  J Σ Jᵀ should embed Σ in upper-left 2×2
        let cov = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        let j = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let out = cov.transform_j3(j);
        assert_abs_diff_eq!(out.xx, cov.xx, epsilon = EPS);
        assert_abs_diff_eq!(out.yy, cov.yy, epsilon = EPS);
        assert_abs_diff_eq!(out.zz, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(out.xy, cov.xy, epsilon = EPS);
        assert_abs_diff_eq!(out.xz, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(out.yz, 0.0, epsilon = EPS);
    }

    #[test]
    fn transform_j3_zero_cov_gives_zero_output() {
        let cov = Cov2::zero();
        let j = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let out = cov.transform_j3(j);
        assert_eq!(out.xx, 0.0);
        assert_eq!(out.yy, 0.0);
        assert_eq!(out.zz, 0.0);
        assert_eq!(out.xy, 0.0);
        assert_eq!(out.xz, 0.0);
        assert_eq!(out.yz, 0.0);
    }

    #[test]
    fn transform_j3_diagonal_input_matches_manual() {
        // Σ = diag(4, 9); J = [[1,1],[1,-1],[0,1]]
        // (J Σ Jᵀ)_xx = 1*4 + 1*9 = 13
        // (J Σ Jᵀ)_yy = 1*4 + 1*9 = 13
        // (J Σ Jᵀ)_zz = 0*4 + 1*9 = 9
        // (J Σ Jᵀ)_xy = 1*4 + (-1)*9 = -5
        // (J Σ Jᵀ)_xz = 0*4 + 1*9 = 9
        // (J Σ Jᵀ)_yz = 0*4 + (-1)*9 = -9
        let cov = Cov2::diag(4.0, 9.0);
        let j = [[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]];
        let out = cov.transform_j3(j);
        assert_abs_diff_eq!(out.xx, 13.0, epsilon = EPS);
        assert_abs_diff_eq!(out.yy, 13.0, epsilon = EPS);
        assert_abs_diff_eq!(out.zz, 9.0, epsilon = EPS);
        assert_abs_diff_eq!(out.xy, -5.0, epsilon = EPS);
        assert_abs_diff_eq!(out.xz, 9.0, epsilon = EPS);
        assert_abs_diff_eq!(out.yz, -9.0, epsilon = EPS);
    }

    #[test]
    fn transform_j3_output_is_symmetric() {
        // The output must be symmetric by construction
        let cov = Cov2 {
            xx: 3.0,
            yy: 7.0,
            xy: 1.5,
        };
        let j = [[0.5, -1.0], [0.2, 0.8], [-0.3, 0.1]];
        let out = cov.transform_j3(j);
        // Off-diagonal entries must equal their symmetric counterparts
        // (they are computed independently; check they match via bilinear symmetry)
        let xy_check = {
            let a = j[0];
            let b = j[1];
            a[0] * b[0] * cov.xx + (a[0] * b[1] + a[1] * b[0]) * cov.xy + a[1] * b[1] * cov.yy
        };
        assert_abs_diff_eq!(out.xy, xy_check, epsilon = EPS);
    }

    #[test]
    fn transform_j3_then_transform_j2_round_trips() {
        // For a full-rank J and matching K=Jᵀ in sense: K J = 2×2 matrix,
        // check that Cov3::transform_j2(K) of transform_j3(J) gives (K J) Σ (K J)ᵀ.
        // Use J = [[1,0],[0,1],[0,0]], K = [[1,0,0],[0,1,0]]
        // Then K J = I₂, so round-trip should recover Σ exactly.
        let cov = Cov2 {
            xx: 5.0,
            yy: 3.0,
            xy: 1.0,
        };
        let j = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let k = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let cov3 = cov.transform_j3(j);
        let recovered = cov3.transform_j2(k);
        assert_abs_diff_eq!(recovered.xx, cov.xx, epsilon = EPS);
        assert_abs_diff_eq!(recovered.yy, cov.yy, epsilon = EPS);
        assert_abs_diff_eq!(recovered.xy, cov.xy, epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // Property-based tests                                                 //
    // ------------------------------------------------------------------ //

    prop_compose! {
        /// Generate a PSD Cov2 via Cholesky: Σ = L Lᵀ with L lower triangular.
        fn psd_cov2()(a in 0.01_f64..10.0, b in -5.0_f64..5.0, c in 0.01_f64..10.0)
            -> Cov2
        {
            // L = [[a, 0], [b, c]]  →  Σ = L Lᵀ = [[a²,ab],[ab, b²+c²]]
            Cov2 { xx: a * a, yy: b * b + c * c, xy: a * b }
        }
    }

    proptest! {
        /// det(Σ) ≥ 0 for any PSD matrix.
        #[test]
        fn det_nonneg_for_psd(cov in psd_cov2()) {
            prop_assert!(cov.det() >= -1e-12);
        }

        /// lambda_max ≥ lambda_min for PSD matrices.
        #[test]
        fn lambda_ordering_for_psd(cov in psd_cov2()) {
            prop_assert!(cov.lambda_max() >= cov.lambda_min() - 1e-12);
        }

        /// tr(Σ) = λ_max + λ_min.
        #[test]
        fn trace_equals_eigenvalue_sum(cov in psd_cov2()) {
            prop_assert!((cov.trace() - (cov.lambda_max() + cov.lambda_min())).abs() < 1e-10);
        }

        /// det(Σ) = λ_max * λ_min.
        #[test]
        fn det_equals_eigenvalue_product(cov in psd_cov2()) {
            prop_assert!((cov.det() - cov.lambda_max() * cov.lambda_min()).abs() < 1e-10);
        }

        /// inflate_isotropic(q) with q ≥ 0 increases lambda_max.
        #[test]
        fn inflate_increases_lambda_max(cov in psd_cov2(), q in 0.0_f64..10.0) {
            prop_assert!(cov.inflate_isotropic(q).lambda_max() >= cov.lambda_max() - 1e-12);
        }

        /// For a non-singular PSD Cov2, C * C^-1 should produce identity.
        #[test]
        fn inverse_correctness(cov in psd_cov2()) {
            if let Some(inv) = cov.inverse() {
                let a00 = cov.xx * inv.xx + cov.xy * inv.xy;
                let a11 = cov.xy * inv.xy + cov.yy * inv.yy;
                prop_assert!((a00 - 1.0).abs() < 1e-8);
                prop_assert!((a11 - 1.0).abs() < 1e-8);
            }
        }

        /// Mahalanobis distance is non-negative for PSD matrices.
        #[test]
        fn mahalanobis_sq_nonneg(cov in psd_cov2(), vx in -10.0_f64..10.0, vy in -10.0_f64..10.0) {
            let v = TangentVec { dx: vx, dy: vy };
            if let Some(m) = cov.mahalanobis_sq(v) {
                prop_assert!(m >= -1e-10);
            }
        }

        /// transform_j3 output has non-negative diagonal for PSD input.
        #[test]
        fn transform_j3_output_diagonal_nonneg(
            cov in psd_cov2(),
            j00 in -3.0_f64..3.0, j01 in -3.0_f64..3.0,
            j10 in -3.0_f64..3.0, j11 in -3.0_f64..3.0,
            j20 in -3.0_f64..3.0, j21 in -3.0_f64..3.0,
        ) {
            let j = [[j00, j01], [j10, j11], [j20, j21]];
            let out = cov.transform_j3(j);
            prop_assert!(out.xx >= -1e-10, "out.xx negative: {}", out.xx);
            prop_assert!(out.yy >= -1e-10, "out.yy negative: {}", out.yy);
            prop_assert!(out.zz >= -1e-10, "out.zz negative: {}", out.zz);
        }
    }

    // ------------------------------------------------------------------ //
    // quad_form                                                            //
    // ------------------------------------------------------------------ //

    #[test]
    fn quad_form_isotropic() {
        // M = q*I  →  vᵀ M v = q * (vx² + vy²)
        let q = 3.0_f64;
        let m = Cov2::isotropic(q);
        let v = TangentVec { dx: 3.0, dy: 4.0 };
        assert_abs_diff_eq!(m.quad_form(v), q * 25.0, epsilon = EPS);
    }

    #[test]
    fn quad_form_zero_cov_is_zero() {
        let m = Cov2::zero();
        let v = TangentVec { dx: 5.0, dy: 7.0 };
        assert_abs_diff_eq!(m.quad_form(v), 0.0, epsilon = EPS);
    }

    #[test]
    fn quad_form_full() {
        // M = [[4,2],[2,9]]; v = [1, 1]
        // vᵀ M v = 4 + 2 + 2 + 9 = 17
        let m = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        let v = TangentVec { dx: 1.0, dy: 1.0 };
        assert_abs_diff_eq!(m.quad_form(v), 17.0, epsilon = EPS);
    }

    // ------------------------------------------------------------------ //
    // cholesky_lower / CholeskyLower2::solve_forward                      //
    // ------------------------------------------------------------------ //

    #[test]
    fn cholesky_lower_diagonal() {
        // Σ = diag(4, 9)  →  L = diag(2, 3)
        let c = Cov2::diag(4.0, 9.0);
        let l = c.cholesky_lower(1e-20).expect("SPD diagonal");
        assert_abs_diff_eq!(l.l00, 2.0, epsilon = EPS);
        assert_abs_diff_eq!(l.l10, 0.0, epsilon = EPS);
        assert_abs_diff_eq!(l.l11, 3.0, epsilon = EPS);
    }

    #[test]
    fn cholesky_lower_reconstructs_cov() {
        // Σ = L Lᵀ check: [[l00²,  l00*l10],[l00*l10, l10²+l11²]]
        let c = Cov2 {
            xx: 4.0,
            yy: 5.0,
            xy: 2.0,
        };
        let l = c.cholesky_lower(1e-20).expect("SPD");
        assert_abs_diff_eq!(l.l00 * l.l00, c.xx, epsilon = 1e-10);
        assert_abs_diff_eq!(l.l10 * l.l10 + l.l11 * l.l11, c.yy, epsilon = 1e-10);
        assert_abs_diff_eq!(l.l00 * l.l10, c.xy, epsilon = 1e-10);
    }

    #[test]
    fn cholesky_lower_singular_returns_none() {
        // Singular: det = 0
        let c = Cov2 {
            xx: 1.0,
            yy: 1.0,
            xy: 1.0,
        };
        assert!(c.cholesky_lower(1e-20).is_none());
    }

    #[test]
    fn cholesky_lower_nonfinite_returns_none() {
        let c = Cov2 {
            xx: f64::NAN,
            yy: 1.0,
            xy: 0.0,
        };
        assert!(c.cholesky_lower(1e-20).is_none());
    }

    #[test]
    fn cholesky_lower_zero_floor_returns_none() {
        let c = Cov2::isotropic(1.0);
        assert!(c.cholesky_lower(0.0).is_none());
    }

    #[test]
    fn cholesky_lower_negative_floor_returns_none() {
        let c = Cov2::isotropic(1.0);
        assert!(c.cholesky_lower(-1e-10).is_none());
    }

    #[test]
    fn solve_forward_diagonal() {
        // L = diag(2, 3); solve L z = v  →  z = [vx/2, vy/3]
        let c = Cov2::diag(4.0, 9.0);
        let l = c.cholesky_lower(1e-20).unwrap();
        let v = TangentVec { dx: 6.0, dy: 9.0 };
        let z = l.solve_forward(v);
        assert_abs_diff_eq!(z.dx, 3.0, epsilon = EPS);
        assert_abs_diff_eq!(z.dy, 3.0, epsilon = EPS);
    }

    #[test]
    fn solve_forward_inverts_cholesky() {
        // L z = v  →  z = L^{-1} v; then L z should recover v
        let c = Cov2 {
            xx: 4.0,
            yy: 5.0,
            xy: 2.0,
        };
        let l = c.cholesky_lower(1e-20).unwrap();
        let v = TangentVec { dx: 1.0, dy: 2.0 };
        let z = l.solve_forward(v);
        // Verify L * z == v
        let lz_x = l.l00 * z.dx;
        let lz_y = l.l10 * z.dx + l.l11 * z.dy;
        assert_abs_diff_eq!(lz_x, v.dx, epsilon = 1e-10);
        assert_abs_diff_eq!(lz_y, v.dy, epsilon = 1e-10);
    }

    // ------------------------------------------------------------------ //
    // whiten_cholesky                                                      //
    // ------------------------------------------------------------------ //

    #[test]
    fn whiten_cholesky_isotropic() {
        // Σ = q*I  →  z = v/sqrt(q); z_sq = (vx²+vy²)/q
        let q = 4.0_f64;
        let c = Cov2::isotropic(q);
        let v = TangentVec { dx: 2.0, dy: 0.0 };
        let (z, z_sq) = c.whiten_cholesky(v, 1e-20).expect("SPD");
        assert_abs_diff_eq!(z.dx, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(z.dy, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(z_sq, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn whiten_cholesky_z_sq_equals_mahalanobis() {
        // For a valid SPD matrix, z_sq must equal mahalanobis_sq
        let c = Cov2 {
            xx: 4.0,
            yy: 9.0,
            xy: 2.0,
        };
        let v = TangentVec { dx: 1.0, dy: 2.0 };
        let (_, z_sq) = c.whiten_cholesky(v, 1e-20).unwrap();
        let mah = c.mahalanobis_sq(v).unwrap();
        assert_abs_diff_eq!(z_sq, mah, epsilon = 1e-8);
    }

    #[test]
    fn whiten_cholesky_singular_returns_none() {
        let c = Cov2 {
            xx: 1.0,
            yy: 1.0,
            xy: 1.0,
        };
        assert!(
            c.whiten_cholesky(TangentVec { dx: 1.0, dy: 0.0 }, 1e-20)
                .is_none()
        );
    }

    #[test]
    fn whiten_cholesky_z_sq_nonneg() {
        let c = Cov2::isotropic(1.0);
        let v = TangentVec { dx: -3.0, dy: -4.0 };
        let (_, z_sq) = c.whiten_cholesky(v, 1e-20).unwrap();
        assert!(z_sq >= 0.0);
    }

    // ------------------------------------------------------------------ //
    // whiten_diag                                                          //
    // ------------------------------------------------------------------ //

    #[test]
    fn whiten_diag_isotropic() {
        // Σ = q*I  →  z = v / sqrt(q)
        let q = 9.0_f64;
        let c = Cov2::isotropic(q);
        let v = TangentVec { dx: 3.0, dy: 6.0 };
        let z = c.whiten_diag(v, 1e-20);
        assert_abs_diff_eq!(z.dx, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(z.dy, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn whiten_diag_independent_axes() {
        // Each axis is whitened independently; off-diagonal xy is ignored.
        let c = Cov2 {
            xx: 4.0,
            yy: 16.0,
            xy: 100.0,
        };
        let v = TangentVec { dx: 2.0, dy: 4.0 };
        let z = c.whiten_diag(v, 1e-20);
        assert_abs_diff_eq!(z.dx, 1.0, epsilon = 1e-10); // 2 / sqrt(4)
        assert_abs_diff_eq!(z.dy, 1.0, epsilon = 1e-10); // 4 / sqrt(16)
    }

    #[test]
    fn whiten_diag_zero_variance_uses_floor() {
        // If variance is 0, floor is used instead, result is finite.
        let c = Cov2::zero();
        let v = TangentVec { dx: 1.0, dy: 1.0 };
        let z = c.whiten_diag(v, 1.0);
        assert!(z.dx.is_finite());
        assert!(z.dy.is_finite());
    }

    // ------------------------------------------------------------------ //
    // Additional property-based tests                                      //
    // ------------------------------------------------------------------ //

    proptest! {
        /// cholesky_lower succeeds for PSD matrices and L Lᵀ ≈ Σ.
        #[test]
        fn cholesky_reconstructs_psd(cov in psd_cov2()) {
            let l = cov.cholesky_lower(1e-12).expect("PSD should decompose");
            // Reconstruct: Σ' = L Lᵀ
            let xx = l.l00 * l.l00;
            let xy = l.l10 * l.l00;
            let yy = l.l10 * l.l10 + l.l11 * l.l11;
            prop_assert!((xx - cov.xx).abs() < 1e-8, "xx mismatch");
            prop_assert!((xy - cov.xy).abs() < 1e-8, "xy mismatch");
            prop_assert!((yy - cov.yy).abs() < 1e-8, "yy mismatch");
        }

        /// cholesky_lower diagonal entries are positive for PSD matrices.
        #[test]
        fn cholesky_diagonal_positive(cov in psd_cov2()) {
            let l = cov.cholesky_lower(1e-12).expect("PSD should decompose");
            prop_assert!(l.l00 > 0.0, "l00 non-positive");
            prop_assert!(l.l11 > 0.0, "l11 non-positive");
        }

        /// whiten_cholesky z² ≈ Mahalanobis distance for PSD matrices.
        #[test]
        fn proptest_whiten_cholesky_z_sq_equals_mahalanobis(
            cov in psd_cov2(),
            vx in -5.0_f64..5.0,
            vy in -5.0_f64..5.0,
        ) {
            let v = TangentVec { dx: vx, dy: vy };
            if let Some((_z, z_sq)) = cov.whiten_cholesky(v, 1e-12)
                && let Some(mah) = cov.mahalanobis_sq(v) {
                    prop_assert!((z_sq - mah).abs() < 1e-8, "z² {} ≠ mah {}", z_sq, mah);
                }
        }

        /// whiten_cholesky z² is non-negative for PSD matrices.
        #[test]
        fn proptest_whiten_cholesky_z_sq_nonneg(
            cov in psd_cov2(),
            vx in -5.0_f64..5.0,
            vy in -5.0_f64..5.0,
        ) {
            let v = TangentVec { dx: vx, dy: vy };
            if let Some((_z, z_sq)) = cov.whiten_cholesky(v, 1e-12) {
                prop_assert!(z_sq >= -1e-10);
            }
        }

        /// whiten_diag output is finite for PSD matrices with positive floor.
        #[test]
        fn whiten_diag_finite_for_psd(
            cov in psd_cov2(),
            vx in -5.0_f64..5.0,
            vy in -5.0_f64..5.0,
        ) {
            let v = TangentVec { dx: vx, dy: vy };
            let z = cov.whiten_diag(v, 1e-20);
            prop_assert!(z.dx.is_finite(), "z.dx is not finite");
            prop_assert!(z.dy.is_finite(), "z.dy is not finite");
        }

        /// quad_form is non-negative for PSD matrices.
        #[test]
        fn quad_form_nonneg_for_psd(
            cov in psd_cov2(),
            vx in -10.0_f64..10.0,
            vy in -10.0_f64..10.0,
        ) {
            let v = TangentVec { dx: vx, dy: vy };
            prop_assert!(cov.quad_form(v) >= -1e-10);
        }

        /// quad_form scales quadratically: (αv)ᵀ Σ (αv) = α² vᵀ Σ v.
        #[test]
        fn quad_form_homogeneous(
            cov in psd_cov2(),
            vx in -5.0_f64..5.0,
            vy in -5.0_f64..5.0,
            alpha in -3.0_f64..3.0,
        ) {
            let v = TangentVec { dx: vx, dy: vy };
            let av = TangentVec { dx: alpha * vx, dy: alpha * vy };
            let ratio = cov.quad_form(av);
            let expected = alpha * alpha * cov.quad_form(v);
            prop_assert!((ratio - expected).abs() < 1e-8, "{} ≠ {}", ratio, expected);
        }

        /// solve_forward inverts cholesky_lower: L (L⁻¹ v) ≈ v.
        #[test]
        fn solve_forward_inverts_cholesky_lower(
            cov in psd_cov2(),
            vx in -5.0_f64..5.0,
            vy in -5.0_f64..5.0,
        ) {
            let v = TangentVec { dx: vx, dy: vy };
            let l = cov.cholesky_lower(1e-12).expect("PSD should decompose");
            let z = l.solve_forward(v);
            // Recover: L z = v  ↔  z = L⁻¹ v
            let rx = l.l00 * z.dx;
            let ry = l.l10 * z.dx + l.l11 * z.dy;
            prop_assert!((rx - vx).abs() < 1e-8, "rx {} ≠ vx {}", rx, vx);
            prop_assert!((ry - vy).abs() < 1e-8, "ry {} ≠ vy {}", ry, vy);
        }
    }
}
