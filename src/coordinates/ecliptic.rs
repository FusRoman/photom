//! Ecliptic coordinate representation with astrometric uncertainties.
//!
//! This module provides [`EclipticCoord`] and [`EclipticCoordCov`], the
//! ecliptic-frame counterparts of [`EquCoord`] and [`EquCoordCov`].
//!
//! ## Motivation
//!
//! The ecliptic frame is the natural reference for solar system object
//! kinematics: asteroid proper motions are nearly parallel to the ecliptic
//! plane, and the ecliptic latitude $\beta$ is a strong prior on the
//! population. Fink-FAT uses ecliptic tangent-plane coordinates as the
//! working frame for its kinematic seed models.
//!
//! ## Coordinate convention
//!
//! - Ecliptic longitude $\lambda \in [0, 2\pi)$ — measured eastward from the
//!   vernal equinox along the ecliptic.
//! - Ecliptic latitude $\beta \in [-\pi/2, \pi/2]$ — measured northward from
//!   the ecliptic plane.
//! - All angles are stored in **radians**.
//! - The obliquity of the ecliptic is fixed at the J2000.0 value
//!   $\varepsilon = 23.439\,291\,1°$, which is sufficient for ZTF and
//!   Vera Rubin survey data.
//!
//! ## Frame transformation
//!
//! The equatorial-to-ecliptic rotation is a right-handed rotation about the
//! $x$-axis by the obliquity angle $\varepsilon$:
//!
//! $$\begin{pmatrix} x_e \\ y_e \\ z_e \end{pmatrix}
//!   = R_x(\varepsilon)
//!   \begin{pmatrix} x_{eq} \\ y_{eq} \\ z_{eq} \end{pmatrix}, \qquad
//!   R_x(\varepsilon) = \begin{pmatrix}
//!     1 & 0 & 0 \\
//!     0 & \cos\varepsilon & \sin\varepsilon \\
//!     0 & -\sin\varepsilon & \cos\varepsilon
//!   \end{pmatrix}$$
//!
//! The ecliptic longitude and latitude are recovered as:
//!
//! $$\lambda = \mathrm{atan2}(y_e,\, x_e), \qquad \beta = \arcsin(z_e)$$
//!
//! ## Types
//!
//! | Type | Role |
//! |------|------|
//! | [`EclipticCoord`] | Ecliptic position $(\lambda, \beta)$ with marginal 1-σ errors |
//! | [`EclipticCoordCov`] | [`EclipticCoord`] with full 2×2 covariance $\Sigma_{\lambda\beta}$ |
//!
//! ## Conversion paths
//!
//! | From | To | Method |
//! |------|----|--------|
//! | [`EquCoord`] | [`EclipticCoord`] | `From` impl — position only, errors discarded |
//! | [`EquCoordCov`] | [`EclipticCoordCov`] | `From` impl — full covariance propagation |
//! | [`EclipticCoord`] | [`EquCoord`] | `From` impl — inverse rotation, errors discarded |
//! | [`EclipticCoordCov`] | [`EquCoordCov`] | `From` impl — full covariance propagation |
//!
//! ## Covariance propagation
//!
//! Errors are propagated via first-order linearisation. The composite
//! Jacobian $J_{\alpha\delta \to \lambda\beta}$ is the product of three
//! factors (see [`EclipticCoordCov`] for the full derivation):
//!
//! $$J = J_{\mathrm{cart}\to\mathrm{ecl}} \cdot R_x(\varepsilon) \cdot J_{\mathrm{equ}\to\mathrm{cart}}$$
//!
//! and the output covariance is $\Sigma_{\lambda\beta} = J\,\Sigma_{\alpha\delta}\,J^\top$.
//!
//! ## Pole singularity
//!
//! Near the ecliptic poles ($r_{xy} = \sqrt{x_e^2 + y_e^2} \to 0$) the
//! longitude partial derivatives diverge; a small numerical floor is applied
//! to $r_{xy}$ before division to avoid NaN in the Jacobian.

use std::{f64::consts::TAU, fmt};

use crate::{
    Degrees, Radians,
    coordinates::{
        COS_OBL, RXY_MIN, SIN_OBL,
        cov2::Cov2,
        equatorial::{EquCoord, EquCoordCov},
    },
};

// ---------------------------------------------------------------------------
// EclipticCoord
// ---------------------------------------------------------------------------

/// An ecliptic sky position with associated astrometric uncertainties.
///
/// All four fields are stored in **radians**.
///
/// - `lon` ($\lambda$) — ecliptic longitude in $[0, 2\pi)$.
/// - `lat` ($\beta$) — ecliptic latitude in $[-\pi/2, \pi/2]$.
/// - `lon_error` ($\sigma_\lambda$) — 1-σ uncertainty on the longitude.
/// - `lat_error` ($\sigma_\beta$) — 1-σ uncertainty on the latitude.
///
/// Use [`EclipticCoord::new`] to construct from radians directly, or
/// [`EclipticCoord::from_degrees`] to supply values in degrees.
/// For full covariance propagation, use [`EclipticCoordCov`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EclipticCoord {
    /// Ecliptic longitude in **radians**, in the range $[0, 2\pi)$.
    pub lon: Radians,
    /// 1-σ uncertainty on the ecliptic longitude, in **radians**.
    pub lon_error: Radians,
    /// Ecliptic latitude in **radians**, in the range $[-\pi/2, \pi/2]$.
    pub lat: Radians,
    /// 1-σ uncertainty on the ecliptic latitude, in **radians**.
    pub lat_error: Radians,
}

impl EclipticCoord {
    /// Construct an [`EclipticCoord`] from values already expressed in radians.
    ///
    /// # Arguments
    ///
    /// - `lon` — ecliptic longitude in **radians**.
    /// - `lon_error` — 1-σ uncertainty on the longitude in **radians**.
    /// - `lat` — ecliptic latitude in **radians**.
    /// - `lat_error` — 1-σ uncertainty on the latitude in **radians**.
    ///
    /// # Returns
    ///
    /// A new [`EclipticCoord`] with the four fields set exactly as provided;
    /// no range checking or unit conversion is performed.
    #[inline]
    pub fn new(lon: Radians, lon_error: Radians, lat: Radians, lat_error: Radians) -> Self {
        Self {
            lon,
            lon_error,
            lat,
            lat_error,
        }
    }

    /// Construct an [`EclipticCoord`] from values expressed in degrees.
    ///
    /// Each argument is converted to radians via $\times\,\pi/180$ before
    /// being stored. No range checking is performed.
    ///
    /// # Arguments
    ///
    /// - `lon_deg` — ecliptic longitude in **degrees**.
    /// - `lon_error_deg` — 1-σ uncertainty on the longitude in **degrees**.
    /// - `lat_deg` — ecliptic latitude in **degrees**.
    /// - `lat_error_deg` — 1-σ uncertainty on the latitude in **degrees**.
    ///
    /// # Returns
    ///
    /// A new [`EclipticCoord`] with all fields stored in radians.
    #[inline]
    pub fn from_degrees(
        lon_deg: Degrees,
        lon_error_deg: Degrees,
        lat_deg: Degrees,
        lat_error_deg: Degrees,
    ) -> Self {
        Self {
            lon: lon_deg.to_radians(),
            lon_error: lon_error_deg.to_radians(),
            lat: lat_deg.to_radians(),
            lat_error: lat_error_deg.to_radians(),
        }
    }

    /// Return the sky position as a `(lon_deg, lat_deg)` tuple in degrees.
    ///
    /// # Returns
    ///
    /// `(lon_deg, lat_deg)` — ecliptic longitude and latitude in degrees.
    /// Uncertainties are not included; use [`EclipticCoord::error_in_degrees`].
    #[inline]
    pub fn to_degrees(&self) -> (Degrees, Degrees) {
        (self.lon.to_degrees(), self.lat.to_degrees())
    }

    /// Return the measurement uncertainties as a `(lon_error_deg, lat_error_deg)`
    /// tuple in degrees.
    ///
    /// # Returns
    ///
    /// `(lon_error_deg, lat_error_deg)` — 1-σ uncertainties in longitude and
    /// latitude, both expressed in degrees.
    #[inline]
    pub fn error_in_degrees(&self) -> (Degrees, Degrees) {
        (self.lon_error.to_degrees(), self.lat_error.to_degrees())
    }
}

impl fmt::Display for EclipticCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lon_deg = self.lon.to_degrees();
        let lat_deg = self.lat.to_degrees();
        let lon_err_arcsec = self.lon_error.to_degrees() * 3600.0;
        let lat_err_arcsec = self.lat_error.to_degrees() * 3600.0;
        write!(
            f,
            "lon: {lon_deg:.6} deg ± {lon_err_arcsec:.3} arcsec, \
             lat: {lat_deg:.6} deg ± {lat_err_arcsec:.3} arcsec"
        )
    }
}

// ---------------------------------------------------------------------------
// EclipticCoordCov
// ---------------------------------------------------------------------------

/// An ecliptic sky position together with its full 2×2 astrometric covariance.
///
/// [`EclipticCoord`] stores only the marginal 1-σ errors
/// $(\sigma_\lambda, \sigma_\beta)$, assuming longitude–latitude independence.
/// `EclipticCoordCov` lifts that limitation by carrying the full [`Cov2`]
/// matrix, including the off-diagonal term $\sigma_{\lambda\beta}$.
///
/// ## Covariance propagation
///
/// The covariance $\Sigma_{\lambda\beta}$ is obtained from the equatorial
/// covariance $\Sigma_{\alpha\delta}$ via first-order linearisation. The
/// composite Jacobian $J$ decomposes as:
///
/// $$J = J_{\mathrm{cart}\to\mathrm{ecl}} \cdot R_x(\varepsilon) \cdot J_{\mathrm{equ}\to\mathrm{cart}}$$
///
/// where:
///
/// - $J_{\mathrm{equ}\to\mathrm{cart}}$ is the $3 \times 2$ Jacobian of
///   $(\alpha,\delta)\to(x,y,z)$:
///
/// $$J_{\mathrm{equ}\to\mathrm{cart}} = \begin{pmatrix}
///   -\cos\delta\sin\alpha & -\sin\delta\cos\alpha \\
///    \cos\delta\cos\alpha & -\sin\delta\sin\alpha \\
///    0                    &  \cos\delta
/// \end{pmatrix}$$
///
/// - $R_x(\varepsilon)$ is the $3\times3$ rotation matrix about the $x$-axis
///   by the obliquity $\varepsilon$:
///
/// $$R_x(\varepsilon) = \begin{pmatrix}
///   1 & 0 & 0 \\
///   0 & \cos\varepsilon &  \sin\varepsilon \\
///   0 & -\sin\varepsilon & \cos\varepsilon
/// \end{pmatrix}$$
///
/// - $J_{\mathrm{cart}\to\mathrm{ecl}}$ is the $2\times3$ Jacobian of
///   $(x_e, y_e, z_e)\to(\lambda, \beta)$, with
///   $r_{xy} = \sqrt{x_e^2 + y_e^2}$:
///
/// $$J_{\mathrm{cart}\to\mathrm{ecl}} = \begin{pmatrix}
///   -y_e / r_{xy}^2 & x_e / r_{xy}^2 & 0 \\
///   -x_e z_e / r_{xy} & -y_e z_e / r_{xy} & r_{xy}
/// \end{pmatrix}$$
///
/// The output covariance is then:
///
/// $$\Sigma_{\lambda\beta} = J\,\Sigma_{\alpha\delta}\,J^\top$$
///
/// ## Construction
///
/// - [`EclipticCoordCov::new`] — provide coordinate and covariance directly.
/// - `From<`[`EquCoordCov`]`>` — full propagation from equatorial covariance.
/// - `From<`[`EquCoord`]`>` — diagonal covariance inferred from marginal errors.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EclipticCoordCov {
    /// Ecliptic sky position. The `lon_error` and `lat_error` fields carry the
    /// **marginal** 1-σ uncertainties (square roots of the diagonal of `cov`).
    pub coord: EclipticCoord,
    /// Full 2×2 astrometric covariance in $(\lambda, \beta)$.
    ///
    /// - `cov.xx` = $\sigma_\lambda^2$
    /// - `cov.yy` = $\sigma_\beta^2$
    /// - `cov.xy` = $\sigma_{\lambda\beta}$
    pub cov: Cov2,
}

impl EclipticCoordCov {
    /// Construct an [`EclipticCoordCov`] from a coordinate and a full [`Cov2`].
    ///
    /// # Arguments
    ///
    /// - `coord` — the ecliptic position; `lon_error` and `lat_error` should
    ///   equal `cov.xx.sqrt()` and `cov.yy.sqrt()` respectively (not enforced).
    /// - `cov` — the 2×2 astrometric covariance.
    ///
    /// # Returns
    ///
    /// A new [`EclipticCoordCov`].
    #[inline]
    pub fn new(coord: EclipticCoord, cov: Cov2) -> Self {
        Self { coord, cov }
    }

    /// Build an [`EclipticCoordCov`] from the marginal errors of an
    /// [`EclipticCoord`], with a diagonal covariance (off-diagonal term zero).
    ///
    /// # Arguments
    ///
    /// - `c` — ecliptic coordinate whose `lon_error` and `lat_error` are
    ///   squared to form the diagonal variances.
    ///
    /// # Returns
    ///
    /// An [`EclipticCoordCov`] with `cov = diag(lon_error², lat_error²)`.
    #[inline]
    pub fn from_ecl(c: EclipticCoord) -> Self {
        let cov = Cov2 {
            xx: c.lon_error * c.lon_error,
            yy: c.lat_error * c.lat_error,
            xy: 0.0,
        };
        Self { coord: c, cov }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply $R_x(\varepsilon)$ to an equatorial Cartesian vector.
///
/// Returns the ecliptic Cartesian components $(x_e, y_e, z_e)$.
#[inline]
fn rotate_to_ecliptic(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    (x, COS_OBL * y + SIN_OBL * z, -SIN_OBL * y + COS_OBL * z)
}

/// Apply $R_x(-\varepsilon)$ (inverse rotation) to an ecliptic Cartesian vector.
///
/// Returns the equatorial Cartesian components $(x_{eq}, y_{eq}, z_{eq})$.
#[inline]
fn rotate_to_equatorial(xe: f64, ye: f64, ze: f64) -> (f64, f64, f64) {
    (xe, COS_OBL * ye - SIN_OBL * ze, SIN_OBL * ye + COS_OBL * ze)
}

/// Compute $(\lambda, \beta)$ from ecliptic Cartesian components.
#[inline]
fn ecl_cart_to_lonlat(xe: f64, ye: f64, ze: f64) -> (f64, f64) {
    let lon = ye.atan2(xe).rem_euclid(TAU);
    let lat = ze.atan2(xe.hypot(ye));
    (lon, lat)
}

/// Compute the $2\times3$ Jacobian $J_{\mathrm{cart}\to\mathrm{ecl}}$ at
/// the ecliptic Cartesian point $(x_e, y_e, z_e)$ (unit sphere assumed).
///
/// Returns `[[∂λ/∂xe, ∂λ/∂ye, ∂λ/∂ze], [∂β/∂xe, ∂β/∂ye, ∂β/∂ze]]`.
#[inline]
fn jac_cart_to_ecl(xe: f64, ye: f64, ze: f64) -> [[f64; 3]; 2] {
    let rxy2 = (xe * xe + ye * ye).max(RXY_MIN * RXY_MIN);
    let rxy = rxy2.sqrt();
    [
        [-ye / rxy2, xe / rxy2, 0.0],
        [-xe * ze / rxy, -ye * ze / rxy, rxy],
    ]
}

// ---------------------------------------------------------------------------
// Conversions: position only (EquCoord ↔ EclipticCoord)
// ---------------------------------------------------------------------------

impl From<&EquCoord> for EclipticCoord {
    /// Convert an equatorial position to ecliptic coordinates.
    ///
    /// Applies the $R_x(\varepsilon)$ rotation and recovers $(\lambda, \beta)$.
    /// Uncertainties carried by the [`EquCoord`] are **discarded**; use
    /// `From<`[`EquCoordCov`]`> for `[`EclipticCoordCov`]` to propagate them.
    fn from(c: &EquCoord) -> Self {
        let (sdec, cdec) = c.dec.sin_cos();
        let (sra, cra) = c.ra.sin_cos();
        let xeq = cdec * cra;
        let yeq = cdec * sra;
        let zeq = sdec;
        let (xe, ye, ze) = rotate_to_ecliptic(xeq, yeq, zeq);
        let (lon, lat) = ecl_cart_to_lonlat(xe, ye, ze);
        EclipticCoord::new(lon, 0.0, lat, 0.0)
    }
}

impl From<EquCoord> for EclipticCoord {
    #[inline]
    fn from(c: EquCoord) -> Self {
        Self::from(&c)
    }
}

impl From<&EclipticCoord> for EquCoord {
    /// Convert an ecliptic position back to equatorial coordinates.
    ///
    /// Applies the inverse rotation $R_x(-\varepsilon)$. Output errors are
    /// set to zero.
    fn from(c: &EclipticCoord) -> Self {
        let (slat, clat) = c.lat.sin_cos();
        let (slon, clon) = c.lon.sin_cos();
        let xe = clat * clon;
        let ye = clat * slon;
        let ze = slat;
        let (xeq, yeq, zeq) = rotate_to_equatorial(xe, ye, ze);
        let rho = xeq.hypot(yeq);
        let dec = zeq.atan2(rho);
        let ra = yeq.atan2(xeq).rem_euclid(TAU);
        EquCoord::new(ra, 0.0, dec, 0.0)
    }
}

impl From<EclipticCoord> for EquCoord {
    #[inline]
    fn from(c: EclipticCoord) -> Self {
        Self::from(&c)
    }
}

// ---------------------------------------------------------------------------
// Conversions: full covariance (EquCoordCov ↔ EclipticCoordCov)
// ---------------------------------------------------------------------------

impl From<EquCoordCov> for EclipticCoordCov {
    /// Convert an equatorial coordinate with covariance to ecliptic coordinates.
    ///
    /// The position is transformed via the rotation $R_x(\varepsilon)$ where
    /// $\varepsilon$ is the obliquity of the ecliptic. The covariance matrix
    /// is propagated using the composite Jacobian
    ///
    /// $$J = J_{\text{cart}\to\text{ecl}} \cdot R_x(\varepsilon) \cdot J_{\text{equ}\to\text{cart}}$$
    ///
    /// where:
    /// - $J_{\text{equ}\to\text{cart}}$ (3×2) maps $(\alpha, \delta)$ to unit-sphere
    ///   Cartesian coordinates,
    /// - $R_x(\varepsilon)$ (3×3) rotates the equatorial frame to the ecliptic frame,
    /// - $J_{\text{cart}\to\text{ecl}}$ (2×3) maps ecliptic Cartesian coordinates to
    ///   $(\lambda, \beta)$.
    ///
    /// The covariance is then propagated as
    ///
    /// $$\Sigma_{\lambda\beta} = J \, \Sigma_{\alpha\delta} \, J^\top$$
    fn from(ec: EquCoordCov) -> Self {
        let (sdec, cdec) = ec.coord.dec.sin_cos();
        let (sra, cra) = ec.coord.ra.sin_cos();

        // Equatorial unit-sphere Cartesian coordinates.
        let xeq = cdec * cra;
        let yeq = cdec * sra;
        let zeq = sdec;

        // Rotate to ecliptic Cartesian coordinates and convert to (λ, β).
        let (xe, ye, ze) = rotate_to_ecliptic(xeq, yeq, zeq);
        let (lon, lat) = ecl_cart_to_lonlat(xe, ye, ze);

        // J_cart→ecl evaluated at the ecliptic Cartesian position (2×3).
        let j_c2e = jac_cart_to_ecl(xe, ye, ze);

        // J_equ→cart (3×2): columns are partial derivatives with respect to
        // α and δ of the unit-sphere map (α,δ) → (cos δ cos α, cos δ sin α, sin δ).
        //
        //   ∂/∂α = (-cos δ sin α,  cos δ cos α,         0)
        //   ∂/∂δ = (-sin δ cos α, -sin δ sin α,  cos δ   )
        let m00 = -cdec * sra; // ∂x/∂α
        let m01 = -sdec * cra; // ∂x/∂δ
        let m10 = cdec * cra; // ∂y/∂α
        let m11 = -sdec * sra; // ∂y/∂δ
        let m20 = 0.0; // ∂z/∂α
        let m21 = cdec; // ∂z/∂δ

        // R_x(ε) · J_equ→cart (3×2).
        //
        // R_x(ε) acts on rows: row 0 is unchanged; rows 1 and 2 are mixed by
        // the obliquity rotation:
        //   row 1 →  cos ε · row 1 + sin ε · row 2
        //   row 2 → -sin ε · row 1 + cos ε · row 2
        let rxj: [[f64; 2]; 3] = [
            [m00, m01],
            [COS_OBL * m10 + SIN_OBL * m20, COS_OBL * m11 + SIN_OBL * m21],
            [
                -SIN_OBL * m10 + COS_OBL * m20,
                -SIN_OBL * m11 + COS_OBL * m21,
            ],
        ];

        // J_total = J_cart→ecl (2×3) · (R_x · J_equ→cart) (3×2) → (2×2).
        let mut jtot = [[0.0f64; 2]; 2];
        for i in 0..2 {
            for k in 0..2 {
                for j in 0..3 {
                    jtot[i][k] += j_c2e[i][j] * rxj[j][k];
                }
            }
        }

        // Σ_λβ = J · Σ_αδ · J^T (2×2 direct propagation).
        //
        // Expanding J Σ J^T with Σ = [[xx, xy], [xy, yy]]:
        //   (J Σ J^T)[i][i] = J[i][0]² xx + 2 J[i][0] J[i][1] xy + J[i][1]² yy
        //   (J Σ J^T)[0][1] = J[0][0] J[1][0] xx
        //                   + (J[0][0] J[1][1] + J[0][1] J[1][0]) xy
        //                   + J[0][1] J[1][1] yy
        let s = &ec.cov;
        let cov2 = Cov2 {
            xx: jtot[0][0] * jtot[0][0] * s.xx
                + 2.0 * jtot[0][0] * jtot[0][1] * s.xy
                + jtot[0][1] * jtot[0][1] * s.yy,
            yy: jtot[1][0] * jtot[1][0] * s.xx
                + 2.0 * jtot[1][0] * jtot[1][1] * s.xy
                + jtot[1][1] * jtot[1][1] * s.yy,
            xy: jtot[0][0] * jtot[1][0] * s.xx
                + (jtot[0][0] * jtot[1][1] + jtot[0][1] * jtot[1][0]) * s.xy
                + jtot[0][1] * jtot[1][1] * s.yy,
        };

        let coord = EclipticCoord::new(lon, cov2.xx.sqrt(), lat, cov2.yy.sqrt());
        EclipticCoordCov::new(coord, cov2)
    }
}

impl From<EclipticCoordCov> for EquCoordCov {
    /// Convert an ecliptic coordinate with covariance to equatorial coordinates.
    ///
    /// The position is transformed via the inverse rotation $R_x(-\varepsilon)$.
    /// The covariance matrix is propagated using the composite Jacobian
    ///
    /// $$J = J_{\text{cart}\to\text{equ}} \cdot R_x(-\varepsilon) \cdot J_{\text{ecl}\to\text{cart}}$$
    ///
    /// where:
    /// - $J_{\text{ecl}\to\text{cart}}$ (3×2) maps $(\lambda, \beta)$ to unit-sphere
    ///   Cartesian coordinates,
    /// - $R_x(-\varepsilon)$ (3×3) rotates the ecliptic frame back to the equatorial frame,
    /// - $J_{\text{cart}\to\text{equ}}$ (2×3) maps equatorial Cartesian coordinates to
    ///   $(\alpha, \delta)$.
    ///
    /// The covariance is then propagated as
    ///
    /// $$\Sigma_{\alpha\delta} = J \, \Sigma_{\lambda\beta} \, J^\top$$
    ///
    /// The cylindrical radius $\rho = \sqrt{x_\text{eq}^2 + y_\text{eq}^2}$ is
    /// clamped to `RXY_MIN` to avoid division by zero near the equatorial poles.
    fn from(ec: EclipticCoordCov) -> Self {
        let (slat, clat) = ec.coord.lat.sin_cos();
        let (slon, clon) = ec.coord.lon.sin_cos();

        // Ecliptic unit-sphere Cartesian coordinates.
        let xe = clat * clon;
        let ye = clat * slon;
        let ze = slat;

        // Rotate to equatorial Cartesian coordinates and convert to (α, δ).
        let (xeq, yeq, zeq) = rotate_to_equatorial(xe, ye, ze);
        let rho2 = (xeq * xeq + yeq * yeq).max(RXY_MIN * RXY_MIN);
        let rho = rho2.sqrt();
        let ra = yeq.atan2(xeq).rem_euclid(TAU);
        let dec = zeq.atan2(rho);

        // J_cart→equ evaluated at the equatorial Cartesian position (2×3).
        //
        //   ∂α/∂(x,y,z) = (-y/ρ², x/ρ², 0)
        //   ∂δ/∂(x,y,z) = (-xz/ρ, -yz/ρ, ρ)   (unnormalized; on the unit sphere ρ ≈ cos δ)
        let j_c2q: [[f64; 3]; 2] = [
            [-yeq / rho2, xeq / rho2, 0.0],
            [-xeq * zeq / rho, -yeq * zeq / rho, rho],
        ];

        // J_ecl→cart (3×2): columns are partial derivatives with respect to
        // λ and β of the unit-sphere map (λ,β) → (cos β cos λ, cos β sin λ, sin β).
        //
        //   ∂/∂λ = (-cos β sin λ,  cos β cos λ,        0)
        //   ∂/∂β = (-sin β cos λ, -sin β sin λ,  cos β  )
        let n00 = -clat * slon; // ∂x/∂λ
        let n01 = -slat * clon; // ∂x/∂β
        let n10 = clat * clon; // ∂y/∂λ
        let n11 = -slat * slon; // ∂y/∂β
        let n20 = 0.0; // ∂z/∂λ
        let n21 = clat; // ∂z/∂β

        // R_x(-ε) · J_ecl→cart (3×2).
        //
        // R_x(-ε) is the transpose of R_x(ε); it mixes rows 1 and 2:
        //   row 1 →  cos ε · row 1 - sin ε · row 2
        //   row 2 →  sin ε · row 1 + cos ε · row 2
        let rxinv_j: [[f64; 2]; 3] = [
            [n00, n01],
            [COS_OBL * n10 - SIN_OBL * n20, COS_OBL * n11 - SIN_OBL * n21],
            [SIN_OBL * n10 + COS_OBL * n20, SIN_OBL * n11 + COS_OBL * n21],
        ];

        // J_total = J_cart→equ (2×3) · (R_x(-ε) · J_ecl→cart) (3×2) → (2×2).
        let mut jtot = [[0.0f64; 2]; 2];
        for i in 0..2 {
            for k in 0..2 {
                for j in 0..3 {
                    jtot[i][k] += j_c2q[i][j] * rxinv_j[j][k];
                }
            }
        }

        // Σ_αδ = J · Σ_λβ · J^T (2×2 direct propagation).
        let s = &ec.cov;
        let cov2 = Cov2 {
            xx: jtot[0][0] * jtot[0][0] * s.xx
                + 2.0 * jtot[0][0] * jtot[0][1] * s.xy
                + jtot[0][1] * jtot[0][1] * s.yy,
            yy: jtot[1][0] * jtot[1][0] * s.xx
                + 2.0 * jtot[1][0] * jtot[1][1] * s.xy
                + jtot[1][1] * jtot[1][1] * s.yy,
            xy: jtot[0][0] * jtot[1][0] * s.xx
                + (jtot[0][0] * jtot[1][1] + jtot[0][1] * jtot[1][0]) * s.xy
                + jtot[0][1] * jtot[1][1] * s.yy,
        };

        let coord = EquCoord::new(ra, cov2.xx.sqrt(), dec, cov2.yy.sqrt());
        EquCoordCov::new(coord, cov2)
    }
}

impl From<EclipticCoord> for EquCoordCov {
    /// Convert an [`EclipticCoord`] to [`EquCoordCov`] using its marginal errors
    /// as a diagonal input covariance.
    #[inline]
    fn from(c: EclipticCoord) -> Self {
        EquCoordCov::from(EclipticCoordCov::from_ecl(c))
    }
}

impl From<EquCoord> for EclipticCoordCov {
    /// Convert an [`EquCoord`] to [`EclipticCoordCov`] using its marginal errors
    /// as a diagonal input covariance.
    ///
    /// Delegates to `From<`[`EquCoordCov`]`>` after building the diagonal
    /// covariance from `ra_error` and `dec_error`.
    #[inline]
    fn from(c: EquCoord) -> Self {
        EclipticCoordCov::from(EquCoordCov::from_equ(c))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod ecliptic_coord_tests {
    use crate::coordinates::OBLIQUITY_J2000;

    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::FRAC_PI_2;

    fn equ(ra_deg: f64, dec_deg: f64, ra_err_arcsec: f64, dec_err_arcsec: f64) -> EquCoord {
        EquCoord::from_degrees(
            ra_deg,
            ra_err_arcsec / 3600.0,
            dec_deg,
            dec_err_arcsec / 3600.0,
        )
    }

    // ------------------------------------------------------------------
    // Round-trip: EquCoord → EclipticCoord → EquCoord
    // ------------------------------------------------------------------

    #[test]
    fn roundtrip_position_vernal_equinox() {
        // Vernal equinox: RA=0, Dec=0 → lon=0, lat=0
        let eq = equ(0.0, 0.0, 0.0, 0.0);
        let ecl = EclipticCoord::from(eq);
        assert_abs_diff_eq!(ecl.lon, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ecl.lat, 0.0, epsilon = 1e-12);
        let eq2 = EquCoord::from(ecl);
        assert_abs_diff_eq!(eq2.ra, eq.ra, epsilon = 1e-12);
        assert_abs_diff_eq!(eq2.dec, eq.dec, epsilon = 1e-12);
    }

    #[test]
    fn roundtrip_position_arbitrary() {
        let eq = equ(123.456, 34.567, 0.0, 0.0);
        let ecl = EclipticCoord::from(eq);
        let eq2 = EquCoord::from(ecl);
        assert_abs_diff_eq!(eq2.ra, eq.ra, epsilon = 1e-7);
        assert_abs_diff_eq!(eq2.dec, eq.dec, epsilon = 1e-7);
    }

    #[test]
    fn roundtrip_position_negative_dec() {
        let eq = equ(270.0, -45.0, 0.0, 0.0);
        let ecl = EclipticCoord::from(eq);
        let eq2 = EquCoord::from(ecl);
        assert_abs_diff_eq!(eq2.ra, eq.ra, epsilon = 1e-11);
        assert_abs_diff_eq!(eq2.dec, eq.dec, epsilon = 1e-11);
    }

    // ------------------------------------------------------------------
    // Known value: north ecliptic pole
    // ------------------------------------------------------------------

    #[test]
    fn north_ecliptic_pole_lat() {
        // The north ecliptic pole is at RA=270°, Dec=90°−ε
        let dec_pole = FRAC_PI_2 - OBLIQUITY_J2000;
        let eq = EquCoord::new(270_f64.to_radians(), 0.0, dec_pole, 0.0);
        let ecl = EclipticCoord::from(eq);
        assert_abs_diff_eq!(ecl.lat, FRAC_PI_2, epsilon = 1e-6);
    }

    // ------------------------------------------------------------------
    // Covariance round-trip: EquCoordCov → EclipticCoordCov → EquCoordCov
    // ------------------------------------------------------------------

    #[test]
    fn roundtrip_covariance_diagonal() {
        let eq = equ(83.82, 22.01, 0.1, 0.1); // near Crab Nebula
        let eq_cov = EquCoordCov::from_equ(eq);
        let ecl_cov = EclipticCoordCov::from(eq_cov);
        let eq_cov2 = EquCoordCov::from(ecl_cov);

        assert_abs_diff_eq!(eq_cov2.coord.ra, eq.ra, epsilon = 1e-7);
        assert_abs_diff_eq!(eq_cov2.coord.dec, eq.dec, epsilon = 1e-7);
        assert_abs_diff_eq!(eq_cov2.cov.xx, eq_cov.cov.xx, epsilon = 1e-17);
        assert_abs_diff_eq!(eq_cov2.cov.yy, eq_cov.cov.yy, epsilon = 1e-17);
        assert_abs_diff_eq!(eq_cov2.cov.xy, eq_cov.cov.xy, epsilon = 1e-17);
    }

    #[test]
    fn roundtrip_covariance_with_correlation() {
        let eq = equ(200.0, -15.0, 0.05, 0.08);
        let cov = Cov2 {
            xx: eq.ra_error * eq.ra_error,
            yy: eq.dec_error * eq.dec_error,
            xy: 0.3 * eq.ra_error * eq.dec_error,
        };
        let eq_cov = EquCoordCov::new(eq, cov);
        let ecl_cov = EclipticCoordCov::from(eq_cov);
        let eq_cov2 = EquCoordCov::from(ecl_cov);

        assert_abs_diff_eq!(eq_cov2.cov.xx, eq_cov.cov.xx, epsilon = 1e-17);
        assert_abs_diff_eq!(eq_cov2.cov.yy, eq_cov.cov.yy, epsilon = 1e-17);
        assert_abs_diff_eq!(eq_cov2.cov.xy, eq_cov.cov.xy, epsilon = 1e-17);
    }

    // ------------------------------------------------------------------
    // Marginal errors are consistent with covariance diagonal
    // ------------------------------------------------------------------

    #[test]
    fn marginal_errors_consistent_with_cov() {
        let eq = equ(45.0, 10.0, 0.2, 0.15);
        let ecl_cov = EclipticCoordCov::from(EquCoordCov::from_equ(eq));
        assert_abs_diff_eq!(
            ecl_cov.coord.lon_error,
            ecl_cov.cov.xx.sqrt(),
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(
            ecl_cov.coord.lat_error,
            ecl_cov.cov.yy.sqrt(),
            epsilon = 1e-15
        );
    }

    // ------------------------------------------------------------------
    // Covariance is positive definite (det > 0)
    // ------------------------------------------------------------------

    #[test]
    fn covariance_positive_definite() {
        let eq = equ(300.0, -60.0, 0.1, 0.1);
        let ecl_cov = EclipticCoordCov::from(EquCoordCov::from_equ(eq));
        let det = ecl_cov.cov.xx * ecl_cov.cov.yy - ecl_cov.cov.xy * ecl_cov.cov.xy;
        assert!(det > 0.0, "covariance must be positive definite, det={det}");
    }
}

#[cfg(test)]
mod ecliptic_coord_proptests {
    use super::*;
    use crate::coordinates::equatorial::{EquCoord, EquCoordCov};
    use proptest::prelude::*;
    use std::f64::consts::{FRAC_PI_2, TAU};

    // ------------------------------------------------------------------
    // Strategies
    // ------------------------------------------------------------------

    /// Arbitrary equatorial position covering the full sky, with zero errors.
    fn arb_equ_position() -> impl Strategy<Value = EquCoord> {
        (0.0..TAU, -FRAC_PI_2..=FRAC_PI_2).prop_map(|(ra, dec)| EquCoord::new(ra, 0.0, dec, 0.0))
    }

    /// Arbitrary equatorial position excluding a 5° cone around each ecliptic
    /// pole, where the longitude Jacobian is numerically ill-conditioned.
    fn arb_equ_away_from_ecl_poles() -> impl Strategy<Value = EquCoord> {
        arb_equ_position().prop_filter("too close to an ecliptic pole", |eq| {
            let ecl = EclipticCoord::from(eq);
            ecl.lat.abs() < FRAC_PI_2 - 5_f64.to_radians()
        })
    }

    /// Positive 1-σ uncertainty uniformly drawn in [0.01, 10] arcseconds,
    /// returned in radians.
    fn arb_err_rad() -> impl Strategy<Value = f64> {
        (0.01_f64..10.0_f64).prop_map(|arcsec| arcsec.to_radians() / 3600.0)
    }

    /// Diagonal [`EquCoordCov`] with independent RA and Dec uncertainties.
    fn arb_equ_cov_diagonal() -> impl Strategy<Value = EquCoordCov> {
        (arb_equ_away_from_ecl_poles(), arb_err_rad(), arb_err_rad()).prop_map(
            |(eq, ra_err, dec_err)| {
                let eq = EquCoord::new(eq.ra, ra_err, eq.dec, dec_err);
                EquCoordCov::from_equ(eq)
            },
        )
    }

    /// [`EquCoordCov`] with an arbitrary correlation coefficient
    /// $\rho \in (-0.9, 0.9)$, ensuring positive definiteness.
    fn arb_equ_cov_correlated() -> impl Strategy<Value = EquCoordCov> {
        (
            arb_equ_away_from_ecl_poles(),
            arb_err_rad(),
            arb_err_rad(),
            -0.9_f64..0.9_f64,
        )
            .prop_map(|(eq, ra_err, dec_err, rho)| {
                let eq = EquCoord::new(eq.ra, ra_err, eq.dec, dec_err);
                let cov = Cov2 {
                    xx: ra_err * ra_err,
                    yy: dec_err * dec_err,
                    xy: rho * ra_err * dec_err,
                };
                EquCoordCov::new(eq, cov)
            })
    }

    // ------------------------------------------------------------------
    // Tolerance helper
    // ------------------------------------------------------------------

    fn angle_tol(v: f64) -> f64 {
        // Relative part: 3e-7 covers worst-case transcendental chain error.
        // Absolute floor: 1e-7 rad (~20 mas) covers near-zero angles where
        // the relative bound collapses below the actual floating-point noise.
        (3e-7 * v.abs()).max(1e-7)
    }

    fn cov_tol(v: f64) -> f64 {
        (1e-6 * v.abs()).max(1e-50)
    }

    // ------------------------------------------------------------------
    // Position round-trip: EquCoord → EclipticCoord → EquCoord
    // ------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_roundtrip_position(eq in arb_equ_away_from_ecl_poles()) {
            let ecl = EclipticCoord::from(eq);
            let eq2 = EquCoord::from(ecl);

            let ra_diff = (eq2.ra - eq.ra).abs();
            let ra_diff = ra_diff.min(TAU - ra_diff);
            let tol_ra = angle_tol(eq.ra).max(angle_tol(eq.dec));
            let tol_dec = angle_tol(eq.dec);

            prop_assert!(
                ra_diff < tol_ra,
                "RA mismatch: {} vs {} (tol {})",
                eq.ra, eq2.ra, tol_ra
            );
            prop_assert!(
                (eq2.dec - eq.dec).abs() < tol_dec,
                "Dec mismatch: {} vs {} (tol {})",
                eq.dec, eq2.dec, tol_dec
            );
        }
    }

    // ------------------------------------------------------------------
    // Marginal errors consistent with covariance diagonal
    // ------------------------------------------------------------------

    proptest! {
        /// The marginal 1-σ errors stored in [`EclipticCoord`] must equal the
        /// square roots of the diagonal entries of the covariance matrix.
        #[test]
        fn prop_marginal_errors_consistent_with_cov(eq_cov in arb_equ_cov_correlated()) {
            let ecl_cov = EclipticCoordCov::from(eq_cov);
            let tol = cov_tol(ecl_cov.cov.xx.sqrt());
            prop_assert!(
                (ecl_cov.coord.lon_error - ecl_cov.cov.xx.sqrt()).abs() < tol,
                "lon_error {} ≠ sqrt(cov.xx) {}",
                ecl_cov.coord.lon_error,
                ecl_cov.cov.xx.sqrt()
            );
            let tol = cov_tol(ecl_cov.cov.yy.sqrt());
            prop_assert!(
                (ecl_cov.coord.lat_error - ecl_cov.cov.yy.sqrt()).abs() < tol,
                "lat_error {} ≠ sqrt(cov.yy) {}",
                ecl_cov.coord.lat_error,
                ecl_cov.cov.yy.sqrt()
            );
        }
    }

    // ------------------------------------------------------------------
    // Covariance positive definiteness
    // ------------------------------------------------------------------

    proptest! {
        /// The propagated ecliptic covariance is positive definite
        /// ($\det\Sigma > 0$) for a diagonal input covariance.
        #[test]
        fn prop_ecliptic_cov_positive_definite(eq_cov in arb_equ_cov_diagonal()) {
            let ecl_cov = EclipticCoordCov::from(eq_cov);
            let det = ecl_cov.cov.xx * ecl_cov.cov.yy - ecl_cov.cov.xy * ecl_cov.cov.xy;
            prop_assert!(det > 0.0, "det = {det}");
        }
    }

    proptest! {
        /// The propagated ecliptic covariance is positive definite
        /// ($\det\Sigma > 0$) for a correlated input covariance.
        #[test]
        fn prop_ecliptic_cov_positive_definite_correlated(eq_cov in arb_equ_cov_correlated()) {
            let ecl_cov = EclipticCoordCov::from(eq_cov);
            let det = ecl_cov.cov.xx * ecl_cov.cov.yy - ecl_cov.cov.xy * ecl_cov.cov.xy;
            prop_assert!(det > 0.0, "det = {det}");
        }
    }

    // ------------------------------------------------------------------
    // Variances non-negative
    // ------------------------------------------------------------------

    proptest! {
        /// Diagonal variances are non-negative after propagation to the
        /// ecliptic frame, guarding against catastrophic cancellation that
        /// would produce `NaN` via `sqrt`.
        #[test]
        fn prop_ecliptic_variances_non_negative(eq_cov in arb_equ_cov_correlated()) {
            let ecl_cov = EclipticCoordCov::from(eq_cov);
            prop_assert!(
                ecl_cov.cov.xx >= 0.0,
                "σ_λλ = {} < 0", ecl_cov.cov.xx
            );
            prop_assert!(
                ecl_cov.cov.yy >= 0.0,
                "σ_ββ = {} < 0", ecl_cov.cov.yy
            );
        }
    }
}
