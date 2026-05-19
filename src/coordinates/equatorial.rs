//! Equatorial coordinate representation with astrometric uncertainties.
//!
//! This module provides [`EquCoord`], a compact structure that bundles a sky
//! position in the equatorial frame with its associated 1-σ astrometric
//! uncertainties. All four fields — `ra`, `dec`, `ra_error`, `dec_error` — are
//! stored in **radians**.
//!
//! ## Construction
//!
//! - [`EquCoord::new`] — accepts all four values already in radians.
//! - [`EquCoord::from_degrees`] — accepts values in degrees and converts them
//!   to radians automatically.
//!
//! ## Accessors
//!
//! - [`EquCoord::to_degrees`] — returns `(ra, dec)` as a degree tuple.
//! - [`EquCoord::error_in_degrees`] — returns `(ra_error, dec_error)` in degrees.
//!
//! ## Geometry
//!
//! - [`EquCoord::angular_separation`] — great-circle distance via the numerically
//!   stable Vincenty formula; result lies in $[0, \pi]$ radians.
//! - [`EquCoord::spherical_midpoint`] — vector-averaging midpoint on the unit sphere;
//!   stable even for nearly-antipodal directions.
//!
//! ## Covariance propagation
//!
//! - [`EquCoordCov::to_cartesian_cov`] — projects to [`CartesianCoordCov`] using a
//!   first-order Jacobian propagation of the full 2×2 input covariance.
//! - [`EquCoordCov`] — extends [`EquCoord`] with a full 2×2 astrometric covariance
//!   (including the RA–Dec correlation term); provides
//!   [`EquCoordCov::to_cartesian_cov`] for full-covariance propagation.
//!
//! ## Conversions
//!
//! - `From<`[`CartesianCoord`]`> for EquCoord` — recovers RA/Dec from a Cartesian
//!   direction; output errors are set to zero.
//! - [`std::fmt::Display`] — formats the position as
//!   `RA: <ra_deg> deg, Dec: <dec_deg> deg`.

use std::{f64::consts::TAU, fmt};

use crate::{
    Degrees, Radians,
    coordinates::{
        NORM_MIN,
        cartesian::{CartesianCoord, CartesianCoordCov},
        cov2::Cov2,
    },
};

/// An equatorial sky position with associated astrometric uncertainties.
///
/// All four fields are stored internally in **radians**.  Use
/// [`EquCoord::new`] to construct from radians directly, or
/// [`EquCoord::from_degrees`] to supply values in degrees.
///
/// The coordinate pair `(ra, dec)` locates a point on the celestial sphere;
/// the companion pair `(ra_error, dec_error)` carries the 1-σ measurement
/// uncertainties of that position.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EquCoord {
    /// Right ascension in **radians**, in the range $[0, 2\pi)$.
    pub ra: Radians,
    /// 1-σ uncertainty on the right ascension, in **radians**.
    pub ra_error: Radians,
    /// Declination in **radians**, in the range $[-\pi/2, \pi/2]$.
    pub dec: Radians,
    /// 1-σ uncertainty on the declination, in **radians**.
    pub dec_error: Radians,
}

impl EquCoord {
    /// Construct an [`EquCoord`] from values already expressed in radians.
    ///
    /// # Arguments
    ///
    /// - `ra` — right ascension in **radians**.
    /// - `ra_error` — 1-σ uncertainty on the right ascension in **radians**.
    /// - `dec` — declination in **radians**.
    /// - `dec_error` — 1-σ uncertainty on the declination in **radians**.
    ///
    /// # Returns
    ///
    /// A new [`EquCoord`] with the four fields set exactly as provided; no
    /// range checking or unit conversion is performed.
    #[inline]
    pub fn new(ra: Radians, ra_error: Radians, dec: Radians, dec_error: Radians) -> Self {
        Self {
            ra,
            ra_error,
            dec,
            dec_error,
        }
    }

    /// Construct an [`EquCoord`] from values expressed in degrees.
    ///
    /// Each argument is converted to radians via the standard $\times\,\pi/180$
    /// factor before being stored.  No range checking is performed.
    ///
    /// # Arguments
    ///
    /// - `ra_deg` — right ascension in **degrees**.
    /// - `ra_error_deg` — 1-σ uncertainty on the right ascension in **degrees**.
    /// - `dec_deg` — declination in **degrees**.
    /// - `dec_error_deg` — 1-σ uncertainty on the declination in **degrees**.
    ///
    /// # Returns
    ///
    /// A new [`EquCoord`] with all fields stored in radians.
    #[inline]
    pub fn from_degrees(
        ra_deg: Degrees,
        ra_error_deg: Degrees,
        dec_deg: Degrees,
        dec_error_deg: Degrees,
    ) -> Self {
        Self {
            ra: ra_deg.to_radians(),
            ra_error: ra_error_deg.to_radians(),
            dec: dec_deg.to_radians(),
            dec_error: dec_error_deg.to_radians(),
        }
    }

    /// Return the sky position as a `(ra_deg, dec_deg)` tuple in degrees.
    ///
    /// The returned tuple contains only the coordinate values; measurement
    /// uncertainties are **not** included.  To obtain the errors in degrees,
    /// use [`EquCoord::error_in_degrees`] instead.
    ///
    /// # Returns
    ///
    /// `(ra_deg, dec_deg)` — right ascension and declination converted to degrees.
    #[inline]
    pub fn to_degrees(&self) -> (Degrees, Degrees) {
        (self.ra.to_degrees(), self.dec.to_degrees())
    }

    /// Return the measurement uncertainties as a `(ra_error_deg, dec_error_deg)` tuple in degrees.
    ///
    /// This is the counterpart to [`EquCoord::to_degrees`]: it converts
    /// [`EquCoord::ra_error`] and [`EquCoord::dec_error`] from radians to degrees.
    ///
    /// # Returns
    ///
    /// `(ra_error_deg, dec_error_deg)` — 1-σ uncertainties in right ascension and
    /// declination, both expressed in degrees.
    #[inline]
    pub fn error_in_degrees(&self) -> (Degrees, Degrees) {
        (self.ra_error.to_degrees(), self.dec_error.to_degrees())
    }

    /// Compute the great-circle angular separation between two sky positions
    /// using the **Vincenty formula**.
    ///
    /// This formulation is numerically stable for all separations, including
    /// near the poles and for antipodal points.  It mirrors the implementation
    /// used in Astropy's `angular_separation`.
    ///
    /// # Arguments
    ///
    /// - `other` — the second sky position.
    ///
    /// # Returns
    ///
    /// Angular separation in **radians**, guaranteed to lie in $[0, \pi]$.
    ///
    /// The result is computed as:
    ///
    /// $$d = \mathrm{atan2}\left(\sqrt{n_1^2 + n_2^2}, D\right)$$
    ///
    /// where, writing $\Delta\lambda = \mathrm{RA}_2 - \mathrm{RA}_1$:
    ///
    /// $$n_1 = \cos(\delta_2)\sin(\Delta\lambda)$$
    ///
    /// $$n_2 = \cos(\delta_1)\sin(\delta_2) - \sin(\delta_1)\cos(\delta_2)\cos(\Delta\lambda)$$
    ///
    /// $$D = \sin(\delta_1)\sin(\delta_2) + \cos(\delta_1)\cos(\delta_2)\cos(\Delta\lambda)$$
    ///
    /// See <https://en.wikipedia.org/wiki/Great-circle_distance> for a full derivation.
    #[inline]
    pub fn angular_separation(&self, other: &EquCoord) -> Radians {
        let dlon = other.ra - self.ra;

        let (slon, clon) = dlon.sin_cos();
        let (slat1, clat1) = self.dec.sin_cos();
        let (slat2, clat2) = other.dec.sin_cos();

        let num1 = clat2 * slon;
        let num2 = clat1 * slat2 - slat1 * clat2 * clon;
        let denom = slat1 * slat2 + clat1 * clat2 * clon;

        num1.hypot(num2).atan2(denom)
    }

    /// Compute a robust spherical midpoint between two sky directions.
    ///
    /// Returns the angular mean via vector averaging: the two unit vectors are
    /// summed and the result is renormalized to the unit sphere.  This is not
    /// the exact geodesic midpoint, but is stable and accurate enough for
    /// defining a tangent-plane center.
    ///
    /// Arguments
    /// ---------
    /// * `ra1`  – Right ascension of the first point (radians).
    /// * `dec1` – Declination of the first point (radians).
    /// * `ra2`  – Right ascension of the second point (radians).
    /// * `dec2` – Declination of the second point (radians).
    ///
    /// Return
    /// ------
    /// `(ra, dec)` of the midpoint in radians, with $\mathrm{ra} \in [0, 2\pi)$.
    ///
    /// Notes
    /// -----
    /// When the two directions are nearly antipodal the sum vector approaches
    /// zero; a numerical floor is applied before normalization to avoid NaN.
    #[inline]
    pub fn spherical_midpoint(&self, other: &EquCoord) -> Self {
        let cart1: CartesianCoord = (*self).into();
        let cart2: CartesianCoord = (*other).into();
        let cart = cart1 + cart2;
        let r = (cart.x * cart.x + cart.y * cart.y + cart.z * cart.z)
            .sqrt()
            .max(NORM_MIN);
        CartesianCoord {
            x: cart.x / r,
            y: cart.y / r,
            z: cart.z / r,
        }
        .into()
    }
}

// ---------------------------------------------------------------------------
// EquCoord Display
// ---------------------------------------------------------------------------

/// Formats the equatorial coordinates in sexagesimal notation with
/// measurement uncertainties expressed in arcseconds.
///
/// # Format
///
/// ```text
/// RA: HHh MMm SS.sss s (DDD.dddddd deg) ± E.eee arcsec,
/// Dec: ±DDd MMm SS.sss s (±DDD.dddddd deg) ± E.eee arcsec
/// ```
///
/// Decimal degree values are shown in parentheses for quick numerical
/// reference.
impl fmt::Display for EquCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ra_deg = self.ra.to_degrees();
        let dec_deg = self.dec.to_degrees();

        // Uncertainties converted from radians to arcseconds (1 rad = 206 264.806 arcsec)
        let ra_err_arcsec = self.ra_error.to_degrees() * 3600.0;
        let dec_err_arcsec = self.dec_error.to_degrees() * 3600.0;

        let ra_sex = deg_to_sexagesimal(ra_deg, true);
        let dec_sex = deg_to_sexagesimal(dec_deg, false);

        write!(
            f,
            "RA: {ra_sex} ({ra_deg:.6} deg) ± {ra_err_arcsec:.3} arcsec, \
             Dec: {dec_sex} ({dec_deg:.6} deg) ± {dec_err_arcsec:.3} arcsec"
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: decimal degrees → sexagesimal string
// ---------------------------------------------------------------------------

/// Converts a decimal degree value to a sexagesimal (DMS) string.
///
/// # Format
///
/// - For right ascension (hours): `HHh MMm SS.sss s`
/// - For declination (degrees):   `±DDd MMm SS.sss s`
///
/// The `is_ra` flag controls whether the value is divided by 15 to convert
/// from degrees to hours (RA convention).
fn deg_to_sexagesimal(deg: f64, is_ra: bool) -> String {
    if is_ra {
        // Convert degrees → hours
        let total_hours = deg / 15.0;
        let h = total_hours.abs().floor() as u32;
        let rem_m = (total_hours.abs() - h as f64) * 60.0;
        let m = rem_m.floor() as u32;
        let s = (rem_m - m as f64) * 60.0;
        format!("{:02}h {:02}m {:06.3}s", h, m, s)
    } else {
        let sign = if deg < 0.0 { '-' } else { '+' };
        let abs = deg.abs();
        let d = abs.floor() as u32;
        let rem_m = (abs - d as f64) * 60.0;
        let m = rem_m.floor() as u32;
        let s = (rem_m - m as f64) * 60.0;
        format!("{}{:02}d {:02}m {:06.3}s", sign, d, m, s)
    }
}

// ---------------------------------------------------------------------------
// EquCoordCov
// ---------------------------------------------------------------------------

/// An equatorial sky position together with its full 2×2 astrometric covariance.
///
/// [`EquCoord`] stores only the marginal 1-σ errors $(\sigma_\alpha, \sigma_\delta)$
/// and assumes the RA–Dec correlation is zero.  `EquCoordCov` lifts that
/// limitation: the [`Cov2`] field retains all three independent entries of the
/// symmetric covariance matrix, including the off-diagonal RA–Dec correlation
/// $\sigma_{\alpha\delta}$.
///
/// ## Construction
///
/// - [`EquCoordCov::new`] — provide coordinate and covariance directly.
/// - [`EquCoordCov::from_equ`] — build a diagonal covariance from an
///   [`EquCoord`]'s marginal errors (off-diagonal term set to zero).
///
/// ## Covariance propagation
///
/// - [`EquCoordCov::to_cartesian_cov`] — propagate to a full
///   [`CartesianCoordCov`] via the 3×2 Jacobian of $(\alpha,\delta)\to(x,y,z)$,
///   using [`Cov2::transform_j3`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EquCoordCov {
    /// Equatorial sky position.  The `ra_error` and `dec_error` fields
    /// carry the **marginal** 1-σ uncertainties (square roots of the diagonal
    /// of `cov`).
    pub coord: EquCoord,
    /// Full 2×2 astrometric covariance in $(\alpha, \delta)$.
    ///
    /// - `cov.xx` = $\sigma_\alpha^2$ (variance of right ascension)
    /// - `cov.yy` = $\sigma_\delta^2$ (variance of declination)
    /// - `cov.xy` = $\sigma_{\alpha\delta}$ (RA–Dec covariance)
    pub cov: Cov2,
}

impl EquCoordCov {
    /// Construct an [`EquCoordCov`] from a coordinate and a full [`Cov2`].
    ///
    /// # Arguments
    ///
    /// - `coord` — the sky position; `ra_error` and `dec_error` should equal
    ///   `cov.xx.sqrt()` and `cov.yy.sqrt()` respectively (this is not enforced
    ///   but inconsistency will cause confusion).
    /// - `cov` — the 2×2 astrometric covariance.
    ///
    /// # Returns
    ///
    /// A new [`EquCoordCov`].
    #[inline]
    pub fn new(coord: EquCoord, cov: Cov2) -> Self {
        Self { coord, cov }
    }

    /// Build an [`EquCoordCov`] from the marginal errors of an [`EquCoord`].
    ///
    /// The off-diagonal covariance term $\sigma_{\alpha\delta}$ is set to zero
    /// (independence assumed).  This is the natural starting point when the
    /// input comes from an [`EquCoord`] that carries no cross-correlation.
    ///
    /// # Arguments
    ///
    /// - `c` — equatorial coordinate whose `ra_error` and `dec_error` are
    ///   squared to form the diagonal variances.
    ///
    /// # Returns
    ///
    /// An [`EquCoordCov`] with `cov = diag(ra_error², dec_error²)`.
    #[inline]
    pub fn from_equ(c: EquCoord) -> Self {
        let cov = Cov2::from_equ(&c);
        Self { coord: c, cov }
    }

    /// Convert to [`CartesianCoordCov`] by propagating the full 2×2 covariance
    /// through the Jacobian $J$ of $(\alpha,\delta)\to(x,y,z)$.
    ///
    /// # Formulation
    ///
    /// The Jacobian is
    ///
    /// $$J = \begin{pmatrix}
    /// -\cos\delta\sin\alpha & -\sin\delta\cos\alpha \\
    ///  \cos\delta\cos\alpha & -\sin\delta\sin\alpha \\
    ///  0                    &  \cos\delta
    /// \end{pmatrix}$$
    ///
    /// and the output covariance is $\Sigma_{xyz} = J\,\Sigma_{\alpha\delta}\,J^\top$,
    /// computed via [`Cov2::transform_j3`].
    ///
    /// # Returns
    ///
    /// A [`CartesianCoordCov`] whose packed `cov` array contains the upper
    /// triangle of $\Sigma_{xyz}$ in row-major order `[xx, xy, xz, yy, yz, zz]`.
    ///
    /// # Notes
    ///
    /// - This method preserves any RA–Dec off-diagonal correlation stored in
    ///   `self.cov.xy`.
    /// - The linear approximation is accurate for small astrometric errors.
    pub fn to_cartesian_cov(&self) -> CartesianCoordCov {
        let (sdec, cdec) = self.coord.dec.sin_cos();
        let (sra, cra) = self.coord.ra.sin_cos();

        // J is 3×2, j[row][col]; col 0 = ∂/∂α, col 1 = ∂/∂δ.
        let j = [
            [-cdec * sra, -sdec * cra],
            [cdec * cra, -sdec * sra],
            [0.0_f64, cdec],
        ];

        let cov3 = self.cov.transform_j3(j);
        CartesianCoordCov {
            coord: CartesianCoord::from(&self.coord),
            cov: cov3,
        }
    }
}

impl From<EquCoord> for EquCoordCov {
    /// Build an [`EquCoordCov`] from an [`EquCoord`] with a diagonal covariance.
    ///
    /// Delegates to [`EquCoordCov::from_equ`]: the off-diagonal term is set to zero.
    #[inline]
    fn from(c: EquCoord) -> Self {
        EquCoordCov::from_equ(c)
    }
}

impl From<&CartesianCoord> for EquCoord {
    /// Recover equatorial angles from a Cartesian direction.
    ///
    /// The vector is **not** required to be unit-normalised; only its
    /// direction matters. Errors on the output are set to zero.
    ///
    /// Numerical notes
    /// ---------------
    /// - $\rho = \sqrt{x^2 + y^2}$ is computed via [`f64::hypot`] for
    ///   stability near the poles.
    /// - $\alpha$ is undefined at the exact pole ($\rho = 0$) and falls back
    ///   to `0` via [`f64::atan2`] semantics.
    fn from(cart: &CartesianCoord) -> Self {
        let CartesianCoord { x, y, z } = *cart;
        let rho = x.hypot(y);
        let dec = z.atan2(rho);
        let ra = y.atan2(x).rem_euclid(TAU);
        EquCoord::new(ra, 0.0, dec, 0.0)
    }
}

impl From<CartesianCoord> for EquCoord {
    /// Convenience wrapper for `From<&CartesianCoord>` to allow direct conversion from owned `CartesianCoord`.
    #[inline]
    fn from(cart: CartesianCoord) -> Self {
        Self::from(&cart)
    }
}

#[cfg(test)]
mod equ_coord_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    // ------------------------------------------------------------------ //
    // Proptest strategies                                                  //
    // ------------------------------------------------------------------ //

    // Generates a valid Right Ascension value in [0, 2π] radians.
    prop_compose! {
        fn valid_ra()(ra in 0.0_f64..=(2.0 * PI)) -> f64 { ra }
    }

    // Generates a valid Declination value in [-π/2, π/2] radians.
    prop_compose! {
        fn valid_dec()(dec in (-PI / 2.0)..=(PI / 2.0)) -> f64 { dec }
    }

    // Generates a valid `EquCoord` with RA ∈ [0, 2π] and Dec ∈ [-π/2, π/2].
    // Errors are set to zero for property-based tests focused on angular separation.
    prop_compose! {
        fn valid_coord()(ra in valid_ra(), dec in valid_dec()) -> EquCoord {
            EquCoord::new(ra, 0.0, dec, 0.0)
        }
    }

    // ------------------------------------------------------------------ //
    // Constructor tests                                                    //
    // ------------------------------------------------------------------ //

    mod constructor {
        use super::*;

        /// Verifies that `new()` stores `ra` and `dec` fields exactly as provided.
        #[test]
        fn new_stores_ra_and_dec_fields() {
            let ra = 1.2345_f64;
            let dec = -0.6789_f64;
            let coord = EquCoord::new(ra, 0.0, dec, 0.0);
            assert_abs_diff_eq!(coord.ra, ra, epsilon = 0.0);
            assert_abs_diff_eq!(coord.dec, dec, epsilon = 0.0);
        }

        /// Verifies that `new()` with zero values produces a coordinate at the origin.
        #[test]
        fn new_with_zero_values() {
            let coord = EquCoord::new(0.0, 0.0, 0.0, 0.0);
            assert_abs_diff_eq!(coord.ra, 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(coord.dec, 0.0, epsilon = 0.0);
        }

        /// Verifies that `new()` stores `ra_error` and `dec_error` fields exactly as provided.
        #[test]
        fn new_stores_error_fields() {
            let ra_err = 0.001_f64;
            let dec_err = 0.002_f64;
            let coord = EquCoord::new(0.0, ra_err, 0.0, dec_err);
            assert_abs_diff_eq!(coord.ra_error, ra_err, epsilon = 0.0);
            assert_abs_diff_eq!(coord.dec_error, dec_err, epsilon = 0.0);
        }

        /// Verifies that `from_degrees()` converts RA and Dec from degrees to radians correctly.
        /// Uses the standard conversion factor π/180 as the expected value.
        #[test]
        fn from_degrees_converts_ra_correctly() {
            let coord = EquCoord::from_degrees(180.0, 0.0, 0.0, 0.0);
            assert_abs_diff_eq!(coord.ra, PI, epsilon = 1e-15);
        }

        /// Verifies that `from_degrees()` converts Dec from degrees to radians correctly.
        #[test]
        fn from_degrees_converts_dec_correctly() {
            let coord = EquCoord::from_degrees(0.0, 0.0, 90.0, 0.0);
            assert_abs_diff_eq!(coord.dec, PI / 2.0, epsilon = 1e-15);
        }

        /// Verifies that `from_degrees()` converts errors from degrees to radians correctly.
        #[test]
        fn from_degrees_converts_errors_correctly() {
            let coord = EquCoord::from_degrees(0.0, 90.0, 0.0, 45.0);
            assert_abs_diff_eq!(coord.ra_error, PI / 2.0, epsilon = 1e-15);
            assert_abs_diff_eq!(coord.dec_error, PI / 4.0, epsilon = 1e-15);
        }

        /// Verifies that `from_degrees()` handles negative Dec values (southern hemisphere).
        #[test]
        fn from_degrees_handles_negative_dec() {
            let coord = EquCoord::from_degrees(0.0, 0.0, -45.0, 0.0);
            assert_abs_diff_eq!(coord.dec, -PI / 4.0, epsilon = 1e-15);
        }

        /// Verifies that `from_degrees()` handles the maximum RA value of 360°.
        #[test]
        fn from_degrees_handles_full_circle_ra() {
            let coord = EquCoord::from_degrees(360.0, 0.0, 0.0, 0.0);
            assert_abs_diff_eq!(coord.ra, 2.0 * PI, epsilon = 1e-15);
        }
    }

    // ------------------------------------------------------------------ //
    // Degree conversion round-trip tests                                  //
    // ------------------------------------------------------------------ //

    mod degree_conversion {
        use super::*;

        /// Verifies that `to_degrees()` round-trips correctly: degrees → radians → degrees.
        /// Epsilon of 1e-13 accounts for floating-point rounding in the two conversions.
        #[test]
        fn to_degrees_round_trips_ra() {
            let ra_deg = 123.456_f64;
            let coord = EquCoord::from_degrees(ra_deg, 0.0, 0.0, 0.0);
            let (ra_out, _) = coord.to_degrees();
            assert_abs_diff_eq!(ra_out, ra_deg, epsilon = 1e-13);
        }

        /// Verifies that `to_degrees()` round-trips Dec correctly.
        #[test]
        fn to_degrees_round_trips_dec() {
            let dec_deg = -33.7_f64;
            let coord = EquCoord::from_degrees(0.0, 0.0, dec_deg, 0.0);
            let (_, dec_out) = coord.to_degrees();
            assert_abs_diff_eq!(dec_out, dec_deg, epsilon = 1e-13);
        }

        /// Verifies that `to_degrees()` returns (0, 0) when both fields are zero.
        #[test]
        fn to_degrees_returns_zero_for_origin() {
            let coord = EquCoord::new(0.0, 0.0, 0.0, 0.0);
            let (ra_deg, dec_deg) = coord.to_degrees();
            assert_abs_diff_eq!(ra_deg, 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(dec_deg, 0.0, epsilon = 0.0);
        }

        /// Verifies that `to_degrees()` converts a known radian value (π rad) to 180°.
        #[test]
        fn to_degrees_converts_pi_to_180() {
            let coord = EquCoord::new(PI, 0.0, 0.0, 0.0);
            let (ra_deg, _) = coord.to_degrees();
            assert_abs_diff_eq!(ra_deg, 180.0, epsilon = 1e-13);
        }
    }

    // ------------------------------------------------------------------ //
    // Display formatting tests                                             //
    // ------------------------------------------------------------------ //

    mod display {
        use super::*;

        /// Verifies that `Display` output contains the "RA:" label.
        #[test]
        fn display_contains_ra_label() {
            let coord = EquCoord::from_degrees(90.0, 0.0, 45.0, 0.0);
            let output = format!("{coord}");
            assert!(
                output.contains("RA:"),
                "Display output missing 'RA:' label: {output}"
            );
        }

        /// Verifies that `Display` output contains the "Dec:" label.
        #[test]
        fn display_contains_dec_label() {
            let coord = EquCoord::from_degrees(90.0, 0.0, 45.0, 0.0);
            let output = format!("{coord}");
            assert!(
                output.contains("Dec:"),
                "Display output missing 'Dec:' label: {output}"
            );
        }

        /// Verifies that `Display` output contains the "deg" unit string.
        #[test]
        fn display_contains_deg_unit() {
            let coord = EquCoord::from_degrees(10.0, 0.0, -20.0, 0.0);
            let output = format!("{coord}");
            assert!(
                output.contains("deg"),
                "Display output missing 'deg' unit: {output}"
            );
        }

        /// Verifies that `Display` output contains the numeric RA value.
        /// The coordinate is set to 180°, which should appear in the formatted string.
        #[test]
        fn display_contains_numeric_ra_value() {
            let coord = EquCoord::from_degrees(180.0, 0.0, 0.0, 0.0);
            let output = format!("{coord}");
            assert!(
                output.contains("180"),
                "Display output missing RA value '180': {output}"
            );
        }

        /// Verifies that `Display` output contains the numeric Dec value for a negative angle.
        #[test]
        fn display_contains_numeric_dec_value() {
            let coord = EquCoord::from_degrees(0.0, 0.0, -90.0, 0.0);
            let output = format!("{coord}");
            assert!(
                output.contains("-90"),
                "Display output missing Dec value '-90': {output}"
            );
        }
    }

    // ------------------------------------------------------------------ //
    // Deterministic angular separation tests                               //
    // ------------------------------------------------------------------ //

    mod angular_separation {
        use super::*;

        /// Verifies that the angular separation of a point with itself is zero.
        #[test]
        fn separation_same_point_is_zero() {
            let c = EquCoord::from_degrees(120.0, 0.0, 45.0, 0.0);
            assert_abs_diff_eq!(c.angular_separation(&c), 0.0, epsilon = 1e-12);
        }

        /// Verifies that two antipodal points have a separation of exactly π radians.
        /// The antipodal pair (ra, dec) and (ra + 180°, -dec) spans the full diameter of the sphere.
        #[test]
        fn separation_antipodal_points_is_pi() {
            let c1 = EquCoord::from_degrees(0.0, 0.0, 30.0, 0.0);
            let c2 = EquCoord::from_degrees(180.0, 0.0, -30.0, 0.0);
            assert_abs_diff_eq!(c1.angular_separation(&c2), PI, epsilon = 1e-12);
        }

        /// Verifies that two equatorial points separated by 90° in RA give a separation of π/2.
        #[test]
        fn separation_equator_90deg() {
            let c1 = EquCoord::from_degrees(0.0, 0.0, 0.0, 0.0);
            let c2 = EquCoord::from_degrees(90.0, 0.0, 0.0, 0.0);
            assert_abs_diff_eq!(c1.angular_separation(&c2), PI / 2.0, epsilon = 1e-12);
        }

        /// Verifies that the north and south celestial poles are separated by exactly π radians.
        #[test]
        fn separation_poles_is_pi() {
            let north = EquCoord::from_degrees(0.0, 0.0, 90.0, 0.0);
            let south = EquCoord::from_degrees(0.0, 0.0, -90.0, 0.0);
            assert_abs_diff_eq!(north.angular_separation(&south), PI, epsilon = 1e-12);
        }

        /// Verifies a known angular separation using real star coordinates.
        /// Sirius (RA=101.2875°, Dec=-16.7161°) and Canopus (RA=95.9879°, Dec=-52.6956°)
        /// have a reference separation of 36.2208° as computed by Astropy.
        #[test]
        fn separation_sirius_canopus_known_value() {
            let sirius = EquCoord::from_degrees(101.2875, 0.0, -16.7161, 0.0);
            let canopus = EquCoord::from_degrees(95.9879, 0.0, -52.6956, 0.0);
            let expected_deg = 36.2208_f64;
            let sep_deg = sirius.angular_separation(&canopus).to_degrees();
            // Tolerance of 0.01° matches the precision of the Astropy reference value.
            assert_abs_diff_eq!(sep_deg, expected_deg, epsilon = 0.01);
        }

        /// Verifies that two points on the same meridian (same RA, different Dec) have a
        /// separation equal to the absolute difference in declination.
        #[test]
        fn separation_same_meridian_equals_dec_difference() {
            let c1 = EquCoord::from_degrees(45.0, 0.0, 10.0, 0.0);
            let c2 = EquCoord::from_degrees(45.0, 0.0, 70.0, 0.0);
            let expected = (70.0_f64 - 10.0_f64).to_radians();
            assert_abs_diff_eq!(c1.angular_separation(&c2), expected, epsilon = 1e-12);
        }

        /// Verifies that two identical points at the north pole have zero separation.
        #[test]
        fn separation_north_pole_with_itself_is_zero() {
            let pole = EquCoord::from_degrees(0.0, 0.0, 90.0, 0.0);
            assert_abs_diff_eq!(pole.angular_separation(&pole), 0.0, epsilon = 1e-12);
        }
    }

    // ------------------------------------------------------------------ //
    // Property-based tests                                                 //
    // ------------------------------------------------------------------ //

    proptest! {
        /// Verifies that the angular separation always lies in the valid range [0, π].
        #[test]
        fn prop_separation_in_range(a in valid_coord(), b in valid_coord()) {
            let sep = a.angular_separation(&b);
            prop_assert!(sep >= 0.0, "negative separation: {sep}");
            prop_assert!(sep <= PI + 1e-12, "separation exceeds π: {sep}");
        }

        /// Verifies that angular separation is symmetric: d(a, b) == d(b, a).
        /// Floating-point arithmetic may introduce differences up to 1e-12.
        #[test]
        fn prop_separation_is_symmetric(a in valid_coord(), b in valid_coord()) {
            let sep_ab = a.angular_separation(&b);
            let sep_ba = b.angular_separation(&a);
            prop_assert!(
                (sep_ab - sep_ba).abs() < 1e-12,
                "asymmetry detected: {sep_ab} vs {sep_ba}"
            );
        }

        /// Verifies that the angular separation of any coordinate with itself is zero.
        #[test]
        fn prop_separation_with_self_is_zero(a in valid_coord()) {
            let sep = a.angular_separation(&a);
            prop_assert!(
                sep.abs() < 1e-12,
                "self-separation is non-zero: {sep}"
            );
        }

        /// Verifies that angular separation is invariant under a uniform RA shift applied
        /// to both coordinates (i.e., a rigid rotation around the polar axis).
        #[test]
        fn prop_separation_invariant_under_ra_shift(
            a in valid_coord(),
            b in valid_coord(),
            shift in 0.0_f64..=(2.0 * PI),
        ) {
            let a_shifted = EquCoord::new((a.ra + shift) % (2.0 * PI), 0.0, a.dec, 0.0);
            let b_shifted = EquCoord::new((b.ra + shift) % (2.0 * PI), 0.0, b.dec, 0.0);
            let sep_orig    = a.angular_separation(&b);
            let sep_shifted = a_shifted.angular_separation(&b_shifted);
            prop_assert!(
                (sep_orig - sep_shifted).abs() < 1e-10,
                "separation changed after RA shift={shift}: {sep_orig} vs {sep_shifted}"
            );
        }
    }

    // ------------------------------------------------------------------ //
    // error_in_degrees tests                                               //
    // ------------------------------------------------------------------ //

    mod error_in_degrees {
        use super::*;

        /// Verifies that `error_in_degrees()` converts ra_error from radians to degrees correctly.
        #[test]
        fn ra_error_converted_to_degrees() {
            let coord = EquCoord::new(0.0, PI / 2.0, 0.0, 0.0);
            let (ra_err_deg, _) = coord.error_in_degrees();
            assert_abs_diff_eq!(ra_err_deg, 90.0, epsilon = 1e-13);
        }

        /// Verifies that `error_in_degrees()` converts dec_error from radians to degrees correctly.
        #[test]
        fn dec_error_converted_to_degrees() {
            let coord = EquCoord::new(0.0, 0.0, 0.0, PI / 4.0);
            let (_, dec_err_deg) = coord.error_in_degrees();
            assert_abs_diff_eq!(dec_err_deg, 45.0, epsilon = 1e-13);
        }

        /// Verifies that zero errors remain zero after conversion.
        #[test]
        fn zero_errors_remain_zero() {
            let coord = EquCoord::new(1.0, 0.0, 0.5, 0.0);
            let (ra_err_deg, dec_err_deg) = coord.error_in_degrees();
            assert_abs_diff_eq!(ra_err_deg, 0.0, epsilon = 0.0);
            assert_abs_diff_eq!(dec_err_deg, 0.0, epsilon = 0.0);
        }

        /// Verifies the round-trip: errors stored via `from_degrees` are recovered by `error_in_degrees`.
        #[test]
        fn error_in_degrees_round_trips_from_degrees() {
            let ra_err_deg = 0.5_f64;
            let dec_err_deg = 1.2_f64;
            let coord = EquCoord::from_degrees(0.0, ra_err_deg, 0.0, dec_err_deg);
            let (ra_out, dec_out) = coord.error_in_degrees();
            assert_abs_diff_eq!(ra_out, ra_err_deg, epsilon = 1e-13);
            assert_abs_diff_eq!(dec_out, dec_err_deg, epsilon = 1e-13);
        }

        /// Verifies that `error_in_degrees` is independent of the coordinate values.
        #[test]
        fn error_in_degrees_independent_of_position() {
            let err_rad = 0.01_f64;
            let c1 = EquCoord::new(0.0, err_rad, 0.0, err_rad);
            let c2 = EquCoord::new(PI / 3.0, err_rad, PI / 6.0, err_rad);
            let (ra1, dec1) = c1.error_in_degrees();
            let (ra2, dec2) = c2.error_in_degrees();
            assert_abs_diff_eq!(ra1, ra2, epsilon = 1e-15);
            assert_abs_diff_eq!(dec1, dec2, epsilon = 1e-15);
        }
    }

    // ------------------------------------------------------------------ //
    // spherical_midpoint deterministic tests                               //
    // ------------------------------------------------------------------ //

    mod spherical_midpoint {
        use super::*;

        /// The midpoint of a point with itself equals the original point.
        #[test]
        fn midpoint_with_self_is_same_point() {
            let c = EquCoord::from_degrees(200.0, 0.0, -33.0, 0.0);
            let mid = c.spherical_midpoint(&c);
            assert_abs_diff_eq!(mid.ra, c.ra, epsilon = 1e-12);
            assert_abs_diff_eq!(mid.dec, c.dec, epsilon = 1e-12);
        }

        /// Two equatorial points (Dec=0) have a midpoint with Dec=0 and averaged RA.
        #[test]
        fn equatorial_midpoint_has_averaged_ra_and_zero_dec() {
            let c1 = EquCoord::from_degrees(20.0, 0.0, 0.0, 0.0);
            let c2 = EquCoord::from_degrees(100.0, 0.0, 0.0, 0.0);
            let mid = c1.spherical_midpoint(&c2);
            assert_abs_diff_eq!(mid.dec, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(mid.ra, 60.0_f64.to_radians(), epsilon = 1e-12);
        }

        /// Spherical midpoint is symmetric: midpoint(a, b) ≈ midpoint(b, a).
        #[test]
        fn midpoint_is_symmetric() {
            let a = EquCoord::from_degrees(50.0, 0.0, 10.0, 0.0);
            let b = EquCoord::from_degrees(110.0, 0.0, -30.0, 0.0);
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
    // EquCoordCov tests                                                    //
    // ------------------------------------------------------------------ //

    mod equ_coord_cov {
        use super::*;
        use crate::coordinates::cov2::Cov2;

        const EPS: f64 = 1e-12;

        /// from_equ builds a diagonal covariance from marginal errors.
        #[test]
        fn from_equ_diagonal() {
            let c = EquCoord::new(0.1, 0.01, 0.2, 0.02);
            let ec = EquCoordCov::from_equ(c);
            assert_abs_diff_eq!(ec.cov.xx, 0.01 * 0.01, epsilon = EPS);
            assert_abs_diff_eq!(ec.cov.yy, 0.02 * 0.02, epsilon = EPS);
            assert_abs_diff_eq!(ec.cov.xy, 0.0, epsilon = EPS);
        }

        /// to_cartesian_cov position matches CartesianCoord::from.
        #[test]
        fn to_cartesian_cov_position_matches() {
            let c = EquCoord::from_degrees(37.0, 1e-5, -15.0, 1e-5);
            let ec = EquCoordCov::from_equ(c);
            let cc = ec.to_cartesian_cov();
            let direct = crate::coordinates::cartesian::CartesianCoord::from(c);
            assert_abs_diff_eq!(cc.coord.x, direct.x, epsilon = EPS);
            assert_abs_diff_eq!(cc.coord.y, direct.y, epsilon = EPS);
            assert_abs_diff_eq!(cc.coord.z, direct.z, epsilon = EPS);
        }

        /// Zero input errors yield an all-zero Cartesian covariance.
        #[test]
        fn zero_errors_give_zero_cartesian_cov() {
            let c = EquCoord::new(0.5, 0.0, 0.3, 0.0);
            let cc = EquCoordCov::from_equ(c).to_cartesian_cov();
            for &v in &[
                cc.cov.xx, cc.cov.xy, cc.cov.xz, cc.cov.yy, cc.cov.yz, cc.cov.zz,
            ] {
                assert!(v.abs() < 1e-30, "expected zero, got {v}");
            }
        }

        /// new() stores coord and cov exactly.
        #[test]
        fn new_stores_fields() {
            let c = EquCoord::new(1.0, 0.01, 0.5, 0.02);
            let cov = Cov2 {
                xx: 1e-4,
                yy: 4e-4,
                xy: 5e-5,
            };
            let ec = EquCoordCov::new(c, cov);
            assert_eq!(ec.coord, c);
            assert_eq!(ec.cov, cov);
        }
    }
}
