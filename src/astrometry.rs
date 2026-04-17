//! Equatorial coordinate representation with measurement uncertainties.
//!
//! This module provides [`EquCoord`], a compact structure that bundles a sky
//! position in the equatorial frame (right ascension and declination) with its
//! associated astrometric uncertainties.  Two construction paths are available:
//!
//! - [`EquCoord::new`] — accepts all four values already in **radians**.
//! - [`EquCoord::from_degrees`] — accepts values in **degrees** and converts
//!   them to radians automatically.
//!
//! The module also exposes [`EquCoord::angular_separation`], which computes the
//! great-circle distance between two sky positions using the numerically stable
//! Vincenty formula.

use std::fmt;

use crate::{Degrees, Radians};

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
}

/// Formats the coordinate as a human-readable string of the form
/// `RA: <ra_deg> deg, Dec: <dec_deg> deg`, where both angles are expressed
/// in degrees.  Measurement uncertainties are not included in the output.
impl fmt::Display for EquCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (ra_deg, dec_deg) = self.to_degrees();
        write!(f, "RA: {} deg, Dec: {} deg", ra_deg, dec_deg)
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
}
