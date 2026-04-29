//! Observer metadata and geodetic conversion utilities.
//!
//! This module defines the [`Observer`] struct, which represents a ground-based
//! observatory site, together with the two constructors that create one from
//! geodetic or from pre-computed geocentric parallax coordinates.  It also
//! exposes the low-level geodetic conversion helpers
//! [`geodetic_to_parallax`] and the error type [`ObserverError`] that covers
//! all failure modes arising during observer
//! construction.
//!
//! Two sub-modules extend this functionality:
//!
//! - [`error_model`] — astrometric bias/RMS tables (FCCT14, CBM10, VFCC17).
//! - [`mpc`] — MPC observatory code types and the network fetch routine.
//!
//! ## Public items
//!
//! | Item | Kind | Description |
//! |------|------|-------------|
//! | [`Observer`] | struct | Ground-based observatory site with parallax constants |
//! | [`ObserverError`] | enum | Errors arising during observer construction |
//! | [`to_opt_notnan`] | fn | Lift `Option<f64>` into `Option<NotNan<f64>>` |
//! | [`geodetic_to_parallax`] | fn | Geodetic-to-geocentric parallax conversion (radians) |

pub mod dataset;
pub mod error_model;
pub mod mpc;

use std::fmt::{self, Display};

use thiserror::Error;

use ordered_float::NotNan;

use crate::{
    Degrees, Meters, Radians, ToNotNan,
    constants::{EARTH_MAJOR_AXIS, EARTH_MINOR_AXIS},
};

/// A ground-based observatory site represented by its geocentric parallax constants.
///
/// The three core fields — [`longitude`](Observer::longitude),
/// [`rho_cos_phi`](Observer::rho_cos_phi), and
/// [`rho_sin_phi`](Observer::rho_sin_phi) — follow the standard MPC
/// parallax convention: $\rho\cos\phi'$ and $\rho\sin\phi'$ are
/// dimensionless quantities expressed in units of the Earth's equatorial
/// radius.
///
/// Use [`Observer::new`] to build from geodetic latitude and elevation, or
/// [`Observer::from_parallax`] to supply the parallax constants directly.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Observer {
    /// Geodetic longitude in **radians** east of Greenwich.
    pub longitude: NotNan<Radians>,

    /// $\rho\cos\phi'$ — geocentric parallax constant, in **Earth radii** (dimensionless).
    pub rho_cos_phi: NotNan<f64>,

    /// $\rho\sin\phi'$ — geocentric parallax constant, in **Earth radii** (dimensionless).
    pub rho_sin_phi: NotNan<f64>,

    /// Optional human-readable site name.
    pub name: Option<String>,

    /// Right ascension measurement accuracy in **radians** (optional).
    pub ra_accuracy: Option<NotNan<Radians>>,

    /// Declination measurement accuracy in **radians** (optional).
    pub dec_accuracy: Option<NotNan<Radians>>,
}

/// Errors that can arise when constructing an [`Observer`].
#[derive(Debug, Error)]
pub enum ObserverError {
    /// A floating-point input was `NaN`, which cannot be stored in a [`NotNan`] wrapper.
    #[error("Invalid floating-point value (NaN encountered): {0}")]
    InvalidFloatValue(ordered_float::FloatIsNan),

    /// The requested MPC code was not found in the observatory lookup table.
    #[error("MPC code not found in lookup: {0:?}")]
    MpcCodeNotFound([u8; 3]),

    /// The supplied string is not a valid three-character MPC code.
    #[error("Invalid MPC code format: {0}")]
    InvalidMpcCode(String),

    /// An observer row in the MPC data was missing its three-character code.
    #[error("Missing MPC code for observer row")]
    MissingMpcCode,
}

/// Converts [`ordered_float::FloatIsNan`] into [`ObserverError::InvalidFloatValue`].
///
/// This blanket conversion lets the `?` operator be used when wrapping
/// floating-point values in [`NotNan`] inside the [`Observer`] constructors.
impl From<ordered_float::FloatIsNan> for ObserverError {
    fn from(e: ordered_float::FloatIsNan) -> Self {
        ObserverError::InvalidFloatValue(e)
    }
}

impl Observer {
    /// Construct an [`Observer`] from geodetic latitude and elevation.
    ///
    /// Converts `(latitude, elevation)` into geocentric parallax coordinates
    /// $(\rho\cos\phi', \rho\sin\phi')$ using [`geodetic_to_parallax`], then
    /// stores all fields as [`NotNan`]-wrapped values.
    ///
    /// # Arguments
    ///
    /// - `longitude` — geodetic longitude in **radians** (east positive).
    /// - `latitude` — geodetic latitude in **radians**.
    /// - `elevation` — height above the reference ellipsoid in **meters**.
    /// - `name` — optional human-readable site name.
    /// - `ra_accuracy` — optional RA measurement accuracy in **radians**.
    /// - `dec_accuracy` — optional declination measurement accuracy in **radians**.
    ///
    /// # Returns
    ///
    /// A new [`Observer`] with the computed parallax constants stored as
    /// [`NotNan`] values.
    ///
    /// # Errors
    ///
    /// Returns [`ObserverError::InvalidFloatValue`] if any of the supplied
    /// floating-point values is `NaN`.
    pub fn new(
        longitude: Radians,
        latitude: Radians,
        elevation: Meters,
        name: Option<String>,
        ra_accuracy: Option<Radians>,
        dec_accuracy: Option<Radians>,
    ) -> Result<Observer, ObserverError> {
        let (rho_cos_phi, rho_sin_phi) = geodetic_to_parallax(latitude, elevation);

        Observer::from_parallax(
            longitude,
            rho_cos_phi,
            rho_sin_phi,
            name,
            ra_accuracy,
            dec_accuracy,
        )
    }

    /// Construct an [`Observer`] from pre-computed geocentric parallax coordinates.
    ///
    /// This constructor skips the geodetic-to-parallax conversion and stores
    /// the supplied $(\rho\cos\phi', \rho\sin\phi')$ values directly.  Use
    /// this path when the parallax constants are already available — for
    /// example, when reading MPC observatory data that lists them explicitly.
    ///
    /// # Arguments
    ///
    /// - `longitude` — geodetic longitude in **radians** (east positive).
    /// - `rho_cos_phi` — $\rho\cos\phi'$ in **Earth radii** (dimensionless).
    /// - `rho_sin_phi` — $\rho\sin\phi'$ in **Earth radii** (dimensionless).
    /// - `name` — optional human-readable site name.
    /// - `ra_accuracy` — optional RA measurement accuracy in **radians**.
    /// - `dec_accuracy` — optional declination measurement accuracy in **radians**.
    ///
    /// # Returns
    ///
    /// A new [`Observer`] with the supplied parallax constants stored as
    /// [`NotNan`] values.
    ///
    /// # Errors
    ///
    /// Returns [`ObserverError::InvalidFloatValue`] if any of the supplied
    /// floating-point values is `NaN`.
    pub fn from_parallax(
        longitude: Radians,
        rho_cos_phi: f64,
        rho_sin_phi: f64,
        name: Option<String>,
        ra_accuracy: Option<f64>,
        dec_accuracy: Option<f64>,
    ) -> Result<Observer, ObserverError> {
        Ok(Observer {
            longitude: NotNan::try_from(longitude)?,
            rho_cos_phi: NotNan::try_from(rho_cos_phi)?,
            rho_sin_phi: NotNan::try_from(rho_sin_phi)?,
            name,
            ra_accuracy: ra_accuracy.to_notnan()?,
            dec_accuracy: dec_accuracy.to_notnan()?,
        })
    }

    /// Recover geodetic latitude and ellipsoidal height (WGS-84) from parallax constants.
    ///
    /// Inverts the stored parallax coordinates `(ρ·cosφ, ρ·sinφ)` to the **geodetic**
    /// latitude `φ` (degrees) and the ellipsoidal height `h` (meters) above the
    /// WGS-84 reference ellipsoid. The computation uses **Bowring’s closed-form**
    /// formula (no iteration), which is usually sufficient for double-precision
    /// accuracy at the centimeter level or better.
    ///
    /// Units & model
    /// -------------
    /// * Inputs: `ρ·cosφ` and `ρ·sinφ` are dimensionless, expressed in **Earth radii**
    ///   (normalized by the equatorial radius). They are scaled internally by `a`
    ///   (the equatorial radius) to meters.
    /// * Output: latitude in **degrees**, height in **meters** (ellipsoidal height, not geoid/orthometric).
    /// * Ellipsoid: WGS-84 radii from `constants.rs` (`EARTH_MAJOR_AXIS` = `a`, `EARTH_MINOR_AXIS` = `b`).
    ///   If you prefer exact GRS-80 reproduction, use consistent `b` there; the difference vs WGS-84 is sub-millimetric.
    ///
    /// Notes
    /// -----
    /// * This routine **does not** compute the geodetic longitude; it only returns `(lat, h)`.
    ///   Your `Observer` already stores the geodetic longitude independently.
    /// * Numerical robustness is good across latitudes, including near the poles.
    /// * If you require bit-for-bit parity with an external reference using a different ellipsoid,
    ///   ensure `a`/`b` match that reference.
    ///
    /// Arguments
    /// -----------------
    /// * None.
    ///
    /// Return
    /// ----------
    /// * `(geodetic_latitude_deg, height_meters)` — latitude in degrees, ellipsoidal height in meters.
    ///
    /// See also
    /// ------------
    /// * [`geodetic_to_parallax`] – Forward conversion used at construction.
    /// * [`Observer::from_parallax`] – Builds an observer from `(ρ·cosφ, ρ·sinφ)`.
    /// * `constants::EARTH_MAJOR_AXIS` / `constants::EARTH_MINOR_AXIS` – Ellipsoid radii used here.
    pub fn geodetic_lat_height_wgs84(&self) -> (Radians, Meters) {
        let a = EARTH_MAJOR_AXIS;
        let b = EARTH_MINOR_AXIS;
        let e2 = 1.0 - (b * b) / (a * a);
        let ep2 = (a * a) / (b * b) - 1.0;

        let p = self.rho_cos_phi.into_inner() * a; // distance in equatorial plane [m]
        let z = self.rho_sin_phi.into_inner() * a; // z [m]

        // Bowring’s formula:
        let theta = (z * a).atan2(p * b);
        let st = theta.sin();
        let ct = theta.cos();
        let phi = (z + ep2 * b * st.powi(3)).atan2(p - e2 * a * ct.powi(3));

        let s = phi.sin();
        let n = a / (1.0 - e2 * s * s).sqrt();
        let h = p / phi.cos() - n;

        (phi, h)
    }

    /// Geocentric latitude derived from the parallax constants.
    ///
    /// Computes the angle between the equatorial plane and the direction from
    /// Earth's center to the observer, expressed in degrees.
    ///
    /// This differs from the geodetic latitude, which is defined relative to
    /// the WGS-84 ellipsoid normal. The two coincide only on a perfect sphere.
    ///
    /// $$\varphi_{\text{geo}} = \arctan\!\left(\frac{\rho\sin\varphi}{\rho\cos\varphi}\right)$$
    ///
    /// Return
    /// ------
    /// Geocentric latitude in degrees, in $[-90°, +90°]$.
    pub fn geocentric_lat_deg(&self) -> Degrees {
        let rc = self.rho_cos_phi.into_inner();
        let rs = self.rho_sin_phi.into_inner();
        rs.atan2(rc).to_degrees()
    }

    /// Geocentric distance (observer–Earth-center) in Earth radii.
    ///
    /// Derived from the IAU parallax constants $(\rho\cos\varphi,\, \rho\sin\varphi)$
    /// stored in the observer record. By definition these constants already encode
    /// the distance in units of the equatorial Earth radius $R_\oplus$, so no
    /// additional scaling is required.
    ///
    /// $$\rho = \sqrt{(\rho\cos\varphi)^2 + (\rho\sin\varphi)^2}$$
    ///
    /// Return
    /// ------
    /// Geocentric distance $\rho$ in Earth radii $R_\oplus$.
    pub fn geocentric_distance_earth_radii(&self) -> f64 {
        let rc = self.rho_cos_phi.into_inner();
        let rs = self.rho_sin_phi.into_inner();
        rc.hypot(rs)
    }
}

/// Convert geodetic latitude in **radians** and height in **meters** into
/// normalized parallax coordinates on the Earth.
///
/// The result accounts for the Earth's oblateness using the reference
/// ellipsoid defined by [`EARTH_MAJOR_AXIS`] (equatorial radius $a$, in
/// metres) and [`EARTH_MINOR_AXIS`] (polar radius $b$, in metres).
///
/// The standard geodetic-to-geocentric conversion is applied:
///
/// ```text
/// u        = atan( (b/a) · sin φ / cos φ )
/// ρ·sin φ' = (b/a) · sin u + (h/a) · sin φ
/// ρ·cos φ' = cos u + (h/a) · cos φ
/// ```
///
/// where $\varphi$ is the geodetic latitude and $h$ is the ellipsoidal height.
///
/// # Arguments
///
/// - `lat` — geodetic latitude of the observer in **radians**.
/// - `height` — observer's altitude above the reference ellipsoid in **meters**.
///
/// # Returns
///
/// A tuple `(rho_cos_phi, rho_sin_phi)` where:
///
/// - `rho_cos_phi` — $\rho\cos\phi'$, normalized projection on the equatorial plane.
/// - `rho_sin_phi` — $\rho\sin\phi'$, normalized projection on the polar axis.
pub fn geodetic_to_parallax(lat: Radians, height: Meters) -> (f64, f64) {
    // Ratio of the Earth's minor to major axis (flattening factor)
    let axis_ratio = EARTH_MINOR_AXIS / EARTH_MAJOR_AXIS;

    // Compute the auxiliary angle u (parametric latitude)
    // This corrects for the Earth's oblateness.
    let u = (lat.sin() * axis_ratio).atan2(lat.cos());

    // Compute the normalized distance along the polar axis
    let rho_sin_phi = axis_ratio * u.sin() + (height / EARTH_MAJOR_AXIS) * lat.sin();

    // Compute the normalized distance along the equatorial plane
    let rho_cos_phi = u.cos() + (height / EARTH_MAJOR_AXIS) * lat.cos();

    (rho_cos_phi, rho_sin_phi)
}

impl Display for Observer {
    /// Pretty-print an observer with optional verbose details using `{:#}` formatting.
    ///
    /// Default formatting (`{}`) prints a compact one-liner:
    /// `Name (lon: XX.XXXXXX°, lat: YY.YYYYYY° geodetic, elev: Z.ZZ km)`.
    ///
    /// Alternate formatting (`{:#}`) prints a multi-line detailed block including:
    /// - Parallax constants `(ρ·cosφ, ρ·sinφ)`,
    /// - Geocentric latitude `φ_geo` and geocentric distance `ρ` (Earth radii),
    /// - Astrometric 1-σ accuracies in arcseconds (if available).
    ///
    /// Arguments
    /// -----------------
    /// * `self`: The observer to format.
    ///
    /// Return
    /// ----------
    /// * A human-readable representation suitable for logs and diagnostics.
    ///
    /// See also
    /// ------------
    /// * [`Observer::geodetic_lat_height_wgs84`] – Geodetic latitude (deg) and ellipsoidal height (m).
    /// * [`geodetic_to_parallax`] – Forward conversion to `(ρ·cosφ, ρ·sinφ)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.name.as_deref().unwrap_or("Unnamed");
        let (lat_deg, h_m) = self.geodetic_lat_height_wgs84();
        let lon_deg = self.longitude.into_inner();

        write!(
            f,
            "{name} (lon: {lon_deg:.6}°, lat: {lat_deg:.6}° geodetic, elev: {h_m:.2} m)"
        )?;

        if !f.alternate() {
            return Ok(());
        }

        let rc = self.rho_cos_phi.into_inner();
        let rs = self.rho_sin_phi.into_inner();

        let fmt_accuracy = |v: Option<NotNan<f64>>| {
            v.map_or_else(
                || "—".to_string(),
                |v| format!("{:.3}″", v.into_inner().to_degrees() * 3600.0),
            )
        };

        writeln!(f)?;
        writeln!(
            f,
            "  parallax: ρ·cosφ={rc:.9}, ρ·sinφ={rs:+.9}  |  φ_geo={:+.6}°  ρ={:.6} RE",
            self.geocentric_lat_deg(),
            self.geocentric_distance_earth_radii(),
        )?;
        write!(
            f,
            "  astrometric 1σ: RA={}, DEC={}",
            fmt_accuracy(self.ra_accuracy),
            fmt_accuracy(self.dec_accuracy),
        )
    }
}

#[cfg(test)]
mod observer_test {
    use super::*;

    fn to_dynamic_observer(
        longitude: Radians,
        latitude: Radians,
        height: Meters,
        name: Option<String>,
        ra_accuracy: Option<f64>,
        dec_accuracy: Option<f64>,
    ) -> Observer {
        Observer::new(longitude, latitude, height, name, ra_accuracy, dec_accuracy)
            .expect("Failed to create Observer")
    }

    #[test]
    fn test_observer_constructor() {
        let observer = to_dynamic_observer(0.0, 0.0, 0.0, None, None, None);
        assert_eq!(observer.longitude, 0.0);
        assert_eq!(observer.rho_cos_phi, 1.0);
        assert_eq!(observer.rho_sin_phi, 0.0);

        let observer = Observer::new(
            289.25058_f64.to_radians(),
            -30.2446_f64.to_radians(),
            2647.,
            Some("Rubin Observatory".to_string()),
            Some(0.0001),
            Some(0.0001),
        )
        .unwrap();

        assert_eq!(observer.longitude, 289.25058_f64.to_radians());
        assert_eq!(observer.rho_cos_phi, 0.8649760504617418);
        assert_eq!(observer.rho_sin_phi, -0.5009551027512434);
    }

    #[test]
    fn geodetic_to_parallax_test() {
        // latitude and height of Pan-STARRS 1, Haleakala
        let (pxy1, pz1) = geodetic_to_parallax(20.707233557_f64.to_radians(), 3067.694);
        assert_eq!(pxy1, 0.9362410003211518);
        assert_eq!(pz1, 0.35154299856304305);
    }

    #[cfg(test)]
    mod geodetic_inverse_tests {
        use super::*;
        use approx::assert_abs_diff_eq;

        /// Round-trip a single site through (lon, lat, h) -> parallax -> inverse
        /// and check that we recover the original geodetic latitude & height.
        ///
        /// Notes
        /// -----
        /// * `Observer::new` is given `h_m` in meters (as per current API usage).
        /// * `geodetic_lat_height_wgs84()` returns height in **meters**; we convert to meters.
        fn roundtrip_site(name: &str, lon_deg: Degrees, lat_deg: Degrees, h_m: Meters) {
            // Build observer (forward: geodetic -> parallax is done inside `Observer::new`)
            let obs = to_dynamic_observer(
                lon_deg.to_radians(),
                lat_deg.to_radians(),
                h_m,
                Some(name.to_string()),
                None,
                None,
            );

            // Inverse: parallax -> geodetic (WGS-84)
            let (lat_rec_rad, h_rec_m) = obs.geodetic_lat_height_wgs84();

            // Tolerances:
            // - Latitude: 1e-6 deg (~3.6 mas) – tight but should pass for double precision Bowring + 0–1 Newton step
            // - Height:   1e-2 m
            let tol_lat_deg = 1e-6;
            let tol_h_m = 1e-2;

            let lat_rec_deg = lat_rec_rad.to_degrees();

            assert_abs_diff_eq!(lat_rec_deg, lat_deg, epsilon = tol_lat_deg);
            assert_abs_diff_eq!(h_rec_m, h_m, epsilon = tol_h_m);
        }

        /// See also
        /// ------------
        /// * [`Observer::new`] – Forward geodetic->parallax conversion under test by round-trip.
        /// * [`Observer::from_parallax`] – Alternative constructor, if you want to inject ρ·cosφ/ρ·sinφ.
        /// * `geodetic_to_parallax` – The forward routine used internally by `Observer::new`.

        #[test]
        fn geodetic_roundtrip_known_observatories_wgs84() {
            // NOTE:
            // The heights below are commonly quoted "above sea level" (orthometric).
            // For pure algorithmic round-trip testing, that's acceptable because we feed
            // the same height into forward and inverse. If you want strict ellipsoidal
            // (h) values, substitute official WGS-84 heights here.
            let sites: &[(&str, Degrees, Degrees, Meters)] = &[
                // name,                      lon_deg (E+), lat_deg (N+), height_m
                ("Haleakala (PS1 I41)", -156.2575, 20.7075, 3055.0),
                ("Mauna Kea (CFHT)", -155.4700, 19.8261, 4205.0),
                ("ESO Paranal", -70.4025, -24.6252, 2635.0),
                ("Cerro Pachon (Rubin)", -70.7366, -30.2407, 2663.0),
                ("La Silla", -70.7346, -29.2613, 2400.0),
                ("Kitt Peak", -111.5967, 31.9583, 2096.0),
                ("Roque de los Muchachos", -17.8947, 28.7606, 2396.0),
            ];

            for (name, lon, lat, h_m) in sites.iter().copied() {
                roundtrip_site(name, lon, lat, h_m);
            }
        }

        #[test]
        fn geodetic_roundtrip_extremes_equator_and_pole() {
            // Equator, sea level
            roundtrip_site("Equator (0°, 0 m)", 0.0, 0.0, 0.0);

            // Near-North-Pole and Near-South-Pole with modest height
            roundtrip_site("Near North Pole", 0.0, 89.999, 1000.0);
            roundtrip_site("Near South Pole", 0.0, -89.999, 1000.0);
        }

        #[test]
        fn geodetic_roundtrip_high_altitude_and_negative() {
            // Very high site (simulate balloon/aircraft)
            roundtrip_site("High Alt 10 m", 10.0, 45.0, 10_000.0);

            // Negative height (below ellipsoid, synthetic but tests robustness)
            roundtrip_site("Below ellipsoid -50 m", -30.0, -10.0, -50.0);
        }
    }

    #[cfg(test)]
    mod observer_display_tests {
        use super::*;

        /// Convert arcseconds to radians.
        #[inline]
        fn arcsec_to_rad(as_val: f64) -> f64 {
            // 1 arcsec = π / (180 * 3600) rad
            std::f64::consts::PI / (180.0 * 3600.0) * as_val
        }

        /// Helper to build an Observer with optional RA/DEC accuracies (in arcseconds).
        fn make_dynamic_observer_with_acc(
            name: Option<&str>,
            lon_deg: f64,
            lat_deg: f64,
            elev_m: f64,
            ra_as: Option<f64>,
            dec_as: Option<f64>,
        ) -> Observer {
            let ra_sigma = ra_as.map(arcsec_to_rad);
            let dec_sigma = dec_as.map(arcsec_to_rad);

            to_dynamic_observer(
                lon_deg,
                lat_deg,
                elev_m, // elevation in meters
                name.map(|s| s.to_string()),
                ra_sigma,
                dec_sigma,
            )
        }

        /// Compact formatting must be a single line with name/lon/lat/elev.
        #[test]
        fn display_compact_single_line() {
            let obs = make_dynamic_observer_with_acc(Some("TestSite"), 10.0, 0.0, 0.0, None, None);

            let s = format!("{obs}");
            // Must not contain newlines in compact form
            assert!(
                !s.contains('\n'),
                "Compact format should be single-line, got:\n{s}"
            );

            // Must contain expected fragments (predictable numbers)
            assert!(
                s.contains("TestSite (lon: 10.000000°"),
                "Missing name/lon fragment. Got:\n{s}"
            );
            assert!(
                s.contains("lat: 0.000000° geodetic"),
                "Missing geodetic latitude fragment. Got:\n{s}"
            );
            assert!(
                s.contains("elev: 0.00 m"),
                "Missing elevation fragment (m). Got:\n{s}"
            );
        }

        /// Alternate formatting must be multi-line and include parallax & uncertainties.
        #[test]
        fn display_verbose_multiline_with_sections() {
            let obs = make_dynamic_observer_with_acc(
                Some("VerboseSite"),
                -70.0,
                -30.0,
                2400.0,
                None,
                None,
            );

            let s = format!("{obs:#}");

            // Must contain multiple lines and the expected section headers/fragments
            assert!(
                s.contains('\n'),
                "Verbose format should be multi-line. Got:\n{s}"
            );
            assert!(
                s.starts_with("VerboseSite (lon: -70.000000°"),
                "First line should start with site name and lon. Got:\n{s}"
            );
            assert!(
                s.contains("\n  parallax: ρ·cosφ="),
                "Missing 'parallax:' line. Got:\n{s}"
            );
            assert!(
                s.contains("φ_geo=") && s.contains("ρ="),
                "Missing φ_geo/ρ fragments. Got:\n{s}"
            );
            assert!(
                s.contains("\n  astrometric 1σ: RA=—, DEC=—"),
                "Missing astrometric 1σ line with em-dashes for None. Got:\n{s}"
            );
        }

        /// When RA/DEC accuracies are provided, they must be printed in arcseconds with three decimals.
        #[test]
        fn display_verbose_shows_ra_dec_sigmas() {
            // RA = 1.0″, DEC = 2.5″ (passed in arcseconds, converted to radians internally)
            let obs = make_dynamic_observer_with_acc(
                Some("AccSite"),
                0.0,
                0.0,
                0.0,
                Some(1.0),
                Some(2.5),
            );

            let s = format!("{obs:#}");

            assert!(
                s.contains("astrometric 1σ: RA=1.000″, DEC=2.500″"),
                "Expected RA/DEC sigma in arcseconds with 3 decimals. Got:\n{s}"
            );
        }

        /// Name fallback should be "Unnamed" when not provided.
        #[test]
        fn display_uses_unnamed_when_missing() {
            let obs = make_dynamic_observer_with_acc(None, 5.0, 0.0, 0.0, None, None);
            let s = format!("{obs}");
            assert!(
                s.starts_with("Unnamed (lon: 5.000000°"),
                "Expected 'Unnamed' fallback. Got:\n{s}"
            );
        }

        /// Basic numeric sanity: geodetic height is shown in kilometers in the display.
        /// For a 3055 m elevation, we expect ~3.055 km (rounded to 2 decimals).
        #[test]
        fn display_elevation_shown_in_km() {
            let obs = make_dynamic_observer_with_acc(
                Some("Haleakala-ish"),
                -156.2575,
                20.7075,
                3055.0,
                None,
                None,
            );

            let s = format!("{obs}");
            // Check the km conversion and rounding only; don't assert on latitude value here.
            assert!(
                s.contains("elev: 3055.00 m"),
                "Expected elevation ~3055 m rounded to 2 decimals. Got:\n{s}"
            );

            // Optional: ensure lon is printed correctly
            assert!(
                s.contains("lon: -156.257500°"),
                "Longitude formatting mismatch. Got:\n{s}"
            );
        }
    }
}
