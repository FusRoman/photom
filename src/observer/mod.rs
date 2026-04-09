//! Observer metadata and geodetic conversion utilities.
//!
//! This module defines the [`Observer`] struct, which represents a ground-based
//! observatory site, together with the two constructors that create one from
//! geodetic or from pre-computed geocentric parallax coordinates.  It also
//! exposes the low-level geodetic conversion helpers
//! ([`geodetic_to_parallax`], [`lat_alt_to_parallax`]) and the error type
//! ([`ObserverError`]) that covers all failure modes arising during observer
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
//! | [`geodetic_to_parallax`] | fn | Degrees-based wrapper for [`lat_alt_to_parallax`] |
//! | [`lat_alt_to_parallax`] | fn | Geodetic-to-geocentric parallax conversion (radians) |

pub mod error_model;
pub mod mpc;

use thiserror::Error;

use ordered_float::NotNan;

use crate::{
    Degrees, Meters, Radians,
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
pub struct Observer {
    /// Geodetic longitude in **degrees** east of Greenwich.
    pub longitude: NotNan<Degrees>,

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
    /// - `longitude` — geodetic longitude in **degrees** (east positive).
    /// - `latitude` — geodetic latitude in **degrees**.
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
        longitude: Degrees,
        latitude: Degrees,
        elevation: Meters,
        name: Option<String>,
        ra_accuracy: Option<f64>,
        dec_accuracy: Option<f64>,
    ) -> Result<Observer, ObserverError> {
        let (rho_cos_phi, rho_sin_phi) = geodetic_to_parallax(latitude, elevation);

        Ok(Observer {
            longitude: NotNan::try_from(longitude)?,
            rho_cos_phi: NotNan::try_from(rho_cos_phi)?,
            rho_sin_phi: NotNan::try_from(rho_sin_phi)?,
            name,
            ra_accuracy: to_opt_notnan(ra_accuracy)?,
            dec_accuracy: to_opt_notnan(dec_accuracy)?,
        })
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
    /// - `longitude` — geodetic longitude in **degrees** (east positive).
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
        longitude: Degrees,
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
            ra_accuracy: to_opt_notnan(ra_accuracy)?,
            dec_accuracy: to_opt_notnan(dec_accuracy)?,
        })
    }
}

/// Lift an `Option<f64>` into an `Option<NotNan<f64>>`, propagating `NaN` as an error.
///
/// - `None` passes through as `Ok(None)`.
/// - `Some(x)` where `x` is finite becomes `Ok(Some(NotNan::new(x)))`.
/// - `Some(NaN)` returns `Err(FloatIsNan)`.
///
/// # Arguments
///
/// - `x` — the optional floating-point value to wrap.
///
/// # Returns
///
/// `Ok(Some(NotNan<f64>))` when `x` is `Some` and finite, `Ok(None)` when
/// `x` is `None`, or `Err(FloatIsNan)` when `x` is `Some(NaN)`.
///
/// # Errors
///
/// Returns [`ordered_float::FloatIsNan`] if `x` is `Some(NaN)`.
#[inline]
pub fn to_opt_notnan(x: Option<f64>) -> Result<Option<NotNan<f64>>, ordered_float::FloatIsNan> {
    x.map(NotNan::new).transpose()
}

/// Convert geodetic latitude in **degrees** and height in **meters** into
/// normalized parallax coordinates.
///
/// This is a thin wrapper around [`lat_alt_to_parallax`] that converts `lat`
/// from degrees to radians before delegating.
///
/// # Arguments
///
/// - `lat` — geodetic latitude of the observer in **degrees**.
/// - `height` — observer's altitude above the reference ellipsoid in **meters**.
///
/// # Returns
///
/// A tuple `(rho_cos_phi, rho_sin_phi)` where:
///
/// - `rho_cos_phi` — $\rho\cos\phi'$, normalized projection on the equatorial plane.
/// - `rho_sin_phi` — $\rho\sin\phi'$, normalized projection on the polar axis.
pub fn geodetic_to_parallax(lat: f64, height: f64) -> (f64, f64) {
    // Convert latitude from degrees to radians
    let latitude_rad = lat.to_radians();

    // Call the main routine that works with radians
    let (rho_cos_phi, rho_sin_phi) = lat_alt_to_parallax(latitude_rad, height);

    (rho_cos_phi, rho_sin_phi)
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
pub fn lat_alt_to_parallax(lat: f64, height: f64) -> (f64, f64) {
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
