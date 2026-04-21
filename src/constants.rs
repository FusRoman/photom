//! Physical and astronomical constants used throughout the crate.
//!
//! All values are plain `f64` constants.  Where a standard definition exists
//! (IAU, GRS1980/WGS84) the source is noted in the individual item's
//! documentation.
//!
//! ## Public items
//!
//! | Constant | Value | Description |
//! |----------|-------|-------------|
//! | [`DPI`] | 2π | Full turn in radians |
//! | [`AU`] | 149 597 870.7 km | Astronomical Unit (IAU 2012) |
//! | [`EARTH_MAJOR_AXIS`] | 6 378 137.0 m | Earth equatorial radius (WGS84) |
//! | [`EARTH_MINOR_AXIS`] | 6 356 752.3 m | Earth polar radius (WGS84) |
//! | [`ERAU`] | ≈ 4.263 × 10⁻⁵ AU | Earth radius in astronomical units |
//! | [`ARCSEC_TO_DEG`] | 1/3600 | Arc-second to degree conversion factor |

/// 2π, useful for trigonometric conversions
pub const DPI: f64 = 2. * std::f64::consts::PI;

/// Astronomical Unit in kilometers (IAU 2012)
pub const AU: f64 = 149_597_870.7;

/// Earth equatorial radius in meters (GRS1980/WGS84)
pub const EARTH_MAJOR_AXIS: f64 = 6_378_137.0;

/// Earth polar radius in meters (GRS1980/WGS84)
pub const EARTH_MINOR_AXIS: f64 = 6_356_752.3;

/// Earth radius expressed in astronomical units
pub const ERAU: f64 = (EARTH_MAJOR_AXIS / 1000.) / AU;

/// Arcseconds to degrees conversion factor.
pub const ARCSEC_TO_DEG: f64 = 1.0 / 3600.0;
