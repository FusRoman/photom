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
