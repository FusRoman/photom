//! Equatorial and Cartesian sky coordinates: construction, geometry, and
//! covariance round-tripping.
//!
//! No feature required:
//!
//! ```sh
//! cargo run --example 08_coordinates_equatorial_cartesian
//! ```

use photom::coordinates::{
    cartesian::CartesianCoord,
    equatorial::{EquCoord, EquCoordCov},
};

const SEED_ERROR_ARCSEC: f64 = 0.05;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. `EquCoord` bundles RA/Dec with their 1-sigma uncertainties, all in
    //    radians internally. `from_degrees` is the convenient constructor
    //    when your source data is in degrees, as most catalogs are.
    // EquCoord is Copy, so both `a` and `b` can be reused freely below.
    let a = EquCoord::from_degrees(
        10.684,
        SEED_ERROR_ARCSEC / 3600.0,
        41.269,
        SEED_ERROR_ARCSEC / 3600.0,
    );
    let b = EquCoord::from_degrees(
        10.700,
        SEED_ERROR_ARCSEC / 3600.0,
        41.280,
        SEED_ERROR_ARCSEC / 3600.0,
    );
    println!("a: {a}");
    println!("b: {b}");

    // 2. Geometry on the sphere: angular separation via the numerically
    //    stable Vincenty formula, and the spherical midpoint (vector
    //    averaging, not a naive average of RA/Dec which breaks near the
    //    poles or across the 0h/24h wrap).
    let separation_arcsec = a.angular_separation(&b).to_degrees() * 3600.0;
    println!("\nangular separation: {separation_arcsec:.2} arcsec");

    let midpoint = a.spherical_midpoint(&b);
    println!("spherical midpoint: {midpoint}");

    // 3. Cartesian conversion: a lossless projection onto the unit sphere,
    //    convenient for dot products and vector algebra. Uncertainties are
    //    not carried over by this simple `From` conversion (see step 4 for
    //    the version that does).
    let cart_a: CartesianCoord = a.into();
    let cart_b: CartesianCoord = b.into();
    println!(
        "\na as Cartesian: ({:.6}, {:.6}, {:.6})",
        cart_a.x, cart_a.y, cart_a.z
    );
    println!(
        "a . b (cosine of the angle between them): {:.9}",
        cart_a.dot(&cart_b)
    );

    // 4. Covariance propagation: `EquCoordCov` carries a full 2x2 covariance
    //    (not just the two marginal sigmas), and `to_cartesian_cov` /
    //    `to_equatorial_cov` propagate it through the (nonlinear)
    //    spherical <-> Cartesian map via first-order Jacobians.
    let cov_a: EquCoordCov = a.into();
    let cart_cov = cov_a.to_cartesian_cov();
    let (ex, ey, ez) = cart_cov.errors();
    println!("\npropagated Cartesian 1-sigma errors: dx={ex:.3e}, dy={ey:.3e}, dz={ez:.3e}");

    let back_to_equ = cart_cov.to_equatorial_cov();
    let (ra_err_deg, dec_err_deg) = back_to_equ.coord.error_in_degrees();
    println!(
        "round-tripped back to equatorial: ra_err={:.6} arcsec, dec_err={:.6} arcsec (started at {SEED_ERROR_ARCSEC:.6} arcsec)",
        ra_err_deg * 3600.0,
        dec_err_deg * 3600.0,
    );

    Ok(())
}
