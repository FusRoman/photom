//! Ecliptic coordinates, gnomonic (tangent-plane) projection, and reading a
//! 2x2 covariance as a confidence ellipse.
//!
//! No feature required:
//!
//! ```sh
//! cargo run --example 09_coordinates_ecliptic_and_projection
//! ```

use photom::coordinates::{
    cov2::Cov2, ecliptic::EclipticCoord, equatorial::EquCoord, gnomonic_projection::TangentPlane,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Ecliptic coordinates are the natural frame for solar-system object
    //    kinematics: proper motions are nearly parallel to the ecliptic
    //    plane. Conversion from equatorial is a fixed rotation by the J2000
    //    obliquity, exposed as `From<EquCoord> for EclipticCoord`.
    let ra_dec = EquCoord::from_degrees(10.684, 5e-6, 41.269, 5e-6);
    let ecliptic: EclipticCoord = ra_dec.into();
    println!("equatorial : {ra_dec}");
    println!("ecliptic   : {ecliptic}");

    // Round-trip back to equatorial to confirm the conversion is lossless
    // for the position itself (only the covariance-aware variant needs a
    // Jacobian; the bare position <-> position map is an exact rotation).
    let back: EquCoord = ecliptic.into();
    println!(
        "round-trip error: {:.3e} deg in RA, {:.3e} deg in Dec",
        (back.ra - ra_dec.ra).to_degrees(),
        (back.dec - ra_dec.dec).to_degrees()
    );

    // 2. Gnomonic (tangent-plane) projection: pick a reference point on the
    //    sky and project nearby positions onto a flat local frame. This is
    //    the standard trick for short-arc astrometry, where working in a
    //    local Euclidean frame is much simpler than on the sphere.
    let tangent_point = EquCoord::from_degrees(10.684, 5e-6, 41.269, 5e-6);
    let plane = TangentPlane::new(tangent_point);

    let nearby = EquCoord::from_degrees(10.700, 5e-6, 41.280, 5e-6);
    let projected = plane.project(&nearby);
    println!(
        "\nprojected offset from tangent point: dx={:.3e} rad, dy={:.3e} rad",
        projected.x, projected.y
    );

    // `unproject` inverts the map; for a point that was itself the tangent
    // point, we recover the exact original RA/Dec.
    let recovered = projected.unproject();
    println!(
        "unprojected back to: ra={:.6} deg, dec={:.6} deg (expected ra={:.6} deg, dec={:.6} deg)",
        recovered.ra.to_degrees(),
        recovered.dec.to_degrees(),
        nearby.ra.to_degrees(),
        nearby.dec.to_degrees()
    );

    // 3. `Cov2` is the tangent-plane covariance: once an astrometric error
    //    ellipse has been projected onto the local plane, its eigenvalues
    //    give the semi-axes of the 1-sigma confidence ellipse, and the
    //    Mahalanobis distance measures "how many sigmas away" a point is.
    let cov = Cov2::from_equ(&nearby);
    println!(
        "\nconfidence ellipse semi-axes: {:.3e} rad (major), {:.3e} rad (minor)",
        cov.lambda_max().sqrt(),
        cov.lambda_min().sqrt()
    );

    // `mahalanobis_sq` needs `Cov2::inverse`, which treats the matrix as
    // singular once |det| underflows `f64::EPSILON`. Since det scales as
    // sigma^4, realistic radian-scale astrometric covariances (sigma ~ 1e-6
    // rad) already underflow that threshold, so `cov` itself cannot be
    // inverted here. Use a coarser, illustrative covariance instead to keep
    // the demo numerically meaningful.
    let coarse_cov = Cov2::diag(1e-4 * 1e-4, 2e-4 * 2e-4);
    let offset = projected - plane.project(&tangent_point);
    match coarse_cov.mahalanobis_sq(offset) {
        Some(d2) => println!(
            "Mahalanobis distance of the projected offset: {:.3} sigma",
            d2.sqrt()
        ),
        None => println!("covariance is singular; Mahalanobis distance undefined"),
    }

    Ok(())
}
