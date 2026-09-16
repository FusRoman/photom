//! Build custom `Observer`s two different ways, inspect their geodetic
//! properties, and attach them to a dataset.
//!
//! No feature required:
//!
//! ```sh
//! cargo run --example 06_observers_basics
//! ```

use photom::{
    observation_dataset::ObsDataset,
    observer::{Observer, geodetic_to_parallax},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. The geodetic constructor: longitude, latitude and elevation are the
    //    numbers you would read off a survey plate or Wikipedia. `Observer::new`
    //    converts them internally to the MPC parallax convention.
    let paranal = Observer::new(
        (-70.4025_f64).to_radians(),
        (-24.6252_f64).to_radians(),
        2635.0,
        Some("ESO Paranal".to_string()),
        None,
        None,
    )?;

    // 2. The parallax constructor: use it directly when rho*cos(phi') and
    //    rho*sin(phi') are already known (e.g. read from an MPC observatory
    //    table). Here we derive them from the same geodetic input as above,
    //    via the free function `geodetic_to_parallax`, to show that both
    //    constructors agree.
    let (rho_cos_phi, rho_sin_phi) = geodetic_to_parallax((-24.6252_f64).to_radians(), 2635.0);
    let paranal_from_parallax = Observer::from_parallax(
        (-70.4025_f64).to_radians(),
        rho_cos_phi,
        rho_sin_phi,
        Some("ESO Paranal (via parallax)".to_string()),
        None,
        None,
    )?;

    println!("geodetic constructor : {paranal}");
    println!("parallax constructor : {paranal_from_parallax}");

    // 3. Geodetic/geocentric readouts derived from the stored parallax
    //    constants. `geodetic_lat_height_wgs84` inverts the WGS-84 ellipsoid
    //    model; `geocentric_lat_deg`/`geocentric_distance_earth_radii` treat
    //    Earth as a sphere centered at its own center of mass.
    let (lat_rad, height_m) = paranal.geodetic_lat_height_wgs84();
    println!(
        "\ngeodetic latitude  : {:.6} deg, height: {:.1} m",
        lat_rad.to_degrees(),
        height_m
    );
    println!(
        "geocentric latitude: {:.6} deg",
        paranal.geocentric_lat_deg()
    );
    println!(
        "geocentric distance: {:.6} Earth radii",
        paranal.geocentric_distance_earth_radii()
    );
    println!("\nverbose display:\n{paranal:#}");

    // 4. Custom observers are attached to a dataset via `push_observer`,
    //    which returns the `ObserverId::IntId` to reference from
    //    `ObservationInput::observer`. `get_observer` resolves it back.
    let (dataset, observer_id) = ObsDataset::empty().push_observer(paranal);
    println!("\nregistered as {observer_id:?} (no observation references it yet)");
    assert!(
        dataset.get_observer(0).is_none(),
        "no observation exists with id=0 yet"
    );

    Ok(())
}
