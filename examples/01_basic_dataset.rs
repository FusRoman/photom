//! Build an `ObsDataset` from scratch, without loading any file.
//!
//! This is the starting point for understanding `photom`'s core data model:
//! an `ObsDataset` is just a collection of `Observation`s, each optionally
//! linked to an `Observer`. Run with:
//!
//! ```sh
//! cargo run --example 01_basic_dataset
//! ```

use photom::{
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, observation::ObservationInput},
    observer::Observer,
    photometry::{Filter, Photometry},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Start from an empty dataset. No observations, no observers yet.
    let dataset = ObsDataset::empty();

    // 2. Register an observer. `push_observer` returns the `ObserverId` we
    //    must reference from each `ObservationInput` that was recorded there.
    let site = Observer::new(
        6.86_f64.to_radians(),  // longitude, radians east of Greenwich
        43.75_f64.to_radians(), // geodetic latitude, radians
        1270.0,                 // elevation, meters
        Some("OCA Calern".to_string()),
        None,
        None,
    )?;
    let (dataset, site_id) = dataset.push_observer(site);

    // 3. Build a few observations by hand. Coordinates are equatorial
    //    (RA/Dec) with their 1-sigma uncertainties, all in radians here via
    //    `from_degrees` for readability.
    let observations = vec![
        ObservationInput::new(
            0,
            EquCoord::from_degrees(10.684_f64, 5e-6, 41.269_f64, 5e-6),
            Photometry {
                magnitude: 18.4,
                error: 0.05,
                filter: Filter::String("g".to_string()),
            },
            60_000.123_456, // epoch, MJD (TT)
            Some(site_id),
        ),
        ObservationInput::new(
            1,
            EquCoord::from_degrees(10.685_f64, 5e-6, 41.270_f64, 5e-6),
            Photometry {
                magnitude: 18.5,
                error: 0.06,
                filter: Filter::String("g".to_string()),
            },
            60_000.223_456,
            Some(site_id),
        ),
    ];

    // 4. `push_observation` consumes the dataset and returns it along with
    //    the internal `ObsIndex` of each newly inserted observation.
    let (dataset, indices) = dataset.push_observation(observations)?;
    println!("inserted at internal indices: {indices:?}");

    // 5. Read the data back: by ObsId, by iteration, or via the observer link.
    println!("observation_count = {}", dataset.observation_count());

    let first = dataset
        .get_observation(0)
        .expect("observation 0 was just inserted");
    println!("\nlookup by id=0:\n{first}");

    for obs in dataset.iter_observations() {
        let observer = dataset.get_observer(*obs.id());
        println!(
            "id={} observer={}",
            obs.id(),
            observer.map(|o| o.to_string()).unwrap_or_default()
        );
    }

    // 6. `ObsDataset` has a `Display` impl: `{}` for a verbose summary block,
    //    `{:#}` for a compact one-liner.
    println!("\n{dataset}");
    println!("{:#}", dataset);

    Ok(())
}
