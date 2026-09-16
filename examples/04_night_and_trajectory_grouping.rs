//! Group observations by trajectory, resolve alternate designations, and
//! see how the night index behaves when a source format does not build one.
//!
//! Requires the `mpc_80_col` feature (to load the fixture file):
//!
//! ```sh
//! cargo run --example 04_night_and_trajectory_grouping --features mpc_80_col
//! ```

use std::collections::BTreeMap;

use camino::Utf8PathBuf;
use photom::{TrajId, observation_dataset::ObsDataset};

fn data(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2015AB.obs holds 37 observations of one physical object, spread over
    // several nights, and referenced under two different provisional
    // designations ("K09R05F" and its later alias "K15A00B").
    let dataset = ObsDataset::from_mpc_80_col(data("2015AB.obs"))?;
    println!("{:#}", dataset);

    // 1. Night grouping is an *optional* index: it is only populated by
    //    ingestion backends that carry an explicit `night_id` column
    //    (Polars, DataFusion). The MPC 80-column and ADES readers do not
    //    build one, so every night-related accessor correctly returns
    //    `None` here rather than an empty-but-present index.
    assert!(dataset.nb_night().is_none());
    assert!(dataset.iter_night_id().is_none());
    println!("\nno night index on this dataset (expected for mpc_80_col ingestion)");

    // When you need "everything seen on the same UT night" from a format
    // without a night index, group by the integer part of `mjd_tt` yourself:
    let mut by_night: BTreeMap<i64, usize> = BTreeMap::new();
    for obs in dataset.iter_observations() {
        *by_night.entry(obs.mjd_tt().floor() as i64).or_default() += 1;
    }
    println!("manually grouped by integer MJD day:");
    for (day, count) in &by_night {
        println!("  MJD {day}: {count} observation(s)");
    }

    // 2. Trajectory grouping, by contrast, is built by every ingestion
    //    backend that carries an object identifier (here, the MPC packed
    //    designation on each 80-column line).
    let traj_ids: Vec<&TrajId> = dataset.iter_traj_id().unwrap().collect();
    println!("\ntrajectories recorded: {}", traj_ids.len());
    for traj_id in &traj_ids {
        let count = dataset.len_trajectory((*traj_id).clone()).unwrap();
        println!("  {traj_id}: {count} observation(s)");
    }

    // 3. `resolve_alias` recovers the canonical TrajId from an alternate
    //    designation registered by the ingestion backend. Here "K15A00B" was
    //    a later designation of the same object as the primary "K09R05F".
    match dataset.resolve_alias("K15A00B") {
        Some(primary) => println!("\n\"K15A00B\" resolves to primary trajectory {primary}"),
        None => println!("\n\"K15A00B\" is not a known alias in this dataset"),
    }
    // The primary designation itself is not registered as an alias of anything.
    assert!(dataset.resolve_alias("K09R05F").is_none());

    // 4. `iter_trajectory_observations` / `materialize_trajectory` give
    //    direct access to the observations of one trajectory, in insertion
    //    order, without touching the rest of the dataset.
    let primary = TrajId::Str("K09R05F".to_string());
    let first_three: Vec<_> = dataset
        .iter_trajectory_observations(primary.clone())
        .expect("trajectory exists")
        .take(3)
        .collect();
    println!("\nfirst 3 observations of {primary}:");
    for obs in first_three {
        println!("  {obs:#}");
    }

    Ok(())
}
