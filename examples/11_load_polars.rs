//! Load observations from a Polars `DataFrame`, with both MPC-coded and
//! custom geodetic observers, plus night/trajectory grouping columns.
//!
//! Unlike the MPC 80-column and ADES readers (see `02_load_mpc80col` and
//! `04_night_and_trajectory_grouping`), the Polars backend *does* build a
//! night index -- whenever the input frame carries a `night_id` column.
//!
//! Requires the `polars` feature:
//!
//! ```sh
//! cargo run --example 11_load_polars --features polars
//! ```

use photom::{io::polars::FromPolarsArgs, observation_dataset::ObsDataset};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build a small in-memory DataFrame matching the schema documented in
    //    the README ("DataFrame / Parquet Schema"): mandatory `id`, `ra`,
    //    `ra_err`, `dec`, `dec_err`, `magnitude`, `mag_err`, `filter`,
    //    `mjd_tt`, plus the optional grouping columns `traj_id`/`night_id`
    //    and the optional observer columns.
    //
    //    Rows 0-1 are seen by an MPC-coded site ("G96"); rows 2-3 by a
    //    custom geodetic site with no MPC code. Both observer-resolution
    //    paths can coexist in the same frame -- resolution is per-row.
    let df = df![
        "id"          => [0u64, 1, 2, 3],
        "ra"          => [0.10_f64, 0.11, 0.30, 0.31],
        "ra_err"      => [1e-6_f64, 1e-6, 1e-6, 1e-6],
        "dec"         => [0.50_f64, 0.50, 0.60, 0.60],
        "dec_err"     => [1e-6_f64, 1e-6, 1e-6, 1e-6],
        "magnitude"   => [18.0_f64, 18.1, 18.2, 18.3],
        "mag_err"     => [0.05_f64, 0.05, 0.05, 0.05],
        "filter"      => ["g", "g", "r", "r"],
        "mjd_tt"      => [60_000.0_f64, 60_000.1, 60_001.0, 60_001.1],
        "traj_id"     => [1u32, 1, 2, 2],
        "night_id"    => [60_000u32, 60_000, 60_001, 60_001],
        "mpc_code_obs" => [Some("G96"), Some("G96"), None, None],
        "obs_lon"     => [None, None, Some(6.86_f64.to_radians()), Some(6.86_f64.to_radians())],
        "obs_lat"     => [None, None, Some(43.75_f64.to_radians()), Some(43.75_f64.to_radians())],
        "obs_alt"     => [None, None, Some(1270.0_f64), Some(1270.0_f64)],
        "obs_ra_acc"  => [None, None, Some(1e-6_f64), Some(1e-6_f64)],
        "obs_dec_acc" => [None, None, Some(1e-6_f64), Some(1e-6_f64)],
    ]?;

    // 2. `from_polars` validates the schema and assembles the dataset.
    //    `FromPolarsArgs::default()` sorts by `night_id` to keep each
    //    night's rows contiguous in memory; the trajectory index falls back
    //    to a scattered ("Split") representation in that case.
    let dataset = ObsDataset::from_polars(&df, FromPolarsArgs::default())?;
    println!("{:#}", dataset);

    // 3. Night grouping works out of the box here, unlike the text-based
    //    ingestion backends (see `04_night_and_trajectory_grouping`).
    println!("\nnights recorded: {}", dataset.nb_night().unwrap());
    for night_id in dataset.iter_night_id().unwrap() {
        println!(
            "  {night_id}: {} observation(s), contiguous={}",
            dataset.len_night(night_id).unwrap(),
            dataset.is_night_contiguous(night_id).unwrap()
        );
    }

    // 4. Observer resolution: rows 0-1 resolve through their MPC code
    //    (network access on first use, see `07_mpc_observer_resolution`),
    //    rows 2-3 through the geodetic columns, interned as a custom site.
    let obs_2 = dataset.get_observation(2).expect("id 2 was inserted");
    let observer = dataset
        .get_observer(2)
        .expect("rows 2-3 carry a full geodetic triplet");
    println!("\nobservation id=2 was recorded by a custom site: {observer}");
    println!("its measurement: {}", obs_2.photometry());

    // 5. `from_lazy` accepts a `LazyFrame` instead, running the same
    //    validation after the lazy plan is collected -- handy when the
    //    frame comes from `scan_parquet`/`scan_csv` rather than being
    //    built in memory.
    let via_lazy = ObsDataset::from_lazy(df.lazy(), FromPolarsArgs::default())?;
    assert_eq!(via_lazy.observation_count(), dataset.observation_count());
    println!(
        "\nfrom_lazy gives the same observation count: {}",
        via_lazy.observation_count()
    );

    Ok(())
}
