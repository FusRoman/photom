//! Parallel iteration over observations and trajectories via `rayon`.
//!
//! Every sequential iterator on `ObsDataset` (see `04_night_and_trajectory_grouping`)
//! has a `par_`-prefixed counterpart with the exact same semantics, once the
//! `parallel` feature is enabled. They return `rayon` parallel iterators, so
//! the usual `rayon::prelude` combinators (`.map`, `.sum`, `.reduce`, ...)
//! apply directly, with zero copying of the underlying data.
//!
//! Requires the `parallel` and `mpc_80_col` features:
//!
//! ```sh
//! cargo run --example 12_parallel_iteration --features parallel,mpc_80_col
//! ```

use camino::Utf8PathBuf;
use photom::observation_dataset::ObsDataset;
use rayon::prelude::*;

fn data(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2015AB.obs: 37 observations of a single trajectory ("K09R05F").
    let dataset = ObsDataset::from_mpc_80_col(data("2015AB.obs"))?;
    println!("{:#}", dataset);

    // 1. `par_iter_observations` mirrors `iter_observations`, but is a
    //    `rayon` `IndexedParallelIterator`: any `rayon::prelude` combinator
    //    works directly on it.
    let sequential_mean: f64 = dataset
        .iter_observations()
        .map(|o| o.photometry().magnitude)
        .sum::<f64>()
        / dataset.observation_count() as f64;

    let parallel_mean: f64 = dataset
        .par_iter_observations()
        .map(|o| o.photometry().magnitude)
        .sum::<f64>()
        / dataset.observation_count() as f64;

    println!(
        "\nmean magnitude: sequential={sequential_mean:.4}, parallel={parallel_mean:.4} (must match)"
    );
    assert!((sequential_mean - parallel_mean).abs() < 1e-12);

    // 2. `par_iter_traj_id` / `par_iter_trajectory_observations` mirror the
    //    trajectory-grouping accessors from `04_night_and_trajectory_grouping`.
    //    Each trajectory can be processed independently and in parallel --
    //    useful once a dataset holds thousands of trajectories rather than
    //    the single one in this small fixture.
    if let Some(traj_ids) = dataset.par_iter_traj_id() {
        let traj_ids: Vec<_> = traj_ids.collect();
        println!("\ntrajectories (parallel-collected): {}", traj_ids.len());

        for traj_id in traj_ids {
            let n = dataset
                .par_iter_trajectory_observations(traj_id.clone())
                .expect("trajectory exists")
                .count();
            println!("  {traj_id}: {n} observation(s)");
        }
    }

    // 3. A parallel `reduce` over declinations: `reduce` takes an identity
    //    element and a combining function, both applied in parallel across
    //    work-stealing chunks -- unlike `min`/`max`, this keeps the
    //    reduction itself parallel rather than falling back to a serial scan.
    let dec_min = dataset
        .par_iter_observations()
        .map(|o| o.equ_coord().dec)
        .reduce(|| f64::INFINITY, f64::min);
    let dec_max = dataset
        .par_iter_observations()
        .map(|o| o.equ_coord().dec)
        .reduce(|| f64::NEG_INFINITY, f64::max);
    println!(
        "\ndeclination range across the dataset: {:.6} deg to {:.6} deg",
        dec_min.to_degrees(),
        dec_max.to_degrees()
    );

    Ok(())
}
