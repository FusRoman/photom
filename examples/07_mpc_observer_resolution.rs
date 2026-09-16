//! Resolve MPC-coded observers lazily from the Minor Planet Center website.
//!
//! Unlike custom `Observer`s (see `06_observers_basics`), observations tied
//! to an MPC three-letter code (e.g. "G96", "T08") do not carry their site's
//! geodetic data directly: `photom` fetches the full MPC observatory table
//! on first use and caches it on disk.
//!
//! This example makes a **network request** the first time it runs (and
//! writes to a local cache directory afterwards, see `PHOTOM_MPC_CACHE_DIR`).
//! It requires the `mpc_80_col` feature:
//!
//! ```sh
//! cargo run --example 07_mpc_observer_resolution --features mpc_80_col
//! ```

use camino::Utf8PathBuf;
use photom::{observation_dataset::ObsDataset, observer::error_model::ObsErrorModel};

fn data(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 8467.obs was recorded by MPC-coded sites ("T08", "W68", ...): each
    // observation stores only the 3-byte code, not the site's coordinates.
    let dataset = ObsDataset::from_mpc_80_col(data("8467.obs"))?
        // An error model is required before an MPC-coded observer can be
        // resolved: it is also used to assign the site's nominal astrometric
        // accuracy once its coordinates are known.
        .with_error_model(ObsErrorModel::FCCT14);

    // `get_observer` triggers, on its *first* call for any MPC-coded
    // observation, a fetch of the full MPC observatory table
    // (https://minorplanetcenter.net/iau/lists/ObsCodes.html). The result is
    // cached in a `OnceLock` for the lifetime of the dataset, and mirrored on
    // disk (default OS cache dir, or `PHOTOM_MPC_CACHE_DIR` if set) so
    // subsequent runs of any program do not repeat the network call.
    let first = dataset
        .iter_observations()
        .next()
        .expect("file is non-empty");
    match dataset.get_observer(*first.id()) {
        Some(observer) => println!("resolved observer for obs id={}:\n{observer:#}", first.id()),
        None => println!(
            "could not resolve the observer for obs id={} (network unavailable?)",
            first.id()
        ),
    }

    // Once the table is cached, `iter_observer` walks every observer touched
    // by this dataset — custom sites and MPC codes alike — without any
    // further network activity.
    match dataset.iter_observer() {
        Ok(iter) => {
            for (id, observer) in iter {
                println!("{id:?} -> {observer}");
            }
        }
        Err(e) => eprintln!("iter_observer failed: {e}"),
    }

    Ok(())
}
