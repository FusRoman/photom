//! Load observations from the classic MPC 80-column ASCII format.
//!
//! Requires the `mpc_80_col` feature:
//!
//! ```sh
//! cargo run --example 02_load_mpc80col --features mpc_80_col
//! ```

use camino::Utf8PathBuf;
use photom::observation_dataset::ObsDataset;

/// Resolve a fixture path relative to the crate root, the same idiom used
/// throughout this crate's own integration tests.
fn data(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. `from_mpc_80_col` reads a single file. Each line is a fixed-width
    //    80-column record; a file can contain several trajectories.
    let single = ObsDataset::from_mpc_80_col(data("8467.obs"))?;
    println!("8467.obs alone -> {:#}", single);

    // 2. `from_mpc_80_col_files` loads several files at once and merges them
    //    into one dataset. Files that fail to parse are *not* fatal: they are
    //    collected as `(path, error)` pairs instead of aborting the whole load.
    let paths = [data("8467.obs"), data("33803.obs")];
    let (combined, warnings) = ObsDataset::from_mpc_80_col_files(&paths);

    for (path, err) in &warnings {
        eprintln!("skipped {path}: {err}");
    }
    println!(
        "\n8467.obs + 33803.obs merged -> {} observations",
        combined.observation_count()
    );
    println!("{:#}", combined);

    // 3. `extend_from_mpc_80_col` does the same thing but starting from an
    //    already-built dataset, useful when files arrive incrementally.
    let extended = single.extend_from_mpc_80_col(&[data("33803.obs")]).0;
    assert_eq!(extended.observation_count(), combined.observation_count());
    println!(
        "\nextend_from_mpc_80_col gives the same total: {}",
        extended.observation_count()
    );

    Ok(())
}
