//! Load observations from MPC ADES XML files, and combine sources with `ObsDatasetBuilder`.
//!
//! Requires the `ades` feature:
//!
//! ```sh
//! cargo run --example 03_load_ades --features ades
//! ```

use camino::Utf8PathBuf;
use photom::{ObsDatasetBuilder, observation_dataset::ObsDataset};

fn data(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. `from_ades` parses one file. `error_ra`/`error_dec` are fallback
    //    1-sigma uncertainties (arcseconds) used when the file itself does
    //    not carry astrometric error columns; `None` keeps whatever the file
    //    provides (or zero, if it provides nothing).
    let single = ObsDataset::from_ades(data("example_ades.xml"), None, None)?;
    println!("example_ades.xml alone -> {:#}", single);

    // 2. `from_ades_files` loads and merges several ADES files in one call,
    //    collecting per-file parse failures as warnings instead of failing
    //    the whole load.
    let paths = [data("example_ades.xml"), data("example_ades2.xml")];
    let (combined, warnings) = ObsDataset::from_ades_files(&paths, None, None);
    for (path, err) in &warnings {
        eprintln!("skipped {path}: {err}");
    }
    println!("\ncombined ADES files -> {:#}", combined);

    // 3. `ObsDatasetBuilder` is the idiomatic way to combine several source
    //    *formats* (ADES, MPC 80-column, ...) into a single dataset in one
    //    pass, still collecting warnings for any file that fails to parse.
    let (built, warnings) = ObsDatasetBuilder::new()
        .add_ades(&paths, None, None)
        .build();
    for w in &warnings {
        eprintln!("builder warning: {w}");
    }
    println!(
        "\nObsDatasetBuilder result matches from_ades_files: {}",
        built.observation_count() == combined.observation_count()
    );

    Ok(())
}
