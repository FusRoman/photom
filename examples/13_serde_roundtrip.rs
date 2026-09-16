//! Serialise and deserialise an `ObsDataset`, and choose the index layout
//! used when reading it back.
//!
//! `ObsDataset` implements `serde::Serialize`/`Deserialize` and works with
//! any serde-compatible format; this example uses `serde_json` since it is
//! already a dependency of the `serde` feature.
//!
//! Requires the `serde` and `mpc_80_col` features:
//!
//! ```sh
//! cargo run --example 13_serde_roundtrip --features serde,mpc_80_col
//! ```

use camino::Utf8PathBuf;
use photom::{
    IndexLayout, ObsDatasetSeed, observation_dataset::ObsDataset,
    observer::error_model::ObsErrorModel,
};
use serde::de::DeserializeSeed as _;

fn data(name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A small real dataset with a trajectory index and an error model
    // attached, so the round-trip exercises more than just the raw
    // observation list.
    let dataset =
        ObsDataset::from_mpc_80_col(data("2015AB.obs"))?.with_error_model(ObsErrorModel::VFCC17);
    println!("original: {:#}", dataset);

    // 1. Plain `serde_json::to_string` / `from_str` round-trip. The default
    //    `Deserialize` impl always rebuilds indices with the `Split` layout,
    //    which is safe regardless of how the data was originally laid out.
    let json = serde_json::to_string(&dataset)?;
    println!("\nserialised to {} bytes of JSON", json.len());

    let restored: ObsDataset = serde_json::from_str(&json)?;
    println!("restored (Split layout): {:#}", restored);
    assert_eq!(restored.observation_count(), dataset.observation_count());

    // Observations, the trajectory index, and the error model all survive
    // the round-trip; only the lazily-initialised MPC network cache does
    // not (it is re-fetched on first use after deserialisation, see
    // `07_mpc_observer_resolution`).
    assert_eq!(restored.get_error_model(), dataset.get_error_model());
    assert_eq!(
        restored.resolve_alias("K15A00B"),
        dataset.resolve_alias("K15A00B")
    );

    // 2. `ObsDatasetSeed` lets you request a contiguous index layout instead,
    //    which can speed up subsequent night/trajectory look-ups. It works
    //    with any format that exposes its `Deserializer` publicly.
    let mut de = serde_json::Deserializer::from_str(&json);
    let restored_contiguous: ObsDataset = ObsDatasetSeed {
        layout: IndexLayout::TryContiguous,
    }
    .deserialize(&mut de)?;
    println!(
        "\nrestored with TryContiguous layout: {:#}",
        restored_contiguous
    );
    assert_eq!(
        restored_contiguous.observation_count(),
        dataset.observation_count()
    );

    Ok(())
}
