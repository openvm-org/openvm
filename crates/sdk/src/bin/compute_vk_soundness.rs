//! Computes the conservative soundness (in bits) of an OpenVM verifying key using the
//! `stark-backend` soundness calculator (`SoundnessCalculator::calculate_from_vk`).
//!
//! Takes a path to a serialized verifying key, as produced by the OpenVM CLI/SDK via
//! `write_object_to_file` (bitcode). The file may hold either an [`AppVerifyingKey`] (e.g.
//! `app.vk`) or a bare [`MultiStarkVerifyingKey`] (e.g. the internal-recursive aggregation vk);
//! the format is auto-detected.
//!
//! Run with:
//!   cargo run --bin compute_vk_soundness --release -p openvm-sdk -- <PATH_TO_VK>

use std::path::PathBuf;

use clap::Parser;
use openvm_sdk::{fs::read_object_from_file, keygen::AppVerifyingKey, SC};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, soundness::SoundnessCalculator};
use serde_json::json;

#[derive(Parser)]
#[command(about = "Compute the conservative soundness (bits) of an OpenVM verifying key")]
struct Args {
    /// Path to a serialized verifying key (bitcode), either an `AppVerifyingKey` (e.g. `app.vk`)
    /// or a bare `MultiStarkVerifyingKey` (e.g. the internal-recursive aggregation vk).
    vk: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();
    let vk = load_vk(&args.vk)?;
    report(&args.vk.display().to_string(), &vk)
}

/// Deserializes a verifying key, auto-detecting whether the file holds an [`AppVerifyingKey`] or a
/// bare [`MultiStarkVerifyingKey`].
fn load_vk(path: &PathBuf) -> eyre::Result<MultiStarkVerifyingKey<SC>> {
    if let Ok(app_vk) = read_object_from_file::<AppVerifyingKey, _>(path) {
        return Ok(app_vk.vk);
    }
    read_object_from_file::<MultiStarkVerifyingKey<SC>, _>(path).map_err(|e| {
        eyre::eyre!(
            "failed to deserialize {} as either an AppVerifyingKey or a MultiStarkVerifyingKey: {e}",
            path.display(),
        )
    })
}

/// Computes and prints the soundness breakdown for a single verifying key.
fn report(name: &str, vk: &MultiStarkVerifyingKey<SC>) -> eyre::Result<()> {
    let s = SoundnessCalculator::calculate_from_vk(vk);
    let params = &vk.inner.params;

    let report = json!({
        "vk": name,
        "params": {
            "num_airs": vk.inner.per_air.len(),
            "l_skip": params.l_skip,
            "n_stack": params.n_stack,
            "log_blowup": params.log_blowup,
            "whir_rounds": params.whir.rounds.len(),
        },
        // Security bits per proof-system component; `total` is the minimum across all of them.
        "security_bits": {
            "logup": s.logup_bits,
            "gkr_sumcheck": s.gkr_sumcheck_bits,
            "gkr_batching": s.gkr_batching_bits,
            "zerocheck_sumcheck": s.zerocheck_sumcheck_bits,
            "constraint_batching": s.constraint_batching_bits,
            "stacked_reduction": s.stacked_reduction_bits,
            "whir": s.whir_bits,
            "total": s.total_bits,
        },
    });

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
