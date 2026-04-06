use std::{fs, path::PathBuf};

use clap::Parser;
use eyre::{Result, WrapErr, eyre};
use openvm_circuit::system::memory::{CHUNK, merkle::public_values::UserPublicValuesProof};
use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config as SC, F};
use openvm_verify_stark_host::{
    NonRootStarkProof, verify_vm_stark_proof_decoded,
    vk::{NonRootStarkVerifyingKey, read_vk_from_file},
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Verify an RTP final proof using only a cached VM verifying key bundle"
)]
struct Args {
    /// Path to the copied RTP final proof file.
    #[arg(long)]
    proof: PathBuf,

    /// Path to a cached VM verifying key bundle.
    #[arg(long)]
    vm_vk: PathBuf,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct RtpProofWithPublicValue<Field> {
    proof: Proof<SC>,
    user_public_values: Option<UserPublicValuesProof<CHUNK, Field>>,
}

fn load_rtp_final_proof(path: &PathBuf) -> Result<NonRootStarkProof> {
    let proof_bytes =
        fs::read(path).wrap_err_with(|| format!("Failed to read RTP final proof {}", path.display()))?;
    let proof: RtpProofWithPublicValue<F> = bincode::deserialize(&proof_bytes)
        .wrap_err_with(|| format!("Failed to deserialize RTP final proof {}", path.display()))?;

    let user_pvs_proof = proof.user_public_values.ok_or_else(|| {
        eyre!(
            "Proof {} does not include user public values; this is not a final RTP proof",
            path.display()
        )
    })?;

    Ok(NonRootStarkProof {
        inner: proof.proof,
        user_pvs_proof,
        deferral_merkle_proofs: None,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let vk: NonRootStarkVerifyingKey = read_vk_from_file(&args.vm_vk)
        .wrap_err_with(|| format!("Failed to read VM verifying key {}", args.vm_vk.display()))?;
    let proof = load_rtp_final_proof(&args.proof)?;

    verify_vm_stark_proof_decoded(&vk, &proof).wrap_err("OpenVM STARK verification failed")?;

    println!("Proof verified successfully: {}", args.proof.display());
    Ok(())
}
