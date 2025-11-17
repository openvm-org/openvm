use std::{env, fs, path::PathBuf};

use verify_stark::{verify_vm_stark_proof, vk::read_vk_from_file};

fn main() -> eyre::Result<()> {
    let mut args = env::args_os();
    // Skip program name
    let _ = args.next();

    let vk_path = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: verify-stark <vk_path> <proof_path>"))?;
    let proof_path = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: verify-stark <vk_path> <proof_path>"))?;

    let vk = read_vk_from_file(vk_path)?;
    let encoded_proof = fs::read(proof_path)?;

    verify_vm_stark_proof(&vk, &encoded_proof)?;
    println!("Proof verified successfully!");

    Ok(())
}
