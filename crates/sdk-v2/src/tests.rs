use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_stark_backend::StarkEngine;
use openvm_transpiler::elf::Elf;

use crate::{
    config::{
        default_app_params, AggregationSystemParams, DEFAULT_APP_LOG_BLOWUP, DEFAULT_APP_L_SKIP,
    },
    Sdk, StdIn,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

#[test]
fn test_root_prover_trace_heights() -> Result<()> {
    let n_stack = 19;
    let app_params = default_app_params(DEFAULT_APP_LOG_BLOWUP, DEFAULT_APP_L_SKIP, n_stack);
    let agg_params = AggregationSystemParams::default();

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;

    let sdk = Sdk::riscv32(app_params, agg_params);
    let app_exe = sdk.convert_to_exe(elf)?;
    let mut evm_prover = sdk.evm_prover(app_exe)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    let proof = evm_prover.prove(stdin)?;
    let vk = evm_prover.root_prover.0.get_vk();
    let engine = E::new(vk.inner.params.clone());
    engine.verify(&vk, &proof)?;

    Ok(())
}
