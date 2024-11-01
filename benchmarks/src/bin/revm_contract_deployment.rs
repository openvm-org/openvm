#![cfg(feature = "revm")]
#![allow(unused_variables)]
#![allow(unused_imports)]
use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
};
use axvm_benchmarks::utils::{bench_from_exe, build_bench_program};
use axvm_circuit::arch::VmConfig;
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_recursion::testing_utils::inner::build_verification_program;
use eyre::Result;
use p3_field::AbstractField;
use revm_interpreter::opcode;
use tracing::info_span;

/// Load number parameter and set to storage with slot 0
const INIT_CODE: &[u8] = &[
    opcode::PUSH1,
    0x01,
    opcode::PUSH1,
    0x17,
    opcode::PUSH1,
    0x1f,
    opcode::CODECOPY,
    opcode::PUSH0,
    opcode::MLOAD,
    opcode::PUSH0,
    opcode::SSTORE,
];

/// Copy runtime bytecode to memory and return
const RET: &[u8] = &[
    opcode::PUSH1,
    0x02,
    opcode::PUSH1,
    0x15,
    opcode::PUSH0,
    opcode::CODECOPY,
    opcode::PUSH1,
    0x02,
    opcode::PUSH0,
    opcode::RETURN,
];

/// Load storage from slot zero to memory
const RUNTIME_BYTECODE: &[u8] = &[opcode::PUSH0, opcode::SLOAD];

fn main() -> Result<()> {
    // TODO[jpw]: benchmark different combinations
    let app_log_blowup = 1;
    // let agg_log_blowup = 3;

    let param = 0x42;
    let bytecode: Vec<u8> = [INIT_CODE, RET, RUNTIME_BYTECODE, &[param]].concat();

    let elf = build_bench_program("revm_contract_deployment")?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vdata = info_span!(
            "revm Contract Deployment",
            group = "revm_contract_deployment"
        )
        .in_scope(|| {
            let engine = BabyBearPoseidon2Engine::new(
                FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
            );
            let bytecode = bytecode
                .into_iter()
                .map(AbstractField::from_canonical_u8)
                .collect();
            bench_from_exe(engine, VmConfig::rv32im(), elf, vec![bytecode])
        })?;
        Ok(())
    })
}
