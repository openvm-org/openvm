use std::rc::Rc;

use ax_stark_sdk::ax_stark_backend::p3_field::AbstractField;
use axvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher,
        instructions::exe::AxVmExe,
        new_vm::{self, VmExecutor as NewVmExecutor},
        VmConfig, VmExecutor,
    },
    system::memory::tree::public_values::UserPublicValuesProof,
    utils::new_air_test_with_min_segments,
};
use axvm_keccak256_circuit::Keccak256Rv32Config;
use axvm_keccak_transpiler::KeccakTranspilerExtension;
use axvm_rv32im_circuit::{Rv32IConfig, Rv32ImConfig};
use axvm_transpiler::{
    axvm_platform::bincode, elf::ELF_DEFAULT_MAX_NUM_PUBLIC_VALUES, transpiler::Transpiler, FromElf,
};
use eyre::Result;
use p3_baby_bear::BabyBear;
use test_case::test_case;

use crate::utils::{build_example_program, build_example_program_with_features};

type F = BabyBear;

/// TODO: remove vm::VmExecutor and use new_vm::VmExecutor everywhere when all VmExtensions are implemented
#[test_case("fibonacci", 1)]
fn test_rv32i_prove(example_name: &str, min_segments: usize) -> Result<()> {
    let elf = build_example_program(example_name)?;
    let config = Rv32IConfig::default();
    new_air_test_with_min_segments(config, elf, vec![], min_segments);
    Ok(())
}

#[test_case("collatz", 1)]
fn test_rv32im_prove(example_name: &str, min_segments: usize) -> Result<()> {
    let elf = build_example_program(example_name)?;
    let config = Rv32ImConfig::default();
    new_air_test_with_min_segments(config, elf, vec![], min_segments);
    Ok(())
}

// #[test_case("fibonacci", 1)]
#[test_case("collatz", 1)]
fn test_rv32im_std_prove(example_name: &str, min_segments: usize) -> Result<()> {
    let elf = build_example_program_with_features(example_name, ["std"])?;
    let config = Rv32ImConfig::default();
    new_air_test_with_min_segments(config, elf, vec![], min_segments);
    Ok(())
}

#[test]
fn test_read_vec_runtime() -> Result<()> {
    let elf = build_example_program("hint")?;
    let config = Rv32IConfig::default();
    let executor = new_vm::VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![[0, 1, 2, 3].map(F::from_canonical_u8).to_vec()])?;
    Ok(())
}

#[test]
fn test_read_runtime() -> Result<()> {
    let elf = build_example_program("read")?;
    let config = Rv32IConfig::default();
    let executor = new_vm::VmExecutor::<F, _>::new(config);

    #[derive(serde::Serialize)]
    struct Foo {
        bar: u32,
        baz: Vec<u32>,
    }
    let foo = Foo {
        bar: 42,
        baz: vec![0, 1, 2, 3],
    };
    let serialized_foo = bincode::serde::encode_to_vec(&foo, bincode::config::standard())
        .expect("serialize to vec failed");
    executor
        .execute(
            elf,
            vec![serialized_foo
                .into_iter()
                .map(F::from_canonical_u8)
                .collect()],
        )
        .unwrap();
    Ok(())
}

#[test]
fn test_reveal_runtime() -> Result<()> {
    let elf = build_example_program("reveal")?;
    let config = Rv32IConfig::default();
    let executor = new_vm::VmExecutor::<F, _>::new(config.clone());
    let final_memory = executor.execute(elf, vec![])?.unwrap();
    let hasher = vm_poseidon2_hasher();
    let pv_proof = UserPublicValuesProof::compute(
        config.system.memory_config.memory_dimensions(),
        ELF_DEFAULT_MAX_NUM_PUBLIC_VALUES,
        &hasher,
        &final_memory,
    );
    assert_eq!(
        pv_proof.public_values,
        [123, 0, 456, 0u32, 0u32, 0u32, 0u32, 0u32]
            .into_iter()
            .flat_map(|x| x.to_le_bytes())
            .map(F::from_canonical_u8)
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn test_keccak256_runtime() -> Result<()> {
    let elf = build_example_program("keccak")?;
    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default_with_intrinsics()
            .with_processor(Rc::new(KeccakTranspilerExtension)),
    );
    let executor = NewVmExecutor::<F, Keccak256Rv32Config>::new(Keccak256Rv32Config::default());
    executor.execute(axvm_exe, vec![])?;
    Ok(())
}

#[test]
fn test_print_runtime() -> Result<()> {
    let elf = build_example_program("print")?;
    let config = Rv32IConfig::default();
    let executor = new_vm::VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_modular_runtime() -> Result<()> {
    let elf = build_example_program("little")?;
    let executor = VmExecutor::<F>::new(VmConfig::rv32im().add_canonical_modulus());
    executor.execute(elf, vec![])?;
    Ok(())
}

// TODO[yi]: add back this test once we have support for modular extension
/*
#[test]
fn test_matrix_power_runtime() -> Result<()> {
    let elf = build_example_program("matrix-power")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_int256_alu()
            .add_int256_m()
            .add_int256_branch(),
    );
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_matrix_power_signed_runtime() -> Result<()> {
    let elf = build_example_program("matrix-power-signed")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_int256_alu()
            .add_int256_m()
            .add_int256_branch(),
    );
    executor.execute(elf, vec![])?;
    Ok(())
}

*/
