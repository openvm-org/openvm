use ax_stark_sdk::ax_stark_backend::p3_field::AbstractField;
use axvm_circuit::{
    arch::{hasher::poseidon2::vm_poseidon2_hasher, VmConfig, VmExecutor},
    system::memory::tree::public_values::compute_user_public_values_proof,
    utils::air_test_with_min_segments,
};
use axvm_transpiler::{axvm_platform::bincode, elf::ELF_DEFAULT_MAX_NUM_PUBLIC_VALUES};
use eyre::Result;
use p3_baby_bear::BabyBear;
use test_case::test_case;

use crate::utils::build_example_program;

type F = BabyBear;

#[test_case("fibonacci", 1)]
fn test_rv32i_prove(example_name: &str, min_segments: usize) -> Result<()> {
    let elf = build_example_program(example_name)?;
    let config = VmConfig::rv32i();
    air_test_with_min_segments(config, elf, vec![], min_segments);
    Ok(())
}

#[test]
fn test_read_vec_runtime() -> Result<()> {
    let elf = build_example_program("hint")?;
    let executor = VmExecutor::<F>::new(VmConfig::rv32i());
    executor.execute(elf, vec![[0, 1, 2, 3].map(F::from_canonical_u8).to_vec()])?;
    Ok(())
}

#[test]
fn test_read_runtime() -> Result<()> {
    let elf = build_example_program("read")?;
    let executor = VmExecutor::<F>::new(VmConfig::rv32i());

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
    let config = VmConfig::rv32i();
    let executor = VmExecutor::<F>::new(config.clone());
    let final_memory = executor.execute(elf, vec![])?.unwrap();
    let hasher = vm_poseidon2_hasher();
    let pv_proof = compute_user_public_values_proof(
        config.memory_config.memory_dimensions(),
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
