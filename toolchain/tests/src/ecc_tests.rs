use std::str::FromStr;

use axvm_circuit::{
    arch::{ExecutorName, VmConfig, VmExecutor},
    intrinsics::modular::SECP256K1_COORD_PRIME,
};
use eyre::Result;
use serde_arrays::*;
use p3_baby_bear::BabyBear;

use crate::utils::{build_example_program, generate_keystore_address, generate_msg_hash, generate_vk_hash};

type F = BabyBear;

#[test]
fn test_moduli_setup_runtime() -> Result<()> {
    let elf = build_example_program("moduli_setup")?;
    let exe = axvm_circuit::arch::instructions::exe::AxVmExe::<F>::from(elf.clone());
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im().add_modular_support(
            exe.custom_op_config
                .intrinsics
                .field_arithmetic
                .primes
                .iter()
                .map(|s| num_bigint_dig::BigUint::from_str(s).unwrap())
                .collect(),
        ),
    );
    executor.execute(elf, vec![])?;
    assert!(!executor.config.supported_modulus.is_empty());
    Ok(())
}

#[test]
fn test_modular_runtime() -> Result<()> {
    let elf = build_example_program("little")?;
    let executor = VmExecutor::<F>::new(VmConfig::rv32im().add_canonical_modulus());
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_complex_runtime() -> Result<()> {
    let elf = build_example_program("complex")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_modular_support(vec![SECP256K1_COORD_PRIME.clone()])
            .add_complex_ext_support(vec![SECP256K1_COORD_PRIME.clone()]),
    );
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_ec_runtime() -> Result<()> {
    let elf = build_example_program("ec")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_canonical_modulus()
            .add_canonical_ec_curves(),
    );
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_ecdsa_runtime() -> Result<()> {
    let elf = build_example_program("ecdsa")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_executor(ExecutorName::Keccak256Rv32)
            .add_canonical_modulus()
            .add_canonical_ec_curves(),
    );
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_m_of_n_ecdsa() -> Result<()> {
    

    let elf = build_example_program("ecdsa_m_of_n")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_executor(ExecutorName::Keccak256Rv32)
            .add_canonical_modulus()
            .add_canonical_ec_curves(),
    );

    #[derive(serde::Serialize)]
    struct Inputs {
        /// calculated keystore address
        keystore_address: [u8; 32],
        /// message hash
        msg_hash: [u8; 32],
        /// m number of signatures
        m: u32,
        /// n number of EOAs
        n: u32,
        /// vector of signatures
        #[serde(with = "BigArray")]
        signatures: Vec<[u8; 64]>,
        /// vector of EOAs
        #[serde(with = "BigArray")]
        eoa_addrs: Vec<[u8; 20]>,
    }

    let eoa_addrs = vec![];
    let vk_hash = generate_vk_hash(vk);

    let inputs = Inputs {
        keystore_address: generate_keystore_address(salt, eoa_addrs, vk_hash),
        msg_hash: generate_msg_hash("message"),
        m: 2,
        n: 3,
        signatures: vec![],
        eoa_addrs,
    };
    let serialized_inputs = bincode::serde::encode_to_vec(&inputs, bincode::config::standard())
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

    executor.execute(elf, vec![])?;
    Ok(())
}
