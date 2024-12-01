use std::str::FromStr;

use ax_stark_sdk::ax_stark_backend::p3_field::AbstractField;
use axvm::intrinsics::keccak256;
use axvm_circuit::{
    arch::{ExecutorName, VmConfig, VmExecutor},
    intrinsics::modular::SECP256K1_COORD_PRIME,
};
use axvm_transpiler::axvm_platform::bincode;
use eyre::Result;
use hex_literal::hex;
use p3_baby_bear::BabyBear;

use crate::utils::{build_example_program, ecdsa_sign, generate_keystore_address, generate_salt};

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
    use crate::utils::{serialize_u8_20_vec, serialize_u8_65_vec};

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
        #[serde(serialize_with = "serialize_u8_65_vec")]
        signatures: Vec<[u8; 65]>,
        /// vector of EOAs
        #[serde(serialize_with = "serialize_u8_20_vec")]
        eoa_addrs: Vec<[u8; 20]>,
    }

    let msg = "message";
    let msg_hash = keccak256(msg.as_bytes());

    #[allow(clippy::useless_vec)]
    let private_keys = vec![
        hex!("a62970f3597fe2a380571fd51e3f962b2d13eb97219beee4cc360165b21d74df"),
        // hex!("0c7580ce67946a4480e671cfa63d68eb9692e59f289a08b1fd114b4fccd18a3b"),
        hex!("15b11c59e8755155ff74f035a97abce181c33c756329337752eeaf5228f49ea8"),
    ];
    let eoa_addrs = vec![
        hex!("4f7a43ff0e4e4224c724cdd39ae69db06e6d3410"),
        hex!("130eb76ab81b76a74cf6820ecf3820cdc5179609"),
        hex!("b7f1d32f02bd66d25c9ff1f60b67acc4790b0930"),
    ];

    let vk_hash = keccak256(&hex!(
        "0000000000000000000000000000000000000000000000000000000000000000"
    ));
    let salt = generate_salt(0);
    let keystore_address = generate_keystore_address(salt, eoa_addrs.clone(), vk_hash);
    let signatures: Vec<[u8; 65]> = private_keys
        .iter()
        .map(|k| ecdsa_sign(*k, keystore_address.clone(), eoa_addrs.clone(), &msg_hash))
        .collect::<Vec<_>>();

    let inputs = Inputs {
        keystore_address,
        msg_hash,
        m: 2,
        n: 3,
        signatures: signatures.clone(),
        eoa_addrs,
    };
    let serialized_inputs = bincode::serde::encode_to_vec(&inputs, bincode::config::standard())
        .expect("serialize to vec failed");
    executor
        .execute(
            elf,
            vec![serialized_inputs
                .into_iter()
                .map(F::from_canonical_u8)
                .collect()],
        )
        .unwrap();
    Ok(())
}
