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

use crate::utils::{
    build_example_program, calculate_data_hash, calculate_keystore_address, calculate_salt,
    ecdsa_sign,
};

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
        /// data hash
        data_hash: [u8; 32],
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

    let m = 2;
    let n = 3;
    let msg = b"message";
    let msg_hash = keccak256(msg);

    // Default Anvil private keys
    #[allow(clippy::useless_vec)]
    let private_keys = vec![
        hex!("ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"),
        // hex!("59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"),
        hex!("5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"),
    ];
    let eoa_addrs = vec![
        hex!("f39fd6e51aad88f6f4ce6ab8827279cfffb92266"),
        hex!("70997970c51812dc3a010c7d01b50e0d17dc79c8"),
        hex!("3c44cdddb6a900fa2b585dd299e03d12fa4293bc"),
    ];
    let vk_hash = keccak256(&hex!(
        "0000000000000000000000000000000000000000000000000000000000000000"
    ));

    let salt = calculate_salt(0);
    let data_hash = calculate_data_hash(m, n, eoa_addrs.clone());
    let keystore_address = calculate_keystore_address(salt, data_hash, vk_hash);
    let full_hash = keccak256(
        [
            keystore_address.to_vec(),
            data_hash.to_vec(),
            msg_hash.to_vec(),
        ]
        .concat()
        .as_slice(),
    );
    let signatures: Vec<[u8; 65]> = private_keys
        .iter()
        .map(|k| ecdsa_sign(*k, full_hash))
        .collect::<Vec<_>>();

    let inputs = Inputs {
        keystore_address,
        data_hash,
        msg_hash,
        m,
        n,
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
