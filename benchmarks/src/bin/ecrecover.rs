#![allow(unused_variables)]
#![allow(unused_imports)]

use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
    p3_keccak::Keccak256Hash,
};
use axvm_benchmarks::utils::{bench_from_exe, build_bench_program, BenchmarkCli};
use axvm_circuit::arch::{instructions::exe::AxVmExe, ExecutorName, VmConfig};
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_recursion::testing_utils::inner::build_verification_program;
use axvm_transpiler::axvm_platform::bincode;
use clap::Parser;
use eyre::Result;
use k256::ecdsa::{RecoveryId, Signature, SigningKey, VerifyingKey};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use tiny_keccak::{Hasher, Keccak};
use tracing::info_span;

fn make_input(signing_key: &SigningKey, msg: &[u8]) -> Vec<BabyBear> {
    let mut hasher = Keccak::v256();
    hasher.update(msg);
    let mut prehash = [0u8; 32];
    hasher.finalize(&mut prehash);
    let (signature, recid) = signing_key.sign_prehash_recoverable(&prehash).unwrap();
    // Input format: https://www.evm.codes/precompiled?fork=cancun#0x01
    let mut input = prehash.to_vec();
    let v = recid.to_byte() + 27u8;
    input.extend_from_slice(&[0; 31]);
    input.push(v);
    input.extend_from_slice(signature.to_bytes().as_ref());

    input.into_iter().map(BabyBear::from_canonical_u8).collect()
}

fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_log_blowup = cli_args.app_log_blowup.unwrap_or(2);
    let agg_log_blowup = cli_args.agg_log_blowup.unwrap_or(2);

    let elf = build_bench_program("ecrecover")?;
    let exe = AxVmExe::<BabyBear>::from(elf.clone());
    let vm_config = VmConfig::rv32im()
        .add_executor(ExecutorName::Keccak256Rv32)
        .add_modular_support(exe.custom_op_config.primes())
        .add_canonical_ec_curves(); // TODO: update sw_setup macros and read it from elf.

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vdata =
            info_span!("ECDSA Recover Program", group = "ecrecover_program").in_scope(|| {
                let mut rng = ChaCha8Rng::seed_from_u64(12345);
                let signing_key: SigningKey = SigningKey::random(&mut rng);
                let verifying_key = VerifyingKey::from(&signing_key);
                let mut hasher = Keccak::v256();
                let mut expected_address = [0u8; 32];
                hasher.update(
                    &verifying_key
                        .to_encoded_point(/* compress = */ false)
                        .as_bytes()[1..],
                );
                hasher.finalize(&mut expected_address);
                expected_address[..12].fill(0); // 20 bytes as the address.
                let mut input_stream = vec![expected_address
                    .into_iter()
                    .map(BabyBear::from_canonical_u8)
                    .collect::<Vec<_>>()];

                let msg = ["Elliptic", "Curve", "Digital", "Signature", "Algorithm"];
                input_stream.extend(
                    msg.iter()
                        .map(|s| make_input(&signing_key, s.as_bytes()))
                        .collect::<Vec<_>>(),
                );

                let engine = BabyBearPoseidon2Engine::new(
                    FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
                );
                bench_from_exe(engine, vm_config, elf, input_stream)
            })?;

        Ok(())
    })
}
