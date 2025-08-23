use clap::Parser;
use eyre::Result;
use k256::ecdsa::{SigningKey, VerifyingKey};
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_sdk::config::{SdkVmBuilder, SdkVmConfig};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{bench::run_with_metric_collection, p3_baby_bear::BabyBear};
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use tiny_keccak::{Hasher, Keccak};

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
    let args = BenchmarkCli::parse();

    let config =
        SdkVmConfig::from_toml(include_str!("../../../guest/ecrecover/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("ecrecover", &config, None)?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
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
        args.bench_from_exe::<SdkVmBuilder, _>(
            "ecrecover_program",
            config,
            elf,
            input_stream.into(),
        )
    })
}
