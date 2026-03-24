use clap::Parser;
use k256::ecdsa::{SigningKey, VerifyingKey};
use openvm_benchmarks_prove::BenchmarkCli;
use openvm_sdk::F;
use openvm_sdk_config::SdkVmConfig;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use tiny_keccak::{Hasher, Keccak};

fn make_input(signing_key: &SigningKey, msg: &[u8]) -> Vec<F> {
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

    input.into_iter().map(F::from_u8).collect()
}

fn main() -> eyre::Result<()> {
    let args = BenchmarkCli::parse();
    let vm_config = SdkVmConfig::from_toml(include_str!("../../../guest/ecrecover/openvm.toml"))?;

    let elf = Elf::decode(
        include_bytes!("../../../guest/ecrecover/elf/openvm-ecdsa-recover-key-program.elf"),
        MEM_SIZE as u32,
    )?;

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
        .map(F::from_u8)
        .collect::<Vec<_>>()];
    let msg = ["Elliptic", "Curve", "Digital", "Signature", "Algorithm"];
    input_stream.extend(
        msg.iter()
            .map(|s| make_input(&signing_key, s.as_bytes()))
            .collect::<Vec<_>>(),
    );
    let stdin = input_stream.into();

    args.run(vm_config, elf, stdin)
}
