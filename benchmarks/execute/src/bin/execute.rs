use eyre::Result;
use k256::ecdsa::{SigningKey, VerifyingKey};
use openvm_algebra_circuit::{Fp2Extension, ModularExtension};
use openvm_benchmarks_utils::{build_and_load_elf, get_programs_dir};
use openvm_circuit::arch::{instructions::exe::VmExe, SystemConfig, VmExecutor};
use openvm_ecc_circuit::{WeierstrassExtension, SECP256K1_CONFIG};
use openvm_pairing_circuit::{PairingCurve, PairingExtension};
use openvm_pairing_guest::bn254::{BN254_MODULUS, BN254_ORDER};
use openvm_sdk::{config::SdkVmConfig, StdIn};
use openvm_stark_sdk::bench::run_with_metric_collection;
use openvm_transpiler::FromElf;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use tiny_keccak::{Hasher, Keccak};

struct ProgramConfig {
    name: &'static str,
    vm_config: fn() -> SdkVmConfig,
    setup_stdin: fn() -> StdIn,
}

static PROGRAMS_TO_RUN: [ProgramConfig; 7] = [
    ProgramConfig {
        name: "fibonacci",
        vm_config: || {
            SdkVmConfig::builder()
                .system(SystemConfig::default().with_continuations().into())
                .rv32i(Default::default())
                .rv32m(Default::default())
                .io(Default::default())
                .build()
        },
        setup_stdin: || {
            let n = 100_000u64;
            let mut stdin = StdIn::default();
            stdin.write(&n);
            stdin
        },
    },
    ProgramConfig {
        name: "revm_transfer",
        vm_config: || {
            SdkVmConfig::builder()
                .system(SystemConfig::default().with_continuations().into())
                .rv32i(Default::default())
                .rv32m(Default::default())
                .io(Default::default())
                .keccak(Default::default())
                .build()
        },
        setup_stdin: || StdIn::default(),
    },
    ProgramConfig {
        name: "base64_json",
        vm_config: || {
            SdkVmConfig::builder()
                .system(SystemConfig::default().with_continuations().into())
                .rv32i(Default::default())
                .rv32m(Default::default())
                .io(Default::default())
                .keccak(Default::default())
                .build()
        },
        setup_stdin: || {
            let data = include_str!("../../../guest/src/base64_json/json_payload_encoded.txt");
            let fe_bytes = data.to_owned().into_bytes();
            StdIn::from_bytes(&fe_bytes)
        },
    },
    ProgramConfig {
        name: "bincode",
        vm_config: || {
            SdkVmConfig::builder()
                .system(SystemConfig::default().with_continuations().into())
                .rv32i(Default::default())
                .rv32m(Default::default())
                .io(Default::default())
                .build()
        },
        setup_stdin: || {
            let file_data = include_bytes!("../../../guest/src/bincode/minecraft_savedata.bin");
            StdIn::from_bytes(file_data)
        },
    },
    // ProgramConfig {
    //     name: "ecrecover",
    //     vm_config: || {
    //         SdkVmConfig::builder()
    //             .system(SystemConfig::default().with_continuations().into())
    //             .rv32i(Default::default())
    //             .rv32m(Default::default())
    //             .io(Default::default())
    //             .modular(ModularExtension::new(vec![
    //                 SECP256K1_CONFIG.modulus.clone(),
    //                 SECP256K1_CONFIG.scalar.clone(),
    //             ]))
    //             .keccak(Default::default())
    //             .ecc(WeierstrassExtension::new(vec![SECP256K1_CONFIG.clone()]))
    //             .build()
    //     },
    //     setup_stdin: || {
    //         let mut rng = ChaCha8Rng::seed_from_u64(12345);
    //         let signing_key: SigningKey = SigningKey::random(&mut rng);
    //         let verifying_key = VerifyingKey::from(&signing_key);

    //         // Create expected address from public key
    //         let mut hasher = Keccak::v256();
    //         let mut expected_address = [0u8; 32];
    //         hasher.update(
    //             &verifying_key
    //                 .to_encoded_point(/* compress = */ false)
    //                 .as_bytes()[1..],
    //         );
    //         hasher.finalize(&mut expected_address);
    //         expected_address[..12].fill(0); // 20 bytes as the address

    //         // Create message and signature
    //         let msg = "Elliptic Curve Digital Signature Algorithm";

    //         let mut hasher = Keccak::v256();
    //         hasher.update(msg.as_bytes());
    //         let mut prehash = [0u8; 32];
    //         hasher.finalize(&mut prehash);

    //         let (signature, recid) = signing_key.sign_prehash_recoverable(&prehash).unwrap();

    //         // Format input according to ecrecover requirements
    //         let mut input = expected_address.to_vec();
    //         input.extend_from_slice(&prehash);
    //         let v = recid.to_byte() + 27u8;
    //         input.extend_from_slice(&[0; 31]);
    //         input.push(v);
    //         input.extend_from_slice(signature.to_bytes().as_ref());

    //         // Create StdIn from prepared input
    //         StdIn::from_bytes(&input)
    //     },
    // },
    ProgramConfig {
        name: "pairing",
        vm_config: || {
            SdkVmConfig::builder()
                .system(SystemConfig::default().with_continuations().into())
                .rv32i(Default::default())
                .rv32m(Default::default())
                .io(Default::default())
                .keccak(Default::default())
                .modular(ModularExtension::new(vec![
                    BN254_MODULUS.clone(),
                    BN254_ORDER.clone(),
                ]))
                .fp2(Fp2Extension::new(vec![BN254_MODULUS.clone()]))
                .ecc(WeierstrassExtension::new(vec![
                    PairingCurve::Bn254.curve_config()
                ]))
                .pairing(PairingExtension::new(vec![PairingCurve::Bn254]))
                .build()
        },
        setup_stdin: || StdIn::default(),
    },
    ProgramConfig {
        name: "regex",
        vm_config: || {
            SdkVmConfig::builder()
                .system(SystemConfig::default().with_continuations().into())
                .rv32i(Default::default())
                .rv32m(Default::default())
                .io(Default::default())
                .keccak(Default::default())
                .build()
        },
        setup_stdin: || {
            let data = include_str!("../../../guest/src/regex/regex_email.txt");
            let fe_bytes = data.to_owned().into_bytes();
            StdIn::from_bytes(&fe_bytes)
        },
    },
    ProgramConfig {
        name: "rkyv",
        vm_config: || {
            SdkVmConfig::builder()
                .system(SystemConfig::default().with_continuations().into())
                .rv32i(Default::default())
                .rv32m(Default::default())
                .io(Default::default())
                .build()
        },
        setup_stdin: || {
            let file_data = include_bytes!("../../../guest/src/rkyv/minecraft_savedata.bin");
            StdIn::from_bytes(file_data)
        },
    },
];

fn main() -> Result<()> {
    tracing::info!("Starting benchmarks with metric collection");

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        for program in &PROGRAMS_TO_RUN {
            tracing::info!("Running program: {}", program.name);

            let program_dir = get_programs_dir().join(program.name);
            let elf = build_and_load_elf(program_dir, "release", false)?;

            let vm_config = (program.vm_config)();
            let exe = VmExe::from_elf(elf, vm_config.transpiler())?;

            let executor = VmExecutor::new(vm_config);
            executor.execute(exe, (program.setup_stdin)())?;
            tracing::info!("Completed program: {}", program.name);
        }
        tracing::info!("All programs executed successfully");
        Ok(())
    })
}
