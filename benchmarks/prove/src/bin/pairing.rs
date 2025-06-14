use clap::Parser;
use eyre::Result;
use openvm_algebra_circuit::{Fp2Extension, ModularExtension};
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::arch::SystemConfig;
use openvm_ecc_circuit::WeierstrassExtension;
use openvm_pairing_circuit::{PairingCurve, PairingExtension};
use openvm_pairing_guest::bn254::{BN254_COMPLEX_STRUCT_NAME, BN254_MODULUS, BN254_ORDER};
use openvm_sdk::{config::SdkVmConfig, Sdk, StdIn};
use openvm_stark_sdk::bench::run_with_metric_collection;

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    let vm_config = SdkVmConfig::builder()
        .system(SystemConfig::default().with_continuations().into())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .keccak(Default::default())
        .modular(ModularExtension::new(vec![
            BN254_MODULUS.clone(),
            BN254_ORDER.clone(),
        ]))
        .fp2(Fp2Extension::new(vec![(
            BN254_COMPLEX_STRUCT_NAME.to_string(),
            BN254_MODULUS.clone(),
        )]))
        .ecc(WeierstrassExtension::new(vec![
            PairingCurve::Bn254.curve_config()
        ]))
        .pairing(PairingExtension::new(vec![PairingCurve::Bn254]))
        .build();
    let elf = args.build_bench_program("pairing", &vm_config, None)?;
    let sdk = Sdk::new();
    let exe = sdk.transpile(elf, vm_config.transpiler()).unwrap();

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        args.bench_from_exe("pairing", vm_config, exe, StdIn::default())
    })
}
