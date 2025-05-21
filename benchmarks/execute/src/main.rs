use clap::{Parser, ValueEnum};
use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_programs_dir, read_elf_file};
use openvm_bigint_circuit::{Int256, Int256Executor, Int256Periphery, Int256Rv32Config};
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::{
    arch::{
        execution_mode::metered::Segment, instructions::exe::VmExe, SystemConfig, VirtualMachine,
        VmExecutor, VmExecutorResult,
    },
    derive::VmConfig,
};
use openvm_keccak256_circuit::{
    Keccak256, Keccak256Executor, Keccak256Periphery, Keccak256Rv32Config,
};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha256Periphery};
use openvm_sha256_transpiler::Sha256TranspilerExtension;
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::{
        baby_bear_blake3::BabyBearBlake3Config,
        baby_bear_poseidon2::{default_engine, BabyBearPoseidon2Config},
    },
    openvm_stark_backend::{
        self,
        p3_field::{FieldExtensionAlgebra, PrimeField32},
    },
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use serde::{Deserialize, Serialize};

// const DEFAULT_APP_CONFIG_PATH: &str = "./openvm.toml";

static AVAILABLE_PROGRAMS: &[&str] = &[
    "fibonacci_recursive",
    "fibonacci_iterative",
    "quicksort",
    "bubblesort",
    "factorial_iterative_u256",
    "revm_snailtracer",
    "keccak256",
    "keccak256_iter",
    "sha256",
    "sha256_iter",
    "revm_transfer",
    // "pairing",
];

#[derive(Parser)]
#[command(author, version, about = "OpenVM Benchmark CLI", long_about = None)]
struct Cli {
    /// Programs to benchmark (if not specified, all programs will be run)
    #[arg(short, long)]
    programs: Vec<String>,

    /// Programs to skip from benchmarking
    #[arg(short, long)]
    skip: Vec<String>,

    /// Output path for benchmark results
    #[arg(short, long, default_value = "OUTPUT_PATH")]
    output: String,

    /// List available benchmark programs and exit
    #[arg(short, long)]
    list: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct ExecuteConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub bigint: Int256,
    #[extension]
    pub keccak: Keccak256,
    #[extension]
    pub sha256: Sha256,
}

impl Default for ExecuteConfig {
    // TODO(ayush): this should be auto-derived as vmconfig should have a with_continuations method
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I::default(),
            rv32m: Rv32M::default(),
            io: Rv32Io::default(),
            bigint: Int256::default(),
            keccak: Keccak256::default(),
            sha256: Sha256::default(),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list {
        println!("Available benchmark programs:");
        for program in AVAILABLE_PROGRAMS {
            println!("  {}", program);
        }
        return Ok(());
    }

    // Set up logging based on verbosity
    if cli.verbose {
        tracing_subscriber::fmt::init();
    }

    let mut programs_to_run = if cli.programs.is_empty() {
        AVAILABLE_PROGRAMS.to_vec()
    } else {
        // Validate provided programs
        for program in &cli.programs {
            if !AVAILABLE_PROGRAMS.contains(&program.as_str()) {
                eprintln!("Unknown program: {}", program);
                eprintln!("Use --list to see available programs");
                std::process::exit(1);
            }
        }
        cli.programs.iter().map(|s| s.as_str()).collect()
    };

    // Remove programs that should be skipped
    if !cli.skip.is_empty() {
        // Validate skipped programs
        for program in &cli.skip {
            if !AVAILABLE_PROGRAMS.contains(&program.as_str()) {
                eprintln!("Unknown program to skip: {}", program);
                eprintln!("Use --list to see available programs");
                std::process::exit(1);
            }
        }

        let skip_set: Vec<&str> = cli.skip.iter().map(|s| s.as_str()).collect();
        programs_to_run.retain(|&program| !skip_set.contains(&program));
    }

    tracing::info!("Starting benchmarks with metric collection");

    run_with_metric_collection(&cli.output, || -> Result<()> {
        for program in &programs_to_run {
            tracing::info!("Running program: {}", program);

            let program_dir = get_programs_dir().join(program);
            let elf_path = get_elf_path(&program_dir);
            let elf = read_elf_file(&elf_path)?;

            // let config_path = program_dir.join(DEFAULT_APP_CONFIG_PATH);
            // let vm_config = read_config_toml_or_default(&config_path)?.app_vm_config;
            // let transpiler = vm_config.transpiler;
            let vm_config = ExecuteConfig::default();

            let transpiler = Transpiler::<BabyBear>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Int256TranspilerExtension)
                .with_extension(Keccak256TranspilerExtension)
                .with_extension(Sha256TranspilerExtension);

            let exe = VmExe::from_elf(elf, transpiler)?;

            let vm = VirtualMachine::new(default_engine(), vm_config.clone());
            let pk = vm.keygen();
            let (widths, interactions): (Vec<usize>, Vec<usize>) = {
                let vk = pk.get_vk();
                vk.inner
                    .per_air
                    .iter()
                    .map(|vk| {
                        let total_width = vk.params.width.preprocessed.unwrap_or(0)
                            + vk.params.width.cached_mains.iter().sum::<usize>()
                            + vk.params.width.common_main
                            // TODO(ayush): no magic value 4. should come from stark config
                            + vk.params.width.after_challenge.iter().sum::<usize>() * 4;
                        (total_width, vk.symbolic_constraints.interactions.len())
                    })
                    .unzip()
            };

            let executor = VmExecutor::new(vm_config);

            // E2 to find segment points
            let segments = executor.execute_metered(exe.clone(), vec![], widths, interactions)?;
            for Segment {
                clk_start,
                num_cycles,
                ..
            } in segments
            {
                // E1 till clk_start
                let state = executor.execute_e1(exe.clone(), vec![], Some(clk_start))?;
                assert!(state.clk == clk_start);
                // E3/tracegen from clk_start for num_cycles beginning with state
                let mut result = executor.execute_and_generate_segment::<BabyBearPoseidon2Config>(
                    exe.clone(),
                    state,
                    num_cycles,
                )?;
                // let proof_input = result.per_segment.pop().unwrap();
                // let proof = tracing::info_span!("prove_single")
                //     .in_scope(|| vm.prove_single(&pk, proof_input));

                // let proof_bytes = bitcode::serialize(&proof)?;
                // tracing::info!("Proof size: {} bytes", proof_bytes.len());
            }

            tracing::info!("Completed program: {}", program);
        }
        tracing::info!("All programs executed successfully");
        Ok(())
    })
}
