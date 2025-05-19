use clap::{Parser, ValueEnum};
use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_programs_dir, read_elf_file};
use openvm_bigint_circuit::Int256Rv32Config;
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::arch::{instructions::exe::VmExe, VirtualMachine, VmExecutor};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection, config::baby_bear_poseidon2::default_engine,
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

#[derive(Debug, Clone, ValueEnum)]
enum BuildProfile {
    Debug,
    Release,
}

// const DEFAULT_APP_CONFIG_PATH: &str = "./openvm.toml";

static AVAILABLE_PROGRAMS: &[&str] = &[
    "fibonacci_recursive",
    "fibonacci_iterative",
    "quicksort",
    "bubblesort",
    "factorial_iterative_u256",
    "revm_snailtracer",
    // "pairing",
    // "keccak256",
    // "keccak256_iter",
    // "sha256",
    // "sha256_iter",
    // "revm_transfer",
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
            let vm_config = Int256Rv32Config::default();

            let transpiler = Transpiler::<BabyBear>::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
                .with_extension(Rv32IoTranspilerExtension)
                .with_extension(Int256TranspilerExtension);

            let exe = VmExe::from_elf(elf, transpiler)?;

            let (widths, interactions): (Vec<usize>, Vec<usize>) = {
                let vm = VirtualMachine::new(default_engine(), vm_config.clone());
                let pk = vm.keygen();
                let vk = pk.get_vk();
                vk.inner
                    .per_air
                    .iter()
                    .map(|vk| {
                        // TODO(ayush): figure out which width to use
                        // let total_width = vk.params.width.preprocessed.unwrap_or(0)
                        //     + vk.params.width.cached_mains.iter().sum::<usize>()
                        //     + vk.params.width.common_main
                        //     + vk.params.width.after_challenge.iter().sum::<usize>();
                        let total_width = vk.params.width.main_widths().iter().sum::<usize>();
                        (total_width, vk.symbolic_constraints.interactions.len())
                    })
                    .unzip()
            };

            let executor = VmExecutor::new(vm_config);
            executor
                .execute_e2(exe.clone(), vec![], widths, interactions)
                .expect("Failed to execute program");

            tracing::info!("Completed program: {}", program);
        }
        tracing::info!("All programs executed successfully");
        Ok(())
    })
}
