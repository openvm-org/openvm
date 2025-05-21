use cargo_openvm::util::read_config_toml_or_default;
use clap::{Parser, ValueEnum};
use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_programs_dir, read_elf_file};
use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
use openvm_sdk::StdIn;
use openvm_stark_sdk::bench::run_with_metric_collection;
use openvm_transpiler::FromElf;

#[derive(Debug, Clone, ValueEnum)]
enum BuildProfile {
    Debug,
    Release,
}

static AVAILABLE_PROGRAMS: &[&str] = &[
    "fibonacci_recursive",
    "fibonacci_iterative",
    "quicksort",
    "bubblesort",
    "pairing",
    "keccak256",
    "keccak256_iter",
    "sha256",
    "sha256_iter",
    "revm_transfer",
    "revm_snailtracer",
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

            let config_path = program_dir.join("openvm.toml");
            let vm_config = read_config_toml_or_default(&config_path)?.app_vm_config;

            let exe = VmExe::from_elf(elf, vm_config.transpiler())?;

            let executor = VmExecutor::new(vm_config);
            executor.execute(exe, StdIn::default())?;
            tracing::info!("Completed program: {}", program);
        }
        tracing::info!("All programs executed successfully");
        Ok(())
    })
}
