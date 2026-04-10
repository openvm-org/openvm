use clap::Parser;
use openvm_benchmarks_prove::BenchmarkCli;
use openvm_sdk::StdIn;
use openvm_sdk_config::SdkVmConfig;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};

fn main() -> eyre::Result<()> {
    let args = BenchmarkCli::parse();
    let vm_config = SdkVmConfig::from_toml(include_str!(
        "../../../guest/sha2_bench/openvm.toml"
    ))?;

    let elf = Elf::decode(
        include_bytes!("../../../guest/sha2_bench/elf/openvm-sha2-bench-program.elf"),
        MEM_SIZE as u32,
    )?;

    // 10 MB
    let num_bytes: u32 = 10 * 1024 * 1024;
    let mut stdin = StdIn::default();
    stdin.write(&num_bytes);

    args.run(vm_config, elf, stdin)
}
