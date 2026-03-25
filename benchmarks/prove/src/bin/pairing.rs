use clap::Parser;
use openvm_benchmarks_prove::BenchmarkCli;
use openvm_sdk::StdIn;
use openvm_sdk_config::SdkVmConfig;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};

fn main() -> eyre::Result<()> {
    let args = BenchmarkCli::parse();
    let vm_config = SdkVmConfig::from_toml(include_str!("../../../guest/pairing/openvm.toml"))?;

    let elf = Elf::decode(
        include_bytes!("../../../guest/pairing/elf/openvm-pairing-program.elf"),
        MEM_SIZE as u32,
    )?;

    args.run(vm_config, elf, StdIn::default())
}
