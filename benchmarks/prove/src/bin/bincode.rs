use clap::Parser;
use openvm_benchmarks_prove::BenchmarkCli;
use openvm_sdk::StdIn;
use openvm_sdk_config::SdkVmConfig;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};

fn main() -> eyre::Result<()> {
    let args = BenchmarkCli::parse();
    let vm_config = SdkVmConfig::from_toml(include_str!("../../../guest/bincode/openvm.toml"))?;

    let elf = Elf::decode(
        include_bytes!("../../../guest/bincode/elf/openvm-bincode-program.elf"),
        MEM_SIZE as u32,
    )?;

    let file_data = include_bytes!("../../../guest/bincode/minecraft_savedata.bin");
    let stdin = StdIn::from_bytes(file_data);

    args.run(vm_config, elf, stdin)
}
