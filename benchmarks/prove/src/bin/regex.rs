use clap::Parser;
use openvm_benchmarks_prove::BenchmarkCli;
use openvm_sdk::StdIn;
use openvm_sdk_config::SdkVmConfig;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};

fn main() -> eyre::Result<()> {
    let args = BenchmarkCli::parse();
    let vm_config = SdkVmConfig::from_toml(include_str!("../../../guest/regex/openvm.toml"))?;

    let elf = Elf::decode(
        include_bytes!("../../../guest/regex/elf/openvm-regex-program.elf"),
        MEM_SIZE as u32,
    )?;

    let data = include_str!("../../../guest/regex/regex_email.txt");
    let stdin = StdIn::from_bytes(&data.to_owned().into_bytes());

    args.run(vm_config, elf, stdin)
}
