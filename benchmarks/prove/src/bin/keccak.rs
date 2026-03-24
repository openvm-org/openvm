use clap::Parser;
use openvm_benchmarks_prove::BenchmarkCli;
use openvm_sdk::StdIn;
use openvm_sdk_config::SdkVmConfig;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};

fn main() -> eyre::Result<()> {
    let args = BenchmarkCli::parse();
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .keccak(Default::default())
        .build()
        .optimize();

    let elf = Elf::decode(
        include_bytes!("../../../guest/keccak256_iter/elf/openvm-keccak256-iter-program.elf"),
        MEM_SIZE as u32,
    )?;

    let num_keccak_iters: u64 = 1 << 12;
    let mut stdin = StdIn::default();
    stdin.write(&num_keccak_iters);

    args.run(vm_config, elf, stdin)
}
