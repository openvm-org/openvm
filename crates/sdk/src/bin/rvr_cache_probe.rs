//! Cross-process cache test helper.
//!
//! Usage: rvr_cache_probe <mode> <cache_dir>
//!   mode: pure | metered | metered_cost
//!
//! Compiles fibonacci with the given cache_dir (expects a cache hit), executes
//! it, and prints the public values as hex to stdout. The parent test compares
//! this output against its own baseline and checks the artifact mtime to verify
//! the cache was reused across processes.

use std::path::PathBuf;

use openvm::platform::memory::MEM_SIZE;
use openvm_sdk::{
    config::{AggregationSystemParams, DEFAULT_APP_L_SKIP},
    Sdk, StdIn,
};
use openvm_stark_sdk::config::app_params_with_100_bits_security;
use openvm_transpiler::elf::Elf;

fn main() -> eyre::Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().expect("usage: rvr_cache_probe <mode> <cache_dir>");
    let cache_dir = PathBuf::from(args.next().expect("usage: rvr_cache_probe <mode> <cache_dir>"));

    let elf_bytes = include_bytes!("../../programs/examples/fibonacci.elf");
    let elf = Elf::decode(elf_bytes, MEM_SIZE as u32)?;

    // Must match make_fib_sdk() exactly so the fingerprint is identical.
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let sdk = Sdk::riscv32(app_params, AggregationSystemParams::default());
    let exe = sdk.convert_to_exe(elf)?;

    let mut stdin = StdIn::default();
    stdin.write(&100u64);

    let pv = match mode.as_str() {
        "pure" => {
            let compiled = sdk.compile_pure_cached(exe, Some(&cache_dir))?;
            sdk.execute_compiled(&compiled, stdin)?
        }
        "metered" => {
            let compiled = sdk.compile_metered_cached(exe, Some(&cache_dir))?;
            let (pv, _) = sdk.execute_compiled_metered(&compiled, stdin)?;
            pv
        }
        "metered_cost" => {
            let compiled = sdk.compile_metered_cost_cached(exe, Some(&cache_dir))?;
            let (pv, _) = sdk.execute_compiled_metered_cost(&compiled, stdin)?;
            pv
        }
        _ => panic!("unknown mode: {mode}"),
    };

    println!("{}", hex::encode(&pv));
    Ok(())
}
