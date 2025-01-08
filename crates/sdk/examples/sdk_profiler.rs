use std::path::PathBuf;

use eyre::Result;
use openvm_build::GuestOptions;
use openvm_sdk::{config::SdkVmConfig, profiler::Profiler, Sdk, StdIn};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SomeStruct {
    pub a: u64,
    pub b: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build the VmConfig with the extensions needed.
    let sdk = Sdk(Profiler::new());
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .build();

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    path.push("guest");
    let target_path = path.to_str().unwrap();

    // 2a. Build the ELF with guest options and a target filter.
    let guest_opts = GuestOptions::default();
    let elf = sdk.build(guest_opts, target_path, &Default::default())?;
    println!("build complete");

    // 3. Transpile the ELF into a VmExe
    let exe = sdk.transpile(elf, vm_config.transpiler())?;
    println!("transpilation done");

    // 4. Format your input into StdIn
    let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
    let mut stdin = StdIn::default();
    stdin.write(&my_input);

    // 5. Run the program
    let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;
    println!("public values output: {:?}", output);

    Ok(())
}
