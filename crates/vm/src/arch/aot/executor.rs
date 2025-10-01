use crate::arch::SystemConfig;
use libloading::{Library, Symbol};
use openvm_instructions::exe::VmExe;
use p3_baby_bear::BabyBearParameters;
use p3_field::PrimeField32;
use std::{env, env::args, fs, path::PathBuf, process::Command};
pub struct AotInstance<F: PrimeField32> {
    exe: VmExe<F>,
}

impl<F: PrimeField32> AotInstance<F> {
    pub fn new(exe: &VmExe<F>) -> Self {
        // let asm_string = Self::compile(exe);

        let output = Command::new("rustc")
            .args([
                "--crate-type=staticlib",
                "--target=x86_64-apple-darwin",
                "./src/arch/aot/rust_function.rs",
                "-o",
                "librust_function_x86.a",
            ])
            .output();

        let output = Command::new("as")
            .args([
                "-arch",
                "x86_64",
                "./src/arch/aot/aot_asm.s",
                "-o",
                "./src/arch/aot/aot_asm.o",
            ])
            .output();

        let output = Command::new("gcc")
            .args([
                "-arch",
                "x86_64",
                "./src/arch/aot/aot_asm.o",
                "-L.",
                "-lrust_function_x86",
                "-o",
                "program",
            ])
            .output();

        Self { exe: exe.clone() }
    }

    // pub fn compile(exe: &VmExe<F>) -> String {
    //     let mut res = String::new();
    //     return res;
    // }

    pub fn execute(&self) {
        unsafe {
            let _ = Command::new("./program").status();
        }
    }
}
