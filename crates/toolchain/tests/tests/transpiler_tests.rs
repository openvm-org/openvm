use std::{
    fs::read,
    path::{Path, PathBuf},
};

use eyre::Result;
use num_bigint::BigUint;
use openvm_algebra_circuit::*;
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_bigint_circuit::*;
use openvm_circuit::{
    arch::{InitFileGenerator, SystemConfig, VmExecutor},
    derive::VmConfig,
    system::SystemExecutor,
    utils::air_test,
};
use openvm_ecc_circuit::{SECP256K1_MODULUS, SECP256K1_ORDER};
use openvm_instructions::exe::VmExe;
use openvm_platform::memory::MEM_SIZE;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImBuilder, Rv32ImConfig, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use serde::{Deserialize, Serialize};
use test_case::test_case;

use openvm_circuit::arch::VmExecState;

type F = BabyBear;

fn get_elf(elf_path: impl AsRef<Path>) -> Result<Elf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    Ok(elf)
}

// An "eyeball test" only: prints the decoded ELF for eyeball inspection
#[test]
fn test_decode_elf() -> Result<()> {
    let elf = get_elf("tests/data/rv32im-empty-program-elf")?;
    dbg!(elf);
    Ok(())
}

// To create ELF directly from .S file, `brew install riscv-gnu-toolchain` and run
// `riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostartfiles -e _start -Ttext 0 fib.S -o
// rv32im-fib-from-as` riscv64-unknown-elf-gcc supports rv32im if you set -march target
#[test_case("tests/data/rv32im-fib-from-as")]
#[test_case("tests/data/rv32im-intrin-from-as")]
fn test_generate_program(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let program = Transpiler::<F>::default()
        .with_extension(Rv32ITranspilerExtension)
        .with_extension(Rv32MTranspilerExtension)
        .with_extension(Rv32IoTranspilerExtension)
        .with_extension(ModularTranspilerExtension)
        .transpile(&elf.instructions)?;
    for instruction in program {
        println!("{:?}", instruction);
    }
    Ok(())
}

#[test_case("tests/data/rv32im-exp-from-as")]
#[test_case("tests/data/rv32im-fib-from-as")]
fn test_rv32im_aot_pure_runtime(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;

    let config = Rv32ImConfig::default();
    let executor = VmExecutor::new(config.clone())?;

    let interpreter = executor.instance(&exe)?;
    let interp_state = interpreter.execute(vec![], None)?;

    let mut aot_instance = executor.aot_instance(&exe)?;
    let aot_state = aot_instance.execute(vec![], None)?;

    // check that the VM state are equal
    assert_eq!(interp_state.instret(), aot_state.instret());
    assert_eq!(interp_state.pc(), aot_state.pc());

    let system_config: &SystemConfig = &config.as_ref();
    let addr_spaces = &system_config.memory_config.addr_spaces; 

    // check memory are equal
    for t in 1..4 {
        for r in 0..addr_spaces[t as usize].num_cells {
            let interp = unsafe {
                interp_state.memory.read::<u8, 1>(t, r as u32)
            };
            let aot_interp = unsafe {
                aot_state.memory.read::<u8, 1>(t, r as u32)
            };
            assert_eq!(interp, aot_interp);
        }
    }
    for r in 0..(addr_spaces[4].num_cells/4) {
        let interp = unsafe {
            interp_state.memory.read::<u32, 4>(4, 4 * r as u32)
        };
        let aot_interp = unsafe {
            aot_state.memory.read::<u32, 4>(4, 4 * r as u32)
        };
        assert_eq!(interp, aot_interp);
    }

    // check streams are equal
    assert_eq!(interp_state.streams.input_stream, aot_state.streams.input_stream);
    assert_eq!(interp_state.streams.hint_stream, aot_state.streams.hint_stream);
    assert_eq!(interp_state.streams.hint_space, aot_state.streams.hint_space);

    Ok(())
}

#[test_case("tests/data/rv32im-exp-from-as")]
#[test_case("tests/data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let config = Rv32ImConfig::default();
    let executor = VmExecutor::new(config)?;
    let interpreter = executor.instance(&exe)?;
    interpreter.execute(vec![], None)?;
    Ok(())
}

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct Rv32ModularFp2Int256Config {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub mul: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub modular: ModularExtension,
    #[extension]
    pub fp2: Fp2Extension,
    #[extension]
    pub int256: Int256,
}

impl Rv32ModularFp2Int256Config {
    pub fn new(modular_moduli: Vec<BigUint>, fp2_moduli: Vec<(String, BigUint)>) -> Self {
        Self {
            system: SystemConfig::default(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(modular_moduli),
            fp2: Fp2Extension::new(fp2_moduli),
            int256: Default::default(),
        }
    }
}

impl InitFileGenerator for Rv32ModularFp2Int256Config {
    fn generate_init_file_contents(&self) -> Option<String> {
        Some(format!(
            "{}\n{}\n",
            self.modular.generate_moduli_init(),
            self.fp2.generate_complex_init(&self.modular)
        ))
    }
}

#[test_case("tests/data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    let config = Rv32ModularFp2Int256Config::new(
        vec![SECP256K1_MODULUS.clone(), SECP256K1_ORDER.clone()],
        vec![("Secp256k1Coord".to_string(), SECP256K1_MODULUS.clone())],
    );
    let elf = get_elf(elf_path)?;
    let openvm_exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension)
            .with_extension(Fp2TranspilerExtension),
    )?;
    let executor = VmExecutor::new(config)?;
    let interpreter = executor.instance(&openvm_exe)?;
    interpreter.execute(vec![], None)?;
    Ok(())
}

#[test]
fn test_terminate_prove() -> Result<()> {
    let config = Rv32ImConfig::default();
    let elf = get_elf("tests/data/rv32im-terminate-from-as")?;
    let openvm_exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    )?;
    air_test(Rv32ImBuilder, config, openvm_exe);
    Ok(())
}
