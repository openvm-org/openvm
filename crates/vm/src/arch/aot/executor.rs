use crate::arch::instructions::VmOpcode;
use crate::{
    arch::{SystemConfig, VmState},
    system::memory::online::GuestMemory,
};
use libloading::{Library, Symbol};
use openvm_instructions::instruction;
use openvm_instructions::LocalOpcode;
use openvm_instructions::{exe::VmExe, instruction::Instruction};
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, Rv32LoadStoreOpcode,
};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::config::fri_params::standard_fri_params_with_100_bits_conjectured_security;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use p3_baby_bear::BabyBearParameters;
use p3_field::PrimeField32;
use std::fs::File;
use std::io::Write;
use std::{env, env::args, fs, path::PathBuf, process::Command};
use tracing::subscriber::SetGlobalDefaultError;

const DEFAULT_PC_JUMP: u32 = 4;
pub struct AotInstance<F: PrimeField32> {
    exe: VmExe<F>,
}

impl<F: PrimeField32> AotInstance<F> {
    pub fn new(exe: &VmExe<F>) -> Self {
        let asm_string = Self::compile(exe);

        let _ = File::create("aot_asm.s").expect("Unable to create file");
        std::fs::write("aot_asm.s", &asm_string).expect("Unable to write file");

        let output = Command::new("rustc")
            .args([
                "--crate-type=staticlib",
                "--target=x86_64-unknown-linux-gnu",
                "rust_function.rs",
                "-o",
                "librust_function.a",
            ])
            .output();

        let output = Command::new("as")
            .args(["aot_asm.s", "-o", "aot_asm.o"])
            .output();

        let output = Command::new("gcc")
            .args([
                "-L.",
                "-lrust_function_x86",
                "-o",
                "program",
            ])
            .output();

        Self { exe: exe.clone() }
    }

    pub fn compile(exe: &VmExe<F>) -> String {
        let mut res = String::new();
        res += r#"
.intel_syntax noprefix
.section    __TEXT,__text,regular,pure_instructions
.globl  _main

"#;
        for r in 0u64..32u64 {
            res += &format!(".comm reg_{r}, 8, 3\n");
        }

        res += r#"
.section    __TEXT,__text,regular,pure_instructions
_main:  
# can do some initializations here

"#;

        for (pc, instruction, _debug_info) in exe.program.enumerate_by_pc() {
            let opcode = instruction.opcode;

            if opcode == BaseAluOpcode::ADD.global_opcode() {
                println!("instruction {pc} is ADD");
                res += &Self::generate_assembly_add(pc, instruction);
            } else if opcode == BaseAluOpcode::XOR.global_opcode() {
                println!("instruction {pc} is XOR");
            } else if opcode == BranchEqualOpcode::BEQ.global_opcode() {
                println!("instruction {pc} is BEQ");
            } else if opcode == BranchLessThanOpcode::BGEU.global_opcode() {
                println!("instruction {pc} is BGEU");
            } else if opcode == Rv32LoadStoreOpcode::LOADB.global_opcode() {
                println!("instruction {pc} is LOADB");
            } else if opcode == Rv32LoadStoreOpcode::STOREB.global_opcode() {
                println!("instruction {pc} is STOREB");
            } else {
                println!("instruction {pc}'s opcode is not implemented yet");
            }
        }

        // // TODO: remove these instructions
        // let instruction = Instruction {
        //     opcode: BaseAluOpcode::ADD.global_opcode(),
        //     a: F::from_canonical_u32(0),
        //     b: F::from_canonical_u32(0),
        //     c: F::from_canonical_u32(18),
        //     d: F::from_canonical_u32(0),
        //     e: F::from_canonical_u32(0),
        //     f: F::from_canonical_u32(0),
        //     g: F::from_canonical_u32(0),
        // };

        // res += &format!("pc_{:x}:\n", 0);
        // res += &Self::generate_assembly_add(instruction);
        // res += &format!("\tmov rdi, qword ptr[rip + reg_0]\n");
        // res += &format!("\tcall _print_register\n");
        // res += "\n";

        // let instruction = Instruction {
        //     opcode: BranchEqualOpcode::BEQ.global_opcode(),
        //     a: F::from_canonical_u32(0),
        //     b: F::from_canonical_u32(0),
        //     c: F::from_canonical_u32(0),
        //     d: F::from_canonical_u32(0),
        //     e: F::from_canonical_u32(0),
        //     f: F::from_canonical_u32(0),
        //     g: F::from_canonical_u32(0),
        // };

        // res += &Self::generate_assembly_beq(instruction, 4);

        // res += "execute_end:\n";
        // res += "\tret\n\n";

        return res;
    }

    pub fn generate_assembly_add(pc: u32, inst: Instruction<F>) -> String {
        let opcode = inst.opcode;

        let asm = {
            // [a:4]_1 = [b:4]_1 + [c:4]_e
            let a = inst.a;
            let b = inst.b;
            let c = inst.c;
            let e = inst.e;

            let mut res = String::new();
            if e == F::ZERO {
                res += &format!("\txor rbx, rbx\n");
                res += &format!("\tadd rbx, qword ptr[rip + reg_{b}]\n");
                res += &format!("\tadd rbx, {c}\n");
                res += &format!("\tmov qword ptr[rip + reg_{a}], rbx\n");
            } else {
                res += &format!("\txor rbx, rbx\n");
                res += &format!("\tadd rbx, qword ptr[rip + reg_{b}]\n");
                res += &format!("\tadd rbx, qword ptr[rip + reg_{c}]\n");
                res += &format!("\tmov qword ptr[rip + reg_{a}], rbx\n");
            }
            res
        };

        return asm;
    }

    pub fn generate_assembly_beq(inst: Instruction<F>, pc: u32) -> String {
        let asm = {
            let a = inst.a;
            let b = inst.b;
            let c = inst.c;

            let mut res = String::new();
            res += &format!("pc_{:x}\n", pc);
            res += &format!("\tmov rax, qword ptr[rip + reg_{a}\n");
            res += &format!("\tmov rbx, qword ptr[rip + reg_{b}\n");
            res += &format!("\tcmp eax, ebx\n");
            res += &format!("\tje pc_{:x}_beq_true\n", pc);
            res += &format!("\tjmp pc_{:x}_beq_done\n", pc);
            res += "\n";
            res += &format!("pc_{:x}_beq_true:\n", pc);
            res += &format!("\tadd r8, {}\n", c);
            res += "\n";

            res += &format!("pc_{:x}_beq_done:\n", pc);
            res += &format!("\tadd r8, {}\n", DEFAULT_PC_JUMP);
            res += "\n";

            // increment pc
            res += &format!("\tadd r8, {}\n", 4);
            res += "\n";
            res
        };
        return asm;
    }

    // TODO: push & pop other caller saved regs too
    pub fn push_caller_saved_regs() -> String {
        let mut res = String::new();
        res += "\tpush r8";
        return res;
    }

    pub fn pop_caller_saved_regs() -> String {
        let mut res = String::new();
        res += "\tpop r8";
        return res;
    }

    pub fn execute(&self) {
        unsafe {
            let _ = Command::new("./program").status();
        }
    }
}
