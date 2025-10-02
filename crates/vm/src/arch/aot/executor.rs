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
                "-no-pie",
                "aot_asm.o",
                "-L.",
                "-lrust_function",
                "-o",
                "program",
            ])
            .output();

        Self { exe: exe.clone() }
    }

    pub fn generate_assembly_header() -> String {
        let mut res = String::new();
        res += &format!(".intel_syntax noprefix\n");
        res += &format!(".code64\n");
        res += &format!(".section .data\n");
        res += &format!(".align 8\n");
        for r in 0u64..32u64 {
            res += &format!(".comm reg_{r}, 8, 8\n");
        }
        res += &format!(".section .text\n");
        res += &format!(".extern print_message\n");
        res += &format!(".extern print_register\n");
        res += &format!(".global main\n");

        res += &format!("main:\n");
        res += &format!("\tsub rsp, 8\n");
        res += &format!("\txor rax, rax\n");
        // set all RISC-V register to 0
        for r in 0u64..32u64 {
            res += &format!("\tmov qword ptr [reg_{}], 0\n", r);
        }
        res += &format!("\n");
        return res;
    }

    pub fn generate_assembly_footer(exe: &VmExe<F>) -> String {
        let mut res = String::new();

        res += &format!("execute_end:\n");
        res += &format!("\txor rax, rax\n");
        res += &format!("\tadd rsp, 8\n");
        res += &format!("\tret\n");
        res += &format!("\n");

        res += &format!(".section .rodata\n");
        res += &format!(".align 64\n");

        // TODO: make sure this list is sorted in increasing order of pc
        for (pc, instruction, _debug_info) in exe.program.enumerate_by_pc() {
            res += &format!("map_pc_{:x}:\t.quad pc_{:x}", pc, pc);
        }
        res += &format!("\n");

        return res;
    }

    pub fn compile(exe: &VmExe<F>) -> String {
        let mut res = String::new();
        res += &Self::generate_assembly_header();
        
        for (pc, instruction, _debug_info) in exe.program.enumerate_by_pc() {
            let opcode = instruction.opcode;
            if opcode == BaseAluOpcode::ADD.global_opcode() {
                res += &Self::generate_assembly_add(pc, instruction);
            } else if opcode == BaseAluOpcode::XOR.global_opcode() {
            } else if opcode == BranchEqualOpcode::BEQ.global_opcode() {
            } else if opcode == BranchLessThanOpcode::BGEU.global_opcode() {
            } else if opcode == Rv32LoadStoreOpcode::LOADB.global_opcode() {
            } else if opcode == Rv32LoadStoreOpcode::STOREB.global_opcode() {
            } else {
            }

            res += &Self::generate_debug_registers(pc);
        }

        res += &Self::generate_assembly_footer(exe);

        return res;
    }

    pub fn generate_debug_registers(pc: u32) -> String {
        let mut res = String::new();
        res += &format!("pc_debug_{:x}:\n", pc);
        for r in 0u64..4u64 {
            res += &format!("\tmov rdi, qword ptr [reg_{}]\n", r);
            res += &format!("\tcall print_register\n");
        }
        res += "\n";
        return res;
    }

    pub fn generate_assembly_add(pc: u32, inst: Instruction<F>) -> String {
        let mut res = String::new();
        res += &format!("pc_{:x}:\n", pc);

        let opcode = inst.opcode;
        let a = inst.a;
        let b = inst.b;
        let c = inst.c;
        let e = inst.e;

        // specs: [a:4]_1 = [b:4]_1 + [c:4]_e

        if e == F::ZERO {
            // specs: [a:4]_1 = [b:4]_1 + [c:4]_0
            res += &format!("\tmov rax, qword ptr [reg_{}]\n", b);
            res += &format!("\tmov rbx, {}\n", c);
            res += &format!("\tadd rax, rbx\n");
            res += &format!("\tmov qword ptr [reg_{}], rax\n", a);
        } else {
            // specs: [a:4]_1 = [b:4]_1 + [c:4]_1
            res += &format!("\tmov rax, qword ptr [reg_{}]\n", b);
            res += &format!("\tmov rbx, qword ptr [reg_{}]\n", c);
            res += &format!("\tadd rax, rbx\n");
            res += &format!("\tmov qword ptr [reg_{}], rax\n", a);
        }

        res += &format!("\n");
        return res;
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
