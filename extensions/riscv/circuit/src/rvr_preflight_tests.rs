use std::{collections::BTreeMap, sync::Arc};

#[cfg(feature = "cuda")]
use openvm_circuit::arch::VmBuilder;
#[cfg(feature = "cuda")]
use openvm_circuit::arch::VmState;
#[cfg(feature = "cuda")]
use openvm_circuit::utils::test_gpu_engine;
use openvm_circuit::{
    arch::{
        rvr::{
            preflight_compile_invocations_for_test, reset_preflight_compile_invocations_for_test,
            LogNativeAssemblerRegistry, RvrPreflightOutput, RvrPreflightRoute,
            VmRvrLogNativeExtension, PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_WRITE,
        },
        verify_segments, ContinuationVmProver, ExecutionError, MatrixRecordArena, Streams,
        VirtualMachine, VmInstance,
    },
    system::SystemRecords,
    utils::test_cpu_engine,
};
#[cfg(feature = "cuda")]
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_matrix_d2h_col_major, GpuBackend};
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::Program,
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SysPhantom, SystemOpcode, DEFERRAL_AS, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::{
    BaseAluOpcode, BaseAluWOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode,
    DivRemWOpcode, LessThanOpcode, MulHOpcode, MulOpcode, MulWOpcode, Rv64AuipcOpcode,
    Rv64HintStoreOpcode, Rv64JalLuiOpcode, Rv64JalrOpcode, Rv64LoadStoreOpcode, Rv64Phantom,
    ShiftOpcode, ShiftWOpcode,
};
use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};
#[cfg(feature = "cuda")]
use openvm_stark_backend::{
    prover::{ColMajorMatrix, MatrixDimensions, ProvingContext},
    StarkEngine,
};
#[cfg(feature = "cuda")]
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

#[cfg(feature = "cuda")]
use crate::Rv64ImGpuBuilder;
use crate::{Rv64ImConfig, Rv64ImCpuBuilder};

type F = BabyBear;

// M4 durable rvr-vs-interpreter preflight suite. Repeat locally with:
// cargo nextest run --cargo-profile=fast -p openvm-riscv-circuit --features rvr rvr_preflight

fn reg(idx: usize) -> usize {
    idx * RV64_REGISTER_NUM_LIMBS
}

fn addi(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(
        BaseAluOpcode::ADD.global_opcode(),
        [reg(rd), reg(rs1), imm, 1, 0],
    )
}

fn alu_r(opcode: BaseAluOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 1])
}

fn sltu(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(
        LessThanOpcode::SLTU.global_opcode(),
        [reg(rd), reg(rs1), reg(rs2), 1, 1],
    )
}

fn less_than(opcode: LessThanOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 1])
}

fn alu_w(opcode: BaseAluWOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 1])
}

fn shift(opcode: ShiftOpcode, rd: usize, rs1: usize, shamt: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), shamt, 1, 0])
}

fn shift_w(opcode: ShiftWOpcode, rd: usize, rs1: usize, shamt: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), shamt, 1, 0])
}

fn beq(rs1: usize, rs2: usize, offset: usize) -> Instruction<F> {
    Instruction::from_usize(
        BranchEqualOpcode::BEQ.global_opcode(),
        [reg(rs1), reg(rs2), offset, 1, 1],
    )
}

fn branch_eq(opcode: BranchEqualOpcode, rs1: usize, rs2: usize, offset: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rs1), reg(rs2), offset, 1, 1])
}

fn branch_lt(
    opcode: BranchLessThanOpcode,
    rs1: usize,
    rs2: usize,
    offset: usize,
) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rs1), reg(rs2), offset, 1, 1])
}

fn jal_lui(opcode: Rv64JalLuiOpcode, rd: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(
        opcode.global_opcode(),
        [
            reg(rd),
            0,
            imm,
            RV64_REGISTER_AS as usize,
            RV64_IMM_AS as usize,
            usize::from(rd != 0),
        ],
    )
}

fn jalr(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(
        Rv64JalrOpcode::JALR.global_opcode(),
        [
            reg(rd),
            reg(rs1),
            imm,
            RV64_REGISTER_AS as usize,
            RV64_IMM_AS as usize,
            usize::from(rd != 0),
            0,
        ],
    )
}

fn auipc(rd: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(
        Rv64AuipcOpcode::AUIPC.global_opcode(),
        [reg(rd), 0, imm, RV64_REGISTER_AS as usize, 0],
    )
}

fn mul(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(
        MulOpcode::MUL.global_opcode(),
        [reg(rd), reg(rs1), reg(rs2), 1, 0],
    )
}

fn mulh(opcode: MulHOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 0])
}

fn mul_w(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(
        MulWOpcode::MULW.global_opcode(),
        [reg(rd), reg(rs1), reg(rs2), 1, 0],
    )
}

fn divrem(opcode: DivRemOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 0])
}

fn divrem_w(opcode: DivRemWOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 0])
}

fn load(opcode: Rv64LoadStoreOpcode, rd: usize, rs1: usize, offset: usize) -> Instruction<F> {
    let enabled = usize::from(rd != 0);
    Instruction::from_usize(
        opcode.global_opcode(),
        [
            reg(rd),
            reg(rs1),
            offset,
            1,
            RV64_MEMORY_AS as usize,
            enabled,
            0,
        ],
    )
}

fn store(opcode: Rv64LoadStoreOpcode, rs2: usize, rs1: usize, offset: usize) -> Instruction<F> {
    Instruction::from_usize(
        opcode.global_opcode(),
        [reg(rs2), reg(rs1), offset, 1, RV64_MEMORY_AS as usize, 1, 0],
    )
}

fn hint_store(opcode: Rv64HintStoreOpcode, num_words: usize, ptr: usize) -> Instruction<F> {
    let a = if opcode == Rv64HintStoreOpcode::HINT_BUFFER {
        reg(num_words)
    } else {
        0
    };
    Instruction::from_usize(
        opcode.global_opcode(),
        [
            a,
            reg(ptr),
            0,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    )
}

fn extension_store(src: usize, ptr: usize, offset: usize) -> Instruction<F> {
    Instruction::from_usize(
        Rv64LoadStoreOpcode::STORED.global_opcode(),
        [
            reg(src),
            reg(ptr),
            offset,
            1,
            PUBLIC_VALUES_AS as usize,
            1,
            0,
        ],
    )
}

fn phantom(sys: SysPhantom) -> Instruction<F> {
    Instruction::from_isize(
        SystemOpcode::PHANTOM.global_opcode(),
        0,
        0,
        sys as isize,
        0,
        0,
    )
}

fn rv64_phantom_with_regs(phantom: Rv64Phantom, a: usize, b: usize) -> Instruction<F> {
    Instruction::from_isize(
        SystemOpcode::PHANTOM.global_opcode(),
        reg(a) as isize,
        reg(b) as isize,
        phantom as isize,
        0,
        0,
    )
}

fn rv64_phantom(phantom: Rv64Phantom) -> Instruction<F> {
    rv64_phantom_with_regs(phantom, 0, 0)
}

fn terminate() -> Instruction<F> {
    Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0)
}

fn exe(instructions: &[Instruction<F>]) -> VmExe<F> {
    VmExe::new(Program::from_instructions(instructions))
}

fn fibonacci_exe() -> VmExe<F> {
    let mut instructions = vec![addi(1, 0, 0), addi(2, 0, 1)];
    for _ in 0..8 {
        instructions.push(alu_r(BaseAluOpcode::ADD, 3, 1, 2));
        instructions.push(alu_r(BaseAluOpcode::ADD, 1, 2, 0));
        instructions.push(alu_r(BaseAluOpcode::ADD, 2, 3, 0));
    }
    instructions.push(terminate());
    exe(&instructions)
}

fn alu_vector_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 9),
        addi(2, 0, 5),
        alu_r(BaseAluOpcode::SUB, 3, 1, 2),
        alu_r(BaseAluOpcode::XOR, 4, 1, 2),
        alu_r(BaseAluOpcode::OR, 5, 1, 2),
        alu_r(BaseAluOpcode::AND, 6, 1, 2),
        sltu(7, 2, 1),
        terminate(),
    ])
}

fn mul_div_vector_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 21),
        addi(2, 0, 5),
        mul(3, 1, 2),
        divrem(DivRemOpcode::DIVU, 4, 1, 2),
        divrem(DivRemOpcode::REMU, 5, 1, 2),
        divrem(DivRemOpcode::DIV, 6, 1, 2),
        divrem(DivRemOpcode::REM, 7, 1, 2),
        terminate(),
    ])
}

fn phantom_subword_read_dominant_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 64),
        addi(6, 0, 72),
        addi(2, 0, 0x5a),
        phantom(SysPhantom::Nop),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 1),
        load(Rv64LoadStoreOpcode::LOADBU, 3, 1, 1),
        phantom(SysPhantom::CtStart),
        load(Rv64LoadStoreOpcode::LOADBU, 4, 1, 1),
        phantom(SysPhantom::CtEnd),
        store(Rv64LoadStoreOpcode::STOREH, 2, 6, 2),
        load(Rv64LoadStoreOpcode::LOADHU, 5, 6, 2),
        terminate(),
    ])
}

fn load_to_x0_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 64),
        addi(2, 0, 0x5a),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 1),
        load(Rv64LoadStoreOpcode::LOADBU, 0, 1, 1),
        addi(3, 0, 7),
        terminate(),
    ])
}

#[cfg(feature = "cuda")]
fn continuation_boundary_memory_exe() -> VmExe<F> {
    const TOUCHED_BLOCKS: usize = 135;
    // The empty initial image has no marked pages, including page zero. Keep the
    // pointer in ADDI's immediate range so interpreter and RVR semantics agree.
    let mut instructions = vec![addi(1, 0, 64), addi(2, 0, 0x5a)];
    for block_idx in 0..TOUCHED_BLOCKS {
        instructions.push(store(
            Rv64LoadStoreOpcode::STORED,
            2,
            1,
            block_idx * size_of::<u64>(),
        ));
    }
    // The taken branch is the exact block-aligned suspension point between segments.
    instructions.push(beq(0, 0, 4));
    for block_idx in 0..TOUCHED_BLOCKS {
        instructions.push(load(
            Rv64LoadStoreOpcode::LOADD,
            3,
            1,
            block_idx * size_of::<u64>(),
        ));
    }
    instructions.push(terminate());
    exe(&instructions)
}

fn block_boundary_branch_exe() -> VmExe<F> {
    exe(&[beq(0, 0, 8), addi(9, 0, 99), addi(1, 0, 7), terminate()])
}

fn push_standard_group_ops(instructions: &mut Vec<Instruction<F>>) {
    instructions.extend([
        alu_r(BaseAluOpcode::SUB, 4, 1, 2),
        alu_r(BaseAluOpcode::XOR, 5, 1, 2),
        alu_r(BaseAluOpcode::OR, 6, 1, 2),
        alu_r(BaseAluOpcode::AND, 7, 1, 2),
        less_than(LessThanOpcode::SLT, 8, 2, 1),
        less_than(LessThanOpcode::SLTU, 9, 2, 1),
        shift(ShiftOpcode::SLL, 12, 1, 2),
        shift(ShiftOpcode::SRL, 13, 12, 2),
        shift(ShiftOpcode::SRA, 14, 12, 2),
        alu_w(BaseAluWOpcode::ADDW, 15, 1, 2),
        alu_w(BaseAluWOpcode::SUBW, 16, 1, 2),
        shift_w(ShiftWOpcode::SLLW, 17, 1, 2),
        shift_w(ShiftWOpcode::SRLW, 18, 17, 2),
        shift_w(ShiftWOpcode::SRAW, 19, 17, 2),
        branch_eq(BranchEqualOpcode::BEQ, 1, 1, 4),
        branch_eq(BranchEqualOpcode::BNE, 1, 2, 4),
        branch_lt(BranchLessThanOpcode::BLT, 2, 1, 4),
        branch_lt(BranchLessThanOpcode::BLTU, 2, 1, 4),
        branch_lt(BranchLessThanOpcode::BGE, 1, 2, 4),
        branch_lt(BranchLessThanOpcode::BGEU, 1, 2, 4),
        jal_lui(Rv64JalLuiOpcode::LUI, 20, 0x12),
        auipc(21, 0),
        jal_lui(Rv64JalLuiOpcode::JAL, 22, 4),
    ]);

    let jalr_next_pc = (instructions.len() + 2) * 4;
    instructions.push(addi(10, 0, jalr_next_pc));
    instructions.push(jalr(23, 10, 0));

    instructions.push(mul(24, 1, 2));
    instructions.extend([
        mulh(MulHOpcode::MULH, 25, 1, 2),
        mulh(MulHOpcode::MULHSU, 26, 1, 2),
        mulh(MulHOpcode::MULHU, 27, 1, 2),
        mul_w(28, 1, 2),
    ]);
}

fn standard_group_exe(repeats: usize) -> VmExe<F> {
    let mut instructions = vec![addi(1, 0, 9), addi(2, 0, 5)];
    for _ in 0..repeats {
        push_standard_group_ops(&mut instructions);
    }
    instructions.push(terminate());
    exe(&instructions)
}

fn standard_group_with_add_tail_exe(add_tail_count: usize) -> VmExe<F> {
    let mut instructions = vec![addi(1, 0, 9), addi(2, 0, 5)];
    push_standard_group_ops(&mut instructions);
    for _ in 0..add_tail_count {
        instructions.push(alu_r(BaseAluOpcode::ADD, 3, 1, 2));
        instructions.push(alu_r(BaseAluOpcode::ADD, 1, 2, 0));
        instructions.push(alu_r(BaseAluOpcode::ADD, 2, 3, 0));
    }
    instructions.push(terminate());
    exe(&instructions)
}

fn push_hard_chip_ops(instructions: &mut Vec<Instruction<F>>) {
    instructions.extend([
        addi(1, 0, 64),
        addi(6, 0, 72),
        addi(7, 0, 96),
        addi(8, 0, 2),
        addi(20, 0, 112),
        addi(21, 0, 80),
        addi(24, 0, 88),
        addi(2, 0, 0x5a),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 1),
        load(Rv64LoadStoreOpcode::LOADBU, 3, 1, 1),
        load(Rv64LoadStoreOpcode::LOADB, 4, 1, 1),
        load(Rv64LoadStoreOpcode::LOADBU, 0, 1, 1),
        store(Rv64LoadStoreOpcode::STOREH, 2, 6, 2),
        load(Rv64LoadStoreOpcode::LOADHU, 5, 6, 2),
        load(Rv64LoadStoreOpcode::LOADH, 9, 6, 2),
        addi(10, 0, 21),
        store(Rv64LoadStoreOpcode::STOREW, 10, 21, 0),
        load(Rv64LoadStoreOpcode::LOADW, 22, 21, 0),
        load(Rv64LoadStoreOpcode::LOADWU, 25, 21, 0),
        store(Rv64LoadStoreOpcode::STORED, 10, 24, 0),
        load(Rv64LoadStoreOpcode::LOADD, 23, 24, 0),
        addi(11, 0, 5),
        divrem(DivRemOpcode::DIVU, 12, 10, 11),
        divrem(DivRemOpcode::REMU, 13, 10, 11),
        divrem(DivRemOpcode::DIV, 14, 10, 11),
        divrem(DivRemOpcode::REM, 15, 10, 11),
        divrem_w(DivRemWOpcode::DIVW, 16, 10, 11),
        divrem_w(DivRemWOpcode::DIVUW, 17, 10, 11),
        divrem_w(DivRemWOpcode::REMW, 18, 10, 11),
        divrem_w(DivRemWOpcode::REMUW, 19, 10, 11),
        phantom(SysPhantom::Nop),
        phantom(SysPhantom::CtStart),
        phantom(SysPhantom::CtEnd),
        hint_store(Rv64HintStoreOpcode::HINT_STORED, 0, 7),
        hint_store(Rv64HintStoreOpcode::HINT_BUFFER, 8, 20),
    ]);
}

fn hard_chip_exe(repeats: usize, branch_separated: bool) -> VmExe<F> {
    let mut instructions = Vec::new();
    for idx in 0..repeats {
        push_hard_chip_ops(&mut instructions);
        if branch_separated && idx + 1 < repeats {
            instructions.push(beq(0, 0, 4));
        }
    }
    instructions.push(terminate());
    exe(&instructions)
}

fn hard_chip_with_add_tail_exe(add_tail_count: usize) -> VmExe<F> {
    let mut instructions = Vec::new();
    push_hard_chip_ops(&mut instructions);
    instructions.push(beq(0, 0, 4));
    for _ in 0..add_tail_count {
        instructions.push(alu_r(BaseAluOpcode::ADD, 3, 1, 2));
        instructions.push(alu_r(BaseAluOpcode::ADD, 1, 2, 0));
        instructions.push(alu_r(BaseAluOpcode::ADD, 2, 3, 0));
    }
    instructions.push(terminate());
    exe(&instructions)
}

fn hard_chip_streams(repeats: usize) -> Streams<F> {
    let mut streams = Streams::default();
    for repeat in 0..repeats {
        for word in 0..3u64 {
            let value = 0x0102_0304_0506_0708u64 + (repeat as u64) * 0x100 + word;
            streams
                .hint_stream
                .extend(value.to_le_bytes().into_iter().map(F::from_u8));
        }
    }
    streams
}

fn full_rv64im_matrix_exe() -> VmExe<F> {
    let mut instructions = vec![addi(1, 0, 9), addi(2, 0, 5)];
    push_standard_group_ops(&mut instructions);
    push_hard_chip_ops(&mut instructions);
    instructions.push(terminate());
    exe(&instructions)
}

fn hint_input_then_memory_exe() -> VmExe<F> {
    exe(&[
        rv64_phantom(Rv64Phantom::HintInput),
        addi(1, 0, 64),
        addi(2, 0, 0x33),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 0),
        load(Rv64LoadStoreOpcode::LOADBU, 3, 1, 0),
        terminate(),
    ])
}

fn hint_random_then_memory_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 64),
        addi(2, 0, 0x44),
        addi(4, 0, 1),
        rv64_phantom_with_regs(Rv64Phantom::HintRandom, 4, 0),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 0),
        load(Rv64LoadStoreOpcode::LOADBU, 3, 1, 0),
        terminate(),
    ])
}

fn print_str_then_memory_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 64),
        addi(2, 0, b'O' as usize),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 0),
        addi(2, 0, b'K' as usize),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 1),
        addi(3, 0, 2),
        addi(4, 0, 0x55),
        rv64_phantom_with_regs(Rv64Phantom::PrintStr, 1, 3),
        store(Rv64LoadStoreOpcode::STOREB, 4, 1, 2),
        load(Rv64LoadStoreOpcode::LOADBU, 5, 1, 2),
        terminate(),
    ])
}

fn deferral_store() -> Instruction<F> {
    Instruction::from_usize(
        Rv64LoadStoreOpcode::STORED.global_opcode(),
        [reg(1), reg(2), 0, 1, DEFERRAL_AS as usize, 1, 0],
    )
}

fn public_values_reveal_exe(offset: usize) -> VmExe<F> {
    exe(&[
        addi(1, 0, 0x2a + offset),
        addi(2, 0, offset),
        extension_store(1, 2, 0),
        terminate(),
    ])
}

/// REVEAL differential fixture: repeated public-values stores (write-after-write
/// on the same AS=3 block, a second AS=3 block, an x0-sourced reveal) interleaved
/// with ordinary main-memory load/store traffic so register/memory
/// prev-timestamps around the reveals are non-trivial.
fn public_values_reveal_differential_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 0x2a),                            // x1 = value A
        addi(2, 0, 0),                               // x2 = public-values base pointer
        addi(3, 0, 0x77),                            // x3 = value B
        addi(4, 0, 64),                              // x4 = main-memory scratch pointer
        extension_store(1, 2, 0),                    // reveal A -> pv[0..8]
        store(Rv64LoadStoreOpcode::STORED, 1, 4, 0), // interleaved AS=2 store
        extension_store(3, 2, 0),                    // reveal B -> pv[0..8] (write-after-write)
        extension_store(1, 2, 8),                    // reveal A -> pv[8..16] (second AS=3 block)
        extension_store(0, 2, 16),                   // reveal x0 -> pv[16..24]
        load(Rv64LoadStoreOpcode::LOADD, 5, 4, 0),   // read back the AS=2 store
        terminate(),
    ])
}

fn hint_input_streams() -> Streams<F> {
    Streams::from(vec![vec![
        F::from_u8(0x11),
        F::from_u8(0x22),
        F::from_u8(0x33),
    ]])
}

fn repeated_adds_exe(count: usize) -> VmExe<F> {
    let mut instructions = vec![addi(1, 0, 1), addi(2, 0, 2)];
    for _ in 0..count {
        instructions.push(alu_r(BaseAluOpcode::ADD, 3, 1, 2));
        instructions.push(alu_r(BaseAluOpcode::ADD, 1, 2, 0));
        instructions.push(alu_r(BaseAluOpcode::ADD, 2, 3, 0));
    }
    instructions.push(terminate());
    exe(&instructions)
}

fn assert_system_records_eq(label: &str, interp: &SystemRecords<F>, rvr: &SystemRecords<F>) {
    assert_eq!(interp.from_state, rvr.from_state, "{label}: from_state");
    assert_eq!(interp.to_state, rvr.to_state, "{label}: to_state");
    assert_eq!(interp.exit_code, rvr.exit_code, "{label}: exit_code");
    assert_eq!(
        interp.filtered_exec_frequencies, rvr.filtered_exec_frequencies,
        "{label}: filtered_exec_frequencies"
    );
    assert_eq!(
        interp.touched_memory, rvr.touched_memory,
        "{label}: touched_memory"
    );
}

fn prove_rvr_preflight_and_verify_with_streams(
    exe: VmExe<F>,
    config: Rv64ImConfig,
    streams: Streams<F>,
) -> usize {
    let engine = test_cpu_engine();
    let (vm, pk) =
        VirtualMachine::new_with_keygen(engine, Rv64ImCpuBuilder, config).expect("vm init");
    let vk = pk.get_vk();
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance =
        VmInstance::new(vm, Arc::new(exe), cached_program_trace).expect("instance init");
    let proof = ContinuationVmProver::prove(&mut instance, streams).expect("prove");
    verify_segments(&instance.vm.engine, &vk, &proof.per_segment).expect("verify segments");
    proof.per_segment.len()
}

fn prove_rvr_preflight_and_verify(exe: VmExe<F>, config: Rv64ImConfig) -> usize {
    prove_rvr_preflight_and_verify_with_streams(exe, config, Streams::default())
}

fn assert_rvr_route_and_proves(
    label: &str,
    exe: VmExe<F>,
    config: Rv64ImConfig,
    streams: Streams<F>,
) -> usize {
    let engine = test_cpu_engine();
    let (vm, pk) =
        VirtualMachine::new_with_keygen(engine, Rv64ImCpuBuilder, config).expect("vm init");
    {
        let route = vm
            .preflight_routed_instance(&exe)
            .expect("route extension program");
        assert!(
            matches!(route, RvrPreflightRoute::Rvr(_)),
            "{label}: must route to rvr preflight"
        );
    }

    let vk = pk.get_vk();
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance =
        VmInstance::new(vm, Arc::new(exe), cached_program_trace).expect("instance init");
    let proof = ContinuationVmProver::prove(&mut instance, streams).expect("prove");
    verify_segments(&instance.vm.engine, &vk, &proof.per_segment).expect("verify segments");
    proof.per_segment.len()
}

fn assert_preflight_matches_interpreter_with_streams(
    label: &str,
    exe: VmExe<F>,
    num_insns: Option<u64>,
    streams: Streams<F>,
) -> RvrPreflightOutput<F> {
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");

    let trace_heights = vec![4096u32; vm.num_airs()];
    let mut interpreter = vm
        .preflight_interpreter(&exe)
        .expect("interpreter preflight");
    let interp_state = vm.create_initial_state(&exe, streams.clone());
    let interp_output = vm
        .execute_preflight(&mut interpreter, interp_state, num_insns, &trace_heights)
        .expect("interpreter execution");

    let route = vm
        .preflight_routed_instance(&exe)
        .expect("routed preflight instance");
    let RvrPreflightRoute::Rvr(instance) = route else {
        panic!("{label}: RV64IM program must route to RVR preflight");
    };
    let rvr_output = instance
        .execute_preflight(streams, num_insns)
        .expect("rvr preflight execution");

    assert_eq!(
        rvr_output.raw_logs.program_log.len(),
        rvr_output.instret as usize,
        "{label}: program-log length must equal instret"
    );
    assert_system_records_eq(
        label,
        &interp_output.system_records,
        &rvr_output.system_records,
    );

    if label == "phantom_subword_read_dominant" {
        assert_read_dominant_memory_aux(&rvr_output);
    }
    if label == "load_to_x0" {
        assert_load_to_x0_timestamp_tick(&rvr_output);
    }
    if let Some(target_instret) = num_insns {
        if rvr_output.suspended {
            assert_eq!(
                rvr_output.instret, target_instret,
                "{label}: suspended rvr preflight must retire exactly the requested block-boundary instret"
            );
        }
    }

    rvr_output
}

fn assert_preflight_matches_interpreter(
    label: &str,
    exe: VmExe<F>,
    num_insns: Option<u64>,
) -> RvrPreflightOutput<F> {
    assert_preflight_matches_interpreter_with_streams(label, exe, num_insns, Streams::default())
}

/// Which AIR traces to byte-compare between interpreter and rvr tracegen.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TraceCompareScope {
    /// Every AIR, including system AIRs.
    All,
    /// Every non-system AIR.
    NonSystem,
    /// Only the LoadStore AIR, selected by opcode→AIR-id from the fixture's
    /// public-values STORED (REVEAL) instruction. Shared periphery/lookup
    /// chips (Poseidon2 etc.) build their rows from unordered parallel
    /// iterators and are excluded from the byte-compare (HARD-5);
    /// `assert_system_records_eq` remains the order-independent oracle.
    RevealLoadStore,
}

fn assert_trace_matches_interpreter(
    label: &str,
    exe: VmExe<F>,
    streams: Streams<F>,
    scope: TraceCompareScope,
) {
    let (mut interp_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("interpreter vm init");
    let trace_heights = vec![4096u32; interp_vm.num_airs()];
    let interp_cached_program_trace = interp_vm.commit_program_on_device(&exe.program);
    interp_vm.load_program(interp_cached_program_trace);
    let interp_state = interp_vm.create_initial_state(&exe, streams.clone());
    interp_vm.transport_init_memory_to_device(&interp_state.memory);
    let mut interpreter = interp_vm
        .preflight_interpreter(&exe)
        .expect("interpreter preflight");
    let interp_output = interp_vm
        .execute_preflight(&mut interpreter, interp_state, None, &trace_heights)
        .expect("interpreter execution");

    let (mut rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("rvr vm init");
    let air_names = rvr_vm.air_names().map(str::to_owned).collect::<Vec<_>>();
    let num_sys_airs = rvr_vm.config().as_ref().num_airs();
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let rvr_initial_state = rvr_vm.create_initial_state(&exe, streams.clone());
    let rvr_output = {
        let route = rvr_vm
            .preflight_routed_instance(&exe)
            .expect("routed preflight instance");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("{label}: program must route to RVR preflight");
        };
        instance
            .execute_preflight(streams, None)
            .expect("rvr preflight execution")
    };
    assert_system_records_eq(
        label,
        &interp_output.system_records,
        &rvr_output.system_records,
    );
    let rvr_record_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        MatrixRecordArena<F>,
    >(&exe, &rvr_output, &capacities, &pc_to_air_idx)
    .expect("rvr log-native record assembly");

    let interp_ctx = interp_vm
        .generate_proving_ctx(interp_output.system_records, interp_output.record_arenas)
        .expect("interpreter trace generation");

    let rvr_cached_program_trace = rvr_vm.commit_program_on_device(&exe.program);
    rvr_vm.load_program(rvr_cached_program_trace);
    rvr_vm.transport_init_memory_to_device(&rvr_initial_state.memory);
    let rvr_ctx = rvr_vm
        .generate_proving_ctx(rvr_output.system_records, rvr_record_arenas)
        .expect("rvr trace generation");

    let reveal_loadstore_air_idx = (scope == TraceCompareScope::RevealLoadStore).then(|| {
        let index = exe
            .program
            .instructions_and_debug_infos
            .iter()
            .position(|slot| {
                slot.as_ref().is_some_and(|(insn, _)| {
                    insn.opcode == Rv64LoadStoreOpcode::STORED.global_opcode()
                        && insn.e.as_canonical_u32() == PUBLIC_VALUES_AS
                })
            })
            .expect("reveal fixture must contain a public-values STORED");
        pc_to_air_idx[index].expect("public-values STORED must map to the LoadStore AIR")
    });
    let in_scope = |air_idx: usize| match scope {
        TraceCompareScope::All => true,
        TraceCompareScope::NonSystem => air_idx >= num_sys_airs,
        TraceCompareScope::RevealLoadStore => Some(air_idx) == reveal_loadstore_air_idx,
    };
    let mut interp_chip_traces = interp_ctx
        .per_trace
        .into_iter()
        .filter(|(air_idx, _)| in_scope(*air_idx))
        .collect::<BTreeMap<_, _>>();
    let mut rvr_chip_traces = rvr_ctx
        .per_trace
        .into_iter()
        .filter(|(air_idx, _)| in_scope(*air_idx))
        .collect::<BTreeMap<_, _>>();
    let interp_air_ids = interp_chip_traces.keys().copied().collect::<Vec<_>>();
    let rvr_air_ids = rvr_chip_traces.keys().copied().collect::<Vec<_>>();
    assert_eq!(interp_air_ids, rvr_air_ids, "{label}: trace AIR set");
    assert!(!interp_air_ids.is_empty(), "{label}: expected traces");

    for air_idx in interp_air_ids {
        let interp_air = interp_chip_traces.remove(&air_idx).unwrap();
        let rvr_air = rvr_chip_traces.remove(&air_idx).unwrap();
        let air_name = air_names
            .get(air_idx)
            .map(String::as_str)
            .unwrap_or("<unknown air>");
        assert_eq!(
            interp_air.common_main.width, rvr_air.common_main.width,
            "{label}: {air_name} width"
        );
        assert_eq!(
            interp_air.common_main.values.len() / interp_air.common_main.width,
            rvr_air.common_main.values.len() / rvr_air.common_main.width,
            "{label}: {air_name} height"
        );
        assert_eq!(
            interp_air.common_main.values, rvr_air.common_main.values,
            "{label}: {air_name} values"
        );
        assert_eq!(
            interp_air.public_values, rvr_air.public_values,
            "{label}: {air_name} public values"
        );
    }
}

fn assert_standard_group_trace_matches_interpreter() {
    assert_trace_matches_interpreter(
        "standard_group_trace",
        standard_group_exe(1),
        Streams::default(),
        TraceCompareScope::NonSystem,
    );
}

#[cfg(feature = "cuda")]
const FULL_RV64IM_INSTRUCTION_AIR_COUNT: usize = 21;

#[cfg(feature = "cuda")]
type CpuProvingCtx = ProvingContext<CpuBackend<BabyBearPoseidon2Config>>;

#[cfg(feature = "cuda")]
type GpuProvingCtx = ProvingContext<GpuBackend>;

#[cfg(feature = "cuda")]
struct HostTrace {
    common_main: ColMajorMatrix<F>,
    public_values: Vec<F>,
}

#[cfg(feature = "cuda")]
fn is_full_rv64im_instruction_air(name: &str) -> bool {
    name.starts_with("VmAirWrapper<Rv64") || name == "Rv64HintStoreAir"
}

#[cfg(feature = "cuda")]
fn full_rv64im_instruction_air_ids(air_names: &[String]) -> Vec<usize> {
    let ids = air_names
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| is_full_rv64im_instruction_air(name).then_some(idx))
        .collect::<Vec<_>>();
    assert_eq!(
        ids.len(),
        FULL_RV64IM_INSTRUCTION_AIR_COUNT,
        "full RV64IM instruction AIR count changed: {:?}",
        ids.iter()
            .map(|idx| air_names[*idx].as_str())
            .collect::<Vec<_>>()
    );
    ids
}

#[cfg(feature = "cuda")]
fn persistent_boundary_air_id(air_names: &[String]) -> usize {
    let ids = air_names
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| name.starts_with("PersistentBoundaryAir<").then_some(idx))
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 1, "expected one persistent-boundary AIR");
    ids[0]
}

#[cfg(feature = "cuda")]
fn collect_cpu_trace_map(ctx: CpuProvingCtx) -> BTreeMap<usize, HostTrace> {
    ctx.per_trace
        .into_iter()
        .map(|(air_idx, ctx)| {
            (
                air_idx,
                HostTrace {
                    common_main: ColMajorMatrix::from_row_major(&ctx.common_main),
                    public_values: ctx.public_values,
                },
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn collect_gpu_trace_map(
    ctx: GpuProvingCtx,
    device_ctx: &GpuDeviceCtx,
) -> BTreeMap<usize, HostTrace> {
    ctx.per_trace
        .into_iter()
        .map(|(air_idx, ctx)| {
            let common_main = transport_matrix_d2h_col_major(&ctx.common_main, device_ctx).unwrap();
            (
                air_idx,
                HostTrace {
                    common_main,
                    public_values: ctx.public_values,
                },
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn assert_trace_values_eq(
    label: &str,
    air_name: &str,
    left_name: &str,
    right_name: &str,
    left: &[F],
    right: &[F],
) {
    if left == right {
        return;
    }
    let first_mismatch = left.iter().zip(right).position(|(l, r)| l != r);
    let detail = first_mismatch
        .map(|idx| {
            format!(
                "first_mismatch={idx} left={} right={}",
                left[idx], right[idx]
            )
        })
        .unwrap_or_else(|| "no common-index mismatch; lengths differ".to_string());
    panic!(
        "{label}: {air_name} values differ between {left_name} and {right_name}: left_len={} right_len={} {detail}",
        left.len(),
        right.len()
    );
}

#[cfg(feature = "cuda")]
fn assert_host_trace_eq(
    label: &str,
    air_name: &str,
    left_name: &str,
    right_name: &str,
    left: &HostTrace,
    right: &HostTrace,
) {
    assert_eq!(
        left.common_main.width(),
        right.common_main.width(),
        "{label}: {air_name} width differs between {left_name} and {right_name}"
    );
    assert_eq!(
        left.common_main.height(),
        right.common_main.height(),
        "{label}: {air_name} height differs between {left_name} and {right_name}"
    );
    assert_trace_values_eq(
        label,
        air_name,
        left_name,
        right_name,
        &left.common_main.values,
        &right.common_main.values,
    );
    assert_eq!(
        left.public_values, right.public_values,
        "{label}: {air_name} public values differ between {left_name} and {right_name}"
    );
}

#[cfg(feature = "cuda")]
fn assert_trace_maps_eq(
    label: &str,
    left_name: &str,
    right_name: &str,
    air_names: &[String],
    left: &BTreeMap<usize, HostTrace>,
    right: &BTreeMap<usize, HostTrace>,
) {
    let left_air_ids = left.keys().copied().collect::<Vec<_>>();
    let right_air_ids = right.keys().copied().collect::<Vec<_>>();
    assert_eq!(
        left_air_ids, right_air_ids,
        "{label}: non-empty AIR set differs between {left_name} and {right_name}"
    );
    assert!(
        !left_air_ids.is_empty(),
        "{label}: expected non-empty traces for {left_name}"
    );
    for air_idx in left_air_ids {
        let air_name = air_names
            .get(air_idx)
            .map(String::as_str)
            .unwrap_or("<unknown air>");
        assert_host_trace_eq(
            label,
            air_name,
            left_name,
            right_name,
            left.get(&air_idx).unwrap(),
            right.get(&air_idx).unwrap(),
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_trace_maps_eq_for_air_ids(
    label: &str,
    left_name: &str,
    right_name: &str,
    air_names: &[String],
    left: &BTreeMap<usize, HostTrace>,
    right: &BTreeMap<usize, HostTrace>,
    air_ids: &[usize],
) {
    assert!(
        !air_ids.is_empty(),
        "{label}: expected at least one active instruction AIR"
    );
    for &air_idx in air_ids {
        let air_name = air_names
            .get(air_idx)
            .map(String::as_str)
            .unwrap_or("<unknown air>");
        let left_trace = left
            .get(&air_idx)
            .unwrap_or_else(|| panic!("{label}: {left_name} missing {air_name}"));
        let right_trace = right
            .get(&air_idx)
            .unwrap_or_else(|| panic!("{label}: {right_name} missing {air_name}"));
        assert_host_trace_eq(
            label,
            air_name,
            left_name,
            right_name,
            left_trace,
            right_trace,
        );
    }
}

#[cfg(feature = "cuda")]
fn prove_gpu_rvr_preflight_and_verify_with_streams(
    exe: VmExe<F>,
    config: Rv64ImConfig,
    streams: Streams<F>,
) -> usize {
    let engine = test_gpu_engine();
    let (vm, pk) =
        VirtualMachine::new_with_keygen(engine, Rv64ImGpuBuilder, config).expect("gpu vm init");
    let vk = pk.get_vk();
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance =
        VmInstance::new(vm, Arc::new(exe), cached_program_trace).expect("gpu instance init");
    let proof = ContinuationVmProver::prove(&mut instance, streams).expect("gpu prove");
    verify_segments(&instance.vm.engine, &vk, &proof.per_segment).expect("gpu verify segments");
    proof.per_segment.len()
}

#[cfg(feature = "cuda")]
fn assert_gpu_rvr_three_way_from_state(
    label: &str,
    exe: &VmExe<F>,
    config: &Rv64ImConfig,
    from_state: VmState<F>,
    num_insns: Option<u64>,
    trace_heights: &[u32],
    expected_active_instruction_air_count: Option<usize>,
) -> VmState<F> {
    let (mut cpu_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("cpu vm init");
    let cpu_air_names = cpu_vm.air_names().map(str::to_owned).collect::<Vec<_>>();
    let mut cpu_interpreter = cpu_vm
        .preflight_interpreter(exe)
        .expect("cpu interpreter preflight");
    let cpu_output = cpu_vm
        .execute_preflight(
            &mut cpu_interpreter,
            from_state.clone(),
            num_insns,
            trace_heights,
        )
        .expect("cpu interpreter execution");

    let (mut gpu_arena_vm, _) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64ImGpuBuilder, config.clone())
            .expect("gpu arena vm init");
    let gpu_arena_air_names = gpu_arena_vm
        .air_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert_eq!(
        cpu_air_names, gpu_arena_air_names,
        "{label}: CPU/GPU AIR name order"
    );
    let mut gpu_interpreter = gpu_arena_vm
        .preflight_interpreter(exe)
        .expect("gpu interpreter preflight");
    let gpu_arena_output = gpu_arena_vm
        .execute_preflight(
            &mut gpu_interpreter,
            from_state.clone(),
            num_insns,
            trace_heights,
        )
        .expect("gpu interpreter execution");

    let (mut gpu_log_vm, _) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64ImGpuBuilder, config.clone())
            .expect("gpu log vm init");
    let air_names = gpu_log_vm
        .air_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert_eq!(
        gpu_arena_air_names, air_names,
        "{label}: GPU log/arena AIR name order"
    );
    let instruction_air_ids = full_rv64im_instruction_air_ids(&air_names);
    let rvr_output = {
        let route = gpu_log_vm
            .preflight_routed_instance(exe)
            .expect("routed preflight instance");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("{label}: program must route to RVR preflight");
        };
        instance
            .execute_preflight_from_state(from_state.clone(), num_insns)
            .expect("gpu rvr preflight execution")
    };
    assert_system_records_eq(
        &format!("{label}: gpu-record-arenas"),
        &cpu_output.system_records,
        &gpu_arena_output.system_records,
    );
    assert_system_records_eq(
        &format!("{label}: gpu-rvr-logs"),
        &cpu_output.system_records,
        &rvr_output.system_records,
    );

    let rvr_to_state = rvr_output.to_state.clone();
    let capacities = trace_heights
        .iter()
        .zip(gpu_log_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = gpu_log_vm.pc_to_air_idx(exe).expect("pc to air mapping");
    let gpu_log_record_arenas = Rv64ImGpuBuilder
        .generate_rvr_record_arenas_from_logs(config, exe, &rvr_output, &capacities, &pc_to_air_idx)
        .expect("gpu builder log-native record assembly")
        .expect("gpu builder must support rvr log-native tracegen");

    let cpu_cached_program_trace = cpu_vm.commit_program_on_device(&exe.program);
    cpu_vm.load_program(cpu_cached_program_trace);
    cpu_vm.transport_init_memory_to_device(&from_state.memory);
    let cpu_ctx = cpu_vm
        .generate_proving_ctx(cpu_output.system_records, cpu_output.record_arenas)
        .expect("cpu interpreter trace generation");

    let gpu_arena_cached_program_trace = gpu_arena_vm.commit_program_on_device(&exe.program);
    gpu_arena_vm.load_program(gpu_arena_cached_program_trace);
    gpu_arena_vm.transport_init_memory_to_device(&from_state.memory);
    let gpu_arena_ctx = gpu_arena_vm
        .generate_proving_ctx(
            gpu_arena_output.system_records,
            gpu_arena_output.record_arenas,
        )
        .expect("gpu record-arena trace generation");
    let gpu_arena_device_ctx = gpu_arena_vm.engine.device().device_ctx.clone();

    let gpu_log_cached_program_trace = gpu_log_vm.commit_program_on_device(&exe.program);
    gpu_log_vm.load_program(gpu_log_cached_program_trace);
    gpu_log_vm.transport_init_memory_to_device(&from_state.memory);
    let gpu_log_ctx = gpu_log_vm
        .generate_proving_ctx(rvr_output.system_records, gpu_log_record_arenas)
        .expect("gpu rvr-log trace generation");
    let gpu_log_device_ctx = gpu_log_vm.engine.device().device_ctx.clone();

    let cpu_traces = collect_cpu_trace_map(cpu_ctx);
    let gpu_arena_traces = collect_gpu_trace_map(gpu_arena_ctx, &gpu_arena_device_ctx);
    let gpu_log_traces = collect_gpu_trace_map(gpu_log_ctx, &gpu_log_device_ctx);

    assert_trace_maps_eq(
        label,
        "gpu_from_rvr_logs",
        "gpu_from_record_arenas",
        &air_names,
        &gpu_log_traces,
        &gpu_arena_traces,
    );

    let active_instruction_air_ids = instruction_air_ids
        .into_iter()
        .filter(|air_idx| gpu_arena_traces.contains_key(air_idx))
        .collect::<Vec<_>>();
    if let Some(expected) = expected_active_instruction_air_count {
        assert_eq!(
            active_instruction_air_ids.len(),
            expected,
            "{label}: active full-RV64IM instruction AIR coverage: {:?}",
            active_instruction_air_ids
                .iter()
                .map(|idx| air_names[*idx].as_str())
                .collect::<Vec<_>>()
        );
    }
    assert_trace_maps_eq_for_air_ids(
        label,
        "gpu_from_record_arenas",
        "cpu_interpreter",
        &air_names,
        &gpu_arena_traces,
        &cpu_traces,
        &active_instruction_air_ids,
    );
    assert_trace_maps_eq_for_air_ids(
        label,
        "gpu_from_record_arenas",
        "cpu_interpreter",
        &air_names,
        &gpu_arena_traces,
        &cpu_traces,
        &[persistent_boundary_air_id(&air_names)],
    );

    rvr_to_state
}

#[cfg(feature = "cuda")]
fn assert_gpu_rvr_three_way_single_segment(
    label: &str,
    exe: VmExe<F>,
    streams: Streams<F>,
    expected_active_instruction_air_count: Option<usize>,
) {
    let config = Rv64ImConfig::default();
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("vm init");
    let trace_heights = vec![4096u32; vm.num_airs()];
    let from_state = vm.create_initial_state(&exe, streams);
    assert_gpu_rvr_three_way_from_state(
        label,
        &exe,
        &config,
        from_state,
        None,
        &trace_heights,
        expected_active_instruction_air_count,
    );
}

#[cfg(feature = "cuda")]
fn assert_gpu_rvr_three_way_multi_segment(label: &str, exe: VmExe<F>, streams: Streams<F>) {
    let mut config = Rv64ImConfig::default();
    config.rv64i.system.segmentation_max_memory = 1;
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("metered vm init");
    let segments = {
        let metered_ctx = vm.build_metered_ctx(&exe);
        let metered = vm.metered_interpreter(&exe).expect("metered interpreter");
        let (segments, _) = metered
            .execute_metered(streams.clone(), metered_ctx)
            .expect("metered execution");
        segments
    };
    assert!(
        segments.len() > 1,
        "{label}: tight segmentation memory limit should force multiple segments"
    );
    let mut state = vm.create_initial_state(&exe, streams);
    for (idx, segment) in segments.into_iter().enumerate() {
        state = assert_gpu_rvr_three_way_from_state(
            &format!("{label}_segment_{idx}"),
            &exe,
            &config,
            state,
            Some(segment.num_insns),
            &segment.trace_heights,
            None,
        );
    }
}

fn assert_read_dominant_memory_aux(output: &RvrPreflightOutput<F>) {
    let block_ptr = 64 / 2;
    let block_addr = (RV64_MEMORY_AS, block_ptr);
    let block_accesses = output
        .access_aux
        .iter()
        .filter(|aux| aux.block_addr == block_addr)
        .collect::<Vec<_>>();
    let write = block_accesses
        .iter()
        .find(|aux| aux.entry.kind == PREFLIGHT_MEMORY_KIND_WRITE)
        .expect("store byte writes memory block");
    let reads_after_write = block_accesses
        .iter()
        .filter(|aux| {
            aux.entry.kind == PREFLIGHT_MEMORY_KIND_READ
                && aux.entry.timestamp > write.entry.timestamp
        })
        .take(2)
        .collect::<Vec<_>>();
    assert_eq!(
        reads_after_write.len(),
        2,
        "program must contain read-after-write followed by read-after-read"
    );

    let expected_values = [
        F::from_u32(0x5a00),
        F::from_u32(0),
        F::from_u32(0),
        F::from_u32(0),
    ];
    assert_eq!(
        reads_after_write[0].prev_timestamp, write.entry.timestamp,
        "read-after-write prev_timestamp must chain to write access"
    );
    assert_eq!(
        reads_after_write[0].prev_data, expected_values,
        "read-after-write prev_data must contain the written byte"
    );
    assert_eq!(
        reads_after_write[1].prev_timestamp, reads_after_write[0].entry.timestamp,
        "read-after-read prev_timestamp must chain to the previous read"
    );
    assert_eq!(
        reads_after_write[1].prev_data, expected_values,
        "read-after-read prev_data must preserve the last written value"
    );

    let touched = output
        .system_records
        .touched_memory
        .iter()
        .find(|(addr, _)| *addr == block_addr)
        .expect("block must be touched");
    assert_eq!(
        touched.1.timestamp, reads_after_write[1].entry.timestamp,
        "touched_memory timestamp must be the trailing read timestamp"
    );
    assert_eq!(
        touched.1.values, expected_values,
        "touched_memory value must remain the last write"
    );
}

fn assert_load_to_x0_timestamp_tick(output: &RvrPreflightOutput<F>) {
    let load_pc = 3u32 * 4;
    let load_idx = output
        .raw_logs
        .program_log
        .iter()
        .position(|entry| entry.pc == u64::from(load_pc))
        .expect("load-to-x0 instruction must be in program log");
    let load_timestamp = output.raw_logs.program_log[load_idx].timestamp;
    let next_timestamp = output.raw_logs.program_log[load_idx + 1].timestamp;
    assert_eq!(
        next_timestamp - load_timestamp,
        3,
        "rd=x0 load must preserve the fixed 3-tick load invariant"
    );

    let load_accesses = output
        .raw_logs
        .memory_log
        .iter()
        .filter(|entry| entry.timestamp >= load_timestamp && entry.timestamp < next_timestamp)
        .collect::<Vec<_>>();
    assert_eq!(
        load_accesses.len(),
        2,
        "rd=x0 load logs rs1 read + memory read, then ticks without a register-write log"
    );
    assert!(load_accesses.iter().any(|entry| {
        entry.timestamp == load_timestamp
            && entry.kind == PREFLIGHT_MEMORY_KIND_READ
            && entry.addr_space == RV64_REGISTER_AS as u8
            && entry.address == reg(1) as u64
    }));
    assert!(load_accesses.iter().any(|entry| {
        entry.timestamp == load_timestamp + 1
            && entry.kind == PREFLIGHT_MEMORY_KIND_READ
            && entry.addr_space == RV64_MEMORY_AS as u8
    }));
    assert!(
        !load_accesses.iter().any(|entry| {
            entry.kind == PREFLIGHT_MEMORY_KIND_WRITE
                && entry.addr_space == RV64_REGISTER_AS as u8
                && entry.address == reg(0) as u64
        }),
        "rd=x0 load must not emit an AS_REGISTER[0] write"
    );
}

fn assert_phantom_timestamp_tick(
    label: &str,
    output: &RvrPreflightOutput<F>,
    phantom_pc: u32,
    next_pc: u32,
) {
    let hint_entry = output
        .raw_logs
        .program_log
        .iter()
        .find(|entry| entry.pc == u64::from(phantom_pc))
        .expect("phantom program-log entry");
    let next_entry = output
        .raw_logs
        .program_log
        .iter()
        .find(|entry| entry.pc == u64::from(next_pc))
        .expect("post-phantom program-log entry");
    assert_eq!(
        next_entry.timestamp,
        hint_entry.timestamp + 1,
        "{label}: Rv64 phantom must tick the shared timestamp before the following instruction"
    );
    assert!(output.raw_logs.memory_log.iter().any(|entry| {
        entry.timestamp >= next_entry.timestamp && entry.addr_space == RV64_MEMORY_AS as u8
    }));
}

#[test]
fn rvr_preflight_differential_suite_system_records_full_rv64im_matrix() {
    let programs = [
        ("fibonacci", fibonacci_exe()),
        ("alu_vector", alu_vector_exe()),
        ("mul_div_vector", mul_div_vector_exe()),
        ("standard_group", standard_group_exe(1)),
        (
            "phantom_subword_read_dominant",
            phantom_subword_read_dominant_exe(),
        ),
        ("load_to_x0", load_to_x0_exe()),
    ];
    for (label, exe) in programs {
        assert_preflight_matches_interpreter(label, exe, None);
    }
    assert_preflight_matches_interpreter_with_streams(
        "hard_chip_full_set",
        hard_chip_exe(1, false),
        None,
        hard_chip_streams(1),
    );
    assert_preflight_matches_interpreter_with_streams(
        "full_rv64im_matrix",
        full_rv64im_matrix_exe(),
        None,
        hard_chip_streams(1),
    );
}

#[test]
fn rvr_preflight_hint_input_phantom_ticks_before_memory_access() {
    let output = assert_preflight_matches_interpreter_with_streams(
        "hint_input_then_memory",
        hint_input_then_memory_exe(),
        None,
        hint_input_streams(),
    );
    assert_phantom_timestamp_tick("hint_input_then_memory", &output, 0, 4);
    assert_trace_matches_interpreter(
        "hint_input_then_memory_trace",
        hint_input_then_memory_exe(),
        hint_input_streams(),
        TraceCompareScope::All,
    );
}

#[test]
fn rvr_preflight_hint_random_phantom_ticks_before_memory_access() {
    let output = assert_preflight_matches_interpreter(
        "hint_random_then_memory",
        hint_random_then_memory_exe(),
        None,
    );
    assert_phantom_timestamp_tick("hint_random_then_memory", &output, 3 * 4, 4 * 4);
    assert_trace_matches_interpreter(
        "hint_random_then_memory_trace",
        hint_random_then_memory_exe(),
        Streams::default(),
        TraceCompareScope::All,
    );
}

#[test]
fn rvr_preflight_print_str_phantom_ticks_before_memory_access() {
    let output = assert_preflight_matches_interpreter(
        "print_str_then_memory",
        print_str_then_memory_exe(),
        None,
    );
    assert_phantom_timestamp_tick("print_str_then_memory", &output, 7 * 4, 8 * 4);
    assert_trace_matches_interpreter(
        "print_str_then_memory_trace",
        print_str_then_memory_exe(),
        Streams::default(),
        TraceCompareScope::All,
    );
}

#[test]
fn rvr_preflight_routing_finalization_rv64im_to_rvr_extension_to_interpreter() {
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let rvr_route = vm
        .preflight_routed_instance(&hard_chip_exe(1, false))
        .expect("route RV64IM program");
    assert!(matches!(rvr_route, RvrPreflightRoute::Rvr(_)));

    // The public-values REVEAL (STORED with e == PUBLIC_VALUES_AS) is a
    // supported base-seam opcode: the classifier must return Supported and
    // route it to the rvr preflight (the exact reth blocker instruction).
    for (label, exe) in [
        ("public_values_reveal_0", public_values_reveal_exe(0)),
        ("public_values_reveal_8", public_values_reveal_exe(8)),
    ] {
        let route = vm
            .preflight_routed_instance(&exe)
            .expect("route reveal program");
        assert!(
            matches!(route, RvrPreflightRoute::Rvr(_)),
            "{label}: reveal must route to rvr preflight"
        );
    }

    // A store to an address space outside {RV64_MEMORY_AS, PUBLIC_VALUES_AS}
    // still finalizes to the interpreter.
    let route = vm
        .preflight_routed_instance(&exe(&[deferral_store(), terminate()]))
        .expect("route deferral-store program");
    assert!(
        route.is_interpreter(),
        "deferral-store: must route to interpreter"
    );
}

#[test]
fn rvr_preflight_reveal_registry_admits_stores_only() {
    // Registry-level HARD-2 proof: the single registered predicate that gates
    // routing also gates assembly, so checking `contains_instruction` checks
    // both. Stores admit e in {RV64_MEMORY_AS, PUBLIC_VALUES_AS}; loads and
    // other address spaces stay rejected.
    let mut registry = LogNativeAssemblerRegistry::<F, MatrixRecordArena<F>>::new();
    Rv64ImConfig::default().extend_rvr_log_native(&mut registry);

    assert!(
        registry.contains_instruction(&extension_store(1, 2, 0)),
        "STORED with e == PUBLIC_VALUES_AS must be admitted"
    );
    assert!(
        registry.contains_instruction(&store(Rv64LoadStoreOpcode::STORED, 1, 2, 0)),
        "STORED with e == RV64_MEMORY_AS must stay admitted"
    );
    assert!(
        !registry.contains_instruction(&deferral_store()),
        "STORED with e == DEFERRAL_AS must stay rejected"
    );

    let mut public_values_load = load(Rv64LoadStoreOpcode::LOADD, 1, 2, 0);
    public_values_load.e = F::from_u32(PUBLIC_VALUES_AS);
    assert!(
        !registry.contains_instruction(&public_values_load),
        "LOADD with e == PUBLIC_VALUES_AS must stay rejected"
    );
    let mut public_values_sign_extend_load = load(Rv64LoadStoreOpcode::LOADW, 1, 2, 0);
    public_values_sign_extend_load.e = F::from_u32(PUBLIC_VALUES_AS);
    assert!(
        !registry.contains_instruction(&public_values_sign_extend_load),
        "LOADW with e == PUBLIC_VALUES_AS must stay rejected"
    );
}

#[test]
fn rvr_preflight_reveal_differential_matches_interpreter() {
    let exe = public_values_reveal_differential_exe();
    // SystemRecords parity: from/to state (next timestamp), exec frequencies,
    // and touched_memory including the AS=3 blocks' values and last-access
    // timestamps.
    assert_preflight_matches_interpreter("public_values_reveal_diff", exe.clone(), None);
    // Byte-compare the LoadStore AIR trace, selected by opcode→AIR-id.
    assert_trace_matches_interpreter(
        "public_values_reveal_diff_trace",
        exe,
        Streams::default(),
        TraceCompareScope::RevealLoadStore,
    );
}

#[test]
fn rvr_preflight_reveal_routes_prove_and_verify_end_to_end() {
    let config = Rv64ImConfig::with_public_values_bytes(16);
    for (label, exe) in [
        ("public_values_reveal_0", public_values_reveal_exe(0)),
        ("public_values_reveal_8", public_values_reveal_exe(8)),
    ] {
        let segments = assert_rvr_route_and_proves(label, exe, config.clone(), Streams::default());
        assert_eq!(
            segments, 1,
            "{label}: small reveal program is single segment"
        );
    }
}

#[test]
fn rvr_preflight_proves_reveal_multi_segment() {
    // The reveal's AS=3 write must be accounted by rvr metered so segment
    // boundaries still coincide with preflight when segmentation is tight.
    let mut config = Rv64ImConfig::with_public_values_bytes(16);
    config.rv64i.system.segmentation_max_memory = 1;
    let mut instructions = vec![
        addi(1, 0, 0x2a),
        addi(2, 0, 0),
        extension_store(1, 2, 0),
        extension_store(1, 2, 8),
    ];
    for _ in 0..400 {
        instructions.push(alu_r(BaseAluOpcode::ADD, 3, 1, 2));
        instructions.push(alu_r(BaseAluOpcode::ADD, 1, 2, 0));
        instructions.push(alu_r(BaseAluOpcode::ADD, 2, 3, 0));
    }
    instructions.push(terminate());
    let segments = prove_rvr_preflight_and_verify(exe(&instructions), config);
    assert!(
        segments > 1,
        "tight segmentation memory limit should force the reveal program into multiple segments"
    );
}

#[test]
fn rvr_preflight_block_aligned_partial_bound_matches_interpreter() {
    let output = assert_preflight_matches_interpreter(
        "block_aligned_branch_boundary_n1",
        block_boundary_branch_exe(),
        Some(1),
    );
    assert!(output.suspended);
    assert_eq!(output.instret, 1);
    assert_eq!(output.raw_logs.program_log.len(), 1);
    assert_eq!(output.system_records.to_state.pc, 8);
    assert_eq!(output.system_records.exit_code, None);
}

#[test]
fn rvr_preflight_rejects_mid_block_partial_bound() {
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let exe = alu_vector_exe();
    let route = vm
        .preflight_routed_instance(&exe)
        .expect("routed preflight instance");
    let RvrPreflightRoute::Rvr(instance) = route else {
        panic!("RV64IM program must route to RVR preflight");
    };
    let err = match instance.execute_preflight(Streams::default(), Some(5)) {
        Ok(_) => panic!("mid-block bounded preflight must be rejected"),
        Err(err) => err,
    };
    match err {
        ExecutionError::RvrExecution(msg) => {
            assert!(
                msg.contains("mid-block rvr preflight suspension unsupported"),
                "unexpected error: {msg}"
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn rvr_preflight_proves_and_verifies_single_segment() {
    let segments = prove_rvr_preflight_and_verify(
        exe(&[
            addi(1, 0, 9),
            addi(2, 0, 5),
            alu_r(BaseAluOpcode::ADD, 3, 1, 2),
            terminate(),
        ]),
        Rv64ImConfig::default(),
    );
    assert_eq!(segments, 1);
}

#[test]
fn rvr_preflight_proves_standard_group_single_segment() {
    assert_preflight_matches_interpreter("standard_group_proof", standard_group_exe(1), None);
    let segments = prove_rvr_preflight_and_verify(standard_group_exe(1), Rv64ImConfig::default());
    assert_eq!(segments, 1);
}

#[test]
fn rvr_preflight_standard_group_trace_matches_interpreter() {
    assert_standard_group_trace_matches_interpreter();
}

#[cfg(feature = "cuda")]
#[test]
fn rvr_gpu_log_native_full_rv64im_three_way_matrix() {
    assert_gpu_rvr_three_way_single_segment(
        "full_rv64im_matrix",
        full_rv64im_matrix_exe(),
        hard_chip_streams(1),
        Some(FULL_RV64IM_INSTRUCTION_AIR_COUNT),
    );
    assert_gpu_rvr_three_way_single_segment(
        "hint_input_then_memory",
        hint_input_then_memory_exe(),
        hint_input_streams(),
        None,
    );
    assert_gpu_rvr_three_way_single_segment(
        "hint_random_then_memory",
        hint_random_then_memory_exe(),
        Streams::default(),
        None,
    );
    assert_gpu_rvr_three_way_single_segment(
        "print_str_then_memory",
        print_str_then_memory_exe(),
        Streams::default(),
        None,
    );
    assert_gpu_rvr_three_way_multi_segment(
        "hard_chip_multi_segment",
        hard_chip_with_add_tail_exe(400),
        hard_chip_streams(1),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rvr_gpu_log_native_continuation_boundary_memory_matches_cpu() {
    const FIRST_SEGMENT_INSNS: u64 = 2 + 135 + 1;
    let exe = continuation_boundary_memory_exe();
    let config = Rv64ImConfig::default();
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("vm init");
    let trace_heights = vec![4096u32; vm.num_airs()];
    let state = vm.create_initial_state(&exe, Streams::default());
    let state = assert_gpu_rvr_three_way_from_state(
        "continuation_boundary_memory_segment_0",
        &exe,
        &config,
        state,
        Some(FIRST_SEGMENT_INSNS),
        &trace_heights,
        None,
    );
    assert_gpu_rvr_three_way_from_state(
        "continuation_boundary_memory_segment_1",
        &exe,
        &config,
        state,
        None,
        &trace_heights,
        None,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rvr_gpu_log_native_full_rv64im_proves_and_verifies() {
    let segments = prove_gpu_rvr_preflight_and_verify_with_streams(
        full_rv64im_matrix_exe(),
        Rv64ImConfig::default(),
        hard_chip_streams(1),
    );
    assert_eq!(segments, 1);
}

#[test]
fn rvr_preflight_hard_chip_trace_matches_interpreter() {
    assert_trace_matches_interpreter(
        "hard_chip_trace",
        hard_chip_exe(1, false),
        hard_chip_streams(1),
        TraceCompareScope::All,
    );
}

#[test]
fn rvr_preflight_proves_hard_chip_single_segment() {
    assert_preflight_matches_interpreter_with_streams(
        "hard_chip_proof",
        hard_chip_exe(1, false),
        None,
        hard_chip_streams(1),
    );
    let segments = prove_rvr_preflight_and_verify_with_streams(
        hard_chip_exe(1, false),
        Rv64ImConfig::default(),
        hard_chip_streams(1),
    );
    assert_eq!(segments, 1);
}

#[test]
fn rvr_preflight_proves_standard_group_multi_segment() {
    let mut config = Rv64ImConfig::default();
    config.rv64i.system.segmentation_max_memory = 1;
    let segments = prove_rvr_preflight_and_verify(standard_group_with_add_tail_exe(400), config);
    assert!(
        segments > 1,
        "tight segmentation memory limit should force standard-group program into multiple segments"
    );
}

#[test]
fn rvr_preflight_proves_hard_chip_multi_segment() {
    let mut config = Rv64ImConfig::default();
    config.rv64i.system.segmentation_max_memory = 1;
    let segments = prove_rvr_preflight_and_verify_with_streams(
        hard_chip_with_add_tail_exe(400),
        config,
        hard_chip_streams(1),
    );
    assert!(
        segments > 1,
        "tight segmentation memory limit should force hard-chip program into multiple segments"
    );
}

#[test]
fn rvr_preflight_proves_and_verifies_multi_segment() {
    reset_preflight_compile_invocations_for_test();
    let mut config = Rv64ImConfig::default();
    config.rv64i.system.segmentation_max_memory = 1;
    let segments = prove_rvr_preflight_and_verify(repeated_adds_exe(400), config);
    assert!(
        segments > 1,
        "tight segmentation memory limit should force multiple segments"
    );
    assert_eq!(
        preflight_compile_invocations_for_test(),
        1,
        "rvr preflight native library must be compiled once and reused across all {segments} segments"
    );
}
