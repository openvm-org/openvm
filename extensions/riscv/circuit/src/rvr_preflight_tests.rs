use std::{collections::BTreeMap, sync::Arc};

#[cfg(feature = "cuda")]
use openvm_circuit::arch::rvr::preflight::RvrArenaNativeTarget;
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
            LogNativeAssemblerRegistry, RvrPreflightEngine, RvrPreflightOutput, RvrPreflightRoute,
            VmRvrLogNativeExtension,
        },
        verify_segments, AddressSpaceHostLayout, Arena, ContinuationVmProver, DenseRecordArena,
        ExecutionError, MatrixRecordArena, Streams, VirtualMachine, VmInstance,
    },
    system::{memory::online::LinearMemory, SystemRecords},
    utils::test_cpu_engine,
};
#[cfg(feature = "cuda")]
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{
    data_transporter::transport_matrix_d2h_col_major, BabyBearPoseidon2GpuEngine, GpuBackend,
};
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SysPhantom, SystemOpcode, DEFERRAL_AS, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWOpcode, BranchEqualOpcode, BranchLessThanOpcode,
    DivRemOpcode, DivRemWOpcode, LessThanOpcode, MulHOpcode, MulOpcode, MulWOpcode,
    Rv64AuipcOpcode, Rv64HintStoreOpcode, Rv64JalLuiOpcode, Rv64JalrOpcode, Rv64LoadStoreOpcode,
    Rv64Phantom, ShiftOpcode, ShiftWOpcode,
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
        BaseAluImmOpcode::ADDI.global_opcode(),
        [reg(rd), reg(rs1), imm, 1, 0],
    )
}

fn alu_r(opcode: BaseAluOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 1])
}

fn alu_imm(opcode: BaseAluOpcode, rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), imm, 1, 0])
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

fn less_than_imm(opcode: LessThanOpcode, rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), imm, 1, 0])
}

fn alu_w(opcode: BaseAluWOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 1])
}

fn alu_w_imm(opcode: BaseAluWOpcode, rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), imm, 1, 0])
}

fn shift(opcode: ShiftOpcode, rd: usize, rs1: usize, shamt: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), shamt, 1, 0])
}

fn shift_reg(opcode: ShiftOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 1])
}

fn shift_w(opcode: ShiftWOpcode, rd: usize, rs1: usize, shamt: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), shamt, 1, 0])
}

fn shift_w_reg(opcode: ShiftWOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(opcode.global_opcode(), [reg(rd), reg(rs1), reg(rs2), 1, 1])
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
    store_as(opcode, rs2, rs1, offset, RV64_MEMORY_AS as usize)
}

fn store_as(
    opcode: Rv64LoadStoreOpcode,
    rs2: usize,
    rs1: usize,
    offset: usize,
    address_space: usize,
) -> Instruction<F> {
    Instruction::from_usize(
        opcode.global_opcode(),
        [reg(rs2), reg(rs1), offset, 1, address_space, 1, 0],
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
    extension_store_width(Rv64LoadStoreOpcode::STORED, src, ptr, offset)
}

fn extension_store_width(
    opcode: Rv64LoadStoreOpcode,
    src: usize,
    ptr: usize,
    offset: usize,
) -> Instruction<F> {
    Instruction::from_usize(
        opcode.global_opcode(),
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

/// The AddSub-heavy fixture for the inline-record differential: reg-reg and
/// reg-imm ADD/SUB with register reuse (non-trivial prev_timestamps), x0 as a
/// source, and a negative immediate (sign path). rd is never x0 (that lifts to
/// a nop). One non-migrated instruction (SLTU) keeps the mixed-mode assembler
/// path live in the same segment.
fn inline_addsub_differential_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 5),                      // x1 = 5      (imm, rs1 = x0)
        addi(2, 0, 100),                    // x2 = 100
        alu_r(BaseAluOpcode::ADD, 3, 1, 2), // x3 = 105    (reg-reg)
        alu_r(BaseAluOpcode::SUB, 4, 2, 1), // x4 = 95
        addi(5, 3, 0x00ff_ffff),            // x5 = x3 + (-1) = 104 (negative imm)
        alu_r(BaseAluOpcode::ADD, 1, 4, 5), // x1 = 199    (rewrites x1 → prev_ts chain)
        alu_r(BaseAluOpcode::SUB, 6, 1, 0), // x6 = x1 - x0 (rs2 = x0)
        // Family 2: LessThan/Shift/Bitwise share the compact alu3 record.
        sltu(7, 2, 1),
        less_than(LessThanOpcode::SLT, 17, 1, 2),
        alu_r(BaseAluOpcode::XOR, 18, 1, 2),
        alu_r(BaseAluOpcode::OR, 19, 1, 2),
        alu_r(BaseAluOpcode::AND, 20, 1, 2),
        shift(ShiftOpcode::SLL, 21, 1, 3), // shamt-immediate form
        shift(ShiftOpcode::SRL, 22, 1, 1),
        shift(ShiftOpcode::SRA, 23, 5, 2), // arithmetic core, negative-ish value
        alu_r(BaseAluOpcode::ADD, 24, 21, 22), // consumes shift results
        // Family 3: branches (branch2), JAL/LUI (wr1, incl. the suppressed
        // x0 link write), AUIPC (wr1), JALR (rw1). Forward targets only; the
        // loop-fixture differential covers the taken backward branch.
        beq(2, 2, 8),                                  // taken: skips the dead slot
        addi(27, 0, 1),                                // dead code (skipped)
        branch_eq(BranchEqualOpcode::BNE, 2, 2, 8),    // untaken: falls through
        branch_lt(BranchLessThanOpcode::BLT, 1, 2, 8), // untaken (199 < 100 is false)
        jal_lui(Rv64JalLuiOpcode::JAL, 28, 8),         // link x28, skip dead slot
        addi(27, 0, 2),                                // dead code (skipped)
        jal_lui(Rv64JalLuiOpcode::JAL, 0, 8),          // jal x0: suppressed link write
        addi(27, 0, 3),                                // dead code (skipped)
        jal_lui(Rv64JalLuiOpcode::LUI, 29, 5),
        auipc(30, 0),    // x30 = this instruction's pc
        jalr(31, 30, 8), // jump to x30 + 8 = the next slot, link x31
        // Family 4: loads/stores are now migrated (alu3 wire; block value +
        // rs1 value + prev data), including sub-word widths, sign extension,
        // and the suppressed rd = x0 write. Phantoms keep the unmigrated
        // assembler path alive in the same segment (mixed mode).
        addi(26, 0, 64), // aligned memory base
        store(Rv64LoadStoreOpcode::STORED, 2, 26, 0),
        load(Rv64LoadStoreOpcode::LOADD, 25, 26, 0),
        store(Rv64LoadStoreOpcode::STOREB, 4, 26, 1),
        load(Rv64LoadStoreOpcode::LOADBU, 25, 26, 1),
        store(Rv64LoadStoreOpcode::STOREH, 4, 26, 2),
        load(Rv64LoadStoreOpcode::LOADHU, 25, 26, 2),
        store(Rv64LoadStoreOpcode::STOREW, 4, 26, 4),
        load(Rv64LoadStoreOpcode::LOADWU, 25, 26, 4),
        load(Rv64LoadStoreOpcode::LOADB, 25, 26, 1),
        load(Rv64LoadStoreOpcode::LOADH, 25, 26, 2),
        load(Rv64LoadStoreOpcode::LOADW, 25, 26, 4),
        load(Rv64LoadStoreOpcode::LOADBU, 0, 26, 1), // rd = x0: suppressed write, tick kept
        phantom(SysPhantom::Nop),
        // Phase-4 family 1: the Mul shapes share the compact alu3 record.
        mul(8, 1, 2),                             // x8 = x1 * x2
        mulh(MulHOpcode::MULH, 9, 1, 2),          // signed high word
        mulh(MulHOpcode::MULHU, 10, 1, 2),        // unsigned high word
        mul_w(11, 1, 2),                          // 32-bit product, sign-extended
        divrem(DivRemOpcode::DIV, 12, 1, 2),      // signed division
        divrem(DivRemOpcode::REM, 13, 1, 0),      // remainder by zero (rs2 = x0)
        divrem(DivRemOpcode::DIVU, 14, 1, 0),     // division by zero
        divrem_w(DivRemWOpcode::DIVW, 15, 1, 2),  // W division
        divrem_w(DivRemWOpcode::REMUW, 16, 1, 0), // W remainder by zero
        terminate(),
    ])
}

/// Main-memory portion of the two-block fixture. Kept separate so CUDA/G2
/// coverage is not whole-AIR-tainted by the deliberate AS=3 fallback rows.
fn two_block_main_memory_instructions() -> Vec<Instruction<F>> {
    vec![
        addi(1, 0, 2044),
        addi(1, 1, 2044), // x1 = 4088, the final block of a 4-KiB page
        addi(2, 0, 0x5a5),
        // Byte accesses are always single-block, including the last page byte.
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 1),
        load(Rv64LoadStoreOpcode::LOADBU, 3, 1, 1),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 7),
        load(Rv64LoadStoreOpcode::LOADB, 3, 1, 7),
        // Unaligned, non-crossing multi-byte accesses.
        store(Rv64LoadStoreOpcode::STOREH, 2, 1, 1),
        load(Rv64LoadStoreOpcode::LOADHU, 3, 1, 1),
        load(Rv64LoadStoreOpcode::LOADH, 3, 1, 1),
        store(Rv64LoadStoreOpcode::STOREW, 2, 1, 1),
        load(Rv64LoadStoreOpcode::LOADWU, 3, 1, 1),
        load(Rv64LoadStoreOpcode::LOADW, 3, 1, 1),
        store(Rv64LoadStoreOpcode::STORED, 2, 1, 0),
        load(Rv64LoadStoreOpcode::LOADD, 3, 1, 0),
        // Each access below crosses both an 8-byte block and the 4-KiB page.
        store(Rv64LoadStoreOpcode::STOREH, 2, 1, 7),
        load(Rv64LoadStoreOpcode::LOADHU, 3, 1, 7),
        load(Rv64LoadStoreOpcode::LOADH, 3, 1, 7),
        store(Rv64LoadStoreOpcode::STOREW, 2, 1, 6),
        load(Rv64LoadStoreOpcode::LOADWU, 3, 1, 6),
        load(Rv64LoadStoreOpcode::LOADW, 3, 1, 6),
        store(Rv64LoadStoreOpcode::STORED, 2, 1, 4),
        load(Rv64LoadStoreOpcode::LOADD, 3, 1, 4),
    ]
}

fn two_block_main_memory_exe() -> VmExe<F> {
    let mut instructions = two_block_main_memory_instructions();
    instructions.push(terminate());
    exe(&instructions)
}

/// Two-block adaptation fixture. Main-memory accesses cover each byte width,
/// unaligned non-crossing subword accesses, every possible crossing width,
/// and crossings whose second block starts on the next 4-KiB page. Public
/// values exercise AS=3 stores on both sides of an 8-byte block boundary.
fn two_block_loadstore_exe() -> VmExe<F> {
    let mut instructions = two_block_main_memory_instructions();
    instructions.extend([
        // AS=3 is store-only upstream. Narrow stores use the verbose helper;
        // STORED shares the compact/residual route with ordinary stores.
        addi(4, 0, 0),
        store_as(
            Rv64LoadStoreOpcode::STOREB,
            2,
            4,
            0,
            PUBLIC_VALUES_AS as usize,
        ),
        store_as(
            Rv64LoadStoreOpcode::STOREH,
            2,
            4,
            1,
            PUBLIC_VALUES_AS as usize,
        ),
        store_as(
            Rv64LoadStoreOpcode::STOREH,
            2,
            4,
            7,
            PUBLIC_VALUES_AS as usize,
        ),
        store_as(
            Rv64LoadStoreOpcode::STOREW,
            2,
            4,
            2,
            PUBLIC_VALUES_AS as usize,
        ),
        store_as(
            Rv64LoadStoreOpcode::STOREW,
            2,
            4,
            7,
            PUBLIC_VALUES_AS as usize,
        ),
        store_as(
            Rv64LoadStoreOpcode::STORED,
            2,
            4,
            0,
            PUBLIC_VALUES_AS as usize,
        ),
        store_as(
            Rv64LoadStoreOpcode::STORED,
            2,
            4,
            7,
            PUBLIC_VALUES_AS as usize,
        ),
        terminate(),
    ]);
    exe(&instructions)
}

/// Execute the fixture through the routed rvr preflight and assemble all
/// record arenas via the standard generation path (which adopts inline C
/// buffers for migrated chips and runs the log assembler for the rest).
fn run_inline_addsub_differential_arm(
    exe: &VmExe<F>,
) -> (
    RvrPreflightOutput<F>,
    Vec<openvm_circuit::arch::DenseRecordArena>,
    Vec<Option<usize>>,
) {
    use openvm_circuit::arch::DenseRecordArena;

    // This harness drives the COMPACT wire + host assembler without arena
    // targets, so it opts out of the (default-on) fused emission. Fused
    // arms re-enable it explicitly afterwards; nextest isolates processes.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("rvr vm init");
    let trace_heights = vec![4096u32; rvr_vm.num_airs()];
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(exe).expect("pc to air mapping");

    let mut rvr_output = {
        let route = rvr_vm
            .preflight_routed_instance(exe)
            .expect("routed preflight instance");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("program must route to RVR preflight");
        };
        instance
            .execute_preflight(Streams::default(), None)
            .expect("rvr preflight execution")
    };
    let arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<F, DenseRecordArena>(
        exe,
        &mut rvr_output,
        &capacities,
        &pc_to_air_idx,
    )
    .expect("record arena generation");
    (rvr_output, arenas, pc_to_air_idx)
}

type TwoBlockArm = (
    RvrPreflightOutput<F>,
    Vec<DenseRecordArena>,
    Vec<Option<usize>>,
    BTreeMap<usize, usize>,
);

fn run_two_block_loadstore_arm(exe: &VmExe<F>) -> TwoBlockArm {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::with_public_values_bytes(32),
    )
    .expect("two-block vm init");
    let trace_heights = vec![4096u32; rvr_vm.num_airs()];
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(exe).expect("pc to air mapping");
    let mut output = {
        let RvrPreflightRoute::Rvr(instance) = rvr_vm
            .preflight_routed_instance(exe)
            .expect("routed two-block preflight instance")
        else {
            panic!("two-block program must route to RVR preflight");
        };
        instance
            .execute_preflight(Streams::default(), None)
            .expect("two-block rvr execution")
    };
    let compact_rows = output
        .inline_records
        .iter()
        .map(|chip| {
            assert_eq!(chip.bytes.len() % chip.record_size, 0);
            (chip.air_idx, chip.bytes.len() / chip.record_size)
        })
        .collect();
    let arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<F, DenseRecordArena>(
        exe,
        &mut output,
        &capacities,
        &pc_to_air_idx,
    )
    .expect("two-block record assembly");
    (output, arenas, pc_to_air_idx, compact_rows)
}

/// R3 Phase-1 differential for the C-consumed record path: the same program is
/// compiled twice — inline records ON (default: AddSub memory-log entries
/// suppressed, host adopts the C-written record buffer) and OFF
/// (`OPENVM_RVR_INLINE_RECORDS=0`: verbose log + host assembler for every
/// opcode). The two runs must agree on every record arena byte-for-byte and on
/// the full tick model (program-log timestamps, system records), and the ON
/// run's memory log must be exactly the OFF log minus the suppressed AddSub
/// entries. nextest runs each test in its own process, so the env mutation is
/// isolated.
#[test]
fn rvr_preflight_inline_addsub_records_match_assembler() {
    let exe = inline_addsub_differential_exe();

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    let (off_output, off_arenas, pc_to_air_idx) = run_inline_addsub_differential_arm(&exe);
    assert!(
        off_output.inline_records.is_empty(),
        "opt-out compile must not produce inline record buffers"
    );

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    let (on_output, on_arenas, _) = run_inline_addsub_differential_arm(&exe);
    let addsub_air_idx = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .find_map(|(i, slot)| {
            slot.as_ref().and_then(|(insn, _)| {
                (insn.opcode == BaseAluOpcode::ADD.global_opcode()
                    || insn.opcode == BaseAluOpcode::SUB.global_opcode())
                .then(|| pc_to_air_idx[i].expect("ADD/SUB must map to an air"))
            })
        })
        .expect("program must contain an ADD/SUB");
    assert!(
        !on_arenas[addsub_air_idx].allocated().is_empty(),
        "inline run must materialize AddSub records (was the inline compile honored?)"
    );

    // Ticking is untouchable: identical program log (pcs and timestamps) and
    // identical system records (to_state timestamp, touched_memory, exec
    // frequencies) between the suppressed and verbose compiles.
    assert_eq!(
        off_output.raw_logs.program_log, on_output.raw_logs.program_log,
        "log suppression must not change the program log or its timestamps"
    );
    assert_system_records_eq(
        "inline_addsub_differential",
        &off_output.system_records,
        &on_output.system_records,
    );

    // The ON memory log is exactly the OFF log minus the suppressed inline
    // entries. AddI has a two-access window; the remaining alu3 families use
    // three accesses.
    let inline_windows: Vec<(u32, u32)> = off_output
        .raw_logs
        .program_log
        .iter()
        .filter(|entry| {
            let slot = ((entry.pc() - exe.program.pc_base) / DEFAULT_PC_STEP) as usize;
            on_output
                .inline_pc_slots
                .get(slot)
                .copied()
                .unwrap_or(false)
        })
        .map(|entry| {
            let slot = ((entry.pc() - exe.program.pc_base) / DEFAULT_PC_STEP) as usize;
            let (instruction, _) = exe.program.instructions_and_debug_infos[slot]
                .as_ref()
                .expect("executed pc must name an instruction");
            let access_count = if instruction.opcode == BaseAluImmOpcode::ADDI.global_opcode() {
                2
            } else if is_loadstore_opcode(instruction.opcode)
                && instruction.opcode != Rv64LoadStoreOpcode::LOADBU.global_opcode()
                && instruction.opcode != Rv64LoadStoreOpcode::LOADB.global_opcode()
                && instruction.opcode != Rv64LoadStoreOpcode::STOREB.global_opcode()
            {
                4
            } else {
                3
            };
            (entry.timestamp, entry.timestamp + access_count)
        })
        .collect();
    assert!(
        !inline_windows.is_empty(),
        "fixture must contain migrated inline instructions"
    );
    let off_log_minus_addsub: Vec<_> = off_output
        .raw_logs
        .memory_log
        .iter()
        .filter(|entry| {
            !inline_windows
                .iter()
                .any(|&(start, end)| entry.timestamp >= start && entry.timestamp < end)
        })
        .cloned()
        .collect();
    assert_eq!(
        on_output.raw_logs.memory_log, off_log_minus_addsub,
        "suppression must drop exactly the AddSub memory-log entries"
    );

    // Byte-identical record arenas across every AIR: the adopted C buffer for
    // AddSub and the untouched assembler output for everything else.
    assert_eq!(off_arenas.len(), on_arenas.len());
    for (air_idx, (off_arena, on_arena)) in off_arenas.iter().zip(on_arenas.iter()).enumerate() {
        assert_eq!(
            off_arena.allocated(),
            on_arena.allocated(),
            "record arena bytes differ for air_idx {air_idx}"
        );
    }
}

/// Stage-2 byte oracle over every migrated RV64IM wire family. The delta arm
/// reconstructs its omitted previous timestamps and partitions the global
/// stream; the resulting Dense arenas must match the established compact arm
/// byte-for-byte for every AIR.
#[test]
fn rvr_preflight_delta_full_rv64im_matrix_is_byte_equal_to_compact() {
    let exe = full_rv64im_matrix_exe();

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let (compact_output, compact_arenas, _) = run_inline_addsub_differential_arm(&exe);

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let (delta_output, delta_arenas, _) = run_inline_addsub_differential_arm(&exe);
    let delta_stream = delta_output
        .delta_records
        .as_ref()
        .expect("delta output must retain its recyclable backing");
    assert_eq!(
        delta_stream.bytes().as_ptr() as usize % 32,
        0,
        "generated C delta target must be 32-byte aligned"
    );
    assert_system_records_eq(
        "delta_full_matrix",
        &compact_output.system_records,
        &delta_output.system_records,
    );
    assert_eq!(
        compact_output.raw_logs.program_log, delta_output.raw_logs.program_log,
        "delta decoder must reconstruct the complete chronological program log"
    );
    assert_eq!(compact_arenas.len(), delta_arenas.len());
    for (air, (compact, delta)) in compact_arenas.iter().zip(delta_arenas.iter()).enumerate() {
        assert_eq!(
            compact.allocated(),
            delta.allocated(),
            "delta-decoded arena differs from compact for AIR {air}"
        );
    }
}

fn is_loadstore_opcode(opcode: openvm_instructions::VmOpcode) -> bool {
    [
        Rv64LoadStoreOpcode::LOADD,
        Rv64LoadStoreOpcode::LOADBU,
        Rv64LoadStoreOpcode::LOADHU,
        Rv64LoadStoreOpcode::LOADWU,
        Rv64LoadStoreOpcode::STORED,
        Rv64LoadStoreOpcode::STOREB,
        Rv64LoadStoreOpcode::STOREH,
        Rv64LoadStoreOpcode::STOREW,
        Rv64LoadStoreOpcode::LOADB,
        Rv64LoadStoreOpcode::LOADH,
        Rv64LoadStoreOpcode::LOADW,
    ]
    .into_iter()
    .any(|candidate| candidate.global_opcode() == opcode)
}

fn is_load_opcode(opcode: openvm_instructions::VmOpcode) -> bool {
    [
        Rv64LoadStoreOpcode::LOADD,
        Rv64LoadStoreOpcode::LOADBU,
        Rv64LoadStoreOpcode::LOADHU,
        Rv64LoadStoreOpcode::LOADWU,
        Rv64LoadStoreOpcode::LOADB,
        Rv64LoadStoreOpcode::LOADH,
        Rv64LoadStoreOpcode::LOADW,
    ]
    .into_iter()
    .any(|candidate| candidate.global_opcode() == opcode)
}

/// Byte oracle for every decoded consumer: the compact CPU assembler and the
/// chronological delta decoder must produce exactly the same upstream-native
/// load/store records as the verbose memory-log route. The explicit row
/// accounting pins `fast non-crossing + residual crossing == metered` per AIR.
#[test]
fn rvr_preflight_two_block_compact_and_delta_match_verbose_records() {
    let exe = two_block_loadstore_exe();

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (verbose, verbose_arenas, _, _) = run_two_block_loadstore_arm(&exe);

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let (compact, compact_arenas, pc_to_air_idx, compact_rows) = run_two_block_loadstore_arm(&exe);

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let (delta, delta_arenas, _, _) = run_two_block_loadstore_arm(&exe);

    assert_system_records_eq(
        "two_block_verbose_compact",
        &verbose.system_records,
        &compact.system_records,
    );
    assert_system_records_eq(
        "two_block_verbose_delta",
        &verbose.system_records,
        &delta.system_records,
    );
    assert_eq!(verbose_arenas.len(), compact_arenas.len());
    assert_eq!(verbose_arenas.len(), delta_arenas.len());
    for air in 0..verbose_arenas.len() {
        assert_eq!(
            verbose_arenas[air].allocated(),
            compact_arenas[air].allocated(),
            "compact two-block records differ from verbose for AIR {air}"
        );
        assert_eq!(
            verbose_arenas[air].allocated(),
            delta_arenas[air].allocated(),
            "delta two-block records differ from verbose for AIR {air}"
        );
    }

    let slot_for_pc = |pc: u32| {
        assert!(pc >= exe.program.pc_base);
        ((pc - exe.program.pc_base) / DEFAULT_PC_STEP) as usize
    };
    let mut metered_by_air = BTreeMap::<usize, usize>::new();
    let mut inline_by_air = BTreeMap::<usize, usize>::new();
    let mut crossing_by_air = BTreeMap::<usize, usize>::new();
    for entry in &compact.raw_logs.program_log {
        let slot = slot_for_pc(entry.pc());
        let (instruction, _) = exe.program.instructions_and_debug_infos[slot]
            .as_ref()
            .expect("executed pc must name an instruction");
        if !is_loadstore_opcode(instruction.opcode) {
            continue;
        }
        let air = pc_to_air_idx[slot].expect("load/store pc must map to an AIR");
        *metered_by_air.entry(air).or_default() += 1;
        if compact.inline_pc_slots[slot] {
            *inline_by_air.entry(air).or_default() += 1;
        }
        if entry.crossing_residual() {
            *crossing_by_air.entry(air).or_default() += 1;
            let is_load = is_load_opcode(instruction.opcode);
            let first_timestamp = entry.timestamp + if is_load { 1 } else { 2 };
            let kind = u8::from(!is_load);
            let events = compact
                .raw_logs
                .memory_log
                .iter()
                .filter(|event| {
                    event.kind == kind
                        && (event.timestamp == first_timestamp
                            || event.timestamp == first_timestamp + 1)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                events.len(),
                2,
                "crossing pc {:#x} must retain exactly two residual blocks",
                entry.pc()
            );
            assert_eq!(events[0].width, 8);
            assert_eq!(events[1].width, 8);
            assert_eq!(events[1].address, events[0].address + 8);
            assert_eq!(events[0].addr_space, events[1].addr_space);
        }
    }
    assert_eq!(
        crossing_by_air.values().sum::<usize>(),
        9,
        "fixture crossing marker count drifted"
    );
    for (&air, &inline_rows) in &inline_by_air {
        let wire_rows = compact_rows.get(&air).copied().unwrap_or(0);
        let residual_rows = crossing_by_air.get(&air).copied().unwrap_or(0);
        assert_eq!(wire_rows, inline_rows, "AIR {air}: compact row count");
        assert!(wire_rows >= residual_rows, "AIR {air}: residual overcount");
        let fast_rows = wire_rows - residual_rows;
        assert_eq!(
            fast_rows + residual_rows,
            inline_rows,
            "AIR {air}: fast + residual must equal metered inline rows"
        );
    }
    for (&air, &metered_rows) in &metered_by_air {
        let inline_rows = inline_by_air.get(&air).copied().unwrap_or(0);
        let crossing_rows = crossing_by_air.get(&air).copied().unwrap_or(0);
        let fast_rows = inline_rows - crossing_rows;
        let verbose_fallback_rows = metered_rows - inline_rows;
        let residual_sourced_rows = crossing_rows + verbose_fallback_rows;
        assert_eq!(
            fast_rows + residual_sourced_rows,
            metered_rows,
            "AIR {air}: fast + residual-sourced rows must equal metered rows"
        );
        let bytes = verbose_arenas[air].allocated().len();
        assert_eq!(bytes % metered_rows, 0, "AIR {air}: partial record row");
        let stride = bytes / metered_rows;
        assert!(
            matches!(stride, 48 | 60),
            "AIR {air}: record stride {stride}"
        );
        assert_eq!(
            compact_arenas[air].allocated().len() / stride,
            metered_rows,
            "AIR {air}: compact decoded row count"
        );
        assert_eq!(
            delta_arenas[air].allocated().len() / stride,
            metered_rows,
            "AIR {air}: delta decoded row count"
        );
    }
}

/// The AS=2-only fixture keeps every width in the device-candidate set. Its
/// CPU compact/delta decoders remain byte oracles for the CUDA/G2 consumers.
#[test]
fn rvr_preflight_two_block_device_candidate_decoders_match_verbose_records() {
    let exe = two_block_main_memory_exe();

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (verbose, verbose_arenas, _, _) = run_two_block_loadstore_arm(&exe);

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let (compact, compact_arenas, pc_to_air_idx, _) = run_two_block_loadstore_arm(&exe);

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let (delta, delta_arenas, _, _) = run_two_block_loadstore_arm(&exe);

    assert_system_records_eq(
        "two_block_device_verbose_compact",
        &verbose.system_records,
        &compact.system_records,
    );
    assert_system_records_eq(
        "two_block_device_verbose_delta",
        &verbose.system_records,
        &delta.system_records,
    );
    for air in 0..verbose_arenas.len() {
        assert_eq!(
            verbose_arenas[air].allocated(),
            compact_arenas[air].allocated(),
            "compact device-candidate records differ for AIR {air}"
        );
        assert_eq!(
            verbose_arenas[air].allocated(),
            delta_arenas[air].allocated(),
            "delta device-candidate records differ for AIR {air}"
        );
    }

    let delta_airs = delta
        .delta_decode_precomputed
        .as_ref()
        .expect("delta fixture must carry decode metadata")
        .kind_to_air
        .iter()
        .map(|&(_, air)| air)
        .collect::<std::collections::BTreeSet<_>>();
    for (slot, entry) in exe.program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = entry else {
            continue;
        };
        if is_loadstore_opcode(instruction.opcode) {
            let air = pc_to_air_idx[slot].expect("load/store pc must map to an AIR");
            assert!(
                delta_airs.contains(&air),
                "AS=2 load/store AIR {air} must remain delta-device-decodable"
            );
        }
    }
}

/// Byte oracle for the arena-native consumer. This stages every eligible AIR
/// directly into its upstream record geometry and compares the written prefix
/// with the verbose assembler, including both crossing and non-crossing rows.
#[test]
fn rvr_preflight_two_block_arena_native_matches_verbose_records() {
    use openvm_circuit::arch::rvr::{ArenaNativeLayout, ChipRecordBuf};

    let exe = two_block_loadstore_exe();
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (oracle_output, oracle_arenas, _, _) = run_two_block_loadstore_arm(&exe);

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    let config = Rv64ImConfig::with_public_values_bytes(32);
    let (vm, _) = VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config)
        .expect("two-block arena-native vm init");
    let heights = vec![4096u32; vm.num_airs()];
    let capacities = heights
        .iter()
        .zip(vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("two-block arena-native route")
    else {
        panic!("two-block program must route to RVR preflight");
    };
    let arena_native = instance
        .compiled()
        .inline_records()
        .arena_native_airs
        .clone();
    assert!(!arena_native.is_empty());

    let mut saw_byte_geometry = false;
    let mut saw_multi_geometry = false;
    let mut stagings = Vec::new();
    let mut targets = BTreeMap::new();
    for &(air, geometry) in &arena_native {
        if let ArenaNativeLayout::LoadStore(offsets) = geometry.layout {
            let is_multi = offsets.read_data_aux_prev_ts2 != usize::MAX
                || offsets.write_prev_ts2 != usize::MAX;
            if is_multi {
                saw_multi_geometry = true;
                assert_eq!(geometry.stride_dense(), 60);
            } else {
                saw_byte_geometry = true;
                assert_eq!(geometry.stride_dense(), 48);
            }
        }
        let stride = geometry.stride_dense();
        let (mut backing, offset) =
            DenseRecordArena::backing_with_capacity(heights[air] as usize * stride);
        let base = unsafe { backing.as_mut_ptr().add(offset) };
        targets.insert(
            air,
            ChipRecordBuf {
                base,
                len: 0,
                cap: (heights[air] as usize * stride) as u32,
                stride: stride as u32,
                core_off: geometry.core_off_dense() as u32,
                flags: openvm_circuit::arch::rvr::PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL,
            },
        );
        stagings.push((air, geometry, stride, backing, base));
    }
    assert!(
        saw_byte_geometry,
        "fixture must stage byte load/store records"
    );
    assert!(
        saw_multi_geometry,
        "fixture must stage multi-byte load/store records"
    );

    let state = vm.create_initial_state(&exe, Streams::default());
    let mut direct_output = instance
        .execute_preflight_from_state_with_arena_targets(
            state,
            Some(oracle_output.instret),
            &heights,
            &targets,
        )
        .expect("two-block arena-native execution");
    let direct_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        DenseRecordArena,
    >(&exe, &mut direct_output, &capacities, &pc_to_air_idx)
    .expect("two-block arena-native residual assembly");
    assert_system_records_eq(
        "two_block_verbose_arena_native",
        &oracle_output.system_records,
        &direct_output.system_records,
    );

    let direct_air_set = stagings
        .iter()
        .map(|&(air, ..)| air)
        .collect::<std::collections::BTreeSet<_>>();
    for (air, geometry, stride, backing, base) in stagings {
        let written = direct_output
            .arena_native_written
            .iter()
            .find(|&&(written_air, _)| written_air == air)
            .map(|&(_, rows)| rows as usize)
            .expect("arena-native AIR must report a written count");
        let direct = DenseRecordArena::from_prewritten(backing, base, written * stride);
        assert_eq!(
            oracle_arenas[air].allocated(),
            direct.allocated(),
            "AIR {air}: arena-native records differ from verbose assembly ({geometry:?})"
        );
    }
    for (air, (oracle, direct)) in oracle_arenas.iter().zip(&direct_arenas).enumerate() {
        if !direct_air_set.contains(&air) {
            assert_eq!(
                oracle.allocated(),
                direct.allocated(),
                "AIR {air}: residual-assembled records differ from verbose assembly"
            );
        }
    }
}

#[test]
fn rvr_preflight_two_block_records_prove_and_verify() {
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let segments = prove_rvr_preflight_and_verify(
        two_block_loadstore_exe(),
        Rv64ImConfig::with_public_values_bytes(32),
    );
    assert_eq!(segments, 1);
}

/// Phase-1 G2 gate: generated C writes only RUN_BLOCK_ID and AddI value
/// lanes. The oracle-only CPU decoder must reconstruct the exact established
/// 44-byte compact consumer records, including every initialized/padding
/// byte, while the production-shaped preflight performs no host record pass.
#[test]
fn rvr_preflight_g2_addi_is_byte_equal_to_compact_consumer() {
    let exe = exe(&[
        addi(1, 0, 7),
        addi(2, 1, 5),
        addi(1, 2, 0x00ff_ffff),
        terminate(),
    ]);
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("G2 oracle VM init");
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("G2 pc mapping");
    let addi_air_idx = pc_to_air_idx[0].expect("AddI must map to an AIR");
    assert_eq!(pc_to_air_idx[1], Some(addi_air_idx));
    assert_eq!(pc_to_air_idx[2], Some(addi_air_idx));
    let mut exact_rows = vec![0u32; rvr_vm.num_airs()];
    exact_rows[addi_air_idx] = 3;

    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let RvrPreflightRoute::Rvr(compact) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("compact routed preflight")
    else {
        panic!("AddI program must route to RVR");
    };
    let mut compact_output = compact
        .execute_preflight_from_state_with_capacities(
            compact.create_initial_state(Streams::default()),
            Some(4),
            &exact_rows,
        )
        .expect("compact AddI preflight");
    let compact_bytes = compact_output
        .inline_records
        .iter()
        .find(|records| records.air_idx == addi_air_idx)
        .expect("compact AddI consumer records");
    assert_eq!(compact_bytes.record_size, 44);
    assert_eq!(compact_bytes.bytes.len(), 3 * 44);
    let compact_consumer_bytes = compact_bytes.bytes.clone();
    let compact_program_log = compact_output.raw_logs.program_log.clone();
    let capacities = exact_rows
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let compact_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        DenseRecordArena,
    >(&exe, &mut compact_output, &capacities, &pc_to_air_idx)
    .expect("compact final-arena oracle");

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let RvrPreflightRoute::Rvr(g2) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("G2 routed preflight")
    else {
        panic!("AddI program must route to G2 RVR");
    };
    assert!(g2.compiled().inline_records().g2.is_some());
    let mut g2_output = g2
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            g2.create_initial_state(Streams::default()),
            Some(4),
            &exact_rows,
            &BTreeMap::new(),
        )
        .expect("G2 AddI preflight");
    let segment = g2_output.g2_segment.take().expect("committed G2 segment");
    let meta = g2_output.g2_meta.as_deref().expect("G2 schema metadata");
    let decode = g2_output
        .delta_decode_precomputed
        .as_deref()
        .expect("G2 static operand table");
    let reference = openvm_circuit::arch::rvr::decode_addi_reference_v1(
        &segment,
        meta,
        decode,
        [0; 32],
        openvm_circuit::arch::rvr::PREFLIGHT_INITIAL_TIMESTAMP,
    )
    .expect("G2 CPU reference decode");

    assert_eq!(
        segment.header_acquire().unwrap().residual_event_count,
        0,
        "standard G2 rows must advance the predecessor shadow without duplicate residual events"
    );
    assert_eq!(reference.compact_records, compact_consumer_bytes);
    assert_eq!(reference.expanded_program_slots, [0, 1, 2, 3]);
    assert_eq!(reference.final_registers[1], 11);
    assert_eq!(reference.final_registers[2], 12);
    assert_eq!(
        reference.final_timestamp,
        openvm_circuit::arch::rvr::PREFLIGHT_INITIAL_TIMESTAMP + 6
    );
    assert!(g2_output.raw_logs.program_log.is_empty());
    assert!(g2_output.raw_logs.memory_log.is_empty());
    assert!(g2_output.raw_logs.delta_memory_log.is_empty());
    assert!(g2_output.access_aux.is_empty());
    assert!(g2_output.inline_records.is_empty());
    assert!(g2_output.delta_records.is_none());
    assert!(!g2_output.access_aux_complete);
    assert!(g2_output.system_records.program_frequencies_on_device);
    assert!(g2_output.system_records.touched_memory_on_device);
    assert_eq!(
        g2_output.system_records.device_replay_oracle,
        meta.checked_emission(),
        "checked G2 must enable the system touched-memory oracle without a second environment flag"
    );

    // Oracle-only final materialization. Production never executes this
    // block: it restores the compact arm's chronology solely to drive the
    // established CPU assembler and compare the complete final arena bytes.
    g2_output.raw_logs.program_log = compact_program_log;
    g2_output.inline_records = vec![openvm_circuit::arch::rvr::RvrInlineChipRecords {
        air_idx: addi_air_idx,
        record_size: 44,
        bytes: reference.compact_records,
    }];
    g2_output.access_aux_complete = true;
    let g2_oracle_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        DenseRecordArena,
    >(&exe, &mut g2_output, &capacities, &pc_to_air_idx)
    .expect("G2 oracle final-arena materialization");
    assert_eq!(compact_arenas.len(), g2_oracle_arenas.len());
    for (air, (compact, g2)) in compact_arenas.iter().zip(&g2_oracle_arenas).enumerate() {
        assert_eq!(
            compact.allocated(),
            g2.allocated(),
            "G2 final arena, including record padding, differs for AIR {air}"
        );
    }
}

#[test]
fn rvr_preflight_g2_phase2a_load_store_reveal_is_byte_equal() {
    let exe = g2_load_store_phase2a_exe();
    let config = Rv64ImConfig::with_public_values_bytes(32);
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config)
        .expect("G2 Phase-2a oracle VM init");
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("G2 Phase-2a pc mapping");
    let mut exact_rows = vec![0u32; rvr_vm.num_airs()];
    for air in pc_to_air_idx.iter().flatten() {
        exact_rows[*air] += 1;
    }
    let capacities = exact_rows
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let instruction_count = exe.program.num_defined_instructions() as u64;

    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let RvrPreflightRoute::Rvr(compact) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("compact Phase-2a routed preflight")
    else {
        panic!("Phase-2a program must route to RVR");
    };
    let mut compact_output = compact
        .execute_preflight_from_state_with_capacities(
            compact.create_initial_state(Streams::default()),
            Some(instruction_count),
            &exact_rows,
        )
        .expect("compact Phase-2a preflight");
    let compact_inline = compact_output
        .inline_records
        .iter()
        .map(|records| (records.air_idx, records.bytes.clone()))
        .collect::<BTreeMap<_, _>>();
    let compact_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        DenseRecordArena,
    >(&exe, &mut compact_output, &capacities, &pc_to_air_idx)
    .expect("compact Phase-2a final arenas");

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let RvrPreflightRoute::Rvr(g2) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("G2 Phase-2a routed preflight")
    else {
        panic!("Phase-2a program must route to G2 RVR");
    };
    let inline = g2.compiled().inline_records();
    assert!(inline.g2.is_some(), "Phase-2a executable must negotiate G2");
    assert!(pc_to_air_idx.iter().enumerate().all(|(slot, air)| {
        air.is_none() || inline.pc_slots.get(slot).copied().unwrap_or(false)
    }));
    let mut g2_output = g2
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            g2.create_initial_state(Streams::default()),
            Some(instruction_count),
            &exact_rows,
            &BTreeMap::new(),
        )
        .expect("G2 Phase-2a preflight");
    let segment = g2_output.g2_segment.take().expect("committed G2 segment");
    let meta = g2_output.g2_meta.as_deref().expect("G2 Phase-2a metadata");
    let decode = g2_output
        .delta_decode_precomputed
        .as_deref()
        .expect("G2 Phase-2a operand table");
    let mut initial_blocks = BTreeMap::new();
    for address in [0u32, 8, 16, 24] {
        initial_blocks.insert((PUBLIC_VALUES_AS as u8, address), 0);
    }
    let reference = openvm_circuit::arch::rvr::decode_reference_v1(
        &segment,
        meta,
        decode,
        [0; 32],
        &initial_blocks,
        openvm_circuit::arch::rvr::PREFLIGHT_INITIAL_TIMESTAMP,
    )
    .expect("G2 Phase-2a CPU reference decode");
    assert_eq!(
        segment.byte_len(),
        3_968,
        "Phase-2a fixture wire must contain only active lane prefixes plus frozen alignment"
    );
    assert_eq!(
        segment.header_acquire().unwrap().residual_event_count,
        0,
        "non-crossing load/store and REVEAL rows must stay entirely in fixed G2 lanes"
    );
    assert_eq!(
        reference.final_registers[0], 0,
        "x0 load must remain suppressed"
    );
    assert!(g2_output.raw_logs.program_log.is_empty());
    assert!(g2_output.raw_logs.memory_log.is_empty());
    assert!(g2_output.raw_logs.delta_memory_log.is_empty());
    assert!(g2_output.access_aux.is_empty());
    assert!(g2_output.inline_records.is_empty());
    assert!(!g2_output.access_aux_complete);

    for binding in meta.air_bindings.iter() {
        if let Some(bytes) = reference.compact_records.get(&binding.kind) {
            let native_multi = matches!(binding.kind, 20..=22 | 24..=28);
            let stride = if native_multi {
                60
            } else {
                openvm_circuit::arch::rvr::PREFLIGHT_ADDSUB_RECORD_SIZE
            };
            assert_eq!(
                bytes.len() / stride,
                exact_rows[binding.air_idx] as usize,
                "decoded rows must equal the metered AIR count for kind {}",
                binding.kind
            );
            if native_multi {
                assert_eq!(
                    bytes,
                    compact_arenas[binding.air_idx].allocated(),
                    "G2 load/store consumer bytes differ for kind {}",
                    binding.kind
                );
            } else if binding.kind == 29 {
                assert_eq!(
                    bytes,
                    compact_inline
                        .get(&binding.air_idx)
                        .expect("compact route must emit the same fixed consumer AIR"),
                    "G2 fixed consumer bytes differ for kind {}",
                    binding.kind
                );
            }
        }
    }
}

#[test]
fn rvr_preflight_g2_two_block_crossings_match_verbose_native_records() {
    let exe = two_block_main_memory_exe();

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (_, verbose_arenas, verbose_pc_to_air, _) = run_two_block_loadstore_arm(&exe);

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::with_public_values_bytes(32),
    )
    .expect("G2 two-block VM init");
    let pc_to_air = vm.pc_to_air_idx(&exe).expect("G2 two-block pc mapping");
    assert_eq!(pc_to_air, verbose_pc_to_air);
    let mut exact_rows = vec![0u32; vm.num_airs()];
    for air in pc_to_air.iter().flatten() {
        exact_rows[*air] += 1;
    }
    let retired = exe.program.num_defined_instructions() as u64;
    let RvrPreflightRoute::Rvr(g2) = vm
        .preflight_routed_instance(&exe)
        .expect("G2 two-block route")
    else {
        panic!("two-block executable must route to G2 RVR");
    };
    assert!(g2.compiled().inline_records().g2.is_some());
    let mut output = g2
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            g2.create_initial_state(Streams::default()),
            Some(retired),
            &exact_rows,
            &BTreeMap::new(),
        )
        .expect("G2 two-block preflight");
    let segment = output.g2_segment.take().expect("G2 two-block segment");
    let meta = output.g2_meta.as_deref().expect("G2 two-block metadata");
    let decode = output
        .delta_decode_precomputed
        .as_deref()
        .expect("G2 two-block operand table");
    assert_eq!(
        segment.header_acquire().unwrap().residual_event_count,
        16,
        "eight crossing accesses must contribute exactly two full-block residuals each"
    );
    let reference = openvm_circuit::arch::rvr::decode_reference_v1(
        &segment,
        meta,
        decode,
        [0; 32],
        &BTreeMap::new(),
        openvm_circuit::arch::rvr::PREFLIGHT_INITIAL_TIMESTAMP,
    )
    .expect("G2 two-block CPU reference decode");
    assert_eq!(
        reference.final_timestamp, output.system_records.to_state.timestamp,
        "G2 two-block replay timestamp"
    );
    assert_eq!(
        reference.final_registers,
        openvm_circuit::arch::rvr::bridge::read_rv64_registers(&output.to_state),
        "G2 two-block replay registers"
    );
    for binding in meta
        .air_bindings
        .iter()
        .filter(|binding| matches!(binding.kind, 20..=22 | 24..=28))
    {
        let bytes = reference
            .compact_records
            .get(&binding.kind)
            .expect("executed multi-block G2 kind");
        assert_eq!(bytes.len() / 60, exact_rows[binding.air_idx] as usize);
        assert_eq!(
            bytes,
            verbose_arenas[binding.air_idx].allocated(),
            "G2 two-block native records differ for kind {}",
            binding.kind
        );
    }
}

#[test]
fn rvr_preflight_g2_store_marks_sparse_continuation_page() {
    let store_exe = exe(&[
        addi(1, 0, 64),
        addi(2, 0, 0x5a),
        store(Rv64LoadStoreOpcode::STORED, 2, 1, 0),
        terminate(),
    ]);
    let (store_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("G2 continuation store VM init");
    let store_pc_to_air = store_vm
        .pc_to_air_idx(&store_exe)
        .expect("G2 continuation store pc mapping");
    let mut store_rows = vec![0u32; store_vm.num_airs()];
    for air in store_pc_to_air.iter().flatten() {
        store_rows[*air] += 1;
    }

    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let RvrPreflightRoute::Rvr(store) = store_vm
        .preflight_routed_instance(&store_exe)
        .expect("G2 continuation store route")
    else {
        panic!("continuation store program must route to G2 RVR");
    };
    let checked_emission = store
        .compiled()
        .inline_records()
        .g2
        .as_deref()
        .expect("G2 continuation metadata")
        .checked_emission();
    let store_output = store
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            store.create_initial_state(Streams::default()),
            Some(store_exe.program.num_defined_instructions() as u64),
            &store_rows,
            &BTreeMap::new(),
        )
        .expect("G2 continuation store preflight");
    let memory = &store_output.to_state.memory.memory;
    let memory_as = RV64_MEMORY_AS as usize;
    let dirty_ranges =
        memory.touched_pages[memory_as].touched_byte_ranges(memory.mem[memory_as].size());
    let host_marked = dirty_ranges
        .iter()
        .any(|&(start, end)| start <= 64 && 64 < end);
    if checked_emission {
        assert!(
            host_marked,
            "checked G2 host oracle must retain continuation page ownership"
        );
    } else {
        assert!(
            !host_marked,
            "production G2 host preflight must delegate continuation pages to device replay"
        );
    }

    // Run the following load as a distinct preflight segment from the carried
    // state. The page assertion above is the sparse-transport contract; this
    // second execution checks the corresponding store-to-load state boundary.
    let load_exe = exe(&[load(Rv64LoadStoreOpcode::LOADD, 3, 1, 0), terminate()]);
    let (load_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("G2 continuation load VM init");
    let load_pc_to_air = load_vm
        .pc_to_air_idx(&load_exe)
        .expect("G2 continuation load pc mapping");
    let mut load_rows = vec![0u32; load_vm.num_airs()];
    for air in load_pc_to_air.iter().flatten() {
        load_rows[*air] += 1;
    }
    let RvrPreflightRoute::Rvr(load) = load_vm
        .preflight_routed_instance(&load_exe)
        .expect("G2 continuation load route")
    else {
        panic!("continuation load program must route to G2 RVR");
    };
    let mut carried_state = store_output.to_state;
    carried_state.set_pc(load_exe.pc_start);
    let load_output = load
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            carried_state,
            Some(load_exe.program.num_defined_instructions() as u64),
            &load_rows,
            &BTreeMap::new(),
        )
        .expect("G2 continuation load preflight");
    let registers = openvm_circuit::arch::rvr::bridge::read_rv64_registers(&load_output.to_state);
    assert_eq!(registers[3], 0x5a, "next segment must observe the G2 store");
}

#[test]
fn rvr_preflight_g2_rejects_high_pointer_before_memory_or_reveal_access() {
    let exe = exe(&[
        load(Rv64LoadStoreOpcode::LOADD, 3, 1, 0),
        extension_store(2, 1, 0),
        extension_store_width(Rv64LoadStoreOpcode::STOREW, 2, 1, 8),
        terminate(),
    ]);
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::with_public_values_bytes(32),
    )
    .expect("G2 high-pointer VM init");
    let pc_to_air = vm.pc_to_air_idx(&exe).expect("G2 high-pointer pc mapping");
    let mut rows = vec![0u32; vm.num_airs()];
    for air in pc_to_air.iter().flatten() {
        rows[*air] += 1;
    }

    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("G2 high-pointer route")
    else {
        panic!("high-pointer fixture must route to G2 RVR");
    };
    if !instance
        .compiled()
        .inline_records()
        .g2
        .as_deref()
        .expect("G2 high-pointer metadata")
        .checked_emission()
    {
        // The floor producer deliberately carries no duplicate pointer check;
        // malformed-pointer validation remains a checked-mode oracle gate.
        return;
    }
    let mut state = instance.create_initial_state(Streams::default());
    let mut registers = [0u64; 32];
    registers[1] = u64::from(u32::MAX) + 1;
    registers[2] = 0x5a;
    openvm_circuit::arch::rvr::bridge::write_rv64_registers(&mut state, &registers);
    let err = match instance.execute_preflight_from_state_with_device_touched_memory_for_test(
        state,
        Some(exe.program.num_defined_instructions() as u64),
        &rows,
        &BTreeMap::new(),
    ) {
        Ok(_) => panic!("G2 must reject a high pointer before dereferencing it"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("G2 wire v1"),
        "high-pointer rejection must fail closed through G2 wire validation: {err}"
    );
}

#[test]
fn rvr_preflight_g2_rejects_high_jalr_pointer_without_truncation() {
    let exe = exe(&[jalr(2, 1, 0), terminate()]);
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("G2 high-Jalr VM init");
    let pc_to_air = vm.pc_to_air_idx(&exe).expect("G2 high-Jalr pc mapping");
    let mut rows = vec![0u32; vm.num_airs()];
    for air in pc_to_air.iter().flatten() {
        rows[*air] += 1;
    }

    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("G2 high-Jalr route")
    else {
        panic!("high-Jalr fixture must route to G2 RVR");
    };
    assert!(instance.compiled().inline_records().g2.is_some());
    if !instance
        .compiled()
        .inline_records()
        .g2
        .as_deref()
        .expect("G2 high-Jalr metadata")
        .checked_emission()
    {
        // Production emits the semantic transition only; checked mode owns
        // the independent high-pointer fail-closed oracle.
        return;
    }
    let mut state = instance.create_initial_state(Streams::default());
    let mut registers = [0u64; 32];
    registers[1] = u64::from(u32::MAX) + 4;
    openvm_circuit::arch::rvr::bridge::write_rv64_registers(&mut state, &registers);
    let err = match instance.execute_preflight_from_state_with_device_touched_memory_for_test(
        state,
        Some(exe.program.num_defined_instructions() as u64),
        &rows,
        &BTreeMap::new(),
    ) {
        Ok(_) => panic!("G2 must reject a high JALR pointer instead of truncating it"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("G2 wire v1"),
        "high-Jalr rejection must fail closed through G2 finalization: {err}"
    );
}

#[test]
fn rvr_preflight_g2_phase2b_full_standard_matrix_is_byte_equal() {
    let exe = g2_full_standard_matrix_exe();
    let config = Rv64ImConfig::with_public_values_bytes(32);
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config)
        .expect("G2 Phase-2b oracle VM init");
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("G2 Phase-2b pc mapping");
    let mut exact_rows = vec![0u32; rvr_vm.num_airs()];
    for air in pc_to_air_idx.iter().flatten() {
        exact_rows[*air] += 1;
    }
    let capacities = exact_rows
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let instruction_count = exe.program.num_defined_instructions() as u64;

    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let RvrPreflightRoute::Rvr(compact) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("compact Phase-2b routed preflight")
    else {
        panic!("Phase-2b matrix must route to compact RVR");
    };
    let mut compact_output = compact
        .execute_preflight_from_state_with_capacities(
            compact.create_initial_state(Streams::default()),
            Some(instruction_count),
            &exact_rows,
        )
        .expect("compact Phase-2b preflight");
    let compact_consumers = compact_output
        .inline_records
        .iter()
        .map(|records| {
            (
                records.air_idx,
                (records.record_size, records.bytes.clone()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let compact_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        DenseRecordArena,
    >(&exe, &mut compact_output, &capacities, &pc_to_air_idx)
    .expect("compact Phase-2b final arenas");

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let RvrPreflightRoute::Rvr(g2) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("G2 Phase-2b routed preflight")
    else {
        panic!("Phase-2b matrix must route to G2 RVR");
    };
    let inline = g2.compiled().inline_records();
    assert!(
        inline.g2.is_some(),
        "full standard matrix must negotiate G2"
    );
    assert!(pc_to_air_idx.iter().enumerate().all(|(slot, air)| {
        air.is_none() || inline.pc_slots.get(slot).copied().unwrap_or(false)
    }));
    let mut g2_output = g2
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            g2.create_initial_state(Streams::default()),
            Some(instruction_count),
            &exact_rows,
            &BTreeMap::new(),
        )
        .expect("G2 Phase-2b preflight");
    let segment = g2_output.g2_segment.take().expect("committed G2 segment");
    let meta = g2_output.g2_meta.as_deref().expect("G2 Phase-2b metadata");
    let decode = g2_output
        .delta_decode_precomputed
        .as_deref()
        .expect("G2 Phase-2b operand table");
    assert_eq!(
        meta.air_bindings
            .iter()
            .map(|binding| binding.kind)
            .collect::<Vec<_>>(),
        (0u8..30).collect::<Vec<_>>(),
        "Phase-2b matrix must bind every standard decoder kind"
    );
    let mut initial_blocks = BTreeMap::new();
    initial_blocks.insert((PUBLIC_VALUES_AS as u8, 0), 0);
    let reference = openvm_circuit::arch::rvr::decode_reference_v1(
        &segment,
        meta,
        decode,
        [0; 32],
        &initial_blocks,
        openvm_circuit::arch::rvr::PREFLIGHT_INITIAL_TIMESTAMP,
    )
    .expect("G2 Phase-2b CPU reference decode");

    let descs = segment
        .validate(&meta.fingerprint)
        .expect("valid G2 descriptors");
    for kind in 1u8..=7 {
        let binding = meta
            .air_bindings
            .iter()
            .find(|binding| binding.kind == kind)
            .expect("variable-arity kind binding");
        let mut total = 0usize;
        let mut register = 0usize;
        for &slot in &reference.expanded_program_slots {
            let entry = decode.entries[slot as usize];
            if entry.air_idx as usize == binding.air_idx {
                total += 1;
                register += usize::from(entry.flags & 1 == 0);
            }
        }
        assert!(
            total > register && register > 0,
            "kind {kind} must cover both arities"
        );
        assert_eq!(
            descs
                .iter()
                .find(|desc| desc.kind == 0x0100 + 2 * u16::from(kind))
                .map_or(0, |desc| desc.count as usize),
            total,
            "kind {kind} V0 count"
        );
        assert_eq!(
            descs
                .iter()
                .find(|desc| desc.kind == 0x0101 + 2 * u16::from(kind))
                .map_or(0, |desc| desc.count as usize),
            register,
            "kind {kind} V1 register-only count"
        );
    }
    for kind in [12u8, 14] {
        assert!(descs
            .iter()
            .all(|desc| desc.kind != 0x0100 + 2 * u16::from(kind)
                && desc.kind != 0x0101 + 2 * u16::from(kind)));
    }

    assert!(g2_output.raw_logs.program_log.is_empty());
    assert!(g2_output.raw_logs.memory_log.is_empty());
    assert!(g2_output.raw_logs.delta_memory_log.is_empty());
    assert!(g2_output.access_aux.is_empty());
    assert!(g2_output.inline_records.is_empty());
    assert!(g2_output.delta_records.is_none());
    assert!(!g2_output.access_aux_complete);
    assert_eq!(
        openvm_circuit::arch::rvr::bridge::read_rv64_registers(&g2_output.to_state),
        reference.final_registers,
        "G2 CPU execution and operand-table replay must finish with identical registers"
    );

    for binding in meta.air_bindings.iter() {
        let bytes = reference
            .compact_records
            .get(&binding.kind)
            .expect("every bound kind has decoded records");
        let native_multi = matches!(binding.kind, 20..=22 | 24..=28);
        let stride = if native_multi {
            assert_eq!(
                bytes,
                compact_arenas[binding.air_idx].allocated(),
                "kind {} native two-block consumer bytes",
                binding.kind
            );
            60
        } else {
            let (stride, expected) = compact_consumers
                .get(&binding.air_idx)
                .expect("compact consumer for bound AIR");
            assert_eq!(
                bytes, expected,
                "kind {} compact consumer bytes",
                binding.kind
            );
            *stride
        };
        assert_eq!(
            bytes.len() / stride,
            exact_rows[binding.air_idx] as usize,
            "kind {} exact AIR count",
            binding.kind
        );
    }
}

#[test]
fn rvr_preflight_g2_rejects_lifted_nop_before_execution() {
    let exe = exe(&[addi(0, 1, 3), terminate()]);
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("G2 lifted-NOP fallback VM init");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("G2 lifted-NOP fallback negotiation")
    else {
        panic!("supported RV64 program must retain the RVR route");
    };
    let inline = instance.compiled().inline_records();
    assert!(inline.g2.is_none());
    assert!(inline.delta_decode.is_none());
    assert!(
        !inline.pc_slots.iter().any(|&slot| slot),
        "lifted NOP must retain the established host-record route"
    );
}

#[test]
fn rvr_preflight_g2_rejects_dso_bound_to_different_vmexe() {
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("G2 manifest VM init");
    let cache_dir = tempfile::tempdir().expect("G2 mismatch cache dir");
    let cache_a = cache_dir.path().join("a.so");
    let cache_b = cache_dir.path().join("b.so");
    let mismatch = cache_dir.path().join("mismatch.so");

    let exe_a = exe(&[addi(1, 0, 3), terminate()]);
    std::env::set_var("OPENVM_RVR_PREFLIGHT_LIB", &cache_a);
    let RvrPreflightRoute::Rvr(_instance_a) = vm
        .preflight_routed_instance(&exe_a)
        .expect("first G2 manifest build")
    else {
        panic!("AddI program must route to RVR");
    };
    let exe_b = exe(&[addi(1, 0, 4), terminate()]);
    std::env::set_var("OPENVM_RVR_PREFLIGHT_LIB", &cache_b);
    let RvrPreflightRoute::Rvr(_instance_b) = vm
        .preflight_routed_instance(&exe_b)
        .expect("second G2 manifest build")
    else {
        panic!("AddI program must route to RVR");
    };

    // Preserve B's input/project keys but A's artifact digest. The cache is
    // therefore structurally valid and reaches the independent DSO-manifest
    // comparison, which must reject the mismatched VmExe binding.
    let manifest_path = |library: &std::path::Path| {
        std::path::PathBuf::from(format!("{}.sha256", library.display()))
    };
    let value = |manifest: &str, key: &str| {
        manifest
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("missing {key} in native cache manifest"))
            .to_string()
    };
    let manifest_a = std::fs::read_to_string(manifest_path(&cache_a)).expect("read A manifest");
    let manifest_b = std::fs::read_to_string(manifest_path(&cache_b)).expect("read B manifest");
    std::fs::copy(&cache_a, &mismatch).expect("copy mismatched G2 artifact");
    std::fs::write(
        manifest_path(&mismatch),
        format!(
            "project={}\nartifact={}\ninput={}\n",
            value(&manifest_b, "project"),
            value(&manifest_a, "artifact"),
            value(&manifest_b, "input"),
        ),
    )
    .expect("write mismatched G2 cache manifest");
    std::env::set_var("OPENVM_RVR_PREFLIGHT_LIB", &mismatch);
    let err = match vm.preflight_routed_instance(&exe_b) {
        Ok(_) => panic!("a DSO bound to another VmExe must fail before native execution"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("G2 DSO manifest"),
        "unexpected G2 manifest rejection: {err}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn rvr_preflight_gpu_default_negotiates_g2_falls_back_and_honors_override() {
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    assert_eq!(
        VmBuilder::<BabyBearPoseidon2GpuEngine>::default_rvr_preflight_engine(
            &Rv64ImGpuBuilder::default(),
        ),
        RvrPreflightEngine::Rvr
    );
    assert_eq!(
        crate::rvr_gpu_decode::configured_emission_mode(),
        Some(crate::rvr_gpu_decode::InlineEmissionMode::G2)
    );

    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("CUDA-default G2 VM init");
    let supported = exe(&[addi(1, 0, 3), terminate()]);
    let RvrPreflightRoute::Rvr(cpu_default) = vm
        .preflight_routed_instance(&supported)
        .expect("CPU default route")
    else {
        panic!("supported AddI program must retain the RVR route");
    };
    assert!(
        cpu_default.compiled().inline_records().g2.is_none(),
        "CUDA-enabled CPU builders must retain the arena route"
    );

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let RvrPreflightRoute::Rvr(g2) = vm
        .preflight_routed_instance(&supported)
        .expect("CUDA-default G2 negotiation")
    else {
        panic!("supported AddI program must retain the RVR route");
    };
    assert!(
        g2.compiled().inline_records().g2.is_some(),
        "negotiable executable must select G2 by default"
    );

    let unsupported = exe(&[addi(0, 1, 3), terminate()]);
    let RvrPreflightRoute::Rvr(arena) = vm
        .preflight_routed_instance(&unsupported)
        .expect("CUDA-default arena fallback")
    else {
        panic!("supported RV64 program must retain the RVR route");
    };
    let fallback = arena.compiled().inline_records();
    assert!(fallback.g2.is_none() && fallback.delta_decode.is_none());
    assert!(
        !fallback.pc_slots.iter().any(|&slot| slot),
        "failed G2 negotiation must fall back before execution"
    );

    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "arena-native");
    assert_eq!(
        crate::rvr_gpu_decode::configured_emission_mode(),
        Some(crate::rvr_gpu_decode::InlineEmissionMode::ArenaNative)
    );
    let RvrPreflightRoute::Rvr(explicit_arena) = vm
        .preflight_routed_instance(&supported)
        .expect("explicit arena override")
    else {
        panic!("supported AddI program must retain the RVR route");
    };
    assert!(explicit_arena.compiled().inline_records().g2.is_none());

    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
}

#[test]
fn rvr_preflight_two_block_compact_bypass_preserves_wire_rows() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let exe = two_block_main_memory_exe();
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("two-block compact-bypass vm init");
    let capacities = vec![4096u32; vm.num_airs()]
        .iter()
        .zip(vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let mut output = {
        let RvrPreflightRoute::Rvr(instance) = vm
            .preflight_routed_instance(&exe)
            .expect("routed compact-bypass instance")
        else {
            panic!("two-block program must route to RVR preflight");
        };
        instance
            .execute_preflight(Streams::default(), None)
            .expect("two-block compact-bypass execution")
    };
    let all_wire = output
        .inline_records
        .iter()
        .map(|chip| (chip.air_idx, chip.bytes.clone()))
        .collect::<BTreeMap<_, _>>();
    let compact_airs = crate::rvr_gpu_decode::RvrGpuDecodeState::compact_record_airs(
        &exe,
        &pc_to_air_idx,
        &output.inline_pc_slots,
        output.delta_decode_precomputed.as_deref(),
    );
    for (slot, entry) in exe.program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = entry else {
            continue;
        };
        if is_loadstore_opcode(instruction.opcode) {
            let air = pc_to_air_idx[slot].expect("load/store pc must map to an AIR");
            assert!(
                compact_airs.contains(&air),
                "AS=2 load/store AIR {air} must remain device-expandable"
            );
        }
    }
    let expected = all_wire
        .into_iter()
        .filter(|(air, _)| compact_airs.contains(air))
        .collect::<BTreeMap<_, _>>();
    let mut registry = LogNativeAssemblerRegistry::<F, DenseRecordArena>::new();
    Rv64ImConfig::default().extend_rvr_log_native(&mut registry);
    let (arenas, wire) = openvm_circuit::arch::rvr::generate_record_arenas_from_logs_with_compact(
        &registry,
        &exe,
        &mut output,
        &capacities,
        &pc_to_air_idx,
        &compact_airs,
    )
    .expect("two-block compact bypass");
    assert_eq!(wire.len(), expected.len());
    for chip in wire {
        assert_eq!(chip.bytes, expected[&chip.air_idx]);
        assert!(arenas[chip.air_idx].allocated().is_empty());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn rvr_gpu_two_block_compact_residual_proves_and_verifies() {
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");
    let segments = prove_gpu_rvr_preflight_and_verify_with_streams(
        two_block_main_memory_exe(),
        Rv64ImConfig::default(),
        Streams::default(),
    );
    assert_eq!(segments, 1);
}

#[cfg(feature = "cuda")]
#[test]
fn rvr_gpu_two_block_delta_residual_proves_and_verifies() {
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let segments = prove_gpu_rvr_preflight_and_verify_with_streams(
        two_block_main_memory_exe(),
        Rv64ImConfig::default(),
        Streams::default(),
    );
    assert_eq!(segments, 1);
}

#[cfg(feature = "cuda")]
#[test]
fn rvr_gpu_two_block_g2_crossings_prove_and_verify() {
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let segments = prove_gpu_rvr_preflight_and_verify_with_streams(
        two_block_main_memory_exe(),
        Rv64ImConfig::default(),
        Streams::default(),
    );
    assert_eq!(segments, 1);
}

#[test]
fn rvr_gpu_operand_table_rebinds_same_shape_different_exe() {
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let exe_a = exe(&[addi(1, 0, 1), terminate()]);
    let exe_len = exe_a.program.instructions_and_debug_infos.len();
    let map_a = vm.pc_to_air_idx(&exe_a).expect("first pc map");
    let state = crate::rvr_gpu_decode::RvrGpuDecodeState::default();
    let identity_a = Arc::new(vec![true; exe_len]);

    state.bind_delta_airs(&exe_a, &map_a, &identity_a, None);
    let first = state.operand_table().expect("first operand table").0[0];
    drop(exe_a);
    drop(identity_a);

    let exe_b = exe(&[addi(2, 0, 1), terminate()]);
    assert_eq!(exe_len, exe_b.program.instructions_and_debug_infos.len());
    let map_b = vm.pc_to_air_idx(&exe_b).expect("second pc map");
    assert_eq!(map_a, map_b, "fixture must isolate the VmExe identity key");
    let identity_b = Arc::new(vec![true; exe_len]);
    state.bind_delta_airs(&exe_b, &map_b, &identity_b, None);
    let second = state.operand_table().expect("second operand table").0[0];

    assert_eq!(first.a, reg(1) as u32);
    assert_eq!(second.a, reg(2) as u32);
    assert_ne!(
        first, second,
        "same-size VmExe must not reuse stale operands"
    );
}

#[test]
fn rvr_preflight_persists_empty_delta_classification() {
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let exe = exe(&[terminate()]);
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("routed preflight instance")
    else {
        panic!("program must route to RVR preflight");
    };
    let meta = instance.compiled().inline_records();
    let precomputed = meta
        .delta_decode
        .as_ref()
        .expect("available AOT classifier must persist an empty result");
    assert!(precomputed.kind_to_air.is_empty());
    assert!(meta.fully_direct_delta);
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

#[test]
fn interpreter_preflight_marks_carried_memory_pages() {
    let exe = exe(&[addi(1, 0, 64), terminate()]);
    let config = Rv64ImConfig::default();
    let (vm, _) = VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config)
        .expect("vm init");
    let trace_heights = vec![4096u32; vm.num_airs()];
    let state = vm.create_initial_state(&exe, Streams::default());
    let mut interpreter = vm
        .preflight_interpreter(&exe)
        .expect("preflight interpreter");

    let output = vm
        .execute_preflight_for(&mut interpreter, state, 1, &trace_heights)
        .expect("segment preflight");
    let register_memory = &output.to_state.memory.memory;
    let register_config = &register_memory.config[RV64_REGISTER_AS as usize];
    let register_bytes = register_config.num_cells * register_config.layout.size();
    let x1_byte_offset = reg(1);
    let x1_bytes = unsafe {
        output.to_state.memory.get_u8_slice(
            RV64_REGISTER_AS,
            x1_byte_offset as u32,
            size_of::<u64>(),
        )
    };
    assert!(
        x1_bytes.iter().any(|&byte| byte != 0),
        "x1 bytes are {x1_bytes:?}, pc={}",
        output.to_state.pc()
    );

    let touched_ranges = register_memory.touched_pages[RV64_REGISTER_AS as usize]
        .touched_byte_ranges(register_bytes);
    assert!(
        touched_ranges
            .iter()
            .any(|&(start, end)| start <= x1_byte_offset && x1_byte_offset < end),
        "x1 at byte offset {x1_byte_offset} was not marked for the next segment's sparse H2D transfer"
    );
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
        alu_w_imm(BaseAluWOpcode::ADDW, 15, 1, 0xff_fffb),
        shift_w(ShiftWOpcode::SLLW, 17, 1, 2),
        shift_w(ShiftWOpcode::SRLW, 18, 17, 2),
        shift_w(ShiftWOpcode::SRAW, 19, 17, 2),
        shift_w_reg(ShiftWOpcode::SLLW, 17, 1, 2),
        shift_w_reg(ShiftWOpcode::SRLW, 18, 17, 2),
        shift_w_reg(ShiftWOpcode::SRAW, 19, 17, 2),
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

    // rd=x0 control flow: the register write is suppressed (f=0, matching the
    // transpiler's rd != 0 encoding) but the timestamp tick and AIR row remain.
    instructions.push(jal_lui(Rv64JalLuiOpcode::JAL, 0, 4));
    let jalr_x0_next_pc = (instructions.len() + 2) * 4;
    instructions.push(addi(10, 0, jalr_x0_next_pc));
    instructions.push(jalr(0, 10, 0));

    instructions.push(mul(24, 1, 2));
    instructions.extend([
        mulh(MulHOpcode::MULH, 25, 1, 2),
        mulh(MulHOpcode::MULHSU, 26, 1, 2),
        mulh(MulHOpcode::MULHU, 27, 1, 2),
        mul_w(28, 1, 2),
    ]);

    // Sign-edge sweep: MSB-set (negative) operands through the sign-sensitive
    // compare/branch/mulh/shift paths; all prior fixture operands are positive.
    instructions.extend([
        alu_r(BaseAluOpcode::SUB, 29, 0, 2),
        less_than(LessThanOpcode::SLT, 30, 29, 1),
        less_than(LessThanOpcode::SLT, 31, 1, 29),
        less_than(LessThanOpcode::SLTU, 30, 29, 1),
        branch_lt(BranchLessThanOpcode::BLT, 29, 1, 4),
        branch_lt(BranchLessThanOpcode::BGE, 1, 29, 4),
        branch_lt(BranchLessThanOpcode::BLTU, 1, 29, 4),
        mulh(MulHOpcode::MULH, 30, 29, 2),
        mulh(MulHOpcode::MULHSU, 31, 29, 2),
        shift(ShiftOpcode::SRA, 30, 29, 3),
        shift_w(ShiftWOpcode::SRAW, 31, 29, 3),
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
        // divrem special cases: zero divisor (rs2 = x0) and signed overflow
        // (MIN / -1), 64- and 32-bit — the only core branches the plain
        // 21/5 operands above never take.
        divrem(DivRemOpcode::DIV, 26, 10, 0),
        divrem(DivRemOpcode::REM, 27, 10, 0),
        divrem(DivRemOpcode::DIVU, 28, 10, 0),
        divrem(DivRemOpcode::REMU, 29, 10, 0),
        divrem_w(DivRemWOpcode::DIVW, 26, 10, 0),
        divrem_w(DivRemWOpcode::REMW, 27, 10, 0),
        addi(28, 0, 1),
        shift(ShiftOpcode::SLL, 28, 28, 63),
        shift(ShiftOpcode::SRA, 29, 28, 63),
        divrem(DivRemOpcode::DIV, 30, 28, 29),
        divrem(DivRemOpcode::REM, 31, 28, 29),
        addi(30, 0, 1),
        shift(ShiftOpcode::SLL, 30, 30, 31),
        divrem_w(DivRemWOpcode::DIVW, 31, 30, 29),
        divrem_w(DivRemWOpcode::REMW, 26, 30, 29),
        // LOADB sign edge: a stored byte with the MSB set must sign-extend.
        addi(9, 0, 0xaa),
        store(Rv64LoadStoreOpcode::STOREB, 9, 1, 3),
        load(Rv64LoadStoreOpcode::LOADB, 22, 1, 3),
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

fn hard_chip_streams(repeats: usize) -> Streams {
    let mut streams = Streams::default();
    for repeat in 0..repeats {
        for word in 0..3u64 {
            let value = 0x0102_0304_0506_0708u64 + (repeat as u64) * 0x100 + word;
            streams.hint_stream.extend(value.to_le_bytes());
        }
    }
    streams
}

fn hintstore_direct_exe() -> VmExe<F> {
    exe(&[
        addi(7, 0, 64),
        addi(8, 0, 3),
        addi(20, 0, 96),
        hint_store(Rv64HintStoreOpcode::HINT_STORED, 0, 7),
        hint_store(Rv64HintStoreOpcode::HINT_BUFFER, 8, 20),
        terminate(),
    ])
}

fn hintstore_direct_streams() -> Streams {
    let mut streams = Streams::default();
    for word in [
        0x0102_0304_0506_0708u64,
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
        0x3132_3334_3536_3738,
    ] {
        streams.hint_stream.extend(word.to_le_bytes());
    }
    streams
}

fn phantom_direct_exe() -> VmExe<F> {
    exe(&[
        phantom(SysPhantom::Nop),
        phantom(SysPhantom::CtStart),
        phantom(SysPhantom::CtEnd),
        rv64_phantom(Rv64Phantom::HintInput),
        rv64_phantom(Rv64Phantom::PrintStr),
        rv64_phantom(Rv64Phantom::HintRandom),
        terminate(),
    ])
}

fn full_rv64im_matrix_exe() -> VmExe<F> {
    let mut instructions = vec![addi(1, 0, 9), addi(2, 0, 5)];
    push_standard_group_ops(&mut instructions);
    push_hard_chip_ops(&mut instructions);
    // REVEAL row: STORED to PUBLIC_VALUES_AS (x0 as the AS3 pointer, r10 = 21
    // from the hard-chip group) so the loadstore mem_as=3 path is locked in the
    // CPU-vs-GPU three-way, not only in the reveal-specific tests.
    instructions.push(extension_store(10, 0, 0));
    instructions.push(terminate());
    exe(&instructions)
}

fn g2_full_standard_matrix_exe() -> VmExe<F> {
    let mut instructions = vec![addi(1, 0, 9), addi(2, 0, 5)];
    push_standard_group_ops(&mut instructions);
    instructions.extend([
        alu_imm(BaseAluOpcode::XOR, 3, 1, 0xff_fffb),
        less_than_imm(LessThanOpcode::SLT, 4, 1, 0xff_fffb),
        shift_reg(ShiftOpcode::SLL, 5, 1, 2),
        shift_reg(ShiftOpcode::SRL, 6, 1, 2),
        shift_reg(ShiftOpcode::SRA, 7, 1, 2),
    ]);
    let standard_end = instructions.len();
    push_hard_chip_ops(&mut instructions);
    // The hard-chip fixture ends in three phantoms and two HintStore custom
    // instructions. Phase 2b covers the complete fixed standard prefix;
    // custom opaque/event-replay transport remains outside this oracle.
    debug_assert!(instructions.len() - standard_end >= 5);
    instructions.truncate(instructions.len() - 5);
    instructions.push(extension_store(10, 0, 0));
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

fn g2_load_store_phase2a_exe() -> VmExe<F> {
    exe(&[
        addi(1, 0, 64),
        addi(2, 0, 0x00ff_ffff), // -1: exercises every sign-extending load.
        store(Rv64LoadStoreOpcode::STORED, 2, 1, 0),
        store(Rv64LoadStoreOpcode::STOREW, 2, 1, 0),
        store(Rv64LoadStoreOpcode::STOREH, 2, 1, 0),
        store(Rv64LoadStoreOpcode::STOREB, 2, 1, 0),
        load(Rv64LoadStoreOpcode::LOADD, 3, 1, 0),
        load(Rv64LoadStoreOpcode::LOADWU, 4, 1, 0),
        load(Rv64LoadStoreOpcode::LOADHU, 5, 1, 0),
        load(Rv64LoadStoreOpcode::LOADBU, 6, 1, 0),
        load(Rv64LoadStoreOpcode::LOADW, 7, 1, 0),
        load(Rv64LoadStoreOpcode::LOADH, 8, 1, 0),
        load(Rv64LoadStoreOpcode::LOADB, 9, 1, 0),
        load(Rv64LoadStoreOpcode::LOADD, 0, 1, 0), // x0 still performs the read.
        addi(10, 0, 0),
        extension_store(2, 10, 0),
        extension_store_width(Rv64LoadStoreOpcode::STOREW, 2, 10, 8),
        extension_store_width(Rv64LoadStoreOpcode::STOREH, 2, 10, 16),
        extension_store_width(Rv64LoadStoreOpcode::STOREB, 2, 10, 24),
        terminate(),
    ])
}

fn hint_input_streams() -> Streams {
    Streams::from(vec![vec![0x11, 0x22, 0x33]])
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
    assert_eq!(
        interp.touched_memory_on_device, rvr.touched_memory_on_device,
        "{label}: touched-memory route"
    );
}

fn prove_rvr_preflight_and_verify_with_streams(
    exe: VmExe<F>,
    config: Rv64ImConfig,
    streams: Streams,
) -> usize {
    let engine = test_cpu_engine();
    let (vm, pk) =
        VirtualMachine::new_with_keygen(engine, Rv64ImCpuBuilder, config).expect("vm init");
    let vk = pk.get_vk();
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance =
        VmInstance::new(vm, Arc::new(exe), cached_program_trace).expect("instance init");
    // CPU proving defaults to the interpreter engine; this fixture exists to
    // prove rvr-generated records, so pin the engine explicitly.
    instance.set_rvr_preflight_engine(Some(RvrPreflightEngine::Rvr));
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
    streams: Streams,
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
    // Pin the rvr engine: the route assertion above is only meaningful if
    // the prove below actually runs on the rvr preflight path.
    instance.set_rvr_preflight_engine(Some(RvrPreflightEngine::Rvr));
    let proof = ContinuationVmProver::prove(&mut instance, streams).expect("prove");
    verify_segments(&instance.vm.engine, &vk, &proof.per_segment).expect("verify segments");
    proof.per_segment.len()
}

fn assert_preflight_matches_interpreter_with_streams(
    label: &str,
    exe: VmExe<F>,
    num_insns: Option<u64>,
    streams: Streams,
) -> RvrPreflightOutput<F> {
    // Direct execute without arena targets: compile compact (fused emission
    // is default-on and has no target-less fallback).
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
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
    let interp_output = match num_insns {
        Some(num_insns) => {
            vm.execute_preflight_for(&mut interpreter, interp_state, num_insns, &trace_heights)
        }
        None => vm.execute_preflight(&mut interpreter, interp_state, &trace_heights),
    }
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
        assert_read_dominant_memory_aux(&rvr_output, &exe);
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
    streams: Streams,
    scope: TraceCompareScope,
) {
    // Direct execute without arena targets: compile compact (fused emission
    // is default-on and has no target-less fallback).
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
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
        .execute_preflight(&mut interpreter, interp_state, &trace_heights)
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
    let mut rvr_output = {
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
    >(&exe, &mut rvr_output, &capacities, &pc_to_air_idx)
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
const FULL_RV64IM_INSTRUCTION_AIR_COUNT: usize = 22;

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

/// System AIRs whose CPU-vs-GPU traces must match byte-for-byte in the three-way
/// comparison: the persistent boundary and memory merkle chips (continuation-state
/// transport), the connector, and the range/lookup count tables fed by every other
/// chip's side effects.
#[cfg(feature = "cuda")]
fn system_compare_air_ids(air_names: &[String]) -> Vec<usize> {
    let ids = air_names
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            (name.starts_with("PersistentBoundaryAir<")
                || name.starts_with("MemoryMerkleAir<")
                || name == "VmConnectorAir"
                || name == "VariableRangeCheckerAir"
                || name.starts_with("RangeTupleCheckerAir<"))
            .then_some(idx)
        })
        .collect::<Vec<_>>();
    // Recorded evidence for the GPU re-gate: this set first *executes* on GPU, so
    // print the resolved ids (visible via `--no-capture` / on failure) rather than
    // relying on the assert message alone.
    eprintln!(
        "system_compare_air_ids: {:?}",
        ids.iter()
            .map(|idx| (*idx, air_names[*idx].as_str()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        ids.len(),
        5,
        "expected boundary/merkle/connector/range system AIRs, found: {:?}",
        ids.iter()
            .map(|idx| air_names[*idx].as_str())
            .collect::<Vec<_>>()
    );
    ids
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

/// MemoryMerkle rows are emitted in backend-specific order (the CPU walks the old/new
/// tree traversal interleaved, the CUDA kernel writes rows at computed per-layer
/// offsets), and the AIR admits any row order that satisfies its constraints — the
/// all-segment stark-debug sweeps validate both orderings. Compare such traces as a
/// row multiset; a genuine per-row value divergence still fails loudly.
#[cfg(feature = "cuda")]
fn assert_host_trace_rows_multiset_eq(
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
    let sorted_rows = |trace: &HostTrace| -> Vec<Vec<u32>> {
        let height = trace.common_main.height();
        let width = trace.common_main.width();
        let mut rows = (0..height)
            .map(|i| {
                (0..width)
                    .map(|j| trace.common_main.values[j * height + i].as_canonical_u32())
                    .collect::<Vec<u32>>()
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    };
    let left_rows = sorted_rows(left);
    let right_rows = sorted_rows(right);
    if let Some(idx) = (0..left_rows.len()).find(|&i| left_rows[i] != right_rows[i]) {
        panic!(
            "{label}: {air_name} row multisets differ between {left_name} and {right_name}: \
             first_sorted_mismatch={idx} left={:?} right={:?}",
            left_rows[idx], right_rows[idx]
        );
    }
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
    streams: Streams,
) -> usize {
    let engine = test_gpu_engine();
    let (vm, pk) = VirtualMachine::new_with_keygen(engine, Rv64ImGpuBuilder::default(), config)
        .expect("gpu vm init");
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
    from_state: VmState,
    num_insns: Option<u64>,
    trace_heights: &[u32],
    expected_active_instruction_air_count: Option<usize>,
) -> VmState {
    let (mut cpu_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("cpu vm init");
    let cpu_air_names = cpu_vm.air_names().map(str::to_owned).collect::<Vec<_>>();
    let mut cpu_interpreter = cpu_vm
        .preflight_interpreter(exe)
        .expect("cpu interpreter preflight");
    let cpu_output = match num_insns {
        Some(num_insns) => cpu_vm.execute_preflight_for(
            &mut cpu_interpreter,
            from_state.clone(),
            num_insns,
            trace_heights,
        ),
        None => cpu_vm.execute_preflight(&mut cpu_interpreter, from_state.clone(), trace_heights),
    }
    .expect("cpu interpreter execution");
    let retired_instructions = cpu_output
        .system_records
        .filtered_exec_frequencies
        .iter()
        .map(|&count| u64::from(count))
        .sum::<u64>();
    if let Some(expected) = num_insns {
        assert_eq!(
            retired_instructions, expected,
            "{label}: interpreter retired instruction count"
        );
    }

    let (mut gpu_arena_vm, _) = VirtualMachine::new_with_keygen(
        test_gpu_engine(),
        Rv64ImGpuBuilder::default(),
        config.clone(),
    )
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
    let gpu_arena_output = match num_insns {
        Some(num_insns) => gpu_arena_vm.execute_preflight_for(
            &mut gpu_interpreter,
            from_state.clone(),
            num_insns,
            trace_heights,
        ),
        None => {
            gpu_arena_vm.execute_preflight(&mut gpu_interpreter, from_state.clone(), trace_heights)
        }
    }
    .expect("gpu interpreter execution");

    let gpu_log_builder = Rv64ImGpuBuilder::default();
    let (mut gpu_log_vm, _) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), gpu_log_builder.clone(), config.clone())
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
    let capacities = trace_heights
        .iter()
        .zip(gpu_log_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let (mut rvr_output, staged_native) = {
        let route = gpu_log_vm
            .preflight_routed_instance(exe)
            .expect("routed preflight instance");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("{label}: program must route to RVR preflight");
        };
        let mut staged = Vec::new();
        let mut targets = BTreeMap::new();
        for &(air, geometry) in &instance.compiled().inline_records().arena_native_airs {
            let (height, width) = capacities[air];
            let (arena, target) = <DenseRecordArena as RvrArenaNativeTarget>::stage_arena_native(
                height, width, &geometry,
            );
            targets.insert(air, target);
            staged.push((air, geometry, arena));
        }
        let output = instance
            .execute_preflight_from_state_with_arena_targets(
                from_state.clone(),
                Some(retired_instructions),
                trace_heights,
                &targets,
            )
            .expect("gpu rvr preflight execution");
        (output, staged)
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

    let mut rvr_to_state = rvr_output.to_state.clone();
    let pc_to_air_idx = gpu_log_vm.pc_to_air_idx(exe).expect("pc to air mapping");
    let mut gpu_log_record_arenas = gpu_log_builder
        .generate_rvr_record_arenas_from_logs(
            config,
            exe,
            &mut rvr_output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("gpu builder log-native record assembly")
        .expect("gpu builder must support rvr log-native tracegen");
    for (air, geometry, mut arena) in staged_native {
        let written = rvr_output
            .arena_native_written
            .iter()
            .find(|&&(written_air, _)| written_air == air)
            .map(|&(_, count)| count as usize)
            .expect("arena-native AIR must report its written count");
        arena.finish_arena_native(written, &geometry);
        gpu_log_record_arenas[air] = arena;
    }

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
    gpu_log_vm.merge_device_continuation_dirty_pages(&mut rvr_to_state.memory);
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
    for &air_idx in &system_compare_air_ids(&air_names) {
        let air_name = air_names[air_idx].as_str();
        let gpu_trace = gpu_arena_traces
            .get(&air_idx)
            .unwrap_or_else(|| panic!("{label}: gpu_from_record_arenas missing {air_name}"));
        let cpu_trace = cpu_traces
            .get(&air_idx)
            .unwrap_or_else(|| panic!("{label}: cpu_interpreter missing {air_name}"));
        if air_name.starts_with("MemoryMerkleAir<") {
            assert_host_trace_rows_multiset_eq(
                label,
                air_name,
                "gpu_from_record_arenas",
                "cpu_interpreter",
                gpu_trace,
                cpu_trace,
            );
        } else {
            assert_host_trace_eq(
                label,
                air_name,
                "gpu_from_record_arenas",
                "cpu_interpreter",
                gpu_trace,
                cpu_trace,
            );
        }
    }

    rvr_to_state
}

#[cfg(feature = "cuda")]
fn assert_gpu_rvr_three_way_single_segment(
    label: &str,
    exe: VmExe<F>,
    streams: Streams,
    expected_active_instruction_air_count: Option<usize>,
) {
    assert_gpu_rvr_three_way_single_segment_with_config(
        label,
        exe,
        streams,
        expected_active_instruction_air_count,
        Rv64ImConfig::default(),
    );
}

#[cfg(feature = "cuda")]
fn assert_gpu_rvr_three_way_single_segment_with_config(
    label: &str,
    exe: VmExe<F>,
    streams: Streams,
    expected_active_instruction_air_count: Option<usize>,
    config: Rv64ImConfig,
) {
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("vm init");
    // Size the arenas from the real metered heights so the test exercises production
    // sizing: DenseRecordArena capacity is exact with no slack, so a metering/assembly
    // count mismatch fails here instead of only at reth scale.
    let segments = {
        let metered_ctx = vm.build_metered_ctx(&exe);
        let metered = vm.metered_interpreter(&exe).expect("metered interpreter");
        let (segments, _) = metered
            .execute_metered(streams.clone(), metered_ctx)
            .expect("metered execution");
        segments
    };
    assert_eq!(
        segments.len(),
        1,
        "{label}: single-segment fixture must meter into one segment"
    );
    let from_state = vm.create_initial_state(&exe, streams);
    assert_gpu_rvr_three_way_from_state(
        label,
        &exe,
        &config,
        from_state,
        None,
        &segments[0].trace_heights,
        expected_active_instruction_air_count,
    );
}

#[cfg(feature = "cuda")]
fn assert_gpu_rvr_three_way_multi_segment(label: &str, exe: VmExe<F>, streams: Streams) {
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

/// Test-side view of one 44-byte compact alu3 record (loads/stores share the
/// shape: p2 = memory access prev_timestamp, pw = write prev_timestamp).
struct CompactAlu3View {
    from_pc: u64,
    reads_prev: [u32; 2],
    write_prev: u32,
}

fn compact_alu3_records(output: &RvrPreflightOutput<F>, air_idx: usize) -> Vec<CompactAlu3View> {
    let chip = output
        .inline_records
        .iter()
        .find(|chip| chip.air_idx == air_idx)
        .expect("air must have inline records");
    assert_eq!(chip.record_size, 44, "expected the alu3 wire shape");
    chip.bytes
        .chunks_exact(44)
        .map(|r| CompactAlu3View {
            from_pc: u64::from(u32::from_le_bytes(r[0..4].try_into().unwrap())),
            reads_prev: [
                u32::from_le_bytes(r[8..12].try_into().unwrap()),
                u32::from_le_bytes(r[12..16].try_into().unwrap()),
            ],
            write_prev: u32::from_le_bytes(r[16..20].try_into().unwrap()),
        })
        .collect()
}

fn program_ts_at_pc(output: &RvrPreflightOutput<F>, pc: u64, nth: usize) -> u32 {
    output
        .raw_logs
        .program_log
        .iter()
        .filter(|entry| u64::from(entry.pc()) == pc)
        .nth(nth)
        .expect("pc must be in the program log")
        .timestamp
}

fn assert_read_dominant_memory_aux(output: &RvrPreflightOutput<F>, exe: &VmExe<F>) {
    // The store and the two following reads of memory block 64 are migrated
    // (inline compact records); the prev-timestamp chain that used to be
    // asserted on the verbose log is asserted on the records instead:
    // read-after-write chains to the store's write tick, read-after-read to
    // the previous read's tick.
    let pc_to_air_idx = {
        let (vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ImCpuBuilder,
            Rv64ImConfig::default(),
        )
        .expect("vm init");
        vm.pc_to_air_idx(exe).expect("pc to air mapping")
    };
    let pc_of = |opcode: openvm_instructions::VmOpcode, nth: usize| {
        exe.program
            .instructions_and_debug_infos
            .iter()
            .enumerate()
            .filter(|(_, s)| s.as_ref().is_some_and(|(insn, _)| insn.opcode == opcode))
            .map(|(i, _)| u64::from(exe.program.pc_base) + (i as u64) * 4)
            .nth(nth)
            .expect("opcode occurrence")
    };
    let store_pc = pc_of(Rv64LoadStoreOpcode::STOREB.global_opcode(), 0);
    let read1_pc = pc_of(Rv64LoadStoreOpcode::LOADBU.global_opcode(), 0);
    let read2_pc = pc_of(Rv64LoadStoreOpcode::LOADBU.global_opcode(), 1);
    let record_at = |pc: u64| {
        let slot = ((pc - u64::from(exe.program.pc_base)) / 4) as usize;
        let air_idx = pc_to_air_idx[slot].expect("memory opcode maps to an air");
        compact_alu3_records(output, air_idx)
            .into_iter()
            .find(|r| r.from_pc == pc)
            .expect("record at pc")
    };
    let store_ts = program_ts_at_pc(output, store_pc, 0);
    let read1_ts = program_ts_at_pc(output, read1_pc, 0);
    let (store, read1, read2) = (
        record_at(store_pc),
        record_at(read1_pc),
        record_at(read2_pc),
    );
    // Store's memory write happens at tick store_ts + 2; the first read's
    // memory access (p2 slot) must chain to it, the second to the first
    // read's memory tick (read1_ts + 1).
    assert_eq!(
        read1.reads_prev[1],
        store_ts + 2,
        "read-after-write prev_timestamp must chain to the store's write tick"
    );
    assert_eq!(
        read2.reads_prev[1],
        read1_ts + 1,
        "read-after-read prev_timestamp must chain to the previous read"
    );
    assert!(
        store.write_prev < store_ts + 2,
        "store prev must precede it"
    );

    let block_addr = (RV64_MEMORY_AS, 64 / 2);
    let touched = output
        .system_records
        .touched_memory
        .iter()
        .find(|(addr, _)| *addr == block_addr)
        .expect("block must be touched");
    let read2_ts = program_ts_at_pc(output, read2_pc, 0);
    assert_eq!(
        touched.1.timestamp,
        read2_ts + 1,
        "touched_memory timestamp must be the trailing read's memory tick"
    );
    let expected_values = [
        F::from_u32(0x5a00),
        F::from_u32(0),
        F::from_u32(0),
        F::from_u32(0),
    ];
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
        .position(|entry| entry.pc() == load_pc)
        .expect("load-to-x0 instruction must be in program log");
    let load_timestamp = output.raw_logs.program_log[load_idx].timestamp;
    let next_timestamp = output.raw_logs.program_log[load_idx + 1].timestamp;
    assert_eq!(
        next_timestamp - load_timestamp,
        3,
        "rd=x0 load must preserve the fixed 3-tick load invariant"
    );
    // The load is migrated: its compact record must exist with the write
    // fields zeroed (suppressed rd write still ticks, logs nothing, and the
    // host selects the disabled branch from the instruction's enable flag).
    let record = output
        .inline_records
        .iter()
        .flat_map(|chip| chip.bytes.chunks_exact(chip.record_size))
        .find(|r| u32::from_le_bytes(r[0..4].try_into().unwrap()) == load_pc)
        .expect("rd=x0 load must emit a compact record");
    assert_eq!(
        u32::from_le_bytes(record[16..20].try_into().unwrap()),
        0,
        "suppressed rd write must record a zero write prev_timestamp"
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
        .find(|entry| entry.pc() == phantom_pc)
        .expect("phantom program-log entry");
    let next_entry = output
        .raw_logs
        .program_log
        .iter()
        .find(|entry| entry.pc() == next_pc)
        .expect("post-phantom program-log entry");
    assert_eq!(
        next_entry.timestamp,
        hint_entry.timestamp + 1,
        "{label}: Rv64 phantom must tick the shared timestamp before the following instruction"
    );
    // The following memory instruction is migrated (inline compact record);
    // its from_timestamp must be exactly the post-phantom tick, proving the
    // phantom's tick landed before the memory access chain.
    let record_from_ts = output
        .inline_records
        .iter()
        .flat_map(|chip| chip.bytes.chunks_exact(chip.record_size))
        .find(|r| u32::from_le_bytes(r[0..4].try_into().unwrap()) == next_entry.pc())
        .map(|r| u32::from_le_bytes(r[4..8].try_into().unwrap()))
        .expect("instruction after the phantom must emit a compact record");
    assert_eq!(
        record_from_ts, next_entry.timestamp,
        "{label}: the post-phantom instruction's record must start at the post-phantom tick"
    );
}

#[test]
fn rvr_preflight_differential_suite_system_records_full_rv64im_matrix() {
    // Direct execute without arena targets (see the harness note).
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
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
    // Exercise the compact AS=3 assembler explicitly; the arena-native path
    // has its own strengthened prove-and-verify gate below.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
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
    // Direct execute without arena targets (see the harness note).
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
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

/// A metered-height undercount must fail loudly at record-assembly time: the GPU-path
/// `DenseRecordArena` is sized exactly from metered heights with no slack, so before the
/// capacity assert was made unconditional an undercount wrote past the buffer and silently
/// corrupted the heap. This pins the loud-failure contract host-side, with no GPU needed.
#[test]
#[should_panic(expected = "failed to allocate")]
fn rvr_preflight_dense_arena_capacity_undercount_panics() {
    // Direct execute without arena targets (see the harness note).
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let exe = standard_group_exe(1);
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let air_names = vm.air_names().map(str::to_owned).collect::<Vec<_>>();
    let add_sub_air_idx = air_names
        .iter()
        .position(|name| {
            name.starts_with("VmAirWrapper<Rv64BaseAluRegU16AdapterAir, AddSubCoreAir<")
        })
        .expect("AddSub AIR present");
    let mut trace_heights = vec![4096u32; vm.num_airs()];
    trace_heights[add_sub_air_idx] = 0;
    let capacities = trace_heights
        .iter()
        .zip(vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let mut rvr_output = {
        let route = vm
            .preflight_routed_instance(&exe)
            .expect("routed preflight instance");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("program must route to RVR preflight");
        };
        instance
            .execute_preflight(Streams::default(), None)
            .expect("rvr preflight execution")
    };
    let _ = crate::log_native::generate_rv64im_record_arenas_from_logs::<F, DenseRecordArena>(
        &exe,
        &mut rvr_output,
        &capacities,
        &pc_to_air_idx,
    );
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
    assert_gpu_rvr_three_way_single_segment_with_config(
        "mixed_loadstore_reveal",
        public_values_reveal_differential_exe(),
        Streams::default(),
        None,
        Rv64ImConfig::with_public_values_bytes(32),
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
fn rvr_gpu_g2_continuation_dirty_pages_prove_and_verify() {
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let mut config = Rv64ImConfig::default();
    config.rv64i.system.segmentation_max_memory = 1;
    let segments = prove_gpu_rvr_preflight_and_verify_with_streams(
        continuation_boundary_memory_exe(),
        config,
        Streams::default(),
    );
    assert!(segments > 1, "dirty-page fixture must cross a continuation");
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

/// Runtime oracle for the GPU-only replay route. The opt-in keeps the host
/// reference alive solely for a byte-for-byte comparison inside the system
/// inventory; trace generation still consumes the device-produced records.
#[cfg(feature = "cuda")]
#[test]
fn rvr_gpu_delta_touched_memory_matches_host_across_continuations() {
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    std::env::set_var("OPENVM_RVR_DEVICE_REPLAY_ORACLE", "1");
    let mut config = Rv64ImConfig::default();
    config.rv64i.system.segmentation_max_memory = 1;
    let segments = prove_gpu_rvr_preflight_and_verify_with_streams(
        hard_chip_with_add_tail_exe(400),
        config,
        hard_chip_streams(1),
    );
    assert!(segments > 1, "oracle fixture must cross a continuation");
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
fn rvr_preflight_parallel_record_merge_matches_serial_bytes() {
    // L4 fail-hard oracle: the production per-AIR parallel merge is compared
    // byte-for-byte with an independent serial merge. This fixture exercises
    // multiple AIR lanes plus HintStore/Phantom residual assembly; the oracle
    // itself requires at least two non-empty lanes and a non-empty byte stream.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_PARALLEL_MERGE_ORACLE", "1");
    let exe = hard_chip_exe(1, false);
    let streams = hard_chip_streams(1);
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("rvr vm init");
    let trace_heights = vec![4096u32; vm.num_airs()];
    let capacities = trace_heights
        .iter()
        .zip(vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let route = vm
        .preflight_routed_instance(&exe)
        .expect("routed preflight instance");
    let RvrPreflightRoute::Rvr(instance) = route else {
        panic!("oracle fixture must route to RVR preflight");
    };

    let mut matrix_output = instance
        .execute_preflight(streams.clone(), None)
        .expect("matrix preflight execution");
    let matrix = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        MatrixRecordArena<F>,
    >(&exe, &mut matrix_output, &capacities, &pc_to_air_idx)
    .expect("matrix parallel record merge oracle");
    assert!(matrix.iter().any(|arena| !arena.is_empty()));

    let mut dense_output = instance
        .execute_preflight(streams, None)
        .expect("dense preflight execution");
    let dense = crate::log_native::generate_rv64im_record_arenas_from_logs::<F, DenseRecordArena>(
        &exe,
        &mut dense_output,
        &capacities,
        &pc_to_air_idx,
    )
    .expect("dense parallel record merge oracle");
    assert!(dense.iter().any(|arena| !arena.is_empty()));

    std::env::remove_var("OPENVM_RVR_PARALLEL_MERGE_ORACLE");
}

#[test]
fn rvr_preflight_proves_hard_chip_single_segment() {
    assert_preflight_matches_interpreter_with_streams(
        "hard_chip_proof",
        hard_chip_exe(1, false),
        None,
        hard_chip_streams(1),
    );
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
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

/// Regression fixture for bug #2 (D1, outcome A): a register set once to a
/// constant, carried across continuation boundaries, and read repeatedly by
/// AddSub. The pre-R1 normalizer produced a deterministic **+4 prev_timestamp
/// offset** for exactly this carried-register-read pattern, leaving the
/// MemoryBus (bus 1) LogUp one-sided on the AddSub AIR (reth CPU seg-28 /
/// GPU seg-18: 15×+1 orphaned sends at T, 15×−1 orphaned receives at T+4).
///
/// R1 replaced that normalizer with shadow-derived prev_timestamps; this fixture
/// is the mandatory guard that R1 keeps the MemoryBus balanced across the
/// boundary. prove+verify is the **global per-bus LogUp balance** gate: a
/// one-sided MemoryBus makes a segment proof invalid (and, with the debug
/// interaction checker on, trips `check_logup` exactly as reth did). x5/x1/x2
/// are never used as `rd`, so x5 stays constant and is carried across every
/// forced boundary while being read as an AddSub operand throughout.
fn carried_register_addsub_exe(reads: usize) -> VmExe<F> {
    let mut ins = vec![addi(5, 0, 0x1f), addi(1, 0, 1), addi(2, 0, 2)];
    for i in 0..reads {
        let rd = 6 + (i % 20); // x6..x25 — never x5/x1/x2
        ins.push(alu_r(BaseAluOpcode::ADD, rd, 5, 1)); // reads x5
        ins.push(alu_r(BaseAluOpcode::SUB, rd, 5, 2)); // reads x5 again
        ins.push(alu_r(BaseAluOpcode::ADD, 1, 1, 2)); // intervening block event (rewrites x1)
    }
    ins.push(terminate());
    exe(&ins)
}

#[test]
fn rvr_preflight_carried_register_addsub_bus_balance_across_boundary() {
    // Force the per-bus LogUp interaction check on regardless of the ambient
    // OPENVM_SKIP_DEBUG (nextest isolates each test in its own process); the
    // continuation prove+verify is the balance gate even if debug is skipped.
    std::env::remove_var("OPENVM_SKIP_DEBUG");
    reset_preflight_compile_invocations_for_test();
    let mut config = Rv64ImConfig::default();
    config.rv64i.system.segmentation_max_memory = 1;
    let segments = prove_rvr_preflight_and_verify(carried_register_addsub_exe(500), config);
    assert!(
        segments > 1,
        "tight segmentation must split the x5-read stream so x5 is carried across a boundary"
    );
}

/// Small looped RV64IM program: `iters` loop iterations of `body_adds` ADDs
/// plus counter/branch, driving a large dynamic instruction count from a tiny
/// static program (single compiled block set, backward branch).
fn checkpoint_loop_exe(iters_base: usize, iters_shift: usize, body_adds: usize) -> VmExe<F> {
    // x3 = iters_base << iters_shift loop iterations (iters_base <= 2047:
    // the addi immediate sign-extends from 12 bits); x1 = step, x2 = counter.
    let mut ins = vec![
        addi(1, 0, 1),
        addi(2, 0, 0),
        addi(3, 0, iters_base),
        shift(ShiftOpcode::SLL, 3, 3, iters_shift),
    ];
    let loop_start = ins.len(); // slot index of the first body instruction
    for k in 0..body_adds {
        ins.push(alu_r(BaseAluOpcode::ADD, 4 + (k % 20), 1, 2));
    }
    ins.push(addi(2, 2, 1));
    let branch_slot = ins.len();
    let offset = -(((branch_slot - loop_start) * 4) as isize);
    ins.push(Instruction::from_isize(
        BranchLessThanOpcode::BLTU.global_opcode(),
        reg(2) as isize,
        reg(3) as isize,
        offset,
        1,
        1,
    ));
    ins.push(terminate());
    exe(&ins)
}

/// Retired instruction count of [`checkpoint_loop_exe`].
fn checkpoint_loop_dyn_insns(iters_base: usize, iters_shift: usize, body_adds: usize) -> u64 {
    let iters = (iters_base as u64) << iters_shift;
    5 + iters * (body_adds as u64 + 2)
}

/// Validates the looped fixture's backward-branch encoding and semantics
/// against the interpreter at a small scale (the ignored checkpoint harness
/// below scales the same program to millions of retired instructions).
#[test]
fn rvr_preflight_loop_fixture_matches_interpreter() {
    let output = assert_preflight_matches_interpreter(
        "checkpoint_loop_small",
        checkpoint_loop_exe(16, 0, 5),
        None,
    );
    assert_eq!(output.instret, checkpoint_loop_dyn_insns(16, 0, 5));
}

/// The generated C records each counter's first 0→nonzero transition. Recycling must scrub
/// exactly those slots before the same compiled instance executes its next segment; otherwise
/// counts silently accumulate and corrupt both AIR heights and the program-frequency trace.
#[test]
fn rvr_preflight_pooled_counters_reset_between_segments() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let exe = checkpoint_loop_exe(16, 0, 5);
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let RvrPreflightRoute::Rvr(instance) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("routed preflight instance")
    else {
        panic!("program must route to RVR preflight")
    };

    let first = instance
        .execute_preflight_from_state(instance.create_initial_state(Streams::default()), None)
        .expect("first pooled-counter execution");
    let expected_chip_counts = first.raw_logs.chip_counts.clone();
    let expected_frequencies = first.system_records.filtered_exec_frequencies.clone();
    assert!(expected_chip_counts.iter().any(|&count| count != 0));
    assert!(expected_frequencies.iter().any(|&count| count != 0));
    assert_eq!(
        first.raw_logs.chip_counts_touched.len(),
        expected_chip_counts
            .iter()
            .filter(|&&count| count != 0)
            .count(),
        "chip first-touch list must cover every and only nonzero counter"
    );
    instance.recycle_output(first);

    let second = instance
        .execute_preflight_from_state(instance.create_initial_state(Streams::default()), None)
        .expect("second pooled-counter execution");
    assert_eq!(second.raw_logs.chip_counts, expected_chip_counts);
    assert_eq!(
        second.system_records.filtered_exec_frequencies,
        expected_frequencies
    );
    instance.recycle_output(second);
}

/// Device chronology commits block runs only after the instret suspender has
/// accepted the block. Otherwise every continuation would gain the first
/// unexecuted block of the following segment and corrupt both chronology and
/// frequencies. Recycling both outputs also exercises the shared cpool seam.
#[test]
fn rvr_preflight_device_chronology_respects_continuation_boundary() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    std::env::set_var("OPENVM_RVR_DEVICE_REPLAY_ORACLE", "1");
    let exe = checkpoint_loop_exe(4, 0, 2);
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("routed preflight instance")
    else {
        panic!("program must route to RVR preflight")
    };
    let heights = vec![256u32; vm.num_airs()];
    let targets = BTreeMap::new();
    let first = instance
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            vm.create_initial_state(&exe, Streams::default()),
            Some(4),
            &heights,
            &targets,
        )
        .expect("first device chronology segment");
    assert!(first.suspended);
    assert_eq!(first.instret, 4);
    assert_eq!(
        first
            .raw_logs
            .program_runs
            .iter()
            .map(|run| u64::from(run.instruction_count))
            .sum::<u64>(),
        first.instret,
        "suspended segment must not retain the rejected next block"
    );
    assert_eq!(
        first.raw_logs.device_program_references.len(),
        first.instret as usize
    );
    let second_state = first.to_state.clone();
    instance.recycle_output(first);

    let second = instance
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            second_state,
            Some(checkpoint_loop_dyn_insns(4, 0, 2) - 4),
            &heights,
            &targets,
        )
        .expect("second device chronology segment");
    assert!(!second.suspended);
    assert_eq!(
        second
            .raw_logs
            .program_runs
            .iter()
            .map(|run| u64::from(run.instruction_count))
            .sum::<u64>(),
        second.instret
    );
    assert_eq!(
        second.raw_logs.device_program_references.len(),
        second.instret as usize
    );
    instance.recycle_output(second);
    std::env::remove_var("OPENVM_RVR_DEVICE_REPLAY_ORACLE");
}

/// Phase-3 <30 ms checkpoint harness: preflight a large RV64IM-dominated
/// segment (~8.2M retired instructions, ~8.2M inline AddSub records) on the
/// single-shot proving path and report the wall time against the r4
/// write-bandwidth model. Manual:
/// `cargo nextest run --cargo-profile=fast -p openvm-riscv-circuit \
///  --features rvr --run-ignored only -- rvr_preflight_addsub_30ms_checkpoint`
#[test]
#[ignore = "manual Phase-3 checkpoint harness, not a correctness gate"]
fn rvr_preflight_addsub_30ms_checkpoint() {
    use std::time::Instant;

    let (iters_base, iters_shift, body_adds) = (1024usize, 3usize, 1000usize); // 8192 x 1002
    let exe = checkpoint_loop_exe(iters_base, iters_shift, body_adds);
    let dyn_insns = checkpoint_loop_dyn_insns(iters_base, iters_shift, body_adds);

    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let addsub_air = pc_to_air_idx[4].expect("body ADD maps to an air");
    let branch_air = pc_to_air_idx[4 + body_adds + 1].expect("branch maps to an air");
    let mut trace_heights = vec![1u32 << 10; rvr_vm.num_airs()];
    trace_heights[addsub_air] = 1 << 24;
    trace_heights[branch_air] = 1 << 14;

    let route = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("routed preflight instance");
    let RvrPreflightRoute::Rvr(instance) = route else {
        panic!("program must route to RVR preflight");
    };

    let mut best = f64::MAX;
    let mut first = f64::MAX;
    let mut record_bytes = 0usize;
    for iteration in 0..5 {
        let init = instance.create_initial_state(Streams::default());
        let t0 = Instant::now();
        let output = instance
            .execute_preflight_from_state_with_capacities(init, Some(dyn_insns), &trace_heights)
            .expect("rvr preflight execution");
        let dt = t0.elapsed().as_secs_f64();
        assert_eq!(output.instret, dyn_insns);
        record_bytes = output
            .inline_records
            .iter()
            .map(|chip| chip.bytes.len())
            .sum();
        // Return the segment buffers like the proving loop does after record
        // assembly, so iterations 2.. measure pooled steady state (iteration
        // 1 stays the cold, pool-filling arm — reported separately).
        instance.recycle_output(output);
        if iteration == 0 {
            first = dt;
        }
        best = best.min(dt);
    }
    eprintln!(
        "checkpoint: dyn_insns={dyn_insns} inline_record_bytes={record_bytes} \
         preflight_execute={:.2}ms cold_iter={:.2}ms record_stream={:.2}GB/s instr_rate={:.1}M/s",
        best * 1e3,
        first * 1e3,
        record_bytes as f64 / best / 1e9,
        dyn_insns as f64 / best / 1e6,
    );
}

/// Mixed-mode residual harness for HintStore (the one unmigrated RV64IM
/// shape): measures the verbose-log volume and host time attributable to
/// HintStore-class instructions at a given word count. Manual; feeds the
/// skip-vs-migrate decision for the multi-row shape.
fn hintstore_residual_exe(bufs: usize, words_per_buf: usize) -> (VmExe<F>, Streams, u64) {
    let mut ins = vec![
        addi(1, 0, 1024),          // ptr base
        addi(2, 0, words_per_buf), // word count register
    ];
    for _ in 0..bufs {
        ins.push(rv64_phantom(Rv64Phantom::HintInput));
        ins.push(hint_store(Rv64HintStoreOpcode::HINT_BUFFER, 2, 1));
    }
    ins.push(terminate());
    let dyn_insns = ins.len() as u64;
    let hint = vec![0xab; words_per_buf * 8];
    let streams = Streams::from(vec![hint; bufs]);
    (exe(&ins), streams, dyn_insns)
}

#[test]
#[ignore = "manual mixed-mode residual harness, not a correctness gate"]
fn rvr_preflight_hintstore_mixed_mode_residual() {
    use std::time::Instant;

    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    for (bufs, words) in [(8usize, 1usize), (8, 1024)] {
        let (exe, streams, dyn_insns) = hintstore_residual_exe(bufs, words);
        let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
        let mut trace_heights = vec![1u32 << 8; rvr_vm.num_airs()];
        let hint_air = pc_to_air_idx[3].expect("hint_store maps to an air");
        trace_heights[hint_air] = ((bufs * words).next_power_of_two().max(256) * 2) as u32;
        let capacities = trace_heights
            .iter()
            .zip(rvr_vm.pk().per_air.iter())
            .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
            .collect::<Vec<_>>();
        let route = rvr_vm
            .preflight_routed_instance(&exe)
            .expect("routed preflight instance");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("program must route to RVR preflight");
        };
        let mut best_exec = f64::MAX;
        let mut best_gen = f64::MAX;
        let mut log_entries = 0usize;
        for _ in 0..9 {
            let init = instance.create_initial_state(streams.clone());
            let t0 = Instant::now();
            let mut output = instance
                .execute_preflight_from_state_with_capacities(init, Some(dyn_insns), &trace_heights)
                .expect("rvr preflight execution");
            let exec_s = t0.elapsed().as_secs_f64();
            log_entries = output.raw_logs.memory_log.len();
            let t1 = Instant::now();
            let arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
                F,
                MatrixRecordArena<F>,
            >(&exe, &mut output, &capacities, &pc_to_air_idx)
            .expect("generation");
            let gen_s = t1.elapsed().as_secs_f64();
            std::hint::black_box(&arenas);
            best_exec = best_exec.min(exec_s);
            best_gen = best_gen.min(gen_s);
        }
        eprintln!(
            "hint_residual: bufs={bufs} words_per_buf={words} total_words={} \
             memory_log_entries={log_entries} execute={:.3}ms generation={:.3}ms",
            bufs * words,
            best_exec * 1e3,
            best_gen * 1e3,
        );
    }
}

/// R3 Phase-1 net timing harness: R1 (verbose log + host assembler) vs R3
/// (log-suppressed inline records + host-adopted C buffers) on an
/// AddSub-dominant program, end to end (preflight execute + record-arena
/// generation), for both arena types. Manual, not a CI gate — run with:
/// `cargo nextest run --cargo-profile=fast -p openvm-riscv-circuit \
///  --features rvr --run-ignored only -- rvr_preflight_inline_addsub_net_timing`
/// (release numbers: swap `--cargo-profile=fast` for `--cargo-profile=release`).
#[test]
#[ignore = "manual Phase-1 timing harness, not a correctness gate"]
fn rvr_preflight_inline_addsub_net_timing() {
    use std::time::Instant;

    use openvm_circuit::arch::DenseRecordArena;

    struct ArmBest {
        execute: f64,
        dense_gen: f64,
        matrix_gen: f64,
        total_dense: f64,
        total_matrix: f64,
    }

    const ITERS: usize = 25;
    let adds = 3000usize; // 2 + 3*3000 + 1 = 9003 instructions, ~9002 AddSub records
    let exe = repeated_adds_exe(adds);

    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("vm init");
    let trace_heights = vec![16384u32; rvr_vm.num_airs()];
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");

    // Compile both arms up front (the env gate is read at compile time), then
    // interleave their measurement iterations so host drift hits both equally.
    let mut instances = Vec::new();
    for (label, env) in [("R1-log", "0"), ("R3-inline", "1")] {
        std::env::set_var("OPENVM_RVR_INLINE_RECORDS", env);
        let route = rvr_vm
            .preflight_routed_instance(&exe)
            .expect("routed preflight instance");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("program must route to RVR preflight");
        };
        instances.push((label, instance));
    }

    // Decompose the records-independent per-call baseline: initial-state
    // construction and the guest-state clone `execute_rvr_preflight` performs.
    {
        let (_, instance) = &instances[0];
        let mut best_init = f64::MAX;
        let mut best_clone = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            let state = instance.create_initial_state(Streams::default());
            best_init = best_init.min(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            let cloned = state.clone();
            best_clone = best_clone.min(t1.elapsed().as_secs_f64());
            std::hint::black_box(&cloned);
        }
        eprintln!(
            "baseline: create_initial_state={:.3}ms state.clone={:.3}ms (min of 5)",
            best_init * 1e3,
            best_clone * 1e3,
        );
    }

    // Structural evidence, printed once per arm: log volume and record source.
    for (label, instance) in &instances {
        let mut output = instance
            .execute_preflight(Streams::default(), None)
            .expect("rvr preflight execution");
        let inline_bytes: usize = output
            .inline_records
            .iter()
            .map(|chip| chip.bytes.len())
            .sum();
        eprintln!(
            "{label}: memory_log_entries={} inline_record_bytes={inline_bytes}",
            output.raw_logs.memory_log.len()
        );
        let arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
            F,
            DenseRecordArena,
        >(&exe, &mut output, &capacities, &pc_to_air_idx)
        .expect("dense generation");
        std::hint::black_box(&arenas);
    }

    let mut bests: Vec<ArmBest> = (0..instances.len())
        .map(|_| ArmBest {
            execute: f64::MAX,
            dense_gen: f64::MAX,
            matrix_gen: f64::MAX,
            total_dense: f64::MAX,
            total_matrix: f64::MAX,
        })
        .collect();
    for _ in 0..ITERS {
        for (arm, (_, instance)) in instances.iter().enumerate() {
            let init_state = instance.create_initial_state(Streams::default());
            let t0 = Instant::now();
            let mut output = instance
                .execute_preflight_from_state_with_capacities(
                    init_state,
                    Some((adds * 3 + 3) as u64),
                    &trace_heights,
                )
                .expect("rvr preflight execution");
            let execute_s = t0.elapsed().as_secs_f64();

            // Dense generation consumes the inline buffers (zero-copy), so time
            // it on this output, then re-execute for the matrix arm.
            let t1 = Instant::now();
            let dense = crate::log_native::generate_rv64im_record_arenas_from_logs::<
                F,
                DenseRecordArena,
            >(&exe, &mut output, &capacities, &pc_to_air_idx)
            .expect("dense generation");
            let dense_s = t1.elapsed().as_secs_f64();
            std::hint::black_box(&dense);

            let init_state2 = instance.create_initial_state(Streams::default());
            let t2 = Instant::now();
            let mut output2 = instance
                .execute_preflight_from_state_with_capacities(
                    init_state2,
                    Some((adds * 3 + 3) as u64),
                    &trace_heights,
                )
                .expect("rvr preflight execution");
            let execute2_s = t2.elapsed().as_secs_f64();
            let t3 = Instant::now();
            let matrix = crate::log_native::generate_rv64im_record_arenas_from_logs::<
                F,
                MatrixRecordArena<F>,
            >(&exe, &mut output2, &capacities, &pc_to_air_idx)
            .expect("matrix generation");
            let matrix_s = t3.elapsed().as_secs_f64();
            std::hint::black_box(&matrix);

            let best = &mut bests[arm];
            best.execute = best.execute.min(execute_s.min(execute2_s));
            best.dense_gen = best.dense_gen.min(dense_s);
            best.matrix_gen = best.matrix_gen.min(matrix_s);
            best.total_dense = best.total_dense.min(execute_s + dense_s);
            best.total_matrix = best.total_matrix.min(execute2_s + matrix_s);
        }
    }
    for (arm, (label, _)) in instances.iter().enumerate() {
        let best = &bests[arm];
        eprintln!(
            "{label}: n_instr={} execute={:.3}ms dense_gen={:.3}ms matrix_gen={:.3}ms \
             total_dense={:.3}ms total_matrix={:.3}ms (min of {ITERS}, interleaved, single-shot)",
            adds * 3 + 3,
            best.execute * 1e3,
            best.dense_gen * 1e3,
            best.matrix_gen * 1e3,
            best.total_dense * 1e3,
            best.total_matrix * 1e3,
        );
    }
    eprintln!(
        "net R1->R3: dense {:.3}ms -> {:.3}ms ({:.2}x); matrix {:.3}ms -> {:.3}ms ({:.2}x)",
        bests[0].total_dense * 1e3,
        bests[1].total_dense * 1e3,
        bests[0].total_dense / bests[1].total_dense,
        bests[0].total_matrix * 1e3,
        bests[1].total_matrix * 1e3,
        bests[0].total_matrix / bests[1].total_matrix,
    );
}

/// Stage-2 oracle: every base RV64IM family, including the W families, uses
/// the chronological delta stream. This is the device-aux ownership contract:
/// W instructions may not retain an arena-native host predecessor path.
/// Host expansion must still produce byte-identical full RV64IM arenas.
#[test]
fn rvr_preflight_delta_with_w_generic_full_rv64im_matrix_matches_assembler() {
    use std::collections::BTreeMap;

    use openvm_circuit::arch::{
        rvr::{preflight::RvrArenaNativeTarget, ChipRecordBuf},
        MatrixRecordArena,
    };

    let exe = full_rv64im_matrix_exe();
    let streams = hard_chip_streams(1);
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("rvr vm init");
    let trace_heights = vec![4096u32; rvr_vm.num_airs()];
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");

    // Arm A: compact wire + host assembler (fused emission opted out; arm F
    // re-enables it below).
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let mut a_output = {
        let RvrPreflightRoute::Rvr(instance) = rvr_vm
            .preflight_routed_instance(&exe)
            .expect("routed preflight instance")
        else {
            panic!("program must route to RVR preflight");
        };
        instance
            .execute_preflight(streams.clone(), None)
            .expect("compact preflight execution")
    };
    let a_instret = a_output.instret;
    let a_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        MatrixRecordArena<F>,
    >(&exe, &mut a_output, &capacities, &pc_to_air_idx)
    .expect("assembler-path record arena generation");

    // Delta arm: all base families stay in the chronological delta stream.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let RvrPreflightRoute::Rvr(f_instance) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("fused routed preflight instance")
    else {
        panic!("fused program must route to RVR preflight");
    };
    let arena_native = f_instance
        .compiled()
        .inline_records()
        .arena_native_airs
        .clone();
    assert!(
        arena_native.iter().all(|(_, geometry)| !matches!(
            geometry.layout,
            openvm_circuit::arch::rvr::ArenaNativeLayout::Alu3(offsets)
                if offsets.w.is_some()
        )),
        "base RV64IM W delta must not retain arena-native host aux paths"
    );
    assert!(f_instance.compiled().inline_records().delta_records);
    for (family, opcodes) in [
        (
            "BaseAluW",
            vec![
                BaseAluWOpcode::ADDW.global_opcode(),
                BaseAluWOpcode::SUBW.global_opcode(),
            ],
        ),
        (
            "ShiftW logical",
            vec![
                ShiftWOpcode::SLLW.global_opcode(),
                ShiftWOpcode::SRLW.global_opcode(),
            ],
        ),
        (
            "ShiftW arithmetic",
            vec![ShiftWOpcode::SRAW.global_opcode()],
        ),
        ("MulW", vec![MulWOpcode::MULW.global_opcode()]),
        (
            "DivRemW",
            vec![
                DivRemWOpcode::DIVW.global_opcode(),
                DivRemWOpcode::DIVUW.global_opcode(),
                DivRemWOpcode::REMW.global_opcode(),
                DivRemWOpcode::REMUW.global_opcode(),
            ],
        ),
    ] {
        let slot = exe
            .program
            .instructions_and_debug_infos
            .iter()
            .position(|entry| {
                entry
                    .as_ref()
                    .is_some_and(|(instruction, _)| opcodes.contains(&instruction.opcode))
            })
            .unwrap_or_else(|| panic!("fixture must contain {family}"));
        let air = pc_to_air_idx[slot].unwrap_or_else(|| panic!("{family} must map to an AIR"));
        assert!(
            f_instance.compiled().inline_records().pc_slots[slot],
            "{family} AIR {air} must emit into the generic delta stream"
        );
        assert!(
            arena_native
                .iter()
                .all(|&(native_air, _)| native_air != air),
            "{family} AIR {air} must not retain arena-native host aux"
        );
    }
    let mut fused_arenas: Vec<(
        usize,
        openvm_circuit::arch::rvr::ArenaNativeGeometry,
        MatrixRecordArena<F>,
    )> = Vec::new();
    let mut targets: BTreeMap<usize, ChipRecordBuf> = BTreeMap::new();
    for &(air, geom) in &arena_native {
        let (arena, target) =
            MatrixRecordArena::<F>::stage_arena_native(capacities[air].0, capacities[air].1, &geom);
        targets.insert(air, target);
        fused_arenas.push((air, geom, arena));
    }
    let f_state = rvr_vm.create_initial_state(&exe, streams);
    let mut f_output = f_instance
        .execute_preflight_from_state_with_arena_targets(
            f_state,
            Some(a_instret),
            &trace_heights,
            &targets,
        )
        .expect("fused preflight execution");
    // The generation loop must accept the all-delta run and assemble every
    // AIR identically.
    let f_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        MatrixRecordArena<F>,
    >(&exe, &mut f_output, &capacities, &pc_to_air_idx)
    .expect("fused-path record arena generation (count oracle)");

    // Direct-final records deliberately suppress their duplicate program-log
    // entries. The generated execution-frequency buffer is the authoritative
    // full-program oracle across both arms.
    assert_eq!(
        a_output.system_records.filtered_exec_frequencies,
        f_output.system_records.filtered_exec_frequencies,
        "execution frequencies must be identical across arms"
    );

    // Byte-equality of every custom fused air's arena over every written row.
    // This fixture has none, but the loop keeps the mixed-route oracle intact.
    let fused_air_set: std::collections::BTreeSet<usize> =
        fused_arenas.iter().map(|&(air, _, _)| air).collect();
    for (fused_air, _, fused_arena) in &fused_arenas {
        let fused_air = *fused_air;
        let rows = a_arenas[fused_air].trace_offset / a_arenas[fused_air].width;
        let written_rows = f_output
            .arena_native_written
            .iter()
            .find(|&&(air, _)| air == fused_air)
            .map(|&(_, count)| count as usize)
            .expect("fused air must report a written count");
        assert_eq!(
            written_rows, rows,
            "air {fused_air}: fused row count must match assembler"
        );
        let n = rows * a_arenas[fused_air].width;
        let assembled = &a_arenas[fused_air].trace_buffer[..n];
        let fused = &fused_arena.trace_buffer[..n];
        if assembled != fused {
            let idx = assembled
                .iter()
                .zip(fused)
                .position(|(left, right)| left != right)
                .expect("different slices have a mismatch");
            panic!(
                "air {fused_air} ({}): fused arena differs at field {idx} (row {}, col {}): assembled={} fused={}",
                rvr_vm.air_names().nth(fused_air).unwrap_or("unknown"),
                idx / a_arenas[fused_air].width,
                idx % a_arenas[fused_air].width,
                assembled[idx],
                fused[idx],
            );
        }
    }

    // Every non-fused air must assemble identically across arms.
    for (air, (a, f)) in a_arenas.iter().zip(f_arenas.iter()).enumerate() {
        if fused_air_set.contains(&air) {
            continue;
        }
        assert_eq!(
            a.trace_offset, f.trace_offset,
            "air {air} row counts must match across arms"
        );
        assert_eq!(
            &a.trace_buffer[..a.trace_offset],
            &f.trace_buffer[..f.trace_offset],
            "air {air} arenas must match across arms"
        );
    }
}

/// R4 fused oracle, Dense flavor (the G1 GPU-staging shape): the fused
/// backing is allocated with `backing_with_capacity`, aimed at the C via
/// `{base = aligned start, stride_dense, core_off_dense}`, adopted zero-copy
/// with `from_prewritten`, and must be byte-identical to the assembler-path
/// DenseRecordArena's `allocated()` region. Fully-C-written Dense records =
/// the upgraded G1 emission.
#[test]
fn rvr_preflight_arena_native_addsub_matches_assembler_dense() {
    use std::collections::BTreeMap;

    use openvm_circuit::arch::{rvr::ChipRecordBuf, DenseRecordArena};

    let exe = inline_addsub_differential_exe();

    // Arm A: compact wire + host assembler into Dense arenas (existing
    // harness; env untouched so the compile stays compact).
    let (a_output, a_arenas, _) = run_inline_addsub_differential_arm(&exe);
    let a_instret = a_output.instret;

    // Arm F: fused compile aiming the AddSub air at a Dense backing.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("fused vm init");
    let trace_heights = vec![4096u32; rvr_vm.num_airs()];
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let RvrPreflightRoute::Rvr(f_instance) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("fused routed preflight instance")
    else {
        panic!("fused program must route to RVR preflight");
    };
    let arena_native = f_instance
        .compiled()
        .inline_records()
        .arena_native_airs
        .clone();
    assert!(!arena_native.is_empty());
    let mut stagings: Vec<(usize, usize, Vec<u8>, *const u8)> = Vec::new();
    let mut targets = BTreeMap::new();
    for &(air, geom) in &arena_native {
        let stride = geom.stride_dense();
        let rows = trace_heights[air] as usize;
        let (mut backing, offset) = DenseRecordArena::backing_with_capacity(rows * stride);
        let base: *const u8 = unsafe { backing.as_mut_ptr().add(offset) };
        targets.insert(
            air,
            ChipRecordBuf {
                base: base.cast_mut(),
                len: 0,
                cap: (rows * stride) as u32,
                stride: stride as u32,
                core_off: geom.core_off_dense() as u32,
                flags: openvm_circuit::arch::rvr::PREFLIGHT_CHIP_RECORD_FLAG_DIRECT_FINAL,
            },
        );
        stagings.push((air, stride, backing, base));
    }
    let f_state = rvr_vm.create_initial_state(&exe, Streams::default());
    let mut f_output = f_instance
        .execute_preflight_from_state_with_arena_targets(
            f_state,
            Some(a_instret),
            &trace_heights,
            &targets,
        )
        .expect("fused preflight execution");
    let f_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        openvm_circuit::arch::DenseRecordArena,
    >(&exe, &mut f_output, &capacities, &pc_to_air_idx)
    .expect("fused-path record arena generation (count oracle)");

    assert_eq!(
        a_output.system_records.filtered_exec_frequencies,
        f_output.system_records.filtered_exec_frequencies,
        "execution frequencies must be identical across arms"
    );

    // Zero-copy adopt each staged backing, then byte-compare against the
    // assembler-path arenas.
    let fused_air_set: std::collections::BTreeSet<usize> =
        stagings.iter().map(|&(air, ..)| air).collect();
    for (fused_air, stride, backing, base) in stagings {
        let written_rows = f_output
            .arena_native_written
            .iter()
            .find(|&&(air, _)| air == fused_air)
            .map(|&(_, count)| count as usize)
            .expect("fused air must report a written count");
        let fused_arena = DenseRecordArena::from_prewritten(backing, base, written_rows * stride);
        assert_eq!(
            a_arenas[fused_air].allocated(),
            fused_arena.allocated(),
            "air {fused_air}: fused Dense records must be byte-identical to assembled"
        );
    }

    // Non-fused airs assemble identically across arms.
    for (air, (a, f)) in a_arenas.iter().zip(f_arenas.iter()).enumerate() {
        if fused_air_set.contains(&air) {
            continue;
        }
        assert_eq!(
            a.allocated(),
            f.allocated(),
            "air {air} arenas must match across arms"
        );
    }
}

/// R4 end-to-end: prove + verify multi-segment THROUGH the prove path with
/// arena-native staging active (the C writes AddSub records into staged
/// arenas inside prove_continuations; finish/substitution included). The
/// strongest fused oracle: the proofs must verify.
#[test]
fn rvr_preflight_arena_native_proves_and_verifies() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    let exe = inline_addsub_differential_exe();
    let segments = prove_rvr_preflight_and_verify(exe, Rv64ImConfig::default());
    assert!(segments >= 1, "expected at least one proven segment");
}

/// HintStore's mixed AIR is a packed variable-row direct-final target: both
/// opcodes migrate together, and the emitted prefix must stay byte-identical
/// to the established verbose assembler for both single and buffer records.
#[test]
fn rvr_preflight_hintstore_direct_final_matches_verbose_twice() {
    use openvm_circuit::{
        arch::rvr::preflight::RvrArenaNativeTarget, system::memory::online::LinearMemory,
    };

    let exe = hintstore_direct_exe();
    let streams = hintstore_direct_streams();
    let config = Rv64ImConfig::default();

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (oracle_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("HintStore oracle vm init");
    let heights = vec![4096u32; oracle_vm.num_airs()];
    let capacities = heights
        .iter()
        .zip(oracle_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = oracle_vm
        .pc_to_air_idx(&exe)
        .expect("HintStore pc-to-air mapping");
    let hint_slots = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .filter_map(|(slot, entry)| {
            entry.as_ref().and_then(|(instruction, _)| {
                (instruction.opcode == Rv64HintStoreOpcode::HINT_STORED.global_opcode()
                    || instruction.opcode == Rv64HintStoreOpcode::HINT_BUFFER.global_opcode())
                .then_some(slot)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        hint_slots.len(),
        2,
        "fixture must cover both HintStore opcodes"
    );
    let hint_air = pc_to_air_idx[hint_slots[0]].expect("HINT_STORED AIR");
    assert_eq!(
        pc_to_air_idx[hint_slots[1]],
        Some(hint_air),
        "HINT_STORED and HINT_BUFFER must share one AIR"
    );
    let RvrPreflightRoute::Rvr(oracle_instance) = oracle_vm
        .preflight_routed_instance(&exe)
        .expect("HintStore oracle route")
    else {
        panic!("HintStore oracle must route to rvr")
    };
    let mut oracle_output = oracle_instance
        .execute_preflight_from_state(oracle_vm.create_initial_state(&exe, streams.clone()), None)
        .expect("HintStore verbose preflight");
    let retired = oracle_output.instret;
    let oracle_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        DenseRecordArena,
    >(&exe, &mut oracle_output, &capacities, &pc_to_air_idx)
    .expect("HintStore verbose record assembly");

    // With no variable arena target, delta compilation must taint the entire
    // mixed AIR back to verbose assembly rather than migrate either opcode.
    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let (tainted_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("HintStore taint vm init");
    let RvrPreflightRoute::Rvr(tainted_instance) = tainted_vm
        .preflight_routed_instance(&exe)
        .expect("HintStore taint route")
    else {
        panic!("HintStore taint arm must route to rvr")
    };
    assert!(
        hint_slots
            .iter()
            .all(|&slot| !tainted_instance.compiled().inline_records().pc_slots[slot]),
        "missing variable target must taint both HintStore opcodes off inline emission"
    );

    for pass in 0..2 {
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
        std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
        let (direct_vm, _) =
            VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
                .expect("HintStore direct vm init");
        let RvrPreflightRoute::Rvr(direct_instance) = direct_vm
            .preflight_routed_instance(&exe)
            .expect("HintStore direct route")
        else {
            panic!("HintStore direct arm must route to rvr")
        };
        let inline = direct_instance.compiled().inline_records();
        assert!(
            hint_slots.iter().all(|&slot| inline.pc_slots[slot]),
            "pass {pass}: both HintStore opcodes must migrate atomically"
        );
        let geometry = inline
            .arena_native_airs
            .iter()
            .find(|&&(air, _)| air == hint_air)
            .map(|&(_, geometry)| geometry)
            .expect("HintStore variable arena geometry");
        assert!(matches!(
            geometry.layout,
            openvm_circuit::arch::rvr::ArenaNativeLayout::CustomVariableRows {
                residual_memory_chronology: true
            }
        ));
        let (mut direct_arena, target) = DenseRecordArena::stage_arena_native(
            heights[hint_air] as usize,
            capacities[hint_air].1,
            &geometry,
        );
        let targets = BTreeMap::from([(hint_air, target)]);
        let mut direct_output = direct_instance
            .execute_preflight_from_state_with_arena_targets(
                direct_vm.create_initial_state(&exe, streams.clone()),
                Some(retired),
                &heights,
                &targets,
            )
            .expect("HintStore direct preflight");
        if pass == 1 {
            let full_memory_log = direct_output.raw_logs.memory_log.clone();
            let (mut compact_arena, compact_target) = DenseRecordArena::stage_arena_native(
                heights[hint_air] as usize,
                capacities[hint_air].1,
                &geometry,
            );
            let compact_targets = BTreeMap::from([(hint_air, compact_target)]);
            let compact_output = direct_instance
                .execute_preflight_from_state_with_compact_delta_memory_for_test(
                    direct_vm.create_initial_state(&exe, streams.clone()),
                    Some(retired),
                    &heights,
                    &compact_targets,
                )
                .expect("HintStore compact direct preflight");
            assert!(
                !full_memory_log.is_empty(),
                "HintStore regression fixture must exercise predecessor-bearing chronology"
            );
            assert!(
                compact_output.raw_logs.memory_log.is_empty()
                    && !compact_output.raw_logs.delta_memory_log.is_empty(),
                "HintStore compact route must retain only the 24-byte residual-memory wire"
            );
            assert_system_records_eq(
                "HintStore compact direct",
                &oracle_output.system_records,
                &compact_output.system_records,
            );
            let (_device_arena, device_target) = DenseRecordArena::stage_arena_native(
                heights[hint_air] as usize,
                capacities[hint_air].1,
                &geometry,
            );
            let device_targets = BTreeMap::from([(hint_air, device_target)]);
            let device_output = direct_instance
                .execute_preflight_from_state_with_device_touched_memory_for_test(
                    direct_vm.create_initial_state(&exe, streams.clone()),
                    Some(retired),
                    &heights,
                    &device_targets,
                )
                .expect("HintStore device-replay routing preflight");
            assert!(device_output.system_records.touched_memory_on_device);
            assert!(
                device_output.system_records.touched_memory.is_empty(),
                "device-owned replay must not collect or sort host touched records"
            );
            assert_eq!(
                device_output.raw_logs.delta_memory_log, compact_output.raw_logs.delta_memory_log,
                "device-owned replay must retain byte-identical compact chronology"
            );
            assert_eq!(
                device_output.raw_logs.touched,
                Vec::new(),
                "production device-owned replay must emit no host first-touch seeds"
            );
            assert!(
                !device_output.raw_logs.program_runs.is_empty(),
                "production device chronology must emit block runs"
            );
            assert!(
                device_output.raw_logs.device_program_references.is_empty(),
                "production device chronology must not retain per-instruction host records"
            );
            assert!(
                device_output
                    .system_records
                    .filtered_exec_frequencies
                    .iter()
                    .all(|&count| count == 0),
                "production device chronology must not compute host program frequencies"
            );
            let memory = &device_output.to_state.memory.memory;
            let memory_as = RV64_MEMORY_AS as usize;
            let dirty_ranges =
                memory.touched_pages[memory_as].touched_byte_ranges(memory.mem[memory_as].size());
            for address in [64usize, 96, 104, 112] {
                assert!(
                    dirty_ranges
                        .iter()
                        .any(|&(start, end)| start <= address && address < end),
                    "device-owned replay must stage the dirty HintStore page containing {address:#x}"
                );
            }
            std::env::set_var("OPENVM_RVR_DEVICE_REPLAY_ORACLE", "1");
            let (_oracle_device_arena, oracle_device_target) = DenseRecordArena::stage_arena_native(
                heights[hint_air] as usize,
                capacities[hint_air].1,
                &geometry,
            );
            let oracle_device_base = oracle_device_target.base as u64;
            let oracle_device_targets = BTreeMap::from([(hint_air, oracle_device_target)]);
            let oracle_device_output = direct_instance
                .execute_preflight_from_state_with_device_touched_memory_for_test(
                    direct_vm.create_initial_state(&exe, streams.clone()),
                    Some(retired),
                    &heights,
                    &oracle_device_targets,
                )
                .expect("HintStore device-replay oracle preflight");
            std::env::remove_var("OPENVM_RVR_DEVICE_REPLAY_ORACLE");
            assert!(
                !oracle_device_output
                    .raw_logs
                    .device_program_references
                    .is_empty(),
                "device chronology oracle must retain a non-vacuous full-vector reference"
            );
            let mut filtered_by_slot =
                Vec::with_capacity(exe.program.instructions_and_debug_infos.len());
            let mut next_filtered = 0u32;
            for instruction in &exe.program.instructions_and_debug_infos {
                filtered_by_slot.push(instruction.as_ref().map(|_| {
                    let index = next_filtered;
                    next_filtered += 1;
                    index
                }));
            }
            let mut reconstructed = Vec::new();
            let mut reconstructed_frequencies = vec![0u32; next_filtered as usize];
            for run in &oracle_device_output.raw_logs.program_runs {
                assert_eq!(run.complete, 1, "device chronology run must be complete");
                assert_eq!(
                    run.chronology_offset as usize,
                    reconstructed.len(),
                    "device chronology runs must be contiguous"
                );
                for local in 0..run.instruction_count {
                    let pc = run.first_pc + local * DEFAULT_PC_STEP;
                    let slot = ((pc - exe.program.pc_base) / DEFAULT_PC_STEP) as usize;
                    let filtered_index = filtered_by_slot
                        .get(slot)
                        .copied()
                        .flatten()
                        .expect("device chronology run referenced an undefined program slot");
                    reconstructed
                        .push(openvm_circuit::arch::rvr::DeviceProgramEntry { pc, filtered_index });
                    reconstructed_frequencies[filtered_index as usize] += 1;
                }
            }
            assert_eq!(
                reconstructed, oracle_device_output.raw_logs.device_program_references,
                "block-run reconstruction must match the complete host chronology vector"
            );
            assert_eq!(
                reconstructed_frequencies,
                oracle_device_output
                    .system_records
                    .filtered_exec_frequencies,
                "block-run reconstruction must match the complete host frequency vector"
            );
            assert_eq!(
                oracle_device_output.raw_logs.device_aux_references.len(),
                oracle_device_output.raw_logs.delta_memory_log.len(),
                "device replay oracle must retain one predecessor per residual event"
            );
            let arena_reference = oracle_device_output
                .raw_logs
                .device_aux_arena_references
                .iter()
                .find(|reference| reference.air_idx == hint_air)
                .expect("HintStore device replay oracle omitted its custom arena");
            assert_eq!(arena_reference.base, oracle_device_base);
            assert!(
                !arena_reference.expected.is_empty(),
                "custom full-arena oracle must not snapshot a placeholder"
            );
            assert_eq!(
                arena_reference.expected.as_slice(),
                oracle_arenas[hint_air].allocated(),
                "custom full-arena oracle must retain the host direct-final bytes"
            );
            let compact_rows = compact_output
                .arena_native_written
                .iter()
                .find(|&&(air, _)| air == hint_air)
                .map(|&(_, rows)| rows as usize)
                .expect("HintStore compact written rows");
            let compact_bytes = compact_output
                .arena_native_written_bytes
                .iter()
                .find(|&&(air, _)| air == hint_air)
                .map(|&(_, bytes)| bytes as usize)
                .expect("HintStore compact written bytes");
            compact_arena.finish_arena_native_sized(compact_rows, compact_bytes, &geometry);
            assert_eq!(
                compact_arena.allocated(),
                oracle_arenas[hint_air].allocated(),
                "HintStore compact direct-final bytes must match the verbose assembler"
            );
        }
        assert_system_records_eq(
            &format!("HintStore direct pass {pass}"),
            &oracle_output.system_records,
            &direct_output.system_records,
        );
        crate::log_native::generate_rv64im_record_arenas_from_logs::<F, DenseRecordArena>(
            &exe,
            &mut direct_output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("HintStore direct residual assembly");
        let written_rows = direct_output
            .arena_native_written
            .iter()
            .find(|&&(air, _)| air == hint_air)
            .map(|&(_, rows)| rows as usize)
            .expect("HintStore written rows");
        let written_bytes = direct_output
            .arena_native_written_bytes
            .iter()
            .find(|&&(air, _)| air == hint_air)
            .map(|&(_, bytes)| bytes as usize)
            .expect("HintStore written bytes");
        assert_eq!(written_rows, 4, "one single row plus three buffer rows");
        assert_eq!(written_bytes, 144, "52-byte single + 92-byte buffer");
        direct_arena.finish_arena_native_sized(written_rows, written_bytes, &geometry);
        assert_eq!(direct_arena.rvr_variable_rows, Some(4));
        assert_eq!(
            direct_arena.allocated(),
            oracle_arenas[hint_air].allocated(),
            "pass {pass}: HintStore direct-final bytes must match the verbose assembler"
        );
    }

    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
}

#[test]
fn rvr_preflight_g2_hintstore_event_replay_is_byte_equal() {
    let exe = hintstore_direct_exe();
    let streams = hintstore_direct_streams();
    let config = Rv64ImConfig::default();

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("HintStore G2 oracle VM init");
    let heights = vec![4096u32; vm.num_airs()];
    let capacities = heights
        .iter()
        .zip(vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = vm.pc_to_air_idx(&exe).expect("HintStore G2 pc mapping");
    let hint_air = pc_to_air_idx[3].expect("HINT_STORED AIR");
    let RvrPreflightRoute::Rvr(oracle) = vm.preflight_routed_instance(&exe).expect("oracle route")
    else {
        panic!("HintStore oracle must route to RVR");
    };
    let mut oracle_output = oracle
        .execute_preflight_from_state(oracle.create_initial_state(streams.clone()), None)
        .expect("HintStore G2 oracle preflight");
    let retired = oracle_output.instret;
    let oracle_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        DenseRecordArena,
    >(&exe, &mut oracle_output, &capacities, &pc_to_air_idx)
    .expect("HintStore G2 oracle arenas");

    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let (g2_vm, _) = VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config)
        .expect("HintStore G2 VM init");
    let RvrPreflightRoute::Rvr(g2) = g2_vm.preflight_routed_instance(&exe).expect("G2 route")
    else {
        panic!("HintStore fixture must route to G2 RVR");
    };
    assert!(g2.compiled().inline_records().g2.is_some());
    assert!(g2
        .compiled()
        .inline_records()
        .arena_native_airs
        .iter()
        .all(|&(air, _)| air != hint_air));
    let mut output = g2
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            g2.create_initial_state(streams),
            Some(retired),
            &heights,
            &BTreeMap::new(),
        )
        .expect("HintStore G2 preflight");
    let segment = output.g2_segment.take().expect("HintStore G2 segment");
    let meta = output.g2_meta.as_deref().expect("HintStore G2 metadata");
    let decode = output
        .delta_decode_precomputed
        .as_deref()
        .expect("HintStore G2 operand table");
    let reference = openvm_circuit::arch::rvr::decode_reference_v1(
        &segment,
        meta,
        decode,
        [0; 32],
        &BTreeMap::new(),
        openvm_circuit::arch::rvr::PREFLIGHT_INITIAL_TIMESTAMP,
    )
    .expect("HintStore G2 CPU event replay");
    let replay = reference
        .compact_records
        .get(&30)
        .expect("HintStore G2 replay rows");
    assert_eq!(replay.len(), 4 * 64);
    let mut dense = Vec::new();
    for row in replay.chunks_exact(64) {
        let local_idx = u32::from_le_bytes(row[8..12].try_into().unwrap());
        if local_idx == 0 {
            dense.extend_from_slice(&row[12..16]);
            dense.extend_from_slice(&row[0..8]);
            dense.extend_from_slice(&row[16..28]);
            dense.extend_from_slice(&row[28..36]);
        }
        dense.extend_from_slice(&row[36..56]);
    }
    assert_eq!(
        dense,
        oracle_arenas[hint_air].allocated(),
        "HintStore event replay must reconstruct the established packed consumer bytes"
    );
    assert_eq!(
        openvm_circuit::arch::rvr::bridge::read_rv64_registers(&output.to_state),
        reference.final_registers
    );
    assert!(output.raw_logs.program_log.is_empty());
    assert!(output.raw_logs.memory_log.is_empty());
    assert!(output.raw_logs.delta_memory_log.is_empty());

    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
}

/// PhantomAir is semantically mixed across every system and extension
/// discriminant. All successful system phantoms and all Rv64I extension
/// phantoms in this configuration must share the same complete-record target;
/// their 20-byte records must exactly match the verbose assembler twice.
#[test]
fn rvr_preflight_phantom_direct_final_matches_verbose_twice() {
    use openvm_circuit::arch::rvr::preflight::RvrArenaNativeTarget;

    let exe = phantom_direct_exe();
    let streams = hint_input_streams();
    let config = Rv64ImConfig::default();

    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (oracle_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("Phantom oracle vm init");
    let heights = vec![4096u32; oracle_vm.num_airs()];
    let capacities = heights
        .iter()
        .zip(oracle_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = oracle_vm
        .pc_to_air_idx(&exe)
        .expect("Phantom pc-to-air mapping");
    let phantom_slots = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .filter_map(|(slot, entry)| {
            entry.as_ref().and_then(|(instruction, _)| {
                (instruction.opcode == SystemOpcode::PHANTOM.global_opcode()).then_some(slot)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(phantom_slots.len(), 6, "fixture discriminator coverage");
    let phantom_air = pc_to_air_idx[phantom_slots[0]].expect("Phantom AIR");
    assert!(
        phantom_slots
            .iter()
            .all(|&slot| pc_to_air_idx[slot] == Some(phantom_air)),
        "every phantom discriminator must share PhantomAir"
    );
    let RvrPreflightRoute::Rvr(oracle_instance) = oracle_vm
        .preflight_routed_instance(&exe)
        .expect("Phantom oracle route")
    else {
        panic!("Phantom oracle must route to rvr")
    };
    let mut oracle_output = oracle_instance
        .execute_preflight_from_state(oracle_vm.create_initial_state(&exe, streams.clone()), None)
        .expect("Phantom verbose preflight");
    let retired = oracle_output.instret;
    let oracle_arenas = crate::log_native::generate_rv64im_record_arenas_from_logs::<
        F,
        DenseRecordArena,
    >(&exe, &mut oracle_output, &capacities, &pc_to_air_idx)
    .expect("Phantom verbose record assembly");

    for pass in 0..2 {
        std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
        std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
        std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
        let (direct_vm, _) =
            VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
                .expect("Phantom direct vm init");
        let RvrPreflightRoute::Rvr(direct_instance) = direct_vm
            .preflight_routed_instance(&exe)
            .expect("Phantom direct route")
        else {
            panic!("Phantom direct arm must route to rvr")
        };
        let inline = direct_instance.compiled().inline_records();
        assert!(
            phantom_slots.iter().all(|&slot| inline.pc_slots[slot]),
            "pass {pass}: every successful Phantom discriminator must migrate atomically"
        );
        let geometry = inline
            .arena_native_airs
            .iter()
            .find(|&&(air, _)| air == phantom_air)
            .map(|&(_, geometry)| geometry)
            .expect("Phantom arena-native geometry");
        assert_eq!(geometry.stride_dense(), 20);
        assert!(matches!(
            geometry.layout,
            openvm_circuit::arch::rvr::ArenaNativeLayout::Custom {
                residual_memory_chronology: true,
                ..
            }
        ));
        let (mut direct_arena, target) = DenseRecordArena::stage_arena_native(
            heights[phantom_air] as usize,
            capacities[phantom_air].1,
            &geometry,
        );
        let targets = BTreeMap::from([(phantom_air, target)]);
        let mut direct_output = direct_instance
            .execute_preflight_from_state_with_arena_targets(
                direct_vm.create_initial_state(&exe, streams.clone()),
                Some(retired),
                &heights,
                &targets,
            )
            .expect("Phantom direct preflight");
        assert_system_records_eq(
            &format!("Phantom direct pass {pass}"),
            &oracle_output.system_records,
            &direct_output.system_records,
        );
        crate::log_native::generate_rv64im_record_arenas_from_logs::<F, DenseRecordArena>(
            &exe,
            &mut direct_output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("Phantom direct residual assembly");
        let written = direct_output
            .arena_native_written
            .iter()
            .find(|&&(air, _)| air == phantom_air)
            .map(|&(_, rows)| rows as usize)
            .expect("Phantom written rows");
        assert_eq!(written, phantom_slots.len());
        direct_arena.finish_arena_native(written, &geometry);
        assert_eq!(
            direct_arena.allocated(),
            oracle_arenas[phantom_air].allocated(),
            "pass {pass}: Phantom direct-final bytes must match the verbose assembler"
        );
    }

    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
}

/// DebugPanic cannot produce a successful Phantom row. Merely having that
/// discriminator in the program must therefore taint every PhantomAir slot
/// back to verbose routing, even when the panic is unreachable at runtime.
#[test]
fn rvr_preflight_phantom_unshaped_discriminator_taints_whole_air() {
    let exe = exe(&[
        phantom(SysPhantom::Nop),
        terminate(),
        phantom(SysPhantom::DebugPanic),
    ]);
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("Phantom taint vm init");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("Phantom taint route")
    else {
        panic!("Phantom taint fixture must route to rvr")
    };
    let inline = instance.compiled().inline_records();
    assert!(
        !inline.pc_slots[0] && !inline.pc_slots[2],
        "one unshaped Phantom discriminator must taint the whole AIR"
    );
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
}

/// Main-memory STORED and REVEAL instructions sharing the doubleword-store AIR
/// must all be arena-native. The four AS=3 rows may never become a secondary
/// source that whole-AIR staging would overwrite.
#[test]
fn rvr_preflight_arena_native_reveal_mixed_air_proves_and_verifies() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    let exe = public_values_reveal_differential_exe();
    let config = Rv64ImConfig::with_public_values_bytes(32);
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("rvr vm init");
    let pc_to_air_idx = vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let store_doubleword_air = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .find(|(_, entry)| {
            entry.as_ref().is_some_and(|(instruction, _)| {
                instruction.opcode == Rv64LoadStoreOpcode::STORED.global_opcode()
            })
        })
        .and_then(|(slot, _)| pc_to_air_idx[slot])
        .expect("STORED maps to an air");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("mixed reveal route")
    else {
        panic!("mixed reveal program must route to rvr")
    };
    let inline = instance.compiled().inline_records();
    assert!(
        inline
            .arena_native_airs
            .iter()
            .any(|&(air, _)| air == store_doubleword_air),
        "doubleword stores must remain arena-native when the program contains REVEAL"
    );
    for (slot, (instruction, _)) in exe
        .program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .filter_map(|(slot, entry)| entry.as_ref().map(|entry| (slot, entry)))
        .filter(|(_, (instruction, _))| instruction.e.as_canonical_u32() == PUBLIC_VALUES_AS)
    {
        assert!(
            inline.pc_slots[slot],
            "REVEAL at program slot {slot} ({:?}) must emit inline",
            instruction.opcode
        );
    }

    let segments = prove_rvr_preflight_and_verify(exe, config);
    assert_eq!(
        segments, 1,
        "small mixed STORED/REVEAL program is single segment"
    );
}

/// G2 precedence + mixed-source invariant: the explicit compact request wins
/// over default-on arena-native emission, and every ordinary AS=2 and REVEAL
/// AS=3 row sharing the doubleword-store AIR remains inline in the compact stream. The host
/// compact assembler is the CPU oracle for the device decoder's expanded
/// arena shape, so proving and verification pin the full record semantics.
#[test]
fn rvr_preflight_compact_request_reveal_mixed_air_proves_and_verifies() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");

    let exe = public_values_reveal_differential_exe();
    let config = Rv64ImConfig::with_public_values_bytes(32);
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("rvr vm init");
    let pc_to_air_idx = vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let store_doubleword_air = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .find(|(_, entry)| {
            entry.as_ref().is_some_and(|(instruction, _)| {
                instruction.opcode == Rv64LoadStoreOpcode::STORED.global_opcode()
            })
        })
        .and_then(|(slot, _)| pc_to_air_idx[slot])
        .expect("STORED maps to an air");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("compact mixed reveal route")
    else {
        panic!("mixed reveal program must route to rvr")
    };
    let inline = instance.compiled().inline_records();
    assert!(
        inline.arena_native_airs.is_empty(),
        "compact request must override every arena-native AIR"
    );
    assert!(
        inline
            .airs
            .iter()
            .any(|&(air, _)| air == store_doubleword_air),
        "doubleword stores must emit compact wire records"
    );
    for (slot, &air) in pc_to_air_idx.iter().enumerate() {
        if air == Some(store_doubleword_air) {
            assert!(
                inline.pc_slots[slot],
                "AS=2/AS=3 doubleword-store row at program slot {slot} must emit compact wire"
            );
        }
    }

    let segments = prove_rvr_preflight_and_verify(exe, config);
    assert_eq!(segments, 1, "mixed STORED/REVEAL fixture is one segment");
}

/// Stage-2 CPU decoder oracle: the chronological 24-byte delta stream omits
/// every access previous-timestamp, then reconstructs and partitions the
/// established compact wires before arena assembly. This fixture combines
/// ordinary AS=2 load/store rows with AS=3 REVEAL rows in the same AIR and
/// proves the reconstructed arena end to end.
#[test]
fn rvr_preflight_delta_request_reveal_mixed_air_proves_and_verifies() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");

    let exe = public_values_reveal_differential_exe();
    let config = Rv64ImConfig::with_public_values_bytes(32);
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Rv64ImCpuBuilder, config.clone())
            .expect("rvr vm init");
    let RvrPreflightRoute::Rvr(instance) = vm
        .preflight_routed_instance(&exe)
        .expect("delta mixed reveal route")
    else {
        panic!("mixed reveal program must route to rvr")
    };
    assert!(instance.compiled().inline_records().delta_records);
    assert!(instance
        .compiled()
        .inline_records()
        .arena_native_airs
        .is_empty());

    let segments = prove_rvr_preflight_and_verify(exe, config);
    assert_eq!(
        segments, 1,
        "mixed delta LoadStore/REVEAL fixture is one segment"
    );
}

/// A segment-boundary checkpoint runs before the next basic block. Delta
/// emission must not reserve that block's record span until the checkpoint
/// admits it; otherwise suspension leaves an unwritten zero-record tail.
#[test]
fn rvr_preflight_delta_multi_segment_does_not_reserve_suspended_block() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");

    let mut config = Rv64ImConfig::default();
    config.rv64i.system.segmentation_max_memory = 1;
    let segments = prove_rvr_preflight_and_verify(repeated_adds_exe(400), config);
    assert!(
        segments > 1,
        "tight segmentation must exercise the delta suspension boundary"
    );
}

/// G2 zero-copy oracle: the same compact-compiled program executed twice —
/// arm A harvests the C-written wire records from the pooled record buffers
/// (the unstaged path), arm B aims every migrated air at a Dense arena staged
/// as a wire target (`stage_rvr_wire`: the C writes packed wire records
/// straight into the 32-aligned backing; adoption is cursor bookkeeping). The
/// staged arenas must be byte-identical to the pooled wire buffers over all
/// four wire formats (the fixture covers alu3/branch2/wr1/rw1 plus the REVEAL
/// loadstore row), must carry `rvr_wire`, and the generation loop's count
/// oracle must accept the staged run.
#[test]
fn rvr_preflight_compact_wire_staged_matches_pooled() {
    use openvm_circuit::arch::rvr::{preflight::RvrArenaNativeTarget, ChipRecordBuf};

    // Exercise the production precedence, not the legacy manual opt-out:
    // an explicit compact GPU request must override default-on arena-native
    // emission. Nextest isolates this process-local environment mutation.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "compact");

    let exe = full_rv64im_matrix_exe();
    let streams = hard_chip_streams(1);
    let (rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("rvr vm init");
    let trace_heights = vec![4096u32; rvr_vm.num_airs()];
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let RvrPreflightRoute::Rvr(instance) = rvr_vm
        .preflight_routed_instance(&exe)
        .expect("routed preflight instance")
    else {
        panic!("program must route to RVR preflight");
    };
    let wire_airs = instance.compiled().inline_records().airs.clone();
    assert!(
        instance.compiled().inline_records().delta_decode.is_some(),
        "compact preflight AOT must persist its CUDA operand/classification artifact"
    );
    let wire_air_set = wire_airs
        .iter()
        .map(|&(air, _)| air)
        .collect::<std::collections::HashSet<_>>();
    let decode_air_set = crate::rvr_gpu_decode::RvrGpuDecodeState::default().bind_compact_segment(
        &exe,
        &pc_to_air_idx,
        &instance.compiled().inline_records().pc_slots,
        instance.compiled().inline_records().delta_decode.as_deref(),
    );
    assert!(
        instance
            .compiled()
            .inline_records()
            .arena_native_airs
            .is_empty(),
        "compact compile must not report arena-native airs"
    );
    assert!(
        decode_air_set.is_subset(&wire_air_set),
        "every independently routed GPU decode AIR must be compiler-inline"
    );
    // ZG2 explicitly encodes the fixture's ordinary AS=2 LoadStore rows and
    // AS=3 REVEAL row in the same compact stream. No mixed-source AIR may be
    // tainted back to host assembly.
    let inline_pc_slots = instance.compiled().inline_records().pc_slots.clone();
    let mixed_airs: std::collections::HashSet<usize> = pc_to_air_idx
        .iter()
        .enumerate()
        .filter_map(|(slot, air)| {
            let air = (*air)?;
            let is_wire_air = wire_airs.iter().any(|&(wire_air, _)| wire_air == air);
            let is_inline_pc = inline_pc_slots.get(slot).copied().unwrap_or(false);
            (is_wire_air && !is_inline_pc).then_some(air)
        })
        .collect();
    assert!(
        mixed_airs.is_empty(),
        "ordinary LoadStore and REVEAL rows must both be compact-decodable"
    );

    // Arm A: pooled record buffers (target-less execute); the wire bytes are
    // the C-written truth this oracle compares against.
    let a_output = instance
        .execute_preflight(streams.clone(), None)
        .expect("pooled compact preflight execution");
    let a_instret = a_output.instret;
    let a_wire: BTreeMap<usize, Vec<u8>> = a_output
        .inline_records
        .iter()
        .map(|chip| (chip.air_idx, chip.bytes.clone()))
        .collect();
    assert_eq!(a_wire.len(), wire_airs.len());
    assert!(
        a_wire.values().any(|bytes| !bytes.is_empty()),
        "fixture must emit wire records"
    );

    // Arm B: every migrated AIR, including mixed-source LoadStore, is staged
    // as a direct-final wire target.
    let mut staged: Vec<(usize, usize, DenseRecordArena)> = Vec::new();
    let mut targets: BTreeMap<usize, ChipRecordBuf> = BTreeMap::new();
    for &(air, wire_size) in &wire_airs {
        if mixed_airs.contains(&air) {
            continue;
        }
        let (arena, buf) = <DenseRecordArena as RvrArenaNativeTarget>::stage_rvr_wire(
            trace_heights[air] as usize,
            wire_size,
        );
        targets.insert(air, buf);
        staged.push((air, wire_size, arena));
    }
    assert!(
        staged.len() >= 10,
        "fixture must stage most migrated airs (got {})",
        staged.len()
    );
    let b_state = rvr_vm.create_initial_state(&exe, streams);
    let mut b_output = instance
        .execute_preflight_from_state_with_arena_targets(
            b_state,
            Some(a_instret),
            &trace_heights,
            &targets,
        )
        .expect("staged wire preflight execution");

    // Direct-final staged records intentionally suppress their duplicate
    // program-log rows. The compile-time-indexed frequency vector remains the
    // complete program-execution oracle across pooled and staged arms.
    assert_eq!(
        a_output.system_records.filtered_exec_frequencies,
        b_output.system_records.filtered_exec_frequencies,
        "program execution frequencies must be identical across arms"
    );

    // The staged run reports counts through the arena-target channel and the
    // generation loop's count oracle must accept it (staged airs skip
    // assembly; every other air assembles normally into full arenas).
    let b_arenas =
        crate::log_native::generate_rv64im_record_arenas_from_logs::<F, DenseRecordArena>(
            &exe,
            &mut b_output,
            &capacities,
            &pc_to_air_idx,
        )
        .expect("staged-path record arena generation (count oracle)");

    for (air, wire_size, mut arena) in staged {
        let bytes = &a_wire[&air];
        assert_eq!(
            bytes.len() % wire_size,
            0,
            "air {air}: pooled wire bytes must be whole records"
        );
        let written = b_output
            .arena_native_written
            .iter()
            .find(|&&(written_air, _)| written_air == air)
            .map(|&(_, count)| count as usize)
            .expect("staged wire air must report a written count");
        assert_eq!(
            written,
            bytes.len() / wire_size,
            "air {air}: staged record count must match the pooled arm"
        );
        arena.finish_rvr_wire(written, wire_size);
        assert!(arena.rvr_wire, "air {air}: staged arena must be wire-mode");
        assert_eq!(
            arena.allocated(),
            bytes.as_slice(),
            "air {air}: staged wire records must be byte-identical to pooled"
        );
        // The placeholder arena the generation loop returned for this air must
        // be empty (the caller substitutes the staged arena).
        assert!(
            b_arenas[air].allocated().is_empty(),
            "air {air}: generation must not write a staged air's arena"
        );
    }

    // Any future mixed/unstaged AIR remains a valid fallback oracle, but this
    // fixture must leave the set empty after explicit REVEAL support.
    for &air in &mixed_airs {
        let b_bytes = b_output
            .inline_records
            .iter()
            .find(|chip| chip.air_idx == air)
            .map(|chip| chip.bytes.as_slice())
            .expect("mixed air must keep a pooled wire buffer");
        assert_eq!(
            b_bytes, a_wire[&air],
            "air {air}: pooled wire bytes must match across arms"
        );
        assert!(
            !b_arenas[air].allocated().is_empty(),
            "air {air}: mixed air must host-expand into a real arena"
        );
    }
}

/// G2 fixture-scale host-write checkpoint (manual): one large AddSub-dominated
/// segment (~8.2M inline records) on the single-shot proving path, three arms
/// over the same program — G1 fused (full records C-written into a staged
/// Dense arena), G2 staged wire (compact records C-written into a staged
/// Dense arena, zero-copy), and the unstaged compact shape (pooled wire
/// buffers + the alloc+memcpy adoption M-GPUDEC measured as INC1). Reports
/// bytes written and wall time per arm. Manual:
/// `cargo nextest run --cargo-profile=fast -p openvm-riscv-circuit \
///  --features rvr --run-ignored only -- rvr_preflight_wire_staged_host_write_checkpoint`
#[test]
#[ignore = "manual G2 host-write checkpoint harness, not a correctness gate"]
fn rvr_preflight_wire_staged_host_write_checkpoint() {
    use std::time::Instant;

    use openvm_circuit::arch::rvr::{
        preflight::RvrArenaNativeTarget, ChipRecordBuf, RvrPreflightBufferPool,
    };

    let (iters_base, iters_shift, body_adds) = (1024usize, 3usize, 1000usize); // 8192 x 1002
    let exe = checkpoint_loop_exe(iters_base, iters_shift, body_adds);
    let dyn_insns = checkpoint_loop_dyn_insns(iters_base, iters_shift, body_adds);
    const ITERS: usize = 5;

    // Execution floors for the same 8.2M-instruction fixture. Pure is the
    // native AOT lower bound; metered retains the page/height bookkeeping
    // needed to size a proving segment but emits no preflight records.
    let (floor_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64ImCpuBuilder,
        Rv64ImConfig::default(),
    )
    .expect("floor vm init");
    let pure = floor_vm
        .executor()
        .rvr_instret_tracking_instance(&exe, None)
        .expect("tracked pure rvr instance");
    let metered = floor_vm
        .metered_interpreter(&exe)
        .expect("metered rvr instance");
    let mut pure_best = f64::MAX;
    let mut metered_best = f64::MAX;
    for _ in 0..ITERS {
        let state = pure.create_initial_vm_state(Streams::default());
        let t0 = Instant::now();
        let state = pure
            .execute_from_state_for(state, dyn_insns)
            .expect("pure rvr execution");
        pure_best = pure_best.min(t0.elapsed().as_secs_f64());
        let _ = std::hint::black_box(state);

        let state = metered.create_initial_vm_state(Streams::default());
        let metered_ctx = floor_vm.build_metered_ctx(&exe);
        let t0 = Instant::now();
        let (segments, state) = metered
            .execute_metered_from_state(state, metered_ctx)
            .expect("metered rvr execution");
        metered_best = metered_best.min(t0.elapsed().as_secs_f64());
        std::hint::black_box((segments, state));
    }
    let mut pc_map_best = f64::MAX;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let map = floor_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
        pc_map_best = pc_map_best.min(t0.elapsed().as_secs_f64());
        std::hint::black_box(map);
    }
    eprintln!(
        "G2_PHASE1_FLOOR dyn_insns={dyn_insns} pure_ms={:.3} metered_ms={:.3} pc_map_ms={:.3}",
        pure_best * 1e3,
        metered_best * 1e3,
        pc_map_best * 1e3,
    );

    let checkpoint_wire_bytes;

    // Arm F (G1 fused): full records at dense stride, C-written into staged
    // arena backings — the Phase C rvr-arm host shape.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    {
        let (rvr_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ImCpuBuilder,
            Rv64ImConfig::default(),
        )
        .expect("fused vm init");
        let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
        let addsub_air = pc_to_air_idx[4].expect("body ADD maps to an air");
        let branch_air = pc_to_air_idx[4 + body_adds + 1].expect("branch maps to an air");
        let mut trace_heights = vec![1u32 << 10; rvr_vm.num_airs()];
        trace_heights[addsub_air] = 1 << 24;
        trace_heights[branch_air] = 1 << 14;
        let RvrPreflightRoute::Rvr(instance) = rvr_vm
            .preflight_routed_instance(&exe)
            .expect("routed preflight instance")
        else {
            panic!("program must route to RVR preflight");
        };
        let arena_native = instance
            .compiled()
            .inline_records()
            .arena_native_airs
            .clone();
        assert!(
            !arena_native.is_empty(),
            "fused compile must be arena-native"
        );
        let mut best = f64::MAX;
        let mut setup_best = f64::MAX;
        let mut execute_best = f64::MAX;
        let mut finish_best = f64::MAX;
        let mut bytes_written = 0usize;
        for _ in 0..ITERS {
            let mut stagings = Vec::new();
            let mut targets = BTreeMap::new();
            let t0 = Instant::now();
            for &(air, geom) in &arena_native {
                let stride = geom.stride_dense();
                let (arena, buf) = <DenseRecordArena as RvrArenaNativeTarget>::stage_arena_native(
                    trace_heights[air] as usize,
                    0,
                    &geom,
                );
                targets.insert(air, buf);
                stagings.push((air, stride, geom, arena));
            }
            let init = instance.create_initial_state(Streams::default());
            let setup_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let output = instance
                .execute_preflight_from_state_with_arena_targets(
                    init,
                    Some(dyn_insns),
                    &trace_heights,
                    &targets,
                )
                .expect("fused preflight execution");
            let execute_s = t1.elapsed().as_secs_f64();
            bytes_written = output
                .arena_native_written
                .iter()
                .map(|&(air, count)| {
                    let stride = stagings
                        .iter()
                        .find(|&&(staged_air, ..)| staged_air == air)
                        .map(|(_, stride, _, _)| *stride)
                        .unwrap();
                    count as usize * stride
                })
                .sum();
            let t2 = Instant::now();
            let mut finished = Vec::new();
            for (air, _, geom, mut arena) in stagings {
                let written = output
                    .arena_native_written
                    .iter()
                    .find(|&&(written_air, _)| written_air == air)
                    .map(|&(_, count)| count as usize)
                    .unwrap();
                arena.finish_arena_native(written, &geom);
                finished.push(arena);
            }
            let finish_s = t2.elapsed().as_secs_f64();
            std::hint::black_box(&finished);
            instance.recycle_output(output);
            setup_best = setup_best.min(setup_s);
            execute_best = execute_best.min(execute_s);
            finish_best = finish_best.min(finish_s);
            best = best.min(setup_s + execute_s + finish_s);
        }
        eprintln!(
            "G2_CHECKPOINT arm=fused_staged(G1) dyn_insns={dyn_insns} bytes={bytes_written} \
             wall_ms={:.2} setup_ms={:.2} execute_emit_ms={:.2} finish_ms={:.3} \
             write_rate={:.2}GB/s",
            best * 1e3,
            setup_best * 1e3,
            execute_best * 1e3,
            finish_best * 1e3,
            bytes_written as f64 / best / 1e9,
        );
    }

    // Arms W (G2 staged wire, zero-copy) + P (pooled wire + adoption memcpy):
    // compact compile.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    {
        let (rvr_vm, _) = VirtualMachine::new_with_keygen(
            test_cpu_engine(),
            Rv64ImCpuBuilder,
            Rv64ImConfig::default(),
        )
        .expect("compact vm init");
        let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
        let addsub_air = pc_to_air_idx[4].expect("body ADD maps to an air");
        let branch_air = pc_to_air_idx[4 + body_adds + 1].expect("branch maps to an air");
        let mut trace_heights = vec![1u32 << 10; rvr_vm.num_airs()];
        trace_heights[addsub_air] = 1 << 24;
        trace_heights[branch_air] = 1 << 14;
        let RvrPreflightRoute::Rvr(instance) = rvr_vm
            .preflight_routed_instance(&exe)
            .expect("routed preflight instance")
        else {
            panic!("program must route to RVR preflight");
        };
        let wire_airs = instance.compiled().inline_records().airs.clone();
        let wire_pool = RvrPreflightBufferPool::from_env();
        for &(air, wire_size) in &wire_airs {
            wire_pool.prepare_wire_backing(air, trace_heights[air] as usize * wire_size);
        }

        // Arm W: staged wire targets, records land in their final backing.
        let mut best = f64::MAX;
        let mut setup_best = f64::MAX;
        let mut execute_best = f64::MAX;
        let mut finish_best = f64::MAX;
        let mut bytes_written = 0usize;
        for _ in 0..ITERS {
            let mut targets: BTreeMap<usize, ChipRecordBuf> = BTreeMap::new();
            let mut staged: Vec<(usize, usize, DenseRecordArena)> = Vec::new();
            let t0 = Instant::now();
            for &(air, wire_size) in &wire_airs {
                let (arena, buf) =
                    <DenseRecordArena as RvrArenaNativeTarget>::stage_rvr_wire_pooled(
                        trace_heights[air] as usize,
                        wire_size,
                        air,
                        &wire_pool,
                    );
                targets.insert(air, buf);
                staged.push((air, wire_size, arena));
            }
            let init = instance.create_initial_state(Streams::default());
            let setup_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let output = instance
                .execute_preflight_from_state_with_arena_targets(
                    init,
                    Some(dyn_insns),
                    &trace_heights,
                    &targets,
                )
                .expect("staged wire preflight execution");
            let execute_s = t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let mut finished = Vec::new();
            for (air, wire_size, mut arena) in staged {
                let written = output
                    .arena_native_written
                    .iter()
                    .find(|&&(written_air, _)| written_air == air)
                    .map(|&(_, count)| count as usize)
                    .unwrap();
                arena.finish_rvr_wire(written, wire_size);
                finished.push(arena);
            }
            let finish_s = t2.elapsed().as_secs_f64();
            std::hint::black_box(&finished);
            bytes_written = output
                .arena_native_written
                .iter()
                .map(|&(air, count)| {
                    let wire_size = wire_airs
                        .iter()
                        .find(|&&(wire_air, _)| wire_air == air)
                        .map(|&(_, size)| size)
                        .unwrap();
                    count as usize * wire_size
                })
                .sum();
            instance.recycle_output(output);
            setup_best = setup_best.min(setup_s);
            execute_best = execute_best.min(execute_s);
            finish_best = finish_best.min(finish_s);
            best = best.min(setup_s + execute_s + finish_s);
        }
        checkpoint_wire_bytes = bytes_written;
        eprintln!(
            "G2_CHECKPOINT arm=wire_staged(G2) dyn_insns={dyn_insns} bytes={bytes_written} \
             wall_ms={:.2} setup_ms={:.2} execute_emit_ms={:.2} finish_ms={:.3} \
             write_rate={:.2}GB/s",
            best * 1e3,
            setup_best * 1e3,
            execute_best * 1e3,
            finish_best * 1e3,
            bytes_written as f64 / best / 1e9,
        );

        // Arm P: pooled wire buffers + the alloc+memcpy adoption (the INC1
        // shape whose fresh-buffer churn M-GPUDEC measured at +86%/seg).
        let mut best = f64::MAX;
        let mut execute_best = f64::MAX;
        let mut adopt_best = f64::MAX;
        let mut bytes_written = 0usize;
        for _ in 0..ITERS {
            let init = instance.create_initial_state(Streams::default());
            let t0 = Instant::now();
            let output = instance
                .execute_preflight_from_state_with_capacities(init, Some(dyn_insns), &trace_heights)
                .expect("pooled compact preflight execution");
            let execute_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let mut adopted = Vec::new();
            for chip in &output.inline_records {
                let mut dense = DenseRecordArena::with_byte_capacity(chip.bytes.len());
                dense
                    .alloc_bytes(chip.bytes.len())
                    .copy_from_slice(&chip.bytes);
                dense.rvr_wire = true;
                adopted.push(dense);
            }
            let adopt_s = t1.elapsed().as_secs_f64();
            bytes_written = output
                .inline_records
                .iter()
                .map(|chip| chip.bytes.len())
                .sum();
            std::hint::black_box(&adopted);
            drop(adopted);
            instance.recycle_output(output);
            execute_best = execute_best.min(execute_s);
            adopt_best = adopt_best.min(adopt_s);
            best = best.min(execute_s + adopt_s);
        }
        eprintln!(
            "G2_CHECKPOINT arm=wire_pooled_memcpy(INC1) dyn_insns={dyn_insns} bytes={bytes_written} \
             wall_ms={:.2} execute_emit_ms={:.2} adopt_ms={:.2} write_rate={:.2}GB/s",
            best * 1e3,
            execute_best * 1e3,
            adopt_best * 1e3,
            bytes_written as f64 / best / 1e9,
        );
    }

    // A warmed contiguous memset/copy over the exact compact payload bounds
    // the raw DRAM-byte term. The checkpoint's hot loop is AddSub-dominated,
    // so its C writes already target one sequential per-AIR stream; a large
    // gap to this floor is per-record compute/store overhead, not scatter.
    assert!(checkpoint_wire_bytes > 0);
    let mut src = vec![0x5au8; checkpoint_wire_bytes];
    let mut dst = vec![0u8; checkpoint_wire_bytes];
    let t0 = Instant::now();
    dst.fill(0xa5);
    let first_touch = t0.elapsed().as_secs_f64();
    let mut fill_best = f64::MAX;
    let mut copy_best = f64::MAX;
    for value in 0..ITERS as u8 {
        let t0 = Instant::now();
        dst.fill(value);
        fill_best = fill_best.min(t0.elapsed().as_secs_f64());

        src[0] = value;
        let t0 = Instant::now();
        dst.copy_from_slice(&src);
        copy_best = copy_best.min(t0.elapsed().as_secs_f64());
        std::hint::black_box(dst[checkpoint_wire_bytes - 1]);
    }
    eprintln!(
        "G2_PHASE1_STREAM bytes={checkpoint_wire_bytes} first_touch_ms={:.3} fill_ms={:.3} \
         fill_GBps={:.2} copy_ms={:.3} copy_GBps={:.2}",
        first_touch * 1e3,
        fill_best * 1e3,
        checkpoint_wire_bytes as f64 / fill_best / 1e9,
        copy_best * 1e3,
        checkpoint_wire_bytes as f64 / copy_best / 1e9,
    );
}
