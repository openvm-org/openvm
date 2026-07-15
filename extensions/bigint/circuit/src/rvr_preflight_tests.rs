use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        rvr::{
            generate_record_arenas_from_logs, LogNativeAssemblerRegistry, RvrPreflightEngine,
            RvrPreflightRoute, VmRvrLogNativeExtension,
        },
        verify_segments, ContinuationVmProver, MatrixRecordArena, Streams, VirtualMachine,
        VmInstance,
    },
    system::SystemRecords,
    utils::test_cpu_engine,
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode,
    MulOpcode, Rv64LoadStoreOpcode, ShiftOpcode,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{Int256Rv64Config, Int256Rv64CpuBuilder};

type F = BabyBear;

fn reg(idx: usize) -> usize {
    idx * RV64_REGISTER_NUM_LIMBS
}

fn addi(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(
        BaseAluImmOpcode::ADDI.global_opcode(),
        [reg(rd), reg(rs1), imm, RV64_REGISTER_AS as usize, 0],
    )
}

fn store_word(rs2: usize, rs1: usize, offset: usize) -> Instruction<F> {
    Instruction::from_usize(
        Rv64LoadStoreOpcode::STORED.global_opcode(),
        [
            reg(rs2),
            reg(rs1),
            offset,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
            1,
            0,
        ],
    )
}

fn int256_alu(opcode: impl LocalOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(
        opcode.global_opcode(),
        [
            reg(rd),
            reg(rs1),
            reg(rs2),
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    )
}

fn int256_branch(
    opcode: impl LocalOpcode,
    rs1: usize,
    rs2: usize,
    offset: usize,
) -> Instruction<F> {
    Instruction::from_usize(
        opcode.global_opcode(),
        [
            reg(rs1),
            reg(rs2),
            offset,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    )
}

fn terminate() -> Instruction<F> {
    Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0)
}

fn exe(instructions: Vec<Instruction<F>>) -> VmExe<F> {
    VmExe::new(Program::from_instructions(&instructions))
}

/// Covers all eight Int256 AIRs. The first branch ends the initial basic block after 19 retired
/// instructions, which provides a stable block-aligned boundary for the two-segment differential.
fn int256_vector_exe() -> VmExe<F> {
    let mut instructions = vec![
        addi(1, 0, 64),
        addi(2, 0, 96),
        addi(3, 0, 128),
        addi(4, 0, 5),
        store_word(4, 1, 0),
        addi(4, 0, 3),
        store_word(4, 2, 0),
    ];
    instructions.extend([
        int256_alu(Rv64BaseAlu256Opcode(BaseAluOpcode::ADD), 3, 1, 2),
        int256_alu(Rv64BaseAlu256Opcode(BaseAluOpcode::SUB), 3, 1, 2),
        int256_alu(Rv64BaseAlu256Opcode(BaseAluOpcode::XOR), 3, 1, 2),
        int256_alu(Rv64BaseAlu256Opcode(BaseAluOpcode::OR), 3, 1, 2),
        int256_alu(Rv64BaseAlu256Opcode(BaseAluOpcode::AND), 3, 1, 2),
        int256_alu(Rv64LessThan256Opcode(LessThanOpcode::SLT), 3, 1, 2),
        int256_alu(Rv64LessThan256Opcode(LessThanOpcode::SLTU), 3, 1, 2),
        int256_alu(Rv64Mul256Opcode(MulOpcode::MUL), 3, 1, 2),
        int256_alu(Rv64Shift256Opcode(ShiftOpcode::SLL), 3, 1, 2),
        int256_alu(Rv64Shift256Opcode(ShiftOpcode::SRL), 3, 1, 2),
        int256_alu(Rv64Shift256Opcode(ShiftOpcode::SRA), 3, 1, 2),
        int256_branch(Rv64BranchEqual256Opcode(BranchEqualOpcode::BEQ), 1, 2, 8),
        addi(5, 0, 1),
        int256_branch(
            Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BLT),
            1,
            2,
            8,
        ),
        addi(5, 0, 2),
        terminate(),
    ]);
    exe(instructions)
}

fn repeated_int256_exe(repeats: usize) -> VmExe<F> {
    let mut instructions = vec![
        addi(1, 0, 64),
        addi(2, 0, 96),
        addi(3, 0, 128),
        addi(4, 0, 5),
        store_word(4, 1, 0),
        addi(4, 0, 3),
        store_word(4, 2, 0),
    ];
    for _ in 0..repeats {
        instructions.push(int256_alu(
            Rv64BaseAlu256Opcode(BaseAluOpcode::ADD),
            3,
            1,
            2,
        ));
        // A branch after every Int256 operation keeps block-aligned segmentation complete.
        instructions.push(int256_branch(
            Rv64BranchEqual256Opcode(BranchEqualOpcode::BEQ),
            1,
            1,
            4,
        ));
    }
    instructions.push(terminate());
    exe(instructions)
}

fn assert_system_records_eq(label: &str, interpreter: &SystemRecords<F>, rvr: &SystemRecords<F>) {
    assert_eq!(
        interpreter.from_state, rvr.from_state,
        "{label}: from_state"
    );
    assert_eq!(interpreter.to_state, rvr.to_state, "{label}: to_state");
    assert_eq!(interpreter.exit_code, rvr.exit_code, "{label}: exit_code");
    assert_eq!(
        interpreter.filtered_exec_frequencies, rvr.filtered_exec_frequencies,
        "{label}: filtered execution frequencies"
    );
    assert_eq!(
        interpreter.touched_memory, rvr.touched_memory,
        "{label}: touched memory"
    );
}

fn is_int256_opcode(opcode: usize) -> bool {
    Rv64BaseAlu256Opcode::iter()
        .map(|opcode| opcode.global_opcode())
        .chain(Rv64Shift256Opcode::iter().map(|opcode| opcode.global_opcode()))
        .chain(Rv64LessThan256Opcode::iter().map(|opcode| opcode.global_opcode()))
        .chain(Rv64BranchEqual256Opcode::iter().map(|opcode| opcode.global_opcode()))
        .chain(Rv64BranchLessThan256Opcode::iter().map(|opcode| opcode.global_opcode()))
        .chain(Rv64Mul256Opcode::iter().map(|opcode| opcode.global_opcode()))
        .any(|candidate| candidate.as_usize() == opcode)
}

fn int256_air_ids(exe: &VmExe<F>, pc_to_air_idx: &[Option<usize>]) -> BTreeSet<usize> {
    let ids = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .zip(pc_to_air_idx)
        .filter_map(|(slot, &air_idx)| {
            let (instruction, _) = slot.as_ref()?;
            is_int256_opcode(instruction.opcode.as_usize()).then_some(air_idx?)
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), 8, "all eight Int256 AIRs must be mapped");
    ids
}

fn assert_single_segment_trace_matches_interpreter(exe: VmExe<F>) {
    let config = Int256Rv64Config::default();
    let (mut interpreter_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Int256Rv64CpuBuilder, config.clone())
            .expect("interpreter vm init");
    let trace_heights = vec![4096; interpreter_vm.num_airs()];
    let interpreter_cached_program_trace = interpreter_vm.commit_program_on_device(&exe.program);
    interpreter_vm.load_program(interpreter_cached_program_trace);
    let interpreter_initial_state = interpreter_vm.create_initial_state(&exe, Streams::default());
    interpreter_vm.transport_init_memory_to_device(&interpreter_initial_state.memory);
    let mut interpreter = interpreter_vm
        .preflight_interpreter(&exe)
        .expect("interpreter preflight");
    let interpreter_output = interpreter_vm
        .execute_preflight(
            &mut interpreter,
            interpreter_initial_state,
            None,
            &trace_heights,
        )
        .expect("interpreter execution");

    let (mut rvr_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Int256Rv64CpuBuilder, config.clone())
            .expect("rvr vm init");
    let air_names = rvr_vm.air_names().map(str::to_owned).collect::<Vec<_>>();
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let int256_air_ids = int256_air_ids(&exe, &pc_to_air_idx);
    let rvr_initial_state = rvr_vm.create_initial_state(&exe, Streams::default());
    let mut rvr_output = {
        let route = rvr_vm.preflight_routed_instance(&exe).expect("rvr route");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("Int256 program must route to rvr");
        };
        instance
            .execute_preflight(Streams::default(), None)
            .expect("rvr execution")
    };
    assert_system_records_eq(
        "int256 single segment",
        &interpreter_output.system_records,
        &rvr_output.system_records,
    );

    let mut registry = LogNativeAssemblerRegistry::<F, MatrixRecordArena<F>>::new();
    config.extend_rvr_log_native(&mut registry);
    let rvr_arenas = generate_record_arenas_from_logs(
        &registry,
        &exe,
        &mut rvr_output,
        &capacities,
        &pc_to_air_idx,
    )
    .expect("rvr record assembly");

    let interpreter_ctx = interpreter_vm
        .generate_proving_ctx(
            interpreter_output.system_records,
            interpreter_output.record_arenas,
        )
        .expect("interpreter trace generation");
    let rvr_cached_program_trace = rvr_vm.commit_program_on_device(&exe.program);
    rvr_vm.load_program(rvr_cached_program_trace);
    rvr_vm.transport_init_memory_to_device(&rvr_initial_state.memory);
    let rvr_ctx = rvr_vm
        .generate_proving_ctx(rvr_output.system_records, rvr_arenas)
        .expect("rvr trace generation");

    // HARD-5: byte-compare only opcode-mapped Int256 AIRs. Shared lookup and
    // Poseidon2 periphery rows are order-independent and are covered by the
    // SystemRecords/touched-memory equality above instead of raw row order.
    let mut interpreter_traces = interpreter_ctx
        .per_trace
        .into_iter()
        .filter(|(air_idx, _)| int256_air_ids.contains(air_idx))
        .collect::<BTreeMap<_, _>>();
    let mut rvr_traces = rvr_ctx
        .per_trace
        .into_iter()
        .filter(|(air_idx, _)| int256_air_ids.contains(air_idx))
        .collect::<BTreeMap<_, _>>();
    let interpreter_air_ids = interpreter_traces.keys().copied().collect::<Vec<_>>();
    assert_eq!(
        interpreter_air_ids,
        rvr_traces.keys().copied().collect::<Vec<_>>(),
        "Int256 trace AIR set"
    );
    assert_eq!(
        interpreter_air_ids,
        int256_air_ids.iter().copied().collect::<Vec<_>>(),
        "all Int256 traces must be active"
    );

    for air_idx in interpreter_air_ids {
        let interpreter_trace = interpreter_traces.remove(&air_idx).unwrap();
        let rvr_trace = rvr_traces.remove(&air_idx).unwrap();
        let air_name = air_names
            .get(air_idx)
            .map(String::as_str)
            .unwrap_or("<unknown air>");
        assert_eq!(
            interpreter_trace.common_main.values, rvr_trace.common_main.values,
            "{air_name}: main trace values"
        );
        assert_eq!(
            interpreter_trace.public_values, rvr_trace.public_values,
            "{air_name}: public values"
        );
    }
}

fn assert_two_segment_system_records_match_interpreter(exe: VmExe<F>) {
    const FIRST_SEGMENT_INSNS: u64 = 19;

    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Int256Rv64CpuBuilder,
        Int256Rv64Config::default(),
    )
    .expect("vm init");
    let trace_heights = vec![4096; vm.num_airs()];
    let mut interpreter = vm
        .preflight_interpreter(&exe)
        .expect("interpreter preflight");
    let initial_state = vm.create_initial_state(&exe, Streams::default());
    let interpreter_first = vm
        .execute_preflight(
            &mut interpreter,
            initial_state.clone(),
            Some(FIRST_SEGMENT_INSNS),
            &trace_heights,
        )
        .expect("first interpreter segment");

    let route = vm.preflight_routed_instance(&exe).expect("rvr route");
    let RvrPreflightRoute::Rvr(instance) = route else {
        panic!("Int256 program must route to rvr");
    };
    let rvr_first = instance
        .execute_preflight_from_state(initial_state, Some(FIRST_SEGMENT_INSNS))
        .expect("first rvr segment");
    assert!(rvr_first.suspended, "first rvr segment must suspend");
    assert_eq!(rvr_first.instret, FIRST_SEGMENT_INSNS);
    assert_system_records_eq(
        "int256 multi segment first",
        &interpreter_first.system_records,
        &rvr_first.system_records,
    );

    let interpreter_second = vm
        .execute_preflight(
            &mut interpreter,
            interpreter_first.to_state,
            None,
            &trace_heights,
        )
        .expect("second interpreter segment");
    let rvr_second = instance
        .execute_preflight_from_state(rvr_first.to_state, None)
        .expect("second rvr segment");
    assert!(!rvr_second.suspended, "second rvr segment must terminate");
    assert_system_records_eq(
        "int256 multi segment second",
        &interpreter_second.system_records,
        &rvr_second.system_records,
    );
}

fn prove_and_verify(exe: VmExe<F>, config: Int256Rv64Config) -> usize {
    let engine = test_cpu_engine();
    let (vm, pk) =
        VirtualMachine::new_with_keygen(engine, Int256Rv64CpuBuilder, config).expect("vm init");
    let vk = pk.get_vk();
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance =
        VmInstance::new(vm, Arc::new(exe), cached_program_trace).expect("instance init");
    // CPU proving defaults to the interpreter engine; this fixture proves
    // rvr-generated records, so pin the engine explicitly.
    instance.set_rvr_preflight_engine(Some(RvrPreflightEngine::Rvr));
    let proof = ContinuationVmProver::prove(&mut instance, Streams::default()).expect("prove");
    verify_segments(&instance.vm.engine, &vk, &proof.per_segment).expect("verify segments");
    proof.per_segment.len()
}

#[test]
fn rvr_preflight_int256_trace_and_system_records_match_interpreter() {
    // Direct execute without arena targets: compile compact (fused emission
    // is default-on and has no target-less fallback).
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let exe = int256_vector_exe();
    assert_single_segment_trace_matches_interpreter(exe.clone());
    assert_two_segment_system_records_match_interpreter(exe);
}

#[test]
fn rvr_preflight_int256_proves_single_and_multi_segment() {
    assert_eq!(
        prove_and_verify(int256_vector_exe(), Int256Rv64Config::default()),
        1
    );

    let mut config = Int256Rv64Config::default();
    config.system.segmentation_max_memory = 1;
    // Segmentation checks run every 1000 instructions. This program exceeds that cadence and has
    // a branch after each Int256 op, so the tight limit can cut at an exact block boundary.
    let segments = prove_and_verify(repeated_int256_exe(600), config);
    assert!(
        segments > 1,
        "tight segmentation memory limit must force multiple segments"
    );
}
