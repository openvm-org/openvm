use std::{collections::BTreeMap, sync::Arc};

use openvm_circuit::{
    arch::{
        rvr::{
            generate_record_arenas_from_logs, preflight::RvrArenaNativeTarget, RvrPreflightEngine,
            RvrPreflightOutput, RvrPreflightRoute, VmRvrLogNativeExtension,
            PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_WRITE,
        },
        testing::assert_vm_states_equivalent,
        verify_segments, ContinuationVmProver, DenseRecordArena, MatrixRecordArena, Streams,
        VirtualMachine, VmInstance,
    },
    system::SystemRecords,
    utils::test_cpu_engine,
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode,
};
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use openvm_riscv_transpiler::{
    BaseAluImmOpcode, BranchEqualOpcode, Rv64HintStoreOpcode, Rv64LoadStoreOpcode,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{Keccak256Rv64Config, Keccak256Rv64CpuBuilder};

type F = BabyBear;

fn reg(idx: usize) -> usize {
    idx * RV64_REGISTER_NUM_LIMBS
}

fn addi(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(
        BaseAluImmOpcode::ADDI.global_opcode(),
        [reg(rd), reg(rs1), imm, 1, 0],
    )
}

fn store_d(rs2: usize, rs1: usize, offset: usize) -> Instruction<F> {
    Instruction::from_usize(
        Rv64LoadStoreOpcode::STORED.global_opcode(),
        [reg(rs2), reg(rs1), offset, 1, RV64_MEMORY_AS as usize, 1, 0],
    )
}

fn branch_boundary() -> Instruction<F> {
    Instruction::from_usize(
        BranchEqualOpcode::BEQ.global_opcode(),
        [reg(0), reg(0), DEFAULT_PC_STEP as usize, 1, 1],
    )
}

fn xorin(buffer: usize, input: usize, len: usize) -> Instruction<F> {
    Instruction::from_usize(
        XorinOpcode::XORIN.global_opcode(),
        [
            reg(buffer),
            reg(input),
            reg(len),
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    )
}

fn keccakf(buffer: usize) -> Instruction<F> {
    Instruction::from_usize(
        KeccakfOpcode::KECCAKF.global_opcode(),
        [
            reg(buffer),
            0,
            0,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    )
}

fn terminate() -> Instruction<F> {
    Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0)
}

fn keccak_exe(rounds: usize, branch_separated: bool) -> VmExe<F> {
    let mut instructions = vec![
        addi(1, 0, 64),
        addi(2, 0, 512),
        addi(3, 0, 16),
        addi(4, 0, 0x234),
        addi(5, 0, 0x678),
        store_d(4, 1, 0),
        store_d(5, 1, 8),
        store_d(5, 2, 0),
        store_d(4, 2, 8),
    ];
    if branch_separated {
        instructions.push(branch_boundary());
    }
    instructions.push(xorin(1, 2, 3));
    for round in 0..rounds {
        instructions.push(keccakf(1));
        if branch_separated && round + 1 < rounds {
            instructions.push(branch_boundary());
        }
    }
    instructions.push(terminate());
    VmExe::new(Program::from_instructions(&instructions))
}

fn keccak_dynamic_xorin_exe() -> VmExe<F> {
    let mut instructions = vec![
        addi(1, 0, 64),
        addi(2, 0, 512),
        addi(4, 0, 0x234),
        addi(5, 0, 0x678),
        store_d(4, 1, 0),
        store_d(5, 1, 8),
        store_d(5, 2, 0),
        store_d(4, 2, 8),
    ];
    for len in [0, 8, 16, 136] {
        instructions.push(addi(3, 0, len));
        instructions.push(xorin(1, 2, 3));
        instructions.push(xorin(1, 2, 3));
        instructions.push(branch_boundary());
    }
    instructions.push(keccakf(1));
    instructions.push(terminate());
    VmExe::new(Program::from_instructions(&instructions))
}

fn hint_store(opcode: Rv64HintStoreOpcode, num_words: usize, ptr: usize) -> Instruction<F> {
    Instruction::from_usize(
        opcode.global_opcode(),
        [
            if opcode == Rv64HintStoreOpcode::HINT_BUFFER {
                reg(num_words)
            } else {
                0
            },
            reg(ptr),
            0,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    )
}

fn keccak_hintstore_interleaved_exe() -> VmExe<F> {
    VmExe::new(Program::from_instructions(&[
        addi(1, 0, 64),
        addi(2, 0, 512),
        addi(3, 0, 16),
        addi(4, 0, 0x234),
        addi(5, 0, 0x678),
        addi(6, 0, 1024),
        addi(7, 0, 2),
        store_d(4, 1, 0),
        store_d(5, 1, 8),
        store_d(5, 2, 0),
        store_d(4, 2, 8),
        xorin(1, 2, 3),
        hint_store(Rv64HintStoreOpcode::HINT_STORED, 0, 6),
        keccakf(1),
        hint_store(Rv64HintStoreOpcode::HINT_BUFFER, 7, 6),
        xorin(1, 2, 3),
        terminate(),
    ]))
}

fn hintstore_streams() -> Streams {
    let mut streams = Streams::default();
    for word in [
        0x0102_0304_0506_0708u64,
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
    ] {
        streams.hint_stream.extend(word.to_le_bytes());
    }
    streams
}

fn keccak_multi_segment_exe(rounds: usize, spacing: usize) -> VmExe<F> {
    let mut instructions = vec![
        addi(1, 0, 64),
        addi(2, 0, 512),
        addi(3, 0, 16),
        addi(4, 0, 0x234),
        addi(5, 0, 0x678),
        store_d(4, 1, 0),
        store_d(5, 1, 8),
        store_d(5, 2, 0),
        store_d(4, 2, 8),
        branch_boundary(),
        xorin(1, 2, 3),
    ];
    for _ in 0..rounds {
        instructions.push(keccakf(1));
        for idx in 0..spacing {
            instructions.push(addi(6, 6, 1));
            if (idx + 1).is_multiple_of(32) {
                instructions.push(branch_boundary());
            }
        }
    }
    instructions.push(terminate());
    VmExe::new(Program::from_instructions(&instructions))
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

fn assert_trace_values_eq(label: &str, air_name: &str, interp: &[F], rvr: &[F]) {
    if interp == rvr {
        return;
    }
    let mismatch = interp
        .iter()
        .zip(rvr)
        .position(|(left, right)| left != right);
    panic!(
        "{label}: {air_name} values differ: interp_len={} rvr_len={} first_mismatch={mismatch:?}",
        interp.len(),
        rvr.len()
    );
}

fn execute_single_segment_differential(
    label: &str,
    exe: &VmExe<F>,
    config: &Keccak256Rv64Config,
) -> RvrPreflightOutput<F> {
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("vm init");
    let trace_heights = vec![4096; vm.num_airs()];
    let initial_state = vm.create_initial_state(exe, Streams::default());
    let mut interpreter = vm
        .preflight_interpreter(exe)
        .expect("interpreter preflight");
    let interp_output = vm
        .execute_preflight(&mut interpreter, initial_state, None, &trace_heights)
        .expect("interpreter execution");
    let route = vm
        .preflight_routed_instance(exe)
        .expect("routed preflight instance");
    let RvrPreflightRoute::Rvr(instance) = route else {
        panic!("{label}: keccak program must route to rvr preflight");
    };
    let rvr_output = instance
        .execute_preflight(Streams::default(), None)
        .expect("rvr preflight execution");
    assert_system_records_eq(
        label,
        &interp_output.system_records,
        &rvr_output.system_records,
    );
    rvr_output
}

fn assert_trace_segment_matches(
    label: &str,
    exe: &VmExe<F>,
    config: &Keccak256Rv64Config,
    from_state: openvm_circuit::arch::VmState,
    num_insns: Option<u64>,
    trace_heights: &[u32],
) -> (openvm_circuit::arch::VmState, Vec<String>) {
    let (mut interp_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("interpreter vm init");
    let mut interpreter = interp_vm
        .preflight_interpreter(exe)
        .expect("interpreter preflight");
    let interp_output = interp_vm
        .execute_preflight(
            &mut interpreter,
            from_state.clone(),
            num_insns,
            trace_heights,
        )
        .expect("interpreter execution");

    let (mut rvr_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("rvr vm init");
    let air_names = rvr_vm.air_names().map(str::to_owned).collect::<Vec<_>>();
    let route = rvr_vm
        .preflight_routed_instance(exe)
        .expect("routed preflight instance");
    let RvrPreflightRoute::Rvr(instance) = route else {
        panic!("{label}: keccak program must route to rvr preflight");
    };
    let mut rvr_output = instance
        .execute_preflight_from_state(from_state.clone(), num_insns)
        .expect("rvr preflight execution");
    assert_system_records_eq(
        label,
        &interp_output.system_records,
        &rvr_output.system_records,
    );

    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(exe).expect("pc to air mapping");
    let mut registry = openvm_circuit::arch::rvr::LogNativeAssemblerRegistry::new();
    config.extend_rvr_log_native(&mut registry);
    let rvr_record_arenas = generate_record_arenas_from_logs::<F, MatrixRecordArena<F>>(
        &registry,
        exe,
        &mut rvr_output,
        &capacities,
        &pc_to_air_idx,
    )
    .expect("rvr record assembly");

    let interp_cached_program_trace = interp_vm.commit_program_on_device(&exe.program);
    interp_vm.load_program(interp_cached_program_trace);
    interp_vm.transport_init_memory_to_device(&from_state.memory);
    let interp_ctx = interp_vm
        .generate_proving_ctx(interp_output.system_records, interp_output.record_arenas)
        .expect("interpreter trace generation");

    let rvr_cached_program_trace = rvr_vm.commit_program_on_device(&exe.program);
    rvr_vm.load_program(rvr_cached_program_trace);
    rvr_vm.transport_init_memory_to_device(&from_state.memory);
    let rvr_ctx = rvr_vm
        .generate_proving_ctx(rvr_output.system_records, rvr_record_arenas)
        .expect("rvr trace generation");

    // HARD-5: these are the three deterministic extension-owned AIRs. Shared
    // lookup/Poseidon2 periphery rows are excluded from raw row-order equality;
    // SystemRecords (including touched_memory) remain compared above.
    let is_keccak_air = |air_idx: &usize| {
        let name = &air_names[*air_idx];
        name.contains("KeccakfOp") || name.contains("KeccakfPerm") || name.contains("XorinVm")
    };
    let mut interp_traces = interp_ctx
        .per_trace
        .into_iter()
        .filter(|(air_idx, _)| is_keccak_air(air_idx))
        .collect::<BTreeMap<_, _>>();
    let mut rvr_traces = rvr_ctx
        .per_trace
        .into_iter()
        .filter(|(air_idx, _)| is_keccak_air(air_idx))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        interp_traces.keys().collect::<Vec<_>>(),
        rvr_traces.keys().collect::<Vec<_>>(),
        "{label}: active AIR set"
    );

    let mut active_keccak_airs = Vec::new();
    for air_idx in interp_traces.keys().copied().collect::<Vec<_>>() {
        let interp = interp_traces.remove(&air_idx).unwrap();
        let rvr = rvr_traces.remove(&air_idx).unwrap();
        let air_name = &air_names[air_idx];
        assert_eq!(
            interp.common_main.width, rvr.common_main.width,
            "{label}: {air_name} width"
        );
        assert_trace_values_eq(
            label,
            air_name,
            &interp.common_main.values,
            &rvr.common_main.values,
        );
        assert_eq!(
            interp.public_values, rvr.public_values,
            "{label}: {air_name} public values"
        );
        active_keccak_airs.push(air_name.clone());
    }

    (rvr_output.to_state, active_keccak_airs)
}

fn assert_keccakf_write_only_accesses(exe: &VmExe<F>, output: &RvrPreflightOutput<F>) {
    let mut checked = 0;
    for program_entry in &output.raw_logs.program_log {
        let instruction = &exe.program.instructions_and_debug_infos
            [(program_entry.pc() / DEFAULT_PC_STEP) as usize]
            .as_ref()
            .unwrap()
            .0;
        if instruction.opcode != KeccakfOpcode::KECCAKF.global_opcode() {
            continue;
        }
        checked += 1;
        let timestamp = program_entry.timestamp;
        let accesses = output
            .raw_logs
            .memory_log
            .iter()
            .filter(|entry| entry.timestamp >= timestamp && entry.timestamp < timestamp + 26)
            .collect::<Vec<_>>();
        assert_eq!(accesses.len(), 26, "keccakf access count");
        assert_eq!(accesses[0].timestamp, timestamp);
        assert_eq!(accesses[0].kind, PREFLIGHT_MEMORY_KIND_READ);
        assert_eq!(accesses[0].addr_space, RV64_REGISTER_AS as u8);
        for (idx, entry) in accesses[1..].iter().enumerate() {
            assert_eq!(entry.timestamp, timestamp + 1 + idx as u32);
            assert_eq!(entry.kind, PREFLIGHT_MEMORY_KIND_WRITE);
            assert_eq!(entry.addr_space, RV64_MEMORY_AS as u8);
        }
    }
    assert!(checked > 0, "expected at least one keccakf instruction");
}

#[test]
fn rvr_preflight_keccak_single_segment_system_records_and_timestamps_match() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let exe = keccak_exe(2, false);
    let output = execute_single_segment_differential(
        "keccak_single_segment",
        &exe,
        &Keccak256Rv64Config::default(),
    );
    assert_keccakf_write_only_accesses(&exe, &output);
}

#[test]
fn rvr_preflight_keccak_single_segment_traces_match_interpreter() {
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let exe = keccak_exe(2, false);
    let config = Keccak256Rv64Config::default();
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("vm init");
    let trace_heights = vec![4096; vm.num_airs()];
    let from_state = vm.create_initial_state(&exe, Streams::default());
    let (_, active_keccak_airs) = assert_trace_segment_matches(
        "keccak_single_segment_trace",
        &exe,
        &config,
        from_state,
        None,
        &trace_heights,
    );
    assert_eq!(
        active_keccak_airs.len(),
        3,
        "KeccakfOp, KeccakfPerm, and Xorin traces must all be active: {active_keccak_airs:?}"
    );
}

#[test]
fn rvr_preflight_keccak_delta_direct_final_matches_host_assembler_bytes() {
    let exe = keccak_dynamic_xorin_exe();
    let config = Keccak256Rv64Config::default();

    // Oracle arm: disable inline records and retain the established verbose
    // log + host assembler path.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (oracle_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("oracle vm init");
    let trace_heights = vec![4096u32; oracle_vm.num_airs()];
    let capacities = trace_heights
        .iter()
        .zip(oracle_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = oracle_vm.pc_to_air_idx(&exe).expect("oracle pc mapping");
    let RvrPreflightRoute::Rvr(oracle_instance) = oracle_vm
        .preflight_routed_instance(&exe)
        .expect("oracle route")
    else {
        panic!("oracle program must route to rvr")
    };
    let mut oracle_output = oracle_instance
        .execute_preflight_from_state(
            oracle_vm.create_initial_state(&exe, Streams::default()),
            None,
        )
        .expect("oracle preflight");
    let oracle_instret = oracle_output.instret;
    let mut oracle_registry = openvm_circuit::arch::rvr::LogNativeAssemblerRegistry::new();
    config.extend_rvr_log_native(&mut oracle_registry);
    let oracle_arenas = generate_record_arenas_from_logs::<F, DenseRecordArena>(
        &oracle_registry,
        &exe,
        &mut oracle_output,
        &capacities,
        &pc_to_air_idx,
    )
    .expect("oracle record assembly");
    let keccakf_air = oracle_vm
        .air_names()
        .position(|name| name.contains("KeccakfOp"))
        .expect("KeccakfOp air");
    let xorin_air = oracle_vm
        .air_names()
        .position(|name| name.contains("XorinVm"))
        .expect("XorinVm air");

    // Direct-final arm: delta remains enabled for the ordinary fixed-shape
    // instructions while KeccakF writes its complete wide record into the
    // consumer arena. Its accesses remain in the residual memory chronology.
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "delta");
    let (direct_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("direct-final vm init");
    let RvrPreflightRoute::Rvr(direct_instance) = direct_vm
        .preflight_routed_instance(&exe)
        .expect("direct-final route")
    else {
        panic!("direct-final program must route to rvr")
    };
    let native_airs = direct_instance
        .compiled()
        .inline_records()
        .arena_native_airs
        .clone();
    assert_eq!(
        native_airs.len(),
        2,
        "only the two Keccak AIRs are custom-native"
    );
    assert!(native_airs.iter().any(|&(air, _)| air == keccakf_air));
    assert!(native_airs.iter().any(|&(air, _)| air == xorin_air));

    let mut staged = Vec::new();
    let mut targets = BTreeMap::new();
    for &(air, geometry) in &native_airs {
        let (arena, target) = DenseRecordArena::stage_arena_native(
            trace_heights[air] as usize,
            capacities[air].1,
            &geometry,
        );
        targets.insert(air, target);
        staged.push((air, geometry, arena));
    }
    let mut direct_output = direct_instance
        .execute_preflight_from_state_with_arena_targets(
            direct_vm.create_initial_state(&exe, Streams::default()),
            Some(oracle_instret),
            &trace_heights,
            &targets,
        )
        .expect("direct-final preflight");
    assert_system_records_eq(
        "Keccakf direct-final",
        &oracle_output.system_records,
        &direct_output.system_records,
    );

    let mut direct_registry = openvm_circuit::arch::rvr::LogNativeAssemblerRegistry::new();
    config.extend_rvr_log_native(&mut direct_registry);
    generate_record_arenas_from_logs::<F, DenseRecordArena>(
        &direct_registry,
        &exe,
        &mut direct_output,
        &capacities,
        &pc_to_air_idx,
    )
    .expect("direct-final residual assembly and delta decode");

    let written = direct_output
        .arena_native_written
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    for (air, geometry, mut direct_arena) in staged {
        direct_arena.finish_arena_native(written[&air] as usize, &geometry);
        assert_eq!(
            oracle_arenas[air].allocated(),
            direct_arena.allocated(),
            "air {air}: direct-final Keccak records must be byte-identical to host assembly"
        );
    }

    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
}

#[test]
fn rvr_preflight_g2_opaque_keccak_xorin_is_byte_equal() {
    let exe = keccak_hintstore_interleaved_exe();
    let streams = hintstore_streams();
    let config = Keccak256Rv64Config::default();

    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    std::env::set_var("OPENVM_RVR_INLINE_RECORDS", "0");
    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    let (oracle_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("G2 opaque oracle VM init");
    let heights = vec![4096u32; oracle_vm.num_airs()];
    let capacities = heights
        .iter()
        .zip(oracle_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = oracle_vm.pc_to_air_idx(&exe).expect("G2 opaque pc mapping");
    let RvrPreflightRoute::Rvr(oracle) = oracle_vm
        .preflight_routed_instance(&exe)
        .expect("G2 opaque oracle route")
    else {
        panic!("G2 opaque oracle must route to RVR");
    };
    let mut oracle_output = oracle
        .execute_preflight_from_state(oracle.create_initial_state(streams.clone()), None)
        .expect("G2 opaque oracle preflight");
    let retired = oracle_output.instret;
    let oracle_arenas = generate_record_arenas_from_logs::<F, DenseRecordArena>(
        &{
            let mut registry = openvm_circuit::arch::rvr::LogNativeAssemblerRegistry::new();
            config.extend_rvr_log_native(&mut registry);
            registry
        },
        &exe,
        &mut oracle_output,
        &capacities,
        &pc_to_air_idx,
    )
    .expect("G2 opaque oracle arenas");

    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "1");
    std::env::set_var("OPENVM_RVR_GPU_RECORDS", "g2");
    let (g2_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config)
            .expect("G2 opaque VM init");
    let RvrPreflightRoute::Rvr(g2) = g2_vm
        .preflight_routed_instance(&exe)
        .expect("G2 opaque route")
    else {
        panic!("Keccak/Xorin fixture must route to G2 RVR");
    };
    let inline = g2.compiled().inline_records();
    let meta = inline.g2.as_deref().expect("G2 opaque negotiation");
    assert_eq!(meta.opaque_bindings.len(), 2);
    let mut staged = Vec::new();
    let mut targets = BTreeMap::new();
    for &(air, geometry) in &inline.arena_native_airs {
        let (arena, target) = DenseRecordArena::stage_arena_native(
            heights[air] as usize,
            capacities[air].1,
            &geometry,
        );
        targets.insert(air, target);
        staged.push((air, geometry, arena));
    }
    let mut output = g2
        .execute_preflight_from_state_with_device_touched_memory_for_test(
            g2.create_initial_state(streams),
            Some(retired),
            &heights,
            &targets,
        )
        .expect("G2 opaque preflight");
    assert!(
        output.raw_logs.device_aux_patches.is_empty(),
        "G2 opaque direct-final records must not defer {} predecessor patches",
        output.raw_logs.device_aux_patches.len()
    );
    let segment = output.g2_segment.take().expect("G2 opaque segment");
    let decode = output
        .delta_decode_precomputed
        .as_deref()
        .expect("G2 opaque operand table");
    let reference = openvm_circuit::arch::rvr::decode_reference_v1(
        &segment,
        meta,
        decode,
        [0; 32],
        &BTreeMap::new(),
        openvm_circuit::arch::rvr::PREFLIGHT_INITIAL_TIMESTAMP,
    )
    .expect("G2 opaque chronology replay");
    assert_eq!(
        openvm_circuit::arch::rvr::bridge::read_rv64_registers(&output.to_state),
        reference.final_registers
    );
    let hint_air = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .zip(&pc_to_air_idx)
        .find_map(|(instruction, &air)| {
            instruction.as_ref().and_then(|(instruction, _)| {
                (instruction.opcode == Rv64HintStoreOpcode::HINT_STORED.global_opcode())
                    .then_some(air)
            })
        })
        .flatten()
        .expect("interleaved HINT_STORED AIR");
    let replay = reference
        .compact_records
        .get(&30)
        .expect("interleaved HintStore replay rows");
    let mut hint_dense = Vec::new();
    for row in replay.chunks_exact(64) {
        let local_idx = u32::from_le_bytes(row[8..12].try_into().unwrap());
        if local_idx == 0 {
            hint_dense.extend_from_slice(&row[12..16]);
            hint_dense.extend_from_slice(&row[0..8]);
            hint_dense.extend_from_slice(&row[16..28]);
            hint_dense.extend_from_slice(&row[28..36]);
        }
        hint_dense.extend_from_slice(&row[36..56]);
    }
    assert_eq!(
        hint_dense,
        oracle_arenas[hint_air].allocated(),
        "interleaved HintStore replay must match the established packed consumer bytes"
    );
    for (air, geometry, mut arena) in staged {
        let written = output
            .arena_native_written
            .iter()
            .find(|&&(written_air, _)| written_air == air)
            .map(|&(_, count)| count as usize)
            .expect("G2 opaque written count");
        let written_bytes = output
            .arena_native_written_bytes
            .iter()
            .find(|&&(written_air, _)| written_air == air)
            .map(|&(_, bytes)| bytes as usize)
            .expect("G2 opaque written bytes");
        arena.finish_arena_native_sized(written, written_bytes, &geometry);
        let first_mismatch = arena
            .allocated()
            .iter()
            .zip(oracle_arenas[air].allocated())
            .position(|(g2, oracle)| g2 != oracle);
        assert_eq!(
            arena.allocated(),
            oracle_arenas[air].allocated(),
            "G2 opaque AIR {air} must pass final consumer bytes unchanged; first mismatch: {first_mismatch:?}"
        );
    }
    assert!(output.raw_logs.program_log.is_empty());
    assert!(output.raw_logs.memory_log.is_empty());
    assert!(output.raw_logs.delta_memory_log.is_empty());

    std::env::remove_var("OPENVM_RVR_GPU_RECORDS");
    std::env::remove_var("OPENVM_RVR_ARENA_NATIVE");
    std::env::remove_var("OPENVM_RVR_INLINE_RECORDS");
}

#[test]
fn rvr_preflight_keccak_multi_segment_traces_and_system_records_match() {
    // Direct execute without arena targets: compile compact (fused emission
    // is default-on and has no target-less fallback).
    std::env::set_var("OPENVM_RVR_ARENA_NATIVE", "0");
    let exe = keccak_multi_segment_exe(4, 400);
    let mut config = Keccak256Rv64Config::default();
    config.system.segmentation_max_memory = 1;
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("metered vm init");
    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered = vm.metered_interpreter(&exe).expect("metered interpreter");
    let (segments, _) = metered
        .execute_metered(Streams::default(), metered_ctx)
        .expect("metered execution");
    assert!(segments.len() > 1, "expected multiple segments");

    let mut state = vm.create_initial_state(&exe, Streams::default());
    let mut keccak_segments = 0;
    for (idx, segment) in segments.into_iter().enumerate() {
        let (to_state, active_keccak_airs) = assert_trace_segment_matches(
            &format!("keccak_multi_segment_{idx}"),
            &exe,
            &config,
            state,
            Some(segment.num_insns),
            &segment.trace_heights,
        );
        if active_keccak_airs
            .iter()
            .any(|name| name.contains("KeccakfOp"))
        {
            keccak_segments += 1;
        }
        state = to_state;
    }
    assert!(
        keccak_segments > 1,
        "expected keccak traces in multiple segments"
    );
}

#[test]
fn rvr_keccak_pure_and_metered_match_interpreter() {
    let exe = keccak_exe(2, true);
    let config = Keccak256Rv64Config::default();
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Keccak256Rv64CpuBuilder, config.clone())
            .expect("vm init");
    let pure_rvr = vm
        .interpreter(&exe)
        .expect("rvr pure instance")
        .execute(Streams::default(), None)
        .expect("rvr pure execution");
    let pure_interpreter = vm
        .naive_interpreter(&exe)
        .expect("pure interpreter")
        .execute(Streams::default(), None)
        .expect("pure interpreter execution");
    assert_vm_states_equivalent(&pure_rvr, &pure_interpreter);

    let (rvr_segments, metered_rvr) = vm
        .get_metered_rvr_instance(&exe)
        .expect("rvr metered instance")
        .execute_metered(Streams::default(), vm.build_metered_ctx(&exe))
        .expect("rvr metered execution");
    let (interpreter_segments, metered_interpreter) = vm
        .naive_metered_interpreter(&exe)
        .expect("metered interpreter")
        .execute_metered(Streams::default(), vm.build_metered_ctx(&exe))
        .expect("metered interpreter execution");
    assert_vm_states_equivalent(&metered_rvr, &metered_interpreter);
    assert_eq!(rvr_segments.len(), interpreter_segments.len());
    for (rvr, interpreter) in rvr_segments.iter().zip(&interpreter_segments) {
        assert_eq!(rvr.instret_start, interpreter.instret_start);
        assert_eq!(rvr.num_insns, interpreter.num_insns);
        assert_eq!(rvr.trace_heights, interpreter.trace_heights);
    }
}

#[test]
fn rvr_preflight_keccak_multi_segment_proves_and_verifies() {
    let exe = keccak_multi_segment_exe(3, 400);
    let mut config = Keccak256Rv64Config::default();
    config.system.segmentation_max_memory = 1;
    let engine = test_cpu_engine();
    let (vm, pk) =
        VirtualMachine::new_with_keygen(engine, Keccak256Rv64CpuBuilder, config).expect("vm init");
    assert!(
        vm.preflight_routed_instance(&exe)
            .expect("route keccak program")
            .is_rvr(),
        "registered keccak opcodes must route to rvr"
    );
    let vk = pk.get_vk();
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance =
        VmInstance::new(vm, Arc::new(exe), cached_program_trace).expect("instance init");
    // CPU proving defaults to the interpreter engine; pin rvr so the route
    // assertion above matches what the prove actually exercises.
    instance.set_rvr_preflight_engine(Some(RvrPreflightEngine::Rvr));
    let proof = ContinuationVmProver::prove(&mut instance, Streams::default()).expect("prove");
    assert!(
        proof.per_segment.len() > 1,
        "expected multiple proof segments"
    );
    verify_segments(&instance.vm.engine, &vk, &proof.per_segment).expect("verify segments");
}
