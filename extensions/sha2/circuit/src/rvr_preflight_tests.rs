use std::collections::BTreeMap;

use openvm_circuit::{
    arch::{
        rvr::{
            generate_record_arenas_from_logs, LogNativeAssemblerRegistry, RvrPreflightRoute,
            VmRvrLogNativeExtension, PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_WRITE,
        },
        testing::assert_vm_states_equivalent,
        Streams, VirtualMachine, VmState,
    },
    system::SystemRecords,
    utils::test_cpu_engine,
};
use openvm_cpu_backend::CpuBackend;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    instruction::Instruction,
    program::Program,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_transpiler::{BaseAluOpcode, BranchEqualOpcode};
use openvm_sha2_air::{Sha256Config, Sha2BlockHasherSubairConfig};
use openvm_sha2_transpiler::Rv64Sha2Opcode;
use openvm_stark_backend::{
    prover::{AirProvingContext, ProvingContext},
    StarkEngine,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};
use sha2::{Digest, Sha256};

use crate::{
    add_padding_to_message, Sha2Config, Sha2MainChipConfig, Sha2Rv64Config, Sha2Rv64CpuBuilder,
};

type F = BabyBear;
type CpuProvingCtx = ProvingContext<CpuBackend<BabyBearPoseidon2Config>>;
type CpuAirProvingCtx = AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>;

const STATE_PTR: u32 = 0x1000;
const INPUT_PTR: u32 = 0x2000;
const STATE_REG: usize = 1;
const INPUT_REG: usize = 3;
const SHA256_TIMESTAMP_DELTA: u32 = 19;

fn reg(idx: usize) -> usize {
    idx * RV64_REGISTER_NUM_LIMBS
}

fn sha256() -> Instruction<F> {
    Instruction::from_usize(
        Rv64Sha2Opcode::SHA256.global_opcode(),
        [
            reg(STATE_REG),
            reg(STATE_REG),
            reg(INPUT_REG),
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    )
}

fn addi(rd: usize, rs1: usize, imm: usize) -> Instruction<F> {
    Instruction::from_usize(
        BaseAluOpcode::ADD.global_opcode(),
        [reg(rd), reg(rs1), imm, 1, 0],
    )
}

fn block_boundary() -> Instruction<F> {
    Instruction::from_usize(
        BranchEqualOpcode::BEQ.global_opcode(),
        [reg(0), reg(0), 4, 1, 1],
    )
}

fn terminate() -> Instruction<F> {
    Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0)
}

fn insert_bytes(image: &mut SparseMemoryImage, address_space: u32, ptr: u32, bytes: &[u8]) {
    image.extend(
        bytes
            .iter()
            .enumerate()
            .map(|(idx, &byte)| ((address_space, ptr + idx as u32), byte)),
    );
}

fn sha256_message_exe(message: &[u8], branch_separated: bool) -> (VmExe<F>, usize) {
    let padded = add_padding_to_message::<Sha256Config>(message.to_vec());
    let num_blocks = padded.len() / Sha256Config::BLOCK_U8S;
    assert_eq!(
        num_blocks,
        (message.len() + 1 + Sha256Config::MESSAGE_LENGTH_BITS / 8)
            .div_ceil(Sha256Config::BLOCK_U8S),
        "SHA-256 padded block-count derivation"
    );

    let mut instructions = Vec::with_capacity(num_blocks * 3 + 1);
    for block_idx in 0..num_blocks {
        instructions.push(sha256());
        if block_idx + 1 < num_blocks {
            instructions.push(addi(INPUT_REG, INPUT_REG, Sha256Config::BLOCK_U8S));
            if branch_separated {
                instructions.push(block_boundary());
            }
        }
    }
    instructions.push(terminate());

    let mut init_memory = SparseMemoryImage::new();
    insert_bytes(
        &mut init_memory,
        RV64_REGISTER_AS,
        reg(STATE_REG) as u32,
        &u64::from(STATE_PTR).to_le_bytes(),
    );
    insert_bytes(
        &mut init_memory,
        RV64_REGISTER_AS,
        reg(INPUT_REG) as u32,
        &u64::from(INPUT_PTR).to_le_bytes(),
    );
    let initial_state = Sha256Config::get_h()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    insert_bytes(&mut init_memory, RV64_MEMORY_AS, STATE_PTR, &initial_state);
    insert_bytes(&mut init_memory, RV64_MEMORY_AS, INPUT_PTR, &padded);

    (
        VmExe::new(Program::from_instructions(&instructions)).with_init_memory(init_memory),
        num_blocks,
    )
}

fn repeated_sha256_block_exe(num_blocks: usize) -> VmExe<F> {
    let mut instructions = vec![sha256(); num_blocks];
    instructions.push(terminate());

    let mut init_memory = SparseMemoryImage::new();
    insert_bytes(
        &mut init_memory,
        RV64_REGISTER_AS,
        reg(STATE_REG) as u32,
        &u64::from(STATE_PTR).to_le_bytes(),
    );
    insert_bytes(
        &mut init_memory,
        RV64_REGISTER_AS,
        reg(INPUT_REG) as u32,
        &u64::from(INPUT_PTR).to_le_bytes(),
    );
    let initial_state = Sha256Config::get_h()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    insert_bytes(&mut init_memory, RV64_MEMORY_AS, STATE_PTR, &initial_state);
    insert_bytes(
        &mut init_memory,
        RV64_MEMORY_AS,
        INPUT_PTR,
        &[0u8; Sha256Config::BLOCK_U8S],
    );
    VmExe::new(Program::from_instructions(&instructions)).with_init_memory(init_memory)
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

fn assert_sha256_access_schedule(
    label: &str,
    exe: &VmExe<F>,
    output: &openvm_circuit::arch::rvr::RvrPreflightOutput<F>,
) -> usize {
    let program_log = &output.raw_logs.program_log;
    let mut num_sha256_entries = 0;
    for (idx, entry) in program_log.iter().enumerate() {
        let instruction_idx = ((entry.pc as u32 - exe.program.pc_base) / 4) as usize;
        let Some((instruction, _)) =
            exe.program.instructions_and_debug_infos[instruction_idx].as_ref()
        else {
            continue;
        };
        if instruction.opcode != Rv64Sha2Opcode::SHA256.global_opcode() {
            continue;
        }
        num_sha256_entries += 1;
        let next_timestamp = program_log
            .get(idx + 1)
            .map_or(output.system_records.to_state.timestamp, |next| {
                next.timestamp
            });
        assert_eq!(
            next_timestamp - entry.timestamp,
            SHA256_TIMESTAMP_DELTA,
            "{label}: SHA-256 timestamp delta"
        );

        let accesses = output
            .raw_logs
            .memory_log
            .iter()
            .filter(|access| {
                access.timestamp >= entry.timestamp && access.timestamp < next_timestamp
            })
            .collect::<Vec<_>>();
        assert_eq!(
            accesses.len(),
            SHA256_TIMESTAMP_DELTA as usize,
            "{label}: SHA-256 memory-log count"
        );
        assert!(accesses[..3].iter().all(|access| {
            access.kind == PREFLIGHT_MEMORY_KIND_READ && access.addr_space == RV64_REGISTER_AS as u8
        }));
        assert!(accesses[3..11].iter().all(|access| {
            access.kind == PREFLIGHT_MEMORY_KIND_READ
                && access.addr_space == RV64_MEMORY_AS as u8
                && access.address >= u64::from(INPUT_PTR)
        }));
        assert!(accesses[11..15].iter().all(|access| {
            access.kind == PREFLIGHT_MEMORY_KIND_READ
                && access.addr_space == RV64_MEMORY_AS as u8
                && access.address >= u64::from(STATE_PTR)
                && access.address < u64::from(STATE_PTR + Sha256Config::STATE_BYTES as u32)
        }));
        assert!(accesses[15..19].iter().all(|access| {
            access.kind == PREFLIGHT_MEMORY_KIND_WRITE
                && access.addr_space == RV64_MEMORY_AS as u8
                && access.address >= u64::from(STATE_PTR)
                && access.address < u64::from(STATE_PTR + Sha256Config::STATE_BYTES as u32)
        }));
    }
    assert!(
        num_sha256_entries > 0,
        "{label}: access-schedule assertion must cover SHA-256 entries"
    );
    num_sha256_entries
}

fn collect_trace_map(
    ctx: CpuProvingCtx,
    num_system_airs: usize,
) -> BTreeMap<usize, CpuAirProvingCtx> {
    ctx.per_trace
        .into_iter()
        .filter(|(air_idx, _)| *air_idx >= num_system_airs)
        .collect()
}

fn assert_trace_maps_eq(
    label: &str,
    air_names: &[String],
    mut interp: BTreeMap<usize, CpuAirProvingCtx>,
    mut rvr: BTreeMap<usize, CpuAirProvingCtx>,
) {
    let interp_ids = interp.keys().copied().collect::<Vec<_>>();
    let rvr_ids = rvr.keys().copied().collect::<Vec<_>>();
    assert_eq!(interp_ids, rvr_ids, "{label}: trace AIR set");
    assert!(!interp_ids.is_empty(), "{label}: expected extension traces");

    for air_idx in interp_ids {
        let interp_air = interp.remove(&air_idx).unwrap();
        let rvr_air = rvr.remove(&air_idx).unwrap();
        let air_name = air_names
            .get(air_idx)
            .map(String::as_str)
            .unwrap_or("<unknown air>");
        assert_eq!(
            interp_air.common_main.width, rvr_air.common_main.width,
            "{label}: {air_name} width"
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

fn assert_sha256_trace_heights(
    label: &str,
    air_names: &[String],
    traces: &BTreeMap<usize, CpuAirProvingCtx>,
    num_blocks: usize,
) {
    let trace_height = |needle: &str| {
        let air_idx = air_names
            .iter()
            .position(|name| name == needle)
            .unwrap_or_else(|| panic!("{label}: missing {needle}"));
        let trace = traces
            .get(&air_idx)
            .unwrap_or_else(|| panic!("{label}: inactive {needle}"));
        trace.common_main.values.len() / trace.common_main.width
    };
    assert_eq!(
        trace_height("Sha2MainAir<Sha256Config>"),
        num_blocks.next_power_of_two(),
        "{label}: one main row per logged SHA-256 block"
    );
    assert_eq!(
        trace_height("Sha2BlockHasherVmAir<Sha256Config>"),
        (num_blocks * Sha256Config::ROWS_PER_BLOCK).next_power_of_two(),
        "{label}: ROWS_PER_BLOCK times the logged SHA-256 block count"
    );
}

fn assert_segment_trace_matches_interpreter(
    label: &str,
    exe: &VmExe<F>,
    config: &Sha2Rv64Config,
    from_state: Option<VmState<F>>,
    num_insns: Option<u64>,
    trace_heights: &[u32],
) -> VmState<F> {
    let (mut interp_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Sha2Rv64CpuBuilder, config.clone())
            .expect("interpreter vm init");
    let interp_cached_program_trace = interp_vm.commit_program_on_device(&exe.program);
    interp_vm.load_program(interp_cached_program_trace);
    let interp_state =
        from_state.unwrap_or_else(|| interp_vm.create_initial_state(exe, Streams::default()));
    let segment_initial_state = interp_state.clone();
    interp_vm.transport_init_memory_to_device(&interp_state.memory);
    let mut interpreter = interp_vm
        .preflight_interpreter(exe)
        .expect("interpreter preflight");
    let interp_output = interp_vm
        .execute_preflight(&mut interpreter, interp_state, num_insns, trace_heights)
        .expect("interpreter execution");

    let (mut rvr_vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Sha2Rv64CpuBuilder, config.clone())
            .expect("rvr vm init");
    let air_names = rvr_vm.air_names().map(str::to_owned).collect::<Vec<_>>();
    let num_system_airs = rvr_vm.config().system.num_airs();
    let capacities = trace_heights
        .iter()
        .zip(rvr_vm.pk().per_air.iter())
        .map(|(&height, pk)| (height as usize, pk.vk.params.width.main_width()))
        .collect::<Vec<_>>();
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(exe).expect("pc to air mapping");
    let route = rvr_vm
        .preflight_routed_instance(exe)
        .expect("routed preflight instance");
    let RvrPreflightRoute::Rvr(instance) = route else {
        panic!("{label}: SHA-256 program must route to rvr preflight");
    };
    let rvr_output = instance
        .execute_preflight_from_state(segment_initial_state.clone(), num_insns)
        .expect("rvr preflight execution");
    assert_system_records_eq(
        label,
        &interp_output.system_records,
        &rvr_output.system_records,
    );
    let num_sha256_entries = assert_sha256_access_schedule(label, exe, &rvr_output);

    let mut registry = LogNativeAssemblerRegistry::new();
    config.extend_rvr_log_native(&mut registry);
    let rvr_record_arenas =
        generate_record_arenas_from_logs(&registry, exe, &rvr_output, &capacities, &pc_to_air_idx)
            .expect("rvr log-native record assembly");

    let interp_ctx = interp_vm
        .generate_proving_ctx(interp_output.system_records, interp_output.record_arenas)
        .expect("interpreter trace generation");

    let rvr_cached_program_trace = rvr_vm.commit_program_on_device(&exe.program);
    rvr_vm.load_program(rvr_cached_program_trace);
    rvr_vm.transport_init_memory_to_device(&segment_initial_state.memory);
    let next_state = rvr_output.to_state;
    let rvr_ctx = rvr_vm
        .generate_proving_ctx(rvr_output.system_records, rvr_record_arenas)
        .expect("rvr trace generation");

    let interp_traces = collect_trace_map(interp_ctx, num_system_airs);
    let rvr_traces = collect_trace_map(rvr_ctx, num_system_airs);
    assert_sha256_trace_heights(label, &air_names, &rvr_traces, num_sha256_entries);
    assert_trace_maps_eq(label, &air_names, interp_traces, rvr_traces);
    next_state
}

fn expected_digest_state(message: &[u8]) -> [u8; Sha256Config::STATE_BYTES] {
    Sha256::digest(message)
        .chunks_exact(size_of::<u32>())
        .flat_map(|word| word.iter().rev().copied())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn assert_final_digest(label: &str, state: &VmState<F>, expected: &[u8]) {
    let digest = unsafe {
        state
            .memory
            .read_bytes::<{ Sha256Config::STATE_BYTES }>(RV64_MEMORY_AS, STATE_PTR)
    };
    assert_eq!(digest, expected, "{label}: digest state");
}

#[test]
fn rvr_preflight_sha256_single_and_multi_block_traces_match_interpreter() {
    for (label, message) in [
        ("single_block", vec![0x42; 55]),
        ("multi_block", (0..120).map(|i| i as u8).collect()),
    ] {
        let (exe, num_blocks) = sha256_message_exe(&message, false);
        assert_eq!(num_blocks, if label == "single_block" { 1 } else { 3 });
        let config = Sha2Rv64Config::default();
        let (vm, _) =
            VirtualMachine::new_with_keygen(test_cpu_engine(), Sha2Rv64CpuBuilder, config.clone())
                .expect("capacity vm init");
        let trace_heights = vec![8192; vm.num_airs()];
        let next_state = assert_segment_trace_matches_interpreter(
            label,
            &exe,
            &config,
            None,
            None,
            &trace_heights,
        );
        assert_final_digest(label, &next_state, &expected_digest_state(&message));
    }
}

#[test]
fn rvr_preflight_sha256_multi_segment_traces_match_interpreter() {
    let message = (0..25_000).map(|i| (i * 31) as u8).collect::<Vec<_>>();
    let (exe, num_blocks) = sha256_message_exe(&message, true);
    assert!(
        num_blocks > 300,
        "fixture must exercise many SHA-256 blocks"
    );

    let mut config = Sha2Rv64Config::default();
    config.system.segmentation_max_memory = 1;
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Sha2Rv64CpuBuilder, config.clone())
            .expect("metered vm init");
    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered = vm.metered_interpreter(&exe).expect("metered rvr instance");
    let (segments, _) = metered
        .execute_metered(Streams::default(), metered_ctx)
        .expect("metered execution");
    assert!(
        segments.len() > 1,
        "tight memory limit and block boundaries must force multiple segments"
    );

    let mut state = vm.create_initial_state(&exe, Streams::default());
    for (idx, segment) in segments.iter().enumerate() {
        state = assert_segment_trace_matches_interpreter(
            &format!("multi_segment_{idx}"),
            &exe,
            &config,
            Some(state),
            Some(segment.num_insns),
            &segment.trace_heights,
        );
    }
    assert_final_digest(
        "multi_segment_final",
        &state,
        &expected_digest_state(&message),
    );
}

#[test]
fn rvr_sha256_pure_and_metered_match_interpreter() {
    let message = (0..120).map(|i| (i * 7) as u8).collect::<Vec<_>>();
    let (exe, _) = sha256_message_exe(&message, true);
    let config = Sha2Rv64Config::default();
    let dimensions = config.system.memory_config.memory_dimensions();
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), Sha2Rv64CpuBuilder, config.clone())
            .expect("vm init");
    let expected = expected_digest_state(&message);

    let pure_rvr = vm
        .interpreter(&exe)
        .expect("pure rvr instance")
        .execute(Streams::default(), None)
        .expect("pure execution");
    let pure_interpreter = vm
        .naive_interpreter(&exe)
        .expect("pure interpreter")
        .execute(Streams::default(), None)
        .expect("pure interpreter execution");
    assert_vm_states_equivalent(&pure_rvr, &pure_interpreter, &dimensions);
    assert_final_digest("pure", &pure_rvr, &expected);

    let (rvr_segments, metered_rvr) = vm
        .get_metered_rvr_instance(&exe)
        .expect("metered rvr instance")
        .execute_metered(Streams::default(), vm.build_metered_ctx(&exe))
        .expect("metered execution");
    let (interpreter_segments, metered_interpreter) = vm
        .naive_metered_interpreter(&exe)
        .expect("metered interpreter")
        .execute_metered(Streams::default(), vm.build_metered_ctx(&exe))
        .expect("metered interpreter execution");
    assert_vm_states_equivalent(&metered_rvr, &metered_interpreter, &dimensions);
    assert_eq!(rvr_segments.len(), interpreter_segments.len());
    for (rvr, interpreter) in rvr_segments.iter().zip(&interpreter_segments) {
        assert_eq!(rvr.instret_start, interpreter.instret_start);
        assert_eq!(rvr.num_insns, interpreter.num_insns);
        assert_eq!(rvr.trace_heights, interpreter.trace_heights);
    }
    assert_final_digest("metered", &metered_rvr, &expected);
}

#[test]
fn rvr_metered_documents_sha256_block_aligned_height_overflow() {
    const NUM_BLOCKS: usize = 1001;
    const TEST_MAX_TRACE_HEIGHT: u32 = 1 << 11;

    let exe = repeated_sha256_block_exe(NUM_BLOCKS);
    let mut params = openvm_stark_backend::SystemParams::new_for_testing(11);
    params.max_constraint_degree = 3;
    let engine: openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine =
        StarkEngine::new(params);
    let (vm, _) =
        VirtualMachine::new_with_keygen(engine, Sha2Rv64CpuBuilder, Sha2Rv64Config::default())
            .expect("small-height vm init");
    let pc_to_air_idx = vm.pc_to_air_idx(&exe).expect("pc to air mapping");
    let main_air_idx = pc_to_air_idx[0].expect("SHA-256 main AIR");
    let block_hasher_air_idx = main_air_idx + 1;

    let metered_ctx = vm.build_metered_ctx(&exe);
    let (segments, _) = vm
        .metered_interpreter(&exe)
        .expect("metered rvr instance")
        .execute_metered(Streams::default(), metered_ctx)
        .expect("metered execution");
    assert_eq!(
        segments.len(),
        2,
        "the rvr compiler caps generated blocks at 1,000 instructions"
    );
    assert_eq!(
        segments[0].num_insns, 1000,
        "the first generated block has no valid interior checkpoint"
    );
    assert_eq!(
        segments[0].trace_heights[block_hasher_air_idx],
        (1000 * Sha256Config::ROWS_PER_BLOCK) as u32
    );
    assert!(
        segments[0].trace_heights[block_hasher_air_idx] > TEST_MAX_TRACE_HEIGHT,
        "the returned block-aligned segment explicitly exceeds the configured height limit"
    );
}
