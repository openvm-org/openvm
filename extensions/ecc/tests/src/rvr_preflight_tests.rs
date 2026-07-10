use std::{
    collections::{BTreeMap, BTreeSet},
    str::FromStr,
};

use eyre::Result;
use num_bigint::BigUint;
use openvm_algebra_transpiler::ModularTranspilerExtension;
use openvm_circuit::{
    arch::{
        rvr::{
            generate_record_arenas_from_logs, LogNativeAssemblerRegistry, RvrPreflightOutput,
            RvrPreflightRoute, VmRvrLogNativeExtension,
        },
        MatrixRecordArena, Streams, VirtualMachine,
    },
    system::SystemRecords,
    utils::{test_cpu_engine, test_system_config},
};
use openvm_ecc_circuit::{
    CurveConfig, Rv64WeierstrassConfig, Rv64WeierstrassCpuBuilder, ECC_BLOCKS_32, ECC_BLOCKS_48,
    SECP256K1_CONFIG,
};
use openvm_ecc_transpiler::{EccTranspilerExtension, Rv64WeierstrassOpcode};
use openvm_instructions::{exe::VmExe, LocalOpcode};
use openvm_riscv_transpiler::{
    Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_toolchain_tests::{build_example_program_at_path_with_features, get_programs_dir};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

type F = BabyBear;
const WEIERSTRASS_OPCODE_COUNT: usize = 4;

fn secp256k1_config() -> Rv64WeierstrassConfig {
    let mut config = Rv64WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone()]);
    *config.as_mut() = test_system_config();
    config
}

fn build_ecc_exe(config: &Rv64WeierstrassConfig) -> Result<VmExe<F>> {
    let elf =
        build_example_program_at_path_with_features(get_programs_dir!(), "ec", ["k256"], config)?;
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv64ITranspilerExtension)
            .with_extension(Rv64MTranspilerExtension)
            .with_extension(Rv64IoTranspilerExtension)
            .with_extension(EccTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    )?)
}

fn bls12_381_config() -> Rv64WeierstrassConfig {
    let curve = CurveConfig {
        struct_name: "Bls12_381G1Affine".to_string(),
        modulus: BigUint::from_str(
            "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787",
        )
        .unwrap(),
        scalar: BigUint::from_str(
            "52435875175126190479447740508185965837690552500527637822603658699938581184513",
        )
        .unwrap(),
        a: BigUint::from(0u8),
        b: BigUint::from(4u8),
    };
    let mut config = Rv64WeierstrassConfig::new(vec![curve]);
    *config.as_mut() = test_system_config();
    config
}

fn build_bls12_381_ecc_exe(config: &Rv64WeierstrassConfig) -> Result<VmExe<F>> {
    let elf = build_example_program_at_path_with_features(
        get_programs_dir!(),
        "ec_bls12_381",
        ["bls12_381"],
        config,
    )?;
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv64ITranspilerExtension)
            .with_extension(Rv64MTranspilerExtension)
            .with_extension(Rv64IoTranspilerExtension)
            .with_extension(EccTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    )?)
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

fn ecc_opcode(opcode: usize) -> Option<(usize, usize)> {
    let offset = opcode.checked_sub(Rv64WeierstrassOpcode::CLASS_OFFSET)?;
    let local_opcode = offset % WEIERSTRASS_OPCODE_COUNT;
    (local_opcode <= Rv64WeierstrassOpcode::SETUP_EC_DOUBLE as usize)
        .then_some((offset / WEIERSTRASS_OPCODE_COUNT, local_opcode))
}

fn assert_ecc_timestamp_deltas(
    exe: &VmExe<F>,
    config: &Rv64WeierstrassConfig,
    output: &RvrPreflightOutput<F>,
) {
    for (idx, entry) in output.raw_logs.program_log.iter().enumerate() {
        let pc = entry.pc as u32;
        let instruction_idx = ((pc - exe.program.pc_base) / 4) as usize;
        let Some((instruction, _)) = &exe.program.instructions_and_debug_infos[instruction_idx]
        else {
            continue;
        };
        let Some((curve_idx, local_opcode)) = ecc_opcode(instruction.opcode.as_usize()) else {
            continue;
        };
        let coordinate_bytes = config.weierstrass.supported_curves[curve_idx]
            .modulus
            .bits()
            .div_ceil(8) as usize;
        let blocks = if coordinate_bytes <= 32 {
            ECC_BLOCKS_32
        } else {
            ECC_BLOCKS_48
        };
        let num_reads = if matches!(
            local_opcode,
            x if x == Rv64WeierstrassOpcode::EC_ADD_NE as usize
                || x == Rv64WeierstrassOpcode::SETUP_EC_ADD_NE as usize
        ) {
            2
        } else {
            1
        };
        let expected_delta = num_reads + 1 + num_reads * blocks + blocks;
        let next_timestamp = output
            .raw_logs
            .program_log
            .get(idx + 1)
            .map(|next| next.timestamp)
            .unwrap_or(output.system_records.to_state.timestamp);
        assert_eq!(
            next_timestamp - entry.timestamp,
            expected_delta as u32,
            "ECC opcode {:#x} at pc {pc:#x} timestamp delta",
            instruction.opcode.as_usize()
        );
    }
}

fn ecc_air_ids(exe: &VmExe<F>, pc_to_air_idx: &[Option<usize>]) -> BTreeSet<usize> {
    let ids = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .zip(pc_to_air_idx)
        .filter_map(|(slot, &air_idx)| {
            let (instruction, _) = slot.as_ref()?;
            ecc_opcode(instruction.opcode.as_usize())?;
            air_idx
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), 2, "EcAddNe and EcDouble AIRs must be mapped");
    ids
}

fn assert_rvr_differential(
    label: &str,
    exe: &VmExe<F>,
    config: &Rv64WeierstrassConfig,
    segments: Vec<(Option<u64>, Vec<u32>)>,
) {
    let (mut interp_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64WeierstrassCpuBuilder,
        config.clone(),
    )
    .expect("interpreter vm init");
    let (mut rvr_vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64WeierstrassCpuBuilder,
        config.clone(),
    )
    .expect("rvr vm init");
    let air_names = rvr_vm.air_names().map(str::to_owned).collect::<Vec<_>>();
    assert_eq!(
        interp_vm.air_names().collect::<Vec<_>>(),
        rvr_vm.air_names().collect::<Vec<_>>(),
        "{label}: AIR order"
    );
    let pc_to_air_idx = rvr_vm.pc_to_air_idx(exe).expect("pc to air mapping");
    let ecc_air_ids = ecc_air_ids(exe, &pc_to_air_idx);
    let widths = rvr_vm
        .pk()
        .per_air
        .iter()
        .map(|pk| pk.vk.params.width.main_width())
        .collect::<Vec<_>>();
    let mut registry = LogNativeAssemblerRegistry::new();
    config.extend_rvr_log_native(&mut registry);
    let mut interpreter = interp_vm
        .preflight_interpreter(exe)
        .expect("interpreter preflight");
    let mut state = interp_vm.create_initial_state(exe, Streams::default());
    let segments = if segments.is_empty() {
        vec![(None, vec![32768; rvr_vm.num_airs()])]
    } else {
        segments
    };

    let segment_outputs = {
        let route = rvr_vm
            .preflight_routed_instance(exe)
            .expect("routed preflight instance");
        let RvrPreflightRoute::Rvr(instance) = route else {
            panic!("{label}: ECC program must route to RVR preflight");
        };
        let mut outputs = Vec::with_capacity(segments.len());
        for (segment_idx, (num_insns, trace_heights)) in segments.into_iter().enumerate() {
            let segment_label = format!("{label}_segment_{segment_idx}");
            let from_state = state.clone();
            let interp_output = interp_vm
                .execute_preflight(
                    &mut interpreter,
                    from_state.clone(),
                    num_insns,
                    &trace_heights,
                )
                .expect("interpreter execution");
            let rvr_output = instance
                .execute_preflight_from_state(from_state.clone(), num_insns)
                .expect("rvr preflight execution");
            assert_system_records_eq(
                &segment_label,
                &interp_output.system_records,
                &rvr_output.system_records,
            );
            assert_ecc_timestamp_deltas(exe, config, &rvr_output);
            let capacities = trace_heights
                .iter()
                .zip(&widths)
                .map(|(&height, &width)| (height as usize, width))
                .collect::<Vec<_>>();
            let rvr_arenas = generate_record_arenas_from_logs::<F, MatrixRecordArena<F>>(
                &registry,
                exe,
                &rvr_output,
                &capacities,
                &pc_to_air_idx,
            )
            .expect("rvr log-native record assembly");
            state = rvr_output.to_state.clone();
            outputs.push((
                from_state,
                interp_output,
                rvr_output.system_records,
                rvr_arenas,
            ));
        }
        outputs
    };

    let interp_program = interp_vm.commit_program_on_device(&exe.program);
    interp_vm.load_program(interp_program);
    let rvr_program = rvr_vm.commit_program_on_device(&exe.program);
    rvr_vm.load_program(rvr_program);
    let mut active_ecc_airs = BTreeSet::new();

    for (segment_idx, (from_state, interp_output, rvr_system_records, rvr_arenas)) in
        segment_outputs.into_iter().enumerate()
    {
        let segment_label = format!("{label}_segment_{segment_idx}");
        interp_vm.transport_init_memory_to_device(&from_state.memory);
        let interp_ctx = interp_vm
            .generate_proving_ctx(interp_output.system_records, interp_output.record_arenas)
            .expect("interpreter trace generation");
        rvr_vm.transport_init_memory_to_device(&from_state.memory);
        let rvr_ctx = rvr_vm
            .generate_proving_ctx(rvr_system_records, rvr_arenas)
            .expect("rvr trace generation");

        // HARD-5: compare only the two deterministic ECC AIRs selected from
        // ECC program counters. Shared lookup/periphery AIR row order is not
        // stable under parallel trace generation and is deliberately excluded.
        let mut interp_traces = interp_ctx
            .per_trace
            .into_iter()
            .filter(|(air_idx, _)| ecc_air_ids.contains(air_idx))
            .collect::<BTreeMap<_, _>>();
        let mut rvr_traces = rvr_ctx
            .per_trace
            .into_iter()
            .filter(|(air_idx, _)| ecc_air_ids.contains(air_idx))
            .collect::<BTreeMap<_, _>>();
        let interp_air_ids = interp_traces.keys().copied().collect::<Vec<_>>();
        assert_eq!(
            interp_air_ids,
            rvr_traces.keys().copied().collect::<Vec<_>>(),
            "{segment_label}: active ECC AIR set"
        );
        for air_idx in interp_air_ids {
            let interp_trace = interp_traces.remove(&air_idx).unwrap();
            let rvr_trace = rvr_traces.remove(&air_idx).unwrap();
            let air_name = &air_names[air_idx];
            assert_eq!(
                interp_trace.common_main.width, rvr_trace.common_main.width,
                "{segment_label}: {air_name} width"
            );
            assert_eq!(
                interp_trace.common_main.values, rvr_trace.common_main.values,
                "{segment_label}: {air_name} values"
            );
            assert_eq!(
                interp_trace.public_values, rvr_trace.public_values,
                "{segment_label}: {air_name} public values"
            );
            active_ecc_airs.insert(air_idx);
        }
    }

    assert_eq!(
        active_ecc_airs, ecc_air_ids,
        "{label}: EcAddNe and EcDouble traces must both be active"
    );
}

#[test]
fn test_weierstrass_rvr_preflight_differential() -> Result<()> {
    let config = secp256k1_config();
    let exe = build_ecc_exe(&config)?;
    assert_rvr_differential("weierstrass_single", &exe, &config, Vec::new());
    Ok(())
}

#[test]
fn test_weierstrass_rvr_preflight_bls12_381_differential() -> Result<()> {
    let config = bls12_381_config();
    let exe = build_bls12_381_ecc_exe(&config)?;
    assert_rvr_differential("weierstrass_bls12_381", &exe, &config, Vec::new());
    Ok(())
}

#[test]
fn test_weierstrass_rvr_preflight_multi_segment_differential() -> Result<()> {
    let mut config = secp256k1_config();
    config.as_mut().segmentation_max_memory = 1;
    let exe = build_ecc_exe(&config)?;
    let (vm, _) = VirtualMachine::new_with_keygen(
        test_cpu_engine(),
        Rv64WeierstrassCpuBuilder,
        config.clone(),
    )
    .expect("metered vm init");
    let metered_ctx = vm.build_metered_ctx(&exe);
    let metered = vm.metered_interpreter(&exe).expect("metered interpreter");
    let (segments, _) = metered
        .execute_metered(Streams::default(), metered_ctx)
        .expect("metered execution");
    assert!(
        segments.len() > 1,
        "tight memory limit must force multiple ECC segments"
    );
    let segments = segments
        .into_iter()
        .map(|segment| (Some(segment.num_insns), segment.trace_heights))
        .collect();
    assert_rvr_differential("weierstrass_multi", &exe, &config, segments);
    Ok(())
}
