#![cfg(all(feature = "cuda", feature = "rvr"))]

use eyre::{ensure, Result};
use halo2curves_axiom::{
    bls12_381::{Fq as BlsFq, Fq2 as BlsFq2, Fr as BlsFr, G1Affine as BlsG1, G2Affine as BlsG2},
    bn256::{Fq as BnFq, Fq2 as BnFq2, Fr as BnFr, G1Affine as BnG1, G2Affine as BnG2},
};
use openvm_algebra_transpiler::{
    Fp2Opcode, Fp2TranspilerExtension, ModularTranspilerExtension, Rv64ModularArithmeticOpcode,
};
use openvm_circuit::{
    arch::{
        rvr::{PreflightEndpoint, PreflightExecution, PreflightLimits},
        verify_segments, VirtualMachine, VmExecutor,
    },
    utils::{test_gpu_engine, test_system_config},
};
use openvm_ecc_circuit::WeierstrassPreflightGpuTracegen;
use openvm_ecc_guest::{algebra::field::FieldExtension, AffinePoint};
use openvm_ecc_transpiler::{EccTranspilerExtension, Rv64WeierstrassOpcode};
use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode, SystemOpcode,
};
use openvm_pairing_circuit::{PairingCurve, Rv64PairingConfig, Rv64PairingGpuBuilder};
use openvm_pairing_guest::{
    bls12_381::BLS12_381_COMPLEX_STRUCT_NAME, bn254::BN254_COMPLEX_STRUCT_NAME,
};
use openvm_pairing_transpiler::{PairingPhantom, PairingTranspilerExtension};
use openvm_riscv_transpiler::{
    Rv64HintStoreOpcode, Rv64ITranspilerExtension, Rv64IoTranspilerExtension,
    Rv64MTranspilerExtension,
};
use openvm_stark_sdk::{
    openvm_stark_backend::{p3_field::PrimeField32, StarkEngine},
    p3_baby_bear::BabyBear,
};
use openvm_toolchain_tests::{build_example_program_at_path_with_features, get_programs_dir};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

const DISCOVERY_INSTRUCTIONS: usize = 512;
const DISCOVERY_REPLAY_VALUES: usize = 4_096;
const PROOF_CHECKPOINT_INTERVAL: usize = 512;

#[derive(Default)]
struct Discovery {
    split: Option<u32>,
    retired: u32,
    replay_values: usize,
}

fn instruction_at(exe: &VmExe<BabyBear>, pc: u32) -> &Instruction<BabyBear> {
    let slot = pc
        .checked_sub(exe.program.pc_base)
        .expect("executed PC precedes the program base")
        / DEFAULT_PC_STEP;
    &exe.program
        .get_instruction_and_debug_info(slot as usize)
        .unwrap_or_else(|| panic!("executed PC {pc:#x} is absent from the program"))
        .0
}

fn block_contains_pc(block_pc: u32, block_len: u32, pc: u32) -> bool {
    pc >= block_pc
        && (pc - block_pc).is_multiple_of(DEFAULT_PC_STEP)
        && (pc - block_pc) / DEFAULT_PC_STEP < block_len
}

fn find_split_after_phantom(execution: &PreflightExecution, pairing_pc: u32) -> Option<u32> {
    let mut blocks = Vec::with_capacity(execution.transcript.checkpoints.len() + 2);
    blocks.push((0, execution.from_state.pc));
    blocks.extend(
        execution
            .transcript
            .checkpoints
            .iter()
            .map(|checkpoint| (checkpoint.retired, checkpoint.pc)),
    );
    blocks.push((execution.retired, execution.to_state.pc));

    blocks.windows(2).find_map(|window| {
        let (block_retired, block_pc) = window[0];
        let next_retired = window[1].0;
        block_contains_pc(block_pc, next_retired - block_retired, pairing_pc)
            .then_some(next_retired)
    })
}

fn discover_pairing_split(
    checkpoint: &openvm_circuit::arch::rvr::PreflightInstance<'_>,
    pairing_pc: u32,
    input: Vec<Vec<u8>>,
) -> Result<Discovery> {
    let mut state = checkpoint.create_initial_vm_state(input);
    let mut discovery = Discovery::default();
    loop {
        let execution = checkpoint.execute_from_state_for(
            state,
            PreflightLimits::new(DISCOVERY_INSTRUCTIONS, DISCOVERY_REPLAY_VALUES, 1),
        )?;
        if discovery.split.is_none() {
            discovery.split = find_split_after_phantom(&execution, pairing_pc)
                .map(|local| discovery.retired + local);
        }
        discovery.retired = discovery
            .retired
            .checked_add(execution.retired)
            .expect("pairing fixture instruction count exceeds u32");
        let chunk_replay_values = execution.transcript.replay_values.len();
        discovery.replay_values += chunk_replay_values;
        state = execution.state;
        if matches!(execution.endpoint, PreflightEndpoint::Terminated) {
            break;
        }
    }
    Ok(discovery)
}

fn prove_pairing_checkpoint(
    mut config: Rv64PairingConfig,
    exe: VmExe<BabyBear>,
    input: Vec<Vec<u8>>,
) -> Result<()> {
    *config.as_mut() = test_system_config();
    let executor = VmExecutor::new(config.clone())?;
    let checkpoint = executor.preflight_instance(&exe)?;
    let pairing_pcs = exe
        .program
        .enumerate_by_pc()
        .iter()
        .filter_map(|(pc, instruction, _)| {
            (instruction.opcode.as_usize() == SystemOpcode::PHANTOM.global_opcode_usize()
                && instruction.c.as_canonical_u32() as u16 == PairingPhantom::HintFinalExp as u16)
                .then_some(*pc)
        })
        .collect::<Vec<_>>();
    ensure!(
        pairing_pcs.len() == 1,
        "expected one static pairing phantom, found {}",
        pairing_pcs.len()
    );
    let pairing_pc = pairing_pcs[0];
    let discovery = discover_pairing_split(&checkpoint, pairing_pc, input.clone())?;
    let split = discovery
        .split
        .ok_or_else(|| eyre::eyre!("pairing phantom was not executed"))?;
    ensure!(
        split < discovery.retired,
        "pairing split must precede termination"
    );

    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(test_gpu_engine(), Rv64PairingGpuBuilder, config.clone())?;
    let cached_program = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program);
    let replay_program = WeierstrassPreflightGpuTracegen::upload_postflight_program(
        &exe.program,
        &config.modular.system.memory_config,
        &config.modular.modular,
        Some(&config.fp2),
        &config.weierstrass,
        &vm.engine.device().device_ctx,
    )?;

    let segment_lengths = [split, discovery.retired - split];
    let mut state = checkpoint.create_initial_vm_state(input);
    let mut expected_pc = state.pc();
    let mut saw_pairing_phantom = false;
    let mut saw_pairing_hint_store = false;
    let mut saw_modular = false;
    let mut saw_fp2 = false;
    let mut saw_ecc = false;
    let mut proofs = Vec::with_capacity(segment_lengths.len());
    for (segment_index, retired) in segment_lengths.into_iter().enumerate() {
        ensure!(
            state.pc() == expected_pc,
            "checkpoint resume PC is discontinuous"
        );
        vm.transport_init_memory_to_device(&state.memory);
        let limits = PreflightLimits::new(
            retired as usize,
            discovery.replay_values.max(1),
            PROOF_CHECKPOINT_INTERVAL,
        );
        let execution = checkpoint.execute_from_state_for(state, limits)?;
        ensure!(
            execution.from_state.pc == expected_pc,
            "execution-bus start does not match the prior endpoint"
        );
        match (segment_index, execution.endpoint) {
            (0, PreflightEndpoint::Suspended) => {}
            (1, PreflightEndpoint::Terminated) => {}
            (0, PreflightEndpoint::Terminated) => {
                eyre::bail!("pairing prefix terminated before HintStore")
            }
            (1, PreflightEndpoint::Suspended) => {
                eyre::bail!("final pairing segment did not terminate")
            }
            _ => unreachable!(),
        }

        let (gpu_transcript, replay_plan) =
            WeierstrassPreflightGpuTracegen::postflight(&vm, &replay_program, &execution, retired)?;
        let program_log = gpu_transcript.program_log_host()?;
        let memory_log = gpu_transcript.memory_log_host()?;
        for event in program_log.iter().take(program_log.len().saturating_sub(1)) {
            let instruction = instruction_at(&exe, event.pc);
            let opcode = instruction.opcode.as_usize();
            let is_pairing_phantom = opcode == SystemOpcode::PHANTOM.global_opcode_usize()
                && instruction.c.as_canonical_u32() as u16 == PairingPhantom::HintFinalExp as u16;
            if is_pairing_phantom {
                saw_pairing_phantom = true;
            } else if saw_pairing_phantom
                && (opcode == Rv64HintStoreOpcode::HINT_STORED.global_opcode_usize()
                    || opcode == Rv64HintStoreOpcode::HINT_BUFFER.global_opcode_usize())
            {
                saw_pairing_hint_store = true;
            }
            saw_modular |= (Rv64ModularArithmeticOpcode::CLASS_OFFSET
                ..Rv64WeierstrassOpcode::CLASS_OFFSET)
                .contains(&opcode);
            saw_ecc |=
                (Rv64WeierstrassOpcode::CLASS_OFFSET..Fp2Opcode::CLASS_OFFSET).contains(&opcode);
            saw_fp2 |= (Fp2Opcode::CLASS_OFFSET..Fp2Opcode::CLASS_OFFSET + 0x100).contains(&opcode);
        }
        if segment_index == 0 {
            ensure!(
                !saw_pairing_hint_store,
                "the pairing advice and HintStore must be in different segments"
            );
            let phantom_step = program_log
                .iter()
                .position(|event| event.pc == pairing_pc)
                .expect("pairing phantom is absent from the expanded program log");
            let phantom = program_log[phantom_step];
            let after_phantom = program_log[phantom_step + 1];
            ensure!(
                after_phantom.timestamp == phantom.timestamp + 1,
                "pairing phantom must advance exactly one timestamp"
            );
            ensure!(
                memory_log
                    .iter()
                    .all(|event| event.timestamp != phantom.timestamp),
                "pairing P/Q peeks must not become timed memory events"
            );
        }

        let proving_ctx = Rv64PairingGpuBuilder::generate_proving_ctx_from_postflight(
            &mut vm,
            &config,
            replay_program.program(),
            &gpu_transcript,
            &replay_plan,
        )?;
        let PreflightExecution {
            state: next_state,
            to_state,
            ..
        } = execution;
        expected_pc = to_state.pc;
        drop(replay_plan);
        drop(gpu_transcript);

        let proof = vm.engine.prove(vm.pk(), proving_ctx)?;
        proofs.push(proof);
        state = next_state;
    }
    verify_segments(&vm.engine, &pk.get_vk(), &proofs)?;
    ensure!(saw_pairing_phantom, "pairing phantom was not replayed");
    ensure!(
        saw_pairing_hint_store,
        "pairing hint was not materialized after its phantom"
    );
    ensure!(
        saw_modular && saw_fp2,
        "pairing fixture did not execute modular and Fp2 AIRs"
    );
    ensure!(saw_ecc, "pairing fixture did not execute an ECC AIR");
    ensure!(state.pc() == expected_pc, "final state PC is inconsistent");
    Ok(())
}

fn transpile_pairing_fixture(
    curve_feature: &str,
    config: &Rv64PairingConfig,
) -> Result<VmExe<BabyBear>> {
    let elf = build_example_program_at_path_with_features(
        get_programs_dir!("tests/programs"),
        "pairing_check",
        [curve_feature, "rvr_checkpoint"],
        config,
    )?;
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv64ITranspilerExtension)
            .with_extension(Rv64MTranspilerExtension)
            .with_extension(Rv64IoTranspilerExtension)
            .with_extension(PairingTranspilerExtension)
            .with_extension(EccTranspilerExtension)
            .with_extension(ModularTranspilerExtension)
            .with_extension(Fp2TranspilerExtension),
    )?)
}

#[cfg(feature = "bn254")]
fn prove_bn254_pairing_checkpoint() -> Result<()> {
    let config = Rv64PairingConfig::new(
        vec![PairingCurve::Bn254],
        vec![BN254_COMPLEX_STRUCT_NAME.to_string()],
    );
    let exe = transpile_pairing_fixture("bn254", &config)?;

    let generator = BnG1::generator();
    let twist_generator = BnG2::generator();
    let mut p = [
        BnG1::from(generator * BnFr::from(1)),
        BnG1::from(generator * BnFr::from(2)),
    ];
    p[1].y = -p[1].y;
    let q = [
        BnG2::from(twist_generator * BnFr::from(2)),
        BnG2::from(twist_generator * BnFr::from(1)),
    ];
    let input = p
        .map(|point| AffinePoint::new(point.x, point.y))
        .into_iter()
        .flat_map(|point| {
            [point.x, point.y]
                .into_iter()
                .flat_map(|value: BnFq| value.to_bytes())
        })
        .chain(
            q.map(|point| AffinePoint::new(point.x, point.y))
                .into_iter()
                .flat_map(|point| [point.x, point.y])
                .flat_map(|value: BnFq2| value.to_coeffs())
                .flat_map(|value| value.to_bytes()),
        )
        .collect();
    prove_pairing_checkpoint(config, exe, vec![input])
}

#[cfg(feature = "bls12_381")]
fn prove_bls12_381_pairing_checkpoint() -> Result<()> {
    let config = Rv64PairingConfig::new(
        vec![PairingCurve::Bls12_381],
        vec![BLS12_381_COMPLEX_STRUCT_NAME.to_string()],
    );
    let exe = transpile_pairing_fixture("bls12_381", &config)?;

    let generator = BlsG1::generator();
    let twist_generator = BlsG2::generator();
    let mut p = [
        BlsG1::from(generator * BlsFr::from(1)),
        BlsG1::from(generator * BlsFr::from(2)),
    ];
    p[1].y = -p[1].y;
    let q = [
        BlsG2::from(twist_generator * BlsFr::from(2)),
        BlsG2::from(twist_generator * BlsFr::from(1)),
    ];
    let input = p
        .map(|point| AffinePoint::new(point.x, point.y))
        .into_iter()
        .flat_map(|point| {
            [point.x, point.y]
                .into_iter()
                .flat_map(|value: BlsFq| value.to_bytes())
        })
        .chain(
            q.map(|point| AffinePoint::new(point.x, point.y))
                .into_iter()
                .flat_map(|point| [point.x, point.y])
                .flat_map(|value: BlsFq2| value.to_coeffs())
                .flat_map(|value| value.to_bytes()),
        )
        .collect();
    prove_pairing_checkpoint(config, exe, vec![input])
}

#[test]
#[cfg(any(feature = "bn254", feature = "bls12_381"))]
fn pairing_checkpoint_replay_proves_across_hint_boundary() -> Result<()> {
    #[cfg(feature = "bn254")]
    prove_bn254_pairing_checkpoint()?;
    #[cfg(feature = "bls12_381")]
    prove_bls12_381_pairing_checkpoint()?;
    Ok(())
}
