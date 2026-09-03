use std::borrow::{Borrow, BorrowMut};

use openvm_cpu_backend::CpuBackend;
use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::Program, LocalOpcode, SystemOpcode::TERMINATE,
};
use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing, prover::AirProvingContext, verifier::VerifierError,
    StarkEngine,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
};

use super::VmConnectorPvs;
use crate::{
    arch::{
        execution_mode::Segment, ExecutionError, PostflightTracegen, Streams, SystemConfig,
        VirtualMachine, VmState, CONNECTOR_AIR_ID,
    },
    system::{
        memory::{online::GuestMemory, AddressMap},
        SystemCpuBuilder,
    },
    utils::test_cpu_engine,
};

type F = BabyBear;
type SC = BabyBearPoseidon2Config;
type PB = CpuBackend<SC>;

#[test]
fn preflight_enforces_exact_terminal_instruction_count() {
    let vm_config = SystemConfig::default();
    let engine = test_cpu_engine();
    let (vm, _) =
        VirtualMachine::new_with_keygen(engine, SystemCpuBuilder, vm_config.clone()).unwrap();
    let instruction = Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 0, 0, 0);
    let vm_exe: VmExe = Program::from_instructions(&[instruction]).into();
    let interpreter = vm.preflight_interpreter(&vm_exe).unwrap();
    let initial_state = || {
        let memory = GuestMemory::new(AddressMap::from_mem_config(&vm_config.memory_config));
        VmState::new_with_defaults(0, memory, Streams::default(), 0)
    };

    let output = interpreter
        .execute_segment(initial_state(), &Segment::new(0, 1, 0, vec![]))
        .unwrap();
    assert_eq!(output.exit_code, Some(0));

    let error = match interpreter.execute_segment(initial_state(), &Segment::new(0, 2, 0, vec![])) {
        Ok(_) => panic!("early termination must not satisfy a longer segment"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        ExecutionError::RetiredInstructionCountMismatch {
            expected: 2,
            actual: 1
        }
    ));

    let instruction = Instruction::from_isize(TERMINATE.global_opcode(), 0, 0, 1, 0, 0);
    let vm_exe: VmExe = Program::from_instructions(&[instruction]).into();
    let interpreter = vm.preflight_interpreter(&vm_exe).unwrap();
    let error = match interpreter.execute_segment(initial_state(), &Segment::new(0, 1, 0, vec![])) {
        Ok(_) => panic!("failed guest termination must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, ExecutionError::FailedWithExitCode(1)));
}

#[test]
fn test_vm_connector_happy_path() {
    let exit_code = 1789;
    test_impl(true, exit_code, |air_ctx| {
        let pvs: &VmConnectorPvs<F> = air_ctx.public_values.as_slice().borrow();
        assert_eq!(pvs.is_terminate, F::ONE);
        assert_eq!(pvs.exit_code, F::from_u32(exit_code));
    });
}

#[test]
fn test_vm_connector_wrong_exit_code() {
    let exit_code = 1789;
    test_impl(false, exit_code, |air_ctx| {
        let pvs: &mut VmConnectorPvs<F> = air_ctx.public_values.as_mut_slice().borrow_mut();
        pvs.exit_code = F::from_u32(exit_code + 1);
    });
}

#[test]
fn test_vm_connector_wrong_is_terminate() {
    let exit_code = 1789;
    test_impl(false, exit_code, |air_ctx| {
        let pvs: &mut VmConnectorPvs<F> = air_ctx.public_values.as_mut_slice().borrow_mut();
        pvs.is_terminate = F::ZERO;
    });
}

fn test_impl(should_pass: bool, exit_code: u32, f: impl FnOnce(&mut AirProvingContext<PB>)) {
    let vm_config = SystemConfig::default();
    let engine = test_cpu_engine();
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(engine, SystemCpuBuilder, vm_config.clone()).unwrap();
    let vk = pk.get_vk();

    let instructions = vec![Instruction::from_isize(
        TERMINATE.global_opcode(),
        0,
        0,
        exit_code as isize,
        0,
        0,
    )];

    let program = Program::from_instructions(&instructions);
    let vm_exe: VmExe = program.into();
    let memory = GuestMemory::new(AddressMap::from_mem_config(&vm_config.memory_config));
    vm.transport_init_memory_to_device(&memory);
    vm.load_program(vm.commit_program_on_device(&vm_exe.program));
    let from_state = VmState::new_with_defaults(0, memory, Streams::default(), 0);
    let interpreter = vm.preflight_interpreter(&vm_exe).unwrap();
    let output = interpreter
        .execute_preflight_from_state(from_state, None)
        .unwrap();
    assert_eq!(output.history.program.len(), 2);
    assert_eq!(output.history.program[0], output.history.program[1]);
    assert_eq!(output.history.program[0].pc, 0);
    assert_eq!(output.history.program[0].timestamp, 1);
    assert!(output.history.memory.accesses.is_empty());
    assert!(output.history.memory.initial_writes.is_empty());
    let prepared = SystemCpuBuilder::prepare_postflight(&vm, &vm_exe.program).unwrap();
    let mut ctx = vm
        .generate_proving_ctx(&vm_exe.program, &prepared, &output)
        .unwrap();
    let connector_air_ctx = &mut ctx
        .per_trace
        .iter_mut()
        .find(|(air_id, _)| *air_id == CONNECTOR_AIR_ID)
        .unwrap()
        .1;
    f(connector_air_ctx);
    let proof = vm.engine.prove(vm.pk(), ctx).unwrap();
    if should_pass {
        vm.engine.verify(&vk, &proof).expect("Verification failed");
    } else {
        let result = vm.engine.verify(&vk, &proof);
        assert!(matches!(
            result,
            Err(VerifierError::BatchConstraintError(_))
        ));
    }
}
