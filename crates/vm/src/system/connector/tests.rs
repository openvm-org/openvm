use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_instructions::{
    instruction::Instruction, program::Program, LocalOpcode, SystemOpcode::TERMINATE,
};
use openvm_stark_backend::{
    config::StarkGenericConfig,
    engine::StarkEngine,
    p3_field::FieldAlgebra,
    prover::{cpu::CpuBackend, types::AirProvingContext},
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};

use super::VmConnectorPvs;
use crate::{
    arch::{
        PreflightExecutionOutput, Streams, SystemConfig, VirtualMachine, VmState, CONNECTOR_AIR_ID,
    },
    system::{
        memory::{online::GuestMemory, AddressMap},
        program::trace::VmCommittedExe,
        SystemCpuBuilder,
    },
};

type F = BabyBear;
type SC = BabyBearPoseidon2Config;
type PB = CpuBackend<SC>;

#[test]
fn test_vm_connector_happy_path() {
    let exit_code = 1789;
    test_impl(true, exit_code, |air_ctx| {
        let pvs: &VmConnectorPvs<F> = air_ctx.public_values.as_slice().borrow();
        assert_eq!(pvs.is_terminate, F::ONE);
        assert_eq!(pvs.exit_code, F::from_canonical_u32(exit_code));
    });
}

#[test]
fn test_vm_connector_wrong_exit_code() {
    let exit_code = 1789;
    test_impl(false, exit_code, |air_ctx| {
        let pvs: &mut VmConnectorPvs<F> = air_ctx.public_values.as_mut_slice().borrow_mut();
        pvs.exit_code = F::from_canonical_u32(exit_code + 1);
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
    let vm_config = SystemConfig::default().without_continuations();
    let engine = BabyBearPoseidon2Engine::new(FriParameters::new_for_testing(1));
    let (mut vm, pk) =
        VirtualMachine::new_with_keygen(engine, SystemCpuBuilder, vm_config.clone()).unwrap();
    let vk = pk.get_vk();

    let instructions = vec![Instruction::<F>::from_isize(
        TERMINATE.global_opcode(),
        0,
        0,
        exit_code as isize,
        0,
        0,
    )];

    let program = Program::from_instructions(&instructions);
    let committed_exe = Arc::new(VmCommittedExe::<SC>::commit(
        program.into(),
        vm.engine.config().pcs(),
    ));
    let max_trace_heights = vec![0; vk.total_widths().len()];
    let memory = GuestMemory::new(AddressMap::from_mem_config(&vm_config.memory_config));
    vm.transport_init_memory_to_device(&memory);
    vm.load_program(committed_exe.get_committed_trace());
    let from_state = VmState::new(
        0,
        0,
        memory,
        Streams::default(),
        0,
        vm_config.num_public_values,
    );
    let mut interpreter = vm.preflight_interpreter(&committed_exe.exe).unwrap();
    let PreflightExecutionOutput {
        system_records,
        record_arenas,
        ..
    } = vm
        .execute_preflight(&mut interpreter, from_state, None, &max_trace_heights)
        .unwrap();
    let mut ctx = vm
        .generate_proving_ctx(system_records, record_arenas)
        .unwrap();
    let connector_air_ctx = &mut ctx
        .per_air
        .iter_mut()
        .find(|(air_id, _)| *air_id == CONNECTOR_AIR_ID)
        .unwrap()
        .1;
    f(connector_air_ctx);
    let proof = vm.engine.prove(vm.pk(), ctx);
    if should_pass {
        vm.engine.verify(&vk, &proof).expect("Verification failed");
    } else {
        assert!(vm.engine.verify(&vk, &proof).is_err());
    }
}
