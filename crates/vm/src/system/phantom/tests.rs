use std::sync::{Arc, Mutex};

use openvm_instructions::{instruction::Instruction, SystemOpcode, VmOpcode};
use openvm_stark_backend::p3_field::{AbstractField, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::PhantomChip;
use crate::arch::{instructions::UsizeOpcode, testing::VmChipTestBuilder, ExecutionState};
type F = BabyBear;

#[test]
fn test_nops_and_terminate() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = PhantomChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        SystemOpcode::default_offset(),
    );
    chip.set_streams(Arc::new(Mutex::new(Default::default())));

    let nop = Instruction::from_isize(
        VmOpcode::with_default_offset(SystemOpcode::PHANTOM),
        0,
        0,
        0,
        0,
        0,
    );
    let mut state: ExecutionState<F> = ExecutionState::new(F::ZERO, F::ONE);
    let num_nops = 5;
    for _ in 0..num_nops {
        tester.execute_with_pc(&mut chip, nop.clone(), state.pc.as_canonical_u32());
        let new_state = tester.execution.records.last().unwrap().final_state;
        assert_eq!(state.pc + F::from_canonical_usize(4), new_state.pc);
        assert_eq!(state.timestamp + F::ONE, new_state.timestamp);
        state = new_state;
    }

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
