use axvm_instructions::{instruction::Instruction, NopOpcode};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};

use super::NopChip;
use crate::arch::{instructions::UsizeOpcode, testing::VmChipTestBuilder, ExecutionState};
type F = BabyBear;

#[test]
fn test_nops_and_terminate() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = NopChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        NopOpcode::default_offset(),
    );

    let nop = Instruction::from_isize(NopOpcode::NOP.with_default_offset(), 0, 0, 0, 0, 0);
    let mut state: ExecutionState<F> = ExecutionState::new(F::zero(), F::one());
    let num_nops = 5;
    for _ in 0..num_nops {
        tester.execute_with_pc(&mut chip, nop.clone(), state.pc.as_canonical_u32());
        let new_state = tester.execution.records.last().unwrap().final_state;
        assert_eq!(state.pc + F::from_canonical_usize(4), new_state.pc);
        assert_eq!(state.timestamp + F::one(), new_state.timestamp);
        state = new_state;
    }

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
