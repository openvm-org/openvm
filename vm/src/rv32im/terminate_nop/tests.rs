use axvm_instructions::{Rv32NopOpcode, TerminateOpcode};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};

use super::Rv32TerminateNopChip;
use crate::{
    arch::{instructions::UsizeOpcode, testing::VmChipTestBuilder, ExecutionState},
    system::program::Instruction,
};

type F = BabyBear;

#[test]
fn test_nops_and_terminate() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32TerminateNopChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        Rv32NopOpcode::default_offset(),
    );

    let nop = Instruction::from_isize(Rv32NopOpcode::NOP.with_default_offset(), 0, 0, 0, 0, 0);
    // let terminate = Instruction::from_isize(
    //     TerminateOpcode::TERMINATE.with_default_offset(),
    //     0,
    //     0,
    //     0,
    //     0,
    //     0,
    // );

    let mut state: ExecutionState<F> = ExecutionState::new(F::zero(), F::one());
    let num_nops = 5;
    for _ in 0..num_nops {
        tester.execute_with_pc(&mut chip, nop.clone(), state.pc.as_canonical_u32());
        let new_state = tester.execution.records.last().unwrap().final_state;
        assert_eq!(state.pc + F::from_canonical_usize(4), new_state.pc);
        assert_eq!(state.timestamp, new_state.timestamp);
        state = new_state;
    }
    // tester.execute_with_pc(&mut chip, terminate, state.pc.as_canonical_u32());
    // let new_state = tester.execution.records.last().unwrap().final_state;
    // assert_eq!(state.pc, new_state.pc);
    // assert_eq!(state.timestamp, new_state.timestamp);

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
