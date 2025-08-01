use openvm_instructions::{instruction::Instruction, SystemOpcode};
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::PhantomExecutor;
use crate::{
    arch::{
        instructions::LocalOpcode,
        testing::{TestChipHarness, VmChipTestBuilder},
        ExecutionState, VmChipWrapper,
    },
    system::phantom::{PhantomAir, PhantomFiller},
};
type F = BabyBear;

#[test]
fn test_nops_and_terminate() {
    let mut tester = VmChipTestBuilder::default();
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
    let executor = PhantomExecutor::<F>::new(Default::default(), phantom_opcode);
    let air = PhantomAir {
        execution_bridge: tester.execution_bridge(),
        phantom_opcode,
    };
    let chip = VmChipWrapper::new(PhantomFiller, tester.memory_helper());
    let num_nops = 5;
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, num_nops);

    let nop = Instruction::from_isize(phantom_opcode, 0, 0, 0, 0, 0);
    let mut state: ExecutionState<F> = ExecutionState::new(F::ZERO, F::ONE);
    for _ in 0..num_nops {
        tester.execute_with_pc(&mut harness, &nop, state.pc.as_canonical_u32());
        let new_state = tester.execution.records.last().unwrap().final_state;
        assert_eq!(state.pc + F::from_canonical_usize(4), new_state.pc);
        assert_eq!(state.timestamp + F::ONE, new_state.timestamp);
        state = new_state;
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}
