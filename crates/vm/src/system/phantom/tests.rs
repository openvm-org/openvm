use openvm_instructions::{instruction::Instruction, SystemOpcode, VmOpcode};
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::PhantomExecutor;
use crate::{
    arch::{
        instructions::LocalOpcode,
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionState, PreflightExecutor,
    },
    system::phantom::{PhantomAir, PhantomChip, PhantomFiller},
};

type F = BabyBear;

fn run_phantom_test<E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    phantom_opcode: VmOpcode,
    num_nops: usize,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
    let nop = Instruction::from_isize(phantom_opcode, 0, 0, 0, 0, 0);
    let mut state: ExecutionState<F> = ExecutionState::new(F::ZERO, F::ONE);

    for _ in 0..num_nops {
        tester.execute_with_pc(executor, arena, &nop, state.pc.as_canonical_u32());
        let new_state = tester.execution_final_state();
        assert_eq!(state.pc + F::from_canonical_usize(4), new_state.pc);
        assert_eq!(state.timestamp + F::ONE, new_state.timestamp);
        state = new_state;
    }
}

#[test]
fn test_nops_and_terminate() {
    const NUM_NOPS: usize = 100;
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();

    let mut tester = VmChipTestBuilder::default();
    let executor = PhantomExecutor::<F>::new(Default::default(), phantom_opcode);
    let chip = PhantomChip::new(PhantomFiller, tester.memory_helper());
    let air = PhantomAir {
        execution_bridge: tester.execution_bridge(),
        phantom_opcode,
    };
    let mut harness = TestChipHarness::with_capacity(executor, air, chip, NUM_NOPS);

    run_phantom_test(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        phantom_opcode,
        NUM_NOPS,
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_phantom_tracegen() {
    use crate::{
        arch::{
            testing::{GpuChipTestBuilder, GpuTestChipHarness},
            EmptyMultiRowLayout,
        },
        system::{cuda::phantom::PhantomChipGPU, phantom::PhantomRecord},
    };

    const NUM_NOPS: usize = 100;
    let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
    let mut tester = GpuChipTestBuilder::default();

    let executor = PhantomExecutor::<F>::new(Default::default(), phantom_opcode);
    let air = PhantomAir {
        execution_bridge: tester.execution_bridge(),
        phantom_opcode,
    };
    let gpu_chip = PhantomChipGPU::new();
    let cpu_chip = PhantomChip::new(PhantomFiller, tester.dummy_memory_helper());
    let mut harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, NUM_NOPS);

    run_phantom_test(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        phantom_opcode,
        NUM_NOPS,
    );

    harness
        .dense_arena
        .get_record_seeker::<&mut PhantomRecord, EmptyMultiRowLayout>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .simple_test()
        .expect("Verification failed");
}
