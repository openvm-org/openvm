use openvm_circuit::arch::{
    testing::BITWISE_OP_LOOKUP_BUS, DenseRecordArena, EmptyAdapterCoreLayout, MatrixRecordArena,
    NewVmChipWrapper, VmAirWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_rv32im_circuit::{
    adapters::{
        Rv32RdWriteAdapterAir, Rv32RdWriteAdapterRecord, Rv32RdWriteAdapterStep, RV32_CELL_BITS,
    },
    Rv32AuipcAir, Rv32AuipcCoreAir, Rv32AuipcCoreRecord, Rv32AuipcStep, Rv32AuipcStepWithAdapter,
};
use openvm_rv32im_transpiler::Rv32AuipcOpcode::*;
use openvm_stark_backend::verifier::VerificationError;
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::Rng;
use stark_backend_gpu::prelude::F;

use crate::{extensions::rv32im::Rv32AuipcChipGpu, testing::GpuChipTestBuilder};

const IMM_BITS: usize = 24;
const MAX_INS_CAPACITY: usize = 128;

type DenseChip<F> = NewVmChipWrapper<F, Rv32AuipcAir, Rv32AuipcStepWithAdapter, DenseRecordArena>;
type SparseChip<F> =
    NewVmChipWrapper<F, Rv32AuipcAir, Rv32AuipcStepWithAdapter, MatrixRecordArena<F>>;

#[test]
fn rand_auipc_tracegen_test() {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let mut tester = GpuChipTestBuilder::default()
        .with_variable_range_checker()
        .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS));
    let mut rng = create_seeded_rng();

    let cpu_bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut dense_chip = create_dense_chip(&tester, cpu_bitwise_chip.clone());

    let mut gpu_chip = Rv32AuipcChipGpu::new(
        dense_chip.air,
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        None,
    );
    let mut cpu_chip = create_sparse_chip(&tester, cpu_bitwise_chip.clone());

    for _ in 0..100 {
        let imm = rng.gen_range(0..(1 << IMM_BITS)) as usize;
        let a = rng.gen_range(0..32) << 2;
        let initial_pc = rng.gen_range(0..(1 << PC_BITS));

        tester.execute_with_pc(
            &mut dense_chip,
            &Instruction::from_usize(AUIPC.global_opcode(), [a, 0, imm, 1, 0]),
            initial_pc,
        );
    }

    // TODO[stephen]: The GPU chip should probably own the arena and lend it to
    // the executor, but because currently in OpenVM the executor owns it we need
    // to do something like this. This should be updated once the Chip definition
    // is updated.
    type Record<'a> = (
        &'a mut Rv32RdWriteAdapterRecord,
        &'a mut Rv32AuipcCoreRecord,
    );
    dense_chip
        .arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut cpu_chip.arena,
            EmptyAdapterCoreLayout::<F, Rv32RdWriteAdapterStep>::new(),
        );
    gpu_chip.arena = Some(&dense_chip.arena);

    // TODO[stephen]: Because memory is not implemented yet, this test fails
    // interaction constraints. Once memory is completed, we should make sure
    // that verification passes.
    tester
        .build()
        .load_and_compare(gpu_chip, cpu_chip)
        .finalize()
        .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
}

fn create_dense_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> DenseChip<F> {
    let mut chip = DenseChip::<F>::new(
        VmAirWrapper::new(
            Rv32RdWriteAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
            Rv32AuipcCoreAir::new(bitwise.bus()),
        ),
        Rv32AuipcStep::new(Rv32RdWriteAdapterStep::new(), bitwise),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn create_sparse_chip(
    tester: &GpuChipTestBuilder,
    bitwise: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) -> SparseChip<F> {
    let mut chip = SparseChip::<F>::new(
        VmAirWrapper::new(
            Rv32RdWriteAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
            Rv32AuipcCoreAir::new(bitwise.bus()),
        ),
        Rv32AuipcStep::new(Rv32RdWriteAdapterStep::new(), bitwise),
        tester.cpu_memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}
