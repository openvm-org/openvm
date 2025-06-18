use std::sync::Arc;

use itertools::izip;
use openvm_circuit::arch::{
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    DenseRecordArena, MemoryConfig, NewVmChipWrapper, VmAirWrapper,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::VariableRangeCheckerBus,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_rv32im_circuit::{
    adapters::{
        Rv32RdWriteAdapterAir, Rv32RdWriteAdapterRecord, Rv32RdWriteAdapterStep, RV32_CELL_BITS,
        RV32_REGISTER_NUM_LIMBS,
    },
    Rv32AuipcAir, Rv32AuipcCoreAir, Rv32AuipcCoreRecord, Rv32AuipcStep, Rv32AuipcStepWithAdapter,
};
use openvm_rv32im_transpiler::Rv32AuipcOpcode::*;
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::Rng;
use stark_backend_gpu::prelude::F;

use crate::{
    extensions::rv32im::Rv32AuipcChipGpu,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

const IMM_BITS: usize = 24;
const MAX_INS_CAPACITY: usize = 128;

#[inline(always)]
fn run_auipc(pc: u32, imm: u32) -> [u8; RV32_REGISTER_NUM_LIMBS] {
    let rd = pc.wrapping_add(imm << RV32_CELL_BITS);
    rd.to_le_bytes()
}

type DenseChip<F> = NewVmChipWrapper<F, Rv32AuipcAir, Rv32AuipcStepWithAdapter, DenseRecordArena>;

fn create_dense_chip(
    tester: &VmChipTestBuilder<F>,
) -> (
    DenseChip<F>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let chip = DenseChip::<F>::new(
        VmAirWrapper::new(
            Rv32RdWriteAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
            Rv32AuipcCoreAir::new(bitwise_bus),
        ),
        Rv32AuipcStep::new(Rv32RdWriteAdapterStep::new(), bitwise_chip.clone()),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    (chip, bitwise_chip)
}

#[test]
fn rand_auipc_tracegen_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, _bitwise_chip) = create_dense_chip(&tester);

    let num_tests: usize = 10;

    let imms = (0..num_tests)
        .map(|_| rng.gen_range(0..(1 << IMM_BITS)) as usize)
        .collect::<Vec<_>>();
    let args = (0..num_tests)
        .map(|_| rng.gen_range(0..32) << 2)
        .collect::<Vec<_>>();
    let from_pcs = (0..num_tests)
        .map(|_| rng.gen_range(0..(1 << PC_BITS)))
        .collect::<Vec<_>>();

    for (&imm, &a, &from_pc) in izip!(imms.iter(), args.iter(), from_pcs.iter()) {
        tester.execute_with_pc(
            &mut chip,
            &Instruction::from_usize(AUIPC.global_opcode(), [a, 0, imm, 1, 0]),
            from_pc,
        );
        let initial_pc = tester.execution.last_from_pc().as_canonical_u32();
        let rd_data = run_auipc(initial_pc, imm as u32);
        assert_eq!(rd_data.map(F::from_canonical_u8), tester.read::<4>(1, a));
    }

    let records = chip
        .arena
        .extract_records::<(Rv32RdWriteAdapterRecord, Rv32AuipcCoreRecord)>();
    eprintln!("{:?}", records);
    assert_eq!(records.len(), num_tests);
    for i in 0..num_tests {
        assert_eq!(records[i].0.from_pc, from_pcs[i]);
        assert_eq!(records[i].0.from_timestamp, 2 * i as u32 + 1);
        assert_eq!(records[i].0.rd_ptr, args[i] as u32);
        assert_eq!(records[i].1.from_pc, from_pcs[i]);
        assert_eq!(records[i].1.imm, imms[i] as u32);
    }

    // GPU part:
    let bitwise_gpu_chip = Arc::new(BitwiseOperationLookupChipGPU::<RV32_CELL_BITS>::new(
        _bitwise_chip.bus(),
    ));
    let mem_config = MemoryConfig::default();
    let var_range_gpu_chip = Arc::new(VariableRangeCheckerChipGPU::new(
        VariableRangeCheckerBus::new(4, mem_config.decomp),
    ));

    let gpu_chip = Rv32AuipcChipGpu::new(
        chip.air,
        var_range_gpu_chip.clone(),
        bitwise_gpu_chip.clone(),
    );

    let records_stream =
        &chip.arena.records_buffer.get_ref()[..chip.arena.records_buffer.position() as usize];
    let auipc_trace = gpu_chip.generate_trace(records_stream);
    // let bitwise_trace = bitwise_gpu_chip.generate_trace();
    // let var_range_trace = var_range_gpu_chip.generate_trace();
    eprintln!("{:?}", auipc_trace);
}
