use std::{cell::RefCell, rc::Rc};

use openvm_algebra_circuit::Fp2;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::{InsExecutorE1, InstructionExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir,
};
use openvm_pairing_transpiler::PairingOpcode;
use openvm_rv32_adapters::{Rv32VecHeapTwoReadsAdapterAir, Rv32VecHeapTwoReadsAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{Fp12, PairingTwoReadsAir, PairingTwoReadsChip, PairingTwoReadsStep};

// Input: Fp12 (12 field elements), [Fp2; 5] (5 x 2 field elements)
// Output: Fp12 (12 field elements)
#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1)]
pub struct EcLineMulBy01234Chip<
    F: PrimeField32,
    const INPUT_BLOCKS1: usize,
    const INPUT_BLOCKS2: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(pub PairingTwoReadsChip<F, INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS1: usize,
        const INPUT_BLOCKS2: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > EcLineMulBy01234Chip<F, INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        mem_helper: SharedMemoryHelper<F>,
        pointer_max_bits: usize,
        config: ExprBuilderConfig,
        xi: [isize; 2],
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker: SharedVariableRangeCheckerChip,
        height: usize,
    ) -> Self {
        assert!(
            xi[0].unsigned_abs() < 1 << config.limb_bits,
            "expect xi to be small"
        ); // not a hard rule, but we expect xi to be small
        assert!(
            xi[1].unsigned_abs() < 1 << config.limb_bits,
            "expect xi to be small"
        );

        let expr = mul_by_01234_expr(config, range_checker.bus(), xi);
        let local_opcode_idx = vec![PairingOpcode::MUL_BY_01234 as usize];

        let air = PairingTwoReadsAir::new(
            Rv32VecHeapTwoReadsAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bitwise_lookup_chip.bus(),
                pointer_max_bits,
            ),
            FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
        );

        let step = PairingTwoReadsStep::new(
            Rv32VecHeapTwoReadsAdapterStep::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            offset,
            local_opcode_idx,
            vec![],
            range_checker,
            "MulBy01234",
            false,
        );
        Self(PairingTwoReadsChip::new(air, step, height, mem_helper))
    }
}

pub fn mul_by_01234_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    xi: [isize; 2],
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config.clone(), range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut f = Fp12::new(builder.clone());
    let mut x0 = Fp2::new(builder.clone());
    let mut x1 = Fp2::new(builder.clone());
    let mut x2 = Fp2::new(builder.clone());
    let mut x3 = Fp2::new(builder.clone());
    let mut x4 = Fp2::new(builder.clone());

    let mut r = f.mul_by_01234(&mut x0, &mut x1, &mut x2, &mut x3, &mut x4, xi);
    r.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus, false)
}
