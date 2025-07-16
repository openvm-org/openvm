use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::{InsExecutorE1, InsExecutorE2, InstructionExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    Chip, ChipUsageGetter,
};
use openvm_ecc_transpiler::Rv32EdwardsOpcode;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{utils::jacobi, EdwardsAir, EdwardsChip, EdwardsStep};

pub fn te_add_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let d = ExprBuilder::new_const(builder.clone(), d_biguint.clone());
    let one = ExprBuilder::new_const(builder.clone(), BigUint::one());

    let x1y2 = x1.clone() * y2.clone();
    let x2y1 = x2.clone() * y1.clone();
    let y1y2 = y1 * y2;
    let x1x2 = x1 * x2;
    let dx1x2y1y2 = d * x1x2.clone() * y1y2.clone();

    let mut x3 = (x1y2 + x2y1) / (one.clone() + dx1x2y1y2.clone());
    let mut y3 = (y1y2 - a * x1x2) / (one - dx1x2y1y2);

    x3.save_output();
    y3.save_output();

    let builder = builder.borrow().clone();

    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_biguint, d_biguint])
}

#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1, InsExecutorE2)]
pub struct TeAddChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub EdwardsChip<F, 2, BLOCKS, BLOCK_SIZE>,
);

#[allow(clippy::too_many_arguments)]
impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    TeAddChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        mem_helper: SharedMemoryHelper<F>,
        pointer_max_bits: usize,
        config: ExprBuilderConfig,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
        range_checker: SharedVariableRangeCheckerChip,
        a: BigUint,
        d: BigUint,
    ) -> Self {
        // Ensure that the addition operation is complete
        assert!(jacobi(&a.clone().into(), &config.modulus.clone().into()) == 1);
        assert!(jacobi(&d.clone().into(), &config.modulus.clone().into()) == -1);

        let expr = te_add_expr(config, range_checker.bus(), a, d);

        let local_opcode_idx = vec![
            Rv32EdwardsOpcode::TE_ADD as usize,
            Rv32EdwardsOpcode::SETUP_TE_ADD as usize,
        ];

        let air = EdwardsAir::new(
            Rv32VecHeapAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bitwise_lookup_chip.bus(),
                pointer_max_bits,
            ),
            FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
        );

        let step = EdwardsStep::new(
            Rv32VecHeapAdapterStep::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            offset,
            local_opcode_idx,
            vec![],
            range_checker,
            "TeEcAdd",
            true,
        );

        Self(EdwardsChip::new(air, step, mem_helper))
    }
}
