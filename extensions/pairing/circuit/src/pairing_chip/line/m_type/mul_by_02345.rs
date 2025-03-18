use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use openvm_algebra_circuit::Fp2;
use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip,
};
use openvm_pairing_transpiler::PairingOpcode;
use openvm_rv32_adapters::Rv32VecHeapTwoReadsAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::Fp12;

// Input: 2 Fp12: 2 x 12 field elements
// Output: Fp12 -> 12 field elements
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct EcLineMulBy02345Chip<
    F: PrimeField32,
    const INPUT_BLOCKS1: usize,
    const INPUT_BLOCKS2: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapTwoReadsAdapterChip<
            F,
            INPUT_BLOCKS1,
            INPUT_BLOCKS2,
            OUTPUT_BLOCKS,
            BLOCK_SIZE,
            BLOCK_SIZE,
        >,
        FieldExpressionCoreChip,
    >,
);

impl<
        F: PrimeField32,
        const INPUT_BLOCKS1: usize,
        const INPUT_BLOCKS2: usize,
        const OUTPUT_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > EcLineMulBy02345Chip<F, INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapTwoReadsAdapterChip<
            F,
            INPUT_BLOCKS1,
            INPUT_BLOCKS2,
            OUTPUT_BLOCKS,
            BLOCK_SIZE,
            BLOCK_SIZE,
        >,
        range_checker: SharedVariableRangeCheckerChip,
        config: ExprBuilderConfig,
        xi: [isize; 2],
        offset: usize,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        assert!(
            xi[0].unsigned_abs() < 1 << config.limb_bits,
            "expect xi to be small"
        ); // not a hard rule, but we expect xi to be small
        assert!(
            xi[1].unsigned_abs() < 1 << config.limb_bits,
            "expect xi to be small"
        );
        let expr = mul_by_02345_expr(config, range_checker.bus(), xi);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![PairingOpcode::MUL_BY_02345 as usize],
            vec![],
            range_checker,
            "MulBy02345",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
    }
}

pub fn mul_by_02345_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    xi: [isize; 2],
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config.clone(), range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut f = Fp12::new(builder.clone());
    let mut x0 = Fp2::new(builder.clone());
    let mut x2 = Fp2::new(builder.clone());
    let mut x3 = Fp2::new(builder.clone());
    let mut x4 = Fp2::new(builder.clone());
    let mut x5 = Fp2::new(builder.clone());

    let mut r = f.mul_by_02345(&mut x0, &mut x2, &mut x3, &mut x4, &mut x5, xi);
    r.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus, false)
}
