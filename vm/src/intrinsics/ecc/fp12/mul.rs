use std::{cell::RefCell, rc::Rc};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_circuit_primitives::var_range::VariableRangeCheckerBus;
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr},
    field_extension::Fp12,
};
use axvm_circuit_derive::InstructionExecutor;
use p3_field::PrimeField32;

use crate::{
    arch::{instructions::Fp12Opcode, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip,
    system::memory::MemoryControllerRef,
};

// Input: Fp12 * 2
// Output: Fp12
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct Fp12MulChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Fp12MulChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        memory_controller: MemoryControllerRef<F>,
        config: ExprBuilderConfig,
        xi: [isize; 2],
        offset: usize,
    ) -> Self {
        let expr = fp12_mul_expr(config, memory_controller.borrow().range_checker.bus(), xi);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![Fp12Opcode::MUL as usize],
            vec![],
            memory_controller.borrow().range_checker.clone(),
            "Fp12Mul",
        );
        Self(VmChipWrapper::new(adapter, core, memory_controller))
    }
}

pub fn fp12_mul_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
    xi: [isize; 2],
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x = Fp12::new(builder.clone());
    let mut y = Fp12::new(builder.clone());
    let mut res = x.mul(&mut y, xi);
    res.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ax_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    };
    use ax_ecc_primitives::{
        field_expression::ExprBuilderConfig,
        test_utils::{bn254_fq12_to_biguint_vec, bn254_fq2_to_biguint_vec},
    };
    use axvm_ecc::algebra::field::FieldExtension;
    use axvm_ecc_constants::BN254;
    use axvm_instructions::{riscv::RV32_CELL_BITS, UsizeOpcode};
    use halo2curves_axiom::{bn256::Fq12, ff::Field};
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        arch::{testing::VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        utils::{biguint_to_limbs, rv32_write_heap_default_with_increment},
    };

    const LIMB_BITS: usize = 8;
    type F = BabyBear;

    #[test]
    fn test_fp12_mul_bn254() {
        const NUM_LIMBS: usize = 32;
        const BLOCK_SIZE: usize = 32;

        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: BN254.MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));
        let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 12, BLOCK_SIZE, BLOCK_SIZE>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
            bitwise_chip.clone(),
        );

        let mut chip = Fp12MulChip::new(
            adapter,
            tester.memory_controller(),
            config,
            BN254.XI,
            Fp12Opcode::default_offset(),
        );

        let mut rng = StdRng::seed_from_u64(64);
        let x = Fq12::random(&mut rng);
        let y = Fq12::random(&mut rng);
        let inputs = [x.to_coeffs(), y.to_coeffs()]
            .concat()
            .iter()
            .flat_map(|&x| bn254_fq2_to_biguint_vec(x))
            .collect::<Vec<_>>();

        let cmp = bn254_fq12_to_biguint_vec(x * y);
        let res = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.clone(), vec![true]);
        assert_eq!(res.len(), cmp.len());
        for i in 0..res.len() {
            assert_eq!(res[i], cmp[i]);
        }

        let x_limbs = inputs[..12]
            .iter()
            .map(|x| {
                biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS)
                    .map(BabyBear::from_canonical_u32)
            })
            .collect_vec();
        let y_limbs = inputs[12..]
            .iter()
            .map(|y| {
                biguint_to_limbs::<NUM_LIMBS>(y.clone(), LIMB_BITS)
                    .map(BabyBear::from_canonical_u32)
            })
            .collect_vec();
        let instruction = rv32_write_heap_default_with_increment(
            &mut tester,
            x_limbs,
            y_limbs,
            512,
            chip.0.core.air.offset + Fp12Opcode::MUL as usize,
        );
        tester.execute(&mut chip, instruction);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
