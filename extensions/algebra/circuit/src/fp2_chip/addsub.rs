use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::Fp2;

// Input: Fp2 * 2
// Output: Fp2
#[derive(Chip, ChipUsageGetter, InstructionExecutor)]
pub struct Fp2AddSubChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub  VmChipWrapper<
        F,
        Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreChip,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Fp2AddSubChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterChip<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        config: ExprBuilderConfig,
        offset: usize,
        range_checker: SharedVariableRangeCheckerChip,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        let (expr, is_add_flag, is_sub_flag) = fp2_addsub_expr(config, range_checker.bus());
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![
                Fp2Opcode::ADD as usize,
                Fp2Opcode::SUB as usize,
                Fp2Opcode::SETUP_ADDSUB as usize,
            ],
            vec![is_add_flag, is_sub_flag],
            range_checker,
            "Fp2AddSub",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
    }
}

pub fn fp2_addsub_expr(
    config: ExprBuilderConfig,
    range_bus: VariableRangeCheckerBus,
) -> (FieldExpr, usize, usize) {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x = Fp2::new(builder.clone());
    let mut y = Fp2::new(builder.clone());
    let add = x.add(&mut y);
    let sub = x.sub(&mut y);

    let is_add_flag = builder.borrow_mut().new_flag();
    let is_sub_flag = builder.borrow_mut().new_flag();
    let diff = Fp2::select(is_sub_flag, &sub, &x);
    let mut z = Fp2::select(is_add_flag, &add, &diff);
    z.save_output();

    let builder = builder.borrow().clone();
    (
        FieldExpr::new(builder, range_bus, true),
        is_add_flag,
        is_sub_flag,
    )
}

#[cfg(test)]
mod tests {

    use halo2curves_axiom::{bn256::Fq2, ff::Field};
    use itertools::Itertools;
    use openvm_algebra_transpiler::Fp2Opcode;
    use openvm_circuit::arch::testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS};
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
    use openvm_mod_circuit_builder::{
        test_utils::{biguint_to_limbs, bn254_fq2_to_biguint_vec, bn254_fq_to_biguint},
        ExprBuilderConfig,
    };
    use openvm_pairing_guest::bn254::BN254_MODULUS;
    use openvm_rv32_adapters::{rv32_write_heap_default, Rv32VecHeapAdapterChip};
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};

    use super::Fp2AddSubChip;

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    type F = BabyBear;

    #[test]
    fn test_fp2_addsub() {
        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let modulus = BN254_MODULUS.clone();
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let adapter = Rv32VecHeapAdapterChip::<F, 2, 2, 2, NUM_LIMBS, NUM_LIMBS>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
        );
        let mut chip = Fp2AddSubChip::new(
            adapter,
            config,
            Fp2Opcode::CLASS_OFFSET,
            tester.range_checker(),
            tester.offline_memory_mutex_arc(),
        );

        let mut rng = StdRng::seed_from_u64(42);
        let x = Fq2::random(&mut rng);
        let y = Fq2::random(&mut rng);
        let inputs = [x.c0, x.c1, y.c0, y.c1].map(bn254_fq_to_biguint);

        let expected_sum = bn254_fq2_to_biguint_vec(x + y);
        let r_sum = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![true, false]);
        assert_eq!(r_sum.len(), 2);
        assert_eq!(r_sum[0], expected_sum[0]);
        assert_eq!(r_sum[1], expected_sum[1]);

        let expected_sub = bn254_fq2_to_biguint_vec(x - y);
        let r_sub = chip
            .0
            .core
            .expr()
            .execute_with_output(inputs.to_vec(), vec![false, true]);
        assert_eq!(r_sub.len(), 2);
        assert_eq!(r_sub[0], expected_sub[0]);
        assert_eq!(r_sub[1], expected_sub[1]);

        let x_limbs = inputs[0..2]
            .iter()
            .map(|x| {
                biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS)
                    .map(BabyBear::from_canonical_u32)
            })
            .collect_vec();
        let y_limbs = inputs[2..4]
            .iter()
            .map(|x| {
                biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS)
                    .map(BabyBear::from_canonical_u32)
            })
            .collect_vec();
        let modulus =
            biguint_to_limbs::<NUM_LIMBS>(modulus, LIMB_BITS).map(BabyBear::from_canonical_u32);
        let zero = [BabyBear::ZERO; NUM_LIMBS];
        let setup_instruction = rv32_write_heap_default(
            &mut tester,
            vec![modulus, zero],
            vec![zero; 2],
            chip.0.core.air.offset + Fp2Opcode::SETUP_ADDSUB as usize,
        );
        let instruction1 = rv32_write_heap_default(
            &mut tester,
            x_limbs.clone(),
            y_limbs.clone(),
            chip.0.core.air.offset + Fp2Opcode::ADD as usize,
        );
        let instruction2 = rv32_write_heap_default(
            &mut tester,
            x_limbs,
            y_limbs,
            chip.0.core.air.offset + Fp2Opcode::SUB as usize,
        );
        tester.execute(&mut chip, &setup_instruction);
        tester.execute(&mut chip, &instruction1);
        tester.execute(&mut chip, &instruction2);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
