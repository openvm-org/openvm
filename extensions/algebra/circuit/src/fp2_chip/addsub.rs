use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_derive::{InsExecutorE1, InstructionExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    Chip, ChipUsageGetter,
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{Fp2Air, Fp2Chip, Fp2Step};
use crate::Fp2;

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

// Input: Fp2 * 2
// Output: Fp2
#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1)]
pub struct Fp2AddSubChip<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Fp2AddSubChip<F, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        mem_helper: SharedMemoryHelper<F>,
        pointer_max_bits: usize,
        config: ExprBuilderConfig,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker: SharedVariableRangeCheckerChip,
        height: usize,
    ) -> Self {
        let (expr, is_add_flag, is_sub_flag) = fp2_addsub_expr(config, range_checker.bus());

        let local_opcode_idx = vec![
            Fp2Opcode::ADD as usize,
            Fp2Opcode::SUB as usize,
            Fp2Opcode::SETUP_ADDSUB as usize,
        ];
        let opcode_flag_idx = vec![is_add_flag, is_sub_flag];
        let air = Fp2Air::new(
            Rv32VecHeapAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bitwise_lookup_chip.bus(),
                pointer_max_bits,
            ),
            FieldExpressionCoreAir::new(
                expr.clone(),
                offset,
                local_opcode_idx.clone(),
                opcode_flag_idx.clone(),
            ),
        );

        let step = Fp2Step::new(
            Rv32VecHeapAdapterStep::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            "Fp2AddSub",
            false,
        );
        Self(Fp2Chip::new(air, step, height, mem_helper))
    }
    pub fn expr(&self) -> &FieldExpr {
        &self.0.step.expr
    }
}

#[cfg(test)]
mod tests {

    use halo2curves_axiom::{bn256::Fq2, ff::Field};
    use itertools::Itertools;
    use num_bigint::BigUint;
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
    use openvm_rv32_adapters::rv32_write_heap_default;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};

    use super::Fp2AddSubChip;

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    const MAX_INS_CAPACITY: usize = 128;
    const OFFSET: usize = Fp2Opcode::CLASS_OFFSET;
    type F = BabyBear;

    fn set_and_execute_rand(
        tester: &mut VmChipTestBuilder<F>,
        chip: &mut Fp2AddSubChip<F, 2, NUM_LIMBS>,
        modulus: &BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let x = Fq2::random(&mut rng);
        let y = Fq2::random(&mut rng);
        let inputs = [x.c0, x.c1, y.c0, y.c1].map(bn254_fq_to_biguint);

        let expected_sum = bn254_fq2_to_biguint_vec(x + y);
        let r_sum = chip
            .expr()
            .execute_with_output(inputs.to_vec(), vec![true, false]);
        assert_eq!(r_sum.len(), 2);
        assert_eq!(r_sum[0], expected_sum[0]);
        assert_eq!(r_sum[1], expected_sum[1]);

        let expected_sub = bn254_fq2_to_biguint_vec(x - y);
        let r_sub = chip
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
        let modulus = biguint_to_limbs::<NUM_LIMBS>(modulus.clone(), LIMB_BITS)
            .map(BabyBear::from_canonical_u32);
        let zero = [BabyBear::ZERO; NUM_LIMBS];
        let setup_instruction = rv32_write_heap_default(
            tester,
            vec![modulus, zero],
            vec![zero; 2],
            OFFSET + Fp2Opcode::SETUP_ADDSUB as usize,
        );
        let instruction1 = rv32_write_heap_default(
            tester,
            x_limbs.clone(),
            y_limbs.clone(),
            OFFSET + Fp2Opcode::ADD as usize,
        );
        let instruction2 =
            rv32_write_heap_default(tester, x_limbs, y_limbs, OFFSET + Fp2Opcode::SUB as usize);

        tester.execute(chip, &setup_instruction);
        tester.execute(chip, &instruction1);
        tester.execute(chip, &instruction2);
    }

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

        let mut chip = Fp2AddSubChip::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            tester.memory_helper(),
            tester.address_bits(),
            config,
            OFFSET,
            bitwise_chip.clone(),
            tester.range_checker(),
            MAX_INS_CAPACITY,
        );

        let num_ops = 10;
        for _ in 0..num_ops {
            set_and_execute_rand(&mut tester, &mut chip, &modulus);
        }
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
