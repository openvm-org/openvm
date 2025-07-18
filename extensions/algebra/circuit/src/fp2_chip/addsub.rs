use std::{cell::RefCell, rc::Rc};

use openvm_algebra_transpiler::Fp2Opcode;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionFiller,
};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller, Rv32VecHeapAdapterStep,
};

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
fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
) -> (FieldExpr, Vec<usize>, Vec<usize>) {
    let (expr, is_add_flag, is_sub_flag) = fp2_addsub_expr(config, range_checker_bus);

    let local_opcode_idx = vec![
        Fp2Opcode::ADD as usize,
        Fp2Opcode::SUB as usize,
        Fp2Opcode::SETUP_ADDSUB as usize,
    ];
    let opcode_flag_idx = vec![is_add_flag, is_sub_flag];

    (expr, local_opcode_idx, opcode_flag_idx)
}

pub fn get_fp2_addsub_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
) -> Fp2Air<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);
    Fp2Air::new(
        Rv32VecHeapAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr, offset, local_opcode_idx, opcode_flag_idx),
    )
}

pub fn get_fp2_addsub_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> Fp2Step<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker_bus);

    Fp2Step::new(
        Rv32VecHeapAdapterStep::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        opcode_flag_idx,
        "Fp2AddSub",
    )
}

pub fn get_fp2_addsub_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
) -> Fp2Chip<F, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx, opcode_flag_idx) = gen_base_expr(config, range_checker.bus());
    Fp2Chip::new(
        FieldExpressionFiller::new(
            Rv32VecHeapAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            false,
        ),
        mem_helper,
    )
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use halo2curves_axiom::{bn256::Fq2, ff::Field};
    use itertools::Itertools;
    use num_bigint::BigUint;
    use openvm_algebra_transpiler::Fp2Opcode;
    use openvm_circuit::arch::testing::{
        TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
    };
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, BitwiseOperationLookupChip,
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

    use crate::fp2_chip::{
        get_fp2_addsub_air, get_fp2_addsub_chip, get_fp2_addsub_step, Fp2Air, Fp2Chip, Fp2Step,
    };

    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    const MAX_INS_CAPACITY: usize = 128;
    const OFFSET: usize = Fp2Opcode::CLASS_OFFSET;
    type F = BabyBear;
    type Harness =
        TestChipHarness<F, Fp2Step<2, NUM_LIMBS>, Fp2Air<2, NUM_LIMBS>, Fp2Chip<F, 2, NUM_LIMBS>>;

    fn set_and_execute_rand(
        tester: &mut VmChipTestBuilder<F>,
        harness: &mut Harness,
        modulus: &BigUint,
    ) {
        let mut rng = create_seeded_rng();
        let x = Fq2::random(&mut rng);
        let y = Fq2::random(&mut rng);
        let inputs = [x.c0, x.c1, y.c0, y.c1].map(bn254_fq_to_biguint);

        let expected_sum = bn254_fq2_to_biguint_vec(x + y);
        let r_sum = harness
            .executor
            .expr
            .execute_with_output(inputs.to_vec(), vec![true, false]);
        assert_eq!(r_sum.len(), 2);
        assert_eq!(r_sum[0], expected_sum[0]);
        assert_eq!(r_sum[1], expected_sum[1]);

        let expected_sub = bn254_fq2_to_biguint_vec(x - y);
        let r_sub = harness
            .executor
            .expr
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

        tester.execute(harness, &setup_instruction);
        tester.execute(harness, &instruction1);
        tester.execute(harness, &instruction2);
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
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_fp2_addsub_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            tester.range_checker().bus(),
            bitwise_bus,
            tester.address_bits(),
            OFFSET,
        );
        let executor = get_fp2_addsub_step(
            config.clone(),
            tester.range_checker().bus(),
            tester.address_bits(),
            OFFSET,
        );
        let chip = get_fp2_addsub_chip(
            config,
            tester.memory_helper(),
            tester.range_checker(),
            bitwise_chip.clone(),
            tester.address_bits(),
        );
        let mut harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

        let num_ops = 10;
        for _ in 0..num_ops {
            set_and_execute_rand(&mut tester, &mut harness, &modulus);
        }
        let tester = tester
            .build()
            .load(harness)
            .load_periphery((bitwise_chip.air, bitwise_chip))
            .finalize();
        tester.simple_test().expect("Verification failed");
    }
}
