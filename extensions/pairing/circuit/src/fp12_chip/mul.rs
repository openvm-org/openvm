use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use openvm_circuit::{arch::VmChipWrapper, system::memory::OfflineMemory};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::{Chip, ChipUsageGetter};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip,
};
use openvm_pairing_transpiler::Fp12Opcode;
use openvm_rv32_adapters::Rv32VecHeapAdapterChip;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::Fp12;
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
        config: ExprBuilderConfig,
        xi: [isize; 2],
        offset: usize,
        range_checker: SharedVariableRangeCheckerChip,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        let expr = fp12_mul_expr(config, range_checker.bus(), xi);
        let core = FieldExpressionCoreChip::new(
            expr,
            offset,
            vec![Fp12Opcode::MUL as usize],
            vec![],
            range_checker,
            "Fp12Mul",
            false,
        );
        Self(VmChipWrapper::new(adapter, core, offline_memory))
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
    FieldExpr::new(builder, range_bus, false)
}

#[cfg(test)]
mod tests {
    use halo2curves_axiom::{bn256::Fq12, ff::Field};
    use itertools::Itertools;
    use openvm_circuit::arch::testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS};
    use openvm_circuit_primitives::bitwise_op_lookup::{
        BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
    };
    use openvm_ecc_guest::algebra::field::FieldExtension;
    use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
    use openvm_mod_circuit_builder::{
        test_utils::{biguint_to_limbs, bn254_fq12_to_biguint_vec, bn254_fq2_to_biguint_vec},
        ExprBuilderConfig,
    };
    use openvm_pairing_guest::bn254::{BN254_MODULUS, BN254_XI_ISIZE};
    use openvm_rv32_adapters::rv32_write_heap_default_with_increment;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    const LIMB_BITS: usize = 8;
    type F = BabyBear;

    #[test]
    fn test_fp12_mul_bn254() {
        const NUM_LIMBS: usize = 32;
        const BLOCK_SIZE: usize = 32;

        let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
        let config = ExprBuilderConfig {
            modulus: BN254_MODULUS.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };
        let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
        let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 12, BLOCK_SIZE, BLOCK_SIZE>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
        );

        let mut chip = Fp12MulChip::new(
            adapter,
            config,
            BN254_XI_ISIZE,
            Fp12Opcode::CLASS_OFFSET,
            tester.range_checker(),
            tester.offline_memory_mutex_arc(),
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
        tester.execute(&mut chip, &instruction);
        let tester = tester.build().load(chip).load(bitwise_chip).finalize();
        tester.simple_test().expect("Verification failed");
    }
}
