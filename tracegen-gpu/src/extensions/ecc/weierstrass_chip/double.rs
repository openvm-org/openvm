use std::sync::Arc;

use derive_new::new;
use num_bigint::BigUint;
use openvm_circuit::arch::{AdapterCoreLayout, DenseRecordArena, RecordSeeker};
use openvm_ecc_circuit::ec_double_ne_expr;
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{
    ExprBuilderConfig, FieldExpressionCoreAir, FieldExpressionMetadata,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterCols, Rv32VecHeapAdapterStep};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend};

use crate::{
    extensions::ecc::EccRecord,
    get_empty_air_proving_ctx,
    mod_builder::field_expression::FieldExpressionChipGPU,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(new)]
pub struct WeierstrassDoubleChipGpu<const BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub config: ExprBuilderConfig,
    pub offset: usize,
    pub a_biguint: BigUint,
    pub pointer_max_bits: u32,
    pub timestamp_max_bits: u32,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for WeierstrassDoubleChipGpu<BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let range_bus = self.range_checker.cpu_chip.as_ref().unwrap().bus();
        let expr = ec_double_ne_expr(self.config.clone(), range_bus, self.a_biguint.clone());

        let total_input_limbs = expr.builder.num_input * expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterStep<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            EccRecord<1, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;

        let local_opcode_idx = vec![
            Rv32WeierstrassOpcode::EC_DOUBLE as usize,
            Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
        ];

        let air = FieldExpressionCoreAir::new(expr, self.offset, local_opcode_idx, vec![]);

        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();

        let d_records = records.to_device().unwrap();

        let field_expr_chip = FieldExpressionChipGPU::new(
            air,
            d_records,
            num_records,
            record_size,
            adapter_width,
            BLOCKS,
            self.range_checker.clone(),
            self.bitwise_lookup.clone(),
            self.pointer_max_bits,
            self.timestamp_max_bits,
        );

        let d_trace = field_expr_chip.generate_field_trace();

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use num_bigint::BigUint;
    use num_traits::{FromPrimitive, Zero};
    use openvm_circuit::arch::testing::memory::gen_pointer;
    use openvm_circuit_primitives::{
        bigint::utils::secp256k1_coord_prime, bitwise_op_lookup::BitwiseOperationLookupChip,
        var_range::VariableRangeCheckerChip,
    };
    use openvm_ecc_circuit::{
        get_ec_double_air, get_ec_double_chip, get_ec_double_step, EcDoubleStep, WeierstrassAir,
        WeierstrassChip,
    };
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        LocalOpcode, VmOpcode,
    };
    use openvm_mod_circuit_builder::{test_utils::biguint_to_limbs, ExprBuilderConfig};
    use openvm_pairing_guest::bls12_381::BLS12_381_MODULUS;
    use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};

    use super::*;
    use crate::testing::{
        default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
        GpuTestChipHarness,
    };

    const LIMB_BITS: usize = 8;
    const MAX_INS_CAPACITY: usize = 128;

    fn create_test_harness<const BLOCKS: usize, const BLOCK_SIZE: usize>(
        tester: &GpuChipTestBuilder,
        config: ExprBuilderConfig,
        offset: usize,
        a_biguint: BigUint,
    ) -> GpuTestChipHarness<
        F,
        EcDoubleStep<BLOCKS, BLOCK_SIZE>,
        WeierstrassAir<1, BLOCKS, BLOCK_SIZE>,
        WeierstrassDoubleChipGpu<BLOCKS, BLOCK_SIZE>,
        WeierstrassChip<F, 1, BLOCKS, BLOCK_SIZE>,
    > {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_ec_double_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            bitwise_bus,
            tester.address_bits(),
            offset,
            a_biguint.clone(),
        );
        let executor = get_ec_double_step(
            config.clone(),
            range_bus,
            tester.address_bits(),
            offset,
            a_biguint.clone(),
        );

        let cpu_chip = get_ec_double_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            dummy_bitwise_chip,
            tester.address_bits(),
            a_biguint.clone(),
        );
        let gpu_chip = WeierstrassDoubleChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            config,
            offset,
            a_biguint,
            tester.address_bits() as u32,
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute_ec_double<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
    >(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<
            F,
            EcDoubleStep<BLOCKS, BLOCK_SIZE>,
            WeierstrassAir<1, BLOCKS, BLOCK_SIZE>,
            WeierstrassDoubleChipGpu<BLOCKS, BLOCK_SIZE>,
            WeierstrassChip<F, 1, BLOCKS, BLOCK_SIZE>,
        >,
        rng: &mut StdRng,
        modulus: &BigUint,
        a_biguint: &BigUint,
        is_setup: bool,
        offset: usize,
    ) {
        let (x1, y1, op_local) = if is_setup {
            (
                modulus.clone(),
                a_biguint.clone(),
                Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
            )
        } else if rng.gen_bool(0.5) {
            if rng.gen_bool(0.5) {
                let x = BigUint::from_u32(2).unwrap();
                let y = BigUint::from_str(
                    "69211104694897500952317515077652022726490027694212560352756646854116994689233",
                )
                .unwrap();
                (x, y, Rv32WeierstrassOpcode::EC_DOUBLE as usize)
            } else {
                let x = BigUint::from_u32(1).unwrap();
                let y = BigUint::from_str(
                    "29896722852569046015560700294576055776214335159245303116488692907525646231534",
                )
                .unwrap();
                (x, y, Rv32WeierstrassOpcode::EC_DOUBLE as usize)
            }
        } else {
            let x_digits: Vec<_> = (0..NUM_LIMBS)
                .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                .collect();
            let mut x = BigUint::new(x_digits);
            x %= modulus;

            let y_digits: Vec<_> = (0..NUM_LIMBS)
                .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                .collect();
            let mut y = BigUint::new(y_digits);
            y %= modulus;

            (x, y, Rv32WeierstrassOpcode::EC_DOUBLE as usize)
        };

        let ptr_as = RV32_REGISTER_AS as usize;
        let data_as = RV32_MEMORY_AS as usize;

        let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        let p1_base_addr = 0u32;
        let result_base_addr = 256u32;

        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs1_ptr,
            p1_base_addr.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            rd_ptr,
            result_base_addr.to_le_bytes().map(F::from_canonical_u8),
        );

        let x1_limbs = biguint_to_limbs::<NUM_LIMBS>(x1, LIMB_BITS).map(F::from_canonical_u32);
        let y1_limbs = biguint_to_limbs::<NUM_LIMBS>(y1, LIMB_BITS).map(F::from_canonical_u32);

        for i in (0..NUM_LIMBS).step_by(BLOCK_SIZE) {
            tester.write::<BLOCK_SIZE>(
                data_as,
                p1_base_addr as usize + i,
                x1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
            );

            tester.write::<BLOCK_SIZE>(
                data_as,
                (p1_base_addr + NUM_LIMBS as u32) as usize + i,
                y1_limbs[i..i + BLOCK_SIZE].try_into().unwrap(),
            );
        }

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op_local),
            rd_ptr as isize,
            rs1_ptr as isize,
            0,
            ptr_as as isize,
            data_as as isize,
        );

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instruction,
        );
    }

    fn run_test_with_config<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
    >(
        modulus: BigUint,
        a_biguint: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let offset = Rv32WeierstrassOpcode::CLASS_OFFSET;
        let config = ExprBuilderConfig {
            modulus: modulus.clone(),
            num_limbs: NUM_LIMBS,
            limb_bits: LIMB_BITS,
        };

        let mut harness =
            create_test_harness::<BLOCKS, BLOCK_SIZE>(&tester, config, offset, a_biguint.clone());

        for i in 0..num_ops {
            set_and_execute_ec_double::<BLOCKS, BLOCK_SIZE, NUM_LIMBS>(
                &mut tester,
                &mut harness,
                &mut rng,
                &modulus,
                &a_biguint,
                i == 0,
                offset,
            );
        }

        harness
            .dense_arena
            .get_record_seeker::<EccRecord<1, BLOCKS, BLOCK_SIZE>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                harness.executor.get_record_layout::<F>(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[test]
    fn test_weierstrass_double_gpu_2x32() {
        run_test_with_config::<2, 32, 32>(secp256k1_coord_prime(), BigUint::zero(), 50);
    }

    #[test]
    fn test_weierstrass_double_gpu_2x32_nonzero_a() {
        let coeff_a = (-halo2curves_axiom::secp256r1::Fp::from(3)).to_bytes();
        run_test_with_config::<2, 32, 32>(
            secp256k1_coord_prime(),
            BigUint::from_bytes_le(&coeff_a),
            2,
        );
    }

    #[test]
    fn test_weierstrass_double_gpu_6x16() {
        run_test_with_config::<6, 16, 48>(BLS12_381_MODULUS.clone(), BigUint::zero(), 50);
    }
}
