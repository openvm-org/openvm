use std::sync::Arc;

use derive_new::new;
use openvm_circuit::arch::{AdapterCoreLayout, DenseRecordArena, RecordSeeker};
use openvm_ecc_circuit::ec_add_ne_expr;
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
pub struct WeierstrassAddNeChipGpu<const BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub config: ExprBuilderConfig,
    pub offset: usize,
    pub pointer_max_bits: u32,
    pub timestamp_max_bits: u32,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for WeierstrassAddNeChipGpu<BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let range_bus = self.range_checker.cpu_chip.as_ref().unwrap().bus();
        let expr = ec_add_ne_expr(self.config.clone(), range_bus);

        let total_input_limbs = expr.builder.num_input * expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            EccRecord<2, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;

        let local_opcode_idx = vec![
            Rv32WeierstrassOpcode::EC_ADD_NE as usize,
            Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
        ];

        let air = FieldExpressionCoreAir::new(expr, self.offset, local_opcode_idx, vec![]);

        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();

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
    use num_traits::{FromPrimitive, One};
    use openvm_circuit::arch::testing::memory::gen_pointer;
    use openvm_circuit_primitives::{
        bigint::utils::secp256k1_coord_prime, bitwise_op_lookup::BitwiseOperationLookupChip,
        var_range::VariableRangeCheckerChip,
    };
    use openvm_ecc_circuit::{
        get_ec_addne_air, get_ec_addne_chip, get_ec_addne_step, EcAddNeStep, WeierstrassAir,
        WeierstrassChip,
    };
    use openvm_instructions::{instruction::Instruction, LocalOpcode, VmOpcode};
    use openvm_mod_circuit_builder::{test_utils::biguint_to_limbs, ExprBuilderConfig};
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
    ) -> GpuTestChipHarness<
        F,
        EcAddNeStep<BLOCKS, BLOCK_SIZE>,
        WeierstrassAir<2, BLOCKS, BLOCK_SIZE>,
        WeierstrassAddNeChipGpu<BLOCKS, BLOCK_SIZE>,
        WeierstrassChip<F, 2, BLOCKS, BLOCK_SIZE>,
    > {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let range_bus = default_var_range_checker_bus();
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        let air = get_ec_addne_air(
            tester.execution_bridge(),
            tester.memory_bridge(),
            config.clone(),
            range_bus,
            bitwise_bus,
            tester.address_bits(),
            offset,
        );
        let executor = get_ec_addne_step(config.clone(), range_bus, tester.address_bits(), offset);

        let cpu_chip = get_ec_addne_chip(
            config.clone(),
            tester.dummy_memory_helper(),
            dummy_range_checker_chip,
            dummy_bitwise_chip,
            tester.address_bits(),
        );
        let gpu_chip = WeierstrassAddNeChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            config,
            offset,
            tester.address_bits() as u32,
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute_ec_add<
        const BLOCKS: usize,
        const BLOCK_SIZE: usize,
        const NUM_LIMBS: usize,
    >(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<
            F,
            EcAddNeStep<BLOCKS, BLOCK_SIZE>,
            WeierstrassAir<2, BLOCKS, BLOCK_SIZE>,
            WeierstrassAddNeChipGpu<BLOCKS, BLOCK_SIZE>,
            WeierstrassChip<F, 2, BLOCKS, BLOCK_SIZE>,
        >,
        rng: &mut StdRng,
        modulus: &BigUint,
        is_setup: bool,
        offset: usize,
    ) {
        let (x1, y1, x2, y2, op_local) = if is_setup {
            (
                modulus.clone(),
                BigUint::one(),
                BigUint::one(),
                BigUint::one(),
                Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
            )
        } else {
            let x1 = BigUint::from_u32(1).unwrap();
            let y1 = BigUint::from_str(
                "29896722852569046015560700294576055776214335159245303116488692907525646231534",
            )
            .unwrap();
            let x2 = BigUint::from_u32(2).unwrap();
            let y2 = BigUint::from_str(
                "69211104694897500952317515077652022726490027694212560352756646854116994689233",
            )
            .unwrap();

            if rng.gen_bool(0.5) {
                (x1, y1, x2, y2, Rv32WeierstrassOpcode::EC_ADD_NE as usize)
            } else {
                (x2, y2, x1, y1, Rv32WeierstrassOpcode::EC_ADD_NE as usize)
            }
        };

        let ptr_as = 1;
        let data_as = 2;

        let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rs2_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        let p1_base_addr = 0u32;
        let p2_base_addr = 256u32;
        let result_base_addr = 512u32;

        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs1_ptr,
            p1_base_addr.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs2_ptr,
            p2_base_addr.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            rd_ptr,
            result_base_addr.to_le_bytes().map(F::from_canonical_u8),
        );

        let x1_limbs = biguint_to_limbs::<NUM_LIMBS>(x1, LIMB_BITS).map(F::from_canonical_u32);
        let y1_limbs = biguint_to_limbs::<NUM_LIMBS>(y1, LIMB_BITS).map(F::from_canonical_u32);
        let x2_limbs = biguint_to_limbs::<NUM_LIMBS>(x2, LIMB_BITS).map(F::from_canonical_u32);
        let y2_limbs = biguint_to_limbs::<NUM_LIMBS>(y2, LIMB_BITS).map(F::from_canonical_u32);

        tester.write(data_as, p1_base_addr as usize, x1_limbs);
        tester.write(
            data_as,
            (p1_base_addr + NUM_LIMBS as u32) as usize,
            y1_limbs,
        );
        tester.write(data_as, p2_base_addr as usize, x2_limbs);
        tester.write(
            data_as,
            (p2_base_addr + NUM_LIMBS as u32) as usize,
            y2_limbs,
        );

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op_local),
            rd_ptr as isize,
            rs1_ptr as isize,
            rs2_ptr as isize,
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

        let mut harness = create_test_harness::<BLOCKS, BLOCK_SIZE>(&tester, config, offset);

        for i in 0..num_ops {
            set_and_execute_ec_add::<BLOCKS, BLOCK_SIZE, NUM_LIMBS>(
                &mut tester,
                &mut harness,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }

        harness
            .dense_arena
            .get_record_seeker::<EccRecord<2, BLOCKS, BLOCK_SIZE>, _>()
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
    fn test_weierstrass_addne_gpu() {
        run_test_with_config::<2, 32, 32>(secp256k1_coord_prime(), 50);
    }
}
