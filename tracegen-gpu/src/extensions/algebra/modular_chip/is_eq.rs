use std::{mem::size_of, sync::Arc};

use derive_new::new;
use num_bigint::BigUint;
use openvm_algebra_circuit::modular_chip::ModularIsEqualRecord;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::bigint::utils::big_uint_to_limbs;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_rv32_adapters::{Rv32IsEqualModAdapterCols, Rv32IsEqualModAdapterRecord};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use crate::{
    extensions::algebra::cuda::is_eq_cuda::tracegen as modular_is_eq_tracegen,
    get_empty_air_proving_ctx,
    primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
    },
};

#[derive(new)]
pub struct ModularIsEqualChipGpu<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub modulus: BigUint,
    pub pointer_max_bits: u32,
    pub timestamp_max_bits: u32,
}

impl<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>
    Chip<DenseRecordArena, GpuBackend>
    for ModularIsEqualChipGpu<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const LIMB_BITS: usize = 8;

        let record_size = size_of::<(
            Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualRecord<TOTAL_LIMBS>,
        )>();

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let trace_width = Rv32IsEqualModAdapterCols::<F, 2, NUM_LANES, LANE_SIZE>::width()
            + openvm_algebra_circuit::modular_chip::ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width(
            );
        let trace_height = next_power_of_two_or_zero(records.len() / record_size);

        let modulus_vec = big_uint_to_limbs(&self.modulus, LIMB_BITS);
        assert!(modulus_vec.len() <= TOTAL_LIMBS);
        let mut modulus_limbs = vec![0u8; TOTAL_LIMBS];
        for (i, &limb) in modulus_vec.iter().enumerate() {
            modulus_limbs[i] = limb as u8;
        }

        let d_records = records.to_device().unwrap();
        let d_modulus = modulus_limbs.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            modular_is_eq_tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &d_modulus,
                TOTAL_LIMBS,
                NUM_LANES,
                LANE_SIZE,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                self.pointer_max_bits,
                self.timestamp_max_bits,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use num_traits::Zero;
    use openvm_algebra_circuit::modular_chip::{
        ModularIsEqualAir, ModularIsEqualChip, ModularIsEqualCoreAir, ModularIsEqualFiller,
        VmModularIsEqualExecutor,
    };
    use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
    use openvm_circuit::arch::{testing::memory::gen_pointer, EmptyAdapterCoreLayout};
    use openvm_circuit_primitives::{
        bigint::utils::{big_uint_to_limbs, secp256k1_coord_prime},
        bitwise_op_lookup::BitwiseOperationLookupChip,
    };
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        LocalOpcode, VmOpcode,
    };
    use openvm_mod_circuit_builder::test_utils::biguint_to_limbs;
    use openvm_pairing_guest::bls12_381::BLS12_381_MODULUS;
    use openvm_rv32_adapters::{
        Rv32IsEqualModAdapterAir, Rv32IsEqualModAdapterExecutor, Rv32IsEqualModAdapterFiller,
    };
    use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};

    use super::*;
    use crate::testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness};

    const LIMB_BITS: usize = 8;
    const MAX_INS_CAPACITY: usize = 128;
    const WRITE_LIMBS: usize = RV32_REGISTER_NUM_LIMBS;

    fn create_test_harness<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const TOTAL_LIMBS: usize,
    >(
        tester: &GpuChipTestBuilder,
        modulus: BigUint,
        offset: usize,
    ) -> GpuTestChipHarness<
        F,
        VmModularIsEqualExecutor<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        ModularIsEqualAir<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        ModularIsEqualChipGpu<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        ModularIsEqualChip<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
    > {
        // getting bus from tester since `gpu_chip` and `air` must use the same bus
        let bitwise_bus = default_bitwise_lookup_bus();
        // creating a dummy chip for Cpu so we only count `add_count`s from GPU
        let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
            bitwise_bus,
        ));

        // Convert modulus to limbs for CPU chip
        let modulus_vec = big_uint_to_limbs(&modulus, LIMB_BITS);
        assert!(modulus_vec.len() <= TOTAL_LIMBS);
        let mut modulus_limbs = [0u8; TOTAL_LIMBS];
        for (i, &limb) in modulus_vec.iter().enumerate() {
            modulus_limbs[i] = limb as u8;
        }

        let adapter_air = Rv32IsEqualModAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
            tester.address_bits(),
        );
        let core_air = ModularIsEqualCoreAir::<TOTAL_LIMBS, WRITE_LIMBS, LIMB_BITS>::new(
            modulus.clone(),
            bitwise_bus,
            offset,
        );
        let air = ModularIsEqualAir::new(adapter_air, core_air);

        let adapter_step = Rv32IsEqualModAdapterExecutor::new(tester.address_bits());
        let executor = VmModularIsEqualExecutor::new(adapter_step, offset, modulus_limbs);

        let adapter_filler =
            Rv32IsEqualModAdapterFiller::new(tester.address_bits(), dummy_bitwise_chip.clone());
        let core_filler = ModularIsEqualFiller::<_, TOTAL_LIMBS, WRITE_LIMBS, LIMB_BITS>::new(
            adapter_filler,
            offset,
            modulus_limbs,
            dummy_bitwise_chip,
        );
        let cpu_chip = ModularIsEqualChip::new(core_filler, tester.dummy_memory_helper());

        let gpu_chip = ModularIsEqualChipGpu::new(
            tester.range_checker(),
            tester.bitwise_op_lookup(),
            modulus,
            tester.address_bits() as u32,
            tester.timestamp_max_bits() as u32,
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    fn set_and_execute_is_eq<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const TOTAL_LIMBS: usize,
    >(
        tester: &mut GpuChipTestBuilder,
        harness: &mut GpuTestChipHarness<
            F,
            VmModularIsEqualExecutor<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualAir<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualChipGpu<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualChip<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        >,
        rng: &mut StdRng,
        modulus: &BigUint,
        is_setup: bool,
        offset: usize,
    ) {
        let ptr_as = RV32_REGISTER_AS as usize;
        let mem_as = RV32_MEMORY_AS as usize;

        let rd_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rs1_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        let rs2_ptr = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);

        // Memory addresses for operands
        let b_base_addr = 0u32;
        let c_base_addr = 128u32;

        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs1_ptr,
            b_base_addr.to_le_bytes().map(F::from_canonical_u8),
        );
        tester.write::<RV32_REGISTER_NUM_LIMBS>(
            ptr_as,
            rs2_ptr,
            c_base_addr.to_le_bytes().map(F::from_canonical_u8),
        );

        let (b, c) = if is_setup {
            (modulus.clone(), BigUint::zero())
        } else {
            let b_digits: Vec<_> = (0..TOTAL_LIMBS)
                .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                .collect();
            let mut b = BigUint::new(b_digits);
            b %= modulus;

            let c = if rng.gen_bool(0.5) {
                b.clone()
            } else {
                let c_digits: Vec<_> = (0..TOTAL_LIMBS)
                    .map(|_| rng.gen_range(0..(1 << LIMB_BITS)))
                    .collect();
                let mut c = BigUint::new(c_digits);
                c %= modulus;
                c
            };

            (b, c)
        };

        let b_limbs =
            biguint_to_limbs::<TOTAL_LIMBS>(b.clone(), LIMB_BITS).map(F::from_canonical_u32);
        let c_limbs =
            biguint_to_limbs::<TOTAL_LIMBS>(c.clone(), LIMB_BITS).map(F::from_canonical_u32);

        for i in (0..TOTAL_LIMBS).step_by(RV32_REGISTER_NUM_LIMBS) {
            tester.write::<RV32_REGISTER_NUM_LIMBS>(
                mem_as,
                b_base_addr as usize + i,
                b_limbs[i..i + RV32_REGISTER_NUM_LIMBS].try_into().unwrap(),
            );
            tester.write::<RV32_REGISTER_NUM_LIMBS>(
                mem_as,
                c_base_addr as usize + i,
                c_limbs[i..i + RV32_REGISTER_NUM_LIMBS].try_into().unwrap(),
            );
        }

        let op_local = if is_setup {
            Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize
        } else {
            Rv32ModularArithmeticOpcode::IS_EQ as usize
        };

        let instruction = Instruction::from_isize(
            VmOpcode::from_usize(offset + op_local),
            rd_ptr as isize,
            rs1_ptr as isize,
            rs2_ptr as isize,
            ptr_as as isize,
            mem_as as isize,
        );

        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instruction,
        );
    }

    fn run_test_with_config<
        const NUM_LANES: usize,
        const LANE_SIZE: usize,
        const TOTAL_LIMBS: usize,
    >(
        modulus: BigUint,
        num_ops: usize,
    ) {
        let mut rng = create_seeded_rng();

        let mut tester =
            GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

        let offset = Rv32ModularArithmeticOpcode::CLASS_OFFSET;
        let mut harness = create_test_harness::<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>(
            &tester,
            modulus.clone(),
            offset,
        );

        for i in 0..num_ops {
            set_and_execute_is_eq::<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>(
                &mut tester,
                &mut harness,
                &mut rng,
                &modulus,
                i == 0,
                offset,
            );
        }

        type Record<'a, const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize> = (
            &'a mut Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            &'a mut ModularIsEqualRecord<TOTAL_LIMBS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<
                    F,
                    Rv32IsEqualModAdapterExecutor<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
                >::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[test]
    fn test_modular_is_eq_gpu_1x32() {
        run_test_with_config::<1, 32, 32>(secp256k1_coord_prime(), 50);
    }

    #[test]
    fn test_modular_is_eq_gpu_3x16() {
        run_test_with_config::<3, 16, 48>(BLS12_381_MODULUS.clone(), 50);
    }
}
