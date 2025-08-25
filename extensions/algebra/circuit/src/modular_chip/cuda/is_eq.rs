use std::sync::Arc;

use derive_new::new;
use num_bigint::BigUint;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs, bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_rv32_adapters::{Rv32IsEqualModAdapterCols, Rv32IsEqualModAdapterRecord};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};

use super::cuda_abi::is_eq_cuda::tracegen as modular_is_equal_tracegen;
use crate::modular_chip::{ModularIsEqualCoreCols, ModularIsEqualRecord};

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
            + ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width();
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
            modular_is_equal_tracegen(
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
