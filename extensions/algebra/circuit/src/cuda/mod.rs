//! GPU tracegen chips for the algebra extension (see AGENTS.md: CUDA prover code
//! lives under `cuda/`). `ModularIsEqualChipGpu` fills the IsEqualMod core and
//! Rv32IsEqualModAdapter columns with the kernel in `cuda/src/modular_is_eq.cu`.

use std::sync::Arc;

use openvm_circuit::arch::DenseRecordArena;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_rv32_adapters::{Rv32IsEqualModAdapterCols, Rv32IsEqualModAdapterRecord};
use openvm_stark_backend::prover::AirProvingContext;

use crate::modular_chip::{ModularIsEqualCoreCols, ModularIsEqualRecord};

pub mod cuda_abi;

pub struct ModularIsEqualChipGpu<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
    d_modulus: openvm_cuda_common::d_buffer::DeviceBuffer<u8>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
}

impl<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>
    ModularIsEqualChipGpu<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    pub fn new(
        modulus_limbs: [u8; TOTAL_LIMBS],
        byte_ptr_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<8>>,
    ) -> Self {
        use openvm_cuda_common::copy::MemCopyH2D;
        let d_modulus = modulus_limbs
            .as_slice()
            .to_device_on(&range_checker.device_ctx)
            .unwrap();
        Self {
            range_checker,
            bitwise_lookup,
            d_modulus,
            pointer_max_bits: byte_ptr_max_bits as u32,
            timestamp_max_bits: timestamp_max_bits as u32,
        }
    }
}

impl<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>
    Chip<DenseRecordArena, GpuBackend>
    for ModularIsEqualChipGpu<NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        use openvm_cuda_common::copy::MemCopyH2D;
        let record_stride = size_of::<(
            Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualRecord<TOTAL_LIMBS>,
        )>();
        let rec_core_offset =
            size_of::<Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_stride, 0);
        let rows_used = records.len() / record_stride;
        let height = rows_used.next_power_of_two();
        let width = Rv32IsEqualModAdapterCols::<F, 2, NUM_LANES, LANE_SIZE>::width()
            + ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width();

        let device_ctx = &self.range_checker.device_ctx;
        let d_records = records.to_device_on(device_ctx).unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(height, width, device_ctx);
        unsafe {
            cuda_abi::tracegen(
                d_trace.buffer(),
                height,
                rows_used,
                &d_records,
                record_stride,
                rec_core_offset,
                &self.d_modulus,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                NUM_LANES,
                self.pointer_max_bits,
                self.timestamp_max_bits,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}
