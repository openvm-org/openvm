//! GPU tracegen chips for the algebra extension (see AGENTS.md: CUDA prover code
//! lives under `cuda/`). `ModularIsEqualChipGpu` fills the IsEqualMod core and
//! Rv64IsEqualModU16Adapter columns with the kernel in `cuda/src/modular_is_eq.cu`.

use std::sync::Arc;

use openvm_circuit::arch::DenseRecordArena;
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_riscv_adapters::{Rv64IsEqualModU16AdapterCols, Rv64IsEqualModU16AdapterRecord};
use openvm_stark_backend::prover::AirProvingContext;

use crate::modular_chip::{ModularIsEqualCoreCols, ModularIsEqualRecord};

pub mod cuda_abi;

pub struct ModularIsEqualChipGpu<const NUM_LANES: usize, const TOTAL_LIMBS: usize> {
    range_checker: Arc<VariableRangeCheckerChipGPU>,
    d_modulus: openvm_cuda_common::d_buffer::DeviceBuffer<u16>,
    pointer_max_bits: u32,
    timestamp_max_bits: u32,
}

impl<const NUM_LANES: usize, const TOTAL_LIMBS: usize>
    ModularIsEqualChipGpu<NUM_LANES, TOTAL_LIMBS>
{
    pub fn new(
        modulus_limbs: [u16; TOTAL_LIMBS],
        byte_ptr_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Self {
        use openvm_cuda_common::copy::MemCopyH2D;
        let d_modulus = modulus_limbs
            .as_slice()
            .to_device_on(&range_checker.device_ctx)
            .unwrap();
        Self {
            range_checker,
            d_modulus,
            pointer_max_bits: byte_ptr_max_bits as u32,
            timestamp_max_bits: timestamp_max_bits as u32,
        }
    }
}

impl<const NUM_LANES: usize, const TOTAL_LIMBS: usize> Chip<DenseRecordArena, GpuBackend>
    for ModularIsEqualChipGpu<NUM_LANES, TOTAL_LIMBS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        use openvm_cuda_common::copy::MemCopyH2D;
        #[cfg(feature = "rvr")]
        let g2_segment_id = arena.rvr_g2_segment_id;
        type ARec<const L: usize> = Rv64IsEqualModU16AdapterRecord<2, L>;
        let record_stride = size_of::<(ARec<NUM_LANES>, ModularIsEqualRecord<TOTAL_LIMBS>)>();
        let rec_core_offset = size_of::<ARec<NUM_LANES>>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_stride, 0);
        let rows_used = records.len() / record_stride;
        let height = rows_used.next_power_of_two();
        let width = Rv64IsEqualModU16AdapterCols::<F, 2, NUM_LANES>::width()
            + ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width();

        let device_ctx = &self.range_checker.device_ctx;
        #[cfg(feature = "rvr")]
        let h2d_timer = g2_segment_id.and_then(|_| {
            openvm_circuit::arch::rvr::gpu_profile::CudaStageTimer::start(device_ctx)
        });
        let d_records = records.to_device_on(device_ctx).unwrap();
        #[cfg(feature = "rvr")]
        if let (Some(timer), Some(segment_id)) = (h2d_timer, g2_segment_id) {
            timer.finish("opaque_h2d", segment_id, records.len());
        }
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
