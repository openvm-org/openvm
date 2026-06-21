use std::sync::Arc;

use derive_new::new;
use openvm_circuit::arch::DenseRecordArena;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, GpuBackend};
use openvm_stark_backend::prover::AirProvingContext;

use crate::adapters::RV64_BYTE_BITS;

fn unsupported_split_loadstore_ctx(arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
    if arena.allocated().is_empty() {
        return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
    }
    unimplemented!("CUDA trace generation for split RV64 loadstore chips is not implemented")
}

#[derive(new)]
pub struct Rv64LoadStoreByteChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV64_BYTE_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64LoadStoreByteChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        unsupported_split_loadstore_ctx(arena)
    }
}

#[derive(new)]
pub struct Rv64LoadStoreHalfwordChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64LoadStoreHalfwordChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        unsupported_split_loadstore_ctx(arena)
    }
}

#[derive(new)]
pub struct Rv64LoadStoreWordChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64LoadStoreWordChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        unsupported_split_loadstore_ctx(arena)
    }
}

#[derive(new)]
pub struct Rv64LoadStoreDoublewordChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64LoadStoreDoublewordChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        unsupported_split_loadstore_ctx(arena)
    }
}
