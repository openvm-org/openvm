use openvm_circuit::{
    arch::BLOCK_FE_WIDTH, system::memory::persistent::PersistentBoundaryCols,
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};

use super::poseidon2::SharedBuffer;
use crate::cuda_abi::boundary::persistent_boundary_tracegen;

pub struct BoundaryChipGPU {
    pub device_ctx: GpuDeviceCtx,
    pub poseidon2_buffer: SharedBuffer<F>,
    /// A `Vec` of pointers to the copied guest memory on device.
    /// This struct cannot own the device memory, hence we take extra care not to use memory we
    /// don't own. TODO: use `Arc<DeviceBuffer>` instead?
    pub initial_leaves: Vec<*const std::ffi::c_void>,
    pub records: Option<DeviceBuffer<u32>>,
    pub num_records: Option<usize>,
    pub trace_width: Option<usize>,
}

const BLOCKS_PER_LEAF: usize = VM_DIGEST_WIDTH / BLOCK_FE_WIDTH;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PersistentBoundaryRecord {
    pub address_space: u32,
    /// AS-native pointer to the first cell of this Merkle leaf.
    pub ptr: u32,
    /// Whether some block of the leaf was *written* during execution (0/1); see
    /// `TouchedBlock`. Not consumed yet.
    pub is_dirty: u32,
    pub timestamps: [u32; BLOCKS_PER_LEAF],
    pub values: [F; VM_DIGEST_WIDTH],
}

impl BoundaryChipGPU {
    pub fn new(poseidon2_buffer: SharedBuffer<F>, device_ctx: GpuDeviceCtx) -> Self {
        Self {
            device_ctx,
            poseidon2_buffer,
            initial_leaves: Vec::new(),
            records: None,
            num_records: None,
            trace_width: None,
        }
    }

    pub fn finalize_records<const DIGEST_WIDTH: usize>(
        &mut self,
        records: Vec<PersistentBoundaryRecord>,
    ) {
        self.num_records = Some(records.len());
        self.trace_width = Some(PersistentBoundaryCols::<F, DIGEST_WIDTH>::width());
        self.records = Some(if records.is_empty() {
            DeviceBuffer::new()
        } else {
            records
                .to_device_on(&self.device_ctx)
                .unwrap()
                .as_buffer::<u32>()
        });
    }

    pub fn finalize_records_device<const DIGEST_WIDTH: usize>(
        &mut self,
        records: DeviceBuffer<u32>,
        num_records: usize,
    ) {
        self.num_records = Some(num_records);
        self.trace_width = Some(PersistentBoundaryCols::<F, DIGEST_WIDTH>::width());
        self.records = Some(records);
    }

    pub fn trace_width(&self) -> usize {
        self.trace_width.expect("Finalize records to get width")
    }

    pub fn records(&self) -> &DeviceBuffer<u32> {
        self.records
            .as_ref()
            .expect("Finalize records to get buffer")
    }
}

impl<RA> Chip<RA, GpuBackend> for BoundaryChipGPU {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let num_records = self.num_records.unwrap();
        if num_records == 0 {
            // Boundary AIR should always be present, so return a single zero-filled
            // padding row.
            let trace =
                DeviceMatrix::<F>::with_capacity_on(1, self.trace_width(), &self.device_ctx);
            trace.buffer().fill_zero_on(&self.device_ctx).unwrap();
            return AirProvingContext::simple_no_pis(trace);
        }
        let unpadded_height = num_records;
        let trace_height = next_power_of_two_or_zero(unpadded_height);
        let trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, self.trace_width(), &self.device_ctx);
        trace.buffer().fill_zero_on(&self.device_ctx).unwrap();
        let mem_ptrs = self.initial_leaves.to_device_on(&self.device_ctx).unwrap();
        let poseidon2_records = self.poseidon2_buffer.records();
        unsafe {
            persistent_boundary_tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &mem_ptrs,
                self.records.as_ref().unwrap(),
                num_records,
                &poseidon2_records,
                &self.poseidon2_buffer.idx,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate boundary trace");
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
