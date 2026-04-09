use std::sync::Arc;

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    cuda_abi::poseidon2::{self, DeferralPoseidon2Count},
    poseidon2::DeferralPoseidon2Cols,
};

#[derive(Clone)]
pub struct DeferralPoseidon2SharedBuffer {
    pub records: Arc<DeviceBuffer<F>>,
    pub counts: Arc<DeviceBuffer<DeferralPoseidon2Count>>,
    pub idx: Arc<DeviceBuffer<u32>>,
}

pub struct DeferralPoseidon2ChipGpu {
    pub ctx: GpuDeviceCtx,
    pub records: Arc<DeviceBuffer<F>>,
    pub counts: Arc<DeviceBuffer<DeferralPoseidon2Count>>,
    pub idx: Arc<DeviceBuffer<u32>>,
    pub sbox_registers: usize,
}

impl DeferralPoseidon2ChipGpu {
    /// Creates a new deferral Poseidon2 chip configured for `max_trace_height` records. Each
    /// Poseidon2 record occupies `POSEIDON2_WIDTH` (16) field elements, and a buffer of that
    /// size is allocated.
    pub fn new(max_trace_height: usize, sbox_registers: usize, ctx: GpuDeviceCtx) -> Self {
        let max_num_records = max_trace_height.next_power_of_two();
        let max_record_buf_size = max_num_records * (DIGEST_SIZE * 2);

        let idx = Arc::new(DeviceBuffer::<u32>::with_capacity_on(1, &ctx));
        idx.fill_zero_on(&ctx).unwrap();

        Self {
            ctx: ctx.clone(),
            records: Arc::new(DeviceBuffer::<F>::with_capacity_on(
                max_record_buf_size,
                &ctx,
            )),
            counts: Arc::new(DeviceBuffer::<DeferralPoseidon2Count>::with_capacity_on(
                max_num_records,
                &ctx,
            )),
            idx,
            sbox_registers,
        }
    }

    pub fn shared_buffer(&self) -> DeferralPoseidon2SharedBuffer {
        DeferralPoseidon2SharedBuffer {
            records: self.records.clone(),
            counts: self.counts.clone(),
            idx: self.idx.clone(),
        }
    }

    pub fn trace_width() -> usize {
        DeferralPoseidon2Cols::<F>::width()
    }
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralPoseidon2ChipGpu {
    fn generate_proving_ctx(&self, _: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let mut num_records = self.idx.to_host_on(&self.ctx).unwrap()[0] as usize;
        if num_records == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let dedup_records =
            DeviceBuffer::<F>::with_capacity_on(num_records * DIGEST_SIZE * 2, &self.ctx);
        let dedup_counts =
            DeviceBuffer::<DeferralPoseidon2Count>::with_capacity_on(num_records, &self.ctx);
        unsafe {
            let d_num_records = [num_records].to_device_on(&self.ctx).unwrap();
            let mut temp_bytes = 0;
            poseidon2::deduplicate_records_get_temp_bytes(
                &self.records,
                &self.counts,
                num_records,
                &d_num_records,
                &mut temp_bytes,
                self.ctx.stream.as_raw(),
            )
            .expect("Failed to get deferral poseidon2 temp bytes");

            let d_temp_storage = if temp_bytes == 0 {
                DeviceBuffer::<u8>::new()
            } else {
                DeviceBuffer::<u8>::with_capacity_on(temp_bytes, &self.ctx)
            };

            poseidon2::deduplicate_records(
                &self.records,
                &self.counts,
                &dedup_records,
                &dedup_counts,
                num_records,
                &d_num_records,
                &d_temp_storage,
                temp_bytes,
                self.ctx.stream.as_raw(),
            )
            .expect("Failed to deduplicate deferral poseidon2 records");

            num_records = *d_num_records
                .to_host_on(&self.ctx)
                .unwrap()
                .first()
                .unwrap();
        }

        let trace_height = next_power_of_two_or_zero(num_records);
        let trace =
            DeviceMatrix::<F>::with_capacity_on(trace_height, Self::trace_width(), &self.ctx);

        unsafe {
            poseidon2::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &dedup_records,
                &dedup_counts,
                num_records,
                self.sbox_registers,
                self.ctx.stream.as_raw(),
            )
            .expect("Failed to generate deferral poseidon2 trace");
        }

        self.idx
            .fill_zero_on(&self.ctx)
            .expect("Failed to reset deferral poseidon2 record index");

        AirProvingContext::simple_no_pis(trace)
    }
}
