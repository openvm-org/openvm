use std::{
    ffi::c_void,
    mem::size_of,
    sync::{Arc, Mutex},
};

use openvm_circuit::arch::DenseRecordArena;
use openvm_circuit_primitives::Chip;
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    cuda_abi::poseidon2::{self, DeferralPoseidon2Count},
    poseidon2::DeferralPoseidon2Cols,
};

pub struct DeferralPoseidon2ProducerBuffer {
    pub records: DeviceBuffer<F>,
    pub counts: DeviceBuffer<DeferralPoseidon2Count>,
    pub idx: DeviceBuffer<u32>,
    pub expected_records: usize,
}

impl DeferralPoseidon2ProducerBuffer {
    pub fn new(expected_records: usize, device_ctx: &GpuDeviceCtx) -> Self {
        assert!(expected_records > 0);
        let idx = DeviceBuffer::<u32>::with_capacity_on(1, device_ctx);
        idx.fill_zero_on(device_ctx).unwrap();
        Self {
            records: DeviceBuffer::<F>::with_capacity_on(
                expected_records * DIGEST_SIZE * 2,
                device_ctx,
            ),
            counts: DeviceBuffer::<DeferralPoseidon2Count>::with_capacity_on(
                expected_records,
                device_ctx,
            ),
            idx,
            expected_records,
        }
    }
}

#[derive(Clone, Default)]
pub struct DeferralPoseidon2SharedBuffer {
    producers: Arc<Mutex<Vec<DeferralPoseidon2ProducerBuffer>>>,
}

impl DeferralPoseidon2SharedBuffer {
    pub fn push(&self, buffer: DeferralPoseidon2ProducerBuffer) {
        self.producers.lock().unwrap().push(buffer);
    }
}

pub struct DeferralPoseidon2ChipGpu {
    pub device_ctx: GpuDeviceCtx,
    pub shared: DeferralPoseidon2SharedBuffer,
    pub sbox_registers: usize,
}

impl DeferralPoseidon2ChipGpu {
    pub fn new(sbox_registers: usize, device_ctx: GpuDeviceCtx) -> Self {
        Self {
            device_ctx,
            shared: DeferralPoseidon2SharedBuffer::default(),
            sbox_registers,
        }
    }

    pub fn shared_buffer(&self) -> DeferralPoseidon2SharedBuffer {
        self.shared.clone()
    }

    pub fn trace_width() -> usize {
        DeferralPoseidon2Cols::<F>::width()
    }

    fn generate_proving_ctx_checked(
        &self,
        max_trace_height: usize,
    ) -> Result<AirProvingContext<GpuBackend>, String> {
        let mut producer_guard = self.shared.producers.lock().unwrap();
        let mut num_records = 0usize;
        for producer in producer_guard.iter() {
            let actual = producer
                .idx
                .to_host_on(&self.device_ctx)
                .map_err(|error| error.to_string())?[0] as usize;
            if actual != producer.expected_records {
                return Err(format!(
                    "Deferral Poseidon2 producer emitted {actual} records, expected {}",
                    producer.expected_records
                ));
            }
            num_records = num_records
                .checked_add(actual)
                .ok_or_else(|| "Deferral Poseidon2 producer record count overflow".to_string())?;
        }
        if num_records == 0 {
            return Ok(AirProvingContext::simple_no_pis(DeviceMatrix::dummy()));
        }

        // Bound every potentially large allocation before consuming producer
        // ownership. The deduplicated trace cannot exceed this raw upper bound.
        let upper_trace_height = num_records
            .checked_next_power_of_two()
            .ok_or_else(|| "Deferral Poseidon2 padded trace height overflow".to_string())?;
        if upper_trace_height > max_trace_height {
            return Err(format!(
                "Deferral Poseidon2 padded trace height {upper_trace_height} exceeds segment limit {max_trace_height}"
            ));
        }
        num_records
            .checked_mul(DIGEST_SIZE * 2)
            .and_then(|elements| elements.checked_mul(size_of::<F>()))
            .ok_or_else(|| "Deferral Poseidon2 record allocation overflow".to_string())?;
        num_records
            .checked_mul(size_of::<DeferralPoseidon2Count>())
            .ok_or_else(|| "Deferral Poseidon2 count allocation overflow".to_string())?;
        upper_trace_height
            .checked_mul(Self::trace_width())
            .and_then(|elements| elements.checked_mul(size_of::<F>()))
            .ok_or_else(|| "Deferral Poseidon2 trace allocation overflow".to_string())?;

        let mut producers = std::mem::take(&mut *producer_guard);
        drop(producer_guard);
        let (records, counts) = if producers.len() == 1 {
            let producer = producers.pop().unwrap();
            (producer.records, producer.counts)
        } else {
            let records = DeviceBuffer::<F>::with_capacity_on(
                num_records * DIGEST_SIZE * 2,
                &self.device_ctx,
            );
            let counts = DeviceBuffer::<DeferralPoseidon2Count>::with_capacity_on(
                num_records,
                &self.device_ctx,
            );
            let mut offset = 0usize;
            for producer in &producers {
                let record_elements = producer.expected_records * DIGEST_SIZE * 2;
                unsafe {
                    cuda_memcpy_on::<true, true>(
                        records.as_mut_ptr().add(offset * DIGEST_SIZE * 2) as *mut c_void,
                        producer.records.as_ptr() as *mut c_void,
                        record_elements * size_of::<F>(),
                        &self.device_ctx,
                    )
                    .map_err(|error| error.to_string())?;
                    cuda_memcpy_on::<true, true>(
                        counts.as_mut_ptr().add(offset) as *mut c_void,
                        producer.counts.as_ptr() as *mut c_void,
                        producer.expected_records * size_of::<DeferralPoseidon2Count>(),
                        &self.device_ctx,
                    )
                    .map_err(|error| error.to_string())?;
                }
                offset += producer.expected_records;
            }
            self.device_ctx
                .stream
                .synchronize()
                .map_err(|error| error.to_string())?;
            drop(producers);
            (records, counts)
        };

        let dedup_records =
            DeviceBuffer::<F>::with_capacity_on(num_records * DIGEST_SIZE * 2, &self.device_ctx);
        let dedup_counts =
            DeviceBuffer::<DeferralPoseidon2Count>::with_capacity_on(num_records, &self.device_ctx);
        unsafe {
            let d_num_records = [num_records]
                .to_device_on(&self.device_ctx)
                .map_err(|error| error.to_string())?;
            let mut temp_bytes = 0;
            poseidon2::deduplicate_records_get_temp_bytes(
                &records,
                &counts,
                num_records,
                &d_num_records,
                &mut temp_bytes,
                self.device_ctx.stream.as_raw(),
            )
            .map_err(|error| error.to_string())?;

            let d_temp_storage = if temp_bytes == 0 {
                DeviceBuffer::<u8>::new()
            } else {
                DeviceBuffer::<u8>::with_capacity_on(temp_bytes, &self.device_ctx)
            };

            poseidon2::deduplicate_records(
                &records,
                &counts,
                &dedup_records,
                &dedup_counts,
                num_records,
                &d_num_records,
                &d_temp_storage,
                temp_bytes,
                self.device_ctx.stream.as_raw(),
            )
            .map_err(|error| error.to_string())?;

            num_records = *d_num_records
                .to_host_on(&self.device_ctx)
                .map_err(|error| error.to_string())?
                .first()
                .ok_or_else(|| "Deferral Poseidon2 dedup count is missing".to_string())?;
        }
        // The D2H dedup count fences the sort/reduce stream. Release the raw
        // combined producers before allocating the final trace; only compact
        // deduplicated inputs remain live through tracegen.
        drop(records);
        drop(counts);

        let trace_height = num_records
            .checked_next_power_of_two()
            .ok_or_else(|| "Deferral Poseidon2 deduplicated trace height overflow".to_string())?;
        if trace_height > max_trace_height {
            return Err(format!(
                "Deferral Poseidon2 deduplicated trace height {trace_height} exceeds segment limit {max_trace_height}"
            ));
        }
        let trace = DeviceMatrix::<F>::with_capacity_on(
            trace_height,
            Self::trace_width(),
            &self.device_ctx,
        );

        unsafe {
            poseidon2::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &dedup_records,
                &dedup_counts,
                num_records,
                self.sbox_registers,
                self.device_ctx.stream.as_raw(),
            )
            .map_err(|error| error.to_string())?;
        }

        Ok(AirProvingContext::simple_no_pis(trace))
    }

    #[cfg(feature = "rvr")]
    pub fn generate_proving_ctx_direct(
        &self,
        max_trace_height: usize,
    ) -> Result<AirProvingContext<GpuBackend>, openvm_circuit::arch::rvr::cuda::GpuRvrInputError>
    {
        self.generate_proving_ctx_checked(max_trace_height)
            .map_err(openvm_circuit::arch::rvr::cuda::GpuRvrInputError::InvalidTranscript)
    }
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralPoseidon2ChipGpu {
    fn generate_proving_ctx(&self, _: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        self.generate_proving_ctx_checked(usize::MAX)
            .expect("Failed to generate deferral poseidon2 trace")
    }
}
