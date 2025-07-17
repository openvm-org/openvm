use std::sync::Arc;

use openvm_circuit::{
    system::memory::{
        persistent::PersistentBoundaryCols, volatile::VolatileBoundaryCols,
        TimestampedEquipartition, TimestampedValues,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator},
    prover::{hal::MatrixDimensions, types::AirProvingContext},
    Chip,
};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
};

use crate::{
    primitives::var_range::VariableRangeCheckerChipGPU,
    system::{
        cuda::boundary::{persistent_boundary_tracegen, volatile_boundary_tracegen},
        poseidon2::SharedBuffer,
    },
};

pub struct PersistentBoundary {
    pub poseidon2_buffer: SharedBuffer<F>,
    pub sbox_regs: usize,
}

pub struct VolatileBoundary {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub as_max_bits: usize,
    pub ptr_max_bits: usize,
}

pub enum BoundaryFields {
    Persistent(PersistentBoundary),
    Volatile(VolatileBoundary),
}

pub struct BoundaryChipGPU {
    pub fields: BoundaryFields,
    pub records: Option<Arc<DeviceBuffer<u32>>>,
    pub trace_width: Option<usize>,
}

impl BoundaryChipGPU {
    pub fn persistent(poseidon2_buffer: SharedBuffer<F>, sbox_regs: usize) -> Self {
        Self {
            fields: BoundaryFields::Persistent(PersistentBoundary {
                poseidon2_buffer,
                sbox_regs,
            }),
            records: None,
            trace_width: None,
        }
    }

    pub fn volatile(
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        as_max_bits: usize,
        ptr_max_bits: usize,
    ) -> Self {
        Self {
            fields: BoundaryFields::Volatile(VolatileBoundary {
                range_checker,
                as_max_bits,
                ptr_max_bits,
            }),
            records: None,
            trace_width: None,
        }
    }

    // Records in the buffer are series of u32s. A single record consts
    // of [as, ptr, timestamp, values[0], ..., values[CHUNK - 1]].
    pub fn finalize_records<const CHUNK: usize>(
        &mut self,
        final_memory: TimestampedEquipartition<F, CHUNK>,
    ) {
        let records: Vec<_> = final_memory
            .par_iter()
            .flat_map(|&((addr_space, ptr), ts_values)| {
                let TimestampedValues { timestamp, values } = ts_values;
                let mut record = vec![addr_space, ptr, timestamp];
                record.extend_from_slice(&values.map(|x| x.as_canonical_u32()));
                record
            })
            .collect();
        self.records = Some(Arc::new(records.to_device().unwrap()));
        self.trace_width = Some(match &self.fields {
            BoundaryFields::Volatile(_) => VolatileBoundaryCols::<F>::width(),
            BoundaryFields::Persistent(_) => PersistentBoundaryCols::<F, CHUNK>::width(),
        });
    }

    pub fn trace_height(&self) -> usize {
        if let Some(records) = &self.records {
            records.len()
        } else {
            0
        }
    }

    pub fn trace_width(&self) -> usize {
        self.trace_width.expect("Finalize records to get width")
    }
}

impl<RA> Chip<RA, GpuBackend> for BoundaryChipGPU {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let records = self
            .records
            .as_ref()
            .expect("Records must be finalized before generating trace");
        let num_records = records.len();
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        match &self.fields {
            BoundaryFields::Persistent(boundary) => unsafe {
                persistent_boundary_tracegen(
                    trace.buffer(),
                    trace.height(),
                    trace.width(),
                    records,
                    num_records,
                    &boundary.poseidon2_buffer.buffer,
                    &boundary.poseidon2_buffer.idx,
                    boundary.sbox_regs,
                )
                .expect("Failed to generate persistent boundary trace");
            },
            BoundaryFields::Volatile(boundary) => unsafe {
                volatile_boundary_tracegen(
                    trace.buffer(),
                    trace.height(),
                    trace.width(),
                    records,
                    num_records,
                    &boundary.range_checker.count,
                    boundary.as_max_bits,
                    boundary.ptr_max_bits,
                )
                .expect("Failed to generate volatile boundary trace");
            },
        }
        AirProvingContext::simple_no_pis(trace)
    }
}
