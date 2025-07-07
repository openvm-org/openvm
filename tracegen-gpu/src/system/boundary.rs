use std::sync::Arc;

use openvm_circuit::{
    system::memory::{
        offline_checker::MemoryBus, persistent::PersistentBoundaryAir,
        volatile::VolatileBoundaryAir, TimestampedEquipartition, TimestampedValues,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeField32,
    p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator},
    prover::hal::MatrixDimensions,
    rap::get_air_name,
    AirRef, ChipUsageGetter,
};
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
    prover_backend::GpuBackend,
    types::SC,
};

use crate::{
    primitives::var_range::VariableRangeCheckerChipGPU,
    system::{
        cuda::boundary::{persistent_boundary_tracegen, volatile_boundary_tracegen},
        poseidon2::SharedBuffer,
    },
    DeviceChip,
};

pub struct PersistentBoundary<const CHUNK: usize> {
    pub air: Arc<PersistentBoundaryAir<CHUNK>>,
    pub poseidon2_buffer: SharedBuffer<F>,
    pub sbox_regs: usize,
}

pub struct VolatileBoundary {
    pub air: Arc<VolatileBoundaryAir>,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub as_max_bits: usize,
    pub ptr_max_bits: usize,
}

pub enum BoundaryFields<const CHUNK: usize> {
    Persistent(PersistentBoundary<CHUNK>),
    Volatile(VolatileBoundary),
}

pub struct BoundaryChipGPU<const CHUNK: usize> {
    pub fields: BoundaryFields<CHUNK>,
    pub records: Option<Arc<DeviceBuffer<u32>>>,
}

impl<const CHUNK: usize> BoundaryChipGPU<CHUNK> {
    pub fn persistent(
        air: PersistentBoundaryAir<CHUNK>,
        poseidon2_buffer: SharedBuffer<F>,
        sbox_regs: usize,
    ) -> Self {
        Self {
            fields: BoundaryFields::Persistent(PersistentBoundary {
                air: Arc::new(air),
                poseidon2_buffer,
                sbox_regs,
            }),
            records: None,
        }
    }

    pub fn volatile(
        memory_bus: MemoryBus,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        as_max_bits: usize,
        ptr_max_bits: usize,
    ) -> Self {
        let air = Arc::new(VolatileBoundaryAir::new(
            memory_bus,
            as_max_bits,
            ptr_max_bits,
            range_checker.air.bus,
        ));
        Self {
            fields: BoundaryFields::Volatile(VolatileBoundary {
                air,
                range_checker,
                as_max_bits,
                ptr_max_bits,
            }),
            records: None,
        }
    }

    // Records in the buffer are series of u32s. A single record consts
    // of [as, ptr, timestamp, values[0], ..., values[CHUNK - 1]].
    pub fn finalize_records(&mut self, final_memory: TimestampedEquipartition<F, CHUNK>) {
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
    }
}

impl<const CHUNK: usize> ChipUsageGetter for BoundaryChipGPU<CHUNK> {
    fn air_name(&self) -> String {
        match &self.fields {
            BoundaryFields::Volatile(boundary) => get_air_name(&boundary.air),
            BoundaryFields::Persistent(boundary) => get_air_name(&boundary.air),
        }
    }

    fn current_trace_height(&self) -> usize {
        if let Some(records) = &self.records {
            records.len()
        } else {
            0
        }
    }

    fn trace_width(&self) -> usize {
        match &self.fields {
            BoundaryFields::Volatile(boundary) => {
                <VolatileBoundaryAir as BaseAir<F>>::width(&boundary.air)
            }
            BoundaryFields::Persistent(boundary) => {
                <PersistentBoundaryAir<CHUNK> as BaseAir<F>>::width(&boundary.air)
            }
        }
    }
}

impl<const CHUNK: usize> DeviceChip<SC, GpuBackend> for BoundaryChipGPU<CHUNK> {
    fn air(&self) -> AirRef<SC> {
        match &self.fields {
            BoundaryFields::Volatile(boundary) => boundary.air.clone(),
            BoundaryFields::Persistent(boundary) => boundary.air.clone(),
        }
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
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
        trace
    }
}
