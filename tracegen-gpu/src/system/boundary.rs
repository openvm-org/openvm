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
    base::DeviceMatrix, cuda::copy::MemCopyH2D, prelude::F, prover_backend::GpuBackend,
};

use crate::{
    primitives::var_range::VariableRangeCheckerChipGPU,
    system::{
        cuda::boundary::{persistent_boundary_tracegen, volatile_boundary_tracegen},
        poseidon2::SharedBuffer,
    },
    testing::get_empty_air_proving_ctx,
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
    pub records: Option<Vec<u32>>,
    pub num_records: Option<usize>,
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
            num_records: None,
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
            num_records: None,
            trace_width: None,
        }
    }

    // Records in the buffer are series of u32s. A single record consts
    // of [as, ptr, timestamp, values[0], ..., values[CHUNK - 1]].
    pub fn finalize_records<const CHUNK: usize>(
        &mut self,
        final_memory: TimestampedEquipartition<F, CHUNK>,
    ) {
        self.num_records = Some(final_memory.len());
        self.trace_width = Some(match &self.fields {
            BoundaryFields::Volatile(_) => VolatileBoundaryCols::<F>::width(),
            BoundaryFields::Persistent(_) => PersistentBoundaryCols::<F, CHUNK>::width(),
        });
        let records: Vec<_> = final_memory
            .par_iter()
            .flat_map(|&((addr_space, ptr), ts_values)| {
                let TimestampedValues { timestamp, values } = ts_values;
                let mut record = vec![addr_space, ptr, timestamp];
                record.extend_from_slice(&values.map(|x| x.as_canonical_u32()));
                record
            })
            .collect();
        self.records = Some(records);
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
        let num_records = self.num_records.unwrap();
        if num_records == 0 {
            return get_empty_air_proving_ctx();
        }
        let records = records.to_device().unwrap();
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity(trace_height, self.trace_width());
        match &self.fields {
            BoundaryFields::Persistent(boundary) => unsafe {
                persistent_boundary_tracegen(
                    trace.buffer(),
                    trace.height(),
                    trace.width(),
                    &records,
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
                    &records,
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

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use openvm_circuit::{
        arch::{testing::MEMORY_BUS, MemoryConfig, ADDR_SPACE_OFFSET},
        system::memory::{
            offline_checker::MemoryBus, volatile::VolatileBoundaryChip, TimestampedEquipartition,
            TimestampedValues,
        },
    };
    use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
    use openvm_stark_backend::{
        p3_util::log2_ceil_usize,
        prover::{cpu::CpuBackend, types::AirProvingContext},
        Chip,
    };
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_field::FieldAlgebra;
    use rand::Rng;
    use stark_backend_gpu::{
        prover_backend::GpuBackend,
        types::{F, SC},
    };

    use crate::{
        primitives::var_range::VariableRangeCheckerChipGPU,
        system::boundary::BoundaryChipGPU,
        testing::{assert_eq_cpu_and_gpu_matrix, default_var_range_checker_bus},
    };

    const MAX_ADDRESS_SPACE: u32 = 4;
    const LIMB_BITS: usize = 15;

    #[test]
    fn test_volatile_boundary_tracegen() {
        const NUM_ADDRESSES: usize = 10;
        let mut rng = create_seeded_rng();

        let mut distinct_addresses = HashSet::new();
        while distinct_addresses.len() < NUM_ADDRESSES {
            let addr_space = rng.gen_range(0..MAX_ADDRESS_SPACE);
            let pointer = rng.gen_range(0..(1 << LIMB_BITS));
            distinct_addresses.insert((addr_space, pointer));
        }

        let mut final_memory = TimestampedEquipartition::<F, 1>::new();
        for (addr_space, pointer) in distinct_addresses.iter().cloned() {
            let final_data = F::from_canonical_u32(rng.gen_range(0..(1 << LIMB_BITS)));
            let final_clk = rng.gen_range(1..(1 << LIMB_BITS)) as u32;

            final_memory.push((
                (addr_space, pointer),
                TimestampedValues {
                    values: [final_data],
                    timestamp: final_clk,
                },
            ));
        }
        final_memory.sort_by_key(|(k, _)| *k);

        let mem_config = MemoryConfig::default();
        let addr_space_max_bits = log2_ceil_usize(
            (ADDR_SPACE_OFFSET + 2u32.pow(mem_config.addr_space_height as u32)) as usize,
        );
        let cpu_rc = Arc::new(VariableRangeCheckerChip::new(
            default_var_range_checker_bus(),
        ));

        let mut gpu_boundary = BoundaryChipGPU::volatile(
            Arc::new(VariableRangeCheckerChipGPU::hybrid(cpu_rc.clone())),
            addr_space_max_bits,
            mem_config.pointer_max_bits,
        );
        let mut cpu_boundary: VolatileBoundaryChip<F> = VolatileBoundaryChip::new(
            MemoryBus::new(MEMORY_BUS),
            addr_space_max_bits,
            mem_config.pointer_max_bits,
            cpu_rc,
        );
        gpu_boundary.finalize_records(final_memory.clone());
        cpu_boundary.finalize(final_memory);
        let gpu_ctx: AirProvingContext<GpuBackend> = gpu_boundary.generate_proving_ctx(());
        let cpu_ctx: AirProvingContext<CpuBackend<SC>> = cpu_boundary.generate_proving_ctx(());
        assert_eq_cpu_and_gpu_matrix(cpu_ctx.common_main.unwrap(), &gpu_ctx.common_main.unwrap());
    }
}
