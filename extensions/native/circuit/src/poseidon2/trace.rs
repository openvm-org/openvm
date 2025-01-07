use std::{borrow::BorrowMut, iter::repeat, sync::Arc};

use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap},
    Chip, ChipUsageGetter,
};

use super::{
    NativePoseidon2BaseChip, NativePoseidon2Cols, NativePoseidon2MemoryCols, NATIVE_POSEIDON2_WIDTH,
};

impl<SC: StarkGenericConfig, const SBOX_REGISTERS: usize> Chip<SC>
    for NativePoseidon2BaseChip<Val<SC>, SBOX_REGISTERS>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        self.air.clone()
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = self.air();
        let height = self.current_trace_height().next_power_of_two();
        let width = self.trace_width();
        let mut records = self.records;
        records.extend(repeat(None).take(height - records.len()));

        let inputs = records
            .par_iter()
            .map(|record| match record {
                Some(record) => record.input,
                None => [Val::<SC>::ZERO; NATIVE_POSEIDON2_WIDTH],
            })
            .collect();
        let inner_trace = self.subchip.generate_trace(inputs);
        let inner_width = self.air.subair.width();

        let memory = self.offline_memory.lock().unwrap();
        let memory_cols = records.par_iter().map(|record| match record {
            Some(record) => record.to_memory_cols(&memory),
            None => NativePoseidon2MemoryCols::blank(),
        });

        let mut values = Val::<SC>::zero_vec(height * width);
        values
            .par_chunks_mut(width)
            .zip(inner_trace.values.par_chunks(inner_width))
            .zip(memory_cols)
            .for_each(|((row, inner_row), memory_cols)| {
                // WARNING: Poseidon2SubCols must be the first field in NativePoseidon2Cols
                row[..inner_width].copy_from_slice(inner_row);
                let cols: &mut NativePoseidon2Cols<Val<SC>, SBOX_REGISTERS> = row.borrow_mut();
                cols.memory = memory_cols;
            });

        AirProofInput::simple_no_pis(air, RowMajorMatrix::new(values, width))
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> ChipUsageGetter
    for NativePoseidon2BaseChip<F, SBOX_REGISTERS>
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        self.records.len()
    }

    fn trace_width(&self) -> usize {
        self.air.width()
    }
}
