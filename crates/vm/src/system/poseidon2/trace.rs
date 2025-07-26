use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    Chip, ChipUsageGetter,
};

use super::{columns::*, Poseidon2PeripheryBaseChip, PERIPHERY_POSEIDON2_WIDTH};

impl<RA, SC: StarkGenericConfig, const SBOX_REGISTERS: usize> Chip<RA, CpuBackend<SC>>
    for Poseidon2PeripheryBaseChip<Val<SC>, SBOX_REGISTERS>
where
    Val<SC>: PrimeField32,
{
    /// Generates trace and clears internal records state.
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let height = next_power_of_two_or_zero(self.current_trace_height());
        let width = self.trace_width();

        let mut inputs = Vec::with_capacity(height);
        let mut multiplicities = Vec::with_capacity(height);
        #[cfg(feature = "parallel")]
        let records_iter = self.records.par_iter();
        #[cfg(not(feature = "parallel"))]
        let records_iter = self.records.iter();
        let (actual_inputs, actual_multiplicities): (Vec<_>, Vec<_>) = records_iter
            .map(|r| {
                let (input, mult) = r.pair();
                (*input, mult.load(std::sync::atomic::Ordering::Relaxed))
            })
            .unzip();
        inputs.extend(actual_inputs);
        multiplicities.extend(actual_multiplicities);
        inputs.resize(height, [Val::<SC>::ZERO; PERIPHERY_POSEIDON2_WIDTH]);
        multiplicities.resize(height, 0);

        // TODO: this would be more optimal if plonky3 made the generate_trace_row function public
        let inner_trace = self.subchip.generate_trace(inputs);
        let inner_width = self.air.subair.width();

        let mut values = Val::<SC>::zero_vec(height * width);
        values
            .par_chunks_mut(width)
            .zip(inner_trace.values.par_chunks(inner_width))
            .zip(multiplicities)
            .for_each(|((row, inner_row), mult)| {
                // WARNING: Poseidon2SubCols must be the first field in Poseidon2PeripheryCols
                row[..inner_width].copy_from_slice(inner_row);
                let cols: &mut Poseidon2PeripheryCols<Val<SC>, SBOX_REGISTERS> = row.borrow_mut();
                cols.mult = Val::<SC>::from_canonical_u32(mult);
            });
        self.records.clear();

        AirProvingContext::simple_no_pis(Arc::new(RowMajorMatrix::new(values, width)))
    }
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> ChipUsageGetter
    for Poseidon2PeripheryBaseChip<F, SBOX_REGISTERS>
{
    fn air_name(&self) -> String {
        format!("Poseidon2PeripheryAir<F, {}>", SBOX_REGISTERS)
    }

    fn current_trace_height(&self) -> usize {
        if self.nonempty.load(std::sync::atomic::Ordering::Relaxed) {
            // Not to call `DashMap::len` too often
            self.records.len()
        } else {
            0
        }
    }

    fn trace_width(&self) -> usize {
        self.air.width()
    }
}
