use std::borrow::BorrowMut;

use openvm_circuit_primitives::{utils::next_power_of_two_or_zero, Chip};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    StarkProtocolConfig, Val,
};

use super::{columns::*, Poseidon2PeripheryBaseChip, PERIPHERY_POSEIDON2_WIDTH};
use crate::arch::VmField;

impl<RA, SC: StarkProtocolConfig, const SBOX_REGISTERS: usize> Chip<RA, CpuBackend<SC>>
    for Poseidon2PeripheryBaseChip<Val<SC>, SBOX_REGISTERS>
where
    Val<SC>: VmField,
{
    /// Generates trace and clears internal records state.
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let current_height = if self.nonempty.load(std::sync::atomic::Ordering::Relaxed) {
            self.records.len()
        } else {
            0
        };
        let height = next_power_of_two_or_zero(current_height);
        let width = Poseidon2PeripheryCols::<Val<SC>, SBOX_REGISTERS>::width();

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
        let inner_width = self.subchip.air.width();

        let mut values = Val::<SC>::zero_vec(height * width);
        values
            .par_chunks_mut(width)
            .zip(inner_trace.values.par_chunks(inner_width))
            .zip(multiplicities)
            .for_each(|((row, inner_row), mult)| {
                // WARNING: Poseidon2SubCols must be the first field in Poseidon2PeripheryCols
                row[..inner_width].copy_from_slice(inner_row);
                let cols: &mut Poseidon2PeripheryCols<Val<SC>, SBOX_REGISTERS> = row.borrow_mut();
                cols.mult = Val::<SC>::from_u32(mult);
            });
        self.records.clear();

        let trace = RowMajorMatrix::new(values, width);
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace))
    }
}
