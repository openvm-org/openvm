use std::{
    array::from_fn,
    borrow::BorrowMut,
    sync::atomic::{AtomicBool, AtomicU32},
};

use dashmap::DashMap;
use openvm_circuit::arch::VmField;
use openvm_circuit_primitives::{utils::next_power_of_two_or_zero, Chip};
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip, POSEIDON2_WIDTH};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::PrimeCharacteristicRing, p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*, prover::AirProvingContext, StarkProtocolConfig, Val,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use rustc_hash::FxBuildHasher;

use super::{DeferralPoseidon2Cols, SBOX_REGISTERS};
use crate::chunks_to_state;

#[derive(Debug)]
pub struct DeferralPoseidon2Chip<F: VmField> {
    pub subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
    pub records: DashMap<[F; POSEIDON2_WIDTH], (AtomicU32, AtomicU32), FxBuildHasher>,
    pub nonempty: AtomicBool,
}

impl<F: VmField> DeferralPoseidon2Chip<F> {
    pub fn new(poseidon2_config: Poseidon2Config<F>) -> Self {
        let subchip = Poseidon2SubChip::new(poseidon2_config.constants);
        Self {
            subchip,
            records: DashMap::default(),
            nonempty: AtomicBool::new(false),
        }
    }

    pub fn perm(
        &self,
        lhs: &[F; DIGEST_SIZE],
        rhs: &[F; DIGEST_SIZE],
        is_compress: bool,
    ) -> [F; DIGEST_SIZE] {
        let output = self.perm_state(lhs, rhs);
        self.select_output_chunk(output, is_compress)
    }

    pub fn perm_state(
        &self,
        lhs: &[F; DIGEST_SIZE],
        rhs: &[F; DIGEST_SIZE],
    ) -> [F; POSEIDON2_WIDTH] {
        let input = chunks_to_state(lhs, rhs);
        self.subchip.permute(input)
    }

    pub fn perm_and_record(
        &self,
        lhs: &[F; DIGEST_SIZE],
        rhs: &[F; DIGEST_SIZE],
        is_compress: bool,
    ) -> [F; DIGEST_SIZE] {
        let input = chunks_to_state(lhs, rhs);
        let output = self.subchip.permute(input);
        let ret = self.select_output_chunk(output, is_compress);
        let count = self
            .records
            .entry(input)
            .or_insert((AtomicU32::new(0), AtomicU32::new(0)));
        let mult = if is_compress { &count.0 } else { &count.1 };
        mult.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.nonempty
            .store(true, std::sync::atomic::Ordering::Relaxed);
        ret
    }

    fn select_output_chunk(
        &self,
        output: [F; POSEIDON2_WIDTH],
        is_compress: bool,
    ) -> [F; DIGEST_SIZE] {
        let offset = if is_compress { 0 } else { DIGEST_SIZE };
        from_fn(|i| output[i + offset])
    }
}

impl<RA, SC: StarkProtocolConfig> Chip<RA, CpuBackend<SC>> for DeferralPoseidon2Chip<Val<SC>>
where
    Val<SC>: VmField,
{
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let width = DeferralPoseidon2Cols::<Val<SC>>::width();
        if !self.nonempty.load(std::sync::atomic::Ordering::Relaxed) {
            let trace = RowMajorMatrix::new(vec![], width);
            return AirProvingContext::simple_no_pis(trace);
        }
        let height = next_power_of_two_or_zero(self.records.len());

        let mut inputs = Vec::with_capacity(height);
        let mut multiplicities = Vec::with_capacity(height);
        #[cfg(feature = "parallel")]
        let records_iter = self.records.par_iter();
        #[cfg(not(feature = "parallel"))]
        let records_iter = self.records.iter();
        let (actual_inputs, actual_multiplicities): (Vec<_>, Vec<_>) = records_iter
            .map(|record| {
                let (input, (compress_mult, capacity_mult)) = record.pair();
                (
                    *input,
                    (
                        compress_mult.load(std::sync::atomic::Ordering::Relaxed),
                        capacity_mult.load(std::sync::atomic::Ordering::Relaxed),
                    ),
                )
            })
            .unzip();
        inputs.extend(actual_inputs);
        multiplicities.extend(actual_multiplicities);
        inputs.resize(height, [Val::<SC>::ZERO; POSEIDON2_WIDTH]);
        multiplicities.resize(height, (0, 0));

        let inner_trace = self.subchip.generate_trace(inputs);
        let inner_width = self.subchip.air.width();

        let mut values = Val::<SC>::zero_vec(height * width);
        values
            .par_chunks_mut(width)
            .zip(inner_trace.values.par_chunks(inner_width))
            .zip(multiplicities)
            .for_each(|((row, inner_row), (compress_mult, capacity_mult))| {
                // WARNING: Poseidon2SubCols must be the first field in DeferralPoseidon2Cols.
                row[..inner_width].copy_from_slice(inner_row);
                let cols: &mut DeferralPoseidon2Cols<Val<SC>> = row.borrow_mut();
                cols.compress_mult = Val::<SC>::from_u32(compress_mult);
                cols.capacity_mult = Val::<SC>::from_u32(capacity_mult);
            });
        self.records.clear();
        self.nonempty
            .store(false, std::sync::atomic::Ordering::Relaxed);

        let trace = RowMajorMatrix::new(values, width);
        AirProvingContext::simple_no_pis(trace)
    }
}
