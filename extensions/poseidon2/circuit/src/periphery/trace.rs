use std::{
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
use rustc_hash::FxBuildHasher;

use super::{Poseidon2PeripheryCols, SBOX_REGISTERS};

#[derive(Debug)]
pub struct Poseidon2PeripheryChip<F: VmField> {
    pub subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
    pub records: DashMap<[F; POSEIDON2_WIDTH], AtomicU32, FxBuildHasher>,
    pub nonempty: AtomicBool,
}

impl<F: VmField> Poseidon2PeripheryChip<F> {
    pub fn new(poseidon2_config: Poseidon2Config<F>) -> Self {
        let subchip = Poseidon2SubChip::new(poseidon2_config.constants);
        Self {
            subchip,
            records: DashMap::default(),
            nonempty: AtomicBool::new(false),
        }
    }

    /// Computes the permutation output without recording the input state.
    ///
    /// Only [`Self::perm_and_record`] contributes rows to the generated trace. Callers that need
    /// the permutation to be checked on the direct bus by the adapter must use
    /// [`Self::perm_and_record`]; this variant is for host-side computation that must not be
    /// recorded (e.g. computing an expected output without double-counting).
    pub fn permute(&self, input: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
        self.subchip.permute(input)
    }

    /// Computes the permutation output and records the input state's multiplicity.
    ///
    /// All recording must complete before [`Self::generate_proving_ctx`] is invoked: the proving
    /// pipeline guarantees this by finishing preflight execution before trace generation.
    pub fn perm_and_record(&self, input: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
        let output = self.subchip.permute(input);
        let count = self.records.entry(input).or_insert(AtomicU32::new(0));
        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.nonempty
            .store(true, std::sync::atomic::Ordering::Relaxed);
        output
    }
}

impl<RA, SC: StarkProtocolConfig> Chip<RA, CpuBackend<SC>> for Poseidon2PeripheryChip<Val<SC>>
where
    Val<SC>: VmField,
{
    /// Generates the trace and clears the recorded state.
    ///
    /// INVARIANT: this must only be called once all [`Self::perm_and_record`] calls are complete
    /// (the VM finishes preflight execution before generating proving contexts). Concurrent
    /// recording during this read-then-clear would silently drop entries, so callers must ensure
    /// recording has quiesced beforehand.
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let width = Poseidon2PeripheryCols::<Val<SC>>::width();
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
                let (input, count) = record.pair();
                (*input, count.load(std::sync::atomic::Ordering::Relaxed))
            })
            .unzip();
        inputs.extend(actual_inputs);
        multiplicities.extend(actual_multiplicities);
        inputs.resize(height, [Val::<SC>::ZERO; POSEIDON2_WIDTH]);
        multiplicities.resize(height, 0);

        let inner_trace = self.subchip.generate_trace(inputs);
        let inner_width = self.subchip.air.width();

        let mut values = Val::<SC>::zero_vec(height * width);
        values
            .par_chunks_mut(width)
            .zip(inner_trace.values.par_chunks(inner_width))
            .zip(multiplicities)
            .for_each(|((row, inner_row), mult)| {
                // WARNING: Poseidon2SubCols must be the first field in Poseidon2PeripheryCols.
                row[..inner_width].copy_from_slice(inner_row);
                let cols: &mut Poseidon2PeripheryCols<Val<SC>> = row.borrow_mut();
                cols.mult = Val::<SC>::from_u32(mult);
            });
        self.records.clear();
        self.nonempty
            .store(false, std::sync::atomic::Ordering::Relaxed);

        let trace = RowMajorMatrix::new(values, width);
        AirProvingContext::simple_no_pis(trace)
    }
}
