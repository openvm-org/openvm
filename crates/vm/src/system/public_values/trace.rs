use std::{borrow::BorrowMut, sync::Arc};

use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{
    p3_matrix::dense::RowMajorMatrix, prover::AirProvingContext, StarkProtocolConfig,
};

use super::{
    public_values_commit, public_values_event_block, value_limbs, PublicValuesAir,
    PublicValuesCols, PublicValuesPvs,
};
use crate::{
    arch::{hasher::HasherChip, PublicValuesState, VmField},
    system::poseidon2::Poseidon2PeripheryChip,
};

pub struct PublicValuesChip<F: VmField> {
    pub air: PublicValuesAir,
    hasher: Arc<Poseidon2PeripheryChip<F>>,
}

impl<F: VmField> PublicValuesChip<F> {
    pub fn new(air: PublicValuesAir, hasher: Arc<Poseidon2PeripheryChip<F>>) -> Self {
        Self { air, hasher }
    }

    pub fn generate_proving_ctx<SC>(
        &self,
        state: &PublicValuesState,
        initial_len: usize,
    ) -> AirProvingContext<CpuBackend<SC>>
    where
        SC: StarkProtocolConfig<F = F>,
    {
        let max_values = self.air.trace_height();
        assert_eq!(state.max_public_values(), max_values);
        assert!(initial_len <= state.len());
        let segment_values = &state.values()[initial_len..];
        assert!(segment_values.len() <= max_values);

        let initial_commit = public_values_commit(
            &state.values()[..initial_len],
            max_values,
            self.hasher.as_ref(),
        );
        let mut commit = initial_commit;
        let width = PublicValuesCols::<F>::width();
        let mut trace = F::zero_vec(width * self.air.trace_height());
        for (ordinal, trace_row) in trace.chunks_exact_mut(width).enumerate() {
            let value = segment_values.get(ordinal).copied();
            let limbs = value_limbs(value.unwrap_or_default());
            let hash = value
                .map(|value| {
                    self.hasher
                        .compress_and_record(&commit, &public_values_event_block(value))
                })
                .unwrap_or(commit);
            *trace_row.borrow_mut() = PublicValuesCols {
                is_valid: F::from_bool(value.is_some()),
                ordinal: F::from_usize(ordinal),
                commit,
                hash,
                value: limbs,
            };
            if value.is_some() {
                commit = hash;
            }
        }

        debug_assert_eq!(
            commit,
            public_values_commit(state.values(), max_values, self.hasher.as_ref())
        );
        let pvs = PublicValuesPvs {
            initial_commit,
            final_commit: commit,
        };
        let pvs = pvs
            .initial_commit
            .into_iter()
            .chain(pvs.final_commit)
            .collect();
        AirProvingContext::simple(RowMajorMatrix::new(trace, width), pvs)
    }
}
