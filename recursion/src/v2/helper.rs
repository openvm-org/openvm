use afs_compiler::prelude::*;
use itertools::Itertools;

use crate::v2::vars::{MultiStarkVerificationAdviceV2Variable, StarkProofV2Variable};

impl<C: Config> StarkProofV2Variable<C> {
    pub fn get_air_ids(&self, builder: &mut Builder<C>) -> Array<C, Usize<C::N>> {
        if builder.flags.static_only {
            builder.vec(
                (0..self.per_air.len().value())
                    .map(Usize::from)
                    .collect_vec(),
            )
        } else {
            let air_ids = builder.array(self.per_air.len());
            builder.range(0, self.per_air.len()).for_each(|i, builder| {
                let air_proof_data = builder.get(&self.per_air, i);
                builder.set_value(&air_ids, i, air_proof_data.air_id);
            });
            air_ids
        }
    }
}

impl<C: Config> MultiStarkVerificationAdviceV2Variable<C> {
    /// Assumption: at most 1 phase is supported.
    pub fn num_challenges_to_sample(&self, builder: &mut Builder<C>) -> Array<C, Usize<C::N>> {
        if self.num_challenges_to_sample_mask.is_empty() {
            return builder.array(0);
        }
        // If all 0s, no phase 1.
        let num_phases: Usize<_> = builder.eval(self.num_challenges_to_sample_mask[0][0].clone());
        let ret = builder.array(num_phases.clone());
        // If phase 1 exists:
        builder
            .if_eq(num_phases.clone(), RVar::one())
            .then(|builder| {
                // Find the biggest index where the mask is 1.
                let num_challenges: Usize<_> = builder.eval(RVar::zero());
                for i in 1..self.num_challenges_to_sample_mask[0].len() {
                    builder
                        .if_eq(
                            self.num_challenges_to_sample_mask[0][i].clone(),
                            RVar::one(),
                        )
                        .then(|builder| {
                            builder.assign(&num_challenges, RVar::from(i));
                        });
                }
                builder.set(&ret, RVar::zero(), num_challenges + RVar::one());
            });
        ret
    }
}
