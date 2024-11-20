//! An AIR with specified interactions can be augmented into a RAP.
//! This module auto-converts any [Air] implemented on an [InteractionBuilder] into a [Rap].

use p3_air::Air;

use super::{InteractionBuilder, RapPhaseSeqKind};
use crate::{
    interaction::stark_log_up::eval_stark_log_up_phase,
    rap::{PermutationAirBuilderWithExposedValues, Rap},
};

/// Used internally to select RAP phase evaluation function.
pub(crate) trait InteractionPhaseAirBuilder {
    fn finalize_interactions(&mut self);
    fn interaction_chunk_size(&self) -> usize;
    fn rap_phase_seq_kind(&self) -> RapPhaseSeqKind;
}

impl<AB, A> Rap<AB> for A
where
    A: Air<AB>,
    AB: InteractionBuilder + PermutationAirBuilderWithExposedValues + InteractionPhaseAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Constraints for the main trace:
        Air::eval(self, builder);
        builder.finalize_interactions();
        if builder.num_interactions() != 0 {
            match builder.rap_phase_seq_kind() {
                RapPhaseSeqKind::StarkLogUp => {
                    eval_stark_log_up_phase(builder, builder.interaction_chunk_size());
                }
                RapPhaseSeqKind::GkrLogUp => todo!(),
            }
        }
    }
}
