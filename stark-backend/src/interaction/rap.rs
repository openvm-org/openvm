//! An AIR with specified interactions can be augmented into a RAP.
//! This module auto-converts any [Air] implemented on an [InteractionBuilder] into a [Rap].

use std::borrow::Borrow;

use p3_air::{Air, ExtensionBuilder, PermutationAirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    interaction::{utils::generate_betas, InteractionType},
    rap::{PermutationAirBuilderWithExposedValues, Rap},
};

use super::{utils::generate_rlc_elements, InteractionBuilder};

impl<AB, A> Rap<AB> for A
where
    A: Air<AB>,
    AB: InteractionBuilder + PermutationAirBuilderWithExposedValues,
{
    fn eval(&self, builder: &mut AB) {
        // Constraits for the main trace:
        Air::eval(self, builder);
        builder.finalize_interactions();
        // Constraints for the permutation trace:
        // If no interactions, nothing to do
        if builder.num_interactions() > 0 {
            let exposed_values = builder.permutation_exposed_values();
            // There are interactions, add constraints for the virtual columns
            assert_eq!(
                exposed_values.len(),
                1,
                "Should have one exposed value for cumulative_sum"
            );
            let cumulative_sum = exposed_values[0];
            eval_permutation_constraints(builder, cumulative_sum);
        }
    }
}

// Initial version taken from valida/machine/src/chip.rs under MIT license.
/// The permutation row consists of 1 column for each bundle of interactions
/// and one column for the partial sum of log derivative. These columns are trace columns
/// "after challenge" phase 0, and they are valued in the extension field.
/// For more details, see the comment in the trace.rs file
pub fn eval_permutation_constraints<AB>(builder: &mut AB, cumulative_sum: AB::VarEF)
where
    AB: InteractionBuilder + PermutationAirBuilder,
{
    let rand_elems = builder.permutation_randomness().to_vec();

    let perm = builder.permutation();
    let [perm_local, perm_next] = [0, 1].map(|i| perm.row_slice(i));
    let perm_local: &[AB::VarEF] = (*perm_local).borrow();
    let perm_next: &[AB::VarEF] = (*perm_next).borrow();

    let all_interactions = builder.all_interactions().to_vec();
    let interaction_chunk_size = builder.interaction_chunk_size();
    let num_interactions = all_interactions.len();
    let perm_width = (num_interactions + interaction_chunk_size - 1) / interaction_chunk_size + 1;
    debug_assert_eq!(perm_width, perm_local.len());
    debug_assert_eq!(perm_width, perm_next.len());
    let phi_local = *perm_local.last().unwrap();
    let phi_next = *perm_next.last().unwrap();

    let alphas = generate_rlc_elements(rand_elems[0].into(), &all_interactions);
    let betas = generate_betas(rand_elems[1].into(), &all_interactions);

    let phi_lhs = phi_next.into() - phi_local.into();
    let mut phi_rhs = AB::ExprEF::zero();
    let mut phi_0 = AB::ExprEF::zero();

    for (chunk_idx, interaction_chunk) in
        all_interactions.chunks(interaction_chunk_size).enumerate()
    {
        let mut denoms = vec![AB::ExprEF::zero(); interaction_chunk.len()];
        let interaction_chunk = interaction_chunk.to_vec();
        for (i, interaction) in interaction_chunk.iter().enumerate() {
            assert!(!interaction.fields.is_empty(), "fields should not be empty");
            let mut denom = alphas[interaction.bus_index].clone();
            for (elem, beta) in interaction.fields.iter().zip(betas.iter()) {
                denom += beta.clone() * elem.clone();
            }
            denoms[i] = denom;
        }

        let mut row_lhs: AB::ExprEF = perm_local[chunk_idx].into();
        for denom in denoms.iter() {
            row_lhs *= denom.clone();
        }

        let mut row_rhs = AB::ExprEF::zero();
        for (i, interaction) in interaction_chunk.into_iter().enumerate() {
            let mut term: AB::ExprEF = interaction.count.into();
            if interaction.interaction_type == InteractionType::Receive {
                term = -term;
            }
            for (j, denom) in denoms.iter().enumerate() {
                if i != j {
                    term *= denom.clone();
                }
            }
            row_rhs += term;
        }

        builder.assert_eq_ext(row_lhs, row_rhs);

        phi_0 += perm_local[chunk_idx].into();
        phi_rhs += perm_next[chunk_idx].into();
    }

    // Running sum constraints
    builder.when_transition().assert_eq_ext(phi_lhs, phi_rhs);
    builder
        .when_first_row()
        .assert_eq_ext(*perm_local.last().unwrap(), phi_0);
    builder
        .when_last_row()
        .assert_eq_ext(*perm_local.last().unwrap(), cumulative_sum);
}
