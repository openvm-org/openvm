//! An AIR with specified interactions can be augmented into a RAP.
//! This module auto-converts any [Air] implemented on an [InteractionBuilder] into a [Rap].

use std::borrow::Borrow;

use p3_air::{Air, ExtensionBuilder, PermutationAirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    interaction::InteractionType,
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
/// The permutation row consists of 1 column for each interaction (send or receive)
/// and one column for the partial sum of log derivative. These columns are trace columns
/// "after challenge" phase 0, and they are valued in the extension field.
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
    let mults_next = builder.all_multiplicities_next();
    let num_interactions = all_interactions.len();
    assert_eq!(num_interactions, mults_next.len());
    assert_eq!(num_interactions + 1, perm_local.len());
    assert_eq!(num_interactions + 1, perm_next.len());
    let phi_local = perm_local[num_interactions];
    let phi_next = perm_next[num_interactions];

    let alphas = generate_rlc_elements(rand_elems[0].into(), &all_interactions);
    let betas = rand_elems[1].into().powers();

    let lhs = phi_next.into() - phi_local.into();
    let mut rhs = AB::ExprEF::zero();
    let mut phi_0 = AB::ExprEF::zero();
    for (i, (interaction, mult_next)) in all_interactions.into_iter().zip(mults_next).enumerate() {
        // Reciprocal constraints
        let mut rlc = AB::ExprEF::zero();
        for (elem, beta) in interaction.fields.into_iter().zip(betas.clone()) {
            rlc += beta * elem;
        }
        rlc += alphas[interaction.bus_index].clone();
        builder.assert_one_ext(rlc * perm_local[i].into());

        let mult_local = interaction.count;

        // Build the RHS of the permutation constraint
        match interaction.interaction_type {
            InteractionType::Send => {
                phi_0 += perm_local[i].into() * mult_local;
                rhs += perm_next[i].into() * mult_next;
            }
            InteractionType::Receive => {
                phi_0 -= perm_local[i].into() * mult_local;
                rhs -= perm_next[i].into() * mult_next;
            }
        }
    }

    // Running sum constraints
    builder.when_transition().assert_eq_ext(lhs, rhs);
    builder
        .when_first_row()
        .assert_eq_ext(*perm_local.last().unwrap(), phi_0);
    builder
        .when_last_row()
        .assert_eq_ext(*perm_local.last().unwrap(), cumulative_sum);
}
