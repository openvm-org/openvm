//! An AIR with specified interactions can be augmented into a RAP.
//! This module implements this construction in the [InteractiveAir] struct.

use std::borrow::Borrow;

use p3_air::{Air, ExtensionBuilder, PairBuilder, PermutationAirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionType,
    rap::{PermutationAirBuilderWithExposedValues, Rap},
};

use super::{utils::generate_rlc_elements, Chip, InteractiveAir};

impl<AB, A> Rap<AB> for A
where
    A: InteractiveAir<AB>,
    AB: PairBuilder + PermutationAirBuilderWithExposedValues + PartitionedAirBuilder + Sync,
{
    fn eval(&self, builder: &mut AB) {
        // Constraits for the main trace:
        Air::eval(self, builder);
        // Constraints for the permutation trace:
        let num_interactions = self.all_interactions().len();
        // If no interactions, nothing to do
        if num_interactions > 0 {
            let exposed_values = builder.permutation_exposed_values();
            // There are interactions, add constraints for the virtual columns
            assert_eq!(
                exposed_values.len(),
                1,
                "Should have one exposed value for cumulative_sum"
            );
            let cumulative_sum = exposed_values[0];
            eval_permutation_constraints(self, builder, cumulative_sum);
        }
    }
}

// Copied from valida/machine/src/chip.rs
/// The permutation row consists of 1 virtual column for each interaction (send or receive)
/// and one virtual column for the partial sum of log derivative.
pub fn eval_permutation_constraints<C, AB>(chip: &C, builder: &mut AB, cumulative_sum: AB::EF)
where
    C: Chip<AB::F>,
    AB: PairBuilder + PermutationAirBuilder + PartitionedAirBuilder,
{
    let rand_elems = builder.permutation_randomness().to_vec();

    let preprocessed = builder.preprocessed();
    let preprocessed_local = preprocessed.row_slice(0);
    let preprocessed_next = preprocessed.row_slice(1);
    let preprocessed_local = (*preprocessed_local).borrow();
    let preprocessed_next = (*preprocessed_next).borrow();

    let partitioned_main = builder.partitioned_main();
    let (main_locals, main_nexts): (Vec<_>, Vec<_>) = partitioned_main
        .iter()
        .map(|mat| (mat.row_slice(0), mat.row_slice(1)))
        .unzip();
    // NEEDS OPTIMIZATION: VirtualPairCol::apply expects `main_local`.
    // Without changing plonky3, we just need to copy and concatenate the partitioned slices together.
    let [main_local, main_next] = [main_locals, main_nexts].map(|row_parts| {
        row_parts
            .iter()
            .flat_map(|r| (*r).borrow() as &[_])
            .copied()
            .collect::<Vec<_>>()
    });

    let perm = builder.permutation();
    let perm_local = perm.row_slice(0);
    let perm_next = perm.row_slice(1);
    let perm_local: &[AB::VarEF] = (*perm_local).borrow();
    let perm_next: &[AB::VarEF] = (*perm_next).borrow();
    let perm_width = perm.width();

    let phi_local = perm_local[perm_width - 1];
    let phi_next = perm_next[perm_width - 1];

    let all_interactions = chip.all_interactions();

    let alphas = generate_rlc_elements(chip, rand_elems[0].into());
    let betas = rand_elems[1].into().powers();

    let lhs = phi_next.into() - phi_local.into();
    let mut rhs = AB::ExprEF::zero();
    let mut phi_0 = AB::ExprEF::zero();
    for (m, (interaction, interaction_type)) in all_interactions.iter().enumerate() {
        // Reciprocal constraints
        let mut rlc = AB::ExprEF::zero();
        for (field, beta) in interaction.fields.iter().zip(betas.clone()) {
            let elem = field.apply::<AB::Expr, AB::Var>(preprocessed_local, &main_local);
            rlc += beta * elem;
        }
        rlc += alphas[interaction.argument_index].clone();
        builder.assert_one_ext(rlc * perm_local[m].into());

        let mult_local = interaction
            .count
            .apply::<AB::Expr, AB::Var>(preprocessed_local, &main_local);
        let mult_next = interaction
            .count
            .apply::<AB::Expr, AB::Var>(preprocessed_next, &main_next);

        // Build the RHS of the permutation constraint
        match interaction_type {
            InteractionType::Send => {
                phi_0 += perm_local[m].into() * mult_local;
                rhs += perm_next[m].into() * mult_next;
            }
            InteractionType::Receive => {
                phi_0 -= perm_local[m].into() * mult_local;
                rhs -= perm_next[m].into() * mult_next;
            }
        }
    }

    // Running sum constraints
    builder.when_transition().assert_eq_ext(lhs, rhs);
    builder
        .when_first_row()
        .assert_eq_ext(*perm_local.last().unwrap(), phi_0);
    builder.when_last_row().assert_eq_ext(
        *perm_local.last().unwrap(),
        AB::ExprEF::from_f(cumulative_sum),
    );
}
