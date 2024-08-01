use std::iter;

use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;

use crate::{
    air_builders::symbolic::{
        symbolic_expression::{SymbolicEvaluator, SymbolicExpression},
        symbolic_variable::{Entry, SymbolicVariable},
    },
    interaction::utils::generate_betas,
};

use super::{utils::generate_rlc_elements, Interaction, InteractionType};

// Copied from valida/machine/src/chip.rs, modified to allow partitioned main trace
/// Generate the permutation trace for a chip given the main trace.
/// The permutation randomness is only available after the main trace from all chips
/// involved in interactions have been committed.
///
/// - `partitioned_main` is the main trace, partitioned into several matrices of the same height
///
/// Returns the permutation trace as a matrix of extension field elements.
///
/// ## Panics
/// - If `partitioned_main` is empty.
pub fn generate_permutation_trace<F, EF>(
    all_interactions: &[Interaction<SymbolicExpression<F>>],
    preprocessed: &Option<RowMajorMatrixView<F>>,
    partitioned_main: &[RowMajorMatrixView<F>],
    public_values: &[F],
    permutation_randomness: Option<[EF; 2]>,
) -> Option<RowMajorMatrix<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if all_interactions.is_empty() {
        return None;
    }
    let [alpha, beta] = permutation_randomness.expect("Not enough permutation challenges");

    let alphas = generate_rlc_elements(alpha, all_interactions);
    let betas = generate_betas(beta, all_interactions);

    // Compute the reciprocal columns
    //
    // Row: | q_1 | q_2 | q_3 | ... | q_n | \phi |
    // * q_i = \frac{1}{\alpha^i + \sum_j \beta^j * f_{i,j}}
    // * f_{i,j} is the jth main trace column for the ith interaction
    // * \phi is the running sum
    //
    // Note: We can optimize this by combining several reciprocal columns into one (the
    // number is subject to a target constraint degree).
    let perm_width = all_interactions.len() + 1;
    let height = partitioned_main[0].height();
    assert!(
        partitioned_main.iter().all(|m| m.height() == height),
        "All main trace parts must have same height"
    );

    // reciprocals is height x all_interactions.len()
    let reciprocals: Vec<EF> = (0..height)
        .into_par_iter()
        .flat_map(|n| -> Vec<_> {
            let evaluator = Evaluator {
                preprocessed,
                partitioned_main,
                public_values,
                height,
                local_index: n,
            };
            all_interactions
                .iter()
                .map(|interaction| {
                    let alpha = alphas[interaction.bus_index];
                    debug_assert!(interaction.fields.len() <= betas.len());
                    let mut fields = interaction.fields.iter();
                    let mut rlc = alpha
                        + evaluator.eval_expr(fields.next().expect("fields should not be empty"));
                    for (expr, &beta) in fields.zip(betas.iter().skip(1)) {
                        rlc += beta * evaluator.eval_expr(expr);
                    }
                    rlc
                })
                .collect()
        })
        .collect();
    // Zero should be vanishingly unlikely if alpha, beta are properly pseudo-randomized
    // The logup reciprocals should never be zero, so trace generation should panic if
    // trying to divide by zero.
    let perm_values = p3_field::batch_multiplicative_inverse(&reciprocals);
    drop(reciprocals);
    // Need to add the `phi` column to perm_values as a RowMajorMatrix
    // TODO[jpw]: is there a more memory efficient way to do this?
    let perm_values = perm_values
        .into_iter()
        .chunks(all_interactions.len())
        .into_iter()
        .flat_map(|row| row.chain(iter::once(EF::zero())))
        .collect();
    let mut perm = RowMajorMatrix::new(perm_values, perm_width);

    let _span = tracing::info_span!("compute logup partial sums").entered();
    // Compute the running sum column
    let mut phi = vec![EF::zero(); perm.height()];
    for n in 0..height {
        let evaluator = Evaluator {
            preprocessed,
            partitioned_main,
            public_values,
            height,
            local_index: n,
        };
        if n > 0 {
            phi[n] = phi[n - 1];
        }
        let perm_row = perm.row_slice(n);
        for (i, interaction) in all_interactions.iter().enumerate() {
            let mult = evaluator.eval_expr(&interaction.count);
            match interaction.interaction_type {
                InteractionType::Send => {
                    phi[n] += perm_row[i] * mult;
                }
                InteractionType::Receive => {
                    phi[n] -= perm_row[i] * mult;
                }
            }
        }
    }

    for (n, row) in perm.as_view_mut().rows_mut().enumerate() {
        *row.last_mut().unwrap() = phi[n];
    }
    _span.exit();

    Some(perm)
}

pub(super) struct Evaluator<'a, F: Field> {
    pub preprocessed: &'a Option<RowMajorMatrixView<'a, F>>,
    pub partitioned_main: &'a [RowMajorMatrixView<'a, F>],
    pub public_values: &'a [F],
    pub height: usize,
    pub local_index: usize,
}

impl<'a, F: Field> SymbolicEvaluator<F, F> for Evaluator<'a, F> {
    fn eval_var(&self, symbolic_var: SymbolicVariable<F>) -> F {
        let n = self.local_index;
        let height = self.height;
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => {
                self.preprocessed.unwrap().get((n + offset) % height, index)
            }
            Entry::Main { part_index, offset } => {
                self.partitioned_main[part_index].get((n + offset) % height, index)
            }
            Entry::Public => self.public_values[index],
            _ => unreachable!("There should be no after challenge variables"),
        }
    }
}
