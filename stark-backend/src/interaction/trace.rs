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
    utils::batch_multiplicative_inverse_allowing_zero,
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
    let betas = beta.powers();

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

    // perm_values is height x perm_width
    let perm_values: Vec<EF> = (0..height)
        .into_par_iter()
        .flat_map(|n| {
            let evaluator = Evaluator {
                preprocessed,
                partitioned_main,
                public_values,
                height,
                local_index: n,
            };
            // Recall: perm_width = all_interactions.len() + 1
            let mut row = vec![EF::zero(); perm_width];
            for (row_j, interaction) in row.iter_mut().zip(all_interactions) {
                let alpha = alphas[interaction.bus_index];
                let mut rlc = EF::zero();
                for (expr, beta) in interaction.fields.iter().zip(betas.clone()) {
                    rlc += beta * evaluator.eval_expr(expr);
                }
                rlc += alpha;
                *row_j = rlc;
            }
            row
        })
        .collect();
    // TODO: Switch to batch_multiplicative_inverse (not allowing zero)?
    // Zero should be vanishingly unlikely if properly randomized?
    let perm_values = batch_multiplicative_inverse_allowing_zero(perm_values);
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
