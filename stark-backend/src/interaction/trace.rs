use itertools::izip;
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
    interaction_chunk_size: usize,
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
    // For every row we do the following
    // We first compute the reciprocals: r_1, r_2, ..., r_n, where
    // r_i = \frac{1}{\alpha^i + \sum_j \beta^j * f_{i, j}}, where
    // f_{i, j} is the jth main trace column for the ith interaction
    //
    // We then bundle every interaction_chunk_size interactions together
    // to get the value perm_i = \sum_{i \in bundle} r_i * m_i, where m_i
    // is the signed count for the interaction.
    //
    // Finally, the last column, \phi, of every row is the running sum of
    // all the previous perm values
    //
    // Row: | perm_1 | perm_2 | perm_3 | ... | perm_s | phi |, where s
    // is the number of bundles
    let num_interactions = all_interactions.len();
    let height = partitioned_main[0].height();
    assert!(
        partitioned_main.iter().all(|m| m.height() == height),
        "All main trace parts must have same height"
    );

    let mut demons = vec![EF::zero(); height * num_interactions];
    for (n, chunk) in demons.chunks_mut(num_interactions).enumerate() {
        let evaluator = Evaluator {
            preprocessed,
            partitioned_main,
            public_values,
            height,
            local_index: n,
        };

        for (i, interaction) in all_interactions.iter().enumerate() {
            let alpha = alphas[interaction.bus_index];
            debug_assert!(interaction.fields.len() <= betas.len());
            let mut fields = interaction.fields.iter();
            let mut denom =
                alpha + evaluator.eval_expr(fields.next().expect("fields should not be empty"));
            for (expr, &beta) in fields.zip(betas.iter().skip(1)) {
                denom += beta * evaluator.eval_expr(expr);
            }
            chunk[i] = denom;
        }
    }

    // Zero should be vanishingly unlikely if alpha, beta are properly pseudo-randomized
    // The logup reciprocals should never be zero, so trace generation should panic if
    // trying to divide by zero.
    let reciprocals = p3_field::batch_multiplicative_inverse(&demons);
    drop(demons);

    let perm_width = (num_interactions + interaction_chunk_size - 1) / interaction_chunk_size + 1;
    let mut perm_values = vec![EF::zero(); height * perm_width];

    perm_values
        .par_chunks_mut(perm_width)
        .zip(reciprocals.par_chunks(num_interactions))
        .enumerate()
        .for_each(|(n, (perm_row, reciprocal_chunk))| {
            debug_assert!(perm_row.len() == perm_width);
            debug_assert!(reciprocal_chunk.len() == num_interactions);

            let evaluator = Evaluator {
                preprocessed,
                partitioned_main,
                public_values,
                height,
                local_index: n,
            };

            let mut row_sum = EF::zero();
            for (perm_val, reciprocal_chunk, interaction_chunk) in izip!(
                perm_row.iter_mut(),
                reciprocal_chunk.chunks(interaction_chunk_size),
                all_interactions.chunks(interaction_chunk_size)
            ) {
                for (reciprocal, interaction) in izip!(reciprocal_chunk, interaction_chunk) {
                    let mut interaction_val = *reciprocal * evaluator.eval_expr(&interaction.count);
                    if interaction.interaction_type == InteractionType::Receive {
                        interaction_val = -interaction_val;
                    }
                    *perm_val += interaction_val;
                }
                row_sum += *perm_val;
            }

            perm_row[perm_width - 1] = row_sum;
        });

    let _span = tracing::info_span!("compute logup partial sums").entered();
    let mut phi = EF::zero();
    for perm_chunk in perm_values.chunks_mut(perm_width) {
        phi += perm_chunk[perm_width - 1];
        perm_chunk[perm_width - 1] = phi;
    }
    _span.exit();

    Some(RowMajorMatrix::new(perm_values, perm_width))
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
