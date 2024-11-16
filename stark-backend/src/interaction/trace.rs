use itertools::izip;
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;

use super::{utils::generate_rlc_elements, Interaction, InteractionType};
use crate::{
    air_builders::symbolic::{
        symbolic_expression::{SymbolicEvaluator, SymbolicExpression},
        symbolic_variable::{Entry, SymbolicVariable},
    },
    interaction::utils::generate_betas,
};

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
    // To optimize memory and parallelism, we split the trace rows into chunks
    // based on the number of cpu threads available, and then do all
    // computations necessary for that chunk within a single thread.
    let perm_width = num_interactions.div_ceil(interaction_chunk_size) + 1;
    let mut perm_values = EF::zero_vec(height * perm_width);
    debug_assert!(
        partitioned_main.iter().all(|m| m.height() == height),
        "All main trace parts must have same height"
    );

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let height_chunk_size = height.div_ceil(num_threads);
    perm_values
        .par_chunks_mut(height_chunk_size * perm_width)
        .enumerate()
        .for_each(|(chunk_idx, perm_values)| {
            // perm_values is now local_height x perm_width row-major matrix
            let num_rows = perm_values.len() / perm_width;
            // the interaction chunking requires more memory because we must
            // allocate separate memory for the denominators and reciprocals
            let mut denoms = EF::zero_vec(num_rows * num_interactions);
            let row_offset = chunk_idx * height_chunk_size;
            // compute the denominators to be inverted:
            for (n, denom_row) in denoms.chunks_exact_mut(num_interactions).enumerate() {
                let evaluator = Evaluator {
                    preprocessed,
                    partitioned_main,
                    public_values,
                    height,
                    local_index: row_offset + n,
                };
                for (denom, interaction) in denom_row.iter_mut().zip(all_interactions.iter()) {
                    let alpha = alphas[interaction.bus_index];
                    debug_assert!(interaction.fields.len() <= betas.len());
                    let mut fields = interaction.fields.iter();
                    *denom = alpha
                        + evaluator.eval_expr(fields.next().expect("fields should not be empty"));
                    for (expr, &beta) in fields.zip(betas.iter().skip(1)) {
                        *denom += beta * evaluator.eval_expr(expr);
                    }
                }
            }

            // Zero should be vanishingly unlikely if alpha, beta are properly pseudo-randomized
            // The logup reciprocals should never be zero, so trace generation should panic if
            // trying to divide by zero.
            let reciprocals = p3_field::batch_multiplicative_inverse(&denoms);
            drop(denoms);
            // This block should already be in a single thread, but rayon is able
            // to do more magic sometimes
            perm_values
                .par_chunks_exact_mut(perm_width)
                .zip(reciprocals.par_chunks_exact(num_interactions))
                .enumerate()
                .for_each(|(n, (perm_row, reciprocal_chunk))| {
                    debug_assert_eq!(perm_row.len(), perm_width);
                    debug_assert_eq!(reciprocal_chunk.len(), num_interactions);

                    let evaluator = Evaluator {
                        preprocessed,
                        partitioned_main,
                        public_values,
                        height,
                        local_index: row_offset + n,
                    };

                    let mut row_sum = EF::ZERO;
                    for (perm_val, reciprocal_chunk, interaction_chunk) in izip!(
                        perm_row.iter_mut(),
                        reciprocal_chunk.chunks(interaction_chunk_size),
                        all_interactions.chunks(interaction_chunk_size)
                    ) {
                        for (reciprocal, interaction) in izip!(reciprocal_chunk, interaction_chunk)
                        {
                            let mut interaction_val =
                                *reciprocal * evaluator.eval_expr(&interaction.count);
                            if interaction.interaction_type == InteractionType::Receive {
                                interaction_val = -interaction_val;
                            }
                            *perm_val += interaction_val;
                        }
                        row_sum += *perm_val;
                    }

                    perm_row[perm_width - 1] = row_sum;
                });
        });

    // At this point, the trace matrix is complete except that the last column
    // has the row sum but not the partial sum
    tracing::trace_span!("compute logup partial sums").in_scope(|| {
        let mut phi = EF::ZERO;
        for perm_chunk in perm_values.chunks_exact_mut(perm_width) {
            phi += *perm_chunk.last().unwrap();
            *perm_chunk.last_mut().unwrap() = phi;
        }
    });

    Some(RowMajorMatrix::new(perm_values, perm_width))
}

pub(super) struct Evaluator<'a, F: Field> {
    pub preprocessed: &'a Option<RowMajorMatrixView<'a, F>>,
    pub partitioned_main: &'a [RowMajorMatrixView<'a, F>],
    pub public_values: &'a [F],
    pub height: usize,
    pub local_index: usize,
}

impl<F: Field> SymbolicEvaluator<F, F> for Evaluator<'_, F> {
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
