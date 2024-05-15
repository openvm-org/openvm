use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::IntoParallelIterator;

use crate::utils::batch_multiplicative_inverse_allowing_zero;

use super::{
    utils::{generate_rlc_elements, reduce_row},
    Chip, InteractionType,
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
pub fn generate_permutation_trace<F, C, EF>(
    chip: &C,
    preprocessed: &Option<RowMajorMatrixView<F>>,
    partitioned_main: &[RowMajorMatrixView<F>],
    permutation_randomness: Option<[EF; 2]>,
) -> Option<RowMajorMatrix<EF>>
where
    F: Field,
    C: Chip<F> + ?Sized,
    EF: ExtensionField<F>,
{
    let all_interactions = chip.all_interactions();
    if all_interactions.is_empty() {
        return None;
    }
    let [alpha, beta] = permutation_randomness.expect("Not enough permutation challenges");

    let alphas = generate_rlc_elements(chip, alpha);
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
            // !!TODO!! This copies all rows, BAD for performance
            let main_row: Vec<F> = partitioned_main
                .iter()
                .flat_map(|main_part| main_part.row_slice(n).to_vec())
                .collect();
            // Recall: perm_width = all_interactions.len() + 1
            let mut row = vec![EF::zero(); perm_width];
            for (row_j, (interaction, _)) in row.iter_mut().zip(&all_interactions) {
                let alpha_m = alphas[interaction.argument_index];
                let preprocessed_row = preprocessed
                    .as_ref()
                    .map(|preprocessed| {
                        // manual implementation of row_slice because of a drop issue
                        &preprocessed.values[n * preprocessed.width..(n + 1) * preprocessed.width]
                    })
                    .unwrap_or(&[]);
                *row_j = reduce_row(
                    preprocessed_row,
                    &main_row,
                    &interaction.fields,
                    alpha_m,
                    betas.clone(),
                );
            }
            row
        })
        .collect();
    // TODO: Switch to batch_multiplicative_inverse (not allowing zero)?
    // Zero should be vanishingly unlikely if properly randomized?
    let perm_values = batch_multiplicative_inverse_allowing_zero(perm_values);
    let mut perm = RowMajorMatrix::new(perm_values, perm_width);

    // Compute the running sum column
    let mut phi = vec![EF::zero(); perm.height()];
    for n in 0..height {
        // !!TODO!! This copies all rows, BAD for performance
        let main_row: Vec<F> = partitioned_main
            .iter()
            .flat_map(|main_part| main_part.row_slice(n).to_vec())
            .collect();
        let perm_row = perm.row_slice(n);
        if n > 0 {
            phi[n] = phi[n - 1];
        }
        let preprocessed_row = preprocessed
            .as_ref()
            .map(|preprocessed| {
                // manual implementation of row_slice because of a drop issue
                &preprocessed.values[n * preprocessed.width..(n + 1) * preprocessed.width]
            })
            .unwrap_or(&[]);
        for (m, (interaction, interaction_type)) in all_interactions.iter().enumerate() {
            let mult = interaction.count.apply::<F, F>(preprocessed_row, &main_row);
            match interaction_type {
                InteractionType::Send => {
                    phi[n] += perm_row[m] * mult;
                }
                InteractionType::Receive => {
                    phi[n] -= perm_row[m] * mult;
                }
            }
        }
    }

    for (n, row) in perm.as_view_mut().rows_mut().enumerate() {
        *row.last_mut().unwrap() = phi[n];
    }

    Some(perm)
}
