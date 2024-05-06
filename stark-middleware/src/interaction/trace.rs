use std::ops::Deref;

use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};

use crate::utils::batch_multiplicative_inverse_allowing_zero;

use super::{
    utils::{generate_rlc_elements, reduce_row},
    Chip, InteractionType,
};

// Copied from valida/machine/src/chip.rs
/// Generate the permutation trace for a chip given the main trace.
/// The permutation randomness is only available after the main trace from all chips
/// involved in interactions have been committed.
///
/// Returns the permutation trace as a matrix of extension field elements.
pub fn generate_permutation_trace<F, C, EF>(
    chip: &C,
    main: &RowMajorMatrixView<F>,
    permutation_randomness: [EF; 2],
) -> RowMajorMatrix<EF>
where
    F: Field,
    C: Chip<F> + ?Sized,
    EF: ExtensionField<F>,
{
    let all_interactions = chip.all_interactions();
    if all_interactions.is_empty() {
        return RowMajorMatrix::new(vec![], 0);
    }
    let [alpha, beta] = permutation_randomness;
    let alphas = generate_rlc_elements(chip, alpha);
    let betas = beta.powers();

    // TODO:
    // let preprocessed = chip.preprocessed_trace();

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
    let height = main.height();
    let mut perm_values = Vec::with_capacity(height * perm_width);

    for n in 0..height {
        let main_row = main.row_slice(n); // main.rows() clones unnecessarily
        let mut row = vec![EF::zero(); perm_width];
        for (m, (interaction, _)) in all_interactions.iter().enumerate() {
            let alpha_m = alphas[interaction.argument_index];
            // let preprocessed_row = preprocessed
            //     .as_ref()
            //     .map(|preprocessed| {
            //         let row = preprocessed.row_slice(n);
            //         let row: &[_] = (*row).borrow();
            //         row.to_vec()
            //     })
            //     .unwrap_or_default();
            row[m] = reduce_row(
                main_row.deref(),
                &[], // preprocessed_row.as_slice(),
                &interaction.fields,
                alpha_m,
                betas.clone(),
            );
        }
        perm_values.extend(row);
    }
    // TODO: Switch to batch_multiplicative_inverse (not allowing zero)?
    // Zero should be vanishingly unlikely if properly randomized?
    let perm_values = batch_multiplicative_inverse_allowing_zero(perm_values);
    let mut perm = RowMajorMatrix::new(perm_values, perm_width);

    // Compute the running sum column
    let mut phi = vec![EF::zero(); perm.height()];
    for n in 0..height {
        let main_row = main.row_slice(n);
        let perm_row = perm.row_slice(n);
        if n > 0 {
            phi[n] = phi[n - 1];
        }
        // TODO:
        let preprocessed_row = &[];
        // let preprocessed_row = preprocessed
        //     .as_ref()
        //     .map(|preprocessed| {
        //         let row = preprocessed.row_slice(n);
        //         let row: &[_] = (*row).borrow();
        //         row.to_vec()
        //     })
        //     .unwrap_or_default();
        for (m, (interaction, interaction_type)) in all_interactions.iter().enumerate() {
            let mult = interaction
                .count
                .apply::<F, F>(preprocessed_row.as_slice(), main_row.deref());
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

    perm
}
