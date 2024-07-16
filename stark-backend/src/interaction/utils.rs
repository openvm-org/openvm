use p3_air::VirtualPairCol;
use p3_field::{AbstractField, ExtensionField, Field, Powers};

use super::Interaction;

/// Returns [random_element, random_element^2, ..., random_element^{max_bus_index + 1}].
pub fn generate_rlc_elements<AF: AbstractField, E>(
    random_element: AF,
    all_interactions: &[Interaction<E>],
) -> Vec<AF> {
    let max_bus_index = all_interactions
        .iter()
        .map(|interaction| interaction.bus_index)
        .max()
        .unwrap_or(0);

    random_element
        .powers()
        .skip(1)
        .take(max_bus_index + 1)
        .collect()
}

// TODO: Use Var and Expr type bounds in place of concrete fields so that
// this function can be used in `eval_permutation_constraints`.
#[allow(dead_code)]
pub fn reduce_row<F, EF>(
    preprocessed_row: &[F],
    main_row: &[F],
    fields: &[VirtualPairCol<F>],
    alpha: EF,
    betas: Powers<EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut rlc = EF::zero();
    for (columns, beta) in fields.iter().zip(betas) {
        rlc += beta * columns.apply::<F, F>(preprocessed_row, main_row)
    }
    rlc += alpha;
    rlc
}
