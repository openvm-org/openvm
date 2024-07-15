use p3_air::VirtualPairCol;
use p3_field::{AbstractField, ExtensionField, Field, Powers};

/// Returns [random_element, random_element^2, ..., random_element^max_power].
pub fn generate_rlc_elements<AF: AbstractField>(random_element: AF, max_power: usize) -> Vec<AF> {
    random_element.powers().skip(1).take(max_power).collect()
}

// TODO: Use Var and Expr type bounds in place of concrete fields so that
// this function can be used in `eval_permutation_constraints`.
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
