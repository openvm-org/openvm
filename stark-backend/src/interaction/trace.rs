use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use rayon::current_num_threads;

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
    let mut reciprocals = vec![EF::one(); height * perm_width];
    reciprocals
        .par_chunks_mut(perm_width)
        .enumerate()
        .for_each(|(i, r)| {
            let evaluator = Evaluator {
                preprocessed,
                partitioned_main,
                public_values,
                height,
                local_index: i,
            };
            for (j, interaction) in all_interactions.iter().enumerate() {
                let alpha = alphas[interaction.bus_index];
                debug_assert!(interaction.fields.len() <= betas.len());
                let mut fields = interaction.fields.iter();
                let mut rlc =
                    alpha + evaluator.eval_expr(fields.next().expect("fields should not be empty"));
                for (expr, &beta) in fields.zip(betas.iter().skip(1)) {
                    rlc += beta * evaluator.eval_expr(expr);
                }
                r[j] = rlc;
            }
        });
    #[cfg(feature = "parallel")]
    let old_perm_values = {
        let num_threads = current_num_threads();
        let chunk_size = (reciprocals.len() + num_threads - 1) / num_threads;
        let mut vals = vec![EF::zero(); reciprocals.len()];
        vals.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, r)| {
                batch_multiplicative_inverse_with_buf(&reciprocals[i * chunk_size..], r);
            });
        vals
    };
    #[cfg(not(feature = "parallel"))]
    let old_perm_values = p3_field::batch_multiplicative_inverse(&reciprocals);

    // Zero should be vanishingly unlikely if alpha, beta are properly pseudo-randomized
    // The logup reciprocals should never be zero, so trace generation should panic if
    // trying to divide by zero.
    // let old_perm_values = p3_field::batch_multiplicative_inverse(&reciprocals);
    drop(reciprocals);
    // Need to add the `phi` column to perm_values as a RowMajorMatrix
    // TODO[jpw]: is there a more memory efficient way to do this?
    // let mut perm_values = vec![EF::zero(); height * perm_width];
    // perm_values
    //     .chunks_mut(perm_width)
    //     .zip(old_perm_values.chunks(perm_width - 1))
    //     .for_each(|(row, old_row)| {
    //         let (left, _) = row.split_at_mut(perm_width - 1);
    //         left.copy_from_slice(old_row)
    //     });

    let _span = tracing::info_span!("compute logup partial sums").entered();
    // Compute the running sum column
    let mut perm = RowMajorMatrix::new(old_perm_values, perm_width);

    let _span = tracing::info_span!("compute logup partial sums").entered();
    // Compute the running sum column
    // let mut phi = vec![EF::zero(); perm.height()];
    let mut phi = Vec::with_capacity(height);
    phi.push(EF::zero());
    for n in 0..height {
        let evaluator = Evaluator {
            preprocessed,
            partitioned_main,
            public_values,
            height,
            local_index: n,
        };
        if n > 0 {
            phi.push(phi[n - 1]);
        }
        let perm_row = perm.row_mut(n);
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
        perm_row[perm_width - 1] = phi[n];
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

pub fn standard<F, EF>(
    all_interactions: &[Interaction<SymbolicExpression<F>>],
    preprocessed: &Option<RowMajorMatrixView<F>>,
    partitioned_main: &[RowMajorMatrixView<F>],
    public_values: &[F],
    alphas: Vec<EF>,
    betas: Vec<EF>,
) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let perm_width = all_interactions.len() + 1;
    let height = partitioned_main[0].height();
    let mut reciprocals = vec![EF::zero(); height * (perm_width - 1)];
    reciprocals
        .par_chunks_mut(perm_width - 1)
        .enumerate()
        .for_each(|(i, r)| {
            let evaluator = Evaluator {
                preprocessed,
                partitioned_main,
                public_values,
                height,
                local_index: i,
            };
            for (j, interaction) in all_interactions.iter().enumerate() {
                let alpha = alphas[interaction.bus_index];
                debug_assert!(interaction.fields.len() <= betas.len());
                let mut fields = interaction.fields.iter();
                let mut rlc =
                    alpha + evaluator.eval_expr(fields.next().expect("fields should not be empty"));
                for (expr, &beta) in fields.zip(betas.iter().skip(1)) {
                    rlc += beta * evaluator.eval_expr(expr);
                }
                r[j] = rlc;
            }
        });
    // Zero should be vanishingly unlikely if alpha, beta are properly pseudo-randomized
    // The logup reciprocals should never be zero, so trace generation should panic if
    // trying to divide by zero.
    let mut old_perm_values = vec![EF::zero(); height * (perm_width - 1)];
    let chunk_size = height;
    old_perm_values
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, r)| {
            batch_multiplicative_inverse_with_buf(
                &reciprocals[i * chunk_size..(i + 1) * chunk_size],
                r,
            );
        });
    // let old_perm_values = p3_field::batch_multiplicative_inverse(&reciprocals);
    drop(reciprocals);
    // Need to add the `phi` column to perm_values as a RowMajorMatrix
    // TODO[jpw]: is there a more memory efficient way to do this?
    let mut perm_values = vec![EF::zero(); height * perm_width];
    perm_values
        .par_chunks_mut(perm_width)
        .enumerate()
        .for_each(|(i, row)| {
            let (left, _) = row.split_at_mut(perm_width - 1);
            left.copy_from_slice(&old_perm_values[i * (perm_width - 1)..(i + 1) * (perm_width - 1)])
        });
    perm_values
}

pub fn substitute_one<F, EF>(
    all_interactions: &[Interaction<SymbolicExpression<F>>],
    preprocessed: &Option<RowMajorMatrixView<F>>,
    partitioned_main: &[RowMajorMatrixView<F>],
    public_values: &[F],
    alphas: Vec<EF>,
    betas: Vec<EF>,
) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let perm_width = all_interactions.len() + 1;
    let height = partitioned_main[0].height();
    let mut reciprocals = vec![EF::one(); height * perm_width];
    reciprocals
        .par_chunks_mut(perm_width)
        .enumerate()
        .for_each(|(i, r)| {
            let evaluator = Evaluator {
                preprocessed,
                partitioned_main,
                public_values,
                height,
                local_index: i,
            };
            for (j, interaction) in all_interactions.iter().enumerate() {
                let alpha = alphas[interaction.bus_index];
                debug_assert!(interaction.fields.len() <= betas.len());
                let mut fields = interaction.fields.iter();
                let mut rlc =
                    alpha + evaluator.eval_expr(fields.next().expect("fields should not be empty"));
                for (expr, &beta) in fields.zip(betas.iter().skip(1)) {
                    rlc += beta * evaluator.eval_expr(expr);
                }
                r[j] = rlc;
            }
        });
    let perm_values = p3_field::batch_multiplicative_inverse(&reciprocals);
    drop(reciprocals);
    perm_values
}

pub fn batch_multiplicative_inverse_with_buf<F: Field>(x: &[F], buf: &mut [F]) {
    // Higher WIDTH increases instruction-level parallelism, but too high a value will cause us
    // to run out of registers.
    const WIDTH: usize = 4;
    // JN note: WIDTH is 4. The code is specialized to this value and will need
    // modification if it is changed. I tried to make it more generic, but Rust's const
    // generics are not yet good enough.

    // Handle special cases. Paradoxically, below is repetitive but concise.
    // The branches should be very predictable.
    let n = buf.len();
    if n == 0 {
        return;
    } else if n == 1 {
        buf[0] = x[0].inverse();
        return;
    } else if n == 2 {
        let x01 = x[0] * x[1];
        let x01inv = x01.inverse();
        buf[0] = x01inv * x[1];
        buf[1] = x01inv * x[0];
        return;
    } else if n == 3 {
        let x01 = x[0] * x[1];
        let x012 = x01 * x[2];
        let x012inv = x012.inverse();
        let x01inv = x012inv * x[2];
        buf[0] = x01inv * x[1];
        buf[1] = x01inv * x[0];
        buf[2] = x012inv * x01;
        return;
    }
    debug_assert!(n >= WIDTH);

    // Buf is reused for a few things to save allocations.
    // Fill buf with cumulative product of x, only taking every 4th value. Concretely, buf will
    // be [
    //   x[0], x[1], x[2], x[3],
    //   x[0] * x[4], x[1] * x[5], x[2] * x[6], x[3] * x[7],
    //   x[0] * x[4] * x[8], x[1] * x[5] * x[9], x[2] * x[6] * x[10], x[3] * x[7] * x[11],
    //   ...
    // ].
    // If n is not a multiple of WIDTH, the result is truncated from the end. For example,
    // for n == 5, we get [x[0], x[1], x[2], x[3], x[0] * x[4]].
    // let mut buf: Vec<F> = Vec::with_capacity(n);
    // cumul_prod holds the last WIDTH elements of buf. This is redundant, but it's how we
    // convince LLVM to keep the values in the registers.
    let mut cumul_prod: [F; WIDTH] = x[..WIDTH].try_into().unwrap();
    buf[0..WIDTH].copy_from_slice(&cumul_prod);
    for (i, &xi) in x[WIDTH..n].iter().enumerate() {
        cumul_prod[i % WIDTH] *= xi;
        buf[WIDTH + i] = cumul_prod[i % WIDTH];
    }
    debug_assert_eq!(buf.len(), n);

    let mut a_inv = {
        // This is where the four dependency chains meet.
        // Take the last four elements of buf and invert them all.
        let c01 = cumul_prod[0] * cumul_prod[1];
        let c23 = cumul_prod[2] * cumul_prod[3];
        let c0123 = c01 * c23;
        let c0123inv = c0123.inverse();
        let c01inv = c0123inv * c23;
        let c23inv = c0123inv * c01;
        [
            c01inv * cumul_prod[1],
            c01inv * cumul_prod[0],
            c23inv * cumul_prod[3],
            c23inv * cumul_prod[2],
        ]
    };

    for i in (WIDTH..n).rev() {
        // buf[i - WIDTH] has not been written to by this loop, so it equals
        // x[i % WIDTH] * x[i % WIDTH + WIDTH] * ... * x[i - WIDTH].
        buf[i] = buf[i - WIDTH] * a_inv[i % WIDTH];
        // buf[i] now holds the inverse of x[i].
        a_inv[i % WIDTH] *= x[i];
    }
    for i in (0..WIDTH).rev() {
        buf[i] = a_inv[i];
    }

    for (&bi, &xi) in buf.iter().zip(x) {
        // Sanity check only.
        debug_assert_eq!(bi * xi, F::one());
    }
}
