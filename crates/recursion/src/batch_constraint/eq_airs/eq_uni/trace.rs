use std::borrow::BorrowMut;

use openvm_stark_sdk::config::baby_bear_poseidon2::{EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    batch_constraint::eq_airs::eq_uni::air::EqUniCols,
    tracegen::{RowMajorChip, StandardTracegenCtx},
};

pub struct EqUniTraceGenerator;

impl RowMajorChip<F> for EqUniTraceGenerator {
    type Ctx<'a> = StandardTracegenCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let vk = ctx.vk;
        let preflights = ctx.preflights;
        let width = EqUniCols::<F>::width();
        let l_skip = vk.inner.params.l_skip;
        let one_height = l_skip + 1;
        let total_height = one_height * preflights.len();
        let padding_height = if let Some(height) = required_height {
            if height < total_height {
                return None;
            }
            height
        } else {
            total_height.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padding_height * width];

        for (pidx, preflight) in preflights.iter().enumerate() {
            let mut x = preflight.batch_constraint.xi[0];
            let mut y = preflight.batch_constraint.sumcheck_rnd[0];
            let mut res = EF::ONE;
            trace[pidx * one_height * width..(pidx + 1) * one_height * width]
                .chunks_exact_mut(width)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let cols: &mut EqUniCols<_> = chunk.borrow_mut();
                    cols.is_valid = F::ONE;
                    cols.proof_idx = F::from_usize(pidx);
                    cols.is_first = F::from_bool(i == 0);

                    cols.idx = F::from_usize(i);
                    cols.x.copy_from_slice(x.as_basis_coefficients_slice());
                    cols.y.copy_from_slice(y.as_basis_coefficients_slice());
                    cols.res.copy_from_slice(res.as_basis_coefficients_slice());

                    res = (x + y) * res + (EF::ONE - x) * (EF::ONE - y);
                    x *= x;
                    y *= y;
                });
        }

        trace[total_height * width..]
            .chunks_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut EqUniCols<F> = chunk.borrow_mut();
                cols.proof_idx = F::from_usize(preflights.len() + i);
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
