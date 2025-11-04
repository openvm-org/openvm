use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_circuit_primitives::{SubAir, utils::not};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    FieldAlgebra, FieldExtensionAlgebra, TwoAdicField, extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::{eval_eq_sharp_uni, eval_eq_uni},
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, EqZeroNBus, EqZeroNMessage,
    },
    bus::{XiRandomnessBus, XiRandomnessMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{
        assert_eq_array, base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_one_minus, ext_field_subtract,
    },
};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct EqNsRecord {
    xi: EF,
    r: EF,
    r_prod: EF,
    eq: EF,
    eq_sharp: EF,
    num_traces: usize,
    n_logup: usize,
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqNsColumns<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    n: T,
    n_less_than_n_logup: T,
    xi_n: [T; D_EF],
    r_n: [T; D_EF],
    r_product: [T; D_EF],
    eq: [T; D_EF],
    eq_sharp: [T; D_EF],
    /// The number of traces with such `n`
    num_traces: T,
}

pub struct EqNsAir {
    pub zero_n_bus: EqZeroNBus,
    pub xi_bus: XiRandomnessBus,
    pub r_xi_bus: BatchConstraintConductorBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqNsAir {}
impl<F> PartitionedBaseAir<F> for EqNsAir {}

impl<F> BaseAir<F> for EqNsAir {
    fn width(&self) -> usize {
        EqNsColumns::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqNsAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqNsColumns<AB::Var> = (*local).borrow();
        let next: &EqNsColumns<AB::Var> = (*next).borrow();

        // Summary:
        // - n consistency: treat `n_less_than_n_logup` as boolean, set `n = 0` on the first row,
        //   increment `n` while continuing within the proof, propagate the “less than n_logup” flag
        //   forward, and clear it whenever a row is invalid.
        // - r consistency: initialize `r_product` to one when the next row begins the proof and,
        //   otherwise, multiply by the current `r_n` values to update the running product.
        // - eq consistency: update both `eq` and `eq_sharp` by multiplying with the shared factor
        //   `1 - (xi + r - 2 * xi * r)` whenever advancing beyond the first row.

        type LoopSubAir = NestedForLoopSubAir<1, 0>;
        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);

        // ========================= n consistency ==============================
        builder.assert_bool(local.n_less_than_n_logup);
        builder.when(local.is_first).assert_zero(local.n);
        builder
            .when(not(next.is_first))
            .assert_one(next.n - local.n);
        builder
            .when(not(next.is_first))
            .when(next.n_less_than_n_logup)
            .assert_one(local.n_less_than_n_logup);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.n_less_than_n_logup);
        // ========================= r consistency ==============================
        assert_eq_array(
            &mut builder.when(local.is_valid * next.is_first),
            local.r_product,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_eq_array(
            &mut builder.when(not(next.is_first)),
            local.r_product,
            ext_field_multiply(next.r_product, local.r_n),
        );
        // ========================= eq consistency ===============================
        let mult = ext_field_one_minus::<AB::Expr>(ext_field_subtract::<AB::Expr>(
            ext_field_add(local.xi_n, local.r_n),
            ext_field_multiply_scalar::<AB::Expr>(
                ext_field_multiply(local.xi_n, local.r_n),
                AB::Expr::TWO,
            ),
        ));
        assert_eq_array(
            &mut builder.when(not(next.is_first)),
            next.eq,
            ext_field_multiply(local.eq, mult.clone()),
        );
        assert_eq_array(
            &mut builder.when(not(next.is_first)),
            next.eq_sharp,
            ext_field_multiply(local.eq_sharp, mult.clone()),
        );

        self.xi_bus.receive(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.n + AB::Expr::from_canonical_usize(self.l_skip),
                xi: local.xi_n.map(|x| x.into()),
            },
            local.is_valid * (AB::Expr::ONE - local.is_last),
        );
        self.r_xi_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: local.n + AB::Expr::from_canonical_usize(self.l_skip),
                value: local.xi_n.map(|x| x.into()),
            },
            local.n_less_than_n_logup,
        );

        self.zero_n_bus.receive(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ZERO,
                value: local.eq.map(|x| x.into()),
            },
            local.is_valid * local.is_first,
        );
        self.zero_n_bus.receive(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ONE,
                value: local.eq_sharp.map(|x| x.into()),
            },
            local.is_valid * local.is_first,
        );

        self.r_xi_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: local.n + AB::Expr::ONE,
                value: local.r_n.map(|x| x.into()),
            },
            local.is_valid * (AB::Expr::ONE - local.is_last),
        );
    }
}

pub(crate) fn generate_eq_ns_trace(
    vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let l_skip = vk.inner.params.l_skip;
    let omega_skip_pows = F::two_adic_generator(l_skip)
        .powers()
        .take(1 << l_skip)
        .collect_vec();
    // TODO: blob, MultiProofVecVec etc
    let records = preflights
        .iter()
        .map(|preflight| {
            let n_global = preflight.proof_shape.n_global();
            let rs = &preflight.batch_constraint.sumcheck_rnd;
            let xi = &preflight.batch_constraint.xi;
            let mut res = Vec::with_capacity(n_global + 1);
            let mut eq = eval_eq_uni(l_skip, xi[0], rs[0]);
            let mut eq_sharp = eval_eq_sharp_uni(&omega_skip_pows, &xi[..l_skip], rs[0]);
            for i in 0..n_global {
                res.push(EqNsRecord {
                    xi: xi[l_skip + i],
                    r: rs[1 + i],
                    r_prod: EF::ONE,
                    eq,
                    eq_sharp,
                    num_traces: 0,
                    n_logup: preflight.proof_shape.n_logup,
                });
                let mult =
                    EF::ONE - xi[l_skip + i] - rs[1 + i] + (xi[l_skip + i] * rs[1 + i]).double();
                eq *= mult;
                eq_sharp *= mult;
            }
            res.push(EqNsRecord {
                xi: EF::ZERO,
                r: EF::ZERO,
                r_prod: EF::ONE,
                eq,
                eq_sharp,
                num_traces: 0,
                n_logup: preflight.proof_shape.n_logup,
            });
            for i in (0..n_global).rev() {
                res[i].r_prod = res[i + 1].r_prod * res[i].r;
            }
            for (_, vdata) in preflight.proof_shape.sorted_trace_vdata.iter() {
                res[vdata.hypercube_dim].num_traces += 1;
            }
            res
        })
        .collect::<Vec<_>>();

    let width = EqNsColumns::<F>::width();
    let total_height = records.iter().map(|rows| rows.len()).sum::<usize>();
    let padded_height = total_height.next_power_of_two();

    let mut trace = vec![F::ZERO; padded_height * width];
    let mut cur_height = 0;
    for (pidx, rows) in records.iter().enumerate() {
        trace[cur_height * width..(cur_height + rows.len()) * width]
            .par_chunks_exact_mut(width)
            .zip(rows.par_iter())
            .enumerate()
            .for_each(|(i, (chunk, record))| {
                let cols: &mut EqNsColumns<_> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(i == 0);
                cols.is_last = F::from_bool(i + 1 == rows.len());
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.n = F::from_canonical_usize(i);
                cols.n_less_than_n_logup = F::from_bool(i < record.n_logup);
                cols.xi_n.copy_from_slice(record.xi.as_base_slice());
                cols.r_n.copy_from_slice(record.r.as_base_slice());
                cols.r_product
                    .copy_from_slice(record.r_prod.as_base_slice());
                cols.eq.copy_from_slice(record.eq.as_base_slice());
                cols.eq_sharp
                    .copy_from_slice(record.eq_sharp.as_base_slice());
                cols.num_traces = F::from_canonical_usize(record.num_traces);
            });
        cur_height += rows.len();
    }
    trace[total_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut EqNsColumns<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            cols.is_first = F::ONE;
            cols.is_last = F::ONE;
        });

    RowMajorMatrix::new(trace, width)
}
