use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, keygen::types::MultiStarkVerifyingKey,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::{
        bus::{
            BatchConstraintConductorBus, BatchConstraintConductorMessage,
            BatchConstraintInnerMessageType, EqNOuterBus, EqNOuterMessage, EqZeroNBus,
            EqZeroNMessage,
        },
        SelectorCount,
    },
    bus::{SelHypercubeBus, SelHypercubeBusMessage, XiRandomnessBus, XiRandomnessMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    tracegen::RowMajorChip,
    utils::{
        base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_one_minus, ext_field_subtract, MultiVecWithBounds,
    },
};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct EqNsRecord {
    xi: EF,
    r: EF,
    eq_r_ones: EF,
    eq_r_zeroes: EF,
    r_prod: EF,
    eq: EF,
    eq_sharp: EF,
    sel_first_count: usize,
    sel_last_and_trans_count: usize,
    n_logup: usize,
    n_max: usize,
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqNsColumns<T> {
    is_valid: T,
    is_first: T,
    proof_idx: T,

    n: T,
    n_less_than_n_logup: T,
    n_less_than_n_max: T,
    is_transition_and_n_less_than_n_max: T,
    xi_n: [T; D_EF],
    r_n: [T; D_EF],
    r_product: [T; D_EF],
    r_pref_product: [T; D_EF],
    one_minus_r_pref_prod: [T; D_EF],
    eq: [T; D_EF],
    eq_sharp: [T; D_EF],

    /// The number of traces whose `n_lift` equals `local.n`.
    /// Note that it cannot be derived from `xi_mult` because
    /// `xi_mult` counts interactions, not AIRs with interactions.
    num_traces: T,
    xi_mult: T,
    sel_first_count: T,
    sel_last_and_trans_count: T,
}

pub struct EqNsAir {
    pub zero_n_bus: EqZeroNBus,
    pub xi_bus: XiRandomnessBus,
    pub r_xi_bus: BatchConstraintConductorBus,
    pub sel_hypercube_bus: SelHypercubeBus,
    pub eq_n_outer_bus: EqNOuterBus,

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
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

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
        builder.when(local.is_first).assert_one(local.is_valid);

        let is_transition = next.is_valid - next.is_first;
        let is_last = local.is_valid - is_transition.clone();

        // ========================= n consistency ==============================
        builder.assert_bool(local.n_less_than_n_max);
        builder.assert_bool(local.n_less_than_n_logup);
        builder.when(local.is_first).assert_zero(local.n);
        builder
            .when(is_transition.clone())
            .assert_one(next.n - local.n);
        builder
            .when(is_transition.clone())
            .when(next.n_less_than_n_logup)
            .assert_one(local.n_less_than_n_logup);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.n_less_than_n_logup);
        builder
            .when(is_transition.clone())
            .when(next.n_less_than_n_max)
            .assert_one(local.n_less_than_n_max);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.n_less_than_n_max);
        // ========================= r consistency ==============================
        assert_array_eq(
            &mut builder.when(local.is_valid - local.n_less_than_n_max),
            local.r_n,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_array_eq(
            &mut builder.when(is_last.clone()),
            local.r_product,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_array_eq(
            &mut builder.when(is_transition.clone()),
            local.r_product,
            ext_field_multiply(next.r_product, local.r_n),
        );
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.r_pref_product,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_array_eq(
            &mut builder.when(is_transition.clone() * local.n_less_than_n_max),
            next.r_pref_product,
            ext_field_multiply(local.r_pref_product, local.r_n),
        );
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.one_minus_r_pref_prod,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_array_eq(
            &mut builder.when(is_transition.clone() * local.n_less_than_n_max),
            next.one_minus_r_pref_prod,
            ext_field_multiply(
                local.one_minus_r_pref_prod,
                ext_field_subtract(base_to_ext::<AB::Expr>(AB::F::ONE), local.r_n),
            ),
        );
        self.sel_hypercube_bus.send(
            builder,
            local.proof_idx,
            SelHypercubeBusMessage {
                n: local.n.into(),
                is_first: AB::Expr::ZERO,
                value: local.r_pref_product.map(Into::into),
            },
            local.is_valid * local.sel_last_and_trans_count,
        );
        self.sel_hypercube_bus.send(
            builder,
            local.proof_idx,
            SelHypercubeBusMessage {
                n: local.n.into(),
                is_first: AB::Expr::ONE,
                value: local.one_minus_r_pref_prod.map(Into::into),
            },
            local.is_valid * local.sel_first_count,
        );
        // ========================= eq consistency ===============================
        let mult = ext_field_one_minus::<AB::Expr>(ext_field_subtract::<AB::Expr>(
            ext_field_add(local.xi_n, local.r_n),
            ext_field_multiply_scalar::<AB::Expr>(
                ext_field_multiply(local.xi_n, local.r_n),
                AB::Expr::TWO,
            ),
        ));
        builder.assert_eq(
            local.is_transition_and_n_less_than_n_max,
            is_transition.clone() * local.n_less_than_n_max,
        );
        assert_array_eq(
            &mut builder.when(local.is_transition_and_n_less_than_n_max),
            next.eq,
            ext_field_multiply(local.eq, mult.clone()),
        );
        assert_array_eq(
            &mut builder.when(local.is_transition_and_n_less_than_n_max),
            next.eq_sharp,
            ext_field_multiply(local.eq_sharp, mult.clone()),
        );

        self.xi_bus.receive(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.n + AB::Expr::from_usize(self.l_skip),
                xi: local.xi_n.map(|x| x.into()),
            },
            is_transition.clone(),
        );
        // Here idx >= l_skip and all idx are different within one proof_idx
        self.r_xi_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: local.n + AB::Expr::from_usize(self.l_skip),
                value: local.xi_n.map(|x| x.into()),
            },
            local.n_less_than_n_logup * local.xi_mult,
        );

        self.zero_n_bus.receive(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ZERO,
                value: local.eq.map(|x| x.into()),
            },
            local.is_first,
        );
        self.zero_n_bus.receive(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ONE,
                value: local.eq_sharp.map(|x| x.into()),
            },
            local.is_first,
        );

        self.r_xi_bus.lookup_key(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: local.n + AB::Expr::ONE,
                value: local.r_n.map(|x| x.into()),
            },
            local.n_less_than_n_max,
        );

        self.eq_n_outer_bus.add_key_with_lookups(
            builder,
            next.proof_idx,
            EqNOuterMessage {
                is_sharp: AB::Expr::ZERO,
                n: next.n.into(),
                value: ext_field_multiply(next.eq, next.r_product),
            },
            next.is_valid * next.num_traces,
        );
        self.eq_n_outer_bus.add_key_with_lookups(
            builder,
            next.proof_idx,
            EqNOuterMessage {
                is_sharp: AB::Expr::ONE,
                n: next.n.into(),
                value: ext_field_multiply(next.eq_sharp, next.r_product),
            },
            next.is_valid * next.num_traces * AB::Expr::TWO, // two because num+denom per trace
        );
    }
}

pub struct EqNsTraceGenerator;

impl RowMajorChip<F> for EqNsTraceGenerator {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a [&'a Preflight],
        &'a MultiVecWithBounds<SelectorCount, 1>,
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (vk, preflights, selector_counts) = ctx;
        let l_skip = vk.inner.params.l_skip;
        let records = preflights
            .iter()
            .enumerate()
            .map(|(pidx, preflight)| {
                let selector_counts = &selector_counts[[pidx]];
                let n_global = preflight.proof_shape.n_global();
                let n_max = preflight.proof_shape.n_max;
                let rs = &preflight.batch_constraint.sumcheck_rnd;
                let xi = &preflight.batch_constraint.xi;
                let mut res = Vec::with_capacity(n_global + 1);
                let mut eq_r_ones = EF::ONE;
                let mut eq_r_zeroes = EF::ONE;
                for i in 0..n_max {
                    let counts = selector_counts[i + l_skip];
                    res.push(EqNsRecord {
                        xi: xi[l_skip + i],
                        r: rs[1 + i],
                        r_prod: EF::ONE,
                        eq: preflight.batch_constraint.eq_ns[i],
                        eq_sharp: preflight.batch_constraint.eq_sharp_ns[i],
                        eq_r_ones,
                        eq_r_zeroes,
                        n_logup: preflight.proof_shape.n_logup,
                        n_max: preflight.proof_shape.n_max,
                        sel_first_count: counts.first,
                        sel_last_and_trans_count: counts.last + counts.transition,
                    });
                    eq_r_ones *= rs[1 + i];
                    eq_r_zeroes *= EF::ONE - rs[1 + i];
                }
                let counts = selector_counts[l_skip + n_max];
                for i in n_max..=n_global {
                    let (sel_first_count, sel_last_and_trans_count) = if i == n_max {
                        (counts.first, counts.last + counts.transition)
                    } else {
                        (0, 0)
                    };
                    let xi = if i == n_global {
                        EF::ZERO
                    } else {
                        xi[l_skip + i]
                    };
                    res.push(EqNsRecord {
                        xi,
                        r: EF::ONE,
                        r_prod: EF::ONE,
                        eq: preflight.batch_constraint.eq_ns[n_max],
                        eq_sharp: preflight.batch_constraint.eq_sharp_ns[n_max],
                        eq_r_ones,
                        eq_r_zeroes,
                        n_logup: preflight.proof_shape.n_logup,
                        n_max: preflight.proof_shape.n_max,
                        sel_first_count,
                        sel_last_and_trans_count,
                    });
                }
                for i in (0..n_global).rev() {
                    res[i].r_prod = res[i + 1].r_prod * res[i].r;
                }
                res
            })
            .collect::<Vec<_>>();

        let width = EqNsColumns::<F>::width();
        let total_height = records.iter().map(|rows| rows.len()).sum::<usize>();
        let padded_height = if let Some(height) = required_height {
            if height < total_height {
                return None;
            }
            height
        } else {
            total_height.next_power_of_two()
        };

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
                    cols.proof_idx = F::from_usize(pidx);
                    cols.n = F::from_usize(i);
                    cols.n_less_than_n_logup = F::from_bool(i < record.n_logup);
                    cols.n_less_than_n_max = F::from_bool(i < record.n_max);
                    cols.is_transition_and_n_less_than_n_max =
                        F::from_bool(i + 1 < rows.len() && i < record.n_max);
                    cols.xi_n
                        .copy_from_slice(record.xi.as_basis_coefficients_slice());
                    cols.r_n
                        .copy_from_slice(record.r.as_basis_coefficients_slice());
                    cols.r_product
                        .copy_from_slice(record.r_prod.as_basis_coefficients_slice());
                    cols.r_pref_product
                        .copy_from_slice(record.eq_r_ones.as_basis_coefficients_slice());
                    cols.one_minus_r_pref_prod
                        .copy_from_slice(record.eq_r_zeroes.as_basis_coefficients_slice());
                    cols.eq
                        .copy_from_slice(record.eq.as_basis_coefficients_slice());
                    cols.eq_sharp
                        .copy_from_slice(record.eq_sharp.as_basis_coefficients_slice());
                    cols.sel_first_count = F::from_usize(record.sel_first_count);
                    cols.sel_last_and_trans_count = F::from_usize(record.sel_last_and_trans_count);
                });
            let mut num_n_lift_int = vec![0; preflights[pidx].proof_shape.n_logup + 1];
            let mut num_n_lift_con = vec![0; preflights[pidx].proof_shape.n_max + 1];
            for (air_idx, vdata) in preflights[pidx].proof_shape.sorted_trace_vdata.iter() {
                let num_interactions = vk.inner.per_air[*air_idx].num_interactions();
                let n_lift = vdata.log_height.saturating_sub(l_skip);
                num_n_lift_con[n_lift] += 1;
                if num_interactions > 0 {
                    num_n_lift_int[n_lift] += num_interactions;
                }
            }
            let mut xi_mult = 0;
            for (chunk, cnt) in trace[cur_height * width..]
                .chunks_mut(width)
                .zip(num_n_lift_int.into_iter())
            {
                xi_mult += cnt;
                let cols: &mut EqNsColumns<_> = chunk.borrow_mut();
                cols.xi_mult = F::from_usize(xi_mult);
            }
            for (chunk, cnt) in trace[cur_height * width..]
                .chunks_mut(width)
                .zip(num_n_lift_con.into_iter())
            {
                let cols: &mut EqNsColumns<_> = chunk.borrow_mut();
                cols.num_traces = F::from_usize(cnt);
            }
            cur_height += rows.len();
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}
