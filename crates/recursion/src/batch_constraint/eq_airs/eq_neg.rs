use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::{and, assert_array_eq, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, keygen::types::MultiStarkVerifyingKey,
    poly_common::eval_eq_uni_at_one, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
    PrimeField32, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::{
        bus::{EqNegInternalBus, EqNegInternalMessage},
        SelectorCount,
    },
    bus::{
        EqNegBaseRandBus, EqNegBaseRandMessage, EqNegResultBus, EqNegResultMessage, SelUniBus,
        SelUniBusMessage,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    tracegen::RowMajorChip,
    utils::{
        base_to_ext, ext_field_add, ext_field_add_scalar, ext_field_multiply,
        ext_field_multiply_scalar, ext_field_subtract, MultiVecWithBounds,
    },
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct EqNegCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Negative hypercube of the current section + section row index
    pub neg_hypercube: F,
    pub neg_hypercube_nz_inv: F,
    pub row_index: F,
    pub is_first_hypercube: F,
    pub is_last_hypercube: F,

    // Value of u^{2^row}, r'^{2^row}, and (r' * omega)^{2^row}, where
    // r' = r^{2^neg_hypercube}
    pub u_pow: [F; D_EF],
    pub r_pow: [F; D_EF],
    pub r_omega_pow: [F; D_EF],

    // Running product of (u^{2^i} + r'^{2^i}) from i to row_index
    pub prod_u_r: [F; D_EF],
    pub prod_u_r_omega: [F; D_EF],
    pub prod_1_r: [F; D_EF],
    pub prod_1_r_omega: [F; D_EF],
    pub sel_first_count: F,
    pub sel_last_trans_count: F,
    pub one_half_pow: F,
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct EqNegTraceGenerator;

impl RowMajorChip<F> for EqNegTraceGenerator {
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
        let width = EqNegCols::<usize>::width();

        if l_skip == 0 {
            let ret = if let Some(height) = required_height {
                // We essentially fill the trace with dummy rows as we do below, but
                // instead of proof_idx being preflights.len() it is 0
                Some(RowMajorMatrix::new(vec![F::ZERO; height * width], width))
            } else {
                Some(RowMajorMatrix::new(vec![], width))
            };
            return ret;
        }

        let total_valid = preflights.len() * (l_skip * (l_skip + 3)) / 2;

        let padded_height = if let Some(height) = required_height {
            if height < total_valid {
                return None;
            }
            height
        } else {
            total_valid.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padded_height * width];
        let mut chunks = trace.chunks_exact_mut(width);

        for (proof_idx, preflight) in preflights.iter().enumerate() {
            let initial_omega = F::two_adic_generator(vk.inner.params.l_skip);
            let initial_u = preflight.stacking.sumcheck_rnd[0];
            let mut initial_r = preflight.batch_constraint.sumcheck_rnd[0];
            let mut initial_r_omega = initial_r * initial_omega;

            for neg_hypercube in 0..l_skip {
                let mut u = initial_u;
                let mut r = initial_r;
                let mut r_omega = initial_r_omega;

                let mut prod_u_r = u * (u + r);
                let mut prod_u_r_omega = u * (u + r_omega);
                let mut prod_1_r = r + F::ONE;
                let mut prod_1_r_omega = r_omega + F::ONE;

                let one_half = F::ONE.halve();
                let mut one_half_pow = one_half;

                for row_idx in 0..=l_skip - neg_hypercube {
                    let chunk = chunks.next().unwrap();
                    let cols: &mut EqNegCols<F> = chunk.borrow_mut();
                    let is_first_hypercube = row_idx == 0;
                    let is_last_hypercube = row_idx == l_skip - neg_hypercube;

                    cols.proof_idx = F::from_usize(proof_idx);
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(neg_hypercube == 0 && is_first_hypercube);
                    cols.is_last = F::from_bool(neg_hypercube + 1 == l_skip && is_last_hypercube);

                    cols.neg_hypercube = F::from_usize(neg_hypercube);
                    cols.neg_hypercube_nz_inv =
                        cols.neg_hypercube.try_inverse().unwrap_or_default();
                    cols.row_index = F::from_usize(row_idx);
                    cols.is_first_hypercube = F::from_bool(is_first_hypercube);
                    cols.is_last_hypercube = F::from_bool(is_last_hypercube);

                    cols.u_pow.copy_from_slice(u.as_basis_coefficients_slice());
                    cols.r_pow.copy_from_slice(r.as_basis_coefficients_slice());
                    cols.r_omega_pow
                        .copy_from_slice(r_omega.as_basis_coefficients_slice());
                    u *= u;
                    r *= r;
                    r_omega *= r_omega;

                    cols.prod_u_r
                        .copy_from_slice(prod_u_r.as_basis_coefficients_slice());
                    cols.prod_u_r_omega
                        .copy_from_slice(prod_u_r_omega.as_basis_coefficients_slice());
                    prod_u_r *= u + r;
                    prod_u_r_omega *= u + r_omega;

                    cols.prod_1_r
                        .copy_from_slice(prod_1_r.as_basis_coefficients_slice());
                    cols.prod_1_r_omega
                        .copy_from_slice(prod_1_r_omega.as_basis_coefficients_slice());
                    cols.one_half_pow = one_half_pow;
                    debug_assert_eq!(
                        prod_1_r * one_half_pow,
                        eval_eq_uni_at_one(row_idx + 1, initial_r)
                    );
                    debug_assert_eq!(
                        prod_1_r_omega * one_half_pow,
                        eval_eq_uni_at_one(row_idx + 1, initial_r_omega)
                    );
                    prod_1_r *= r + F::ONE;
                    prod_1_r_omega *= r_omega + F::ONE;
                    one_half_pow *= one_half;

                    // We only use the count for the last row of `hypercube` or the very first row.
                    // We use the very first row to hard-code the lookup for log_height = 0 since
                    // it's disjoint from the other condition.
                    if neg_hypercube == 0 && row_idx == 0 {
                        let counts = selector_counts[[proof_idx]][0];
                        cols.sel_first_count = F::from_usize(counts.first);
                        cols.sel_last_trans_count = F::from_usize(counts.last + counts.transition);
                    } else if row_idx == l_skip - neg_hypercube {
                        let mut counts = selector_counts[[proof_idx]][row_idx];
                        // On `neg_hypercube`, we collect counts for all heights >= l_skip.
                        if neg_hypercube == 0 {
                            for count in &selector_counts[[proof_idx]][l_skip - neg_hypercube + 1..]
                            {
                                counts.first += count.first;
                                counts.last += count.last;
                                counts.transition += count.transition;
                            }
                        }
                        cols.sel_first_count = F::from_usize(counts.first);
                        cols.sel_last_trans_count = F::from_usize(counts.last + counts.transition);
                    }
                }

                initial_r *= initial_r;
                initial_r_omega *= initial_r_omega;
            }
        }

        for chunk in chunks {
            let cols: &mut EqNegCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_usize(preflights.len());
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct EqNegAir {
    pub result_bus: EqNegResultBus,
    pub base_rand_bus: EqNegBaseRandBus,
    pub internal_bus: EqNegInternalBus,
    pub sel_uni_bus: SelUniBus,
    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqNegAir {}
impl<F> PartitionedBaseAir<F> for EqNegAir {}
impl<F> BaseAir<F> for EqNegAir {
    fn width(&self) -> usize {
        EqNegCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqNegAir
where
    AB::F: PrimeField32 + TwoAdicField,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &EqNegCols<AB::Var> = (*local).borrow();
        let next: &EqNegCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<3, 2> {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid.into(),
                        counter: [
                            local.proof_idx.into(),
                            local.neg_hypercube.into(),
                            local.row_index.into(),
                        ],
                        is_first: [
                            local.is_first.into(),
                            local.is_first_hypercube.into(),
                            local.is_valid.into(),
                        ],
                    },
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid.into(),
                        counter: [
                            next.proof_idx.into(),
                            next.neg_hypercube.into(),
                            next.row_index.into(),
                        ],
                        is_first: [
                            next.is_first.into(),
                            next.is_first_hypercube.into(),
                            next.is_valid.into(),
                        ],
                    },
                ),
                NestedForLoopAuxCols {
                    is_transition: [
                        local.is_valid - local.is_last,
                        local.is_valid - local.is_last_hypercube,
                    ],
                },
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(local.is_first_hypercube);
        builder.assert_bool(local.is_last_hypercube);
        builder.when(local.is_first).assert_one(local.is_valid);
        builder.when(local.is_last).assert_one(local.is_valid);
        builder
            .when(local.is_first_hypercube)
            .assert_one(local.is_valid);
        builder
            .when(local.is_last_hypercube)
            .assert_one(local.is_valid);

        /*
         * Constrain that neg_hypercube dimension starts at 0 and increments, and
         * that row_idx increments by 1 in each neg_hypercube section. Additionally,
         * each section should have exactly l_skip - neg_hypercube + 1 rows.
         */
        builder
            .when(local.is_first)
            .assert_zero(local.neg_hypercube);
        builder
            .when(local.is_last)
            .assert_eq(local.neg_hypercube, AB::F::from_usize(self.l_skip - 1));
        builder
            .when(local.is_first_hypercube)
            .assert_zero(local.row_index);

        builder.when(local.is_last_hypercube).assert_eq(
            local.row_index,
            AB::Expr::from_usize(self.l_skip) - local.neg_hypercube,
        );

        builder
            .when(local.is_valid - local.is_last_hypercube)
            .assert_one(next.row_index - local.row_index);
        builder
            .when(local.is_valid - local.is_last_hypercube)
            .assert_eq(next.neg_hypercube, local.neg_hypercube);

        /*
         * Receive u_0 and r_0 from the AIRs that sample them on the first row. Also,
         * send (u, r^2, (r * omega)^2) on the first row of non-last hypercubes and
         * receive (u, r, r * omega) on the first row of non-first hypercubes.
         */
        let initial_omega = AB::F::two_adic_generator(self.l_skip);
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.r_omega_pow,
            ext_field_multiply_scalar(local.r_pow, initial_omega),
        );

        self.base_rand_bus.receive(
            builder,
            local.proof_idx,
            EqNegBaseRandMessage {
                u: local.u_pow.map(Into::into),
                r_squared: ext_field_multiply::<AB::Expr>(local.r_pow, local.r_pow),
            },
            local.is_first,
        );

        self.internal_bus.send(
            builder,
            local.proof_idx,
            EqNegInternalMessage {
                neg_n: local.neg_hypercube + AB::F::ONE,
                u: local.u_pow.map(Into::into),
                r: ext_field_multiply(local.r_pow, local.r_pow),
                r_omega: ext_field_multiply(local.r_omega_pow, local.r_omega_pow),
            },
            and(local.is_first_hypercube, not(next.is_last)),
        );

        self.internal_bus.receive(
            builder,
            local.proof_idx,
            EqNegInternalMessage {
                neg_n: local.neg_hypercube,
                u: local.u_pow,
                r: local.r_pow,
                r_omega: local.r_omega_pow,
            },
            and(local.is_first_hypercube, not(local.is_first)),
        );

        /*
         * Constrain the running product of (u^{2^i} + r'^{2^i})
         */
        assert_array_eq(
            &mut builder.when(local.is_first_hypercube),
            local.prod_u_r,
            ext_field_multiply(local.u_pow, ext_field_add(local.u_pow, local.r_pow)),
        );

        assert_array_eq(
            &mut builder.when(local.is_first_hypercube),
            local.prod_1_r,
            ext_field_add(local.r_pow, base_to_ext::<AB::Expr>(AB::F::ONE)),
        );

        assert_array_eq(
            &mut builder.when(local.is_valid - local.is_last_hypercube),
            ext_field_multiply(local.u_pow, local.u_pow),
            next.u_pow,
        );

        assert_array_eq(
            &mut builder.when(local.is_valid - local.is_last_hypercube),
            ext_field_multiply(local.r_pow, local.r_pow),
            next.r_pow,
        );

        assert_array_eq(
            &mut builder.when(local.is_valid - local.is_last_hypercube),
            ext_field_multiply(local.prod_u_r, ext_field_add(next.u_pow, next.r_pow)),
            next.prod_u_r,
        );

        assert_array_eq(
            &mut builder.when(local.is_valid - local.is_last_hypercube),
            ext_field_multiply(
                local.prod_1_r,
                ext_field_add(next.r_pow, base_to_ext::<AB::Expr>(AB::F::ONE)),
            ),
            next.prod_1_r,
        );

        /*
         * Constrain the running product that is used to compute eq_0(u, r * omega).
         */
        assert_array_eq(
            &mut builder.when(local.is_first_hypercube),
            local.prod_u_r_omega,
            ext_field_multiply(local.u_pow, ext_field_add(local.u_pow, local.r_omega_pow)),
        );

        assert_array_eq(
            &mut builder.when(local.is_valid - local.is_last_hypercube),
            ext_field_multiply(local.r_omega_pow, local.r_omega_pow),
            next.r_omega_pow,
        );

        assert_array_eq(
            &mut builder.when(local.is_valid - local.is_last_hypercube),
            ext_field_multiply(
                local.prod_u_r_omega,
                ext_field_add(next.u_pow, next.r_omega_pow),
            ),
            next.prod_u_r_omega,
        );

        self.sel_uni_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            SelUniBusMessage {
                n: -local.neg_hypercube.into(),
                is_first: AB::Expr::ONE,
                value: local.prod_1_r.map(|x| x * local.one_half_pow),
            },
            next.is_last_hypercube * next.sel_first_count,
        );
        self.sel_uni_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            SelUniBusMessage {
                n: -local.neg_hypercube.into(),
                is_first: AB::Expr::ZERO,
                value: local.prod_1_r_omega.map(|x| x * local.one_half_pow),
            },
            next.is_last_hypercube * next.sel_last_trans_count,
        );

        // This is kind of ugly. But we use the first row as the lookup table for
        // selector for log_height=0.
        self.sel_uni_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            SelUniBusMessage {
                n: -AB::Expr::from_usize(self.l_skip),
                is_first: AB::Expr::ONE,
                value: [
                    AB::Expr::ONE,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
            },
            local.is_first * local.sel_first_count,
        );
        self.sel_uni_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            SelUniBusMessage {
                n: -AB::Expr::from_usize(self.l_skip),
                is_first: AB::Expr::ZERO,
                value: [
                    AB::Expr::ONE,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
            },
            local.is_first * local.sel_last_trans_count,
        );

        /*
         * Compute eq_n(u, r) and k_rot_n(u, r), and send them to EqBaseAir (without
         * the omega_pow_inv).
         */
        let eq = ext_field_add_scalar::<AB::Expr>(
            ext_field_subtract(local.prod_u_r, next.u_pow),
            AB::F::ONE,
        );
        let k_rot = ext_field_add_scalar::<AB::Expr>(
            ext_field_subtract(local.prod_u_r_omega, next.u_pow),
            AB::F::ONE,
        );

        let is_neg = local.neg_hypercube * local.neg_hypercube_nz_inv;
        builder.when(local.neg_hypercube).assert_one(is_neg.clone());
        self.result_bus.send(
            builder,
            local.proof_idx,
            EqNegResultMessage {
                n: AB::Expr::ZERO - local.neg_hypercube,
                eq,
                k_rot,
            },
            is_neg * next.is_last_hypercube,
        );
    }
}
