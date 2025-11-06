use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    SubAir,
    utils::{and, assert_array_eq, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField,
    extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    stacking::bus::{
        EqNegBaseRandBus, EqNegBaseRandMessage, EqNegInternalBus, EqNegInternalMessage,
        EqNegResultBus, EqNegResultMessage,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{
        ext_field_add, ext_field_add_scalar, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_subtract,
    },
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct EqNegCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Negative hypercube of the current section + section row index
    pub neg_hypercube: F,
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
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct EqNegTraceGenerator;

impl EqNegTraceGenerator {
    pub fn generate_trace(
        vk: &MultiStarkVerifyingKeyV2,
        preflights: &[Preflight],
    ) -> RowMajorMatrix<F> {
        let l_skip = vk.inner.params.l_skip;
        let width = EqNegCols::<usize>::width();
        if l_skip == 0 {
            return RowMajorMatrix::new(vec![], width);
        }
        let height = (l_skip * (l_skip + 1)) / 2 - 1;

        let mut trace = vec![F::ZERO; (preflights.len() * height).next_power_of_two() * width];
        let mut chunks = trace.chunks_exact_mut(width);

        for (proof_idx, preflight) in preflights.iter().enumerate() {
            let initial_omega = F::two_adic_generator(vk.inner.params.l_skip - 1);
            let initial_u = preflight.stacking.sumcheck_rnd[0];
            let mut initial_r = preflight.batch_constraint.sumcheck_rnd[0].square();
            let mut initial_r_omega = initial_r * initial_omega;

            for neg_hypercube in 1..l_skip {
                let mut u = initial_u;
                let mut r = initial_r;
                let mut r_omega = initial_r_omega;

                let mut prod_u_r = u * (u + r);
                let mut prod_u_r_omega = u * (u + r_omega);

                for row_idx in 0..=l_skip - neg_hypercube {
                    let chunk = chunks.next().unwrap();
                    let cols: &mut EqNegCols<F> = chunk.borrow_mut();
                    let is_first_hypercube = row_idx == 0;
                    let is_last_hypercube = row_idx == l_skip - neg_hypercube;

                    cols.proof_idx = F::from_canonical_usize(proof_idx);
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(neg_hypercube == 1 && is_first_hypercube);
                    cols.is_last = F::from_bool(neg_hypercube + 1 == l_skip && is_last_hypercube);

                    cols.neg_hypercube = F::from_canonical_usize(neg_hypercube);
                    cols.row_index = F::from_canonical_usize(row_idx);
                    cols.is_first_hypercube = F::from_bool(is_first_hypercube);
                    cols.is_last_hypercube = F::from_bool(is_last_hypercube);

                    cols.u_pow.copy_from_slice(u.as_base_slice());
                    cols.r_pow.copy_from_slice(r.as_base_slice());
                    cols.r_omega_pow.copy_from_slice(r_omega.as_base_slice());
                    u *= u;
                    r *= r;
                    r_omega *= r_omega;

                    cols.prod_u_r.copy_from_slice(prod_u_r.as_base_slice());
                    cols.prod_u_r_omega
                        .copy_from_slice(prod_u_r_omega.as_base_slice());
                    prod_u_r *= u + r;
                    prod_u_r_omega *= u + r_omega;
                }

                initial_r *= initial_r;
                initial_r_omega *= initial_r_omega;
            }
        }

        for chunk in chunks {
            let cols: &mut EqNegCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len());
        }

        RowMajorMatrix::new(trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct EqNegAir {
    pub result_bus: EqNegResultBus,
    pub base_rand_bus: EqNegBaseRandBus,
    pub internal_bus: EqNegInternalBus,
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
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

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
                            local.neg_hypercube - AB::F::ONE,
                            local.row_index.into(),
                        ],
                        is_first: [
                            local.is_first.into(),
                            local.is_first_hypercube.into(),
                            AB::Expr::ONE,
                        ],
                    },
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid.into(),
                        counter: [
                            next.proof_idx.into(),
                            next.neg_hypercube - AB::F::ONE,
                            next.row_index.into(),
                        ],
                        is_first: [
                            next.is_first.into(),
                            next.is_first_hypercube.into(),
                            AB::Expr::ONE,
                        ],
                    },
                ),
                NestedForLoopAuxCols {
                    is_transition: [
                        local.is_valid * (AB::Expr::ONE - local.is_last),
                        local.is_valid * (AB::Expr::ONE - local.is_last_hypercube),
                    ],
                },
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(local.is_first_hypercube);
        builder.assert_bool(local.is_last_hypercube);

        /*
         * Constrain that neg_hypercube dimension starts at 1 and increments, and
         * that row_idx increments by 1 in each neg_hypercube section. Additionally,
         * each section should have exactly l_skip - neg_hypercube + 1 rows.
         */
        builder.when(local.is_first).assert_one(local.neg_hypercube);
        builder.when(local.is_last).assert_eq(
            local.neg_hypercube,
            AB::F::from_canonical_usize(self.l_skip - 1),
        );

        builder.when(local.is_last_hypercube).assert_eq(
            local.row_index,
            AB::Expr::from_canonical_usize(self.l_skip) - local.neg_hypercube,
        );

        builder
            .when(and(local.is_valid, not(local.is_last_hypercube)))
            .assert_one(next.row_index - local.row_index);
        builder
            .when(and(local.is_valid, not(local.is_last_hypercube)))
            .assert_eq(next.neg_hypercube, local.neg_hypercube);

        /*
         * Receive u_0 and r_0 from the AIRs that sample them on the first row. Also,
         * send (u, r^2, (r * omega)^2) on the first row of non-last hypercubes and
         * receive (u, r, r * omega) on the first row of non-first hypercubes.
         */
        let initial_omega = AB::F::two_adic_generator(self.l_skip - 1);
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.r_omega_pow,
            ext_field_multiply_scalar(local.r_pow, initial_omega),
        );

        self.base_rand_bus.receive(
            builder,
            local.proof_idx,
            EqNegBaseRandMessage {
                u: local.u_pow,
                r_squared: local.r_pow,
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
            &mut builder.when(and(local.is_valid, not(local.is_last_hypercube))),
            ext_field_multiply(local.u_pow, local.u_pow),
            next.u_pow,
        );

        assert_array_eq(
            &mut builder.when(and(local.is_valid, not(local.is_last_hypercube))),
            ext_field_multiply(local.r_pow, local.r_pow),
            next.r_pow,
        );

        assert_array_eq(
            &mut builder.when(and(local.is_valid, not(local.is_last_hypercube))),
            ext_field_multiply(local.prod_u_r, ext_field_add(next.u_pow, next.r_pow)),
            next.prod_u_r,
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
            &mut builder.when(and(local.is_valid, not(local.is_last_hypercube))),
            ext_field_multiply(local.r_omega_pow, local.r_omega_pow),
            next.r_omega_pow,
        );

        assert_array_eq(
            &mut builder.when(and(local.is_valid, not(local.is_last_hypercube))),
            ext_field_multiply(
                local.prod_u_r_omega,
                ext_field_add(next.u_pow, next.r_omega_pow),
            ),
            next.prod_u_r_omega,
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

        self.result_bus.send(
            builder,
            local.proof_idx,
            EqNegResultMessage {
                n: AB::Expr::ZERO - local.neg_hypercube,
                eq,
                k_rot,
            },
            and(next.is_last_hypercube, next.is_valid),
        );
    }
}
