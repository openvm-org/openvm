use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_circuit_primitives::{
    SubAir,
    utils::{and, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    Field, FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField,
    extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, WhirOpeningPointBus,
        WhirOpeningPointMessage,
    },
    stacking::bus::{
        EqBaseBus, EqBaseMessage, EqKernelLookupBus, EqKernelLookupMessage, EqRandValuesLookupBus,
        EqRandValuesLookupMessage,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{
        assert_eq_array, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_subtract,
    },
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct EqBaseCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Row index for the given proof
    pub row_idx: F,

    // Value of u^{2^row}, r^{2^row}, and (r * omega)^{2^row}
    pub u_pow: [F; D_EF],
    pub r_pow: [F; D_EF],
    pub r_omega_pow: [F; D_EF],

    // Running product of (u^{2^i} + r^{2^i}) from i to row
    pub prod_u_r: [F; D_EF],
    pub prod_u_r_omega: [F; D_EF],
    pub prod_u_1: [F; D_EF],
    pub prod_r_omega_1: [F; D_EF],

    // Lookup multiplicity for eq_0(u, r) and k_rot_0(u, r)
    pub mult: F,
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct EqBaseTraceGenerator;

impl EqBaseTraceGenerator {
    pub fn generate_trace(
        vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> RowMajorMatrix<F> {
        debug_assert_eq!(proofs.len(), preflights.len());

        let width = EqBaseCols::<usize>::width();

        if proofs.is_empty() {
            return RowMajorMatrix::new(vec![F::ZERO; width], width);
        }

        let mut combined_trace = Vec::<F>::new();
        let mut total_rows = 0usize;

        for (proof_idx, (proof, preflight)) in proofs.iter().zip(preflights).enumerate() {
            let mut mult = 0usize;
            for (sort_idx, (_, vdata)) in
                preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
            {
                if vdata.hypercube_dim == 0 {
                    mult += proof.batch_constraint_proof.column_openings[sort_idx]
                        .iter()
                        .flatten()
                        .collect_vec()
                        .len();
                }
            }

            let num_rows = vk.inner.params.l_skip + 1;
            let proof_idx_value = F::from_canonical_usize(proof_idx);

            let mut trace = vec![F::ZERO; num_rows * width];

            for chunk in trace.chunks_mut(width) {
                let cols: &mut EqBaseCols<F> = chunk.borrow_mut();
                cols.proof_idx = proof_idx_value;
            }

            let omega = F::two_adic_generator(vk.inner.params.l_skip);
            let mut u = preflight.stacking.sumcheck_rnd[0];
            let mut r = preflight.batch_constraint.sumcheck_rnd[0];
            let mut r_omega = r * omega;

            let mut prod_u_r = u * (u + r);
            let mut prod_u_r_omega = u * (u + r_omega);
            let mut prod_u_1 = u + F::ONE;
            let mut prod_r_omega_1 = r_omega + F::ONE;

            for (row_idx, chunk) in trace.chunks_mut(width).take(num_rows).enumerate() {
                let cols: &mut EqBaseCols<F> = chunk.borrow_mut();
                let is_last = row_idx + 1 == num_rows;

                cols.proof_idx = proof_idx_value;
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(row_idx == 0);
                cols.is_last = F::from_bool(is_last);

                cols.row_idx = F::from_canonical_usize(row_idx);

                cols.u_pow.copy_from_slice(u.as_base_slice());
                cols.r_pow.copy_from_slice(r.as_base_slice());
                cols.r_omega_pow.copy_from_slice(r_omega.as_base_slice());

                cols.prod_u_r.copy_from_slice(prod_u_r.as_base_slice());
                cols.prod_u_r_omega
                    .copy_from_slice(prod_u_r_omega.as_base_slice());
                cols.prod_u_1.copy_from_slice(prod_u_1.as_base_slice());
                cols.prod_r_omega_1
                    .copy_from_slice(prod_r_omega_1.as_base_slice());

                if is_last {
                    cols.mult = F::from_canonical_usize(mult);
                }

                u *= u;
                r *= r;
                r_omega *= r_omega;

                prod_u_r *= u + r;
                prod_u_r_omega *= u + r_omega;
                prod_u_1 *= u + F::ONE;
                prod_r_omega_1 *= r_omega + F::ONE;
            }

            combined_trace.extend(trace);
            total_rows += num_rows;
        }

        let padded_rows = total_rows.next_power_of_two();
        if padded_rows > total_rows {
            let padding_start = combined_trace.len();
            combined_trace.resize(padded_rows * width, F::ZERO);

            let padding_proof_idx = F::from_canonical_usize(proofs.len());
            let mut chunks = combined_trace[padding_start..].chunks_mut(width);
            let num_padded_rows = padded_rows - total_rows;
            for i in 0..num_padded_rows {
                let chunk = chunks.next().unwrap();
                let cols: &mut EqBaseCols<F> = chunk.borrow_mut();
                cols.proof_idx = padding_proof_idx;
                if i + 1 == num_padded_rows {
                    cols.is_last = F::ONE;
                }
            }
        }

        RowMajorMatrix::new(combined_trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct EqBaseAir {
    // External buses
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,

    // Internal buses
    pub eq_base_bus: EqBaseBus,
    pub eq_rand_values_bus: EqRandValuesLookupBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,

    // Other fields
    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for EqBaseAir {}
impl PartitionedBaseAir<F> for EqBaseAir {}

impl<F> BaseAir<F> for EqBaseAir {
    fn width(&self) -> usize {
        EqBaseCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqBaseAir
where
    AB::F: PrimeField32 + TwoAdicField,
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqBaseCols<AB::Var> = (*local).borrow();
        let next: &EqBaseCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1, 0> {}.eval(
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
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_zero((local.proof_idx + AB::F::ONE - next.proof_idx) * next.proof_idx);
        builder
            .when(and(not(local.is_valid), local.is_last))
            .assert_zero(next.proof_idx);

        /*
         * Constrain value of row_idx and send u^{2^row} to WhirOpeningPointBus when
         * row_idx < l_skip.
         */
        let is_valid_transition = and(local.is_valid, not(local.is_last));

        builder.when(local.is_first).assert_zero(local.row_idx);
        builder
            .when(is_valid_transition.clone())
            .assert_eq(local.row_idx + AB::F::ONE, next.row_idx);
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_eq(local.row_idx, AB::F::from_canonical_usize(self.l_skip));

        self.whir_opening_point_bus.send(
            builder,
            local.proof_idx,
            WhirOpeningPointMessage {
                idx: local.row_idx,
                value: local.u_pow,
            },
            is_valid_transition,
        );

        /*
         * Receive the values of u_0 and r_0 from the AIRs that sample them.
         */
        self.constraint_randomness_bus.receive(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: AB::Expr::ZERO,
                challenge: local.r_pow.map(Into::into),
            },
            local.is_first,
        );

        self.eq_rand_values_bus.receive(
            builder,
            local.proof_idx,
            EqRandValuesLookupMessage {
                idx: AB::Expr::ZERO,
                u: local.u_pow.map(Into::into),
            },
            local.is_first,
        );

        /*
         * Constrain the running product of (u^{2^i} + r^{2^i}) from i to row, which is
         * used to compute eq_0(u, r).
         */
        assert_eq_array(
            &mut builder.when(local.is_first),
            local.prod_u_r,
            ext_field_multiply(local.u_pow, ext_field_add(local.u_pow, local.r_pow)),
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.u_pow, local.u_pow),
            next.u_pow,
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.r_pow, local.r_pow),
            next.r_pow,
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.prod_u_r, ext_field_add(next.u_pow, next.r_pow)),
            next.prod_u_r,
        );

        /*
         * Constrain the running product that is used to compute eq_0(u, r * omega).
         */
        let omega = AB::F::two_adic_generator(self.l_skip);

        assert_eq_array(
            &mut builder.when(local.is_first),
            local.prod_u_r_omega,
            ext_field_multiply(local.u_pow, ext_field_add(local.u_pow, local.r_omega_pow)),
        );

        assert_eq_array(
            &mut builder.when(local.is_first),
            local.r_omega_pow,
            ext_field_multiply_scalar(local.r_pow, omega),
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.r_omega_pow, local.r_omega_pow),
            next.r_omega_pow,
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(
                local.prod_u_r_omega,
                ext_field_add(next.u_pow, next.r_omega_pow),
            ),
            next.prod_u_r_omega,
        );

        /*
         * Constrain the running products that are used to compute eq_0(u, 1)
         * and eq_0(r * omega, 1).
         */
        let ef_one = [AB::F::ONE, AB::F::ZERO, AB::F::ZERO, AB::F::ZERO];

        assert_eq_array(
            &mut builder.when(local.is_first),
            local.prod_u_1,
            ext_field_add(local.u_pow, ef_one),
        );

        assert_eq_array(
            &mut builder.when(local.is_first),
            local.prod_r_omega_1,
            ext_field_add(local.r_omega_pow, ef_one),
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.prod_u_1, ext_field_add(next.u_pow, ef_one)),
            next.prod_u_1,
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(
                local.prod_r_omega_1,
                ext_field_add(next.r_omega_pow, ef_one),
            ),
            next.prod_r_omega_1,
        );

        /*
         * Compute eq_0(u, r) and eq_0(u, r * omega), which are sent to the lookup
         * bus. Note that k_rot_0(u, r) = eq_0(u, r * omega).
         */
        let omega_pow_inv = AB::F::from_canonical_usize(1 << self.l_skip).inverse();

        let eq_u_r = ext_field_multiply_scalar(
            ext_field_add::<AB::Expr>(ext_field_subtract(local.prod_u_r, next.u_pow), ef_one),
            omega_pow_inv,
        );

        let eq_u_r_omega = ext_field_multiply_scalar(
            ext_field_add::<AB::Expr>(ext_field_subtract(local.prod_u_r_omega, next.u_pow), ef_one),
            omega_pow_inv,
        );

        builder.when(next.mult).assert_one(next.is_last);

        self.eq_kernel_lookup_bus.send(
            builder,
            local.proof_idx,
            EqKernelLookupMessage {
                n: AB::Expr::ZERO,
                eq: eq_u_r.clone(),
                k_rot: eq_u_r_omega.clone(),
            },
            next.mult,
        );

        /*
         * Compute eq_0(u, 1) and eq_0(r * omega, 1) and send to SumcheckRoundsAir.
         */
        let eq_u_1 = ext_field_multiply_scalar(local.prod_u_1, omega_pow_inv);
        let eq_r_omega_1 = ext_field_multiply_scalar(local.prod_r_omega_1, omega_pow_inv);

        self.eq_base_bus.send(
            builder,
            local.proof_idx,
            EqBaseMessage {
                eq_u_r,
                eq_u_r_omega,
                eq_u_r_prod: ext_field_multiply(eq_u_1, eq_r_omega_1),
            },
            and(next.is_last, next.is_valid),
        );
    }
}
