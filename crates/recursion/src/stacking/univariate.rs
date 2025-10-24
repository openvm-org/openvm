use std::borrow::{Borrow, BorrowMut};

use itertools::{Itertools, izip};
use openvm_circuit_primitives::utils::{and, not};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    FieldAlgebra, FieldExtensionAlgebra, PrimeField32, extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        StackingSumcheckRandomnessBus, StackingSumcheckRandomnessMessage, TranscriptBus,
        TranscriptBusMessage,
    },
    stacking::bus::{
        EqBitsLookupBus, EqKernelLookupBus, EqRandValuesLookupBus, EqRandValuesLookupMessage,
        StackingModuleTidxBus, StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
    },
    system::Preflight,
    utils::{
        assert_eq_array, assert_one_ext, ext_field_add, ext_field_multiply,
        ext_field_multiply_scalar,
    },
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UnivariateRoundCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Sampled transcript values
    pub tidx: F,
    pub u_0: [F; D_EF],
    pub u_0_pow: [F; D_EF],

    // Coefficients of univariate round (s_0) polynomial
    pub coeff: [F; D_EF],

    // Columns to compute s_0(z) sum over all z in D
    pub coeff_idx: F,
    pub coeff_is_d: F,
    pub s_0_sum_over_d: [F; D_EF],

    // Evaluation of s_0 polynomial at u_0
    pub poly_rand_eval: [F; D_EF],
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct UnivariateRoundTraceGenerator;

impl UnivariateRoundTraceGenerator {
    pub fn generate_trace(
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight,
    ) -> RowMajorMatrix<F> {
        let coeffs = &proof.stacking_proof.univariate_round_coeffs;

        let width = UnivariateRoundCols::<usize>::width();
        let num_valid = coeffs.len();
        let num_rows = num_valid.next_power_of_two();

        let mut trace = vec![F::ZERO; num_rows * width];

        let u_0 = preflight.stacking.sumcheck_rnd[0];
        let u_0_pows = u_0.powers().take(num_valid).collect_vec();

        let initial_tidx = preflight.stacking.intermediate_tidx[0];

        let d_card = 1usize << vk.inner.params.l_skip;
        let mut s_0_sum_over_d = coeffs[0] * F::from_canonical_usize(d_card);
        let mut poly_rand_eval = EF::ZERO;

        for (i, (&coeff, chunk, &u_0_pow)) in
            izip!(coeffs.iter(), trace.chunks_mut(width), u_0_pows.iter()).enumerate()
        {
            let cols: &mut UnivariateRoundCols<F> = chunk.borrow_mut();
            // TODO: handle proof_idx
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(i == 0);
            cols.is_last = F::from_bool(i + 1 == num_valid);

            cols.tidx = F::from_canonical_usize(initial_tidx + (D_EF * i));
            cols.u_0.copy_from_slice(u_0.as_base_slice());
            cols.u_0_pow.copy_from_slice(u_0_pow.as_base_slice());

            cols.coeff.copy_from_slice(coeff.as_base_slice());

            cols.coeff_idx = F::from_canonical_usize(i);
            if i == d_card {
                s_0_sum_over_d += coeff * F::from_canonical_usize(d_card);
                cols.coeff_is_d = F::ONE;
            }
            cols.s_0_sum_over_d
                .copy_from_slice(s_0_sum_over_d.as_base_slice());

            poly_rand_eval += coeff * u_0_pow;
            cols.poly_rand_eval
                .copy_from_slice(poly_rand_eval.as_base_slice());
        }

        if num_valid < num_rows {
            let last_row: &mut UnivariateRoundCols<F> =
                trace[(num_rows - 1) * width..].borrow_mut();
            last_row.is_last = F::ONE;
        }

        RowMajorMatrix::new(trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct UnivariateRoundAir {
    // External buses
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
    pub eq_rand_values_bus: EqRandValuesLookupBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,
    pub eq_bits_lookup_bus: EqBitsLookupBus,

    // Other fields
    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for UnivariateRoundAir {}
impl PartitionedBaseAir<F> for UnivariateRoundAir {}

impl<F> BaseAir<F> for UnivariateRoundAir {
    fn width(&self) -> usize {
        UnivariateRoundCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UnivariateRoundAir
where
    AB::F: PrimeField32,
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &UnivariateRoundCols<AB::Var> = (*local).borrow();
        let next: &UnivariateRoundCols<AB::Var> = (*next).borrow();

        // TODO[stephen]: constrain proof_idx, is_valid, is_first, is_last

        /*
         * Constrain that the sum of s_0(z) for z in D via interaction equals the RLC of column
         * claims from OpeningClaimsAir. We use the properties of D to do this efficiently -
         * since D is a multiplicative subgroup, it turns out this sum is |D| * (a_0 + a_{|D|}).
         */
        let d_card = 1usize << self.l_skip;

        builder.when(local.is_first).assert_zero(local.coeff_idx);
        builder.when(and(local.is_last, local.is_valid)).assert_eq(
            local.coeff_idx,
            AB::F::from_canonical_usize(2 * (d_card - 1)),
        );
        builder
            .when(and(not(local.is_last), local.is_valid))
            .assert_one(next.coeff_idx - local.coeff_idx);

        builder.assert_bool(local.coeff_is_d);
        builder.when(local.coeff_is_d).assert_one(local.is_valid);
        builder
            .when(local.coeff_is_d)
            .assert_eq(local.coeff_idx, AB::F::from_canonical_usize(d_card));

        assert_eq_array(
            &mut builder.when(local.is_first),
            ext_field_multiply_scalar(local.coeff, AB::F::from_canonical_usize(d_card)),
            local.s_0_sum_over_d,
        );

        assert_eq_array(
            &mut builder.when(next.coeff_is_d),
            ext_field_add(
                local.s_0_sum_over_d,
                ext_field_multiply_scalar(next.coeff, AB::F::from_canonical_usize(d_card)),
            ),
            next.s_0_sum_over_d,
        );

        self.sumcheck_claims_bus.receive(
            builder,
            next.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ZERO,
                value: next.s_0_sum_over_d.map(Into::into),
            },
            next.coeff_is_d,
        );

        /*
         * Compute evaluation of polynomial s_0(u_0) and send it to SumcheckRoundsAir, where
         * it'll be used to constrain the correctness of s_1(0).
         */
        assert_one_ext(&mut builder.when(local.is_first), local.u_0_pow);
        assert_eq_array(&mut builder.when(not(local.is_last)), local.u_0, next.u_0);

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.u_0, local.u_0_pow),
            next.u_0_pow,
        );

        assert_eq_array(
            &mut builder.when(local.is_first),
            ext_field_multiply(local.coeff, local.u_0_pow),
            local.poly_rand_eval,
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_add(
                local.poly_rand_eval,
                ext_field_multiply(next.coeff, next.u_0_pow),
            ),
            next.poly_rand_eval,
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ONE,
                value: local.poly_rand_eval.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Because we sample u_0 from the transcript here, we send u_0 to other AIRs that
         * need to use it.
         */
        self.stacking_randomness_bus.send(
            builder,
            local.proof_idx,
            StackingSumcheckRandomnessMessage {
                idx: AB::Expr::ZERO,
                challenge: local.u_0.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        self.eq_rand_values_bus.send(
            builder,
            local.proof_idx,
            EqRandValuesLookupMessage {
                idx: AB::Expr::ZERO,
                u: local.u_0.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Constrain transcript operations and send the final tidx to SumcheckRoundsAir.
         */
        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ZERO,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx,
                    value: local.coeff[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i + D_EF) + local.tidx,
                    value: local.u_0[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                and(local.is_last, local.is_valid),
            );
        }

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ONE,
                tidx: AB::Expr::from_canonical_usize(2 * D_EF) + local.tidx,
            },
            and(local.is_last, local.is_valid),
        );
    }
}
