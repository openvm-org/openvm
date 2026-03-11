use core::borrow::Borrow;

use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::{
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{eq_1, ext_field_add, ext_field_multiply},
    whir::bus::{
        FinalPolyQueryEvalBus, FinalPolyQueryEvalMessage, WhirAlphaBus, WhirAlphaMessage,
        WhirFinalPolyBus, WhirFinalPolyBusMessage, WhirGammaBus, WhirGammaMessage, WhirQueryBus,
        WhirQueryBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub(in crate::whir::final_poly_query_eval) struct FinalPolyQueryEvalCols<T> {
    pub is_enabled: T,
    // loop indices
    pub proof_idx: T,
    pub whir_round: T,
    pub query_idx: T,
    pub phase_idx: T,
    pub eval_idx: T,
    // is_first flags
    pub is_first_in_proof: T,
    pub is_first_in_round: T,
    pub is_first_in_query: T,
    pub is_first_in_phase: T,
    pub is_last_round: T,
    pub is_query_zero: T,
    pub query_pow: [T; D_EF],
    pub alpha: [T; D_EF],
    pub gamma: [T; D_EF],
    pub gamma_pow: [T; D_EF],
    pub final_poly_coeff: [T; D_EF],
    pub final_value_acc: [T; D_EF],
    pub gamma_eq_acc: [T; D_EF],
    pub horner_acc: [T; D_EF],
    pub do_carry: T,
}

#[derive(Debug)]
pub struct FinalPolyQueryEvalAir {
    pub query_bus: WhirQueryBus,
    pub alpha_bus: WhirAlphaBus,
    pub gamma_bus: WhirGammaBus,
    pub final_poly_bus: WhirFinalPolyBus,
    pub final_poly_query_eval_bus: FinalPolyQueryEvalBus,
    pub num_whir_rounds: usize,
    pub k_whir: usize,
    pub log_final_poly_len: usize,
}

impl BaseAirWithPublicValues<F> for FinalPolyQueryEvalAir {}
impl PartitionedBaseAir<F> for FinalPolyQueryEvalAir {}

impl<F> BaseAir<F> for FinalPolyQueryEvalAir {
    fn width(&self) -> usize {
        FinalPolyQueryEvalCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FinalPolyQueryEvalAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &FinalPolyQueryEvalCols<AB::Var> = (*local).borrow();
        let next: &FinalPolyQueryEvalCols<AB::Var> = (*next).borrow();

        let k_whir_f = AB::Expr::from_usize(self.k_whir);
        let eq_phase_len =
            k_whir_f.clone() * (AB::Expr::from_usize(self.num_whir_rounds - 1) - local.whir_round);
        let final_poly_phase_len = AB::Expr::from_usize((1 << self.log_final_poly_len) - 1);

        let round_base = (local.whir_round + AB::Expr::ONE) * k_whir_f;

        let proof_idx = local.proof_idx;

        builder.assert_bool(local.phase_idx);
        builder.when(local.phase_idx).assert_one(local.is_enabled);
        let local_is_eq_phase = local.is_enabled - local.phase_idx;

        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        let is_same_round = next.is_enabled - next.is_first_in_round;
        let is_same_query = next.is_enabled - next.is_first_in_query;
        let is_same_phase = next.is_enabled - next.is_first_in_phase;

        builder
            .when(local.phase_idx)
            .when(local.is_enabled - is_same_phase.clone())
            .assert_eq(local.eval_idx, final_poly_phase_len.clone());

        builder
            .when(AB::Expr::ONE - local.phase_idx)
            .when(local.is_enabled - is_same_phase.clone())
            .assert_eq(local.eval_idx, eq_phase_len - AB::Expr::ONE);

        NestedForLoopSubAir.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled.into(),
                    counter: [
                        local.proof_idx.into(),
                        local.whir_round.into(),
                        local.query_idx.into(),
                        // Adjust the phase counter so every enabled query still starts at 0:
                        // non-final rounds must begin in eq-phase (phase_idx = 0), while the
                        // last round may start at phase_idx = 1 because its eq-phase is empty.
                        local.phase_idx - local.is_last_round,
                        local.eval_idx.into(),
                    ],
                    is_first: [
                        local.is_first_in_proof,
                        local.is_first_in_round,
                        local.is_first_in_query,
                        local.is_first_in_phase,
                        local.is_enabled,
                    ]
                    .map(Into::into),
                },
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled.into(),
                    counter: [
                        next.proof_idx.into(),
                        next.whir_round.into(),
                        next.query_idx.into(),
                        next.phase_idx - next.is_last_round,
                        next.eval_idx.into(),
                    ],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_round,
                        next.is_first_in_query,
                        next.is_first_in_phase,
                        next.is_enabled,
                    ]
                    .map(Into::into),
                },
            ),
        );

        assert_array_eq(
            &mut builder.when(local_is_eq_phase.clone()),
            next.query_pow,
            ext_field_multiply::<AB::Expr>(local.query_pow, local.query_pow),
        );

        assert_array_eq(
            &mut builder.when(local.phase_idx * is_same_query.clone()),
            next.query_pow,
            local.query_pow,
        );

        // The value of post_horner_acc *after* this inner-most loop iteration, if we are not in
        // eq_phase. Degree 2.
        let post_horner_acc = ext_field_add(
            ext_field_multiply(local.horner_acc, local.query_pow),
            local.final_poly_coeff,
        );

        assert_array_eq(
            &mut builder.when(local.is_first_in_query),
            local.gamma_eq_acc,
            local.gamma_pow,
        );
        assert_array_eq(
            &mut builder.when(local_is_eq_phase.clone()),
            next.gamma_eq_acc,
            ext_field_multiply(local.gamma_eq_acc, eq_1(local.alpha, local.query_pow)),
        );
        assert_array_eq(
            &mut builder.when((AB::Expr::ONE - local_is_eq_phase.clone()) * is_same_query.clone()),
            next.gamma_eq_acc,
            local.gamma_eq_acc,
        );
        assert_array_eq(
            &mut builder.when(local.is_first_in_round),
            local.gamma_pow,
            local.gamma,
        );
        assert_array_eq(
            &mut builder.when(is_same_query.clone()),
            next.gamma_pow,
            local.gamma_pow,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone() * next.is_first_in_query),
            next.gamma_pow,
            ext_field_multiply(local.gamma_pow, local.gamma),
        );

        assert_array_eq(
            &mut builder.when(is_same_query.clone()),
            next.final_value_acc,
            local.final_value_acc,
        );
        builder
            .when(local.is_first_in_proof)
            .assert_zeros(local.final_value_acc);
        builder
            .when(local.is_first_in_round)
            .assert_one(local.is_query_zero);
        builder
            .when(is_same_query.clone())
            .assert_eq(local.is_query_zero, next.is_query_zero);
        builder
            .when(local.query_idx)
            .assert_zero(local.is_query_zero);
        builder
            .when(local.is_enabled - is_same_proof.clone())
            .assert_one(local.is_last_round);
        builder
            .when(is_same_round.clone())
            .assert_eq(local.is_last_round, next.is_last_round);
        builder
            .when(local.whir_round - AB::Expr::from_usize(self.num_whir_rounds - 1))
            .assert_zero(local.is_last_round);

        builder.assert_bool(local.do_carry);
        builder.assert_eq(
            local.is_enabled * local.do_carry,
            (is_same_proof.clone() - is_same_query.clone())
                * (AB::Expr::ONE - local.is_query_zero * local.is_last_round),
        );

        let gamma_eq_post = ext_field_multiply(local.gamma_eq_acc, post_horner_acc.clone());

        builder
            .when(local_is_eq_phase.clone())
            .assert_zeros(local.horner_acc);
        builder
            .when(local.is_first_in_query)
            .assert_zeros(local.horner_acc);
        builder
            .when(local_is_eq_phase.clone())
            .assert_zeros(local.final_poly_coeff);
        assert_array_eq(
            &mut builder.when(is_same_query.clone()),
            next.horner_acc,
            post_horner_acc.clone(),
        );
        let contrib = ext_field_multiply::<AB::Expr>(local.gamma_eq_acc, post_horner_acc.clone());
        let post_final_value_acc = ext_field_add(local.final_value_acc, contrib);
        assert_array_eq(
            &mut builder.when(local.do_carry),
            next.final_value_acc,
            post_final_value_acc,
        );
        assert_array_eq(
            &mut builder.when(
                (is_same_proof.clone() - is_same_query.clone())
                    * (local.is_query_zero * local.is_last_round),
            ),
            next.final_value_acc,
            local.final_value_acc,
        );

        self.query_bus.receive(
            builder,
            proof_idx,
            WhirQueryBusMessage {
                whir_round: local.whir_round,
                query_idx: local.query_idx,
                value: local.query_pow,
            },
            local.is_first_in_query,
        );
        let alpha_idx = round_base + local.eval_idx;
        self.alpha_bus.lookup_key(
            builder,
            proof_idx,
            WhirAlphaMessage {
                idx: alpha_idx,
                challenge: local.alpha.map(Into::into),
            },
            local_is_eq_phase.clone(),
        );
        self.gamma_bus.receive(
            builder,
            proof_idx,
            WhirGammaMessage {
                idx: local.whir_round,
                challenge: local.gamma,
            },
            local.is_first_in_round,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            local.gamma,
            next.gamma,
        );
        let is_last = local.is_enabled - is_same_proof.clone();
        let final_value = ext_field_add(local.final_value_acc, gamma_eq_post);
        self.final_poly_query_eval_bus.receive(
            builder,
            proof_idx,
            FinalPolyQueryEvalMessage {
                last_whir_round: AB::Expr::from_usize(self.num_whir_rounds),
                value: final_value,
            },
            is_last,
        );

        let coeff_idx = final_poly_phase_len - local.eval_idx;
        self.final_poly_bus.lookup_key(
            builder,
            proof_idx,
            WhirFinalPolyBusMessage {
                idx: coeff_idx,
                coeff: local.final_poly_coeff.map(Into::into),
            },
            local.is_enabled * local.phase_idx,
        );
    }
}
