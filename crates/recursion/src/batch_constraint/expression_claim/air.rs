use std::borrow::Borrow;

use openvm_circuit_primitives::utils::{assert_array_eq, not};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, EqNOuterBus, EqNOuterMessage, ExpressionClaimBus,
        ExpressionClaimMessage, SumcheckClaimBus, SumcheckClaimMessage,
    },
    bus::{ExpressionClaimNMaxBus, ExpressionClaimNMaxMessage, HyperdimBus, HyperdimBusMessage},
    primitives::bus::{PowerCheckerBus, PowerCheckerBusMessage},
    utils::{base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
};

/// For each proof, this AIR will receive 2t interaction claims and t constraint claims.
/// (2 interaction claims and 1 constraint claim per trace).
/// These values are folded (algebraic batching) with mu into a single value, which
/// should match the final sumcheck claim.
#[derive(AlignedBorrow, Copy, Clone, Debug)]
#[repr(C)]
pub struct ExpressionClaimCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub proof_idx: T,

    pub is_interaction: T,
    /// Index within the proof, 0 ~ 2t-1 are interaction claims, 0~t-1 are constraint claims.
    pub idx: T,
    pub idx_parity: T,
    pub trace_idx: T,
    /// The received evaluation claim. Note that for interactions, this is without norm_factor and
    /// eq_sharp_ns. These are interactions_evals (without norm_factor and eq_sharp_ns) and
    /// constraint_evals in the rust verifier.
    pub value: [T; D_EF],
    /// Receive from eq_ns AIR
    pub eq_sharp_ns: [T; D_EF],

    /// For folding with mu.
    pub cur_sum: [T; D_EF],
    pub mu: [T; D_EF],
    pub multiplier: [T; D_EF],

    /// Need to know n as if n<0, we need to multiply some norm_factor.
    pub n_abs: T,
    pub n_abs_pow: T,
    pub n_sign: T,
    /// The round idx for final sumcheck claim.
    pub num_multilinear_sumcheck_rounds: T,
}

pub struct ExpressionClaimAir {
    pub expression_claim_n_max_bus: ExpressionClaimNMaxBus,
    pub expr_claim_bus: ExpressionClaimBus,
    pub mu_bus: BatchConstraintConductorBus,
    pub sumcheck_claim_bus: SumcheckClaimBus,
    pub eq_n_outer_bus: EqNOuterBus,
    pub pow_checker_bus: PowerCheckerBus,
    pub hyperdim_bus: HyperdimBus,
}

impl<F> BaseAirWithPublicValues<F> for ExpressionClaimAir {}
impl<F> PartitionedBaseAir<F> for ExpressionClaimAir {}

impl<F> BaseAir<F> for ExpressionClaimAir {
    fn width(&self) -> usize {
        ExpressionClaimCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpressionClaimAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &ExpressionClaimCols<AB::Var> = (*local).borrow();
        let next: &ExpressionClaimCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_interaction);
        builder.assert_bool(local.idx_parity);
        builder.assert_bool(local.n_sign);
        builder
            .when(local.is_first)
            .assert_one(local.is_interaction);
        builder.when(local.is_first).assert_zero(local.idx_parity);
        builder
            .when(local.is_interaction)
            .assert_eq(local.idx_parity + next.idx_parity, AB::Expr::ONE);
        builder
            .when(local.idx_parity)
            .assert_one(local.is_interaction);

        // === cum sum folding ===
        // cur_sum = next_cur_sum * mu + value * multiplier
        assert_array_eq(
            &mut builder.when(local.is_valid * not(next.is_first)),
            local.cur_sum,
            ext_field_add::<AB::Expr>(
                ext_field_multiply::<AB::Expr>(local.value, local.multiplier),
                ext_field_multiply::<AB::Expr>(next.cur_sum, local.mu),
            ),
        );
        // multiplier = 1 if not interaction
        assert_array_eq(
            &mut builder.when(not(local.is_interaction)).when(local.is_valid),
            local.multiplier,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );

        // IF negative n and numerator
        assert_array_eq(
            &mut builder.when(local.n_sign * (local.is_interaction - local.idx_parity)),
            ext_field_multiply_scalar::<AB::Expr>(local.multiplier, local.n_abs_pow),
            local.eq_sharp_ns,
        );
        // ELSE 1
        assert_array_eq(
            &mut builder.when(local.is_interaction * (AB::Expr::ONE - local.n_sign)),
            local.multiplier,
            local.eq_sharp_ns,
        );
        // ELSE 2
        assert_array_eq(
            &mut builder.when(local.idx_parity),
            local.multiplier,
            local.eq_sharp_ns,
        );

        // === interactions ===
        self.expr_claim_bus.receive(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: local.is_interaction,
                idx: local.idx,
                value: local.value,
            },
            local.is_valid,
        );

        self.mu_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Mu.to_field(),
                idx: AB::Expr::ZERO,
                value: local.mu.map(Into::into),
            },
            local.is_first * local.is_valid,
        );

        // Receive n_max value from proof shape air
        self.expression_claim_n_max_bus.receive(
            builder,
            local.proof_idx,
            ExpressionClaimNMaxMessage {
                n_max: local.num_multilinear_sumcheck_rounds,
            },
            local.is_first * local.is_valid,
        );

        self.sumcheck_claim_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimMessage::<AB::Expr> {
                round: local.num_multilinear_sumcheck_rounds.into(),
                value: local.cur_sum.map(Into::into),
            },
            local.is_first * local.is_valid,
        );

        self.hyperdim_bus.receive(
            builder,
            local.proof_idx,
            HyperdimBusMessage {
                sort_idx: local.trace_idx.into(),
                n_abs: local.n_abs.into(),
                n_sign_bit: local.n_sign.into(),
            },
            local.is_valid * (local.is_interaction - local.idx_parity),
        );

        self.eq_n_outer_bus.lookup_key(
            builder,
            local.proof_idx,
            EqNOuterMessage {
                is_sharp: AB::Expr::ONE,
                n: local.n_abs * (AB::Expr::ONE - local.n_sign),
                value: local.eq_sharp_ns.map(Into::into),
            },
            local.is_valid * local.is_interaction,
        );

        self.pow_checker_bus.lookup_key(
            builder,
            PowerCheckerBusMessage {
                log: local.n_abs.into(),
                exp: local.n_abs_pow.into(),
            },
            local.is_valid * local.is_interaction,
        );
    }
}
