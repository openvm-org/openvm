use std::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    SubAir,
};
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
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
};

/// For each proof, this AIR will receive 2t interaction claims and t constraint claims.
/// (2 interaction claims and 1 constraint claim per trace).
/// These values are folded (algebraic batching) with mu into a single value, which
/// should match the final sumcheck claim.
///
/// Rows are structured as a nested loop: for each proof, group 0 (interactions) comes first,
/// then group 1 (constraints). Within the interaction group, each trace occupies 2 rows
/// (numerator then denominator). Within the constraint group, each trace occupies 1 row.
/// `NestedForLoopSubAir<2>` enforces canonical enumeration of `(proof_idx, group_idx)`;
/// `trace_idx` and `idx_parity` are caller-managed within each group.
///
/// Example for t = 2 traces (one proof):
///
/// ```text
/// row | is_first | group_idx | is_first_in_group | trace_idx | idx_parity | idx
/// ----|----------|-----------|-------------------|-----------|------------|----
///   0 |    1     |     0     |         1         |     0     |     0      |  0   ← numerator trace 0
///   1 |    0     |     0     |         0         |     0     |     1      |  1   ← denominator trace 0
///   2 |    0     |     0     |         0         |     1     |     0      |  2   ← numerator trace 1
///   3 |    0     |     0     |         0         |     1     |     1      |  3   ← denominator trace 1
///   4 |    0     |     1     |         1         |     0     |     0      |  0   ← constraint trace 0
///   5 |    0     |     1     |         0         |     1     |     0      |  1   ← constraint trace 1
/// ```
///
/// `is_interaction` is derived as `1 - group_idx` (not a separate column).
#[derive(AlignedBorrow, Copy, Clone, Debug)]
#[repr(C)]
pub struct ExpressionClaimCols<T> {
    // --- Loop structure (enforced by NestedForLoopSubAir<2>) ---
    pub is_valid: T,
    /// First row of a proof. Marks proof boundaries.
    pub is_first: T,
    pub proof_idx: T,
    /// 0 = interaction group, 1 = constraint group. Monotone within a proof.
    pub group_idx: T,
    /// Marks the first row of each group (set at proof start and interaction→constraint boundary).
    pub is_first_in_group: T,

    // --- Claim indexing (derived from loop counters) ---
    /// Claim index within its group. For interactions: `2 * trace_idx + idx_parity` (0..2t).
    /// For constraints: `trace_idx` (0..t).
    pub idx: T,
    /// 0 = numerator, 1 = denominator. Always 0 on constraint rows. Alternates on interaction
    /// rows.
    pub idx_parity: T,
    /// Sorted trace index within the group. Monotone non-decreasing; resets at group boundaries.
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

        // === Loop structure via NestedForLoopSubAir<2> ===
        // Enforces: is_valid contiguity, proof_idx and group_idx monotone (0/1 increments),
        // is_first/is_first_in_group boolean and correctly placed at boundaries.
        type LoopSubAir = NestedForLoopSubAir<2>;
        LoopSubAir {}.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_valid,
                    counter: [local.proof_idx, local.group_idx],
                    is_first: [local.is_first, local.is_first_in_group],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_valid,
                    counter: [next.proof_idx, next.group_idx],
                    is_first: [next.is_first, next.is_first_in_group],
                }
                .map_into(),
            ),
        );


        // Derived expressions:
        // is_interaction: true for interaction group (group_idx == 0)
        let is_interaction: AB::Expr = AB::Expr::ONE - local.group_idx.into();
        // is_same_proof: next row is valid and within the same proof
        let is_same_proof: AB::Expr = next.is_valid - next.is_first;
        // is_same_group: next row is valid and within the same group
        let is_same_group: AB::Expr = next.is_valid - next.is_first_in_group;
        // is_last_in_proof: current row is the last row of its proof
        let is_last_in_proof: AB::Expr =
            LoopSubAir::local_is_last(local.is_valid, next.is_valid, next.is_first);

        // Each proof starts with group 0 (interactions) and ends with 1 (constraints).
        builder.when(local.is_first).assert_zero(local.group_idx);
        builder.when(is_last_in_proof.clone()).assert_one(local.group_idx);

        // === Claim indexing constraints ===
        builder.assert_bool(local.idx_parity);
        builder.assert_bool(local.n_sign);

        // trace_idx resets to 0 at each group boundary and is monotone within a group
        builder
            .when(local.is_first_in_group)
            .assert_zero(local.trace_idx);
        // For constraints (group_idx=1), idx_parity is 0, so trace_idx always increase by 1.
        // For interactions (group_idx=0), trace_idx increase by idx_parity.
        builder
            .when(is_same_group.clone())
            .assert_eq(
                next.trace_idx - local.trace_idx,
                local.idx_parity + local.group_idx,
            );

        // idx_parity: 0 at group start, alternates within interactions
        builder
            .when(local.is_first_in_group)
            .assert_zero(local.idx_parity);
        // idx_parity alternates 0/1.
        builder
            .when(local.is_valid)
            .when(is_interaction.clone())
            .assert_eq(local.idx_parity + next.idx_parity, AB::Expr::ONE);
        // only group 0 can have idx_parity set.
        builder.when(local.idx_parity).assert_zero(local.group_idx);

        // idx binding to trace_idx / idx_parity (INT-6397 / INT-6439)
        // Interaction rows: idx = 2 * trace_idx + idx_parity
        builder
            .when(local.is_valid)
            .when(is_interaction.clone())
            .assert_eq(
                local.idx,
                local.trace_idx * AB::Expr::TWO + local.idx_parity,
            );
        // Constraint rows: idx = trace_idx
        builder
            .when(local.is_valid)
            .when(local.group_idx)
            .assert_eq(local.idx, local.trace_idx);

        // === mu constancy within a proof ===
        assert_array_eq(
            &mut builder.when(is_same_proof.clone()),
            next.mu,
            local.mu.map(Into::into),
        );

        // === Hyperdim metadata constancy within numerator/denominator pairs (INT-6399) ===
        // A numerator row (idx_parity=0, is_interaction=1) is always followed by its
        // denominator (idx_parity=1) due to the alternation constraint. Ensure they
        // share the same trace metadata so the hyperdim lookup on the numerator binds both.
        builder
            .when(local.is_valid)
            .when(is_interaction.clone())
            .when(not(local.idx_parity))
            .assert_eq(next.n_abs, local.n_abs);
        builder
            .when(local.is_valid)
            .when(is_interaction.clone())
            .when(not(local.idx_parity))
            .assert_eq(next.n_sign, local.n_sign);

        // === cum sum folding ===
        // Fold recurrence within a proof: cur_sum = value * multiplier + next_cur_sum * mu
        assert_array_eq(
            &mut builder.when(is_same_proof),
            local.cur_sum,
            ext_field_add::<AB::Expr>(
                ext_field_multiply::<AB::Expr>(local.value, local.multiplier),
                ext_field_multiply::<AB::Expr>(next.cur_sum, local.mu),
            ),
        );
        // Terminal base case: last row of each proof's fold (INT-6402)
        assert_array_eq(
            &mut builder.when(is_last_in_proof),
            local.cur_sum,
            ext_field_multiply::<AB::Expr>(local.value, local.multiplier),
        );

        // multiplier = 1 if not interaction
        assert_array_eq(
            &mut builder.when(local.group_idx).when(local.is_valid),
            local.multiplier,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );

        // IF negative n and numerator
        assert_array_eq(
            &mut builder.when(local.n_sign * (is_interaction.clone() - local.idx_parity)),
            ext_field_multiply_scalar::<AB::Expr>(local.multiplier, local.n_abs_pow),
            local.eq_sharp_ns,
        );
        // ELSE 1: positive n, interaction row
        assert_array_eq(
            &mut builder.when(is_interaction.clone() * (AB::Expr::ONE - local.n_sign)),
            local.multiplier,
            local.eq_sharp_ns,
        );
        // ELSE 2: denominator row
        assert_array_eq(
            &mut builder.when(local.idx_parity),
            local.multiplier,
            local.eq_sharp_ns,
        );

        // === bus interactions ===
        self.expr_claim_bus.receive(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: is_interaction.clone(),
                idx: local.idx.into(),
                value: local.value.map(Into::into),
            },
            local.is_valid,
        );

        self.mu_bus.lookup_key(
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

        self.hyperdim_bus.lookup_key(
            builder,
            local.proof_idx,
            HyperdimBusMessage {
                sort_idx: local.trace_idx.into(),
                n_abs: local.n_abs.into(),
                n_sign_bit: local.n_sign.into(),
            },
            local.is_valid * (is_interaction.clone() - local.idx_parity),
        );

        self.eq_n_outer_bus.lookup_key(
            builder,
            local.proof_idx,
            EqNOuterMessage {
                is_sharp: AB::Expr::ONE,
                n: local.n_abs * (AB::Expr::ONE - local.n_sign),
                value: local.eq_sharp_ns.map(Into::into),
            },
            local.is_valid * is_interaction.clone(),
        );

        self.pow_checker_bus.lookup_key(
            builder,
            PowerCheckerBusMessage {
                log: local.n_abs.into(),
                exp: local.n_abs_pow.into(),
            },
            local.is_valid * is_interaction,
        );
    }
}
