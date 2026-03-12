use core::borrow::Borrow;

use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::TranscriptBus,
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{ext_field_add, ext_field_multiply},
    whir::bus::{
        VerifyQueriesBus, VerifyQueriesBusMessage, VerifyQueryBus, VerifyQueryBusMessage,
        WhirQueryBus, WhirQueryBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct WhirQueryCols<T> {
    pub is_enabled: T,
    // loop counters
    pub proof_idx: T,
    pub whir_round: T,
    pub query_idx: T,
    // first flags
    pub is_first_in_proof: T,
    pub is_first_in_round: T,
    pub tidx: T,
    pub num_queries: T,
    pub omega: T,
    pub sample: T,
    pub zi_root: T,
    pub zi: T,
    pub yi: [T; D_EF],
    pub gamma: [T; D_EF],
    pub gamma_pow: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub post_claim: [T; D_EF],
}

pub struct WhirQueryAir {
    pub transcript_bus: TranscriptBus,
    pub verify_queries_bus: VerifyQueriesBus,
    pub query_bus: WhirQueryBus,
    pub verify_query_bus: VerifyQueryBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub k: usize,
    pub initial_log_domain_size: usize,
}

impl BaseAirWithPublicValues<F> for WhirQueryAir {}
impl PartitionedBaseAir<F> for WhirQueryAir {}

impl<F> BaseAir<F> for WhirQueryAir {
    fn width(&self) -> usize {
        WhirQueryCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for WhirQueryAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &WhirQueryCols<AB::Var> = (*local).borrow();
        let next: &WhirQueryCols<AB::Var> = (*next).borrow();

        let proof_idx = local.proof_idx;
        let is_enabled = local.is_enabled;
        let is_same_round = next.is_enabled - next.is_first_in_round;

        NestedForLoopSubAir.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled,
                    counter: [local.proof_idx, local.whir_round, local.query_idx],
                    is_first: [local.is_first_in_proof, local.is_first_in_round, is_enabled],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx, next.whir_round, next.query_idx],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_round,
                        next.is_enabled,
                    ],
                }
                .map_into(),
            ),
        );

        assert_array_eq(
            &mut builder.when(local.is_first_in_round),
            local.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma),
        );

        builder
            .when(local.is_enabled - is_same_round.clone())
            .assert_eq(local.query_idx + AB::Expr::ONE, local.num_queries);

        // num_queries is constrained via the bus interaction with WhirRoundAir
        self.verify_queries_bus.receive(
            builder,
            proof_idx,
            VerifyQueriesBusMessage {
                tidx: local.tidx,
                whir_round: local.whir_round,
                num_queries: local.num_queries,
                omega: local.omega,
                gamma: local.gamma,
                pre_claim: local.pre_claim,
                post_claim: local.post_claim,
            },
            local.is_first_in_round,
        );
        builder
            .when(is_same_round.clone())
            .assert_one(next.tidx - local.tidx);
        builder
            .when(is_same_round.clone())
            .assert_eq(next.omega, local.omega);
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.gamma,
            local.gamma,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma_pow),
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.pre_claim,
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.gamma_pow, local.yi),
            ),
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.post_claim,
            local.post_claim,
        );
        assert_array_eq(
            &mut builder.when(local.is_enabled - is_same_round.clone()),
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.yi, local.gamma_pow),
            ),
            local.post_claim,
        );

        self.transcript_bus
            .sample(builder, proof_idx, local.tidx, local.sample, is_enabled);
        self.query_bus.send(
            builder,
            proof_idx,
            WhirQueryBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: local.query_idx.into() + AB::Expr::ONE,
                value: [
                    local.zi.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
            },
            is_enabled,
        );
        self.verify_query_bus.send(
            builder,
            proof_idx,
            VerifyQueryBusMessage {
                whir_round: local.whir_round,
                query_idx: local.query_idx,
                merkle_idx_bit_src: local.sample,
                zi_root: local.zi_root,
                zi: local.zi,
                yi: local.yi,
            },
            is_enabled,
        );
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.omega.into(),
                bit_src: local.sample.into(),
                num_bits: AB::Expr::from_usize(self.initial_log_domain_size - self.k)
                    - local.whir_round,
                result: local.zi_root.into(),
            },
            is_enabled,
        );
    }
}
