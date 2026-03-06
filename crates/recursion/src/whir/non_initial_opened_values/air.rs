use core::borrow::Borrow;
use std::array::from_fn;

use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{CHUNK, D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        MerkleVerifyBus, MerkleVerifyBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    whir::bus::{VerifyQueryBus, VerifyQueryBusMessage, WhirFoldingBus, WhirFoldingBusMessage},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub(in crate::whir::non_initial_opened_values) struct NonInitialOpenedValuesCols<T> {
    pub is_enabled: T,
    // Indices
    pub proof_idx: T,
    pub whir_round: T,
    pub query_idx: T,
    pub coset_idx: T,
    // Flags
    pub is_first_in_proof: T,
    pub is_first_in_round: T,
    pub is_first_in_query: T,
    pub merkle_idx_bit_src: T,
    pub zi_root: T,
    pub zi: T,
    pub twiddle: T,
    pub value: [T; D_EF],
    pub value_hash: [T; CHUNK],
    pub yi: [T; D_EF],
}

pub struct NonInitialOpenedValuesAir {
    pub verify_query_bus: VerifyQueryBus,
    pub folding_bus: WhirFoldingBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub merkle_verify_bus: MerkleVerifyBus,
    pub k: usize,
    pub initial_log_domain_size: usize,
}

impl BaseAirWithPublicValues<F> for NonInitialOpenedValuesAir {}
impl PartitionedBaseAir<F> for NonInitialOpenedValuesAir {}

impl<F> BaseAir<F> for NonInitialOpenedValuesAir {
    fn width(&self) -> usize {
        NonInitialOpenedValuesCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for NonInitialOpenedValuesAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &NonInitialOpenedValuesCols<AB::Var> = (*local).borrow();
        let next: &NonInitialOpenedValuesCols<AB::Var> = (*next).borrow();

        let is_same_query = next.is_enabled - next.is_first_in_query;

        let max_coset_idx = AB::Expr::from_usize((1 << self.k) - 1);
        builder
            .when(local.is_enabled - is_same_query.clone())
            .assert_eq(local.coset_idx, max_coset_idx);

        NestedForLoopSubAir.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled.into(),
                    counter: [
                        local.proof_idx.into(),
                        local.whir_round - AB::F::ONE,
                        local.query_idx.into(),
                        local.coset_idx.into(),
                    ],
                    is_first: [
                        local.is_first_in_proof,
                        local.is_first_in_round,
                        local.is_first_in_query,
                        local.is_enabled,
                    ]
                    .map(Into::into),
                },
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled.into(),
                    counter: [
                        next.proof_idx.into(),
                        next.whir_round - AB::F::ONE,
                        next.query_idx.into(),
                        next.coset_idx.into(),
                    ],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_round,
                        next.is_first_in_query,
                        next.is_enabled,
                    ]
                    .map(Into::into),
                },
            ),
        );

        self.verify_query_bus.receive(
            builder,
            local.proof_idx,
            VerifyQueryBusMessage {
                whir_round: local.whir_round,
                query_idx: local.query_idx,
                merkle_idx_bit_src: local.merkle_idx_bit_src,
                zi_root: local.zi_root,
                zi: local.zi,
                yi: local.yi,
            },
            local.is_first_in_query,
        );

        let omega_k = AB::Expr::from_prime_subfield(
            <<AB::Expr as PrimeCharacteristicRing>::PrimeSubfield as TwoAdicField>::two_adic_generator(
                self.k,
            ),
        );
        builder
            .when(local.is_first_in_round)
            .assert_eq(local.twiddle, AB::Expr::ONE);
        builder
            .when(is_same_query.clone())
            .assert_eq(next.twiddle, local.twiddle * omega_k);

        assert_array_eq(&mut builder.when(is_same_query.clone()), local.yi, next.yi);
        builder
            .when(is_same_query.clone())
            .assert_eq(local.zi, next.zi);
        builder
            .when(is_same_query.clone())
            .assert_eq(local.zi_root, next.zi_root);

        self.folding_bus.send(
            builder,
            local.proof_idx,
            WhirFoldingBusMessage {
                whir_round: local.whir_round.into(),
                query_idx: local.query_idx.into(),
                height: AB::Expr::ZERO,
                coset_shift: local.zi_root.into(),
                coset_size: AB::Expr::from_usize(1 << self.k),
                coset_idx: local.coset_idx.into(),
                twiddle: local.twiddle.into(),
                value: local.value.map(Into::into),
                z_final: local.zi.into(),
                y_final: local.yi.map(Into::into),
            },
            local.is_enabled,
        );

        let pre_state: [AB::Expr; POSEIDON2_WIDTH] = from_fn(|i| {
            if i < D_EF {
                local.value[i].into()
            } else {
                AB::Expr::ZERO
            }
        });
        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: pre_state,
                output: local.value_hash.map(Into::into),
            },
            local.is_enabled,
        );

        self.merkle_verify_bus.send(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                value: local.value_hash.map(Into::into),
                merkle_idx: local.merkle_idx_bit_src.into(),
                // There are two parts: hashing leaves (depth k) and merkle proof
                total_depth: AB::Expr::from_usize(self.initial_log_domain_size + 1)
                    - local.whir_round,
                height: AB::Expr::ZERO,
                leaf_sub_idx: local.coset_idx.into(),
                commit_major: local.whir_round.into(),
                commit_minor: AB::Expr::ZERO,
            },
            local.is_enabled,
        );
    }
}
