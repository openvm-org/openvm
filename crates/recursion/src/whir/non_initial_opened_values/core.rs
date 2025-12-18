use core::borrow::{Borrow, BorrowMut};
use std::array::from_fn;

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::Permutation;
use stark_backend_v2::{
    D_EF, F, SystemParams,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::{WIDTH, poseidon2_perm},
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{MerkleVerifyBus, MerkleVerifyBusMessage, Poseidon2Bus, Poseidon2BusMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    whir::bus::{VerifyQueryBus, VerifyQueryBusMessage, WhirFoldingBus, WhirFoldingBusMessage},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub(in crate::whir::non_initial_opened_values) struct NonInitialOpenedValuesCols<T> {
    is_enabled: T,
    // Indices
    proof_idx: T,
    whir_round: T,
    query_idx: T,
    coset_idx: T,
    // Flags
    is_first_in_proof: T,
    is_first_in_round: T,
    is_first_in_query: T,
    merkle_idx_bit_src: T,
    // TODO: extract
    zi_root: T,
    zi: T,
    twiddle: T,
    value: [T; D_EF],
    value_hash: [T; WIDTH],
    yi: [T; D_EF],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::whir) struct NonInitialOpenedValueRecord {
    pub value: [F; D_EF],
    pub value_hash: [F; WIDTH],
}

pub struct NonInitialOpenedValuesAir {
    pub verify_query_bus: VerifyQueryBus,
    pub folding_bus: WhirFoldingBus,
    pub poseidon_bus: Poseidon2Bus,
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
    <AB::Expr as FieldAlgebra>::F: TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &NonInitialOpenedValuesCols<AB::Var> = (*local).borrow();
        let next: &NonInitialOpenedValuesCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_enabled);
        builder
            .when(local.is_first_in_proof)
            .assert_one(local.is_enabled);
        builder
            .when(local.is_first_in_round)
            .assert_one(local.is_enabled);
        builder
            .when(local.is_first_in_query)
            .assert_one(local.is_enabled);

        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        let is_same_round = next.is_enabled - next.is_first_in_round;
        let is_same_query = next.is_enabled - next.is_first_in_query;

        let max_coset_idx = AB::Expr::from_canonical_usize((1 << self.k) - 1);
        builder
            .when(local.is_enabled - is_same_query.clone())
            .assert_eq(local.coset_idx, max_coset_idx);

        NestedForLoopSubAir.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_enabled,
                        counter: [
                            local.proof_idx,
                            local.whir_round,
                            local.query_idx,
                            local.coset_idx,
                        ],
                        is_first: [
                            local.is_first_in_proof,
                            local.is_first_in_round,
                            local.is_first_in_query,
                            local.is_enabled,
                        ],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_enabled,
                        counter: [
                            next.proof_idx,
                            next.whir_round,
                            next.query_idx,
                            next.coset_idx,
                        ],
                        is_first: [
                            next.is_first_in_proof,
                            next.is_first_in_round,
                            next.is_first_in_query,
                            next.is_enabled,
                        ],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols {
                    is_transition: [
                        is_same_proof.clone(),
                        is_same_round.clone(),
                        is_same_query.clone(),
                    ],
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

        let omega_k = AB::Expr::from_f(
            <<AB::Expr as FieldAlgebra>::F as TwoAdicField>::two_adic_generator(self.k),
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
                coset_size: AB::Expr::from_canonical_usize(1 << self.k),
                coset_idx: local.coset_idx.into(),
                twiddle: local.twiddle.into(),
                value: local.value.map(Into::into),
                z_final: local.zi.into(),
                y_final: local.yi.map(Into::into),
            },
            local.is_enabled,
        );

        let pre_state: [AB::Expr; WIDTH] = from_fn(|i| {
            if i < D_EF {
                local.value[i].into()
            } else {
                AB::Expr::ZERO
            }
        });
        self.poseidon_bus.lookup_key(
            builder,
            Poseidon2BusMessage {
                input: pre_state,
                output: local.value_hash.map(Into::into),
            },
            local.is_enabled,
        );

        self.merkle_verify_bus.send(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                value: from_fn(|i| local.value_hash[i].into()),
                merkle_idx: local.merkle_idx_bit_src.into(),
                // There are two parts: hashing leaves (depth k) and merkle proof
                total_depth: AB::Expr::from_canonical_usize(self.initial_log_domain_size + 1)
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

pub(in crate::whir) fn build_non_initial_opened_value_records(
    params: &SystemParams,
    proofs: &[&Proof],
) -> Vec<NonInitialOpenedValueRecord> {
    let perm = poseidon2_perm();
    let num_queries_per_round: Vec<usize> =
        params.whir.rounds.iter().map(|r| r.num_queries).collect();

    let mut records = Vec::new();
    for proof in proofs {
        for (whir_round, &num_queries) in num_queries_per_round.iter().enumerate().skip(1) {
            for query_idx in 0..num_queries {
                let opened_values =
                    proof.whir_proof.codeword_opened_values[whir_round - 1][query_idx].clone();

                for opened_value in opened_values {
                    let mut state = [F::ZERO; WIDTH];
                    state[..D_EF].copy_from_slice(opened_value.as_base_slice());
                    perm.permute_mut(&mut state);

                    let mut value = [F::ZERO; D_EF];
                    value.copy_from_slice(opened_value.as_base_slice());

                    records.push(NonInitialOpenedValueRecord {
                        value,
                        value_hash: state,
                    });
                }
            }
        }
    }

    records
}

#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    preflights: &[&Preflight],
    records: &[NonInitialOpenedValueRecord],
) -> RowMajorMatrix<F> {
    let params = &mvk.inner.params;

    let num_rounds = params.num_whir_rounds();
    let num_queries_per_round: Vec<usize> =
        params.whir.rounds.iter().map(|r| r.num_queries).collect();
    let k_whir = params.k_whir();
    let omega_k = F::two_adic_generator(k_whir);
    let rows_per_query = 1 << k_whir;

    let mut round_row_offsets = Vec::with_capacity(num_rounds);
    round_row_offsets.push(0usize);
    for &num_queries in num_queries_per_round.iter().skip(1) {
        let rows_this_round = num_queries * rows_per_query;
        round_row_offsets.push(round_row_offsets.last().unwrap() + rows_this_round);
    }
    let num_rows_per_proof = *round_row_offsets.last().unwrap();

    let num_valid_rows = num_rows_per_proof * preflights.len();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = NonInitialOpenedValuesCols::<F>::width();

    debug_assert_eq!(records.len(), num_valid_rows);
    let mut trace = vec![F::ZERO; num_rows * width];

    trace
        .par_chunks_exact_mut(width)
        .zip(records)
        .enumerate()
        .for_each(|(row_idx, (row, record))| {
            let proof_idx = row_idx / num_rows_per_proof;
            let i = row_idx % num_rows_per_proof;

            let preflight = &preflights[proof_idx];

            let cols: &mut NonInitialOpenedValuesCols<F> = row.borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_canonical_usize(proof_idx);

            let round_minus_1 = round_row_offsets[1..].partition_point(|&offset| offset <= i);
            let whir_round = round_minus_1 + 1;
            let row_in_round = i - round_row_offsets[round_minus_1];

            let coset_idx = row_in_round % rows_per_query;
            let query_idx = row_in_round / rows_per_query;

            let is_first_in_proof = i == 0;
            let is_first_in_query = coset_idx == 0;
            let is_first_in_round = is_first_in_query && query_idx == 0;

            cols.whir_round = F::from_canonical_usize(whir_round);
            cols.query_idx = F::from_canonical_usize(query_idx);
            cols.coset_idx = F::from_canonical_usize(coset_idx);
            cols.twiddle = omega_k.exp_u64(coset_idx as u64);
            cols.is_first_in_proof = F::from_bool(is_first_in_proof);
            cols.is_first_in_round = F::from_bool(is_first_in_round);
            cols.is_first_in_query = F::from_bool(is_first_in_query);
            cols.zi_root = preflight.whir.zj_roots[whir_round][query_idx];
            cols.value_hash = record.value_hash;
            cols.value.copy_from_slice(&record.value);
            let query_offset = preflight.whir.query_offsets[whir_round];
            cols.merkle_idx_bit_src = preflight.whir.queries[query_offset + query_idx];
            cols.zi = preflight.whir.zjs[whir_round][query_idx];
            cols.yi
                .copy_from_slice(preflight.whir.yjs[whir_round][query_idx].as_base_slice());
        });

    RowMajorMatrix::new(trace, width)
}
