use core::borrow::{Borrow, BorrowMut};
use std::array::from_fn;

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::Permutation;
use stark_backend_v2::{
    D_EF, F,
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
struct NonInitialOpenedValuesCols<T> {
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
    is_same_proof: T,
    is_same_round: T,
    is_same_query: T,
    merkle_idx_bit_src: T,
    coset_idx_max_aux: T,
    // TODO: extract
    zi_root: T,
    zi: T,
    twiddle: T,
    value: [T; D_EF],
    value_hash: [T; WIDTH],
    yi: [T; D_EF],
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

        let max_coset_idx = AB::Expr::from_canonical_usize((1 << self.k) - 1);
        let coset_idx_diff = max_coset_idx - local.coset_idx;
        let is_not_last_in_query = coset_idx_diff.clone() * local.coset_idx_max_aux;
        builder
            .when(local.is_enabled * coset_idx_diff.clone())
            .assert_one(is_not_last_in_query.clone());
        builder
            .when(local.is_enabled - is_not_last_in_query)
            .assert_zero(next.coset_idx);

        NestedForLoopSubAir::<4, 3>.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_enabled.into(),
                        counter: [
                            local.proof_idx.into(),
                            local.whir_round.into(),
                            local.query_idx.into(),
                            local.coset_idx.into(),
                        ],
                        is_first: [
                            local.is_first_in_proof.into(),
                            local.is_first_in_round.into(),
                            local.is_first_in_query.into(),
                            AB::Expr::ONE,
                        ],
                    },
                    NestedForLoopIoCols {
                        is_enabled: next.is_enabled.into(),
                        counter: [
                            next.proof_idx.into(),
                            next.whir_round.into(),
                            next.query_idx.into(),
                            next.coset_idx.into(),
                        ],
                        is_first: [
                            next.is_first_in_proof.into(),
                            next.is_first_in_round.into(),
                            next.is_first_in_query.into(),
                            AB::Expr::ONE,
                        ],
                    },
                ),
                NestedForLoopAuxCols {
                    is_transition: [
                        local.is_same_proof,
                        local.is_same_round,
                        local.is_same_query,
                    ],
                }
                .map_into(),
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
            .when(local.is_same_query)
            .assert_eq(next.twiddle, local.twiddle * omega_k);

        assert_array_eq(&mut builder.when(local.is_same_query), local.yi, next.yi);
        builder
            .when(local.is_same_query)
            .assert_eq(local.zi, next.zi);
        builder
            .when(local.is_same_query)
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

pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let params = mvk.inner.params;

    let num_rounds = params.num_whir_rounds();
    let num_queries = params.num_whir_queries;
    let k_whir = params.k_whir;
    let omega_k = F::two_adic_generator(k_whir);
    let rows_per_query = 1 << k_whir;
    let max_coset_idx = rows_per_query - 1;
    let rows_per_round = rows_per_query * num_queries;

    let num_rows_per_proof: usize = (num_rounds - 1) * rows_per_round;
    let num_valid_rows = num_rows_per_proof * proofs.len();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = NonInitialOpenedValuesCols::<F>::width();

    struct Record {
        value_hash: [F; WIDTH],
    }

    let perm = poseidon2_perm();

    let mut records = vec![];

    for proof in proofs {
        for whir_round in 1..params.num_whir_rounds() {
            for query_idx in 0..params.num_whir_queries {
                let opened_values =
                    proof.whir_proof.codeword_opened_values[whir_round - 1][query_idx].clone();

                for opened_value in opened_values {
                    let mut state = [F::ZERO; WIDTH];
                    state[..D_EF].copy_from_slice(opened_value.as_base_slice());
                    perm.permute_mut(&mut state);

                    records.push(Record { value_hash: state });
                }
            }
        }
    }

    let mut trace = vec![F::ZERO; num_rows * width];

    trace
        .par_chunks_exact_mut(width)
        .zip(records)
        .enumerate()
        .for_each(|(row_idx, (row, record))| {
            let proof_idx = row_idx / num_rows_per_proof;
            let i = row_idx % num_rows_per_proof;

            let proof = &proofs[proof_idx];
            let preflight = &preflights[proof_idx];

            let cols: &mut NonInitialOpenedValuesCols<F> = row.borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_canonical_usize(proof_idx);

            let coset_idx = i % rows_per_query;
            let query_idx = (i / rows_per_query) % num_queries;
            let whir_round = if rows_per_round == 0 {
                1
            } else {
                (i / rows_per_round) + 1
            };
            let row_in_proof = i;
            let is_first_in_proof = row_in_proof == 0;
            let is_last_in_proof = row_in_proof + 1 == num_rows_per_proof;
            let is_first_in_query = coset_idx == 0;
            let is_first_in_round = is_first_in_query && query_idx == 0;
            let is_last_in_query = coset_idx + 1 == rows_per_query;
            let is_last_query_in_round = query_idx + 1 == num_queries;
            let is_same_query = !is_last_in_proof && !is_last_in_query;
            let is_same_round = !is_last_in_proof && (!is_last_in_query || !is_last_query_in_round);

            cols.whir_round = F::from_canonical_usize(whir_round);
            cols.query_idx = F::from_canonical_usize(query_idx);
            cols.coset_idx = F::from_canonical_usize(coset_idx);
            cols.coset_idx_max_aux = F::from_canonical_usize(max_coset_idx - coset_idx)
                .try_inverse()
                .unwrap_or_default();
            cols.twiddle = omega_k.exp_u64(coset_idx as u64);
            cols.is_first_in_proof = F::from_bool(is_first_in_proof);
            cols.is_same_proof = F::from_bool(!is_last_in_proof);
            cols.is_first_in_round = F::from_bool(is_first_in_round);
            cols.is_first_in_query = F::from_bool(is_first_in_query);
            cols.is_same_round = F::from_bool(is_same_round);
            cols.is_same_query = F::from_bool(is_same_query);
            cols.zi_root = preflight.whir.zj_roots[whir_round][query_idx];
            cols.value.copy_from_slice(
                proof.whir_proof.codeword_opened_values[whir_round - 1][query_idx][coset_idx]
                    .as_base_slice(),
            );
            cols.value_hash = record.value_hash;
            cols.merkle_idx_bit_src = preflight.whir.queries[whir_round * num_queries + query_idx];
            cols.zi = preflight.whir.zjs[whir_round][query_idx];
            cols.yi
                .copy_from_slice(preflight.whir.yjs[whir_round][query_idx].as_base_slice());
        });

    trace
        .par_chunks_exact_mut(width)
        .skip(num_valid_rows)
        .for_each(|row| {
            let cols: &mut NonInitialOpenedValuesCols<F> = row.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len());
            cols.coset_idx_max_aux = F::from_canonical_usize(max_coset_idx)
                .try_inverse()
                .unwrap_or_default();
        });

    RowMajorMatrix::new(trace, width)
}
