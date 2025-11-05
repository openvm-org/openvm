use core::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    Field, FieldAlgebra, FieldExtensionAlgebra, TwoAdicField, extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::SystemParams,
    poseidon2::{CHUNK, WIDTH},
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        MerkleVerifyBus, MerkleVerifyBusMessage, Poseidon2Bus, StackingIndexMessage,
        StackingIndicesBus,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{assert_eq_array, ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
    whir::bus::{VerifyQueryBus, VerifyQueryBusMessage, WhirFoldingBus, WhirFoldingBusMessage},
};

// (proof_idx, query_idx, coset_idx, commit_idx, col_chunk_idx)
#[repr(C)]
#[derive(AlignedBorrow)]
struct InitialOpenedValuesCols<T> {
    proof_idx: T,
    query_idx: T,
    commit_idx: T,
    coset_idx: T,
    col_chunk_idx: T,
    is_first_in_proof: T,
    is_first_in_query: T,
    is_first_in_commit: T,
    is_first_in_coset: T,
    is_same_proof: T,
    is_same_query: T,
    is_same_commit: T,
    is_same_coset_idx: T,
    coset_idx_max_aux: T,
    flags: [T; CHUNK],
    codeword_value_acc: [T; 4],
    // TODO: reduce number of mu pows
    mu_pows: [[T; 4]; CHUNK],
    mu: [T; 4],
    pre_state: [T; WIDTH],
    post_state: [T; WIDTH],
    // TODO: consider removing these from this AIR and passing them directly to `WhirFoldingAir`.
    twiddle: T,
    zi_root: T,
    zi: T,
    yi: [T; D_EF],
    merkle_idx_bit_src: T,
}

pub struct InitialOpenedValuesAir {
    pub stacking_indices_bus: StackingIndicesBus,
    pub verify_query_bus: VerifyQueryBus,
    pub folding_bus: WhirFoldingBus,
    pub _poseidon_bus: Poseidon2Bus,
    pub merkle_verify_bus: MerkleVerifyBus,
    pub initial_log_domain_size: usize,
    pub k: usize,
}

impl BaseAirWithPublicValues<F> for InitialOpenedValuesAir {}
impl PartitionedBaseAir<F> for InitialOpenedValuesAir {}

impl<F> BaseAir<F> for InitialOpenedValuesAir {
    fn width(&self) -> usize {
        InitialOpenedValuesCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for InitialOpenedValuesAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &InitialOpenedValuesCols<AB::Var> = (*local).borrow();
        let next: &InitialOpenedValuesCols<AB::Var> = (*next).borrow();
        let omega_k = AB::Expr::from_f(
            <<AB::Expr as FieldAlgebra>::F as TwoAdicField>::two_adic_generator(self.k),
        );

        NestedForLoopSubAir::<5, 4>.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.flags[0].into(),
                        counter: [
                            local.proof_idx.into(),
                            local.query_idx.into(),
                            local.coset_idx.into(),
                            local.commit_idx.into(),
                            local.col_chunk_idx.into(),
                        ],
                        is_first: [
                            local.is_first_in_proof.into(),
                            local.is_first_in_query.into(),
                            local.is_first_in_coset.into(),
                            local.is_first_in_commit.into(),
                            AB::Expr::ONE,
                        ],
                    },
                    NestedForLoopIoCols {
                        is_enabled: next.flags[0].into(),
                        counter: [
                            next.proof_idx.into(),
                            next.query_idx.into(),
                            next.coset_idx.into(),
                            next.commit_idx.into(),
                            next.col_chunk_idx.into(),
                        ],
                        is_first: [
                            next.is_first_in_proof.into(),
                            next.is_first_in_query.into(),
                            next.is_first_in_coset.into(),
                            next.is_first_in_commit.into(),
                            AB::Expr::ONE,
                        ],
                    },
                ),
                NestedForLoopAuxCols {
                    is_transition: [
                        local.is_same_proof,
                        local.is_same_query,
                        local.is_same_coset_idx,
                        local.is_same_commit,
                    ],
                }
                .map_into(),
            ),
        );

        let is_enabled = local.flags[0];
        for flag in local.flags {
            builder.assert_bool(flag);
        }
        for i in 0..CHUNK - 1 {
            builder.when(local.flags[i + 1]).assert_one(local.flags[i]);
        }

        let mut chunk_len = AB::Expr::ZERO;
        let mut codeword_value_slice_acc = local.codeword_value_acc.map(Into::into);

        assert_eq_array(&mut builder.when(local.is_same_proof), local.mu, next.mu);
        assert_array_eq(
            &mut builder.when(local.is_first_in_coset),
            local.mu_pows[0],
            [AB::F::ONE, AB::F::ZERO, AB::F::ZERO, AB::F::ZERO],
        );
        assert_array_eq(
            &mut builder.when(local.is_first_in_coset),
            local.codeword_value_acc,
            [AB::F::ZERO; 4],
        );

        let diff = AB::Expr::from_canonical_usize((1 << self.k) - 1) - local.coset_idx;
        builder
            .when(is_enabled)
            .when(diff.clone())
            .assert_one(diff.clone() * local.coset_idx_max_aux);
        let is_coset_idx_max = AB::Expr::ONE - diff * local.coset_idx_max_aux;
        builder
            .when(is_coset_idx_max * (is_enabled - local.is_same_coset_idx))
            .assert_zero(next.coset_idx);

        for i in 0..CHUNK {
            if i < CHUNK - 1 {
                assert_array_eq(
                    &mut builder.when(local.flags[i + 1]),
                    local.mu_pows[i + 1],
                    ext_field_multiply(local.mu, local.mu_pows[i]),
                );
                assert_array_eq(
                    &mut builder.when(AB::Expr::ONE - local.flags[i + 1]),
                    local.mu_pows[i + 1],
                    local.mu_pows[i],
                );
            } else {
                assert_eq_array(
                    &mut builder.when(local.is_same_coset_idx),
                    next.mu_pows[0],
                    ext_field_multiply(local.mu, local.mu_pows[CHUNK - 1]),
                );
            }

            builder
                .when(AB::Expr::ONE - local.flags[i])
                .assert_zero(local.pre_state[i]);

            // !local.flags[i] => pre_state[i] == 0, so this leaves the
            // accumulator unchanged on invalid rows.
            codeword_value_slice_acc = ext_field_add(
                codeword_value_slice_acc,
                ext_field_multiply_scalar(local.mu_pows[i], local.pre_state[i]),
            );

            builder
                .when(local.is_first_in_commit)
                .assert_zero(next.pre_state[CHUNK + i]);
            builder
                .when(local.is_same_commit)
                .assert_eq(next.pre_state[CHUNK + i], local.post_state[CHUNK + i]);

            let col_idx = local.col_chunk_idx * AB::Expr::from_canonical_usize(CHUNK)
                + AB::Expr::from_canonical_usize(i);

            self.stacking_indices_bus.receive(
                builder,
                local.proof_idx,
                StackingIndexMessage {
                    commit_idx: local.commit_idx.into(),
                    col_idx: col_idx.clone(),
                },
                local.flags[i],
            );

            chunk_len += local.flags[i].into();
        }
        self.verify_query_bus.receive(
            builder,
            local.proof_idx,
            VerifyQueryBusMessage {
                whir_round: AB::Expr::ZERO,
                query_idx: local.query_idx.into(),
                merkle_idx_bit_src: local.merkle_idx_bit_src.into(),
                zi_root: local.zi_root.into(),
                zi: local.zi.into(),
                yi: local.yi.map(Into::into),
            },
            local.is_first_in_query,
        );
        assert_array_eq(
            &mut builder.when(local.is_same_coset_idx),
            codeword_value_slice_acc.clone(),
            next.codeword_value_acc,
        );

        // self.poseidon_bus.lookup_key(
        //     builder,
        //     Poseidon2BusMessage {
        //         input: local.pre_state,
        //         output: local.post_state,
        //     },
        //     local.is_enabled,
        // );

        let is_last_in_commit = is_enabled - local.is_same_commit;
        self.merkle_verify_bus.send(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                merkle_idx: local.merkle_idx_bit_src.into(),
                total_depth: AB::Expr::from_canonical_usize(self.initial_log_domain_size + 1),
                height: AB::Expr::ZERO,
                leaf_sub_idx: local.coset_idx.into(),
                value: array::from_fn(|i| local.post_state[i].into()),
                commit_major: AB::Expr::ZERO,
                commit_minor: local.commit_idx.into(),
            },
            is_last_in_commit,
        );

        builder
            .when(local.is_first_in_proof)
            .assert_eq(local.twiddle, AB::Expr::ONE);
        builder
            .when(local.is_same_coset_idx)
            .assert_eq(next.twiddle, local.twiddle);
        builder
            .when((is_enabled - local.is_same_coset_idx) * next.flags[0])
            .assert_eq(next.twiddle, local.twiddle * omega_k);

        builder
            .when(local.is_same_query)
            .assert_eq(next.zi_root, local.zi_root);
        builder
            .when(local.is_same_query)
            .assert_eq(next.zi, local.zi);
        assert_array_eq(&mut builder.when(local.is_same_query), next.yi, local.yi);
        self.folding_bus.send(
            builder,
            local.proof_idx,
            WhirFoldingBusMessage {
                whir_round: AB::Expr::ZERO,
                query_idx: local.query_idx.into(),
                height: AB::Expr::ZERO,
                coset_shift: local.zi_root.into(),
                coset_idx: local.coset_idx.into(),
                coset_size: AB::Expr::from_canonical_usize(1 << self.k),
                twiddle: local.twiddle.into(),
                value: codeword_value_slice_acc,
                z_final: local.zi.into(),
                y_final: local.yi.map(Into::into),
            },
            is_enabled - local.is_same_coset_idx,
        );
    }
}

pub(in crate::whir) struct InitialOpenedValueRecord {
    pub proof_idx: usize,
    pub query_idx: usize,
    pub commit_idx: usize,
    pub chunk_idx: usize,
    pub chunk_len: usize,
    pub coset_idx: usize,
    pub mu_pow: EF,
    pub codeword_slice_val_acc: EF,
    pub pre_state: [F; WIDTH],
    pub post_state: [F; WIDTH],
}

pub(crate) fn generate_trace(
    params: SystemParams,
    proofs: &[Proof],
    preflights: &[Preflight],
    records: &[InitialOpenedValueRecord],
) -> RowMajorMatrix<F> {
    debug_assert_eq!(proofs.len(), preflights.len());

    let k_whir = params.k_whir;

    let omega_k = F::two_adic_generator(k_whir);
    let height = records.len().next_power_of_two();
    let width = InitialOpenedValuesCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];

    let num_valid_rows = records.len();
    trace
        .par_chunks_exact_mut(width)
        .take(num_valid_rows)
        .zip(records)
        .for_each(|(row, record)| {
            let proof_idx = record.proof_idx;
            let proof = &proofs[proof_idx];
            let preflight = &preflights[proof_idx];
            let mu = preflight.stacking.stacking_batching_challenge;

            let cols: &mut InitialOpenedValuesCols<F> = row.borrow_mut();

            let is_first_in_commit = record.chunk_idx == 0;
            let is_first_in_coset = is_first_in_commit && record.commit_idx == 0;
            let is_first_in_query = is_first_in_coset && record.coset_idx == 0;
            let is_first_in_proof = is_first_in_query && record.query_idx == 0;

            let num_chunks = proof.whir_proof.initial_round_opened_rows[record.commit_idx]
                [record.query_idx][record.coset_idx]
                .len()
                .div_ceil(CHUNK);
            let is_same_commit = record.chunk_idx < num_chunks - 1;
            let is_same_coset_idx = is_same_commit
                || record.commit_idx < proof.whir_proof.initial_round_opened_rows.len() - 1;
            let is_same_query = is_same_coset_idx || record.coset_idx < (1 << k_whir) - 1;
            let is_same_proof = is_same_query || record.query_idx < params.num_whir_queries - 1;

            cols.proof_idx = F::from_canonical_usize(proof_idx);
            cols.is_first_in_proof = F::from_bool(is_first_in_proof);
            cols.is_first_in_query = F::from_bool(is_first_in_query);
            cols.is_first_in_coset = F::from_bool(is_first_in_coset);
            cols.is_first_in_commit = F::from_bool(is_first_in_commit);
            cols.query_idx = F::from_canonical_usize(record.query_idx);
            cols.commit_idx = F::from_canonical_usize(record.commit_idx);
            cols.col_chunk_idx = F::from_canonical_usize(record.chunk_idx);
            cols.coset_idx = F::from_canonical_usize(record.coset_idx);
            for flag in cols.flags.iter_mut().take(record.chunk_len) {
                *flag = F::ONE;
            }
            cols.twiddle = omega_k.exp_u64(record.coset_idx as u64);
            cols.codeword_value_acc
                .copy_from_slice(record.codeword_slice_val_acc.as_base_slice());
            cols.zi_root = preflight.whir.zj_roots[0][record.query_idx];
            cols.zi = preflight.whir.zjs[0][record.query_idx];
            cols.yi
                .copy_from_slice(preflight.whir.yjs[0][record.query_idx].as_base_slice());
            cols.mu.copy_from_slice(
                preflight
                    .stacking
                    .stacking_batching_challenge
                    .as_base_slice(),
            );
            cols.merkle_idx_bit_src = preflight.whir.queries[record.query_idx];
            let mut mu_pow = record.mu_pow;
            for offset in 0..CHUNK {
                cols.mu_pows[offset].copy_from_slice(mu_pow.as_base_slice());
                if offset < record.chunk_len - 1 {
                    mu_pow *= mu;
                }
            }
            cols.pre_state = record.pre_state;
            cols.post_state = record.post_state;
            cols.coset_idx_max_aux = F::from_canonical_usize((1 << k_whir) - 1 - record.coset_idx)
                .try_inverse()
                .unwrap_or_default();

            cols.is_same_proof = F::from_bool(is_same_proof);
            cols.is_same_query = F::from_bool(is_same_query);
            cols.is_same_commit = F::from_bool(is_same_commit);
            cols.is_same_coset_idx = F::from_bool(is_same_coset_idx);
        });

    trace
        .par_chunks_exact_mut(width)
        .skip(num_valid_rows)
        .for_each(|row| {
            let cols: &mut InitialOpenedValuesCols<F> = row.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len());
        });

    RowMajorMatrix::new(trace, width)
}
