use core::{
    array,
    borrow::{Borrow, BorrowMut},
    cmp::min,
};

use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    interaction::InteractionBuilder, keygen::types::MultiStarkVerifyingKey, proof::Proof,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, CHUNK, D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        MerkleVerifyBus, MerkleVerifyBusMessage, Poseidon2PermuteBus, Poseidon2PermuteMessage,
        StackingIndexMessage, StackingIndicesBus, WhirMuBus, WhirMuMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    tracegen::RowMajorChip,
    utils::{ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
    whir::{
        bus::{VerifyQueryBus, VerifyQueryBusMessage, WhirFoldingBus, WhirFoldingBusMessage},
        WhirBlobCpu,
    },
};

// (proof_idx, query_idx, coset_idx, commit_idx, col_chunk_idx)
#[repr(C)]
#[derive(AlignedBorrow)]
pub(in crate::whir::initial_opened_values) struct InitialOpenedValuesCols<T> {
    proof_idx: T,
    query_idx: T,
    commit_idx: T,
    coset_idx: T,
    col_chunk_idx: T,
    is_first_in_proof: T,
    is_first_in_query: T,
    is_first_in_commit: T,
    is_first_in_coset: T,
    flags: [T; CHUNK],
    codeword_value_acc: [T; 4],
    codeword_value_next_acc: [T; 4],
    /// Stores clamped even powers for the current chunk base exponent `b`:
    /// mu^(min(b + 2k, opened_row_len - 1)) for k in [0, CHUNK / 2).
    mu_pows_even_clamped: [[T; 4]; CHUNK / 2],
    /// Stores mu^(min(b + CHUNK - 1, opened_row_len - 1)) for the current
    /// chunk base exponent `b`.
    /// Here opened_row_len is the opened row length for the current commit.
    mu_pow_last_clamped: [T; 4],
    mu: [T; 4],
    pre_state: [T; POSEIDON2_WIDTH],
    post_state: [T; POSEIDON2_WIDTH],
    twiddle: T,
    zi_root: T,
    zi: T,
    yi: [T; D_EF],
    merkle_idx_bit_src: T,
}

pub struct InitialOpenedValuesAir {
    pub stacking_indices_bus: StackingIndicesBus,
    pub whir_mu_bus: WhirMuBus,
    pub verify_query_bus: VerifyQueryBus,
    pub folding_bus: WhirFoldingBus,
    pub poseidon_permute_bus: Poseidon2PermuteBus,
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
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        BinomiallyExtendable<{ D_EF }> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &InitialOpenedValuesCols<AB::Var> = (*local).borrow();
        let next: &InitialOpenedValuesCols<AB::Var> = (*next).borrow();
        let omega_k = AB::Expr::from_prime_subfield(
            <<AB::Expr as PrimeCharacteristicRing>::PrimeSubfield as TwoAdicField>::two_adic_generator(
                self.k,
            ),
        );

        let is_same_proof = next.flags[0] - next.is_first_in_proof;
        let is_same_query = next.flags[0] - next.is_first_in_query;
        let is_same_coset_idx = next.flags[0] - next.is_first_in_coset;
        let is_same_commit = next.flags[0] - next.is_first_in_commit;

        NestedForLoopSubAir::<5>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.flags[0],
                    counter: [
                        local.proof_idx,
                        local.query_idx,
                        local.coset_idx,
                        local.commit_idx,
                        local.col_chunk_idx,
                    ],
                    is_first: [
                        local.is_first_in_proof,
                        local.is_first_in_query,
                        local.is_first_in_coset,
                        local.is_first_in_commit,
                        local.flags[0],
                    ],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.flags[0],
                    counter: [
                        next.proof_idx,
                        next.query_idx,
                        next.coset_idx,
                        next.commit_idx,
                        next.col_chunk_idx,
                    ],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_query,
                        next.is_first_in_coset,
                        next.is_first_in_commit,
                        next.flags[0],
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

        self.whir_mu_bus.receive(
            builder,
            local.proof_idx,
            WhirMuMessage {
                mu: local.mu.map(Into::into),
            },
            local.is_first_in_proof,
        );

        assert_array_eq(&mut builder.when(is_same_proof.clone()), local.mu, next.mu);
        assert_array_eq(
            &mut builder.when(local.is_first_in_coset),
            local.mu_pows_even_clamped[0],
            [AB::F::ONE, AB::F::ZERO, AB::F::ZERO, AB::F::ZERO],
        );
        assert_array_eq(
            &mut builder.when(local.is_first_in_coset),
            local.codeword_value_acc,
            [AB::F::ZERO; 4],
        );

        builder
            .when(is_enabled - is_same_query.clone())
            .assert_eq(local.coset_idx, AB::Expr::from_usize((1 << self.k) - 1));

        let select = |cond: AB::Expr, a: [AB::Expr; D_EF], b: [AB::Expr; D_EF]| {
            array::from_fn(|j| a[j].clone() + cond.clone() * (b[j].clone() - a[j].clone()))
        };

        // Even-power recurrence.
        // Starting from mu^(min(b + 2k, opened_row_len - 1)):
        // - apply +1 if flags[2k+1] = 1
        // - apply another +1 if flags[2k+2] = 1
        for k in 0..CHUNK / 2 - 1 {
            let p0 = local.mu_pows_even_clamped[k].map(Into::into);
            let p1 = ext_field_multiply(local.mu, p0.clone());
            let p2 = ext_field_multiply(local.mu, p1.clone());
            let p_after_odd = select(local.flags[2 * k + 1].into(), p0, p1);
            let p_after_even = select(local.flags[2 * k + 2].into(), p_after_odd, p2);
            assert_array_eq(builder, local.mu_pows_even_clamped[k + 1], p_after_even);
        }

        // mu_pow_last_clamped stores mu^(min(b + CHUNK - 1, opened_row_len - 1)).
        // Starting from mu_pows_even_clamped[last] = mu^(min(b + CHUNK - 2, opened_row_len - 1)),
        // apply one final +1 step iff the last slot is valid.
        {
            let last = CHUNK / 2 - 1;
            let p_last = local.mu_pows_even_clamped[last].map(Into::into);
            let p_last_next = ext_field_multiply(local.mu, p_last.clone());
            let expected_last = select(local.flags[CHUNK - 1].into(), p_last, p_last_next);
            assert_array_eq(builder, local.mu_pow_last_clamped, expected_last);
        }

        assert_array_eq(
            &mut builder.when(is_same_coset_idx.clone()),
            next.mu_pows_even_clamped[0],
            ext_field_multiply(local.mu, local.mu_pow_last_clamped),
        );

        // For degree reasons, odd entries intentionally use the raw
        // mu * mu_pows_even_clamped[i/2] form (unclamped). This is safe
        // because each contribution is masked by flags[i].
        // - even i: mu^(min(b + i, opened_row_len - 1))
        // - odd i: raw mu * mu_pows_even_clamped[i/2]
        // The clamped tail power mu_pow_last_clamped is still used for cross-row
        // chaining into next.mu_pows_even_clamped[0].
        let mu_pows_acc_odd_unclamped: [[AB::Expr; D_EF]; CHUNK] = array::from_fn(|i| {
            if i % 2 == 0 {
                local.mu_pows_even_clamped[i / 2].map(Into::into)
            } else {
                ext_field_multiply(local.mu, local.mu_pows_even_clamped[i / 2])
            }
        });

        let mut codeword_value_slice_acc: [AB::Expr; D_EF] =
            local.codeword_value_acc.map(Into::into);
        for (i, mu_pow_acc_odd_unclamped) in mu_pows_acc_odd_unclamped.iter().enumerate() {
            builder
                .when(is_same_commit.clone())
                .when(AB::Expr::ONE - next.flags[i])
                .assert_eq(local.post_state[i], next.pre_state[i]);

            // flags[i] masks invalid slots, so those contribute zero regardless
            // of pre_state[i].
            let contribution: [AB::Expr; D_EF] = ext_field_multiply_scalar(
                mu_pow_acc_odd_unclamped.clone(),
                local.pre_state[i] * local.flags[i],
            );
            codeword_value_slice_acc = ext_field_add(codeword_value_slice_acc, contribution);

            builder
                .when(local.is_first_in_commit)
                .assert_zero(local.pre_state[CHUNK + i]);
            builder
                .when(is_same_commit.clone())
                .assert_eq(next.pre_state[CHUNK + i], local.post_state[CHUNK + i]);

            let col_idx =
                local.col_chunk_idx * AB::Expr::from_usize(CHUNK) + AB::Expr::from_usize(i);

            self.stacking_indices_bus.lookup_key(
                builder,
                local.proof_idx,
                StackingIndexMessage {
                    commit_idx: local.commit_idx.into(),
                    col_idx: col_idx.clone(),
                },
                local.flags[i],
            );
        }

        assert_array_eq(
            builder,
            local.codeword_value_next_acc,
            codeword_value_slice_acc,
        );
        assert_array_eq(
            &mut builder.when(is_same_coset_idx.clone()),
            next.codeword_value_acc,
            local.codeword_value_next_acc,
        );

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

        self.poseidon_permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: local.pre_state,
                output: local.post_state,
            },
            is_enabled,
        );

        let is_last_in_commit = is_enabled - is_same_commit.clone();
        self.merkle_verify_bus.send(
            builder,
            local.proof_idx,
            MerkleVerifyBusMessage {
                merkle_idx: local.merkle_idx_bit_src.into(),
                total_depth: AB::Expr::from_usize(self.initial_log_domain_size + 1),
                height: AB::Expr::ZERO,
                leaf_sub_idx: local.coset_idx.into(),
                value: array::from_fn(|i| local.post_state[i].into()),
                commit_major: AB::Expr::ZERO,
                commit_minor: local.commit_idx.into(),
            },
            is_last_in_commit,
        );

        builder
            .when(local.is_first_in_query)
            .assert_eq(local.twiddle, AB::Expr::ONE);
        builder
            .when(is_same_coset_idx.clone())
            .assert_eq(next.twiddle, local.twiddle);
        builder
            .when((is_enabled - is_same_coset_idx.clone()) * next.flags[0])
            .assert_eq(next.twiddle, local.twiddle * omega_k);

        builder
            .when(is_same_query.clone())
            .assert_eq(next.zi_root, local.zi_root);
        builder
            .when(is_same_query.clone())
            .assert_eq(next.zi, local.zi);
        assert_array_eq(&mut builder.when(is_same_query.clone()), next.yi, local.yi);
        self.folding_bus.send(
            builder,
            local.proof_idx,
            WhirFoldingBusMessage {
                whir_round: AB::Expr::ZERO,
                query_idx: local.query_idx.into(),
                height: AB::Expr::ZERO,
                coset_shift: local.zi_root.into(),
                coset_idx: local.coset_idx.into(),
                coset_size: AB::Expr::from_usize(1 << self.k),
                twiddle: local.twiddle.into(),
                value: local.codeword_value_next_acc.map(Into::into),
                z_final: local.zi.into(),
                y_final: local.yi.map(Into::into),
            },
            is_enabled - is_same_coset_idx.clone(),
        );
    }
}

pub(crate) struct InitialOpenedValuesTraceGenerator;

pub(crate) struct InitialOpenedValuesCtx<'a> {
    pub vk: &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    pub proofs: &'a [&'a Proof<BabyBearPoseidon2Config>],
    pub preflights: &'a [&'a Preflight],
    pub blob: &'a WhirBlobCpu,
}

impl RowMajorChip<F> for InitialOpenedValuesTraceGenerator {
    type Ctx<'a> = InitialOpenedValuesCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let params = &ctx.vk.inner.params;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        let blob = ctx.blob;
        let zi_roots = &blob.zi_roots;
        let zis = &blob.zis;
        let yis = &blob.yis;
        let accs_layout = blob.codeword_value_accs.layout();
        let codeword_value_accs = blob.codeword_value_accs.as_slice();
        debug_assert_eq!(proofs.len(), preflights.len());

        let k_whir = params.k_whir();

        let omega_k = F::two_adic_generator(k_whir);
        let num_valid_rows = codeword_value_accs.len();
        let height = if let Some(h) = required_height {
            if h < num_valid_rows {
                return None;
            }
            h
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = InitialOpenedValuesCols::<F>::width();
        let mut trace = vec![F::ZERO; height * width];
        let mu_pows = &blob.mu_pows;
        trace
            .par_chunks_exact_mut(width)
            .take(num_valid_rows)
            .zip(codeword_value_accs)
            .enumerate()
            .for_each(|(row_idx, (row, &codeword_value_acc))| {
                let (proof_idx, query_idx, coset_idx, commit_idx, chunk_idx) =
                    accs_layout.decompose(row_idx);
                let preflight = &preflights[proof_idx];
                let chunk_len = accs_layout.chunk_len(commit_idx, chunk_idx);

                let mu = preflight.stacking.stacking_batching_challenge;

                let cols: &mut InitialOpenedValuesCols<F> = row.borrow_mut();

                let is_first_in_commit = chunk_idx == 0;
                let is_first_in_coset = is_first_in_commit && commit_idx == 0;
                let is_first_in_query = is_first_in_coset && coset_idx == 0;
                let is_first_in_proof = is_first_in_query && query_idx == 0;

                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_first_in_proof = F::from_bool(is_first_in_proof);
                cols.is_first_in_query = F::from_bool(is_first_in_query);
                cols.is_first_in_coset = F::from_bool(is_first_in_coset);
                cols.is_first_in_commit = F::from_bool(is_first_in_commit);
                cols.query_idx = F::from_usize(query_idx);
                cols.commit_idx = F::from_usize(commit_idx);
                cols.col_chunk_idx = F::from_usize(chunk_idx);
                cols.coset_idx = F::from_usize(coset_idx);
                for flag in cols.flags.iter_mut().take(chunk_len) {
                    *flag = F::ONE;
                }
                cols.twiddle = omega_k.exp_u64(coset_idx as u64);
                cols.codeword_value_acc
                    .copy_from_slice(codeword_value_acc.as_basis_coefficients_slice());
                let query = (proof_idx, 0, query_idx);
                cols.zi_root = zi_roots[query];
                cols.zi = zis[query];
                cols.yi
                    .copy_from_slice(yis[query].as_basis_coefficients_slice());
                cols.mu.copy_from_slice(mu.as_basis_coefficients_slice());
                cols.merkle_idx_bit_src = preflight.whir.queries[query_idx];

                let exponent_base = accs_layout.commit_width_offset(commit_idx);
                let chunk_base = exponent_base + chunk_idx * CHUNK;

                // Fill mu_pows_even_clamped[k] = mu^(min(b + 2k, opened_row_len - 1)),
                // where b = exponent_base + chunk_idx*CHUNK.
                for k in 0..CHUNK / 2 {
                    let offset = 2 * k;
                    let exponent = chunk_base + min(offset, chunk_len - 1);
                    let mu_pow = mu_pows[(proof_idx, exponent)];
                    cols.mu_pows_even_clamped[k]
                        .copy_from_slice(mu_pow.as_basis_coefficients_slice());
                }
                let exponent = chunk_base + chunk_len - 1;
                let mu_pow = mu_pows[(proof_idx, exponent)];
                cols.mu_pow_last_clamped
                    .copy_from_slice(mu_pow.as_basis_coefficients_slice());

                let states = &preflight.initial_row_states[commit_idx][query_idx][coset_idx];
                let opened_row = &proofs[proof_idx].whir_proof.initial_round_opened_rows
                    [commit_idx][query_idx][coset_idx];
                let chunk_start = chunk_idx * CHUNK;

                cols.pre_state = if chunk_idx > 0 {
                    states[chunk_idx - 1]
                } else {
                    [F::ZERO; POSEIDON2_WIDTH]
                };
                cols.pre_state[..chunk_len]
                    .copy_from_slice(&opened_row[chunk_start..chunk_start + chunk_len]);
                cols.post_state = states[chunk_idx];

                let mut next_acc = cols.codeword_value_acc;
                for i in 0..chunk_len {
                    let exponent = chunk_base + i;
                    let mu_pow_coeffs: &[F] =
                        mu_pows[(proof_idx, exponent)].as_basis_coefficients_slice();
                    let scalar = cols.pre_state[i];
                    for j in 0..D_EF {
                        next_acc[j] += mu_pow_coeffs[j] * scalar;
                    }
                }
                cols.codeword_value_next_acc = next_acc;
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
