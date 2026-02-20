use std::borrow::{Borrow, BorrowMut};

use itertools::{izip, Itertools};
use openvm_circuit_primitives::{
    utils::{and, assert_array_eq, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        StackingIndexMessage, StackingIndicesBus, TranscriptBus, TranscriptBusMessage,
        WhirModuleBus, WhirModuleMessage, WhirMuBus, WhirMuMessage,
    },
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    stacking::{
        bus::{
            ClaimCoefficientsBus, ClaimCoefficientsMessage, StackingModuleTidxBus,
            StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
        },
        utils::{compute_coefficients, get_stacked_slice_data},
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    tracegen::{RowMajorChip, StandardTracegenCtx},
    utils::{assert_one_ext, ext_field_add, ext_field_multiply, pow_tidx_count},
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct StackingClaimsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Correspond to stacking_claim
    pub commit_idx: F,
    pub stacked_col_idx: F,

    // Sampled transcript values
    pub tidx: F,
    pub mu: [F; D_EF],
    pub mu_pow: [F; D_EF],

    // μ PoW witness and sample for proof-of-work check
    pub mu_pow_witness: F,
    pub mu_pow_sample: F,

    // Stacking claim and batched coefficient computed in OpeningClaimsCols
    pub stacking_claim: [F; D_EF],
    pub claim_coefficient: [F; D_EF],

    // Sum of each stacking_claim * claim_coefficient
    pub final_s_eval: [F; D_EF],

    // RLC of stacking claims using mu
    pub whir_claim: [F; D_EF],
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct StackingClaimsAir {
    // External buses
    pub stacking_indices_bus: StackingIndicesBus,
    pub whir_module_bus: WhirModuleBus,
    pub whir_mu_bus: WhirMuBus,
    pub transcript_bus: TranscriptBus,
    pub exp_bits_len_bus: ExpBitsLenBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub claim_coefficients_bus: ClaimCoefficientsBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,

    pub stacking_index_mult: usize,
    /// Number of PoW bits for μ batching challenge.
    pub mu_pow_bits: usize,
}

impl BaseAirWithPublicValues<F> for StackingClaimsAir {}
impl PartitionedBaseAir<F> for StackingClaimsAir {}

impl<F> BaseAir<F> for StackingClaimsAir {
    fn width(&self) -> usize {
        StackingClaimsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for StackingClaimsAir
where
    AB::F: PrimeField32,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &StackingClaimsCols<AB::Var> = (*local).borrow();
        let next: &StackingClaimsCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1, 0> {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_zero((local.proof_idx + AB::F::ONE - next.proof_idx) * next.proof_idx);
        builder
            .when(and(not(local.is_valid), local.is_last))
            .assert_zero(next.proof_idx);
        builder.when(local.is_first).assert_one(local.is_valid);

        /*
         * Constrain that commit_idx and stacked_col_idx increment correctly.
         */
        builder.when(local.is_first).assert_zero(local.commit_idx);
        builder
            .when(local.is_first)
            .assert_zero(local.stacked_col_idx);

        builder.when(not(local.is_last)).assert_zero(
            (next.commit_idx - local.commit_idx - AB::Expr::ONE)
                * (next.stacked_col_idx - local.stacked_col_idx - AB::Expr::ONE)
                - not(local.is_valid),
        );

        self.stacking_indices_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            StackingIndexMessage {
                commit_idx: local.commit_idx,
                col_idx: local.stacked_col_idx,
            },
            local.is_valid * AB::Expr::from_usize(self.stacking_index_mult),
        );

        /*
         * Compute the running sum of stacking_claim * claim_coefficient values and then
         * constrain the final result to be equal to s_{n_stack}(u_{n_stack}), which is
         * sent from SumcheckRoundsAir.
         */
        self.claim_coefficients_bus.receive(
            builder,
            local.proof_idx,
            ClaimCoefficientsMessage {
                commit_idx: local.commit_idx,
                stacked_col_idx: local.stacked_col_idx,
                coefficient: local.claim_coefficient,
            },
            local.is_valid,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_multiply(local.stacking_claim, local.claim_coefficient),
            local.final_s_eval,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_add(
                ext_field_multiply(next.stacking_claim, next.claim_coefficient),
                local.final_s_eval,
            ),
            next.final_s_eval,
        );

        self.sumcheck_claims_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::TWO,
                value: local.final_s_eval.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Constrain transcript operations and send the final tidx to the WHIR module.
         */
        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::TWO,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        builder
            .when(not(local.is_last) * local.is_valid)
            .assert_eq(local.tidx + AB::F::from_usize(D_EF), next.tidx);

        let mu_pow_offset = pow_tidx_count(self.mu_pow_bits);

        for i in 0..D_EF {
            // Observe stacking_claim at tidx + 0..D_EF
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i) + local.tidx,
                    value: local.stacking_claim[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            // Sample μ at tidx + D_EF + mu_pow_offset + i (after μ PoW observe/sample if any)
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(i + D_EF + mu_pow_offset) + local.tidx,
                    value: local.mu[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                and(local.is_last, local.is_valid),
            );
        }

        if self.mu_pow_bits > 0 {
            // μ PoW: observe mu_pow_witness at tidx + D_EF (on last row only)
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(D_EF) + local.tidx,
                    value: local.mu_pow_witness.into(),
                    is_sample: AB::Expr::ZERO,
                },
                and(local.is_last, local.is_valid),
            );

            // μ PoW: sample mu_pow_sample at tidx + D_EF + 1 (on last row only)
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(D_EF + 1) + local.tidx,
                    value: local.mu_pow_sample.into(),
                    is_sample: AB::Expr::ONE,
                },
                and(local.is_last, local.is_valid),
            );

            // μ PoW check: g^{mu_pow_sample[0:mu_pow_bits]} = 1
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: AB::F::GENERATOR.into(),
                    bit_src: local.mu_pow_sample.into(),
                    num_bits: AB::Expr::from_usize(self.mu_pow_bits),
                    result: AB::Expr::ONE,
                },
                and(local.is_last, local.is_valid),
            );
        }

        /*
         * Compute the RLC of the stacking claims and send it to the WHIR module.
         */
        assert_one_ext(&mut builder.when(local.is_first), local.mu_pow);
        assert_array_eq(&mut builder.when(not(local.is_last)), local.mu, next.mu);
        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.mu, local.mu_pow),
            next.mu_pow,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.stacking_claim,
            local.whir_claim,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_add(
                ext_field_multiply(next.stacking_claim, next.mu_pow),
                local.whir_claim,
            ),
            next.whir_claim,
        );

        // Send to WHIR module with tidx after all transcript operations
        self.whir_module_bus.send(
            builder,
            local.proof_idx,
            WhirModuleMessage {
                tidx: AB::Expr::from_usize(2 * D_EF + mu_pow_offset) + local.tidx,
                claim: local.whir_claim.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );
        self.whir_mu_bus.send(
            builder,
            local.proof_idx,
            WhirMuMessage {
                mu: local.mu.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );
    }
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct StackingClaimsTraceGenerator;

impl RowMajorChip<F> for StackingClaimsTraceGenerator {
    type Ctx<'a> = StandardTracegenCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let vk = ctx.vk;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        debug_assert_eq!(proofs.len(), preflights.len());

        let width = StackingClaimsCols::<usize>::width();
        let minimum_height: usize = proofs.iter().fold(0, |acc, proof| {
            acc + proof
                .stacking_proof
                .stacking_openings
                .iter()
                .fold(0, |acc, openings| acc + openings.len())
        });
        let height = if let Some(height) = required_height {
            if height < minimum_height {
                return None;
            }
            height
        } else {
            minimum_height.next_power_of_two()
        };

        let mut trace = vec![F::ZERO; height * width];
        let mut chunks = trace.chunks_mut(width);

        for (proof_idx, (proof, preflight)) in proofs.iter().zip(preflights).enumerate() {
            let claims = proof
                .stacking_proof
                .stacking_openings
                .iter()
                .enumerate()
                .flat_map(|(commit_idx, openings)| {
                    openings
                        .iter()
                        .enumerate()
                        .map(move |(stacked_col_idx, opening)| {
                            (commit_idx, stacked_col_idx, opening)
                        })
                })
                .collect_vec();
            let stacked_slices =
                get_stacked_slice_data(vk, &preflight.proof_shape.sorted_trace_vdata);

            let coeffs = compute_coefficients(
                proof,
                &stacked_slices,
                &preflight.stacking.sumcheck_rnd,
                &preflight.batch_constraint.sumcheck_rnd,
                &preflight.stacking.lambda,
                vk.inner.params.l_skip,
                vk.inner.params.n_stack,
            )
            .0
            .into_iter()
            .flatten()
            .collect_vec();

            let num_rows = claims.len();
            let proof_idx_value = F::from_usize(proof_idx);

            let initial_tidx = preflight.stacking.intermediate_tidx[2];

            let mu = preflight.stacking.stacking_batching_challenge;
            let mu_pows = mu.powers().take(claims.len()).collect_vec();

            // μ PoW witness and sample from preflight
            let mu_pow_witness = preflight.stacking.mu_pow_witness;
            let mu_pow_sample = preflight.stacking.mu_pow_sample;

            let mut final_s_eval = EF::ZERO;
            let mut whir_claim = EF::ZERO;

            for (idx, (&(commit_idx, stacked_col_idx, &claim), coeff)) in
                izip!(&claims, coeffs).enumerate()
            {
                let chunk = chunks.next().unwrap();
                let cols: &mut StackingClaimsCols<F> = chunk.borrow_mut();

                cols.proof_idx = proof_idx_value;
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(idx == 0);
                cols.is_last = F::from_bool(idx + 1 == num_rows);

                cols.commit_idx = F::from_usize(commit_idx);
                cols.stacked_col_idx = F::from_usize(stacked_col_idx);

                cols.tidx = F::from_usize(initial_tidx + (D_EF * idx));
                cols.mu.copy_from_slice(mu.as_basis_coefficients_slice());
                cols.mu_pow
                    .copy_from_slice(mu_pows[idx].as_basis_coefficients_slice());

                cols.stacking_claim
                    .copy_from_slice(claim.as_basis_coefficients_slice());

                // μ PoW columns (only used on last row, but set for all rows for simplicity)
                cols.mu_pow_witness = mu_pow_witness;
                cols.mu_pow_sample = mu_pow_sample;

                cols.claim_coefficient
                    .copy_from_slice(coeff.as_basis_coefficients_slice());
                final_s_eval += claim * coeff;
                cols.final_s_eval
                    .copy_from_slice(final_s_eval.as_basis_coefficients_slice());

                whir_claim += mu_pows[idx] * claim;
                cols.whir_claim
                    .copy_from_slice(whir_claim.as_basis_coefficients_slice());
            }
        }

        let padding_proof_idx = F::from_usize(proofs.len());
        let mut chunks = chunks.peekable();

        while let Some(chunk) = chunks.next() {
            let cols: &mut StackingClaimsCols<F> = chunk.borrow_mut();
            cols.proof_idx = padding_proof_idx;
            if chunks.peek().is_none() {
                cols.is_last = F::ONE;
            }
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use openvm_cuda_backend::{base::DeviceMatrix, GpuBackend};
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
    use openvm_stark_backend::prover::AirProvingContext;

    use super::*;
    use crate::{
        stacking::{
            cuda_abi::{
                stacking_claims_tracegen, stacking_claims_tracegen_temp_bytes,
                ClaimsRecordsPerProof, StackingClaim,
            },
            cuda_tracegen::StackingBlob,
        },
        tracegen::{cuda::StandardTracegenGpuCtx, ModuleChip},
    };

    pub struct StackingClaimsTraceGeneratorGpu;

    impl ModuleChip<GpuBackend> for StackingClaimsTraceGeneratorGpu {
        type Ctx<'a> = (StandardTracegenGpuCtx<'a>, &'a StackingBlob);

        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<openvm_stark_backend::prover::AirProvingContext<GpuBackend>> {
            let proofs_gpu = ctx.0.proofs;
            let preflights_gpu = ctx.0.preflights;
            let blob = ctx.1;

            let mut num_valid_rows = 0;
            let mut row_bounds = Vec::with_capacity(proofs_gpu.len());
            let claims = proofs_gpu
                .iter()
                .map(|proof| {
                    let ret = proof
                        .cpu
                        .stacking_proof
                        .stacking_openings
                        .iter()
                        .enumerate()
                        .flat_map(|(commit_idx, openings)| {
                            openings
                                .iter()
                                .enumerate()
                                .map(move |(stacked_col_idx, opening)| StackingClaim {
                                    commit_idx: commit_idx as u32,
                                    stacked_col_idx: stacked_col_idx as u32,
                                    claim: *opening,
                                })
                        })
                        .collect_vec();
                    num_valid_rows += ret.len();
                    row_bounds.push(num_valid_rows as u32);
                    ret.to_device().unwrap()
                })
                .collect_vec();
            let mu_pows = preflights_gpu
                .iter()
                .enumerate()
                .map(|(proof_idx, preflight)| {
                    let mu = preflight.cpu.stacking.stacking_batching_challenge;
                    mu.powers()
                        .take(claims[proof_idx].len())
                        .collect_vec()
                        .to_device()
                        .unwrap()
                })
                .collect_vec();

            let height = if let Some(height) = required_height {
                if height < num_valid_rows {
                    return None;
                }
                height
            } else {
                num_valid_rows.next_power_of_two()
            };
            let width = StackingClaimsCols::<usize>::width();
            let d_trace = DeviceMatrix::with_capacity(height, width);

            let d_claims = claims.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_coeffs = blob.coeffs.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_mu_pows = mu_pows.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_records = preflights_gpu
                .iter()
                .map(|preflight| ClaimsRecordsPerProof {
                    initial_tidx: preflight.cpu.stacking.intermediate_tidx[2] as u32,
                    mu: preflight.cpu.stacking.stacking_batching_challenge,
                    mu_pow_witness: preflight.cpu.stacking.mu_pow_witness,
                    mu_pow_sample: preflight.cpu.stacking.mu_pow_sample,
                })
                .collect_vec()
                .to_device()
                .unwrap();

            unsafe {
                let temp_bytes =
                    stacking_claims_tracegen_temp_bytes(d_trace.buffer(), height).unwrap();
                let d_temp_buffer = DeviceBuffer::<u8>::with_capacity(temp_bytes);
                stacking_claims_tracegen(
                    d_trace.buffer(),
                    height,
                    width,
                    &row_bounds,
                    d_claims,
                    d_coeffs,
                    d_mu_pows,
                    &d_records,
                    proofs_gpu.len() as u32,
                    &d_temp_buffer,
                    temp_bytes,
                )
                .unwrap();
            }

            Some(AirProvingContext::simple_no_pis(d_trace))
        }
    }
}
