use std::sync::Arc;

use itertools::{izip, Itertools};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_maybe_rayon::prelude::*,
    proof::{Proof, StackingProof},
    prover::{AirProvingContext, CpuBackend},
    AirRef, FiatShamirTranscript, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use strum::{EnumCount, EnumDiscriminants};

use crate::{
    stacking::{
        bus::*,
        claims::{StackingClaimsAir, StackingClaimsTraceGenerator},
        eq_base::{EqBaseAir, EqBaseTraceGenerator},
        eq_bits::{EqBitsAir, EqBitsTraceGenerator},
        opening::{OpeningClaimsAir, OpeningClaimsTraceGenerator},
        sumcheck::{SumcheckRoundsAir, SumcheckRoundsTraceGenerator},
        univariate::{UnivariateRoundAir, UnivariateRoundTraceGenerator},
    },
    system::{
        AirModule, BusIndexManager, BusInventory, GlobalCtxCpu, Preflight, StackingPreflight,
        TraceGenModule,
    },
    tracegen::{ModuleChip, RowMajorChip, StandardTracegenCtx},
    utils::pow_observe_sample,
};

mod bus;
pub mod claims;
pub mod eq_base;
pub mod eq_bits;
pub mod opening;
pub mod sumcheck;
pub mod univariate;
mod utils;

#[cfg(feature = "cuda")]
mod cuda_abi;

pub struct StackingModule {
    bus_inventory: BusInventory,

    // Internal buses
    stacking_tidx_bus: StackingModuleTidxBus,
    claim_coefficients_bus: ClaimCoefficientsBus,
    sumcheck_claims_bus: SumcheckClaimsBus,
    eq_rand_values_bus: EqRandValuesLookupBus,
    eq_base_bus: EqBaseBus,
    eq_bits_internal_bus: EqBitsInternalBus,
    eq_kernel_lookup_bus: EqKernelLookupBus,
    eq_bits_lookup_bus: EqBitsLookupBus,

    l_skip: usize,
    n_stack: usize,
    stacking_index_mult: usize,
    /// Number of PoW bits for μ batching challenge.
    mu_pow_bits: usize,
}

impl StackingModule {
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Self {
        Self {
            bus_inventory,
            stacking_tidx_bus: StackingModuleTidxBus::new(b.new_bus_idx()),
            claim_coefficients_bus: ClaimCoefficientsBus::new(b.new_bus_idx()),
            sumcheck_claims_bus: SumcheckClaimsBus::new(b.new_bus_idx()),
            eq_rand_values_bus: EqRandValuesLookupBus::new(b.new_bus_idx()),
            eq_base_bus: EqBaseBus::new(b.new_bus_idx()),
            eq_bits_internal_bus: EqBitsInternalBus::new(b.new_bus_idx()),
            eq_kernel_lookup_bus: EqKernelLookupBus::new(b.new_bus_idx()),
            eq_bits_lookup_bus: EqBitsLookupBus::new(b.new_bus_idx()),
            l_skip: child_vk.inner.params.l_skip,
            n_stack: child_vk.inner.params.n_stack,
            stacking_index_mult: child_vk
                .inner
                .params
                .whir
                .rounds
                .first()
                .map(|round| round.num_queries)
                .unwrap_or(0)
                << child_vk.inner.params.k_whir(),
            mu_pow_bits: child_vk.inner.params.whir.mu_pow_bits,
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    pub fn run_preflight<TS>(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory,
    {
        let mut sumcheck_rnd = vec![];
        let mut intermediate_tidx = [0; 3];

        let StackingProof {
            univariate_round_coeffs,
            sumcheck_round_polys,
            stacking_openings,
        } = &proof.stacking_proof;

        let lambda = ts.sample_ext();
        intermediate_tidx[0] = ts.len();

        for coef in univariate_round_coeffs {
            ts.observe_ext(*coef);
        }
        let u0 = ts.sample_ext();
        let univariate_poly_rand_eval = izip!(univariate_round_coeffs, u0.powers())
            .map(|(&coef, pow)| coef * pow)
            .sum();
        sumcheck_rnd.push(u0);
        intermediate_tidx[1] = ts.len();

        for poly in sumcheck_round_polys {
            for eval in poly {
                ts.observe_ext(*eval);
            }
            let ui = ts.sample_ext();
            sumcheck_rnd.push(ui);
        }
        intermediate_tidx[2] = ts.len();

        for matrix_openings in stacking_openings {
            for col_opening in matrix_openings {
                ts.observe_ext(*col_opening);
            }
        }

        // μ PoW: observe witness and sample before sampling μ
        let mu_pow_witness = proof.whir_proof.mu_pow_witness;
        let mu_pow_sample = pow_observe_sample(ts, self.mu_pow_bits, mu_pow_witness);

        let stacking_batching_challenge = ts.sample_ext();

        preflight.stacking = StackingPreflight {
            intermediate_tidx,
            post_tidx: ts.len(),
            univariate_poly_rand_eval,
            stacking_batching_challenge,
            mu_pow_witness,
            mu_pow_sample,
            lambda,
            sumcheck_rnd,
        };
    }
}

impl AirModule for StackingModule {
    fn num_airs(&self) -> usize {
        StackingModuleChipDiscriminants::COUNT
    }

    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let opening_air = OpeningClaimsAir {
            lifted_heights_bus: self.bus_inventory.lifted_heights_bus,
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            column_claims_bus: self.bus_inventory.column_claims_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            claim_coefficients_bus: self.claim_coefficients_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            l_skip: self.l_skip,
            n_stack: self.n_stack,
        };
        let univariate_round_air = UnivariateRoundAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            l_skip: self.l_skip,
        };
        let sumcheck_rounds_air = SumcheckRoundsAir {
            constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            eq_base_bus: self.eq_base_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            l_skip: self.l_skip,
        };
        let stacking_claims_air = StackingClaimsAir {
            stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
            whir_module_bus: self.bus_inventory.whir_module_bus,
            whir_mu_bus: self.bus_inventory.whir_mu_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
            stacking_tidx_bus: self.stacking_tidx_bus,
            claim_coefficients_bus: self.claim_coefficients_bus,
            sumcheck_claims_bus: self.sumcheck_claims_bus,
            stacking_index_mult: self.stacking_index_mult,
            mu_pow_bits: self.mu_pow_bits,
        };
        let eq_base_air = EqBaseAir {
            constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            eq_base_bus: self.eq_base_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            eq_kernel_lookup_bus: self.eq_kernel_lookup_bus,
            eq_neg_base_rand_bus: self.bus_inventory.eq_neg_base_rand_bus,
            eq_neg_result_bus: self.bus_inventory.eq_neg_result_bus,
            l_skip: self.l_skip,
        };
        let eq_bits_air = EqBitsAir {
            eq_bits_internal_bus: self.eq_bits_internal_bus,
            eq_bits_lookup_bus: self.eq_bits_lookup_bus,
            eq_rand_values_bus: self.eq_rand_values_bus,
            n_stack: self.n_stack,
            l_skip: self.l_skip,
        };
        vec![
            Arc::new(opening_air),
            Arc::new(univariate_round_air),
            Arc::new(sumcheck_rounds_air),
            Arc::new(stacking_claims_air),
            Arc::new(eq_base_air),
            Arc::new(eq_bits_air),
        ]
    }
}

#[derive(Clone, Copy, strum_macros::Display, EnumDiscriminants)]
#[strum_discriminants(derive(strum_macros::EnumCount))]
#[strum_discriminants(repr(usize))]
enum StackingModuleChip {
    OpeningClaims,
    UnivariateRound,
    SumcheckRounds,
    StackingClaims,
    EqBase,
    EqBits,
}

impl StackingModuleChip {
    fn index(&self) -> usize {
        StackingModuleChipDiscriminants::from(self) as usize
    }
}

impl RowMajorChip<F> for StackingModuleChip {
    type Ctx<'a> = StandardTracegenCtx<'a>;

    #[tracing::instrument(
        name = "wrapper.generate_trace",
        level = "trace",
        skip_all,
        fields(air = %self)
    )]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        match self {
            StackingModuleChip::OpeningClaims => {
                OpeningClaimsTraceGenerator.generate_trace(ctx, required_height)
            }
            StackingModuleChip::UnivariateRound => {
                UnivariateRoundTraceGenerator.generate_trace(ctx, required_height)
            }
            StackingModuleChip::SumcheckRounds => {
                SumcheckRoundsTraceGenerator.generate_trace(ctx, required_height)
            }
            StackingModuleChip::StackingClaims => {
                StackingClaimsTraceGenerator.generate_trace(ctx, required_height)
            }
            StackingModuleChip::EqBase => EqBaseTraceGenerator.generate_trace(ctx, required_height),
            StackingModuleChip::EqBits => EqBitsTraceGenerator.generate_trace(ctx, required_height),
        }
    }
}

impl TraceGenModule<GlobalCtxCpu, CpuBackend<BabyBearPoseidon2Config>> for StackingModule {
    type ModuleSpecificCtx<'a> = ();

    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        _module_ctx: &(),
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>> {
        let ctx = StandardTracegenCtx {
            vk: child_vk,
            proofs: &proofs.iter().collect_vec(),
            preflights: &preflights.iter().collect_vec(),
        };
        let chips = [
            StackingModuleChip::OpeningClaims,
            StackingModuleChip::UnivariateRound,
            StackingModuleChip::SumcheckRounds,
            StackingModuleChip::StackingClaims,
            StackingModuleChip::EqBase,
            StackingModuleChip::EqBits,
        ];
        let span = tracing::Span::current();
        chips
            .par_iter()
            .map(|chip| {
                let _guard = span.enter();
                chip.generate_proving_ctx(
                    &ctx,
                    required_heights.map(|heights| heights[chip.index()]),
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use itertools::Itertools;
    use openvm_cuda_backend::{prelude::EF, GpuBackend};
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
    use openvm_stark_backend::p3_maybe_rayon::prelude::*;

    use super::*;
    use crate::{
        cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu, GlobalCtxGpu},
        stacking::{
            claims::cuda::StackingClaimsTraceGeneratorGpu,
            cuda_abi::{
                compute_coefficients, compute_coefficients_temp_bytes, stacked_slice_data,
                PolyPrecomputation, StackedSliceData, StackedTraceData,
            },
            opening::cuda::OpeningClaimsTraceGeneratorGpu,
        },
        tracegen::cuda::{generate_gpu_proving_ctx, StandardTracegenGpuCtx},
    };

    impl ModuleChip<GpuBackend> for StackingModuleChip {
        type Ctx<'a> = (StandardTracegenGpuCtx<'a>, &'a StackingBlob);

        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            match self {
                StackingModuleChip::OpeningClaims => {
                    OpeningClaimsTraceGeneratorGpu.generate_proving_ctx(ctx, required_height)
                }
                StackingModuleChip::StackingClaims => {
                    StackingClaimsTraceGeneratorGpu.generate_proving_ctx(ctx, required_height)
                }
                _ => {
                    let proofs_cpu = ctx.0.proofs.iter().map(|p| &p.cpu).collect_vec();
                    let preflights_cpu = ctx.0.preflights.iter().map(|p| &p.cpu).collect_vec();
                    let cpu_ctx = StandardTracegenCtx {
                        vk: &ctx.0.vk.cpu,
                        proofs: &proofs_cpu,
                        preflights: &preflights_cpu,
                    };
                    generate_gpu_proving_ctx(self, &cpu_ctx, required_height)
                }
            }
        }
    }

    pub(crate) struct StackingBlob {
        pub(crate) slice_data: Vec<DeviceBuffer<StackedSliceData>>,
        pub(crate) coeffs: Vec<DeviceBuffer<EF>>,
        pub(crate) precomps: Vec<DeviceBuffer<PolyPrecomputation>>,
    }

    impl StackingBlob {
        pub fn new(
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
        ) -> Self {
            let l_skip = child_vk.system_params.l_skip as u32;
            let n_stack = child_vk.system_params.n_stack as u32;
            let log_stacked_height = l_skip + n_stack;
            let stacked_height = 1u32 << log_stacked_height;

            let mut slice_data = Vec::with_capacity(preflights.len());
            let mut coeffs = Vec::with_capacity(preflights.len());
            let mut precomps = Vec::with_capacity(preflights.len());

            for (proof, preflight) in izip!(proofs, preflights) {
                let sorted_trace_data = &preflight.cpu.proof_shape.sorted_trace_vdata;
                let num_airs = sorted_trace_data.len();
                let mut num_commits = 1;

                let mut stacked_trace_data = Vec::with_capacity(num_airs);
                let mut other_trace_data = vec![];

                let mut current_col_idx = 0;
                let mut current_row_idx = 0;

                for (air_idx, vdata) in sorted_trace_data {
                    let stark_vk = &child_vk.cpu.inner.per_air[*air_idx];
                    let trace_widths = &stark_vk.params.width;
                    let need_rot = stark_vk.params.need_rot as u32;
                    let log_height = vdata.log_height as u32;
                    // IMPORTANT: This must match CPU `get_stacked_slice_data`, which stacks ONLY
                    // the common-main columns here (cached mains are handled as separate commits).
                    let common_width = trace_widths.common_main as u32;

                    stacked_trace_data.push(StackedTraceData {
                        commit_idx: 0,
                        start_col_idx: current_col_idx,
                        start_row_idx: current_row_idx,
                        log_height,
                        width: common_width,
                        need_rot,
                    });

                    let lifted_height = log_height.max(l_skip);
                    let unbounded_row_idx = current_row_idx + (common_width << lifted_height);
                    current_col_idx += unbounded_row_idx >> log_stacked_height;
                    current_row_idx = unbounded_row_idx & (stacked_height - 1);

                    other_trace_data.extend(
                        trace_widths
                            .preprocessed
                            .iter()
                            .chain(trace_widths.cached_mains.iter())
                            .map(|&part_width| {
                                let ret = StackedTraceData {
                                    commit_idx: num_commits,
                                    start_col_idx: 0,
                                    start_row_idx: 0,
                                    log_height,
                                    width: part_width as u32,
                                    need_rot,
                                };
                                num_commits += 1;
                                ret
                            }),
                    );
                }

                stacked_trace_data.extend(other_trace_data.iter());

                let mut num_slices = 0u32;
                let slice_offsets = stacked_trace_data
                    .iter()
                    .map(|trace_data| {
                        let ret = num_slices;
                        num_slices += trace_data.width;
                        ret
                    })
                    .collect_vec();

                let d_slice_offsets = slice_offsets.to_device().unwrap();
                let d_stacked_trace_data = stacked_trace_data.to_device().unwrap();

                let d_slice_data =
                    DeviceBuffer::<StackedSliceData>::with_capacity(num_slices as usize);

                unsafe {
                    stacked_slice_data(
                        &d_slice_data,
                        &d_slice_offsets,
                        &d_stacked_trace_data,
                        num_airs as u32,
                        num_commits,
                        num_slices,
                        n_stack,
                        l_skip,
                    )
                    .unwrap();
                }

                let d_lambda_pows = preflight
                    .cpu
                    .stacking
                    .lambda
                    .powers()
                    .take((num_slices << 1) as usize)
                    .collect_vec()
                    .to_device()
                    .unwrap();

                let d_coeff_terms = DeviceBuffer::<EF>::with_capacity(num_slices as usize);
                let d_coeff_term_keys = DeviceBuffer::<u64>::with_capacity(num_slices as usize);
                let d_precomps =
                    DeviceBuffer::<PolyPrecomputation>::with_capacity(num_slices as usize);
                let d_num_coeffs = DeviceBuffer::<usize>::with_capacity(1);

                let num_claims = proof
                    .cpu
                    .stacking_proof
                    .stacking_openings
                    .iter()
                    .fold(0, |acc, v| acc + v.len());
                let d_coeffs = DeviceBuffer::<EF>::with_capacity(num_claims);
                let d_coeff_keys = DeviceBuffer::<u64>::with_capacity(num_claims);

                unsafe {
                    let temp_bytes = compute_coefficients_temp_bytes(
                        &d_coeff_terms,
                        &d_coeff_term_keys,
                        &d_coeffs,
                        &d_coeff_keys,
                        num_slices,
                        &d_num_coeffs,
                    )
                    .unwrap();
                    let d_temp_buffer = DeviceBuffer::<u8>::with_capacity(temp_bytes);
                    compute_coefficients(
                        &d_coeff_terms,
                        &d_coeff_term_keys,
                        &d_coeffs,
                        &d_coeff_keys,
                        &d_precomps,
                        &d_slice_data,
                        &preflight.stacking.sumcheck_rnd,
                        &preflight.batch_constraint.sumcheck_rnd,
                        &d_lambda_pows,
                        num_commits,
                        num_slices,
                        n_stack,
                        l_skip,
                        &d_temp_buffer,
                        temp_bytes,
                        &d_num_coeffs,
                    )
                    .unwrap();
                }

                slice_data.push(d_slice_data);
                coeffs.push(d_coeffs);
                precomps.push(d_precomps);
            }

            Self {
                slice_data,
                coeffs,
                precomps,
            }
        }
    }

    impl TraceGenModule<GlobalCtxGpu, GpuBackend> for StackingModule {
        type ModuleSpecificCtx<'a> = ();

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            _module_ctx: &(),
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContext<GpuBackend>>> {
            let blob = StackingBlob::new(child_vk, proofs, preflights);
            let ctx = (
                StandardTracegenGpuCtx {
                    vk: child_vk,
                    proofs,
                    preflights,
                },
                &blob,
            );

            let gpu_chips = [
                StackingModuleChip::OpeningClaims,
                StackingModuleChip::StackingClaims,
            ];
            let cpu_chips = [
                StackingModuleChip::UnivariateRound,
                StackingModuleChip::SumcheckRounds,
                StackingModuleChip::EqBase,
                StackingModuleChip::EqBits,
            ];

            // Launch all GPU tracegen kernels serially first (default stream).
            let indexed_gpu_ctxs = gpu_chips
                .iter()
                .map(|chip| {
                    (
                        chip.index(),
                        chip.generate_proving_ctx(
                            &ctx,
                            required_heights.map(|heights| heights[chip.index()]),
                        ),
                    )
                })
                .collect::<Vec<_>>();

            // Then run CPU tracegen for remaining AIRs in parallel
            let span = tracing::Span::current();
            let indexed_cpu_ctxs = cpu_chips
                .par_iter()
                .map(|chip| {
                    let _guard = span.enter();
                    (
                        chip.index(),
                        chip.generate_proving_ctx(
                            &ctx,
                            required_heights.map(|heights| heights[chip.index()]),
                        ),
                    )
                })
                .collect::<Vec<_>>();

            indexed_gpu_ctxs
                .into_iter()
                .chain(indexed_cpu_ctxs)
                .sorted_by(|a, b| a.0.cmp(&b.0))
                .map(|(_idx, ctx)| ctx)
                .collect()
        }
    }
}
