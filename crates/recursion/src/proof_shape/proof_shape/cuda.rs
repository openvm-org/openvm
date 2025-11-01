use std::sync::Arc;

use cuda_backend_v2::{Digest, GpuBackendV2};
use itertools::Itertools;
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::copy::MemCopyH2D;
use stark_backend_v2::{DIGEST_SIZE, prover::AirProvingContextV2};

use crate::{
    cuda::{preflight::PreflightGpu, vk::VerifyingKeyGpu},
    primitives::{
        pow::cuda::PowerCheckerGpuTraceGenerator, range::cuda::RangeCheckerGpuTraceGenerator,
    },
    proof_shape::{cuda_abi::proof_shape_tracegen, proof_shape::ProofShapeCols},
};

#[repr(C)]
pub(crate) struct ProofShapePerProof {
    num_present: usize,
    n_max: usize,
    n_logup: usize,
    final_cidx: usize,
    final_total_interactions: usize,
    main_commit: Digest,
}

#[repr(C)]
pub(crate) struct ProofShapeTracegenInputs {
    num_airs: usize,
    l_skip: usize,
    max_cached: usize,
    min_cached_idx: usize,
    range_checker_8_ptr: *mut u32,
    range_checker_5_ptr: *mut u32,
    pow_checker_ptr: *mut u32,
}

pub(crate) fn generate_proving_ctx<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    vk_gpu: &VerifyingKeyGpu,
    preflights_gpu: &[PreflightGpu],
    encoder_width: usize,
    min_cached_idx: usize,
    max_cached: usize,
    range_checker: Arc<RangeCheckerGpuTraceGenerator<LIMB_BITS>>,
    pow_checker: Arc<PowerCheckerGpuTraceGenerator<2, 32>>,
) -> AirProvingContextV2<GpuBackendV2> {
    debug_assert_eq!(NUM_LIMBS, 4);
    debug_assert_eq!(LIMB_BITS, 8);

    let num_airs = vk_gpu.per_air.len();
    let height = (preflights_gpu.len() * (num_airs + 1)).next_power_of_two();
    let width =
        ProofShapeCols::<u8, NUM_LIMBS>::width() + encoder_width + max_cached * (DIGEST_SIZE + 1);
    let trace = DeviceMatrix::with_capacity(height, width);

    let sorted_trace_data = preflights_gpu
        .iter()
        .map(|preflight| preflight.proof_shape.sorted_trace_vdata.as_ptr())
        .collect_vec();
    let cached_commits = preflights_gpu
        .iter()
        .map(|preflight| preflight.proof_shape.sorted_cached_commits.as_ptr())
        .collect_vec();
    let per_proof = preflights_gpu
        .iter()
        .map(|preflight| ProofShapePerProof {
            num_present: preflight.proof_shape.num_present,
            n_max: preflight.proof_shape.n_max,
            n_logup: preflight.proof_shape.n_logup,
            final_cidx: preflight.proof_shape.final_cidx,
            final_total_interactions: preflight.proof_shape.final_total_interactions,
            main_commit: preflight.proof_shape.main_commit,
        })
        .collect_vec()
        .to_device()
        .unwrap();
    let inputs = ProofShapeTracegenInputs {
        num_airs,
        l_skip: vk_gpu.system_params.l_skip,
        max_cached,
        min_cached_idx,
        range_checker_8_ptr: range_checker.count_mut_ptr(),
        range_checker_5_ptr: pow_checker.range_count_mut_ptr(),
        pow_checker_ptr: pow_checker.pow_count_mut_ptr(),
    };

    unsafe {
        proof_shape_tracegen(
            trace.buffer(),
            height,
            &vk_gpu.per_air,
            sorted_trace_data,
            cached_commits,
            &per_proof,
            preflights_gpu.len(),
            &inputs,
        )
        .unwrap();
    }
    AirProvingContextV2::simple_no_pis(trace)
}
