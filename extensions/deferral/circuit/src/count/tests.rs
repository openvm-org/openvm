use std::sync::Arc;

use openvm_circuit::arch::{Arena, DenseRecordArena};
use openvm_circuit_primitives::Chip;
use openvm_cpu_backend::CpuBackend;
use openvm_cuda_backend::{
    data_transporter::assert_eq_host_and_device_matrix_col_maj, prelude::BabyBearPoseidon2Config,
};
use openvm_cuda_common::{
    common::get_device,
    copy::MemCopyH2D,
    stream::{CudaStream, GpuDeviceCtx, StreamGuard},
};
use openvm_stark_backend::prover::ColMajorMatrix;
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::Rng;

use crate::count::{DeferralCircuitCountChip, DeferralCircuitCountChipGpu};

#[test]
fn test_cuda_deferral_count_tracegen_equivalence() {
    const NUM_DEFERRAL_CIRCUITS: usize = 16;

    let mut rng = create_seeded_rng();
    let counts = (0..NUM_DEFERRAL_CIRCUITS)
        .map(|_| rng.random_range(0..200))
        .collect::<Vec<_>>();

    let cpu_chip = DeferralCircuitCountChip::new(NUM_DEFERRAL_CIRCUITS);
    for (idx, mult) in counts.iter().copied().enumerate() {
        for _ in 0..mult {
            cpu_chip.add_count(idx as u32);
        }
    }

    let device_ctx = GpuDeviceCtx {
        device_id: get_device().unwrap() as u32,
        stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
    };
    let count = Arc::new(counts.to_device_on(&device_ctx).unwrap());
    let gpu_chip =
        DeferralCircuitCountChipGpu::new(count, NUM_DEFERRAL_CIRCUITS, device_ctx.clone());

    let cpu_trace = <DeferralCircuitCountChip as Chip<
        (),
        CpuBackend<BabyBearPoseidon2Config>,
    >>::generate_proving_ctx(&cpu_chip, ())
    .common_main;
    let gpu_trace = gpu_chip
        .generate_proving_ctx(DenseRecordArena::with_capacity(1, 1))
        .common_main;

    let cpu_trace_cm = ColMajorMatrix::from_row_major(&cpu_trace);
    assert_eq_host_and_device_matrix_col_maj(&cpu_trace_cm, &gpu_trace, &device_ctx);
}
