use std::sync::Arc;

use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::Rng;
use stark_backend_gpu::cuda::copy::MemCopyH2D;

use crate::{
    dummy::cuda::utils::send_bitwise_operation_lookups,
    primitives::bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    utils::assert_eq_cpu_and_gpu_matrix,
};

#[test]
fn test_bitwise_op_lookup_tracegen() {
    const NUM_BITS: usize = 8;
    const NUM_PAIRS: usize = 1 << 16;

    let mut rng = create_seeded_rng();

    let pairs: Vec<[u32; 2]> = (0..NUM_PAIRS)
        .map(|_| {
            [
                rng.gen_range(0..(1 << NUM_BITS)),
                rng.gen_range(0..(1 << NUM_BITS)),
            ]
        })
        .collect();

    let bus = BitwiseOperationLookupBus::new(0);
    let cpu_chip = BitwiseOperationLookupChip::<NUM_BITS>::new(bus);
    for (i, pair) in pairs.iter().enumerate() {
        if i % 2 == 0 {
            cpu_chip.request_range(pair[0], pair[1]);
        } else {
            cpu_chip.request_xor(pair[0], pair[1]);
        }
    }
    let cpu_trace = Arc::new(cpu_chip.generate_trace());

    let gpu_chip = BitwiseOperationLookupChipGPU::<NUM_BITS>::new(bus);
    let pairs_device = pairs
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();
    unsafe {
        send_bitwise_operation_lookups(&gpu_chip.count, &pairs_device, NUM_BITS as u32).unwrap();
    }
    let trace = gpu_chip.generate_trace();

    assert_eq_cpu_and_gpu_matrix(cpu_trace, &trace);
}
