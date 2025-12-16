#![cfg(all(test, feature = "cuda"))]

use openvm_circuit::{
    arch::{DenseRecordArena, SystemConfig, VmConfig},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use openvm_new_keccak256_circuit::{
    extension::cuda::{Keccak256Rv32GpuBuilder, Keccak256GpuProverExt},
    Keccak256Rv32Config,
};
use openvm_cuda_backend::engine::GpuBabyBearPoseidon2Engine;
use openvm_stark_sdk::engine::{StarkEngine, VmBuilder};
use std::sync::Arc;

#[test]
fn test_cuda_build() {
    // This test verifies that CUDA components can be built
    let config = Keccak256Rv32Config::default();
    let builder = Keccak256Rv32GpuBuilder;
    
    // Create AIR inventory
    let circuit = config.circuit::<GpuBabyBearPoseidon2Engine>();
    
    // Try to build the chip complex - this should compile successfully
    let chip_complex = builder.create_chip_complex(&config, circuit).unwrap();
    
    assert!(chip_complex.inventory.airs().len() > 0, "Should have AIRs in inventory");
}

#[test]
fn test_xorin_gpu_chip_instantiation() {
    use openvm_new_keccak256_circuit::xorin::cuda::XorinVmChipGpu;
    
    let bitwise_bus = BitwiseOperationLookupBus::<8>::new(0);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<8>::new(bitwise_bus));
    let pointer_max_bits = 16;
    
    let gpu_chip = XorinVmChipGpu::new(bitwise_chip.gpu.clone(), pointer_max_bits);
    
    // Create empty arena to test basic functionality
    let arena = DenseRecordArena::new();
    let air_context = gpu_chip.generate_proving_ctx(arena);
    
    // Should produce valid (though empty) trace
    assert!(air_context.get_trace(0).height() >= 1);
}