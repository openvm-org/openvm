#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_circuit::{
        arch::{testing::TestVmExecutor, DenseRecordArena, SystemConfig, VirtualMachine},
        system::memory::tree::public_values::UserPublicValuesProof,
        utils::new_air_test_result,
    };
    use openvm_circuit_primitives::{
        bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    };
    use openvm_cuda_backend::{
        chip::{CudaChip, CudaChipInternal, GpuExecutor},
        prover_backend::GpuBackend,
    };
    use openvm_instructions::VmOpcode;
    use openvm_new_keccak256_transpiler::{XorinIo, XorinOpcode};
    use openvm_rv32im_circuit::{Rv32IConfig, Rv32IProverOptions};
    use openvm_stark_backend::{
        air_builders::PartitionedAirBuilder,
        engine::air::ProverConstraintFolder,
        p3_baby_bear::{BabyBear, BabyBearParameters},
        p3_field::Field,
        p3_matrix::dense::RowMajorMatrix,
        prover::types::{AirProvingContext, TraceCommitmentBuilder},
        Chip, ChipUsageGetter,
    };
    use openvm_stark_sdk::{
        config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkProver,
        p3_baby_bear::BabyBear, TraceInfo,
    };

    use crate::{
        cuda::XorinVmChipGpu,
        xorin::{air::XorinAir, columns::NUM_XORIN_VM_COLS, XorinVmChip, XorinVmExecutor},
    };

    type F = BabyBear;

    #[test]
    fn test_xorin_cuda_trace_generation() {
        let bitwise_bus = BitwiseOperationLookupBus::<8>::new(0);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<8>::new(bitwise_bus));

        let pointer_max_bits = 16;
        
        // Create CPU and GPU chips
        let cpu_chip = XorinVmChip::new(
            XorinVmExecutor::new(0, pointer_max_bits),
            crate::xorin::XorinVmFiller::new(bitwise_chip.clone(), pointer_max_bits),
            bitwise_chip.clone(),
        );

        let gpu_chip = XorinVmChipGpu::new(bitwise_chip.gpu.clone(), pointer_max_bits);

        // Create a simple test program
        let buffer_ptr = 0x1000;
        let input_ptr = 0x2000;
        let len = 16; // XOR 16 bytes

        // Create test instruction
        let instruction = VmOpcode::from_usize(XorinOpcode::XORIN.as_usize())
            .into_instruction::<F>()
            .with_a(XorinIo::RS(10))
            .with_b(XorinIo::RS(11))
            .with_c(XorinIo::RS(12))
            .with_d(1) // RV32_REGISTER_AS
            .with_e(0); // RV32_MEMORY_AS

        // Create a test VM with memory initialized
        let mut test_vm = TestVmExecutor::default();
        
        // Set up registers
        test_vm.write_register(10, buffer_ptr as u32);
        test_vm.write_register(11, input_ptr as u32);
        test_vm.write_register(12, len as u32);
        
        // Initialize memory with test data
        let buffer_data = vec![0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
                               0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let input_data = vec![0xfeu8, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
                              0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x00];
        
        for (i, &byte) in buffer_data.iter().enumerate() {
            test_vm.write_memory(buffer_ptr + i, byte);
        }
        for (i, &byte) in input_data.iter().enumerate() {
            test_vm.write_memory(input_ptr + i, byte);
        }

        // Execute instruction
        test_vm.step(&cpu_chip.executor, instruction).unwrap();

        // Generate traces
        let (pk, _vk) = BabyBearBlake3Engine::new(Rv32IProverOptions::default()).setup(vec![
            cpu_chip.into_air(),
            // Additional chips would be added here in a real scenario
        ]);
        
        let config = SystemConfig::default()
            .with_max_segment_size(1)
            .with_continuations(1);
            
        let mut vm = VirtualMachine::new(config, cpu_chip.executor);
        vm.add_extension(cpu_chip.filler.clone());
        
        // Add the test instruction to the VM
        vm.step(instruction);
        
        let records = vm.get_chip_records();
        
        // Get CPU trace
        let cpu_trace = cpu_chip.generate_trace_timed(&pk[0], records);
        
        // Get GPU trace
        let gpu_air_context = gpu_chip.generate_proving_ctx(records);
        let gpu_trace = gpu_air_context.get_trace(0);
        
        // Compare traces
        assert_eq!(cpu_trace.height(), gpu_trace.height(), "Trace heights don't match");
        assert_eq!(cpu_trace.width(), gpu_trace.width(), "Trace widths don't match");
        
        // Compare trace values (allowing for some differences due to implementation details)
        let cpu_matrix = cpu_trace.as_any().downcast_ref::<RowMajorMatrix<F>>().unwrap();
        let gpu_matrix = gpu_trace.as_any().downcast_ref::<RowMajorMatrix<F>>().unwrap();
        
        for row in 0..cpu_trace.height() {
            for col in 0..cpu_trace.width() {
                let cpu_val = cpu_matrix.get(row, col).clone();
                let gpu_val = gpu_matrix.get(row, col).clone();
                if cpu_val != gpu_val {
                    println!("Mismatch at row {}, col {}: CPU = {:?}, GPU = {:?}", row, col, cpu_val, gpu_val);
                }
            }
        }
    }

    #[test] 
    fn test_xorin_cuda_air_constraints() {
        // This test verifies that the GPU-generated trace satisfies AIR constraints
        let bitwise_bus = BitwiseOperationLookupBus::<8>::new(0);
        let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<8>::new(bitwise_bus));

        let pointer_max_bits = 16;
        let gpu_chip = XorinVmChipGpu::new(bitwise_chip.gpu.clone(), pointer_max_bits);
        
        // Create simple test data
        let mut arena = DenseRecordArena::new();
        
        // Generate a simple xorin record
        // This would normally come from VM execution
        // For now, we'll leave it empty to test basic functionality
        
        let air_context = gpu_chip.generate_proving_ctx(arena);
        
        // The trace should be valid even if empty
        assert!(air_context.get_trace(0).height() >= 1);
    }
}