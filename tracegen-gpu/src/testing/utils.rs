use core::fmt::Debug;
use std::sync::Arc;

use openvm_circuit::{
    arch::{
        testing::{BITWISE_OP_LOOKUP_BUS, RANGE_CHECKER_BUS},
        MemoryConfig,
    },
    system::memory::{adapter::records::arena_size_bound, online::TracingMemory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, var_range::VariableRangeCheckerBus,
};
use openvm_stark_backend::{
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_util::log2_strict_usize,
    prover::{
        hal::{MatrixDimensions, ProverBackend},
        types::AirProvingContext,
    },
};
use stark_backend_gpu::{base::DeviceMatrix, data_transporter::transport_device_matrix_to_host};

// Asserts that two DeviceMatrix are equal
pub fn assert_eq_gpu_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    a: &DeviceMatrix<T>,
    b: &DeviceMatrix<T>,
) {
    assert_eq!(a.height(), b.height());
    assert_eq!(a.width(), b.width());
    assert_eq!(a.buffer().len(), b.buffer().len());
    let a_host = transport_device_matrix_to_host(a);
    let b_host = transport_device_matrix_to_host(b);
    for r in 0..a_host.height() {
        for c in 0..a_host.width() {
            assert_eq!(
                a_host.get(r, c),
                b_host.get(r, c),
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}

// Asserts that a RowMajorMatrix and a DeviceMatrix (a column-major matrix)
// are equal
pub fn assert_eq_cpu_and_gpu_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    cpu: Arc<RowMajorMatrix<T>>,
    gpu: &DeviceMatrix<T>,
) {
    assert_eq!(gpu.width(), cpu.width());
    assert_eq!(gpu.height(), cpu.height());
    let gpu = transport_device_matrix_to_host(gpu);
    for r in 0..cpu.height() {
        for c in 0..cpu.width() {
            assert_eq!(
                gpu.get(r, c),
                cpu.get(r, c),
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}

// Utility function to print out a DeviceMatrix as a RowMajorMatrix for easy
// comparison during debugging
pub fn print_gpu_matrix_as_row_major_matrix<T: Clone + Send + Sync + Debug>(
    gpu_matrix: &DeviceMatrix<T>,
) {
    println!("{:?}", transport_device_matrix_to_host(gpu_matrix));
}

pub(crate) fn default_tracing_memory(
    mem_config: &MemoryConfig,
    init_block_size: usize,
) -> TracingMemory {
    let max_access_adapter_n = log2_strict_usize(mem_config.max_access_adapter_n);
    let arena_size_bound = arena_size_bound(&vec![1 << 16; max_access_adapter_n]);
    TracingMemory::new(mem_config, init_block_size, arena_size_bound)
}

pub fn default_var_range_checker_bus() -> VariableRangeCheckerBus {
    // setting default range_max_bits to 17 because that's the default decomp value in MemoryConfig
    VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, 17)
}

pub fn default_bitwise_lookup_bus() -> BitwiseOperationLookupBus {
    BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS)
}

pub fn get_empty_air_proving_ctx<PB: ProverBackend>() -> AirProvingContext<PB> {
    AirProvingContext {
        cached_mains: vec![],
        common_main: None,
        public_values: vec![],
    }
}
