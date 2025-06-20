#![allow(clippy::missing_safety_doc)]
use stark_backend_gpu::cuda::{d_buffer::DeviceBuffer, error::CudaError};

pub mod poseidon2 {

    /// Poseidon2 tracegen on GPU
    ///
    /// # Arguments
    ///
    /// * `d_output` - DeviceBuffer for the output (column major)
    /// * `d_inputs` - DeviceBuffer for the inputs (column major)
    /// * `sbox_regs` - Number of sbox registers (0 or 1)
    /// * `n` - Number of rows
    /// Parallelized over rows
    ///
    /// Currently only supports same constants as  
    /// https://github.com/openvm-org/openvm/blob/08bbf79368b07437271aeacb25fb8857980ca863/crates/circuits/poseidon2-air/src/lib.rs
    /// so:
    /// * `WIDTH` - 16
    /// * `SBOX_DEGREE` - 7
    /// * `HALF_FULL_ROUNDS` - 4
    /// * `PARTIAL_ROUNDS` - 13
    ///
    ///
    use super::*;

    extern "C" {
        fn _poseidon2_tracegen(
            output: *mut std::ffi::c_void,
            inputs: *mut std::ffi::c_void,
            sbox_regs: u32,
            n: u32,
        ) -> i32;

        fn _print_poseidon2_constants() -> i32;
    }

    pub unsafe fn tracegen<T>(
        d_output: &DeviceBuffer<T>,
        d_inputs: &DeviceBuffer<T>,
        sbox_regs: u32,
        n: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_poseidon2_tracegen(
            d_output.as_mut_raw_ptr(),
            d_inputs.as_mut_raw_ptr(),
            sbox_regs,
            n,
        ))
    }
}
