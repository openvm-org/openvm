#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

pub mod poseidon2 {
    /// Poseidon2 tracegen on GPU (parallelized over rows)
    ///
    /// # Arguments
    ///
    /// * `d_output` - DeviceBuffer for the output (column major)
    /// * `d_inputs` - DeviceBuffer for the inputs (column major)
    /// * `sbox_regs` - Number of sbox registers (0 or 1)
    /// * `n` - Number of rows
    ///
    /// Currently only supports same constants as  
    /// https://github.com/openvm-org/openvm/blob/08bbf79368b07437271aeacb25fb8857980ca863/crates/circuits/poseidon2-air/src/lib.rs
    /// so:
    /// * `WIDTH` - 16
    /// * `SBOX_DEGREE` - 7
    /// * `HALF_FULL_ROUNDS` - 4
    /// * `PARTIAL_ROUNDS` - 13
    use super::*;

    extern "C" {
        fn _poseidon2_dummy_tracegen(output: *mut F, inputs: *mut F, sbox_regs: u32, n: u32)
            -> i32;
    }

    pub unsafe fn dummy_tracegen(
        d_output: &DeviceBuffer<F>,
        d_inputs: &DeviceBuffer<F>,
        sbox_regs: u32,
        n: u32,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_poseidon2_dummy_tracegen(
            d_output.as_mut_ptr(),
            d_inputs.as_mut_ptr(),
            sbox_regs,
            n,
        ))
    }
}
