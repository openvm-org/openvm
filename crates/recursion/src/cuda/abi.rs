#![allow(clippy::missing_safety_doc)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError};

extern "C" {
    fn _get_fp_prefix_scan_temp_bytes(d_arr: *mut F, n: usize, temp_n: *mut usize) -> i32;
    fn _fp_prefix_scan(
        d_arr: *mut F,
        n: usize,
        d_temp: *mut std::ffi::c_void,
        temp_n: usize,
    ) -> i32;
}

/*
 * Stores the minimum temporary buffer size (in bytes) required to call prefix_scan
 * on size_n DeviceBuffer d_arr.
 */
pub unsafe fn get_prefix_scan_temp_bytes(
    d_arr: &DeviceBuffer<F>,
    n: usize,
    temp_n: &mut usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_get_fp_prefix_scan_temp_bytes(
        d_arr.as_mut_ptr(),
        n,
        temp_n as *mut usize,
    ))
}

/*
 * Takes a size-n F DeviceBuffer and performs an in-place inclusive prefix scan.
 * Note this requires a temporary buffer, which you need to allocate in Rust due
 * to VPMM. To get the minimum size of this buffer, use get_prefix_scan_temp_bytes.
 */
pub unsafe fn prefix_scan(
    d_arr: &DeviceBuffer<F>,
    n: usize,
    d_temp: &DeviceBuffer<u8>,
    temp_n: usize,
) -> Result<(), CudaError> {
    CudaError::from_result(_fp_prefix_scan(
        d_arr.as_mut_ptr(),
        n,
        d_temp.as_mut_ptr() as *mut std::ffi::c_void,
        temp_n,
    ))
}

#[cfg(test)]
mod tests {
    use eyre::Result;
    use openvm_cuda_backend::types::F;
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};
    use p3_field::FieldAlgebra;

    use super::*;

    #[test]
    fn test_cuda_prefix_scan() -> Result<()> {
        const N: usize = 64;
        let d_arr = [F::ONE; N].to_device()?;
        let mut temp_n = 0usize;
        unsafe {
            get_prefix_scan_temp_bytes(&d_arr, N, &mut temp_n)?;
        }
        let d_temp = DeviceBuffer::<u8>::with_capacity(temp_n);
        unsafe {
            prefix_scan(&d_arr, N, &d_temp, temp_n)?;
        }
        for (i, val) in d_arr.to_host()?.iter().enumerate() {
            assert_eq!(*val, F::from_canonical_usize(i + 1));
        }
        Ok(())
    }
}
