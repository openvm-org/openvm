#[cfg(any(test, feature = "test-utils"))]
mod stark_utils;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

use std::mem::size_of_val;

pub use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_stark_backend::p3_field::PrimeField32;
#[cfg(any(test, feature = "test-utils"))]
pub use stark_utils::*;
#[cfg(any(test, feature = "test-utils"))]
pub use test_utils::*;

#[inline(always)]
pub fn transmute_field_to_u32<F: PrimeField32>(field: &F) -> u32 {
    debug_assert_eq!(
        std::mem::size_of::<F>(),
        std::mem::size_of::<u32>(),
        "Field type F must have the same size as u32"
    );
    debug_assert_eq!(
        std::mem::align_of::<F>(),
        std::mem::align_of::<u32>(),
        "Field type F must have the same alignment as u32"
    );
    // SAFETY: This assumes that F has the same memory layout as u32.
    // This is only safe for field types that are guaranteed to be represented
    // as a single u32 internally
    unsafe { *(field as *const F as *const u32) }
}

#[inline(always)]
pub fn transmute_u32_to_field<F: PrimeField32>(value: &u32) -> F {
    debug_assert_eq!(
        std::mem::size_of::<F>(),
        std::mem::size_of::<u32>(),
        "Field type F must have the same size as u32"
    );
    debug_assert_eq!(
        std::mem::align_of::<F>(),
        std::mem::align_of::<u32>(),
        "Field type F must have the same alignment as u32"
    );
    // SAFETY: This assumes that F has the same memory layout as u32.
    // This is only safe for field types that are guaranteed to be represented
    // as a single u32 internally
    unsafe { *(value as *const u32 as *const F) }
}

/// # Safety
/// The type `T` should be plain old data so there is no worry about [Drop] behavior in the
/// transmutation.
#[inline(always)]
pub unsafe fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    let len = size_of_val(slice);
    // SAFETY: length and alignment are correct.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, len) }
}

/// Allocates a boxed slice of `T` with all bytes zeroed.
/// SAFETY: The caller must ensure that zero representation is valid for `T`.
pub fn get_zeroed_array<T: Sized>(len: usize) -> Box<[T]> {
    if len == 0 {
        return Box::new([]);
    }

    let layout = std::alloc::Layout::array::<T>(len).expect("Layout calculation failed");

    // SAFETY: We're allocating memory and zero-initializing it directly.
    // This is safe because:
    // 1. T is Sized, so we know its exact size and alignment
    // 2. The caller guarantees that zero representation is valid for T
    // 3. We use the global allocator with the correct layout
    // 4. We only create the Box after successful allocation and initialization
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let slice = std::slice::from_raw_parts_mut(ptr, len);
        Box::from_raw(slice)
    }
}
