// 2 reads, 1 write, imm support
pub mod native_adapter;
// 2 reads, 1 write, arbitrary read size, arbitrary write size, no imm support
pub mod native_basic_adapter;
// 1 read, 1 write, arbitrary read size, arbitrary write size, no imm support
pub mod convert_adapter;
// 2 reads, 1 write, read size = write size = N, no imm support, read/write to address space d
pub mod native_vectorized_adapter;
// 2 reads and 1 write from/to heap memory
pub mod native_vec_heap_adapter;

use crate::{
    arch::{BasicAdapterInterface, MinimalInstruction},
    kernels::adapters::native_adapter::GenericNativeAdapterInterface,
};

pub type NativeAdapterInterface<T> = GenericNativeAdapterInterface<T, 2, 1>;
pub type NativeVectorizedAdapterInterface<T, const N: usize> =
    BasicAdapterInterface<T, MinimalInstruction<T>, 2, 1, N, N>;
