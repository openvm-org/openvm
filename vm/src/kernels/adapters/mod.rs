pub mod native_adapter;
pub mod native_vectorized_adapter;

use crate::arch::BasicAdapterInterface;

pub type NativeAdapterInterface<T> = BasicAdapterInterface<T, 2, 1, 1, 1>;
pub type NativeVectorizedAdapterInterface<T, const N: usize> = BasicAdapterInterface<T, 2, 1, N, N>;