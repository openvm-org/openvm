pub mod native_adapter;

use crate::arch::BasicAdapterInterface;

pub type NativeAdapterInterface<T> = BasicAdapterInterface<T, 2, 1, 1, 1>;
