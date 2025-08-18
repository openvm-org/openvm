#[cfg(any(test, feature = "test-utils"))]
mod stark_utils;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

use std::{marker::PhantomData, mem::size_of_val};

pub use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_stark_backend::{
    p3_field::PrimeField32,
    prover::{
        cpu::CpuBackend,
        hal::{MatrixDimensions, ProverBackend},
        types::AirProvingContext,
    },
    Chip,
};
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

#[cfg(feature = "cuda")]
pub fn get_empty_air_proving_ctx<PB: ProverBackend>() -> AirProvingContext<PB> {
    AirProvingContext {
        cached_mains: vec![],
        common_main: None,
        public_values: vec![],
    }
}

#[cfg(feature = "cuda")]
use stark_backend_gpu::{
    data_transporter::transport_matrix_to_device, prover_backend::GpuBackend, types::SC,
};

#[cfg(feature = "cuda")]
// Wraps a CPU chip for use with GpuBackend
pub struct HybridChip<RA, C: Chip<RA, CpuBackend<SC>>> {
    pub cpu_chip: C,
    _marker: PhantomData<RA>,
}

#[cfg(feature = "cuda")]
impl<RA, C: Chip<RA, CpuBackend<SC>>> HybridChip<RA, C> {
    pub fn new(cpu_chip: C) -> Self {
        Self {
            cpu_chip,
            _marker: PhantomData,
        }
    }
}

#[cfg(feature = "cuda")]
impl<RA, C: Chip<RA, CpuBackend<SC>>> Chip<RA, GpuBackend> for HybridChip<RA, C> {
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<GpuBackend> {
        let ctx = self.cpu_chip.generate_proving_ctx(arena);
        cpu_proving_ctx_to_gpu(ctx)
    }
}

#[cfg(feature = "cuda")]
pub fn cpu_proving_ctx_to_gpu(
    cpu_ctx: AirProvingContext<CpuBackend<SC>>,
) -> AirProvingContext<GpuBackend> {
    assert!(
        cpu_ctx.cached_mains.is_empty(),
        "CPU to GPU transfer of cached traces not supported"
    );
    let trace = cpu_ctx
        .common_main
        .filter(|trace| trace.height() > 0)
        .map(transport_matrix_to_device);
    AirProvingContext {
        cached_mains: vec![],
        common_main: trace,
        public_values: cpu_ctx.public_values,
    }
}
