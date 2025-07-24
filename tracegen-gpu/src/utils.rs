use std::marker::PhantomData;

use derive_new::new;
use openvm_stark_backend::{
    prover::{
        cpu::CpuBackend,
        hal::{MatrixDimensions, ProverBackend},
        types::AirProvingContext,
    },
    Chip,
};
use stark_backend_gpu::{
    data_transporter::transport_matrix_to_device, prover_backend::GpuBackend, types::SC,
};

/// A struct that has the same memory layout as `uint2` to be used in FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, new)]
pub struct UInt2 {
    pub x: u32,
    pub y: u32,
}

pub fn get_empty_air_proving_ctx<PB: ProverBackend>() -> AirProvingContext<PB> {
    AirProvingContext {
        cached_mains: vec![],
        common_main: None,
        public_values: vec![],
    }
}

// Wraps a CPU chip for use with GpuBackend
pub struct HybridChip<RA, C: Chip<RA, CpuBackend<SC>>> {
    pub cpu_chip: C,
    _marker: PhantomData<RA>,
}

impl<RA, C: Chip<RA, CpuBackend<SC>>> HybridChip<RA, C> {
    pub fn new(cpu_chip: C) -> Self {
        Self {
            cpu_chip,
            _marker: PhantomData,
        }
    }
}

impl<RA, C: Chip<RA, CpuBackend<SC>>> Chip<RA, GpuBackend> for HybridChip<RA, C> {
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<GpuBackend> {
        let ctx = self.cpu_chip.generate_proving_ctx(arena);
        cpu_proving_ctx_to_gpu(ctx)
    }
}

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
