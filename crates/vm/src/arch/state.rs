use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use rand::{rngs::StdRng, SeedableRng};

use super::{ExecutionError, Streams};
#[cfg(feature = "metrics")]
use crate::metrics::VmMetrics;
use crate::{arch::execution_mode::E1ExecutionCtx, system::memory::online::GuestMemory};

/// Represents the core state of a VM.
pub struct VmState<F, MEM = GuestMemory> {
    pub instret: u64,
    pub pc: u32,
    pub memory: MEM,
    pub streams: Streams<F>,
    pub rng: StdRng,
    #[cfg(feature = "metrics")]
    pub metrics: VmMetrics,
}

impl<F, MEM> VmState<F, MEM> {
    pub fn new(
        instret: u64,
        pc: u32,
        memory: MEM,
        streams: impl Into<Streams<F>>,
        seed: u64,
    ) -> Self {
        Self {
            instret,
            pc,
            memory,
            streams: streams.into(),
            rng: StdRng::seed_from_u64(seed),
            #[cfg(feature = "metrics")]
            metrics: VmMetrics::default(),
        }
    }
}

/// Represents the full execution state of a VM during execution.
/// The global state is generic in guest memory `MEM` and additional context `CTX`.
/// The host state is execution context specific.
pub struct VmSegmentState<F, MEM, CTX> {
    /// Core VM state
    pub vm_state: VmState<F, MEM>,
    /// Execution-specific fields
    pub exit_code: Result<Option<u32>, ExecutionError>,
    pub ctx: CTX,
}

impl<F, MEM, CTX> VmSegmentState<F, MEM, CTX> {
    pub fn new(vm_state: VmState<F, MEM>, ctx: CTX) -> Self {
        Self {
            vm_state,
            ctx,
            exit_code: Ok(None),
        }
    }
}

impl<F, MEM, CTX> Deref for VmSegmentState<F, MEM, CTX> {
    type Target = VmState<F, MEM>;

    fn deref(&self) -> &Self::Target {
        &self.vm_state
    }
}

impl<F, MEM, CTX> DerefMut for VmSegmentState<F, MEM, CTX> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vm_state
    }
}

impl<F, CTX> VmSegmentState<F, GuestMemory, CTX>
where
    CTX: E1ExecutionCtx,
{
    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn vm_read<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE] {
        self.ctx
            .on_memory_operation(addr_space, ptr, BLOCK_SIZE as u32);
        self.host_read(addr_space, ptr)
    }

    /// Runtime write operation for a block of memory
    #[inline(always)]
    pub fn vm_write<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; BLOCK_SIZE],
    ) {
        self.ctx
            .on_memory_operation(addr_space, ptr, BLOCK_SIZE as u32);
        self.host_write(addr_space, ptr, data)
    }

    #[inline(always)]
    pub fn vm_read_slice<T: Copy + Debug>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        len: usize,
    ) -> &[T] {
        self.ctx.on_memory_operation(addr_space, ptr, len as u32);
        self.host_read_slice(addr_space, ptr, len)
    }

    #[inline(always)]
    pub fn host_read<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE] {
        unsafe { self.memory.read(addr_space, ptr) }
    }

    #[inline(always)]
    pub fn host_write<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; BLOCK_SIZE],
    ) {
        unsafe { self.memory.write(addr_space, ptr, *data) }
    }

    #[inline(always)]
    pub fn host_read_slice<T: Copy + Debug>(&self, addr_space: u32, ptr: u32, len: usize) -> &[T] {
        unsafe { self.memory.get_slice(addr_space, ptr, len) }
    }
}
