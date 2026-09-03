use std::{
    fmt::Debug,
    mem::size_of,
    ops::{Deref, DerefMut},
};

use eyre::eyre;
use getset::{CopyGetters, MutGetters};
use openvm_instructions::exe::SparseMemoryImage;
use rand::{rngs::StdRng, SeedableRng};
use tracing::instrument;

use super::{create_memory_image, ExecutionError, Streams};
#[cfg(feature = "metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{execution_mode::ExecutionCtxTrait, SystemConfig, VmStateMut},
    system::memory::online::{GuestMemory, LinearMemory},
};

/// Represents the core state of a VM.
#[repr(C)]
#[derive(derive_new::new, CopyGetters, MutGetters, Clone)]
pub struct VmState<MEM = GuestMemory> {
    #[getset(get_copy = "pub", get_mut = "pub")]
    pc: u32,
    pub memory: MEM,
    pub streams: Streams,
    pub rng: StdRng,
    #[cfg(feature = "metrics")]
    pub metrics: VmMetrics,
}

pub(super) const DEFAULT_RNG_SEED: u64 = 0;

impl<MEM> VmState<MEM> {
    #[inline(always)]
    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    pub fn new_with_defaults(pc: u32, memory: MEM, streams: impl Into<Streams>, seed: u64) -> Self {
        Self {
            pc,
            memory,
            streams: streams.into(),
            rng: StdRng::seed_from_u64(seed),
            #[cfg(feature = "metrics")]
            metrics: VmMetrics::default(),
        }
    }

    #[inline(always)]
    pub fn into_mut<'a, CTX>(&'a mut self, ctx: &'a mut CTX) -> VmStateMut<'a, MEM, CTX> {
        VmStateMut {
            pc: &mut self.pc,
            memory: &mut self.memory,
            streams: &mut self.streams,
            rng: &mut self.rng,
            ctx,
            #[cfg(feature = "metrics")]
            metrics: &mut self.metrics,
        }
    }
}

impl VmState<GuestMemory> {
    #[instrument(name = "VmState::initial", level = "debug", skip_all)]
    pub fn initial(
        system_config: &SystemConfig,
        init_memory: &SparseMemoryImage,
        pc_start: u32,
        inputs: impl Into<Streams>,
    ) -> Self {
        let memory = create_memory_image(&system_config.memory_config, init_memory);
        VmState::new_with_defaults(pc_start, memory, inputs.into(), DEFAULT_RNG_SEED)
    }

    pub fn reset(
        &mut self,
        init_memory: &SparseMemoryImage,
        pc_start: u32,
        streams: impl Into<Streams>,
    ) {
        self.pc = pc_start;
        self.memory.memory.fill_zero();
        self.memory.memory.set_from_sparse(init_memory);
        self.streams = streams.into();
        self.rng = StdRng::seed_from_u64(DEFAULT_RNG_SEED);
    }

    /// Deeply clone this state while copying only memory pages recorded in
    /// [`AddressMap::touched_pages`](crate::system::memory::AddressMap::touched_pages).
    ///
    /// The returned memory mappings are independent from `self`. Initial states, interpreter
    /// states, and RVR segment-boundary states maintain the required metadata. Other callers must
    /// ensure that all nonzero memory pages are marked; ordinary [`Clone`] remains the exact
    /// dense-copy operation for states that cannot provide that invariant.
    pub fn sparse_clone(&self) -> Self {
        Self {
            pc: self.pc,
            memory: GuestMemory::new(self.memory.memory.sparse_clone()),
            streams: self.streams.clone(),
            rng: self.rng.clone(),
            #[cfg(feature = "metrics")]
            metrics: self.metrics.clone(),
        }
    }
}

/// Represents the full execution state of a VM during execution.
/// The global state is generic in guest memory `MEM` and additional context `CTX`.
/// The host state is execution context specific.
// @dev: Do not confuse with `ExecutionState` struct.
#[repr(C)]
pub struct VmExecState<MEM, CTX> {
    /// Core VM state
    pub vm_state: VmState<MEM>,
    pub ctx: CTX,
    /// Execution-specific fields
    pub exit_code: Result<Option<u32>, ExecutionError>,
}

impl<CTX: ExecutionCtxTrait> VmExecState<GuestMemory, CTX> {
    #[inline(always)]
    pub fn should_suspend(&mut self) -> bool {
        CTX::should_suspend(self)
    }
}

impl<MEM, CTX> VmExecState<MEM, CTX> {
    pub fn new(vm_state: VmState<MEM>, ctx: CTX) -> Self {
        Self {
            vm_state,
            ctx,
            exit_code: Ok(None),
        }
    }

    /// Try to clone VmExecState. Return an error if `exit_code` is an error because `ExecutionEror`
    /// cannot be cloned.
    pub fn try_clone(&self) -> eyre::Result<Self>
    where
        VmState<MEM>: Clone,
        CTX: Clone,
    {
        if self.exit_code.is_err() {
            return Err(eyre!(
                "failed to clone VmExecState because exit_code is an error"
            ));
        }
        Ok(Self {
            vm_state: self.vm_state.clone(),
            exit_code: Ok(*self.exit_code.as_ref().unwrap()),
            ctx: self.ctx.clone(),
        })
    }
}

impl<MEM, CTX> Deref for VmExecState<MEM, CTX> {
    type Target = VmState<MEM>;

    fn deref(&self) -> &Self::Target {
        &self.vm_state
    }
}

impl<MEM, CTX> DerefMut for VmExecState<MEM, CTX> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vm_state
    }
}

impl<CTX> VmExecState<GuestMemory, CTX>
where
    CTX: ExecutionCtxTrait,
{
    /// Runtime byte read: `byte_ptr` is a raw byte offset into the AS's
    /// linear storage. Returns `N` bytes.
    #[inline(always)]
    pub fn vm_read_bytes<const N: usize>(&mut self, addr_space: u32, byte_ptr: u32) -> [u8; N] {
        self.ctx.on_memory_operation(addr_space, byte_ptr, N as u32);
        let value = self.host_read_bytes(addr_space, byte_ptr);
        self.ctx
            .on_memory_read(&self.vm_state.memory, addr_space, byte_ptr, N as u32);
        value
    }

    /// Runtime byte write: `byte_ptr` is a raw byte offset.
    #[inline(always)]
    pub fn vm_write_bytes<const N: usize>(
        &mut self,
        addr_space: u32,
        byte_ptr: u32,
        data: &[u8; N],
    ) {
        self.ctx.on_memory_operation(addr_space, byte_ptr, N as u32);
        self.ctx
            .on_memory_write_start(&self.vm_state.memory, addr_space, byte_ptr, N as u32);
        self.host_write_bytes(addr_space, byte_ptr, data);
        self.ctx.on_memory_write_end(&self.vm_state.memory);
    }

    /// Runtime cell read: `ptr` is an AS-native pointer.
    /// `T` must match the AS cell type (e.g. `F` for `DEFERRAL_AS`).
    #[inline(always)]
    pub fn vm_read<T: Copy + Debug, const N: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; N] {
        self.ctx.on_memory_operation(addr_space, ptr, N as u32);
        let value = self.host_read(addr_space, ptr);
        self.ctx.on_memory_read(
            &self.vm_state.memory,
            addr_space,
            ptr * size_of::<T>() as u32,
            (N * size_of::<T>()) as u32,
        );
        value
    }

    /// Runtime cell write: `ptr` is an AS-native pointer.
    #[inline(always)]
    pub fn vm_write<T: Copy + Debug, const N: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; N],
    ) {
        self.ctx.on_memory_operation(addr_space, ptr, N as u32);
        self.ctx.on_memory_write_start(
            &self.vm_state.memory,
            addr_space,
            ptr * size_of::<T>() as u32,
            (N * size_of::<T>()) as u32,
        );
        self.host_write(addr_space, ptr, data);
        self.ctx.on_memory_write_end(&self.vm_state.memory);
    }

    #[inline(always)]
    pub fn vm_read_slice<T: Copy + Debug>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        len: usize,
    ) -> &[T] {
        self.ctx.on_memory_operation(addr_space, ptr, len as u32);
        self.ctx.on_memory_read(
            &self.vm_state.memory,
            addr_space,
            ptr * size_of::<T>() as u32,
            (len * size_of::<T>()) as u32,
        );
        self.host_read_slice(addr_space, ptr, len)
    }

    #[inline(always)]
    pub fn host_read_bytes<const N: usize>(&self, addr_space: u32, byte_ptr: u32) -> [u8; N] {
        // SAFETY: caller guarantees the byte range is in bounds.
        unsafe {
            self.memory
                .memory
                .get_memory()
                .get_unchecked(addr_space as usize)
                .read(byte_ptr as usize)
        }
    }

    #[inline(always)]
    pub fn host_write_bytes<const N: usize>(
        &mut self,
        addr_space: u32,
        byte_ptr: u32,
        data: &[u8; N],
    ) {
        // SAFETY: caller guarantees the byte range is in bounds.
        unsafe { self.memory.write_bytes(addr_space, byte_ptr, *data) }
    }

    /// Cell read: `ptr` is a pointer; byte offset is `ptr * size_of::<T>()`.
    #[inline(always)]
    pub fn host_read<T: Copy + Debug, const N: usize>(&self, addr_space: u32, ptr: u32) -> [T; N] {
        // SAFETY: caller guarantees T matches the AS layout and the range is
        // in bounds.
        unsafe {
            self.memory
                .memory
                .get_memory()
                .get_unchecked(addr_space as usize)
                .read((ptr as usize) * size_of::<T>())
        }
    }

    /// Cell write: `ptr` is a pointer; byte offset is `ptr * size_of::<T>()`.
    #[inline(always)]
    pub fn host_write<T: Copy + Debug, const N: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; N],
    ) {
        // SAFETY: caller guarantees T matches the AS layout and the range is
        // in bounds.
        unsafe { self.memory.write(addr_space, ptr, *data) }
    }

    #[inline(always)]
    pub fn host_read_slice<T: Copy + Debug>(&self, addr_space: u32, ptr: u32, len: usize) -> &[T] {
        // SAFETY:
        // - T must match the AS cell type.
        // - panics if the slice is out of bounds
        unsafe { self.memory.get_slice(addr_space, ptr, len) }
    }

    #[inline(always)]
    pub fn host_read_u8_slice(&self, addr_space: u32, byte_ptr: u32, len: usize) -> &[u8] {
        // SAFETY:
        // - panics if the byte range is out of bounds
        unsafe { self.memory.get_u8_slice(addr_space, byte_ptr, len) }
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{exe::SparseMemoryImage, riscv::MEMORY_AS};

    use super::*;
    use crate::{
        arch::execution_mode::ExecutionCtx, system::memory::online::PAGE_SIZE,
        utils::test_system_config,
    };

    #[test]
    fn host_write_bytes_survives_sparse_state_clone() {
        let vm_state = VmState::initial(
            &test_system_config(),
            &SparseMemoryImage::new(),
            0x1234,
            Streams::default(),
        );
        let mut exec_state = VmExecState::new(vm_state, ExecutionCtx::new(None));
        let byte_ptr = (2 * PAGE_SIZE + 17) as u32;

        exec_state.host_write_bytes(MEMORY_AS, byte_ptr, &[1, 2, 3, 4]);
        let snapshot = exec_state.vm_state.sparse_clone();
        exec_state.host_write_bytes(MEMORY_AS, byte_ptr, &[5, 6, 7, 8]);

        assert_eq!(snapshot.pc(), 0x1234);
        assert_eq!(
            unsafe { snapshot.memory.read_bytes::<4>(MEMORY_AS, byte_ptr) },
            [1, 2, 3, 4]
        );
    }
}
