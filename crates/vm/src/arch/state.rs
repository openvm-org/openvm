use std::{
    fmt::Debug,
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
    system::memory::online::GuestMemory,
};

/// Represents the core state of a VM.
#[derive(derive_new::new, CopyGetters, MutGetters, Clone)]
pub struct VmState<F, MEM = GuestMemory> {
    #[getset(get_copy = "pub", get_mut = "pub")]
    instret: u64,
    #[getset(get_copy = "pub", get_mut = "pub")]
    pc: u32,
    pub memory: MEM,
    pub streams: Streams<F>,
    pub rng: StdRng,
    /// The public values of the PublicValuesAir when it exists
    pub(crate) custom_pvs: Vec<Option<F>>,
    #[cfg(feature = "metrics")]
    pub metrics: VmMetrics,
}

pub(super) const DEFAULT_RNG_SEED: u64 = 0;

impl<F: Clone, MEM> VmState<F, MEM> {
    /// `num_custom_pvs` should only be nonzero when the PublicValuesAir exists.
    pub fn new_with_defaults(
        instret: u64,
        pc: u32,
        memory: MEM,
        streams: impl Into<Streams<F>>,
        seed: u64,
        num_custom_pvs: usize,
    ) -> Self {
        Self {
            instret,
            pc,
            memory,
            streams: streams.into(),
            rng: StdRng::seed_from_u64(seed),
            custom_pvs: vec![None; num_custom_pvs],
            #[cfg(feature = "metrics")]
            metrics: VmMetrics::default(),
        }
    }

    #[inline(always)]
    pub fn set_instret_and_pc(&mut self, instret: u64, pc: u32) {
        self.instret = instret;
        self.pc = pc;
    }

    #[inline(always)]
    pub fn into_mut<'a, RA>(&'a mut self, ctx: &'a mut RA) -> VmStateMut<'a, F, MEM, RA> {
        VmStateMut {
            pc: &mut self.pc,
            memory: &mut self.memory,
            streams: &mut self.streams,
            rng: &mut self.rng,
            custom_pvs: &mut self.custom_pvs,
            ctx,
            #[cfg(feature = "metrics")]
            metrics: &mut self.metrics,
        }
    }
}

impl<F: Clone> VmState<F, GuestMemory> {
    #[instrument(name = "VmState::initial", level = "debug", skip_all)]
    pub fn initial(
        system_config: &SystemConfig,
        init_memory: &SparseMemoryImage,
        pc_start: u32,
        inputs: impl Into<Streams<F>>,
    ) -> Self {
        let memory = create_memory_image(&system_config.memory_config, init_memory);
        let num_custom_pvs = if system_config.has_public_values_chip() {
            system_config.num_public_values
        } else {
            0
        };
        VmState::new_with_defaults(
            0,
            pc_start,
            memory,
            inputs.into(),
            DEFAULT_RNG_SEED,
            num_custom_pvs,
        )
    }

    pub fn reset(
        &mut self,
        init_memory: &SparseMemoryImage,
        pc_start: u32,
        streams: impl Into<Streams<F>>,
    ) {
        self.instret = 0;
        self.pc = pc_start;
        self.memory.memory.fill_zero();
        self.memory.memory.set_from_sparse(init_memory);
        self.streams = streams.into();
        self.rng = StdRng::seed_from_u64(DEFAULT_RNG_SEED);
    }
}

/// Represents the full execution state of a VM during execution.
/// The global state is generic in guest memory `MEM` and additional context `CTX`.
/// The host state is execution context specific.
// @dev: Do not confuse with `ExecutionState` struct.
pub struct VmExecState<F, MEM, CTX> {
    /// Core VM state
    pub vm_state: VmState<F, MEM>,
    /// Execution-specific fields
    pub exit_code: Result<Option<u32>, ExecutionError>,
    pub ctx: CTX,
}

impl<F, MEM, CTX> VmExecState<F, MEM, CTX> {
    pub fn new(vm_state: VmState<F, MEM>, ctx: CTX) -> Self {
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
        VmState<F, MEM>: Clone,
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

impl<F, MEM, CTX> Deref for VmExecState<F, MEM, CTX> {
    type Target = VmState<F, MEM>;

    fn deref(&self) -> &Self::Target {
        &self.vm_state
    }
}

impl<F, MEM, CTX> DerefMut for VmExecState<F, MEM, CTX> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vm_state
    }
}

impl<F, CTX> VmExecState<F, GuestMemory, CTX>
where
    CTX: ExecutionCtxTrait,
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
        // SAFETY:
        // - T is stack-allocated repr(C) or repr(transparent), usually u8 or F where F is the base
        //   field
        // - T is the exact memory cell type for this address space, satisfying the type requirement
        unsafe { self.memory.read(addr_space, ptr) }
    }

    #[inline(always)]
    pub fn host_write<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; BLOCK_SIZE],
    ) {
        // SAFETY:
        // - T is stack-allocated repr(C) or repr(transparent), usually u8 or F where F is the base
        //   field
        // - T is the exact memory cell type for this address space, satisfying the type requirement
        unsafe { self.memory.write(addr_space, ptr, *data) }
    }

    #[inline(always)]
    pub fn host_read_slice<T: Copy + Debug>(&self, addr_space: u32, ptr: u32, len: usize) -> &[T] {
        // SAFETY:
        // - T is stack-allocated repr(C) or repr(transparent), usually u8 or F where F is the base
        //   field
        // - T is the exact memory cell type for this address space, satisfying the type requirement
        // - panics if the slice is out of bounds
        unsafe { self.memory.get_slice(addr_space, ptr, len) }
    }
}
