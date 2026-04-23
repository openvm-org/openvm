//! Metered cost execution: per-chip trace cost tracking matching OpenVM's `MeteredCostCtx`.

use std::sync::Arc;

use openvm_instructions::{exe::VmExe, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_platform::memory::MEM_SIZE;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_lift::ExtensionRegistry;
use rvr_state::{GuardedMemory, TracerState};

use super::{
    build_callbacks, build_io_state,
    compile::ChipMapping,
    compile_metered_cost, compile_metered_cost_with_extensions, execute_metered_cost,
    register_and_execute,
    state::{init_rvr_state_with_metered_cost, state_as_void_ptr},
};
use crate::{
    arch::{
        execution_mode::MeteredCostCtx,
        vm::{
            copy_guest_memory_to_rvr_memory, ensure_rvr_outcome, map_rvr_compile_error,
            map_rvr_execute_error, read_public_values_from_guest_memory,
            read_rv32_regs_from_guest_memory, state_from_rvr, streams_from_io_state,
            streams_to_io_seed, write_rvr_memory_to_guest_memory,
        },
        ExecutionError, ExecutorInventory, MeteredExecutor, StaticProgramError, Streams,
        SystemConfig, VmState,
    },
    system::memory::online::GuestMemory,
};

/// Configuration for mapping PCs and memory operations to metering costs.
pub struct MeteredCostConfig {
    /// pc_index -> chip_idx. pc_index = (pc - pc_base) / 4.
    /// `u32::MAX` means no chip (e.g., TERMINATE).
    pub pc_to_chip: Vec<u32>,
    pub pc_base: u32,
    /// Per-AIR widths (for cost = height * width).
    pub widths: Vec<usize>,
    /// Chip index for the HINT_STOREW/HINT_BUFFER executor (if present in program).
    pub hint_store_chip_idx: Option<u32>,
}

impl MeteredCostConfig {
    /// Extract chip mapping for compilation.
    pub fn chip_mapping(&self) -> ChipMapping {
        ChipMapping {
            pc_to_chip: self.pc_to_chip.clone(),
            hint_store_chip_idx: self.hint_store_chip_idx,
            chip_widths: Some(self.widths.iter().map(|&w| w as u64).collect()),
        }
    }
}

const NO_CHIP: u32 = u32::MAX;

pub struct RvrMeteredCostInstance<F: PrimeField32, E> {
    pub(crate) system_config: SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) inventory: Arc<ExecutorInventory<E>>,
    pub(crate) executor_idx_to_air_idx: Vec<usize>,
    pub(crate) extensions: ExtensionRegistry<F>,
}

/// Build a `MeteredCostConfig` from the program, executor inventory, AIR index mapping, and widths.
pub fn build_metered_cost_config<F, E>(
    exe: &VmExe<F>,
    inventory: &ExecutorInventory<E>,
    executor_idx_to_air_idx: &[usize],
    widths: &[usize],
    hint_buffer_opcode: Option<VmOpcode>,
) -> MeteredCostConfig
where
    F: PrimeField32,
{
    let program = &exe.program;
    let pc_base = program.pc_base;

    let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();

    let hint_store_chip_idx = hint_buffer_opcode.and_then(|opcode| {
        inventory
            .instruction_lookup
            .get(&opcode)
            .map(|&executor_idx| executor_idx_to_air_idx[executor_idx as usize] as u32)
    });

    let pc_to_chip: Vec<u32> = program
        .instructions_and_debug_infos
        .iter()
        .map(|slot| {
            if let Some((inst, _)) = slot {
                let opcode: VmOpcode = inst.opcode;
                if opcode == terminate_opcode {
                    NO_CHIP
                } else if let Some(&executor_idx) = inventory.instruction_lookup.get(&opcode) {
                    let air_idx = executor_idx_to_air_idx[executor_idx as usize];
                    air_idx as u32
                } else {
                    NO_CHIP
                }
            } else {
                NO_CHIP
            }
        })
        .collect();

    MeteredCostConfig {
        pc_to_chip,
        pc_base,
        widths: widths.to_vec(),
        hint_store_chip_idx,
    }
}

/// C-compatible metered cost meter data.
///
/// Layout must exactly match the C `Tracer` struct in `openvm_tracer_metered_cost.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeteredCostData {
    pub cost: u64,
    pub chip_widths: *const u64,
}

impl Default for MeteredCostData {
    fn default() -> Self {
        Self {
            cost: 0,
            chip_widths: std::ptr::null(),
        }
    }
}

/// Pointer wrapper stored in RvState's tracer field. Matches C `Tracer*` (8 bytes).
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct MeteredCostMeter(pub *mut MeteredCostData);

impl Default for MeteredCostMeter {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl TracerState for MeteredCostMeter {
    const KIND: u32 = 10;
}

impl std::ops::Deref for MeteredCostMeter {
    type Target = MeteredCostData;
    fn deref(&self) -> &MeteredCostData {
        unsafe { &*self.0 }
    }
}

impl std::ops::DerefMut for MeteredCostMeter {
    fn deref_mut(&mut self) -> &mut MeteredCostData {
        unsafe { &mut *self.0 }
    }
}

/// C-compatible pure tracer data.
///
/// Layout must exactly match the C `Tracer` struct in `openvm_tracer_pure.h`.
/// All tracing is no-op; suspension is handled by RvState's target_instret.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PureTracerData;

/// Pointer wrapper stored in RvState's tracer field. Matches C `Tracer*` (8 bytes).
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct PureTracer(pub *mut PureTracerData);

impl Default for PureTracer {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl TracerState for PureTracer {
    const KIND: u32 = 12;
}

impl std::ops::Deref for PureTracer {
    type Target = PureTracerData;
    fn deref(&self) -> &PureTracerData {
        unsafe { &*self.0 }
    }
}

impl std::ops::DerefMut for PureTracer {
    fn deref_mut(&mut self) -> &mut PureTracerData {
        unsafe { &mut *self.0 }
    }
}

/// Prepare metering data from a `MeteredCostConfig`.
///
/// Returns `widths_u64` for the C tracer.
pub fn prepare_metered_cost(config: &MeteredCostConfig) -> Vec<u64> {
    config.widths.iter().map(|&w| w as u64).collect()
}

impl<F, E> RvrMeteredCostInstance<F, E>
where
    F: PrimeField32,
    E: MeteredExecutor<F>,
{
    pub fn execute_metered_cost(
        &self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<F, GuestMemory>), ExecutionError> {
        let inputs = inputs.into();
        let metered_cost_config = build_metered_cost_config(
            self.exe.as_ref(),
            self.inventory.as_ref(),
            &self.executor_idx_to_air_idx,
            &ctx.widths,
            None,
        );
        let chips = metered_cost_config.chip_mapping();

        let compiled = if self.extensions.is_empty() {
            compile_metered_cost(self.exe.as_ref(), &chips)
        } else {
            compile_metered_cost_with_extensions(self.exe.as_ref(), &self.extensions, &chips)
        }
        .map_err(map_rvr_compile_error)?;

        let result = execute_metered_cost(
            &compiled,
            self.exe.as_ref(),
            inputs.input_stream,
            metered_cost_config,
            Default::default(),
        )
        .map_err(map_rvr_execute_error)?;

        let mut output_ctx = ctx;
        output_ctx.instret = result.instret;
        output_ctx.cost = result.cost;

        let state = state_from_rvr(
            &self.system_config,
            self.exe.as_ref(),
            result.state.pc,
            &result.state.regs,
            &result.memory,
            &[],
        );

        Ok((output_ctx, state))
    }

    pub fn execute_metered_cost_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<F, GuestMemory>), ExecutionError> {
        let pc = from_state.pc();
        let mut guest_memory = from_state.memory;
        let (input_stream, hint_stream, deferrals) = streams_to_io_seed(from_state.streams);
        let rng = from_state.rng;
        #[cfg(feature = "metrics")]
        let metrics = from_state.metrics;

        let metered_cost_config = build_metered_cost_config(
            self.exe.as_ref(),
            self.inventory.as_ref(),
            &self.executor_idx_to_air_idx,
            &ctx.widths,
            None,
        );
        let chips = metered_cost_config.chip_mapping();

        let compiled = if self.extensions.is_empty() {
            compile_metered_cost(self.exe.as_ref(), &chips)
        } else {
            compile_metered_cost_with_extensions(self.exe.as_ref(), &self.extensions, &chips)
        }
        .map_err(map_rvr_compile_error)?;

        let mut memory = GuardedMemory::new(MEM_SIZE).map_err(|err| {
            ExecutionError::Static(StaticProgramError::FailToGenerateDynamicLibrary {
                err: err.to_string(),
            })
        })?;
        let mut tracer_data = MeteredCostData::default();
        let mut state = init_rvr_state_with_metered_cost(self.exe.as_ref(), &mut memory);
        state.tracer = MeteredCostMeter(&mut tracer_data);
        state.pc = pc;
        state
            .regs
            .copy_from_slice(&read_rv32_regs_from_guest_memory(&guest_memory));
        copy_guest_memory_to_rvr_memory(&guest_memory, &mut memory);

        let widths_u64 = prepare_metered_cost(&metered_cost_config);
        state.tracer.chip_widths = widths_u64.as_ptr();
        state.tracer.cost = 0;

        let mut io_state = build_io_state(input_stream, memory.as_ptr(), Default::default());
        io_state.hint_stream = hint_stream;
        io_state.hint_pos = 0;
        io_state.public_values = read_public_values_from_guest_memory(&guest_memory);
        io_state.rng = rng;
        let callbacks = build_callbacks(&mut io_state);
        unsafe { register_and_execute(&compiled, &callbacks, state_as_void_ptr(&mut state)) }
            .map_err(map_rvr_execute_error)?;
        ensure_rvr_outcome(
            "metered-cost execution from state",
            state.is_terminated(),
            state.is_suspended(),
            state.result_code(),
            false,
        )?;

        let mut output_ctx = ctx;
        output_ctx.instret = state.instret;
        output_ctx.cost = state.tracer.cost;

        write_rvr_memory_to_guest_memory(
            &mut guest_memory,
            &state.regs,
            &memory,
            &io_state.public_values,
        );
        let to_state = VmState::new(
            state.pc,
            guest_memory,
            streams_from_io_state(&io_state, deferrals),
            io_state.rng,
            #[cfg(feature = "metrics")]
            metrics,
        );

        Ok((output_ctx, to_state))
    }
}
