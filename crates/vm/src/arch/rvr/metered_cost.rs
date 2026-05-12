//! Metered cost execution: per-chip trace cost tracking matching OpenVM's `MeteredCostCtx`.

use std::sync::Arc;

use openvm_instructions::{exe::VmExe, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_lift::{ExtensionRegistry, NO_CHIP};

use super::{
    bridge::{map_rvr_compile_error, map_rvr_execute_error},
    compile::ChipMapping,
    compile_metered_cost, compile_metered_cost_with_extensions, execute_metered_cost,
    state::{TracerPayload, TracerPtr},
};
use crate::{
    arch::{
        execution_mode::MeteredCostCtx, ExecutionError, ExecutorInventory, MeteredExecutor,
        Streams, SystemConfig, VmState,
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
}

impl MeteredCostConfig {
    /// Extract chip mapping for compilation.
    pub fn chip_mapping(&self) -> ChipMapping {
        ChipMapping {
            pc_to_chip: self.pc_to_chip.clone(),
            chip_widths: Some(self.widths.iter().map(|&w| w as u64).collect()),
        }
    }
}

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
) -> MeteredCostConfig
where
    F: PrimeField32,
{
    let program = &exe.program;
    let pc_base = program.pc_base;

    let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();

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

impl TracerPayload for MeteredCostData {
    const KIND: u32 = 10;
}

pub type MeteredCostMeter = TracerPtr<MeteredCostData>;

/// C-compatible pure tracer data.
///
/// Layout must exactly match the C `Tracer` struct in `openvm_tracer_pure.h`.
/// All tracing is no-op; suspension is handled by RvState's target_instret.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PureTracerData;

impl TracerPayload for PureTracerData {
    const KIND: u32 = 12;
}

pub type PureTracer = TracerPtr<PureTracerData>;

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
        let vm_state = VmState::initial(
            &self.system_config,
            &self.exe.init_memory,
            self.exe.pc_start,
            inputs,
        );
        self.execute_metered_cost_from_state(vm_state, ctx)
    }

    pub fn execute_metered_cost_from_state(
        &self,
        mut vm_state: VmState<F, GuestMemory>,
        ctx: MeteredCostCtx,
    ) -> Result<(MeteredCostCtx, VmState<F, GuestMemory>), ExecutionError> {
        let metered_cost_config = build_metered_cost_config(
            self.exe.as_ref(),
            self.inventory.as_ref(),
            &self.executor_idx_to_air_idx,
            &ctx.widths,
        );
        let chips = metered_cost_config.chip_mapping();

        let compiled = if self.extensions.is_empty() {
            compile_metered_cost(self.exe.as_ref(), &chips)
        } else {
            compile_metered_cost_with_extensions(self.exe.as_ref(), &self.extensions, &chips)
        }
        .map_err(map_rvr_compile_error)?;

        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        let result = tracing::info_span!("execute_metered_cost")
            .in_scope(|| execute_metered_cost(&compiled, &mut vm_state, metered_cost_config, None))
            .map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed();
            let insns = result.state.instret;
            tracing::info!("instructions_executed={insns}");
            metrics::counter!("execute_metered_cost_insns").absolute(insns);
            metrics::gauge!("execute_metered_cost_insn_mi/s")
                .set(insns as f64 / elapsed.as_micros() as f64);
        }

        let mut output_ctx = ctx;
        output_ctx.instret = result.state.instret;
        output_ctx.cost = result.cost;

        Ok((output_ctx, vm_state))
    }
}
