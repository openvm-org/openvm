//! Metered cost execution: per-chip trace cost tracking matching OpenVM's `MeteredCostCtx`.

use openvm_circuit::arch::{ExecutorInventory, SystemConfig};
use openvm_instructions::{exe::VmExe, LocalOpcode, SystemOpcode, VmOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::TracerState;

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
    pub fn chip_mapping(&self) -> crate::compile::ChipMapping {
        crate::compile::ChipMapping {
            pc_to_chip: self.pc_to_chip.clone(),
            hint_store_chip_idx: self.hint_store_chip_idx,
            chip_widths: Some(self.widths.iter().map(|&w| w as u64).collect()),
        }
    }
}

const NO_CHIP: u32 = u32::MAX;

/// Build a `MeteredCostConfig` from the program, executor inventory, AIR index mapping, and widths.
pub fn build_metered_cost_config<F, E>(
    exe: &VmExe<F>,
    inventory: &ExecutorInventory<E>,
    executor_idx_to_air_idx: &[usize],
    widths: &[usize],
    _system_config: &SystemConfig,
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
