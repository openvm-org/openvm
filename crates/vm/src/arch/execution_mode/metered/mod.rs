pub mod bounded;
pub mod exact;

// pub use exact::MeteredCtxExact as MeteredCtx;
pub use bounded::MeteredCtxBounded as MeteredCtx;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{p3_field::PrimeField32, ChipUsageGetter};
use p3_baby_bear::BabyBear;

use crate::arch::{
    execution_control::ExecutionControl, execution_mode::metered::bounded::Segment, ChipId,
    ExecutionError, InsExecutorE1, VmChipComplex, VmConfig, VmSegmentState, VmStateMut,
    CONNECTOR_AIR_ID, PROGRAM_AIR_ID, PUBLIC_VALUES_AIR_ID,
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: u64 = 100;

// TODO(ayush): fix these values
const MAX_TRACE_HEIGHT: u32 = (1 << 23) - 100;
const MAX_TRACE_CELLS: usize = 1_200_000_000; // 1.2B
const MAX_INTERACTIONS: usize = BabyBear::ORDER_U32 as usize;

pub struct MeteredExecutionControl<'a> {
    // Constants
    air_names: &'a [String],
    pub widths: &'a [usize],
    pub interactions: &'a [usize],
}

impl<'a> MeteredExecutionControl<'a> {
    pub fn new(air_names: &'a [String], widths: &'a [usize], interactions: &'a [usize]) -> Self {
        Self {
            air_names,
            widths,
            interactions,
        }
    }

    /// Calculate the total cells used based on trace heights and widths
    fn calculate_total_cells(&self, trace_heights: &[u32]) -> usize {
        trace_heights
            .iter()
            .zip(self.widths)
            .map(|(&height, &width)| height as usize * width)
            .sum()
    }

    /// Calculate the total interactions based on trace heights and interaction counts
    fn calculate_total_interactions(&self, trace_heights: &[u32]) -> usize {
        trace_heights
            .iter()
            .zip(self.interactions)
            // We add 1 for the zero messages from the padding rows
            .map(|(&height, &interactions)| (height + 1) as usize * interactions)
            .sum()
    }

    fn should_segment(&self, state: &mut VmSegmentState<MeteredCtx>) -> bool {
        let trace_heights = state.ctx.trace_heights_if_finalized();
        for (i, &height) in trace_heights.iter().enumerate() {
            if height > MAX_TRACE_HEIGHT {
                tracing::info!(
                    "Segment {:2} | clk {:9} | chip {} ({}) height ({:8}) > max ({:8})",
                    state.ctx.segments.len(),
                    state.ctx.clk_last_segment_check,
                    i,
                    self.air_names[i],
                    height,
                    MAX_TRACE_HEIGHT
                );
                return true;
            }
        }

        let total_cells = self.calculate_total_cells(&trace_heights);
        if total_cells > MAX_TRACE_CELLS {
            tracing::info!(
                "Segment {:2} | clk {:9} | total cells ({:10}) > max ({:10})",
                state.ctx.segments.len(),
                state.ctx.clk_last_segment_check,
                total_cells,
                MAX_TRACE_CELLS
            );
            return true;
        }

        let total_interactions = self.calculate_total_interactions(&trace_heights);
        if total_interactions > MAX_INTERACTIONS {
            tracing::info!(
                "Segment {:2} | clk {:9} | total interactions ({:11}) > max ({:11})",
                state.ctx.segments.len(),
                state.ctx.clk_last_segment_check,
                total_interactions,
                MAX_INTERACTIONS
            );
            return true;
        }

        false
    }

    fn reset_segment<F, VC>(
        &self,
        state: &mut VmSegmentState<MeteredCtx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) where
        F: PrimeField32,
        VC: VmConfig<F>,
    {
        state.ctx.leaf_indices.clear();

        // TODO(ayush): only reset trace heights for chips that are not constant height instead
        // of refilling again
        state.ctx.trace_heights.fill(0);

        // Program | Connector | Public Values | Memory ... | Executors (except Public Values) |
        // Range Checker
        state.ctx.trace_heights[PROGRAM_AIR_ID] =
            chip_complex.program_chip().true_program_length as u32;
        state.ctx.trace_heights[CONNECTOR_AIR_ID] = 2;

        let mut offset = if chip_complex.config().has_public_values_chip() {
            PUBLIC_VALUES_AIR_ID + 1
        } else {
            PUBLIC_VALUES_AIR_ID
        };
        offset += chip_complex.memory_controller().num_airs();

        // Periphery chips with constant heights
        for (i, chip_id) in chip_complex
            .inventory
            .insertion_order
            .iter()
            .rev()
            .enumerate()
        {
            if let &ChipId::Periphery(id) = chip_id {
                if let Some(constant_height) =
                    chip_complex.inventory.periphery[id].constant_trace_height()
                {
                    state.ctx.trace_heights[offset + i] = constant_height as u32;
                }
            }
        }

        // Range checker chip
        if let (Some(range_checker_height), Some(last_height)) = (
            chip_complex.range_checker_chip().constant_trace_height(),
            state.ctx.trace_heights.last_mut(),
        ) {
            *last_height = range_checker_height as u32;
        }
    }

    fn check_segment_limits<F, VC>(
        &self,
        state: &mut VmSegmentState<MeteredCtx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) where
        F: PrimeField32,
        VC: VmConfig<F>,
    {
        // Avoid checking segment too often.
        if state.clk < state.ctx.clk_last_segment_check + SEGMENT_CHECK_INTERVAL {
            return;
        }

        if self.should_segment(state) {
            let clk_start = state
                .ctx
                .segments
                .last()
                .map_or(0, |s| s.clk_start + s.num_cycles);
            let segment = Segment {
                clk_start,
                num_cycles: state.ctx.clk_last_segment_check - clk_start,
                // TODO(ayush): this is trace heights after overflow so an overestimate
                trace_heights: state.ctx.trace_heights.clone(),
            };
            state.ctx.segments.push(segment);
            self.reset_segment::<F, VC>(state, chip_complex);
        }

        state.ctx.clk_last_segment_check = state.clk;
    }
}

impl<F, VC> ExecutionControl<F, VC> for MeteredExecutionControl<'_>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    type Ctx = MeteredCtx;

    fn initialize_context(&self) -> Self::Ctx {
        todo!()
    }

    fn should_suspend(
        &self,
        _state: &mut VmSegmentState<Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        false
    }

    fn on_start(
        &self,
        state: &mut VmSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        self.reset_segment::<F, VC>(state, chip_complex);
    }

    fn on_suspend_or_terminate(
        &self,
        state: &mut VmSegmentState<Self::Ctx>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        _exit_code: Option<u32>,
    ) {
        state.ctx.finalize_access_adapter_heights();

        tracing::info!(
            "Segment {:2} | clk {:9} | terminated",
            state.ctx.segments.len(),
            state.clk,
        );
        // Add the last segment
        let clk_start = state
            .ctx
            .segments
            .last()
            .map_or(0, |s| s.clk_start + s.num_cycles);
        let segment = Segment {
            clk_start,
            num_cycles: state.clk - clk_start,
            // TODO(ayush): this is trace heights after overflow so an overestimate
            trace_heights: state.ctx.trace_heights.clone(),
        };
        state.ctx.segments.push(segment);
    }

    /// Execute a single instruction
    fn execute_instruction(
        &self,
        state: &mut VmSegmentState<Self::Ctx>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        // Check if segmentation needs to happen
        self.check_segment_limits::<F, VC>(state, chip_complex);

        let mut offset = if chip_complex.config().has_public_values_chip() {
            PUBLIC_VALUES_AIR_ID + 1
        } else {
            PUBLIC_VALUES_AIR_ID
        };
        offset += chip_complex.memory_controller().num_airs();

        let &Instruction { opcode, .. } = instruction;
        if let Some((executor, i)) = chip_complex.inventory.get_mut_executor_with_index(&opcode) {
            let mut vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: state.memory.as_mut().unwrap(),
                ctx: &mut state.ctx,
            };
            executor.execute_metered(&mut vm_state, instruction, offset + i)?;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };
        state.clk += 1;

        Ok(())
    }
}
