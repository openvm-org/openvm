//! GPU postflight expansion, indexing, and read-only replay for compiled preflight.
//!
//! This module owns three phases:
//!
//! 1. re-execute checkpoint intervals to derive logical program and memory events;
//! 2. build memory chronology and opcode indexes once for the segment; and
//! 3. expose the immutable result to system and instruction trace generators.
//!
//! Checkpoints and residuals are authoritative executor output. Program events,
//! memory events, predecessors, and first-write values are derived GPU data.

use std::sync::Arc;

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::{CudaError, MemCopyError},
    stream::GpuDeviceCtx,
};
use openvm_instructions::{
    instruction::Instruction,
    program::Program,
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_BYTES},
    LocalOpcode, SystemOpcode, VmOpcode, DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use p3_baby_bear::BabyBear;
use rvr_state::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
    RvrCheckpoint,
};
use thiserror::Error;

#[cfg(feature = "test-utils")]
use super::PreflightEventLog;
use super::{bridge::read_rv64_registers, preflight::PreflightExecution, PreflightEndpoint};
use crate::{
    arch::{
        to_byte_ptr_bits, ExecutionState, MemoryCellType, MemoryConfig, PreflightHistory,
        ADDR_SPACE_OFFSET, BLOCK_FE_WIDTH,
    },
    cuda_abi::{rvr_checkpoint_replay, rvr_postflight},
    system::TouchedBlock,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PostflightInstruction {
    /// Global opcode followed by the seven canonical instruction operands.
    pub words: [u32; 8],
}

/// Location of one executed instruction and the first timed memory event in
/// its timestamp interval. The final program sentinel has no entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RvrReplayStep {
    pub program_index: u32,
    pub memory_start: u32,
}

const _: () = assert!(size_of::<PostflightInstruction>() == 32);
const _: () = assert!(size_of::<TouchedBlock<BabyBear>>() == size_of::<[u32; 8]>());
const _: () = assert!(RV64_REGISTER_AS == ADDR_SPACE_OFFSET);

type PostflightFieldBlock = PreflightFieldBlock;

const _: () = assert!(size_of::<PostflightFieldBlock>() == 4 * size_of::<u32>());

/// Compact opcode-family ABI for checkpoint replay. One base identifies each
/// supported contiguous family; this is passed by value and never uploaded as
/// a per-segment opcode table.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PostflightOpcodeBases {
    pub base_alu: u32,
    pub shift: u32,
    pub less_than: u32,
    pub load_store: u32,
    pub branch_equal: u32,
    pub branch_less_than: u32,
    pub jal_lui: u32,
    pub jalr: u32,
    pub auipc: u32,
    pub mul: u32,
    pub mulh: u32,
    pub divrem: u32,
    pub base_alu_w: u32,
    pub shift_w: u32,
    pub mul_w: u32,
    pub divrem_w: u32,
    pub base_alu_imm: u32,
    pub shift_imm: u32,
    pub less_than_imm: u32,
    pub base_alu_w_imm: u32,
    pub shift_w_imm: u32,
    pub hint_store: u32,
    pub phantom: u32,
    pub terminate: u32,
}

const _: () = assert!(size_of::<PostflightOpcodeBases>() == 24 * size_of::<u32>());

impl PostflightOpcodeBases {
    #[doc(hidden)]
    pub fn owns(self, opcode: u32) -> bool {
        let family =
            |base: u32, count: u32| opcode.checked_sub(base).is_some_and(|local| local < count);
        family(self.base_alu, 5)
            || family(self.shift, 3)
            || family(self.less_than, 2)
            || family(self.load_store, 11)
            || family(self.branch_equal, 2)
            || family(self.branch_less_than, 4)
            || family(self.jal_lui, 2)
            || opcode == self.jalr
            || opcode == self.auipc
            || opcode == self.mul
            || family(self.mulh, 3)
            || family(self.divrem, 4)
            || family(self.base_alu_w, 2)
            || family(self.shift_w, 3)
            || opcode == self.mul_w
            || family(self.divrem_w, 4)
            || family(self.base_alu_imm, 4)
            || family(self.shift_imm, 3)
            || family(self.less_than_imm, 2)
            || opcode == self.base_alu_w_imm
            || family(self.shift_w_imm, 3)
            || family(self.hint_store, 2)
            || opcode == self.phantom
            || opcode == self.terminate
    }
}

const RVR_CHECKPOINT_NO_SCHEDULE: u32 = u32::MAX;
const RVR_CHECKPOINT_MAX_DENSE_OPCODE: u32 = u16::MAX as u32;
const RVR_CHECKPOINT_EFFECT_NEXT: u8 = 0;
const RVR_CHECKPOINT_EFFECT_BRANCH_RESIDUAL: u8 = 1;
const RVR_CHECKPOINT_REGISTER_WRITE_NONE: u8 = 0;
const RVR_CHECKPOINT_REGISTER_WRITE_ZERO: u8 = 1;
const RVR_CHECKPOINT_REGISTER_WRITE_RESIDUAL: u8 = 2;
const RVR_CHECKPOINT_SPAN_BASE_REGISTER: u8 = 0;
const RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_INPUT: u8 = 1;
const RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_OUTPUT: u8 = 2;
const RVR_CHECKPOINT_SPAN_COUNT_FIXED: u8 = 0;
const RVR_CHECKPOINT_SPAN_COUNT_REGISTER: u8 = 1;
const RVR_CHECKPOINT_SPAN_COUNT_RESIDUAL: u8 = 2;
const RVR_CHECKPOINT_SPAN_READ_U16: u8 = 0;
const RVR_CHECKPOINT_SPAN_WRITE_U16_RESIDUAL: u8 = 1;
const RVR_CHECKPOINT_SPAN_WRITE_U16_ZERO: u8 = 2;
const RVR_CHECKPOINT_SPAN_READ_FIELD32: u8 = 3;
const RVR_CHECKPOINT_SPAN_WRITE_FIELD32_CANONICAL_PAIRS: u8 = 4;
const RVR_CHECKPOINT_SPAN_WRITE_U16_STATIC: u8 = 5;
const RVR_CHECKPOINT_DEFERRAL_DIGEST_BLOCKS: u32 = 2;

/// One contiguous sequence of fixed-width memory-bus accesses in an
/// extension-owned checkpoint replay schedule. U16 spans access eight bytes per
/// event; FIELD32 spans access four field cells per event.
///
/// The finite source tags distinguish ordinary RV64 heap blocks from Deferral
/// accumulator blocks. This is intentionally not a general address-expression
/// language: each supported source has one canonical interpretation in replay.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PostflightAccessSpan {
    address_space: u32,
    count: u32,
    base_index: u8,
    base_source: u8,
    count_register: u8,
    count_shift: u8,
    count_source: u8,
    value_source: u8,
    value_index: u16,
}

const _: () = assert!(size_of::<PostflightAccessSpan>() == 16);
const _: () = assert!(std::mem::offset_of!(PostflightAccessSpan, address_space) == 0);
const _: () = assert!(std::mem::offset_of!(PostflightAccessSpan, count) == 4);
const _: () = assert!(std::mem::offset_of!(PostflightAccessSpan, base_index) == 8);
const _: () = assert!(std::mem::offset_of!(PostflightAccessSpan, value_source) == 13);
const _: () = assert!(std::mem::offset_of!(PostflightAccessSpan, value_index) == 14);

impl PostflightAccessSpan {
    pub const fn read_fixed(address_space: u32, base_register: u8, count: u32) -> Self {
        Self {
            address_space,
            count,
            base_index: base_register,
            base_source: RVR_CHECKPOINT_SPAN_BASE_REGISTER,
            count_register: 0,
            count_shift: 0,
            count_source: RVR_CHECKPOINT_SPAN_COUNT_FIXED,
            value_source: RVR_CHECKPOINT_SPAN_READ_U16,
            value_index: 0,
        }
    }

    pub const fn write_fixed_from_residuals(
        address_space: u32,
        base_register: u8,
        count: u32,
    ) -> Self {
        Self {
            value_source: RVR_CHECKPOINT_SPAN_WRITE_U16_RESIDUAL,
            ..Self::read_fixed(address_space, base_register, count)
        }
    }

    /// A fixed span whose postimage is statically zero. This consumes the AIR's
    /// write slots without adding redundant zeroes to the serial transcript.
    pub const fn write_fixed_zero(address_space: u32, base_register: u8, count: u32) -> Self {
        Self {
            value_source: RVR_CHECKPOINT_SPAN_WRITE_U16_ZERO,
            ..Self::read_fixed(address_space, base_register, count)
        }
    }

    pub const fn read_count_from_register(
        address_space: u32,
        base_register: u8,
        count_register: u8,
        count_shift: u8,
        max_count: u32,
    ) -> Self {
        Self {
            address_space,
            count: max_count,
            base_index: base_register,
            base_source: RVR_CHECKPOINT_SPAN_BASE_REGISTER,
            count_register,
            count_shift,
            count_source: RVR_CHECKPOINT_SPAN_COUNT_REGISTER,
            value_source: RVR_CHECKPOINT_SPAN_READ_U16,
            value_index: 0,
        }
    }

    pub const fn write_count_from_register_from_residuals(
        address_space: u32,
        base_register: u8,
        count_register: u8,
        count_shift: u8,
        max_count: u32,
    ) -> Self {
        Self {
            value_source: RVR_CHECKPOINT_SPAN_WRITE_U16_RESIDUAL,
            ..Self::read_count_from_register(
                address_space,
                base_register,
                count_register,
                count_shift,
                max_count,
            )
        }
    }

    /// A variable-size write whose next residual is the number of eight-byte
    /// blocks and whose following residuals are the block postimages.
    pub const fn write_count_from_residual_from_residuals(
        address_space: u32,
        base_register: u8,
        max_count: u32,
    ) -> Self {
        Self {
            count_source: RVR_CHECKPOINT_SPAN_COUNT_RESIDUAL,
            value_source: RVR_CHECKPOINT_SPAN_WRITE_U16_RESIDUAL,
            ..Self::read_fixed(address_space, base_register, max_count)
        }
    }

    /// Two consecutive four-cell reads of a Deferral input accumulator. The
    /// base is `16 * instruction[deferral_idx_operand]` in AS4 cell units.
    pub const fn read_deferral_input_accumulator(deferral_idx_operand: u8) -> Self {
        Self::deferral_accumulator(
            deferral_idx_operand,
            RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_INPUT,
            RVR_CHECKPOINT_SPAN_READ_FIELD32,
        )
    }

    /// Two consecutive four-cell reads of a Deferral output accumulator. The
    /// base is `16 * instruction[deferral_idx_operand] + 8` in AS4 cell units.
    pub const fn read_deferral_output_accumulator(deferral_idx_operand: u8) -> Self {
        Self::deferral_accumulator(
            deferral_idx_operand,
            RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_OUTPUT,
            RVR_CHECKPOINT_SPAN_READ_FIELD32,
        )
    }

    /// Two consecutive four-cell writes of a Deferral input accumulator. Each
    /// block consumes two u64 residuals containing four canonical u32 cells.
    pub const fn write_deferral_input_accumulator(deferral_idx_operand: u8) -> Self {
        Self::deferral_accumulator(
            deferral_idx_operand,
            RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_INPUT,
            RVR_CHECKPOINT_SPAN_WRITE_FIELD32_CANONICAL_PAIRS,
        )
    }

    /// Two consecutive four-cell writes of a Deferral output accumulator.
    pub const fn write_deferral_output_accumulator(deferral_idx_operand: u8) -> Self {
        Self::deferral_accumulator(
            deferral_idx_operand,
            RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_OUTPUT,
            RVR_CHECKPOINT_SPAN_WRITE_FIELD32_CANONICAL_PAIRS,
        )
    }

    const fn deferral_accumulator(
        deferral_idx_operand: u8,
        base_source: u8,
        value_source: u8,
    ) -> Self {
        Self {
            address_space: DEFERRAL_AS,
            count: RVR_CHECKPOINT_DEFERRAL_DIGEST_BLOCKS,
            base_index: deferral_idx_operand,
            base_source,
            count_register: 0,
            count_shift: 0,
            count_source: RVR_CHECKPOINT_SPAN_COUNT_FIXED,
            value_source,
            value_index: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PostflightEventCount {
    pub(crate) memory: u32,
    pub(crate) field: u32,
}

const _: () = assert!(size_of::<PostflightEventCount>() == 2 * size_of::<u32>());
const _: () = assert!(std::mem::offset_of!(PostflightEventCount, memory) == 0);
const _: () = assert!(std::mem::offset_of!(PostflightEventCount, field) == 4);

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RvrCheckpointAccessSchedule {
    first_span: u32,
    num_spans: u32,
    register_operands: [u8; 3],
    num_register_reads: u8,
    effect: u8,
    effect_operand: u8,
    register_write_source: u8,
    register_write_operand: u8,
}

const _: () = assert!(size_of::<RvrCheckpointAccessSchedule>() == 16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RvrCheckpointInstructionLayout {
    zero_operand_mask: u32,
    register_as_operand: u8,
    memory_as_operand: u8,
}

/// Static extension access schedules uploaded once with a GPU program. They
/// describe access order only and are not part of the preflight transcript.
///
/// This is an internal composition seam, not a stable extension API. The
/// supported sources remain a finite POD set: residuals, zero, program-owned
/// constants, and Deferral's field accumulator blocks.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct PostflightAccessRegistry {
    dispatch: Vec<u32>,
    schedules: Vec<RvrCheckpointAccessSchedule>,
    instruction_layouts: Vec<RvrCheckpointInstructionLayout>,
    spans: Vec<PostflightAccessSpan>,
    static_values: Vec<u64>,
}

impl PostflightAccessRegistry {
    /// Adds one fixed sequence of eight-byte write values to the program-owned
    /// replay data and returns the span that consumes it. This is used for
    /// setup instructions whose postimage depends only on VM configuration.
    #[doc(hidden)]
    pub fn write_fixed_from_static(
        &mut self,
        address_space: u32,
        base_register: u8,
        values: &[u64],
    ) -> Result<PostflightAccessSpan, GpuPostflightError> {
        let count = u32::try_from(values.len()).map_err(|_| {
            GpuPostflightError::InvalidAccessSchedule("static write span is too large".to_string())
        })?;
        if count == 0 {
            return Err(GpuPostflightError::InvalidAccessSchedule(
                "static write span must not be empty".to_string(),
            ));
        }
        let value_index = u16::try_from(self.static_values.len()).map_err(|_| {
            GpuPostflightError::InvalidAccessSchedule(
                "static write table exceeds its u16 index domain".to_string(),
            )
        })?;
        self.static_values
            .len()
            .checked_add(values.len())
            .filter(|&end| end <= usize::from(u16::MAX) + 1)
            .ok_or_else(|| {
                GpuPostflightError::InvalidAccessSchedule(
                    "static write table exceeds its u16 index domain".to_string(),
                )
            })?;
        self.static_values.extend_from_slice(values);
        Ok(PostflightAccessSpan {
            value_source: RVR_CHECKPOINT_SPAN_WRITE_U16_STATIC,
            value_index,
            ..PostflightAccessSpan::read_fixed(address_space, base_register, count)
        })
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &mut self,
        opcode: u32,
        register_operands: &[u8],
        zero_operand_mask: u32,
        register_as_operand: u8,
        memory_as_operand: u8,
        spans: &[PostflightAccessSpan],
    ) -> Result<(), GpuPostflightError> {
        self.register_with_effect(
            opcode,
            register_operands,
            zero_operand_mask,
            register_as_operand,
            memory_as_operand,
            spans,
            RVR_CHECKPOINT_EFFECT_NEXT,
            0,
            RVR_CHECKPOINT_REGISTER_WRITE_NONE,
            0,
        )
    }

    /// Registers a schedule whose final clock slot writes zero to a register.
    /// An x0 destination reserves the slot but emits no memory event.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn register_with_zero_register_write(
        &mut self,
        opcode: u32,
        register_operands: &[u8],
        zero_operand_mask: u32,
        register_as_operand: u8,
        memory_as_operand: u8,
        spans: &[PostflightAccessSpan],
        write_operand: u8,
    ) -> Result<(), GpuPostflightError> {
        self.register_with_effect(
            opcode,
            register_operands,
            zero_operand_mask,
            register_as_operand,
            memory_as_operand,
            spans,
            RVR_CHECKPOINT_EFFECT_NEXT,
            0,
            RVR_CHECKPOINT_REGISTER_WRITE_ZERO,
            write_operand,
        )
    }

    /// Registers a schedule whose final clock slot writes one residual to a
    /// register. An x0 destination consumes the residual and slot but emits no
    /// memory event.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn register_with_residual_register_write(
        &mut self,
        opcode: u32,
        register_operands: &[u8],
        zero_operand_mask: u32,
        register_as_operand: u8,
        memory_as_operand: u8,
        spans: &[PostflightAccessSpan],
        write_operand: u8,
    ) -> Result<(), GpuPostflightError> {
        self.register_with_effect(
            opcode,
            register_operands,
            zero_operand_mask,
            register_as_operand,
            memory_as_operand,
            spans,
            RVR_CHECKPOINT_EFFECT_NEXT,
            0,
            RVR_CHECKPOINT_REGISTER_WRITE_RESIDUAL,
            write_operand,
        )
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn register_branch_residual(
        &mut self,
        opcode: u32,
        register_operands: &[u8],
        zero_operand_mask: u32,
        register_as_operand: u8,
        memory_as_operand: u8,
        spans: &[PostflightAccessSpan],
        branch_operand: u8,
    ) -> Result<(), GpuPostflightError> {
        self.register_with_effect(
            opcode,
            register_operands,
            zero_operand_mask,
            register_as_operand,
            memory_as_operand,
            spans,
            RVR_CHECKPOINT_EFFECT_BRANCH_RESIDUAL,
            branch_operand,
            RVR_CHECKPOINT_REGISTER_WRITE_NONE,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_with_effect(
        &mut self,
        opcode: u32,
        register_operands: &[u8],
        zero_operand_mask: u32,
        register_as_operand: u8,
        memory_as_operand: u8,
        spans: &[PostflightAccessSpan],
        effect: u8,
        effect_operand: u8,
        register_write_source: u8,
        register_write_operand: u8,
    ) -> Result<(), GpuPostflightError> {
        if register_operands.len() > 3
            || register_operands
                .iter()
                .any(|&word| !(1..8).contains(&word))
            || !(1..8).contains(&register_as_operand)
            || !(1..8).contains(&memory_as_operand)
            || register_as_operand == memory_as_operand
            || register_operands
                .iter()
                .any(|&word| word == register_as_operand || word == memory_as_operand)
            || zero_operand_mask & 1 != 0
            || zero_operand_mask >> 8 != 0
            || register_operands
                .iter()
                .chain([register_as_operand, memory_as_operand].iter())
                .any(|&word| zero_operand_mask & (1 << word) != 0)
            || (effect == RVR_CHECKPOINT_EFFECT_NEXT && effect_operand != 0)
            || (effect == RVR_CHECKPOINT_EFFECT_BRANCH_RESIDUAL
                && (!(1..8).contains(&effect_operand)
                    || register_operands.contains(&effect_operand)
                    || effect_operand == register_as_operand
                    || effect_operand == memory_as_operand
                    || zero_operand_mask & (1 << effect_operand) != 0))
            || !matches!(
                effect,
                RVR_CHECKPOINT_EFFECT_NEXT | RVR_CHECKPOINT_EFFECT_BRANCH_RESIDUAL
            )
            || !matches!(
                register_write_source,
                RVR_CHECKPOINT_REGISTER_WRITE_NONE
                    | RVR_CHECKPOINT_REGISTER_WRITE_ZERO
                    | RVR_CHECKPOINT_REGISTER_WRITE_RESIDUAL
            )
            || (register_write_source == RVR_CHECKPOINT_REGISTER_WRITE_NONE
                && register_write_operand != 0)
            || (register_write_source != RVR_CHECKPOINT_REGISTER_WRITE_NONE
                && (!(1..8).contains(&register_write_operand)
                    || register_write_operand == register_as_operand
                    || register_write_operand == memory_as_operand
                    || zero_operand_mask & (1 << register_write_operand) != 0))
            || spans.is_empty()
        {
            return Err(GpuPostflightError::InvalidAccessSchedule(
                "invalid operand layout".to_string(),
            ));
        }
        if opcode > RVR_CHECKPOINT_MAX_DENSE_OPCODE {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "opcode {opcode} exceeds the dense checkpoint dispatch bound"
            )));
        }
        let dispatch_len = usize::try_from(opcode)
            .ok()
            .and_then(|opcode| opcode.checked_add(1))
            .ok_or_else(|| {
                GpuPostflightError::InvalidAccessSchedule(
                    "opcode dispatch length overflow".to_string(),
                )
            })?;
        if self.dispatch.len() < dispatch_len {
            self.dispatch
                .resize(dispatch_len, RVR_CHECKPOINT_NO_SCHEDULE);
        }
        if self.dispatch[opcode as usize] != RVR_CHECKPOINT_NO_SCHEDULE {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "duplicate checkpoint access schedule for opcode {opcode}"
            )));
        }
        for span in spans {
            let field = matches!(
                span.value_source,
                RVR_CHECKPOINT_SPAN_READ_FIELD32
                    | RVR_CHECKPOINT_SPAN_WRITE_FIELD32_CANONICAL_PAIRS
            );
            let base_valid = match span.base_source {
                RVR_CHECKPOINT_SPAN_BASE_REGISTER => {
                    usize::from(span.base_index) < register_operands.len()
                }
                RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_INPUT
                | RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_OUTPUT => {
                    (1..8).contains(&span.base_index)
                        && !register_operands.contains(&span.base_index)
                        && span.base_index != register_as_operand
                        && span.base_index != memory_as_operand
                        && zero_operand_mask & (1 << span.base_index) == 0
                }
                _ => false,
            };
            let count_valid = match span.count_source {
                RVR_CHECKPOINT_SPAN_COUNT_FIXED => {
                    span.count != 0 && span.count_register == 0 && span.count_shift == 0
                }
                RVR_CHECKPOINT_SPAN_COUNT_REGISTER => {
                    usize::from(span.count_register) < register_operands.len()
                        && span.count_shift < u64::BITS as u8
                }
                RVR_CHECKPOINT_SPAN_COUNT_RESIDUAL => {
                    span.count != 0 && span.count_register == 0 && span.count_shift == 0
                }
                _ => false,
            };
            let value_valid = matches!(
                span.value_source,
                RVR_CHECKPOINT_SPAN_READ_U16
                    | RVR_CHECKPOINT_SPAN_WRITE_U16_RESIDUAL
                    | RVR_CHECKPOINT_SPAN_WRITE_U16_ZERO
                    | RVR_CHECKPOINT_SPAN_READ_FIELD32
                    | RVR_CHECKPOINT_SPAN_WRITE_FIELD32_CANONICAL_PAIRS
                    | RVR_CHECKPOINT_SPAN_WRITE_U16_STATIC
            );
            let static_end = usize::from(span.value_index)
                .checked_add(span.count as usize)
                .filter(|&end| end <= self.static_values.len());
            if !base_valid
                || !count_valid
                || !value_valid
                || field != (span.address_space == DEFERRAL_AS)
                || (!field && span.address_space != RV64_MEMORY_AS)
                || (field
                    && !matches!(
                        span.base_source,
                        RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_INPUT
                            | RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_OUTPUT
                    ))
                || (!field && span.base_source != RVR_CHECKPOINT_SPAN_BASE_REGISTER)
                || (field && span.count_source != RVR_CHECKPOINT_SPAN_COUNT_FIXED)
                || (field && span.count != RVR_CHECKPOINT_DEFERRAL_DIGEST_BLOCKS)
                || (span.value_source == RVR_CHECKPOINT_SPAN_WRITE_U16_STATIC
                    && (span.count_source != RVR_CHECKPOINT_SPAN_COUNT_FIXED
                        || static_end.is_none()))
                || (span.value_source != RVR_CHECKPOINT_SPAN_WRITE_U16_STATIC
                    && span.value_index != 0)
            {
                return Err(GpuPostflightError::InvalidAccessSchedule(
                    "invalid access span".to_string(),
                ));
            }
        }
        let first_span = u32::try_from(self.spans.len()).map_err(|_| {
            GpuPostflightError::InvalidAccessSchedule("too many access spans".to_string())
        })?;
        let num_spans = u32::try_from(spans.len()).map_err(|_| {
            GpuPostflightError::InvalidAccessSchedule("too many spans in schedule".to_string())
        })?;
        let schedule_index = u32::try_from(self.schedules.len()).map_err(|_| {
            GpuPostflightError::InvalidAccessSchedule("too many access schedules".to_string())
        })?;
        let mut operand_words = [0u8; 3];
        operand_words[..register_operands.len()].copy_from_slice(register_operands);
        self.spans.extend_from_slice(spans);
        self.schedules.push(RvrCheckpointAccessSchedule {
            first_span,
            num_spans,
            register_operands: operand_words,
            num_register_reads: register_operands.len() as u8,
            effect,
            effect_operand,
            register_write_source,
            register_write_operand,
        });
        self.instruction_layouts
            .push(RvrCheckpointInstructionLayout {
                zero_operand_mask,
                register_as_operand,
                memory_as_operand,
            });
        self.dispatch[opcode as usize] = schedule_index;
        Ok(())
    }

    #[doc(hidden)]
    pub fn validate_no_native_collisions(
        &self,
        opcodes: PostflightOpcodeBases,
    ) -> Result<(), GpuPostflightError> {
        if let Some(opcode) = self
            .dispatch
            .iter()
            .enumerate()
            .find_map(|(opcode, &schedule)| {
                (schedule != RVR_CHECKPOINT_NO_SCHEDULE && opcodes.owns(opcode as u32))
                    .then_some(opcode)
            })
        {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "opcode {opcode} is owned by both native replay and an extension schedule"
            )));
        }
        Ok(())
    }

    fn validate_instruction(
        &self,
        instruction: &PostflightInstruction,
        cell_pointer_max_bits: usize,
        deferral_num_cells: usize,
    ) -> Result<(), GpuPostflightError> {
        let opcode = instruction.words[0] as usize;
        let Some(&schedule_index) = self.dispatch.get(opcode) else {
            return Ok(());
        };
        if schedule_index == RVR_CHECKPOINT_NO_SCHEDULE {
            return Ok(());
        }
        let schedule = self.schedules.get(schedule_index as usize).ok_or_else(|| {
            GpuPostflightError::InvalidAccessSchedule(
                "dispatch references a missing schedule".to_string(),
            )
        })?;
        let layout = self
            .instruction_layouts
            .get(schedule_index as usize)
            .ok_or_else(|| {
                GpuPostflightError::InvalidAccessSchedule(
                    "schedule is missing its host instruction layout".to_string(),
                )
            })?;
        let span_start = schedule.first_span as usize;
        let span_end = schedule
            .first_span
            .checked_add(schedule.num_spans)
            .map(|end| end as usize)
            .ok_or_else(|| {
                GpuPostflightError::InvalidAccessSchedule(
                    "schedule span range exceeds the u32 index domain".to_string(),
                )
            })?;
        let schedule_spans = self.spans.get(span_start..span_end).ok_or_else(|| {
            GpuPostflightError::InvalidAccessSchedule(
                "schedule references a missing access span".to_string(),
            )
        })?;
        let invalid_deferral_span = schedule_spans
            .iter()
            .filter(|span| {
                matches!(
                    span.base_source,
                    RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_INPUT
                        | RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_OUTPUT
                )
            })
            .any(|span| {
                let base = u64::from(instruction.words[span.base_index as usize]) * 16
                    + u64::from(span.base_source == RVR_CHECKPOINT_SPAN_BASE_DEFERRAL_OUTPUT) * 8;
                let end = base + u64::from(span.count) * BLOCK_FE_WIDTH as u64;
                let pointer_limit = 1u64 << cell_pointer_max_bits;
                end < base || end > pointer_limit || end > deferral_num_cells as u64
            });
        if instruction.words[layout.register_as_operand as usize] != RV64_REGISTER_AS
            || instruction.words[layout.memory_as_operand as usize] != RV64_MEMORY_AS
            || (1..8).any(|word| {
                layout.zero_operand_mask & (1 << word) != 0 && instruction.words[word] != 0
            })
            || schedule
                .register_operands
                .iter()
                .take(schedule.num_register_reads as usize)
                .any(|&word| {
                    let pointer = u64::from(instruction.words[word as usize]);
                    pointer >= 32 * RV64_REGISTER_BYTES || pointer % RV64_REGISTER_BYTES != 0
                })
            || (schedule.register_write_source != RVR_CHECKPOINT_REGISTER_WRITE_NONE && {
                let pointer =
                    u64::from(instruction.words[schedule.register_write_operand as usize]);
                pointer >= 32 * RV64_REGISTER_BYTES || pointer % RV64_REGISTER_BYTES != 0
            })
            || (schedule.effect == RVR_CHECKPOINT_EFFECT_BRANCH_RESIDUAL
                && instruction.words[schedule.effect_operand as usize] >= BabyBear::ORDER_U32)
            || invalid_deferral_span
        {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "opcode {} has an instruction incompatible with its access schedule",
                instruction.words[0]
            )));
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RvrMemoryAddressSpace {
    num_cells: u64,
    cell_kind: u32,
    _padding: u32,
}

const _: () = assert!(size_of::<RvrMemoryAddressSpace>() == 16);

const RVR_MEMORY_CELL_UNSUPPORTED: u32 = 0;
const RVR_MEMORY_CELL_U16: u32 = 1;
const RVR_MEMORY_CELL_FIELD32: u32 = 2;

fn memory_cell_kind(layout: MemoryCellType) -> u32 {
    match layout {
        MemoryCellType::U16 => RVR_MEMORY_CELL_U16,
        MemoryCellType::F { size: 4 } => RVR_MEMORY_CELL_FIELD32,
        _ => RVR_MEMORY_CELL_UNSUPPORTED,
    }
}

#[derive(Debug, Error)]
pub enum GpuPostflightError {
    #[error("program opcode {0} does not fit the GPU replay ABI")]
    OpcodeTooLarge(usize),
    #[error("invalid GPU postflight memory configuration: {0}")]
    InvalidMemoryConfig(String),
    #[error("invalid GPU postflight access schedule: {0}")]
    InvalidAccessSchedule(String),
    #[error("{0}")]
    InvalidTranscript(String),
    #[error("GPU postflight input belongs to another CUDA device or stream")]
    ContextMismatch,
    #[error("GPU postflight transcript belongs to another uploaded program")]
    ProgramMismatch,
    #[error("GPU postflight plan belongs to another transcript segment")]
    SegmentMismatch,
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    Copy(#[from] MemCopyError),
}

fn upload<T>(values: &[T], device_ctx: &GpuDeviceCtx) -> Result<DeviceBuffer<T>, MemCopyError> {
    if values.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        values.to_device_on(device_ctx)
    }
}

pub(crate) type ConnectorBoundary = (ExecutionState<u32>, ExecutionState<u32>, Option<u32>);

#[cfg(feature = "test-utils")]
fn replay_boundary(
    transcript: &PreflightEventLog,
    endpoint: PreflightEndpoint,
) -> Result<ConnectorBoundary, GpuPostflightError> {
    let first = transcript.program_log.first().ok_or_else(|| {
        GpuPostflightError::InvalidTranscript(
            "transcript must contain an initial event and final sentinel".to_string(),
        )
    })?;
    let last = transcript.program_log.last().unwrap();
    Ok((
        ExecutionState::new(first.pc, first.timestamp),
        ExecutionState::new(last.pc, last.timestamp),
        matches!(endpoint, PreflightEndpoint::Terminated).then_some(0),
    ))
}

/// Static program data uploaded once and shared by every replayed segment.
pub struct GpuPostflightProgram {
    instructions: DeviceBuffer<PostflightInstruction>,
    dense_program_rows: DeviceBuffer<u32>,
    num_program_rows: usize,
    active_opcodes: Vec<u32>,
    d_active_opcodes: DeviceBuffer<u32>,
    checkpoint_schedule_dispatch: DeviceBuffer<u32>,
    checkpoint_schedules: DeviceBuffer<RvrCheckpointAccessSchedule>,
    checkpoint_spans: DeviceBuffer<PostflightAccessSpan>,
    checkpoint_static_values: DeviceBuffer<u64>,
    checkpoint_schedule_opcodes: Vec<u32>,
    memory_address_spaces: DeviceBuffer<RvrMemoryAddressSpace>,
    /// Host layout for validating and uploading expanded interpreter logs.
    memory_config: MemoryConfig,
    address_space_height: u32,
    /// Pointer width used by chronology keys, whose pointers count AS-native cells.
    cell_pointer_max_bits: u32,
    /// Pointer width used by instruction replay, whose pointers count bytes.
    /// Guest addresses are u32, so wider configured domains saturate at 32 bits.
    byte_pointer_max_bits: u32,
    timestamp_max_bits: u32,
    pc_base: u32,
    device_ctx: GpuDeviceCtx,
    identity: Arc<()>,
}

impl GpuPostflightProgram {
    #[cfg(any(test, feature = "test-utils"))]
    pub fn upload<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        Self::upload_with_postflight_access_registry(
            program,
            memory_config,
            &PostflightAccessRegistry::default(),
            device_ctx,
        )
    }

    #[doc(hidden)]
    pub fn upload_with_postflight_access_registry<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        registry: &PostflightAccessRegistry,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        if F::ORDER_U32 != BabyBear::ORDER_U32 || size_of::<F>() != size_of::<BabyBear>() {
            return Err(GpuPostflightError::InvalidMemoryConfig(
                "GPU postflight currently requires the BabyBear proof field".to_string(),
            ));
        }
        if memory_config.pointer_max_bits > u32::BITS as usize
            || memory_config.addr_space_height >= u32::BITS as usize
            || memory_config.timestamp_max_bits >= u32::BITS as usize
        {
            return Err(GpuPostflightError::InvalidMemoryConfig(
                "address-space height, pointer width, and timestamp width must fit u32".to_string(),
            ));
        }
        let address_space_count = 1usize
            .checked_shl(memory_config.addr_space_height as u32)
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig("address-space count overflow".to_string())
            })?;
        let expected_address_spaces = (ADDR_SPACE_OFFSET as usize)
            .checked_add(address_space_count)
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig("address-space count overflow".to_string())
            })?;
        if memory_config.addr_spaces.len() != expected_address_spaces {
            return Err(GpuPostflightError::InvalidMemoryConfig(format!(
                "expected {expected_address_spaces} address-space layouts, found {}",
                memory_config.addr_spaces.len()
            )));
        }
        let memory_address_spaces = memory_config
            .addr_spaces
            .iter()
            .map(|config| RvrMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                cell_kind: memory_cell_kind(config.layout),
                _padding: 0,
            })
            .collect::<Vec<_>>();
        let block_pointer_bits = memory_config
            .pointer_max_bits
            .checked_sub(BLOCK_FE_WIDTH.ilog2() as usize)
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig(
                    "pointer width is smaller than one memory block".to_string(),
                )
            })?;
        let label_bits = memory_config
            .addr_space_height
            .checked_add(block_pointer_bits)
            .ok_or_else(|| {
                GpuPostflightError::InvalidMemoryConfig(
                    "address-space and block-pointer label width overflow".to_string(),
                )
            })?;
        if label_bits > u32::BITS as usize {
            return Err(GpuPostflightError::InvalidMemoryConfig(
                "address-space and block-pointer label does not fit u32".to_string(),
            ));
        }
        let instructions = program
            .instructions_and_debug_infos
            .iter()
            .map(|entry| match entry {
                Some((instruction, _)) => instruction_to_replay(instruction),
                None => Ok(PostflightInstruction {
                    words: [u32::MAX, 0, 0, 0, 0, 0, 0, 0],
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        for instruction in &instructions {
            if instruction.words[0] != u32::MAX {
                registry.validate_instruction(
                    instruction,
                    memory_config.pointer_max_bits,
                    memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells,
                )?;
            }
        }
        let opcodes: Vec<u32> = instructions
            .iter()
            .map(|instruction| instruction.words[0])
            .collect();
        let mut next_program_row = 0u32;
        let dense_program_rows = opcodes
            .iter()
            .map(|&opcode| {
                if opcode == u32::MAX {
                    u32::MAX
                } else {
                    let row = next_program_row;
                    next_program_row += 1;
                    row
                }
            })
            .collect::<Vec<_>>();
        let mut active_opcodes = opcodes
            .iter()
            .copied()
            .filter(|&opcode| opcode != u32::MAX)
            .collect::<Vec<_>>();
        active_opcodes.sort_unstable();
        active_opcodes.dedup();
        let checkpoint_schedule_opcodes = registry
            .dispatch
            .iter()
            .enumerate()
            .filter_map(|(opcode, &schedule)| {
                (schedule != RVR_CHECKPOINT_NO_SCHEDULE).then_some(opcode as u32)
            })
            .collect();
        Ok(Self {
            instructions: upload(&instructions, device_ctx)?,
            dense_program_rows: upload(&dense_program_rows, device_ctx)?,
            num_program_rows: next_program_row as usize,
            d_active_opcodes: upload(&active_opcodes, device_ctx)?,
            active_opcodes,
            checkpoint_schedule_dispatch: upload(&registry.dispatch, device_ctx)?,
            checkpoint_schedules: upload(&registry.schedules, device_ctx)?,
            checkpoint_spans: upload(&registry.spans, device_ctx)?,
            checkpoint_static_values: upload(&registry.static_values, device_ctx)?,
            checkpoint_schedule_opcodes,
            memory_address_spaces: upload(&memory_address_spaces, device_ctx)?,
            memory_config: memory_config.clone(),
            address_space_height: memory_config.addr_space_height as u32,
            cell_pointer_max_bits: memory_config.pointer_max_bits as u32,
            byte_pointer_max_bits: to_byte_ptr_bits(memory_config.pointer_max_bits)
                .min(u32::BITS as usize) as u32,
            timestamp_max_bits: memory_config.timestamp_max_bits as u32,
            pc_base: program.pc_base,
            device_ctx: device_ctx.clone(),
            identity: Arc::new(()),
        })
    }

    pub fn instructions(&self) -> DeviceBufferView {
        self.instructions.view()
    }

    pub const fn pc_base(&self) -> u32 {
        self.pc_base
    }

    /// Upload one segment's logs and its derived replay work. Deriving the work
    /// through the uploaded program prevents mixing a plan with another static
    /// program that happens to have the same length.
    #[cfg(feature = "test-utils")]
    pub fn upload_transcript(
        &self,
        transcript: &PreflightEventLog,
        endpoint: PreflightEndpoint,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let boundary = replay_boundary(transcript, endpoint)?;
        let segment_identity = Arc::new(());
        let gpu = GpuPostflightTranscript::upload(
            transcript,
            &self.memory_config,
            &self.device_ctx,
            self.identity.clone(),
            segment_identity.clone(),
        )?;
        let plan = GpuPostflightPlan::build(
            self,
            &gpu,
            endpoint,
            boundary,
            self.identity.clone(),
            segment_identity,
        )?;
        Ok((gpu, plan))
    }

    /// Uploads expanded history produced by serial interpreter preflight.
    ///
    /// The interpreter already records full block values. Chronology still
    /// runs once on the GPU to derive predecessor indexes and touched blocks
    /// in the same format as compiled preflight.
    pub(crate) fn upload_history(
        &self,
        history: &PreflightHistory,
        endpoint: PreflightEndpoint,
        boundary: ConnectorBoundary,
        initial_memory_images: &[DeviceBufferView],
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        if history.program.is_empty() {
            return Err(GpuPostflightError::InvalidTranscript(
                "preflight history must contain a final program sentinel".to_string(),
            ));
        }
        if history.memory.accesses.len() >= (1usize << 31) {
            return Err(GpuPostflightError::InvalidTranscript(
                "preflight memory log exceeds packed predecessor indexes".to_string(),
            ));
        }

        let mut write_masks = Vec::with_capacity(history.memory.accesses.len());
        let field_values = &history.memory.field_values;
        let mut field_cursor = 0usize;
        for event in &history.memory.accesses {
            let address_space = event.address_space() as usize;
            let layout = self
                .memory_config
                .addr_spaces
                .get(address_space)
                .ok_or_else(|| {
                    GpuPostflightError::InvalidTranscript(format!(
                        "memory event uses unknown address space {address_space}"
                    ))
                })?
                .layout;
            write_masks.push(if event.is_write() { u8::MAX } else { 0 });
            if layout == MemoryCellType::field32() {
                let reference =
                    usize::try_from(u32::from(event.value[0]) | (u32::from(event.value[1]) << 16))
                        .unwrap();
                if event.value[2] != 0
                    || event.value[3] != 0
                    || reference != field_cursor
                    || reference >= field_values.len()
                {
                    return Err(GpuPostflightError::InvalidTranscript(
                        "field memory events must use dense ordered sidecar references".to_string(),
                    ));
                }
                field_cursor += 1;
            }
        }
        if field_cursor != field_values.len() {
            return Err(GpuPostflightError::InvalidTranscript(
                "field sidecar contains unreferenced values".to_string(),
            ));
        }

        let program_log = upload(&history.program, &self.device_ctx)?;
        let memory_log = upload(&history.memory.accesses, &self.device_ctx)?;
        let field_values = upload(field_values, &self.device_ctx)?;
        let write_masks = upload(&write_masks, &self.device_ctx)?;
        let error = [0u32].to_device_on(&self.device_ctx)?;
        let (initial_write_log, field_initial_values, memory_index) = build_gpu_memory_chronology(
            &memory_log,
            &write_masks,
            &field_values,
            initial_memory_images,
            self.address_space_height,
            self.cell_pointer_max_bits,
            self.memory_address_spaces.view(),
            &error,
            &self.device_ctx,
        )?;
        drop(write_masks);

        let segment_identity = Arc::new(());
        let transcript = GpuPostflightTranscript {
            program_log,
            memory_log,
            initial_write_log,
            field_values,
            field_initial_values,
            memory_predecessors: memory_index.predecessors,
            touched_blocks: memory_index.touched_blocks,
            num_touched_blocks: memory_index.num_touched_blocks,
            error,
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity: segment_identity.clone(),
        };
        let plan = GpuPostflightPlan::build(
            self,
            &transcript,
            endpoint,
            boundary,
            self.identity.clone(),
            segment_identity,
        )?;
        Ok((transcript, plan))
    }

    /// Expands checkpoint execution into ordinary program and memory logs.
    /// Independent chunks emit byte-masked block intents; one device-wide
    /// chronology pass resolves them against the segment-start memory image.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn postflight(
        &self,
        execution: &PreflightExecution,
        expected_retired: u32,
        initial_registers: DeviceBufferView,
        initial_memory: DeviceBufferView,
        initial_memory_images: &[DeviceBufferView],
        opcodes: PostflightOpcodeBases,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        if execution.retired != expected_retired {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "preflight retired {} instructions, expected {expected_retired}",
                execution.retired
            )));
        }
        if let Some(opcode) = self
            .checkpoint_schedule_opcodes
            .iter()
            .copied()
            .find(|&opcode| opcodes.owns(opcode))
        {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "opcode {opcode} is owned by both native replay and an extension schedule"
            )));
        }
        let (endpoint_kind, exit_code) = match execution.endpoint {
            PreflightEndpoint::Terminated => (0, Some(0)),
            PreflightEndpoint::Suspended => (1, None),
        };
        if execution.from_state.timestamp != 1
            || execution.to_state.pc != execution.state.pc()
            || execution.to_state.timestamp >= (1u32 << self.timestamp_max_bits)
        {
            return Err(GpuPostflightError::InvalidTranscript(
                "preflight has an invalid boundary".to_string(),
            ));
        }
        let residual_cursor =
            u32::try_from(execution.transcript.residuals.len()).map_err(|_| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint residual stream exceeds the u32 cursor domain".to_string(),
                )
            })?;
        let final_registers = read_rv64_registers(&execution.state);
        let mut final_anchor = RvrCheckpoint {
            pc: execution.to_state.pc,
            timestamp: execution.to_state.timestamp,
            retired: execution.retired,
            residual_cursor,
            regs: [0; 31],
        };
        final_anchor.regs.copy_from_slice(&final_registers[1..]);
        let mut anchors = execution.transcript.checkpoints.clone();
        anchors.push(final_anchor);
        let anchors = upload(&anchors, &self.device_ctx)?;
        let residuals = upload(&execution.transcript.residuals, &self.device_ctx)?;
        let error = [0u32].to_device_on(&self.device_ctx)?;
        let event_counts = gpu_buffer::<PostflightEventCount>(anchors.len(), &self.device_ctx);
        event_counts.fill_zero_on(&self.device_ctx)?;
        let address_spaces = [RV64_REGISTER_AS, RV64_MEMORY_AS, RV64_IMM_AS, DEFERRAL_AS];
        let count_span = tracing::info_span!("postflight_replay_count").entered();
        unsafe {
            rvr_checkpoint_replay::count(
                self.instructions.view(),
                self.pc_base,
                initial_registers,
                initial_memory,
                anchors.view(),
                residuals.view(),
                self.checkpoint_schedule_dispatch.view(),
                self.checkpoint_schedules.view(),
                self.checkpoint_spans.view(),
                self.checkpoint_static_values.view(),
                opcodes,
                address_spaces,
                self.byte_pointer_max_bits,
                self.cell_pointer_max_bits,
                execution.from_state.pc,
                execution.from_state.timestamp,
                endpoint_kind,
                &event_counts,
                &error,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        let count_error = error.to_host_on(&self.device_ctx)?[0];
        if count_error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "checkpoint GPU count replay rejected execution with code {count_error}"
            )));
        }
        let counts = event_counts.to_host_on(&self.device_ctx)?;
        let mut total_memory = 0u32;
        let mut total_fields = 0u32;
        let mut offsets = Vec::with_capacity(counts.len());
        for count in counts {
            offsets.push(PostflightEventCount {
                memory: total_memory,
                field: total_fields,
            });
            total_memory = total_memory.checked_add(count.memory).ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint replay memory-event count exceeds u32".to_string(),
                )
            })?;
            total_fields = total_fields.checked_add(count.field).ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint replay field-event count exceeds u32".to_string(),
                )
            })?;
        }
        if total_memory >= (1u32 << 31) {
            return Err(GpuPostflightError::InvalidTranscript(
                "checkpoint replay memory log exceeds packed predecessor indexes".to_string(),
            ));
        }
        drop(count_span);
        let program_len = usize::try_from(execution.retired)
            .ok()
            .and_then(|retired| retired.checked_add(1))
            .ok_or_else(|| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint replay program-log length overflow".to_string(),
                )
            })?;
        let program_log = gpu_buffer::<PreflightProgramEvent>(program_len, &self.device_ctx);
        let memory_log =
            gpu_buffer::<PreflightMemoryEvent>(total_memory as usize, &self.device_ctx);
        let field_values =
            gpu_buffer::<PostflightFieldBlock>(total_fields as usize, &self.device_ctx);
        // One transient byte per event is enough to distinguish reads, full
        // writes, and partial block writes. The chronology pass consumes and
        // releases this before opcode trace generation.
        let write_masks = gpu_buffer::<u8>(total_memory as usize, &self.device_ctx);
        let offsets = upload(&offsets, &self.device_ctx)?;
        let emit_span = tracing::info_span!("postflight_replay_emit").entered();
        unsafe {
            rvr_checkpoint_replay::emit(
                self.instructions.view(),
                self.pc_base,
                initial_registers,
                initial_memory,
                anchors.view(),
                residuals.view(),
                offsets.view(),
                self.checkpoint_schedule_dispatch.view(),
                self.checkpoint_schedules.view(),
                self.checkpoint_spans.view(),
                self.checkpoint_static_values.view(),
                opcodes,
                address_spaces,
                self.byte_pointer_max_bits,
                self.cell_pointer_max_bits,
                execution.from_state.pc,
                execution.from_state.timestamp,
                endpoint_kind,
                program_log.view(),
                memory_log.view(),
                write_masks.view(),
                field_values.view(),
                &error,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        let emit_error = error.to_host_on(&self.device_ctx)?[0];
        if emit_error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "checkpoint GPU emit replay rejected execution with code {emit_error}"
            )));
        }
        drop(emit_span);
        // The host reads above synchronize the emit stream. Release compact
        // replay inputs before chronology allocates its sort/scan scratch.
        drop(anchors);
        drop(residuals);
        drop(event_counts);
        drop(offsets);
        let (initial_write_log, field_initial_values, memory_index) =
            tracing::info_span!("postflight_memory_chronology").in_scope(|| {
                build_gpu_memory_chronology(
                    &memory_log,
                    &write_masks,
                    &field_values,
                    initial_memory_images,
                    self.address_space_height,
                    self.cell_pointer_max_bits,
                    self.memory_address_spaces.view(),
                    &error,
                    &self.device_ctx,
                )
            })?;
        drop(write_masks);

        let segment_identity = Arc::new(());
        let transcript = GpuPostflightTranscript {
            program_log,
            memory_log,
            initial_write_log,
            field_values,
            field_initial_values,
            memory_predecessors: memory_index.predecessors,
            touched_blocks: memory_index.touched_blocks,
            num_touched_blocks: memory_index.num_touched_blocks,
            error,
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity: segment_identity.clone(),
        };
        let boundary = (execution.from_state, execution.to_state, exit_code);
        let plan = tracing::info_span!("postflight_program_index").in_scope(|| {
            GpuPostflightPlan::build(
                self,
                &transcript,
                execution.endpoint,
                boundary,
                self.identity.clone(),
                segment_identity,
            )
        })?;
        Ok((transcript, plan))
    }

    pub fn ensure_replay_inputs(
        &self,
        transcript: &GpuPostflightTranscript,
        plan: &GpuPostflightPlan,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<(), GpuPostflightError> {
        ensure_same_context(&self.device_ctx, device_ctx)?;
        ensure_same_context(&transcript.device_ctx, device_ctx)?;
        ensure_same_context(&plan.device_ctx, device_ctx)?;
        if !Arc::ptr_eq(&self.identity, &transcript.program_identity)
            || !Arc::ptr_eq(&self.identity, &plan.program_identity)
        {
            return Err(GpuPostflightError::ProgramMismatch);
        }
        if !Arc::ptr_eq(&transcript.segment_identity, &plan.segment_identity) {
            return Err(GpuPostflightError::SegmentMismatch);
        }
        Ok(())
    }
}

fn ensure_same_context(
    expected: &GpuDeviceCtx,
    actual: &GpuDeviceCtx,
) -> Result<(), GpuPostflightError> {
    if expected.device_id == actual.device_id && expected.stream == actual.stream {
        Ok(())
    } else {
        Err(GpuPostflightError::ContextMismatch)
    }
}

fn instruction_to_replay<F: PrimeField32>(
    instruction: &Instruction<F>,
) -> Result<PostflightInstruction, GpuPostflightError> {
    let opcode = u32::try_from(instruction.opcode.as_usize())
        .map_err(|_| GpuPostflightError::OpcodeTooLarge(instruction.opcode.as_usize()))?;
    Ok(PostflightInstruction {
        words: [
            opcode,
            instruction.a.as_canonical_u32(),
            instruction.b.as_canonical_u32(),
            instruction.c.as_canonical_u32(),
            instruction.d.as_canonical_u32(),
            instruction.e.as_canonical_u32(),
            instruction.f.as_canonical_u32(),
            instruction.g.as_canonical_u32(),
        ],
    })
}

fn gpu_buffer<T>(len: usize, device_ctx: &GpuDeviceCtx) -> DeviceBuffer<T> {
    if len == 0 {
        DeviceBuffer::new()
    } else {
        DeviceBuffer::with_capacity_on(len, device_ctx)
    }
}

struct GpuMemoryIndex {
    predecessors: DeviceBuffer<u32>,
    touched_blocks: DeviceBuffer<TouchedBlock<BabyBear>>,
    num_touched_blocks: usize,
}

#[allow(clippy::too_many_arguments)]
fn build_gpu_memory_chronology(
    memory: &DeviceBuffer<PreflightMemoryEvent>,
    write_masks: &DeviceBuffer<u8>,
    field_values: &DeviceBuffer<PostflightFieldBlock>,
    initial_memory: &[DeviceBufferView],
    address_space_height: u32,
    pointer_max_bits: u32,
    address_spaces: DeviceBufferView,
    error: &DeviceBuffer<u32>,
    device_ctx: &GpuDeviceCtx,
) -> Result<
    (
        DeviceBuffer<PreflightInitialWrite>,
        DeviceBuffer<PostflightFieldBlock>,
        GpuMemoryIndex,
    ),
    GpuPostflightError,
> {
    // Trusted count/emit expansion assigns the k-th FIELD32 event in memory-log
    // order reference k and allocates exactly one sidecar entry per such event.
    // That dense unique mapping is the race-freedom invariant for in-place GPU
    // resolution; chronology range-checks references but deliberately does not
    // allocate a claimed bitmap or perform another full-log scan to re-prove it.
    if memory.len() != write_masks.len() {
        return Err(GpuPostflightError::InvalidTranscript(
            "checkpoint memory intent and mask lengths differ".to_string(),
        ));
    }
    if memory.is_empty() {
        if !field_values.is_empty() {
            return Err(GpuPostflightError::InvalidTranscript(
                "field values exist without memory events".to_string(),
            ));
        }
        return Ok((
            DeviceBuffer::new(),
            DeviceBuffer::new(),
            GpuMemoryIndex {
                predecessors: DeviceBuffer::new(),
                touched_blocks: DeviceBuffer::new(),
                num_touched_blocks: 0,
            },
        ));
    }

    let num_entries = memory.len();
    // Chronology count/sort uses one u64 per event. Value resolution reuses
    // the same allocation as one 16-byte scan element per event.
    let workspace = gpu_buffer::<u64>(2 * num_entries, device_ctx);
    let sorted_keys = gpu_buffer::<u64>(num_entries, device_ctx);
    let predecessors = gpu_buffer::<u32>(num_entries, device_ctx);
    let initial_memory = upload(initial_memory, device_ctx)?;
    // Keep the U16-only allocation exactly as before. Field metadata extends
    // this tiny counter buffer only when a field sidecar actually exists.
    let count_len = if field_values.is_empty() { 2 } else { 6 };
    let device_counts = upload(&vec![0u32; count_len], device_ctx)?;
    let mut temp_bytes = 0usize;
    unsafe {
        rvr_postflight::memory_chronology_get_temp_bytes(
            num_entries,
            &mut temp_bytes,
            device_ctx.stream.as_raw(),
        )?;
    }
    let temp_storage = gpu_buffer::<u8>(temp_bytes, device_ctx);
    unsafe {
        rvr_postflight::memory_chronology_sort_and_count(
            memory.view(),
            write_masks.view(),
            field_values.view(),
            address_spaces,
            ADDR_SPACE_OFFSET,
            address_space_height,
            pointer_max_bits,
            DEFERRAL_AS,
            !field_values.is_empty(),
            &workspace,
            &sorted_keys,
            &device_counts,
            &temp_storage,
            temp_bytes,
            error,
            device_ctx.stream.as_raw(),
        )?;
    }
    let counts = device_counts.to_host_on(device_ctx)?;
    drop(device_counts);
    let (num_seeds, num_touched) = (counts[0], counts[1]);
    let (field_begin, field_end, field_seed_base, num_field_seeds) = if let [_, _, field_begin, field_end, field_seed_base, num_field_seeds] =
        counts.as_slice()
    {
        (*field_begin, *field_end, *field_seed_base, *num_field_seeds)
    } else {
        (0, 0, 0, 0)
    };
    let sort_error = error.to_host_on(device_ctx)?[0];
    if sort_error != 0 {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "checkpoint GPU memory chronology rejected intents with code {sort_error}"
        )));
    }
    let num_seeds = num_seeds as usize;
    let num_touched = num_touched as usize;
    let num_field_seeds = num_field_seeds as usize;
    if num_seeds > num_entries
        || num_touched > num_entries
        || field_end < field_begin
        || field_end as usize > num_entries
        || (field_end - field_begin) as usize != field_values.len()
        || field_seed_base as usize > num_seeds
        || num_field_seeds > num_seeds - field_seed_base as usize
    {
        return Err(GpuPostflightError::InvalidTranscript(
            "checkpoint GPU memory chronology produced invalid counts".to_string(),
        ));
    }
    let seeds = gpu_buffer::<PreflightInitialWrite>(num_seeds, device_ctx);
    let field_seeds = gpu_buffer::<PostflightFieldBlock>(num_field_seeds, device_ctx);
    let touched_blocks = gpu_buffer::<TouchedBlock<BabyBear>>(num_touched, device_ctx);
    unsafe {
        rvr_postflight::memory_chronology_resolve(
            memory.view(),
            write_masks.view(),
            address_spaces,
            initial_memory.view(),
            field_values.view(),
            RV64_REGISTER_AS,
            &sorted_keys,
            &workspace,
            &predecessors,
            seeds.view(),
            field_seeds.view(),
            field_begin,
            field_end,
            field_seed_base,
            touched_blocks.view(),
            &temp_storage,
            temp_bytes,
            error,
            device_ctx.stream.as_raw(),
        )?;
    }
    let resolve_error = error.to_host_on(device_ctx)?[0];
    if resolve_error != 0 {
        return Err(GpuPostflightError::InvalidTranscript(format!(
            "checkpoint GPU memory chronology failed with code {resolve_error}"
        )));
    }
    // The error copy fences chronology. Release every sort/scan-only
    // allocation before trace generation receives the retained logs and
    // predecessor index.
    drop((workspace, sorted_keys, initial_memory, temp_storage));
    Ok((
        seeds,
        field_seeds,
        GpuMemoryIndex {
            predecessors,
            touched_blocks,
            num_touched_blocks: num_touched,
        },
    ))
}

/// Device-resident transcript and its one generic predecessor index.
///
/// This object is shared across all opcode kernels for the segment. `error` is
/// also shared so replay validation can fail closed without a per-chip copy.
/// Read it once after all replay kernels. A nonzero result is terminal for that
/// proving attempt because threads from other rows may already have updated
/// shared lookup histograms.
pub struct GpuPostflightTranscript {
    program_log: DeviceBuffer<PreflightProgramEvent>,
    memory_log: DeviceBuffer<PreflightMemoryEvent>,
    initial_write_log: DeviceBuffer<PreflightInitialWrite>,
    field_values: DeviceBuffer<PostflightFieldBlock>,
    field_initial_values: DeviceBuffer<PostflightFieldBlock>,
    memory_predecessors: DeviceBuffer<u32>,
    touched_blocks: DeviceBuffer<TouchedBlock<BabyBear>>,
    num_touched_blocks: usize,
    error: DeviceBuffer<u32>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
}

impl GpuPostflightTranscript {
    #[cfg(feature = "test-utils")]
    fn upload(
        transcript: &PreflightEventLog,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
        program_identity: Arc<()>,
        segment_identity: Arc<()>,
    ) -> Result<Self, GpuPostflightError> {
        let memory_index = super::postflight::build_memory_index(
            &transcript.memory_log,
            &transcript.initial_write_log,
            memory_config,
        )
        .map_err(|error| GpuPostflightError::InvalidTranscript(error.to_string()))?;
        let program_log = upload(&transcript.program_log, device_ctx)?;
        let memory_log = upload(&transcript.memory_log, device_ctx)?;
        let initial_write_log = upload(&transcript.initial_write_log, device_ctx)?;
        let error = [0u32].to_device_on(device_ctx)?;
        let memory_predecessors = upload(&memory_index.predecessors, device_ctx)?;
        let touched_blocks = upload(&memory_index.touched_blocks, device_ctx)?;
        let num_touched_blocks = touched_blocks.len();
        Ok(Self {
            program_log,
            memory_log,
            initial_write_log,
            field_values: DeviceBuffer::new(),
            field_initial_values: DeviceBuffer::new(),
            memory_predecessors,
            touched_blocks,
            num_touched_blocks,
            error,
            device_ctx: device_ctx.clone(),
            program_identity,
            segment_identity,
        })
    }

    pub fn error_code(&self) -> Result<u32, MemCopyError> {
        Ok(self.error.to_host_on(&self.device_ctx)?[0])
    }

    /// Waits for every replay kernel submitted on this transcript's stream.
    ///
    /// Safe orchestration code must call this before returning ownership to a
    /// caller that may release the transcript or replay plan. This fence is
    /// separate from [`Self::error_code`] so a failed D2H enqueue cannot skip
    /// synchronization of earlier kernels.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.device_ctx.stream.synchronize()
    }

    pub fn program_log(&self) -> DeviceBufferView {
        self.program_log.view()
    }

    pub fn memory_log(&self) -> DeviceBufferView {
        self.memory_log.view()
    }

    pub fn initial_write_log(&self) -> DeviceBufferView {
        self.initial_write_log.view()
    }

    /// Full-width raw-Montgomery values for field-cell memory events.
    pub fn field_values(&self) -> DeviceBufferView {
        self.field_values.view()
    }

    /// Full-width raw-Montgomery values for first-write field seeds.
    pub fn field_initial_values(&self) -> DeviceBufferView {
        self.field_initial_values.view()
    }

    pub fn memory_predecessors(&self) -> DeviceBufferView {
        self.memory_predecessors.view()
    }

    /// Sorted unique final state of every block touched by a timed memory
    /// event. The view is the initialized prefix of the retained capacity.
    pub fn touched_blocks(&self) -> DeviceBufferView {
        DeviceBufferView {
            ptr: self.touched_blocks.as_raw_ptr(),
            size: self.num_touched_blocks * size_of::<TouchedBlock<BabyBear>>(),
        }
    }

    pub const fn num_touched_blocks(&self) -> usize {
        self.num_touched_blocks
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn program_log_host(&self) -> Result<Vec<PreflightProgramEvent>, MemCopyError> {
        self.program_log.to_host_on(&self.device_ctx)
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn replace_program_log_for_test(
        &mut self,
        program_log: &[PreflightProgramEvent],
    ) -> Result<(), MemCopyError> {
        assert_eq!(
            program_log.len(),
            self.program_log.len(),
            "a replay plan is valid only for its original program-log length"
        );
        self.program_log = program_log.to_device_on(&self.device_ctx)?;
        Ok(())
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn memory_log_host(&self) -> Result<Vec<PreflightMemoryEvent>, MemCopyError> {
        self.memory_log.to_host_on(&self.device_ctx)
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn initial_write_log_host(&self) -> Result<Vec<PreflightInitialWrite>, MemCopyError> {
        self.initial_write_log.to_host_on(&self.device_ctx)
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn field_values_host(&self) -> Result<Vec<[u32; BLOCK_FE_WIDTH]>, MemCopyError> {
        Ok(self
            .field_values
            .to_host_on(&self.device_ctx)?
            .into_iter()
            .map(|block| block.values)
            .collect())
    }

    pub fn error_ptr(&self) -> *mut u32 {
        self.error.as_mut_ptr()
    }
}

/// The opcode-partitioned replay work list, uploaded once per segment.
pub struct GpuPostflightPlan {
    steps: DeviceBuffer<RvrReplayStep>,
    program_frequencies: DeviceBuffer<u32>,
    opcode_ranges: std::collections::BTreeMap<u32, std::ops::Range<usize>>,
    from_state: ExecutionState<u32>,
    to_state: ExecutionState<u32>,
    exit_code: Option<u32>,
    device_ctx: GpuDeviceCtx,
    program_identity: Arc<()>,
    segment_identity: Arc<()>,
}

impl GpuPostflightPlan {
    fn build(
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        endpoint: PreflightEndpoint,
        boundary: ConnectorBoundary,
        program_identity: Arc<()>,
        segment_identity: Arc<()>,
    ) -> Result<Self, GpuPostflightError> {
        let num_program_events = transcript.program_log.len();
        if num_program_events == 0 {
            return Err(GpuPostflightError::InvalidTranscript(
                "transcript must contain a final sentinel".to_string(),
            ));
        }
        let num_steps = num_program_events - 1;
        if num_steps >= u32::MAX as usize {
            return Err(GpuPostflightError::InvalidTranscript(
                "program log has more than u32::MAX entries".to_string(),
            ));
        }
        let opcode_keys_in = gpu_buffer::<u32>(num_steps, &program.device_ctx);
        let opcode_keys_out = gpu_buffer::<u32>(num_steps, &program.device_ctx);
        let steps_in = gpu_buffer::<RvrReplayStep>(num_steps, &program.device_ctx);
        let steps_out = gpu_buffer::<RvrReplayStep>(num_steps, &program.device_ctx);
        let ranges = gpu_buffer::<u32>(2 * program.active_opcodes.len(), &program.device_ctx);
        let program_frequencies = gpu_buffer::<u32>(program.num_program_rows, &program.device_ctx);
        program_frequencies.fill_zero_on(&program.device_ctx)?;
        let mut temp_bytes = 0usize;
        unsafe {
            rvr_postflight::program_index_get_temp_bytes(
                num_steps,
                &mut temp_bytes,
                program.device_ctx.stream.as_raw(),
            )?;
        }
        let temp_storage = gpu_buffer::<u8>(temp_bytes, &program.device_ctx);
        let (endpoint_kind, resume_pc, final_timestamp) = match endpoint {
            PreflightEndpoint::Terminated => (0, 0, 0),
            PreflightEndpoint::Suspended => (1, boundary.1.pc, boundary.1.timestamp),
        };
        unsafe {
            rvr_postflight::program_index(
                program.instructions.view(),
                program.dense_program_rows.view(),
                program.pc_base,
                transcript.program_log.view(),
                transcript.memory_log.view(),
                program.d_active_opcodes.view(),
                program.timestamp_max_bits,
                endpoint_kind,
                resume_pc,
                final_timestamp,
                SystemOpcode::TERMINATE.global_opcode().as_usize() as u32,
                &opcode_keys_in,
                &opcode_keys_out,
                steps_in.as_mut_raw_ptr(),
                steps_out.as_mut_raw_ptr(),
                &ranges,
                &program_frequencies,
                &temp_storage,
                temp_bytes,
                &transcript.error,
                program.device_ctx.stream.as_raw(),
            )?;
        }
        let ranges = ranges.to_host_on(&program.device_ctx)?;
        let error = transcript.error.to_host_on(&program.device_ctx)?[0];
        if error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "GPU postflight rejected transcript with code {error}"
            )));
        }
        let mut opcode_ranges = std::collections::BTreeMap::new();
        let mut covered = 0usize;
        for (&opcode, range) in program.active_opcodes.iter().zip(ranges.chunks_exact(2)) {
            let start = range[0] as usize;
            let end = range[1] as usize;
            if start != covered || start > end || end > num_steps {
                return Err(GpuPostflightError::InvalidTranscript(
                    "GPU opcode ranges do not form a complete partition".to_string(),
                ));
            }
            if start != end {
                opcode_ranges.insert(opcode, start..end);
            }
            covered = end;
        }
        if covered != num_steps {
            return Err(GpuPostflightError::InvalidTranscript(
                "GPU opcode ranges do not cover every execution step".to_string(),
            ));
        }
        Ok(Self {
            steps: steps_out,
            program_frequencies,
            opcode_ranges,
            from_state: boundary.0,
            to_state: boundary.1,
            exit_code: boundary.2,
            device_ctx: program.device_ctx.clone(),
            program_identity,
            segment_identity,
        })
    }

    pub fn steps(&self) -> DeviceBufferView {
        self.steps.view()
    }

    /// Dense execution frequencies in cached-program row order. Static program
    /// gaps are omitted and unexecuted defined instructions remain zero.
    pub fn program_frequencies(&self) -> DeviceBufferView {
        self.program_frequencies.view()
    }

    /// Connector inputs derived from the same host events uploaded into this
    /// validated replay plan. This metadata is cold and adds nothing to the
    /// preflight hot-path logs.
    pub(crate) const fn connector_boundary(&self) -> ConnectorBoundary {
        (self.from_state, self.to_state, self.exit_code)
    }

    pub fn opcode_range(&self, opcode: VmOpcode) -> std::ops::Range<usize> {
        u32::try_from(opcode.as_usize())
            .ok()
            .and_then(|opcode| self.opcode_ranges.get(&opcode).cloned())
            .unwrap_or(0..0)
    }

    /// Global opcodes that were actually executed in this segment.
    ///
    /// The iterator is sorted and contains no duplicates because it follows
    /// the replay plan's opcode partition. Tracegen coordinators use this
    /// before launching any opcode kernel to reject unported instructions.
    pub fn executed_opcodes(&self) -> impl Iterator<Item = u32> + '_ {
        self.opcode_ranges.keys().copied()
    }

    #[cfg(feature = "test-utils")]
    #[doc(hidden)]
    pub fn steps_host(&self) -> Result<Vec<[u32; 2]>, MemCopyError> {
        Ok(self
            .steps
            .to_host_on(&self.device_ctx)?
            .into_iter()
            .map(|step| [step.program_index, step.memory_start])
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        stream::GpuDeviceCtx,
    };
    use openvm_instructions::riscv::RV64_MEMORY_AS;
    use p3_baby_bear::BabyBear;
    use rvr_state::{
        PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
    };

    use super::{super::postflight::MEMORY_PREDECESSOR_SEED_BIT, *};

    fn event_value(
        timestamp: u32,
        address_space: u32,
        pointer: u32,
        is_write: bool,
        value: [u16; 4],
    ) -> PreflightMemoryEvent {
        PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: address_space | if is_write { PREFLIGHT_WRITE_BIT } else { 0 },
            pointer,
            value,
        }
    }

    fn field_event(
        timestamp: u32,
        pointer: u32,
        is_write: bool,
        value_index: u32,
    ) -> PreflightMemoryEvent {
        PreflightMemoryEvent {
            timestamp,
            address_space_and_kind: DEFERRAL_AS | if is_write { PREFLIGHT_WRITE_BIT } else { 0 },
            pointer,
            value: [value_index as u16, (value_index >> 16) as u16, 0, 0],
        }
    }

    fn raw_baby_bear(value: BabyBear) -> u32 {
        // BabyBear and the CUDA `Fp` ABI are both one raw Montgomery u32.
        unsafe { std::mem::transmute(value) }
    }

    #[allow(clippy::type_complexity)]
    fn gpu_chronology_with_fields(
        memory: &[PreflightMemoryEvent],
        write_masks: &[u8],
        field_values: &[PostflightFieldBlock],
        initial_memory: &[Vec<u8>],
        config: &MemoryConfig,
    ) -> Result<
        (
            Vec<PreflightMemoryEvent>,
            Vec<PreflightInitialWrite>,
            Vec<PostflightFieldBlock>,
            Vec<PostflightFieldBlock>,
            Vec<u32>,
            Vec<TouchedBlock<BabyBear>>,
        ),
        GpuPostflightError,
    > {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let memory = upload(memory, &device_ctx).unwrap();
        let write_masks = upload(write_masks, &device_ctx).unwrap();
        let field_values = upload(field_values, &device_ctx).unwrap();
        let initial_memory = initial_memory
            .iter()
            .map(|image| upload(image, &device_ctx).unwrap())
            .collect::<Vec<_>>();
        let initial_memory_views = initial_memory
            .iter()
            .map(|image| image.view())
            .collect::<Vec<_>>();
        let address_spaces = config
            .addr_spaces
            .iter()
            .map(|config| RvrMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                cell_kind: memory_cell_kind(config.layout),
                _padding: 0,
            })
            .collect::<Vec<_>>();
        let address_spaces = upload(&address_spaces, &device_ctx).unwrap();
        let error = [0u32].to_device_on(&device_ctx).unwrap();
        let (seeds, field_seeds, index) = build_gpu_memory_chronology(
            &memory,
            &write_masks,
            &field_values,
            &initial_memory_views,
            config.addr_space_height as u32,
            config.pointer_max_bits as u32,
            address_spaces.view(),
            &error,
            &device_ctx,
        )?;
        let mut touched = index.touched_blocks.to_host_on(&device_ctx).unwrap();
        touched.truncate(index.num_touched_blocks);
        Ok((
            memory.to_host_on(&device_ctx).unwrap(),
            seeds.to_host_on(&device_ctx).unwrap(),
            field_values.to_host_on(&device_ctx).unwrap(),
            field_seeds.to_host_on(&device_ctx).unwrap(),
            index.predecessors.to_host_on(&device_ctx).unwrap(),
            touched,
        ))
    }

    fn gpu_program(opcodes: &[u32], device_ctx: &GpuDeviceCtx) -> GpuPostflightProgram {
        let instructions = opcodes
            .iter()
            .map(|&opcode| PostflightInstruction {
                words: [opcode, 0, 0, 0, 0, 0, 0, 0],
            })
            .collect::<Vec<_>>();
        let mut active_opcodes = opcodes.to_vec();
        active_opcodes.retain(|&opcode| opcode != u32::MAX);
        active_opcodes.sort_unstable();
        active_opcodes.dedup();
        let mut next_program_row = 0u32;
        let dense_program_rows = opcodes
            .iter()
            .map(|&opcode| {
                if opcode == u32::MAX {
                    u32::MAX
                } else {
                    let row = next_program_row;
                    next_program_row += 1;
                    row
                }
            })
            .collect::<Vec<_>>();
        let config = MemoryConfig::default();
        let memory_address_spaces = config
            .addr_spaces
            .iter()
            .map(|config| RvrMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                cell_kind: memory_cell_kind(config.layout),
                _padding: 0,
            })
            .collect::<Vec<_>>();
        GpuPostflightProgram {
            instructions: upload(&instructions, device_ctx).unwrap(),
            dense_program_rows: upload(&dense_program_rows, device_ctx).unwrap(),
            num_program_rows: next_program_row as usize,
            d_active_opcodes: upload(&active_opcodes, device_ctx).unwrap(),
            active_opcodes,
            checkpoint_schedule_dispatch: DeviceBuffer::new(),
            checkpoint_schedules: DeviceBuffer::new(),
            checkpoint_spans: DeviceBuffer::new(),
            checkpoint_static_values: DeviceBuffer::new(),
            checkpoint_schedule_opcodes: Vec::new(),
            memory_address_spaces: upload(&memory_address_spaces, device_ctx).unwrap(),
            memory_config: config.clone(),
            address_space_height: config.addr_space_height as u32,
            cell_pointer_max_bits: config.pointer_max_bits as u32,
            byte_pointer_max_bits: to_byte_ptr_bits(config.pointer_max_bits).min(u32::BITS as usize)
                as u32,
            timestamp_max_bits: config.timestamp_max_bits as u32,
            pc_base: 0,
            device_ctx: device_ctx.clone(),
            identity: Arc::new(()),
        }
    }

    fn gpu_plan(
        program: &GpuPostflightProgram,
        transcript: &PreflightEventLog,
        endpoint: PreflightEndpoint,
    ) -> Result<GpuPostflightPlan, GpuPostflightError> {
        let boundary = replay_boundary(transcript, endpoint)?;
        let segment_identity = Arc::new(());
        let gpu_transcript = GpuPostflightTranscript::upload(
            transcript,
            &program.memory_config,
            &program.device_ctx,
            program.identity.clone(),
            segment_identity.clone(),
        )?;
        GpuPostflightPlan::build(
            program,
            &gpu_transcript,
            endpoint,
            boundary,
            program.identity.clone(),
            segment_identity,
        )
    }

    #[test]
    fn interpreter_history_uses_the_standard_gpu_indexes() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let opcode = 17;
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let program = gpu_program(&[opcode, terminate], &device_ctx);
        let first = [1u16, 2, 3, 4];
        let memory_read = [21u16, 22, 23, 24];
        let initial_second = [5u16, 6, 7, 8];
        let written_second = [9u16, 10, 11, 12];
        let history = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 4,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 4,
                },
            ],
            memory: crate::arch::PreflightMemoryLog {
                accesses: vec![
                    event_value(1, RV64_REGISTER_AS, 0, false, first),
                    event_value(2, RV64_MEMORY_AS, 0, false, memory_read),
                    event_value(
                        3,
                        RV64_REGISTER_AS,
                        BLOCK_FE_WIDTH as u32,
                        true,
                        written_second,
                    ),
                ],
                initial_writes: vec![PreflightInitialWrite {
                    address_space: RV64_REGISTER_AS,
                    pointer: BLOCK_FE_WIDTH as u32,
                    initial_value: initial_second,
                }],
                field_values: vec![],
                field_initial_values: vec![],
            },
        };
        let mut initial_registers = Vec::new();
        for value in first.into_iter().chain(initial_second) {
            initial_registers.extend_from_slice(&value.to_le_bytes());
        }
        let initial_memory_values = memory_read
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let initial_memory = (0..program.memory_config.addr_spaces.len())
            .map(|address_space| {
                let image = if address_space == RV64_REGISTER_AS as usize {
                    initial_registers.as_slice()
                } else if address_space == RV64_MEMORY_AS as usize {
                    initial_memory_values.as_slice()
                } else {
                    &[]
                };
                upload(image, &device_ctx).unwrap()
            })
            .collect::<Vec<_>>();
        let initial_memory_views = initial_memory
            .iter()
            .map(|image| image.view())
            .collect::<Vec<_>>();

        let (transcript, plan) = program
            .upload_history(
                &history,
                PreflightEndpoint::Terminated,
                (
                    ExecutionState::new(0u32, 1u32),
                    ExecutionState::new(4u32, 4u32),
                    Some(0),
                ),
                &initial_memory_views,
            )
            .unwrap();

        assert_eq!(
            transcript
                .initial_write_log
                .to_host_on(&device_ctx)
                .unwrap(),
            history.memory.initial_writes
        );
        assert_eq!(
            transcript
                .memory_predecessors
                .to_host_on(&device_ctx)
                .unwrap(),
            vec![0, 0, MEMORY_PREDECESSOR_SEED_BIT]
        );
        assert_eq!(
            plan.opcode_range(VmOpcode::from_usize(opcode as usize))
                .len(),
            1
        );
        assert_eq!(
            plan.opcode_range(SystemOpcode::TERMINATE.global_opcode())
                .len(),
            1
        );
    }

    fn mixed_chronology_fixture() -> (MemoryConfig, Vec<Vec<u8>>) {
        let mut config = MemoryConfig::default();
        for address_space in &mut config.addr_spaces {
            address_space.num_cells = 0;
        }
        config.addr_spaces[RV64_MEMORY_AS as usize].num_cells = 8;
        config.addr_spaces[DEFERRAL_AS as usize].num_cells = 8;
        let mut images = config
            .addr_spaces
            .iter()
            .map(|space| {
                let cell_bytes = match space.layout {
                    MemoryCellType::Null | MemoryCellType::U8 => 1,
                    MemoryCellType::U16 => 2,
                    MemoryCellType::U32 | MemoryCellType::F { size: 4 } => 4,
                    MemoryCellType::F { size } => size as usize,
                };
                vec![0u8; space.num_cells * cell_bytes]
            })
            .collect::<Vec<_>>();
        for (index, value) in [1u16, 2, 3, 4].into_iter().enumerate() {
            images[RV64_MEMORY_AS as usize][2 * index..2 * index + 2]
                .copy_from_slice(&value.to_le_bytes());
        }
        for (index, value) in [11u32, 12, 13, 14, 21, 22, 23, 24].into_iter().enumerate() {
            images[DEFERRAL_AS as usize][4 * index..4 * index + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
        (config, images)
    }

    #[test]
    fn gpu_chronology_resolves_mixed_u16_and_field_blocks_with_one_predecessor_order() {
        let (config, initial_memory) = mixed_chronology_fixture();
        let memory = [
            field_event(1, 0, false, 0),
            event_value(2, RV64_MEMORY_AS, 0, true, [0x00aa, 0, 0, 0]),
            field_event(3, 0, true, 1),
            field_event(4, 0, false, 2),
            event_value(5, RV64_MEMORY_AS, 0, false, [0; 4]),
            field_event(6, 4, true, 3),
            field_event(7, 4, false, 4),
        ];
        let first_write = PostflightFieldBlock {
            values: [31, 32, 33, 34],
        };
        let second_write = PostflightFieldBlock {
            values: [41, 42, 43, 44],
        };
        let field_values = [
            PostflightFieldBlock::default(),
            first_write,
            PostflightFieldBlock::default(),
            second_write,
            PostflightFieldBlock::default(),
        ];
        let (resolved, seeds, resolved_fields, field_seeds, predecessors, touched) =
            gpu_chronology_with_fields(
                &memory,
                &[0, 0x01, 0xff, 0, 0, 0xff, 0],
                &field_values,
                &initial_memory,
                &config,
            )
            .unwrap();

        assert_eq!(predecessors, [0, 1 << 31, 1, 3, 2, (1 << 31) | 1, 6]);
        assert_eq!(resolved[1].value, [0x00aa, 2, 3, 4]);
        assert_eq!(resolved[4].value, [0x00aa, 2, 3, 4]);
        assert_eq!(resolved_fields[0].values, [11, 12, 13, 14]);
        assert_eq!(resolved_fields[1], first_write);
        assert_eq!(resolved_fields[2], first_write);
        assert_eq!(resolved_fields[3], second_write);
        assert_eq!(resolved_fields[4], second_write);

        assert_eq!(seeds.len(), 2);
        assert_eq!(seeds[0].address_space, RV64_MEMORY_AS);
        assert_eq!(seeds[0].initial_value, [1, 2, 3, 4]);
        assert_eq!(seeds[1].address_space, DEFERRAL_AS);
        assert_eq!(seeds[1].initial_value, [0, 0, 0, 0]);
        assert_eq!(
            field_seeds,
            [PostflightFieldBlock {
                values: [21, 22, 23, 24]
            }]
        );

        assert_eq!(
            touched
                .iter()
                .map(|block| (block.address_space, block.ptr, block.timestamp))
                .collect::<Vec<_>>(),
            [
                (RV64_MEMORY_AS, 0, 5),
                (DEFERRAL_AS, 0, 4),
                (DEFERRAL_AS, 4, 7),
            ]
        );
        assert_eq!(
            touched[0].values.map(|value| value.as_canonical_u32()),
            [0x00aa, 2, 3, 4]
        );
        assert_eq!(touched[1].values.map(raw_baby_bear), first_write.values);
        assert_eq!(touched[2].values.map(raw_baby_bear), second_write.values);
        assert_eq!(
            touched
                .iter()
                .map(|block| block.is_dirty)
                .collect::<Vec<_>>(),
            [1, 1, 1]
        );
    }

    #[test]
    fn gpu_chronology_keeps_narrow_u16_only_path() {
        let mut config = MemoryConfig {
            addr_space_height: 1,
            ..Default::default()
        };
        config.addr_spaces.truncate(3);
        config.addr_spaces[RV64_MEMORY_AS as usize].num_cells = 4;
        let mut initial_memory = vec![Vec::new(), Vec::new(), vec![0u8; 8]];
        for (index, value) in [1u16, 2, 3, 4].into_iter().enumerate() {
            initial_memory[RV64_MEMORY_AS as usize][2 * index..2 * index + 2]
                .copy_from_slice(&value.to_le_bytes());
        }
        let read = event_value(1, RV64_MEMORY_AS, 0, false, [0; 4]);
        let (resolved, seeds, field_values, field_seeds, predecessors, touched) =
            gpu_chronology_with_fields(&[read], &[0], &[], &initial_memory, &config).unwrap();

        assert_eq!(resolved[0].value, [1, 2, 3, 4]);
        assert!(seeds.is_empty());
        assert!(field_values.is_empty());
        assert!(field_seeds.is_empty());
        assert_eq!(predecessors, [0]);
        assert_eq!(touched.len(), 1);
        assert_eq!(touched[0].is_dirty, 0);

        // Dirtiness records the write itself, even when the value is unchanged
        // and a later read is the block's final event.
        let write = event_value(1, RV64_MEMORY_AS, 0, true, [1, 2, 3, 4]);
        let read = event_value(2, RV64_MEMORY_AS, 0, false, [0; 4]);
        let (_, _, _, _, _, touched) =
            gpu_chronology_with_fields(&[write, read], &[0xff, 0], &[], &initial_memory, &config)
                .unwrap();
        assert_eq!(touched.len(), 1);
        assert_eq!(touched[0].is_dirty, 1);
    }

    #[test]
    fn gpu_chronology_rejects_partial_or_noncanonical_field_values() {
        let (config, initial_memory) = mixed_chronology_fixture();
        let write = field_event(1, 0, true, 0);
        let valid = [PostflightFieldBlock {
            values: [1, 2, 3, 4],
        }];

        assert!(
            gpu_chronology_with_fields(&[write], &[0x0f], &valid, &initial_memory, &config,)
                .is_err()
        );

        let invalid = [PostflightFieldBlock {
            values: [BabyBear::ORDER_U32, 2, 3, 4],
        }];
        assert!(
            gpu_chronology_with_fields(&[write], &[0xff], &invalid, &initial_memory, &config,)
                .is_err()
        );

        let malformed_reference = PreflightMemoryEvent {
            value: [0, 0, 1, 0],
            ..write
        };
        assert!(gpu_chronology_with_fields(
            &[malformed_reference],
            &[0xff],
            &valid,
            &initial_memory,
            &config,
        )
        .is_err());

        let out_of_bounds_reference = field_event(1, 0, true, 1);
        assert!(gpu_chronology_with_fields(
            &[out_of_bounds_reference],
            &[0xff],
            &valid,
            &initial_memory,
            &config,
        )
        .is_err());

        let nonzero_read = field_event(1, 0, false, 0);
        assert!(gpu_chronology_with_fields(
            &[nonzero_read],
            &[0],
            &valid,
            &initial_memory,
            &config,
        )
        .is_err());

        let mut short_initial_memory = initial_memory.clone();
        short_initial_memory[DEFERRAL_AS as usize].truncate(8);
        assert!(gpu_chronology_with_fields(
            &[write],
            &[0xff],
            &valid,
            &short_initial_memory,
            &config,
        )
        .is_err());

        let mut noncanonical_initial_memory = initial_memory.clone();
        noncanonical_initial_memory[DEFERRAL_AS as usize][0..4]
            .copy_from_slice(&BabyBear::ORDER_U32.to_le_bytes());
        assert!(gpu_chronology_with_fields(
            &[write],
            &[0xff],
            &valid,
            &noncanonical_initial_memory,
            &config,
        )
        .is_err());

        let mut wrong_field_space = config.clone();
        let wrong_address_space = DEFERRAL_AS + 1;
        wrong_field_space.addr_spaces[wrong_address_space as usize].num_cells = 4;
        let mut wrong_field_memory = initial_memory;
        wrong_field_memory[wrong_address_space as usize].resize(16, 0);
        let wrong_space_event = PreflightMemoryEvent {
            address_space_and_kind: wrong_address_space | PREFLIGHT_WRITE_BIT,
            ..write
        };
        assert!(gpu_chronology_with_fields(
            &[wrong_space_event],
            &[0xff],
            &valid,
            &wrong_field_memory,
            &wrong_field_space,
        )
        .is_err());
    }

    #[test]
    fn gpu_program_rejects_memory_configs_outside_the_compact_key_abi() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let program = Program::<BabyBear>::from_instructions(&[]);
        let assert_invalid = |config: &MemoryConfig| {
            assert!(matches!(
                GpuPostflightProgram::upload(&program, config, &device_ctx),
                Err(GpuPostflightError::InvalidMemoryConfig(_))
            ));
        };

        let ordinary = MemoryConfig::default();
        let uploaded = GpuPostflightProgram::upload(&program, &ordinary, &device_ctx).unwrap();
        assert_eq!(
            uploaded.byte_pointer_max_bits,
            to_byte_ptr_bits(ordinary.pointer_max_bits).min(u32::BITS as usize) as u32
        );
        assert_eq!(
            uploaded.cell_pointer_max_bits,
            ordinary.pointer_max_bits as u32
        );

        for pointer_max_bits in [1, 33] {
            let config = MemoryConfig {
                pointer_max_bits,
                ..MemoryConfig::default()
            };
            assert_invalid(&config);
        }

        let timestamp_too_wide = MemoryConfig {
            timestamp_max_bits: 32,
            ..MemoryConfig::default()
        };
        assert_invalid(&timestamp_too_wide);

        let label_too_wide = MemoryConfig {
            pointer_max_bits: 32,
            ..MemoryConfig::default()
        };
        assert_invalid(&label_too_wide);

        let mut malformed_layout = MemoryConfig::default();
        malformed_layout.addr_spaces.pop();
        assert_invalid(&malformed_layout);

        let mut maximum = MemoryConfig {
            addr_space_height: 2,
            pointer_max_bits: 32,
            ..MemoryConfig::default()
        };
        maximum
            .addr_spaces
            .truncate(ADDR_SPACE_OFFSET as usize + (1 << maximum.addr_space_height));
        GpuPostflightProgram::upload(&program, &maximum, &device_ctx).unwrap();
    }

    #[test]
    fn gpu_program_index_matches_cpu_oracle_and_preserves_order() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let opcodes = [100, 200, terminate];
        let program = gpu_program(&opcodes, &device_ctx);
        let transcript = PreflightEventLog {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 2,
                },
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 3,
                },
                PreflightProgramEvent {
                    pc: 4,
                    timestamp: 4,
                },
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: 5,
                },
                PreflightProgramEvent {
                    pc: 8,
                    timestamp: 5,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let endpoint = PreflightEndpoint::Terminated;
        let expected =
            super::super::postflight::ReplayData::build(0, &opcodes, &transcript, endpoint)
                .unwrap();
        let actual = gpu_plan(&program, &transcript, endpoint).unwrap();
        let actual_steps = actual.steps.to_host_on(&device_ctx).unwrap();
        assert_eq!(actual_steps, expected.steps());
        assert_eq!(&actual.opcode_ranges, expected.opcode_ranges());
        assert_eq!(
            actual_steps[actual.opcode_ranges[&100].clone()]
                .iter()
                .map(|step| step.program_index)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            actual_steps[actual.opcode_ranges[&200].clone()]
                .iter()
                .map(|step| step.program_index)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn gpu_program_frequencies_are_dense_and_exclude_the_sentinel() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let mut program = gpu_program(&[100, u32::MAX, 200, 300, terminate], &device_ctx);
        program.pc_base = 0x100;
        let transcript = PreflightEventLog {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0x100,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0x108,
                    timestamp: 2,
                },
                PreflightProgramEvent {
                    pc: 0x100,
                    timestamp: 3,
                },
                PreflightProgramEvent {
                    pc: 0x110,
                    timestamp: 4,
                },
                PreflightProgramEvent {
                    pc: 0x110,
                    timestamp: 4,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let plan = gpu_plan(&program, &transcript, PreflightEndpoint::Terminated).unwrap();
        assert_eq!(
            plan.program_frequencies.to_host_on(&device_ctx).unwrap(),
            vec![2, 1, 0, 1]
        );
        assert_eq!(
            plan.connector_boundary(),
            (
                ExecutionState::new(0x100u32, 1u32),
                ExecutionState::new(0x110u32, 4u32),
                Some(0)
            )
        );

        let suspended = PreflightEventLog {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0x100,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0x108,
                    timestamp: 2,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let plan = gpu_plan(&program, &suspended, PreflightEndpoint::Suspended).unwrap();
        assert_eq!(
            plan.program_frequencies.to_host_on(&device_ctx).unwrap(),
            vec![1, 0, 0, 0]
        );
        assert_eq!(
            plan.connector_boundary(),
            (
                ExecutionState::new(0x100u32, 1u32),
                ExecutionState::new(0x108u32, 2u32),
                None
            )
        );

        let empty = PreflightEventLog {
            program_log: vec![PreflightProgramEvent {
                pc: 0x100,
                timestamp: 1,
            }],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let plan = gpu_plan(&program, &empty, PreflightEndpoint::Suspended).unwrap();
        assert_eq!(
            plan.program_frequencies.to_host_on(&device_ctx).unwrap(),
            vec![0; 4]
        );
    }

    #[test]
    fn gpu_program_frequency_input_rejects_invalid_program_counters() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let mut program = gpu_program(&[100, u32::MAX, 200], &device_ctx);
        program.pc_base = 0x100;
        for invalid_pc in [0xfc, 0x102, 0x104, 0x10c] {
            let transcript = PreflightEventLog {
                program_log: vec![
                    PreflightProgramEvent {
                        pc: invalid_pc,
                        timestamp: 1,
                    },
                    PreflightProgramEvent {
                        pc: invalid_pc,
                        timestamp: 2,
                    },
                ],
                memory_log: vec![],
                initial_write_log: vec![],
            };
            assert!(gpu_plan(&program, &transcript, PreflightEndpoint::Suspended,).is_err());
        }
    }

    #[test]
    fn gpu_program_index_accepts_an_empty_suspended_segment() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let program = gpu_program(&[100], &device_ctx);
        let transcript = PreflightEventLog {
            program_log: vec![PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            }],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        let endpoint = PreflightEndpoint::Suspended;
        let plan = gpu_plan(&program, &transcript, endpoint).unwrap();
        assert!(plan.steps.is_empty());
        assert!(plan.opcode_ranges.is_empty());
    }

    #[test]
    fn gpu_program_index_rejects_the_timestamp_domain_limit() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let mut program = gpu_program(&[100], &device_ctx);
        program.timestamp_max_bits = 2;
        let transcript = |final_timestamp| PreflightEventLog {
            program_log: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: final_timestamp,
                },
            ],
            memory_log: vec![],
            initial_write_log: vec![],
        };
        program
            .upload_transcript(&transcript(3), PreflightEndpoint::Suspended)
            .unwrap();
        assert!(program
            .upload_transcript(&transcript(4), PreflightEndpoint::Suspended,)
            .is_err());
    }

    #[test]
    fn gpu_program_index_rejects_malformed_boundaries() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let program = gpu_program(&[100, terminate], &device_ctx);
        let transcript = |program_log| PreflightEventLog {
            program_log,
            memory_log: vec![],
            initial_write_log: vec![],
        };

        let undefined_pc = transcript(vec![
            PreflightProgramEvent {
                pc: 12,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 12,
                timestamp: 2,
            },
        ]);
        assert!(gpu_plan(&program, &undefined_pc, PreflightEndpoint::Suspended,).is_err());

        let missing_terminate = transcript(vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 0,
                timestamp: 2,
            },
        ]);
        assert!(gpu_plan(&program, &missing_terminate, PreflightEndpoint::Terminated,).is_err());

        let timestamp_regression = transcript(vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 2,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 1,
            },
        ]);
        assert!(gpu_plan(
            &program,
            &timestamp_regression,
            PreflightEndpoint::Terminated,
        )
        .is_err());
    }
}
