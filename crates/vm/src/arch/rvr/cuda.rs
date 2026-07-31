//! Compiled checkpoint expansion for GPU preflight.
//!
//! This module converts compact RVR checkpoints into the same immutable
//! history consumed by the generic CUDA postflight indexer. Chronology,
//! opcode indexing, and trace generation live in `arch::cuda::postflight`.

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::{DeviceBuffer, DeviceBufferView},
    stream::GpuDeviceCtx,
};
use openvm_instructions::{
    program::Program,
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_BYTES},
    DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use p3_baby_bear::BabyBear;
use rvr_state::{PreflightMemoryEvent, PreflightProgramEvent, RvrCheckpoint};

use crate::{
    arch::{
        cuda::postflight::{
            gpu_buffer, upload, GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram,
            GpuPostflightTranscript, GpuReplayInstruction,
        },
        rvr::{bridge::read_rv64_registers, preflight::PreflightExecution, PreflightEndpoint},
        to_byte_ptr_bits, MemoryConfig, PreflightFieldBlock, BLOCK_FE_WIDTH,
    },
    cuda_abi::rvr_checkpoint_replay,
};

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

const RVR_REPLAY_NO_SCHEDULE: u32 = u32::MAX;
const RVR_REPLAY_MAX_DENSE_OPCODE: u32 = u16::MAX as u32;
const RVR_REPLAY_EFFECT_NEXT: u8 = 0;
const RVR_REPLAY_EFFECT_BRANCH_REPLAY_VALUE: u8 = 1;
const RVR_REPLAY_REGISTER_WRITE_NONE: u8 = 0;
const RVR_REPLAY_REGISTER_WRITE_ZERO: u8 = 1;
const RVR_REPLAY_REGISTER_WRITE_REPLAY_VALUE: u8 = 2;
const RVR_REPLAY_SPAN_BASE_REGISTER: u8 = 0;
const RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT: u8 = 1;
const RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT: u8 = 2;
const RVR_REPLAY_SPAN_COUNT_FIXED: u8 = 0;
const RVR_REPLAY_SPAN_COUNT_REGISTER: u8 = 1;
const RVR_REPLAY_SPAN_COUNT_REPLAY_VALUE: u8 = 2;
const RVR_REPLAY_SPAN_READ_U16: u8 = 0;
const RVR_REPLAY_SPAN_WRITE_U16_REPLAY_VALUE: u8 = 1;
const RVR_REPLAY_SPAN_WRITE_U16_ZERO: u8 = 2;
const RVR_REPLAY_SPAN_READ_FIELD32: u8 = 3;
const RVR_REPLAY_SPAN_WRITE_FIELD32_CANONICAL_PAIRS: u8 = 4;
const RVR_REPLAY_SPAN_WRITE_U16_STATIC: u8 = 5;
const RVR_REPLAY_DEFERRAL_DIGEST_BLOCKS: u32 = 2;

/// One contiguous sequence of fixed-width memory-bus accesses in an
/// extension-owned checkpoint replay schedule. U16 spans access eight bytes per
/// event; FIELD32 spans access four field cells per event.
///
/// The finite source tags distinguish ordinary RV64 heap blocks from Deferral
/// accumulator blocks. This is intentionally not a general address-expression
/// language: each supported source has one canonical interpretation in replay.
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
            base_source: RVR_REPLAY_SPAN_BASE_REGISTER,
            count_register: 0,
            count_shift: 0,
            count_source: RVR_REPLAY_SPAN_COUNT_FIXED,
            value_source: RVR_REPLAY_SPAN_READ_U16,
            value_index: 0,
        }
    }

    pub const fn write_fixed_from_replay_values(
        address_space: u32,
        base_register: u8,
        count: u32,
    ) -> Self {
        Self {
            value_source: RVR_REPLAY_SPAN_WRITE_U16_REPLAY_VALUE,
            ..Self::read_fixed(address_space, base_register, count)
        }
    }

    /// A fixed span whose postimage is statically zero. This consumes the AIR's
    /// write slots without adding redundant zeroes to the serial transcript.
    pub const fn write_fixed_zero(address_space: u32, base_register: u8, count: u32) -> Self {
        Self {
            value_source: RVR_REPLAY_SPAN_WRITE_U16_ZERO,
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
            base_source: RVR_REPLAY_SPAN_BASE_REGISTER,
            count_register,
            count_shift,
            count_source: RVR_REPLAY_SPAN_COUNT_REGISTER,
            value_source: RVR_REPLAY_SPAN_READ_U16,
            value_index: 0,
        }
    }

    pub const fn write_register_count_from_replay_values(
        address_space: u32,
        base_register: u8,
        count_register: u8,
        count_shift: u8,
        max_count: u32,
    ) -> Self {
        Self {
            value_source: RVR_REPLAY_SPAN_WRITE_U16_REPLAY_VALUE,
            ..Self::read_count_from_register(
                address_space,
                base_register,
                count_register,
                count_shift,
                max_count,
            )
        }
    }

    /// A variable-size write whose next replay value is the number of eight-byte
    /// blocks and whose following replay values are the block postimages.
    pub const fn write_dynamic_from_replay_values(
        address_space: u32,
        base_register: u8,
        max_count: u32,
    ) -> Self {
        Self {
            count_source: RVR_REPLAY_SPAN_COUNT_REPLAY_VALUE,
            value_source: RVR_REPLAY_SPAN_WRITE_U16_REPLAY_VALUE,
            ..Self::read_fixed(address_space, base_register, max_count)
        }
    }

    /// Two consecutive four-cell reads of a Deferral input accumulator. The
    /// base is `16 * instruction[deferral_idx_operand]` in AS4 cell units.
    pub const fn read_deferral_input_accumulator(deferral_idx_operand: u8) -> Self {
        Self::deferral_accumulator(
            deferral_idx_operand,
            RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT,
            RVR_REPLAY_SPAN_READ_FIELD32,
        )
    }

    /// Two consecutive four-cell reads of a Deferral output accumulator. The
    /// base is `16 * instruction[deferral_idx_operand] + 8` in AS4 cell units.
    pub const fn read_deferral_output_accumulator(deferral_idx_operand: u8) -> Self {
        Self::deferral_accumulator(
            deferral_idx_operand,
            RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT,
            RVR_REPLAY_SPAN_READ_FIELD32,
        )
    }

    /// Two consecutive four-cell writes of a Deferral input accumulator. Each
    /// block consumes two u64 replay values containing four canonical u32 cells.
    pub const fn write_deferral_input_accumulator(deferral_idx_operand: u8) -> Self {
        Self::deferral_accumulator(
            deferral_idx_operand,
            RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT,
            RVR_REPLAY_SPAN_WRITE_FIELD32_CANONICAL_PAIRS,
        )
    }

    /// Two consecutive four-cell writes of a Deferral output accumulator.
    pub const fn write_deferral_output_accumulator(deferral_idx_operand: u8) -> Self {
        Self::deferral_accumulator(
            deferral_idx_operand,
            RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT,
            RVR_REPLAY_SPAN_WRITE_FIELD32_CANONICAL_PAIRS,
        )
    }

    const fn deferral_accumulator(
        deferral_idx_operand: u8,
        base_source: u8,
        value_source: u8,
    ) -> Self {
        Self {
            address_space: DEFERRAL_AS,
            count: RVR_REPLAY_DEFERRAL_DIGEST_BLOCKS,
            base_index: deferral_idx_operand,
            base_source,
            count_register: 0,
            count_shift: 0,
            count_source: RVR_REPLAY_SPAN_COUNT_FIXED,
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
struct RvrReplayAccessSchedule {
    first_span: u32,
    num_spans: u32,
    register_operands: [u8; 3],
    num_register_reads: u8,
    effect: u8,
    effect_operand: u8,
    register_write_source: u8,
    register_write_operand: u8,
}

const _: () = assert!(size_of::<RvrReplayAccessSchedule>() == 16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RvrReplayInstructionLayout {
    zero_operand_mask: u32,
    register_as_operand: u8,
    memory_as_operand: u8,
}

/// Static extension access schedules uploaded once with a GPU program. They
/// describe access order only and are not part of the preflight transcript.
///
/// This is an internal composition seam, not a stable extension API. The
/// supported sources remain a finite POD set: replay values, zero, program-owned
/// constants, and Deferral's field accumulator blocks.
#[derive(Clone, Debug, Default)]
pub struct PostflightAccessRegistry {
    dispatch: Vec<u32>,
    schedules: Vec<RvrReplayAccessSchedule>,
    instruction_layouts: Vec<RvrReplayInstructionLayout>,
    spans: Vec<PostflightAccessSpan>,
    static_values: Vec<u64>,
}

/// Borrowed semantic description of one extension instruction's replay
/// accesses. Registration validates the operand roles and converts this into
/// the compact CUDA schedule stored with the program.
#[derive(Clone, Copy, Debug)]
pub struct PostflightAccessSchedule<'a> {
    pub register_operands: &'a [u8],
    pub zero_operand_mask: u32,
    pub register_as_operand: u8,
    pub memory_as_operand: u8,
    pub spans: &'a [PostflightAccessSpan],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PostflightEffect {
    Next,
    BranchFromReplayValue { operand: u8 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PostflightRegisterWrite {
    None,
    Zero { operand: u8 },
    ReplayValue { operand: u8 },
}

impl PostflightAccessRegistry {
    /// Adds one fixed sequence of eight-byte write values to the program-owned
    /// replay data and returns the span that consumes it. This is used for
    /// setup instructions whose postimage depends only on VM configuration.
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
            value_source: RVR_REPLAY_SPAN_WRITE_U16_STATIC,
            value_index,
            ..PostflightAccessSpan::read_fixed(address_space, base_register, count)
        })
    }

    pub fn register(
        &mut self,
        opcode: u32,
        schedule: PostflightAccessSchedule<'_>,
    ) -> Result<(), GpuPostflightError> {
        self.register_with_effect(
            opcode,
            schedule,
            PostflightEffect::Next,
            PostflightRegisterWrite::None,
        )
    }

    /// Registers a schedule whose final clock slot writes zero to a register.
    /// An x0 destination reserves the slot but emits no memory event.
    pub fn register_with_zero_register_write(
        &mut self,
        opcode: u32,
        schedule: PostflightAccessSchedule<'_>,
        write_operand: u8,
    ) -> Result<(), GpuPostflightError> {
        self.register_with_effect(
            opcode,
            schedule,
            PostflightEffect::Next,
            PostflightRegisterWrite::Zero {
                operand: write_operand,
            },
        )
    }

    /// Registers a schedule whose final clock slot writes one replay value to a
    /// register. An x0 destination consumes the replay value and slot but emits no
    /// memory event.
    pub fn register_with_replay_value_write(
        &mut self,
        opcode: u32,
        schedule: PostflightAccessSchedule<'_>,
        write_operand: u8,
    ) -> Result<(), GpuPostflightError> {
        self.register_with_effect(
            opcode,
            schedule,
            PostflightEffect::Next,
            PostflightRegisterWrite::ReplayValue {
                operand: write_operand,
            },
        )
    }

    pub fn register_branch_from_replay_value(
        &mut self,
        opcode: u32,
        schedule: PostflightAccessSchedule<'_>,
        branch_operand: u8,
    ) -> Result<(), GpuPostflightError> {
        self.register_with_effect(
            opcode,
            schedule,
            PostflightEffect::BranchFromReplayValue {
                operand: branch_operand,
            },
            PostflightRegisterWrite::None,
        )
    }

    fn register_with_effect(
        &mut self,
        opcode: u32,
        schedule: PostflightAccessSchedule<'_>,
        effect: PostflightEffect,
        register_write: PostflightRegisterWrite,
    ) -> Result<(), GpuPostflightError> {
        let PostflightAccessSchedule {
            register_operands,
            zero_operand_mask,
            register_as_operand,
            memory_as_operand,
            spans,
        } = schedule;
        let effect_operand = match effect {
            PostflightEffect::Next => 0,
            PostflightEffect::BranchFromReplayValue { operand } => operand,
        };
        let register_write_operand = match register_write {
            PostflightRegisterWrite::None => 0,
            PostflightRegisterWrite::Zero { operand }
            | PostflightRegisterWrite::ReplayValue { operand } => operand,
        };
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
            || (matches!(effect, PostflightEffect::BranchFromReplayValue { .. })
                && (!(1..8).contains(&effect_operand)
                    || register_operands.contains(&effect_operand)
                    || effect_operand == register_as_operand
                    || effect_operand == memory_as_operand
                    || zero_operand_mask & (1 << effect_operand) != 0))
            || (!matches!(register_write, PostflightRegisterWrite::None)
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
        if opcode > RVR_REPLAY_MAX_DENSE_OPCODE {
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
            self.dispatch.resize(dispatch_len, RVR_REPLAY_NO_SCHEDULE);
        }
        if self.dispatch[opcode as usize] != RVR_REPLAY_NO_SCHEDULE {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "duplicate checkpoint access schedule for opcode {opcode}"
            )));
        }
        for span in spans {
            let field = matches!(
                span.value_source,
                RVR_REPLAY_SPAN_READ_FIELD32 | RVR_REPLAY_SPAN_WRITE_FIELD32_CANONICAL_PAIRS
            );
            let base_valid = match span.base_source {
                RVR_REPLAY_SPAN_BASE_REGISTER => {
                    usize::from(span.base_index) < register_operands.len()
                }
                RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT | RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT => {
                    (1..8).contains(&span.base_index)
                        && !register_operands.contains(&span.base_index)
                        && span.base_index != register_as_operand
                        && span.base_index != memory_as_operand
                        && zero_operand_mask & (1 << span.base_index) == 0
                }
                _ => false,
            };
            let count_valid = match span.count_source {
                RVR_REPLAY_SPAN_COUNT_FIXED => {
                    span.count != 0 && span.count_register == 0 && span.count_shift == 0
                }
                RVR_REPLAY_SPAN_COUNT_REGISTER => {
                    usize::from(span.count_register) < register_operands.len()
                        && span.count_shift < u64::BITS as u8
                }
                RVR_REPLAY_SPAN_COUNT_REPLAY_VALUE => {
                    span.count != 0 && span.count_register == 0 && span.count_shift == 0
                }
                _ => false,
            };
            let value_valid = matches!(
                span.value_source,
                RVR_REPLAY_SPAN_READ_U16
                    | RVR_REPLAY_SPAN_WRITE_U16_REPLAY_VALUE
                    | RVR_REPLAY_SPAN_WRITE_U16_ZERO
                    | RVR_REPLAY_SPAN_READ_FIELD32
                    | RVR_REPLAY_SPAN_WRITE_FIELD32_CANONICAL_PAIRS
                    | RVR_REPLAY_SPAN_WRITE_U16_STATIC
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
                        RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT | RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT
                    ))
                || (!field && span.base_source != RVR_REPLAY_SPAN_BASE_REGISTER)
                || (field && span.count_source != RVR_REPLAY_SPAN_COUNT_FIXED)
                || (field && span.count != RVR_REPLAY_DEFERRAL_DIGEST_BLOCKS)
                || (span.value_source == RVR_REPLAY_SPAN_WRITE_U16_STATIC
                    && (span.count_source != RVR_REPLAY_SPAN_COUNT_FIXED || static_end.is_none()))
                || (span.value_source != RVR_REPLAY_SPAN_WRITE_U16_STATIC && span.value_index != 0)
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
        first_span.checked_add(num_spans).ok_or_else(|| {
            GpuPostflightError::InvalidAccessSchedule(
                "access schedule exceeds the u32 span index domain".to_string(),
            )
        })?;
        let schedule_index = u32::try_from(self.schedules.len()).map_err(|_| {
            GpuPostflightError::InvalidAccessSchedule("too many access schedules".to_string())
        })?;
        let mut operand_words = [0u8; 3];
        operand_words[..register_operands.len()].copy_from_slice(register_operands);
        self.spans.extend_from_slice(spans);
        let (effect, effect_operand) = match effect {
            PostflightEffect::Next => (RVR_REPLAY_EFFECT_NEXT, 0),
            PostflightEffect::BranchFromReplayValue { operand } => {
                (RVR_REPLAY_EFFECT_BRANCH_REPLAY_VALUE, operand)
            }
        };
        let (register_write_source, register_write_operand) = match register_write {
            PostflightRegisterWrite::None => (RVR_REPLAY_REGISTER_WRITE_NONE, 0),
            PostflightRegisterWrite::Zero { operand } => (RVR_REPLAY_REGISTER_WRITE_ZERO, operand),
            PostflightRegisterWrite::ReplayValue { operand } => {
                (RVR_REPLAY_REGISTER_WRITE_REPLAY_VALUE, operand)
            }
        };
        self.schedules.push(RvrReplayAccessSchedule {
            first_span,
            num_spans,
            register_operands: operand_words,
            num_register_reads: register_operands.len() as u8,
            effect,
            effect_operand,
            register_write_source,
            register_write_operand,
        });
        self.instruction_layouts.push(RvrReplayInstructionLayout {
            zero_operand_mask,
            register_as_operand,
            memory_as_operand,
        });
        debug_assert_eq!(self.schedules.len(), self.instruction_layouts.len());
        self.dispatch[opcode as usize] = schedule_index;
        Ok(())
    }

    pub fn validate_no_native_collisions(
        &self,
        opcodes: PostflightOpcodeBases,
    ) -> Result<(), GpuPostflightError> {
        if let Some(opcode) = self
            .dispatch
            .iter()
            .enumerate()
            .find_map(|(opcode, &schedule)| {
                (schedule != RVR_REPLAY_NO_SCHEDULE && opcodes.owns(opcode as u32))
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
        instruction: &GpuReplayInstruction,
        cell_pointer_max_bits: usize,
        deferral_num_cells: usize,
    ) -> Result<(), GpuPostflightError> {
        let opcode = instruction.words[0] as usize;
        let Some(&schedule_index) = self.dispatch.get(opcode) else {
            return Ok(());
        };
        if schedule_index == RVR_REPLAY_NO_SCHEDULE {
            return Ok(());
        }
        debug_assert_eq!(self.schedules.len(), self.instruction_layouts.len());
        let schedule = self
            .schedules
            .get(schedule_index as usize)
            .expect("registered dispatch must reference a schedule");
        let layout = self
            .instruction_layouts
            .get(schedule_index as usize)
            .expect("registered schedule must have a host instruction layout");
        let span_start = schedule.first_span as usize;
        let span_end = (schedule.first_span + schedule.num_spans) as usize;
        let schedule_spans = self
            .spans
            .get(span_start..span_end)
            .expect("registered schedule must reference valid access spans");
        let invalid_deferral_span = schedule_spans
            .iter()
            .filter(|span| {
                matches!(
                    span.base_source,
                    RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT | RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT
                )
            })
            .any(|span| {
                let base = u64::from(instruction.words[span.base_index as usize]) * 16
                    + u64::from(span.base_source == RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT) * 8;
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
            || (schedule.register_write_source != RVR_REPLAY_REGISTER_WRITE_NONE && {
                let pointer =
                    u64::from(instruction.words[schedule.register_write_operand as usize]);
                pointer >= 32 * RV64_REGISTER_BYTES || pointer % RV64_REGISTER_BYTES != 0
            })
            || (schedule.effect == RVR_REPLAY_EFFECT_BRANCH_REPLAY_VALUE
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

pub struct PreflightReplayProgram {
    program: GpuPostflightProgram,
    schedule_dispatch: DeviceBuffer<u32>,
    access_schedules: DeviceBuffer<RvrReplayAccessSchedule>,
    access_spans: DeviceBuffer<PostflightAccessSpan>,
    static_values: DeviceBuffer<u64>,
    scheduled_opcodes: Vec<u32>,
    byte_pointer_max_bits: u32,
}

impl PreflightReplayProgram {
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

    /// Uploads one program together with the extension schedules used only by
    /// compiled checkpoint expansion.
    pub fn upload_with_postflight_access_registry<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        registry: &PostflightAccessRegistry,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        let program =
            GpuPostflightProgram::upload_validated(program, memory_config, device_ctx, |instr| {
                registry.validate_instruction(
                    instr,
                    memory_config.pointer_max_bits,
                    memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells,
                )
            })?;
        let scheduled_opcodes = registry
            .dispatch
            .iter()
            .enumerate()
            .filter_map(|(opcode, &schedule)| {
                (schedule != RVR_REPLAY_NO_SCHEDULE).then_some(opcode as u32)
            })
            .collect();
        Ok(Self {
            program,
            schedule_dispatch: upload(&registry.dispatch, device_ctx)?,
            access_schedules: upload(&registry.schedules, device_ctx)?,
            access_spans: upload(&registry.spans, device_ctx)?,
            static_values: upload(&registry.static_values, device_ctx)?,
            scheduled_opcodes,
            byte_pointer_max_bits: to_byte_ptr_bits(memory_config.pointer_max_bits)
                .min(u32::BITS as usize) as u32,
        })
    }

    /// Generic immutable-program view consumed by postflight indexing and
    /// read-only trace generation.
    pub const fn program(&self) -> &GpuPostflightProgram {
        &self.program
    }
}

impl PreflightReplayProgram {
    /// Expands checkpoint execution into ordinary program and memory logs.
    /// Independent chunks emit byte-masked block intents; one device-wide
    /// chronology pass resolves them against the segment-start memory image.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn postflight(
        &self,
        execution: &PreflightExecution,
        num_insns: u32,
        initial_registers: DeviceBufferView,
        initial_memory: DeviceBufferView,
        initial_memory_images: &[DeviceBufferView],
        opcodes: PostflightOpcodeBases,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let program = self.program();
        let instret = execution.retired;
        if instret != num_insns {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "preflight instret {instret} does not match segment num_insns {num_insns}"
            )));
        }
        if let Some(opcode) = self
            .scheduled_opcodes
            .iter()
            .copied()
            .find(|&opcode| opcodes.owns(opcode))
        {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "opcode {opcode} is owned by both native replay and an extension schedule"
            )));
        }
        let endpoint_kind = match execution.endpoint {
            PreflightEndpoint::Terminated => 0,
            PreflightEndpoint::Suspended => 1,
        };
        if execution.from_state.timestamp != 1
            || execution.to_state.pc != execution.state.pc()
            || execution.to_state.timestamp >= (1u32 << program.timestamp_max_bits())
        {
            return Err(GpuPostflightError::InvalidTranscript(
                "preflight has an invalid boundary".to_string(),
            ));
        }
        let replay_value_cursor =
            u32::try_from(execution.transcript.replay_values.len()).map_err(|_| {
                GpuPostflightError::InvalidTranscript(
                    "checkpoint replay-value stream exceeds the u32 cursor domain".to_string(),
                )
            })?;
        let final_registers = read_rv64_registers(&execution.state);
        let mut final_anchor = RvrCheckpoint {
            pc: execution.to_state.pc,
            timestamp: execution.to_state.timestamp,
            retired: execution.retired,
            replay_value_cursor,
            regs: [0; 31],
        };
        final_anchor.regs.copy_from_slice(&final_registers[1..]);
        let mut anchors = execution.transcript.checkpoints.clone();
        anchors.push(final_anchor);
        let anchors = upload(&anchors, program.device_ctx())?;
        let replay_values = upload(&execution.transcript.replay_values, program.device_ctx())?;
        let error = [0u32].to_device_on(program.device_ctx())?;
        let event_counts = gpu_buffer::<PostflightEventCount>(anchors.len(), program.device_ctx());
        event_counts.fill_zero_on(program.device_ctx())?;
        let address_spaces = [RV64_REGISTER_AS, RV64_MEMORY_AS, RV64_IMM_AS, DEFERRAL_AS];
        let count_span = tracing::info_span!("postflight_replay_count").entered();
        unsafe {
            rvr_checkpoint_replay::count(
                program.instructions(),
                program.pc_base(),
                initial_registers,
                initial_memory,
                anchors.view(),
                replay_values.view(),
                self.schedule_dispatch.view(),
                self.access_schedules.view(),
                self.access_spans.view(),
                self.static_values.view(),
                opcodes,
                address_spaces,
                self.byte_pointer_max_bits,
                program.cell_pointer_max_bits(),
                execution.from_state.pc,
                execution.from_state.timestamp,
                endpoint_kind,
                &event_counts,
                &error,
                program.device_ctx().stream.as_raw(),
            )?;
        }
        let count_error = error.to_host_on(program.device_ctx())?[0];
        if count_error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "checkpoint GPU count replay rejected execution with code {count_error}"
            )));
        }
        let counts = event_counts.to_host_on(program.device_ctx())?;
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
        let program_log = gpu_buffer::<PreflightProgramEvent>(program_len, program.device_ctx());
        let memory_log =
            gpu_buffer::<PreflightMemoryEvent>(total_memory as usize, program.device_ctx());
        let field_values =
            gpu_buffer::<PreflightFieldBlock>(total_fields as usize, program.device_ctx());
        // One transient byte per event is enough to distinguish reads, full
        // writes, and partial block writes. The chronology pass consumes and
        // releases this before opcode trace generation.
        let write_masks = gpu_buffer::<u8>(total_memory as usize, program.device_ctx());
        let offsets = upload(&offsets, program.device_ctx())?;
        let emit_span = tracing::info_span!("postflight_replay_emit").entered();
        unsafe {
            rvr_checkpoint_replay::emit(
                program.instructions(),
                program.pc_base(),
                initial_registers,
                initial_memory,
                anchors.view(),
                replay_values.view(),
                offsets.view(),
                self.schedule_dispatch.view(),
                self.access_schedules.view(),
                self.access_spans.view(),
                self.static_values.view(),
                opcodes,
                address_spaces,
                self.byte_pointer_max_bits,
                program.cell_pointer_max_bits(),
                execution.from_state.pc,
                execution.from_state.timestamp,
                endpoint_kind,
                program_log.view(),
                memory_log.view(),
                write_masks.view(),
                field_values.view(),
                &error,
                program.device_ctx().stream.as_raw(),
            )?;
        }
        let emit_error = error.to_host_on(program.device_ctx())?[0];
        if emit_error != 0 {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "checkpoint GPU emit replay rejected execution with code {emit_error}"
            )));
        }
        drop(emit_span);
        // The host reads above synchronize the emit stream. Release compact
        // replay inputs before chronology allocates its sort/scan scratch.
        drop(anchors);
        drop(replay_values);
        drop(event_counts);
        drop(offsets);
        let boundary = (
            execution.from_state,
            execution.to_state,
            matches!(execution.endpoint, PreflightEndpoint::Terminated).then_some(0),
        );
        program.index_device_history(
            program_log,
            memory_log,
            field_values,
            write_masks,
            error,
            initial_memory_images,
            boundary,
        )
    }
}

#[cfg(test)]
mod tests {
    use openvm_cuda_common::stream::GpuDeviceCtx;
    use openvm_instructions::{
        instruction::Instruction, riscv::RV64_MEMORY_AS, LocalOpcode, SystemOpcode, VmOpcode,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use rvr_state::{
        PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
    };

    use super::*;
    use crate::arch::{
        cuda::postflight::{
            build_memory_chronology_for_test as gpu_chronology_with_fields,
            empty_chronology_counts_for_test,
        },
        postflight::PREDECESSOR_SEED_BIT,
        ExecutionState, MemoryCellType, Postflight, PreflightHistory, ADDR_SPACE_OFFSET,
    };

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

    #[test]
    fn empty_gpu_chronology_zeroes_every_counter() {
        assert_eq!(empty_chronology_counts_for_test(false).unwrap(), vec![0; 3]);
        assert_eq!(empty_chronology_counts_for_test(true).unwrap(), vec![0; 7]);
    }

    fn gpu_program(opcodes: &[u32], device_ctx: &GpuDeviceCtx) -> GpuPostflightProgram {
        GpuPostflightProgram::synthetic_for_test(
            opcodes,
            0,
            MemoryConfig::default().timestamp_max_bits as u32,
            device_ctx,
        )
        .unwrap()
    }

    fn gpu_plan(
        program: &GpuPostflightProgram,
        history: &PreflightHistory,
        endpoint: PreflightEndpoint,
    ) -> Result<GpuPostflightPlan, GpuPostflightError> {
        assert!(history.memory.accesses.is_empty());
        assert!(history.memory.initial_writes.is_empty());
        let first = history.program.first().unwrap();
        let last = history.program.last().unwrap();
        let boundary = (
            ExecutionState::new(first.pc, first.timestamp),
            ExecutionState::new(last.pc, last.timestamp),
            matches!(endpoint, PreflightEndpoint::Terminated).then_some(0),
        );
        program
            .index_program_log_for_test(&history.program, boundary)
            .map(|(_, plan)| plan)
    }

    #[test]
    fn empty_program_cannot_terminate_without_a_terminate_step() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let program = gpu_program(&[terminate], &device_ctx);
        let history = PreflightHistory {
            program: vec![PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            }],
            ..Default::default()
        };
        let state = ExecutionState::new(0u32, 1u32);
        let error =
            match program.index_program_log_for_test(&history.program, (state, state, Some(0))) {
                Ok(_) => panic!("empty program must not terminate without a terminate step"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("code 115"));
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
        let initial_memory = (0..MemoryConfig::default().addr_spaces.len())
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
                (
                    ExecutionState::new(0u32, 1u32),
                    ExecutionState::new(4u32, 4u32),
                    Some(0),
                ),
                &initial_memory_views,
            )
            .unwrap();

        assert_eq!(
            transcript.initial_write_log_host().unwrap(),
            history.memory.initial_writes
        );
        assert_eq!(
            transcript.memory_predecessors_host().unwrap(),
            vec![0, 0, PREDECESSOR_SEED_BIT]
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
                .copy_from_slice(&raw_baby_bear(BabyBear::from_u32(value)).to_le_bytes());
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
        let first_write = PreflightFieldBlock {
            values: [31, 32, 33, 34],
        };
        let second_write = PreflightFieldBlock {
            values: [41, 42, 43, 44],
        };
        let field_values = [
            PreflightFieldBlock::default(),
            first_write,
            PreflightFieldBlock::default(),
            second_write,
            PreflightFieldBlock::default(),
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
            [PreflightFieldBlock {
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
        assert_eq!(
            touched[1].values.map(|value| value.as_canonical_u32()),
            first_write.values
        );
        assert_eq!(
            touched[2].values.map(|value| value.as_canonical_u32()),
            second_write.values
        );
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

        let observed_read = event_value(1, RV64_MEMORY_AS, 0, false, [1, 2, 3, 4]);
        assert!(
            gpu_chronology_with_fields(&[observed_read], &[0], &[], &initial_memory, &config)
                .is_ok()
        );
        let incorrect_read = event_value(1, RV64_MEMORY_AS, 0, false, [9, 2, 3, 4]);
        assert!(
            gpu_chronology_with_fields(&[incorrect_read], &[0], &[], &initial_memory, &config)
                .is_err()
        );

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
        let valid = [PreflightFieldBlock {
            values: [1, 2, 3, 4],
        }];

        assert!(
            gpu_chronology_with_fields(&[write], &[0x0f], &valid, &initial_memory, &config,)
                .is_err()
        );

        let invalid = [PreflightFieldBlock {
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
        let observed = [PreflightFieldBlock {
            values: [11, 12, 13, 14],
        }];
        assert!(gpu_chronology_with_fields(
            &[nonzero_read],
            &[0],
            &observed,
            &initial_memory,
            &config,
        )
        .is_ok());
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
        let uploaded = PreflightReplayProgram::upload(&program, &ordinary, &device_ctx).unwrap();
        assert_eq!(uploaded.byte_pointer_max_bits, 32);
        assert_eq!(uploaded.program().cell_pointer_max_bits(), 31);

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
        let history = PreflightHistory {
            program: vec![
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
            ..Default::default()
        };
        let endpoint = PreflightEndpoint::Terminated;
        let cpu_program = Program::<BabyBear>::new_without_debug_infos(
            &[
                Instruction::from_usize(VmOpcode::from_usize(100), [0; 5]),
                Instruction::from_usize(VmOpcode::from_usize(200), [0; 5]),
                Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0; 5]),
            ],
            0,
        );
        let expected =
            Postflight::new(&cpu_program, &history, &MemoryConfig::default(), Some(0)).unwrap();
        let actual = gpu_plan(&program, &history, endpoint).unwrap();
        let actual_steps = actual.steps_host().unwrap();
        let expected_steps = expected
            .replay_steps_for_test()
            .map(|(program_index, memory_start)| [program_index, memory_start])
            .collect::<Vec<_>>();
        assert_eq!(actual_steps, expected_steps);
        for &opcode in &[100, 200, terminate] {
            assert_eq!(
                actual.opcode_range(VmOpcode::from_usize(opcode as usize)),
                expected
                    .opcode_ranges_for_test()
                    .get(&opcode)
                    .cloned()
                    .unwrap_or(0..0)
            );
        }
        let opcode_100 = actual.opcode_range(VmOpcode::from_usize(100));
        let opcode_200 = actual.opcode_range(VmOpcode::from_usize(200));
        assert_eq!(
            actual_steps[opcode_100]
                .iter()
                .map(|step| step[0])
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            actual_steps[opcode_200]
                .iter()
                .map(|step| step[0])
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn gpu_program_frequencies_are_dense_and_exclude_the_sentinel() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let program = GpuPostflightProgram::synthetic_for_test(
            &[100, u32::MAX, 200, 300, terminate],
            0x100,
            MemoryConfig::default().timestamp_max_bits as u32,
            &device_ctx,
        )
        .unwrap();
        let history = PreflightHistory {
            program: vec![
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
            ..Default::default()
        };
        let plan = gpu_plan(&program, &history, PreflightEndpoint::Terminated).unwrap();
        assert_eq!(plan.program_frequencies_host().unwrap(), vec![2, 1, 0, 1]);
        assert_eq!(
            plan.connector_boundary(),
            (
                ExecutionState::new(0x100u32, 1u32),
                ExecutionState::new(0x110u32, 4u32),
                Some(0)
            )
        );

        let suspended = PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0x100,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0x108,
                    timestamp: 2,
                },
            ],
            ..Default::default()
        };
        let plan = gpu_plan(&program, &suspended, PreflightEndpoint::Suspended).unwrap();
        assert_eq!(plan.program_frequencies_host().unwrap(), vec![1, 0, 0, 0]);
        assert_eq!(
            plan.connector_boundary(),
            (
                ExecutionState::new(0x100u32, 1u32),
                ExecutionState::new(0x108u32, 2u32),
                None
            )
        );

        let empty = PreflightHistory {
            program: vec![PreflightProgramEvent {
                pc: 0x100,
                timestamp: 1,
            }],
            ..Default::default()
        };
        let plan = gpu_plan(&program, &empty, PreflightEndpoint::Suspended).unwrap();
        assert_eq!(plan.program_frequencies_host().unwrap(), vec![0; 4]);
    }

    #[test]
    fn gpu_program_frequency_input_rejects_invalid_program_counters() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let program = GpuPostflightProgram::synthetic_for_test(
            &[100, u32::MAX, 200],
            0x100,
            MemoryConfig::default().timestamp_max_bits as u32,
            &device_ctx,
        )
        .unwrap();
        for invalid_pc in [0xfc, 0x102, 0x104, 0x10c] {
            let history = PreflightHistory {
                program: vec![
                    PreflightProgramEvent {
                        pc: invalid_pc,
                        timestamp: 1,
                    },
                    PreflightProgramEvent {
                        pc: invalid_pc,
                        timestamp: 2,
                    },
                ],
                ..Default::default()
            };
            assert!(gpu_plan(&program, &history, PreflightEndpoint::Suspended,).is_err());
        }
    }

    #[test]
    fn gpu_program_index_accepts_an_empty_suspended_segment() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let program = gpu_program(&[100], &device_ctx);
        let history = PreflightHistory {
            program: vec![PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            }],
            ..Default::default()
        };
        let endpoint = PreflightEndpoint::Suspended;
        let plan = gpu_plan(&program, &history, endpoint).unwrap();
        assert!(plan.steps_host().unwrap().is_empty());
        assert_eq!(plan.executed_opcodes().count(), 0);
    }

    #[test]
    fn gpu_program_index_rejects_the_timestamp_domain_limit() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let program = GpuPostflightProgram::synthetic_for_test(&[100], 0, 2, &device_ctx).unwrap();
        let history = |final_timestamp| PreflightHistory {
            program: vec![
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: 1,
                },
                PreflightProgramEvent {
                    pc: 0,
                    timestamp: final_timestamp,
                },
            ],
            ..Default::default()
        };
        gpu_plan(&program, &history(3), PreflightEndpoint::Suspended).unwrap();
        assert!(gpu_plan(&program, &history(4), PreflightEndpoint::Suspended,).is_err());
    }

    #[test]
    fn gpu_program_index_rejects_malformed_boundaries() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let terminate = SystemOpcode::TERMINATE.global_opcode().as_usize() as u32;
        let program = gpu_program(&[100, terminate], &device_ctx);
        let history = |program| PreflightHistory {
            program,
            ..Default::default()
        };

        let undefined_pc = history(vec![
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

        let missing_terminate = history(vec![
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

        let timestamp_regression = history(vec![
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
