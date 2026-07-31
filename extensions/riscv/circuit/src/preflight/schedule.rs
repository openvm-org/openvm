use openvm_circuit::arch::{
    cuda::postflight::{GpuPostflightError, GpuReplayInstruction, POSTFLIGHT_INSTRUCTION_FIELDS},
    BLOCK_FE_WIDTH,
};
use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_NUM_REGISTERS, RV64_REGISTER_AS, RV64_REGISTER_BYTES},
    DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

const RVR_REPLAY_NO_SCHEDULE: u32 = u32::MAX;
const RVR_REPLAY_MAX_DENSE_OPCODE: u32 = u16::MAX as u32;
const RVR_REPLAY_EFFECT_NEXT: u8 = 0;
const RVR_REPLAY_EFFECT_BRANCH_REPLAY_VALUE: u8 = 1;
const RVR_REPLAY_REGISTER_WRITE_NONE: u8 = 0;
const RVR_REPLAY_REGISTER_WRITE_ZERO: u8 = 1;
const RVR_REPLAY_REGISTER_WRITE_REPLAY_VALUE: u8 = 2;
/// The replay schedule stores at most three register-read operand indexes.
const RVR_REPLAY_REGISTER_OPERANDS: usize = 3;
/// Bits for instruction operands `a..g`; bit zero is the opcode word.
const RVR_REPLAY_OPERAND_MASK: u32 = ((1u32 << POSTFLIGHT_INSTRUCTION_FIELDS) - 1) & !1;
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
const DEFERRAL_PAIR_STRIDE_CELLS: u64 = 2 * BLOCK_FE_WIDTH as u64;
const DEFERRAL_OUTPUT_OFFSET_CELLS: u64 = BLOCK_FE_WIDTH as u64;

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RvrReplayAccessSchedule {
    first_span: u32,
    num_spans: u32,
    register_operands: [u8; RVR_REPLAY_REGISTER_OPERANDS],
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
    pub(super) dispatch: Vec<u32>,
    pub(super) schedules: Vec<RvrReplayAccessSchedule>,
    instruction_layouts: Vec<RvrReplayInstructionLayout>,
    pub(super) spans: Vec<PostflightAccessSpan>,
    pub(super) static_values: Vec<u64>,
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

fn is_instruction_operand(word: u8) -> bool {
    (1..POSTFLIGHT_INSTRUCTION_FIELDS as u8).contains(&word)
}

fn invalid_access_schedule(message: impl Into<String>) -> GpuPostflightError {
    GpuPostflightError::InvalidAccessSchedule(message.into())
}

fn validate_instruction_operand(operand: u8, role: &str) -> Result<(), GpuPostflightError> {
    if !is_instruction_operand(operand) {
        return Err(invalid_access_schedule(format!(
            "{role} must reference instruction operand a..g"
        )));
    }
    Ok(())
}

fn validate_address_space_operands(
    register_as_operand: u8,
    memory_as_operand: u8,
) -> Result<(), GpuPostflightError> {
    validate_instruction_operand(register_as_operand, "register address-space operand")?;
    validate_instruction_operand(memory_as_operand, "memory address-space operand")?;
    if register_as_operand == memory_as_operand {
        return Err(invalid_access_schedule(
            "register and memory address spaces must use distinct operands",
        ));
    }
    Ok(())
}

fn validate_zero_operands(
    zero_operand_mask: u32,
    occupied_operands: impl IntoIterator<Item = u8>,
) -> Result<(), GpuPostflightError> {
    if zero_operand_mask & !RVR_REPLAY_OPERAND_MASK != 0 {
        return Err(invalid_access_schedule(
            "zero-operand mask may only reference instruction operands a..g",
        ));
    }
    if occupied_operands
        .into_iter()
        .any(|operand| zero_operand_mask & (1 << operand) != 0)
    {
        return Err(invalid_access_schedule(
            "zero operands overlap an address-space or register operand",
        ));
    }
    Ok(())
}

fn validate_register_operands(
    operands: &[u8],
    register_as_operand: u8,
    memory_as_operand: u8,
) -> Result<(), GpuPostflightError> {
    if operands.len() > RVR_REPLAY_REGISTER_OPERANDS {
        return Err(invalid_access_schedule(format!(
            "at most {RVR_REPLAY_REGISTER_OPERANDS} register operands are supported"
        )));
    }
    for &operand in operands {
        validate_instruction_operand(operand, "register operand")?;
        if operand == register_as_operand || operand == memory_as_operand {
            return Err(invalid_access_schedule(
                "register operands overlap an address-space operand",
            ));
        }
    }
    Ok(())
}

fn validate_effect_operand(
    effect: PostflightEffect,
    schedule: PostflightAccessSchedule<'_>,
) -> Result<(), GpuPostflightError> {
    let PostflightEffect::BranchFromReplayValue { operand } = effect else {
        return Ok(());
    };
    validate_instruction_operand(operand, "branch effect operand")?;
    if schedule.register_operands.contains(&operand)
        || operand == schedule.register_as_operand
        || operand == schedule.memory_as_operand
        || schedule.zero_operand_mask & (1 << operand) != 0
    {
        return Err(invalid_access_schedule(
            "branch effect operand overlaps another operand role",
        ));
    }
    Ok(())
}

fn validate_register_write_operand(
    register_write: PostflightRegisterWrite,
    schedule: PostflightAccessSchedule<'_>,
) -> Result<(), GpuPostflightError> {
    let operand = match register_write {
        PostflightRegisterWrite::None => return Ok(()),
        PostflightRegisterWrite::Zero { operand }
        | PostflightRegisterWrite::ReplayValue { operand } => operand,
    };
    validate_instruction_operand(operand, "register-write operand")?;
    if operand == schedule.register_as_operand
        || operand == schedule.memory_as_operand
        || schedule.zero_operand_mask & (1 << operand) != 0
    {
        return Err(invalid_access_schedule(
            "register-write operand overlaps another operand role",
        ));
    }
    Ok(())
}

fn validate_access_schedule(
    schedule: PostflightAccessSchedule<'_>,
    effect: PostflightEffect,
    register_write: PostflightRegisterWrite,
) -> Result<(), GpuPostflightError> {
    if schedule.spans.is_empty() {
        return Err(invalid_access_schedule(
            "access schedule must contain at least one span",
        ));
    }
    validate_address_space_operands(schedule.register_as_operand, schedule.memory_as_operand)?;
    validate_register_operands(
        schedule.register_operands,
        schedule.register_as_operand,
        schedule.memory_as_operand,
    )?;
    validate_zero_operands(
        schedule.zero_operand_mask,
        schedule
            .register_operands
            .iter()
            .copied()
            .chain([schedule.register_as_operand, schedule.memory_as_operand]),
    )?;
    validate_effect_operand(effect, schedule)?;
    validate_register_write_operand(register_write, schedule)?;
    Ok(())
}

fn validate_span_base(
    span: &PostflightAccessSpan,
    schedule: PostflightAccessSchedule<'_>,
) -> Result<(), GpuPostflightError> {
    match span.base_source {
        RVR_REPLAY_SPAN_BASE_REGISTER => {
            if usize::from(span.base_index) >= schedule.register_operands.len() {
                return Err(invalid_access_schedule(
                    "span base references a missing register operand",
                ));
            }
        }
        RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT | RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT => {
            validate_instruction_operand(span.base_index, "deferral span base")?;
            if schedule.register_operands.contains(&span.base_index)
                || span.base_index == schedule.register_as_operand
                || span.base_index == schedule.memory_as_operand
                || schedule.zero_operand_mask & (1 << span.base_index) != 0
            {
                return Err(invalid_access_schedule(
                    "deferral span base overlaps another operand role",
                ));
            }
        }
        _ => return Err(invalid_access_schedule("span has an unknown base source")),
    }
    Ok(())
}

fn validate_span_count(
    span: &PostflightAccessSpan,
    register_operand_count: usize,
) -> Result<(), GpuPostflightError> {
    match span.count_source {
        RVR_REPLAY_SPAN_COUNT_FIXED | RVR_REPLAY_SPAN_COUNT_REPLAY_VALUE => {
            if span.count == 0 || span.count_register != 0 || span.count_shift != 0 {
                return Err(invalid_access_schedule(
                    "fixed and replay-value span counts require a nonzero bound",
                ));
            }
        }
        RVR_REPLAY_SPAN_COUNT_REGISTER => {
            if usize::from(span.count_register) >= register_operand_count {
                return Err(invalid_access_schedule(
                    "span count references a missing register operand",
                ));
            }
            if span.count_shift >= u64::BITS as u8 {
                return Err(invalid_access_schedule(
                    "span count shift exceeds the register width",
                ));
            }
        }
        _ => return Err(invalid_access_schedule("span has an unknown count source")),
    }
    Ok(())
}

fn validate_span_address_space(span: &PostflightAccessSpan) -> Result<(), GpuPostflightError> {
    let field_span = matches!(
        span.value_source,
        RVR_REPLAY_SPAN_READ_FIELD32 | RVR_REPLAY_SPAN_WRITE_FIELD32_CANONICAL_PAIRS
    );
    if field_span {
        if span.address_space != DEFERRAL_AS
            || !matches!(
                span.base_source,
                RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT | RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT
            )
            || span.count_source != RVR_REPLAY_SPAN_COUNT_FIXED
            || span.count != RVR_REPLAY_DEFERRAL_DIGEST_BLOCKS
        {
            return Err(invalid_access_schedule(
                "field spans must describe one fixed deferral digest",
            ));
        }
    } else if span.address_space != RV64_MEMORY_AS
        || span.base_source != RVR_REPLAY_SPAN_BASE_REGISTER
    {
        return Err(invalid_access_schedule(
            "u16 spans must use register-based RV64 main-memory addresses",
        ));
    }
    Ok(())
}

fn validate_span_value(
    span: &PostflightAccessSpan,
    static_values_len: usize,
) -> Result<(), GpuPostflightError> {
    if !matches!(
        span.value_source,
        RVR_REPLAY_SPAN_READ_U16
            | RVR_REPLAY_SPAN_WRITE_U16_REPLAY_VALUE
            | RVR_REPLAY_SPAN_WRITE_U16_ZERO
            | RVR_REPLAY_SPAN_READ_FIELD32
            | RVR_REPLAY_SPAN_WRITE_FIELD32_CANONICAL_PAIRS
            | RVR_REPLAY_SPAN_WRITE_U16_STATIC
    ) {
        return Err(invalid_access_schedule("span has an unknown value source"));
    }
    if span.value_source == RVR_REPLAY_SPAN_WRITE_U16_STATIC {
        let static_end = usize::from(span.value_index)
            .checked_add(span.count as usize)
            .filter(|&end| end <= static_values_len);
        if span.count_source != RVR_REPLAY_SPAN_COUNT_FIXED || static_end.is_none() {
            return Err(invalid_access_schedule(
                "static write span exceeds the registered value table",
            ));
        }
    } else if span.value_index != 0 {
        return Err(invalid_access_schedule(
            "non-static span must not reference the static value table",
        ));
    }
    Ok(())
}

fn validate_access_span(
    span: &PostflightAccessSpan,
    schedule: PostflightAccessSchedule<'_>,
    static_values_len: usize,
) -> Result<(), GpuPostflightError> {
    validate_span_base(span, schedule)?;
    validate_span_count(span, schedule.register_operands.len())?;
    validate_span_address_space(span)?;
    validate_span_value(span, static_values_len)?;
    Ok(())
}

fn is_canonical_register_pointer(pointer: u32) -> bool {
    let pointer = u64::from(pointer);
    pointer < RV64_NUM_REGISTERS as u64 * RV64_REGISTER_BYTES
        && pointer.is_multiple_of(RV64_REGISTER_BYTES)
}

fn instruction_operands_match(
    instruction: &GpuReplayInstruction,
    schedule: &RvrReplayAccessSchedule,
    layout: &RvrReplayInstructionLayout,
) -> bool {
    instruction.words[layout.register_as_operand as usize] == RV64_REGISTER_AS
        && instruction.words[layout.memory_as_operand as usize] == RV64_MEMORY_AS
        && !(1..POSTFLIGHT_INSTRUCTION_FIELDS)
            .any(|word| layout.zero_operand_mask & (1 << word) != 0 && instruction.words[word] != 0)
        && schedule
            .register_operands
            .iter()
            .take(schedule.num_register_reads as usize)
            .all(|&word| is_canonical_register_pointer(instruction.words[word as usize]))
        && (schedule.register_write_source == RVR_REPLAY_REGISTER_WRITE_NONE
            || is_canonical_register_pointer(
                instruction.words[schedule.register_write_operand as usize],
            ))
        && (schedule.effect != RVR_REPLAY_EFFECT_BRANCH_REPLAY_VALUE
            || instruction.words[schedule.effect_operand as usize] < BabyBear::ORDER_U32)
}

fn deferral_spans_fit(
    instruction: &GpuReplayInstruction,
    spans: &[PostflightAccessSpan],
    cell_pointer_max_bits: usize,
    deferral_num_cells: usize,
) -> bool {
    let pointer_limit = 1u64 << cell_pointer_max_bits;
    spans
        .iter()
        .filter(|span| {
            matches!(
                span.base_source,
                RVR_REPLAY_SPAN_BASE_DEFERRAL_INPUT | RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT
            )
        })
        .all(|span| {
            let base = u64::from(instruction.words[span.base_index as usize])
                * DEFERRAL_PAIR_STRIDE_CELLS
                + u64::from(span.base_source == RVR_REPLAY_SPAN_BASE_DEFERRAL_OUTPUT)
                    * DEFERRAL_OUTPUT_OFFSET_CELLS;
            let Some(end) = base.checked_add(u64::from(span.count) * BLOCK_FE_WIDTH as u64) else {
                return false;
            };
            end <= pointer_limit && end <= deferral_num_cells as u64
        })
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
        validate_access_schedule(schedule, effect, register_write)?;
        let PostflightAccessSchedule {
            register_operands,
            zero_operand_mask,
            register_as_operand,
            memory_as_operand,
            spans,
        } = schedule;
        for span in spans {
            validate_access_span(span, schedule, self.static_values.len())?;
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
        if self
            .dispatch
            .get(opcode as usize)
            .is_some_and(|&schedule| schedule != RVR_REPLAY_NO_SCHEDULE)
        {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "duplicate checkpoint access schedule for opcode {opcode}"
            )));
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
        let mut operand_words = [0u8; RVR_REPLAY_REGISTER_OPERANDS];
        operand_words[..register_operands.len()].copy_from_slice(register_operands);
        if self.dispatch.len() < dispatch_len {
            self.dispatch.resize(dispatch_len, RVR_REPLAY_NO_SCHEDULE);
        }
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

    pub(super) fn validate_disjoint_opcodes(
        &self,
        native_opcodes: impl IntoIterator<Item = u32>,
    ) -> Result<(), GpuPostflightError> {
        if let Some(opcode) = native_opcodes.into_iter().find(|&opcode| {
            self.dispatch
                .get(opcode as usize)
                .is_some_and(|&schedule| schedule != RVR_REPLAY_NO_SCHEDULE)
        }) {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "opcode {opcode} is owned by both native replay and an extension schedule"
            )));
        }
        Ok(())
    }

    pub(super) fn validate_instruction(
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
        if !instruction_operands_match(instruction, schedule, layout)
            || !deferral_spans_fit(
                instruction,
                schedule_spans,
                cell_pointer_max_bits,
                deferral_num_cells,
            )
        {
            return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                "opcode {} has an instruction incompatible with its access schedule",
                instruction.words[0]
            )));
        }
        Ok(())
    }
}
