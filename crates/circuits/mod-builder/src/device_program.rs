//! Serialization of a finalized field expression for a device-side trace interpreter.
//!
//! This module only defines the pure data transformation. Device allocation, launch policy, and
//! trace-generation ownership remain with the prover extension that consumes the serialized
//! program.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};
use openvm_circuit_primitives::bigint::{
    check_carry_to_zero::get_carry_max_abs_and_bits, OverflowInt,
};

use crate::{FieldExpr, FieldExpressionFiller, FieldExpressionTraceError, SymbolicExpr};

const NO_FLAG: u32 = u32::MAX;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
enum ValueOpcode {
    LoadInput = 0,
    Constant = 1,
    Add = 2,
    Sub = 3,
    Mul = 4,
    Div = 5,
    IntAdd = 6,
    IntMul = 7,
    Select = 8,
    SaveVariable = 9,
    LoadOutput = 10,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
enum LimbOpcode {
    Input = 0,
    Variable = 1,
    Constant = 2,
    Add = 3,
    Sub = 4,
    Mul = 5,
    IntAdd = 6,
    IntMul = 7,
    Select = 8,
}

#[derive(Clone, Copy, Debug, Default)]
struct ValueOp {
    opcode: u32,
    flag: u32,
    /// Execute this operation only when all these flags are set.
    guard_true: u32,
    /// Execute this operation only when all these flags are clear.
    guard_false: u32,
    dst: usize,
    a: usize,
    b: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct ValueGuard {
    required_true: u32,
    required_false: u32,
}

impl ValueGuard {
    fn with_flag(self, flag: usize, value: bool) -> Result<Self, FieldExpressionTraceError> {
        let bit = 1u32
            .checked_shl(
                u32::try_from(flag)
                    .map_err(|_| FieldExpressionTraceError::InvalidFlagIndex(flag))?,
            )
            .ok_or(FieldExpressionTraceError::InvalidFlagIndex(flag))?;
        Ok(if value {
            Self {
                required_true: self.required_true | bit,
                ..self
            }
        } else {
            Self {
                required_false: self.required_false | bit,
                ..self
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LimbOp {
    opcode: u32,
    flag: u32,
    dst_offset: usize,
    dst_len: usize,
    a_offset: usize,
    a_len: usize,
    b_offset: usize,
    b_len: usize,
    immediate: i32,
}

#[derive(Clone, Copy, Debug)]
struct ConstraintMetadata {
    tape_start: usize,
    tape_len: usize,
    result_offset: usize,
    result_len: usize,
    quotient_limbs: usize,
    carry_limbs: usize,
    carry_min_abs: u32,
    carry_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OpcodeMetadata {
    local_opcode: u32,
    flag: u32,
}

#[derive(Clone, Debug)]
struct DeviceFieldExprProgram {
    num_limbs: usize,
    limb_bits: usize,
    u32_limbs: usize,
    num_inputs: usize,
    num_vars: usize,
    num_flags: usize,
    needs_setup: bool,
    should_finalize: bool,
    core_width: usize,
    value_ops: Vec<ValueOp>,
    num_value_slots: usize,
    limb_ops: Vec<LimbOp>,
    limb_scratch_len: usize,
    constraints: Vec<ConstraintMetadata>,
    max_quotient_limbs: usize,
    prime_u32: Vec<u32>,
    montgomery_inverse: u32,
    montgomery_r2: Vec<u32>,
    prime_minus_two: Vec<u32>,
    prime_inverse: Vec<u32>,
    prime_limbs: Vec<i32>,
    montgomery_payload: Vec<u32>,
    constant_limb_payload: Vec<i32>,
    opcode_metadata: Vec<OpcodeMetadata>,
    setup_value_limbs: Vec<u32>,
    output_indices: Vec<u32>,
    dummy_outputs: Vec<u32>,
    aux_words_per_thread: usize,
}

/// A field-expression program ready to upload to a device-side interpreter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SerializedFieldExpr {
    pub blob: Vec<u32>,
    pub core_width: usize,
    pub aux_words_per_thread: usize,
}

struct Serializer<'a> {
    expr: &'a FieldExpr,
    u32_limbs: usize,
    value_ops: Vec<ValueOp>,
    value_slot_base: usize,
    next_value_slot: usize,
    num_value_slots: usize,
    montgomery_payload: Vec<u32>,
    limb_ops: Vec<LimbOp>,
    limb_scratch_cursor: usize,
    limb_scratch_len: usize,
    constant_limb_payload: Vec<i32>,
    montgomery_r: BigUint,
}

impl Serializer<'_> {
    fn montgomery_limbs(&self, value: &BigUint) -> Vec<u32> {
        biguint_to_u32_limbs(
            &((value * &self.montgomery_r) % self.expr.program().prime()),
            self.u32_limbs,
        )
    }

    fn push_montgomery_payload(&mut self, value: &BigUint) -> usize {
        let index = self.montgomery_payload.len() / self.u32_limbs;
        let limbs = self.montgomery_limbs(value);
        self.montgomery_payload.extend(limbs);
        index
    }

    fn immediate_as_field(&self, immediate: isize) -> Result<BigUint, FieldExpressionTraceError> {
        if immediate >= 0 {
            Ok(BigUint::from(immediate as u64) % self.expr.program().prime())
        } else {
            let magnitude = BigUint::from(immediate.unsigned_abs() as u64);
            if &magnitude > self.expr.program().prime() {
                return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                    "negative field-expression immediate exceeds the modulus",
                ));
            }
            Ok(self.expr.program().prime() - magnitude)
        }
    }

    fn allocate_value_slot(&mut self) -> Result<usize, FieldExpressionTraceError> {
        let slot = self.next_value_slot;
        self.next_value_slot = self
            .next_value_slot
            .checked_add(1)
            .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
        self.num_value_slots = self.num_value_slots.max(self.next_value_slot);
        Ok(slot)
    }

    fn emit_value(
        &mut self,
        expr: &SymbolicExpr,
        guard: ValueGuard,
    ) -> Result<usize, FieldExpressionTraceError> {
        match expr {
            SymbolicExpr::Input(index) => {
                if *index >= self.expr.program().num_inputs() {
                    return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                        "field-expression input index is out of bounds",
                    ));
                }
                Ok(*index)
            }
            SymbolicExpr::Var(index) => {
                if *index >= self.expr.program().num_vars() {
                    return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                        "field-expression variable index is out of bounds",
                    ));
                }
                let slot = self
                    .expr
                    .program()
                    .num_inputs()
                    .checked_add(*index)
                    .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
                Ok(slot)
            }
            SymbolicExpr::Const(index, value, expr_limbs) => {
                let Some((indexed_value, indexed_limbs)) =
                    self.expr.program().builder().constants.get(*index)
                else {
                    return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                        "field-expression constant index is out of bounds",
                    ));
                };
                if indexed_value != value
                    || indexed_limbs.len() != *expr_limbs
                    || value >= self.expr.program().prime()
                {
                    return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                        "field-expression constant metadata is inconsistent",
                    ));
                }
                let payload = self.push_montgomery_payload(value);
                let dst = self.allocate_value_slot()?;
                self.value_ops.push(ValueOp {
                    opcode: ValueOpcode::Constant as u32,
                    guard_true: guard.required_true,
                    guard_false: guard.required_false,
                    dst,
                    a: payload,
                    ..Default::default()
                });
                Ok(dst)
            }
            SymbolicExpr::Add(lhs, rhs)
            | SymbolicExpr::Sub(lhs, rhs)
            | SymbolicExpr::Mul(lhs, rhs)
            | SymbolicExpr::Div(lhs, rhs) => {
                let a = self.emit_value(lhs, guard)?;
                let b = self.emit_value(rhs, guard)?;
                let opcode = match expr {
                    SymbolicExpr::Add(..) => ValueOpcode::Add,
                    SymbolicExpr::Sub(..) => ValueOpcode::Sub,
                    SymbolicExpr::Mul(..) => ValueOpcode::Mul,
                    SymbolicExpr::Div(..) => ValueOpcode::Div,
                    _ => unreachable!(),
                };
                let dst = self.allocate_value_slot()?;
                self.value_ops.push(ValueOp {
                    opcode: opcode as u32,
                    guard_true: guard.required_true,
                    guard_false: guard.required_false,
                    dst,
                    a,
                    b,
                    ..Default::default()
                });
                Ok(dst)
            }
            SymbolicExpr::IntAdd(lhs, immediate) | SymbolicExpr::IntMul(lhs, immediate) => {
                let a = self.emit_value(lhs, guard)?;
                let immediate = self.immediate_as_field(*immediate)?;
                let payload = self.push_montgomery_payload(&immediate);
                let opcode = if matches!(expr, SymbolicExpr::IntAdd(..)) {
                    ValueOpcode::IntAdd
                } else {
                    ValueOpcode::IntMul
                };
                let dst = self.allocate_value_slot()?;
                self.value_ops.push(ValueOp {
                    opcode: opcode as u32,
                    guard_true: guard.required_true,
                    guard_false: guard.required_false,
                    dst,
                    a,
                    b: payload,
                    ..Default::default()
                });
                Ok(dst)
            }
            SymbolicExpr::Select(flag, lhs, rhs) => {
                if *flag >= self.expr.program().num_flags() {
                    return Err(FieldExpressionTraceError::InvalidFlagIndex(*flag));
                }
                let a = self.emit_value(lhs, guard.with_flag(*flag, true)?)?;
                let b = self.emit_value(rhs, guard.with_flag(*flag, false)?)?;
                let dst = self.allocate_value_slot()?;
                self.value_ops.push(ValueOp {
                    opcode: ValueOpcode::Select as u32,
                    flag: to_u32(*flag)?,
                    guard_true: guard.required_true,
                    guard_false: guard.required_false,
                    dst,
                    a,
                    b,
                });
                Ok(dst)
            }
        }
    }

    fn allocate_limb_scratch(&mut self, len: usize) -> Result<usize, FieldExpressionTraceError> {
        let offset = self.limb_scratch_cursor;
        self.limb_scratch_cursor = self
            .limb_scratch_cursor
            .checked_add(len)
            .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
        self.limb_scratch_len = self.limb_scratch_len.max(self.limb_scratch_cursor);
        Ok(offset)
    }

    fn emit_limbs(
        &mut self,
        expr: &SymbolicExpr,
    ) -> Result<(usize, usize), FieldExpressionTraceError> {
        let program = self.expr.program();
        let builder = program.builder();
        if expr.constraint_limb_max_abs(builder.limb_bits, builder.num_limbs) > i32::MAX as usize {
            return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                "an intermediate constraint limb exceeds the device interpreter's i32 range",
            ));
        }
        let num_limbs = program.canonical_num_limbs();
        match expr {
            SymbolicExpr::Input(index) => {
                if *index >= program.num_inputs() {
                    return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                        "field-expression input index is out of bounds",
                    ));
                }
                let offset = self.allocate_limb_scratch(num_limbs)?;
                self.limb_ops.push(LimbOp {
                    opcode: LimbOpcode::Input as u32,
                    dst_offset: offset,
                    dst_len: num_limbs,
                    a_offset: *index,
                    ..Default::default()
                });
                Ok((offset, num_limbs))
            }
            SymbolicExpr::Var(index) => {
                if *index >= program.num_vars() {
                    return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                        "field-expression variable index is out of bounds",
                    ));
                }
                let offset = self.allocate_limb_scratch(num_limbs)?;
                self.limb_ops.push(LimbOp {
                    opcode: LimbOpcode::Variable as u32,
                    dst_offset: offset,
                    dst_len: num_limbs,
                    a_offset: *index,
                    ..Default::default()
                });
                Ok((offset, num_limbs))
            }
            SymbolicExpr::Const(index, value, expr_limbs) => {
                let Some((indexed_value, limbs)) = builder.constants.get(*index) else {
                    return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                        "field-expression constant index is out of bounds",
                    ));
                };
                if indexed_value != value || limbs.len() != *expr_limbs || value >= program.prime()
                {
                    return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
                        "field-expression constant metadata is inconsistent",
                    ));
                }
                let payload_offset = self.constant_limb_payload.len();
                self.constant_limb_payload.extend(
                    limbs
                        .iter()
                        .copied()
                        .map(i32::try_from)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| FieldExpressionTraceError::ProgramTooLarge)?,
                );
                let len = *expr_limbs;
                let offset = self.allocate_limb_scratch(len)?;
                self.limb_ops.push(LimbOp {
                    opcode: LimbOpcode::Constant as u32,
                    dst_offset: offset,
                    dst_len: len,
                    a_offset: payload_offset,
                    ..Default::default()
                });
                Ok((offset, len))
            }
            SymbolicExpr::Add(lhs, rhs) | SymbolicExpr::Sub(lhs, rhs) => {
                let (a_offset, a_len) = self.emit_limbs(lhs)?;
                let (b_offset, b_len) = self.emit_limbs(rhs)?;
                let len = a_len.max(b_len);
                let offset = self.allocate_limb_scratch(len)?;
                let opcode = if matches!(expr, SymbolicExpr::Add(..)) {
                    LimbOpcode::Add
                } else {
                    LimbOpcode::Sub
                };
                self.limb_ops.push(LimbOp {
                    opcode: opcode as u32,
                    dst_offset: offset,
                    dst_len: len,
                    a_offset,
                    a_len,
                    b_offset,
                    b_len,
                    ..Default::default()
                });
                Ok((offset, len))
            }
            SymbolicExpr::Mul(lhs, rhs) => {
                let (a_offset, a_len) = self.emit_limbs(lhs)?;
                let (b_offset, b_len) = self.emit_limbs(rhs)?;
                let len = a_len
                    .checked_add(b_len)
                    .and_then(|sum| sum.checked_sub(1))
                    .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
                let offset = self.allocate_limb_scratch(len)?;
                self.limb_ops.push(LimbOp {
                    opcode: LimbOpcode::Mul as u32,
                    dst_offset: offset,
                    dst_len: len,
                    a_offset,
                    a_len,
                    b_offset,
                    b_len,
                    ..Default::default()
                });
                Ok((offset, len))
            }
            SymbolicExpr::IntAdd(lhs, immediate) | SymbolicExpr::IntMul(lhs, immediate) => {
                let (a_offset, a_len) = self.emit_limbs(lhs)?;
                let offset = self.allocate_limb_scratch(a_len)?;
                let opcode = if matches!(expr, SymbolicExpr::IntAdd(..)) {
                    LimbOpcode::IntAdd
                } else {
                    LimbOpcode::IntMul
                };
                self.limb_ops.push(LimbOp {
                    opcode: opcode as u32,
                    dst_offset: offset,
                    dst_len: a_len,
                    a_offset,
                    a_len,
                    immediate: i32::try_from(*immediate)
                        .map_err(|_| FieldExpressionTraceError::ProgramTooLarge)?,
                    ..Default::default()
                });
                Ok((offset, a_len))
            }
            SymbolicExpr::Select(flag, lhs, rhs) => {
                if *flag >= program.num_flags() {
                    return Err(FieldExpressionTraceError::InvalidFlagIndex(*flag));
                }
                let (a_offset, a_len) = self.emit_limbs(lhs)?;
                let (b_offset, b_len) = self.emit_limbs(rhs)?;
                let len = a_len.max(b_len);
                let offset = self.allocate_limb_scratch(len)?;
                self.limb_ops.push(LimbOp {
                    opcode: LimbOpcode::Select as u32,
                    flag: to_u32(*flag)?,
                    dst_offset: offset,
                    dst_len: len,
                    a_offset,
                    a_len,
                    b_offset,
                    b_len,
                    ..Default::default()
                });
                Ok((offset, len))
            }
            SymbolicExpr::Div(..) => {
                unreachable!("division is not permitted in field-expression constraints")
            }
        }
    }
}

/// Serializes the expression and already-normalized opcode metadata held by `filler`.
pub fn serialize_field_expr<A>(
    filler: &FieldExpressionFiller<A>,
) -> Result<SerializedFieldExpr, FieldExpressionTraceError> {
    let program = build_device_program(filler)?;
    let blob = program.to_blob()?;
    Ok(SerializedFieldExpr {
        blob,
        core_width: program.core_width,
        aux_words_per_thread: program.aux_words_per_thread,
    })
}

fn build_device_program<A>(
    filler: &FieldExpressionFiller<A>,
) -> Result<DeviceFieldExprProgram, FieldExpressionTraceError> {
    build_device_program_inner(
        &filler.expr,
        &filler.local_opcode_idx,
        &filler.opcode_flag_idx,
        filler.should_finalize,
    )
}

fn build_device_program_inner(
    expr: &FieldExpr,
    local_opcode_idx: &[usize],
    opcode_flag_idx: &[usize],
    should_finalize: bool,
) -> Result<DeviceFieldExprProgram, FieldExpressionTraceError> {
    let builder = expr.program().builder();
    let opcode_metadata = normalize_opcode_metadata(
        expr.program().needs_setup(),
        expr.program().num_flags(),
        local_opcode_idx,
        opcode_flag_idx,
    )?;

    let bit_width = builder
        .num_limbs
        .checked_mul(builder.limb_bits)
        .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
    let u32_limbs = bit_width.div_ceil(32);
    if builder.limb_bits != 8 {
        return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
            "device field expressions require 8-bit limbs",
        ));
    }
    if builder.num_flags > 32 {
        return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
            "device field expressions support at most 32 flags",
        ));
    }
    if u32_limbs == 0 || u32_limbs > 12 {
        return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
            "device field expressions support 1 through 12 u32 limbs",
        ));
    }
    if builder.num_variables != builder.computes.len()
        || builder.num_variables != builder.constraints.len()
        || builder.num_variables != builder.q_limbs.len()
        || builder.num_variables != builder.carry_limbs.len()
    {
        return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
            "field-expression variable metadata is inconsistent",
        ));
    }
    if builder.prime <= BigUint::from(2u32)
        || !builder.prime.bit(0)
        || builder.prime.bits()
            > u64::try_from(bit_width).map_err(|_| FieldExpressionTraceError::ProgramTooLarge)?
    {
        return Err(FieldExpressionTraceError::UnsupportedDeviceProgram(
            "device field expressions require an odd modulus fitting the declared limb width",
        ));
    }
    let montgomery_r = (BigUint::one() << checked_mul(32, u32_limbs)?) % &builder.prime;
    let mut output_positions = vec![None; builder.num_variables];
    let output_indices = expr
        .program()
        .output_indices()
        .iter()
        .copied()
        .enumerate()
        .map(|(position, index)| {
            if index >= builder.num_variables {
                Err(FieldExpressionTraceError::InvalidProgramOutput(index))
            } else {
                output_positions[index].get_or_insert(position);
                to_u32(index)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let value_slot_base = builder
        .num_input
        .checked_add(builder.num_variables)
        .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
    let mut serializer = Serializer {
        expr,
        u32_limbs,
        value_ops: Vec::new(),
        value_slot_base,
        next_value_slot: value_slot_base,
        num_value_slots: value_slot_base,
        montgomery_payload: Vec::new(),
        limb_ops: Vec::new(),
        limb_scratch_cursor: 0,
        limb_scratch_len: 0,
        constant_limb_payload: Vec::new(),
        montgomery_r: montgomery_r.clone(),
    };

    for input in 0..builder.num_input {
        serializer.value_ops.push(ValueOp {
            opcode: ValueOpcode::LoadInput as u32,
            dst: input,
            a: input,
            ..Default::default()
        });
    }
    for (variable, compute) in builder.computes.iter().enumerate() {
        serializer.next_value_slot = serializer.value_slot_base;
        // Output variables are already present in the read-only execution transcript.
        // Load them directly; the constraint tape below validates their relation to the inputs.
        let source = if let Some(output_position) = output_positions[variable] {
            let source = serializer.allocate_value_slot()?;
            serializer.value_ops.push(ValueOp {
                opcode: ValueOpcode::LoadOutput as u32,
                dst: source,
                a: output_position,
                ..Default::default()
            });
            source
        } else {
            serializer.emit_value(compute, ValueGuard::default())?
        };
        let slot = builder
            .num_input
            .checked_add(variable)
            .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
        serializer.value_ops.push(ValueOp {
            opcode: ValueOpcode::SaveVariable as u32,
            dst: slot,
            a: variable,
            b: source,
            ..Default::default()
        });
    }

    let zero_inputs = (0..builder.num_input)
        .map(|_| {
            OverflowInt::<isize>::from_biguint(
                &BigUint::zero(),
                builder.limb_bits,
                Some(builder.num_limbs),
            )
        })
        .collect::<Vec<_>>();
    let zero_vars = (0..builder.num_variables)
        .map(|_| {
            OverflowInt::<isize>::from_biguint(
                &BigUint::zero(),
                builder.limb_bits,
                Some(builder.num_limbs),
            )
        })
        .collect::<Vec<_>>();
    let zero_constants = builder
        .constants
        .iter()
        .map(|(_, limbs)| {
            OverflowInt::<isize>::from_unsigned_limbs(vec![0; limbs.len()], builder.limb_bits)
        })
        .collect::<Vec<_>>();
    let flags = vec![false; builder.num_flags];
    let prime_overflow =
        OverflowInt::<isize>::from_biguint(&builder.prime, builder.limb_bits, None);
    let mut constraints = Vec::with_capacity(builder.constraints.len());
    for (index, constraint) in builder.constraints.iter().enumerate() {
        serializer.limb_scratch_cursor = 0;
        let tape_start = serializer.limb_ops.len();
        let (result_offset, result_len) = serializer.emit_limbs(constraint)?;
        let tape_len = serializer
            .limb_ops
            .len()
            .checked_sub(tape_start)
            .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;

        let expression_bound =
            constraint.evaluate_overflow_isize(&zero_inputs, &zero_vars, &zero_constants, &flags);
        let quotient_bound = OverflowInt::<isize>::from_signed_limbs(
            vec![0; builder.q_limbs[index]],
            builder.limb_bits,
        );
        let total = expression_bound - quotient_bound * prime_overflow.clone();
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(total.max_overflow_bits(), builder.limb_bits);
        if total.num_limbs() != builder.carry_limbs[index] {
            return Err(FieldExpressionTraceError::ProgramTooLarge);
        }
        let quotient_bits = checked_mul(builder.q_limbs[index], builder.limb_bits)?
            .checked_add(1)
            .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
        if quotient_bits >= checked_mul(64, u32_limbs)? {
            return Err(FieldExpressionTraceError::ProgramTooLarge);
        }
        constraints.push(ConstraintMetadata {
            tape_start,
            tape_len,
            result_offset,
            result_len,
            quotient_limbs: builder.q_limbs[index],
            carry_limbs: builder.carry_limbs[index],
            carry_min_abs: to_u32(carry_min_abs)?,
            carry_bits: to_u32(carry_bits)?,
        });
    }

    let max_quotient_limbs = builder.q_limbs.iter().copied().max().unwrap_or(0);
    let aux_words_per_thread = aux_words_per_thread(
        builder.num_variables,
        u32_limbs,
        serializer.num_value_slots,
        serializer.limb_scratch_len,
        max_quotient_limbs,
    )?;
    let prime_u32 = biguint_to_u32_limbs(&builder.prime, u32_limbs);
    let mut inverse = 1u32;
    for _ in 0..5 {
        inverse = inverse.wrapping_mul(2u32.wrapping_sub(prime_u32[0].wrapping_mul(inverse)));
    }
    let montgomery_inverse = inverse.wrapping_neg();
    let montgomery_r2 = biguint_to_u32_limbs(
        &((&montgomery_r * &montgomery_r) % &builder.prime),
        u32_limbs,
    );
    let prime_minus_two = biguint_to_u32_limbs(&(&builder.prime - BigUint::from(2u32)), u32_limbs);
    let wide_bits = checked_mul(64, u32_limbs)?;
    let wide_modulus = BigInt::one() << wide_bits;
    let prime_inverse = BigInt::from_biguint(Sign::Plus, builder.prime.clone())
        .modinv(&wide_modulus)
        .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
    let (_, mut prime_inverse) = prime_inverse.to_u32_digits();
    prime_inverse.resize(checked_mul(2, u32_limbs)?, 0);
    let prime_limbs = builder
        .prime_limbs
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| FieldExpressionTraceError::ProgramTooLarge)?;
    let setup_value_limbs = fixed_byte_limbs(expr.program().setup_values(), builder.num_limbs)?;
    let dummy_outputs = if should_finalize {
        let inputs = vec![BigUint::zero(); builder.num_input];
        let flags = vec![false; builder.num_flags];
        let mut variables = vec![BigUint::zero(); builder.num_variables];
        for (index, compute) in builder.computes.iter().enumerate() {
            variables[index] = compute.compute(&inputs, &variables, &flags, &builder.prime);
        }
        output_indices
            .iter()
            .flat_map(|&index| biguint_to_u32_limbs(&variables[index as usize], u32_limbs))
            .collect()
    } else {
        Vec::new()
    };
    Ok(DeviceFieldExprProgram {
        num_limbs: builder.num_limbs,
        limb_bits: builder.limb_bits,
        u32_limbs,
        num_inputs: builder.num_input,
        num_vars: builder.num_variables,
        num_flags: builder.num_flags,
        needs_setup: builder.needs_setup(),
        should_finalize,
        core_width: expr.program().width(),
        value_ops: serializer.value_ops,
        num_value_slots: serializer.num_value_slots,
        limb_ops: serializer.limb_ops,
        limb_scratch_len: serializer.limb_scratch_len,
        constraints,
        max_quotient_limbs,
        prime_u32,
        montgomery_inverse,
        montgomery_r2,
        prime_minus_two,
        prime_inverse,
        prime_limbs,
        montgomery_payload: serializer.montgomery_payload,
        constant_limb_payload: serializer.constant_limb_payload,
        opcode_metadata,
        setup_value_limbs,
        output_indices,
        dummy_outputs,
        aux_words_per_thread,
    })
}

fn normalize_opcode_metadata(
    needs_setup: bool,
    num_flags: usize,
    local_opcodes: &[usize],
    opcode_flags: &[usize],
) -> Result<Vec<OpcodeMetadata>, FieldExpressionTraceError> {
    let valid_shape = if needs_setup {
        !local_opcodes.is_empty() && opcode_flags.len() + 1 == local_opcodes.len()
    } else {
        local_opcodes.len() == 1 && opcode_flags.is_empty() && num_flags == 0
    };
    if !valid_shape
        || local_opcodes
            .iter()
            .enumerate()
            .any(|(index, opcode)| local_opcodes[..index].contains(opcode))
        || opcode_flags.iter().any(|flag| *flag >= num_flags)
        || opcode_flags
            .iter()
            .enumerate()
            .any(|(index, flag)| opcode_flags[..index].contains(flag))
    {
        return Err(FieldExpressionTraceError::InvalidFlagLayout);
    }

    local_opcodes
        .iter()
        .enumerate()
        .map(|(index, opcode)| {
            Ok(OpcodeMetadata {
                local_opcode: to_u32(*opcode)?,
                flag: opcode_flags
                    .get(index)
                    .copied()
                    .map(to_u32)
                    .transpose()?
                    .unwrap_or(NO_FLAG),
            })
        })
        .collect()
}

fn fixed_byte_limbs(
    values: &[BigUint],
    num_limbs: usize,
) -> Result<Vec<u32>, FieldExpressionTraceError> {
    let capacity = values
        .len()
        .checked_mul(num_limbs)
        .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
    let mut result = Vec::with_capacity(capacity);
    for value in values {
        let bytes = value.to_bytes_le();
        if bytes.len() > num_limbs {
            return Err(FieldExpressionTraceError::ProgramTooLarge);
        }
        result.extend(bytes.iter().copied().map(u32::from));
        result.resize(
            result
                .len()
                .checked_add(num_limbs - bytes.len())
                .ok_or(FieldExpressionTraceError::ProgramTooLarge)?,
            0,
        );
    }
    Ok(result)
}

fn aux_words_per_thread(
    num_vars: usize,
    u32_limbs: usize,
    num_value_slots: usize,
    limb_scratch_len: usize,
    max_quotient_limbs: usize,
) -> Result<usize, FieldExpressionTraceError> {
    let persistent = checked_mul(num_vars, u32_limbs)?;
    let value_workspace = checked_mul(num_value_slots, u32_limbs)?
        .checked_add(checked_mul(3, u32_limbs)?)
        .and_then(|value| value.checked_add(2))
        .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
    let constraint_workspace = limb_scratch_len
        .checked_add(checked_mul(4, u32_limbs)?)
        .and_then(|value| value.checked_add(max_quotient_limbs))
        .ok_or(FieldExpressionTraceError::ProgramTooLarge)?;
    persistent
        .checked_add(value_workspace.max(constraint_workspace))
        .ok_or(FieldExpressionTraceError::ProgramTooLarge)
}

fn biguint_to_u32_limbs(value: &BigUint, len: usize) -> Vec<u32> {
    let mut limbs = value.to_u32_digits();
    assert!(limbs.len() <= len, "field value exceeds its declared width");
    limbs.resize(len, 0);
    limbs
}

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize, FieldExpressionTraceError> {
    lhs.checked_mul(rhs)
        .ok_or(FieldExpressionTraceError::ProgramTooLarge)
}

fn to_u32(value: usize) -> Result<u32, FieldExpressionTraceError> {
    u32::try_from(value).map_err(|_| FieldExpressionTraceError::ProgramTooLarge)
}

// Header words. This is an internal host/device ABI, not a persistent file format.
const HEADER_WORDS: usize = 34;
const H_NUM_LIMBS: usize = 0;
const H_LIMB_BITS: usize = 1;
const H_U32_LIMBS: usize = 2;
const H_NUM_INPUTS: usize = 3;
const H_NUM_VARS: usize = 4;
const H_NUM_FLAGS: usize = 5;
const H_NEEDS_SETUP: usize = 6;
const H_SHOULD_FINALIZE: usize = 7;
const H_CORE_WIDTH: usize = 8;
const H_NUM_VALUE_SLOTS: usize = 9;
const H_NUM_VALUE_OPS: usize = 10;
const H_NUM_LIMB_OPS: usize = 11;
const H_NUM_CONSTRAINTS: usize = 12;
const H_LIMB_SCRATCH_LEN: usize = 13;
const H_PRIME_LIMBS_LEN: usize = 14;
const H_NUM_OPCODE_METADATA: usize = 15;
const H_NUM_SETUP_VALUES: usize = 16;
const H_NUM_OUTPUTS: usize = 17;
const H_MAX_QUOTIENT_LIMBS: usize = 18;
const H_AUX_WORDS_PER_THREAD: usize = 19;
const H_VALUE_OPS_OFFSET: usize = 20;
const H_LIMB_OPS_OFFSET: usize = 21;
const H_CONSTRAINTS_OFFSET: usize = 22;
const H_PRIME_U32_OFFSET: usize = 23;
const H_MONTGOMERY_R2_OFFSET: usize = 24;
const H_PRIME_MINUS_TWO_OFFSET: usize = 25;
const H_PRIME_INVERSE_OFFSET: usize = 26;
const H_PRIME_LIMBS_OFFSET: usize = 27;
const H_MONTGOMERY_PAYLOAD_OFFSET: usize = 28;
const H_CONSTANT_LIMB_PAYLOAD_OFFSET: usize = 29;
const H_OPCODE_METADATA_OFFSET: usize = 30;
const H_SETUP_VALUES_OFFSET: usize = 31;
const H_OUTPUT_INDICES_OFFSET: usize = 32;
const H_MONTGOMERY_INVERSE: usize = 33;

impl DeviceFieldExprProgram {
    fn to_blob(&self) -> Result<Vec<u32>, FieldExpressionTraceError> {
        let mut blob = vec![0; HEADER_WORDS];
        blob[H_NUM_LIMBS] = to_u32(self.num_limbs)?;
        blob[H_LIMB_BITS] = to_u32(self.limb_bits)?;
        blob[H_U32_LIMBS] = to_u32(self.u32_limbs)?;
        blob[H_NUM_INPUTS] = to_u32(self.num_inputs)?;
        blob[H_NUM_VARS] = to_u32(self.num_vars)?;
        blob[H_NUM_FLAGS] = to_u32(self.num_flags)?;
        blob[H_NEEDS_SETUP] = self.needs_setup as u32;
        blob[H_SHOULD_FINALIZE] = self.should_finalize as u32;
        blob[H_CORE_WIDTH] = to_u32(self.core_width)?;
        blob[H_NUM_VALUE_SLOTS] = to_u32(self.num_value_slots)?;
        blob[H_NUM_VALUE_OPS] = to_u32(self.value_ops.len())?;
        blob[H_NUM_LIMB_OPS] = to_u32(self.limb_ops.len())?;
        blob[H_NUM_CONSTRAINTS] = to_u32(self.constraints.len())?;
        blob[H_LIMB_SCRATCH_LEN] = to_u32(self.limb_scratch_len)?;
        blob[H_PRIME_LIMBS_LEN] = to_u32(self.prime_limbs.len())?;
        blob[H_NUM_OPCODE_METADATA] = to_u32(self.opcode_metadata.len())?;
        blob[H_NUM_SETUP_VALUES] = to_u32(self.setup_value_limbs.len() / self.num_limbs)?;
        blob[H_NUM_OUTPUTS] = to_u32(self.output_indices.len())?;
        blob[H_MAX_QUOTIENT_LIMBS] = to_u32(self.max_quotient_limbs)?;
        blob[H_AUX_WORDS_PER_THREAD] = to_u32(self.aux_words_per_thread)?;

        blob[H_VALUE_OPS_OFFSET] = to_u32(blob.len())?;
        for op in &self.value_ops {
            blob.extend([
                op.opcode,
                op.flag,
                op.guard_true,
                op.guard_false,
                to_u32(op.dst)?,
                to_u32(op.a)?,
                to_u32(op.b)?,
            ]);
        }
        blob[H_LIMB_OPS_OFFSET] = to_u32(blob.len())?;
        for op in &self.limb_ops {
            blob.extend([
                op.opcode,
                op.flag,
                to_u32(op.dst_offset)?,
                to_u32(op.dst_len)?,
                to_u32(op.a_offset)?,
                to_u32(op.a_len)?,
                to_u32(op.b_offset)?,
                to_u32(op.b_len)?,
                op.immediate as u32,
            ]);
        }
        blob[H_CONSTRAINTS_OFFSET] = to_u32(blob.len())?;
        for constraint in &self.constraints {
            blob.extend([
                to_u32(constraint.tape_start)?,
                to_u32(constraint.tape_len)?,
                to_u32(constraint.result_offset)?,
                to_u32(constraint.result_len)?,
                to_u32(constraint.quotient_limbs)?,
                to_u32(constraint.carry_limbs)?,
                constraint.carry_min_abs,
                constraint.carry_bits,
            ]);
        }
        blob[H_PRIME_U32_OFFSET] = to_u32(blob.len())?;
        blob.extend(&self.prime_u32);
        blob[H_MONTGOMERY_R2_OFFSET] = to_u32(blob.len())?;
        blob.extend(&self.montgomery_r2);
        blob[H_PRIME_MINUS_TWO_OFFSET] = to_u32(blob.len())?;
        blob.extend(&self.prime_minus_two);
        blob[H_PRIME_INVERSE_OFFSET] = to_u32(blob.len())?;
        blob.extend(&self.prime_inverse);
        blob[H_PRIME_LIMBS_OFFSET] = to_u32(blob.len())?;
        blob.extend(self.prime_limbs.iter().map(|value| *value as u32));
        blob[H_MONTGOMERY_PAYLOAD_OFFSET] = to_u32(blob.len())?;
        blob.extend(&self.montgomery_payload);
        blob[H_CONSTANT_LIMB_PAYLOAD_OFFSET] = to_u32(blob.len())?;
        blob.extend(self.constant_limb_payload.iter().map(|value| *value as u32));
        blob[H_OPCODE_METADATA_OFFSET] = to_u32(blob.len())?;
        for metadata in &self.opcode_metadata {
            blob.extend([metadata.local_opcode, metadata.flag]);
        }
        blob[H_SETUP_VALUES_OFFSET] = to_u32(blob.len())?;
        blob.extend(&self.setup_value_limbs);
        blob[H_OUTPUT_INDICES_OFFSET] = to_u32(blob.len())?;
        blob.extend(&self.output_indices);
        blob.extend(&self.dummy_outputs);
        blob[H_MONTGOMERY_INVERSE] = self.montgomery_inverse;
        Ok(blob)
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use openvm_circuit_primitives::bigint::utils::secp256k1_coord_prime;

    use super::{
        biguint_to_u32_limbs, build_device_program, serialize_field_expr, ValueOpcode,
        H_AUX_WORDS_PER_THREAD, H_CORE_WIDTH, H_MAX_QUOTIENT_LIMBS, H_NUM_OPCODE_METADATA,
        H_NUM_OUTPUTS, H_NUM_SETUP_VALUES, H_OPCODE_METADATA_OFFSET, H_OUTPUT_INDICES_OFFSET,
        H_SETUP_VALUES_OFFSET, NO_FLAG,
    };
    use crate::{
        test_utils::setup, ExprBuilder, FieldExpr, FieldExpressionFiller, FieldExpressionProgram,
        FieldExpressionTraceError, FieldVariable, SymbolicExpr,
    };

    fn setup_filler(needs_setup: bool, setup_values: Vec<BigUint>) -> FieldExpressionFiller<()> {
        let prime = secp256k1_coord_prime();
        let (range_checker, builder) = setup(&prime);
        let lhs = ExprBuilder::new_input(builder.clone());
        let rhs = ExprBuilder::new_input(builder.clone());
        let mut sum = lhs + rhs;
        sum.save_output();
        let program = FieldExpressionProgram::new_with_setup_values(
            builder.borrow().clone(),
            needs_setup,
            setup_values,
        );
        let expr = FieldExpr::new(program, range_checker.bus());
        FieldExpressionFiller::new(
            (),
            expr,
            if needs_setup { vec![7, 9] } else { vec![7] },
            Vec::new(),
            range_checker,
            true,
        )
    }

    #[test]
    fn serializes_normalized_opcode_setup_and_output_metadata() {
        let filler = setup_filler(true, vec![BigUint::from(123u32)]);
        let serialized = serialize_field_expr(&filler).unwrap();
        let blob = &serialized.blob;

        assert_eq!(blob[H_CORE_WIDTH] as usize, serialized.core_width);
        assert_eq!(
            blob[H_AUX_WORDS_PER_THREAD] as usize,
            serialized.aux_words_per_thread
        );
        assert_eq!(blob[H_NUM_OPCODE_METADATA], 2);
        let opcode_offset = blob[H_OPCODE_METADATA_OFFSET] as usize;
        assert_eq!(&blob[opcode_offset..opcode_offset + 4], &[7, 0, 9, NO_FLAG]);

        assert_eq!(blob[H_NUM_SETUP_VALUES], 1);
        let setup_offset = blob[H_SETUP_VALUES_OFFSET] as usize;
        assert_eq!(blob[setup_offset], 123);
        assert!(blob[setup_offset + 1..setup_offset + 32]
            .iter()
            .all(|word| *word == 0));

        assert_eq!(blob[H_NUM_OUTPUTS], 1);
        let output_offset = blob[H_OUTPUT_INDICES_OFFSET] as usize;
        assert_eq!(blob[output_offset], 0);
        assert!(blob[H_MAX_QUOTIENT_LIMBS] > 0);
    }

    #[test]
    fn serializes_no_setup_opcode_without_a_flag() {
        let filler = setup_filler(false, Vec::new());
        let serialized = serialize_field_expr(&filler).unwrap();
        let offset = serialized.blob[H_OPCODE_METADATA_OFFSET] as usize;
        assert_eq!(&serialized.blob[offset..offset + 2], &[7, NO_FLAG]);
    }

    #[test]
    fn rejects_mutated_flag_metadata() {
        let mut filler = setup_filler(true, Vec::new());
        filler.opcode_flag_idx[0] = 1;
        assert_eq!(
            serialize_field_expr(&filler),
            Err(FieldExpressionTraceError::InvalidFlagLayout)
        );
    }

    #[test]
    fn loads_output_without_replaying_its_division() {
        let prime = secp256k1_coord_prime();
        let (range_checker, builder) = setup(&prime);
        let lhs = ExprBuilder::new_input(builder.clone());
        let rhs = ExprBuilder::new_input(builder.clone());
        let (output_index, output) = builder.borrow_mut().new_var();
        let mut output = FieldVariable::from_var(builder.clone(), output);
        let mul_flag = builder.borrow_mut().new_flag();
        let div_flag = builder.borrow_mut().new_flag();
        builder
            .borrow_mut()
            .set_constraint(output_index, output.expr.clone() - lhs.expr.clone());
        builder.borrow_mut().set_compute(
            output_index,
            SymbolicExpr::Select(
                mul_flag,
                Box::new(lhs.expr.clone() * rhs.expr.clone()),
                Box::new(SymbolicExpr::Select(
                    div_flag,
                    Box::new(lhs.expr.clone() / rhs.expr.clone()),
                    Box::new(lhs.expr),
                )),
            ),
        );
        output.save_output();
        let program = FieldExpressionProgram::new(builder.borrow().clone(), true);
        let expr = FieldExpr::new(program, range_checker.bus());
        let filler = FieldExpressionFiller::new(
            (),
            expr,
            vec![2, 3, 4],
            vec![mul_flag, div_flag],
            range_checker,
            true,
        );

        let program = build_device_program(&filler).unwrap();
        assert!(program
            .value_ops
            .iter()
            .all(|op| op.opcode != ValueOpcode::Div as u32));
        let load = program
            .value_ops
            .iter()
            .find(|op| op.opcode == ValueOpcode::LoadOutput as u32)
            .unwrap();
        assert_eq!(load.a, 0);
    }

    #[test]
    fn serializes_outputs_for_finalized_dummy_rows() {
        let prime = secp256k1_coord_prime();
        let (range_checker, builder) = setup(&prime);
        let input = ExprBuilder::new_input(builder.clone());
        let constant = ExprBuilder::new_const(builder.clone(), BigUint::from(7u32));
        let mut output = input + constant;
        output.save_output();
        let program = FieldExpressionProgram::new(builder.borrow().clone(), false);
        let expr = FieldExpr::new(program, range_checker.bus());
        let filler = FieldExpressionFiller::new((), expr, vec![7], Vec::new(), range_checker, true);

        let program = build_device_program(&filler).unwrap();
        assert_eq!(
            program.dummy_outputs,
            biguint_to_u32_limbs(&BigUint::from(7u32), program.u32_limbs)
        );
    }

    #[test]
    fn preserves_unreachable_nested_select_guards() {
        let prime = secp256k1_coord_prime();
        let (range_checker, builder) = setup(&prime);
        let lhs = ExprBuilder::new_input(builder.clone());
        let rhs = ExprBuilder::new_input(builder.clone());
        let (output_index, output) = builder.borrow_mut().new_var();
        let output = FieldVariable::from_var(builder.clone(), output);
        let flag = builder.borrow_mut().new_flag();
        builder
            .borrow_mut()
            .set_constraint(output_index, output.expr.clone() - lhs.expr.clone());
        builder.borrow_mut().set_compute(
            output_index,
            SymbolicExpr::Select(
                flag,
                Box::new(SymbolicExpr::Select(
                    flag,
                    Box::new(lhs.expr.clone()),
                    Box::new(lhs.expr.clone() / rhs.expr),
                )),
                Box::new(lhs.expr),
            ),
        );
        let program = FieldExpressionProgram::new(builder.borrow().clone(), true);
        let expr = FieldExpr::new(program, range_checker.bus());
        let filler =
            FieldExpressionFiller::new((), expr, vec![2, 3], vec![flag], range_checker, true);

        let program = build_device_program(&filler).unwrap();
        let division = program
            .value_ops
            .iter()
            .find(|op| op.opcode == ValueOpcode::Div as u32)
            .unwrap();
        assert_eq!(division.guard_true, 1 << flag);
        assert_eq!(division.guard_false, 1 << flag);
        serialize_field_expr(&filler).unwrap();
    }

    #[test]
    fn reuses_sequential_value_and_constraint_scratch() {
        let prime = secp256k1_coord_prime();
        let (range_checker, builder) = setup(&prime);
        let lhs = ExprBuilder::new_input(builder.clone());
        let rhs = ExprBuilder::new_input(builder.clone());
        let mut sum = lhs.clone() + rhs.clone();
        sum.save();
        let mut product = lhs * rhs;
        product.save();
        let program = FieldExpressionProgram::new(builder.borrow().clone(), false);
        let expr = FieldExpr::new(program, range_checker.bus());
        let filler = FieldExpressionFiller::new((), expr, vec![7], Vec::new(), range_checker, true);

        let program = build_device_program(&filler).unwrap();
        let temporary_slot = program.num_inputs + program.num_vars;
        let temporary_destinations = program
            .value_ops
            .iter()
            .filter(|op| {
                matches!(
                    op.opcode,
                    opcode if opcode == ValueOpcode::Add as u32
                        || opcode == ValueOpcode::Mul as u32
                )
            })
            .map(|op| op.dst)
            .collect::<Vec<_>>();
        assert_eq!(temporary_destinations, [temporary_slot, temporary_slot]);

        assert_eq!(program.constraints.len(), 2);
        for constraint in &program.constraints {
            assert_eq!(program.limb_ops[constraint.tape_start].dst_offset, 0);
        }
    }
}
