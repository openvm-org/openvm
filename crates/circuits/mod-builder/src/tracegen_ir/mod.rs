//! Compilation of a finalized [`crate::FieldExpr`] into trace-generation IR
//! ([`compile_tracegen_ir`]), encoding of that IR for the GPU interpreter
//! ([`TracegenIr::encode`]), and the types shared by the compiler and encoder.
//!
//! See `README.md` for the design overview.

mod abi;
mod compiler;
mod encoding;
#[cfg(test)]
mod tests;

use abi::*;
pub use compiler::{compile_tracegen_ir, TracegenCompileError};

/// Operations that compute field-variable values during trace generation.
///
/// The CUDA interpreter executes this tape over `K`-word Montgomery field elements, reducing
/// arithmetic modulo the configured prime. Its results populate the variable columns later used
/// by the witness phase and exposed as instruction outputs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum EvalOpcode {
    #[default]
    LoadInput = EVAL_OP_LOAD_INPUT,
    Constant = EVAL_OP_CONST,
    Add = EVAL_OP_ADD,
    Sub = EVAL_OP_SUB,
    Mul = EVAL_OP_MUL,
    Div = EVAL_OP_DIV,
    IntAdd = EVAL_OP_INTADD,
    IntMul = EVAL_OP_INTMUL,
    Select = EVAL_OP_SELECT,
    SaveVar = EVAL_OP_SAVE_VAR,
}

/// Operations that generate witnesses proving the evaluated field variables are correct.
///
/// The CUDA interpreter executes one such tape per constraint using unreduced signed limb
/// arithmetic. The resulting integer expression is used to derive the quotient and carry columns
/// that prove it is zero modulo the configured prime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum WitnessOpcode {
    #[default]
    Input = WITNESS_OP_INPUT,
    Var = WITNESS_OP_VAR,
    Constant = WITNESS_OP_CONST,
    Add = WITNESS_OP_ADD,
    Sub = WITNESS_OP_SUB,
    Mul = WITNESS_OP_MUL,
    IntAdd = WITNESS_OP_INTADD,
    IntMul = WITNESS_OP_INTMUL,
    Select = WITNESS_OP_SELECT,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalOp {
    pub opcode: EvalOpcode,
    pub flag: u32,
    pub dst: u32,
    pub a: u32,
    pub b: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WitnessOp {
    pub opcode: WitnessOpcode,
    pub flag: u32,
    pub dst_off: u32,
    pub dst_len: u32,
    pub a_off: u32,
    pub a_len: u32,
    pub b_off: u32,
    pub b_len: u32,
    pub imm: i32,
}

#[derive(Clone, Debug)]
pub struct ConstraintMeta {
    pub tape_start: usize,
    pub tape_len: usize,
    /// Scratch offset/len of the evaluated constraint expression limbs.
    pub result_off: u32,
    pub result_len: u32,
    pub q_limbs: usize,
    pub carry_limbs: usize,
    /// Range-check shift/bits for carries, from `get_carry_max_abs_and_bits` on the
    /// bound-propagated `expr - q * p` overflow int (data independent).
    pub carry_min_abs: u32,
    pub carry_bits: u32,
}

/// Validated host-side trace-generation IR. [`Self::encode`] flattens it for CUDA.
#[derive(Clone, Debug)]
pub struct TracegenIr {
    num_limbs: usize,
    limb_bits: usize,
    /// K: number of u32 limbs per field element.
    k: usize,
    num_input: usize,
    num_vars: usize,
    num_flags: usize,
    needs_setup: bool,
    /// Trace sub-row width (must equal `BaseAir::width(&expr)`).
    width: usize,

    eval_ops: Vec<EvalOp>,
    num_eval_slots: usize,
    witness_ops: Vec<WitnessOp>,
    scratch_len: usize,
    constraints: Vec<ConstraintMeta>,

    // Field constants (u32 little-endian limbs)
    p_u32: Vec<u32>,
    mprime: u32,
    r2_u32: Vec<u32>,
    pm2_u32: Vec<u32>,
    /// p^{-1} mod 2^(32*2K), 2K limbs (for exact division).
    pinv_u32: Vec<u32>,
    /// Prime as `ceil(p.bits()/limb_bits)` canonical limbs (matches `prime_overflow`).
    p8: Vec<i32>,

    /// Montgomery-form payload for EVAL_OP_CONST / EVAL_OP_INTADD / EVAL_OP_INTMUL
    /// (K limbs each).
    mont_payload: Vec<u32>,
    /// Limb payload for WITNESS_OP_CONST (concatenated, offsets stored in op.a_off).
    const_limbs_payload: Vec<i32>,

    /// Opcode -> flags mapping (from FieldExpressionFiller): position of the record's
    /// local opcode in `local_opcode_idx`; if < opcode_flag_idx.len(), that flag is set.
    local_opcode_idx: Vec<usize>,
    opcode_flag_idx: Vec<usize>,
}

impl TracegenIr {
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn aux_words(&self) -> usize {
        (self.num_eval_slots + self.num_vars) * self.k + self.scratch_len + 4 * self.k
    }
}
