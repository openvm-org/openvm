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

/// Value-phase operations over `K`-word Montgomery field elements.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum ValueOpcode {
    #[default]
    LoadInput = VOP_LOAD_INPUT,
    Constant = VOP_CONST,
    Add = VOP_ADD,
    Sub = VOP_SUB,
    Mul = VOP_MUL,
    Div = VOP_DIV,
    IntAdd = VOP_INTADD,
    IntMul = VOP_INTMUL,
    Select = VOP_SELECT,
    SaveVar = VOP_SAVE_VAR,
}

/// Limb-phase operations over signed vectors in the scratch arena.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum LimbOpcode {
    #[default]
    Input = LOP_INPUT,
    Var = LOP_VAR,
    Constant = LOP_CONST,
    Add = LOP_ADD,
    Sub = LOP_SUB,
    Mul = LOP_MUL,
    IntAdd = LOP_INTADD,
    IntMul = LOP_INTMUL,
    Select = LOP_SELECT,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ValueOp {
    pub opcode: ValueOpcode,
    pub flag: u32,
    pub dst: u32,
    pub a: u32,
    pub b: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LimbOp {
    pub opcode: LimbOpcode,
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

    value_ops: Vec<ValueOp>,
    num_value_slots: usize,
    limb_ops: Vec<LimbOp>,
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

    /// Montgomery-form payload for VOP_CONST / VOP_INTADD / VOP_INTMUL (K limbs each).
    mont_payload: Vec<u32>,
    /// Limb payload for LOP_CONST (concatenated, offsets stored in op.b_off).
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
        (self.num_value_slots + self.num_vars) * self.k + self.scratch_len + 4 * self.k
    }
}
