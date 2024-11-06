use backtrace::Backtrace;
use p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::{utils::isize_to_field, PhantomInstruction, SystemOpcode, UsizeOpcode};

/// Number of operands of an instruction.
pub const NUM_OPERANDS: usize = 7;

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, PartialEq, Eq, derive_new::new, Serialize, Deserialize)]
pub struct Instruction<F> {
    pub opcode: usize,
    pub a: F,
    pub b: F,
    pub c: F,
    pub d: F,
    pub e: F,
    pub f: F,
    pub g: F,
}

impl<F: Field> Instruction<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn from_isize(opcode: usize, a: isize, b: isize, c: isize, d: isize, e: isize) -> Self {
        Self {
            opcode,
            a: isize_to_field::<F>(a),
            b: isize_to_field::<F>(b),
            c: isize_to_field::<F>(c),
            d: isize_to_field::<F>(d),
            e: isize_to_field::<F>(e),
            f: isize_to_field::<F>(0),
            g: isize_to_field::<F>(0),
        }
    }

    pub fn from_usize<const N: usize>(opcode: usize, operands: [usize; N]) -> Self {
        let mut operands = operands.map(F::from_canonical_usize).to_vec();
        operands.resize(NUM_OPERANDS, F::zero());
        Self {
            opcode,
            a: operands[0],
            b: operands[1],
            c: operands[2],
            d: operands[3],
            e: operands[4],
            f: operands[5],
            g: operands[6],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn large_from_isize(
        opcode: usize,
        a: isize,
        b: isize,
        c: isize,
        d: isize,
        e: isize,
        f: isize,
        g: isize,
    ) -> Self {
        Self {
            opcode,
            a: isize_to_field::<F>(a),
            b: isize_to_field::<F>(b),
            c: isize_to_field::<F>(c),
            d: isize_to_field::<F>(d),
            e: isize_to_field::<F>(e),
            f: isize_to_field::<F>(f),
            g: isize_to_field::<F>(g),
        }
    }

    pub fn phantom(kind: PhantomInstruction, a: F, b: F, c_upper: u16) -> Self {
        Self {
            opcode: SystemOpcode::PHANTOM.with_default_offset(),
            a,
            b,
            c: F::from_canonical_u32((kind as u32) | ((c_upper as u32) << 16)),
            ..Default::default()
        }
    }

    pub fn debug(phantom: PhantomInstruction) -> Self {
        Self {
            opcode: SystemOpcode::PHANTOM.with_default_offset(),
            c: F::from_canonical_usize(phantom as usize),
            ..Default::default()
        }
    }
}

impl<T: Default> Default for Instruction<T> {
    fn default() -> Self {
        Self {
            opcode: 0, // there is no real default opcode, this field must always be set
            a: T::default(),
            b: T::default(),
            c: T::default(),
            d: T::default(),
            e: T::default(),
            f: T::default(),
            g: T::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugInfo {
    pub dsl_instruction: String,
    pub trace: Option<Backtrace>,
}

impl DebugInfo {
    pub fn new(dsl_instruction: String, trace: Option<Backtrace>) -> Self {
        Self {
            dsl_instruction,
            trace,
        }
    }
}
