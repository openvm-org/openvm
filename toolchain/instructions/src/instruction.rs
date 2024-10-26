use backtrace::Backtrace;
use p3_field::Field;

use crate::utils::isize_to_field;

/// Number of operands of an instruction.
pub const NUM_OPERANDS: usize = 7;

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, PartialEq, Eq, derive_new::new)]
pub struct Instruction<F> {
    pub opcode: usize,
    pub a: F,
    pub b: F,
    pub c: F,
    pub d: F,
    pub e: F,
    pub f: F,
    pub g: F,
    pub debug: String,
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
            debug: String::new(),
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
            debug: String::new(),
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
            debug: String::new(),
        }
    }

    pub fn debug(opcode: usize, debug: &str) -> Self {
        Self {
            opcode,
            a: F::zero(),
            b: F::zero(),
            c: F::zero(),
            d: F::zero(),
            e: F::zero(),
            f: F::zero(),
            g: F::zero(),
            debug: String::from(debug),
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
            debug: String::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
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
