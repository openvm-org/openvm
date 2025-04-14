use backtrace::Backtrace;
use openvm_stark_backend::p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::{utils::isize_to_field, LocalOpcode, PhantomDiscriminant, SystemOpcode, VmOpcode};

/// Number of operands of an instruction.
pub const NUM_OPERANDS: usize = 7;

#[repr(C)]
#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, PartialEq, Eq, derive_new::new, Serialize, Deserialize)]
pub struct Instruction<F> {
    pub opcode: VmOpcode,
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
    pub fn from_isize(opcode: VmOpcode, a: isize, b: isize, c: isize, d: isize, e: isize) -> Self {
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

    pub fn from_usize<const N: usize>(opcode: VmOpcode, operands: [usize; N]) -> Self {
        let mut operands = operands.map(F::from_canonical_usize).to_vec();
        operands.resize(NUM_OPERANDS, F::ZERO);
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
        opcode: VmOpcode,
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

    pub fn phantom(discriminant: PhantomDiscriminant, a: F, b: F, c_upper: u16) -> Self {
        Self {
            opcode: SystemOpcode::PHANTOM.global_opcode(),
            a,
            b,
            c: F::from_canonical_u32((discriminant.0 as u32) | ((c_upper as u32) << 16)),
            ..Default::default()
        }
    }

    pub fn debug(discriminant: PhantomDiscriminant) -> Self {
        Self {
            opcode: SystemOpcode::PHANTOM.global_opcode(),
            c: F::from_canonical_u32(discriminant.0 as u32),
            ..Default::default()
        }
    }

    pub fn operands(&self) -> Vec<F> {
        vec![self.a, self.b, self.c, self.d, self.e, self.f, self.g]
    }
}

impl<T: Default> Default for Instruction<T> {
    fn default() -> Self {
        Self {
            opcode: VmOpcode::from_usize(0), /* there is no real default opcode, this field must
                                              * always be set */
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
