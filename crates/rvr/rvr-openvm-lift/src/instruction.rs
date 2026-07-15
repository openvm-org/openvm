use openvm_instructions::{instruction::Instruction, VmOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

/// Canonical integer representation of an OpenVM instruction used by rvr lifting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RvrInstruction {
    pub opcode: VmOpcode,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub e: u32,
    pub f: u32,
    pub g: u32,
    signed_c: i32,
}

impl RvrInstruction {
    /// Lower an OpenVM instruction to its canonical integer representation.
    #[inline]
    pub fn from_field<F: PrimeField32>(insn: &Instruction<F>) -> Self {
        Self {
            opcode: insn.opcode,
            a: insn.a.as_canonical_u32(),
            b: insn.b.as_canonical_u32(),
            c: insn.c.as_canonical_u32(),
            d: insn.d.as_canonical_u32(),
            e: insn.e.as_canonical_u32(),
            f: insn.f.as_canonical_u32(),
            g: insn.g.as_canonical_u32(),
            signed_c: decode_signed(insn.c.as_canonical_u32(), F::ORDER_U32),
        }
    }

    pub const fn from_canonical(
        opcode: VmOpcode,
        [a, b, c, d, e, f, g]: [u32; 7],
        field_order: u32,
    ) -> Self {
        Self {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            signed_c: decode_signed(c, field_order),
        }
    }

    #[inline]
    pub const fn operands(&self) -> [u32; 7] {
        [self.a, self.b, self.c, self.d, self.e, self.f, self.g]
    }

    #[inline]
    pub const fn signed_c(&self) -> i32 {
        self.signed_c
    }
}

const fn decode_signed(value: u32, field_order: u32) -> i32 {
    if value > field_order / 2 {
        value.wrapping_sub(field_order) as i32
    } else {
        value as i32
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{instruction::Instruction, VmOpcode};
    use p3_baby_bear::BabyBear;

    use super::*;

    #[test]
    fn lowers_all_operands_to_canonical_u32() {
        let insn = Instruction::<BabyBear>::from_isize(VmOpcode::from_usize(7), 1, -1, 3, -4, 5);
        let lowered = RvrInstruction::from_field(&insn);

        assert_eq!(lowered.opcode, insn.opcode);
        assert_eq!(lowered.a, 1);
        assert_eq!(lowered.b, BabyBear::ORDER_U32 - 1);
        assert_eq!(lowered.c, 3);
        assert_eq!(lowered.d, BabyBear::ORDER_U32 - 4);
        assert_eq!(lowered.e, 5);
        assert_eq!(lowered.f, 0);
        assert_eq!(lowered.g, 0);
    }

    #[test]
    fn signed_decoding_uses_source_field_order() {
        let insn = RvrInstruction::from_canonical(VmOpcode::from_usize(0), [0; 7], 101);
        assert_eq!(insn.signed_c(), 0);

        let insn =
            RvrInstruction::from_canonical(VmOpcode::from_usize(0), [0, 0, 51, 0, 0, 0, 0], 101);
        assert_eq!(insn.signed_c(), -50);

        let insn =
            RvrInstruction::from_canonical(VmOpcode::from_usize(0), [0, 0, 100, 0, 0, 0, 0], 101);
        assert_eq!(insn.signed_c(), -1);
    }
}
