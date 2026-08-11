use std::{error::Error, fmt};

use backtrace::Backtrace;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

use crate::{LocalOpcode, PhantomDiscriminant, SystemOpcode, VmOpcode};

/// Number of operands of an instruction.
pub const NUM_OPERANDS: usize = 7;

/// Field-independent value stored in an OpenVM instruction operand.
///
/// Operands use an `i32` representation so signed values such as control-flow offsets do not
/// depend on a field modulus. Every operand is restricted to the signed 30-bit interval
/// `[-2^29, 2^29)`. This makes conversion into any supported proof field (whose order is at least
/// `2^30`) injective while keeping instructions field-independent and compact.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct InstructionOperand(i32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionOperandOutOfRange;

impl fmt::Display for InstructionOperandOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("instruction operand must fit in signed 30 bits")
    }
}

impl Error for InstructionOperandOutOfRange {}

impl InstructionOperand {
    pub const MIN: i32 = -(1 << 29);
    pub const MAX: i32 = (1 << 29) - 1;

    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const TWO: Self = Self(2);

    /// Creates an operand from a signed 32-bit value.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not fit in the signed 30-bit operand domain.
    pub const fn from_i32(value: i32) -> Self {
        assert!(
            value >= Self::MIN && value <= Self::MAX,
            "instruction operand must fit in signed 30 bits"
        );
        Self(value)
    }

    /// Creates an operand from a non-negative value.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not fit in the signed 30-bit operand domain.
    pub fn from_u32(value: u32) -> Self {
        Self::from_i32(
            value
                .try_into()
                .expect("instruction operand must fit in signed 30 bits"),
        )
    }

    /// Creates an operand from a signed, pointer-sized value.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not fit in the signed 30-bit operand domain.
    pub fn from_isize(value: isize) -> Self {
        Self::from_i32(
            value
                .try_into()
                .expect("instruction operand must fit in signed 30 bits"),
        )
    }

    /// Creates an operand from a non-negative, pointer-sized value.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not fit in the signed 30-bit operand domain.
    pub fn from_usize(value: usize) -> Self {
        Self::from_i32(
            value
                .try_into()
                .expect("instruction operand must fit in signed 30 bits"),
        )
    }

    pub const fn as_i32(self) -> i32 {
        self.0
    }

    /// Returns the raw two's-complement representation of this operand.
    pub const fn as_u32(self) -> u32 {
        self.0 as u32
    }

    /// Returns this operand as a non-negative integer, or `None` if it is signed.
    pub fn checked_as_u32(self) -> Option<u32> {
        self.0.try_into().ok()
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    pub const fn is_one(self) -> bool {
        self.0 == 1
    }
}

impl<'de> Deserialize<'de> for InstructionOperand {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = i32::deserialize(deserializer)?;
        if !(Self::MIN..=Self::MAX).contains(&value) {
            return Err(D::Error::custom(format_args!(
                "instruction operand {value} is outside the signed 30-bit domain"
            )));
        }
        Ok(Self(value))
    }
}

impl TryFrom<i32> for InstructionOperand {
    type Error = InstructionOperandOutOfRange;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        (Self::MIN..=Self::MAX)
            .contains(&value)
            .then_some(Self(value))
            .ok_or(InstructionOperandOutOfRange)
    }
}

impl TryFrom<isize> for InstructionOperand {
    type Error = InstructionOperandOutOfRange;

    fn try_from(value: isize) -> Result<Self, Self::Error> {
        i32::try_from(value)
            .map_err(|_| InstructionOperandOutOfRange)?
            .try_into()
    }
}

impl TryFrom<u32> for InstructionOperand {
    type Error = InstructionOperandOutOfRange;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        i32::try_from(value)
            .map_err(|_| InstructionOperandOutOfRange)?
            .try_into()
    }
}

impl TryFrom<usize> for InstructionOperand {
    type Error = InstructionOperandOutOfRange;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        i32::try_from(value)
            .map_err(|_| InstructionOperandOutOfRange)?
            .try_into()
    }
}

impl From<bool> for InstructionOperand {
    fn from(value: bool) -> Self {
        Self::from_u32(value.into())
    }
}

macro_rules! impl_infallible_operand_from {
    ($($ty:ty),* $(,)?) => {
        $(impl From<$ty> for InstructionOperand {
            fn from(value: $ty) -> Self {
                Self(value.into())
            }
        })*
    };
}

impl_infallible_operand_from!(i8, u8, i16, u16);

impl fmt::Display for InstructionOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[repr(C)]
#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: VmOpcode,
    pub a: InstructionOperand,
    pub b: InstructionOperand,
    pub c: InstructionOperand,
    pub d: InstructionOperand,
    pub e: InstructionOperand,
    pub f: InstructionOperand,
    pub g: InstructionOperand,
}

impl Instruction {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        opcode: VmOpcode,
        a: impl Into<InstructionOperand>,
        b: impl Into<InstructionOperand>,
        c: impl Into<InstructionOperand>,
        d: impl Into<InstructionOperand>,
        e: impl Into<InstructionOperand>,
        f: impl Into<InstructionOperand>,
        g: impl Into<InstructionOperand>,
    ) -> Self {
        Self {
            opcode,
            a: a.into(),
            b: b.into(),
            c: c.into(),
            d: d.into(),
            e: e.into(),
            f: f.into(),
            g: g.into(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_isize(opcode: VmOpcode, a: isize, b: isize, c: isize, d: isize, e: isize) -> Self {
        Self {
            opcode,
            a: InstructionOperand::from_isize(a),
            b: InstructionOperand::from_isize(b),
            c: InstructionOperand::from_isize(c),
            d: InstructionOperand::from_isize(d),
            e: InstructionOperand::from_isize(e),
            f: InstructionOperand::ZERO,
            g: InstructionOperand::ZERO,
        }
    }

    pub fn from_usize<const N: usize>(opcode: VmOpcode, operands: [usize; N]) -> Self {
        let operand = |index| {
            operands
                .get(index)
                .copied()
                .map(InstructionOperand::from_usize)
                .unwrap_or(InstructionOperand::ZERO)
        };
        Self {
            opcode,
            a: operand(0),
            b: operand(1),
            c: operand(2),
            d: operand(3),
            e: operand(4),
            f: operand(5),
            g: operand(6),
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
            a: InstructionOperand::from_isize(a),
            b: InstructionOperand::from_isize(b),
            c: InstructionOperand::from_isize(c),
            d: InstructionOperand::from_isize(d),
            e: InstructionOperand::from_isize(e),
            f: InstructionOperand::from_isize(f),
            g: InstructionOperand::from_isize(g),
        }
    }

    pub fn phantom(
        discriminant: PhantomDiscriminant,
        a: impl Into<InstructionOperand>,
        b: impl Into<InstructionOperand>,
        c_upper: u16,
    ) -> Self {
        Self {
            opcode: SystemOpcode::PHANTOM.global_opcode(),
            a: a.into(),
            b: b.into(),
            c: InstructionOperand::from_u32(discriminant.0.into()),
            d: InstructionOperand::from_u32(c_upper.into()),
            ..Default::default()
        }
    }

    pub fn debug(discriminant: PhantomDiscriminant) -> Self {
        Self {
            opcode: SystemOpcode::PHANTOM.global_opcode(),
            c: InstructionOperand::from_u32(discriminant.0 as u32),
            ..Default::default()
        }
    }

    /// Returns validated system-phantom operands `[a, b, discriminant, c_upper]`.
    ///
    /// System phantoms require non-negative `a` and `b`, 16-bit `discriminant` and `c_upper`, and
    /// zero for every unused operand.
    pub fn checked_phantom_operands(&self) -> Option<[u32; 4]> {
        if !self.e.is_zero() || !self.f.is_zero() || !self.g.is_zero() {
            return None;
        }
        let a = self.a.checked_as_u32()?;
        let b = self.b.checked_as_u32()?;
        let discriminant = self.c.checked_as_u32()?;
        let c_upper = self.d.checked_as_u32()?;
        u16::try_from(discriminant).ok()?;
        u16::try_from(c_upper).ok()?;
        Some([a, b, discriminant, c_upper])
    }

    pub const fn operands(&self) -> [InstructionOperand; NUM_OPERANDS] {
        [self.a, self.b, self.c, self.d, self.e, self.f, self.g]
    }
}

impl Default for Instruction {
    fn default() -> Self {
        Self {
            opcode: VmOpcode::from_usize(0), /* there is no real default opcode, this field must
                                              * always be set */
            a: InstructionOperand::ZERO,
            b: InstructionOperand::ZERO,
            c: InstructionOperand::ZERO,
            d: InstructionOperand::ZERO,
            e: InstructionOperand::ZERO,
            f: InstructionOperand::ZERO,
            g: InstructionOperand::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::de::value::{Error, I32Deserializer};

    use super::*;

    #[test]
    fn operand_i32_boundaries() {
        assert_eq!(
            InstructionOperand::from_i32(InstructionOperand::MIN).as_i32(),
            InstructionOperand::MIN
        );
        assert_eq!(InstructionOperand::from_i32(-1).as_i32(), -1);
        assert_eq!(InstructionOperand::ZERO.as_i32(), 0);
        assert_eq!(
            InstructionOperand::from_u32(InstructionOperand::MAX as u32).as_u32(),
            InstructionOperand::MAX as u32
        );
    }

    #[test]
    #[should_panic(expected = "instruction operand must fit in signed 30 bits")]
    fn operand_rejects_value_above_signed_30_bit_domain() {
        InstructionOperand::from_u32(InstructionOperand::MAX as u32 + 1);
    }

    #[test]
    #[should_panic(expected = "instruction operand must fit in signed 30 bits")]
    fn operand_rejects_value_below_signed_30_bit_domain() {
        InstructionOperand::from_i32(InstructionOperand::MIN - 1);
    }

    #[test]
    fn operand_deserialization_enforces_signed_30_bit_domain() {
        let valid =
            InstructionOperand::deserialize(I32Deserializer::<Error>::new(InstructionOperand::MIN))
                .unwrap();
        assert_eq!(valid.as_i32(), InstructionOperand::MIN);
        assert!(
            InstructionOperand::deserialize(I32Deserializer::<Error>::new(
                InstructionOperand::MAX + 1,
            ))
            .is_err()
        );
    }

    #[test]
    fn signed_operand_exposes_raw_bits_and_checked_unsigned_access() {
        let operand = InstructionOperand::from_i32(-1);
        assert_eq!(operand.as_u32(), u32::MAX);
        assert_eq!(operand.checked_as_u32(), None);
    }

    #[test]
    fn from_usize_pads_without_allocating() {
        let instruction = Instruction::from_usize(VmOpcode::from_usize(7), [1, 2, 3]);
        assert_eq!(
            instruction.operands(),
            [
                InstructionOperand::ONE,
                InstructionOperand::TWO,
                InstructionOperand::from_i32(3),
                InstructionOperand::ZERO,
                InstructionOperand::ZERO,
                InstructionOperand::ZERO,
                InstructionOperand::ZERO,
            ]
        );
    }

    #[test]
    fn phantom_keeps_discriminant_and_upper_bits_in_separate_operands() {
        let instruction = Instruction::phantom(PhantomDiscriminant(u16::MAX), 0u8, 0u8, u16::MAX);
        assert_eq!(instruction.c.as_u32(), u32::from(u16::MAX));
        assert_eq!(instruction.d.as_u32(), u32::from(u16::MAX));
        assert_eq!(
            instruction.c.as_u32() | (instruction.d.as_u32() << 16),
            u32::MAX
        );
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
