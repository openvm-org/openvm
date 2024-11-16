use ax_circuit_derive::AlignedBorrow;

use crate::arch::DynArray;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct MinimalInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
}

// This ProcessedInstruction is used by rv32_jalr and rv32_rdwrite
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ImmInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    pub immediate: T,
}

mod conversions {
    use super::*;

    impl<T> From<MinimalInstruction<T>> for DynArray<T> {
        fn from(m: MinimalInstruction<T>) -> Self {
            Self(vec![m.is_valid, m.opcode])
        }
    }

    impl<T> From<DynArray<T>> for MinimalInstruction<T> {
        fn from(m: DynArray<T>) -> Self {
            let mut m = m.0.into_iter();
            MinimalInstruction {
                is_valid: m.next().unwrap(),
                opcode: m.next().unwrap(),
            }
        }
    }

    impl<T> From<DynArray<T>> for ImmInstruction<T> {
        fn from(m: DynArray<T>) -> Self {
            let mut m = m.0.into_iter();
            ImmInstruction {
                is_valid: m.next().unwrap(),
                opcode: m.next().unwrap(),
                immediate: m.next().unwrap(),
            }
        }
    }

    impl<T> From<ImmInstruction<T>> for DynArray<T> {
        fn from(instruction: ImmInstruction<T>) -> Self {
            DynArray::from(vec![
                instruction.is_valid,
                instruction.opcode,
                instruction.immediate,
            ])
        }
    }
}
