use std::{array::from_fn, ops::Mul};

use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::riscv::{RV64_REGISTER_AS, RV64_WORD_NUM_LIMBS};
use openvm_riscv_circuit::adapters::{tracing_read, RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS};
use openvm_stark_backend::{interaction::InteractionBuilder, p3_field::PrimeCharacteristicRing};

/// Zero-pads the 4 materialized bytes of an RV64 register into the full 8-byte register value
/// expected by the memory bus. The upper 4 bytes are hardcoded to zero in the interaction, which
/// enforces at proving time that the stored register fits in 32 bits without an extra assertion.
#[inline]
pub(crate) fn pad_reg_val<AB: InteractionBuilder>(
    val: [AB::Var; RV64_WORD_NUM_LIMBS],
) -> [AB::Expr; RV64_REGISTER_NUM_LIMBS] {
    from_fn(|i| {
        if i < RV64_WORD_NUM_LIMBS {
            val[i].into()
        } else {
            AB::Expr::ZERO
        }
    })
}

/// Composes the 4 materialized bytes of an RV64 register into a single field element, to be used
/// as a memory pointer.
#[inline]
pub(crate) fn compose_ptr<T, V>(data: [V; RV64_WORD_NUM_LIMBS]) -> T
where
    T: PrimeCharacteristicRing,
    V: Mul<T, Output = T>,
{
    data.into_iter()
        .enumerate()
        .fold(T::ZERO, |acc, (i, limb)| {
            acc + limb * T::from_u32(1 << (i * RV64_CELL_BITS))
        })
}

/// Reads an 8-byte register, debug-asserts that the upper 4 bytes are zero (enforced at proving
/// time by zero padding in the memory bus interaction), and returns the low 4 bytes as a u32
/// pointer.
#[inline(always)]
pub(crate) fn tracing_read_reg_ptr(
    memory: &mut TracingMemory,
    ptr: u32,
    prev_timestamp: &mut u32,
) -> u32 {
    let bytes: [u8; RV64_REGISTER_NUM_LIMBS] =
        tracing_read(memory, RV64_REGISTER_AS, ptr, prev_timestamp);
    debug_assert_eq!(
        bytes[RV64_WORD_NUM_LIMBS..],
        [0u8; RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS],
        "upper 4 bytes of register must be zero"
    );
    u32::from_le_bytes(bytes[..RV64_WORD_NUM_LIMBS].try_into().unwrap())
}
