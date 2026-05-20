use openvm_riscv_circuit::adapters::{RV64_PTR_U16_LIMBS, U16_BITS};

#[inline(always)]
pub(crate) fn u32_to_le_u16_limbs(value: u32) -> [u16; RV64_PTR_U16_LIMBS] {
    core::array::from_fn(|i| ((value >> (U16_BITS * i)) & (u16::MAX as u32)) as u16)
}
