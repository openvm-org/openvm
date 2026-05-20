use openvm_riscv_circuit::adapters::{
    RV64_LOW32_U16_LIMBS, RV64_U16_LIMB_BITS, RV64_U16_LIMB_MASK,
};

#[inline(always)]
pub(crate) fn u32_to_le_u16_limbs(value: u32) -> [u16; RV64_LOW32_U16_LIMBS] {
    core::array::from_fn(|i| ((value >> (RV64_U16_LIMB_BITS * i)) & RV64_U16_LIMB_MASK) as u16)
}
