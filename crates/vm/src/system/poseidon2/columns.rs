use openvm_circuit_primitives::{AlignedBorrow, StructReflectionHelper};
use openvm_poseidon2_air::Poseidon2SubCols;

/// Columns for Poseidon2Vm AIR.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Poseidon2PeripheryCols<F, const SBOX_REGISTERS: usize> {
    pub inner: Poseidon2SubCols<F, SBOX_REGISTERS>,
    pub mult: F,
}

/// Manual impl because Poseidon2SubCols is an external type without StructReflection.
impl<F, const SBOX_REGISTERS: usize> StructReflectionHelper
    for Poseidon2PeripheryCols<F, SBOX_REGISTERS>
{
    fn struct_reflection() -> Option<Vec<String>> {
        None
    }
}
