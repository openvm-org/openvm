mod final_exp;
mod line;
mod miller_loop;
mod miller_step;
mod operations;
mod sextic_ext_field;

use axvm_algebra::{
    field::{ComplexConjugate, FieldExtension},
    Field, IntMod,
};
use axvm_platform::constants::{PairingBaseFunct7, PAIRING_MAX_KINDS};
pub use final_exp::*;
pub use line::*;
pub use miller_loop::*;
pub use miller_step::*;
pub(crate) use operations::*;
pub use sextic_ext_field::*;

pub trait PairingIntrinsics {
    type Fp: Field + IntMod;
    type Fp2: Field + FieldExtension<Self::Fp> + ComplexConjugate;
    type Fp12: FieldExtension<Self::Fp2> + ComplexConjugate;

    /// Index for custom intrinsic opcode determination.
    const PAIRING_IDX: usize;
    /// The sextic extension `Fp12` is `Fp2[X] / (X^6 - \xi)`, where `\xi` is a non-residue.
    const XI: Self::Fp2;
}

pub const fn shifted_funct7<P: PairingIntrinsics>(funct7: PairingBaseFunct7) -> usize {
    P::PAIRING_IDX * (PAIRING_MAX_KINDS as usize) + funct7 as usize
}
