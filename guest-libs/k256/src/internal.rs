use core::ops::{Add, AddAssign, Neg};

use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_algebra_moduli_macros::moduli_declare;
use openvm_ecc_guest::{
    weierstrass::{CachedMulTable, IntrinsicCurve, WeierstrassPoint},
    CyclicGroup, Group,
};
use openvm_ecc_sw_macros::sw_declare;

use crate::Secp256k1;

// --- Define the OpenVM modular arithmetic and ecc types ---

const CURVE_B: Secp256k1Coord = Secp256k1Coord::from_const_bytes(seven_le());
pub const fn seven_le() -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[0] = 7;
    buf
}

moduli_declare! {
    Secp256k1Coord { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F" },
    Secp256k1Scalar { modulus = "0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141" },
}

sw_declare! {
    Secp256k1Point { mod_type = Secp256k1Coord, b = CURVE_B },
}

// --- Implement internal traits ---

impl CyclicGroup for Secp256k1Point {
    // The constants are taken from: https://en.bitcoin.it/wiki/Secp256k1
    const GENERATOR: Self = Secp256k1Point {
        // from_const_bytes takes a little endian byte string
        x: Secp256k1Coord::from_const_bytes(hex!(
            "9817F8165B81F259D928CE2DDBFC9B02070B87CE9562A055ACBBDCF97E66BE79"
        )),
        y: Secp256k1Coord::from_const_bytes(hex!(
            "B8D410FB8FD0479C195485A648B417FDA808110EFCFBA45D65C4A32677DA3A48"
        )),
    };
    const NEG_GENERATOR: Self = Secp256k1Point {
        x: Secp256k1Coord::from_const_bytes(hex!(
            "9817F8165B81F259D928CE2DDBFC9B02070B87CE9562A055ACBBDCF97E66BE79"
        )),
        y: Secp256k1Coord::from_const_bytes(hex!(
            "7727EF046F2FB863E6AB7A59B74BE80257F7EEF103045BA29A3B5CD98825C5B7"
        )),
    };
}

impl IntrinsicCurve for Secp256k1 {
    type Scalar = Secp256k1Scalar;
    type Point = Secp256k1Point;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
    where
        for<'a> &'a Self::Point: Add<&'a Self::Point, Output = Self::Point>,
    {
        #[cfg(not(target_os = "zkvm"))]
        {
            // heuristic
            if coeffs.len() < 25 {
                let table = CachedMulTable::<Self>::new_with_prime_order(bases, 4);
                table.windowed_mul(coeffs)
            } else {
                openvm_ecc_guest::msm(coeffs, bases)
            }
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut acc = <Self::Point as Group>::IDENTITY;
            for (coeff, base) in coeffs.iter().zip(bases.iter()) {
                unsafe {
                    let mut uninit: core::mem::MaybeUninit<Self::Point> =
                        core::mem::MaybeUninit::uninit();
                    secp256k1_ec_mul(
                        uninit.as_mut_ptr() as usize,
                        coeff as *const Self::Scalar as usize,
                        base as *const Self::Point as usize,
                    );
                    acc.add_assign(&uninit.assume_init());
                }
            }
            acc
        }
    }
}

#[cfg(target_os = "zkvm")]
#[inline(always)]
#[no_mangle]
extern "C" fn secp256k1_ec_mul(rd: usize, rs1: usize, rs2: usize) {
    openvm_platform::custom_insn_r!(
        opcode = openvm_ecc_guest::OPCODE,
        funct3 = openvm_ecc_guest::SW_FUNCT3 as usize,
        funct7 = openvm_ecc_guest::SwBaseFunct7::SwEcMul as usize,
        rd = In rd,
        rs1 = In rs1,
        rs2 = In rs2
    );
}

// --- Implement helpful methods mimicking the structs in k256 ---

impl Secp256k1Point {
    pub fn x_be_bytes(&self) -> [u8; 32] {
        <Self as WeierstrassPoint>::x(self).to_be_bytes()
    }

    pub fn y_be_bytes(&self) -> [u8; 32] {
        <Self as WeierstrassPoint>::y(self).to_be_bytes()
    }
}
