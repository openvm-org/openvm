use core::ops::Add;

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

macro_rules! generator_multiple {
    ($x:literal, $y:literal) => {
        Secp256k1Point {
            x: Secp256k1Coord::from_const_bytes(hex!($x)),
            y: Secp256k1Coord::from_const_bytes(hex!($y)),
            z: Secp256k1Coord::from_const_u8(1),
        }
    };
}

/// Affine multiples `2G..15G` for the four-bit ECDSA generator window.
const GENERATOR_MULTIPLES: [Secp256k1Point; 14] = [
    generator_multiple!(
        "E59E705CB909ACABA73CEF8C4B8E775CD87CC0956E4045306D7DED41947F04C6",
        "2AE5CF50A9316423E1D066326532F6F7EEEA6C461984C5A339C33DA6FE68E11A"
    ),
    generator_multiple!(
        "F936E0BC13F10186B0996F8345C831B529529DF8854F344910C35892018A30F9",
        "72E6B88475FDB96C1B23C23499A9006556F3372AE637E30F14E82D630F7B8F38"
    ),
    generator_multiple!(
        "13CDC4E8AB94FA748475E00E90136CCC04140B9304491E58F3800DC1F1DB93E4",
        "22997347DC7BE9CF40FEBDBF33AE67D94814A58E09E24256B755D4A03E99ED51"
    ),
    generator_multiple!(
        "E4EF40B269D5A8CBB79A61DCBD848BE828515C0A25A7B4559320071A4DDE8B2F",
        "D662ACA63A7DA8DC40680DAB1B2788F726C4C9A6DDA9DBD4D6E3E5362622ACD8"
    ),
    generator_multiple!(
        "56752960147A052F8BA168852F47F682D3355235143A4520A4EE5E75D57BF9FF",
        "97F275B0360C873CA0E48F51F6F080DE60C5457F0196BEF320B6FBAC7A7712AE"
    ),
    generator_multiple!(
        "BCF9C4CAEDDD2BE99CE330037E9B413D0E7AEAF265F398A3EAB45D6E64F0BD5C",
        "DA647208282608A5B5E7FD13B8D013A8DB541A866D8D17A3605925BA40CAEB6A"
    ),
    generator_multiple!(
        "012A0AE1F34E78678A88AFE505DD1B0A2F3C0FB73F84F3AF1D35CA5CE1E5012F",
        "04E9BD6CB72CDAB517765BBAD613E2C2B4132D132A083D2949995341A7A84D5C"
    ),
    generator_multiple!(
        "BECC27FC0D115FC314E7574C979697E0BD9A559F8A17AD0953F6C7F0E284D4AC",
        "379C4FC62A26CC050F8E5F37A488D8ADE9613B7671093864FDD9A7B0218933CC"
    ),
    generator_multiple!(
        "C747E2472A8EA652B7C243199BD442345DAEE61A7B7C473562C8F3479E4D43A0",
        "D76873033BE5BE3C59A177D82E4C796F694CA293E6C7B6A327BC195442BA3A89"
    ),
    generator_multiple!(
        "CB08A05D8917ECBB9178C1E50B984956AC5AC6706B24F45E1E41A958F8E74A77",
        "1BC653C9C9741D30A8D6F9DFE2B12D3765B3B7D756DD4302195E6BEB32A084D9"
    ),
    generator_multiple!(
        "5AE8AF7070F4B0C55B09209641F47C683346734D008FC3151B56E748D51511D0",
        "272306F4131B056B526DA8D95D8C237915D87BE13745B6A8D7E015C8FD4FF3A9"
    ),
    generator_multiple!(
        "A85A40198FDFEDDECD580E61C6FB75B0518674C305D2D1C78B2875D9C27387F2",
        "81ED03DB52CBB5291FA91F52DA061A3A47AFCD65EB128275890A888D2E90B00A"
    ),
    generator_multiple!(
        "E423E8601A249BE4E6498967637BAA26328ED3077FE664FD9C715E899EDF9F49",
        "5B3FA103D4405FC6BC953F7AC279424664D4B3A7E444F09051854EB5C4F6C2CA"
    ),
    generator_multiple!(
        "0E087EE2F8BCAD449EF7853C6F94E53111F45F09E35A465A96EA437D4F4D92D7",
        "586BA2F69FDC04C5A5D396D82BAF40EAEF6DCC28C22E8483A6726CA872281E58"
    ),
];

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
        z: Secp256k1Coord::from_const_u8(1),
    };
    const NEG_GENERATOR: Self = Secp256k1Point {
        x: Secp256k1Coord::from_const_bytes(hex!(
            "9817F8165B81F259D928CE2DDBFC9B02070B87CE9562A055ACBBDCF97E66BE79"
        )),
        y: Secp256k1Coord::from_const_bytes(hex!(
            "7727EF046F2FB863E6AB7A59B74BE80257F7EEF103045BA29A3B5CD98825C5B7"
        )),
        z: Secp256k1Coord::from_const_u8(1),
    };
}

impl IntrinsicCurve for Secp256k1 {
    type Scalar = Secp256k1Scalar;
    type Point = Secp256k1Point;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
    where
        for<'a> &'a Self::Point: Add<&'a Self::Point, Output = Self::Point>,
    {
        // heuristic
        if coeffs.len() < 25 {
            let table = CachedMulTable::<Self>::new_with_prime_order(bases, 4);
            table.windowed_mul(coeffs)
        } else {
            openvm_ecc_guest::msm(coeffs, bases)
        }
    }

    fn lincomb_generator(
        generator_scalar: &Self::Scalar,
        point_scalar: &Self::Scalar,
        point: &Self::Point,
    ) -> Self::Point {
        let bases = [Self::Point::GENERATOR, *point];
        let precomputed = [Some(GENERATOR_MULTIPLES.as_slice()), None];
        CachedMulTable::<Self>::new_with_prime_order_and_precomputed(&bases, 4, &precomputed)
            .windowed_mul(&[*generator_scalar, *point_scalar])
    }

    fn lincomb_neg_generator(
        generator_scalar: &Self::Scalar,
        point_scalar: &Self::Scalar,
        point: &Self::Point,
    ) -> Self::Point {
        Self::lincomb_generator(&(-generator_scalar), point_scalar, point)
    }
}

// --- Implement helpful methods mimicking the structs in k256 ---

impl Secp256k1Point {
    pub fn x_be_bytes(&self) -> [u8; 32] {
        let n = self.normalize();
        <Self as WeierstrassPoint>::x(&n).to_be_bytes()
    }

    pub fn y_be_bytes(&self) -> [u8; 32] {
        let n = self.normalize();
        <Self as WeierstrassPoint>::y(&n).to_be_bytes()
    }
}
