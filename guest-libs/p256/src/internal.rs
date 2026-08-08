use core::ops::Add;

use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_algebra_moduli_macros::moduli_declare;
use openvm_ecc_guest::{
    weierstrass::{CachedMulTable, IntrinsicCurve, WeierstrassPoint},
    CyclicGroup, Group,
};
use openvm_ecc_sw_macros::sw_declare;

use crate::NistP256;

// --- Define the OpenVM modular arithmetic and ecc types ---

moduli_declare! {
    P256Coord { modulus = "0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff" },
    P256Scalar { modulus = "0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551" },
}

// from_const_bytes is little endian
pub const CURVE_A: P256Coord = P256Coord::from_const_bytes(hex!(
    "fcffffffffffffffffffffff00000000000000000000000001000000ffffffff"
));
pub const CURVE_B: P256Coord = P256Coord::from_const_bytes(hex!(
    "4b60d2273e3cce3bf6b053ccb0061d65bc86987655bdebb3e7933aaad835c65a"
));

sw_declare! {
    P256Point { mod_type = P256Coord, a = CURVE_A, b = CURVE_B },
}

macro_rules! generator_multiple {
    ($x:literal, $y:literal) => {
        P256Point {
            x: P256Coord::from_const_bytes(hex!($x)),
            y: P256Coord::from_const_bytes(hex!($y)),
            z: P256Coord::from_const_u8(1),
        }
    };
}

/// Affine multiples `2G..15G` for the four-bit ECDSA generator window.
const GENERATOR_MULTIPLES: [P256Point; 14] = [
    generator_multiple!(
        "78996647fc480ba6351bf277e26989c0c31ab5040338528a7e4f038d187bf27c",
        "d17378229db7049e2982e93ce6ad7dbadb30749fc69a3d2940d08edb10557707"
    ),
    generator_multiple!(
        "6cfde7c61b6641fb85a9adef21b7c6e665f14b1d95eff7c8440a33a6d1e4cb5e",
        "32507da227b1799a3db84f3836b02ad8eca2641ace064b377eff98490c643487"
    ),
    generator_multiple!(
        "5208036b44029350ef965578dbe21f03d02be69e65de2da0bb8fd032354a53e2",
        "c6d84e183fc2425c05e00ef3c396fc4e762d86da5feedf19c73c634c5a57f1e0"
    ),
    generator_multiple!(
        "ed33d0c30d4a552124e55b1ffd828cefdf8f660856c884d7d24051517a0b5951",
        "a46da1fd44bbd0d18808d8d4002f010d26798abf36bfe18a7d724a90a87dc1e0"
    ),
    generator_multiple!(
        "a991223ce9aab0c6b415b2eb0d744c02e3dd97b82c24d3922c60a4762a171ab0",
        "e27fc78f53487cfdbd167e1c70f7001c7903a7fb2d0eec6fd5da373274105ce8"
    ),
    generator_multiple!(
        "a3b28731702806305bef0fa8b8f8f97e60fb017c6630bb25467bbfa06f3b538e",
        "b400f4c1861a5ec5211b04cb3336c7530090f5a6839f066d361833e0bd1deb73"
    ),
    generator_multiple!(
        "93b36fdbc19dddb4db97ce0f9838d2c1ad4cb53a2d74424053b0e9be9d77d962",
        "7e95090f6a0a54da786ae7bbf651eda2e0ce6711775df14f24d8e991bdcc5aad"
    ),
    generator_multiple!(
        "e09e94904b8a9ed7b3f86d2c8ccb0a9e72f8711dd5388987710bdffeb6d768ea",
        "fa48d04d4a225ae83f82dea4ea4f714dc8a08e4a964a0187e7fcc972c944272a"
    ),
    generator_multiple!(
        "3f72c5049406364c6e30481c476cca45b53f22ead11412593e993a2a6b6df6ce",
        "7307af44aabb34caee1e75fe29ed0d59104c3b9ddd3c126e90aeaa29a2628687"
    ),
    generator_multiple!(
        "d121bc74d3913343bf485025d02e7416da1cc2b09d373806594c3b88b713d13e",
        "40372ae8fceef8e2da89985eda040d098ac6f4a4af43c824a2c8c4cc9a209990"
    ),
    generator_multiple!(
        "c4e32486eec500d5992cf8b22830987951d5e520735326465ed917a8bdd51d74",
        "d38144cd22ff9519a75cba352c91eb8e54b1874855837356dc5f389c6ab47007"
    ),
    generator_multiple!(
        "012c07469d5de1988ad5ea654b282e79fce25ed8f25d80615a49ace07a837c17",
        "d8bfc7efe2bb439cf34dfba1c314ee26724e0fb4ad9140a258a5be4ecd58bb63"
    ),
    generator_multiple!(
        "0b92d224732709575e9c067abeac26f13cdf36437f64767ab962381c007ae754",
        "75b3d0602fc8a71b0890507377ea7171c3e7a2058c1f12427531f429bbf199f5"
    ),
    generator_multiple!(
        "5f9d9be5638c6663f10e3ade92af03ae658288998937fbade7ba1a97c64d45f0",
        "364f030dde9ce5473ffab575ce213b2ae643961fe594654e1f2d2e59e33eb9b5"
    ),
];

// --- Implement internal traits ---

impl CyclicGroup for P256Point {
    // The constants are taken from: https://neuromancer.sk/std/secg/secp256r1
    const GENERATOR: Self = P256Point {
        // from_const_bytes takes a little endian byte string
        x: P256Coord::from_const_bytes(hex!(
            "96c298d84539a1f4a033eb2d817d0377f240a463e5e6bcf847422ce1f2d1176b"
        )),
        y: P256Coord::from_const_bytes(hex!(
            "f551bf376840b6cbce5e316b5733ce2b169e0f7c4aebe78e9b7f1afee242e34f"
        )),
        z: P256Coord::from_const_u8(1),
    };
    const NEG_GENERATOR: Self = P256Point {
        x: P256Coord::from_const_bytes(hex!(
            "96c298d84539a1f4a033eb2d817d0377f240a463e5e6bcf847422ce1f2d1176b"
        )),
        y: P256Coord::from_const_bytes(hex!(
            "0aae40c897bf493431a1ce94a9cc31d4e961f083b51418716580e5011cbd1cb0"
        )),
        z: P256Coord::from_const_u8(1),
    };
}

impl IntrinsicCurve for NistP256 {
    type Scalar = P256Scalar;
    type Point = P256Point;

    fn msm(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
    where
        for<'a> &'a Self::Point: Add<&'a Self::Point, Output = Self::Point>,
    {
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

// --- Implement helpful methods mimicking the structs in p256 ---

impl P256Point {
    pub fn x_be_bytes(&self) -> [u8; 32] {
        let n = self.normalize();
        <Self as WeierstrassPoint>::x(&n).to_be_bytes()
    }

    pub fn y_be_bytes(&self) -> [u8; 32] {
        let n = self.normalize();
        <Self as WeierstrassPoint>::y(&n).to_be_bytes()
    }
}
