#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::str::FromStr;

use num_bigint_dig::BigUint;
use openvm_algebra_guest::{
    moduli_setup::{moduli_declare, moduli_init},
    Field, IntMod,
};
use openvm_ecc_guest::{
    edwards::TwistedEdwardsPoint,
    //k256::{Secp256k1Coord, Secp256k1Point, Secp256k1Scalar},
    //msm,
    //weierstrass::WeierstrassPoint,
    te_setup::{te_declare, te_init},
    Group,
};

moduli_declare! {
    Edwards25519Coord { modulus = "57896044618658097711785492504343953926634992332820282019728792003956564819949" },
    //Edwards25519_a { modulus = "57896044618658097711785492504343953926634992332820282019728792003956564819948" },
    //Edwards25519_d { modulus = "37095705934669439343138083508754565189542113879843219016388785533085940283555" },
}

moduli_init! {
    "57896044618658097711785492504343953926634992332820282019728792003956564819949",
    //"57896044618658097711785492504343953926634992332820282019728792003956564819948",
    //"37095705934669439343138083508754565189542113879843219016388785533085940283555",
}

impl Field for Edwards25519Coord {
    const ZERO: Self = <Self as IntMod>::ZERO;
    const ONE: Self = <Self as IntMod>::ONE;

    type SelfRef<'a> = &'a Self;

    fn double_assign(&mut self) {
        IntMod::double_assign(self);
    }

    fn square_assign(&mut self) {
        IntMod::square_assign(self);
    }
}

te_declare! {
    Edwards25519Point {
        mod_type = Edwards25519Coord,
        a = "57896044618658097711785492504343953926634992332820282019728792003956564819948",
        d = "37095705934669439343138083508754565189542113879843219016388785533085940283555",
    }
}

te_init! {
    Edwards25519Point,
}

openvm::entry!(main);

fn string_to_coord(s: &str) -> Edwards25519Coord {
    Edwards25519Coord::from_le_bytes(&BigUint::from_str(s).unwrap().to_bytes_le())
}

pub fn main() {
    setup_all_moduli();
    setup_all_curves();

    // Base point of edwards25519
    let x1 = string_to_coord(
        "15112221349535400772501151409588531511454012693041857206046113283949847762202",
    );
    let y1 = string_to_coord(
        "46316835694926478169428394003475163141307993866256225615783033603165251855960",
    );

    // random point on edwards25519
    let x2 = Edwards25519Coord::from_u32(2);
    let y2 = string_to_coord(
        "11879831548380997166425477238087913000047176376829905612296558668626594440753",
    );

    // This is the sum of (x1, y1) and (x2, y2).
    let x3 = string_to_coord(
        "44969869612046584870714054830543834361257841801051546235130567688769346152934",
    );
    let y3 = string_to_coord(
        "50796027728050908782231253190819121962159170739537197094456293084373503699602",
    );

    // This is 2 * (x1, y1)
    let x4 = string_to_coord(
        "24727413235106541002554574571675588834622768167397638456726423682521233608206",
    );
    let y4 = string_to_coord(
        "15549675580280190176352668710449542251549572066445060580507079593062643049417",
    );

    let mut p1 = Edwards25519Point::from_xy(x1.clone(), y1.clone()).unwrap();
    let mut p2 = Edwards25519Point::from_xy(x2, y2).unwrap();

    // Generic add can handle equal or unequal points.
    let p3 = &p1 + &p2;
    if p3.x() != &x3 || p3.y() != &y3 {
        panic!();
    }
    let p4 = &p2 + &p2;
    if p4.x() != &x4 || p4.y() != &y4 {
        panic!();
    }

    // Add assign and double assign
    p1 += &p2;
    if p1.x() != &x3 || p1.y() != &y3 {
        panic!();
    }
    p2.double_assign();
    if p2.x() != &x4 || p2.y() != &y4 {
        panic!();
    }
}
