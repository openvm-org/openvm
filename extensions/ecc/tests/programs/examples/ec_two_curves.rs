#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use hex_literal::hex;
use openvm_algebra_guest::IntMod;
use openvm_ecc_guest::{msm, weierstrass::WeierstrassPoint, Group};
use openvm_k256::{Secp256k1Coord, Secp256k1Point, Secp256k1Scalar};
use openvm_p256::{P256Coord, P256Point};

openvm::init!("openvm_init_ec_two_curves_k256_p256.rs");

openvm::entry!(main);

pub fn main() {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
    // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
    let x1 = Secp256k1Coord::from_u32(1);
    let y1 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "EEA7767E580D75BC6FDD7F58D2A84C2614FB22586068DB63B346C6E60AF21842"
    ));
    let x2 = Secp256k1Coord::from_u32(2);
    let y2 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "D1A847A8F879E0AEE32544DA5BA0B3BD1703A1F52867A5601FF6454DD8180499"
    ));
    // This is the sum of (x1, y1) and (x2, y2).
    let x3 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "BE675E31F8AC8200CBCC6B10CECCD6EB93FB07D99BB9E7C99CC9245C862D3AF2"
    ));
    let y3 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "B44573B48FD3416DD256A8C0E1BAD03E88A78BF176778682589B9CB478FC1D79"
    ));
    // This is the double of (x2, y2).
    let x4 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "3BFFFFFF32333333333333333333333333333333333333333333333333333333"
    ));
    let y4 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "AC54ECC4254A4EDCAB10CC557A9811ED1EF7CB8AFDC64820C6803D2C5F481639"
    ));

    let mut p1 = Secp256k1Point::from_xy(x1.clone(), y1.clone()).unwrap();
    let mut p2 = Secp256k1Point::from_xy(x2, y2).unwrap();

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

    // Ec Mul
    let p1 = Secp256k1Point::from_xy(x1, y1).unwrap();
    let scalar = Secp256k1Scalar::from_u32(12345678);
    // Calculated with https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#ec-multiply-tool
    let x5 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "194A93387F790803D972AF9C4A40CB89D106A36F58EE2F31DC48A41768216D6D"
    ));
    let y5 = Secp256k1Coord::from_le_bytes_unchecked(&hex!(
        "9E272F746DA7BED171E522610212B6AEEAAFDB2AD9F4B530B8E1B27293B19B2C"
    ));
    let result = msm(&[scalar], &[p1]);
    if result.x() != &x5 || result.y() != &y5 {
        panic!();
    }

    // Sample points got from https://asecuritysite.com/ecc/p256p
    let x1 = P256Coord::from_u32(5);
    let y1 = P256Coord::from_le_bytes_unchecked(&hex!(
        "ccfb4832085c4133c5a3d9643c50ca11de7a8199ce3b91fe061858aab9439245"
    ));
    let p1 = P256Point::from_xy(x1.clone(), y1.clone()).unwrap();
    let x2 = P256Coord::from_u32(6);
    let y2 = P256Coord::from_le_bytes_unchecked(&hex!(
        "cb23828228510d22e9c0e70fb802d1dc47007233e5856946c20a25542c4cb236"
    ));
    let p2 = P256Point::from_xy(x2.clone(), y2.clone()).unwrap();

    // Generic add can handle equal or unequal points.
    let p3 = &p1 + &p2;
    let p4 = &p2 + &p2;

    // Add assign and double assign
    let mut sum = P256Point::from_xy(x1, y1).unwrap();
    sum += &p2;
    if sum.x() != p3.x() || sum.y() != p3.y() {
        panic!();
    }
    let mut double = P256Point::from_xy(x2, y2).unwrap();
    double.double_assign();
    if double.x() != p4.x() || double.y() != p4.y() {
        panic!();
    }
}
