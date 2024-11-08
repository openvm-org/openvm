#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use axvm::intrinsics::IntMod;
use axvm_ecc::sw::{EcPointN, IntModN};
use hex_literal::hex;

axvm::entry!(main);

pub fn main() {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
    // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
    let x1 = IntModN::from_u32(1);
    let y1 = IntModN::from_le_bytes(&hex!(
        "EEA7767E580D75BC6FDD7F58D2A84C2614FB22586068DB63B346C6E60AF21842"
    ));
    let x2 = IntModN::from_u32(2);
    let y2 = IntModN::from_le_bytes(&hex!(
        "D1A847A8F879E0AEE32544DA5BA0B3BD1703A1F52867A5601FF6454DD8180499"
    ));
    // This is the sum of (x1, y1) and (x2, y2).
    let x3 = IntModN::from_le_bytes(&hex!(
        "BE675E31F8AC8200CBCC6B10CECCD6EB93FB07D99BB9E7C99CC9245C862D3AF2"
    ));
    let y3 = IntModN::from_le_bytes(&hex!(
        "B44573B48FD3416DD256A8C0E1BAD03E88A78BF176778682589B9CB478FC1D79"
    ));
    // This is the double of (x2, y2).
    let x4 = IntModN::from_le_bytes(&hex!(
        "3BFFFFFF32333333333333333333333333333333333333333333333333333333"
    ));
    let y4 = IntModN::from_le_bytes(&hex!(
        "AC54ECC4254A4EDCAB10CC557A9811ED1EF7CB8AFDC64820C6803D2C5F481639"
    ));

    let mut p1 = black_box(EcPointN {
        x: x1.clone(),
        y: y1.clone(),
    });
    let mut p2 = black_box(EcPointN { x: x2, y: y2 });

    // Generic add can handle equal or unequal points.
    let p3 = p1.add(&p2);
    if p3.x != x3 || p3.y != y3 {
        panic!();
    }
    let p4 = p2.add(&p2);
    if p4.x != x4 || p4.y != y4 {
        panic!();
    }

    // Add assign and double assign
    p1.add_assign(&p2);
    if p1.x != x3 || p1.y != y3 {
        panic!();
    }
    p2.double_assign();
    if p2.x != x4 || p2.y != y4 {
        panic!();
    }

    // Ec Mul
    let p1 = black_box(EcPointN { x: x1, y: y1 });
    let scalar = IntModN::from_u32(12345678);
    // Calculated with https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#ec-multiply-tool
    let x5 = IntModN::from_le_bytes(&hex!(
        "194A93387F790803D972AF9C4A40CB89D106A36F58EE2F31DC48A41768216D6D"
    ));
    let y5 = IntModN::from_le_bytes(&hex!(
        "9E272F746DA7BED171E522610212B6AEEAAFDB2AD9F4B530B8E1B27293B19B2C"
    ));
    let result = EcPointN::msm(&[scalar], &[p1]);
    if result.x != x5 || result.y != y5 {
        panic!();
    }
}
