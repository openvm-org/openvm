#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::intrinsics::IntModN;
use axvm_ecc::sw::EcPoint;
use hex_literal::hex;

axvm::entry!(main);

pub fn main() {
    // Sample points got from https://asecuritysite.com/ecc/ecc_points2 and
    // https://learnmeabitcoin.com/technical/cryptography/elliptic-curve/#add
    let x1 = IntModN::from_u32(1);
    let y1 = IntModN::from_bytes(hex!(
        "EEA7767E580D75BC6FDD7F58D2A84C2614FB22586068DB63B346C6E60AF21842"
    ));
    let x2 = IntModN::from_u32(2);
    let y2 = IntModN::from_bytes(hex!(
        "D1A847A8F879E0AEE32544DA5BA0B3BD1703A1F52867A5601FF6454DD8180499"
    ));
    // This is the sum of (x1, y1) and (x2, y2).
    let x3 = IntModN::from_bytes(hex!(
        "BE675E31F8AC8200CBCC6B10CECCD6EB93FB07D99BB9E7C99CC9245C862D3AF2"
    ));
    let y3 = IntModN::from_bytes(hex!(
        "B44573B48FD3416DD256A8C0E1BAD03E88A78BF176778682589B9CB478FC1D79"
    ));
    // This is the double of (x2, y2).
    let x4 = IntModN::from_bytes(hex!(
        "3BFFFFFF32333333333333333333333333333333333333333333333333333333"
    ));
    let y4 = IntModN::from_bytes(hex!(
        "AC54ECC4254A4EDCAB10CC557A9811ED1EF7CB8AFDC64820C6803D2C5F481639"
    ));

    let p1 = EcPoint { x: x1, y: y1 };
    let p2 = EcPoint { x: x2, y: y2 };

    let p3 = EcPoint::add(&p1, &p2);

    if p3.x != x3 || p3.y != y3 {
        axvm::process::panic();
    }

    let p4 = EcPoint::add(&p2, &p2);

    if p4.x != x4 || p4.y != y4 {
        axvm::process::panic();
    }
}
