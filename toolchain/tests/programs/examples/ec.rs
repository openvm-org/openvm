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
        "EE7A67E785D057CBF6DDF7852D8AC46241BF22850686BD363B646C6EA02F8124"
    ));
    let x2 = IntModN::from_u32(2);
    let y2 = IntModN::from_bytes(hex!(
        "1D8A748A8F970EEA3E5244ADB50A3BDB71301A5F82765A06F16F54D48D814099"
    ));
    // This is the sum of (x1, y1) and (x2, y2).
    let x3 = IntModN::from_bytes(hex!(
        "EB76E5138FCA2800BCCCB601ECCC6DBE39BF709DB99B7E9CC99C42C568D2A32F"
    ));
    let y3 = IntModN::from_bytes(hex!(
        "4B54374BF83D14D62D658A0C1EAB0DE3887AB81F6777682885B9C94B87CFD197"
    ));

    let p1 = EcPoint { x: x1, y: y1 };
    let p2 = EcPoint { x: x2, y: y2 };

    let p3 = EcPoint::add(&p1, &p2);

    if p3.x != x3 {
        axvm::process::panic();
    }
}
