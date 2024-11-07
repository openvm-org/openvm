#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::intrinsics::IntModN;
use axvm_ecc::sw::EcPoint;
use hex_literal::hex;

axvm::entry!(main);

pub fn main() {
    let x1 = IntModN::from_u32(1);
    let y1 = IntModN::from_bytes(hex!(
        "4218F20AE6C646B363DB68605822FB14264CA8D2587FDD6FBC750D587E76A7EE"
    ));
    let x2 = BigUint::from_u32(2).unwrap();
    let y2 = IntModN::from_bytes(hex!(
        "990418D84D45F61F60A56728F5A10317BDB3A05BDA4425E3AEE079F8A847A8D1"
    ));
    let x3 = IntModN::from_bytes(hex!(
        "F23A2D865C24C99CC9E7B99BD907FB93EBD6CCCE106BCCCB0082ACF8315E67BE"
    ));
    let y3 = IntModN::from_bytes(hex!(
        "791DFC78B49C9B5882867776F18BA7883ED0BAE1C0A856D26D41D38FB47345B4"
    ));

    // let p1 = ??
}
