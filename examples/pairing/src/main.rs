// [!region pre]
use hex_literal::hex;
// [!endregion pre]
// [!region imports]
use openvm_algebra_guest::{field::FieldExtension, IntMod};
use openvm_ecc_guest::AffinePoint;
use openvm_pairing::{
    bls12_381::{Bls12_381, Fp, Fp2},
    PairingCheck,
};
// [!endregion imports]

// [!region init]
openvm::init!();
/* The init! macro will expand to the following
openvm_algebra_moduli_macros::moduli_init! {
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
    "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
}

openvm_algebra_complex_macros::complex_init! {
    "Bls12_381Fp2" { mod_idx = 0 },
}
*/
// [!endregion init]

// [!region main]
pub fn main() {
    let p0 = AffinePoint::new(
        Fp::from_be_bytes_unchecked(&hex!("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb")),
        Fp::from_be_bytes_unchecked(&hex!("08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"))
    );
    let p1 = AffinePoint::new(
        Fp2::from_coeffs([
            Fp::from_be_bytes_unchecked(&hex!("1638533957d540a9d2370f17cc7ed5863bc0b995b8825e0ee1ea1e1e4d00dbae81f14b0bf3611b78c952aacab827a053")),
            Fp::from_be_bytes_unchecked(&hex!("0a4edef9c1ed7f729f520e47730a124fd70662a904ba1074728114d1031e1572c6c886f6b57ec72a6178288c47c33577"))
        ]),
        Fp2::from_coeffs([
            Fp::from_be_bytes_unchecked(&hex!("0468fb440d82b0630aeb8dca2b5256789a66da69bf91009cbfe6bd221e47aa8ae88dece9764bf3bd999d95d71e4c9899")),
            Fp::from_be_bytes_unchecked(&hex!("0f6d4552fa65dd2638b361543f887136a43253d9c66c411697003f7a13c308f5422e1aa0a59c8967acdefd8b6e36ccf3"))
        ]),
    );
    let q0 = AffinePoint::new(
        Fp::from_be_bytes_unchecked(&hex!("0572cbea904d67468808c8eb50a9450c9721db309128012543902d0ac358a62ae28f75bb8f1c7c42c39a8c5529bf0f4e")),
        Fp::from_be_bytes_unchecked(&hex!("166a9d8cabc673a322fda673779d8e3822ba3ecb8670e461f73bb9021d5fd76a4c56d9d4cd16bd1bba86881979749d28"))
    );
    let q1 = AffinePoint::new(
        Fp2::from_coeffs([
            Fp::from_be_bytes_unchecked(&hex!("024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8")),
            Fp::from_be_bytes_unchecked(&hex!("13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e"))
        ]),
        Fp2::from_coeffs([
            Fp::from_be_bytes_unchecked(&hex!("0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801")),
            Fp::from_be_bytes_unchecked(&hex!("0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be"))
        ]),
    );

    // [!region pairing_check]
    let res = Bls12_381::pairing_check(&[p0, -q0], &[p1, q1]);
    assert!(res.is_ok());
    // [!endregion pairing_check]
}
// [!endregion main]
