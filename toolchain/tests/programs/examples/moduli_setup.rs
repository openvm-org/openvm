#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

axvm::entry!(main);
axvm::moduli_setup! {
    bls12381 = "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787";
}

pub fn main() {
    let x = IntMod_bls12381::from_bytes(core::array::from_fn(|i| i as u8));
    assert_eq!(x.0.len(), 48);
    core::hint::black_box(AXIOM_SERIALIZED_MODULI);
}
