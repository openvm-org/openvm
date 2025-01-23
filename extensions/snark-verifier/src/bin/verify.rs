use openvm_algebra_guest::{complex_macros::complex_init, moduli_macros::moduli_init};
use openvm_ecc_guest::sw_macros::sw_init;
#[allow(unused_imports)]
use openvm_pairing_guest::bn254::Bn254G1Affine;
use openvm_snark_verifier::PlonkVerifierContext;

moduli_init! {
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
}

complex_init! {
    Bn254Fp2 { mod_idx = 0 },
}

sw_init! {
    Bn254G1Affine
}

fn main() {
    setup_all_moduli();
    setup_all_complex_extensions();
    setup_all_curves();

    let ctx: PlonkVerifierContext = openvm::io::read();
    ctx.verify().unwrap();
}
