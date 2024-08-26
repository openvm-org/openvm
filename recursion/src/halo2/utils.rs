use lazy_static::lazy_static;
use snark_verifier_sdk::{
    halo2::{PoseidonTranscript, POSEIDON_SPEC},
    snark_verifier::{
        halo2_base::halo2_proofs::{
            halo2curves::bn256::{Bn256, G1Affine},
            poly::{
                commitment::CommitmentScheme,
                kzg::commitment::{KZGCommitmentScheme, ParamsKZG},
            },
        },
        pcs::kzg::KzgDecidingKey,
        verifier::{plonk::PlonkProof, SnarkVerifier},
    },
    NativeLoader, PlonkVerifier, Snark, SHPLONK,
};

lazy_static! {
    // TODO: this should be dynamic. hard code for now.
    static ref SVK: G1Affine =
        serde_json::from_str("\"0100000000000000000000000000000000000000000000000000000000000000\"")
            .unwrap();

    /// TODO: this is also stored in the pinning jsons. We should read it from the pinning if possible.
    /// This commits to the trusted setup used to generate all proving keys.
    /// This MUST be updated whenever the trusted setup is changed.
    pub static ref DK: KzgDecidingKey<Bn256> = serde_json::from_str(r#"
          {
            "_marker": null,
            "g2": "edf692d95cbdde46ddda5ef7d422436779445c5e66006a42761e1f12efde0018c212f3aeb785e49712e7a9353349aaf1255dfb31b7bf60723a480d9293938e19",
            "s_g2": "0016e2a0605f771222637bae45148c8faebb4598ee98f30f20f790a0c3c8e02a7bf78bf67c4aac19dcc690b9ca0abef445d9a576c92ad6041e6ef1413ca92a17",
            "svk": {
              "g": "0100000000000000000000000000000000000000000000000000000000000000"
            }
          }
       "#).unwrap();
    /// Hacking because of bad interface. This is to construct a fake KZG params to pass Svk(which only requires ParamsKZG.g[0]) to AggregationCircuit.
    static ref FAKE_KZG_PARAMS: ParamsKZG<Bn256> = KZGCommitmentScheme::new_params(1);
    pub static ref KZG_PARAMS_FOR_SVK: ParamsKZG<Bn256> = build_kzg_params_for_svk(*SVK);
}

fn build_kzg_params_for_svk(g: G1Affine) -> ParamsKZG<Bn256> {
    FAKE_KZG_PARAMS.from_parts(
        1,
        vec![g],
        Some(vec![g]),
        Default::default(),
        Default::default(),
    )
}

#[allow(dead_code)]
pub(crate) fn verify_snark(dk: &KzgDecidingKey<Bn256>, snark: &Snark) {
    let mut transcript =
        PoseidonTranscript::<NativeLoader, &[u8]>::from_spec(snark.proof(), POSEIDON_SPEC.clone());
    let proof: PlonkProof<_, _, SHPLONK> =
        PlonkVerifier::read_proof(dk, &snark.protocol, &snark.instances, &mut transcript)
            .expect("Failed to read PlonkProof");
    PlonkVerifier::verify(dk, &snark.protocol, &snark.instances, &proof)
        .expect("PlonkVerifier failed");
}
