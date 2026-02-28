use halo2_base::halo2_proofs::dev::MockProver;

use super::*;

#[test]
fn scaffold_mockprover_smoke_test() {
    let circuit = StaticVerifierCircuit::default();
    let (builder, params) = circuit.prepare_mock_builder();
    StaticVerifierCircuit::assert_phase0_shape(&params);

    MockProver::run(circuit.shape.k as u32, &builder, vec![vec![]])
        .expect("mock prover should initialize")
        .assert_satisfied();
}
