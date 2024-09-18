/// Run with RANDOM_SRS=1 if you don't want to download the SRS.
use ax_sdk::config::{baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig, setup_tracing};

use crate::{halo2::testing_utils::run_evm_verifier_e2e_test, tests::fibonacci_stark_for_test};

#[test]
fn fibonacci_evm_verifier_e2e() {
    setup_tracing();
    run_evm_verifier_e2e_test(
        &fibonacci_stark_for_test::<BabyBearPoseidon2OuterConfig>(16),
        None,
    )
}
