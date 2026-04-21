use openvm_circuit::arch::testing::TestSC;
use openvm_circuit_primitives::Chip;
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::poseidon2::DeferralPoseidon2Chip;

#[test]
fn deferral_poseidon2_empty_trace() {
    let chip = DeferralPoseidon2Chip::<BabyBear>::new(Poseidon2Config::default());
    for _ in 0..2 {
        let ctx: AirProvingContext<CpuBackend<TestSC>> = chip.generate_proving_ctx(());
        assert_eq!(
            ctx.common_main.height(),
            0,
            "DeferralPoseidon2Chip with no records should return an empty trace",
        );
    }
}
