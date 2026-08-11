use openvm_circuit::arch::VmCircuitConfig;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    extension::Poseidon2Rv32Config, periphery::Poseidon2PeripheryAir, permute::Poseidon2PermuteAir,
};

type SC = BabyBearPoseidon2Config;
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

/// Pins the index arithmetic that `permute::execution::execute_e2_impl` relies on: it is handed
/// the permute AIR's index as `chip_idx` and derives the periphery AIR's index as `chip_idx + 1`,
/// so the two AIRs must be adjacent in that order in verifying-key order.
///
/// `extend_circuit` arranges this by adding the periphery AIR immediately before the permute AIR,
/// which [`openvm_circuit::arch::AirInventory::into_airs`] then reverses. Inserting any AIR between
/// them would make metered execution attribute periphery rows to an unrelated AIR, mis-sizing
/// segments with nothing else failing.
#[test]
fn periphery_air_immediately_follows_permute_air_in_vk_order() {
    let config = Poseidon2Rv32Config::default();
    let airs: Vec<_> = <Poseidon2Rv32Config as VmCircuitConfig<SC>>::create_airs(&config)
        .unwrap()
        .into_airs()
        .collect();

    let index_of = |pred: &dyn Fn(&dyn std::any::Any) -> bool| {
        airs.iter()
            .position(|air| pred(air.as_any()))
            .expect("AIR should be present in the circuit")
    };
    let permute_idx = index_of(&|any| any.is::<Poseidon2PermuteAir>());
    let periphery_idx = index_of(&|any| any.is::<Poseidon2PeripheryAir<F>>());

    assert_eq!(
        periphery_idx,
        permute_idx + 1,
        "periphery AIR must directly follow the permute AIR in VK order; \
         see `execute_e2_impl`'s `periphery_air_idx = op_air_idx + 1`"
    );
}
