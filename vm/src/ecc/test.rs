use ax_sdk::config::setup_tracing;
use p3_baby_bear::BabyBear;

use crate::arch::testing::MachineChipTestBuilder;

#[test]
fn test_ec_add() {
    setup_tracing();

    let mut tester: MachineChipTestBuilder<BabyBear> = MachineChipTestBuilder::default();
}
