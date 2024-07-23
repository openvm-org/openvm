use afs_compiler::asm::AsmBuilder;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn test_cycle_tracker() {
    let mut builder = AsmBuilder::<F, EF>::default();

    // let x = builder.get(&bits, 0);
}
