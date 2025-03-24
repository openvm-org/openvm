use openvm_native_compiler::asm::AsmConfig;
use openvm_native_recursion::hints::Hintable;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    p3_field::{extension::BinomialExtensionField, ExtensionField, PrimeField32, TwoAdicField},
    proof::Proof,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Config,
        baby_bear_poseidon2_root::BabyBearPoseidon2RootConfig,
    },
    p3_baby_bear::BabyBear,
};

pub mod static_verifier;
pub mod verifier;

// pub type SC = BabyBearPoseidon2Config;
// pub type C = InnerConfig;
// pub type F = BabyBear;
// pub type RootSC = BabyBearPoseidon2RootConfig;

pub trait SdkConfig
where
    Self: Sized,
    Proof<Self::SC>: Hintable<C<Self>>,
{
    type F: PrimeField32 + TwoAdicField;
    type EF: ExtensionField<Self::F> + TwoAdicField;
    type SC: StarkGenericConfig;
    type RootSC: StarkGenericConfig;
}

pub type C<SdkC: SdkConfig> = AsmConfig<SdkC::F, SdkC::EF>;

pub struct BabyBearSdkConfig;

impl SdkConfig for BabyBearSdkConfig {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type SC = BabyBearPoseidon2Config;
    type RootSC = BabyBearPoseidon2RootConfig;
}
