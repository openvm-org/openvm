use axvm_instructions::exe::AxVmExe;
use p3_baby_bear::BabyBear;
use serde::{de::DeserializeOwned, Serialize};
use static_assertions::assert_impl_all;

assert_impl_all!(AxVmExe<BabyBear>: Serialize, DeserializeOwned);
