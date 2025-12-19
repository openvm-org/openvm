use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_sdk::{Sdk, F};
use openvm_stark_backend::p3_field::PrimeField32;

fn main() {
    let sdk = Sdk::standard();
    let (agg_pk, _agg_vk) = sdk.agg_keygen().expect("agg_keygen failed");

    let internal_commit: [F; DIGEST_SIZE] =
        agg_pk.internal_committed_exe.get_program_commit().into();
    let internal_u32 = internal_commit.map(|x| x.as_canonical_u32());
    println!("internal_program_commit_u32: {internal_u32:?}");
}
