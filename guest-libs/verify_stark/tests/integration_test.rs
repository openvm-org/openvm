use std::path::PathBuf;

use eyre::Result;
use openvm_circuit::arch::{SystemConfig, DEFAULT_MAX_NUM_PUBLIC_VALUES};
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_sdk::{
    config::{AggregationConfig, AppConfig, SdkSystemConfig, SdkVmConfig},
    keygen::AggProvingKey,
    Sdk, StdIn,
};
use openvm_stark_sdk::config::FriParameters;
use openvm_verify_stark::host::{
    compute_hint_key_for_verify_openvm_stark, encode_proof_to_kv_store_value,
};

const LEAF_LOG_BLOWUP: usize = 2;
const INTERNAL_LOG_BLOWUP: usize = 3;
const ROOT_LOG_BLOWUP: usize = 4;

#[test]
fn test_verify_openvm_stark_e2e() -> Result<()> {
    const ASM_FILENAME: &str = "root_verifier.asm";
    let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    pkg_dir.pop();
    pkg_dir.pop();
    pkg_dir.push("crates/sdk/guest/fib");

    let vm_config = SdkVmConfig::builder()
        .system(SdkSystemConfig {
            config: SystemConfig::default(),
        })
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .native(Default::default())
        .build();
    let fri_params = FriParameters::new_for_testing(LEAF_LOG_BLOWUP);
    let app_config = AppConfig::new_with_leaf_fri_params(fri_params, vm_config.clone(), fri_params);
    let sdk = Sdk::new(app_config)?;
    assert!(vm_config.system.config.continuation_enabled);
    let elf = sdk.build(Default::default(), pkg_dir, &None, None)?;

    let (e2e_stark_proof, app_commit) = sdk.prove(elf, StdIn::default())?;
    let exe_commit = app_commit.app_exe_commit.to_u32_digest();
    let vm_commit = app_commit.app_vm_commit.to_u32_digest();

    let agg_pk = AggProvingKey::keygen(AggregationConfig {
        max_num_user_public_values: DEFAULT_MAX_NUM_PUBLIC_VALUES,
        leaf_fri_params: FriParameters::new_for_testing(LEAF_LOG_BLOWUP),
        internal_fri_params: FriParameters::new_for_testing(INTERNAL_LOG_BLOWUP),
        root_fri_params: FriParameters::new_for_testing(ROOT_LOG_BLOWUP),
        profiling: false,
        compiler_options: CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        },
        root_max_constraint_degree: (1 << ROOT_LOG_BLOWUP) + 1,
    })?;
    let _ = sdk.set_agg_pk(agg_pk);
    let asm = sdk.generate_root_verifier_asm();
    let asm_path = format!(
        "{}/examples/verify_openvm_stark/{}",
        env!("CARGO_MANIFEST_DIR"),
        ASM_FILENAME
    );
    std::fs::write(asm_path, asm)?;

    let verify_elf = {
        let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        pkg_dir.push("examples/verify_openvm_stark");
        sdk.build(Default::default(), pkg_dir, &None, None)?
    };

    // app_exe publishes 31st and 32nd fibonacci numbers.
    let pvs: Vec<u8> = [1346269, 2178309, 0, 0, 0, 0, 0, 0u32]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();

    let mut stdin = StdIn::default();
    let key = compute_hint_key_for_verify_openvm_stark(ASM_FILENAME, &exe_commit, &vm_commit, &pvs);
    let value = encode_proof_to_kv_store_value(&e2e_stark_proof.inner);
    stdin.add_key_value(key, value);

    stdin.write(&exe_commit);
    stdin.write(&vm_commit);
    stdin.write(&pvs);

    sdk.execute(verify_elf, stdin)?;

    Ok(())
}
