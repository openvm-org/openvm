#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use eyre::Result;
    use openvm_circuit::arch::{SystemConfig, DEFAULT_MAX_NUM_PUBLIC_VALUES};
    use openvm_native_compiler::conversion::CompilerOptions;
    use openvm_sdk::{
        commit::AppExecutionCommit,
        config::{AggStarkConfig, AppConfig, SdkSystemConfig, SdkVmConfig},
        keygen::AggStarkProvingKey,
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
        let sdk = Sdk::new();
        let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        pkg_dir.pop();
        pkg_dir.pop();
        pkg_dir.push("crates/sdk/guest/fib");

        let vm_config = SdkVmConfig::builder()
            .system(SdkSystemConfig {
                config: SystemConfig::default().with_continuations(),
            })
            .rv32i(Default::default())
            .rv32m(Default::default())
            .io(Default::default())
            .native(Default::default())
            .build();
        assert!(vm_config.system.config.continuation_enabled);
        let elf = sdk.build(
            Default::default(),
            &vm_config,
            pkg_dir,
            &Default::default(),
            None,
        )?;

        let app_exe = sdk.transpile(elf, vm_config.transpiler())?;
        let fri_params = FriParameters::new_for_testing(LEAF_LOG_BLOWUP);
        let app_config =
            AppConfig::new_with_leaf_fri_params(fri_params, vm_config.clone(), fri_params);

        let app_pk = sdk.app_keygen(app_config.clone())?;
        let committed_app_exe = sdk.commit_app_exe(fri_params, app_exe.clone())?;

        let commits =
            AppExecutionCommit::compute(&vm_config, &committed_app_exe, &app_pk.leaf_committed_exe);
        let exe_commit = commits.app_exe_commit.to_u32_digest();
        let vm_commit = commits.app_vm_commit.to_u32_digest();

        let agg_pk = AggStarkProvingKey::keygen(AggStarkConfig {
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
        });
        let asm = sdk.generate_root_verifier_asm(&agg_pk);
        let asm_path = format!(
            "{}/examples/verify_openvm_stark/{}",
            env!("CARGO_MANIFEST_DIR"),
            ASM_FILENAME
        );
        std::fs::write(asm_path, asm)?;

        let e2e_stark_proof = sdk.generate_e2e_stark_proof(
            Arc::new(app_pk),
            committed_app_exe,
            agg_pk,
            StdIn::default(),
        )?;

        let verify_exe = {
            let mut pkg_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
            pkg_dir.push("examples/verify_openvm_stark");
            let elf = sdk.build(
                Default::default(),
                &vm_config,
                pkg_dir,
                &Default::default(),
                None,
            )?;
            sdk.transpile(elf, vm_config.transpiler())?
        };

        // app_exe publishes 7th and 8th fibonacci numbers.
        let pvs: Vec<u8> = [13u32, 21, 0, 0, 0, 0, 0, 0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();

        let mut stdin = StdIn::default();
        let key =
            compute_hint_key_for_verify_openvm_stark(ASM_FILENAME, &exe_commit, &vm_commit, &pvs);
        let value = encode_proof_to_kv_store_value(&e2e_stark_proof.proof);
        stdin.add_key_value(key, value);

        stdin.write(&exe_commit);
        stdin.write(&vm_commit);
        stdin.write(&pvs);

        sdk.execute(verify_exe, vm_config, stdin)?;

        Ok(())
    }
}
