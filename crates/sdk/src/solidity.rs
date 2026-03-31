use std::{
    fs::{create_dir_all, write},
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

use eyre::Context;
use serde_json::{json, Value};
use tempfile::tempdir;

use crate::{
    error::SdkError,
    fs::{
        EVM_HALO2_VERIFIER_BASE_NAME, EVM_HALO2_VERIFIER_INTERFACE_NAME,
        EVM_HALO2_VERIFIER_PARENT_NAME,
    },
    types::{EvmHalo2Verifier, EvmVerifierByteCode},
    OPENVM_VERSION,
};

const EVM_HALO2_VERIFIER_TEMPLATE: &str =
    include_str!("../contracts/template/OpenVmHalo2Verifier.sol");
const EVM_HALO2_VERIFIER_INTERFACE: &str =
    include_str!("../contracts/src/IOpenVmHalo2Verifier.sol");

alloy_sol_types::sol! {
    #[allow(missing_docs)]
    interface IOpenVmHalo2Verifier {
        function verify(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit, bytes32 appVmCommit) external view;
    }
}

/// Generate the EVM Halo2 verifier Solidity contract, compile it with solc, and return
/// the verifier artifact.
pub(crate) fn generate_halo2_verifier_solidity(
    halo2_pk: &crate::keygen::Halo2ProvingKey,
    halo2_params_reader: &crate::halo2_params::CacheHalo2ParamsReader,
) -> Result<EvmHalo2Verifier, SdkError> {
    let wrapper_k = halo2_pk.wrapper.pinning.metadata.config_params.k;
    let params = halo2_params_reader.read_params(wrapper_k);

    // Generate the base Halo2Verifier Solidity code from snark-verifier
    // (via the wrapper circuit, which is what produces the final EVM proof)
    let fallback_verifier = halo2_pk.wrapper.generate_fallback_evm_verifier(&params);
    let halo2_verifier_code = fallback_verifier.sol_code;

    // Compute public values length from the wrapper circuit's instances.
    // The wrapper's instances layout is:
    //   [0..12]: KZG accumulator
    //   [12]: app_exe_commit
    //   [13]: app_vk_commit
    //   [14..]: user public values
    let num_pvs = halo2_pk
        .wrapper
        .pinning
        .metadata
        .num_pvs
        .first()
        .expect("Expected at least one instance column");
    // Subtract 12 (accumulator) + 2 (commits) = 14
    let pvs_length = num_pvs
        .checked_sub(crate::types::NUM_BN254_ACCUMULATOR + 2)
        .expect("Unexpected number of wrapper circuit public values");

    assert!(
        pvs_length <= 8192,
        "OpenVM Halo2 verifier contract does not support more than 8192 public values"
    );

    // PROOF_DATA_LENGTH is now a constant in the template: (12 + 43) * 32
    // Fill out template placeholders
    let openvm_verifier_code = EVM_HALO2_VERIFIER_TEMPLATE
        .replace("{PUBLIC_VALUES_LENGTH}", &pvs_length.to_string())
        .replace("{OPENVM_VERSION}", OPENVM_VERSION);

    // Format Solidity code if forge-fmt is available (requires Rust 1.91+)
    let (formatted_interface, formatted_halo2_verifier_code, formatted_openvm_verifier_code) =
        format_solidity_sources(
            EVM_HALO2_VERIFIER_INTERFACE,
            &halo2_verifier_code,
            &openvm_verifier_code,
        );

    // Create temp dir
    let temp_dir = tempdir()
        .wrap_err("Failed to create temp dir")
        .map_err(SdkError::Other)?;
    let temp_path = temp_dir.path();
    let root_path = Path::new("src").join(format!("v{OPENVM_VERSION}"));

    // Make interfaces dir
    let interfaces_path = root_path.join("interfaces");

    // This will also create the dir for root_path, so no need to explicitly
    // create it
    create_dir_all(temp_path.join(&interfaces_path))?;

    let interface_file_path = interfaces_path.join(EVM_HALO2_VERIFIER_INTERFACE_NAME);
    let parent_file_path = root_path.join(EVM_HALO2_VERIFIER_PARENT_NAME);
    let base_file_path = root_path.join(EVM_HALO2_VERIFIER_BASE_NAME);

    // Write the files to the temp dir. This is only for compilation
    // purposes.
    write(temp_path.join(&interface_file_path), &formatted_interface)?;
    write(
        temp_path.join(&parent_file_path),
        &formatted_halo2_verifier_code,
    )?;
    write(
        temp_path.join(&base_file_path),
        &formatted_openvm_verifier_code,
    )?;

    // Run solc from the temp dir
    let solc_input = json!({
        "language": "Solidity",
        "sources": {
            interface_file_path.to_str().unwrap(): {
                "content": formatted_interface
            },
            parent_file_path.to_str().unwrap(): {
                "content": formatted_halo2_verifier_code
            },
            base_file_path.to_str().unwrap(): {
                "content": formatted_openvm_verifier_code
            }
        },
        "settings": {
            "remappings": ["forge-std/=lib/forge-std/src/"],
            "optimizer": {
                "enabled": true,
                "runs": 100000,
                "details": {
                    "constantOptimizer": false,
                    "yul": false
                }
            },
            "evmVersion": "paris",
            "viaIR": false,
            "outputSelection": {
                "*": {
                    "*": ["metadata", "evm.bytecode.object"]
                }
            }
        }
    });

    let mut child = Command::new("solc")
        .current_dir(temp_path)
        .arg("--standard-json")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn solc");

    child
        .stdin
        .as_mut()
        .expect("Failed to open stdin")
        .write_all(solc_input.to_string().as_bytes())
        .expect("Failed to write to stdin");

    let output = child.wait_with_output().expect("Failed to read output");

    if !output.status.success() {
        return Err(SdkError::Other(eyre::eyre!(
            "solc exited with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let parsed: Value =
        serde_json::from_slice(&output.stdout).map_err(|e| SdkError::Other(e.into()))?;

    let bytecode = parsed
        .get("contracts")
        .expect("No 'contracts' field found")
        .get(format!("src/v{OPENVM_VERSION}/OpenVmHalo2Verifier.sol"))
        .unwrap_or_else(|| panic!("No 'src/v{OPENVM_VERSION}/OpenVmHalo2Verifier.sol' field found"))
        .get("OpenVmHalo2Verifier")
        .expect("No 'OpenVmHalo2Verifier' field found")
        .get("evm")
        .expect("No 'evm' field found")
        .get("bytecode")
        .expect("No 'bytecode' field found")
        .get("object")
        .expect("No 'object' field found")
        .as_str()
        .expect("No 'object' field found");

    let bytecode = hex::decode(bytecode).expect("Invalid hex in Binary");

    let evm_verifier = EvmHalo2Verifier {
        halo2_verifier_code: formatted_halo2_verifier_code,
        openvm_verifier_code: formatted_openvm_verifier_code,
        openvm_verifier_interface: formatted_interface,
        artifact: EvmVerifierByteCode {
            sol_compiler_version: "0.8.19".to_string(),
            sol_compiler_options: solc_input.get("settings").unwrap().to_string(),
            bytecode,
        },
    };
    Ok(evm_verifier)
}

/// Verify an EVM Halo2 proof by deploying the verifier bytecode in a local EVM.
pub(crate) fn verify_evm_halo2_proof(
    openvm_verifier: &EvmHalo2Verifier,
    evm_proof: crate::types::EvmProof,
) -> Result<u64, SdkError> {
    // Convert EvmProof → RawEvmProof for the static verifier's evm_verify
    let raw_evm_proof: openvm_static_verifier::keygen::RawEvmProof = evm_proof.into();
    let deployment_code = &openvm_verifier.artifact.bytecode;

    let gas_cost = openvm_static_verifier::keygen::evm_verify(deployment_code, &raw_evm_proof)
        .map_err(|reason| SdkError::Other(eyre::eyre!("Sdk::verify_openvm_evm_proof: {reason}")))?;

    Ok(gas_cost)
}

/// Format Solidity sources using forge-fmt when available, or return them as-is.
fn format_solidity_sources(
    interface: &str,
    halo2_verifier: &str,
    openvm_verifier: &str,
) -> (String, String, String) {
    #[cfg(feature = "evm-verify-fmt")]
    {
        use forge_fmt::{
            format, FormatterConfig, IntTypes, MultilineFuncHeaderStyle, NumberUnderscore,
            QuoteStyle, SingleLineBlockStyle,
        };

        let config = FormatterConfig {
            line_length: 120,
            tab_width: 4,
            bracket_spacing: true,
            int_types: IntTypes::Long,
            multiline_func_header: MultilineFuncHeaderStyle::AttributesFirst,
            quote_style: QuoteStyle::Double,
            number_underscore: NumberUnderscore::Thousands,
            single_line_statement_blocks: SingleLineBlockStyle::Preserve,
            override_spacing: false,
            wrap_comments: false,
            ignore: vec![],
            contract_new_lines: false,
            sort_imports: false,
            ..Default::default()
        };

        let formatted_interface = format(interface, config.clone())
            .into_result()
            .expect("Failed to format interface");
        let formatted_halo2 = format(halo2_verifier, config.clone())
            .into_result()
            .expect("Failed to format halo2 verifier code");
        let formatted_openvm = format(openvm_verifier, config)
            .into_result()
            .expect("Failed to format openvm verifier code");

        (formatted_interface, formatted_halo2, formatted_openvm)
    }
    #[cfg(not(feature = "evm-verify-fmt"))]
    {
        (
            interface.to_string(),
            halo2_verifier.to_string(),
            openvm_verifier.to_string(),
        )
    }
}
