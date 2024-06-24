use std::time::Instant;

use afs_stark_backend::{keygen::types::MultiStarkPartialVerifyingKey, prover::types::Proof};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use stark_vm::vm::config::VMConfig;

use crate::{commands::read_from_path, isa::get_vm};

/// `afs verify` command
/// Uses information from config.toml to verify a proof using the verifying key in `output-folder`
/// as */prove.bin.
#[derive(Debug, Parser)]
pub struct VerifyCommand {
    #[arg(
        long = "proof-file",
        short = 'p',
        help = "The path to the proof file",
        required = true
    )]
    pub proof_file: String,

    #[arg(
        long = "isa-file",
        short = 'f',
        help = "The .isa file input",
        required = true
    )]
    pub isa_file_path: String,

    #[arg(
        long = "keys-folder",
        short = 'k',
        help = "The folder that contains keys",
        required = false,
        default_value = "keys"
    )]
    pub keys_folder: String,
}

impl VerifyCommand {
    /// Execute the `verify` command
    pub fn execute(&self, config: VMConfig) -> Result<()> {
        let start = Instant::now();

        self.execute_helper(config)?;

        let duration = start.elapsed();
        println!("Verified table operations in {:?}", duration);

        Ok(())
    }

    pub fn execute_helper(&self, config: VMConfig) -> Result<()> {
        println!("Verifying proof file: {}", self.proof_file);
        let vm = get_vm::<BabyBearPoseidon2Config>(config, &self.isa_file_path)?;
        // verify::verify_ops(&self.proof_file).await?;
        let encoded_vk = read_from_path(self.keys_folder.clone() + "/partial.vk").unwrap();
        let partial_vk: MultiStarkPartialVerifyingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_vk).unwrap();

        let encoded_proof = read_from_path(self.proof_file.clone()).unwrap();
        let proof: Proof<BabyBearPoseidon2Config> = bincode::deserialize(&encoded_proof).unwrap();

        let engine = config::baby_bear_poseidon2::default_engine(vm.max_log_degree());

        let mut challenger = engine.new_challenger();
        let verifier = engine.verifier();
        let result = verifier.verify(
            &mut challenger,
            partial_vk,
            vm.chips(),
            proof,
            &vec![vec![]; vm.chips().len()],
        );

        if result.is_err() {
            println!("Verification Unsuccessful");
        } else {
            println!("Verification Succeeded!");
        }
        Ok(())
    }
}
