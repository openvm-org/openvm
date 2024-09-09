use std::{path::Path, time::Instant};

use afs_stark_backend::{keygen::types::MultiStarkVerifyingKey, prover::types::Proof};
use ax_sdk::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use stark_vm::{
    program::Program,
    vm::{config::VmConfig, VirtualMachine},
};

use crate::{asm::parse_asm_file, commands::read_from_path};

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
        long = "asm-file",
        short = 'f',
        help = "The .asm file input",
        required = true
    )]
    pub asm_file_path: String,

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
    pub fn execute(&self, config: VmConfig) -> Result<()> {
        let start = Instant::now();

        self.execute_helper(config)?;

        let duration = start.elapsed();
        println!("Verified table operations in {:?}", duration);

        Ok(())
    }

    pub fn execute_helper(&self, config: VmConfig) -> Result<()> {
        println!("Verifying proof file: {}", self.proof_file);
        let instructions = parse_asm_file::<BabyBear>(Path::new(&self.asm_file_path))?;
        let program_len = instructions.len();
        let program = Program {
            instructions,
            debug_infos: vec![None; program_len],
        };
        let vm = VirtualMachine::new(config, program, vec![]);
        let encoded_vk = read_from_path(&Path::new(&self.keys_folder).join("vk"))?;
        let vk: MultiStarkVerifyingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_vk)?;

        let encoded_proof = read_from_path(Path::new(&self.proof_file))?;
        let proof: Proof<BabyBearPoseidon2Config> = bincode::deserialize(&encoded_proof)?;

        // FIXME: verify should not have to execute
        let result = vm.execute_and_generate::<BabyBearPoseidon2Config>()?;
        assert_eq!(
            result.segment_results.len(),
            1,
            "continuations not currently supported"
        );
        let result = result.segment_results.into_iter().next().unwrap();

        let engine = config::baby_bear_poseidon2::default_engine(result.max_log_degree());

        let mut challenger = engine.new_challenger();
        let verifier = engine.verifier();
        let result = verifier.verify(&mut challenger, &vk, &proof, &result.public_values);

        if result.is_err() {
            println!("Verification Unsuccessful");
        } else {
            println!("Verification Succeeded!");
        }
        Ok(())
    }
}
