use std::{path::Path, time::Instant};

use afs_stark_backend::{keygen::types::MultiStarkPartialVerifyingKey, prover::types::Proof};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use stark_vm::vm::{config::VmConfig, get_chips, VirtualMachine};

use crate::{
    asm::parse_asm_file,
    commands::{read_from_path, WORD_SIZE},
};

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
        let instructions = parse_asm_file(Path::new(&self.asm_file_path))?;
        let mut vm = VirtualMachine::<WORD_SIZE, _>::new(config, instructions, vec![]);
        let encoded_vk = read_from_path(&Path::new(&self.keys_folder).join("partial.vk"))?;
        let partial_vk: MultiStarkPartialVerifyingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_vk)?;

        let encoded_proof = read_from_path(Path::new(&self.proof_file))?;
        let proof: Proof<BabyBearPoseidon2Config> = bincode::deserialize(&encoded_proof)?;

        let engine = config::baby_bear_poseidon2::default_engine(vm.max_log_degree()?);

        let chips = get_chips(&vm);
        let num_chips = chips.len();

        let mut challenger = engine.new_challenger();
        let verifier = engine.verifier();
        let result = verifier.verify(
            &mut challenger,
            &partial_vk,
            chips,
            &proof,
            &vec![vec![]; num_chips],
        );

        if result.is_err() {
            println!("Verification Unsuccessful");
        } else {
            println!("Verification Succeeded!");
        }
        Ok(())
    }
}
