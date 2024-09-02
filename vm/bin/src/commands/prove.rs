use std::{ops::Deref, path::Path, time::Instant};

use afs_stark_backend::{
    keygen::types::MultiStarkProvingKey, prover::trace::TraceCommitmentBuilder,
};
use ax_sdk::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use stark_vm::{
    program::Program,
    vm::{config::VmConfig, VirtualMachine},
};

use crate::{
    asm::parse_asm_file,
    commands::{read_from_path, write_bytes},
};

/// `afs prove` command
/// Uses information from config.toml to generate a proof of the changes made by a .afi file to a table
/// saves the proof in `output-folder` as */prove.bin.
#[derive(Debug, Parser)]
pub struct ProveCommand {
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

impl ProveCommand {
    /// Execute the `prove` command
    pub fn execute(&self, config: VmConfig) -> Result<()> {
        let start = Instant::now();
        self.execute_helper(config)?;

        let duration = start.elapsed();
        println!("Proved table operations in {:?}", duration);

        Ok(())
    }

    pub fn execute_helper(&self, config: VmConfig) -> Result<()> {
        println!("Proving program: {}", self.asm_file_path);
        let instructions = parse_asm_file(Path::new(&self.asm_file_path.clone()))?;
        let program_len = instructions.len();
        let program = Program {
            instructions,
            debug_infos: vec![None; program_len],
        };
        let vm = VirtualMachine::new(config, program, vec![]);

        let result = vm.execute_and_generate()?;
        assert_eq!(
            result.segment_results.len(),
            1,
            "continuations not currently supported"
        );
        let result = result.segment_results.into_iter().next().unwrap();

        let engine = config::baby_bear_poseidon2::default_engine(result.max_log_degree());
        let encoded_pk = read_from_path(&Path::new(&self.keys_folder.clone()).join("pk"))?;
        let pk: MultiStarkProvingKey<BabyBearPoseidon2Config> = bincode::deserialize(&encoded_pk)?;

        let vk = pk.vk();

        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        for trace in result.traces {
            trace_builder.load_trace(trace);
        }
        trace_builder.commit_current();

        let airs = result.airs.iter().map(Box::deref).collect();

        let main_trace_data = trace_builder.view(&vk, airs);

        let mut challenger = engine.new_challenger();
        let proof = prover.prove(&mut challenger, &pk, main_trace_data, &result.public_values);

        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let proof_path = Path::new(&self.asm_file_path.clone()).with_extension("prove.bin");
        write_bytes(&encoded_proof, &proof_path)?;
        Ok(())
    }
}
