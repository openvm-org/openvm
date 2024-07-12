use std::{path::Path, time::Instant};

use afs_stark_backend::{
    keygen::types::MultiStarkProvingKey, prover::trace::TraceCommitmentBuilder,
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use stark_vm::vm::{config::VmParamsConfig, get_chips, VirtualMachine};

use crate::{
    asm::parse_asm_file,
    commands::{read_from_path, write_bytes, WORD_SIZE},
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
    pub fn execute(&self, config: VmParamsConfig) -> Result<()> {
        let start = Instant::now();
        self.execute_helper(config)?;

        let duration = start.elapsed();
        println!("Proved table operations in {:?}", duration);

        Ok(())
    }

    pub fn execute_helper(&self, config: VmParamsConfig) -> Result<()> {
        println!("Proving program: {}", self.asm_file_path);
        let instructions = parse_asm_file(Path::new(&self.asm_file_path.clone()))?;
        let mut vm = VirtualMachine::<WORD_SIZE, _>::new(config, instructions, vec![]);

        let engine = config::baby_bear_poseidon2::default_engine(vm.segments[0].max_log_degree()?);
        let encoded_pk = read_from_path(&Path::new(&self.keys_folder.clone()).join("pk"))?;
        let pk: MultiStarkProvingKey<BabyBearPoseidon2Config> = bincode::deserialize(&encoded_pk)?;

        let vk = pk.vk();

        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        for trace in vm.segments[0].traces()? {
            trace_builder.load_trace(trace);
        }
        trace_builder.commit_current();

        let chips = get_chips(&vm.segments[0]);
        let num_chips = chips.len();

        let main_trace_data = trace_builder.view(&vk, chips);

        let mut challenger = engine.new_challenger();
        let proof = prover.prove(
            &mut challenger,
            &pk,
            main_trace_data,
            &vec![vec![]; num_chips],
        );

        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let proof_path = Path::new(&self.asm_file_path.clone()).with_extension("prove.bin");
        write_bytes(&encoded_proof, &proof_path)?;
        Ok(())
    }
}
