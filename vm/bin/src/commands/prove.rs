use std::time::Instant;

use afs_stark_backend::{
    keygen::types::MultiStarkPartialProvingKey, prover::trace::TraceCommitmentBuilder,
};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use stark_vm::vm::config::VmConfig;

use crate::{
    commands::{read_from_path, write_bytes},
    isa::get_vm,
};

/// `afs prove` command
/// Uses information from config.toml to generate a proof of the changes made by a .afi file to a table
/// saves the proof in `output-folder` as */prove.bin.
#[derive(Debug, Parser)]
pub struct ProveCommand {
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
        println!("Proving program: {}", self.isa_file_path);
        let vm = get_vm::<BabyBearPoseidon2Config>(config, &self.isa_file_path)?;

        let engine = config::baby_bear_poseidon2::default_engine(vm.max_log_degree());
        let encoded_pk = read_from_path(self.keys_folder.clone() + "/partial.pk").unwrap();
        let partial_pk: MultiStarkPartialProvingKey<BabyBearPoseidon2Config> =
            bincode::deserialize(&encoded_pk).unwrap();

        let partial_vk = partial_pk.partial_vk();

        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        for trace in vm.traces() {
            trace_builder.load_trace(trace);
        }
        trace_builder.commit_current();

        let main_trace_data = trace_builder.view(&partial_vk, vm.chips());

        let mut challenger = engine.new_challenger();
        let proof = prover.prove(
            &mut challenger,
            &partial_pk,
            main_trace_data,
            &vec![vec![]; vm.chips().len()],
        );

        let encoded_proof: Vec<u8> = bincode::serialize(&proof).unwrap();
        let proof_path = self.isa_file_path.clone() + ".prove.bin";
        write_bytes(&encoded_proof, proof_path).unwrap();
        Ok(())
    }
}
