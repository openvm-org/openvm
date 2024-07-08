use std::{
    fs::{self},
    path::Path,
    time::Instant,
};

use afs_test_utils::{
    config::{self},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use itertools::Itertools;
use p3_matrix::Matrix;
use stark_vm::vm::{config::VmConfig, get_chips, VirtualMachine};

use crate::asm::parse_asm_file;

use super::{write_bytes, WORD_SIZE};

/// `afs keygen` command
/// Uses information from config.toml to generate partial proving and verifying keys and
/// saves them to the specified `output-folder` as *.partial.pk and *.partial.vk.
#[derive(Debug, Parser)]
pub struct KeygenCommand {
    #[arg(
        long = "asm-file",
        short = 'f',
        help = "The .asm file input",
        required = true
    )]
    pub asm_file_path: String,
    #[arg(
        long = "output-folder",
        short = 'o',
        help = "The folder to output the keys to",
        required = false,
        default_value = "keys"
    )]
    pub output_folder: String,
}

impl KeygenCommand {
    /// Execute the `keygen` command
    pub fn execute(self, config: VmConfig) -> Result<()> {
        let start = Instant::now();
        self.execute_helper(config)?;
        let duration = start.elapsed();
        println!("Generated keys in {:?}", duration);
        Ok(())
    }

    fn execute_helper(self, config: VmConfig) -> Result<()> {
        let instructions = parse_asm_file(Path::new(&self.asm_file_path.clone()))?;
        let mut vm = VirtualMachine::<WORD_SIZE, _>::new(config, instructions, vec![]);
        let engine = config::baby_bear_poseidon2::default_engine(vm.max_log_degree()?);
        let mut keygen_builder = engine.keygen_builder();

        let traces = vm.traces()?;
        let chips = get_chips(&vm);

        for (chip, trace) in chips.into_iter().zip_eq(traces) {
            keygen_builder.add_air(chip, trace.height(), 0);
        }

        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();
        let encoded_pk: Vec<u8> = bincode::serialize(&partial_pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&partial_vk)?;
        fs::create_dir_all(Path::new(&self.output_folder))?;
        let pk_path = Path::new(&self.output_folder).join("partial.pk");
        let vk_path = Path::new(&self.output_folder).join("partial.vk");
        fs::create_dir_all(self.output_folder)?;
        write_bytes(&encoded_pk, &pk_path)?;
        write_bytes(&encoded_vk, &vk_path)?;
        Ok(())
    }
}
