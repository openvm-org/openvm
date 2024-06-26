use std::{
    fs::{self, File}, io::{BufWriter, Write}, path::Path, time::Instant
};

use afs_test_utils::{
    config::{self},
    engine::StarkEngine,
};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_matrix::Matrix;
use stark_vm::vm::{config::VmConfig, VirtualMachine};

use crate::isa::parse_isa_file;

/// `afs keygen` command
/// Uses information from config.toml to generate partial proving and verifying keys and
/// saves them to the specified `output-folder` as *.partial.pk and *.partial.vk.
#[derive(Debug, Parser)]
pub struct KeygenCommand {
    #[arg(
        long = "isa-file",
        short = 'f',
        help = "The .isa file input",
        required = true
    )]
    pub isa_file_path: String,
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
        let instructions = parse_isa_file(Path::new(&self.isa_file_path.clone()))?;
        let vm = VirtualMachine::new(config, instructions);
        let engine = config::baby_bear_poseidon2::default_engine(vm.max_log_degree());
        let mut keygen_builder = engine.keygen_builder();

        let chips = vm.chips();
        let traces = vm.traces();

        for (chip, trace) in chips.into_iter().zip(traces) {
            keygen_builder.add_air(chip, trace.height(), 0);
        }

        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();
        let encoded_pk: Vec<u8> = bincode::serialize(&partial_pk)?;
        let encoded_vk: Vec<u8> = bincode::serialize(&partial_vk)?;
        fs::create_dir_all(Path::new(&self.output_folder.clone()))?;
        let pk_path = Path::new(&self.output_folder).join("partial.pk");
        let vk_path = Path::new(&self.output_folder).join("partial.vk");
        fs::create_dir_all(self.output_folder)?;
        write_bytes(&encoded_pk, &pk_path)?;
        write_bytes(&encoded_vk, &vk_path)?;
        Ok(())
    }
}

fn write_bytes(bytes: &[u8], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    Ok(())
}
