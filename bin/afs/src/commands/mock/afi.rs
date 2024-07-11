use clap::Parser;
use color_eyre::eyre::Result;
<<<<<<< HEAD
use logical_interface::afs_input_instructions::AfsInputInstructions;
=======
use logical_interface::afs_input::AfsInputFile;
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b

#[derive(Debug, Parser)]
pub struct AfiCommand {
    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
}

/// `mock afi` subcommand
impl AfiCommand {
    /// Execute the `mock afi` command
<<<<<<< HEAD
    pub fn execute(self) -> Result<()> {
        println!("afi_file_path: {}", self.afi_file_path);
        let instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;
=======
    pub fn execute(&self) -> Result<()> {
        println!("afi_file_path: {}", self.afi_file_path);
        let instructions = AfsInputFile::open(&self.afi_file_path)?;
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b
        if !self.silent {
            println!("{:?}", instructions.header);
            for op in instructions.operations {
                println!("{:?}", op);
            }
        }
        Ok(())
    }
}
