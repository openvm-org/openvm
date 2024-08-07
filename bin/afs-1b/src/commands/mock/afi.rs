use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::AfsInputFile;

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
    pub fn execute(self) -> Result<()> {
        println!("afi_file_path: {}", self.afi_file_path);
        let instructions = AfsInputFile::open(&self.afi_file_path)?;
        if !self.silent {
            println!("{:?}", instructions.header);
            for op in instructions.operations {
                println!("{:?}", op);
            }
        }
        Ok(())
    }
}
