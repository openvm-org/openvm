use afs_chips::single_page_index_scan::page_index_scan_input::Comp;
use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use color_eyre::eyre::Result;

use super::common::{execute_predicate_common, CommonCommands};

#[derive(Debug, Parser)]
pub struct LteCommand {
    #[command(flatten)]
    pub args: CommonCommands,
}

/// `lte` command
impl LteCommand {
    /// Execute the `lte` command
    pub fn execute(self, config: &PageConfig) -> Result<()> {
        println!(
            "Running `index <= {}` command on table: {}",
            self.args.value, self.args.table_id
        );
        execute_predicate_common(self.args, config, Comp::Lte)
    }
}
