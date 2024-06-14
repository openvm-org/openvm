use afs_test_utils::page_config::PageConfig;
use alloy_primitives::U256;
use clap::Parser;
use color_eyre::eyre::Result;

use crate::utils::get_table_from_db;

use super::common::CommonCommands;

#[derive(Debug, Parser)]
pub struct EqCommand {
    #[command(flatten)]
    pub args: CommonCommands,
}

/// `eq` command
impl EqCommand {
    /// Execute the `eq` command
    pub fn execute(self, config: &PageConfig) -> Result<()> {
        println!(
            "Running `index == {}` command on table: {}",
            self.args.value, self.args.table_id
        );
        let table = get_table_from_db::<U256, U256>(self.args.table_id, self.args.db_file_path);
        let _page = table.to_page(config.page.height);
        let _predicate = "==";

        // Handle the page and predicate in ZK and get the resulting page back

        // Convert back to a Table

        // Save the output to a file or print it

        Ok(())
    }
}
