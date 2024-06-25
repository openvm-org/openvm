use crate::commands::run::{PageConfig, RunCommand};
use color_eyre::eyre::Result;
use logical_interface::{afs_input::operation::InnerJoinOp, mock_db::MockDb};
use p3_uni_stark::StarkGenericConfig;

pub fn execute_inner_join<SC: StarkGenericConfig>(
    config: &PageConfig,
    cli: &RunCommand,
    db: &mut MockDb,
    op: InnerJoinOp,
) -> Result<()> {
    println!("inner_join: {:?}", op);
    Ok(())
}
