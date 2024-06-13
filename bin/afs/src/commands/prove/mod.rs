use afs_chips::page_rw_checker::page_controller::{OpType, Operation};
use alloy_primitives::{Uint, U256};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions, AfsOperation},
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::{
        codec::{self, fixed_bytes::FixedBytesCodec},
        types::TableMetadata,
        Table,
    },
    types::{Data, Index},
    utils::string_to_fixed_bytes_be_vec,
};

/// `afs prove` command
#[derive(Debug, Parser)]
pub struct ProveCommand {
    #[arg(long = "table-id", short = 't', help = "The table ID", required = true)]
    pub table_id: String,

    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "DB file input (default: new empty DB)",
        required = false
    )]
    pub db_file_path: Option<String>,

    #[arg(
        long = "output-file",
        short = 'o',
        help = "The path to the output file",
        required = false,
        default_value = "output/prove.bin"
    )]
    pub output_file: String,
}

impl ProveCommand {
    /// Execute the `prove` command
    pub fn execute(self) -> Result<()> {
        println!("Proving ops file: {}", self.afi_file_path);
        // prove::prove_ops(&self.ops_file).await?;
        let instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;
        let mut db = if let Some(db_file_path) = self.db_file_path {
            println!("db_file_path: {}", db_file_path);
            MockDb::from_file(&db_file_path)
        } else {
            let default_table_metadata = TableMetadata::new(32, 1024);
            MockDb::new(default_table_metadata)
        };
        let codec = FixedBytesCodec::new(
            db.default_table_metadata.index_bytes,
            db.default_table_metadata.data_bytes,
        );
        let mut interface = AfsInterface::<U256, U256>::new(&mut db);
        let zk_ops = instructions
            .operations
            .iter()
            .enumerate()
            .map(|(i, op)| afi_op_conv(op, self.table_id.clone(), &mut interface, i + 1, &codec))
            .collect::<Vec<_>>();

        Ok(())
    }
}

fn afi_op_conv<I, D>(
    afi_op: &AfsOperation,
    table_id: String,
    interface: &mut AfsInterface<I, D>,
    clk: usize,
    codec: &FixedBytesCodec<I, D>,
) -> Operation
where
    I: Index,
    D: Data,
{
    let idx_u8 = string_to_fixed_bytes_be_vec(afi_op.args[0].clone(), 32);
    let idx_u16 = vec_u8_to_vec_u16(&idx_u8);
    let idx = codec.fixed_bytes_to_index(idx_u8.clone());
    match afi_op.operation {
        InputFileBodyOperation::Read => {
            assert!(afi_op.args.len() == 1);
            let data = interface
                .read(table_id, codec.fixed_bytes_to_index(idx_u8))
                .unwrap();
            let data_bytes = codec.data_to_fixed_bytes(data.clone());
            let data_u16 = vec_u8_to_vec_u16(&data_bytes);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Read,
            }
        }
        InputFileBodyOperation::Insert => {
            assert!(afi_op.args.len() == 2);
            let data_u8 = string_to_fixed_bytes_be_vec(afi_op.args[1].clone(), D::MEMORY_SIZE);
            let data_u16 = vec_u8_to_vec_u16(&data_u8);
            let data = codec.fixed_bytes_to_data(data_u8);
            interface.insert(table_id, idx, data);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
        InputFileBodyOperation::Write => {
            assert!(afi_op.args.len() == 2);
            let data_u8 = string_to_fixed_bytes_be_vec(afi_op.args[1].clone(), D::MEMORY_SIZE);
            let data_u16 = vec_u8_to_vec_u16(&data_u8);
            let data = codec.fixed_bytes_to_data(data_u8);
            interface.insert(table_id, idx, data);
            Operation {
                clk,
                idx: idx_u16,
                data: data_u16,
                op_type: OpType::Write,
            }
        }
    }
}

// while the output is u32, each element is less than 2 << 16.
fn vec_u8_to_vec_u16(bytes: &Vec<u8>) -> Vec<u32> {
    let mut ans = vec![];
    for i in 0..(bytes.len() + 1) / 2 {
        if bytes.len() % 2 == 1 && i == 0 {
            ans.push(bytes[0] as u32);
        } else {
            ans.push((256 * bytes[2 * i + 1] + bytes[2 * i]) as u32);
        }
    }
    ans
}

fn table_to_page