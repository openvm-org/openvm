use afs_chips::{
    group_by::group_by_input::GroupByOperation, single_page_index_scan::page_index_scan_input::Comp,
};

use crate::{
    afs_input::{
        operation::{InnerJoinOp, Operand, ReadOp, WhereOp, WriteOp},
        types::InputFileOp,
    },
    afs_interface::utils::string_to_table_id,
};

use super::{operation::GroupByOp, AfsInputFile, AfsOperation};

#[test]
pub fn test_read_file() {
    let file_path = "tests/data/test_input_file_32_1024.afi";
    let instructions = AfsInputFile::open(file_path).unwrap();
    let header = instructions.header;
    assert_eq!(
        header.table_id,
        "0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711"
    );
    assert_eq!(header.index_bytes, 32);
    assert_eq!(header.data_bytes, 1024);

    let operations = instructions.operations;
    assert_eq!(operations.len(), 5);
    assert_eq!(operations[0].operation, InputFileOp::Insert);
    assert_eq!(operations[1].operation, InputFileOp::Read);
    assert_eq!(operations[2].operation, InputFileOp::Insert);
    assert_eq!(operations[3].operation, InputFileOp::Insert);
    assert_eq!(operations[4].operation, InputFileOp::Read);
    assert_eq!(operations[0].args, ["18000001", "0x01"]);
    assert_eq!(operations[1].args, ["18000001"]);
    assert_eq!(
        operations[2].args,
        ["19000050", "0x69963768F8407dE501029680dE46945F838Fc98B"]
    );
    assert_eq!(
        operations[3].args,
        ["19000051", "0xe76a90E3069c9d86e666DcC687e76fcecf4429cF"]
    );
    assert_eq!(operations[4].args, ["19000051"]);
}

#[test]
pub fn test_parse_read_op() {
    let args = vec!["18000001".to_string()];
    let op = ReadOp::parse(args).unwrap();
    assert_eq!(
        op,
        ReadOp {
            index: "18000001".to_string(),
        }
    );
}

#[test]
pub fn test_parse_write_op() {
    let args = vec!["18000001".to_string(), "0x01".to_string()];
    let op = WriteOp::parse(args).unwrap();
    assert_eq!(
        op,
        WriteOp {
            index: "18000001".to_string(),
            data: "0x01".to_string()
        }
    );
}

#[test]
pub fn test_parse_insert_op() {
    let args = vec!["19000000".to_string(), "0x05".to_string()];
    let op = WriteOp::parse(args).unwrap();
    assert_eq!(
        op,
        WriteOp {
            index: "19000000".to_string(),
            data: "0x05".to_string()
        }
    );
}

#[test]
pub fn test_parse_where_op() {
    let args = vec![
        "0x15".to_string(),
        "INDEX".to_string(),
        "<".to_string(),
        "0x55".to_string(),
    ];
    let op = WhereOp::parse(args).unwrap();
    assert_eq!(
        op,
        WhereOp {
            table_id: string_to_table_id("0x15".to_string()),
            operand: Operand::Index,
            predicate: Comp::Lt,
            value: "0x55".to_string()
        }
    );
}

#[test]
pub fn test_parse_group_by_op() {
    let args = vec![
        "0x11".to_string(),
        "5".to_string(),
        "10".to_string(),
        "20".to_string(),
        "61".to_string(),
        "SUM".to_string(),
    ];
    let op = GroupByOp::parse(args).unwrap();
    assert_eq!(
        op,
        GroupByOp {
            table_id: string_to_table_id("0x11".to_string()),
            group_by_cols: vec![5, 10, 20],
            agg_col: 61,
            op: GroupByOperation::Sum,
        }
    );
}

#[test]
pub fn test_parse_inner_join_op() {
    let args = vec![
        "0x11".to_string(),
        "0x12".to_string(),
        "1".to_string(),
        "32".to_string(),
    ];
    let op = InnerJoinOp::parse(args).unwrap();
    assert_eq!(
        op,
        InnerJoinOp {
            table_id_left: string_to_table_id("0x11".to_string()),
            table_id_right: string_to_table_id("0x12".to_string()),
            fkey_start: 1,
            fkey_end: 32,
        }
    );
}

#[test]
pub fn test_generate_large_afi_rw() {
    let file_path = "tests/data/256_write_32_1024.afi";
    let mut instructions = AfsInputFile::new(file_path, "0x0a", 32, 1024);
    for i in 0..256 {
        instructions.add_operations(vec![AfsOperation {
            operation: InputFileOp::Insert,
            args: vec![format!("0x{:08x}", i), format!("0x{:08x}", i * 2)],
        }]);
    }
    instructions.save_to_file().unwrap();

    let file_path = "tests/data/256_read_32_1024.afi";
    let mut instructions = AfsInputFile::new(file_path, "0x0a", 32, 1024);
    for i in 0..256 {
        instructions.add_operations(vec![AfsOperation {
            operation: InputFileOp::Read,
            args: vec![format!("0x{:08x}", i)],
        }]);
    }
    instructions.save_to_file().unwrap();
}
