use crate::afs_input_instructions::types::InputFileBodyOperation;

use super::{AfsInputInstructions, AfsOperation};

#[test]
pub fn test_read_file() {
    let file_path = "tests/data/test_input_file_32_1024.afi";
    let instructions = AfsInputInstructions::from_file(file_path).unwrap();
    let header = instructions.header;
    assert_eq!(
        header.table_id,
        "0x155687649d5789a399211641b38bb93139f8ceca042466aa98e500a904657711"
    );
    assert_eq!(header.index_bytes, 32);
    assert_eq!(header.data_bytes, 1024);

    let operations = instructions.operations;
    assert_eq!(operations.len(), 5);
    assert_eq!(operations[0].operation, InputFileBodyOperation::Insert);
    assert_eq!(operations[1].operation, InputFileBodyOperation::Read);
    assert_eq!(operations[2].operation, InputFileBodyOperation::Insert);
    assert_eq!(operations[3].operation, InputFileBodyOperation::Insert);
    assert_eq!(operations[4].operation, InputFileBodyOperation::Read);
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
pub fn test_generate_large_afi_rw() {
    let file_path = "tests/data/256_write_32_1024.afi";
    let mut instructions = AfsInputInstructions::new(file_path, "0x0a", 32, 1024);
    for i in 0..256 {
        instructions.add_operations(vec![AfsOperation {
            operation: InputFileBodyOperation::Insert,
            args: vec![format!("0x{:08x}", i), format!("0x{:08x}", i * 2)],
        }]);
    }
    instructions.save_to_file().unwrap();

    let file_path = "tests/data/256_read_32_1024.afi";
    let mut instructions = AfsInputInstructions::new(file_path, "0x0a", 32, 1024);
    for i in 0..256 {
        instructions.add_operations(vec![AfsOperation {
            operation: InputFileBodyOperation::Read,
            args: vec![format!("0x{:08x}", i)],
        }]);
    }
    instructions.save_to_file().unwrap();
}
