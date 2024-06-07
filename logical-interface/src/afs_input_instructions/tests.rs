use super::AfsInputInstructions;
use crate::mock_db::MockDb;

#[test]
pub fn test_read_file() {
    let file_path = "tests/test_input_file.afi";
    let afs_input_file = AfsInputInstructions::from_file(file_path.to_string());
    println!("{:?}", afs_input_file);
}

#[test]
pub fn test_write_mock_db_from_file() {
    let file_path = "tests/test_input_file.afi";
    let afi = AfsInputInstructions::from_file(file_path.to_string());
    let mock_db: MockDb<32, 1024> = MockDb::new_from_afi(&afi);
    println!("{:?}", mock_db.map);
}
