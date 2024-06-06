use super::AfsInputFile;

#[test]
pub fn test_read_file() {
    let file_path = "tests/test_input_file.afi";
    let afs_input_file = AfsInputFile::new(file_path.to_string());
    println!("{:?}", afs_input_file);
}
