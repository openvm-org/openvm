fn main() {
    vergen::EmitBuilder::builder().git_sha(true).emit().unwrap();
}
