pub const MAX: usize = 1000000;
pub const MAX_OPS: usize = 65536;
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
};

pub fn main() {
    let args: Vec<String> = env::args().collect();
    let i: usize = args[1].parse().unwrap();
    let is_proof_input: usize = args[2].parse().unwrap();
    if is_proof_input == 0 {
        let file = File::create("tmp.afi").unwrap();
        let mut writer = BufWriter::new(file);
        writer.write_all(b"TABLE_ID big_tree\n").unwrap();
        writer.write_all(b"INDEX_BYTES 16\n").unwrap();
        writer.write_all(b"DATA_BYTES 32\n").unwrap();
        for j in 0..MAX {
            let s = (2 * (MAX * i + j)).to_string();
            let s = "INSERT ".to_owned() + &s + " " + &s + "\n";
            writer.write_all(s.as_bytes()).unwrap();
        }
    } else {
        let file = File::create("proof_input.afi").unwrap();
        let mut writer = BufWriter::new(file);
        writer.write_all(b"TABLE_ID big_tree\n").unwrap();
        writer.write_all(b"INDEX_BYTES 16\n").unwrap();
        writer.write_all(b"DATA_BYTES 32\n").unwrap();
        for j in 0..MAX_OPS {
            let s = (MAX_OPS * i + j).to_string();
            let s = "INSERT ".to_owned() + &s + " " + &s + "\n";
            writer.write_all(s.as_bytes()).unwrap();
        }
    }
}
