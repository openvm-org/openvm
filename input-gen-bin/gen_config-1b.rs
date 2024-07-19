pub const MAX: usize = 1000000;
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
};

// [page]
// index_bytes = 16
// data_bytes = 32
// bits_per_fe = 16
// leaf_height = 1048576
// internal_height = 1024
// mode = "ReadWrite" # options: "ReadOnly", "ReadWrite"
// max_rw_ops = 65536

// [tree]
// init_leaf_cap = 1024
// init_internal_cap = 1
// final_leaf_cap = 1024
// final_internal_cap = 1

// [schema]
// key_length = 2
// limb_size = 4

pub fn main() {
    let args: Vec<String> = env::args().collect();
    let leaf_height: String = args[1].clone();
    let internal_height: String = args[2].clone();
    let leaf_cap: String = args[3].clone();
    let internal_cap: String = args[4].clone();
    let num_ops: String = args[5].clone();

    let file = File::create("config-1b.toml").unwrap();
    let mut writer = BufWriter::new(file);
    writer
        .write_all(b"[page]\nindex_bytes = 16\ndata_bytes = 32\nbits_per_fe = 16\n")
        .unwrap();
    writer.write_all(b"leaf_height = ").unwrap();
    writer.write_all(leaf_height.as_bytes()).unwrap();
    writer.write_all(b"\ninternal_height = ").unwrap();
    writer.write_all(internal_height.as_bytes()).unwrap();
    writer
        .write_all(b"\nmode = \"ReadWrite\" # options: \"ReadOnly\", \"ReadWrite\"\nmax_rw_ops = ")
        .unwrap();
    writer.write_all(num_ops.as_bytes()).unwrap();
    writer.write_all(b"\n\n[tree]\n").unwrap();
    writer.write_all(b"init_leaf_cap = ").unwrap();
    writer.write_all(leaf_cap.as_bytes()).unwrap();
    writer.write_all(b"\ninit_internal_cap = ").unwrap();
    writer.write_all(internal_cap.as_bytes()).unwrap();
    writer.write_all(b"\nfinal_leaf_cap = ").unwrap();
    writer.write_all(leaf_cap.as_bytes()).unwrap();
    writer.write_all(b"\nfinal_internal_cap = ").unwrap();
    writer.write_all(internal_cap.as_bytes()).unwrap();
    writer
        .write_all(b"\n\n[schema]\nkey_length = 2\nlimb_size=4\n")
        .unwrap();
}
