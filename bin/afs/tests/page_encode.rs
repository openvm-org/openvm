use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{Read, Write};

#[test]
pub fn test_encode() {
    let seed = [42; 32];
    let mut rng = StdRng::from_seed(seed);

    let log_page_height = 3;
    let page_width = 4;
    let page_height = 1 << log_page_height;

    let pages = (0..page_height)
        .map(|_| {
            (0..page_width)
                .map(|_| rng.gen::<u32>())
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<Vec<u32>>>();

    let serialized = bincode::serialize(&pages).unwrap();
    let mut file = File::create("output.afp").unwrap();
    file.write_all(&serialized).unwrap();
}

#[test]
pub fn test_decode() {
    let file = File::open("output.afp").unwrap();
    let mut reader = std::io::BufReader::new(file);
    let mut serialized = Vec::new();
    reader.read_to_end(&mut serialized).unwrap();
    let deserialized: Vec<Vec<u32>> = bincode::deserialize(&serialized).unwrap();
    println!("{:?}", deserialized);
}
