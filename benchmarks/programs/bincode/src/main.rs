mod generate;
mod types;

use bincode::{config::standard, decode_from_slice};
use types::Players;

fn main() {
    // nothing up our sleeves, state and stream are first 20 digits of pi
    // const STATE: u64 = 3141592653;
    // const STREAM: u64 = 5897932384;

    // let mut rng = Lcg64Xsh32::new(STATE, STREAM);

    // const PLAYERS: usize = 500;
    // let data = Players {
    //     players: generate_vec::<_, Player>(&mut rng, PLAYERS..PLAYERS + 1),
    // };

    // let ser = encode_to_vec(&data, config);
    // let ser = ser.unwrap();

    // let mut file = File::create("minecraft_savedata.bin").expect("Failed to create file");
    // file.write_all(&ser).expect("Failed to write to file");

    let config = standard();

    let data = openvm::io::read_vec();

    let _deser: (Players, usize) = decode_from_slice(&data, config).expect("Failed to deserialize");
}
