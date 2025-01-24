#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), no_main)]

mod generate;
mod types;

use bincode::{config::standard, decode_from_slice};
use types::Players;

openvm::entry!(main);

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

    let mut running_product: usize = 1;
    for _ in 0..40000 {
        let mut running_sum: usize = 0;
        for x in openvm::io::read_vec() {
            running_sum += x as usize;
        }
        running_product *= running_sum;
        running_product %= 25565;
    }

    //let data: (Players, usize) = openvm::io::read();

    //let data = openvm::io::read_vec();

    //let _deser: (Players, usize) = decode_from_slice(&data, config).expect("Failed to deserialize");*/
}
