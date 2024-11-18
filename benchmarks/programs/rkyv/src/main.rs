#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

mod generate;
mod types;

use rkyv::{access, rancor::Panic, Archive};
use types::Players;

axvm::entry!(main);

fn main() {
    // nothing up our sleeves, state and stream are first 20 digits of pi
    // const STATE: u64 = 3141592653;
    // const STREAM: u64 = 5897932384;

    // let mut rng = Lcg64Xsh32::new(STATE, STREAM);

    // const PLAYERS: usize = 500;
    // let data = Players {
    //     players: generate_vec::<_, Player>(&mut rng, PLAYERS..PLAYERS + 1),
    // };

    // let ser: Result<AlignedVec, Panic> = to_bytes(&data);
    // let ser = ser.unwrap();

    // let mut file = File::create("minecraft_savedata.bin").expect("Failed to create file");
    // file.write_all(&ser).expect("Failed to write to file");

    let data = axvm::io::read_vec();

    access::<<Players as Archive>::Archived, Panic>(&data).expect("Failed to deserialize");
}
