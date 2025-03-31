mod generate;
mod types;

use core::hint::black_box;

use rkyv::{access_unchecked, Archive};
use types::Players;

fn main() {
    // Uncomment to generate a minecraft save file:
    // // nothing up our sleeves, state and stream are first 20 digits of pi
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

    let data = openvm::io::read_vec();

    // The zkVM does not need alignment guarantees, and in fact because of how read_vec works,
    // `data` will only be 4-aligned since it first reads a u32 for the vec length.
    let _archived = black_box(unsafe { access_unchecked::<<Players as Archive>::Archived>(&data) });
}
