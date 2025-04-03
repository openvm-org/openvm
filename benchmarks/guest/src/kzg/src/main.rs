use core::hint::black_box;

use hex_literal::hex;
use revm_precompile::kzg_point_evaluation;
use revm_primitives::{Bytes, Env};

pub fn main() {
    let input = &hex!("01e798154708fe7789429634053cbf9f99b619f9f084048927333fce637f549b564c0a11a0f704f4fc3e8acfe0f8245f0ad1347b378fbf96e206da11a5d3630624d25032e67a7e6a4910df5834b8fe70e6bcfeeac0352434196bdf4b2485d5a18f59a8d2a1a625a17f3fea0fe5eb8c896db3764f3185481bc22f91b4aaffcca25f26936857bc3a7c2539ea8ec3a952b7873033e038326e87ed3e1276fd140253fa08e9fc25fb2d9a98527fc22a2c9612fbeafdad446cbc7bcdbdcd780af2c16a");

    let input_bytes = Bytes::from_static(input);
    let result = kzg_point_evaluation::run(&input_bytes, u64::MAX, &Env::default());

    black_box(result);
}
