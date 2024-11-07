use num_bigint_dig::BigUint;

mod bls12381;
mod bn254;
mod utils;
pub use bls12381::*;
pub use bn254::*;
pub use utils::*;

#[allow(non_snake_case)]
pub struct CurveConst {
    pub MODULUS: BigUint,
    pub ORDER: BigUint,
    pub XI: [isize; 2],
}
