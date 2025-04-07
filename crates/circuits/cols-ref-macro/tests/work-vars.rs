use openvm_sha2_air::{Sha256Config, Sha2Config};
use openvm_sha_macros::ColsRef;

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2Config)]
pub struct ShaWorkVarsCols<
    T,
    const WORD_BITS: usize,
    const ROUNDS_PER_ROW: usize,
    const WORD_U16S: usize,
> {
    /// `a` and `e` after each iteration as 32-bits
    pub a: [[T; WORD_BITS]; ROUNDS_PER_ROW],
    pub e: [[T; WORD_BITS]; ROUNDS_PER_ROW],
    /// The carry's used for addition during each iteration when computing `a` and `e`
    pub carry_a: [[T; WORD_U16S]; ROUNDS_PER_ROW],
    pub carry_e: [[T; WORD_U16S]; ROUNDS_PER_ROW],
}

#[test]
fn work_vars() {
    let input = [2; 32 * 4 + 32 * 4 + 2 * 4 + 2 * 4];
    let test: ShaWorkVarsColsRef<u32> = ShaWorkVarsColsRef::from::<Sha256Config>(&input);
    println!(
        "{:?}\n{:?}\n{:?}\n{:?}",
        test.a, test.e, test.carry_a, test.carry_e
    );
}
