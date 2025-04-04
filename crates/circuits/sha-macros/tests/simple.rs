use openvm_sha2_air::{Sha256Config, Sha2Config};
use openvm_sha_macros::ColsRef;

#[allow(dead_code)]
#[derive(ColsRef)]
#[config(Sha2Config)]
struct Test<T, const WORD_BITS: usize, const ROUNDS_PER_ROW: usize, const WORD_U16S: usize> {
    a: T,
    b: [T; WORD_BITS],
    c: [[T; WORD_BITS]; ROUNDS_PER_ROW],
}

#[test]
fn simple() {
    let input = [0; 1 + 32 + 32 * 4];
    let test: TestRef<u32> = TestRef::from::<Sha256Config>(&input);
    println!("{}, {}, {}", test.a, test.b[0], test.b[1]);
    println!("{}", test.c);
}
