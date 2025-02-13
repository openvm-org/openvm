use openvm_sha_air::{Sha256Config, ShaConfig};
use openvm_sha_macros::ColsRef;

#[derive(ColsRef)]
#[config(ShaConfig)]
struct ArrayTest<T> {
    a: T,
    b: [T; 4],
    c: [[T; 4]; 4],
}

#[test]
fn arrays() {
    let input = [1; 1 + 4 + 4 * 4];
    let test: ArrayTestRef<u32> = ArrayTestRef::from::<Sha256Config>(&input);
    println!("{}", test.a);
    println!("{}", test.b);
    println!("{}", test.c);
}
