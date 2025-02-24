use openvm_sha_air::{Sha256Config, ShaConfig};
use openvm_sha_macros::ColsRef;

#[derive(ColsRef)]
struct Test1Cols<T, const WORD_BITS: usize> {
    pub a: T,
    pub nested: Test2Cols<T, WORD_BITS>,
}

#[derive(ColsRef)]
struct Test2Cols<T, const WORD_BITS: usize> {
    pub b: T,
    pub c: [T; WORD_BITS],
}

#[test]
fn nested_const() {
    let input = [0; 1 + 1 + 32];
    let test: Test1ColsRef<u32> = Test1ColsRef::from::<Sha256Config>(&input);
    println!("{}, {}, {}", test.a, test.nested.b, test.nested.c);
}

#[test]
fn nested_mut() {
    let mut input = [0; 1 + 1 + 32];
    let mut test: Test1ColsRefMut<u32> = Test1ColsRefMut::from::<Sha256Config>(&mut input);
    *test.nested.b = 1u32;
    test.nested.c[0] = 1u32;
    println!("{}, {}, {}", test.a, test.nested.b, test.nested.c);
}

#[test]
fn nested_from_mut() {
    let mut mut_input = [0; 1 + 1 + 32];
    let mut mut_test: Test1ColsRefMut<u32> = Test1ColsRefMut::from::<Sha256Config>(&mut mut_input);
    let const_test: Test1ColsRef<u32> = Test1ColsRef::from_mut::<Sha256Config>(&mut mut_test);
    println!(
        "{}, {}, {}",
        const_test.a, const_test.nested.b, const_test.nested.c
    );
}
