use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_sha_macros::ColsRef;

mod test_config;
use test_config::{TestConfig, TestConfigImpl};

#[derive(ColsRef)]
#[config(TestConfig)]
struct TestCols<T, const N: usize> {
    a: [T; N],
    // Forces the field to be treated as a struct that derives AlignedBorrow.
    // In particular, ignores the fact that it ends with `Cols` and doesn't
    // expect a `PlainTestColsRef` type.
    #[plain]
    b: PlainCols<T>,
}

#[derive(Clone, Copy, Debug, AlignedBorrow)]
struct PlainCols<T> {
    a: T,
    b: [T; 4],
}

#[test]
fn plain() {
    let input = [1; TestConfigImpl::N + 1 + 4];
    let test: TestColsRef<u32> = TestColsRef::from::<TestConfigImpl>(&input);
    println!("{}", test.a);
    println!("{:?}", test.b);
}

#[test]
fn plain_mut() {
    let mut input = [1; TestConfigImpl::N + 1 + 4];
    let mut test: TestColsRefMut<u32> = TestColsRefMut::from::<TestConfigImpl>(&mut input);
    test.a[0] = 1;
    test.b.a = 1;
    test.b.b[0] = 1;
    println!("{}", test.a);
    println!("{:?}", test.b);
}

#[test]
fn plain_from_mut() {
    let mut input = [1; TestConfigImpl::N + 1 + 4];
    let mut test: TestColsRefMut<u32> = TestColsRefMut::from::<TestConfigImpl>(&mut input);
    test.a[0] = 1;
    test.b.a = 1;
    test.b.b[0] = 1;
    let test2: TestColsRef<u32> = TestColsRef::from_mut::<TestConfigImpl>(&mut test);
    println!("{}", test2.a);
    println!("{:?}", test2.b);
}
