use openvm_sha_macros::ColsRef;

const ONE: usize = 1;
const TWO: usize = 2;
const THREE: usize = 3;

mod test_config;
use test_config::{TestConfig, TestConfigImpl};

#[allow(dead_code)]
#[derive(ColsRef)]
#[config(TestConfig)]
struct ConstLenArrayTest<T, const N: usize> {
    a: T,
    b: [T; N],
    c: [[T; ONE]; TWO],
    d: [[[T; ONE]; TWO]; THREE],
}

#[test]
fn const_len_arrays() {
    let input = [1; 1 + TestConfigImpl::N * 2 + 2 * 3];
    let test: ConstLenArrayTestRef<u32> = ConstLenArrayTestRef::from::<TestConfigImpl>(&input);
    println!("{}", test.a);
    println!("{}", test.b);
    println!("{}", test.c);
    println!("{}", test.d);
}
