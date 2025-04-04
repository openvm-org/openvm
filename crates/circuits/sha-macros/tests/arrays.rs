use openvm_sha_macros::ColsRef;

mod test_config;
use test_config::{TestConfig, TestConfigImpl};

#[derive(ColsRef)]
#[config(TestConfig)]
struct ArrayTest<T, const N: usize> {
    a: T,
    b: [T; N],
    c: [[T; 4]; N],
}

#[test]
fn arrays() {
    let input = [1; 1 + TestConfigImpl::N + 4 * TestConfigImpl::N];
    let test: ArrayTestRef<u32> = ArrayTestRef::from::<TestConfigImpl>(&input);
    println!("{}", test.a);
    println!("{:?}", test.b);
    println!("{:?}", test.c);
}

/*
#[derive(Clone)]
struct ArrayTestRef<'a, T> {
    pub a: &'a T,
    pub b: ndarray::ArrayView1<'a, T>,
    pub c: ndarray::ArrayView2<'a, T>,
}

impl<'a, T> ArrayTestRef<'a, T> {
    pub fn from<C: TestConfig>(slice: &'a [T]) -> Self {
        let a_length = 1;
        let (a_slice, slice) = slice.split_at(a_length);
        let (b_slice, slice) = slice.split_at(1 * C::N);
        let b_slice = ndarray::ArrayView1::from_shape((C::N), b_slice).unwrap();
        let (c_slice, slice) = slice.split_at(1 * C::N * 4);
        let c_slice = ndarray::ArrayView2::from_shape((C::N, 4), c_slice).unwrap();
        Self {
            a: &a_slice[0],
            b: b_slice,
            c: c_slice,
        }
    }
    pub const fn width<C: TestConfig>() -> usize {
        0 + 1 + 1 * C::N + 1 * C::N * 4
    }
}

impl<'b, T> ArrayTestRef<'b, T> {
    pub fn from_mut<'a, C: TestConfig>(other: &'b ArrayTestRefMut<'a, T>) -> Self {
        Self {
            a: &other.a,
            b: other.b.view(),
            c: other.c.view(),
        }
    }
}
struct ArrayTestRefMut<'a, T> {
    pub a: &'a mut T,
    pub b: ndarray::ArrayViewMut1<'a, T>,
    pub c: ndarray::ArrayViewMut2<'a, T>,
}

impl<'a, T> ArrayTestRefMut<'a, T> {
    pub fn from<C: TestConfig>(slice: &'a mut [T]) -> Self {
        let a_length = 1;
        let (mut a_slice, mut slice) = slice.split_at_mut(a_length);
        let (mut b_slice, mut slice) = slice.split_at_mut(1 * C::N);
        let b_slice = ndarray::ArrayViewMut1::from_shape((C::N), b_slice).unwrap();
        let (mut c_slice, mut slice) = slice.split_at_mut(1 * C::N * 4);
        let c_slice = ndarray::ArrayViewMut2::from_shape((C::N, 4), c_slice).unwrap();
        Self {
            a: &mut a_slice[0],
            b: b_slice,
            c: c_slice,
        }
    }
    pub const fn width<C: TestConfig>() -> usize {
        0 + 1 + 1 * C::N + 1 * C::N * 4
    }
}
*/
