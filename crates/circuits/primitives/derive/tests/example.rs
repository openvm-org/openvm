use openvm_circuit_primitives_derive::ColsRef;

pub trait ExampleConfig {
    const N: usize;
}
pub struct ExampleConfigImplA;
impl ExampleConfig for ExampleConfigImplA {
    const N: usize = 5;
}
pub struct ExampleConfigImplB;
impl ExampleConfig for ExampleConfigImplB {
    const N: usize = 10;
}

#[allow(dead_code)]
#[derive(ColsRef)]
#[config(ExampleConfig)]
struct ExampleCols<T, const N: usize> {
    arr: [T; N],
    sum: T,
}

#[test]
fn example() {
    let input = [1, 2, 3, 4, 5, 15];
    let test: ExampleColsRef<u32> = ExampleColsRef::from::<ExampleConfigImplA>(&input);
    println!("{}, {}", test.arr, test.sum);
}

/*
 * For reference, this is what the ColsRef macro expands to.
 * The `cargo expand` tool is helpful for understanding how the ColsRef macro works.
 * See https://github.com/dtolnay/cargo-expand

#[derive(Debug, Clone)]
struct ExampleColsRef<'a, T> {
    pub arr: ndarray::ArrayView1<'a, T>,
    pub sum: &'a T,
}

impl<'a, T> ExampleColsRef<'a, T> {
    pub fn from<C: ExampleConfig>(slice: &'a [T]) -> Self {
        let (arr_slice, slice) = slice.split_at(1 * C::N);
        let arr_slice = ndarray::ArrayView1::from_shape((C::N), arr_slice).unwrap();
        let sum_length = 1;
        let (sum_slice, slice) = slice.split_at(sum_length);
        Self {
            arr: arr_slice,
            sum: &sum_slice[0],
        }
    }
    pub const fn width<C: ExampleConfig>() -> usize {
        0 + 1 * C::N + 1
    }
}

impl<'b, T> ExampleColsRef<'b, T> {
    pub fn from_mut<'a, C: ExampleConfig>(other: &'b ExampleColsRefMut<'a, T>) -> Self {
        Self {
            arr: other.arr.view(),
            sum: &other.sum,
        }
    }
}

#[derive(Debug)]
struct ExampleColsRefMut<'a, T> {
    pub arr: ndarray::ArrayViewMut1<'a, T>,
    pub sum: &'a mut T,
}

impl<'a, T> ExampleColsRefMut<'a, T> {
    pub fn from<C: ExampleConfig>(slice: &'a mut [T]) -> Self {
        let (arr_slice, slice) = slice.split_at_mut(1 * C::N);
        let arr_slice = ndarray::ArrayViewMut1::from_shape((C::N), arr_slice).unwrap();
        let sum_length = 1;
        let (sum_slice, slice) = slice.split_at_mut(sum_length);
        Self {
            arr: arr_slice,
            sum: &mut sum_slice[0],
        }
    }
    pub const fn width<C: ExampleConfig>() -> usize {
        0 + 1 * C::N + 1
    }
}
*/
