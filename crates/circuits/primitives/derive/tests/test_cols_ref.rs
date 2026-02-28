use openvm_circuit_primitives_derive::{AlignedBorrow, ColsRef};

pub trait TestConfig {
    const N: usize;
    const M: usize;
}
pub struct TestConfigImpl;
impl TestConfig for TestConfigImpl {
    const N: usize = 5;
    const M: usize = 2;
}

#[allow(dead_code)] // TestCols isn't actually used in the code. silence clippy warning
#[derive(ColsRef)]
#[config(TestConfig)]
struct TestCols<T, const N: usize, const M: usize> {
    single_field_element: T,
    array_of_t: [T; N],
    nested_array_of_t: [[T; N]; N],
    cols_struct: TestSubCols<T, M>,
    #[aligned_borrow]
    array_of_aligned_borrow: [TestAlignedBorrow<T>; N],
    #[aligned_borrow]
    nested_array_of_aligned_borrow: [[TestAlignedBorrow<T>; N]; N],
}

#[allow(dead_code)] // TestSubCols isn't actually used in the code. silence clippy warning
#[derive(ColsRef, Debug)]
#[config(TestConfig)]
struct TestSubCols<T, const M: usize> {
    // TestSubCols can have fields of any type that TestCols can have
    a: T,
    b: [T; M],
    #[aligned_borrow]
    c: TestAlignedBorrow<T>,
}

#[derive(AlignedBorrow, Debug)]
struct TestAlignedBorrow<T> {
    a: T,
    b: [T; 5],
}

#[test]
fn test_cols_ref() {
    assert_eq!(
        TestColsRef::<i32>::width::<TestConfigImpl>(),
        TestColsRefMut::<i32>::width::<TestConfigImpl>()
    );
    const WIDTH: usize = TestColsRef::<i32>::width::<TestConfigImpl>();
    let mut input = vec![0; WIDTH];
    let mut cols: TestColsRefMut<i32> = TestColsRefMut::from::<TestConfigImpl>(&mut input);

    *cols.single_field_element = 1;
    cols.array_of_t[0] = 2;
    cols.nested_array_of_t[[0, 0]] = 3;
    *cols.cols_struct.a = 4;
    cols.cols_struct.b[0] = 5;
    cols.cols_struct.c.a = 6;
    cols.cols_struct.c.b[0] = 7;
    cols.array_of_aligned_borrow[0].a = 8;
    cols.array_of_aligned_borrow[0].b[0] = 9;
    cols.nested_array_of_aligned_borrow[[0, 0]].a = 10;
    cols.nested_array_of_aligned_borrow[[0, 0]].b[0] = 11;

    let cols: TestColsRef<i32> = TestColsRef::from::<TestConfigImpl>(&input);
    println!("{cols:?}");
    assert_eq!(*cols.single_field_element, 1);
    assert_eq!(cols.array_of_t[0], 2);
    assert_eq!(cols.nested_array_of_t[[0, 0]], 3);
    assert_eq!(*cols.cols_struct.a, 4);
    assert_eq!(cols.cols_struct.b[0], 5);
    assert_eq!(cols.cols_struct.c.a, 6);
    assert_eq!(cols.cols_struct.c.b[0], 7);
    assert_eq!(cols.array_of_aligned_borrow[0].a, 8);
    assert_eq!(cols.array_of_aligned_borrow[0].b[0], 9);
    assert_eq!(cols.nested_array_of_aligned_borrow[[0, 0]].a, 10);
    assert_eq!(cols.nested_array_of_aligned_borrow[[0, 0]].b[0], 11);
}

/*
 * For reference, this is what the ColsRef macro expands to.
 * The `cargo expand` tool is helpful for understanding how the ColsRef macro works.
 * See https://github.com/dtolnay/cargo-expand

#[derive(Debug, Clone)]
struct TestColsRef<'a, T> {
    pub single_field_element: &'a T,
    pub array_of_t: ndarray::ArrayView1<'a, T>,
    pub nested_array_of_t: ndarray::ArrayView2<'a, T>,
    pub cols_struct: TestSubColsRef<'a, T>,
    pub array_of_aligned_borrow: ndarray::ArrayView1<'a, TestAlignedBorrow<T>>,
    pub nested_array_of_aligned_borrow: ndarray::ArrayView2<'a, TestAlignedBorrow<T>>,
}

impl<'a, T> TestColsRef<'a, T> {
    pub fn from<C: TestConfig>(slice: &'a [T]) -> Self {
        let single_field_element_length = 1;
        let (single_field_element_slice, slice) = slice
            .split_at(single_field_element_length);
        let (array_of_t_slice, slice) = slice.split_at(1 * C::N);
        let array_of_t_slice = ndarray::ArrayView1::from_shape((C::N), array_of_t_slice)
            .unwrap();
        let (nested_array_of_t_slice, slice) = slice.split_at(1 * C::N * C::N);
        let nested_array_of_t_slice = ndarray::ArrayView2::from_shape(
                (C::N, C::N),
                nested_array_of_t_slice,
            )
            .unwrap();
        let cols_struct_length = <TestSubColsRef<'a, T>>::width::<C>();
        let (cols_struct_slice, slice) = slice.split_at(cols_struct_length);
        let cols_struct_slice = <TestSubColsRef<'a, T>>::from::<C>(cols_struct_slice);
        let (array_of_aligned_borrow_slice, slice) = slice
            .split_at(<TestAlignedBorrow<T>>::width() * C::N);
        let array_of_aligned_borrow_slice: &[TestAlignedBorrow<T>] = unsafe {
            &*(array_of_aligned_borrow_slice as *const [T]
                as *const [TestAlignedBorrow<T>])
        };
        let array_of_aligned_borrow_slice = ndarray::ArrayView1::from_shape(
                (C::N),
                array_of_aligned_borrow_slice,
            )
            .unwrap();
        let (nested_array_of_aligned_borrow_slice, slice) = slice
            .split_at(<TestAlignedBorrow<T>>::width() * C::N * C::N);
        let nested_array_of_aligned_borrow_slice: &[TestAlignedBorrow<T>] = unsafe {
            &*(nested_array_of_aligned_borrow_slice as *const [T]
                as *const [TestAlignedBorrow<T>])
        };
        let nested_array_of_aligned_borrow_slice = ndarray::ArrayView2::from_shape(
                (C::N, C::N),
                nested_array_of_aligned_borrow_slice,
            )
            .unwrap();
        Self {
            single_field_element: &single_field_element_slice[0],
            array_of_t: array_of_t_slice,
            nested_array_of_t: nested_array_of_t_slice,
            cols_struct: cols_struct_slice,
            array_of_aligned_borrow: array_of_aligned_borrow_slice,
            nested_array_of_aligned_borrow: nested_array_of_aligned_borrow_slice,
        }
    }
    pub const fn width<C: TestConfig>() -> usize {
        0 + 1 + 1 * C::N + 1 * C::N * C::N + <TestSubColsRef<'a, T>>::width::<C>()
            + <TestAlignedBorrow<T>>::width() * C::N
            + <TestAlignedBorrow<T>>::width() * C::N * C::N
    }
}

impl<'b, T> TestColsRef<'b, T> {
    pub fn from_mut<'a, C: TestConfig>(other: &'b TestColsRefMut<'a, T>) -> Self {
        Self {
            single_field_element: &other.single_field_element,
            array_of_t: other.array_of_t.view(),
            nested_array_of_t: other.nested_array_of_t.view(),
            cols_struct: <TestSubColsRef<'b, T>>::from_mut::<C>(&other.cols_struct),
            array_of_aligned_borrow: other.array_of_aligned_borrow.view(),
            nested_array_of_aligned_borrow: other.nested_array_of_aligned_borrow.view(),
        }
    }
}

#[derive(Debug)]
struct TestColsRefMut<'a, T> {
    pub single_field_element: &'a mut T,
    pub array_of_t: ndarray::ArrayViewMut1<'a, T>,
    pub nested_array_of_t: ndarray::ArrayViewMut2<'a, T>,
    pub cols_struct: TestSubColsRefMut<'a, T>,
    pub array_of_aligned_borrow: ndarray::ArrayViewMut1<'a, TestAlignedBorrow<T>>,
    pub nested_array_of_aligned_borrow: ndarray::ArrayViewMut2<'a, TestAlignedBorrow<T>>,
}

impl<'a, T> TestColsRefMut<'a, T> {
    pub fn from<C: TestConfig>(slice: &'a mut [T]) -> Self {
        let single_field_element_length = 1;
        let (single_field_element_slice, slice) = slice
            .split_at_mut(single_field_element_length);
        let (array_of_t_slice, slice) = slice.split_at_mut(1 * C::N);
        let array_of_t_slice = ndarray::ArrayViewMut1::from_shape(
                (C::N),
                array_of_t_slice,
            )
            .unwrap();
        let (nested_array_of_t_slice, slice) = slice.split_at_mut(1 * C::N * C::N);
        let nested_array_of_t_slice = ndarray::ArrayViewMut2::from_shape(
                (C::N, C::N),
                nested_array_of_t_slice,
            )
            .unwrap();
        let cols_struct_length = <TestSubColsRefMut<'a, T>>::width::<C>();
        let (cols_struct_slice, slice) = slice.split_at_mut(cols_struct_length);
        let cols_struct_slice = <TestSubColsRefMut<'a, T>>::from::<C>(cols_struct_slice);
        let (array_of_aligned_borrow_slice, slice) = slice
            .split_at_mut(<TestAlignedBorrow<T>>::width() * C::N);
        let array_of_aligned_borrow_slice: &mut [TestAlignedBorrow<T>] = unsafe {
            &mut *(array_of_aligned_borrow_slice as *mut [T]
                as *mut [TestAlignedBorrow<T>])
        };
        let array_of_aligned_borrow_slice = ndarray::ArrayViewMut1::from_shape(
                (C::N),
                array_of_aligned_borrow_slice,
            )
            .unwrap();
        let (nested_array_of_aligned_borrow_slice, slice) = slice
            .split_at_mut(<TestAlignedBorrow<T>>::width() * C::N * C::N);
        let nested_array_of_aligned_borrow_slice: &mut [TestAlignedBorrow<T>] = unsafe {
            &mut *(nested_array_of_aligned_borrow_slice as *mut [T]
                as *mut [TestAlignedBorrow<T>])
        };
        let nested_array_of_aligned_borrow_slice = ndarray::ArrayViewMut2::from_shape(
                (C::N, C::N),
                nested_array_of_aligned_borrow_slice,
            )
            .unwrap();
        Self {
            single_field_element: &mut single_field_element_slice[0],
            array_of_t: array_of_t_slice,
            nested_array_of_t: nested_array_of_t_slice,
            cols_struct: cols_struct_slice,
            array_of_aligned_borrow: array_of_aligned_borrow_slice,
            nested_array_of_aligned_borrow: nested_array_of_aligned_borrow_slice,
        }
    }
    pub const fn width<C: TestConfig>() -> usize {
        0 + 1 + 1 * C::N + 1 * C::N * C::N + <TestSubColsRefMut<'a, T>>::width::<C>()
            + <TestAlignedBorrow<T>>::width() * C::N
            + <TestAlignedBorrow<T>>::width() * C::N * C::N
    }
}

#[derive(Debug, Clone)]
struct TestSubColsRef<'a, T> {
    pub a: &'a T,
    pub b: ndarray::ArrayView1<'a, T>,
    pub c: &'a TestAlignedBorrow<T>,
}

impl<'a, T> TestSubColsRef<'a, T> {
    pub fn from<C: TestConfig>(slice: &'a [T]) -> Self {
        let a_length = 1;
        let (a_slice, slice) = slice.split_at(a_length);
        let (b_slice, slice) = slice.split_at(1 * C::M);
        let b_slice = ndarray::ArrayView1::from_shape((C::M), b_slice).unwrap();
        let c_length = <TestAlignedBorrow<T>>::width();
        let (c_slice, slice) = slice.split_at(c_length);
        Self {
            a: &a_slice[0],
            b: b_slice,
            c: {
                use core::borrow::Borrow;
                c_slice.borrow()
            },
        }
    }
    pub const fn width<C: TestConfig>() -> usize {
        0 + 1 + 1 * C::M + <TestAlignedBorrow<T>>::width()
    }
}

impl<'b, T> TestSubColsRef<'b, T> {
    pub fn from_mut<'a, C: TestConfig>(other: &'b TestSubColsRefMut<'a, T>) -> Self {
        Self {
            a: &other.a,
            b: other.b.view(),
            c: other.c,
        }
    }
}

#[derive(Debug)]
struct TestSubColsRefMut<'a, T> {
    pub a: &'a mut T,
    pub b: ndarray::ArrayViewMut1<'a, T>,
    pub c: &'a mut TestAlignedBorrow<T>,
}

impl<'a, T> TestSubColsRefMut<'a, T> {
    pub fn from<C: TestConfig>(slice: &'a mut [T]) -> Self {
        let a_length = 1;
        let (a_slice, slice) = slice.split_at_mut(a_length);
        let (b_slice, slice) = slice.split_at_mut(1 * C::M);
        let b_slice = ndarray::ArrayViewMut1::from_shape((C::M), b_slice).unwrap();
        let c_length = <TestAlignedBorrow<T>>::width();
        let (c_slice, slice) = slice.split_at_mut(c_length);
        Self {
            a: &mut a_slice[0],
            b: b_slice,
            c: {
                use core::borrow::BorrowMut;
                c_slice.borrow_mut()
            },
        }
    }
    pub const fn width<C: TestConfig>() -> usize {
        0 + 1 + 1 * C::M + <TestAlignedBorrow<T>>::width()
    }
}
*/
