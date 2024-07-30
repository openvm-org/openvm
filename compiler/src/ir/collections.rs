use alloc::rc::Rc;
use itertools::Itertools;
use p3_field::AbstractField;
use std::cell::RefCell;

use super::{Builder, Config, FromConstant, MemIndex, MemVariable, Ptr, Usize, Var, Variable};

/// A logical array.
#[derive(Debug, Clone)]
pub enum Array<C: Config, T> {
    /// Array of some local variables or constants, which can only be manipulated statically. It
    /// only exists in the DSL syntax and isn't backed by memory.
    Fixed(Rc<RefCell<Vec<Option<T>>>>),
    /// Array on heap. Index access can use variables. Length could be determined on runtime but
    /// cannot change after initialization.
    Dyn(Ptr<C::N>, Usize<C::N>),
}

impl<C: Config, V: MemVariable<C>> Array<C, V> {
    /// Gets the length of the array as a variable inside the DSL.
    pub fn len(&self) -> Usize<C::N> {
        match self {
            Self::Fixed(vec) => Usize::from(vec.borrow().len()),
            Self::Dyn(_, len) => *len,
        }
    }

    /// Asserts that an array has a certain length. Change its length to constant if it is a variable.
    pub fn assert_len(&self, builder: &mut Builder<C>, len: usize) {
        match self {
            Self::Fixed(vec) => {
                assert_eq!(vec.borrow().len(), len);
            }
            Self::Dyn(_, c_len) => match c_len {
                Usize::Const(c_len) => {
                    assert_eq!(*c_len, len);
                }
                Usize::Var(c_len) => {
                    builder.assert_usize_eq(*c_len, len);
                }
            },
        }
    }

    /// Shifts the array by `shift` elements.
    /// !Attention!: the behavior of `Fixed` and `Dyn` is different. For Dyn, the shift is a view
    /// and shares memory with the original. For `Fixed`, `set`/`set_value` on slices won't impact
    /// the original array.
    pub fn shift(&self, builder: &mut Builder<C>, shift: Usize<C::N>) -> Array<C, V> {
        match self {
            Self::Fixed(v) => {
                if let Usize::Const(shift) = shift {
                    Array::Fixed(Rc::new(RefCell::new(v.borrow()[shift..].to_vec())))
                } else {
                    panic!("Cannot shift a fixed array with a variable shift");
                }
            }
            Self::Dyn(ptr, len) => {
                assert!(V::size_of() == 1, "only support variables of size 1");
                let new_address = builder.eval(ptr.address + shift);
                let new_ptr = Ptr::<C::N> {
                    address: new_address,
                };
                let len_var = len.materialize(builder);
                let new_length = builder.eval(len_var - shift);
                Array::Dyn(new_ptr, Usize::Var(new_length))
            }
        }
    }

    /// Truncates the array to `len` elements.
    pub fn truncate(&self, builder: &mut Builder<C>, len: Usize<C::N>) {
        match self {
            Self::Fixed(_) => {
                todo!()
            }
            Self::Dyn(_, old_len) => {
                builder.assign(*old_len, len);
            }
        };
    }

    /// Slices the array from `start` to `end`.
    /// !Attention!: the behavior of `Fixed` and `Dyn` is different. For Dyn, the shift is a view
    /// and shares memory with the original. For `Fixed`, `set`/`set_value` on slices won't impact
    /// the original array.
    pub fn slice(
        &self,
        builder: &mut Builder<C>,
        start: Usize<C::N>,
        end: Usize<C::N>,
    ) -> Array<C, V> {
        match self {
            Self::Fixed(v) => {
                if let (Usize::Const(start), Usize::Const(end)) = (start, end) {
                    Array::Fixed(Rc::new(RefCell::new(v.borrow()[start..end].to_vec())))
                } else {
                    panic!("Cannot slice a fixed array with a variable start or end");
                }
            }
            Self::Dyn(_, len) => {
                if builder.flags.debug {
                    let start_v = start.materialize(builder);
                    let end_v = end.materialize(builder);
                    let valid = builder.lt(start_v, end_v);
                    builder.assert_var_eq(valid, C::N::one());

                    let len_v = len.materialize(builder);
                    let len_plus_1_v = builder.eval(len_v + C::N::one());
                    let valid = builder.lt(end_v, len_plus_1_v);
                    builder.assert_var_eq(valid, C::N::one());
                }

                let slice_len: Usize<_> = builder.eval(end - start);
                let mut slice = builder.dyn_array(slice_len);
                builder.range(0, slice_len).for_each(|i, builder| {
                    let idx: Usize<_> = builder.eval(start + i);
                    let value = builder.get(self, idx);
                    builder.set(&mut slice, i, value);
                });

                slice
            }
        }
    }
}

impl<C: Config> Builder<C> {
    /// Initialize an array of fixed length `len`. The entries will be uninitialized.
    pub fn array<V: MemVariable<C>>(&mut self, len: impl Into<Usize<C::N>>) -> Array<C, V> {
        self.dyn_array(len)
    }

    /// Creates an array from a vector.
    pub fn vec<V: MemVariable<C>>(&mut self, v: Vec<V>) -> Array<C, V> {
        Array::Fixed(Rc::new(RefCell::new(
            v.into_iter().map(|x| Some(x)).collect(),
        )))
    }

    /// Create an uninitialized Array::Fixed.
    pub fn uninit_fixed_array<V: Variable<C>>(&mut self, len: usize) -> Array<C, V> {
        Array::Fixed(Rc::new(RefCell::new(vec![None::<V>; len])))
    }

    /// Creates a dynamic array for a length.
    pub fn dyn_array<V: MemVariable<C>>(&mut self, len: impl Into<Usize<C::N>>) -> Array<C, V> {
        let len = match len.into() {
            Usize::Const(len) => self.eval(C::N::from_canonical_usize(len)),
            Usize::Var(len) => len,
        };
        let len = Usize::Var(len);
        let ptr = self.alloc(len, V::size_of());
        Array::Dyn(ptr, len)
    }

    pub fn get<V: MemVariable<C>, I: Into<Usize<C::N>>>(
        &mut self,
        slice: &Array<C, V>,
        index: I,
    ) -> V {
        let index = index.into();

        match slice {
            Array::Fixed(slice) => {
                if let Usize::Const(idx) = index {
                    if let Some(ele) = &slice.borrow()[idx] {
                        ele.clone()
                    } else {
                        panic!("Cannot get an uninitialized element in a fixed slice");
                    }
                } else {
                    panic!("Cannot index into a fixed slice with a variable size")
                }
            }
            Array::Dyn(ptr, len) => {
                if self.flags.debug {
                    let index_v = index.materialize(self);
                    let len_v = len.materialize(self);
                    let valid = self.lt(index_v, len_v);
                    self.assert_var_eq(valid, C::N::one());
                }
                let index = MemIndex {
                    index,
                    offset: 0,
                    size: V::size_of(),
                };
                let var: V = self.uninit();
                self.load(var.clone(), *ptr, index);
                var
            }
        }
    }

    pub fn get_ptr<V: MemVariable<C>, I: Into<Usize<C::N>>>(
        &mut self,
        slice: &Array<C, V>,
        index: I,
    ) -> Ptr<C::N> {
        let index = index.into();

        match slice {
            Array::Fixed(_) => {
                todo!()
            }
            Array::Dyn(ptr, len) => {
                if self.flags.debug {
                    let index_v = index.materialize(self);
                    let len_v = len.materialize(self);
                    let valid = self.lt(index_v, len_v);
                    self.assert_var_eq(valid, C::N::one());
                }
                let index = MemIndex {
                    index,
                    offset: 0,
                    size: V::size_of(),
                };
                let var: Ptr<C::N> = self.uninit();
                self.load(var, *ptr, index);
                var
            }
        }
    }

    pub fn set<V: MemVariable<C>, I: Into<Usize<C::N>>, Expr: Into<V::Expression>>(
        &mut self,
        slice: &mut Array<C, V>,
        index: I,
        value: Expr,
    ) {
        let index = index.into();

        match slice {
            Array::Fixed(v) => {
                if let Usize::Const(idx) = index {
                    let value = self.eval(value);
                    v.borrow_mut()[idx] = Some(value);
                } else {
                    panic!("Cannot index into a fixed slice with a variable index")
                }
            }
            Array::Dyn(ptr, len) => {
                if self.flags.debug {
                    let index_v = index.materialize(self);
                    let len_v = len.materialize(self);
                    let valid = self.lt(index_v, len_v);
                    self.assert_var_eq(valid, C::N::one());
                }
                let index = MemIndex {
                    index,
                    offset: 0,
                    size: V::size_of(),
                };
                let value: V = self.eval(value);
                self.store(*ptr, index, value);
            }
        }
    }

    pub fn set_value<V: MemVariable<C>, I: Into<Usize<C::N>>>(
        &mut self,
        slice: &mut Array<C, V>,
        index: I,
        value: V,
    ) {
        let index = index.into();

        match slice {
            Array::Fixed(v) => {
                if let Usize::Const(idx) = index {
                    v.borrow_mut()[idx] = Some(value);
                } else {
                    panic!("Cannot index into a fixed slice with a variable size")
                }
            }
            Array::Dyn(ptr, _) => {
                let index = MemIndex {
                    index,
                    offset: 0,
                    size: V::size_of(),
                };
                self.store(*ptr, index, value);
            }
        }
    }
}

impl<C: Config, T: MemVariable<C>> Variable<C> for Array<C, T> {
    type Expression = Self;

    fn uninit(builder: &mut Builder<C>) -> Self {
        Array::Dyn(builder.uninit(), builder.uninit())
    }

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
        match (self, src.clone()) {
            (Array::Dyn(lhs_ptr, lhs_len), Array::Dyn(rhs_ptr, rhs_len)) => {
                builder.assign(*lhs_ptr, rhs_ptr);
                builder.assign(*lhs_len, rhs_len);
            }
            _ => unreachable!(),
        }
    }

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs.clone(), rhs.clone()) {
            (Array::Fixed(lhs), Array::Fixed(rhs)) => {
                // No need to compare if they are the same reference. The same reference will
                // also cause borrow errors in the following loop.
                if Rc::ptr_eq(&lhs, &rhs) {
                    return;
                }
                for (l, r) in lhs.borrow().iter().zip_eq(rhs.borrow().iter()) {
                    assert!(l.is_some(), "lhs array is not fully initialized");
                    assert!(r.is_some(), "rhs array is not fully initialized");
                    T::assert_eq(
                        T::Expression::from(l.as_ref().unwrap().clone()),
                        T::Expression::from(r.as_ref().unwrap().clone()),
                        builder,
                    );
                }
            }
            (Array::Dyn(_, lhs_len), Array::Dyn(_, rhs_len)) => {
                let lhs_len_var = builder.materialize(lhs_len);
                let rhs_len_var = builder.materialize(rhs_len);
                builder.assert_eq::<Var<_>>(lhs_len_var, rhs_len_var);

                let start = Usize::Const(0);
                let end = lhs_len;
                builder.range(start, end).for_each(|i, builder| {
                    let a = builder.get(&lhs, i);
                    let b = builder.get(&rhs, i);
                    builder.assert_eq::<T>(a, b);
                });
            }
            _ => panic!("cannot compare arrays of different types"),
        }
    }

    fn assert_ne(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs.clone(), rhs.clone()) {
            (Array::Fixed(lhs), Array::Fixed(rhs)) => {
                // No need to compare if they are the same reference. The same reference will
                // also cause borrow errors.
                if Rc::ptr_eq(&lhs, &rhs) {
                    panic!("assert not equal on the same array");
                }
                for (l, r) in lhs.borrow().iter().zip_eq(rhs.borrow().iter()) {
                    assert!(l.is_some(), "lhs array is not fully initialized");
                    assert!(r.is_some(), "rhs array is not fully initialized");
                    T::assert_ne(
                        T::Expression::from(l.as_ref().unwrap().clone()),
                        T::Expression::from(r.as_ref().unwrap().clone()),
                        builder,
                    );
                }
            }
            (Array::Dyn(_, lhs_len), Array::Dyn(_, rhs_len)) => {
                builder.assert_usize_eq(lhs_len, rhs_len);

                let end = lhs_len;
                builder.range(0, end).for_each(|i, builder| {
                    let a = builder.get(&lhs, i);
                    let b = builder.get(&rhs, i);
                    builder.assert_ne::<T>(a, b);
                });
            }
            _ => panic!("cannot compare arrays of different types"),
        }
    }

    // The default version calls `uninit`. If `expr` is `Fixed`, it will be converted into `Dyn`.
    fn eval(_builder: &mut Builder<C>, expr: impl Into<Self::Expression>) -> Self {
        expr.into()
    }
}

impl<C: Config, T: MemVariable<C>> MemVariable<C> for Array<C, T> {
    fn size_of() -> usize {
        2
    }

    fn load(&self, src: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        match self {
            Array::Dyn(dst, Usize::Var(len)) => {
                let mut index = index;
                dst.load(src, index, builder);
                index.offset += <Ptr<C::N> as MemVariable<C>>::size_of();
                len.load(src, index, builder);
            }
            _ => unreachable!(),
        }
    }

    fn store(&self, dst: Ptr<<C as Config>::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        match self {
            Array::Dyn(src, Usize::Var(len)) => {
                let mut index = index;
                src.store(dst, index, builder);
                index.offset += <Ptr<C::N> as MemVariable<C>>::size_of();
                len.store(dst, index, builder);
            }
            _ => unreachable!(),
        }
    }
}

impl<C: Config, V: FromConstant<C> + MemVariable<C>> FromConstant<C> for Array<C, V> {
    type Constant = Vec<V::Constant>;

    fn constant(value: Self::Constant, builder: &mut Builder<C>) -> Self {
        let mut array = builder.dyn_array(value.len());
        for (i, val) in value.into_iter().enumerate() {
            let val = V::constant(val, builder);
            builder.set(&mut array, i, val);
        }
        array
    }
}
