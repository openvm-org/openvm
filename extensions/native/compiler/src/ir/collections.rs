use alloc::rc::Rc;
use std::cell::RefCell;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::{
    Builder, Config, FromConstant, MemIndex, MemVariable, Ptr, RVar, Usize, Var, Variable,
};

/// A logical array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Array<C: Config, T> {
    /// Array of some local variables or constants, which can only be manipulated statically. It
    /// only exists in the DSL syntax and isn't backed by memory.
    Fixed(Rc<RefCell<Vec<Option<T>>>>),
    /// Array on heap. Index access can use variables. Length could be determined on runtime but
    /// cannot change after initialization.
    Dyn(Ptr<C::N>, Usize<C::N>),
}

impl<C: Config, V: MemVariable<C>> Array<C, V> {
    /// Gets a right value of the array.
    pub fn vec(&self) -> Vec<V> {
        match self {
            Self::Fixed(vec) => vec.borrow().iter().map(|x| x.clone().unwrap()).collect(),
            _ => panic!("array is dynamic, not fixed"),
        }
    }

    pub fn ptr(&self) -> Ptr<C::N> {
        match *self {
            Array::Dyn(ptr, _) => ptr,
            Array::Fixed(_) => panic!("cannot retrieve pointer for a compile-time array"),
        }
    }

    /// Gets the length of the array as a variable inside the DSL.
    pub fn len(&self) -> Usize<C::N> {
        match self {
            Self::Fixed(vec) => Usize::from(vec.borrow().len()),
            Self::Dyn(_, len) => len.clone(),
        }
    }

    /// Shifts the array by `shift` elements.
    /// !Attention!: the behavior of `Fixed` and `Dyn` is different. For Dyn, the shift is a view
    /// and shares memory with the original. For `Fixed`, `set`/`set_value` on slices won't impact
    /// the original array.
    pub fn shift(&self, builder: &mut Builder<C>, shift: impl Into<RVar<C::N>>) -> Array<C, V> {
        match self {
            Self::Fixed(v) => {
                let shift = shift.into();
                if let RVar::Const(_) = shift {
                    let shift = shift.value();
                    Array::Fixed(Rc::new(RefCell::new(v.borrow()[shift..].to_vec())))
                } else {
                    panic!("Cannot shift a fixed array with a variable shift");
                }
            }
            Self::Dyn(ptr, len) => {
                let len = RVar::from(len.clone());
                let shift = shift.into();
                let new_ptr = builder.eval(*ptr + shift * RVar::from(V::size_of()));
                let new_length = builder.eval(len - shift);
                Array::Dyn(new_ptr, Usize::Var(new_length))
            }
        }
    }

    /// Truncates the array to `len` elements.
    pub fn truncate(&self, builder: &mut Builder<C>, len: Usize<C::N>) {
        match self {
            Self::Fixed(v) => {
                let len = len.value();
                v.borrow_mut().truncate(len);
            }
            Self::Dyn(_, old_len) => {
                builder.assign(old_len, len);
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
        start: impl Into<RVar<C::N>>,
        end: impl Into<RVar<C::N>>,
    ) -> Array<C, V> {
        let start = start.into();
        let end = end.into();
        match self {
            Self::Fixed(v) => {
                if let (RVar::Const(_), RVar::Const(_)) = (&start, &end) {
                    Array::Fixed(Rc::new(RefCell::new(
                        v.borrow()[start.value()..end.value()].to_vec(),
                    )))
                } else {
                    panic!("Cannot slice a fixed array with a variable start or end");
                }
            }
            Self::Dyn(ptr, _) => {
                let slice_len = builder.eval(end - start);
                let address = builder.eval(ptr.address + start * RVar::from(V::size_of()));
                let ptr = Ptr { address };
                Array::Dyn(ptr, Usize::Var(slice_len))
            }
        }
    }
}

impl<C: Config> Builder<C> {
    /// Initialize an array of fixed length `len`. The entries will be uninitialized.
    pub fn array<V: MemVariable<C>>(&mut self, len: impl Into<RVar<C::N>>) -> Array<C, V> {
        let len = len.into();
        if self.flags.static_only {
            self.uninit_fixed_array(len.value())
        } else {
            self.dyn_array(len)
        }
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
    pub fn dyn_array<V: MemVariable<C>>(&mut self, len: impl Into<RVar<C::N>>) -> Array<C, V> {
        let len: Var<_> = self.eval(len.into());
        let ptr = self.alloc(len, V::size_of());
        Array::Dyn(ptr, Usize::Var(len))
    }

    pub fn get<V: MemVariable<C>, I: Into<RVar<C::N>>>(
        &mut self,
        slice: &Array<C, V>,
        index: I,
    ) -> V {
        let index = index.into();

        match slice {
            Array::Fixed(slice) => {
                if let RVar::Const(_) = index {
                    let idx = index.value();
                    if let Some(ele) = &slice.borrow()[idx] {
                        ele.clone()
                    } else {
                        panic!("Cannot get an uninitialized element in a fixed slice");
                    }
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
                let var: V = self.uninit();
                self.load(var.clone(), *ptr, index);
                var
            }
        }
    }

    /// Intended to be used with `ptr` from `zip`. Assumes that:
    /// - if `slice` is `Array::Fixed`, then `ptr` is a constant index in [0, slice.len()).
    /// - if `slice` is `Array::Dyn`, then `ptr` is a variable iterator over the entries of `slice`.
    ///
    /// In both cases, loads and returns the corresponding element of `slice`.
    pub fn iter_ptr_get<V: MemVariable<C>>(&mut self, slice: &Array<C, V>, ptr: RVar<C::N>) -> V {
        match slice {
            Array::Fixed(v) => {
                if let RVar::Const(_) = ptr {
                    let idx = ptr.value();
                    v.borrow()[idx].clone().unwrap()
                } else {
                    panic!("Cannot index into a fixed slice with a variable index")
                }
            }
            Array::Dyn(_, _) => {
                let index = MemIndex {
                    index: 0.into(),
                    offset: 0,
                    size: V::size_of(),
                };
                let var: V = self.uninit();
                self.load(
                    var.clone(),
                    Ptr {
                        address: match ptr {
                            RVar::Const(_) => panic!(
                                "iter_ptr_get on dynamic array not supported for constant ptr"
                            ),
                            RVar::Val(v) => v,
                        },
                    },
                    index,
                );
                var
            }
        }
    }

    /// Intended to be used with `ptr` from `zip`. Assumes that:
    /// - if `slice` is `Array::Fixed`, then `ptr` is a constant index in [0, slice.len()).
    /// - if `slice` is `Array::Dyn`, then `ptr` is a variable iterator over the entries of `slice`.
    ///
    /// In both cases, stores the given `value` at the corresponding element of `slice`.
    pub fn iter_ptr_set<V: MemVariable<C>, Expr: Into<V::Expression>>(
        &mut self,
        slice: &Array<C, V>,
        ptr: RVar<C::N>,
        value: Expr,
    ) {
        match slice {
            Array::Fixed(v) => {
                if let RVar::Const(_) = ptr {
                    let idx = ptr.value();
                    let value = self.eval(value);
                    v.borrow_mut()[idx] = Some(value);
                } else {
                    panic!("Cannot index into a fixed slice with a variable index")
                }
            }
            Array::Dyn(_, _) => {
                let value: V = self.eval(value);
                self.store(
                    Ptr {
                        address: match ptr {
                            RVar::Const(_) => panic!(
                                "iter_ptr_set on dynamic array not supported for constant ptr"
                            ),
                            RVar::Val(v) => v,
                        },
                    },
                    MemIndex {
                        index: 0.into(),
                        offset: 0,
                        size: V::size_of(),
                    },
                    value,
                );
            }
        }
    }

    pub fn set<V: MemVariable<C>, I: Into<RVar<C::N>>, Expr: Into<V::Expression>>(
        &mut self,
        slice: &Array<C, V>,
        index: I,
        value: Expr,
    ) {
        let index = index.into();

        match slice {
            Array::Fixed(v) => {
                if let RVar::Const(_) = index {
                    let idx = index.value();
                    let value = self.eval(value);
                    v.borrow_mut()[idx] = Some(value);
                } else {
                    panic!("Cannot index into a fixed slice with a variable index")
                }
            }
            Array::Dyn(ptr, _) => {
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

    pub fn set_value<V: MemVariable<C>, I: Into<RVar<C::N>>>(
        &mut self,
        slice: &Array<C, V>,
        index: I,
        value: V,
    ) {
        let index = index.into();

        match slice {
            Array::Fixed(v) => {
                if let RVar::Const(_) = index {
                    let idx = index.value();
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
                builder.assign(lhs_ptr, rhs_ptr);
                builder.assign(lhs_len, rhs_len);
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
                builder.assert_eq::<Usize<_>>(lhs_len.clone(), rhs_len);

                builder.range(0, lhs_len).for_each(|idx_vec, builder| {
                    let a = builder.get(&lhs, idx_vec[0]);
                    let b = builder.get(&rhs, idx_vec[0]);
                    builder.assert_eq::<T>(a, b);
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
        let array = builder.dyn_array(value.len());
        for (i, val) in value.into_iter().enumerate() {
            let val = V::constant(val, builder);
            builder.set(&array, i, val);
        }
        array
    }
}

/// Unsafe transmute from array of one type to another.
///
/// SAFETY: only use this if the memory layout of types `S` and `T` align.
/// Only usable for `Array::Dyn`, will panic otherwise.
pub fn unsafe_array_transmute<C: Config, S, T>(arr: Array<C, S>) -> Array<C, T> {
    if let Array::Dyn(ptr, len) = arr {
        Array::Dyn(ptr, len)
    } else {
        unreachable!()
    }
}

#[allow(clippy::len_without_is_empty)]
pub trait ArrayLike<C: Config> {
    fn len(&self) -> Usize<C::N>;

    fn ptr(&self) -> Ptr<C::N>;

    fn is_fixed(&self) -> bool;

    fn element_size_of(&self) -> usize;
}

impl<C: Config, T: MemVariable<C>> ArrayLike<C> for Array<C, T> {
    fn len(&self) -> Usize<C::N> {
        self.len()
    }

    fn ptr(&self) -> Ptr<C::N> {
        self.ptr()
    }

    fn is_fixed(&self) -> bool {
        match self {
            Array::Fixed(_) => true,
            Array::Dyn(_, _) => false,
        }
    }

    fn element_size_of(&self) -> usize {
        T::size_of()
    }
}
