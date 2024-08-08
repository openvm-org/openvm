use super::{Builder, Config, FromConstant, MemIndex, MemVariable, Ptr, Usize, Variable};

/// A logical array.
#[derive(Debug, Clone)]
pub struct RefPtr<C: Config, T> {
    pub ptr: Ptr<C::N>,
    phantom: std::marker::PhantomData<T>,
}

impl<C: Config> Builder<C> {
    /// Initialize a new instance of type T. The entries will be uninitialized.
    pub fn new_ref_ptr<V: MemVariable<C>>(&mut self) -> RefPtr<C, V> {
        let ptr = self.alloc(Usize::Const(1), V::size_of());
        RefPtr {
            ptr,
            phantom: std::marker::PhantomData,
        }
    }

    /// Copies the referenced data onto the stack
    pub fn deref<V: MemVariable<C>>(&mut self, ptr: &RefPtr<C, V>) -> V {
        let index = MemIndex {
            index: Usize::Const(0),
            offset: 0,
            size: V::size_of(),
        };
        let var: V = self.uninit();
        self.load(var.clone(), ptr.ptr, index);
        var
    }

    pub fn set_to_expr<V: MemVariable<C>, Expr: Into<V::Expression>>(
        &mut self,
        ptr: &mut RefPtr<C, V>,
        value: Expr,
    ) {
        let index = MemIndex {
            index: Usize::Const(0),
            offset: 0,
            size: V::size_of(),
        };
        let value: V = self.eval(value);
        self.store(ptr.ptr, index, value);
    }

    pub fn set_to_value<V: MemVariable<C>>(&mut self, ptr: &mut RefPtr<C, V>, value: V) {
        let index = MemIndex {
            index: Usize::Const(0),
            offset: 0,
            size: V::size_of(),
        };
        self.store(ptr.ptr, index, value);
    }
}

impl<C: Config, T: MemVariable<C>> Variable<C> for RefPtr<C, T> {
    type Expression = Self;

    fn uninit(builder: &mut Builder<C>) -> Self {
        builder.new_ref_ptr::<T>()
    }

    fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
        let (RefPtr { ptr: lhs_ptr, .. }, RefPtr { ptr: rhs_ptr, .. }) = (self, src.clone());
        {
            builder.assign(*lhs_ptr, rhs_ptr);
        }
    }

    fn assert_eq(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let a = builder.deref(&lhs);
        let b = builder.deref(&rhs);
        builder.assert_eq::<T>(a, b);
    }

    fn assert_ne(
        lhs: impl Into<Self::Expression>,
        rhs: impl Into<Self::Expression>,
        builder: &mut Builder<C>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let a = builder.deref(&lhs);
        let b = builder.deref(&rhs);
        builder.assert_ne::<T>(a, b);
    }

    // The default version calls `uninit`. If `expr` is `Fixed`, it will be converted into `Dyn`.
    fn eval(_builder: &mut Builder<C>, expr: impl Into<Self::Expression>) -> Self {
        expr.into()
    }
}

impl<C: Config, T: MemVariable<C>> MemVariable<C> for RefPtr<C, T> {
    fn size_of() -> usize {
        1
    }

    fn load(&self, src: Ptr<C::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        self.ptr.load(src, index, builder);
    }

    fn store(&self, dst: Ptr<<C as Config>::N>, index: MemIndex<C::N>, builder: &mut Builder<C>) {
        self.ptr.store(dst, index, builder);
    }
}

impl<C: Config, V: FromConstant<C> + MemVariable<C>> FromConstant<C> for RefPtr<C, V> {
    type Constant = V::Constant;

    fn constant(value: Self::Constant, builder: &mut Builder<C>) -> Self {
        let mut ref_ptr = builder.new_ref_ptr();
        let val = V::constant(value, builder);
        builder.set_to_expr(&mut ref_ptr, val);
        ref_ptr
    }
}

impl<C: Config, V: MemVariable<C>> RefPtr<C, V> {
    pub fn from_ptr(ptr: Ptr<C::N>) -> Self {
        Self {
            ptr,
            phantom: std::marker::PhantomData,
        }
    }
}
