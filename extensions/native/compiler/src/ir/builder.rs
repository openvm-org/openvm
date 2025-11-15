use std::{iter::Zip, vec::IntoIter};

use backtrace::Backtrace;
use itertools::izip;
use openvm_native_compiler_derive::iter_zip;
use openvm_stark_backend::p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra};
use serde::{Deserialize, Serialize};

use super::{
    Array, Config, DslIr, Ext, Felt, FromConstant, MemIndex, MemVariable, RVar, SymbolicExt,
    SymbolicFelt, SymbolicVar, Usize, Var, Variable, WitnessRef,
};
use crate::ir::{collections::ArrayLike, Ptr};

/// TracedVec is a Vec wrapper that records a trace whenever an element is pushed. When extending
/// from another TracedVec, the traces are copied over.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracedVec<T> {
    pub vec: Vec<T>,
    pub traces: Vec<Option<Backtrace>>,
}

impl<T> Default for TracedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> From<Vec<T>> for TracedVec<T> {
    fn from(vec: Vec<T>) -> Self {
        let len = vec.len();
        Self {
            vec,
            traces: vec![None; len],
        }
    }
}

impl<T> TracedVec<T> {
    pub const fn new() -> Self {
        Self {
            vec: Vec::new(),
            traces: Vec::new(),
        }
    }

    pub fn push(&mut self, value: T) {
        self.vec.push(value);
        self.traces.push(None);
    }

    /// Pushes a value to the vector and records a backtrace if RUST_BACKTRACE is enabled
    pub fn trace_push(&mut self, value: T) {
        self.vec.push(value);
        if std::env::var_os("RUST_BACKTRACE").is_none() {
            self.traces.push(None);
        } else {
            self.traces.push(Some(Backtrace::new_unresolved()));
        }
    }

    pub fn extend<I: IntoIterator<Item = (T, Option<Backtrace>)>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let len = iter.size_hint().0;
        self.vec.reserve(len);
        self.traces.reserve(len);
        for (value, trace) in iter {
            self.vec.push(value);
            self.traces.push(trace);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }
}

impl<T> IntoIterator for TracedVec<T> {
    type Item = (T, Option<Backtrace>);
    type IntoIter = Zip<IntoIter<T>, IntoIter<Option<Backtrace>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter().zip(self.traces)
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct BuilderFlags {
    pub debug: bool,
    /// If true, branching/looping/heap memory is disabled.
    pub static_only: bool,
}

/// A builder for the DSL.
///
/// Can compile to both assembly and a set of constraints.
#[derive(Debug, Clone, Default)]
pub struct Builder<C: Config> {
    pub(crate) var_count: u32,
    pub(crate) felt_count: u32,
    pub(crate) ext_count: u32,
    pub operations: TracedVec<DslIr<C>>,
    pub(crate) nb_public_values: Option<Var<C::N>>,
    pub(crate) witness_var_count: u32,
    pub(crate) witness_felt_count: u32,
    pub(crate) witness_ext_count: u32,
    pub(crate) witness_space: Vec<Vec<WitnessRef>>,
    pub flags: BuilderFlags,
    pub is_sub_builder: bool,
}

impl<C: Config> Builder<C> {
    /// Creates a new builder with a given number of counts for each type.
    pub fn create_sub_builder(&self) -> Self {
        Self {
            var_count: self.var_count,
            felt_count: self.felt_count,
            ext_count: self.ext_count,
            // Witness counts are only used when the target is a circuit.  And sub-builders are
            // not used when the target is a circuit, so it is fine to set the witness counts to 0.
            witness_var_count: 0,
            witness_felt_count: 0,
            witness_ext_count: 0,
            witness_space: Default::default(),
            operations: Default::default(),
            nb_public_values: self.nb_public_values,
            flags: self.flags,
            is_sub_builder: true,
        }
    }

    /// Pushes an operation to the builder.
    pub fn push(&mut self, op: DslIr<C>) {
        self.operations.push(op);
    }

    /// Pushes an operation to the builder and records a trace if RUST_BACKTRACE=1.
    pub fn trace_push(&mut self, op: DslIr<C>) {
        self.operations.trace_push(op);
    }

    /// Creates an uninitialized variable.
    pub fn uninit<V: Variable<C>>(&mut self) -> V {
        V::uninit(self)
    }

    /// Evaluates an expression and returns a variable.
    pub fn eval<V: Variable<C>, E: Into<V::Expression>>(&mut self, expr: E) -> V {
        V::eval(self, expr)
    }

    /// Evaluates an expression and returns a right value.
    pub fn eval_expr(&mut self, expr: impl Into<SymbolicVar<C::N>>) -> RVar<C::N> {
        let expr = expr.into();
        match expr {
            SymbolicVar::Const(c, _) => RVar::Const(c),
            SymbolicVar::Val(val, _) => RVar::Val(val),
            _ => {
                let ret: Var<_> = self.eval(expr);
                RVar::Val(ret)
            }
        }
    }

    /// Increments Usize by one.
    pub fn inc(&mut self, u: &Usize<C::N>) {
        self.assign(u, u.clone() + RVar::one());
    }

    /// Evaluates a constant expression and returns a variable.
    pub fn constant<V: FromConstant<C>>(&mut self, value: V::Constant) -> V {
        V::constant(value, self)
    }

    /// Assigns an expression to a variable.
    pub fn assign<V: Variable<C>, E: Into<V::Expression>>(&mut self, dst: &V, expr: E) {
        dst.assign(expr.into(), self);
    }

    /// Casts a Felt to a Var.
    pub fn cast_felt_to_var(&mut self, felt: Felt<C::F>) -> Var<C::N> {
        let var: Var<_> = self.uninit();
        self.operations.push(DslIr::CastFV(var, felt));
        var
    }
    /// Casts a Var to a Felt.
    pub fn unsafe_cast_var_to_felt(&mut self, var: Var<C::N>) -> Felt<C::F> {
        assert!(!self.flags.static_only, "dynamic mode only");
        let felt: Felt<_> = self.uninit();
        self.operations.push(DslIr::UnsafeCastVF(felt, var));
        felt
    }

    /// Asserts that a Usize is non-zero
    pub fn assert_nonzero(&mut self, u: &Usize<C::N>) {
        self.operations.push(DslIr::AssertNonZero(u.clone()));
    }

    /// Asserts that two expressions are equal.
    pub fn assert_eq<V: Variable<C>>(
        &mut self,
        lhs: impl Into<V::Expression>,
        rhs: impl Into<V::Expression>,
    ) {
        V::assert_eq(lhs, rhs, self);
    }

    /// Assert that two vars are equal.
    pub fn assert_var_eq<LhsExpr: Into<SymbolicVar<C::N>>, RhsExpr: Into<SymbolicVar<C::N>>>(
        &mut self,
        lhs: LhsExpr,
        rhs: RhsExpr,
    ) {
        self.assert_eq::<Var<C::N>>(lhs, rhs);
    }

    /// Assert that two felts are equal.
    pub fn assert_felt_eq<LhsExpr: Into<SymbolicFelt<C::F>>, RhsExpr: Into<SymbolicFelt<C::F>>>(
        &mut self,
        lhs: LhsExpr,
        rhs: RhsExpr,
    ) {
        self.assert_eq::<Felt<C::F>>(lhs, rhs);
    }

    /// Assert that two exts are equal.
    pub fn assert_ext_eq<
        LhsExpr: Into<SymbolicExt<C::F, C::EF>>,
        RhsExpr: Into<SymbolicExt<C::F, C::EF>>,
    >(
        &mut self,
        lhs: LhsExpr,
        rhs: RhsExpr,
    ) {
        self.assert_eq::<Ext<C::F, C::EF>>(lhs, rhs);
    }

    pub fn assert_usize_eq<LhsExpr: Into<SymbolicVar<C::N>>, RhsExpr: Into<SymbolicVar<C::N>>>(
        &mut self,
        lhs: LhsExpr,
        rhs: RhsExpr,
    ) {
        self.assert_eq::<Usize<C::N>>(lhs, rhs);
    }

    /// Assert that two arrays are equal.
    pub fn assert_var_array_eq(&mut self, lhs: &Array<C, Var<C::N>>, rhs: &Array<C, Var<C::N>>) {
        self.assert_var_eq(lhs.len(), rhs.len());
        self.range(0, lhs.len()).for_each(|idx_vec, builder| {
            let l = builder.get(lhs, idx_vec[0]);
            let r = builder.get(rhs, idx_vec[0]);
            builder.assert_var_eq(l, r);
        });
    }

    /// Evaluate a block of operations if two expressions are equal.
    pub fn if_eq<LhsExpr: Into<SymbolicVar<C::N>>, RhsExpr: Into<SymbolicVar<C::N>>>(
        &mut self,
        lhs: LhsExpr,
        rhs: RhsExpr,
    ) -> IfBuilder<'_, C> {
        IfBuilder {
            lhs: lhs.into(),
            rhs: rhs.into(),
            is_eq: true,
            builder: self,
        }
    }

    /// Evaluate a block of operations if two expressions are not equal.
    pub fn if_ne<LhsExpr: Into<SymbolicVar<C::N>>, RhsExpr: Into<SymbolicVar<C::N>>>(
        &mut self,
        lhs: LhsExpr,
        rhs: RhsExpr,
    ) -> IfBuilder<'_, C> {
        IfBuilder {
            lhs: lhs.into(),
            rhs: rhs.into(),
            is_eq: false,
            builder: self,
        }
    }

    /// Asserts that lhs is less than rhs in time O(rhs).
    pub fn assert_less_than_slow_small_rhs<
        LhsExpr: Into<SymbolicVar<C::N>>,
        RhsExpr: Into<SymbolicVar<C::N>>,
    >(
        &mut self,
        lhs: LhsExpr,
        rhs: RhsExpr,
    ) {
        let lhs: Usize<_> = self.eval(lhs.into());
        let rhs: Usize<_> = self.eval(rhs.into());
        let product: Usize<_> = self.eval(lhs.clone());
        self.range(1, rhs).for_each(|i_vec, builder| {
            let i = i_vec[0];
            let diff: Usize<_> = builder.eval(lhs.clone() - i);
            builder.assign(&product, product.clone() * diff);
        });
        self.assert_usize_eq(product, RVar::from(0));
    }

    /// Asserts that lhs is less than rhs in time O(log(lhs) + log(rhs)).
    ///
    /// Only works for Felt == BabyBear and in the VM.
    ///
    /// Uses bit decomposition hint, which has large constant factor overhead, so prefer
    /// [Self::assert_less_than_slow_small_rhs] when rhs is small.
    pub fn assert_less_than_slow_bit_decomp(&mut self, lhs: Var<C::N>, rhs: Var<C::N>) {
        let lhs = self.unsafe_cast_var_to_felt(lhs);
        let rhs = self.unsafe_cast_var_to_felt(rhs);

        let lhs_bits = self.num2bits_f(lhs, C::N::bits() as u32);
        let rhs_bits = self.num2bits_f(rhs, C::N::bits() as u32);

        let is_lt: Var<_> = self.eval(C::N::ZERO);

        iter_zip!(self, lhs_bits, rhs_bits).for_each(|ptr_vec, builder| {
            let lhs_bit = builder.iter_ptr_get(&lhs_bits, ptr_vec[0]);
            let rhs_bit = builder.iter_ptr_get(&rhs_bits, ptr_vec[1]);

            builder.if_ne(lhs_bit, rhs_bit).then(|builder| {
                builder.assign(&is_lt, rhs_bit);
            });
        });
        self.assert_var_eq(is_lt, C::N::ONE);
    }

    /// asserts that x has at most num_bits bits
    pub fn range_check_var(&mut self, x: Var<C::N>, num_bits: usize) {
        assert!(!self.flags.static_only, "range_check_var is dynamic");
        assert!(num_bits <= 30);
        self.trace_push(DslIr::RangeCheckV(x, num_bits));
    }

    /// Evaluate a block of operations over a range from start to end.
    pub fn range(
        &mut self,
        start: impl Into<RVar<C::N>>,
        end: impl Into<RVar<C::N>>,
    ) -> IteratorBuilder<'_, C> {
        self.range_with_step(start, end, C::N::ONE)
    }
    /// Evaluate a block of operations over a range from start to end with a custom step.
    pub fn range_with_step(
        &mut self,
        start: impl Into<RVar<C::N>>,
        end: impl Into<RVar<C::N>>,
        step: C::N,
    ) -> IteratorBuilder<'_, C> {
        let start = start.into();
        let end0 = end.into();
        IteratorBuilder {
            starts: vec![start],
            end0,
            step_sizes: vec![step],
            builder: self,
        }
    }

    pub fn zip<'a>(
        &'a mut self,
        arrays: &'a [Box<dyn ArrayLike<C> + 'a>],
    ) -> IteratorBuilder<'a, C> {
        assert!(!arrays.is_empty());
        if arrays.iter().all(|array| array.is_fixed()) {
            IteratorBuilder {
                starts: vec![RVar::zero(); arrays.len()],
                end0: arrays[0].len().into(),
                step_sizes: vec![C::N::ONE; arrays.len()],
                builder: self,
            }
        } else if arrays.iter().all(|array| !array.is_fixed()) {
            IteratorBuilder {
                starts: arrays
                    .iter()
                    .map(|array| array.ptr().address.into())
                    .collect(),
                end0: {
                    let len: RVar<C::N> = arrays[0].len().into();
                    let size = arrays[0].element_size_of();
                    let end: Var<C::N> =
                        self.eval(arrays[0].ptr().address + len * RVar::from(size));
                    end.into()
                },
                step_sizes: arrays
                    .iter()
                    .map(|array| C::N::from_canonical_usize(array.element_size_of()))
                    .collect(),
                builder: self,
            }
        } else {
            panic!("Cannot use zipped pointer iterator with mixed arrays");
        }
    }

    pub fn print_debug(&mut self, val: usize) {
        let constant = self.eval(C::N::from_canonical_usize(val));
        self.print_v(constant);
    }

    /// Print a variable.
    pub fn print_v(&mut self, dst: Var<C::N>) {
        self.operations.push(DslIr::PrintV(dst));
    }

    /// Print a felt.
    pub fn print_f(&mut self, dst: Felt<C::F>) {
        self.operations.push(DslIr::PrintF(dst));
    }

    /// Print an ext.
    pub fn print_e(&mut self, dst: Ext<C::F, C::EF>) {
        self.operations.push(DslIr::PrintE(dst));
    }

    pub fn hint_var(&mut self) -> Var<C::N> {
        let ptr = self.alloc(RVar::one(), 1);
        // Prepare data for hinting.
        self.operations.push(DslIr::HintFelt());
        let index = MemIndex {
            index: RVar::zero(),
            offset: 0,
            size: 1,
        };
        self.operations.push(DslIr::StoreHintWord(ptr, index));
        let v: Var<C::N> = self.uninit();
        self.load(v, ptr, index);
        v
    }

    pub fn hint_felt(&mut self) -> Felt<C::F> {
        let ptr = self.alloc(RVar::one(), 1);
        // Prepare data for hinting.
        self.operations.push(DslIr::HintFelt());
        let index = MemIndex {
            index: RVar::zero(),
            offset: 0,
            size: 1,
        };
        self.operations.push(DslIr::StoreHintWord(ptr, index));
        let f: Felt<C::F> = self.uninit();
        self.load(f, ptr, index);
        f
    }

    pub fn hint_ext(&mut self) -> Ext<C::F, C::EF> {
        let flattened = self.hint_felts_fixed(C::EF::D);

        // Simply recast memory as Array<Ext>.
        let array: Array<C, Ext<_, _>> = match flattened {
            Array::Fixed(_) => unreachable!(),
            Array::Dyn(ptr, _) => Array::Dyn(ptr, Usize::from(1)),
        };
        self.get(&array, 0)
    }

    /// Hint a vector of variables.
    ///
    /// Writes the next element of the witness stream into memory and returns it.
    pub fn hint_vars(&mut self) -> Array<C, Var<C::N>> {
        self.hint_words()
    }

    /// Hint a vector of felts.
    pub fn hint_felts(&mut self) -> Array<C, Felt<C::F>> {
        self.hint_words()
    }

    pub fn hint_felts_fixed(&mut self, len: impl Into<RVar<C::N>>) -> Array<C, Felt<C::F>> {
        self.hint_words_fixed(len)
    }

    /// Hints an array of V and assumes V::size_of() == 1.
    fn hint_words<V: MemVariable<C>>(&mut self) -> Array<C, V> {
        assert_eq!(V::size_of(), 1);

        // Allocate space for the length variable. We assume that mem[ptr..] is empty.
        let ptr = self.alloc(RVar::one(), 1);

        // Prepare length + data for hinting.
        self.operations.push(DslIr::HintInputVec());

        // Write and retrieve length hint.
        let index = MemIndex {
            index: RVar::zero(),
            offset: 0,
            size: 1,
        };
        // MemIndex.index share the same pointer, but it doesn't matter.
        self.operations.push(DslIr::StoreHintWord(ptr, index));

        let vlen: Var<C::N> = self.uninit();
        self.load(vlen, ptr, index);
        let arr = self.dyn_array(vlen);

        // Write the content hints directly into the array memory.
        iter_zip!(self, arr).for_each(|ptr_vec, builder| {
            let index = MemIndex {
                index: 0.into(),
                offset: 0,
                size: 1,
            };
            builder.operations.push(DslIr::StoreHintWord(
                Ptr {
                    address: ptr_vec[0].variable(),
                },
                index,
            ));
        });
        arr
    }

    /// Hints an array of V and assumes V::size_of() == 1.
    fn hint_words_fixed<V: MemVariable<C>>(&mut self, len: impl Into<RVar<C::N>>) -> Array<C, V> {
        assert_eq!(V::size_of(), 1);

        let arr = self.dyn_array(len.into());
        // Write the content hints directly into the array memory.
        iter_zip!(self, arr).for_each(|ptr_vec, builder| {
            let index = MemIndex {
                index: 0.into(),
                offset: 0,
                size: 1,
            };
            builder.operations.push(DslIr::HintFelt());
            builder.operations.push(DslIr::StoreHintWord(
                Ptr {
                    address: ptr_vec[0].variable(),
                },
                index,
            ));
        });
        arr
    }

    /// Hint a vector of exts.
    ///
    /// Emits two hint opcodes: the first for the number of exts, the second for the list of exts
    /// themselves.
    pub fn hint_exts(&mut self) -> Array<C, Ext<C::F, C::EF>> {
        let len = self.hint_var();
        let flattened = self.hint_felts();

        let size = <Ext<C::F, C::EF> as MemVariable<C>>::size_of();
        self.assert_usize_eq(flattened.len(), len * C::N::from_canonical_usize(size));

        // Simply recast memory as Array<Ext>.
        match flattened {
            Array::Fixed(_) => unreachable!(),
            Array::Dyn(ptr, _) => Array::Dyn(ptr, Usize::Var(len)),
        }
    }

    /// Move data from input stream into hint space. Return an ID which can be used to load the
    /// data at runtime.
    pub fn hint_load(&mut self) -> Var<C::N> {
        self.trace_push(DslIr::HintLoad());
        let ptr = self.alloc(RVar::one(), 1);
        let index = MemIndex {
            index: RVar::zero(),
            offset: 0,
            size: 1,
        };
        // MemIndex.index share the same pointer, but it doesn't matter.
        self.operations.push(DslIr::StoreHintWord(ptr, index));
        let id: Var<C::N> = self.uninit();
        self.load(id, ptr, index);
        id
    }

    pub fn witness_var(&mut self) -> Var<C::N> {
        assert!(
            !self.is_sub_builder,
            "Cannot create a witness var with a sub builder"
        );
        let witness = self.uninit();
        self.operations
            .push(DslIr::WitnessVar(witness, self.witness_var_count));
        self.witness_var_count += 1;
        witness
    }

    pub fn witness_felt(&mut self) -> Felt<C::F> {
        assert!(
            !self.is_sub_builder,
            "Cannot create a witness felt with a sub builder"
        );
        let witness = self.uninit();
        self.operations
            .push(DslIr::WitnessFelt(witness, self.witness_felt_count));
        self.witness_felt_count += 1;
        witness
    }

    pub fn witness_ext(&mut self) -> Ext<C::F, C::EF> {
        assert!(
            !self.is_sub_builder,
            "Cannot create a witness ext with a sub builder"
        );
        let witness = self.uninit();
        self.operations
            .push(DslIr::WitnessExt(witness, self.witness_ext_count));
        self.witness_ext_count += 1;
        witness
    }

    pub fn witness_load(&mut self, witness_refs: Vec<WitnessRef>) -> Usize<C::N> {
        assert!(
            !self.is_sub_builder,
            "Cannot load witness refs with a sub builder"
        );
        let ret = self.witness_space.len();
        self.witness_space.push(witness_refs);
        ret.into()
    }

    pub fn get_witness_refs(&self, id: Usize<C::N>) -> &[WitnessRef] {
        self.witness_space.get(id.value()).unwrap()
    }

    /// Throws an error.
    pub fn error(&mut self) {
        self.operations.trace_push(DslIr::Error());
    }

    fn get_nb_public_values(&mut self) -> Var<C::N> {
        assert!(
            !self.is_sub_builder,
            "Cannot commit to public values with a sub builder"
        );
        if self.nb_public_values.is_none() {
            self.nb_public_values = Some(self.eval(C::N::ZERO));
        }
        *self.nb_public_values.as_ref().unwrap()
    }

    fn commit_public_value_and_increment(&mut self, val: Felt<C::F>, nb_public_values: Var<C::N>) {
        assert!(
            !self.flags.static_only,
            "Static mode should use static_commit_public_value"
        );
        self.operations.push(DslIr::Publish(val, nb_public_values));
        self.assign(&nb_public_values, nb_public_values + C::N::ONE);
    }

    /// Commits a Var as public value. This value will be constrained when verified. This method
    /// should only be used in static mode.
    pub fn static_commit_public_value(&mut self, index: usize, val: Var<C::N>) {
        assert!(
            self.flags.static_only,
            "Dynamic mode should use commit_public_value instead."
        );
        self.operations.push(DslIr::CircuitPublish(val, index));
    }

    /// Register and commits a felt as public value.  This value will be constrained when verified.
    pub fn commit_public_value(&mut self, val: Felt<C::F>) {
        let nb_public_values = self.get_nb_public_values();
        self.commit_public_value_and_increment(val, nb_public_values);
    }

    /// Commits an array of felts in public values.
    pub fn commit_public_values(&mut self, vals: &Array<C, Felt<C::F>>) {
        let nb_public_values = self.get_nb_public_values();
        let len = vals.len();
        self.range(0, len).for_each(|idx_vec, builder| {
            let val = builder.get(vals, idx_vec[0]);
            builder.commit_public_value_and_increment(val, nb_public_values);
        });
    }

    pub fn cycle_tracker_start(&mut self, name: &str) {
        self.operations
            .push(DslIr::CycleTrackerStart(name.to_string()));
    }

    pub fn cycle_tracker_end(&mut self, name: &str) {
        self.operations
            .push(DslIr::CycleTrackerEnd(name.to_string()));
    }

    pub fn halt(&mut self) {
        self.operations.push(DslIr::Halt);
    }
}

/// A builder for the DSL that handles if statements.
pub struct IfBuilder<'a, C: Config> {
    lhs: SymbolicVar<C::N>,
    rhs: SymbolicVar<C::N>,
    is_eq: bool,
    pub(crate) builder: &'a mut Builder<C>,
}

/// A set of conditions that if statements can be based on.
enum IfCondition<N> {
    EqConst(N, N),
    NeConst(N, N),
    Eq(Var<N>, Var<N>),
    EqI(Var<N>, N),
    Ne(Var<N>, Var<N>),
    NeI(Var<N>, N),
}

impl<C: Config> IfBuilder<'_, C> {
    pub fn then(&mut self, mut f: impl FnMut(&mut Builder<C>)) {
        // Get the condition reduced from the expressions for lhs and rhs.
        let condition = self.condition();
        // Early return for const branches.
        match condition {
            IfCondition::EqConst(lhs, rhs) => {
                if lhs == rhs {
                    return f(self.builder);
                }
                return;
            }
            IfCondition::NeConst(lhs, rhs) => {
                if lhs != rhs {
                    return f(self.builder);
                }
                return;
            }
            _ => (),
        }
        assert!(
            !self.builder.flags.static_only,
            "Cannot use dynamic branch in static mode"
        );

        // Execute the `then` block and collect the instructions.
        let mut f_builder = self.builder.create_sub_builder();
        f(&mut f_builder);
        let then_instructions = f_builder.operations;

        // Dispatch instructions to the correct conditional block.
        match condition {
            IfCondition::Eq(lhs, rhs) => {
                let op = DslIr::IfEq(lhs, rhs, then_instructions, Default::default());
                self.builder.operations.push(op);
            }
            IfCondition::EqI(lhs, rhs) => {
                let op = DslIr::IfEqI(lhs, rhs, then_instructions, Default::default());
                self.builder.operations.push(op);
            }
            IfCondition::Ne(lhs, rhs) => {
                let op = DslIr::IfNe(lhs, rhs, then_instructions, Default::default());
                self.builder.operations.push(op);
            }
            IfCondition::NeI(lhs, rhs) => {
                let op = DslIr::IfNeI(lhs, rhs, then_instructions, Default::default());
                self.builder.operations.push(op);
            }
            _ => unreachable!("Const if should have returned early"),
        }
    }

    pub fn then_or_else(
        &mut self,
        mut then_f: impl FnMut(&mut Builder<C>),
        mut else_f: impl FnMut(&mut Builder<C>),
    ) {
        // Get the condition reduced from the expressions for lhs and rhs.
        let condition = self.condition();
        // Early return for const branches.
        match condition {
            IfCondition::EqConst(lhs, rhs) => {
                if lhs == rhs {
                    return then_f(self.builder);
                }
                return else_f(self.builder);
            }
            IfCondition::NeConst(lhs, rhs) => {
                if lhs != rhs {
                    return then_f(self.builder);
                }
                return else_f(self.builder);
            }
            _ => (),
        }
        assert!(
            !self.builder.flags.static_only,
            "Cannot use dynamic branch in static mode"
        );
        let mut then_builder = self.builder.create_sub_builder();

        // Execute the `then` and `else_then` blocks and collect the instructions.
        then_f(&mut then_builder);
        let then_instructions = then_builder.operations;

        let mut else_builder = self.builder.create_sub_builder();
        else_f(&mut else_builder);
        let else_instructions = else_builder.operations;

        // Dispatch instructions to the correct conditional block.
        match condition {
            IfCondition::Eq(lhs, rhs) => {
                let op = DslIr::IfEq(lhs, rhs, then_instructions, else_instructions);
                self.builder.operations.push(op);
            }
            IfCondition::EqI(lhs, rhs) => {
                let op = DslIr::IfEqI(lhs, rhs, then_instructions, else_instructions);
                self.builder.operations.push(op);
            }
            IfCondition::Ne(lhs, rhs) => {
                let op = DslIr::IfNe(lhs, rhs, then_instructions, else_instructions);
                self.builder.operations.push(op);
            }
            IfCondition::NeI(lhs, rhs) => {
                let op = DslIr::IfNeI(lhs, rhs, then_instructions, else_instructions);
                self.builder.operations.push(op);
            }
            _ => unreachable!("Const if should have returned early"),
        }
    }

    fn condition(&mut self) -> IfCondition<C::N> {
        match (self.lhs.clone(), self.rhs.clone(), self.is_eq) {
            (SymbolicVar::Const(lhs, _), SymbolicVar::Const(rhs, _), true) => {
                IfCondition::EqConst(lhs, rhs)
            }
            (SymbolicVar::Const(lhs, _), SymbolicVar::Const(rhs, _), false) => {
                IfCondition::NeConst(lhs, rhs)
            }
            (SymbolicVar::Const(lhs, _), SymbolicVar::Val(rhs, _), true) => {
                IfCondition::EqI(rhs, lhs)
            }
            (SymbolicVar::Const(lhs, _), SymbolicVar::Val(rhs, _), false) => {
                IfCondition::NeI(rhs, lhs)
            }
            (SymbolicVar::Const(lhs, _), rhs, true) => {
                let rhs: Var<C::N> = self.builder.eval(rhs);
                IfCondition::EqI(rhs, lhs)
            }
            (SymbolicVar::Const(lhs, _), rhs, false) => {
                let rhs: Var<C::N> = self.builder.eval(rhs);
                IfCondition::NeI(rhs, lhs)
            }
            (SymbolicVar::Val(lhs, _), SymbolicVar::Const(rhs, _), true) => {
                let lhs: Var<C::N> = self.builder.eval(lhs);
                IfCondition::EqI(lhs, rhs)
            }
            (SymbolicVar::Val(lhs, _), SymbolicVar::Const(rhs, _), false) => {
                let lhs: Var<C::N> = self.builder.eval(lhs);
                IfCondition::NeI(lhs, rhs)
            }
            (lhs, SymbolicVar::Const(rhs, _), true) => {
                let lhs: Var<C::N> = self.builder.eval(lhs);
                IfCondition::EqI(lhs, rhs)
            }
            (lhs, SymbolicVar::Const(rhs, _), false) => {
                let lhs: Var<C::N> = self.builder.eval(lhs);
                IfCondition::NeI(lhs, rhs)
            }
            (SymbolicVar::Val(lhs, _), SymbolicVar::Val(rhs, _), true) => IfCondition::Eq(lhs, rhs),
            (SymbolicVar::Val(lhs, _), SymbolicVar::Val(rhs, _), false) => {
                IfCondition::Ne(lhs, rhs)
            }
            (SymbolicVar::Val(lhs, _), rhs, true) => {
                let rhs: Var<C::N> = self.builder.eval(rhs);
                IfCondition::Eq(lhs, rhs)
            }
            (SymbolicVar::Val(lhs, _), rhs, false) => {
                let rhs: Var<C::N> = self.builder.eval(rhs);
                IfCondition::Ne(lhs, rhs)
            }
            (lhs, SymbolicVar::Val(rhs, _), true) => {
                let lhs: Var<C::N> = self.builder.eval(lhs);
                IfCondition::Eq(lhs, rhs)
            }
            (lhs, SymbolicVar::Val(rhs, _), false) => {
                let lhs: Var<C::N> = self.builder.eval(lhs);
                IfCondition::Ne(lhs, rhs)
            }
            (lhs, rhs, true) => {
                let lhs: Var<C::N> = self.builder.eval(lhs);
                let rhs: Var<C::N> = self.builder.eval(rhs);
                IfCondition::Eq(lhs, rhs)
            }
            (lhs, rhs, false) => {
                let lhs: Var<C::N> = self.builder.eval(lhs);
                let rhs: Var<C::N> = self.builder.eval(rhs);
                IfCondition::Ne(lhs, rhs)
            }
        }
    }
}

// iterates through zipped pointers
pub struct IteratorBuilder<'a, C: Config> {
    starts: Vec<RVar<C::N>>,
    end0: RVar<C::N>,
    step_sizes: Vec<C::N>,
    builder: &'a mut Builder<C>,
}

impl<C: Config> IteratorBuilder<'_, C> {
    pub fn for_each(&mut self, mut f: impl FnMut(Vec<RVar<C::N>>, &mut Builder<C>)) {
        assert!(self.starts.len() == self.step_sizes.len());
        assert!(!self.starts.is_empty());

        if self.starts.iter().all(|start| start.is_const()) && self.end0.is_const() {
            self.for_each_unrolled(|ptrs, builder| {
                f(ptrs, builder);
            });
            return;
        }

        self.for_each_dynamic(|ptrs, builder| {
            f(ptrs, builder);
        });
    }

    fn for_each_unrolled(&mut self, mut f: impl FnMut(Vec<RVar<C::N>>, &mut Builder<C>)) {
        let mut ptrs: Vec<_> = self
            .starts
            .iter()
            .map(|start| start.field_value())
            .collect();
        let end0 = self.end0.field_value();
        while ptrs[0] != end0 {
            f(
                ptrs.iter().map(|ptr| RVar::Const(*ptr)).collect(),
                self.builder,
            );
            for (ptr, step_size) in izip!(&mut ptrs, &self.step_sizes) {
                *ptr += *step_size;
            }
        }
    }

    fn for_each_dynamic(&mut self, mut f: impl FnMut(Vec<RVar<C::N>>, &mut Builder<C>)) {
        assert!(
            !self.builder.flags.static_only,
            "Cannot use dynamic loop in static mode"
        );

        let loop_variables: Vec<Var<C::N>> = (0..self.starts.len())
            .map(|_| self.builder.uninit())
            .collect();
        let mut loop_body_builder = self.builder.create_sub_builder();

        f(
            loop_variables.iter().map(|&v| v.into()).collect(),
            &mut loop_body_builder,
        );

        let loop_instructions = loop_body_builder.operations;
        let op = DslIr::ZipFor(
            self.starts.clone(),
            self.end0,
            self.step_sizes.clone(),
            loop_variables,
            loop_instructions,
        );
        self.builder.operations.push(op);
    }
}
