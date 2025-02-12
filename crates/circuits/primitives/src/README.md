# Circuit Primitives

This crate contains a collection of primitives for use when building circuits. The primitives are separated into two types: standalone [Air](https://github.com/Plonky3/Plonky3/blob/main/air/src/air.rs) and [SubAir](./sub_air.rs).

The following modules contain standalone `Air`'s:
- [range](./range/README.md)
- [range_gate](./range_gate/README.md)
- [range_tuple](./range_tuple/mod.rs)
- [var_range](./var_range/mod.rs)
- [xor](./xor/README.md)
- [bitwise_op_lookup](./bitwise_op_lookup/mod.rs)

The following modules contain `SubAir`'s:
- [assert_less_than](./assert_less_than/mod.rs)
- [bigint](./bigint/README.md)
- [encoder](./encoder/mod.rs)
- [is_equal](./is_equal/mod.rs)
- [is_equal_array](./is_equal_array/mod.rs)
- [is_less_than](./is_less_than/mod.rs)
- [is_less_than_array](./is_less_than_array/mod.rs)
- [is_zero](./is_zero/mod.rs)

## SubAir

Trait with associated types intended to allow reuse of constraint logic inside other AIRs.

A `SubAir` is **not** an `Air` itself.
It is a struct that holds the means to generate a particular set of constraints, meant to be reusable within other AIRs.

The trait is designed to be maximally flexible, but typical implementations will separate the `AirContext` into two parts: `Io` and `AuxCols`. 
The `Io` part will consist of expressions (built using `AB::Expr`) that the `SubAir` does not own, while the `AuxCols` are any internal columns that the `SubAir` requires to generate its constraints. 
The `AuxCols` are columns that the `SubAir` fully owns and should be internally determined by the `SubAir` from the `Io` part. These `AuxCols` are typically just slices of `AB::Var`.

This trait only owns the constraints, but it is expected that the `TraceSubRowGenerator` trait
or some analogous functionality is also implemented so that the trace generation of the `AuxCols`
of each row can be done purely in terms of the `Io` part.

```rust
pub trait SubAir<AB: AirBuilder> {
    /// Type to define the context, typically in terms of `AB::Expr` that are needed
    /// to define the SubAir's constraints.
    type AirContext<'a>
    where
        Self: 'a,
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, ctx: Self::AirContext<'a>)
    where
        AB::Var: 'a,
        AB::Expr: 'a;
}
```

Helper for generation of the trace on a subset of the columns in a single row
of the trace matrix.

```rust
pub trait TraceSubRowGenerator<F> {
    /// The minimal amount of information needed to generate the sub-row of the trace matrix.
    /// This type has a lifetime so other context, such as references to other chips, can be provided.
    type TraceContext<'a>
    where
        Self: 'a;
    /// The type for the columns to mutate. Often this can be `&'a mut Cols<F>` if `Cols` is on the stack.
    /// For structs that use the heap, this should be a struct that contains mutable slices.
    type ColsMut<'a>
    where
        Self: 'a;

    fn generate_subrow<'a>(&'a self, ctx: Self::TraceContext<'a>, sub_row: Self::ColsMut<'a>);
}
```
