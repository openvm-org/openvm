# ColsRef macro

The `ColsRef` procedural macro is used in constraint generation to create column structs that have dynamic sizes.

Note: this macro was originally created for use in the SHA-2 VM extension, where we reuse the same constraint generation code for three different circuits (SHA-256, SHA-512, and SHA-384).
See the [SHA-2 VM extension](../../../../../../extensions/sha2/circuit/src/sha2_chip/air.rs) for an example of how to use the `ColsRef` macro to reuse constraint generation code over multiple circuits.

## Overview

As an illustrative example, consider the following columns struct:
```rust
struct ExampleCols<T, const N: usize> {
    arr: [T; N],
    sum: T,
}
```
Let's say we want to constrain `sum` to be the sum of the elements of `arr`, and `N` can be either 5 or 10.
We can define a trait that stores the config parameters.
```rust
pub trait ExampleConfig {
    const N: usize;
}
```
and then implement it for the two different configs.
```rust
pub struct ExampleConfigImplA;
impl ExampleConfig for ExampleConfigImplA {
    const N: usize = 5;
}
pub struct ExampleConfigImplB;
impl ExampleConfig for ExampleConfigImplB {
    const N: usize = 10;
}
```
Then we can use the `ColsRef` macro like this
```rust
#[derive(ColsRef)]
#[config(ExampleConfig)]
struct ExampleCols<T, const N: usize> {
    arr: [T; N],
    sum: T,
}
```
which will generate a columns struct that uses references to the fields.
```rust
struct ExampleColsRef<'a, T, const N: usize> {
    arr: ndarray::ArrayView1<'a, T>, // an n-dimensional view into the input slice (ArrayView2 for 2D arrays, etc.)
    sum: &'a T,
}
```
The `ColsRef` macro will also generate a `from` method that takes a slice of the correct length and returns an instance of the columns struct.
The `from` method is parameterized by a struct that implements the `ExampleConfig` trait, and it uses the associated constants to determine how to split the input slice into the fields of the columns struct.

So, the constraint generation code can be written as
```rust
impl<AB: InteractionBuilder, C: ExampleConfig> Air<AB> for ExampleAir<C> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, _) = (main.row_slice(0), main.row_slice(1));
        let local_cols = ExampleColsRef::<AB::Var>::from::<C>(&local[..C::N + 1]);
        let sum = local_cols.arr.iter().sum();
        builder.assert_eq(local_cols.sum, sum);
    }
}
```
Notes:
- the `arr` and `sum` fields of `ExampleColsRef` are references to the elements of the `local` slice.
- the name, `N`, of the const generic parameter must match the name of the associated constant `N` in the `ExampleConfig` trait.

The `ColsRef` macro also generates a `ExampleColsRefMut` struct that stores mutable references to the fields, for use in trace generation.

The `ColsRef` macro supports more than just variable-length array fields.
The field types can also be:
- any type that derives `AlignedBorrow` via `#[derive(AlignedBorrow)]`
- any type that derives `ColsRef` via `#[derive(ColsRef)]`
- (possibly nested) arrays of `T` or (possibly nested) arrays of a type that derives `AlignedBorrow`

Note that we currently do not support arrays of types that derive `ColsRef`.

## Specification

Annotating a struct named `ExampleCols` with `#[derive(ColsRef)]` and `#[config(ExampleConfig)]` produces two structs, `ExampleColsRef` and `ExampleColsRefMut`.
- we assume `ExampleCols` has exactly one generic type parameter, typically named `T`, and any number of const generic parameters. Each const generic parameter must have a name that matches an associated constant in the `ExampleConfig` trait

The fields of `ExampleColsRef` have the same names as the fields of `ExampleCols`, but their types are transformed as follows:
- type `T` becomes `&T`
- type `[T; LEN]` becomes `&ArrayView1<T>` (see [ndarray](https://docs.rs/ndarray/latest/ndarray/index.html)) where `LEN` is an associated constant in `ExampleConfig`
    - the `ExampleColsRef::from` method will correctly infer the length of the array from the config
- fields with names that end in `Cols` are assumed to be a columns struct that derives `ColsRef` and are transformed into the appropriate `ColsRef` type recursively
    - one restriction is that any nested `ColsRef` type must have the same config as the outer `ColsRef` type
- fields that are annotated with `#[aligned_borrow]` are assumed to derive `AlignedBorrow` and are borrowed from the input slice. The new type is a reference to the `AlignedBorrow` type
    - if a field whose name ends in `Cols` is annotated with `#[aligned_borrow]`, then the aligned borrow takes precedence, and the field is not transformed into an `ArrayView`
- nested arrays of `U` become `&ArrayViewX<U>` where `X` is the number of dimensions in the nested array type
    - `U` can be either the generic type `T` or a type that derives `AlignedBorrow`. In the latter case, the field must be annotated with `#[aligned_borrow]`
    - the `ArrayViewX` type provides a `X`-dimensional view into the row slice 

The fields of `ExampleColsRefMut` are almost the same as the fields of `ExampleColsRef`, but they are mutable references.
- the `ArrayViewMutX` type is used instead of `ArrayViewX` for the array fields.
- fields that derive `ColsRef` are transformed into the appropriate `ColsRefMut` type recursively.

Each of the `ExampleColsRef` and `ExampleColsRefMut` types has the following methods implemented:
```rust
// Takes a slice of the correct length and returns an instance of the columns struct.
pub const fn from<C: ExampleConfig>(slice: &[T]) -> Self;
// Returns the number of cells in the struct
pub const fn width<C: ExampleConfig>() -> usize;
```
Note that the `width` method on both structs returns the same value.

Additionally, the `ExampleColsRef` struct has a `from_mut` method that takes a `ExampleColsRefMut` and returns a `ExampleColsRef`.
This may be useful in trace generation to pass a `ExampleColsRefMut` to a function that expects a `ExampleColsRef`.

See the [tests](../../tests/test_cols_ref.rs) for concrete examples of how the `ColsRef` macro handles each of the supported field types.