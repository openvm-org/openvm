# ZK Functional Language Reference

This document describes the syntax and semantics of the language.

Table of contents:
- [Program Structure](#program-structure)
  - [Functions](#functions)
  - [Algebraic Types](#algebraic-types)
- [Variables](#variables)
  - [Scopes](#scopes)
  - [Declaration, definition, and representation of variables](#declaration-definition-and-representation-of-variables)
- [Dematerialization](#dematerialization)
- [Program Features](#program-features)
  - [Types](#types)
  - [Expressions](#expressions)
  - [Statements](#statements)
- [Guided Examples](#guided-examples)
  - [Fibonacci](#fibonacci)
  - [Merkle verify](#merkle-verify)

#### Notes on syntax

Most languages use `{...}` to define blocks; however, in this language,
we reserve the use of `{...}` to always refer to dematerialization.
Therefore, we use `(...)` to define blocks instead, and only use `{...}` for blocks if we want to dematerialize them.

Also note that comments can be used just like in Rust, so you can use `//` for a single-line comment and `/* ... */` for a multiline comment.

## Program Structure

A programs consists of a series of functions and algebraic types; these are explained in the below subsections.

### Functions

The main building block of a program is a function.
Functions are like functions in most languages, with the added constraint that since
this is a purely functional language, functions cannot have side effects, meaning that they are fully determined by their inputs.

A function contains the following components:
- A name
- A list of arguments
- A body
- Optionally, a specification as an inline function
The syntax of a function is as follows:
```rust
(fn|inline) name((in|out)[alloc|unalloc]<type1> arg1, ...) body
```
An example, with the body abbreviated for now, is as follows:
```rust
fn fibonacci(in<F> n, out<F> a, out<F> b) (...)
```

#### Function Arguments

A function argument has four components:
- A behavior, which is either `in` or `out`.
- A declaration type, which is either `alloc` or `unalloc`, but can be (and usually is) omitted.
- A type
- A name

As seen above, the syntax is as follows:
```rust
behavior declaration_type<type> name
```
The name and type are as in most languages.

To understand the behavior, note that the list of components of a function did not include a return type.
This is because return values are regarded as also being arguments. Specifically, return values are now `out` arguments,
while `in` arguments correspond to usual arguments that are inputs to a function.
Alternatively, in order to simulate the behavior of languages where functions can mutate their arguments, `in` arguments
can be seen as the previous state of the arguments, and `out` arguments can be seen as the next state of the arguments.

Finally, the declaration type specifies whether we are explicitly representing the argument using its own cells (`alloc`),
or if we will later represent it in terms of other expressions (`unalloc`). This will be discussed in more detail in [Variables](#variables).
For now, we note that the declaration type is usually omitted, in which case the compiler assumes `alloc` for `in` arguments and `unalloc` for out arguments.

#### Execution Order

The order in which arguments are specified in a function carries meaning:
they specify that an `out` argument is determined once the preceding `in` arguments are specified. Therefore, functions that are like functions in most languages will have all `in` arguments listed before all `out` arguments,
so that all `in` arguments must be specified before the `out` arguments are determined. However, more complicated behavior can be achieved by specifying some `in` arguments after some `out` arguments.
This allows for a caller to perform computation based on the outputs of a callee, then provide additional input to the callee based on said computation.

In contrast, the order in which statements are specified in the function body does not carry any meaning.
Thus, variables can be used before they are defined or even declared.
The compiler, given the argument order specified by the function, will determine a valid execution order for the statements,
or determine that no such order exists, in which case the program is considered to be semantically incorrect.

More details about this ordering are as follows:
- Each statement can be evaluated once the variables it depends on have been defined in the statement's scope.
For a `match` statement, this is any variables upon which the match expression depends.
- Statements are evaluated atomically, meaning that the entire statement is evaluated before any other statement.
This stands in contrast to Haskell, where statements can partially evaluate in order to determine certain variables or even parts of certain variables.
  - An exception to this is function calls, which allow for partial evaluation of a prefix of the remaining arguments. This is only possible if some `in` arguments appear after some `out` arguments.

#### Function Body

The body of a function is a series of three types of components:
- Normal statements; the various types of statements are described in [Statements](#statements), except for variable declarations, which are more thoroughly described in [Variables](#variables).
- Function calls, which are described below in [Function Calls](#function-calls).
- Matches, which are described below in [Matches](#matches).

The syntax of a function body is as follows: we enclose the above components in either `(...)` or `{...}`; the latter will cause the entire contents of the body to be dematerialized,
meaning that none of said body will be constrained in the proof. For more detail, see [Dematerialization](#dematerialization).

#### Function Calls

The syntax of a function call is the same as in Rust, with the key difference that in Rust, function calls are expressions,
while here, they are an entire statement. Thus:
```rust
function_name(arg1, ...);
```
The callee specifies each argument as either `in` or `out`; correspondingly, each argument either must be specified, or will be determined by the callee.
An `out` argument can also be specified elsewhere, as in the following example:
```rust
def n = 12;
def b = 144;
fibonacci(n, def a, b);
```
Here, the `in` argument `n` is specified, the out argument `a` is determined by the callee, and the out argument `b` is specified elsewhere.
Code like this will constrain the callee's value for `b` to be equal to the caller's; i.e. it is an implicit form of this:

```rust
def n = 12;
fibonacci(n, def a, def b);
b = 144;
```

If the callee is specified as inline, then the callee's body will be inlined into the caller's body.
This is mostly a performance optimization and does not affect the semantics of the program,
except that inline functions cannot recursively call each other.

#### Matches

Like Rust, we can use `match` to achieve branching. In fact, `match` is the only way to branch in the language.
There are four differences of `match` in the language from Rust:
- Like function calls, `match` statements are statements, not expressions.
- Unlike Rust, where we follow the fisrt arm that matches, the language requires arms to be disjoint.
- Match arms need not be exhaustive. In this case, the program constrains the matched value to correspond to one of the arms.
- Currently, `match` statements are very simple: they take in an algebraic type, and each arm destructures it into one of its variants.

The syntax of a `match` statement is as follows:
```rust
match expr (
    constructor1([alloc|unalloc] component1, ...) => body1
    ...
)
```
The syntax is mostly identical to Rust. Components can also be preceded by either `alloc` or `unalloc`,
which specifies that they are either explicitly represented here or are not (for more detail, see [Variables](#variables)).
The default behavior is `alloc`.

Additionally, `match expr` can be surrounded by `{...}`. This will dematerialize the `match` check,
meaning that it is not constrained that `expr` is in fact the expected variant of its type;
however, the statements in the body of each arm will not be dematerialized by this.

A pattern that relies on this is as follows: we use `{match expr}`, then in each branch, constrain in some other way that that branch actually holds.
For example, we use `{match a == b}`, then in the `True` branch, constrain `a` and `b` to be equal, and in the `False` branch, constrain `a` and `b` to be unequal.

### Algebraic Types

The language allows for user-defined types in the form of algebraic types.
Algebraic types are essentially the same as Rust enums, and can be defined with the below syntax that is identical to Rust (other than `(...)` instead of `{...}`):
```rust
enum Name (
    Variant1(ComponentType1, ...),
    Variant2(ComponentType2, ...),
    ...
)
```
Algebraic types include as a special case types with only one variant, which can be specified as above.
However, this is somewhat cumbersome, and so they can also be specified like `struct`s in Rust:

```rust
struct Name (
    ComponentType1,
    ComponentType2,
    ...
)
```
Note that algebraic types are not allowed to have zero variants.

## Variables

The language, like most, relies on variables for storing and referring to data.
A variable has a name and a type, and belongs to a certain scope.

### Scopes

Scopes are defined by `match` statements.
Formally, a function's body can be interpreted as a tree, where the root is a scope, and scopes have children corresponding to statements, function calls, and `match`es; each `match` can then have children which are also scopes.

A variable defined in a scope can be referenced in that scope and any descendant scopes, but not in any scope above it.
Two variables that can be referenced in a common scope cannot have the same name.

### Declaration, definition, and representation of variables

In most languages, a variable must be declared and defined.
Declaration means specifying that a variable exists with a certain name and type.
Definition means specifying that a variable has a certain value.
Declaration and definition are usually done simultaneously, as in the Rust code `let x = 5;`,
but can be done separately, as in the Rust code `let x; x = 5;`.

The same is true for our language.
Note that as this language is purely functional, variables can only be defined once.

More importantly, the language additionally has the concept of representation of variables.
Representation means specifying how a variable is represented as an expression in the AIR.
Therefore, when dealing with variables, we not only declare them and define them, but also represent them.
Additionally, just as declaration and definition are usually done simultaneously but can be done separately,
the same is true for representation, meaning that there are 2^3 = 8 different ways to use a variable.

This gives rise to 7 keywords for declaring, defining, and representing variables, which we describe now.

The first two keywords correspond to variable declaration statements.
Variable declarations explicitly specify the type of a variable and don't do anything else.
There are two ways to declare a variable:
```rust
alloc<type> name;
unalloc<type> name;
```
Both declare a variable named `name` of type `type` in the current scope.
However, the first specifies that we explicitly allocate cells with which to represent the variable,
while the second does not, meaning that we will need to represent the variable in some other way later.

Additionally, function arguments and components in match arms also declare variables,
explaining why `alloc` and `unalloc` can be used in those places.

The remaining five keywords are used in expressions.
Each keyword can be folllowed by a variable name,
and will do some combination of declaring, defining, and representing the variable.
The keywords are:
- `def`: declares, defines, and represents the variable.
- `let`: declares and defines the variable.
- `fix`: defines and represents the variable.
- `set`: defines the variable.
- `rep`: represents the variable.

When representing a variable in this way (as opposed to `alloc`, which explicitly allocates cells for the variable),
what the representation is depends on the statement in which the variable is used.
For example, if the variable is being used in a function call, then the variable will be allocated some cells,
so that those cells can be used the corresponding interaction. However, if the variable is used in a simple equality,
then the representation will simply be copied from the other side of the equality, and no additional cells will be allocated.


The compiler requires that each variable is defined exactly once, which can potentially mean
being defined once in each branch of a `match`, and therefore being "defined once" by the match.

Similarly, the compiler requires that each variable is represented exactly once;
however, an exception is made for variables of types that do not take up any cells,
which are usually dematerialized types.

Thus, for instance, the following two pieces of code are equivalent:
- `def x = 5;`
- `unalloc<F> x; fix x = 5;`

In contrast, `alloc<F> x; set x = 5;` will be semantically identical but will allocate an unnecessary cell that always contains the value `5`, `let x = 5` will mean that `x` is still yet to be represented, while code such as `alloc<F> x; fix x = 5;` is semantically invalid as it represents `x` twice.

Finally note that the keywords `def, let, fix, set, rep` can precede not only variable names, but more complicated expressions.
In this case, they refer to any variables that appear in the expression.
For example, we can do `def Point(x, y) = Point(3, 4);`, which is equivalent to `Point(def x, def y) = Point(3, 4);`.
### Practical usage of variables

For programs which do not need to be super tightly optimized,
you don't want to have to worry about representation. In that case, you should follow the following guide:
- For most variables, use `def` to declare, define, and represent them.
- For most `out` arguments, use `fix` to define and represent them.
- For variables (including `out` arguments) whose values are branch-dependent, use `alloc` to declare and represent them and `set` to define them.
- Do not use `alloc` or `unalloc` for function arguments or match components except when an `out` argument's value is branch-dependent, in which case you should use `alloc`.

Following this guide will represent variables when they are defined except when their values are branch-dependent,
in which case we explicitly allocate cells in order to prevent representations with high degree.

## Dematerialization

A fundamental feature of the language is dematerialization.
Dematerialization allows for parts of the language, i.e. types, variables, statements, etc., that are unchanged
from the perspective of execution, but which are not present in the AIR.
Thus, we can take "hints" as input and rely on their values without materializing them as cells,
or perform dematerialized computation in a more expensive way, then verify the result of the computation more cheaply.

To dematerialize something, we surround it with `{...}`. The following language components can be dematerialized:
- Types: If `T` is a type, then `{T}` is also a type that refers to a dematerialized `T`. Thus, `{T}` takes up zero cells.
- Expressions: If `expr` is an expression of type `T`, then `{expr}` is an expression of type `{T}`.
- Statements: Any statement, match, or function call can be surrounded by `{...}` in order to be dematerialized, meaning that the corresponding constraints will not be enforced by the AIR.
- Matches: As mentioned before, the `match expr` component of a `match` can be surrounded by `{...}` in order to cause just the `match`'s own check to be dematerialized, without dematerializing the bodies of the arms.
- Bodies: For convenience, an entire body (i.e. a function body or a body of a match arm) can be dematerialized by replacing `(...)` with `{...}`.

The important thing to note is that anywhere inside `{...}`, further `{...}` has no effect.
This is to say that inside `{...}`, the two types `T` and `{T}` are the same type. Therefore:
- In a dematerialized expression `{expr}`, wherever an expression of type `T` is expected, an expression of type `{T}` can be used (and vice versa).
- In a dematerialized statement `{statement};`, wherever an expression of type `T` is expected, an expression of type `{T}` can be used (and vice versa).
- The two types `{T}` and `{{T}}` are identical.

In mathematical terms, `{...}` is idempotent.

## Program Features

Having covered the fundamental structure of the language,
we now describe the different types, expressions, and statements that are available.

### Types

#### Field elements

The field element type is `F`, corresponding to whatever field the proof system uses.

#### References

Given a type `T`, the type `&T` denote a reference to a value of type `T`, just as in Rust. A `&T` can be created from a `T` and vice versa using the reference and dereference statements.
There is no functional difference between a `T` and a `&T`, so using `&T` only serves as a performance optimization.

### Arrays and associated types

Given a type `T`, the type `@[T]` denotes an in-memory array of `T`s. An `@[T]` can be accessed at an index to yield a `T`.

To construct an array, we have an additional type known as an appendable prefix. Given a type `T` and an *expression* `len` of type `F`, the type `#..len[T]` denotes an appendable prefix of `T`s of length `len`.
The type signature itself constrains the length of the prefix to be exactly some value, allowing further appends to occur at the correct location.

Appendable prefixes are a somewhat unique type, as they have a notion of "consumption" of the prefix.
Appending to a prefix consumes the prefix (and returns a new prefix); appendable prefixes can also be converted to arrays
(an `#..len[T]` can be converted to a `@[T]`), which also consumes the prefix.
Furthermore, almost any other generic usage of an appendable prefix will consume it.
As implied by the name, a prefix can be consumed only once.
This constrains a prefix to not be appended to more than once at the same index, and also ensures that an array refers to the entirety of a prefix.

### User-defined types

User-defined types are referred to by name; see [Algebraic Types](#algebraic-types) for details on their definition.

### Boolean

The type `Bool` is an algebraic type that is required to be defined as
```rust
enum Bool ( True, False )
```
It is special as it is the return value of the built-in equality operator `==`.
It is also transpiled to the Rust `bool` type rather than to a new enum.

### Fixed-size arrays

Given a type `T` and a literal number `len`, the type `[T; len]` denotes a fixed-size array of `T`s of length `len`.
This is a convenience type; it can be simulated by an algebraic type with one variant and `len` components, each equal to `T`.

### Unmaterialized type

GIven a type `T`, the type `{T}` is a type that works identically to `T` during execution,
but is not materialized in the AIR. See [Dematerialization](#dematerialization) for more information.

### Expressions

The language has a notion similar to that of lvalues and rvalues in C++.
Specifically, we will say that a statement depends on some of the expressions it contains being defined,
and other expressions that it contains being assign-ready.
We can then define expressions into two categories:
- Assignable expressions are those that are assign-ready once all of their subexpressions are assign-ready.
- Unassignable expressions are those that are assign-ready once they are defined.

For example, in the statement `def x = 5;`, the statement depends on `5` being defined and on `x` being assign-ready.
`x`, being a variable prefixed by `def`, is assignable, and as it does not contain any other expressions, it is always assign-ready.
In contrast, if we were to write `x + y = 5;`, then this statement would require `x` and `y` to be already defined before it could execute.

An important aspect of the language is that expressions that are not assignable can still be assigned to:
given an unassignable expression `x` and an expression `y`, a statement which assigns `y` to `x` will actually constrain `x` and `y` to be equal.
This then allows us to write piecemeal statements such as `Point(1, def x) = Point(y, 4);` which will constrain `y` to be equal to `1` and assign `x` the value `4`.

The following expressions are assignable:
- Variable expressions when prefixed (directly or indirectly) by `def`, `let`, `fix`, or `set`.
- Algebraic expressions.
- Fixed-length arrays.

#### Constant

Number literals are expressions of type `F`.

#### Variable

A variable name is an expression referring to that variable.

#### Variable keywords

The keywords `def, let, fix, set, rep` can precede an expression to specify how any contained variables are being used; see [Variables](#variables) for more information.

#### Algebraic

Given an algebraic type `T` one of whose variants is named `A` and has component types `T_1, ..., T_n`,  and given some expressions `e_1, ..., e_n` of types `T_1, ..., T_n`, the expression `A(e_1, ..., e_n)` is an expression of type `T`.

#### Arithmetic

Given two expressions `e_1` and `e_2` of type `F`, the expressions `e_1 + e_2`, `e_1 - e_2`, and `e_1 * e_2` are expressions of type `F`.

Additionally, given an expression `e` of type `F`, the expression `-e` is an expression of type `F`.

#### Readable view of prefix

Given an expression `e` of type `#..len[T]`, the expression `@..[e]` is an expression of type `@..len[T]`.

#### Prefix into array

Given an expression `e` of type `@..len[T]`, the expression `@[e]` is an expression of type `@[T]` that consumes `e`.

#### Fixed length array

Given a type `T` and expressions `e_1, ..., e_n` of type `T`, the expression `[e_1, ..., e_n]` is an expression of type `[T; n]`.

#### Empty fixed length array

Given a type `T`, the expression `[<T>]` is an expression of type `[T; 0]`.

#### Fixed length array concatenation

Given a type `T` and expressions `e_1` and `e_2` of types `[T; n]` and `[T; m]` respectively, the expression `e_1 ++ e_2` is an expression of type `[T; n + m]`.

#### Fixed length array access

Given an expression `e` of type `[T; n]` and a literal `i` among `0, ..., n - 1`, the expression `e[i]` is an expression of type `T`.

#### Fixed length array slice

Given an expression `e` of type `[T; n]` and literals `i` and `j` with `i <= j` and `j <= n`, the expression `e[i..j]` is an expression of type `[T; j - i]`.

#### Fixed length array repeated

Given an expression `e` of type `T` and a literal `n`, the expression `[e; n]` is an expression of type `[T; n]`.
`T` cannot contain an appendable prefix.

#### Dematerialized expression

Given an expression `e` of type `T`, the expression `{e}` is an expression of type `{T}`.

#### Expressions that can only be used in a dematerialized context

The following expressions cannot be translated into polynomial expressions, and so are "execution-time only", meaning that in order to use them, you must dematerialize them.

#### Division

Given expressions `e_1` and `e_2` of type `F`, the expression `e_1 / e_2` is an expression of type `F`.

#### Equality

Given expressions `e_1` and `e_2` of type `T`, the expression `e_1 == e_2` is an expression of type `Bool`.
`T` cannot contain any memory type, meaning that it cannot contain a reference, array, or appendable or readable prefix.

### Statements

#### Variable declaration

The statements `alloc<T> name;` and `unalloc<T> name;` declare a variable `name` of type `T` in the current scope.
For more details, see [Variables](#variables).

#### Equality

Given expressions `e_1` and `e_2` of type `T`, the statement

```rust
e_1 = e_2;
```

assigns `e_1` to be equal to `e_2`.
It therefore relies on `e_2` being defined and `e_1` being assign-ready.

#### Reference

Given an expression `data` of type `T` and an expression `ref` of type `&T`, the statement

```rust
ref -> data;
```

assigns `ref` to be a reference to `data`.
It therefore relies on `data` being defined and `ref` being assign-ready.

#### Dereference

Given an expression `ref` of type `&T` and an expression `data` of type `T`, the statement

```rust
data <- ref;
```

assigns `data` to be the value pointed to by `ref`.
It therefore relies on `ref` being defined and `data` being assign-ready.

#### Empty prefix

Given an expression `prefix` of type `#..len[T]`, the statement

```rust
prefix -> |T;
```

assigns `prefix` to be an empty appendable prefix.
It therefore relies on `prefix` being assign-ready. It also constrains `len` to be equal to `0`.

#### Prefix append

Given an expression `old_prefix` of type `#..len1[T]`, an expression `new_prefix` of type `#..len2[T]`, and an expression `elem` of type `T`, the statement

```rust
new_prefix -> old_prefix | elem;
```

assigns `new_prefix` to be the appendable prefix formed by appending `elem` to `old_prefix`.
It therefore relies on `old_prefix` and `elem` being defined and `new_prefix` being assign-ready. It also constrains `len2` to be equal to `len1 + 1`.

#### Array access

Given an expression `array` of type `@[T]`, an expression `index` of type `F`, and an expression `elem` of type `T`, the statement

```rust
elem <- array !! index
```

assigns `elem` to be the element of `array` at index `index`.
It therefore relies on `array` and `index` being defined and `elem` being assign-ready.

## Guided Examples

### Fibonacci

```rust
enum Bool ( True, False )

fn fibonacci(in<F> n, out<F> a, out<F> b) (
    {match n == 0} (
        True => (
            n = 0;
            fix a = 0;
            fix b = 1;
        )
        False => (
            fibonacci(n - 1, def x, def y);
            fix a = y;
            fix b = x + y;
        )
    )
)
```

The above program defines a function named `fibonacci` with three arguments, all of which are field elements:
- An input argument `n`, corresponding to the index of the Fibonacci number to compute.
- An output argument `a`, corresponding to `F_n`.
- An output argument `b`, corresponding to `F_{n + 1}`.

As `alloc`/`unalloc` were not specified explicitly, `n` is `alloc` and `a, b` are `unalloc` by default,
meaning that `n` is already represented and `a, b` are not.

The body consists of a match statement, which performs a *dematerialized* check (because of the `{...}` in `{match n == 0}`) on whether `n` is equal to `0` by matching on the expression `n == 0`, which is a `Bool`.
This match therefore does not introduce any AIR constraints.

In the `True` branch, we importantly constrain `n` to be equal to `0`, verifying that the branch holds since the match itself was unconstrained.
We then `fix` the variables `a` and `b` to be their correct values, both defining them to both values and representing them as those constants.

In the `False` branch, we recursively call `fibonacci` with `n - 1` as the input arguments;
for the output arguments, we `def` the variables `x` and `y`, declaring them to be field elements, defining them as the outputs of the recursive call,
and representing them; since this is a function call, the representation simply introduces a new cell for each.
We then `fix` the variables `a` and `b` in terms of `y` and `x + y`, both defining them and representing them in terms of those values.

### Merkle verify

The below program is commented inline instead of having a separate explanation.

```rust
enum Bool ( True, False )

/*
    `leaf` is a reference to a `[F; 8]`,
    `bits` is an array of `Bool`s,
    and `siblings` is an unmaterialized array of `[F; 8]`s.
    
    The only output is `commit`, which is an `[F; 8]`;
    we have explicitly allocated `commit`.
*/
fn merkle_verify(in<&[F; 8]> leaf, in<F> length, in<@[Bool]> bits, in<{@[[F; 8]]}> siblings, out alloc<[F; 8]> commit) (
    // We explicitly allocate `left` and `right` so that they are degree 1
    alloc<[F; 8]> left;
    alloc<[F; 8]> right;
    /* This call to hash is inlined
       and declares/defines/represents a new variable `hash_result`.
       The call depends `left` and `right`,
       which are defined later in each branch;
       it can only be executed once both are defined in all branches.
        */
    hash(left, right, def hash_result);
    // The match is unmaterialized like in fibonacci
    {match length == 0} (
        True => (
            length = 0;
            /* In this branch, we define `left`, `right`, and `commit`.
               We use set as they already have a representation. */
            set left = [0; 8];
            set right = [0; 8];
            // This statement dereferences `leaf` into `commit`
            set commit <- leaf;
        )
        False => (
            /* Recursive call to `merkle_verify`
               which declares/defines but does not represent `child` by using `let` */
            merkle_verify(leaf, length - 1, bits, siblings, let child);
            def i = length - 1;
            /* Array access, which again does not represent `bit` */
            let bit <- bits !! i;
            /* Dematerialized array access:
               at execution time we access the siblings array to determine what sibling to use,
               but in the AIR it is treated as a hint, and no details of the array or access remain. */
            {let sibling <- siblings !! i};
            /* Another match, this time materialized;
               we also use the match to represent `bit`, saving a column */
            match rep bit (
                True => (
                    /* Dematerialized set: because it is dematerialized, the variable `sibling`,
                       which is of type `{[F; 8]}`,
                       can be assigned to the variable `left`, which is of type `[F; 8]`,
                       because those types are identical inside {...}. */
                     */
                     
                    {set left = sibling};
                    /* We specify that `right` should be the child,
                       then knowing that, use `right`, which is already represented, to represent `child`,
                       keeping us from having to allocate cells for `child` and saving 8 cells.
                        */
                    set right = child;
                    rep child = right;
                )
                False => (
                    // Analogous to the True branch
                    set left = child;
                    rep child = left;
                    {set right = sibling};
                )
            )
            set commit = hash_result;
        )
    )
)

// Random dumb hash function
inline hash(in<[F; 8]> left, in<[F; 8]> right, out<[F; 8]> result) (
    fix result = [
        left[0] + right[0],
        left[1] * right[1],
        left[2] - right[2],
        left[3],
        right[4],
        115,
        left[6] * left[7],
        right[6] * right[7],
    ];
)
```