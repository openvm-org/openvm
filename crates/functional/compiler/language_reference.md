# ZK Functional Language Reference

This document describes the syntax and semantics of the language.

Table of contents:
- [Program Structure](#program-structure)
  - [Functions](#functions)
  - [Algebraic Types](#algebraic-types)
- [Variables](#variables)
- [Dematerialization](#dematerialization)
- [Program Features](#program-features)
- [Guided Examples](#guided-examples)

#### Note on syntax

Most languages use `{...}` to define blocks; however, in this language,
we reserve the use of `{...}` to always refer to dematerialization.
Therefore, we use `(...)` to define blocks instead, and only use `{...}` for blocks if we want to dematerialize them.

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
```text
(fn|inline) name((in|out)[alloc|unalloc]<type1> arg1, ...) body
```
An example, with the body abbreviated for now, is as follows:
```text
fn fibonacci(in<F> n, out<F> a, out<F> b) (...)
```

#### Function Arguments

A function argument has four components:
- A behavior, which is either `in` or `out`.
- A declaration type, which is either `alloc` or `unalloc`, but can be (and usually is) omitted.
- A type
- A name

As seen above, the syntax is as follows:
```text
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
```text
function_name(arg1, ...);
```
The callee specifies each argument as either `in` or `out`; correspondingly, each argument either must be specified, or will be determined by the callee.
An `out` argument can also be specified elsewhere, as in the following example:
```text
def n = 12;
def b = 144;
fibonacci(n, def a, b);
```
Here, the `in` argument `n` is specified, the out argument `a` is determined by the callee, and the out argument `b` is specified elsewhere.
Code like this will constrain the callee's value for `b` to be equal to the caller's; i.e. it is an implicit form of this:

```text
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
```text
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
```text
enum Name (
    Variant1(ComponentType1, ...),
    Variant2(ComponentType2, ...),
    ...
)
```
Algebraic types include as a special case types with only one variant, which can be specified as above.
However, this is somewhat cumbersome, and so they can also be specified like `struct`s in Rust:

```text
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
```text
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

#### Field element

The field element type is `F`, corresponding to whatever field the proof system uses.

#### Reference

Given a type `T`, the type `&T` denote a reference to a value of type `T`, just as in Rust.
There is no functional difference between a `T` and a `&T`, so using `&T` only serves as a performance optimization.

### Expressions

### Statements

## Guided Examples