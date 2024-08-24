# BigInt modular arithmetic by checking carry to zero

Google doc [spec](https://docs.google.com/document/d/1YYZ2mH__FR_CZGtteeNagTy1TpnYjc-O6_Bj3w4NOUc/edit#heading=h.8xamf5s02nq3)

## Details

*Added `OverflowInt<T>`*
It's big integer represented as limbs, and also tracks the value limit of each limb.
- it supports arithmetic like +-*, and updates the value limit accordingly.
- the generic type can be real values like `isize` or expression like `AB::Expr`.
- can generate the carries array for the limbs automatically.

Use case:

1. Trace generation: The parent AIR (e.g. multiplication AIR) can compose the expression (e.g. AB - PQ - R) as `OverflowInt<isize>` and generates the carries array (see the tests for an example).
1. AIR eval: The parent AIR `eval` function should compose the expression as `OverflowInt<AB::Expr>` and pass it to the check carry subair `constrain_carry_to_zero` along with the carries column.
