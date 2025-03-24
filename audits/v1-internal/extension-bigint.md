# VM Extension: BigInt

Author: [manh9203](https://github.com/manh9203)

## 1. Introduction

Scope: [BigInt extension](https://github.com/openvm-org/openvm/tree/main/extensions/bigint)

Commit: <https://github.com/openvm-org/openvm/commit/830053d599606fd5c7dc8f8346710f9d6854beae>

Describe the main focus and any additional context:

- OpenVM ISA - check they conform to framework
- Pseudoproofs that each chip’s constraints exactly match the ISA
- Check all chips conform to circuit architecture
- Transpiler

## 2. Findings

Classify by **severity** according to [cantina critiera](https://docs.cantina.xyz/cantina-docs/cantina-competitions/judging-process/finding-severity-criteria) in terms of likelihood and impact.

Findings include anything that could warrant change or unexpected behavior that should be brought to light. They range from severe to informational.

No findings found.

## 3. Discussion

Discussion is for general discussion or additional writing about what has been studied, considered, or understood that did not result in a concrete finding. Discussions are useful both to see what were important areas that are security critical and why they were satisfied. In the good case, the review should have few findings but lots of discussion.

### 3.1 Handling negative shift amounts for `I256`

When using the `I256` type, if the shift amount for `shl` or `shr` is negative, it is internally converted to `(shift_amount % 256) as usize`. For example, `Int256::from_i8(1) << Int256::from_i8(-1)` ends up being `Int256::from_i8(1) << 255`.

This differs from Rust’s native shift operations, which will return a compilation error if the shift amount is negative.

While there is no universal convention for handling negative shift amounts in BigInt-like types, it may be more intuitive to follow Rust’s native behavior for consistency. For instance, the standard BigInt library doesn't allow the shift amount to be BigInt, and will panic if the shift amount is negative. Otherwise, it would be helpful to document how negative shift amounts are handled in `I256`.
