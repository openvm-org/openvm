# VM Extension: Elliptic Curve Cryptography

Author: [manh9203](https://github.com/manh9203)

## 1. Introduction

### 1.1 Scope

[ECC extension](https://github.com/openvm-org/openvm/tree/main/extensions/ecc)

### 1.2 Commit

<https://github.com/openvm-org/openvm/commit/4285a4f974db90ce11bd21c0642d059cbb8975d0>

### 1.3 Describe the main focus and any additional context

- Check the ECC extension conforms to the OpenVM ISA framework
- Pseudoproofs that each chip’s constraints exactly match the ISA
- Check all chips conform to circuit architecture
- Check the transpiler matches the RISC-V specs

## 2. Findings

Findings include anything that could warrant change or unexpected behavior that should be brought to light. They range from severe to informational.

### 2.1 P256 Generator Endianness

**Severity:** High

**Context:** <https://github.com/openvm-org/openvm/blob/4285a4f974db90ce11bd21c0642d059cbb8975d0/extensions/ecc/guest/src/p256.rs#L61>

**Description:**
The `GENERATOR` constant is big endian, while `from_const_bytes` accepts little endian.

Duplicate of https://cantina.xyz/code/c486d600-bed0-4fc6-aed1-de759fd29fa2/findings/102

**Proof of concept:** Omitted

**Recommendation:**
Change the constant to the correct one.

**Resolution:** <https://github.com/openvm-org/openvm/commit/76a8c98f8d3ccfce91d300eaf21717cc518b2766>

### 2.2 Non-zkvm Weierstrass double operation missing `a`

Author: [manh9203](https://github.com/manh9203)

**Severity:** Medium
**Context:** <https://github.com/openvm-org/openvm/blob/4285a4f974db90ce11bd21c0642d059cbb8975d0/extensions/ecc/sw-macros/src/lib.rs#L168>

**Description:**
In the `#[cfg(not(target_os = "zkvm"))]` implementation of `double_impl` in the Weierstrass curve macro, the `a` coefficient of the curve is missing.
This only affects the non-zkVM implementation used for host testing. It does not affect the VM execution or proving.

**Proof of concept:** Omitted

**Recommendation:**
Update the implementation to include the `a` term.

**Resolution:** <https://github.com/openvm-org/openvm/commit/76a8c98f8d3ccfce91d300eaf21717cc518b2766>

### 2.3 Malicious host can cause decompress method to panic

Author: [Avaneesh](https://github.com/Avaneesh-axiom)

**Severity:** High

**Context:**
This issue involves the [decompress method](https://github.com/openvm-org/openvm/blob/4285a4f974db90ce11bd21c0642d059cbb8975d0/extensions/ecc/guest/src/weierstrass.rs#L71C1-L84C1).

**Description:**
If the host provides invalid hints, the decompress method can panic even if the point being decompressed is valid.
This can result in a malicious host proving that a guest program panics, even when it doesn't panic under an honest host.

A related problem is that an honest host cannot detect if an invalid point is being decompressed.
In this case, the decompress method will panic, which results in the same behavior as if the host had provided an invalid hint.

**Proof of concept:** See [`extensions/ecc/tests/programs/examples/decompress_invalid_hint.rs`](https://github.com/openvm-org/openvm/commit/aeb308c69ab449c7930c684832899a8dfdec6fa9#diff-3dbf606a27db24d8358b8376d23e41ffd5690dfe9eff97247ac1d4a0507c6791)

**Recommendation:**
Under an honest host, decompress should either successfully produce a decompressed point or prove that no point with the given x-coordinate exists.
If an invalid hint is detected, the program should enter an infinite loop to prevent malicious hosts from being able to prove that the program panics.

**Resolution:** <https://github.com/openvm-org/openvm/commit/aeb308c69ab449c7930c684832899a8dfdec6fa9>

Under an honest host, `decompress` either successfully produces a
decompressed point or proves that no point with the given x-coordinate
exists.

If an invalid hint is detected, the program enters an infinite loop.

Here is the implementation we decided on:

- For each curve, hint a non quadratic residue element in the setup
function and save it as a global variable in the same scope as the curve
definition (i.e. the invocation of `sw_declare!`)
- Decompression has two possible outcomes: either the point is
successfully decompressed or it is proven that it cannot be decompressed
(i.e. no point on the curve exists with the given x-coordinate). The
decompression hint indicates which outcome it is.
- The latter case is handled as follows. If `rhs := x^3 + ax + b`, hint
for an element sqrt of the coordinate field satisfying `sqrt^2 = rhs * nonqr` where nonqr is the non quadratic residue for that curve that was
initialized in setup.
- The guest code verifies that every hint is valid (i.e. the `nonqr` is
indeed a non-qr, any hinted field element is less than the modulus,
etc). If an invalid hint is encountered, it enters an infinite loop.
This prevents a malicious host from proving a valid guest program panics
by supplying invalid hints.

### 2.4 Malicious host can cause ECDSA verification function to fail

Author: [lispc](https://github.com/lispc)

**Severity:** High

**Context:**
This issue involves the [verify_prehashed](https://github.com/openvm-org/openvm/blob/5e5558e8c4998797eb9ec3918c662c9ea818a81e/extensions/ecc/guest/src/ecdsa.rs#L213C50-L214C1) function.

**Description:**
This issue is related to issue 2.3 in nature. The current code asserts that the input validation in ECDSA signature verification is done via:

```rust
        let (r_be, s_be) = sig.split_at(<C as IntrinsicCurve>::Scalar::NUM_LIMBS);
        // Note: Scalar internally stores using little endian
        let r = Scalar::<C>::from_be_bytes(r_be);
        let s = Scalar::<C>::from_be_bytes(s_be);
        // The PartialEq implementation of Scalar: IntMod will constrain `r, s`
        // are in the canonical unique form (i.e., less than the modulus).
        assert_ne!(r, Scalar::<C>::ZERO);
        assert_ne!(s, Scalar::<C>::ZERO);
```

where `r, s` are asserted to be less than the modulus of `Scalar` using `PartialEq`, which uses the custom `iseqmod` intrinsic RISC-V instruction.
This instruction constrains that `r, s` are less than the modulus and the VM circuit will not successfully prove otherwise. This means that invalid
inputs will cause the proving to fail, whereas the expected behavior is that invalid inputs should return an `Error` in the guest program.

**Proof of concept:** N/A

**Recommendation:**
The program should check `r < Scalar::MODULUS` using `&[u8]` comparison, and return `Error` early if the check fails. Same with `s`.

**Resolution:** <https://github.com/openvm-org/openvm/pull/1458>
https://github.com/openvm-org/openvm/commit/d1321b1c52f45eadaba53185a994d3fc0a497072

- (style) rename previous `IntMod::assert_unique` to `IntMod::assert_reduced` and move the implementation using `iseqmod` into macro since it's specific to the use of special intrinsic.
- Add new `is_reduced() -> bool` that checks if an integer representation is less than the modulus. Does simple byte-wise comparison check.
- Update ECDSA verify and recover functions so it returns Error when `r` or `s` are not in `[1,n]`.

### 2.5 `find_non_qr` does not work for `p = 1 mod 4`

**Severity:** High

**Context:** <https://github.com/openvm-org/openvm/blob/b92feee7496903f6de42aef66b0c0ac146ed1438/extensions/ecc/circuit/src/weierstrass_extension.rs#L496>

**Description:**
The typo causes the function to fail to find a quadratic non-residue when `p = 1 (mod 4)`.
The tests didn't catch it because the moduli in the tests are either the special cases (`3 mod 4` or `5 mod 8`).

**Proof of concept:** Omitted

**Recommendation:**
Fix the typo.

**Resolution:** <https://github.com/openvm-org/openvm/pull/1469>

### 2.6 `prime_limbs` is tight bound in `FieldExpr::eval`

**Severity:** Medium

**Context:** <https://github.com/openvm-org/openvm/blob/b92feee7496903f6de42aef66b0c0ac146ed1438/crates/circuits/mod-builder/src/builder.rs#L334>

**Description:**
The `prime_limbs` is used as a tight bound in `FieldExpr::eval` while the limbs from trace generation are either `32` or `48` in length.
This causes the circuit to fail to prove when `prime` is not 256 or 384 bits.

**Proof of concept:** Omitted

**Recommendation:**
Pad the limbs with zeros to the correct length.

**Resolution:** <https://github.com/openvm-org/openvm/pull/1469>
https://github.com/openvm-org/openvm/commit/a1c1087196ba5da44a80ae91addcb7da51482854

### 2.7 `HintNonQr` is missing in specs

**Severity:** Medium

**Context:**
Check the [RISC-V, ISA, and transpiler specs](https://github.com/openvm-org/openvm/tree/main/docs/specs)

**Description:**
We recently added a new phantom instruction `HintNonQr` for the ECC extension, but we didn't update the RISC-V, ISA, and transpiler specs accordingly.

**Proof of concept:** Omitted

**Recommendation:**
Update the specs.

**Resolution:** <https://github.com/openvm-org/openvm/pull/1456>
https://github.com/openvm-org/openvm/commit/7f87e1ddfb420a2c73c141c8ce35990f4c93086a

## 3. Discussion
