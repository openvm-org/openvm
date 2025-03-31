# Algebra (Modular Arithmetic)

The OpenVM Algebra extension provides tools to create and manipulate modular arithmetic structures and their complex extensions. For example, if \\(p\\) is prime, OpenVM Algebra can handle modular arithmetic in \\(\mathbb{F}\_p\\)​ and its quadratic extension fields \\(\mathbb{F}\_p[x]/(x^2 + 1)\\).

The functional part is provided by the `openvm-algebra-guest` crate, which is a guest library that can be used in any OpenVM program. The macros for creating corresponding structs are in the `openvm-algebra-moduli-macros` and `openvm-algebra-complex-macros` crates.

## Available traits and methods

- `IntMod` trait:
  Defines the type `Repr` and constants `MODULUS`, `NUM_LIMBS`, `ZERO`, and `ONE`. It also provides basic methods for constructing a modular arithmetic object and performing arithmetic operations.

  - `Repr` typically is `[u8; NUM_LIMBS]`, representing the number's underlying storage.
  - `MODULUS` is the compile-time known modulus.
  - `ZERO` and `ONE` represent the additive and multiplicative identities, respectively.
  - Constructors include `from_repr`, `from_le_bytes`, `from_be_bytes`, `from_u8`, `from_u32`, and `from_u64`.

- `Field` trait:
  Provides constants `ZERO` and `ONE` and methods for basic arithmetic operations within a field.

## Modular arithmetic

To [leverage](./overview.md) compile-time known moduli for performance, you declare, initialize, and then set up the arithmetic structures:

1. **Declare**: Use the `moduli_declare!` macro to define a modular arithmetic struct. This can be done multiple times in various crates or modules:

```rust
moduli_declare! {
    Bls12_381Fp { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" },
    Bn254Fp { modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583" },
}
```

This creates `Bls12_381Fp` and `Bn254Fp` structs, each implementing the `IntMod` trait. The modulus parameter must be a string literal in decimal or hexadecimal format.

2. **Init**: Use the `moduli_init!` macro exactly once in the final binary:

```rust
moduli_init! {
    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
    "21888242871839275222246405745257275088696311157297823662689037894645226208583"
}
```

This step enumerates the declared moduli (e.g., `0` for the first one, `1` for the second one) and sets up internal linkage so the compiler can generate the appropriate RISC-V instructions associated with each modulus.

3. **Setup**: At runtime, before performing arithmetic, a setup instruction must be sent to ensure security and correctness. For the \\(i\\)-th modulus, you call `setup_<i>()` (e.g., `setup_0()` or `setup_1()`). Alternatively, `setup_all_moduli()` can be used to handle all declared moduli.

**Summary**:

- `moduli_declare!`: Declares modular arithmetic structures and can be done multiple times.
- `moduli_init!`: Called once in the final binary to assign and lock in the moduli.
- `setup_<i>()`/`setup_all_moduli()`: Ensures at runtime that the correct modulus is in use, providing a security check and finalizing the environment for safe arithmetic operations.

## Complex field extension

Complex extensions, such as \\(\mathbb{F}\_p[x]/(x^2 + 1)\\), are defined similarly using `complex_declare!` and `complex_init!`:

1. **Declare**:

```rust
complex_declare! {
    Bn254Fp2 { mod_type = Bn254Fp }
}
```

This creates a `Bn254Fp2` struct, representing a complex extension field. The `mod_type` must implement `IntMod`.

2. **Init**: Called once, after `moduli_init!`, to enumerate these extensions and generate corresponding instructions:

```rust
complex_init! {
    Bn254Fp2 { mod_idx = 0 },
}
```

Note that you need to use the same type name in `complex_declare!` and `complex_init!`. For example, the following code will **fail** to compile:

```rust
// moduli related macros...

complex_declare! {
    Bn254Fp2 { mod_type = Bn254Fp },
}

pub type Fp2 = Bn254Fp2;

complex_init! {
    Fp2 { mod_idx = 0 },
}
```

Here, `mod_idx` refers to the index of the underlying modulus as initialized by `moduli_init!`

3. **Setup**: Similar to moduli, call `setup_complex_<i>()` or `setup_all_complex_extensions()` at runtime to secure the environment.

### Config parameters

For the guest program to build successfully, all used moduli must be declared in the `.toml` config file in the following format:

```toml
[app_vm_config.modular]
supported_modulus = ["115792089237316195423570985008687907853269984665640564039457584007908834671663"]

[app_vm_config.fp2]
supported_modulus = ["115792089237316195423570985008687907853269984665640564039457584007908834671663"]
```

The `supported_modulus` parameter is a list of moduli that the guest program will use. They must be provided in decimal format in the `.toml` file.
The order of moduli in `[app_vm_config.modular]` must match the order in the `moduli_init!` macro.
Similarly, the order of moduli in `[app_vm_config.fp2]` must match the order in the `complex_init!` macro.

### Example program

Here is a toy example using both the modular arithmetic and complex field extension capabilities:

```rust,no_run,noplayground
{{ #include ../../../examples/algebra/src/main.rs }}
```

To have the correct imports for the above example, add the following to the `Cargo.toml` file:

```toml
[dependencies]
openvm = { git = "https://github.com/openvm-org/openvm.git" }
openvm-platform = { git = "https://github.com/openvm-org/openvm.git" }
openvm-algebra-guest = { git = "https://github.com/openvm-org/openvm.git" }
openvm-algebra-complex-macros = { git = "https://github.com/openvm-org/openvm.git" }
serde = { version = "1.0.216", default-features = false }
```

Here is the full `openvm.toml` to accompany the above example:

```toml
[app_vm_config.rv32i]
[app_vm_config.rv32m]
[app_vm_config.io]
[app_vm_config.modular]
supported_modulus = ["998244353","1000000007"]

[app_vm_config.fp2]
supported_modulus = ["998244353","1000000007"]
```
