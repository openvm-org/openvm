# OpenVM ECC

For elliptic curve cryptography, the `openvm-ecc` crate provides macros similar to those in [`openvm-algebra`](./algebra.md):

1. **Declare**: Use `sw_declare!` to define elliptic curves over the previously declared moduli. For example:

```rust
sw_declare! {
    Bls12_381G1Affine { mod_type = Bls12_381Fp, b = BLS12_381_B },
    Bn254G1Affine { mod_type = Bn254Fp, b = BN254_B },
}
```

Each declared curve must specify the `mod_type` (implementing `IntMod`) and a constant `b` for the Weierstrass curve equation $y^2 = x^3 + b$.

2. **Init**: Called once, it enumerates these curves and allows the compiler to produce optimized instructions:

```rust
sw_init! {
    Bls12_381Fp, Bn254Fp,
}
```

3. **Setup**: Similar to the moduli and complex extensions, runtime setup instructions ensure that the correct curve parameters are being used, guaranteeing secure operation.

**Summary**:

- `sw_declare!`: Declares elliptic curve structures.
- `sw_init!`: Initializes them once, linking them to the underlying moduli.
- `setup_sw_<i>()`/`setup_all_curves()`: Secures runtime correctness.
