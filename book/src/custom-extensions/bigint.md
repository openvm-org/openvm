# Big Integers

The OpenVM BigInt extension (aka `Int256`) provides two structs: `U256` and `I256`. These structs can be used to perform 256 bit arithmetic operations. The functional part is provided by the `openvm-bigint-guest` crate, which is a guest library that can be used in any OpenVM program.

## `U256`

The `U256` struct is a 256-bit unsigned integer type.

### Constants

The `U256` struct has the following constants:

- `MAX`: The maximum value of a `U256`.
- `MIN`: The minimum value of a `U256`.
- `ZERO`: The zero constant.

### Constructors

The `U256` struct implements the following constructors: `from_u8`, `from_u32`, and `from_u64`.

### Binary Operations

The `U256` struct implements the following binary operations: `addition`, `subtraction`, `multiplication`, `bitwise and`, `bitwise or`, `bitwise xor`, `bitwise shift right`, and `bitwise shift left`. All operations will wrap the result when the result is outside the range of the `U256` type.

All of the operations can be used in 6 different ways:
`U256 op U256` or `U256 op &U256` or `&U256 op U256` or `&U256 op &U256` or `U256 op= U256` or `&U256 op= U256`.

### Other

When using the `U256` struct with `target_os = "zkvm"`, the struct utilizes efficient implementations of comparison operators as well as the `clone` method.

## `I256`

The `I256` struct is a 256-bit signed integer type. The `I256` struct is very similar to the `U256` struct.

### Constants

The `I256` struct has the following constants:

- `MAX`: The maximum value of a `I256`.
- `MIN`: The minimum value of a `I256`.
- `ZERO`: The zero constant.

### Binary Operations

The `I256` struct implements the following binary operations: `addition`, `subtraction`, `multiplication`, `bitwise and`, `bitwise or`, `bitwise xor`, `bitwise shift right`, and `bitwise shift left`. All operations will wrap the result when the result is outside the range of the `I256` type. Note that unlike the `U256`, when performing the shift right operation `I256` will perform an arithmetic shift right (i.e. sign extends the result).

All of the operations can be used in 6 different ways:
`I256 op I256` or `I256 op &I256` or `&I256 op I256` or `&I256 op &I256` or `I256 op= I256` or `&I256 op= I256`.

### Constructors

The `I256` struct implements the following constructors: `from_i8`, `from_i32`, and `from_i64`.

### Other

When using the `I256` struct with `target_os = "zkvm"`, the struct utilizes efficient implementations of comparison operators as well as the `clone` method.



## External Linking

The Bigint Guest extension provides another way to use the native implementation. It provides external functions that are meant to be linked to other external libraries. The external libraries can use these functions as a hook for the 256 bit integer native implementations. Enabled only when the `target_os = "zkvm"`. All of the functions are defined as `unsafe extern "C" fn`. Also, note that you must enable the feature `export-intrinsics` to make them globally linkable.

- `zkvm_u256_wrapping_add_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a + b`.
- `zkvm_u256_wrapping_sub_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a - b`.
- `zkvm_u256_wrapping_mul_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a * b`.
- `zkvm_u256_bitxor_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a ^ b`.
- `zkvm_u256_bitand_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a & b`.
- `zkvm_u256_bitor_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a | b`.
- `zkvm_u256_wrapping_shl_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a << b`.
- `zkvm_u256_wrapping_shr_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a >> b`.
- `zkvm_u256_arithmetic_shr_impl(result: *mut u8, a: *const u8, b: *const u8)`: takes in a pointer to the result, and two pointers to the inputs. `result = a.arithmetic_shr(b)`.
- `zkvm_u256_eq_impl(a: *const u8, b: *const u8) -> bool`: takes in two pointers to the inputs. Returns `true` if `a == b`, otherwise `false`.
- `zkvm_u256_cmp_impl(a: *const u8, b: *const u8) -> Ordering`: takes in two pointers to the inputs. Returns the ordering of `a` and `b`.
- `zkvm_u256_clone_impl(result: *mut u8, a: *const u8)`: takes in a pointer to the result buffer, and a pointer to the input. `result = a`.

And in the external library, you can do the following:

```rust
extern "C" {
    fn zkvm_u256_wrapping_add_impl(result: *mut u8, a: *const u8, b: *const u8);
}

fn wrapping_add(a: &Custom_U256, b: &Custom_U256) -> Custom_U256 {
    #[cfg(target_os = "zkvm")] {
        let mut result: MaybeUninit<Custom_U256> = MaybeUninit::uninit();
        unsafe {
            zkvm_u256_wrapping_add_impl(result.as_mut_ptr() as *mut u8, a as *const u8, b as *const u8);
        }
        unsafe { result.assume_init() }
    }
    #[cfg(not(target_os = "zkvm"))] {
        // Regular wrapping add implementation
    }
}
```

### Config parameters

For the guest program to build successfully add the following to your `.toml` file:

```toml
[app_vm_config.bigint]
```
