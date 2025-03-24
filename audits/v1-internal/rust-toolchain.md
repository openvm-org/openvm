# Rust Toolchain

Author: [Golovanov399](https://github.com/Golovanov399)

## 1. Introduction

Scope: `crates/toolchain`
Commit: `a46aad0a9b01cb455ac151b12b83afb8ec61536a`

The review is focused on establishing the safety and correctness of our rust toolchain.

## 2. Findings

### 2.1 `string_to_bytes` is very inconsistent about spaces and in general

**Severity:** Informational

**Context:** `crates/toolchain/macros/src/lib.rs`

**Description:** First of all, regarding the fact that it's only used in the algebra moduli macros -- see Disc. 3.5. Second, it is very weird that we skip whitespaces when decoding `0x...` and don't when parsing a decimal.

<!-- **Proof of concept:** (optional) sample code/patch or other way to demonstrate the exploit -->

**Recommendation:** Sort out the relation to other similar functions, then `///` document the usage.

<!-- **Resolution:** link to the PR where this is resolved, once done -->


### 2.2 Potential underflow in the heap embedded allocator

**Severity:** Low

**Context:** `crates/toolchain/platform/src/heap/embedded.rs`

**Description:** There is this function:

```rust
pub fn init() {
    extern "C" {
        static _end: u8;
    }
    let heap_pos: usize = unsafe { (&_end) as *const u8 as usize };
    let heap_size: usize = crate::memory::GUEST_MAX_MEM - heap_pos;
    unsafe { HEAP.init(heap_pos, heap_size) }
}
```

If we turn out to have a lot of unitialized data or otherwise make `_end` very far in memory, then `heap_pos` may exceed `crate::memory::GUEST_MAX_MEM`, which will probably lead to just a crash.

<!-- **Proof of concept:** (optional) sample code/patch or other way to demonstrate the exploit -->

**Recommendation:** It is better to produce a meaningful error.

**Resolution:** https://github.com/openvm-org/openvm/pull/1483

### 2.3 Irrelevant comment

**Severity:** Informational

**Context:** `crates/toolchain/openvm/src/io/mod.rs`

**Description:** In this code:

```rust
/// Read the next vec and deserialize it into a type `T`.
pub fn read<T: DeserializeOwned>() -> T {
    let reader = read::Reader::new();
    let mut deserializer = Deserializer::new(reader);
    T::deserialize(&mut deserializer).unwrap()
}
```

the comment is outdated.

<!-- **Proof of concept:** (optional) sample code/patch or other way to demonstrate the exploit -->

**Recommendation:** Update it.

**Resolution:** https://github.com/openvm-org/openvm/pull/1484

### 2.4 `read_n_bytes` is quadratic

**Severity:** Informational

**Context:** `crates/toolchain/openvm/src/host.rs`

**Description:** In this code:

```rust
pub fn read_n_bytes(n: usize) -> Vec<u8> {
    HINT_STREAM.borrow_mut().drain(..n).collect()
}
```

it takes linear time to read `n` bytes in the remaining size of the stream. However, this will only be used for testing, if ever, so not a problem.

<!-- **Proof of concept:** (optional) sample code/patch or other way to demonstrate the exploit -->

**Recommendation:** Keep the stream reversed and drain from the end.

<!-- **Resolution:** link to the PR where this is resolved, once done -->


### 2.5 `sys_alloc_aligned` should check that align is a power of two since it relies on it

**Severity:** Informational

**Context:** `crates/toolchain/platform/src/memory.rs`

**Description:** `sys_alloc_aligned` uses `& (align - 1)` but never checks that `align` is a power of two -- although in all usages it is.

<!-- **Proof of concept:** (optional) sample code/patch or other way to demonstrate the exploit -->

**Recommendation:** Add a check.

**Resolution:** None. The rust `alloc::Layout` already says that `align` must be a power of two.

### 2.6 `read_vec_by_len` uses unsafe Rust that could misbehave with a different global allocator

**Severity:** Low 

**Context:** https://github.com/openvm-org/openvm/blob/aa99bbf8c136541487d6bbefdc29eb06dae94c6c/crates/toolchain/openvm/src/io/mod.rs#L74

**Description:** In the implementation, `ptr_start` is allocated with `Layout::from_size_align(capacity, 4)` and then `Vec::from_raw_parts` is called to create a `Vec<u8>`. 
The safety requirements in the documentation for `from_raw_parts` states:

> `T` needs to have the same alignment as what `ptr` was allocated with. (`T` having a less strict alignment is not sufficient, the alignment really needs to be equal to satisfy the `dealloc` requirement that memory must be allocated and deallocated with the same layout.)

Here `T = u8` has alignment `1` so this requirement is actually not satisfied.
It is currently not a problem because both global allocators that can be used -- bump and `embedded-alloc` -- have implementations where alignment is always rounded up to at least `4`.
The `embedded-alloc` crate uses `linked-list-allocator`, which has a minimum alignment of
`sizeof(usize) * 2 = 8` on 32-bit architectures: https://github.com/rust-osdev/linked-list-allocator/blob/b5caf3271259ddda60927752fa26527e0ccd2d56/src/hole.rs#L429

However, if the global allocator was changed to one without this minimum alignment property, then the `Vec` dealloc implementation may not be consistent with how it was allocated.

**Recommendation:** Just allocate using `Vec::with_capacity`. The allocator will round the alignment up to `4` for the aforementioned reasons anyways.
Since `hint_buffer_u32` does not require `ptr_start` to be 4-byte aligned, using the `u8` alignment of `1` is also technically correct.

**Resolution:** https://github.com/openvm-org/openvm/pull/1489


## 3. Discussion

Discussion is for general discussion or additional writing about what has been studied, considered, or understood that did not result in a concrete finding. Discussions are useful both to see what were important areas that are security critical and why they were satisfied. In the good case, the review should have few findings but lots of discussion.

### 3.1 DDoS

We should be careful about users not ddosing us, although, as already discussed, this should not be a problem -- but as a precaution, we could at least add a possibility to set a time limit on building the program (not sure how to set a time limit on the actual execution).

### 3.2 `TargetFilter` kind

is always either `bin` or `example`, not a big issue but maybe it would be a good design decision to make an enum for it to reflect that fact. Also, I hope that there are no scenarios where our target matching can let us down (I think we should be fine, but I don't know what to look for).

### 3.3 `from_isize` and `from_usize` have too different signatures

Probably it makes sense to create something generic that would accept everything we wanted.

### 3.4 Why is stuff where it is?

- Is toolchain a good place for `Instruction`?
- Is `instructions` module a good place for `parse_biguint_auto`? Also, we don't use it.
- Same about where `biguint_to_limbs` is. And `string_to_bytes`. Clearly this whole thing was written to match our specific use cases back then and needs restructuring.
