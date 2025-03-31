# Circuit Architecture

Author: [Golovanov399](https://github.com/Golovanov399)

## 1. Introduction

Scope: https://github.com/openvm-org/openvm

Commit: 6d019463b50dc315553aa81007dfc85a0c0736e4

This review is about establishing that using our circuit model properly guarantees the correctness of the program workflow, and defining what "properly" means as well as our circuit model.

## 2. Findings

### 2.1 VolatileBoundaryAir does not range check address

Author: [jonathanpwang](https://github.com/jonathanpwang)

**Severity:** High

**Context:** https://github.com/openvm-org/openvm/blob/f3461e8e71dcd57d3c7ed5cc42592a4cf492c434/crates/vm/src/system/memory/volatile/mod.rs#L102

**Description:**
The `VolatileBoundaryAir` uses `IsLtArrayWhenTransitionAir` to assert that the `address = (address_space, pointer)` on the rows are all sorted in strictly increasing order, which 
enforces the addresses are unique. However the `IsLtArrayWhenTransitionAir` has an **assumption** that the two arrays to compare have elements that are already range checked.
The `VolatileBoundaryAir` does not constrain these range checks, which means it is possible to initialize the memory bus with addresses that are out of bounds, breaking an invariant of the memory bus.

I believe one can construct a case where the difference between pointers remains in `[0, 2^29)` and the differences are large enough that summing the differences results in a number congruent to `0 mod p`
where `p` is the modulus of `F`. This would break the uniqueness claim of the addresses.

**Recommendation:** Add range checks to `address`.

**Resolution:** https://github.com/openvm-org/openvm/pull/1490
https://github.com/openvm-org/openvm/commit/c9339e6ee8c52ee047eab2fefc94fea0926f04b8

VolatileBoundaryAir was using IsLessThanArray, which compares `x < y` **assuming** that `x, y` are both in a range `[0, max_bits)`. But for the volatile boundary case, we **must** range check this assumption, where `x = (addr_space, pointer)`.

We fix this by decomposing `addr_space, pointer` both into limbs and range checking.
I slightly optimized to only support `addr_space in [0, 2^range_max_bits)` and `pointer in [0, 2^{2*range_max_bits})` but did it with configurable constants so this could be changed to support a greater max bits in address space in the future.

### 2.2 Initial timestamp is not constrained to be `0`

Author: [zlangley](https://github.com/zlangley)

**Severity:** High

**Description:**
In the connector chip, it is constrained that the final timestamp is less than `2^29`, which, together with the fact that the actual value of the timestamp does not exceed the number of interactions, which is less than `p`, guaranteed that the total increase of the timestamp is at most `2^29`, and memory is sound. However, if the initial timestamp turns out to be greater than the final one, the total increase of the timestamp may be up to `p`, which renders all the memory timestamp checks unsound.

**Recommendation:** Check that the initial timestamp is `0` in the connector chip.

**Resolution:** https://github.com/openvm-org/openvm/pull/1495
https://github.com/openvm-org/openvm/commit/c136e5788d59fe7f4563cd1d3112af1e2066101f

### 2.3 `MemoryController::with_volatile_memory` constructor underestimates max address space bits

Author: [jonathanpwang](https://github.com/jonathanpwang)

**Severity:** Low

**Context:** https://github.com/openvm-org/openvm/blob/f3461e8e71dcd57d3c7ed5cc42592a4cf492c434/crates/vm/src/system/memory/controller/mod.rs#L241

**Description:**
The `MemoryController::with_volatile_memory` constructor uses `as_height` as `addr_space_max_bits` in the `VolatileBoundaryChip` constructor. However, the address spaces are in the range `[0, AS_OFFSET + 2^addr_space_max_bits)` where `AS_OFFSET = 1`, so `addr_space_max_bits` is actually one higher. This would cause a runtime panic in trace generation if the full range of address spaces is used.

**Recommendation:** Fix the constructor.

**Resolution:** https://github.com/openvm-org/openvm/pull/1496
https://github.com/openvm-org/openvm/commit/8d934b3d61ffd72cec7de914a72010f3de9f238b


## 3. Discussion

### 3.1 Circuit architecture must constrain guest execution

_In the ISA spec we define a notion of [guest instruction execution](https://github.com/axiom-crypto/openvm-private/blob/specs/update/docs/specs/ISA.md#guest-instruction-execution). We discuss how the circuit architecture ensures that the guest execution is fully constrained by the VM circuit._

We range check stuff from streams in the proof of `hintstore`.

### 3.2 System Buses

_Discuss how they fall into STARK backend interactions framework._

See https://github.com/openvm-org/openvm/blob/main/docs/specs/circuit.md.

### 3.3 Timestamp overflow

_Timestamp overflow was previously found in cantina [https://cantina.xyz/code/c486d600-bed0-4fc6-aed1-de759fd29fa2/findings/11]. We discuss the issue and how it is currently safeguarded against._

We ensured that the total timestamp increment is bounded from above by the number of interactions, which is `< p` because the backend ensures that, and now we just check in the connector that the final timestamp `< 2^29`.

For completeness, we also need to check that the initial timestamp is less than the final one, but we just check that it's zero: https://github.com/openvm-org/openvm/pull/1467.

### 3.4 In `MemoryReadOrImmediateOperation::eval`, I think we can decrease the degree of expressions

We use `enabled * not(is_immediate)`, while we can use `enabled - is_immediate` -- `enabled` is boolean because I checked all occurrences, and `is_immediate` is boolean because it's a result of `IsZeroAir` stuff.

### 3.5 Can memory reads please not increase the timestamp?

I claim that they cannot. Reason: if memory read has `timestamp == prev_timestamp`, then it basically does two opposite interactions with `[address, data, timestamp]`, therefore does not contribute anything to the LogUp thing. We could just change the data in the trace to whatever we want, and it would still be accepted.

One way to think about it is that the corresponding edge in the timeline graph/network would be a self-loop, and any flow can just not go there -- and all our arguments there are essentially about a flow having to go through all meaningful edges.

### 3.6 When we split-merge stuff in memory controller, we sometimes do operations that cancel each other out within one split-merge.

For example, when we need to split `[0, 8)` until we have `[0, 1)`, we do:

- receive `[0, 8)`, send `[0, 4)`, send `[4, 8)`,
- receive `[0, 4)`, send `[0, 2)`, send `[2, 4)`,
- receive `[0, 2)`, send `[0, 1)`, send `[1, 2)`.
  We just did 9 interactions, where `[0, 4)` and `[0, 2)` could have been omitted (so we could only do 5).

Is it worth optimizing? Or is it rare enough not to bother?
