# openvm-continuations

Basic provers for the continuation aggregation pipeline. Each basic prover generates a single aggregated `Proof` from child `Proof`s for a specific aggregation subcircuit (i.e. **inner** and **root**).

For the full specification (layer architecture, subcircuit constraints, public value layouts), see the [continuations spec](../../docs/vocs/docs/pages/specs/architecture/continuations.mdx).

## Inner Basic Prover

Used for the **leaf**, **internal-for-leaf**, and **internal-recursive** layers, which all share the inner aggregation subcircuit.

### Constructor

```rust
pub fn new(
    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    system_params: SystemParams,
    is_self_recursive: bool,
    def_hook_cached_commit: Option<Digest>,
) -> Self
```

- `child_vk` — verifying key for child proofs
- `system_params` — parent system parameters
- `is_self_recursive` — indicates whether this prover is internal-recursive (i.e., can use its own `vk` as `child_vk`)
- `def_hook_cached_commit` — optional cached trace commit for the deferral hook verifier; use `None` for VM-only aggregation

Constructing the prover pre-generates the child layer's **cached trace commit** (and, together with the child VK's pre-hash, the **vk commit**) as well as the parent layer's proving and verifying keys. Functions to construct the prover from a saved `pk` and cached trace are available.

Each inner layer should use the following constructor arguments:

| Layer | `child_vk` | `system_params` | `is_self_recursive` |
|-------|-----------------|------------------|-------|
| leaf | `app_vk` | leaf | false |
| internal-for-leaf | `leaf_vk` | internal | false |
| internal-recursive | `internal_for_leaf_vk` | internal | true |

Note that internal-for-leaf and internal-recursive share the same system parameters, and that there is only one internal-recursive prover.

### Prove API

For VM-only aggregation, call:

```rust
pub fn agg_prove_no_def(
    &self,
    proofs: &[Proof<SC>],
    child_vk_kind: ChildVkKind,
) -> Result<Proof<SC>>
```

For mixed VM/deferral aggregation in the combined tree, call:

```rust
pub fn agg_prove(
    &self,
    proofs: &[Proof<SC>],
    child_vk_kind: ChildVkKind,
    proofs_type: ProofsType,
    absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
) -> Result<Proof<SC>>
```

- `proofs` — child `Proof`s to aggregate
- `child_vk_kind` — enum that indicates whether the child proofs are from the app layer (`App`), the same layer recursively (`RecursiveSelf`), or a different layer (`Standard`)
- `proofs_type` — distinguishes VM-only, deferral-only, and mixed aggregation in the combined tree
- `absent_trace_pvs` — optional absent-proof public values used when padding deferral proofs in the mixed tree

Each layer uses the following inputs:

| Layer | `proofs` | `child_vk_kind` |
|-------|-----------------|------------------|
| leaf | app proofs | `App` |
| internal-for-leaf | leaf proofs | `Standard` |
| internal-recursive (layer 0) | internal-for-leaf proofs | `Standard` |
| internal-recursive (layer 1+) | internal-recursive proofs | `RecursiveSelf` |

## Root Basic Prover

Used for the **root** layer, which wraps a single internal-recursive `Proof` and constrains user public values' presence in memory.

### Constructor

```rust
pub fn new(
    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    internal_recursive_cached_commit: CommitBytes,
    system_params: SystemParams,
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
    def_hook_commit: Option<CommitBytes>,
    trace_heights: Option<Vec<usize>>,
) -> Self
```

- `child_vk` — the `internal_recursive_vk`
- `internal_recursive_cached_commit` — the cached trace commit of the internal-recursive verifier circuit (the `vk_pre_hash` is derived from `child_vk` internally)
- `system_params` — parent system parameters
- `memory_dimensions` — the memory dimensions used for app execution, used to compute whether each sibling hash in the Merkle proof should be the left or right sibling
- `num_user_pvs` — number of user public values
- `def_hook_commit` — commitment hash of the deferral hook aggregation tree
- `trace_heights` - constant heights that the traces of the root proof must be

The constructor pre-generates the parent proving and verifying keys, which can be saved.

### Prove API

Root proving is a two-step flow:

```rust
pub fn generate_proving_ctx_no_def<PB>(
    &self,
    proof: Proof<SC>,
    user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
) -> Option<ProvingContext<PB>>

pub fn root_prove_from_ctx<E>(&self, ctx: ProvingContext<E::PB>) -> Result<Proof<RootSC>>
```

If deferrals are enabled, use `generate_proving_ctx(...)` instead and pass the deferral Merkle proofs.

- `proof` — the child internal-recursive `Proof`
- `user_pvs_proof` — the user public values Merkle proof (proving presence in memory); generated at the app layer
- `ctx` — the fully assembled proving context returned by `generate_proving_ctx*`
