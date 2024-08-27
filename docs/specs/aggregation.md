# Aggregation

We describe our strategy for aggregating STARK proofs at a high-level.

<!--Some details are subject to change-->

## Static Aggregation

Assume that we have a static (i.e., known ahead of time) list `allowed_vks` of STARK verifying keys (unique identifiers for STARK circuits).

Suppose we have a variable-length list of proofs `proofs` where `proofs.len()` is independent of `allowed_vks.len()`. The goal is to produce a single STARK proof that asserts that `proofs[i]` verifies with respect verifying key `vk[i]` where `allowed_vks` contains `vk[i]`, for all `i`. Additionally, there should be the optionality to store a commitment to the ordered list of `(hash(vk[i]), public_values[i])` where `public_values[i]` are the public values of proof `i`.

We aggregate `proofs` using a tree-structure. The arity of the tree can be adjusted for performance;
by default it is 2. The height of the tree is variable and equal to $\lceil \log{n} \rceil$ where $n$ is the number of proofs and the base of logarithm is the arity.

We distinguish between three types of nodes in the tree:

- Leaf
- Internal
- Root

Each node of the tree will be a STARK VM circuit, _without continuations_, proving a VM program that runs STARK verification on an `arity` number of proofs. We make the distinction that each type of node in the tree may be a **different** VM circuit, meaning with different chip configurations. All VM circuits must support the opcodes necessary to do STARK verification.

For each node type, a different program is run in the VM circuit:

- Leaf: the program verifies `<=leaf_arity` proofs, where each proof is verified with respect to one of the verification keys in `allowed_vks`. The leaf program will have the proof, public values, and verifying keys of each proof in program memory, and the program can be augmented with additional checks (for example, state transitions checks are necessary for continuations).
- Internal: the program verifies `<= internal_arity` proofs, where all proofs are verified with respect to the same verifying key. This verifying key is either that of a leaf circuit or that of an internal circuit (the present circuit itself). The circuit cannot know the verifying key of itself, so to avoid a circular dependency, the hash of the verifying key is made a public value.
- Root: this program _may_ just be the same as the Internal program, but for the purposes of optimizing [on-chain aggregation](#on-chain-aggregation), there is the possiblity for it to be different. The root program verifies `<= root_arity` proofs, where all proofs are of the internal circuit. Note that `root_arity` may be `1`.

### STARK Configurations

Before proceeding, we must discuss the topic of STARK configurations: any STARK proof depends on at least three configuration parameters:

- `F` the base field of the AIRs
- `EF` the extension field of the AIRs used for challenge values
- the hash function used for the FRI PCS. This hash function must be able to hash `F` and `EF` elements, where elements can be packed before hashing.

For all Leaf and Internal circuits [above](#static-aggregation), we use an **Inner Config**. Example Inner Configs are:

- `F` is BabyBear, `EF` is quartic extension of BabyBear, hash is BabyBearPoseidon2
- `F` is BabyBear, `EF` is quartic extension of BabyBear, hash is SHA256
- `F` is Mersenne31, `EF` is quartic extension of Mersenne31, hash is Mersenne31Poseidon2
- `F` is Mersenne31, `EF` is quartic extension of Mersenne31, hash is SHA256

We discuss considerations for choice of hash below.

On the other hand, the Root circuit will use an **Outer Config**. Example Outer Configs are:

- `F` is BabyBear, `EF` is quartic extension of BabyBear, hash is BN254FrPoseidon2 (or BN254FrPoseidon1)
- ~~`F` is BabyBear, `EF` is quartic extension of BabyBear, hash is SHA256~~
- `F` is BN254Fr, `EF` is BN254Fr, hash is BN254FrPoseidon2 (or BN254FrPoseidon1)
- ~~`F` is BN254Fr, `EF` is BN254Fr, hash is SHA256~~
- Analogous configurations with BabyBear replaced with Mersenne31.

To explain, since `31 * 8 < 254`, eight BabyBear field elements can be packed together and embedded (non-algebraically) into a BN254Fr field element. In this way BN254FrPoseidon2 can be used to hash BabyBear elements.

The choice of hash function in the Outer Config only affects what hash must be verified in the Halo2 circuit for on-chain aggregation (see [below](#on-chain-aggregation)). For performance, it is therefore always better to use BN254FrPoseidon2 for the Outer Config.

### On-chain Aggregation

The Root circuit above is the last STARK circuit, whose single proof will in turn verify all initial `proofs`. Due to the size of STARK proofs, for on-chain verification we must wrap this proof inside an elliptic curve based SNARK proof so that the final SNARK proof can be verified on-chain (where on-chain currently means within an Ethereum Virtual Machine).

We create a Halo2 circuit that verifies any proof of the Root STARK circuit. This is a non-universal circuit whose verifying key depends on the specific STARK circuit to be verified. The majority of the verification logic can be code-generated into the `halo2-lib` eDSL which uses a special vertical custom gate specialized for cheap on-chain verification cost. There are two main performance considerations:

#### 1. Hash

To perform FRI verification in the Halo2 circuit, the circuit must constrain calculations of STARK Outer Config hashes. As mentioned above, this hash will be BN254FrPoseidon2. The constraints for this hash can either be implemented directly using the `halo2-base` vertical gate, or with a custom gate. The custom gate will be faster but with higher verification cost. There are two approaches to consider:

Approach A

- Use a single Halo2 circuit with only thinnest `halo2-base` vertical gate to verify the Root STARK circuit proof.

Approach B

- Use a first Halo2 circuit with custom gate for BN254FrPoseidon2 to verify the Root STARK circuit proof.
- Use a second Halo2 circuit with only the thinnest `halo2-base` vertical gate to verify the previous Halo2 circuit.

Approach B is likely better, provided that the time to generate both proofs is faster than the time to generate the single proof in Approach A.

#### 2. Outer Config Base Field

The Outer Config base field `F` can be either a 31-bit field or BN254Fr.

When `F` is 31-bit field:

- For FRI folding and other arithmetic in STARK verification, the Halo2 circuit must perform BabyBear prime field arithmetic and extension field arithmetic inside the halo2 circuit. These are non-native arithmetic operations.

When `F` is BN254Fr and `EF` is BN254Fr:

- Halo2 circuit only needs to perform native field arithmetic inside the halo2 circuit.
- The Root STARK circuit must now perform non-native BabyBear field arithmetic and extension field arithmetic inside the STARK to support the verification of the STARKs with the Inner Config. This non-native arithmetic is still expected to be much faster in the STARK than in Halo2, but the added chip complexity may also increase verifier cost in the Halo2 circuit.
- If the Inner Config hash is BabyBearPoseidon2, now the Root STARK circuit must constrain BabyBearPoseidon2 inside a circuit with base field BN254Fr. This is definitely not efficient. **Therefore it is not possible for the Outer Config base field to be BN254Fr if the Inner Config hash is BabyBearPoseidon2.**
- This Outer Config is only possible if the Inner Config hash is a hash that does not depend on the native field (e.g., SHA256 or Blake2b or Blake3).
  - **Observation:** even if the hash used for the Internal circuit is SHA256, the Leaf circuit can still be proven using BabyBearPoseidon2. Likewise, it is even possible to have the Internal circuits use BabyBearPoseidon2 at higher depths in the tree (away from the root). The only requirement is that the last Internal circuit proof, which will be verified by the Root circuit, needs to be proven with SHA256 as the hash.

TODO: to determine which Outer Config is best, we will:

- Instrument the cost of non-native small field arithmetic in the Halo2 circuit.
- Benchmark an aggregation VM with Inner Config hash BabyBearPoseidon2 proven over BabyBearPoseidon2 versus one with Inner Config hash SHA256 proven over SHA256.

## Dynamic Aggregation

TODO
