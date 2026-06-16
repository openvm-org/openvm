# Recursion Crate

In continuations, a **layer** is one level of the aggregation tree: it takes proofs from the layer below, verifies them, enforces the required public-value consistency, and outputs proofs for the layer above. The following layers' circuits are defined using the recursion crate:

- `app`: VM execution produces one proof per segment.
- `leaf`: aggregates app segment proofs.
- `internal-for-leaf`: aggregates leaf proofs.
- `internal-recursive`: repeatedly aggregates internal proofs until one remains.
- `root`: wraps the final internal-recursive proof and decommits user public values.

## Circuit Fields and Dependencies

To ensure the `internal_recursive_vk` and `root_vk` are constant for all possible `app_vk`'s, we impose several restrictions on how verifier circuit AIRs can and cannot depend on the child's `vk`. To start, we divide the `vk` fields into 4 categories:

- symbolic constraints DAG - per-AIR evaluation constraints (including symbolic interactions) represented as a directed acyclic graph
- system (configurable) parameters - per-layer configurable parameters that modify the verifier protocol
- constant fields - per-layer non-configurable parameters that the verifier circuit assumes to be constant across the leaf and internal layers
- dependent fields - per-layer non-configurable parameters that depend on constant parameters

Note we choose which non-configurable fields are constant and dependent ([see below](#Non-Configurable-Fields)).

By definition, parent system parameters and constant fields must be independent of the child `vk`.  We also impose that:

- Parent dependent fields may depend on the child's constant fields
- The parent symbolic constraint DAG can depend on the child system parameters, constant fields, and dependent fields at all levels
- No other parent-child dependency may exist

In table form (where ✅ means the parent category can depend on the child category):

| **Parent \\ Child**                      | Symbolic Constraints DAG | System (Configurable) Parameters | Constant Fields | Dependent Fields |
|------------------------------------------|--------------------------|----------------------------------|-----------------|------------------|
| **Symbolic Constraints DAG**             | ❌                       | ✅                               | ✅              | ✅               |
| **System (Configurable) Parameters**     | ❌                       | ❌                               | ❌              | ❌               |
| **Constant Fields**                      | ❌                       | ❌                               | ❌              | ❌               |
| **Dependent Fields**                     | ❌                       | ❌                               | ✅              | ❌               |

## Non-Configurable Fields

We get to choose which non-configurable fields are constant and which are dependent. As of now, the following need to be constant across the leaf and internal layers:

- Number and order of AIRs
- Number and location of cached traces
- Number, location, and hypercube dimension of preprocessed traces
- Number of interactions per row
- Number, location, and order of public values

The following can depend on constant fields:

- AIR common main widths
- Number of constraints

We categorize other non-configurable fields as needed.
