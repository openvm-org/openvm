# Recursion Crate

TODO: Explain recursion layers

## AIR Development Guide

To ensure the `root_vk` is constant for all possible `app_vk`'s, we impose several restrictions on how verifier circuit AIRs can and cannot depend on the child's `vk`. To start, we divide the `vk` fields into 4 categories:

- symbolic constraints DAG - per-AIR evaluation constraints (including symbolic interactions) represented as a directed acyclic graph
- system (configurable) parameters - per-layer configurable parameters that modify the verifier protocol
- constant fields - per-layer non-configurable parameters that the verifier circuit assumes to be constant across all verifier layers (but not the app layer)
- dependent fields - per-layer non-configurable parameters that depend on constant parameters

Note we choose which non-configurable fields are constant and dependent ([see below](#Non-Configurable-Fields)).

By definition, parent system parameters and constant fields must be independent of the child `vk`.  We also impose that:

- Parent dependent fields may depend on the child's constant fields
- The parent symbolic constraint DAG can depend on the child constant and dependent fields at all levels, and the child system parameters at the leaf and internal levels
- No other parent-child dependency may exist

In table form (where ✅ means the parent category can depend on the child category):

| **Parent \\ Child** | Symbolic Constraints DAG | System (Configurable) Parameters | Constant Fields | Dependent Fields |
|----------------------|--------------------------|----------------------------------|-----------------|------------------|
| **Symbolic Constraints DAG** | ❌ | ✅ if non-root| ✅  | ✅ |
| **System (Configurable) Parameters** | ❌ | ❌ | ❌ | ❌ |
| **Constant Fields** | ❌ | ❌ | ❌ | ❌ |
| **Dependent Fields** | ❌ | ❌ | ✅ | ❌ |

#### Non-Configurable Fields

We get to choose which non-configurable fields are constant and which are dependent. As of now, the following need to be constant at all verifier layers:

- Number and order of AIRs
- Number and location of cached traces
- Number, location, and hypercube dimension of preprocessed traces
- Number of interactions per row
- Number, location, and order of public values

The following can depend on constant fields:

- AIR common main widths
- Number of constraints

We categorize other non-configurable fields as needed. If a non-configurable field is not categorized, it should **not** be used.
