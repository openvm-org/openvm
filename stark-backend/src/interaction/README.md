# AIR Interactions (Cross-table lookups)

We explain the interface and implementation of the communication protocol between different AIR matrices introduced by Valida here. We note that this allows AIRs with matrices of
different heights to communicate. See [here](https://hackmd.io/@shuklaayush/rJHhuWGfR) for another reference.

## Interface

The main interface is controlled by the trait [`Chip`](./chip.rs)

```rust
pub trait Chip<F: Field> {
    fn sends(&self) -> Vec<Interaction<F>> {
        vec![]
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        vec![]
    }
}

#[derive(Clone, Debug)]
pub struct Interaction<F: Field> {
    pub fields: Vec<VirtualPairCol<F>>,
    pub count: VirtualPairCol<F>,
    pub argument_index: usize,
}
```

The `Chip` trait should be implemented on a struct implementing the `Air` trait.
For a given chip, the interface allows to specify sends and receives via the
`Interaction` struct. A single interaction $\sigma$ specifies a [communication] bus
to communicate over -- this bus is an abstract concept that is not explicitly materialized.
The index of this bus is `argument_index`, which we call $i_\sigma$ in the following.
The interaction specifies `fields` $(f_j)$ and `count` $m$ where each $f_j$ and $m$ is a polynomial expression
on the main and preprocessed trace polynomials with rotations. This means that we want to send the tuple
$(f_1(\mathbf T),\dotsc,f_{len}(\mathbf T))$ to the $i$-th bus with multiplicity $m(\mathbf T)$, where $\mathbf T$
refers to the trace (including preprocessed columns) as polynomials (as well as rotations).

### Outcome

If all row values for `count` for sends are small enough that the sum of all `count` values across all `sends` is strictly smaller than the field characteristic (so no overflows are possible), this enforces that:

> for each bus, each unique row of `fields` occurs with the same total `count` in sends and receives across all chips.

In other words, for each bus, there is a multiset equality between

> the multiset union of the rows of `fields` with multiplicity `count` across all sends

and

> the multiset union of the rows of `fields` with multiplicity `count` across all receives.

One important consequence is that:

> for each bus, each row of a `fields` with non-zero `count` from a send coincides with some row of a `fields` of a receive (possibly in another chip).

In other words, it enforces a cross-chip lookup of the rows of the send tables with non-zero `count` into the concatenation of the receive tables.

### Conventions

Following Valida, we will follow the convention that if an individual chip is the owner of some functionality, say `f(x) = y`, then the chip itself should add `receive`
interactions to _receive_ requests with fields `(x, y)` and constrain correctness of `f(x) = y`. Any other chip in a system that wants to use this functionality should
add `send` interactions to _send_ requests for this functionality.

## Backend implementation via logUp

The backend implementation of the prover will constrain the computation of a cumulative sum
_for just this AIR_
$$\sum_r \left(\sum_\sigma sign(\sigma) \frac {m_\sigma[r]}{\alpha^{i_\sigma} + \sum_j \beta^j \cdot f_{\sigma,j}(\mathbf T[r])} \right)$$
where $r$ sums over all row indices, $\sigma$ sums over all sends and receives, $sign(\sigma) = 1$ if $\sigma$ is a send, $sign(\sigma) = -1$ if $\sigma$ is a receive.

- $\alpha,\beta$ are two random challenge extension field elements.
- The reciprocal is the logUp logarithmic derivative argument.
- $\alpha^{i_\sigma}$ is used to distinguish the bus index.
- $\sum_j \beta^j \cdot f_{\sigma,j}$ is the RLC of the $(f_{\sigma,j})$ tuple.
- Add the sends, subtract the receives.

Globally, the prover will sum this per-AIR cumulative sum over all AIRs and lastly constrain that the sum is $0$. This will enforce that the sends and receives are balanced globally across all AIRs. Note that the multiplicity allows a single send to a bus to be received by multiple AIRs.

### Virtual columns and constraints

In theory the $f_j, m$ can be any multi-variate polynomial expression. Currently plonky3 only supports affine expressions (degree <= 1 polynomials), which are constructed via the `VirtualPairCol` struct.
A `VirtualPairCol` is an affine function over a set of columns of the form $f(\mathbf T) = b + \sum w_i T_i$.

```rust
pub struct VirtualPairCol<F: Field> {
    column_weights: Vec<(PairCol, F)>,
    constant: F,
}
```

As such, the RLC $\sum_j \beta^j \cdot f_j$ is a linear polynomial.

For each send/receive interaction, we must add one virtual column $q_\sigma$ with row $r$ equal to
$$q_\sigma[r] = \frac {m_\sigma[r]}{\alpha^{i_\sigma} + \sum_j \beta^j \cdot f_{\sigma,j}(\mathbf T[r])}$$
The constraint is
$$q_\sigma \cdot \left(\alpha^{i_\sigma} + \sum_j \beta^j \cdot f_{\sigma,j}(\mathbf T) \right) = m_\sigma(\mathbf T)$$
has degree $max(1 + max_j deg(f_{\sigma,j}), deg(m_\sigma))$ ($=2$ in the case all functions are affine).

Note: we could save columns by combining $q$ columns together, at the cost of increasing the constraint degree.

We need one more virtual column $\phi$ for the cumulative sum of all sends and receives. The row $r$ of $\phi$ contains the partial sum of all reciprocals up to row $r$.
$$\phi[r] = \sum_{r' \leq r} \left(\sum_\sigma q_\sigma[r'] - \sum_\tau q_\tau[r'] \right)$$

The constraints are:

- $sel_{first} \cdot \phi = sel_{first} \cdot (\sum_\sigma q_\sigma + \sum_\tau q_\tau)$
- $sel_{transition} \cdot (\phi' - \phi) = sel_{transition} \cdot (\sum_\sigma q_\sigma' - \sum_\tau q_\tau')$ where $\phi'$ and $q'$ mean the next row (rotation by $1$).
- $sel_{last} \cdot \phi = sum$

where $sum$ is exposed to the verifier.

In summarize, we need 1 additional virtual column for each send or receive interaction, and 1 additional virtual column to track the partial sum. These columns are all virtual in the sense that they are only materialized by the prover, after the main trace was committed, because a random challenge is needed.
