Optimal Ate Pairing
==

We support the [optimal Ate pairing](https://datatracker.ietf.org/doc/html/draft-yonezawa-pairing-friendly-curves-02#appendix-A) in OpenVM for the Bn254 and BLS12-381 curves.

For present use cases, we will only need to **assert** that the pairing is one. 
This leads to significant optimizations in ZK. We refer to this as `pairing_check` instead of `pairing`.
We implement most of the pairing check in Rust since it was found that doing certain pairing operations in circuits would cause performance issues.
We still use opcodes for Fp and Fp2 arithmetic.

Specifically this will be a pairing
$$e: \mathbb G_1 \times \mathbb G_2 \to \mathbb G_T$$
with $\mathbb G_1 = E(\mathbb F_p)[r]$ and $\mathbb G_2 = E'(\mathbb F_{p^2})[r]$ and $\mathbb G_T$ are the $r$ th roots of unity in $\mathbb F_{p^{12}}$. The primes $p, r$ will be fixed constants. The curve $E: y^2 = x^3 + b$ is a short Weierstrass curve and $E'$ is a twist of $E$ of either D-type $E': Y^2 = X^3 + b/\xi$ or M-type $E': Y^2 = X^3 + b\xi$ where $\xi \in \mathbb F_{p^2}$ is specially chosen so that $\mathbb F_{p^{12}} = \mathbb F_{p^2}[X]/(X^6 - \xi)$. 

Note: BN254 is D-type and BLS12-381 is M-type.

Jump to [Summary](#summary).

## Field extension towers

For the purposes of pairings, we need to choose a representation of the field extension $\mathbb F_{p^{12}}$. The standard way to do this is to represent $\mathbb F_{p^{12}}$ as a specific tower of field extensions of $\mathbb F_p$: $$
\begin{align}
\mathbb F_{p^2} &= \mathbb F_p[u] / (u^2 - \beta) \\
\mathbb F_{p^6} &= \mathbb F_{p^2}[v] / (v^3 - \xi) \\
\mathbb F_{p^{12}} &= \mathbb F_{p^6}[w] / (w^2 - v)
\end{align}$$ where $\beta$ is a quadratic non-residue in $\mathbb F_p$ and $\xi$ is not a quadratic residue or a cubic residue in $\mathbb F_{p^2}$. The condition on $\xi$ is equivalent to saying that the polynomial $X^6 - \xi$ is irreducible over $\mathbb F_{p^2}[X]$. 

We can set $\beta = -1$ whenever $-1$ is not a quadratic residue, which is the case when $p \equiv 3 \pmod 4$. This is true for BN254 and BLS12-381.
For BN254, we use $\xi = 9 + u$.  For [BLS12-381](https://hackmd.io/@benjaminion/bls12-381#Extension-towers), we use $\xi = 1 + u$.
The most salient point is that $\beta,\xi$ can be represented using single signed limbs instead of signed BigInts (this is important for the implementation).

The most important equations are $u^2 = -1$ and $w^6 = \xi$. 

## Twists

Recall that $X^6 - \xi$ is irreducible over $\mathbb F_{p^2}[X]$. By [Section 3, [Barreto-Naehrig]](https://eprint.iacr.org/2005/133.pdf), we can use $\xi$ to define a [D-type](https://datatracker.ietf.org/doc/html/draft-kasamatsu-bncurves-02#section-4.1) **sextic twist** of $E : y^2 = x^3 + b$ by 
$$ E' : Y^2 = X^3 + b / \xi $$
We have $w \in \mathbb F_{p^{12}}$ is a root of $X^6 - \xi$, so 
$$ \Psi(X,Y) = (w^2 X, w^3 Y) $$
defines a group homomorphism $\Psi : E'(\mathbb F_{p^2}) \to E(\mathbb F_{p^{12}})$. 

We define an M-type sextic twist by
$$ E' : Y^2 = X^3 + b \xi $$
and the twist morphism 
$$ \Psi(X,Y) = (w^{-2} X, w^{-3} Y) = (w^4 X / \xi, w^3 Y / \xi) $$
defines a group homomorphism $\Psi : E'(\mathbb F_{p^2}) \to E(\mathbb F_{p^{12}})$.

### Frobenius morphism
Define the Frobenius morphism $\phi_p : E({\mathbb F}_{p^{12}}) \to E({\mathbb F}_{p^{12}})$ by $\phi_p(x,y) = (x^p, y^p)$.
Fact: we have an alternate description of $\Psi(\mathbb G_2)$ as 
$$
\Psi(\mathbb G_2) = E(\mathbb F_{p^{12}})[r] \cap \ker(\phi_p - [p]).
$$

## Optimal Ate pairing
We only consider curves with embedding degree $12$ (this is the case for BN254 and BLS12-381). The optimal Ate pairing takes the form 
$$ e(P,Q) = f_{miller,Q}(P)^{\frac{p^{12}-1} r} $$
where $p^{12} - 1$ comes from the embedding degree.

The form of $f_{miller,Q}(P)$ is different for BN curves versus BLS curves:

For BN curves, $f_{miller,Q}(P)$ is defined by 
$$ f_{miller,Q}(P) = f_{6x + 2 + p - p^2 + p^3,Q}(P) = f_{6x+2, Q}(P) \cdot l_{[6x+2]\Psi(Q), \phi_p(\Psi(Q))}(P) \cdot l_{[6x+2]Q + \phi_p(\Psi(Q)), -\phi_p^2(\Psi(Q))}(P) $$
where 
- $x$ is the seed used to generate the curve
- $f_{i,Q}$ is computed recursively using Miller's algorithm $$
f_{i + j,Q} = f_{i,Q} \cdot f_{j,Q} \cdot \frac{l_{[i]\Psi(Q), [j]\Psi(Q)}}{v_{[i+j]\Psi(Q)}}.$$ with the base case that $f_{1,Q}$ is the identity function. Here $l_{P_1, P_2}$ denotes the equation for a line passing through $P_1, P_2$, while $v_{P}$ denotes the equation for a vertical line passing through $P$. 
- A well-known optimization due to the final exponentiation is that the final value of $e(P,Q)$ remains unchanged if we omit the division by $v_{[i+j]\Psi(Q)}$ in the equation above. **This means we can skip all computations of $v$.** Thus we omit mention of $v$ below.

For BLS curves, $f_{miller,Q}(P)$ is defined by 
$$ f_{miller,Q}(P) = f_{x, Q}(P) $$
where $x$ is the seed of the curve and $f$ is computed using Miller's algorithm as above.

The computation of pairing is broken into two parts:
- Miller loop to compute $f_{miller,Q}(P)$
- Final exponentiation by $(p^{12} - 1)/r$. 

### Applications and Important Optimizations
An important consideration to keep in mind is that the two primary use cases for optimal Ate pairing are:
- KZG commitment opening
- BLS signature verification
For these cases, there are two optimizations that the [Gnark team](https://hackmd.io/@yelhousni/emulated-pairing#Fixed-argument-pairings) has pointed out:
- Specifics around product of Miller loops: we do not compute one pairing but **two**: $e(P_1,Q_1)e(P_2,Q_2)$. For these purposes we should compute $f_{miller,Q_1}(P_1)f_{miller,Q_2}(P_2)$ and then final exponentiate **once**. We will discuss below additional optimizations around when computing a product of Miller loops simultaneously.
- Fixed-argument pairing: in KZG commitment opening, we compute $e(P_1,Q_1)e(P_2,Q_2)$ where $Q_1,Q_2$ are both known at compile time. For BLS signatures, we compute $e(P_1,Q_1)e(P_2,Q_2)$ where $Q_1$ is known at compile time. In these cases, much of the computation does not need to be done in-program and is instead a compiler constant. We go over this in more detail below.
There is one final important optimization in [Novakovic-Eagen](https://eprint.iacr.org/2024/640.pdf):
- For the applications above, we only need to assert $\prod_i e(P_i,Q_i) = 1 \in \mathbb F_{p^{12}}$.
- It is significantly cheaper to do this assertion check alone without computing the full pairing explicitly.

We will focus on the implementation of an eDSL function
```rust
fn pairing_check(P: &[EcPoint<Fp>], Q: &[EcPoint<Fp2>]) -> ();
```
that asserts that $\prod_i e(P_i, Q_i) = 1$. This does not check that the $P_i, Q_i$ are valid or in the correct subgroups but assumes that each $P_i \in \mathbb G_1$ and $Q_i \in \mathbb G_2$ and are valid canonical representations. The gnark implementations for [BN254](https://github.com/Consensys/gnark/blob/3a0fa0f316437854d56bf10a1e75811df9697f46/std/algebra/emulated/sw_bn254/pairing.go#L247) and [BLS12-381](https://github.com/Consensys/gnark/blob/3a0fa0f316437854d56bf10a1e75811df9697f46/std/algebra/emulated/sw_bls12381/pairing.go#L247) are useful references.

If $P$ or $Q$ are identity, then $e(P,Q) = 1$. We will therefore assume $P,Q$ are not identity below because identity point always requires special handling.
### Miller Loop
 $f_{i,Q}$ is computed recursively using Miller's algorithm $$
f_{i + j,Q} = f_{i,Q} \cdot f_{j,Q} \cdot \frac{l_{[i]\Psi(Q), [j]\Psi(Q)}}{v_{[i+j]\Psi(Q)}}.$$Observe that for pairing we want to compute $f_{\lambda,Q}(P)$ for a fixed $\lambda$ depending on the curve. We compute $f_{\lambda,Q}(P)$ recursively by using a pseudo-binary encoding of $\lambda$ (for BN curves there is some additional Frobenius step we don't mention here). This means we represent $\lambda = \sum_i \sigma_i 2^i$ where each $\sigma_i \in \{-1,0,1\}$. This decomposition is known at compile-time and can be optimally chosen for each curve.
- We are given $P,Q$ as inputs so we evaluate the line functions at specific input $P$.
- Here $[i]\Psi(Q) = \Psi([i]Q)$ is the twist homomorphism applied to $[i]Q$ where the latter is scalar multiplication in $\mathbb G_2$. 
- The output of line function is in $\mathbb F_{p^{12}}$, so it is 12 $\mathbb F_p$ elements.
- Due to a special property of final exponentiation, we **do not** need to calculate the $v$ vertical line denominators above. We omit the $v$ term below.

The pseudo-code to compute $f_{\lambda,Q}(P)$ without vertical lines is as follows:
Simple Version ([skip below](#Multi-Miller-Loop-Pseudo-Code) to see the final version):
1. Let $f = l_{\Psi(Q),\Psi(Q)}(P)$, assuming `pseudo_binary_encoding.last() == 1` where `pseudo_binary_encoding[i]` is $\sigma_i$ above. 
2. Set $Q_{acc} = Q$.
3. For `i` in `pseudo_binary_encoding.len() - 2 ..=0`:
	1. $f := f^2$ 
	2. Double step: $f := f \cdot l_{\Psi(Q_{acc}), \Psi(Q_{acc})}(P)$ and $Q_{acc} := [2]Q_{acc}$. 
	3. If $\sigma_i \ne 0$:
		1. $f := f \cdot l_{\Psi(Q_{acc}), \Psi([\sigma_i] Q)}(P)$ and $Q_{acc} := Q_{acc} + [\sigma_i]Q$. Here $[\sigma_i]Q = \pm Q$ should be cached ahead.
Above $Q_{acc}$ is always of the form $[x_{acc}]Q$ for some $1 < x_{acc} < r - 1$. This means $Q_{acc} \ne \pm Q$ because $x_{acc} \not\equiv \pm 1 \pmod r$ because $Q$ is not identity and has $r$-torsion. 

#### Double and Add
The [Gnark team](https://hackmd.io/@yelhousni/emulated-pairing#Miller-loop-optimizations) pointed out an optimizations:
When $\sigma_i \ne 0$, the double and add elliptic curve operations can be combined and optimized: $Q_{acc} := [2]Q_{acc} + [\sigma_i Q] = (Q_{acc} + [\sigma_i]Q) + Q_{acc}$. 

The general formula for double and add is: $[2](x_S, y_S)+(x_Q, y_Q) = (x_{(S+Q)+S}, y_{(S+Q)+S})$
$$
\begin{align*}
            \lambda_1 &= (y_S-y_Q)/(x_S - x_Q) \\
            x_{S+Q} &= \lambda_1^2 - x_S - x_Q \\
            \lambda_2 &= -\lambda_1 - 2y_S / (x_{S+Q}-x_S) \\
            x_{(S+Q)+S} &= \lambda_2^2 - x_S - x_{S+Q} \\
            y_{(S+Q)+S} &= \lambda_2(x_S - x_{(S+Q)+S}) - y_S
\end{align*}$$
This saves a computation of intermediate $y$ coordinate! Due to https://arxiv.org/abs/math/0208038 

The corresponding $f$ update becomes
- $f := f^2 \cdot l_{\Psi(Q_{acc}),\Psi([\sigma_i]Q)} \cdot l_{\Psi(Q_{acc} + [\sigma_i] Q),\Psi(Q_{acc})}$ 

#### Sparse Lines
When the twist $\Psi$ is D-type, the line functions $l_{\Psi(S),\Psi(Q)}$ can always be written in the form $$a Y + w \cdot b X + w^3 \cdot c$$ where the $w,w^3$ come form the twist $\Psi$ and $a,b,c \in \mathbb F_{p^2}$. 

When the twist $\Psi$ is M-type, the line functions $l_{\Psi(S),\Psi(Q)}$ can always be written in the form $$a Y + w^{-1} \cdot b X + w^{-3} \cdot c$$ where the $w,w^3$ come form the twist $\Psi$ and $a,b,c \in \mathbb F_{p^2}$. Note that $w^{-i} = w^{6-i} / \xi$ using the identity $w^6 = \xi$.

**WARNING**: BEWARE OF THE TWIST. Let $\tilde w = w$ for D-type and $\tilde w = w^{-1}$ for M-type.

We evaluate the point at $P = (x_P,y_P) \in (\mathbb F_p, \mathbb F_p)$.  Gnark uses the fact that division by $\mathbb F_{p^2}^\times$ does not change the result after final exponentiation. This means you can represent any line evaluated at $P$ as $$
1 + b' \frac{x_P}{y_P} \cdot \tilde w + c' \cdot \frac 1{y_P} \cdot \tilde w^3.$$

##### Sparse Line Representation

We need to discuss how to represent a sparse line and how to convert it back into an $\mathbb F_{p^{12}}$ element in the basis $1,w,\dotsc,w^5$. **The representation is different for D-type versus M-type.**

For D-type, we use
```rust!
struct UnevaluatedLine<Fp2> {
    b: Fp2,
    c: Fp2
}
```
to represent the line _function_ $(x_P, y_P) \mapsto 1 + b \frac{x_P}{y_P} w + c \frac 1{y_P} w^3$. Given a point $(x_P, y_P) \in E(\mathbb F_p)$, we can evaluate `unevaluated_line.eval(x_P, y_P)` to get 
```rust!
impl UnevaluatedLine<Fp2> {
    fn evaluate(self, x_over_y: Fp2, y_inv: Fp2) -> EvaluatedLine<Fp2> {
        EvaluatedLine {
            b: self.b * x_over_y,
            c: self.c * y_inv
        }
    }
}
struct EvaluatedLine<Fp2> {
    b: Fp2,
    c: Fp2
}
```
which represents the _point_ $1 + b w + c w^3$. Concretely we should have `Fp12::from_evaluated_line_d_type(line: EvaluatedLine)`.


For M-type, we need to do a little more work because $\tilde w = w^{-1}$. We can still represent an evaluated line as $1 + b \frac{x_P}{y_P} w^{-1} + c \frac 1{y_P} w^{-3}$ but this is not optimal for computations. Instead, we follow [gnark](https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/fields_bls12381/e12_pairing.go#L235) and use the fact that multiplication by $w^3 \in \mathbb F_{p^4}$ (proper subfield of $\mathbb F_{p^{12}}$) does not change the Miller loop result. Hence up to final exponentiation, we can still use `UnevaluatedLine` but now to represent the line function $(x_P, y_P) \mapsto w^3 + b \frac{x_P}{y_P} w^2 + c \frac 1{y_P}$. The `UnevaluatedLine::evaluate` function remains the same. What is different is the `Fp12::from_evaluated_line_m_type(line: EvaluatedLine)` function is now different. 


We can precompute $x_P/y_P, 1/y_P \in \mathbb F_p$ once at the beginning of the pairing function. This means an evaluated line can be represented using only **two** $\mathbb F_{p^2}$ elements (this is better than what we did in halo2-ecc where we used 3).
- Here we make the assumption that there is not a non-identity point in $E(\mathbb F_p)$ with $y$-coordinate zero. This is true for most curves (it corresponds to the constant term in equation not being a cube).
 
The equations for line functions from [Gnark paper](https://eprint.iacr.org/2022/1162.pdf) are reproduced here.
$$
\begin{align}
l_{\Psi(S),\Psi(S)}(P) &= 1 - \lambda \cdot \frac{x_P}{y_P}\cdot \tilde w + (\lambda x_S - y_S)\cdot \frac 1 {y_P} \cdot \tilde w^3, \quad\quad \lambda = \frac{3x_S^2}{2y_S} \\
l_{\Psi(S),\Psi(Q)}(P) &= 1 - \lambda_1 \cdot \frac{x_P}{y_P} \cdot \tilde w + (\lambda_1 x_S - y_S) \cdot \frac 1{y_P} \cdot \tilde w^3, \quad\quad \lambda_1 = \frac{y_Q - y_S}{x_Q - x_S} \\
l_{\Psi(S+Q),\Psi(S)}(P) &= 1 - \lambda_2 \cdot \frac{x_P}{y_P} \cdot \tilde w + (\lambda_2 x_S - y_S)\cdot \frac 1{y_P} \cdot \tilde w^3, \quad\quad \lambda_2 = -\lambda_1 - 2 \frac{y_S}{x_{S+Q} - x_S}
\end{align}
$$
The first line is the tangent line used in a double step. The latter two are non-tangent lines used in double-and-add step.

#### Multi-Miller Loop Pseudo-Code
```rust
// Assumes P != O, Q != O
fn multi_miller_loop(P: &[EcPoint<Fp>], Q: &[EcPoint<Fp2>], pseudo_binary_encoding: &[i32]) -> Fp12 {
	// assuming pseudo_binary_encoding.last() == 1
	let x_over_ys = // P.x/P.y ... for each P
	let y_invs = // 1/P.y ... for each P
	// if pseudo_binary_encoding has -1s, cache -Q's
	let mut res = Fp12::one();
	let mut Q_acc = Q.clone();

	for i in pseudo_binary_encoding.len() - 2 .. =0 {
		res = fp12_square(res); // skip if first iteration of loop

		if pseudo_binary_encoding[i] == 0 {
			let (Q_acc, lines) = Q_acc.map(miller_double_step);
			// Evaluate lines at different P_i's:
			for (line, x_over_y, y_inv) in (lines, x_over_ys, y_invs) {
				line = evaluate_line(line, x_over_y, y_inv);
			}
		} else {
			let Q_signed = // compile-time select Q or Q_neg based on pseudo_binary_encoding[i]
			let (Q_acc, lines0, lines1) = zip(Q_acc, Q_signed).map(miller_double_and_add_step);
			// Evaluate lines at different P_i's:
			for (line0, line1, x_over_y, y_inv) in (lines0, lines1, x_over_y, y_inv) {
				line0 = evaluate_line(line0, x_over_y, y_inv);
				line1 = evaluate_line(line1, x_over_y, y_inv);
			}
			lines = [lines0, lines1].concat();
		}
		// lines have all been evaluated at P_i's now
		if lines.len() % 2 == 1 {
			res = mul_by_013(res, lines.pop())
		}
		for (line0, line1) in lines.chunks(2) {
			prod = mul_013_by_013(line0, line1);
			res = mul_by_01234(res, prod);
		}
	}
	res
}
```

### Assert Final Exponentiation Is One
After computing $f_{miller,Q}(P)$, the optimal Ate pairing is given by $$e(P,Q) = f_{miller,Q}(P)^{(p^{12}-1)/r}$$ where the exponentiation is called the final exponentiation. 
For current applications, we only need to prove the assertion that $e(P,Q) = 1 \in \mathbb F_{p^{12}}$. This turns out to be significantly easier to prove by [Novakovic-Eagen](https://eprint.iacr.org/2024/640.pdf). 

We will provide functionality to compute the final exponentiation itself in the future, but at present we focus on the optimized AssertFinalExponentiationIsOne implementation. 

Theorem 3 of the paper says that $$\prod_i e(P_i,Q_i) = 1 \Longleftrightarrow \exists c,u \in \mathbb F_{p^{12}}^* : \prod_i f_{miller,Q_i}(P_i) = c^\lambda u \text{ and } u^d = 1$$
where $d = gcd(\lambda/r, (q^{12}-1)/r)$. (The paper seems to have a typo).

The main idea is that the values of $c,u$ can be **hinted** in the VM. Then within the VM program all we need to verify is the equality on the right. (The paper notes in some cases the $u^d = 1$ check can also be skipped).

The paper makes very clever observation that $c^\lambda$ uses the same $\lambda$ as in the multi-Miller loop. Technically $\lambda = \lambda' + \dotsc$ where there are some $p$-power terms, which need to be dealt with separately. This means that we can modify `multi_miller_loop` to a new 
```rust
fn multi_miller_loop_embedded_exp(P: &[EcPoint<Fp>], Q: &[EcPoint<Fp2>], c: Fp12, pseudo_binary_encoding: &[i32]) {
```
where in the main loop, we do an extra
```rust
	if pseudo_binary_encoding[i] == 1 {
		res = fp12_multiply(res, c)
	} else if pseudo_binary_encoding[i] == -1 {
		res = fp12_multiply(res, c_inv)
	}
```
where we compute `c_inv` once at the beginning of the function.
Some additional Frobenius operations on `c` are still necessary, but we have saved every squaring of `c` by sharing it with the squaring of `res` in the Miller loop above! [Gnark implementation](https://github.com/Consensys/gnark/blob/3a0fa0f316437854d56bf10a1e75811df9697f46/std/algebra/emulated/sw_bn254/pairing.go#L689).

To summarize the main savings are:
- only need to compute `Fp12` exponentiation by $\lambda$ once, whereas in normal final exponentiation it requires three times (though there's compression optimizations)
- squaring in the square-and-multiply exponentiation by $\lambda$ is free by sharing with Miller loop
For reference this turns a ~240-bit exponentiation into only ~30 `fp12_multiply`.

#### What if final exponentiation is not 1?

If the above approach fails, we fallback to doing the final exponentiation with square-and-multiply. This is slow.

## References
- Reference on BLS12-381: https://hackmd.io/@benjaminion/bls12-381
- Reference on BN254: https://hackmd.io/@jpw/bn254
- Gnark pairing implementation explainer: https://hackmd.io/@yelhousni/emulated-pairing
- Implementation of BN254 optimal Ate pairing in halo2-lib: https://github.com/axiom-crypto/halo2-lib/blob/main/halo2-ecc/src/bn254/pairing.rs
