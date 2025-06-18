# ECDSA Signature Verification and Recovery

We give a proof that the ECDSA public key recovery algorithm used in the `ecdsa` module automatically implies valid signature verification.

## `verify_prehashed`
We start by giving an overview of the ECDSA signature verification algorithm following the [Digital Signature Standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-5.pdf).

We describe the algorithm for `verify_prehashed`, which does not include the hashing of the signature itself.

**Inputs:**
1. Digest $H$ which _should_ be the digest of a cryptographic hash function on the message.
2. Signature consisting of pair of integers $r, s$.
3. Purported signature verification key $Q$.

The `verify_prehashed` function assumes `sig: &[u8]` is properly encoded, but the `PrehashVerifier::verify_prehash` function takes in `signature: &Signature<C>` which is a protected type ensuring correct encoding. The protected type `VerifyingKey<C>` and its inner type `PublicKey<C>` are protected to ensure $Q$ is a non-identity affine point on the elliptic curve.

**Output:** Accept or reject the signature over $H$ as originating from the owner of public key $Q$.

**Process:**
Let $n$ be the order of the elliptic curve $C$ and $L_n$ the number of bits in $n$. We assume $n$ is of prime order. Let $G$ be a fixed generator point of $C$. Let $p$ be the modulus of the coordinate field of $C$.

1. Verify that $r$ and $s$ are integers in the interval $[1, n-1]$. Reject otherwise.
2. Derive the integer $e$ from $H$ as follows:
  - If the bit length of $H$ is at most $L_n$, set $E = H$. Otherwise set $E$ equal to the leftmost $L_n$ bits of $H$.
  - Let $e$ denote the integer representation of $E$ in big endian.
3. Compute $u = e \cdot s^{-1} \mod n$ and $v = r \cdot s^{-1} \mod n$.
4. Compute $R_1 = uG + vQ$. Reject if $R_1$ is the identity.
5. Set $x_R$ to the $x$-coordinate of $R_1 = (x_R, y_R)$.
6. Convert $x_R$ to the unique integer $r_1$ in $[0, p - 1]$.
7. Accept if and only if $r = r_1 \mod n$.

## `recover_from_prehash_noverify`

We describe the ECDSA public key recovery algorithm following [SEC 1, Section 4.1.6](https://www.secg.org/sec1-v2.pdf). Like above, we describe this algorithm for `recover_from_prehash_noverify` which operates on the message digest without hashing the message.

**Inputs:**
1. Digest $H$ which _should_ be the digest of a cryptographic hash function on the message.
2. Signature consisting of pair of integers $r, s$.
3. Recovery ID `recovery_id` in the range $[0, 3]$.

**Output:** An elliptic curve public key $Q$ for which $(r, s)$ is a valid signature on digest $H$, or "invalid".

**Process:**
Let $n, L_n, G, p, C$ be as in the previous section.

1. Verify that $r$ and $s$ are integers in the interval $[1, n-1]$. Reject otherwise.
2. Let $j = 0$ if the high bit (3/4) of `recovery_id` is $0$, otherwise $j = 1$. Note that the current `RecoveryId` from the `ecdsa` crate only supports $j$ in $[0, 1]$.
3. Let $x = r + j n$ as integer. Reject if $x \geq p$.
4. Calculate the curve point $R = (x_1, y_1)$ where $x_1 = x \mod p$. Reject if no such $y_1$ exists. If more than one $y_1$ exists, then choose the unique $y_1$ such that the parity of $y_1$ as an integer in canonical form matches the low bit of `recovery_id`.
5. Compute $e$ from $H$ as in Step 2 of ECDSA signature verification above.
6. Compute $Q = r^{-1}( s R - e G)$.
7. Reject if $Q$ is the identity. Accept otherwise.

### Proof of signature verification

Above, we skip the verification of the original signature again with the recovered $Q$. We give a proof below that the recovered $Q$ will always pass the signature verification.

We refer to the steps in verification as V1-7 and the steps in recovery as R1-7.
- V1 is automatic from R1.
- V2 is the same as R5.
- V3: We compute $u = e \cdot s^{-1} \mod n$ and $v = r \cdot s^{-1} \mod n$.
- V4: compute $$R_1 = uG + vQ = e \cdot s^{-1} G + (r \cdot s^{-1})\cdot r^{-1} \cdot (s R - e G) = R.$$
R4 guarantees $R_1$ is not identity.
- V5: We set $x_R = x_1$.
- V6: Then $r_1 = r + j n$ since R3 guarantees that $r + j n$ is in the range $[0, p-1]$.
- V7: It is evident that $r_1 = r+jn$ is congruent to $r$ modulo $n$.

Therefore the signature verifies with respect to $Q$.
