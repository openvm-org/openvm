//! Graph-IR generation backend for the [`chip_traits`](crate::chip_traits) traits.
//!
//! [`Halo2IRBuilder`] implements the same chip traits as
//! [`Halo2Backend`](crate::halo2_backend::Halo2Backend), but instead of assigning halo2
//! advice cells it records a dataflow graph of [`Halo2GraphNode`]s. Each node corresponds
//! to a statically-sized slice of the halo2 witness tape (advice cells) and range-check
//! tape, so executing the nodes **in tape order** reproduces the exact witness stream the
//! halo2 backend produces. To guarantee this, the builder mirrors the concrete chips'
//! host-side bookkeeping exactly:
//!
//! - **Lazy reduction**: `max_bits` tracking follows `BabyBearChip` verbatim. Where a chip reduces
//!   an operand *before* its gate cells (add/sub/mul/mul_add), the builder emits an explicit
//!   [`Halo2Opcode::BBReduce`] node first. Where reduces are interleaved inside an op (`div`, ext
//!   `mul`/`div`), the op stays atomic and the executor re-derives the reduce decisions from the
//!   operand bit bounds stored in each [`GraphCell`].
//! - **Constant caching**: the builder keeps a `zero_cell` (mirroring `Context::load_zero`) and a
//!   BabyBear constant cache (mirroring `BabyBearChip::const_cache`), so cached constant loads
//!   insert no node/tape cells. Atomic ops that internally load cacheable constants (`BBDiv` loads
//!   ONE, `ExtMul` loads W, `ExtDiv` loads ONE/ZERO/W) expose those cells as extra node outputs;
//!   whether each constant materializes is decided **at build time** and recorded per node in
//!   [`NodeMeta::constant_skip_inds`], so executors replay nodes statelessly (and in parallel)
//!   without tracking any cache.
//! - **Per-node metadata**: every emitted node gets a [`NodeMeta`] (context/range tape offsets and
//!   lengths, constant-skip indices, operand tape offsets) derived by replaying the op on a
//!   [`CalculateOffsetsTape`](crate::halo2_opcode_impl::CalculateOffsetsTape) seeded with the
//!   builder's current cache state.
//! - **Copy constraints**: `constrain_equal` assigns no advice cells, so it produces no node.
//! - **Transcript / digest hashing**: the builder re-implements `TranscriptChip`'s sponge and
//!   buffer bookkeeping, emitting [`Halo2Opcode::PoseidonPermute2T3`] /
//!   [`Halo2Opcode::InnerProduct`] (base-2^31 packing) / [`Halo2Opcode::DecomposeBn254ToBabyBear`]
//!   / [`Halo2Opcode::RangeDiv`] nodes.
//!
//! Constants that halo2 assigns as *fixed-column* `QuantumCell::Constant`s (select branch
//! values, inner-product coefficients, the value of a `Const` witness cell) are not node
//! outputs; they appear as [`GraphCell::Const`] operands and write no advice cells.
//!
//! A few opcodes beyond the core arithmetic set are required to cover the full populate
//! pipeline: witness loading (`LoadWitness`), inner products (`InnerProduct`), and the
//! transcript hint decompositions (`DecomposeBn254ToBabyBear`, `RangeDiv`).

use core::{array, iter};
use std::collections::{HashMap, HashSet};

use halo2_base::{
    halo2_proofs::{
        arithmetic::Field as _,
        halo2curves::{bn256::Fr, ff::PrimeField as _},
    },
    utils::{bit_length, fe_to_biguint},
};
use itertools::Itertools;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, PrimeField64,
    },
    p3_baby_bear::BabyBear,
};

use crate::{
    chip_traits::{
        BabyBearExt4Inst, BabyBearInst, ChipBase, DigestHashInst, GateInst, PopulateInputs,
        TranscriptInst,
    },
    field::baby_bear::{
        BabyBearExt4, BabyBearExt4Wire, BabyBearWire, ReducedBabyBearExt4Wire, ReducedBabyBearWire,
        BABYBEAR_MAX_BITS, BABY_BEAR_MODULUS_U64, RESERVED_HIGH_BITS,
    },
    halo2_opcode_impl::{derive_opcode_metadata, UNMATERIALIZED},
    hash::{
        poseidon2::{MULTI_FIELD32_NUM_F_ELMS, MULTI_FIELD32_RATE, POSEIDON2_RATE},
        POSEIDON2_WIDTH,
    },
    transcript::{DigestWire, NUM_OBS_PER_WORD, NUM_SAMPLES_PER_WORD},
};

/// Unique id of an IR node; also its index in [`Halo2IRBuilder::nodes`].
pub type NodeId = u32;

/// Reduce when a bound would exceed this, mirroring `BabyBearChip`.
const REDUCE_THRESHOLD: usize = Fr::CAPACITY as usize - RESERVED_HIGH_BITS;
/// Bit bound for raw Bn254 cells (digests, sponge state, packed words).
const RAW_MAX_BITS: usize = Fr::NUM_BITS as usize;

const _: () = assert!(POSEIDON2_WIDTH == 3);

/// An IR operand/result: either a value produced by a node, or a fixed-column constant.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GraphCell {
    /// `(node id, absolute context-tape offset the cell is written at, bits bound)`.
    ///
    /// The bits bound is the `max_bits` invariant of [`BabyBearWire`]: the (signed) value
    /// is guaranteed `< 2^bits`. Executors use operand bit bounds to replay the chips'
    /// internal reduce decisions inside atomic ops.
    Cell(NodeId, usize, u16),
    /// A constant operand assigned as a halo2 fixed-column `QuantumCell::Constant`;
    /// writes no advice cells.
    Const(Fr),
}

impl GraphCell {
    pub fn bits(&self) -> usize {
        match self {
            Self::Cell(_, _, bits) => *bits as usize,
            Self::Const(value) => fr_bits(value),
        }
    }
}

/// IR opcodes. Each op writes a statically-known slice of the witness tape and of the
/// range-check tape when executed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Halo2Opcode {
    // -- gate ops --
    /// One constant advice cell (`ctx.load_constant` / `ctx.load_zero`); the single
    /// operand is the [`GraphCell::Const`] holding the value.
    Const,
    /// `if cond { a } else { b }` on raw cells; operands `[a, b, cond]` (`a`/`b` may be
    /// [`GraphCell::Const`], covering `select_const`).
    Select,
    /// Little-endian bit decomposition of the operand into `n` bit cells.
    Num2Bits(u16),
    // -- babybear ops --
    /// Barrett-style signed reduction of the operand to `[0, p)`; the operand's bit bound
    /// determines the quotient range-check width.
    BBReduce,
    BBAdd,
    BBNeg,
    BBSub,
    BBMul,
    BBMulAdd,
    /// Atomic BabyBear division `a / b` mirroring `BabyBearChip::div` (inverse hint,
    /// non-zero check, `a = b*c` check, internal reduces). Output 0 is the quotient;
    /// output 1 is the internally loaded ONE constant cell (materialized only when ONE was
    /// not already cached).
    BBDiv,
    /// Asserts the operand is `0 mod p` (exact-quotient hint + range check).
    BBAssertZero,
    // -- extension ops --
    /// Atomic `BabyBearExt4Chip::mul`; operands `[a0..a3, b0..b3]`. Outputs 0-3 are the
    /// product coefficients; output 4 is the internally loaded W constant cell
    /// (materialized only when W was not already cached).
    ExtMul,
    /// Atomic `BabyBearExt4Chip::div`; operands `[a0..a3, b0..b3]`. Outputs 0-3 are the
    /// quotient coefficients; outputs 4-6 are the internally loaded ONE/ZERO/W constant
    /// cells (each materialized only when not already cached).
    ExtDiv,
    // -- poseidon ops --
    /// Width-2 Poseidon2 permutation (digest compression).
    PoseidonPermute2T2,
    /// Width-3 Poseidon2 permutation (transcript sponge, digest hashing).
    PoseidonPermute2T3,
    // -- ops beyond the core set, required to cover the full populate pipeline --
    /// One proof-input advice cell; the value comes from the builder's input stream.
    LoadWitness,
    /// One proof-input advice cell constrained to `[0, p)` (`check_less_than_safe`).
    CheckLessThanSafe,
    /// Inner product of `n` `(value, coefficient)` operand pairs, interleaved
    /// `[v0, c0, v1, c1, ..]`; coefficients are [`GraphCell::Const`] in practice
    /// (`inner_product_const` and base-2^31 transcript/digest packing).
    InnerProduct(u16),
    /// Base-BabyBear hint decomposition of one squeezed Bn254 word into
    /// [`NUM_SAMPLES_PER_WORD`] digit cells (plus internal top-quotient hint and
    /// boundary checks), mirroring `decompose_bn254_to_base_baby_bear_digits`.
    DecomposeBn254ToBabyBear,
    /// `rem` of `range.div_mod(operand, 2^n)`, mirroring `TranscriptChip::sample_bits`.
    RangeDiv(u16),
}

impl Halo2Opcode {
    pub fn num_operands(&self) -> usize {
        match self {
            Self::LoadWitness => 0,
            Self::CheckLessThanSafe => 1,
            Self::Const
            | Self::Num2Bits(_)
            | Self::BBReduce
            | Self::BBNeg
            | Self::BBAssertZero
            | Self::DecomposeBn254ToBabyBear
            | Self::RangeDiv(_) => 1,
            Self::BBAdd | Self::BBSub | Self::BBMul | Self::BBDiv | Self::PoseidonPermute2T2 => 2,
            Self::Select | Self::BBMulAdd | Self::PoseidonPermute2T3 => 3,
            Self::ExtMul | Self::ExtDiv => 8,
            Self::InnerProduct(n) => 2 * *n as usize,
        }
    }

    pub fn num_results(&self) -> usize {
        match self {
            Self::BBAssertZero | Self::CheckLessThanSafe => 0,
            Self::Const
            | Self::Select
            | Self::BBReduce
            | Self::BBAdd
            | Self::BBNeg
            | Self::BBSub
            | Self::BBMul
            | Self::BBMulAdd
            | Self::LoadWitness
            | Self::InnerProduct(_)
            | Self::RangeDiv(_) => 1,
            Self::BBDiv | Self::PoseidonPermute2T2 => 2,
            Self::PoseidonPermute2T3 => 3,
            Self::ExtMul => 5,
            Self::DecomposeBn254ToBabyBear => NUM_SAMPLES_PER_WORD,
            Self::ExtDiv => 7,
            Self::Num2Bits(n) => *n as usize,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Const => "Const",
            Self::Select => "Select",
            Self::Num2Bits(_) => "Num2Bits",
            Self::BBReduce => "BBReduce",
            Self::BBAdd => "BBAdd",
            Self::BBNeg => "BBNeg",
            Self::BBSub => "BBSub",
            Self::BBMul => "BBMul",
            Self::BBMulAdd => "BBMulAdd",
            Self::BBDiv => "BBDiv",
            Self::BBAssertZero => "BBAssertZero",
            Self::ExtMul => "ExtMul",
            Self::ExtDiv => "ExtDiv",
            Self::PoseidonPermute2T2 => "PoseidonPermute2T2",
            Self::PoseidonPermute2T3 => "PoseidonPermute2T3",
            Self::LoadWitness => "LoadWitness",
            Self::CheckLessThanSafe => "CheckLessThanSafe",
            Self::InnerProduct(_) => "InnerProduct",
            Self::DecomposeBn254ToBabyBear => "DecomposeBn254ToBabyBear",
            Self::RangeDiv(_) => "RangeDiv",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Halo2GraphNode {
    pub opcode: Halo2Opcode,
    pub operands: Vec<GraphCell>,
    pub id: NodeId,
}

/// Per-node replay metadata deduced at build time, indexed by [`NodeId`]
/// (see [`Halo2IRBuilder::node_meta`]).
///
/// Together with the operand values this is everything an executor needs to
/// replay a node in isolation: no constant cache is consulted at runtime, so
/// nodes can be interpreted in parallel.
#[derive(Clone, Debug)]
pub struct NodeMeta {
    /// Number of context-tape (advice) slots the node writes.
    pub ctx_len: usize,
    /// Absolute context-tape offset the node begins writing at.
    pub ctx_offset: usize,
    /// Number of range-tape (lookup) slots the node writes.
    pub lookups_len: usize,
    /// Absolute range-tape offset the node begins writing at.
    pub lookup_offset: usize,
    /// Whether the node writes any constant cells, i.e. some internal
    /// `load_constant` call missed the chips' caches at build time.
    pub requires_constant_skip: bool,
    /// Indices of the node's `load_constant` calls that write a cell, in call
    /// order (e.g. `[1, 3]` when the second and fourth of five calls missed the
    /// cache); calls not listed hit a cache and must not write. Drives
    /// `WitnessTape` at replay time.
    pub constant_skip_inds: Vec<u32>,
    /// Absolute context-tape offset of each operand (`UNMATERIALIZED` for
    /// [`GraphCell::Const`] operands, whose values live in the node's operand
    /// list, not the context tape).
    pub arg_offsets: Vec<usize>,
}

/// Transcript sponge/buffer state, mirroring `TranscriptChip`.
#[derive(Clone, Debug)]
struct IrTranscript {
    sponge_state: [GraphCell; POSEIDON2_WIDTH],
    absorb_idx: usize,
    sample_idx: usize,
    observe_buf: Vec<ReducedBabyBearWire<GraphCell>>,
    sample_buf: Vec<BabyBearWire<GraphCell>>,
}

/// Backend that records the circuit-population trace as a graph IR.
pub struct Halo2IRBuilder {
    /// Node tape, in exact witness-tape order.
    pub nodes: Vec<Halo2GraphNode>,
    /// Replay metadata of node `i`, in node-tape order.
    pub node_meta: Vec<NodeMeta>,
    /// Proof-input witness stream, one entry per `LoadWitness` node, in
    /// node-tape order.
    pub input_values: Vec<Fr>,
    /// Range-check lookup bits of the target halo2 circuit (drives limb
    /// decompositions, so it is part of the tape shape).
    lookup_bits: usize,
    /// Context-tape (advice) write cursor; the next node's `ctx_offset`.
    ctx_offset: usize,
    /// Range-tape (lookup) write cursor; the next node's `lookup_offset`.
    lookup_offset: usize,
    /// `levels[i]` = dataflow depth of node `i` (0 for source nodes).
    levels: Vec<u32>,
    /// Mirrors `Context::zero_cell` caching in `Context::load_zero`.
    zero_cell: Option<GraphCell>,
    /// Mirrors `BabyBearChip::const_cache` (keyed by canonical u64).
    bb_const_cache: HashMap<u64, BabyBearWire<GraphCell>>,
    /// Constants currently materialized in the mirrored caches (`zero_cell` +
    /// `bb_const_cache`); in-op `load_constant` calls hit these without
    /// assigning a cell.
    warm_consts: HashSet<Fr>,
    transcript: Option<IrTranscript>,
}

fn fr_bits(value: &Fr) -> usize {
    fe_to_biguint(value).bits() as usize
}

fn bb_wire(value: GraphCell, max_bits: usize) -> BabyBearWire<GraphCell> {
    BabyBearWire { value, max_bits }
}

impl Halo2IRBuilder {
    pub fn new(lookup_bits: usize) -> Self {
        Halo2IRBuilder {
            nodes: Vec::new(),
            node_meta: Vec::new(),
            input_values: Vec::new(),
            lookup_bits,
            ctx_offset: 0,
            lookup_offset: 0,
            levels: Vec::new(),
            zero_cell: None,
            bb_const_cache: HashMap::new(),
            warm_consts: HashSet::new(),
            transcript: None,
        }
    }

    /// Total number of context-tape (advice) cells written by all nodes.
    pub fn total_ctx_len(&self) -> usize {
        self.ctx_offset
    }

    /// Total number of range-tape (lookup) cells written by all nodes.
    pub fn total_lookups_len(&self) -> usize {
        self.lookup_offset
    }

    /// Range-check lookup bits the IR was built for.
    pub fn lookup_bits(&self) -> usize {
        self.lookup_bits
    }

    /// Dataflow depth of each node (0 for source nodes), indexed by [`NodeId`].
    pub fn node_levels(&self) -> &[u32] {
        &self.levels
    }

    /// Emits a node and deduces its [`NodeMeta`] by replaying the op against the
    /// current cache state. Returns the node id and the **absolute** context-tape
    /// offset of each logical output ([`UNMATERIALIZED`] for constants that hit a
    /// cache and assigned no cell).
    fn emit(&mut self, opcode: Halo2Opcode, operands: Vec<GraphCell>) -> (NodeId, Vec<usize>) {
        debug_assert_eq!(operands.len(), opcode.num_operands());
        let id = self.nodes.len() as NodeId;
        let level = operands
            .iter()
            .filter_map(|cell| match cell {
                GraphCell::Cell(node, _, _) => Some(self.levels[*node as usize] + 1),
                GraphCell::Const(_) => None,
            })
            .max()
            .unwrap_or(0);
        self.levels.push(level);

        // Tape shape depends only on constant operand values, bit bounds,
        // lookup_bits, and cache state — a nonzero dummy stands in for cell values.
        let args: Vec<Fr> = operands
            .iter()
            .map(|cell| match cell {
                GraphCell::Cell(..) => Fr::ONE,
                GraphCell::Const(value) => *value,
            })
            .collect();
        let bits: Vec<u16> = operands.iter().map(|cell| cell.bits() as u16).collect();
        let meta =
            derive_opcode_metadata(&opcode, &args, &bits, self.lookup_bits, &self.warm_consts);

        let ctx_offset = self.ctx_offset;
        let output_offsets: Vec<usize> = meta
            .output_offsets
            .iter()
            .map(|&rel| {
                if rel == UNMATERIALIZED {
                    UNMATERIALIZED
                } else {
                    ctx_offset + rel
                }
            })
            .collect();
        let arg_offsets = operands
            .iter()
            .map(|cell| match cell {
                GraphCell::Cell(_, offset, _) => *offset,
                GraphCell::Const(_) => UNMATERIALIZED,
            })
            .collect();
        self.node_meta.push(NodeMeta {
            ctx_len: meta.ctx_len,
            ctx_offset,
            lookups_len: meta.lookups_len,
            lookup_offset: self.lookup_offset,
            requires_constant_skip: !meta.constant_skip_inds.is_empty(),
            constant_skip_inds: meta.constant_skip_inds,
            arg_offsets,
        });
        self.ctx_offset += meta.ctx_len;
        self.lookup_offset += meta.lookups_len;

        self.nodes.push(Halo2GraphNode {
            opcode,
            operands,
            id,
        });
        (id, output_offsets)
    }

    fn emit1(&mut self, opcode: Halo2Opcode, operands: Vec<GraphCell>, bits: usize) -> GraphCell {
        let (id, offsets) = self.emit(opcode, operands);
        GraphCell::Cell(id, offsets[0], bits as u16)
    }

    /// Mirrors `Context::load_zero`.
    fn load_zero(&mut self) -> GraphCell {
        if let Some(zero) = self.zero_cell {
            return zero;
        }
        let zero = self.emit1(Halo2Opcode::Const, vec![GraphCell::Const(Fr::ZERO)], 0);
        self.zero_cell = Some(zero);
        self.warm_consts.insert(Fr::ZERO);
        zero
    }

    /// Records the cache effect of a BabyBear constant loaded *inside* an atomic node
    /// (`BBDiv`/`ExtMul`/`ExtDiv`), whose cell sits at absolute context-tape `offset`
    /// when it materializes. Mirrors `BabyBearChip::load_constant` cache/zero-cell
    /// behavior.
    fn note_internal_bb_const(&mut self, id: NodeId, offset: usize, value: BabyBear) {
        let key = value.as_canonical_u64();
        if self.bb_const_cache.contains_key(&key) {
            return;
        }
        let max_bits = bit_length(key);
        let cell = if key == 0 {
            match self.zero_cell {
                Some(zero) => zero,
                None => {
                    debug_assert_ne!(offset, UNMATERIALIZED);
                    let zero = GraphCell::Cell(id, offset, 0);
                    self.zero_cell = Some(zero);
                    zero
                }
            }
        } else {
            debug_assert_ne!(offset, UNMATERIALIZED);
            GraphCell::Cell(id, offset, max_bits as u16)
        };
        self.bb_const_cache.insert(key, bb_wire(cell, max_bits));
        self.warm_consts.insert(Fr::from(key));
    }

    fn bb_reduce_wire(&mut self, a: BabyBearWire<GraphCell>) -> BabyBearWire<GraphCell> {
        assert!(a.max_bits <= REDUCE_THRESHOLD);
        let cell = self.emit1(Halo2Opcode::BBReduce, vec![a.value], BABYBEAR_MAX_BITS);
        bb_wire(cell, BABYBEAR_MAX_BITS)
    }

    fn inner_product(&mut self, values: &[GraphCell], coeffs: &[Fr]) -> GraphCell {
        assert_eq!(values.len(), coeffs.len());
        let operands = values
            .iter()
            .zip(coeffs)
            .flat_map(|(&value, &coeff)| [value, GraphCell::Const(coeff)])
            .collect_vec();
        self.emit1(
            Halo2Opcode::InnerProduct(values.len() as u16),
            operands,
            RAW_MAX_BITS,
        )
    }

    /// Base-2^31 packing of reduced BabyBear wires, mirroring `pack_base_2_31_cells`.
    fn pack_base_2_31(&mut self, values: &[ReducedBabyBearWire<GraphCell>]) -> GraphCell {
        assert!(values.len() <= MULTI_FIELD32_NUM_F_ELMS);
        let base = Fr::from(1u64 << 31);
        let coeffs = iter::successors(Some(Fr::ONE), |power| Some(*power * base))
            .take(values.len())
            .collect_vec();
        let operands = values.iter().map(|value| value.value()).collect_vec();
        self.inner_product(&operands, &coeffs)
    }

    fn permute_t3(&mut self, state: &mut [GraphCell; POSEIDON2_WIDTH]) {
        let (id, offsets) = self.emit(Halo2Opcode::PoseidonPermute2T3, state.to_vec());
        *state = array::from_fn(|i| GraphCell::Cell(id, offsets[i], RAW_MAX_BITS as u16));
    }

    // --- transcript internals mirroring `TranscriptChip` ---

    fn take_transcript(&mut self) -> IrTranscript {
        self.transcript
            .take()
            .expect("transcript not initialized; call init_transcript first")
    }

    fn sponge_absorb(&mut self, t: &mut IrTranscript, value: GraphCell) {
        t.sponge_state[t.absorb_idx] = value;
        t.absorb_idx += 1;
        if t.absorb_idx == POSEIDON2_RATE {
            self.permute_t3(&mut t.sponge_state);
            t.absorb_idx = 0;
            t.sample_idx = POSEIDON2_RATE;
        }
    }

    fn sponge_squeeze(&mut self, t: &mut IrTranscript) -> GraphCell {
        if t.absorb_idx != 0 || t.sample_idx == 0 {
            self.permute_t3(&mut t.sponge_state);
            t.absorb_idx = 0;
            t.sample_idx = POSEIDON2_RATE;
        }
        t.sample_idx -= 1;
        t.sponge_state[t.sample_idx]
    }

    fn flush_observe_buf(&mut self, t: &mut IrTranscript) {
        if !t.observe_buf.is_empty() {
            let packed = self.pack_base_2_31(&t.observe_buf);
            self.sponge_absorb(t, packed);
            t.observe_buf.clear();
        }
    }

    fn observe_inner(&mut self, t: &mut IrTranscript, value: &ReducedBabyBearWire<GraphCell>) {
        t.sample_buf.clear();
        t.observe_buf.push(*value);
        if t.observe_buf.len() == NUM_OBS_PER_WORD {
            self.flush_observe_buf(t);
        }
    }

    fn sample_inner(&mut self, t: &mut IrTranscript) -> BabyBearWire<GraphCell> {
        if let Some(val) = t.sample_buf.pop() {
            return val;
        }
        self.flush_observe_buf(t);
        let squeezed = self.sponge_squeeze(t);
        let (id, offsets) = self.emit(Halo2Opcode::DecomposeBn254ToBabyBear, vec![squeezed]);
        t.sample_buf = (0..NUM_SAMPLES_PER_WORD)
            .map(|i| {
                bb_wire(
                    GraphCell::Cell(id, offsets[i], BABYBEAR_MAX_BITS as u16),
                    BABYBEAR_MAX_BITS,
                )
            })
            .collect();
        // Reverse so pop() returns digits in order (b_0 first).
        t.sample_buf.reverse();
        t.sample_buf.pop().expect("sample_buf should be non-empty")
    }

    fn sample_bits_inner(&mut self, t: &mut IrTranscript, bits: usize) -> GraphCell {
        assert!(
            bits < (u32::BITS as usize),
            "sample_bits requires bits < 32: {bits}"
        );
        assert!(
            (1u64 << bits) < BABY_BEAR_MODULUS_U64,
            "sample_bits requires (1 << bits) < modulus: bits={bits}"
        );
        let sampled = self.sample_inner(t);
        if bits == 0 {
            return self.load_zero();
        }
        self.emit1(
            Halo2Opcode::RangeDiv(bits as u16),
            vec![sampled.value],
            bits,
        )
    }

    // --- graph statistics ---

    pub fn stats(&self) -> IrStats {
        let mut level_widths: Vec<usize> = Vec::new();
        for &level in &self.levels {
            let level = level as usize;
            if level >= level_widths.len() {
                level_widths.resize(level + 1, 0);
            }
            level_widths[level] += 1;
        }
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for node in &self.nodes {
            *counts.entry(node.opcode.name()).or_default() += 1;
        }
        let mut opcode_counts = counts.into_iter().collect_vec();
        opcode_counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let num_levels = level_widths.len();
        IrStats {
            num_nodes: self.nodes.len(),
            num_inputs: self.input_values.len(),
            num_levels,
            max_width: level_widths.iter().copied().max().unwrap_or(0),
            avg_width: self.nodes.len() as f64 / num_levels.max(1) as f64,
            opcode_counts,
            level_widths,
        }
    }
}

/// Structural statistics of the generated IR graph.
#[derive(Clone, Debug)]
pub struct IrStats {
    pub num_nodes: usize,
    pub num_inputs: usize,
    /// Maximum dataflow depth (number of levels).
    pub num_levels: usize,
    /// Maximum number of nodes on any single level.
    pub max_width: usize,
    /// Average number of nodes per level.
    pub avg_width: f64,
    /// Opcode distribution, sorted by descending count.
    pub opcode_counts: Vec<(&'static str, usize)>,
    /// Number of nodes at each level.
    pub level_widths: Vec<usize>,
}

impl core::fmt::Display for IrStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "nodes: {}", self.num_nodes)?;
        writeln!(f, "proof-input witnesses: {}", self.num_inputs)?;
        writeln!(f, "max depth (levels): {}", self.num_levels)?;
        writeln!(f, "max width: {}", self.max_width)?;
        writeln!(f, "avg width per level: {:.1}", self.avg_width)?;
        writeln!(f, "opcode distribution:")?;
        for (name, count) in &self.opcode_counts {
            writeln!(
                f,
                "  {name:<22} {count:>10} ({:.2}%)",
                100.0 * *count as f64 / self.num_nodes as f64
            )?;
        }
        Ok(())
    }
}

// --- static bit-bound simulation of atomic extension ops ---

fn w_bits() -> usize {
    bit_length(<BabyBear as BinomiallyExtendable<4>>::W.as_canonical_u64())
}

/// Mirrors the `max_bits` and reduce logic of `BabyBearChip::mul_add`.
fn mul_add_result_bits(mut a: usize, mut b: usize, mut c: usize) -> usize {
    if a < b {
        core::mem::swap(&mut a, &mut b);
    }
    if a + b + 1 > REDUCE_THRESHOLD {
        a = BABYBEAR_MAX_BITS;
        if a + b + 1 > REDUCE_THRESHOLD {
            b = BABYBEAR_MAX_BITS;
        }
    }
    if c + 1 > REDUCE_THRESHOLD {
        c = BABYBEAR_MAX_BITS;
    }
    c.max(a + b) + 1
}

/// Mirrors the `max_bits` and reduce logic of `BabyBearChip::special_inner_product`,
/// mutating the bit bounds like the chip mutates the wires.
fn special_inner_product_bits(a: &mut [usize; 4], b: &mut [usize; 4], s: usize) -> usize {
    let mut max_bits = 0;
    let lb = s.saturating_sub(3);
    let ub = 4.min(s + 1);
    let len = if s < 3 { s + 1 } else { 7 - s };
    for (i, (ci, di)) in (lb..ub).zip(((s + 1 - ub)..(s + 1 - lb)).rev()).enumerate() {
        let limit = REDUCE_THRESHOLD - len + i;
        if a[ci] + b[di] > limit {
            if a[ci] >= b[di] {
                a[ci] = BABYBEAR_MAX_BITS;
                if a[ci] + b[di] > limit {
                    b[di] = BABYBEAR_MAX_BITS;
                }
            } else {
                b[di] = BABYBEAR_MAX_BITS;
                if a[ci] + b[di] > limit {
                    a[ci] = BABYBEAR_MAX_BITS;
                }
            }
        }
        max_bits = if i == 0 {
            a[ci] + b[di]
        } else {
            max_bits.max(a[ci] + b[di]) + 1
        };
    }
    max_bits
}

/// Mirrors the `max_bits` bookkeeping of `BabyBearExt4Chip::mul`.
fn ext_mul_result_bits(mut a: [usize; 4], mut b: [usize; 4]) -> [usize; 4] {
    let mut coeffs = Vec::with_capacity(7);
    for s in 0..7 {
        coeffs.push(special_inner_product_bits(&mut a, &mut b, s));
    }
    let w = w_bits();
    for i in 4..7 {
        coeffs[i - 4] = mul_add_result_bits(coeffs[i], w, coeffs[i - 4]);
    }
    [coeffs[0], coeffs[1], coeffs[2], coeffs[3]]
}

impl ChipBase for Halo2IRBuilder {
    type F = GraphCell;
}

impl PopulateInputs for Halo2IRBuilder {
    fn load_witness(&mut self, value: Fr) -> GraphCell {
        self.input_values.push(value);
        self.emit1(Halo2Opcode::LoadWitness, vec![], RAW_MAX_BITS)
    }

    fn bb_load_reduced_witness(&mut self, value: BabyBear) -> ReducedBabyBearWire<GraphCell> {
        self.input_values.push(Fr::from(value.as_canonical_u64()));
        let cell = self.emit1(Halo2Opcode::LoadWitness, vec![], BABYBEAR_MAX_BITS);
        self.emit(Halo2Opcode::CheckLessThanSafe, vec![cell]);
        ReducedBabyBearWire::assume_reduced(bb_wire(cell, BABYBEAR_MAX_BITS))
    }

    fn ext_load_reduced_witness(
        &mut self,
        value: BabyBearExt4,
    ) -> ReducedBabyBearExt4Wire<GraphCell> {
        let coeffs = value.as_basis_coefficients_slice();
        ReducedBabyBearExt4Wire::assume_reduced(array::from_fn(|i| {
            self.bb_load_reduced_witness(coeffs[i])
        }))
    }
}

impl GateInst for Halo2IRBuilder {
    /// Mirrors `ctx.load_constant`: always assigns a fresh cell (no cache).
    fn load_constant(&mut self, value: Fr) -> GraphCell {
        let bits = fr_bits(&value);
        self.emit1(Halo2Opcode::Const, vec![GraphCell::Const(value)], bits)
    }

    /// Copy constraint only; assigns no advice cells, so no node is emitted.
    fn constrain_equal(&mut self, _a: GraphCell, _b: GraphCell) {}

    fn select(
        &mut self,
        when_true: GraphCell,
        when_false: GraphCell,
        cond: GraphCell,
    ) -> GraphCell {
        let bits = when_true.bits().max(when_false.bits());
        self.emit1(Halo2Opcode::Select, vec![when_true, when_false, cond], bits)
    }

    fn select_const(&mut self, when_true: Fr, when_false: Fr, cond: GraphCell) -> GraphCell {
        self.select(
            GraphCell::Const(when_true),
            GraphCell::Const(when_false),
            cond,
        )
    }

    fn num_to_bits(&mut self, a: GraphCell, range_bits: usize) -> Vec<GraphCell> {
        let (id, offsets) = self.emit(Halo2Opcode::Num2Bits(range_bits as u16), vec![a]);
        (0..range_bits)
            .map(|i| GraphCell::Cell(id, offsets[i], 1))
            .collect()
    }

    fn inner_product_const(&mut self, values: &[GraphCell], coeffs: &[Fr]) -> GraphCell {
        self.inner_product(values, coeffs)
    }

    fn cell_count(&self) -> usize {
        self.nodes.len()
    }
}

impl BabyBearInst for Halo2IRBuilder {
    /// Mirrors `BabyBearChip::load_constant` (const cache + `load_zero` for zero).
    fn bb_load_constant(&mut self, value: BabyBear) -> BabyBearWire<GraphCell> {
        let key = value.as_canonical_u64();
        if let Some(&cached) = self.bb_const_cache.get(&key) {
            return cached;
        }
        let max_bits = bit_length(key);
        let cell = if key == 0 {
            self.load_zero()
        } else {
            self.emit1(
                Halo2Opcode::Const,
                vec![GraphCell::Const(Fr::from(key))],
                max_bits,
            )
        };
        let wire = bb_wire(cell, max_bits);
        self.bb_const_cache.insert(key, wire);
        self.warm_consts.insert(Fr::from(key));
        wire
    }

    fn bb_load_reduced_constant(&mut self, value: BabyBear) -> ReducedBabyBearWire<GraphCell> {
        // Constants are canonical by construction.
        ReducedBabyBearWire::assume_reduced(self.bb_load_constant(value))
    }

    fn bb_reduce(&mut self, a: BabyBearWire<GraphCell>) -> BabyBearWire<GraphCell> {
        self.bb_reduce_wire(a)
    }

    fn bb_reduce_max_bits(&mut self, a: BabyBearWire<GraphCell>) -> BabyBearWire<GraphCell> {
        if a.max_bits > BABYBEAR_MAX_BITS {
            self.bb_reduce_wire(a)
        } else {
            a
        }
    }

    fn bb_add(
        &mut self,
        mut a: BabyBearWire<GraphCell>,
        mut b: BabyBearWire<GraphCell>,
    ) -> BabyBearWire<GraphCell> {
        if a.max_bits + 1 > REDUCE_THRESHOLD {
            a = self.bb_reduce_wire(a);
        }
        if b.max_bits + 1 > REDUCE_THRESHOLD {
            b = self.bb_reduce_wire(b);
        }
        let max_bits = a.max_bits.max(b.max_bits) + 1;
        let cell = self.emit1(Halo2Opcode::BBAdd, vec![a.value, b.value], max_bits);
        bb_wire(cell, max_bits)
    }

    fn bb_neg(&mut self, a: BabyBearWire<GraphCell>) -> BabyBearWire<GraphCell> {
        let cell = self.emit1(Halo2Opcode::BBNeg, vec![a.value], a.max_bits);
        bb_wire(cell, a.max_bits)
    }

    fn bb_sub(
        &mut self,
        mut a: BabyBearWire<GraphCell>,
        mut b: BabyBearWire<GraphCell>,
    ) -> BabyBearWire<GraphCell> {
        if a.max_bits + 1 > REDUCE_THRESHOLD {
            a = self.bb_reduce_wire(a);
        }
        if b.max_bits + 1 > REDUCE_THRESHOLD {
            b = self.bb_reduce_wire(b);
        }
        let max_bits = a.max_bits.max(b.max_bits) + 1;
        let cell = self.emit1(Halo2Opcode::BBSub, vec![a.value, b.value], max_bits);
        bb_wire(cell, max_bits)
    }

    fn bb_mul(
        &mut self,
        mut a: BabyBearWire<GraphCell>,
        mut b: BabyBearWire<GraphCell>,
    ) -> BabyBearWire<GraphCell> {
        if a.max_bits < b.max_bits {
            core::mem::swap(&mut a, &mut b);
        }
        if a.max_bits + b.max_bits > REDUCE_THRESHOLD {
            a = self.bb_reduce_wire(a);
            if a.max_bits + b.max_bits > REDUCE_THRESHOLD {
                b = self.bb_reduce_wire(b);
            }
        }
        let max_bits = a.max_bits + b.max_bits;
        let cell = self.emit1(Halo2Opcode::BBMul, vec![a.value, b.value], max_bits);
        bb_wire(cell, max_bits)
    }

    fn bb_mul_add(
        &mut self,
        mut a: BabyBearWire<GraphCell>,
        mut b: BabyBearWire<GraphCell>,
        mut c: BabyBearWire<GraphCell>,
    ) -> BabyBearWire<GraphCell> {
        if a.max_bits < b.max_bits {
            core::mem::swap(&mut a, &mut b);
        }
        if a.max_bits + b.max_bits + 1 > REDUCE_THRESHOLD {
            a = self.bb_reduce_wire(a);
            if a.max_bits + b.max_bits + 1 > REDUCE_THRESHOLD {
                b = self.bb_reduce_wire(b);
            }
        }
        if c.max_bits + 1 > REDUCE_THRESHOLD {
            c = self.bb_reduce_wire(c);
        }
        let max_bits = c.max_bits.max(a.max_bits + b.max_bits) + 1;
        let cell = self.emit1(
            Halo2Opcode::BBMulAdd,
            vec![a.value, b.value, c.value],
            max_bits,
        );
        bb_wire(cell, max_bits)
    }

    fn bb_div(
        &mut self,
        a: BabyBearWire<GraphCell>,
        b: BabyBearWire<GraphCell>,
    ) -> BabyBearWire<GraphCell> {
        let (id, offsets) = self.emit(Halo2Opcode::BBDiv, vec![a.value, b.value]);
        // `BabyBearChip::div` internally loads the ONE constant (output 1 when uncached).
        self.note_internal_bb_const(id, offsets[1], BabyBear::new(1));
        bb_wire(
            GraphCell::Cell(id, offsets[0], BABYBEAR_MAX_BITS as u16),
            BABYBEAR_MAX_BITS,
        )
    }

    fn bb_assert_zero(&mut self, a: BabyBearWire<GraphCell>) {
        assert!(a.max_bits <= REDUCE_THRESHOLD);
        self.emit(Halo2Opcode::BBAssertZero, vec![a.value]);
    }

    fn bb_assert_equal(&mut self, a: BabyBearWire<GraphCell>, b: BabyBearWire<GraphCell>) {
        let diff = self.bb_sub(a, b);
        self.bb_assert_zero(diff);
    }

    fn bb_zero(&mut self) -> BabyBearWire<GraphCell> {
        self.bb_load_constant(BabyBear::new(0))
    }

    fn bb_one(&mut self) -> BabyBearWire<GraphCell> {
        self.bb_load_constant(BabyBear::new(1))
    }

    fn bb_mul_const(&mut self, a: BabyBearWire<GraphCell>, c: BabyBear) -> BabyBearWire<GraphCell> {
        let c_wire = self.bb_load_constant(c);
        self.bb_mul(a, c_wire)
    }

    fn bb_square(&mut self, a: BabyBearWire<GraphCell>) -> BabyBearWire<GraphCell> {
        self.bb_mul(a, a)
    }

    fn bb_pow_power_of_two(
        &mut self,
        a: BabyBearWire<GraphCell>,
        n: usize,
    ) -> BabyBearWire<GraphCell> {
        let mut result = a;
        for _ in 0..n {
            result = self.bb_square(result);
        }
        result
    }
}

impl BabyBearExt4Inst for Halo2IRBuilder {
    fn ext_load_constant(&mut self, value: BabyBearExt4) -> BabyBearExt4Wire<GraphCell> {
        let coeffs = value.as_basis_coefficients_slice();
        BabyBearExt4Wire(array::from_fn(|i| self.bb_load_constant(coeffs[i])))
    }

    fn ext_load_reduced_constant(
        &mut self,
        value: BabyBearExt4,
    ) -> ReducedBabyBearExt4Wire<GraphCell> {
        let coeffs = value.as_basis_coefficients_slice();
        ReducedBabyBearExt4Wire::assume_reduced(array::from_fn(|i| {
            self.bb_load_reduced_constant(coeffs[i])
        }))
    }

    fn ext_add(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
        b: BabyBearExt4Wire<GraphCell>,
    ) -> BabyBearExt4Wire<GraphCell> {
        BabyBearExt4Wire(array::from_fn(|i| self.bb_add(a.0[i], b.0[i])))
    }

    fn ext_neg(&mut self, a: BabyBearExt4Wire<GraphCell>) -> BabyBearExt4Wire<GraphCell> {
        BabyBearExt4Wire(array::from_fn(|i| self.bb_neg(a.0[i])))
    }

    fn ext_sub(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
        b: BabyBearExt4Wire<GraphCell>,
    ) -> BabyBearExt4Wire<GraphCell> {
        BabyBearExt4Wire(array::from_fn(|i| self.bb_sub(a.0[i], b.0[i])))
    }

    fn ext_scalar_mul(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
        b: BabyBearWire<GraphCell>,
    ) -> BabyBearExt4Wire<GraphCell> {
        BabyBearExt4Wire(array::from_fn(|i| self.bb_mul(a.0[i], b)))
    }

    fn ext_scalar_mul_add(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
        b: BabyBearWire<GraphCell>,
        c: BabyBearExt4Wire<GraphCell>,
    ) -> BabyBearExt4Wire<GraphCell> {
        BabyBearExt4Wire(array::from_fn(|i| self.bb_mul_add(a.0[i], b, c.0[i])))
    }

    fn ext_assert_zero(&mut self, a: BabyBearExt4Wire<GraphCell>) {
        for x in a.0 {
            self.bb_assert_zero(x);
        }
    }

    fn ext_assert_equal(&mut self, a: BabyBearExt4Wire<GraphCell>, b: BabyBearExt4Wire<GraphCell>) {
        for (a, b) in a.0.into_iter().zip(b.0) {
            self.bb_assert_equal(a, b);
        }
    }

    fn ext_mul(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
        b: BabyBearExt4Wire<GraphCell>,
    ) -> BabyBearExt4Wire<GraphCell> {
        let operands = a.0.iter().chain(b.0.iter()).map(|w| w.value).collect_vec();
        let (id, offsets) = self.emit(Halo2Opcode::ExtMul, operands);
        // `BabyBearExt4Chip::mul` internally loads the W constant (output 4 when uncached).
        self.note_internal_bb_const(id, offsets[4], <BabyBear as BinomiallyExtendable<4>>::W);
        let bits = ext_mul_result_bits(a.0.map(|w| w.max_bits), b.0.map(|w| w.max_bits));
        BabyBearExt4Wire(array::from_fn(|i| {
            bb_wire(GraphCell::Cell(id, offsets[i], bits[i] as u16), bits[i])
        }))
    }

    fn ext_div(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
        b: BabyBearExt4Wire<GraphCell>,
    ) -> BabyBearExt4Wire<GraphCell> {
        let operands = a.0.iter().chain(b.0.iter()).map(|w| w.value).collect_vec();
        let (id, offsets) = self.emit(Halo2Opcode::ExtDiv, operands);
        // `BabyBearExt4Chip::div` internally loads ext ONE = bb [1, 0, 0, 0] (outputs 4/5
        // when uncached) and, via the internal ext mul, the W constant (output 6).
        self.note_internal_bb_const(id, offsets[4], BabyBear::new(1));
        self.note_internal_bb_const(id, offsets[5], BabyBear::new(0));
        self.note_internal_bb_const(id, offsets[6], <BabyBear as BinomiallyExtendable<4>>::W);
        // The quotient is loaded as an ext witness: each coefficient is 31 bits.
        BabyBearExt4Wire(array::from_fn(|i| {
            bb_wire(
                GraphCell::Cell(id, offsets[i], BABYBEAR_MAX_BITS as u16),
                BABYBEAR_MAX_BITS,
            )
        }))
    }

    fn ext_reduce_max_bits(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
    ) -> BabyBearExt4Wire<GraphCell> {
        BabyBearExt4Wire(array::from_fn(|i| self.bb_reduce_max_bits(a.0[i])))
    }

    fn ext_zero(&mut self) -> BabyBearExt4Wire<GraphCell> {
        self.ext_from_base_const(BabyBear::new(0))
    }

    fn ext_from_base_const(&mut self, value: BabyBear) -> BabyBearExt4Wire<GraphCell> {
        let base_val = self.bb_load_constant(value);
        let z = self.bb_load_constant(BabyBear::new(0));
        BabyBearExt4Wire([base_val, z, z, z])
    }

    fn ext_from_base_var(&mut self, value: BabyBearWire<GraphCell>) -> BabyBearExt4Wire<GraphCell> {
        let z = self.bb_load_constant(BabyBear::new(0));
        BabyBearExt4Wire([value, z, z, z])
    }

    fn ext_mul_base_const(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
        c: BabyBear,
    ) -> BabyBearExt4Wire<GraphCell> {
        let c_wire = self.bb_load_constant(c);
        self.ext_scalar_mul(a, c_wire)
    }

    fn ext_square(&mut self, a: BabyBearExt4Wire<GraphCell>) -> BabyBearExt4Wire<GraphCell> {
        self.ext_mul(a, a)
    }

    fn ext_pow_power_of_two(
        &mut self,
        a: BabyBearExt4Wire<GraphCell>,
        n: usize,
    ) -> BabyBearExt4Wire<GraphCell> {
        let mut result = a;
        for _ in 0..n {
            result = self.ext_square(result);
        }
        result
    }
}

impl DigestHashInst for Halo2IRBuilder {
    /// Mirrors `hash::poseidon2::hash_babybear_slice_to_digest`.
    fn hash_babybear_slice_to_digest(
        &mut self,
        values: &[ReducedBabyBearWire<GraphCell>],
    ) -> GraphCell {
        let zero = self.load_zero();
        let mut state = [zero; POSEIDON2_WIDTH];
        for block_chunk in values.chunks(MULTI_FIELD32_RATE) {
            for (chunk_id, chunk) in block_chunk.chunks(MULTI_FIELD32_NUM_F_ELMS).enumerate() {
                state[chunk_id] = self.pack_base_2_31(chunk);
            }
            self.permute_t3(&mut state);
        }
        state[0]
    }

    fn compress_digests(&mut self, left: GraphCell, right: GraphCell) -> GraphCell {
        let (id, offsets) = self.emit(Halo2Opcode::PoseidonPermute2T2, vec![left, right]);
        GraphCell::Cell(id, offsets[0], RAW_MAX_BITS as u16)
    }
}

impl TranscriptInst for Halo2IRBuilder {
    fn init_transcript(&mut self) {
        let zero = self.load_zero();
        self.transcript = Some(IrTranscript {
            sponge_state: [zero; POSEIDON2_WIDTH],
            absorb_idx: 0,
            sample_idx: 0,
            observe_buf: Vec::with_capacity(NUM_OBS_PER_WORD),
            sample_buf: Vec::with_capacity(NUM_SAMPLES_PER_WORD),
        });
    }

    fn observe(&mut self, value: &ReducedBabyBearWire<GraphCell>) {
        let mut t = self.take_transcript();
        self.observe_inner(&mut t, value);
        self.transcript = Some(t);
    }

    fn observe_ext(&mut self, value: &ReducedBabyBearExt4Wire<GraphCell>) {
        let mut t = self.take_transcript();
        for coeff in value.coeffs() {
            self.observe_inner(&mut t, coeff);
        }
        self.transcript = Some(t);
    }

    fn observe_commit(&mut self, digest: &DigestWire<GraphCell>) {
        let mut t = self.take_transcript();
        t.sample_buf.clear();
        self.flush_observe_buf(&mut t);
        for &elem in &digest.elems {
            self.sponge_absorb(&mut t, elem);
        }
        self.transcript = Some(t);
    }

    fn sample(&mut self) -> BabyBearWire<GraphCell> {
        let mut t = self.take_transcript();
        let out = self.sample_inner(&mut t);
        self.transcript = Some(t);
        out
    }

    fn sample_ext(&mut self) -> BabyBearExt4Wire<GraphCell> {
        let mut t = self.take_transcript();
        let coeffs = array::from_fn(|_| self.sample_inner(&mut t));
        self.transcript = Some(t);
        BabyBearExt4Wire(coeffs)
    }

    fn sample_bits(&mut self, bits: usize) -> GraphCell {
        let mut t = self.take_transcript();
        let out = self.sample_bits_inner(&mut t, bits);
        self.transcript = Some(t);
        out
    }

    fn check_witness(&mut self, bits: usize, witness: &ReducedBabyBearWire<GraphCell>) {
        if bits == 0 {
            return;
        }
        let mut t = self.take_transcript();
        self.observe_inner(&mut t, witness);
        // `assert_is_const(sampled_bits, 0)` is a copy constraint to a fixed cell and
        // assigns no advice cells, so only the sample_bits cells appear in the tape.
        let _sampled_bits = self.sample_bits_inner(&mut t, bits);
        self.transcript = Some(t);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Read},
        path::PathBuf,
    };

    use openvm_stark_sdk::{
        config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
        openvm_stark_backend::proof::Proof,
    };

    use super::*;
    use crate::{config::StaticVerifierShape, StaticVerifierCircuit};

    fn cache_dir() -> PathBuf {
        std::env::var("STATIC_VERIFIER_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../bin/static-verifier-tracegen/cache")
            })
    }

    /// Reads only the length-prefixed JSON `(circuit, shape)` header of a
    /// `StaticVerifierProvingKey` encoding, skipping the halo2 proving key bytes.
    fn read_static_circuit(path: &std::path::Path) -> (StaticVerifierCircuit, StaticVerifierShape) {
        let mut reader = BufReader::new(File::open(path).expect("open static verifier pk"));
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes).expect("read JSON length");
        let len = u64::from_le_bytes(len_bytes) as usize;
        let mut bytes = vec![0u8; len];
        reader.read_exact(&mut bytes).expect("read JSON section");
        serde_json::from_slice(&bytes).expect("deserialize (circuit, shape)")
    }

    #[test]
    #[ignore = "requires cached static verifier pk + root proof from bin/static-verifier-tracegen"]
    fn ir_generation_stats_for_root_proof() {
        let dir = cache_dir();
        let (circuit, shape) = read_static_circuit(&dir.join("static_verifier_pk.bin"));
        let proof: Proof<RootConfig> = bitcode::deserialize(
            &std::fs::read(dir.join("root_proof.bitcode")).expect("read root proof"),
        )
        .expect("deserialize root proof");

        let mut builder = Halo2IRBuilder::new(shape.lookup_bits);
        circuit.populate_verify_stark_constraints(&mut builder, &proof);

        let stats = builder.stats();
        println!("=== static verifier IR stats ===");
        print!("{stats}");
        println!("ctx cells: {}", builder.total_ctx_len());
        println!("lookup cells: {}", builder.total_lookups_len());
        assert!(stats.num_nodes > 0);
        assert_eq!(builder.nodes.len(), builder.levels.len());
        assert_eq!(builder.nodes.len(), builder.node_meta.len());
    }
}
