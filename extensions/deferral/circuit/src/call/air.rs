use std::{array::from_fn, borrow::Borrow};

use itertools::{izip, Itertools as _};
use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
        VmAdapterInterface, VmCoreAir, BLOCK_FE_WIDTH, BUS_PTR_SCALE, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{pack_u8_for_bus, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, var_range::VariableRangeCheckerBus, ColumnsAir,
    StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_CELL_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_WORD_NUM_LIMBS},
    LocalOpcode, DEFERRAL_AS,
};
use openvm_riscv_circuit::adapters::expand_to_rv64_register;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::PrimeField32;

use crate::{
    canonicity::{CanonicityAuxCols, CanonicitySubAir},
    count::DeferralCircuitCountBus,
    poseidon2::DeferralPoseidon2Bus,
    utils::{
        bytes_to_f, split_f_memory_ops, u16_commit_to_f, COMMIT_MEMORY_OPS, COMMIT_NUM_U16S,
        DIGEST_F_MEMORY_OPS, F_NUM_U16S, OUTPUT_LEN_NUM_U16S, OUTPUT_TOTAL_MEMORY_OPS,
    },
};

/// Number of accumulators owned by each `deferral_idx` (input + output).
const NUM_ACCUMULATORS_PER_IDX: usize = 2;

// ========================= CORE ==============================

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct DeferralCallReads<B, F> {
    /// Commit to a specific deferral input, stored as u16 cells (matching the
    /// underlying Pattern B memory granularity).
    pub input_commit: [B; COMMIT_NUM_U16S],

    // Deferral address space accumulators immediately prior to the current deferral call
    pub old_input_acc: [F; DIGEST_SIZE],
    pub old_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct DeferralCallWrites<B, F> {
    /// Output key for raw output + its length in bytes. These cells are written
    /// as one contiguous heap write, with layout `[output_commit || output_len]`
    /// (all u16-celled). `output_len` is the byte length of the raw output and
    /// **must** be divisible by `DIGEST_SIZE`.
    pub output_commit: [B; COMMIT_NUM_U16S],
    pub output_len: [B; OUTPUT_LEN_NUM_U16S],

    // Deferral address space accumulators after incorporating the current deferral call
    pub new_input_acc: [F; DIGEST_SIZE],
    pub new_output_acc: [F; DIGEST_SIZE],
}

/// Byte-shaped sibling of [`DeferralCallReads`] used for the tracegen record.
/// The underlying heap is byte-granular, so the executor naturally reads bytes;
/// the trace filler packs bytes pairs into the u16-shaped column.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeferralCallReadsBytes<F> {
    pub input_commit: [u8; crate::utils::COMMIT_NUM_BYTES],
    pub old_input_acc: [F; DIGEST_SIZE],
    pub old_output_acc: [F; DIGEST_SIZE],
}

/// Byte-shaped sibling of [`DeferralCallWrites`] used for the tracegen record.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeferralCallWritesBytes<F> {
    pub output_commit: [u8; crate::utils::COMMIT_NUM_BYTES],
    pub output_len: [u8; crate::utils::F_NUM_BYTES],
    pub new_input_acc: [F; DIGEST_SIZE],
    pub new_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DeferralCallCoreCols<T> {
    pub is_valid: T,
    pub deferral_idx: T,
    pub reads: DeferralCallReads<T, T>,
    pub writes: DeferralCallWrites<T, T>,

    pub input_commit_lt_aux: [CanonicityAuxCols<T>; DIGEST_SIZE],
    pub output_commit_lt_aux: [CanonicityAuxCols<T>; DIGEST_SIZE],
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(DeferralCallCoreCols<u8>)]
pub struct DeferralCallCoreAir {
    pub count_bus: DeferralCircuitCountBus,
    pub poseidon2_bus: DeferralPoseidon2Bus,
    pub bitwise_bus: BitwiseOperationLookupBus,
    /// 16-bit range checker bus used for per-cell range checks on
    /// `input_commit` / `output_commit` / `output_len` u16 cells, and for the
    /// canonicity sub-AIR's `diff_val - 1` outputs (also 16-bit values).
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for DeferralCallCoreAir {
    fn width(&self) -> usize {
        DeferralCallCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for DeferralCallCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for DeferralCallCoreAir
where
    AB: InteractionBuilder,
    AB::F: PrimeField32,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<DeferralCallReads<AB::Expr, AB::Expr>>,
    I::Writes: From<DeferralCallWrites<AB::Expr, AB::Expr>>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &DeferralCallCoreCols<_> = local_core.borrow();
        builder.assert_bool(cols.is_valid);

        // Constrain the canonicity of both commits, i.e. that every `F_NUM_U16S`
        // u16 cells uniquely represent an element of F. Each canonicity walk
        // returns a `diff - 1` value in `[0, 2^16)` that we range-check below.
        let input_commit_rcs = izip!(
            cols.reads.input_commit.chunks_exact(F_NUM_U16S),
            cols.input_commit_lt_aux
        )
        .map(|(cells, aux)| {
            CanonicitySubAir.assert_canonicity(builder, cells, &aux, cols.is_valid.into())
        })
        .collect_vec();

        let output_commit_rcs = izip!(
            cols.writes.output_commit.chunks_exact(F_NUM_U16S),
            cols.output_commit_lt_aux
        )
        .map(|(cells, aux)| {
            CanonicitySubAir.assert_canonicity(builder, cells, &aux, cols.is_valid.into())
        })
        .collect_vec();

        // Range-check each canonicity `diff - 1` to 16 bits via the variable
        // range checker (one interaction per cell, mirroring the SHA-2 pattern).
        for rc in input_commit_rcs {
            self.range_bus
                .range_check(rc, 16)
                .eval(builder, cols.is_valid);
        }
        for rc in output_commit_rcs {
            self.range_bus
                .range_check(rc, 16)
                .eval(builder, cols.is_valid);
        }

        // Range-check the u16 cells we write to heap memory (and the output_len
        // cells we hand to the memory bus). Each cell goes through one 16-bit
        // range check.
        for &cell in cols.writes.output_commit.iter() {
            self.range_bus
                .range_check(cell, 16)
                .eval(builder, cols.is_valid);
        }
        for &cell in cols.writes.output_len.iter() {
            self.range_bus
                .range_check(cell, 16)
                .eval(builder, cols.is_valid);
        }

        // Constrain the updated accumulators.
        let input_f_commit = u16_commit_to_f(&cols.reads.input_commit);
        let output_f_commit = u16_commit_to_f(&cols.writes.output_commit);

        self.poseidon2_bus
            .lookup(
                cols.reads.old_input_acc,
                input_f_commit,
                cols.writes.new_input_acc,
                AB::Expr::ONE,
            )
            .eval(builder, cols.is_valid);

        self.poseidon2_bus
            .lookup(
                cols.reads.old_output_acc,
                output_f_commit,
                cols.writes.new_output_acc,
                AB::Expr::ONE,
            )
            .eval(builder, cols.is_valid);

        self.count_bus
            .send(cols.deferral_idx)
            .eval(builder, cols.is_valid);

        AdapterAirContext {
            to_pc: None,
            reads: DeferralCallReads {
                input_commit: cols.reads.input_commit.map(Into::into),
                old_input_acc: cols.reads.old_input_acc.map(Into::into),
                old_output_acc: cols.reads.old_output_acc.map(Into::into),
            }
            .into(),
            writes: DeferralCallWrites {
                output_commit: cols.writes.output_commit.map(Into::into),
                output_len: cols.writes.output_len.map(Into::into),
                new_input_acc: cols.writes.new_input_acc.map(Into::into),
                new_output_acc: cols.writes.new_output_acc.map(Into::into),
            }
            .into(),
            instruction: ImmInstruction {
                is_valid: cols.is_valid.into(),
                opcode: AB::Expr::from_usize(DeferralOpcode::CALL.global_opcode_usize()),
                immediate: cols.deferral_idx.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        DeferralOpcode::CLASS_OFFSET
    }
}

// ========================= ADAPTER ==============================

pub struct DeferralCallAdapterInterface;

impl<T> VmAdapterInterface<T> for DeferralCallAdapterInterface {
    type Reads = DeferralCallReads<T, T>;
    type Writes = DeferralCallWrites<T, T>;
    type ProcessedInstruction = ImmInstruction<T>;
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DeferralCallAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs_ptr: T,

    // Heap pointers and aux columns
    pub rd_val: [T; RV64_WORD_NUM_LIMBS],
    pub rs_val: [T; RV64_WORD_NUM_LIMBS],
    pub rd_aux: MemoryReadAuxCols<T>,
    pub rs_aux: MemoryReadAuxCols<T>,

    // Read auxiliary columns. `input_commit` is on byte-AS RV64_MEMORY_AS; the
    // two accumulator reads are on F-celled DEFERRAL_AS.
    pub input_commit_aux: [MemoryReadAuxCols<T>; COMMIT_MEMORY_OPS],
    pub old_input_acc_aux: [MemoryReadAuxCols<T>; DIGEST_F_MEMORY_OPS],
    pub old_output_acc_aux: [MemoryReadAuxCols<T>; DIGEST_F_MEMORY_OPS],

    // Write auxiliary columns. `output_commit_and_len` is on byte-AS
    // RV64_MEMORY_AS (data width `MEMORY_BLOCK_BYTES`); the two accumulator
    // writes are on F-celled DEFERRAL_AS (data width `BLOCK_FE_WIDTH`).
    pub output_commit_and_len_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; OUTPUT_TOTAL_MEMORY_OPS],
    pub new_input_acc_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; DIGEST_F_MEMORY_OPS],
    pub new_output_acc_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; DIGEST_F_MEMORY_OPS],
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(DeferralCallAdapterCols<u8>)]
pub struct DeferralCallAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_bus: BitwiseOperationLookupBus,
    /// 16-bit range checker bus used for the `output_len` high-cell range
    /// check (scaled to enforce `output_len < 2^address_bits`).
    pub range_bus: VariableRangeCheckerBus,
    pub address_bits: usize,
}

impl<F: Field> BaseAir<F> for DeferralCallAdapterAir {
    fn width(&self) -> usize {
        DeferralCallAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for DeferralCallAdapterAir {
    type Interface = DeferralCallAdapterInterface;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &DeferralCallAdapterCols<_> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Operands a and b are register pointers. Their values are read first
        // to get heap pointers for output write and input commit read respectively.
        let d = AB::Expr::from_u32(RV64_REGISTER_AS);
        let e = AB::Expr::from_u32(RV64_MEMORY_AS);

        // `rd_val` / `rs_val` remain byte-shaped to keep the existing pointer
        // range-check + `bytes_to_f` decoding. Pad them to 8 bytes (upper 4
        // limbs hardcoded zero) and pack to 4 u16 cells for the bus payload.
        let rd_full = expand_to_rv64_register(&cols.rd_val);
        let rs_full = expand_to_rv64_register(&cols.rs_val);

        // Heap pointers are first read from their respective registers.
        self.memory_bridge
            .read_4(
                MemoryAddress::new(d.clone(), cols.rd_ptr),
                pack_u8_for_bus::<AB>(&rd_full),
                timestamp_pp(),
                &cols.rd_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read_4(
                MemoryAddress::new(d.clone(), cols.rs_ptr),
                pack_u8_for_bus::<AB>(&rs_full),
                timestamp_pp(),
                &cols.rs_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // We range check the top byte of both heap pointers to ensure that each
        // access is in [0, 2^address_bits). The memory merkle argument ensures
        // that each read/write pointer is less than 2^addr_bits, and this range
        // check ensures the accesses don't wrap around P.
        debug_assert!(RV64_CELL_BITS * RV64_WORD_NUM_LIMBS >= self.address_bits);
        let limb_shift =
            AB::F::from_usize(1 << (RV64_CELL_BITS * RV64_WORD_NUM_LIMBS - self.address_bits));

        self.bitwise_bus
            .send_range(
                cols.rd_val[RV64_WORD_NUM_LIMBS - 1] * limb_shift,
                cols.rs_val[RV64_WORD_NUM_LIMBS - 1] * limb_shift,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Accumulators are read then updated in the deferral address space,
        // using deferral_idx (instruction immediate / operand c) to determine
        // the accumulator memory address.
        let input_ptr = bytes_to_f(&cols.rs_val);
        let output_ptr = bytes_to_f(&cols.rd_val);

        let deferral_idx = ctx.instruction.immediate;
        let deferral_as = AB::Expr::from_u32(DEFERRAL_AS);

        // DEFERRAL_AS bus pointers. Each deferral idx owns
        // NUM_ACCUMULATORS_PER_IDX × DIGEST_SIZE F cells (input + output
        // accumulators). The bridge expects the normalized memory-bus pointer
        // `bus_ptr = BUS_PTR_SCALE * cell_idx`; for RV64 byte-AS chips this
        // multiplication is already baked into the byte stride, but DEFERRAL_AS
        // has no equivalent so we apply it explicitly.
        let digest_size = AB::F::from_usize(DIGEST_SIZE);
        let bus_ptr_scale = AB::F::from_usize(BUS_PTR_SCALE);
        let num_accumulators = AB::F::from_usize(NUM_ACCUMULATORS_PER_IDX);
        let input_acc_ptr = deferral_idx.clone() * num_accumulators * digest_size * bus_ptr_scale;
        let output_acc_ptr = input_acc_ptr.clone() + AB::Expr::from(digest_size * bus_ptr_scale);

        let DeferralCallReads {
            input_commit,
            old_input_acc,
            old_output_acc,
        } = ctx.reads;
        let DeferralCallWrites {
            output_commit,
            output_len,
            new_input_acc,
            new_output_acc,
        } = ctx.writes;

        // Constrain output_len to be under 2^address_bits. `output_len` is 2
        // u16 cells; the high cell holds bits [16, 32). Scale it so the
        // 16-bit range check forces the value into `[0, 2^(address_bits - 16))`.
        debug_assert!(self.address_bits >= 16);
        let address_bits_high_shift = AB::F::from_usize(1 << (F_NUM_U16S * 16 - self.address_bits));
        self.range_bus
            .range_check(
                output_len[OUTPUT_LEN_NUM_U16S - 1].clone() * address_bits_high_shift,
                16,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Zero-pad `output_len` to `BLOCK_FE_WIDTH` cells (the bus message
        // width) so the upper half of the 8-byte register access is zero.
        let output_len_full: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|i| {
            if i < OUTPUT_LEN_NUM_U16S {
                output_len[i].clone()
            } else {
                AB::Expr::ZERO
            }
        });

        // `input_commit` is 16 u16 cells = `COMMIT_MEMORY_OPS` chunks of
        // `BLOCK_FE_WIDTH` cells each; pass them straight to the bridge.
        let input_commit_chunks: [[AB::Expr; BLOCK_FE_WIDTH]; COMMIT_MEMORY_OPS] =
            split_f_memory_ops::<AB::Expr, COMMIT_NUM_U16S, COMMIT_MEMORY_OPS>(input_commit);
        for (chunk_idx, (data, aux)) in input_commit_chunks
            .into_iter()
            .zip(&cols.input_commit_aux)
            .enumerate()
        {
            self.memory_bridge
                .read_4(
                    MemoryAddress::new(
                        e.clone(),
                        input_ptr.clone() + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let old_input_acc_chunks =
            split_f_memory_ops::<_, DIGEST_SIZE, DIGEST_F_MEMORY_OPS>(old_input_acc);
        for (chunk_idx, (data, aux)) in old_input_acc_chunks
            .into_iter()
            .zip(&cols.old_input_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .read_4(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        input_acc_ptr.clone()
                            + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let old_output_acc_chunks =
            split_f_memory_ops::<_, DIGEST_SIZE, DIGEST_F_MEMORY_OPS>(old_output_acc);
        for (chunk_idx, (data, aux)) in old_output_acc_chunks
            .into_iter()
            .zip(&cols.old_output_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .read_4(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        output_acc_ptr.clone()
                            + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // The output write spans `OUTPUT_TOTAL_MEMORY_OPS` memory ops of
        // `BLOCK_FE_WIDTH` u16 cells each. The first `COMMIT_MEMORY_OPS` ops
        // carry `output_commit` cells; the last op carries `output_len`
        // zero-padded to `BLOCK_FE_WIDTH`.
        let output_commit_chunks: [[AB::Expr; BLOCK_FE_WIDTH]; COMMIT_MEMORY_OPS] =
            split_f_memory_ops::<AB::Expr, COMMIT_NUM_U16S, COMMIT_MEMORY_OPS>(output_commit);
        let mut combined_chunks_iter = output_commit_chunks
            .into_iter()
            .chain(std::iter::once(output_len_full));
        for (chunk_idx, aux) in cols.output_commit_and_len_aux.iter().enumerate() {
            let data = combined_chunks_iter.next().unwrap();
            self.memory_bridge
                .write_4(
                    MemoryAddress::new(
                        e.clone(),
                        output_ptr.clone() + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }
        debug_assert!(combined_chunks_iter.next().is_none());

        let new_input_acc_chunks =
            split_f_memory_ops::<_, DIGEST_SIZE, DIGEST_F_MEMORY_OPS>(new_input_acc);
        for (chunk_idx, (data, aux)) in new_input_acc_chunks
            .into_iter()
            .zip(&cols.new_input_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .write_4(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        input_acc_ptr.clone()
                            + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let new_output_acc_chunks =
            split_f_memory_ops::<_, DIGEST_SIZE, DIGEST_F_MEMORY_OPS>(new_output_acc);
        for (chunk_idx, (data, aux)) in new_output_acc_chunks
            .into_iter()
            .zip(&cols.new_output_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .write_4(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        output_acc_ptr.clone()
                            + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.rd_ptr.into(),
                    cols.rs_ptr.into(),
                    deferral_idx,
                    d.clone(),
                    e.clone(),
                ],
                cols.from_state,
                AB::Expr::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &DeferralCallAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}
