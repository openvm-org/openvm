use std::{array::from_fn, borrow::Borrow};

use itertools::{izip, Itertools as _};
use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
        VmAdapterInterface, VmCoreAir, DEFAULT_BLOCK_SIZE,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode, DEFERRAL_AS,
};
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
        byte_commit_to_f, bytes_to_f, combine_output, split_memory_ops, COMMIT_MEMORY_OPS,
        COMMIT_NUM_BYTES, DIGEST_MEMORY_OPS, F_NUM_BYTES, OUTPUT_TOTAL_BYTES,
        OUTPUT_TOTAL_MEMORY_OPS,
    },
};

// ========================= CORE ==============================

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct DeferralCallReads<B, F> {
    // Commit to a specific deferral input, passed in by the user as a pointer
    pub input_commit: [B; COMMIT_NUM_BYTES],

    // Native address space accumulators immediately prior to the current deferral call
    pub old_input_acc: [F; DIGEST_SIZE],
    pub old_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct DeferralCallWrites<B, F> {
    // Output key for raw output + its length in bytes. These bytes are written as one
    // contiguous heap write, with layout [output_commit || output_len_le]. Note output_len
    // **must** be divisible by DIGEST_SIZE.
    pub output_commit: [B; COMMIT_NUM_BYTES],
    pub output_len: [B; F_NUM_BYTES],

    // Native address space accumulators after incorporating the current deferral call
    pub new_input_acc: [F; DIGEST_SIZE],
    pub new_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralCallCoreCols<T> {
    pub is_valid: T,
    pub deferral_idx: T,
    pub reads: DeferralCallReads<T, T>,
    pub writes: DeferralCallWrites<T, T>,

    pub input_commit_lt_aux: [CanonicityAuxCols<T>; DIGEST_SIZE],
    pub output_commit_lt_aux: [CanonicityAuxCols<T>; DIGEST_SIZE],
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct DeferralCallCoreAir {
    pub count_bus: DeferralCircuitCountBus,
    pub poseidon2_bus: DeferralPoseidon2Bus,
    pub bitwise_bus: BitwiseOperationLookupBus,
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

        // Constrain the canonicity of both commits and output_len, i.e. that every
        // F_NUM_BYTES bytes uniquely represents an element of F.
        let input_commit_rcs = izip!(
            cols.reads.input_commit.chunks_exact(F_NUM_BYTES),
            cols.input_commit_lt_aux
        )
        .map(|(bytes, aux)| {
            CanonicitySubAir.assert_canonicity(builder, bytes, &aux, cols.is_valid.into())
        })
        .collect_vec();

        let output_commit_rcs = izip!(
            cols.writes.output_commit.chunks_exact(F_NUM_BYTES),
            cols.output_commit_lt_aux
        )
        .map(|(bytes, aux)| {
            CanonicitySubAir.assert_canonicity(builder, bytes, &aux, cols.is_valid.into())
        })
        .collect_vec();

        for rc_pair in input_commit_rcs.chunks_exact(2) {
            self.bitwise_bus
                .send_range(rc_pair[0].clone(), rc_pair[1].clone())
                .eval(builder, cols.is_valid);
        }
        for rc_pair in output_commit_rcs.chunks_exact(2) {
            self.bitwise_bus
                .send_range(rc_pair[0].clone(), rc_pair[1].clone())
                .eval(builder, cols.is_valid);
        }

        // Range check the bytes that we write to RV32 heap memory.
        for bytes in cols.writes.output_commit.chunks_exact(2) {
            self.bitwise_bus
                .send_range(bytes[0], bytes[1])
                .eval(builder, cols.is_valid);
        }

        for bytes in cols.writes.output_len.chunks_exact(2) {
            self.bitwise_bus
                .send_range(bytes[0], bytes[1])
                .eval(builder, cols.is_valid);
        }

        // Constrain the updated accumulators.
        let input_f_commit = byte_commit_to_f(&cols.reads.input_commit);
        let output_f_commit = byte_commit_to_f(&cols.writes.output_commit);

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
#[derive(AlignedBorrow)]
pub struct DeferralCallAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs_ptr: T,

    // Heap pointers and aux columns
    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rd_aux: MemoryReadAuxCols<T>,
    pub rs_aux: MemoryReadAuxCols<T>,

    // Read auxiliary columns
    pub input_commit_aux: [MemoryReadAuxCols<T>; COMMIT_MEMORY_OPS],
    pub old_input_acc_aux: [MemoryReadAuxCols<T>; DIGEST_MEMORY_OPS],
    pub old_output_acc_aux: [MemoryReadAuxCols<T>; DIGEST_MEMORY_OPS],

    // Write auxiliary columns
    pub output_commit_and_len_aux:
        [MemoryWriteAuxCols<T, DEFAULT_BLOCK_SIZE>; OUTPUT_TOTAL_MEMORY_OPS],
    pub new_input_acc_aux: [MemoryWriteAuxCols<T, DEFAULT_BLOCK_SIZE>; DIGEST_MEMORY_OPS],
    pub new_output_acc_aux: [MemoryWriteAuxCols<T, DEFAULT_BLOCK_SIZE>; DIGEST_MEMORY_OPS],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralCallAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_bus: BitwiseOperationLookupBus,
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

        // Operands a and b are RV32 register pointers. Their values are read first
        // to get heap pointers for output write and input commit read respectively.
        let d = AB::Expr::from_u32(RV32_REGISTER_AS);
        let e = AB::Expr::from_u32(RV32_MEMORY_AS);

        // Heap pointers are first read from their respective registers.
        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), cols.rd_ptr),
                cols.rd_val,
                timestamp_pp(),
                &cols.rd_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), cols.rs_ptr),
                cols.rs_val,
                timestamp_pp(),
                &cols.rs_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // We range check the top byte of both heap pointers to ensure that each
        // access is in [0, 2^address_bits). The memory merkle argument ensures
        // that each read/write pointer is less than 2^addr_bits, and this range
        // check ensures the accesses don't wrap around P.
        debug_assert!(RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS >= self.address_bits);
        let limb_shift =
            AB::F::from_usize(1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits));

        self.bitwise_bus
            .send_range(
                cols.rd_val[RV32_REGISTER_NUM_LIMBS - 1] * limb_shift,
                cols.rs_val[RV32_REGISTER_NUM_LIMBS - 1] * limb_shift,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Accumulators are read then updated in the native address space, using
        // deferral_idx (instruction immediate / operand c) to determine the
        // accumulator memory address.
        let input_ptr = bytes_to_f(&cols.rs_val);
        let output_ptr = bytes_to_f(&cols.rd_val);

        let deferral_idx = ctx.instruction.immediate;
        let deferral_as = AB::Expr::from_u32(DEFERRAL_AS);

        let digest_size = AB::F::from_usize(DIGEST_SIZE);
        let input_acc_ptr = deferral_idx.clone() * AB::Expr::TWO * digest_size;
        let output_acc_ptr = input_acc_ptr.clone() + digest_size;

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

        // Constrain output_len to be under 2^address_bits also.
        self.bitwise_bus
            .send_range(
                output_len[RV32_REGISTER_NUM_LIMBS - 1].clone() * limb_shift,
                AB::Expr::ZERO,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        let output_len_full = from_fn(|i| {
            if i < F_NUM_BYTES {
                output_len[i].clone()
            } else {
                AB::Expr::ZERO
            }
        });

        let input_commit_chunks =
            split_memory_ops::<_, COMMIT_NUM_BYTES, COMMIT_MEMORY_OPS>(input_commit);
        for (chunk_idx, (data, aux)) in input_commit_chunks
            .into_iter()
            .zip(&cols.input_commit_aux)
            .enumerate()
        {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e.clone(),
                        input_ptr.clone() + AB::Expr::from_usize(chunk_idx * DEFAULT_BLOCK_SIZE),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let old_input_acc_chunks =
            split_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(old_input_acc);
        for (chunk_idx, (data, aux)) in old_input_acc_chunks
            .into_iter()
            .zip(&cols.old_input_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        input_acc_ptr.clone()
                            + AB::Expr::from_usize(chunk_idx * DEFAULT_BLOCK_SIZE),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let old_output_acc_chunks =
            split_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(old_output_acc);
        for (chunk_idx, (data, aux)) in old_output_acc_chunks
            .into_iter()
            .zip(&cols.old_output_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        output_acc_ptr.clone()
                            + AB::Expr::from_usize(chunk_idx * DEFAULT_BLOCK_SIZE),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let output_commit_and_len = combine_output(output_commit, output_len_full);
        let output_commit_and_len_chunks =
            split_memory_ops::<_, OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS>(
                output_commit_and_len,
            );
        for (chunk_idx, (data, aux)) in output_commit_and_len_chunks
            .into_iter()
            .zip(&cols.output_commit_and_len_aux)
            .enumerate()
        {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e.clone(),
                        output_ptr.clone() + AB::Expr::from_usize(chunk_idx * DEFAULT_BLOCK_SIZE),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let new_input_acc_chunks =
            split_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(new_input_acc);
        for (chunk_idx, (data, aux)) in new_input_acc_chunks
            .into_iter()
            .zip(&cols.new_input_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        input_acc_ptr.clone()
                            + AB::Expr::from_usize(chunk_idx * DEFAULT_BLOCK_SIZE),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let new_output_acc_chunks =
            split_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(new_output_acc);
        for (chunk_idx, (data, aux)) in new_output_acc_chunks
            .into_iter()
            .zip(&cols.new_output_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        output_acc_ptr.clone()
                            + AB::Expr::from_usize(chunk_idx * DEFAULT_BLOCK_SIZE),
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
