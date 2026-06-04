use std::borrow::Borrow;

use itertools::{izip, Itertools as _};
use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
        VmAdapterInterface, VmCoreAir, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    var_range::VariableRangeCheckerBus, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, DEFERRAL_AS,
};
use openvm_riscv_circuit::adapters::{byte_ptr_to_u16_ptr, expand_to_rv64_block};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::PrimeField32;

use super::NUM_ACCUMULATORS_PER_IDX;
use crate::{
    canonicity::{CanonicityAuxCols, CanonicitySubAir},
    count::DeferralCircuitCountBus,
    poseidon2::DeferralPoseidon2Bus,
    utils::{
        combine_output_cells, scale_output_len, scale_rv64_ptr_high_u16, split_f_memory_ops,
        u16_commit_to_f, u16s_to_f, COMMIT_MEMORY_OPS, COMMIT_NUM_U16S, DIGEST_F_MEMORY_OPS,
        F_NUM_U16S, OUTPUT_TOTAL_MEMORY_OPS, OUTPUT_TOTAL_NUM_U16S, RV64_PTR_U16S, U16_BITS,
    },
};

// ========================= CORE ==============================

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct DeferralCallReads<B, F> {
    // Commit to a specific deferral input, passed in by the user as a pointer
    pub input_commit: [B; COMMIT_NUM_U16S],

    // Deferral address space accumulators immediately prior to the current deferral call
    pub old_input_acc: [F; DIGEST_SIZE],
    pub old_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct DeferralCallWrites<B, F> {
    // Output key for raw output + its length in bytes. These cells are written as one
    // contiguous heap write, with layout [output_commit || output_len_le]. Note output_len
    // **must** be divisible by SPONGE_BYTES_PER_ROW.
    pub output_commit: [B; COMMIT_NUM_U16S],
    pub output_len: [B; F_NUM_U16S],

    // Deferral address space accumulators after incorporating the current deferral call
    pub new_input_acc: [F; DIGEST_SIZE],
    pub new_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeferralCallReadsBytes<F> {
    pub input_commit: [u8; crate::utils::COMMIT_NUM_BYTES],
    pub old_input_acc: [F; DIGEST_SIZE],
    pub old_output_acc: [F; DIGEST_SIZE],
}

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

        // Constrain the canonicity of both commits, i.e. that every
        // F_NUM_U16S u16 cells uniquely represent an element of F.
        let input_commit_rcs = izip!(
            cols.reads.input_commit.chunks_exact(F_NUM_U16S),
            cols.input_commit_lt_aux
        )
        .map(|(cells, aux)| {
            let cells: &[_; F_NUM_U16S] = cells.try_into().unwrap();
            CanonicitySubAir.assert_canonicity(builder, cells, &aux, cols.is_valid.into())
        })
        .collect_vec();

        let output_commit_rcs = izip!(
            cols.writes.output_commit.chunks_exact(F_NUM_U16S),
            cols.output_commit_lt_aux
        )
        .map(|(cells, aux)| {
            let cells: &[_; F_NUM_U16S] = cells.try_into().unwrap();
            CanonicitySubAir.assert_canonicity(builder, cells, &aux, cols.is_valid.into())
        })
        .collect_vec();

        for rc in input_commit_rcs {
            self.range_bus
                .range_check(rc, U16_BITS)
                .eval(builder, cols.is_valid);
        }
        for rc in output_commit_rcs {
            self.range_bus
                .range_check(rc, U16_BITS)
                .eval(builder, cols.is_valid);
        }

        // Range-check the u16 cells written to heap memory.
        for &cell in cols
            .writes
            .output_commit
            .iter()
            .chain(cols.writes.output_len.iter())
        {
            self.range_bus
                .range_check(cell, U16_BITS)
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
    pub rd_val: [T; RV64_PTR_U16S],
    pub rs_val: [T; RV64_PTR_U16S],
    pub rd_aux: MemoryReadAuxCols<T>,
    pub rs_aux: MemoryReadAuxCols<T>,

    // Heap commit reads use byte chunks; accumulator reads use DEFERRAL_AS
    // cell chunks.
    pub input_commit_aux: [MemoryReadAuxCols<T>; COMMIT_MEMORY_OPS],
    pub old_input_acc_aux: [MemoryReadAuxCols<T>; DIGEST_F_MEMORY_OPS],
    pub old_output_acc_aux: [MemoryReadAuxCols<T>; DIGEST_F_MEMORY_OPS],

    // Heap output writes use byte chunks; accumulator writes use DEFERRAL_AS
    // cell chunks.
    pub output_commit_and_len_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; OUTPUT_TOTAL_MEMORY_OPS],
    pub new_input_acc_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; DIGEST_F_MEMORY_OPS],
    pub new_output_acc_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; DIGEST_F_MEMORY_OPS],
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(DeferralCallAdapterCols<u8>)]
pub struct DeferralCallAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
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

        let rd_bus: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_rv64_block(&cols.rd_val);
        let rs_bus: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_rv64_block(&cols.rs_val);

        // Heap pointers are first read from their respective registers.
        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), byte_ptr_to_u16_ptr::<AB>(cols.rd_ptr)),
                rd_bus,
                timestamp_pp(),
                &cols.rd_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), byte_ptr_to_u16_ptr::<AB>(cols.rs_ptr)),
                rs_bus,
                timestamp_pp(),
                &cols.rs_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // We range check the high u16 of both heap pointers to ensure that each
        // access is in [0, 2^address_bits).
        self.range_bus
            .range_check(
                scale_rv64_ptr_high_u16::<AB::Expr, _>(
                    cols.rd_val[RV64_PTR_U16S - 1],
                    self.address_bits,
                ),
                U16_BITS,
            )
            .eval(builder, ctx.instruction.is_valid.clone());
        self.range_bus
            .range_check(
                scale_rv64_ptr_high_u16::<AB::Expr, _>(
                    cols.rs_val[RV64_PTR_U16S - 1],
                    self.address_bits,
                ),
                U16_BITS,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Accumulators are read then updated in the deferral address space,
        // using deferral_idx (instruction immediate / operand c) to determine
        // the accumulator memory address.
        let input_ptr = u16s_to_f(&cols.rs_val);
        let output_ptr = u16s_to_f(&cols.rd_val);

        let deferral_idx = ctx.instruction.immediate;
        let deferral_as = AB::Expr::from_u32(DEFERRAL_AS);

        // Accumulators are consecutive DEFERRAL_AS cell ranges.
        let digest_size = AB::F::from_usize(DIGEST_SIZE);
        let num_accumulators = AB::F::from_usize(NUM_ACCUMULATORS_PER_IDX);
        let input_acc_ptr = deferral_idx.clone() * num_accumulators * digest_size;
        let output_acc_ptr = input_acc_ptr.clone() + AB::Expr::from(digest_size);

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
        self.range_bus
            .range_check(
                scale_output_len::<AB::Expr, _>(&output_len, self.address_bits),
                U16_BITS,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        let output_len_full: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_rv64_block(&output_len);

        let input_commit_chunks: [[AB::Expr; BLOCK_FE_WIDTH]; COMMIT_MEMORY_OPS] =
            split_f_memory_ops::<AB::Expr, COMMIT_NUM_U16S, COMMIT_MEMORY_OPS>(input_commit);
        for (chunk_idx, (data, aux)) in input_commit_chunks
            .into_iter()
            .zip(&cols.input_commit_aux)
            .enumerate()
        {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e.clone(),
                        byte_ptr_to_u16_ptr::<AB>(
                            input_ptr.clone()
                                + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                        ),
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
                .read(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        input_acc_ptr.clone() + AB::Expr::from_usize(chunk_idx * BLOCK_FE_WIDTH),
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
                .read(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        output_acc_ptr.clone() + AB::Expr::from_usize(chunk_idx * BLOCK_FE_WIDTH),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let output_commit_and_len = combine_output_cells(output_commit, output_len_full);
        let output_chunks =
            split_f_memory_ops::<AB::Expr, OUTPUT_TOTAL_NUM_U16S, OUTPUT_TOTAL_MEMORY_OPS>(
                output_commit_and_len,
            );
        for (chunk_idx, (data, aux)) in output_chunks
            .into_iter()
            .zip(&cols.output_commit_and_len_aux)
            .enumerate()
        {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e.clone(),
                        byte_ptr_to_u16_ptr::<AB>(
                            output_ptr.clone()
                                + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let new_input_acc_chunks =
            split_f_memory_ops::<_, DIGEST_SIZE, DIGEST_F_MEMORY_OPS>(new_input_acc);
        for (chunk_idx, (data, aux)) in new_input_acc_chunks
            .into_iter()
            .zip(&cols.new_input_acc_aux)
            .enumerate()
        {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        input_acc_ptr.clone() + AB::Expr::from_usize(chunk_idx * BLOCK_FE_WIDTH),
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
                .write(
                    MemoryAddress::new(
                        deferral_as.clone(),
                        output_acc_ptr.clone() + AB::Expr::from_usize(chunk_idx * BLOCK_FE_WIDTH),
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
