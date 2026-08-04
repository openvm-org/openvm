use std::{array::from_fn, borrow::Borrow};

use itertools::{izip, Itertools as _};
use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir,
        VmAdapterInterface, VmCoreAir, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES, U16_CELL_SIZE,
    },
    system::memory::{
        offline_checker::{pack_u8_block, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
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
    riscv::{RV64_BYTE_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_WORD_NUM_LIMBS},
    LocalOpcode, DEFERRAL_AS,
};
use openvm_riscv_circuit::adapters::{
    eval_add_const_u16_limbs, eval_byte_ptr_limbs_to_u16_cell_ptr_limbs, expand_to_rv64_register,
    pack_u8_pair, reg_byte_ptr_to_cell_ptr_limbs,
};
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
        byte_commit_to_f, combine_output, split_byte_memory_ops, split_f_memory_ops,
        COMMIT_MEMORY_OPS, COMMIT_NUM_BYTES, DIGEST_F_MEMORY_OPS, F_NUM_BYTES, OUTPUT_TOTAL_BYTES,
        OUTPUT_TOTAL_MEMORY_OPS,
    },
};

// ========================= CORE ==============================

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct DeferralCallReads<B, F> {
    // Commit to a specific deferral input, passed in by the user as a pointer
    pub input_commit: [B; COMMIT_NUM_BYTES],

    // Deferral address space accumulators immediately prior to the current deferral call
    pub old_input_acc: [F; DIGEST_SIZE],
    pub old_output_acc: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Copy, Debug)]
pub struct DeferralCallWrites<B, F> {
    // Output key for raw output + its length in bytes. These bytes are written as one
    // contiguous heap write, with layout [output_commit || output_len_le]. Note output_len
    // **must** be divisible by DIGEST_SIZE.
    pub output_commit: [B; COMMIT_NUM_BYTES],
    pub output_len: [B; F_NUM_BYTES],

    // Deferral address space accumulators after incorporating the current deferral call
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

        // Range check heap byte decompositions used by canonicity and packed memory ops.
        for bytes in cols.reads.input_commit.chunks_exact(2) {
            self.bitwise_bus
                .send_range(bytes[0], bytes[1])
                .eval(builder, cols.is_valid);
        }
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

    // Carries for converting the heap `input`/`output` base *byte* pointers to AS-native u16
    // *cell* pointer limbs.
    pub input_cell_carry: T,
    pub output_cell_carry: T,
    // Per-block carries for adding the cell offset `chunk_idx * (MEMORY_BLOCK_BYTES /
    // U16_CELL_SIZE)` to each base cell pointer.
    pub input_commit_add_carry: [T; COMMIT_MEMORY_OPS],
    pub output_add_carry: [T; OUTPUT_TOTAL_MEMORY_OPS],
    // The DEFERRAL_AS accumulator cell pointers need no limb decomposition or add-carry columns:
    // they are bounded below 2^16 (see the static assert in `super`), so the high cell limb is
    // identically zero and the low limb is the algebraic cell pointer.
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(DeferralCallAdapterCols<u8>)]
pub struct DeferralCallAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_bus: BitwiseOperationLookupBus,
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

        // Build full 8-element data arrays with upper 4 limbs hardcoded to zero
        let rd_full = expand_to_rv64_register(&cols.rd_val);
        let rs_full = expand_to_rv64_register(&cols.rs_val);

        // Heap pointers are first read from their respective registers. Register byte pointers are
        // small: `ptr / 2` in the low cell limb, high cell limb zero.
        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), reg_byte_ptr_to_cell_ptr_limbs::<AB>(cols.rd_ptr)),
                pack_u8_block::<AB>(&rd_full),
                timestamp_pp(),
                &cols.rd_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), reg_byte_ptr_to_cell_ptr_limbs::<AB>(cols.rs_ptr)),
                pack_u8_block::<AB>(&rs_full),
                timestamp_pp(),
                &cols.rs_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // We range check the top byte of both heap pointers to ensure that each
        // access is in [0, 2^address_bits). The memory merkle argument ensures
        // that each read/write pointer is less than 2^addr_bits, and this range
        // check ensures the accesses don't wrap around P.
        debug_assert!(RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS >= self.address_bits);
        let limb_shift =
            AB::F::from_usize(1 << (RV64_BYTE_BITS * RV64_WORD_NUM_LIMBS - self.address_bits));

        self.bitwise_bus
            .send_range(
                cols.rd_val[RV64_WORD_NUM_LIMBS - 1] * limb_shift,
                cols.rs_val[RV64_WORD_NUM_LIMBS - 1] * limb_shift,
            )
            .eval(builder, ctx.instruction.is_valid.clone());
        for val in [&cols.rd_val, &cols.rs_val] {
            for bytes in val.chunks_exact(2) {
                self.bitwise_bus
                    .send_range(bytes[0], bytes[1])
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Convert the heap `input`/`output` base *byte* pointers (read from registers) into
        // AS-native u16 *cell* pointer limbs `[cell_lo, cell_hi]`. The byte pointers are
        // little-endian 16-bit limbs packed from the low 4 register bytes.
        let input_byte_limbs: [AB::Expr; 2] = [
            pack_u8_pair(cols.rs_val[0].into(), cols.rs_val[1].into()),
            pack_u8_pair(cols.rs_val[2].into(), cols.rs_val[3].into()),
        ];
        let input_base_cell = eval_byte_ptr_limbs_to_u16_cell_ptr_limbs::<AB>(
            builder,
            self.range_bus,
            input_byte_limbs,
            cols.input_cell_carry,
            self.address_bits,
            ctx.instruction.is_valid.clone(),
        );
        let output_byte_limbs: [AB::Expr; 2] = [
            pack_u8_pair(cols.rd_val[0].into(), cols.rd_val[1].into()),
            pack_u8_pair(cols.rd_val[2].into(), cols.rd_val[3].into()),
        ];
        let output_base_cell = eval_byte_ptr_limbs_to_u16_cell_ptr_limbs::<AB>(
            builder,
            self.range_bus,
            output_byte_limbs,
            cols.output_cell_carry,
            self.address_bits,
            ctx.instruction.is_valid.clone(),
        );

        // Cell offset (in u16 cells) between consecutive heap blocks.
        let heap_cell_stride = (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32;

        // Accumulators are read then updated in the deferral address space,
        // using deferral_idx (instruction immediate / operand c) to determine
        // the accumulator memory address.
        let deferral_idx = ctx.instruction.immediate;
        let deferral_as = AB::Expr::from_u32(DEFERRAL_AS);

        // Accumulators are consecutive DEFERRAL_AS *cell* ranges. The base accumulator pointer
        // `NUM_ACCUMULATORS_PER_IDX * deferral_idx * DIGEST_SIZE` and every accumulator cell access
        // are bounded below 2^16 because the count bus constrains `deferral_idx < MAX_DEF_CIRCUITS`
        // (see the static assert in `super`). The pointer therefore fits entirely in the low cell
        // limb with a high limb of zero, so — unlike the heap pointers — no limb decomposition,
        // range checks, or add carries are needed: each cell pointer is just `[base + offset, 0]`.
        let acc_base_ptr =
            deferral_idx.clone() * AB::Expr::from_usize(NUM_ACCUMULATORS_PER_IDX * DIGEST_SIZE);
        let acc_cell_ptr = |offset: usize| -> [AB::Expr; 2] {
            [
                acc_base_ptr.clone() + AB::Expr::from_usize(offset),
                AB::Expr::ZERO,
            ]
        };

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
                output_len[F_NUM_BYTES - 1].clone() * limb_shift,
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
            split_byte_memory_ops::<_, COMMIT_NUM_BYTES, COMMIT_MEMORY_OPS>(input_commit);
        for (chunk_idx, (data, aux, carry)) in izip!(
            input_commit_chunks,
            &cols.input_commit_aux,
            &cols.input_commit_add_carry
        )
        .enumerate()
        {
            let block_cell_ptr = eval_add_const_u16_limbs::<AB>(
                builder,
                self.range_bus,
                input_base_cell.clone(),
                chunk_idx as u32 * heap_cell_stride,
                *carry,
                ctx.instruction.is_valid.clone(),
            );
            self.memory_bridge
                .read(
                    MemoryAddress::new(e.clone(), block_cell_ptr),
                    pack_u8_block::<AB>(&data),
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
                        acc_cell_ptr(chunk_idx * BLOCK_FE_WIDTH),
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
                        acc_cell_ptr(DIGEST_SIZE + chunk_idx * BLOCK_FE_WIDTH),
                    ),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let output_commit_and_len = combine_output(output_commit, output_len_full);
        let output_commit_and_len_chunks =
            split_byte_memory_ops::<_, OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS>(
                output_commit_and_len,
            );
        for (chunk_idx, (data, aux, carry)) in izip!(
            output_commit_and_len_chunks,
            &cols.output_commit_and_len_aux,
            &cols.output_add_carry
        )
        .enumerate()
        {
            let block_cell_ptr = eval_add_const_u16_limbs::<AB>(
                builder,
                self.range_bus,
                output_base_cell.clone(),
                chunk_idx as u32 * heap_cell_stride,
                *carry,
                ctx.instruction.is_valid.clone(),
            );
            self.memory_bridge
                .write(
                    MemoryAddress::new(e.clone(), block_cell_ptr),
                    pack_u8_block::<AB>(&data),
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
                        acc_cell_ptr(chunk_idx * BLOCK_FE_WIDTH),
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
                        acc_cell_ptr(DIGEST_SIZE + chunk_idx * BLOCK_FE_WIDTH),
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
