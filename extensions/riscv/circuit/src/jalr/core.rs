use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64JalrOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::adapters::{
    Rv64JalrAdapterExecutor, Rv64JalrAdapterFiller, RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS,
    WORD_NUM_LIMBS,
};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct Rv64JalrCoreCols<T> {
    pub imm: T,
    // Keep the same 32-bit decomposition columns and zero-extend to RV64 at the adapter boundary.
    pub rs1_data: [T; WORD_NUM_LIMBS],
    // To save a column, we only store the 3 most significant limbs of low-32 rd_data.
    // The least significant limb can be derived from from_pc and these limbs.
    pub rd_data: [T; WORD_NUM_LIMBS - 1],
    pub is_valid: T,

    pub to_pc_least_sig_bit: T,
    /// These are the limbs of `to_pc * 2`.
    pub to_pc_limbs: [T; 2],
    pub imm_sign: T,
}

#[derive(Debug, Clone, derive_new::new)]
pub struct Rv64JalrCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64JalrCoreAir {
    fn width(&self) -> usize {
        Rv64JalrCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv64JalrCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv64JalrCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV64_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; RV64_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<SignedImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv64JalrCoreCols<AB::Var> = (*local_core).borrow();
        let Rv64JalrCoreCols::<AB::Var> {
            imm,
            rs1_data: rs1,
            rd_data: rd,
            is_valid,
            imm_sign,
            to_pc_least_sig_bit,
            to_pc_limbs,
        } = *cols;

        builder.assert_bool(is_valid);

        // composed is the composition of 3 most significant limbs of low-32 rd.
        let composed = rd
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << ((i + 1) * RV64_CELL_BITS))
            });

        let least_sig_limb = from_pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP) - composed;

        // low-32 rd_data decomposition.
        let rd_data_low: [AB::Expr; WORD_NUM_LIMBS] = array::from_fn(|i| {
            if i == 0 {
                least_sig_limb.clone()
            } else {
                rd[i - 1].into().clone()
            }
        });

        // Constrain rd_data_low.
        // Assumes only from_pc in [0,2^PC_BITS) is allowed by program bus.
        self.bitwise_lookup_bus
            .send_range(rd_data_low[0].clone(), rd_data_low[1].clone())
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data_low[2].clone(), RV64_CELL_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data_low[3].clone(), PC_BITS - RV64_CELL_BITS * 3)
            .eval(builder, is_valid);

        builder.assert_bool(imm_sign);

        // Constrain to_pc_least_sig_bit + 2 * to_pc_limbs = rs1 + imm as an i32 addition with 2
        // 16-bit limbs. RISC-V spec explicitly sets the least significant bit of `to_pc` to 0.
        let rs1_limbs_01 = rs1[0] + rs1[1] * AB::F::from_canonical_u32(1 << RV64_CELL_BITS);
        let rs1_limbs_23 = rs1[2] + rs1[3] * AB::F::from_canonical_u32(1 << RV64_CELL_BITS);
        let inv = AB::F::from_canonical_u32(1 << 16).inverse();

        builder.assert_bool(to_pc_least_sig_bit);
        let carry = (rs1_limbs_01 + imm - to_pc_limbs[0] * AB::F::TWO - to_pc_least_sig_bit) * inv;
        builder.when(is_valid).assert_bool(carry.clone());

        let imm_extend_limb = imm_sign * AB::F::from_canonical_u32((1 << 16) - 1);
        let carry = (rs1_limbs_23 + imm_extend_limb + carry - to_pc_limbs[1]) * inv;
        builder.when(is_valid).assert_bool(carry);

        // Preventing to_pc overflow.
        self.range_bus
            .range_check(to_pc_limbs[1], PC_BITS - 16)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(to_pc_limbs[0], 15)
            .eval(builder, is_valid);
        let to_pc =
            to_pc_limbs[0] * AB::F::TWO + to_pc_limbs[1] * AB::F::from_canonical_u32(1 << 16);

        // Zero-extend low-32 rs1/rd at the adapter interface.
        let rs1_data = array::from_fn(|i| {
            if i < WORD_NUM_LIMBS {
                rs1[i].into()
            } else {
                AB::Expr::ZERO
            }
        });
        let rd_data = array::from_fn(|i| {
            if i < WORD_NUM_LIMBS {
                rd_data_low[i].clone()
            } else {
                AB::Expr::ZERO
            }
        });

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, JALR);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [rs1_data].into(),
            writes: [rd_data].into(),
            instruction: SignedImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm.into(),
                imm_sign: imm_sign.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv64JalrOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64JalrCoreRecord {
    pub imm: u16,
    pub from_pc: u32,
    pub rs1_val: u32,
    pub imm_sign: bool,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64JalrExecutor<A = Rv64JalrAdapterExecutor> {
    adapter: A,
}

#[derive(Clone)]
pub struct Rv64JalrFiller<A = Rv64JalrAdapterFiller> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A> Rv64JalrFiller<A> {
    pub fn new(
        adapter: A,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        assert!(range_checker_chip.range_max_bits() >= 16);
        Self {
            adapter,
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F, A, RA> PreflightExecutor<F, RA> for Rv64JalrExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = [u8; RV64_REGISTER_NUM_LIMBS],
            WriteData = [u8; RV64_REGISTER_NUM_LIMBS],
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut Rv64JalrCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv64JalrOpcode::from_usize(opcode - Rv64JalrOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, c, g, .. } = *instruction;

        debug_assert_eq!(
            opcode.local_opcode_idx(Rv64JalrOpcode::CLASS_OFFSET),
            JALR as usize
        );

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let rs1_data = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);
        core_record.rs1_val =
            u32::from_le_bytes([rs1_data[0], rs1_data[1], rs1_data[2], rs1_data[3]]);

        core_record.imm = c.as_canonical_u32() as u16;
        core_record.imm_sign = g.is_one();
        core_record.from_pc = *state.pc;

        let (to_pc, rd_data) = run_jalr(
            core_record.from_pc,
            core_record.rs1_val,
            core_record.imm,
            core_record.imm_sign,
        );

        self.adapter
            .write(state.memory, instruction, rd_data, &mut adapter_record);

        // RISC-V spec explicitly sets the least significant bit of `to_pc` to 0.
        *state.pc = to_pc & !1;

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for Rv64JalrFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // Rv64JalrCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid Rv64JalrCoreRecord written by the executor
        // during trace generation
        let record: &Rv64JalrCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut Rv64JalrCoreCols<F> = core_row.borrow_mut();

        let (to_pc, rd_data) =
            run_jalr(record.from_pc, record.rs1_val, record.imm, record.imm_sign);
        let to_pc_limbs = [(to_pc & ((1 << 16) - 1)) >> 1, to_pc >> 16];
        self.range_checker_chip.add_count(to_pc_limbs[0], 15);
        self.range_checker_chip
            .add_count(to_pc_limbs[1], PC_BITS - 16);
        self.bitwise_lookup_chip
            .request_range(rd_data[0] as u32, rd_data[1] as u32);

        self.range_checker_chip
            .add_count(rd_data[2] as u32, RV64_CELL_BITS);
        self.range_checker_chip
            .add_count(rd_data[3] as u32, PC_BITS - RV64_CELL_BITS * 3);

        // Write in reverse order.
        core_row.imm_sign = F::from_bool(record.imm_sign);
        core_row.to_pc_limbs = to_pc_limbs.map(F::from_canonical_u32);
        core_row.to_pc_least_sig_bit = F::from_bool(to_pc & 1 == 1);
        // fill_trace_row is called only on valid rows.
        core_row.is_valid = F::ONE;
        core_row.rs1_data = record.rs1_val.to_le_bytes().map(F::from_canonical_u8);
        core_row
            .rd_data
            .iter_mut()
            .rev()
            .zip(rd_data[..WORD_NUM_LIMBS].iter().skip(1).rev())
            .for_each(|(dst, src)| {
                *dst = F::from_canonical_u8(*src);
            });
        core_row.imm = F::from_canonical_u16(record.imm);
    }
}

// returns (to_pc, rd_data)
#[inline(always)]
pub(super) fn run_jalr(
    pc: u32,
    rs1: u32,
    imm: u16,
    imm_sign: bool,
) -> (u32, [u8; RV64_REGISTER_NUM_LIMBS]) {
    let to_pc = rs1.wrapping_add(imm as u32 + (imm_sign as u32 * 0xffff0000));
    assert!(to_pc < (1 << PC_BITS));

    let mut rd_data = [0u8; RV64_REGISTER_NUM_LIMBS];
    rd_data[..WORD_NUM_LIMBS].copy_from_slice(&pc.wrapping_add(DEFAULT_PC_STEP).to_le_bytes());
    (to_pc, rd_data)
}
