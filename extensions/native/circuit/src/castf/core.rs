use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, InsExecutorE1, MinimalInstruction, Result,
        VmAdapterInterface, VmCoreAir, VmCoreChip, VmExecutionState,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::CastfOpcode;
use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{Deserialize, Serialize};

// LIMB_BITS is the size of the limbs in bits.
pub(crate) const LIMB_BITS: usize = 8;
// the final limb has only 6 bits
pub(crate) const FINAL_LIMB_BITS: usize = 6;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct CastFCoreCols<T> {
    pub in_val: T,
    pub out_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug)]
pub struct CastFCoreAir {
    pub bus: VariableRangeCheckerBus, /* to communicate with the range checker that checks that
                                       * all limbs are < 2^LIMB_BITS */
}

impl<F: Field> BaseAir<F> for CastFCoreAir {
    fn width(&self) -> usize {
        CastFCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for CastFCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for CastFCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 1]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &CastFCoreCols<_> = local_core.borrow();

        builder.assert_bool(cols.is_valid);

        let intermed_val = cols
            .out_val
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &limb)| {
                acc + limb * AB::Expr::from_canonical_u32(1 << (i * LIMB_BITS))
            });

        for i in 0..4 {
            self.bus
                .range_check(
                    cols.out_val[i],
                    match i {
                        0..=2 => LIMB_BITS,
                        3 => FINAL_LIMB_BITS,
                        _ => unreachable!(),
                    },
                )
                .eval(builder, cols.is_valid);
        }

        AdapterAirContext {
            to_pc: None,
            reads: [[intermed_val]].into(),
            writes: [cols.out_val.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
                opcode: AB::Expr::from_canonical_usize(
                    CastfOpcode::CASTF.global_opcode().as_usize(),
                ),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        CastfOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub struct CastFRecord<F> {
    pub in_val: F,
    pub out_val: [u32; RV32_REGISTER_NUM_LIMBS],
}

pub struct CastFStep<A> {
    pub air: CastFCoreAir,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
    phantom: PhantomData<A>,
}

impl CastFStep<A> {
    pub fn new(range_checker_chip: SharedVariableRangeCheckerChip) -> Self {
        Self {
            air: CastFCoreAir {
                bus: range_checker_chip.bus(),
            },
            range_checker_chip,
            phantom: PhantomData,
        }
    }
}

impl<A> CastFStep<A> {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
        offset: usize,
    ) -> Self {
        Self {
            air: BaseAluCoreAir {
                bus: bitwise_lookup_chip.bus(),
                offset,
            },
            bitwise_lookup_chip,
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn execute_core<F: PrimeField32>(
        &self,
        instruction: &Instruction<F>,
        [x, y]: [[u8; NUM_LIMBS]; 2],
        core_row: &mut [F],
    ) -> [u8; NUM_LIMBS] {
        let opcode = instruction.opcode;
        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        let z = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &x, &y);
        println!("{local_opcode:?} {x:?}, {y:?}: {z:?}");

        let core_row: &mut BaseAluCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();
        core_row.a = z.map(F::from_canonical_u8);
        core_row.b = x.map(F::from_canonical_u8);
        core_row.c = y.map(F::from_canonical_u8);
        core_row.opcode_add_flag = F::from_bool(local_opcode == BaseAluOpcode::ADD);
        core_row.opcode_sub_flag = F::from_bool(local_opcode == BaseAluOpcode::SUB);
        core_row.opcode_xor_flag = F::from_bool(local_opcode == BaseAluOpcode::XOR);
        core_row.opcode_or_flag = F::from_bool(local_opcode == BaseAluOpcode::OR);
        core_row.opcode_and_flag = F::from_bool(local_opcode == BaseAluOpcode::AND);

        z
    }

    pub fn fill_trace_row_core<F: PrimeField32>(&self, core_row: &mut [F]) {
        let core_row: &mut BaseAluCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        if core_row.opcode_add_flag == F::ONE || core_row.opcode_sub_flag == F::ONE {
            for a_val in core_row.a.map(|x| x.as_canonical_u32()) {
                self.bitwise_lookup_chip.request_xor(a_val, a_val);
            }
        } else {
            let b = core_row.b.map(|x| x.as_canonical_u32());
            let c = core_row.c.map(|x| x.as_canonical_u32());
            for (b_val, c_val) in zip(b, c) {
                self.bitwise_lookup_chip.request_xor(b_val, c_val);
            }
        }
    }
}

impl<F, CTX, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> SingleTraceStep<F, CTX>
    for BaseAluStep<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData = [[u8; NUM_LIMBS]; 2],
            WriteData = [u8; NUM_LIMBS],
            TraceContext<'a> = &'a BitwiseOperationLookupChip<LIMB_BITS>,
        >,
{
    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory, CTX>,
        instruction: &Instruction<F>,
        row_slice: &mut [F],
    ) -> Result<()> {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        A::start(*state.pc, state.memory, adapter_row);
        let [rs1, rs2] = A::read(state.memory, instruction, adapter_row);
        let output = self.execute_core(instruction, [rs1, rs2], core_row);
        A::write(state.memory, instruction, adapter_row, &output);

        *state.pc += DEFAULT_PC_STEP;
        Ok(())
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BaseAluOpcode::from_usize(opcode - self.air.offset))
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        A::fill_trace_row(mem_helper, self.bitwise_lookup_chip.as_ref(), adapter_row);
        self.fill_trace_row_core(core_row);
    }
}

impl<Mem, Ctx, F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> StepExecutorE1<Mem, Ctx, F>
    for BaseAluStep<A, NUM_LIMBS, LIMB_BITS>
where
    Mem: GuestMemory,
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<
            Mem,
            F,
            ReadData = [[u8; NUM_LIMBS]; 2],
            WriteData = [u8; NUM_LIMBS],
        >,
{
    fn execute_e1(
        &mut self,
        state: &mut VmExecutionState<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = instruction;

        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        let [rs1_bytes, rs2_bytes] = A::read(&mut state.memory, instruction);

        let rd_bytes = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &rs1_bytes, &rs2_bytes);

        A::write(&mut state.memory, instruction, &rd_bytes);

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, I> VmCoreChip<F, I> for CastFCoreChip
where
    F: PrimeField32,
    I: VmAdapterInterface<F>,
    I::Reads: Into<[[F; 1]; 1]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = CastFRecord<F>;
    type Air = CastFCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;

        assert_eq!(
            opcode.local_opcode_idx(CastfOpcode::CLASS_OFFSET),
            CastfOpcode::CASTF as usize
        );

        let y = reads.into()[0][0];
        let x = CastF::solve(y.as_canonical_u32());

        let output = AdapterRuntimeContext {
            to_pc: None,
            writes: [x.map(F::from_canonical_u32)].into(),
        };

        let record = CastFRecord {
            in_val: y,
            out_val: x,
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", CastfOpcode::CASTF)
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        for (i, limb) in record.out_val.iter().enumerate() {
            if i == 3 {
                self.range_checker_chip.add_count(*limb, FINAL_LIMB_BITS);
            } else {
                self.range_checker_chip.add_count(*limb, LIMB_BITS);
            }
        }

        let cols: &mut CastFCoreCols<F> = row_slice.borrow_mut();
        cols.in_val = record.in_val;
        cols.out_val = record.out_val.map(F::from_canonical_u32);
        cols.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

impl<Mem, Ctx, F> InsExecutorE1<Mem, Ctx, F> for CastFCoreChip
where
    Mem: GuestMemory,
    F: PrimeField32,
{
    fn execute_e1(
        &mut self,
        state: &mut VmExecutionState<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()> {
        let Instruction {
            opcode, a, b, d, e, ..
        } = instruction;

        assert_eq!(
            opcode.local_opcode_idx(CastfOpcode::CLASS_OFFSET),
            CastfOpcode::CASTF as usize
        );

        // TODO(ayush): check if can be read directly as [u8; 4] or u32?
        let [y]: [F; 1] = unsafe {
            state
                .memory
                .read(e.as_canonical_u32(), b.as_canonical_u32())
        };
        let x = CastF::solve(y.as_canonical_u32());
        let x = x.map(F::from_canonical_u32);

        unsafe {
            state
                .memory
                .write::<F, 4>(d.as_canonical_u32(), a.as_canonical_u32(), &x);
        };

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

pub struct CastF;
impl CastF {
    pub(super) fn solve(y: u32) -> [u32; RV32_REGISTER_NUM_LIMBS] {
        let mut x = [0; 4];
        for (i, limb) in x.iter_mut().enumerate() {
            *limb = (y >> (8 * i)) & 0xFF;
        }
        x
    }
}
