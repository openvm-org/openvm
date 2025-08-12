use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use getset::Setters;
use openvm_circuit_primitives::{encoder::Encoder, AlignedBytesBorrow, SubAir};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::RV32_IMM_AS,
    LocalOpcode,
    PublishOpcode::{self, PUBLISH},
    NATIVE_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::{
    arch::{
        execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait},
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, E2PreCompute, EmptyAdapterCoreLayout, ExecuteFunc, ExecutionError,
        Executor, MeteredExecutor, MinimalInstruction, PreflightExecutor, RecordArena,
        StaticProgramError, TraceFiller, VmCoreAir, VmExecState, VmStateMut,
    },
    system::{
        memory::{
            online::{GuestMemory, TracingMemory},
            MemoryAuxColsFactory,
        },
        native_adapter::NativeAdapterExecutor,
        public_values::columns::PublicValuesCoreColsView,
    },
    utils::{transmute_field_to_u32, transmute_u32_to_field},
};

pub(crate) type AdapterInterface<F> = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 0, 1, 1>;

#[derive(Clone, Debug)]
pub struct PublicValuesCoreAir {
    /// Number of custom public values to publish.
    pub num_custom_pvs: usize,
    encoder: Encoder,
}

impl PublicValuesCoreAir {
    pub fn new(num_custom_pvs: usize, max_degree: u32) -> Self {
        Self {
            num_custom_pvs,
            encoder: Encoder::new(num_custom_pvs, max_degree, true),
        }
    }
}

impl<F: Field> BaseAir<F> for PublicValuesCoreAir {
    fn width(&self) -> usize {
        3 + self.encoder.width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for PublicValuesCoreAir {
    fn num_public_values(&self) -> usize {
        self.num_custom_pvs
    }
}

impl<AB: InteractionBuilder + AirBuilderWithPublicValues> VmCoreAir<AB, AdapterInterface<AB::Expr>>
    for PublicValuesCoreAir
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, AdapterInterface<AB::Expr>> {
        let cols = PublicValuesCoreColsView::<_, &AB::Var>::borrow(local_core);
        debug_assert_eq!(cols.width(), BaseAir::<AB::F>::width(self));
        let is_valid = *cols.is_valid;
        let value = *cols.value;
        let index = *cols.index;

        let vars = cols.custom_pv_vars.iter().map(|&&x| x).collect::<Vec<_>>();
        self.encoder.eval(builder, &vars);

        let flags = self.encoder.flags::<AB>(&vars);

        let mut match_public_value_index = AB::Expr::ZERO;
        let mut match_public_value = AB::Expr::ZERO;
        for (i, flag) in flags.iter().enumerate() {
            match_public_value_index += flag.clone() * AB::F::from_canonical_usize(i);
            match_public_value += flag.clone() * builder.public_values()[i].into();
        }
        builder.assert_eq(is_valid, self.encoder.is_valid::<AB>(&vars));

        let mut when_publish = builder.when(is_valid);
        when_publish.assert_eq(index, match_public_value_index);
        when_publish.assert_eq(value, match_public_value);

        AdapterAirContext {
            to_pc: None,
            reads: [[value.into()], [index.into()]],
            writes: [],
            instruction: MinimalInstruction {
                is_valid: is_valid.into(),
                opcode: AB::Expr::from_canonical_usize(PUBLISH.global_opcode().as_usize()),
            },
        }
    }

    fn start_offset(&self) -> usize {
        PublishOpcode::CLASS_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct PublicValuesRecord<F> {
    pub value: F,
    pub index: F,
}

/// ATTENTION: If a specific public value is not provided, a default 0 will be used when generating
/// the proof but in the perspective of constraints, it could be any value.
#[derive(Clone)]
pub struct PublicValuesExecutor<F, A = NativeAdapterExecutor<F, 2, 0>> {
    adapter: A,
    phantom: PhantomData<F>,
}

#[derive(Clone, Setters)]
pub struct PublicValuesFiller<F, A = NativeAdapterExecutor<F, 2, 0>> {
    adapter: A,
    encoder: Encoder,
    num_custom_pvs: usize,
    public_values: Vec<F>,
}

impl<F: Clone, A> PublicValuesExecutor<F, A> {
    pub fn new(adapter: A) -> Self {
        Self {
            adapter,
            phantom: PhantomData,
        }
    }
}

impl<F: Clone, A> PublicValuesFiller<F, A> {
    /// **Note:** `max_degree` is the maximum degree of the constraint polynomials to represent the
    /// flags. If you want the overall AIR's constraint degree to be `<= max_constraint_degree`,
    /// then typically you should set `max_degree` to `max_constraint_degree - 1`.
    pub fn new(adapter: A, num_custom_pvs: usize, max_degree: u32) -> Self {
        Self {
            adapter,
            encoder: Encoder::new(num_custom_pvs, max_degree, true),
            num_custom_pvs,
            public_values: Vec::new(),
        }
    }

    pub fn set_public_values(&mut self, public_values: Vec<F>)
    where
        F: Field,
    {
        assert_eq!(public_values.len(), self.num_custom_pvs);
        self.public_values = public_values;
    }
}

impl<F, A, RA> PreflightExecutor<F, RA> for PublicValuesExecutor<F, A>
where
    F: PrimeField32,
    A: 'static + Clone + AdapterTraceExecutor<F, ReadData = [[F; 1]; 2], WriteData = [[F; 1]; 0]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut PublicValuesRecord<F>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            PublishOpcode::from_usize(opcode - PublishOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [[core_record.value], [core_record.index]] =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);
        {
            let idx: usize = core_record.index.as_canonical_u32() as usize;
            let custom_pvs = state.custom_pvs;

            if custom_pvs[idx].is_none() {
                custom_pvs[idx] = Some(core_record.value);
            } else {
                // Not a hard constraint violation when publishing the same value twice but the
                // program should avoid that.
                panic!("Custom public value {} already set", idx);
            }
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A> TraceFiller<F> for PublicValuesFiller<F, A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &PublicValuesRecord<F> = unsafe { get_record_from_slice(&mut core_row, ()) };
        let cols = PublicValuesCoreColsView::<_, &mut F>::borrow_mut(core_row);

        let idx: usize = record.index.as_canonical_u32() as usize;
        let pt = self.encoder.get_flag_pt(idx);

        cols.custom_pv_vars
            .into_iter()
            .zip(pt.iter())
            .for_each(|(var, &val)| {
                *var = F::from_canonical_u32(val);
            });

        *cols.index = record.index;
        *cols.value = record.value;
        *cols.is_valid = F::ONE;
    }

    fn generate_public_values(&self) -> Vec<F> {
        assert_eq!(
            self.public_values.len(),
            self.num_custom_pvs,
            "Did not set public values"
        );
        self.public_values.clone()
    }
}

#[derive(AlignedBytesBorrow)]
#[repr(C)]
struct PublicValuesPreCompute {
    b_or_imm: u32,
    c_or_imm: u32,
}

impl<F, A> Executor<F> for PublicValuesExecutor<F, A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<PublicValuesPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut PublicValuesPreCompute = data.borrow_mut();
        let (b_is_imm, c_is_imm) = self.pre_compute_impl(inst, data);

        let fn_ptr = match (b_is_imm, c_is_imm) {
            (true, true) => execute_e1_impl::<_, _, true, true>,
            (true, false) => execute_e1_impl::<_, _, true, false>,
            (false, true) => execute_e1_impl::<_, _, false, true>,
            (false, false) => execute_e1_impl::<_, _, false, false>,
        };
        Ok(fn_ptr)
    }
}

impl<F, A> MeteredExecutor<F> for PublicValuesExecutor<F, A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<PublicValuesPreCompute>>()
    }

    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<PublicValuesPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (b_is_imm, c_is_imm) = self.pre_compute_impl(inst, &mut data.data);

        let fn_ptr = match (b_is_imm, c_is_imm) {
            (true, true) => execute_e2_impl::<_, _, true, true>,
            (true, false) => execute_e2_impl::<_, _, true, false>,
            (false, true) => execute_e2_impl::<_, _, false, true>,
            (false, false) => execute_e2_impl::<_, _, false, false>,
        };
        Ok(fn_ptr)
    }
}

#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX, const B_IS_IMM: bool, const C_IS_IMM: bool>(
    pre_compute: &[u8],
    state: &mut VmExecState<F, GuestMemory, CTX>,
) where
    CTX: ExecutionCtxTrait,
{
    let pre_compute: &PublicValuesPreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, B_IS_IMM, C_IS_IMM>(pre_compute, state);
}

#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX, const B_IS_IMM: bool, const C_IS_IMM: bool>(
    pre_compute: &[u8],
    state: &mut VmExecState<F, GuestMemory, CTX>,
) where
    CTX: MeteredExecutionCtxTrait,
{
    let pre_compute: &E2PreCompute<PublicValuesPreCompute> = pre_compute.borrow();
    state.ctx.on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, B_IS_IMM, C_IS_IMM>(&pre_compute.data, state);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX, const B_IS_IMM: bool, const C_IS_IMM: bool>(
    pre_compute: &PublicValuesPreCompute,
    state: &mut VmExecState<F, GuestMemory, CTX>,
) where
    CTX: ExecutionCtxTrait,
{
    let value = if B_IS_IMM {
        transmute_u32_to_field(&pre_compute.b_or_imm)
    } else {
        state.vm_read::<F, 1>(NATIVE_AS, pre_compute.b_or_imm)[0]
    };
    let index = if C_IS_IMM {
        transmute_u32_to_field(&pre_compute.c_or_imm)
    } else {
        state.vm_read::<F, 1>(NATIVE_AS, pre_compute.c_or_imm)[0]
    };

    let idx: usize = index.as_canonical_u32() as usize;
    {
        let custom_pvs = &mut state.vm_state.custom_pvs;

        if custom_pvs[idx].is_none() {
            custom_pvs[idx] = Some(value);
        } else {
            // Not a hard constraint violation when publishing the same value twice but the
            // program should avoid that.
            panic!("Custom public value {} already set", idx);
        }
    }
    state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
    state.instret += 1;
}

impl<F, A> PublicValuesExecutor<F, A>
where
    F: PrimeField32,
{
    fn pre_compute_impl(
        &self,
        inst: &Instruction<F>,
        data: &mut PublicValuesPreCompute,
    ) -> (bool, bool) {
        let &Instruction { b, c, e, f, .. } = inst;

        let e = e.as_canonical_u32();
        let f = f.as_canonical_u32();

        let b_is_imm = e == RV32_IMM_AS;
        let c_is_imm = f == RV32_IMM_AS;

        let b_or_imm = if b_is_imm {
            transmute_field_to_u32(&b)
        } else {
            b.as_canonical_u32()
        };
        let c_or_imm = if c_is_imm {
            transmute_field_to_u32(&c)
        } else {
            c.as_canonical_u32()
        };

        *data = PublicValuesPreCompute { b_or_imm, c_or_imm };

        (b_is_imm, c_is_imm)
    }
}
