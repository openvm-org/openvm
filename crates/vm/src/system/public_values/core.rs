use std::sync::Mutex;

use openvm_circuit_primitives::{encoder::Encoder, AlignedBytesBorrow, SubAir};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    LocalOpcode,
    PublishOpcode::{self, PUBLISH},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};

use crate::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        get_record_from_slice, AdapterAirContext, AdapterExecutorE1, AdapterTraceFiller,
        AdapterTraceStep, BasicAdapterInterface, EmptyAdapterCoreLayout, InsExecutorE1,
        InstructionExecutor, MinimalInstruction, RecordArena, Result, TraceFiller, TraceStep,
        VmCoreAir, VmStateMut,
    },
    system::{
        memory::{
            online::{GuestMemory, TracingMemory},
            MemoryAuxColsFactory,
        },
        native_adapter::NativeAdapterStep,
        public_values::columns::PublicValuesCoreColsView,
    },
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
pub struct PublicValuesStep<F, A = NativeAdapterStep<F, 2, 0>> {
    adapter: A,
    encoder: Encoder,
    // Mutex is to make the struct Sync. But it actually won't be accessed by multiple threads.
    pub(crate) custom_pvs: Mutex<Vec<Option<F>>>,
}

impl<F: Clone, A> PublicValuesStep<F, A> {
    /// **Note:** `max_degree` is the maximum degree of the constraint polynomials to represent the
    /// flags. If you want the overall AIR's constraint degree to be `<= max_constraint_degree`,
    /// then typically you should set `max_degree` to `max_constraint_degree - 1`.
    pub fn new(adapter: A, num_custom_pvs: usize, max_degree: u32) -> Self {
        Self {
            adapter,
            encoder: Encoder::new(num_custom_pvs, max_degree, true),
            custom_pvs: Mutex::new(vec![None; num_custom_pvs]),
        }
    }

    pub fn set_public_values(&mut self, public_values: &[F]) {
        let mut custom_pvs = self.custom_pvs.lock().unwrap();
        assert_eq!(public_values.len(), custom_pvs.len());
        for (pv_mut, value) in custom_pvs.iter_mut().zip(public_values) {
            *pv_mut = Some(value.clone());
        }
    }
}

// We clone when we want to run a new instance of the program, so we reset the custom public values.
impl<F: Clone, A: Clone> Clone for PublicValuesStep<F, A> {
    fn clone(&self) -> Self {
        Self {
            adapter: self.adapter.clone(),
            encoder: self.encoder.clone(),
            custom_pvs: Mutex::new(vec![None; self.custom_pvs.lock().unwrap().len()]),
        }
    }
}

impl<F, A> TraceStep<F> for PublicValuesStep<F, A>
where
    F: 'static,
    A: 'static + AdapterTraceStep<F, ReadData = [[F; 1]; 2], WriteData = [[F; 1]; 0]>,
{
    type RecordLayout = EmptyAdapterCoreLayout<F, A>;
    type RecordMut<'a> = (A::RecordMut<'a>, &'a mut PublicValuesRecord<F>);
}

impl<F, A, RA> InstructionExecutor<F, RA> for PublicValuesStep<F, A>
where
    F: PrimeField32,
    A: 'static + Clone + AdapterTraceStep<F, ReadData = [[F; 1]; 2], WriteData = [[F; 1]; 0]>,
    for<'buf> RA: RecordArena<
        'buf,
        <Self as TraceStep<F>>::RecordLayout,
        <Self as TraceStep<F>>::RecordMut<'buf>,
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            PublishOpcode::from_usize(opcode - PublishOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<()> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [[core_record.value], [core_record.index]] =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);
        {
            let idx: usize = core_record.index.as_canonical_u32() as usize;
            let mut custom_pvs = self.custom_pvs.lock().unwrap();

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

impl<F, A> TraceFiller<F> for PublicValuesStep<F, A>
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
        self.custom_pvs
            .lock()
            .unwrap()
            .iter()
            .map(|&x| x.unwrap_or(F::ZERO))
            .collect()
    }
}

impl<F, A> InsExecutorE1<F> for PublicValuesStep<F, A>
where
    F: PrimeField32,
    A: 'static + for<'a> AdapterExecutorE1<F, ReadData = [F; 2], WriteData = [F; 0]>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let [value, index] = self.adapter.read(state, instruction);

        let idx: usize = index.as_canonical_u32() as usize;
        {
            let mut custom_pvs = self.custom_pvs.lock().unwrap();

            if custom_pvs[idx].is_none() {
                custom_pvs[idx] = Some(value);
            } else {
                // Not a hard constraint violation when publishing the same value twice but the
                // program should avoid that.
                panic!("Custom public value {} already set", idx);
            }
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}
