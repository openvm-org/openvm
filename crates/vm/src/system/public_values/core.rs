use std::sync::{Arc, Mutex};

use openvm_circuit_primitives::{encoder::Encoder, AlignedBytesBorrow, SubAir};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    LocalOpcode,
    PublishOpcode::{self, PUBLISH},
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::{ParallelIterator, ParallelSliceMut},
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues},
    AirRef, Chip, ChipUsageGetter,
};

use crate::{
    arch::{
        execution_mode::{metered::MeteredCtx, tracegen::TracegenCtx, E1E2ExecutionCtx},
        get_record_from_slice, AdapterAirContext, AdapterExecutorE1, AdapterTraceFiller,
        AdapterTraceStep, BasicAdapterInterface, EmptyLayout, ExecutionState, InsExecutor,
        InsExecutorE1, InstructionExecutor, MinimalInstruction, RecordArena, Result, Streams,
        VmAirWrapper, VmCoreAir, VmStateMut,
    },
    system::{
        memory::{
            online::{GuestMemory, TracingMemory},
            MemoryAuxColsFactory, MemoryController,
        },
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
pub struct PublicValuesCoreStep<AdapterAir, AdapterStep, F> {
    air: VmAirWrapper<AdapterAir, PublicValuesCoreAir>,
    adapter: AdapterStep,
    // TODO(ayush): put air here and take from air
    // Mutex is to make the struct Sync. But it actually won't be accessed by multiple threads.
    pub(crate) custom_pvs: Mutex<Vec<Option<F>>>,
}

impl<AdapterAir, AdapterStep, F> PublicValuesCoreStep<AdapterAir, AdapterStep, F>
where
    F: PrimeField32,
    AdapterStep: 'static + AdapterTraceFiller<F>,
{
    /// **Note:** `max_degree` is the maximum degree of the constraint polynomials to represent the
    /// flags. If you want the overall AIR's constraint degree to be `<= max_constraint_degree`,
    /// then typically you should set `max_degree` to `max_constraint_degree - 1`.
    pub fn new(
        adapter_air: AdapterAir,
        adapter_step: AdapterStep,
        num_custom_pvs: usize,
        max_degree: u32,
    ) -> Self {
        Self {
            air: VmAirWrapper::new(
                adapter_air,
                PublicValuesCoreAir::new(num_custom_pvs, max_degree),
            ),
            adapter: adapter_step,
            custom_pvs: Mutex::new(vec![None; num_custom_pvs]),
        }
    }

    pub fn get_custom_public_values(&self) -> Vec<Option<F>> {
        self.custom_pvs.lock().unwrap().clone()
    }

    pub fn generate_public_values(&self) -> Vec<F> {
        self.get_custom_public_values()
            .into_iter()
            .map(|x| x.unwrap_or(F::ZERO))
            .collect()
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) =
            unsafe { row_slice.split_at_mut_unchecked(AdapterStep::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &PublicValuesRecord<F> = unsafe { get_record_from_slice(&mut core_row, ()) };
        let cols = PublicValuesCoreColsView::<_, &mut F>::borrow_mut(core_row);

        let idx: usize = record.index.as_canonical_u32() as usize;
        let pt = self.air.core.encoder.get_flag_pt(idx);

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

    fn fill_dummy_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, _row_slice: &mut [F]) {
        // By default, the row is filled with zeroes
    }

    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) where
        Self: Send + Sync,
        F: Send + Sync + Clone,
    {
        let width = trace.width();
        trace.values[..rows_used * width]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_trace_row(mem_helper, row_slice);
            });
        trace.values[rows_used * width..]
            .par_chunks_exact_mut(width)
            .for_each(|row_slice| {
                self.fill_dummy_trace_row(mem_helper, row_slice);
            });
    }
}

impl<AdapterAir, AdapterStep, F> InstructionExecutor<F>
    for PublicValuesCoreStep<AdapterAir, AdapterStep, F>
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            PublishOpcode::from_usize(opcode - PublishOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &mut self,
        _memory: &mut MemoryController<F>,
        _streams: &mut Streams<F>,
        _instruction: &Instruction<F>,
        _from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>> {
        unimplemented!()
    }
}

impl<AdapterAir, AdapterStep, F> InsExecutorE1<F>
    for PublicValuesCoreStep<AdapterAir, AdapterStep, F>
where
    F: PrimeField32,
    AdapterStep: 'static + for<'a> AdapterExecutorE1<F, ReadData = [F; 2], WriteData = [F; 0]>,
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

impl<AdapterAir, AdapterStep, F, RA> InsExecutor<F, RA>
    for PublicValuesCoreStep<AdapterAir, AdapterStep, F>
where
    F: PrimeField32,
    AdapterStep: AdapterTraceStep<F, ReadData = [[F; 1]; 2], WriteData = [[F; 1]; 0]> + 'static,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyLayout<AdapterStep>,
        (
            AdapterStep::RecordMut<'buf>,
            &'buf mut PublicValuesRecord<F>,
        ),
    >,
{
    fn execute_tracegen(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, TracegenCtx<RA>>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()>
    where
        F: PrimeField32,
    {
        let arena = &mut state.ctx.arenas[chip_index];
        let (mut adapter_record, core_record) = arena.alloc(EmptyLayout::new());

        AdapterStep::start(*state.pc, state.memory, &mut adapter_record);

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

impl<AdapterAir, AdapterStep, F> ChipUsageGetter
    for PublicValuesCoreStep<AdapterAir, AdapterStep, F>
where
    F: Field,
    AdapterAir: BaseAir<F>,
{
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn trace_width(&self) -> usize {
        BaseAir::width(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        // TODO(ayush): fix this
        // unimplemented!()
        0
    }
}

impl<SC, AdapterAir, AdapterStep> Chip<SC>
    for PublicValuesCoreStep<AdapterAir, AdapterStep, Val<SC>>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    AdapterAir: BaseAir<Val<SC>>,
    VmAirWrapper<AdapterAir, PublicValuesCoreAir>: Clone + AnyRap<SC> + 'static,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        unimplemented!("generate_air_proof_input isn't implemented")
    }

    // fn generate_air_proof_input_with_trace(
    //     self,
    //     mut trace: RowMajorMatrix<F>,
    // ) -> AirProofInput<SC> {
    //     assert!(
    //         trace.height().is_power_of_two(),
    //         "Trace height must be a power of two"
    //     );
    //     self.fill_trace(&mut trace.values);

    //     let public_values = self.generate_public_values();
    //     AirProofInput::simple(trace, public_values)
    // }
}
