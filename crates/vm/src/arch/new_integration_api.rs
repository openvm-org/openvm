use std::{
    borrow::Borrow,
    sync::{Arc, Mutex},
};

use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    air_builders::{debug::DebugConstraintBuilder, symbolic::SymbolicRapBuilder},
    config::{StarkGenericConfig, Val},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter, Stateful,
};
use serde::{de::DeserializeOwned, Serialize};

use super::{ExecutionState, InstructionExecutor, Result};
use crate::system::memory::{MemoryController, OfflineMemory};

pub trait VmAdapter<F>: BaseAir<F> + Clone {
    type ExecuteTx;

    type TraceTx<'tx>
    where
        Self: 'tx,
        F: 'tx;

    fn execute_tx(&self) -> Self::ExecuteTx;

    fn trace_tx<'a>(
        &self,
        memory: &'a OfflineMemory<F>,
        row_buffer: &'a mut [F],
        from_state: ExecutionState<u32>,
    ) -> Self::TraceTx<'a>;
}

// Notes[jpw]:
// - Separate trait from VmAdapter because it needs the AB generic
// - Everything in this trait is **not** performance sensitive because it is only used for keygen (we serialize symbolic constraints afterwards), whereas VmAdapter trait is performance sensitive for both execution and trace generation.
/// Trait to be implemented on a struct that has enough information to determine
/// the adapter row width.
pub trait VmAdapterAir<AB: AirBuilder>: BaseAir<AB::F> {
    type AirTx<'tx>
    where
        Self: 'tx,
        AB: 'tx;

    fn air_tx<'a>(&self, local_adapter: &'a [AB::Var]) -> Self::AirTx<'a>;
}

/// Trait to be implemented on primitive chip to integrate with the machine.
pub trait VmCoreChip<F, A: VmAdapter<F>> {
    /// Minimum data that must be recorded to be able to generate trace for one row.
    type Record: Send + Serialize + DeserializeOwned;
    type Air: BaseAir<F> + Clone;

    /// Returns `(to_pc, record)`.
    fn execute_instruction(
        &self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_pc: u32,
        tx: &mut A::ExecuteTx,
    ) -> Result<(u32, Self::Record)>;

    fn get_opcode_name(&self, opcode: usize) -> String;

    /// Populates `row_core` with values corresponding to `record`.
    /// The provided `row_core` will correspond to the core columns, and
    /// **does not** include the adapter columns.
    /// However this function does also generate the adapter trace
    /// through `tx`.
    ///
    /// This function will be called for each row in the trace which is being used, and all other
    /// rows in the trace will be filled with zeroes.
    fn generate_trace_row(&self, row_core: &mut [F], record: Self::Record, tx: &mut A::TraceTx<'_>);

    /// Returns a list of public values to publish.
    fn generate_public_values(&self) -> Vec<F> {
        vec![]
    }

    fn air(&self) -> &Self::Air;

    /// Finalize the trace, especially the padded rows if the all-zero rows don't satisfy the constraints.
    /// This is done **after** records are consumed and the trace matrix is generated.
    /// Most implementations should just leave the default implementation if padding with rows of all 0s satisfies the constraints.
    fn finalize(&self, _trace: &mut RowMajorMatrix<F>, _num_records: usize) {
        // do nothing by default
    }
}

/// The generic `TX` should be an `AirTx` type.
pub trait VmCoreAir<AB, TX>: BaseAirWithPublicValues<AB::F>
where
    AB: AirBuilder,
{
    fn eval(&self, builder: &mut AB, local_core: &[AB::Var], tx: &mut TX);

    /// The offset the opcodes by this chip start from.
    /// This is usually just `CorrespondingOpcode::CLASS_OFFSET`,
    /// but sometimes (for modular chips, for example) it also depends on something else.
    fn start_offset(&self) -> usize;

    fn start_offset_expr(&self) -> AB::Expr {
        AB::Expr::from_canonical_usize(self.start_offset())
    }

    fn expr_to_global_expr(&self, local_expr: impl Into<AB::Expr>) -> AB::Expr {
        self.start_offset_expr() + local_expr.into()
    }

    fn opcode_to_global_expr(&self, local_opcode: impl LocalOpcode) -> AB::Expr {
        self.expr_to_global_expr(AB::Expr::from_canonical_usize(local_opcode.local_usize()))
    }
}

pub struct VmChipWrapper<F, A: VmAdapter<F>, C: VmCoreChip<F, A>> {
    pub adapter: A,
    pub core: C,
    /// Each record is of the form `(from_state, core_record)`.
    pub records: Vec<(ExecutionState<u32>, C::Record)>,
    offline_memory: Arc<Mutex<OfflineMemory<F>>>,
}

// TODO: Make this configurable.
const DEFAULT_RECORDS_CAPACITY: usize = 1 << 20;

impl<F, A, C> VmChipWrapper<F, A, C>
where
    A: VmAdapter<F>,
    C: VmCoreChip<F, A>,
{
    pub fn new(adapter: A, core: C, offline_memory: Arc<Mutex<OfflineMemory<F>>>) -> Self {
        Self {
            adapter,
            core,
            records: Vec::with_capacity(DEFAULT_RECORDS_CAPACITY),
            offline_memory,
        }
    }
}

impl<F, A: VmAdapter<F>, C: VmCoreChip<F, A>> Stateful<Vec<u8>> for VmChipWrapper<F, A, C> {
    fn load_state(&mut self, state: Vec<u8>) {
        self.records = bitcode::deserialize(&state).unwrap();
    }

    fn store_state(&self) -> Vec<u8> {
        bitcode::serialize(&self.records).unwrap()
    }
}

impl<F, A, C> InstructionExecutor<F> for VmChipWrapper<F, A, C>
where
    F: PrimeField32,
    A: VmAdapter<F> + Send + Sync,
    C: VmCoreChip<F, A> + Send + Sync,
{
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>> {
        let mut tx = self.adapter.execute_tx();
        let (to_pc, core_record) =
            self.core
                .execute_instruction(memory, instruction, from_state.pc, &mut tx)?;
        self.records.push((from_state, core_record));
        let to_timestamp = memory.timestamp();
        let to_state = ExecutionState::new(to_pc, to_timestamp);
        Ok(to_state)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.core.get_opcode_name(opcode)
    }
}

// Note[jpw]: the statement we want is:
// - when A is an AdapterAir for all AirBuilders needed by stark-backend
// - and when C::Air is an CoreAir for all AirBuilders needed by stark-backend,
// then VmAirWrapper<A, C::Air> is an Air for all AirBuilders needed
// by stark-backend, which is equivalent to saying it can be represented as AnyRap<SC>
// The where clauses to achieve this statement is unfortunately really verbose.
impl<SC, A, C> Chip<SC> for VmChipWrapper<Val<SC>, A, C>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    A: VmAdapter<Val<SC>> + Send + Sync + 'static,
    C: VmCoreChip<Val<SC>, A> + Send + Sync,
    A: VmAdapterAir<SymbolicRapBuilder<Val<SC>>>
        + for<'a> VmAdapterAir<DebugConstraintBuilder<'a, SC>>, // AirRef bound
    C::Air: Send + Sync + 'static,
    C::Air: for<'tx> VmCoreAir<
        SymbolicRapBuilder<Val<SC>>,
        <A as VmAdapterAir<SymbolicRapBuilder<Val<SC>>>>::AirTx<'tx>,
    >,
    C::Air: for<'tx, 'a> VmCoreAir<
        DebugConstraintBuilder<'a, SC>,
        <A as VmAdapterAir<DebugConstraintBuilder<'a, SC>>>::AirTx<'tx>,
    >,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        let air: VmAirWrapper<A, C::Air> = VmAirWrapper {
            adapter: self.adapter.clone(),
            core: self.core.air().clone(),
        };
        Arc::new(air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = self.air();
        let num_records = self.records.len();
        let height = next_power_of_two_or_zero(num_records);
        let core_width = self.core.air().width();
        let adapter_width = self.adapter.width();
        let width = core_width + adapter_width;
        let mut values = Val::<SC>::zero_vec(height * width);

        let memory = self.offline_memory.lock().unwrap();

        // This zip only goes through records.
        // The padding rows between records.len()..height are filled with zeros.
        values
            .par_chunks_mut(width)
            .zip(self.records.into_par_iter())
            .for_each(|(row_slice, (from_state, core_record))| {
                let (adapter_row, core_row) = row_slice.split_at_mut(adapter_width);
                let mut tx = self.adapter.trace_tx(&memory, adapter_row, from_state);
                self.core.generate_trace_row(core_row, core_record, &mut tx);
            });

        let mut trace = RowMajorMatrix::new(values, width);
        self.core.finalize(&mut trace, num_records);

        AirProofInput::simple(air, trace, self.core.generate_public_values())
    }
}

impl<F, A, C> ChipUsageGetter for VmChipWrapper<F, A, C>
where
    A: VmAdapter<F> + Sync,
    C: VmCoreChip<F, A> + Sync,
{
    fn air_name(&self) -> String {
        format!(
            "<{},{}>",
            get_air_name(&self.adapter),
            get_air_name(self.core.air())
        )
    }
    fn current_trace_height(&self) -> usize {
        self.records.len()
    }
    fn trace_width(&self) -> usize {
        self.adapter.width() + self.core.air().width()
    }
}

pub struct VmAirWrapper<A, C> {
    pub adapter: A,
    pub core: C,
}

impl<F, A, C> BaseAir<F> for VmAirWrapper<A, C>
where
    A: BaseAir<F>,
    C: BaseAir<F>,
{
    fn width(&self) -> usize {
        self.adapter.width() + self.core.width()
    }
}

impl<F, A, C> BaseAirWithPublicValues<F> for VmAirWrapper<A, C>
where
    A: BaseAir<F>,
    C: BaseAirWithPublicValues<F>,
{
    fn num_public_values(&self) -> usize {
        self.core.num_public_values()
    }
}

// Current cached trace is not supported
impl<F, A, C> PartitionedBaseAir<F> for VmAirWrapper<A, C>
where
    A: BaseAir<F>,
    C: BaseAir<F>,
{
}

impl<AB, A, C> Air<AB> for VmAirWrapper<A, C>
where
    AB: AirBuilder,
    A: VmAdapterAir<AB>,
    C: for<'tx> VmCoreAir<AB, A::AirTx<'tx>>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let (local_adapter, local_core) = local.split_at(self.adapter.width());
        let mut tx = self.adapter.air_tx(local_adapter);

        self.core.eval(builder, local_core, &mut tx);
    }
}
