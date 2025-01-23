use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    p3_air::AirBuilder, p3_field::FieldAlgebra, p3_matrix::dense::RowMajorMatrix,
    rap::BaseAirWithPublicValues,
};
use serde::{de::DeserializeOwned, Serialize};

use super::Result;
use crate::system::memory::MemoryController;

pub trait VmAdapter<F> {
    type ExecuteTx<'tx>
    where
        Self: 'tx;

    type TraceTx<'tx>
    where
        Self: 'tx;
}

/// Trait to be implemented on primitive chip to integrate with the machine.
pub trait VmCoreChip<F, A: VmAdapter<F>> {
    /// Minimum data that must be recorded to be able to generate trace for one row.
    type Record: Send + Serialize + DeserializeOwned;
    type Air;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_pc: u32,
        tx: &mut A::ExecuteTx<'_>,
    ) -> Result<Self::Record>;

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
