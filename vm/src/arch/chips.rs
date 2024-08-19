use enum_dispatch::enum_dispatch;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use afs_stark_backend::{interaction::InteractionBuilder, rap::AnyRap};
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    arch::columns::ExecutionState, cpu::trace::Instruction,
    field_extension::FieldExtensionArithmeticChip,
};

#[enum_dispatch]
pub trait OpCodeExecutor<F> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize>;
}

pub trait MachineAir<AB: AirBuilder>: Air<AB> + BaseAir<AB::F> {}

#[enum_dispatch]
pub trait MachineChip<F> {
    fn generate_trace(&mut self) -> RowMajorMatrix<F>;
    fn air<AB: AirBuilder<F = F> + InteractionBuilder + AirBuilderWithPublicValues>(
        &self,
    ) -> &dyn MachineAir<AB>;
    fn get_public_values(&mut self) -> Vec<F> {
        vec![]
    }
}

#[enum_dispatch(OpCodeExecutor<F>)]
pub enum OpCodeExecutorVariant<F: PrimeField32> {
    FieldExtension(FieldExtensionArithmeticChip<F>),
}

#[enum_dispatch(MachineChip<F>)]
pub enum MachineChipVariant<F: PrimeField32> {
    FieldExtension(FieldExtensionArithmeticChip<F>),
}

fn bing<C: MachineChip<BabyBear>>(chip: &C) -> &dyn AnyRap<BabyBearPoseidon2Config> {
    chip.air() as &dyn AnyRap<BabyBearPoseidon2Config>
}

struct RandomAir<F: Field> {}

impl<F: Field> BaseAir<F> for RandomAir<F> {
    fn width(&self) -> usize {
        todo!()
    }
}

impl<F: Field, AB: AirBuilder<F = F>> Air<AB> for RandomAir<F> {
    fn eval(&self, builder: &mut AB) {
        todo!()
    }
}

fn bong0<SC: StarkGenericConfig, C: MachineChip<Val<SC>>>(chip: &C) {
    let sh = chip.air() as &dyn AnyRap<SC>;

    todo!()
}

fn bong<SC: StarkGenericConfig>(chip: &RandomAir<Val<SC>>) -> &dyn AnyRap<SC> {
    let sh = chip as &dyn AnyRap<SC>;

    todo!()
}
