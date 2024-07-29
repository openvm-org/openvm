use std::cmp::Reverse;

use afs_recursion::{hints::InnerVal, stark::DynRapForRecursion, types::InnerConfig};
use afs_stark_backend::rap::AnyRap;
use afs_test_utils::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use itertools::{izip, Itertools};
use p3_baby_bear::BabyBear;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use stark_vm::vm::VirtualMachine;

pub fn get_rec_raps<const WORD_SIZE: usize>(
    vm: &VirtualMachine<WORD_SIZE, InnerVal>,
) -> Vec<&dyn DynRapForRecursion<InnerConfig>> {
    let mut result: Vec<&dyn DynRapForRecursion<InnerConfig>> = vec![
        &vm.cpu_air,
        &vm.program_chip.air,
        &vm.memory_chip.air,
        &vm.range_checker.air,
    ];
    if vm.options().field_arithmetic_enabled {
        result.push(&vm.field_arithmetic_chip.air);
    }
    if vm.options().field_extension_enabled {
        result.push(&vm.field_extension_chip.air);
    }
    if vm.options().poseidon2_enabled() {
        result.push(&vm.poseidon2_chip.air);
    }
    result
}

#[allow(clippy::type_complexity)]
pub fn sort_chips<'a>(
    chips: Vec<&'a dyn AnyRap<BabyBearPoseidon2Config>>,
    rec_raps: Vec<&'a dyn DynRapForRecursion<InnerConfig>>,
    traces: Vec<RowMajorMatrix<BabyBear>>,
    pvs: Vec<Vec<BabyBear>>,
) -> (
    Vec<&'a dyn AnyRap<BabyBearPoseidon2Config>>,
    Vec<&'a dyn DynRapForRecursion<InnerConfig>>,
    Vec<RowMajorMatrix<BabyBear>>,
    Vec<Vec<BabyBear>>,
) {
    let mut groups = izip!(chips, rec_raps, traces, pvs).collect_vec();
    groups.sort_by_key(|(_, _, trace, _)| Reverse(trace.height()));

    let chips = groups.iter().map(|(x, _, _, _)| *x).collect_vec();
    let rec_raps = groups.iter().map(|(_, x, _, _)| *x).collect_vec();
    let pvs = groups.iter().map(|(_, _, _, x)| x.clone()).collect_vec();
    let traces = groups.into_iter().map(|(_, _, x, _)| x).collect_vec();

    (chips, rec_raps, traces, pvs)
}
