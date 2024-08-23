use std::{array::from_fn, cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::interaction::InteractionBuilder;
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use itertools::izip;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{seq::SliceRandom, Rng, RngCore};

use crate::{
    cpu::RANGE_CHECKER_BUS,
    memory::{
        manager::{dimensions::MemoryDimensions, trace_builder::MemoryTraceBuilder, MemoryManager},
        offline_checker::{bridge::MemoryOfflineChecker, columns::MemoryOfflineCheckerCols},
    },
    vm::config::MemoryConfig,
};

const TEST_NUM_WORDS: usize = 1;
const TEST_WORD_SIZE: usize = 4;

type Val = BabyBear;

struct OfflineCheckerDummyAir {
    offline_checker: MemoryOfflineChecker,
}

impl<F: Field> BaseAir<F> for OfflineCheckerDummyAir {
    fn width(&self) -> usize {
        MemoryOfflineCheckerCols::<TEST_WORD_SIZE, usize>::width(&self.offline_checker)
    }
}

impl<AB: InteractionBuilder> Air<AB> for OfflineCheckerDummyAir {
    fn eval(&self, builder: &mut AB) {
        let main = &builder.main();

        let local = main.row_slice(0);
        let local = MemoryOfflineCheckerCols::<TEST_WORD_SIZE, AB::Var>::from_slice(&local);

        self.offline_checker
            .subair_eval(builder, local.io.into_expr::<AB>(), local.aux);
    }
}

#[test]
fn volatile_memory_offline_checker_test() {
    let mut rng = create_seeded_rng();

    const MAX_VAL: u32 = 1 << 20;

    let memory_dimensions = MemoryDimensions::new(1, 20, 1);
    let mem_config = MemoryConfig::new(29, 29, 29, 16);

    let range_checker = Arc::new(RangeCheckerGateChip::new(
        RANGE_CHECKER_BUS,
        (1 << mem_config.decomp) as u32,
    ));
    let memory_manager = Rc::new(RefCell::new(MemoryManager::<
        TEST_NUM_WORDS,
        TEST_WORD_SIZE,
        Val,
    >::with_volatile_memory(
        mem_config, range_checker.clone()
    )));
    let offline_checker = MemoryOfflineChecker::new(mem_config.clk_max_bits, mem_config.decomp);

    let num_addresses = rng.gen_range(1..=10);
    let mut all_addresses = vec![];
    for _ in 0..num_addresses {
        let addr_space = Val::from_canonical_usize(
            memory_dimensions.as_offset + rng.gen_range(0..(1 << memory_dimensions.as_height)),
        );
        let pointer = Val::from_canonical_u32(
            rng.gen_range(0..(1 << memory_dimensions.address_height)) as u32
                / TEST_WORD_SIZE as u32
                * TEST_WORD_SIZE as u32,
        );

        all_addresses.push((addr_space, pointer));
    }

    let mut mem_ops = vec![];
    let mut mem_trace_builder = MemoryTraceBuilder::new(
        memory_manager.clone(),
        range_checker.clone(),
        offline_checker,
    );
    // First, write to all addresses
    for (addr_space, pointer) in all_addresses.iter() {
        let word = from_fn(|_| Val::from_canonical_u32(rng.next_u32() % MAX_VAL));
        mem_ops.push(mem_trace_builder.write_word(*addr_space, *pointer, word));
    }

    // Second, do some random memory accesses
    let num_accesses = rng.gen_range(1..=10);
    for _ in 0..num_accesses {
        let (addr_space, pointer) = *all_addresses.choose(&mut rng).unwrap();
        let word = from_fn(|_| Val::from_canonical_u32(rng.next_u32() % MAX_VAL));

        let mem_op = if rng.gen_bool(0.5) {
            mem_trace_builder.write_word(addr_space, pointer, word)
        } else {
            mem_trace_builder.read_word(addr_space, pointer)
        };

        mem_ops.push(mem_op);
    }

    let diff = mem_ops.len().next_power_of_two() - mem_ops.len();
    for _ in 0..diff {
        mem_ops.push(mem_trace_builder.disabled_read(BabyBear::one()));
    }

    let mut checker_trace = vec![];
    for (op, aux_cols) in izip!(
        mem_ops.into_iter(),
        mem_trace_builder.take_accesses_buffer().into_iter()
    ) {
        checker_trace.extend(op.flatten());
        checker_trace.extend(aux_cols.flatten());
    }

    let checker_width = MemoryOfflineCheckerCols::<TEST_WORD_SIZE, Val>::width(&offline_checker);
    let checker_trace = RowMajorMatrix::new(checker_trace, checker_width);
    let memory_interface_trace = memory_manager.borrow().generate_memory_interface_trace();
    let range_checker_trace = range_checker.generate_trace();
    let audit_air = memory_manager.borrow().get_audit_air();
    let offline_checker_air = OfflineCheckerDummyAir { offline_checker };

    run_simple_test_no_pis(
        vec![&range_checker.air, &offline_checker_air, &audit_air],
        vec![range_checker_trace, checker_trace, memory_interface_trace],
    )
    .expect("Verification failed");
}
