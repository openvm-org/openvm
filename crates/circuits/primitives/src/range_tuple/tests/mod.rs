use std::{array, iter, sync::Arc};

use openvm_stark_backend::{
    any_air_arc_vec,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    prover::{AirProvingContext, ColMajorMatrix},
    test_utils::dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    utils::disable_debug_builder,
    AirRef, StarkEngine,
};
#[cfg(not(feature = "cuda"))]
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::Rng;
#[cfg(feature = "cuda")]
use {
    crate::{
        range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerChipGPU},
        utils::test_gpu_engine_small,
        Chip,
    },
    array::from_fn,
    dummy::DummyInteractionChipGPU,
    openvm_cuda_backend::{
        base::DeviceMatrix,
        prelude::{F, SC},
    },
    openvm_cuda_common::copy::MemCopyH2D as _,
    openvm_stark_backend::p3_air::BaseAir,
};

use crate::{
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
    utils::test_engine_small,
};

#[cfg(feature = "cuda")]
mod dummy;

#[test]
fn test_range_tuple_chip() {
    fn test_with_size<const N: usize>() {
        let mut rng = create_seeded_rng();

        const LIST_LEN: usize = 64;

        let bus_index = 0;
        let sizes: [u32; N] = array::from_fn(|_| 1 << rng.random_range(1..5));

        let bus = RangeTupleCheckerBus::new(bus_index, sizes);
        let range_checker = RangeTupleCheckerChip::new(bus);

        // generates a valid random tuple given sizes
        let mut gen_tuple = || {
            sizes
                .iter()
                .map(|&size| rng.random_range(0..size))
                .collect::<Vec<_>>()
        };

        // generates a list of random valid tuples
        let num_lists = 10;
        let lists_vals = (0..num_lists)
            .map(|_| (0..LIST_LEN).map(|_| gen_tuple()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        // generate dummy AIR chips for each list
        let lists_airs = (0..num_lists)
            .map(|_| DummyInteractionAir::new(sizes.len(), true, bus_index))
            .collect::<Vec<DummyInteractionAir>>();

        let mut all_chips = lists_airs
            .into_iter()
            .map(|list| Arc::new(list) as AirRef<_>)
            .collect::<Vec<_>>();
        all_chips.push(Arc::new(range_checker.air));

        // generate traces for each list
        let lists_traces = lists_vals
            .par_iter()
            .map(|list| {
                RowMajorMatrix::new(
                    list.clone()
                        .into_iter()
                        .flat_map(|v| {
                            range_checker.add_count(&v);
                            iter::once(1).chain(v)
                        })
                        .map(PrimeCharacteristicRing::from_u32)
                        .collect(),
                    sizes.len() + 1,
                )
            })
            .collect::<Vec<RowMajorMatrix<F>>>();

        let range_trace = range_checker.generate_trace();

        let all_traces = lists_traces
            .into_iter()
            .chain(iter::once(range_trace))
            .collect::<Vec<RowMajorMatrix<F>>>();

        let all_traces = all_traces
            .iter()
            .map(ColMajorMatrix::from_row_major)
            .map(AirProvingContext::simple_no_pis)
            .collect::<Vec<_>>();
        test_engine_small()
            .run_test(all_chips, all_traces)
            .expect("Verification failed");
    }

    test_with_size::<2>();
    test_with_size::<3>();
    test_with_size::<4>();
}

#[test]
#[should_panic]
fn negative_test_range_tuple_chip() {
    let bus_index = 0;
    let sizes = [2, 2, 8];

    let bus = RangeTupleCheckerBus::new(bus_index, sizes);
    let range_checker = RangeTupleCheckerChip::new(bus);

    let valid_tuples: Vec<Vec<u32>> = (0..2)
        .flat_map(|i| (0..2).flat_map(move |j| (0..8).map(move |k| vec![i, j, k])))
        .collect();

    let dummy_air = DummyInteractionAir::new(sizes.len(), true, bus_index);

    let dummy_trace = RowMajorMatrix::new(
        valid_tuples
            .iter()
            .flat_map(|v| {
                range_checker.add_count(v);
                iter::once(1).chain(v.iter().cloned())
            })
            .map(PrimeCharacteristicRing::from_u32)
            .collect(),
        sizes.len() + 1,
    );

    let mut range_trace = range_checker.generate_trace();

    let mut all_chips: Vec<AirRef<_>> = vec![Arc::new(dummy_air) as AirRef<_>];
    all_chips.push(Arc::new(range_checker.air));

    {
        let traces = [&dummy_trace, &range_trace]
            .into_iter()
            .map(ColMajorMatrix::from_row_major)
            .map(AirProvingContext::simple_no_pis)
            .collect::<Vec<_>>();
        test_engine_small()
            .run_test(all_chips, traces)
            .expect("This should not fail");
    }

    // Corrupt the trace to make it invalid
    range_trace.values[0] = F::from_u32(99);

    disable_debug_builder();
    let error = test_engine_small()
        .run_test(
            any_air_arc_vec![range_checker.air],
            [range_trace]
                .iter()
                .map(ColMajorMatrix::from_row_major)
                .map(AirProvingContext::simple_no_pis)
                .collect::<Vec<_>>(),
        )
        .err();
    assert!(error.is_some(), "Expected constraint to fail, got success",);
}

#[test]
fn negative_test_range_tuple_chip_size_overflow() {
    let bus_index = 0;
    let sizes = [256, 256, 256, 256];

    let result = std::panic::catch_unwind(|| RangeTupleCheckerBus::new(bus_index, sizes));
    assert!(
        result.is_err(),
        "Expected RangeTupleCheckerBus::new to panic on overflow"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_range_tuple() {
    const TUPLE_SIZE: usize = 3;
    const NUM_INPUTS: usize = 1 << 16;

    let mut rng = create_seeded_rng();
    let sizes: [u32; TUPLE_SIZE] = from_fn(|_| 1 << rng.random_range(1..5));
    let bus = RangeTupleCheckerBus::<TUPLE_SIZE>::new(0, sizes);
    let random_values = (0..NUM_INPUTS)
        .flat_map(|_| {
            sizes
                .iter()
                .map(|&size| rng.random_range(0..size))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let range_tuple_checker = Arc::new(RangeTupleCheckerChipGPU::new(bus.sizes));
    let dummy_chip = DummyInteractionChipGPU::new(range_tuple_checker.clone(), random_values);

    let airs: Vec<AirRef<SC>> = vec![
        Arc::new(DummyInteractionAir::new(TUPLE_SIZE, true, bus.inner.index)),
        Arc::new(RangeTupleCheckerAir::<TUPLE_SIZE> { bus }),
    ];
    let ctxs = vec![
        dummy_chip.generate_proving_ctx(()),
        range_tuple_checker.generate_proving_ctx(()),
    ];

    test_gpu_engine_small()
        .run_test(airs, ctxs)
        .expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_range_tuple_hybrid() {
    const TUPLE_SIZE: usize = 3;
    const NUM_INPUTS: usize = 1 << 16;

    let mut rng = create_seeded_rng();
    let sizes: [u32; TUPLE_SIZE] = from_fn(|_| 1 << rng.random_range(1..5));
    let bus = RangeTupleCheckerBus::<TUPLE_SIZE>::new(0, sizes);
    let range_tuple_checker = Arc::new(RangeTupleCheckerChipGPU::hybrid(Arc::new(
        RangeTupleCheckerChip::new(bus),
    )));

    let gpu_values = (0..NUM_INPUTS)
        .flat_map(|_| {
            sizes
                .iter()
                .map(|&size| rng.random_range(0..size))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let gpu_dummy_chip = DummyInteractionChipGPU::new(range_tuple_checker.clone(), gpu_values);

    let cpu_chip = range_tuple_checker.cpu_chip.clone().unwrap();
    let cpu_values = (0..NUM_INPUTS)
        .map(|_| {
            let values = sizes
                .iter()
                .map(|&size| rng.random_range(0..size))
                .collect::<Vec<_>>();
            cpu_chip.add_count(&values);
            values
        })
        .collect::<Vec<_>>();
    let cpu_dummy_trace = (0..NUM_INPUTS)
        .map(|_| F::ONE)
        .chain(
            cpu_values
                .iter()
                .map(|v| F::from_u32(v[0]))
                .chain(cpu_values.iter().map(|v| F::from_u32(v[1])))
                .chain(cpu_values.iter().map(|v| F::from_u32(v[2]))),
        )
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    let dummy_air = DummyInteractionAir::new(TUPLE_SIZE, true, bus.inner.index);
    let cpu_proving_ctx = AirProvingContext {
        cached_mains: vec![],
        common_main: DeviceMatrix::new(
            Arc::new(cpu_dummy_trace),
            NUM_INPUTS,
            BaseAir::<F>::width(&dummy_air),
        ),
        public_values: vec![],
    };

    let airs: Vec<AirRef<SC>> = vec![
        Arc::new(dummy_air),
        Arc::new(dummy_air),
        Arc::new(RangeTupleCheckerAir::<TUPLE_SIZE> { bus }),
    ];
    let ctxs = vec![
        cpu_proving_ctx,
        gpu_dummy_chip.generate_proving_ctx(()),
        range_tuple_checker.generate_proving_ctx(()),
    ];

    test_gpu_engine_small()
        .run_test(airs, ctxs)
        .expect("Verification failed");
}
