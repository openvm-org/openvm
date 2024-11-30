use std::{
    array,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
use ax_poseidon2_air::poseidon2::Poseidon2Config;
use ax_stark_backend::{
    interaction::InteractionBuilder,
    prover::types::AirProofInput,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    Chip,
};
use ax_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{
    prelude::{SliceRandom, StdRng},
    Rng,
};

use super::{
    merkle::DirectCompressionBus, Equipartition, MemoryAuxColsFactory, MemoryController,
    MemoryReadRecord,
};
use crate::{
    arch::{
        testing::memory::gen_pointer, ExecutionBus, MemoryConfig, EXECUTION_BUS, MEMORY_BUS,
        MEMORY_MERKLE_BUS, POSEIDON2_DIRECT_BUS, READ_INSTRUCTION_BUS,
    },
    system::{
        memory::{
            merkle::MemoryMerkleBus,
            offline_checker::{MemoryBridge, MemoryBus, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryWriteRecord,
        },
        poseidon2::Poseidon2Chip,
        program::ProgramBus,
    },
};

const MAX: usize = 64;
const RANGE_CHECKER_BUS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
struct MemoryRequesterCols<T> {
    address_space: T,
    pointer: T,
    data_1: [T; 1],
    data_4: [T; 4],
    data_max: [T; MAX],
    timestamp: T,
    write_1_aux: MemoryWriteAuxCols<T, 1>,
    write_4_aux: MemoryWriteAuxCols<T, 4>,
    read_1_aux: MemoryReadAuxCols<T, 1>,
    read_4_aux: MemoryReadAuxCols<T, 4>,
    read_max_aux: MemoryReadAuxCols<T, MAX>,
    is_write_1: T,
    is_write_4: T,
    is_read_1: T,
    is_read_4: T,
    is_read_max: T,
}

struct MemoryRequesterAir {
    memory_bridge: MemoryBridge,
}

impl<T> BaseAirWithPublicValues<T> for MemoryRequesterAir {}
impl<T> PartitionedBaseAir<T> for MemoryRequesterAir {}
impl<T> BaseAir<T> for MemoryRequesterAir {
    fn width(&self) -> usize {
        MemoryRequesterCols::<T>::width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for MemoryRequesterAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryRequesterCols<AB::Var> = (*local).borrow();

        let flags = [
            local.is_read_1,
            local.is_write_1,
            local.is_read_4,
            local.is_write_4,
            local.is_read_max,
        ];

        let mut sum = AB::Expr::ZERO;
        for flag in flags {
            builder.assert_bool(flag);
            sum += flag.into();
        }
        builder.assert_one(sum);

        self.memory_bridge
            .read(
                MemoryAddress::new(local.address_space, local.pointer),
                local.data_1,
                local.timestamp,
                &local.read_1_aux,
            )
            .eval(builder, local.is_read_1);

        self.memory_bridge
            .read(
                MemoryAddress::new(local.address_space, local.pointer),
                local.data_4,
                local.timestamp,
                &local.read_4_aux,
            )
            .eval(builder, local.is_read_4);

        self.memory_bridge
            .write(
                MemoryAddress::new(local.address_space, local.pointer),
                local.data_1,
                local.timestamp,
                &local.write_1_aux,
            )
            .eval(builder, local.is_write_1);

        self.memory_bridge
            .write(
                MemoryAddress::new(local.address_space, local.pointer),
                local.data_4,
                local.timestamp,
                &local.write_4_aux,
            )
            .eval(builder, local.is_write_4);

        self.memory_bridge
            .read(
                MemoryAddress::new(local.address_space, local.pointer),
                local.data_max,
                local.timestamp,
                &local.read_max_aux,
            )
            .eval(builder, local.is_read_max);
    }
}

#[allow(clippy::large_enum_variant)]
enum Record<F> {
    Write(MemoryWriteRecord<F, 1>),
    Read(MemoryReadRecord<F, 1>),
    Read4(MemoryReadRecord<F, 4>),
    Write4(MemoryWriteRecord<F, 4>),
    ReadMax(MemoryReadRecord<F, MAX>),
}

fn generate_trace<F: PrimeField32>(
    records: Vec<Record<F>>,
    aux_factory: MemoryAuxColsFactory<F>,
) -> RowMajorMatrix<F> {
    let height = records.len().next_power_of_two();
    let width = MemoryRequesterCols::<F>::width();
    let mut values = F::zero_vec(height * width);

    for (row, record) in values.chunks_mut(width).zip(records) {
        let row: &mut MemoryRequesterCols<F> = row.borrow_mut();
        match record {
            Record::Write(record) => {
                row.address_space = record.address_space;
                row.pointer = record.pointer;
                row.timestamp = F::from_canonical_u32(record.timestamp);

                row.data_1 = record.data;
                row.write_1_aux = aux_factory.make_write_aux_cols(record);
                row.is_write_1 = F::ONE;
            }
            Record::Read(record) => {
                row.address_space = record.address_space;
                row.pointer = record.pointer;
                row.timestamp = F::from_canonical_u32(record.timestamp);

                row.data_1 = record.data;
                row.read_1_aux = aux_factory.make_read_aux_cols(record);
                row.is_read_1 = F::ONE;
            }
            Record::Read4(record) => {
                row.address_space = record.address_space;
                row.pointer = record.pointer;
                row.timestamp = F::from_canonical_u32(record.timestamp);

                row.data_4 = record.data;
                row.read_4_aux = aux_factory.make_read_aux_cols(record);
                row.is_read_4 = F::ONE;
            }
            Record::Write4(record) => {
                row.address_space = record.address_space;
                row.pointer = record.pointer;
                row.timestamp = F::from_canonical_u32(record.timestamp);

                row.data_4 = record.data;
                row.write_4_aux = aux_factory.make_write_aux_cols(record);
                row.is_write_4 = F::ONE;
            }
            Record::ReadMax(record) => {
                row.address_space = record.address_space;
                row.pointer = record.pointer;
                row.timestamp = F::from_canonical_u32(record.timestamp);

                row.data_max = record.data;
                row.read_max_aux = aux_factory.make_read_aux_cols(record);
                row.is_read_max = F::ONE;
            }
        }
    }
    RowMajorMatrix::new(values, width)
}

/// Simple integration test for memory chip.
///
/// Creates a bunch of random read/write records, used to generate a trace for [MemoryRequesterAir],
/// which sends reads/writes over [MemoryBridge].
#[test]
fn test_memory_controller() {
    let memory_bus = MemoryBus(MEMORY_BUS);
    let memory_config = MemoryConfig::default();
    let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, memory_config.decomp);
    let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let mut memory_controller = MemoryController::with_volatile_memory(
        memory_bus,
        memory_config,
        range_checker.clone(),
        None,
    );
    let aux_factory = memory_controller.aux_cols_factory();

    let mut rng = create_seeded_rng();
    let records = make_random_accesses(&mut memory_controller, &mut rng);
    let memory_requester_air = Arc::new(MemoryRequesterAir {
        memory_bridge: memory_controller.memory_bridge(),
    });
    let memory_requester_trace = generate_trace(records, aux_factory);

    memory_controller.finalize(None::<&mut Poseidon2Chip<BabyBear>>);

    let mut air_proof_inputs = memory_controller.generate_air_proof_inputs();
    air_proof_inputs.push(AirProofInput::simple_no_pis(
        memory_requester_air,
        memory_requester_trace,
    ));
    air_proof_inputs.push(range_checker.generate_air_proof_input());

    BabyBearPoseidon2Engine::run_test_fast(air_proof_inputs).expect("Verification failed");
}

#[test]
fn test_memory_controller_persistent() {
    let memory_bus = MemoryBus(MEMORY_BUS);
    let merkle_bus = MemoryMerkleBus(MEMORY_MERKLE_BUS);
    let compression_bus = DirectCompressionBus(POSEIDON2_DIRECT_BUS);
    let memory_config = MemoryConfig::default();
    let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, memory_config.decomp);
    let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let mut memory_controller = MemoryController::with_persistent_memory(
        memory_bus,
        memory_config,
        range_checker.clone(),
        merkle_bus,
        compression_bus,
        Equipartition::new(),
        None,
    );
    let aux_factory = memory_controller.aux_cols_factory();

    let mut rng = create_seeded_rng();
    let records = make_random_accesses(&mut memory_controller, &mut rng);
    let memory_requester_trace = generate_trace(records, aux_factory);

    let memory_requester_air = MemoryRequesterAir {
        memory_bridge: memory_controller.memory_bridge(),
    };

    // This never gets used because poseido2_chip will only have direct compression interactions
    let dummy_memory_controller = MemoryController::with_volatile_memory(
        MemoryBus(MEMORY_BUS),
        MemoryConfig::default(),
        range_checker.clone(),
        None,
    );

    let mut poseidon_chip = Poseidon2Chip::from_poseidon2_config(
        Poseidon2Config::<16, BabyBear>::new_p3_baby_bear_16(),
        3,
        ExecutionBus(EXECUTION_BUS),
        ProgramBus(READ_INSTRUCTION_BUS),
        Rc::new(RefCell::new(dummy_memory_controller)),
        POSEIDON2_DIRECT_BUS,
        0,
    );

    memory_controller.finalize(Some(&mut poseidon_chip));
    let mut air_proof_inputs = memory_controller.generate_air_proof_inputs();
    air_proof_inputs.push(AirProofInput::simple_no_pis(
        Arc::new(memory_requester_air),
        memory_requester_trace,
    ));
    air_proof_inputs.push(poseidon_chip.generate_air_proof_input());
    air_proof_inputs.push(range_checker.generate_air_proof_input());

    BabyBearPoseidon2Engine::run_test_fast(air_proof_inputs).expect("Verification failed");
}

fn make_random_accesses<F: PrimeField32>(
    memory_controller: &mut MemoryController<F>,
    mut rng: &mut StdRng,
) -> Vec<Record<F>> {
    (0..1024)
        .map(|_| {
            let address_space = F::from_canonical_u32(*[1, 2].choose(&mut rng).unwrap());

            match rng.gen_range(0..5) {
                0 => {
                    let pointer = F::from_canonical_usize(gen_pointer(rng, 1));
                    let data = F::from_canonical_u32(rng.gen_range(0..1 << 30));
                    Record::Write(memory_controller.write(address_space, pointer, [data]))
                }
                1 => {
                    let pointer = F::from_canonical_usize(gen_pointer(rng, 1));
                    Record::Read(memory_controller.read::<1>(address_space, pointer))
                }
                2 => {
                    let pointer = F::from_canonical_usize(gen_pointer(rng, 4));
                    Record::Read4(memory_controller.read::<4>(address_space, pointer))
                }
                3 => {
                    let pointer = F::from_canonical_usize(gen_pointer(rng, 4));
                    let data = array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..1 << 30)));
                    Record::Write4(memory_controller.write::<4>(address_space, pointer, data))
                }
                4 => {
                    let pointer = F::from_canonical_usize(gen_pointer(rng, MAX));
                    Record::ReadMax(memory_controller.read::<MAX>(address_space, pointer))
                }
                _ => unreachable!(),
            }
        })
        .collect_vec()
}
