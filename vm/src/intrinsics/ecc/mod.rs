use std::{rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::utils::big_uint_mod_inverse,
    ecc::{EcAddUnequalAir, EcAirConfig, EcDoubleAir},
    var_range::VariableRangeCheckerChip,
};
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
use p3_field::PrimeField32;

use crate::{
    arch::{
        ExecutionBridge,
        ExecutionBus,
        ExecutionState, // instructions::Opcode,
        InstructionExecutor,
    },
    intrinsics::modular_addsub::{FIELD_ELEMENT_BITS, SECP256K1_COORD_PRIME},
    system::memory::{MemoryControllerRef, MemoryHeapReadRecord, MemoryHeapWriteRecord},
    system::program::{bridge::ProgramBus, ExecutionError, Instruction},
    utils::{biguint_to_limbs, limbs_to_biguint},
};

mod air;
mod bridge;
mod columns;
mod trace;

pub use air::*;
pub use columns::*;

#[cfg(test)]
mod test;

const NUM_LIMBS: usize = 32;
const LIMB_SIZE: usize = 8;
const TWO_NUM_LIMBS: usize = NUM_LIMBS * 2;

fn read_ec_points<T: PrimeField32>(
    memory_controller: MemoryControllerRef<T>,
    ptr_as: T,
    data_as: T,
    ptr_pointer: T,
) -> (BigUint, BigUint, MemoryHeapReadRecord<T, TWO_NUM_LIMBS>) {
    let mut memory_controller = memory_controller.borrow_mut();
    let array_read = memory_controller.read_heap::<TWO_NUM_LIMBS>(ptr_as, data_as, ptr_pointer);
    let u32_array = array_read.data_read.data.map(|x| x.as_canonical_u32());
    let x = limbs_to_biguint(&u32_array[..NUM_LIMBS], LIMB_SIZE);
    let y = limbs_to_biguint(&u32_array[NUM_LIMBS..], LIMB_SIZE);
    (x, y, array_read)
}

fn write_ec_points<T: PrimeField32>(
    memory_controller: MemoryControllerRef<T>,
    x: BigUint,
    y: BigUint,
    ptr_as: T,
    data_as: T,
    ptr_pointer: T,
) -> MemoryHeapWriteRecord<T, TWO_NUM_LIMBS> {
    let mut memory_controller = memory_controller.borrow_mut();
    let x_limbs = biguint_to_limbs::<NUM_LIMBS>(x, LIMB_SIZE);
    let y_limbs = biguint_to_limbs::<NUM_LIMBS>(y, LIMB_SIZE);
    let mut array = [0; 64];
    array[..NUM_LIMBS].copy_from_slice(&x_limbs);
    array[NUM_LIMBS..].copy_from_slice(&y_limbs);
    let array: [T; 64] = array.map(|x| T::from_canonical_u32(x));
    memory_controller.write_heap::<TWO_NUM_LIMBS>(ptr_as, data_as, ptr_pointer, array)
}

#[derive(Clone, Debug)]
pub struct EcAddUnequalRecord<T: PrimeField32> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    // Each limb is 8 bits (byte), 32 limbs for 256 bits, 2 coordinates for each point..
    pub p1_array_read: MemoryHeapReadRecord<T, TWO_NUM_LIMBS>,
    pub p2_array_read: MemoryHeapReadRecord<T, TWO_NUM_LIMBS>,
    pub p3_array_write: MemoryHeapWriteRecord<T, TWO_NUM_LIMBS>,
}

#[derive(Clone, Debug)]
pub struct EcChipConfig<T: PrimeField32> {
    memory_controller: MemoryControllerRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
    prime: BigUint,
}

#[derive(Clone, Debug)]
pub struct EcAddUnequalChip<T: PrimeField32> {
    pub air: EcAddUnequalVmAir,
    pub data: Vec<EcAddUnequalRecord<T>>,
    pub config: EcChipConfig<T>,

    _offset: usize,
}

fn make_ec_config<T: PrimeField32>(memory_controller: &MemoryControllerRef<T>) -> EcAirConfig {
    let range_checker_chip = memory_controller.borrow().range_checker.clone();
    let prime = SECP256K1_COORD_PRIME.clone();
    EcAirConfig::new(
        prime.clone(),
        BigUint::from_u32(7).unwrap(),
        range_checker_chip.bus().index,
        range_checker_chip.range_max_bits(),
        LIMB_SIZE,
        FIELD_ELEMENT_BITS,
    )
}

fn make_ec_chip_config<T: PrimeField32>(
    memory_controller: MemoryControllerRef<T>,
) -> EcChipConfig<T> {
    let range_checker_chip = memory_controller.borrow().range_checker.clone();
    let prime = SECP256K1_COORD_PRIME.clone();
    EcChipConfig {
        memory_controller,
        range_checker_chip,
        prime,
    }
}

impl<T: PrimeField32> EcAddUnequalChip<T> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<T>,
        offset: usize,
    ) -> Self {
        let memory_bridge = memory_controller.borrow().memory_bridge();

        let ec_config = make_ec_config(&memory_controller);
        let air = EcAddUnequalVmAir {
            air: EcAddUnequalAir { config: ec_config },
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
            offset,
        };
        let config = make_ec_chip_config(memory_controller);

        Self {
            air,
            config,
            data: vec![],
            _offset: offset,
        }
    }
}

impl<T: PrimeField32> InstructionExecutor<T> for EcAddUnequalChip<T> {
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        let Instruction {
            opcode: _,
            op_a: p3_address_ptr,
            op_b: p1_address_ptr,
            op_c: p2_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();

        let (p1_x, p1_y, p1_array_read) = read_ec_points(
            Rc::clone(&self.config.memory_controller),
            d,
            e,
            p1_address_ptr,
        );
        let (p2_x, p2_y, p2_array_read) = read_ec_points(
            Rc::clone(&self.config.memory_controller),
            d,
            e,
            p2_address_ptr,
        );

        let prime = self.config.prime.clone();
        let dx = &prime + &p1_x - &p2_x;
        let dy = &prime + &p1_y - &p2_y;
        let dx_inv = big_uint_mod_inverse(&dx, &prime);
        let lambda: BigUint = (dy * dx_inv) % &prime;
        let p3_x: BigUint = (&lambda * &lambda + &prime + &prime - &p1_x - &p2_x) % &prime;
        let p3_y: BigUint = (&lambda * (&prime + &p1_x - &p3_x) + &prime - &p1_y) % &prime;

        let p3_array_write = write_ec_points(
            Rc::clone(&self.config.memory_controller),
            p3_x,
            p3_y,
            d,
            e,
            p3_address_ptr,
        );

        let record = EcAddUnequalRecord {
            from_state,
            instruction,
            p1_array_read,
            p2_array_read,
            p3_array_write,
        };
        self.data.push(record);

        let memory_controller = self.config.memory_controller.borrow();
        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_controller.timestamp().as_canonical_u32() as usize,
        })
    }

    fn get_opcode_name(&self, _: usize) -> String {
        "SECP256K1_ADD_NE".to_string()
    }
}

#[derive(Clone, Debug)]
pub struct EcDoubleRecord<T: PrimeField32> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    // Each limb is 8 bits (byte), 32 limbs for 256 bits, 2 coordinates for each point..
    pub p1_array_read: MemoryHeapReadRecord<T, TWO_NUM_LIMBS>,
    pub p2_array_write: MemoryHeapWriteRecord<T, TWO_NUM_LIMBS>,
}

#[derive(Clone, Debug)]
pub struct EcDoubleChip<T: PrimeField32> {
    pub air: EcDoubleVmAir,
    pub data: Vec<EcDoubleRecord<T>>,
    pub config: EcChipConfig<T>,

    _offset: usize,
}

impl<T: PrimeField32> EcDoubleChip<T> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<T>,
        offset: usize,
    ) -> Self {
        let memory_bridge = memory_controller.borrow().memory_bridge();

        let ec_config = make_ec_config(&memory_controller);
        let air = EcDoubleVmAir {
            air: EcDoubleAir { config: ec_config },
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
            offset,
        };
        let config = make_ec_chip_config(memory_controller);

        Self {
            air,
            config,
            data: vec![],
            _offset: offset,
        }
    }
}

impl<T: PrimeField32> InstructionExecutor<T> for EcDoubleChip<T> {
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        let Instruction {
            opcode: _,
            op_a: p2_address_ptr,
            op_b: p1_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();

        let (p1_x, p1_y, p1_array_read) = read_ec_points(
            Rc::clone(&self.config.memory_controller),
            d,
            e,
            p1_address_ptr,
        );

        let prime = self.config.prime.clone();
        let two_y = &p1_y + &p1_y;
        let two_y_inv = big_uint_mod_inverse(&two_y, &prime);
        let three = BigUint::from_u32(3).unwrap();
        let lambda: BigUint = three * &p1_x * &p1_x * two_y_inv;
        let p3_x: BigUint = (&lambda * &lambda + &prime + &prime - &p1_x - &p1_x) % &prime;
        let p3_y: BigUint = (&lambda * (&prime + &p1_x - &p3_x) + &prime - &p1_y) % &prime;

        let p2_array_write = write_ec_points(
            Rc::clone(&self.config.memory_controller),
            p3_x,
            p3_y,
            d,
            e,
            p2_address_ptr,
        );

        let record = EcDoubleRecord {
            from_state,
            instruction,
            p1_array_read,
            p2_array_write,
        };
        self.data.push(record);

        let memory_controller = self.config.memory_controller.borrow();
        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_controller.timestamp().as_canonical_u32() as usize,
        })
    }

    fn get_opcode_name(&self, _: usize) -> String {
        "SECP256K1_DOUBLE".to_string()
    }
}
