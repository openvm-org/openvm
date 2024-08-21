use std::{
    array,
    cell::RefCell,
    ops::{Add, Mul, Sub},
    rc::Rc,
    sync::Arc,
};

use afs_primitives::range_gate::RangeCheckerGateChip;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    cpu::{trace::Instruction, OpCode, FIELD_EXTENSION_INSTRUCTIONS},
    field_extension::{
        air::FieldExtensionArithmeticAir,
        columns::{
            FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols,
            FieldExtensionArithmeticIoCols,
        },
    },
    memory::{
        manager::{trace_builder::MemoryTraceBuilder, MemoryManager},
        offline_checker::bridge::MemoryOfflineChecker,
        OpType,
    },
    vm::config::MemoryConfig,
};

pub const BETA: usize = 11;
pub const EXTENSION_DEGREE: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldExtensionArithmeticOperation<const WORD_SIZE: usize, F> {
    pub clk: usize,
    pub opcode: OpCode,
    pub op_a: F,
    pub op_b: F,
    pub op_c: F,
    pub d: F,
    pub e: F,
    pub operand1: [F; EXTENSION_DEGREE],
    pub operand2: [F; EXTENSION_DEGREE],
    pub result: [F; EXTENSION_DEGREE],
}

pub struct FieldExtensionArithmeticChip<
    const NUM_WORDS: usize,
    const WORD_SIZE: usize,
    F: PrimeField32,
> {
    pub air: FieldExtensionArithmeticAir<WORD_SIZE>,
    pub operations: Vec<FieldExtensionArithmeticOperation<WORD_SIZE, F>>,

    pub memory_manager: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
    pub memory: MemoryTraceBuilder<NUM_WORDS, WORD_SIZE, F>,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    FieldExtensionArithmeticChip<NUM_WORDS, WORD_SIZE, F>
{
    #[allow(clippy::new_without_default)]
    pub fn new(
        mem_config: MemoryConfig,
        memory_manager: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        let air = FieldExtensionArithmeticAir {
            mem_oc: MemoryOfflineChecker::new(mem_config.clk_max_bits, mem_config.decomp),
        };
        let memory = MemoryTraceBuilder::<NUM_WORDS, WORD_SIZE, F>::new(
            memory_manager.clone(),
            range_checker.clone(),
            air.mem_oc,
        );
        Self {
            air,
            operations: vec![],
            memory_manager,
            memory,
            range_checker,
        }
    }

    pub fn accesses_per_instruction(opcode: OpCode) -> usize {
        assert!(FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode));
        match opcode {
            OpCode::BBE4INV => 8,
            _ => 12,
        }
    }

    pub fn calculate(&mut self, clk: usize, instruction: Instruction<F>) -> [F; EXTENSION_DEGREE] {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            op_f: _f,
            op_g: _g,
            debug: _debug,
        } = instruction;
        assert!(FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode));

        let operand1 = self.read_extension_element(d, op_b);
        let operand2 = if opcode == OpCode::BBE4INV {
            // 4 disabled reads
            for _ in 0..4 {
                self.memory.disabled_op(e, OpType::Read);
            }
            [F::zero(); EXTENSION_DEGREE]
        } else {
            self.read_extension_element(e, op_c)
        };

        let result = FieldExtensionArithmetic::solve(opcode, operand1, operand2).unwrap();

        self.write_extension_element(d, op_a, result);

        self.operations.push(FieldExtensionArithmeticOperation {
            clk,
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            operand1,
            operand2,
            result,
        });

        result
    }

    fn read_extension_element(&mut self, address_space: F, address: F) -> [F; EXTENSION_DEGREE] {
        assert_ne!(address_space, F::zero());

        let mut result = [F::zero(); EXTENSION_DEGREE];

        for (i, result_elem) in result.iter_mut().enumerate() {
            let data = self.memory.read_elem(
                address_space,
                address + F::from_canonical_usize(i * WORD_SIZE),
            );

            *result_elem = data;
        }

        result
    }

    fn write_extension_element(
        &mut self,
        address_space: F,
        address: F,
        result: [F; EXTENSION_DEGREE],
    ) {
        assert_ne!(address_space, F::zero());

        for (i, row) in result.iter().enumerate() {
            self.memory.write_elem(
                address_space,
                address + F::from_canonical_usize(i * WORD_SIZE),
                *row,
            );
        }
    }

    pub fn current_height(&self) -> usize {
        self.operations.len()
    }

    pub fn make_blank_row(&self) -> FieldExtensionArithmeticCols<WORD_SIZE, F> {
        let mut trace_builder = MemoryTraceBuilder::<NUM_WORDS, WORD_SIZE, F>::new(
            self.memory_manager.clone(),
            self.range_checker.clone(),
            self.air.mem_oc,
        );

        let clk = self.memory_manager.borrow().get_clk();

        for _ in 0..8 {
            trace_builder.disabled_op(F::zero(), OpType::Read);
        }
        for _ in 0..4 {
            trace_builder.disabled_op(F::zero(), OpType::Write);
        }
        let mut mem_oc_aux_iter = trace_builder.take_accesses_buffer().into_iter();

        FieldExtensionArithmeticCols {
            io: FieldExtensionArithmeticIoCols {
                opcode: F::from_canonical_u32(OpCode::FE4ADD as u32),
                clk,
                x: [F::zero(); EXTENSION_DEGREE],
                y: [F::zero(); EXTENSION_DEGREE],
                z: [F::zero(); EXTENSION_DEGREE],
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: F::zero(),
                valid_y_read: F::zero(),
                op_a: F::zero(),
                op_b: F::zero(),
                op_c: F::zero(),
                d: F::zero(),
                e: F::zero(),
                is_add: F::one(),
                is_sub: F::zero(),
                is_mul: F::zero(),
                is_inv: F::zero(),
                inv: [F::zero(); EXTENSION_DEGREE],
                mem_oc_aux_cols: array::from_fn(|_| mem_oc_aux_iter.next().unwrap()),
            },
        }
    }
}

pub struct FieldExtensionArithmetic;

impl FieldExtensionArithmetic {
    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for opcodes not in cpu::FIELD_EXTENSION_INSTRUCTIONS.
    pub fn solve<T: Field>(
        op: OpCode,
        operand1: [T; EXTENSION_DEGREE],
        operand2: [T; EXTENSION_DEGREE],
    ) -> Option<[T; EXTENSION_DEGREE]> {
        match op {
            OpCode::FE4ADD => Some(Self::add(operand1, operand2)),
            OpCode::FE4SUB => Some(Self::subtract(operand1, operand2)),
            OpCode::BBE4MUL => Some(Self::multiply(operand1, operand2)),
            OpCode::BBE4INV => Some(Self::invert(operand1)),
            _ => None,
        }
    }

    pub(crate) fn add<V, E>(x: [V; 4], y: [V; 4]) -> [E; 4]
    where
        V: Copy,
        V: Add<V, Output = E>,
    {
        array::from_fn(|i| x[i] + y[i])
    }

    pub(crate) fn subtract<V, E>(x: [V; 4], y: [V; 4]) -> [E; 4]
    where
        V: Copy,
        V: Sub<V, Output = E>,
    {
        array::from_fn(|i| x[i] - y[i])
    }

    pub(crate) fn multiply<V, E>(x: [V; 4], y: [V; 4]) -> [E; 4]
    where
        E: AbstractField,
        V: Copy,
        V: Mul<V, Output = E>,
        E: Mul<V, Output = E>,
        V: Add<V, Output = E>,
        E: Add<V, Output = E>,
    {
        let [x0, x1, x2, x3] = x;
        let [y0, y1, y2, y3] = y;
        [
            x0 * y0 + (x1 * y3 + x2 * y2 + x3 * y1) * E::from_canonical_usize(BETA),
            x0 * y1 + x1 * y0 + (x2 * y3 + x3 * y2) * E::from_canonical_usize(BETA),
            x0 * y2 + x1 * y1 + x2 * y0 + (x3 * y3) * E::from_canonical_usize(BETA),
            x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0,
        ]
    }

    fn invert<T: Field>(a: [T; 4]) -> [T; 4] {
        // Let a = (a0, a1, a2, a3) represent the element we want to invert.
        // Define a' = (a0, -a1, a2, -a3).  By construction, the product b = a * a' will have zero
        // degree-1 and degree-3 coefficients.
        // Let b = (b0, 0, b2, 0) and define b' = (x, 0, -y, 0).
        // Note that c = b * b' = x^2 - BETA * y^2, which is an element of the base field.
        // Therefore, the inverse of a is 1 / a = a' / (a * a') = a' * b' / (b * b') = a' * b' / c.

        let [a0, a1, a2, a3] = a;

        let beta = T::from_canonical_usize(BETA);

        let mut b0 = a0 * a0 - beta * (T::two() * a1 * a3 - a2 * a2);
        let mut b2 = T::two() * a0 * a2 - a1 * a1 - beta * a3 * a3;

        let c = b0 * b0 - beta * b2 * b2;
        let inv_c = c.inverse();

        b0 *= inv_c;
        b2 *= inv_c;

        [
            a0 * b0 - a2 * b2 * beta,
            -a1 * b0 + a3 * b2 * beta,
            -a0 * b2 + a2 * b0,
            a1 * b2 - a3 * b0,
        ]
    }
}
