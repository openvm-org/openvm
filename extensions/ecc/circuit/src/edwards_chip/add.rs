use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
};

use derive_more::derive::{Deref, DerefMut};
use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit::{
    arch::{ExecutionBridge, *},
    system::memory::{
        offline_checker::MemoryBridge, online::GuestMemory, SharedMemoryHelper, POINTER_MAX_BITS,
    },
};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_ecc_transpiler::Rv32EdwardsOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, ExprBuilder, ExprBuilderConfig, FieldExpr,
    FieldExpressionCoreAir, FieldExpressionExecutor, FieldExpressionFiller,
};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterExecutor, Rv32VecHeapAdapterFiller,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{utils::jacobi, EdwardsAir, EdwardsChip};
use crate::edwards_chip::curves::{get_te_curve_type, te_add, TeCurveType};

pub fn te_add_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let d = ExprBuilder::new_const(builder.clone(), d_biguint.clone());
    let one = ExprBuilder::new_const(builder.clone(), BigUint::one());

    let x1y2 = x1.clone() * y2.clone();
    let x2y1 = x2.clone() * y1.clone();
    let y1y2 = y1 * y2;
    let x1x2 = x1 * x2;
    let dx1x2y1y2 = d * x1x2.clone() * y1y2.clone();

    let mut x3 = (x1y2 + x2y1) / (one.clone() + dx1x2y1y2.clone());
    let mut y3 = (y1y2 - a * x1x2) / (one - dx1x2y1y2);

    x3.save_output();
    y3.save_output();

    let builder = (*builder).borrow().clone();

    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_biguint, d_biguint])
}

#[derive(Clone, PreflightExecutor, Deref, DerefMut)]
pub struct TeAddExecutor<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub(crate)  FieldExpressionExecutor<
        Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    >,
);

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> (FieldExpr, Vec<usize>) {
    let expr = te_add_expr(config, range_checker_bus, a_biguint, d_biguint);

    let local_opcode_idx = vec![
        Rv32EdwardsOpcode::TE_ADD as usize,
        Rv32EdwardsOpcode::SETUP_TE_ADD as usize,
    ];

    (expr, local_opcode_idx)
}

pub fn get_te_add_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> EdwardsAir<2, BLOCKS, BLOCK_SIZE> {
    // Ensure that the addition operation is complete
    assert!(jacobi(&a_biguint.clone().into(), &config.modulus.clone().into()) == 1);
    assert!(jacobi(&d_biguint.clone().into(), &config.modulus.clone().into()) == -1);

    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint, d_biguint);
    EdwardsAir::new(
        Rv32VecHeapAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
    )
}

pub fn get_te_add_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> TeAddExecutor<BLOCKS, BLOCK_SIZE> {
    // Ensure that the addition operation is complete
    assert!(jacobi(&a_biguint.clone().into(), &config.modulus.clone().into()) == 1);
    assert!(jacobi(&d_biguint.clone().into(), &config.modulus.clone().into()) == -1);

    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint, d_biguint);
    TeAddExecutor(FieldExpressionExecutor::new(
        Rv32VecHeapAdapterExecutor::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        vec![],
        "TeAdd",
    ))
}

pub fn get_te_add_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> EdwardsChip<F, 2, BLOCKS, BLOCK_SIZE> {
    // Ensure that the addition operation is complete
    assert!(jacobi(&a_biguint.clone().into(), &config.modulus.clone().into()) == 1);
    assert!(jacobi(&d_biguint.clone().into(), &config.modulus.clone().into()) == -1);

    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker.bus(), a_biguint, d_biguint);
    EdwardsChip::new(
        FieldExpressionFiller::new(
            Rv32VecHeapAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            true,
        ),
        mem_helper,
    )
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct TeAddPreCompute<'a> {
    expr: &'a FieldExpr,
    rs_addrs: [u8; 2],
    a: u8,
    flag_idx: u8,
}

impl<'a, const BLOCKS: usize, const BLOCK_SIZE: usize> TeAddExecutor<BLOCKS, BLOCK_SIZE> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut TeAddPreCompute<'a>,
    ) -> Result<bool, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        // Validate instruction format
        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        if d != RV32_REGISTER_AS || e != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = opcode.local_opcode_idx(self.offset);

        // Pre-compute flag_idx
        let needs_setup = self.expr.needs_setup();
        let mut flag_idx = self.expr.num_flags() as u8;
        if needs_setup {
            // Find which opcode this is in our local_opcode_idx list
            if let Some(opcode_position) = self
                .local_opcode_idx
                .iter()
                .position(|&idx| idx == local_opcode)
            {
                // If this is NOT the last opcode (setup), get the corresponding flag_idx
                if opcode_position < self.opcode_flag_idx.len() {
                    flag_idx = self.opcode_flag_idx[opcode_position] as u8;
                }
            }
        }

        let rs_addrs = from_fn(|i| if i == 0 { b } else { c } as u8);
        *data = TeAddPreCompute {
            expr: &self.expr,
            rs_addrs,
            a: a as u8,
            flag_idx,
        };

        let local_opcode = opcode.local_opcode_idx(self.offset);
        let is_setup = local_opcode == Rv32EdwardsOpcode::SETUP_TE_ADD as usize;

        Ok(is_setup)
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> Executor<F>
    for TeAddExecutor<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<TeAddPreCompute>()
    }

    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: E1ExecutionCtx,
    {
        let pre_compute: &mut TeAddPreCompute = data.borrow_mut();

        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        if let Some(curve_type) = {
            let modulus = &pre_compute.expr.builder.prime;
            let a_coeff = &pre_compute.expr.setup_values[0];
            let d_coeff = &pre_compute.expr.setup_values[1];
            get_te_curve_type(modulus, a_coeff, d_coeff)
        } {
            match (is_setup, curve_type) {
                (true, TeCurveType::ED25519) => Ok(execute_e12_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { TeCurveType::ED25519 as u8 },
                    true,
                >),
                (false, TeCurveType::ED25519) => Ok(execute_e12_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { TeCurveType::ED25519 as u8 },
                    false,
                >),
                _ => panic!("Unsupported curve type"),
            }
        } else if is_setup {
            Ok(execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, { u8::MAX }, true>)
        } else {
            Ok(execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, { u8::MAX }, false>)
        }
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> MeteredExecutor<F>
    for TeAddExecutor<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<TeAddPreCompute>>()
    }

    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: E2ExecutionCtx,
    {
        let pre_compute: &mut E2PreCompute<TeAddPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let is_setup = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        if let Some(curve_type) = {
            let modulus = &pre_compute.data.expr.builder.prime;
            let a_coeff = &pre_compute.data.expr.setup_values[0];
            let d_coeff = &pre_compute.data.expr.setup_values[1];
            get_te_curve_type(modulus, a_coeff, d_coeff)
        } {
            match (is_setup, curve_type) {
                (true, TeCurveType::ED25519) => Ok(execute_e2_setup_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { TeCurveType::ED25519 as u8 },
                >),
                (false, TeCurveType::ED25519) => {
                    Ok(execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { TeCurveType::ED25519 as u8 }>)
                }
                _ => panic!("Unsupported curve type"),
            }
        } else if is_setup {
            Ok(execute_e2_setup_impl::<_, _, BLOCKS, BLOCK_SIZE, { u8::MAX }>)
        } else {
            Ok(execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { u8::MAX }>)
        }
    }
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let e2_pre_compute: &E2PreCompute<TeAddPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(e2_pre_compute.chip_idx as usize, 1);
    let pre_compute = unsafe {
        std::slice::from_raw_parts(
            &e2_pre_compute.data as *const _ as *const u8,
            std::mem::size_of::<TeAddPreCompute>(),
        )
    };
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, CURVE_TYPE, false>(pre_compute, vm_state);
}

unsafe fn execute_e2_setup_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let e2_pre_compute: &E2PreCompute<TeAddPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(e2_pre_compute.chip_idx as usize, 1);
    let pre_compute = unsafe {
        std::slice::from_raw_parts(
            &e2_pre_compute.data as *const _ as *const u8,
            std::mem::size_of::<TeAddPreCompute>(),
        )
    };
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, CURVE_TYPE, true>(&pre_compute, vm_state);
}

unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &TeAddPreCompute = pre_compute.borrow();
    // Read register values
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read memory values for both points
    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });

    if IS_SETUP {
        let input_prime = BigUint::from_bytes_le(read_data[0][..BLOCKS / 2].as_flattened());
        let input_a = BigUint::from_bytes_le(read_data[0][BLOCKS / 2..].as_flattened());
        let input_d = BigUint::from_bytes_le(read_data[1][..BLOCKS / 2].as_flattened());

        if input_prime != pre_compute.expr.prime {
            vm_state.exit_code = Err(ExecutionError::Fail {
                pc: vm_state.pc,
                msg: "TeAdd: mismatched prime",
            });
            return;
        }

        if input_a != pre_compute.expr.setup_values[0] {
            vm_state.exit_code = Err(ExecutionError::Fail {
                pc: vm_state.pc,
                msg: "TeAdd: mismatched a",
            });
            return;
        }

        if input_d != pre_compute.expr.setup_values[1] {
            vm_state.exit_code = Err(ExecutionError::Fail {
                pc: vm_state.pc,
                msg: "TeAdd: mismatched d",
            });
            return;
        }
    }

    let output_data = if CURVE_TYPE == u8::MAX {
        let read_data: DynArray<u8> = read_data.into();
        run_field_expression_precomputed::<true>(
            pre_compute.expr,
            pre_compute.flag_idx as usize,
            &read_data.0,
        )
        .into()
    } else {
        te_add::<CURVE_TYPE, BLOCKS, BLOCK_SIZE>(read_data)
    };

    let rd_val = u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    // Write output data to memory
    for (i, block) in output_data.into_iter().enumerate() {
        vm_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}
