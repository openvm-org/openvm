use alloc::{collections::BTreeMap, vec};
use std::collections::BTreeSet;

use openvm_circuit::arch::instructions::instruction::DebugInfo;
use openvm_stark_backend::p3_field::{ExtensionField, Field, PrimeField32, TwoAdicField};

use super::{config::AsmConfig, AssemblyCode, BasicBlock, IndexTriple, ValueOrConst};
use crate::{
    asm::AsmInstruction,
    ir::{Array, DslIr, Ext, Felt, Ptr, RVar, Usize, Var},
    prelude::TracedVec,
};

/// The memory location for the top of memory
pub const MEMORY_TOP: u32 = (1 << 29) - 4;

// The memory location for the start of the heap.
pub(crate) const HEAP_START_ADDRESS: i32 = 1 << 24;

/// The heap pointer address.
pub(crate) const HEAP_PTR: i32 = HEAP_START_ADDRESS - 4;
/// Utility register.
pub(crate) const A0: i32 = HEAP_START_ADDRESS - 8;

/// The memory location for the top of the stack.
pub(crate) const STACK_TOP: i32 = HEAP_START_ADDRESS - 64;

/// The assembly compiler.
// #[derive(Debug, Clone, Default)]
pub struct AsmCompiler<F, EF> {
    basic_blocks: Vec<BasicBlock<F, EF>>,
    break_label: Option<F>,
    break_label_map: BTreeMap<F, F>,
    break_counter: usize,
    contains_break: BTreeSet<F>,
    function_labels: BTreeMap<String, F>,
    trap_label: F,
    word_size: usize,
}

impl<F> Var<F> {
    /// Gets the frame pointer for a var.
    pub const fn fp(&self) -> i32 {
        // Vars are stored in stack positions 1, 2, 9, 10, 17, 18, ...
        STACK_TOP - (8 * (self.0 / 2) + 1 + (self.0 % 2)) as i32
    }
}

impl<F> Felt<F> {
    /// Gets the frame pointer for a felt.
    pub const fn fp(&self) -> i32 {
        // Felts are stored in stack positions 3, 4, 11, 12, 19, 20, ...
        STACK_TOP - (((self.0 >> 1) << 3) + 3 + (self.0 & 1)) as i32
    }
}

impl<F, EF> Ext<F, EF> {
    /// Gets the frame pointer for an extension element
    pub const fn fp(&self) -> i32 {
        // Exts are stored in stack positions 5-8, 13-16, 21-24, ...
        STACK_TOP - 8 * self.0 as i32
    }
}

impl<F> Ptr<F> {
    /// Gets the frame pointer for a pointer.
    pub const fn fp(&self) -> i32 {
        self.address.fp()
    }
}

impl<F: PrimeField32 + TwoAdicField, EF: ExtensionField<F> + TwoAdicField> AsmCompiler<F, EF> {
    /// Creates a new [AsmCompiler].
    pub fn new(word_size: usize) -> Self {
        Self {
            basic_blocks: vec![BasicBlock::new()],
            break_label: None,
            break_label_map: BTreeMap::new(),
            contains_break: BTreeSet::new(),
            function_labels: BTreeMap::new(),
            break_counter: 0,
            trap_label: F::ONE,
            word_size,
        }
    }

    /// Creates a new break label.
    pub fn new_break_label(&mut self) -> F {
        let label = self.break_counter;
        self.break_counter += 1;
        let label = F::from_canonical_usize(label);
        self.break_label = Some(label);
        label
    }

    /// Builds the operations into assembly instructions.
    pub fn build(&mut self, operations: TracedVec<DslIr<AsmConfig<F, EF>>>) {
        if self.block_label().is_zero() {
            // Initialize the heap pointer value.
            let heap_start = F::from_canonical_u32(HEAP_START_ADDRESS as u32);
            self.push(AsmInstruction::ImmF(HEAP_PTR, heap_start), None);
            // Jump over the TRAP instruction we are about to add.
            self.push(AsmInstruction::j(self.trap_label + F::ONE), None);
            self.basic_block();
            // Add a TRAP instruction used as jump destination for all failed assertions.
            assert_eq!(self.block_label(), self.trap_label);
            self.push(AsmInstruction::Trap, None);
            self.basic_block();
        }
        // For each operation, generate assembly instructions.
        for (op, trace) in operations.clone() {
            let debug_info = Some(DebugInfo::new(op.to_string(), trace));
            match op {
                DslIr::ImmV(dst, src) => {
                    self.push(AsmInstruction::ImmF(dst.fp(), src), debug_info);
                }
                DslIr::ImmF(dst, src) => {
                    self.push(AsmInstruction::ImmF(dst.fp(), src), debug_info);
                }
                DslIr::ImmE(dst, src) => {
                    self.assign_exti(dst.fp(), src, debug_info);
                }
                DslIr::AddV(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::AddF(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::AddVI(dst, lhs, rhs) => {
                    self.push(AsmInstruction::AddFI(dst.fp(), lhs.fp(), rhs), debug_info);
                }
                DslIr::AddF(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::AddF(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::AddFI(dst, lhs, rhs) => {
                    self.push(AsmInstruction::AddFI(dst.fp(), lhs.fp(), rhs), debug_info);
                }
                DslIr::AddE(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::AddE(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::AddEI(dst, lhs, rhs) => {
                    self.add_ext_exti(dst, lhs, rhs, debug_info);
                }
                DslIr::AddEF(dst, lhs, rhs) => {
                    self.add_ext_felt(dst, lhs, rhs, debug_info);
                }
                DslIr::AddEFFI(dst, lhs, rhs) => {
                    self.add_felt_exti(dst, lhs, rhs, debug_info);
                }
                DslIr::AddEFI(dst, lhs, rhs) => {
                    self.add_ext_exti(dst, lhs, EF::from_base(rhs), debug_info);
                }
                DslIr::SubV(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::SubF(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::SubVI(dst, lhs, rhs) => {
                    self.push(AsmInstruction::SubFI(dst.fp(), lhs.fp(), rhs), debug_info);
                }
                DslIr::SubVIN(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::SubFIN(dst.fp(), lhs, rhs.fp()),
                        debug_info.clone(),
                    );
                }
                DslIr::SubF(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::SubF(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::SubFI(dst, lhs, rhs) => {
                    self.push(AsmInstruction::SubFI(dst.fp(), lhs.fp(), rhs), debug_info);
                }
                DslIr::SubFIN(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::SubFIN(dst.fp(), lhs, rhs.fp()),
                        debug_info.clone(),
                    );
                }
                DslIr::NegV(dst, src) => {
                    self.push(
                        AsmInstruction::MulFI(dst.fp(), src.fp(), F::NEG_ONE),
                        debug_info,
                    );
                }
                DslIr::NegF(dst, src) => {
                    self.push(
                        AsmInstruction::MulFI(dst.fp(), src.fp(), F::NEG_ONE),
                        debug_info,
                    );
                }
                DslIr::DivF(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::DivF(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::DivFI(dst, lhs, rhs) => {
                    self.push(AsmInstruction::DivFI(dst.fp(), lhs.fp(), rhs), debug_info);
                }
                DslIr::DivFIN(dst, lhs, rhs) => {
                    self.push(AsmInstruction::DivFIN(dst.fp(), lhs, rhs.fp()), debug_info);
                }
                DslIr::DivEIN(dst, lhs, rhs) => {
                    self.assign_exti(A0, lhs, debug_info.clone());
                    self.push(AsmInstruction::DivE(dst.fp(), A0, rhs.fp()), debug_info);
                }
                DslIr::DivE(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::DivE(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::DivEI(dst, lhs, rhs) => {
                    self.assign_exti(A0, rhs, debug_info.clone());
                    self.push(AsmInstruction::DivE(dst.fp(), lhs.fp(), A0), debug_info);
                }
                DslIr::DivEF(dst, lhs, rhs) => {
                    self.div_ext_felt(dst, lhs, rhs, debug_info);
                }
                DslIr::DivEFI(dst, lhs, rhs) => {
                    self.mul_ext_felti(dst, lhs, rhs.inverse(), debug_info);
                }
                DslIr::SubEF(dst, lhs, rhs) => {
                    self.sub_ext_felt(dst, lhs, rhs, debug_info);
                }
                DslIr::SubEFI(dst, lhs, rhs) => {
                    self.add_ext_exti(dst, lhs, EF::from_base(rhs.neg()), debug_info);
                }
                DslIr::SubEIN(dst, lhs, rhs) => {
                    self.sub_exti_ext(dst, lhs, rhs, debug_info.clone());
                }
                DslIr::SubE(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::SubE(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::SubEI(dst, lhs, rhs) => {
                    self.add_ext_exti(dst, lhs, rhs.neg(), debug_info);
                }
                DslIr::NegE(dst, src) => {
                    self.mul_ext_felti(dst, src, F::NEG_ONE, debug_info);
                }
                DslIr::MulV(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::MulF(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::MulVI(dst, lhs, rhs) => {
                    self.push(AsmInstruction::MulFI(dst.fp(), lhs.fp(), rhs), debug_info);
                }
                DslIr::MulF(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::MulF(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::MulFI(dst, lhs, rhs) => {
                    self.push(AsmInstruction::MulFI(dst.fp(), lhs.fp(), rhs), debug_info);
                }
                DslIr::MulE(dst, lhs, rhs) => {
                    self.push(
                        AsmInstruction::MulE(dst.fp(), lhs.fp(), rhs.fp()),
                        debug_info,
                    );
                }
                DslIr::MulEI(dst, lhs, rhs) => {
                    self.assign_exti(A0, rhs, debug_info.clone());
                    self.push(AsmInstruction::MulE(dst.fp(), lhs.fp(), A0), debug_info);
                }
                DslIr::MulEF(dst, lhs, rhs) => {
                    self.mul_ext_felt(dst, lhs, rhs, debug_info);
                }
                DslIr::MulEFI(dst, lhs, rhs) => {
                    self.mul_ext_felti(dst, lhs, rhs, debug_info);
                }
                DslIr::CastFV(dst, src) => {
                    self.push(
                        AsmInstruction::AddFI(dst.fp(), src.fp(), F::ZERO),
                        debug_info,
                    );
                }
                DslIr::UnsafeCastVF(dst, src) => {
                    self.push(
                        AsmInstruction::AddFI(dst.fp(), src.fp(), F::ZERO),
                        debug_info,
                    );
                }
                DslIr::IfEq(lhs, rhs, then_block, else_block) => {
                    let if_compiler = IfCompiler {
                        compiler: self,
                        lhs: lhs.fp(),
                        rhs: ValueOrConst::Val(rhs.fp()),
                        is_eq: true,
                    };
                    if else_block.is_empty() {
                        if_compiler.then(|builder| builder.build(then_block), debug_info);
                    } else {
                        if_compiler.then_or_else(
                            |builder| builder.build(then_block),
                            |builder| builder.build(else_block),
                            debug_info,
                        );
                    }
                }
                DslIr::IfNe(lhs, rhs, then_block, else_block) => {
                    let if_compiler = IfCompiler {
                        compiler: self,
                        lhs: lhs.fp(),
                        rhs: ValueOrConst::Val(rhs.fp()),
                        is_eq: false,
                    };
                    if else_block.is_empty() {
                        if_compiler.then(|builder| builder.build(then_block), debug_info);
                    } else {
                        if_compiler.then_or_else(
                            |builder| builder.build(then_block),
                            |builder| builder.build(else_block),
                            debug_info,
                        );
                    }
                }
                DslIr::IfEqI(lhs, rhs, then_block, else_block) => {
                    let if_compiler = IfCompiler {
                        compiler: self,
                        lhs: lhs.fp(),
                        rhs: ValueOrConst::Const(rhs),
                        is_eq: true,
                    };
                    if else_block.is_empty() {
                        if_compiler.then(|builder| builder.build(then_block), debug_info);
                    } else {
                        if_compiler.then_or_else(
                            |builder| builder.build(then_block),
                            |builder| builder.build(else_block),
                            debug_info,
                        );
                    }
                }
                DslIr::IfNeI(lhs, rhs, then_block, else_block) => {
                    let if_compiler = IfCompiler {
                        compiler: self,
                        lhs: lhs.fp(),
                        rhs: ValueOrConst::Const(rhs),
                        is_eq: false,
                    };
                    if else_block.is_empty() {
                        if_compiler.then(|builder| builder.build(then_block), debug_info);
                    } else {
                        if_compiler.then_or_else(
                            |builder| builder.build(then_block),
                            |builder| builder.build(else_block),
                            debug_info,
                        );
                    }
                }
                DslIr::Break => {
                    let label = self.break_label.expect("No break label set");
                    let current_block = self.block_label();
                    self.contains_break.insert(current_block);
                    self.push(AsmInstruction::Break(label), debug_info);
                }
                DslIr::For(start, end, step_size, loop_var, block) => {
                    let for_compiler = ForCompiler {
                        compiler: self,
                        start,
                        end,
                        step_size,
                        loop_var,
                    };
                    for_compiler.for_each(move |_, builder| builder.build(block), debug_info);
                }
                DslIr::Loop(block) => {
                    let loop_compiler = LoopCompiler { compiler: self };
                    loop_compiler.compile(move |builder| builder.build(block), debug_info);
                }
                DslIr::AssertEqV(lhs, rhs) => {
                    // If lhs != rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::Val(rhs.fp()), false, debug_info)
                }
                DslIr::AssertEqVI(lhs, rhs) => {
                    // If lhs != rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::Const(rhs), false, debug_info)
                }
                DslIr::AssertNeV(lhs, rhs) => {
                    // If lhs == rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::Val(rhs.fp()), true, debug_info)
                }
                DslIr::AssertNeVI(lhs, rhs) => {
                    // If lhs == rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::Const(rhs), true, debug_info)
                }
                DslIr::AssertEqF(lhs, rhs) => {
                    // If lhs != rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::Val(rhs.fp()), false, debug_info)
                }
                DslIr::AssertEqFI(lhs, rhs) => {
                    // If lhs != rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::Const(rhs), false, debug_info)
                }
                DslIr::AssertNeF(lhs, rhs) => {
                    // If lhs == rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::Val(rhs.fp()), true, debug_info)
                }
                DslIr::AssertNeFI(lhs, rhs) => {
                    // If lhs == rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::Const(rhs), true, debug_info)
                }
                DslIr::AssertEqE(lhs, rhs) => {
                    // If lhs != rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::ExtVal(rhs.fp()), false, debug_info)
                }
                DslIr::AssertEqEI(lhs, rhs) => {
                    // If lhs != rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::ExtConst(rhs), false, debug_info)
                }
                DslIr::AssertNeE(lhs, rhs) => {
                    // If lhs == rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::ExtVal(rhs.fp()), true, debug_info)
                }
                DslIr::AssertNeEI(lhs, rhs) => {
                    // If lhs == rhs, execute TRAP
                    self.assert(lhs.fp(), ValueOrConst::ExtConst(rhs), true, debug_info)
                }
                DslIr::Alloc(ptr, len, size) => {
                    self.alloc(ptr, len, size, debug_info);
                }
                DslIr::LoadV(var, ptr, index) => match index.fp() {
                    IndexTriple::Const(index, offset, size) => self.push(
                        AsmInstruction::LoadFI(var.fp(), ptr.fp(), index, size, offset),
                        debug_info.clone(),
                    ),
                    IndexTriple::Var(index, offset, size) => {
                        self.add_scaled(A0, ptr.fp(), index, size, debug_info.clone());
                        self.push(
                            AsmInstruction::LoadFI(var.fp(), A0, F::ZERO, F::ZERO, offset),
                            debug_info.clone(),
                        )
                    }
                },
                DslIr::LoadF(var, ptr, index) => match index.fp() {
                    IndexTriple::Const(index, offset, size) => self.push(
                        AsmInstruction::LoadFI(var.fp(), ptr.fp(), index, size, offset),
                        debug_info.clone(),
                    ),
                    IndexTriple::Var(index, offset, size) => {
                        self.add_scaled(A0, ptr.fp(), index, size, debug_info.clone());
                        self.push(
                            AsmInstruction::LoadFI(var.fp(), A0, F::ZERO, F::ZERO, offset),
                            debug_info.clone(),
                        )
                    }
                },
                DslIr::LoadE(var, ptr, index) => match index.fp() {
                    IndexTriple::Const(index, offset, size) => {
                        self.load_ext(var, ptr.fp(), index * size + offset, debug_info)
                    }
                    IndexTriple::Var(index, offset, size) => {
                        self.add_scaled(A0, ptr.fp(), index, size, debug_info.clone());
                        self.load_ext(var, A0, offset, debug_info)
                    }
                },
                DslIr::LoadHeapPtr(ptr) => self.push(
                    AsmInstruction::AddFI(ptr.fp(), HEAP_PTR, F::ZERO),
                    debug_info,
                ),
                DslIr::StoreV(var, ptr, index) => match index.fp() {
                    IndexTriple::Const(index, offset, size) => self.push(
                        AsmInstruction::StoreFI(var.fp(), ptr.fp(), index, size, offset),
                        debug_info.clone(),
                    ),
                    IndexTriple::Var(index, offset, size) => {
                        self.add_scaled(A0, ptr.fp(), index, size, debug_info.clone());
                        self.push(
                            AsmInstruction::StoreFI(var.fp(), A0, F::ZERO, F::ZERO, offset),
                            debug_info.clone(),
                        )
                    }
                },
                DslIr::StoreF(var, ptr, index) => match index.fp() {
                    IndexTriple::Const(index, offset, size) => self.push(
                        AsmInstruction::StoreFI(var.fp(), ptr.fp(), index, size, offset),
                        debug_info.clone(),
                    ),
                    IndexTriple::Var(index, offset, size) => {
                        self.add_scaled(A0, ptr.fp(), index, size, debug_info.clone());
                        self.push(
                            AsmInstruction::StoreFI(var.fp(), A0, F::ZERO, F::ZERO, offset),
                            debug_info.clone(),
                        )
                    }
                },
                DslIr::StoreE(var, ptr, index) => match index.fp() {
                    IndexTriple::Const(index, offset, size) => {
                        self.store_ext(var, ptr.fp(), index * size + offset, debug_info)
                    }
                    IndexTriple::Var(index, offset, size) => {
                        self.add_scaled(A0, ptr.fp(), index, size, debug_info.clone());
                        self.store_ext(var, A0, offset, debug_info)
                    }
                },
                DslIr::StoreHeapPtr(ptr) => self.push(
                    AsmInstruction::AddFI(HEAP_PTR, ptr.fp(), F::ZERO),
                    debug_info,
                ),
                DslIr::HintBitsF(var, len) => {
                    self.push(AsmInstruction::HintBits(var.fp(), len), debug_info);
                }
                DslIr::HintBitsV(var, len) => {
                    self.push(AsmInstruction::HintBits(var.fp(), len), debug_info);
                }
                DslIr::HintBitsU(_) => {
                    todo!()
                }
                DslIr::Poseidon2PermuteBabyBear(dst, src) => match (dst, src) {
                    (Array::Dyn(dst, _), Array::Dyn(src, _)) => self.push(
                        AsmInstruction::Poseidon2Permute(dst.fp(), src.fp()),
                        debug_info,
                    ),
                    _ => unimplemented!(),
                },
                DslIr::Poseidon2CompressBabyBear(result, left, right) => {
                    match (result, left, right) {
                        (Array::Dyn(result, _), Array::Dyn(left, _), Array::Dyn(right, _)) => self
                            .push(
                                AsmInstruction::Poseidon2Compress(
                                    result.fp(),
                                    left.fp(),
                                    right.fp(),
                                ),
                                debug_info,
                            ),
                        _ => unimplemented!(),
                    }
                }
                DslIr::Error() => self.push(AsmInstruction::j(self.trap_label), debug_info),
                DslIr::PrintF(dst) => {
                    self.push(AsmInstruction::PrintF(dst.fp()), debug_info);
                }
                DslIr::PrintV(dst) => {
                    self.push(AsmInstruction::PrintV(dst.fp()), debug_info);
                }
                DslIr::PrintE(dst) => {
                    self.push(AsmInstruction::PrintE(dst.fp()), debug_info);
                }
                DslIr::HintInputVec() => {
                    self.push(AsmInstruction::HintInputVec(), debug_info);
                }
                DslIr::StoreHintWord(ptr, index) => match index.fp() {
                    IndexTriple::Const(index, offset, size) => self.push(
                        AsmInstruction::StoreHintWordI(ptr.fp(), size * index + offset),
                        debug_info.clone(),
                    ),
                    IndexTriple::Var(index, offset, size) => {
                        self.add_scaled(A0, ptr.fp(), index, size, debug_info.clone());
                        self.push(AsmInstruction::StoreHintWordI(A0, offset), debug_info)
                    }
                },
                DslIr::Publish(val, index) => {
                    self.push(AsmInstruction::Publish(val.fp(), index.fp()), debug_info);
                }
                DslIr::CycleTrackerStart(name) => {
                    self.push(
                        AsmInstruction::CycleTrackerStart(),
                        Some(DebugInfo {
                            dsl_instruction: format!("CT-{}", name),
                            trace: None,
                        }),
                    );
                }
                DslIr::CycleTrackerEnd(name) => {
                    self.push(
                        AsmInstruction::CycleTrackerEnd(),
                        Some(DebugInfo {
                            dsl_instruction: format!("CT-{}", name),
                            trace: None,
                        }),
                    );
                }
                DslIr::Halt => {
                    self.push(AsmInstruction::Halt, debug_info);
                }
                DslIr::FriReducedOpening(alpha, curr_alpha_pow, at_x_array, at_z_array, result) => {
                    self.push(
                        AsmInstruction::FriReducedOpening(
                            at_x_array.ptr().fp(),
                            at_z_array.ptr().fp(),
                            result.fp(),
                            match at_z_array.len() {
                                Usize::Const(_) => panic!(
                                    "FriFold does not currently support constant length arrays"
                                ),
                                Usize::Var(len) => len.fp(),
                            },
                            alpha.fp(),
                            curr_alpha_pow.fp(),
                        ),
                        debug_info,
                    );
                }
                _ => unimplemented!(),
            }
        }
    }

    pub fn alloc(
        &mut self,
        ptr: Ptr<F>,
        len: impl Into<RVar<F>>,
        size: usize,
        debug_info: Option<DebugInfo>,
    ) {
        let word_size = self.word_size;
        let align = |x: usize| x.div_ceil(word_size) * word_size;
        // Load the current heap ptr address to the stack value and advance the heap ptr.
        let len = len.into();
        match len {
            RVar::Const(len) => {
                self.push(
                    AsmInstruction::CopyF(ptr.fp(), HEAP_PTR),
                    debug_info.clone(),
                );
                let inc = F::from_canonical_usize(align((len.as_canonical_u32() as usize) * size));
                self.push(AsmInstruction::AddFI(HEAP_PTR, HEAP_PTR, inc), debug_info);
            }
            RVar::Val(len) => {
                self.push(
                    AsmInstruction::CopyF(ptr.fp(), HEAP_PTR),
                    debug_info.clone(),
                );
                let size = F::from_canonical_usize(align(size));
                self.push(
                    AsmInstruction::MulFI(A0, len.fp(), size),
                    debug_info.clone(),
                );
                self.push(AsmInstruction::AddF(HEAP_PTR, HEAP_PTR, A0), debug_info);
            }
        }
    }

    pub fn assert(
        &mut self,
        lhs: i32,
        rhs: ValueOrConst<F, EF>,
        is_eq: bool,
        debug_info: Option<DebugInfo>,
    ) {
        let trap_label = self.trap_label;
        let if_compiler = IfCompiler {
            compiler: self,
            lhs,
            rhs,
            is_eq: !is_eq,
        };
        if_compiler.then_label(trap_label, debug_info);
    }

    pub fn code(self) -> AssemblyCode<F, EF> {
        let labels = self
            .function_labels
            .into_iter()
            .map(|(k, v)| (v, k))
            .collect();
        AssemblyCode::new(self.basic_blocks, labels)
    }

    fn basic_block(&mut self) {
        self.basic_blocks.push(BasicBlock::new());
    }

    fn block_label(&mut self) -> F {
        F::from_canonical_usize(self.basic_blocks.len() - 1)
    }

    fn push_to_block(
        &mut self,
        block_label: F,
        instruction: AsmInstruction<F, EF>,
        debug_info: Option<DebugInfo>,
    ) {
        self.basic_blocks
            .get_mut(block_label.as_canonical_u32() as usize)
            .unwrap_or_else(|| panic!("Missing block at label: {:?}", block_label))
            .push(instruction, debug_info);
    }

    fn push(&mut self, instruction: AsmInstruction<F, EF>, debug_info: Option<DebugInfo>) {
        self.basic_blocks
            .last_mut()
            .unwrap()
            .push(instruction, debug_info);
    }

    // mem[dst] <- mem[src] + c * mem[val]
    // assumes dst != src
    fn add_scaled(&mut self, dst: i32, src: i32, val: i32, c: F, debug_info: Option<DebugInfo>) {
        if c == F::ONE {
            self.push(AsmInstruction::AddF(dst, src, val), debug_info);
        } else {
            self.push(AsmInstruction::MulFI(dst, val, c), debug_info.clone());
            self.push(AsmInstruction::AddF(dst, dst, src), debug_info);
        }
    }
}

pub struct IfCompiler<'a, F, EF> {
    compiler: &'a mut AsmCompiler<F, EF>,
    lhs: i32,
    rhs: ValueOrConst<F, EF>,
    is_eq: bool,
}

impl<F: PrimeField32 + TwoAdicField, EF: ExtensionField<F> + TwoAdicField> IfCompiler<'_, F, EF> {
    pub fn then<Func>(self, f: Func, debug_info: Option<DebugInfo>)
    where
        Func: FnOnce(&mut AsmCompiler<F, EF>),
    {
        let Self {
            compiler,
            lhs,
            rhs,
            is_eq,
        } = self;

        // Get the label for the current block.
        let current_block = compiler.block_label();

        // Generate the blocks for the then branch.
        compiler.basic_block();
        f(compiler);

        // Generate the block for returning to the main flow.
        compiler.basic_block();
        let after_if_block = compiler.block_label();

        // Get the branch instruction to push to the `current_block`.
        let instr = Self::branch(lhs, rhs, is_eq, after_if_block);
        compiler.push_to_block(current_block, instr, debug_info);
    }

    pub fn then_label(self, label: F, debug_info: Option<DebugInfo>) {
        let Self {
            compiler,
            lhs,
            rhs,
            is_eq,
        } = self;

        // Get the label for the current block.
        let current_block = compiler.block_label();

        // Get the branch instruction to push to the `current_block`.
        let instr = Self::branch(lhs, rhs, is_eq, label);
        compiler.push_to_block(current_block, instr, debug_info);
    }

    pub fn then_or_else<ThenFunc, ElseFunc>(
        self,
        then_f: ThenFunc,
        else_f: ElseFunc,
        debug_info: Option<DebugInfo>,
    ) where
        ThenFunc: FnOnce(&mut AsmCompiler<F, EF>),
        ElseFunc: FnOnce(&mut AsmCompiler<F, EF>),
    {
        let Self {
            compiler,
            lhs,
            rhs,
            is_eq,
        } = self;

        // Get the label for the current block, so we can generate the jump instruction into it.
        // conditional branc instruction to it, if the condition is not met.
        let if_branching_block = compiler.block_label();

        // Generate the block for the then branch.
        compiler.basic_block();
        then_f(compiler);
        let last_if_block = compiler.block_label();

        // Generate the block for the else branch.
        compiler.basic_block();
        let else_block = compiler.block_label();
        else_f(compiler);

        // Generate the jump instruction to the else block
        let instr = Self::branch(lhs, rhs, is_eq, else_block);
        compiler.push_to_block(if_branching_block, instr, debug_info.clone());

        // Generate the block for returning to the main flow.
        compiler.basic_block();
        let main_flow_block = compiler.block_label();
        let instr = AsmInstruction::j(main_flow_block);
        compiler.push_to_block(last_if_block, instr, debug_info.clone());
    }

    const fn branch(
        lhs: i32,
        rhs: ValueOrConst<F, EF>,
        is_eq: bool,
        block: F,
    ) -> AsmInstruction<F, EF> {
        match (rhs, is_eq) {
            (ValueOrConst::Const(rhs), true) => AsmInstruction::BneI(block, lhs, rhs),
            (ValueOrConst::Const(rhs), false) => AsmInstruction::BeqI(block, lhs, rhs),
            (ValueOrConst::ExtConst(rhs), true) => AsmInstruction::BneEI(block, lhs, rhs),
            (ValueOrConst::ExtConst(rhs), false) => AsmInstruction::BeqEI(block, lhs, rhs),
            (ValueOrConst::Val(rhs), true) => AsmInstruction::Bne(block, lhs, rhs),
            (ValueOrConst::Val(rhs), false) => AsmInstruction::Beq(block, lhs, rhs),
            (ValueOrConst::ExtVal(rhs), true) => AsmInstruction::BneE(block, lhs, rhs),
            (ValueOrConst::ExtVal(rhs), false) => AsmInstruction::BeqE(block, lhs, rhs),
        }
    }
}

/// A builder for a for loop.
///
/// SAFETY: Starting with end < start will lead to undefined behavior.
pub struct ForCompiler<'a, F: Field, EF> {
    compiler: &'a mut AsmCompiler<F, EF>,
    start: RVar<F>,
    end: RVar<F>,
    step_size: F,
    loop_var: Var<F>,
}

impl<F: PrimeField32 + TwoAdicField, EF: ExtensionField<F> + TwoAdicField> ForCompiler<'_, F, EF> {
    pub(super) fn for_each(
        mut self,
        f: impl FnOnce(Var<F>, &mut AsmCompiler<F, EF>),
        debug_info: Option<DebugInfo>,
    ) {
        // The function block structure:
        // - Setting the loop range
        // - Executing the loop body and incrementing the loop variable
        // - the loop condition

        // Set the loop variable to the start of the range.
        self.set_loop_var(debug_info.clone());

        // Save the label of the for loop call.
        let loop_call_label = self.compiler.block_label();

        // Initialize a break label for this loop.
        let break_label = self.compiler.new_break_label();
        self.compiler.break_label = Some(break_label);

        // A basic block for the loop body
        self.compiler.basic_block();

        // Save the loop body label for the loop condition.
        let loop_label = self.compiler.block_label();

        // The loop body.
        f(self.loop_var, self.compiler);

        // Increment the loop variable.
        self.compiler.push(
            AsmInstruction::AddFI(self.loop_var.fp(), self.loop_var.fp(), self.step_size),
            debug_info.clone(),
        );

        // Add a basic block for the loop condition.
        self.compiler.basic_block();

        // Jump to loop body if the loop condition still holds.
        self.jump_to_loop_body(loop_label, debug_info.clone());

        // Add a jump instruction to the loop condition in the loop call block.
        let label = self.compiler.block_label();
        let instr = AsmInstruction::j(label);
        self.compiler
            .push_to_block(loop_call_label, instr, debug_info.clone());

        // Initialize the after loop block.
        self.compiler.basic_block();

        // Resolve the break label.
        let label = self.compiler.block_label();
        self.compiler.break_label_map.insert(break_label, label);

        // Replace the break instruction with a jump to the after loop block.
        for block in self.compiler.contains_break.iter() {
            for instruction in self.compiler.basic_blocks[block.as_canonical_u32() as usize]
                .0
                .iter_mut()
            {
                if let AsmInstruction::Break(l) = instruction {
                    if *l == break_label {
                        *instruction = AsmInstruction::j(label);
                    }
                }
            }
        }

        // self.compiler.contains_break.clear();
    }

    fn set_loop_var(&mut self, debug_info: Option<DebugInfo>) {
        match self.start {
            RVar::Const(start) => {
                self.compiler.push(
                    AsmInstruction::ImmF(self.loop_var.fp(), start),
                    debug_info.clone(),
                );
            }
            RVar::Val(var) => {
                self.compiler.push(
                    AsmInstruction::CopyF(self.loop_var.fp(), var.fp()),
                    debug_info.clone(),
                );
            }
        }
    }

    fn jump_to_loop_body(&mut self, loop_label: F, debug_info: Option<DebugInfo>) {
        match self.end {
            RVar::Const(end) => {
                let instr = AsmInstruction::BneI(loop_label, self.loop_var.fp(), end);
                self.compiler.push(instr, debug_info.clone());
            }
            RVar::Val(end) => {
                let instr = AsmInstruction::Bne(loop_label, self.loop_var.fp(), end.fp());
                self.compiler.push(instr, debug_info.clone());
            }
        }
    }
}

struct LoopCompiler<'a, F: Field, EF> {
    compiler: &'a mut AsmCompiler<F, EF>,
}

impl<F: PrimeField32 + TwoAdicField, EF: ExtensionField<F> + TwoAdicField> LoopCompiler<'_, F, EF> {
    fn compile(
        self,
        compile_body: impl FnOnce(&mut AsmCompiler<F, EF>),
        debug_info: Option<DebugInfo>,
    ) {
        // Initialize a break label for this loop.
        let break_label = self.compiler.new_break_label();
        self.compiler.break_label = Some(break_label);

        // Loop block.
        self.compiler.basic_block();
        let loop_label = self.compiler.block_label();

        compile_body(self.compiler);
        self.compiler
            .push(AsmInstruction::j(loop_label), debug_info.clone());

        // After loop block.
        self.compiler.basic_block();
        let after_loop_label = self.compiler.block_label();
        self.compiler
            .break_label_map
            .insert(break_label, after_loop_label);

        // Replace break instructions with a jump to the after loop block.
        for block in self.compiler.contains_break.iter() {
            for instruction in self.compiler.basic_blocks[block.as_canonical_u32() as usize]
                .0
                .iter_mut()
            {
                if let AsmInstruction::Break(l) = instruction {
                    if *l == break_label {
                        *instruction = AsmInstruction::j(after_loop_label);
                    }
                }
            }
        }
    }
}

// Ext compiler logic.
impl<F: PrimeField32 + TwoAdicField, EF: ExtensionField<F> + TwoAdicField> AsmCompiler<F, EF> {
    fn assign_exti(&mut self, dst: i32, imm: EF, debug_info: Option<DebugInfo>) {
        let imm = imm.as_base_slice();
        for i in 0..EF::D {
            self.push(
                AsmInstruction::ImmF(dst + i as i32, imm[i]),
                debug_info.clone(),
            );
        }
    }

    fn load_ext(&mut self, val: Ext<F, EF>, addr: i32, offset: F, debug_info: Option<DebugInfo>) {
        for i in 0..EF::D {
            self.push(
                AsmInstruction::LoadFI(
                    val.fp() + i as i32,
                    addr,
                    F::from_canonical_usize(i),
                    F::ONE,
                    offset,
                ),
                debug_info.clone(),
            )
        }
    }

    fn store_ext(&mut self, val: Ext<F, EF>, addr: i32, offset: F, debug_info: Option<DebugInfo>) {
        for i in 0..EF::D {
            self.push(
                AsmInstruction::StoreFI(
                    val.fp() + i as i32,
                    addr,
                    F::from_canonical_usize(i),
                    F::ONE,
                    offset,
                ),
                debug_info.clone(),
            )
        }
    }

    fn add_ext_exti(
        &mut self,
        dst: Ext<F, EF>,
        lhs: Ext<F, EF>,
        rhs: EF,
        debug_info: Option<DebugInfo>,
    ) {
        let rhs = rhs.as_base_slice();
        for i in 0..EF::D {
            let j = i as i32;
            self.push(
                AsmInstruction::AddFI(dst.fp() + j, lhs.fp() + j, rhs[i]),
                debug_info.clone(),
            );
        }
    }

    fn sub_exti_ext(
        &mut self,
        dst: Ext<F, EF>,
        lhs: EF,
        rhs: Ext<F, EF>,
        debug_info: Option<DebugInfo>,
    ) {
        let lhs = lhs.as_base_slice();
        for i in 0..EF::D {
            let j = i as i32;
            self.push(
                AsmInstruction::SubFIN(dst.fp() + j, lhs[i], rhs.fp() + j),
                debug_info.clone(),
            );
        }
    }

    fn add_ext_felt(
        &mut self,
        dst: Ext<F, EF>,
        lhs: Ext<F, EF>,
        rhs: Felt<F>,
        debug_info: Option<DebugInfo>,
    ) {
        self.push(
            AsmInstruction::AddF(dst.fp(), lhs.fp(), rhs.fp()),
            debug_info.clone(),
        );
        for i in 1..EF::D {
            let j = i as i32;
            self.push(
                AsmInstruction::CopyF(dst.fp() + j, lhs.fp() + j),
                debug_info.clone(),
            );
        }
    }

    fn sub_ext_felt(
        &mut self,
        dst: Ext<F, EF>,
        lhs: Ext<F, EF>,
        rhs: Felt<F>,
        debug_info: Option<DebugInfo>,
    ) {
        self.push(
            AsmInstruction::SubF(dst.fp(), lhs.fp(), rhs.fp()),
            debug_info.clone(),
        );
        for i in 1..EF::D {
            let j = i as i32;
            self.push(
                AsmInstruction::CopyF(dst.fp() + j, lhs.fp() + j),
                debug_info.clone(),
            );
        }
    }

    fn add_felt_exti(
        &mut self,
        dst: Ext<F, EF>,
        lhs: Felt<F>,
        rhs: EF,
        debug_info: Option<DebugInfo>,
    ) {
        let rhs = rhs.as_base_slice();

        self.push(
            AsmInstruction::CopyF(dst.fp(), lhs.fp()),
            debug_info.clone(),
        );

        for i in 1..EF::D {
            let j = i as i32;
            self.push(
                AsmInstruction::ImmF(dst.fp() + j, rhs[i]),
                debug_info.clone(),
            );
        }
    }

    fn mul_ext_felt(
        &mut self,
        dst: Ext<F, EF>,
        lhs: Ext<F, EF>,
        rhs: Felt<F>,
        debug_info: Option<DebugInfo>,
    ) {
        for i in 0..EF::D {
            let j = i as i32;
            self.push(
                AsmInstruction::MulF(dst.fp() + j, lhs.fp() + j, rhs.fp()),
                debug_info.clone(),
            );
        }
    }

    fn mul_ext_felti(
        &mut self,
        dst: Ext<F, EF>,
        lhs: Ext<F, EF>,
        rhs: F,
        debug_info: Option<DebugInfo>,
    ) {
        for i in 0..EF::D {
            let j = i as i32;
            self.push(
                AsmInstruction::MulFI(dst.fp() + j, lhs.fp() + j, rhs),
                debug_info.clone(),
            );
        }
    }

    fn div_ext_felt(
        &mut self,
        dst: Ext<F, EF>,
        lhs: Ext<F, EF>,
        rhs: Felt<F>,
        debug_info: Option<DebugInfo>,
    ) {
        for i in 0..EF::D {
            let j = i as i32;
            self.push(
                AsmInstruction::DivF(dst.fp() + j, lhs.fp() + j, rhs.fp()),
                debug_info.clone(),
            );
        }
    }
}
