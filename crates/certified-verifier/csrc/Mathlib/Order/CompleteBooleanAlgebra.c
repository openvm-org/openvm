// Lean compiler output
// Module: Mathlib.Order.CompleteBooleanAlgebra
// Imports: public import Init public meta import Init public import Mathlib.Logic.Equiv.Set public import Mathlib.Logic.Pairwise public import Mathlib.Order.CompleteLattice.Lemmas public import Mathlib.Order.Directed public import Mathlib.Order.GaloisConnection.Basic
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lp_mathlib_Prod_instCompleteLattice___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instHeytingAlgebra___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instCoheytingAlgebra___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
extern lean_object* lp_mathlib_PUnit_instCompleteLinearOrder;
extern lean_object* lp_mathlib_PUnit_instBooleanAlgebra;
lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_SemilatticeInf_toMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeSup_toMax___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instBooleanAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instHeytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(lean_object*);
extern lean_object* lp_mathlib_Prop_instCompleteLattice;
lean_object* lp_mathlib_Prod_instBooleanAlgebra___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toHeytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_toCoframe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_toCoframe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_ofMinimalAxioms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_ofMinimalAxioms(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter___redArg(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoframe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoframe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instFrame___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instFrame(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFrame___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFrame___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFrame___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFrame(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instFrame___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instFrame(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoframe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoframe(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoframe___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoframe___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoframe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoframe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteDistribLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteDistribLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteDistribLattice___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteDistribLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompletelyDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompletelyDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompletelyDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompletelyDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompletelyDistribLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompletelyDistribLattice___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompletelyDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompletelyDistribLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompletelyDistribLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompletelyDistribLattice___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompletelyDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompletelyDistribLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteBooleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteBooleanAlgebra___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteBooleanAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteBooleanAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteBooleanAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteBooleanAlgebra___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteAtomicBooleanAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteAtomicBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteAtomicBooleanAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_Prop_instCompleteBooleanAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_frame___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_frame___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_frame(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_frame___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coframe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coframe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coframe___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeDistribLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeDistribLattice___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completelyDistribLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completelyDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completelyDistribLattice___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeBooleanAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeBooleanAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeAtomicBooleanAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeAtomicBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeAtomicBooleanAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_frame___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_frame___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_frame___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_frame___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coframe___redArg___lam__17(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coframe___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coframe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coframe(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeDistribLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completelyDistribLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completelyDistribLattice___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completelyDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completelyDistribLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeBooleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeAtomicBooleanAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeAtomicBooleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_PUnit_instCompleteBooleanAlgebra___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PUnit_instCompleteBooleanAlgebra___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteBooleanAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteAtomicBooleanAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toCompleteLattice_2_; lean_object* v_toBoundedOrder_3_; lean_object* v_toHImp_4_; lean_object* v_toCompl_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_16_; 
v_toCompleteLattice_2_ = lean_ctor_get(v_self_1_, 0);
lean_inc_ref(v_toCompleteLattice_2_);
v_toBoundedOrder_3_ = lean_ctor_get(v_toCompleteLattice_2_, 3);
lean_inc_ref(v_toBoundedOrder_3_);
v_toHImp_4_ = lean_ctor_get(v_self_1_, 1);
v_toCompl_5_ = lean_ctor_get(v_self_1_, 2);
v_isSharedCheck_16_ = !lean_is_exclusive(v_self_1_);
if (v_isSharedCheck_16_ == 0)
{
lean_object* v_unused_17_; 
v_unused_17_ = lean_ctor_get(v_self_1_, 0);
lean_dec(v_unused_17_);
v___x_7_ = v_self_1_;
v_isShared_8_ = v_isSharedCheck_16_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_toCompl_5_);
lean_inc(v_toHImp_4_);
lean_dec(v_self_1_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_16_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
lean_object* v_toLattice_9_; lean_object* v_toOrderTop_10_; lean_object* v_toOrderBot_11_; lean_object* v___x_13_; 
v_toLattice_9_ = lean_ctor_get(v_toCompleteLattice_2_, 0);
lean_inc_ref(v_toLattice_9_);
lean_dec_ref(v_toCompleteLattice_2_);
v_toOrderTop_10_ = lean_ctor_get(v_toBoundedOrder_3_, 0);
lean_inc(v_toOrderTop_10_);
v_toOrderBot_11_ = lean_ctor_get(v_toBoundedOrder_3_, 1);
lean_inc(v_toOrderBot_11_);
lean_dec_ref(v_toBoundedOrder_3_);
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 2, v_toHImp_4_);
lean_ctor_set(v___x_7_, 1, v_toOrderTop_10_);
lean_ctor_set(v___x_7_, 0, v_toLattice_9_);
v___x_13_ = v___x_7_;
goto v_reusejp_12_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v_toLattice_9_);
lean_ctor_set(v_reuseFailAlloc_15_, 1, v_toOrderTop_10_);
lean_ctor_set(v_reuseFailAlloc_15_, 2, v_toHImp_4_);
v___x_13_ = v_reuseFailAlloc_15_;
goto v_reusejp_12_;
}
v_reusejp_12_:
{
lean_object* v___x_14_; 
v___x_14_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v_toOrderBot_11_);
lean_ctor_set(v___x_14_, 2, v_toCompl_5_);
return v___x_14_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toHeytingAlgebra(lean_object* v_00_u03b1_18_, lean_object* v_self_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v_self_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(lean_object* v_self_21_){
_start:
{
lean_object* v_toCompleteLattice_22_; lean_object* v_toBoundedOrder_23_; lean_object* v_toSDiff_24_; lean_object* v_toHNot_25_; lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_36_; 
v_toCompleteLattice_22_ = lean_ctor_get(v_self_21_, 0);
lean_inc_ref(v_toCompleteLattice_22_);
v_toBoundedOrder_23_ = lean_ctor_get(v_toCompleteLattice_22_, 3);
lean_inc_ref(v_toBoundedOrder_23_);
v_toSDiff_24_ = lean_ctor_get(v_self_21_, 1);
v_toHNot_25_ = lean_ctor_get(v_self_21_, 2);
v_isSharedCheck_36_ = !lean_is_exclusive(v_self_21_);
if (v_isSharedCheck_36_ == 0)
{
lean_object* v_unused_37_; 
v_unused_37_ = lean_ctor_get(v_self_21_, 0);
lean_dec(v_unused_37_);
v___x_27_ = v_self_21_;
v_isShared_28_ = v_isSharedCheck_36_;
goto v_resetjp_26_;
}
else
{
lean_inc(v_toHNot_25_);
lean_inc(v_toSDiff_24_);
lean_dec(v_self_21_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_36_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v_toLattice_29_; lean_object* v_toOrderTop_30_; lean_object* v_toOrderBot_31_; lean_object* v___x_33_; 
v_toLattice_29_ = lean_ctor_get(v_toCompleteLattice_22_, 0);
lean_inc_ref(v_toLattice_29_);
lean_dec_ref(v_toCompleteLattice_22_);
v_toOrderTop_30_ = lean_ctor_get(v_toBoundedOrder_23_, 0);
lean_inc(v_toOrderTop_30_);
v_toOrderBot_31_ = lean_ctor_get(v_toBoundedOrder_23_, 1);
lean_inc(v_toOrderBot_31_);
lean_dec_ref(v_toBoundedOrder_23_);
if (v_isShared_28_ == 0)
{
lean_ctor_set(v___x_27_, 2, v_toSDiff_24_);
lean_ctor_set(v___x_27_, 1, v_toOrderBot_31_);
lean_ctor_set(v___x_27_, 0, v_toLattice_29_);
v___x_33_ = v___x_27_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_35_; 
v_reuseFailAlloc_35_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_35_, 0, v_toLattice_29_);
lean_ctor_set(v_reuseFailAlloc_35_, 1, v_toOrderBot_31_);
lean_ctor_set(v_reuseFailAlloc_35_, 2, v_toSDiff_24_);
v___x_33_ = v_reuseFailAlloc_35_;
goto v_reusejp_32_;
}
v_reusejp_32_:
{
lean_object* v___x_34_; 
v___x_34_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v_toOrderTop_30_);
lean_ctor_set(v___x_34_, 2, v_toHNot_25_);
return v___x_34_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toCoheytingAlgebra(lean_object* v_00_u03b1_38_, lean_object* v_self_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v_self_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_toCoframe___redArg(lean_object* v_self_41_){
_start:
{
lean_object* v_toFrame_42_; lean_object* v_toSDiff_43_; lean_object* v_toHNot_44_; lean_object* v_toCompleteLattice_45_; lean_object* v___x_47_; uint8_t v_isShared_48_; uint8_t v_isSharedCheck_52_; 
v_toFrame_42_ = lean_ctor_get(v_self_41_, 0);
lean_inc_ref(v_toFrame_42_);
v_toSDiff_43_ = lean_ctor_get(v_self_41_, 1);
lean_inc(v_toSDiff_43_);
v_toHNot_44_ = lean_ctor_get(v_self_41_, 2);
lean_inc(v_toHNot_44_);
lean_dec_ref(v_self_41_);
v_toCompleteLattice_45_ = lean_ctor_get(v_toFrame_42_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v_toFrame_42_);
if (v_isSharedCheck_52_ == 0)
{
lean_object* v_unused_53_; lean_object* v_unused_54_; 
v_unused_53_ = lean_ctor_get(v_toFrame_42_, 2);
lean_dec(v_unused_53_);
v_unused_54_ = lean_ctor_get(v_toFrame_42_, 1);
lean_dec(v_unused_54_);
v___x_47_ = v_toFrame_42_;
v_isShared_48_ = v_isSharedCheck_52_;
goto v_resetjp_46_;
}
else
{
lean_inc(v_toCompleteLattice_45_);
lean_dec(v_toFrame_42_);
v___x_47_ = lean_box(0);
v_isShared_48_ = v_isSharedCheck_52_;
goto v_resetjp_46_;
}
v_resetjp_46_:
{
lean_object* v___x_50_; 
if (v_isShared_48_ == 0)
{
lean_ctor_set(v___x_47_, 2, v_toHNot_44_);
lean_ctor_set(v___x_47_, 1, v_toSDiff_43_);
v___x_50_ = v___x_47_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v_toCompleteLattice_45_);
lean_ctor_set(v_reuseFailAlloc_51_, 1, v_toSDiff_43_);
lean_ctor_set(v_reuseFailAlloc_51_, 2, v_toHNot_44_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
return v___x_50_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_toCoframe(lean_object* v_00_u03b1_55_, lean_object* v_self_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v_self_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra___redArg(lean_object* v_self_58_){
_start:
{
lean_object* v_toFrame_59_; lean_object* v_toCompleteLattice_60_; lean_object* v_toBoundedOrder_61_; lean_object* v_toSDiff_62_; lean_object* v_toHNot_63_; lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_84_; 
v_toFrame_59_ = lean_ctor_get(v_self_58_, 0);
lean_inc_ref(v_toFrame_59_);
v_toCompleteLattice_60_ = lean_ctor_get(v_toFrame_59_, 0);
lean_inc_ref(v_toCompleteLattice_60_);
v_toBoundedOrder_61_ = lean_ctor_get(v_toCompleteLattice_60_, 3);
lean_inc_ref(v_toBoundedOrder_61_);
v_toSDiff_62_ = lean_ctor_get(v_self_58_, 1);
v_toHNot_63_ = lean_ctor_get(v_self_58_, 2);
v_isSharedCheck_84_ = !lean_is_exclusive(v_self_58_);
if (v_isSharedCheck_84_ == 0)
{
lean_object* v_unused_85_; 
v_unused_85_ = lean_ctor_get(v_self_58_, 0);
lean_dec(v_unused_85_);
v___x_65_ = v_self_58_;
v_isShared_66_ = v_isSharedCheck_84_;
goto v_resetjp_64_;
}
else
{
lean_inc(v_toHNot_63_);
lean_inc(v_toSDiff_62_);
lean_dec(v_self_58_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_84_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
lean_object* v_toHImp_67_; lean_object* v_toCompl_68_; lean_object* v___x_70_; uint8_t v_isShared_71_; uint8_t v_isSharedCheck_82_; 
v_toHImp_67_ = lean_ctor_get(v_toFrame_59_, 1);
v_toCompl_68_ = lean_ctor_get(v_toFrame_59_, 2);
v_isSharedCheck_82_ = !lean_is_exclusive(v_toFrame_59_);
if (v_isSharedCheck_82_ == 0)
{
lean_object* v_unused_83_; 
v_unused_83_ = lean_ctor_get(v_toFrame_59_, 0);
lean_dec(v_unused_83_);
v___x_70_ = v_toFrame_59_;
v_isShared_71_ = v_isSharedCheck_82_;
goto v_resetjp_69_;
}
else
{
lean_inc(v_toCompl_68_);
lean_inc(v_toHImp_67_);
lean_dec(v_toFrame_59_);
v___x_70_ = lean_box(0);
v_isShared_71_ = v_isSharedCheck_82_;
goto v_resetjp_69_;
}
v_resetjp_69_:
{
lean_object* v_toLattice_72_; lean_object* v_toOrderTop_73_; lean_object* v_toOrderBot_74_; lean_object* v___x_76_; 
v_toLattice_72_ = lean_ctor_get(v_toCompleteLattice_60_, 0);
lean_inc_ref(v_toLattice_72_);
lean_dec_ref(v_toCompleteLattice_60_);
v_toOrderTop_73_ = lean_ctor_get(v_toBoundedOrder_61_, 0);
lean_inc(v_toOrderTop_73_);
v_toOrderBot_74_ = lean_ctor_get(v_toBoundedOrder_61_, 1);
lean_inc(v_toOrderBot_74_);
lean_dec_ref(v_toBoundedOrder_61_);
if (v_isShared_71_ == 0)
{
lean_ctor_set(v___x_70_, 2, v_toHImp_67_);
lean_ctor_set(v___x_70_, 1, v_toOrderTop_73_);
lean_ctor_set(v___x_70_, 0, v_toLattice_72_);
v___x_76_ = v___x_70_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_81_; 
v_reuseFailAlloc_81_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_81_, 0, v_toLattice_72_);
lean_ctor_set(v_reuseFailAlloc_81_, 1, v_toOrderTop_73_);
lean_ctor_set(v_reuseFailAlloc_81_, 2, v_toHImp_67_);
v___x_76_ = v_reuseFailAlloc_81_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
lean_object* v___x_78_; 
if (v_isShared_66_ == 0)
{
lean_ctor_set(v___x_65_, 2, v_toCompl_68_);
lean_ctor_set(v___x_65_, 1, v_toOrderBot_74_);
lean_ctor_set(v___x_65_, 0, v___x_76_);
v___x_78_ = v___x_65_;
goto v_reusejp_77_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v___x_76_);
lean_ctor_set(v_reuseFailAlloc_80_, 1, v_toOrderBot_74_);
lean_ctor_set(v_reuseFailAlloc_80_, 2, v_toCompl_68_);
v___x_78_ = v_reuseFailAlloc_80_;
goto v_reusejp_77_;
}
v_reusejp_77_:
{
lean_object* v___x_79_; 
v___x_79_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_toSDiff_62_);
lean_ctor_set(v___x_79_, 2, v_toHNot_63_);
return v___x_79_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra(lean_object* v_00_u03b1_86_, lean_object* v_self_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra___redArg(v_self_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra___redArg(lean_object* v_self_89_){
_start:
{
lean_object* v_toCompleteLattice_90_; lean_object* v_toBoundedOrder_91_; lean_object* v_toHImp_92_; lean_object* v_toCompl_93_; lean_object* v_toSDiff_94_; lean_object* v_toHNot_95_; lean_object* v_toLattice_96_; lean_object* v_toOrderTop_97_; lean_object* v_toOrderBot_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v_toCompleteLattice_90_ = lean_ctor_get(v_self_89_, 0);
v_toBoundedOrder_91_ = lean_ctor_get(v_toCompleteLattice_90_, 3);
v_toHImp_92_ = lean_ctor_get(v_self_89_, 1);
v_toCompl_93_ = lean_ctor_get(v_self_89_, 2);
v_toSDiff_94_ = lean_ctor_get(v_self_89_, 3);
v_toHNot_95_ = lean_ctor_get(v_self_89_, 4);
v_toLattice_96_ = lean_ctor_get(v_toCompleteLattice_90_, 0);
v_toOrderTop_97_ = lean_ctor_get(v_toBoundedOrder_91_, 0);
v_toOrderBot_98_ = lean_ctor_get(v_toBoundedOrder_91_, 1);
lean_inc(v_toHImp_92_);
lean_inc(v_toOrderTop_97_);
lean_inc_ref(v_toLattice_96_);
v___x_99_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_99_, 0, v_toLattice_96_);
lean_ctor_set(v___x_99_, 1, v_toOrderTop_97_);
lean_ctor_set(v___x_99_, 2, v_toHImp_92_);
lean_inc(v_toCompl_93_);
lean_inc(v_toOrderBot_98_);
v___x_100_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
lean_ctor_set(v___x_100_, 1, v_toOrderBot_98_);
lean_ctor_set(v___x_100_, 2, v_toCompl_93_);
lean_inc(v_toHNot_95_);
lean_inc(v_toSDiff_94_);
v___x_101_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
lean_ctor_set(v___x_101_, 1, v_toSDiff_94_);
lean_ctor_set(v___x_101_, 2, v_toHNot_95_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra___redArg___boxed(lean_object* v_self_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra___redArg(v_self_102_);
lean_dec_ref(v_self_102_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra(lean_object* v_00_u03b1_104_, lean_object* v_self_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra___redArg(v_self_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra___boxed(lean_object* v_00_u03b1_107_, lean_object* v_self_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_CompletelyDistribLattice_toBiheytingAlgebra(v_00_u03b1_107_, v_self_108_);
lean_dec_ref(v_self_108_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0(lean_object* v_toSupSet_110_, lean_object* v_a_111_, lean_object* v_b_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lean_apply_1(v_toSupSet_110_, lean_box(0));
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0___boxed(lean_object* v_toSupSet_114_, lean_object* v_a_115_, lean_object* v_b_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0(v_toSupSet_114_, v_a_115_, v_b_116_);
lean_dec(v_b_116_);
lean_dec(v_a_115_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1(lean_object* v_toSupSet_118_, lean_object* v_a_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lean_apply_1(v_toSupSet_118_, lean_box(0));
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1___boxed(lean_object* v_toSupSet_121_, lean_object* v_a_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1(v_toSupSet_121_, v_a_122_);
lean_dec(v_a_122_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms___redArg(lean_object* v_inst_124_){
_start:
{
lean_object* v___x_125_; lean_object* v_toSupSet_126_; lean_object* v___f_127_; lean_object* v___f_128_; lean_object* v___x_129_; 
lean_inc_ref(v_inst_124_);
v___x_125_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_124_);
v_toSupSet_126_ = lean_ctor_get(v___x_125_, 1);
lean_inc_n(v_toSupSet_126_, 2);
lean_dec_ref(v___x_125_);
v___f_127_ = lean_alloc_closure((void*)(lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_127_, 0, v_toSupSet_126_);
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_128_, 0, v_toSupSet_126_);
v___x_129_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_129_, 0, v_inst_124_);
lean_ctor_set(v___x_129_, 1, v___f_127_);
lean_ctor_set(v___x_129_, 2, v___f_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_ofMinimalAxioms(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_, lean_object* v_minAx_132_){
_start:
{
lean_object* v___x_133_; lean_object* v_toSupSet_134_; lean_object* v___f_135_; lean_object* v___f_136_; lean_object* v___x_137_; 
lean_inc_ref(v_inst_131_);
v___x_133_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_131_);
v_toSupSet_134_ = lean_ctor_get(v___x_133_, 1);
lean_inc_n(v_toSupSet_134_, 2);
lean_dec_ref(v___x_133_);
v___f_135_ = lean_alloc_closure((void*)(lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_135_, 0, v_toSupSet_134_);
v___f_136_ = lean_alloc_closure((void*)(lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_136_, 0, v_toSupSet_134_);
v___x_137_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_137_, 0, v_inst_131_);
lean_ctor_set(v___x_137_, 1, v___f_135_);
lean_ctor_set(v___x_137_, 2, v___f_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0(lean_object* v_toInfSet_138_, lean_object* v_a_139_, lean_object* v_b_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lean_apply_1(v_toInfSet_138_, lean_box(0));
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0___boxed(lean_object* v_toInfSet_142_, lean_object* v_a_143_, lean_object* v_b_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0(v_toInfSet_142_, v_a_143_, v_b_144_);
lean_dec(v_b_144_);
lean_dec(v_a_143_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1(lean_object* v_toInfSet_146_, lean_object* v_a_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_apply_1(v_toInfSet_146_, lean_box(0));
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1___boxed(lean_object* v_toInfSet_149_, lean_object* v_a_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1(v_toInfSet_149_, v_a_150_);
lean_dec(v_a_150_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg(lean_object* v_inst_152_){
_start:
{
lean_object* v___x_153_; lean_object* v_toInfSet_154_; lean_object* v___f_155_; lean_object* v___f_156_; lean_object* v___x_157_; 
lean_inc_ref(v_inst_152_);
v___x_153_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_152_);
v_toInfSet_154_ = lean_ctor_get(v___x_153_, 1);
lean_inc_n(v_toInfSet_154_, 2);
lean_dec_ref(v___x_153_);
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_155_, 0, v_toInfSet_154_);
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_156_, 0, v_toInfSet_154_);
v___x_157_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_157_, 0, v_inst_152_);
lean_ctor_set(v___x_157_, 1, v___f_155_);
lean_ctor_set(v___x_157_, 2, v___f_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_ofMinimalAxioms(lean_object* v_00_u03b1_158_, lean_object* v_inst_159_, lean_object* v_minAx_160_){
_start:
{
lean_object* v___x_161_; lean_object* v_toInfSet_162_; lean_object* v___f_163_; lean_object* v___f_164_; lean_object* v___x_165_; 
lean_inc_ref(v_inst_159_);
v___x_161_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_159_);
v_toInfSet_162_ = lean_ctor_get(v___x_161_, 1);
lean_inc_n(v_toInfSet_162_, 2);
lean_dec_ref(v___x_161_);
v___f_163_ = lean_alloc_closure((void*)(lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_163_, 0, v_toInfSet_162_);
v___f_164_ = lean_alloc_closure((void*)(lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_164_, 0, v_toInfSet_162_);
v___x_165_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_165_, 0, v_inst_159_);
lean_ctor_set(v___x_165_, 1, v___f_163_);
lean_ctor_set(v___x_165_, 2, v___f_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_ofMinimalAxioms___redArg(lean_object* v_inst_166_){
_start:
{
lean_object* v___x_167_; lean_object* v_toSupSet_168_; lean_object* v___f_169_; lean_object* v___f_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v_toInfSet_173_; lean_object* v___f_174_; lean_object* v___f_175_; lean_object* v___x_176_; 
lean_inc_ref_n(v_inst_166_, 2);
v___x_167_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_166_);
v_toSupSet_168_ = lean_ctor_get(v___x_167_, 1);
lean_inc_n(v_toSupSet_168_, 2);
lean_dec_ref(v___x_167_);
v___f_169_ = lean_alloc_closure((void*)(lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_169_, 0, v_toSupSet_168_);
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_170_, 0, v_toSupSet_168_);
v___x_171_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_171_, 0, v_inst_166_);
lean_ctor_set(v___x_171_, 1, v___f_170_);
lean_ctor_set(v___x_171_, 2, v___f_169_);
v___x_172_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_166_);
v_toInfSet_173_ = lean_ctor_get(v___x_172_, 1);
lean_inc_n(v_toInfSet_173_, 2);
lean_dec_ref(v___x_172_);
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_174_, 0, v_toInfSet_173_);
v___f_175_ = lean_alloc_closure((void*)(lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_175_, 0, v_toInfSet_173_);
v___x_176_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_176_, 0, v___x_171_);
lean_ctor_set(v___x_176_, 1, v___f_175_);
lean_ctor_set(v___x_176_, 2, v___f_174_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteDistribLattice_ofMinimalAxioms(lean_object* v_00_u03b1_177_, lean_object* v_inst_178_, lean_object* v_minAx_179_){
_start:
{
lean_object* v___x_180_; lean_object* v_toSupSet_181_; lean_object* v___f_182_; lean_object* v___f_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v_toInfSet_186_; lean_object* v___f_187_; lean_object* v___f_188_; lean_object* v___x_189_; 
lean_inc_ref_n(v_inst_178_, 2);
v___x_180_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_178_);
v_toSupSet_181_ = lean_ctor_get(v___x_180_, 1);
lean_inc_n(v_toSupSet_181_, 2);
lean_dec_ref(v___x_180_);
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_182_, 0, v_toSupSet_181_);
v___f_183_ = lean_alloc_closure((void*)(lp_mathlib_Order_Frame_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_183_, 0, v_toSupSet_181_);
v___x_184_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_184_, 0, v_inst_178_);
lean_ctor_set(v___x_184_, 1, v___f_183_);
lean_ctor_set(v___x_184_, 2, v___f_182_);
v___x_185_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_178_);
v_toInfSet_186_ = lean_ctor_get(v___x_185_, 1);
lean_inc_n(v_toInfSet_186_, 2);
lean_dec_ref(v___x_185_);
v___f_187_ = lean_alloc_closure((void*)(lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_187_, 0, v_toInfSet_186_);
v___f_188_ = lean_alloc_closure((void*)(lp_mathlib_Order_Coframe_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_188_, 0, v_toInfSet_186_);
v___x_189_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_189_, 0, v___x_184_);
lean_ctor_set(v___x_189_, 1, v___f_188_);
lean_ctor_set(v___x_189_, 2, v___f_187_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter___redArg(uint8_t v_i_190_, lean_object* v_j_191_, lean_object* v_h__1_192_, lean_object* v_h__2_193_){
_start:
{
if (v_i_190_ == 0)
{
lean_object* v___x_194_; 
lean_dec(v_h__1_192_);
v___x_194_ = lean_apply_1(v_h__2_193_, v_j_191_);
return v___x_194_;
}
else
{
lean_object* v___x_195_; 
lean_dec(v_h__2_193_);
v___x_195_ = lean_apply_1(v_h__1_192_, v_j_191_);
return v___x_195_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter___redArg___boxed(lean_object* v_i_196_, lean_object* v_j_197_, lean_object* v_h__1_198_, lean_object* v_h__2_199_){
_start:
{
uint8_t v_i_59__boxed_200_; lean_object* v_res_201_; 
v_i_59__boxed_200_ = lean_unbox(v_i_196_);
v_res_201_ = lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter___redArg(v_i_59__boxed_200_, v_j_197_, v_h__1_198_, v_h__2_199_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter(lean_object* v_00_u03b1_202_, lean_object* v_s_203_, lean_object* v_motive_204_, uint8_t v_i_205_, lean_object* v_j_206_, lean_object* v_h__1_207_, lean_object* v_h__2_208_){
_start:
{
if (v_i_205_ == 0)
{
lean_object* v___x_209_; 
lean_dec(v_h__1_207_);
v___x_209_ = lean_apply_1(v_h__2_208_, v_j_206_);
return v___x_209_;
}
else
{
lean_object* v___x_210_; 
lean_dec(v_h__2_208_);
v___x_210_ = lean_apply_1(v_h__1_207_, v_j_206_);
return v___x_210_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter___boxed(lean_object* v_00_u03b1_211_, lean_object* v_s_212_, lean_object* v_motive_213_, lean_object* v_i_214_, lean_object* v_j_215_, lean_object* v_h__1_216_, lean_object* v_h__2_217_){
_start:
{
uint8_t v_i_66__boxed_218_; lean_object* v_res_219_; 
v_i_66__boxed_218_ = lean_unbox(v_i_214_);
v_res_219_ = lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__3_splitter(v_00_u03b1_211_, v_s_212_, v_motive_213_, v_i_66__boxed_218_, v_j_215_, v_h__1_216_, v_h__2_217_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter___redArg(uint8_t v_i_220_, lean_object* v_h__1_221_, lean_object* v_h__2_222_){
_start:
{
if (v_i_220_ == 0)
{
lean_object* v___x_223_; lean_object* v___x_224_; 
lean_dec(v_h__1_221_);
v___x_223_ = lean_box(0);
v___x_224_ = lean_apply_1(v_h__2_222_, v___x_223_);
return v___x_224_;
}
else
{
lean_object* v___x_225_; lean_object* v___x_226_; 
lean_dec(v_h__2_222_);
v___x_225_ = lean_box(0);
v___x_226_ = lean_apply_1(v_h__1_221_, v___x_225_);
return v___x_226_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter___redArg___boxed(lean_object* v_i_227_, lean_object* v_h__1_228_, lean_object* v_h__2_229_){
_start:
{
uint8_t v_i_26__boxed_230_; lean_object* v_res_231_; 
v_i_26__boxed_230_ = lean_unbox(v_i_227_);
v_res_231_ = lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter___redArg(v_i_26__boxed_230_, v_h__1_228_, v_h__2_229_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter(lean_object* v_motive_232_, uint8_t v_i_233_, lean_object* v_h__1_234_, lean_object* v_h__2_235_){
_start:
{
if (v_i_233_ == 0)
{
lean_object* v___x_236_; lean_object* v___x_237_; 
lean_dec(v_h__1_234_);
v___x_236_ = lean_box(0);
v___x_237_ = lean_apply_1(v_h__2_235_, v___x_236_);
return v___x_237_;
}
else
{
lean_object* v___x_238_; lean_object* v___x_239_; 
lean_dec(v_h__2_235_);
v___x_238_ = lean_box(0);
v___x_239_ = lean_apply_1(v_h__1_234_, v___x_238_);
return v___x_239_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter___boxed(lean_object* v_motive_240_, lean_object* v_i_241_, lean_object* v_h__1_242_, lean_object* v_h__2_243_){
_start:
{
uint8_t v_i_37__boxed_244_; lean_object* v_res_245_; 
v_i_37__boxed_244_ = lean_unbox(v_i_241_);
v_res_245_ = lp_mathlib___private_Mathlib_Order_CompleteBooleanAlgebra_0__CompletelyDistribLattice_MinimalAxioms_toCompleteDistribLattice_match__1__1_splitter(v_motive_240_, v_i_37__boxed_244_, v_h__1_242_, v_h__2_243_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__0(lean_object* v_toSupSet_246_, lean_object* v_a_247_, lean_object* v_a_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lean_apply_1(v_toSupSet_246_, lean_box(0));
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__0___boxed(lean_object* v_toSupSet_250_, lean_object* v_a_251_, lean_object* v_a_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__0(v_toSupSet_250_, v_a_251_, v_a_252_);
lean_dec(v_a_252_);
lean_dec(v_a_251_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__1(lean_object* v_toSupSet_254_, lean_object* v_a_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lean_apply_1(v_toSupSet_254_, lean_box(0));
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__1___boxed(lean_object* v_toSupSet_257_, lean_object* v_a_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__1(v_toSupSet_257_, v_a_258_);
lean_dec(v_a_258_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__2(lean_object* v_toInfSet_260_, lean_object* v_a_261_, lean_object* v_a_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lean_apply_1(v_toInfSet_260_, lean_box(0));
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__2___boxed(lean_object* v_toInfSet_264_, lean_object* v_a_265_, lean_object* v_a_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__2(v_toInfSet_264_, v_a_265_, v_a_266_);
lean_dec(v_a_266_);
lean_dec(v_a_265_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__3(lean_object* v_toInfSet_268_, lean_object* v_a_269_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lean_apply_1(v_toInfSet_268_, lean_box(0));
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__3___boxed(lean_object* v_toInfSet_271_, lean_object* v_a_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__3(v_toInfSet_271_, v_a_272_);
lean_dec(v_a_272_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg(lean_object* v_inst_274_){
_start:
{
lean_object* v___x_275_; lean_object* v_toSupSet_276_; lean_object* v___x_277_; lean_object* v_toInfSet_278_; lean_object* v___f_279_; lean_object* v___f_280_; lean_object* v___f_281_; lean_object* v___f_282_; lean_object* v___x_283_; 
lean_inc_ref_n(v_inst_274_, 2);
v___x_275_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_274_);
v_toSupSet_276_ = lean_ctor_get(v___x_275_, 1);
lean_inc_n(v_toSupSet_276_, 2);
lean_dec_ref(v___x_275_);
v___x_277_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_274_);
v_toInfSet_278_ = lean_ctor_get(v___x_277_, 1);
lean_inc_n(v_toInfSet_278_, 2);
lean_dec_ref(v___x_277_);
v___f_279_ = lean_alloc_closure((void*)(lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_279_, 0, v_toSupSet_276_);
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_280_, 0, v_toSupSet_276_);
v___f_281_ = lean_alloc_closure((void*)(lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_281_, 0, v_toInfSet_278_);
v___f_282_ = lean_alloc_closure((void*)(lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_282_, 0, v_toInfSet_278_);
v___x_283_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_283_, 0, v_inst_274_);
lean_ctor_set(v___x_283_, 1, v___f_279_);
lean_ctor_set(v___x_283_, 2, v___f_280_);
lean_ctor_set(v___x_283_, 3, v___f_281_);
lean_ctor_set(v___x_283_, 4, v___f_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms(lean_object* v_00_u03b1_284_, lean_object* v_inst_285_, lean_object* v_minAx_286_){
_start:
{
lean_object* v___x_287_; lean_object* v_toSupSet_288_; lean_object* v___x_289_; lean_object* v_toInfSet_290_; lean_object* v___f_291_; lean_object* v___f_292_; lean_object* v___f_293_; lean_object* v___f_294_; lean_object* v___x_295_; 
lean_inc_ref_n(v_inst_285_, 2);
v___x_287_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_285_);
v_toSupSet_288_ = lean_ctor_get(v___x_287_, 1);
lean_inc_n(v_toSupSet_288_, 2);
lean_dec_ref(v___x_287_);
v___x_289_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_285_);
v_toInfSet_290_ = lean_ctor_get(v___x_289_, 1);
lean_inc_n(v_toInfSet_290_, 2);
lean_dec_ref(v___x_289_);
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_291_, 0, v_toSupSet_288_);
v___f_292_ = lean_alloc_closure((void*)(lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_292_, 0, v_toSupSet_288_);
v___f_293_ = lean_alloc_closure((void*)(lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_293_, 0, v_toInfSet_290_);
v___f_294_ = lean_alloc_closure((void*)(lp_mathlib_CompletelyDistribLattice_ofMinimalAxioms___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_294_, 0, v_toInfSet_290_);
v___x_295_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_295_, 0, v_inst_285_);
lean_ctor_set(v___x_295_, 1, v___f_291_);
lean_ctor_set(v___x_295_, 2, v___f_292_);
lean_ctor_set(v___x_295_, 3, v___f_293_);
lean_ctor_set(v___x_295_, 4, v___f_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(lean_object* v_inst_296_){
_start:
{
lean_object* v_toCompleteLattice_297_; lean_object* v_toHImp_298_; lean_object* v_toCompl_299_; lean_object* v_toSDiff_300_; lean_object* v_toHNot_301_; lean_object* v___x_302_; lean_object* v___x_303_; 
v_toCompleteLattice_297_ = lean_ctor_get(v_inst_296_, 0);
v_toHImp_298_ = lean_ctor_get(v_inst_296_, 1);
v_toCompl_299_ = lean_ctor_get(v_inst_296_, 2);
v_toSDiff_300_ = lean_ctor_get(v_inst_296_, 3);
v_toHNot_301_ = lean_ctor_get(v_inst_296_, 4);
lean_inc(v_toCompl_299_);
lean_inc(v_toHImp_298_);
lean_inc_ref(v_toCompleteLattice_297_);
v___x_302_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_302_, 0, v_toCompleteLattice_297_);
lean_ctor_set(v___x_302_, 1, v_toHImp_298_);
lean_ctor_set(v___x_302_, 2, v_toCompl_299_);
lean_inc(v_toHNot_301_);
lean_inc(v_toSDiff_300_);
v___x_303_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_303_, 0, v___x_302_);
lean_ctor_set(v___x_303_, 1, v_toSDiff_300_);
lean_ctor_set(v___x_303_, 2, v_toHNot_301_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg___boxed(lean_object* v_inst_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v_inst_304_);
lean_dec_ref(v_inst_304_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice(lean_object* v_00_u03b1_306_, lean_object* v_inst_307_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v_inst_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___boxed(lean_object* v_00_u03b1_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice(v_00_u03b1_309_, v_inst_310_);
lean_dec_ref(v_inst_310_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___redArg(lean_object* v_inst_312_){
_start:
{
lean_object* v_toCompleteLattice_313_; lean_object* v_toHImp_314_; lean_object* v_toCompl_315_; lean_object* v_toSDiff_316_; lean_object* v_toHNot_317_; lean_object* v___x_318_; 
v_toCompleteLattice_313_ = lean_ctor_get(v_inst_312_, 0);
v_toHImp_314_ = lean_ctor_get(v_inst_312_, 1);
v_toCompl_315_ = lean_ctor_get(v_inst_312_, 2);
v_toSDiff_316_ = lean_ctor_get(v_inst_312_, 3);
v_toHNot_317_ = lean_ctor_get(v_inst_312_, 4);
lean_inc(v_toHNot_317_);
lean_inc(v_toSDiff_316_);
lean_inc(v_toCompl_315_);
lean_inc(v_toHImp_314_);
lean_inc_ref(v_toCompleteLattice_313_);
v___x_318_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_318_, 0, v_toCompleteLattice_313_);
lean_ctor_set(v___x_318_, 1, v_toHImp_314_);
lean_ctor_set(v___x_318_, 2, v_toCompl_315_);
lean_ctor_set(v___x_318_, 3, v_toSDiff_316_);
lean_ctor_set(v___x_318_, 4, v_toHNot_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___redArg___boxed(lean_object* v_inst_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___redArg(v_inst_319_);
lean_dec_ref(v_inst_319_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice(lean_object* v_00_u03b1_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___redArg(v_inst_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___boxed(lean_object* v_00_u03b1_324_, lean_object* v_inst_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice(v_00_u03b1_324_, v_inst_325_);
lean_dec_ref(v_inst_325_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoframe___redArg(lean_object* v_inst_327_){
_start:
{
lean_object* v_toCompleteLattice_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v_toGeneralizedCoheytingAlgebra_332_; lean_object* v_toHNot_333_; lean_object* v_toSDiff_334_; lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_341_; 
v_toCompleteLattice_328_ = lean_ctor_get(v_inst_327_, 0);
lean_inc_ref(v_toCompleteLattice_328_);
v___x_329_ = lp_mathlib_OrderDual_instCompleteLattice___redArg(v_toCompleteLattice_328_);
v___x_330_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v_inst_327_);
v___x_331_ = lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(v___x_330_);
v_toGeneralizedCoheytingAlgebra_332_ = lean_ctor_get(v___x_331_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_332_);
v_toHNot_333_ = lean_ctor_get(v___x_331_, 2);
lean_inc(v_toHNot_333_);
lean_dec_ref(v___x_331_);
v_toSDiff_334_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_332_, 2);
v_isSharedCheck_341_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_332_);
if (v_isSharedCheck_341_ == 0)
{
lean_object* v_unused_342_; lean_object* v_unused_343_; 
v_unused_342_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_332_, 1);
lean_dec(v_unused_342_);
v_unused_343_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_332_, 0);
lean_dec(v_unused_343_);
v___x_336_ = v_toGeneralizedCoheytingAlgebra_332_;
v_isShared_337_ = v_isSharedCheck_341_;
goto v_resetjp_335_;
}
else
{
lean_inc(v_toSDiff_334_);
lean_dec(v_toGeneralizedCoheytingAlgebra_332_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_341_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v___x_339_; 
if (v_isShared_337_ == 0)
{
lean_ctor_set(v___x_336_, 2, v_toHNot_333_);
lean_ctor_set(v___x_336_, 1, v_toSDiff_334_);
lean_ctor_set(v___x_336_, 0, v___x_329_);
v___x_339_ = v___x_336_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v___x_329_);
lean_ctor_set(v_reuseFailAlloc_340_, 1, v_toSDiff_334_);
lean_ctor_set(v_reuseFailAlloc_340_, 2, v_toHNot_333_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoframe(lean_object* v_00_u03b1_344_, lean_object* v_inst_345_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_mathlib_OrderDual_instCoframe___redArg(v_inst_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toDistribLattice___redArg(lean_object* v_inst_347_){
_start:
{
lean_object* v_toCompleteLattice_348_; lean_object* v_toLattice_349_; 
v_toCompleteLattice_348_ = lean_ctor_get(v_inst_347_, 0);
v_toLattice_349_ = lean_ctor_get(v_toCompleteLattice_348_, 0);
lean_inc_ref(v_toLattice_349_);
return v_toLattice_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toDistribLattice___redArg___boxed(lean_object* v_inst_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_Order_Frame_toDistribLattice___redArg(v_inst_350_);
lean_dec_ref(v_inst_350_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toDistribLattice(lean_object* v_00_u03b1_352_, lean_object* v_inst_353_){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = lp_mathlib_Order_Frame_toDistribLattice___redArg(v_inst_353_);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Frame_toDistribLattice___boxed(lean_object* v_00_u03b1_355_, lean_object* v_inst_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_Order_Frame_toDistribLattice(v_00_u03b1_355_, v_inst_356_);
lean_dec_ref(v_inst_356_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toDistribLattice___redArg(lean_object* v_inst_358_){
_start:
{
lean_object* v_toCompleteLattice_359_; lean_object* v_toLattice_360_; 
v_toCompleteLattice_359_ = lean_ctor_get(v_inst_358_, 0);
v_toLattice_360_ = lean_ctor_get(v_toCompleteLattice_359_, 0);
lean_inc_ref(v_toLattice_360_);
return v_toLattice_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toDistribLattice___redArg___boxed(lean_object* v_inst_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_Order_Coframe_toDistribLattice___redArg(v_inst_361_);
lean_dec_ref(v_inst_361_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toDistribLattice(lean_object* v_00_u03b1_363_, lean_object* v_inst_364_){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lp_mathlib_Order_Coframe_toDistribLattice___redArg(v_inst_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_Coframe_toDistribLattice___boxed(lean_object* v_00_u03b1_366_, lean_object* v_inst_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_mathlib_Order_Coframe_toDistribLattice(v_00_u03b1_366_, v_inst_367_);
lean_dec_ref(v_inst_367_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instFrame___redArg(lean_object* v_inst_369_, lean_object* v_inst_370_){
_start:
{
lean_object* v_toCompleteLattice_371_; lean_object* v_toCompleteLattice_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v_toGeneralizedHeytingAlgebra_377_; lean_object* v_toCompl_378_; lean_object* v_toHImp_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_386_; 
v_toCompleteLattice_371_ = lean_ctor_get(v_inst_369_, 0);
v_toCompleteLattice_372_ = lean_ctor_get(v_inst_370_, 0);
lean_inc_ref(v_toCompleteLattice_372_);
lean_inc_ref(v_toCompleteLattice_371_);
v___x_373_ = lp_mathlib_Prod_instCompleteLattice___redArg(v_toCompleteLattice_371_, v_toCompleteLattice_372_);
v___x_374_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v_inst_369_);
v___x_375_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v_inst_370_);
v___x_376_ = lp_mathlib_Prod_instHeytingAlgebra___redArg(v___x_374_, v___x_375_);
v_toGeneralizedHeytingAlgebra_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_377_);
v_toCompl_378_ = lean_ctor_get(v___x_376_, 2);
lean_inc(v_toCompl_378_);
lean_dec_ref(v___x_376_);
v_toHImp_379_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_377_, 2);
v_isSharedCheck_386_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_377_);
if (v_isSharedCheck_386_ == 0)
{
lean_object* v_unused_387_; lean_object* v_unused_388_; 
v_unused_387_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_377_, 1);
lean_dec(v_unused_387_);
v_unused_388_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_377_, 0);
lean_dec(v_unused_388_);
v___x_381_ = v_toGeneralizedHeytingAlgebra_377_;
v_isShared_382_ = v_isSharedCheck_386_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_toHImp_379_);
lean_dec(v_toGeneralizedHeytingAlgebra_377_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_386_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v___x_384_; 
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 2, v_toCompl_378_);
lean_ctor_set(v___x_381_, 1, v_toHImp_379_);
lean_ctor_set(v___x_381_, 0, v___x_373_);
v___x_384_ = v___x_381_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v___x_373_);
lean_ctor_set(v_reuseFailAlloc_385_, 1, v_toHImp_379_);
lean_ctor_set(v_reuseFailAlloc_385_, 2, v_toCompl_378_);
v___x_384_ = v_reuseFailAlloc_385_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
return v___x_384_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instFrame(lean_object* v_00_u03b1_389_, lean_object* v_00_u03b2_390_, lean_object* v_inst_391_, lean_object* v_inst_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_Prod_instFrame___redArg(v_inst_391_, v_inst_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFrame___redArg___lam__0(lean_object* v_inst_394_, lean_object* v_i_395_){
_start:
{
lean_object* v___x_396_; lean_object* v_toCompleteLattice_397_; 
v___x_396_ = lean_apply_1(v_inst_394_, v_i_395_);
v_toCompleteLattice_397_ = lean_ctor_get(v___x_396_, 0);
lean_inc_ref(v_toCompleteLattice_397_);
lean_dec_ref(v___x_396_);
return v_toCompleteLattice_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFrame___redArg___lam__1(lean_object* v_inst_398_, lean_object* v_i_399_){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; 
v___x_400_ = lean_apply_1(v_inst_398_, v_i_399_);
v___x_401_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v___x_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFrame___redArg(lean_object* v_inst_402_){
_start:
{
lean_object* v___f_403_; lean_object* v___f_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v_toGeneralizedHeytingAlgebra_407_; lean_object* v_toCompl_408_; lean_object* v_toHImp_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_416_; 
lean_inc_ref(v_inst_402_);
v___f_403_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instFrame___redArg___lam__0), 2, 1);
lean_closure_set(v___f_403_, 0, v_inst_402_);
v___f_404_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instFrame___redArg___lam__1), 2, 1);
lean_closure_set(v___f_404_, 0, v_inst_402_);
v___x_405_ = lp_mathlib_Pi_instCompleteLattice___redArg(v___f_403_);
v___x_406_ = lp_mathlib_Pi_instHeytingAlgebra___redArg(v___f_404_);
v_toGeneralizedHeytingAlgebra_407_ = lean_ctor_get(v___x_406_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_407_);
v_toCompl_408_ = lean_ctor_get(v___x_406_, 2);
lean_inc(v_toCompl_408_);
lean_dec_ref(v___x_406_);
v_toHImp_409_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_407_, 2);
v_isSharedCheck_416_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_407_);
if (v_isSharedCheck_416_ == 0)
{
lean_object* v_unused_417_; lean_object* v_unused_418_; 
v_unused_417_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_407_, 1);
lean_dec(v_unused_417_);
v_unused_418_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_407_, 0);
lean_dec(v_unused_418_);
v___x_411_ = v_toGeneralizedHeytingAlgebra_407_;
v_isShared_412_ = v_isSharedCheck_416_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_toHImp_409_);
lean_dec(v_toGeneralizedHeytingAlgebra_407_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_416_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
lean_object* v___x_414_; 
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 2, v_toCompl_408_);
lean_ctor_set(v___x_411_, 1, v_toHImp_409_);
lean_ctor_set(v___x_411_, 0, v___x_405_);
v___x_414_ = v___x_411_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v___x_405_);
lean_ctor_set(v_reuseFailAlloc_415_, 1, v_toHImp_409_);
lean_ctor_set(v_reuseFailAlloc_415_, 2, v_toCompl_408_);
v___x_414_ = v_reuseFailAlloc_415_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
return v___x_414_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFrame(lean_object* v_00_u03b9_419_, lean_object* v_00_u03c0_420_, lean_object* v_inst_421_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lp_mathlib_Pi_instFrame___redArg(v_inst_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instFrame___redArg(lean_object* v_inst_423_){
_start:
{
lean_object* v_toCompleteLattice_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v_toGeneralizedHeytingAlgebra_428_; lean_object* v_toCompl_429_; lean_object* v_toHImp_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_437_; 
v_toCompleteLattice_424_ = lean_ctor_get(v_inst_423_, 0);
lean_inc_ref(v_toCompleteLattice_424_);
v___x_425_ = lp_mathlib_OrderDual_instCompleteLattice___redArg(v_toCompleteLattice_424_);
v___x_426_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v_inst_423_);
v___x_427_ = lp_mathlib_OrderDual_instHeytingAlgebra___redArg(v___x_426_);
v_toGeneralizedHeytingAlgebra_428_ = lean_ctor_get(v___x_427_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_428_);
v_toCompl_429_ = lean_ctor_get(v___x_427_, 2);
lean_inc(v_toCompl_429_);
lean_dec_ref(v___x_427_);
v_toHImp_430_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_428_, 2);
v_isSharedCheck_437_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_428_);
if (v_isSharedCheck_437_ == 0)
{
lean_object* v_unused_438_; lean_object* v_unused_439_; 
v_unused_438_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_428_, 1);
lean_dec(v_unused_438_);
v_unused_439_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_428_, 0);
lean_dec(v_unused_439_);
v___x_432_ = v_toGeneralizedHeytingAlgebra_428_;
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_toHImp_430_);
lean_dec(v_toGeneralizedHeytingAlgebra_428_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_435_; 
if (v_isShared_433_ == 0)
{
lean_ctor_set(v___x_432_, 2, v_toCompl_429_);
lean_ctor_set(v___x_432_, 1, v_toHImp_430_);
lean_ctor_set(v___x_432_, 0, v___x_425_);
v___x_435_ = v___x_432_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v___x_425_);
lean_ctor_set(v_reuseFailAlloc_436_, 1, v_toHImp_430_);
lean_ctor_set(v_reuseFailAlloc_436_, 2, v_toCompl_429_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instFrame(lean_object* v_00_u03b1_440_, lean_object* v_inst_441_){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = lp_mathlib_OrderDual_instFrame___redArg(v_inst_441_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoframe___redArg(lean_object* v_inst_443_, lean_object* v_inst_444_){
_start:
{
lean_object* v_toCompleteLattice_445_; lean_object* v_toCompleteLattice_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v_toGeneralizedCoheytingAlgebra_451_; lean_object* v_toHNot_452_; lean_object* v_toSDiff_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_460_; 
v_toCompleteLattice_445_ = lean_ctor_get(v_inst_443_, 0);
v_toCompleteLattice_446_ = lean_ctor_get(v_inst_444_, 0);
lean_inc_ref(v_toCompleteLattice_446_);
lean_inc_ref(v_toCompleteLattice_445_);
v___x_447_ = lp_mathlib_Prod_instCompleteLattice___redArg(v_toCompleteLattice_445_, v_toCompleteLattice_446_);
v___x_448_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v_inst_443_);
v___x_449_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v_inst_444_);
v___x_450_ = lp_mathlib_Prod_instCoheytingAlgebra___redArg(v___x_448_, v___x_449_);
v_toGeneralizedCoheytingAlgebra_451_ = lean_ctor_get(v___x_450_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_451_);
v_toHNot_452_ = lean_ctor_get(v___x_450_, 2);
lean_inc(v_toHNot_452_);
lean_dec_ref(v___x_450_);
v_toSDiff_453_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_451_, 2);
v_isSharedCheck_460_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_451_);
if (v_isSharedCheck_460_ == 0)
{
lean_object* v_unused_461_; lean_object* v_unused_462_; 
v_unused_461_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_451_, 1);
lean_dec(v_unused_461_);
v_unused_462_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_451_, 0);
lean_dec(v_unused_462_);
v___x_455_ = v_toGeneralizedCoheytingAlgebra_451_;
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_toSDiff_453_);
lean_dec(v_toGeneralizedCoheytingAlgebra_451_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_458_; 
if (v_isShared_456_ == 0)
{
lean_ctor_set(v___x_455_, 2, v_toHNot_452_);
lean_ctor_set(v___x_455_, 1, v_toSDiff_453_);
lean_ctor_set(v___x_455_, 0, v___x_447_);
v___x_458_ = v___x_455_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v___x_447_);
lean_ctor_set(v_reuseFailAlloc_459_, 1, v_toSDiff_453_);
lean_ctor_set(v_reuseFailAlloc_459_, 2, v_toHNot_452_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoframe(lean_object* v_00_u03b1_463_, lean_object* v_00_u03b2_464_, lean_object* v_inst_465_, lean_object* v_inst_466_){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lp_mathlib_Prod_instCoframe___redArg(v_inst_465_, v_inst_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoframe___redArg___lam__0(lean_object* v_inst_468_, lean_object* v_i_469_){
_start:
{
lean_object* v___x_470_; lean_object* v_toCompleteLattice_471_; 
v___x_470_ = lean_apply_1(v_inst_468_, v_i_469_);
v_toCompleteLattice_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc_ref(v_toCompleteLattice_471_);
lean_dec_ref(v___x_470_);
return v_toCompleteLattice_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoframe___redArg___lam__1(lean_object* v_inst_472_, lean_object* v_i_473_){
_start:
{
lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_474_ = lean_apply_1(v_inst_472_, v_i_473_);
v___x_475_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v___x_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoframe___redArg(lean_object* v_inst_476_){
_start:
{
lean_object* v___f_477_; lean_object* v___f_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v_toGeneralizedCoheytingAlgebra_481_; lean_object* v_toHNot_482_; lean_object* v_toSDiff_483_; lean_object* v___x_485_; uint8_t v_isShared_486_; uint8_t v_isSharedCheck_490_; 
lean_inc_ref(v_inst_476_);
v___f_477_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoframe___redArg___lam__0), 2, 1);
lean_closure_set(v___f_477_, 0, v_inst_476_);
v___f_478_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoframe___redArg___lam__1), 2, 1);
lean_closure_set(v___f_478_, 0, v_inst_476_);
v___x_479_ = lp_mathlib_Pi_instCompleteLattice___redArg(v___f_477_);
v___x_480_ = lp_mathlib_Pi_instCoheytingAlgebra___redArg(v___f_478_);
v_toGeneralizedCoheytingAlgebra_481_ = lean_ctor_get(v___x_480_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_481_);
v_toHNot_482_ = lean_ctor_get(v___x_480_, 2);
lean_inc(v_toHNot_482_);
lean_dec_ref(v___x_480_);
v_toSDiff_483_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_481_, 2);
v_isSharedCheck_490_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_481_);
if (v_isSharedCheck_490_ == 0)
{
lean_object* v_unused_491_; lean_object* v_unused_492_; 
v_unused_491_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_481_, 1);
lean_dec(v_unused_491_);
v_unused_492_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_481_, 0);
lean_dec(v_unused_492_);
v___x_485_ = v_toGeneralizedCoheytingAlgebra_481_;
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
else
{
lean_inc(v_toSDiff_483_);
lean_dec(v_toGeneralizedCoheytingAlgebra_481_);
v___x_485_ = lean_box(0);
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
v_resetjp_484_:
{
lean_object* v___x_488_; 
if (v_isShared_486_ == 0)
{
lean_ctor_set(v___x_485_, 2, v_toHNot_482_);
lean_ctor_set(v___x_485_, 1, v_toSDiff_483_);
lean_ctor_set(v___x_485_, 0, v___x_479_);
v___x_488_ = v___x_485_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v___x_479_);
lean_ctor_set(v_reuseFailAlloc_489_, 1, v_toSDiff_483_);
lean_ctor_set(v_reuseFailAlloc_489_, 2, v_toHNot_482_);
v___x_488_ = v_reuseFailAlloc_489_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
return v___x_488_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoframe(lean_object* v_00_u03b9_493_, lean_object* v_00_u03c0_494_, lean_object* v_inst_495_){
_start:
{
lean_object* v___x_496_; 
v___x_496_ = lp_mathlib_Pi_instCoframe___redArg(v_inst_495_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteDistribLattice___redArg(lean_object* v_inst_497_){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v_toFrame_500_; lean_object* v___x_502_; uint8_t v_isShared_503_; uint8_t v_isSharedCheck_510_; 
lean_inc_ref(v_inst_497_);
v___x_498_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v_inst_497_);
v___x_499_ = lp_mathlib_OrderDual_instFrame___redArg(v___x_498_);
v_toFrame_500_ = lean_ctor_get(v_inst_497_, 0);
v_isSharedCheck_510_ = !lean_is_exclusive(v_inst_497_);
if (v_isSharedCheck_510_ == 0)
{
lean_object* v_unused_511_; lean_object* v_unused_512_; 
v_unused_511_ = lean_ctor_get(v_inst_497_, 2);
lean_dec(v_unused_511_);
v_unused_512_ = lean_ctor_get(v_inst_497_, 1);
lean_dec(v_unused_512_);
v___x_502_ = v_inst_497_;
v_isShared_503_ = v_isSharedCheck_510_;
goto v_resetjp_501_;
}
else
{
lean_inc(v_toFrame_500_);
lean_dec(v_inst_497_);
v___x_502_ = lean_box(0);
v_isShared_503_ = v_isSharedCheck_510_;
goto v_resetjp_501_;
}
v_resetjp_501_:
{
lean_object* v___x_504_; lean_object* v_toSDiff_505_; lean_object* v_toHNot_506_; lean_object* v___x_508_; 
v___x_504_ = lp_mathlib_OrderDual_instCoframe___redArg(v_toFrame_500_);
v_toSDiff_505_ = lean_ctor_get(v___x_504_, 1);
lean_inc(v_toSDiff_505_);
v_toHNot_506_ = lean_ctor_get(v___x_504_, 2);
lean_inc(v_toHNot_506_);
lean_dec_ref(v___x_504_);
if (v_isShared_503_ == 0)
{
lean_ctor_set(v___x_502_, 2, v_toHNot_506_);
lean_ctor_set(v___x_502_, 1, v_toSDiff_505_);
lean_ctor_set(v___x_502_, 0, v___x_499_);
v___x_508_ = v___x_502_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v___x_499_);
lean_ctor_set(v_reuseFailAlloc_509_, 1, v_toSDiff_505_);
lean_ctor_set(v_reuseFailAlloc_509_, 2, v_toHNot_506_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteDistribLattice(lean_object* v_00_u03b1_513_, lean_object* v_inst_514_){
_start:
{
lean_object* v___x_515_; 
v___x_515_ = lp_mathlib_OrderDual_instCompleteDistribLattice___redArg(v_inst_514_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteDistribLattice___redArg(lean_object* v_inst_516_, lean_object* v_inst_517_){
_start:
{
lean_object* v_toFrame_518_; lean_object* v_toFrame_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v_toSDiff_524_; lean_object* v_toHNot_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_532_; 
v_toFrame_518_ = lean_ctor_get(v_inst_516_, 0);
v_toFrame_519_ = lean_ctor_get(v_inst_517_, 0);
lean_inc_ref(v_toFrame_519_);
lean_inc_ref(v_toFrame_518_);
v___x_520_ = lp_mathlib_Prod_instFrame___redArg(v_toFrame_518_, v_toFrame_519_);
v___x_521_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v_inst_516_);
v___x_522_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v_inst_517_);
v___x_523_ = lp_mathlib_Prod_instCoframe___redArg(v___x_521_, v___x_522_);
v_toSDiff_524_ = lean_ctor_get(v___x_523_, 1);
v_toHNot_525_ = lean_ctor_get(v___x_523_, 2);
v_isSharedCheck_532_ = !lean_is_exclusive(v___x_523_);
if (v_isSharedCheck_532_ == 0)
{
lean_object* v_unused_533_; 
v_unused_533_ = lean_ctor_get(v___x_523_, 0);
lean_dec(v_unused_533_);
v___x_527_ = v___x_523_;
v_isShared_528_ = v_isSharedCheck_532_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_toHNot_525_);
lean_inc(v_toSDiff_524_);
lean_dec(v___x_523_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_532_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
lean_object* v___x_530_; 
if (v_isShared_528_ == 0)
{
lean_ctor_set(v___x_527_, 0, v___x_520_);
v___x_530_ = v___x_527_;
goto v_reusejp_529_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v___x_520_);
lean_ctor_set(v_reuseFailAlloc_531_, 1, v_toSDiff_524_);
lean_ctor_set(v_reuseFailAlloc_531_, 2, v_toHNot_525_);
v___x_530_ = v_reuseFailAlloc_531_;
goto v_reusejp_529_;
}
v_reusejp_529_:
{
return v___x_530_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteDistribLattice(lean_object* v_00_u03b1_534_, lean_object* v_00_u03b2_535_, lean_object* v_inst_536_, lean_object* v_inst_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = lp_mathlib_Prod_instCompleteDistribLattice___redArg(v_inst_536_, v_inst_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteDistribLattice___redArg___lam__0(lean_object* v_inst_539_, lean_object* v_i_540_){
_start:
{
lean_object* v___x_541_; lean_object* v_toFrame_542_; 
v___x_541_ = lean_apply_1(v_inst_539_, v_i_540_);
v_toFrame_542_ = lean_ctor_get(v___x_541_, 0);
lean_inc_ref(v_toFrame_542_);
lean_dec_ref(v___x_541_);
return v_toFrame_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteDistribLattice___redArg___lam__1(lean_object* v_inst_543_, lean_object* v_i_544_){
_start:
{
lean_object* v___x_545_; lean_object* v___x_546_; 
v___x_545_ = lean_apply_1(v_inst_543_, v_i_544_);
v___x_546_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_545_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteDistribLattice___redArg(lean_object* v_inst_547_){
_start:
{
lean_object* v___f_548_; lean_object* v___f_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v_toSDiff_552_; lean_object* v_toHNot_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_560_; 
lean_inc_ref(v_inst_547_);
v___f_548_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteDistribLattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_548_, 0, v_inst_547_);
v___f_549_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteDistribLattice___redArg___lam__1), 2, 1);
lean_closure_set(v___f_549_, 0, v_inst_547_);
v___x_550_ = lp_mathlib_Pi_instFrame___redArg(v___f_548_);
v___x_551_ = lp_mathlib_Pi_instCoframe___redArg(v___f_549_);
v_toSDiff_552_ = lean_ctor_get(v___x_551_, 1);
v_toHNot_553_ = lean_ctor_get(v___x_551_, 2);
v_isSharedCheck_560_ = !lean_is_exclusive(v___x_551_);
if (v_isSharedCheck_560_ == 0)
{
lean_object* v_unused_561_; 
v_unused_561_ = lean_ctor_get(v___x_551_, 0);
lean_dec(v_unused_561_);
v___x_555_ = v___x_551_;
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_toHNot_553_);
lean_inc(v_toSDiff_552_);
lean_dec(v___x_551_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_558_; 
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 0, v___x_550_);
v___x_558_ = v___x_555_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v___x_550_);
lean_ctor_set(v_reuseFailAlloc_559_, 1, v_toSDiff_552_);
lean_ctor_set(v_reuseFailAlloc_559_, 2, v_toHNot_553_);
v___x_558_ = v_reuseFailAlloc_559_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
return v___x_558_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteDistribLattice(lean_object* v_00_u03b9_562_, lean_object* v_00_u03c0_563_, lean_object* v_inst_564_){
_start:
{
lean_object* v___x_565_; 
v___x_565_ = lp_mathlib_Pi_instCompleteDistribLattice___redArg(v_inst_564_);
return v___x_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompletelyDistribLattice___redArg(lean_object* v_inst_566_){
_start:
{
lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v_toFrame_570_; lean_object* v___x_571_; lean_object* v_toCompleteLattice_572_; lean_object* v_toHImp_573_; lean_object* v_toCompl_574_; lean_object* v_toSDiff_575_; lean_object* v_toHNot_576_; lean_object* v___x_577_; 
v___x_567_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v_inst_566_);
lean_inc_ref(v___x_567_);
v___x_568_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_567_);
v___x_569_ = lp_mathlib_OrderDual_instFrame___redArg(v___x_568_);
v_toFrame_570_ = lean_ctor_get(v___x_567_, 0);
lean_inc_ref(v_toFrame_570_);
lean_dec_ref(v___x_567_);
v___x_571_ = lp_mathlib_OrderDual_instCoframe___redArg(v_toFrame_570_);
v_toCompleteLattice_572_ = lean_ctor_get(v___x_569_, 0);
lean_inc_ref(v_toCompleteLattice_572_);
v_toHImp_573_ = lean_ctor_get(v___x_569_, 1);
lean_inc(v_toHImp_573_);
v_toCompl_574_ = lean_ctor_get(v___x_569_, 2);
lean_inc(v_toCompl_574_);
lean_dec_ref(v___x_569_);
v_toSDiff_575_ = lean_ctor_get(v___x_571_, 1);
lean_inc(v_toSDiff_575_);
v_toHNot_576_ = lean_ctor_get(v___x_571_, 2);
lean_inc(v_toHNot_576_);
lean_dec_ref(v___x_571_);
v___x_577_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_577_, 0, v_toCompleteLattice_572_);
lean_ctor_set(v___x_577_, 1, v_toHImp_573_);
lean_ctor_set(v___x_577_, 2, v_toCompl_574_);
lean_ctor_set(v___x_577_, 3, v_toSDiff_575_);
lean_ctor_set(v___x_577_, 4, v_toHNot_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompletelyDistribLattice___redArg___boxed(lean_object* v_inst_578_){
_start:
{
lean_object* v_res_579_; 
v_res_579_ = lp_mathlib_OrderDual_instCompletelyDistribLattice___redArg(v_inst_578_);
lean_dec_ref(v_inst_578_);
return v_res_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompletelyDistribLattice(lean_object* v_00_u03b1_580_, lean_object* v_inst_581_){
_start:
{
lean_object* v___x_582_; 
v___x_582_ = lp_mathlib_OrderDual_instCompletelyDistribLattice___redArg(v_inst_581_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompletelyDistribLattice___boxed(lean_object* v_00_u03b1_583_, lean_object* v_inst_584_){
_start:
{
lean_object* v_res_585_; 
v_res_585_ = lp_mathlib_OrderDual_instCompletelyDistribLattice(v_00_u03b1_583_, v_inst_584_);
lean_dec_ref(v_inst_584_);
return v_res_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompletelyDistribLattice___redArg(lean_object* v_inst_586_, lean_object* v_inst_587_){
_start:
{
lean_object* v___x_588_; lean_object* v_toFrame_589_; lean_object* v___x_590_; lean_object* v_toFrame_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v_toCompleteLattice_596_; lean_object* v_toHImp_597_; lean_object* v_toCompl_598_; lean_object* v_toSDiff_599_; lean_object* v_toHNot_600_; lean_object* v___x_601_; 
v___x_588_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v_inst_586_);
v_toFrame_589_ = lean_ctor_get(v___x_588_, 0);
lean_inc_ref(v_toFrame_589_);
v___x_590_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v_inst_587_);
v_toFrame_591_ = lean_ctor_get(v___x_590_, 0);
lean_inc_ref(v_toFrame_591_);
v___x_592_ = lp_mathlib_Prod_instFrame___redArg(v_toFrame_589_, v_toFrame_591_);
v___x_593_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_588_);
v___x_594_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_590_);
v___x_595_ = lp_mathlib_Prod_instCoframe___redArg(v___x_593_, v___x_594_);
v_toCompleteLattice_596_ = lean_ctor_get(v___x_592_, 0);
lean_inc_ref(v_toCompleteLattice_596_);
v_toHImp_597_ = lean_ctor_get(v___x_592_, 1);
lean_inc(v_toHImp_597_);
v_toCompl_598_ = lean_ctor_get(v___x_592_, 2);
lean_inc(v_toCompl_598_);
lean_dec_ref(v___x_592_);
v_toSDiff_599_ = lean_ctor_get(v___x_595_, 1);
lean_inc(v_toSDiff_599_);
v_toHNot_600_ = lean_ctor_get(v___x_595_, 2);
lean_inc(v_toHNot_600_);
lean_dec_ref(v___x_595_);
v___x_601_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_601_, 0, v_toCompleteLattice_596_);
lean_ctor_set(v___x_601_, 1, v_toHImp_597_);
lean_ctor_set(v___x_601_, 2, v_toCompl_598_);
lean_ctor_set(v___x_601_, 3, v_toSDiff_599_);
lean_ctor_set(v___x_601_, 4, v_toHNot_600_);
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompletelyDistribLattice___redArg___boxed(lean_object* v_inst_602_, lean_object* v_inst_603_){
_start:
{
lean_object* v_res_604_; 
v_res_604_ = lp_mathlib_Prod_instCompletelyDistribLattice___redArg(v_inst_602_, v_inst_603_);
lean_dec_ref(v_inst_603_);
lean_dec_ref(v_inst_602_);
return v_res_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompletelyDistribLattice(lean_object* v_00_u03b1_605_, lean_object* v_00_u03b2_606_, lean_object* v_inst_607_, lean_object* v_inst_608_){
_start:
{
lean_object* v___x_609_; 
v___x_609_ = lp_mathlib_Prod_instCompletelyDistribLattice___redArg(v_inst_607_, v_inst_608_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompletelyDistribLattice___boxed(lean_object* v_00_u03b1_610_, lean_object* v_00_u03b2_611_, lean_object* v_inst_612_, lean_object* v_inst_613_){
_start:
{
lean_object* v_res_614_; 
v_res_614_ = lp_mathlib_Prod_instCompletelyDistribLattice(v_00_u03b1_610_, v_00_u03b2_611_, v_inst_612_, v_inst_613_);
lean_dec_ref(v_inst_613_);
lean_dec_ref(v_inst_612_);
return v_res_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompletelyDistribLattice___redArg___lam__0(lean_object* v_inst_615_, lean_object* v_i_616_){
_start:
{
lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v_toFrame_619_; 
v___x_617_ = lean_apply_1(v_inst_615_, v_i_616_);
v___x_618_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v___x_617_);
lean_dec_ref(v___x_617_);
v_toFrame_619_ = lean_ctor_get(v___x_618_, 0);
lean_inc_ref(v_toFrame_619_);
lean_dec_ref(v___x_618_);
return v_toFrame_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompletelyDistribLattice___redArg___lam__1(lean_object* v_inst_620_, lean_object* v_i_621_){
_start:
{
lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; 
v___x_622_ = lean_apply_1(v_inst_620_, v_i_621_);
v___x_623_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v___x_622_);
lean_dec_ref(v___x_622_);
v___x_624_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_623_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompletelyDistribLattice___redArg(lean_object* v_inst_625_){
_start:
{
lean_object* v___f_626_; lean_object* v___x_627_; lean_object* v_toCompleteLattice_628_; lean_object* v_toHImp_629_; lean_object* v_toCompl_630_; lean_object* v___f_631_; lean_object* v___x_632_; lean_object* v_toSDiff_633_; lean_object* v_toHNot_634_; lean_object* v___x_635_; 
lean_inc_ref(v_inst_625_);
v___f_626_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompletelyDistribLattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_626_, 0, v_inst_625_);
v___x_627_ = lp_mathlib_Pi_instFrame___redArg(v___f_626_);
v_toCompleteLattice_628_ = lean_ctor_get(v___x_627_, 0);
lean_inc_ref(v_toCompleteLattice_628_);
v_toHImp_629_ = lean_ctor_get(v___x_627_, 1);
lean_inc(v_toHImp_629_);
v_toCompl_630_ = lean_ctor_get(v___x_627_, 2);
lean_inc(v_toCompl_630_);
lean_dec_ref(v___x_627_);
v___f_631_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompletelyDistribLattice___redArg___lam__1), 2, 1);
lean_closure_set(v___f_631_, 0, v_inst_625_);
v___x_632_ = lp_mathlib_Pi_instCoframe___redArg(v___f_631_);
v_toSDiff_633_ = lean_ctor_get(v___x_632_, 1);
lean_inc(v_toSDiff_633_);
v_toHNot_634_ = lean_ctor_get(v___x_632_, 2);
lean_inc(v_toHNot_634_);
lean_dec_ref(v___x_632_);
v___x_635_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_635_, 0, v_toCompleteLattice_628_);
lean_ctor_set(v___x_635_, 1, v_toHImp_629_);
lean_ctor_set(v___x_635_, 2, v_toCompl_630_);
lean_ctor_set(v___x_635_, 3, v_toSDiff_633_);
lean_ctor_set(v___x_635_, 4, v_toHNot_634_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompletelyDistribLattice(lean_object* v_00_u03b9_636_, lean_object* v_00_u03c0_637_, lean_object* v_inst_638_){
_start:
{
lean_object* v___x_639_; 
v___x_639_ = lp_mathlib_Pi_instCompletelyDistribLattice___redArg(v_inst_638_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(lean_object* v_self_640_){
_start:
{
lean_object* v_toCompleteLattice_641_; lean_object* v_toBoundedOrder_642_; lean_object* v_toCompl_643_; lean_object* v_toSDiff_644_; lean_object* v_toHImp_645_; lean_object* v_toLattice_646_; lean_object* v_toOrderTop_647_; lean_object* v_toOrderBot_648_; lean_object* v___x_649_; 
v_toCompleteLattice_641_ = lean_ctor_get(v_self_640_, 0);
v_toBoundedOrder_642_ = lean_ctor_get(v_toCompleteLattice_641_, 3);
v_toCompl_643_ = lean_ctor_get(v_self_640_, 1);
v_toSDiff_644_ = lean_ctor_get(v_self_640_, 2);
v_toHImp_645_ = lean_ctor_get(v_self_640_, 3);
v_toLattice_646_ = lean_ctor_get(v_toCompleteLattice_641_, 0);
v_toOrderTop_647_ = lean_ctor_get(v_toBoundedOrder_642_, 0);
v_toOrderBot_648_ = lean_ctor_get(v_toBoundedOrder_642_, 1);
lean_inc(v_toOrderBot_648_);
lean_inc(v_toOrderTop_647_);
lean_inc(v_toHImp_645_);
lean_inc(v_toSDiff_644_);
lean_inc(v_toCompl_643_);
lean_inc_ref(v_toLattice_646_);
v___x_649_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_649_, 0, v_toLattice_646_);
lean_ctor_set(v___x_649_, 1, v_toCompl_643_);
lean_ctor_set(v___x_649_, 2, v_toSDiff_644_);
lean_ctor_set(v___x_649_, 3, v_toHImp_645_);
lean_ctor_set(v___x_649_, 4, v_toOrderTop_647_);
lean_ctor_set(v___x_649_, 5, v_toOrderBot_648_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg___boxed(lean_object* v_self_650_){
_start:
{
lean_object* v_res_651_; 
v_res_651_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_self_650_);
lean_dec_ref(v_self_650_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra(lean_object* v_00_u03b1_652_, lean_object* v_self_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_self_653_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___boxed(lean_object* v_00_u03b1_655_, lean_object* v_self_656_){
_start:
{
lean_object* v_res_657_; 
v_res_657_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra(v_00_u03b1_655_, v_self_656_);
lean_dec_ref(v_self_656_);
return v_res_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(lean_object* v_inst_658_){
_start:
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v_toCompleteLattice_661_; lean_object* v_toCompl_662_; lean_object* v_toSDiff_663_; lean_object* v_toHImp_664_; lean_object* v___x_665_; lean_object* v_toHNot_666_; lean_object* v___x_668_; uint8_t v_isShared_669_; uint8_t v_isSharedCheck_673_; 
v___x_659_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_658_);
v___x_660_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(v___x_659_);
lean_dec_ref(v___x_659_);
v_toCompleteLattice_661_ = lean_ctor_get(v_inst_658_, 0);
v_toCompl_662_ = lean_ctor_get(v_inst_658_, 1);
v_toSDiff_663_ = lean_ctor_get(v_inst_658_, 2);
v_toHImp_664_ = lean_ctor_get(v_inst_658_, 3);
lean_inc(v_toCompl_662_);
lean_inc(v_toHImp_664_);
lean_inc_ref(v_toCompleteLattice_661_);
v___x_665_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_665_, 0, v_toCompleteLattice_661_);
lean_ctor_set(v___x_665_, 1, v_toHImp_664_);
lean_ctor_set(v___x_665_, 2, v_toCompl_662_);
v_toHNot_666_ = lean_ctor_get(v___x_660_, 2);
v_isSharedCheck_673_ = !lean_is_exclusive(v___x_660_);
if (v_isSharedCheck_673_ == 0)
{
lean_object* v_unused_674_; lean_object* v_unused_675_; 
v_unused_674_ = lean_ctor_get(v___x_660_, 1);
lean_dec(v_unused_674_);
v_unused_675_ = lean_ctor_get(v___x_660_, 0);
lean_dec(v_unused_675_);
v___x_668_ = v___x_660_;
v_isShared_669_ = v_isSharedCheck_673_;
goto v_resetjp_667_;
}
else
{
lean_inc(v_toHNot_666_);
lean_dec(v___x_660_);
v___x_668_ = lean_box(0);
v_isShared_669_ = v_isSharedCheck_673_;
goto v_resetjp_667_;
}
v_resetjp_667_:
{
lean_object* v___x_671_; 
lean_inc(v_toSDiff_663_);
if (v_isShared_669_ == 0)
{
lean_ctor_set(v___x_668_, 1, v_toSDiff_663_);
lean_ctor_set(v___x_668_, 0, v___x_665_);
v___x_671_ = v___x_668_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v___x_665_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v_toSDiff_663_);
lean_ctor_set(v_reuseFailAlloc_672_, 2, v_toHNot_666_);
v___x_671_ = v_reuseFailAlloc_672_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
return v___x_671_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg___boxed(lean_object* v_inst_676_){
_start:
{
lean_object* v_res_677_; 
v_res_677_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(v_inst_676_);
lean_dec_ref(v_inst_676_);
return v_res_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice(lean_object* v_00_u03b1_678_, lean_object* v_inst_679_){
_start:
{
lean_object* v___x_680_; 
v___x_680_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(v_inst_679_);
return v___x_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___boxed(lean_object* v_00_u03b1_681_, lean_object* v_inst_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice(v_00_u03b1_681_, v_inst_682_);
lean_dec_ref(v_inst_682_);
return v_res_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteBooleanAlgebra___redArg(lean_object* v_inst_684_, lean_object* v_inst_685_){
_start:
{
lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v_toFrame_692_; lean_object* v_toCompleteLattice_693_; lean_object* v_toDistribLattice_694_; lean_object* v_toCompl_695_; lean_object* v_toSDiff_696_; lean_object* v_toHImp_697_; lean_object* v_toTop_698_; lean_object* v_toBot_699_; lean_object* v_toSupSet_700_; lean_object* v_toInfSet_701_; lean_object* v___x_703_; uint8_t v_isShared_704_; uint8_t v_isSharedCheck_710_; 
v___x_686_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_684_);
v___x_687_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_685_);
v___x_688_ = lp_mathlib_Prod_instBooleanAlgebra___redArg(v___x_686_, v___x_687_);
v___x_689_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(v_inst_684_);
v___x_690_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(v_inst_685_);
v___x_691_ = lp_mathlib_Prod_instCompleteDistribLattice___redArg(v___x_689_, v___x_690_);
v_toFrame_692_ = lean_ctor_get(v___x_691_, 0);
lean_inc_ref(v_toFrame_692_);
lean_dec_ref(v___x_691_);
v_toCompleteLattice_693_ = lean_ctor_get(v_toFrame_692_, 0);
lean_inc_ref(v_toCompleteLattice_693_);
lean_dec_ref(v_toFrame_692_);
v_toDistribLattice_694_ = lean_ctor_get(v___x_688_, 0);
lean_inc_ref(v_toDistribLattice_694_);
v_toCompl_695_ = lean_ctor_get(v___x_688_, 1);
lean_inc(v_toCompl_695_);
v_toSDiff_696_ = lean_ctor_get(v___x_688_, 2);
lean_inc(v_toSDiff_696_);
v_toHImp_697_ = lean_ctor_get(v___x_688_, 3);
lean_inc(v_toHImp_697_);
v_toTop_698_ = lean_ctor_get(v___x_688_, 4);
lean_inc(v_toTop_698_);
v_toBot_699_ = lean_ctor_get(v___x_688_, 5);
lean_inc(v_toBot_699_);
lean_dec_ref(v___x_688_);
v_toSupSet_700_ = lean_ctor_get(v_toCompleteLattice_693_, 1);
v_toInfSet_701_ = lean_ctor_get(v_toCompleteLattice_693_, 2);
v_isSharedCheck_710_ = !lean_is_exclusive(v_toCompleteLattice_693_);
if (v_isSharedCheck_710_ == 0)
{
lean_object* v_unused_711_; lean_object* v_unused_712_; 
v_unused_711_ = lean_ctor_get(v_toCompleteLattice_693_, 3);
lean_dec(v_unused_711_);
v_unused_712_ = lean_ctor_get(v_toCompleteLattice_693_, 0);
lean_dec(v_unused_712_);
v___x_703_ = v_toCompleteLattice_693_;
v_isShared_704_ = v_isSharedCheck_710_;
goto v_resetjp_702_;
}
else
{
lean_inc(v_toInfSet_701_);
lean_inc(v_toSupSet_700_);
lean_dec(v_toCompleteLattice_693_);
v___x_703_ = lean_box(0);
v_isShared_704_ = v_isSharedCheck_710_;
goto v_resetjp_702_;
}
v_resetjp_702_:
{
lean_object* v___x_705_; lean_object* v___x_707_; 
v___x_705_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_705_, 0, v_toTop_698_);
lean_ctor_set(v___x_705_, 1, v_toBot_699_);
if (v_isShared_704_ == 0)
{
lean_ctor_set(v___x_703_, 3, v___x_705_);
lean_ctor_set(v___x_703_, 0, v_toDistribLattice_694_);
v___x_707_ = v___x_703_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v_toDistribLattice_694_);
lean_ctor_set(v_reuseFailAlloc_709_, 1, v_toSupSet_700_);
lean_ctor_set(v_reuseFailAlloc_709_, 2, v_toInfSet_701_);
lean_ctor_set(v_reuseFailAlloc_709_, 3, v___x_705_);
v___x_707_ = v_reuseFailAlloc_709_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
lean_object* v___x_708_; 
v___x_708_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_708_, 0, v___x_707_);
lean_ctor_set(v___x_708_, 1, v_toCompl_695_);
lean_ctor_set(v___x_708_, 2, v_toSDiff_696_);
lean_ctor_set(v___x_708_, 3, v_toHImp_697_);
return v___x_708_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteBooleanAlgebra___redArg___boxed(lean_object* v_inst_713_, lean_object* v_inst_714_){
_start:
{
lean_object* v_res_715_; 
v_res_715_ = lp_mathlib_Prod_instCompleteBooleanAlgebra___redArg(v_inst_713_, v_inst_714_);
lean_dec_ref(v_inst_714_);
lean_dec_ref(v_inst_713_);
return v_res_715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteBooleanAlgebra(lean_object* v_00_u03b1_716_, lean_object* v_00_u03b2_717_, lean_object* v_inst_718_, lean_object* v_inst_719_){
_start:
{
lean_object* v___x_720_; 
v___x_720_ = lp_mathlib_Prod_instCompleteBooleanAlgebra___redArg(v_inst_718_, v_inst_719_);
return v___x_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteBooleanAlgebra___boxed(lean_object* v_00_u03b1_721_, lean_object* v_00_u03b2_722_, lean_object* v_inst_723_, lean_object* v_inst_724_){
_start:
{
lean_object* v_res_725_; 
v_res_725_ = lp_mathlib_Prod_instCompleteBooleanAlgebra(v_00_u03b1_721_, v_00_u03b2_722_, v_inst_723_, v_inst_724_);
lean_dec_ref(v_inst_724_);
lean_dec_ref(v_inst_723_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg___lam__0(lean_object* v_inst_726_, lean_object* v_i_727_){
_start:
{
lean_object* v___x_728_; lean_object* v___x_729_; 
v___x_728_ = lean_apply_1(v_inst_726_, v_i_727_);
v___x_729_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v___x_728_);
lean_dec_ref(v___x_728_);
return v___x_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg___lam__1(lean_object* v_inst_730_, lean_object* v_i_731_){
_start:
{
lean_object* v___x_732_; lean_object* v___x_733_; 
v___x_732_ = lean_apply_1(v_inst_730_, v_i_731_);
v___x_733_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(v___x_732_);
lean_dec_ref(v___x_732_);
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg(lean_object* v_inst_734_){
_start:
{
lean_object* v___f_735_; lean_object* v___x_736_; lean_object* v_toDistribLattice_737_; lean_object* v_toCompl_738_; lean_object* v_toSDiff_739_; lean_object* v_toHImp_740_; lean_object* v_toTop_741_; lean_object* v_toBot_742_; lean_object* v___f_743_; lean_object* v___x_744_; lean_object* v_toFrame_745_; lean_object* v_toCompleteLattice_746_; lean_object* v_toSupSet_747_; lean_object* v_toInfSet_748_; lean_object* v___x_750_; uint8_t v_isShared_751_; uint8_t v_isSharedCheck_757_; 
lean_inc_ref(v_inst_734_);
v___f_735_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_735_, 0, v_inst_734_);
v___x_736_ = lp_mathlib_Pi_instBooleanAlgebra___redArg(v___f_735_);
v_toDistribLattice_737_ = lean_ctor_get(v___x_736_, 0);
lean_inc_ref(v_toDistribLattice_737_);
v_toCompl_738_ = lean_ctor_get(v___x_736_, 1);
lean_inc(v_toCompl_738_);
v_toSDiff_739_ = lean_ctor_get(v___x_736_, 2);
lean_inc(v_toSDiff_739_);
v_toHImp_740_ = lean_ctor_get(v___x_736_, 3);
lean_inc(v_toHImp_740_);
v_toTop_741_ = lean_ctor_get(v___x_736_, 4);
lean_inc(v_toTop_741_);
v_toBot_742_ = lean_ctor_get(v___x_736_, 5);
lean_inc(v_toBot_742_);
lean_dec_ref(v___x_736_);
v___f_743_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(v___f_743_, 0, v_inst_734_);
v___x_744_ = lp_mathlib_Pi_instCompleteDistribLattice___redArg(v___f_743_);
v_toFrame_745_ = lean_ctor_get(v___x_744_, 0);
lean_inc_ref(v_toFrame_745_);
lean_dec_ref(v___x_744_);
v_toCompleteLattice_746_ = lean_ctor_get(v_toFrame_745_, 0);
lean_inc_ref(v_toCompleteLattice_746_);
lean_dec_ref(v_toFrame_745_);
v_toSupSet_747_ = lean_ctor_get(v_toCompleteLattice_746_, 1);
v_toInfSet_748_ = lean_ctor_get(v_toCompleteLattice_746_, 2);
v_isSharedCheck_757_ = !lean_is_exclusive(v_toCompleteLattice_746_);
if (v_isSharedCheck_757_ == 0)
{
lean_object* v_unused_758_; lean_object* v_unused_759_; 
v_unused_758_ = lean_ctor_get(v_toCompleteLattice_746_, 3);
lean_dec(v_unused_758_);
v_unused_759_ = lean_ctor_get(v_toCompleteLattice_746_, 0);
lean_dec(v_unused_759_);
v___x_750_ = v_toCompleteLattice_746_;
v_isShared_751_ = v_isSharedCheck_757_;
goto v_resetjp_749_;
}
else
{
lean_inc(v_toInfSet_748_);
lean_inc(v_toSupSet_747_);
lean_dec(v_toCompleteLattice_746_);
v___x_750_ = lean_box(0);
v_isShared_751_ = v_isSharedCheck_757_;
goto v_resetjp_749_;
}
v_resetjp_749_:
{
lean_object* v___x_752_; lean_object* v___x_754_; 
v___x_752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_752_, 0, v_toTop_741_);
lean_ctor_set(v___x_752_, 1, v_toBot_742_);
if (v_isShared_751_ == 0)
{
lean_ctor_set(v___x_750_, 3, v___x_752_);
lean_ctor_set(v___x_750_, 0, v_toDistribLattice_737_);
v___x_754_ = v___x_750_;
goto v_reusejp_753_;
}
else
{
lean_object* v_reuseFailAlloc_756_; 
v_reuseFailAlloc_756_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_756_, 0, v_toDistribLattice_737_);
lean_ctor_set(v_reuseFailAlloc_756_, 1, v_toSupSet_747_);
lean_ctor_set(v_reuseFailAlloc_756_, 2, v_toInfSet_748_);
lean_ctor_set(v_reuseFailAlloc_756_, 3, v___x_752_);
v___x_754_ = v_reuseFailAlloc_756_;
goto v_reusejp_753_;
}
v_reusejp_753_:
{
lean_object* v___x_755_; 
v___x_755_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_755_, 0, v___x_754_);
lean_ctor_set(v___x_755_, 1, v_toCompl_738_);
lean_ctor_set(v___x_755_, 2, v_toSDiff_739_);
lean_ctor_set(v___x_755_, 3, v_toHImp_740_);
return v___x_755_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteBooleanAlgebra(lean_object* v_00_u03b9_760_, lean_object* v_00_u03c0_761_, lean_object* v_inst_762_){
_start:
{
lean_object* v___x_763_; 
v___x_763_ = lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg(v_inst_762_);
return v___x_763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteBooleanAlgebra___redArg(lean_object* v_inst_764_){
_start:
{
lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v_toFrame_769_; lean_object* v_toCompleteLattice_770_; lean_object* v_toDistribLattice_771_; lean_object* v_toCompl_772_; lean_object* v_toSDiff_773_; lean_object* v_toHImp_774_; lean_object* v_toTop_775_; lean_object* v_toBot_776_; lean_object* v_toSupSet_777_; lean_object* v_toInfSet_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_787_; 
v___x_765_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_764_);
v___x_766_ = lp_mathlib_OrderDual_instBooleanAlgebra___redArg(v___x_765_);
v___x_767_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(v_inst_764_);
v___x_768_ = lp_mathlib_OrderDual_instCompleteDistribLattice___redArg(v___x_767_);
v_toFrame_769_ = lean_ctor_get(v___x_768_, 0);
lean_inc_ref(v_toFrame_769_);
lean_dec_ref(v___x_768_);
v_toCompleteLattice_770_ = lean_ctor_get(v_toFrame_769_, 0);
lean_inc_ref(v_toCompleteLattice_770_);
lean_dec_ref(v_toFrame_769_);
v_toDistribLattice_771_ = lean_ctor_get(v___x_766_, 0);
lean_inc_ref(v_toDistribLattice_771_);
v_toCompl_772_ = lean_ctor_get(v___x_766_, 1);
lean_inc(v_toCompl_772_);
v_toSDiff_773_ = lean_ctor_get(v___x_766_, 2);
lean_inc(v_toSDiff_773_);
v_toHImp_774_ = lean_ctor_get(v___x_766_, 3);
lean_inc(v_toHImp_774_);
v_toTop_775_ = lean_ctor_get(v___x_766_, 4);
lean_inc(v_toTop_775_);
v_toBot_776_ = lean_ctor_get(v___x_766_, 5);
lean_inc(v_toBot_776_);
lean_dec_ref(v___x_766_);
v_toSupSet_777_ = lean_ctor_get(v_toCompleteLattice_770_, 1);
v_toInfSet_778_ = lean_ctor_get(v_toCompleteLattice_770_, 2);
v_isSharedCheck_787_ = !lean_is_exclusive(v_toCompleteLattice_770_);
if (v_isSharedCheck_787_ == 0)
{
lean_object* v_unused_788_; lean_object* v_unused_789_; 
v_unused_788_ = lean_ctor_get(v_toCompleteLattice_770_, 3);
lean_dec(v_unused_788_);
v_unused_789_ = lean_ctor_get(v_toCompleteLattice_770_, 0);
lean_dec(v_unused_789_);
v___x_780_ = v_toCompleteLattice_770_;
v_isShared_781_ = v_isSharedCheck_787_;
goto v_resetjp_779_;
}
else
{
lean_inc(v_toInfSet_778_);
lean_inc(v_toSupSet_777_);
lean_dec(v_toCompleteLattice_770_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_787_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v___x_782_; lean_object* v___x_784_; 
v___x_782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_782_, 0, v_toTop_775_);
lean_ctor_set(v___x_782_, 1, v_toBot_776_);
if (v_isShared_781_ == 0)
{
lean_ctor_set(v___x_780_, 3, v___x_782_);
lean_ctor_set(v___x_780_, 0, v_toDistribLattice_771_);
v___x_784_ = v___x_780_;
goto v_reusejp_783_;
}
else
{
lean_object* v_reuseFailAlloc_786_; 
v_reuseFailAlloc_786_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_786_, 0, v_toDistribLattice_771_);
lean_ctor_set(v_reuseFailAlloc_786_, 1, v_toSupSet_777_);
lean_ctor_set(v_reuseFailAlloc_786_, 2, v_toInfSet_778_);
lean_ctor_set(v_reuseFailAlloc_786_, 3, v___x_782_);
v___x_784_ = v_reuseFailAlloc_786_;
goto v_reusejp_783_;
}
v_reusejp_783_:
{
lean_object* v___x_785_; 
v___x_785_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_785_, 0, v___x_784_);
lean_ctor_set(v___x_785_, 1, v_toCompl_772_);
lean_ctor_set(v___x_785_, 2, v_toSDiff_773_);
lean_ctor_set(v___x_785_, 3, v_toHImp_774_);
return v___x_785_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteBooleanAlgebra___redArg___boxed(lean_object* v_inst_790_){
_start:
{
lean_object* v_res_791_; 
v_res_791_ = lp_mathlib_OrderDual_instCompleteBooleanAlgebra___redArg(v_inst_790_);
lean_dec_ref(v_inst_790_);
return v_res_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteBooleanAlgebra(lean_object* v_00_u03b1_792_, lean_object* v_inst_793_){
_start:
{
lean_object* v___x_794_; 
v___x_794_ = lp_mathlib_OrderDual_instCompleteBooleanAlgebra___redArg(v_inst_793_);
return v___x_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteBooleanAlgebra___boxed(lean_object* v_00_u03b1_795_, lean_object* v_inst_796_){
_start:
{
lean_object* v_res_797_; 
v_res_797_ = lp_mathlib_OrderDual_instCompleteBooleanAlgebra(v_00_u03b1_795_, v_inst_796_);
lean_dec_ref(v_inst_796_);
return v_res_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___redArg(lean_object* v_inst_798_){
_start:
{
lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v_toCompleteLattice_801_; lean_object* v_toCompl_802_; lean_object* v_toSDiff_803_; lean_object* v_toHImp_804_; lean_object* v_toHNot_805_; lean_object* v___x_806_; 
v___x_799_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_798_);
v___x_800_ = lp_mathlib_BooleanAlgebra_toBiheytingAlgebra___redArg(v___x_799_);
lean_dec_ref(v___x_799_);
v_toCompleteLattice_801_ = lean_ctor_get(v_inst_798_, 0);
v_toCompl_802_ = lean_ctor_get(v_inst_798_, 1);
v_toSDiff_803_ = lean_ctor_get(v_inst_798_, 2);
v_toHImp_804_ = lean_ctor_get(v_inst_798_, 3);
v_toHNot_805_ = lean_ctor_get(v___x_800_, 2);
lean_inc(v_toHNot_805_);
lean_dec_ref(v___x_800_);
lean_inc(v_toSDiff_803_);
lean_inc(v_toCompl_802_);
lean_inc(v_toHImp_804_);
lean_inc_ref(v_toCompleteLattice_801_);
v___x_806_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_806_, 0, v_toCompleteLattice_801_);
lean_ctor_set(v___x_806_, 1, v_toHImp_804_);
lean_ctor_set(v___x_806_, 2, v_toCompl_802_);
lean_ctor_set(v___x_806_, 3, v_toSDiff_803_);
lean_ctor_set(v___x_806_, 4, v_toHNot_805_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___redArg___boxed(lean_object* v_inst_807_){
_start:
{
lean_object* v_res_808_; 
v_res_808_ = lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___redArg(v_inst_807_);
lean_dec_ref(v_inst_807_);
return v_res_808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice(lean_object* v_00_u03b1_809_, lean_object* v_inst_810_){
_start:
{
lean_object* v___x_811_; 
v___x_811_ = lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___redArg(v_inst_810_);
return v___x_811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___boxed(lean_object* v_00_u03b1_812_, lean_object* v_inst_813_){
_start:
{
lean_object* v_res_814_; 
v_res_814_ = lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice(v_00_u03b1_812_, v_inst_813_);
lean_dec_ref(v_inst_813_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra___redArg(lean_object* v_inst_815_, lean_object* v_inst_816_){
_start:
{
lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v_toCompleteLattice_823_; lean_object* v_toDistribLattice_824_; lean_object* v_toCompl_825_; lean_object* v_toSDiff_826_; lean_object* v_toHImp_827_; lean_object* v_toTop_828_; lean_object* v_toBot_829_; lean_object* v_toSupSet_830_; lean_object* v_toInfSet_831_; lean_object* v___x_833_; uint8_t v_isShared_834_; uint8_t v_isSharedCheck_840_; 
v___x_817_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_815_);
v___x_818_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_816_);
v___x_819_ = lp_mathlib_Prod_instBooleanAlgebra___redArg(v___x_817_, v___x_818_);
v___x_820_ = lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___redArg(v_inst_815_);
v___x_821_ = lp_mathlib_CompleteAtomicBooleanAlgebra_toCompletelyDistribLattice___redArg(v_inst_816_);
v___x_822_ = lp_mathlib_Prod_instCompletelyDistribLattice___redArg(v___x_820_, v___x_821_);
lean_dec_ref(v___x_821_);
lean_dec_ref(v___x_820_);
v_toCompleteLattice_823_ = lean_ctor_get(v___x_822_, 0);
lean_inc_ref(v_toCompleteLattice_823_);
lean_dec_ref(v___x_822_);
v_toDistribLattice_824_ = lean_ctor_get(v___x_819_, 0);
lean_inc_ref(v_toDistribLattice_824_);
v_toCompl_825_ = lean_ctor_get(v___x_819_, 1);
lean_inc(v_toCompl_825_);
v_toSDiff_826_ = lean_ctor_get(v___x_819_, 2);
lean_inc(v_toSDiff_826_);
v_toHImp_827_ = lean_ctor_get(v___x_819_, 3);
lean_inc(v_toHImp_827_);
v_toTop_828_ = lean_ctor_get(v___x_819_, 4);
lean_inc(v_toTop_828_);
v_toBot_829_ = lean_ctor_get(v___x_819_, 5);
lean_inc(v_toBot_829_);
lean_dec_ref(v___x_819_);
v_toSupSet_830_ = lean_ctor_get(v_toCompleteLattice_823_, 1);
v_toInfSet_831_ = lean_ctor_get(v_toCompleteLattice_823_, 2);
v_isSharedCheck_840_ = !lean_is_exclusive(v_toCompleteLattice_823_);
if (v_isSharedCheck_840_ == 0)
{
lean_object* v_unused_841_; lean_object* v_unused_842_; 
v_unused_841_ = lean_ctor_get(v_toCompleteLattice_823_, 3);
lean_dec(v_unused_841_);
v_unused_842_ = lean_ctor_get(v_toCompleteLattice_823_, 0);
lean_dec(v_unused_842_);
v___x_833_ = v_toCompleteLattice_823_;
v_isShared_834_ = v_isSharedCheck_840_;
goto v_resetjp_832_;
}
else
{
lean_inc(v_toInfSet_831_);
lean_inc(v_toSupSet_830_);
lean_dec(v_toCompleteLattice_823_);
v___x_833_ = lean_box(0);
v_isShared_834_ = v_isSharedCheck_840_;
goto v_resetjp_832_;
}
v_resetjp_832_:
{
lean_object* v___x_835_; lean_object* v___x_837_; 
v___x_835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_835_, 0, v_toTop_828_);
lean_ctor_set(v___x_835_, 1, v_toBot_829_);
if (v_isShared_834_ == 0)
{
lean_ctor_set(v___x_833_, 3, v___x_835_);
lean_ctor_set(v___x_833_, 0, v_toDistribLattice_824_);
v___x_837_ = v___x_833_;
goto v_reusejp_836_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v_toDistribLattice_824_);
lean_ctor_set(v_reuseFailAlloc_839_, 1, v_toSupSet_830_);
lean_ctor_set(v_reuseFailAlloc_839_, 2, v_toInfSet_831_);
lean_ctor_set(v_reuseFailAlloc_839_, 3, v___x_835_);
v___x_837_ = v_reuseFailAlloc_839_;
goto v_reusejp_836_;
}
v_reusejp_836_:
{
lean_object* v___x_838_; 
v___x_838_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_838_, 0, v___x_837_);
lean_ctor_set(v___x_838_, 1, v_toCompl_825_);
lean_ctor_set(v___x_838_, 2, v_toSDiff_826_);
lean_ctor_set(v___x_838_, 3, v_toHImp_827_);
return v___x_838_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra___redArg___boxed(lean_object* v_inst_843_, lean_object* v_inst_844_){
_start:
{
lean_object* v_res_845_; 
v_res_845_ = lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra___redArg(v_inst_843_, v_inst_844_);
lean_dec_ref(v_inst_844_);
lean_dec_ref(v_inst_843_);
return v_res_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra(lean_object* v_00_u03b1_846_, lean_object* v_00_u03b2_847_, lean_object* v_inst_848_, lean_object* v_inst_849_){
_start:
{
lean_object* v___x_850_; 
v___x_850_ = lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra___redArg(v_inst_848_, v_inst_849_);
return v___x_850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra___boxed(lean_object* v_00_u03b1_851_, lean_object* v_00_u03b2_852_, lean_object* v_inst_853_, lean_object* v_inst_854_){
_start:
{
lean_object* v_res_855_; 
v_res_855_ = lp_mathlib_Prod_instCompleteAtomicBooleanAlgebra(v_00_u03b1_851_, v_00_u03b2_852_, v_inst_853_, v_inst_854_);
lean_dec_ref(v_inst_854_);
lean_dec_ref(v_inst_853_);
return v_res_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteAtomicBooleanAlgebra___redArg___lam__0(lean_object* v_inst_856_, lean_object* v_i_857_){
_start:
{
lean_object* v___x_858_; 
v___x_858_ = lean_apply_1(v_inst_856_, v_i_857_);
return v___x_858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteAtomicBooleanAlgebra___redArg(lean_object* v_inst_859_){
_start:
{
lean_object* v___f_860_; lean_object* v___x_861_; 
v___f_860_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteAtomicBooleanAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_860_, 0, v_inst_859_);
v___x_861_ = lp_mathlib_Pi_instCompleteBooleanAlgebra___redArg(v___f_860_);
return v___x_861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteAtomicBooleanAlgebra(lean_object* v_00_u03b9_862_, lean_object* v_00_u03c0_863_, lean_object* v_inst_864_){
_start:
{
lean_object* v___x_865_; 
v___x_865_ = lp_mathlib_Pi_instCompleteAtomicBooleanAlgebra___redArg(v_inst_864_);
return v___x_865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra___redArg(lean_object* v_inst_866_){
_start:
{
lean_object* v___x_867_; 
v___x_867_ = lp_mathlib_OrderDual_instCompleteBooleanAlgebra___redArg(v_inst_866_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra___redArg___boxed(lean_object* v_inst_868_){
_start:
{
lean_object* v_res_869_; 
v_res_869_ = lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra___redArg(v_inst_868_);
lean_dec_ref(v_inst_868_);
return v_res_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra(lean_object* v_00_u03b1_870_, lean_object* v_inst_871_){
_start:
{
lean_object* v___x_872_; 
v___x_872_ = lp_mathlib_OrderDual_instCompleteBooleanAlgebra___redArg(v_inst_871_);
return v___x_872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra___boxed(lean_object* v_00_u03b1_873_, lean_object* v_inst_874_){
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_mathlib_OrderDual_instCompleteAtomicBooleanAlgebra(v_00_u03b1_873_, v_inst_874_);
lean_dec_ref(v_inst_874_);
return v_res_875_;
}
}
static lean_object* _init_lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra___closed__0(void){
_start:
{
lean_object* v___x_876_; lean_object* v___x_877_; 
v___x_876_ = lp_mathlib_Prop_instCompleteLattice;
v___x_877_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_877_, 0, v___x_876_);
lean_ctor_set(v___x_877_, 1, lean_box(0));
lean_ctor_set(v___x_877_, 2, lean_box(0));
lean_ctor_set(v___x_877_, 3, lean_box(0));
return v___x_877_;
}
}
static lean_object* _init_lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra(void){
_start:
{
lean_object* v___x_878_; 
v___x_878_ = lean_obj_once(&lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra___closed__0, &lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra___closed__0_once, _init_lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra___closed__0);
return v___x_878_;
}
}
static lean_object* _init_lp_mathlib_Prop_instCompleteBooleanAlgebra(void){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra;
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_frame___redArg___lam__0(lean_object* v_inst_880_, lean_object* v_a_881_, lean_object* v_b_882_){
_start:
{
lean_object* v___x_883_; 
v___x_883_ = lean_apply_2(v_inst_880_, v_a_881_, v_b_882_);
return v___x_883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_frame___redArg(lean_object* v_inst_884_, lean_object* v_inst_885_, lean_object* v_inst_886_, lean_object* v_inst_887_, lean_object* v_inst_888_, lean_object* v_inst_889_, lean_object* v_inst_890_, lean_object* v_inst_891_, lean_object* v_inst_892_, lean_object* v_inst_893_){
_start:
{
lean_object* v___f_894_; lean_object* v___f_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; 
v___f_894_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_894_, 0, v_inst_884_);
v___f_895_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_895_, 0, v_inst_885_);
v___x_896_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_896_, 0, v_inst_886_);
lean_ctor_set(v___x_896_, 1, v_inst_887_);
v___x_897_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_897_, 0, v___x_896_);
lean_ctor_set(v___x_897_, 1, v___f_894_);
v___x_898_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_898_, 0, v___x_897_);
lean_ctor_set(v___x_898_, 1, v___f_895_);
v___x_899_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_899_, 0, v_inst_890_);
lean_ctor_set(v___x_899_, 1, v_inst_891_);
v___x_900_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_900_, 0, v___x_898_);
lean_ctor_set(v___x_900_, 1, v_inst_888_);
lean_ctor_set(v___x_900_, 2, v_inst_889_);
lean_ctor_set(v___x_900_, 3, v___x_899_);
v___x_901_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_901_, 0, v___x_900_);
lean_ctor_set(v___x_901_, 1, v_inst_893_);
lean_ctor_set(v___x_901_, 2, v_inst_892_);
return v___x_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_frame(lean_object* v_00_u03b1_902_, lean_object* v_00_u03b2_903_, lean_object* v_inst_904_, lean_object* v_inst_905_, lean_object* v_inst_906_, lean_object* v_inst_907_, lean_object* v_inst_908_, lean_object* v_inst_909_, lean_object* v_inst_910_, lean_object* v_inst_911_, lean_object* v_inst_912_, lean_object* v_inst_913_, lean_object* v_inst_914_, lean_object* v_f_915_, lean_object* v_hf_916_, lean_object* v_le_917_, lean_object* v_lt_918_, lean_object* v_map__sup_919_, lean_object* v_map__inf_920_, lean_object* v_map__sSup_921_, lean_object* v_map__sInf_922_, lean_object* v_map__top_923_, lean_object* v_map__bot_924_, lean_object* v_map__compl_925_, lean_object* v_map__himp_926_){
_start:
{
lean_object* v___f_927_; lean_object* v___f_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; 
v___f_927_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_927_, 0, v_inst_904_);
v___f_928_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_928_, 0, v_inst_905_);
v___x_929_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_929_, 0, v_inst_906_);
lean_ctor_set(v___x_929_, 1, v_inst_907_);
v___x_930_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_930_, 0, v___x_929_);
lean_ctor_set(v___x_930_, 1, v___f_927_);
v___x_931_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_931_, 0, v___x_930_);
lean_ctor_set(v___x_931_, 1, v___f_928_);
v___x_932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_932_, 0, v_inst_910_);
lean_ctor_set(v___x_932_, 1, v_inst_911_);
v___x_933_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_933_, 0, v___x_931_);
lean_ctor_set(v___x_933_, 1, v_inst_908_);
lean_ctor_set(v___x_933_, 2, v_inst_909_);
lean_ctor_set(v___x_933_, 3, v___x_932_);
v___x_934_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_934_, 0, v___x_933_);
lean_ctor_set(v___x_934_, 1, v_inst_913_);
lean_ctor_set(v___x_934_, 2, v_inst_912_);
return v___x_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_frame___boxed(lean_object** _args){
lean_object* v_00_u03b1_935_ = _args[0];
lean_object* v_00_u03b2_936_ = _args[1];
lean_object* v_inst_937_ = _args[2];
lean_object* v_inst_938_ = _args[3];
lean_object* v_inst_939_ = _args[4];
lean_object* v_inst_940_ = _args[5];
lean_object* v_inst_941_ = _args[6];
lean_object* v_inst_942_ = _args[7];
lean_object* v_inst_943_ = _args[8];
lean_object* v_inst_944_ = _args[9];
lean_object* v_inst_945_ = _args[10];
lean_object* v_inst_946_ = _args[11];
lean_object* v_inst_947_ = _args[12];
lean_object* v_f_948_ = _args[13];
lean_object* v_hf_949_ = _args[14];
lean_object* v_le_950_ = _args[15];
lean_object* v_lt_951_ = _args[16];
lean_object* v_map__sup_952_ = _args[17];
lean_object* v_map__inf_953_ = _args[18];
lean_object* v_map__sSup_954_ = _args[19];
lean_object* v_map__sInf_955_ = _args[20];
lean_object* v_map__top_956_ = _args[21];
lean_object* v_map__bot_957_ = _args[22];
lean_object* v_map__compl_958_ = _args[23];
lean_object* v_map__himp_959_ = _args[24];
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_mathlib_Function_Injective_frame(v_00_u03b1_935_, v_00_u03b2_936_, v_inst_937_, v_inst_938_, v_inst_939_, v_inst_940_, v_inst_941_, v_inst_942_, v_inst_943_, v_inst_944_, v_inst_945_, v_inst_946_, v_inst_947_, v_f_948_, v_hf_949_, v_le_950_, v_lt_951_, v_map__sup_952_, v_map__inf_953_, v_map__sSup_954_, v_map__sInf_955_, v_map__top_956_, v_map__bot_957_, v_map__compl_958_, v_map__himp_959_);
lean_dec(v_f_948_);
lean_dec_ref(v_inst_947_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coframe___redArg(lean_object* v_inst_961_, lean_object* v_inst_962_, lean_object* v_inst_963_, lean_object* v_inst_964_, lean_object* v_inst_965_, lean_object* v_inst_966_, lean_object* v_inst_967_, lean_object* v_inst_968_, lean_object* v_inst_969_, lean_object* v_inst_970_){
_start:
{
lean_object* v___f_971_; lean_object* v___f_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v_toGeneralizedCoheytingAlgebra_979_; lean_object* v_toHNot_980_; lean_object* v_toSDiff_981_; lean_object* v___x_983_; uint8_t v_isShared_984_; uint8_t v_isSharedCheck_988_; 
lean_inc(v_inst_961_);
v___f_971_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_971_, 0, v_inst_961_);
lean_inc(v_inst_962_);
v___f_972_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_972_, 0, v_inst_962_);
v___x_973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_973_, 0, v_inst_963_);
lean_ctor_set(v___x_973_, 1, v_inst_964_);
v___x_974_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_974_, 0, v___x_973_);
lean_ctor_set(v___x_974_, 1, v___f_971_);
v___x_975_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_975_, 0, v___x_974_);
lean_ctor_set(v___x_975_, 1, v___f_972_);
lean_inc(v_inst_968_);
lean_inc(v_inst_967_);
v___x_976_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_976_, 0, v_inst_967_);
lean_ctor_set(v___x_976_, 1, v_inst_968_);
v___x_977_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_977_, 0, v___x_975_);
lean_ctor_set(v___x_977_, 1, v_inst_965_);
lean_ctor_set(v___x_977_, 2, v_inst_966_);
lean_ctor_set(v___x_977_, 3, v___x_976_);
v___x_978_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_962_, v_inst_961_, v_inst_963_, v_inst_964_, v_inst_968_, v_inst_967_, v_inst_969_, v_inst_970_);
v_toGeneralizedCoheytingAlgebra_979_ = lean_ctor_get(v___x_978_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_979_);
v_toHNot_980_ = lean_ctor_get(v___x_978_, 2);
lean_inc(v_toHNot_980_);
lean_dec_ref(v___x_978_);
v_toSDiff_981_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_979_, 2);
v_isSharedCheck_988_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_979_);
if (v_isSharedCheck_988_ == 0)
{
lean_object* v_unused_989_; lean_object* v_unused_990_; 
v_unused_989_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_979_, 1);
lean_dec(v_unused_989_);
v_unused_990_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_979_, 0);
lean_dec(v_unused_990_);
v___x_983_ = v_toGeneralizedCoheytingAlgebra_979_;
v_isShared_984_ = v_isSharedCheck_988_;
goto v_resetjp_982_;
}
else
{
lean_inc(v_toSDiff_981_);
lean_dec(v_toGeneralizedCoheytingAlgebra_979_);
v___x_983_ = lean_box(0);
v_isShared_984_ = v_isSharedCheck_988_;
goto v_resetjp_982_;
}
v_resetjp_982_:
{
lean_object* v___x_986_; 
if (v_isShared_984_ == 0)
{
lean_ctor_set(v___x_983_, 2, v_toHNot_980_);
lean_ctor_set(v___x_983_, 1, v_toSDiff_981_);
lean_ctor_set(v___x_983_, 0, v___x_977_);
v___x_986_ = v___x_983_;
goto v_reusejp_985_;
}
else
{
lean_object* v_reuseFailAlloc_987_; 
v_reuseFailAlloc_987_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_987_, 0, v___x_977_);
lean_ctor_set(v_reuseFailAlloc_987_, 1, v_toSDiff_981_);
lean_ctor_set(v_reuseFailAlloc_987_, 2, v_toHNot_980_);
v___x_986_ = v_reuseFailAlloc_987_;
goto v_reusejp_985_;
}
v_reusejp_985_:
{
return v___x_986_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coframe(lean_object* v_00_u03b1_991_, lean_object* v_00_u03b2_992_, lean_object* v_inst_993_, lean_object* v_inst_994_, lean_object* v_inst_995_, lean_object* v_inst_996_, lean_object* v_inst_997_, lean_object* v_inst_998_, lean_object* v_inst_999_, lean_object* v_inst_1000_, lean_object* v_inst_1001_, lean_object* v_inst_1002_, lean_object* v_inst_1003_, lean_object* v_f_1004_, lean_object* v_hf_1005_, lean_object* v_le_1006_, lean_object* v_lt_1007_, lean_object* v_map__sup_1008_, lean_object* v_map__inf_1009_, lean_object* v_map__sSup_1010_, lean_object* v_map__sInf_1011_, lean_object* v_map__top_1012_, lean_object* v_map__bot_1013_, lean_object* v_map__hnot_1014_, lean_object* v_map__sdiff_1015_){
_start:
{
lean_object* v___f_1016_; lean_object* v___f_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v_toGeneralizedCoheytingAlgebra_1024_; lean_object* v_toHNot_1025_; lean_object* v_toSDiff_1026_; lean_object* v___x_1028_; uint8_t v_isShared_1029_; uint8_t v_isSharedCheck_1033_; 
lean_inc(v_inst_993_);
v___f_1016_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1016_, 0, v_inst_993_);
lean_inc(v_inst_994_);
v___f_1017_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1017_, 0, v_inst_994_);
v___x_1018_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1018_, 0, v_inst_995_);
lean_ctor_set(v___x_1018_, 1, v_inst_996_);
v___x_1019_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1019_, 0, v___x_1018_);
lean_ctor_set(v___x_1019_, 1, v___f_1016_);
v___x_1020_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1020_, 0, v___x_1019_);
lean_ctor_set(v___x_1020_, 1, v___f_1017_);
lean_inc(v_inst_1000_);
lean_inc(v_inst_999_);
v___x_1021_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1021_, 0, v_inst_999_);
lean_ctor_set(v___x_1021_, 1, v_inst_1000_);
v___x_1022_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1022_, 0, v___x_1020_);
lean_ctor_set(v___x_1022_, 1, v_inst_997_);
lean_ctor_set(v___x_1022_, 2, v_inst_998_);
lean_ctor_set(v___x_1022_, 3, v___x_1021_);
v___x_1023_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_994_, v_inst_993_, v_inst_995_, v_inst_996_, v_inst_1000_, v_inst_999_, v_inst_1001_, v_inst_1002_);
v_toGeneralizedCoheytingAlgebra_1024_ = lean_ctor_get(v___x_1023_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1024_);
v_toHNot_1025_ = lean_ctor_get(v___x_1023_, 2);
lean_inc(v_toHNot_1025_);
lean_dec_ref(v___x_1023_);
v_toSDiff_1026_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1024_, 2);
v_isSharedCheck_1033_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_1024_);
if (v_isSharedCheck_1033_ == 0)
{
lean_object* v_unused_1034_; lean_object* v_unused_1035_; 
v_unused_1034_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1024_, 1);
lean_dec(v_unused_1034_);
v_unused_1035_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1024_, 0);
lean_dec(v_unused_1035_);
v___x_1028_ = v_toGeneralizedCoheytingAlgebra_1024_;
v_isShared_1029_ = v_isSharedCheck_1033_;
goto v_resetjp_1027_;
}
else
{
lean_inc(v_toSDiff_1026_);
lean_dec(v_toGeneralizedCoheytingAlgebra_1024_);
v___x_1028_ = lean_box(0);
v_isShared_1029_ = v_isSharedCheck_1033_;
goto v_resetjp_1027_;
}
v_resetjp_1027_:
{
lean_object* v___x_1031_; 
if (v_isShared_1029_ == 0)
{
lean_ctor_set(v___x_1028_, 2, v_toHNot_1025_);
lean_ctor_set(v___x_1028_, 1, v_toSDiff_1026_);
lean_ctor_set(v___x_1028_, 0, v___x_1022_);
v___x_1031_ = v___x_1028_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v___x_1022_);
lean_ctor_set(v_reuseFailAlloc_1032_, 1, v_toSDiff_1026_);
lean_ctor_set(v_reuseFailAlloc_1032_, 2, v_toHNot_1025_);
v___x_1031_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
return v___x_1031_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coframe___boxed(lean_object** _args){
lean_object* v_00_u03b1_1036_ = _args[0];
lean_object* v_00_u03b2_1037_ = _args[1];
lean_object* v_inst_1038_ = _args[2];
lean_object* v_inst_1039_ = _args[3];
lean_object* v_inst_1040_ = _args[4];
lean_object* v_inst_1041_ = _args[5];
lean_object* v_inst_1042_ = _args[6];
lean_object* v_inst_1043_ = _args[7];
lean_object* v_inst_1044_ = _args[8];
lean_object* v_inst_1045_ = _args[9];
lean_object* v_inst_1046_ = _args[10];
lean_object* v_inst_1047_ = _args[11];
lean_object* v_inst_1048_ = _args[12];
lean_object* v_f_1049_ = _args[13];
lean_object* v_hf_1050_ = _args[14];
lean_object* v_le_1051_ = _args[15];
lean_object* v_lt_1052_ = _args[16];
lean_object* v_map__sup_1053_ = _args[17];
lean_object* v_map__inf_1054_ = _args[18];
lean_object* v_map__sSup_1055_ = _args[19];
lean_object* v_map__sInf_1056_ = _args[20];
lean_object* v_map__top_1057_ = _args[21];
lean_object* v_map__bot_1058_ = _args[22];
lean_object* v_map__hnot_1059_ = _args[23];
lean_object* v_map__sdiff_1060_ = _args[24];
_start:
{
lean_object* v_res_1061_; 
v_res_1061_ = lp_mathlib_Function_Injective_coframe(v_00_u03b1_1036_, v_00_u03b2_1037_, v_inst_1038_, v_inst_1039_, v_inst_1040_, v_inst_1041_, v_inst_1042_, v_inst_1043_, v_inst_1044_, v_inst_1045_, v_inst_1046_, v_inst_1047_, v_inst_1048_, v_f_1049_, v_hf_1050_, v_le_1051_, v_lt_1052_, v_map__sup_1053_, v_map__inf_1054_, v_map__sSup_1055_, v_map__sInf_1056_, v_map__top_1057_, v_map__bot_1058_, v_map__hnot_1059_, v_map__sdiff_1060_);
lean_dec(v_f_1049_);
lean_dec_ref(v_inst_1048_);
return v_res_1061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeDistribLattice___redArg(lean_object* v_inst_1062_, lean_object* v_inst_1063_, lean_object* v_inst_1064_, lean_object* v_inst_1065_, lean_object* v_inst_1066_, lean_object* v_inst_1067_, lean_object* v_inst_1068_, lean_object* v_inst_1069_, lean_object* v_inst_1070_, lean_object* v_inst_1071_, lean_object* v_inst_1072_, lean_object* v_inst_1073_){
_start:
{
lean_object* v___f_1074_; lean_object* v___f_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v_toGeneralizedCoheytingAlgebra_1083_; lean_object* v_toHNot_1084_; lean_object* v_toSDiff_1085_; lean_object* v___x_1087_; uint8_t v_isShared_1088_; uint8_t v_isSharedCheck_1092_; 
lean_inc(v_inst_1062_);
v___f_1074_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1074_, 0, v_inst_1062_);
lean_inc(v_inst_1063_);
v___f_1075_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1075_, 0, v_inst_1063_);
v___x_1076_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1076_, 0, v_inst_1064_);
lean_ctor_set(v___x_1076_, 1, v_inst_1065_);
v___x_1077_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1077_, 0, v___x_1076_);
lean_ctor_set(v___x_1077_, 1, v___f_1074_);
v___x_1078_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1078_, 0, v___x_1077_);
lean_ctor_set(v___x_1078_, 1, v___f_1075_);
lean_inc(v_inst_1069_);
lean_inc(v_inst_1068_);
v___x_1079_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1079_, 0, v_inst_1068_);
lean_ctor_set(v___x_1079_, 1, v_inst_1069_);
v___x_1080_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1080_, 0, v___x_1078_);
lean_ctor_set(v___x_1080_, 1, v_inst_1066_);
lean_ctor_set(v___x_1080_, 2, v_inst_1067_);
lean_ctor_set(v___x_1080_, 3, v___x_1079_);
v___x_1081_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1081_, 0, v___x_1080_);
lean_ctor_set(v___x_1081_, 1, v_inst_1071_);
lean_ctor_set(v___x_1081_, 2, v_inst_1070_);
v___x_1082_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_1063_, v_inst_1062_, v_inst_1064_, v_inst_1065_, v_inst_1069_, v_inst_1068_, v_inst_1072_, v_inst_1073_);
v_toGeneralizedCoheytingAlgebra_1083_ = lean_ctor_get(v___x_1082_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1083_);
v_toHNot_1084_ = lean_ctor_get(v___x_1082_, 2);
lean_inc(v_toHNot_1084_);
lean_dec_ref(v___x_1082_);
v_toSDiff_1085_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1083_, 2);
v_isSharedCheck_1092_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_1083_);
if (v_isSharedCheck_1092_ == 0)
{
lean_object* v_unused_1093_; lean_object* v_unused_1094_; 
v_unused_1093_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1083_, 1);
lean_dec(v_unused_1093_);
v_unused_1094_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1083_, 0);
lean_dec(v_unused_1094_);
v___x_1087_ = v_toGeneralizedCoheytingAlgebra_1083_;
v_isShared_1088_ = v_isSharedCheck_1092_;
goto v_resetjp_1086_;
}
else
{
lean_inc(v_toSDiff_1085_);
lean_dec(v_toGeneralizedCoheytingAlgebra_1083_);
v___x_1087_ = lean_box(0);
v_isShared_1088_ = v_isSharedCheck_1092_;
goto v_resetjp_1086_;
}
v_resetjp_1086_:
{
lean_object* v___x_1090_; 
if (v_isShared_1088_ == 0)
{
lean_ctor_set(v___x_1087_, 2, v_toHNot_1084_);
lean_ctor_set(v___x_1087_, 1, v_toSDiff_1085_);
lean_ctor_set(v___x_1087_, 0, v___x_1081_);
v___x_1090_ = v___x_1087_;
goto v_reusejp_1089_;
}
else
{
lean_object* v_reuseFailAlloc_1091_; 
v_reuseFailAlloc_1091_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1091_, 0, v___x_1081_);
lean_ctor_set(v_reuseFailAlloc_1091_, 1, v_toSDiff_1085_);
lean_ctor_set(v_reuseFailAlloc_1091_, 2, v_toHNot_1084_);
v___x_1090_ = v_reuseFailAlloc_1091_;
goto v_reusejp_1089_;
}
v_reusejp_1089_:
{
return v___x_1090_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeDistribLattice(lean_object* v_00_u03b1_1095_, lean_object* v_00_u03b2_1096_, lean_object* v_inst_1097_, lean_object* v_inst_1098_, lean_object* v_inst_1099_, lean_object* v_inst_1100_, lean_object* v_inst_1101_, lean_object* v_inst_1102_, lean_object* v_inst_1103_, lean_object* v_inst_1104_, lean_object* v_inst_1105_, lean_object* v_inst_1106_, lean_object* v_inst_1107_, lean_object* v_inst_1108_, lean_object* v_inst_1109_, lean_object* v_f_1110_, lean_object* v_hf_1111_, lean_object* v_le_1112_, lean_object* v_lt_1113_, lean_object* v_map__sup_1114_, lean_object* v_map__inf_1115_, lean_object* v_map__sSup_1116_, lean_object* v_map__sInf_1117_, lean_object* v_map__top_1118_, lean_object* v_map__bot_1119_, lean_object* v_map__compl_1120_, lean_object* v_map__himp_1121_, lean_object* v_map__hnot_1122_, lean_object* v_map__sdiff_1123_){
_start:
{
lean_object* v___f_1124_; lean_object* v___f_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v_toGeneralizedCoheytingAlgebra_1133_; lean_object* v_toHNot_1134_; lean_object* v_toSDiff_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1142_; 
lean_inc(v_inst_1097_);
v___f_1124_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1124_, 0, v_inst_1097_);
lean_inc(v_inst_1098_);
v___f_1125_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1125_, 0, v_inst_1098_);
v___x_1126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1126_, 0, v_inst_1099_);
lean_ctor_set(v___x_1126_, 1, v_inst_1100_);
v___x_1127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1127_, 0, v___x_1126_);
lean_ctor_set(v___x_1127_, 1, v___f_1124_);
v___x_1128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1128_, 0, v___x_1127_);
lean_ctor_set(v___x_1128_, 1, v___f_1125_);
lean_inc(v_inst_1104_);
lean_inc(v_inst_1103_);
v___x_1129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1129_, 0, v_inst_1103_);
lean_ctor_set(v___x_1129_, 1, v_inst_1104_);
v___x_1130_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1130_, 0, v___x_1128_);
lean_ctor_set(v___x_1130_, 1, v_inst_1101_);
lean_ctor_set(v___x_1130_, 2, v_inst_1102_);
lean_ctor_set(v___x_1130_, 3, v___x_1129_);
v___x_1131_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1131_, 0, v___x_1130_);
lean_ctor_set(v___x_1131_, 1, v_inst_1106_);
lean_ctor_set(v___x_1131_, 2, v_inst_1105_);
v___x_1132_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_1098_, v_inst_1097_, v_inst_1099_, v_inst_1100_, v_inst_1104_, v_inst_1103_, v_inst_1107_, v_inst_1108_);
v_toGeneralizedCoheytingAlgebra_1133_ = lean_ctor_get(v___x_1132_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1133_);
v_toHNot_1134_ = lean_ctor_get(v___x_1132_, 2);
lean_inc(v_toHNot_1134_);
lean_dec_ref(v___x_1132_);
v_toSDiff_1135_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1133_, 2);
v_isSharedCheck_1142_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_1133_);
if (v_isSharedCheck_1142_ == 0)
{
lean_object* v_unused_1143_; lean_object* v_unused_1144_; 
v_unused_1143_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1133_, 1);
lean_dec(v_unused_1143_);
v_unused_1144_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1133_, 0);
lean_dec(v_unused_1144_);
v___x_1137_ = v_toGeneralizedCoheytingAlgebra_1133_;
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_toSDiff_1135_);
lean_dec(v_toGeneralizedCoheytingAlgebra_1133_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v___x_1140_; 
if (v_isShared_1138_ == 0)
{
lean_ctor_set(v___x_1137_, 2, v_toHNot_1134_);
lean_ctor_set(v___x_1137_, 1, v_toSDiff_1135_);
lean_ctor_set(v___x_1137_, 0, v___x_1131_);
v___x_1140_ = v___x_1137_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v___x_1131_);
lean_ctor_set(v_reuseFailAlloc_1141_, 1, v_toSDiff_1135_);
lean_ctor_set(v_reuseFailAlloc_1141_, 2, v_toHNot_1134_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeDistribLattice___boxed(lean_object** _args){
lean_object* v_00_u03b1_1145_ = _args[0];
lean_object* v_00_u03b2_1146_ = _args[1];
lean_object* v_inst_1147_ = _args[2];
lean_object* v_inst_1148_ = _args[3];
lean_object* v_inst_1149_ = _args[4];
lean_object* v_inst_1150_ = _args[5];
lean_object* v_inst_1151_ = _args[6];
lean_object* v_inst_1152_ = _args[7];
lean_object* v_inst_1153_ = _args[8];
lean_object* v_inst_1154_ = _args[9];
lean_object* v_inst_1155_ = _args[10];
lean_object* v_inst_1156_ = _args[11];
lean_object* v_inst_1157_ = _args[12];
lean_object* v_inst_1158_ = _args[13];
lean_object* v_inst_1159_ = _args[14];
lean_object* v_f_1160_ = _args[15];
lean_object* v_hf_1161_ = _args[16];
lean_object* v_le_1162_ = _args[17];
lean_object* v_lt_1163_ = _args[18];
lean_object* v_map__sup_1164_ = _args[19];
lean_object* v_map__inf_1165_ = _args[20];
lean_object* v_map__sSup_1166_ = _args[21];
lean_object* v_map__sInf_1167_ = _args[22];
lean_object* v_map__top_1168_ = _args[23];
lean_object* v_map__bot_1169_ = _args[24];
lean_object* v_map__compl_1170_ = _args[25];
lean_object* v_map__himp_1171_ = _args[26];
lean_object* v_map__hnot_1172_ = _args[27];
lean_object* v_map__sdiff_1173_ = _args[28];
_start:
{
lean_object* v_res_1174_; 
v_res_1174_ = lp_mathlib_Function_Injective_completeDistribLattice(v_00_u03b1_1145_, v_00_u03b2_1146_, v_inst_1147_, v_inst_1148_, v_inst_1149_, v_inst_1150_, v_inst_1151_, v_inst_1152_, v_inst_1153_, v_inst_1154_, v_inst_1155_, v_inst_1156_, v_inst_1157_, v_inst_1158_, v_inst_1159_, v_f_1160_, v_hf_1161_, v_le_1162_, v_lt_1163_, v_map__sup_1164_, v_map__inf_1165_, v_map__sSup_1166_, v_map__sInf_1167_, v_map__top_1168_, v_map__bot_1169_, v_map__compl_1170_, v_map__himp_1171_, v_map__hnot_1172_, v_map__sdiff_1173_);
lean_dec(v_f_1160_);
lean_dec_ref(v_inst_1159_);
return v_res_1174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completelyDistribLattice___redArg(lean_object* v_inst_1175_, lean_object* v_inst_1176_, lean_object* v_inst_1177_, lean_object* v_inst_1178_, lean_object* v_inst_1179_, lean_object* v_inst_1180_, lean_object* v_inst_1181_, lean_object* v_inst_1182_, lean_object* v_inst_1183_, lean_object* v_inst_1184_, lean_object* v_inst_1185_, lean_object* v_inst_1186_){
_start:
{
lean_object* v___f_1187_; lean_object* v___f_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v_toGeneralizedCoheytingAlgebra_1195_; lean_object* v_toHNot_1196_; lean_object* v_toSDiff_1197_; lean_object* v___x_1198_; 
lean_inc(v_inst_1175_);
v___f_1187_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1187_, 0, v_inst_1175_);
lean_inc(v_inst_1176_);
v___f_1188_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1188_, 0, v_inst_1176_);
v___x_1189_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1189_, 0, v_inst_1177_);
lean_ctor_set(v___x_1189_, 1, v_inst_1178_);
v___x_1190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1190_, 0, v___x_1189_);
lean_ctor_set(v___x_1190_, 1, v___f_1187_);
v___x_1191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1191_, 0, v___x_1190_);
lean_ctor_set(v___x_1191_, 1, v___f_1188_);
lean_inc(v_inst_1182_);
lean_inc(v_inst_1181_);
v___x_1192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1192_, 0, v_inst_1181_);
lean_ctor_set(v___x_1192_, 1, v_inst_1182_);
v___x_1193_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1193_, 0, v___x_1191_);
lean_ctor_set(v___x_1193_, 1, v_inst_1179_);
lean_ctor_set(v___x_1193_, 2, v_inst_1180_);
lean_ctor_set(v___x_1193_, 3, v___x_1192_);
v___x_1194_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_1176_, v_inst_1175_, v_inst_1177_, v_inst_1178_, v_inst_1182_, v_inst_1181_, v_inst_1185_, v_inst_1186_);
v_toGeneralizedCoheytingAlgebra_1195_ = lean_ctor_get(v___x_1194_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1195_);
v_toHNot_1196_ = lean_ctor_get(v___x_1194_, 2);
lean_inc(v_toHNot_1196_);
lean_dec_ref(v___x_1194_);
v_toSDiff_1197_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1195_, 2);
lean_inc(v_toSDiff_1197_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1195_);
v___x_1198_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1198_, 0, v___x_1193_);
lean_ctor_set(v___x_1198_, 1, v_inst_1184_);
lean_ctor_set(v___x_1198_, 2, v_inst_1183_);
lean_ctor_set(v___x_1198_, 3, v_toSDiff_1197_);
lean_ctor_set(v___x_1198_, 4, v_toHNot_1196_);
return v___x_1198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completelyDistribLattice(lean_object* v_00_u03b1_1199_, lean_object* v_00_u03b2_1200_, lean_object* v_inst_1201_, lean_object* v_inst_1202_, lean_object* v_inst_1203_, lean_object* v_inst_1204_, lean_object* v_inst_1205_, lean_object* v_inst_1206_, lean_object* v_inst_1207_, lean_object* v_inst_1208_, lean_object* v_inst_1209_, lean_object* v_inst_1210_, lean_object* v_inst_1211_, lean_object* v_inst_1212_, lean_object* v_inst_1213_, lean_object* v_f_1214_, lean_object* v_hf_1215_, lean_object* v_le_1216_, lean_object* v_lt_1217_, lean_object* v_map__sup_1218_, lean_object* v_map__inf_1219_, lean_object* v_map__sSup_1220_, lean_object* v_map__sInf_1221_, lean_object* v_map__top_1222_, lean_object* v_map__bot_1223_, lean_object* v_map__compl_1224_, lean_object* v_map__himp_1225_, lean_object* v_map__hnot_1226_, lean_object* v_map__sdiff_1227_){
_start:
{
lean_object* v___f_1228_; lean_object* v___f_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v_toGeneralizedCoheytingAlgebra_1236_; lean_object* v_toHNot_1237_; lean_object* v_toSDiff_1238_; lean_object* v___x_1239_; 
lean_inc(v_inst_1201_);
v___f_1228_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1228_, 0, v_inst_1201_);
lean_inc(v_inst_1202_);
v___f_1229_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1229_, 0, v_inst_1202_);
v___x_1230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1230_, 0, v_inst_1203_);
lean_ctor_set(v___x_1230_, 1, v_inst_1204_);
v___x_1231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1231_, 0, v___x_1230_);
lean_ctor_set(v___x_1231_, 1, v___f_1228_);
v___x_1232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1232_, 0, v___x_1231_);
lean_ctor_set(v___x_1232_, 1, v___f_1229_);
lean_inc(v_inst_1208_);
lean_inc(v_inst_1207_);
v___x_1233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1233_, 0, v_inst_1207_);
lean_ctor_set(v___x_1233_, 1, v_inst_1208_);
v___x_1234_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1234_, 0, v___x_1232_);
lean_ctor_set(v___x_1234_, 1, v_inst_1205_);
lean_ctor_set(v___x_1234_, 2, v_inst_1206_);
lean_ctor_set(v___x_1234_, 3, v___x_1233_);
v___x_1235_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_1202_, v_inst_1201_, v_inst_1203_, v_inst_1204_, v_inst_1208_, v_inst_1207_, v_inst_1211_, v_inst_1212_);
v_toGeneralizedCoheytingAlgebra_1236_ = lean_ctor_get(v___x_1235_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1236_);
v_toHNot_1237_ = lean_ctor_get(v___x_1235_, 2);
lean_inc(v_toHNot_1237_);
lean_dec_ref(v___x_1235_);
v_toSDiff_1238_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1236_, 2);
lean_inc(v_toSDiff_1238_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1236_);
v___x_1239_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1239_, 0, v___x_1234_);
lean_ctor_set(v___x_1239_, 1, v_inst_1210_);
lean_ctor_set(v___x_1239_, 2, v_inst_1209_);
lean_ctor_set(v___x_1239_, 3, v_toSDiff_1238_);
lean_ctor_set(v___x_1239_, 4, v_toHNot_1237_);
return v___x_1239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completelyDistribLattice___boxed(lean_object** _args){
lean_object* v_00_u03b1_1240_ = _args[0];
lean_object* v_00_u03b2_1241_ = _args[1];
lean_object* v_inst_1242_ = _args[2];
lean_object* v_inst_1243_ = _args[3];
lean_object* v_inst_1244_ = _args[4];
lean_object* v_inst_1245_ = _args[5];
lean_object* v_inst_1246_ = _args[6];
lean_object* v_inst_1247_ = _args[7];
lean_object* v_inst_1248_ = _args[8];
lean_object* v_inst_1249_ = _args[9];
lean_object* v_inst_1250_ = _args[10];
lean_object* v_inst_1251_ = _args[11];
lean_object* v_inst_1252_ = _args[12];
lean_object* v_inst_1253_ = _args[13];
lean_object* v_inst_1254_ = _args[14];
lean_object* v_f_1255_ = _args[15];
lean_object* v_hf_1256_ = _args[16];
lean_object* v_le_1257_ = _args[17];
lean_object* v_lt_1258_ = _args[18];
lean_object* v_map__sup_1259_ = _args[19];
lean_object* v_map__inf_1260_ = _args[20];
lean_object* v_map__sSup_1261_ = _args[21];
lean_object* v_map__sInf_1262_ = _args[22];
lean_object* v_map__top_1263_ = _args[23];
lean_object* v_map__bot_1264_ = _args[24];
lean_object* v_map__compl_1265_ = _args[25];
lean_object* v_map__himp_1266_ = _args[26];
lean_object* v_map__hnot_1267_ = _args[27];
lean_object* v_map__sdiff_1268_ = _args[28];
_start:
{
lean_object* v_res_1269_; 
v_res_1269_ = lp_mathlib_Function_Injective_completelyDistribLattice(v_00_u03b1_1240_, v_00_u03b2_1241_, v_inst_1242_, v_inst_1243_, v_inst_1244_, v_inst_1245_, v_inst_1246_, v_inst_1247_, v_inst_1248_, v_inst_1249_, v_inst_1250_, v_inst_1251_, v_inst_1252_, v_inst_1253_, v_inst_1254_, v_f_1255_, v_hf_1256_, v_le_1257_, v_lt_1258_, v_map__sup_1259_, v_map__inf_1260_, v_map__sSup_1261_, v_map__sInf_1262_, v_map__top_1263_, v_map__bot_1264_, v_map__compl_1265_, v_map__himp_1266_, v_map__hnot_1267_, v_map__sdiff_1268_);
lean_dec(v_f_1255_);
lean_dec_ref(v_inst_1254_);
return v_res_1269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeBooleanAlgebra___redArg(lean_object* v_inst_1270_, lean_object* v_inst_1271_, lean_object* v_inst_1272_, lean_object* v_inst_1273_, lean_object* v_inst_1274_, lean_object* v_inst_1275_, lean_object* v_inst_1276_, lean_object* v_inst_1277_, lean_object* v_inst_1278_, lean_object* v_inst_1279_, lean_object* v_inst_1280_){
_start:
{
lean_object* v___f_1281_; lean_object* v___f_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; 
v___f_1281_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1281_, 0, v_inst_1270_);
v___f_1282_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1282_, 0, v_inst_1271_);
v___x_1283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1283_, 0, v_inst_1272_);
lean_ctor_set(v___x_1283_, 1, v_inst_1273_);
v___x_1284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1284_, 0, v___x_1283_);
lean_ctor_set(v___x_1284_, 1, v___f_1281_);
v___x_1285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1285_, 0, v___x_1284_);
lean_ctor_set(v___x_1285_, 1, v___f_1282_);
v___x_1286_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1286_, 0, v_inst_1276_);
lean_ctor_set(v___x_1286_, 1, v_inst_1277_);
v___x_1287_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1287_, 0, v___x_1285_);
lean_ctor_set(v___x_1287_, 1, v_inst_1274_);
lean_ctor_set(v___x_1287_, 2, v_inst_1275_);
lean_ctor_set(v___x_1287_, 3, v___x_1286_);
v___x_1288_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1287_);
lean_ctor_set(v___x_1288_, 1, v_inst_1278_);
lean_ctor_set(v___x_1288_, 2, v_inst_1280_);
lean_ctor_set(v___x_1288_, 3, v_inst_1279_);
return v___x_1288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeBooleanAlgebra(lean_object* v_00_u03b1_1289_, lean_object* v_00_u03b2_1290_, lean_object* v_inst_1291_, lean_object* v_inst_1292_, lean_object* v_inst_1293_, lean_object* v_inst_1294_, lean_object* v_inst_1295_, lean_object* v_inst_1296_, lean_object* v_inst_1297_, lean_object* v_inst_1298_, lean_object* v_inst_1299_, lean_object* v_inst_1300_, lean_object* v_inst_1301_, lean_object* v_inst_1302_, lean_object* v_f_1303_, lean_object* v_hf_1304_, lean_object* v_le_1305_, lean_object* v_lt_1306_, lean_object* v_map__sup_1307_, lean_object* v_map__inf_1308_, lean_object* v_map__sSup_1309_, lean_object* v_map__sInf_1310_, lean_object* v_map__top_1311_, lean_object* v_map__bot_1312_, lean_object* v_map__compl_1313_, lean_object* v_map__himp_1314_, lean_object* v_map__sdiff_1315_){
_start:
{
lean_object* v___f_1316_; lean_object* v___f_1317_; lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; 
v___f_1316_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1316_, 0, v_inst_1291_);
v___f_1317_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1317_, 0, v_inst_1292_);
v___x_1318_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1318_, 0, v_inst_1293_);
lean_ctor_set(v___x_1318_, 1, v_inst_1294_);
v___x_1319_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1319_, 0, v___x_1318_);
lean_ctor_set(v___x_1319_, 1, v___f_1316_);
v___x_1320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1320_, 0, v___x_1319_);
lean_ctor_set(v___x_1320_, 1, v___f_1317_);
v___x_1321_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1321_, 0, v_inst_1297_);
lean_ctor_set(v___x_1321_, 1, v_inst_1298_);
v___x_1322_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1322_, 0, v___x_1320_);
lean_ctor_set(v___x_1322_, 1, v_inst_1295_);
lean_ctor_set(v___x_1322_, 2, v_inst_1296_);
lean_ctor_set(v___x_1322_, 3, v___x_1321_);
v___x_1323_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1323_, 0, v___x_1322_);
lean_ctor_set(v___x_1323_, 1, v_inst_1299_);
lean_ctor_set(v___x_1323_, 2, v_inst_1301_);
lean_ctor_set(v___x_1323_, 3, v_inst_1300_);
return v___x_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeBooleanAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_1324_ = _args[0];
lean_object* v_00_u03b2_1325_ = _args[1];
lean_object* v_inst_1326_ = _args[2];
lean_object* v_inst_1327_ = _args[3];
lean_object* v_inst_1328_ = _args[4];
lean_object* v_inst_1329_ = _args[5];
lean_object* v_inst_1330_ = _args[6];
lean_object* v_inst_1331_ = _args[7];
lean_object* v_inst_1332_ = _args[8];
lean_object* v_inst_1333_ = _args[9];
lean_object* v_inst_1334_ = _args[10];
lean_object* v_inst_1335_ = _args[11];
lean_object* v_inst_1336_ = _args[12];
lean_object* v_inst_1337_ = _args[13];
lean_object* v_f_1338_ = _args[14];
lean_object* v_hf_1339_ = _args[15];
lean_object* v_le_1340_ = _args[16];
lean_object* v_lt_1341_ = _args[17];
lean_object* v_map__sup_1342_ = _args[18];
lean_object* v_map__inf_1343_ = _args[19];
lean_object* v_map__sSup_1344_ = _args[20];
lean_object* v_map__sInf_1345_ = _args[21];
lean_object* v_map__top_1346_ = _args[22];
lean_object* v_map__bot_1347_ = _args[23];
lean_object* v_map__compl_1348_ = _args[24];
lean_object* v_map__himp_1349_ = _args[25];
lean_object* v_map__sdiff_1350_ = _args[26];
_start:
{
lean_object* v_res_1351_; 
v_res_1351_ = lp_mathlib_Function_Injective_completeBooleanAlgebra(v_00_u03b1_1324_, v_00_u03b2_1325_, v_inst_1326_, v_inst_1327_, v_inst_1328_, v_inst_1329_, v_inst_1330_, v_inst_1331_, v_inst_1332_, v_inst_1333_, v_inst_1334_, v_inst_1335_, v_inst_1336_, v_inst_1337_, v_f_1338_, v_hf_1339_, v_le_1340_, v_lt_1341_, v_map__sup_1342_, v_map__inf_1343_, v_map__sSup_1344_, v_map__sInf_1345_, v_map__top_1346_, v_map__bot_1347_, v_map__compl_1348_, v_map__himp_1349_, v_map__sdiff_1350_);
lean_dec(v_f_1338_);
lean_dec_ref(v_inst_1337_);
return v_res_1351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeAtomicBooleanAlgebra___redArg(lean_object* v_inst_1352_, lean_object* v_inst_1353_, lean_object* v_inst_1354_, lean_object* v_inst_1355_, lean_object* v_inst_1356_, lean_object* v_inst_1357_, lean_object* v_inst_1358_, lean_object* v_inst_1359_, lean_object* v_inst_1360_, lean_object* v_inst_1361_, lean_object* v_inst_1362_, lean_object* v_inst_1363_){
_start:
{
lean_object* v___f_1364_; lean_object* v___f_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v_toGeneralizedCoheytingAlgebra_1372_; lean_object* v_toSDiff_1373_; lean_object* v___x_1374_; 
lean_inc(v_inst_1352_);
v___f_1364_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1364_, 0, v_inst_1352_);
lean_inc(v_inst_1353_);
v___f_1365_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1365_, 0, v_inst_1353_);
v___x_1366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1366_, 0, v_inst_1354_);
lean_ctor_set(v___x_1366_, 1, v_inst_1355_);
v___x_1367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1367_, 0, v___x_1366_);
lean_ctor_set(v___x_1367_, 1, v___f_1364_);
v___x_1368_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1368_, 0, v___x_1367_);
lean_ctor_set(v___x_1368_, 1, v___f_1365_);
lean_inc(v_inst_1359_);
lean_inc(v_inst_1358_);
v___x_1369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1369_, 0, v_inst_1358_);
lean_ctor_set(v___x_1369_, 1, v_inst_1359_);
v___x_1370_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1370_, 0, v___x_1368_);
lean_ctor_set(v___x_1370_, 1, v_inst_1356_);
lean_ctor_set(v___x_1370_, 2, v_inst_1357_);
lean_ctor_set(v___x_1370_, 3, v___x_1369_);
v___x_1371_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_1353_, v_inst_1352_, v_inst_1354_, v_inst_1355_, v_inst_1359_, v_inst_1358_, v_inst_1362_, v_inst_1363_);
v_toGeneralizedCoheytingAlgebra_1372_ = lean_ctor_get(v___x_1371_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1372_);
lean_dec_ref(v___x_1371_);
v_toSDiff_1373_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1372_, 2);
lean_inc(v_toSDiff_1373_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1372_);
v___x_1374_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1374_, 0, v___x_1370_);
lean_ctor_set(v___x_1374_, 1, v_inst_1360_);
lean_ctor_set(v___x_1374_, 2, v_toSDiff_1373_);
lean_ctor_set(v___x_1374_, 3, v_inst_1361_);
return v___x_1374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeAtomicBooleanAlgebra(lean_object* v_00_u03b1_1375_, lean_object* v_00_u03b2_1376_, lean_object* v_inst_1377_, lean_object* v_inst_1378_, lean_object* v_inst_1379_, lean_object* v_inst_1380_, lean_object* v_inst_1381_, lean_object* v_inst_1382_, lean_object* v_inst_1383_, lean_object* v_inst_1384_, lean_object* v_inst_1385_, lean_object* v_inst_1386_, lean_object* v_inst_1387_, lean_object* v_inst_1388_, lean_object* v_inst_1389_, lean_object* v_f_1390_, lean_object* v_hf_1391_, lean_object* v_le_1392_, lean_object* v_lt_1393_, lean_object* v_map__sup_1394_, lean_object* v_map__inf_1395_, lean_object* v_map__sSup_1396_, lean_object* v_map__sInf_1397_, lean_object* v_map__top_1398_, lean_object* v_map__bot_1399_, lean_object* v_map__compl_1400_, lean_object* v_map__himp_1401_, lean_object* v_map__hnot_1402_, lean_object* v_map__sdiff_1403_){
_start:
{
lean_object* v___f_1404_; lean_object* v___f_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v_toGeneralizedCoheytingAlgebra_1412_; lean_object* v_toSDiff_1413_; lean_object* v___x_1414_; 
lean_inc(v_inst_1377_);
v___f_1404_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1404_, 0, v_inst_1377_);
lean_inc(v_inst_1378_);
v___f_1405_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_frame___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1405_, 0, v_inst_1378_);
v___x_1406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1406_, 0, v_inst_1379_);
lean_ctor_set(v___x_1406_, 1, v_inst_1380_);
v___x_1407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1407_, 0, v___x_1406_);
lean_ctor_set(v___x_1407_, 1, v___f_1404_);
v___x_1408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1408_, 0, v___x_1407_);
lean_ctor_set(v___x_1408_, 1, v___f_1405_);
lean_inc(v_inst_1384_);
lean_inc(v_inst_1383_);
v___x_1409_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1409_, 0, v_inst_1383_);
lean_ctor_set(v___x_1409_, 1, v_inst_1384_);
v___x_1410_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1410_, 0, v___x_1408_);
lean_ctor_set(v___x_1410_, 1, v_inst_1381_);
lean_ctor_set(v___x_1410_, 2, v_inst_1382_);
lean_ctor_set(v___x_1410_, 3, v___x_1409_);
v___x_1411_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_1378_, v_inst_1377_, v_inst_1379_, v_inst_1380_, v_inst_1384_, v_inst_1383_, v_inst_1387_, v_inst_1388_);
v_toGeneralizedCoheytingAlgebra_1412_ = lean_ctor_get(v___x_1411_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1412_);
lean_dec_ref(v___x_1411_);
v_toSDiff_1413_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1412_, 2);
lean_inc(v_toSDiff_1413_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1412_);
v___x_1414_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1414_, 0, v___x_1410_);
lean_ctor_set(v___x_1414_, 1, v_inst_1385_);
lean_ctor_set(v___x_1414_, 2, v_toSDiff_1413_);
lean_ctor_set(v___x_1414_, 3, v_inst_1386_);
return v___x_1414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeAtomicBooleanAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_1415_ = _args[0];
lean_object* v_00_u03b2_1416_ = _args[1];
lean_object* v_inst_1417_ = _args[2];
lean_object* v_inst_1418_ = _args[3];
lean_object* v_inst_1419_ = _args[4];
lean_object* v_inst_1420_ = _args[5];
lean_object* v_inst_1421_ = _args[6];
lean_object* v_inst_1422_ = _args[7];
lean_object* v_inst_1423_ = _args[8];
lean_object* v_inst_1424_ = _args[9];
lean_object* v_inst_1425_ = _args[10];
lean_object* v_inst_1426_ = _args[11];
lean_object* v_inst_1427_ = _args[12];
lean_object* v_inst_1428_ = _args[13];
lean_object* v_inst_1429_ = _args[14];
lean_object* v_f_1430_ = _args[15];
lean_object* v_hf_1431_ = _args[16];
lean_object* v_le_1432_ = _args[17];
lean_object* v_lt_1433_ = _args[18];
lean_object* v_map__sup_1434_ = _args[19];
lean_object* v_map__inf_1435_ = _args[20];
lean_object* v_map__sSup_1436_ = _args[21];
lean_object* v_map__sInf_1437_ = _args[22];
lean_object* v_map__top_1438_ = _args[23];
lean_object* v_map__bot_1439_ = _args[24];
lean_object* v_map__compl_1440_ = _args[25];
lean_object* v_map__himp_1441_ = _args[26];
lean_object* v_map__hnot_1442_ = _args[27];
lean_object* v_map__sdiff_1443_ = _args[28];
_start:
{
lean_object* v_res_1444_; 
v_res_1444_ = lp_mathlib_Function_Injective_completeAtomicBooleanAlgebra(v_00_u03b1_1415_, v_00_u03b2_1416_, v_inst_1417_, v_inst_1418_, v_inst_1419_, v_inst_1420_, v_inst_1421_, v_inst_1422_, v_inst_1423_, v_inst_1424_, v_inst_1425_, v_inst_1426_, v_inst_1427_, v_inst_1428_, v_inst_1429_, v_f_1430_, v_hf_1431_, v_le_1432_, v_lt_1433_, v_map__sup_1434_, v_map__inf_1435_, v_map__sSup_1436_, v_map__sInf_1437_, v_map__top_1438_, v_map__bot_1439_, v_map__compl_1440_, v_map__himp_1441_, v_map__hnot_1442_, v_map__sdiff_1443_);
lean_dec(v_f_1430_);
lean_dec_ref(v_inst_1429_);
return v_res_1444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__0(lean_object* v_self_1445_, lean_object* v___y_1446_){
_start:
{
lean_object* v_toFun_1447_; lean_object* v___x_1448_; 
v_toFun_1447_ = lean_ctor_get(v_self_1445_, 0);
lean_inc(v_toFun_1447_);
lean_dec_ref(v_self_1445_);
v___x_1448_ = lean_apply_1(v_toFun_1447_, v___y_1446_);
return v___x_1448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__1(lean_object* v___f_1449_, lean_object* v_e_1450_, lean_object* v_inf_1451_, lean_object* v_toFun_1452_, lean_object* v_a_1453_, lean_object* v_b_1454_){
_start:
{
lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; 
lean_inc(v___f_1449_);
lean_inc_ref(v_e_1450_);
v___x_1455_ = lean_apply_2(v___f_1449_, v_e_1450_, v_a_1453_);
v___x_1456_ = lean_apply_2(v___f_1449_, v_e_1450_, v_b_1454_);
v___x_1457_ = lean_apply_2(v_inf_1451_, v___x_1455_, v___x_1456_);
v___x_1458_ = lean_apply_1(v_toFun_1452_, v___x_1457_);
return v___x_1458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__3(lean_object* v_min_1459_, lean_object* v_a_1460_, lean_object* v_b_1461_){
_start:
{
lean_object* v___x_1462_; 
v___x_1462_ = lean_apply_2(v_min_1459_, v_a_1460_, v_b_1461_);
return v___x_1462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__2(lean_object* v_toSemilatticeSup_1463_, lean_object* v___f_1464_, lean_object* v_e_1465_, lean_object* v_toFun_1466_, lean_object* v_a_1467_, lean_object* v_b_1468_){
_start:
{
lean_object* v_sup_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; 
v_sup_1469_ = lean_ctor_get(v_toSemilatticeSup_1463_, 1);
lean_inc(v_sup_1469_);
lean_dec_ref(v_toSemilatticeSup_1463_);
lean_inc(v___f_1464_);
lean_inc_ref(v_e_1465_);
v___x_1470_ = lean_apply_2(v___f_1464_, v_e_1465_, v_a_1467_);
v___x_1471_ = lean_apply_2(v___f_1464_, v_e_1465_, v_b_1468_);
v___x_1472_ = lean_apply_2(v_sup_1469_, v___x_1470_, v___x_1471_);
v___x_1473_ = lean_apply_1(v_toFun_1466_, v___x_1472_);
return v___x_1473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__5(lean_object* v_toSupSet_1474_, lean_object* v_toFun_1475_, lean_object* v_s_1476_){
_start:
{
lean_object* v___x_1477_; lean_object* v___x_1478_; 
v___x_1477_ = lean_apply_1(v_toSupSet_1474_, lean_box(0));
v___x_1478_ = lean_apply_1(v_toFun_1475_, v___x_1477_);
return v___x_1478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__4(lean_object* v_toInfSet_1479_, lean_object* v_toFun_1480_, lean_object* v_s_1481_){
_start:
{
lean_object* v___x_1482_; lean_object* v___x_1483_; 
v___x_1482_ = lean_apply_1(v_toInfSet_1479_, lean_box(0));
v___x_1483_ = lean_apply_1(v_toFun_1480_, v___x_1482_);
return v___x_1483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__6(lean_object* v___f_1484_, lean_object* v_a_1485_, lean_object* v_b_1486_){
_start:
{
lean_object* v___x_1487_; 
v___x_1487_ = lean_apply_2(v___f_1484_, v_a_1485_, v_b_1486_);
return v___x_1487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__10(lean_object* v_e_1488_, lean_object* v_toCompl_1489_, lean_object* v_toFun_1490_, lean_object* v_a_1491_){
_start:
{
lean_object* v_toFun_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v_toFun_1492_ = lean_ctor_get(v_e_1488_, 0);
lean_inc(v_toFun_1492_);
lean_dec_ref(v_e_1488_);
v___x_1493_ = lean_apply_1(v_toFun_1492_, v_a_1491_);
v___x_1494_ = lean_apply_1(v_toCompl_1489_, v___x_1493_);
v___x_1495_ = lean_apply_1(v_toFun_1490_, v___x_1494_);
return v___x_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg___lam__7(lean_object* v___f_1496_, lean_object* v_e_1497_, lean_object* v_toHImp_1498_, lean_object* v_toFun_1499_, lean_object* v_a_1500_, lean_object* v_b_1501_){
_start:
{
lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; 
lean_inc(v___f_1496_);
lean_inc_ref(v_e_1497_);
v___x_1502_ = lean_apply_2(v___f_1496_, v_e_1497_, v_a_1500_);
v___x_1503_ = lean_apply_2(v___f_1496_, v_e_1497_, v_b_1501_);
v___x_1504_ = lean_apply_2(v_toHImp_1498_, v___x_1502_, v___x_1503_);
v___x_1505_ = lean_apply_1(v_toFun_1499_, v___x_1504_);
return v___x_1505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame___redArg(lean_object* v_e_1507_, lean_object* v_inst_1508_){
_start:
{
lean_object* v_toCompleteLattice_1509_; lean_object* v_toBoundedOrder_1510_; lean_object* v_toLattice_1511_; lean_object* v_toOrderTop_1512_; lean_object* v_toOrderBot_1513_; lean_object* v___x_1515_; uint8_t v_isShared_1516_; uint8_t v_isSharedCheck_1648_; 
v_toCompleteLattice_1509_ = lean_ctor_get(v_inst_1508_, 0);
v_toBoundedOrder_1510_ = lean_ctor_get(v_toCompleteLattice_1509_, 3);
lean_inc_ref(v_toBoundedOrder_1510_);
v_toLattice_1511_ = lean_ctor_get(v_toCompleteLattice_1509_, 0);
lean_inc_ref(v_toLattice_1511_);
v_toOrderTop_1512_ = lean_ctor_get(v_toBoundedOrder_1510_, 0);
v_toOrderBot_1513_ = lean_ctor_get(v_toBoundedOrder_1510_, 1);
v_isSharedCheck_1648_ = !lean_is_exclusive(v_toBoundedOrder_1510_);
if (v_isSharedCheck_1648_ == 0)
{
v___x_1515_ = v_toBoundedOrder_1510_;
v_isShared_1516_ = v_isSharedCheck_1648_;
goto v_resetjp_1514_;
}
else
{
lean_inc(v_toOrderBot_1513_);
lean_inc(v_toOrderTop_1512_);
lean_dec(v_toBoundedOrder_1510_);
v___x_1515_ = lean_box(0);
v_isShared_1516_ = v_isSharedCheck_1648_;
goto v_resetjp_1514_;
}
v_resetjp_1514_:
{
lean_object* v___x_1517_; lean_object* v_toFun_1518_; lean_object* v___x_1519_; lean_object* v_toSupSet_1520_; lean_object* v___x_1522_; uint8_t v_isShared_1523_; uint8_t v_isSharedCheck_1646_; 
lean_inc_ref(v_e_1507_);
v___x_1517_ = lp_mathlib_Equiv_symm___redArg(v_e_1507_);
v_toFun_1518_ = lean_ctor_get(v___x_1517_, 0);
lean_inc(v_toFun_1518_);
lean_dec_ref(v___x_1517_);
lean_inc_ref(v_toCompleteLattice_1509_);
v___x_1519_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_1509_);
v_toSupSet_1520_ = lean_ctor_get(v___x_1519_, 1);
v_isSharedCheck_1646_ = !lean_is_exclusive(v___x_1519_);
if (v_isSharedCheck_1646_ == 0)
{
lean_object* v_unused_1647_; 
v_unused_1647_ = lean_ctor_get(v___x_1519_, 0);
lean_dec(v_unused_1647_);
v___x_1522_ = v___x_1519_;
v_isShared_1523_ = v_isSharedCheck_1646_;
goto v_resetjp_1521_;
}
else
{
lean_inc(v_toSupSet_1520_);
lean_dec(v___x_1519_);
v___x_1522_ = lean_box(0);
v_isShared_1523_ = v_isSharedCheck_1646_;
goto v_resetjp_1521_;
}
v_resetjp_1521_:
{
lean_object* v___x_1524_; lean_object* v_toInfSet_1525_; lean_object* v___x_1527_; uint8_t v_isShared_1528_; uint8_t v_isSharedCheck_1644_; 
lean_inc_ref(v_toCompleteLattice_1509_);
v___x_1524_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_1509_);
v_toInfSet_1525_ = lean_ctor_get(v___x_1524_, 1);
v_isSharedCheck_1644_ = !lean_is_exclusive(v___x_1524_);
if (v_isSharedCheck_1644_ == 0)
{
lean_object* v_unused_1645_; 
v_unused_1645_ = lean_ctor_get(v___x_1524_, 0);
lean_dec(v_unused_1645_);
v___x_1527_ = v___x_1524_;
v_isShared_1528_ = v_isSharedCheck_1644_;
goto v_resetjp_1526_;
}
else
{
lean_inc(v_toInfSet_1525_);
lean_dec(v___x_1524_);
v___x_1527_ = lean_box(0);
v_isShared_1528_ = v_isSharedCheck_1644_;
goto v_resetjp_1526_;
}
v_resetjp_1526_:
{
lean_object* v_toSemilatticeSup_1529_; lean_object* v_inf_1530_; lean_object* v___x_1532_; uint8_t v_isShared_1533_; uint8_t v_isSharedCheck_1643_; 
v_toSemilatticeSup_1529_ = lean_ctor_get(v_toLattice_1511_, 0);
v_inf_1530_ = lean_ctor_get(v_toLattice_1511_, 1);
v_isSharedCheck_1643_ = !lean_is_exclusive(v_toLattice_1511_);
if (v_isSharedCheck_1643_ == 0)
{
v___x_1532_ = v_toLattice_1511_;
v_isShared_1533_ = v_isSharedCheck_1643_;
goto v_resetjp_1531_;
}
else
{
lean_inc(v_inf_1530_);
lean_inc(v_toSemilatticeSup_1529_);
lean_dec(v_toLattice_1511_);
v___x_1532_ = lean_box(0);
v_isShared_1533_ = v_isSharedCheck_1643_;
goto v_resetjp_1531_;
}
v_resetjp_1531_:
{
lean_object* v___f_1534_; lean_object* v_min_1535_; lean_object* v_le_1536_; lean_object* v_lt_1537_; lean_object* v_semilatticeInf_1538_; lean_object* v_toPartialOrder_1539_; lean_object* v___x_1541_; uint8_t v_isShared_1542_; uint8_t v_isSharedCheck_1641_; 
v___f_1534_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_1518_);
lean_inc_ref(v_e_1507_);
v_min_1535_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_1535_, 0, v___f_1534_);
lean_closure_set(v_min_1535_, 1, v_e_1507_);
lean_closure_set(v_min_1535_, 2, v_inf_1530_);
lean_closure_set(v_min_1535_, 3, v_toFun_1518_);
v_le_1536_ = lean_box(0);
v_lt_1537_ = lean_box(0);
lean_inc_ref(v_min_1535_);
v_semilatticeInf_1538_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1535_, v_le_1536_, v_lt_1537_);
v_toPartialOrder_1539_ = lean_ctor_get(v_semilatticeInf_1538_, 0);
v_isSharedCheck_1641_ = !lean_is_exclusive(v_semilatticeInf_1538_);
if (v_isSharedCheck_1641_ == 0)
{
lean_object* v_unused_1642_; 
v_unused_1642_ = lean_ctor_get(v_semilatticeInf_1538_, 1);
lean_dec(v_unused_1642_);
v___x_1541_ = v_semilatticeInf_1538_;
v_isShared_1542_ = v_isSharedCheck_1641_;
goto v_resetjp_1540_;
}
else
{
lean_inc(v_toPartialOrder_1539_);
lean_dec(v_semilatticeInf_1538_);
v___x_1541_ = lean_box(0);
v_isShared_1542_ = v_isSharedCheck_1641_;
goto v_resetjp_1540_;
}
v_resetjp_1540_:
{
lean_object* v_toLE_1543_; lean_object* v_toLT_1544_; lean_object* v___x_1546_; uint8_t v_isShared_1547_; uint8_t v_isSharedCheck_1640_; 
v_toLE_1543_ = lean_ctor_get(v_toPartialOrder_1539_, 0);
v_toLT_1544_ = lean_ctor_get(v_toPartialOrder_1539_, 1);
v_isSharedCheck_1640_ = !lean_is_exclusive(v_toPartialOrder_1539_);
if (v_isSharedCheck_1640_ == 0)
{
v___x_1546_ = v_toPartialOrder_1539_;
v_isShared_1547_ = v_isSharedCheck_1640_;
goto v_resetjp_1545_;
}
else
{
lean_inc(v_toLT_1544_);
lean_inc(v_toLE_1543_);
lean_dec(v_toPartialOrder_1539_);
v___x_1546_ = lean_box(0);
v_isShared_1547_ = v_isSharedCheck_1640_;
goto v_resetjp_1545_;
}
v_resetjp_1545_:
{
lean_object* v___f_1548_; lean_object* v___f_1549_; lean_object* v___x_1551_; 
v___f_1548_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_1548_, 0, v_min_1535_);
lean_inc(v_toFun_1518_);
lean_inc_ref(v_e_1507_);
v___f_1549_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_1549_, 0, v_toSemilatticeSup_1529_);
lean_closure_set(v___f_1549_, 1, v___f_1534_);
lean_closure_set(v___f_1549_, 2, v_e_1507_);
lean_closure_set(v___f_1549_, 3, v_toFun_1518_);
if (v_isShared_1547_ == 0)
{
v___x_1551_ = v___x_1546_;
goto v_reusejp_1550_;
}
else
{
lean_object* v_reuseFailAlloc_1639_; 
v_reuseFailAlloc_1639_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1639_, 0, v_toLE_1543_);
lean_ctor_set(v_reuseFailAlloc_1639_, 1, v_toLT_1544_);
v___x_1551_ = v_reuseFailAlloc_1639_;
goto v_reusejp_1550_;
}
v_reusejp_1550_:
{
lean_object* v___x_1553_; 
lean_inc_ref(v___f_1549_);
if (v_isShared_1542_ == 0)
{
lean_ctor_set(v___x_1541_, 1, v___f_1549_);
lean_ctor_set(v___x_1541_, 0, v___x_1551_);
v___x_1553_ = v___x_1541_;
goto v_reusejp_1552_;
}
else
{
lean_object* v_reuseFailAlloc_1638_; 
v_reuseFailAlloc_1638_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1638_, 0, v___x_1551_);
lean_ctor_set(v_reuseFailAlloc_1638_, 1, v___f_1549_);
v___x_1553_ = v_reuseFailAlloc_1638_;
goto v_reusejp_1552_;
}
v_reusejp_1552_:
{
lean_object* v_lattice_1555_; 
lean_inc_ref(v___f_1548_);
if (v_isShared_1533_ == 0)
{
lean_ctor_set(v___x_1532_, 1, v___f_1548_);
lean_ctor_set(v___x_1532_, 0, v___x_1553_);
v_lattice_1555_ = v___x_1532_;
goto v_reusejp_1554_;
}
else
{
lean_object* v_reuseFailAlloc_1637_; 
v_reuseFailAlloc_1637_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1637_, 0, v___x_1553_);
lean_ctor_set(v_reuseFailAlloc_1637_, 1, v___f_1548_);
v_lattice_1555_ = v_reuseFailAlloc_1637_;
goto v_reusejp_1554_;
}
v_reusejp_1554_:
{
lean_object* v___x_1556_; lean_object* v_toPartialOrder_1557_; lean_object* v___x_1559_; uint8_t v_isShared_1560_; uint8_t v_isSharedCheck_1635_; 
v___x_1556_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1555_);
v_toPartialOrder_1557_ = lean_ctor_get(v___x_1556_, 0);
v_isSharedCheck_1635_ = !lean_is_exclusive(v___x_1556_);
if (v_isSharedCheck_1635_ == 0)
{
lean_object* v_unused_1636_; 
v_unused_1636_ = lean_ctor_get(v___x_1556_, 1);
lean_dec(v_unused_1636_);
v___x_1559_ = v___x_1556_;
v_isShared_1560_ = v_isSharedCheck_1635_;
goto v_resetjp_1558_;
}
else
{
lean_inc(v_toPartialOrder_1557_);
lean_dec(v___x_1556_);
v___x_1559_ = lean_box(0);
v_isShared_1560_ = v_isSharedCheck_1635_;
goto v_resetjp_1558_;
}
v_resetjp_1558_:
{
lean_object* v_toLE_1561_; lean_object* v_toLT_1562_; lean_object* v___x_1564_; uint8_t v_isShared_1565_; uint8_t v_isSharedCheck_1634_; 
v_toLE_1561_ = lean_ctor_get(v_toPartialOrder_1557_, 0);
v_toLT_1562_ = lean_ctor_get(v_toPartialOrder_1557_, 1);
v_isSharedCheck_1634_ = !lean_is_exclusive(v_toPartialOrder_1557_);
if (v_isSharedCheck_1634_ == 0)
{
v___x_1564_ = v_toPartialOrder_1557_;
v_isShared_1565_ = v_isSharedCheck_1634_;
goto v_resetjp_1563_;
}
else
{
lean_inc(v_toLT_1562_);
lean_inc(v_toLE_1561_);
lean_dec(v_toPartialOrder_1557_);
v___x_1564_ = lean_box(0);
v_isShared_1565_ = v_isSharedCheck_1634_;
goto v_resetjp_1563_;
}
v_resetjp_1563_:
{
lean_object* v_top_1566_; lean_object* v_bot_1567_; lean_object* v_supSet_1568_; lean_object* v_infSet_1569_; lean_object* v___f_1570_; lean_object* v___x_1572_; 
lean_inc_n(v_toFun_1518_, 4);
v_top_1566_ = lean_apply_1(v_toFun_1518_, v_toOrderTop_1512_);
v_bot_1567_ = lean_apply_1(v_toFun_1518_, v_toOrderBot_1513_);
v_supSet_1568_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_1568_, 0, v_toSupSet_1520_);
lean_closure_set(v_supSet_1568_, 1, v_toFun_1518_);
v_infSet_1569_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_1569_, 0, v_toInfSet_1525_);
lean_closure_set(v_infSet_1569_, 1, v_toFun_1518_);
v___f_1570_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_1570_, 0, v___f_1549_);
if (v_isShared_1565_ == 0)
{
v___x_1572_ = v___x_1564_;
goto v_reusejp_1571_;
}
else
{
lean_object* v_reuseFailAlloc_1633_; 
v_reuseFailAlloc_1633_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1633_, 0, v_toLE_1561_);
lean_ctor_set(v_reuseFailAlloc_1633_, 1, v_toLT_1562_);
v___x_1572_ = v_reuseFailAlloc_1633_;
goto v_reusejp_1571_;
}
v_reusejp_1571_:
{
lean_object* v___x_1574_; 
lean_inc_ref(v___f_1570_);
if (v_isShared_1560_ == 0)
{
lean_ctor_set(v___x_1559_, 1, v___f_1570_);
lean_ctor_set(v___x_1559_, 0, v___x_1572_);
v___x_1574_ = v___x_1559_;
goto v_reusejp_1573_;
}
else
{
lean_object* v_reuseFailAlloc_1632_; 
v_reuseFailAlloc_1632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1632_, 0, v___x_1572_);
lean_ctor_set(v_reuseFailAlloc_1632_, 1, v___f_1570_);
v___x_1574_ = v_reuseFailAlloc_1632_;
goto v_reusejp_1573_;
}
v_reusejp_1573_:
{
lean_object* v___x_1576_; 
lean_inc_ref(v___f_1548_);
if (v_isShared_1528_ == 0)
{
lean_ctor_set(v___x_1527_, 1, v___f_1548_);
lean_ctor_set(v___x_1527_, 0, v___x_1574_);
v___x_1576_ = v___x_1527_;
goto v_reusejp_1575_;
}
else
{
lean_object* v_reuseFailAlloc_1631_; 
v_reuseFailAlloc_1631_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1631_, 0, v___x_1574_);
lean_ctor_set(v_reuseFailAlloc_1631_, 1, v___f_1548_);
v___x_1576_ = v_reuseFailAlloc_1631_;
goto v_reusejp_1575_;
}
v_reusejp_1575_:
{
lean_object* v___x_1578_; 
lean_inc(v_top_1566_);
if (v_isShared_1516_ == 0)
{
lean_ctor_set(v___x_1515_, 1, v_bot_1567_);
lean_ctor_set(v___x_1515_, 0, v_top_1566_);
v___x_1578_ = v___x_1515_;
goto v_reusejp_1577_;
}
else
{
lean_object* v_reuseFailAlloc_1630_; 
v_reuseFailAlloc_1630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1630_, 0, v_top_1566_);
lean_ctor_set(v_reuseFailAlloc_1630_, 1, v_bot_1567_);
v___x_1578_ = v_reuseFailAlloc_1630_;
goto v_reusejp_1577_;
}
v_reusejp_1577_:
{
lean_object* v_completeLattice_1579_; lean_object* v___x_1580_; lean_object* v_toGeneralizedHeytingAlgebra_1581_; lean_object* v_toOrderBot_1582_; lean_object* v_toCompl_1583_; lean_object* v_toHImp_1584_; lean_object* v___x_1586_; uint8_t v_isShared_1587_; uint8_t v_isSharedCheck_1627_; 
v_completeLattice_1579_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_1579_, 0, v___x_1576_);
lean_ctor_set(v_completeLattice_1579_, 1, v_supSet_1568_);
lean_ctor_set(v_completeLattice_1579_, 2, v_infSet_1569_);
lean_ctor_set(v_completeLattice_1579_, 3, v___x_1578_);
v___x_1580_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v_inst_1508_);
v_toGeneralizedHeytingAlgebra_1581_ = lean_ctor_get(v___x_1580_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_1581_);
v_toOrderBot_1582_ = lean_ctor_get(v___x_1580_, 1);
lean_inc(v_toOrderBot_1582_);
v_toCompl_1583_ = lean_ctor_get(v___x_1580_, 2);
lean_inc(v_toCompl_1583_);
lean_dec_ref(v___x_1580_);
v_toHImp_1584_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1581_, 2);
v_isSharedCheck_1627_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_1581_);
if (v_isSharedCheck_1627_ == 0)
{
lean_object* v_unused_1628_; lean_object* v_unused_1629_; 
v_unused_1628_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1581_, 1);
lean_dec(v_unused_1628_);
v_unused_1629_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1581_, 0);
lean_dec(v_unused_1629_);
v___x_1586_ = v_toGeneralizedHeytingAlgebra_1581_;
v_isShared_1587_ = v_isSharedCheck_1627_;
goto v_resetjp_1585_;
}
else
{
lean_inc(v_toHImp_1584_);
lean_dec(v_toGeneralizedHeytingAlgebra_1581_);
v___x_1586_ = lean_box(0);
v_isShared_1587_ = v_isSharedCheck_1627_;
goto v_resetjp_1585_;
}
v_resetjp_1585_:
{
lean_object* v___x_1588_; lean_object* v_toPartialOrder_1589_; lean_object* v_toInfSet_1590_; lean_object* v___x_1592_; uint8_t v_isShared_1593_; uint8_t v_isSharedCheck_1626_; 
lean_inc_ref(v_completeLattice_1579_);
v___x_1588_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_1579_);
v_toPartialOrder_1589_ = lean_ctor_get(v___x_1588_, 0);
v_toInfSet_1590_ = lean_ctor_get(v___x_1588_, 1);
v_isSharedCheck_1626_ = !lean_is_exclusive(v___x_1588_);
if (v_isSharedCheck_1626_ == 0)
{
v___x_1592_ = v___x_1588_;
v_isShared_1593_ = v_isSharedCheck_1626_;
goto v_resetjp_1591_;
}
else
{
lean_inc(v_toInfSet_1590_);
lean_inc(v_toPartialOrder_1589_);
lean_dec(v___x_1588_);
v___x_1592_ = lean_box(0);
v_isShared_1593_ = v_isSharedCheck_1626_;
goto v_resetjp_1591_;
}
v_resetjp_1591_:
{
lean_object* v_toLE_1594_; lean_object* v_toLT_1595_; lean_object* v___x_1597_; uint8_t v_isShared_1598_; uint8_t v_isSharedCheck_1625_; 
v_toLE_1594_ = lean_ctor_get(v_toPartialOrder_1589_, 0);
v_toLT_1595_ = lean_ctor_get(v_toPartialOrder_1589_, 1);
v_isSharedCheck_1625_ = !lean_is_exclusive(v_toPartialOrder_1589_);
if (v_isSharedCheck_1625_ == 0)
{
v___x_1597_ = v_toPartialOrder_1589_;
v_isShared_1598_ = v_isSharedCheck_1625_;
goto v_resetjp_1596_;
}
else
{
lean_inc(v_toLT_1595_);
lean_inc(v_toLE_1594_);
lean_dec(v_toPartialOrder_1589_);
v___x_1597_ = lean_box(0);
v_isShared_1598_ = v_isSharedCheck_1625_;
goto v_resetjp_1596_;
}
v_resetjp_1596_:
{
lean_object* v___x_1599_; lean_object* v_toSupSet_1600_; lean_object* v___x_1602_; uint8_t v_isShared_1603_; uint8_t v_isSharedCheck_1623_; 
v___x_1599_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_1579_);
v_toSupSet_1600_ = lean_ctor_get(v___x_1599_, 1);
v_isSharedCheck_1623_ = !lean_is_exclusive(v___x_1599_);
if (v_isSharedCheck_1623_ == 0)
{
lean_object* v_unused_1624_; 
v_unused_1624_ = lean_ctor_get(v___x_1599_, 0);
lean_dec(v_unused_1624_);
v___x_1602_ = v___x_1599_;
v_isShared_1603_ = v_isSharedCheck_1623_;
goto v_resetjp_1601_;
}
else
{
lean_inc(v_toSupSet_1600_);
lean_dec(v___x_1599_);
v___x_1602_ = lean_box(0);
v_isShared_1603_ = v_isSharedCheck_1623_;
goto v_resetjp_1601_;
}
v_resetjp_1601_:
{
lean_object* v_compl_1604_; lean_object* v_himp_1605_; lean_object* v_bot_1606_; lean_object* v___x_1608_; 
lean_inc_n(v_toFun_1518_, 2);
lean_inc_ref(v_e_1507_);
v_compl_1604_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_1604_, 0, v_e_1507_);
lean_closure_set(v_compl_1604_, 1, v_toCompl_1583_);
lean_closure_set(v_compl_1604_, 2, v_toFun_1518_);
v_himp_1605_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_1605_, 0, v___f_1534_);
lean_closure_set(v_himp_1605_, 1, v_e_1507_);
lean_closure_set(v_himp_1605_, 2, v_toHImp_1584_);
lean_closure_set(v_himp_1605_, 3, v_toFun_1518_);
v_bot_1606_ = lean_apply_1(v_toFun_1518_, v_toOrderBot_1582_);
if (v_isShared_1598_ == 0)
{
v___x_1608_ = v___x_1597_;
goto v_reusejp_1607_;
}
else
{
lean_object* v_reuseFailAlloc_1622_; 
v_reuseFailAlloc_1622_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1622_, 0, v_toLE_1594_);
lean_ctor_set(v_reuseFailAlloc_1622_, 1, v_toLT_1595_);
v___x_1608_ = v_reuseFailAlloc_1622_;
goto v_reusejp_1607_;
}
v_reusejp_1607_:
{
lean_object* v___x_1610_; 
if (v_isShared_1603_ == 0)
{
lean_ctor_set(v___x_1602_, 1, v___f_1570_);
lean_ctor_set(v___x_1602_, 0, v___x_1608_);
v___x_1610_ = v___x_1602_;
goto v_reusejp_1609_;
}
else
{
lean_object* v_reuseFailAlloc_1621_; 
v_reuseFailAlloc_1621_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1621_, 0, v___x_1608_);
lean_ctor_set(v_reuseFailAlloc_1621_, 1, v___f_1570_);
v___x_1610_ = v_reuseFailAlloc_1621_;
goto v_reusejp_1609_;
}
v_reusejp_1609_:
{
lean_object* v___x_1612_; 
if (v_isShared_1593_ == 0)
{
lean_ctor_set(v___x_1592_, 1, v___f_1548_);
lean_ctor_set(v___x_1592_, 0, v___x_1610_);
v___x_1612_ = v___x_1592_;
goto v_reusejp_1611_;
}
else
{
lean_object* v_reuseFailAlloc_1620_; 
v_reuseFailAlloc_1620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1620_, 0, v___x_1610_);
lean_ctor_set(v_reuseFailAlloc_1620_, 1, v___f_1548_);
v___x_1612_ = v_reuseFailAlloc_1620_;
goto v_reusejp_1611_;
}
v_reusejp_1611_:
{
lean_object* v___x_1614_; 
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 1, v_bot_1606_);
lean_ctor_set(v___x_1522_, 0, v_top_1566_);
v___x_1614_ = v___x_1522_;
goto v_reusejp_1613_;
}
else
{
lean_object* v_reuseFailAlloc_1619_; 
v_reuseFailAlloc_1619_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1619_, 0, v_top_1566_);
lean_ctor_set(v_reuseFailAlloc_1619_, 1, v_bot_1606_);
v___x_1614_ = v_reuseFailAlloc_1619_;
goto v_reusejp_1613_;
}
v_reusejp_1613_:
{
lean_object* v___x_1615_; lean_object* v___x_1617_; 
v___x_1615_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1615_, 0, v___x_1612_);
lean_ctor_set(v___x_1615_, 1, v_toSupSet_1600_);
lean_ctor_set(v___x_1615_, 2, v_toInfSet_1590_);
lean_ctor_set(v___x_1615_, 3, v___x_1614_);
if (v_isShared_1587_ == 0)
{
lean_ctor_set(v___x_1586_, 2, v_compl_1604_);
lean_ctor_set(v___x_1586_, 1, v_himp_1605_);
lean_ctor_set(v___x_1586_, 0, v___x_1615_);
v___x_1617_ = v___x_1586_;
goto v_reusejp_1616_;
}
else
{
lean_object* v_reuseFailAlloc_1618_; 
v_reuseFailAlloc_1618_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1618_, 0, v___x_1615_);
lean_ctor_set(v_reuseFailAlloc_1618_, 1, v_himp_1605_);
lean_ctor_set(v_reuseFailAlloc_1618_, 2, v_compl_1604_);
v___x_1617_ = v_reuseFailAlloc_1618_;
goto v_reusejp_1616_;
}
v_reusejp_1616_:
{
return v___x_1617_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_frame(lean_object* v_00_u03b1_1649_, lean_object* v_00_u03b2_1650_, lean_object* v_e_1651_, lean_object* v_inst_1652_){
_start:
{
lean_object* v_toCompleteLattice_1653_; lean_object* v_toBoundedOrder_1654_; lean_object* v_toLattice_1655_; lean_object* v_toOrderTop_1656_; lean_object* v_toOrderBot_1657_; lean_object* v___x_1659_; uint8_t v_isShared_1660_; uint8_t v_isSharedCheck_1792_; 
v_toCompleteLattice_1653_ = lean_ctor_get(v_inst_1652_, 0);
v_toBoundedOrder_1654_ = lean_ctor_get(v_toCompleteLattice_1653_, 3);
lean_inc_ref(v_toBoundedOrder_1654_);
v_toLattice_1655_ = lean_ctor_get(v_toCompleteLattice_1653_, 0);
lean_inc_ref(v_toLattice_1655_);
v_toOrderTop_1656_ = lean_ctor_get(v_toBoundedOrder_1654_, 0);
v_toOrderBot_1657_ = lean_ctor_get(v_toBoundedOrder_1654_, 1);
v_isSharedCheck_1792_ = !lean_is_exclusive(v_toBoundedOrder_1654_);
if (v_isSharedCheck_1792_ == 0)
{
v___x_1659_ = v_toBoundedOrder_1654_;
v_isShared_1660_ = v_isSharedCheck_1792_;
goto v_resetjp_1658_;
}
else
{
lean_inc(v_toOrderBot_1657_);
lean_inc(v_toOrderTop_1656_);
lean_dec(v_toBoundedOrder_1654_);
v___x_1659_ = lean_box(0);
v_isShared_1660_ = v_isSharedCheck_1792_;
goto v_resetjp_1658_;
}
v_resetjp_1658_:
{
lean_object* v___x_1661_; lean_object* v_toFun_1662_; lean_object* v___x_1663_; lean_object* v_toSupSet_1664_; lean_object* v___x_1666_; uint8_t v_isShared_1667_; uint8_t v_isSharedCheck_1790_; 
lean_inc_ref(v_e_1651_);
v___x_1661_ = lp_mathlib_Equiv_symm___redArg(v_e_1651_);
v_toFun_1662_ = lean_ctor_get(v___x_1661_, 0);
lean_inc(v_toFun_1662_);
lean_dec_ref(v___x_1661_);
lean_inc_ref(v_toCompleteLattice_1653_);
v___x_1663_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_1653_);
v_toSupSet_1664_ = lean_ctor_get(v___x_1663_, 1);
v_isSharedCheck_1790_ = !lean_is_exclusive(v___x_1663_);
if (v_isSharedCheck_1790_ == 0)
{
lean_object* v_unused_1791_; 
v_unused_1791_ = lean_ctor_get(v___x_1663_, 0);
lean_dec(v_unused_1791_);
v___x_1666_ = v___x_1663_;
v_isShared_1667_ = v_isSharedCheck_1790_;
goto v_resetjp_1665_;
}
else
{
lean_inc(v_toSupSet_1664_);
lean_dec(v___x_1663_);
v___x_1666_ = lean_box(0);
v_isShared_1667_ = v_isSharedCheck_1790_;
goto v_resetjp_1665_;
}
v_resetjp_1665_:
{
lean_object* v___x_1668_; lean_object* v_toInfSet_1669_; lean_object* v___x_1671_; uint8_t v_isShared_1672_; uint8_t v_isSharedCheck_1788_; 
lean_inc_ref(v_toCompleteLattice_1653_);
v___x_1668_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_1653_);
v_toInfSet_1669_ = lean_ctor_get(v___x_1668_, 1);
v_isSharedCheck_1788_ = !lean_is_exclusive(v___x_1668_);
if (v_isSharedCheck_1788_ == 0)
{
lean_object* v_unused_1789_; 
v_unused_1789_ = lean_ctor_get(v___x_1668_, 0);
lean_dec(v_unused_1789_);
v___x_1671_ = v___x_1668_;
v_isShared_1672_ = v_isSharedCheck_1788_;
goto v_resetjp_1670_;
}
else
{
lean_inc(v_toInfSet_1669_);
lean_dec(v___x_1668_);
v___x_1671_ = lean_box(0);
v_isShared_1672_ = v_isSharedCheck_1788_;
goto v_resetjp_1670_;
}
v_resetjp_1670_:
{
lean_object* v_toSemilatticeSup_1673_; lean_object* v_inf_1674_; lean_object* v___x_1676_; uint8_t v_isShared_1677_; uint8_t v_isSharedCheck_1787_; 
v_toSemilatticeSup_1673_ = lean_ctor_get(v_toLattice_1655_, 0);
v_inf_1674_ = lean_ctor_get(v_toLattice_1655_, 1);
v_isSharedCheck_1787_ = !lean_is_exclusive(v_toLattice_1655_);
if (v_isSharedCheck_1787_ == 0)
{
v___x_1676_ = v_toLattice_1655_;
v_isShared_1677_ = v_isSharedCheck_1787_;
goto v_resetjp_1675_;
}
else
{
lean_inc(v_inf_1674_);
lean_inc(v_toSemilatticeSup_1673_);
lean_dec(v_toLattice_1655_);
v___x_1676_ = lean_box(0);
v_isShared_1677_ = v_isSharedCheck_1787_;
goto v_resetjp_1675_;
}
v_resetjp_1675_:
{
lean_object* v___f_1678_; lean_object* v_min_1679_; lean_object* v_le_1680_; lean_object* v_lt_1681_; lean_object* v_semilatticeInf_1682_; lean_object* v_toPartialOrder_1683_; lean_object* v___x_1685_; uint8_t v_isShared_1686_; uint8_t v_isSharedCheck_1785_; 
v___f_1678_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_1662_);
lean_inc_ref(v_e_1651_);
v_min_1679_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_1679_, 0, v___f_1678_);
lean_closure_set(v_min_1679_, 1, v_e_1651_);
lean_closure_set(v_min_1679_, 2, v_inf_1674_);
lean_closure_set(v_min_1679_, 3, v_toFun_1662_);
v_le_1680_ = lean_box(0);
v_lt_1681_ = lean_box(0);
lean_inc_ref(v_min_1679_);
v_semilatticeInf_1682_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1679_, v_le_1680_, v_lt_1681_);
v_toPartialOrder_1683_ = lean_ctor_get(v_semilatticeInf_1682_, 0);
v_isSharedCheck_1785_ = !lean_is_exclusive(v_semilatticeInf_1682_);
if (v_isSharedCheck_1785_ == 0)
{
lean_object* v_unused_1786_; 
v_unused_1786_ = lean_ctor_get(v_semilatticeInf_1682_, 1);
lean_dec(v_unused_1786_);
v___x_1685_ = v_semilatticeInf_1682_;
v_isShared_1686_ = v_isSharedCheck_1785_;
goto v_resetjp_1684_;
}
else
{
lean_inc(v_toPartialOrder_1683_);
lean_dec(v_semilatticeInf_1682_);
v___x_1685_ = lean_box(0);
v_isShared_1686_ = v_isSharedCheck_1785_;
goto v_resetjp_1684_;
}
v_resetjp_1684_:
{
lean_object* v_toLE_1687_; lean_object* v_toLT_1688_; lean_object* v___x_1690_; uint8_t v_isShared_1691_; uint8_t v_isSharedCheck_1784_; 
v_toLE_1687_ = lean_ctor_get(v_toPartialOrder_1683_, 0);
v_toLT_1688_ = lean_ctor_get(v_toPartialOrder_1683_, 1);
v_isSharedCheck_1784_ = !lean_is_exclusive(v_toPartialOrder_1683_);
if (v_isSharedCheck_1784_ == 0)
{
v___x_1690_ = v_toPartialOrder_1683_;
v_isShared_1691_ = v_isSharedCheck_1784_;
goto v_resetjp_1689_;
}
else
{
lean_inc(v_toLT_1688_);
lean_inc(v_toLE_1687_);
lean_dec(v_toPartialOrder_1683_);
v___x_1690_ = lean_box(0);
v_isShared_1691_ = v_isSharedCheck_1784_;
goto v_resetjp_1689_;
}
v_resetjp_1689_:
{
lean_object* v___f_1692_; lean_object* v___f_1693_; lean_object* v___x_1695_; 
v___f_1692_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_1692_, 0, v_min_1679_);
lean_inc(v_toFun_1662_);
lean_inc_ref(v_e_1651_);
v___f_1693_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_1693_, 0, v_toSemilatticeSup_1673_);
lean_closure_set(v___f_1693_, 1, v___f_1678_);
lean_closure_set(v___f_1693_, 2, v_e_1651_);
lean_closure_set(v___f_1693_, 3, v_toFun_1662_);
if (v_isShared_1691_ == 0)
{
v___x_1695_ = v___x_1690_;
goto v_reusejp_1694_;
}
else
{
lean_object* v_reuseFailAlloc_1783_; 
v_reuseFailAlloc_1783_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1783_, 0, v_toLE_1687_);
lean_ctor_set(v_reuseFailAlloc_1783_, 1, v_toLT_1688_);
v___x_1695_ = v_reuseFailAlloc_1783_;
goto v_reusejp_1694_;
}
v_reusejp_1694_:
{
lean_object* v___x_1697_; 
lean_inc_ref(v___f_1693_);
if (v_isShared_1686_ == 0)
{
lean_ctor_set(v___x_1685_, 1, v___f_1693_);
lean_ctor_set(v___x_1685_, 0, v___x_1695_);
v___x_1697_ = v___x_1685_;
goto v_reusejp_1696_;
}
else
{
lean_object* v_reuseFailAlloc_1782_; 
v_reuseFailAlloc_1782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1782_, 0, v___x_1695_);
lean_ctor_set(v_reuseFailAlloc_1782_, 1, v___f_1693_);
v___x_1697_ = v_reuseFailAlloc_1782_;
goto v_reusejp_1696_;
}
v_reusejp_1696_:
{
lean_object* v_lattice_1699_; 
lean_inc_ref(v___f_1692_);
if (v_isShared_1677_ == 0)
{
lean_ctor_set(v___x_1676_, 1, v___f_1692_);
lean_ctor_set(v___x_1676_, 0, v___x_1697_);
v_lattice_1699_ = v___x_1676_;
goto v_reusejp_1698_;
}
else
{
lean_object* v_reuseFailAlloc_1781_; 
v_reuseFailAlloc_1781_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1781_, 0, v___x_1697_);
lean_ctor_set(v_reuseFailAlloc_1781_, 1, v___f_1692_);
v_lattice_1699_ = v_reuseFailAlloc_1781_;
goto v_reusejp_1698_;
}
v_reusejp_1698_:
{
lean_object* v___x_1700_; lean_object* v_toPartialOrder_1701_; lean_object* v___x_1703_; uint8_t v_isShared_1704_; uint8_t v_isSharedCheck_1779_; 
v___x_1700_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1699_);
v_toPartialOrder_1701_ = lean_ctor_get(v___x_1700_, 0);
v_isSharedCheck_1779_ = !lean_is_exclusive(v___x_1700_);
if (v_isSharedCheck_1779_ == 0)
{
lean_object* v_unused_1780_; 
v_unused_1780_ = lean_ctor_get(v___x_1700_, 1);
lean_dec(v_unused_1780_);
v___x_1703_ = v___x_1700_;
v_isShared_1704_ = v_isSharedCheck_1779_;
goto v_resetjp_1702_;
}
else
{
lean_inc(v_toPartialOrder_1701_);
lean_dec(v___x_1700_);
v___x_1703_ = lean_box(0);
v_isShared_1704_ = v_isSharedCheck_1779_;
goto v_resetjp_1702_;
}
v_resetjp_1702_:
{
lean_object* v_toLE_1705_; lean_object* v_toLT_1706_; lean_object* v___x_1708_; uint8_t v_isShared_1709_; uint8_t v_isSharedCheck_1778_; 
v_toLE_1705_ = lean_ctor_get(v_toPartialOrder_1701_, 0);
v_toLT_1706_ = lean_ctor_get(v_toPartialOrder_1701_, 1);
v_isSharedCheck_1778_ = !lean_is_exclusive(v_toPartialOrder_1701_);
if (v_isSharedCheck_1778_ == 0)
{
v___x_1708_ = v_toPartialOrder_1701_;
v_isShared_1709_ = v_isSharedCheck_1778_;
goto v_resetjp_1707_;
}
else
{
lean_inc(v_toLT_1706_);
lean_inc(v_toLE_1705_);
lean_dec(v_toPartialOrder_1701_);
v___x_1708_ = lean_box(0);
v_isShared_1709_ = v_isSharedCheck_1778_;
goto v_resetjp_1707_;
}
v_resetjp_1707_:
{
lean_object* v_top_1710_; lean_object* v_bot_1711_; lean_object* v_supSet_1712_; lean_object* v_infSet_1713_; lean_object* v___f_1714_; lean_object* v___x_1716_; 
lean_inc_n(v_toFun_1662_, 4);
v_top_1710_ = lean_apply_1(v_toFun_1662_, v_toOrderTop_1656_);
v_bot_1711_ = lean_apply_1(v_toFun_1662_, v_toOrderBot_1657_);
v_supSet_1712_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_1712_, 0, v_toSupSet_1664_);
lean_closure_set(v_supSet_1712_, 1, v_toFun_1662_);
v_infSet_1713_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_1713_, 0, v_toInfSet_1669_);
lean_closure_set(v_infSet_1713_, 1, v_toFun_1662_);
v___f_1714_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_1714_, 0, v___f_1693_);
if (v_isShared_1709_ == 0)
{
v___x_1716_ = v___x_1708_;
goto v_reusejp_1715_;
}
else
{
lean_object* v_reuseFailAlloc_1777_; 
v_reuseFailAlloc_1777_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1777_, 0, v_toLE_1705_);
lean_ctor_set(v_reuseFailAlloc_1777_, 1, v_toLT_1706_);
v___x_1716_ = v_reuseFailAlloc_1777_;
goto v_reusejp_1715_;
}
v_reusejp_1715_:
{
lean_object* v___x_1718_; 
lean_inc_ref(v___f_1714_);
if (v_isShared_1704_ == 0)
{
lean_ctor_set(v___x_1703_, 1, v___f_1714_);
lean_ctor_set(v___x_1703_, 0, v___x_1716_);
v___x_1718_ = v___x_1703_;
goto v_reusejp_1717_;
}
else
{
lean_object* v_reuseFailAlloc_1776_; 
v_reuseFailAlloc_1776_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1776_, 0, v___x_1716_);
lean_ctor_set(v_reuseFailAlloc_1776_, 1, v___f_1714_);
v___x_1718_ = v_reuseFailAlloc_1776_;
goto v_reusejp_1717_;
}
v_reusejp_1717_:
{
lean_object* v___x_1720_; 
lean_inc_ref(v___f_1692_);
if (v_isShared_1672_ == 0)
{
lean_ctor_set(v___x_1671_, 1, v___f_1692_);
lean_ctor_set(v___x_1671_, 0, v___x_1718_);
v___x_1720_ = v___x_1671_;
goto v_reusejp_1719_;
}
else
{
lean_object* v_reuseFailAlloc_1775_; 
v_reuseFailAlloc_1775_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1775_, 0, v___x_1718_);
lean_ctor_set(v_reuseFailAlloc_1775_, 1, v___f_1692_);
v___x_1720_ = v_reuseFailAlloc_1775_;
goto v_reusejp_1719_;
}
v_reusejp_1719_:
{
lean_object* v___x_1722_; 
lean_inc(v_top_1710_);
if (v_isShared_1660_ == 0)
{
lean_ctor_set(v___x_1659_, 1, v_bot_1711_);
lean_ctor_set(v___x_1659_, 0, v_top_1710_);
v___x_1722_ = v___x_1659_;
goto v_reusejp_1721_;
}
else
{
lean_object* v_reuseFailAlloc_1774_; 
v_reuseFailAlloc_1774_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1774_, 0, v_top_1710_);
lean_ctor_set(v_reuseFailAlloc_1774_, 1, v_bot_1711_);
v___x_1722_ = v_reuseFailAlloc_1774_;
goto v_reusejp_1721_;
}
v_reusejp_1721_:
{
lean_object* v_completeLattice_1723_; lean_object* v___x_1724_; lean_object* v_toGeneralizedHeytingAlgebra_1725_; lean_object* v_toOrderBot_1726_; lean_object* v_toCompl_1727_; lean_object* v_toHImp_1728_; lean_object* v___x_1730_; uint8_t v_isShared_1731_; uint8_t v_isSharedCheck_1771_; 
v_completeLattice_1723_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_1723_, 0, v___x_1720_);
lean_ctor_set(v_completeLattice_1723_, 1, v_supSet_1712_);
lean_ctor_set(v_completeLattice_1723_, 2, v_infSet_1713_);
lean_ctor_set(v_completeLattice_1723_, 3, v___x_1722_);
v___x_1724_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v_inst_1652_);
v_toGeneralizedHeytingAlgebra_1725_ = lean_ctor_get(v___x_1724_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_1725_);
v_toOrderBot_1726_ = lean_ctor_get(v___x_1724_, 1);
lean_inc(v_toOrderBot_1726_);
v_toCompl_1727_ = lean_ctor_get(v___x_1724_, 2);
lean_inc(v_toCompl_1727_);
lean_dec_ref(v___x_1724_);
v_toHImp_1728_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1725_, 2);
v_isSharedCheck_1771_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_1725_);
if (v_isSharedCheck_1771_ == 0)
{
lean_object* v_unused_1772_; lean_object* v_unused_1773_; 
v_unused_1772_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1725_, 1);
lean_dec(v_unused_1772_);
v_unused_1773_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1725_, 0);
lean_dec(v_unused_1773_);
v___x_1730_ = v_toGeneralizedHeytingAlgebra_1725_;
v_isShared_1731_ = v_isSharedCheck_1771_;
goto v_resetjp_1729_;
}
else
{
lean_inc(v_toHImp_1728_);
lean_dec(v_toGeneralizedHeytingAlgebra_1725_);
v___x_1730_ = lean_box(0);
v_isShared_1731_ = v_isSharedCheck_1771_;
goto v_resetjp_1729_;
}
v_resetjp_1729_:
{
lean_object* v___x_1732_; lean_object* v_toPartialOrder_1733_; lean_object* v_toInfSet_1734_; lean_object* v___x_1736_; uint8_t v_isShared_1737_; uint8_t v_isSharedCheck_1770_; 
lean_inc_ref(v_completeLattice_1723_);
v___x_1732_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_1723_);
v_toPartialOrder_1733_ = lean_ctor_get(v___x_1732_, 0);
v_toInfSet_1734_ = lean_ctor_get(v___x_1732_, 1);
v_isSharedCheck_1770_ = !lean_is_exclusive(v___x_1732_);
if (v_isSharedCheck_1770_ == 0)
{
v___x_1736_ = v___x_1732_;
v_isShared_1737_ = v_isSharedCheck_1770_;
goto v_resetjp_1735_;
}
else
{
lean_inc(v_toInfSet_1734_);
lean_inc(v_toPartialOrder_1733_);
lean_dec(v___x_1732_);
v___x_1736_ = lean_box(0);
v_isShared_1737_ = v_isSharedCheck_1770_;
goto v_resetjp_1735_;
}
v_resetjp_1735_:
{
lean_object* v_toLE_1738_; lean_object* v_toLT_1739_; lean_object* v___x_1741_; uint8_t v_isShared_1742_; uint8_t v_isSharedCheck_1769_; 
v_toLE_1738_ = lean_ctor_get(v_toPartialOrder_1733_, 0);
v_toLT_1739_ = lean_ctor_get(v_toPartialOrder_1733_, 1);
v_isSharedCheck_1769_ = !lean_is_exclusive(v_toPartialOrder_1733_);
if (v_isSharedCheck_1769_ == 0)
{
v___x_1741_ = v_toPartialOrder_1733_;
v_isShared_1742_ = v_isSharedCheck_1769_;
goto v_resetjp_1740_;
}
else
{
lean_inc(v_toLT_1739_);
lean_inc(v_toLE_1738_);
lean_dec(v_toPartialOrder_1733_);
v___x_1741_ = lean_box(0);
v_isShared_1742_ = v_isSharedCheck_1769_;
goto v_resetjp_1740_;
}
v_resetjp_1740_:
{
lean_object* v___x_1743_; lean_object* v_toSupSet_1744_; lean_object* v___x_1746_; uint8_t v_isShared_1747_; uint8_t v_isSharedCheck_1767_; 
v___x_1743_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_1723_);
v_toSupSet_1744_ = lean_ctor_get(v___x_1743_, 1);
v_isSharedCheck_1767_ = !lean_is_exclusive(v___x_1743_);
if (v_isSharedCheck_1767_ == 0)
{
lean_object* v_unused_1768_; 
v_unused_1768_ = lean_ctor_get(v___x_1743_, 0);
lean_dec(v_unused_1768_);
v___x_1746_ = v___x_1743_;
v_isShared_1747_ = v_isSharedCheck_1767_;
goto v_resetjp_1745_;
}
else
{
lean_inc(v_toSupSet_1744_);
lean_dec(v___x_1743_);
v___x_1746_ = lean_box(0);
v_isShared_1747_ = v_isSharedCheck_1767_;
goto v_resetjp_1745_;
}
v_resetjp_1745_:
{
lean_object* v_compl_1748_; lean_object* v_himp_1749_; lean_object* v_bot_1750_; lean_object* v___x_1752_; 
lean_inc_n(v_toFun_1662_, 2);
lean_inc_ref(v_e_1651_);
v_compl_1748_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_1748_, 0, v_e_1651_);
lean_closure_set(v_compl_1748_, 1, v_toCompl_1727_);
lean_closure_set(v_compl_1748_, 2, v_toFun_1662_);
v_himp_1749_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_1749_, 0, v___f_1678_);
lean_closure_set(v_himp_1749_, 1, v_e_1651_);
lean_closure_set(v_himp_1749_, 2, v_toHImp_1728_);
lean_closure_set(v_himp_1749_, 3, v_toFun_1662_);
v_bot_1750_ = lean_apply_1(v_toFun_1662_, v_toOrderBot_1726_);
if (v_isShared_1742_ == 0)
{
v___x_1752_ = v___x_1741_;
goto v_reusejp_1751_;
}
else
{
lean_object* v_reuseFailAlloc_1766_; 
v_reuseFailAlloc_1766_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1766_, 0, v_toLE_1738_);
lean_ctor_set(v_reuseFailAlloc_1766_, 1, v_toLT_1739_);
v___x_1752_ = v_reuseFailAlloc_1766_;
goto v_reusejp_1751_;
}
v_reusejp_1751_:
{
lean_object* v___x_1754_; 
if (v_isShared_1747_ == 0)
{
lean_ctor_set(v___x_1746_, 1, v___f_1714_);
lean_ctor_set(v___x_1746_, 0, v___x_1752_);
v___x_1754_ = v___x_1746_;
goto v_reusejp_1753_;
}
else
{
lean_object* v_reuseFailAlloc_1765_; 
v_reuseFailAlloc_1765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1765_, 0, v___x_1752_);
lean_ctor_set(v_reuseFailAlloc_1765_, 1, v___f_1714_);
v___x_1754_ = v_reuseFailAlloc_1765_;
goto v_reusejp_1753_;
}
v_reusejp_1753_:
{
lean_object* v___x_1756_; 
if (v_isShared_1737_ == 0)
{
lean_ctor_set(v___x_1736_, 1, v___f_1692_);
lean_ctor_set(v___x_1736_, 0, v___x_1754_);
v___x_1756_ = v___x_1736_;
goto v_reusejp_1755_;
}
else
{
lean_object* v_reuseFailAlloc_1764_; 
v_reuseFailAlloc_1764_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1764_, 0, v___x_1754_);
lean_ctor_set(v_reuseFailAlloc_1764_, 1, v___f_1692_);
v___x_1756_ = v_reuseFailAlloc_1764_;
goto v_reusejp_1755_;
}
v_reusejp_1755_:
{
lean_object* v___x_1758_; 
if (v_isShared_1667_ == 0)
{
lean_ctor_set(v___x_1666_, 1, v_bot_1750_);
lean_ctor_set(v___x_1666_, 0, v_top_1710_);
v___x_1758_ = v___x_1666_;
goto v_reusejp_1757_;
}
else
{
lean_object* v_reuseFailAlloc_1763_; 
v_reuseFailAlloc_1763_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1763_, 0, v_top_1710_);
lean_ctor_set(v_reuseFailAlloc_1763_, 1, v_bot_1750_);
v___x_1758_ = v_reuseFailAlloc_1763_;
goto v_reusejp_1757_;
}
v_reusejp_1757_:
{
lean_object* v___x_1759_; lean_object* v___x_1761_; 
v___x_1759_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1759_, 0, v___x_1756_);
lean_ctor_set(v___x_1759_, 1, v_toSupSet_1744_);
lean_ctor_set(v___x_1759_, 2, v_toInfSet_1734_);
lean_ctor_set(v___x_1759_, 3, v___x_1758_);
if (v_isShared_1731_ == 0)
{
lean_ctor_set(v___x_1730_, 2, v_compl_1748_);
lean_ctor_set(v___x_1730_, 1, v_himp_1749_);
lean_ctor_set(v___x_1730_, 0, v___x_1759_);
v___x_1761_ = v___x_1730_;
goto v_reusejp_1760_;
}
else
{
lean_object* v_reuseFailAlloc_1762_; 
v_reuseFailAlloc_1762_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1762_, 0, v___x_1759_);
lean_ctor_set(v_reuseFailAlloc_1762_, 1, v_himp_1749_);
lean_ctor_set(v_reuseFailAlloc_1762_, 2, v_compl_1748_);
v___x_1761_ = v_reuseFailAlloc_1762_;
goto v_reusejp_1760_;
}
v_reusejp_1760_:
{
return v___x_1761_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coframe___redArg___lam__17(lean_object* v_e_1793_, lean_object* v_toHNot_1794_, lean_object* v_toFun_1795_, lean_object* v_a_1796_){
_start:
{
lean_object* v_toFun_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; 
v_toFun_1797_ = lean_ctor_get(v_e_1793_, 0);
lean_inc(v_toFun_1797_);
lean_dec_ref(v_e_1793_);
v___x_1798_ = lean_apply_1(v_toFun_1797_, v_a_1796_);
v___x_1799_ = lean_apply_1(v_toHNot_1794_, v___x_1798_);
v___x_1800_ = lean_apply_1(v_toFun_1795_, v___x_1799_);
return v___x_1800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coframe___redArg___lam__0(lean_object* v___f_1801_, lean_object* v_e_1802_, lean_object* v_toSDiff_1803_, lean_object* v_toFun_1804_, lean_object* v_a_1805_, lean_object* v_b_1806_){
_start:
{
lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; 
lean_inc(v___f_1801_);
lean_inc_ref(v_e_1802_);
v___x_1807_ = lean_apply_2(v___f_1801_, v_e_1802_, v_a_1805_);
v___x_1808_ = lean_apply_2(v___f_1801_, v_e_1802_, v_b_1806_);
v___x_1809_ = lean_apply_2(v_toSDiff_1803_, v___x_1807_, v___x_1808_);
v___x_1810_ = lean_apply_1(v_toFun_1804_, v___x_1809_);
return v___x_1810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coframe___redArg(lean_object* v_e_1811_, lean_object* v_inst_1812_){
_start:
{
lean_object* v_toCompleteLattice_1813_; lean_object* v_toBoundedOrder_1814_; lean_object* v_toLattice_1815_; lean_object* v_toOrderTop_1816_; lean_object* v_toOrderBot_1817_; lean_object* v___x_1819_; uint8_t v_isShared_1820_; uint8_t v_isSharedCheck_2031_; 
v_toCompleteLattice_1813_ = lean_ctor_get(v_inst_1812_, 0);
v_toBoundedOrder_1814_ = lean_ctor_get(v_toCompleteLattice_1813_, 3);
lean_inc_ref(v_toBoundedOrder_1814_);
v_toLattice_1815_ = lean_ctor_get(v_toCompleteLattice_1813_, 0);
lean_inc_ref(v_toLattice_1815_);
v_toOrderTop_1816_ = lean_ctor_get(v_toBoundedOrder_1814_, 0);
v_toOrderBot_1817_ = lean_ctor_get(v_toBoundedOrder_1814_, 1);
v_isSharedCheck_2031_ = !lean_is_exclusive(v_toBoundedOrder_1814_);
if (v_isSharedCheck_2031_ == 0)
{
v___x_1819_ = v_toBoundedOrder_1814_;
v_isShared_1820_ = v_isSharedCheck_2031_;
goto v_resetjp_1818_;
}
else
{
lean_inc(v_toOrderBot_1817_);
lean_inc(v_toOrderTop_1816_);
lean_dec(v_toBoundedOrder_1814_);
v___x_1819_ = lean_box(0);
v_isShared_1820_ = v_isSharedCheck_2031_;
goto v_resetjp_1818_;
}
v_resetjp_1818_:
{
lean_object* v___x_1821_; lean_object* v_toFun_1822_; lean_object* v___x_1823_; lean_object* v_toSupSet_1824_; lean_object* v___x_1826_; uint8_t v_isShared_1827_; uint8_t v_isSharedCheck_2029_; 
lean_inc_ref(v_e_1811_);
v___x_1821_ = lp_mathlib_Equiv_symm___redArg(v_e_1811_);
v_toFun_1822_ = lean_ctor_get(v___x_1821_, 0);
lean_inc(v_toFun_1822_);
lean_dec_ref(v___x_1821_);
lean_inc_ref(v_toCompleteLattice_1813_);
v___x_1823_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_1813_);
v_toSupSet_1824_ = lean_ctor_get(v___x_1823_, 1);
v_isSharedCheck_2029_ = !lean_is_exclusive(v___x_1823_);
if (v_isSharedCheck_2029_ == 0)
{
lean_object* v_unused_2030_; 
v_unused_2030_ = lean_ctor_get(v___x_1823_, 0);
lean_dec(v_unused_2030_);
v___x_1826_ = v___x_1823_;
v_isShared_1827_ = v_isSharedCheck_2029_;
goto v_resetjp_1825_;
}
else
{
lean_inc(v_toSupSet_1824_);
lean_dec(v___x_1823_);
v___x_1826_ = lean_box(0);
v_isShared_1827_ = v_isSharedCheck_2029_;
goto v_resetjp_1825_;
}
v_resetjp_1825_:
{
lean_object* v___x_1828_; lean_object* v_toInfSet_1829_; lean_object* v___x_1831_; uint8_t v_isShared_1832_; uint8_t v_isSharedCheck_2027_; 
lean_inc_ref(v_toCompleteLattice_1813_);
v___x_1828_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_1813_);
v_toInfSet_1829_ = lean_ctor_get(v___x_1828_, 1);
v_isSharedCheck_2027_ = !lean_is_exclusive(v___x_1828_);
if (v_isSharedCheck_2027_ == 0)
{
lean_object* v_unused_2028_; 
v_unused_2028_ = lean_ctor_get(v___x_1828_, 0);
lean_dec(v_unused_2028_);
v___x_1831_ = v___x_1828_;
v_isShared_1832_ = v_isSharedCheck_2027_;
goto v_resetjp_1830_;
}
else
{
lean_inc(v_toInfSet_1829_);
lean_dec(v___x_1828_);
v___x_1831_ = lean_box(0);
v_isShared_1832_ = v_isSharedCheck_2027_;
goto v_resetjp_1830_;
}
v_resetjp_1830_:
{
lean_object* v_toSemilatticeSup_1833_; lean_object* v_inf_1834_; lean_object* v___x_1836_; uint8_t v_isShared_1837_; uint8_t v_isSharedCheck_2026_; 
v_toSemilatticeSup_1833_ = lean_ctor_get(v_toLattice_1815_, 0);
v_inf_1834_ = lean_ctor_get(v_toLattice_1815_, 1);
v_isSharedCheck_2026_ = !lean_is_exclusive(v_toLattice_1815_);
if (v_isSharedCheck_2026_ == 0)
{
v___x_1836_ = v_toLattice_1815_;
v_isShared_1837_ = v_isSharedCheck_2026_;
goto v_resetjp_1835_;
}
else
{
lean_inc(v_inf_1834_);
lean_inc(v_toSemilatticeSup_1833_);
lean_dec(v_toLattice_1815_);
v___x_1836_ = lean_box(0);
v_isShared_1837_ = v_isSharedCheck_2026_;
goto v_resetjp_1835_;
}
v_resetjp_1835_:
{
lean_object* v___f_1838_; lean_object* v_min_1839_; lean_object* v_le_1840_; lean_object* v_lt_1841_; lean_object* v_semilatticeInf_1842_; lean_object* v_toPartialOrder_1843_; lean_object* v___x_1845_; uint8_t v_isShared_1846_; uint8_t v_isSharedCheck_2024_; 
v___f_1838_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_1822_);
lean_inc_ref(v_e_1811_);
v_min_1839_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_1839_, 0, v___f_1838_);
lean_closure_set(v_min_1839_, 1, v_e_1811_);
lean_closure_set(v_min_1839_, 2, v_inf_1834_);
lean_closure_set(v_min_1839_, 3, v_toFun_1822_);
v_le_1840_ = lean_box(0);
v_lt_1841_ = lean_box(0);
lean_inc_ref(v_min_1839_);
v_semilatticeInf_1842_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1839_, v_le_1840_, v_lt_1841_);
v_toPartialOrder_1843_ = lean_ctor_get(v_semilatticeInf_1842_, 0);
v_isSharedCheck_2024_ = !lean_is_exclusive(v_semilatticeInf_1842_);
if (v_isSharedCheck_2024_ == 0)
{
lean_object* v_unused_2025_; 
v_unused_2025_ = lean_ctor_get(v_semilatticeInf_1842_, 1);
lean_dec(v_unused_2025_);
v___x_1845_ = v_semilatticeInf_1842_;
v_isShared_1846_ = v_isSharedCheck_2024_;
goto v_resetjp_1844_;
}
else
{
lean_inc(v_toPartialOrder_1843_);
lean_dec(v_semilatticeInf_1842_);
v___x_1845_ = lean_box(0);
v_isShared_1846_ = v_isSharedCheck_2024_;
goto v_resetjp_1844_;
}
v_resetjp_1844_:
{
lean_object* v_toLE_1847_; lean_object* v_toLT_1848_; lean_object* v___x_1850_; uint8_t v_isShared_1851_; uint8_t v_isSharedCheck_2023_; 
v_toLE_1847_ = lean_ctor_get(v_toPartialOrder_1843_, 0);
v_toLT_1848_ = lean_ctor_get(v_toPartialOrder_1843_, 1);
v_isSharedCheck_2023_ = !lean_is_exclusive(v_toPartialOrder_1843_);
if (v_isSharedCheck_2023_ == 0)
{
v___x_1850_ = v_toPartialOrder_1843_;
v_isShared_1851_ = v_isSharedCheck_2023_;
goto v_resetjp_1849_;
}
else
{
lean_inc(v_toLT_1848_);
lean_inc(v_toLE_1847_);
lean_dec(v_toPartialOrder_1843_);
v___x_1850_ = lean_box(0);
v_isShared_1851_ = v_isSharedCheck_2023_;
goto v_resetjp_1849_;
}
v_resetjp_1849_:
{
lean_object* v___f_1852_; lean_object* v___f_1853_; lean_object* v___x_1855_; 
v___f_1852_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_1852_, 0, v_min_1839_);
lean_inc(v_toFun_1822_);
lean_inc_ref(v_e_1811_);
v___f_1853_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_1853_, 0, v_toSemilatticeSup_1833_);
lean_closure_set(v___f_1853_, 1, v___f_1838_);
lean_closure_set(v___f_1853_, 2, v_e_1811_);
lean_closure_set(v___f_1853_, 3, v_toFun_1822_);
if (v_isShared_1851_ == 0)
{
v___x_1855_ = v___x_1850_;
goto v_reusejp_1854_;
}
else
{
lean_object* v_reuseFailAlloc_2022_; 
v_reuseFailAlloc_2022_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2022_, 0, v_toLE_1847_);
lean_ctor_set(v_reuseFailAlloc_2022_, 1, v_toLT_1848_);
v___x_1855_ = v_reuseFailAlloc_2022_;
goto v_reusejp_1854_;
}
v_reusejp_1854_:
{
lean_object* v___x_1857_; 
lean_inc_ref(v___f_1853_);
if (v_isShared_1846_ == 0)
{
lean_ctor_set(v___x_1845_, 1, v___f_1853_);
lean_ctor_set(v___x_1845_, 0, v___x_1855_);
v___x_1857_ = v___x_1845_;
goto v_reusejp_1856_;
}
else
{
lean_object* v_reuseFailAlloc_2021_; 
v_reuseFailAlloc_2021_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2021_, 0, v___x_1855_);
lean_ctor_set(v_reuseFailAlloc_2021_, 1, v___f_1853_);
v___x_1857_ = v_reuseFailAlloc_2021_;
goto v_reusejp_1856_;
}
v_reusejp_1856_:
{
lean_object* v_lattice_1859_; 
lean_inc_ref(v___f_1852_);
if (v_isShared_1837_ == 0)
{
lean_ctor_set(v___x_1836_, 1, v___f_1852_);
lean_ctor_set(v___x_1836_, 0, v___x_1857_);
v_lattice_1859_ = v___x_1836_;
goto v_reusejp_1858_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v___x_1857_);
lean_ctor_set(v_reuseFailAlloc_2020_, 1, v___f_1852_);
v_lattice_1859_ = v_reuseFailAlloc_2020_;
goto v_reusejp_1858_;
}
v_reusejp_1858_:
{
lean_object* v___x_1860_; lean_object* v_toPartialOrder_1861_; lean_object* v___x_1863_; uint8_t v_isShared_1864_; uint8_t v_isSharedCheck_2018_; 
v___x_1860_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1859_);
v_toPartialOrder_1861_ = lean_ctor_get(v___x_1860_, 0);
v_isSharedCheck_2018_ = !lean_is_exclusive(v___x_1860_);
if (v_isSharedCheck_2018_ == 0)
{
lean_object* v_unused_2019_; 
v_unused_2019_ = lean_ctor_get(v___x_1860_, 1);
lean_dec(v_unused_2019_);
v___x_1863_ = v___x_1860_;
v_isShared_1864_ = v_isSharedCheck_2018_;
goto v_resetjp_1862_;
}
else
{
lean_inc(v_toPartialOrder_1861_);
lean_dec(v___x_1860_);
v___x_1863_ = lean_box(0);
v_isShared_1864_ = v_isSharedCheck_2018_;
goto v_resetjp_1862_;
}
v_resetjp_1862_:
{
lean_object* v_toLE_1865_; lean_object* v_toLT_1866_; lean_object* v___x_1868_; uint8_t v_isShared_1869_; uint8_t v_isSharedCheck_2017_; 
v_toLE_1865_ = lean_ctor_get(v_toPartialOrder_1861_, 0);
v_toLT_1866_ = lean_ctor_get(v_toPartialOrder_1861_, 1);
v_isSharedCheck_2017_ = !lean_is_exclusive(v_toPartialOrder_1861_);
if (v_isSharedCheck_2017_ == 0)
{
v___x_1868_ = v_toPartialOrder_1861_;
v_isShared_1869_ = v_isSharedCheck_2017_;
goto v_resetjp_1867_;
}
else
{
lean_inc(v_toLT_1866_);
lean_inc(v_toLE_1865_);
lean_dec(v_toPartialOrder_1861_);
v___x_1868_ = lean_box(0);
v_isShared_1869_ = v_isSharedCheck_2017_;
goto v_resetjp_1867_;
}
v_resetjp_1867_:
{
lean_object* v_top_1870_; lean_object* v_bot_1871_; lean_object* v_supSet_1872_; lean_object* v_infSet_1873_; lean_object* v___f_1874_; lean_object* v___x_1876_; 
lean_inc_n(v_toFun_1822_, 4);
v_top_1870_ = lean_apply_1(v_toFun_1822_, v_toOrderTop_1816_);
v_bot_1871_ = lean_apply_1(v_toFun_1822_, v_toOrderBot_1817_);
v_supSet_1872_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_1872_, 0, v_toSupSet_1824_);
lean_closure_set(v_supSet_1872_, 1, v_toFun_1822_);
v_infSet_1873_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_1873_, 0, v_toInfSet_1829_);
lean_closure_set(v_infSet_1873_, 1, v_toFun_1822_);
v___f_1874_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_1874_, 0, v___f_1853_);
if (v_isShared_1869_ == 0)
{
v___x_1876_ = v___x_1868_;
goto v_reusejp_1875_;
}
else
{
lean_object* v_reuseFailAlloc_2016_; 
v_reuseFailAlloc_2016_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2016_, 0, v_toLE_1865_);
lean_ctor_set(v_reuseFailAlloc_2016_, 1, v_toLT_1866_);
v___x_1876_ = v_reuseFailAlloc_2016_;
goto v_reusejp_1875_;
}
v_reusejp_1875_:
{
lean_object* v___x_1878_; 
lean_inc_ref(v___f_1874_);
if (v_isShared_1864_ == 0)
{
lean_ctor_set(v___x_1863_, 1, v___f_1874_);
lean_ctor_set(v___x_1863_, 0, v___x_1876_);
v___x_1878_ = v___x_1863_;
goto v_reusejp_1877_;
}
else
{
lean_object* v_reuseFailAlloc_2015_; 
v_reuseFailAlloc_2015_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2015_, 0, v___x_1876_);
lean_ctor_set(v_reuseFailAlloc_2015_, 1, v___f_1874_);
v___x_1878_ = v_reuseFailAlloc_2015_;
goto v_reusejp_1877_;
}
v_reusejp_1877_:
{
lean_object* v___x_1880_; 
lean_inc_ref(v___f_1852_);
lean_inc_ref(v___x_1878_);
if (v_isShared_1832_ == 0)
{
lean_ctor_set(v___x_1831_, 1, v___f_1852_);
lean_ctor_set(v___x_1831_, 0, v___x_1878_);
v___x_1880_ = v___x_1831_;
goto v_reusejp_1879_;
}
else
{
lean_object* v_reuseFailAlloc_2014_; 
v_reuseFailAlloc_2014_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2014_, 0, v___x_1878_);
lean_ctor_set(v_reuseFailAlloc_2014_, 1, v___f_1852_);
v___x_1880_ = v_reuseFailAlloc_2014_;
goto v_reusejp_1879_;
}
v_reusejp_1879_:
{
lean_object* v___x_1882_; 
lean_inc(v_bot_1871_);
if (v_isShared_1820_ == 0)
{
lean_ctor_set(v___x_1819_, 1, v_bot_1871_);
lean_ctor_set(v___x_1819_, 0, v_top_1870_);
v___x_1882_ = v___x_1819_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_2013_; 
v_reuseFailAlloc_2013_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2013_, 0, v_top_1870_);
lean_ctor_set(v_reuseFailAlloc_2013_, 1, v_bot_1871_);
v___x_1882_ = v_reuseFailAlloc_2013_;
goto v_reusejp_1881_;
}
v_reusejp_1881_:
{
lean_object* v_completeLattice_1883_; lean_object* v___x_1884_; lean_object* v_toGeneralizedCoheytingAlgebra_1885_; lean_object* v_toLattice_1886_; lean_object* v_toOrderTop_1887_; lean_object* v_toHNot_1888_; lean_object* v_toOrderBot_1889_; lean_object* v_toSDiff_1890_; lean_object* v_toSemilatticeSup_1891_; lean_object* v_inf_1892_; lean_object* v___x_1894_; uint8_t v_isShared_1895_; uint8_t v_isSharedCheck_2012_; 
lean_inc_ref(v___x_1880_);
v_completeLattice_1883_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_1883_, 0, v___x_1880_);
lean_ctor_set(v_completeLattice_1883_, 1, v_supSet_1872_);
lean_ctor_set(v_completeLattice_1883_, 2, v_infSet_1873_);
lean_ctor_set(v_completeLattice_1883_, 3, v___x_1882_);
v___x_1884_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v_inst_1812_);
v_toGeneralizedCoheytingAlgebra_1885_ = lean_ctor_get(v___x_1884_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1885_);
v_toLattice_1886_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1885_, 0);
lean_inc_ref(v_toLattice_1886_);
v_toOrderTop_1887_ = lean_ctor_get(v___x_1884_, 1);
lean_inc(v_toOrderTop_1887_);
v_toHNot_1888_ = lean_ctor_get(v___x_1884_, 2);
lean_inc(v_toHNot_1888_);
lean_dec_ref(v___x_1884_);
v_toOrderBot_1889_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1885_, 1);
lean_inc(v_toOrderBot_1889_);
v_toSDiff_1890_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1885_, 2);
lean_inc(v_toSDiff_1890_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1885_);
v_toSemilatticeSup_1891_ = lean_ctor_get(v_toLattice_1886_, 0);
v_inf_1892_ = lean_ctor_get(v_toLattice_1886_, 1);
v_isSharedCheck_2012_ = !lean_is_exclusive(v_toLattice_1886_);
if (v_isSharedCheck_2012_ == 0)
{
v___x_1894_ = v_toLattice_1886_;
v_isShared_1895_ = v_isSharedCheck_2012_;
goto v_resetjp_1893_;
}
else
{
lean_inc(v_inf_1892_);
lean_inc(v_toSemilatticeSup_1891_);
lean_dec(v_toLattice_1886_);
v___x_1894_ = lean_box(0);
v_isShared_1895_ = v_isSharedCheck_2012_;
goto v_resetjp_1893_;
}
v_resetjp_1893_:
{
lean_object* v_min_1896_; lean_object* v_semilatticeInf_1897_; lean_object* v_toPartialOrder_1898_; lean_object* v___x_1900_; uint8_t v_isShared_1901_; uint8_t v_isSharedCheck_2010_; 
lean_inc(v_toFun_1822_);
lean_inc_ref(v_e_1811_);
v_min_1896_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_1896_, 0, v___f_1838_);
lean_closure_set(v_min_1896_, 1, v_e_1811_);
lean_closure_set(v_min_1896_, 2, v_inf_1892_);
lean_closure_set(v_min_1896_, 3, v_toFun_1822_);
lean_inc_ref(v_min_1896_);
v_semilatticeInf_1897_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1896_, v_le_1840_, v_lt_1841_);
v_toPartialOrder_1898_ = lean_ctor_get(v_semilatticeInf_1897_, 0);
v_isSharedCheck_2010_ = !lean_is_exclusive(v_semilatticeInf_1897_);
if (v_isSharedCheck_2010_ == 0)
{
lean_object* v_unused_2011_; 
v_unused_2011_ = lean_ctor_get(v_semilatticeInf_1897_, 1);
lean_dec(v_unused_2011_);
v___x_1900_ = v_semilatticeInf_1897_;
v_isShared_1901_ = v_isSharedCheck_2010_;
goto v_resetjp_1899_;
}
else
{
lean_inc(v_toPartialOrder_1898_);
lean_dec(v_semilatticeInf_1897_);
v___x_1900_ = lean_box(0);
v_isShared_1901_ = v_isSharedCheck_2010_;
goto v_resetjp_1899_;
}
v_resetjp_1899_:
{
lean_object* v_toLE_1902_; lean_object* v_toLT_1903_; lean_object* v___x_1905_; uint8_t v_isShared_1906_; uint8_t v_isSharedCheck_2009_; 
v_toLE_1902_ = lean_ctor_get(v_toPartialOrder_1898_, 0);
v_toLT_1903_ = lean_ctor_get(v_toPartialOrder_1898_, 1);
v_isSharedCheck_2009_ = !lean_is_exclusive(v_toPartialOrder_1898_);
if (v_isSharedCheck_2009_ == 0)
{
v___x_1905_ = v_toPartialOrder_1898_;
v_isShared_1906_ = v_isSharedCheck_2009_;
goto v_resetjp_1904_;
}
else
{
lean_inc(v_toLT_1903_);
lean_inc(v_toLE_1902_);
lean_dec(v_toPartialOrder_1898_);
v___x_1905_ = lean_box(0);
v_isShared_1906_ = v_isSharedCheck_2009_;
goto v_resetjp_1904_;
}
v_resetjp_1904_:
{
lean_object* v___f_1907_; lean_object* v___f_1908_; lean_object* v___x_1910_; 
v___f_1907_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_1907_, 0, v_min_1896_);
lean_inc(v_toFun_1822_);
lean_inc_ref(v_e_1811_);
v___f_1908_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_1908_, 0, v_toSemilatticeSup_1891_);
lean_closure_set(v___f_1908_, 1, v___f_1838_);
lean_closure_set(v___f_1908_, 2, v_e_1811_);
lean_closure_set(v___f_1908_, 3, v_toFun_1822_);
if (v_isShared_1906_ == 0)
{
v___x_1910_ = v___x_1905_;
goto v_reusejp_1909_;
}
else
{
lean_object* v_reuseFailAlloc_2008_; 
v_reuseFailAlloc_2008_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2008_, 0, v_toLE_1902_);
lean_ctor_set(v_reuseFailAlloc_2008_, 1, v_toLT_1903_);
v___x_1910_ = v_reuseFailAlloc_2008_;
goto v_reusejp_1909_;
}
v_reusejp_1909_:
{
lean_object* v___x_1912_; 
lean_inc_ref(v___f_1908_);
if (v_isShared_1901_ == 0)
{
lean_ctor_set(v___x_1900_, 1, v___f_1908_);
lean_ctor_set(v___x_1900_, 0, v___x_1910_);
v___x_1912_ = v___x_1900_;
goto v_reusejp_1911_;
}
else
{
lean_object* v_reuseFailAlloc_2007_; 
v_reuseFailAlloc_2007_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2007_, 0, v___x_1910_);
lean_ctor_set(v_reuseFailAlloc_2007_, 1, v___f_1908_);
v___x_1912_ = v_reuseFailAlloc_2007_;
goto v_reusejp_1911_;
}
v_reusejp_1911_:
{
lean_object* v_lattice_1914_; 
lean_inc_ref(v___f_1907_);
if (v_isShared_1895_ == 0)
{
lean_ctor_set(v___x_1894_, 1, v___f_1907_);
lean_ctor_set(v___x_1894_, 0, v___x_1912_);
v_lattice_1914_ = v___x_1894_;
goto v_reusejp_1913_;
}
else
{
lean_object* v_reuseFailAlloc_2006_; 
v_reuseFailAlloc_2006_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2006_, 0, v___x_1912_);
lean_ctor_set(v_reuseFailAlloc_2006_, 1, v___f_1907_);
v_lattice_1914_ = v_reuseFailAlloc_2006_;
goto v_reusejp_1913_;
}
v_reusejp_1913_:
{
lean_object* v___x_1915_; lean_object* v_toPartialOrder_1916_; lean_object* v___x_1918_; uint8_t v_isShared_1919_; uint8_t v_isSharedCheck_2004_; 
v___x_1915_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1914_);
v_toPartialOrder_1916_ = lean_ctor_get(v___x_1915_, 0);
v_isSharedCheck_2004_ = !lean_is_exclusive(v___x_1915_);
if (v_isSharedCheck_2004_ == 0)
{
lean_object* v_unused_2005_; 
v_unused_2005_ = lean_ctor_get(v___x_1915_, 1);
lean_dec(v_unused_2005_);
v___x_1918_ = v___x_1915_;
v_isShared_1919_ = v_isSharedCheck_2004_;
goto v_resetjp_1917_;
}
else
{
lean_inc(v_toPartialOrder_1916_);
lean_dec(v___x_1915_);
v___x_1918_ = lean_box(0);
v_isShared_1919_ = v_isSharedCheck_2004_;
goto v_resetjp_1917_;
}
v_resetjp_1917_:
{
lean_object* v_toLE_1920_; lean_object* v_toLT_1921_; lean_object* v___x_1923_; uint8_t v_isShared_1924_; uint8_t v_isSharedCheck_2003_; 
v_toLE_1920_ = lean_ctor_get(v_toPartialOrder_1916_, 0);
v_toLT_1921_ = lean_ctor_get(v_toPartialOrder_1916_, 1);
v_isSharedCheck_2003_ = !lean_is_exclusive(v_toPartialOrder_1916_);
if (v_isSharedCheck_2003_ == 0)
{
v___x_1923_ = v_toPartialOrder_1916_;
v_isShared_1924_ = v_isSharedCheck_2003_;
goto v_resetjp_1922_;
}
else
{
lean_inc(v_toLT_1921_);
lean_inc(v_toLE_1920_);
lean_dec(v_toPartialOrder_1916_);
v___x_1923_ = lean_box(0);
v_isShared_1924_ = v_isSharedCheck_2003_;
goto v_resetjp_1922_;
}
v_resetjp_1922_:
{
lean_object* v___f_1925_; lean_object* v___x_1927_; 
v___f_1925_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_1925_, 0, v___f_1908_);
if (v_isShared_1924_ == 0)
{
v___x_1927_ = v___x_1923_;
goto v_reusejp_1926_;
}
else
{
lean_object* v_reuseFailAlloc_2002_; 
v_reuseFailAlloc_2002_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2002_, 0, v_toLE_1920_);
lean_ctor_set(v_reuseFailAlloc_2002_, 1, v_toLT_1921_);
v___x_1927_ = v_reuseFailAlloc_2002_;
goto v_reusejp_1926_;
}
v_reusejp_1926_:
{
lean_object* v___x_1929_; 
if (v_isShared_1919_ == 0)
{
lean_ctor_set(v___x_1918_, 1, v___f_1925_);
lean_ctor_set(v___x_1918_, 0, v___x_1927_);
v___x_1929_ = v___x_1918_;
goto v_reusejp_1928_;
}
else
{
lean_object* v_reuseFailAlloc_2001_; 
v_reuseFailAlloc_2001_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2001_, 0, v___x_1927_);
lean_ctor_set(v_reuseFailAlloc_2001_, 1, v___f_1925_);
v___x_1929_ = v_reuseFailAlloc_2001_;
goto v_reusejp_1928_;
}
v_reusejp_1928_:
{
lean_object* v___x_1931_; 
lean_inc_ref(v___x_1929_);
if (v_isShared_1827_ == 0)
{
lean_ctor_set(v___x_1826_, 1, v___f_1907_);
lean_ctor_set(v___x_1826_, 0, v___x_1929_);
v___x_1931_ = v___x_1826_;
goto v_reusejp_1930_;
}
else
{
lean_object* v_reuseFailAlloc_2000_; 
v_reuseFailAlloc_2000_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2000_, 0, v___x_1929_);
lean_ctor_set(v_reuseFailAlloc_2000_, 1, v___f_1907_);
v___x_1931_ = v_reuseFailAlloc_2000_;
goto v_reusejp_1930_;
}
v_reusejp_1930_:
{
lean_object* v___x_1932_; lean_object* v_toPartialOrder_1933_; lean_object* v_toLE_1934_; lean_object* v_toLT_1935_; lean_object* v___x_1937_; uint8_t v_isShared_1938_; uint8_t v_isSharedCheck_1999_; 
v___x_1932_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_1931_);
v_toPartialOrder_1933_ = lean_ctor_get(v___x_1932_, 0);
lean_inc_ref(v_toPartialOrder_1933_);
v_toLE_1934_ = lean_ctor_get(v_toPartialOrder_1933_, 0);
v_toLT_1935_ = lean_ctor_get(v_toPartialOrder_1933_, 1);
v_isSharedCheck_1999_ = !lean_is_exclusive(v_toPartialOrder_1933_);
if (v_isSharedCheck_1999_ == 0)
{
v___x_1937_ = v_toPartialOrder_1933_;
v_isShared_1938_ = v_isSharedCheck_1999_;
goto v_resetjp_1936_;
}
else
{
lean_inc(v_toLT_1935_);
lean_inc(v_toLE_1934_);
lean_dec(v_toPartialOrder_1933_);
v___x_1937_ = lean_box(0);
v_isShared_1938_ = v_isSharedCheck_1999_;
goto v_resetjp_1936_;
}
v_resetjp_1936_:
{
lean_object* v_bot_1939_; lean_object* v_hnot_1940_; lean_object* v_sdiff_1941_; lean_object* v_top_1942_; lean_object* v___f_1943_; lean_object* v___f_1944_; lean_object* v_coheytingAlgebra_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v_toPartialOrder_1948_; lean_object* v_toInfSet_1949_; lean_object* v___x_1951_; uint8_t v_isShared_1952_; uint8_t v_isSharedCheck_1998_; 
lean_inc_n(v_toFun_1822_, 3);
v_bot_1939_ = lean_apply_1(v_toFun_1822_, v_toOrderBot_1889_);
lean_inc_ref(v_e_1811_);
v_hnot_1940_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__17), 4, 3);
lean_closure_set(v_hnot_1940_, 0, v_e_1811_);
lean_closure_set(v_hnot_1940_, 1, v_toHNot_1888_);
lean_closure_set(v_hnot_1940_, 2, v_toFun_1822_);
v_sdiff_1941_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_1941_, 0, v___f_1838_);
lean_closure_set(v_sdiff_1941_, 1, v_e_1811_);
lean_closure_set(v_sdiff_1941_, 2, v_toSDiff_1890_);
lean_closure_set(v_sdiff_1941_, 3, v_toFun_1822_);
v_top_1942_ = lean_apply_1(v_toFun_1822_, v_toOrderTop_1887_);
v___f_1943_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1943_, 0, v___x_1932_);
v___f_1944_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1944_, 0, v___x_1929_);
v_coheytingAlgebra_1945_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_1943_, v___f_1944_, v_toLE_1934_, v_toLT_1935_, v_bot_1939_, v_top_1942_, v_hnot_1940_, v_sdiff_1941_);
v___x_1946_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_1880_);
lean_inc_ref(v_completeLattice_1883_);
v___x_1947_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_1883_);
v_toPartialOrder_1948_ = lean_ctor_get(v___x_1947_, 0);
v_toInfSet_1949_ = lean_ctor_get(v___x_1947_, 1);
v_isSharedCheck_1998_ = !lean_is_exclusive(v___x_1947_);
if (v_isSharedCheck_1998_ == 0)
{
v___x_1951_ = v___x_1947_;
v_isShared_1952_ = v_isSharedCheck_1998_;
goto v_resetjp_1950_;
}
else
{
lean_inc(v_toInfSet_1949_);
lean_inc(v_toPartialOrder_1948_);
lean_dec(v___x_1947_);
v___x_1951_ = lean_box(0);
v_isShared_1952_ = v_isSharedCheck_1998_;
goto v_resetjp_1950_;
}
v_resetjp_1950_:
{
lean_object* v_toLE_1953_; lean_object* v_toLT_1954_; lean_object* v___x_1956_; uint8_t v_isShared_1957_; uint8_t v_isSharedCheck_1997_; 
v_toLE_1953_ = lean_ctor_get(v_toPartialOrder_1948_, 0);
v_toLT_1954_ = lean_ctor_get(v_toPartialOrder_1948_, 1);
v_isSharedCheck_1997_ = !lean_is_exclusive(v_toPartialOrder_1948_);
if (v_isSharedCheck_1997_ == 0)
{
v___x_1956_ = v_toPartialOrder_1948_;
v_isShared_1957_ = v_isSharedCheck_1997_;
goto v_resetjp_1955_;
}
else
{
lean_inc(v_toLT_1954_);
lean_inc(v_toLE_1953_);
lean_dec(v_toPartialOrder_1948_);
v___x_1956_ = lean_box(0);
v_isShared_1957_ = v_isSharedCheck_1997_;
goto v_resetjp_1955_;
}
v_resetjp_1955_:
{
lean_object* v___x_1958_; lean_object* v_toGeneralizedCoheytingAlgebra_1959_; lean_object* v_toSupSet_1960_; lean_object* v___x_1962_; uint8_t v_isShared_1963_; uint8_t v_isSharedCheck_1995_; 
v___x_1958_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_1883_);
v_toGeneralizedCoheytingAlgebra_1959_ = lean_ctor_get(v_coheytingAlgebra_1945_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1959_);
v_toSupSet_1960_ = lean_ctor_get(v___x_1958_, 1);
v_isSharedCheck_1995_ = !lean_is_exclusive(v___x_1958_);
if (v_isSharedCheck_1995_ == 0)
{
lean_object* v_unused_1996_; 
v_unused_1996_ = lean_ctor_get(v___x_1958_, 0);
lean_dec(v_unused_1996_);
v___x_1962_ = v___x_1958_;
v_isShared_1963_ = v_isSharedCheck_1995_;
goto v_resetjp_1961_;
}
else
{
lean_inc(v_toSupSet_1960_);
lean_dec(v___x_1958_);
v___x_1962_ = lean_box(0);
v_isShared_1963_ = v_isSharedCheck_1995_;
goto v_resetjp_1961_;
}
v_resetjp_1961_:
{
lean_object* v_toOrderTop_1964_; lean_object* v_toHNot_1965_; lean_object* v_toSDiff_1966_; lean_object* v___f_1967_; lean_object* v___f_1968_; lean_object* v___x_1970_; 
v_toOrderTop_1964_ = lean_ctor_get(v_coheytingAlgebra_1945_, 1);
lean_inc(v_toOrderTop_1964_);
v_toHNot_1965_ = lean_ctor_get(v_coheytingAlgebra_1945_, 2);
lean_inc(v_toHNot_1965_);
lean_dec_ref(v_coheytingAlgebra_1945_);
v_toSDiff_1966_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1959_, 2);
lean_inc(v_toSDiff_1966_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1959_);
v___f_1967_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1967_, 0, v___x_1878_);
v___f_1968_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1968_, 0, v___x_1946_);
if (v_isShared_1957_ == 0)
{
v___x_1970_ = v___x_1956_;
goto v_reusejp_1969_;
}
else
{
lean_object* v_reuseFailAlloc_1994_; 
v_reuseFailAlloc_1994_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1994_, 0, v_toLE_1953_);
lean_ctor_set(v_reuseFailAlloc_1994_, 1, v_toLT_1954_);
v___x_1970_ = v_reuseFailAlloc_1994_;
goto v_reusejp_1969_;
}
v_reusejp_1969_:
{
lean_object* v___x_1972_; 
if (v_isShared_1963_ == 0)
{
lean_ctor_set(v___x_1962_, 1, v___f_1874_);
lean_ctor_set(v___x_1962_, 0, v___x_1970_);
v___x_1972_ = v___x_1962_;
goto v_reusejp_1971_;
}
else
{
lean_object* v_reuseFailAlloc_1993_; 
v_reuseFailAlloc_1993_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1993_, 0, v___x_1970_);
lean_ctor_set(v_reuseFailAlloc_1993_, 1, v___f_1874_);
v___x_1972_ = v_reuseFailAlloc_1993_;
goto v_reusejp_1971_;
}
v_reusejp_1971_:
{
lean_object* v___x_1974_; 
if (v_isShared_1952_ == 0)
{
lean_ctor_set(v___x_1951_, 1, v___f_1852_);
lean_ctor_set(v___x_1951_, 0, v___x_1972_);
v___x_1974_ = v___x_1951_;
goto v_reusejp_1973_;
}
else
{
lean_object* v_reuseFailAlloc_1992_; 
v_reuseFailAlloc_1992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1992_, 0, v___x_1972_);
lean_ctor_set(v_reuseFailAlloc_1992_, 1, v___f_1852_);
v___x_1974_ = v_reuseFailAlloc_1992_;
goto v_reusejp_1973_;
}
v_reusejp_1973_:
{
lean_object* v___x_1976_; 
lean_inc(v_bot_1871_);
lean_inc(v_toOrderTop_1964_);
if (v_isShared_1938_ == 0)
{
lean_ctor_set(v___x_1937_, 1, v_bot_1871_);
lean_ctor_set(v___x_1937_, 0, v_toOrderTop_1964_);
v___x_1976_ = v___x_1937_;
goto v_reusejp_1975_;
}
else
{
lean_object* v_reuseFailAlloc_1991_; 
v_reuseFailAlloc_1991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1991_, 0, v_toOrderTop_1964_);
lean_ctor_set(v_reuseFailAlloc_1991_, 1, v_bot_1871_);
v___x_1976_ = v_reuseFailAlloc_1991_;
goto v_reusejp_1975_;
}
v_reusejp_1975_:
{
lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v_toGeneralizedCoheytingAlgebra_1979_; lean_object* v_toHNot_1980_; lean_object* v_toSDiff_1981_; lean_object* v___x_1983_; uint8_t v_isShared_1984_; uint8_t v_isSharedCheck_1988_; 
v___x_1977_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1977_, 0, v___x_1974_);
lean_ctor_set(v___x_1977_, 1, v_toSupSet_1960_);
lean_ctor_set(v___x_1977_, 2, v_toInfSet_1949_);
lean_ctor_set(v___x_1977_, 3, v___x_1976_);
v___x_1978_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_1968_, v___f_1967_, v_toLE_1953_, v_toLT_1954_, v_bot_1871_, v_toOrderTop_1964_, v_toHNot_1965_, v_toSDiff_1966_);
v_toGeneralizedCoheytingAlgebra_1979_ = lean_ctor_get(v___x_1978_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1979_);
v_toHNot_1980_ = lean_ctor_get(v___x_1978_, 2);
lean_inc(v_toHNot_1980_);
lean_dec_ref(v___x_1978_);
v_toSDiff_1981_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1979_, 2);
v_isSharedCheck_1988_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_1979_);
if (v_isSharedCheck_1988_ == 0)
{
lean_object* v_unused_1989_; lean_object* v_unused_1990_; 
v_unused_1989_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1979_, 1);
lean_dec(v_unused_1989_);
v_unused_1990_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1979_, 0);
lean_dec(v_unused_1990_);
v___x_1983_ = v_toGeneralizedCoheytingAlgebra_1979_;
v_isShared_1984_ = v_isSharedCheck_1988_;
goto v_resetjp_1982_;
}
else
{
lean_inc(v_toSDiff_1981_);
lean_dec(v_toGeneralizedCoheytingAlgebra_1979_);
v___x_1983_ = lean_box(0);
v_isShared_1984_ = v_isSharedCheck_1988_;
goto v_resetjp_1982_;
}
v_resetjp_1982_:
{
lean_object* v___x_1986_; 
if (v_isShared_1984_ == 0)
{
lean_ctor_set(v___x_1983_, 2, v_toHNot_1980_);
lean_ctor_set(v___x_1983_, 1, v_toSDiff_1981_);
lean_ctor_set(v___x_1983_, 0, v___x_1977_);
v___x_1986_ = v___x_1983_;
goto v_reusejp_1985_;
}
else
{
lean_object* v_reuseFailAlloc_1987_; 
v_reuseFailAlloc_1987_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1987_, 0, v___x_1977_);
lean_ctor_set(v_reuseFailAlloc_1987_, 1, v_toSDiff_1981_);
lean_ctor_set(v_reuseFailAlloc_1987_, 2, v_toHNot_1980_);
v___x_1986_ = v_reuseFailAlloc_1987_;
goto v_reusejp_1985_;
}
v_reusejp_1985_:
{
return v___x_1986_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coframe(lean_object* v_00_u03b1_2032_, lean_object* v_00_u03b2_2033_, lean_object* v_e_2034_, lean_object* v_inst_2035_){
_start:
{
lean_object* v_toCompleteLattice_2036_; lean_object* v_toBoundedOrder_2037_; lean_object* v_toLattice_2038_; lean_object* v_toOrderTop_2039_; lean_object* v_toOrderBot_2040_; lean_object* v___x_2042_; uint8_t v_isShared_2043_; uint8_t v_isSharedCheck_2254_; 
v_toCompleteLattice_2036_ = lean_ctor_get(v_inst_2035_, 0);
v_toBoundedOrder_2037_ = lean_ctor_get(v_toCompleteLattice_2036_, 3);
lean_inc_ref(v_toBoundedOrder_2037_);
v_toLattice_2038_ = lean_ctor_get(v_toCompleteLattice_2036_, 0);
lean_inc_ref(v_toLattice_2038_);
v_toOrderTop_2039_ = lean_ctor_get(v_toBoundedOrder_2037_, 0);
v_toOrderBot_2040_ = lean_ctor_get(v_toBoundedOrder_2037_, 1);
v_isSharedCheck_2254_ = !lean_is_exclusive(v_toBoundedOrder_2037_);
if (v_isSharedCheck_2254_ == 0)
{
v___x_2042_ = v_toBoundedOrder_2037_;
v_isShared_2043_ = v_isSharedCheck_2254_;
goto v_resetjp_2041_;
}
else
{
lean_inc(v_toOrderBot_2040_);
lean_inc(v_toOrderTop_2039_);
lean_dec(v_toBoundedOrder_2037_);
v___x_2042_ = lean_box(0);
v_isShared_2043_ = v_isSharedCheck_2254_;
goto v_resetjp_2041_;
}
v_resetjp_2041_:
{
lean_object* v___x_2044_; lean_object* v_toFun_2045_; lean_object* v___x_2046_; lean_object* v_toSupSet_2047_; lean_object* v___x_2049_; uint8_t v_isShared_2050_; uint8_t v_isSharedCheck_2252_; 
lean_inc_ref(v_e_2034_);
v___x_2044_ = lp_mathlib_Equiv_symm___redArg(v_e_2034_);
v_toFun_2045_ = lean_ctor_get(v___x_2044_, 0);
lean_inc(v_toFun_2045_);
lean_dec_ref(v___x_2044_);
lean_inc_ref(v_toCompleteLattice_2036_);
v___x_2046_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_2036_);
v_toSupSet_2047_ = lean_ctor_get(v___x_2046_, 1);
v_isSharedCheck_2252_ = !lean_is_exclusive(v___x_2046_);
if (v_isSharedCheck_2252_ == 0)
{
lean_object* v_unused_2253_; 
v_unused_2253_ = lean_ctor_get(v___x_2046_, 0);
lean_dec(v_unused_2253_);
v___x_2049_ = v___x_2046_;
v_isShared_2050_ = v_isSharedCheck_2252_;
goto v_resetjp_2048_;
}
else
{
lean_inc(v_toSupSet_2047_);
lean_dec(v___x_2046_);
v___x_2049_ = lean_box(0);
v_isShared_2050_ = v_isSharedCheck_2252_;
goto v_resetjp_2048_;
}
v_resetjp_2048_:
{
lean_object* v___x_2051_; lean_object* v_toInfSet_2052_; lean_object* v___x_2054_; uint8_t v_isShared_2055_; uint8_t v_isSharedCheck_2250_; 
lean_inc_ref(v_toCompleteLattice_2036_);
v___x_2051_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_2036_);
v_toInfSet_2052_ = lean_ctor_get(v___x_2051_, 1);
v_isSharedCheck_2250_ = !lean_is_exclusive(v___x_2051_);
if (v_isSharedCheck_2250_ == 0)
{
lean_object* v_unused_2251_; 
v_unused_2251_ = lean_ctor_get(v___x_2051_, 0);
lean_dec(v_unused_2251_);
v___x_2054_ = v___x_2051_;
v_isShared_2055_ = v_isSharedCheck_2250_;
goto v_resetjp_2053_;
}
else
{
lean_inc(v_toInfSet_2052_);
lean_dec(v___x_2051_);
v___x_2054_ = lean_box(0);
v_isShared_2055_ = v_isSharedCheck_2250_;
goto v_resetjp_2053_;
}
v_resetjp_2053_:
{
lean_object* v_toSemilatticeSup_2056_; lean_object* v_inf_2057_; lean_object* v___x_2059_; uint8_t v_isShared_2060_; uint8_t v_isSharedCheck_2249_; 
v_toSemilatticeSup_2056_ = lean_ctor_get(v_toLattice_2038_, 0);
v_inf_2057_ = lean_ctor_get(v_toLattice_2038_, 1);
v_isSharedCheck_2249_ = !lean_is_exclusive(v_toLattice_2038_);
if (v_isSharedCheck_2249_ == 0)
{
v___x_2059_ = v_toLattice_2038_;
v_isShared_2060_ = v_isSharedCheck_2249_;
goto v_resetjp_2058_;
}
else
{
lean_inc(v_inf_2057_);
lean_inc(v_toSemilatticeSup_2056_);
lean_dec(v_toLattice_2038_);
v___x_2059_ = lean_box(0);
v_isShared_2060_ = v_isSharedCheck_2249_;
goto v_resetjp_2058_;
}
v_resetjp_2058_:
{
lean_object* v___f_2061_; lean_object* v_min_2062_; lean_object* v_le_2063_; lean_object* v_lt_2064_; lean_object* v_semilatticeInf_2065_; lean_object* v_toPartialOrder_2066_; lean_object* v___x_2068_; uint8_t v_isShared_2069_; uint8_t v_isSharedCheck_2247_; 
v___f_2061_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_2045_);
lean_inc_ref(v_e_2034_);
v_min_2062_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2062_, 0, v___f_2061_);
lean_closure_set(v_min_2062_, 1, v_e_2034_);
lean_closure_set(v_min_2062_, 2, v_inf_2057_);
lean_closure_set(v_min_2062_, 3, v_toFun_2045_);
v_le_2063_ = lean_box(0);
v_lt_2064_ = lean_box(0);
lean_inc_ref(v_min_2062_);
v_semilatticeInf_2065_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2062_, v_le_2063_, v_lt_2064_);
v_toPartialOrder_2066_ = lean_ctor_get(v_semilatticeInf_2065_, 0);
v_isSharedCheck_2247_ = !lean_is_exclusive(v_semilatticeInf_2065_);
if (v_isSharedCheck_2247_ == 0)
{
lean_object* v_unused_2248_; 
v_unused_2248_ = lean_ctor_get(v_semilatticeInf_2065_, 1);
lean_dec(v_unused_2248_);
v___x_2068_ = v_semilatticeInf_2065_;
v_isShared_2069_ = v_isSharedCheck_2247_;
goto v_resetjp_2067_;
}
else
{
lean_inc(v_toPartialOrder_2066_);
lean_dec(v_semilatticeInf_2065_);
v___x_2068_ = lean_box(0);
v_isShared_2069_ = v_isSharedCheck_2247_;
goto v_resetjp_2067_;
}
v_resetjp_2067_:
{
lean_object* v_toLE_2070_; lean_object* v_toLT_2071_; lean_object* v___x_2073_; uint8_t v_isShared_2074_; uint8_t v_isSharedCheck_2246_; 
v_toLE_2070_ = lean_ctor_get(v_toPartialOrder_2066_, 0);
v_toLT_2071_ = lean_ctor_get(v_toPartialOrder_2066_, 1);
v_isSharedCheck_2246_ = !lean_is_exclusive(v_toPartialOrder_2066_);
if (v_isSharedCheck_2246_ == 0)
{
v___x_2073_ = v_toPartialOrder_2066_;
v_isShared_2074_ = v_isSharedCheck_2246_;
goto v_resetjp_2072_;
}
else
{
lean_inc(v_toLT_2071_);
lean_inc(v_toLE_2070_);
lean_dec(v_toPartialOrder_2066_);
v___x_2073_ = lean_box(0);
v_isShared_2074_ = v_isSharedCheck_2246_;
goto v_resetjp_2072_;
}
v_resetjp_2072_:
{
lean_object* v___f_2075_; lean_object* v___f_2076_; lean_object* v___x_2078_; 
v___f_2075_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_2075_, 0, v_min_2062_);
lean_inc(v_toFun_2045_);
lean_inc_ref(v_e_2034_);
v___f_2076_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_2076_, 0, v_toSemilatticeSup_2056_);
lean_closure_set(v___f_2076_, 1, v___f_2061_);
lean_closure_set(v___f_2076_, 2, v_e_2034_);
lean_closure_set(v___f_2076_, 3, v_toFun_2045_);
if (v_isShared_2074_ == 0)
{
v___x_2078_ = v___x_2073_;
goto v_reusejp_2077_;
}
else
{
lean_object* v_reuseFailAlloc_2245_; 
v_reuseFailAlloc_2245_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2245_, 0, v_toLE_2070_);
lean_ctor_set(v_reuseFailAlloc_2245_, 1, v_toLT_2071_);
v___x_2078_ = v_reuseFailAlloc_2245_;
goto v_reusejp_2077_;
}
v_reusejp_2077_:
{
lean_object* v___x_2080_; 
lean_inc_ref(v___f_2076_);
if (v_isShared_2069_ == 0)
{
lean_ctor_set(v___x_2068_, 1, v___f_2076_);
lean_ctor_set(v___x_2068_, 0, v___x_2078_);
v___x_2080_ = v___x_2068_;
goto v_reusejp_2079_;
}
else
{
lean_object* v_reuseFailAlloc_2244_; 
v_reuseFailAlloc_2244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2244_, 0, v___x_2078_);
lean_ctor_set(v_reuseFailAlloc_2244_, 1, v___f_2076_);
v___x_2080_ = v_reuseFailAlloc_2244_;
goto v_reusejp_2079_;
}
v_reusejp_2079_:
{
lean_object* v_lattice_2082_; 
lean_inc_ref(v___f_2075_);
if (v_isShared_2060_ == 0)
{
lean_ctor_set(v___x_2059_, 1, v___f_2075_);
lean_ctor_set(v___x_2059_, 0, v___x_2080_);
v_lattice_2082_ = v___x_2059_;
goto v_reusejp_2081_;
}
else
{
lean_object* v_reuseFailAlloc_2243_; 
v_reuseFailAlloc_2243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2243_, 0, v___x_2080_);
lean_ctor_set(v_reuseFailAlloc_2243_, 1, v___f_2075_);
v_lattice_2082_ = v_reuseFailAlloc_2243_;
goto v_reusejp_2081_;
}
v_reusejp_2081_:
{
lean_object* v___x_2083_; lean_object* v_toPartialOrder_2084_; lean_object* v___x_2086_; uint8_t v_isShared_2087_; uint8_t v_isSharedCheck_2241_; 
v___x_2083_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2082_);
v_toPartialOrder_2084_ = lean_ctor_get(v___x_2083_, 0);
v_isSharedCheck_2241_ = !lean_is_exclusive(v___x_2083_);
if (v_isSharedCheck_2241_ == 0)
{
lean_object* v_unused_2242_; 
v_unused_2242_ = lean_ctor_get(v___x_2083_, 1);
lean_dec(v_unused_2242_);
v___x_2086_ = v___x_2083_;
v_isShared_2087_ = v_isSharedCheck_2241_;
goto v_resetjp_2085_;
}
else
{
lean_inc(v_toPartialOrder_2084_);
lean_dec(v___x_2083_);
v___x_2086_ = lean_box(0);
v_isShared_2087_ = v_isSharedCheck_2241_;
goto v_resetjp_2085_;
}
v_resetjp_2085_:
{
lean_object* v_toLE_2088_; lean_object* v_toLT_2089_; lean_object* v___x_2091_; uint8_t v_isShared_2092_; uint8_t v_isSharedCheck_2240_; 
v_toLE_2088_ = lean_ctor_get(v_toPartialOrder_2084_, 0);
v_toLT_2089_ = lean_ctor_get(v_toPartialOrder_2084_, 1);
v_isSharedCheck_2240_ = !lean_is_exclusive(v_toPartialOrder_2084_);
if (v_isSharedCheck_2240_ == 0)
{
v___x_2091_ = v_toPartialOrder_2084_;
v_isShared_2092_ = v_isSharedCheck_2240_;
goto v_resetjp_2090_;
}
else
{
lean_inc(v_toLT_2089_);
lean_inc(v_toLE_2088_);
lean_dec(v_toPartialOrder_2084_);
v___x_2091_ = lean_box(0);
v_isShared_2092_ = v_isSharedCheck_2240_;
goto v_resetjp_2090_;
}
v_resetjp_2090_:
{
lean_object* v_top_2093_; lean_object* v_bot_2094_; lean_object* v_supSet_2095_; lean_object* v_infSet_2096_; lean_object* v___f_2097_; lean_object* v___x_2099_; 
lean_inc_n(v_toFun_2045_, 4);
v_top_2093_ = lean_apply_1(v_toFun_2045_, v_toOrderTop_2039_);
v_bot_2094_ = lean_apply_1(v_toFun_2045_, v_toOrderBot_2040_);
v_supSet_2095_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_2095_, 0, v_toSupSet_2047_);
lean_closure_set(v_supSet_2095_, 1, v_toFun_2045_);
v_infSet_2096_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_2096_, 0, v_toInfSet_2052_);
lean_closure_set(v_infSet_2096_, 1, v_toFun_2045_);
v___f_2097_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_2097_, 0, v___f_2076_);
if (v_isShared_2092_ == 0)
{
v___x_2099_ = v___x_2091_;
goto v_reusejp_2098_;
}
else
{
lean_object* v_reuseFailAlloc_2239_; 
v_reuseFailAlloc_2239_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2239_, 0, v_toLE_2088_);
lean_ctor_set(v_reuseFailAlloc_2239_, 1, v_toLT_2089_);
v___x_2099_ = v_reuseFailAlloc_2239_;
goto v_reusejp_2098_;
}
v_reusejp_2098_:
{
lean_object* v___x_2101_; 
lean_inc_ref(v___f_2097_);
if (v_isShared_2087_ == 0)
{
lean_ctor_set(v___x_2086_, 1, v___f_2097_);
lean_ctor_set(v___x_2086_, 0, v___x_2099_);
v___x_2101_ = v___x_2086_;
goto v_reusejp_2100_;
}
else
{
lean_object* v_reuseFailAlloc_2238_; 
v_reuseFailAlloc_2238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2238_, 0, v___x_2099_);
lean_ctor_set(v_reuseFailAlloc_2238_, 1, v___f_2097_);
v___x_2101_ = v_reuseFailAlloc_2238_;
goto v_reusejp_2100_;
}
v_reusejp_2100_:
{
lean_object* v___x_2103_; 
lean_inc_ref(v___f_2075_);
lean_inc_ref(v___x_2101_);
if (v_isShared_2055_ == 0)
{
lean_ctor_set(v___x_2054_, 1, v___f_2075_);
lean_ctor_set(v___x_2054_, 0, v___x_2101_);
v___x_2103_ = v___x_2054_;
goto v_reusejp_2102_;
}
else
{
lean_object* v_reuseFailAlloc_2237_; 
v_reuseFailAlloc_2237_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2237_, 0, v___x_2101_);
lean_ctor_set(v_reuseFailAlloc_2237_, 1, v___f_2075_);
v___x_2103_ = v_reuseFailAlloc_2237_;
goto v_reusejp_2102_;
}
v_reusejp_2102_:
{
lean_object* v___x_2105_; 
lean_inc(v_bot_2094_);
if (v_isShared_2043_ == 0)
{
lean_ctor_set(v___x_2042_, 1, v_bot_2094_);
lean_ctor_set(v___x_2042_, 0, v_top_2093_);
v___x_2105_ = v___x_2042_;
goto v_reusejp_2104_;
}
else
{
lean_object* v_reuseFailAlloc_2236_; 
v_reuseFailAlloc_2236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2236_, 0, v_top_2093_);
lean_ctor_set(v_reuseFailAlloc_2236_, 1, v_bot_2094_);
v___x_2105_ = v_reuseFailAlloc_2236_;
goto v_reusejp_2104_;
}
v_reusejp_2104_:
{
lean_object* v_completeLattice_2106_; lean_object* v___x_2107_; lean_object* v_toGeneralizedCoheytingAlgebra_2108_; lean_object* v_toLattice_2109_; lean_object* v_toOrderTop_2110_; lean_object* v_toHNot_2111_; lean_object* v_toOrderBot_2112_; lean_object* v_toSDiff_2113_; lean_object* v_toSemilatticeSup_2114_; lean_object* v_inf_2115_; lean_object* v___x_2117_; uint8_t v_isShared_2118_; uint8_t v_isSharedCheck_2235_; 
lean_inc_ref(v___x_2103_);
v_completeLattice_2106_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_2106_, 0, v___x_2103_);
lean_ctor_set(v_completeLattice_2106_, 1, v_supSet_2095_);
lean_ctor_set(v_completeLattice_2106_, 2, v_infSet_2096_);
lean_ctor_set(v_completeLattice_2106_, 3, v___x_2105_);
v___x_2107_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v_inst_2035_);
v_toGeneralizedCoheytingAlgebra_2108_ = lean_ctor_get(v___x_2107_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2108_);
v_toLattice_2109_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2108_, 0);
lean_inc_ref(v_toLattice_2109_);
v_toOrderTop_2110_ = lean_ctor_get(v___x_2107_, 1);
lean_inc(v_toOrderTop_2110_);
v_toHNot_2111_ = lean_ctor_get(v___x_2107_, 2);
lean_inc(v_toHNot_2111_);
lean_dec_ref(v___x_2107_);
v_toOrderBot_2112_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2108_, 1);
lean_inc(v_toOrderBot_2112_);
v_toSDiff_2113_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2108_, 2);
lean_inc(v_toSDiff_2113_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_2108_);
v_toSemilatticeSup_2114_ = lean_ctor_get(v_toLattice_2109_, 0);
v_inf_2115_ = lean_ctor_get(v_toLattice_2109_, 1);
v_isSharedCheck_2235_ = !lean_is_exclusive(v_toLattice_2109_);
if (v_isSharedCheck_2235_ == 0)
{
v___x_2117_ = v_toLattice_2109_;
v_isShared_2118_ = v_isSharedCheck_2235_;
goto v_resetjp_2116_;
}
else
{
lean_inc(v_inf_2115_);
lean_inc(v_toSemilatticeSup_2114_);
lean_dec(v_toLattice_2109_);
v___x_2117_ = lean_box(0);
v_isShared_2118_ = v_isSharedCheck_2235_;
goto v_resetjp_2116_;
}
v_resetjp_2116_:
{
lean_object* v_min_2119_; lean_object* v_semilatticeInf_2120_; lean_object* v_toPartialOrder_2121_; lean_object* v___x_2123_; uint8_t v_isShared_2124_; uint8_t v_isSharedCheck_2233_; 
lean_inc(v_toFun_2045_);
lean_inc_ref(v_e_2034_);
v_min_2119_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2119_, 0, v___f_2061_);
lean_closure_set(v_min_2119_, 1, v_e_2034_);
lean_closure_set(v_min_2119_, 2, v_inf_2115_);
lean_closure_set(v_min_2119_, 3, v_toFun_2045_);
lean_inc_ref(v_min_2119_);
v_semilatticeInf_2120_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2119_, v_le_2063_, v_lt_2064_);
v_toPartialOrder_2121_ = lean_ctor_get(v_semilatticeInf_2120_, 0);
v_isSharedCheck_2233_ = !lean_is_exclusive(v_semilatticeInf_2120_);
if (v_isSharedCheck_2233_ == 0)
{
lean_object* v_unused_2234_; 
v_unused_2234_ = lean_ctor_get(v_semilatticeInf_2120_, 1);
lean_dec(v_unused_2234_);
v___x_2123_ = v_semilatticeInf_2120_;
v_isShared_2124_ = v_isSharedCheck_2233_;
goto v_resetjp_2122_;
}
else
{
lean_inc(v_toPartialOrder_2121_);
lean_dec(v_semilatticeInf_2120_);
v___x_2123_ = lean_box(0);
v_isShared_2124_ = v_isSharedCheck_2233_;
goto v_resetjp_2122_;
}
v_resetjp_2122_:
{
lean_object* v_toLE_2125_; lean_object* v_toLT_2126_; lean_object* v___x_2128_; uint8_t v_isShared_2129_; uint8_t v_isSharedCheck_2232_; 
v_toLE_2125_ = lean_ctor_get(v_toPartialOrder_2121_, 0);
v_toLT_2126_ = lean_ctor_get(v_toPartialOrder_2121_, 1);
v_isSharedCheck_2232_ = !lean_is_exclusive(v_toPartialOrder_2121_);
if (v_isSharedCheck_2232_ == 0)
{
v___x_2128_ = v_toPartialOrder_2121_;
v_isShared_2129_ = v_isSharedCheck_2232_;
goto v_resetjp_2127_;
}
else
{
lean_inc(v_toLT_2126_);
lean_inc(v_toLE_2125_);
lean_dec(v_toPartialOrder_2121_);
v___x_2128_ = lean_box(0);
v_isShared_2129_ = v_isSharedCheck_2232_;
goto v_resetjp_2127_;
}
v_resetjp_2127_:
{
lean_object* v___f_2130_; lean_object* v___f_2131_; lean_object* v___x_2133_; 
v___f_2130_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_2130_, 0, v_min_2119_);
lean_inc(v_toFun_2045_);
lean_inc_ref(v_e_2034_);
v___f_2131_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_2131_, 0, v_toSemilatticeSup_2114_);
lean_closure_set(v___f_2131_, 1, v___f_2061_);
lean_closure_set(v___f_2131_, 2, v_e_2034_);
lean_closure_set(v___f_2131_, 3, v_toFun_2045_);
if (v_isShared_2129_ == 0)
{
v___x_2133_ = v___x_2128_;
goto v_reusejp_2132_;
}
else
{
lean_object* v_reuseFailAlloc_2231_; 
v_reuseFailAlloc_2231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2231_, 0, v_toLE_2125_);
lean_ctor_set(v_reuseFailAlloc_2231_, 1, v_toLT_2126_);
v___x_2133_ = v_reuseFailAlloc_2231_;
goto v_reusejp_2132_;
}
v_reusejp_2132_:
{
lean_object* v___x_2135_; 
lean_inc_ref(v___f_2131_);
if (v_isShared_2124_ == 0)
{
lean_ctor_set(v___x_2123_, 1, v___f_2131_);
lean_ctor_set(v___x_2123_, 0, v___x_2133_);
v___x_2135_ = v___x_2123_;
goto v_reusejp_2134_;
}
else
{
lean_object* v_reuseFailAlloc_2230_; 
v_reuseFailAlloc_2230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2230_, 0, v___x_2133_);
lean_ctor_set(v_reuseFailAlloc_2230_, 1, v___f_2131_);
v___x_2135_ = v_reuseFailAlloc_2230_;
goto v_reusejp_2134_;
}
v_reusejp_2134_:
{
lean_object* v_lattice_2137_; 
lean_inc_ref(v___f_2130_);
if (v_isShared_2118_ == 0)
{
lean_ctor_set(v___x_2117_, 1, v___f_2130_);
lean_ctor_set(v___x_2117_, 0, v___x_2135_);
v_lattice_2137_ = v___x_2117_;
goto v_reusejp_2136_;
}
else
{
lean_object* v_reuseFailAlloc_2229_; 
v_reuseFailAlloc_2229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2229_, 0, v___x_2135_);
lean_ctor_set(v_reuseFailAlloc_2229_, 1, v___f_2130_);
v_lattice_2137_ = v_reuseFailAlloc_2229_;
goto v_reusejp_2136_;
}
v_reusejp_2136_:
{
lean_object* v___x_2138_; lean_object* v_toPartialOrder_2139_; lean_object* v___x_2141_; uint8_t v_isShared_2142_; uint8_t v_isSharedCheck_2227_; 
v___x_2138_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2137_);
v_toPartialOrder_2139_ = lean_ctor_get(v___x_2138_, 0);
v_isSharedCheck_2227_ = !lean_is_exclusive(v___x_2138_);
if (v_isSharedCheck_2227_ == 0)
{
lean_object* v_unused_2228_; 
v_unused_2228_ = lean_ctor_get(v___x_2138_, 1);
lean_dec(v_unused_2228_);
v___x_2141_ = v___x_2138_;
v_isShared_2142_ = v_isSharedCheck_2227_;
goto v_resetjp_2140_;
}
else
{
lean_inc(v_toPartialOrder_2139_);
lean_dec(v___x_2138_);
v___x_2141_ = lean_box(0);
v_isShared_2142_ = v_isSharedCheck_2227_;
goto v_resetjp_2140_;
}
v_resetjp_2140_:
{
lean_object* v_toLE_2143_; lean_object* v_toLT_2144_; lean_object* v___x_2146_; uint8_t v_isShared_2147_; uint8_t v_isSharedCheck_2226_; 
v_toLE_2143_ = lean_ctor_get(v_toPartialOrder_2139_, 0);
v_toLT_2144_ = lean_ctor_get(v_toPartialOrder_2139_, 1);
v_isSharedCheck_2226_ = !lean_is_exclusive(v_toPartialOrder_2139_);
if (v_isSharedCheck_2226_ == 0)
{
v___x_2146_ = v_toPartialOrder_2139_;
v_isShared_2147_ = v_isSharedCheck_2226_;
goto v_resetjp_2145_;
}
else
{
lean_inc(v_toLT_2144_);
lean_inc(v_toLE_2143_);
lean_dec(v_toPartialOrder_2139_);
v___x_2146_ = lean_box(0);
v_isShared_2147_ = v_isSharedCheck_2226_;
goto v_resetjp_2145_;
}
v_resetjp_2145_:
{
lean_object* v___f_2148_; lean_object* v___x_2150_; 
v___f_2148_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_2148_, 0, v___f_2131_);
if (v_isShared_2147_ == 0)
{
v___x_2150_ = v___x_2146_;
goto v_reusejp_2149_;
}
else
{
lean_object* v_reuseFailAlloc_2225_; 
v_reuseFailAlloc_2225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2225_, 0, v_toLE_2143_);
lean_ctor_set(v_reuseFailAlloc_2225_, 1, v_toLT_2144_);
v___x_2150_ = v_reuseFailAlloc_2225_;
goto v_reusejp_2149_;
}
v_reusejp_2149_:
{
lean_object* v___x_2152_; 
if (v_isShared_2142_ == 0)
{
lean_ctor_set(v___x_2141_, 1, v___f_2148_);
lean_ctor_set(v___x_2141_, 0, v___x_2150_);
v___x_2152_ = v___x_2141_;
goto v_reusejp_2151_;
}
else
{
lean_object* v_reuseFailAlloc_2224_; 
v_reuseFailAlloc_2224_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2224_, 0, v___x_2150_);
lean_ctor_set(v_reuseFailAlloc_2224_, 1, v___f_2148_);
v___x_2152_ = v_reuseFailAlloc_2224_;
goto v_reusejp_2151_;
}
v_reusejp_2151_:
{
lean_object* v___x_2154_; 
lean_inc_ref(v___x_2152_);
if (v_isShared_2050_ == 0)
{
lean_ctor_set(v___x_2049_, 1, v___f_2130_);
lean_ctor_set(v___x_2049_, 0, v___x_2152_);
v___x_2154_ = v___x_2049_;
goto v_reusejp_2153_;
}
else
{
lean_object* v_reuseFailAlloc_2223_; 
v_reuseFailAlloc_2223_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2223_, 0, v___x_2152_);
lean_ctor_set(v_reuseFailAlloc_2223_, 1, v___f_2130_);
v___x_2154_ = v_reuseFailAlloc_2223_;
goto v_reusejp_2153_;
}
v_reusejp_2153_:
{
lean_object* v___x_2155_; lean_object* v_toPartialOrder_2156_; lean_object* v_toLE_2157_; lean_object* v_toLT_2158_; lean_object* v___x_2160_; uint8_t v_isShared_2161_; uint8_t v_isSharedCheck_2222_; 
v___x_2155_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2154_);
v_toPartialOrder_2156_ = lean_ctor_get(v___x_2155_, 0);
lean_inc_ref(v_toPartialOrder_2156_);
v_toLE_2157_ = lean_ctor_get(v_toPartialOrder_2156_, 0);
v_toLT_2158_ = lean_ctor_get(v_toPartialOrder_2156_, 1);
v_isSharedCheck_2222_ = !lean_is_exclusive(v_toPartialOrder_2156_);
if (v_isSharedCheck_2222_ == 0)
{
v___x_2160_ = v_toPartialOrder_2156_;
v_isShared_2161_ = v_isSharedCheck_2222_;
goto v_resetjp_2159_;
}
else
{
lean_inc(v_toLT_2158_);
lean_inc(v_toLE_2157_);
lean_dec(v_toPartialOrder_2156_);
v___x_2160_ = lean_box(0);
v_isShared_2161_ = v_isSharedCheck_2222_;
goto v_resetjp_2159_;
}
v_resetjp_2159_:
{
lean_object* v_bot_2162_; lean_object* v_hnot_2163_; lean_object* v_sdiff_2164_; lean_object* v_top_2165_; lean_object* v___f_2166_; lean_object* v___f_2167_; lean_object* v_coheytingAlgebra_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v_toPartialOrder_2171_; lean_object* v_toInfSet_2172_; lean_object* v___x_2174_; uint8_t v_isShared_2175_; uint8_t v_isSharedCheck_2221_; 
lean_inc_n(v_toFun_2045_, 3);
v_bot_2162_ = lean_apply_1(v_toFun_2045_, v_toOrderBot_2112_);
lean_inc_ref(v_e_2034_);
v_hnot_2163_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__17), 4, 3);
lean_closure_set(v_hnot_2163_, 0, v_e_2034_);
lean_closure_set(v_hnot_2163_, 1, v_toHNot_2111_);
lean_closure_set(v_hnot_2163_, 2, v_toFun_2045_);
v_sdiff_2164_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_2164_, 0, v___f_2061_);
lean_closure_set(v_sdiff_2164_, 1, v_e_2034_);
lean_closure_set(v_sdiff_2164_, 2, v_toSDiff_2113_);
lean_closure_set(v_sdiff_2164_, 3, v_toFun_2045_);
v_top_2165_ = lean_apply_1(v_toFun_2045_, v_toOrderTop_2110_);
v___f_2166_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2166_, 0, v___x_2155_);
v___f_2167_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2167_, 0, v___x_2152_);
v_coheytingAlgebra_2168_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2166_, v___f_2167_, v_toLE_2157_, v_toLT_2158_, v_bot_2162_, v_top_2165_, v_hnot_2163_, v_sdiff_2164_);
v___x_2169_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2103_);
lean_inc_ref(v_completeLattice_2106_);
v___x_2170_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_2106_);
v_toPartialOrder_2171_ = lean_ctor_get(v___x_2170_, 0);
v_toInfSet_2172_ = lean_ctor_get(v___x_2170_, 1);
v_isSharedCheck_2221_ = !lean_is_exclusive(v___x_2170_);
if (v_isSharedCheck_2221_ == 0)
{
v___x_2174_ = v___x_2170_;
v_isShared_2175_ = v_isSharedCheck_2221_;
goto v_resetjp_2173_;
}
else
{
lean_inc(v_toInfSet_2172_);
lean_inc(v_toPartialOrder_2171_);
lean_dec(v___x_2170_);
v___x_2174_ = lean_box(0);
v_isShared_2175_ = v_isSharedCheck_2221_;
goto v_resetjp_2173_;
}
v_resetjp_2173_:
{
lean_object* v_toLE_2176_; lean_object* v_toLT_2177_; lean_object* v___x_2179_; uint8_t v_isShared_2180_; uint8_t v_isSharedCheck_2220_; 
v_toLE_2176_ = lean_ctor_get(v_toPartialOrder_2171_, 0);
v_toLT_2177_ = lean_ctor_get(v_toPartialOrder_2171_, 1);
v_isSharedCheck_2220_ = !lean_is_exclusive(v_toPartialOrder_2171_);
if (v_isSharedCheck_2220_ == 0)
{
v___x_2179_ = v_toPartialOrder_2171_;
v_isShared_2180_ = v_isSharedCheck_2220_;
goto v_resetjp_2178_;
}
else
{
lean_inc(v_toLT_2177_);
lean_inc(v_toLE_2176_);
lean_dec(v_toPartialOrder_2171_);
v___x_2179_ = lean_box(0);
v_isShared_2180_ = v_isSharedCheck_2220_;
goto v_resetjp_2178_;
}
v_resetjp_2178_:
{
lean_object* v___x_2181_; lean_object* v_toGeneralizedCoheytingAlgebra_2182_; lean_object* v_toSupSet_2183_; lean_object* v___x_2185_; uint8_t v_isShared_2186_; uint8_t v_isSharedCheck_2218_; 
v___x_2181_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_2106_);
v_toGeneralizedCoheytingAlgebra_2182_ = lean_ctor_get(v_coheytingAlgebra_2168_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2182_);
v_toSupSet_2183_ = lean_ctor_get(v___x_2181_, 1);
v_isSharedCheck_2218_ = !lean_is_exclusive(v___x_2181_);
if (v_isSharedCheck_2218_ == 0)
{
lean_object* v_unused_2219_; 
v_unused_2219_ = lean_ctor_get(v___x_2181_, 0);
lean_dec(v_unused_2219_);
v___x_2185_ = v___x_2181_;
v_isShared_2186_ = v_isSharedCheck_2218_;
goto v_resetjp_2184_;
}
else
{
lean_inc(v_toSupSet_2183_);
lean_dec(v___x_2181_);
v___x_2185_ = lean_box(0);
v_isShared_2186_ = v_isSharedCheck_2218_;
goto v_resetjp_2184_;
}
v_resetjp_2184_:
{
lean_object* v_toOrderTop_2187_; lean_object* v_toHNot_2188_; lean_object* v_toSDiff_2189_; lean_object* v___f_2190_; lean_object* v___f_2191_; lean_object* v___x_2193_; 
v_toOrderTop_2187_ = lean_ctor_get(v_coheytingAlgebra_2168_, 1);
lean_inc(v_toOrderTop_2187_);
v_toHNot_2188_ = lean_ctor_get(v_coheytingAlgebra_2168_, 2);
lean_inc(v_toHNot_2188_);
lean_dec_ref(v_coheytingAlgebra_2168_);
v_toSDiff_2189_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2182_, 2);
lean_inc(v_toSDiff_2189_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_2182_);
v___f_2190_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2190_, 0, v___x_2101_);
v___f_2191_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2191_, 0, v___x_2169_);
if (v_isShared_2180_ == 0)
{
v___x_2193_ = v___x_2179_;
goto v_reusejp_2192_;
}
else
{
lean_object* v_reuseFailAlloc_2217_; 
v_reuseFailAlloc_2217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2217_, 0, v_toLE_2176_);
lean_ctor_set(v_reuseFailAlloc_2217_, 1, v_toLT_2177_);
v___x_2193_ = v_reuseFailAlloc_2217_;
goto v_reusejp_2192_;
}
v_reusejp_2192_:
{
lean_object* v___x_2195_; 
if (v_isShared_2186_ == 0)
{
lean_ctor_set(v___x_2185_, 1, v___f_2097_);
lean_ctor_set(v___x_2185_, 0, v___x_2193_);
v___x_2195_ = v___x_2185_;
goto v_reusejp_2194_;
}
else
{
lean_object* v_reuseFailAlloc_2216_; 
v_reuseFailAlloc_2216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2216_, 0, v___x_2193_);
lean_ctor_set(v_reuseFailAlloc_2216_, 1, v___f_2097_);
v___x_2195_ = v_reuseFailAlloc_2216_;
goto v_reusejp_2194_;
}
v_reusejp_2194_:
{
lean_object* v___x_2197_; 
if (v_isShared_2175_ == 0)
{
lean_ctor_set(v___x_2174_, 1, v___f_2075_);
lean_ctor_set(v___x_2174_, 0, v___x_2195_);
v___x_2197_ = v___x_2174_;
goto v_reusejp_2196_;
}
else
{
lean_object* v_reuseFailAlloc_2215_; 
v_reuseFailAlloc_2215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2215_, 0, v___x_2195_);
lean_ctor_set(v_reuseFailAlloc_2215_, 1, v___f_2075_);
v___x_2197_ = v_reuseFailAlloc_2215_;
goto v_reusejp_2196_;
}
v_reusejp_2196_:
{
lean_object* v___x_2199_; 
lean_inc(v_bot_2094_);
lean_inc(v_toOrderTop_2187_);
if (v_isShared_2161_ == 0)
{
lean_ctor_set(v___x_2160_, 1, v_bot_2094_);
lean_ctor_set(v___x_2160_, 0, v_toOrderTop_2187_);
v___x_2199_ = v___x_2160_;
goto v_reusejp_2198_;
}
else
{
lean_object* v_reuseFailAlloc_2214_; 
v_reuseFailAlloc_2214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2214_, 0, v_toOrderTop_2187_);
lean_ctor_set(v_reuseFailAlloc_2214_, 1, v_bot_2094_);
v___x_2199_ = v_reuseFailAlloc_2214_;
goto v_reusejp_2198_;
}
v_reusejp_2198_:
{
lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v_toGeneralizedCoheytingAlgebra_2202_; lean_object* v_toHNot_2203_; lean_object* v_toSDiff_2204_; lean_object* v___x_2206_; uint8_t v_isShared_2207_; uint8_t v_isSharedCheck_2211_; 
v___x_2200_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2200_, 0, v___x_2197_);
lean_ctor_set(v___x_2200_, 1, v_toSupSet_2183_);
lean_ctor_set(v___x_2200_, 2, v_toInfSet_2172_);
lean_ctor_set(v___x_2200_, 3, v___x_2199_);
v___x_2201_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2191_, v___f_2190_, v_toLE_2176_, v_toLT_2177_, v_bot_2094_, v_toOrderTop_2187_, v_toHNot_2188_, v_toSDiff_2189_);
v_toGeneralizedCoheytingAlgebra_2202_ = lean_ctor_get(v___x_2201_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2202_);
v_toHNot_2203_ = lean_ctor_get(v___x_2201_, 2);
lean_inc(v_toHNot_2203_);
lean_dec_ref(v___x_2201_);
v_toSDiff_2204_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2202_, 2);
v_isSharedCheck_2211_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_2202_);
if (v_isSharedCheck_2211_ == 0)
{
lean_object* v_unused_2212_; lean_object* v_unused_2213_; 
v_unused_2212_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2202_, 1);
lean_dec(v_unused_2212_);
v_unused_2213_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2202_, 0);
lean_dec(v_unused_2213_);
v___x_2206_ = v_toGeneralizedCoheytingAlgebra_2202_;
v_isShared_2207_ = v_isSharedCheck_2211_;
goto v_resetjp_2205_;
}
else
{
lean_inc(v_toSDiff_2204_);
lean_dec(v_toGeneralizedCoheytingAlgebra_2202_);
v___x_2206_ = lean_box(0);
v_isShared_2207_ = v_isSharedCheck_2211_;
goto v_resetjp_2205_;
}
v_resetjp_2205_:
{
lean_object* v___x_2209_; 
if (v_isShared_2207_ == 0)
{
lean_ctor_set(v___x_2206_, 2, v_toHNot_2203_);
lean_ctor_set(v___x_2206_, 1, v_toSDiff_2204_);
lean_ctor_set(v___x_2206_, 0, v___x_2200_);
v___x_2209_ = v___x_2206_;
goto v_reusejp_2208_;
}
else
{
lean_object* v_reuseFailAlloc_2210_; 
v_reuseFailAlloc_2210_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2210_, 0, v___x_2200_);
lean_ctor_set(v_reuseFailAlloc_2210_, 1, v_toSDiff_2204_);
lean_ctor_set(v_reuseFailAlloc_2210_, 2, v_toHNot_2203_);
v___x_2209_ = v_reuseFailAlloc_2210_;
goto v_reusejp_2208_;
}
v_reusejp_2208_:
{
return v___x_2209_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeDistribLattice___redArg(lean_object* v_e_2255_, lean_object* v_inst_2256_){
_start:
{
lean_object* v___x_2257_; lean_object* v_toCompleteLattice_2258_; lean_object* v_toBoundedOrder_2259_; lean_object* v_toLattice_2260_; lean_object* v_toOrderTop_2261_; lean_object* v_toOrderBot_2262_; lean_object* v___x_2264_; uint8_t v_isShared_2265_; uint8_t v_isSharedCheck_2552_; 
lean_inc_ref(v_inst_2256_);
v___x_2257_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v_inst_2256_);
v_toCompleteLattice_2258_ = lean_ctor_get(v___x_2257_, 0);
lean_inc_ref(v_toCompleteLattice_2258_);
lean_dec_ref(v___x_2257_);
v_toBoundedOrder_2259_ = lean_ctor_get(v_toCompleteLattice_2258_, 3);
lean_inc_ref(v_toBoundedOrder_2259_);
v_toLattice_2260_ = lean_ctor_get(v_toCompleteLattice_2258_, 0);
lean_inc_ref(v_toLattice_2260_);
v_toOrderTop_2261_ = lean_ctor_get(v_toBoundedOrder_2259_, 0);
v_toOrderBot_2262_ = lean_ctor_get(v_toBoundedOrder_2259_, 1);
v_isSharedCheck_2552_ = !lean_is_exclusive(v_toBoundedOrder_2259_);
if (v_isSharedCheck_2552_ == 0)
{
v___x_2264_ = v_toBoundedOrder_2259_;
v_isShared_2265_ = v_isSharedCheck_2552_;
goto v_resetjp_2263_;
}
else
{
lean_inc(v_toOrderBot_2262_);
lean_inc(v_toOrderTop_2261_);
lean_dec(v_toBoundedOrder_2259_);
v___x_2264_ = lean_box(0);
v_isShared_2265_ = v_isSharedCheck_2552_;
goto v_resetjp_2263_;
}
v_resetjp_2263_:
{
lean_object* v___x_2266_; lean_object* v_toFun_2267_; lean_object* v___x_2269_; uint8_t v_isShared_2270_; uint8_t v_isSharedCheck_2550_; 
lean_inc_ref(v_e_2255_);
v___x_2266_ = lp_mathlib_Equiv_symm___redArg(v_e_2255_);
v_toFun_2267_ = lean_ctor_get(v___x_2266_, 0);
v_isSharedCheck_2550_ = !lean_is_exclusive(v___x_2266_);
if (v_isSharedCheck_2550_ == 0)
{
lean_object* v_unused_2551_; 
v_unused_2551_ = lean_ctor_get(v___x_2266_, 1);
lean_dec(v_unused_2551_);
v___x_2269_ = v___x_2266_;
v_isShared_2270_ = v_isSharedCheck_2550_;
goto v_resetjp_2268_;
}
else
{
lean_inc(v_toFun_2267_);
lean_dec(v___x_2266_);
v___x_2269_ = lean_box(0);
v_isShared_2270_ = v_isSharedCheck_2550_;
goto v_resetjp_2268_;
}
v_resetjp_2268_:
{
lean_object* v___x_2271_; lean_object* v_toSupSet_2272_; lean_object* v___x_2274_; uint8_t v_isShared_2275_; uint8_t v_isSharedCheck_2548_; 
lean_inc_ref(v_toCompleteLattice_2258_);
v___x_2271_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_2258_);
v_toSupSet_2272_ = lean_ctor_get(v___x_2271_, 1);
v_isSharedCheck_2548_ = !lean_is_exclusive(v___x_2271_);
if (v_isSharedCheck_2548_ == 0)
{
lean_object* v_unused_2549_; 
v_unused_2549_ = lean_ctor_get(v___x_2271_, 0);
lean_dec(v_unused_2549_);
v___x_2274_ = v___x_2271_;
v_isShared_2275_ = v_isSharedCheck_2548_;
goto v_resetjp_2273_;
}
else
{
lean_inc(v_toSupSet_2272_);
lean_dec(v___x_2271_);
v___x_2274_ = lean_box(0);
v_isShared_2275_ = v_isSharedCheck_2548_;
goto v_resetjp_2273_;
}
v_resetjp_2273_:
{
lean_object* v___x_2276_; lean_object* v_toInfSet_2277_; lean_object* v___x_2279_; uint8_t v_isShared_2280_; uint8_t v_isSharedCheck_2546_; 
v___x_2276_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_2258_);
v_toInfSet_2277_ = lean_ctor_get(v___x_2276_, 1);
v_isSharedCheck_2546_ = !lean_is_exclusive(v___x_2276_);
if (v_isSharedCheck_2546_ == 0)
{
lean_object* v_unused_2547_; 
v_unused_2547_ = lean_ctor_get(v___x_2276_, 0);
lean_dec(v_unused_2547_);
v___x_2279_ = v___x_2276_;
v_isShared_2280_ = v_isSharedCheck_2546_;
goto v_resetjp_2278_;
}
else
{
lean_inc(v_toInfSet_2277_);
lean_dec(v___x_2276_);
v___x_2279_ = lean_box(0);
v_isShared_2280_ = v_isSharedCheck_2546_;
goto v_resetjp_2278_;
}
v_resetjp_2278_:
{
lean_object* v_toSemilatticeSup_2281_; lean_object* v_inf_2282_; lean_object* v___x_2284_; uint8_t v_isShared_2285_; uint8_t v_isSharedCheck_2545_; 
v_toSemilatticeSup_2281_ = lean_ctor_get(v_toLattice_2260_, 0);
v_inf_2282_ = lean_ctor_get(v_toLattice_2260_, 1);
v_isSharedCheck_2545_ = !lean_is_exclusive(v_toLattice_2260_);
if (v_isSharedCheck_2545_ == 0)
{
v___x_2284_ = v_toLattice_2260_;
v_isShared_2285_ = v_isSharedCheck_2545_;
goto v_resetjp_2283_;
}
else
{
lean_inc(v_inf_2282_);
lean_inc(v_toSemilatticeSup_2281_);
lean_dec(v_toLattice_2260_);
v___x_2284_ = lean_box(0);
v_isShared_2285_ = v_isSharedCheck_2545_;
goto v_resetjp_2283_;
}
v_resetjp_2283_:
{
lean_object* v___f_2286_; lean_object* v_min_2287_; lean_object* v_le_2288_; lean_object* v_lt_2289_; lean_object* v_semilatticeInf_2290_; lean_object* v_toPartialOrder_2291_; lean_object* v___x_2293_; uint8_t v_isShared_2294_; uint8_t v_isSharedCheck_2543_; 
v___f_2286_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_2267_);
lean_inc_ref(v_e_2255_);
v_min_2287_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2287_, 0, v___f_2286_);
lean_closure_set(v_min_2287_, 1, v_e_2255_);
lean_closure_set(v_min_2287_, 2, v_inf_2282_);
lean_closure_set(v_min_2287_, 3, v_toFun_2267_);
v_le_2288_ = lean_box(0);
v_lt_2289_ = lean_box(0);
lean_inc_ref(v_min_2287_);
v_semilatticeInf_2290_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2287_, v_le_2288_, v_lt_2289_);
v_toPartialOrder_2291_ = lean_ctor_get(v_semilatticeInf_2290_, 0);
v_isSharedCheck_2543_ = !lean_is_exclusive(v_semilatticeInf_2290_);
if (v_isSharedCheck_2543_ == 0)
{
lean_object* v_unused_2544_; 
v_unused_2544_ = lean_ctor_get(v_semilatticeInf_2290_, 1);
lean_dec(v_unused_2544_);
v___x_2293_ = v_semilatticeInf_2290_;
v_isShared_2294_ = v_isSharedCheck_2543_;
goto v_resetjp_2292_;
}
else
{
lean_inc(v_toPartialOrder_2291_);
lean_dec(v_semilatticeInf_2290_);
v___x_2293_ = lean_box(0);
v_isShared_2294_ = v_isSharedCheck_2543_;
goto v_resetjp_2292_;
}
v_resetjp_2292_:
{
lean_object* v_toLE_2295_; lean_object* v_toLT_2296_; lean_object* v___x_2298_; uint8_t v_isShared_2299_; uint8_t v_isSharedCheck_2542_; 
v_toLE_2295_ = lean_ctor_get(v_toPartialOrder_2291_, 0);
v_toLT_2296_ = lean_ctor_get(v_toPartialOrder_2291_, 1);
v_isSharedCheck_2542_ = !lean_is_exclusive(v_toPartialOrder_2291_);
if (v_isSharedCheck_2542_ == 0)
{
v___x_2298_ = v_toPartialOrder_2291_;
v_isShared_2299_ = v_isSharedCheck_2542_;
goto v_resetjp_2297_;
}
else
{
lean_inc(v_toLT_2296_);
lean_inc(v_toLE_2295_);
lean_dec(v_toPartialOrder_2291_);
v___x_2298_ = lean_box(0);
v_isShared_2299_ = v_isSharedCheck_2542_;
goto v_resetjp_2297_;
}
v_resetjp_2297_:
{
lean_object* v___f_2300_; lean_object* v___f_2301_; lean_object* v___x_2303_; 
v___f_2300_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_2300_, 0, v_min_2287_);
lean_inc(v_toFun_2267_);
lean_inc_ref(v_e_2255_);
v___f_2301_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_2301_, 0, v_toSemilatticeSup_2281_);
lean_closure_set(v___f_2301_, 1, v___f_2286_);
lean_closure_set(v___f_2301_, 2, v_e_2255_);
lean_closure_set(v___f_2301_, 3, v_toFun_2267_);
if (v_isShared_2299_ == 0)
{
v___x_2303_ = v___x_2298_;
goto v_reusejp_2302_;
}
else
{
lean_object* v_reuseFailAlloc_2541_; 
v_reuseFailAlloc_2541_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2541_, 0, v_toLE_2295_);
lean_ctor_set(v_reuseFailAlloc_2541_, 1, v_toLT_2296_);
v___x_2303_ = v_reuseFailAlloc_2541_;
goto v_reusejp_2302_;
}
v_reusejp_2302_:
{
lean_object* v___x_2305_; 
lean_inc_ref(v___f_2301_);
if (v_isShared_2294_ == 0)
{
lean_ctor_set(v___x_2293_, 1, v___f_2301_);
lean_ctor_set(v___x_2293_, 0, v___x_2303_);
v___x_2305_ = v___x_2293_;
goto v_reusejp_2304_;
}
else
{
lean_object* v_reuseFailAlloc_2540_; 
v_reuseFailAlloc_2540_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2540_, 0, v___x_2303_);
lean_ctor_set(v_reuseFailAlloc_2540_, 1, v___f_2301_);
v___x_2305_ = v_reuseFailAlloc_2540_;
goto v_reusejp_2304_;
}
v_reusejp_2304_:
{
lean_object* v_lattice_2307_; 
lean_inc_ref(v___f_2300_);
if (v_isShared_2285_ == 0)
{
lean_ctor_set(v___x_2284_, 1, v___f_2300_);
lean_ctor_set(v___x_2284_, 0, v___x_2305_);
v_lattice_2307_ = v___x_2284_;
goto v_reusejp_2306_;
}
else
{
lean_object* v_reuseFailAlloc_2539_; 
v_reuseFailAlloc_2539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2539_, 0, v___x_2305_);
lean_ctor_set(v_reuseFailAlloc_2539_, 1, v___f_2300_);
v_lattice_2307_ = v_reuseFailAlloc_2539_;
goto v_reusejp_2306_;
}
v_reusejp_2306_:
{
lean_object* v___x_2308_; lean_object* v_toPartialOrder_2309_; lean_object* v___x_2311_; uint8_t v_isShared_2312_; uint8_t v_isSharedCheck_2537_; 
v___x_2308_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2307_);
v_toPartialOrder_2309_ = lean_ctor_get(v___x_2308_, 0);
v_isSharedCheck_2537_ = !lean_is_exclusive(v___x_2308_);
if (v_isSharedCheck_2537_ == 0)
{
lean_object* v_unused_2538_; 
v_unused_2538_ = lean_ctor_get(v___x_2308_, 1);
lean_dec(v_unused_2538_);
v___x_2311_ = v___x_2308_;
v_isShared_2312_ = v_isSharedCheck_2537_;
goto v_resetjp_2310_;
}
else
{
lean_inc(v_toPartialOrder_2309_);
lean_dec(v___x_2308_);
v___x_2311_ = lean_box(0);
v_isShared_2312_ = v_isSharedCheck_2537_;
goto v_resetjp_2310_;
}
v_resetjp_2310_:
{
lean_object* v_toLE_2313_; lean_object* v_toLT_2314_; lean_object* v___x_2316_; uint8_t v_isShared_2317_; uint8_t v_isSharedCheck_2536_; 
v_toLE_2313_ = lean_ctor_get(v_toPartialOrder_2309_, 0);
v_toLT_2314_ = lean_ctor_get(v_toPartialOrder_2309_, 1);
v_isSharedCheck_2536_ = !lean_is_exclusive(v_toPartialOrder_2309_);
if (v_isSharedCheck_2536_ == 0)
{
v___x_2316_ = v_toPartialOrder_2309_;
v_isShared_2317_ = v_isSharedCheck_2536_;
goto v_resetjp_2315_;
}
else
{
lean_inc(v_toLT_2314_);
lean_inc(v_toLE_2313_);
lean_dec(v_toPartialOrder_2309_);
v___x_2316_ = lean_box(0);
v_isShared_2317_ = v_isSharedCheck_2536_;
goto v_resetjp_2315_;
}
v_resetjp_2315_:
{
lean_object* v_top_2318_; lean_object* v_bot_2319_; lean_object* v_supSet_2320_; lean_object* v_infSet_2321_; lean_object* v___f_2322_; lean_object* v___x_2324_; 
lean_inc_n(v_toFun_2267_, 4);
v_top_2318_ = lean_apply_1(v_toFun_2267_, v_toOrderTop_2261_);
v_bot_2319_ = lean_apply_1(v_toFun_2267_, v_toOrderBot_2262_);
v_supSet_2320_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_2320_, 0, v_toSupSet_2272_);
lean_closure_set(v_supSet_2320_, 1, v_toFun_2267_);
v_infSet_2321_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_2321_, 0, v_toInfSet_2277_);
lean_closure_set(v_infSet_2321_, 1, v_toFun_2267_);
v___f_2322_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_2322_, 0, v___f_2301_);
if (v_isShared_2317_ == 0)
{
v___x_2324_ = v___x_2316_;
goto v_reusejp_2323_;
}
else
{
lean_object* v_reuseFailAlloc_2535_; 
v_reuseFailAlloc_2535_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2535_, 0, v_toLE_2313_);
lean_ctor_set(v_reuseFailAlloc_2535_, 1, v_toLT_2314_);
v___x_2324_ = v_reuseFailAlloc_2535_;
goto v_reusejp_2323_;
}
v_reusejp_2323_:
{
lean_object* v___x_2326_; 
lean_inc_ref(v___f_2322_);
if (v_isShared_2312_ == 0)
{
lean_ctor_set(v___x_2311_, 1, v___f_2322_);
lean_ctor_set(v___x_2311_, 0, v___x_2324_);
v___x_2326_ = v___x_2311_;
goto v_reusejp_2325_;
}
else
{
lean_object* v_reuseFailAlloc_2534_; 
v_reuseFailAlloc_2534_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2534_, 0, v___x_2324_);
lean_ctor_set(v_reuseFailAlloc_2534_, 1, v___f_2322_);
v___x_2326_ = v_reuseFailAlloc_2534_;
goto v_reusejp_2325_;
}
v_reusejp_2325_:
{
lean_object* v___x_2328_; 
lean_inc_ref(v___f_2300_);
lean_inc_ref(v___x_2326_);
if (v_isShared_2280_ == 0)
{
lean_ctor_set(v___x_2279_, 1, v___f_2300_);
lean_ctor_set(v___x_2279_, 0, v___x_2326_);
v___x_2328_ = v___x_2279_;
goto v_reusejp_2327_;
}
else
{
lean_object* v_reuseFailAlloc_2533_; 
v_reuseFailAlloc_2533_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2533_, 0, v___x_2326_);
lean_ctor_set(v_reuseFailAlloc_2533_, 1, v___f_2300_);
v___x_2328_ = v_reuseFailAlloc_2533_;
goto v_reusejp_2327_;
}
v_reusejp_2327_:
{
lean_object* v___x_2330_; 
if (v_isShared_2265_ == 0)
{
lean_ctor_set(v___x_2264_, 1, v_bot_2319_);
lean_ctor_set(v___x_2264_, 0, v_top_2318_);
v___x_2330_ = v___x_2264_;
goto v_reusejp_2329_;
}
else
{
lean_object* v_reuseFailAlloc_2532_; 
v_reuseFailAlloc_2532_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2532_, 0, v_top_2318_);
lean_ctor_set(v_reuseFailAlloc_2532_, 1, v_bot_2319_);
v___x_2330_ = v_reuseFailAlloc_2532_;
goto v_reusejp_2329_;
}
v_reusejp_2329_:
{
lean_object* v_completeLattice_2331_; lean_object* v___x_2332_; lean_object* v_toHeytingAlgebra_2333_; lean_object* v_toGeneralizedHeytingAlgebra_2334_; lean_object* v_toOrderBot_2335_; lean_object* v_toCompl_2336_; lean_object* v___x_2338_; uint8_t v_isShared_2339_; uint8_t v_isSharedCheck_2531_; 
lean_inc_ref(v___x_2328_);
v_completeLattice_2331_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_2331_, 0, v___x_2328_);
lean_ctor_set(v_completeLattice_2331_, 1, v_supSet_2320_);
lean_ctor_set(v_completeLattice_2331_, 2, v_infSet_2321_);
lean_ctor_set(v_completeLattice_2331_, 3, v___x_2330_);
v___x_2332_ = lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra___redArg(v_inst_2256_);
v_toHeytingAlgebra_2333_ = lean_ctor_get(v___x_2332_, 0);
lean_inc_ref(v_toHeytingAlgebra_2333_);
v_toGeneralizedHeytingAlgebra_2334_ = lean_ctor_get(v_toHeytingAlgebra_2333_, 0);
v_toOrderBot_2335_ = lean_ctor_get(v_toHeytingAlgebra_2333_, 1);
v_toCompl_2336_ = lean_ctor_get(v_toHeytingAlgebra_2333_, 2);
v_isSharedCheck_2531_ = !lean_is_exclusive(v_toHeytingAlgebra_2333_);
if (v_isSharedCheck_2531_ == 0)
{
v___x_2338_ = v_toHeytingAlgebra_2333_;
v_isShared_2339_ = v_isSharedCheck_2531_;
goto v_resetjp_2337_;
}
else
{
lean_inc(v_toCompl_2336_);
lean_inc(v_toOrderBot_2335_);
lean_inc(v_toGeneralizedHeytingAlgebra_2334_);
lean_dec(v_toHeytingAlgebra_2333_);
v___x_2338_ = lean_box(0);
v_isShared_2339_ = v_isSharedCheck_2531_;
goto v_resetjp_2337_;
}
v_resetjp_2337_:
{
lean_object* v_toHImp_2340_; lean_object* v___x_2342_; uint8_t v_isShared_2343_; uint8_t v_isSharedCheck_2528_; 
v_toHImp_2340_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2334_, 2);
v_isSharedCheck_2528_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_2334_);
if (v_isSharedCheck_2528_ == 0)
{
lean_object* v_unused_2529_; lean_object* v_unused_2530_; 
v_unused_2529_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2334_, 1);
lean_dec(v_unused_2529_);
v_unused_2530_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2334_, 0);
lean_dec(v_unused_2530_);
v___x_2342_ = v_toGeneralizedHeytingAlgebra_2334_;
v_isShared_2343_ = v_isSharedCheck_2528_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_toHImp_2340_);
lean_dec(v_toGeneralizedHeytingAlgebra_2334_);
v___x_2342_ = lean_box(0);
v_isShared_2343_ = v_isSharedCheck_2528_;
goto v_resetjp_2341_;
}
v_resetjp_2341_:
{
lean_object* v___x_2344_; lean_object* v_toGeneralizedCoheytingAlgebra_2345_; lean_object* v_toLattice_2346_; lean_object* v_toOrderTop_2347_; lean_object* v_toHNot_2348_; lean_object* v_toOrderBot_2349_; lean_object* v_toSDiff_2350_; lean_object* v_toSemilatticeSup_2351_; lean_object* v_inf_2352_; lean_object* v___x_2354_; uint8_t v_isShared_2355_; uint8_t v_isSharedCheck_2527_; 
v___x_2344_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v___x_2332_);
v_toGeneralizedCoheytingAlgebra_2345_ = lean_ctor_get(v___x_2344_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2345_);
v_toLattice_2346_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2345_, 0);
lean_inc_ref(v_toLattice_2346_);
v_toOrderTop_2347_ = lean_ctor_get(v___x_2344_, 1);
lean_inc(v_toOrderTop_2347_);
v_toHNot_2348_ = lean_ctor_get(v___x_2344_, 2);
lean_inc(v_toHNot_2348_);
lean_dec_ref(v___x_2344_);
v_toOrderBot_2349_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2345_, 1);
lean_inc(v_toOrderBot_2349_);
v_toSDiff_2350_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2345_, 2);
lean_inc(v_toSDiff_2350_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_2345_);
v_toSemilatticeSup_2351_ = lean_ctor_get(v_toLattice_2346_, 0);
v_inf_2352_ = lean_ctor_get(v_toLattice_2346_, 1);
v_isSharedCheck_2527_ = !lean_is_exclusive(v_toLattice_2346_);
if (v_isSharedCheck_2527_ == 0)
{
v___x_2354_ = v_toLattice_2346_;
v_isShared_2355_ = v_isSharedCheck_2527_;
goto v_resetjp_2353_;
}
else
{
lean_inc(v_inf_2352_);
lean_inc(v_toSemilatticeSup_2351_);
lean_dec(v_toLattice_2346_);
v___x_2354_ = lean_box(0);
v_isShared_2355_ = v_isSharedCheck_2527_;
goto v_resetjp_2353_;
}
v_resetjp_2353_:
{
lean_object* v_min_2356_; lean_object* v_semilatticeInf_2357_; lean_object* v_toPartialOrder_2358_; lean_object* v___x_2360_; uint8_t v_isShared_2361_; uint8_t v_isSharedCheck_2525_; 
lean_inc(v_toFun_2267_);
lean_inc_ref(v_e_2255_);
v_min_2356_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2356_, 0, v___f_2286_);
lean_closure_set(v_min_2356_, 1, v_e_2255_);
lean_closure_set(v_min_2356_, 2, v_inf_2352_);
lean_closure_set(v_min_2356_, 3, v_toFun_2267_);
lean_inc_ref(v_min_2356_);
v_semilatticeInf_2357_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2356_, v_le_2288_, v_lt_2289_);
v_toPartialOrder_2358_ = lean_ctor_get(v_semilatticeInf_2357_, 0);
v_isSharedCheck_2525_ = !lean_is_exclusive(v_semilatticeInf_2357_);
if (v_isSharedCheck_2525_ == 0)
{
lean_object* v_unused_2526_; 
v_unused_2526_ = lean_ctor_get(v_semilatticeInf_2357_, 1);
lean_dec(v_unused_2526_);
v___x_2360_ = v_semilatticeInf_2357_;
v_isShared_2361_ = v_isSharedCheck_2525_;
goto v_resetjp_2359_;
}
else
{
lean_inc(v_toPartialOrder_2358_);
lean_dec(v_semilatticeInf_2357_);
v___x_2360_ = lean_box(0);
v_isShared_2361_ = v_isSharedCheck_2525_;
goto v_resetjp_2359_;
}
v_resetjp_2359_:
{
lean_object* v_toLE_2362_; lean_object* v_toLT_2363_; lean_object* v___x_2365_; uint8_t v_isShared_2366_; uint8_t v_isSharedCheck_2524_; 
v_toLE_2362_ = lean_ctor_get(v_toPartialOrder_2358_, 0);
v_toLT_2363_ = lean_ctor_get(v_toPartialOrder_2358_, 1);
v_isSharedCheck_2524_ = !lean_is_exclusive(v_toPartialOrder_2358_);
if (v_isSharedCheck_2524_ == 0)
{
v___x_2365_ = v_toPartialOrder_2358_;
v_isShared_2366_ = v_isSharedCheck_2524_;
goto v_resetjp_2364_;
}
else
{
lean_inc(v_toLT_2363_);
lean_inc(v_toLE_2362_);
lean_dec(v_toPartialOrder_2358_);
v___x_2365_ = lean_box(0);
v_isShared_2366_ = v_isSharedCheck_2524_;
goto v_resetjp_2364_;
}
v_resetjp_2364_:
{
lean_object* v___f_2367_; lean_object* v___f_2368_; lean_object* v___x_2370_; 
v___f_2367_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_2367_, 0, v_min_2356_);
lean_inc(v_toFun_2267_);
lean_inc_ref(v_e_2255_);
v___f_2368_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_2368_, 0, v_toSemilatticeSup_2351_);
lean_closure_set(v___f_2368_, 1, v___f_2286_);
lean_closure_set(v___f_2368_, 2, v_e_2255_);
lean_closure_set(v___f_2368_, 3, v_toFun_2267_);
if (v_isShared_2366_ == 0)
{
v___x_2370_ = v___x_2365_;
goto v_reusejp_2369_;
}
else
{
lean_object* v_reuseFailAlloc_2523_; 
v_reuseFailAlloc_2523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2523_, 0, v_toLE_2362_);
lean_ctor_set(v_reuseFailAlloc_2523_, 1, v_toLT_2363_);
v___x_2370_ = v_reuseFailAlloc_2523_;
goto v_reusejp_2369_;
}
v_reusejp_2369_:
{
lean_object* v___x_2372_; 
lean_inc_ref(v___f_2368_);
if (v_isShared_2361_ == 0)
{
lean_ctor_set(v___x_2360_, 1, v___f_2368_);
lean_ctor_set(v___x_2360_, 0, v___x_2370_);
v___x_2372_ = v___x_2360_;
goto v_reusejp_2371_;
}
else
{
lean_object* v_reuseFailAlloc_2522_; 
v_reuseFailAlloc_2522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2522_, 0, v___x_2370_);
lean_ctor_set(v_reuseFailAlloc_2522_, 1, v___f_2368_);
v___x_2372_ = v_reuseFailAlloc_2522_;
goto v_reusejp_2371_;
}
v_reusejp_2371_:
{
lean_object* v_lattice_2374_; 
lean_inc_ref(v___f_2367_);
if (v_isShared_2355_ == 0)
{
lean_ctor_set(v___x_2354_, 1, v___f_2367_);
lean_ctor_set(v___x_2354_, 0, v___x_2372_);
v_lattice_2374_ = v___x_2354_;
goto v_reusejp_2373_;
}
else
{
lean_object* v_reuseFailAlloc_2521_; 
v_reuseFailAlloc_2521_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2521_, 0, v___x_2372_);
lean_ctor_set(v_reuseFailAlloc_2521_, 1, v___f_2367_);
v_lattice_2374_ = v_reuseFailAlloc_2521_;
goto v_reusejp_2373_;
}
v_reusejp_2373_:
{
lean_object* v___x_2375_; lean_object* v_toPartialOrder_2376_; lean_object* v___x_2378_; uint8_t v_isShared_2379_; uint8_t v_isSharedCheck_2519_; 
v___x_2375_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2374_);
v_toPartialOrder_2376_ = lean_ctor_get(v___x_2375_, 0);
v_isSharedCheck_2519_ = !lean_is_exclusive(v___x_2375_);
if (v_isSharedCheck_2519_ == 0)
{
lean_object* v_unused_2520_; 
v_unused_2520_ = lean_ctor_get(v___x_2375_, 1);
lean_dec(v_unused_2520_);
v___x_2378_ = v___x_2375_;
v_isShared_2379_ = v_isSharedCheck_2519_;
goto v_resetjp_2377_;
}
else
{
lean_inc(v_toPartialOrder_2376_);
lean_dec(v___x_2375_);
v___x_2378_ = lean_box(0);
v_isShared_2379_ = v_isSharedCheck_2519_;
goto v_resetjp_2377_;
}
v_resetjp_2377_:
{
lean_object* v_toLE_2380_; lean_object* v_toLT_2381_; lean_object* v___x_2383_; uint8_t v_isShared_2384_; uint8_t v_isSharedCheck_2518_; 
v_toLE_2380_ = lean_ctor_get(v_toPartialOrder_2376_, 0);
v_toLT_2381_ = lean_ctor_get(v_toPartialOrder_2376_, 1);
v_isSharedCheck_2518_ = !lean_is_exclusive(v_toPartialOrder_2376_);
if (v_isSharedCheck_2518_ == 0)
{
v___x_2383_ = v_toPartialOrder_2376_;
v_isShared_2384_ = v_isSharedCheck_2518_;
goto v_resetjp_2382_;
}
else
{
lean_inc(v_toLT_2381_);
lean_inc(v_toLE_2380_);
lean_dec(v_toPartialOrder_2376_);
v___x_2383_ = lean_box(0);
v_isShared_2384_ = v_isSharedCheck_2518_;
goto v_resetjp_2382_;
}
v_resetjp_2382_:
{
lean_object* v_bot_2385_; lean_object* v___f_2386_; lean_object* v___x_2388_; 
lean_inc(v_toFun_2267_);
v_bot_2385_ = lean_apply_1(v_toFun_2267_, v_toOrderBot_2335_);
v___f_2386_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_2386_, 0, v___f_2368_);
if (v_isShared_2384_ == 0)
{
v___x_2388_ = v___x_2383_;
goto v_reusejp_2387_;
}
else
{
lean_object* v_reuseFailAlloc_2517_; 
v_reuseFailAlloc_2517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2517_, 0, v_toLE_2380_);
lean_ctor_set(v_reuseFailAlloc_2517_, 1, v_toLT_2381_);
v___x_2388_ = v_reuseFailAlloc_2517_;
goto v_reusejp_2387_;
}
v_reusejp_2387_:
{
lean_object* v___x_2390_; 
lean_inc_ref(v___f_2386_);
if (v_isShared_2379_ == 0)
{
lean_ctor_set(v___x_2378_, 1, v___f_2386_);
lean_ctor_set(v___x_2378_, 0, v___x_2388_);
v___x_2390_ = v___x_2378_;
goto v_reusejp_2389_;
}
else
{
lean_object* v_reuseFailAlloc_2516_; 
v_reuseFailAlloc_2516_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2516_, 0, v___x_2388_);
lean_ctor_set(v_reuseFailAlloc_2516_, 1, v___f_2386_);
v___x_2390_ = v_reuseFailAlloc_2516_;
goto v_reusejp_2389_;
}
v_reusejp_2389_:
{
lean_object* v___x_2392_; 
lean_inc_ref(v___f_2367_);
lean_inc_ref(v___x_2390_);
if (v_isShared_2275_ == 0)
{
lean_ctor_set(v___x_2274_, 1, v___f_2367_);
lean_ctor_set(v___x_2274_, 0, v___x_2390_);
v___x_2392_ = v___x_2274_;
goto v_reusejp_2391_;
}
else
{
lean_object* v_reuseFailAlloc_2515_; 
v_reuseFailAlloc_2515_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2515_, 0, v___x_2390_);
lean_ctor_set(v_reuseFailAlloc_2515_, 1, v___f_2367_);
v___x_2392_ = v_reuseFailAlloc_2515_;
goto v_reusejp_2391_;
}
v_reusejp_2391_:
{
lean_object* v___x_2393_; lean_object* v_toPartialOrder_2394_; lean_object* v_toLE_2395_; lean_object* v_toLT_2396_; lean_object* v___x_2398_; uint8_t v_isShared_2399_; uint8_t v_isSharedCheck_2514_; 
v___x_2393_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2392_);
v_toPartialOrder_2394_ = lean_ctor_get(v___x_2393_, 0);
lean_inc_ref(v_toPartialOrder_2394_);
v_toLE_2395_ = lean_ctor_get(v_toPartialOrder_2394_, 0);
v_toLT_2396_ = lean_ctor_get(v_toPartialOrder_2394_, 1);
v_isSharedCheck_2514_ = !lean_is_exclusive(v_toPartialOrder_2394_);
if (v_isSharedCheck_2514_ == 0)
{
v___x_2398_ = v_toPartialOrder_2394_;
v_isShared_2399_ = v_isSharedCheck_2514_;
goto v_resetjp_2397_;
}
else
{
lean_inc(v_toLT_2396_);
lean_inc(v_toLE_2395_);
lean_dec(v_toPartialOrder_2394_);
v___x_2398_ = lean_box(0);
v_isShared_2399_ = v_isSharedCheck_2514_;
goto v_resetjp_2397_;
}
v_resetjp_2397_:
{
lean_object* v_bot_2400_; lean_object* v_hnot_2401_; lean_object* v_sdiff_2402_; lean_object* v_top_2403_; lean_object* v___f_2404_; lean_object* v___f_2405_; lean_object* v_coheytingAlgebra_2406_; lean_object* v_toGeneralizedCoheytingAlgebra_2407_; lean_object* v_toLattice_2408_; lean_object* v_toOrderTop_2409_; lean_object* v_toHNot_2410_; lean_object* v_toSDiff_2411_; lean_object* v_toSemilatticeSup_2412_; lean_object* v___x_2413_; lean_object* v_toPartialOrder_2414_; lean_object* v_toLE_2415_; lean_object* v_toLT_2416_; lean_object* v___x_2418_; uint8_t v_isShared_2419_; uint8_t v_isSharedCheck_2513_; 
lean_inc_n(v_toFun_2267_, 4);
v_bot_2400_ = lean_apply_1(v_toFun_2267_, v_toOrderBot_2349_);
lean_inc_ref_n(v_e_2255_, 2);
v_hnot_2401_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__17), 4, 3);
lean_closure_set(v_hnot_2401_, 0, v_e_2255_);
lean_closure_set(v_hnot_2401_, 1, v_toHNot_2348_);
lean_closure_set(v_hnot_2401_, 2, v_toFun_2267_);
v_sdiff_2402_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_2402_, 0, v___f_2286_);
lean_closure_set(v_sdiff_2402_, 1, v_e_2255_);
lean_closure_set(v_sdiff_2402_, 2, v_toSDiff_2350_);
lean_closure_set(v_sdiff_2402_, 3, v_toFun_2267_);
v_top_2403_ = lean_apply_1(v_toFun_2267_, v_toOrderTop_2347_);
v___f_2404_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2404_, 0, v___x_2393_);
v___f_2405_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2405_, 0, v___x_2390_);
v_coheytingAlgebra_2406_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2404_, v___f_2405_, v_toLE_2395_, v_toLT_2396_, v_bot_2400_, v_top_2403_, v_hnot_2401_, v_sdiff_2402_);
v_toGeneralizedCoheytingAlgebra_2407_ = lean_ctor_get(v_coheytingAlgebra_2406_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2407_);
v_toLattice_2408_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2407_, 0);
lean_inc_ref(v_toLattice_2408_);
v_toOrderTop_2409_ = lean_ctor_get(v_coheytingAlgebra_2406_, 1);
lean_inc(v_toOrderTop_2409_);
v_toHNot_2410_ = lean_ctor_get(v_coheytingAlgebra_2406_, 2);
lean_inc(v_toHNot_2410_);
lean_dec_ref(v_coheytingAlgebra_2406_);
v_toSDiff_2411_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2407_, 2);
lean_inc(v_toSDiff_2411_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_2407_);
v_toSemilatticeSup_2412_ = lean_ctor_get(v_toLattice_2408_, 0);
lean_inc_ref(v_toSemilatticeSup_2412_);
v___x_2413_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_2408_);
v_toPartialOrder_2414_ = lean_ctor_get(v___x_2413_, 0);
lean_inc_ref(v_toPartialOrder_2414_);
v_toLE_2415_ = lean_ctor_get(v_toPartialOrder_2414_, 0);
v_toLT_2416_ = lean_ctor_get(v_toPartialOrder_2414_, 1);
v_isSharedCheck_2513_ = !lean_is_exclusive(v_toPartialOrder_2414_);
if (v_isSharedCheck_2513_ == 0)
{
v___x_2418_ = v_toPartialOrder_2414_;
v_isShared_2419_ = v_isSharedCheck_2513_;
goto v_resetjp_2417_;
}
else
{
lean_inc(v_toLT_2416_);
lean_inc(v_toLE_2415_);
lean_dec(v_toPartialOrder_2414_);
v___x_2418_ = lean_box(0);
v_isShared_2419_ = v_isSharedCheck_2513_;
goto v_resetjp_2417_;
}
v_resetjp_2417_:
{
lean_object* v_compl_2420_; lean_object* v_himp_2421_; lean_object* v___f_2422_; lean_object* v___f_2423_; lean_object* v___x_2425_; 
lean_inc(v_toFun_2267_);
lean_inc_ref(v_e_2255_);
v_compl_2420_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_2420_, 0, v_e_2255_);
lean_closure_set(v_compl_2420_, 1, v_toCompl_2336_);
lean_closure_set(v_compl_2420_, 2, v_toFun_2267_);
v_himp_2421_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_2421_, 0, v___f_2286_);
lean_closure_set(v_himp_2421_, 1, v_e_2255_);
lean_closure_set(v_himp_2421_, 2, v_toHImp_2340_);
lean_closure_set(v_himp_2421_, 3, v_toFun_2267_);
v___f_2422_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2422_, 0, v_toSemilatticeSup_2412_);
v___f_2423_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2423_, 0, v___x_2413_);
if (v_isShared_2419_ == 0)
{
v___x_2425_ = v___x_2418_;
goto v_reusejp_2424_;
}
else
{
lean_object* v_reuseFailAlloc_2512_; 
v_reuseFailAlloc_2512_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2512_, 0, v_toLE_2415_);
lean_ctor_set(v_reuseFailAlloc_2512_, 1, v_toLT_2416_);
v___x_2425_ = v_reuseFailAlloc_2512_;
goto v_reusejp_2424_;
}
v_reusejp_2424_:
{
lean_object* v___x_2427_; 
if (v_isShared_2399_ == 0)
{
lean_ctor_set(v___x_2398_, 1, v___f_2386_);
lean_ctor_set(v___x_2398_, 0, v___x_2425_);
v___x_2427_ = v___x_2398_;
goto v_reusejp_2426_;
}
else
{
lean_object* v_reuseFailAlloc_2511_; 
v_reuseFailAlloc_2511_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2511_, 0, v___x_2425_);
lean_ctor_set(v_reuseFailAlloc_2511_, 1, v___f_2386_);
v___x_2427_ = v_reuseFailAlloc_2511_;
goto v_reusejp_2426_;
}
v_reusejp_2426_:
{
lean_object* v___x_2429_; 
if (v_isShared_2270_ == 0)
{
lean_ctor_set(v___x_2269_, 1, v___f_2367_);
lean_ctor_set(v___x_2269_, 0, v___x_2427_);
v___x_2429_ = v___x_2269_;
goto v_reusejp_2428_;
}
else
{
lean_object* v_reuseFailAlloc_2510_; 
v_reuseFailAlloc_2510_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2510_, 0, v___x_2427_);
lean_ctor_set(v_reuseFailAlloc_2510_, 1, v___f_2367_);
v___x_2429_ = v_reuseFailAlloc_2510_;
goto v_reusejp_2428_;
}
v_reusejp_2428_:
{
lean_object* v___x_2431_; 
lean_inc_ref(v_himp_2421_);
lean_inc(v_toOrderTop_2409_);
if (v_isShared_2343_ == 0)
{
lean_ctor_set(v___x_2342_, 2, v_himp_2421_);
lean_ctor_set(v___x_2342_, 1, v_toOrderTop_2409_);
lean_ctor_set(v___x_2342_, 0, v___x_2429_);
v___x_2431_ = v___x_2342_;
goto v_reusejp_2430_;
}
else
{
lean_object* v_reuseFailAlloc_2509_; 
v_reuseFailAlloc_2509_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2509_, 0, v___x_2429_);
lean_ctor_set(v_reuseFailAlloc_2509_, 1, v_toOrderTop_2409_);
lean_ctor_set(v_reuseFailAlloc_2509_, 2, v_himp_2421_);
v___x_2431_ = v_reuseFailAlloc_2509_;
goto v_reusejp_2430_;
}
v_reusejp_2430_:
{
lean_object* v___x_2433_; 
lean_inc_ref(v_compl_2420_);
lean_inc(v_bot_2385_);
if (v_isShared_2339_ == 0)
{
lean_ctor_set(v___x_2338_, 2, v_compl_2420_);
lean_ctor_set(v___x_2338_, 1, v_bot_2385_);
lean_ctor_set(v___x_2338_, 0, v___x_2431_);
v___x_2433_ = v___x_2338_;
goto v_reusejp_2432_;
}
else
{
lean_object* v_reuseFailAlloc_2508_; 
v_reuseFailAlloc_2508_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2508_, 0, v___x_2431_);
lean_ctor_set(v_reuseFailAlloc_2508_, 1, v_bot_2385_);
lean_ctor_set(v_reuseFailAlloc_2508_, 2, v_compl_2420_);
v___x_2433_ = v_reuseFailAlloc_2508_;
goto v_reusejp_2432_;
}
v_reusejp_2432_:
{
lean_object* v___x_2434_; lean_object* v_toGeneralizedCoheytingAlgebra_2435_; lean_object* v_toHNot_2436_; lean_object* v_toSDiff_2437_; lean_object* v___x_2439_; uint8_t v_isShared_2440_; uint8_t v_isSharedCheck_2505_; 
lean_inc(v_bot_2385_);
v___x_2434_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2423_, v___f_2422_, v_toLE_2415_, v_toLT_2416_, v_bot_2385_, v_toOrderTop_2409_, v_toHNot_2410_, v_toSDiff_2411_);
v_toGeneralizedCoheytingAlgebra_2435_ = lean_ctor_get(v___x_2434_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2435_);
v_toHNot_2436_ = lean_ctor_get(v___x_2434_, 2);
lean_inc(v_toHNot_2436_);
lean_dec_ref(v___x_2434_);
v_toSDiff_2437_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2435_, 2);
v_isSharedCheck_2505_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_2435_);
if (v_isSharedCheck_2505_ == 0)
{
lean_object* v_unused_2506_; lean_object* v_unused_2507_; 
v_unused_2506_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2435_, 1);
lean_dec(v_unused_2506_);
v_unused_2507_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2435_, 0);
lean_dec(v_unused_2507_);
v___x_2439_ = v_toGeneralizedCoheytingAlgebra_2435_;
v_isShared_2440_ = v_isSharedCheck_2505_;
goto v_resetjp_2438_;
}
else
{
lean_inc(v_toSDiff_2437_);
lean_dec(v_toGeneralizedCoheytingAlgebra_2435_);
v___x_2439_ = lean_box(0);
v_isShared_2440_ = v_isSharedCheck_2505_;
goto v_resetjp_2438_;
}
v_resetjp_2438_:
{
lean_object* v_biheytingAlgebra_2442_; 
if (v_isShared_2440_ == 0)
{
lean_ctor_set(v___x_2439_, 2, v_toHNot_2436_);
lean_ctor_set(v___x_2439_, 1, v_toSDiff_2437_);
lean_ctor_set(v___x_2439_, 0, v___x_2433_);
v_biheytingAlgebra_2442_ = v___x_2439_;
goto v_reusejp_2441_;
}
else
{
lean_object* v_reuseFailAlloc_2504_; 
v_reuseFailAlloc_2504_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2504_, 0, v___x_2433_);
lean_ctor_set(v_reuseFailAlloc_2504_, 1, v_toSDiff_2437_);
lean_ctor_set(v_reuseFailAlloc_2504_, 2, v_toHNot_2436_);
v_biheytingAlgebra_2442_ = v_reuseFailAlloc_2504_;
goto v_reusejp_2441_;
}
v_reusejp_2441_:
{
lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v_toPartialOrder_2445_; lean_object* v_toInfSet_2446_; lean_object* v___x_2448_; uint8_t v_isShared_2449_; uint8_t v_isSharedCheck_2503_; 
v___x_2443_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2328_);
lean_inc_ref(v_completeLattice_2331_);
v___x_2444_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_2331_);
v_toPartialOrder_2445_ = lean_ctor_get(v___x_2444_, 0);
v_toInfSet_2446_ = lean_ctor_get(v___x_2444_, 1);
v_isSharedCheck_2503_ = !lean_is_exclusive(v___x_2444_);
if (v_isSharedCheck_2503_ == 0)
{
v___x_2448_ = v___x_2444_;
v_isShared_2449_ = v_isSharedCheck_2503_;
goto v_resetjp_2447_;
}
else
{
lean_inc(v_toInfSet_2446_);
lean_inc(v_toPartialOrder_2445_);
lean_dec(v___x_2444_);
v___x_2448_ = lean_box(0);
v_isShared_2449_ = v_isSharedCheck_2503_;
goto v_resetjp_2447_;
}
v_resetjp_2447_:
{
lean_object* v_toLE_2450_; lean_object* v_toLT_2451_; lean_object* v___x_2453_; uint8_t v_isShared_2454_; uint8_t v_isSharedCheck_2502_; 
v_toLE_2450_ = lean_ctor_get(v_toPartialOrder_2445_, 0);
v_toLT_2451_ = lean_ctor_get(v_toPartialOrder_2445_, 1);
v_isSharedCheck_2502_ = !lean_is_exclusive(v_toPartialOrder_2445_);
if (v_isSharedCheck_2502_ == 0)
{
v___x_2453_ = v_toPartialOrder_2445_;
v_isShared_2454_ = v_isSharedCheck_2502_;
goto v_resetjp_2452_;
}
else
{
lean_inc(v_toLT_2451_);
lean_inc(v_toLE_2450_);
lean_dec(v_toPartialOrder_2445_);
v___x_2453_ = lean_box(0);
v_isShared_2454_ = v_isSharedCheck_2502_;
goto v_resetjp_2452_;
}
v_resetjp_2452_:
{
lean_object* v___x_2455_; lean_object* v_toSupSet_2456_; lean_object* v___x_2458_; uint8_t v_isShared_2459_; uint8_t v_isSharedCheck_2500_; 
v___x_2455_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_2331_);
v_toSupSet_2456_ = lean_ctor_get(v___x_2455_, 1);
v_isSharedCheck_2500_ = !lean_is_exclusive(v___x_2455_);
if (v_isSharedCheck_2500_ == 0)
{
lean_object* v_unused_2501_; 
v_unused_2501_ = lean_ctor_get(v___x_2455_, 0);
lean_dec(v_unused_2501_);
v___x_2458_ = v___x_2455_;
v_isShared_2459_ = v_isSharedCheck_2500_;
goto v_resetjp_2457_;
}
else
{
lean_inc(v_toSupSet_2456_);
lean_dec(v___x_2455_);
v___x_2458_ = lean_box(0);
v_isShared_2459_ = v_isSharedCheck_2500_;
goto v_resetjp_2457_;
}
v_resetjp_2457_:
{
lean_object* v___x_2460_; lean_object* v_toGeneralizedCoheytingAlgebra_2461_; lean_object* v_toOrderTop_2462_; lean_object* v_toHNot_2463_; lean_object* v_toSDiff_2464_; lean_object* v___x_2466_; uint8_t v_isShared_2467_; uint8_t v_isSharedCheck_2497_; 
v___x_2460_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_biheytingAlgebra_2442_);
v_toGeneralizedCoheytingAlgebra_2461_ = lean_ctor_get(v___x_2460_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2461_);
v_toOrderTop_2462_ = lean_ctor_get(v___x_2460_, 1);
lean_inc(v_toOrderTop_2462_);
v_toHNot_2463_ = lean_ctor_get(v___x_2460_, 2);
lean_inc(v_toHNot_2463_);
lean_dec_ref(v___x_2460_);
v_toSDiff_2464_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2461_, 2);
v_isSharedCheck_2497_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_2461_);
if (v_isSharedCheck_2497_ == 0)
{
lean_object* v_unused_2498_; lean_object* v_unused_2499_; 
v_unused_2498_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2461_, 1);
lean_dec(v_unused_2498_);
v_unused_2499_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2461_, 0);
lean_dec(v_unused_2499_);
v___x_2466_ = v_toGeneralizedCoheytingAlgebra_2461_;
v_isShared_2467_ = v_isSharedCheck_2497_;
goto v_resetjp_2465_;
}
else
{
lean_inc(v_toSDiff_2464_);
lean_dec(v_toGeneralizedCoheytingAlgebra_2461_);
v___x_2466_ = lean_box(0);
v_isShared_2467_ = v_isSharedCheck_2497_;
goto v_resetjp_2465_;
}
v_resetjp_2465_:
{
lean_object* v___f_2468_; lean_object* v___f_2469_; lean_object* v___x_2471_; 
v___f_2468_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2468_, 0, v___x_2326_);
v___f_2469_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2469_, 0, v___x_2443_);
if (v_isShared_2454_ == 0)
{
v___x_2471_ = v___x_2453_;
goto v_reusejp_2470_;
}
else
{
lean_object* v_reuseFailAlloc_2496_; 
v_reuseFailAlloc_2496_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2496_, 0, v_toLE_2450_);
lean_ctor_set(v_reuseFailAlloc_2496_, 1, v_toLT_2451_);
v___x_2471_ = v_reuseFailAlloc_2496_;
goto v_reusejp_2470_;
}
v_reusejp_2470_:
{
lean_object* v___x_2473_; 
if (v_isShared_2459_ == 0)
{
lean_ctor_set(v___x_2458_, 1, v___f_2322_);
lean_ctor_set(v___x_2458_, 0, v___x_2471_);
v___x_2473_ = v___x_2458_;
goto v_reusejp_2472_;
}
else
{
lean_object* v_reuseFailAlloc_2495_; 
v_reuseFailAlloc_2495_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2495_, 0, v___x_2471_);
lean_ctor_set(v_reuseFailAlloc_2495_, 1, v___f_2322_);
v___x_2473_ = v_reuseFailAlloc_2495_;
goto v_reusejp_2472_;
}
v_reusejp_2472_:
{
lean_object* v___x_2475_; 
if (v_isShared_2449_ == 0)
{
lean_ctor_set(v___x_2448_, 1, v___f_2300_);
lean_ctor_set(v___x_2448_, 0, v___x_2473_);
v___x_2475_ = v___x_2448_;
goto v_reusejp_2474_;
}
else
{
lean_object* v_reuseFailAlloc_2494_; 
v_reuseFailAlloc_2494_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2494_, 0, v___x_2473_);
lean_ctor_set(v_reuseFailAlloc_2494_, 1, v___f_2300_);
v___x_2475_ = v_reuseFailAlloc_2494_;
goto v_reusejp_2474_;
}
v_reusejp_2474_:
{
lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2479_; 
lean_inc(v_bot_2385_);
lean_inc(v_toOrderTop_2462_);
v___x_2476_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2476_, 0, v_toOrderTop_2462_);
lean_ctor_set(v___x_2476_, 1, v_bot_2385_);
v___x_2477_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2477_, 0, v___x_2475_);
lean_ctor_set(v___x_2477_, 1, v_toSupSet_2456_);
lean_ctor_set(v___x_2477_, 2, v_toInfSet_2446_);
lean_ctor_set(v___x_2477_, 3, v___x_2476_);
if (v_isShared_2467_ == 0)
{
lean_ctor_set(v___x_2466_, 2, v_compl_2420_);
lean_ctor_set(v___x_2466_, 1, v_himp_2421_);
lean_ctor_set(v___x_2466_, 0, v___x_2477_);
v___x_2479_ = v___x_2466_;
goto v_reusejp_2478_;
}
else
{
lean_object* v_reuseFailAlloc_2493_; 
v_reuseFailAlloc_2493_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2493_, 0, v___x_2477_);
lean_ctor_set(v_reuseFailAlloc_2493_, 1, v_himp_2421_);
lean_ctor_set(v_reuseFailAlloc_2493_, 2, v_compl_2420_);
v___x_2479_ = v_reuseFailAlloc_2493_;
goto v_reusejp_2478_;
}
v_reusejp_2478_:
{
lean_object* v___x_2480_; lean_object* v_toGeneralizedCoheytingAlgebra_2481_; lean_object* v_toHNot_2482_; lean_object* v_toSDiff_2483_; lean_object* v___x_2485_; uint8_t v_isShared_2486_; uint8_t v_isSharedCheck_2490_; 
v___x_2480_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2469_, v___f_2468_, v_toLE_2450_, v_toLT_2451_, v_bot_2385_, v_toOrderTop_2462_, v_toHNot_2463_, v_toSDiff_2464_);
v_toGeneralizedCoheytingAlgebra_2481_ = lean_ctor_get(v___x_2480_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2481_);
v_toHNot_2482_ = lean_ctor_get(v___x_2480_, 2);
lean_inc(v_toHNot_2482_);
lean_dec_ref(v___x_2480_);
v_toSDiff_2483_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2481_, 2);
v_isSharedCheck_2490_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_2481_);
if (v_isSharedCheck_2490_ == 0)
{
lean_object* v_unused_2491_; lean_object* v_unused_2492_; 
v_unused_2491_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2481_, 1);
lean_dec(v_unused_2491_);
v_unused_2492_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2481_, 0);
lean_dec(v_unused_2492_);
v___x_2485_ = v_toGeneralizedCoheytingAlgebra_2481_;
v_isShared_2486_ = v_isSharedCheck_2490_;
goto v_resetjp_2484_;
}
else
{
lean_inc(v_toSDiff_2483_);
lean_dec(v_toGeneralizedCoheytingAlgebra_2481_);
v___x_2485_ = lean_box(0);
v_isShared_2486_ = v_isSharedCheck_2490_;
goto v_resetjp_2484_;
}
v_resetjp_2484_:
{
lean_object* v___x_2488_; 
if (v_isShared_2486_ == 0)
{
lean_ctor_set(v___x_2485_, 2, v_toHNot_2482_);
lean_ctor_set(v___x_2485_, 1, v_toSDiff_2483_);
lean_ctor_set(v___x_2485_, 0, v___x_2479_);
v___x_2488_ = v___x_2485_;
goto v_reusejp_2487_;
}
else
{
lean_object* v_reuseFailAlloc_2489_; 
v_reuseFailAlloc_2489_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2489_, 0, v___x_2479_);
lean_ctor_set(v_reuseFailAlloc_2489_, 1, v_toSDiff_2483_);
lean_ctor_set(v_reuseFailAlloc_2489_, 2, v_toHNot_2482_);
v___x_2488_ = v_reuseFailAlloc_2489_;
goto v_reusejp_2487_;
}
v_reusejp_2487_:
{
return v___x_2488_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeDistribLattice(lean_object* v_00_u03b1_2553_, lean_object* v_00_u03b2_2554_, lean_object* v_e_2555_, lean_object* v_inst_2556_){
_start:
{
lean_object* v___x_2557_; lean_object* v_toCompleteLattice_2558_; lean_object* v_toBoundedOrder_2559_; lean_object* v_toLattice_2560_; lean_object* v_toOrderTop_2561_; lean_object* v_toOrderBot_2562_; lean_object* v___x_2564_; uint8_t v_isShared_2565_; uint8_t v_isSharedCheck_2852_; 
lean_inc_ref(v_inst_2556_);
v___x_2557_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v_inst_2556_);
v_toCompleteLattice_2558_ = lean_ctor_get(v___x_2557_, 0);
lean_inc_ref(v_toCompleteLattice_2558_);
lean_dec_ref(v___x_2557_);
v_toBoundedOrder_2559_ = lean_ctor_get(v_toCompleteLattice_2558_, 3);
lean_inc_ref(v_toBoundedOrder_2559_);
v_toLattice_2560_ = lean_ctor_get(v_toCompleteLattice_2558_, 0);
lean_inc_ref(v_toLattice_2560_);
v_toOrderTop_2561_ = lean_ctor_get(v_toBoundedOrder_2559_, 0);
v_toOrderBot_2562_ = lean_ctor_get(v_toBoundedOrder_2559_, 1);
v_isSharedCheck_2852_ = !lean_is_exclusive(v_toBoundedOrder_2559_);
if (v_isSharedCheck_2852_ == 0)
{
v___x_2564_ = v_toBoundedOrder_2559_;
v_isShared_2565_ = v_isSharedCheck_2852_;
goto v_resetjp_2563_;
}
else
{
lean_inc(v_toOrderBot_2562_);
lean_inc(v_toOrderTop_2561_);
lean_dec(v_toBoundedOrder_2559_);
v___x_2564_ = lean_box(0);
v_isShared_2565_ = v_isSharedCheck_2852_;
goto v_resetjp_2563_;
}
v_resetjp_2563_:
{
lean_object* v___x_2566_; lean_object* v_toFun_2567_; lean_object* v___x_2569_; uint8_t v_isShared_2570_; uint8_t v_isSharedCheck_2850_; 
lean_inc_ref(v_e_2555_);
v___x_2566_ = lp_mathlib_Equiv_symm___redArg(v_e_2555_);
v_toFun_2567_ = lean_ctor_get(v___x_2566_, 0);
v_isSharedCheck_2850_ = !lean_is_exclusive(v___x_2566_);
if (v_isSharedCheck_2850_ == 0)
{
lean_object* v_unused_2851_; 
v_unused_2851_ = lean_ctor_get(v___x_2566_, 1);
lean_dec(v_unused_2851_);
v___x_2569_ = v___x_2566_;
v_isShared_2570_ = v_isSharedCheck_2850_;
goto v_resetjp_2568_;
}
else
{
lean_inc(v_toFun_2567_);
lean_dec(v___x_2566_);
v___x_2569_ = lean_box(0);
v_isShared_2570_ = v_isSharedCheck_2850_;
goto v_resetjp_2568_;
}
v_resetjp_2568_:
{
lean_object* v___x_2571_; lean_object* v_toSupSet_2572_; lean_object* v___x_2574_; uint8_t v_isShared_2575_; uint8_t v_isSharedCheck_2848_; 
lean_inc_ref(v_toCompleteLattice_2558_);
v___x_2571_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_2558_);
v_toSupSet_2572_ = lean_ctor_get(v___x_2571_, 1);
v_isSharedCheck_2848_ = !lean_is_exclusive(v___x_2571_);
if (v_isSharedCheck_2848_ == 0)
{
lean_object* v_unused_2849_; 
v_unused_2849_ = lean_ctor_get(v___x_2571_, 0);
lean_dec(v_unused_2849_);
v___x_2574_ = v___x_2571_;
v_isShared_2575_ = v_isSharedCheck_2848_;
goto v_resetjp_2573_;
}
else
{
lean_inc(v_toSupSet_2572_);
lean_dec(v___x_2571_);
v___x_2574_ = lean_box(0);
v_isShared_2575_ = v_isSharedCheck_2848_;
goto v_resetjp_2573_;
}
v_resetjp_2573_:
{
lean_object* v___x_2576_; lean_object* v_toInfSet_2577_; lean_object* v___x_2579_; uint8_t v_isShared_2580_; uint8_t v_isSharedCheck_2846_; 
v___x_2576_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_2558_);
v_toInfSet_2577_ = lean_ctor_get(v___x_2576_, 1);
v_isSharedCheck_2846_ = !lean_is_exclusive(v___x_2576_);
if (v_isSharedCheck_2846_ == 0)
{
lean_object* v_unused_2847_; 
v_unused_2847_ = lean_ctor_get(v___x_2576_, 0);
lean_dec(v_unused_2847_);
v___x_2579_ = v___x_2576_;
v_isShared_2580_ = v_isSharedCheck_2846_;
goto v_resetjp_2578_;
}
else
{
lean_inc(v_toInfSet_2577_);
lean_dec(v___x_2576_);
v___x_2579_ = lean_box(0);
v_isShared_2580_ = v_isSharedCheck_2846_;
goto v_resetjp_2578_;
}
v_resetjp_2578_:
{
lean_object* v_toSemilatticeSup_2581_; lean_object* v_inf_2582_; lean_object* v___x_2584_; uint8_t v_isShared_2585_; uint8_t v_isSharedCheck_2845_; 
v_toSemilatticeSup_2581_ = lean_ctor_get(v_toLattice_2560_, 0);
v_inf_2582_ = lean_ctor_get(v_toLattice_2560_, 1);
v_isSharedCheck_2845_ = !lean_is_exclusive(v_toLattice_2560_);
if (v_isSharedCheck_2845_ == 0)
{
v___x_2584_ = v_toLattice_2560_;
v_isShared_2585_ = v_isSharedCheck_2845_;
goto v_resetjp_2583_;
}
else
{
lean_inc(v_inf_2582_);
lean_inc(v_toSemilatticeSup_2581_);
lean_dec(v_toLattice_2560_);
v___x_2584_ = lean_box(0);
v_isShared_2585_ = v_isSharedCheck_2845_;
goto v_resetjp_2583_;
}
v_resetjp_2583_:
{
lean_object* v___f_2586_; lean_object* v_min_2587_; lean_object* v_le_2588_; lean_object* v_lt_2589_; lean_object* v_semilatticeInf_2590_; lean_object* v_toPartialOrder_2591_; lean_object* v___x_2593_; uint8_t v_isShared_2594_; uint8_t v_isSharedCheck_2843_; 
v___f_2586_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_2567_);
lean_inc_ref(v_e_2555_);
v_min_2587_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2587_, 0, v___f_2586_);
lean_closure_set(v_min_2587_, 1, v_e_2555_);
lean_closure_set(v_min_2587_, 2, v_inf_2582_);
lean_closure_set(v_min_2587_, 3, v_toFun_2567_);
v_le_2588_ = lean_box(0);
v_lt_2589_ = lean_box(0);
lean_inc_ref(v_min_2587_);
v_semilatticeInf_2590_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2587_, v_le_2588_, v_lt_2589_);
v_toPartialOrder_2591_ = lean_ctor_get(v_semilatticeInf_2590_, 0);
v_isSharedCheck_2843_ = !lean_is_exclusive(v_semilatticeInf_2590_);
if (v_isSharedCheck_2843_ == 0)
{
lean_object* v_unused_2844_; 
v_unused_2844_ = lean_ctor_get(v_semilatticeInf_2590_, 1);
lean_dec(v_unused_2844_);
v___x_2593_ = v_semilatticeInf_2590_;
v_isShared_2594_ = v_isSharedCheck_2843_;
goto v_resetjp_2592_;
}
else
{
lean_inc(v_toPartialOrder_2591_);
lean_dec(v_semilatticeInf_2590_);
v___x_2593_ = lean_box(0);
v_isShared_2594_ = v_isSharedCheck_2843_;
goto v_resetjp_2592_;
}
v_resetjp_2592_:
{
lean_object* v_toLE_2595_; lean_object* v_toLT_2596_; lean_object* v___x_2598_; uint8_t v_isShared_2599_; uint8_t v_isSharedCheck_2842_; 
v_toLE_2595_ = lean_ctor_get(v_toPartialOrder_2591_, 0);
v_toLT_2596_ = lean_ctor_get(v_toPartialOrder_2591_, 1);
v_isSharedCheck_2842_ = !lean_is_exclusive(v_toPartialOrder_2591_);
if (v_isSharedCheck_2842_ == 0)
{
v___x_2598_ = v_toPartialOrder_2591_;
v_isShared_2599_ = v_isSharedCheck_2842_;
goto v_resetjp_2597_;
}
else
{
lean_inc(v_toLT_2596_);
lean_inc(v_toLE_2595_);
lean_dec(v_toPartialOrder_2591_);
v___x_2598_ = lean_box(0);
v_isShared_2599_ = v_isSharedCheck_2842_;
goto v_resetjp_2597_;
}
v_resetjp_2597_:
{
lean_object* v___f_2600_; lean_object* v___f_2601_; lean_object* v___x_2603_; 
v___f_2600_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_2600_, 0, v_min_2587_);
lean_inc(v_toFun_2567_);
lean_inc_ref(v_e_2555_);
v___f_2601_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_2601_, 0, v_toSemilatticeSup_2581_);
lean_closure_set(v___f_2601_, 1, v___f_2586_);
lean_closure_set(v___f_2601_, 2, v_e_2555_);
lean_closure_set(v___f_2601_, 3, v_toFun_2567_);
if (v_isShared_2599_ == 0)
{
v___x_2603_ = v___x_2598_;
goto v_reusejp_2602_;
}
else
{
lean_object* v_reuseFailAlloc_2841_; 
v_reuseFailAlloc_2841_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2841_, 0, v_toLE_2595_);
lean_ctor_set(v_reuseFailAlloc_2841_, 1, v_toLT_2596_);
v___x_2603_ = v_reuseFailAlloc_2841_;
goto v_reusejp_2602_;
}
v_reusejp_2602_:
{
lean_object* v___x_2605_; 
lean_inc_ref(v___f_2601_);
if (v_isShared_2594_ == 0)
{
lean_ctor_set(v___x_2593_, 1, v___f_2601_);
lean_ctor_set(v___x_2593_, 0, v___x_2603_);
v___x_2605_ = v___x_2593_;
goto v_reusejp_2604_;
}
else
{
lean_object* v_reuseFailAlloc_2840_; 
v_reuseFailAlloc_2840_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2840_, 0, v___x_2603_);
lean_ctor_set(v_reuseFailAlloc_2840_, 1, v___f_2601_);
v___x_2605_ = v_reuseFailAlloc_2840_;
goto v_reusejp_2604_;
}
v_reusejp_2604_:
{
lean_object* v_lattice_2607_; 
lean_inc_ref(v___f_2600_);
if (v_isShared_2585_ == 0)
{
lean_ctor_set(v___x_2584_, 1, v___f_2600_);
lean_ctor_set(v___x_2584_, 0, v___x_2605_);
v_lattice_2607_ = v___x_2584_;
goto v_reusejp_2606_;
}
else
{
lean_object* v_reuseFailAlloc_2839_; 
v_reuseFailAlloc_2839_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2839_, 0, v___x_2605_);
lean_ctor_set(v_reuseFailAlloc_2839_, 1, v___f_2600_);
v_lattice_2607_ = v_reuseFailAlloc_2839_;
goto v_reusejp_2606_;
}
v_reusejp_2606_:
{
lean_object* v___x_2608_; lean_object* v_toPartialOrder_2609_; lean_object* v___x_2611_; uint8_t v_isShared_2612_; uint8_t v_isSharedCheck_2837_; 
v___x_2608_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2607_);
v_toPartialOrder_2609_ = lean_ctor_get(v___x_2608_, 0);
v_isSharedCheck_2837_ = !lean_is_exclusive(v___x_2608_);
if (v_isSharedCheck_2837_ == 0)
{
lean_object* v_unused_2838_; 
v_unused_2838_ = lean_ctor_get(v___x_2608_, 1);
lean_dec(v_unused_2838_);
v___x_2611_ = v___x_2608_;
v_isShared_2612_ = v_isSharedCheck_2837_;
goto v_resetjp_2610_;
}
else
{
lean_inc(v_toPartialOrder_2609_);
lean_dec(v___x_2608_);
v___x_2611_ = lean_box(0);
v_isShared_2612_ = v_isSharedCheck_2837_;
goto v_resetjp_2610_;
}
v_resetjp_2610_:
{
lean_object* v_toLE_2613_; lean_object* v_toLT_2614_; lean_object* v___x_2616_; uint8_t v_isShared_2617_; uint8_t v_isSharedCheck_2836_; 
v_toLE_2613_ = lean_ctor_get(v_toPartialOrder_2609_, 0);
v_toLT_2614_ = lean_ctor_get(v_toPartialOrder_2609_, 1);
v_isSharedCheck_2836_ = !lean_is_exclusive(v_toPartialOrder_2609_);
if (v_isSharedCheck_2836_ == 0)
{
v___x_2616_ = v_toPartialOrder_2609_;
v_isShared_2617_ = v_isSharedCheck_2836_;
goto v_resetjp_2615_;
}
else
{
lean_inc(v_toLT_2614_);
lean_inc(v_toLE_2613_);
lean_dec(v_toPartialOrder_2609_);
v___x_2616_ = lean_box(0);
v_isShared_2617_ = v_isSharedCheck_2836_;
goto v_resetjp_2615_;
}
v_resetjp_2615_:
{
lean_object* v_top_2618_; lean_object* v_bot_2619_; lean_object* v_supSet_2620_; lean_object* v_infSet_2621_; lean_object* v___f_2622_; lean_object* v___x_2624_; 
lean_inc_n(v_toFun_2567_, 4);
v_top_2618_ = lean_apply_1(v_toFun_2567_, v_toOrderTop_2561_);
v_bot_2619_ = lean_apply_1(v_toFun_2567_, v_toOrderBot_2562_);
v_supSet_2620_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_2620_, 0, v_toSupSet_2572_);
lean_closure_set(v_supSet_2620_, 1, v_toFun_2567_);
v_infSet_2621_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_2621_, 0, v_toInfSet_2577_);
lean_closure_set(v_infSet_2621_, 1, v_toFun_2567_);
v___f_2622_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_2622_, 0, v___f_2601_);
if (v_isShared_2617_ == 0)
{
v___x_2624_ = v___x_2616_;
goto v_reusejp_2623_;
}
else
{
lean_object* v_reuseFailAlloc_2835_; 
v_reuseFailAlloc_2835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2835_, 0, v_toLE_2613_);
lean_ctor_set(v_reuseFailAlloc_2835_, 1, v_toLT_2614_);
v___x_2624_ = v_reuseFailAlloc_2835_;
goto v_reusejp_2623_;
}
v_reusejp_2623_:
{
lean_object* v___x_2626_; 
lean_inc_ref(v___f_2622_);
if (v_isShared_2612_ == 0)
{
lean_ctor_set(v___x_2611_, 1, v___f_2622_);
lean_ctor_set(v___x_2611_, 0, v___x_2624_);
v___x_2626_ = v___x_2611_;
goto v_reusejp_2625_;
}
else
{
lean_object* v_reuseFailAlloc_2834_; 
v_reuseFailAlloc_2834_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2834_, 0, v___x_2624_);
lean_ctor_set(v_reuseFailAlloc_2834_, 1, v___f_2622_);
v___x_2626_ = v_reuseFailAlloc_2834_;
goto v_reusejp_2625_;
}
v_reusejp_2625_:
{
lean_object* v___x_2628_; 
lean_inc_ref(v___f_2600_);
lean_inc_ref(v___x_2626_);
if (v_isShared_2580_ == 0)
{
lean_ctor_set(v___x_2579_, 1, v___f_2600_);
lean_ctor_set(v___x_2579_, 0, v___x_2626_);
v___x_2628_ = v___x_2579_;
goto v_reusejp_2627_;
}
else
{
lean_object* v_reuseFailAlloc_2833_; 
v_reuseFailAlloc_2833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2833_, 0, v___x_2626_);
lean_ctor_set(v_reuseFailAlloc_2833_, 1, v___f_2600_);
v___x_2628_ = v_reuseFailAlloc_2833_;
goto v_reusejp_2627_;
}
v_reusejp_2627_:
{
lean_object* v___x_2630_; 
if (v_isShared_2565_ == 0)
{
lean_ctor_set(v___x_2564_, 1, v_bot_2619_);
lean_ctor_set(v___x_2564_, 0, v_top_2618_);
v___x_2630_ = v___x_2564_;
goto v_reusejp_2629_;
}
else
{
lean_object* v_reuseFailAlloc_2832_; 
v_reuseFailAlloc_2832_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2832_, 0, v_top_2618_);
lean_ctor_set(v_reuseFailAlloc_2832_, 1, v_bot_2619_);
v___x_2630_ = v_reuseFailAlloc_2832_;
goto v_reusejp_2629_;
}
v_reusejp_2629_:
{
lean_object* v_completeLattice_2631_; lean_object* v___x_2632_; lean_object* v_toHeytingAlgebra_2633_; lean_object* v_toGeneralizedHeytingAlgebra_2634_; lean_object* v_toOrderBot_2635_; lean_object* v_toCompl_2636_; lean_object* v___x_2638_; uint8_t v_isShared_2639_; uint8_t v_isSharedCheck_2831_; 
lean_inc_ref(v___x_2628_);
v_completeLattice_2631_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_2631_, 0, v___x_2628_);
lean_ctor_set(v_completeLattice_2631_, 1, v_supSet_2620_);
lean_ctor_set(v_completeLattice_2631_, 2, v_infSet_2621_);
lean_ctor_set(v_completeLattice_2631_, 3, v___x_2630_);
v___x_2632_ = lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra___redArg(v_inst_2556_);
v_toHeytingAlgebra_2633_ = lean_ctor_get(v___x_2632_, 0);
lean_inc_ref(v_toHeytingAlgebra_2633_);
v_toGeneralizedHeytingAlgebra_2634_ = lean_ctor_get(v_toHeytingAlgebra_2633_, 0);
v_toOrderBot_2635_ = lean_ctor_get(v_toHeytingAlgebra_2633_, 1);
v_toCompl_2636_ = lean_ctor_get(v_toHeytingAlgebra_2633_, 2);
v_isSharedCheck_2831_ = !lean_is_exclusive(v_toHeytingAlgebra_2633_);
if (v_isSharedCheck_2831_ == 0)
{
v___x_2638_ = v_toHeytingAlgebra_2633_;
v_isShared_2639_ = v_isSharedCheck_2831_;
goto v_resetjp_2637_;
}
else
{
lean_inc(v_toCompl_2636_);
lean_inc(v_toOrderBot_2635_);
lean_inc(v_toGeneralizedHeytingAlgebra_2634_);
lean_dec(v_toHeytingAlgebra_2633_);
v___x_2638_ = lean_box(0);
v_isShared_2639_ = v_isSharedCheck_2831_;
goto v_resetjp_2637_;
}
v_resetjp_2637_:
{
lean_object* v_toHImp_2640_; lean_object* v___x_2642_; uint8_t v_isShared_2643_; uint8_t v_isSharedCheck_2828_; 
v_toHImp_2640_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2634_, 2);
v_isSharedCheck_2828_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_2634_);
if (v_isSharedCheck_2828_ == 0)
{
lean_object* v_unused_2829_; lean_object* v_unused_2830_; 
v_unused_2829_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2634_, 1);
lean_dec(v_unused_2829_);
v_unused_2830_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2634_, 0);
lean_dec(v_unused_2830_);
v___x_2642_ = v_toGeneralizedHeytingAlgebra_2634_;
v_isShared_2643_ = v_isSharedCheck_2828_;
goto v_resetjp_2641_;
}
else
{
lean_inc(v_toHImp_2640_);
lean_dec(v_toGeneralizedHeytingAlgebra_2634_);
v___x_2642_ = lean_box(0);
v_isShared_2643_ = v_isSharedCheck_2828_;
goto v_resetjp_2641_;
}
v_resetjp_2641_:
{
lean_object* v___x_2644_; lean_object* v_toGeneralizedCoheytingAlgebra_2645_; lean_object* v_toLattice_2646_; lean_object* v_toOrderTop_2647_; lean_object* v_toHNot_2648_; lean_object* v_toOrderBot_2649_; lean_object* v_toSDiff_2650_; lean_object* v_toSemilatticeSup_2651_; lean_object* v_inf_2652_; lean_object* v___x_2654_; uint8_t v_isShared_2655_; uint8_t v_isSharedCheck_2827_; 
v___x_2644_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v___x_2632_);
v_toGeneralizedCoheytingAlgebra_2645_ = lean_ctor_get(v___x_2644_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2645_);
v_toLattice_2646_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2645_, 0);
lean_inc_ref(v_toLattice_2646_);
v_toOrderTop_2647_ = lean_ctor_get(v___x_2644_, 1);
lean_inc(v_toOrderTop_2647_);
v_toHNot_2648_ = lean_ctor_get(v___x_2644_, 2);
lean_inc(v_toHNot_2648_);
lean_dec_ref(v___x_2644_);
v_toOrderBot_2649_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2645_, 1);
lean_inc(v_toOrderBot_2649_);
v_toSDiff_2650_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2645_, 2);
lean_inc(v_toSDiff_2650_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_2645_);
v_toSemilatticeSup_2651_ = lean_ctor_get(v_toLattice_2646_, 0);
v_inf_2652_ = lean_ctor_get(v_toLattice_2646_, 1);
v_isSharedCheck_2827_ = !lean_is_exclusive(v_toLattice_2646_);
if (v_isSharedCheck_2827_ == 0)
{
v___x_2654_ = v_toLattice_2646_;
v_isShared_2655_ = v_isSharedCheck_2827_;
goto v_resetjp_2653_;
}
else
{
lean_inc(v_inf_2652_);
lean_inc(v_toSemilatticeSup_2651_);
lean_dec(v_toLattice_2646_);
v___x_2654_ = lean_box(0);
v_isShared_2655_ = v_isSharedCheck_2827_;
goto v_resetjp_2653_;
}
v_resetjp_2653_:
{
lean_object* v_min_2656_; lean_object* v_semilatticeInf_2657_; lean_object* v_toPartialOrder_2658_; lean_object* v___x_2660_; uint8_t v_isShared_2661_; uint8_t v_isSharedCheck_2825_; 
lean_inc(v_toFun_2567_);
lean_inc_ref(v_e_2555_);
v_min_2656_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2656_, 0, v___f_2586_);
lean_closure_set(v_min_2656_, 1, v_e_2555_);
lean_closure_set(v_min_2656_, 2, v_inf_2652_);
lean_closure_set(v_min_2656_, 3, v_toFun_2567_);
lean_inc_ref(v_min_2656_);
v_semilatticeInf_2657_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2656_, v_le_2588_, v_lt_2589_);
v_toPartialOrder_2658_ = lean_ctor_get(v_semilatticeInf_2657_, 0);
v_isSharedCheck_2825_ = !lean_is_exclusive(v_semilatticeInf_2657_);
if (v_isSharedCheck_2825_ == 0)
{
lean_object* v_unused_2826_; 
v_unused_2826_ = lean_ctor_get(v_semilatticeInf_2657_, 1);
lean_dec(v_unused_2826_);
v___x_2660_ = v_semilatticeInf_2657_;
v_isShared_2661_ = v_isSharedCheck_2825_;
goto v_resetjp_2659_;
}
else
{
lean_inc(v_toPartialOrder_2658_);
lean_dec(v_semilatticeInf_2657_);
v___x_2660_ = lean_box(0);
v_isShared_2661_ = v_isSharedCheck_2825_;
goto v_resetjp_2659_;
}
v_resetjp_2659_:
{
lean_object* v_toLE_2662_; lean_object* v_toLT_2663_; lean_object* v___x_2665_; uint8_t v_isShared_2666_; uint8_t v_isSharedCheck_2824_; 
v_toLE_2662_ = lean_ctor_get(v_toPartialOrder_2658_, 0);
v_toLT_2663_ = lean_ctor_get(v_toPartialOrder_2658_, 1);
v_isSharedCheck_2824_ = !lean_is_exclusive(v_toPartialOrder_2658_);
if (v_isSharedCheck_2824_ == 0)
{
v___x_2665_ = v_toPartialOrder_2658_;
v_isShared_2666_ = v_isSharedCheck_2824_;
goto v_resetjp_2664_;
}
else
{
lean_inc(v_toLT_2663_);
lean_inc(v_toLE_2662_);
lean_dec(v_toPartialOrder_2658_);
v___x_2665_ = lean_box(0);
v_isShared_2666_ = v_isSharedCheck_2824_;
goto v_resetjp_2664_;
}
v_resetjp_2664_:
{
lean_object* v___f_2667_; lean_object* v___f_2668_; lean_object* v___x_2670_; 
v___f_2667_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_2667_, 0, v_min_2656_);
lean_inc(v_toFun_2567_);
lean_inc_ref(v_e_2555_);
v___f_2668_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_2668_, 0, v_toSemilatticeSup_2651_);
lean_closure_set(v___f_2668_, 1, v___f_2586_);
lean_closure_set(v___f_2668_, 2, v_e_2555_);
lean_closure_set(v___f_2668_, 3, v_toFun_2567_);
if (v_isShared_2666_ == 0)
{
v___x_2670_ = v___x_2665_;
goto v_reusejp_2669_;
}
else
{
lean_object* v_reuseFailAlloc_2823_; 
v_reuseFailAlloc_2823_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2823_, 0, v_toLE_2662_);
lean_ctor_set(v_reuseFailAlloc_2823_, 1, v_toLT_2663_);
v___x_2670_ = v_reuseFailAlloc_2823_;
goto v_reusejp_2669_;
}
v_reusejp_2669_:
{
lean_object* v___x_2672_; 
lean_inc_ref(v___f_2668_);
if (v_isShared_2661_ == 0)
{
lean_ctor_set(v___x_2660_, 1, v___f_2668_);
lean_ctor_set(v___x_2660_, 0, v___x_2670_);
v___x_2672_ = v___x_2660_;
goto v_reusejp_2671_;
}
else
{
lean_object* v_reuseFailAlloc_2822_; 
v_reuseFailAlloc_2822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2822_, 0, v___x_2670_);
lean_ctor_set(v_reuseFailAlloc_2822_, 1, v___f_2668_);
v___x_2672_ = v_reuseFailAlloc_2822_;
goto v_reusejp_2671_;
}
v_reusejp_2671_:
{
lean_object* v_lattice_2674_; 
lean_inc_ref(v___f_2667_);
if (v_isShared_2655_ == 0)
{
lean_ctor_set(v___x_2654_, 1, v___f_2667_);
lean_ctor_set(v___x_2654_, 0, v___x_2672_);
v_lattice_2674_ = v___x_2654_;
goto v_reusejp_2673_;
}
else
{
lean_object* v_reuseFailAlloc_2821_; 
v_reuseFailAlloc_2821_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2821_, 0, v___x_2672_);
lean_ctor_set(v_reuseFailAlloc_2821_, 1, v___f_2667_);
v_lattice_2674_ = v_reuseFailAlloc_2821_;
goto v_reusejp_2673_;
}
v_reusejp_2673_:
{
lean_object* v___x_2675_; lean_object* v_toPartialOrder_2676_; lean_object* v___x_2678_; uint8_t v_isShared_2679_; uint8_t v_isSharedCheck_2819_; 
v___x_2675_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2674_);
v_toPartialOrder_2676_ = lean_ctor_get(v___x_2675_, 0);
v_isSharedCheck_2819_ = !lean_is_exclusive(v___x_2675_);
if (v_isSharedCheck_2819_ == 0)
{
lean_object* v_unused_2820_; 
v_unused_2820_ = lean_ctor_get(v___x_2675_, 1);
lean_dec(v_unused_2820_);
v___x_2678_ = v___x_2675_;
v_isShared_2679_ = v_isSharedCheck_2819_;
goto v_resetjp_2677_;
}
else
{
lean_inc(v_toPartialOrder_2676_);
lean_dec(v___x_2675_);
v___x_2678_ = lean_box(0);
v_isShared_2679_ = v_isSharedCheck_2819_;
goto v_resetjp_2677_;
}
v_resetjp_2677_:
{
lean_object* v_toLE_2680_; lean_object* v_toLT_2681_; lean_object* v___x_2683_; uint8_t v_isShared_2684_; uint8_t v_isSharedCheck_2818_; 
v_toLE_2680_ = lean_ctor_get(v_toPartialOrder_2676_, 0);
v_toLT_2681_ = lean_ctor_get(v_toPartialOrder_2676_, 1);
v_isSharedCheck_2818_ = !lean_is_exclusive(v_toPartialOrder_2676_);
if (v_isSharedCheck_2818_ == 0)
{
v___x_2683_ = v_toPartialOrder_2676_;
v_isShared_2684_ = v_isSharedCheck_2818_;
goto v_resetjp_2682_;
}
else
{
lean_inc(v_toLT_2681_);
lean_inc(v_toLE_2680_);
lean_dec(v_toPartialOrder_2676_);
v___x_2683_ = lean_box(0);
v_isShared_2684_ = v_isSharedCheck_2818_;
goto v_resetjp_2682_;
}
v_resetjp_2682_:
{
lean_object* v_bot_2685_; lean_object* v___f_2686_; lean_object* v___x_2688_; 
lean_inc(v_toFun_2567_);
v_bot_2685_ = lean_apply_1(v_toFun_2567_, v_toOrderBot_2635_);
v___f_2686_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_2686_, 0, v___f_2668_);
if (v_isShared_2684_ == 0)
{
v___x_2688_ = v___x_2683_;
goto v_reusejp_2687_;
}
else
{
lean_object* v_reuseFailAlloc_2817_; 
v_reuseFailAlloc_2817_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2817_, 0, v_toLE_2680_);
lean_ctor_set(v_reuseFailAlloc_2817_, 1, v_toLT_2681_);
v___x_2688_ = v_reuseFailAlloc_2817_;
goto v_reusejp_2687_;
}
v_reusejp_2687_:
{
lean_object* v___x_2690_; 
lean_inc_ref(v___f_2686_);
if (v_isShared_2679_ == 0)
{
lean_ctor_set(v___x_2678_, 1, v___f_2686_);
lean_ctor_set(v___x_2678_, 0, v___x_2688_);
v___x_2690_ = v___x_2678_;
goto v_reusejp_2689_;
}
else
{
lean_object* v_reuseFailAlloc_2816_; 
v_reuseFailAlloc_2816_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2816_, 0, v___x_2688_);
lean_ctor_set(v_reuseFailAlloc_2816_, 1, v___f_2686_);
v___x_2690_ = v_reuseFailAlloc_2816_;
goto v_reusejp_2689_;
}
v_reusejp_2689_:
{
lean_object* v___x_2692_; 
lean_inc_ref(v___f_2667_);
lean_inc_ref(v___x_2690_);
if (v_isShared_2575_ == 0)
{
lean_ctor_set(v___x_2574_, 1, v___f_2667_);
lean_ctor_set(v___x_2574_, 0, v___x_2690_);
v___x_2692_ = v___x_2574_;
goto v_reusejp_2691_;
}
else
{
lean_object* v_reuseFailAlloc_2815_; 
v_reuseFailAlloc_2815_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2815_, 0, v___x_2690_);
lean_ctor_set(v_reuseFailAlloc_2815_, 1, v___f_2667_);
v___x_2692_ = v_reuseFailAlloc_2815_;
goto v_reusejp_2691_;
}
v_reusejp_2691_:
{
lean_object* v___x_2693_; lean_object* v_toPartialOrder_2694_; lean_object* v_toLE_2695_; lean_object* v_toLT_2696_; lean_object* v___x_2698_; uint8_t v_isShared_2699_; uint8_t v_isSharedCheck_2814_; 
v___x_2693_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2692_);
v_toPartialOrder_2694_ = lean_ctor_get(v___x_2693_, 0);
lean_inc_ref(v_toPartialOrder_2694_);
v_toLE_2695_ = lean_ctor_get(v_toPartialOrder_2694_, 0);
v_toLT_2696_ = lean_ctor_get(v_toPartialOrder_2694_, 1);
v_isSharedCheck_2814_ = !lean_is_exclusive(v_toPartialOrder_2694_);
if (v_isSharedCheck_2814_ == 0)
{
v___x_2698_ = v_toPartialOrder_2694_;
v_isShared_2699_ = v_isSharedCheck_2814_;
goto v_resetjp_2697_;
}
else
{
lean_inc(v_toLT_2696_);
lean_inc(v_toLE_2695_);
lean_dec(v_toPartialOrder_2694_);
v___x_2698_ = lean_box(0);
v_isShared_2699_ = v_isSharedCheck_2814_;
goto v_resetjp_2697_;
}
v_resetjp_2697_:
{
lean_object* v_bot_2700_; lean_object* v_hnot_2701_; lean_object* v_sdiff_2702_; lean_object* v_top_2703_; lean_object* v___f_2704_; lean_object* v___f_2705_; lean_object* v_coheytingAlgebra_2706_; lean_object* v_toGeneralizedCoheytingAlgebra_2707_; lean_object* v_toLattice_2708_; lean_object* v_toOrderTop_2709_; lean_object* v_toHNot_2710_; lean_object* v_toSDiff_2711_; lean_object* v_toSemilatticeSup_2712_; lean_object* v___x_2713_; lean_object* v_toPartialOrder_2714_; lean_object* v_toLE_2715_; lean_object* v_toLT_2716_; lean_object* v___x_2718_; uint8_t v_isShared_2719_; uint8_t v_isSharedCheck_2813_; 
lean_inc_n(v_toFun_2567_, 4);
v_bot_2700_ = lean_apply_1(v_toFun_2567_, v_toOrderBot_2649_);
lean_inc_ref_n(v_e_2555_, 2);
v_hnot_2701_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__17), 4, 3);
lean_closure_set(v_hnot_2701_, 0, v_e_2555_);
lean_closure_set(v_hnot_2701_, 1, v_toHNot_2648_);
lean_closure_set(v_hnot_2701_, 2, v_toFun_2567_);
v_sdiff_2702_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_2702_, 0, v___f_2586_);
lean_closure_set(v_sdiff_2702_, 1, v_e_2555_);
lean_closure_set(v_sdiff_2702_, 2, v_toSDiff_2650_);
lean_closure_set(v_sdiff_2702_, 3, v_toFun_2567_);
v_top_2703_ = lean_apply_1(v_toFun_2567_, v_toOrderTop_2647_);
v___f_2704_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2704_, 0, v___x_2693_);
v___f_2705_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2705_, 0, v___x_2690_);
v_coheytingAlgebra_2706_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2704_, v___f_2705_, v_toLE_2695_, v_toLT_2696_, v_bot_2700_, v_top_2703_, v_hnot_2701_, v_sdiff_2702_);
v_toGeneralizedCoheytingAlgebra_2707_ = lean_ctor_get(v_coheytingAlgebra_2706_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2707_);
v_toLattice_2708_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2707_, 0);
lean_inc_ref(v_toLattice_2708_);
v_toOrderTop_2709_ = lean_ctor_get(v_coheytingAlgebra_2706_, 1);
lean_inc(v_toOrderTop_2709_);
v_toHNot_2710_ = lean_ctor_get(v_coheytingAlgebra_2706_, 2);
lean_inc(v_toHNot_2710_);
lean_dec_ref(v_coheytingAlgebra_2706_);
v_toSDiff_2711_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2707_, 2);
lean_inc(v_toSDiff_2711_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_2707_);
v_toSemilatticeSup_2712_ = lean_ctor_get(v_toLattice_2708_, 0);
lean_inc_ref(v_toSemilatticeSup_2712_);
v___x_2713_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_2708_);
v_toPartialOrder_2714_ = lean_ctor_get(v___x_2713_, 0);
lean_inc_ref(v_toPartialOrder_2714_);
v_toLE_2715_ = lean_ctor_get(v_toPartialOrder_2714_, 0);
v_toLT_2716_ = lean_ctor_get(v_toPartialOrder_2714_, 1);
v_isSharedCheck_2813_ = !lean_is_exclusive(v_toPartialOrder_2714_);
if (v_isSharedCheck_2813_ == 0)
{
v___x_2718_ = v_toPartialOrder_2714_;
v_isShared_2719_ = v_isSharedCheck_2813_;
goto v_resetjp_2717_;
}
else
{
lean_inc(v_toLT_2716_);
lean_inc(v_toLE_2715_);
lean_dec(v_toPartialOrder_2714_);
v___x_2718_ = lean_box(0);
v_isShared_2719_ = v_isSharedCheck_2813_;
goto v_resetjp_2717_;
}
v_resetjp_2717_:
{
lean_object* v_compl_2720_; lean_object* v_himp_2721_; lean_object* v___f_2722_; lean_object* v___f_2723_; lean_object* v___x_2725_; 
lean_inc(v_toFun_2567_);
lean_inc_ref(v_e_2555_);
v_compl_2720_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_2720_, 0, v_e_2555_);
lean_closure_set(v_compl_2720_, 1, v_toCompl_2636_);
lean_closure_set(v_compl_2720_, 2, v_toFun_2567_);
v_himp_2721_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_2721_, 0, v___f_2586_);
lean_closure_set(v_himp_2721_, 1, v_e_2555_);
lean_closure_set(v_himp_2721_, 2, v_toHImp_2640_);
lean_closure_set(v_himp_2721_, 3, v_toFun_2567_);
v___f_2722_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2722_, 0, v_toSemilatticeSup_2712_);
v___f_2723_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2723_, 0, v___x_2713_);
if (v_isShared_2719_ == 0)
{
v___x_2725_ = v___x_2718_;
goto v_reusejp_2724_;
}
else
{
lean_object* v_reuseFailAlloc_2812_; 
v_reuseFailAlloc_2812_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2812_, 0, v_toLE_2715_);
lean_ctor_set(v_reuseFailAlloc_2812_, 1, v_toLT_2716_);
v___x_2725_ = v_reuseFailAlloc_2812_;
goto v_reusejp_2724_;
}
v_reusejp_2724_:
{
lean_object* v___x_2727_; 
if (v_isShared_2699_ == 0)
{
lean_ctor_set(v___x_2698_, 1, v___f_2686_);
lean_ctor_set(v___x_2698_, 0, v___x_2725_);
v___x_2727_ = v___x_2698_;
goto v_reusejp_2726_;
}
else
{
lean_object* v_reuseFailAlloc_2811_; 
v_reuseFailAlloc_2811_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2811_, 0, v___x_2725_);
lean_ctor_set(v_reuseFailAlloc_2811_, 1, v___f_2686_);
v___x_2727_ = v_reuseFailAlloc_2811_;
goto v_reusejp_2726_;
}
v_reusejp_2726_:
{
lean_object* v___x_2729_; 
if (v_isShared_2570_ == 0)
{
lean_ctor_set(v___x_2569_, 1, v___f_2667_);
lean_ctor_set(v___x_2569_, 0, v___x_2727_);
v___x_2729_ = v___x_2569_;
goto v_reusejp_2728_;
}
else
{
lean_object* v_reuseFailAlloc_2810_; 
v_reuseFailAlloc_2810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2810_, 0, v___x_2727_);
lean_ctor_set(v_reuseFailAlloc_2810_, 1, v___f_2667_);
v___x_2729_ = v_reuseFailAlloc_2810_;
goto v_reusejp_2728_;
}
v_reusejp_2728_:
{
lean_object* v___x_2731_; 
lean_inc_ref(v_himp_2721_);
lean_inc(v_toOrderTop_2709_);
if (v_isShared_2643_ == 0)
{
lean_ctor_set(v___x_2642_, 2, v_himp_2721_);
lean_ctor_set(v___x_2642_, 1, v_toOrderTop_2709_);
lean_ctor_set(v___x_2642_, 0, v___x_2729_);
v___x_2731_ = v___x_2642_;
goto v_reusejp_2730_;
}
else
{
lean_object* v_reuseFailAlloc_2809_; 
v_reuseFailAlloc_2809_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2809_, 0, v___x_2729_);
lean_ctor_set(v_reuseFailAlloc_2809_, 1, v_toOrderTop_2709_);
lean_ctor_set(v_reuseFailAlloc_2809_, 2, v_himp_2721_);
v___x_2731_ = v_reuseFailAlloc_2809_;
goto v_reusejp_2730_;
}
v_reusejp_2730_:
{
lean_object* v___x_2733_; 
lean_inc_ref(v_compl_2720_);
lean_inc(v_bot_2685_);
if (v_isShared_2639_ == 0)
{
lean_ctor_set(v___x_2638_, 2, v_compl_2720_);
lean_ctor_set(v___x_2638_, 1, v_bot_2685_);
lean_ctor_set(v___x_2638_, 0, v___x_2731_);
v___x_2733_ = v___x_2638_;
goto v_reusejp_2732_;
}
else
{
lean_object* v_reuseFailAlloc_2808_; 
v_reuseFailAlloc_2808_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2808_, 0, v___x_2731_);
lean_ctor_set(v_reuseFailAlloc_2808_, 1, v_bot_2685_);
lean_ctor_set(v_reuseFailAlloc_2808_, 2, v_compl_2720_);
v___x_2733_ = v_reuseFailAlloc_2808_;
goto v_reusejp_2732_;
}
v_reusejp_2732_:
{
lean_object* v___x_2734_; lean_object* v_toGeneralizedCoheytingAlgebra_2735_; lean_object* v_toHNot_2736_; lean_object* v_toSDiff_2737_; lean_object* v___x_2739_; uint8_t v_isShared_2740_; uint8_t v_isSharedCheck_2805_; 
lean_inc(v_bot_2685_);
v___x_2734_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2723_, v___f_2722_, v_toLE_2715_, v_toLT_2716_, v_bot_2685_, v_toOrderTop_2709_, v_toHNot_2710_, v_toSDiff_2711_);
v_toGeneralizedCoheytingAlgebra_2735_ = lean_ctor_get(v___x_2734_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2735_);
v_toHNot_2736_ = lean_ctor_get(v___x_2734_, 2);
lean_inc(v_toHNot_2736_);
lean_dec_ref(v___x_2734_);
v_toSDiff_2737_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2735_, 2);
v_isSharedCheck_2805_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_2735_);
if (v_isSharedCheck_2805_ == 0)
{
lean_object* v_unused_2806_; lean_object* v_unused_2807_; 
v_unused_2806_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2735_, 1);
lean_dec(v_unused_2806_);
v_unused_2807_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2735_, 0);
lean_dec(v_unused_2807_);
v___x_2739_ = v_toGeneralizedCoheytingAlgebra_2735_;
v_isShared_2740_ = v_isSharedCheck_2805_;
goto v_resetjp_2738_;
}
else
{
lean_inc(v_toSDiff_2737_);
lean_dec(v_toGeneralizedCoheytingAlgebra_2735_);
v___x_2739_ = lean_box(0);
v_isShared_2740_ = v_isSharedCheck_2805_;
goto v_resetjp_2738_;
}
v_resetjp_2738_:
{
lean_object* v_biheytingAlgebra_2742_; 
if (v_isShared_2740_ == 0)
{
lean_ctor_set(v___x_2739_, 2, v_toHNot_2736_);
lean_ctor_set(v___x_2739_, 1, v_toSDiff_2737_);
lean_ctor_set(v___x_2739_, 0, v___x_2733_);
v_biheytingAlgebra_2742_ = v___x_2739_;
goto v_reusejp_2741_;
}
else
{
lean_object* v_reuseFailAlloc_2804_; 
v_reuseFailAlloc_2804_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2804_, 0, v___x_2733_);
lean_ctor_set(v_reuseFailAlloc_2804_, 1, v_toSDiff_2737_);
lean_ctor_set(v_reuseFailAlloc_2804_, 2, v_toHNot_2736_);
v_biheytingAlgebra_2742_ = v_reuseFailAlloc_2804_;
goto v_reusejp_2741_;
}
v_reusejp_2741_:
{
lean_object* v___x_2743_; lean_object* v___x_2744_; lean_object* v_toPartialOrder_2745_; lean_object* v_toInfSet_2746_; lean_object* v___x_2748_; uint8_t v_isShared_2749_; uint8_t v_isSharedCheck_2803_; 
v___x_2743_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2628_);
lean_inc_ref(v_completeLattice_2631_);
v___x_2744_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_2631_);
v_toPartialOrder_2745_ = lean_ctor_get(v___x_2744_, 0);
v_toInfSet_2746_ = lean_ctor_get(v___x_2744_, 1);
v_isSharedCheck_2803_ = !lean_is_exclusive(v___x_2744_);
if (v_isSharedCheck_2803_ == 0)
{
v___x_2748_ = v___x_2744_;
v_isShared_2749_ = v_isSharedCheck_2803_;
goto v_resetjp_2747_;
}
else
{
lean_inc(v_toInfSet_2746_);
lean_inc(v_toPartialOrder_2745_);
lean_dec(v___x_2744_);
v___x_2748_ = lean_box(0);
v_isShared_2749_ = v_isSharedCheck_2803_;
goto v_resetjp_2747_;
}
v_resetjp_2747_:
{
lean_object* v_toLE_2750_; lean_object* v_toLT_2751_; lean_object* v___x_2753_; uint8_t v_isShared_2754_; uint8_t v_isSharedCheck_2802_; 
v_toLE_2750_ = lean_ctor_get(v_toPartialOrder_2745_, 0);
v_toLT_2751_ = lean_ctor_get(v_toPartialOrder_2745_, 1);
v_isSharedCheck_2802_ = !lean_is_exclusive(v_toPartialOrder_2745_);
if (v_isSharedCheck_2802_ == 0)
{
v___x_2753_ = v_toPartialOrder_2745_;
v_isShared_2754_ = v_isSharedCheck_2802_;
goto v_resetjp_2752_;
}
else
{
lean_inc(v_toLT_2751_);
lean_inc(v_toLE_2750_);
lean_dec(v_toPartialOrder_2745_);
v___x_2753_ = lean_box(0);
v_isShared_2754_ = v_isSharedCheck_2802_;
goto v_resetjp_2752_;
}
v_resetjp_2752_:
{
lean_object* v___x_2755_; lean_object* v_toSupSet_2756_; lean_object* v___x_2758_; uint8_t v_isShared_2759_; uint8_t v_isSharedCheck_2800_; 
v___x_2755_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_2631_);
v_toSupSet_2756_ = lean_ctor_get(v___x_2755_, 1);
v_isSharedCheck_2800_ = !lean_is_exclusive(v___x_2755_);
if (v_isSharedCheck_2800_ == 0)
{
lean_object* v_unused_2801_; 
v_unused_2801_ = lean_ctor_get(v___x_2755_, 0);
lean_dec(v_unused_2801_);
v___x_2758_ = v___x_2755_;
v_isShared_2759_ = v_isSharedCheck_2800_;
goto v_resetjp_2757_;
}
else
{
lean_inc(v_toSupSet_2756_);
lean_dec(v___x_2755_);
v___x_2758_ = lean_box(0);
v_isShared_2759_ = v_isSharedCheck_2800_;
goto v_resetjp_2757_;
}
v_resetjp_2757_:
{
lean_object* v___x_2760_; lean_object* v_toGeneralizedCoheytingAlgebra_2761_; lean_object* v_toOrderTop_2762_; lean_object* v_toHNot_2763_; lean_object* v_toSDiff_2764_; lean_object* v___x_2766_; uint8_t v_isShared_2767_; uint8_t v_isSharedCheck_2797_; 
v___x_2760_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_biheytingAlgebra_2742_);
v_toGeneralizedCoheytingAlgebra_2761_ = lean_ctor_get(v___x_2760_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2761_);
v_toOrderTop_2762_ = lean_ctor_get(v___x_2760_, 1);
lean_inc(v_toOrderTop_2762_);
v_toHNot_2763_ = lean_ctor_get(v___x_2760_, 2);
lean_inc(v_toHNot_2763_);
lean_dec_ref(v___x_2760_);
v_toSDiff_2764_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2761_, 2);
v_isSharedCheck_2797_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_2761_);
if (v_isSharedCheck_2797_ == 0)
{
lean_object* v_unused_2798_; lean_object* v_unused_2799_; 
v_unused_2798_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2761_, 1);
lean_dec(v_unused_2798_);
v_unused_2799_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2761_, 0);
lean_dec(v_unused_2799_);
v___x_2766_ = v_toGeneralizedCoheytingAlgebra_2761_;
v_isShared_2767_ = v_isSharedCheck_2797_;
goto v_resetjp_2765_;
}
else
{
lean_inc(v_toSDiff_2764_);
lean_dec(v_toGeneralizedCoheytingAlgebra_2761_);
v___x_2766_ = lean_box(0);
v_isShared_2767_ = v_isSharedCheck_2797_;
goto v_resetjp_2765_;
}
v_resetjp_2765_:
{
lean_object* v___f_2768_; lean_object* v___f_2769_; lean_object* v___x_2771_; 
v___f_2768_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2768_, 0, v___x_2626_);
v___f_2769_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2769_, 0, v___x_2743_);
if (v_isShared_2754_ == 0)
{
v___x_2771_ = v___x_2753_;
goto v_reusejp_2770_;
}
else
{
lean_object* v_reuseFailAlloc_2796_; 
v_reuseFailAlloc_2796_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2796_, 0, v_toLE_2750_);
lean_ctor_set(v_reuseFailAlloc_2796_, 1, v_toLT_2751_);
v___x_2771_ = v_reuseFailAlloc_2796_;
goto v_reusejp_2770_;
}
v_reusejp_2770_:
{
lean_object* v___x_2773_; 
if (v_isShared_2759_ == 0)
{
lean_ctor_set(v___x_2758_, 1, v___f_2622_);
lean_ctor_set(v___x_2758_, 0, v___x_2771_);
v___x_2773_ = v___x_2758_;
goto v_reusejp_2772_;
}
else
{
lean_object* v_reuseFailAlloc_2795_; 
v_reuseFailAlloc_2795_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2795_, 0, v___x_2771_);
lean_ctor_set(v_reuseFailAlloc_2795_, 1, v___f_2622_);
v___x_2773_ = v_reuseFailAlloc_2795_;
goto v_reusejp_2772_;
}
v_reusejp_2772_:
{
lean_object* v___x_2775_; 
if (v_isShared_2749_ == 0)
{
lean_ctor_set(v___x_2748_, 1, v___f_2600_);
lean_ctor_set(v___x_2748_, 0, v___x_2773_);
v___x_2775_ = v___x_2748_;
goto v_reusejp_2774_;
}
else
{
lean_object* v_reuseFailAlloc_2794_; 
v_reuseFailAlloc_2794_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2794_, 0, v___x_2773_);
lean_ctor_set(v_reuseFailAlloc_2794_, 1, v___f_2600_);
v___x_2775_ = v_reuseFailAlloc_2794_;
goto v_reusejp_2774_;
}
v_reusejp_2774_:
{
lean_object* v___x_2776_; lean_object* v___x_2777_; lean_object* v___x_2779_; 
lean_inc(v_bot_2685_);
lean_inc(v_toOrderTop_2762_);
v___x_2776_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2776_, 0, v_toOrderTop_2762_);
lean_ctor_set(v___x_2776_, 1, v_bot_2685_);
v___x_2777_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2777_, 0, v___x_2775_);
lean_ctor_set(v___x_2777_, 1, v_toSupSet_2756_);
lean_ctor_set(v___x_2777_, 2, v_toInfSet_2746_);
lean_ctor_set(v___x_2777_, 3, v___x_2776_);
if (v_isShared_2767_ == 0)
{
lean_ctor_set(v___x_2766_, 2, v_compl_2720_);
lean_ctor_set(v___x_2766_, 1, v_himp_2721_);
lean_ctor_set(v___x_2766_, 0, v___x_2777_);
v___x_2779_ = v___x_2766_;
goto v_reusejp_2778_;
}
else
{
lean_object* v_reuseFailAlloc_2793_; 
v_reuseFailAlloc_2793_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2793_, 0, v___x_2777_);
lean_ctor_set(v_reuseFailAlloc_2793_, 1, v_himp_2721_);
lean_ctor_set(v_reuseFailAlloc_2793_, 2, v_compl_2720_);
v___x_2779_ = v_reuseFailAlloc_2793_;
goto v_reusejp_2778_;
}
v_reusejp_2778_:
{
lean_object* v___x_2780_; lean_object* v_toGeneralizedCoheytingAlgebra_2781_; lean_object* v_toHNot_2782_; lean_object* v_toSDiff_2783_; lean_object* v___x_2785_; uint8_t v_isShared_2786_; uint8_t v_isSharedCheck_2790_; 
v___x_2780_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2769_, v___f_2768_, v_toLE_2750_, v_toLT_2751_, v_bot_2685_, v_toOrderTop_2762_, v_toHNot_2763_, v_toSDiff_2764_);
v_toGeneralizedCoheytingAlgebra_2781_ = lean_ctor_get(v___x_2780_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2781_);
v_toHNot_2782_ = lean_ctor_get(v___x_2780_, 2);
lean_inc(v_toHNot_2782_);
lean_dec_ref(v___x_2780_);
v_toSDiff_2783_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2781_, 2);
v_isSharedCheck_2790_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_2781_);
if (v_isSharedCheck_2790_ == 0)
{
lean_object* v_unused_2791_; lean_object* v_unused_2792_; 
v_unused_2791_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2781_, 1);
lean_dec(v_unused_2791_);
v_unused_2792_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2781_, 0);
lean_dec(v_unused_2792_);
v___x_2785_ = v_toGeneralizedCoheytingAlgebra_2781_;
v_isShared_2786_ = v_isSharedCheck_2790_;
goto v_resetjp_2784_;
}
else
{
lean_inc(v_toSDiff_2783_);
lean_dec(v_toGeneralizedCoheytingAlgebra_2781_);
v___x_2785_ = lean_box(0);
v_isShared_2786_ = v_isSharedCheck_2790_;
goto v_resetjp_2784_;
}
v_resetjp_2784_:
{
lean_object* v___x_2788_; 
if (v_isShared_2786_ == 0)
{
lean_ctor_set(v___x_2785_, 2, v_toHNot_2782_);
lean_ctor_set(v___x_2785_, 1, v_toSDiff_2783_);
lean_ctor_set(v___x_2785_, 0, v___x_2779_);
v___x_2788_ = v___x_2785_;
goto v_reusejp_2787_;
}
else
{
lean_object* v_reuseFailAlloc_2789_; 
v_reuseFailAlloc_2789_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2789_, 0, v___x_2779_);
lean_ctor_set(v_reuseFailAlloc_2789_, 1, v_toSDiff_2783_);
lean_ctor_set(v_reuseFailAlloc_2789_, 2, v_toHNot_2782_);
v___x_2788_ = v_reuseFailAlloc_2789_;
goto v_reusejp_2787_;
}
v_reusejp_2787_:
{
return v___x_2788_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completelyDistribLattice___redArg(lean_object* v_e_2853_, lean_object* v_inst_2854_){
_start:
{
lean_object* v___x_2855_; lean_object* v___x_2856_; lean_object* v_toCompleteLattice_2857_; lean_object* v_toBoundedOrder_2858_; lean_object* v_toLattice_2859_; lean_object* v_toOrderTop_2860_; lean_object* v_toOrderBot_2861_; lean_object* v___x_2863_; uint8_t v_isShared_2864_; uint8_t v_isSharedCheck_3204_; 
v___x_2855_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v_inst_2854_);
lean_inc_ref(v___x_2855_);
v___x_2856_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_2855_);
v_toCompleteLattice_2857_ = lean_ctor_get(v___x_2856_, 0);
lean_inc_ref(v_toCompleteLattice_2857_);
lean_dec_ref(v___x_2856_);
v_toBoundedOrder_2858_ = lean_ctor_get(v_toCompleteLattice_2857_, 3);
lean_inc_ref(v_toBoundedOrder_2858_);
v_toLattice_2859_ = lean_ctor_get(v_toCompleteLattice_2857_, 0);
lean_inc_ref(v_toLattice_2859_);
v_toOrderTop_2860_ = lean_ctor_get(v_toBoundedOrder_2858_, 0);
v_toOrderBot_2861_ = lean_ctor_get(v_toBoundedOrder_2858_, 1);
v_isSharedCheck_3204_ = !lean_is_exclusive(v_toBoundedOrder_2858_);
if (v_isSharedCheck_3204_ == 0)
{
v___x_2863_ = v_toBoundedOrder_2858_;
v_isShared_2864_ = v_isSharedCheck_3204_;
goto v_resetjp_2862_;
}
else
{
lean_inc(v_toOrderBot_2861_);
lean_inc(v_toOrderTop_2860_);
lean_dec(v_toBoundedOrder_2858_);
v___x_2863_ = lean_box(0);
v_isShared_2864_ = v_isSharedCheck_3204_;
goto v_resetjp_2862_;
}
v_resetjp_2862_:
{
lean_object* v___x_2865_; lean_object* v_toFun_2866_; lean_object* v___x_2868_; uint8_t v_isShared_2869_; uint8_t v_isSharedCheck_3202_; 
lean_inc_ref(v_e_2853_);
v___x_2865_ = lp_mathlib_Equiv_symm___redArg(v_e_2853_);
v_toFun_2866_ = lean_ctor_get(v___x_2865_, 0);
v_isSharedCheck_3202_ = !lean_is_exclusive(v___x_2865_);
if (v_isSharedCheck_3202_ == 0)
{
lean_object* v_unused_3203_; 
v_unused_3203_ = lean_ctor_get(v___x_2865_, 1);
lean_dec(v_unused_3203_);
v___x_2868_ = v___x_2865_;
v_isShared_2869_ = v_isSharedCheck_3202_;
goto v_resetjp_2867_;
}
else
{
lean_inc(v_toFun_2866_);
lean_dec(v___x_2865_);
v___x_2868_ = lean_box(0);
v_isShared_2869_ = v_isSharedCheck_3202_;
goto v_resetjp_2867_;
}
v_resetjp_2867_:
{
lean_object* v___x_2870_; lean_object* v_toSupSet_2871_; lean_object* v___x_2873_; uint8_t v_isShared_2874_; uint8_t v_isSharedCheck_3200_; 
lean_inc_ref(v_toCompleteLattice_2857_);
v___x_2870_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_2857_);
v_toSupSet_2871_ = lean_ctor_get(v___x_2870_, 1);
v_isSharedCheck_3200_ = !lean_is_exclusive(v___x_2870_);
if (v_isSharedCheck_3200_ == 0)
{
lean_object* v_unused_3201_; 
v_unused_3201_ = lean_ctor_get(v___x_2870_, 0);
lean_dec(v_unused_3201_);
v___x_2873_ = v___x_2870_;
v_isShared_2874_ = v_isSharedCheck_3200_;
goto v_resetjp_2872_;
}
else
{
lean_inc(v_toSupSet_2871_);
lean_dec(v___x_2870_);
v___x_2873_ = lean_box(0);
v_isShared_2874_ = v_isSharedCheck_3200_;
goto v_resetjp_2872_;
}
v_resetjp_2872_:
{
lean_object* v___x_2875_; lean_object* v_toInfSet_2876_; lean_object* v___x_2878_; uint8_t v_isShared_2879_; uint8_t v_isSharedCheck_3198_; 
v___x_2875_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_2857_);
v_toInfSet_2876_ = lean_ctor_get(v___x_2875_, 1);
v_isSharedCheck_3198_ = !lean_is_exclusive(v___x_2875_);
if (v_isSharedCheck_3198_ == 0)
{
lean_object* v_unused_3199_; 
v_unused_3199_ = lean_ctor_get(v___x_2875_, 0);
lean_dec(v_unused_3199_);
v___x_2878_ = v___x_2875_;
v_isShared_2879_ = v_isSharedCheck_3198_;
goto v_resetjp_2877_;
}
else
{
lean_inc(v_toInfSet_2876_);
lean_dec(v___x_2875_);
v___x_2878_ = lean_box(0);
v_isShared_2879_ = v_isSharedCheck_3198_;
goto v_resetjp_2877_;
}
v_resetjp_2877_:
{
lean_object* v_toSemilatticeSup_2880_; lean_object* v_inf_2881_; lean_object* v___x_2883_; uint8_t v_isShared_2884_; uint8_t v_isSharedCheck_3197_; 
v_toSemilatticeSup_2880_ = lean_ctor_get(v_toLattice_2859_, 0);
v_inf_2881_ = lean_ctor_get(v_toLattice_2859_, 1);
v_isSharedCheck_3197_ = !lean_is_exclusive(v_toLattice_2859_);
if (v_isSharedCheck_3197_ == 0)
{
v___x_2883_ = v_toLattice_2859_;
v_isShared_2884_ = v_isSharedCheck_3197_;
goto v_resetjp_2882_;
}
else
{
lean_inc(v_inf_2881_);
lean_inc(v_toSemilatticeSup_2880_);
lean_dec(v_toLattice_2859_);
v___x_2883_ = lean_box(0);
v_isShared_2884_ = v_isSharedCheck_3197_;
goto v_resetjp_2882_;
}
v_resetjp_2882_:
{
lean_object* v___f_2885_; lean_object* v_min_2886_; lean_object* v_le_2887_; lean_object* v_lt_2888_; lean_object* v_semilatticeInf_2889_; lean_object* v_toPartialOrder_2890_; lean_object* v___x_2892_; uint8_t v_isShared_2893_; uint8_t v_isSharedCheck_3195_; 
v___f_2885_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_2866_);
lean_inc_ref(v_e_2853_);
v_min_2886_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2886_, 0, v___f_2885_);
lean_closure_set(v_min_2886_, 1, v_e_2853_);
lean_closure_set(v_min_2886_, 2, v_inf_2881_);
lean_closure_set(v_min_2886_, 3, v_toFun_2866_);
v_le_2887_ = lean_box(0);
v_lt_2888_ = lean_box(0);
lean_inc_ref(v_min_2886_);
v_semilatticeInf_2889_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2886_, v_le_2887_, v_lt_2888_);
v_toPartialOrder_2890_ = lean_ctor_get(v_semilatticeInf_2889_, 0);
v_isSharedCheck_3195_ = !lean_is_exclusive(v_semilatticeInf_2889_);
if (v_isSharedCheck_3195_ == 0)
{
lean_object* v_unused_3196_; 
v_unused_3196_ = lean_ctor_get(v_semilatticeInf_2889_, 1);
lean_dec(v_unused_3196_);
v___x_2892_ = v_semilatticeInf_2889_;
v_isShared_2893_ = v_isSharedCheck_3195_;
goto v_resetjp_2891_;
}
else
{
lean_inc(v_toPartialOrder_2890_);
lean_dec(v_semilatticeInf_2889_);
v___x_2892_ = lean_box(0);
v_isShared_2893_ = v_isSharedCheck_3195_;
goto v_resetjp_2891_;
}
v_resetjp_2891_:
{
lean_object* v_toLE_2894_; lean_object* v_toLT_2895_; lean_object* v___x_2897_; uint8_t v_isShared_2898_; uint8_t v_isSharedCheck_3194_; 
v_toLE_2894_ = lean_ctor_get(v_toPartialOrder_2890_, 0);
v_toLT_2895_ = lean_ctor_get(v_toPartialOrder_2890_, 1);
v_isSharedCheck_3194_ = !lean_is_exclusive(v_toPartialOrder_2890_);
if (v_isSharedCheck_3194_ == 0)
{
v___x_2897_ = v_toPartialOrder_2890_;
v_isShared_2898_ = v_isSharedCheck_3194_;
goto v_resetjp_2896_;
}
else
{
lean_inc(v_toLT_2895_);
lean_inc(v_toLE_2894_);
lean_dec(v_toPartialOrder_2890_);
v___x_2897_ = lean_box(0);
v_isShared_2898_ = v_isSharedCheck_3194_;
goto v_resetjp_2896_;
}
v_resetjp_2896_:
{
lean_object* v___f_2899_; lean_object* v___f_2900_; lean_object* v___x_2902_; 
v___f_2899_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_2899_, 0, v_min_2886_);
lean_inc(v_toFun_2866_);
lean_inc_ref(v_e_2853_);
v___f_2900_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_2900_, 0, v_toSemilatticeSup_2880_);
lean_closure_set(v___f_2900_, 1, v___f_2885_);
lean_closure_set(v___f_2900_, 2, v_e_2853_);
lean_closure_set(v___f_2900_, 3, v_toFun_2866_);
if (v_isShared_2898_ == 0)
{
v___x_2902_ = v___x_2897_;
goto v_reusejp_2901_;
}
else
{
lean_object* v_reuseFailAlloc_3193_; 
v_reuseFailAlloc_3193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3193_, 0, v_toLE_2894_);
lean_ctor_set(v_reuseFailAlloc_3193_, 1, v_toLT_2895_);
v___x_2902_ = v_reuseFailAlloc_3193_;
goto v_reusejp_2901_;
}
v_reusejp_2901_:
{
lean_object* v___x_2904_; 
lean_inc_ref(v___f_2900_);
if (v_isShared_2893_ == 0)
{
lean_ctor_set(v___x_2892_, 1, v___f_2900_);
lean_ctor_set(v___x_2892_, 0, v___x_2902_);
v___x_2904_ = v___x_2892_;
goto v_reusejp_2903_;
}
else
{
lean_object* v_reuseFailAlloc_3192_; 
v_reuseFailAlloc_3192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3192_, 0, v___x_2902_);
lean_ctor_set(v_reuseFailAlloc_3192_, 1, v___f_2900_);
v___x_2904_ = v_reuseFailAlloc_3192_;
goto v_reusejp_2903_;
}
v_reusejp_2903_:
{
lean_object* v_lattice_2906_; 
lean_inc_ref(v___f_2899_);
if (v_isShared_2884_ == 0)
{
lean_ctor_set(v___x_2883_, 1, v___f_2899_);
lean_ctor_set(v___x_2883_, 0, v___x_2904_);
v_lattice_2906_ = v___x_2883_;
goto v_reusejp_2905_;
}
else
{
lean_object* v_reuseFailAlloc_3191_; 
v_reuseFailAlloc_3191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3191_, 0, v___x_2904_);
lean_ctor_set(v_reuseFailAlloc_3191_, 1, v___f_2899_);
v_lattice_2906_ = v_reuseFailAlloc_3191_;
goto v_reusejp_2905_;
}
v_reusejp_2905_:
{
lean_object* v___x_2907_; lean_object* v_toPartialOrder_2908_; lean_object* v___x_2910_; uint8_t v_isShared_2911_; uint8_t v_isSharedCheck_3189_; 
v___x_2907_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2906_);
v_toPartialOrder_2908_ = lean_ctor_get(v___x_2907_, 0);
v_isSharedCheck_3189_ = !lean_is_exclusive(v___x_2907_);
if (v_isSharedCheck_3189_ == 0)
{
lean_object* v_unused_3190_; 
v_unused_3190_ = lean_ctor_get(v___x_2907_, 1);
lean_dec(v_unused_3190_);
v___x_2910_ = v___x_2907_;
v_isShared_2911_ = v_isSharedCheck_3189_;
goto v_resetjp_2909_;
}
else
{
lean_inc(v_toPartialOrder_2908_);
lean_dec(v___x_2907_);
v___x_2910_ = lean_box(0);
v_isShared_2911_ = v_isSharedCheck_3189_;
goto v_resetjp_2909_;
}
v_resetjp_2909_:
{
lean_object* v_toLE_2912_; lean_object* v_toLT_2913_; lean_object* v___x_2915_; uint8_t v_isShared_2916_; uint8_t v_isSharedCheck_3188_; 
v_toLE_2912_ = lean_ctor_get(v_toPartialOrder_2908_, 0);
v_toLT_2913_ = lean_ctor_get(v_toPartialOrder_2908_, 1);
v_isSharedCheck_3188_ = !lean_is_exclusive(v_toPartialOrder_2908_);
if (v_isSharedCheck_3188_ == 0)
{
v___x_2915_ = v_toPartialOrder_2908_;
v_isShared_2916_ = v_isSharedCheck_3188_;
goto v_resetjp_2914_;
}
else
{
lean_inc(v_toLT_2913_);
lean_inc(v_toLE_2912_);
lean_dec(v_toPartialOrder_2908_);
v___x_2915_ = lean_box(0);
v_isShared_2916_ = v_isSharedCheck_3188_;
goto v_resetjp_2914_;
}
v_resetjp_2914_:
{
lean_object* v_top_2917_; lean_object* v_bot_2918_; lean_object* v_supSet_2919_; lean_object* v_infSet_2920_; lean_object* v___f_2921_; lean_object* v___x_2923_; 
lean_inc_n(v_toFun_2866_, 4);
v_top_2917_ = lean_apply_1(v_toFun_2866_, v_toOrderTop_2860_);
v_bot_2918_ = lean_apply_1(v_toFun_2866_, v_toOrderBot_2861_);
v_supSet_2919_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_2919_, 0, v_toSupSet_2871_);
lean_closure_set(v_supSet_2919_, 1, v_toFun_2866_);
v_infSet_2920_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_2920_, 0, v_toInfSet_2876_);
lean_closure_set(v_infSet_2920_, 1, v_toFun_2866_);
v___f_2921_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_2921_, 0, v___f_2900_);
if (v_isShared_2916_ == 0)
{
v___x_2923_ = v___x_2915_;
goto v_reusejp_2922_;
}
else
{
lean_object* v_reuseFailAlloc_3187_; 
v_reuseFailAlloc_3187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3187_, 0, v_toLE_2912_);
lean_ctor_set(v_reuseFailAlloc_3187_, 1, v_toLT_2913_);
v___x_2923_ = v_reuseFailAlloc_3187_;
goto v_reusejp_2922_;
}
v_reusejp_2922_:
{
lean_object* v___x_2925_; 
lean_inc_ref(v___f_2921_);
if (v_isShared_2911_ == 0)
{
lean_ctor_set(v___x_2910_, 1, v___f_2921_);
lean_ctor_set(v___x_2910_, 0, v___x_2923_);
v___x_2925_ = v___x_2910_;
goto v_reusejp_2924_;
}
else
{
lean_object* v_reuseFailAlloc_3186_; 
v_reuseFailAlloc_3186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3186_, 0, v___x_2923_);
lean_ctor_set(v_reuseFailAlloc_3186_, 1, v___f_2921_);
v___x_2925_ = v_reuseFailAlloc_3186_;
goto v_reusejp_2924_;
}
v_reusejp_2924_:
{
lean_object* v___x_2927_; 
lean_inc_ref(v___f_2899_);
lean_inc_ref(v___x_2925_);
if (v_isShared_2879_ == 0)
{
lean_ctor_set(v___x_2878_, 1, v___f_2899_);
lean_ctor_set(v___x_2878_, 0, v___x_2925_);
v___x_2927_ = v___x_2878_;
goto v_reusejp_2926_;
}
else
{
lean_object* v_reuseFailAlloc_3185_; 
v_reuseFailAlloc_3185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3185_, 0, v___x_2925_);
lean_ctor_set(v_reuseFailAlloc_3185_, 1, v___f_2899_);
v___x_2927_ = v_reuseFailAlloc_3185_;
goto v_reusejp_2926_;
}
v_reusejp_2926_:
{
lean_object* v___x_2929_; 
if (v_isShared_2864_ == 0)
{
lean_ctor_set(v___x_2863_, 1, v_bot_2918_);
lean_ctor_set(v___x_2863_, 0, v_top_2917_);
v___x_2929_ = v___x_2863_;
goto v_reusejp_2928_;
}
else
{
lean_object* v_reuseFailAlloc_3184_; 
v_reuseFailAlloc_3184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3184_, 0, v_top_2917_);
lean_ctor_set(v_reuseFailAlloc_3184_, 1, v_bot_2918_);
v___x_2929_ = v_reuseFailAlloc_3184_;
goto v_reusejp_2928_;
}
v_reusejp_2928_:
{
lean_object* v_completeLattice_2930_; lean_object* v___x_2931_; lean_object* v_toHeytingAlgebra_2932_; lean_object* v_toGeneralizedHeytingAlgebra_2933_; lean_object* v_toOrderBot_2934_; lean_object* v_toCompl_2935_; lean_object* v___x_2937_; uint8_t v_isShared_2938_; uint8_t v_isSharedCheck_3183_; 
lean_inc_ref(v___x_2927_);
v_completeLattice_2930_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_2930_, 0, v___x_2927_);
lean_ctor_set(v_completeLattice_2930_, 1, v_supSet_2919_);
lean_ctor_set(v_completeLattice_2930_, 2, v_infSet_2920_);
lean_ctor_set(v_completeLattice_2930_, 3, v___x_2929_);
v___x_2931_ = lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra___redArg(v___x_2855_);
v_toHeytingAlgebra_2932_ = lean_ctor_get(v___x_2931_, 0);
lean_inc_ref(v_toHeytingAlgebra_2932_);
v_toGeneralizedHeytingAlgebra_2933_ = lean_ctor_get(v_toHeytingAlgebra_2932_, 0);
v_toOrderBot_2934_ = lean_ctor_get(v_toHeytingAlgebra_2932_, 1);
v_toCompl_2935_ = lean_ctor_get(v_toHeytingAlgebra_2932_, 2);
v_isSharedCheck_3183_ = !lean_is_exclusive(v_toHeytingAlgebra_2932_);
if (v_isSharedCheck_3183_ == 0)
{
v___x_2937_ = v_toHeytingAlgebra_2932_;
v_isShared_2938_ = v_isSharedCheck_3183_;
goto v_resetjp_2936_;
}
else
{
lean_inc(v_toCompl_2935_);
lean_inc(v_toOrderBot_2934_);
lean_inc(v_toGeneralizedHeytingAlgebra_2933_);
lean_dec(v_toHeytingAlgebra_2932_);
v___x_2937_ = lean_box(0);
v_isShared_2938_ = v_isSharedCheck_3183_;
goto v_resetjp_2936_;
}
v_resetjp_2936_:
{
lean_object* v_toHImp_2939_; lean_object* v___x_2941_; uint8_t v_isShared_2942_; uint8_t v_isSharedCheck_3180_; 
v_toHImp_2939_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2933_, 2);
v_isSharedCheck_3180_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_2933_);
if (v_isSharedCheck_3180_ == 0)
{
lean_object* v_unused_3181_; lean_object* v_unused_3182_; 
v_unused_3181_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2933_, 1);
lean_dec(v_unused_3181_);
v_unused_3182_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2933_, 0);
lean_dec(v_unused_3182_);
v___x_2941_ = v_toGeneralizedHeytingAlgebra_2933_;
v_isShared_2942_ = v_isSharedCheck_3180_;
goto v_resetjp_2940_;
}
else
{
lean_inc(v_toHImp_2939_);
lean_dec(v_toGeneralizedHeytingAlgebra_2933_);
v___x_2941_ = lean_box(0);
v_isShared_2942_ = v_isSharedCheck_3180_;
goto v_resetjp_2940_;
}
v_resetjp_2940_:
{
lean_object* v___x_2943_; lean_object* v_toGeneralizedCoheytingAlgebra_2944_; lean_object* v_toLattice_2945_; lean_object* v_toOrderTop_2946_; lean_object* v_toHNot_2947_; lean_object* v_toOrderBot_2948_; lean_object* v_toSDiff_2949_; lean_object* v_toSemilatticeSup_2950_; lean_object* v_inf_2951_; lean_object* v___x_2953_; uint8_t v_isShared_2954_; uint8_t v_isSharedCheck_3179_; 
v___x_2943_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v___x_2931_);
v_toGeneralizedCoheytingAlgebra_2944_ = lean_ctor_get(v___x_2943_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2944_);
v_toLattice_2945_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2944_, 0);
lean_inc_ref(v_toLattice_2945_);
v_toOrderTop_2946_ = lean_ctor_get(v___x_2943_, 1);
lean_inc(v_toOrderTop_2946_);
v_toHNot_2947_ = lean_ctor_get(v___x_2943_, 2);
lean_inc(v_toHNot_2947_);
lean_dec_ref(v___x_2943_);
v_toOrderBot_2948_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2944_, 1);
lean_inc(v_toOrderBot_2948_);
v_toSDiff_2949_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2944_, 2);
lean_inc(v_toSDiff_2949_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_2944_);
v_toSemilatticeSup_2950_ = lean_ctor_get(v_toLattice_2945_, 0);
v_inf_2951_ = lean_ctor_get(v_toLattice_2945_, 1);
v_isSharedCheck_3179_ = !lean_is_exclusive(v_toLattice_2945_);
if (v_isSharedCheck_3179_ == 0)
{
v___x_2953_ = v_toLattice_2945_;
v_isShared_2954_ = v_isSharedCheck_3179_;
goto v_resetjp_2952_;
}
else
{
lean_inc(v_inf_2951_);
lean_inc(v_toSemilatticeSup_2950_);
lean_dec(v_toLattice_2945_);
v___x_2953_ = lean_box(0);
v_isShared_2954_ = v_isSharedCheck_3179_;
goto v_resetjp_2952_;
}
v_resetjp_2952_:
{
lean_object* v_min_2955_; lean_object* v_semilatticeInf_2956_; lean_object* v_toPartialOrder_2957_; lean_object* v___x_2959_; uint8_t v_isShared_2960_; uint8_t v_isSharedCheck_3177_; 
lean_inc(v_toFun_2866_);
lean_inc_ref(v_e_2853_);
v_min_2955_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2955_, 0, v___f_2885_);
lean_closure_set(v_min_2955_, 1, v_e_2853_);
lean_closure_set(v_min_2955_, 2, v_inf_2951_);
lean_closure_set(v_min_2955_, 3, v_toFun_2866_);
lean_inc_ref(v_min_2955_);
v_semilatticeInf_2956_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2955_, v_le_2887_, v_lt_2888_);
v_toPartialOrder_2957_ = lean_ctor_get(v_semilatticeInf_2956_, 0);
v_isSharedCheck_3177_ = !lean_is_exclusive(v_semilatticeInf_2956_);
if (v_isSharedCheck_3177_ == 0)
{
lean_object* v_unused_3178_; 
v_unused_3178_ = lean_ctor_get(v_semilatticeInf_2956_, 1);
lean_dec(v_unused_3178_);
v___x_2959_ = v_semilatticeInf_2956_;
v_isShared_2960_ = v_isSharedCheck_3177_;
goto v_resetjp_2958_;
}
else
{
lean_inc(v_toPartialOrder_2957_);
lean_dec(v_semilatticeInf_2956_);
v___x_2959_ = lean_box(0);
v_isShared_2960_ = v_isSharedCheck_3177_;
goto v_resetjp_2958_;
}
v_resetjp_2958_:
{
lean_object* v_toLE_2961_; lean_object* v_toLT_2962_; lean_object* v___x_2964_; uint8_t v_isShared_2965_; uint8_t v_isSharedCheck_3176_; 
v_toLE_2961_ = lean_ctor_get(v_toPartialOrder_2957_, 0);
v_toLT_2962_ = lean_ctor_get(v_toPartialOrder_2957_, 1);
v_isSharedCheck_3176_ = !lean_is_exclusive(v_toPartialOrder_2957_);
if (v_isSharedCheck_3176_ == 0)
{
v___x_2964_ = v_toPartialOrder_2957_;
v_isShared_2965_ = v_isSharedCheck_3176_;
goto v_resetjp_2963_;
}
else
{
lean_inc(v_toLT_2962_);
lean_inc(v_toLE_2961_);
lean_dec(v_toPartialOrder_2957_);
v___x_2964_ = lean_box(0);
v_isShared_2965_ = v_isSharedCheck_3176_;
goto v_resetjp_2963_;
}
v_resetjp_2963_:
{
lean_object* v___f_2966_; lean_object* v___f_2967_; lean_object* v___x_2969_; 
v___f_2966_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_2966_, 0, v_min_2955_);
lean_inc(v_toFun_2866_);
lean_inc_ref(v_e_2853_);
v___f_2967_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_2967_, 0, v_toSemilatticeSup_2950_);
lean_closure_set(v___f_2967_, 1, v___f_2885_);
lean_closure_set(v___f_2967_, 2, v_e_2853_);
lean_closure_set(v___f_2967_, 3, v_toFun_2866_);
if (v_isShared_2965_ == 0)
{
v___x_2969_ = v___x_2964_;
goto v_reusejp_2968_;
}
else
{
lean_object* v_reuseFailAlloc_3175_; 
v_reuseFailAlloc_3175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3175_, 0, v_toLE_2961_);
lean_ctor_set(v_reuseFailAlloc_3175_, 1, v_toLT_2962_);
v___x_2969_ = v_reuseFailAlloc_3175_;
goto v_reusejp_2968_;
}
v_reusejp_2968_:
{
lean_object* v___x_2971_; 
lean_inc_ref(v___f_2967_);
if (v_isShared_2960_ == 0)
{
lean_ctor_set(v___x_2959_, 1, v___f_2967_);
lean_ctor_set(v___x_2959_, 0, v___x_2969_);
v___x_2971_ = v___x_2959_;
goto v_reusejp_2970_;
}
else
{
lean_object* v_reuseFailAlloc_3174_; 
v_reuseFailAlloc_3174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3174_, 0, v___x_2969_);
lean_ctor_set(v_reuseFailAlloc_3174_, 1, v___f_2967_);
v___x_2971_ = v_reuseFailAlloc_3174_;
goto v_reusejp_2970_;
}
v_reusejp_2970_:
{
lean_object* v_lattice_2973_; 
lean_inc_ref(v___f_2966_);
if (v_isShared_2954_ == 0)
{
lean_ctor_set(v___x_2953_, 1, v___f_2966_);
lean_ctor_set(v___x_2953_, 0, v___x_2971_);
v_lattice_2973_ = v___x_2953_;
goto v_reusejp_2972_;
}
else
{
lean_object* v_reuseFailAlloc_3173_; 
v_reuseFailAlloc_3173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3173_, 0, v___x_2971_);
lean_ctor_set(v_reuseFailAlloc_3173_, 1, v___f_2966_);
v_lattice_2973_ = v_reuseFailAlloc_3173_;
goto v_reusejp_2972_;
}
v_reusejp_2972_:
{
lean_object* v___x_2974_; lean_object* v_toPartialOrder_2975_; lean_object* v___x_2977_; uint8_t v_isShared_2978_; uint8_t v_isSharedCheck_3171_; 
v___x_2974_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2973_);
v_toPartialOrder_2975_ = lean_ctor_get(v___x_2974_, 0);
v_isSharedCheck_3171_ = !lean_is_exclusive(v___x_2974_);
if (v_isSharedCheck_3171_ == 0)
{
lean_object* v_unused_3172_; 
v_unused_3172_ = lean_ctor_get(v___x_2974_, 1);
lean_dec(v_unused_3172_);
v___x_2977_ = v___x_2974_;
v_isShared_2978_ = v_isSharedCheck_3171_;
goto v_resetjp_2976_;
}
else
{
lean_inc(v_toPartialOrder_2975_);
lean_dec(v___x_2974_);
v___x_2977_ = lean_box(0);
v_isShared_2978_ = v_isSharedCheck_3171_;
goto v_resetjp_2976_;
}
v_resetjp_2976_:
{
lean_object* v_toLE_2979_; lean_object* v_toLT_2980_; lean_object* v___x_2982_; uint8_t v_isShared_2983_; uint8_t v_isSharedCheck_3170_; 
v_toLE_2979_ = lean_ctor_get(v_toPartialOrder_2975_, 0);
v_toLT_2980_ = lean_ctor_get(v_toPartialOrder_2975_, 1);
v_isSharedCheck_3170_ = !lean_is_exclusive(v_toPartialOrder_2975_);
if (v_isSharedCheck_3170_ == 0)
{
v___x_2982_ = v_toPartialOrder_2975_;
v_isShared_2983_ = v_isSharedCheck_3170_;
goto v_resetjp_2981_;
}
else
{
lean_inc(v_toLT_2980_);
lean_inc(v_toLE_2979_);
lean_dec(v_toPartialOrder_2975_);
v___x_2982_ = lean_box(0);
v_isShared_2983_ = v_isSharedCheck_3170_;
goto v_resetjp_2981_;
}
v_resetjp_2981_:
{
lean_object* v_bot_2984_; lean_object* v___f_2985_; lean_object* v___x_2987_; 
lean_inc(v_toFun_2866_);
v_bot_2984_ = lean_apply_1(v_toFun_2866_, v_toOrderBot_2934_);
v___f_2985_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_2985_, 0, v___f_2967_);
if (v_isShared_2983_ == 0)
{
v___x_2987_ = v___x_2982_;
goto v_reusejp_2986_;
}
else
{
lean_object* v_reuseFailAlloc_3169_; 
v_reuseFailAlloc_3169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3169_, 0, v_toLE_2979_);
lean_ctor_set(v_reuseFailAlloc_3169_, 1, v_toLT_2980_);
v___x_2987_ = v_reuseFailAlloc_3169_;
goto v_reusejp_2986_;
}
v_reusejp_2986_:
{
lean_object* v___x_2989_; 
lean_inc_ref(v___f_2985_);
if (v_isShared_2978_ == 0)
{
lean_ctor_set(v___x_2977_, 1, v___f_2985_);
lean_ctor_set(v___x_2977_, 0, v___x_2987_);
v___x_2989_ = v___x_2977_;
goto v_reusejp_2988_;
}
else
{
lean_object* v_reuseFailAlloc_3168_; 
v_reuseFailAlloc_3168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3168_, 0, v___x_2987_);
lean_ctor_set(v_reuseFailAlloc_3168_, 1, v___f_2985_);
v___x_2989_ = v_reuseFailAlloc_3168_;
goto v_reusejp_2988_;
}
v_reusejp_2988_:
{
lean_object* v___x_2991_; 
lean_inc_ref(v___f_2966_);
lean_inc_ref(v___x_2989_);
if (v_isShared_2874_ == 0)
{
lean_ctor_set(v___x_2873_, 1, v___f_2966_);
lean_ctor_set(v___x_2873_, 0, v___x_2989_);
v___x_2991_ = v___x_2873_;
goto v_reusejp_2990_;
}
else
{
lean_object* v_reuseFailAlloc_3167_; 
v_reuseFailAlloc_3167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3167_, 0, v___x_2989_);
lean_ctor_set(v_reuseFailAlloc_3167_, 1, v___f_2966_);
v___x_2991_ = v_reuseFailAlloc_3167_;
goto v_reusejp_2990_;
}
v_reusejp_2990_:
{
lean_object* v___x_2992_; lean_object* v_toPartialOrder_2993_; lean_object* v_toLE_2994_; lean_object* v_toLT_2995_; lean_object* v___x_2997_; uint8_t v_isShared_2998_; uint8_t v_isSharedCheck_3166_; 
v___x_2992_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2991_);
v_toPartialOrder_2993_ = lean_ctor_get(v___x_2992_, 0);
lean_inc_ref(v_toPartialOrder_2993_);
v_toLE_2994_ = lean_ctor_get(v_toPartialOrder_2993_, 0);
v_toLT_2995_ = lean_ctor_get(v_toPartialOrder_2993_, 1);
v_isSharedCheck_3166_ = !lean_is_exclusive(v_toPartialOrder_2993_);
if (v_isSharedCheck_3166_ == 0)
{
v___x_2997_ = v_toPartialOrder_2993_;
v_isShared_2998_ = v_isSharedCheck_3166_;
goto v_resetjp_2996_;
}
else
{
lean_inc(v_toLT_2995_);
lean_inc(v_toLE_2994_);
lean_dec(v_toPartialOrder_2993_);
v___x_2997_ = lean_box(0);
v_isShared_2998_ = v_isSharedCheck_3166_;
goto v_resetjp_2996_;
}
v_resetjp_2996_:
{
lean_object* v_bot_2999_; lean_object* v_hnot_3000_; lean_object* v_sdiff_3001_; lean_object* v_top_3002_; lean_object* v___f_3003_; lean_object* v___f_3004_; lean_object* v_coheytingAlgebra_3005_; lean_object* v_toGeneralizedCoheytingAlgebra_3006_; lean_object* v_toLattice_3007_; lean_object* v_toOrderTop_3008_; lean_object* v_toHNot_3009_; lean_object* v_toSDiff_3010_; lean_object* v_toSemilatticeSup_3011_; lean_object* v___x_3012_; lean_object* v_toPartialOrder_3013_; lean_object* v_toLE_3014_; lean_object* v_toLT_3015_; lean_object* v___x_3017_; uint8_t v_isShared_3018_; uint8_t v_isSharedCheck_3165_; 
lean_inc_n(v_toFun_2866_, 4);
v_bot_2999_ = lean_apply_1(v_toFun_2866_, v_toOrderBot_2948_);
lean_inc_ref_n(v_e_2853_, 2);
v_hnot_3000_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__17), 4, 3);
lean_closure_set(v_hnot_3000_, 0, v_e_2853_);
lean_closure_set(v_hnot_3000_, 1, v_toHNot_2947_);
lean_closure_set(v_hnot_3000_, 2, v_toFun_2866_);
v_sdiff_3001_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_3001_, 0, v___f_2885_);
lean_closure_set(v_sdiff_3001_, 1, v_e_2853_);
lean_closure_set(v_sdiff_3001_, 2, v_toSDiff_2949_);
lean_closure_set(v_sdiff_3001_, 3, v_toFun_2866_);
v_top_3002_ = lean_apply_1(v_toFun_2866_, v_toOrderTop_2946_);
v___f_3003_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3003_, 0, v___x_2992_);
v___f_3004_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3004_, 0, v___x_2989_);
v_coheytingAlgebra_3005_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_3003_, v___f_3004_, v_toLE_2994_, v_toLT_2995_, v_bot_2999_, v_top_3002_, v_hnot_3000_, v_sdiff_3001_);
v_toGeneralizedCoheytingAlgebra_3006_ = lean_ctor_get(v_coheytingAlgebra_3005_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3006_);
v_toLattice_3007_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3006_, 0);
lean_inc_ref(v_toLattice_3007_);
v_toOrderTop_3008_ = lean_ctor_get(v_coheytingAlgebra_3005_, 1);
lean_inc(v_toOrderTop_3008_);
v_toHNot_3009_ = lean_ctor_get(v_coheytingAlgebra_3005_, 2);
lean_inc(v_toHNot_3009_);
lean_dec_ref(v_coheytingAlgebra_3005_);
v_toSDiff_3010_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3006_, 2);
lean_inc(v_toSDiff_3010_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_3006_);
v_toSemilatticeSup_3011_ = lean_ctor_get(v_toLattice_3007_, 0);
lean_inc_ref(v_toSemilatticeSup_3011_);
v___x_3012_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_3007_);
v_toPartialOrder_3013_ = lean_ctor_get(v___x_3012_, 0);
lean_inc_ref(v_toPartialOrder_3013_);
v_toLE_3014_ = lean_ctor_get(v_toPartialOrder_3013_, 0);
v_toLT_3015_ = lean_ctor_get(v_toPartialOrder_3013_, 1);
v_isSharedCheck_3165_ = !lean_is_exclusive(v_toPartialOrder_3013_);
if (v_isSharedCheck_3165_ == 0)
{
v___x_3017_ = v_toPartialOrder_3013_;
v_isShared_3018_ = v_isSharedCheck_3165_;
goto v_resetjp_3016_;
}
else
{
lean_inc(v_toLT_3015_);
lean_inc(v_toLE_3014_);
lean_dec(v_toPartialOrder_3013_);
v___x_3017_ = lean_box(0);
v_isShared_3018_ = v_isSharedCheck_3165_;
goto v_resetjp_3016_;
}
v_resetjp_3016_:
{
lean_object* v_compl_3019_; lean_object* v_himp_3020_; lean_object* v___f_3021_; lean_object* v___f_3022_; lean_object* v___x_3024_; 
lean_inc(v_toFun_2866_);
lean_inc_ref(v_e_2853_);
v_compl_3019_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_3019_, 0, v_e_2853_);
lean_closure_set(v_compl_3019_, 1, v_toCompl_2935_);
lean_closure_set(v_compl_3019_, 2, v_toFun_2866_);
v_himp_3020_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_3020_, 0, v___f_2885_);
lean_closure_set(v_himp_3020_, 1, v_e_2853_);
lean_closure_set(v_himp_3020_, 2, v_toHImp_2939_);
lean_closure_set(v_himp_3020_, 3, v_toFun_2866_);
v___f_3021_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3021_, 0, v_toSemilatticeSup_3011_);
v___f_3022_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3022_, 0, v___x_3012_);
if (v_isShared_3018_ == 0)
{
v___x_3024_ = v___x_3017_;
goto v_reusejp_3023_;
}
else
{
lean_object* v_reuseFailAlloc_3164_; 
v_reuseFailAlloc_3164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3164_, 0, v_toLE_3014_);
lean_ctor_set(v_reuseFailAlloc_3164_, 1, v_toLT_3015_);
v___x_3024_ = v_reuseFailAlloc_3164_;
goto v_reusejp_3023_;
}
v_reusejp_3023_:
{
lean_object* v___x_3026_; 
if (v_isShared_2998_ == 0)
{
lean_ctor_set(v___x_2997_, 1, v___f_2985_);
lean_ctor_set(v___x_2997_, 0, v___x_3024_);
v___x_3026_ = v___x_2997_;
goto v_reusejp_3025_;
}
else
{
lean_object* v_reuseFailAlloc_3163_; 
v_reuseFailAlloc_3163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3163_, 0, v___x_3024_);
lean_ctor_set(v_reuseFailAlloc_3163_, 1, v___f_2985_);
v___x_3026_ = v_reuseFailAlloc_3163_;
goto v_reusejp_3025_;
}
v_reusejp_3025_:
{
lean_object* v___x_3028_; 
if (v_isShared_2869_ == 0)
{
lean_ctor_set(v___x_2868_, 1, v___f_2966_);
lean_ctor_set(v___x_2868_, 0, v___x_3026_);
v___x_3028_ = v___x_2868_;
goto v_reusejp_3027_;
}
else
{
lean_object* v_reuseFailAlloc_3162_; 
v_reuseFailAlloc_3162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3162_, 0, v___x_3026_);
lean_ctor_set(v_reuseFailAlloc_3162_, 1, v___f_2966_);
v___x_3028_ = v_reuseFailAlloc_3162_;
goto v_reusejp_3027_;
}
v_reusejp_3027_:
{
lean_object* v___x_3030_; 
lean_inc_ref(v_himp_3020_);
lean_inc(v_toOrderTop_3008_);
if (v_isShared_2942_ == 0)
{
lean_ctor_set(v___x_2941_, 2, v_himp_3020_);
lean_ctor_set(v___x_2941_, 1, v_toOrderTop_3008_);
lean_ctor_set(v___x_2941_, 0, v___x_3028_);
v___x_3030_ = v___x_2941_;
goto v_reusejp_3029_;
}
else
{
lean_object* v_reuseFailAlloc_3161_; 
v_reuseFailAlloc_3161_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3161_, 0, v___x_3028_);
lean_ctor_set(v_reuseFailAlloc_3161_, 1, v_toOrderTop_3008_);
lean_ctor_set(v_reuseFailAlloc_3161_, 2, v_himp_3020_);
v___x_3030_ = v_reuseFailAlloc_3161_;
goto v_reusejp_3029_;
}
v_reusejp_3029_:
{
lean_object* v___x_3032_; 
lean_inc_ref(v_compl_3019_);
lean_inc(v_bot_2984_);
if (v_isShared_2938_ == 0)
{
lean_ctor_set(v___x_2937_, 2, v_compl_3019_);
lean_ctor_set(v___x_2937_, 1, v_bot_2984_);
lean_ctor_set(v___x_2937_, 0, v___x_3030_);
v___x_3032_ = v___x_2937_;
goto v_reusejp_3031_;
}
else
{
lean_object* v_reuseFailAlloc_3160_; 
v_reuseFailAlloc_3160_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3160_, 0, v___x_3030_);
lean_ctor_set(v_reuseFailAlloc_3160_, 1, v_bot_2984_);
lean_ctor_set(v_reuseFailAlloc_3160_, 2, v_compl_3019_);
v___x_3032_ = v_reuseFailAlloc_3160_;
goto v_reusejp_3031_;
}
v_reusejp_3031_:
{
lean_object* v___x_3033_; lean_object* v_toGeneralizedCoheytingAlgebra_3034_; lean_object* v_toHNot_3035_; lean_object* v_toSDiff_3036_; lean_object* v___x_3038_; uint8_t v_isShared_3039_; uint8_t v_isSharedCheck_3157_; 
lean_inc(v_bot_2984_);
v___x_3033_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_3022_, v___f_3021_, v_toLE_3014_, v_toLT_3015_, v_bot_2984_, v_toOrderTop_3008_, v_toHNot_3009_, v_toSDiff_3010_);
v_toGeneralizedCoheytingAlgebra_3034_ = lean_ctor_get(v___x_3033_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3034_);
v_toHNot_3035_ = lean_ctor_get(v___x_3033_, 2);
lean_inc(v_toHNot_3035_);
lean_dec_ref(v___x_3033_);
v_toSDiff_3036_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3034_, 2);
v_isSharedCheck_3157_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_3034_);
if (v_isSharedCheck_3157_ == 0)
{
lean_object* v_unused_3158_; lean_object* v_unused_3159_; 
v_unused_3158_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3034_, 1);
lean_dec(v_unused_3158_);
v_unused_3159_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3034_, 0);
lean_dec(v_unused_3159_);
v___x_3038_ = v_toGeneralizedCoheytingAlgebra_3034_;
v_isShared_3039_ = v_isSharedCheck_3157_;
goto v_resetjp_3037_;
}
else
{
lean_inc(v_toSDiff_3036_);
lean_dec(v_toGeneralizedCoheytingAlgebra_3034_);
v___x_3038_ = lean_box(0);
v_isShared_3039_ = v_isSharedCheck_3157_;
goto v_resetjp_3037_;
}
v_resetjp_3037_:
{
lean_object* v_biheytingAlgebra_3041_; 
if (v_isShared_3039_ == 0)
{
lean_ctor_set(v___x_3038_, 2, v_toHNot_3035_);
lean_ctor_set(v___x_3038_, 1, v_toSDiff_3036_);
lean_ctor_set(v___x_3038_, 0, v___x_3032_);
v_biheytingAlgebra_3041_ = v___x_3038_;
goto v_reusejp_3040_;
}
else
{
lean_object* v_reuseFailAlloc_3156_; 
v_reuseFailAlloc_3156_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3156_, 0, v___x_3032_);
lean_ctor_set(v_reuseFailAlloc_3156_, 1, v_toSDiff_3036_);
lean_ctor_set(v_reuseFailAlloc_3156_, 2, v_toHNot_3035_);
v_biheytingAlgebra_3041_ = v_reuseFailAlloc_3156_;
goto v_reusejp_3040_;
}
v_reusejp_3040_:
{
lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v_toPartialOrder_3044_; lean_object* v_toInfSet_3045_; lean_object* v___x_3047_; uint8_t v_isShared_3048_; uint8_t v_isSharedCheck_3155_; 
v___x_3042_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2927_);
lean_inc_ref(v_completeLattice_2930_);
v___x_3043_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_2930_);
v_toPartialOrder_3044_ = lean_ctor_get(v___x_3043_, 0);
v_toInfSet_3045_ = lean_ctor_get(v___x_3043_, 1);
v_isSharedCheck_3155_ = !lean_is_exclusive(v___x_3043_);
if (v_isSharedCheck_3155_ == 0)
{
v___x_3047_ = v___x_3043_;
v_isShared_3048_ = v_isSharedCheck_3155_;
goto v_resetjp_3046_;
}
else
{
lean_inc(v_toInfSet_3045_);
lean_inc(v_toPartialOrder_3044_);
lean_dec(v___x_3043_);
v___x_3047_ = lean_box(0);
v_isShared_3048_ = v_isSharedCheck_3155_;
goto v_resetjp_3046_;
}
v_resetjp_3046_:
{
lean_object* v_toLE_3049_; lean_object* v_toLT_3050_; lean_object* v___x_3052_; uint8_t v_isShared_3053_; uint8_t v_isSharedCheck_3154_; 
v_toLE_3049_ = lean_ctor_get(v_toPartialOrder_3044_, 0);
v_toLT_3050_ = lean_ctor_get(v_toPartialOrder_3044_, 1);
v_isSharedCheck_3154_ = !lean_is_exclusive(v_toPartialOrder_3044_);
if (v_isSharedCheck_3154_ == 0)
{
v___x_3052_ = v_toPartialOrder_3044_;
v_isShared_3053_ = v_isSharedCheck_3154_;
goto v_resetjp_3051_;
}
else
{
lean_inc(v_toLT_3050_);
lean_inc(v_toLE_3049_);
lean_dec(v_toPartialOrder_3044_);
v___x_3052_ = lean_box(0);
v_isShared_3053_ = v_isSharedCheck_3154_;
goto v_resetjp_3051_;
}
v_resetjp_3051_:
{
lean_object* v___x_3054_; lean_object* v_toSupSet_3055_; lean_object* v___x_3057_; uint8_t v_isShared_3058_; uint8_t v_isSharedCheck_3152_; 
v___x_3054_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_2930_);
v_toSupSet_3055_ = lean_ctor_get(v___x_3054_, 1);
v_isSharedCheck_3152_ = !lean_is_exclusive(v___x_3054_);
if (v_isSharedCheck_3152_ == 0)
{
lean_object* v_unused_3153_; 
v_unused_3153_ = lean_ctor_get(v___x_3054_, 0);
lean_dec(v_unused_3153_);
v___x_3057_ = v___x_3054_;
v_isShared_3058_ = v_isSharedCheck_3152_;
goto v_resetjp_3056_;
}
else
{
lean_inc(v_toSupSet_3055_);
lean_dec(v___x_3054_);
v___x_3057_ = lean_box(0);
v_isShared_3058_ = v_isSharedCheck_3152_;
goto v_resetjp_3056_;
}
v_resetjp_3056_:
{
lean_object* v___x_3059_; lean_object* v_toGeneralizedCoheytingAlgebra_3060_; lean_object* v_toOrderTop_3061_; lean_object* v_toHNot_3062_; lean_object* v_toSDiff_3063_; lean_object* v___x_3065_; uint8_t v_isShared_3066_; uint8_t v_isSharedCheck_3149_; 
v___x_3059_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_biheytingAlgebra_3041_);
v_toGeneralizedCoheytingAlgebra_3060_ = lean_ctor_get(v___x_3059_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3060_);
v_toOrderTop_3061_ = lean_ctor_get(v___x_3059_, 1);
lean_inc(v_toOrderTop_3061_);
v_toHNot_3062_ = lean_ctor_get(v___x_3059_, 2);
lean_inc(v_toHNot_3062_);
lean_dec_ref(v___x_3059_);
v_toSDiff_3063_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3060_, 2);
v_isSharedCheck_3149_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_3060_);
if (v_isSharedCheck_3149_ == 0)
{
lean_object* v_unused_3150_; lean_object* v_unused_3151_; 
v_unused_3150_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3060_, 1);
lean_dec(v_unused_3150_);
v_unused_3151_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3060_, 0);
lean_dec(v_unused_3151_);
v___x_3065_ = v_toGeneralizedCoheytingAlgebra_3060_;
v_isShared_3066_ = v_isSharedCheck_3149_;
goto v_resetjp_3064_;
}
else
{
lean_inc(v_toSDiff_3063_);
lean_dec(v_toGeneralizedCoheytingAlgebra_3060_);
v___x_3065_ = lean_box(0);
v_isShared_3066_ = v_isSharedCheck_3149_;
goto v_resetjp_3064_;
}
v_resetjp_3064_:
{
lean_object* v___f_3067_; lean_object* v___f_3068_; lean_object* v___x_3070_; 
v___f_3067_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3067_, 0, v___x_2925_);
v___f_3068_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3068_, 0, v___x_3042_);
if (v_isShared_3053_ == 0)
{
v___x_3070_ = v___x_3052_;
goto v_reusejp_3069_;
}
else
{
lean_object* v_reuseFailAlloc_3148_; 
v_reuseFailAlloc_3148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3148_, 0, v_toLE_3049_);
lean_ctor_set(v_reuseFailAlloc_3148_, 1, v_toLT_3050_);
v___x_3070_ = v_reuseFailAlloc_3148_;
goto v_reusejp_3069_;
}
v_reusejp_3069_:
{
lean_object* v___x_3072_; 
lean_inc_ref(v___f_2921_);
if (v_isShared_3058_ == 0)
{
lean_ctor_set(v___x_3057_, 1, v___f_2921_);
lean_ctor_set(v___x_3057_, 0, v___x_3070_);
v___x_3072_ = v___x_3057_;
goto v_reusejp_3071_;
}
else
{
lean_object* v_reuseFailAlloc_3147_; 
v_reuseFailAlloc_3147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3147_, 0, v___x_3070_);
lean_ctor_set(v_reuseFailAlloc_3147_, 1, v___f_2921_);
v___x_3072_ = v_reuseFailAlloc_3147_;
goto v_reusejp_3071_;
}
v_reusejp_3071_:
{
lean_object* v___x_3074_; 
lean_inc_ref(v___f_2899_);
if (v_isShared_3048_ == 0)
{
lean_ctor_set(v___x_3047_, 1, v___f_2899_);
lean_ctor_set(v___x_3047_, 0, v___x_3072_);
v___x_3074_ = v___x_3047_;
goto v_reusejp_3073_;
}
else
{
lean_object* v_reuseFailAlloc_3146_; 
v_reuseFailAlloc_3146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3146_, 0, v___x_3072_);
lean_ctor_set(v_reuseFailAlloc_3146_, 1, v___f_2899_);
v___x_3074_ = v_reuseFailAlloc_3146_;
goto v_reusejp_3073_;
}
v_reusejp_3073_:
{
lean_object* v___x_3075_; lean_object* v___x_3076_; lean_object* v___x_3078_; 
lean_inc(v_bot_2984_);
lean_inc(v_toOrderTop_3061_);
v___x_3075_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3075_, 0, v_toOrderTop_3061_);
lean_ctor_set(v___x_3075_, 1, v_bot_2984_);
v___x_3076_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3076_, 0, v___x_3074_);
lean_ctor_set(v___x_3076_, 1, v_toSupSet_3055_);
lean_ctor_set(v___x_3076_, 2, v_toInfSet_3045_);
lean_ctor_set(v___x_3076_, 3, v___x_3075_);
if (v_isShared_3066_ == 0)
{
lean_ctor_set(v___x_3065_, 2, v_compl_3019_);
lean_ctor_set(v___x_3065_, 1, v_himp_3020_);
lean_ctor_set(v___x_3065_, 0, v___x_3076_);
v___x_3078_ = v___x_3065_;
goto v_reusejp_3077_;
}
else
{
lean_object* v_reuseFailAlloc_3145_; 
v_reuseFailAlloc_3145_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3145_, 0, v___x_3076_);
lean_ctor_set(v_reuseFailAlloc_3145_, 1, v_himp_3020_);
lean_ctor_set(v_reuseFailAlloc_3145_, 2, v_compl_3019_);
v___x_3078_ = v_reuseFailAlloc_3145_;
goto v_reusejp_3077_;
}
v_reusejp_3077_:
{
lean_object* v___x_3079_; lean_object* v_toGeneralizedCoheytingAlgebra_3080_; lean_object* v_toHNot_3081_; lean_object* v_toSDiff_3082_; lean_object* v___x_3084_; uint8_t v_isShared_3085_; uint8_t v_isSharedCheck_3142_; 
v___x_3079_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_3068_, v___f_3067_, v_toLE_3049_, v_toLT_3050_, v_bot_2984_, v_toOrderTop_3061_, v_toHNot_3062_, v_toSDiff_3063_);
v_toGeneralizedCoheytingAlgebra_3080_ = lean_ctor_get(v___x_3079_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3080_);
v_toHNot_3081_ = lean_ctor_get(v___x_3079_, 2);
lean_inc(v_toHNot_3081_);
lean_dec_ref(v___x_3079_);
v_toSDiff_3082_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3080_, 2);
v_isSharedCheck_3142_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_3080_);
if (v_isSharedCheck_3142_ == 0)
{
lean_object* v_unused_3143_; lean_object* v_unused_3144_; 
v_unused_3143_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3080_, 1);
lean_dec(v_unused_3143_);
v_unused_3144_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3080_, 0);
lean_dec(v_unused_3144_);
v___x_3084_ = v_toGeneralizedCoheytingAlgebra_3080_;
v_isShared_3085_ = v_isSharedCheck_3142_;
goto v_resetjp_3083_;
}
else
{
lean_inc(v_toSDiff_3082_);
lean_dec(v_toGeneralizedCoheytingAlgebra_3080_);
v___x_3084_ = lean_box(0);
v_isShared_3085_ = v_isSharedCheck_3142_;
goto v_resetjp_3083_;
}
v_resetjp_3083_:
{
lean_object* v_completeDistribLattice_3087_; 
lean_inc_ref(v___x_3078_);
if (v_isShared_3085_ == 0)
{
lean_ctor_set(v___x_3084_, 2, v_toHNot_3081_);
lean_ctor_set(v___x_3084_, 1, v_toSDiff_3082_);
lean_ctor_set(v___x_3084_, 0, v___x_3078_);
v_completeDistribLattice_3087_ = v___x_3084_;
goto v_reusejp_3086_;
}
else
{
lean_object* v_reuseFailAlloc_3141_; 
v_reuseFailAlloc_3141_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3141_, 0, v___x_3078_);
lean_ctor_set(v_reuseFailAlloc_3141_, 1, v_toSDiff_3082_);
lean_ctor_set(v_reuseFailAlloc_3141_, 2, v_toHNot_3081_);
v_completeDistribLattice_3087_ = v_reuseFailAlloc_3141_;
goto v_reusejp_3086_;
}
v_reusejp_3086_:
{
lean_object* v___x_3088_; lean_object* v_toCompleteLattice_3089_; lean_object* v_toLattice_3090_; lean_object* v_toSemilatticeSup_3091_; lean_object* v___x_3092_; lean_object* v___x_3093_; lean_object* v_toPartialOrder_3094_; lean_object* v_toInfSet_3095_; lean_object* v___x_3097_; uint8_t v_isShared_3098_; uint8_t v_isSharedCheck_3140_; 
v___x_3088_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v_completeDistribLattice_3087_);
v_toCompleteLattice_3089_ = lean_ctor_get(v___x_3088_, 0);
lean_inc_ref_n(v_toCompleteLattice_3089_, 2);
v_toLattice_3090_ = lean_ctor_get(v_toCompleteLattice_3089_, 0);
v_toSemilatticeSup_3091_ = lean_ctor_get(v_toLattice_3090_, 0);
lean_inc_ref(v_toSemilatticeSup_3091_);
lean_inc_ref(v_toLattice_3090_);
v___x_3092_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_3090_);
v___x_3093_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_3089_);
v_toPartialOrder_3094_ = lean_ctor_get(v___x_3093_, 0);
v_toInfSet_3095_ = lean_ctor_get(v___x_3093_, 1);
v_isSharedCheck_3140_ = !lean_is_exclusive(v___x_3093_);
if (v_isSharedCheck_3140_ == 0)
{
v___x_3097_ = v___x_3093_;
v_isShared_3098_ = v_isSharedCheck_3140_;
goto v_resetjp_3096_;
}
else
{
lean_inc(v_toInfSet_3095_);
lean_inc(v_toPartialOrder_3094_);
lean_dec(v___x_3093_);
v___x_3097_ = lean_box(0);
v_isShared_3098_ = v_isSharedCheck_3140_;
goto v_resetjp_3096_;
}
v_resetjp_3096_:
{
lean_object* v_toLE_3099_; lean_object* v_toLT_3100_; lean_object* v___x_3102_; uint8_t v_isShared_3103_; uint8_t v_isSharedCheck_3139_; 
v_toLE_3099_ = lean_ctor_get(v_toPartialOrder_3094_, 0);
v_toLT_3100_ = lean_ctor_get(v_toPartialOrder_3094_, 1);
v_isSharedCheck_3139_ = !lean_is_exclusive(v_toPartialOrder_3094_);
if (v_isSharedCheck_3139_ == 0)
{
v___x_3102_ = v_toPartialOrder_3094_;
v_isShared_3103_ = v_isSharedCheck_3139_;
goto v_resetjp_3101_;
}
else
{
lean_inc(v_toLT_3100_);
lean_inc(v_toLE_3099_);
lean_dec(v_toPartialOrder_3094_);
v___x_3102_ = lean_box(0);
v_isShared_3103_ = v_isSharedCheck_3139_;
goto v_resetjp_3101_;
}
v_resetjp_3101_:
{
lean_object* v___x_3104_; lean_object* v_toSupSet_3105_; lean_object* v___x_3107_; uint8_t v_isShared_3108_; uint8_t v_isSharedCheck_3137_; 
v___x_3104_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_3089_);
v_toSupSet_3105_ = lean_ctor_get(v___x_3104_, 1);
v_isSharedCheck_3137_ = !lean_is_exclusive(v___x_3104_);
if (v_isSharedCheck_3137_ == 0)
{
lean_object* v_unused_3138_; 
v_unused_3138_ = lean_ctor_get(v___x_3104_, 0);
lean_dec(v_unused_3138_);
v___x_3107_ = v___x_3104_;
v_isShared_3108_ = v_isSharedCheck_3137_;
goto v_resetjp_3106_;
}
else
{
lean_inc(v_toSupSet_3105_);
lean_dec(v___x_3104_);
v___x_3107_ = lean_box(0);
v_isShared_3108_ = v_isSharedCheck_3137_;
goto v_resetjp_3106_;
}
v_resetjp_3106_:
{
lean_object* v___x_3109_; lean_object* v_toGeneralizedCoheytingAlgebra_3110_; lean_object* v_toOrderTop_3111_; lean_object* v_toHNot_3112_; lean_object* v___x_3113_; lean_object* v_toGeneralizedHeytingAlgebra_3114_; lean_object* v_toOrderBot_3115_; lean_object* v_toCompl_3116_; lean_object* v_toHImp_3117_; lean_object* v_toSDiff_3118_; lean_object* v___f_3119_; lean_object* v___f_3120_; lean_object* v___x_3122_; 
v___x_3109_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v___x_3088_);
v_toGeneralizedCoheytingAlgebra_3110_ = lean_ctor_get(v___x_3109_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3110_);
v_toOrderTop_3111_ = lean_ctor_get(v___x_3109_, 1);
lean_inc(v_toOrderTop_3111_);
v_toHNot_3112_ = lean_ctor_get(v___x_3109_, 2);
lean_inc(v_toHNot_3112_);
lean_dec_ref(v___x_3109_);
v___x_3113_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v___x_3078_);
v_toGeneralizedHeytingAlgebra_3114_ = lean_ctor_get(v___x_3113_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_3114_);
v_toOrderBot_3115_ = lean_ctor_get(v___x_3113_, 1);
lean_inc(v_toOrderBot_3115_);
v_toCompl_3116_ = lean_ctor_get(v___x_3113_, 2);
lean_inc(v_toCompl_3116_);
lean_dec_ref(v___x_3113_);
v_toHImp_3117_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_3114_, 2);
lean_inc(v_toHImp_3117_);
lean_dec_ref(v_toGeneralizedHeytingAlgebra_3114_);
v_toSDiff_3118_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3110_, 2);
lean_inc(v_toSDiff_3118_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_3110_);
v___f_3119_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3119_, 0, v_toSemilatticeSup_3091_);
v___f_3120_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3120_, 0, v___x_3092_);
if (v_isShared_3103_ == 0)
{
v___x_3122_ = v___x_3102_;
goto v_reusejp_3121_;
}
else
{
lean_object* v_reuseFailAlloc_3136_; 
v_reuseFailAlloc_3136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3136_, 0, v_toLE_3099_);
lean_ctor_set(v_reuseFailAlloc_3136_, 1, v_toLT_3100_);
v___x_3122_ = v_reuseFailAlloc_3136_;
goto v_reusejp_3121_;
}
v_reusejp_3121_:
{
lean_object* v___x_3124_; 
if (v_isShared_3108_ == 0)
{
lean_ctor_set(v___x_3107_, 1, v___f_2921_);
lean_ctor_set(v___x_3107_, 0, v___x_3122_);
v___x_3124_ = v___x_3107_;
goto v_reusejp_3123_;
}
else
{
lean_object* v_reuseFailAlloc_3135_; 
v_reuseFailAlloc_3135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3135_, 0, v___x_3122_);
lean_ctor_set(v_reuseFailAlloc_3135_, 1, v___f_2921_);
v___x_3124_ = v_reuseFailAlloc_3135_;
goto v_reusejp_3123_;
}
v_reusejp_3123_:
{
lean_object* v___x_3126_; 
if (v_isShared_3098_ == 0)
{
lean_ctor_set(v___x_3097_, 1, v___f_2899_);
lean_ctor_set(v___x_3097_, 0, v___x_3124_);
v___x_3126_ = v___x_3097_;
goto v_reusejp_3125_;
}
else
{
lean_object* v_reuseFailAlloc_3134_; 
v_reuseFailAlloc_3134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3134_, 0, v___x_3124_);
lean_ctor_set(v_reuseFailAlloc_3134_, 1, v___f_2899_);
v___x_3126_ = v_reuseFailAlloc_3134_;
goto v_reusejp_3125_;
}
v_reusejp_3125_:
{
lean_object* v___x_3127_; lean_object* v___x_3128_; lean_object* v___x_3129_; lean_object* v_toGeneralizedCoheytingAlgebra_3130_; lean_object* v_toHNot_3131_; lean_object* v_toSDiff_3132_; lean_object* v___x_3133_; 
lean_inc(v_toOrderBot_3115_);
lean_inc(v_toOrderTop_3111_);
v___x_3127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3127_, 0, v_toOrderTop_3111_);
lean_ctor_set(v___x_3127_, 1, v_toOrderBot_3115_);
v___x_3128_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3128_, 0, v___x_3126_);
lean_ctor_set(v___x_3128_, 1, v_toSupSet_3105_);
lean_ctor_set(v___x_3128_, 2, v_toInfSet_3095_);
lean_ctor_set(v___x_3128_, 3, v___x_3127_);
v___x_3129_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_3120_, v___f_3119_, v_toLE_3099_, v_toLT_3100_, v_toOrderBot_3115_, v_toOrderTop_3111_, v_toHNot_3112_, v_toSDiff_3118_);
v_toGeneralizedCoheytingAlgebra_3130_ = lean_ctor_get(v___x_3129_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3130_);
v_toHNot_3131_ = lean_ctor_get(v___x_3129_, 2);
lean_inc(v_toHNot_3131_);
lean_dec_ref(v___x_3129_);
v_toSDiff_3132_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3130_, 2);
lean_inc(v_toSDiff_3132_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_3130_);
v___x_3133_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3133_, 0, v___x_3128_);
lean_ctor_set(v___x_3133_, 1, v_toHImp_3117_);
lean_ctor_set(v___x_3133_, 2, v_toCompl_3116_);
lean_ctor_set(v___x_3133_, 3, v_toSDiff_3132_);
lean_ctor_set(v___x_3133_, 4, v_toHNot_3131_);
return v___x_3133_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completelyDistribLattice___redArg___boxed(lean_object* v_e_3205_, lean_object* v_inst_3206_){
_start:
{
lean_object* v_res_3207_; 
v_res_3207_ = lp_mathlib_Equiv_completelyDistribLattice___redArg(v_e_3205_, v_inst_3206_);
lean_dec_ref(v_inst_3206_);
return v_res_3207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completelyDistribLattice(lean_object* v_00_u03b1_3208_, lean_object* v_00_u03b2_3209_, lean_object* v_e_3210_, lean_object* v_inst_3211_){
_start:
{
lean_object* v___x_3212_; lean_object* v___x_3213_; lean_object* v_toCompleteLattice_3214_; lean_object* v_toBoundedOrder_3215_; lean_object* v_toLattice_3216_; lean_object* v_toOrderTop_3217_; lean_object* v_toOrderBot_3218_; lean_object* v___x_3220_; uint8_t v_isShared_3221_; uint8_t v_isSharedCheck_3561_; 
v___x_3212_ = lp_mathlib_CompletelyDistribLattice_toCompleteDistribLattice___redArg(v_inst_3211_);
lean_inc_ref(v___x_3212_);
v___x_3213_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_3212_);
v_toCompleteLattice_3214_ = lean_ctor_get(v___x_3213_, 0);
lean_inc_ref(v_toCompleteLattice_3214_);
lean_dec_ref(v___x_3213_);
v_toBoundedOrder_3215_ = lean_ctor_get(v_toCompleteLattice_3214_, 3);
lean_inc_ref(v_toBoundedOrder_3215_);
v_toLattice_3216_ = lean_ctor_get(v_toCompleteLattice_3214_, 0);
lean_inc_ref(v_toLattice_3216_);
v_toOrderTop_3217_ = lean_ctor_get(v_toBoundedOrder_3215_, 0);
v_toOrderBot_3218_ = lean_ctor_get(v_toBoundedOrder_3215_, 1);
v_isSharedCheck_3561_ = !lean_is_exclusive(v_toBoundedOrder_3215_);
if (v_isSharedCheck_3561_ == 0)
{
v___x_3220_ = v_toBoundedOrder_3215_;
v_isShared_3221_ = v_isSharedCheck_3561_;
goto v_resetjp_3219_;
}
else
{
lean_inc(v_toOrderBot_3218_);
lean_inc(v_toOrderTop_3217_);
lean_dec(v_toBoundedOrder_3215_);
v___x_3220_ = lean_box(0);
v_isShared_3221_ = v_isSharedCheck_3561_;
goto v_resetjp_3219_;
}
v_resetjp_3219_:
{
lean_object* v___x_3222_; lean_object* v_toFun_3223_; lean_object* v___x_3225_; uint8_t v_isShared_3226_; uint8_t v_isSharedCheck_3559_; 
lean_inc_ref(v_e_3210_);
v___x_3222_ = lp_mathlib_Equiv_symm___redArg(v_e_3210_);
v_toFun_3223_ = lean_ctor_get(v___x_3222_, 0);
v_isSharedCheck_3559_ = !lean_is_exclusive(v___x_3222_);
if (v_isSharedCheck_3559_ == 0)
{
lean_object* v_unused_3560_; 
v_unused_3560_ = lean_ctor_get(v___x_3222_, 1);
lean_dec(v_unused_3560_);
v___x_3225_ = v___x_3222_;
v_isShared_3226_ = v_isSharedCheck_3559_;
goto v_resetjp_3224_;
}
else
{
lean_inc(v_toFun_3223_);
lean_dec(v___x_3222_);
v___x_3225_ = lean_box(0);
v_isShared_3226_ = v_isSharedCheck_3559_;
goto v_resetjp_3224_;
}
v_resetjp_3224_:
{
lean_object* v___x_3227_; lean_object* v_toSupSet_3228_; lean_object* v___x_3230_; uint8_t v_isShared_3231_; uint8_t v_isSharedCheck_3557_; 
lean_inc_ref(v_toCompleteLattice_3214_);
v___x_3227_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_3214_);
v_toSupSet_3228_ = lean_ctor_get(v___x_3227_, 1);
v_isSharedCheck_3557_ = !lean_is_exclusive(v___x_3227_);
if (v_isSharedCheck_3557_ == 0)
{
lean_object* v_unused_3558_; 
v_unused_3558_ = lean_ctor_get(v___x_3227_, 0);
lean_dec(v_unused_3558_);
v___x_3230_ = v___x_3227_;
v_isShared_3231_ = v_isSharedCheck_3557_;
goto v_resetjp_3229_;
}
else
{
lean_inc(v_toSupSet_3228_);
lean_dec(v___x_3227_);
v___x_3230_ = lean_box(0);
v_isShared_3231_ = v_isSharedCheck_3557_;
goto v_resetjp_3229_;
}
v_resetjp_3229_:
{
lean_object* v___x_3232_; lean_object* v_toInfSet_3233_; lean_object* v___x_3235_; uint8_t v_isShared_3236_; uint8_t v_isSharedCheck_3555_; 
v___x_3232_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_3214_);
v_toInfSet_3233_ = lean_ctor_get(v___x_3232_, 1);
v_isSharedCheck_3555_ = !lean_is_exclusive(v___x_3232_);
if (v_isSharedCheck_3555_ == 0)
{
lean_object* v_unused_3556_; 
v_unused_3556_ = lean_ctor_get(v___x_3232_, 0);
lean_dec(v_unused_3556_);
v___x_3235_ = v___x_3232_;
v_isShared_3236_ = v_isSharedCheck_3555_;
goto v_resetjp_3234_;
}
else
{
lean_inc(v_toInfSet_3233_);
lean_dec(v___x_3232_);
v___x_3235_ = lean_box(0);
v_isShared_3236_ = v_isSharedCheck_3555_;
goto v_resetjp_3234_;
}
v_resetjp_3234_:
{
lean_object* v_toSemilatticeSup_3237_; lean_object* v_inf_3238_; lean_object* v___x_3240_; uint8_t v_isShared_3241_; uint8_t v_isSharedCheck_3554_; 
v_toSemilatticeSup_3237_ = lean_ctor_get(v_toLattice_3216_, 0);
v_inf_3238_ = lean_ctor_get(v_toLattice_3216_, 1);
v_isSharedCheck_3554_ = !lean_is_exclusive(v_toLattice_3216_);
if (v_isSharedCheck_3554_ == 0)
{
v___x_3240_ = v_toLattice_3216_;
v_isShared_3241_ = v_isSharedCheck_3554_;
goto v_resetjp_3239_;
}
else
{
lean_inc(v_inf_3238_);
lean_inc(v_toSemilatticeSup_3237_);
lean_dec(v_toLattice_3216_);
v___x_3240_ = lean_box(0);
v_isShared_3241_ = v_isSharedCheck_3554_;
goto v_resetjp_3239_;
}
v_resetjp_3239_:
{
lean_object* v___f_3242_; lean_object* v_min_3243_; lean_object* v_le_3244_; lean_object* v_lt_3245_; lean_object* v_semilatticeInf_3246_; lean_object* v_toPartialOrder_3247_; lean_object* v___x_3249_; uint8_t v_isShared_3250_; uint8_t v_isSharedCheck_3552_; 
v___f_3242_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_3223_);
lean_inc_ref(v_e_3210_);
v_min_3243_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_3243_, 0, v___f_3242_);
lean_closure_set(v_min_3243_, 1, v_e_3210_);
lean_closure_set(v_min_3243_, 2, v_inf_3238_);
lean_closure_set(v_min_3243_, 3, v_toFun_3223_);
v_le_3244_ = lean_box(0);
v_lt_3245_ = lean_box(0);
lean_inc_ref(v_min_3243_);
v_semilatticeInf_3246_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_3243_, v_le_3244_, v_lt_3245_);
v_toPartialOrder_3247_ = lean_ctor_get(v_semilatticeInf_3246_, 0);
v_isSharedCheck_3552_ = !lean_is_exclusive(v_semilatticeInf_3246_);
if (v_isSharedCheck_3552_ == 0)
{
lean_object* v_unused_3553_; 
v_unused_3553_ = lean_ctor_get(v_semilatticeInf_3246_, 1);
lean_dec(v_unused_3553_);
v___x_3249_ = v_semilatticeInf_3246_;
v_isShared_3250_ = v_isSharedCheck_3552_;
goto v_resetjp_3248_;
}
else
{
lean_inc(v_toPartialOrder_3247_);
lean_dec(v_semilatticeInf_3246_);
v___x_3249_ = lean_box(0);
v_isShared_3250_ = v_isSharedCheck_3552_;
goto v_resetjp_3248_;
}
v_resetjp_3248_:
{
lean_object* v_toLE_3251_; lean_object* v_toLT_3252_; lean_object* v___x_3254_; uint8_t v_isShared_3255_; uint8_t v_isSharedCheck_3551_; 
v_toLE_3251_ = lean_ctor_get(v_toPartialOrder_3247_, 0);
v_toLT_3252_ = lean_ctor_get(v_toPartialOrder_3247_, 1);
v_isSharedCheck_3551_ = !lean_is_exclusive(v_toPartialOrder_3247_);
if (v_isSharedCheck_3551_ == 0)
{
v___x_3254_ = v_toPartialOrder_3247_;
v_isShared_3255_ = v_isSharedCheck_3551_;
goto v_resetjp_3253_;
}
else
{
lean_inc(v_toLT_3252_);
lean_inc(v_toLE_3251_);
lean_dec(v_toPartialOrder_3247_);
v___x_3254_ = lean_box(0);
v_isShared_3255_ = v_isSharedCheck_3551_;
goto v_resetjp_3253_;
}
v_resetjp_3253_:
{
lean_object* v___f_3256_; lean_object* v___f_3257_; lean_object* v___x_3259_; 
v___f_3256_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_3256_, 0, v_min_3243_);
lean_inc(v_toFun_3223_);
lean_inc_ref(v_e_3210_);
v___f_3257_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_3257_, 0, v_toSemilatticeSup_3237_);
lean_closure_set(v___f_3257_, 1, v___f_3242_);
lean_closure_set(v___f_3257_, 2, v_e_3210_);
lean_closure_set(v___f_3257_, 3, v_toFun_3223_);
if (v_isShared_3255_ == 0)
{
v___x_3259_ = v___x_3254_;
goto v_reusejp_3258_;
}
else
{
lean_object* v_reuseFailAlloc_3550_; 
v_reuseFailAlloc_3550_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3550_, 0, v_toLE_3251_);
lean_ctor_set(v_reuseFailAlloc_3550_, 1, v_toLT_3252_);
v___x_3259_ = v_reuseFailAlloc_3550_;
goto v_reusejp_3258_;
}
v_reusejp_3258_:
{
lean_object* v___x_3261_; 
lean_inc_ref(v___f_3257_);
if (v_isShared_3250_ == 0)
{
lean_ctor_set(v___x_3249_, 1, v___f_3257_);
lean_ctor_set(v___x_3249_, 0, v___x_3259_);
v___x_3261_ = v___x_3249_;
goto v_reusejp_3260_;
}
else
{
lean_object* v_reuseFailAlloc_3549_; 
v_reuseFailAlloc_3549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3549_, 0, v___x_3259_);
lean_ctor_set(v_reuseFailAlloc_3549_, 1, v___f_3257_);
v___x_3261_ = v_reuseFailAlloc_3549_;
goto v_reusejp_3260_;
}
v_reusejp_3260_:
{
lean_object* v_lattice_3263_; 
lean_inc_ref(v___f_3256_);
if (v_isShared_3241_ == 0)
{
lean_ctor_set(v___x_3240_, 1, v___f_3256_);
lean_ctor_set(v___x_3240_, 0, v___x_3261_);
v_lattice_3263_ = v___x_3240_;
goto v_reusejp_3262_;
}
else
{
lean_object* v_reuseFailAlloc_3548_; 
v_reuseFailAlloc_3548_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3548_, 0, v___x_3261_);
lean_ctor_set(v_reuseFailAlloc_3548_, 1, v___f_3256_);
v_lattice_3263_ = v_reuseFailAlloc_3548_;
goto v_reusejp_3262_;
}
v_reusejp_3262_:
{
lean_object* v___x_3264_; lean_object* v_toPartialOrder_3265_; lean_object* v___x_3267_; uint8_t v_isShared_3268_; uint8_t v_isSharedCheck_3546_; 
v___x_3264_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_3263_);
v_toPartialOrder_3265_ = lean_ctor_get(v___x_3264_, 0);
v_isSharedCheck_3546_ = !lean_is_exclusive(v___x_3264_);
if (v_isSharedCheck_3546_ == 0)
{
lean_object* v_unused_3547_; 
v_unused_3547_ = lean_ctor_get(v___x_3264_, 1);
lean_dec(v_unused_3547_);
v___x_3267_ = v___x_3264_;
v_isShared_3268_ = v_isSharedCheck_3546_;
goto v_resetjp_3266_;
}
else
{
lean_inc(v_toPartialOrder_3265_);
lean_dec(v___x_3264_);
v___x_3267_ = lean_box(0);
v_isShared_3268_ = v_isSharedCheck_3546_;
goto v_resetjp_3266_;
}
v_resetjp_3266_:
{
lean_object* v_toLE_3269_; lean_object* v_toLT_3270_; lean_object* v___x_3272_; uint8_t v_isShared_3273_; uint8_t v_isSharedCheck_3545_; 
v_toLE_3269_ = lean_ctor_get(v_toPartialOrder_3265_, 0);
v_toLT_3270_ = lean_ctor_get(v_toPartialOrder_3265_, 1);
v_isSharedCheck_3545_ = !lean_is_exclusive(v_toPartialOrder_3265_);
if (v_isSharedCheck_3545_ == 0)
{
v___x_3272_ = v_toPartialOrder_3265_;
v_isShared_3273_ = v_isSharedCheck_3545_;
goto v_resetjp_3271_;
}
else
{
lean_inc(v_toLT_3270_);
lean_inc(v_toLE_3269_);
lean_dec(v_toPartialOrder_3265_);
v___x_3272_ = lean_box(0);
v_isShared_3273_ = v_isSharedCheck_3545_;
goto v_resetjp_3271_;
}
v_resetjp_3271_:
{
lean_object* v_top_3274_; lean_object* v_bot_3275_; lean_object* v_supSet_3276_; lean_object* v_infSet_3277_; lean_object* v___f_3278_; lean_object* v___x_3280_; 
lean_inc_n(v_toFun_3223_, 4);
v_top_3274_ = lean_apply_1(v_toFun_3223_, v_toOrderTop_3217_);
v_bot_3275_ = lean_apply_1(v_toFun_3223_, v_toOrderBot_3218_);
v_supSet_3276_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_3276_, 0, v_toSupSet_3228_);
lean_closure_set(v_supSet_3276_, 1, v_toFun_3223_);
v_infSet_3277_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_3277_, 0, v_toInfSet_3233_);
lean_closure_set(v_infSet_3277_, 1, v_toFun_3223_);
v___f_3278_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_3278_, 0, v___f_3257_);
if (v_isShared_3273_ == 0)
{
v___x_3280_ = v___x_3272_;
goto v_reusejp_3279_;
}
else
{
lean_object* v_reuseFailAlloc_3544_; 
v_reuseFailAlloc_3544_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3544_, 0, v_toLE_3269_);
lean_ctor_set(v_reuseFailAlloc_3544_, 1, v_toLT_3270_);
v___x_3280_ = v_reuseFailAlloc_3544_;
goto v_reusejp_3279_;
}
v_reusejp_3279_:
{
lean_object* v___x_3282_; 
lean_inc_ref(v___f_3278_);
if (v_isShared_3268_ == 0)
{
lean_ctor_set(v___x_3267_, 1, v___f_3278_);
lean_ctor_set(v___x_3267_, 0, v___x_3280_);
v___x_3282_ = v___x_3267_;
goto v_reusejp_3281_;
}
else
{
lean_object* v_reuseFailAlloc_3543_; 
v_reuseFailAlloc_3543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3543_, 0, v___x_3280_);
lean_ctor_set(v_reuseFailAlloc_3543_, 1, v___f_3278_);
v___x_3282_ = v_reuseFailAlloc_3543_;
goto v_reusejp_3281_;
}
v_reusejp_3281_:
{
lean_object* v___x_3284_; 
lean_inc_ref(v___f_3256_);
lean_inc_ref(v___x_3282_);
if (v_isShared_3236_ == 0)
{
lean_ctor_set(v___x_3235_, 1, v___f_3256_);
lean_ctor_set(v___x_3235_, 0, v___x_3282_);
v___x_3284_ = v___x_3235_;
goto v_reusejp_3283_;
}
else
{
lean_object* v_reuseFailAlloc_3542_; 
v_reuseFailAlloc_3542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3542_, 0, v___x_3282_);
lean_ctor_set(v_reuseFailAlloc_3542_, 1, v___f_3256_);
v___x_3284_ = v_reuseFailAlloc_3542_;
goto v_reusejp_3283_;
}
v_reusejp_3283_:
{
lean_object* v___x_3286_; 
if (v_isShared_3221_ == 0)
{
lean_ctor_set(v___x_3220_, 1, v_bot_3275_);
lean_ctor_set(v___x_3220_, 0, v_top_3274_);
v___x_3286_ = v___x_3220_;
goto v_reusejp_3285_;
}
else
{
lean_object* v_reuseFailAlloc_3541_; 
v_reuseFailAlloc_3541_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3541_, 0, v_top_3274_);
lean_ctor_set(v_reuseFailAlloc_3541_, 1, v_bot_3275_);
v___x_3286_ = v_reuseFailAlloc_3541_;
goto v_reusejp_3285_;
}
v_reusejp_3285_:
{
lean_object* v_completeLattice_3287_; lean_object* v___x_3288_; lean_object* v_toHeytingAlgebra_3289_; lean_object* v_toGeneralizedHeytingAlgebra_3290_; lean_object* v_toOrderBot_3291_; lean_object* v_toCompl_3292_; lean_object* v___x_3294_; uint8_t v_isShared_3295_; uint8_t v_isSharedCheck_3540_; 
lean_inc_ref(v___x_3284_);
v_completeLattice_3287_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_3287_, 0, v___x_3284_);
lean_ctor_set(v_completeLattice_3287_, 1, v_supSet_3276_);
lean_ctor_set(v_completeLattice_3287_, 2, v_infSet_3277_);
lean_ctor_set(v_completeLattice_3287_, 3, v___x_3286_);
v___x_3288_ = lp_mathlib_CompleteDistribLattice_toBiheytingAlgebra___redArg(v___x_3212_);
v_toHeytingAlgebra_3289_ = lean_ctor_get(v___x_3288_, 0);
lean_inc_ref(v_toHeytingAlgebra_3289_);
v_toGeneralizedHeytingAlgebra_3290_ = lean_ctor_get(v_toHeytingAlgebra_3289_, 0);
v_toOrderBot_3291_ = lean_ctor_get(v_toHeytingAlgebra_3289_, 1);
v_toCompl_3292_ = lean_ctor_get(v_toHeytingAlgebra_3289_, 2);
v_isSharedCheck_3540_ = !lean_is_exclusive(v_toHeytingAlgebra_3289_);
if (v_isSharedCheck_3540_ == 0)
{
v___x_3294_ = v_toHeytingAlgebra_3289_;
v_isShared_3295_ = v_isSharedCheck_3540_;
goto v_resetjp_3293_;
}
else
{
lean_inc(v_toCompl_3292_);
lean_inc(v_toOrderBot_3291_);
lean_inc(v_toGeneralizedHeytingAlgebra_3290_);
lean_dec(v_toHeytingAlgebra_3289_);
v___x_3294_ = lean_box(0);
v_isShared_3295_ = v_isSharedCheck_3540_;
goto v_resetjp_3293_;
}
v_resetjp_3293_:
{
lean_object* v_toHImp_3296_; lean_object* v___x_3298_; uint8_t v_isShared_3299_; uint8_t v_isSharedCheck_3537_; 
v_toHImp_3296_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_3290_, 2);
v_isSharedCheck_3537_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_3290_);
if (v_isSharedCheck_3537_ == 0)
{
lean_object* v_unused_3538_; lean_object* v_unused_3539_; 
v_unused_3538_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_3290_, 1);
lean_dec(v_unused_3538_);
v_unused_3539_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_3290_, 0);
lean_dec(v_unused_3539_);
v___x_3298_ = v_toGeneralizedHeytingAlgebra_3290_;
v_isShared_3299_ = v_isSharedCheck_3537_;
goto v_resetjp_3297_;
}
else
{
lean_inc(v_toHImp_3296_);
lean_dec(v_toGeneralizedHeytingAlgebra_3290_);
v___x_3298_ = lean_box(0);
v_isShared_3299_ = v_isSharedCheck_3537_;
goto v_resetjp_3297_;
}
v_resetjp_3297_:
{
lean_object* v___x_3300_; lean_object* v_toGeneralizedCoheytingAlgebra_3301_; lean_object* v_toLattice_3302_; lean_object* v_toOrderTop_3303_; lean_object* v_toHNot_3304_; lean_object* v_toOrderBot_3305_; lean_object* v_toSDiff_3306_; lean_object* v_toSemilatticeSup_3307_; lean_object* v_inf_3308_; lean_object* v___x_3310_; uint8_t v_isShared_3311_; uint8_t v_isSharedCheck_3536_; 
v___x_3300_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v___x_3288_);
v_toGeneralizedCoheytingAlgebra_3301_ = lean_ctor_get(v___x_3300_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3301_);
v_toLattice_3302_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3301_, 0);
lean_inc_ref(v_toLattice_3302_);
v_toOrderTop_3303_ = lean_ctor_get(v___x_3300_, 1);
lean_inc(v_toOrderTop_3303_);
v_toHNot_3304_ = lean_ctor_get(v___x_3300_, 2);
lean_inc(v_toHNot_3304_);
lean_dec_ref(v___x_3300_);
v_toOrderBot_3305_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3301_, 1);
lean_inc(v_toOrderBot_3305_);
v_toSDiff_3306_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3301_, 2);
lean_inc(v_toSDiff_3306_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_3301_);
v_toSemilatticeSup_3307_ = lean_ctor_get(v_toLattice_3302_, 0);
v_inf_3308_ = lean_ctor_get(v_toLattice_3302_, 1);
v_isSharedCheck_3536_ = !lean_is_exclusive(v_toLattice_3302_);
if (v_isSharedCheck_3536_ == 0)
{
v___x_3310_ = v_toLattice_3302_;
v_isShared_3311_ = v_isSharedCheck_3536_;
goto v_resetjp_3309_;
}
else
{
lean_inc(v_inf_3308_);
lean_inc(v_toSemilatticeSup_3307_);
lean_dec(v_toLattice_3302_);
v___x_3310_ = lean_box(0);
v_isShared_3311_ = v_isSharedCheck_3536_;
goto v_resetjp_3309_;
}
v_resetjp_3309_:
{
lean_object* v_min_3312_; lean_object* v_semilatticeInf_3313_; lean_object* v_toPartialOrder_3314_; lean_object* v___x_3316_; uint8_t v_isShared_3317_; uint8_t v_isSharedCheck_3534_; 
lean_inc(v_toFun_3223_);
lean_inc_ref(v_e_3210_);
v_min_3312_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_3312_, 0, v___f_3242_);
lean_closure_set(v_min_3312_, 1, v_e_3210_);
lean_closure_set(v_min_3312_, 2, v_inf_3308_);
lean_closure_set(v_min_3312_, 3, v_toFun_3223_);
lean_inc_ref(v_min_3312_);
v_semilatticeInf_3313_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_3312_, v_le_3244_, v_lt_3245_);
v_toPartialOrder_3314_ = lean_ctor_get(v_semilatticeInf_3313_, 0);
v_isSharedCheck_3534_ = !lean_is_exclusive(v_semilatticeInf_3313_);
if (v_isSharedCheck_3534_ == 0)
{
lean_object* v_unused_3535_; 
v_unused_3535_ = lean_ctor_get(v_semilatticeInf_3313_, 1);
lean_dec(v_unused_3535_);
v___x_3316_ = v_semilatticeInf_3313_;
v_isShared_3317_ = v_isSharedCheck_3534_;
goto v_resetjp_3315_;
}
else
{
lean_inc(v_toPartialOrder_3314_);
lean_dec(v_semilatticeInf_3313_);
v___x_3316_ = lean_box(0);
v_isShared_3317_ = v_isSharedCheck_3534_;
goto v_resetjp_3315_;
}
v_resetjp_3315_:
{
lean_object* v_toLE_3318_; lean_object* v_toLT_3319_; lean_object* v___x_3321_; uint8_t v_isShared_3322_; uint8_t v_isSharedCheck_3533_; 
v_toLE_3318_ = lean_ctor_get(v_toPartialOrder_3314_, 0);
v_toLT_3319_ = lean_ctor_get(v_toPartialOrder_3314_, 1);
v_isSharedCheck_3533_ = !lean_is_exclusive(v_toPartialOrder_3314_);
if (v_isSharedCheck_3533_ == 0)
{
v___x_3321_ = v_toPartialOrder_3314_;
v_isShared_3322_ = v_isSharedCheck_3533_;
goto v_resetjp_3320_;
}
else
{
lean_inc(v_toLT_3319_);
lean_inc(v_toLE_3318_);
lean_dec(v_toPartialOrder_3314_);
v___x_3321_ = lean_box(0);
v_isShared_3322_ = v_isSharedCheck_3533_;
goto v_resetjp_3320_;
}
v_resetjp_3320_:
{
lean_object* v___f_3323_; lean_object* v___f_3324_; lean_object* v___x_3326_; 
v___f_3323_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_3323_, 0, v_min_3312_);
lean_inc(v_toFun_3223_);
lean_inc_ref(v_e_3210_);
v___f_3324_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_3324_, 0, v_toSemilatticeSup_3307_);
lean_closure_set(v___f_3324_, 1, v___f_3242_);
lean_closure_set(v___f_3324_, 2, v_e_3210_);
lean_closure_set(v___f_3324_, 3, v_toFun_3223_);
if (v_isShared_3322_ == 0)
{
v___x_3326_ = v___x_3321_;
goto v_reusejp_3325_;
}
else
{
lean_object* v_reuseFailAlloc_3532_; 
v_reuseFailAlloc_3532_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3532_, 0, v_toLE_3318_);
lean_ctor_set(v_reuseFailAlloc_3532_, 1, v_toLT_3319_);
v___x_3326_ = v_reuseFailAlloc_3532_;
goto v_reusejp_3325_;
}
v_reusejp_3325_:
{
lean_object* v___x_3328_; 
lean_inc_ref(v___f_3324_);
if (v_isShared_3317_ == 0)
{
lean_ctor_set(v___x_3316_, 1, v___f_3324_);
lean_ctor_set(v___x_3316_, 0, v___x_3326_);
v___x_3328_ = v___x_3316_;
goto v_reusejp_3327_;
}
else
{
lean_object* v_reuseFailAlloc_3531_; 
v_reuseFailAlloc_3531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3531_, 0, v___x_3326_);
lean_ctor_set(v_reuseFailAlloc_3531_, 1, v___f_3324_);
v___x_3328_ = v_reuseFailAlloc_3531_;
goto v_reusejp_3327_;
}
v_reusejp_3327_:
{
lean_object* v_lattice_3330_; 
lean_inc_ref(v___f_3323_);
if (v_isShared_3311_ == 0)
{
lean_ctor_set(v___x_3310_, 1, v___f_3323_);
lean_ctor_set(v___x_3310_, 0, v___x_3328_);
v_lattice_3330_ = v___x_3310_;
goto v_reusejp_3329_;
}
else
{
lean_object* v_reuseFailAlloc_3530_; 
v_reuseFailAlloc_3530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3530_, 0, v___x_3328_);
lean_ctor_set(v_reuseFailAlloc_3530_, 1, v___f_3323_);
v_lattice_3330_ = v_reuseFailAlloc_3530_;
goto v_reusejp_3329_;
}
v_reusejp_3329_:
{
lean_object* v___x_3331_; lean_object* v_toPartialOrder_3332_; lean_object* v___x_3334_; uint8_t v_isShared_3335_; uint8_t v_isSharedCheck_3528_; 
v___x_3331_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_3330_);
v_toPartialOrder_3332_ = lean_ctor_get(v___x_3331_, 0);
v_isSharedCheck_3528_ = !lean_is_exclusive(v___x_3331_);
if (v_isSharedCheck_3528_ == 0)
{
lean_object* v_unused_3529_; 
v_unused_3529_ = lean_ctor_get(v___x_3331_, 1);
lean_dec(v_unused_3529_);
v___x_3334_ = v___x_3331_;
v_isShared_3335_ = v_isSharedCheck_3528_;
goto v_resetjp_3333_;
}
else
{
lean_inc(v_toPartialOrder_3332_);
lean_dec(v___x_3331_);
v___x_3334_ = lean_box(0);
v_isShared_3335_ = v_isSharedCheck_3528_;
goto v_resetjp_3333_;
}
v_resetjp_3333_:
{
lean_object* v_toLE_3336_; lean_object* v_toLT_3337_; lean_object* v___x_3339_; uint8_t v_isShared_3340_; uint8_t v_isSharedCheck_3527_; 
v_toLE_3336_ = lean_ctor_get(v_toPartialOrder_3332_, 0);
v_toLT_3337_ = lean_ctor_get(v_toPartialOrder_3332_, 1);
v_isSharedCheck_3527_ = !lean_is_exclusive(v_toPartialOrder_3332_);
if (v_isSharedCheck_3527_ == 0)
{
v___x_3339_ = v_toPartialOrder_3332_;
v_isShared_3340_ = v_isSharedCheck_3527_;
goto v_resetjp_3338_;
}
else
{
lean_inc(v_toLT_3337_);
lean_inc(v_toLE_3336_);
lean_dec(v_toPartialOrder_3332_);
v___x_3339_ = lean_box(0);
v_isShared_3340_ = v_isSharedCheck_3527_;
goto v_resetjp_3338_;
}
v_resetjp_3338_:
{
lean_object* v_bot_3341_; lean_object* v___f_3342_; lean_object* v___x_3344_; 
lean_inc(v_toFun_3223_);
v_bot_3341_ = lean_apply_1(v_toFun_3223_, v_toOrderBot_3291_);
v___f_3342_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_3342_, 0, v___f_3324_);
if (v_isShared_3340_ == 0)
{
v___x_3344_ = v___x_3339_;
goto v_reusejp_3343_;
}
else
{
lean_object* v_reuseFailAlloc_3526_; 
v_reuseFailAlloc_3526_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3526_, 0, v_toLE_3336_);
lean_ctor_set(v_reuseFailAlloc_3526_, 1, v_toLT_3337_);
v___x_3344_ = v_reuseFailAlloc_3526_;
goto v_reusejp_3343_;
}
v_reusejp_3343_:
{
lean_object* v___x_3346_; 
lean_inc_ref(v___f_3342_);
if (v_isShared_3335_ == 0)
{
lean_ctor_set(v___x_3334_, 1, v___f_3342_);
lean_ctor_set(v___x_3334_, 0, v___x_3344_);
v___x_3346_ = v___x_3334_;
goto v_reusejp_3345_;
}
else
{
lean_object* v_reuseFailAlloc_3525_; 
v_reuseFailAlloc_3525_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3525_, 0, v___x_3344_);
lean_ctor_set(v_reuseFailAlloc_3525_, 1, v___f_3342_);
v___x_3346_ = v_reuseFailAlloc_3525_;
goto v_reusejp_3345_;
}
v_reusejp_3345_:
{
lean_object* v___x_3348_; 
lean_inc_ref(v___f_3323_);
lean_inc_ref(v___x_3346_);
if (v_isShared_3231_ == 0)
{
lean_ctor_set(v___x_3230_, 1, v___f_3323_);
lean_ctor_set(v___x_3230_, 0, v___x_3346_);
v___x_3348_ = v___x_3230_;
goto v_reusejp_3347_;
}
else
{
lean_object* v_reuseFailAlloc_3524_; 
v_reuseFailAlloc_3524_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3524_, 0, v___x_3346_);
lean_ctor_set(v_reuseFailAlloc_3524_, 1, v___f_3323_);
v___x_3348_ = v_reuseFailAlloc_3524_;
goto v_reusejp_3347_;
}
v_reusejp_3347_:
{
lean_object* v___x_3349_; lean_object* v_toPartialOrder_3350_; lean_object* v_toLE_3351_; lean_object* v_toLT_3352_; lean_object* v___x_3354_; uint8_t v_isShared_3355_; uint8_t v_isSharedCheck_3523_; 
v___x_3349_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_3348_);
v_toPartialOrder_3350_ = lean_ctor_get(v___x_3349_, 0);
lean_inc_ref(v_toPartialOrder_3350_);
v_toLE_3351_ = lean_ctor_get(v_toPartialOrder_3350_, 0);
v_toLT_3352_ = lean_ctor_get(v_toPartialOrder_3350_, 1);
v_isSharedCheck_3523_ = !lean_is_exclusive(v_toPartialOrder_3350_);
if (v_isSharedCheck_3523_ == 0)
{
v___x_3354_ = v_toPartialOrder_3350_;
v_isShared_3355_ = v_isSharedCheck_3523_;
goto v_resetjp_3353_;
}
else
{
lean_inc(v_toLT_3352_);
lean_inc(v_toLE_3351_);
lean_dec(v_toPartialOrder_3350_);
v___x_3354_ = lean_box(0);
v_isShared_3355_ = v_isSharedCheck_3523_;
goto v_resetjp_3353_;
}
v_resetjp_3353_:
{
lean_object* v_bot_3356_; lean_object* v_hnot_3357_; lean_object* v_sdiff_3358_; lean_object* v_top_3359_; lean_object* v___f_3360_; lean_object* v___f_3361_; lean_object* v_coheytingAlgebra_3362_; lean_object* v_toGeneralizedCoheytingAlgebra_3363_; lean_object* v_toLattice_3364_; lean_object* v_toOrderTop_3365_; lean_object* v_toHNot_3366_; lean_object* v_toSDiff_3367_; lean_object* v_toSemilatticeSup_3368_; lean_object* v___x_3369_; lean_object* v_toPartialOrder_3370_; lean_object* v_toLE_3371_; lean_object* v_toLT_3372_; lean_object* v___x_3374_; uint8_t v_isShared_3375_; uint8_t v_isSharedCheck_3522_; 
lean_inc_n(v_toFun_3223_, 4);
v_bot_3356_ = lean_apply_1(v_toFun_3223_, v_toOrderBot_3305_);
lean_inc_ref_n(v_e_3210_, 2);
v_hnot_3357_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__17), 4, 3);
lean_closure_set(v_hnot_3357_, 0, v_e_3210_);
lean_closure_set(v_hnot_3357_, 1, v_toHNot_3304_);
lean_closure_set(v_hnot_3357_, 2, v_toFun_3223_);
v_sdiff_3358_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_3358_, 0, v___f_3242_);
lean_closure_set(v_sdiff_3358_, 1, v_e_3210_);
lean_closure_set(v_sdiff_3358_, 2, v_toSDiff_3306_);
lean_closure_set(v_sdiff_3358_, 3, v_toFun_3223_);
v_top_3359_ = lean_apply_1(v_toFun_3223_, v_toOrderTop_3303_);
v___f_3360_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3360_, 0, v___x_3349_);
v___f_3361_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3361_, 0, v___x_3346_);
v_coheytingAlgebra_3362_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_3360_, v___f_3361_, v_toLE_3351_, v_toLT_3352_, v_bot_3356_, v_top_3359_, v_hnot_3357_, v_sdiff_3358_);
v_toGeneralizedCoheytingAlgebra_3363_ = lean_ctor_get(v_coheytingAlgebra_3362_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3363_);
v_toLattice_3364_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3363_, 0);
lean_inc_ref(v_toLattice_3364_);
v_toOrderTop_3365_ = lean_ctor_get(v_coheytingAlgebra_3362_, 1);
lean_inc(v_toOrderTop_3365_);
v_toHNot_3366_ = lean_ctor_get(v_coheytingAlgebra_3362_, 2);
lean_inc(v_toHNot_3366_);
lean_dec_ref(v_coheytingAlgebra_3362_);
v_toSDiff_3367_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3363_, 2);
lean_inc(v_toSDiff_3367_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_3363_);
v_toSemilatticeSup_3368_ = lean_ctor_get(v_toLattice_3364_, 0);
lean_inc_ref(v_toSemilatticeSup_3368_);
v___x_3369_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_3364_);
v_toPartialOrder_3370_ = lean_ctor_get(v___x_3369_, 0);
lean_inc_ref(v_toPartialOrder_3370_);
v_toLE_3371_ = lean_ctor_get(v_toPartialOrder_3370_, 0);
v_toLT_3372_ = lean_ctor_get(v_toPartialOrder_3370_, 1);
v_isSharedCheck_3522_ = !lean_is_exclusive(v_toPartialOrder_3370_);
if (v_isSharedCheck_3522_ == 0)
{
v___x_3374_ = v_toPartialOrder_3370_;
v_isShared_3375_ = v_isSharedCheck_3522_;
goto v_resetjp_3373_;
}
else
{
lean_inc(v_toLT_3372_);
lean_inc(v_toLE_3371_);
lean_dec(v_toPartialOrder_3370_);
v___x_3374_ = lean_box(0);
v_isShared_3375_ = v_isSharedCheck_3522_;
goto v_resetjp_3373_;
}
v_resetjp_3373_:
{
lean_object* v_compl_3376_; lean_object* v_himp_3377_; lean_object* v___f_3378_; lean_object* v___f_3379_; lean_object* v___x_3381_; 
lean_inc(v_toFun_3223_);
lean_inc_ref(v_e_3210_);
v_compl_3376_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_3376_, 0, v_e_3210_);
lean_closure_set(v_compl_3376_, 1, v_toCompl_3292_);
lean_closure_set(v_compl_3376_, 2, v_toFun_3223_);
v_himp_3377_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_3377_, 0, v___f_3242_);
lean_closure_set(v_himp_3377_, 1, v_e_3210_);
lean_closure_set(v_himp_3377_, 2, v_toHImp_3296_);
lean_closure_set(v_himp_3377_, 3, v_toFun_3223_);
v___f_3378_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3378_, 0, v_toSemilatticeSup_3368_);
v___f_3379_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3379_, 0, v___x_3369_);
if (v_isShared_3375_ == 0)
{
v___x_3381_ = v___x_3374_;
goto v_reusejp_3380_;
}
else
{
lean_object* v_reuseFailAlloc_3521_; 
v_reuseFailAlloc_3521_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3521_, 0, v_toLE_3371_);
lean_ctor_set(v_reuseFailAlloc_3521_, 1, v_toLT_3372_);
v___x_3381_ = v_reuseFailAlloc_3521_;
goto v_reusejp_3380_;
}
v_reusejp_3380_:
{
lean_object* v___x_3383_; 
if (v_isShared_3355_ == 0)
{
lean_ctor_set(v___x_3354_, 1, v___f_3342_);
lean_ctor_set(v___x_3354_, 0, v___x_3381_);
v___x_3383_ = v___x_3354_;
goto v_reusejp_3382_;
}
else
{
lean_object* v_reuseFailAlloc_3520_; 
v_reuseFailAlloc_3520_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3520_, 0, v___x_3381_);
lean_ctor_set(v_reuseFailAlloc_3520_, 1, v___f_3342_);
v___x_3383_ = v_reuseFailAlloc_3520_;
goto v_reusejp_3382_;
}
v_reusejp_3382_:
{
lean_object* v___x_3385_; 
if (v_isShared_3226_ == 0)
{
lean_ctor_set(v___x_3225_, 1, v___f_3323_);
lean_ctor_set(v___x_3225_, 0, v___x_3383_);
v___x_3385_ = v___x_3225_;
goto v_reusejp_3384_;
}
else
{
lean_object* v_reuseFailAlloc_3519_; 
v_reuseFailAlloc_3519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3519_, 0, v___x_3383_);
lean_ctor_set(v_reuseFailAlloc_3519_, 1, v___f_3323_);
v___x_3385_ = v_reuseFailAlloc_3519_;
goto v_reusejp_3384_;
}
v_reusejp_3384_:
{
lean_object* v___x_3387_; 
lean_inc_ref(v_himp_3377_);
lean_inc(v_toOrderTop_3365_);
if (v_isShared_3299_ == 0)
{
lean_ctor_set(v___x_3298_, 2, v_himp_3377_);
lean_ctor_set(v___x_3298_, 1, v_toOrderTop_3365_);
lean_ctor_set(v___x_3298_, 0, v___x_3385_);
v___x_3387_ = v___x_3298_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3518_; 
v_reuseFailAlloc_3518_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3518_, 0, v___x_3385_);
lean_ctor_set(v_reuseFailAlloc_3518_, 1, v_toOrderTop_3365_);
lean_ctor_set(v_reuseFailAlloc_3518_, 2, v_himp_3377_);
v___x_3387_ = v_reuseFailAlloc_3518_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
lean_object* v___x_3389_; 
lean_inc_ref(v_compl_3376_);
lean_inc(v_bot_3341_);
if (v_isShared_3295_ == 0)
{
lean_ctor_set(v___x_3294_, 2, v_compl_3376_);
lean_ctor_set(v___x_3294_, 1, v_bot_3341_);
lean_ctor_set(v___x_3294_, 0, v___x_3387_);
v___x_3389_ = v___x_3294_;
goto v_reusejp_3388_;
}
else
{
lean_object* v_reuseFailAlloc_3517_; 
v_reuseFailAlloc_3517_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3517_, 0, v___x_3387_);
lean_ctor_set(v_reuseFailAlloc_3517_, 1, v_bot_3341_);
lean_ctor_set(v_reuseFailAlloc_3517_, 2, v_compl_3376_);
v___x_3389_ = v_reuseFailAlloc_3517_;
goto v_reusejp_3388_;
}
v_reusejp_3388_:
{
lean_object* v___x_3390_; lean_object* v_toGeneralizedCoheytingAlgebra_3391_; lean_object* v_toHNot_3392_; lean_object* v_toSDiff_3393_; lean_object* v___x_3395_; uint8_t v_isShared_3396_; uint8_t v_isSharedCheck_3514_; 
lean_inc(v_bot_3341_);
v___x_3390_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_3379_, v___f_3378_, v_toLE_3371_, v_toLT_3372_, v_bot_3341_, v_toOrderTop_3365_, v_toHNot_3366_, v_toSDiff_3367_);
v_toGeneralizedCoheytingAlgebra_3391_ = lean_ctor_get(v___x_3390_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3391_);
v_toHNot_3392_ = lean_ctor_get(v___x_3390_, 2);
lean_inc(v_toHNot_3392_);
lean_dec_ref(v___x_3390_);
v_toSDiff_3393_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3391_, 2);
v_isSharedCheck_3514_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_3391_);
if (v_isSharedCheck_3514_ == 0)
{
lean_object* v_unused_3515_; lean_object* v_unused_3516_; 
v_unused_3515_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3391_, 1);
lean_dec(v_unused_3515_);
v_unused_3516_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3391_, 0);
lean_dec(v_unused_3516_);
v___x_3395_ = v_toGeneralizedCoheytingAlgebra_3391_;
v_isShared_3396_ = v_isSharedCheck_3514_;
goto v_resetjp_3394_;
}
else
{
lean_inc(v_toSDiff_3393_);
lean_dec(v_toGeneralizedCoheytingAlgebra_3391_);
v___x_3395_ = lean_box(0);
v_isShared_3396_ = v_isSharedCheck_3514_;
goto v_resetjp_3394_;
}
v_resetjp_3394_:
{
lean_object* v_biheytingAlgebra_3398_; 
if (v_isShared_3396_ == 0)
{
lean_ctor_set(v___x_3395_, 2, v_toHNot_3392_);
lean_ctor_set(v___x_3395_, 1, v_toSDiff_3393_);
lean_ctor_set(v___x_3395_, 0, v___x_3389_);
v_biheytingAlgebra_3398_ = v___x_3395_;
goto v_reusejp_3397_;
}
else
{
lean_object* v_reuseFailAlloc_3513_; 
v_reuseFailAlloc_3513_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3513_, 0, v___x_3389_);
lean_ctor_set(v_reuseFailAlloc_3513_, 1, v_toSDiff_3393_);
lean_ctor_set(v_reuseFailAlloc_3513_, 2, v_toHNot_3392_);
v_biheytingAlgebra_3398_ = v_reuseFailAlloc_3513_;
goto v_reusejp_3397_;
}
v_reusejp_3397_:
{
lean_object* v___x_3399_; lean_object* v___x_3400_; lean_object* v_toPartialOrder_3401_; lean_object* v_toInfSet_3402_; lean_object* v___x_3404_; uint8_t v_isShared_3405_; uint8_t v_isSharedCheck_3512_; 
v___x_3399_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_3284_);
lean_inc_ref(v_completeLattice_3287_);
v___x_3400_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_3287_);
v_toPartialOrder_3401_ = lean_ctor_get(v___x_3400_, 0);
v_toInfSet_3402_ = lean_ctor_get(v___x_3400_, 1);
v_isSharedCheck_3512_ = !lean_is_exclusive(v___x_3400_);
if (v_isSharedCheck_3512_ == 0)
{
v___x_3404_ = v___x_3400_;
v_isShared_3405_ = v_isSharedCheck_3512_;
goto v_resetjp_3403_;
}
else
{
lean_inc(v_toInfSet_3402_);
lean_inc(v_toPartialOrder_3401_);
lean_dec(v___x_3400_);
v___x_3404_ = lean_box(0);
v_isShared_3405_ = v_isSharedCheck_3512_;
goto v_resetjp_3403_;
}
v_resetjp_3403_:
{
lean_object* v_toLE_3406_; lean_object* v_toLT_3407_; lean_object* v___x_3409_; uint8_t v_isShared_3410_; uint8_t v_isSharedCheck_3511_; 
v_toLE_3406_ = lean_ctor_get(v_toPartialOrder_3401_, 0);
v_toLT_3407_ = lean_ctor_get(v_toPartialOrder_3401_, 1);
v_isSharedCheck_3511_ = !lean_is_exclusive(v_toPartialOrder_3401_);
if (v_isSharedCheck_3511_ == 0)
{
v___x_3409_ = v_toPartialOrder_3401_;
v_isShared_3410_ = v_isSharedCheck_3511_;
goto v_resetjp_3408_;
}
else
{
lean_inc(v_toLT_3407_);
lean_inc(v_toLE_3406_);
lean_dec(v_toPartialOrder_3401_);
v___x_3409_ = lean_box(0);
v_isShared_3410_ = v_isSharedCheck_3511_;
goto v_resetjp_3408_;
}
v_resetjp_3408_:
{
lean_object* v___x_3411_; lean_object* v_toSupSet_3412_; lean_object* v___x_3414_; uint8_t v_isShared_3415_; uint8_t v_isSharedCheck_3509_; 
v___x_3411_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_3287_);
v_toSupSet_3412_ = lean_ctor_get(v___x_3411_, 1);
v_isSharedCheck_3509_ = !lean_is_exclusive(v___x_3411_);
if (v_isSharedCheck_3509_ == 0)
{
lean_object* v_unused_3510_; 
v_unused_3510_ = lean_ctor_get(v___x_3411_, 0);
lean_dec(v_unused_3510_);
v___x_3414_ = v___x_3411_;
v_isShared_3415_ = v_isSharedCheck_3509_;
goto v_resetjp_3413_;
}
else
{
lean_inc(v_toSupSet_3412_);
lean_dec(v___x_3411_);
v___x_3414_ = lean_box(0);
v_isShared_3415_ = v_isSharedCheck_3509_;
goto v_resetjp_3413_;
}
v_resetjp_3413_:
{
lean_object* v___x_3416_; lean_object* v_toGeneralizedCoheytingAlgebra_3417_; lean_object* v_toOrderTop_3418_; lean_object* v_toHNot_3419_; lean_object* v_toSDiff_3420_; lean_object* v___x_3422_; uint8_t v_isShared_3423_; uint8_t v_isSharedCheck_3506_; 
v___x_3416_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_biheytingAlgebra_3398_);
v_toGeneralizedCoheytingAlgebra_3417_ = lean_ctor_get(v___x_3416_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3417_);
v_toOrderTop_3418_ = lean_ctor_get(v___x_3416_, 1);
lean_inc(v_toOrderTop_3418_);
v_toHNot_3419_ = lean_ctor_get(v___x_3416_, 2);
lean_inc(v_toHNot_3419_);
lean_dec_ref(v___x_3416_);
v_toSDiff_3420_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3417_, 2);
v_isSharedCheck_3506_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_3417_);
if (v_isSharedCheck_3506_ == 0)
{
lean_object* v_unused_3507_; lean_object* v_unused_3508_; 
v_unused_3507_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3417_, 1);
lean_dec(v_unused_3507_);
v_unused_3508_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3417_, 0);
lean_dec(v_unused_3508_);
v___x_3422_ = v_toGeneralizedCoheytingAlgebra_3417_;
v_isShared_3423_ = v_isSharedCheck_3506_;
goto v_resetjp_3421_;
}
else
{
lean_inc(v_toSDiff_3420_);
lean_dec(v_toGeneralizedCoheytingAlgebra_3417_);
v___x_3422_ = lean_box(0);
v_isShared_3423_ = v_isSharedCheck_3506_;
goto v_resetjp_3421_;
}
v_resetjp_3421_:
{
lean_object* v___f_3424_; lean_object* v___f_3425_; lean_object* v___x_3427_; 
v___f_3424_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3424_, 0, v___x_3282_);
v___f_3425_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3425_, 0, v___x_3399_);
if (v_isShared_3410_ == 0)
{
v___x_3427_ = v___x_3409_;
goto v_reusejp_3426_;
}
else
{
lean_object* v_reuseFailAlloc_3505_; 
v_reuseFailAlloc_3505_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3505_, 0, v_toLE_3406_);
lean_ctor_set(v_reuseFailAlloc_3505_, 1, v_toLT_3407_);
v___x_3427_ = v_reuseFailAlloc_3505_;
goto v_reusejp_3426_;
}
v_reusejp_3426_:
{
lean_object* v___x_3429_; 
lean_inc_ref(v___f_3278_);
if (v_isShared_3415_ == 0)
{
lean_ctor_set(v___x_3414_, 1, v___f_3278_);
lean_ctor_set(v___x_3414_, 0, v___x_3427_);
v___x_3429_ = v___x_3414_;
goto v_reusejp_3428_;
}
else
{
lean_object* v_reuseFailAlloc_3504_; 
v_reuseFailAlloc_3504_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3504_, 0, v___x_3427_);
lean_ctor_set(v_reuseFailAlloc_3504_, 1, v___f_3278_);
v___x_3429_ = v_reuseFailAlloc_3504_;
goto v_reusejp_3428_;
}
v_reusejp_3428_:
{
lean_object* v___x_3431_; 
lean_inc_ref(v___f_3256_);
if (v_isShared_3405_ == 0)
{
lean_ctor_set(v___x_3404_, 1, v___f_3256_);
lean_ctor_set(v___x_3404_, 0, v___x_3429_);
v___x_3431_ = v___x_3404_;
goto v_reusejp_3430_;
}
else
{
lean_object* v_reuseFailAlloc_3503_; 
v_reuseFailAlloc_3503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3503_, 0, v___x_3429_);
lean_ctor_set(v_reuseFailAlloc_3503_, 1, v___f_3256_);
v___x_3431_ = v_reuseFailAlloc_3503_;
goto v_reusejp_3430_;
}
v_reusejp_3430_:
{
lean_object* v___x_3432_; lean_object* v___x_3433_; lean_object* v___x_3435_; 
lean_inc(v_bot_3341_);
lean_inc(v_toOrderTop_3418_);
v___x_3432_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3432_, 0, v_toOrderTop_3418_);
lean_ctor_set(v___x_3432_, 1, v_bot_3341_);
v___x_3433_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3433_, 0, v___x_3431_);
lean_ctor_set(v___x_3433_, 1, v_toSupSet_3412_);
lean_ctor_set(v___x_3433_, 2, v_toInfSet_3402_);
lean_ctor_set(v___x_3433_, 3, v___x_3432_);
if (v_isShared_3423_ == 0)
{
lean_ctor_set(v___x_3422_, 2, v_compl_3376_);
lean_ctor_set(v___x_3422_, 1, v_himp_3377_);
lean_ctor_set(v___x_3422_, 0, v___x_3433_);
v___x_3435_ = v___x_3422_;
goto v_reusejp_3434_;
}
else
{
lean_object* v_reuseFailAlloc_3502_; 
v_reuseFailAlloc_3502_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3502_, 0, v___x_3433_);
lean_ctor_set(v_reuseFailAlloc_3502_, 1, v_himp_3377_);
lean_ctor_set(v_reuseFailAlloc_3502_, 2, v_compl_3376_);
v___x_3435_ = v_reuseFailAlloc_3502_;
goto v_reusejp_3434_;
}
v_reusejp_3434_:
{
lean_object* v___x_3436_; lean_object* v_toGeneralizedCoheytingAlgebra_3437_; lean_object* v_toHNot_3438_; lean_object* v_toSDiff_3439_; lean_object* v___x_3441_; uint8_t v_isShared_3442_; uint8_t v_isSharedCheck_3499_; 
v___x_3436_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_3425_, v___f_3424_, v_toLE_3406_, v_toLT_3407_, v_bot_3341_, v_toOrderTop_3418_, v_toHNot_3419_, v_toSDiff_3420_);
v_toGeneralizedCoheytingAlgebra_3437_ = lean_ctor_get(v___x_3436_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3437_);
v_toHNot_3438_ = lean_ctor_get(v___x_3436_, 2);
lean_inc(v_toHNot_3438_);
lean_dec_ref(v___x_3436_);
v_toSDiff_3439_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3437_, 2);
v_isSharedCheck_3499_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_3437_);
if (v_isSharedCheck_3499_ == 0)
{
lean_object* v_unused_3500_; lean_object* v_unused_3501_; 
v_unused_3500_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3437_, 1);
lean_dec(v_unused_3500_);
v_unused_3501_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3437_, 0);
lean_dec(v_unused_3501_);
v___x_3441_ = v_toGeneralizedCoheytingAlgebra_3437_;
v_isShared_3442_ = v_isSharedCheck_3499_;
goto v_resetjp_3440_;
}
else
{
lean_inc(v_toSDiff_3439_);
lean_dec(v_toGeneralizedCoheytingAlgebra_3437_);
v___x_3441_ = lean_box(0);
v_isShared_3442_ = v_isSharedCheck_3499_;
goto v_resetjp_3440_;
}
v_resetjp_3440_:
{
lean_object* v_completeDistribLattice_3444_; 
lean_inc_ref(v___x_3435_);
if (v_isShared_3442_ == 0)
{
lean_ctor_set(v___x_3441_, 2, v_toHNot_3438_);
lean_ctor_set(v___x_3441_, 1, v_toSDiff_3439_);
lean_ctor_set(v___x_3441_, 0, v___x_3435_);
v_completeDistribLattice_3444_ = v___x_3441_;
goto v_reusejp_3443_;
}
else
{
lean_object* v_reuseFailAlloc_3498_; 
v_reuseFailAlloc_3498_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3498_, 0, v___x_3435_);
lean_ctor_set(v_reuseFailAlloc_3498_, 1, v_toSDiff_3439_);
lean_ctor_set(v_reuseFailAlloc_3498_, 2, v_toHNot_3438_);
v_completeDistribLattice_3444_ = v_reuseFailAlloc_3498_;
goto v_reusejp_3443_;
}
v_reusejp_3443_:
{
lean_object* v___x_3445_; lean_object* v_toCompleteLattice_3446_; lean_object* v_toLattice_3447_; lean_object* v_toSemilatticeSup_3448_; lean_object* v___x_3449_; lean_object* v___x_3450_; lean_object* v_toPartialOrder_3451_; lean_object* v_toInfSet_3452_; lean_object* v___x_3454_; uint8_t v_isShared_3455_; uint8_t v_isSharedCheck_3497_; 
v___x_3445_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v_completeDistribLattice_3444_);
v_toCompleteLattice_3446_ = lean_ctor_get(v___x_3445_, 0);
lean_inc_ref_n(v_toCompleteLattice_3446_, 2);
v_toLattice_3447_ = lean_ctor_get(v_toCompleteLattice_3446_, 0);
v_toSemilatticeSup_3448_ = lean_ctor_get(v_toLattice_3447_, 0);
lean_inc_ref(v_toSemilatticeSup_3448_);
lean_inc_ref(v_toLattice_3447_);
v___x_3449_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_3447_);
v___x_3450_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_3446_);
v_toPartialOrder_3451_ = lean_ctor_get(v___x_3450_, 0);
v_toInfSet_3452_ = lean_ctor_get(v___x_3450_, 1);
v_isSharedCheck_3497_ = !lean_is_exclusive(v___x_3450_);
if (v_isSharedCheck_3497_ == 0)
{
v___x_3454_ = v___x_3450_;
v_isShared_3455_ = v_isSharedCheck_3497_;
goto v_resetjp_3453_;
}
else
{
lean_inc(v_toInfSet_3452_);
lean_inc(v_toPartialOrder_3451_);
lean_dec(v___x_3450_);
v___x_3454_ = lean_box(0);
v_isShared_3455_ = v_isSharedCheck_3497_;
goto v_resetjp_3453_;
}
v_resetjp_3453_:
{
lean_object* v_toLE_3456_; lean_object* v_toLT_3457_; lean_object* v___x_3459_; uint8_t v_isShared_3460_; uint8_t v_isSharedCheck_3496_; 
v_toLE_3456_ = lean_ctor_get(v_toPartialOrder_3451_, 0);
v_toLT_3457_ = lean_ctor_get(v_toPartialOrder_3451_, 1);
v_isSharedCheck_3496_ = !lean_is_exclusive(v_toPartialOrder_3451_);
if (v_isSharedCheck_3496_ == 0)
{
v___x_3459_ = v_toPartialOrder_3451_;
v_isShared_3460_ = v_isSharedCheck_3496_;
goto v_resetjp_3458_;
}
else
{
lean_inc(v_toLT_3457_);
lean_inc(v_toLE_3456_);
lean_dec(v_toPartialOrder_3451_);
v___x_3459_ = lean_box(0);
v_isShared_3460_ = v_isSharedCheck_3496_;
goto v_resetjp_3458_;
}
v_resetjp_3458_:
{
lean_object* v___x_3461_; lean_object* v_toSupSet_3462_; lean_object* v___x_3464_; uint8_t v_isShared_3465_; uint8_t v_isSharedCheck_3494_; 
v___x_3461_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_3446_);
v_toSupSet_3462_ = lean_ctor_get(v___x_3461_, 1);
v_isSharedCheck_3494_ = !lean_is_exclusive(v___x_3461_);
if (v_isSharedCheck_3494_ == 0)
{
lean_object* v_unused_3495_; 
v_unused_3495_ = lean_ctor_get(v___x_3461_, 0);
lean_dec(v_unused_3495_);
v___x_3464_ = v___x_3461_;
v_isShared_3465_ = v_isSharedCheck_3494_;
goto v_resetjp_3463_;
}
else
{
lean_inc(v_toSupSet_3462_);
lean_dec(v___x_3461_);
v___x_3464_ = lean_box(0);
v_isShared_3465_ = v_isSharedCheck_3494_;
goto v_resetjp_3463_;
}
v_resetjp_3463_:
{
lean_object* v___x_3466_; lean_object* v_toGeneralizedCoheytingAlgebra_3467_; lean_object* v_toOrderTop_3468_; lean_object* v_toHNot_3469_; lean_object* v___x_3470_; lean_object* v_toGeneralizedHeytingAlgebra_3471_; lean_object* v_toOrderBot_3472_; lean_object* v_toCompl_3473_; lean_object* v_toHImp_3474_; lean_object* v_toSDiff_3475_; lean_object* v___f_3476_; lean_object* v___f_3477_; lean_object* v___x_3479_; 
v___x_3466_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v___x_3445_);
v_toGeneralizedCoheytingAlgebra_3467_ = lean_ctor_get(v___x_3466_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3467_);
v_toOrderTop_3468_ = lean_ctor_get(v___x_3466_, 1);
lean_inc(v_toOrderTop_3468_);
v_toHNot_3469_ = lean_ctor_get(v___x_3466_, 2);
lean_inc(v_toHNot_3469_);
lean_dec_ref(v___x_3466_);
v___x_3470_ = lp_mathlib_Order_Frame_toHeytingAlgebra___redArg(v___x_3435_);
v_toGeneralizedHeytingAlgebra_3471_ = lean_ctor_get(v___x_3470_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_3471_);
v_toOrderBot_3472_ = lean_ctor_get(v___x_3470_, 1);
lean_inc(v_toOrderBot_3472_);
v_toCompl_3473_ = lean_ctor_get(v___x_3470_, 2);
lean_inc(v_toCompl_3473_);
lean_dec_ref(v___x_3470_);
v_toHImp_3474_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_3471_, 2);
lean_inc(v_toHImp_3474_);
lean_dec_ref(v_toGeneralizedHeytingAlgebra_3471_);
v_toSDiff_3475_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3467_, 2);
lean_inc(v_toSDiff_3475_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_3467_);
v___f_3476_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3476_, 0, v_toSemilatticeSup_3448_);
v___f_3477_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_3477_, 0, v___x_3449_);
if (v_isShared_3460_ == 0)
{
v___x_3479_ = v___x_3459_;
goto v_reusejp_3478_;
}
else
{
lean_object* v_reuseFailAlloc_3493_; 
v_reuseFailAlloc_3493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3493_, 0, v_toLE_3456_);
lean_ctor_set(v_reuseFailAlloc_3493_, 1, v_toLT_3457_);
v___x_3479_ = v_reuseFailAlloc_3493_;
goto v_reusejp_3478_;
}
v_reusejp_3478_:
{
lean_object* v___x_3481_; 
if (v_isShared_3465_ == 0)
{
lean_ctor_set(v___x_3464_, 1, v___f_3278_);
lean_ctor_set(v___x_3464_, 0, v___x_3479_);
v___x_3481_ = v___x_3464_;
goto v_reusejp_3480_;
}
else
{
lean_object* v_reuseFailAlloc_3492_; 
v_reuseFailAlloc_3492_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3492_, 0, v___x_3479_);
lean_ctor_set(v_reuseFailAlloc_3492_, 1, v___f_3278_);
v___x_3481_ = v_reuseFailAlloc_3492_;
goto v_reusejp_3480_;
}
v_reusejp_3480_:
{
lean_object* v___x_3483_; 
if (v_isShared_3455_ == 0)
{
lean_ctor_set(v___x_3454_, 1, v___f_3256_);
lean_ctor_set(v___x_3454_, 0, v___x_3481_);
v___x_3483_ = v___x_3454_;
goto v_reusejp_3482_;
}
else
{
lean_object* v_reuseFailAlloc_3491_; 
v_reuseFailAlloc_3491_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3491_, 0, v___x_3481_);
lean_ctor_set(v_reuseFailAlloc_3491_, 1, v___f_3256_);
v___x_3483_ = v_reuseFailAlloc_3491_;
goto v_reusejp_3482_;
}
v_reusejp_3482_:
{
lean_object* v___x_3484_; lean_object* v___x_3485_; lean_object* v___x_3486_; lean_object* v_toGeneralizedCoheytingAlgebra_3487_; lean_object* v_toHNot_3488_; lean_object* v_toSDiff_3489_; lean_object* v___x_3490_; 
lean_inc(v_toOrderBot_3472_);
lean_inc(v_toOrderTop_3468_);
v___x_3484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3484_, 0, v_toOrderTop_3468_);
lean_ctor_set(v___x_3484_, 1, v_toOrderBot_3472_);
v___x_3485_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3485_, 0, v___x_3483_);
lean_ctor_set(v___x_3485_, 1, v_toSupSet_3462_);
lean_ctor_set(v___x_3485_, 2, v_toInfSet_3452_);
lean_ctor_set(v___x_3485_, 3, v___x_3484_);
v___x_3486_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_3477_, v___f_3476_, v_toLE_3456_, v_toLT_3457_, v_toOrderBot_3472_, v_toOrderTop_3468_, v_toHNot_3469_, v_toSDiff_3475_);
v_toGeneralizedCoheytingAlgebra_3487_ = lean_ctor_get(v___x_3486_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_3487_);
v_toHNot_3488_ = lean_ctor_get(v___x_3486_, 2);
lean_inc(v_toHNot_3488_);
lean_dec_ref(v___x_3486_);
v_toSDiff_3489_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_3487_, 2);
lean_inc(v_toSDiff_3489_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_3487_);
v___x_3490_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3490_, 0, v___x_3485_);
lean_ctor_set(v___x_3490_, 1, v_toHImp_3474_);
lean_ctor_set(v___x_3490_, 2, v_toCompl_3473_);
lean_ctor_set(v___x_3490_, 3, v_toSDiff_3489_);
lean_ctor_set(v___x_3490_, 4, v_toHNot_3488_);
return v___x_3490_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completelyDistribLattice___boxed(lean_object* v_00_u03b1_3562_, lean_object* v_00_u03b2_3563_, lean_object* v_e_3564_, lean_object* v_inst_3565_){
_start:
{
lean_object* v_res_3566_; 
v_res_3566_ = lp_mathlib_Equiv_completelyDistribLattice(v_00_u03b1_3562_, v_00_u03b2_3563_, v_e_3564_, v_inst_3565_);
lean_dec_ref(v_inst_3565_);
return v_res_3566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeBooleanAlgebra___redArg(lean_object* v_e_3567_, lean_object* v_inst_3568_){
_start:
{
lean_object* v_toCompleteLattice_3569_; lean_object* v_toBoundedOrder_3570_; lean_object* v_toLattice_3571_; lean_object* v_toOrderTop_3572_; lean_object* v_toOrderBot_3573_; lean_object* v___x_3575_; uint8_t v_isShared_3576_; uint8_t v_isSharedCheck_3714_; 
v_toCompleteLattice_3569_ = lean_ctor_get(v_inst_3568_, 0);
v_toBoundedOrder_3570_ = lean_ctor_get(v_toCompleteLattice_3569_, 3);
lean_inc_ref(v_toBoundedOrder_3570_);
v_toLattice_3571_ = lean_ctor_get(v_toCompleteLattice_3569_, 0);
lean_inc_ref(v_toLattice_3571_);
v_toOrderTop_3572_ = lean_ctor_get(v_toBoundedOrder_3570_, 0);
v_toOrderBot_3573_ = lean_ctor_get(v_toBoundedOrder_3570_, 1);
v_isSharedCheck_3714_ = !lean_is_exclusive(v_toBoundedOrder_3570_);
if (v_isSharedCheck_3714_ == 0)
{
v___x_3575_ = v_toBoundedOrder_3570_;
v_isShared_3576_ = v_isSharedCheck_3714_;
goto v_resetjp_3574_;
}
else
{
lean_inc(v_toOrderBot_3573_);
lean_inc(v_toOrderTop_3572_);
lean_dec(v_toBoundedOrder_3570_);
v___x_3575_ = lean_box(0);
v_isShared_3576_ = v_isSharedCheck_3714_;
goto v_resetjp_3574_;
}
v_resetjp_3574_:
{
lean_object* v___x_3577_; lean_object* v_toFun_3578_; lean_object* v___x_3579_; lean_object* v_toSupSet_3580_; lean_object* v___x_3582_; uint8_t v_isShared_3583_; uint8_t v_isSharedCheck_3712_; 
lean_inc_ref(v_e_3567_);
v___x_3577_ = lp_mathlib_Equiv_symm___redArg(v_e_3567_);
v_toFun_3578_ = lean_ctor_get(v___x_3577_, 0);
lean_inc(v_toFun_3578_);
lean_dec_ref(v___x_3577_);
lean_inc_ref(v_toCompleteLattice_3569_);
v___x_3579_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_3569_);
v_toSupSet_3580_ = lean_ctor_get(v___x_3579_, 1);
v_isSharedCheck_3712_ = !lean_is_exclusive(v___x_3579_);
if (v_isSharedCheck_3712_ == 0)
{
lean_object* v_unused_3713_; 
v_unused_3713_ = lean_ctor_get(v___x_3579_, 0);
lean_dec(v_unused_3713_);
v___x_3582_ = v___x_3579_;
v_isShared_3583_ = v_isSharedCheck_3712_;
goto v_resetjp_3581_;
}
else
{
lean_inc(v_toSupSet_3580_);
lean_dec(v___x_3579_);
v___x_3582_ = lean_box(0);
v_isShared_3583_ = v_isSharedCheck_3712_;
goto v_resetjp_3581_;
}
v_resetjp_3581_:
{
lean_object* v___x_3584_; lean_object* v_toInfSet_3585_; lean_object* v___x_3587_; uint8_t v_isShared_3588_; uint8_t v_isSharedCheck_3710_; 
lean_inc_ref(v_toCompleteLattice_3569_);
v___x_3584_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_3569_);
v_toInfSet_3585_ = lean_ctor_get(v___x_3584_, 1);
v_isSharedCheck_3710_ = !lean_is_exclusive(v___x_3584_);
if (v_isSharedCheck_3710_ == 0)
{
lean_object* v_unused_3711_; 
v_unused_3711_ = lean_ctor_get(v___x_3584_, 0);
lean_dec(v_unused_3711_);
v___x_3587_ = v___x_3584_;
v_isShared_3588_ = v_isSharedCheck_3710_;
goto v_resetjp_3586_;
}
else
{
lean_inc(v_toInfSet_3585_);
lean_dec(v___x_3584_);
v___x_3587_ = lean_box(0);
v_isShared_3588_ = v_isSharedCheck_3710_;
goto v_resetjp_3586_;
}
v_resetjp_3586_:
{
lean_object* v_toSemilatticeSup_3589_; lean_object* v_inf_3590_; lean_object* v___x_3592_; uint8_t v_isShared_3593_; uint8_t v_isSharedCheck_3709_; 
v_toSemilatticeSup_3589_ = lean_ctor_get(v_toLattice_3571_, 0);
v_inf_3590_ = lean_ctor_get(v_toLattice_3571_, 1);
v_isSharedCheck_3709_ = !lean_is_exclusive(v_toLattice_3571_);
if (v_isSharedCheck_3709_ == 0)
{
v___x_3592_ = v_toLattice_3571_;
v_isShared_3593_ = v_isSharedCheck_3709_;
goto v_resetjp_3591_;
}
else
{
lean_inc(v_inf_3590_);
lean_inc(v_toSemilatticeSup_3589_);
lean_dec(v_toLattice_3571_);
v___x_3592_ = lean_box(0);
v_isShared_3593_ = v_isSharedCheck_3709_;
goto v_resetjp_3591_;
}
v_resetjp_3591_:
{
lean_object* v___f_3594_; lean_object* v_min_3595_; lean_object* v_le_3596_; lean_object* v_lt_3597_; lean_object* v_semilatticeInf_3598_; lean_object* v_toPartialOrder_3599_; lean_object* v___x_3601_; uint8_t v_isShared_3602_; uint8_t v_isSharedCheck_3707_; 
v___f_3594_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_3578_);
lean_inc_ref(v_e_3567_);
v_min_3595_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_3595_, 0, v___f_3594_);
lean_closure_set(v_min_3595_, 1, v_e_3567_);
lean_closure_set(v_min_3595_, 2, v_inf_3590_);
lean_closure_set(v_min_3595_, 3, v_toFun_3578_);
v_le_3596_ = lean_box(0);
v_lt_3597_ = lean_box(0);
lean_inc_ref(v_min_3595_);
v_semilatticeInf_3598_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_3595_, v_le_3596_, v_lt_3597_);
v_toPartialOrder_3599_ = lean_ctor_get(v_semilatticeInf_3598_, 0);
v_isSharedCheck_3707_ = !lean_is_exclusive(v_semilatticeInf_3598_);
if (v_isSharedCheck_3707_ == 0)
{
lean_object* v_unused_3708_; 
v_unused_3708_ = lean_ctor_get(v_semilatticeInf_3598_, 1);
lean_dec(v_unused_3708_);
v___x_3601_ = v_semilatticeInf_3598_;
v_isShared_3602_ = v_isSharedCheck_3707_;
goto v_resetjp_3600_;
}
else
{
lean_inc(v_toPartialOrder_3599_);
lean_dec(v_semilatticeInf_3598_);
v___x_3601_ = lean_box(0);
v_isShared_3602_ = v_isSharedCheck_3707_;
goto v_resetjp_3600_;
}
v_resetjp_3600_:
{
lean_object* v_toLE_3603_; lean_object* v_toLT_3604_; lean_object* v___x_3606_; uint8_t v_isShared_3607_; uint8_t v_isSharedCheck_3706_; 
v_toLE_3603_ = lean_ctor_get(v_toPartialOrder_3599_, 0);
v_toLT_3604_ = lean_ctor_get(v_toPartialOrder_3599_, 1);
v_isSharedCheck_3706_ = !lean_is_exclusive(v_toPartialOrder_3599_);
if (v_isSharedCheck_3706_ == 0)
{
v___x_3606_ = v_toPartialOrder_3599_;
v_isShared_3607_ = v_isSharedCheck_3706_;
goto v_resetjp_3605_;
}
else
{
lean_inc(v_toLT_3604_);
lean_inc(v_toLE_3603_);
lean_dec(v_toPartialOrder_3599_);
v___x_3606_ = lean_box(0);
v_isShared_3607_ = v_isSharedCheck_3706_;
goto v_resetjp_3605_;
}
v_resetjp_3605_:
{
lean_object* v___f_3608_; lean_object* v___f_3609_; lean_object* v___x_3611_; 
v___f_3608_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_3608_, 0, v_min_3595_);
lean_inc(v_toFun_3578_);
lean_inc_ref(v_e_3567_);
v___f_3609_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_3609_, 0, v_toSemilatticeSup_3589_);
lean_closure_set(v___f_3609_, 1, v___f_3594_);
lean_closure_set(v___f_3609_, 2, v_e_3567_);
lean_closure_set(v___f_3609_, 3, v_toFun_3578_);
if (v_isShared_3607_ == 0)
{
v___x_3611_ = v___x_3606_;
goto v_reusejp_3610_;
}
else
{
lean_object* v_reuseFailAlloc_3705_; 
v_reuseFailAlloc_3705_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3705_, 0, v_toLE_3603_);
lean_ctor_set(v_reuseFailAlloc_3705_, 1, v_toLT_3604_);
v___x_3611_ = v_reuseFailAlloc_3705_;
goto v_reusejp_3610_;
}
v_reusejp_3610_:
{
lean_object* v___x_3613_; 
lean_inc_ref(v___f_3609_);
if (v_isShared_3602_ == 0)
{
lean_ctor_set(v___x_3601_, 1, v___f_3609_);
lean_ctor_set(v___x_3601_, 0, v___x_3611_);
v___x_3613_ = v___x_3601_;
goto v_reusejp_3612_;
}
else
{
lean_object* v_reuseFailAlloc_3704_; 
v_reuseFailAlloc_3704_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3704_, 0, v___x_3611_);
lean_ctor_set(v_reuseFailAlloc_3704_, 1, v___f_3609_);
v___x_3613_ = v_reuseFailAlloc_3704_;
goto v_reusejp_3612_;
}
v_reusejp_3612_:
{
lean_object* v_lattice_3615_; 
lean_inc_ref(v___f_3608_);
if (v_isShared_3593_ == 0)
{
lean_ctor_set(v___x_3592_, 1, v___f_3608_);
lean_ctor_set(v___x_3592_, 0, v___x_3613_);
v_lattice_3615_ = v___x_3592_;
goto v_reusejp_3614_;
}
else
{
lean_object* v_reuseFailAlloc_3703_; 
v_reuseFailAlloc_3703_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3703_, 0, v___x_3613_);
lean_ctor_set(v_reuseFailAlloc_3703_, 1, v___f_3608_);
v_lattice_3615_ = v_reuseFailAlloc_3703_;
goto v_reusejp_3614_;
}
v_reusejp_3614_:
{
lean_object* v___x_3616_; lean_object* v_toPartialOrder_3617_; lean_object* v___x_3619_; uint8_t v_isShared_3620_; uint8_t v_isSharedCheck_3701_; 
v___x_3616_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_3615_);
v_toPartialOrder_3617_ = lean_ctor_get(v___x_3616_, 0);
v_isSharedCheck_3701_ = !lean_is_exclusive(v___x_3616_);
if (v_isSharedCheck_3701_ == 0)
{
lean_object* v_unused_3702_; 
v_unused_3702_ = lean_ctor_get(v___x_3616_, 1);
lean_dec(v_unused_3702_);
v___x_3619_ = v___x_3616_;
v_isShared_3620_ = v_isSharedCheck_3701_;
goto v_resetjp_3618_;
}
else
{
lean_inc(v_toPartialOrder_3617_);
lean_dec(v___x_3616_);
v___x_3619_ = lean_box(0);
v_isShared_3620_ = v_isSharedCheck_3701_;
goto v_resetjp_3618_;
}
v_resetjp_3618_:
{
lean_object* v_toLE_3621_; lean_object* v_toLT_3622_; lean_object* v___x_3624_; uint8_t v_isShared_3625_; uint8_t v_isSharedCheck_3700_; 
v_toLE_3621_ = lean_ctor_get(v_toPartialOrder_3617_, 0);
v_toLT_3622_ = lean_ctor_get(v_toPartialOrder_3617_, 1);
v_isSharedCheck_3700_ = !lean_is_exclusive(v_toPartialOrder_3617_);
if (v_isSharedCheck_3700_ == 0)
{
v___x_3624_ = v_toPartialOrder_3617_;
v_isShared_3625_ = v_isSharedCheck_3700_;
goto v_resetjp_3623_;
}
else
{
lean_inc(v_toLT_3622_);
lean_inc(v_toLE_3621_);
lean_dec(v_toPartialOrder_3617_);
v___x_3624_ = lean_box(0);
v_isShared_3625_ = v_isSharedCheck_3700_;
goto v_resetjp_3623_;
}
v_resetjp_3623_:
{
lean_object* v_top_3626_; lean_object* v_bot_3627_; lean_object* v_supSet_3628_; lean_object* v_infSet_3629_; lean_object* v___f_3630_; lean_object* v___x_3632_; 
lean_inc_n(v_toFun_3578_, 4);
v_top_3626_ = lean_apply_1(v_toFun_3578_, v_toOrderTop_3572_);
v_bot_3627_ = lean_apply_1(v_toFun_3578_, v_toOrderBot_3573_);
v_supSet_3628_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_3628_, 0, v_toSupSet_3580_);
lean_closure_set(v_supSet_3628_, 1, v_toFun_3578_);
v_infSet_3629_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_3629_, 0, v_toInfSet_3585_);
lean_closure_set(v_infSet_3629_, 1, v_toFun_3578_);
v___f_3630_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_3630_, 0, v___f_3609_);
if (v_isShared_3625_ == 0)
{
v___x_3632_ = v___x_3624_;
goto v_reusejp_3631_;
}
else
{
lean_object* v_reuseFailAlloc_3699_; 
v_reuseFailAlloc_3699_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3699_, 0, v_toLE_3621_);
lean_ctor_set(v_reuseFailAlloc_3699_, 1, v_toLT_3622_);
v___x_3632_ = v_reuseFailAlloc_3699_;
goto v_reusejp_3631_;
}
v_reusejp_3631_:
{
lean_object* v___x_3634_; 
lean_inc_ref(v___f_3630_);
if (v_isShared_3620_ == 0)
{
lean_ctor_set(v___x_3619_, 1, v___f_3630_);
lean_ctor_set(v___x_3619_, 0, v___x_3632_);
v___x_3634_ = v___x_3619_;
goto v_reusejp_3633_;
}
else
{
lean_object* v_reuseFailAlloc_3698_; 
v_reuseFailAlloc_3698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3698_, 0, v___x_3632_);
lean_ctor_set(v_reuseFailAlloc_3698_, 1, v___f_3630_);
v___x_3634_ = v_reuseFailAlloc_3698_;
goto v_reusejp_3633_;
}
v_reusejp_3633_:
{
lean_object* v___x_3636_; 
lean_inc_ref(v___f_3608_);
if (v_isShared_3588_ == 0)
{
lean_ctor_set(v___x_3587_, 1, v___f_3608_);
lean_ctor_set(v___x_3587_, 0, v___x_3634_);
v___x_3636_ = v___x_3587_;
goto v_reusejp_3635_;
}
else
{
lean_object* v_reuseFailAlloc_3697_; 
v_reuseFailAlloc_3697_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3697_, 0, v___x_3634_);
lean_ctor_set(v_reuseFailAlloc_3697_, 1, v___f_3608_);
v___x_3636_ = v_reuseFailAlloc_3697_;
goto v_reusejp_3635_;
}
v_reusejp_3635_:
{
lean_object* v___x_3638_; 
if (v_isShared_3576_ == 0)
{
lean_ctor_set(v___x_3575_, 1, v_bot_3627_);
lean_ctor_set(v___x_3575_, 0, v_top_3626_);
v___x_3638_ = v___x_3575_;
goto v_reusejp_3637_;
}
else
{
lean_object* v_reuseFailAlloc_3696_; 
v_reuseFailAlloc_3696_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3696_, 0, v_top_3626_);
lean_ctor_set(v_reuseFailAlloc_3696_, 1, v_bot_3627_);
v___x_3638_ = v_reuseFailAlloc_3696_;
goto v_reusejp_3637_;
}
v_reusejp_3637_:
{
lean_object* v_completeLattice_3639_; lean_object* v___x_3640_; lean_object* v___x_3642_; uint8_t v_isShared_3643_; uint8_t v_isSharedCheck_3691_; 
v_completeLattice_3639_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_3639_, 0, v___x_3636_);
lean_ctor_set(v_completeLattice_3639_, 1, v_supSet_3628_);
lean_ctor_set(v_completeLattice_3639_, 2, v_infSet_3629_);
lean_ctor_set(v_completeLattice_3639_, 3, v___x_3638_);
v___x_3640_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_3568_);
v_isSharedCheck_3691_ = !lean_is_exclusive(v_inst_3568_);
if (v_isSharedCheck_3691_ == 0)
{
lean_object* v_unused_3692_; lean_object* v_unused_3693_; lean_object* v_unused_3694_; lean_object* v_unused_3695_; 
v_unused_3692_ = lean_ctor_get(v_inst_3568_, 3);
lean_dec(v_unused_3692_);
v_unused_3693_ = lean_ctor_get(v_inst_3568_, 2);
lean_dec(v_unused_3693_);
v_unused_3694_ = lean_ctor_get(v_inst_3568_, 1);
lean_dec(v_unused_3694_);
v_unused_3695_ = lean_ctor_get(v_inst_3568_, 0);
lean_dec(v_unused_3695_);
v___x_3642_ = v_inst_3568_;
v_isShared_3643_ = v_isSharedCheck_3691_;
goto v_resetjp_3641_;
}
else
{
lean_dec(v_inst_3568_);
v___x_3642_ = lean_box(0);
v_isShared_3643_ = v_isSharedCheck_3691_;
goto v_resetjp_3641_;
}
v_resetjp_3641_:
{
lean_object* v_toCompl_3644_; lean_object* v_toHImp_3645_; lean_object* v_toTop_3646_; lean_object* v___x_3647_; lean_object* v_toSDiff_3648_; lean_object* v_toBot_3649_; lean_object* v___x_3650_; lean_object* v_toPartialOrder_3651_; lean_object* v_toInfSet_3652_; lean_object* v___x_3654_; uint8_t v_isShared_3655_; uint8_t v_isSharedCheck_3690_; 
v_toCompl_3644_ = lean_ctor_get(v___x_3640_, 1);
lean_inc(v_toCompl_3644_);
v_toHImp_3645_ = lean_ctor_get(v___x_3640_, 3);
lean_inc(v_toHImp_3645_);
v_toTop_3646_ = lean_ctor_get(v___x_3640_, 4);
lean_inc(v_toTop_3646_);
v___x_3647_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(v___x_3640_);
lean_dec_ref(v___x_3640_);
v_toSDiff_3648_ = lean_ctor_get(v___x_3647_, 1);
lean_inc(v_toSDiff_3648_);
v_toBot_3649_ = lean_ctor_get(v___x_3647_, 2);
lean_inc(v_toBot_3649_);
lean_dec_ref(v___x_3647_);
lean_inc_ref(v_completeLattice_3639_);
v___x_3650_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_3639_);
v_toPartialOrder_3651_ = lean_ctor_get(v___x_3650_, 0);
v_toInfSet_3652_ = lean_ctor_get(v___x_3650_, 1);
v_isSharedCheck_3690_ = !lean_is_exclusive(v___x_3650_);
if (v_isSharedCheck_3690_ == 0)
{
v___x_3654_ = v___x_3650_;
v_isShared_3655_ = v_isSharedCheck_3690_;
goto v_resetjp_3653_;
}
else
{
lean_inc(v_toInfSet_3652_);
lean_inc(v_toPartialOrder_3651_);
lean_dec(v___x_3650_);
v___x_3654_ = lean_box(0);
v_isShared_3655_ = v_isSharedCheck_3690_;
goto v_resetjp_3653_;
}
v_resetjp_3653_:
{
lean_object* v_toLE_3656_; lean_object* v_toLT_3657_; lean_object* v___x_3659_; uint8_t v_isShared_3660_; uint8_t v_isSharedCheck_3689_; 
v_toLE_3656_ = lean_ctor_get(v_toPartialOrder_3651_, 0);
v_toLT_3657_ = lean_ctor_get(v_toPartialOrder_3651_, 1);
v_isSharedCheck_3689_ = !lean_is_exclusive(v_toPartialOrder_3651_);
if (v_isSharedCheck_3689_ == 0)
{
v___x_3659_ = v_toPartialOrder_3651_;
v_isShared_3660_ = v_isSharedCheck_3689_;
goto v_resetjp_3658_;
}
else
{
lean_inc(v_toLT_3657_);
lean_inc(v_toLE_3656_);
lean_dec(v_toPartialOrder_3651_);
v___x_3659_ = lean_box(0);
v_isShared_3660_ = v_isSharedCheck_3689_;
goto v_resetjp_3658_;
}
v_resetjp_3658_:
{
lean_object* v___x_3661_; lean_object* v_toSupSet_3662_; lean_object* v___x_3664_; uint8_t v_isShared_3665_; uint8_t v_isSharedCheck_3687_; 
v___x_3661_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_3639_);
v_toSupSet_3662_ = lean_ctor_get(v___x_3661_, 1);
v_isSharedCheck_3687_ = !lean_is_exclusive(v___x_3661_);
if (v_isSharedCheck_3687_ == 0)
{
lean_object* v_unused_3688_; 
v_unused_3688_ = lean_ctor_get(v___x_3661_, 0);
lean_dec(v_unused_3688_);
v___x_3664_ = v___x_3661_;
v_isShared_3665_ = v_isSharedCheck_3687_;
goto v_resetjp_3663_;
}
else
{
lean_inc(v_toSupSet_3662_);
lean_dec(v___x_3661_);
v___x_3664_ = lean_box(0);
v_isShared_3665_ = v_isSharedCheck_3687_;
goto v_resetjp_3663_;
}
v_resetjp_3663_:
{
lean_object* v_top_3666_; lean_object* v_himp_3667_; lean_object* v_compl_3668_; lean_object* v_sdiff_3669_; lean_object* v_bot_3670_; lean_object* v___x_3672_; 
lean_inc_n(v_toFun_3578_, 4);
v_top_3666_ = lean_apply_1(v_toFun_3578_, v_toTop_3646_);
lean_inc_ref_n(v_e_3567_, 2);
v_himp_3667_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_3667_, 0, v___f_3594_);
lean_closure_set(v_himp_3667_, 1, v_e_3567_);
lean_closure_set(v_himp_3667_, 2, v_toHImp_3645_);
lean_closure_set(v_himp_3667_, 3, v_toFun_3578_);
v_compl_3668_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_3668_, 0, v_e_3567_);
lean_closure_set(v_compl_3668_, 1, v_toCompl_3644_);
lean_closure_set(v_compl_3668_, 2, v_toFun_3578_);
v_sdiff_3669_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_3669_, 0, v___f_3594_);
lean_closure_set(v_sdiff_3669_, 1, v_e_3567_);
lean_closure_set(v_sdiff_3669_, 2, v_toSDiff_3648_);
lean_closure_set(v_sdiff_3669_, 3, v_toFun_3578_);
v_bot_3670_ = lean_apply_1(v_toFun_3578_, v_toBot_3649_);
if (v_isShared_3660_ == 0)
{
v___x_3672_ = v___x_3659_;
goto v_reusejp_3671_;
}
else
{
lean_object* v_reuseFailAlloc_3686_; 
v_reuseFailAlloc_3686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3686_, 0, v_toLE_3656_);
lean_ctor_set(v_reuseFailAlloc_3686_, 1, v_toLT_3657_);
v___x_3672_ = v_reuseFailAlloc_3686_;
goto v_reusejp_3671_;
}
v_reusejp_3671_:
{
lean_object* v___x_3674_; 
if (v_isShared_3665_ == 0)
{
lean_ctor_set(v___x_3664_, 1, v___f_3630_);
lean_ctor_set(v___x_3664_, 0, v___x_3672_);
v___x_3674_ = v___x_3664_;
goto v_reusejp_3673_;
}
else
{
lean_object* v_reuseFailAlloc_3685_; 
v_reuseFailAlloc_3685_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3685_, 0, v___x_3672_);
lean_ctor_set(v_reuseFailAlloc_3685_, 1, v___f_3630_);
v___x_3674_ = v_reuseFailAlloc_3685_;
goto v_reusejp_3673_;
}
v_reusejp_3673_:
{
lean_object* v___x_3676_; 
if (v_isShared_3655_ == 0)
{
lean_ctor_set(v___x_3654_, 1, v___f_3608_);
lean_ctor_set(v___x_3654_, 0, v___x_3674_);
v___x_3676_ = v___x_3654_;
goto v_reusejp_3675_;
}
else
{
lean_object* v_reuseFailAlloc_3684_; 
v_reuseFailAlloc_3684_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3684_, 0, v___x_3674_);
lean_ctor_set(v_reuseFailAlloc_3684_, 1, v___f_3608_);
v___x_3676_ = v_reuseFailAlloc_3684_;
goto v_reusejp_3675_;
}
v_reusejp_3675_:
{
lean_object* v___x_3678_; 
if (v_isShared_3583_ == 0)
{
lean_ctor_set(v___x_3582_, 1, v_bot_3670_);
lean_ctor_set(v___x_3582_, 0, v_top_3666_);
v___x_3678_ = v___x_3582_;
goto v_reusejp_3677_;
}
else
{
lean_object* v_reuseFailAlloc_3683_; 
v_reuseFailAlloc_3683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3683_, 0, v_top_3666_);
lean_ctor_set(v_reuseFailAlloc_3683_, 1, v_bot_3670_);
v___x_3678_ = v_reuseFailAlloc_3683_;
goto v_reusejp_3677_;
}
v_reusejp_3677_:
{
lean_object* v___x_3679_; lean_object* v___x_3681_; 
v___x_3679_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3679_, 0, v___x_3676_);
lean_ctor_set(v___x_3679_, 1, v_toSupSet_3662_);
lean_ctor_set(v___x_3679_, 2, v_toInfSet_3652_);
lean_ctor_set(v___x_3679_, 3, v___x_3678_);
if (v_isShared_3643_ == 0)
{
lean_ctor_set(v___x_3642_, 3, v_himp_3667_);
lean_ctor_set(v___x_3642_, 2, v_sdiff_3669_);
lean_ctor_set(v___x_3642_, 1, v_compl_3668_);
lean_ctor_set(v___x_3642_, 0, v___x_3679_);
v___x_3681_ = v___x_3642_;
goto v_reusejp_3680_;
}
else
{
lean_object* v_reuseFailAlloc_3682_; 
v_reuseFailAlloc_3682_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3682_, 0, v___x_3679_);
lean_ctor_set(v_reuseFailAlloc_3682_, 1, v_compl_3668_);
lean_ctor_set(v_reuseFailAlloc_3682_, 2, v_sdiff_3669_);
lean_ctor_set(v_reuseFailAlloc_3682_, 3, v_himp_3667_);
v___x_3681_ = v_reuseFailAlloc_3682_;
goto v_reusejp_3680_;
}
v_reusejp_3680_:
{
return v___x_3681_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeBooleanAlgebra(lean_object* v_00_u03b1_3715_, lean_object* v_00_u03b2_3716_, lean_object* v_e_3717_, lean_object* v_inst_3718_){
_start:
{
lean_object* v_toCompleteLattice_3719_; lean_object* v_toBoundedOrder_3720_; lean_object* v_toLattice_3721_; lean_object* v_toOrderTop_3722_; lean_object* v_toOrderBot_3723_; lean_object* v___x_3725_; uint8_t v_isShared_3726_; uint8_t v_isSharedCheck_3864_; 
v_toCompleteLattice_3719_ = lean_ctor_get(v_inst_3718_, 0);
v_toBoundedOrder_3720_ = lean_ctor_get(v_toCompleteLattice_3719_, 3);
lean_inc_ref(v_toBoundedOrder_3720_);
v_toLattice_3721_ = lean_ctor_get(v_toCompleteLattice_3719_, 0);
lean_inc_ref(v_toLattice_3721_);
v_toOrderTop_3722_ = lean_ctor_get(v_toBoundedOrder_3720_, 0);
v_toOrderBot_3723_ = lean_ctor_get(v_toBoundedOrder_3720_, 1);
v_isSharedCheck_3864_ = !lean_is_exclusive(v_toBoundedOrder_3720_);
if (v_isSharedCheck_3864_ == 0)
{
v___x_3725_ = v_toBoundedOrder_3720_;
v_isShared_3726_ = v_isSharedCheck_3864_;
goto v_resetjp_3724_;
}
else
{
lean_inc(v_toOrderBot_3723_);
lean_inc(v_toOrderTop_3722_);
lean_dec(v_toBoundedOrder_3720_);
v___x_3725_ = lean_box(0);
v_isShared_3726_ = v_isSharedCheck_3864_;
goto v_resetjp_3724_;
}
v_resetjp_3724_:
{
lean_object* v___x_3727_; lean_object* v_toFun_3728_; lean_object* v___x_3729_; lean_object* v_toSupSet_3730_; lean_object* v___x_3732_; uint8_t v_isShared_3733_; uint8_t v_isSharedCheck_3862_; 
lean_inc_ref(v_e_3717_);
v___x_3727_ = lp_mathlib_Equiv_symm___redArg(v_e_3717_);
v_toFun_3728_ = lean_ctor_get(v___x_3727_, 0);
lean_inc(v_toFun_3728_);
lean_dec_ref(v___x_3727_);
lean_inc_ref(v_toCompleteLattice_3719_);
v___x_3729_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_3719_);
v_toSupSet_3730_ = lean_ctor_get(v___x_3729_, 1);
v_isSharedCheck_3862_ = !lean_is_exclusive(v___x_3729_);
if (v_isSharedCheck_3862_ == 0)
{
lean_object* v_unused_3863_; 
v_unused_3863_ = lean_ctor_get(v___x_3729_, 0);
lean_dec(v_unused_3863_);
v___x_3732_ = v___x_3729_;
v_isShared_3733_ = v_isSharedCheck_3862_;
goto v_resetjp_3731_;
}
else
{
lean_inc(v_toSupSet_3730_);
lean_dec(v___x_3729_);
v___x_3732_ = lean_box(0);
v_isShared_3733_ = v_isSharedCheck_3862_;
goto v_resetjp_3731_;
}
v_resetjp_3731_:
{
lean_object* v___x_3734_; lean_object* v_toInfSet_3735_; lean_object* v___x_3737_; uint8_t v_isShared_3738_; uint8_t v_isSharedCheck_3860_; 
lean_inc_ref(v_toCompleteLattice_3719_);
v___x_3734_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_3719_);
v_toInfSet_3735_ = lean_ctor_get(v___x_3734_, 1);
v_isSharedCheck_3860_ = !lean_is_exclusive(v___x_3734_);
if (v_isSharedCheck_3860_ == 0)
{
lean_object* v_unused_3861_; 
v_unused_3861_ = lean_ctor_get(v___x_3734_, 0);
lean_dec(v_unused_3861_);
v___x_3737_ = v___x_3734_;
v_isShared_3738_ = v_isSharedCheck_3860_;
goto v_resetjp_3736_;
}
else
{
lean_inc(v_toInfSet_3735_);
lean_dec(v___x_3734_);
v___x_3737_ = lean_box(0);
v_isShared_3738_ = v_isSharedCheck_3860_;
goto v_resetjp_3736_;
}
v_resetjp_3736_:
{
lean_object* v_toSemilatticeSup_3739_; lean_object* v_inf_3740_; lean_object* v___x_3742_; uint8_t v_isShared_3743_; uint8_t v_isSharedCheck_3859_; 
v_toSemilatticeSup_3739_ = lean_ctor_get(v_toLattice_3721_, 0);
v_inf_3740_ = lean_ctor_get(v_toLattice_3721_, 1);
v_isSharedCheck_3859_ = !lean_is_exclusive(v_toLattice_3721_);
if (v_isSharedCheck_3859_ == 0)
{
v___x_3742_ = v_toLattice_3721_;
v_isShared_3743_ = v_isSharedCheck_3859_;
goto v_resetjp_3741_;
}
else
{
lean_inc(v_inf_3740_);
lean_inc(v_toSemilatticeSup_3739_);
lean_dec(v_toLattice_3721_);
v___x_3742_ = lean_box(0);
v_isShared_3743_ = v_isSharedCheck_3859_;
goto v_resetjp_3741_;
}
v_resetjp_3741_:
{
lean_object* v___f_3744_; lean_object* v_min_3745_; lean_object* v_le_3746_; lean_object* v_lt_3747_; lean_object* v_semilatticeInf_3748_; lean_object* v_toPartialOrder_3749_; lean_object* v___x_3751_; uint8_t v_isShared_3752_; uint8_t v_isSharedCheck_3857_; 
v___f_3744_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_3728_);
lean_inc_ref(v_e_3717_);
v_min_3745_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_3745_, 0, v___f_3744_);
lean_closure_set(v_min_3745_, 1, v_e_3717_);
lean_closure_set(v_min_3745_, 2, v_inf_3740_);
lean_closure_set(v_min_3745_, 3, v_toFun_3728_);
v_le_3746_ = lean_box(0);
v_lt_3747_ = lean_box(0);
lean_inc_ref(v_min_3745_);
v_semilatticeInf_3748_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_3745_, v_le_3746_, v_lt_3747_);
v_toPartialOrder_3749_ = lean_ctor_get(v_semilatticeInf_3748_, 0);
v_isSharedCheck_3857_ = !lean_is_exclusive(v_semilatticeInf_3748_);
if (v_isSharedCheck_3857_ == 0)
{
lean_object* v_unused_3858_; 
v_unused_3858_ = lean_ctor_get(v_semilatticeInf_3748_, 1);
lean_dec(v_unused_3858_);
v___x_3751_ = v_semilatticeInf_3748_;
v_isShared_3752_ = v_isSharedCheck_3857_;
goto v_resetjp_3750_;
}
else
{
lean_inc(v_toPartialOrder_3749_);
lean_dec(v_semilatticeInf_3748_);
v___x_3751_ = lean_box(0);
v_isShared_3752_ = v_isSharedCheck_3857_;
goto v_resetjp_3750_;
}
v_resetjp_3750_:
{
lean_object* v_toLE_3753_; lean_object* v_toLT_3754_; lean_object* v___x_3756_; uint8_t v_isShared_3757_; uint8_t v_isSharedCheck_3856_; 
v_toLE_3753_ = lean_ctor_get(v_toPartialOrder_3749_, 0);
v_toLT_3754_ = lean_ctor_get(v_toPartialOrder_3749_, 1);
v_isSharedCheck_3856_ = !lean_is_exclusive(v_toPartialOrder_3749_);
if (v_isSharedCheck_3856_ == 0)
{
v___x_3756_ = v_toPartialOrder_3749_;
v_isShared_3757_ = v_isSharedCheck_3856_;
goto v_resetjp_3755_;
}
else
{
lean_inc(v_toLT_3754_);
lean_inc(v_toLE_3753_);
lean_dec(v_toPartialOrder_3749_);
v___x_3756_ = lean_box(0);
v_isShared_3757_ = v_isSharedCheck_3856_;
goto v_resetjp_3755_;
}
v_resetjp_3755_:
{
lean_object* v___f_3758_; lean_object* v___f_3759_; lean_object* v___x_3761_; 
v___f_3758_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_3758_, 0, v_min_3745_);
lean_inc(v_toFun_3728_);
lean_inc_ref(v_e_3717_);
v___f_3759_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_3759_, 0, v_toSemilatticeSup_3739_);
lean_closure_set(v___f_3759_, 1, v___f_3744_);
lean_closure_set(v___f_3759_, 2, v_e_3717_);
lean_closure_set(v___f_3759_, 3, v_toFun_3728_);
if (v_isShared_3757_ == 0)
{
v___x_3761_ = v___x_3756_;
goto v_reusejp_3760_;
}
else
{
lean_object* v_reuseFailAlloc_3855_; 
v_reuseFailAlloc_3855_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3855_, 0, v_toLE_3753_);
lean_ctor_set(v_reuseFailAlloc_3855_, 1, v_toLT_3754_);
v___x_3761_ = v_reuseFailAlloc_3855_;
goto v_reusejp_3760_;
}
v_reusejp_3760_:
{
lean_object* v___x_3763_; 
lean_inc_ref(v___f_3759_);
if (v_isShared_3752_ == 0)
{
lean_ctor_set(v___x_3751_, 1, v___f_3759_);
lean_ctor_set(v___x_3751_, 0, v___x_3761_);
v___x_3763_ = v___x_3751_;
goto v_reusejp_3762_;
}
else
{
lean_object* v_reuseFailAlloc_3854_; 
v_reuseFailAlloc_3854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3854_, 0, v___x_3761_);
lean_ctor_set(v_reuseFailAlloc_3854_, 1, v___f_3759_);
v___x_3763_ = v_reuseFailAlloc_3854_;
goto v_reusejp_3762_;
}
v_reusejp_3762_:
{
lean_object* v_lattice_3765_; 
lean_inc_ref(v___f_3758_);
if (v_isShared_3743_ == 0)
{
lean_ctor_set(v___x_3742_, 1, v___f_3758_);
lean_ctor_set(v___x_3742_, 0, v___x_3763_);
v_lattice_3765_ = v___x_3742_;
goto v_reusejp_3764_;
}
else
{
lean_object* v_reuseFailAlloc_3853_; 
v_reuseFailAlloc_3853_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3853_, 0, v___x_3763_);
lean_ctor_set(v_reuseFailAlloc_3853_, 1, v___f_3758_);
v_lattice_3765_ = v_reuseFailAlloc_3853_;
goto v_reusejp_3764_;
}
v_reusejp_3764_:
{
lean_object* v___x_3766_; lean_object* v_toPartialOrder_3767_; lean_object* v___x_3769_; uint8_t v_isShared_3770_; uint8_t v_isSharedCheck_3851_; 
v___x_3766_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_3765_);
v_toPartialOrder_3767_ = lean_ctor_get(v___x_3766_, 0);
v_isSharedCheck_3851_ = !lean_is_exclusive(v___x_3766_);
if (v_isSharedCheck_3851_ == 0)
{
lean_object* v_unused_3852_; 
v_unused_3852_ = lean_ctor_get(v___x_3766_, 1);
lean_dec(v_unused_3852_);
v___x_3769_ = v___x_3766_;
v_isShared_3770_ = v_isSharedCheck_3851_;
goto v_resetjp_3768_;
}
else
{
lean_inc(v_toPartialOrder_3767_);
lean_dec(v___x_3766_);
v___x_3769_ = lean_box(0);
v_isShared_3770_ = v_isSharedCheck_3851_;
goto v_resetjp_3768_;
}
v_resetjp_3768_:
{
lean_object* v_toLE_3771_; lean_object* v_toLT_3772_; lean_object* v___x_3774_; uint8_t v_isShared_3775_; uint8_t v_isSharedCheck_3850_; 
v_toLE_3771_ = lean_ctor_get(v_toPartialOrder_3767_, 0);
v_toLT_3772_ = lean_ctor_get(v_toPartialOrder_3767_, 1);
v_isSharedCheck_3850_ = !lean_is_exclusive(v_toPartialOrder_3767_);
if (v_isSharedCheck_3850_ == 0)
{
v___x_3774_ = v_toPartialOrder_3767_;
v_isShared_3775_ = v_isSharedCheck_3850_;
goto v_resetjp_3773_;
}
else
{
lean_inc(v_toLT_3772_);
lean_inc(v_toLE_3771_);
lean_dec(v_toPartialOrder_3767_);
v___x_3774_ = lean_box(0);
v_isShared_3775_ = v_isSharedCheck_3850_;
goto v_resetjp_3773_;
}
v_resetjp_3773_:
{
lean_object* v_top_3776_; lean_object* v_bot_3777_; lean_object* v_supSet_3778_; lean_object* v_infSet_3779_; lean_object* v___f_3780_; lean_object* v___x_3782_; 
lean_inc_n(v_toFun_3728_, 4);
v_top_3776_ = lean_apply_1(v_toFun_3728_, v_toOrderTop_3722_);
v_bot_3777_ = lean_apply_1(v_toFun_3728_, v_toOrderBot_3723_);
v_supSet_3778_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_3778_, 0, v_toSupSet_3730_);
lean_closure_set(v_supSet_3778_, 1, v_toFun_3728_);
v_infSet_3779_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_3779_, 0, v_toInfSet_3735_);
lean_closure_set(v_infSet_3779_, 1, v_toFun_3728_);
v___f_3780_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_3780_, 0, v___f_3759_);
if (v_isShared_3775_ == 0)
{
v___x_3782_ = v___x_3774_;
goto v_reusejp_3781_;
}
else
{
lean_object* v_reuseFailAlloc_3849_; 
v_reuseFailAlloc_3849_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3849_, 0, v_toLE_3771_);
lean_ctor_set(v_reuseFailAlloc_3849_, 1, v_toLT_3772_);
v___x_3782_ = v_reuseFailAlloc_3849_;
goto v_reusejp_3781_;
}
v_reusejp_3781_:
{
lean_object* v___x_3784_; 
lean_inc_ref(v___f_3780_);
if (v_isShared_3770_ == 0)
{
lean_ctor_set(v___x_3769_, 1, v___f_3780_);
lean_ctor_set(v___x_3769_, 0, v___x_3782_);
v___x_3784_ = v___x_3769_;
goto v_reusejp_3783_;
}
else
{
lean_object* v_reuseFailAlloc_3848_; 
v_reuseFailAlloc_3848_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3848_, 0, v___x_3782_);
lean_ctor_set(v_reuseFailAlloc_3848_, 1, v___f_3780_);
v___x_3784_ = v_reuseFailAlloc_3848_;
goto v_reusejp_3783_;
}
v_reusejp_3783_:
{
lean_object* v___x_3786_; 
lean_inc_ref(v___f_3758_);
if (v_isShared_3738_ == 0)
{
lean_ctor_set(v___x_3737_, 1, v___f_3758_);
lean_ctor_set(v___x_3737_, 0, v___x_3784_);
v___x_3786_ = v___x_3737_;
goto v_reusejp_3785_;
}
else
{
lean_object* v_reuseFailAlloc_3847_; 
v_reuseFailAlloc_3847_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3847_, 0, v___x_3784_);
lean_ctor_set(v_reuseFailAlloc_3847_, 1, v___f_3758_);
v___x_3786_ = v_reuseFailAlloc_3847_;
goto v_reusejp_3785_;
}
v_reusejp_3785_:
{
lean_object* v___x_3788_; 
if (v_isShared_3726_ == 0)
{
lean_ctor_set(v___x_3725_, 1, v_bot_3777_);
lean_ctor_set(v___x_3725_, 0, v_top_3776_);
v___x_3788_ = v___x_3725_;
goto v_reusejp_3787_;
}
else
{
lean_object* v_reuseFailAlloc_3846_; 
v_reuseFailAlloc_3846_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3846_, 0, v_top_3776_);
lean_ctor_set(v_reuseFailAlloc_3846_, 1, v_bot_3777_);
v___x_3788_ = v_reuseFailAlloc_3846_;
goto v_reusejp_3787_;
}
v_reusejp_3787_:
{
lean_object* v_completeLattice_3789_; lean_object* v___x_3790_; lean_object* v___x_3792_; uint8_t v_isShared_3793_; uint8_t v_isSharedCheck_3841_; 
v_completeLattice_3789_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_3789_, 0, v___x_3786_);
lean_ctor_set(v_completeLattice_3789_, 1, v_supSet_3778_);
lean_ctor_set(v_completeLattice_3789_, 2, v_infSet_3779_);
lean_ctor_set(v_completeLattice_3789_, 3, v___x_3788_);
v___x_3790_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_3718_);
v_isSharedCheck_3841_ = !lean_is_exclusive(v_inst_3718_);
if (v_isSharedCheck_3841_ == 0)
{
lean_object* v_unused_3842_; lean_object* v_unused_3843_; lean_object* v_unused_3844_; lean_object* v_unused_3845_; 
v_unused_3842_ = lean_ctor_get(v_inst_3718_, 3);
lean_dec(v_unused_3842_);
v_unused_3843_ = lean_ctor_get(v_inst_3718_, 2);
lean_dec(v_unused_3843_);
v_unused_3844_ = lean_ctor_get(v_inst_3718_, 1);
lean_dec(v_unused_3844_);
v_unused_3845_ = lean_ctor_get(v_inst_3718_, 0);
lean_dec(v_unused_3845_);
v___x_3792_ = v_inst_3718_;
v_isShared_3793_ = v_isSharedCheck_3841_;
goto v_resetjp_3791_;
}
else
{
lean_dec(v_inst_3718_);
v___x_3792_ = lean_box(0);
v_isShared_3793_ = v_isSharedCheck_3841_;
goto v_resetjp_3791_;
}
v_resetjp_3791_:
{
lean_object* v_toCompl_3794_; lean_object* v_toHImp_3795_; lean_object* v_toTop_3796_; lean_object* v___x_3797_; lean_object* v_toSDiff_3798_; lean_object* v_toBot_3799_; lean_object* v___x_3800_; lean_object* v_toPartialOrder_3801_; lean_object* v_toInfSet_3802_; lean_object* v___x_3804_; uint8_t v_isShared_3805_; uint8_t v_isSharedCheck_3840_; 
v_toCompl_3794_ = lean_ctor_get(v___x_3790_, 1);
lean_inc(v_toCompl_3794_);
v_toHImp_3795_ = lean_ctor_get(v___x_3790_, 3);
lean_inc(v_toHImp_3795_);
v_toTop_3796_ = lean_ctor_get(v___x_3790_, 4);
lean_inc(v_toTop_3796_);
v___x_3797_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(v___x_3790_);
lean_dec_ref(v___x_3790_);
v_toSDiff_3798_ = lean_ctor_get(v___x_3797_, 1);
lean_inc(v_toSDiff_3798_);
v_toBot_3799_ = lean_ctor_get(v___x_3797_, 2);
lean_inc(v_toBot_3799_);
lean_dec_ref(v___x_3797_);
lean_inc_ref(v_completeLattice_3789_);
v___x_3800_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_3789_);
v_toPartialOrder_3801_ = lean_ctor_get(v___x_3800_, 0);
v_toInfSet_3802_ = lean_ctor_get(v___x_3800_, 1);
v_isSharedCheck_3840_ = !lean_is_exclusive(v___x_3800_);
if (v_isSharedCheck_3840_ == 0)
{
v___x_3804_ = v___x_3800_;
v_isShared_3805_ = v_isSharedCheck_3840_;
goto v_resetjp_3803_;
}
else
{
lean_inc(v_toInfSet_3802_);
lean_inc(v_toPartialOrder_3801_);
lean_dec(v___x_3800_);
v___x_3804_ = lean_box(0);
v_isShared_3805_ = v_isSharedCheck_3840_;
goto v_resetjp_3803_;
}
v_resetjp_3803_:
{
lean_object* v_toLE_3806_; lean_object* v_toLT_3807_; lean_object* v___x_3809_; uint8_t v_isShared_3810_; uint8_t v_isSharedCheck_3839_; 
v_toLE_3806_ = lean_ctor_get(v_toPartialOrder_3801_, 0);
v_toLT_3807_ = lean_ctor_get(v_toPartialOrder_3801_, 1);
v_isSharedCheck_3839_ = !lean_is_exclusive(v_toPartialOrder_3801_);
if (v_isSharedCheck_3839_ == 0)
{
v___x_3809_ = v_toPartialOrder_3801_;
v_isShared_3810_ = v_isSharedCheck_3839_;
goto v_resetjp_3808_;
}
else
{
lean_inc(v_toLT_3807_);
lean_inc(v_toLE_3806_);
lean_dec(v_toPartialOrder_3801_);
v___x_3809_ = lean_box(0);
v_isShared_3810_ = v_isSharedCheck_3839_;
goto v_resetjp_3808_;
}
v_resetjp_3808_:
{
lean_object* v___x_3811_; lean_object* v_toSupSet_3812_; lean_object* v___x_3814_; uint8_t v_isShared_3815_; uint8_t v_isSharedCheck_3837_; 
v___x_3811_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_3789_);
v_toSupSet_3812_ = lean_ctor_get(v___x_3811_, 1);
v_isSharedCheck_3837_ = !lean_is_exclusive(v___x_3811_);
if (v_isSharedCheck_3837_ == 0)
{
lean_object* v_unused_3838_; 
v_unused_3838_ = lean_ctor_get(v___x_3811_, 0);
lean_dec(v_unused_3838_);
v___x_3814_ = v___x_3811_;
v_isShared_3815_ = v_isSharedCheck_3837_;
goto v_resetjp_3813_;
}
else
{
lean_inc(v_toSupSet_3812_);
lean_dec(v___x_3811_);
v___x_3814_ = lean_box(0);
v_isShared_3815_ = v_isSharedCheck_3837_;
goto v_resetjp_3813_;
}
v_resetjp_3813_:
{
lean_object* v_top_3816_; lean_object* v_himp_3817_; lean_object* v_compl_3818_; lean_object* v_sdiff_3819_; lean_object* v_bot_3820_; lean_object* v___x_3822_; 
lean_inc_n(v_toFun_3728_, 4);
v_top_3816_ = lean_apply_1(v_toFun_3728_, v_toTop_3796_);
lean_inc_ref_n(v_e_3717_, 2);
v_himp_3817_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_3817_, 0, v___f_3744_);
lean_closure_set(v_himp_3817_, 1, v_e_3717_);
lean_closure_set(v_himp_3817_, 2, v_toHImp_3795_);
lean_closure_set(v_himp_3817_, 3, v_toFun_3728_);
v_compl_3818_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_3818_, 0, v_e_3717_);
lean_closure_set(v_compl_3818_, 1, v_toCompl_3794_);
lean_closure_set(v_compl_3818_, 2, v_toFun_3728_);
v_sdiff_3819_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_3819_, 0, v___f_3744_);
lean_closure_set(v_sdiff_3819_, 1, v_e_3717_);
lean_closure_set(v_sdiff_3819_, 2, v_toSDiff_3798_);
lean_closure_set(v_sdiff_3819_, 3, v_toFun_3728_);
v_bot_3820_ = lean_apply_1(v_toFun_3728_, v_toBot_3799_);
if (v_isShared_3810_ == 0)
{
v___x_3822_ = v___x_3809_;
goto v_reusejp_3821_;
}
else
{
lean_object* v_reuseFailAlloc_3836_; 
v_reuseFailAlloc_3836_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3836_, 0, v_toLE_3806_);
lean_ctor_set(v_reuseFailAlloc_3836_, 1, v_toLT_3807_);
v___x_3822_ = v_reuseFailAlloc_3836_;
goto v_reusejp_3821_;
}
v_reusejp_3821_:
{
lean_object* v___x_3824_; 
if (v_isShared_3815_ == 0)
{
lean_ctor_set(v___x_3814_, 1, v___f_3780_);
lean_ctor_set(v___x_3814_, 0, v___x_3822_);
v___x_3824_ = v___x_3814_;
goto v_reusejp_3823_;
}
else
{
lean_object* v_reuseFailAlloc_3835_; 
v_reuseFailAlloc_3835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3835_, 0, v___x_3822_);
lean_ctor_set(v_reuseFailAlloc_3835_, 1, v___f_3780_);
v___x_3824_ = v_reuseFailAlloc_3835_;
goto v_reusejp_3823_;
}
v_reusejp_3823_:
{
lean_object* v___x_3826_; 
if (v_isShared_3805_ == 0)
{
lean_ctor_set(v___x_3804_, 1, v___f_3758_);
lean_ctor_set(v___x_3804_, 0, v___x_3824_);
v___x_3826_ = v___x_3804_;
goto v_reusejp_3825_;
}
else
{
lean_object* v_reuseFailAlloc_3834_; 
v_reuseFailAlloc_3834_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3834_, 0, v___x_3824_);
lean_ctor_set(v_reuseFailAlloc_3834_, 1, v___f_3758_);
v___x_3826_ = v_reuseFailAlloc_3834_;
goto v_reusejp_3825_;
}
v_reusejp_3825_:
{
lean_object* v___x_3828_; 
if (v_isShared_3733_ == 0)
{
lean_ctor_set(v___x_3732_, 1, v_bot_3820_);
lean_ctor_set(v___x_3732_, 0, v_top_3816_);
v___x_3828_ = v___x_3732_;
goto v_reusejp_3827_;
}
else
{
lean_object* v_reuseFailAlloc_3833_; 
v_reuseFailAlloc_3833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3833_, 0, v_top_3816_);
lean_ctor_set(v_reuseFailAlloc_3833_, 1, v_bot_3820_);
v___x_3828_ = v_reuseFailAlloc_3833_;
goto v_reusejp_3827_;
}
v_reusejp_3827_:
{
lean_object* v___x_3829_; lean_object* v___x_3831_; 
v___x_3829_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3829_, 0, v___x_3826_);
lean_ctor_set(v___x_3829_, 1, v_toSupSet_3812_);
lean_ctor_set(v___x_3829_, 2, v_toInfSet_3802_);
lean_ctor_set(v___x_3829_, 3, v___x_3828_);
if (v_isShared_3793_ == 0)
{
lean_ctor_set(v___x_3792_, 3, v_himp_3817_);
lean_ctor_set(v___x_3792_, 2, v_sdiff_3819_);
lean_ctor_set(v___x_3792_, 1, v_compl_3818_);
lean_ctor_set(v___x_3792_, 0, v___x_3829_);
v___x_3831_ = v___x_3792_;
goto v_reusejp_3830_;
}
else
{
lean_object* v_reuseFailAlloc_3832_; 
v_reuseFailAlloc_3832_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_3832_, 0, v___x_3829_);
lean_ctor_set(v_reuseFailAlloc_3832_, 1, v_compl_3818_);
lean_ctor_set(v_reuseFailAlloc_3832_, 2, v_sdiff_3819_);
lean_ctor_set(v_reuseFailAlloc_3832_, 3, v_himp_3817_);
v___x_3831_ = v_reuseFailAlloc_3832_;
goto v_reusejp_3830_;
}
v_reusejp_3830_:
{
return v___x_3831_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeAtomicBooleanAlgebra___redArg(lean_object* v_e_3865_, lean_object* v_inst_3866_){
_start:
{
lean_object* v_toCompleteLattice_3867_; lean_object* v_toBoundedOrder_3868_; lean_object* v_toLattice_3869_; lean_object* v_toOrderTop_3870_; lean_object* v_toOrderBot_3871_; lean_object* v___x_3873_; uint8_t v_isShared_3874_; uint8_t v_isSharedCheck_4067_; 
v_toCompleteLattice_3867_ = lean_ctor_get(v_inst_3866_, 0);
v_toBoundedOrder_3868_ = lean_ctor_get(v_toCompleteLattice_3867_, 3);
lean_inc_ref(v_toBoundedOrder_3868_);
v_toLattice_3869_ = lean_ctor_get(v_toCompleteLattice_3867_, 0);
lean_inc_ref(v_toLattice_3869_);
v_toOrderTop_3870_ = lean_ctor_get(v_toBoundedOrder_3868_, 0);
v_toOrderBot_3871_ = lean_ctor_get(v_toBoundedOrder_3868_, 1);
v_isSharedCheck_4067_ = !lean_is_exclusive(v_toBoundedOrder_3868_);
if (v_isSharedCheck_4067_ == 0)
{
v___x_3873_ = v_toBoundedOrder_3868_;
v_isShared_3874_ = v_isSharedCheck_4067_;
goto v_resetjp_3872_;
}
else
{
lean_inc(v_toOrderBot_3871_);
lean_inc(v_toOrderTop_3870_);
lean_dec(v_toBoundedOrder_3868_);
v___x_3873_ = lean_box(0);
v_isShared_3874_ = v_isSharedCheck_4067_;
goto v_resetjp_3872_;
}
v_resetjp_3872_:
{
lean_object* v___x_3875_; lean_object* v_toFun_3876_; lean_object* v___x_3878_; uint8_t v_isShared_3879_; uint8_t v_isSharedCheck_4065_; 
lean_inc_ref(v_e_3865_);
v___x_3875_ = lp_mathlib_Equiv_symm___redArg(v_e_3865_);
v_toFun_3876_ = lean_ctor_get(v___x_3875_, 0);
v_isSharedCheck_4065_ = !lean_is_exclusive(v___x_3875_);
if (v_isSharedCheck_4065_ == 0)
{
lean_object* v_unused_4066_; 
v_unused_4066_ = lean_ctor_get(v___x_3875_, 1);
lean_dec(v_unused_4066_);
v___x_3878_ = v___x_3875_;
v_isShared_3879_ = v_isSharedCheck_4065_;
goto v_resetjp_3877_;
}
else
{
lean_inc(v_toFun_3876_);
lean_dec(v___x_3875_);
v___x_3878_ = lean_box(0);
v_isShared_3879_ = v_isSharedCheck_4065_;
goto v_resetjp_3877_;
}
v_resetjp_3877_:
{
lean_object* v___x_3880_; lean_object* v_toSupSet_3881_; lean_object* v___x_3883_; uint8_t v_isShared_3884_; uint8_t v_isSharedCheck_4063_; 
lean_inc_ref(v_toCompleteLattice_3867_);
v___x_3880_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_3867_);
v_toSupSet_3881_ = lean_ctor_get(v___x_3880_, 1);
v_isSharedCheck_4063_ = !lean_is_exclusive(v___x_3880_);
if (v_isSharedCheck_4063_ == 0)
{
lean_object* v_unused_4064_; 
v_unused_4064_ = lean_ctor_get(v___x_3880_, 0);
lean_dec(v_unused_4064_);
v___x_3883_ = v___x_3880_;
v_isShared_3884_ = v_isSharedCheck_4063_;
goto v_resetjp_3882_;
}
else
{
lean_inc(v_toSupSet_3881_);
lean_dec(v___x_3880_);
v___x_3883_ = lean_box(0);
v_isShared_3884_ = v_isSharedCheck_4063_;
goto v_resetjp_3882_;
}
v_resetjp_3882_:
{
lean_object* v___x_3885_; lean_object* v_toInfSet_3886_; lean_object* v___x_3888_; uint8_t v_isShared_3889_; uint8_t v_isSharedCheck_4061_; 
lean_inc_ref(v_toCompleteLattice_3867_);
v___x_3885_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_3867_);
v_toInfSet_3886_ = lean_ctor_get(v___x_3885_, 1);
v_isSharedCheck_4061_ = !lean_is_exclusive(v___x_3885_);
if (v_isSharedCheck_4061_ == 0)
{
lean_object* v_unused_4062_; 
v_unused_4062_ = lean_ctor_get(v___x_3885_, 0);
lean_dec(v_unused_4062_);
v___x_3888_ = v___x_3885_;
v_isShared_3889_ = v_isSharedCheck_4061_;
goto v_resetjp_3887_;
}
else
{
lean_inc(v_toInfSet_3886_);
lean_dec(v___x_3885_);
v___x_3888_ = lean_box(0);
v_isShared_3889_ = v_isSharedCheck_4061_;
goto v_resetjp_3887_;
}
v_resetjp_3887_:
{
lean_object* v_toSemilatticeSup_3890_; lean_object* v_inf_3891_; lean_object* v___x_3893_; uint8_t v_isShared_3894_; uint8_t v_isSharedCheck_4060_; 
v_toSemilatticeSup_3890_ = lean_ctor_get(v_toLattice_3869_, 0);
v_inf_3891_ = lean_ctor_get(v_toLattice_3869_, 1);
v_isSharedCheck_4060_ = !lean_is_exclusive(v_toLattice_3869_);
if (v_isSharedCheck_4060_ == 0)
{
v___x_3893_ = v_toLattice_3869_;
v_isShared_3894_ = v_isSharedCheck_4060_;
goto v_resetjp_3892_;
}
else
{
lean_inc(v_inf_3891_);
lean_inc(v_toSemilatticeSup_3890_);
lean_dec(v_toLattice_3869_);
v___x_3893_ = lean_box(0);
v_isShared_3894_ = v_isSharedCheck_4060_;
goto v_resetjp_3892_;
}
v_resetjp_3892_:
{
lean_object* v___f_3895_; lean_object* v_min_3896_; lean_object* v_le_3897_; lean_object* v_lt_3898_; lean_object* v_semilatticeInf_3899_; lean_object* v_toPartialOrder_3900_; lean_object* v___x_3902_; uint8_t v_isShared_3903_; uint8_t v_isSharedCheck_4058_; 
v___f_3895_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_3876_);
lean_inc_ref(v_e_3865_);
v_min_3896_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_3896_, 0, v___f_3895_);
lean_closure_set(v_min_3896_, 1, v_e_3865_);
lean_closure_set(v_min_3896_, 2, v_inf_3891_);
lean_closure_set(v_min_3896_, 3, v_toFun_3876_);
v_le_3897_ = lean_box(0);
v_lt_3898_ = lean_box(0);
lean_inc_ref(v_min_3896_);
v_semilatticeInf_3899_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_3896_, v_le_3897_, v_lt_3898_);
v_toPartialOrder_3900_ = lean_ctor_get(v_semilatticeInf_3899_, 0);
v_isSharedCheck_4058_ = !lean_is_exclusive(v_semilatticeInf_3899_);
if (v_isSharedCheck_4058_ == 0)
{
lean_object* v_unused_4059_; 
v_unused_4059_ = lean_ctor_get(v_semilatticeInf_3899_, 1);
lean_dec(v_unused_4059_);
v___x_3902_ = v_semilatticeInf_3899_;
v_isShared_3903_ = v_isSharedCheck_4058_;
goto v_resetjp_3901_;
}
else
{
lean_inc(v_toPartialOrder_3900_);
lean_dec(v_semilatticeInf_3899_);
v___x_3902_ = lean_box(0);
v_isShared_3903_ = v_isSharedCheck_4058_;
goto v_resetjp_3901_;
}
v_resetjp_3901_:
{
lean_object* v_toLE_3904_; lean_object* v_toLT_3905_; lean_object* v___x_3907_; uint8_t v_isShared_3908_; uint8_t v_isSharedCheck_4057_; 
v_toLE_3904_ = lean_ctor_get(v_toPartialOrder_3900_, 0);
v_toLT_3905_ = lean_ctor_get(v_toPartialOrder_3900_, 1);
v_isSharedCheck_4057_ = !lean_is_exclusive(v_toPartialOrder_3900_);
if (v_isSharedCheck_4057_ == 0)
{
v___x_3907_ = v_toPartialOrder_3900_;
v_isShared_3908_ = v_isSharedCheck_4057_;
goto v_resetjp_3906_;
}
else
{
lean_inc(v_toLT_3905_);
lean_inc(v_toLE_3904_);
lean_dec(v_toPartialOrder_3900_);
v___x_3907_ = lean_box(0);
v_isShared_3908_ = v_isSharedCheck_4057_;
goto v_resetjp_3906_;
}
v_resetjp_3906_:
{
lean_object* v___f_3909_; lean_object* v___f_3910_; lean_object* v___x_3912_; 
v___f_3909_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_3909_, 0, v_min_3896_);
lean_inc(v_toFun_3876_);
lean_inc_ref(v_e_3865_);
v___f_3910_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_3910_, 0, v_toSemilatticeSup_3890_);
lean_closure_set(v___f_3910_, 1, v___f_3895_);
lean_closure_set(v___f_3910_, 2, v_e_3865_);
lean_closure_set(v___f_3910_, 3, v_toFun_3876_);
if (v_isShared_3908_ == 0)
{
v___x_3912_ = v___x_3907_;
goto v_reusejp_3911_;
}
else
{
lean_object* v_reuseFailAlloc_4056_; 
v_reuseFailAlloc_4056_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4056_, 0, v_toLE_3904_);
lean_ctor_set(v_reuseFailAlloc_4056_, 1, v_toLT_3905_);
v___x_3912_ = v_reuseFailAlloc_4056_;
goto v_reusejp_3911_;
}
v_reusejp_3911_:
{
lean_object* v___x_3914_; 
lean_inc_ref(v___f_3910_);
if (v_isShared_3903_ == 0)
{
lean_ctor_set(v___x_3902_, 1, v___f_3910_);
lean_ctor_set(v___x_3902_, 0, v___x_3912_);
v___x_3914_ = v___x_3902_;
goto v_reusejp_3913_;
}
else
{
lean_object* v_reuseFailAlloc_4055_; 
v_reuseFailAlloc_4055_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4055_, 0, v___x_3912_);
lean_ctor_set(v_reuseFailAlloc_4055_, 1, v___f_3910_);
v___x_3914_ = v_reuseFailAlloc_4055_;
goto v_reusejp_3913_;
}
v_reusejp_3913_:
{
lean_object* v_lattice_3916_; 
lean_inc_ref(v___f_3909_);
if (v_isShared_3894_ == 0)
{
lean_ctor_set(v___x_3893_, 1, v___f_3909_);
lean_ctor_set(v___x_3893_, 0, v___x_3914_);
v_lattice_3916_ = v___x_3893_;
goto v_reusejp_3915_;
}
else
{
lean_object* v_reuseFailAlloc_4054_; 
v_reuseFailAlloc_4054_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4054_, 0, v___x_3914_);
lean_ctor_set(v_reuseFailAlloc_4054_, 1, v___f_3909_);
v_lattice_3916_ = v_reuseFailAlloc_4054_;
goto v_reusejp_3915_;
}
v_reusejp_3915_:
{
lean_object* v___x_3917_; lean_object* v_toPartialOrder_3918_; lean_object* v___x_3920_; uint8_t v_isShared_3921_; uint8_t v_isSharedCheck_4052_; 
v___x_3917_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_3916_);
v_toPartialOrder_3918_ = lean_ctor_get(v___x_3917_, 0);
v_isSharedCheck_4052_ = !lean_is_exclusive(v___x_3917_);
if (v_isSharedCheck_4052_ == 0)
{
lean_object* v_unused_4053_; 
v_unused_4053_ = lean_ctor_get(v___x_3917_, 1);
lean_dec(v_unused_4053_);
v___x_3920_ = v___x_3917_;
v_isShared_3921_ = v_isSharedCheck_4052_;
goto v_resetjp_3919_;
}
else
{
lean_inc(v_toPartialOrder_3918_);
lean_dec(v___x_3917_);
v___x_3920_ = lean_box(0);
v_isShared_3921_ = v_isSharedCheck_4052_;
goto v_resetjp_3919_;
}
v_resetjp_3919_:
{
lean_object* v_toLE_3922_; lean_object* v_toLT_3923_; lean_object* v___x_3925_; uint8_t v_isShared_3926_; uint8_t v_isSharedCheck_4051_; 
v_toLE_3922_ = lean_ctor_get(v_toPartialOrder_3918_, 0);
v_toLT_3923_ = lean_ctor_get(v_toPartialOrder_3918_, 1);
v_isSharedCheck_4051_ = !lean_is_exclusive(v_toPartialOrder_3918_);
if (v_isSharedCheck_4051_ == 0)
{
v___x_3925_ = v_toPartialOrder_3918_;
v_isShared_3926_ = v_isSharedCheck_4051_;
goto v_resetjp_3924_;
}
else
{
lean_inc(v_toLT_3923_);
lean_inc(v_toLE_3922_);
lean_dec(v_toPartialOrder_3918_);
v___x_3925_ = lean_box(0);
v_isShared_3926_ = v_isSharedCheck_4051_;
goto v_resetjp_3924_;
}
v_resetjp_3924_:
{
lean_object* v_top_3927_; lean_object* v_bot_3928_; lean_object* v_supSet_3929_; lean_object* v_infSet_3930_; lean_object* v___f_3931_; lean_object* v___x_3933_; 
lean_inc_n(v_toFun_3876_, 4);
v_top_3927_ = lean_apply_1(v_toFun_3876_, v_toOrderTop_3870_);
v_bot_3928_ = lean_apply_1(v_toFun_3876_, v_toOrderBot_3871_);
v_supSet_3929_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_3929_, 0, v_toSupSet_3881_);
lean_closure_set(v_supSet_3929_, 1, v_toFun_3876_);
v_infSet_3930_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_3930_, 0, v_toInfSet_3886_);
lean_closure_set(v_infSet_3930_, 1, v_toFun_3876_);
v___f_3931_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_3931_, 0, v___f_3910_);
if (v_isShared_3926_ == 0)
{
v___x_3933_ = v___x_3925_;
goto v_reusejp_3932_;
}
else
{
lean_object* v_reuseFailAlloc_4050_; 
v_reuseFailAlloc_4050_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4050_, 0, v_toLE_3922_);
lean_ctor_set(v_reuseFailAlloc_4050_, 1, v_toLT_3923_);
v___x_3933_ = v_reuseFailAlloc_4050_;
goto v_reusejp_3932_;
}
v_reusejp_3932_:
{
lean_object* v___x_3935_; 
lean_inc_ref(v___f_3931_);
if (v_isShared_3921_ == 0)
{
lean_ctor_set(v___x_3920_, 1, v___f_3931_);
lean_ctor_set(v___x_3920_, 0, v___x_3933_);
v___x_3935_ = v___x_3920_;
goto v_reusejp_3934_;
}
else
{
lean_object* v_reuseFailAlloc_4049_; 
v_reuseFailAlloc_4049_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4049_, 0, v___x_3933_);
lean_ctor_set(v_reuseFailAlloc_4049_, 1, v___f_3931_);
v___x_3935_ = v_reuseFailAlloc_4049_;
goto v_reusejp_3934_;
}
v_reusejp_3934_:
{
lean_object* v___x_3937_; 
lean_inc_ref(v___f_3909_);
if (v_isShared_3889_ == 0)
{
lean_ctor_set(v___x_3888_, 1, v___f_3909_);
lean_ctor_set(v___x_3888_, 0, v___x_3935_);
v___x_3937_ = v___x_3888_;
goto v_reusejp_3936_;
}
else
{
lean_object* v_reuseFailAlloc_4048_; 
v_reuseFailAlloc_4048_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4048_, 0, v___x_3935_);
lean_ctor_set(v_reuseFailAlloc_4048_, 1, v___f_3909_);
v___x_3937_ = v_reuseFailAlloc_4048_;
goto v_reusejp_3936_;
}
v_reusejp_3936_:
{
lean_object* v___x_3939_; 
if (v_isShared_3874_ == 0)
{
lean_ctor_set(v___x_3873_, 1, v_bot_3928_);
lean_ctor_set(v___x_3873_, 0, v_top_3927_);
v___x_3939_ = v___x_3873_;
goto v_reusejp_3938_;
}
else
{
lean_object* v_reuseFailAlloc_4047_; 
v_reuseFailAlloc_4047_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4047_, 0, v_top_3927_);
lean_ctor_set(v_reuseFailAlloc_4047_, 1, v_bot_3928_);
v___x_3939_ = v_reuseFailAlloc_4047_;
goto v_reusejp_3938_;
}
v_reusejp_3938_:
{
lean_object* v_completeLattice_3940_; lean_object* v___x_3941_; lean_object* v___x_3943_; uint8_t v_isShared_3944_; uint8_t v_isSharedCheck_4042_; 
v_completeLattice_3940_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_3940_, 0, v___x_3937_);
lean_ctor_set(v_completeLattice_3940_, 1, v_supSet_3929_);
lean_ctor_set(v_completeLattice_3940_, 2, v_infSet_3930_);
lean_ctor_set(v_completeLattice_3940_, 3, v___x_3939_);
v___x_3941_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_3866_);
v_isSharedCheck_4042_ = !lean_is_exclusive(v_inst_3866_);
if (v_isSharedCheck_4042_ == 0)
{
lean_object* v_unused_4043_; lean_object* v_unused_4044_; lean_object* v_unused_4045_; lean_object* v_unused_4046_; 
v_unused_4043_ = lean_ctor_get(v_inst_3866_, 3);
lean_dec(v_unused_4043_);
v_unused_4044_ = lean_ctor_get(v_inst_3866_, 2);
lean_dec(v_unused_4044_);
v_unused_4045_ = lean_ctor_get(v_inst_3866_, 1);
lean_dec(v_unused_4045_);
v_unused_4046_ = lean_ctor_get(v_inst_3866_, 0);
lean_dec(v_unused_4046_);
v___x_3943_ = v_inst_3866_;
v_isShared_3944_ = v_isSharedCheck_4042_;
goto v_resetjp_3942_;
}
else
{
lean_dec(v_inst_3866_);
v___x_3943_ = lean_box(0);
v_isShared_3944_ = v_isSharedCheck_4042_;
goto v_resetjp_3942_;
}
v_resetjp_3942_:
{
lean_object* v_toCompl_3945_; lean_object* v_toHImp_3946_; lean_object* v_toTop_3947_; lean_object* v___x_3948_; lean_object* v_toSDiff_3949_; lean_object* v_toBot_3950_; lean_object* v___x_3951_; lean_object* v_toPartialOrder_3952_; lean_object* v_toInfSet_3953_; lean_object* v___x_3955_; uint8_t v_isShared_3956_; uint8_t v_isSharedCheck_4041_; 
v_toCompl_3945_ = lean_ctor_get(v___x_3941_, 1);
lean_inc(v_toCompl_3945_);
v_toHImp_3946_ = lean_ctor_get(v___x_3941_, 3);
lean_inc(v_toHImp_3946_);
v_toTop_3947_ = lean_ctor_get(v___x_3941_, 4);
lean_inc(v_toTop_3947_);
v___x_3948_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(v___x_3941_);
lean_dec_ref(v___x_3941_);
v_toSDiff_3949_ = lean_ctor_get(v___x_3948_, 1);
lean_inc(v_toSDiff_3949_);
v_toBot_3950_ = lean_ctor_get(v___x_3948_, 2);
lean_inc(v_toBot_3950_);
lean_dec_ref(v___x_3948_);
lean_inc_ref(v_completeLattice_3940_);
v___x_3951_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_3940_);
v_toPartialOrder_3952_ = lean_ctor_get(v___x_3951_, 0);
v_toInfSet_3953_ = lean_ctor_get(v___x_3951_, 1);
v_isSharedCheck_4041_ = !lean_is_exclusive(v___x_3951_);
if (v_isSharedCheck_4041_ == 0)
{
v___x_3955_ = v___x_3951_;
v_isShared_3956_ = v_isSharedCheck_4041_;
goto v_resetjp_3954_;
}
else
{
lean_inc(v_toInfSet_3953_);
lean_inc(v_toPartialOrder_3952_);
lean_dec(v___x_3951_);
v___x_3955_ = lean_box(0);
v_isShared_3956_ = v_isSharedCheck_4041_;
goto v_resetjp_3954_;
}
v_resetjp_3954_:
{
lean_object* v_toLE_3957_; lean_object* v_toLT_3958_; lean_object* v___x_3960_; uint8_t v_isShared_3961_; uint8_t v_isSharedCheck_4040_; 
v_toLE_3957_ = lean_ctor_get(v_toPartialOrder_3952_, 0);
v_toLT_3958_ = lean_ctor_get(v_toPartialOrder_3952_, 1);
v_isSharedCheck_4040_ = !lean_is_exclusive(v_toPartialOrder_3952_);
if (v_isSharedCheck_4040_ == 0)
{
v___x_3960_ = v_toPartialOrder_3952_;
v_isShared_3961_ = v_isSharedCheck_4040_;
goto v_resetjp_3959_;
}
else
{
lean_inc(v_toLT_3958_);
lean_inc(v_toLE_3957_);
lean_dec(v_toPartialOrder_3952_);
v___x_3960_ = lean_box(0);
v_isShared_3961_ = v_isSharedCheck_4040_;
goto v_resetjp_3959_;
}
v_resetjp_3959_:
{
lean_object* v___x_3962_; lean_object* v_toSupSet_3963_; lean_object* v___x_3965_; uint8_t v_isShared_3966_; uint8_t v_isSharedCheck_4038_; 
v___x_3962_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_3940_);
v_toSupSet_3963_ = lean_ctor_get(v___x_3962_, 1);
v_isSharedCheck_4038_ = !lean_is_exclusive(v___x_3962_);
if (v_isSharedCheck_4038_ == 0)
{
lean_object* v_unused_4039_; 
v_unused_4039_ = lean_ctor_get(v___x_3962_, 0);
lean_dec(v_unused_4039_);
v___x_3965_ = v___x_3962_;
v_isShared_3966_ = v_isSharedCheck_4038_;
goto v_resetjp_3964_;
}
else
{
lean_inc(v_toSupSet_3963_);
lean_dec(v___x_3962_);
v___x_3965_ = lean_box(0);
v_isShared_3966_ = v_isSharedCheck_4038_;
goto v_resetjp_3964_;
}
v_resetjp_3964_:
{
lean_object* v_top_3967_; lean_object* v_himp_3968_; lean_object* v_compl_3969_; lean_object* v_sdiff_3970_; lean_object* v_bot_3971_; lean_object* v___x_3973_; 
lean_inc_n(v_toFun_3876_, 4);
v_top_3967_ = lean_apply_1(v_toFun_3876_, v_toTop_3947_);
lean_inc_ref_n(v_e_3865_, 2);
v_himp_3968_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_3968_, 0, v___f_3895_);
lean_closure_set(v_himp_3968_, 1, v_e_3865_);
lean_closure_set(v_himp_3968_, 2, v_toHImp_3946_);
lean_closure_set(v_himp_3968_, 3, v_toFun_3876_);
v_compl_3969_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_3969_, 0, v_e_3865_);
lean_closure_set(v_compl_3969_, 1, v_toCompl_3945_);
lean_closure_set(v_compl_3969_, 2, v_toFun_3876_);
v_sdiff_3970_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_3970_, 0, v___f_3895_);
lean_closure_set(v_sdiff_3970_, 1, v_e_3865_);
lean_closure_set(v_sdiff_3970_, 2, v_toSDiff_3949_);
lean_closure_set(v_sdiff_3970_, 3, v_toFun_3876_);
v_bot_3971_ = lean_apply_1(v_toFun_3876_, v_toBot_3950_);
if (v_isShared_3961_ == 0)
{
v___x_3973_ = v___x_3960_;
goto v_reusejp_3972_;
}
else
{
lean_object* v_reuseFailAlloc_4037_; 
v_reuseFailAlloc_4037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4037_, 0, v_toLE_3957_);
lean_ctor_set(v_reuseFailAlloc_4037_, 1, v_toLT_3958_);
v___x_3973_ = v_reuseFailAlloc_4037_;
goto v_reusejp_3972_;
}
v_reusejp_3972_:
{
lean_object* v___x_3975_; 
lean_inc_ref(v___f_3931_);
if (v_isShared_3966_ == 0)
{
lean_ctor_set(v___x_3965_, 1, v___f_3931_);
lean_ctor_set(v___x_3965_, 0, v___x_3973_);
v___x_3975_ = v___x_3965_;
goto v_reusejp_3974_;
}
else
{
lean_object* v_reuseFailAlloc_4036_; 
v_reuseFailAlloc_4036_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4036_, 0, v___x_3973_);
lean_ctor_set(v_reuseFailAlloc_4036_, 1, v___f_3931_);
v___x_3975_ = v_reuseFailAlloc_4036_;
goto v_reusejp_3974_;
}
v_reusejp_3974_:
{
lean_object* v___x_3977_; 
lean_inc_ref(v___f_3909_);
lean_inc_ref(v___x_3975_);
if (v_isShared_3956_ == 0)
{
lean_ctor_set(v___x_3955_, 1, v___f_3909_);
lean_ctor_set(v___x_3955_, 0, v___x_3975_);
v___x_3977_ = v___x_3955_;
goto v_reusejp_3976_;
}
else
{
lean_object* v_reuseFailAlloc_4035_; 
v_reuseFailAlloc_4035_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4035_, 0, v___x_3975_);
lean_ctor_set(v_reuseFailAlloc_4035_, 1, v___f_3909_);
v___x_3977_ = v_reuseFailAlloc_4035_;
goto v_reusejp_3976_;
}
v_reusejp_3976_:
{
lean_object* v___x_3979_; 
if (v_isShared_3884_ == 0)
{
lean_ctor_set(v___x_3883_, 1, v_bot_3971_);
lean_ctor_set(v___x_3883_, 0, v_top_3967_);
v___x_3979_ = v___x_3883_;
goto v_reusejp_3978_;
}
else
{
lean_object* v_reuseFailAlloc_4034_; 
v_reuseFailAlloc_4034_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4034_, 0, v_top_3967_);
lean_ctor_set(v_reuseFailAlloc_4034_, 1, v_bot_3971_);
v___x_3979_ = v_reuseFailAlloc_4034_;
goto v_reusejp_3978_;
}
v_reusejp_3978_:
{
lean_object* v___x_3980_; lean_object* v_completeBooleanAlgebra_3982_; 
lean_inc_ref(v___x_3977_);
v___x_3980_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3980_, 0, v___x_3977_);
lean_ctor_set(v___x_3980_, 1, v_toSupSet_3963_);
lean_ctor_set(v___x_3980_, 2, v_toInfSet_3953_);
lean_ctor_set(v___x_3980_, 3, v___x_3979_);
lean_inc_ref(v___x_3980_);
if (v_isShared_3944_ == 0)
{
lean_ctor_set(v___x_3943_, 3, v_himp_3968_);
lean_ctor_set(v___x_3943_, 2, v_sdiff_3970_);
lean_ctor_set(v___x_3943_, 1, v_compl_3969_);
lean_ctor_set(v___x_3943_, 0, v___x_3980_);
v_completeBooleanAlgebra_3982_ = v___x_3943_;
goto v_reusejp_3981_;
}
else
{
lean_object* v_reuseFailAlloc_4033_; 
v_reuseFailAlloc_4033_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_4033_, 0, v___x_3980_);
lean_ctor_set(v_reuseFailAlloc_4033_, 1, v_compl_3969_);
lean_ctor_set(v_reuseFailAlloc_4033_, 2, v_sdiff_3970_);
lean_ctor_set(v_reuseFailAlloc_4033_, 3, v_himp_3968_);
v_completeBooleanAlgebra_3982_ = v_reuseFailAlloc_4033_;
goto v_reusejp_3981_;
}
v_reusejp_3981_:
{
lean_object* v___x_3983_; lean_object* v___x_3984_; lean_object* v_toPartialOrder_3985_; lean_object* v_toInfSet_3986_; lean_object* v___x_3988_; uint8_t v_isShared_3989_; uint8_t v_isSharedCheck_4032_; 
v___x_3983_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_3977_);
lean_inc_ref(v___x_3980_);
v___x_3984_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v___x_3980_);
v_toPartialOrder_3985_ = lean_ctor_get(v___x_3984_, 0);
v_toInfSet_3986_ = lean_ctor_get(v___x_3984_, 1);
v_isSharedCheck_4032_ = !lean_is_exclusive(v___x_3984_);
if (v_isSharedCheck_4032_ == 0)
{
v___x_3988_ = v___x_3984_;
v_isShared_3989_ = v_isSharedCheck_4032_;
goto v_resetjp_3987_;
}
else
{
lean_inc(v_toInfSet_3986_);
lean_inc(v_toPartialOrder_3985_);
lean_dec(v___x_3984_);
v___x_3988_ = lean_box(0);
v_isShared_3989_ = v_isSharedCheck_4032_;
goto v_resetjp_3987_;
}
v_resetjp_3987_:
{
lean_object* v_toLE_3990_; lean_object* v_toLT_3991_; lean_object* v___x_3993_; uint8_t v_isShared_3994_; uint8_t v_isSharedCheck_4031_; 
v_toLE_3990_ = lean_ctor_get(v_toPartialOrder_3985_, 0);
v_toLT_3991_ = lean_ctor_get(v_toPartialOrder_3985_, 1);
v_isSharedCheck_4031_ = !lean_is_exclusive(v_toPartialOrder_3985_);
if (v_isSharedCheck_4031_ == 0)
{
v___x_3993_ = v_toPartialOrder_3985_;
v_isShared_3994_ = v_isSharedCheck_4031_;
goto v_resetjp_3992_;
}
else
{
lean_inc(v_toLT_3991_);
lean_inc(v_toLE_3990_);
lean_dec(v_toPartialOrder_3985_);
v___x_3993_ = lean_box(0);
v_isShared_3994_ = v_isSharedCheck_4031_;
goto v_resetjp_3992_;
}
v_resetjp_3992_:
{
lean_object* v___x_3995_; lean_object* v_toSupSet_3996_; lean_object* v___x_3998_; uint8_t v_isShared_3999_; uint8_t v_isSharedCheck_4029_; 
v___x_3995_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v___x_3980_);
v_toSupSet_3996_ = lean_ctor_get(v___x_3995_, 1);
v_isSharedCheck_4029_ = !lean_is_exclusive(v___x_3995_);
if (v_isSharedCheck_4029_ == 0)
{
lean_object* v_unused_4030_; 
v_unused_4030_ = lean_ctor_get(v___x_3995_, 0);
lean_dec(v_unused_4030_);
v___x_3998_ = v___x_3995_;
v_isShared_3999_ = v_isSharedCheck_4029_;
goto v_resetjp_3997_;
}
else
{
lean_inc(v_toSupSet_3996_);
lean_dec(v___x_3995_);
v___x_3998_ = lean_box(0);
v_isShared_3999_ = v_isSharedCheck_4029_;
goto v_resetjp_3997_;
}
v_resetjp_3997_:
{
lean_object* v___x_4000_; lean_object* v_toCompl_4001_; lean_object* v_toSDiff_4002_; lean_object* v_toHImp_4003_; lean_object* v_toTop_4004_; lean_object* v_toBot_4005_; lean_object* v___x_4006_; lean_object* v___x_4007_; lean_object* v___x_4008_; lean_object* v_toHNot_4009_; lean_object* v___f_4010_; lean_object* v___f_4011_; lean_object* v___x_4013_; 
v___x_4000_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_completeBooleanAlgebra_3982_);
v_toCompl_4001_ = lean_ctor_get(v___x_4000_, 1);
lean_inc(v_toCompl_4001_);
v_toSDiff_4002_ = lean_ctor_get(v___x_4000_, 2);
lean_inc(v_toSDiff_4002_);
v_toHImp_4003_ = lean_ctor_get(v___x_4000_, 3);
lean_inc(v_toHImp_4003_);
v_toTop_4004_ = lean_ctor_get(v___x_4000_, 4);
lean_inc(v_toTop_4004_);
v_toBot_4005_ = lean_ctor_get(v___x_4000_, 5);
lean_inc(v_toBot_4005_);
lean_dec_ref(v___x_4000_);
v___x_4006_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(v_completeBooleanAlgebra_3982_);
lean_dec_ref(v_completeBooleanAlgebra_3982_);
v___x_4007_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_4006_);
v___x_4008_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v___x_4007_);
v_toHNot_4009_ = lean_ctor_get(v___x_4008_, 2);
lean_inc(v_toHNot_4009_);
lean_dec_ref(v___x_4008_);
v___f_4010_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_4010_, 0, v___x_3975_);
v___f_4011_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_4011_, 0, v___x_3983_);
if (v_isShared_3994_ == 0)
{
v___x_4013_ = v___x_3993_;
goto v_reusejp_4012_;
}
else
{
lean_object* v_reuseFailAlloc_4028_; 
v_reuseFailAlloc_4028_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4028_, 0, v_toLE_3990_);
lean_ctor_set(v_reuseFailAlloc_4028_, 1, v_toLT_3991_);
v___x_4013_ = v_reuseFailAlloc_4028_;
goto v_reusejp_4012_;
}
v_reusejp_4012_:
{
lean_object* v___x_4015_; 
if (v_isShared_3999_ == 0)
{
lean_ctor_set(v___x_3998_, 1, v___f_3931_);
lean_ctor_set(v___x_3998_, 0, v___x_4013_);
v___x_4015_ = v___x_3998_;
goto v_reusejp_4014_;
}
else
{
lean_object* v_reuseFailAlloc_4027_; 
v_reuseFailAlloc_4027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4027_, 0, v___x_4013_);
lean_ctor_set(v_reuseFailAlloc_4027_, 1, v___f_3931_);
v___x_4015_ = v_reuseFailAlloc_4027_;
goto v_reusejp_4014_;
}
v_reusejp_4014_:
{
lean_object* v___x_4017_; 
if (v_isShared_3989_ == 0)
{
lean_ctor_set(v___x_3988_, 1, v___f_3909_);
lean_ctor_set(v___x_3988_, 0, v___x_4015_);
v___x_4017_ = v___x_3988_;
goto v_reusejp_4016_;
}
else
{
lean_object* v_reuseFailAlloc_4026_; 
v_reuseFailAlloc_4026_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4026_, 0, v___x_4015_);
lean_ctor_set(v_reuseFailAlloc_4026_, 1, v___f_3909_);
v___x_4017_ = v_reuseFailAlloc_4026_;
goto v_reusejp_4016_;
}
v_reusejp_4016_:
{
lean_object* v___x_4019_; 
lean_inc(v_toBot_4005_);
lean_inc(v_toTop_4004_);
if (v_isShared_3879_ == 0)
{
lean_ctor_set(v___x_3878_, 1, v_toBot_4005_);
lean_ctor_set(v___x_3878_, 0, v_toTop_4004_);
v___x_4019_ = v___x_3878_;
goto v_reusejp_4018_;
}
else
{
lean_object* v_reuseFailAlloc_4025_; 
v_reuseFailAlloc_4025_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4025_, 0, v_toTop_4004_);
lean_ctor_set(v_reuseFailAlloc_4025_, 1, v_toBot_4005_);
v___x_4019_ = v_reuseFailAlloc_4025_;
goto v_reusejp_4018_;
}
v_reusejp_4018_:
{
lean_object* v___x_4020_; lean_object* v___x_4021_; lean_object* v_toGeneralizedCoheytingAlgebra_4022_; lean_object* v_toSDiff_4023_; lean_object* v___x_4024_; 
v___x_4020_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4020_, 0, v___x_4017_);
lean_ctor_set(v___x_4020_, 1, v_toSupSet_3996_);
lean_ctor_set(v___x_4020_, 2, v_toInfSet_3986_);
lean_ctor_set(v___x_4020_, 3, v___x_4019_);
v___x_4021_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_4011_, v___f_4010_, v_toLE_3990_, v_toLT_3991_, v_toBot_4005_, v_toTop_4004_, v_toHNot_4009_, v_toSDiff_4002_);
v_toGeneralizedCoheytingAlgebra_4022_ = lean_ctor_get(v___x_4021_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_4022_);
lean_dec_ref(v___x_4021_);
v_toSDiff_4023_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_4022_, 2);
lean_inc(v_toSDiff_4023_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_4022_);
v___x_4024_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4024_, 0, v___x_4020_);
lean_ctor_set(v___x_4024_, 1, v_toCompl_4001_);
lean_ctor_set(v___x_4024_, 2, v_toSDiff_4023_);
lean_ctor_set(v___x_4024_, 3, v_toHImp_4003_);
return v___x_4024_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeAtomicBooleanAlgebra(lean_object* v_00_u03b1_4068_, lean_object* v_00_u03b2_4069_, lean_object* v_e_4070_, lean_object* v_inst_4071_){
_start:
{
lean_object* v_toCompleteLattice_4072_; lean_object* v_toBoundedOrder_4073_; lean_object* v_toLattice_4074_; lean_object* v_toOrderTop_4075_; lean_object* v_toOrderBot_4076_; lean_object* v___x_4078_; uint8_t v_isShared_4079_; uint8_t v_isSharedCheck_4272_; 
v_toCompleteLattice_4072_ = lean_ctor_get(v_inst_4071_, 0);
v_toBoundedOrder_4073_ = lean_ctor_get(v_toCompleteLattice_4072_, 3);
lean_inc_ref(v_toBoundedOrder_4073_);
v_toLattice_4074_ = lean_ctor_get(v_toCompleteLattice_4072_, 0);
lean_inc_ref(v_toLattice_4074_);
v_toOrderTop_4075_ = lean_ctor_get(v_toBoundedOrder_4073_, 0);
v_toOrderBot_4076_ = lean_ctor_get(v_toBoundedOrder_4073_, 1);
v_isSharedCheck_4272_ = !lean_is_exclusive(v_toBoundedOrder_4073_);
if (v_isSharedCheck_4272_ == 0)
{
v___x_4078_ = v_toBoundedOrder_4073_;
v_isShared_4079_ = v_isSharedCheck_4272_;
goto v_resetjp_4077_;
}
else
{
lean_inc(v_toOrderBot_4076_);
lean_inc(v_toOrderTop_4075_);
lean_dec(v_toBoundedOrder_4073_);
v___x_4078_ = lean_box(0);
v_isShared_4079_ = v_isSharedCheck_4272_;
goto v_resetjp_4077_;
}
v_resetjp_4077_:
{
lean_object* v___x_4080_; lean_object* v_toFun_4081_; lean_object* v___x_4083_; uint8_t v_isShared_4084_; uint8_t v_isSharedCheck_4270_; 
lean_inc_ref(v_e_4070_);
v___x_4080_ = lp_mathlib_Equiv_symm___redArg(v_e_4070_);
v_toFun_4081_ = lean_ctor_get(v___x_4080_, 0);
v_isSharedCheck_4270_ = !lean_is_exclusive(v___x_4080_);
if (v_isSharedCheck_4270_ == 0)
{
lean_object* v_unused_4271_; 
v_unused_4271_ = lean_ctor_get(v___x_4080_, 1);
lean_dec(v_unused_4271_);
v___x_4083_ = v___x_4080_;
v_isShared_4084_ = v_isSharedCheck_4270_;
goto v_resetjp_4082_;
}
else
{
lean_inc(v_toFun_4081_);
lean_dec(v___x_4080_);
v___x_4083_ = lean_box(0);
v_isShared_4084_ = v_isSharedCheck_4270_;
goto v_resetjp_4082_;
}
v_resetjp_4082_:
{
lean_object* v___x_4085_; lean_object* v_toSupSet_4086_; lean_object* v___x_4088_; uint8_t v_isShared_4089_; uint8_t v_isSharedCheck_4268_; 
lean_inc_ref(v_toCompleteLattice_4072_);
v___x_4085_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_toCompleteLattice_4072_);
v_toSupSet_4086_ = lean_ctor_get(v___x_4085_, 1);
v_isSharedCheck_4268_ = !lean_is_exclusive(v___x_4085_);
if (v_isSharedCheck_4268_ == 0)
{
lean_object* v_unused_4269_; 
v_unused_4269_ = lean_ctor_get(v___x_4085_, 0);
lean_dec(v_unused_4269_);
v___x_4088_ = v___x_4085_;
v_isShared_4089_ = v_isSharedCheck_4268_;
goto v_resetjp_4087_;
}
else
{
lean_inc(v_toSupSet_4086_);
lean_dec(v___x_4085_);
v___x_4088_ = lean_box(0);
v_isShared_4089_ = v_isSharedCheck_4268_;
goto v_resetjp_4087_;
}
v_resetjp_4087_:
{
lean_object* v___x_4090_; lean_object* v_toInfSet_4091_; lean_object* v___x_4093_; uint8_t v_isShared_4094_; uint8_t v_isSharedCheck_4266_; 
lean_inc_ref(v_toCompleteLattice_4072_);
v___x_4090_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_toCompleteLattice_4072_);
v_toInfSet_4091_ = lean_ctor_get(v___x_4090_, 1);
v_isSharedCheck_4266_ = !lean_is_exclusive(v___x_4090_);
if (v_isSharedCheck_4266_ == 0)
{
lean_object* v_unused_4267_; 
v_unused_4267_ = lean_ctor_get(v___x_4090_, 0);
lean_dec(v_unused_4267_);
v___x_4093_ = v___x_4090_;
v_isShared_4094_ = v_isSharedCheck_4266_;
goto v_resetjp_4092_;
}
else
{
lean_inc(v_toInfSet_4091_);
lean_dec(v___x_4090_);
v___x_4093_ = lean_box(0);
v_isShared_4094_ = v_isSharedCheck_4266_;
goto v_resetjp_4092_;
}
v_resetjp_4092_:
{
lean_object* v_toSemilatticeSup_4095_; lean_object* v_inf_4096_; lean_object* v___x_4098_; uint8_t v_isShared_4099_; uint8_t v_isSharedCheck_4265_; 
v_toSemilatticeSup_4095_ = lean_ctor_get(v_toLattice_4074_, 0);
v_inf_4096_ = lean_ctor_get(v_toLattice_4074_, 1);
v_isSharedCheck_4265_ = !lean_is_exclusive(v_toLattice_4074_);
if (v_isSharedCheck_4265_ == 0)
{
v___x_4098_ = v_toLattice_4074_;
v_isShared_4099_ = v_isSharedCheck_4265_;
goto v_resetjp_4097_;
}
else
{
lean_inc(v_inf_4096_);
lean_inc(v_toSemilatticeSup_4095_);
lean_dec(v_toLattice_4074_);
v___x_4098_ = lean_box(0);
v_isShared_4099_ = v_isSharedCheck_4265_;
goto v_resetjp_4097_;
}
v_resetjp_4097_:
{
lean_object* v___f_4100_; lean_object* v_min_4101_; lean_object* v_le_4102_; lean_object* v_lt_4103_; lean_object* v_semilatticeInf_4104_; lean_object* v_toPartialOrder_4105_; lean_object* v___x_4107_; uint8_t v_isShared_4108_; uint8_t v_isSharedCheck_4263_; 
v___f_4100_ = ((lean_object*)(lp_mathlib_Equiv_frame___redArg___closed__0));
lean_inc(v_toFun_4081_);
lean_inc_ref(v_e_4070_);
v_min_4101_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__1), 6, 4);
lean_closure_set(v_min_4101_, 0, v___f_4100_);
lean_closure_set(v_min_4101_, 1, v_e_4070_);
lean_closure_set(v_min_4101_, 2, v_inf_4096_);
lean_closure_set(v_min_4101_, 3, v_toFun_4081_);
v_le_4102_ = lean_box(0);
v_lt_4103_ = lean_box(0);
lean_inc_ref(v_min_4101_);
v_semilatticeInf_4104_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_4101_, v_le_4102_, v_lt_4103_);
v_toPartialOrder_4105_ = lean_ctor_get(v_semilatticeInf_4104_, 0);
v_isSharedCheck_4263_ = !lean_is_exclusive(v_semilatticeInf_4104_);
if (v_isSharedCheck_4263_ == 0)
{
lean_object* v_unused_4264_; 
v_unused_4264_ = lean_ctor_get(v_semilatticeInf_4104_, 1);
lean_dec(v_unused_4264_);
v___x_4107_ = v_semilatticeInf_4104_;
v_isShared_4108_ = v_isSharedCheck_4263_;
goto v_resetjp_4106_;
}
else
{
lean_inc(v_toPartialOrder_4105_);
lean_dec(v_semilatticeInf_4104_);
v___x_4107_ = lean_box(0);
v_isShared_4108_ = v_isSharedCheck_4263_;
goto v_resetjp_4106_;
}
v_resetjp_4106_:
{
lean_object* v_toLE_4109_; lean_object* v_toLT_4110_; lean_object* v___x_4112_; uint8_t v_isShared_4113_; uint8_t v_isSharedCheck_4262_; 
v_toLE_4109_ = lean_ctor_get(v_toPartialOrder_4105_, 0);
v_toLT_4110_ = lean_ctor_get(v_toPartialOrder_4105_, 1);
v_isSharedCheck_4262_ = !lean_is_exclusive(v_toPartialOrder_4105_);
if (v_isSharedCheck_4262_ == 0)
{
v___x_4112_ = v_toPartialOrder_4105_;
v_isShared_4113_ = v_isSharedCheck_4262_;
goto v_resetjp_4111_;
}
else
{
lean_inc(v_toLT_4110_);
lean_inc(v_toLE_4109_);
lean_dec(v_toPartialOrder_4105_);
v___x_4112_ = lean_box(0);
v_isShared_4113_ = v_isSharedCheck_4262_;
goto v_resetjp_4111_;
}
v_resetjp_4111_:
{
lean_object* v___f_4114_; lean_object* v___f_4115_; lean_object* v___x_4117_; 
v___f_4114_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__3), 3, 1);
lean_closure_set(v___f_4114_, 0, v_min_4101_);
lean_inc(v_toFun_4081_);
lean_inc_ref(v_e_4070_);
v___f_4115_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__2), 6, 4);
lean_closure_set(v___f_4115_, 0, v_toSemilatticeSup_4095_);
lean_closure_set(v___f_4115_, 1, v___f_4100_);
lean_closure_set(v___f_4115_, 2, v_e_4070_);
lean_closure_set(v___f_4115_, 3, v_toFun_4081_);
if (v_isShared_4113_ == 0)
{
v___x_4117_ = v___x_4112_;
goto v_reusejp_4116_;
}
else
{
lean_object* v_reuseFailAlloc_4261_; 
v_reuseFailAlloc_4261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4261_, 0, v_toLE_4109_);
lean_ctor_set(v_reuseFailAlloc_4261_, 1, v_toLT_4110_);
v___x_4117_ = v_reuseFailAlloc_4261_;
goto v_reusejp_4116_;
}
v_reusejp_4116_:
{
lean_object* v___x_4119_; 
lean_inc_ref(v___f_4115_);
if (v_isShared_4108_ == 0)
{
lean_ctor_set(v___x_4107_, 1, v___f_4115_);
lean_ctor_set(v___x_4107_, 0, v___x_4117_);
v___x_4119_ = v___x_4107_;
goto v_reusejp_4118_;
}
else
{
lean_object* v_reuseFailAlloc_4260_; 
v_reuseFailAlloc_4260_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4260_, 0, v___x_4117_);
lean_ctor_set(v_reuseFailAlloc_4260_, 1, v___f_4115_);
v___x_4119_ = v_reuseFailAlloc_4260_;
goto v_reusejp_4118_;
}
v_reusejp_4118_:
{
lean_object* v_lattice_4121_; 
lean_inc_ref(v___f_4114_);
if (v_isShared_4099_ == 0)
{
lean_ctor_set(v___x_4098_, 1, v___f_4114_);
lean_ctor_set(v___x_4098_, 0, v___x_4119_);
v_lattice_4121_ = v___x_4098_;
goto v_reusejp_4120_;
}
else
{
lean_object* v_reuseFailAlloc_4259_; 
v_reuseFailAlloc_4259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4259_, 0, v___x_4119_);
lean_ctor_set(v_reuseFailAlloc_4259_, 1, v___f_4114_);
v_lattice_4121_ = v_reuseFailAlloc_4259_;
goto v_reusejp_4120_;
}
v_reusejp_4120_:
{
lean_object* v___x_4122_; lean_object* v_toPartialOrder_4123_; lean_object* v___x_4125_; uint8_t v_isShared_4126_; uint8_t v_isSharedCheck_4257_; 
v___x_4122_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_4121_);
v_toPartialOrder_4123_ = lean_ctor_get(v___x_4122_, 0);
v_isSharedCheck_4257_ = !lean_is_exclusive(v___x_4122_);
if (v_isSharedCheck_4257_ == 0)
{
lean_object* v_unused_4258_; 
v_unused_4258_ = lean_ctor_get(v___x_4122_, 1);
lean_dec(v_unused_4258_);
v___x_4125_ = v___x_4122_;
v_isShared_4126_ = v_isSharedCheck_4257_;
goto v_resetjp_4124_;
}
else
{
lean_inc(v_toPartialOrder_4123_);
lean_dec(v___x_4122_);
v___x_4125_ = lean_box(0);
v_isShared_4126_ = v_isSharedCheck_4257_;
goto v_resetjp_4124_;
}
v_resetjp_4124_:
{
lean_object* v_toLE_4127_; lean_object* v_toLT_4128_; lean_object* v___x_4130_; uint8_t v_isShared_4131_; uint8_t v_isSharedCheck_4256_; 
v_toLE_4127_ = lean_ctor_get(v_toPartialOrder_4123_, 0);
v_toLT_4128_ = lean_ctor_get(v_toPartialOrder_4123_, 1);
v_isSharedCheck_4256_ = !lean_is_exclusive(v_toPartialOrder_4123_);
if (v_isSharedCheck_4256_ == 0)
{
v___x_4130_ = v_toPartialOrder_4123_;
v_isShared_4131_ = v_isSharedCheck_4256_;
goto v_resetjp_4129_;
}
else
{
lean_inc(v_toLT_4128_);
lean_inc(v_toLE_4127_);
lean_dec(v_toPartialOrder_4123_);
v___x_4130_ = lean_box(0);
v_isShared_4131_ = v_isSharedCheck_4256_;
goto v_resetjp_4129_;
}
v_resetjp_4129_:
{
lean_object* v_top_4132_; lean_object* v_bot_4133_; lean_object* v_supSet_4134_; lean_object* v_infSet_4135_; lean_object* v___f_4136_; lean_object* v___x_4138_; 
lean_inc_n(v_toFun_4081_, 4);
v_top_4132_ = lean_apply_1(v_toFun_4081_, v_toOrderTop_4075_);
v_bot_4133_ = lean_apply_1(v_toFun_4081_, v_toOrderBot_4076_);
v_supSet_4134_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_4134_, 0, v_toSupSet_4086_);
lean_closure_set(v_supSet_4134_, 1, v_toFun_4081_);
v_infSet_4135_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_4135_, 0, v_toInfSet_4091_);
lean_closure_set(v_infSet_4135_, 1, v_toFun_4081_);
v___f_4136_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__6), 3, 1);
lean_closure_set(v___f_4136_, 0, v___f_4115_);
if (v_isShared_4131_ == 0)
{
v___x_4138_ = v___x_4130_;
goto v_reusejp_4137_;
}
else
{
lean_object* v_reuseFailAlloc_4255_; 
v_reuseFailAlloc_4255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4255_, 0, v_toLE_4127_);
lean_ctor_set(v_reuseFailAlloc_4255_, 1, v_toLT_4128_);
v___x_4138_ = v_reuseFailAlloc_4255_;
goto v_reusejp_4137_;
}
v_reusejp_4137_:
{
lean_object* v___x_4140_; 
lean_inc_ref(v___f_4136_);
if (v_isShared_4126_ == 0)
{
lean_ctor_set(v___x_4125_, 1, v___f_4136_);
lean_ctor_set(v___x_4125_, 0, v___x_4138_);
v___x_4140_ = v___x_4125_;
goto v_reusejp_4139_;
}
else
{
lean_object* v_reuseFailAlloc_4254_; 
v_reuseFailAlloc_4254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4254_, 0, v___x_4138_);
lean_ctor_set(v_reuseFailAlloc_4254_, 1, v___f_4136_);
v___x_4140_ = v_reuseFailAlloc_4254_;
goto v_reusejp_4139_;
}
v_reusejp_4139_:
{
lean_object* v___x_4142_; 
lean_inc_ref(v___f_4114_);
if (v_isShared_4094_ == 0)
{
lean_ctor_set(v___x_4093_, 1, v___f_4114_);
lean_ctor_set(v___x_4093_, 0, v___x_4140_);
v___x_4142_ = v___x_4093_;
goto v_reusejp_4141_;
}
else
{
lean_object* v_reuseFailAlloc_4253_; 
v_reuseFailAlloc_4253_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4253_, 0, v___x_4140_);
lean_ctor_set(v_reuseFailAlloc_4253_, 1, v___f_4114_);
v___x_4142_ = v_reuseFailAlloc_4253_;
goto v_reusejp_4141_;
}
v_reusejp_4141_:
{
lean_object* v___x_4144_; 
if (v_isShared_4079_ == 0)
{
lean_ctor_set(v___x_4078_, 1, v_bot_4133_);
lean_ctor_set(v___x_4078_, 0, v_top_4132_);
v___x_4144_ = v___x_4078_;
goto v_reusejp_4143_;
}
else
{
lean_object* v_reuseFailAlloc_4252_; 
v_reuseFailAlloc_4252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4252_, 0, v_top_4132_);
lean_ctor_set(v_reuseFailAlloc_4252_, 1, v_bot_4133_);
v___x_4144_ = v_reuseFailAlloc_4252_;
goto v_reusejp_4143_;
}
v_reusejp_4143_:
{
lean_object* v_completeLattice_4145_; lean_object* v___x_4146_; lean_object* v___x_4148_; uint8_t v_isShared_4149_; uint8_t v_isSharedCheck_4247_; 
v_completeLattice_4145_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_completeLattice_4145_, 0, v___x_4142_);
lean_ctor_set(v_completeLattice_4145_, 1, v_supSet_4134_);
lean_ctor_set(v_completeLattice_4145_, 2, v_infSet_4135_);
lean_ctor_set(v_completeLattice_4145_, 3, v___x_4144_);
v___x_4146_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_inst_4071_);
v_isSharedCheck_4247_ = !lean_is_exclusive(v_inst_4071_);
if (v_isSharedCheck_4247_ == 0)
{
lean_object* v_unused_4248_; lean_object* v_unused_4249_; lean_object* v_unused_4250_; lean_object* v_unused_4251_; 
v_unused_4248_ = lean_ctor_get(v_inst_4071_, 3);
lean_dec(v_unused_4248_);
v_unused_4249_ = lean_ctor_get(v_inst_4071_, 2);
lean_dec(v_unused_4249_);
v_unused_4250_ = lean_ctor_get(v_inst_4071_, 1);
lean_dec(v_unused_4250_);
v_unused_4251_ = lean_ctor_get(v_inst_4071_, 0);
lean_dec(v_unused_4251_);
v___x_4148_ = v_inst_4071_;
v_isShared_4149_ = v_isSharedCheck_4247_;
goto v_resetjp_4147_;
}
else
{
lean_dec(v_inst_4071_);
v___x_4148_ = lean_box(0);
v_isShared_4149_ = v_isSharedCheck_4247_;
goto v_resetjp_4147_;
}
v_resetjp_4147_:
{
lean_object* v_toCompl_4150_; lean_object* v_toHImp_4151_; lean_object* v_toTop_4152_; lean_object* v___x_4153_; lean_object* v_toSDiff_4154_; lean_object* v_toBot_4155_; lean_object* v___x_4156_; lean_object* v_toPartialOrder_4157_; lean_object* v_toInfSet_4158_; lean_object* v___x_4160_; uint8_t v_isShared_4161_; uint8_t v_isSharedCheck_4246_; 
v_toCompl_4150_ = lean_ctor_get(v___x_4146_, 1);
lean_inc(v_toCompl_4150_);
v_toHImp_4151_ = lean_ctor_get(v___x_4146_, 3);
lean_inc(v_toHImp_4151_);
v_toTop_4152_ = lean_ctor_get(v___x_4146_, 4);
lean_inc(v_toTop_4152_);
v___x_4153_ = lp_mathlib_BooleanAlgebra_toGeneralizedBooleanAlgebra___redArg(v___x_4146_);
lean_dec_ref(v___x_4146_);
v_toSDiff_4154_ = lean_ctor_get(v___x_4153_, 1);
lean_inc(v_toSDiff_4154_);
v_toBot_4155_ = lean_ctor_get(v___x_4153_, 2);
lean_inc(v_toBot_4155_);
lean_dec_ref(v___x_4153_);
lean_inc_ref(v_completeLattice_4145_);
v___x_4156_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_completeLattice_4145_);
v_toPartialOrder_4157_ = lean_ctor_get(v___x_4156_, 0);
v_toInfSet_4158_ = lean_ctor_get(v___x_4156_, 1);
v_isSharedCheck_4246_ = !lean_is_exclusive(v___x_4156_);
if (v_isSharedCheck_4246_ == 0)
{
v___x_4160_ = v___x_4156_;
v_isShared_4161_ = v_isSharedCheck_4246_;
goto v_resetjp_4159_;
}
else
{
lean_inc(v_toInfSet_4158_);
lean_inc(v_toPartialOrder_4157_);
lean_dec(v___x_4156_);
v___x_4160_ = lean_box(0);
v_isShared_4161_ = v_isSharedCheck_4246_;
goto v_resetjp_4159_;
}
v_resetjp_4159_:
{
lean_object* v_toLE_4162_; lean_object* v_toLT_4163_; lean_object* v___x_4165_; uint8_t v_isShared_4166_; uint8_t v_isSharedCheck_4245_; 
v_toLE_4162_ = lean_ctor_get(v_toPartialOrder_4157_, 0);
v_toLT_4163_ = lean_ctor_get(v_toPartialOrder_4157_, 1);
v_isSharedCheck_4245_ = !lean_is_exclusive(v_toPartialOrder_4157_);
if (v_isSharedCheck_4245_ == 0)
{
v___x_4165_ = v_toPartialOrder_4157_;
v_isShared_4166_ = v_isSharedCheck_4245_;
goto v_resetjp_4164_;
}
else
{
lean_inc(v_toLT_4163_);
lean_inc(v_toLE_4162_);
lean_dec(v_toPartialOrder_4157_);
v___x_4165_ = lean_box(0);
v_isShared_4166_ = v_isSharedCheck_4245_;
goto v_resetjp_4164_;
}
v_resetjp_4164_:
{
lean_object* v___x_4167_; lean_object* v_toSupSet_4168_; lean_object* v___x_4170_; uint8_t v_isShared_4171_; uint8_t v_isSharedCheck_4243_; 
v___x_4167_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_completeLattice_4145_);
v_toSupSet_4168_ = lean_ctor_get(v___x_4167_, 1);
v_isSharedCheck_4243_ = !lean_is_exclusive(v___x_4167_);
if (v_isSharedCheck_4243_ == 0)
{
lean_object* v_unused_4244_; 
v_unused_4244_ = lean_ctor_get(v___x_4167_, 0);
lean_dec(v_unused_4244_);
v___x_4170_ = v___x_4167_;
v_isShared_4171_ = v_isSharedCheck_4243_;
goto v_resetjp_4169_;
}
else
{
lean_inc(v_toSupSet_4168_);
lean_dec(v___x_4167_);
v___x_4170_ = lean_box(0);
v_isShared_4171_ = v_isSharedCheck_4243_;
goto v_resetjp_4169_;
}
v_resetjp_4169_:
{
lean_object* v_top_4172_; lean_object* v_himp_4173_; lean_object* v_compl_4174_; lean_object* v_sdiff_4175_; lean_object* v_bot_4176_; lean_object* v___x_4178_; 
lean_inc_n(v_toFun_4081_, 4);
v_top_4172_ = lean_apply_1(v_toFun_4081_, v_toTop_4152_);
lean_inc_ref_n(v_e_4070_, 2);
v_himp_4173_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__7), 6, 4);
lean_closure_set(v_himp_4173_, 0, v___f_4100_);
lean_closure_set(v_himp_4173_, 1, v_e_4070_);
lean_closure_set(v_himp_4173_, 2, v_toHImp_4151_);
lean_closure_set(v_himp_4173_, 3, v_toFun_4081_);
v_compl_4174_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_frame___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_4174_, 0, v_e_4070_);
lean_closure_set(v_compl_4174_, 1, v_toCompl_4150_);
lean_closure_set(v_compl_4174_, 2, v_toFun_4081_);
v_sdiff_4175_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coframe___redArg___lam__0), 6, 4);
lean_closure_set(v_sdiff_4175_, 0, v___f_4100_);
lean_closure_set(v_sdiff_4175_, 1, v_e_4070_);
lean_closure_set(v_sdiff_4175_, 2, v_toSDiff_4154_);
lean_closure_set(v_sdiff_4175_, 3, v_toFun_4081_);
v_bot_4176_ = lean_apply_1(v_toFun_4081_, v_toBot_4155_);
if (v_isShared_4166_ == 0)
{
v___x_4178_ = v___x_4165_;
goto v_reusejp_4177_;
}
else
{
lean_object* v_reuseFailAlloc_4242_; 
v_reuseFailAlloc_4242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4242_, 0, v_toLE_4162_);
lean_ctor_set(v_reuseFailAlloc_4242_, 1, v_toLT_4163_);
v___x_4178_ = v_reuseFailAlloc_4242_;
goto v_reusejp_4177_;
}
v_reusejp_4177_:
{
lean_object* v___x_4180_; 
lean_inc_ref(v___f_4136_);
if (v_isShared_4171_ == 0)
{
lean_ctor_set(v___x_4170_, 1, v___f_4136_);
lean_ctor_set(v___x_4170_, 0, v___x_4178_);
v___x_4180_ = v___x_4170_;
goto v_reusejp_4179_;
}
else
{
lean_object* v_reuseFailAlloc_4241_; 
v_reuseFailAlloc_4241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4241_, 0, v___x_4178_);
lean_ctor_set(v_reuseFailAlloc_4241_, 1, v___f_4136_);
v___x_4180_ = v_reuseFailAlloc_4241_;
goto v_reusejp_4179_;
}
v_reusejp_4179_:
{
lean_object* v___x_4182_; 
lean_inc_ref(v___f_4114_);
lean_inc_ref(v___x_4180_);
if (v_isShared_4161_ == 0)
{
lean_ctor_set(v___x_4160_, 1, v___f_4114_);
lean_ctor_set(v___x_4160_, 0, v___x_4180_);
v___x_4182_ = v___x_4160_;
goto v_reusejp_4181_;
}
else
{
lean_object* v_reuseFailAlloc_4240_; 
v_reuseFailAlloc_4240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4240_, 0, v___x_4180_);
lean_ctor_set(v_reuseFailAlloc_4240_, 1, v___f_4114_);
v___x_4182_ = v_reuseFailAlloc_4240_;
goto v_reusejp_4181_;
}
v_reusejp_4181_:
{
lean_object* v___x_4184_; 
if (v_isShared_4089_ == 0)
{
lean_ctor_set(v___x_4088_, 1, v_bot_4176_);
lean_ctor_set(v___x_4088_, 0, v_top_4172_);
v___x_4184_ = v___x_4088_;
goto v_reusejp_4183_;
}
else
{
lean_object* v_reuseFailAlloc_4239_; 
v_reuseFailAlloc_4239_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4239_, 0, v_top_4172_);
lean_ctor_set(v_reuseFailAlloc_4239_, 1, v_bot_4176_);
v___x_4184_ = v_reuseFailAlloc_4239_;
goto v_reusejp_4183_;
}
v_reusejp_4183_:
{
lean_object* v___x_4185_; lean_object* v_completeBooleanAlgebra_4187_; 
lean_inc_ref(v___x_4182_);
v___x_4185_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4185_, 0, v___x_4182_);
lean_ctor_set(v___x_4185_, 1, v_toSupSet_4168_);
lean_ctor_set(v___x_4185_, 2, v_toInfSet_4158_);
lean_ctor_set(v___x_4185_, 3, v___x_4184_);
lean_inc_ref(v___x_4185_);
if (v_isShared_4149_ == 0)
{
lean_ctor_set(v___x_4148_, 3, v_himp_4173_);
lean_ctor_set(v___x_4148_, 2, v_sdiff_4175_);
lean_ctor_set(v___x_4148_, 1, v_compl_4174_);
lean_ctor_set(v___x_4148_, 0, v___x_4185_);
v_completeBooleanAlgebra_4187_ = v___x_4148_;
goto v_reusejp_4186_;
}
else
{
lean_object* v_reuseFailAlloc_4238_; 
v_reuseFailAlloc_4238_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_4238_, 0, v___x_4185_);
lean_ctor_set(v_reuseFailAlloc_4238_, 1, v_compl_4174_);
lean_ctor_set(v_reuseFailAlloc_4238_, 2, v_sdiff_4175_);
lean_ctor_set(v_reuseFailAlloc_4238_, 3, v_himp_4173_);
v_completeBooleanAlgebra_4187_ = v_reuseFailAlloc_4238_;
goto v_reusejp_4186_;
}
v_reusejp_4186_:
{
lean_object* v___x_4188_; lean_object* v___x_4189_; lean_object* v_toPartialOrder_4190_; lean_object* v_toInfSet_4191_; lean_object* v___x_4193_; uint8_t v_isShared_4194_; uint8_t v_isSharedCheck_4237_; 
v___x_4188_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_4182_);
lean_inc_ref(v___x_4185_);
v___x_4189_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v___x_4185_);
v_toPartialOrder_4190_ = lean_ctor_get(v___x_4189_, 0);
v_toInfSet_4191_ = lean_ctor_get(v___x_4189_, 1);
v_isSharedCheck_4237_ = !lean_is_exclusive(v___x_4189_);
if (v_isSharedCheck_4237_ == 0)
{
v___x_4193_ = v___x_4189_;
v_isShared_4194_ = v_isSharedCheck_4237_;
goto v_resetjp_4192_;
}
else
{
lean_inc(v_toInfSet_4191_);
lean_inc(v_toPartialOrder_4190_);
lean_dec(v___x_4189_);
v___x_4193_ = lean_box(0);
v_isShared_4194_ = v_isSharedCheck_4237_;
goto v_resetjp_4192_;
}
v_resetjp_4192_:
{
lean_object* v_toLE_4195_; lean_object* v_toLT_4196_; lean_object* v___x_4198_; uint8_t v_isShared_4199_; uint8_t v_isSharedCheck_4236_; 
v_toLE_4195_ = lean_ctor_get(v_toPartialOrder_4190_, 0);
v_toLT_4196_ = lean_ctor_get(v_toPartialOrder_4190_, 1);
v_isSharedCheck_4236_ = !lean_is_exclusive(v_toPartialOrder_4190_);
if (v_isSharedCheck_4236_ == 0)
{
v___x_4198_ = v_toPartialOrder_4190_;
v_isShared_4199_ = v_isSharedCheck_4236_;
goto v_resetjp_4197_;
}
else
{
lean_inc(v_toLT_4196_);
lean_inc(v_toLE_4195_);
lean_dec(v_toPartialOrder_4190_);
v___x_4198_ = lean_box(0);
v_isShared_4199_ = v_isSharedCheck_4236_;
goto v_resetjp_4197_;
}
v_resetjp_4197_:
{
lean_object* v___x_4200_; lean_object* v_toSupSet_4201_; lean_object* v___x_4203_; uint8_t v_isShared_4204_; uint8_t v_isSharedCheck_4234_; 
v___x_4200_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v___x_4185_);
v_toSupSet_4201_ = lean_ctor_get(v___x_4200_, 1);
v_isSharedCheck_4234_ = !lean_is_exclusive(v___x_4200_);
if (v_isSharedCheck_4234_ == 0)
{
lean_object* v_unused_4235_; 
v_unused_4235_ = lean_ctor_get(v___x_4200_, 0);
lean_dec(v_unused_4235_);
v___x_4203_ = v___x_4200_;
v_isShared_4204_ = v_isSharedCheck_4234_;
goto v_resetjp_4202_;
}
else
{
lean_inc(v_toSupSet_4201_);
lean_dec(v___x_4200_);
v___x_4203_ = lean_box(0);
v_isShared_4204_ = v_isSharedCheck_4234_;
goto v_resetjp_4202_;
}
v_resetjp_4202_:
{
lean_object* v___x_4205_; lean_object* v_toCompl_4206_; lean_object* v_toSDiff_4207_; lean_object* v_toHImp_4208_; lean_object* v_toTop_4209_; lean_object* v_toBot_4210_; lean_object* v___x_4211_; lean_object* v___x_4212_; lean_object* v___x_4213_; lean_object* v_toHNot_4214_; lean_object* v___f_4215_; lean_object* v___f_4216_; lean_object* v___x_4218_; 
v___x_4205_ = lp_mathlib_CompleteBooleanAlgebra_toBooleanAlgebra___redArg(v_completeBooleanAlgebra_4187_);
v_toCompl_4206_ = lean_ctor_get(v___x_4205_, 1);
lean_inc(v_toCompl_4206_);
v_toSDiff_4207_ = lean_ctor_get(v___x_4205_, 2);
lean_inc(v_toSDiff_4207_);
v_toHImp_4208_ = lean_ctor_get(v___x_4205_, 3);
lean_inc(v_toHImp_4208_);
v_toTop_4209_ = lean_ctor_get(v___x_4205_, 4);
lean_inc(v_toTop_4209_);
v_toBot_4210_ = lean_ctor_get(v___x_4205_, 5);
lean_inc(v_toBot_4210_);
lean_dec_ref(v___x_4205_);
v___x_4211_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteDistribLattice___redArg(v_completeBooleanAlgebra_4187_);
lean_dec_ref(v_completeBooleanAlgebra_4187_);
v___x_4212_ = lp_mathlib_CompleteDistribLattice_toCoframe___redArg(v___x_4211_);
v___x_4213_ = lp_mathlib_Order_Coframe_toCoheytingAlgebra___redArg(v___x_4212_);
v_toHNot_4214_ = lean_ctor_get(v___x_4213_, 2);
lean_inc(v_toHNot_4214_);
lean_dec_ref(v___x_4213_);
v___f_4215_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_4215_, 0, v___x_4180_);
v___f_4216_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_4216_, 0, v___x_4188_);
if (v_isShared_4199_ == 0)
{
v___x_4218_ = v___x_4198_;
goto v_reusejp_4217_;
}
else
{
lean_object* v_reuseFailAlloc_4233_; 
v_reuseFailAlloc_4233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4233_, 0, v_toLE_4195_);
lean_ctor_set(v_reuseFailAlloc_4233_, 1, v_toLT_4196_);
v___x_4218_ = v_reuseFailAlloc_4233_;
goto v_reusejp_4217_;
}
v_reusejp_4217_:
{
lean_object* v___x_4220_; 
if (v_isShared_4204_ == 0)
{
lean_ctor_set(v___x_4203_, 1, v___f_4136_);
lean_ctor_set(v___x_4203_, 0, v___x_4218_);
v___x_4220_ = v___x_4203_;
goto v_reusejp_4219_;
}
else
{
lean_object* v_reuseFailAlloc_4232_; 
v_reuseFailAlloc_4232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4232_, 0, v___x_4218_);
lean_ctor_set(v_reuseFailAlloc_4232_, 1, v___f_4136_);
v___x_4220_ = v_reuseFailAlloc_4232_;
goto v_reusejp_4219_;
}
v_reusejp_4219_:
{
lean_object* v___x_4222_; 
if (v_isShared_4194_ == 0)
{
lean_ctor_set(v___x_4193_, 1, v___f_4114_);
lean_ctor_set(v___x_4193_, 0, v___x_4220_);
v___x_4222_ = v___x_4193_;
goto v_reusejp_4221_;
}
else
{
lean_object* v_reuseFailAlloc_4231_; 
v_reuseFailAlloc_4231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4231_, 0, v___x_4220_);
lean_ctor_set(v_reuseFailAlloc_4231_, 1, v___f_4114_);
v___x_4222_ = v_reuseFailAlloc_4231_;
goto v_reusejp_4221_;
}
v_reusejp_4221_:
{
lean_object* v___x_4224_; 
lean_inc(v_toBot_4210_);
lean_inc(v_toTop_4209_);
if (v_isShared_4084_ == 0)
{
lean_ctor_set(v___x_4083_, 1, v_toBot_4210_);
lean_ctor_set(v___x_4083_, 0, v_toTop_4209_);
v___x_4224_ = v___x_4083_;
goto v_reusejp_4223_;
}
else
{
lean_object* v_reuseFailAlloc_4230_; 
v_reuseFailAlloc_4230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4230_, 0, v_toTop_4209_);
lean_ctor_set(v_reuseFailAlloc_4230_, 1, v_toBot_4210_);
v___x_4224_ = v_reuseFailAlloc_4230_;
goto v_reusejp_4223_;
}
v_reusejp_4223_:
{
lean_object* v___x_4225_; lean_object* v___x_4226_; lean_object* v_toGeneralizedCoheytingAlgebra_4227_; lean_object* v_toSDiff_4228_; lean_object* v___x_4229_; 
v___x_4225_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4225_, 0, v___x_4222_);
lean_ctor_set(v___x_4225_, 1, v_toSupSet_4201_);
lean_ctor_set(v___x_4225_, 2, v_toInfSet_4191_);
lean_ctor_set(v___x_4225_, 3, v___x_4224_);
v___x_4226_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_4216_, v___f_4215_, v_toLE_4195_, v_toLT_4196_, v_toBot_4210_, v_toTop_4209_, v_toHNot_4214_, v_toSDiff_4207_);
v_toGeneralizedCoheytingAlgebra_4227_ = lean_ctor_get(v___x_4226_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_4227_);
lean_dec_ref(v___x_4226_);
v_toSDiff_4228_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_4227_, 2);
lean_inc(v_toSDiff_4228_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_4227_);
v___x_4229_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4229_, 0, v___x_4225_);
lean_ctor_set(v___x_4229_, 1, v_toCompl_4206_);
lean_ctor_set(v___x_4229_, 2, v_toSDiff_4228_);
lean_ctor_set(v___x_4229_, 3, v_toHImp_4208_);
return v___x_4229_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteBooleanAlgebra___closed__0(void){
_start:
{
lean_object* v___x_4273_; lean_object* v___x_4274_; 
v___x_4273_ = lp_mathlib_PUnit_instCompleteLinearOrder;
v___x_4274_ = lp_mathlib_CompleteLinearOrder_toCompletelyDistribLattice___redArg(v___x_4273_);
return v___x_4274_;
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteBooleanAlgebra(void){
_start:
{
lean_object* v___x_4275_; lean_object* v_toCompleteLattice_4276_; lean_object* v___x_4277_; lean_object* v_toCompl_4278_; lean_object* v_toSDiff_4279_; lean_object* v_toHImp_4280_; lean_object* v___x_4281_; 
v___x_4275_ = lean_obj_once(&lp_mathlib_PUnit_instCompleteBooleanAlgebra___closed__0, &lp_mathlib_PUnit_instCompleteBooleanAlgebra___closed__0_once, _init_lp_mathlib_PUnit_instCompleteBooleanAlgebra___closed__0);
v_toCompleteLattice_4276_ = lean_ctor_get(v___x_4275_, 0);
v___x_4277_ = lp_mathlib_PUnit_instBooleanAlgebra;
v_toCompl_4278_ = lean_ctor_get(v___x_4277_, 1);
v_toSDiff_4279_ = lean_ctor_get(v___x_4277_, 2);
v_toHImp_4280_ = lean_ctor_get(v___x_4277_, 3);
lean_inc(v_toHImp_4280_);
lean_inc(v_toSDiff_4279_);
lean_inc(v_toCompl_4278_);
lean_inc_ref(v_toCompleteLattice_4276_);
v___x_4281_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4281_, 0, v_toCompleteLattice_4276_);
lean_ctor_set(v___x_4281_, 1, v_toCompl_4278_);
lean_ctor_set(v___x_4281_, 2, v_toSDiff_4279_);
lean_ctor_set(v___x_4281_, 3, v_toHImp_4280_);
return v___x_4281_;
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteAtomicBooleanAlgebra(void){
_start:
{
lean_object* v___x_4282_; 
v___x_4282_ = lp_mathlib_PUnit_instCompleteBooleanAlgebra;
return v___x_4282_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Pairwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra = _init_lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra();
lean_mark_persistent(lp_mathlib_Prop_instCompleteAtomicBooleanAlgebra);
lp_mathlib_Prop_instCompleteBooleanAlgebra = _init_lp_mathlib_Prop_instCompleteBooleanAlgebra();
lean_mark_persistent(lp_mathlib_Prop_instCompleteBooleanAlgebra);
lp_mathlib_PUnit_instCompleteBooleanAlgebra = _init_lp_mathlib_PUnit_instCompleteBooleanAlgebra();
lean_mark_persistent(lp_mathlib_PUnit_instCompleteBooleanAlgebra);
lp_mathlib_PUnit_instCompleteAtomicBooleanAlgebra = _init_lp_mathlib_PUnit_instCompleteAtomicBooleanAlgebra();
lean_mark_persistent(lp_mathlib_PUnit_instCompleteAtomicBooleanAlgebra);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Pairwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(builtin);
}
#ifdef __cplusplus
}
#endif
