// Lean compiler output
// Module: Mathlib.Order.Heyting.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.PropInstances public import Mathlib.Order.GaloisConnection.Defs
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
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
extern lean_object* lp_mathlib_Prop_instDistribLattice;
lean_object* lp_mathlib_Pi_instLattice___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instOrderTop___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instHImp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instOrderBot___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instCompl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instLattice___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeInf_toMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeSup_toMax___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instLattice___redArg(lean_object*, lean_object*);
extern lean_object* lp_mathlib_PUnit_instLinearOrder;
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0 = (const lean_object*)&lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedHeytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedCoheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedCoheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__2(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instHeytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHeytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHeytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Prop_instHeytingAlgebra___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prop_instHeytingAlgebra___closed__0;
static lean_once_cell_t lp_mathlib_Prop_instHeytingAlgebra___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prop_instHeytingAlgebra___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Prop_instHeytingAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBiheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBiheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBiheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBiheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedCoheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_heytingAlgebra___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_heytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_heytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coheytingAlgebra___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_biheytingAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_biheytingAlgebra___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_biheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_biheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PUnit_instBiheytingAlgebra___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instBiheytingAlgebra___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___closed__0 = (const lean_object*)&lp_mathlib_PUnit_instBiheytingAlgebra___closed__0_value;
static const lean_closure_object lp_mathlib_PUnit_instBiheytingAlgebra___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_instBiheytingAlgebra___lam__1, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___closed__1 = (const lean_object*)&lp_mathlib_PUnit_instBiheytingAlgebra___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_a_3_, lean_object* v_b_4_){
_start:
{
lean_object* v_fst_5_; lean_object* v_snd_6_; lean_object* v_fst_7_; lean_object* v_snd_8_; lean_object* v___x_10_; uint8_t v_isShared_11_; uint8_t v_isSharedCheck_17_; 
v_fst_5_ = lean_ctor_get(v_a_3_, 0);
lean_inc(v_fst_5_);
v_snd_6_ = lean_ctor_get(v_a_3_, 1);
lean_inc(v_snd_6_);
lean_dec_ref(v_a_3_);
v_fst_7_ = lean_ctor_get(v_b_4_, 0);
v_snd_8_ = lean_ctor_get(v_b_4_, 1);
v_isSharedCheck_17_ = !lean_is_exclusive(v_b_4_);
if (v_isSharedCheck_17_ == 0)
{
v___x_10_ = v_b_4_;
v_isShared_11_ = v_isSharedCheck_17_;
goto v_resetjp_9_;
}
else
{
lean_inc(v_snd_8_);
lean_inc(v_fst_7_);
lean_dec(v_b_4_);
v___x_10_ = lean_box(0);
v_isShared_11_ = v_isSharedCheck_17_;
goto v_resetjp_9_;
}
v_resetjp_9_:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_15_; 
v___x_12_ = lean_apply_2(v_inst_1_, v_fst_5_, v_fst_7_);
v___x_13_ = lean_apply_2(v_inst_2_, v_snd_6_, v_snd_8_);
if (v_isShared_11_ == 0)
{
lean_ctor_set(v___x_10_, 1, v___x_13_);
lean_ctor_set(v___x_10_, 0, v___x_12_);
v___x_15_ = v___x_10_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v___x_12_);
lean_ctor_set(v_reuseFailAlloc_16_, 1, v___x_13_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp___redArg(lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(v___f_20_, 0, v_inst_18_);
lean_closure_set(v___f_20_, 1, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp(lean_object* v_00_u03b1_21_, lean_object* v_00_u03b2_22_, lean_object* v_inst_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(v___f_25_, 0, v_inst_23_);
lean_closure_set(v___f_25_, 1, v_inst_24_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff___redArg___lam__0(lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_b_28_, lean_object* v_a_29_){
_start:
{
lean_object* v_fst_30_; lean_object* v_snd_31_; lean_object* v_fst_32_; lean_object* v_snd_33_; lean_object* v___x_35_; uint8_t v_isShared_36_; uint8_t v_isSharedCheck_42_; 
v_fst_30_ = lean_ctor_get(v_b_28_, 0);
lean_inc(v_fst_30_);
v_snd_31_ = lean_ctor_get(v_b_28_, 1);
lean_inc(v_snd_31_);
lean_dec_ref(v_b_28_);
v_fst_32_ = lean_ctor_get(v_a_29_, 0);
v_snd_33_ = lean_ctor_get(v_a_29_, 1);
v_isSharedCheck_42_ = !lean_is_exclusive(v_a_29_);
if (v_isSharedCheck_42_ == 0)
{
v___x_35_ = v_a_29_;
v_isShared_36_ = v_isSharedCheck_42_;
goto v_resetjp_34_;
}
else
{
lean_inc(v_snd_33_);
lean_inc(v_fst_32_);
lean_dec(v_a_29_);
v___x_35_ = lean_box(0);
v_isShared_36_ = v_isSharedCheck_42_;
goto v_resetjp_34_;
}
v_resetjp_34_:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_40_; 
v___x_37_ = lean_apply_2(v_inst_26_, v_fst_30_, v_fst_32_);
v___x_38_ = lean_apply_2(v_inst_27_, v_snd_31_, v_snd_33_);
if (v_isShared_36_ == 0)
{
lean_ctor_set(v___x_35_, 1, v___x_38_);
lean_ctor_set(v___x_35_, 0, v___x_37_);
v___x_40_ = v___x_35_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v___x_37_);
lean_ctor_set(v_reuseFailAlloc_41_, 1, v___x_38_);
v___x_40_ = v_reuseFailAlloc_41_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
return v___x_40_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff___redArg(lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___f_45_; 
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSDiff___redArg___lam__0), 4, 2);
lean_closure_set(v___f_45_, 0, v_inst_43_);
lean_closure_set(v___f_45_, 1, v_inst_44_);
return v___f_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff(lean_object* v_00_u03b1_46_, lean_object* v_00_u03b2_47_, lean_object* v_inst_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v___f_50_; 
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSDiff___redArg___lam__0), 4, 2);
lean_closure_set(v___f_50_, 0, v_inst_48_);
lean_closure_set(v___f_50_, 1, v_inst_49_);
return v___f_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot___redArg___lam__0(lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_a_53_){
_start:
{
lean_object* v_fst_54_; lean_object* v_snd_55_; lean_object* v___x_57_; uint8_t v_isShared_58_; uint8_t v_isSharedCheck_64_; 
v_fst_54_ = lean_ctor_get(v_a_53_, 0);
v_snd_55_ = lean_ctor_get(v_a_53_, 1);
v_isSharedCheck_64_ = !lean_is_exclusive(v_a_53_);
if (v_isSharedCheck_64_ == 0)
{
v___x_57_ = v_a_53_;
v_isShared_58_ = v_isSharedCheck_64_;
goto v_resetjp_56_;
}
else
{
lean_inc(v_snd_55_);
lean_inc(v_fst_54_);
lean_dec(v_a_53_);
v___x_57_ = lean_box(0);
v_isShared_58_ = v_isSharedCheck_64_;
goto v_resetjp_56_;
}
v_resetjp_56_:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_62_; 
v___x_59_ = lean_apply_1(v_inst_51_, v_fst_54_);
v___x_60_ = lean_apply_1(v_inst_52_, v_snd_55_);
if (v_isShared_58_ == 0)
{
lean_ctor_set(v___x_57_, 1, v___x_60_);
lean_ctor_set(v___x_57_, 0, v___x_59_);
v___x_62_ = v___x_57_;
goto v_reusejp_61_;
}
else
{
lean_object* v_reuseFailAlloc_63_; 
v_reuseFailAlloc_63_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_63_, 0, v___x_59_);
lean_ctor_set(v_reuseFailAlloc_63_, 1, v___x_60_);
v___x_62_ = v_reuseFailAlloc_63_;
goto v_reusejp_61_;
}
v_reusejp_61_:
{
return v___x_62_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot___redArg(lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___f_67_; 
v___f_67_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_67_, 0, v_inst_65_);
lean_closure_set(v___f_67_, 1, v_inst_66_);
return v___f_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot(lean_object* v_00_u03b1_68_, lean_object* v_00_u03b2_69_, lean_object* v_inst_70_, lean_object* v_inst_71_){
_start:
{
lean_object* v___f_72_; 
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_72_, 0, v_inst_70_);
lean_closure_set(v___f_72_, 1, v_inst_71_);
return v___f_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompl___redArg(lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___f_75_; 
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_75_, 0, v_inst_73_);
lean_closure_set(v___f_75_, 1, v_inst_74_);
return v___f_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompl(lean_object* v_00_u03b1_76_, lean_object* v_00_u03b2_77_, lean_object* v_inst_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v___f_80_; 
v___f_80_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_80_, 0, v_inst_78_);
lean_closure_set(v___f_80_, 1, v_inst_79_);
return v___f_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(lean_object* v_self_81_){
_start:
{
lean_object* v_toHeytingAlgebra_82_; lean_object* v_toGeneralizedHeytingAlgebra_83_; lean_object* v_toSDiff_84_; lean_object* v_toHNot_85_; lean_object* v_toOrderBot_86_; lean_object* v___x_88_; uint8_t v_isShared_89_; uint8_t v_isSharedCheck_103_; 
v_toHeytingAlgebra_82_ = lean_ctor_get(v_self_81_, 0);
lean_inc_ref(v_toHeytingAlgebra_82_);
v_toGeneralizedHeytingAlgebra_83_ = lean_ctor_get(v_toHeytingAlgebra_82_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_83_);
v_toSDiff_84_ = lean_ctor_get(v_self_81_, 1);
lean_inc(v_toSDiff_84_);
v_toHNot_85_ = lean_ctor_get(v_self_81_, 2);
lean_inc(v_toHNot_85_);
lean_dec_ref(v_self_81_);
v_toOrderBot_86_ = lean_ctor_get(v_toHeytingAlgebra_82_, 1);
v_isSharedCheck_103_ = !lean_is_exclusive(v_toHeytingAlgebra_82_);
if (v_isSharedCheck_103_ == 0)
{
lean_object* v_unused_104_; lean_object* v_unused_105_; 
v_unused_104_ = lean_ctor_get(v_toHeytingAlgebra_82_, 2);
lean_dec(v_unused_104_);
v_unused_105_ = lean_ctor_get(v_toHeytingAlgebra_82_, 0);
lean_dec(v_unused_105_);
v___x_88_ = v_toHeytingAlgebra_82_;
v_isShared_89_ = v_isSharedCheck_103_;
goto v_resetjp_87_;
}
else
{
lean_inc(v_toOrderBot_86_);
lean_dec(v_toHeytingAlgebra_82_);
v___x_88_ = lean_box(0);
v_isShared_89_ = v_isSharedCheck_103_;
goto v_resetjp_87_;
}
v_resetjp_87_:
{
lean_object* v_toLattice_90_; lean_object* v_toOrderTop_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_101_; 
v_toLattice_90_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_83_, 0);
v_toOrderTop_91_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_83_, 1);
v_isSharedCheck_101_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_83_);
if (v_isSharedCheck_101_ == 0)
{
lean_object* v_unused_102_; 
v_unused_102_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_83_, 2);
lean_dec(v_unused_102_);
v___x_93_ = v_toGeneralizedHeytingAlgebra_83_;
v_isShared_94_ = v_isSharedCheck_101_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_toOrderTop_91_);
lean_inc(v_toLattice_90_);
lean_dec(v_toGeneralizedHeytingAlgebra_83_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_101_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
lean_object* v___x_96_; 
if (v_isShared_94_ == 0)
{
lean_ctor_set(v___x_93_, 2, v_toSDiff_84_);
lean_ctor_set(v___x_93_, 1, v_toOrderBot_86_);
v___x_96_ = v___x_93_;
goto v_reusejp_95_;
}
else
{
lean_object* v_reuseFailAlloc_100_; 
v_reuseFailAlloc_100_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_100_, 0, v_toLattice_90_);
lean_ctor_set(v_reuseFailAlloc_100_, 1, v_toOrderBot_86_);
lean_ctor_set(v_reuseFailAlloc_100_, 2, v_toSDiff_84_);
v___x_96_ = v_reuseFailAlloc_100_;
goto v_reusejp_95_;
}
v_reusejp_95_:
{
lean_object* v___x_98_; 
if (v_isShared_89_ == 0)
{
lean_ctor_set(v___x_88_, 2, v_toHNot_85_);
lean_ctor_set(v___x_88_, 1, v_toOrderTop_91_);
lean_ctor_set(v___x_88_, 0, v___x_96_);
v___x_98_ = v___x_88_;
goto v_reusejp_97_;
}
else
{
lean_object* v_reuseFailAlloc_99_; 
v_reuseFailAlloc_99_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_99_, 0, v___x_96_);
lean_ctor_set(v_reuseFailAlloc_99_, 1, v_toOrderTop_91_);
lean_ctor_set(v_reuseFailAlloc_99_, 2, v_toHNot_85_);
v___x_98_ = v_reuseFailAlloc_99_;
goto v_reusejp_97_;
}
v_reusejp_97_:
{
return v___x_98_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra(lean_object* v_00_u03b1_106_, lean_object* v_self_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_self_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(lean_object* v_inst_109_){
_start:
{
lean_object* v_toGeneralizedHeytingAlgebra_110_; lean_object* v_toOrderBot_111_; lean_object* v_toOrderTop_112_; lean_object* v___x_113_; 
v_toGeneralizedHeytingAlgebra_110_ = lean_ctor_get(v_inst_109_, 0);
v_toOrderBot_111_ = lean_ctor_get(v_inst_109_, 1);
v_toOrderTop_112_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_110_, 1);
lean_inc(v_toOrderBot_111_);
lean_inc(v_toOrderTop_112_);
v___x_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_113_, 0, v_toOrderTop_112_);
lean_ctor_set(v___x_113_, 1, v_toOrderBot_111_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg___boxed(lean_object* v_inst_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(v_inst_114_);
lean_dec_ref(v_inst_114_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder(lean_object* v_00_u03b1_116_, lean_object* v_inst_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(v_inst_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___boxed(lean_object* v_00_u03b1_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_HeytingAlgebra_toBoundedOrder(v_00_u03b1_119_, v_inst_120_);
lean_dec_ref(v_inst_120_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(lean_object* v_inst_122_){
_start:
{
lean_object* v_toGeneralizedCoheytingAlgebra_123_; lean_object* v_toOrderTop_124_; lean_object* v_toOrderBot_125_; lean_object* v___x_126_; 
v_toGeneralizedCoheytingAlgebra_123_ = lean_ctor_get(v_inst_122_, 0);
v_toOrderTop_124_ = lean_ctor_get(v_inst_122_, 1);
v_toOrderBot_125_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_123_, 1);
lean_inc(v_toOrderBot_125_);
lean_inc(v_toOrderTop_124_);
v___x_126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_126_, 0, v_toOrderTop_124_);
lean_ctor_set(v___x_126_, 1, v_toOrderBot_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg___boxed(lean_object* v_inst_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(v_inst_127_);
lean_dec_ref(v_inst_127_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder(lean_object* v_00_u03b1_129_, lean_object* v_inst_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(v_inst_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___boxed(lean_object* v_00_u03b1_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_CoheytingAlgebra_toBoundedOrder(v_00_u03b1_132_, v_inst_133_);
lean_dec_ref(v_inst_133_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp___redArg___lam__0(lean_object* v_himp_135_, lean_object* v_toOrderBot_136_, lean_object* v_a_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lean_apply_2(v_himp_135_, v_a_137_, v_toOrderBot_136_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp___redArg(lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_himp_141_){
_start:
{
lean_object* v_toOrderTop_142_; lean_object* v_toOrderBot_143_; lean_object* v___f_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v_toOrderTop_142_ = lean_ctor_get(v_inst_140_, 0);
lean_inc(v_toOrderTop_142_);
v_toOrderBot_143_ = lean_ctor_get(v_inst_140_, 1);
lean_inc_n(v_toOrderBot_143_, 2);
lean_dec_ref(v_inst_140_);
lean_inc(v_himp_141_);
v___f_144_ = lean_alloc_closure((void*)(lp_mathlib_HeytingAlgebra_ofHImp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_144_, 0, v_himp_141_);
lean_closure_set(v___f_144_, 1, v_toOrderBot_143_);
v___x_145_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_145_, 0, v_inst_139_);
lean_ctor_set(v___x_145_, 1, v_toOrderTop_142_);
lean_ctor_set(v___x_145_, 2, v_himp_141_);
v___x_146_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
lean_ctor_set(v___x_146_, 1, v_toOrderBot_143_);
lean_ctor_set(v___x_146_, 2, v___f_144_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp(lean_object* v_00_u03b1_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_himp_150_, lean_object* v_le__himp__iff_151_){
_start:
{
lean_object* v_toOrderTop_152_; lean_object* v_toOrderBot_153_; lean_object* v___f_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v_toOrderTop_152_ = lean_ctor_get(v_inst_149_, 0);
lean_inc(v_toOrderTop_152_);
v_toOrderBot_153_ = lean_ctor_get(v_inst_149_, 1);
lean_inc_n(v_toOrderBot_153_, 2);
lean_dec_ref(v_inst_149_);
lean_inc(v_himp_150_);
v___f_154_ = lean_alloc_closure((void*)(lp_mathlib_HeytingAlgebra_ofHImp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_154_, 0, v_himp_150_);
lean_closure_set(v___f_154_, 1, v_toOrderBot_153_);
v___x_155_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_155_, 0, v_inst_148_);
lean_ctor_set(v___x_155_, 1, v_toOrderTop_152_);
lean_ctor_set(v___x_155_, 2, v_himp_150_);
v___x_156_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
lean_ctor_set(v___x_156_, 1, v_toOrderBot_153_);
lean_ctor_set(v___x_156_, 2, v___f_154_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg___lam__0(lean_object* v_toSemilatticeSup_157_, lean_object* v_compl_158_, lean_object* v_x1_159_, lean_object* v_x2_160_){
_start:
{
lean_object* v_sup_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v_sup_161_ = lean_ctor_get(v_toSemilatticeSup_157_, 1);
lean_inc(v_sup_161_);
lean_dec_ref(v_toSemilatticeSup_157_);
v___x_162_ = lean_apply_1(v_compl_158_, v_x1_159_);
v___x_163_ = lean_apply_2(v_sup_161_, v___x_162_, v_x2_160_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg(lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_compl_166_){
_start:
{
lean_object* v_toOrderTop_167_; lean_object* v_toOrderBot_168_; lean_object* v_toSemilatticeSup_169_; lean_object* v___f_170_; lean_object* v___x_171_; lean_object* v___x_172_; 
v_toOrderTop_167_ = lean_ctor_get(v_inst_165_, 0);
v_toOrderBot_168_ = lean_ctor_get(v_inst_165_, 1);
v_toSemilatticeSup_169_ = lean_ctor_get(v_inst_164_, 0);
lean_inc(v_compl_166_);
lean_inc_ref(v_toSemilatticeSup_169_);
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_HeytingAlgebra_ofCompl___redArg___lam__0), 4, 2);
lean_closure_set(v___f_170_, 0, v_toSemilatticeSup_169_);
lean_closure_set(v___f_170_, 1, v_compl_166_);
lean_inc(v_toOrderTop_167_);
v___x_171_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_171_, 0, v_inst_164_);
lean_ctor_set(v___x_171_, 1, v_toOrderTop_167_);
lean_ctor_set(v___x_171_, 2, v___f_170_);
lean_inc(v_toOrderBot_168_);
v___x_172_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v_toOrderBot_168_);
lean_ctor_set(v___x_172_, 2, v_compl_166_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg___boxed(lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_compl_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_HeytingAlgebra_ofCompl___redArg(v_inst_173_, v_inst_174_, v_compl_175_);
lean_dec_ref(v_inst_174_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl(lean_object* v_00_u03b1_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_compl_180_, lean_object* v_le__himp__iff_181_){
_start:
{
lean_object* v_toOrderTop_182_; lean_object* v_toOrderBot_183_; lean_object* v_toSemilatticeSup_184_; lean_object* v___f_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v_toOrderTop_182_ = lean_ctor_get(v_inst_179_, 0);
v_toOrderBot_183_ = lean_ctor_get(v_inst_179_, 1);
v_toSemilatticeSup_184_ = lean_ctor_get(v_inst_178_, 0);
lean_inc(v_compl_180_);
lean_inc_ref(v_toSemilatticeSup_184_);
v___f_185_ = lean_alloc_closure((void*)(lp_mathlib_HeytingAlgebra_ofCompl___redArg___lam__0), 4, 2);
lean_closure_set(v___f_185_, 0, v_toSemilatticeSup_184_);
lean_closure_set(v___f_185_, 1, v_compl_180_);
lean_inc(v_toOrderTop_182_);
v___x_186_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_186_, 0, v_inst_178_);
lean_ctor_set(v___x_186_, 1, v_toOrderTop_182_);
lean_ctor_set(v___x_186_, 2, v___f_185_);
lean_inc(v_toOrderBot_183_);
v___x_187_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_187_, 0, v___x_186_);
lean_ctor_set(v___x_187_, 1, v_toOrderBot_183_);
lean_ctor_set(v___x_187_, 2, v_compl_180_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___boxed(lean_object* v_00_u03b1_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_compl_191_, lean_object* v_le__himp__iff_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_HeytingAlgebra_ofCompl(v_00_u03b1_188_, v_inst_189_, v_inst_190_, v_compl_191_, v_le__himp__iff_192_);
lean_dec_ref(v_inst_190_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff___redArg___lam__0(lean_object* v_sdiff_194_, lean_object* v_toOrderTop_195_, lean_object* v_a_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lean_apply_2(v_sdiff_194_, v_toOrderTop_195_, v_a_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff___redArg(lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_sdiff_200_){
_start:
{
lean_object* v_toOrderTop_201_; lean_object* v_toOrderBot_202_; lean_object* v___f_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v_toOrderTop_201_ = lean_ctor_get(v_inst_199_, 0);
lean_inc_n(v_toOrderTop_201_, 2);
v_toOrderBot_202_ = lean_ctor_get(v_inst_199_, 1);
lean_inc(v_toOrderBot_202_);
lean_dec_ref(v_inst_199_);
lean_inc(v_sdiff_200_);
v___f_203_ = lean_alloc_closure((void*)(lp_mathlib_CoheytingAlgebra_ofSDiff___redArg___lam__0), 3, 2);
lean_closure_set(v___f_203_, 0, v_sdiff_200_);
lean_closure_set(v___f_203_, 1, v_toOrderTop_201_);
v___x_204_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_204_, 0, v_inst_198_);
lean_ctor_set(v___x_204_, 1, v_toOrderBot_202_);
lean_ctor_set(v___x_204_, 2, v_sdiff_200_);
v___x_205_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
lean_ctor_set(v___x_205_, 1, v_toOrderTop_201_);
lean_ctor_set(v___x_205_, 2, v___f_203_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff(lean_object* v_00_u03b1_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_sdiff_209_, lean_object* v_sdiff__le__iff_210_){
_start:
{
lean_object* v_toOrderTop_211_; lean_object* v_toOrderBot_212_; lean_object* v___f_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v_toOrderTop_211_ = lean_ctor_get(v_inst_208_, 0);
lean_inc_n(v_toOrderTop_211_, 2);
v_toOrderBot_212_ = lean_ctor_get(v_inst_208_, 1);
lean_inc(v_toOrderBot_212_);
lean_dec_ref(v_inst_208_);
lean_inc(v_sdiff_209_);
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_CoheytingAlgebra_ofSDiff___redArg___lam__0), 3, 2);
lean_closure_set(v___f_213_, 0, v_sdiff_209_);
lean_closure_set(v___f_213_, 1, v_toOrderTop_211_);
v___x_214_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_214_, 0, v_inst_207_);
lean_ctor_set(v___x_214_, 1, v_toOrderBot_212_);
lean_ctor_set(v___x_214_, 2, v_sdiff_209_);
v___x_215_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_toOrderTop_211_);
lean_ctor_set(v___x_215_, 2, v___f_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg___lam__0(lean_object* v_inst_216_, lean_object* v_hnot_217_, lean_object* v_a_218_, lean_object* v_b_219_){
_start:
{
lean_object* v_inf_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v_inf_220_ = lean_ctor_get(v_inst_216_, 1);
lean_inc(v_inf_220_);
lean_dec_ref(v_inst_216_);
v___x_221_ = lean_apply_1(v_hnot_217_, v_b_219_);
v___x_222_ = lean_apply_2(v_inf_220_, v_a_218_, v___x_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg(lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_hnot_225_){
_start:
{
lean_object* v_toOrderTop_226_; lean_object* v_toOrderBot_227_; lean_object* v___f_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v_toOrderTop_226_ = lean_ctor_get(v_inst_224_, 0);
v_toOrderBot_227_ = lean_ctor_get(v_inst_224_, 1);
lean_inc(v_hnot_225_);
lean_inc_ref(v_inst_223_);
v___f_228_ = lean_alloc_closure((void*)(lp_mathlib_CoheytingAlgebra_ofHNot___redArg___lam__0), 4, 2);
lean_closure_set(v___f_228_, 0, v_inst_223_);
lean_closure_set(v___f_228_, 1, v_hnot_225_);
lean_inc(v_toOrderBot_227_);
v___x_229_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_229_, 0, v_inst_223_);
lean_ctor_set(v___x_229_, 1, v_toOrderBot_227_);
lean_ctor_set(v___x_229_, 2, v___f_228_);
lean_inc(v_toOrderTop_226_);
v___x_230_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
lean_ctor_set(v___x_230_, 1, v_toOrderTop_226_);
lean_ctor_set(v___x_230_, 2, v_hnot_225_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg___boxed(lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_hnot_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_CoheytingAlgebra_ofHNot___redArg(v_inst_231_, v_inst_232_, v_hnot_233_);
lean_dec_ref(v_inst_232_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot(lean_object* v_00_u03b1_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_hnot_238_, lean_object* v_sdiff__le__iff_239_){
_start:
{
lean_object* v_toOrderTop_240_; lean_object* v_toOrderBot_241_; lean_object* v___f_242_; lean_object* v___x_243_; lean_object* v___x_244_; 
v_toOrderTop_240_ = lean_ctor_get(v_inst_237_, 0);
v_toOrderBot_241_ = lean_ctor_get(v_inst_237_, 1);
lean_inc(v_hnot_238_);
lean_inc_ref(v_inst_236_);
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib_CoheytingAlgebra_ofHNot___redArg___lam__0), 4, 2);
lean_closure_set(v___f_242_, 0, v_inst_236_);
lean_closure_set(v___f_242_, 1, v_hnot_238_);
lean_inc(v_toOrderBot_241_);
v___x_243_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_243_, 0, v_inst_236_);
lean_ctor_set(v___x_243_, 1, v_toOrderBot_241_);
lean_ctor_set(v___x_243_, 2, v___f_242_);
lean_inc(v_toOrderTop_240_);
v___x_244_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_244_, 0, v___x_243_);
lean_ctor_set(v___x_244_, 1, v_toOrderTop_240_);
lean_ctor_set(v___x_244_, 2, v_hnot_238_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___boxed(lean_object* v_00_u03b1_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_hnot_248_, lean_object* v_sdiff__le__iff_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_CoheytingAlgebra_ofHNot(v_00_u03b1_245_, v_inst_246_, v_inst_247_, v_hnot_248_, v_sdiff__le__iff_249_);
lean_dec_ref(v_inst_247_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg(lean_object* v_inst_251_){
_start:
{
lean_object* v_toLattice_252_; 
v_toLattice_252_ = lean_ctor_get(v_inst_251_, 0);
lean_inc_ref(v_toLattice_252_);
return v_toLattice_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg___boxed(lean_object* v_inst_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg(v_inst_253_);
lean_dec_ref(v_inst_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice(lean_object* v_00_u03b1_255_, lean_object* v_inst_256_){
_start:
{
lean_object* v_toLattice_257_; 
v_toLattice_257_ = lean_ctor_get(v_inst_256_, 0);
lean_inc_ref(v_toLattice_257_);
return v_toLattice_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___boxed(lean_object* v_00_u03b1_258_, lean_object* v_inst_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice(v_00_u03b1_258_, v_inst_259_);
lean_dec_ref(v_inst_259_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object* v_self_261_, lean_object* v___y_262_){
_start:
{
lean_object* v_toFun_263_; lean_object* v___x_264_; 
v_toFun_263_ = lean_ctor_get(v_self_261_, 0);
lean_inc(v_toFun_263_);
lean_dec_ref(v_self_261_);
v___x_264_ = lean_apply_1(v_toFun_263_, v___y_262_);
return v___x_264_;
}
}
static lean_object* _init_lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1(lean_object* v___f_266_, lean_object* v_toHImp_267_, lean_object* v_a_268_, lean_object* v_b_269_){
_start:
{
lean_object* v___x_270_; lean_object* v_toFun_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_270_ = lean_obj_once(&lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0);
v_toFun_271_ = lean_ctor_get(v___x_270_, 0);
lean_inc(v___f_266_);
v___x_272_ = lean_apply_2(v___f_266_, v___x_270_, v_b_269_);
v___x_273_ = lean_apply_2(v___f_266_, v___x_270_, v_a_268_);
v___x_274_ = lean_apply_2(v_toHImp_267_, v___x_272_, v___x_273_);
lean_inc(v_toFun_271_);
v___x_275_ = lean_apply_1(v_toFun_271_, v___x_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg(lean_object* v_inst_277_){
_start:
{
lean_object* v_toLattice_278_; lean_object* v_toOrderTop_279_; lean_object* v_toHImp_280_; lean_object* v___x_282_; uint8_t v_isShared_283_; uint8_t v_isSharedCheck_290_; 
v_toLattice_278_ = lean_ctor_get(v_inst_277_, 0);
v_toOrderTop_279_ = lean_ctor_get(v_inst_277_, 1);
v_toHImp_280_ = lean_ctor_get(v_inst_277_, 2);
v_isSharedCheck_290_ = !lean_is_exclusive(v_inst_277_);
if (v_isSharedCheck_290_ == 0)
{
v___x_282_ = v_inst_277_;
v_isShared_283_ = v_isSharedCheck_290_;
goto v_resetjp_281_;
}
else
{
lean_inc(v_toHImp_280_);
lean_inc(v_toOrderTop_279_);
lean_inc(v_toLattice_278_);
lean_dec(v_inst_277_);
v___x_282_ = lean_box(0);
v_isShared_283_ = v_isSharedCheck_290_;
goto v_resetjp_281_;
}
v_resetjp_281_:
{
lean_object* v___f_284_; lean_object* v___f_285_; lean_object* v___x_286_; lean_object* v___x_288_; 
v___f_284_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
v___f_285_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1), 4, 2);
lean_closure_set(v___f_285_, 0, v___f_284_);
lean_closure_set(v___f_285_, 1, v_toHImp_280_);
v___x_286_ = lp_mathlib_OrderDual_instLattice___redArg(v_toLattice_278_);
if (v_isShared_283_ == 0)
{
lean_ctor_set(v___x_282_, 2, v___f_285_);
lean_ctor_set(v___x_282_, 0, v___x_286_);
v___x_288_ = v___x_282_;
goto v_reusejp_287_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v___x_286_);
lean_ctor_set(v_reuseFailAlloc_289_, 1, v_toOrderTop_279_);
lean_ctor_set(v_reuseFailAlloc_289_, 2, v___f_285_);
v___x_288_ = v_reuseFailAlloc_289_;
goto v_reusejp_287_;
}
v_reusejp_287_:
{
return v___x_288_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra(lean_object* v_00_u03b1_291_, lean_object* v_inst_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg(v_inst_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(lean_object* v_inst_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v_toLattice_296_; lean_object* v_toOrderTop_297_; lean_object* v_toHImp_298_; lean_object* v_toLattice_299_; lean_object* v_toOrderTop_300_; lean_object* v_toHImp_301_; lean_object* v___x_303_; uint8_t v_isShared_304_; uint8_t v_isSharedCheck_311_; 
v_toLattice_296_ = lean_ctor_get(v_inst_294_, 0);
lean_inc_ref(v_toLattice_296_);
v_toOrderTop_297_ = lean_ctor_get(v_inst_294_, 1);
lean_inc(v_toOrderTop_297_);
v_toHImp_298_ = lean_ctor_get(v_inst_294_, 2);
lean_inc(v_toHImp_298_);
lean_dec_ref(v_inst_294_);
v_toLattice_299_ = lean_ctor_get(v_inst_295_, 0);
v_toOrderTop_300_ = lean_ctor_get(v_inst_295_, 1);
v_toHImp_301_ = lean_ctor_get(v_inst_295_, 2);
v_isSharedCheck_311_ = !lean_is_exclusive(v_inst_295_);
if (v_isSharedCheck_311_ == 0)
{
v___x_303_ = v_inst_295_;
v_isShared_304_ = v_isSharedCheck_311_;
goto v_resetjp_302_;
}
else
{
lean_inc(v_toHImp_301_);
lean_inc(v_toOrderTop_300_);
lean_inc(v_toLattice_299_);
lean_dec(v_inst_295_);
v___x_303_ = lean_box(0);
v_isShared_304_ = v_isSharedCheck_311_;
goto v_resetjp_302_;
}
v_resetjp_302_:
{
lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___f_307_; lean_object* v___x_309_; 
v___x_305_ = lp_mathlib_Prod_instLattice___redArg(v_toLattice_296_, v_toLattice_299_);
v___x_306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_306_, 0, v_toOrderTop_297_);
lean_ctor_set(v___x_306_, 1, v_toOrderTop_300_);
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(v___f_307_, 0, v_toHImp_298_);
lean_closure_set(v___f_307_, 1, v_toHImp_301_);
if (v_isShared_304_ == 0)
{
lean_ctor_set(v___x_303_, 2, v___f_307_);
lean_ctor_set(v___x_303_, 1, v___x_306_);
lean_ctor_set(v___x_303_, 0, v___x_305_);
v___x_309_ = v___x_303_;
goto v_reusejp_308_;
}
else
{
lean_object* v_reuseFailAlloc_310_; 
v_reuseFailAlloc_310_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_310_, 0, v___x_305_);
lean_ctor_set(v_reuseFailAlloc_310_, 1, v___x_306_);
lean_ctor_set(v_reuseFailAlloc_310_, 2, v___f_307_);
v___x_309_ = v_reuseFailAlloc_310_;
goto v_reusejp_308_;
}
v_reusejp_308_:
{
return v___x_309_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedHeytingAlgebra(lean_object* v_00_u03b1_312_, lean_object* v_00_u03b2_313_, lean_object* v_inst_314_, lean_object* v_inst_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(v_inst_314_, v_inst_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__0(lean_object* v_inst_317_, lean_object* v_i_318_){
_start:
{
lean_object* v___x_319_; lean_object* v_toLattice_320_; 
v___x_319_ = lean_apply_1(v_inst_317_, v_i_318_);
v_toLattice_320_ = lean_ctor_get(v___x_319_, 0);
lean_inc_ref(v_toLattice_320_);
lean_dec_ref(v___x_319_);
return v_toLattice_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__1(lean_object* v_inst_321_, lean_object* v_i_322_){
_start:
{
lean_object* v___x_323_; lean_object* v_toOrderTop_324_; 
v___x_323_ = lean_apply_1(v_inst_321_, v_i_322_);
v_toOrderTop_324_ = lean_ctor_get(v___x_323_, 1);
lean_inc(v_toOrderTop_324_);
lean_dec_ref(v___x_323_);
return v_toOrderTop_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__2(lean_object* v_inst_325_, lean_object* v_i_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v___x_329_; lean_object* v_toHImp_330_; lean_object* v___x_331_; 
v___x_329_ = lean_apply_1(v_inst_325_, v_i_326_);
v_toHImp_330_ = lean_ctor_get(v___x_329_, 2);
lean_inc(v_toHImp_330_);
lean_dec_ref(v___x_329_);
v___x_331_ = lean_apply_2(v_toHImp_330_, v___y_327_, v___y_328_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg(lean_object* v_inst_332_){
_start:
{
lean_object* v___f_333_; lean_object* v___f_334_; lean_object* v___f_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___f_338_; lean_object* v___x_339_; 
lean_inc_ref_n(v_inst_332_, 2);
v___f_333_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_333_, 0, v_inst_332_);
v___f_334_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(v___f_334_, 0, v_inst_332_);
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__2), 4, 1);
lean_closure_set(v___f_335_, 0, v_inst_332_);
v___x_336_ = lp_mathlib_Pi_instLattice___redArg(v___f_333_);
v___x_337_ = lp_mathlib_Pi_instOrderTop___redArg(v___f_334_);
v___f_338_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instHImp___redArg___lam__0), 4, 1);
lean_closure_set(v___f_338_, 0, v___f_335_);
v___x_339_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_339_, 0, v___x_336_);
lean_ctor_set(v___x_339_, 1, v___x_337_);
lean_ctor_set(v___x_339_, 2, v___f_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra(lean_object* v_00_u03b9_340_, lean_object* v_00_u03b1_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg(v_inst_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg(lean_object* v_inst_344_){
_start:
{
lean_object* v_toLattice_345_; 
v_toLattice_345_ = lean_ctor_get(v_inst_344_, 0);
lean_inc_ref(v_toLattice_345_);
return v_toLattice_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg___boxed(lean_object* v_inst_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg(v_inst_346_);
lean_dec_ref(v_inst_346_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice(lean_object* v_00_u03b1_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v_toLattice_350_; 
v_toLattice_350_ = lean_ctor_get(v_inst_349_, 0);
lean_inc_ref(v_toLattice_350_);
return v_toLattice_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___boxed(lean_object* v_00_u03b1_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice(v_00_u03b1_351_, v_inst_352_);
lean_dec_ref(v_inst_352_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__1(lean_object* v___f_354_, lean_object* v_toSDiff_355_, lean_object* v_a_356_, lean_object* v_b_357_){
_start:
{
lean_object* v___x_358_; lean_object* v_toFun_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_358_ = lean_obj_once(&lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0);
v_toFun_359_ = lean_ctor_get(v___x_358_, 0);
lean_inc(v___f_354_);
v___x_360_ = lean_apply_2(v___f_354_, v___x_358_, v_b_357_);
v___x_361_ = lean_apply_2(v___f_354_, v___x_358_, v_a_356_);
v___x_362_ = lean_apply_2(v_toSDiff_355_, v___x_360_, v___x_361_);
lean_inc(v_toFun_359_);
v___x_363_ = lean_apply_1(v_toFun_359_, v___x_362_);
return v___x_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg(lean_object* v_inst_364_){
_start:
{
lean_object* v_toLattice_365_; lean_object* v_toOrderBot_366_; lean_object* v_toSDiff_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_377_; 
v_toLattice_365_ = lean_ctor_get(v_inst_364_, 0);
v_toOrderBot_366_ = lean_ctor_get(v_inst_364_, 1);
v_toSDiff_367_ = lean_ctor_get(v_inst_364_, 2);
v_isSharedCheck_377_ = !lean_is_exclusive(v_inst_364_);
if (v_isSharedCheck_377_ == 0)
{
v___x_369_ = v_inst_364_;
v_isShared_370_ = v_isSharedCheck_377_;
goto v_resetjp_368_;
}
else
{
lean_inc(v_toSDiff_367_);
lean_inc(v_toOrderBot_366_);
lean_inc(v_toLattice_365_);
lean_dec(v_inst_364_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_377_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
lean_object* v___f_371_; lean_object* v___f_372_; lean_object* v___x_373_; lean_object* v___x_375_; 
v___f_371_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
v___f_372_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__1), 4, 2);
lean_closure_set(v___f_372_, 0, v___f_371_);
lean_closure_set(v___f_372_, 1, v_toSDiff_367_);
v___x_373_ = lp_mathlib_OrderDual_instLattice___redArg(v_toLattice_365_);
if (v_isShared_370_ == 0)
{
lean_ctor_set(v___x_369_, 2, v___f_372_);
lean_ctor_set(v___x_369_, 0, v___x_373_);
v___x_375_ = v___x_369_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v___x_373_);
lean_ctor_set(v_reuseFailAlloc_376_, 1, v_toOrderBot_366_);
lean_ctor_set(v_reuseFailAlloc_376_, 2, v___f_372_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra(lean_object* v_00_u03b1_378_, lean_object* v_inst_379_){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg(v_inst_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedCoheytingAlgebra___redArg(lean_object* v_inst_381_, lean_object* v_inst_382_){
_start:
{
lean_object* v_toLattice_383_; lean_object* v_toOrderBot_384_; lean_object* v_toSDiff_385_; lean_object* v_toLattice_386_; lean_object* v_toOrderBot_387_; lean_object* v_toSDiff_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_398_; 
v_toLattice_383_ = lean_ctor_get(v_inst_381_, 0);
lean_inc_ref(v_toLattice_383_);
v_toOrderBot_384_ = lean_ctor_get(v_inst_381_, 1);
lean_inc(v_toOrderBot_384_);
v_toSDiff_385_ = lean_ctor_get(v_inst_381_, 2);
lean_inc(v_toSDiff_385_);
lean_dec_ref(v_inst_381_);
v_toLattice_386_ = lean_ctor_get(v_inst_382_, 0);
v_toOrderBot_387_ = lean_ctor_get(v_inst_382_, 1);
v_toSDiff_388_ = lean_ctor_get(v_inst_382_, 2);
v_isSharedCheck_398_ = !lean_is_exclusive(v_inst_382_);
if (v_isSharedCheck_398_ == 0)
{
v___x_390_ = v_inst_382_;
v_isShared_391_ = v_isSharedCheck_398_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_toSDiff_388_);
lean_inc(v_toOrderBot_387_);
lean_inc(v_toLattice_386_);
lean_dec(v_inst_382_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_398_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___f_394_; lean_object* v___x_396_; 
v___x_392_ = lp_mathlib_Prod_instLattice___redArg(v_toLattice_383_, v_toLattice_386_);
v___x_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_393_, 0, v_toOrderBot_384_);
lean_ctor_set(v___x_393_, 1, v_toOrderBot_387_);
v___f_394_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSDiff___redArg___lam__0), 4, 2);
lean_closure_set(v___f_394_, 0, v_toSDiff_385_);
lean_closure_set(v___f_394_, 1, v_toSDiff_388_);
if (v_isShared_391_ == 0)
{
lean_ctor_set(v___x_390_, 2, v___f_394_);
lean_ctor_set(v___x_390_, 1, v___x_393_);
lean_ctor_set(v___x_390_, 0, v___x_392_);
v___x_396_ = v___x_390_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v___x_392_);
lean_ctor_set(v_reuseFailAlloc_397_, 1, v___x_393_);
lean_ctor_set(v_reuseFailAlloc_397_, 2, v___f_394_);
v___x_396_ = v_reuseFailAlloc_397_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
return v___x_396_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedCoheytingAlgebra(lean_object* v_00_u03b1_399_, lean_object* v_00_u03b2_400_, lean_object* v_inst_401_, lean_object* v_inst_402_){
_start:
{
lean_object* v___x_403_; 
v___x_403_ = lp_mathlib_Prod_instGeneralizedCoheytingAlgebra___redArg(v_inst_401_, v_inst_402_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object* v_inst_404_, lean_object* v_i_405_){
_start:
{
lean_object* v___x_406_; lean_object* v_toLattice_407_; 
v___x_406_ = lean_apply_1(v_inst_404_, v_i_405_);
v_toLattice_407_ = lean_ctor_get(v___x_406_, 0);
lean_inc_ref(v_toLattice_407_);
lean_dec_ref(v___x_406_);
return v_toLattice_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__1(lean_object* v_inst_408_, lean_object* v_i_409_){
_start:
{
lean_object* v___x_410_; lean_object* v_toOrderBot_411_; 
v___x_410_ = lean_apply_1(v_inst_408_, v_i_409_);
v_toOrderBot_411_ = lean_ctor_get(v___x_410_, 1);
lean_inc(v_toOrderBot_411_);
lean_dec_ref(v___x_410_);
return v_toOrderBot_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__2(lean_object* v_inst_412_, lean_object* v_i_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
lean_object* v___x_416_; lean_object* v_toSDiff_417_; lean_object* v___x_418_; 
v___x_416_ = lean_apply_1(v_inst_412_, v_i_413_);
v_toSDiff_417_ = lean_ctor_get(v___x_416_, 2);
lean_inc(v_toSDiff_417_);
lean_dec_ref(v___x_416_);
v___x_418_ = lean_apply_2(v_toSDiff_417_, v___y_414_, v___y_415_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg(lean_object* v_inst_419_){
_start:
{
lean_object* v___f_420_; lean_object* v___f_421_; lean_object* v___f_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___f_425_; lean_object* v___x_426_; 
lean_inc_ref_n(v_inst_419_, 2);
v___f_420_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_420_, 0, v_inst_419_);
v___f_421_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(v___f_421_, 0, v_inst_419_);
v___f_422_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__2), 4, 1);
lean_closure_set(v___f_422_, 0, v_inst_419_);
v___x_423_ = lp_mathlib_Pi_instLattice___redArg(v___f_420_);
v___x_424_ = lp_mathlib_Pi_instOrderBot___redArg(v___f_421_);
v___f_425_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSDiff___redArg___lam__0), 4, 1);
lean_closure_set(v___f_425_, 0, v___f_422_);
v___x_426_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_426_, 0, v___x_423_);
lean_ctor_set(v___x_426_, 1, v___x_424_);
lean_ctor_set(v___x_426_, 2, v___f_425_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra(lean_object* v_00_u03b9_427_, lean_object* v_00_u03b1_428_, lean_object* v_inst_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg(v_inst_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__2(lean_object* v___x_431_, lean_object* v___y_432_){
_start:
{
lean_object* v_toFun_433_; lean_object* v___x_434_; 
v_toFun_433_ = lean_ctor_get(v___x_431_, 0);
lean_inc(v_toFun_433_);
lean_dec_ref(v___x_431_);
v___x_434_ = lean_apply_1(v_toFun_433_, v___y_432_);
return v___x_434_;
}
}
static lean_object* _init_lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0(void){
_start:
{
lean_object* v___x_435_; lean_object* v___f_436_; 
v___x_435_ = lean_obj_once(&lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1___closed__0);
v___f_436_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__2), 2, 1);
lean_closure_set(v___f_436_, 0, v___x_435_);
return v___f_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(lean_object* v_inst_437_){
_start:
{
lean_object* v_toGeneralizedHeytingAlgebra_438_; lean_object* v_toOrderBot_439_; lean_object* v_toCompl_440_; lean_object* v_toLattice_441_; lean_object* v_toHImp_442_; lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_467_; 
v_toGeneralizedHeytingAlgebra_438_ = lean_ctor_get(v_inst_437_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_438_);
v_toOrderBot_439_ = lean_ctor_get(v_inst_437_, 1);
lean_inc(v_toOrderBot_439_);
v_toCompl_440_ = lean_ctor_get(v_inst_437_, 2);
lean_inc(v_toCompl_440_);
v_toLattice_441_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_438_, 0);
v_toHImp_442_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_438_, 2);
v_isSharedCheck_467_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_438_);
if (v_isSharedCheck_467_ == 0)
{
lean_object* v_unused_468_; 
v_unused_468_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_438_, 1);
lean_dec(v_unused_468_);
v___x_444_ = v_toGeneralizedHeytingAlgebra_438_;
v_isShared_445_ = v_isSharedCheck_467_;
goto v_resetjp_443_;
}
else
{
lean_inc(v_toHImp_442_);
lean_inc(v_toLattice_441_);
lean_dec(v_toGeneralizedHeytingAlgebra_438_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_467_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_449_; uint8_t v_isShared_450_; uint8_t v_isSharedCheck_463_; 
v___x_446_ = lp_mathlib_OrderDual_instLattice___redArg(v_toLattice_441_);
v___x_447_ = lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(v_inst_437_);
v_isSharedCheck_463_ = !lean_is_exclusive(v_inst_437_);
if (v_isSharedCheck_463_ == 0)
{
lean_object* v_unused_464_; lean_object* v_unused_465_; lean_object* v_unused_466_; 
v_unused_464_ = lean_ctor_get(v_inst_437_, 2);
lean_dec(v_unused_464_);
v_unused_465_ = lean_ctor_get(v_inst_437_, 1);
lean_dec(v_unused_465_);
v_unused_466_ = lean_ctor_get(v_inst_437_, 0);
lean_dec(v_unused_466_);
v___x_449_ = v_inst_437_;
v_isShared_450_ = v_isSharedCheck_463_;
goto v_resetjp_448_;
}
else
{
lean_dec(v_inst_437_);
v___x_449_ = lean_box(0);
v_isShared_450_ = v_isSharedCheck_463_;
goto v_resetjp_448_;
}
v_resetjp_448_:
{
lean_object* v_toOrderTop_451_; lean_object* v___f_452_; lean_object* v___f_453_; lean_object* v___x_455_; 
v_toOrderTop_451_ = lean_ctor_get(v___x_447_, 0);
lean_inc(v_toOrderTop_451_);
lean_dec_ref(v___x_447_);
v___f_452_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
v___f_453_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__1), 4, 2);
lean_closure_set(v___f_453_, 0, v___f_452_);
lean_closure_set(v___f_453_, 1, v_toHImp_442_);
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 2, v___f_453_);
lean_ctor_set(v___x_444_, 1, v_toOrderTop_451_);
lean_ctor_set(v___x_444_, 0, v___x_446_);
v___x_455_ = v___x_444_;
goto v_reusejp_454_;
}
else
{
lean_object* v_reuseFailAlloc_462_; 
v_reuseFailAlloc_462_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_462_, 0, v___x_446_);
lean_ctor_set(v_reuseFailAlloc_462_, 1, v_toOrderTop_451_);
lean_ctor_set(v_reuseFailAlloc_462_, 2, v___f_453_);
v___x_455_ = v_reuseFailAlloc_462_;
goto v_reusejp_454_;
}
v_reusejp_454_:
{
lean_object* v___f_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_460_; 
v___f_456_ = lean_obj_once(&lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0, &lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0_once, _init_lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0);
v___x_457_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_457_, 0, lean_box(0));
lean_closure_set(v___x_457_, 1, lean_box(0));
lean_closure_set(v___x_457_, 2, lean_box(0));
lean_closure_set(v___x_457_, 3, v_toCompl_440_);
lean_closure_set(v___x_457_, 4, v___f_456_);
v___x_458_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_458_, 0, lean_box(0));
lean_closure_set(v___x_458_, 1, lean_box(0));
lean_closure_set(v___x_458_, 2, lean_box(0));
lean_closure_set(v___x_458_, 3, v___f_456_);
lean_closure_set(v___x_458_, 4, v___x_457_);
if (v_isShared_450_ == 0)
{
lean_ctor_set(v___x_449_, 2, v___x_458_);
lean_ctor_set(v___x_449_, 0, v___x_455_);
v___x_460_ = v___x_449_;
goto v_reusejp_459_;
}
else
{
lean_object* v_reuseFailAlloc_461_; 
v_reuseFailAlloc_461_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_461_, 0, v___x_455_);
lean_ctor_set(v_reuseFailAlloc_461_, 1, v_toOrderBot_439_);
lean_ctor_set(v_reuseFailAlloc_461_, 2, v___x_458_);
v___x_460_ = v_reuseFailAlloc_461_;
goto v_reusejp_459_;
}
v_reusejp_459_:
{
return v___x_460_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra(lean_object* v_00_u03b1_469_, lean_object* v_inst_470_){
_start:
{
lean_object* v___x_471_; 
v___x_471_ = lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(v_inst_470_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instHeytingAlgebra___redArg(lean_object* v_inst_472_){
_start:
{
lean_object* v_toGeneralizedCoheytingAlgebra_473_; lean_object* v_toOrderTop_474_; lean_object* v_toHNot_475_; lean_object* v_toLattice_476_; lean_object* v_toSDiff_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_502_; 
v_toGeneralizedCoheytingAlgebra_473_ = lean_ctor_get(v_inst_472_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_473_);
v_toOrderTop_474_ = lean_ctor_get(v_inst_472_, 1);
lean_inc(v_toOrderTop_474_);
v_toHNot_475_ = lean_ctor_get(v_inst_472_, 2);
lean_inc(v_toHNot_475_);
v_toLattice_476_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_473_, 0);
v_toSDiff_477_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_473_, 2);
v_isSharedCheck_502_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_473_);
if (v_isSharedCheck_502_ == 0)
{
lean_object* v_unused_503_; 
v_unused_503_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_473_, 1);
lean_dec(v_unused_503_);
v___x_479_ = v_toGeneralizedCoheytingAlgebra_473_;
v_isShared_480_ = v_isSharedCheck_502_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_toSDiff_477_);
lean_inc(v_toLattice_476_);
lean_dec(v_toGeneralizedCoheytingAlgebra_473_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_502_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_498_; 
v___x_481_ = lp_mathlib_OrderDual_instLattice___redArg(v_toLattice_476_);
v___x_482_ = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(v_inst_472_);
v_isSharedCheck_498_ = !lean_is_exclusive(v_inst_472_);
if (v_isSharedCheck_498_ == 0)
{
lean_object* v_unused_499_; lean_object* v_unused_500_; lean_object* v_unused_501_; 
v_unused_499_ = lean_ctor_get(v_inst_472_, 2);
lean_dec(v_unused_499_);
v_unused_500_ = lean_ctor_get(v_inst_472_, 1);
lean_dec(v_unused_500_);
v_unused_501_ = lean_ctor_get(v_inst_472_, 0);
lean_dec(v_unused_501_);
v___x_484_ = v_inst_472_;
v_isShared_485_ = v_isSharedCheck_498_;
goto v_resetjp_483_;
}
else
{
lean_dec(v_inst_472_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_498_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v_toOrderBot_486_; lean_object* v___f_487_; lean_object* v___f_488_; lean_object* v___x_490_; 
v_toOrderBot_486_ = lean_ctor_get(v___x_482_, 1);
lean_inc(v_toOrderBot_486_);
lean_dec_ref(v___x_482_);
v___f_487_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
v___f_488_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__1), 4, 2);
lean_closure_set(v___f_488_, 0, v___f_487_);
lean_closure_set(v___f_488_, 1, v_toSDiff_477_);
if (v_isShared_480_ == 0)
{
lean_ctor_set(v___x_479_, 2, v___f_488_);
lean_ctor_set(v___x_479_, 1, v_toOrderBot_486_);
lean_ctor_set(v___x_479_, 0, v___x_481_);
v___x_490_ = v___x_479_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_497_; 
v_reuseFailAlloc_497_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_497_, 0, v___x_481_);
lean_ctor_set(v_reuseFailAlloc_497_, 1, v_toOrderBot_486_);
lean_ctor_set(v_reuseFailAlloc_497_, 2, v___f_488_);
v___x_490_ = v_reuseFailAlloc_497_;
goto v_reusejp_489_;
}
v_reusejp_489_:
{
lean_object* v___f_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_495_; 
v___f_491_ = lean_obj_once(&lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0, &lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0_once, _init_lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___closed__0);
v___x_492_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_492_, 0, lean_box(0));
lean_closure_set(v___x_492_, 1, lean_box(0));
lean_closure_set(v___x_492_, 2, lean_box(0));
lean_closure_set(v___x_492_, 3, v_toHNot_475_);
lean_closure_set(v___x_492_, 4, v___f_491_);
v___x_493_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_493_, 0, lean_box(0));
lean_closure_set(v___x_493_, 1, lean_box(0));
lean_closure_set(v___x_493_, 2, lean_box(0));
lean_closure_set(v___x_493_, 3, v___f_491_);
lean_closure_set(v___x_493_, 4, v___x_492_);
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 2, v___x_493_);
lean_ctor_set(v___x_484_, 0, v___x_490_);
v___x_495_ = v___x_484_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v___x_490_);
lean_ctor_set(v_reuseFailAlloc_496_, 1, v_toOrderTop_474_);
lean_ctor_set(v_reuseFailAlloc_496_, 2, v___x_493_);
v___x_495_ = v_reuseFailAlloc_496_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
return v___x_495_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instHeytingAlgebra(lean_object* v_00_u03b1_504_, lean_object* v_inst_505_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_mathlib_OrderDual_instHeytingAlgebra___redArg(v_inst_505_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHeytingAlgebra___redArg(lean_object* v_inst_507_, lean_object* v_inst_508_){
_start:
{
lean_object* v_toGeneralizedHeytingAlgebra_509_; lean_object* v_toOrderBot_510_; lean_object* v_toCompl_511_; lean_object* v_toGeneralizedHeytingAlgebra_512_; lean_object* v_toOrderBot_513_; lean_object* v_toCompl_514_; lean_object* v___x_516_; uint8_t v_isShared_517_; uint8_t v_isSharedCheck_524_; 
v_toGeneralizedHeytingAlgebra_509_ = lean_ctor_get(v_inst_507_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_509_);
v_toOrderBot_510_ = lean_ctor_get(v_inst_507_, 1);
lean_inc(v_toOrderBot_510_);
v_toCompl_511_ = lean_ctor_get(v_inst_507_, 2);
lean_inc(v_toCompl_511_);
lean_dec_ref(v_inst_507_);
v_toGeneralizedHeytingAlgebra_512_ = lean_ctor_get(v_inst_508_, 0);
v_toOrderBot_513_ = lean_ctor_get(v_inst_508_, 1);
v_toCompl_514_ = lean_ctor_get(v_inst_508_, 2);
v_isSharedCheck_524_ = !lean_is_exclusive(v_inst_508_);
if (v_isSharedCheck_524_ == 0)
{
v___x_516_ = v_inst_508_;
v_isShared_517_ = v_isSharedCheck_524_;
goto v_resetjp_515_;
}
else
{
lean_inc(v_toCompl_514_);
lean_inc(v_toOrderBot_513_);
lean_inc(v_toGeneralizedHeytingAlgebra_512_);
lean_dec(v_inst_508_);
v___x_516_ = lean_box(0);
v_isShared_517_ = v_isSharedCheck_524_;
goto v_resetjp_515_;
}
v_resetjp_515_:
{
lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___f_520_; lean_object* v___x_522_; 
v___x_518_ = lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(v_toGeneralizedHeytingAlgebra_509_, v_toGeneralizedHeytingAlgebra_512_);
v___x_519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_519_, 0, v_toOrderBot_510_);
lean_ctor_set(v___x_519_, 1, v_toOrderBot_513_);
v___f_520_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_520_, 0, v_toCompl_511_);
lean_closure_set(v___f_520_, 1, v_toCompl_514_);
if (v_isShared_517_ == 0)
{
lean_ctor_set(v___x_516_, 2, v___f_520_);
lean_ctor_set(v___x_516_, 1, v___x_519_);
lean_ctor_set(v___x_516_, 0, v___x_518_);
v___x_522_ = v___x_516_;
goto v_reusejp_521_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v___x_518_);
lean_ctor_set(v_reuseFailAlloc_523_, 1, v___x_519_);
lean_ctor_set(v_reuseFailAlloc_523_, 2, v___f_520_);
v___x_522_ = v_reuseFailAlloc_523_;
goto v_reusejp_521_;
}
v_reusejp_521_:
{
return v___x_522_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHeytingAlgebra(lean_object* v_00_u03b1_525_, lean_object* v_00_u03b2_526_, lean_object* v_inst_527_, lean_object* v_inst_528_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_mathlib_Prod_instHeytingAlgebra___redArg(v_inst_527_, v_inst_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__0(lean_object* v_inst_530_, lean_object* v_i_531_){
_start:
{
lean_object* v___x_532_; lean_object* v_toGeneralizedHeytingAlgebra_533_; 
v___x_532_ = lean_apply_1(v_inst_530_, v_i_531_);
v_toGeneralizedHeytingAlgebra_533_ = lean_ctor_get(v___x_532_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_533_);
lean_dec_ref(v___x_532_);
return v_toGeneralizedHeytingAlgebra_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__1(lean_object* v_inst_534_, lean_object* v_i_535_){
_start:
{
lean_object* v___x_536_; lean_object* v_toOrderBot_537_; 
v___x_536_ = lean_apply_1(v_inst_534_, v_i_535_);
v_toOrderBot_537_ = lean_ctor_get(v___x_536_, 1);
lean_inc(v_toOrderBot_537_);
lean_dec_ref(v___x_536_);
return v_toOrderBot_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__2(lean_object* v_inst_538_, lean_object* v_i_539_, lean_object* v___y_540_){
_start:
{
lean_object* v___x_541_; lean_object* v_toCompl_542_; lean_object* v___x_543_; 
v___x_541_ = lean_apply_1(v_inst_538_, v_i_539_);
v_toCompl_542_ = lean_ctor_get(v___x_541_, 2);
lean_inc(v_toCompl_542_);
lean_dec_ref(v___x_541_);
v___x_543_ = lean_apply_1(v_toCompl_542_, v___y_540_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg(lean_object* v_inst_544_){
_start:
{
lean_object* v___f_545_; lean_object* v___f_546_; lean_object* v___f_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___f_550_; lean_object* v___x_551_; 
lean_inc_ref_n(v_inst_544_, 2);
v___f_545_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_545_, 0, v_inst_544_);
v___f_546_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(v___f_546_, 0, v_inst_544_);
v___f_547_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_547_, 0, v_inst_544_);
v___x_548_ = lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg(v___f_545_);
v___x_549_ = lp_mathlib_Pi_instOrderBot___redArg(v___f_546_);
v___f_550_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_550_, 0, v___f_547_);
v___x_551_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_551_, 0, v___x_548_);
lean_ctor_set(v___x_551_, 1, v___x_549_);
lean_ctor_set(v___x_551_, 2, v___f_550_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra(lean_object* v_00_u03b9_552_, lean_object* v_00_u03b1_553_, lean_object* v_inst_554_){
_start:
{
lean_object* v___x_555_; 
v___x_555_ = lp_mathlib_Pi_instHeytingAlgebra___redArg(v_inst_554_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoheytingAlgebra___redArg(lean_object* v_inst_556_, lean_object* v_inst_557_){
_start:
{
lean_object* v_toGeneralizedCoheytingAlgebra_558_; lean_object* v_toGeneralizedCoheytingAlgebra_559_; lean_object* v_toOrderTop_560_; lean_object* v_toHNot_561_; lean_object* v_toLattice_562_; lean_object* v_toSDiff_563_; lean_object* v_toOrderTop_564_; lean_object* v_toHNot_565_; lean_object* v_toLattice_566_; lean_object* v_toSDiff_567_; lean_object* v___x_569_; uint8_t v_isShared_570_; uint8_t v_isSharedCheck_607_; 
v_toGeneralizedCoheytingAlgebra_558_ = lean_ctor_get(v_inst_556_, 0);
v_toGeneralizedCoheytingAlgebra_559_ = lean_ctor_get(v_inst_557_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_559_);
v_toOrderTop_560_ = lean_ctor_get(v_inst_556_, 1);
lean_inc(v_toOrderTop_560_);
v_toHNot_561_ = lean_ctor_get(v_inst_556_, 2);
lean_inc(v_toHNot_561_);
v_toLattice_562_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_558_, 0);
v_toSDiff_563_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_558_, 2);
lean_inc(v_toSDiff_563_);
v_toOrderTop_564_ = lean_ctor_get(v_inst_557_, 1);
lean_inc(v_toOrderTop_564_);
v_toHNot_565_ = lean_ctor_get(v_inst_557_, 2);
lean_inc(v_toHNot_565_);
v_toLattice_566_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_559_, 0);
v_toSDiff_567_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_559_, 2);
v_isSharedCheck_607_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_559_);
if (v_isSharedCheck_607_ == 0)
{
lean_object* v_unused_608_; 
v_unused_608_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_559_, 1);
lean_dec(v_unused_608_);
v___x_569_ = v_toGeneralizedCoheytingAlgebra_559_;
v_isShared_570_ = v_isSharedCheck_607_;
goto v_resetjp_568_;
}
else
{
lean_inc(v_toSDiff_567_);
lean_inc(v_toLattice_566_);
lean_dec(v_toGeneralizedCoheytingAlgebra_559_);
v___x_569_ = lean_box(0);
v_isShared_570_ = v_isSharedCheck_607_;
goto v_resetjp_568_;
}
v_resetjp_568_:
{
lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v_toOrderBot_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_605_; 
lean_inc_ref(v_toLattice_562_);
v___x_571_ = lp_mathlib_Prod_instLattice___redArg(v_toLattice_562_, v_toLattice_566_);
v___x_572_ = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(v_inst_556_);
lean_dec_ref(v_inst_556_);
v_toOrderBot_573_ = lean_ctor_get(v___x_572_, 1);
v_isSharedCheck_605_ = !lean_is_exclusive(v___x_572_);
if (v_isSharedCheck_605_ == 0)
{
lean_object* v_unused_606_; 
v_unused_606_ = lean_ctor_get(v___x_572_, 0);
lean_dec(v_unused_606_);
v___x_575_ = v___x_572_;
v_isShared_576_ = v_isSharedCheck_605_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_toOrderBot_573_);
lean_dec(v___x_572_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_605_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
lean_object* v___x_577_; lean_object* v___x_579_; uint8_t v_isShared_580_; uint8_t v_isSharedCheck_601_; 
v___x_577_ = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(v_inst_557_);
v_isSharedCheck_601_ = !lean_is_exclusive(v_inst_557_);
if (v_isSharedCheck_601_ == 0)
{
lean_object* v_unused_602_; lean_object* v_unused_603_; lean_object* v_unused_604_; 
v_unused_602_ = lean_ctor_get(v_inst_557_, 2);
lean_dec(v_unused_602_);
v_unused_603_ = lean_ctor_get(v_inst_557_, 1);
lean_dec(v_unused_603_);
v_unused_604_ = lean_ctor_get(v_inst_557_, 0);
lean_dec(v_unused_604_);
v___x_579_ = v_inst_557_;
v_isShared_580_ = v_isSharedCheck_601_;
goto v_resetjp_578_;
}
else
{
lean_dec(v_inst_557_);
v___x_579_ = lean_box(0);
v_isShared_580_ = v_isSharedCheck_601_;
goto v_resetjp_578_;
}
v_resetjp_578_:
{
lean_object* v_toOrderBot_581_; lean_object* v___x_583_; uint8_t v_isShared_584_; uint8_t v_isSharedCheck_599_; 
v_toOrderBot_581_ = lean_ctor_get(v___x_577_, 1);
v_isSharedCheck_599_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_599_ == 0)
{
lean_object* v_unused_600_; 
v_unused_600_ = lean_ctor_get(v___x_577_, 0);
lean_dec(v_unused_600_);
v___x_583_ = v___x_577_;
v_isShared_584_ = v_isSharedCheck_599_;
goto v_resetjp_582_;
}
else
{
lean_inc(v_toOrderBot_581_);
lean_dec(v___x_577_);
v___x_583_ = lean_box(0);
v_isShared_584_ = v_isSharedCheck_599_;
goto v_resetjp_582_;
}
v_resetjp_582_:
{
lean_object* v___x_586_; 
if (v_isShared_584_ == 0)
{
lean_ctor_set(v___x_583_, 0, v_toOrderBot_573_);
v___x_586_ = v___x_583_;
goto v_reusejp_585_;
}
else
{
lean_object* v_reuseFailAlloc_598_; 
v_reuseFailAlloc_598_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_598_, 0, v_toOrderBot_573_);
lean_ctor_set(v_reuseFailAlloc_598_, 1, v_toOrderBot_581_);
v___x_586_ = v_reuseFailAlloc_598_;
goto v_reusejp_585_;
}
v_reusejp_585_:
{
lean_object* v___f_587_; lean_object* v___x_589_; 
v___f_587_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSDiff___redArg___lam__0), 4, 2);
lean_closure_set(v___f_587_, 0, v_toSDiff_563_);
lean_closure_set(v___f_587_, 1, v_toSDiff_567_);
if (v_isShared_570_ == 0)
{
lean_ctor_set(v___x_569_, 2, v___f_587_);
lean_ctor_set(v___x_569_, 1, v___x_586_);
lean_ctor_set(v___x_569_, 0, v___x_571_);
v___x_589_ = v___x_569_;
goto v_reusejp_588_;
}
else
{
lean_object* v_reuseFailAlloc_597_; 
v_reuseFailAlloc_597_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_597_, 0, v___x_571_);
lean_ctor_set(v_reuseFailAlloc_597_, 1, v___x_586_);
lean_ctor_set(v_reuseFailAlloc_597_, 2, v___f_587_);
v___x_589_ = v_reuseFailAlloc_597_;
goto v_reusejp_588_;
}
v_reusejp_588_:
{
lean_object* v___x_591_; 
if (v_isShared_576_ == 0)
{
lean_ctor_set(v___x_575_, 1, v_toOrderTop_564_);
lean_ctor_set(v___x_575_, 0, v_toOrderTop_560_);
v___x_591_ = v___x_575_;
goto v_reusejp_590_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v_toOrderTop_560_);
lean_ctor_set(v_reuseFailAlloc_596_, 1, v_toOrderTop_564_);
v___x_591_ = v_reuseFailAlloc_596_;
goto v_reusejp_590_;
}
v_reusejp_590_:
{
lean_object* v___f_592_; lean_object* v___x_594_; 
v___f_592_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_592_, 0, v_toHNot_561_);
lean_closure_set(v___f_592_, 1, v_toHNot_565_);
if (v_isShared_580_ == 0)
{
lean_ctor_set(v___x_579_, 2, v___f_592_);
lean_ctor_set(v___x_579_, 1, v___x_591_);
lean_ctor_set(v___x_579_, 0, v___x_589_);
v___x_594_ = v___x_579_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v___x_589_);
lean_ctor_set(v_reuseFailAlloc_595_, 1, v___x_591_);
lean_ctor_set(v_reuseFailAlloc_595_, 2, v___f_592_);
v___x_594_ = v_reuseFailAlloc_595_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
return v___x_594_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoheytingAlgebra(lean_object* v_00_u03b1_609_, lean_object* v_00_u03b2_610_, lean_object* v_inst_611_, lean_object* v_inst_612_){
_start:
{
lean_object* v___x_613_; 
v___x_613_ = lp_mathlib_Prod_instCoheytingAlgebra___redArg(v_inst_611_, v_inst_612_);
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__0(lean_object* v_inst_614_, lean_object* v_i_615_){
_start:
{
lean_object* v___x_616_; lean_object* v_toGeneralizedCoheytingAlgebra_617_; 
v___x_616_ = lean_apply_1(v_inst_614_, v_i_615_);
v_toGeneralizedCoheytingAlgebra_617_ = lean_ctor_get(v___x_616_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_617_);
lean_dec_ref(v___x_616_);
return v_toGeneralizedCoheytingAlgebra_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__1(lean_object* v_inst_618_, lean_object* v_i_619_){
_start:
{
lean_object* v___x_620_; lean_object* v_toOrderTop_621_; 
v___x_620_ = lean_apply_1(v_inst_618_, v_i_619_);
v_toOrderTop_621_ = lean_ctor_get(v___x_620_, 1);
lean_inc(v_toOrderTop_621_);
lean_dec_ref(v___x_620_);
return v_toOrderTop_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__2(lean_object* v_inst_622_, lean_object* v_i_623_, lean_object* v___y_624_){
_start:
{
lean_object* v___x_625_; lean_object* v_toHNot_626_; lean_object* v___x_627_; 
v___x_625_ = lean_apply_1(v_inst_622_, v_i_623_);
v_toHNot_626_ = lean_ctor_get(v___x_625_, 2);
lean_inc(v_toHNot_626_);
lean_dec_ref(v___x_625_);
v___x_627_ = lean_apply_1(v_toHNot_626_, v___y_624_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg(lean_object* v_inst_628_){
_start:
{
lean_object* v___f_629_; lean_object* v___f_630_; lean_object* v___f_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___f_634_; lean_object* v___x_635_; 
lean_inc_ref_n(v_inst_628_, 2);
v___f_629_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_629_, 0, v_inst_628_);
v___f_630_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(v___f_630_, 0, v_inst_628_);
v___f_631_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_631_, 0, v_inst_628_);
v___x_632_ = lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg(v___f_629_);
v___x_633_ = lp_mathlib_Pi_instOrderTop___redArg(v___f_630_);
v___f_634_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_634_, 0, v___f_631_);
v___x_635_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_635_, 0, v___x_632_);
lean_ctor_set(v___x_635_, 1, v___x_633_);
lean_ctor_set(v___x_635_, 2, v___f_634_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra(lean_object* v_00_u03b9_636_, lean_object* v_00_u03b1_637_, lean_object* v_inst_638_){
_start:
{
lean_object* v___x_639_; 
v___x_639_ = lp_mathlib_Pi_instCoheytingAlgebra___redArg(v_inst_638_);
return v___x_639_;
}
}
static lean_object* _init_lp_mathlib_Prop_instHeytingAlgebra___closed__0(void){
_start:
{
lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_640_ = lp_mathlib_Prop_instDistribLattice;
v___x_641_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_641_, 0, v___x_640_);
lean_ctor_set(v___x_641_, 1, lean_box(0));
lean_ctor_set(v___x_641_, 2, lean_box(0));
return v___x_641_;
}
}
static lean_object* _init_lp_mathlib_Prop_instHeytingAlgebra___closed__1(void){
_start:
{
lean_object* v___x_642_; lean_object* v___x_643_; 
v___x_642_ = lean_obj_once(&lp_mathlib_Prop_instHeytingAlgebra___closed__0, &lp_mathlib_Prop_instHeytingAlgebra___closed__0_once, _init_lp_mathlib_Prop_instHeytingAlgebra___closed__0);
v___x_643_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_643_, 0, v___x_642_);
lean_ctor_set(v___x_643_, 1, lean_box(0));
lean_ctor_set(v___x_643_, 2, lean_box(0));
return v___x_643_;
}
}
static lean_object* _init_lp_mathlib_Prop_instHeytingAlgebra(void){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lean_obj_once(&lp_mathlib_Prop_instHeytingAlgebra___closed__1, &lp_mathlib_Prop_instHeytingAlgebra___closed__1_once, _init_lp_mathlib_Prop_instHeytingAlgebra___closed__1);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0(lean_object* v_inst_645_, lean_object* v_toOrderBot_646_, lean_object* v_a_647_, lean_object* v_b_648_){
_start:
{
lean_object* v_toDecidableLE_649_; lean_object* v___x_650_; uint8_t v___x_651_; 
v_toDecidableLE_649_ = lean_ctor_get(v_inst_645_, 4);
lean_inc_ref(v_toDecidableLE_649_);
lean_dec_ref(v_inst_645_);
lean_inc(v_a_647_);
v___x_650_ = lean_apply_2(v_toDecidableLE_649_, v_a_647_, v_b_648_);
v___x_651_ = lean_unbox(v___x_650_);
if (v___x_651_ == 0)
{
return v_a_647_;
}
else
{
lean_dec(v_a_647_);
lean_inc(v_toOrderBot_646_);
return v_toOrderBot_646_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0___boxed(lean_object* v_inst_652_, lean_object* v_toOrderBot_653_, lean_object* v_a_654_, lean_object* v_b_655_){
_start:
{
lean_object* v_res_656_; 
v_res_656_ = lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0(v_inst_652_, v_toOrderBot_653_, v_a_654_, v_b_655_);
lean_dec(v_toOrderBot_653_);
return v_res_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1(lean_object* v_inst_657_, lean_object* v_toOrderTop_658_, lean_object* v_a_659_, lean_object* v_b_660_){
_start:
{
lean_object* v_toDecidableLE_661_; lean_object* v___x_662_; uint8_t v___x_663_; 
v_toDecidableLE_661_ = lean_ctor_get(v_inst_657_, 4);
lean_inc_ref(v_toDecidableLE_661_);
lean_dec_ref(v_inst_657_);
lean_inc(v_b_660_);
v___x_662_ = lean_apply_2(v_toDecidableLE_661_, v_a_659_, v_b_660_);
v___x_663_ = lean_unbox(v___x_662_);
if (v___x_663_ == 0)
{
return v_b_660_;
}
else
{
lean_dec(v_b_660_);
lean_inc(v_toOrderTop_658_);
return v_toOrderTop_658_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1___boxed(lean_object* v_inst_664_, lean_object* v_toOrderTop_665_, lean_object* v_a_666_, lean_object* v_b_667_){
_start:
{
lean_object* v_res_668_; 
v_res_668_ = lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1(v_inst_664_, v_toOrderTop_665_, v_a_666_, v_b_667_);
lean_dec(v_toOrderTop_665_);
return v_res_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2(lean_object* v_inst_669_, lean_object* v_toOrderBot_670_, lean_object* v_toOrderTop_671_, lean_object* v_a_672_){
_start:
{
lean_object* v_toDecidableEq_673_; lean_object* v___x_674_; uint8_t v___x_675_; 
v_toDecidableEq_673_ = lean_ctor_get(v_inst_669_, 5);
lean_inc_ref(v_toDecidableEq_673_);
lean_dec_ref(v_inst_669_);
lean_inc(v_toOrderBot_670_);
v___x_674_ = lean_apply_2(v_toDecidableEq_673_, v_a_672_, v_toOrderBot_670_);
v___x_675_ = lean_unbox(v___x_674_);
if (v___x_675_ == 0)
{
return v_toOrderBot_670_;
}
else
{
lean_dec(v_toOrderBot_670_);
lean_inc(v_toOrderTop_671_);
return v_toOrderTop_671_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2___boxed(lean_object* v_inst_676_, lean_object* v_toOrderBot_677_, lean_object* v_toOrderTop_678_, lean_object* v_a_679_){
_start:
{
lean_object* v_res_680_; 
v_res_680_ = lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2(v_inst_676_, v_toOrderBot_677_, v_toOrderTop_678_, v_a_679_);
lean_dec(v_toOrderTop_678_);
return v_res_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3(lean_object* v_inst_681_, lean_object* v_toOrderTop_682_, lean_object* v_toOrderBot_683_, lean_object* v_a_684_){
_start:
{
lean_object* v_toDecidableEq_685_; lean_object* v___x_686_; uint8_t v___x_687_; 
v_toDecidableEq_685_ = lean_ctor_get(v_inst_681_, 5);
lean_inc_ref(v_toDecidableEq_685_);
lean_dec_ref(v_inst_681_);
lean_inc(v_toOrderTop_682_);
v___x_686_ = lean_apply_2(v_toDecidableEq_685_, v_a_684_, v_toOrderTop_682_);
v___x_687_ = lean_unbox(v___x_686_);
if (v___x_687_ == 0)
{
return v_toOrderTop_682_;
}
else
{
lean_dec(v_toOrderTop_682_);
lean_inc(v_toOrderBot_683_);
return v_toOrderBot_683_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3___boxed(lean_object* v_inst_688_, lean_object* v_toOrderTop_689_, lean_object* v_toOrderBot_690_, lean_object* v_a_691_){
_start:
{
lean_object* v_res_692_; 
v_res_692_ = lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3(v_inst_688_, v_toOrderTop_689_, v_toOrderBot_690_, v_a_691_);
lean_dec(v_toOrderBot_690_);
return v_res_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg(lean_object* v_inst_693_, lean_object* v_inst_694_){
_start:
{
lean_object* v___x_695_; lean_object* v_toOrderTop_696_; lean_object* v_toOrderBot_697_; lean_object* v___f_698_; lean_object* v___f_699_; lean_object* v___f_700_; lean_object* v___f_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; 
v___x_695_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_693_);
v_toOrderTop_696_ = lean_ctor_get(v_inst_694_, 0);
lean_inc_n(v_toOrderTop_696_, 4);
v_toOrderBot_697_ = lean_ctor_get(v_inst_694_, 1);
lean_inc_n(v_toOrderBot_697_, 4);
lean_dec_ref(v_inst_694_);
lean_inc_ref_n(v_inst_693_, 3);
v___f_698_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_698_, 0, v_inst_693_);
lean_closure_set(v___f_698_, 1, v_toOrderBot_697_);
v___f_699_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_699_, 0, v_inst_693_);
lean_closure_set(v___f_699_, 1, v_toOrderTop_696_);
v___f_700_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2___boxed), 4, 3);
lean_closure_set(v___f_700_, 0, v_inst_693_);
lean_closure_set(v___f_700_, 1, v_toOrderBot_697_);
lean_closure_set(v___f_700_, 2, v_toOrderTop_696_);
v___f_701_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_701_, 0, v_inst_693_);
lean_closure_set(v___f_701_, 1, v_toOrderTop_696_);
lean_closure_set(v___f_701_, 2, v_toOrderBot_697_);
v___x_702_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_702_, 0, v___x_695_);
lean_ctor_set(v___x_702_, 1, v_toOrderTop_696_);
lean_ctor_set(v___x_702_, 2, v___f_699_);
v___x_703_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_703_, 0, v___x_702_);
lean_ctor_set(v___x_703_, 1, v_toOrderBot_697_);
lean_ctor_set(v___x_703_, 2, v___f_700_);
v___x_704_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_704_, 0, v___x_703_);
lean_ctor_set(v___x_704_, 1, v___f_698_);
lean_ctor_set(v___x_704_, 2, v___f_701_);
return v___x_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra(lean_object* v_00_u03b1_705_, lean_object* v_inst_706_, lean_object* v_inst_707_){
_start:
{
lean_object* v___x_708_; lean_object* v_toOrderTop_709_; lean_object* v_toOrderBot_710_; lean_object* v___f_711_; lean_object* v___f_712_; lean_object* v___f_713_; lean_object* v___f_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; 
v___x_708_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_706_);
v_toOrderTop_709_ = lean_ctor_get(v_inst_707_, 0);
lean_inc_n(v_toOrderTop_709_, 4);
v_toOrderBot_710_ = lean_ctor_get(v_inst_707_, 1);
lean_inc_n(v_toOrderBot_710_, 4);
lean_dec_ref(v_inst_707_);
lean_inc_ref_n(v_inst_706_, 3);
v___f_711_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_711_, 0, v_inst_706_);
lean_closure_set(v___f_711_, 1, v_toOrderBot_710_);
v___f_712_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_712_, 0, v_inst_706_);
lean_closure_set(v___f_712_, 1, v_toOrderTop_709_);
v___f_713_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2___boxed), 4, 3);
lean_closure_set(v___f_713_, 0, v_inst_706_);
lean_closure_set(v___f_713_, 1, v_toOrderBot_710_);
lean_closure_set(v___f_713_, 2, v_toOrderTop_709_);
v___f_714_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_714_, 0, v_inst_706_);
lean_closure_set(v___f_714_, 1, v_toOrderTop_709_);
lean_closure_set(v___f_714_, 2, v_toOrderBot_710_);
v___x_715_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_715_, 0, v___x_708_);
lean_ctor_set(v___x_715_, 1, v_toOrderTop_709_);
lean_ctor_set(v___x_715_, 2, v___f_712_);
v___x_716_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_716_, 0, v___x_715_);
lean_ctor_set(v___x_716_, 1, v_toOrderBot_710_);
lean_ctor_set(v___x_716_, 2, v___f_713_);
v___x_717_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_717_, 0, v___x_716_);
lean_ctor_set(v___x_717_, 1, v___f_711_);
lean_ctor_set(v___x_717_, 2, v___f_714_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBiheytingAlgebra___redArg(lean_object* v_inst_718_){
_start:
{
lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v_toHeytingAlgebra_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_732_; 
lean_inc_ref(v_inst_718_);
v___x_719_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_inst_718_);
v___x_720_ = lp_mathlib_OrderDual_instHeytingAlgebra___redArg(v___x_719_);
v_toHeytingAlgebra_721_ = lean_ctor_get(v_inst_718_, 0);
v_isSharedCheck_732_ = !lean_is_exclusive(v_inst_718_);
if (v_isSharedCheck_732_ == 0)
{
lean_object* v_unused_733_; lean_object* v_unused_734_; 
v_unused_733_ = lean_ctor_get(v_inst_718_, 2);
lean_dec(v_unused_733_);
v_unused_734_ = lean_ctor_get(v_inst_718_, 1);
lean_dec(v_unused_734_);
v___x_723_ = v_inst_718_;
v_isShared_724_ = v_isSharedCheck_732_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_toHeytingAlgebra_721_);
lean_dec(v_inst_718_);
v___x_723_ = lean_box(0);
v_isShared_724_ = v_isSharedCheck_732_;
goto v_resetjp_722_;
}
v_resetjp_722_:
{
lean_object* v___x_725_; lean_object* v_toGeneralizedCoheytingAlgebra_726_; lean_object* v_toHNot_727_; lean_object* v_toSDiff_728_; lean_object* v___x_730_; 
v___x_725_ = lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(v_toHeytingAlgebra_721_);
v_toGeneralizedCoheytingAlgebra_726_ = lean_ctor_get(v___x_725_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_726_);
v_toHNot_727_ = lean_ctor_get(v___x_725_, 2);
lean_inc(v_toHNot_727_);
lean_dec_ref(v___x_725_);
v_toSDiff_728_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_726_, 2);
lean_inc(v_toSDiff_728_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_726_);
if (v_isShared_724_ == 0)
{
lean_ctor_set(v___x_723_, 2, v_toHNot_727_);
lean_ctor_set(v___x_723_, 1, v_toSDiff_728_);
lean_ctor_set(v___x_723_, 0, v___x_720_);
v___x_730_ = v___x_723_;
goto v_reusejp_729_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v___x_720_);
lean_ctor_set(v_reuseFailAlloc_731_, 1, v_toSDiff_728_);
lean_ctor_set(v_reuseFailAlloc_731_, 2, v_toHNot_727_);
v___x_730_ = v_reuseFailAlloc_731_;
goto v_reusejp_729_;
}
v_reusejp_729_:
{
return v___x_730_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBiheytingAlgebra(lean_object* v_00_u03b1_735_, lean_object* v_inst_736_){
_start:
{
lean_object* v___x_737_; 
v___x_737_ = lp_mathlib_OrderDual_instBiheytingAlgebra___redArg(v_inst_736_);
return v___x_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBiheytingAlgebra___redArg(lean_object* v_inst_738_, lean_object* v_inst_739_){
_start:
{
lean_object* v_toHeytingAlgebra_740_; lean_object* v_toHeytingAlgebra_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v_toGeneralizedCoheytingAlgebra_746_; lean_object* v_toHNot_747_; lean_object* v_toSDiff_748_; lean_object* v___x_750_; uint8_t v_isShared_751_; uint8_t v_isSharedCheck_755_; 
v_toHeytingAlgebra_740_ = lean_ctor_get(v_inst_738_, 0);
v_toHeytingAlgebra_741_ = lean_ctor_get(v_inst_739_, 0);
lean_inc_ref(v_toHeytingAlgebra_741_);
lean_inc_ref(v_toHeytingAlgebra_740_);
v___x_742_ = lp_mathlib_Prod_instHeytingAlgebra___redArg(v_toHeytingAlgebra_740_, v_toHeytingAlgebra_741_);
v___x_743_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_inst_738_);
v___x_744_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_inst_739_);
v___x_745_ = lp_mathlib_Prod_instCoheytingAlgebra___redArg(v___x_743_, v___x_744_);
v_toGeneralizedCoheytingAlgebra_746_ = lean_ctor_get(v___x_745_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_746_);
v_toHNot_747_ = lean_ctor_get(v___x_745_, 2);
lean_inc(v_toHNot_747_);
lean_dec_ref(v___x_745_);
v_toSDiff_748_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_746_, 2);
v_isSharedCheck_755_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_746_);
if (v_isSharedCheck_755_ == 0)
{
lean_object* v_unused_756_; lean_object* v_unused_757_; 
v_unused_756_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_746_, 1);
lean_dec(v_unused_756_);
v_unused_757_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_746_, 0);
lean_dec(v_unused_757_);
v___x_750_ = v_toGeneralizedCoheytingAlgebra_746_;
v_isShared_751_ = v_isSharedCheck_755_;
goto v_resetjp_749_;
}
else
{
lean_inc(v_toSDiff_748_);
lean_dec(v_toGeneralizedCoheytingAlgebra_746_);
v___x_750_ = lean_box(0);
v_isShared_751_ = v_isSharedCheck_755_;
goto v_resetjp_749_;
}
v_resetjp_749_:
{
lean_object* v___x_753_; 
if (v_isShared_751_ == 0)
{
lean_ctor_set(v___x_750_, 2, v_toHNot_747_);
lean_ctor_set(v___x_750_, 1, v_toSDiff_748_);
lean_ctor_set(v___x_750_, 0, v___x_742_);
v___x_753_ = v___x_750_;
goto v_reusejp_752_;
}
else
{
lean_object* v_reuseFailAlloc_754_; 
v_reuseFailAlloc_754_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_754_, 0, v___x_742_);
lean_ctor_set(v_reuseFailAlloc_754_, 1, v_toSDiff_748_);
lean_ctor_set(v_reuseFailAlloc_754_, 2, v_toHNot_747_);
v___x_753_ = v_reuseFailAlloc_754_;
goto v_reusejp_752_;
}
v_reusejp_752_:
{
return v___x_753_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBiheytingAlgebra(lean_object* v_00_u03b1_758_, lean_object* v_00_u03b2_759_, lean_object* v_inst_760_, lean_object* v_inst_761_){
_start:
{
lean_object* v___x_762_; 
v___x_762_ = lp_mathlib_Prod_instBiheytingAlgebra___redArg(v_inst_760_, v_inst_761_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__0(lean_object* v_inst_763_, lean_object* v_i_764_){
_start:
{
lean_object* v___x_765_; lean_object* v_toHeytingAlgebra_766_; 
v___x_765_ = lean_apply_1(v_inst_763_, v_i_764_);
v_toHeytingAlgebra_766_ = lean_ctor_get(v___x_765_, 0);
lean_inc_ref(v_toHeytingAlgebra_766_);
lean_dec_ref(v___x_765_);
return v_toHeytingAlgebra_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__1(lean_object* v_inst_767_, lean_object* v_i_768_){
_start:
{
lean_object* v___x_769_; lean_object* v___x_770_; 
v___x_769_ = lean_apply_1(v_inst_767_, v_i_768_);
v___x_770_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v___x_769_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg(lean_object* v_inst_771_){
_start:
{
lean_object* v___f_772_; lean_object* v___f_773_; lean_object* v___x_774_; lean_object* v___f_775_; lean_object* v___f_776_; lean_object* v___f_777_; lean_object* v___f_778_; lean_object* v___f_779_; lean_object* v___x_780_; 
lean_inc_ref(v_inst_771_);
v___f_772_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_772_, 0, v_inst_771_);
v___f_773_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(v___f_773_, 0, v_inst_771_);
v___x_774_ = lp_mathlib_Pi_instHeytingAlgebra___redArg(v___f_772_);
lean_inc_ref(v___f_773_);
v___f_775_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(v___f_775_, 0, v___f_773_);
v___f_776_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__2), 4, 1);
lean_closure_set(v___f_776_, 0, v___f_775_);
v___f_777_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSDiff___redArg___lam__0), 4, 1);
lean_closure_set(v___f_777_, 0, v___f_776_);
v___f_778_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_778_, 0, v___f_773_);
v___f_779_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_779_, 0, v___f_778_);
v___x_780_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_780_, 0, v___x_774_);
lean_ctor_set(v___x_780_, 1, v___f_777_);
lean_ctor_set(v___x_780_, 2, v___f_779_);
return v___x_780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra(lean_object* v_00_u03b9_781_, lean_object* v_00_u03b1_782_, lean_object* v_inst_783_){
_start:
{
lean_object* v___x_784_; 
v___x_784_ = lp_mathlib_Pi_instBiheytingAlgebra___redArg(v_inst_783_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0(lean_object* v_inst_785_, lean_object* v_a_786_, lean_object* v_b_787_){
_start:
{
lean_object* v___x_788_; 
v___x_788_ = lean_apply_2(v_inst_785_, v_a_786_, v_b_787_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg(lean_object* v_inst_789_, lean_object* v_inst_790_, lean_object* v_inst_791_, lean_object* v_inst_792_, lean_object* v_inst_793_, lean_object* v_inst_794_){
_start:
{
lean_object* v___f_795_; lean_object* v___f_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; 
v___f_795_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_795_, 0, v_inst_789_);
v___f_796_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_796_, 0, v_inst_790_);
v___x_797_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_797_, 0, v_inst_791_);
lean_ctor_set(v___x_797_, 1, v_inst_792_);
v___x_798_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_798_, 0, v___x_797_);
lean_ctor_set(v___x_798_, 1, v___f_795_);
v___x_799_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_799_, 0, v___x_798_);
lean_ctor_set(v___x_799_, 1, v___f_796_);
v___x_800_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_800_, 0, v___x_799_);
lean_ctor_set(v___x_800_, 1, v_inst_793_);
lean_ctor_set(v___x_800_, 2, v_inst_794_);
return v___x_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra(lean_object* v_00_u03b1_801_, lean_object* v_00_u03b2_802_, lean_object* v_inst_803_, lean_object* v_inst_804_, lean_object* v_inst_805_, lean_object* v_inst_806_, lean_object* v_inst_807_, lean_object* v_inst_808_, lean_object* v_inst_809_, lean_object* v_f_810_, lean_object* v_hf_811_, lean_object* v_le_812_, lean_object* v_lt_813_, lean_object* v_map__sup_814_, lean_object* v_map__inf_815_, lean_object* v_map__top_816_, lean_object* v_map__himp_817_){
_start:
{
lean_object* v___f_818_; lean_object* v___f_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; 
v___f_818_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_818_, 0, v_inst_803_);
v___f_819_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_819_, 0, v_inst_804_);
v___x_820_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_820_, 0, v_inst_805_);
lean_ctor_set(v___x_820_, 1, v_inst_806_);
v___x_821_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_821_, 0, v___x_820_);
lean_ctor_set(v___x_821_, 1, v___f_818_);
v___x_822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_822_, 0, v___x_821_);
lean_ctor_set(v___x_822_, 1, v___f_819_);
v___x_823_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_823_, 0, v___x_822_);
lean_ctor_set(v___x_823_, 1, v_inst_807_);
lean_ctor_set(v___x_823_, 2, v_inst_808_);
return v___x_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_824_ = _args[0];
lean_object* v_00_u03b2_825_ = _args[1];
lean_object* v_inst_826_ = _args[2];
lean_object* v_inst_827_ = _args[3];
lean_object* v_inst_828_ = _args[4];
lean_object* v_inst_829_ = _args[5];
lean_object* v_inst_830_ = _args[6];
lean_object* v_inst_831_ = _args[7];
lean_object* v_inst_832_ = _args[8];
lean_object* v_f_833_ = _args[9];
lean_object* v_hf_834_ = _args[10];
lean_object* v_le_835_ = _args[11];
lean_object* v_lt_836_ = _args[12];
lean_object* v_map__sup_837_ = _args[13];
lean_object* v_map__inf_838_ = _args[14];
lean_object* v_map__top_839_ = _args[15];
lean_object* v_map__himp_840_ = _args[16];
_start:
{
lean_object* v_res_841_; 
v_res_841_ = lp_mathlib_Function_Injective_generalizedHeytingAlgebra(v_00_u03b1_824_, v_00_u03b2_825_, v_inst_826_, v_inst_827_, v_inst_828_, v_inst_829_, v_inst_830_, v_inst_831_, v_inst_832_, v_f_833_, v_hf_834_, v_le_835_, v_lt_836_, v_map__sup_837_, v_map__inf_838_, v_map__top_839_, v_map__himp_840_);
lean_dec(v_f_833_);
lean_dec_ref(v_inst_832_);
return v_res_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra___redArg(lean_object* v_inst_842_, lean_object* v_inst_843_, lean_object* v_inst_844_, lean_object* v_inst_845_, lean_object* v_inst_846_, lean_object* v_inst_847_){
_start:
{
lean_object* v___f_848_; lean_object* v___f_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; 
v___f_848_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_848_, 0, v_inst_842_);
v___f_849_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_849_, 0, v_inst_843_);
v___x_850_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_850_, 0, v_inst_844_);
lean_ctor_set(v___x_850_, 1, v_inst_845_);
v___x_851_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_851_, 0, v___x_850_);
lean_ctor_set(v___x_851_, 1, v___f_848_);
v___x_852_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_852_, 0, v___x_851_);
lean_ctor_set(v___x_852_, 1, v___f_849_);
v___x_853_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_853_, 0, v___x_852_);
lean_ctor_set(v___x_853_, 1, v_inst_846_);
lean_ctor_set(v___x_853_, 2, v_inst_847_);
return v___x_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra(lean_object* v_00_u03b1_854_, lean_object* v_00_u03b2_855_, lean_object* v_inst_856_, lean_object* v_inst_857_, lean_object* v_inst_858_, lean_object* v_inst_859_, lean_object* v_inst_860_, lean_object* v_inst_861_, lean_object* v_inst_862_, lean_object* v_f_863_, lean_object* v_hf_864_, lean_object* v_le_865_, lean_object* v_lt_866_, lean_object* v_map__sup_867_, lean_object* v_map__inf_868_, lean_object* v_map__bot_869_, lean_object* v_map__sdiff_870_){
_start:
{
lean_object* v___f_871_; lean_object* v___f_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; 
v___f_871_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_871_, 0, v_inst_856_);
v___f_872_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_872_, 0, v_inst_857_);
v___x_873_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_873_, 0, v_inst_858_);
lean_ctor_set(v___x_873_, 1, v_inst_859_);
v___x_874_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_874_, 0, v___x_873_);
lean_ctor_set(v___x_874_, 1, v___f_871_);
v___x_875_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_875_, 0, v___x_874_);
lean_ctor_set(v___x_875_, 1, v___f_872_);
v___x_876_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_876_, 0, v___x_875_);
lean_ctor_set(v___x_876_, 1, v_inst_860_);
lean_ctor_set(v___x_876_, 2, v_inst_861_);
return v___x_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_877_ = _args[0];
lean_object* v_00_u03b2_878_ = _args[1];
lean_object* v_inst_879_ = _args[2];
lean_object* v_inst_880_ = _args[3];
lean_object* v_inst_881_ = _args[4];
lean_object* v_inst_882_ = _args[5];
lean_object* v_inst_883_ = _args[6];
lean_object* v_inst_884_ = _args[7];
lean_object* v_inst_885_ = _args[8];
lean_object* v_f_886_ = _args[9];
lean_object* v_hf_887_ = _args[10];
lean_object* v_le_888_ = _args[11];
lean_object* v_lt_889_ = _args[12];
lean_object* v_map__sup_890_ = _args[13];
lean_object* v_map__inf_891_ = _args[14];
lean_object* v_map__bot_892_ = _args[15];
lean_object* v_map__sdiff_893_ = _args[16];
_start:
{
lean_object* v_res_894_; 
v_res_894_ = lp_mathlib_Function_Injective_generalizedCoheytingAlgebra(v_00_u03b1_877_, v_00_u03b2_878_, v_inst_879_, v_inst_880_, v_inst_881_, v_inst_882_, v_inst_883_, v_inst_884_, v_inst_885_, v_f_886_, v_hf_887_, v_le_888_, v_lt_889_, v_map__sup_890_, v_map__inf_891_, v_map__bot_892_, v_map__sdiff_893_);
lean_dec(v_f_886_);
lean_dec_ref(v_inst_885_);
return v_res_894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra___redArg(lean_object* v_inst_895_, lean_object* v_inst_896_, lean_object* v_inst_897_, lean_object* v_inst_898_, lean_object* v_inst_899_, lean_object* v_inst_900_, lean_object* v_inst_901_, lean_object* v_inst_902_){
_start:
{
lean_object* v___f_903_; lean_object* v___f_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; 
v___f_903_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_903_, 0, v_inst_895_);
v___f_904_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_904_, 0, v_inst_896_);
v___x_905_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_905_, 0, v_inst_897_);
lean_ctor_set(v___x_905_, 1, v_inst_898_);
v___x_906_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_906_, 0, v___x_905_);
lean_ctor_set(v___x_906_, 1, v___f_903_);
v___x_907_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_907_, 0, v___x_906_);
lean_ctor_set(v___x_907_, 1, v___f_904_);
v___x_908_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_908_, 0, v___x_907_);
lean_ctor_set(v___x_908_, 1, v_inst_899_);
lean_ctor_set(v___x_908_, 2, v_inst_902_);
v___x_909_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_909_, 0, v___x_908_);
lean_ctor_set(v___x_909_, 1, v_inst_900_);
lean_ctor_set(v___x_909_, 2, v_inst_901_);
return v___x_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra(lean_object* v_00_u03b1_910_, lean_object* v_00_u03b2_911_, lean_object* v_inst_912_, lean_object* v_inst_913_, lean_object* v_inst_914_, lean_object* v_inst_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_inst_918_, lean_object* v_inst_919_, lean_object* v_inst_920_, lean_object* v_f_921_, lean_object* v_hf_922_, lean_object* v_le_923_, lean_object* v_lt_924_, lean_object* v_map__sup_925_, lean_object* v_map__inf_926_, lean_object* v_map__top_927_, lean_object* v_map__bot_928_, lean_object* v_map__compl_929_, lean_object* v_map__himp_930_){
_start:
{
lean_object* v___f_931_; lean_object* v___f_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; 
v___f_931_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_931_, 0, v_inst_912_);
v___f_932_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_932_, 0, v_inst_913_);
v___x_933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_933_, 0, v_inst_914_);
lean_ctor_set(v___x_933_, 1, v_inst_915_);
v___x_934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_934_, 0, v___x_933_);
lean_ctor_set(v___x_934_, 1, v___f_931_);
v___x_935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_935_, 0, v___x_934_);
lean_ctor_set(v___x_935_, 1, v___f_932_);
v___x_936_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_936_, 0, v___x_935_);
lean_ctor_set(v___x_936_, 1, v_inst_916_);
lean_ctor_set(v___x_936_, 2, v_inst_919_);
v___x_937_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_937_, 0, v___x_936_);
lean_ctor_set(v___x_937_, 1, v_inst_917_);
lean_ctor_set(v___x_937_, 2, v_inst_918_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_938_ = _args[0];
lean_object* v_00_u03b2_939_ = _args[1];
lean_object* v_inst_940_ = _args[2];
lean_object* v_inst_941_ = _args[3];
lean_object* v_inst_942_ = _args[4];
lean_object* v_inst_943_ = _args[5];
lean_object* v_inst_944_ = _args[6];
lean_object* v_inst_945_ = _args[7];
lean_object* v_inst_946_ = _args[8];
lean_object* v_inst_947_ = _args[9];
lean_object* v_inst_948_ = _args[10];
lean_object* v_f_949_ = _args[11];
lean_object* v_hf_950_ = _args[12];
lean_object* v_le_951_ = _args[13];
lean_object* v_lt_952_ = _args[14];
lean_object* v_map__sup_953_ = _args[15];
lean_object* v_map__inf_954_ = _args[16];
lean_object* v_map__top_955_ = _args[17];
lean_object* v_map__bot_956_ = _args[18];
lean_object* v_map__compl_957_ = _args[19];
lean_object* v_map__himp_958_ = _args[20];
_start:
{
lean_object* v_res_959_; 
v_res_959_ = lp_mathlib_Function_Injective_heytingAlgebra(v_00_u03b1_938_, v_00_u03b2_939_, v_inst_940_, v_inst_941_, v_inst_942_, v_inst_943_, v_inst_944_, v_inst_945_, v_inst_946_, v_inst_947_, v_inst_948_, v_f_949_, v_hf_950_, v_le_951_, v_lt_952_, v_map__sup_953_, v_map__inf_954_, v_map__top_955_, v_map__bot_956_, v_map__compl_957_, v_map__himp_958_);
lean_dec(v_f_949_);
lean_dec_ref(v_inst_948_);
return v_res_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___redArg(lean_object* v_inst_960_, lean_object* v_inst_961_, lean_object* v_inst_962_, lean_object* v_inst_963_, lean_object* v_inst_964_, lean_object* v_inst_965_, lean_object* v_inst_966_, lean_object* v_inst_967_){
_start:
{
lean_object* v___f_968_; lean_object* v___f_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; 
v___f_968_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_968_, 0, v_inst_961_);
v___f_969_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_969_, 0, v_inst_960_);
v___x_970_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_970_, 0, v_inst_962_);
lean_ctor_set(v___x_970_, 1, v_inst_963_);
v___x_971_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_971_, 0, v___x_970_);
lean_ctor_set(v___x_971_, 1, v___f_968_);
v___x_972_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_972_, 0, v___x_971_);
lean_ctor_set(v___x_972_, 1, v___f_969_);
v___x_973_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_973_, 0, v___x_972_);
lean_ctor_set(v___x_973_, 1, v_inst_964_);
lean_ctor_set(v___x_973_, 2, v_inst_967_);
v___x_974_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_974_, 0, v___x_973_);
lean_ctor_set(v___x_974_, 1, v_inst_965_);
lean_ctor_set(v___x_974_, 2, v_inst_966_);
return v___x_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra(lean_object* v_00_u03b1_975_, lean_object* v_00_u03b2_976_, lean_object* v_inst_977_, lean_object* v_inst_978_, lean_object* v_inst_979_, lean_object* v_inst_980_, lean_object* v_inst_981_, lean_object* v_inst_982_, lean_object* v_inst_983_, lean_object* v_inst_984_, lean_object* v_inst_985_, lean_object* v_f_986_, lean_object* v_hf_987_, lean_object* v_le_988_, lean_object* v_lt_989_, lean_object* v_map__inf_990_, lean_object* v_map__sup_991_, lean_object* v_map__bot_992_, lean_object* v_map__top_993_, lean_object* v_map__compl_994_, lean_object* v_map__himp_995_){
_start:
{
lean_object* v___x_996_; 
v___x_996_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v_inst_977_, v_inst_978_, v_inst_979_, v_inst_980_, v_inst_981_, v_inst_982_, v_inst_983_, v_inst_984_);
return v___x_996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_997_ = _args[0];
lean_object* v_00_u03b2_998_ = _args[1];
lean_object* v_inst_999_ = _args[2];
lean_object* v_inst_1000_ = _args[3];
lean_object* v_inst_1001_ = _args[4];
lean_object* v_inst_1002_ = _args[5];
lean_object* v_inst_1003_ = _args[6];
lean_object* v_inst_1004_ = _args[7];
lean_object* v_inst_1005_ = _args[8];
lean_object* v_inst_1006_ = _args[9];
lean_object* v_inst_1007_ = _args[10];
lean_object* v_f_1008_ = _args[11];
lean_object* v_hf_1009_ = _args[12];
lean_object* v_le_1010_ = _args[13];
lean_object* v_lt_1011_ = _args[14];
lean_object* v_map__inf_1012_ = _args[15];
lean_object* v_map__sup_1013_ = _args[16];
lean_object* v_map__bot_1014_ = _args[17];
lean_object* v_map__top_1015_ = _args[18];
lean_object* v_map__compl_1016_ = _args[19];
lean_object* v_map__himp_1017_ = _args[20];
_start:
{
lean_object* v_res_1018_; 
v_res_1018_ = lp_mathlib_Function_Injective_coheytingAlgebra(v_00_u03b1_997_, v_00_u03b2_998_, v_inst_999_, v_inst_1000_, v_inst_1001_, v_inst_1002_, v_inst_1003_, v_inst_1004_, v_inst_1005_, v_inst_1006_, v_inst_1007_, v_f_1008_, v_hf_1009_, v_le_1010_, v_lt_1011_, v_map__inf_1012_, v_map__sup_1013_, v_map__bot_1014_, v_map__top_1015_, v_map__compl_1016_, v_map__himp_1017_);
lean_dec(v_f_1008_);
lean_dec_ref(v_inst_1007_);
return v_res_1018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra___redArg(lean_object* v_inst_1019_, lean_object* v_inst_1020_, lean_object* v_inst_1021_, lean_object* v_inst_1022_, lean_object* v_inst_1023_, lean_object* v_inst_1024_, lean_object* v_inst_1025_, lean_object* v_inst_1026_, lean_object* v_inst_1027_, lean_object* v_inst_1028_){
_start:
{
lean_object* v___f_1029_; lean_object* v___f_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; 
v___f_1029_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1029_, 0, v_inst_1019_);
v___f_1030_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1030_, 0, v_inst_1020_);
v___x_1031_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1031_, 0, v_inst_1021_);
lean_ctor_set(v___x_1031_, 1, v_inst_1022_);
v___x_1032_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1032_, 0, v___x_1031_);
lean_ctor_set(v___x_1032_, 1, v___f_1029_);
v___x_1033_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1033_, 0, v___x_1032_);
lean_ctor_set(v___x_1033_, 1, v___f_1030_);
v___x_1034_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1034_, 0, v___x_1033_);
lean_ctor_set(v___x_1034_, 1, v_inst_1023_);
lean_ctor_set(v___x_1034_, 2, v_inst_1027_);
v___x_1035_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1035_, 0, v___x_1034_);
lean_ctor_set(v___x_1035_, 1, v_inst_1024_);
lean_ctor_set(v___x_1035_, 2, v_inst_1025_);
v___x_1036_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1036_, 0, v___x_1035_);
lean_ctor_set(v___x_1036_, 1, v_inst_1028_);
lean_ctor_set(v___x_1036_, 2, v_inst_1026_);
return v___x_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra(lean_object* v_00_u03b1_1037_, lean_object* v_00_u03b2_1038_, lean_object* v_inst_1039_, lean_object* v_inst_1040_, lean_object* v_inst_1041_, lean_object* v_inst_1042_, lean_object* v_inst_1043_, lean_object* v_inst_1044_, lean_object* v_inst_1045_, lean_object* v_inst_1046_, lean_object* v_inst_1047_, lean_object* v_inst_1048_, lean_object* v_inst_1049_, lean_object* v_f_1050_, lean_object* v_hf_1051_, lean_object* v_le_1052_, lean_object* v_lt_1053_, lean_object* v_map__sup_1054_, lean_object* v_map__inf_1055_, lean_object* v_map__top_1056_, lean_object* v_map__bot_1057_, lean_object* v_map__compl_1058_, lean_object* v_map__hnot_1059_, lean_object* v_map__himp_1060_, lean_object* v_map__sdiff_1061_){
_start:
{
lean_object* v___f_1062_; lean_object* v___f_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; 
v___f_1062_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1062_, 0, v_inst_1039_);
v___f_1063_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1063_, 0, v_inst_1040_);
v___x_1064_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1064_, 0, v_inst_1041_);
lean_ctor_set(v___x_1064_, 1, v_inst_1042_);
v___x_1065_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1065_, 0, v___x_1064_);
lean_ctor_set(v___x_1065_, 1, v___f_1062_);
v___x_1066_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1066_, 0, v___x_1065_);
lean_ctor_set(v___x_1066_, 1, v___f_1063_);
v___x_1067_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1067_, 0, v___x_1066_);
lean_ctor_set(v___x_1067_, 1, v_inst_1043_);
lean_ctor_set(v___x_1067_, 2, v_inst_1047_);
v___x_1068_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1068_, 0, v___x_1067_);
lean_ctor_set(v___x_1068_, 1, v_inst_1044_);
lean_ctor_set(v___x_1068_, 2, v_inst_1045_);
v___x_1069_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1069_, 0, v___x_1068_);
lean_ctor_set(v___x_1069_, 1, v_inst_1048_);
lean_ctor_set(v___x_1069_, 2, v_inst_1046_);
return v___x_1069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra___boxed(lean_object** _args){
lean_object* v_00_u03b1_1070_ = _args[0];
lean_object* v_00_u03b2_1071_ = _args[1];
lean_object* v_inst_1072_ = _args[2];
lean_object* v_inst_1073_ = _args[3];
lean_object* v_inst_1074_ = _args[4];
lean_object* v_inst_1075_ = _args[5];
lean_object* v_inst_1076_ = _args[6];
lean_object* v_inst_1077_ = _args[7];
lean_object* v_inst_1078_ = _args[8];
lean_object* v_inst_1079_ = _args[9];
lean_object* v_inst_1080_ = _args[10];
lean_object* v_inst_1081_ = _args[11];
lean_object* v_inst_1082_ = _args[12];
lean_object* v_f_1083_ = _args[13];
lean_object* v_hf_1084_ = _args[14];
lean_object* v_le_1085_ = _args[15];
lean_object* v_lt_1086_ = _args[16];
lean_object* v_map__sup_1087_ = _args[17];
lean_object* v_map__inf_1088_ = _args[18];
lean_object* v_map__top_1089_ = _args[19];
lean_object* v_map__bot_1090_ = _args[20];
lean_object* v_map__compl_1091_ = _args[21];
lean_object* v_map__hnot_1092_ = _args[22];
lean_object* v_map__himp_1093_ = _args[23];
lean_object* v_map__sdiff_1094_ = _args[24];
_start:
{
lean_object* v_res_1095_; 
v_res_1095_ = lp_mathlib_Function_Injective_biheytingAlgebra(v_00_u03b1_1070_, v_00_u03b2_1071_, v_inst_1072_, v_inst_1073_, v_inst_1074_, v_inst_1075_, v_inst_1076_, v_inst_1077_, v_inst_1078_, v_inst_1079_, v_inst_1080_, v_inst_1081_, v_inst_1082_, v_f_1083_, v_hf_1084_, v_le_1085_, v_lt_1086_, v_map__sup_1087_, v_map__inf_1088_, v_map__top_1089_, v_map__bot_1090_, v_map__compl_1091_, v_map__hnot_1092_, v_map__himp_1093_, v_map__sdiff_1094_);
lean_dec(v_f_1083_);
lean_dec_ref(v_inst_1082_);
return v_res_1095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1(lean_object* v_e_1096_, lean_object* v___f_1097_, lean_object* v_inf_1098_, lean_object* v_a_1099_, lean_object* v_b_1100_){
_start:
{
lean_object* v___x_1101_; lean_object* v_toFun_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; 
lean_inc_ref_n(v_e_1096_, 2);
v___x_1101_ = lp_mathlib_Equiv_symm___redArg(v_e_1096_);
v_toFun_1102_ = lean_ctor_get(v___x_1101_, 0);
lean_inc(v_toFun_1102_);
lean_dec_ref(v___x_1101_);
lean_inc(v___f_1097_);
v___x_1103_ = lean_apply_2(v___f_1097_, v_e_1096_, v_a_1099_);
v___x_1104_ = lean_apply_2(v___f_1097_, v_e_1096_, v_b_1100_);
v___x_1105_ = lean_apply_2(v_inf_1098_, v___x_1103_, v___x_1104_);
v___x_1106_ = lean_apply_1(v_toFun_1102_, v___x_1105_);
return v___x_1106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2(lean_object* v_min_1107_, lean_object* v_a_1108_, lean_object* v_b_1109_){
_start:
{
lean_object* v___x_1110_; 
v___x_1110_ = lean_apply_2(v_min_1107_, v_a_1108_, v_b_1109_);
return v___x_1110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0(lean_object* v_toSemilatticeSup_1111_, lean_object* v_e_1112_, lean_object* v___f_1113_, lean_object* v_a_1114_, lean_object* v_b_1115_){
_start:
{
lean_object* v_sup_1116_; lean_object* v___x_1117_; lean_object* v_toFun_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; 
v_sup_1116_ = lean_ctor_get(v_toSemilatticeSup_1111_, 1);
lean_inc(v_sup_1116_);
lean_dec_ref(v_toSemilatticeSup_1111_);
lean_inc_ref_n(v_e_1112_, 2);
v___x_1117_ = lp_mathlib_Equiv_symm___redArg(v_e_1112_);
v_toFun_1118_ = lean_ctor_get(v___x_1117_, 0);
lean_inc(v_toFun_1118_);
lean_dec_ref(v___x_1117_);
lean_inc(v___f_1113_);
v___x_1119_ = lean_apply_2(v___f_1113_, v_e_1112_, v_a_1114_);
v___x_1120_ = lean_apply_2(v___f_1113_, v_e_1112_, v_b_1115_);
v___x_1121_ = lean_apply_2(v_sup_1116_, v___x_1119_, v___x_1120_);
v___x_1122_ = lean_apply_1(v_toFun_1118_, v___x_1121_);
return v___x_1122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5(lean_object* v___f_1123_, lean_object* v_a_1124_, lean_object* v_b_1125_){
_start:
{
lean_object* v___x_1126_; 
v___x_1126_ = lean_apply_2(v___f_1123_, v_a_1124_, v_b_1125_);
return v___x_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__3(lean_object* v___f_1127_, lean_object* v_e_1128_, lean_object* v_toHImp_1129_, lean_object* v_toFun_1130_, lean_object* v_a_1131_, lean_object* v_b_1132_){
_start:
{
lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; 
lean_inc(v___f_1127_);
lean_inc_ref(v_e_1128_);
v___x_1133_ = lean_apply_2(v___f_1127_, v_e_1128_, v_a_1131_);
v___x_1134_ = lean_apply_2(v___f_1127_, v_e_1128_, v_b_1132_);
v___x_1135_ = lean_apply_2(v_toHImp_1129_, v___x_1133_, v___x_1134_);
v___x_1136_ = lean_apply_1(v_toFun_1130_, v___x_1135_);
return v___x_1136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg(lean_object* v_e_1137_, lean_object* v_inst_1138_){
_start:
{
lean_object* v_toLattice_1139_; lean_object* v_toOrderTop_1140_; lean_object* v_toHImp_1141_; lean_object* v___x_1143_; uint8_t v_isShared_1144_; uint8_t v_isSharedCheck_1214_; 
v_toLattice_1139_ = lean_ctor_get(v_inst_1138_, 0);
v_toOrderTop_1140_ = lean_ctor_get(v_inst_1138_, 1);
v_toHImp_1141_ = lean_ctor_get(v_inst_1138_, 2);
v_isSharedCheck_1214_ = !lean_is_exclusive(v_inst_1138_);
if (v_isSharedCheck_1214_ == 0)
{
v___x_1143_ = v_inst_1138_;
v_isShared_1144_ = v_isSharedCheck_1214_;
goto v_resetjp_1142_;
}
else
{
lean_inc(v_toHImp_1141_);
lean_inc(v_toOrderTop_1140_);
lean_inc(v_toLattice_1139_);
lean_dec(v_inst_1138_);
v___x_1143_ = lean_box(0);
v_isShared_1144_ = v_isSharedCheck_1214_;
goto v_resetjp_1142_;
}
v_resetjp_1142_:
{
lean_object* v_toSemilatticeSup_1145_; lean_object* v_inf_1146_; lean_object* v___x_1148_; uint8_t v_isShared_1149_; uint8_t v_isSharedCheck_1213_; 
v_toSemilatticeSup_1145_ = lean_ctor_get(v_toLattice_1139_, 0);
v_inf_1146_ = lean_ctor_get(v_toLattice_1139_, 1);
v_isSharedCheck_1213_ = !lean_is_exclusive(v_toLattice_1139_);
if (v_isSharedCheck_1213_ == 0)
{
v___x_1148_ = v_toLattice_1139_;
v_isShared_1149_ = v_isSharedCheck_1213_;
goto v_resetjp_1147_;
}
else
{
lean_inc(v_inf_1146_);
lean_inc(v_toSemilatticeSup_1145_);
lean_dec(v_toLattice_1139_);
v___x_1148_ = lean_box(0);
v_isShared_1149_ = v_isSharedCheck_1213_;
goto v_resetjp_1147_;
}
v_resetjp_1147_:
{
lean_object* v___f_1150_; lean_object* v_min_1151_; lean_object* v_le_1152_; lean_object* v_lt_1153_; lean_object* v_semilatticeInf_1154_; lean_object* v_toPartialOrder_1155_; lean_object* v___x_1157_; uint8_t v_isShared_1158_; uint8_t v_isSharedCheck_1211_; 
v___f_1150_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc_ref(v_e_1137_);
v_min_1151_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1151_, 0, v_e_1137_);
lean_closure_set(v_min_1151_, 1, v___f_1150_);
lean_closure_set(v_min_1151_, 2, v_inf_1146_);
v_le_1152_ = lean_box(0);
v_lt_1153_ = lean_box(0);
lean_inc_ref(v_min_1151_);
v_semilatticeInf_1154_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1151_, v_le_1152_, v_lt_1153_);
v_toPartialOrder_1155_ = lean_ctor_get(v_semilatticeInf_1154_, 0);
v_isSharedCheck_1211_ = !lean_is_exclusive(v_semilatticeInf_1154_);
if (v_isSharedCheck_1211_ == 0)
{
lean_object* v_unused_1212_; 
v_unused_1212_ = lean_ctor_get(v_semilatticeInf_1154_, 1);
lean_dec(v_unused_1212_);
v___x_1157_ = v_semilatticeInf_1154_;
v_isShared_1158_ = v_isSharedCheck_1211_;
goto v_resetjp_1156_;
}
else
{
lean_inc(v_toPartialOrder_1155_);
lean_dec(v_semilatticeInf_1154_);
v___x_1157_ = lean_box(0);
v_isShared_1158_ = v_isSharedCheck_1211_;
goto v_resetjp_1156_;
}
v_resetjp_1156_:
{
lean_object* v_toLE_1159_; lean_object* v_toLT_1160_; lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1210_; 
v_toLE_1159_ = lean_ctor_get(v_toPartialOrder_1155_, 0);
v_toLT_1160_ = lean_ctor_get(v_toPartialOrder_1155_, 1);
v_isSharedCheck_1210_ = !lean_is_exclusive(v_toPartialOrder_1155_);
if (v_isSharedCheck_1210_ == 0)
{
v___x_1162_ = v_toPartialOrder_1155_;
v_isShared_1163_ = v_isSharedCheck_1210_;
goto v_resetjp_1161_;
}
else
{
lean_inc(v_toLT_1160_);
lean_inc(v_toLE_1159_);
lean_dec(v_toPartialOrder_1155_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1210_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v___f_1164_; lean_object* v___f_1165_; lean_object* v___x_1167_; 
v___f_1164_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1164_, 0, v_min_1151_);
lean_inc_ref(v_e_1137_);
v___f_1165_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1165_, 0, v_toSemilatticeSup_1145_);
lean_closure_set(v___f_1165_, 1, v_e_1137_);
lean_closure_set(v___f_1165_, 2, v___f_1150_);
if (v_isShared_1163_ == 0)
{
v___x_1167_ = v___x_1162_;
goto v_reusejp_1166_;
}
else
{
lean_object* v_reuseFailAlloc_1209_; 
v_reuseFailAlloc_1209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1209_, 0, v_toLE_1159_);
lean_ctor_set(v_reuseFailAlloc_1209_, 1, v_toLT_1160_);
v___x_1167_ = v_reuseFailAlloc_1209_;
goto v_reusejp_1166_;
}
v_reusejp_1166_:
{
lean_object* v___x_1169_; 
lean_inc_ref(v___f_1165_);
if (v_isShared_1158_ == 0)
{
lean_ctor_set(v___x_1157_, 1, v___f_1165_);
lean_ctor_set(v___x_1157_, 0, v___x_1167_);
v___x_1169_ = v___x_1157_;
goto v_reusejp_1168_;
}
else
{
lean_object* v_reuseFailAlloc_1208_; 
v_reuseFailAlloc_1208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1208_, 0, v___x_1167_);
lean_ctor_set(v_reuseFailAlloc_1208_, 1, v___f_1165_);
v___x_1169_ = v_reuseFailAlloc_1208_;
goto v_reusejp_1168_;
}
v_reusejp_1168_:
{
lean_object* v_lattice_1171_; 
lean_inc_ref(v___f_1164_);
if (v_isShared_1149_ == 0)
{
lean_ctor_set(v___x_1148_, 1, v___f_1164_);
lean_ctor_set(v___x_1148_, 0, v___x_1169_);
v_lattice_1171_ = v___x_1148_;
goto v_reusejp_1170_;
}
else
{
lean_object* v_reuseFailAlloc_1207_; 
v_reuseFailAlloc_1207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1207_, 0, v___x_1169_);
lean_ctor_set(v_reuseFailAlloc_1207_, 1, v___f_1164_);
v_lattice_1171_ = v_reuseFailAlloc_1207_;
goto v_reusejp_1170_;
}
v_reusejp_1170_:
{
lean_object* v___x_1172_; lean_object* v_toFun_1173_; lean_object* v___x_1175_; uint8_t v_isShared_1176_; uint8_t v_isSharedCheck_1205_; 
lean_inc_ref(v_e_1137_);
v___x_1172_ = lp_mathlib_Equiv_symm___redArg(v_e_1137_);
v_toFun_1173_ = lean_ctor_get(v___x_1172_, 0);
v_isSharedCheck_1205_ = !lean_is_exclusive(v___x_1172_);
if (v_isSharedCheck_1205_ == 0)
{
lean_object* v_unused_1206_; 
v_unused_1206_ = lean_ctor_get(v___x_1172_, 1);
lean_dec(v_unused_1206_);
v___x_1175_ = v___x_1172_;
v_isShared_1176_ = v_isSharedCheck_1205_;
goto v_resetjp_1174_;
}
else
{
lean_inc(v_toFun_1173_);
lean_dec(v___x_1172_);
v___x_1175_ = lean_box(0);
v_isShared_1176_ = v_isSharedCheck_1205_;
goto v_resetjp_1174_;
}
v_resetjp_1174_:
{
lean_object* v___x_1177_; lean_object* v_toPartialOrder_1178_; lean_object* v___x_1180_; uint8_t v_isShared_1181_; uint8_t v_isSharedCheck_1203_; 
v___x_1177_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1171_);
v_toPartialOrder_1178_ = lean_ctor_get(v___x_1177_, 0);
v_isSharedCheck_1203_ = !lean_is_exclusive(v___x_1177_);
if (v_isSharedCheck_1203_ == 0)
{
lean_object* v_unused_1204_; 
v_unused_1204_ = lean_ctor_get(v___x_1177_, 1);
lean_dec(v_unused_1204_);
v___x_1180_ = v___x_1177_;
v_isShared_1181_ = v_isSharedCheck_1203_;
goto v_resetjp_1179_;
}
else
{
lean_inc(v_toPartialOrder_1178_);
lean_dec(v___x_1177_);
v___x_1180_ = lean_box(0);
v_isShared_1181_ = v_isSharedCheck_1203_;
goto v_resetjp_1179_;
}
v_resetjp_1179_:
{
lean_object* v_toLE_1182_; lean_object* v_toLT_1183_; lean_object* v___x_1185_; uint8_t v_isShared_1186_; uint8_t v_isSharedCheck_1202_; 
v_toLE_1182_ = lean_ctor_get(v_toPartialOrder_1178_, 0);
v_toLT_1183_ = lean_ctor_get(v_toPartialOrder_1178_, 1);
v_isSharedCheck_1202_ = !lean_is_exclusive(v_toPartialOrder_1178_);
if (v_isSharedCheck_1202_ == 0)
{
v___x_1185_ = v_toPartialOrder_1178_;
v_isShared_1186_ = v_isSharedCheck_1202_;
goto v_resetjp_1184_;
}
else
{
lean_inc(v_toLT_1183_);
lean_inc(v_toLE_1182_);
lean_dec(v_toPartialOrder_1178_);
v___x_1185_ = lean_box(0);
v_isShared_1186_ = v_isSharedCheck_1202_;
goto v_resetjp_1184_;
}
v_resetjp_1184_:
{
lean_object* v___f_1187_; lean_object* v_himp_1188_; lean_object* v_top_1189_; lean_object* v___x_1191_; 
v___f_1187_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1187_, 0, v___f_1165_);
lean_inc(v_toFun_1173_);
v_himp_1188_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__3), 6, 4);
lean_closure_set(v_himp_1188_, 0, v___f_1150_);
lean_closure_set(v_himp_1188_, 1, v_e_1137_);
lean_closure_set(v_himp_1188_, 2, v_toHImp_1141_);
lean_closure_set(v_himp_1188_, 3, v_toFun_1173_);
v_top_1189_ = lean_apply_1(v_toFun_1173_, v_toOrderTop_1140_);
if (v_isShared_1186_ == 0)
{
v___x_1191_ = v___x_1185_;
goto v_reusejp_1190_;
}
else
{
lean_object* v_reuseFailAlloc_1201_; 
v_reuseFailAlloc_1201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1201_, 0, v_toLE_1182_);
lean_ctor_set(v_reuseFailAlloc_1201_, 1, v_toLT_1183_);
v___x_1191_ = v_reuseFailAlloc_1201_;
goto v_reusejp_1190_;
}
v_reusejp_1190_:
{
lean_object* v___x_1193_; 
if (v_isShared_1181_ == 0)
{
lean_ctor_set(v___x_1180_, 1, v___f_1187_);
lean_ctor_set(v___x_1180_, 0, v___x_1191_);
v___x_1193_ = v___x_1180_;
goto v_reusejp_1192_;
}
else
{
lean_object* v_reuseFailAlloc_1200_; 
v_reuseFailAlloc_1200_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1200_, 0, v___x_1191_);
lean_ctor_set(v_reuseFailAlloc_1200_, 1, v___f_1187_);
v___x_1193_ = v_reuseFailAlloc_1200_;
goto v_reusejp_1192_;
}
v_reusejp_1192_:
{
lean_object* v___x_1195_; 
if (v_isShared_1176_ == 0)
{
lean_ctor_set(v___x_1175_, 1, v___f_1164_);
lean_ctor_set(v___x_1175_, 0, v___x_1193_);
v___x_1195_ = v___x_1175_;
goto v_reusejp_1194_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v___x_1193_);
lean_ctor_set(v_reuseFailAlloc_1199_, 1, v___f_1164_);
v___x_1195_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1194_;
}
v_reusejp_1194_:
{
lean_object* v___x_1197_; 
if (v_isShared_1144_ == 0)
{
lean_ctor_set(v___x_1143_, 2, v_himp_1188_);
lean_ctor_set(v___x_1143_, 1, v_top_1189_);
lean_ctor_set(v___x_1143_, 0, v___x_1195_);
v___x_1197_ = v___x_1143_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v___x_1195_);
lean_ctor_set(v_reuseFailAlloc_1198_, 1, v_top_1189_);
lean_ctor_set(v_reuseFailAlloc_1198_, 2, v_himp_1188_);
v___x_1197_ = v_reuseFailAlloc_1198_;
goto v_reusejp_1196_;
}
v_reusejp_1196_:
{
return v___x_1197_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedHeytingAlgebra(lean_object* v_00_u03b1_1215_, lean_object* v_00_u03b2_1216_, lean_object* v_e_1217_, lean_object* v_inst_1218_){
_start:
{
lean_object* v_toLattice_1219_; lean_object* v_toOrderTop_1220_; lean_object* v_toHImp_1221_; lean_object* v___x_1223_; uint8_t v_isShared_1224_; uint8_t v_isSharedCheck_1294_; 
v_toLattice_1219_ = lean_ctor_get(v_inst_1218_, 0);
v_toOrderTop_1220_ = lean_ctor_get(v_inst_1218_, 1);
v_toHImp_1221_ = lean_ctor_get(v_inst_1218_, 2);
v_isSharedCheck_1294_ = !lean_is_exclusive(v_inst_1218_);
if (v_isSharedCheck_1294_ == 0)
{
v___x_1223_ = v_inst_1218_;
v_isShared_1224_ = v_isSharedCheck_1294_;
goto v_resetjp_1222_;
}
else
{
lean_inc(v_toHImp_1221_);
lean_inc(v_toOrderTop_1220_);
lean_inc(v_toLattice_1219_);
lean_dec(v_inst_1218_);
v___x_1223_ = lean_box(0);
v_isShared_1224_ = v_isSharedCheck_1294_;
goto v_resetjp_1222_;
}
v_resetjp_1222_:
{
lean_object* v_toSemilatticeSup_1225_; lean_object* v_inf_1226_; lean_object* v___x_1228_; uint8_t v_isShared_1229_; uint8_t v_isSharedCheck_1293_; 
v_toSemilatticeSup_1225_ = lean_ctor_get(v_toLattice_1219_, 0);
v_inf_1226_ = lean_ctor_get(v_toLattice_1219_, 1);
v_isSharedCheck_1293_ = !lean_is_exclusive(v_toLattice_1219_);
if (v_isSharedCheck_1293_ == 0)
{
v___x_1228_ = v_toLattice_1219_;
v_isShared_1229_ = v_isSharedCheck_1293_;
goto v_resetjp_1227_;
}
else
{
lean_inc(v_inf_1226_);
lean_inc(v_toSemilatticeSup_1225_);
lean_dec(v_toLattice_1219_);
v___x_1228_ = lean_box(0);
v_isShared_1229_ = v_isSharedCheck_1293_;
goto v_resetjp_1227_;
}
v_resetjp_1227_:
{
lean_object* v___f_1230_; lean_object* v_min_1231_; lean_object* v_le_1232_; lean_object* v_lt_1233_; lean_object* v_semilatticeInf_1234_; lean_object* v_toPartialOrder_1235_; lean_object* v___x_1237_; uint8_t v_isShared_1238_; uint8_t v_isSharedCheck_1291_; 
v___f_1230_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc_ref(v_e_1217_);
v_min_1231_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1231_, 0, v_e_1217_);
lean_closure_set(v_min_1231_, 1, v___f_1230_);
lean_closure_set(v_min_1231_, 2, v_inf_1226_);
v_le_1232_ = lean_box(0);
v_lt_1233_ = lean_box(0);
lean_inc_ref(v_min_1231_);
v_semilatticeInf_1234_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1231_, v_le_1232_, v_lt_1233_);
v_toPartialOrder_1235_ = lean_ctor_get(v_semilatticeInf_1234_, 0);
v_isSharedCheck_1291_ = !lean_is_exclusive(v_semilatticeInf_1234_);
if (v_isSharedCheck_1291_ == 0)
{
lean_object* v_unused_1292_; 
v_unused_1292_ = lean_ctor_get(v_semilatticeInf_1234_, 1);
lean_dec(v_unused_1292_);
v___x_1237_ = v_semilatticeInf_1234_;
v_isShared_1238_ = v_isSharedCheck_1291_;
goto v_resetjp_1236_;
}
else
{
lean_inc(v_toPartialOrder_1235_);
lean_dec(v_semilatticeInf_1234_);
v___x_1237_ = lean_box(0);
v_isShared_1238_ = v_isSharedCheck_1291_;
goto v_resetjp_1236_;
}
v_resetjp_1236_:
{
lean_object* v_toLE_1239_; lean_object* v_toLT_1240_; lean_object* v___x_1242_; uint8_t v_isShared_1243_; uint8_t v_isSharedCheck_1290_; 
v_toLE_1239_ = lean_ctor_get(v_toPartialOrder_1235_, 0);
v_toLT_1240_ = lean_ctor_get(v_toPartialOrder_1235_, 1);
v_isSharedCheck_1290_ = !lean_is_exclusive(v_toPartialOrder_1235_);
if (v_isSharedCheck_1290_ == 0)
{
v___x_1242_ = v_toPartialOrder_1235_;
v_isShared_1243_ = v_isSharedCheck_1290_;
goto v_resetjp_1241_;
}
else
{
lean_inc(v_toLT_1240_);
lean_inc(v_toLE_1239_);
lean_dec(v_toPartialOrder_1235_);
v___x_1242_ = lean_box(0);
v_isShared_1243_ = v_isSharedCheck_1290_;
goto v_resetjp_1241_;
}
v_resetjp_1241_:
{
lean_object* v___f_1244_; lean_object* v___f_1245_; lean_object* v___x_1247_; 
v___f_1244_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1244_, 0, v_min_1231_);
lean_inc_ref(v_e_1217_);
v___f_1245_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1245_, 0, v_toSemilatticeSup_1225_);
lean_closure_set(v___f_1245_, 1, v_e_1217_);
lean_closure_set(v___f_1245_, 2, v___f_1230_);
if (v_isShared_1243_ == 0)
{
v___x_1247_ = v___x_1242_;
goto v_reusejp_1246_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v_toLE_1239_);
lean_ctor_set(v_reuseFailAlloc_1289_, 1, v_toLT_1240_);
v___x_1247_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1246_;
}
v_reusejp_1246_:
{
lean_object* v___x_1249_; 
lean_inc_ref(v___f_1245_);
if (v_isShared_1238_ == 0)
{
lean_ctor_set(v___x_1237_, 1, v___f_1245_);
lean_ctor_set(v___x_1237_, 0, v___x_1247_);
v___x_1249_ = v___x_1237_;
goto v_reusejp_1248_;
}
else
{
lean_object* v_reuseFailAlloc_1288_; 
v_reuseFailAlloc_1288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1288_, 0, v___x_1247_);
lean_ctor_set(v_reuseFailAlloc_1288_, 1, v___f_1245_);
v___x_1249_ = v_reuseFailAlloc_1288_;
goto v_reusejp_1248_;
}
v_reusejp_1248_:
{
lean_object* v_lattice_1251_; 
lean_inc_ref(v___f_1244_);
if (v_isShared_1229_ == 0)
{
lean_ctor_set(v___x_1228_, 1, v___f_1244_);
lean_ctor_set(v___x_1228_, 0, v___x_1249_);
v_lattice_1251_ = v___x_1228_;
goto v_reusejp_1250_;
}
else
{
lean_object* v_reuseFailAlloc_1287_; 
v_reuseFailAlloc_1287_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1287_, 0, v___x_1249_);
lean_ctor_set(v_reuseFailAlloc_1287_, 1, v___f_1244_);
v_lattice_1251_ = v_reuseFailAlloc_1287_;
goto v_reusejp_1250_;
}
v_reusejp_1250_:
{
lean_object* v___x_1252_; lean_object* v_toFun_1253_; lean_object* v___x_1255_; uint8_t v_isShared_1256_; uint8_t v_isSharedCheck_1285_; 
lean_inc_ref(v_e_1217_);
v___x_1252_ = lp_mathlib_Equiv_symm___redArg(v_e_1217_);
v_toFun_1253_ = lean_ctor_get(v___x_1252_, 0);
v_isSharedCheck_1285_ = !lean_is_exclusive(v___x_1252_);
if (v_isSharedCheck_1285_ == 0)
{
lean_object* v_unused_1286_; 
v_unused_1286_ = lean_ctor_get(v___x_1252_, 1);
lean_dec(v_unused_1286_);
v___x_1255_ = v___x_1252_;
v_isShared_1256_ = v_isSharedCheck_1285_;
goto v_resetjp_1254_;
}
else
{
lean_inc(v_toFun_1253_);
lean_dec(v___x_1252_);
v___x_1255_ = lean_box(0);
v_isShared_1256_ = v_isSharedCheck_1285_;
goto v_resetjp_1254_;
}
v_resetjp_1254_:
{
lean_object* v___x_1257_; lean_object* v_toPartialOrder_1258_; lean_object* v___x_1260_; uint8_t v_isShared_1261_; uint8_t v_isSharedCheck_1283_; 
v___x_1257_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1251_);
v_toPartialOrder_1258_ = lean_ctor_get(v___x_1257_, 0);
v_isSharedCheck_1283_ = !lean_is_exclusive(v___x_1257_);
if (v_isSharedCheck_1283_ == 0)
{
lean_object* v_unused_1284_; 
v_unused_1284_ = lean_ctor_get(v___x_1257_, 1);
lean_dec(v_unused_1284_);
v___x_1260_ = v___x_1257_;
v_isShared_1261_ = v_isSharedCheck_1283_;
goto v_resetjp_1259_;
}
else
{
lean_inc(v_toPartialOrder_1258_);
lean_dec(v___x_1257_);
v___x_1260_ = lean_box(0);
v_isShared_1261_ = v_isSharedCheck_1283_;
goto v_resetjp_1259_;
}
v_resetjp_1259_:
{
lean_object* v_toLE_1262_; lean_object* v_toLT_1263_; lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1282_; 
v_toLE_1262_ = lean_ctor_get(v_toPartialOrder_1258_, 0);
v_toLT_1263_ = lean_ctor_get(v_toPartialOrder_1258_, 1);
v_isSharedCheck_1282_ = !lean_is_exclusive(v_toPartialOrder_1258_);
if (v_isSharedCheck_1282_ == 0)
{
v___x_1265_ = v_toPartialOrder_1258_;
v_isShared_1266_ = v_isSharedCheck_1282_;
goto v_resetjp_1264_;
}
else
{
lean_inc(v_toLT_1263_);
lean_inc(v_toLE_1262_);
lean_dec(v_toPartialOrder_1258_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1282_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
lean_object* v___f_1267_; lean_object* v_himp_1268_; lean_object* v_top_1269_; lean_object* v___x_1271_; 
v___f_1267_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1267_, 0, v___f_1245_);
lean_inc(v_toFun_1253_);
v_himp_1268_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__3), 6, 4);
lean_closure_set(v_himp_1268_, 0, v___f_1230_);
lean_closure_set(v_himp_1268_, 1, v_e_1217_);
lean_closure_set(v_himp_1268_, 2, v_toHImp_1221_);
lean_closure_set(v_himp_1268_, 3, v_toFun_1253_);
v_top_1269_ = lean_apply_1(v_toFun_1253_, v_toOrderTop_1220_);
if (v_isShared_1266_ == 0)
{
v___x_1271_ = v___x_1265_;
goto v_reusejp_1270_;
}
else
{
lean_object* v_reuseFailAlloc_1281_; 
v_reuseFailAlloc_1281_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1281_, 0, v_toLE_1262_);
lean_ctor_set(v_reuseFailAlloc_1281_, 1, v_toLT_1263_);
v___x_1271_ = v_reuseFailAlloc_1281_;
goto v_reusejp_1270_;
}
v_reusejp_1270_:
{
lean_object* v___x_1273_; 
if (v_isShared_1261_ == 0)
{
lean_ctor_set(v___x_1260_, 1, v___f_1267_);
lean_ctor_set(v___x_1260_, 0, v___x_1271_);
v___x_1273_ = v___x_1260_;
goto v_reusejp_1272_;
}
else
{
lean_object* v_reuseFailAlloc_1280_; 
v_reuseFailAlloc_1280_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1280_, 0, v___x_1271_);
lean_ctor_set(v_reuseFailAlloc_1280_, 1, v___f_1267_);
v___x_1273_ = v_reuseFailAlloc_1280_;
goto v_reusejp_1272_;
}
v_reusejp_1272_:
{
lean_object* v___x_1275_; 
if (v_isShared_1256_ == 0)
{
lean_ctor_set(v___x_1255_, 1, v___f_1244_);
lean_ctor_set(v___x_1255_, 0, v___x_1273_);
v___x_1275_ = v___x_1255_;
goto v_reusejp_1274_;
}
else
{
lean_object* v_reuseFailAlloc_1279_; 
v_reuseFailAlloc_1279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1279_, 0, v___x_1273_);
lean_ctor_set(v_reuseFailAlloc_1279_, 1, v___f_1244_);
v___x_1275_ = v_reuseFailAlloc_1279_;
goto v_reusejp_1274_;
}
v_reusejp_1274_:
{
lean_object* v___x_1277_; 
if (v_isShared_1224_ == 0)
{
lean_ctor_set(v___x_1223_, 2, v_himp_1268_);
lean_ctor_set(v___x_1223_, 1, v_top_1269_);
lean_ctor_set(v___x_1223_, 0, v___x_1275_);
v___x_1277_ = v___x_1223_;
goto v_reusejp_1276_;
}
else
{
lean_object* v_reuseFailAlloc_1278_; 
v_reuseFailAlloc_1278_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1278_, 0, v___x_1275_);
lean_ctor_set(v_reuseFailAlloc_1278_, 1, v_top_1269_);
lean_ctor_set(v_reuseFailAlloc_1278_, 2, v_himp_1268_);
v___x_1277_ = v_reuseFailAlloc_1278_;
goto v_reusejp_1276_;
}
v_reusejp_1276_:
{
return v___x_1277_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg___lam__8(lean_object* v___f_1295_, lean_object* v_e_1296_, lean_object* v_toSDiff_1297_, lean_object* v_toFun_1298_, lean_object* v_a_1299_, lean_object* v_b_1300_){
_start:
{
lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; 
lean_inc(v___f_1295_);
lean_inc_ref(v_e_1296_);
v___x_1301_ = lean_apply_2(v___f_1295_, v_e_1296_, v_a_1299_);
v___x_1302_ = lean_apply_2(v___f_1295_, v_e_1296_, v_b_1300_);
v___x_1303_ = lean_apply_2(v_toSDiff_1297_, v___x_1301_, v___x_1302_);
v___x_1304_ = lean_apply_1(v_toFun_1298_, v___x_1303_);
return v___x_1304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg(lean_object* v_e_1305_, lean_object* v_inst_1306_){
_start:
{
lean_object* v_toLattice_1307_; lean_object* v_toOrderBot_1308_; lean_object* v_toSDiff_1309_; lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1382_; 
v_toLattice_1307_ = lean_ctor_get(v_inst_1306_, 0);
v_toOrderBot_1308_ = lean_ctor_get(v_inst_1306_, 1);
v_toSDiff_1309_ = lean_ctor_get(v_inst_1306_, 2);
v_isSharedCheck_1382_ = !lean_is_exclusive(v_inst_1306_);
if (v_isSharedCheck_1382_ == 0)
{
v___x_1311_ = v_inst_1306_;
v_isShared_1312_ = v_isSharedCheck_1382_;
goto v_resetjp_1310_;
}
else
{
lean_inc(v_toSDiff_1309_);
lean_inc(v_toOrderBot_1308_);
lean_inc(v_toLattice_1307_);
lean_dec(v_inst_1306_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1382_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
lean_object* v_toSemilatticeSup_1313_; lean_object* v_inf_1314_; lean_object* v___x_1316_; uint8_t v_isShared_1317_; uint8_t v_isSharedCheck_1381_; 
v_toSemilatticeSup_1313_ = lean_ctor_get(v_toLattice_1307_, 0);
v_inf_1314_ = lean_ctor_get(v_toLattice_1307_, 1);
v_isSharedCheck_1381_ = !lean_is_exclusive(v_toLattice_1307_);
if (v_isSharedCheck_1381_ == 0)
{
v___x_1316_ = v_toLattice_1307_;
v_isShared_1317_ = v_isSharedCheck_1381_;
goto v_resetjp_1315_;
}
else
{
lean_inc(v_inf_1314_);
lean_inc(v_toSemilatticeSup_1313_);
lean_dec(v_toLattice_1307_);
v___x_1316_ = lean_box(0);
v_isShared_1317_ = v_isSharedCheck_1381_;
goto v_resetjp_1315_;
}
v_resetjp_1315_:
{
lean_object* v___f_1318_; lean_object* v_min_1319_; lean_object* v_le_1320_; lean_object* v_lt_1321_; lean_object* v_semilatticeInf_1322_; lean_object* v_toPartialOrder_1323_; lean_object* v___x_1325_; uint8_t v_isShared_1326_; uint8_t v_isSharedCheck_1379_; 
v___f_1318_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc_ref(v_e_1305_);
v_min_1319_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1319_, 0, v_e_1305_);
lean_closure_set(v_min_1319_, 1, v___f_1318_);
lean_closure_set(v_min_1319_, 2, v_inf_1314_);
v_le_1320_ = lean_box(0);
v_lt_1321_ = lean_box(0);
lean_inc_ref(v_min_1319_);
v_semilatticeInf_1322_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1319_, v_le_1320_, v_lt_1321_);
v_toPartialOrder_1323_ = lean_ctor_get(v_semilatticeInf_1322_, 0);
v_isSharedCheck_1379_ = !lean_is_exclusive(v_semilatticeInf_1322_);
if (v_isSharedCheck_1379_ == 0)
{
lean_object* v_unused_1380_; 
v_unused_1380_ = lean_ctor_get(v_semilatticeInf_1322_, 1);
lean_dec(v_unused_1380_);
v___x_1325_ = v_semilatticeInf_1322_;
v_isShared_1326_ = v_isSharedCheck_1379_;
goto v_resetjp_1324_;
}
else
{
lean_inc(v_toPartialOrder_1323_);
lean_dec(v_semilatticeInf_1322_);
v___x_1325_ = lean_box(0);
v_isShared_1326_ = v_isSharedCheck_1379_;
goto v_resetjp_1324_;
}
v_resetjp_1324_:
{
lean_object* v_toLE_1327_; lean_object* v_toLT_1328_; lean_object* v___x_1330_; uint8_t v_isShared_1331_; uint8_t v_isSharedCheck_1378_; 
v_toLE_1327_ = lean_ctor_get(v_toPartialOrder_1323_, 0);
v_toLT_1328_ = lean_ctor_get(v_toPartialOrder_1323_, 1);
v_isSharedCheck_1378_ = !lean_is_exclusive(v_toPartialOrder_1323_);
if (v_isSharedCheck_1378_ == 0)
{
v___x_1330_ = v_toPartialOrder_1323_;
v_isShared_1331_ = v_isSharedCheck_1378_;
goto v_resetjp_1329_;
}
else
{
lean_inc(v_toLT_1328_);
lean_inc(v_toLE_1327_);
lean_dec(v_toPartialOrder_1323_);
v___x_1330_ = lean_box(0);
v_isShared_1331_ = v_isSharedCheck_1378_;
goto v_resetjp_1329_;
}
v_resetjp_1329_:
{
lean_object* v___f_1332_; lean_object* v___f_1333_; lean_object* v___x_1335_; 
v___f_1332_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1332_, 0, v_min_1319_);
lean_inc_ref(v_e_1305_);
v___f_1333_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1333_, 0, v_toSemilatticeSup_1313_);
lean_closure_set(v___f_1333_, 1, v_e_1305_);
lean_closure_set(v___f_1333_, 2, v___f_1318_);
if (v_isShared_1331_ == 0)
{
v___x_1335_ = v___x_1330_;
goto v_reusejp_1334_;
}
else
{
lean_object* v_reuseFailAlloc_1377_; 
v_reuseFailAlloc_1377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1377_, 0, v_toLE_1327_);
lean_ctor_set(v_reuseFailAlloc_1377_, 1, v_toLT_1328_);
v___x_1335_ = v_reuseFailAlloc_1377_;
goto v_reusejp_1334_;
}
v_reusejp_1334_:
{
lean_object* v___x_1337_; 
lean_inc_ref(v___f_1333_);
if (v_isShared_1326_ == 0)
{
lean_ctor_set(v___x_1325_, 1, v___f_1333_);
lean_ctor_set(v___x_1325_, 0, v___x_1335_);
v___x_1337_ = v___x_1325_;
goto v_reusejp_1336_;
}
else
{
lean_object* v_reuseFailAlloc_1376_; 
v_reuseFailAlloc_1376_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1376_, 0, v___x_1335_);
lean_ctor_set(v_reuseFailAlloc_1376_, 1, v___f_1333_);
v___x_1337_ = v_reuseFailAlloc_1376_;
goto v_reusejp_1336_;
}
v_reusejp_1336_:
{
lean_object* v_lattice_1339_; 
lean_inc_ref(v___f_1332_);
if (v_isShared_1317_ == 0)
{
lean_ctor_set(v___x_1316_, 1, v___f_1332_);
lean_ctor_set(v___x_1316_, 0, v___x_1337_);
v_lattice_1339_ = v___x_1316_;
goto v_reusejp_1338_;
}
else
{
lean_object* v_reuseFailAlloc_1375_; 
v_reuseFailAlloc_1375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1375_, 0, v___x_1337_);
lean_ctor_set(v_reuseFailAlloc_1375_, 1, v___f_1332_);
v_lattice_1339_ = v_reuseFailAlloc_1375_;
goto v_reusejp_1338_;
}
v_reusejp_1338_:
{
lean_object* v___x_1340_; lean_object* v_toFun_1341_; lean_object* v___x_1343_; uint8_t v_isShared_1344_; uint8_t v_isSharedCheck_1373_; 
lean_inc_ref(v_e_1305_);
v___x_1340_ = lp_mathlib_Equiv_symm___redArg(v_e_1305_);
v_toFun_1341_ = lean_ctor_get(v___x_1340_, 0);
v_isSharedCheck_1373_ = !lean_is_exclusive(v___x_1340_);
if (v_isSharedCheck_1373_ == 0)
{
lean_object* v_unused_1374_; 
v_unused_1374_ = lean_ctor_get(v___x_1340_, 1);
lean_dec(v_unused_1374_);
v___x_1343_ = v___x_1340_;
v_isShared_1344_ = v_isSharedCheck_1373_;
goto v_resetjp_1342_;
}
else
{
lean_inc(v_toFun_1341_);
lean_dec(v___x_1340_);
v___x_1343_ = lean_box(0);
v_isShared_1344_ = v_isSharedCheck_1373_;
goto v_resetjp_1342_;
}
v_resetjp_1342_:
{
lean_object* v___x_1345_; lean_object* v_toPartialOrder_1346_; lean_object* v___x_1348_; uint8_t v_isShared_1349_; uint8_t v_isSharedCheck_1371_; 
v___x_1345_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1339_);
v_toPartialOrder_1346_ = lean_ctor_get(v___x_1345_, 0);
v_isSharedCheck_1371_ = !lean_is_exclusive(v___x_1345_);
if (v_isSharedCheck_1371_ == 0)
{
lean_object* v_unused_1372_; 
v_unused_1372_ = lean_ctor_get(v___x_1345_, 1);
lean_dec(v_unused_1372_);
v___x_1348_ = v___x_1345_;
v_isShared_1349_ = v_isSharedCheck_1371_;
goto v_resetjp_1347_;
}
else
{
lean_inc(v_toPartialOrder_1346_);
lean_dec(v___x_1345_);
v___x_1348_ = lean_box(0);
v_isShared_1349_ = v_isSharedCheck_1371_;
goto v_resetjp_1347_;
}
v_resetjp_1347_:
{
lean_object* v_toLE_1350_; lean_object* v_toLT_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1370_; 
v_toLE_1350_ = lean_ctor_get(v_toPartialOrder_1346_, 0);
v_toLT_1351_ = lean_ctor_get(v_toPartialOrder_1346_, 1);
v_isSharedCheck_1370_ = !lean_is_exclusive(v_toPartialOrder_1346_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1353_ = v_toPartialOrder_1346_;
v_isShared_1354_ = v_isSharedCheck_1370_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_toLT_1351_);
lean_inc(v_toLE_1350_);
lean_dec(v_toPartialOrder_1346_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1370_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v___f_1355_; lean_object* v_sdiff_1356_; lean_object* v_bot_1357_; lean_object* v___x_1359_; 
v___f_1355_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1355_, 0, v___f_1333_);
lean_inc(v_toFun_1341_);
v_sdiff_1356_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg___lam__8), 6, 4);
lean_closure_set(v_sdiff_1356_, 0, v___f_1318_);
lean_closure_set(v_sdiff_1356_, 1, v_e_1305_);
lean_closure_set(v_sdiff_1356_, 2, v_toSDiff_1309_);
lean_closure_set(v_sdiff_1356_, 3, v_toFun_1341_);
v_bot_1357_ = lean_apply_1(v_toFun_1341_, v_toOrderBot_1308_);
if (v_isShared_1354_ == 0)
{
v___x_1359_ = v___x_1353_;
goto v_reusejp_1358_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_toLE_1350_);
lean_ctor_set(v_reuseFailAlloc_1369_, 1, v_toLT_1351_);
v___x_1359_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1358_;
}
v_reusejp_1358_:
{
lean_object* v___x_1361_; 
if (v_isShared_1349_ == 0)
{
lean_ctor_set(v___x_1348_, 1, v___f_1355_);
lean_ctor_set(v___x_1348_, 0, v___x_1359_);
v___x_1361_ = v___x_1348_;
goto v_reusejp_1360_;
}
else
{
lean_object* v_reuseFailAlloc_1368_; 
v_reuseFailAlloc_1368_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1368_, 0, v___x_1359_);
lean_ctor_set(v_reuseFailAlloc_1368_, 1, v___f_1355_);
v___x_1361_ = v_reuseFailAlloc_1368_;
goto v_reusejp_1360_;
}
v_reusejp_1360_:
{
lean_object* v___x_1363_; 
if (v_isShared_1344_ == 0)
{
lean_ctor_set(v___x_1343_, 1, v___f_1332_);
lean_ctor_set(v___x_1343_, 0, v___x_1361_);
v___x_1363_ = v___x_1343_;
goto v_reusejp_1362_;
}
else
{
lean_object* v_reuseFailAlloc_1367_; 
v_reuseFailAlloc_1367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1367_, 0, v___x_1361_);
lean_ctor_set(v_reuseFailAlloc_1367_, 1, v___f_1332_);
v___x_1363_ = v_reuseFailAlloc_1367_;
goto v_reusejp_1362_;
}
v_reusejp_1362_:
{
lean_object* v___x_1365_; 
if (v_isShared_1312_ == 0)
{
lean_ctor_set(v___x_1311_, 2, v_sdiff_1356_);
lean_ctor_set(v___x_1311_, 1, v_bot_1357_);
lean_ctor_set(v___x_1311_, 0, v___x_1363_);
v___x_1365_ = v___x_1311_;
goto v_reusejp_1364_;
}
else
{
lean_object* v_reuseFailAlloc_1366_; 
v_reuseFailAlloc_1366_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1366_, 0, v___x_1363_);
lean_ctor_set(v_reuseFailAlloc_1366_, 1, v_bot_1357_);
lean_ctor_set(v_reuseFailAlloc_1366_, 2, v_sdiff_1356_);
v___x_1365_ = v_reuseFailAlloc_1366_;
goto v_reusejp_1364_;
}
v_reusejp_1364_:
{
return v___x_1365_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_generalizedCoheytingAlgebra(lean_object* v_00_u03b1_1383_, lean_object* v_00_u03b2_1384_, lean_object* v_e_1385_, lean_object* v_inst_1386_){
_start:
{
lean_object* v_toLattice_1387_; lean_object* v_toOrderBot_1388_; lean_object* v_toSDiff_1389_; lean_object* v___x_1391_; uint8_t v_isShared_1392_; uint8_t v_isSharedCheck_1462_; 
v_toLattice_1387_ = lean_ctor_get(v_inst_1386_, 0);
v_toOrderBot_1388_ = lean_ctor_get(v_inst_1386_, 1);
v_toSDiff_1389_ = lean_ctor_get(v_inst_1386_, 2);
v_isSharedCheck_1462_ = !lean_is_exclusive(v_inst_1386_);
if (v_isSharedCheck_1462_ == 0)
{
v___x_1391_ = v_inst_1386_;
v_isShared_1392_ = v_isSharedCheck_1462_;
goto v_resetjp_1390_;
}
else
{
lean_inc(v_toSDiff_1389_);
lean_inc(v_toOrderBot_1388_);
lean_inc(v_toLattice_1387_);
lean_dec(v_inst_1386_);
v___x_1391_ = lean_box(0);
v_isShared_1392_ = v_isSharedCheck_1462_;
goto v_resetjp_1390_;
}
v_resetjp_1390_:
{
lean_object* v_toSemilatticeSup_1393_; lean_object* v_inf_1394_; lean_object* v___x_1396_; uint8_t v_isShared_1397_; uint8_t v_isSharedCheck_1461_; 
v_toSemilatticeSup_1393_ = lean_ctor_get(v_toLattice_1387_, 0);
v_inf_1394_ = lean_ctor_get(v_toLattice_1387_, 1);
v_isSharedCheck_1461_ = !lean_is_exclusive(v_toLattice_1387_);
if (v_isSharedCheck_1461_ == 0)
{
v___x_1396_ = v_toLattice_1387_;
v_isShared_1397_ = v_isSharedCheck_1461_;
goto v_resetjp_1395_;
}
else
{
lean_inc(v_inf_1394_);
lean_inc(v_toSemilatticeSup_1393_);
lean_dec(v_toLattice_1387_);
v___x_1396_ = lean_box(0);
v_isShared_1397_ = v_isSharedCheck_1461_;
goto v_resetjp_1395_;
}
v_resetjp_1395_:
{
lean_object* v___f_1398_; lean_object* v_min_1399_; lean_object* v_le_1400_; lean_object* v_lt_1401_; lean_object* v_semilatticeInf_1402_; lean_object* v_toPartialOrder_1403_; lean_object* v___x_1405_; uint8_t v_isShared_1406_; uint8_t v_isSharedCheck_1459_; 
v___f_1398_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc_ref(v_e_1385_);
v_min_1399_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1399_, 0, v_e_1385_);
lean_closure_set(v_min_1399_, 1, v___f_1398_);
lean_closure_set(v_min_1399_, 2, v_inf_1394_);
v_le_1400_ = lean_box(0);
v_lt_1401_ = lean_box(0);
lean_inc_ref(v_min_1399_);
v_semilatticeInf_1402_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1399_, v_le_1400_, v_lt_1401_);
v_toPartialOrder_1403_ = lean_ctor_get(v_semilatticeInf_1402_, 0);
v_isSharedCheck_1459_ = !lean_is_exclusive(v_semilatticeInf_1402_);
if (v_isSharedCheck_1459_ == 0)
{
lean_object* v_unused_1460_; 
v_unused_1460_ = lean_ctor_get(v_semilatticeInf_1402_, 1);
lean_dec(v_unused_1460_);
v___x_1405_ = v_semilatticeInf_1402_;
v_isShared_1406_ = v_isSharedCheck_1459_;
goto v_resetjp_1404_;
}
else
{
lean_inc(v_toPartialOrder_1403_);
lean_dec(v_semilatticeInf_1402_);
v___x_1405_ = lean_box(0);
v_isShared_1406_ = v_isSharedCheck_1459_;
goto v_resetjp_1404_;
}
v_resetjp_1404_:
{
lean_object* v_toLE_1407_; lean_object* v_toLT_1408_; lean_object* v___x_1410_; uint8_t v_isShared_1411_; uint8_t v_isSharedCheck_1458_; 
v_toLE_1407_ = lean_ctor_get(v_toPartialOrder_1403_, 0);
v_toLT_1408_ = lean_ctor_get(v_toPartialOrder_1403_, 1);
v_isSharedCheck_1458_ = !lean_is_exclusive(v_toPartialOrder_1403_);
if (v_isSharedCheck_1458_ == 0)
{
v___x_1410_ = v_toPartialOrder_1403_;
v_isShared_1411_ = v_isSharedCheck_1458_;
goto v_resetjp_1409_;
}
else
{
lean_inc(v_toLT_1408_);
lean_inc(v_toLE_1407_);
lean_dec(v_toPartialOrder_1403_);
v___x_1410_ = lean_box(0);
v_isShared_1411_ = v_isSharedCheck_1458_;
goto v_resetjp_1409_;
}
v_resetjp_1409_:
{
lean_object* v___f_1412_; lean_object* v___f_1413_; lean_object* v___x_1415_; 
v___f_1412_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1412_, 0, v_min_1399_);
lean_inc_ref(v_e_1385_);
v___f_1413_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1413_, 0, v_toSemilatticeSup_1393_);
lean_closure_set(v___f_1413_, 1, v_e_1385_);
lean_closure_set(v___f_1413_, 2, v___f_1398_);
if (v_isShared_1411_ == 0)
{
v___x_1415_ = v___x_1410_;
goto v_reusejp_1414_;
}
else
{
lean_object* v_reuseFailAlloc_1457_; 
v_reuseFailAlloc_1457_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1457_, 0, v_toLE_1407_);
lean_ctor_set(v_reuseFailAlloc_1457_, 1, v_toLT_1408_);
v___x_1415_ = v_reuseFailAlloc_1457_;
goto v_reusejp_1414_;
}
v_reusejp_1414_:
{
lean_object* v___x_1417_; 
lean_inc_ref(v___f_1413_);
if (v_isShared_1406_ == 0)
{
lean_ctor_set(v___x_1405_, 1, v___f_1413_);
lean_ctor_set(v___x_1405_, 0, v___x_1415_);
v___x_1417_ = v___x_1405_;
goto v_reusejp_1416_;
}
else
{
lean_object* v_reuseFailAlloc_1456_; 
v_reuseFailAlloc_1456_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1456_, 0, v___x_1415_);
lean_ctor_set(v_reuseFailAlloc_1456_, 1, v___f_1413_);
v___x_1417_ = v_reuseFailAlloc_1456_;
goto v_reusejp_1416_;
}
v_reusejp_1416_:
{
lean_object* v_lattice_1419_; 
lean_inc_ref(v___f_1412_);
if (v_isShared_1397_ == 0)
{
lean_ctor_set(v___x_1396_, 1, v___f_1412_);
lean_ctor_set(v___x_1396_, 0, v___x_1417_);
v_lattice_1419_ = v___x_1396_;
goto v_reusejp_1418_;
}
else
{
lean_object* v_reuseFailAlloc_1455_; 
v_reuseFailAlloc_1455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1455_, 0, v___x_1417_);
lean_ctor_set(v_reuseFailAlloc_1455_, 1, v___f_1412_);
v_lattice_1419_ = v_reuseFailAlloc_1455_;
goto v_reusejp_1418_;
}
v_reusejp_1418_:
{
lean_object* v___x_1420_; lean_object* v_toFun_1421_; lean_object* v___x_1423_; uint8_t v_isShared_1424_; uint8_t v_isSharedCheck_1453_; 
lean_inc_ref(v_e_1385_);
v___x_1420_ = lp_mathlib_Equiv_symm___redArg(v_e_1385_);
v_toFun_1421_ = lean_ctor_get(v___x_1420_, 0);
v_isSharedCheck_1453_ = !lean_is_exclusive(v___x_1420_);
if (v_isSharedCheck_1453_ == 0)
{
lean_object* v_unused_1454_; 
v_unused_1454_ = lean_ctor_get(v___x_1420_, 1);
lean_dec(v_unused_1454_);
v___x_1423_ = v___x_1420_;
v_isShared_1424_ = v_isSharedCheck_1453_;
goto v_resetjp_1422_;
}
else
{
lean_inc(v_toFun_1421_);
lean_dec(v___x_1420_);
v___x_1423_ = lean_box(0);
v_isShared_1424_ = v_isSharedCheck_1453_;
goto v_resetjp_1422_;
}
v_resetjp_1422_:
{
lean_object* v___x_1425_; lean_object* v_toPartialOrder_1426_; lean_object* v___x_1428_; uint8_t v_isShared_1429_; uint8_t v_isSharedCheck_1451_; 
v___x_1425_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1419_);
v_toPartialOrder_1426_ = lean_ctor_get(v___x_1425_, 0);
v_isSharedCheck_1451_ = !lean_is_exclusive(v___x_1425_);
if (v_isSharedCheck_1451_ == 0)
{
lean_object* v_unused_1452_; 
v_unused_1452_ = lean_ctor_get(v___x_1425_, 1);
lean_dec(v_unused_1452_);
v___x_1428_ = v___x_1425_;
v_isShared_1429_ = v_isSharedCheck_1451_;
goto v_resetjp_1427_;
}
else
{
lean_inc(v_toPartialOrder_1426_);
lean_dec(v___x_1425_);
v___x_1428_ = lean_box(0);
v_isShared_1429_ = v_isSharedCheck_1451_;
goto v_resetjp_1427_;
}
v_resetjp_1427_:
{
lean_object* v_toLE_1430_; lean_object* v_toLT_1431_; lean_object* v___x_1433_; uint8_t v_isShared_1434_; uint8_t v_isSharedCheck_1450_; 
v_toLE_1430_ = lean_ctor_get(v_toPartialOrder_1426_, 0);
v_toLT_1431_ = lean_ctor_get(v_toPartialOrder_1426_, 1);
v_isSharedCheck_1450_ = !lean_is_exclusive(v_toPartialOrder_1426_);
if (v_isSharedCheck_1450_ == 0)
{
v___x_1433_ = v_toPartialOrder_1426_;
v_isShared_1434_ = v_isSharedCheck_1450_;
goto v_resetjp_1432_;
}
else
{
lean_inc(v_toLT_1431_);
lean_inc(v_toLE_1430_);
lean_dec(v_toPartialOrder_1426_);
v___x_1433_ = lean_box(0);
v_isShared_1434_ = v_isSharedCheck_1450_;
goto v_resetjp_1432_;
}
v_resetjp_1432_:
{
lean_object* v___f_1435_; lean_object* v_sdiff_1436_; lean_object* v_bot_1437_; lean_object* v___x_1439_; 
v___f_1435_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1435_, 0, v___f_1413_);
lean_inc(v_toFun_1421_);
v_sdiff_1436_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg___lam__8), 6, 4);
lean_closure_set(v_sdiff_1436_, 0, v___f_1398_);
lean_closure_set(v_sdiff_1436_, 1, v_e_1385_);
lean_closure_set(v_sdiff_1436_, 2, v_toSDiff_1389_);
lean_closure_set(v_sdiff_1436_, 3, v_toFun_1421_);
v_bot_1437_ = lean_apply_1(v_toFun_1421_, v_toOrderBot_1388_);
if (v_isShared_1434_ == 0)
{
v___x_1439_ = v___x_1433_;
goto v_reusejp_1438_;
}
else
{
lean_object* v_reuseFailAlloc_1449_; 
v_reuseFailAlloc_1449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1449_, 0, v_toLE_1430_);
lean_ctor_set(v_reuseFailAlloc_1449_, 1, v_toLT_1431_);
v___x_1439_ = v_reuseFailAlloc_1449_;
goto v_reusejp_1438_;
}
v_reusejp_1438_:
{
lean_object* v___x_1441_; 
if (v_isShared_1429_ == 0)
{
lean_ctor_set(v___x_1428_, 1, v___f_1435_);
lean_ctor_set(v___x_1428_, 0, v___x_1439_);
v___x_1441_ = v___x_1428_;
goto v_reusejp_1440_;
}
else
{
lean_object* v_reuseFailAlloc_1448_; 
v_reuseFailAlloc_1448_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1448_, 0, v___x_1439_);
lean_ctor_set(v_reuseFailAlloc_1448_, 1, v___f_1435_);
v___x_1441_ = v_reuseFailAlloc_1448_;
goto v_reusejp_1440_;
}
v_reusejp_1440_:
{
lean_object* v___x_1443_; 
if (v_isShared_1424_ == 0)
{
lean_ctor_set(v___x_1423_, 1, v___f_1412_);
lean_ctor_set(v___x_1423_, 0, v___x_1441_);
v___x_1443_ = v___x_1423_;
goto v_reusejp_1442_;
}
else
{
lean_object* v_reuseFailAlloc_1447_; 
v_reuseFailAlloc_1447_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1447_, 0, v___x_1441_);
lean_ctor_set(v_reuseFailAlloc_1447_, 1, v___f_1412_);
v___x_1443_ = v_reuseFailAlloc_1447_;
goto v_reusejp_1442_;
}
v_reusejp_1442_:
{
lean_object* v___x_1445_; 
if (v_isShared_1392_ == 0)
{
lean_ctor_set(v___x_1391_, 2, v_sdiff_1436_);
lean_ctor_set(v___x_1391_, 1, v_bot_1437_);
lean_ctor_set(v___x_1391_, 0, v___x_1443_);
v___x_1445_ = v___x_1391_;
goto v_reusejp_1444_;
}
else
{
lean_object* v_reuseFailAlloc_1446_; 
v_reuseFailAlloc_1446_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1446_, 0, v___x_1443_);
lean_ctor_set(v_reuseFailAlloc_1446_, 1, v_bot_1437_);
lean_ctor_set(v_reuseFailAlloc_1446_, 2, v_sdiff_1436_);
v___x_1445_ = v_reuseFailAlloc_1446_;
goto v_reusejp_1444_;
}
v_reusejp_1444_:
{
return v___x_1445_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_heytingAlgebra___redArg___lam__10(lean_object* v_e_1463_, lean_object* v_toCompl_1464_, lean_object* v_toFun_1465_, lean_object* v_a_1466_){
_start:
{
lean_object* v_toFun_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; 
v_toFun_1467_ = lean_ctor_get(v_e_1463_, 0);
lean_inc(v_toFun_1467_);
lean_dec_ref(v_e_1463_);
v___x_1468_ = lean_apply_1(v_toFun_1467_, v_a_1466_);
v___x_1469_ = lean_apply_1(v_toCompl_1464_, v___x_1468_);
v___x_1470_ = lean_apply_1(v_toFun_1465_, v___x_1469_);
return v___x_1470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_heytingAlgebra___redArg(lean_object* v_e_1471_, lean_object* v_inst_1472_){
_start:
{
lean_object* v_toGeneralizedHeytingAlgebra_1473_; lean_object* v_toLattice_1474_; lean_object* v_toOrderBot_1475_; lean_object* v_toCompl_1476_; lean_object* v___x_1478_; uint8_t v_isShared_1479_; uint8_t v_isSharedCheck_1581_; 
v_toGeneralizedHeytingAlgebra_1473_ = lean_ctor_get(v_inst_1472_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_1473_);
v_toLattice_1474_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1473_, 0);
lean_inc_ref(v_toLattice_1474_);
v_toOrderBot_1475_ = lean_ctor_get(v_inst_1472_, 1);
v_toCompl_1476_ = lean_ctor_get(v_inst_1472_, 2);
v_isSharedCheck_1581_ = !lean_is_exclusive(v_inst_1472_);
if (v_isSharedCheck_1581_ == 0)
{
lean_object* v_unused_1582_; 
v_unused_1582_ = lean_ctor_get(v_inst_1472_, 0);
lean_dec(v_unused_1582_);
v___x_1478_ = v_inst_1472_;
v_isShared_1479_ = v_isSharedCheck_1581_;
goto v_resetjp_1477_;
}
else
{
lean_inc(v_toCompl_1476_);
lean_inc(v_toOrderBot_1475_);
lean_dec(v_inst_1472_);
v___x_1478_ = lean_box(0);
v_isShared_1479_ = v_isSharedCheck_1581_;
goto v_resetjp_1477_;
}
v_resetjp_1477_:
{
lean_object* v_toOrderTop_1480_; lean_object* v_toHImp_1481_; lean_object* v___x_1483_; uint8_t v_isShared_1484_; uint8_t v_isSharedCheck_1579_; 
v_toOrderTop_1480_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1473_, 1);
v_toHImp_1481_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1473_, 2);
v_isSharedCheck_1579_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_1473_);
if (v_isSharedCheck_1579_ == 0)
{
lean_object* v_unused_1580_; 
v_unused_1580_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1473_, 0);
lean_dec(v_unused_1580_);
v___x_1483_ = v_toGeneralizedHeytingAlgebra_1473_;
v_isShared_1484_ = v_isSharedCheck_1579_;
goto v_resetjp_1482_;
}
else
{
lean_inc(v_toHImp_1481_);
lean_inc(v_toOrderTop_1480_);
lean_dec(v_toGeneralizedHeytingAlgebra_1473_);
v___x_1483_ = lean_box(0);
v_isShared_1484_ = v_isSharedCheck_1579_;
goto v_resetjp_1482_;
}
v_resetjp_1482_:
{
lean_object* v_toSemilatticeSup_1485_; lean_object* v_inf_1486_; lean_object* v___x_1488_; uint8_t v_isShared_1489_; uint8_t v_isSharedCheck_1578_; 
v_toSemilatticeSup_1485_ = lean_ctor_get(v_toLattice_1474_, 0);
v_inf_1486_ = lean_ctor_get(v_toLattice_1474_, 1);
v_isSharedCheck_1578_ = !lean_is_exclusive(v_toLattice_1474_);
if (v_isSharedCheck_1578_ == 0)
{
v___x_1488_ = v_toLattice_1474_;
v_isShared_1489_ = v_isSharedCheck_1578_;
goto v_resetjp_1487_;
}
else
{
lean_inc(v_inf_1486_);
lean_inc(v_toSemilatticeSup_1485_);
lean_dec(v_toLattice_1474_);
v___x_1488_ = lean_box(0);
v_isShared_1489_ = v_isSharedCheck_1578_;
goto v_resetjp_1487_;
}
v_resetjp_1487_:
{
lean_object* v___f_1490_; lean_object* v_min_1491_; lean_object* v_le_1492_; lean_object* v_lt_1493_; lean_object* v_semilatticeInf_1494_; lean_object* v_toPartialOrder_1495_; lean_object* v___x_1497_; uint8_t v_isShared_1498_; uint8_t v_isSharedCheck_1576_; 
v___f_1490_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc_ref(v_e_1471_);
v_min_1491_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1491_, 0, v_e_1471_);
lean_closure_set(v_min_1491_, 1, v___f_1490_);
lean_closure_set(v_min_1491_, 2, v_inf_1486_);
v_le_1492_ = lean_box(0);
v_lt_1493_ = lean_box(0);
lean_inc_ref(v_min_1491_);
v_semilatticeInf_1494_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1491_, v_le_1492_, v_lt_1493_);
v_toPartialOrder_1495_ = lean_ctor_get(v_semilatticeInf_1494_, 0);
v_isSharedCheck_1576_ = !lean_is_exclusive(v_semilatticeInf_1494_);
if (v_isSharedCheck_1576_ == 0)
{
lean_object* v_unused_1577_; 
v_unused_1577_ = lean_ctor_get(v_semilatticeInf_1494_, 1);
lean_dec(v_unused_1577_);
v___x_1497_ = v_semilatticeInf_1494_;
v_isShared_1498_ = v_isSharedCheck_1576_;
goto v_resetjp_1496_;
}
else
{
lean_inc(v_toPartialOrder_1495_);
lean_dec(v_semilatticeInf_1494_);
v___x_1497_ = lean_box(0);
v_isShared_1498_ = v_isSharedCheck_1576_;
goto v_resetjp_1496_;
}
v_resetjp_1496_:
{
lean_object* v_toLE_1499_; lean_object* v_toLT_1500_; lean_object* v___x_1502_; uint8_t v_isShared_1503_; uint8_t v_isSharedCheck_1575_; 
v_toLE_1499_ = lean_ctor_get(v_toPartialOrder_1495_, 0);
v_toLT_1500_ = lean_ctor_get(v_toPartialOrder_1495_, 1);
v_isSharedCheck_1575_ = !lean_is_exclusive(v_toPartialOrder_1495_);
if (v_isSharedCheck_1575_ == 0)
{
v___x_1502_ = v_toPartialOrder_1495_;
v_isShared_1503_ = v_isSharedCheck_1575_;
goto v_resetjp_1501_;
}
else
{
lean_inc(v_toLT_1500_);
lean_inc(v_toLE_1499_);
lean_dec(v_toPartialOrder_1495_);
v___x_1502_ = lean_box(0);
v_isShared_1503_ = v_isSharedCheck_1575_;
goto v_resetjp_1501_;
}
v_resetjp_1501_:
{
lean_object* v___f_1504_; lean_object* v___f_1505_; lean_object* v___x_1507_; 
v___f_1504_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1504_, 0, v_min_1491_);
lean_inc_ref(v_e_1471_);
v___f_1505_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1505_, 0, v_toSemilatticeSup_1485_);
lean_closure_set(v___f_1505_, 1, v_e_1471_);
lean_closure_set(v___f_1505_, 2, v___f_1490_);
if (v_isShared_1503_ == 0)
{
v___x_1507_ = v___x_1502_;
goto v_reusejp_1506_;
}
else
{
lean_object* v_reuseFailAlloc_1574_; 
v_reuseFailAlloc_1574_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1574_, 0, v_toLE_1499_);
lean_ctor_set(v_reuseFailAlloc_1574_, 1, v_toLT_1500_);
v___x_1507_ = v_reuseFailAlloc_1574_;
goto v_reusejp_1506_;
}
v_reusejp_1506_:
{
lean_object* v___x_1509_; 
lean_inc_ref(v___f_1505_);
if (v_isShared_1498_ == 0)
{
lean_ctor_set(v___x_1497_, 1, v___f_1505_);
lean_ctor_set(v___x_1497_, 0, v___x_1507_);
v___x_1509_ = v___x_1497_;
goto v_reusejp_1508_;
}
else
{
lean_object* v_reuseFailAlloc_1573_; 
v_reuseFailAlloc_1573_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1573_, 0, v___x_1507_);
lean_ctor_set(v_reuseFailAlloc_1573_, 1, v___f_1505_);
v___x_1509_ = v_reuseFailAlloc_1573_;
goto v_reusejp_1508_;
}
v_reusejp_1508_:
{
lean_object* v_lattice_1511_; 
lean_inc_ref(v___f_1504_);
if (v_isShared_1489_ == 0)
{
lean_ctor_set(v___x_1488_, 1, v___f_1504_);
lean_ctor_set(v___x_1488_, 0, v___x_1509_);
v_lattice_1511_ = v___x_1488_;
goto v_reusejp_1510_;
}
else
{
lean_object* v_reuseFailAlloc_1572_; 
v_reuseFailAlloc_1572_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1572_, 0, v___x_1509_);
lean_ctor_set(v_reuseFailAlloc_1572_, 1, v___f_1504_);
v_lattice_1511_ = v_reuseFailAlloc_1572_;
goto v_reusejp_1510_;
}
v_reusejp_1510_:
{
lean_object* v___x_1512_; lean_object* v_toFun_1513_; lean_object* v___x_1515_; uint8_t v_isShared_1516_; uint8_t v_isSharedCheck_1570_; 
lean_inc_ref(v_e_1471_);
v___x_1512_ = lp_mathlib_Equiv_symm___redArg(v_e_1471_);
v_toFun_1513_ = lean_ctor_get(v___x_1512_, 0);
v_isSharedCheck_1570_ = !lean_is_exclusive(v___x_1512_);
if (v_isSharedCheck_1570_ == 0)
{
lean_object* v_unused_1571_; 
v_unused_1571_ = lean_ctor_get(v___x_1512_, 1);
lean_dec(v_unused_1571_);
v___x_1515_ = v___x_1512_;
v_isShared_1516_ = v_isSharedCheck_1570_;
goto v_resetjp_1514_;
}
else
{
lean_inc(v_toFun_1513_);
lean_dec(v___x_1512_);
v___x_1515_ = lean_box(0);
v_isShared_1516_ = v_isSharedCheck_1570_;
goto v_resetjp_1514_;
}
v_resetjp_1514_:
{
lean_object* v___x_1517_; lean_object* v_toPartialOrder_1518_; lean_object* v___x_1520_; uint8_t v_isShared_1521_; uint8_t v_isSharedCheck_1568_; 
v___x_1517_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1511_);
v_toPartialOrder_1518_ = lean_ctor_get(v___x_1517_, 0);
v_isSharedCheck_1568_ = !lean_is_exclusive(v___x_1517_);
if (v_isSharedCheck_1568_ == 0)
{
lean_object* v_unused_1569_; 
v_unused_1569_ = lean_ctor_get(v___x_1517_, 1);
lean_dec(v_unused_1569_);
v___x_1520_ = v___x_1517_;
v_isShared_1521_ = v_isSharedCheck_1568_;
goto v_resetjp_1519_;
}
else
{
lean_inc(v_toPartialOrder_1518_);
lean_dec(v___x_1517_);
v___x_1520_ = lean_box(0);
v_isShared_1521_ = v_isSharedCheck_1568_;
goto v_resetjp_1519_;
}
v_resetjp_1519_:
{
lean_object* v_toLE_1522_; lean_object* v_toLT_1523_; lean_object* v___x_1525_; uint8_t v_isShared_1526_; uint8_t v_isSharedCheck_1567_; 
v_toLE_1522_ = lean_ctor_get(v_toPartialOrder_1518_, 0);
v_toLT_1523_ = lean_ctor_get(v_toPartialOrder_1518_, 1);
v_isSharedCheck_1567_ = !lean_is_exclusive(v_toPartialOrder_1518_);
if (v_isSharedCheck_1567_ == 0)
{
v___x_1525_ = v_toPartialOrder_1518_;
v_isShared_1526_ = v_isSharedCheck_1567_;
goto v_resetjp_1524_;
}
else
{
lean_inc(v_toLT_1523_);
lean_inc(v_toLE_1522_);
lean_dec(v_toPartialOrder_1518_);
v___x_1525_ = lean_box(0);
v_isShared_1526_ = v_isSharedCheck_1567_;
goto v_resetjp_1524_;
}
v_resetjp_1524_:
{
lean_object* v___f_1527_; lean_object* v___x_1529_; 
v___f_1527_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1527_, 0, v___f_1505_);
if (v_isShared_1526_ == 0)
{
v___x_1529_ = v___x_1525_;
goto v_reusejp_1528_;
}
else
{
lean_object* v_reuseFailAlloc_1566_; 
v_reuseFailAlloc_1566_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1566_, 0, v_toLE_1522_);
lean_ctor_set(v_reuseFailAlloc_1566_, 1, v_toLT_1523_);
v___x_1529_ = v_reuseFailAlloc_1566_;
goto v_reusejp_1528_;
}
v_reusejp_1528_:
{
lean_object* v___x_1531_; 
lean_inc_ref(v___f_1527_);
if (v_isShared_1521_ == 0)
{
lean_ctor_set(v___x_1520_, 1, v___f_1527_);
lean_ctor_set(v___x_1520_, 0, v___x_1529_);
v___x_1531_ = v___x_1520_;
goto v_reusejp_1530_;
}
else
{
lean_object* v_reuseFailAlloc_1565_; 
v_reuseFailAlloc_1565_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1565_, 0, v___x_1529_);
lean_ctor_set(v_reuseFailAlloc_1565_, 1, v___f_1527_);
v___x_1531_ = v_reuseFailAlloc_1565_;
goto v_reusejp_1530_;
}
v_reusejp_1530_:
{
lean_object* v___x_1533_; 
lean_inc_ref(v___f_1504_);
if (v_isShared_1516_ == 0)
{
lean_ctor_set(v___x_1515_, 1, v___f_1504_);
lean_ctor_set(v___x_1515_, 0, v___x_1531_);
v___x_1533_ = v___x_1515_;
goto v_reusejp_1532_;
}
else
{
lean_object* v_reuseFailAlloc_1564_; 
v_reuseFailAlloc_1564_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1564_, 0, v___x_1531_);
lean_ctor_set(v_reuseFailAlloc_1564_, 1, v___f_1504_);
v___x_1533_ = v_reuseFailAlloc_1564_;
goto v_reusejp_1532_;
}
v_reusejp_1532_:
{
lean_object* v___x_1534_; lean_object* v_toPartialOrder_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1562_; 
v___x_1534_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_1533_);
v_toPartialOrder_1535_ = lean_ctor_get(v___x_1534_, 0);
v_isSharedCheck_1562_ = !lean_is_exclusive(v___x_1534_);
if (v_isSharedCheck_1562_ == 0)
{
lean_object* v_unused_1563_; 
v_unused_1563_ = lean_ctor_get(v___x_1534_, 1);
lean_dec(v_unused_1563_);
v___x_1537_ = v___x_1534_;
v_isShared_1538_ = v_isSharedCheck_1562_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_toPartialOrder_1535_);
lean_dec(v___x_1534_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1562_;
goto v_resetjp_1536_;
}
v_resetjp_1536_:
{
lean_object* v_toLE_1539_; lean_object* v_toLT_1540_; lean_object* v___x_1542_; uint8_t v_isShared_1543_; uint8_t v_isSharedCheck_1561_; 
v_toLE_1539_ = lean_ctor_get(v_toPartialOrder_1535_, 0);
v_toLT_1540_ = lean_ctor_get(v_toPartialOrder_1535_, 1);
v_isSharedCheck_1561_ = !lean_is_exclusive(v_toPartialOrder_1535_);
if (v_isSharedCheck_1561_ == 0)
{
v___x_1542_ = v_toPartialOrder_1535_;
v_isShared_1543_ = v_isSharedCheck_1561_;
goto v_resetjp_1541_;
}
else
{
lean_inc(v_toLT_1540_);
lean_inc(v_toLE_1539_);
lean_dec(v_toPartialOrder_1535_);
v___x_1542_ = lean_box(0);
v_isShared_1543_ = v_isSharedCheck_1561_;
goto v_resetjp_1541_;
}
v_resetjp_1541_:
{
lean_object* v_top_1544_; lean_object* v_compl_1545_; lean_object* v_himp_1546_; lean_object* v_bot_1547_; lean_object* v___x_1549_; 
lean_inc_n(v_toFun_1513_, 3);
v_top_1544_ = lean_apply_1(v_toFun_1513_, v_toOrderTop_1480_);
lean_inc_ref(v_e_1471_);
v_compl_1545_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_heytingAlgebra___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_1545_, 0, v_e_1471_);
lean_closure_set(v_compl_1545_, 1, v_toCompl_1476_);
lean_closure_set(v_compl_1545_, 2, v_toFun_1513_);
v_himp_1546_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__3), 6, 4);
lean_closure_set(v_himp_1546_, 0, v___f_1490_);
lean_closure_set(v_himp_1546_, 1, v_e_1471_);
lean_closure_set(v_himp_1546_, 2, v_toHImp_1481_);
lean_closure_set(v_himp_1546_, 3, v_toFun_1513_);
v_bot_1547_ = lean_apply_1(v_toFun_1513_, v_toOrderBot_1475_);
if (v_isShared_1543_ == 0)
{
v___x_1549_ = v___x_1542_;
goto v_reusejp_1548_;
}
else
{
lean_object* v_reuseFailAlloc_1560_; 
v_reuseFailAlloc_1560_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1560_, 0, v_toLE_1539_);
lean_ctor_set(v_reuseFailAlloc_1560_, 1, v_toLT_1540_);
v___x_1549_ = v_reuseFailAlloc_1560_;
goto v_reusejp_1548_;
}
v_reusejp_1548_:
{
lean_object* v___x_1551_; 
if (v_isShared_1538_ == 0)
{
lean_ctor_set(v___x_1537_, 1, v___f_1527_);
lean_ctor_set(v___x_1537_, 0, v___x_1549_);
v___x_1551_ = v___x_1537_;
goto v_reusejp_1550_;
}
else
{
lean_object* v_reuseFailAlloc_1559_; 
v_reuseFailAlloc_1559_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1559_, 0, v___x_1549_);
lean_ctor_set(v_reuseFailAlloc_1559_, 1, v___f_1527_);
v___x_1551_ = v_reuseFailAlloc_1559_;
goto v_reusejp_1550_;
}
v_reusejp_1550_:
{
lean_object* v___x_1552_; lean_object* v___x_1554_; 
v___x_1552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1552_, 0, v___x_1551_);
lean_ctor_set(v___x_1552_, 1, v___f_1504_);
if (v_isShared_1484_ == 0)
{
lean_ctor_set(v___x_1483_, 2, v_himp_1546_);
lean_ctor_set(v___x_1483_, 1, v_top_1544_);
lean_ctor_set(v___x_1483_, 0, v___x_1552_);
v___x_1554_ = v___x_1483_;
goto v_reusejp_1553_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v___x_1552_);
lean_ctor_set(v_reuseFailAlloc_1558_, 1, v_top_1544_);
lean_ctor_set(v_reuseFailAlloc_1558_, 2, v_himp_1546_);
v___x_1554_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1553_;
}
v_reusejp_1553_:
{
lean_object* v___x_1556_; 
if (v_isShared_1479_ == 0)
{
lean_ctor_set(v___x_1478_, 2, v_compl_1545_);
lean_ctor_set(v___x_1478_, 1, v_bot_1547_);
lean_ctor_set(v___x_1478_, 0, v___x_1554_);
v___x_1556_ = v___x_1478_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1557_; 
v_reuseFailAlloc_1557_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1557_, 0, v___x_1554_);
lean_ctor_set(v_reuseFailAlloc_1557_, 1, v_bot_1547_);
lean_ctor_set(v_reuseFailAlloc_1557_, 2, v_compl_1545_);
v___x_1556_ = v_reuseFailAlloc_1557_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
return v___x_1556_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_heytingAlgebra(lean_object* v_00_u03b1_1583_, lean_object* v_00_u03b2_1584_, lean_object* v_e_1585_, lean_object* v_inst_1586_){
_start:
{
lean_object* v_toGeneralizedHeytingAlgebra_1587_; lean_object* v_toLattice_1588_; lean_object* v_toOrderBot_1589_; lean_object* v_toCompl_1590_; lean_object* v___x_1592_; uint8_t v_isShared_1593_; uint8_t v_isSharedCheck_1695_; 
v_toGeneralizedHeytingAlgebra_1587_ = lean_ctor_get(v_inst_1586_, 0);
lean_inc_ref(v_toGeneralizedHeytingAlgebra_1587_);
v_toLattice_1588_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1587_, 0);
lean_inc_ref(v_toLattice_1588_);
v_toOrderBot_1589_ = lean_ctor_get(v_inst_1586_, 1);
v_toCompl_1590_ = lean_ctor_get(v_inst_1586_, 2);
v_isSharedCheck_1695_ = !lean_is_exclusive(v_inst_1586_);
if (v_isSharedCheck_1695_ == 0)
{
lean_object* v_unused_1696_; 
v_unused_1696_ = lean_ctor_get(v_inst_1586_, 0);
lean_dec(v_unused_1696_);
v___x_1592_ = v_inst_1586_;
v_isShared_1593_ = v_isSharedCheck_1695_;
goto v_resetjp_1591_;
}
else
{
lean_inc(v_toCompl_1590_);
lean_inc(v_toOrderBot_1589_);
lean_dec(v_inst_1586_);
v___x_1592_ = lean_box(0);
v_isShared_1593_ = v_isSharedCheck_1695_;
goto v_resetjp_1591_;
}
v_resetjp_1591_:
{
lean_object* v_toOrderTop_1594_; lean_object* v_toHImp_1595_; lean_object* v___x_1597_; uint8_t v_isShared_1598_; uint8_t v_isSharedCheck_1693_; 
v_toOrderTop_1594_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1587_, 1);
v_toHImp_1595_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1587_, 2);
v_isSharedCheck_1693_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_1587_);
if (v_isSharedCheck_1693_ == 0)
{
lean_object* v_unused_1694_; 
v_unused_1694_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1587_, 0);
lean_dec(v_unused_1694_);
v___x_1597_ = v_toGeneralizedHeytingAlgebra_1587_;
v_isShared_1598_ = v_isSharedCheck_1693_;
goto v_resetjp_1596_;
}
else
{
lean_inc(v_toHImp_1595_);
lean_inc(v_toOrderTop_1594_);
lean_dec(v_toGeneralizedHeytingAlgebra_1587_);
v___x_1597_ = lean_box(0);
v_isShared_1598_ = v_isSharedCheck_1693_;
goto v_resetjp_1596_;
}
v_resetjp_1596_:
{
lean_object* v_toSemilatticeSup_1599_; lean_object* v_inf_1600_; lean_object* v___x_1602_; uint8_t v_isShared_1603_; uint8_t v_isSharedCheck_1692_; 
v_toSemilatticeSup_1599_ = lean_ctor_get(v_toLattice_1588_, 0);
v_inf_1600_ = lean_ctor_get(v_toLattice_1588_, 1);
v_isSharedCheck_1692_ = !lean_is_exclusive(v_toLattice_1588_);
if (v_isSharedCheck_1692_ == 0)
{
v___x_1602_ = v_toLattice_1588_;
v_isShared_1603_ = v_isSharedCheck_1692_;
goto v_resetjp_1601_;
}
else
{
lean_inc(v_inf_1600_);
lean_inc(v_toSemilatticeSup_1599_);
lean_dec(v_toLattice_1588_);
v___x_1602_ = lean_box(0);
v_isShared_1603_ = v_isSharedCheck_1692_;
goto v_resetjp_1601_;
}
v_resetjp_1601_:
{
lean_object* v___f_1604_; lean_object* v_min_1605_; lean_object* v_le_1606_; lean_object* v_lt_1607_; lean_object* v_semilatticeInf_1608_; lean_object* v_toPartialOrder_1609_; lean_object* v___x_1611_; uint8_t v_isShared_1612_; uint8_t v_isSharedCheck_1690_; 
v___f_1604_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc_ref(v_e_1585_);
v_min_1605_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1605_, 0, v_e_1585_);
lean_closure_set(v_min_1605_, 1, v___f_1604_);
lean_closure_set(v_min_1605_, 2, v_inf_1600_);
v_le_1606_ = lean_box(0);
v_lt_1607_ = lean_box(0);
lean_inc_ref(v_min_1605_);
v_semilatticeInf_1608_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1605_, v_le_1606_, v_lt_1607_);
v_toPartialOrder_1609_ = lean_ctor_get(v_semilatticeInf_1608_, 0);
v_isSharedCheck_1690_ = !lean_is_exclusive(v_semilatticeInf_1608_);
if (v_isSharedCheck_1690_ == 0)
{
lean_object* v_unused_1691_; 
v_unused_1691_ = lean_ctor_get(v_semilatticeInf_1608_, 1);
lean_dec(v_unused_1691_);
v___x_1611_ = v_semilatticeInf_1608_;
v_isShared_1612_ = v_isSharedCheck_1690_;
goto v_resetjp_1610_;
}
else
{
lean_inc(v_toPartialOrder_1609_);
lean_dec(v_semilatticeInf_1608_);
v___x_1611_ = lean_box(0);
v_isShared_1612_ = v_isSharedCheck_1690_;
goto v_resetjp_1610_;
}
v_resetjp_1610_:
{
lean_object* v_toLE_1613_; lean_object* v_toLT_1614_; lean_object* v___x_1616_; uint8_t v_isShared_1617_; uint8_t v_isSharedCheck_1689_; 
v_toLE_1613_ = lean_ctor_get(v_toPartialOrder_1609_, 0);
v_toLT_1614_ = lean_ctor_get(v_toPartialOrder_1609_, 1);
v_isSharedCheck_1689_ = !lean_is_exclusive(v_toPartialOrder_1609_);
if (v_isSharedCheck_1689_ == 0)
{
v___x_1616_ = v_toPartialOrder_1609_;
v_isShared_1617_ = v_isSharedCheck_1689_;
goto v_resetjp_1615_;
}
else
{
lean_inc(v_toLT_1614_);
lean_inc(v_toLE_1613_);
lean_dec(v_toPartialOrder_1609_);
v___x_1616_ = lean_box(0);
v_isShared_1617_ = v_isSharedCheck_1689_;
goto v_resetjp_1615_;
}
v_resetjp_1615_:
{
lean_object* v___f_1618_; lean_object* v___f_1619_; lean_object* v___x_1621_; 
v___f_1618_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1618_, 0, v_min_1605_);
lean_inc_ref(v_e_1585_);
v___f_1619_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1619_, 0, v_toSemilatticeSup_1599_);
lean_closure_set(v___f_1619_, 1, v_e_1585_);
lean_closure_set(v___f_1619_, 2, v___f_1604_);
if (v_isShared_1617_ == 0)
{
v___x_1621_ = v___x_1616_;
goto v_reusejp_1620_;
}
else
{
lean_object* v_reuseFailAlloc_1688_; 
v_reuseFailAlloc_1688_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1688_, 0, v_toLE_1613_);
lean_ctor_set(v_reuseFailAlloc_1688_, 1, v_toLT_1614_);
v___x_1621_ = v_reuseFailAlloc_1688_;
goto v_reusejp_1620_;
}
v_reusejp_1620_:
{
lean_object* v___x_1623_; 
lean_inc_ref(v___f_1619_);
if (v_isShared_1612_ == 0)
{
lean_ctor_set(v___x_1611_, 1, v___f_1619_);
lean_ctor_set(v___x_1611_, 0, v___x_1621_);
v___x_1623_ = v___x_1611_;
goto v_reusejp_1622_;
}
else
{
lean_object* v_reuseFailAlloc_1687_; 
v_reuseFailAlloc_1687_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1687_, 0, v___x_1621_);
lean_ctor_set(v_reuseFailAlloc_1687_, 1, v___f_1619_);
v___x_1623_ = v_reuseFailAlloc_1687_;
goto v_reusejp_1622_;
}
v_reusejp_1622_:
{
lean_object* v_lattice_1625_; 
lean_inc_ref(v___f_1618_);
if (v_isShared_1603_ == 0)
{
lean_ctor_set(v___x_1602_, 1, v___f_1618_);
lean_ctor_set(v___x_1602_, 0, v___x_1623_);
v_lattice_1625_ = v___x_1602_;
goto v_reusejp_1624_;
}
else
{
lean_object* v_reuseFailAlloc_1686_; 
v_reuseFailAlloc_1686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1686_, 0, v___x_1623_);
lean_ctor_set(v_reuseFailAlloc_1686_, 1, v___f_1618_);
v_lattice_1625_ = v_reuseFailAlloc_1686_;
goto v_reusejp_1624_;
}
v_reusejp_1624_:
{
lean_object* v___x_1626_; lean_object* v_toFun_1627_; lean_object* v___x_1629_; uint8_t v_isShared_1630_; uint8_t v_isSharedCheck_1684_; 
lean_inc_ref(v_e_1585_);
v___x_1626_ = lp_mathlib_Equiv_symm___redArg(v_e_1585_);
v_toFun_1627_ = lean_ctor_get(v___x_1626_, 0);
v_isSharedCheck_1684_ = !lean_is_exclusive(v___x_1626_);
if (v_isSharedCheck_1684_ == 0)
{
lean_object* v_unused_1685_; 
v_unused_1685_ = lean_ctor_get(v___x_1626_, 1);
lean_dec(v_unused_1685_);
v___x_1629_ = v___x_1626_;
v_isShared_1630_ = v_isSharedCheck_1684_;
goto v_resetjp_1628_;
}
else
{
lean_inc(v_toFun_1627_);
lean_dec(v___x_1626_);
v___x_1629_ = lean_box(0);
v_isShared_1630_ = v_isSharedCheck_1684_;
goto v_resetjp_1628_;
}
v_resetjp_1628_:
{
lean_object* v___x_1631_; lean_object* v_toPartialOrder_1632_; lean_object* v___x_1634_; uint8_t v_isShared_1635_; uint8_t v_isSharedCheck_1682_; 
v___x_1631_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1625_);
v_toPartialOrder_1632_ = lean_ctor_get(v___x_1631_, 0);
v_isSharedCheck_1682_ = !lean_is_exclusive(v___x_1631_);
if (v_isSharedCheck_1682_ == 0)
{
lean_object* v_unused_1683_; 
v_unused_1683_ = lean_ctor_get(v___x_1631_, 1);
lean_dec(v_unused_1683_);
v___x_1634_ = v___x_1631_;
v_isShared_1635_ = v_isSharedCheck_1682_;
goto v_resetjp_1633_;
}
else
{
lean_inc(v_toPartialOrder_1632_);
lean_dec(v___x_1631_);
v___x_1634_ = lean_box(0);
v_isShared_1635_ = v_isSharedCheck_1682_;
goto v_resetjp_1633_;
}
v_resetjp_1633_:
{
lean_object* v_toLE_1636_; lean_object* v_toLT_1637_; lean_object* v___x_1639_; uint8_t v_isShared_1640_; uint8_t v_isSharedCheck_1681_; 
v_toLE_1636_ = lean_ctor_get(v_toPartialOrder_1632_, 0);
v_toLT_1637_ = lean_ctor_get(v_toPartialOrder_1632_, 1);
v_isSharedCheck_1681_ = !lean_is_exclusive(v_toPartialOrder_1632_);
if (v_isSharedCheck_1681_ == 0)
{
v___x_1639_ = v_toPartialOrder_1632_;
v_isShared_1640_ = v_isSharedCheck_1681_;
goto v_resetjp_1638_;
}
else
{
lean_inc(v_toLT_1637_);
lean_inc(v_toLE_1636_);
lean_dec(v_toPartialOrder_1632_);
v___x_1639_ = lean_box(0);
v_isShared_1640_ = v_isSharedCheck_1681_;
goto v_resetjp_1638_;
}
v_resetjp_1638_:
{
lean_object* v___f_1641_; lean_object* v___x_1643_; 
v___f_1641_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1641_, 0, v___f_1619_);
if (v_isShared_1640_ == 0)
{
v___x_1643_ = v___x_1639_;
goto v_reusejp_1642_;
}
else
{
lean_object* v_reuseFailAlloc_1680_; 
v_reuseFailAlloc_1680_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1680_, 0, v_toLE_1636_);
lean_ctor_set(v_reuseFailAlloc_1680_, 1, v_toLT_1637_);
v___x_1643_ = v_reuseFailAlloc_1680_;
goto v_reusejp_1642_;
}
v_reusejp_1642_:
{
lean_object* v___x_1645_; 
lean_inc_ref(v___f_1641_);
if (v_isShared_1635_ == 0)
{
lean_ctor_set(v___x_1634_, 1, v___f_1641_);
lean_ctor_set(v___x_1634_, 0, v___x_1643_);
v___x_1645_ = v___x_1634_;
goto v_reusejp_1644_;
}
else
{
lean_object* v_reuseFailAlloc_1679_; 
v_reuseFailAlloc_1679_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1679_, 0, v___x_1643_);
lean_ctor_set(v_reuseFailAlloc_1679_, 1, v___f_1641_);
v___x_1645_ = v_reuseFailAlloc_1679_;
goto v_reusejp_1644_;
}
v_reusejp_1644_:
{
lean_object* v___x_1647_; 
lean_inc_ref(v___f_1618_);
if (v_isShared_1630_ == 0)
{
lean_ctor_set(v___x_1629_, 1, v___f_1618_);
lean_ctor_set(v___x_1629_, 0, v___x_1645_);
v___x_1647_ = v___x_1629_;
goto v_reusejp_1646_;
}
else
{
lean_object* v_reuseFailAlloc_1678_; 
v_reuseFailAlloc_1678_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1678_, 0, v___x_1645_);
lean_ctor_set(v_reuseFailAlloc_1678_, 1, v___f_1618_);
v___x_1647_ = v_reuseFailAlloc_1678_;
goto v_reusejp_1646_;
}
v_reusejp_1646_:
{
lean_object* v___x_1648_; lean_object* v_toPartialOrder_1649_; lean_object* v___x_1651_; uint8_t v_isShared_1652_; uint8_t v_isSharedCheck_1676_; 
v___x_1648_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_1647_);
v_toPartialOrder_1649_ = lean_ctor_get(v___x_1648_, 0);
v_isSharedCheck_1676_ = !lean_is_exclusive(v___x_1648_);
if (v_isSharedCheck_1676_ == 0)
{
lean_object* v_unused_1677_; 
v_unused_1677_ = lean_ctor_get(v___x_1648_, 1);
lean_dec(v_unused_1677_);
v___x_1651_ = v___x_1648_;
v_isShared_1652_ = v_isSharedCheck_1676_;
goto v_resetjp_1650_;
}
else
{
lean_inc(v_toPartialOrder_1649_);
lean_dec(v___x_1648_);
v___x_1651_ = lean_box(0);
v_isShared_1652_ = v_isSharedCheck_1676_;
goto v_resetjp_1650_;
}
v_resetjp_1650_:
{
lean_object* v_toLE_1653_; lean_object* v_toLT_1654_; lean_object* v___x_1656_; uint8_t v_isShared_1657_; uint8_t v_isSharedCheck_1675_; 
v_toLE_1653_ = lean_ctor_get(v_toPartialOrder_1649_, 0);
v_toLT_1654_ = lean_ctor_get(v_toPartialOrder_1649_, 1);
v_isSharedCheck_1675_ = !lean_is_exclusive(v_toPartialOrder_1649_);
if (v_isSharedCheck_1675_ == 0)
{
v___x_1656_ = v_toPartialOrder_1649_;
v_isShared_1657_ = v_isSharedCheck_1675_;
goto v_resetjp_1655_;
}
else
{
lean_inc(v_toLT_1654_);
lean_inc(v_toLE_1653_);
lean_dec(v_toPartialOrder_1649_);
v___x_1656_ = lean_box(0);
v_isShared_1657_ = v_isSharedCheck_1675_;
goto v_resetjp_1655_;
}
v_resetjp_1655_:
{
lean_object* v_top_1658_; lean_object* v_compl_1659_; lean_object* v_himp_1660_; lean_object* v_bot_1661_; lean_object* v___x_1663_; 
lean_inc_n(v_toFun_1627_, 3);
v_top_1658_ = lean_apply_1(v_toFun_1627_, v_toOrderTop_1594_);
lean_inc_ref(v_e_1585_);
v_compl_1659_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_heytingAlgebra___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_1659_, 0, v_e_1585_);
lean_closure_set(v_compl_1659_, 1, v_toCompl_1590_);
lean_closure_set(v_compl_1659_, 2, v_toFun_1627_);
v_himp_1660_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__3), 6, 4);
lean_closure_set(v_himp_1660_, 0, v___f_1604_);
lean_closure_set(v_himp_1660_, 1, v_e_1585_);
lean_closure_set(v_himp_1660_, 2, v_toHImp_1595_);
lean_closure_set(v_himp_1660_, 3, v_toFun_1627_);
v_bot_1661_ = lean_apply_1(v_toFun_1627_, v_toOrderBot_1589_);
if (v_isShared_1657_ == 0)
{
v___x_1663_ = v___x_1656_;
goto v_reusejp_1662_;
}
else
{
lean_object* v_reuseFailAlloc_1674_; 
v_reuseFailAlloc_1674_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1674_, 0, v_toLE_1653_);
lean_ctor_set(v_reuseFailAlloc_1674_, 1, v_toLT_1654_);
v___x_1663_ = v_reuseFailAlloc_1674_;
goto v_reusejp_1662_;
}
v_reusejp_1662_:
{
lean_object* v___x_1665_; 
if (v_isShared_1652_ == 0)
{
lean_ctor_set(v___x_1651_, 1, v___f_1641_);
lean_ctor_set(v___x_1651_, 0, v___x_1663_);
v___x_1665_ = v___x_1651_;
goto v_reusejp_1664_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v___x_1663_);
lean_ctor_set(v_reuseFailAlloc_1673_, 1, v___f_1641_);
v___x_1665_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1664_;
}
v_reusejp_1664_:
{
lean_object* v___x_1666_; lean_object* v___x_1668_; 
v___x_1666_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1666_, 0, v___x_1665_);
lean_ctor_set(v___x_1666_, 1, v___f_1618_);
if (v_isShared_1598_ == 0)
{
lean_ctor_set(v___x_1597_, 2, v_himp_1660_);
lean_ctor_set(v___x_1597_, 1, v_top_1658_);
lean_ctor_set(v___x_1597_, 0, v___x_1666_);
v___x_1668_ = v___x_1597_;
goto v_reusejp_1667_;
}
else
{
lean_object* v_reuseFailAlloc_1672_; 
v_reuseFailAlloc_1672_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1672_, 0, v___x_1666_);
lean_ctor_set(v_reuseFailAlloc_1672_, 1, v_top_1658_);
lean_ctor_set(v_reuseFailAlloc_1672_, 2, v_himp_1660_);
v___x_1668_ = v_reuseFailAlloc_1672_;
goto v_reusejp_1667_;
}
v_reusejp_1667_:
{
lean_object* v___x_1670_; 
if (v_isShared_1593_ == 0)
{
lean_ctor_set(v___x_1592_, 2, v_compl_1659_);
lean_ctor_set(v___x_1592_, 1, v_bot_1661_);
lean_ctor_set(v___x_1592_, 0, v___x_1668_);
v___x_1670_ = v___x_1592_;
goto v_reusejp_1669_;
}
else
{
lean_object* v_reuseFailAlloc_1671_; 
v_reuseFailAlloc_1671_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1671_, 0, v___x_1668_);
lean_ctor_set(v_reuseFailAlloc_1671_, 1, v_bot_1661_);
lean_ctor_set(v_reuseFailAlloc_1671_, 2, v_compl_1659_);
v___x_1670_ = v_reuseFailAlloc_1671_;
goto v_reusejp_1669_;
}
v_reusejp_1669_:
{
return v___x_1670_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coheytingAlgebra___redArg___lam__8(lean_object* v_e_1697_, lean_object* v_toHNot_1698_, lean_object* v_toFun_1699_, lean_object* v_a_1700_){
_start:
{
lean_object* v_toFun_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; 
v_toFun_1701_ = lean_ctor_get(v_e_1697_, 0);
lean_inc(v_toFun_1701_);
lean_dec_ref(v_e_1697_);
v___x_1702_ = lean_apply_1(v_toFun_1701_, v_a_1700_);
v___x_1703_ = lean_apply_1(v_toHNot_1698_, v___x_1702_);
v___x_1704_ = lean_apply_1(v_toFun_1699_, v___x_1703_);
return v___x_1704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coheytingAlgebra___redArg(lean_object* v_e_1705_, lean_object* v_inst_1706_){
_start:
{
lean_object* v_toGeneralizedCoheytingAlgebra_1707_; lean_object* v_toLattice_1708_; lean_object* v_toOrderTop_1709_; lean_object* v_toHNot_1710_; lean_object* v_toOrderBot_1711_; lean_object* v_toSDiff_1712_; lean_object* v_toSemilatticeSup_1713_; lean_object* v_inf_1714_; lean_object* v___x_1716_; uint8_t v_isShared_1717_; uint8_t v_isSharedCheck_1787_; 
v_toGeneralizedCoheytingAlgebra_1707_ = lean_ctor_get(v_inst_1706_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1707_);
v_toLattice_1708_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1707_, 0);
lean_inc_ref(v_toLattice_1708_);
v_toOrderTop_1709_ = lean_ctor_get(v_inst_1706_, 1);
lean_inc(v_toOrderTop_1709_);
v_toHNot_1710_ = lean_ctor_get(v_inst_1706_, 2);
lean_inc(v_toHNot_1710_);
lean_dec_ref(v_inst_1706_);
v_toOrderBot_1711_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1707_, 1);
lean_inc(v_toOrderBot_1711_);
v_toSDiff_1712_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1707_, 2);
lean_inc(v_toSDiff_1712_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1707_);
v_toSemilatticeSup_1713_ = lean_ctor_get(v_toLattice_1708_, 0);
v_inf_1714_ = lean_ctor_get(v_toLattice_1708_, 1);
v_isSharedCheck_1787_ = !lean_is_exclusive(v_toLattice_1708_);
if (v_isSharedCheck_1787_ == 0)
{
v___x_1716_ = v_toLattice_1708_;
v_isShared_1717_ = v_isSharedCheck_1787_;
goto v_resetjp_1715_;
}
else
{
lean_inc(v_inf_1714_);
lean_inc(v_toSemilatticeSup_1713_);
lean_dec(v_toLattice_1708_);
v___x_1716_ = lean_box(0);
v_isShared_1717_ = v_isSharedCheck_1787_;
goto v_resetjp_1715_;
}
v_resetjp_1715_:
{
lean_object* v___f_1718_; lean_object* v_min_1719_; lean_object* v_le_1720_; lean_object* v_lt_1721_; lean_object* v_semilatticeInf_1722_; lean_object* v_toPartialOrder_1723_; lean_object* v___x_1725_; uint8_t v_isShared_1726_; uint8_t v_isSharedCheck_1785_; 
v___f_1718_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc_ref(v_e_1705_);
v_min_1719_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1719_, 0, v_e_1705_);
lean_closure_set(v_min_1719_, 1, v___f_1718_);
lean_closure_set(v_min_1719_, 2, v_inf_1714_);
v_le_1720_ = lean_box(0);
v_lt_1721_ = lean_box(0);
lean_inc_ref(v_min_1719_);
v_semilatticeInf_1722_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1719_, v_le_1720_, v_lt_1721_);
v_toPartialOrder_1723_ = lean_ctor_get(v_semilatticeInf_1722_, 0);
v_isSharedCheck_1785_ = !lean_is_exclusive(v_semilatticeInf_1722_);
if (v_isSharedCheck_1785_ == 0)
{
lean_object* v_unused_1786_; 
v_unused_1786_ = lean_ctor_get(v_semilatticeInf_1722_, 1);
lean_dec(v_unused_1786_);
v___x_1725_ = v_semilatticeInf_1722_;
v_isShared_1726_ = v_isSharedCheck_1785_;
goto v_resetjp_1724_;
}
else
{
lean_inc(v_toPartialOrder_1723_);
lean_dec(v_semilatticeInf_1722_);
v___x_1725_ = lean_box(0);
v_isShared_1726_ = v_isSharedCheck_1785_;
goto v_resetjp_1724_;
}
v_resetjp_1724_:
{
lean_object* v_toLE_1727_; lean_object* v_toLT_1728_; lean_object* v___x_1730_; uint8_t v_isShared_1731_; uint8_t v_isSharedCheck_1784_; 
v_toLE_1727_ = lean_ctor_get(v_toPartialOrder_1723_, 0);
v_toLT_1728_ = lean_ctor_get(v_toPartialOrder_1723_, 1);
v_isSharedCheck_1784_ = !lean_is_exclusive(v_toPartialOrder_1723_);
if (v_isSharedCheck_1784_ == 0)
{
v___x_1730_ = v_toPartialOrder_1723_;
v_isShared_1731_ = v_isSharedCheck_1784_;
goto v_resetjp_1729_;
}
else
{
lean_inc(v_toLT_1728_);
lean_inc(v_toLE_1727_);
lean_dec(v_toPartialOrder_1723_);
v___x_1730_ = lean_box(0);
v_isShared_1731_ = v_isSharedCheck_1784_;
goto v_resetjp_1729_;
}
v_resetjp_1729_:
{
lean_object* v___f_1732_; lean_object* v___f_1733_; lean_object* v___x_1735_; 
v___f_1732_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1732_, 0, v_min_1719_);
lean_inc_ref(v_e_1705_);
v___f_1733_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1733_, 0, v_toSemilatticeSup_1713_);
lean_closure_set(v___f_1733_, 1, v_e_1705_);
lean_closure_set(v___f_1733_, 2, v___f_1718_);
if (v_isShared_1731_ == 0)
{
v___x_1735_ = v___x_1730_;
goto v_reusejp_1734_;
}
else
{
lean_object* v_reuseFailAlloc_1783_; 
v_reuseFailAlloc_1783_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1783_, 0, v_toLE_1727_);
lean_ctor_set(v_reuseFailAlloc_1783_, 1, v_toLT_1728_);
v___x_1735_ = v_reuseFailAlloc_1783_;
goto v_reusejp_1734_;
}
v_reusejp_1734_:
{
lean_object* v___x_1737_; 
lean_inc_ref(v___f_1733_);
if (v_isShared_1726_ == 0)
{
lean_ctor_set(v___x_1725_, 1, v___f_1733_);
lean_ctor_set(v___x_1725_, 0, v___x_1735_);
v___x_1737_ = v___x_1725_;
goto v_reusejp_1736_;
}
else
{
lean_object* v_reuseFailAlloc_1782_; 
v_reuseFailAlloc_1782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1782_, 0, v___x_1735_);
lean_ctor_set(v_reuseFailAlloc_1782_, 1, v___f_1733_);
v___x_1737_ = v_reuseFailAlloc_1782_;
goto v_reusejp_1736_;
}
v_reusejp_1736_:
{
lean_object* v_lattice_1739_; 
lean_inc_ref(v___f_1732_);
if (v_isShared_1717_ == 0)
{
lean_ctor_set(v___x_1716_, 1, v___f_1732_);
lean_ctor_set(v___x_1716_, 0, v___x_1737_);
v_lattice_1739_ = v___x_1716_;
goto v_reusejp_1738_;
}
else
{
lean_object* v_reuseFailAlloc_1781_; 
v_reuseFailAlloc_1781_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1781_, 0, v___x_1737_);
lean_ctor_set(v_reuseFailAlloc_1781_, 1, v___f_1732_);
v_lattice_1739_ = v_reuseFailAlloc_1781_;
goto v_reusejp_1738_;
}
v_reusejp_1738_:
{
lean_object* v___x_1740_; lean_object* v_toFun_1741_; lean_object* v___x_1743_; uint8_t v_isShared_1744_; uint8_t v_isSharedCheck_1779_; 
lean_inc_ref(v_e_1705_);
v___x_1740_ = lp_mathlib_Equiv_symm___redArg(v_e_1705_);
v_toFun_1741_ = lean_ctor_get(v___x_1740_, 0);
v_isSharedCheck_1779_ = !lean_is_exclusive(v___x_1740_);
if (v_isSharedCheck_1779_ == 0)
{
lean_object* v_unused_1780_; 
v_unused_1780_ = lean_ctor_get(v___x_1740_, 1);
lean_dec(v_unused_1780_);
v___x_1743_ = v___x_1740_;
v_isShared_1744_ = v_isSharedCheck_1779_;
goto v_resetjp_1742_;
}
else
{
lean_inc(v_toFun_1741_);
lean_dec(v___x_1740_);
v___x_1743_ = lean_box(0);
v_isShared_1744_ = v_isSharedCheck_1779_;
goto v_resetjp_1742_;
}
v_resetjp_1742_:
{
lean_object* v___x_1745_; lean_object* v_toPartialOrder_1746_; lean_object* v___x_1748_; uint8_t v_isShared_1749_; uint8_t v_isSharedCheck_1777_; 
v___x_1745_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1739_);
v_toPartialOrder_1746_ = lean_ctor_get(v___x_1745_, 0);
v_isSharedCheck_1777_ = !lean_is_exclusive(v___x_1745_);
if (v_isSharedCheck_1777_ == 0)
{
lean_object* v_unused_1778_; 
v_unused_1778_ = lean_ctor_get(v___x_1745_, 1);
lean_dec(v_unused_1778_);
v___x_1748_ = v___x_1745_;
v_isShared_1749_ = v_isSharedCheck_1777_;
goto v_resetjp_1747_;
}
else
{
lean_inc(v_toPartialOrder_1746_);
lean_dec(v___x_1745_);
v___x_1748_ = lean_box(0);
v_isShared_1749_ = v_isSharedCheck_1777_;
goto v_resetjp_1747_;
}
v_resetjp_1747_:
{
lean_object* v_toLE_1750_; lean_object* v_toLT_1751_; lean_object* v___x_1753_; uint8_t v_isShared_1754_; uint8_t v_isSharedCheck_1776_; 
v_toLE_1750_ = lean_ctor_get(v_toPartialOrder_1746_, 0);
v_toLT_1751_ = lean_ctor_get(v_toPartialOrder_1746_, 1);
v_isSharedCheck_1776_ = !lean_is_exclusive(v_toPartialOrder_1746_);
if (v_isSharedCheck_1776_ == 0)
{
v___x_1753_ = v_toPartialOrder_1746_;
v_isShared_1754_ = v_isSharedCheck_1776_;
goto v_resetjp_1752_;
}
else
{
lean_inc(v_toLT_1751_);
lean_inc(v_toLE_1750_);
lean_dec(v_toPartialOrder_1746_);
v___x_1753_ = lean_box(0);
v_isShared_1754_ = v_isSharedCheck_1776_;
goto v_resetjp_1752_;
}
v_resetjp_1752_:
{
lean_object* v___f_1755_; lean_object* v___x_1757_; 
v___f_1755_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1755_, 0, v___f_1733_);
if (v_isShared_1754_ == 0)
{
v___x_1757_ = v___x_1753_;
goto v_reusejp_1756_;
}
else
{
lean_object* v_reuseFailAlloc_1775_; 
v_reuseFailAlloc_1775_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1775_, 0, v_toLE_1750_);
lean_ctor_set(v_reuseFailAlloc_1775_, 1, v_toLT_1751_);
v___x_1757_ = v_reuseFailAlloc_1775_;
goto v_reusejp_1756_;
}
v_reusejp_1756_:
{
lean_object* v___x_1759_; 
if (v_isShared_1749_ == 0)
{
lean_ctor_set(v___x_1748_, 1, v___f_1755_);
lean_ctor_set(v___x_1748_, 0, v___x_1757_);
v___x_1759_ = v___x_1748_;
goto v_reusejp_1758_;
}
else
{
lean_object* v_reuseFailAlloc_1774_; 
v_reuseFailAlloc_1774_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1774_, 0, v___x_1757_);
lean_ctor_set(v_reuseFailAlloc_1774_, 1, v___f_1755_);
v___x_1759_ = v_reuseFailAlloc_1774_;
goto v_reusejp_1758_;
}
v_reusejp_1758_:
{
lean_object* v___x_1761_; 
lean_inc_ref(v___x_1759_);
if (v_isShared_1744_ == 0)
{
lean_ctor_set(v___x_1743_, 1, v___f_1732_);
lean_ctor_set(v___x_1743_, 0, v___x_1759_);
v___x_1761_ = v___x_1743_;
goto v_reusejp_1760_;
}
else
{
lean_object* v_reuseFailAlloc_1773_; 
v_reuseFailAlloc_1773_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1773_, 0, v___x_1759_);
lean_ctor_set(v_reuseFailAlloc_1773_, 1, v___f_1732_);
v___x_1761_ = v_reuseFailAlloc_1773_;
goto v_reusejp_1760_;
}
v_reusejp_1760_:
{
lean_object* v___x_1762_; lean_object* v_toPartialOrder_1763_; lean_object* v_toLE_1764_; lean_object* v_toLT_1765_; lean_object* v_bot_1766_; lean_object* v_hnot_1767_; lean_object* v_sdiff_1768_; lean_object* v_top_1769_; lean_object* v___f_1770_; lean_object* v___f_1771_; lean_object* v___x_1772_; 
v___x_1762_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_1761_);
v_toPartialOrder_1763_ = lean_ctor_get(v___x_1762_, 0);
lean_inc_ref(v_toPartialOrder_1763_);
v_toLE_1764_ = lean_ctor_get(v_toPartialOrder_1763_, 0);
lean_inc(v_toLE_1764_);
v_toLT_1765_ = lean_ctor_get(v_toPartialOrder_1763_, 1);
lean_inc(v_toLT_1765_);
lean_dec_ref(v_toPartialOrder_1763_);
lean_inc_n(v_toFun_1741_, 3);
v_bot_1766_ = lean_apply_1(v_toFun_1741_, v_toOrderBot_1711_);
lean_inc_ref(v_e_1705_);
v_hnot_1767_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coheytingAlgebra___redArg___lam__8), 4, 3);
lean_closure_set(v_hnot_1767_, 0, v_e_1705_);
lean_closure_set(v_hnot_1767_, 1, v_toHNot_1710_);
lean_closure_set(v_hnot_1767_, 2, v_toFun_1741_);
v_sdiff_1768_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg___lam__8), 6, 4);
lean_closure_set(v_sdiff_1768_, 0, v___f_1718_);
lean_closure_set(v_sdiff_1768_, 1, v_e_1705_);
lean_closure_set(v_sdiff_1768_, 2, v_toSDiff_1712_);
lean_closure_set(v_sdiff_1768_, 3, v_toFun_1741_);
v_top_1769_ = lean_apply_1(v_toFun_1741_, v_toOrderTop_1709_);
v___f_1770_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1770_, 0, v___x_1762_);
v___f_1771_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1771_, 0, v___x_1759_);
v___x_1772_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_1770_, v___f_1771_, v_toLE_1764_, v_toLT_1765_, v_bot_1766_, v_top_1769_, v_hnot_1767_, v_sdiff_1768_);
return v___x_1772_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coheytingAlgebra(lean_object* v_00_u03b1_1788_, lean_object* v_00_u03b2_1789_, lean_object* v_e_1790_, lean_object* v_inst_1791_){
_start:
{
lean_object* v_toGeneralizedCoheytingAlgebra_1792_; lean_object* v_toLattice_1793_; lean_object* v_toOrderTop_1794_; lean_object* v_toHNot_1795_; lean_object* v_toOrderBot_1796_; lean_object* v_toSDiff_1797_; lean_object* v_toSemilatticeSup_1798_; lean_object* v_inf_1799_; lean_object* v___x_1801_; uint8_t v_isShared_1802_; uint8_t v_isSharedCheck_1872_; 
v_toGeneralizedCoheytingAlgebra_1792_ = lean_ctor_get(v_inst_1791_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1792_);
v_toLattice_1793_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1792_, 0);
lean_inc_ref(v_toLattice_1793_);
v_toOrderTop_1794_ = lean_ctor_get(v_inst_1791_, 1);
lean_inc(v_toOrderTop_1794_);
v_toHNot_1795_ = lean_ctor_get(v_inst_1791_, 2);
lean_inc(v_toHNot_1795_);
lean_dec_ref(v_inst_1791_);
v_toOrderBot_1796_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1792_, 1);
lean_inc(v_toOrderBot_1796_);
v_toSDiff_1797_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1792_, 2);
lean_inc(v_toSDiff_1797_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1792_);
v_toSemilatticeSup_1798_ = lean_ctor_get(v_toLattice_1793_, 0);
v_inf_1799_ = lean_ctor_get(v_toLattice_1793_, 1);
v_isSharedCheck_1872_ = !lean_is_exclusive(v_toLattice_1793_);
if (v_isSharedCheck_1872_ == 0)
{
v___x_1801_ = v_toLattice_1793_;
v_isShared_1802_ = v_isSharedCheck_1872_;
goto v_resetjp_1800_;
}
else
{
lean_inc(v_inf_1799_);
lean_inc(v_toSemilatticeSup_1798_);
lean_dec(v_toLattice_1793_);
v___x_1801_ = lean_box(0);
v_isShared_1802_ = v_isSharedCheck_1872_;
goto v_resetjp_1800_;
}
v_resetjp_1800_:
{
lean_object* v___f_1803_; lean_object* v_min_1804_; lean_object* v_le_1805_; lean_object* v_lt_1806_; lean_object* v_semilatticeInf_1807_; lean_object* v_toPartialOrder_1808_; lean_object* v___x_1810_; uint8_t v_isShared_1811_; uint8_t v_isSharedCheck_1870_; 
v___f_1803_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc_ref(v_e_1790_);
v_min_1804_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1804_, 0, v_e_1790_);
lean_closure_set(v_min_1804_, 1, v___f_1803_);
lean_closure_set(v_min_1804_, 2, v_inf_1799_);
v_le_1805_ = lean_box(0);
v_lt_1806_ = lean_box(0);
lean_inc_ref(v_min_1804_);
v_semilatticeInf_1807_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1804_, v_le_1805_, v_lt_1806_);
v_toPartialOrder_1808_ = lean_ctor_get(v_semilatticeInf_1807_, 0);
v_isSharedCheck_1870_ = !lean_is_exclusive(v_semilatticeInf_1807_);
if (v_isSharedCheck_1870_ == 0)
{
lean_object* v_unused_1871_; 
v_unused_1871_ = lean_ctor_get(v_semilatticeInf_1807_, 1);
lean_dec(v_unused_1871_);
v___x_1810_ = v_semilatticeInf_1807_;
v_isShared_1811_ = v_isSharedCheck_1870_;
goto v_resetjp_1809_;
}
else
{
lean_inc(v_toPartialOrder_1808_);
lean_dec(v_semilatticeInf_1807_);
v___x_1810_ = lean_box(0);
v_isShared_1811_ = v_isSharedCheck_1870_;
goto v_resetjp_1809_;
}
v_resetjp_1809_:
{
lean_object* v_toLE_1812_; lean_object* v_toLT_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1869_; 
v_toLE_1812_ = lean_ctor_get(v_toPartialOrder_1808_, 0);
v_toLT_1813_ = lean_ctor_get(v_toPartialOrder_1808_, 1);
v_isSharedCheck_1869_ = !lean_is_exclusive(v_toPartialOrder_1808_);
if (v_isSharedCheck_1869_ == 0)
{
v___x_1815_ = v_toPartialOrder_1808_;
v_isShared_1816_ = v_isSharedCheck_1869_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_toLT_1813_);
lean_inc(v_toLE_1812_);
lean_dec(v_toPartialOrder_1808_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1869_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___f_1817_; lean_object* v___f_1818_; lean_object* v___x_1820_; 
v___f_1817_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1817_, 0, v_min_1804_);
lean_inc_ref(v_e_1790_);
v___f_1818_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1818_, 0, v_toSemilatticeSup_1798_);
lean_closure_set(v___f_1818_, 1, v_e_1790_);
lean_closure_set(v___f_1818_, 2, v___f_1803_);
if (v_isShared_1816_ == 0)
{
v___x_1820_ = v___x_1815_;
goto v_reusejp_1819_;
}
else
{
lean_object* v_reuseFailAlloc_1868_; 
v_reuseFailAlloc_1868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1868_, 0, v_toLE_1812_);
lean_ctor_set(v_reuseFailAlloc_1868_, 1, v_toLT_1813_);
v___x_1820_ = v_reuseFailAlloc_1868_;
goto v_reusejp_1819_;
}
v_reusejp_1819_:
{
lean_object* v___x_1822_; 
lean_inc_ref(v___f_1818_);
if (v_isShared_1811_ == 0)
{
lean_ctor_set(v___x_1810_, 1, v___f_1818_);
lean_ctor_set(v___x_1810_, 0, v___x_1820_);
v___x_1822_ = v___x_1810_;
goto v_reusejp_1821_;
}
else
{
lean_object* v_reuseFailAlloc_1867_; 
v_reuseFailAlloc_1867_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1867_, 0, v___x_1820_);
lean_ctor_set(v_reuseFailAlloc_1867_, 1, v___f_1818_);
v___x_1822_ = v_reuseFailAlloc_1867_;
goto v_reusejp_1821_;
}
v_reusejp_1821_:
{
lean_object* v_lattice_1824_; 
lean_inc_ref(v___f_1817_);
if (v_isShared_1802_ == 0)
{
lean_ctor_set(v___x_1801_, 1, v___f_1817_);
lean_ctor_set(v___x_1801_, 0, v___x_1822_);
v_lattice_1824_ = v___x_1801_;
goto v_reusejp_1823_;
}
else
{
lean_object* v_reuseFailAlloc_1866_; 
v_reuseFailAlloc_1866_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1866_, 0, v___x_1822_);
lean_ctor_set(v_reuseFailAlloc_1866_, 1, v___f_1817_);
v_lattice_1824_ = v_reuseFailAlloc_1866_;
goto v_reusejp_1823_;
}
v_reusejp_1823_:
{
lean_object* v___x_1825_; lean_object* v_toFun_1826_; lean_object* v___x_1828_; uint8_t v_isShared_1829_; uint8_t v_isSharedCheck_1864_; 
lean_inc_ref(v_e_1790_);
v___x_1825_ = lp_mathlib_Equiv_symm___redArg(v_e_1790_);
v_toFun_1826_ = lean_ctor_get(v___x_1825_, 0);
v_isSharedCheck_1864_ = !lean_is_exclusive(v___x_1825_);
if (v_isSharedCheck_1864_ == 0)
{
lean_object* v_unused_1865_; 
v_unused_1865_ = lean_ctor_get(v___x_1825_, 1);
lean_dec(v_unused_1865_);
v___x_1828_ = v___x_1825_;
v_isShared_1829_ = v_isSharedCheck_1864_;
goto v_resetjp_1827_;
}
else
{
lean_inc(v_toFun_1826_);
lean_dec(v___x_1825_);
v___x_1828_ = lean_box(0);
v_isShared_1829_ = v_isSharedCheck_1864_;
goto v_resetjp_1827_;
}
v_resetjp_1827_:
{
lean_object* v___x_1830_; lean_object* v_toPartialOrder_1831_; lean_object* v___x_1833_; uint8_t v_isShared_1834_; uint8_t v_isSharedCheck_1862_; 
v___x_1830_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1824_);
v_toPartialOrder_1831_ = lean_ctor_get(v___x_1830_, 0);
v_isSharedCheck_1862_ = !lean_is_exclusive(v___x_1830_);
if (v_isSharedCheck_1862_ == 0)
{
lean_object* v_unused_1863_; 
v_unused_1863_ = lean_ctor_get(v___x_1830_, 1);
lean_dec(v_unused_1863_);
v___x_1833_ = v___x_1830_;
v_isShared_1834_ = v_isSharedCheck_1862_;
goto v_resetjp_1832_;
}
else
{
lean_inc(v_toPartialOrder_1831_);
lean_dec(v___x_1830_);
v___x_1833_ = lean_box(0);
v_isShared_1834_ = v_isSharedCheck_1862_;
goto v_resetjp_1832_;
}
v_resetjp_1832_:
{
lean_object* v_toLE_1835_; lean_object* v_toLT_1836_; lean_object* v___x_1838_; uint8_t v_isShared_1839_; uint8_t v_isSharedCheck_1861_; 
v_toLE_1835_ = lean_ctor_get(v_toPartialOrder_1831_, 0);
v_toLT_1836_ = lean_ctor_get(v_toPartialOrder_1831_, 1);
v_isSharedCheck_1861_ = !lean_is_exclusive(v_toPartialOrder_1831_);
if (v_isSharedCheck_1861_ == 0)
{
v___x_1838_ = v_toPartialOrder_1831_;
v_isShared_1839_ = v_isSharedCheck_1861_;
goto v_resetjp_1837_;
}
else
{
lean_inc(v_toLT_1836_);
lean_inc(v_toLE_1835_);
lean_dec(v_toPartialOrder_1831_);
v___x_1838_ = lean_box(0);
v_isShared_1839_ = v_isSharedCheck_1861_;
goto v_resetjp_1837_;
}
v_resetjp_1837_:
{
lean_object* v___f_1840_; lean_object* v___x_1842_; 
v___f_1840_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1840_, 0, v___f_1818_);
if (v_isShared_1839_ == 0)
{
v___x_1842_ = v___x_1838_;
goto v_reusejp_1841_;
}
else
{
lean_object* v_reuseFailAlloc_1860_; 
v_reuseFailAlloc_1860_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1860_, 0, v_toLE_1835_);
lean_ctor_set(v_reuseFailAlloc_1860_, 1, v_toLT_1836_);
v___x_1842_ = v_reuseFailAlloc_1860_;
goto v_reusejp_1841_;
}
v_reusejp_1841_:
{
lean_object* v___x_1844_; 
if (v_isShared_1834_ == 0)
{
lean_ctor_set(v___x_1833_, 1, v___f_1840_);
lean_ctor_set(v___x_1833_, 0, v___x_1842_);
v___x_1844_ = v___x_1833_;
goto v_reusejp_1843_;
}
else
{
lean_object* v_reuseFailAlloc_1859_; 
v_reuseFailAlloc_1859_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1859_, 0, v___x_1842_);
lean_ctor_set(v_reuseFailAlloc_1859_, 1, v___f_1840_);
v___x_1844_ = v_reuseFailAlloc_1859_;
goto v_reusejp_1843_;
}
v_reusejp_1843_:
{
lean_object* v___x_1846_; 
lean_inc_ref(v___x_1844_);
if (v_isShared_1829_ == 0)
{
lean_ctor_set(v___x_1828_, 1, v___f_1817_);
lean_ctor_set(v___x_1828_, 0, v___x_1844_);
v___x_1846_ = v___x_1828_;
goto v_reusejp_1845_;
}
else
{
lean_object* v_reuseFailAlloc_1858_; 
v_reuseFailAlloc_1858_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1858_, 0, v___x_1844_);
lean_ctor_set(v_reuseFailAlloc_1858_, 1, v___f_1817_);
v___x_1846_ = v_reuseFailAlloc_1858_;
goto v_reusejp_1845_;
}
v_reusejp_1845_:
{
lean_object* v___x_1847_; lean_object* v_toPartialOrder_1848_; lean_object* v_toLE_1849_; lean_object* v_toLT_1850_; lean_object* v_bot_1851_; lean_object* v_hnot_1852_; lean_object* v_sdiff_1853_; lean_object* v_top_1854_; lean_object* v___f_1855_; lean_object* v___f_1856_; lean_object* v___x_1857_; 
v___x_1847_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_1846_);
v_toPartialOrder_1848_ = lean_ctor_get(v___x_1847_, 0);
lean_inc_ref(v_toPartialOrder_1848_);
v_toLE_1849_ = lean_ctor_get(v_toPartialOrder_1848_, 0);
lean_inc(v_toLE_1849_);
v_toLT_1850_ = lean_ctor_get(v_toPartialOrder_1848_, 1);
lean_inc(v_toLT_1850_);
lean_dec_ref(v_toPartialOrder_1848_);
lean_inc_n(v_toFun_1826_, 3);
v_bot_1851_ = lean_apply_1(v_toFun_1826_, v_toOrderBot_1796_);
lean_inc_ref(v_e_1790_);
v_hnot_1852_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coheytingAlgebra___redArg___lam__8), 4, 3);
lean_closure_set(v_hnot_1852_, 0, v_e_1790_);
lean_closure_set(v_hnot_1852_, 1, v_toHNot_1795_);
lean_closure_set(v_hnot_1852_, 2, v_toFun_1826_);
v_sdiff_1853_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg___lam__8), 6, 4);
lean_closure_set(v_sdiff_1853_, 0, v___f_1803_);
lean_closure_set(v_sdiff_1853_, 1, v_e_1790_);
lean_closure_set(v_sdiff_1853_, 2, v_toSDiff_1797_);
lean_closure_set(v_sdiff_1853_, 3, v_toFun_1826_);
v_top_1854_ = lean_apply_1(v_toFun_1826_, v_toOrderTop_1794_);
v___f_1855_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1855_, 0, v___x_1847_);
v___f_1856_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1856_, 0, v___x_1844_);
v___x_1857_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_1855_, v___f_1856_, v_toLE_1849_, v_toLT_1850_, v_bot_1851_, v_top_1854_, v_hnot_1852_, v_sdiff_1853_);
return v___x_1857_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_biheytingAlgebra___redArg___lam__1(lean_object* v___f_1873_, lean_object* v_e_1874_, lean_object* v_inf_1875_, lean_object* v_toFun_1876_, lean_object* v_a_1877_, lean_object* v_b_1878_){
_start:
{
lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; 
lean_inc(v___f_1873_);
lean_inc_ref(v_e_1874_);
v___x_1879_ = lean_apply_2(v___f_1873_, v_e_1874_, v_a_1877_);
v___x_1880_ = lean_apply_2(v___f_1873_, v_e_1874_, v_b_1878_);
v___x_1881_ = lean_apply_2(v_inf_1875_, v___x_1879_, v___x_1880_);
v___x_1882_ = lean_apply_1(v_toFun_1876_, v___x_1881_);
return v___x_1882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_biheytingAlgebra___redArg___lam__3(lean_object* v_toSemilatticeSup_1883_, lean_object* v___f_1884_, lean_object* v_e_1885_, lean_object* v_toFun_1886_, lean_object* v_a_1887_, lean_object* v_b_1888_){
_start:
{
lean_object* v_sup_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; 
v_sup_1889_ = lean_ctor_get(v_toSemilatticeSup_1883_, 1);
lean_inc(v_sup_1889_);
lean_dec_ref(v_toSemilatticeSup_1883_);
lean_inc(v___f_1884_);
lean_inc_ref(v_e_1885_);
v___x_1890_ = lean_apply_2(v___f_1884_, v_e_1885_, v_a_1887_);
v___x_1891_ = lean_apply_2(v___f_1884_, v_e_1885_, v_b_1888_);
v___x_1892_ = lean_apply_2(v_sup_1889_, v___x_1890_, v___x_1891_);
v___x_1893_ = lean_apply_1(v_toFun_1886_, v___x_1892_);
return v___x_1893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_biheytingAlgebra___redArg(lean_object* v_e_1894_, lean_object* v_inst_1895_){
_start:
{
lean_object* v_toHeytingAlgebra_1896_; lean_object* v_toGeneralizedHeytingAlgebra_1897_; lean_object* v_toOrderBot_1898_; lean_object* v_toCompl_1899_; lean_object* v___x_1901_; uint8_t v_isShared_1902_; uint8_t v_isSharedCheck_2038_; 
v_toHeytingAlgebra_1896_ = lean_ctor_get(v_inst_1895_, 0);
lean_inc_ref(v_toHeytingAlgebra_1896_);
v_toGeneralizedHeytingAlgebra_1897_ = lean_ctor_get(v_toHeytingAlgebra_1896_, 0);
v_toOrderBot_1898_ = lean_ctor_get(v_toHeytingAlgebra_1896_, 1);
v_toCompl_1899_ = lean_ctor_get(v_toHeytingAlgebra_1896_, 2);
v_isSharedCheck_2038_ = !lean_is_exclusive(v_toHeytingAlgebra_1896_);
if (v_isSharedCheck_2038_ == 0)
{
v___x_1901_ = v_toHeytingAlgebra_1896_;
v_isShared_1902_ = v_isSharedCheck_2038_;
goto v_resetjp_1900_;
}
else
{
lean_inc(v_toCompl_1899_);
lean_inc(v_toOrderBot_1898_);
lean_inc(v_toGeneralizedHeytingAlgebra_1897_);
lean_dec(v_toHeytingAlgebra_1896_);
v___x_1901_ = lean_box(0);
v_isShared_1902_ = v_isSharedCheck_2038_;
goto v_resetjp_1900_;
}
v_resetjp_1900_:
{
lean_object* v___x_1903_; lean_object* v_toFun_1904_; lean_object* v___x_1906_; uint8_t v_isShared_1907_; uint8_t v_isSharedCheck_2036_; 
lean_inc_ref(v_e_1894_);
v___x_1903_ = lp_mathlib_Equiv_symm___redArg(v_e_1894_);
v_toFun_1904_ = lean_ctor_get(v___x_1903_, 0);
v_isSharedCheck_2036_ = !lean_is_exclusive(v___x_1903_);
if (v_isSharedCheck_2036_ == 0)
{
lean_object* v_unused_2037_; 
v_unused_2037_ = lean_ctor_get(v___x_1903_, 1);
lean_dec(v_unused_2037_);
v___x_1906_ = v___x_1903_;
v_isShared_1907_ = v_isSharedCheck_2036_;
goto v_resetjp_1905_;
}
else
{
lean_inc(v_toFun_1904_);
lean_dec(v___x_1903_);
v___x_1906_ = lean_box(0);
v_isShared_1907_ = v_isSharedCheck_2036_;
goto v_resetjp_1905_;
}
v_resetjp_1905_:
{
lean_object* v_toHImp_1908_; lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_2033_; 
v_toHImp_1908_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1897_, 2);
v_isSharedCheck_2033_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_1897_);
if (v_isSharedCheck_2033_ == 0)
{
lean_object* v_unused_2034_; lean_object* v_unused_2035_; 
v_unused_2034_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1897_, 1);
lean_dec(v_unused_2034_);
v_unused_2035_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_1897_, 0);
lean_dec(v_unused_2035_);
v___x_1910_ = v_toGeneralizedHeytingAlgebra_1897_;
v_isShared_1911_ = v_isSharedCheck_2033_;
goto v_resetjp_1909_;
}
else
{
lean_inc(v_toHImp_1908_);
lean_dec(v_toGeneralizedHeytingAlgebra_1897_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_2033_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
lean_object* v___x_1912_; lean_object* v_toGeneralizedCoheytingAlgebra_1913_; lean_object* v_toLattice_1914_; lean_object* v_toOrderTop_1915_; lean_object* v_toHNot_1916_; lean_object* v_toOrderBot_1917_; lean_object* v_toSDiff_1918_; lean_object* v_toSemilatticeSup_1919_; lean_object* v_inf_1920_; lean_object* v___x_1922_; uint8_t v_isShared_1923_; uint8_t v_isSharedCheck_2032_; 
v___x_1912_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_inst_1895_);
v_toGeneralizedCoheytingAlgebra_1913_ = lean_ctor_get(v___x_1912_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1913_);
v_toLattice_1914_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1913_, 0);
lean_inc_ref(v_toLattice_1914_);
v_toOrderTop_1915_ = lean_ctor_get(v___x_1912_, 1);
lean_inc(v_toOrderTop_1915_);
v_toHNot_1916_ = lean_ctor_get(v___x_1912_, 2);
lean_inc(v_toHNot_1916_);
lean_dec_ref(v___x_1912_);
v_toOrderBot_1917_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1913_, 1);
lean_inc(v_toOrderBot_1917_);
v_toSDiff_1918_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1913_, 2);
lean_inc(v_toSDiff_1918_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_1913_);
v_toSemilatticeSup_1919_ = lean_ctor_get(v_toLattice_1914_, 0);
v_inf_1920_ = lean_ctor_get(v_toLattice_1914_, 1);
v_isSharedCheck_2032_ = !lean_is_exclusive(v_toLattice_1914_);
if (v_isSharedCheck_2032_ == 0)
{
v___x_1922_ = v_toLattice_1914_;
v_isShared_1923_ = v_isSharedCheck_2032_;
goto v_resetjp_1921_;
}
else
{
lean_inc(v_inf_1920_);
lean_inc(v_toSemilatticeSup_1919_);
lean_dec(v_toLattice_1914_);
v___x_1922_ = lean_box(0);
v_isShared_1923_ = v_isSharedCheck_2032_;
goto v_resetjp_1921_;
}
v_resetjp_1921_:
{
lean_object* v___f_1924_; lean_object* v_min_1925_; lean_object* v_le_1926_; lean_object* v_lt_1927_; lean_object* v_semilatticeInf_1928_; lean_object* v_toPartialOrder_1929_; lean_object* v___x_1931_; uint8_t v_isShared_1932_; uint8_t v_isSharedCheck_2030_; 
v___f_1924_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc(v_toFun_1904_);
lean_inc_ref(v_e_1894_);
v_min_1925_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_biheytingAlgebra___redArg___lam__1), 6, 4);
lean_closure_set(v_min_1925_, 0, v___f_1924_);
lean_closure_set(v_min_1925_, 1, v_e_1894_);
lean_closure_set(v_min_1925_, 2, v_inf_1920_);
lean_closure_set(v_min_1925_, 3, v_toFun_1904_);
v_le_1926_ = lean_box(0);
v_lt_1927_ = lean_box(0);
lean_inc_ref(v_min_1925_);
v_semilatticeInf_1928_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1925_, v_le_1926_, v_lt_1927_);
v_toPartialOrder_1929_ = lean_ctor_get(v_semilatticeInf_1928_, 0);
v_isSharedCheck_2030_ = !lean_is_exclusive(v_semilatticeInf_1928_);
if (v_isSharedCheck_2030_ == 0)
{
lean_object* v_unused_2031_; 
v_unused_2031_ = lean_ctor_get(v_semilatticeInf_1928_, 1);
lean_dec(v_unused_2031_);
v___x_1931_ = v_semilatticeInf_1928_;
v_isShared_1932_ = v_isSharedCheck_2030_;
goto v_resetjp_1930_;
}
else
{
lean_inc(v_toPartialOrder_1929_);
lean_dec(v_semilatticeInf_1928_);
v___x_1931_ = lean_box(0);
v_isShared_1932_ = v_isSharedCheck_2030_;
goto v_resetjp_1930_;
}
v_resetjp_1930_:
{
lean_object* v_toLE_1933_; lean_object* v_toLT_1934_; lean_object* v___x_1936_; uint8_t v_isShared_1937_; uint8_t v_isSharedCheck_2029_; 
v_toLE_1933_ = lean_ctor_get(v_toPartialOrder_1929_, 0);
v_toLT_1934_ = lean_ctor_get(v_toPartialOrder_1929_, 1);
v_isSharedCheck_2029_ = !lean_is_exclusive(v_toPartialOrder_1929_);
if (v_isSharedCheck_2029_ == 0)
{
v___x_1936_ = v_toPartialOrder_1929_;
v_isShared_1937_ = v_isSharedCheck_2029_;
goto v_resetjp_1935_;
}
else
{
lean_inc(v_toLT_1934_);
lean_inc(v_toLE_1933_);
lean_dec(v_toPartialOrder_1929_);
v___x_1936_ = lean_box(0);
v_isShared_1937_ = v_isSharedCheck_2029_;
goto v_resetjp_1935_;
}
v_resetjp_1935_:
{
lean_object* v___f_1938_; lean_object* v___f_1939_; lean_object* v___x_1941_; 
v___f_1938_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1938_, 0, v_min_1925_);
lean_inc(v_toFun_1904_);
lean_inc_ref(v_e_1894_);
v___f_1939_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_biheytingAlgebra___redArg___lam__3), 6, 4);
lean_closure_set(v___f_1939_, 0, v_toSemilatticeSup_1919_);
lean_closure_set(v___f_1939_, 1, v___f_1924_);
lean_closure_set(v___f_1939_, 2, v_e_1894_);
lean_closure_set(v___f_1939_, 3, v_toFun_1904_);
if (v_isShared_1937_ == 0)
{
v___x_1941_ = v___x_1936_;
goto v_reusejp_1940_;
}
else
{
lean_object* v_reuseFailAlloc_2028_; 
v_reuseFailAlloc_2028_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2028_, 0, v_toLE_1933_);
lean_ctor_set(v_reuseFailAlloc_2028_, 1, v_toLT_1934_);
v___x_1941_ = v_reuseFailAlloc_2028_;
goto v_reusejp_1940_;
}
v_reusejp_1940_:
{
lean_object* v___x_1943_; 
lean_inc_ref(v___f_1939_);
if (v_isShared_1932_ == 0)
{
lean_ctor_set(v___x_1931_, 1, v___f_1939_);
lean_ctor_set(v___x_1931_, 0, v___x_1941_);
v___x_1943_ = v___x_1931_;
goto v_reusejp_1942_;
}
else
{
lean_object* v_reuseFailAlloc_2027_; 
v_reuseFailAlloc_2027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2027_, 0, v___x_1941_);
lean_ctor_set(v_reuseFailAlloc_2027_, 1, v___f_1939_);
v___x_1943_ = v_reuseFailAlloc_2027_;
goto v_reusejp_1942_;
}
v_reusejp_1942_:
{
lean_object* v_lattice_1945_; 
lean_inc_ref(v___f_1938_);
if (v_isShared_1923_ == 0)
{
lean_ctor_set(v___x_1922_, 1, v___f_1938_);
lean_ctor_set(v___x_1922_, 0, v___x_1943_);
v_lattice_1945_ = v___x_1922_;
goto v_reusejp_1944_;
}
else
{
lean_object* v_reuseFailAlloc_2026_; 
v_reuseFailAlloc_2026_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2026_, 0, v___x_1943_);
lean_ctor_set(v_reuseFailAlloc_2026_, 1, v___f_1938_);
v_lattice_1945_ = v_reuseFailAlloc_2026_;
goto v_reusejp_1944_;
}
v_reusejp_1944_:
{
lean_object* v___x_1946_; lean_object* v_toPartialOrder_1947_; lean_object* v___x_1949_; uint8_t v_isShared_1950_; uint8_t v_isSharedCheck_2024_; 
v___x_1946_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1945_);
v_toPartialOrder_1947_ = lean_ctor_get(v___x_1946_, 0);
v_isSharedCheck_2024_ = !lean_is_exclusive(v___x_1946_);
if (v_isSharedCheck_2024_ == 0)
{
lean_object* v_unused_2025_; 
v_unused_2025_ = lean_ctor_get(v___x_1946_, 1);
lean_dec(v_unused_2025_);
v___x_1949_ = v___x_1946_;
v_isShared_1950_ = v_isSharedCheck_2024_;
goto v_resetjp_1948_;
}
else
{
lean_inc(v_toPartialOrder_1947_);
lean_dec(v___x_1946_);
v___x_1949_ = lean_box(0);
v_isShared_1950_ = v_isSharedCheck_2024_;
goto v_resetjp_1948_;
}
v_resetjp_1948_:
{
lean_object* v_toLE_1951_; lean_object* v_toLT_1952_; lean_object* v___x_1954_; uint8_t v_isShared_1955_; uint8_t v_isSharedCheck_2023_; 
v_toLE_1951_ = lean_ctor_get(v_toPartialOrder_1947_, 0);
v_toLT_1952_ = lean_ctor_get(v_toPartialOrder_1947_, 1);
v_isSharedCheck_2023_ = !lean_is_exclusive(v_toPartialOrder_1947_);
if (v_isSharedCheck_2023_ == 0)
{
v___x_1954_ = v_toPartialOrder_1947_;
v_isShared_1955_ = v_isSharedCheck_2023_;
goto v_resetjp_1953_;
}
else
{
lean_inc(v_toLT_1952_);
lean_inc(v_toLE_1951_);
lean_dec(v_toPartialOrder_1947_);
v___x_1954_ = lean_box(0);
v_isShared_1955_ = v_isSharedCheck_2023_;
goto v_resetjp_1953_;
}
v_resetjp_1953_:
{
lean_object* v_bot_1956_; lean_object* v___f_1957_; lean_object* v___x_1959_; 
lean_inc(v_toFun_1904_);
v_bot_1956_ = lean_apply_1(v_toFun_1904_, v_toOrderBot_1898_);
v___f_1957_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_1957_, 0, v___f_1939_);
if (v_isShared_1955_ == 0)
{
v___x_1959_ = v___x_1954_;
goto v_reusejp_1958_;
}
else
{
lean_object* v_reuseFailAlloc_2022_; 
v_reuseFailAlloc_2022_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2022_, 0, v_toLE_1951_);
lean_ctor_set(v_reuseFailAlloc_2022_, 1, v_toLT_1952_);
v___x_1959_ = v_reuseFailAlloc_2022_;
goto v_reusejp_1958_;
}
v_reusejp_1958_:
{
lean_object* v___x_1961_; 
lean_inc_ref(v___f_1957_);
if (v_isShared_1950_ == 0)
{
lean_ctor_set(v___x_1949_, 1, v___f_1957_);
lean_ctor_set(v___x_1949_, 0, v___x_1959_);
v___x_1961_ = v___x_1949_;
goto v_reusejp_1960_;
}
else
{
lean_object* v_reuseFailAlloc_2021_; 
v_reuseFailAlloc_2021_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2021_, 0, v___x_1959_);
lean_ctor_set(v_reuseFailAlloc_2021_, 1, v___f_1957_);
v___x_1961_ = v_reuseFailAlloc_2021_;
goto v_reusejp_1960_;
}
v_reusejp_1960_:
{
lean_object* v___x_1963_; 
lean_inc_ref(v___f_1938_);
lean_inc_ref(v___x_1961_);
if (v_isShared_1907_ == 0)
{
lean_ctor_set(v___x_1906_, 1, v___f_1938_);
lean_ctor_set(v___x_1906_, 0, v___x_1961_);
v___x_1963_ = v___x_1906_;
goto v_reusejp_1962_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v___x_1961_);
lean_ctor_set(v_reuseFailAlloc_2020_, 1, v___f_1938_);
v___x_1963_ = v_reuseFailAlloc_2020_;
goto v_reusejp_1962_;
}
v_reusejp_1962_:
{
lean_object* v___x_1964_; lean_object* v_toPartialOrder_1965_; lean_object* v_toLE_1966_; lean_object* v_toLT_1967_; lean_object* v___x_1969_; uint8_t v_isShared_1970_; uint8_t v_isSharedCheck_2019_; 
v___x_1964_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_1963_);
v_toPartialOrder_1965_ = lean_ctor_get(v___x_1964_, 0);
lean_inc_ref(v_toPartialOrder_1965_);
v_toLE_1966_ = lean_ctor_get(v_toPartialOrder_1965_, 0);
v_toLT_1967_ = lean_ctor_get(v_toPartialOrder_1965_, 1);
v_isSharedCheck_2019_ = !lean_is_exclusive(v_toPartialOrder_1965_);
if (v_isSharedCheck_2019_ == 0)
{
v___x_1969_ = v_toPartialOrder_1965_;
v_isShared_1970_ = v_isSharedCheck_2019_;
goto v_resetjp_1968_;
}
else
{
lean_inc(v_toLT_1967_);
lean_inc(v_toLE_1966_);
lean_dec(v_toPartialOrder_1965_);
v___x_1969_ = lean_box(0);
v_isShared_1970_ = v_isSharedCheck_2019_;
goto v_resetjp_1968_;
}
v_resetjp_1968_:
{
lean_object* v_bot_1971_; lean_object* v_hnot_1972_; lean_object* v_sdiff_1973_; lean_object* v_top_1974_; lean_object* v___f_1975_; lean_object* v___f_1976_; lean_object* v_coheytingAlgebra_1977_; lean_object* v_toGeneralizedCoheytingAlgebra_1978_; lean_object* v_toLattice_1979_; lean_object* v___x_1981_; uint8_t v_isShared_1982_; uint8_t v_isSharedCheck_2016_; 
lean_inc_n(v_toFun_1904_, 4);
v_bot_1971_ = lean_apply_1(v_toFun_1904_, v_toOrderBot_1917_);
lean_inc_ref_n(v_e_1894_, 2);
v_hnot_1972_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coheytingAlgebra___redArg___lam__8), 4, 3);
lean_closure_set(v_hnot_1972_, 0, v_e_1894_);
lean_closure_set(v_hnot_1972_, 1, v_toHNot_1916_);
lean_closure_set(v_hnot_1972_, 2, v_toFun_1904_);
v_sdiff_1973_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg___lam__8), 6, 4);
lean_closure_set(v_sdiff_1973_, 0, v___f_1924_);
lean_closure_set(v_sdiff_1973_, 1, v_e_1894_);
lean_closure_set(v_sdiff_1973_, 2, v_toSDiff_1918_);
lean_closure_set(v_sdiff_1973_, 3, v_toFun_1904_);
v_top_1974_ = lean_apply_1(v_toFun_1904_, v_toOrderTop_1915_);
v___f_1975_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1975_, 0, v___x_1964_);
v___f_1976_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1976_, 0, v___x_1961_);
lean_inc_ref(v_sdiff_1973_);
lean_inc_ref(v_hnot_1972_);
lean_inc(v_top_1974_);
v_coheytingAlgebra_1977_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_1975_, v___f_1976_, v_toLE_1966_, v_toLT_1967_, v_bot_1971_, v_top_1974_, v_hnot_1972_, v_sdiff_1973_);
v_toGeneralizedCoheytingAlgebra_1978_ = lean_ctor_get(v_coheytingAlgebra_1977_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_1978_);
lean_dec_ref(v_coheytingAlgebra_1977_);
v_toLattice_1979_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1978_, 0);
v_isSharedCheck_2016_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_1978_);
if (v_isSharedCheck_2016_ == 0)
{
lean_object* v_unused_2017_; lean_object* v_unused_2018_; 
v_unused_2017_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1978_, 2);
lean_dec(v_unused_2017_);
v_unused_2018_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_1978_, 1);
lean_dec(v_unused_2018_);
v___x_1981_ = v_toGeneralizedCoheytingAlgebra_1978_;
v_isShared_1982_ = v_isSharedCheck_2016_;
goto v_resetjp_1980_;
}
else
{
lean_inc(v_toLattice_1979_);
lean_dec(v_toGeneralizedCoheytingAlgebra_1978_);
v___x_1981_ = lean_box(0);
v_isShared_1982_ = v_isSharedCheck_2016_;
goto v_resetjp_1980_;
}
v_resetjp_1980_:
{
lean_object* v___x_1983_; lean_object* v_toPartialOrder_1984_; lean_object* v___x_1986_; uint8_t v_isShared_1987_; uint8_t v_isSharedCheck_2014_; 
v___x_1983_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_1979_);
v_toPartialOrder_1984_ = lean_ctor_get(v___x_1983_, 0);
v_isSharedCheck_2014_ = !lean_is_exclusive(v___x_1983_);
if (v_isSharedCheck_2014_ == 0)
{
lean_object* v_unused_2015_; 
v_unused_2015_ = lean_ctor_get(v___x_1983_, 1);
lean_dec(v_unused_2015_);
v___x_1986_ = v___x_1983_;
v_isShared_1987_ = v_isSharedCheck_2014_;
goto v_resetjp_1985_;
}
else
{
lean_inc(v_toPartialOrder_1984_);
lean_dec(v___x_1983_);
v___x_1986_ = lean_box(0);
v_isShared_1987_ = v_isSharedCheck_2014_;
goto v_resetjp_1985_;
}
v_resetjp_1985_:
{
lean_object* v_toLE_1988_; lean_object* v_toLT_1989_; lean_object* v___x_1991_; uint8_t v_isShared_1992_; uint8_t v_isSharedCheck_2013_; 
v_toLE_1988_ = lean_ctor_get(v_toPartialOrder_1984_, 0);
v_toLT_1989_ = lean_ctor_get(v_toPartialOrder_1984_, 1);
v_isSharedCheck_2013_ = !lean_is_exclusive(v_toPartialOrder_1984_);
if (v_isSharedCheck_2013_ == 0)
{
v___x_1991_ = v_toPartialOrder_1984_;
v_isShared_1992_ = v_isSharedCheck_2013_;
goto v_resetjp_1990_;
}
else
{
lean_inc(v_toLT_1989_);
lean_inc(v_toLE_1988_);
lean_dec(v_toPartialOrder_1984_);
v___x_1991_ = lean_box(0);
v_isShared_1992_ = v_isSharedCheck_2013_;
goto v_resetjp_1990_;
}
v_resetjp_1990_:
{
lean_object* v_compl_1993_; lean_object* v_himp_1994_; lean_object* v___x_1996_; 
lean_inc(v_toFun_1904_);
lean_inc_ref(v_e_1894_);
v_compl_1993_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_heytingAlgebra___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_1993_, 0, v_e_1894_);
lean_closure_set(v_compl_1993_, 1, v_toCompl_1899_);
lean_closure_set(v_compl_1993_, 2, v_toFun_1904_);
v_himp_1994_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__3), 6, 4);
lean_closure_set(v_himp_1994_, 0, v___f_1924_);
lean_closure_set(v_himp_1994_, 1, v_e_1894_);
lean_closure_set(v_himp_1994_, 2, v_toHImp_1908_);
lean_closure_set(v_himp_1994_, 3, v_toFun_1904_);
if (v_isShared_1992_ == 0)
{
v___x_1996_ = v___x_1991_;
goto v_reusejp_1995_;
}
else
{
lean_object* v_reuseFailAlloc_2012_; 
v_reuseFailAlloc_2012_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2012_, 0, v_toLE_1988_);
lean_ctor_set(v_reuseFailAlloc_2012_, 1, v_toLT_1989_);
v___x_1996_ = v_reuseFailAlloc_2012_;
goto v_reusejp_1995_;
}
v_reusejp_1995_:
{
lean_object* v___x_1998_; 
if (v_isShared_1987_ == 0)
{
lean_ctor_set(v___x_1986_, 1, v___f_1957_);
lean_ctor_set(v___x_1986_, 0, v___x_1996_);
v___x_1998_ = v___x_1986_;
goto v_reusejp_1997_;
}
else
{
lean_object* v_reuseFailAlloc_2011_; 
v_reuseFailAlloc_2011_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2011_, 0, v___x_1996_);
lean_ctor_set(v_reuseFailAlloc_2011_, 1, v___f_1957_);
v___x_1998_ = v_reuseFailAlloc_2011_;
goto v_reusejp_1997_;
}
v_reusejp_1997_:
{
lean_object* v___x_2000_; 
if (v_isShared_1970_ == 0)
{
lean_ctor_set(v___x_1969_, 1, v___f_1938_);
lean_ctor_set(v___x_1969_, 0, v___x_1998_);
v___x_2000_ = v___x_1969_;
goto v_reusejp_1999_;
}
else
{
lean_object* v_reuseFailAlloc_2010_; 
v_reuseFailAlloc_2010_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2010_, 0, v___x_1998_);
lean_ctor_set(v_reuseFailAlloc_2010_, 1, v___f_1938_);
v___x_2000_ = v_reuseFailAlloc_2010_;
goto v_reusejp_1999_;
}
v_reusejp_1999_:
{
lean_object* v___x_2002_; 
if (v_isShared_1911_ == 0)
{
lean_ctor_set(v___x_1910_, 2, v_himp_1994_);
lean_ctor_set(v___x_1910_, 1, v_top_1974_);
lean_ctor_set(v___x_1910_, 0, v___x_2000_);
v___x_2002_ = v___x_1910_;
goto v_reusejp_2001_;
}
else
{
lean_object* v_reuseFailAlloc_2009_; 
v_reuseFailAlloc_2009_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2009_, 0, v___x_2000_);
lean_ctor_set(v_reuseFailAlloc_2009_, 1, v_top_1974_);
lean_ctor_set(v_reuseFailAlloc_2009_, 2, v_himp_1994_);
v___x_2002_ = v_reuseFailAlloc_2009_;
goto v_reusejp_2001_;
}
v_reusejp_2001_:
{
lean_object* v___x_2004_; 
if (v_isShared_1902_ == 0)
{
lean_ctor_set(v___x_1901_, 2, v_compl_1993_);
lean_ctor_set(v___x_1901_, 1, v_bot_1956_);
lean_ctor_set(v___x_1901_, 0, v___x_2002_);
v___x_2004_ = v___x_1901_;
goto v_reusejp_2003_;
}
else
{
lean_object* v_reuseFailAlloc_2008_; 
v_reuseFailAlloc_2008_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2008_, 0, v___x_2002_);
lean_ctor_set(v_reuseFailAlloc_2008_, 1, v_bot_1956_);
lean_ctor_set(v_reuseFailAlloc_2008_, 2, v_compl_1993_);
v___x_2004_ = v_reuseFailAlloc_2008_;
goto v_reusejp_2003_;
}
v_reusejp_2003_:
{
lean_object* v___x_2006_; 
if (v_isShared_1982_ == 0)
{
lean_ctor_set(v___x_1981_, 2, v_hnot_1972_);
lean_ctor_set(v___x_1981_, 1, v_sdiff_1973_);
lean_ctor_set(v___x_1981_, 0, v___x_2004_);
v___x_2006_ = v___x_1981_;
goto v_reusejp_2005_;
}
else
{
lean_object* v_reuseFailAlloc_2007_; 
v_reuseFailAlloc_2007_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2007_, 0, v___x_2004_);
lean_ctor_set(v_reuseFailAlloc_2007_, 1, v_sdiff_1973_);
lean_ctor_set(v_reuseFailAlloc_2007_, 2, v_hnot_1972_);
v___x_2006_ = v_reuseFailAlloc_2007_;
goto v_reusejp_2005_;
}
v_reusejp_2005_:
{
return v___x_2006_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_biheytingAlgebra(lean_object* v_00_u03b1_2039_, lean_object* v_00_u03b2_2040_, lean_object* v_e_2041_, lean_object* v_inst_2042_){
_start:
{
lean_object* v_toHeytingAlgebra_2043_; lean_object* v_toGeneralizedHeytingAlgebra_2044_; lean_object* v_toOrderBot_2045_; lean_object* v_toCompl_2046_; lean_object* v___x_2048_; uint8_t v_isShared_2049_; uint8_t v_isSharedCheck_2185_; 
v_toHeytingAlgebra_2043_ = lean_ctor_get(v_inst_2042_, 0);
lean_inc_ref(v_toHeytingAlgebra_2043_);
v_toGeneralizedHeytingAlgebra_2044_ = lean_ctor_get(v_toHeytingAlgebra_2043_, 0);
v_toOrderBot_2045_ = lean_ctor_get(v_toHeytingAlgebra_2043_, 1);
v_toCompl_2046_ = lean_ctor_get(v_toHeytingAlgebra_2043_, 2);
v_isSharedCheck_2185_ = !lean_is_exclusive(v_toHeytingAlgebra_2043_);
if (v_isSharedCheck_2185_ == 0)
{
v___x_2048_ = v_toHeytingAlgebra_2043_;
v_isShared_2049_ = v_isSharedCheck_2185_;
goto v_resetjp_2047_;
}
else
{
lean_inc(v_toCompl_2046_);
lean_inc(v_toOrderBot_2045_);
lean_inc(v_toGeneralizedHeytingAlgebra_2044_);
lean_dec(v_toHeytingAlgebra_2043_);
v___x_2048_ = lean_box(0);
v_isShared_2049_ = v_isSharedCheck_2185_;
goto v_resetjp_2047_;
}
v_resetjp_2047_:
{
lean_object* v___x_2050_; lean_object* v_toFun_2051_; lean_object* v___x_2053_; uint8_t v_isShared_2054_; uint8_t v_isSharedCheck_2183_; 
lean_inc_ref(v_e_2041_);
v___x_2050_ = lp_mathlib_Equiv_symm___redArg(v_e_2041_);
v_toFun_2051_ = lean_ctor_get(v___x_2050_, 0);
v_isSharedCheck_2183_ = !lean_is_exclusive(v___x_2050_);
if (v_isSharedCheck_2183_ == 0)
{
lean_object* v_unused_2184_; 
v_unused_2184_ = lean_ctor_get(v___x_2050_, 1);
lean_dec(v_unused_2184_);
v___x_2053_ = v___x_2050_;
v_isShared_2054_ = v_isSharedCheck_2183_;
goto v_resetjp_2052_;
}
else
{
lean_inc(v_toFun_2051_);
lean_dec(v___x_2050_);
v___x_2053_ = lean_box(0);
v_isShared_2054_ = v_isSharedCheck_2183_;
goto v_resetjp_2052_;
}
v_resetjp_2052_:
{
lean_object* v_toHImp_2055_; lean_object* v___x_2057_; uint8_t v_isShared_2058_; uint8_t v_isSharedCheck_2180_; 
v_toHImp_2055_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2044_, 2);
v_isSharedCheck_2180_ = !lean_is_exclusive(v_toGeneralizedHeytingAlgebra_2044_);
if (v_isSharedCheck_2180_ == 0)
{
lean_object* v_unused_2181_; lean_object* v_unused_2182_; 
v_unused_2181_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2044_, 1);
lean_dec(v_unused_2181_);
v_unused_2182_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_2044_, 0);
lean_dec(v_unused_2182_);
v___x_2057_ = v_toGeneralizedHeytingAlgebra_2044_;
v_isShared_2058_ = v_isSharedCheck_2180_;
goto v_resetjp_2056_;
}
else
{
lean_inc(v_toHImp_2055_);
lean_dec(v_toGeneralizedHeytingAlgebra_2044_);
v___x_2057_ = lean_box(0);
v_isShared_2058_ = v_isSharedCheck_2180_;
goto v_resetjp_2056_;
}
v_resetjp_2056_:
{
lean_object* v___x_2059_; lean_object* v_toGeneralizedCoheytingAlgebra_2060_; lean_object* v_toLattice_2061_; lean_object* v_toOrderTop_2062_; lean_object* v_toHNot_2063_; lean_object* v_toOrderBot_2064_; lean_object* v_toSDiff_2065_; lean_object* v_toSemilatticeSup_2066_; lean_object* v_inf_2067_; lean_object* v___x_2069_; uint8_t v_isShared_2070_; uint8_t v_isSharedCheck_2179_; 
v___x_2059_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v_inst_2042_);
v_toGeneralizedCoheytingAlgebra_2060_ = lean_ctor_get(v___x_2059_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2060_);
v_toLattice_2061_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2060_, 0);
lean_inc_ref(v_toLattice_2061_);
v_toOrderTop_2062_ = lean_ctor_get(v___x_2059_, 1);
lean_inc(v_toOrderTop_2062_);
v_toHNot_2063_ = lean_ctor_get(v___x_2059_, 2);
lean_inc(v_toHNot_2063_);
lean_dec_ref(v___x_2059_);
v_toOrderBot_2064_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2060_, 1);
lean_inc(v_toOrderBot_2064_);
v_toSDiff_2065_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2060_, 2);
lean_inc(v_toSDiff_2065_);
lean_dec_ref(v_toGeneralizedCoheytingAlgebra_2060_);
v_toSemilatticeSup_2066_ = lean_ctor_get(v_toLattice_2061_, 0);
v_inf_2067_ = lean_ctor_get(v_toLattice_2061_, 1);
v_isSharedCheck_2179_ = !lean_is_exclusive(v_toLattice_2061_);
if (v_isSharedCheck_2179_ == 0)
{
v___x_2069_ = v_toLattice_2061_;
v_isShared_2070_ = v_isSharedCheck_2179_;
goto v_resetjp_2068_;
}
else
{
lean_inc(v_inf_2067_);
lean_inc(v_toSemilatticeSup_2066_);
lean_dec(v_toLattice_2061_);
v___x_2069_ = lean_box(0);
v_isShared_2070_ = v_isSharedCheck_2179_;
goto v_resetjp_2068_;
}
v_resetjp_2068_:
{
lean_object* v___f_2071_; lean_object* v_min_2072_; lean_object* v_le_2073_; lean_object* v_lt_2074_; lean_object* v_semilatticeInf_2075_; lean_object* v_toPartialOrder_2076_; lean_object* v___x_2078_; uint8_t v_isShared_2079_; uint8_t v_isSharedCheck_2177_; 
v___f_2071_ = ((lean_object*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___closed__0));
lean_inc(v_toFun_2051_);
lean_inc_ref(v_e_2041_);
v_min_2072_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_biheytingAlgebra___redArg___lam__1), 6, 4);
lean_closure_set(v_min_2072_, 0, v___f_2071_);
lean_closure_set(v_min_2072_, 1, v_e_2041_);
lean_closure_set(v_min_2072_, 2, v_inf_2067_);
lean_closure_set(v_min_2072_, 3, v_toFun_2051_);
v_le_2073_ = lean_box(0);
v_lt_2074_ = lean_box(0);
lean_inc_ref(v_min_2072_);
v_semilatticeInf_2075_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_2072_, v_le_2073_, v_lt_2074_);
v_toPartialOrder_2076_ = lean_ctor_get(v_semilatticeInf_2075_, 0);
v_isSharedCheck_2177_ = !lean_is_exclusive(v_semilatticeInf_2075_);
if (v_isSharedCheck_2177_ == 0)
{
lean_object* v_unused_2178_; 
v_unused_2178_ = lean_ctor_get(v_semilatticeInf_2075_, 1);
lean_dec(v_unused_2178_);
v___x_2078_ = v_semilatticeInf_2075_;
v_isShared_2079_ = v_isSharedCheck_2177_;
goto v_resetjp_2077_;
}
else
{
lean_inc(v_toPartialOrder_2076_);
lean_dec(v_semilatticeInf_2075_);
v___x_2078_ = lean_box(0);
v_isShared_2079_ = v_isSharedCheck_2177_;
goto v_resetjp_2077_;
}
v_resetjp_2077_:
{
lean_object* v_toLE_2080_; lean_object* v_toLT_2081_; lean_object* v___x_2083_; uint8_t v_isShared_2084_; uint8_t v_isSharedCheck_2176_; 
v_toLE_2080_ = lean_ctor_get(v_toPartialOrder_2076_, 0);
v_toLT_2081_ = lean_ctor_get(v_toPartialOrder_2076_, 1);
v_isSharedCheck_2176_ = !lean_is_exclusive(v_toPartialOrder_2076_);
if (v_isSharedCheck_2176_ == 0)
{
v___x_2083_ = v_toPartialOrder_2076_;
v_isShared_2084_ = v_isSharedCheck_2176_;
goto v_resetjp_2082_;
}
else
{
lean_inc(v_toLT_2081_);
lean_inc(v_toLE_2080_);
lean_dec(v_toPartialOrder_2076_);
v___x_2083_ = lean_box(0);
v_isShared_2084_ = v_isSharedCheck_2176_;
goto v_resetjp_2082_;
}
v_resetjp_2082_:
{
lean_object* v___f_2085_; lean_object* v___f_2086_; lean_object* v___x_2088_; 
v___f_2085_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(v___f_2085_, 0, v_min_2072_);
lean_inc(v_toFun_2051_);
lean_inc_ref(v_e_2041_);
v___f_2086_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_biheytingAlgebra___redArg___lam__3), 6, 4);
lean_closure_set(v___f_2086_, 0, v_toSemilatticeSup_2066_);
lean_closure_set(v___f_2086_, 1, v___f_2071_);
lean_closure_set(v___f_2086_, 2, v_e_2041_);
lean_closure_set(v___f_2086_, 3, v_toFun_2051_);
if (v_isShared_2084_ == 0)
{
v___x_2088_ = v___x_2083_;
goto v_reusejp_2087_;
}
else
{
lean_object* v_reuseFailAlloc_2175_; 
v_reuseFailAlloc_2175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2175_, 0, v_toLE_2080_);
lean_ctor_set(v_reuseFailAlloc_2175_, 1, v_toLT_2081_);
v___x_2088_ = v_reuseFailAlloc_2175_;
goto v_reusejp_2087_;
}
v_reusejp_2087_:
{
lean_object* v___x_2090_; 
lean_inc_ref(v___f_2086_);
if (v_isShared_2079_ == 0)
{
lean_ctor_set(v___x_2078_, 1, v___f_2086_);
lean_ctor_set(v___x_2078_, 0, v___x_2088_);
v___x_2090_ = v___x_2078_;
goto v_reusejp_2089_;
}
else
{
lean_object* v_reuseFailAlloc_2174_; 
v_reuseFailAlloc_2174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2174_, 0, v___x_2088_);
lean_ctor_set(v_reuseFailAlloc_2174_, 1, v___f_2086_);
v___x_2090_ = v_reuseFailAlloc_2174_;
goto v_reusejp_2089_;
}
v_reusejp_2089_:
{
lean_object* v_lattice_2092_; 
lean_inc_ref(v___f_2085_);
if (v_isShared_2070_ == 0)
{
lean_ctor_set(v___x_2069_, 1, v___f_2085_);
lean_ctor_set(v___x_2069_, 0, v___x_2090_);
v_lattice_2092_ = v___x_2069_;
goto v_reusejp_2091_;
}
else
{
lean_object* v_reuseFailAlloc_2173_; 
v_reuseFailAlloc_2173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2173_, 0, v___x_2090_);
lean_ctor_set(v_reuseFailAlloc_2173_, 1, v___f_2085_);
v_lattice_2092_ = v_reuseFailAlloc_2173_;
goto v_reusejp_2091_;
}
v_reusejp_2091_:
{
lean_object* v___x_2093_; lean_object* v_toPartialOrder_2094_; lean_object* v___x_2096_; uint8_t v_isShared_2097_; uint8_t v_isSharedCheck_2171_; 
v___x_2093_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_2092_);
v_toPartialOrder_2094_ = lean_ctor_get(v___x_2093_, 0);
v_isSharedCheck_2171_ = !lean_is_exclusive(v___x_2093_);
if (v_isSharedCheck_2171_ == 0)
{
lean_object* v_unused_2172_; 
v_unused_2172_ = lean_ctor_get(v___x_2093_, 1);
lean_dec(v_unused_2172_);
v___x_2096_ = v___x_2093_;
v_isShared_2097_ = v_isSharedCheck_2171_;
goto v_resetjp_2095_;
}
else
{
lean_inc(v_toPartialOrder_2094_);
lean_dec(v___x_2093_);
v___x_2096_ = lean_box(0);
v_isShared_2097_ = v_isSharedCheck_2171_;
goto v_resetjp_2095_;
}
v_resetjp_2095_:
{
lean_object* v_toLE_2098_; lean_object* v_toLT_2099_; lean_object* v___x_2101_; uint8_t v_isShared_2102_; uint8_t v_isSharedCheck_2170_; 
v_toLE_2098_ = lean_ctor_get(v_toPartialOrder_2094_, 0);
v_toLT_2099_ = lean_ctor_get(v_toPartialOrder_2094_, 1);
v_isSharedCheck_2170_ = !lean_is_exclusive(v_toPartialOrder_2094_);
if (v_isSharedCheck_2170_ == 0)
{
v___x_2101_ = v_toPartialOrder_2094_;
v_isShared_2102_ = v_isSharedCheck_2170_;
goto v_resetjp_2100_;
}
else
{
lean_inc(v_toLT_2099_);
lean_inc(v_toLE_2098_);
lean_dec(v_toPartialOrder_2094_);
v___x_2101_ = lean_box(0);
v_isShared_2102_ = v_isSharedCheck_2170_;
goto v_resetjp_2100_;
}
v_resetjp_2100_:
{
lean_object* v_bot_2103_; lean_object* v___f_2104_; lean_object* v___x_2106_; 
lean_inc(v_toFun_2051_);
v_bot_2103_ = lean_apply_1(v_toFun_2051_, v_toOrderBot_2045_);
v___f_2104_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__5), 3, 1);
lean_closure_set(v___f_2104_, 0, v___f_2086_);
if (v_isShared_2102_ == 0)
{
v___x_2106_ = v___x_2101_;
goto v_reusejp_2105_;
}
else
{
lean_object* v_reuseFailAlloc_2169_; 
v_reuseFailAlloc_2169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2169_, 0, v_toLE_2098_);
lean_ctor_set(v_reuseFailAlloc_2169_, 1, v_toLT_2099_);
v___x_2106_ = v_reuseFailAlloc_2169_;
goto v_reusejp_2105_;
}
v_reusejp_2105_:
{
lean_object* v___x_2108_; 
lean_inc_ref(v___f_2104_);
if (v_isShared_2097_ == 0)
{
lean_ctor_set(v___x_2096_, 1, v___f_2104_);
lean_ctor_set(v___x_2096_, 0, v___x_2106_);
v___x_2108_ = v___x_2096_;
goto v_reusejp_2107_;
}
else
{
lean_object* v_reuseFailAlloc_2168_; 
v_reuseFailAlloc_2168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2168_, 0, v___x_2106_);
lean_ctor_set(v_reuseFailAlloc_2168_, 1, v___f_2104_);
v___x_2108_ = v_reuseFailAlloc_2168_;
goto v_reusejp_2107_;
}
v_reusejp_2107_:
{
lean_object* v___x_2110_; 
lean_inc_ref(v___f_2085_);
lean_inc_ref(v___x_2108_);
if (v_isShared_2054_ == 0)
{
lean_ctor_set(v___x_2053_, 1, v___f_2085_);
lean_ctor_set(v___x_2053_, 0, v___x_2108_);
v___x_2110_ = v___x_2053_;
goto v_reusejp_2109_;
}
else
{
lean_object* v_reuseFailAlloc_2167_; 
v_reuseFailAlloc_2167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2167_, 0, v___x_2108_);
lean_ctor_set(v_reuseFailAlloc_2167_, 1, v___f_2085_);
v___x_2110_ = v_reuseFailAlloc_2167_;
goto v_reusejp_2109_;
}
v_reusejp_2109_:
{
lean_object* v___x_2111_; lean_object* v_toPartialOrder_2112_; lean_object* v_toLE_2113_; lean_object* v_toLT_2114_; lean_object* v___x_2116_; uint8_t v_isShared_2117_; uint8_t v_isSharedCheck_2166_; 
v___x_2111_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_2110_);
v_toPartialOrder_2112_ = lean_ctor_get(v___x_2111_, 0);
lean_inc_ref(v_toPartialOrder_2112_);
v_toLE_2113_ = lean_ctor_get(v_toPartialOrder_2112_, 0);
v_toLT_2114_ = lean_ctor_get(v_toPartialOrder_2112_, 1);
v_isSharedCheck_2166_ = !lean_is_exclusive(v_toPartialOrder_2112_);
if (v_isSharedCheck_2166_ == 0)
{
v___x_2116_ = v_toPartialOrder_2112_;
v_isShared_2117_ = v_isSharedCheck_2166_;
goto v_resetjp_2115_;
}
else
{
lean_inc(v_toLT_2114_);
lean_inc(v_toLE_2113_);
lean_dec(v_toPartialOrder_2112_);
v___x_2116_ = lean_box(0);
v_isShared_2117_ = v_isSharedCheck_2166_;
goto v_resetjp_2115_;
}
v_resetjp_2115_:
{
lean_object* v_bot_2118_; lean_object* v_hnot_2119_; lean_object* v_sdiff_2120_; lean_object* v_top_2121_; lean_object* v___f_2122_; lean_object* v___f_2123_; lean_object* v_coheytingAlgebra_2124_; lean_object* v_toGeneralizedCoheytingAlgebra_2125_; lean_object* v_toLattice_2126_; lean_object* v___x_2128_; uint8_t v_isShared_2129_; uint8_t v_isSharedCheck_2163_; 
lean_inc_n(v_toFun_2051_, 4);
v_bot_2118_ = lean_apply_1(v_toFun_2051_, v_toOrderBot_2064_);
lean_inc_ref_n(v_e_2041_, 2);
v_hnot_2119_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_coheytingAlgebra___redArg___lam__8), 4, 3);
lean_closure_set(v_hnot_2119_, 0, v_e_2041_);
lean_closure_set(v_hnot_2119_, 1, v_toHNot_2063_);
lean_closure_set(v_hnot_2119_, 2, v_toFun_2051_);
v_sdiff_2120_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedCoheytingAlgebra___redArg___lam__8), 6, 4);
lean_closure_set(v_sdiff_2120_, 0, v___f_2071_);
lean_closure_set(v_sdiff_2120_, 1, v_e_2041_);
lean_closure_set(v_sdiff_2120_, 2, v_toSDiff_2065_);
lean_closure_set(v_sdiff_2120_, 3, v_toFun_2051_);
v_top_2121_ = lean_apply_1(v_toFun_2051_, v_toOrderTop_2062_);
v___f_2122_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2122_, 0, v___x_2111_);
v___f_2123_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2123_, 0, v___x_2108_);
lean_inc_ref(v_sdiff_2120_);
lean_inc_ref(v_hnot_2119_);
lean_inc(v_top_2121_);
v_coheytingAlgebra_2124_ = lp_mathlib_Function_Injective_coheytingAlgebra___redArg(v___f_2122_, v___f_2123_, v_toLE_2113_, v_toLT_2114_, v_bot_2118_, v_top_2121_, v_hnot_2119_, v_sdiff_2120_);
v_toGeneralizedCoheytingAlgebra_2125_ = lean_ctor_get(v_coheytingAlgebra_2124_, 0);
lean_inc_ref(v_toGeneralizedCoheytingAlgebra_2125_);
lean_dec_ref(v_coheytingAlgebra_2124_);
v_toLattice_2126_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2125_, 0);
v_isSharedCheck_2163_ = !lean_is_exclusive(v_toGeneralizedCoheytingAlgebra_2125_);
if (v_isSharedCheck_2163_ == 0)
{
lean_object* v_unused_2164_; lean_object* v_unused_2165_; 
v_unused_2164_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2125_, 2);
lean_dec(v_unused_2164_);
v_unused_2165_ = lean_ctor_get(v_toGeneralizedCoheytingAlgebra_2125_, 1);
lean_dec(v_unused_2165_);
v___x_2128_ = v_toGeneralizedCoheytingAlgebra_2125_;
v_isShared_2129_ = v_isSharedCheck_2163_;
goto v_resetjp_2127_;
}
else
{
lean_inc(v_toLattice_2126_);
lean_dec(v_toGeneralizedCoheytingAlgebra_2125_);
v___x_2128_ = lean_box(0);
v_isShared_2129_ = v_isSharedCheck_2163_;
goto v_resetjp_2127_;
}
v_resetjp_2127_:
{
lean_object* v___x_2130_; lean_object* v_toPartialOrder_2131_; lean_object* v___x_2133_; uint8_t v_isShared_2134_; uint8_t v_isSharedCheck_2161_; 
v___x_2130_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_toLattice_2126_);
v_toPartialOrder_2131_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2161_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2161_ == 0)
{
lean_object* v_unused_2162_; 
v_unused_2162_ = lean_ctor_get(v___x_2130_, 1);
lean_dec(v_unused_2162_);
v___x_2133_ = v___x_2130_;
v_isShared_2134_ = v_isSharedCheck_2161_;
goto v_resetjp_2132_;
}
else
{
lean_inc(v_toPartialOrder_2131_);
lean_dec(v___x_2130_);
v___x_2133_ = lean_box(0);
v_isShared_2134_ = v_isSharedCheck_2161_;
goto v_resetjp_2132_;
}
v_resetjp_2132_:
{
lean_object* v_toLE_2135_; lean_object* v_toLT_2136_; lean_object* v___x_2138_; uint8_t v_isShared_2139_; uint8_t v_isSharedCheck_2160_; 
v_toLE_2135_ = lean_ctor_get(v_toPartialOrder_2131_, 0);
v_toLT_2136_ = lean_ctor_get(v_toPartialOrder_2131_, 1);
v_isSharedCheck_2160_ = !lean_is_exclusive(v_toPartialOrder_2131_);
if (v_isSharedCheck_2160_ == 0)
{
v___x_2138_ = v_toPartialOrder_2131_;
v_isShared_2139_ = v_isSharedCheck_2160_;
goto v_resetjp_2137_;
}
else
{
lean_inc(v_toLT_2136_);
lean_inc(v_toLE_2135_);
lean_dec(v_toPartialOrder_2131_);
v___x_2138_ = lean_box(0);
v_isShared_2139_ = v_isSharedCheck_2160_;
goto v_resetjp_2137_;
}
v_resetjp_2137_:
{
lean_object* v_compl_2140_; lean_object* v_himp_2141_; lean_object* v___x_2143_; 
lean_inc(v_toFun_2051_);
lean_inc_ref(v_e_2041_);
v_compl_2140_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_heytingAlgebra___redArg___lam__10), 4, 3);
lean_closure_set(v_compl_2140_, 0, v_e_2041_);
lean_closure_set(v_compl_2140_, 1, v_toCompl_2046_);
lean_closure_set(v_compl_2140_, 2, v_toFun_2051_);
v_himp_2141_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_generalizedHeytingAlgebra___redArg___lam__3), 6, 4);
lean_closure_set(v_himp_2141_, 0, v___f_2071_);
lean_closure_set(v_himp_2141_, 1, v_e_2041_);
lean_closure_set(v_himp_2141_, 2, v_toHImp_2055_);
lean_closure_set(v_himp_2141_, 3, v_toFun_2051_);
if (v_isShared_2139_ == 0)
{
v___x_2143_ = v___x_2138_;
goto v_reusejp_2142_;
}
else
{
lean_object* v_reuseFailAlloc_2159_; 
v_reuseFailAlloc_2159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2159_, 0, v_toLE_2135_);
lean_ctor_set(v_reuseFailAlloc_2159_, 1, v_toLT_2136_);
v___x_2143_ = v_reuseFailAlloc_2159_;
goto v_reusejp_2142_;
}
v_reusejp_2142_:
{
lean_object* v___x_2145_; 
if (v_isShared_2134_ == 0)
{
lean_ctor_set(v___x_2133_, 1, v___f_2104_);
lean_ctor_set(v___x_2133_, 0, v___x_2143_);
v___x_2145_ = v___x_2133_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2158_; 
v_reuseFailAlloc_2158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2158_, 0, v___x_2143_);
lean_ctor_set(v_reuseFailAlloc_2158_, 1, v___f_2104_);
v___x_2145_ = v_reuseFailAlloc_2158_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
lean_object* v___x_2147_; 
if (v_isShared_2117_ == 0)
{
lean_ctor_set(v___x_2116_, 1, v___f_2085_);
lean_ctor_set(v___x_2116_, 0, v___x_2145_);
v___x_2147_ = v___x_2116_;
goto v_reusejp_2146_;
}
else
{
lean_object* v_reuseFailAlloc_2157_; 
v_reuseFailAlloc_2157_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2157_, 0, v___x_2145_);
lean_ctor_set(v_reuseFailAlloc_2157_, 1, v___f_2085_);
v___x_2147_ = v_reuseFailAlloc_2157_;
goto v_reusejp_2146_;
}
v_reusejp_2146_:
{
lean_object* v___x_2149_; 
if (v_isShared_2058_ == 0)
{
lean_ctor_set(v___x_2057_, 2, v_himp_2141_);
lean_ctor_set(v___x_2057_, 1, v_top_2121_);
lean_ctor_set(v___x_2057_, 0, v___x_2147_);
v___x_2149_ = v___x_2057_;
goto v_reusejp_2148_;
}
else
{
lean_object* v_reuseFailAlloc_2156_; 
v_reuseFailAlloc_2156_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2156_, 0, v___x_2147_);
lean_ctor_set(v_reuseFailAlloc_2156_, 1, v_top_2121_);
lean_ctor_set(v_reuseFailAlloc_2156_, 2, v_himp_2141_);
v___x_2149_ = v_reuseFailAlloc_2156_;
goto v_reusejp_2148_;
}
v_reusejp_2148_:
{
lean_object* v___x_2151_; 
if (v_isShared_2049_ == 0)
{
lean_ctor_set(v___x_2048_, 2, v_compl_2140_);
lean_ctor_set(v___x_2048_, 1, v_bot_2103_);
lean_ctor_set(v___x_2048_, 0, v___x_2149_);
v___x_2151_ = v___x_2048_;
goto v_reusejp_2150_;
}
else
{
lean_object* v_reuseFailAlloc_2155_; 
v_reuseFailAlloc_2155_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2155_, 0, v___x_2149_);
lean_ctor_set(v_reuseFailAlloc_2155_, 1, v_bot_2103_);
lean_ctor_set(v_reuseFailAlloc_2155_, 2, v_compl_2140_);
v___x_2151_ = v_reuseFailAlloc_2155_;
goto v_reusejp_2150_;
}
v_reusejp_2150_:
{
lean_object* v___x_2153_; 
if (v_isShared_2129_ == 0)
{
lean_ctor_set(v___x_2128_, 2, v_hnot_2119_);
lean_ctor_set(v___x_2128_, 1, v_sdiff_2120_);
lean_ctor_set(v___x_2128_, 0, v___x_2151_);
v___x_2153_ = v___x_2128_;
goto v_reusejp_2152_;
}
else
{
lean_object* v_reuseFailAlloc_2154_; 
v_reuseFailAlloc_2154_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2154_, 0, v___x_2151_);
lean_ctor_set(v_reuseFailAlloc_2154_, 1, v_sdiff_2120_);
lean_ctor_set(v_reuseFailAlloc_2154_, 2, v_hnot_2119_);
v___x_2153_ = v_reuseFailAlloc_2154_;
goto v_reusejp_2152_;
}
v_reusejp_2152_:
{
return v___x_2153_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___lam__0(lean_object* v_x_2186_, lean_object* v_x_2187_){
_start:
{
lean_object* v___x_2188_; 
v___x_2188_ = lean_box(0);
return v___x_2188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___lam__1(lean_object* v___x_2189_, lean_object* v_x_2190_){
_start:
{
return v___x_2189_;
}
}
static lean_object* _init_lp_mathlib_PUnit_instBiheytingAlgebra(void){
_start:
{
lean_object* v___x_2194_; lean_object* v_toPartialOrder_2195_; lean_object* v___f_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___f_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; 
v___x_2194_ = lp_mathlib_PUnit_instLinearOrder;
v_toPartialOrder_2195_ = lean_ctor_get(v___x_2194_, 0);
v___f_2196_ = ((lean_object*)(lp_mathlib_PUnit_instBiheytingAlgebra___closed__0));
lean_inc_ref(v_toPartialOrder_2195_);
v___x_2197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2197_, 0, v_toPartialOrder_2195_);
lean_ctor_set(v___x_2197_, 1, v___f_2196_);
v___x_2198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2198_, 0, v___x_2197_);
lean_ctor_set(v___x_2198_, 1, v___f_2196_);
v___x_2199_ = lean_box(0);
v___f_2200_ = ((lean_object*)(lp_mathlib_PUnit_instBiheytingAlgebra___closed__1));
v___x_2201_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2201_, 0, v___x_2198_);
lean_ctor_set(v___x_2201_, 1, v___x_2199_);
lean_ctor_set(v___x_2201_, 2, v___f_2196_);
v___x_2202_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2202_, 0, v___x_2201_);
lean_ctor_set(v___x_2202_, 1, v___x_2199_);
lean_ctor_set(v___x_2202_, 2, v___f_2200_);
v___x_2203_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2203_, 0, v___x_2202_);
lean_ctor_set(v___x_2203_, 1, v___f_2196_);
lean_ctor_set(v___x_2203_, 2, v___f_2200_);
return v___x_2203_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_PropInstances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_PropInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Prop_instHeytingAlgebra = _init_lp_mathlib_Prop_instHeytingAlgebra();
lean_mark_persistent(lp_mathlib_Prop_instHeytingAlgebra);
lp_mathlib_PUnit_instBiheytingAlgebra = _init_lp_mathlib_PUnit_instBiheytingAlgebra();
lean_mark_persistent(lp_mathlib_PUnit_instBiheytingAlgebra);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_PropInstances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_PropInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
