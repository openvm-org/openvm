// Lean compiler output
// Module: Mathlib.Order.Lattice
// Imports: public import Init public meta import Init public import Mathlib.Data.Bool.Basic public import Mathlib.Logic.Pairwise public import Mathlib.Order.Monotone.Basic public import Mathlib.Order.ULift import Mathlib.Tactic.GRewrite
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Pi_partialOrder___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instPreorder(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ULift_instMax__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_preorder(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Bool_linearOrder;
extern lean_object* lp_mathlib_Nat_instLinearOrder;
extern lean_object* lp_mathlib_Int_instLinearOrder;
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toMax___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toMax___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toMax(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_mk_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeSup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toSemilatticeInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_mkDual___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_mkDual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_mk_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribLattice_ofInfSupLe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribLattice_ofInfSupLe___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribLattice_ofInfSupLe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribLattice_ofInfSupLe___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lattice_toLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toLinearOrder___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toLinearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeOfLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeOfLinearOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeOfLinearOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeOfLinearOrder___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instDistribLatticeNat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instDistribLatticeNat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeNat;
static lean_once_cell_t lp_mathlib_instLatticeInt___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLatticeInt___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instLatticeInt;
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMaxForall__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMaxForall__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMaxForall__mathlib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMinForall__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMinForall__mathlib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeSup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeSup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeInf___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeInf___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeInf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instDistribLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instDistribLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMax__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMax__mathlib___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMax__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMin__mathlib___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMin__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDistribLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_lattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_lattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_lattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeSup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_lattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_lattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_lattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_distribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_distribLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_preorder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_preorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_partialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_partialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_linearOrder___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_linearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_linearOrder___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_linearOrder___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_linearOrder___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_linearOrder___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeSup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeInf___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribLattice___redArg___lam__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSemilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSemilatticeSup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSemilatticeInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ULift_instLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ULift_instLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ULift_instLinearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Bool_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Bool_instPartialOrder___closed__0;
static lean_once_cell_t lp_mathlib_Bool_instPartialOrder___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Bool_instPartialOrder___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Bool_instPartialOrder;
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDistribLattice;
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toMax___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
lean_object* v_sup_4_; lean_object* v___x_5_; 
v_sup_4_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_sup_4_);
lean_dec_ref(v_inst_1_);
v___x_5_ = lean_apply_2(v_sup_4_, v_a_2_, v_b_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toMax___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v_inst_6_);
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toMax(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toMin___redArg___lam__0(lean_object* v_inst_11_, lean_object* v_a_12_, lean_object* v_b_13_){
_start:
{
lean_object* v_inf_14_; lean_object* v___x_15_; 
v_inf_14_ = lean_ctor_get(v_inst_11_, 1);
lean_inc(v_inf_14_);
lean_dec_ref(v_inst_11_);
v___x_15_ = lean_apply_2(v_inf_14_, v_a_12_, v_b_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toMin___redArg(lean_object* v_inst_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_17_, 0, v_inst_16_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toMin(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_mk_x27___redArg___lam__0(lean_object* v_inst_21_, lean_object* v_x1_22_, lean_object* v_x2_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_apply_2(v_inst_21_, v_x1_22_, v_x2_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_mk_x27___redArg(lean_object* v_inst_28_){
_start:
{
lean_object* v___f_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_29_, 0, v_inst_28_);
v___x_30_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
v___x_31_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_31_, 0, v___x_30_);
lean_ctor_set(v___x_31_, 1, v___f_29_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_mk_x27(lean_object* v_00_u03b1_32_, lean_object* v_inst_33_, lean_object* v_sup__comm_34_, lean_object* v_sup__assoc_35_, lean_object* v_sup__idem_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_SemilatticeSup_mk_x27___redArg(v_inst_33_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_mk_x27___redArg(lean_object* v_inst_38_){
_start:
{
lean_object* v___f_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_39_, 0, v_inst_38_);
v___x_40_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
v___x_41_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_41_, 0, v___x_40_);
lean_ctor_set(v___x_41_, 1, v___f_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_mk_x27(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_, lean_object* v_inf__comm_44_, lean_object* v_inf__assoc_45_, lean_object* v_inf__idem_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_SemilatticeInf_mk_x27___redArg(v_inst_43_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeSup___redArg___lam__0(lean_object* v_inf_48_, lean_object* v_a_49_, lean_object* v_b_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lean_apply_2(v_inf_48_, v_a_49_, v_b_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeSup___redArg(lean_object* v_h_52_){
_start:
{
lean_object* v_toPartialOrder_53_; lean_object* v_inf_54_; lean_object* v___x_56_; uint8_t v_isShared_57_; uint8_t v_isSharedCheck_63_; 
v_toPartialOrder_53_ = lean_ctor_get(v_h_52_, 0);
v_inf_54_ = lean_ctor_get(v_h_52_, 1);
v_isSharedCheck_63_ = !lean_is_exclusive(v_h_52_);
if (v_isSharedCheck_63_ == 0)
{
v___x_56_ = v_h_52_;
v_isShared_57_ = v_isSharedCheck_63_;
goto v_resetjp_55_;
}
else
{
lean_inc(v_inf_54_);
lean_inc(v_toPartialOrder_53_);
lean_dec(v_h_52_);
v___x_56_ = lean_box(0);
v_isShared_57_ = v_isSharedCheck_63_;
goto v_resetjp_55_;
}
v_resetjp_55_:
{
lean_object* v___f_58_; lean_object* v___x_59_; lean_object* v___x_61_; 
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instSemilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_58_, 0, v_inf_54_);
v___x_59_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_toPartialOrder_53_);
lean_dec_ref(v_toPartialOrder_53_);
if (v_isShared_57_ == 0)
{
lean_ctor_set(v___x_56_, 1, v___f_58_);
lean_ctor_set(v___x_56_, 0, v___x_59_);
v___x_61_ = v___x_56_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_62_; 
v_reuseFailAlloc_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_62_, 0, v___x_59_);
lean_ctor_set(v_reuseFailAlloc_62_, 1, v___f_58_);
v___x_61_ = v_reuseFailAlloc_62_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
return v___x_61_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeSup(lean_object* v_00_u03b1_64_, lean_object* v_h_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_OrderDual_instSemilatticeSup___redArg(v_h_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeInf___redArg___lam__0(lean_object* v_sup_67_, lean_object* v_a_68_, lean_object* v_b_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lean_apply_2(v_sup_67_, v_a_68_, v_b_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeInf___redArg(lean_object* v_h_71_){
_start:
{
lean_object* v_toPartialOrder_72_; lean_object* v_sup_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_82_; 
v_toPartialOrder_72_ = lean_ctor_get(v_h_71_, 0);
v_sup_73_ = lean_ctor_get(v_h_71_, 1);
v_isSharedCheck_82_ = !lean_is_exclusive(v_h_71_);
if (v_isSharedCheck_82_ == 0)
{
v___x_75_ = v_h_71_;
v_isShared_76_ = v_isSharedCheck_82_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_sup_73_);
lean_inc(v_toPartialOrder_72_);
lean_dec(v_h_71_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_82_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v___f_77_; lean_object* v___x_78_; lean_object* v___x_80_; 
v___f_77_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instSemilatticeInf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_77_, 0, v_sup_73_);
v___x_78_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_toPartialOrder_72_);
lean_dec_ref(v_toPartialOrder_72_);
if (v_isShared_76_ == 0)
{
lean_ctor_set(v___x_75_, 1, v___f_77_);
lean_ctor_set(v___x_75_, 0, v___x_78_);
v___x_80_ = v___x_75_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_81_; 
v_reuseFailAlloc_81_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_81_, 0, v___x_78_);
lean_ctor_set(v_reuseFailAlloc_81_, 1, v___f_77_);
v___x_80_ = v_reuseFailAlloc_81_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
return v___x_80_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemilatticeInf(lean_object* v_00_u03b1_83_, lean_object* v_h_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_OrderDual_instSemilatticeInf___redArg(v_h_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object* v_self_86_){
_start:
{
lean_object* v_toSemilatticeSup_87_; lean_object* v_inf_88_; lean_object* v_toPartialOrder_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_96_; 
v_toSemilatticeSup_87_ = lean_ctor_get(v_self_86_, 0);
lean_inc_ref(v_toSemilatticeSup_87_);
v_inf_88_ = lean_ctor_get(v_self_86_, 1);
lean_inc(v_inf_88_);
lean_dec_ref(v_self_86_);
v_toPartialOrder_89_ = lean_ctor_get(v_toSemilatticeSup_87_, 0);
v_isSharedCheck_96_ = !lean_is_exclusive(v_toSemilatticeSup_87_);
if (v_isSharedCheck_96_ == 0)
{
lean_object* v_unused_97_; 
v_unused_97_ = lean_ctor_get(v_toSemilatticeSup_87_, 1);
lean_dec(v_unused_97_);
v___x_91_ = v_toSemilatticeSup_87_;
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_toPartialOrder_89_);
lean_dec(v_toSemilatticeSup_87_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_94_; 
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 1, v_inf_88_);
v___x_94_ = v___x_91_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v_toPartialOrder_89_);
lean_ctor_set(v_reuseFailAlloc_95_, 1, v_inf_88_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toSemilatticeInf(lean_object* v_00_u03b1_98_, lean_object* v_self_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_self_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_mkDual___redArg(lean_object* v_inst_101_, lean_object* v_sup_102_){
_start:
{
lean_object* v_toPartialOrder_103_; lean_object* v_inf_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_112_; 
v_toPartialOrder_103_ = lean_ctor_get(v_inst_101_, 0);
v_inf_104_ = lean_ctor_get(v_inst_101_, 1);
v_isSharedCheck_112_ = !lean_is_exclusive(v_inst_101_);
if (v_isSharedCheck_112_ == 0)
{
v___x_106_ = v_inst_101_;
v_isShared_107_ = v_isSharedCheck_112_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_inf_104_);
lean_inc(v_toPartialOrder_103_);
lean_dec(v_inst_101_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_112_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___x_109_; 
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 1, v_sup_102_);
v___x_109_ = v___x_106_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_111_; 
v_reuseFailAlloc_111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_111_, 0, v_toPartialOrder_103_);
lean_ctor_set(v_reuseFailAlloc_111_, 1, v_sup_102_);
v___x_109_ = v_reuseFailAlloc_111_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
lean_object* v___x_110_; 
v___x_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_inf_104_);
return v___x_110_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_mkDual(lean_object* v_00_u03b1_113_, lean_object* v_inst_114_, lean_object* v_sup_115_, lean_object* v_le__sup__left_116_, lean_object* v_le__sup__right_117_, lean_object* v_sup__le_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_mathlib_Lattice_mkDual___redArg(v_inst_114_, v_sup_115_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLattice___redArg___lam__0(lean_object* v_toSemilatticeSup_120_, lean_object* v_a_121_, lean_object* v_b_122_){
_start:
{
lean_object* v_sup_123_; lean_object* v___x_124_; 
v_sup_123_ = lean_ctor_get(v_toSemilatticeSup_120_, 1);
lean_inc(v_sup_123_);
lean_dec_ref(v_toSemilatticeSup_120_);
v___x_124_ = lean_apply_2(v_sup_123_, v_a_121_, v_b_122_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLattice___redArg(lean_object* v_inst_125_){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v_toSemilatticeSup_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_136_; 
lean_inc_ref(v_inst_125_);
v___x_126_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_125_);
v___x_127_ = lp_mathlib_OrderDual_instSemilatticeSup___redArg(v___x_126_);
v_toSemilatticeSup_128_ = lean_ctor_get(v_inst_125_, 0);
v_isSharedCheck_136_ = !lean_is_exclusive(v_inst_125_);
if (v_isSharedCheck_136_ == 0)
{
lean_object* v_unused_137_; 
v_unused_137_ = lean_ctor_get(v_inst_125_, 1);
lean_dec(v_unused_137_);
v___x_130_ = v_inst_125_;
v_isShared_131_ = v_isSharedCheck_136_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_toSemilatticeSup_128_);
lean_dec(v_inst_125_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_136_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___f_132_; lean_object* v___x_134_; 
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_132_, 0, v_toSemilatticeSup_128_);
if (v_isShared_131_ == 0)
{
lean_ctor_set(v___x_130_, 1, v___f_132_);
lean_ctor_set(v___x_130_, 0, v___x_127_);
v___x_134_ = v___x_130_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v___x_127_);
lean_ctor_set(v_reuseFailAlloc_135_, 1, v___f_132_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLattice(lean_object* v_00_u03b1_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lp_mathlib_OrderDual_instLattice___redArg(v_inst_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_mk_x27___redArg(lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v___f_143_; lean_object* v_semilatt__sup__inst_144_; lean_object* v___x_145_; 
v___f_143_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_143_, 0, v_inst_142_);
v_semilatt__sup__inst_144_ = lp_mathlib_SemilatticeSup_mk_x27___redArg(v_inst_141_);
v___x_145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_145_, 0, v_semilatt__sup__inst_144_);
lean_ctor_set(v___x_145_, 1, v___f_143_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_mk_x27(lean_object* v_00_u03b1_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_sup__comm_149_, lean_object* v_sup__assoc_150_, lean_object* v_inf__comm_151_, lean_object* v_inf__assoc_152_, lean_object* v_sup__inf__self_153_, lean_object* v_inf__sup__self_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Lattice_mk_x27___redArg(v_inst_147_, v_inst_148_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDistribLattice___redArg(lean_object* v_inst_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lp_mathlib_OrderDual_instLattice___redArg(v_inst_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDistribLattice(lean_object* v_00_u03b1_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_OrderDual_instLattice___redArg(v_inst_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribLattice_ofInfSupLe___redArg(lean_object* v_inst_161_){
_start:
{
lean_inc_ref(v_inst_161_);
return v_inst_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribLattice_ofInfSupLe___redArg___boxed(lean_object* v_inst_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_DistribLattice_ofInfSupLe___redArg(v_inst_162_);
lean_dec_ref(v_inst_162_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribLattice_ofInfSupLe(lean_object* v_00_u03b1_164_, lean_object* v_inst_165_, lean_object* v_inf__sup__le_166_){
_start:
{
lean_inc_ref(v_inst_165_);
return v_inst_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribLattice_ofInfSupLe___boxed(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_, lean_object* v_inf__sup__le_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_DistribLattice_ofInfSupLe(v_00_u03b1_167_, v_inst_168_, v_inf__sup__le_169_);
lean_dec_ref(v_inst_168_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object* v_inst_171_){
_start:
{
lean_object* v_toPartialOrder_172_; lean_object* v_toMin_173_; lean_object* v_toMax_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
v_toPartialOrder_172_ = lean_ctor_get(v_inst_171_, 0);
v_toMin_173_ = lean_ctor_get(v_inst_171_, 1);
v_toMax_174_ = lean_ctor_get(v_inst_171_, 2);
lean_inc(v_toMax_174_);
lean_inc_ref(v_toPartialOrder_172_);
v___x_175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_175_, 0, v_toPartialOrder_172_);
lean_ctor_set(v___x_175_, 1, v_toMax_174_);
lean_inc(v_toMin_173_);
v___x_176_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
lean_ctor_set(v___x_176_, 1, v_toMin_173_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toLattice___redArg___boxed(lean_object* v_inst_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_177_);
lean_dec_ref(v_inst_177_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toLattice(lean_object* v_00_u03b1_179_, lean_object* v_inst_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toLattice___boxed(lean_object* v_00_u03b1_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_LinearOrder_toLattice(v_00_u03b1_182_, v_inst_183_);
lean_dec_ref(v_inst_183_);
return v_res_184_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lattice_toLinearOrder___redArg___lam__0(lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_a_187_, lean_object* v_b_188_){
_start:
{
lean_object* v___x_189_; uint8_t v___x_190_; 
lean_inc(v_b_188_);
lean_inc(v_a_187_);
v___x_189_ = lean_apply_2(v_inst_185_, v_a_187_, v_b_188_);
v___x_190_ = lean_unbox(v___x_189_);
if (v___x_190_ == 0)
{
lean_object* v___x_191_; uint8_t v___x_192_; 
v___x_191_ = lean_apply_2(v_inst_186_, v_a_187_, v_b_188_);
v___x_192_ = lean_unbox(v___x_191_);
if (v___x_192_ == 0)
{
uint8_t v___x_193_; 
v___x_193_ = 2;
return v___x_193_;
}
else
{
uint8_t v___x_194_; 
v___x_194_ = 1;
return v___x_194_;
}
}
else
{
uint8_t v___x_195_; 
lean_dec(v_b_188_);
lean_dec(v_a_187_);
lean_dec_ref(v_inst_186_);
v___x_195_ = 0;
return v___x_195_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toLinearOrder___redArg___lam__0___boxed(lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_a_198_, lean_object* v_b_199_){
_start:
{
uint8_t v_res_200_; lean_object* v_r_201_; 
v_res_200_ = lp_mathlib_Lattice_toLinearOrder___redArg___lam__0(v_inst_196_, v_inst_197_, v_a_198_, v_b_199_);
v_r_201_ = lean_box(v_res_200_);
return v_r_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toLinearOrder___redArg(lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; lean_object* v_toPartialOrder_207_; lean_object* v_toSemilatticeSup_208_; lean_object* v___f_209_; lean_object* v___f_210_; lean_object* v___f_211_; lean_object* v___x_212_; 
lean_inc_ref(v_inst_202_);
v___x_206_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_202_);
v_toPartialOrder_207_ = lean_ctor_get(v___x_206_, 0);
lean_inc_ref(v_toPartialOrder_207_);
v_toSemilatticeSup_208_ = lean_ctor_get(v_inst_202_, 0);
lean_inc_ref(v_toSemilatticeSup_208_);
lean_dec_ref(v_inst_202_);
lean_inc_ref(v_inst_203_);
lean_inc_ref(v_inst_205_);
v___f_209_ = lean_alloc_closure((void*)(lp_mathlib_Lattice_toLinearOrder___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_209_, 0, v_inst_205_);
lean_closure_set(v___f_209_, 1, v_inst_203_);
v___f_210_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_210_, 0, v___x_206_);
v___f_211_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_211_, 0, v_toSemilatticeSup_208_);
v___x_212_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_212_, 0, v_toPartialOrder_207_);
lean_ctor_set(v___x_212_, 1, v___f_210_);
lean_ctor_set(v___x_212_, 2, v___f_211_);
lean_ctor_set(v___x_212_, 3, v___f_209_);
lean_ctor_set(v___x_212_, 4, v_inst_204_);
lean_ctor_set(v___x_212_, 5, v_inst_203_);
lean_ctor_set(v___x_212_, 6, v_inst_205_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lattice_toLinearOrder(lean_object* v_00_u03b1_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v___x_219_; lean_object* v_toPartialOrder_220_; lean_object* v_toSemilatticeSup_221_; lean_object* v___f_222_; lean_object* v___f_223_; lean_object* v___f_224_; lean_object* v___x_225_; 
lean_inc_ref(v_inst_214_);
v___x_219_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_214_);
v_toPartialOrder_220_ = lean_ctor_get(v___x_219_, 0);
lean_inc_ref(v_toPartialOrder_220_);
v_toSemilatticeSup_221_ = lean_ctor_get(v_inst_214_, 0);
lean_inc_ref(v_toSemilatticeSup_221_);
lean_dec_ref(v_inst_214_);
lean_inc_ref(v_inst_215_);
lean_inc_ref(v_inst_217_);
v___f_222_ = lean_alloc_closure((void*)(lp_mathlib_Lattice_toLinearOrder___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_222_, 0, v_inst_217_);
lean_closure_set(v___f_222_, 1, v_inst_215_);
v___f_223_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_223_, 0, v___x_219_);
v___f_224_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_224_, 0, v_toSemilatticeSup_221_);
v___x_225_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_225_, 0, v_toPartialOrder_220_);
lean_ctor_set(v___x_225_, 1, v___f_223_);
lean_ctor_set(v___x_225_, 2, v___f_224_);
lean_ctor_set(v___x_225_, 3, v___f_222_);
lean_ctor_set(v___x_225_, 4, v_inst_216_);
lean_ctor_set(v___x_225_, 5, v_inst_215_);
lean_ctor_set(v___x_225_, 6, v_inst_217_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeOfLinearOrder___redArg(lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeOfLinearOrder___redArg___boxed(lean_object* v_inst_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_instDistribLatticeOfLinearOrder___redArg(v_inst_228_);
lean_dec_ref(v_inst_228_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeOfLinearOrder(lean_object* v_00_u03b1_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribLatticeOfLinearOrder___boxed(lean_object* v_00_u03b1_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_instDistribLatticeOfLinearOrder(v_00_u03b1_233_, v_inst_234_);
lean_dec_ref(v_inst_234_);
return v_res_235_;
}
}
static lean_object* _init_lp_mathlib_instDistribLatticeNat___closed__0(void){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_236_ = lp_mathlib_Nat_instLinearOrder;
v___x_237_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_236_);
return v___x_237_;
}
}
static lean_object* _init_lp_mathlib_instDistribLatticeNat(void){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lean_obj_once(&lp_mathlib_instDistribLatticeNat___closed__0, &lp_mathlib_instDistribLatticeNat___closed__0_once, _init_lp_mathlib_instDistribLatticeNat___closed__0);
return v___x_238_;
}
}
static lean_object* _init_lp_mathlib_instLatticeInt___closed__0(void){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_239_ = lp_mathlib_Int_instLinearOrder;
v___x_240_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_239_);
return v___x_240_;
}
}
static lean_object* _init_lp_mathlib_instLatticeInt(void){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lean_obj_once(&lp_mathlib_instLatticeInt___closed__0, &lp_mathlib_instLatticeInt___closed__0_once, _init_lp_mathlib_instLatticeInt___closed__0);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMaxForall__mathlib___redArg___lam__0(lean_object* v_inst_242_, lean_object* v_f_243_, lean_object* v_g_244_, lean_object* v_i_245_){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
lean_inc_n(v_i_245_, 2);
v___x_246_ = lean_apply_1(v_f_243_, v_i_245_);
v___x_247_ = lean_apply_1(v_g_244_, v_i_245_);
v___x_248_ = lean_apply_3(v_inst_242_, v_i_245_, v___x_246_, v___x_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMaxForall__mathlib___redArg(lean_object* v_inst_249_){
_start:
{
lean_object* v___f_250_; 
v___f_250_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMaxForall__mathlib___redArg___lam__0), 4, 1);
lean_closure_set(v___f_250_, 0, v_inst_249_);
return v___f_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMaxForall__mathlib(lean_object* v_00_u03b9_251_, lean_object* v_00_u03b1_x27_252_, lean_object* v_inst_253_){
_start:
{
lean_object* v___f_254_; 
v___f_254_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMaxForall__mathlib___redArg___lam__0), 4, 1);
lean_closure_set(v___f_254_, 0, v_inst_253_);
return v___f_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMinForall__mathlib___redArg(lean_object* v_inst_255_){
_start:
{
lean_object* v___f_256_; 
v___f_256_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMaxForall__mathlib___redArg___lam__0), 4, 1);
lean_closure_set(v___f_256_, 0, v_inst_255_);
return v___f_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instMinForall__mathlib(lean_object* v_00_u03b9_257_, lean_object* v_00_u03b1_x27_258_, lean_object* v_inst_259_){
_start:
{
lean_object* v___f_260_; 
v___f_260_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMaxForall__mathlib___redArg___lam__0), 4, 1);
lean_closure_set(v___f_260_, 0, v_inst_259_);
return v___f_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeSup___redArg___lam__0(lean_object* v_inst_261_, lean_object* v_i_262_){
_start:
{
lean_object* v___x_263_; lean_object* v_toPartialOrder_264_; 
v___x_263_ = lean_apply_1(v_inst_261_, v_i_262_);
v_toPartialOrder_264_ = lean_ctor_get(v___x_263_, 0);
lean_inc_ref(v_toPartialOrder_264_);
lean_dec_ref(v___x_263_);
return v_toPartialOrder_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeSup___redArg___lam__1(lean_object* v_inst_265_, lean_object* v_x_266_, lean_object* v_y_267_, lean_object* v_i_268_){
_start:
{
lean_object* v___x_269_; lean_object* v_sup_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; 
lean_inc_n(v_i_268_, 2);
v___x_269_ = lean_apply_1(v_inst_265_, v_i_268_);
v_sup_270_ = lean_ctor_get(v___x_269_, 1);
lean_inc(v_sup_270_);
lean_dec_ref(v___x_269_);
v___x_271_ = lean_apply_1(v_x_266_, v_i_268_);
v___x_272_ = lean_apply_1(v_y_267_, v_i_268_);
v___x_273_ = lean_apply_2(v_sup_270_, v___x_271_, v___x_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeSup___redArg(lean_object* v_inst_274_){
_start:
{
lean_object* v___f_275_; lean_object* v___f_276_; lean_object* v___x_277_; lean_object* v___x_278_; 
lean_inc_ref(v_inst_274_);
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSemilatticeSup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_275_, 0, v_inst_274_);
v___f_276_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSemilatticeSup___redArg___lam__1), 4, 1);
lean_closure_set(v___f_276_, 0, v_inst_274_);
v___x_277_ = lp_mathlib_Pi_partialOrder___redArg(v___f_275_);
v___x_278_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_278_, 0, v___x_277_);
lean_ctor_set(v___x_278_, 1, v___f_276_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeSup(lean_object* v_00_u03b9_279_, lean_object* v_00_u03b1_x27_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lp_mathlib_Pi_instSemilatticeSup___redArg(v_inst_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeInf___redArg___lam__0(lean_object* v_inst_283_, lean_object* v_i_284_){
_start:
{
lean_object* v___x_285_; lean_object* v_toPartialOrder_286_; 
v___x_285_ = lean_apply_1(v_inst_283_, v_i_284_);
v_toPartialOrder_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc_ref(v_toPartialOrder_286_);
lean_dec_ref(v___x_285_);
return v_toPartialOrder_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeInf___redArg___lam__1(lean_object* v_inst_287_, lean_object* v_x_288_, lean_object* v_y_289_, lean_object* v_i_290_){
_start:
{
lean_object* v___x_291_; lean_object* v_inf_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
lean_inc_n(v_i_290_, 2);
v___x_291_ = lean_apply_1(v_inst_287_, v_i_290_);
v_inf_292_ = lean_ctor_get(v___x_291_, 1);
lean_inc(v_inf_292_);
lean_dec_ref(v___x_291_);
v___x_293_ = lean_apply_1(v_x_288_, v_i_290_);
v___x_294_ = lean_apply_1(v_y_289_, v_i_290_);
v___x_295_ = lean_apply_2(v_inf_292_, v___x_293_, v___x_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeInf___redArg(lean_object* v_inst_296_){
_start:
{
lean_object* v___f_297_; lean_object* v___f_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
lean_inc_ref(v_inst_296_);
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSemilatticeInf___redArg___lam__0), 2, 1);
lean_closure_set(v___f_297_, 0, v_inst_296_);
v___f_298_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSemilatticeInf___redArg___lam__1), 4, 1);
lean_closure_set(v___f_298_, 0, v_inst_296_);
v___x_299_ = lp_mathlib_Pi_partialOrder___redArg(v___f_297_);
v___x_300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_300_, 0, v___x_299_);
lean_ctor_set(v___x_300_, 1, v___f_298_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSemilatticeInf(lean_object* v_00_u03b9_301_, lean_object* v_00_u03b1_x27_302_, lean_object* v_inst_303_){
_start:
{
lean_object* v___x_304_; 
v___x_304_ = lp_mathlib_Pi_instSemilatticeInf___redArg(v_inst_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLattice___redArg___lam__0(lean_object* v_inst_305_, lean_object* v_i_306_){
_start:
{
lean_object* v___x_307_; lean_object* v_toSemilatticeSup_308_; 
v___x_307_ = lean_apply_1(v_inst_305_, v_i_306_);
v_toSemilatticeSup_308_ = lean_ctor_get(v___x_307_, 0);
lean_inc_ref(v_toSemilatticeSup_308_);
lean_dec_ref(v___x_307_);
return v_toSemilatticeSup_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLattice___redArg___lam__1(lean_object* v_inst_309_, lean_object* v_x_310_, lean_object* v_y_311_, lean_object* v_i_312_){
_start:
{
lean_object* v___x_313_; lean_object* v_inf_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; 
lean_inc_n(v_i_312_, 2);
v___x_313_ = lean_apply_1(v_inst_309_, v_i_312_);
v_inf_314_ = lean_ctor_get(v___x_313_, 1);
lean_inc(v_inf_314_);
lean_dec_ref(v___x_313_);
v___x_315_ = lean_apply_1(v_x_310_, v_i_312_);
v___x_316_ = lean_apply_1(v_y_311_, v_i_312_);
v___x_317_ = lean_apply_2(v_inf_314_, v___x_315_, v___x_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLattice___redArg(lean_object* v_inst_318_){
_start:
{
lean_object* v___f_319_; lean_object* v___f_320_; lean_object* v___x_321_; lean_object* v___x_322_; 
lean_inc_ref(v_inst_318_);
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instLattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_319_, 0, v_inst_318_);
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instLattice___redArg___lam__1), 4, 1);
lean_closure_set(v___f_320_, 0, v_inst_318_);
v___x_321_ = lp_mathlib_Pi_instSemilatticeSup___redArg(v___f_319_);
v___x_322_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_321_);
lean_ctor_set(v___x_322_, 1, v___f_320_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instLattice(lean_object* v_00_u03b9_323_, lean_object* v_00_u03b1_x27_324_, lean_object* v_inst_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lp_mathlib_Pi_instLattice___redArg(v_inst_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instDistribLattice___redArg___lam__0(lean_object* v_inst_327_, lean_object* v_i_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lean_apply_1(v_inst_327_, v_i_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instDistribLattice___redArg(lean_object* v_inst_330_){
_start:
{
lean_object* v___f_331_; lean_object* v___x_332_; 
v___f_331_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instDistribLattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_331_, 0, v_inst_330_);
v___x_332_ = lp_mathlib_Pi_instLattice___redArg(v___f_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instDistribLattice(lean_object* v_00_u03b9_333_, lean_object* v_00_u03b1_x27_334_, lean_object* v_inst_335_){
_start:
{
lean_object* v___x_336_; 
v___x_336_ = lp_mathlib_Pi_instDistribLattice___redArg(v_inst_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMax__mathlib___redArg___lam__0(lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_p_339_, lean_object* v_q_340_){
_start:
{
lean_object* v_fst_341_; lean_object* v_snd_342_; lean_object* v_fst_343_; lean_object* v_snd_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_353_; 
v_fst_341_ = lean_ctor_get(v_p_339_, 0);
lean_inc(v_fst_341_);
v_snd_342_ = lean_ctor_get(v_p_339_, 1);
lean_inc(v_snd_342_);
lean_dec_ref(v_p_339_);
v_fst_343_ = lean_ctor_get(v_q_340_, 0);
v_snd_344_ = lean_ctor_get(v_q_340_, 1);
v_isSharedCheck_353_ = !lean_is_exclusive(v_q_340_);
if (v_isSharedCheck_353_ == 0)
{
v___x_346_ = v_q_340_;
v_isShared_347_ = v_isSharedCheck_353_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_snd_344_);
lean_inc(v_fst_343_);
lean_dec(v_q_340_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_353_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_351_; 
v___x_348_ = lean_apply_2(v_inst_337_, v_fst_341_, v_fst_343_);
v___x_349_ = lean_apply_2(v_inst_338_, v_snd_342_, v_snd_344_);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 1, v___x_349_);
lean_ctor_set(v___x_346_, 0, v___x_348_);
v___x_351_ = v___x_346_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v___x_348_);
lean_ctor_set(v_reuseFailAlloc_352_, 1, v___x_349_);
v___x_351_ = v_reuseFailAlloc_352_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
return v___x_351_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMax__mathlib___redArg(lean_object* v_inst_354_, lean_object* v_inst_355_){
_start:
{
lean_object* v___f_356_; 
v___f_356_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMax__mathlib___redArg___lam__0), 4, 2);
lean_closure_set(v___f_356_, 0, v_inst_354_);
lean_closure_set(v___f_356_, 1, v_inst_355_);
return v___f_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMax__mathlib(lean_object* v_00_u03b1_357_, lean_object* v_00_u03b2_358_, lean_object* v_inst_359_, lean_object* v_inst_360_){
_start:
{
lean_object* v___f_361_; 
v___f_361_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMax__mathlib___redArg___lam__0), 4, 2);
lean_closure_set(v___f_361_, 0, v_inst_359_);
lean_closure_set(v___f_361_, 1, v_inst_360_);
return v___f_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMin__mathlib___redArg(lean_object* v_inst_362_, lean_object* v_inst_363_){
_start:
{
lean_object* v___f_364_; 
v___f_364_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMax__mathlib___redArg___lam__0), 4, 2);
lean_closure_set(v___f_364_, 0, v_inst_362_);
lean_closure_set(v___f_364_, 1, v_inst_363_);
return v___f_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMin__mathlib(lean_object* v_00_u03b1_365_, lean_object* v_00_u03b2_366_, lean_object* v_inst_367_, lean_object* v_inst_368_){
_start:
{
lean_object* v___f_369_; 
v___f_369_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMax__mathlib___redArg___lam__0), 4, 2);
lean_closure_set(v___f_369_, 0, v_inst_367_);
lean_closure_set(v___f_369_, 1, v_inst_368_);
return v___f_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeSup___redArg___lam__0(lean_object* v_sup_370_, lean_object* v_sup_371_, lean_object* v_a_372_, lean_object* v_b_373_){
_start:
{
lean_object* v_fst_374_; lean_object* v_snd_375_; lean_object* v_fst_376_; lean_object* v_snd_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_386_; 
v_fst_374_ = lean_ctor_get(v_a_372_, 0);
lean_inc(v_fst_374_);
v_snd_375_ = lean_ctor_get(v_a_372_, 1);
lean_inc(v_snd_375_);
lean_dec_ref(v_a_372_);
v_fst_376_ = lean_ctor_get(v_b_373_, 0);
v_snd_377_ = lean_ctor_get(v_b_373_, 1);
v_isSharedCheck_386_ = !lean_is_exclusive(v_b_373_);
if (v_isSharedCheck_386_ == 0)
{
v___x_379_ = v_b_373_;
v_isShared_380_ = v_isSharedCheck_386_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_snd_377_);
lean_inc(v_fst_376_);
lean_dec(v_b_373_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_386_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_384_; 
v___x_381_ = lean_apply_2(v_sup_370_, v_fst_374_, v_fst_376_);
v___x_382_ = lean_apply_2(v_sup_371_, v_snd_375_, v_snd_377_);
if (v_isShared_380_ == 0)
{
lean_ctor_set(v___x_379_, 1, v___x_382_);
lean_ctor_set(v___x_379_, 0, v___x_381_);
v___x_384_ = v___x_379_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v___x_381_);
lean_ctor_set(v_reuseFailAlloc_385_, 1, v___x_382_);
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
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeSup___redArg(lean_object* v_inst_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v_toPartialOrder_389_; lean_object* v_sup_390_; lean_object* v_toPartialOrder_391_; lean_object* v_sup_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_401_; 
v_toPartialOrder_389_ = lean_ctor_get(v_inst_387_, 0);
lean_inc_ref(v_toPartialOrder_389_);
v_sup_390_ = lean_ctor_get(v_inst_387_, 1);
lean_inc(v_sup_390_);
lean_dec_ref(v_inst_387_);
v_toPartialOrder_391_ = lean_ctor_get(v_inst_388_, 0);
v_sup_392_ = lean_ctor_get(v_inst_388_, 1);
v_isSharedCheck_401_ = !lean_is_exclusive(v_inst_388_);
if (v_isSharedCheck_401_ == 0)
{
v___x_394_ = v_inst_388_;
v_isShared_395_ = v_isSharedCheck_401_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_sup_392_);
lean_inc(v_toPartialOrder_391_);
lean_dec(v_inst_388_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_401_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v___f_396_; lean_object* v___x_397_; lean_object* v___x_399_; 
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSemilatticeSup___redArg___lam__0), 4, 2);
lean_closure_set(v___f_396_, 0, v_sup_390_);
lean_closure_set(v___f_396_, 1, v_sup_392_);
v___x_397_ = lp_mathlib_Prod_instPreorder(lean_box(0), lean_box(0), v_toPartialOrder_389_, v_toPartialOrder_391_);
lean_dec_ref(v_toPartialOrder_391_);
lean_dec_ref(v_toPartialOrder_389_);
if (v_isShared_395_ == 0)
{
lean_ctor_set(v___x_394_, 1, v___f_396_);
lean_ctor_set(v___x_394_, 0, v___x_397_);
v___x_399_ = v___x_394_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v___x_397_);
lean_ctor_set(v_reuseFailAlloc_400_, 1, v___f_396_);
v___x_399_ = v_reuseFailAlloc_400_;
goto v_reusejp_398_;
}
v_reusejp_398_:
{
return v___x_399_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeSup(lean_object* v_00_u03b1_402_, lean_object* v_00_u03b2_403_, lean_object* v_inst_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_Prod_instSemilatticeSup___redArg(v_inst_404_, v_inst_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeInf___redArg___lam__0(lean_object* v_inf_407_, lean_object* v_inf_408_, lean_object* v_a_409_, lean_object* v_b_410_){
_start:
{
lean_object* v_fst_411_; lean_object* v_snd_412_; lean_object* v_fst_413_; lean_object* v_snd_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_423_; 
v_fst_411_ = lean_ctor_get(v_a_409_, 0);
lean_inc(v_fst_411_);
v_snd_412_ = lean_ctor_get(v_a_409_, 1);
lean_inc(v_snd_412_);
lean_dec_ref(v_a_409_);
v_fst_413_ = lean_ctor_get(v_b_410_, 0);
v_snd_414_ = lean_ctor_get(v_b_410_, 1);
v_isSharedCheck_423_ = !lean_is_exclusive(v_b_410_);
if (v_isSharedCheck_423_ == 0)
{
v___x_416_ = v_b_410_;
v_isShared_417_ = v_isSharedCheck_423_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_snd_414_);
lean_inc(v_fst_413_);
lean_dec(v_b_410_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_423_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_421_; 
v___x_418_ = lean_apply_2(v_inf_407_, v_fst_411_, v_fst_413_);
v___x_419_ = lean_apply_2(v_inf_408_, v_snd_412_, v_snd_414_);
if (v_isShared_417_ == 0)
{
lean_ctor_set(v___x_416_, 1, v___x_419_);
lean_ctor_set(v___x_416_, 0, v___x_418_);
v___x_421_ = v___x_416_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v___x_418_);
lean_ctor_set(v_reuseFailAlloc_422_, 1, v___x_419_);
v___x_421_ = v_reuseFailAlloc_422_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
return v___x_421_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeInf___redArg(lean_object* v_inst_424_, lean_object* v_inst_425_){
_start:
{
lean_object* v_toPartialOrder_426_; lean_object* v_inf_427_; lean_object* v_toPartialOrder_428_; lean_object* v_inf_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_438_; 
v_toPartialOrder_426_ = lean_ctor_get(v_inst_424_, 0);
lean_inc_ref(v_toPartialOrder_426_);
v_inf_427_ = lean_ctor_get(v_inst_424_, 1);
lean_inc(v_inf_427_);
lean_dec_ref(v_inst_424_);
v_toPartialOrder_428_ = lean_ctor_get(v_inst_425_, 0);
v_inf_429_ = lean_ctor_get(v_inst_425_, 1);
v_isSharedCheck_438_ = !lean_is_exclusive(v_inst_425_);
if (v_isSharedCheck_438_ == 0)
{
v___x_431_ = v_inst_425_;
v_isShared_432_ = v_isSharedCheck_438_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_inf_429_);
lean_inc(v_toPartialOrder_428_);
lean_dec(v_inst_425_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_438_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v___f_433_; lean_object* v___x_434_; lean_object* v___x_436_; 
v___f_433_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSemilatticeInf___redArg___lam__0), 4, 2);
lean_closure_set(v___f_433_, 0, v_inf_427_);
lean_closure_set(v___f_433_, 1, v_inf_429_);
v___x_434_ = lp_mathlib_Prod_instPreorder(lean_box(0), lean_box(0), v_toPartialOrder_426_, v_toPartialOrder_428_);
lean_dec_ref(v_toPartialOrder_428_);
lean_dec_ref(v_toPartialOrder_426_);
if (v_isShared_432_ == 0)
{
lean_ctor_set(v___x_431_, 1, v___f_433_);
lean_ctor_set(v___x_431_, 0, v___x_434_);
v___x_436_ = v___x_431_;
goto v_reusejp_435_;
}
else
{
lean_object* v_reuseFailAlloc_437_; 
v_reuseFailAlloc_437_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_437_, 0, v___x_434_);
lean_ctor_set(v_reuseFailAlloc_437_, 1, v___f_433_);
v___x_436_ = v_reuseFailAlloc_437_;
goto v_reusejp_435_;
}
v_reusejp_435_:
{
return v___x_436_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemilatticeInf(lean_object* v_00_u03b1_439_, lean_object* v_00_u03b2_440_, lean_object* v_inst_441_, lean_object* v_inst_442_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lp_mathlib_Prod_instSemilatticeInf___redArg(v_inst_441_, v_inst_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLattice___redArg(lean_object* v_inst_444_, lean_object* v_inst_445_){
_start:
{
lean_object* v_toSemilatticeSup_446_; lean_object* v_inf_447_; lean_object* v_toSemilatticeSup_448_; lean_object* v_inf_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_458_; 
v_toSemilatticeSup_446_ = lean_ctor_get(v_inst_444_, 0);
lean_inc_ref(v_toSemilatticeSup_446_);
v_inf_447_ = lean_ctor_get(v_inst_444_, 1);
lean_inc(v_inf_447_);
lean_dec_ref(v_inst_444_);
v_toSemilatticeSup_448_ = lean_ctor_get(v_inst_445_, 0);
v_inf_449_ = lean_ctor_get(v_inst_445_, 1);
v_isSharedCheck_458_ = !lean_is_exclusive(v_inst_445_);
if (v_isSharedCheck_458_ == 0)
{
v___x_451_ = v_inst_445_;
v_isShared_452_ = v_isSharedCheck_458_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_inf_449_);
lean_inc(v_toSemilatticeSup_448_);
lean_dec(v_inst_445_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_458_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v___f_453_; lean_object* v___x_454_; lean_object* v___x_456_; 
v___f_453_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSemilatticeInf___redArg___lam__0), 4, 2);
lean_closure_set(v___f_453_, 0, v_inf_447_);
lean_closure_set(v___f_453_, 1, v_inf_449_);
v___x_454_ = lp_mathlib_Prod_instSemilatticeSup___redArg(v_toSemilatticeSup_446_, v_toSemilatticeSup_448_);
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 1, v___f_453_);
lean_ctor_set(v___x_451_, 0, v___x_454_);
v___x_456_ = v___x_451_;
goto v_reusejp_455_;
}
else
{
lean_object* v_reuseFailAlloc_457_; 
v_reuseFailAlloc_457_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_457_, 0, v___x_454_);
lean_ctor_set(v_reuseFailAlloc_457_, 1, v___f_453_);
v___x_456_ = v_reuseFailAlloc_457_;
goto v_reusejp_455_;
}
v_reusejp_455_:
{
return v___x_456_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLattice(lean_object* v_00_u03b1_459_, lean_object* v_00_u03b2_460_, lean_object* v_inst_461_, lean_object* v_inst_462_){
_start:
{
lean_object* v___x_463_; 
v___x_463_ = lp_mathlib_Prod_instLattice___redArg(v_inst_461_, v_inst_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDistribLattice___redArg(lean_object* v_inst_464_, lean_object* v_inst_465_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_mathlib_Prod_instLattice___redArg(v_inst_464_, v_inst_465_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDistribLattice(lean_object* v_00_u03b1_467_, lean_object* v_00_u03b2_468_, lean_object* v_inst_469_, lean_object* v_inst_470_){
_start:
{
lean_object* v___x_471_; 
v___x_471_ = lp_mathlib_Prod_instLattice___redArg(v_inst_469_, v_inst_470_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeSup___redArg___lam__0(lean_object* v_sup_472_, lean_object* v_x_473_, lean_object* v_y_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lean_apply_2(v_sup_472_, v_x_473_, v_y_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeSup___redArg(lean_object* v_inst_476_){
_start:
{
lean_object* v_toPartialOrder_477_; lean_object* v_sup_478_; lean_object* v___x_480_; uint8_t v_isShared_481_; uint8_t v_isSharedCheck_487_; 
v_toPartialOrder_477_ = lean_ctor_get(v_inst_476_, 0);
v_sup_478_ = lean_ctor_get(v_inst_476_, 1);
v_isSharedCheck_487_ = !lean_is_exclusive(v_inst_476_);
if (v_isSharedCheck_487_ == 0)
{
v___x_480_ = v_inst_476_;
v_isShared_481_ = v_isSharedCheck_487_;
goto v_resetjp_479_;
}
else
{
lean_inc(v_sup_478_);
lean_inc(v_toPartialOrder_477_);
lean_dec(v_inst_476_);
v___x_480_ = lean_box(0);
v_isShared_481_ = v_isSharedCheck_487_;
goto v_resetjp_479_;
}
v_resetjp_479_:
{
lean_object* v___f_482_; lean_object* v___x_483_; lean_object* v___x_485_; 
v___f_482_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_482_, 0, v_sup_478_);
v___x_483_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_477_, lean_box(0));
lean_dec_ref(v_toPartialOrder_477_);
if (v_isShared_481_ == 0)
{
lean_ctor_set(v___x_480_, 1, v___f_482_);
lean_ctor_set(v___x_480_, 0, v___x_483_);
v___x_485_ = v___x_480_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v___x_483_);
lean_ctor_set(v_reuseFailAlloc_486_, 1, v___f_482_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeSup(lean_object* v_00_u03b1_488_, lean_object* v_inst_489_, lean_object* v_P_490_, lean_object* v_Psup_491_){
_start:
{
lean_object* v_toPartialOrder_492_; lean_object* v_sup_493_; lean_object* v___x_495_; uint8_t v_isShared_496_; uint8_t v_isSharedCheck_502_; 
v_toPartialOrder_492_ = lean_ctor_get(v_inst_489_, 0);
v_sup_493_ = lean_ctor_get(v_inst_489_, 1);
v_isSharedCheck_502_ = !lean_is_exclusive(v_inst_489_);
if (v_isSharedCheck_502_ == 0)
{
v___x_495_ = v_inst_489_;
v_isShared_496_ = v_isSharedCheck_502_;
goto v_resetjp_494_;
}
else
{
lean_inc(v_sup_493_);
lean_inc(v_toPartialOrder_492_);
lean_dec(v_inst_489_);
v___x_495_ = lean_box(0);
v_isShared_496_ = v_isSharedCheck_502_;
goto v_resetjp_494_;
}
v_resetjp_494_:
{
lean_object* v___f_497_; lean_object* v___x_498_; lean_object* v___x_500_; 
v___f_497_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_497_, 0, v_sup_493_);
v___x_498_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_492_, lean_box(0));
lean_dec_ref(v_toPartialOrder_492_);
if (v_isShared_496_ == 0)
{
lean_ctor_set(v___x_495_, 1, v___f_497_);
lean_ctor_set(v___x_495_, 0, v___x_498_);
v___x_500_ = v___x_495_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v___x_498_);
lean_ctor_set(v_reuseFailAlloc_501_, 1, v___f_497_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeInf___redArg___lam__0(lean_object* v_inf_503_, lean_object* v_x_504_, lean_object* v_y_505_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lean_apply_2(v_inf_503_, v_x_504_, v_y_505_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeInf___redArg(lean_object* v_inst_507_){
_start:
{
lean_object* v_toPartialOrder_508_; lean_object* v_inf_509_; lean_object* v___x_511_; uint8_t v_isShared_512_; uint8_t v_isSharedCheck_518_; 
v_toPartialOrder_508_ = lean_ctor_get(v_inst_507_, 0);
v_inf_509_ = lean_ctor_get(v_inst_507_, 1);
v_isSharedCheck_518_ = !lean_is_exclusive(v_inst_507_);
if (v_isSharedCheck_518_ == 0)
{
v___x_511_ = v_inst_507_;
v_isShared_512_ = v_isSharedCheck_518_;
goto v_resetjp_510_;
}
else
{
lean_inc(v_inf_509_);
lean_inc(v_toPartialOrder_508_);
lean_dec(v_inst_507_);
v___x_511_ = lean_box(0);
v_isShared_512_ = v_isSharedCheck_518_;
goto v_resetjp_510_;
}
v_resetjp_510_:
{
lean_object* v___f_513_; lean_object* v___x_514_; lean_object* v___x_516_; 
v___f_513_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_semilatticeInf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_513_, 0, v_inf_509_);
v___x_514_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_508_, lean_box(0));
lean_dec_ref(v_toPartialOrder_508_);
if (v_isShared_512_ == 0)
{
lean_ctor_set(v___x_511_, 1, v___f_513_);
lean_ctor_set(v___x_511_, 0, v___x_514_);
v___x_516_ = v___x_511_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v___x_514_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v___f_513_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_semilatticeInf(lean_object* v_00_u03b1_519_, lean_object* v_inst_520_, lean_object* v_P_521_, lean_object* v_Psup_522_){
_start:
{
lean_object* v___x_523_; 
v___x_523_ = lp_mathlib_Subtype_semilatticeInf___redArg(v_inst_520_);
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_lattice___redArg___lam__1(lean_object* v_toSemilatticeSup_524_, lean_object* v_x_525_, lean_object* v_y_526_){
_start:
{
lean_object* v_sup_527_; lean_object* v___x_528_; 
v_sup_527_ = lean_ctor_get(v_toSemilatticeSup_524_, 1);
lean_inc(v_sup_527_);
lean_dec_ref(v_toSemilatticeSup_524_);
v___x_528_ = lean_apply_2(v_sup_527_, v_x_525_, v_y_526_);
return v___x_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_lattice___redArg(lean_object* v_inst_529_){
_start:
{
lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v_toSemilatticeSup_532_; lean_object* v_inf_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_551_; 
lean_inc_ref(v_inst_529_);
v___x_530_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_529_);
v___x_531_ = lp_mathlib_Subtype_semilatticeInf___redArg(v___x_530_);
v_toSemilatticeSup_532_ = lean_ctor_get(v_inst_529_, 0);
v_inf_533_ = lean_ctor_get(v_inst_529_, 1);
v_isSharedCheck_551_ = !lean_is_exclusive(v_inst_529_);
if (v_isSharedCheck_551_ == 0)
{
v___x_535_ = v_inst_529_;
v_isShared_536_ = v_isSharedCheck_551_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_inf_533_);
lean_inc(v_toSemilatticeSup_532_);
lean_dec(v_inst_529_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_551_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v_toPartialOrder_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_549_; 
v_toPartialOrder_537_ = lean_ctor_get(v___x_531_, 0);
v_isSharedCheck_549_ = !lean_is_exclusive(v___x_531_);
if (v_isSharedCheck_549_ == 0)
{
lean_object* v_unused_550_; 
v_unused_550_ = lean_ctor_get(v___x_531_, 1);
lean_dec(v_unused_550_);
v___x_539_ = v___x_531_;
v_isShared_540_ = v_isSharedCheck_549_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_toPartialOrder_537_);
lean_dec(v___x_531_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_549_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
lean_object* v___f_541_; lean_object* v___f_542_; lean_object* v___x_544_; 
v___f_541_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_semilatticeInf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_541_, 0, v_inf_533_);
v___f_542_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_lattice___redArg___lam__1), 3, 1);
lean_closure_set(v___f_542_, 0, v_toSemilatticeSup_532_);
if (v_isShared_540_ == 0)
{
lean_ctor_set(v___x_539_, 1, v___f_542_);
v___x_544_ = v___x_539_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_548_; 
v_reuseFailAlloc_548_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_548_, 0, v_toPartialOrder_537_);
lean_ctor_set(v_reuseFailAlloc_548_, 1, v___f_542_);
v___x_544_ = v_reuseFailAlloc_548_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
lean_object* v___x_546_; 
if (v_isShared_536_ == 0)
{
lean_ctor_set(v___x_535_, 1, v___f_541_);
lean_ctor_set(v___x_535_, 0, v___x_544_);
v___x_546_ = v___x_535_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v___x_544_);
lean_ctor_set(v_reuseFailAlloc_547_, 1, v___f_541_);
v___x_546_ = v_reuseFailAlloc_547_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
return v___x_546_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_lattice(lean_object* v_00_u03b1_552_, lean_object* v_inst_553_, lean_object* v_P_554_, lean_object* v_Psup_555_, lean_object* v_Pinf_556_){
_start:
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v_toSemilatticeSup_559_; lean_object* v_inf_560_; lean_object* v___x_562_; uint8_t v_isShared_563_; uint8_t v_isSharedCheck_578_; 
lean_inc_ref(v_inst_553_);
v___x_557_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_553_);
v___x_558_ = lp_mathlib_Subtype_semilatticeInf___redArg(v___x_557_);
v_toSemilatticeSup_559_ = lean_ctor_get(v_inst_553_, 0);
v_inf_560_ = lean_ctor_get(v_inst_553_, 1);
v_isSharedCheck_578_ = !lean_is_exclusive(v_inst_553_);
if (v_isSharedCheck_578_ == 0)
{
v___x_562_ = v_inst_553_;
v_isShared_563_ = v_isSharedCheck_578_;
goto v_resetjp_561_;
}
else
{
lean_inc(v_inf_560_);
lean_inc(v_toSemilatticeSup_559_);
lean_dec(v_inst_553_);
v___x_562_ = lean_box(0);
v_isShared_563_ = v_isSharedCheck_578_;
goto v_resetjp_561_;
}
v_resetjp_561_:
{
lean_object* v_toPartialOrder_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_576_; 
v_toPartialOrder_564_ = lean_ctor_get(v___x_558_, 0);
v_isSharedCheck_576_ = !lean_is_exclusive(v___x_558_);
if (v_isSharedCheck_576_ == 0)
{
lean_object* v_unused_577_; 
v_unused_577_ = lean_ctor_get(v___x_558_, 1);
lean_dec(v_unused_577_);
v___x_566_ = v___x_558_;
v_isShared_567_ = v_isSharedCheck_576_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_toPartialOrder_564_);
lean_dec(v___x_558_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_576_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
lean_object* v___f_568_; lean_object* v___f_569_; lean_object* v___x_571_; 
v___f_568_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_semilatticeInf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_568_, 0, v_inf_560_);
v___f_569_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_lattice___redArg___lam__1), 3, 1);
lean_closure_set(v___f_569_, 0, v_toSemilatticeSup_559_);
if (v_isShared_567_ == 0)
{
lean_ctor_set(v___x_566_, 1, v___f_569_);
v___x_571_ = v___x_566_;
goto v_reusejp_570_;
}
else
{
lean_object* v_reuseFailAlloc_575_; 
v_reuseFailAlloc_575_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_575_, 0, v_toPartialOrder_564_);
lean_ctor_set(v_reuseFailAlloc_575_, 1, v___f_569_);
v___x_571_ = v_reuseFailAlloc_575_;
goto v_reusejp_570_;
}
v_reusejp_570_:
{
lean_object* v___x_573_; 
if (v_isShared_563_ == 0)
{
lean_ctor_set(v___x_562_, 1, v___f_568_);
lean_ctor_set(v___x_562_, 0, v___x_571_);
v___x_573_ = v___x_562_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v___x_571_);
lean_ctor_set(v_reuseFailAlloc_574_, 1, v___f_568_);
v___x_573_ = v_reuseFailAlloc_574_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
return v___x_573_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0(lean_object* v_inst_579_, lean_object* v_a_580_, lean_object* v_b_581_){
_start:
{
lean_object* v___x_582_; 
v___x_582_ = lean_apply_2(v_inst_579_, v_a_580_, v_b_581_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeSup___redArg(lean_object* v_inst_583_, lean_object* v_inst_584_, lean_object* v_inst_585_){
_start:
{
lean_object* v___f_586_; lean_object* v___x_587_; lean_object* v___x_588_; 
v___f_586_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_586_, 0, v_inst_583_);
v___x_587_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_587_, 0, v_inst_584_);
lean_ctor_set(v___x_587_, 1, v_inst_585_);
v___x_588_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_588_, 0, v___x_587_);
lean_ctor_set(v___x_588_, 1, v___f_586_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeSup(lean_object* v_00_u03b1_589_, lean_object* v_00_u03b2_590_, lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_f_595_, lean_object* v_hf__inj_596_, lean_object* v_le_597_, lean_object* v_lt_598_, lean_object* v_map__sup_599_){
_start:
{
lean_object* v___f_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
v___f_600_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_600_, 0, v_inst_591_);
v___x_601_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_601_, 0, v_inst_592_);
lean_ctor_set(v___x_601_, 1, v_inst_593_);
v___x_602_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_602_, 0, v___x_601_);
lean_ctor_set(v___x_602_, 1, v___f_600_);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeSup___boxed(lean_object* v_00_u03b1_603_, lean_object* v_00_u03b2_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_inst_607_, lean_object* v_inst_608_, lean_object* v_f_609_, lean_object* v_hf__inj_610_, lean_object* v_le_611_, lean_object* v_lt_612_, lean_object* v_map__sup_613_){
_start:
{
lean_object* v_res_614_; 
v_res_614_ = lp_mathlib_Function_Injective_semilatticeSup(v_00_u03b1_603_, v_00_u03b2_604_, v_inst_605_, v_inst_606_, v_inst_607_, v_inst_608_, v_f_609_, v_hf__inj_610_, v_le_611_, v_lt_612_, v_map__sup_613_);
lean_dec(v_f_609_);
lean_dec_ref(v_inst_608_);
return v_res_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object* v_inst_615_, lean_object* v_inst_616_, lean_object* v_inst_617_){
_start:
{
lean_object* v___f_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v___f_618_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_618_, 0, v_inst_615_);
v___x_619_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_619_, 0, v_inst_616_);
lean_ctor_set(v___x_619_, 1, v_inst_617_);
v___x_620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_620_, 0, v___x_619_);
lean_ctor_set(v___x_620_, 1, v___f_618_);
return v___x_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeInf(lean_object* v_00_u03b1_621_, lean_object* v_00_u03b2_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_inst_625_, lean_object* v_inst_626_, lean_object* v_f_627_, lean_object* v_hf__inj_628_, lean_object* v_le_629_, lean_object* v_lt_630_, lean_object* v_map__sup_631_){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_inst_623_, v_inst_624_, v_inst_625_);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semilatticeInf___boxed(lean_object* v_00_u03b1_633_, lean_object* v_00_u03b2_634_, lean_object* v_inst_635_, lean_object* v_inst_636_, lean_object* v_inst_637_, lean_object* v_inst_638_, lean_object* v_f_639_, lean_object* v_hf__inj_640_, lean_object* v_le_641_, lean_object* v_lt_642_, lean_object* v_map__sup_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_mathlib_Function_Injective_semilatticeInf(v_00_u03b1_633_, v_00_u03b2_634_, v_inst_635_, v_inst_636_, v_inst_637_, v_inst_638_, v_f_639_, v_hf__inj_640_, v_le_641_, v_lt_642_, v_map__sup_643_);
lean_dec(v_f_639_);
lean_dec_ref(v_inst_638_);
return v_res_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_lattice___redArg(lean_object* v_inst_645_, lean_object* v_inst_646_, lean_object* v_inst_647_, lean_object* v_inst_648_){
_start:
{
lean_object* v___f_649_; lean_object* v___f_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; 
v___f_649_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_649_, 0, v_inst_645_);
v___f_650_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_650_, 0, v_inst_646_);
v___x_651_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_651_, 0, v_inst_647_);
lean_ctor_set(v___x_651_, 1, v_inst_648_);
v___x_652_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_652_, 0, v___x_651_);
lean_ctor_set(v___x_652_, 1, v___f_649_);
v___x_653_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_653_, 0, v___x_652_);
lean_ctor_set(v___x_653_, 1, v___f_650_);
return v___x_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_lattice(lean_object* v_00_u03b1_654_, lean_object* v_00_u03b2_655_, lean_object* v_inst_656_, lean_object* v_inst_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_inst_660_, lean_object* v_f_661_, lean_object* v_hf__inj_662_, lean_object* v_le_663_, lean_object* v_lt_664_, lean_object* v_map__sup_665_, lean_object* v_map__inf_666_){
_start:
{
lean_object* v___f_667_; lean_object* v___f_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v___f_667_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_667_, 0, v_inst_656_);
v___f_668_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_668_, 0, v_inst_657_);
v___x_669_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_669_, 0, v_inst_658_);
lean_ctor_set(v___x_669_, 1, v_inst_659_);
v___x_670_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_670_, 0, v___x_669_);
lean_ctor_set(v___x_670_, 1, v___f_667_);
v___x_671_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_671_, 0, v___x_670_);
lean_ctor_set(v___x_671_, 1, v___f_668_);
return v___x_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_lattice___boxed(lean_object* v_00_u03b1_672_, lean_object* v_00_u03b2_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_inst_677_, lean_object* v_inst_678_, lean_object* v_f_679_, lean_object* v_hf__inj_680_, lean_object* v_le_681_, lean_object* v_lt_682_, lean_object* v_map__sup_683_, lean_object* v_map__inf_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_mathlib_Function_Injective_lattice(v_00_u03b1_672_, v_00_u03b2_673_, v_inst_674_, v_inst_675_, v_inst_676_, v_inst_677_, v_inst_678_, v_f_679_, v_hf__inj_680_, v_le_681_, v_lt_682_, v_map__sup_683_, v_map__inf_684_);
lean_dec(v_f_679_);
lean_dec_ref(v_inst_678_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribLattice___redArg(lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_inst_688_, lean_object* v_inst_689_){
_start:
{
lean_object* v___f_690_; lean_object* v___f_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; 
v___f_690_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_690_, 0, v_inst_686_);
v___f_691_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_691_, 0, v_inst_687_);
v___x_692_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_692_, 0, v_inst_688_);
lean_ctor_set(v___x_692_, 1, v_inst_689_);
v___x_693_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_693_, 0, v___x_692_);
lean_ctor_set(v___x_693_, 1, v___f_690_);
v___x_694_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_694_, 0, v___x_693_);
lean_ctor_set(v___x_694_, 1, v___f_691_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribLattice(lean_object* v_00_u03b1_695_, lean_object* v_00_u03b2_696_, lean_object* v_inst_697_, lean_object* v_inst_698_, lean_object* v_inst_699_, lean_object* v_inst_700_, lean_object* v_inst_701_, lean_object* v_f_702_, lean_object* v_hf__inj_703_, lean_object* v_le_704_, lean_object* v_lt_705_, lean_object* v_map__sup_706_, lean_object* v_map__inf_707_){
_start:
{
lean_object* v___f_708_; lean_object* v___f_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; 
v___f_708_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_708_, 0, v_inst_697_);
v___f_709_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_709_, 0, v_inst_698_);
v___x_710_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_710_, 0, v_inst_699_);
lean_ctor_set(v___x_710_, 1, v_inst_700_);
v___x_711_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_711_, 0, v___x_710_);
lean_ctor_set(v___x_711_, 1, v___f_708_);
v___x_712_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_712_, 0, v___x_711_);
lean_ctor_set(v___x_712_, 1, v___f_709_);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distribLattice___boxed(lean_object* v_00_u03b1_713_, lean_object* v_00_u03b2_714_, lean_object* v_inst_715_, lean_object* v_inst_716_, lean_object* v_inst_717_, lean_object* v_inst_718_, lean_object* v_inst_719_, lean_object* v_f_720_, lean_object* v_hf__inj_721_, lean_object* v_le_722_, lean_object* v_lt_723_, lean_object* v_map__sup_724_, lean_object* v_map__inf_725_){
_start:
{
lean_object* v_res_726_; 
v_res_726_ = lp_mathlib_Function_Injective_distribLattice(v_00_u03b1_713_, v_00_u03b2_714_, v_inst_715_, v_inst_716_, v_inst_717_, v_inst_718_, v_inst_719_, v_f_720_, v_hf__inj_721_, v_le_722_, v_lt_723_, v_map__sup_724_, v_map__inf_725_);
lean_dec(v_f_720_);
lean_dec_ref(v_inst_719_);
return v_res_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_distribLattice___redArg(lean_object* v_inst_727_){
_start:
{
lean_object* v_toSemilatticeSup_728_; lean_object* v_inf_729_; lean_object* v___x_731_; uint8_t v_isShared_732_; uint8_t v_isSharedCheck_740_; 
v_toSemilatticeSup_728_ = lean_ctor_get(v_inst_727_, 0);
v_inf_729_ = lean_ctor_get(v_inst_727_, 1);
v_isSharedCheck_740_ = !lean_is_exclusive(v_inst_727_);
if (v_isSharedCheck_740_ == 0)
{
v___x_731_ = v_inst_727_;
v_isShared_732_ = v_isSharedCheck_740_;
goto v_resetjp_730_;
}
else
{
lean_inc(v_inf_729_);
lean_inc(v_toSemilatticeSup_728_);
lean_dec(v_inst_727_);
v___x_731_ = lean_box(0);
v_isShared_732_ = v_isSharedCheck_740_;
goto v_resetjp_730_;
}
v_resetjp_730_:
{
lean_object* v___f_733_; lean_object* v___f_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_738_; 
v___f_733_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instSemilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_733_, 0, v_inf_729_);
v___f_734_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_734_, 0, v_toSemilatticeSup_728_);
v___x_735_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
v___x_736_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_736_, 0, v___x_735_);
lean_ctor_set(v___x_736_, 1, v___f_734_);
if (v_isShared_732_ == 0)
{
lean_ctor_set(v___x_731_, 1, v___f_733_);
lean_ctor_set(v___x_731_, 0, v___x_736_);
v___x_738_ = v___x_731_;
goto v_reusejp_737_;
}
else
{
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v___x_736_);
lean_ctor_set(v_reuseFailAlloc_739_, 1, v___f_733_);
v___x_738_ = v_reuseFailAlloc_739_;
goto v_reusejp_737_;
}
v_reusejp_737_:
{
return v___x_738_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_distribLattice(lean_object* v_00_u03b1_741_, lean_object* v_inst_742_, lean_object* v_P_743_, lean_object* v_Psup_744_, lean_object* v_Pinf_745_){
_start:
{
lean_object* v_toSemilatticeSup_746_; lean_object* v_inf_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_758_; 
v_toSemilatticeSup_746_ = lean_ctor_get(v_inst_742_, 0);
v_inf_747_ = lean_ctor_get(v_inst_742_, 1);
v_isSharedCheck_758_ = !lean_is_exclusive(v_inst_742_);
if (v_isSharedCheck_758_ == 0)
{
v___x_749_ = v_inst_742_;
v_isShared_750_ = v_isSharedCheck_758_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_inf_747_);
lean_inc(v_toSemilatticeSup_746_);
lean_dec(v_inst_742_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_758_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___f_751_; lean_object* v___f_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_756_; 
v___f_751_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instSemilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_751_, 0, v_inf_747_);
v___f_752_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_752_, 0, v_toSemilatticeSup_746_);
v___x_753_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
v___x_754_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_754_, 0, v___x_753_);
lean_ctor_set(v___x_754_, 1, v___f_752_);
if (v_isShared_750_ == 0)
{
lean_ctor_set(v___x_749_, 1, v___f_751_);
lean_ctor_set(v___x_749_, 0, v___x_754_);
v___x_756_ = v___x_749_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v___x_754_);
lean_ctor_set(v_reuseFailAlloc_757_, 1, v___f_751_);
v___x_756_ = v_reuseFailAlloc_757_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
return v___x_756_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_preorder(lean_object* v_00_u03b1_759_, lean_object* v_00_u03b2_760_, lean_object* v_e_761_, lean_object* v_inst_762_){
_start:
{
lean_object* v___x_763_; 
v___x_763_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
return v___x_763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_preorder___boxed(lean_object* v_00_u03b1_764_, lean_object* v_00_u03b2_765_, lean_object* v_e_766_, lean_object* v_inst_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_mathlib_Equiv_preorder(v_00_u03b1_764_, v_00_u03b2_765_, v_e_766_, v_inst_767_);
lean_dec_ref(v_inst_767_);
lean_dec_ref(v_e_766_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_partialOrder(lean_object* v_00_u03b1_769_, lean_object* v_00_u03b2_770_, lean_object* v_e_771_, lean_object* v_inst_772_){
_start:
{
lean_object* v___x_773_; 
v___x_773_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_partialOrder___boxed(lean_object* v_00_u03b1_774_, lean_object* v_00_u03b2_775_, lean_object* v_e_776_, lean_object* v_inst_777_){
_start:
{
lean_object* v_res_778_; 
v_res_778_ = lp_mathlib_Equiv_partialOrder(v_00_u03b1_774_, v_00_u03b2_775_, v_e_776_, v_inst_777_);
lean_dec_ref(v_inst_777_);
lean_dec_ref(v_e_776_);
return v_res_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__0(lean_object* v_self_779_, lean_object* v___y_780_){
_start:
{
lean_object* v_toFun_781_; lean_object* v___x_782_; 
v_toFun_781_ = lean_ctor_get(v_self_779_, 0);
lean_inc(v_toFun_781_);
lean_dec_ref(v_self_779_);
v___x_782_ = lean_apply_1(v_toFun_781_, v___y_780_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__1(lean_object* v_e_783_, lean_object* v___f_784_, lean_object* v_toMax_785_, lean_object* v_a_786_, lean_object* v_b_787_){
_start:
{
lean_object* v___x_788_; lean_object* v_toFun_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; 
lean_inc_ref_n(v_e_783_, 2);
v___x_788_ = lp_mathlib_Equiv_symm___redArg(v_e_783_);
v_toFun_789_ = lean_ctor_get(v___x_788_, 0);
lean_inc(v_toFun_789_);
lean_dec_ref(v___x_788_);
lean_inc(v___f_784_);
v___x_790_ = lean_apply_2(v___f_784_, v_e_783_, v_a_786_);
v___x_791_ = lean_apply_2(v___f_784_, v_e_783_, v_b_787_);
v___x_792_ = lean_apply_2(v_toMax_785_, v___x_790_, v___x_791_);
v___x_793_ = lean_apply_1(v_toFun_789_, v___x_792_);
return v___x_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__3(lean_object* v_e_794_, lean_object* v___f_795_, lean_object* v_toMin_796_, lean_object* v_a_797_, lean_object* v_b_798_){
_start:
{
lean_object* v___x_799_; lean_object* v_toFun_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; 
lean_inc_ref_n(v_e_794_, 2);
v___x_799_ = lp_mathlib_Equiv_symm___redArg(v_e_794_);
v_toFun_800_ = lean_ctor_get(v___x_799_, 0);
lean_inc(v_toFun_800_);
lean_dec_ref(v___x_799_);
lean_inc(v___f_795_);
v___x_801_ = lean_apply_2(v___f_795_, v_e_794_, v_a_797_);
v___x_802_ = lean_apply_2(v___f_795_, v_e_794_, v_b_798_);
v___x_803_ = lean_apply_2(v_toMin_796_, v___x_801_, v___x_802_);
v___x_804_ = lean_apply_1(v_toFun_800_, v___x_803_);
return v___x_804_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_linearOrder___redArg___lam__5(lean_object* v___f_805_, lean_object* v_e_806_, lean_object* v_toDecidableLE_807_, lean_object* v_a_808_, lean_object* v_b_809_){
_start:
{
lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; uint8_t v___x_813_; 
lean_inc(v___f_805_);
lean_inc_ref(v_e_806_);
v___x_810_ = lean_apply_2(v___f_805_, v_e_806_, v_a_808_);
v___x_811_ = lean_apply_2(v___f_805_, v_e_806_, v_b_809_);
v___x_812_ = lean_apply_2(v_toDecidableLE_807_, v___x_810_, v___x_811_);
v___x_813_ = lean_unbox(v___x_812_);
return v___x_813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__5___boxed(lean_object* v___f_814_, lean_object* v_e_815_, lean_object* v_toDecidableLE_816_, lean_object* v_a_817_, lean_object* v_b_818_){
_start:
{
uint8_t v_res_819_; lean_object* v_r_820_; 
v_res_819_ = lp_mathlib_Equiv_linearOrder___redArg___lam__5(v___f_814_, v_e_815_, v_toDecidableLE_816_, v_a_817_, v_b_818_);
v_r_820_ = lean_box(v_res_819_);
return v_r_820_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_linearOrder___redArg___lam__4(lean_object* v___f_821_, lean_object* v_e_822_, lean_object* v_toDecidableLT_823_, lean_object* v_a_824_, lean_object* v_b_825_){
_start:
{
lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; uint8_t v___x_829_; 
lean_inc(v___f_821_);
lean_inc_ref(v_e_822_);
v___x_826_ = lean_apply_2(v___f_821_, v_e_822_, v_a_824_);
v___x_827_ = lean_apply_2(v___f_821_, v_e_822_, v_b_825_);
v___x_828_ = lean_apply_2(v_toDecidableLT_823_, v___x_826_, v___x_827_);
v___x_829_ = lean_unbox(v___x_828_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__4___boxed(lean_object* v___f_830_, lean_object* v_e_831_, lean_object* v_toDecidableLT_832_, lean_object* v_a_833_, lean_object* v_b_834_){
_start:
{
uint8_t v_res_835_; lean_object* v_r_836_; 
v_res_835_ = lp_mathlib_Equiv_linearOrder___redArg___lam__4(v___f_830_, v_e_831_, v_toDecidableLT_832_, v_a_833_, v_b_834_);
v_r_836_ = lean_box(v_res_835_);
return v_r_836_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_linearOrder___redArg___lam__2(lean_object* v___f_837_, lean_object* v_e_838_, lean_object* v_toOrd_839_, lean_object* v_a_840_, lean_object* v_b_841_){
_start:
{
lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; uint8_t v___x_845_; 
lean_inc(v___f_837_);
lean_inc_ref(v_e_838_);
v___x_842_ = lean_apply_2(v___f_837_, v_e_838_, v_a_840_);
v___x_843_ = lean_apply_2(v___f_837_, v_e_838_, v_b_841_);
v___x_844_ = lean_apply_2(v_toOrd_839_, v___x_842_, v___x_843_);
v___x_845_ = lean_unbox(v___x_844_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg___lam__2___boxed(lean_object* v___f_846_, lean_object* v_e_847_, lean_object* v_toOrd_848_, lean_object* v_a_849_, lean_object* v_b_850_){
_start:
{
uint8_t v_res_851_; lean_object* v_r_852_; 
v_res_851_ = lp_mathlib_Equiv_linearOrder___redArg___lam__2(v___f_846_, v_e_847_, v_toOrd_848_, v_a_849_, v_b_850_);
v_r_852_ = lean_box(v_res_851_);
return v_r_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder___redArg(lean_object* v_e_854_, lean_object* v_inst_855_, lean_object* v_inst_856_){
_start:
{
lean_object* v_toMin_857_; lean_object* v_toMax_858_; lean_object* v_toOrd_859_; lean_object* v_toDecidableLE_860_; lean_object* v_toDecidableLT_861_; lean_object* v___x_863_; uint8_t v_isShared_864_; uint8_t v_isSharedCheck_875_; 
v_toMin_857_ = lean_ctor_get(v_inst_855_, 1);
v_toMax_858_ = lean_ctor_get(v_inst_855_, 2);
v_toOrd_859_ = lean_ctor_get(v_inst_855_, 3);
v_toDecidableLE_860_ = lean_ctor_get(v_inst_855_, 4);
v_toDecidableLT_861_ = lean_ctor_get(v_inst_855_, 6);
v_isSharedCheck_875_ = !lean_is_exclusive(v_inst_855_);
if (v_isSharedCheck_875_ == 0)
{
lean_object* v_unused_876_; lean_object* v_unused_877_; 
v_unused_876_ = lean_ctor_get(v_inst_855_, 5);
lean_dec(v_unused_876_);
v_unused_877_ = lean_ctor_get(v_inst_855_, 0);
lean_dec(v_unused_877_);
v___x_863_ = v_inst_855_;
v_isShared_864_ = v_isSharedCheck_875_;
goto v_resetjp_862_;
}
else
{
lean_inc(v_toDecidableLT_861_);
lean_inc(v_toDecidableLE_860_);
lean_inc(v_toOrd_859_);
lean_inc(v_toMax_858_);
lean_inc(v_toMin_857_);
lean_dec(v_inst_855_);
v___x_863_ = lean_box(0);
v_isShared_864_ = v_isSharedCheck_875_;
goto v_resetjp_862_;
}
v_resetjp_862_:
{
lean_object* v___f_865_; lean_object* v_max_866_; lean_object* v_min_867_; lean_object* v___f_868_; lean_object* v___f_869_; lean_object* v_compare_870_; lean_object* v___x_871_; lean_object* v___x_873_; 
v___f_865_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
lean_inc_ref_n(v_e_854_, 4);
v_max_866_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__1), 5, 3);
lean_closure_set(v_max_866_, 0, v_e_854_);
lean_closure_set(v_max_866_, 1, v___f_865_);
lean_closure_set(v_max_866_, 2, v_toMax_858_);
v_min_867_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__3), 5, 3);
lean_closure_set(v_min_867_, 0, v_e_854_);
lean_closure_set(v_min_867_, 1, v___f_865_);
lean_closure_set(v_min_867_, 2, v_toMin_857_);
v___f_868_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__5___boxed), 5, 3);
lean_closure_set(v___f_868_, 0, v___f_865_);
lean_closure_set(v___f_868_, 1, v_e_854_);
lean_closure_set(v___f_868_, 2, v_toDecidableLE_860_);
v___f_869_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__4___boxed), 5, 3);
lean_closure_set(v___f_869_, 0, v___f_865_);
lean_closure_set(v___f_869_, 1, v_e_854_);
lean_closure_set(v___f_869_, 2, v_toDecidableLT_861_);
v_compare_870_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__2___boxed), 5, 3);
lean_closure_set(v_compare_870_, 0, v___f_865_);
lean_closure_set(v_compare_870_, 1, v_e_854_);
lean_closure_set(v_compare_870_, 2, v_toOrd_859_);
v___x_871_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
if (v_isShared_864_ == 0)
{
lean_ctor_set(v___x_863_, 6, v___f_869_);
lean_ctor_set(v___x_863_, 5, v_inst_856_);
lean_ctor_set(v___x_863_, 4, v___f_868_);
lean_ctor_set(v___x_863_, 3, v_compare_870_);
lean_ctor_set(v___x_863_, 2, v_max_866_);
lean_ctor_set(v___x_863_, 1, v_min_867_);
lean_ctor_set(v___x_863_, 0, v___x_871_);
v___x_873_ = v___x_863_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v___x_871_);
lean_ctor_set(v_reuseFailAlloc_874_, 1, v_min_867_);
lean_ctor_set(v_reuseFailAlloc_874_, 2, v_max_866_);
lean_ctor_set(v_reuseFailAlloc_874_, 3, v_compare_870_);
lean_ctor_set(v_reuseFailAlloc_874_, 4, v___f_868_);
lean_ctor_set(v_reuseFailAlloc_874_, 5, v_inst_856_);
lean_ctor_set(v_reuseFailAlloc_874_, 6, v___f_869_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearOrder(lean_object* v_00_u03b1_878_, lean_object* v_00_u03b2_879_, lean_object* v_e_880_, lean_object* v_inst_881_, lean_object* v_inst_882_){
_start:
{
lean_object* v_toMin_883_; lean_object* v_toMax_884_; lean_object* v_toOrd_885_; lean_object* v_toDecidableLE_886_; lean_object* v_toDecidableLT_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_901_; 
v_toMin_883_ = lean_ctor_get(v_inst_881_, 1);
v_toMax_884_ = lean_ctor_get(v_inst_881_, 2);
v_toOrd_885_ = lean_ctor_get(v_inst_881_, 3);
v_toDecidableLE_886_ = lean_ctor_get(v_inst_881_, 4);
v_toDecidableLT_887_ = lean_ctor_get(v_inst_881_, 6);
v_isSharedCheck_901_ = !lean_is_exclusive(v_inst_881_);
if (v_isSharedCheck_901_ == 0)
{
lean_object* v_unused_902_; lean_object* v_unused_903_; 
v_unused_902_ = lean_ctor_get(v_inst_881_, 5);
lean_dec(v_unused_902_);
v_unused_903_ = lean_ctor_get(v_inst_881_, 0);
lean_dec(v_unused_903_);
v___x_889_ = v_inst_881_;
v_isShared_890_ = v_isSharedCheck_901_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_toDecidableLT_887_);
lean_inc(v_toDecidableLE_886_);
lean_inc(v_toOrd_885_);
lean_inc(v_toMax_884_);
lean_inc(v_toMin_883_);
lean_dec(v_inst_881_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_901_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v___f_891_; lean_object* v_max_892_; lean_object* v_min_893_; lean_object* v___f_894_; lean_object* v___f_895_; lean_object* v_compare_896_; lean_object* v___x_897_; lean_object* v___x_899_; 
v___f_891_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
lean_inc_ref_n(v_e_880_, 4);
v_max_892_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__1), 5, 3);
lean_closure_set(v_max_892_, 0, v_e_880_);
lean_closure_set(v_max_892_, 1, v___f_891_);
lean_closure_set(v_max_892_, 2, v_toMax_884_);
v_min_893_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__3), 5, 3);
lean_closure_set(v_min_893_, 0, v_e_880_);
lean_closure_set(v_min_893_, 1, v___f_891_);
lean_closure_set(v_min_893_, 2, v_toMin_883_);
v___f_894_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__5___boxed), 5, 3);
lean_closure_set(v___f_894_, 0, v___f_891_);
lean_closure_set(v___f_894_, 1, v_e_880_);
lean_closure_set(v___f_894_, 2, v_toDecidableLE_886_);
v___f_895_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__4___boxed), 5, 3);
lean_closure_set(v___f_895_, 0, v___f_891_);
lean_closure_set(v___f_895_, 1, v_e_880_);
lean_closure_set(v___f_895_, 2, v_toDecidableLT_887_);
v_compare_896_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_linearOrder___redArg___lam__2___boxed), 5, 3);
lean_closure_set(v_compare_896_, 0, v___f_891_);
lean_closure_set(v_compare_896_, 1, v_e_880_);
lean_closure_set(v_compare_896_, 2, v_toOrd_885_);
v___x_897_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
if (v_isShared_890_ == 0)
{
lean_ctor_set(v___x_889_, 6, v___f_895_);
lean_ctor_set(v___x_889_, 5, v_inst_882_);
lean_ctor_set(v___x_889_, 4, v___f_894_);
lean_ctor_set(v___x_889_, 3, v_compare_896_);
lean_ctor_set(v___x_889_, 2, v_max_892_);
lean_ctor_set(v___x_889_, 1, v_min_893_);
lean_ctor_set(v___x_889_, 0, v___x_897_);
v___x_899_ = v___x_889_;
goto v_reusejp_898_;
}
else
{
lean_object* v_reuseFailAlloc_900_; 
v_reuseFailAlloc_900_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_900_, 0, v___x_897_);
lean_ctor_set(v_reuseFailAlloc_900_, 1, v_min_893_);
lean_ctor_set(v_reuseFailAlloc_900_, 2, v_max_892_);
lean_ctor_set(v_reuseFailAlloc_900_, 3, v_compare_896_);
lean_ctor_set(v_reuseFailAlloc_900_, 4, v___f_894_);
lean_ctor_set(v_reuseFailAlloc_900_, 5, v_inst_882_);
lean_ctor_set(v_reuseFailAlloc_900_, 6, v___f_895_);
v___x_899_ = v_reuseFailAlloc_900_;
goto v_reusejp_898_;
}
v_reusejp_898_:
{
return v___x_899_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeSup___redArg___lam__1(lean_object* v_e_904_, lean_object* v_inst_905_, lean_object* v___f_906_, lean_object* v_a_907_, lean_object* v_b_908_){
_start:
{
lean_object* v___x_909_; lean_object* v_sup_910_; lean_object* v_toFun_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; 
lean_inc_ref_n(v_e_904_, 2);
v___x_909_ = lp_mathlib_Equiv_symm___redArg(v_e_904_);
v_sup_910_ = lean_ctor_get(v_inst_905_, 1);
lean_inc(v_sup_910_);
lean_dec_ref(v_inst_905_);
v_toFun_911_ = lean_ctor_get(v___x_909_, 0);
lean_inc(v_toFun_911_);
lean_dec_ref(v___x_909_);
lean_inc(v___f_906_);
v___x_912_ = lean_apply_2(v___f_906_, v_e_904_, v_a_907_);
v___x_913_ = lean_apply_2(v___f_906_, v_e_904_, v_b_908_);
v___x_914_ = lean_apply_2(v_sup_910_, v___x_912_, v___x_913_);
v___x_915_ = lean_apply_1(v_toFun_911_, v___x_914_);
return v___x_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeSup___redArg(lean_object* v_e_916_, lean_object* v_inst_917_){
_start:
{
lean_object* v___f_918_; lean_object* v___f_919_; lean_object* v___x_920_; lean_object* v___x_921_; 
v___f_918_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
v___f_919_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_semilatticeSup___redArg___lam__1), 5, 3);
lean_closure_set(v___f_919_, 0, v_e_916_);
lean_closure_set(v___f_919_, 1, v_inst_917_);
lean_closure_set(v___f_919_, 2, v___f_918_);
v___x_920_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
v___x_921_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_921_, 0, v___x_920_);
lean_ctor_set(v___x_921_, 1, v___f_919_);
return v___x_921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeSup(lean_object* v_00_u03b1_922_, lean_object* v_00_u03b2_923_, lean_object* v_e_924_, lean_object* v_inst_925_){
_start:
{
lean_object* v___f_926_; lean_object* v___f_927_; lean_object* v___x_928_; lean_object* v___x_929_; 
v___f_926_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
v___f_927_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_semilatticeSup___redArg___lam__1), 5, 3);
lean_closure_set(v___f_927_, 0, v_e_924_);
lean_closure_set(v___f_927_, 1, v_inst_925_);
lean_closure_set(v___f_927_, 2, v___f_926_);
v___x_928_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
v___x_929_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_929_, 0, v___x_928_);
lean_ctor_set(v___x_929_, 1, v___f_927_);
return v___x_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeInf___redArg___lam__1(lean_object* v_e_930_, lean_object* v_inst_931_, lean_object* v___f_932_, lean_object* v_a_933_, lean_object* v_b_934_){
_start:
{
lean_object* v___x_935_; lean_object* v_inf_936_; lean_object* v_toFun_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; 
lean_inc_ref_n(v_e_930_, 2);
v___x_935_ = lp_mathlib_Equiv_symm___redArg(v_e_930_);
v_inf_936_ = lean_ctor_get(v_inst_931_, 1);
lean_inc(v_inf_936_);
lean_dec_ref(v_inst_931_);
v_toFun_937_ = lean_ctor_get(v___x_935_, 0);
lean_inc(v_toFun_937_);
lean_dec_ref(v___x_935_);
lean_inc(v___f_932_);
v___x_938_ = lean_apply_2(v___f_932_, v_e_930_, v_a_933_);
v___x_939_ = lean_apply_2(v___f_932_, v_e_930_, v_b_934_);
v___x_940_ = lean_apply_2(v_inf_936_, v___x_938_, v___x_939_);
v___x_941_ = lean_apply_1(v_toFun_937_, v___x_940_);
return v___x_941_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeInf___redArg(lean_object* v_e_942_, lean_object* v_inst_943_){
_start:
{
lean_object* v___f_944_; lean_object* v_min_945_; lean_object* v_le_946_; lean_object* v_lt_947_; lean_object* v___x_948_; 
v___f_944_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
v_min_945_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_semilatticeInf___redArg___lam__1), 5, 3);
lean_closure_set(v_min_945_, 0, v_e_942_);
lean_closure_set(v_min_945_, 1, v_inst_943_);
lean_closure_set(v_min_945_, 2, v___f_944_);
v_le_946_ = lean_box(0);
v_lt_947_ = lean_box(0);
v___x_948_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_945_, v_le_946_, v_lt_947_);
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_semilatticeInf(lean_object* v_00_u03b1_949_, lean_object* v_00_u03b2_950_, lean_object* v_e_951_, lean_object* v_inst_952_){
_start:
{
lean_object* v___f_953_; lean_object* v_min_954_; lean_object* v_le_955_; lean_object* v_lt_956_; lean_object* v___x_957_; 
v___f_953_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
v_min_954_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_semilatticeInf___redArg___lam__1), 5, 3);
lean_closure_set(v_min_954_, 0, v_e_951_);
lean_closure_set(v_min_954_, 1, v_inst_952_);
lean_closure_set(v_min_954_, 2, v___f_953_);
v_le_955_ = lean_box(0);
v_lt_956_ = lean_box(0);
v___x_957_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_954_, v_le_955_, v_lt_956_);
return v___x_957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice___redArg___lam__1(lean_object* v_e_958_, lean_object* v___f_959_, lean_object* v_inf_960_, lean_object* v_a_961_, lean_object* v_b_962_){
_start:
{
lean_object* v___x_963_; lean_object* v_toFun_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; 
lean_inc_ref_n(v_e_958_, 2);
v___x_963_ = lp_mathlib_Equiv_symm___redArg(v_e_958_);
v_toFun_964_ = lean_ctor_get(v___x_963_, 0);
lean_inc(v_toFun_964_);
lean_dec_ref(v___x_963_);
lean_inc(v___f_959_);
v___x_965_ = lean_apply_2(v___f_959_, v_e_958_, v_a_961_);
v___x_966_ = lean_apply_2(v___f_959_, v_e_958_, v_b_962_);
v___x_967_ = lean_apply_2(v_inf_960_, v___x_965_, v___x_966_);
v___x_968_ = lean_apply_1(v_toFun_964_, v___x_967_);
return v___x_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice___redArg___lam__2(lean_object* v_min_969_, lean_object* v_a_970_, lean_object* v_b_971_){
_start:
{
lean_object* v___x_972_; 
v___x_972_ = lean_apply_2(v_min_969_, v_a_970_, v_b_971_);
return v___x_972_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice___redArg___lam__0(lean_object* v_toSemilatticeSup_973_, lean_object* v_e_974_, lean_object* v___f_975_, lean_object* v_a_976_, lean_object* v_b_977_){
_start:
{
lean_object* v_sup_978_; lean_object* v___x_979_; lean_object* v_toFun_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; 
v_sup_978_ = lean_ctor_get(v_toSemilatticeSup_973_, 1);
lean_inc(v_sup_978_);
lean_dec_ref(v_toSemilatticeSup_973_);
lean_inc_ref_n(v_e_974_, 2);
v___x_979_ = lp_mathlib_Equiv_symm___redArg(v_e_974_);
v_toFun_980_ = lean_ctor_get(v___x_979_, 0);
lean_inc(v_toFun_980_);
lean_dec_ref(v___x_979_);
lean_inc(v___f_975_);
v___x_981_ = lean_apply_2(v___f_975_, v_e_974_, v_a_976_);
v___x_982_ = lean_apply_2(v___f_975_, v_e_974_, v_b_977_);
v___x_983_ = lean_apply_2(v_sup_978_, v___x_981_, v___x_982_);
v___x_984_ = lean_apply_1(v_toFun_980_, v___x_983_);
return v___x_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice___redArg(lean_object* v_e_985_, lean_object* v_inst_986_){
_start:
{
lean_object* v_toSemilatticeSup_987_; lean_object* v_inf_988_; lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_1020_; 
v_toSemilatticeSup_987_ = lean_ctor_get(v_inst_986_, 0);
v_inf_988_ = lean_ctor_get(v_inst_986_, 1);
v_isSharedCheck_1020_ = !lean_is_exclusive(v_inst_986_);
if (v_isSharedCheck_1020_ == 0)
{
v___x_990_ = v_inst_986_;
v_isShared_991_ = v_isSharedCheck_1020_;
goto v_resetjp_989_;
}
else
{
lean_inc(v_inf_988_);
lean_inc(v_toSemilatticeSup_987_);
lean_dec(v_inst_986_);
v___x_990_ = lean_box(0);
v_isShared_991_ = v_isSharedCheck_1020_;
goto v_resetjp_989_;
}
v_resetjp_989_:
{
lean_object* v___f_992_; lean_object* v_min_993_; lean_object* v_le_994_; lean_object* v_lt_995_; lean_object* v_semilatticeInf_996_; lean_object* v_toPartialOrder_997_; lean_object* v___x_999_; uint8_t v_isShared_1000_; uint8_t v_isSharedCheck_1018_; 
v___f_992_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
lean_inc_ref(v_e_985_);
v_min_993_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__1), 5, 3);
lean_closure_set(v_min_993_, 0, v_e_985_);
lean_closure_set(v_min_993_, 1, v___f_992_);
lean_closure_set(v_min_993_, 2, v_inf_988_);
v_le_994_ = lean_box(0);
v_lt_995_ = lean_box(0);
lean_inc_ref(v_min_993_);
v_semilatticeInf_996_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_993_, v_le_994_, v_lt_995_);
v_toPartialOrder_997_ = lean_ctor_get(v_semilatticeInf_996_, 0);
v_isSharedCheck_1018_ = !lean_is_exclusive(v_semilatticeInf_996_);
if (v_isSharedCheck_1018_ == 0)
{
lean_object* v_unused_1019_; 
v_unused_1019_ = lean_ctor_get(v_semilatticeInf_996_, 1);
lean_dec(v_unused_1019_);
v___x_999_ = v_semilatticeInf_996_;
v_isShared_1000_ = v_isSharedCheck_1018_;
goto v_resetjp_998_;
}
else
{
lean_inc(v_toPartialOrder_997_);
lean_dec(v_semilatticeInf_996_);
v___x_999_ = lean_box(0);
v_isShared_1000_ = v_isSharedCheck_1018_;
goto v_resetjp_998_;
}
v_resetjp_998_:
{
lean_object* v_toLE_1001_; lean_object* v_toLT_1002_; lean_object* v___x_1004_; uint8_t v_isShared_1005_; uint8_t v_isSharedCheck_1017_; 
v_toLE_1001_ = lean_ctor_get(v_toPartialOrder_997_, 0);
v_toLT_1002_ = lean_ctor_get(v_toPartialOrder_997_, 1);
v_isSharedCheck_1017_ = !lean_is_exclusive(v_toPartialOrder_997_);
if (v_isSharedCheck_1017_ == 0)
{
v___x_1004_ = v_toPartialOrder_997_;
v_isShared_1005_ = v_isSharedCheck_1017_;
goto v_resetjp_1003_;
}
else
{
lean_inc(v_toLT_1002_);
lean_inc(v_toLE_1001_);
lean_dec(v_toPartialOrder_997_);
v___x_1004_ = lean_box(0);
v_isShared_1005_ = v_isSharedCheck_1017_;
goto v_resetjp_1003_;
}
v_resetjp_1003_:
{
lean_object* v___f_1006_; lean_object* v___f_1007_; lean_object* v___x_1009_; 
v___f_1006_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1006_, 0, v_min_993_);
v___f_1007_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1007_, 0, v_toSemilatticeSup_987_);
lean_closure_set(v___f_1007_, 1, v_e_985_);
lean_closure_set(v___f_1007_, 2, v___f_992_);
if (v_isShared_1005_ == 0)
{
v___x_1009_ = v___x_1004_;
goto v_reusejp_1008_;
}
else
{
lean_object* v_reuseFailAlloc_1016_; 
v_reuseFailAlloc_1016_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1016_, 0, v_toLE_1001_);
lean_ctor_set(v_reuseFailAlloc_1016_, 1, v_toLT_1002_);
v___x_1009_ = v_reuseFailAlloc_1016_;
goto v_reusejp_1008_;
}
v_reusejp_1008_:
{
lean_object* v___x_1011_; 
if (v_isShared_1000_ == 0)
{
lean_ctor_set(v___x_999_, 1, v___f_1007_);
lean_ctor_set(v___x_999_, 0, v___x_1009_);
v___x_1011_ = v___x_999_;
goto v_reusejp_1010_;
}
else
{
lean_object* v_reuseFailAlloc_1015_; 
v_reuseFailAlloc_1015_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1015_, 0, v___x_1009_);
lean_ctor_set(v_reuseFailAlloc_1015_, 1, v___f_1007_);
v___x_1011_ = v_reuseFailAlloc_1015_;
goto v_reusejp_1010_;
}
v_reusejp_1010_:
{
lean_object* v___x_1013_; 
if (v_isShared_991_ == 0)
{
lean_ctor_set(v___x_990_, 1, v___f_1006_);
lean_ctor_set(v___x_990_, 0, v___x_1011_);
v___x_1013_ = v___x_990_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1014_; 
v_reuseFailAlloc_1014_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1014_, 0, v___x_1011_);
lean_ctor_set(v_reuseFailAlloc_1014_, 1, v___f_1006_);
v___x_1013_ = v_reuseFailAlloc_1014_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
return v___x_1013_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_lattice(lean_object* v_00_u03b1_1021_, lean_object* v_00_u03b2_1022_, lean_object* v_e_1023_, lean_object* v_inst_1024_){
_start:
{
lean_object* v_toSemilatticeSup_1025_; lean_object* v_inf_1026_; lean_object* v___x_1028_; uint8_t v_isShared_1029_; uint8_t v_isSharedCheck_1058_; 
v_toSemilatticeSup_1025_ = lean_ctor_get(v_inst_1024_, 0);
v_inf_1026_ = lean_ctor_get(v_inst_1024_, 1);
v_isSharedCheck_1058_ = !lean_is_exclusive(v_inst_1024_);
if (v_isSharedCheck_1058_ == 0)
{
v___x_1028_ = v_inst_1024_;
v_isShared_1029_ = v_isSharedCheck_1058_;
goto v_resetjp_1027_;
}
else
{
lean_inc(v_inf_1026_);
lean_inc(v_toSemilatticeSup_1025_);
lean_dec(v_inst_1024_);
v___x_1028_ = lean_box(0);
v_isShared_1029_ = v_isSharedCheck_1058_;
goto v_resetjp_1027_;
}
v_resetjp_1027_:
{
lean_object* v___f_1030_; lean_object* v_min_1031_; lean_object* v_le_1032_; lean_object* v_lt_1033_; lean_object* v_semilatticeInf_1034_; lean_object* v_toPartialOrder_1035_; lean_object* v___x_1037_; uint8_t v_isShared_1038_; uint8_t v_isSharedCheck_1056_; 
v___f_1030_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
lean_inc_ref(v_e_1023_);
v_min_1031_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1031_, 0, v_e_1023_);
lean_closure_set(v_min_1031_, 1, v___f_1030_);
lean_closure_set(v_min_1031_, 2, v_inf_1026_);
v_le_1032_ = lean_box(0);
v_lt_1033_ = lean_box(0);
lean_inc_ref(v_min_1031_);
v_semilatticeInf_1034_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1031_, v_le_1032_, v_lt_1033_);
v_toPartialOrder_1035_ = lean_ctor_get(v_semilatticeInf_1034_, 0);
v_isSharedCheck_1056_ = !lean_is_exclusive(v_semilatticeInf_1034_);
if (v_isSharedCheck_1056_ == 0)
{
lean_object* v_unused_1057_; 
v_unused_1057_ = lean_ctor_get(v_semilatticeInf_1034_, 1);
lean_dec(v_unused_1057_);
v___x_1037_ = v_semilatticeInf_1034_;
v_isShared_1038_ = v_isSharedCheck_1056_;
goto v_resetjp_1036_;
}
else
{
lean_inc(v_toPartialOrder_1035_);
lean_dec(v_semilatticeInf_1034_);
v___x_1037_ = lean_box(0);
v_isShared_1038_ = v_isSharedCheck_1056_;
goto v_resetjp_1036_;
}
v_resetjp_1036_:
{
lean_object* v_toLE_1039_; lean_object* v_toLT_1040_; lean_object* v___x_1042_; uint8_t v_isShared_1043_; uint8_t v_isSharedCheck_1055_; 
v_toLE_1039_ = lean_ctor_get(v_toPartialOrder_1035_, 0);
v_toLT_1040_ = lean_ctor_get(v_toPartialOrder_1035_, 1);
v_isSharedCheck_1055_ = !lean_is_exclusive(v_toPartialOrder_1035_);
if (v_isSharedCheck_1055_ == 0)
{
v___x_1042_ = v_toPartialOrder_1035_;
v_isShared_1043_ = v_isSharedCheck_1055_;
goto v_resetjp_1041_;
}
else
{
lean_inc(v_toLT_1040_);
lean_inc(v_toLE_1039_);
lean_dec(v_toPartialOrder_1035_);
v___x_1042_ = lean_box(0);
v_isShared_1043_ = v_isSharedCheck_1055_;
goto v_resetjp_1041_;
}
v_resetjp_1041_:
{
lean_object* v___f_1044_; lean_object* v___f_1045_; lean_object* v___x_1047_; 
v___f_1044_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1044_, 0, v_min_1031_);
v___f_1045_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1045_, 0, v_toSemilatticeSup_1025_);
lean_closure_set(v___f_1045_, 1, v_e_1023_);
lean_closure_set(v___f_1045_, 2, v___f_1030_);
if (v_isShared_1043_ == 0)
{
v___x_1047_ = v___x_1042_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v_toLE_1039_);
lean_ctor_set(v_reuseFailAlloc_1054_, 1, v_toLT_1040_);
v___x_1047_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
lean_object* v___x_1049_; 
if (v_isShared_1038_ == 0)
{
lean_ctor_set(v___x_1037_, 1, v___f_1045_);
lean_ctor_set(v___x_1037_, 0, v___x_1047_);
v___x_1049_ = v___x_1037_;
goto v_reusejp_1048_;
}
else
{
lean_object* v_reuseFailAlloc_1053_; 
v_reuseFailAlloc_1053_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1053_, 0, v___x_1047_);
lean_ctor_set(v_reuseFailAlloc_1053_, 1, v___f_1045_);
v___x_1049_ = v_reuseFailAlloc_1053_;
goto v_reusejp_1048_;
}
v_reusejp_1048_:
{
lean_object* v___x_1051_; 
if (v_isShared_1029_ == 0)
{
lean_ctor_set(v___x_1028_, 1, v___f_1044_);
lean_ctor_set(v___x_1028_, 0, v___x_1049_);
v___x_1051_ = v___x_1028_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v___x_1049_);
lean_ctor_set(v_reuseFailAlloc_1052_, 1, v___f_1044_);
v___x_1051_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
return v___x_1051_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribLattice___redArg___lam__6(lean_object* v___f_1059_, lean_object* v_a_1060_, lean_object* v_b_1061_){
_start:
{
lean_object* v___x_1062_; 
v___x_1062_ = lean_apply_2(v___f_1059_, v_a_1060_, v_b_1061_);
return v___x_1062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribLattice___redArg(lean_object* v_e_1063_, lean_object* v_inst_1064_){
_start:
{
lean_object* v_toSemilatticeSup_1065_; lean_object* v_inf_1066_; lean_object* v___x_1068_; uint8_t v_isShared_1069_; uint8_t v_isSharedCheck_1119_; 
v_toSemilatticeSup_1065_ = lean_ctor_get(v_inst_1064_, 0);
v_inf_1066_ = lean_ctor_get(v_inst_1064_, 1);
v_isSharedCheck_1119_ = !lean_is_exclusive(v_inst_1064_);
if (v_isSharedCheck_1119_ == 0)
{
v___x_1068_ = v_inst_1064_;
v_isShared_1069_ = v_isSharedCheck_1119_;
goto v_resetjp_1067_;
}
else
{
lean_inc(v_inf_1066_);
lean_inc(v_toSemilatticeSup_1065_);
lean_dec(v_inst_1064_);
v___x_1068_ = lean_box(0);
v_isShared_1069_ = v_isSharedCheck_1119_;
goto v_resetjp_1067_;
}
v_resetjp_1067_:
{
lean_object* v___f_1070_; lean_object* v_min_1071_; lean_object* v_le_1072_; lean_object* v_lt_1073_; lean_object* v_semilatticeInf_1074_; lean_object* v_toPartialOrder_1075_; lean_object* v___x_1077_; uint8_t v_isShared_1078_; uint8_t v_isSharedCheck_1117_; 
v___f_1070_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
lean_inc_ref(v_e_1063_);
v_min_1071_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1071_, 0, v_e_1063_);
lean_closure_set(v_min_1071_, 1, v___f_1070_);
lean_closure_set(v_min_1071_, 2, v_inf_1066_);
v_le_1072_ = lean_box(0);
v_lt_1073_ = lean_box(0);
lean_inc_ref(v_min_1071_);
v_semilatticeInf_1074_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1071_, v_le_1072_, v_lt_1073_);
v_toPartialOrder_1075_ = lean_ctor_get(v_semilatticeInf_1074_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v_semilatticeInf_1074_);
if (v_isSharedCheck_1117_ == 0)
{
lean_object* v_unused_1118_; 
v_unused_1118_ = lean_ctor_get(v_semilatticeInf_1074_, 1);
lean_dec(v_unused_1118_);
v___x_1077_ = v_semilatticeInf_1074_;
v_isShared_1078_ = v_isSharedCheck_1117_;
goto v_resetjp_1076_;
}
else
{
lean_inc(v_toPartialOrder_1075_);
lean_dec(v_semilatticeInf_1074_);
v___x_1077_ = lean_box(0);
v_isShared_1078_ = v_isSharedCheck_1117_;
goto v_resetjp_1076_;
}
v_resetjp_1076_:
{
lean_object* v_toLE_1079_; lean_object* v_toLT_1080_; lean_object* v___x_1082_; uint8_t v_isShared_1083_; uint8_t v_isSharedCheck_1116_; 
v_toLE_1079_ = lean_ctor_get(v_toPartialOrder_1075_, 0);
v_toLT_1080_ = lean_ctor_get(v_toPartialOrder_1075_, 1);
v_isSharedCheck_1116_ = !lean_is_exclusive(v_toPartialOrder_1075_);
if (v_isSharedCheck_1116_ == 0)
{
v___x_1082_ = v_toPartialOrder_1075_;
v_isShared_1083_ = v_isSharedCheck_1116_;
goto v_resetjp_1081_;
}
else
{
lean_inc(v_toLT_1080_);
lean_inc(v_toLE_1079_);
lean_dec(v_toPartialOrder_1075_);
v___x_1082_ = lean_box(0);
v_isShared_1083_ = v_isSharedCheck_1116_;
goto v_resetjp_1081_;
}
v_resetjp_1081_:
{
lean_object* v___f_1084_; lean_object* v___f_1085_; lean_object* v___x_1087_; 
v___f_1084_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1084_, 0, v_min_1071_);
v___f_1085_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1085_, 0, v_toSemilatticeSup_1065_);
lean_closure_set(v___f_1085_, 1, v_e_1063_);
lean_closure_set(v___f_1085_, 2, v___f_1070_);
if (v_isShared_1083_ == 0)
{
v___x_1087_ = v___x_1082_;
goto v_reusejp_1086_;
}
else
{
lean_object* v_reuseFailAlloc_1115_; 
v_reuseFailAlloc_1115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1115_, 0, v_toLE_1079_);
lean_ctor_set(v_reuseFailAlloc_1115_, 1, v_toLT_1080_);
v___x_1087_ = v_reuseFailAlloc_1115_;
goto v_reusejp_1086_;
}
v_reusejp_1086_:
{
lean_object* v___x_1089_; 
lean_inc_ref(v___f_1085_);
if (v_isShared_1078_ == 0)
{
lean_ctor_set(v___x_1077_, 1, v___f_1085_);
lean_ctor_set(v___x_1077_, 0, v___x_1087_);
v___x_1089_ = v___x_1077_;
goto v_reusejp_1088_;
}
else
{
lean_object* v_reuseFailAlloc_1114_; 
v_reuseFailAlloc_1114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1114_, 0, v___x_1087_);
lean_ctor_set(v_reuseFailAlloc_1114_, 1, v___f_1085_);
v___x_1089_ = v_reuseFailAlloc_1114_;
goto v_reusejp_1088_;
}
v_reusejp_1088_:
{
lean_object* v_lattice_1091_; 
lean_inc_ref(v___f_1084_);
if (v_isShared_1069_ == 0)
{
lean_ctor_set(v___x_1068_, 1, v___f_1084_);
lean_ctor_set(v___x_1068_, 0, v___x_1089_);
v_lattice_1091_ = v___x_1068_;
goto v_reusejp_1090_;
}
else
{
lean_object* v_reuseFailAlloc_1113_; 
v_reuseFailAlloc_1113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1113_, 0, v___x_1089_);
lean_ctor_set(v_reuseFailAlloc_1113_, 1, v___f_1084_);
v_lattice_1091_ = v_reuseFailAlloc_1113_;
goto v_reusejp_1090_;
}
v_reusejp_1090_:
{
lean_object* v___x_1092_; lean_object* v_toPartialOrder_1093_; lean_object* v___x_1095_; uint8_t v_isShared_1096_; uint8_t v_isSharedCheck_1111_; 
v___x_1092_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1091_);
v_toPartialOrder_1093_ = lean_ctor_get(v___x_1092_, 0);
v_isSharedCheck_1111_ = !lean_is_exclusive(v___x_1092_);
if (v_isSharedCheck_1111_ == 0)
{
lean_object* v_unused_1112_; 
v_unused_1112_ = lean_ctor_get(v___x_1092_, 1);
lean_dec(v_unused_1112_);
v___x_1095_ = v___x_1092_;
v_isShared_1096_ = v_isSharedCheck_1111_;
goto v_resetjp_1094_;
}
else
{
lean_inc(v_toPartialOrder_1093_);
lean_dec(v___x_1092_);
v___x_1095_ = lean_box(0);
v_isShared_1096_ = v_isSharedCheck_1111_;
goto v_resetjp_1094_;
}
v_resetjp_1094_:
{
lean_object* v_toLE_1097_; lean_object* v_toLT_1098_; lean_object* v___x_1100_; uint8_t v_isShared_1101_; uint8_t v_isSharedCheck_1110_; 
v_toLE_1097_ = lean_ctor_get(v_toPartialOrder_1093_, 0);
v_toLT_1098_ = lean_ctor_get(v_toPartialOrder_1093_, 1);
v_isSharedCheck_1110_ = !lean_is_exclusive(v_toPartialOrder_1093_);
if (v_isSharedCheck_1110_ == 0)
{
v___x_1100_ = v_toPartialOrder_1093_;
v_isShared_1101_ = v_isSharedCheck_1110_;
goto v_resetjp_1099_;
}
else
{
lean_inc(v_toLT_1098_);
lean_inc(v_toLE_1097_);
lean_dec(v_toPartialOrder_1093_);
v___x_1100_ = lean_box(0);
v_isShared_1101_ = v_isSharedCheck_1110_;
goto v_resetjp_1099_;
}
v_resetjp_1099_:
{
lean_object* v___f_1102_; lean_object* v___x_1104_; 
v___f_1102_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_distribLattice___redArg___lam__6), 3, 1);
lean_closure_set(v___f_1102_, 0, v___f_1085_);
if (v_isShared_1101_ == 0)
{
v___x_1104_ = v___x_1100_;
goto v_reusejp_1103_;
}
else
{
lean_object* v_reuseFailAlloc_1109_; 
v_reuseFailAlloc_1109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1109_, 0, v_toLE_1097_);
lean_ctor_set(v_reuseFailAlloc_1109_, 1, v_toLT_1098_);
v___x_1104_ = v_reuseFailAlloc_1109_;
goto v_reusejp_1103_;
}
v_reusejp_1103_:
{
lean_object* v___x_1106_; 
if (v_isShared_1096_ == 0)
{
lean_ctor_set(v___x_1095_, 1, v___f_1102_);
lean_ctor_set(v___x_1095_, 0, v___x_1104_);
v___x_1106_ = v___x_1095_;
goto v_reusejp_1105_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v___x_1104_);
lean_ctor_set(v_reuseFailAlloc_1108_, 1, v___f_1102_);
v___x_1106_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1105_;
}
v_reusejp_1105_:
{
lean_object* v___x_1107_; 
v___x_1107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1107_, 0, v___x_1106_);
lean_ctor_set(v___x_1107_, 1, v___f_1084_);
return v___x_1107_;
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_distribLattice(lean_object* v_00_u03b1_1120_, lean_object* v_00_u03b2_1121_, lean_object* v_e_1122_, lean_object* v_inst_1123_){
_start:
{
lean_object* v_toSemilatticeSup_1124_; lean_object* v_inf_1125_; lean_object* v___x_1127_; uint8_t v_isShared_1128_; uint8_t v_isSharedCheck_1178_; 
v_toSemilatticeSup_1124_ = lean_ctor_get(v_inst_1123_, 0);
v_inf_1125_ = lean_ctor_get(v_inst_1123_, 1);
v_isSharedCheck_1178_ = !lean_is_exclusive(v_inst_1123_);
if (v_isSharedCheck_1178_ == 0)
{
v___x_1127_ = v_inst_1123_;
v_isShared_1128_ = v_isSharedCheck_1178_;
goto v_resetjp_1126_;
}
else
{
lean_inc(v_inf_1125_);
lean_inc(v_toSemilatticeSup_1124_);
lean_dec(v_inst_1123_);
v___x_1127_ = lean_box(0);
v_isShared_1128_ = v_isSharedCheck_1178_;
goto v_resetjp_1126_;
}
v_resetjp_1126_:
{
lean_object* v___f_1129_; lean_object* v_min_1130_; lean_object* v_le_1131_; lean_object* v_lt_1132_; lean_object* v_semilatticeInf_1133_; lean_object* v_toPartialOrder_1134_; lean_object* v___x_1136_; uint8_t v_isShared_1137_; uint8_t v_isSharedCheck_1176_; 
v___f_1129_ = ((lean_object*)(lp_mathlib_Equiv_linearOrder___redArg___closed__0));
lean_inc_ref(v_e_1122_);
v_min_1130_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__1), 5, 3);
lean_closure_set(v_min_1130_, 0, v_e_1122_);
lean_closure_set(v_min_1130_, 1, v___f_1129_);
lean_closure_set(v_min_1130_, 2, v_inf_1125_);
v_le_1131_ = lean_box(0);
v_lt_1132_ = lean_box(0);
lean_inc_ref(v_min_1130_);
v_semilatticeInf_1133_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_1130_, v_le_1131_, v_lt_1132_);
v_toPartialOrder_1134_ = lean_ctor_get(v_semilatticeInf_1133_, 0);
v_isSharedCheck_1176_ = !lean_is_exclusive(v_semilatticeInf_1133_);
if (v_isSharedCheck_1176_ == 0)
{
lean_object* v_unused_1177_; 
v_unused_1177_ = lean_ctor_get(v_semilatticeInf_1133_, 1);
lean_dec(v_unused_1177_);
v___x_1136_ = v_semilatticeInf_1133_;
v_isShared_1137_ = v_isSharedCheck_1176_;
goto v_resetjp_1135_;
}
else
{
lean_inc(v_toPartialOrder_1134_);
lean_dec(v_semilatticeInf_1133_);
v___x_1136_ = lean_box(0);
v_isShared_1137_ = v_isSharedCheck_1176_;
goto v_resetjp_1135_;
}
v_resetjp_1135_:
{
lean_object* v_toLE_1138_; lean_object* v_toLT_1139_; lean_object* v___x_1141_; uint8_t v_isShared_1142_; uint8_t v_isSharedCheck_1175_; 
v_toLE_1138_ = lean_ctor_get(v_toPartialOrder_1134_, 0);
v_toLT_1139_ = lean_ctor_get(v_toPartialOrder_1134_, 1);
v_isSharedCheck_1175_ = !lean_is_exclusive(v_toPartialOrder_1134_);
if (v_isSharedCheck_1175_ == 0)
{
v___x_1141_ = v_toPartialOrder_1134_;
v_isShared_1142_ = v_isSharedCheck_1175_;
goto v_resetjp_1140_;
}
else
{
lean_inc(v_toLT_1139_);
lean_inc(v_toLE_1138_);
lean_dec(v_toPartialOrder_1134_);
v___x_1141_ = lean_box(0);
v_isShared_1142_ = v_isSharedCheck_1175_;
goto v_resetjp_1140_;
}
v_resetjp_1140_:
{
lean_object* v___f_1143_; lean_object* v___f_1144_; lean_object* v___x_1146_; 
v___f_1143_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1143_, 0, v_min_1130_);
v___f_1144_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_lattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1144_, 0, v_toSemilatticeSup_1124_);
lean_closure_set(v___f_1144_, 1, v_e_1122_);
lean_closure_set(v___f_1144_, 2, v___f_1129_);
if (v_isShared_1142_ == 0)
{
v___x_1146_ = v___x_1141_;
goto v_reusejp_1145_;
}
else
{
lean_object* v_reuseFailAlloc_1174_; 
v_reuseFailAlloc_1174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1174_, 0, v_toLE_1138_);
lean_ctor_set(v_reuseFailAlloc_1174_, 1, v_toLT_1139_);
v___x_1146_ = v_reuseFailAlloc_1174_;
goto v_reusejp_1145_;
}
v_reusejp_1145_:
{
lean_object* v___x_1148_; 
lean_inc_ref(v___f_1144_);
if (v_isShared_1137_ == 0)
{
lean_ctor_set(v___x_1136_, 1, v___f_1144_);
lean_ctor_set(v___x_1136_, 0, v___x_1146_);
v___x_1148_ = v___x_1136_;
goto v_reusejp_1147_;
}
else
{
lean_object* v_reuseFailAlloc_1173_; 
v_reuseFailAlloc_1173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1173_, 0, v___x_1146_);
lean_ctor_set(v_reuseFailAlloc_1173_, 1, v___f_1144_);
v___x_1148_ = v_reuseFailAlloc_1173_;
goto v_reusejp_1147_;
}
v_reusejp_1147_:
{
lean_object* v_lattice_1150_; 
lean_inc_ref(v___f_1143_);
if (v_isShared_1128_ == 0)
{
lean_ctor_set(v___x_1127_, 1, v___f_1143_);
lean_ctor_set(v___x_1127_, 0, v___x_1148_);
v_lattice_1150_ = v___x_1127_;
goto v_reusejp_1149_;
}
else
{
lean_object* v_reuseFailAlloc_1172_; 
v_reuseFailAlloc_1172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1172_, 0, v___x_1148_);
lean_ctor_set(v_reuseFailAlloc_1172_, 1, v___f_1143_);
v_lattice_1150_ = v_reuseFailAlloc_1172_;
goto v_reusejp_1149_;
}
v_reusejp_1149_:
{
lean_object* v___x_1151_; lean_object* v_toPartialOrder_1152_; lean_object* v___x_1154_; uint8_t v_isShared_1155_; uint8_t v_isSharedCheck_1170_; 
v___x_1151_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_1150_);
v_toPartialOrder_1152_ = lean_ctor_get(v___x_1151_, 0);
v_isSharedCheck_1170_ = !lean_is_exclusive(v___x_1151_);
if (v_isSharedCheck_1170_ == 0)
{
lean_object* v_unused_1171_; 
v_unused_1171_ = lean_ctor_get(v___x_1151_, 1);
lean_dec(v_unused_1171_);
v___x_1154_ = v___x_1151_;
v_isShared_1155_ = v_isSharedCheck_1170_;
goto v_resetjp_1153_;
}
else
{
lean_inc(v_toPartialOrder_1152_);
lean_dec(v___x_1151_);
v___x_1154_ = lean_box(0);
v_isShared_1155_ = v_isSharedCheck_1170_;
goto v_resetjp_1153_;
}
v_resetjp_1153_:
{
lean_object* v_toLE_1156_; lean_object* v_toLT_1157_; lean_object* v___x_1159_; uint8_t v_isShared_1160_; uint8_t v_isSharedCheck_1169_; 
v_toLE_1156_ = lean_ctor_get(v_toPartialOrder_1152_, 0);
v_toLT_1157_ = lean_ctor_get(v_toPartialOrder_1152_, 1);
v_isSharedCheck_1169_ = !lean_is_exclusive(v_toPartialOrder_1152_);
if (v_isSharedCheck_1169_ == 0)
{
v___x_1159_ = v_toPartialOrder_1152_;
v_isShared_1160_ = v_isSharedCheck_1169_;
goto v_resetjp_1158_;
}
else
{
lean_inc(v_toLT_1157_);
lean_inc(v_toLE_1156_);
lean_dec(v_toPartialOrder_1152_);
v___x_1159_ = lean_box(0);
v_isShared_1160_ = v_isSharedCheck_1169_;
goto v_resetjp_1158_;
}
v_resetjp_1158_:
{
lean_object* v___f_1161_; lean_object* v___x_1163_; 
v___f_1161_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_distribLattice___redArg___lam__6), 3, 1);
lean_closure_set(v___f_1161_, 0, v___f_1144_);
if (v_isShared_1160_ == 0)
{
v___x_1163_ = v___x_1159_;
goto v_reusejp_1162_;
}
else
{
lean_object* v_reuseFailAlloc_1168_; 
v_reuseFailAlloc_1168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1168_, 0, v_toLE_1156_);
lean_ctor_set(v_reuseFailAlloc_1168_, 1, v_toLT_1157_);
v___x_1163_ = v_reuseFailAlloc_1168_;
goto v_reusejp_1162_;
}
v_reusejp_1162_:
{
lean_object* v___x_1165_; 
if (v_isShared_1155_ == 0)
{
lean_ctor_set(v___x_1154_, 1, v___f_1161_);
lean_ctor_set(v___x_1154_, 0, v___x_1163_);
v___x_1165_ = v___x_1154_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1167_; 
v_reuseFailAlloc_1167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1167_, 0, v___x_1163_);
lean_ctor_set(v_reuseFailAlloc_1167_, 1, v___f_1161_);
v___x_1165_ = v_reuseFailAlloc_1167_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
lean_object* v___x_1166_; 
v___x_1166_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1166_, 0, v___x_1165_);
lean_ctor_set(v___x_1166_, 1, v___f_1143_);
return v___x_1166_;
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSemilatticeSup___redArg(lean_object* v_inst_1179_){
_start:
{
lean_object* v___f_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; 
v___f_1180_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1180_, 0, v_inst_1179_);
v___x_1181_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
v___x_1182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1182_, 0, v___x_1181_);
lean_ctor_set(v___x_1182_, 1, v___f_1180_);
return v___x_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSemilatticeSup(lean_object* v_00_u03b1_1183_, lean_object* v_inst_1184_){
_start:
{
lean_object* v___x_1185_; 
v___x_1185_ = lp_mathlib_ULift_instSemilatticeSup___redArg(v_inst_1184_);
return v___x_1185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSemilatticeInf___redArg(lean_object* v_inst_1186_){
_start:
{
lean_object* v___f_1187_; lean_object* v___f_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; 
v___f_1187_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1187_, 0, v_inst_1186_);
v___f_1188_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1188_, 0, v___f_1187_);
v___x_1189_ = lean_box(0);
v___x_1190_ = lean_box(0);
v___x_1191_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v___f_1188_, v___x_1189_, v___x_1190_);
return v___x_1191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instSemilatticeInf(lean_object* v_00_u03b1_1192_, lean_object* v_inst_1193_){
_start:
{
lean_object* v___x_1194_; 
v___x_1194_ = lp_mathlib_ULift_instSemilatticeInf___redArg(v_inst_1193_);
return v___x_1194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLattice___redArg(lean_object* v_inst_1195_){
_start:
{
lean_object* v_toSemilatticeSup_1196_; lean_object* v_inf_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1206_; 
v_toSemilatticeSup_1196_ = lean_ctor_get(v_inst_1195_, 0);
v_inf_1197_ = lean_ctor_get(v_inst_1195_, 1);
v_isSharedCheck_1206_ = !lean_is_exclusive(v_inst_1195_);
if (v_isSharedCheck_1206_ == 0)
{
v___x_1199_ = v_inst_1195_;
v_isShared_1200_ = v_isSharedCheck_1206_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_inf_1197_);
lean_inc(v_toSemilatticeSup_1196_);
lean_dec(v_inst_1195_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1206_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
lean_object* v___f_1201_; lean_object* v___x_1202_; lean_object* v___x_1204_; 
v___f_1201_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instSemilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1201_, 0, v_inf_1197_);
v___x_1202_ = lp_mathlib_ULift_instSemilatticeSup___redArg(v_toSemilatticeSup_1196_);
if (v_isShared_1200_ == 0)
{
lean_ctor_set(v___x_1199_, 1, v___f_1201_);
lean_ctor_set(v___x_1199_, 0, v___x_1202_);
v___x_1204_ = v___x_1199_;
goto v_reusejp_1203_;
}
else
{
lean_object* v_reuseFailAlloc_1205_; 
v_reuseFailAlloc_1205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1205_, 0, v___x_1202_);
lean_ctor_set(v_reuseFailAlloc_1205_, 1, v___f_1201_);
v___x_1204_ = v_reuseFailAlloc_1205_;
goto v_reusejp_1203_;
}
v_reusejp_1203_:
{
return v___x_1204_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLattice(lean_object* v_00_u03b1_1207_, lean_object* v_inst_1208_){
_start:
{
lean_object* v___x_1209_; 
v___x_1209_ = lp_mathlib_ULift_instLattice___redArg(v_inst_1208_);
return v___x_1209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instDistribLattice___redArg(lean_object* v_inst_1210_){
_start:
{
lean_object* v_toSemilatticeSup_1211_; lean_object* v_inf_1212_; lean_object* v___x_1214_; uint8_t v_isShared_1215_; uint8_t v_isSharedCheck_1223_; 
v_toSemilatticeSup_1211_ = lean_ctor_get(v_inst_1210_, 0);
v_inf_1212_ = lean_ctor_get(v_inst_1210_, 1);
v_isSharedCheck_1223_ = !lean_is_exclusive(v_inst_1210_);
if (v_isSharedCheck_1223_ == 0)
{
v___x_1214_ = v_inst_1210_;
v_isShared_1215_ = v_isSharedCheck_1223_;
goto v_resetjp_1213_;
}
else
{
lean_inc(v_inf_1212_);
lean_inc(v_toSemilatticeSup_1211_);
lean_dec(v_inst_1210_);
v___x_1214_ = lean_box(0);
v_isShared_1215_ = v_isSharedCheck_1223_;
goto v_resetjp_1213_;
}
v_resetjp_1213_:
{
lean_object* v___f_1216_; lean_object* v___f_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1221_; 
v___f_1216_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instSemilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1216_, 0, v_inf_1212_);
v___f_1217_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1217_, 0, v_toSemilatticeSup_1211_);
v___x_1218_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
v___x_1219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1219_, 0, v___x_1218_);
lean_ctor_set(v___x_1219_, 1, v___f_1217_);
if (v_isShared_1215_ == 0)
{
lean_ctor_set(v___x_1214_, 1, v___f_1216_);
lean_ctor_set(v___x_1214_, 0, v___x_1219_);
v___x_1221_ = v___x_1214_;
goto v_reusejp_1220_;
}
else
{
lean_object* v_reuseFailAlloc_1222_; 
v_reuseFailAlloc_1222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1222_, 0, v___x_1219_);
lean_ctor_set(v_reuseFailAlloc_1222_, 1, v___f_1216_);
v___x_1221_ = v_reuseFailAlloc_1222_;
goto v_reusejp_1220_;
}
v_reusejp_1220_:
{
return v___x_1221_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instDistribLattice(lean_object* v_00_u03b1_1224_, lean_object* v_inst_1225_){
_start:
{
lean_object* v___x_1226_; 
v___x_1226_ = lp_mathlib_ULift_instDistribLattice___redArg(v_inst_1225_);
return v___x_1226_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ULift_instLinearOrder___redArg___lam__0(lean_object* v_toDecidableEq_1227_, lean_object* v_a_1228_, lean_object* v_b_1229_){
_start:
{
lean_object* v___x_1230_; uint8_t v___x_1231_; 
v___x_1230_ = lean_apply_2(v_toDecidableEq_1227_, v_a_1228_, v_b_1229_);
v___x_1231_ = lean_unbox(v___x_1230_);
return v___x_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder___redArg___lam__0___boxed(lean_object* v_toDecidableEq_1232_, lean_object* v_a_1233_, lean_object* v_b_1234_){
_start:
{
uint8_t v_res_1235_; lean_object* v_r_1236_; 
v_res_1235_ = lp_mathlib_ULift_instLinearOrder___redArg___lam__0(v_toDecidableEq_1232_, v_a_1233_, v_b_1234_);
v_r_1236_ = lean_box(v_res_1235_);
return v_r_1236_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ULift_instLinearOrder___redArg___lam__1(lean_object* v_toDecidableLE_1237_, lean_object* v_a_1238_, lean_object* v_b_1239_){
_start:
{
lean_object* v___x_1240_; uint8_t v___x_1241_; 
v___x_1240_ = lean_apply_2(v_toDecidableLE_1237_, v_a_1238_, v_b_1239_);
v___x_1241_ = lean_unbox(v___x_1240_);
return v___x_1241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder___redArg___lam__1___boxed(lean_object* v_toDecidableLE_1242_, lean_object* v_a_1243_, lean_object* v_b_1244_){
_start:
{
uint8_t v_res_1245_; lean_object* v_r_1246_; 
v_res_1245_ = lp_mathlib_ULift_instLinearOrder___redArg___lam__1(v_toDecidableLE_1242_, v_a_1243_, v_b_1244_);
v_r_1246_ = lean_box(v_res_1245_);
return v_r_1246_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ULift_instLinearOrder___redArg___lam__2(lean_object* v_toDecidableLT_1247_, lean_object* v_a_1248_, lean_object* v_b_1249_){
_start:
{
lean_object* v___x_1250_; uint8_t v___x_1251_; 
v___x_1250_ = lean_apply_2(v_toDecidableLT_1247_, v_a_1248_, v_b_1249_);
v___x_1251_ = lean_unbox(v___x_1250_);
return v___x_1251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder___redArg___lam__2___boxed(lean_object* v_toDecidableLT_1252_, lean_object* v_a_1253_, lean_object* v_b_1254_){
_start:
{
uint8_t v_res_1255_; lean_object* v_r_1256_; 
v_res_1255_ = lp_mathlib_ULift_instLinearOrder___redArg___lam__2(v_toDecidableLT_1252_, v_a_1253_, v_b_1254_);
v_r_1256_ = lean_box(v_res_1255_);
return v_r_1256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder___redArg(lean_object* v_inst_1257_){
_start:
{
lean_object* v_toMin_1258_; lean_object* v_toMax_1259_; lean_object* v_toOrd_1260_; lean_object* v_toDecidableLE_1261_; lean_object* v_toDecidableEq_1262_; lean_object* v_toDecidableLT_1263_; lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1277_; 
v_toMin_1258_ = lean_ctor_get(v_inst_1257_, 1);
v_toMax_1259_ = lean_ctor_get(v_inst_1257_, 2);
v_toOrd_1260_ = lean_ctor_get(v_inst_1257_, 3);
v_toDecidableLE_1261_ = lean_ctor_get(v_inst_1257_, 4);
v_toDecidableEq_1262_ = lean_ctor_get(v_inst_1257_, 5);
v_toDecidableLT_1263_ = lean_ctor_get(v_inst_1257_, 6);
v_isSharedCheck_1277_ = !lean_is_exclusive(v_inst_1257_);
if (v_isSharedCheck_1277_ == 0)
{
lean_object* v_unused_1278_; 
v_unused_1278_ = lean_ctor_get(v_inst_1257_, 0);
lean_dec(v_unused_1278_);
v___x_1265_ = v_inst_1257_;
v_isShared_1266_ = v_isSharedCheck_1277_;
goto v_resetjp_1264_;
}
else
{
lean_inc(v_toDecidableLT_1263_);
lean_inc(v_toDecidableEq_1262_);
lean_inc(v_toDecidableLE_1261_);
lean_inc(v_toOrd_1260_);
lean_inc(v_toMax_1259_);
lean_inc(v_toMin_1258_);
lean_dec(v_inst_1257_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1277_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
lean_object* v___f_1267_; lean_object* v___f_1268_; lean_object* v___f_1269_; lean_object* v___f_1270_; lean_object* v___f_1271_; lean_object* v___f_1272_; lean_object* v___x_1273_; lean_object* v___x_1275_; 
v___f_1267_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1267_, 0, v_toDecidableEq_1262_);
v___f_1268_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instLinearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_1268_, 0, v_toDecidableLE_1261_);
v___f_1269_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instLinearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_1269_, 0, v_toDecidableLT_1263_);
v___f_1270_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1270_, 0, v_toMax_1259_);
v___f_1271_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instMax__mathlib___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1271_, 0, v_toMin_1258_);
v___f_1272_ = lean_alloc_closure((void*)(lp_mathlib_ULift_instOrd__mathlib___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1272_, 0, v_toOrd_1260_);
v___x_1273_ = ((lean_object*)(lp_mathlib_SemilatticeSup_mk_x27___redArg___closed__0));
if (v_isShared_1266_ == 0)
{
lean_ctor_set(v___x_1265_, 6, v___f_1269_);
lean_ctor_set(v___x_1265_, 5, v___f_1267_);
lean_ctor_set(v___x_1265_, 4, v___f_1268_);
lean_ctor_set(v___x_1265_, 3, v___f_1272_);
lean_ctor_set(v___x_1265_, 2, v___f_1270_);
lean_ctor_set(v___x_1265_, 1, v___f_1271_);
lean_ctor_set(v___x_1265_, 0, v___x_1273_);
v___x_1275_ = v___x_1265_;
goto v_reusejp_1274_;
}
else
{
lean_object* v_reuseFailAlloc_1276_; 
v_reuseFailAlloc_1276_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_1276_, 0, v___x_1273_);
lean_ctor_set(v_reuseFailAlloc_1276_, 1, v___f_1271_);
lean_ctor_set(v_reuseFailAlloc_1276_, 2, v___f_1270_);
lean_ctor_set(v_reuseFailAlloc_1276_, 3, v___f_1272_);
lean_ctor_set(v_reuseFailAlloc_1276_, 4, v___f_1268_);
lean_ctor_set(v_reuseFailAlloc_1276_, 5, v___f_1267_);
lean_ctor_set(v_reuseFailAlloc_1276_, 6, v___f_1269_);
v___x_1275_ = v_reuseFailAlloc_1276_;
goto v_reusejp_1274_;
}
v_reusejp_1274_:
{
return v___x_1275_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instLinearOrder(lean_object* v_00_u03b1_1279_, lean_object* v_inst_1280_){
_start:
{
lean_object* v___x_1281_; 
v___x_1281_ = lp_mathlib_ULift_instLinearOrder___redArg(v_inst_1280_);
return v___x_1281_;
}
}
static lean_object* _init_lp_mathlib_Bool_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_1282_; lean_object* v___x_1283_; 
v___x_1282_ = lp_mathlib_Bool_linearOrder;
v___x_1283_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_1282_);
return v___x_1283_;
}
}
static lean_object* _init_lp_mathlib_Bool_instPartialOrder___closed__1(void){
_start:
{
lean_object* v___x_1284_; lean_object* v___x_1285_; 
v___x_1284_ = lean_obj_once(&lp_mathlib_Bool_instPartialOrder___closed__0, &lp_mathlib_Bool_instPartialOrder___closed__0_once, _init_lp_mathlib_Bool_instPartialOrder___closed__0);
v___x_1285_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_1284_);
return v___x_1285_;
}
}
static lean_object* _init_lp_mathlib_Bool_instPartialOrder(void){
_start:
{
lean_object* v___x_1286_; lean_object* v_toPartialOrder_1287_; 
v___x_1286_ = lean_obj_once(&lp_mathlib_Bool_instPartialOrder___closed__1, &lp_mathlib_Bool_instPartialOrder___closed__1_once, _init_lp_mathlib_Bool_instPartialOrder___closed__1);
v_toPartialOrder_1287_ = lean_ctor_get(v___x_1286_, 0);
lean_inc_ref(v_toPartialOrder_1287_);
return v_toPartialOrder_1287_;
}
}
static lean_object* _init_lp_mathlib_Bool_instDistribLattice(void){
_start:
{
lean_object* v___x_1288_; 
v___x_1288_ = lean_obj_once(&lp_mathlib_Bool_instPartialOrder___closed__0, &lp_mathlib_Bool_instPartialOrder___closed__0_once, _init_lp_mathlib_Bool_instPartialOrder___closed__0);
return v___x_1288_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Bool_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Pairwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Monotone_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GRewrite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Bool_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Monotone_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instDistribLatticeNat = _init_lp_mathlib_instDistribLatticeNat();
lean_mark_persistent(lp_mathlib_instDistribLatticeNat);
lp_mathlib_instLatticeInt = _init_lp_mathlib_instLatticeInt();
lean_mark_persistent(lp_mathlib_instLatticeInt);
lp_mathlib_Bool_instPartialOrder = _init_lp_mathlib_Bool_instPartialOrder();
lean_mark_persistent(lp_mathlib_Bool_instPartialOrder);
lp_mathlib_Bool_instDistribLattice = _init_lp_mathlib_Bool_instDistribLattice();
lean_mark_persistent(lp_mathlib_Bool_instDistribLattice);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Bool_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Pairwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Monotone_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GRewrite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Bool_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Monotone_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Lattice(builtin);
}
#ifdef __cplusplus
}
#endif
