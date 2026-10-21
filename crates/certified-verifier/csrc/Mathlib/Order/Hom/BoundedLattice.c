// Lean compiler output
// Module: Mathlib.Order.Hom.BoundedLattice
// Imports: public import Init public meta import Init public import Mathlib.Order.Hom.Bounded public import Mathlib.Order.Hom.Lattice public import Mathlib.Order.SymmDiff
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
lean_object* lp_mathlib_InfHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_SupHom_const___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_SupHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SupHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SupHom_subtypeVal___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_InfHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LatticeHom_dual___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_toBotHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_toBotHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_toBotHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_toBotHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_toTopHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_toTopHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_toTopHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_toTopHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toInfTopHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toInfTopHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toInfTopHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toInfTopHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toSupBotHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toSupBotHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toSupBotHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toSupBotHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfTopHomOfInfTopHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfTopHomOfInfTopHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfTopHomOfInfTopHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedLatticeHomOfBoundedLatticeHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedLatticeHomOfBoundedLatticeHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedLatticeHomOfBoundedLatticeHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instFunLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SupBotHom_instFunLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SupBotHom_instFunLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SupBotHom_instFunLike___closed__0 = (const lean_object*)&lp_mathlib_SupBotHom_instFunLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SupBotHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_SupBotHom_id___closed__0 = (const lean_object*)&lp_mathlib_SupBotHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instMax___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instMax___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instMax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instMax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instMin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_SupBotHom_instPartialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SupBotHom_instPartialOrder___closed__0 = (const lean_object*)&lp_mathlib_SupBotHom_instPartialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instSemilatticeInf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instSemilatticeInf___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instSemilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SupBotHom_subtypeVal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SupHom_subtypeVal___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SupBotHom_subtypeVal___closed__0 = (const lean_object*)&lp_mathlib_SupBotHom_subtypeVal___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_subtypeVal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_subtypeVal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_subtypeVal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_subtypeVal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toBoundedOrderHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toBoundedOrderHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toBoundedOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toBoundedOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_subtypeVal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_subtypeVal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_toBotHom___redArg(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_toBotHom___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_SupBotHom_toBotHom___redArg(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_toBotHom(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_self_10_){
_start:
{
lean_inc(v_self_10_);
return v_self_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_toBotHom___boxed(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_self_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_SupBotHom_toBotHom(v_00_u03b1_11_, v_00_u03b2_12_, v_inst_13_, v_inst_14_, v_inst_15_, v_inst_16_, v_self_17_);
lean_dec(v_self_17_);
lean_dec(v_inst_16_);
lean_dec(v_inst_15_);
lean_dec(v_inst_14_);
lean_dec(v_inst_13_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_toTopHom___redArg(lean_object* v_self_19_){
_start:
{
lean_inc(v_self_19_);
return v_self_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_toTopHom___redArg___boxed(lean_object* v_self_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_InfTopHom_toTopHom___redArg(v_self_20_);
lean_dec(v_self_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_toTopHom(lean_object* v_00_u03b1_22_, lean_object* v_00_u03b2_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_self_28_){
_start:
{
lean_inc(v_self_28_);
return v_self_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_toTopHom___boxed(lean_object* v_00_u03b1_29_, lean_object* v_00_u03b2_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_self_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_InfTopHom_toTopHom(v_00_u03b1_29_, v_00_u03b2_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_self_35_);
lean_dec(v_self_35_);
lean_dec(v_inst_34_);
lean_dec(v_inst_33_);
lean_dec(v_inst_32_);
lean_dec(v_inst_31_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toInfTopHom___redArg(lean_object* v_self_37_){
_start:
{
lean_inc(v_self_37_);
return v_self_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toInfTopHom___redArg___boxed(lean_object* v_self_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_BoundedLatticeHom_toInfTopHom___redArg(v_self_38_);
lean_dec(v_self_38_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toInfTopHom(lean_object* v_00_u03b1_40_, lean_object* v_00_u03b2_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_self_46_){
_start:
{
lean_inc(v_self_46_);
return v_self_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toInfTopHom___boxed(lean_object* v_00_u03b1_47_, lean_object* v_00_u03b2_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_self_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_BoundedLatticeHom_toInfTopHom(v_00_u03b1_47_, v_00_u03b2_48_, v_inst_49_, v_inst_50_, v_inst_51_, v_inst_52_, v_self_53_);
lean_dec(v_self_53_);
lean_dec_ref(v_inst_52_);
lean_dec_ref(v_inst_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toSupBotHom___redArg(lean_object* v_self_55_){
_start:
{
lean_inc(v_self_55_);
return v_self_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toSupBotHom___redArg___boxed(lean_object* v_self_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_BoundedLatticeHom_toSupBotHom___redArg(v_self_56_);
lean_dec(v_self_56_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toSupBotHom(lean_object* v_00_u03b1_58_, lean_object* v_00_u03b2_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_self_64_){
_start:
{
lean_inc(v_self_64_);
return v_self_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toSupBotHom___boxed(lean_object* v_00_u03b1_65_, lean_object* v_00_u03b2_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_self_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_BoundedLatticeHom_toSupBotHom(v_00_u03b1_65_, v_00_u03b2_66_, v_inst_67_, v_inst_68_, v_inst_69_, v_inst_70_, v_self_71_);
lean_dec(v_self_71_);
lean_dec_ref(v_inst_70_);
lean_dec_ref(v_inst_69_);
lean_dec_ref(v_inst_68_);
lean_dec_ref(v_inst_67_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg___lam__0(lean_object* v_inst_73_, lean_object* v_f_74_, lean_object* v___y_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lean_apply_2(v_inst_73_, v_f_74_, v___y_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg(lean_object* v_inst_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_78_, 0, v_inst_77_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass(lean_object* v_F_79_, lean_object* v_00_u03b1_80_, lean_object* v_00_u03b2_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_){
_start:
{
lean_object* v___f_88_; 
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_88_, 0, v_inst_82_);
return v___f_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___boxed(lean_object* v_F_89_, lean_object* v_00_u03b1_90_, lean_object* v_00_u03b2_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass(v_F_89_, v_00_u03b1_90_, v_00_u03b2_91_, v_inst_92_, v_inst_93_, v_inst_94_, v_inst_95_, v_inst_96_, v_inst_97_);
lean_dec(v_inst_96_);
lean_dec(v_inst_95_);
lean_dec(v_inst_94_);
lean_dec(v_inst_93_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfTopHomOfInfTopHomClass___redArg(lean_object* v_inst_99_){
_start:
{
lean_object* v___f_100_; 
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_100_, 0, v_inst_99_);
return v___f_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfTopHomOfInfTopHomClass(lean_object* v_F_101_, lean_object* v_00_u03b1_102_, lean_object* v_00_u03b2_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___f_110_; 
v___f_110_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_110_, 0, v_inst_104_);
return v___f_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfTopHomOfInfTopHomClass___boxed(lean_object* v_F_111_, lean_object* v_00_u03b1_112_, lean_object* v_00_u03b2_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_instCoeTCInfTopHomOfInfTopHomClass(v_F_111_, v_00_u03b1_112_, v_00_u03b2_113_, v_inst_114_, v_inst_115_, v_inst_116_, v_inst_117_, v_inst_118_, v_inst_119_);
lean_dec(v_inst_118_);
lean_dec(v_inst_117_);
lean_dec(v_inst_116_);
lean_dec(v_inst_115_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedLatticeHomOfBoundedLatticeHomClass___redArg(lean_object* v_inst_121_){
_start:
{
lean_object* v___f_122_; 
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_122_, 0, v_inst_121_);
return v___f_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedLatticeHomOfBoundedLatticeHomClass(lean_object* v_F_123_, lean_object* v_00_u03b1_124_, lean_object* v_00_u03b2_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___f_132_; 
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupBotHomOfSupBotHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_132_, 0, v_inst_126_);
return v___f_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedLatticeHomOfBoundedLatticeHomClass___boxed(lean_object* v_F_133_, lean_object* v_00_u03b1_134_, lean_object* v_00_u03b2_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_instCoeTCBoundedLatticeHomOfBoundedLatticeHomClass(v_F_133_, v_00_u03b1_134_, v_00_u03b2_135_, v_inst_136_, v_inst_137_, v_inst_138_, v_inst_139_, v_inst_140_, v_inst_141_);
lean_dec_ref(v_inst_140_);
lean_dec_ref(v_inst_139_);
lean_dec_ref(v_inst_138_);
lean_dec_ref(v_inst_137_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instFunLike___lam__0(lean_object* v_f_143_, lean_object* v___y_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_apply_1(v_f_143_, v___y_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instFunLike(lean_object* v_00_u03b1_147_, lean_object* v_00_u03b2_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_){
_start:
{
lean_object* v___f_153_; 
v___f_153_ = ((lean_object*)(lp_mathlib_SupBotHom_instFunLike___closed__0));
return v___f_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instFunLike___boxed(lean_object* v_00_u03b1_154_, lean_object* v_00_u03b2_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_SupBotHom_instFunLike(v_00_u03b1_154_, v_00_u03b2_155_, v_inst_156_, v_inst_157_, v_inst_158_, v_inst_159_);
lean_dec(v_inst_159_);
lean_dec(v_inst_158_);
lean_dec(v_inst_157_);
lean_dec(v_inst_156_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instFunLike(lean_object* v_00_u03b1_161_, lean_object* v_00_u03b2_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v___f_167_; 
v___f_167_ = ((lean_object*)(lp_mathlib_SupBotHom_instFunLike___closed__0));
return v___f_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instFunLike___boxed(lean_object* v_00_u03b1_168_, lean_object* v_00_u03b2_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_InfTopHom_instFunLike(v_00_u03b1_168_, v_00_u03b2_169_, v_inst_170_, v_inst_171_, v_inst_172_, v_inst_173_);
lean_dec(v_inst_173_);
lean_dec(v_inst_172_);
lean_dec(v_inst_171_);
lean_dec(v_inst_170_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_copy___redArg(lean_object* v_f_x27_175_){
_start:
{
lean_inc(v_f_x27_175_);
return v_f_x27_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_copy___redArg___boxed(lean_object* v_f_x27_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib_SupBotHom_copy___redArg(v_f_x27_176_);
lean_dec(v_f_x27_176_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_copy(lean_object* v_00_u03b1_178_, lean_object* v_00_u03b2_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_f_184_, lean_object* v_f_x27_185_, lean_object* v_h_186_){
_start:
{
lean_inc(v_f_x27_185_);
return v_f_x27_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_copy___boxed(lean_object* v_00_u03b1_187_, lean_object* v_00_u03b2_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_f_193_, lean_object* v_f_x27_194_, lean_object* v_h_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_SupBotHom_copy(v_00_u03b1_187_, v_00_u03b2_188_, v_inst_189_, v_inst_190_, v_inst_191_, v_inst_192_, v_f_193_, v_f_x27_194_, v_h_195_);
lean_dec(v_f_x27_194_);
lean_dec(v_f_193_);
lean_dec(v_inst_192_);
lean_dec(v_inst_191_);
lean_dec(v_inst_190_);
lean_dec(v_inst_189_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_copy___redArg(lean_object* v_f_x27_197_){
_start:
{
lean_inc(v_f_x27_197_);
return v_f_x27_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_copy___redArg___boxed(lean_object* v_f_x27_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_InfTopHom_copy___redArg(v_f_x27_198_);
lean_dec(v_f_x27_198_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_copy(lean_object* v_00_u03b1_200_, lean_object* v_00_u03b2_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_f_206_, lean_object* v_f_x27_207_, lean_object* v_h_208_){
_start:
{
lean_inc(v_f_x27_207_);
return v_f_x27_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_copy___boxed(lean_object* v_00_u03b1_209_, lean_object* v_00_u03b2_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_f_215_, lean_object* v_f_x27_216_, lean_object* v_h_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_InfTopHom_copy(v_00_u03b1_209_, v_00_u03b2_210_, v_inst_211_, v_inst_212_, v_inst_213_, v_inst_214_, v_f_215_, v_f_x27_216_, v_h_217_);
lean_dec(v_f_x27_216_);
lean_dec(v_f_215_);
lean_dec(v_inst_214_);
lean_dec(v_inst_213_);
lean_dec(v_inst_212_);
lean_dec(v_inst_211_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_id(lean_object* v_00_u03b1_220_, lean_object* v_inst_221_, lean_object* v_inst_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = ((lean_object*)(lp_mathlib_SupBotHom_id___closed__0));
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_id___boxed(lean_object* v_00_u03b1_224_, lean_object* v_inst_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_mathlib_SupBotHom_id(v_00_u03b1_224_, v_inst_225_, v_inst_226_);
lean_dec(v_inst_226_);
lean_dec(v_inst_225_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_id(lean_object* v_00_u03b1_228_, lean_object* v_inst_229_, lean_object* v_inst_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = ((lean_object*)(lp_mathlib_SupBotHom_id___closed__0));
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_id___boxed(lean_object* v_00_u03b1_232_, lean_object* v_inst_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_InfTopHom_id(v_00_u03b1_232_, v_inst_233_, v_inst_234_);
lean_dec(v_inst_234_);
lean_dec(v_inst_233_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instInhabited(lean_object* v_00_u03b1_236_, lean_object* v_inst_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = ((lean_object*)(lp_mathlib_SupBotHom_id___closed__0));
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instInhabited___boxed(lean_object* v_00_u03b1_240_, lean_object* v_inst_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_SupBotHom_instInhabited(v_00_u03b1_240_, v_inst_241_, v_inst_242_);
lean_dec(v_inst_242_);
lean_dec(v_inst_241_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instInhabited(lean_object* v_00_u03b1_244_, lean_object* v_inst_245_, lean_object* v_inst_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = ((lean_object*)(lp_mathlib_SupBotHom_id___closed__0));
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instInhabited___boxed(lean_object* v_00_u03b1_248_, lean_object* v_inst_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_InfTopHom_instInhabited(v_00_u03b1_248_, v_inst_249_, v_inst_250_);
lean_dec(v_inst_250_);
lean_dec(v_inst_249_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_comp___redArg(lean_object* v_f_252_, lean_object* v_g_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lp_mathlib_SupHom_comp___redArg(v_f_252_, v_g_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_comp(lean_object* v_00_u03b1_255_, lean_object* v_00_u03b2_256_, lean_object* v_00_u03b3_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_f_264_, lean_object* v_g_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lp_mathlib_SupHom_comp___redArg(v_f_264_, v_g_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_comp___boxed(lean_object* v_00_u03b1_267_, lean_object* v_00_u03b2_268_, lean_object* v_00_u03b3_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_f_276_, lean_object* v_g_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_SupBotHom_comp(v_00_u03b1_267_, v_00_u03b2_268_, v_00_u03b3_269_, v_inst_270_, v_inst_271_, v_inst_272_, v_inst_273_, v_inst_274_, v_inst_275_, v_f_276_, v_g_277_);
lean_dec(v_inst_275_);
lean_dec(v_inst_274_);
lean_dec(v_inst_273_);
lean_dec(v_inst_272_);
lean_dec(v_inst_271_);
lean_dec(v_inst_270_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_comp___redArg(lean_object* v_f_279_, lean_object* v_g_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_mathlib_InfHom_comp___redArg(v_f_279_, v_g_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_comp(lean_object* v_00_u03b1_282_, lean_object* v_00_u03b2_283_, lean_object* v_00_u03b3_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_f_291_, lean_object* v_g_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lp_mathlib_InfHom_comp___redArg(v_f_291_, v_g_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_comp___boxed(lean_object* v_00_u03b1_294_, lean_object* v_00_u03b2_295_, lean_object* v_00_u03b3_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_f_303_, lean_object* v_g_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_InfTopHom_comp(v_00_u03b1_294_, v_00_u03b2_295_, v_00_u03b3_296_, v_inst_297_, v_inst_298_, v_inst_299_, v_inst_300_, v_inst_301_, v_inst_302_, v_f_303_, v_g_304_);
lean_dec(v_inst_302_);
lean_dec(v_inst_301_);
lean_dec(v_inst_300_);
lean_dec(v_inst_299_);
lean_dec(v_inst_298_);
lean_dec(v_inst_297_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instMax___redArg___lam__0(lean_object* v_inst_306_, lean_object* v_f_307_, lean_object* v_g_308_, lean_object* v___y_309_){
_start:
{
lean_object* v_sup_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v_sup_310_ = lean_ctor_get(v_inst_306_, 1);
lean_inc(v_sup_310_);
lean_dec_ref(v_inst_306_);
lean_inc(v___y_309_);
v___x_311_ = lean_apply_1(v_f_307_, v___y_309_);
v___x_312_ = lean_apply_1(v_g_308_, v___y_309_);
v___x_313_ = lean_apply_2(v_sup_310_, v___x_311_, v___x_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instMax___redArg(lean_object* v_inst_314_){
_start:
{
lean_object* v___f_315_; 
v___f_315_ = lean_alloc_closure((void*)(lp_mathlib_SupBotHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_315_, 0, v_inst_314_);
return v___f_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instMax(lean_object* v_00_u03b1_316_, lean_object* v_00_u03b2_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_){
_start:
{
lean_object* v___f_322_; 
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib_SupBotHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_322_, 0, v_inst_320_);
return v___f_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instMax___boxed(lean_object* v_00_u03b1_323_, lean_object* v_00_u03b2_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_inst_327_, lean_object* v_inst_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_SupBotHom_instMax(v_00_u03b1_323_, v_00_u03b2_324_, v_inst_325_, v_inst_326_, v_inst_327_, v_inst_328_);
lean_dec(v_inst_328_);
lean_dec(v_inst_326_);
lean_dec(v_inst_325_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instMin___redArg___lam__0(lean_object* v_inst_330_, lean_object* v_f_331_, lean_object* v_g_332_, lean_object* v___y_333_){
_start:
{
lean_object* v_inf_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v_inf_334_ = lean_ctor_get(v_inst_330_, 1);
lean_inc(v_inf_334_);
lean_dec_ref(v_inst_330_);
lean_inc(v___y_333_);
v___x_335_ = lean_apply_1(v_f_331_, v___y_333_);
v___x_336_ = lean_apply_1(v_g_332_, v___y_333_);
v___x_337_ = lean_apply_2(v_inf_334_, v___x_335_, v___x_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instMin___redArg(lean_object* v_inst_338_){
_start:
{
lean_object* v___f_339_; 
v___f_339_ = lean_alloc_closure((void*)(lp_mathlib_InfTopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_339_, 0, v_inst_338_);
return v___f_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instMin(lean_object* v_00_u03b1_340_, lean_object* v_00_u03b2_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_){
_start:
{
lean_object* v___f_346_; 
v___f_346_ = lean_alloc_closure((void*)(lp_mathlib_InfTopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_346_, 0, v_inst_344_);
return v___f_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instMin___boxed(lean_object* v_00_u03b1_347_, lean_object* v_00_u03b2_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib_InfTopHom_instMin(v_00_u03b1_347_, v_00_u03b2_348_, v_inst_349_, v_inst_350_, v_inst_351_, v_inst_352_);
lean_dec(v_inst_352_);
lean_dec(v_inst_350_);
lean_dec(v_inst_349_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instPartialOrder(lean_object* v_00_u03b1_357_, lean_object* v_00_u03b2_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_){
_start:
{
lean_object* v___x_363_; 
v___x_363_ = ((lean_object*)(lp_mathlib_SupBotHom_instPartialOrder___closed__0));
return v___x_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instPartialOrder___boxed(lean_object* v_00_u03b1_364_, lean_object* v_00_u03b2_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_mathlib_SupBotHom_instPartialOrder(v_00_u03b1_364_, v_00_u03b2_365_, v_inst_366_, v_inst_367_, v_inst_368_, v_inst_369_);
lean_dec(v_inst_369_);
lean_dec_ref(v_inst_368_);
lean_dec(v_inst_367_);
lean_dec(v_inst_366_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instPartialOrder(lean_object* v_00_u03b1_371_, lean_object* v_00_u03b2_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = ((lean_object*)(lp_mathlib_SupBotHom_instPartialOrder___closed__0));
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instPartialOrder___boxed(lean_object* v_00_u03b1_378_, lean_object* v_00_u03b2_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_inst_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_mathlib_InfTopHom_instPartialOrder(v_00_u03b1_378_, v_00_u03b2_379_, v_inst_380_, v_inst_381_, v_inst_382_, v_inst_383_);
lean_dec(v_inst_383_);
lean_dec_ref(v_inst_382_);
lean_dec(v_inst_381_);
lean_dec(v_inst_380_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup___redArg___lam__0(lean_object* v_inst_385_, lean_object* v_a_386_, lean_object* v_b_387_, lean_object* v___y_388_){
_start:
{
lean_object* v_sup_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
v_sup_389_ = lean_ctor_get(v_inst_385_, 1);
lean_inc(v_sup_389_);
lean_dec_ref(v_inst_385_);
lean_inc(v___y_388_);
v___x_390_ = lean_apply_1(v_a_386_, v___y_388_);
v___x_391_ = lean_apply_1(v_b_387_, v___y_388_);
v___x_392_ = lean_apply_2(v_sup_389_, v___x_390_, v___x_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup___redArg(lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_){
_start:
{
lean_object* v___x_397_; lean_object* v_toLE_398_; lean_object* v_toLT_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_408_; 
v___x_397_ = lp_mathlib_SupBotHom_instPartialOrder(lean_box(0), lean_box(0), v_inst_393_, v_inst_394_, v_inst_395_, v_inst_396_);
v_toLE_398_ = lean_ctor_get(v___x_397_, 0);
v_toLT_399_ = lean_ctor_get(v___x_397_, 1);
v_isSharedCheck_408_ = !lean_is_exclusive(v___x_397_);
if (v_isSharedCheck_408_ == 0)
{
v___x_401_ = v___x_397_;
v_isShared_402_ = v_isSharedCheck_408_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_toLT_399_);
lean_inc(v_toLE_398_);
lean_dec(v___x_397_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_408_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___f_403_; lean_object* v___x_405_; 
v___f_403_ = lean_alloc_closure((void*)(lp_mathlib_SupBotHom_instSemilatticeSup___redArg___lam__0), 4, 1);
lean_closure_set(v___f_403_, 0, v_inst_395_);
if (v_isShared_402_ == 0)
{
v___x_405_ = v___x_401_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_407_; 
v_reuseFailAlloc_407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_407_, 0, v_toLE_398_);
lean_ctor_set(v_reuseFailAlloc_407_, 1, v_toLT_399_);
v___x_405_ = v_reuseFailAlloc_407_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
lean_object* v___x_406_; 
v___x_406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_406_, 0, v___x_405_);
lean_ctor_set(v___x_406_, 1, v___f_403_);
return v___x_406_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup___redArg___boxed(lean_object* v_inst_409_, lean_object* v_inst_410_, lean_object* v_inst_411_, lean_object* v_inst_412_){
_start:
{
lean_object* v_res_413_; 
v_res_413_ = lp_mathlib_SupBotHom_instSemilatticeSup___redArg(v_inst_409_, v_inst_410_, v_inst_411_, v_inst_412_);
lean_dec(v_inst_412_);
lean_dec(v_inst_410_);
lean_dec(v_inst_409_);
return v_res_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup(lean_object* v_00_u03b1_414_, lean_object* v_00_u03b2_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_inst_419_){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = lp_mathlib_SupBotHom_instSemilatticeSup___redArg(v_inst_416_, v_inst_417_, v_inst_418_, v_inst_419_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instSemilatticeSup___boxed(lean_object* v_00_u03b1_421_, lean_object* v_00_u03b2_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_SupBotHom_instSemilatticeSup(v_00_u03b1_421_, v_00_u03b2_422_, v_inst_423_, v_inst_424_, v_inst_425_, v_inst_426_);
lean_dec(v_inst_426_);
lean_dec(v_inst_424_);
lean_dec(v_inst_423_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instSemilatticeInf___redArg(lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_){
_start:
{
lean_object* v___x_432_; lean_object* v_toLE_433_; lean_object* v_toLT_434_; lean_object* v___f_435_; lean_object* v___x_436_; 
v___x_432_ = lp_mathlib_InfTopHom_instPartialOrder(lean_box(0), lean_box(0), v_inst_428_, v_inst_429_, v_inst_430_, v_inst_431_);
v_toLE_433_ = lean_ctor_get(v___x_432_, 0);
lean_inc(v_toLE_433_);
v_toLT_434_ = lean_ctor_get(v___x_432_, 1);
lean_inc(v_toLT_434_);
lean_dec_ref(v___x_432_);
v___f_435_ = lean_alloc_closure((void*)(lp_mathlib_InfTopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_435_, 0, v_inst_430_);
v___x_436_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v___f_435_, v_toLE_433_, v_toLT_434_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instSemilatticeInf___redArg___boxed(lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_mathlib_InfTopHom_instSemilatticeInf___redArg(v_inst_437_, v_inst_438_, v_inst_439_, v_inst_440_);
lean_dec(v_inst_440_);
lean_dec(v_inst_438_);
lean_dec(v_inst_437_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instSemilatticeInf(lean_object* v_00_u03b1_442_, lean_object* v_00_u03b2_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_inst_447_){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lp_mathlib_InfTopHom_instSemilatticeInf___redArg(v_inst_444_, v_inst_445_, v_inst_446_, v_inst_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instSemilatticeInf___boxed(lean_object* v_00_u03b1_449_, lean_object* v_00_u03b2_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_inst_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_InfTopHom_instSemilatticeInf(v_00_u03b1_449_, v_00_u03b2_450_, v_inst_451_, v_inst_452_, v_inst_453_, v_inst_454_);
lean_dec(v_inst_454_);
lean_dec(v_inst_452_);
lean_dec(v_inst_451_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instOrderBot___redArg(lean_object* v_inst_456_){
_start:
{
lean_object* v___f_457_; 
v___f_457_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_457_, 0, v_inst_456_);
return v___f_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instOrderBot(lean_object* v_00_u03b1_458_, lean_object* v_00_u03b2_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_inst_463_){
_start:
{
lean_object* v___f_464_; 
v___f_464_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_464_, 0, v_inst_463_);
return v___f_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_instOrderBot___boxed(lean_object* v_00_u03b1_465_, lean_object* v_00_u03b2_466_, lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_inst_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_SupBotHom_instOrderBot(v_00_u03b1_465_, v_00_u03b2_466_, v_inst_467_, v_inst_468_, v_inst_469_, v_inst_470_);
lean_dec_ref(v_inst_469_);
lean_dec(v_inst_468_);
lean_dec(v_inst_467_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instOrderTop___redArg(lean_object* v_inst_472_){
_start:
{
lean_object* v___f_473_; 
v___f_473_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_473_, 0, v_inst_472_);
return v___f_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instOrderTop(lean_object* v_00_u03b1_474_, lean_object* v_00_u03b2_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_){
_start:
{
lean_object* v___f_480_; 
v___f_480_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_480_, 0, v_inst_479_);
return v___f_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_instOrderTop___boxed(lean_object* v_00_u03b1_481_, lean_object* v_00_u03b2_482_, lean_object* v_inst_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib_InfTopHom_instOrderTop(v_00_u03b1_481_, v_00_u03b2_482_, v_inst_483_, v_inst_484_, v_inst_485_, v_inst_486_);
lean_dec_ref(v_inst_485_);
lean_dec(v_inst_484_);
lean_dec(v_inst_483_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_subtypeVal(lean_object* v_00_u03b2_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_P_492_, lean_object* v_Pbot_493_, lean_object* v_Psup_494_){
_start:
{
lean_object* v___f_495_; 
v___f_495_ = ((lean_object*)(lp_mathlib_SupBotHom_subtypeVal___closed__0));
return v___f_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_subtypeVal___boxed(lean_object* v_00_u03b2_496_, lean_object* v_inst_497_, lean_object* v_inst_498_, lean_object* v_P_499_, lean_object* v_Pbot_500_, lean_object* v_Psup_501_){
_start:
{
lean_object* v_res_502_; 
v_res_502_ = lp_mathlib_SupBotHom_subtypeVal(v_00_u03b2_496_, v_inst_497_, v_inst_498_, v_P_499_, v_Pbot_500_, v_Psup_501_);
lean_dec(v_inst_498_);
lean_dec_ref(v_inst_497_);
return v_res_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_subtypeVal(lean_object* v_00_u03b2_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_P_506_, lean_object* v_Pbot_507_, lean_object* v_Psup_508_){
_start:
{
lean_object* v___f_509_; 
v___f_509_ = ((lean_object*)(lp_mathlib_SupBotHom_subtypeVal___closed__0));
return v___f_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_subtypeVal___boxed(lean_object* v_00_u03b2_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_P_513_, lean_object* v_Pbot_514_, lean_object* v_Psup_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_mathlib_InfTopHom_subtypeVal(v_00_u03b2_510_, v_inst_511_, v_inst_512_, v_P_513_, v_Pbot_514_, v_Psup_515_);
lean_dec(v_inst_512_);
lean_dec_ref(v_inst_511_);
return v_res_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toBoundedOrderHom___redArg(lean_object* v_f_517_){
_start:
{
lean_inc(v_f_517_);
return v_f_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toBoundedOrderHom___redArg___boxed(lean_object* v_f_518_){
_start:
{
lean_object* v_res_519_; 
v_res_519_ = lp_mathlib_BoundedLatticeHom_toBoundedOrderHom___redArg(v_f_518_);
lean_dec(v_f_518_);
return v_res_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toBoundedOrderHom(lean_object* v_00_u03b1_520_, lean_object* v_00_u03b2_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_f_526_){
_start:
{
lean_inc(v_f_526_);
return v_f_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_toBoundedOrderHom___boxed(lean_object* v_00_u03b1_527_, lean_object* v_00_u03b2_528_, lean_object* v_inst_529_, lean_object* v_inst_530_, lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_f_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_mathlib_BoundedLatticeHom_toBoundedOrderHom(v_00_u03b1_527_, v_00_u03b2_528_, v_inst_529_, v_inst_530_, v_inst_531_, v_inst_532_, v_f_533_);
lean_dec(v_f_533_);
lean_dec_ref(v_inst_532_);
lean_dec_ref(v_inst_531_);
lean_dec_ref(v_inst_530_);
lean_dec_ref(v_inst_529_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_copy___redArg(lean_object* v_f_x27_535_){
_start:
{
lean_inc(v_f_x27_535_);
return v_f_x27_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_copy___redArg___boxed(lean_object* v_f_x27_536_){
_start:
{
lean_object* v_res_537_; 
v_res_537_ = lp_mathlib_BoundedLatticeHom_copy___redArg(v_f_x27_536_);
lean_dec(v_f_x27_536_);
return v_res_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_copy(lean_object* v_00_u03b1_538_, lean_object* v_00_u03b2_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_f_544_, lean_object* v_f_x27_545_, lean_object* v_h_546_){
_start:
{
lean_inc(v_f_x27_545_);
return v_f_x27_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_copy___boxed(lean_object* v_00_u03b1_547_, lean_object* v_00_u03b2_548_, lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_f_553_, lean_object* v_f_x27_554_, lean_object* v_h_555_){
_start:
{
lean_object* v_res_556_; 
v_res_556_ = lp_mathlib_BoundedLatticeHom_copy(v_00_u03b1_547_, v_00_u03b2_548_, v_inst_549_, v_inst_550_, v_inst_551_, v_inst_552_, v_f_553_, v_f_x27_554_, v_h_555_);
lean_dec(v_f_x27_554_);
lean_dec(v_f_553_);
lean_dec_ref(v_inst_552_);
lean_dec_ref(v_inst_551_);
lean_dec_ref(v_inst_550_);
lean_dec_ref(v_inst_549_);
return v_res_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_id(lean_object* v_00_u03b1_557_, lean_object* v_inst_558_, lean_object* v_inst_559_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = ((lean_object*)(lp_mathlib_SupBotHom_id___closed__0));
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_id___boxed(lean_object* v_00_u03b1_561_, lean_object* v_inst_562_, lean_object* v_inst_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_BoundedLatticeHom_id(v_00_u03b1_561_, v_inst_562_, v_inst_563_);
lean_dec_ref(v_inst_563_);
lean_dec_ref(v_inst_562_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_instInhabited(lean_object* v_00_u03b1_565_, lean_object* v_inst_566_, lean_object* v_inst_567_){
_start:
{
lean_object* v___x_568_; 
v___x_568_ = ((lean_object*)(lp_mathlib_SupBotHom_id___closed__0));
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_instInhabited___boxed(lean_object* v_00_u03b1_569_, lean_object* v_inst_570_, lean_object* v_inst_571_){
_start:
{
lean_object* v_res_572_; 
v_res_572_ = lp_mathlib_BoundedLatticeHom_instInhabited(v_00_u03b1_569_, v_inst_570_, v_inst_571_);
lean_dec_ref(v_inst_571_);
lean_dec_ref(v_inst_570_);
return v_res_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_comp___redArg(lean_object* v_f_573_, lean_object* v_g_574_){
_start:
{
lean_object* v___x_575_; 
v___x_575_ = lp_mathlib_SupHom_comp___redArg(v_f_573_, v_g_574_);
return v___x_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_comp(lean_object* v_00_u03b1_576_, lean_object* v_00_u03b2_577_, lean_object* v_00_u03b3_578_, lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_inst_583_, lean_object* v_inst_584_, lean_object* v_f_585_, lean_object* v_g_586_){
_start:
{
lean_object* v___x_587_; 
v___x_587_ = lp_mathlib_SupHom_comp___redArg(v_f_585_, v_g_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_comp___boxed(lean_object* v_00_u03b1_588_, lean_object* v_00_u03b2_589_, lean_object* v_00_u03b3_590_, lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_f_597_, lean_object* v_g_598_){
_start:
{
lean_object* v_res_599_; 
v_res_599_ = lp_mathlib_BoundedLatticeHom_comp(v_00_u03b1_588_, v_00_u03b2_589_, v_00_u03b3_590_, v_inst_591_, v_inst_592_, v_inst_593_, v_inst_594_, v_inst_595_, v_inst_596_, v_f_597_, v_g_598_);
lean_dec_ref(v_inst_596_);
lean_dec_ref(v_inst_595_);
lean_dec_ref(v_inst_594_);
lean_dec_ref(v_inst_593_);
lean_dec_ref(v_inst_592_);
lean_dec_ref(v_inst_591_);
return v_res_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_subtypeVal(lean_object* v_00_u03b2_600_, lean_object* v_inst_601_, lean_object* v_inst_602_, lean_object* v_P_603_, lean_object* v_Pbot_604_, lean_object* v_Ptop_605_, lean_object* v_Psup_606_, lean_object* v_Pinf_607_){
_start:
{
lean_object* v___f_608_; 
v___f_608_ = ((lean_object*)(lp_mathlib_SupBotHom_subtypeVal___closed__0));
return v___f_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_subtypeVal___boxed(lean_object* v_00_u03b2_609_, lean_object* v_inst_610_, lean_object* v_inst_611_, lean_object* v_P_612_, lean_object* v_Pbot_613_, lean_object* v_Ptop_614_, lean_object* v_Psup_615_, lean_object* v_Pinf_616_){
_start:
{
lean_object* v_res_617_; 
v_res_617_ = lp_mathlib_BoundedLatticeHom_subtypeVal(v_00_u03b2_609_, v_inst_610_, v_inst_611_, v_P_612_, v_Pbot_613_, v_Ptop_614_, v_Psup_615_, v_Pinf_616_);
lean_dec_ref(v_inst_611_);
lean_dec_ref(v_inst_610_);
return v_res_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg___lam__0(lean_object* v_inst_618_, lean_object* v_inst_619_, lean_object* v_f_620_, lean_object* v___y_621_){
_start:
{
lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v_toFun_624_; lean_object* v___x_625_; 
v___x_622_ = lp_mathlib_SupHom_dual(lean_box(0), lean_box(0), v_inst_618_, v_inst_619_);
v___x_623_ = lp_mathlib_Equiv_symm___redArg(v___x_622_);
v_toFun_624_ = lean_ctor_get(v___x_623_, 0);
lean_inc(v_toFun_624_);
lean_dec_ref(v___x_623_);
v___x_625_ = lean_apply_2(v_toFun_624_, v_f_620_, v___y_621_);
return v___x_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg___lam__0___boxed(lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_f_628_, lean_object* v___y_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_mathlib_SupBotHom_dual___redArg___lam__0(v_inst_626_, v_inst_627_, v_f_628_, v___y_629_);
lean_dec(v_inst_627_);
lean_dec(v_inst_626_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg___lam__1(lean_object* v_inst_631_, lean_object* v_inst_632_, lean_object* v_f_633_, lean_object* v___y_634_){
_start:
{
lean_object* v___x_635_; lean_object* v_toFun_636_; lean_object* v___x_637_; 
v___x_635_ = lp_mathlib_SupHom_dual(lean_box(0), lean_box(0), v_inst_631_, v_inst_632_);
v_toFun_636_ = lean_ctor_get(v___x_635_, 0);
lean_inc(v_toFun_636_);
lean_dec_ref(v___x_635_);
v___x_637_ = lean_apply_2(v_toFun_636_, v_f_633_, v___y_634_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg___lam__1___boxed(lean_object* v_inst_638_, lean_object* v_inst_639_, lean_object* v_f_640_, lean_object* v___y_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_mathlib_SupBotHom_dual___redArg___lam__1(v_inst_638_, v_inst_639_, v_f_640_, v___y_641_);
lean_dec(v_inst_639_);
lean_dec(v_inst_638_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___redArg(lean_object* v_inst_643_, lean_object* v_inst_644_){
_start:
{
lean_object* v___f_645_; lean_object* v___f_646_; lean_object* v___x_647_; 
lean_inc(v_inst_644_);
lean_inc(v_inst_643_);
v___f_645_ = lean_alloc_closure((void*)(lp_mathlib_SupBotHom_dual___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_645_, 0, v_inst_643_);
lean_closure_set(v___f_645_, 1, v_inst_644_);
v___f_646_ = lean_alloc_closure((void*)(lp_mathlib_SupBotHom_dual___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_646_, 0, v_inst_643_);
lean_closure_set(v___f_646_, 1, v_inst_644_);
v___x_647_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_647_, 0, v___f_646_);
lean_ctor_set(v___x_647_, 1, v___f_645_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual(lean_object* v_00_u03b1_648_, lean_object* v_00_u03b2_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_inst_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_mathlib_SupBotHom_dual___redArg(v_inst_650_, v_inst_652_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupBotHom_dual___boxed(lean_object* v_00_u03b1_655_, lean_object* v_00_u03b2_656_, lean_object* v_inst_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_inst_660_){
_start:
{
lean_object* v_res_661_; 
v_res_661_ = lp_mathlib_SupBotHom_dual(v_00_u03b1_655_, v_00_u03b2_656_, v_inst_657_, v_inst_658_, v_inst_659_, v_inst_660_);
lean_dec(v_inst_660_);
lean_dec(v_inst_658_);
return v_res_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg___lam__0(lean_object* v_inst_662_, lean_object* v_inst_663_, lean_object* v_f_664_, lean_object* v___y_665_){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v_toFun_668_; lean_object* v___x_669_; 
v___x_666_ = lp_mathlib_InfHom_dual(lean_box(0), lean_box(0), v_inst_662_, v_inst_663_);
v___x_667_ = lp_mathlib_Equiv_symm___redArg(v___x_666_);
v_toFun_668_ = lean_ctor_get(v___x_667_, 0);
lean_inc(v_toFun_668_);
lean_dec_ref(v___x_667_);
v___x_669_ = lean_apply_2(v_toFun_668_, v_f_664_, v___y_665_);
return v___x_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg___lam__0___boxed(lean_object* v_inst_670_, lean_object* v_inst_671_, lean_object* v_f_672_, lean_object* v___y_673_){
_start:
{
lean_object* v_res_674_; 
v_res_674_ = lp_mathlib_InfTopHom_dual___redArg___lam__0(v_inst_670_, v_inst_671_, v_f_672_, v___y_673_);
lean_dec(v_inst_671_);
lean_dec(v_inst_670_);
return v_res_674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg___lam__1(lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_f_677_, lean_object* v___y_678_){
_start:
{
lean_object* v___x_679_; lean_object* v_toFun_680_; lean_object* v___x_681_; 
v___x_679_ = lp_mathlib_InfHom_dual(lean_box(0), lean_box(0), v_inst_675_, v_inst_676_);
v_toFun_680_ = lean_ctor_get(v___x_679_, 0);
lean_inc(v_toFun_680_);
lean_dec_ref(v___x_679_);
v___x_681_ = lean_apply_2(v_toFun_680_, v_f_677_, v___y_678_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg___lam__1___boxed(lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_f_684_, lean_object* v___y_685_){
_start:
{
lean_object* v_res_686_; 
v_res_686_ = lp_mathlib_InfTopHom_dual___redArg___lam__1(v_inst_682_, v_inst_683_, v_f_684_, v___y_685_);
lean_dec(v_inst_683_);
lean_dec(v_inst_682_);
return v_res_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___redArg(lean_object* v_inst_687_, lean_object* v_inst_688_){
_start:
{
lean_object* v___f_689_; lean_object* v___f_690_; lean_object* v___x_691_; 
lean_inc(v_inst_688_);
lean_inc(v_inst_687_);
v___f_689_ = lean_alloc_closure((void*)(lp_mathlib_InfTopHom_dual___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_689_, 0, v_inst_687_);
lean_closure_set(v___f_689_, 1, v_inst_688_);
v___f_690_ = lean_alloc_closure((void*)(lp_mathlib_InfTopHom_dual___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_690_, 0, v_inst_687_);
lean_closure_set(v___f_690_, 1, v_inst_688_);
v___x_691_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_691_, 0, v___f_690_);
lean_ctor_set(v___x_691_, 1, v___f_689_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual(lean_object* v_00_u03b1_692_, lean_object* v_00_u03b2_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_inst_696_, lean_object* v_inst_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_mathlib_InfTopHom_dual___redArg(v_inst_694_, v_inst_696_);
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfTopHom_dual___boxed(lean_object* v_00_u03b1_699_, lean_object* v_00_u03b2_700_, lean_object* v_inst_701_, lean_object* v_inst_702_, lean_object* v_inst_703_, lean_object* v_inst_704_){
_start:
{
lean_object* v_res_705_; 
v_res_705_ = lp_mathlib_InfTopHom_dual(v_00_u03b1_699_, v_00_u03b2_700_, v_inst_701_, v_inst_702_, v_inst_703_, v_inst_704_);
lean_dec(v_inst_704_);
lean_dec(v_inst_702_);
return v_res_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual___redArg___lam__0(lean_object* v_inst_706_, lean_object* v_inst_707_, lean_object* v_f_708_, lean_object* v___y_709_){
_start:
{
lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v_toFun_712_; lean_object* v___x_713_; 
v___x_710_ = lp_mathlib_LatticeHom_dual___redArg(v_inst_706_, v_inst_707_);
v___x_711_ = lp_mathlib_Equiv_symm___redArg(v___x_710_);
v_toFun_712_ = lean_ctor_get(v___x_711_, 0);
lean_inc(v_toFun_712_);
lean_dec_ref(v___x_711_);
v___x_713_ = lean_apply_2(v_toFun_712_, v_f_708_, v___y_709_);
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual___redArg___lam__1(lean_object* v_inst_714_, lean_object* v_inst_715_, lean_object* v_f_716_, lean_object* v___y_717_){
_start:
{
lean_object* v___x_718_; lean_object* v_toFun_719_; lean_object* v___x_720_; 
v___x_718_ = lp_mathlib_LatticeHom_dual___redArg(v_inst_714_, v_inst_715_);
v_toFun_719_ = lean_ctor_get(v___x_718_, 0);
lean_inc(v_toFun_719_);
lean_dec_ref(v___x_718_);
v___x_720_ = lean_apply_2(v_toFun_719_, v_f_716_, v___y_717_);
return v___x_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual___redArg(lean_object* v_inst_721_, lean_object* v_inst_722_){
_start:
{
lean_object* v___f_723_; lean_object* v___f_724_; lean_object* v___x_725_; 
lean_inc_ref(v_inst_722_);
lean_inc_ref(v_inst_721_);
v___f_723_ = lean_alloc_closure((void*)(lp_mathlib_BoundedLatticeHom_dual___redArg___lam__0), 4, 2);
lean_closure_set(v___f_723_, 0, v_inst_721_);
lean_closure_set(v___f_723_, 1, v_inst_722_);
v___f_724_ = lean_alloc_closure((void*)(lp_mathlib_BoundedLatticeHom_dual___redArg___lam__1), 4, 2);
lean_closure_set(v___f_724_, 0, v_inst_721_);
lean_closure_set(v___f_724_, 1, v_inst_722_);
v___x_725_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_725_, 0, v___f_724_);
lean_ctor_set(v___x_725_, 1, v___f_723_);
return v___x_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual(lean_object* v_00_u03b1_726_, lean_object* v_00_u03b2_727_, lean_object* v_inst_728_, lean_object* v_inst_729_, lean_object* v_inst_730_, lean_object* v_inst_731_){
_start:
{
lean_object* v___x_732_; 
v___x_732_ = lp_mathlib_BoundedLatticeHom_dual___redArg(v_inst_728_, v_inst_730_);
return v___x_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedLatticeHom_dual___boxed(lean_object* v_00_u03b1_733_, lean_object* v_00_u03b2_734_, lean_object* v_inst_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_mathlib_BoundedLatticeHom_dual(v_00_u03b1_733_, v_00_u03b2_734_, v_inst_735_, v_inst_736_, v_inst_737_, v_inst_738_);
lean_dec_ref(v_inst_738_);
lean_dec_ref(v_inst_736_);
return v_res_739_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Bounded(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SymmDiff(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Hom_Bounded(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SymmDiff(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
}
#ifdef __cplusplus
}
#endif
