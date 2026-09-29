// Lean compiler output
// Module: Mathlib.Algebra.Ring.Subsemiring.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.BigOperators public import Mathlib.Algebra.Ring.Action.Subobjects public import Mathlib.Algebra.Ring.Equiv public import Mathlib.Algebra.Ring.Prod public import Mathlib.Algebra.Ring.Subsemiring.Defs public import Mathlib.GroupTheory.Submonoid.Centralizer public import Mathlib.RingTheory.NonUnitalSubsemiring.Basic public import Mathlib.Algebra.Module.Defs
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
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subsemiring_toSemiring___redArg(lean_object*);
lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___redArg(lean_object*);
lean_object* lp_mathlib_SubsemiringClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_domRestrict___redArg(lean_object*);
lean_object* lp_mathlib_Subsemiring_instPartialOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubsemiringClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubsemiringClass_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Subsemiring_topEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subsemiring_topEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subsemiring_topEquiv___closed__0 = (const lean_object*)&lp_mathlib_Subsemiring_topEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Subsemiring_topEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Subsemiring_topEquiv___closed__0_value),((lean_object*)&lp_mathlib_Subsemiring_topEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Subsemiring_topEquiv___closed__1 = (const lean_object*)&lp_mathlib_Subsemiring_topEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRangeS___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRangeS___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRangeS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRangeS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Subsemiring_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subsemiring_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subsemiring_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_Subsemiring_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subsemiring_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subsemiring_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subsemiring_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_Subsemiring_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Subsemiring_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Subsemiring_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_Subsemiring_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center_commSemiring_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center_commSemiring_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerToMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerToMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center_commSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center_commSemiring(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subsemiring_decidableMemCenter___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_decidableMemCenter___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subsemiring_decidableMemCenter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_decidableMemCenter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subsemiring_decidableMemCenter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_decidableMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_subsemiringClosure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_subsemiringClosure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subsemiring_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subsemiring_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subsemiring_gi___closed__0 = (const lean_object*)&lp_mathlib_Subsemiring_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subsemiring_prodEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subsemiring_prodEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_restrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_restrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeSRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeSRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeSRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subsemiring_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubsemiringClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subsemiring_inclusion___closed__0 = (const lean_object*)&lp_mathlib_Subsemiring_inclusion___closed__0_value;
static const lean_closure_object lp_mathlib_Subsemiring_inclusion___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_codRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Subsemiring_inclusion___closed__0_value)} };
static const lean_object* lp_mathlib_Subsemiring_inclusion___closed__1 = (const lean_object*)&lp_mathlib_Subsemiring_inclusion___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_subsemiringCongr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_subsemiringCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_smul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulActionWithZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulActionWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulActionWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulActionWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulActionWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instModuleSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instModuleSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instModuleSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulSemiringActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulSemiringActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulSemiringActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closureCommSemiringOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closureCommSemiringOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topEquiv___lam__0(lean_object* v_r_1_){
_start:
{
lean_inc(v_r_1_);
return v_r_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topEquiv___lam__0___boxed(lean_object* v_r_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Subsemiring_topEquiv___lam__0(v_r_2_);
lean_dec(v_r_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topEquiv(lean_object* v_R_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = ((lean_object*)(lp_mathlib_Subsemiring_topEquiv___closed__1));
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_topEquiv___boxed(lean_object* v_R_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Subsemiring_topEquiv(v_R_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_comap(lean_object* v_R_13_, lean_object* v_S_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_f_17_, lean_object* v_s_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_box(0);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_comap___boxed(lean_object* v_R_20_, lean_object* v_S_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_f_24_, lean_object* v_s_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Subsemiring_comap(v_R_20_, v_S_21_, v_inst_22_, v_inst_23_, v_f_24_, v_s_25_);
lean_dec(v_f_24_);
lean_dec_ref(v_inst_23_);
lean_dec_ref(v_inst_22_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_map(lean_object* v_R_27_, lean_object* v_S_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_f_31_, lean_object* v_s_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_box(0);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_map___boxed(lean_object* v_R_34_, lean_object* v_S_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_f_38_, lean_object* v_s_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_Subsemiring_map(v_R_34_, v_S_35_, v_inst_36_, v_inst_37_, v_f_38_, v_s_39_);
lean_dec(v_f_38_);
lean_dec_ref(v_inst_37_);
lean_dec_ref(v_inst_36_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeS(lean_object* v_R_41_, lean_object* v_S_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_f_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeS___boxed(lean_object* v_R_47_, lean_object* v_S_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_f_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_RingHom_rangeS(v_R_47_, v_S_48_, v_inst_49_, v_inst_50_, v_f_51_);
lean_dec(v_f_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRangeS___redArg___lam__0(lean_object* v_f_53_, lean_object* v___y_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_apply_1(v_f_53_, v___y_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRangeS___redArg(lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_f_58_){
_start:
{
lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_fintypeRangeS___redArg___lam__0), 2, 1);
lean_closure_set(v___f_59_, 0, v_f_58_);
v___x_60_ = lp_mathlib_PLift_fintype___redArg(v_inst_56_);
v___x_61_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_57_, v___f_59_, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRangeS(lean_object* v_R_62_, lean_object* v_S_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_f_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_RingHom_fintypeRangeS___redArg(v_inst_66_, v_inst_67_, v_f_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRangeS___boxed(lean_object* v_R_70_, lean_object* v_S_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_f_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_RingHom_fintypeRangeS(v_R_70_, v_S_71_, v_inst_72_, v_inst_73_, v_inst_74_, v_inst_75_, v_f_76_);
lean_dec_ref(v_inst_73_);
lean_dec_ref(v_inst_72_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instBot(lean_object* v_R_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lean_box(0);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instBot___boxed(lean_object* v_R_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Subsemiring_instBot(v_R_81_, v_inst_82_);
lean_dec_ref(v_inst_82_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInhabited(lean_object* v_R_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_box(0);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInhabited___boxed(lean_object* v_R_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Subsemiring_instInhabited(v_R_87_, v_inst_88_);
lean_dec_ref(v_inst_88_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInfSet___lam__0(lean_object* v_s_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lean_box(0);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInfSet(lean_object* v_R_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___f_95_; 
v___f_95_ = ((lean_object*)(lp_mathlib_Subsemiring_instInfSet___closed__0));
return v___f_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instInfSet___boxed(lean_object* v_R_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_Subsemiring_instInfSet(v_R_96_, v_inst_97_);
lean_dec_ref(v_inst_97_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice___redArg___lam__0(lean_object* v_x1_99_, lean_object* v_x2_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lean_box(0);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice___redArg(lean_object* v_inst_105_){
_start:
{
lean_object* v___x_106_; lean_object* v___f_107_; lean_object* v___x_108_; lean_object* v_toLattice_109_; lean_object* v_toSupSet_110_; lean_object* v_toInfSet_111_; lean_object* v___x_113_; uint8_t v_isShared_114_; uint8_t v_isSharedCheck_129_; 
v___x_106_ = lp_mathlib_Subsemiring_instPartialOrder(lean_box(0), v_inst_105_);
v___f_107_ = ((lean_object*)(lp_mathlib_Subsemiring_instInfSet___closed__0));
v___x_108_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_106_, v___f_107_);
v_toLattice_109_ = lean_ctor_get(v___x_108_, 0);
v_toSupSet_110_ = lean_ctor_get(v___x_108_, 1);
v_toInfSet_111_ = lean_ctor_get(v___x_108_, 2);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_129_ == 0)
{
lean_object* v_unused_130_; 
v_unused_130_ = lean_ctor_get(v___x_108_, 3);
lean_dec(v_unused_130_);
v___x_113_ = v___x_108_;
v_isShared_114_ = v_isSharedCheck_129_;
goto v_resetjp_112_;
}
else
{
lean_inc(v_toInfSet_111_);
lean_inc(v_toSupSet_110_);
lean_inc(v_toLattice_109_);
lean_dec(v___x_108_);
v___x_113_ = lean_box(0);
v_isShared_114_ = v_isSharedCheck_129_;
goto v_resetjp_112_;
}
v_resetjp_112_:
{
lean_object* v_toSemilatticeSup_115_; lean_object* v___x_117_; uint8_t v_isShared_118_; uint8_t v_isSharedCheck_127_; 
v_toSemilatticeSup_115_ = lean_ctor_get(v_toLattice_109_, 0);
v_isSharedCheck_127_ = !lean_is_exclusive(v_toLattice_109_);
if (v_isSharedCheck_127_ == 0)
{
lean_object* v_unused_128_; 
v_unused_128_ = lean_ctor_get(v_toLattice_109_, 1);
lean_dec(v_unused_128_);
v___x_117_ = v_toLattice_109_;
v_isShared_118_ = v_isSharedCheck_127_;
goto v_resetjp_116_;
}
else
{
lean_inc(v_toSemilatticeSup_115_);
lean_dec(v_toLattice_109_);
v___x_117_ = lean_box(0);
v_isShared_118_ = v_isSharedCheck_127_;
goto v_resetjp_116_;
}
v_resetjp_116_:
{
lean_object* v___f_119_; lean_object* v___x_121_; 
v___f_119_ = ((lean_object*)(lp_mathlib_Subsemiring_instCompleteLattice___redArg___closed__0));
if (v_isShared_118_ == 0)
{
lean_ctor_set(v___x_117_, 1, v___f_119_);
v___x_121_ = v___x_117_;
goto v_reusejp_120_;
}
else
{
lean_object* v_reuseFailAlloc_126_; 
v_reuseFailAlloc_126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_126_, 0, v_toSemilatticeSup_115_);
lean_ctor_set(v_reuseFailAlloc_126_, 1, v___f_119_);
v___x_121_ = v_reuseFailAlloc_126_;
goto v_reusejp_120_;
}
v_reusejp_120_:
{
lean_object* v___x_122_; lean_object* v___x_124_; 
v___x_122_ = ((lean_object*)(lp_mathlib_Subsemiring_instCompleteLattice___redArg___closed__1));
if (v_isShared_114_ == 0)
{
lean_ctor_set(v___x_113_, 3, v___x_122_);
lean_ctor_set(v___x_113_, 0, v___x_121_);
v___x_124_ = v___x_113_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v___x_121_);
lean_ctor_set(v_reuseFailAlloc_125_, 1, v_toSupSet_110_);
lean_ctor_set(v_reuseFailAlloc_125_, 2, v_toInfSet_111_);
lean_ctor_set(v_reuseFailAlloc_125_, 3, v___x_122_);
v___x_124_ = v_reuseFailAlloc_125_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
return v___x_124_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice___redArg___boxed(lean_object* v_inst_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Subsemiring_instCompleteLattice___redArg(v_inst_131_);
lean_dec_ref(v_inst_131_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice(lean_object* v_R_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_Subsemiring_instCompleteLattice___redArg(v_inst_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instCompleteLattice___boxed(lean_object* v_R_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Subsemiring_instCompleteLattice(v_R_136_, v_inst_137_);
lean_dec_ref(v_inst_137_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center(lean_object* v_R_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lean_box(0);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center___boxed(lean_object* v_R_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Subsemiring_center(v_R_142_, v_inst_143_);
lean_dec_ref(v_inst_143_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center_commSemiring_x27___redArg(lean_object* v_inst_145_){
_start:
{
lean_object* v___x_146_; lean_object* v_toMulOneClass_147_; lean_object* v___x_148_; lean_object* v_toOne_149_; lean_object* v_toMul_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v_toNonUnitalNonAssocSemiring_154_; lean_object* v_toNatCast_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_163_; 
lean_inc_ref(v_inst_145_);
v___x_146_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_145_);
v_toMulOneClass_147_ = lean_ctor_get(v___x_146_, 0);
lean_inc_ref(v_toMulOneClass_147_);
lean_dec_ref(v___x_146_);
v___x_148_ = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(v_toMulOneClass_147_);
v_toOne_149_ = lean_ctor_get(v___x_148_, 0);
lean_inc_n(v_toOne_149_, 2);
v_toMul_150_ = lean_ctor_get(v___x_148_, 1);
lean_inc_n(v_toMul_150_, 2);
lean_dec_ref(v___x_148_);
v___x_151_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_151_, 0, lean_box(0));
lean_closure_set(v___x_151_, 1, v_toMul_150_);
lean_closure_set(v___x_151_, 2, v_toOne_149_);
v___x_152_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_152_, 0, v_toOne_149_);
lean_ctor_set(v___x_152_, 1, v_toMul_150_);
lean_ctor_set(v___x_152_, 2, v___x_151_);
v___x_153_ = lp_mathlib_SubsemiringClass_toNonAssocSemiring___redArg(v_inst_145_);
v_toNonUnitalNonAssocSemiring_154_ = lean_ctor_get(v___x_153_, 0);
v_toNatCast_155_ = lean_ctor_get(v___x_153_, 2);
v_isSharedCheck_163_ = !lean_is_exclusive(v___x_153_);
if (v_isSharedCheck_163_ == 0)
{
lean_object* v_unused_164_; 
v_unused_164_ = lean_ctor_get(v___x_153_, 1);
lean_dec(v_unused_164_);
v___x_157_ = v___x_153_;
v_isShared_158_ = v_isSharedCheck_163_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_toNatCast_155_);
lean_inc(v_toNonUnitalNonAssocSemiring_154_);
lean_dec(v___x_153_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_163_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v_toAddCommMonoid_159_; lean_object* v___x_161_; 
v_toAddCommMonoid_159_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_154_, 0);
lean_inc_ref(v_toAddCommMonoid_159_);
lean_dec_ref(v_toNonUnitalNonAssocSemiring_154_);
if (v_isShared_158_ == 0)
{
lean_ctor_set(v___x_157_, 1, v___x_152_);
lean_ctor_set(v___x_157_, 0, v_toAddCommMonoid_159_);
v___x_161_ = v___x_157_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_toAddCommMonoid_159_);
lean_ctor_set(v_reuseFailAlloc_162_, 1, v___x_152_);
lean_ctor_set(v_reuseFailAlloc_162_, 2, v_toNatCast_155_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center_commSemiring_x27(lean_object* v_R_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v___x_167_; lean_object* v_toMulOneClass_168_; lean_object* v___x_169_; lean_object* v_toOne_170_; lean_object* v_toMul_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v_toNonUnitalNonAssocSemiring_175_; lean_object* v_toNatCast_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_184_; 
lean_inc_ref(v_inst_166_);
v___x_167_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_166_);
v_toMulOneClass_168_ = lean_ctor_get(v___x_167_, 0);
lean_inc_ref(v_toMulOneClass_168_);
lean_dec_ref(v___x_167_);
v___x_169_ = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(v_toMulOneClass_168_);
v_toOne_170_ = lean_ctor_get(v___x_169_, 0);
lean_inc_n(v_toOne_170_, 2);
v_toMul_171_ = lean_ctor_get(v___x_169_, 1);
lean_inc_n(v_toMul_171_, 2);
lean_dec_ref(v___x_169_);
v___x_172_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_172_, 0, lean_box(0));
lean_closure_set(v___x_172_, 1, v_toMul_171_);
lean_closure_set(v___x_172_, 2, v_toOne_170_);
v___x_173_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_173_, 0, v_toOne_170_);
lean_ctor_set(v___x_173_, 1, v_toMul_171_);
lean_ctor_set(v___x_173_, 2, v___x_172_);
v___x_174_ = lp_mathlib_SubsemiringClass_toNonAssocSemiring___redArg(v_inst_166_);
v_toNonUnitalNonAssocSemiring_175_ = lean_ctor_get(v___x_174_, 0);
v_toNatCast_176_ = lean_ctor_get(v___x_174_, 2);
v_isSharedCheck_184_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_184_ == 0)
{
lean_object* v_unused_185_; 
v_unused_185_ = lean_ctor_get(v___x_174_, 1);
lean_dec(v_unused_185_);
v___x_178_ = v___x_174_;
v_isShared_179_ = v_isSharedCheck_184_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_toNatCast_176_);
lean_inc(v_toNonUnitalNonAssocSemiring_175_);
lean_dec(v___x_174_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_184_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v_toAddCommMonoid_180_; lean_object* v___x_182_; 
v_toAddCommMonoid_180_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_175_, 0);
lean_inc_ref(v_toAddCommMonoid_180_);
lean_dec_ref(v_toNonUnitalNonAssocSemiring_175_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 1, v___x_173_);
lean_ctor_set(v___x_178_, 0, v_toAddCommMonoid_180_);
v___x_182_ = v___x_178_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_183_; 
v_reuseFailAlloc_183_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_183_, 0, v_toAddCommMonoid_180_);
lean_ctor_set(v_reuseFailAlloc_183_, 1, v___x_173_);
lean_ctor_set(v_reuseFailAlloc_183_, 2, v_toNatCast_176_);
v___x_182_ = v_reuseFailAlloc_183_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
return v___x_182_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerCongr___redArg(lean_object* v_e_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(v_e_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerCongr(lean_object* v_R_188_, lean_object* v_S_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_e_192_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(v_e_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerCongr___boxed(lean_object* v_R_194_, lean_object* v_S_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_e_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_Subsemiring_centerCongr(v_R_194_, v_S_195_, v_inst_196_, v_inst_197_, v_e_198_);
lean_dec_ref(v_inst_197_);
lean_dec_ref(v_inst_196_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerToMulOpposite___redArg(lean_object* v_inst_200_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_201_; lean_object* v___x_202_; 
v_toNonUnitalNonAssocSemiring_201_ = lean_ctor_get(v_inst_200_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_201_);
lean_dec_ref(v_inst_200_);
v___x_202_ = lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(v_toNonUnitalNonAssocSemiring_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centerToMulOpposite(lean_object* v_R_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_Subsemiring_centerToMulOpposite___redArg(v_inst_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center_commSemiring___redArg(lean_object* v_inst_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_center_commSemiring(lean_object* v_R_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_209_);
return v___x_210_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subsemiring_decidableMemCenter___redArg___lam__0(lean_object* v_toMul_211_, lean_object* v_x_212_, lean_object* v_inst_213_, lean_object* v_a_214_){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; uint8_t v___x_218_; 
lean_inc(v_toMul_211_);
lean_inc(v_x_212_);
lean_inc(v_a_214_);
v___x_215_ = lean_apply_2(v_toMul_211_, v_a_214_, v_x_212_);
v___x_216_ = lean_apply_2(v_toMul_211_, v_x_212_, v_a_214_);
v___x_217_ = lean_apply_2(v_inst_213_, v___x_215_, v___x_216_);
v___x_218_ = lean_unbox(v___x_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_decidableMemCenter___redArg___lam__0___boxed(lean_object* v_toMul_219_, lean_object* v_x_220_, lean_object* v_inst_221_, lean_object* v_a_222_){
_start:
{
uint8_t v_res_223_; lean_object* v_r_224_; 
v_res_223_ = lp_mathlib_Subsemiring_decidableMemCenter___redArg___lam__0(v_toMul_219_, v_x_220_, v_inst_221_, v_a_222_);
v_r_224_ = lean_box(v_res_223_);
return v_r_224_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subsemiring_decidableMemCenter___redArg(lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_x_228_){
_start:
{
lean_object* v___x_229_; lean_object* v_toMul_230_; lean_object* v___f_231_; uint8_t v___x_232_; 
v___x_229_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_225_);
v_toMul_230_ = lean_ctor_get(v___x_229_, 0);
lean_inc(v_toMul_230_);
lean_dec_ref(v___x_229_);
v___f_231_ = lean_alloc_closure((void*)(lp_mathlib_Subsemiring_decidableMemCenter___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_231_, 0, v_toMul_230_);
lean_closure_set(v___f_231_, 1, v_x_228_);
lean_closure_set(v___f_231_, 2, v_inst_226_);
v___x_232_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_231_, v_inst_227_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_decidableMemCenter___redArg___boxed(lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_x_236_){
_start:
{
uint8_t v_res_237_; lean_object* v_r_238_; 
v_res_237_ = lp_mathlib_Subsemiring_decidableMemCenter___redArg(v_inst_233_, v_inst_234_, v_inst_235_, v_x_236_);
v_r_238_ = lean_box(v_res_237_);
return v_r_238_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subsemiring_decidableMemCenter(lean_object* v_R_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_x_243_){
_start:
{
uint8_t v___x_244_; 
v___x_244_ = lp_mathlib_Subsemiring_decidableMemCenter___redArg(v_inst_240_, v_inst_241_, v_inst_242_, v_x_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_decidableMemCenter___boxed(lean_object* v_R_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_x_249_){
_start:
{
uint8_t v_res_250_; lean_object* v_r_251_; 
v_res_250_ = lp_mathlib_Subsemiring_decidableMemCenter(v_R_245_, v_inst_246_, v_inst_247_, v_inst_248_, v_x_249_);
v_r_251_ = lean_box(v_res_250_);
return v_r_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centralizer(lean_object* v_R_252_, lean_object* v_inst_253_, lean_object* v_s_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lean_box(0);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_centralizer___boxed(lean_object* v_R_256_, lean_object* v_inst_257_, lean_object* v_s_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_Subsemiring_centralizer(v_R_256_, v_inst_257_, v_s_258_);
lean_dec_ref(v_inst_257_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closure(lean_object* v_R_260_, lean_object* v_inst_261_, lean_object* v_s_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lean_box(0);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closure___boxed(lean_object* v_R_264_, lean_object* v_inst_265_, lean_object* v_s_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_Subsemiring_closure(v_R_264_, v_inst_265_, v_s_266_);
lean_dec_ref(v_inst_265_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_subsemiringClosure(lean_object* v_R_268_, lean_object* v_inst_269_, lean_object* v_M_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lean_box(0);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_subsemiringClosure___boxed(lean_object* v_R_272_, lean_object* v_inst_273_, lean_object* v_M_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_mathlib_Submonoid_subsemiringClosure(v_R_272_, v_inst_273_, v_M_274_);
lean_dec_ref(v_inst_273_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_gi___lam__0(lean_object* v_s_276_, lean_object* v_x_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lean_box(0);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_gi(lean_object* v_R_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___f_282_; 
v___f_282_ = ((lean_object*)(lp_mathlib_Subsemiring_gi___closed__0));
return v___f_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_gi___boxed(lean_object* v_R_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_Subsemiring_gi(v_R_283_, v_inst_284_);
lean_dec_ref(v_inst_284_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_prod(lean_object* v_R_286_, lean_object* v_S_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_s_290_, lean_object* v_t_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lean_box(0);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_prod___boxed(lean_object* v_R_293_, lean_object* v_S_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_s_297_, lean_object* v_t_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_Subsemiring_prod(v_R_293_, v_S_294_, v_inst_295_, v_inst_296_, v_s_297_, v_t_298_);
lean_dec_ref(v_inst_296_);
lean_dec_ref(v_inst_295_);
return v_res_299_;
}
}
static lean_object* _init_lp_mathlib_Subsemiring_prodEquiv___closed__0(void){
_start:
{
lean_object* v___x_300_; 
v___x_300_ = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_prodEquiv(lean_object* v_R_301_, lean_object* v_S_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_s_305_, lean_object* v_t_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lean_obj_once(&lp_mathlib_Subsemiring_prodEquiv___closed__0, &lp_mathlib_Subsemiring_prodEquiv___closed__0_once, _init_lp_mathlib_Subsemiring_prodEquiv___closed__0);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_prodEquiv___boxed(lean_object* v_R_308_, lean_object* v_S_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_s_312_, lean_object* v_t_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_Subsemiring_prodEquiv(v_R_308_, v_S_309_, v_inst_310_, v_inst_311_, v_s_312_, v_t_313_);
lean_dec_ref(v_inst_311_);
lean_dec_ref(v_inst_310_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_codRestrict___redArg___lam__0(lean_object* v_f_315_, lean_object* v_n_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lean_apply_1(v_f_315_, v_n_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_codRestrict___redArg(lean_object* v_f_318_){
_start:
{
lean_object* v___f_319_; 
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_319_, 0, v_f_318_);
return v___f_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_codRestrict(lean_object* v_R_320_, lean_object* v_S_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_00_u03c3S_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_f_327_, lean_object* v_s_328_, lean_object* v_h_329_){
_start:
{
lean_object* v___f_330_; 
v___f_330_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_330_, 0, v_f_327_);
return v___f_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_codRestrict___boxed(lean_object* v_R_331_, lean_object* v_S_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_00_u03c3S_335_, lean_object* v_inst_336_, lean_object* v_inst_337_, lean_object* v_f_338_, lean_object* v_s_339_, lean_object* v_h_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_RingHom_codRestrict(v_R_331_, v_S_332_, v_inst_333_, v_inst_334_, v_00_u03c3S_335_, v_inst_336_, v_inst_337_, v_f_338_, v_s_339_, v_h_340_);
lean_dec(v_s_339_);
lean_dec_ref(v_inst_334_);
lean_dec_ref(v_inst_333_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_restrict___redArg(lean_object* v_f_342_){
_start:
{
lean_object* v___x_343_; lean_object* v___f_344_; 
v___x_343_ = lp_mathlib_RingHom_domRestrict___redArg(v_f_342_);
v___f_344_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_344_, 0, v___x_343_);
return v___f_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_restrict(lean_object* v_R_345_, lean_object* v_S_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_00_u03c3R_349_, lean_object* v_00_u03c3S_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_f_355_, lean_object* v_s_x27_356_, lean_object* v_s_357_, lean_object* v_h_358_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lp_mathlib_RingHom_restrict___redArg(v_f_355_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_restrict___boxed(lean_object* v_R_360_, lean_object* v_S_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_00_u03c3R_364_, lean_object* v_00_u03c3S_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_f_370_, lean_object* v_s_x27_371_, lean_object* v_s_372_, lean_object* v_h_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_mathlib_RingHom_restrict(v_R_360_, v_S_361_, v_inst_362_, v_inst_363_, v_00_u03c3R_364_, v_00_u03c3S_365_, v_inst_366_, v_inst_367_, v_inst_368_, v_inst_369_, v_f_370_, v_s_x27_371_, v_s_372_, v_h_373_);
lean_dec(v_s_372_);
lean_dec(v_s_x27_371_);
lean_dec_ref(v_inst_363_);
lean_dec_ref(v_inst_362_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeSRestrict___redArg(lean_object* v_f_375_){
_start:
{
lean_object* v___f_376_; 
v___f_376_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_376_, 0, v_f_375_);
return v___f_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeSRestrict(lean_object* v_R_377_, lean_object* v_S_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_f_381_){
_start:
{
lean_object* v___f_382_; 
v___f_382_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_382_, 0, v_f_381_);
return v___f_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeSRestrict___boxed(lean_object* v_R_383_, lean_object* v_S_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_f_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_mathlib_RingHom_rangeSRestrict(v_R_383_, v_S_384_, v_inst_385_, v_inst_386_, v_f_387_);
lean_dec_ref(v_inst_386_);
lean_dec_ref(v_inst_385_);
return v_res_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_inclusion(lean_object* v_R_392_, lean_object* v_inst_393_, lean_object* v_S_394_, lean_object* v_T_395_, lean_object* v_h_396_){
_start:
{
lean_object* v___f_397_; 
v___f_397_ = ((lean_object*)(lp_mathlib_Subsemiring_inclusion___closed__1));
return v___f_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_inclusion___boxed(lean_object* v_R_398_, lean_object* v_inst_399_, lean_object* v_S_400_, lean_object* v_T_401_, lean_object* v_h_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_Subsemiring_inclusion(v_R_398_, v_inst_399_, v_S_400_, v_T_401_, v_h_402_);
lean_dec_ref(v_inst_399_);
return v_res_403_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_subsemiringCongr___closed__0(void){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringCongr(lean_object* v_R_405_, lean_object* v_inst_406_, lean_object* v_s_407_, lean_object* v_t_408_, lean_object* v_h_409_){
_start:
{
lean_object* v___x_410_; 
v___x_410_ = lean_obj_once(&lp_mathlib_RingEquiv_subsemiringCongr___closed__0, &lp_mathlib_RingEquiv_subsemiringCongr___closed__0_once, _init_lp_mathlib_RingEquiv_subsemiringCongr___closed__0);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringCongr___boxed(lean_object* v_R_411_, lean_object* v_inst_412_, lean_object* v_s_413_, lean_object* v_t_414_, lean_object* v_h_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_RingEquiv_subsemiringCongr(v_R_411_, v_inst_412_, v_s_413_, v_t_414_, v_h_415_);
lean_dec_ref(v_inst_412_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__0(lean_object* v_f_417_, lean_object* v_x_418_){
_start:
{
lean_object* v___x_419_; 
v___x_419_ = lean_apply_1(v_f_417_, v_x_418_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__1(lean_object* v_g_420_, lean_object* v_x_421_){
_start:
{
lean_object* v___x_422_; lean_object* v___x_423_; 
v___x_422_ = lp_mathlib_SubsemiringClass_subtype___lam__0(v_x_421_);
v___x_423_ = lean_apply_1(v_g_420_, v___x_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__1___boxed(lean_object* v_g_424_, lean_object* v_x_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__1(v_g_424_, v_x_425_);
lean_dec(v_x_425_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___redArg(lean_object* v_g_427_, lean_object* v_f_428_){
_start:
{
lean_object* v___f_429_; lean_object* v___f_430_; lean_object* v___x_431_; 
v___f_429_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__0), 2, 1);
lean_closure_set(v___f_429_, 0, v_f_428_);
v___f_430_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofLeftInverseS___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_430_, 0, v_g_427_);
v___x_431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_431_, 0, v___f_429_);
lean_ctor_set(v___x_431_, 1, v___f_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS(lean_object* v_R_432_, lean_object* v_S_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_g_436_, lean_object* v_f_437_, lean_object* v_h_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_RingEquiv_ofLeftInverseS___redArg(v_g_436_, v_f_437_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverseS___boxed(lean_object* v_R_440_, lean_object* v_S_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_g_444_, lean_object* v_f_445_, lean_object* v_h_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib_RingEquiv_ofLeftInverseS(v_R_440_, v_S_441_, v_inst_442_, v_inst_443_, v_g_444_, v_f_445_, v_h_446_);
lean_dec_ref(v_inst_443_);
lean_dec_ref(v_inst_442_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringMap___redArg(lean_object* v_e_448_){
_start:
{
lean_object* v___x_449_; 
v___x_449_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringMap(lean_object* v_R_450_, lean_object* v_S_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_e_454_, lean_object* v_s_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_454_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subsemiringMap___boxed(lean_object* v_R_457_, lean_object* v_S_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_e_461_, lean_object* v_s_462_){
_start:
{
lean_object* v_res_463_; 
v_res_463_ = lp_mathlib_RingEquiv_subsemiringMap(v_R_457_, v_S_458_, v_inst_459_, v_inst_460_, v_e_461_, v_s_462_);
lean_dec_ref(v_inst_460_);
lean_dec_ref(v_inst_459_);
return v_res_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_smul___redArg(lean_object* v_inst_464_){
_start:
{
lean_object* v___f_465_; 
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_465_, 0, v_inst_464_);
return v___f_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_smul(lean_object* v_R_x27_466_, lean_object* v_00_u03b1_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_S_470_){
_start:
{
lean_object* v___f_471_; 
v___f_471_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_471_, 0, v_inst_469_);
return v___f_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_smul___boxed(lean_object* v_R_x27_472_, lean_object* v_00_u03b1_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_S_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_mathlib_Subsemiring_smul(v_R_x27_472_, v_00_u03b1_473_, v_inst_474_, v_inst_475_, v_S_476_);
lean_dec_ref(v_inst_474_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem___redArg(lean_object* v_inst_478_){
_start:
{
lean_object* v___f_479_; 
v___f_479_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_479_, 0, v_inst_478_);
return v___f_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem(lean_object* v_R_x27_480_, lean_object* v_00_u03b1_481_, lean_object* v_inst_482_, lean_object* v_S_x27_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_s_486_, lean_object* v_inst_487_, lean_object* v_inst_488_){
_start:
{
lean_object* v___f_489_; 
v___f_489_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_489_, 0, v_inst_488_);
return v___f_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem___boxed(lean_object* v_R_x27_490_, lean_object* v_00_u03b1_491_, lean_object* v_inst_492_, lean_object* v_S_x27_493_, lean_object* v_inst_494_, lean_object* v_inst_495_, lean_object* v_s_496_, lean_object* v_inst_497_, lean_object* v_inst_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem(v_R_x27_490_, v_00_u03b1_491_, v_inst_492_, v_S_x27_493_, v_inst_494_, v_inst_495_, v_s_496_, v_inst_497_, v_inst_498_);
lean_dec(v_inst_497_);
lean_dec(v_s_496_);
lean_dec_ref(v_inst_492_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem__1___redArg(lean_object* v_inst_500_){
_start:
{
lean_object* v___f_501_; 
v___f_501_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_501_, 0, v_inst_500_);
return v___f_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem__1(lean_object* v_R_x27_502_, lean_object* v_00_u03b1_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_S_507_){
_start:
{
lean_object* v___f_508_; 
v___f_508_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_508_, 0, v_inst_506_);
return v___f_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem__1___boxed(lean_object* v_R_x27_509_, lean_object* v_00_u03b1_510_, lean_object* v_inst_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_S_514_){
_start:
{
lean_object* v_res_515_; 
v_res_515_ = lp_mathlib_Subsemiring_instSMulWithZeroSubtypeMem__1(v_R_x27_509_, v_00_u03b1_510_, v_inst_511_, v_inst_512_, v_inst_513_, v_S_514_);
lean_dec(v_inst_512_);
lean_dec_ref(v_inst_511_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulAction___redArg(lean_object* v_inst_516_){
_start:
{
lean_object* v___f_517_; 
v___f_517_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_517_, 0, v_inst_516_);
return v___f_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulAction(lean_object* v_R_x27_518_, lean_object* v_00_u03b1_519_, lean_object* v_inst_520_, lean_object* v_inst_521_, lean_object* v_S_522_){
_start:
{
lean_object* v___f_523_; 
v___f_523_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_523_, 0, v_inst_521_);
return v___f_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulAction___boxed(lean_object* v_R_x27_524_, lean_object* v_00_u03b1_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_S_528_){
_start:
{
lean_object* v_res_529_; 
v_res_529_ = lp_mathlib_Subsemiring_mulAction(v_R_x27_524_, v_00_u03b1_525_, v_inst_526_, v_inst_527_, v_S_528_);
lean_dec_ref(v_inst_526_);
return v_res_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_distribMulAction___redArg(lean_object* v_inst_530_){
_start:
{
lean_object* v___f_531_; 
v___f_531_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_531_, 0, v_inst_530_);
return v___f_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_distribMulAction(lean_object* v_R_x27_532_, lean_object* v_00_u03b1_533_, lean_object* v_inst_534_, lean_object* v_inst_535_, lean_object* v_inst_536_, lean_object* v_S_537_){
_start:
{
lean_object* v___f_538_; 
v___f_538_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_538_, 0, v_inst_536_);
return v___f_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_distribMulAction___boxed(lean_object* v_R_x27_539_, lean_object* v_00_u03b1_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_S_544_){
_start:
{
lean_object* v_res_545_; 
v_res_545_ = lp_mathlib_Subsemiring_distribMulAction(v_R_x27_539_, v_00_u03b1_540_, v_inst_541_, v_inst_542_, v_inst_543_, v_S_544_);
lean_dec_ref(v_inst_542_);
lean_dec_ref(v_inst_541_);
return v_res_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulDistribMulAction___redArg(lean_object* v_inst_546_){
_start:
{
lean_object* v___f_547_; 
v___f_547_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_547_, 0, v_inst_546_);
return v___f_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulDistribMulAction(lean_object* v_R_x27_548_, lean_object* v_00_u03b1_549_, lean_object* v_inst_550_, lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_S_553_){
_start:
{
lean_object* v___f_554_; 
v___f_554_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_554_, 0, v_inst_552_);
return v___f_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulDistribMulAction___boxed(lean_object* v_R_x27_555_, lean_object* v_00_u03b1_556_, lean_object* v_inst_557_, lean_object* v_inst_558_, lean_object* v_inst_559_, lean_object* v_S_560_){
_start:
{
lean_object* v_res_561_; 
v_res_561_ = lp_mathlib_Subsemiring_mulDistribMulAction(v_R_x27_555_, v_00_u03b1_556_, v_inst_557_, v_inst_558_, v_inst_559_, v_S_560_);
lean_dec_ref(v_inst_558_);
lean_dec_ref(v_inst_557_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulActionWithZeroSubtypeMem___redArg(lean_object* v_inst_562_){
_start:
{
lean_object* v___f_563_; 
v___f_563_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_563_, 0, v_inst_562_);
return v___f_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulActionWithZeroSubtypeMem(lean_object* v_R_x27_564_, lean_object* v_00_u03b1_565_, lean_object* v_inst_566_, lean_object* v_S_x27_567_, lean_object* v_inst_568_, lean_object* v_inst_569_, lean_object* v_s_570_, lean_object* v_inst_571_, lean_object* v_inst_572_){
_start:
{
lean_object* v___f_573_; 
v___f_573_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_573_, 0, v_inst_572_);
return v___f_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulActionWithZeroSubtypeMem___boxed(lean_object* v_R_x27_574_, lean_object* v_00_u03b1_575_, lean_object* v_inst_576_, lean_object* v_S_x27_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_s_580_, lean_object* v_inst_581_, lean_object* v_inst_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_mathlib_Subsemiring_instMulActionWithZeroSubtypeMem(v_R_x27_574_, v_00_u03b1_575_, v_inst_576_, v_S_x27_577_, v_inst_578_, v_inst_579_, v_s_580_, v_inst_581_, v_inst_582_);
lean_dec(v_inst_581_);
lean_dec(v_s_580_);
lean_dec_ref(v_inst_576_);
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulActionWithZero___redArg(lean_object* v_inst_584_){
_start:
{
lean_object* v___f_585_; 
v___f_585_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_585_, 0, v_inst_584_);
return v___f_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulActionWithZero(lean_object* v_R_x27_586_, lean_object* v_00_u03b1_587_, lean_object* v_inst_588_, lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_S_591_){
_start:
{
lean_object* v___f_592_; 
v___f_592_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_592_, 0, v_inst_590_);
return v___f_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mulActionWithZero___boxed(lean_object* v_R_x27_593_, lean_object* v_00_u03b1_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_S_598_){
_start:
{
lean_object* v_res_599_; 
v_res_599_ = lp_mathlib_Subsemiring_mulActionWithZero(v_R_x27_593_, v_00_u03b1_594_, v_inst_595_, v_inst_596_, v_inst_597_, v_S_598_);
lean_dec(v_inst_596_);
lean_dec_ref(v_inst_595_);
return v_res_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instModuleSubtypeMem___redArg(lean_object* v_inst_600_){
_start:
{
lean_object* v___f_601_; 
v___f_601_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_601_, 0, v_inst_600_);
return v___f_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instModuleSubtypeMem(lean_object* v_R_x27_602_, lean_object* v_00_u03b1_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_S_x27_607_, lean_object* v_inst_608_, lean_object* v_inst_609_, lean_object* v_s_610_){
_start:
{
lean_object* v___f_611_; 
v___f_611_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_611_, 0, v_inst_606_);
return v___f_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instModuleSubtypeMem___boxed(lean_object* v_R_x27_612_, lean_object* v_00_u03b1_613_, lean_object* v_inst_614_, lean_object* v_inst_615_, lean_object* v_inst_616_, lean_object* v_S_x27_617_, lean_object* v_inst_618_, lean_object* v_inst_619_, lean_object* v_s_620_){
_start:
{
lean_object* v_res_621_; 
v_res_621_ = lp_mathlib_Subsemiring_instModuleSubtypeMem(v_R_x27_612_, v_00_u03b1_613_, v_inst_614_, v_inst_615_, v_inst_616_, v_S_x27_617_, v_inst_618_, v_inst_619_, v_s_620_);
lean_dec(v_s_620_);
lean_dec_ref(v_inst_615_);
lean_dec_ref(v_inst_614_);
return v_res_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_module___redArg(lean_object* v_inst_622_){
_start:
{
lean_object* v___f_623_; 
v___f_623_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_623_, 0, v_inst_622_);
return v___f_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_module(lean_object* v_R_x27_624_, lean_object* v_00_u03b1_625_, lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_inst_628_, lean_object* v_S_629_){
_start:
{
lean_object* v___f_630_; 
v___f_630_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_630_, 0, v_inst_628_);
return v___f_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_module___boxed(lean_object* v_R_x27_631_, lean_object* v_00_u03b1_632_, lean_object* v_inst_633_, lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_S_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_mathlib_Subsemiring_module(v_R_x27_631_, v_00_u03b1_632_, v_inst_633_, v_inst_634_, v_inst_635_, v_S_636_);
lean_dec_ref(v_inst_634_);
lean_dec_ref(v_inst_633_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulSemiringActionSubtypeMem___redArg(lean_object* v_inst_638_){
_start:
{
lean_object* v___f_639_; 
v___f_639_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_639_, 0, v_inst_638_);
return v___f_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulSemiringActionSubtypeMem(lean_object* v_R_x27_640_, lean_object* v_00_u03b1_641_, lean_object* v_inst_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_S_645_){
_start:
{
lean_object* v___f_646_; 
v___f_646_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_646_, 0, v_inst_644_);
return v___f_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_instMulSemiringActionSubtypeMem___boxed(lean_object* v_R_x27_647_, lean_object* v_00_u03b1_648_, lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_S_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_mathlib_Subsemiring_instMulSemiringActionSubtypeMem(v_R_x27_647_, v_00_u03b1_648_, v_inst_649_, v_inst_650_, v_inst_651_, v_S_652_);
lean_dec_ref(v_inst_650_);
lean_dec_ref(v_inst_649_);
return v_res_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closureCommSemiringOfComm___redArg(lean_object* v_inst_654_){
_start:
{
lean_object* v___x_655_; 
v___x_655_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_654_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_closureCommSemiringOfComm(lean_object* v_R_x27_656_, lean_object* v_inst_657_, lean_object* v_s_658_, lean_object* v_hcomm_659_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lp_mathlib_Subsemiring_toSemiring___redArg(v_inst_657_);
return v___x_660_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Subobjects(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Subobjects(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Subobjects(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Action_Subobjects(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
