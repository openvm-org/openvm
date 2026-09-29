// Lean compiler output
// Module: Mathlib.Algebra.Ring.Subring.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Group.Subgroup.Basic public import Mathlib.Algebra.Ring.Subring.Defs public import Mathlib.Algebra.Ring.Subsemiring.Basic public import Mathlib.RingTheory.NonUnitalSubring.Basic public import Mathlib.Data.Set.Finite.Basic
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
lean_object* lp_mathlib_Equiv_Set_univ(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(lean_object*);
lean_object* lp_mathlib_SubringClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_RingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubringClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___redArg(lean_object*);
lean_object* lp_mathlib_SubringClass_toRing___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subring_instPartialOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(lean_object*);
lean_object* lp_mathlib_RingHom_restrict___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Subsemiring_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subring_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subring_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subring_instMin___closed__0 = (const lean_object*)&lp_mathlib_Subring_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Subring_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subring_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subring_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_Subring_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subring_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subring_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subring_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_Subring_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Subring_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Subring_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_Subring_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_center(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_center___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerToMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerToMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureCommRingOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureCommRingOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subring_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subring_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subring_gi___closed__0 = (const lean_object*)&lp_mathlib_Subring_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subring_prodEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subring_prodEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subring_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subring_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubringClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subring_inclusion___closed__0 = (const lean_object*)&lp_mathlib_Subring_inclusion___closed__0_value;
static const lean_closure_object lp_mathlib_Subring_inclusion___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_codRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Subring_inclusion___closed__0_value)} };
static const lean_object* lp_mathlib_Subring_inclusion___closed__1 = (const lean_object*)&lp_mathlib_Subring_inclusion___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subring_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_subringCongr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_subringCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instTop(lean_object* v_R_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instTop___boxed(lean_object* v_R_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Subring_instTop(v_R_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_7_);
v___x_9_ = lp_mathlib_Subsemiring_topEquiv(lean_box(0), v___x_8_);
lean_dec_ref(v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_topEquiv(lean_object* v_R_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Subring_topEquiv___redArg(v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___lam__0(lean_object* v___x_13_, lean_object* v___y_14_){
_start:
{
lean_object* v_toFun_15_; lean_object* v___x_16_; 
v_toFun_15_ = lean_ctor_get(v___x_13_, 0);
lean_inc(v_toFun_15_);
lean_dec_ref(v___x_13_);
v___x_16_ = lean_apply_1(v_toFun_15_, v___y_14_);
return v___x_16_;
}
}
static lean_object* _init_lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__0(void){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_Equiv_Set_univ(lean_box(0));
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__1(void){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_18_ = lean_obj_once(&lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__0, &lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__0_once, _init_lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__0);
v___x_19_ = lp_mathlib_Equiv_symm___redArg(v___x_18_);
return v___x_19_;
}
}
static lean_object* _init_lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__2(void){
_start:
{
lean_object* v___x_20_; lean_object* v___f_21_; 
v___x_20_ = lean_obj_once(&lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__1, &lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__1_once, _init_lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__1);
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___lam__0), 2, 1);
lean_closure_set(v___f_21_, 0, v___x_20_);
return v___f_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg(lean_object* v_inst_22_){
_start:
{
lean_object* v___f_23_; lean_object* v___x_24_; 
v___f_23_ = lean_obj_once(&lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__2, &lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__2_once, _init_lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___closed__2);
v___x_24_ = lp_mathlib_Finset_map___redArg(v___f_23_, v_inst_22_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop(lean_object* v_R_25_, lean_object* v_inst_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg(v_inst_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instFintypeSubtypeMemTop___boxed(lean_object* v_R_29_, lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Subring_instFintypeSubtypeMemTop(v_R_29_, v_inst_30_, v_inst_31_);
lean_dec_ref(v_inst_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_comap(lean_object* v_R_33_, lean_object* v_S_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_f_37_, lean_object* v_s_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lean_box(0);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_comap___boxed(lean_object* v_R_40_, lean_object* v_S_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_f_44_, lean_object* v_s_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Subring_comap(v_R_40_, v_S_41_, v_inst_42_, v_inst_43_, v_f_44_, v_s_45_);
lean_dec(v_f_44_);
lean_dec_ref(v_inst_43_);
lean_dec_ref(v_inst_42_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_map(lean_object* v_R_47_, lean_object* v_S_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_f_51_, lean_object* v_s_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lean_box(0);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_map___boxed(lean_object* v_R_54_, lean_object* v_S_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_f_58_, lean_object* v_s_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_Subring_map(v_R_54_, v_S_55_, v_inst_56_, v_inst_57_, v_f_58_, v_s_59_);
lean_dec(v_f_58_);
lean_dec_ref(v_inst_57_);
lean_dec_ref(v_inst_56_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_range(lean_object* v_R_61_, lean_object* v_S_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_f_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lean_box(0);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_range___boxed(lean_object* v_R_67_, lean_object* v_S_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_f_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_RingHom_range(v_R_67_, v_S_68_, v_inst_69_, v_inst_70_, v_f_71_);
lean_dec(v_f_71_);
lean_dec_ref(v_inst_70_);
lean_dec_ref(v_inst_69_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___redArg___lam__0(lean_object* v_f_73_, lean_object* v___y_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lean_apply_1(v_f_73_, v___y_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___redArg(lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_f_78_){
_start:
{
lean_object* v___f_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_fintypeRange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_79_, 0, v_f_78_);
v___x_80_ = lp_mathlib_PLift_fintype___redArg(v_inst_76_);
v___x_81_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_77_, v___f_79_, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange(lean_object* v_R_82_, lean_object* v_S_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_f_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_RingHom_fintypeRange___redArg(v_inst_86_, v_inst_87_, v_f_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeRange___boxed(lean_object* v_R_90_, lean_object* v_S_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_f_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_RingHom_fintypeRange(v_R_90_, v_S_91_, v_inst_92_, v_inst_93_, v_inst_94_, v_inst_95_, v_f_96_);
lean_dec_ref(v_inst_93_);
lean_dec_ref(v_inst_92_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instBot(lean_object* v_R_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lean_box(0);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instBot___boxed(lean_object* v_R_101_, lean_object* v_inst_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_Subring_instBot(v_R_101_, v_inst_102_);
lean_dec_ref(v_inst_102_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInhabited(lean_object* v_R_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lean_box(0);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInhabited___boxed(lean_object* v_R_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_Subring_instInhabited(v_R_107_, v_inst_108_);
lean_dec_ref(v_inst_108_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin___lam__0(lean_object* v_s_110_, lean_object* v_t_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lean_box(0);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin(lean_object* v_R_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___f_116_; 
v___f_116_ = ((lean_object*)(lp_mathlib_Subring_instMin___closed__0));
return v___f_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instMin___boxed(lean_object* v_R_117_, lean_object* v_inst_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_Subring_instMin(v_R_117_, v_inst_118_);
lean_dec_ref(v_inst_118_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet___lam__0(lean_object* v_s_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lean_box(0);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet(lean_object* v_R_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___f_125_; 
v___f_125_ = ((lean_object*)(lp_mathlib_Subring_instInfSet___closed__0));
return v___f_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instInfSet___boxed(lean_object* v_R_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_Subring_instInfSet(v_R_126_, v_inst_127_);
lean_dec_ref(v_inst_127_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___redArg___lam__0(lean_object* v_x1_129_, lean_object* v_x2_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lean_box(0);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___redArg(lean_object* v_inst_135_){
_start:
{
lean_object* v___x_136_; lean_object* v___f_137_; lean_object* v___x_138_; lean_object* v_toLattice_139_; lean_object* v_toSupSet_140_; lean_object* v_toInfSet_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_159_; 
v___x_136_ = lp_mathlib_Subring_instPartialOrder(lean_box(0), v_inst_135_);
v___f_137_ = ((lean_object*)(lp_mathlib_Subring_instInfSet___closed__0));
v___x_138_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_136_, v___f_137_);
v_toLattice_139_ = lean_ctor_get(v___x_138_, 0);
v_toSupSet_140_ = lean_ctor_get(v___x_138_, 1);
v_toInfSet_141_ = lean_ctor_get(v___x_138_, 2);
v_isSharedCheck_159_ = !lean_is_exclusive(v___x_138_);
if (v_isSharedCheck_159_ == 0)
{
lean_object* v_unused_160_; 
v_unused_160_ = lean_ctor_get(v___x_138_, 3);
lean_dec(v_unused_160_);
v___x_143_ = v___x_138_;
v_isShared_144_ = v_isSharedCheck_159_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_toInfSet_141_);
lean_inc(v_toSupSet_140_);
lean_inc(v_toLattice_139_);
lean_dec(v___x_138_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_159_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v_toSemilatticeSup_145_; lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_157_; 
v_toSemilatticeSup_145_ = lean_ctor_get(v_toLattice_139_, 0);
v_isSharedCheck_157_ = !lean_is_exclusive(v_toLattice_139_);
if (v_isSharedCheck_157_ == 0)
{
lean_object* v_unused_158_; 
v_unused_158_ = lean_ctor_get(v_toLattice_139_, 1);
lean_dec(v_unused_158_);
v___x_147_ = v_toLattice_139_;
v_isShared_148_ = v_isSharedCheck_157_;
goto v_resetjp_146_;
}
else
{
lean_inc(v_toSemilatticeSup_145_);
lean_dec(v_toLattice_139_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_157_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v___f_149_; lean_object* v___x_151_; 
v___f_149_ = ((lean_object*)(lp_mathlib_Subring_instCompleteLattice___redArg___closed__0));
if (v_isShared_148_ == 0)
{
lean_ctor_set(v___x_147_, 1, v___f_149_);
v___x_151_ = v___x_147_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v_toSemilatticeSup_145_);
lean_ctor_set(v_reuseFailAlloc_156_, 1, v___f_149_);
v___x_151_ = v_reuseFailAlloc_156_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
lean_object* v___x_152_; lean_object* v___x_154_; 
v___x_152_ = ((lean_object*)(lp_mathlib_Subring_instCompleteLattice___redArg___closed__1));
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 3, v___x_152_);
lean_ctor_set(v___x_143_, 0, v___x_151_);
v___x_154_ = v___x_143_;
goto v_reusejp_153_;
}
else
{
lean_object* v_reuseFailAlloc_155_; 
v_reuseFailAlloc_155_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_155_, 0, v___x_151_);
lean_ctor_set(v_reuseFailAlloc_155_, 1, v_toSupSet_140_);
lean_ctor_set(v_reuseFailAlloc_155_, 2, v_toInfSet_141_);
lean_ctor_set(v_reuseFailAlloc_155_, 3, v___x_152_);
v___x_154_ = v_reuseFailAlloc_155_;
goto v_reusejp_153_;
}
v_reusejp_153_:
{
return v___x_154_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___redArg___boxed(lean_object* v_inst_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_Subring_instCompleteLattice___redArg(v_inst_161_);
lean_dec_ref(v_inst_161_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice(lean_object* v_R_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_Subring_instCompleteLattice___redArg(v_inst_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCompleteLattice___boxed(lean_object* v_R_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Subring_instCompleteLattice(v_R_166_, v_inst_167_);
lean_dec_ref(v_inst_167_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_center(lean_object* v_R_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_box(0);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_center___boxed(lean_object* v_R_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Subring_center(v_R_172_, v_inst_173_);
lean_dec_ref(v_inst_173_);
return v_res_174_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter___redArg___lam__0(lean_object* v_toMul_175_, lean_object* v_x_176_, lean_object* v_inst_177_, lean_object* v_a_178_){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; uint8_t v___x_182_; 
lean_inc(v_toMul_175_);
lean_inc(v_x_176_);
lean_inc(v_a_178_);
v___x_179_ = lean_apply_2(v_toMul_175_, v_a_178_, v_x_176_);
v___x_180_ = lean_apply_2(v_toMul_175_, v_x_176_, v_a_178_);
v___x_181_ = lean_apply_2(v_inst_177_, v___x_179_, v___x_180_);
v___x_182_ = lean_unbox(v___x_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___redArg___lam__0___boxed(lean_object* v_toMul_183_, lean_object* v_x_184_, lean_object* v_inst_185_, lean_object* v_a_186_){
_start:
{
uint8_t v_res_187_; lean_object* v_r_188_; 
v_res_187_ = lp_mathlib_Subring_decidableMemCenter___redArg___lam__0(v_toMul_183_, v_x_184_, v_inst_185_, v_a_186_);
v_r_188_ = lean_box(v_res_187_);
return v_r_188_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter___redArg(lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_x_192_){
_start:
{
lean_object* v_toSemiring_193_; lean_object* v___x_194_; lean_object* v_toMul_195_; lean_object* v___f_196_; uint8_t v___x_197_; 
v_toSemiring_193_ = lean_ctor_get(v_inst_189_, 0);
lean_inc_ref(v_toSemiring_193_);
lean_dec_ref(v_inst_189_);
v___x_194_ = lp_mathlib_instDistribOfSemiring___redArg(v_toSemiring_193_);
v_toMul_195_ = lean_ctor_get(v___x_194_, 0);
lean_inc(v_toMul_195_);
lean_dec_ref(v___x_194_);
v___f_196_ = lean_alloc_closure((void*)(lp_mathlib_Subring_decidableMemCenter___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_196_, 0, v_toMul_195_);
lean_closure_set(v___f_196_, 1, v_x_192_);
lean_closure_set(v___f_196_, 2, v_inst_190_);
v___x_197_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_196_, v_inst_191_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___redArg___boxed(lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_x_201_){
_start:
{
uint8_t v_res_202_; lean_object* v_r_203_; 
v_res_202_ = lp_mathlib_Subring_decidableMemCenter___redArg(v_inst_198_, v_inst_199_, v_inst_200_, v_x_201_);
v_r_203_ = lean_box(v_res_202_);
return v_r_203_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Subring_decidableMemCenter(lean_object* v_R_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_x_208_){
_start:
{
uint8_t v___x_209_; 
v___x_209_ = lp_mathlib_Subring_decidableMemCenter___redArg(v_inst_205_, v_inst_206_, v_inst_207_, v_x_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_decidableMemCenter___boxed(lean_object* v_R_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_x_214_){
_start:
{
uint8_t v_res_215_; lean_object* v_r_216_; 
v_res_215_ = lp_mathlib_Subring_decidableMemCenter(v_R_210_, v_inst_211_, v_inst_212_, v_inst_213_, v_x_214_);
v_r_216_ = lean_box(v_res_215_);
return v_r_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter___redArg(lean_object* v_inst_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instCommRingSubtypeMemCenter(lean_object* v_R_219_, lean_object* v_inst_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr___redArg(lean_object* v_e_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(v_e_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr(lean_object* v_R_224_, lean_object* v_S_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_e_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(v_e_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerCongr___boxed(lean_object* v_R_230_, lean_object* v_S_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_e_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_Subring_centerCongr(v_R_230_, v_S_231_, v_inst_232_, v_inst_233_, v_e_234_);
lean_dec_ref(v_inst_233_);
lean_dec_ref(v_inst_232_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerToMulOpposite___redArg(lean_object* v_inst_236_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v_toNonUnitalNonAssocRing_237_ = lean_ctor_get(v_inst_236_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_237_);
lean_dec_ref(v_inst_236_);
v___x_238_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocRing_237_);
v___x_239_ = lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(v___x_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centerToMulOpposite(lean_object* v_R_240_, lean_object* v_inst_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_Subring_centerToMulOpposite___redArg(v_inst_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__0(lean_object* v_toInv_243_, lean_object* v_a_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lean_apply_1(v_toInv_243_, v_a_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__1(lean_object* v_toDiv_246_, lean_object* v_a_247_, lean_object* v_b_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lean_apply_2(v_toDiv_246_, v_a_247_, v_b_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__2(lean_object* v_toRing_250_, lean_object* v_toNatCast_251_, lean_object* v_toIntCast_252_, lean_object* v___f_253_, lean_object* v_x_254_, lean_object* v___y_255_){
_start:
{
lean_object* v_toSemiring_256_; lean_object* v_toMonoid_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v_toMul_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v_toSemiring_256_ = lean_ctor_get(v_toRing_250_, 0);
v_toMonoid_257_ = lean_ctor_get(v_toSemiring_256_, 1);
v___x_258_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_257_);
v___x_259_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_258_);
v_toMul_260_ = lean_ctor_get(v___x_259_, 1);
lean_inc(v_toMul_260_);
lean_dec_ref(v___x_259_);
v___x_261_ = lp_mathlib_Rat_castRec___redArg(v_toNatCast_251_, v_toIntCast_252_, v___f_253_, v_x_254_);
v___x_262_ = lean_apply_2(v_toMul_260_, v___x_261_, v___y_255_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__2___boxed(lean_object* v_toRing_263_, lean_object* v_toNatCast_264_, lean_object* v_toIntCast_265_, lean_object* v___f_266_, lean_object* v_x_267_, lean_object* v___y_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_Subring_instField___redArg___lam__2(v_toRing_263_, v_toNatCast_264_, v_toIntCast_265_, v___f_266_, v_x_267_, v___y_268_);
lean_dec_ref(v_toRing_263_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__3(lean_object* v_toRing_270_, lean_object* v_toNatCast_271_, lean_object* v___f_272_, lean_object* v_x_273_, lean_object* v___y_274_){
_start:
{
lean_object* v_toSemiring_275_; lean_object* v_toMonoid_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v_toMul_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v_toSemiring_275_ = lean_ctor_get(v_toRing_270_, 0);
v_toMonoid_276_ = lean_ctor_get(v_toSemiring_275_, 1);
v___x_277_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_276_);
v___x_278_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_277_);
v_toMul_279_ = lean_ctor_get(v___x_278_, 1);
lean_inc(v_toMul_279_);
lean_dec_ref(v___x_278_);
v___x_280_ = lp_mathlib_NNRat_castRec___redArg(v_toNatCast_271_, v___f_272_, v_x_273_);
v___x_281_ = lean_apply_2(v_toMul_279_, v___x_280_, v___y_274_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg___lam__3___boxed(lean_object* v_toRing_282_, lean_object* v_toNatCast_283_, lean_object* v___f_284_, lean_object* v_x_285_, lean_object* v___y_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_Subring_instField___redArg___lam__3(v_toRing_282_, v_toNatCast_283_, v___f_284_, v_x_285_, v___y_286_);
lean_dec_ref(v_toRing_282_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField___redArg(lean_object* v_inst_288_){
_start:
{
lean_object* v_toRing_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v_toInv_295_; lean_object* v___x_296_; lean_object* v___x_298_; uint8_t v_isShared_299_; uint8_t v_isSharedCheck_318_; 
v_toRing_289_ = lean_ctor_get(v_inst_288_, 0);
lean_inc_ref_n(v_toRing_289_, 2);
v___x_290_ = lp_mathlib_SubringClass_toRing___redArg(v_toRing_289_);
v___x_291_ = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(v_inst_288_);
v___x_292_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_291_);
lean_dec_ref(v___x_291_);
v___x_293_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_292_);
v___x_294_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_293_);
lean_dec_ref(v___x_293_);
v_toInv_295_ = lean_ctor_get(v___x_294_, 1);
lean_inc(v_toInv_295_);
lean_dec_ref(v___x_294_);
v___x_296_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_inst_288_);
v_isSharedCheck_318_ = !lean_is_exclusive(v_inst_288_);
if (v_isSharedCheck_318_ == 0)
{
lean_object* v_unused_319_; lean_object* v_unused_320_; lean_object* v_unused_321_; lean_object* v_unused_322_; lean_object* v_unused_323_; lean_object* v_unused_324_; lean_object* v_unused_325_; lean_object* v_unused_326_; 
v_unused_319_ = lean_ctor_get(v_inst_288_, 7);
lean_dec(v_unused_319_);
v_unused_320_ = lean_ctor_get(v_inst_288_, 6);
lean_dec(v_unused_320_);
v_unused_321_ = lean_ctor_get(v_inst_288_, 5);
lean_dec(v_unused_321_);
v_unused_322_ = lean_ctor_get(v_inst_288_, 4);
lean_dec(v_unused_322_);
v_unused_323_ = lean_ctor_get(v_inst_288_, 3);
lean_dec(v_unused_323_);
v_unused_324_ = lean_ctor_get(v_inst_288_, 2);
lean_dec(v_unused_324_);
v_unused_325_ = lean_ctor_get(v_inst_288_, 1);
lean_dec(v_unused_325_);
v_unused_326_ = lean_ctor_get(v_inst_288_, 0);
lean_dec(v_unused_326_);
v___x_298_ = v_inst_288_;
v_isShared_299_ = v_isSharedCheck_318_;
goto v_resetjp_297_;
}
else
{
lean_dec(v_inst_288_);
v___x_298_ = lean_box(0);
v_isShared_299_ = v_isSharedCheck_318_;
goto v_resetjp_297_;
}
v_resetjp_297_:
{
lean_object* v_toSemiring_300_; lean_object* v_toMonoid_301_; lean_object* v_toDiv_302_; lean_object* v_toIntCast_303_; lean_object* v_toNatCast_304_; lean_object* v_toOne_305_; lean_object* v_toMul_306_; lean_object* v___f_307_; lean_object* v___f_308_; lean_object* v___f_309_; lean_object* v___f_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_316_; 
v_toSemiring_300_ = lean_ctor_get(v___x_290_, 0);
lean_inc_ref(v_toSemiring_300_);
v_toMonoid_301_ = lean_ctor_get(v_toSemiring_300_, 1);
lean_inc_ref(v_toMonoid_301_);
v_toDiv_302_ = lean_ctor_get(v___x_296_, 2);
lean_inc(v_toDiv_302_);
lean_dec_ref(v___x_296_);
v_toIntCast_303_ = lean_ctor_get(v___x_290_, 4);
lean_inc_n(v_toIntCast_303_, 2);
v_toNatCast_304_ = lean_ctor_get(v_toSemiring_300_, 2);
lean_inc_n(v_toNatCast_304_, 4);
lean_dec_ref(v_toSemiring_300_);
v_toOne_305_ = lean_ctor_get(v_toMonoid_301_, 0);
lean_inc_n(v_toOne_305_, 2);
v_toMul_306_ = lean_ctor_get(v_toMonoid_301_, 1);
lean_inc_n(v_toMul_306_, 2);
lean_dec_ref(v_toMonoid_301_);
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_Subring_instField___redArg___lam__0), 2, 1);
lean_closure_set(v___f_307_, 0, v_toInv_295_);
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_Subring_instField___redArg___lam__1), 3, 1);
lean_closure_set(v___f_308_, 0, v_toDiv_302_);
lean_inc_ref_n(v___f_308_, 4);
lean_inc_ref(v_toRing_289_);
v___f_309_ = lean_alloc_closure((void*)(lp_mathlib_Subring_instField___redArg___lam__2___boxed), 6, 4);
lean_closure_set(v___f_309_, 0, v_toRing_289_);
lean_closure_set(v___f_309_, 1, v_toNatCast_304_);
lean_closure_set(v___f_309_, 2, v_toIntCast_303_);
lean_closure_set(v___f_309_, 3, v___f_308_);
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_Subring_instField___redArg___lam__3___boxed), 5, 3);
lean_closure_set(v___f_310_, 0, v_toRing_289_);
lean_closure_set(v___f_310_, 1, v_toNatCast_304_);
lean_closure_set(v___f_310_, 2, v___f_308_);
v___x_311_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_311_, 0, lean_box(0));
lean_closure_set(v___x_311_, 1, v_toOne_305_);
lean_closure_set(v___x_311_, 2, v_toMul_306_);
lean_inc_ref(v___f_307_);
v___x_312_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_312_, 0, lean_box(0));
lean_closure_set(v___x_312_, 1, v_toOne_305_);
lean_closure_set(v___x_312_, 2, v_toMul_306_);
lean_closure_set(v___x_312_, 3, v___f_307_);
lean_closure_set(v___x_312_, 4, v___x_311_);
v___x_313_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(v___x_313_, 0, lean_box(0));
lean_closure_set(v___x_313_, 1, v_toNatCast_304_);
lean_closure_set(v___x_313_, 2, v___f_308_);
v___x_314_ = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(v___x_314_, 0, lean_box(0));
lean_closure_set(v___x_314_, 1, v_toNatCast_304_);
lean_closure_set(v___x_314_, 2, v_toIntCast_303_);
lean_closure_set(v___x_314_, 3, v___f_308_);
if (v_isShared_299_ == 0)
{
lean_ctor_set(v___x_298_, 7, v___f_309_);
lean_ctor_set(v___x_298_, 6, v___f_310_);
lean_ctor_set(v___x_298_, 5, v___x_314_);
lean_ctor_set(v___x_298_, 4, v___x_313_);
lean_ctor_set(v___x_298_, 3, v___x_312_);
lean_ctor_set(v___x_298_, 2, v___f_308_);
lean_ctor_set(v___x_298_, 1, v___f_307_);
lean_ctor_set(v___x_298_, 0, v___x_290_);
v___x_316_ = v___x_298_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_317_; 
v_reuseFailAlloc_317_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_317_, 0, v___x_290_);
lean_ctor_set(v_reuseFailAlloc_317_, 1, v___f_307_);
lean_ctor_set(v_reuseFailAlloc_317_, 2, v___f_308_);
lean_ctor_set(v_reuseFailAlloc_317_, 3, v___x_312_);
lean_ctor_set(v_reuseFailAlloc_317_, 4, v___x_313_);
lean_ctor_set(v_reuseFailAlloc_317_, 5, v___x_314_);
lean_ctor_set(v_reuseFailAlloc_317_, 6, v___f_310_);
lean_ctor_set(v_reuseFailAlloc_317_, 7, v___f_309_);
v___x_316_ = v_reuseFailAlloc_317_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
return v___x_316_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instField(lean_object* v_K_327_, lean_object* v_inst_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lp_mathlib_Subring_instField___redArg(v_inst_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centralizer(lean_object* v_R_330_, lean_object* v_inst_331_, lean_object* v_s_332_){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = lean_box(0);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_centralizer___boxed(lean_object* v_R_334_, lean_object* v_inst_335_, lean_object* v_s_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_Subring_centralizer(v_R_334_, v_inst_335_, v_s_336_);
lean_dec_ref(v_inst_335_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closure(lean_object* v_R_338_, lean_object* v_inst_339_, lean_object* v_s_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lean_box(0);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closure___boxed(lean_object* v_R_342_, lean_object* v_inst_343_, lean_object* v_s_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Subring_closure(v_R_342_, v_inst_343_, v_s_344_);
lean_dec_ref(v_inst_343_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureCommRingOfComm___redArg(lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_closureCommRingOfComm(lean_object* v_R_348_, lean_object* v_inst_349_, lean_object* v_s_350_, lean_object* v_hcomm_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_349_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi___lam__0(lean_object* v_s_353_, lean_object* v_x_354_){
_start:
{
lean_object* v___x_355_; 
v___x_355_ = lean_box(0);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi(lean_object* v_R_357_, lean_object* v_inst_358_){
_start:
{
lean_object* v___f_359_; 
v___f_359_ = ((lean_object*)(lp_mathlib_Subring_gi___closed__0));
return v___f_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_gi___boxed(lean_object* v_R_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_Subring_gi(v_R_360_, v_inst_361_);
lean_dec_ref(v_inst_361_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_prod(lean_object* v_R_363_, lean_object* v_S_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_s_367_, lean_object* v_t_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lean_box(0);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_prod___boxed(lean_object* v_R_370_, lean_object* v_S_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_s_374_, lean_object* v_t_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib_Subring_prod(v_R_370_, v_S_371_, v_inst_372_, v_inst_373_, v_s_374_, v_t_375_);
lean_dec_ref(v_inst_373_);
lean_dec_ref(v_inst_372_);
return v_res_376_;
}
}
static lean_object* _init_lp_mathlib_Subring_prodEquiv___closed__0(void){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_prodEquiv(lean_object* v_R_378_, lean_object* v_S_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_s_382_, lean_object* v_t_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lean_obj_once(&lp_mathlib_Subring_prodEquiv___closed__0, &lp_mathlib_Subring_prodEquiv___closed__0_once, _init_lp_mathlib_Subring_prodEquiv___closed__0);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_prodEquiv___boxed(lean_object* v_R_385_, lean_object* v_S_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_s_389_, lean_object* v_t_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_Subring_prodEquiv(v_R_385_, v_S_386_, v_inst_387_, v_inst_388_, v_s_389_, v_t_390_);
lean_dec_ref(v_inst_388_);
lean_dec_ref(v_inst_387_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict___redArg(lean_object* v_f_392_){
_start:
{
lean_object* v___f_393_; 
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_393_, 0, v_f_392_);
return v___f_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict(lean_object* v_R_394_, lean_object* v_S_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_f_398_){
_start:
{
lean_object* v___f_399_; 
v___f_399_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_399_, 0, v_f_398_);
return v___f_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrict___boxed(lean_object* v_R_400_, lean_object* v_S_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_f_404_){
_start:
{
lean_object* v_res_405_; 
v_res_405_ = lp_mathlib_RingHom_rangeRestrict(v_R_400_, v_S_401_, v_inst_402_, v_inst_403_, v_f_404_);
lean_dec_ref(v_inst_403_);
lean_dec_ref(v_inst_402_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocus(lean_object* v_R_406_, lean_object* v_inst_407_, lean_object* v_S_408_, lean_object* v_inst_409_, lean_object* v_f_410_, lean_object* v_g_411_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lean_box(0);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocus___boxed(lean_object* v_R_413_, lean_object* v_inst_414_, lean_object* v_S_415_, lean_object* v_inst_416_, lean_object* v_f_417_, lean_object* v_g_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_mathlib_RingHom_eqLocus(v_R_413_, v_inst_414_, v_S_415_, v_inst_416_, v_f_417_, v_g_418_);
lean_dec(v_g_418_);
lean_dec(v_f_417_);
lean_dec_ref(v_inst_416_);
lean_dec_ref(v_inst_414_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_inclusion(lean_object* v_R_423_, lean_object* v_inst_424_, lean_object* v_S_425_, lean_object* v_T_426_, lean_object* v_h_427_){
_start:
{
lean_object* v___f_428_; 
v___f_428_ = ((lean_object*)(lp_mathlib_Subring_inclusion___closed__1));
return v___f_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_inclusion___boxed(lean_object* v_R_429_, lean_object* v_inst_430_, lean_object* v_S_431_, lean_object* v_T_432_, lean_object* v_h_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_Subring_inclusion(v_R_429_, v_inst_430_, v_S_431_, v_T_432_, v_h_433_);
lean_dec_ref(v_inst_430_);
return v_res_434_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_subringCongr___closed__0(void){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringCongr(lean_object* v_R_436_, lean_object* v_inst_437_, lean_object* v_s_438_, lean_object* v_t_439_, lean_object* v_h_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lean_obj_once(&lp_mathlib_RingEquiv_subringCongr___closed__0, &lp_mathlib_RingEquiv_subringCongr___closed__0_once, _init_lp_mathlib_RingEquiv_subringCongr___closed__0);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringCongr___boxed(lean_object* v_R_442_, lean_object* v_inst_443_, lean_object* v_s_444_, lean_object* v_t_445_, lean_object* v_h_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib_RingEquiv_subringCongr(v_R_442_, v_inst_443_, v_s_444_, v_t_445_, v_h_446_);
lean_dec_ref(v_inst_443_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__0(lean_object* v_f_448_, lean_object* v_x_449_){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lp_mathlib_RingHom_codRestrict___redArg___lam__0(v_f_448_, v_x_449_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1(lean_object* v_g_451_, lean_object* v_x_452_){
_start:
{
lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_453_ = lp_mathlib_SubringClass_subtype___lam__0(v_x_452_);
v___x_454_ = lean_apply_1(v_g_451_, v___x_453_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1___boxed(lean_object* v_g_455_, lean_object* v_x_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1(v_g_455_, v_x_456_);
lean_dec(v_x_456_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___redArg(lean_object* v_g_458_, lean_object* v_f_459_){
_start:
{
lean_object* v___f_460_; lean_object* v___f_461_; lean_object* v___x_462_; 
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_460_, 0, v_f_459_);
v___f_461_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofLeftInverse___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_461_, 0, v_g_458_);
v___x_462_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_462_, 0, v___f_460_);
lean_ctor_set(v___x_462_, 1, v___f_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse(lean_object* v_R_463_, lean_object* v_S_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_g_467_, lean_object* v_f_468_, lean_object* v_h_469_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_RingEquiv_ofLeftInverse___redArg(v_g_467_, v_f_468_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse___boxed(lean_object* v_R_471_, lean_object* v_S_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_g_475_, lean_object* v_f_476_, lean_object* v_h_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib_RingEquiv_ofLeftInverse(v_R_471_, v_S_472_, v_inst_473_, v_inst_474_, v_g_475_, v_f_476_, v_h_477_);
lean_dec_ref(v_inst_474_);
lean_dec_ref(v_inst_473_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___redArg(lean_object* v_e_479_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap(lean_object* v_R_481_, lean_object* v_S_482_, lean_object* v_inst_483_, lean_object* v_inst_484_, lean_object* v_s_485_, lean_object* v_e_486_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subringMap___boxed(lean_object* v_R_488_, lean_object* v_S_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_s_492_, lean_object* v_e_493_){
_start:
{
lean_object* v_res_494_; 
v_res_494_ = lp_mathlib_RingEquiv_subringMap(v_R_488_, v_S_489_, v_inst_490_, v_inst_491_, v_s_492_, v_e_493_);
lean_dec_ref(v_inst_491_);
lean_dec_ref(v_inst_490_);
return v_res_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg___lam__0(lean_object* v_e_495_, lean_object* v___y_496_){
_start:
{
lean_object* v_toFun_497_; lean_object* v___x_498_; 
v_toFun_497_ = lean_ctor_get(v_e_495_, 0);
lean_inc(v_toFun_497_);
lean_dec_ref(v_e_495_);
v___x_498_ = lean_apply_1(v_toFun_497_, v___y_496_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg___lam__2(lean_object* v___f_499_, lean_object* v___y_500_){
_start:
{
lean_object* v___x_103__overap_501_; lean_object* v___x_502_; 
v___x_103__overap_501_ = lp_mathlib_RingHom_restrict___redArg(v___f_499_);
v___x_502_ = lean_apply_1(v___x_103__overap_501_, v___y_500_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___redArg(lean_object* v_e_503_){
_start:
{
lean_object* v___f_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___f_507_; lean_object* v___f_508_; lean_object* v___x_509_; 
lean_inc_ref(v_e_503_);
v___f_504_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_restrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_504_, 0, v_e_503_);
v___x_505_ = lp_mathlib_RingHom_restrict___redArg(v___f_504_);
v___x_506_ = lp_mathlib_Equiv_symm___redArg(v_e_503_);
v___f_507_ = lean_alloc_closure((void*)(lp_mathlib_Subring_instFintypeSubtypeMemTop___redArg___lam__0), 2, 1);
lean_closure_set(v___f_507_, 0, v___x_506_);
v___f_508_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_restrict___redArg___lam__2), 2, 1);
lean_closure_set(v___f_508_, 0, v___f_507_);
v___x_509_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_509_, 0, v___x_505_);
lean_ctor_set(v___x_509_, 1, v___f_508_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict(lean_object* v_R_510_, lean_object* v_S_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_00_u03c3R_514_, lean_object* v_00_u03c3S_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_inst_518_, lean_object* v_inst_519_, lean_object* v_e_520_, lean_object* v_s_x27_521_, lean_object* v_s_522_, lean_object* v_h_523_){
_start:
{
lean_object* v___x_524_; 
v___x_524_ = lp_mathlib_RingEquiv_restrict___redArg(v_e_520_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_restrict___boxed(lean_object* v_R_525_, lean_object* v_S_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_00_u03c3R_529_, lean_object* v_00_u03c3S_530_, lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_e_535_, lean_object* v_s_x27_536_, lean_object* v_s_537_, lean_object* v_h_538_){
_start:
{
lean_object* v_res_539_; 
v_res_539_ = lp_mathlib_RingEquiv_restrict(v_R_525_, v_S_526_, v_inst_527_, v_inst_528_, v_00_u03c3R_529_, v_00_u03c3S_530_, v_inst_531_, v_inst_532_, v_inst_533_, v_inst_534_, v_e_535_, v_s_x27_536_, v_s_537_, v_h_538_);
lean_dec(v_s_537_);
lean_dec(v_s_x27_536_);
lean_dec_ref(v_inst_528_);
lean_dec_ref(v_inst_527_);
return v_res_539_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
