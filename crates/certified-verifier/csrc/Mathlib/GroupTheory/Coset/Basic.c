// Lean compiler output
// Module: Mathlib.GroupTheory.Coset.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Pointwise.Set.Basic public import Mathlib.Algebra.Group.Subgroup.Basic public import Mathlib.Data.Setoid.Basic public import Mathlib.GroupTheory.Coset.Defs
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
lean_object* lp_mathlib_SubgroupClass_inclusion___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Quotient_map_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l_Quotient_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_QuotientGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubgroupClass_inclusion___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__0_value;
static const lean_closure_object lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Quotient_map_x27, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__1 = (const lean_object*)&lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfEmbeddingOfLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfEmbeddingOfLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientMapOfLE___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientMapOfLE___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientMapOfLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientMapOfLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientMapOfLE___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientMapOfLE___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientMapOfLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientMapOfLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientiInfAddSubgroupOfEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientiInfAddSubgroupOfEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfEmbedding(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientiInfEmbedding(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientiInfEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquivKer___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquivKer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquivKer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquivKer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquivKer___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquivKer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquivKer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquivKer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__0 = (const lean_object*)&lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Quotient_lift, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__1 = (const lean_object*)&lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientEquivSelf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientEquivSelf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientEquivSelf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientEquivSelf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg___lam__0(lean_object* v_toMul_1_, lean_object* v_g_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toMul_1_, v_g_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg___lam__1(lean_object* v_toInv_5_, lean_object* v_g_6_, lean_object* v_toMul_7_, lean_object* v_x_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_apply_1(v_toInv_5_, v_g_6_);
v___x_10_ = lean_apply_2(v_toMul_7_, v___x_9_, v_x_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg(lean_object* v_inst_11_, lean_object* v_g_12_){
_start:
{
lean_object* v_toMonoid_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v_toMul_16_; lean_object* v___x_17_; lean_object* v_toInv_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_27_; 
v_toMonoid_13_ = lean_ctor_get(v_inst_11_, 0);
v___x_14_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_13_);
v___x_15_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_14_);
v_toMul_16_ = lean_ctor_get(v___x_15_, 1);
lean_inc(v_toMul_16_);
lean_dec_ref(v___x_15_);
v___x_17_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_11_);
v_toInv_18_ = lean_ctor_get(v___x_17_, 1);
v_isSharedCheck_27_ = !lean_is_exclusive(v___x_17_);
if (v_isSharedCheck_27_ == 0)
{
lean_object* v_unused_28_; 
v_unused_28_ = lean_ctor_get(v___x_17_, 0);
lean_dec(v_unused_28_);
v___x_20_ = v___x_17_;
v_isShared_21_ = v_isSharedCheck_27_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_toInv_18_);
lean_dec(v___x_17_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_27_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___f_22_; lean_object* v___f_23_; lean_object* v___x_25_; 
lean_inc(v_g_12_);
lean_inc(v_toMul_16_);
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_22_, 0, v_toMul_16_);
lean_closure_set(v___f_22_, 1, v_g_12_);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg___lam__1), 4, 3);
lean_closure_set(v___f_23_, 0, v_toInv_18_);
lean_closure_set(v___f_23_, 1, v_g_12_);
lean_closure_set(v___f_23_, 2, v_toMul_16_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 1, v___f_22_);
lean_ctor_set(v___x_20_, 0, v___f_23_);
v___x_25_ = v___x_20_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___f_23_);
lean_ctor_set(v_reuseFailAlloc_26_, 1, v___f_22_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg___boxed(lean_object* v_inst_29_, lean_object* v_g_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg(v_inst_29_, v_g_30_);
lean_dec_ref(v_inst_29_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup(lean_object* v_00_u03b1_32_, lean_object* v_inst_33_, lean_object* v_s_34_, lean_object* v_g_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg(v_inst_33_, v_g_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_leftCosetEquivSubgroup___boxed(lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_s_39_, lean_object* v_g_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Subgroup_leftCosetEquivSubgroup(v_00_u03b1_37_, v_inst_38_, v_s_39_, v_g_40_);
lean_dec_ref(v_inst_38_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg___lam__0(lean_object* v_toAdd_42_, lean_object* v_g_43_, lean_object* v_x_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lean_apply_2(v_toAdd_42_, v_g_43_, v_x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg___lam__1(lean_object* v_toNeg_46_, lean_object* v_g_47_, lean_object* v_toAdd_48_, lean_object* v_x_49_){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = lean_apply_1(v_toNeg_46_, v_g_47_);
v___x_51_ = lean_apply_2(v_toAdd_48_, v___x_50_, v_x_49_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg(lean_object* v_inst_52_, lean_object* v_g_53_){
_start:
{
lean_object* v_toAddMonoid_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v_toAdd_57_; lean_object* v___x_58_; lean_object* v_toNeg_59_; lean_object* v___x_61_; uint8_t v_isShared_62_; uint8_t v_isSharedCheck_68_; 
v_toAddMonoid_54_ = lean_ctor_get(v_inst_52_, 0);
v___x_55_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_54_);
v___x_56_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_55_);
v_toAdd_57_ = lean_ctor_get(v___x_56_, 1);
lean_inc(v_toAdd_57_);
lean_dec_ref(v___x_56_);
v___x_58_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_52_);
v_toNeg_59_ = lean_ctor_get(v___x_58_, 1);
v_isSharedCheck_68_ = !lean_is_exclusive(v___x_58_);
if (v_isSharedCheck_68_ == 0)
{
lean_object* v_unused_69_; 
v_unused_69_ = lean_ctor_get(v___x_58_, 0);
lean_dec(v_unused_69_);
v___x_61_ = v___x_58_;
v_isShared_62_ = v_isSharedCheck_68_;
goto v_resetjp_60_;
}
else
{
lean_inc(v_toNeg_59_);
lean_dec(v___x_58_);
v___x_61_ = lean_box(0);
v_isShared_62_ = v_isSharedCheck_68_;
goto v_resetjp_60_;
}
v_resetjp_60_:
{
lean_object* v___f_63_; lean_object* v___f_64_; lean_object* v___x_66_; 
lean_inc(v_g_53_);
lean_inc(v_toAdd_57_);
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_63_, 0, v_toAdd_57_);
lean_closure_set(v___f_63_, 1, v_g_53_);
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg___lam__1), 4, 3);
lean_closure_set(v___f_64_, 0, v_toNeg_59_);
lean_closure_set(v___f_64_, 1, v_g_53_);
lean_closure_set(v___f_64_, 2, v_toAdd_57_);
if (v_isShared_62_ == 0)
{
lean_ctor_set(v___x_61_, 1, v___f_63_);
lean_ctor_set(v___x_61_, 0, v___f_64_);
v___x_66_ = v___x_61_;
goto v_reusejp_65_;
}
else
{
lean_object* v_reuseFailAlloc_67_; 
v_reuseFailAlloc_67_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_67_, 0, v___f_64_);
lean_ctor_set(v_reuseFailAlloc_67_, 1, v___f_63_);
v___x_66_ = v_reuseFailAlloc_67_;
goto v_reusejp_65_;
}
v_reusejp_65_:
{
return v___x_66_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg___boxed(lean_object* v_inst_70_, lean_object* v_g_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg(v_inst_70_, v_g_71_);
lean_dec_ref(v_inst_70_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup(lean_object* v_00_u03b1_73_, lean_object* v_inst_74_, lean_object* v_s_75_, lean_object* v_g_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg(v_inst_74_, v_g_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___boxed(lean_object* v_00_u03b1_78_, lean_object* v_inst_79_, lean_object* v_s_80_, lean_object* v_g_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup(v_00_u03b1_78_, v_inst_79_, v_s_80_, v_g_81_);
lean_dec_ref(v_inst_79_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg___lam__0(lean_object* v_toMul_83_, lean_object* v_g_84_, lean_object* v_x_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_apply_2(v_toMul_83_, v_x_85_, v_g_84_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg___lam__1(lean_object* v_toInv_87_, lean_object* v_g_88_, lean_object* v_toMul_89_, lean_object* v_x_90_){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_91_ = lean_apply_1(v_toInv_87_, v_g_88_);
v___x_92_ = lean_apply_2(v_toMul_89_, v_x_90_, v___x_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg(lean_object* v_inst_93_, lean_object* v_g_94_){
_start:
{
lean_object* v_toMonoid_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v_toMul_98_; lean_object* v___x_99_; lean_object* v_toInv_100_; lean_object* v___x_102_; uint8_t v_isShared_103_; uint8_t v_isSharedCheck_109_; 
v_toMonoid_95_ = lean_ctor_get(v_inst_93_, 0);
v___x_96_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_95_);
v___x_97_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_96_);
v_toMul_98_ = lean_ctor_get(v___x_97_, 1);
lean_inc(v_toMul_98_);
lean_dec_ref(v___x_97_);
v___x_99_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_93_);
v_toInv_100_ = lean_ctor_get(v___x_99_, 1);
v_isSharedCheck_109_ = !lean_is_exclusive(v___x_99_);
if (v_isSharedCheck_109_ == 0)
{
lean_object* v_unused_110_; 
v_unused_110_ = lean_ctor_get(v___x_99_, 0);
lean_dec(v_unused_110_);
v___x_102_ = v___x_99_;
v_isShared_103_ = v_isSharedCheck_109_;
goto v_resetjp_101_;
}
else
{
lean_inc(v_toInv_100_);
lean_dec(v___x_99_);
v___x_102_ = lean_box(0);
v_isShared_103_ = v_isSharedCheck_109_;
goto v_resetjp_101_;
}
v_resetjp_101_:
{
lean_object* v___f_104_; lean_object* v___f_105_; lean_object* v___x_107_; 
lean_inc(v_g_94_);
lean_inc(v_toMul_98_);
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_104_, 0, v_toMul_98_);
lean_closure_set(v___f_104_, 1, v_g_94_);
v___f_105_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg___lam__1), 4, 3);
lean_closure_set(v___f_105_, 0, v_toInv_100_);
lean_closure_set(v___f_105_, 1, v_g_94_);
lean_closure_set(v___f_105_, 2, v_toMul_98_);
if (v_isShared_103_ == 0)
{
lean_ctor_set(v___x_102_, 1, v___f_104_);
lean_ctor_set(v___x_102_, 0, v___f_105_);
v___x_107_ = v___x_102_;
goto v_reusejp_106_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v___f_105_);
lean_ctor_set(v_reuseFailAlloc_108_, 1, v___f_104_);
v___x_107_ = v_reuseFailAlloc_108_;
goto v_reusejp_106_;
}
v_reusejp_106_:
{
return v___x_107_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg___boxed(lean_object* v_inst_111_, lean_object* v_g_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg(v_inst_111_, v_g_112_);
lean_dec_ref(v_inst_111_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup(lean_object* v_00_u03b1_114_, lean_object* v_inst_115_, lean_object* v_s_116_, lean_object* v_g_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib_Subgroup_rightCosetEquivSubgroup___redArg(v_inst_115_, v_g_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_rightCosetEquivSubgroup___boxed(lean_object* v_00_u03b1_119_, lean_object* v_inst_120_, lean_object* v_s_121_, lean_object* v_g_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_Subgroup_rightCosetEquivSubgroup(v_00_u03b1_119_, v_inst_120_, v_s_121_, v_g_122_);
lean_dec_ref(v_inst_120_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg___lam__0(lean_object* v_toAdd_124_, lean_object* v_g_125_, lean_object* v_x_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lean_apply_2(v_toAdd_124_, v_x_126_, v_g_125_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg___lam__1(lean_object* v_toNeg_128_, lean_object* v_g_129_, lean_object* v_toAdd_130_, lean_object* v_x_131_){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_132_ = lean_apply_1(v_toNeg_128_, v_g_129_);
v___x_133_ = lean_apply_2(v_toAdd_130_, v_x_131_, v___x_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg(lean_object* v_inst_134_, lean_object* v_g_135_){
_start:
{
lean_object* v_toAddMonoid_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v_toAdd_139_; lean_object* v___x_140_; lean_object* v_toNeg_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_150_; 
v_toAddMonoid_136_ = lean_ctor_get(v_inst_134_, 0);
v___x_137_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_136_);
v___x_138_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_137_);
v_toAdd_139_ = lean_ctor_get(v___x_138_, 1);
lean_inc(v_toAdd_139_);
lean_dec_ref(v___x_138_);
v___x_140_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_134_);
v_toNeg_141_ = lean_ctor_get(v___x_140_, 1);
v_isSharedCheck_150_ = !lean_is_exclusive(v___x_140_);
if (v_isSharedCheck_150_ == 0)
{
lean_object* v_unused_151_; 
v_unused_151_ = lean_ctor_get(v___x_140_, 0);
lean_dec(v_unused_151_);
v___x_143_ = v___x_140_;
v_isShared_144_ = v_isSharedCheck_150_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_toNeg_141_);
lean_dec(v___x_140_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_150_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v___f_145_; lean_object* v___f_146_; lean_object* v___x_148_; 
lean_inc(v_g_135_);
lean_inc(v_toAdd_139_);
v___f_145_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_145_, 0, v_toAdd_139_);
lean_closure_set(v___f_145_, 1, v_g_135_);
v___f_146_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg___lam__1), 4, 3);
lean_closure_set(v___f_146_, 0, v_toNeg_141_);
lean_closure_set(v___f_146_, 1, v_g_135_);
lean_closure_set(v___f_146_, 2, v_toAdd_139_);
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 1, v___f_145_);
lean_ctor_set(v___x_143_, 0, v___f_146_);
v___x_148_ = v___x_143_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_149_; 
v_reuseFailAlloc_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_149_, 0, v___f_146_);
lean_ctor_set(v_reuseFailAlloc_149_, 1, v___f_145_);
v___x_148_ = v_reuseFailAlloc_149_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
return v___x_148_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg___boxed(lean_object* v_inst_152_, lean_object* v_g_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg(v_inst_152_, v_g_153_);
lean_dec_ref(v_inst_152_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_, lean_object* v_s_157_, lean_object* v_g_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___redArg(v_inst_156_, v_g_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup___boxed(lean_object* v_00_u03b1_160_, lean_object* v_inst_161_, lean_object* v_s_162_, lean_object* v_g_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_AddSubgroup_rightCosetEquivAddSubgroup(v_00_u03b1_160_, v_inst_161_, v_s_162_, v_g_163_);
lean_dec_ref(v_inst_161_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg___lam__0(lean_object* v_f_165_, lean_object* v_toMul_166_, lean_object* v_a_167_){
_start:
{
lean_object* v_fst_168_; lean_object* v_snd_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v_fst_168_ = lean_ctor_get(v_a_167_, 0);
lean_inc(v_fst_168_);
v_snd_169_ = lean_ctor_get(v_a_167_, 1);
lean_inc(v_snd_169_);
lean_dec_ref(v_a_167_);
v___x_170_ = lean_apply_1(v_f_165_, v_fst_168_);
v___x_171_ = lean_apply_2(v_toMul_166_, v___x_170_, v_snd_169_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg___lam__1(lean_object* v_f_172_, lean_object* v_toInv_173_, lean_object* v_toMul_174_, lean_object* v_a_175_){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
lean_inc_n(v_a_175_, 2);
v___x_176_ = lean_apply_1(v_f_172_, v_a_175_);
v___x_177_ = lean_apply_1(v_toInv_173_, v___x_176_);
v___x_178_ = lean_apply_2(v_toMul_174_, v___x_177_, v_a_175_);
v___x_179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_179_, 0, v_a_175_);
lean_ctor_set(v___x_179_, 1, v___x_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg(lean_object* v_inst_180_, lean_object* v_f_181_){
_start:
{
lean_object* v_toMonoid_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v_toMul_185_; lean_object* v___x_186_; lean_object* v_toInv_187_; lean_object* v___x_189_; uint8_t v_isShared_190_; uint8_t v_isSharedCheck_196_; 
v_toMonoid_182_ = lean_ctor_get(v_inst_180_, 0);
v___x_183_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_182_);
v___x_184_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_183_);
v_toMul_185_ = lean_ctor_get(v___x_184_, 1);
lean_inc(v_toMul_185_);
lean_dec_ref(v___x_184_);
v___x_186_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_180_);
v_toInv_187_ = lean_ctor_get(v___x_186_, 1);
v_isSharedCheck_196_ = !lean_is_exclusive(v___x_186_);
if (v_isSharedCheck_196_ == 0)
{
lean_object* v_unused_197_; 
v_unused_197_ = lean_ctor_get(v___x_186_, 0);
lean_dec(v_unused_197_);
v___x_189_ = v___x_186_;
v_isShared_190_ = v_isSharedCheck_196_;
goto v_resetjp_188_;
}
else
{
lean_inc(v_toInv_187_);
lean_dec(v___x_186_);
v___x_189_ = lean_box(0);
v_isShared_190_ = v_isSharedCheck_196_;
goto v_resetjp_188_;
}
v_resetjp_188_:
{
lean_object* v___f_191_; lean_object* v___f_192_; lean_object* v___x_194_; 
lean_inc(v_toMul_185_);
lean_inc(v_f_181_);
v___f_191_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg___lam__0), 3, 2);
lean_closure_set(v___f_191_, 0, v_f_181_);
lean_closure_set(v___f_191_, 1, v_toMul_185_);
v___f_192_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg___lam__1), 4, 3);
lean_closure_set(v___f_192_, 0, v_f_181_);
lean_closure_set(v___f_192_, 1, v_toInv_187_);
lean_closure_set(v___f_192_, 2, v_toMul_185_);
if (v_isShared_190_ == 0)
{
lean_ctor_set(v___x_189_, 1, v___f_191_);
lean_ctor_set(v___x_189_, 0, v___f_192_);
v___x_194_ = v___x_189_;
goto v_reusejp_193_;
}
else
{
lean_object* v_reuseFailAlloc_195_; 
v_reuseFailAlloc_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_195_, 0, v___f_192_);
lean_ctor_set(v_reuseFailAlloc_195_, 1, v___f_191_);
v___x_194_ = v_reuseFailAlloc_195_;
goto v_reusejp_193_;
}
v_reusejp_193_:
{
return v___x_194_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg___boxed(lean_object* v_inst_198_, lean_object* v_f_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg(v_inst_198_, v_f_199_);
lean_dec_ref(v_inst_198_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27(lean_object* v_00_u03b1_201_, lean_object* v_inst_202_, lean_object* v_s_203_, lean_object* v_t_204_, lean_object* v_h__le_205_, lean_object* v_f_206_, lean_object* v_hf_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___redArg(v_inst_202_, v_f_206_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivProdOfLE_x27___boxed(lean_object* v_00_u03b1_209_, lean_object* v_inst_210_, lean_object* v_s_211_, lean_object* v_t_212_, lean_object* v_h__le_213_, lean_object* v_f_214_, lean_object* v_hf_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_Subgroup_quotientEquivProdOfLE_x27(v_00_u03b1_209_, v_inst_210_, v_s_211_, v_t_212_, v_h__le_213_, v_f_214_, v_hf_215_);
lean_dec_ref(v_inst_210_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg___lam__0(lean_object* v_f_217_, lean_object* v_toAdd_218_, lean_object* v_a_219_){
_start:
{
lean_object* v_fst_220_; lean_object* v_snd_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v_fst_220_ = lean_ctor_get(v_a_219_, 0);
lean_inc(v_fst_220_);
v_snd_221_ = lean_ctor_get(v_a_219_, 1);
lean_inc(v_snd_221_);
lean_dec_ref(v_a_219_);
v___x_222_ = lean_apply_1(v_f_217_, v_fst_220_);
v___x_223_ = lean_apply_2(v_toAdd_218_, v___x_222_, v_snd_221_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg___lam__1(lean_object* v_f_224_, lean_object* v_toNeg_225_, lean_object* v_toAdd_226_, lean_object* v_a_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
lean_inc_n(v_a_227_, 2);
v___x_228_ = lean_apply_1(v_f_224_, v_a_227_);
v___x_229_ = lean_apply_1(v_toNeg_225_, v___x_228_);
v___x_230_ = lean_apply_2(v_toAdd_226_, v___x_229_, v_a_227_);
v___x_231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_231_, 0, v_a_227_);
lean_ctor_set(v___x_231_, 1, v___x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg(lean_object* v_inst_232_, lean_object* v_f_233_){
_start:
{
lean_object* v_toAddMonoid_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v_toAdd_237_; lean_object* v___x_238_; lean_object* v_toNeg_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_248_; 
v_toAddMonoid_234_ = lean_ctor_get(v_inst_232_, 0);
v___x_235_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_234_);
v___x_236_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_235_);
v_toAdd_237_ = lean_ctor_get(v___x_236_, 1);
lean_inc(v_toAdd_237_);
lean_dec_ref(v___x_236_);
v___x_238_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_232_);
v_toNeg_239_ = lean_ctor_get(v___x_238_, 1);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_248_ == 0)
{
lean_object* v_unused_249_; 
v_unused_249_ = lean_ctor_get(v___x_238_, 0);
lean_dec(v_unused_249_);
v___x_241_ = v___x_238_;
v_isShared_242_ = v_isSharedCheck_248_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_toNeg_239_);
lean_dec(v___x_238_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_248_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___f_243_; lean_object* v___f_244_; lean_object* v___x_246_; 
lean_inc(v_toAdd_237_);
lean_inc(v_f_233_);
v___f_243_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg___lam__0), 3, 2);
lean_closure_set(v___f_243_, 0, v_f_233_);
lean_closure_set(v___f_243_, 1, v_toAdd_237_);
v___f_244_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg___lam__1), 4, 3);
lean_closure_set(v___f_244_, 0, v_f_233_);
lean_closure_set(v___f_244_, 1, v_toNeg_239_);
lean_closure_set(v___f_244_, 2, v_toAdd_237_);
if (v_isShared_242_ == 0)
{
lean_ctor_set(v___x_241_, 1, v___f_243_);
lean_ctor_set(v___x_241_, 0, v___f_244_);
v___x_246_ = v___x_241_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v___f_244_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v___f_243_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg___boxed(lean_object* v_inst_250_, lean_object* v_f_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg(v_inst_250_, v_f_251_);
lean_dec_ref(v_inst_250_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27(lean_object* v_00_u03b1_253_, lean_object* v_inst_254_, lean_object* v_s_255_, lean_object* v_t_256_, lean_object* v_h__le_257_, lean_object* v_f_258_, lean_object* v_hf_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___redArg(v_inst_254_, v_f_258_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27___boxed(lean_object* v_00_u03b1_261_, lean_object* v_inst_262_, lean_object* v_s_263_, lean_object* v_t_264_, lean_object* v_h__le_265_, lean_object* v_f_266_, lean_object* v_hf_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_AddSubgroup_quotientEquivProdOfLE_x27(v_00_u03b1_261_, v_inst_262_, v_s_263_, v_t_264_, v_h__le_265_, v_f_266_, v_hf_267_);
lean_dec_ref(v_inst_262_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE(lean_object* v_00_u03b1_273_, lean_object* v_inst_274_, lean_object* v_s_275_, lean_object* v_t_276_, lean_object* v_H_277_, lean_object* v_h_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = ((lean_object*)(lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__1));
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___boxed(lean_object* v_00_u03b1_280_, lean_object* v_inst_281_, lean_object* v_s_282_, lean_object* v_t_283_, lean_object* v_H_284_, lean_object* v_h_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE(v_00_u03b1_280_, v_inst_281_, v_s_282_, v_t_283_, v_H_284_, v_h_285_);
lean_dec_ref(v_inst_281_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfEmbeddingOfLE(lean_object* v_00_u03b1_287_, lean_object* v_inst_288_, lean_object* v_s_289_, lean_object* v_t_290_, lean_object* v_H_291_, lean_object* v_h_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = ((lean_object*)(lp_mathlib_Subgroup_quotientSubgroupOfEmbeddingOfLE___closed__1));
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfEmbeddingOfLE___boxed(lean_object* v_00_u03b1_294_, lean_object* v_inst_295_, lean_object* v_s_296_, lean_object* v_t_297_, lean_object* v_H_298_, lean_object* v_h_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_AddSubgroup_quotientAddSubgroupOfEmbeddingOfLE(v_00_u03b1_294_, v_inst_295_, v_s_296_, v_t_297_, v_H_298_, v_h_299_);
lean_dec_ref(v_inst_295_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE___redArg(lean_object* v_a_301_){
_start:
{
lean_inc(v_a_301_);
return v_a_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE___redArg___boxed(lean_object* v_a_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE___redArg(v_a_302_);
lean_dec(v_a_302_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE(lean_object* v_00_u03b1_304_, lean_object* v_inst_305_, lean_object* v_s_306_, lean_object* v_t_307_, lean_object* v_H_308_, lean_object* v_h_309_, lean_object* v_a_310_){
_start:
{
lean_inc(v_a_310_);
return v_a_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE___boxed(lean_object* v_00_u03b1_311_, lean_object* v_inst_312_, lean_object* v_s_313_, lean_object* v_t_314_, lean_object* v_H_315_, lean_object* v_h_316_, lean_object* v_a_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_mathlib_Subgroup_quotientSubgroupOfMapOfLE(v_00_u03b1_311_, v_inst_312_, v_s_313_, v_t_314_, v_H_315_, v_h_316_, v_a_317_);
lean_dec(v_a_317_);
lean_dec_ref(v_inst_312_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE___redArg(lean_object* v_a_319_){
_start:
{
lean_inc(v_a_319_);
return v_a_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE___redArg___boxed(lean_object* v_a_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE___redArg(v_a_320_);
lean_dec(v_a_320_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE(lean_object* v_00_u03b1_322_, lean_object* v_inst_323_, lean_object* v_s_324_, lean_object* v_t_325_, lean_object* v_H_326_, lean_object* v_h_327_, lean_object* v_a_328_){
_start:
{
lean_inc(v_a_328_);
return v_a_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE___boxed(lean_object* v_00_u03b1_329_, lean_object* v_inst_330_, lean_object* v_s_331_, lean_object* v_t_332_, lean_object* v_H_333_, lean_object* v_h_334_, lean_object* v_a_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_AddSubgroup_quotientAddSubgroupOfMapOfLE(v_00_u03b1_329_, v_inst_330_, v_s_331_, v_t_332_, v_H_333_, v_h_334_, v_a_335_);
lean_dec(v_a_335_);
lean_dec_ref(v_inst_330_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientMapOfLE___redArg(lean_object* v_a_337_){
_start:
{
lean_inc(v_a_337_);
return v_a_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientMapOfLE___redArg___boxed(lean_object* v_a_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib_Subgroup_quotientMapOfLE___redArg(v_a_338_);
lean_dec(v_a_338_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientMapOfLE(lean_object* v_00_u03b1_340_, lean_object* v_inst_341_, lean_object* v_s_342_, lean_object* v_t_343_, lean_object* v_h_344_, lean_object* v_a_345_){
_start:
{
lean_inc(v_a_345_);
return v_a_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientMapOfLE___boxed(lean_object* v_00_u03b1_346_, lean_object* v_inst_347_, lean_object* v_s_348_, lean_object* v_t_349_, lean_object* v_h_350_, lean_object* v_a_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_mathlib_Subgroup_quotientMapOfLE(v_00_u03b1_346_, v_inst_347_, v_s_348_, v_t_349_, v_h_350_, v_a_351_);
lean_dec(v_a_351_);
lean_dec_ref(v_inst_347_);
return v_res_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientMapOfLE___redArg(lean_object* v_a_353_){
_start:
{
lean_inc(v_a_353_);
return v_a_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientMapOfLE___redArg___boxed(lean_object* v_a_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib_AddSubgroup_quotientMapOfLE___redArg(v_a_354_);
lean_dec(v_a_354_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientMapOfLE(lean_object* v_00_u03b1_356_, lean_object* v_inst_357_, lean_object* v_s_358_, lean_object* v_t_359_, lean_object* v_h_360_, lean_object* v_a_361_){
_start:
{
lean_inc(v_a_361_);
return v_a_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientMapOfLE___boxed(lean_object* v_00_u03b1_362_, lean_object* v_inst_363_, lean_object* v_s_364_, lean_object* v_t_365_, lean_object* v_h_366_, lean_object* v_a_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_mathlib_AddSubgroup_quotientMapOfLE(v_00_u03b1_362_, v_inst_363_, v_s_364_, v_t_365_, v_h_366_, v_a_367_);
lean_dec(v_a_367_);
lean_dec_ref(v_inst_363_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___lam__0(lean_object* v_q_369_, lean_object* v_i_370_){
_start:
{
lean_inc(v_q_369_);
return v_q_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___lam__0___boxed(lean_object* v_q_371_, lean_object* v_i_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___lam__0(v_q_371_, v_i_372_);
lean_dec(v_i_372_);
lean_dec(v_q_371_);
return v_res_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding(lean_object* v_00_u03b1_375_, lean_object* v_inst_376_, lean_object* v_00_u03b9_377_, lean_object* v_f_378_, lean_object* v_H_379_){
_start:
{
lean_object* v___f_380_; 
v___f_380_ = ((lean_object*)(lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___closed__0));
return v___f_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___boxed(lean_object* v_00_u03b1_381_, lean_object* v_inst_382_, lean_object* v_00_u03b9_383_, lean_object* v_f_384_, lean_object* v_H_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding(v_00_u03b1_381_, v_inst_382_, v_00_u03b9_383_, v_f_384_, v_H_385_);
lean_dec_ref(v_f_384_);
lean_dec_ref(v_inst_382_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientiInfAddSubgroupOfEmbedding(lean_object* v_00_u03b1_387_, lean_object* v_inst_388_, lean_object* v_00_u03b9_389_, lean_object* v_f_390_, lean_object* v_H_391_){
_start:
{
lean_object* v___f_392_; 
v___f_392_ = ((lean_object*)(lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___closed__0));
return v___f_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientiInfAddSubgroupOfEmbedding___boxed(lean_object* v_00_u03b1_393_, lean_object* v_inst_394_, lean_object* v_00_u03b9_395_, lean_object* v_f_396_, lean_object* v_H_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_mathlib_AddSubgroup_quotientiInfAddSubgroupOfEmbedding(v_00_u03b1_393_, v_inst_394_, v_00_u03b9_395_, v_f_396_, v_H_397_);
lean_dec_ref(v_f_396_);
lean_dec_ref(v_inst_394_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfEmbedding(lean_object* v_00_u03b1_399_, lean_object* v_inst_400_, lean_object* v_00_u03b9_401_, lean_object* v_f_402_){
_start:
{
lean_object* v___f_403_; 
v___f_403_ = ((lean_object*)(lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___closed__0));
return v___f_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientiInfEmbedding___boxed(lean_object* v_00_u03b1_404_, lean_object* v_inst_405_, lean_object* v_00_u03b9_406_, lean_object* v_f_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_Subgroup_quotientiInfEmbedding(v_00_u03b1_404_, v_inst_405_, v_00_u03b9_406_, v_f_407_);
lean_dec_ref(v_f_407_);
lean_dec_ref(v_inst_405_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientiInfEmbedding(lean_object* v_00_u03b1_409_, lean_object* v_inst_410_, lean_object* v_00_u03b9_411_, lean_object* v_f_412_){
_start:
{
lean_object* v___f_413_; 
v___f_413_ = ((lean_object*)(lp_mathlib_Subgroup_quotientiInfSubgroupOfEmbedding___closed__0));
return v___f_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientiInfEmbedding___boxed(lean_object* v_00_u03b1_414_, lean_object* v_inst_415_, lean_object* v_00_u03b9_416_, lean_object* v_f_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_AddSubgroup_quotientiInfEmbedding(v_00_u03b1_414_, v_inst_415_, v_00_u03b9_416_, v_f_417_);
lean_dec_ref(v_f_417_);
lean_dec_ref(v_inst_415_);
return v_res_418_;
}
}
static lean_object* _init_lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0(void){
_start:
{
lean_object* v___x_419_; 
v___x_419_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquivKer___redArg(lean_object* v_inst_420_, lean_object* v_a_421_){
_start:
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_422_ = lean_obj_once(&lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0, &lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0_once, _init_lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0);
v___x_423_ = lp_mathlib_Subgroup_leftCosetEquivSubgroup___redArg(v_inst_420_, v_a_421_);
v___x_424_ = lp_mathlib_Equiv_trans___redArg(v___x_422_, v___x_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquivKer___redArg___boxed(lean_object* v_inst_425_, lean_object* v_a_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_MonoidHom_fiberEquivKer___redArg(v_inst_425_, v_a_426_);
lean_dec_ref(v_inst_425_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquivKer(lean_object* v_00_u03b1_428_, lean_object* v_inst_429_, lean_object* v_H_430_, lean_object* v_inst_431_, lean_object* v_f_432_, lean_object* v_a_433_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lp_mathlib_MonoidHom_fiberEquivKer___redArg(v_inst_429_, v_a_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquivKer___boxed(lean_object* v_00_u03b1_435_, lean_object* v_inst_436_, lean_object* v_H_437_, lean_object* v_inst_438_, lean_object* v_f_439_, lean_object* v_a_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_mathlib_MonoidHom_fiberEquivKer(v_00_u03b1_435_, v_inst_436_, v_H_437_, v_inst_438_, v_f_439_, v_a_440_);
lean_dec(v_f_439_);
lean_dec_ref(v_inst_438_);
lean_dec_ref(v_inst_436_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquivKer___redArg(lean_object* v_inst_442_, lean_object* v_a_443_){
_start:
{
lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_444_ = lean_obj_once(&lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0, &lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0_once, _init_lp_mathlib_MonoidHom_fiberEquivKer___redArg___closed__0);
v___x_445_ = lp_mathlib_AddSubgroup_leftCosetEquivAddSubgroup___redArg(v_inst_442_, v_a_443_);
v___x_446_ = lp_mathlib_Equiv_trans___redArg(v___x_444_, v___x_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquivKer___redArg___boxed(lean_object* v_inst_447_, lean_object* v_a_448_){
_start:
{
lean_object* v_res_449_; 
v_res_449_ = lp_mathlib_AddMonoidHom_fiberEquivKer___redArg(v_inst_447_, v_a_448_);
lean_dec_ref(v_inst_447_);
return v_res_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquivKer(lean_object* v_00_u03b1_450_, lean_object* v_inst_451_, lean_object* v_H_452_, lean_object* v_inst_453_, lean_object* v_f_454_, lean_object* v_a_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_AddMonoidHom_fiberEquivKer___redArg(v_inst_451_, v_a_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquivKer___boxed(lean_object* v_00_u03b1_457_, lean_object* v_inst_458_, lean_object* v_H_459_, lean_object* v_inst_460_, lean_object* v_f_461_, lean_object* v_a_462_){
_start:
{
lean_object* v_res_463_; 
v_res_463_ = lp_mathlib_AddMonoidHom_fiberEquivKer(v_00_u03b1_457_, v_inst_458_, v_H_459_, v_inst_460_, v_f_461_, v_a_462_);
lean_dec(v_f_461_);
lean_dec_ref(v_inst_460_);
lean_dec_ref(v_inst_458_);
return v_res_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquiv___redArg(lean_object* v_inst_464_, lean_object* v_a_465_, lean_object* v_b_466_){
_start:
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; 
v___x_467_ = lp_mathlib_MonoidHom_fiberEquivKer___redArg(v_inst_464_, v_a_465_);
v___x_468_ = lp_mathlib_MonoidHom_fiberEquivKer___redArg(v_inst_464_, v_b_466_);
v___x_469_ = lp_mathlib_Equiv_symm___redArg(v___x_468_);
v___x_470_ = lp_mathlib_Equiv_trans___redArg(v___x_467_, v___x_469_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquiv___redArg___boxed(lean_object* v_inst_471_, lean_object* v_a_472_, lean_object* v_b_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_mathlib_MonoidHom_fiberEquiv___redArg(v_inst_471_, v_a_472_, v_b_473_);
lean_dec_ref(v_inst_471_);
return v_res_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquiv(lean_object* v_00_u03b1_475_, lean_object* v_inst_476_, lean_object* v_H_477_, lean_object* v_inst_478_, lean_object* v_f_479_, lean_object* v_a_480_, lean_object* v_b_481_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lp_mathlib_MonoidHom_fiberEquiv___redArg(v_inst_476_, v_a_480_, v_b_481_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fiberEquiv___boxed(lean_object* v_00_u03b1_483_, lean_object* v_inst_484_, lean_object* v_H_485_, lean_object* v_inst_486_, lean_object* v_f_487_, lean_object* v_a_488_, lean_object* v_b_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_mathlib_MonoidHom_fiberEquiv(v_00_u03b1_483_, v_inst_484_, v_H_485_, v_inst_486_, v_f_487_, v_a_488_, v_b_489_);
lean_dec(v_f_487_);
lean_dec_ref(v_inst_486_);
lean_dec_ref(v_inst_484_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquiv___redArg(lean_object* v_inst_491_, lean_object* v_a_492_, lean_object* v_b_493_){
_start:
{
lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; 
v___x_494_ = lp_mathlib_AddMonoidHom_fiberEquivKer___redArg(v_inst_491_, v_a_492_);
v___x_495_ = lp_mathlib_AddMonoidHom_fiberEquivKer___redArg(v_inst_491_, v_b_493_);
v___x_496_ = lp_mathlib_Equiv_symm___redArg(v___x_495_);
v___x_497_ = lp_mathlib_Equiv_trans___redArg(v___x_494_, v___x_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquiv___redArg___boxed(lean_object* v_inst_498_, lean_object* v_a_499_, lean_object* v_b_500_){
_start:
{
lean_object* v_res_501_; 
v_res_501_ = lp_mathlib_AddMonoidHom_fiberEquiv___redArg(v_inst_498_, v_a_499_, v_b_500_);
lean_dec_ref(v_inst_498_);
return v_res_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquiv(lean_object* v_00_u03b1_502_, lean_object* v_inst_503_, lean_object* v_H_504_, lean_object* v_inst_505_, lean_object* v_f_506_, lean_object* v_a_507_, lean_object* v_b_508_){
_start:
{
lean_object* v___x_509_; 
v___x_509_ = lp_mathlib_AddMonoidHom_fiberEquiv___redArg(v_inst_503_, v_a_507_, v_b_508_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fiberEquiv___boxed(lean_object* v_00_u03b1_510_, lean_object* v_inst_511_, lean_object* v_H_512_, lean_object* v_inst_513_, lean_object* v_f_514_, lean_object* v_a_515_, lean_object* v_b_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_mathlib_AddMonoidHom_fiberEquiv(v_00_u03b1_510_, v_inst_511_, v_H_512_, v_inst_513_, v_f_514_, v_a_515_, v_b_516_);
lean_dec(v_f_514_);
lean_dec_ref(v_inst_513_);
lean_dec_ref(v_inst_511_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientEquivSelf___redArg(lean_object* v_inst_522_){
_start:
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; 
v___x_523_ = lean_box(0);
v___x_524_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__1));
v___x_525_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_525_, 0, lean_box(0));
lean_closure_set(v___x_525_, 1, v_inst_522_);
lean_closure_set(v___x_525_, 2, v___x_523_);
v___x_526_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_526_, 0, v___x_524_);
lean_ctor_set(v___x_526_, 1, v___x_525_);
return v___x_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientEquivSelf(lean_object* v_00_u03b1_527_, lean_object* v_inst_528_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_mathlib_QuotientGroup_quotientEquivSelf___redArg(v_inst_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientEquivSelf___redArg(lean_object* v_inst_530_){
_start:
{
lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; 
v___x_531_ = lean_box(0);
v___x_532_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientEquivSelf___redArg___closed__1));
v___x_533_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_533_, 0, lean_box(0));
lean_closure_set(v___x_533_, 1, v_inst_530_);
lean_closure_set(v___x_533_, 2, v___x_531_);
v___x_534_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_532_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientEquivSelf(lean_object* v_00_u03b1_535_, lean_object* v_inst_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lp_mathlib_QuotientAddGroup_quotientEquivSelf___redArg(v_inst_536_);
return v___x_537_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
