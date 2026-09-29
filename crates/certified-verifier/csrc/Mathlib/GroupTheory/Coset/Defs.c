// Lean compiler output
// Module: Mathlib.GroupTheory.Coset.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Quotient public import Mathlib.Algebra.Group.Action.Opposite public import Mathlib.Algebra.Group.Subgroup.MulOpposite public import Mathlib.GroupTheory.GroupAction.Defs public import Mathlib.Algebra.Group.Pointwise.Set.Basic
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_map_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_leftRel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_leftRel___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_leftRel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_leftRel___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_leftRelDecidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_leftRelDecidable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_leftRelDecidable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_leftRelDecidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_leftRelDecidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_leftRelDecidable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_leftRelDecidable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_leftRelDecidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instHasQuotientSubgroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instHasQuotientSubgroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instHasQuotientAddSubgroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instHasQuotientAddSubgroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rightRel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rightRel___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rightRel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rightRel___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_rightRelDecidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rightRelDecidable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_rightRelDecidable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rightRelDecidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_rightRelDecidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rightRelDecidable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_rightRelDecidable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rightRelDecidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instCoeQuotientSubgroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instCoeQuotientSubgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instCoeQuotientAddSubgroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instCoeQuotientAddSubgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_quotientEquivOfEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Subgroup_quotientEquivOfEq___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_quotientEquivOfEq___closed__0_value;
static const lean_closure_object lp_mathlib_Subgroup_quotientEquivOfEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Quotient_map_x27, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Subgroup_quotientEquivOfEq___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Subgroup_quotientEquivOfEq___closed__1 = (const lean_object*)&lp_mathlib_Subgroup_quotientEquivOfEq___closed__1_value;
static const lean_ctor_object lp_mathlib_Subgroup_quotientEquivOfEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Subgroup_quotientEquivOfEq___closed__1_value),((lean_object*)&lp_mathlib_Subgroup_quotientEquivOfEq___closed__1_value)}};
static const lean_object* lp_mathlib_Subgroup_quotientEquivOfEq___closed__2 = (const lean_object*)&lp_mathlib_Subgroup_quotientEquivOfEq___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivOfEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivOfEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_leftRel(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_, lean_object* v_s_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_leftRel___boxed(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_, lean_object* v_s_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_QuotientGroup_leftRel(v_00_u03b1_5_, v_inst_6_, v_s_7_);
lean_dec_ref(v_inst_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_leftRel(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_, lean_object* v_s_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_leftRel___boxed(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_, lean_object* v_s_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_QuotientAddGroup_leftRel(v_00_u03b1_13_, v_inst_14_, v_s_15_);
lean_dec_ref(v_inst_14_);
return v_res_16_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_leftRelDecidable___redArg(lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_x_19_, lean_object* v_y_20_){
_start:
{
lean_object* v_toMonoid_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v_toMul_24_; lean_object* v___x_25_; lean_object* v_toInv_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; uint8_t v___x_30_; 
v_toMonoid_21_ = lean_ctor_get(v_inst_17_, 0);
v___x_22_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_21_);
v___x_23_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_22_);
v_toMul_24_ = lean_ctor_get(v___x_23_, 1);
lean_inc(v_toMul_24_);
lean_dec_ref(v___x_23_);
v___x_25_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_17_);
v_toInv_26_ = lean_ctor_get(v___x_25_, 1);
lean_inc(v_toInv_26_);
lean_dec_ref(v___x_25_);
v___x_27_ = lean_apply_1(v_toInv_26_, v_x_19_);
v___x_28_ = lean_apply_2(v_toMul_24_, v___x_27_, v_y_20_);
v___x_29_ = lean_apply_1(v_inst_18_, v___x_28_);
v___x_30_ = lean_unbox(v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_leftRelDecidable___redArg___boxed(lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_x_33_, lean_object* v_y_34_){
_start:
{
uint8_t v_res_35_; lean_object* v_r_36_; 
v_res_35_ = lp_mathlib_QuotientGroup_leftRelDecidable___redArg(v_inst_31_, v_inst_32_, v_x_33_, v_y_34_);
lean_dec_ref(v_inst_31_);
v_r_36_ = lean_box(v_res_35_);
return v_r_36_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_leftRelDecidable(lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_s_39_, lean_object* v_inst_40_, lean_object* v_x_41_, lean_object* v_y_42_){
_start:
{
uint8_t v___x_43_; 
v___x_43_ = lp_mathlib_QuotientGroup_leftRelDecidable___redArg(v_inst_38_, v_inst_40_, v_x_41_, v_y_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_leftRelDecidable___boxed(lean_object* v_00_u03b1_44_, lean_object* v_inst_45_, lean_object* v_s_46_, lean_object* v_inst_47_, lean_object* v_x_48_, lean_object* v_y_49_){
_start:
{
uint8_t v_res_50_; lean_object* v_r_51_; 
v_res_50_ = lp_mathlib_QuotientGroup_leftRelDecidable(v_00_u03b1_44_, v_inst_45_, v_s_46_, v_inst_47_, v_x_48_, v_y_49_);
lean_dec_ref(v_inst_45_);
v_r_51_ = lean_box(v_res_50_);
return v_r_51_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_leftRelDecidable___redArg(lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_x_54_, lean_object* v_y_55_){
_start:
{
lean_object* v_toAddMonoid_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v_toAdd_59_; lean_object* v___x_60_; lean_object* v_toNeg_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; uint8_t v___x_65_; 
v_toAddMonoid_56_ = lean_ctor_get(v_inst_52_, 0);
v___x_57_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_56_);
v___x_58_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_57_);
v_toAdd_59_ = lean_ctor_get(v___x_58_, 1);
lean_inc(v_toAdd_59_);
lean_dec_ref(v___x_58_);
v___x_60_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_52_);
v_toNeg_61_ = lean_ctor_get(v___x_60_, 1);
lean_inc(v_toNeg_61_);
lean_dec_ref(v___x_60_);
v___x_62_ = lean_apply_1(v_toNeg_61_, v_x_54_);
v___x_63_ = lean_apply_2(v_toAdd_59_, v___x_62_, v_y_55_);
v___x_64_ = lean_apply_1(v_inst_53_, v___x_63_);
v___x_65_ = lean_unbox(v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_leftRelDecidable___redArg___boxed(lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_x_68_, lean_object* v_y_69_){
_start:
{
uint8_t v_res_70_; lean_object* v_r_71_; 
v_res_70_ = lp_mathlib_QuotientAddGroup_leftRelDecidable___redArg(v_inst_66_, v_inst_67_, v_x_68_, v_y_69_);
lean_dec_ref(v_inst_66_);
v_r_71_ = lean_box(v_res_70_);
return v_r_71_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_leftRelDecidable(lean_object* v_00_u03b1_72_, lean_object* v_inst_73_, lean_object* v_s_74_, lean_object* v_inst_75_, lean_object* v_x_76_, lean_object* v_y_77_){
_start:
{
uint8_t v___x_78_; 
v___x_78_ = lp_mathlib_QuotientAddGroup_leftRelDecidable___redArg(v_inst_73_, v_inst_75_, v_x_76_, v_y_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_leftRelDecidable___boxed(lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_s_81_, lean_object* v_inst_82_, lean_object* v_x_83_, lean_object* v_y_84_){
_start:
{
uint8_t v_res_85_; lean_object* v_r_86_; 
v_res_85_ = lp_mathlib_QuotientAddGroup_leftRelDecidable(v_00_u03b1_79_, v_inst_80_, v_s_81_, v_inst_82_, v_x_83_, v_y_84_);
lean_dec_ref(v_inst_80_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instHasQuotientSubgroup(lean_object* v_00_u03b1_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_box(0);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instHasQuotientSubgroup___boxed(lean_object* v_00_u03b1_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_QuotientGroup_instHasQuotientSubgroup(v_00_u03b1_90_, v_inst_91_);
lean_dec_ref(v_inst_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instHasQuotientAddSubgroup(lean_object* v_00_u03b1_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lean_box(0);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instHasQuotientAddSubgroup___boxed(lean_object* v_00_u03b1_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_QuotientAddGroup_instHasQuotientAddSubgroup(v_00_u03b1_96_, v_inst_97_);
lean_dec_ref(v_inst_97_);
return v_res_98_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem___redArg(lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_a_101_, lean_object* v_b_102_){
_start:
{
uint8_t v___x_103_; 
v___x_103_ = lp_mathlib_QuotientGroup_leftRelDecidable___redArg(v_inst_99_, v_inst_100_, v_a_101_, v_b_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem___redArg___boxed(lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_a_106_, lean_object* v_b_107_){
_start:
{
uint8_t v_res_108_; lean_object* v_r_109_; 
v_res_108_ = lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem___redArg(v_inst_104_, v_inst_105_, v_a_106_, v_b_107_);
lean_dec_ref(v_inst_104_);
v_r_109_ = lean_box(v_res_108_);
return v_r_109_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem(lean_object* v_00_u03b1_110_, lean_object* v_inst_111_, lean_object* v_s_112_, lean_object* v_inst_113_, lean_object* v_a_114_, lean_object* v_b_115_){
_start:
{
uint8_t v___x_116_; 
v___x_116_ = lp_mathlib_QuotientGroup_leftRelDecidable___redArg(v_inst_111_, v_inst_113_, v_a_114_, v_b_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem___boxed(lean_object* v_00_u03b1_117_, lean_object* v_inst_118_, lean_object* v_s_119_, lean_object* v_inst_120_, lean_object* v_a_121_, lean_object* v_b_122_){
_start:
{
uint8_t v_res_123_; lean_object* v_r_124_; 
v_res_123_ = lp_mathlib_QuotientGroup_instDecidableEqQuotientSubgroupOfDecidablePredMem(v_00_u03b1_117_, v_inst_118_, v_s_119_, v_inst_120_, v_a_121_, v_b_122_);
lean_dec_ref(v_inst_118_);
v_r_124_ = lean_box(v_res_123_);
return v_r_124_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem___redArg(lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_a_127_, lean_object* v_b_128_){
_start:
{
uint8_t v___x_129_; 
v___x_129_ = lp_mathlib_QuotientAddGroup_leftRelDecidable___redArg(v_inst_125_, v_inst_126_, v_a_127_, v_b_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem___redArg___boxed(lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_a_132_, lean_object* v_b_133_){
_start:
{
uint8_t v_res_134_; lean_object* v_r_135_; 
v_res_134_ = lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem___redArg(v_inst_130_, v_inst_131_, v_a_132_, v_b_133_);
lean_dec_ref(v_inst_130_);
v_r_135_ = lean_box(v_res_134_);
return v_r_135_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem(lean_object* v_00_u03b1_136_, lean_object* v_inst_137_, lean_object* v_s_138_, lean_object* v_inst_139_, lean_object* v_a_140_, lean_object* v_b_141_){
_start:
{
uint8_t v___x_142_; 
v___x_142_ = lp_mathlib_QuotientAddGroup_leftRelDecidable___redArg(v_inst_137_, v_inst_139_, v_a_140_, v_b_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem___boxed(lean_object* v_00_u03b1_143_, lean_object* v_inst_144_, lean_object* v_s_145_, lean_object* v_inst_146_, lean_object* v_a_147_, lean_object* v_b_148_){
_start:
{
uint8_t v_res_149_; lean_object* v_r_150_; 
v_res_149_ = lp_mathlib_QuotientAddGroup_instDecidableEqQuotientAddSubgroupOfDecidablePredMem(v_00_u03b1_143_, v_inst_144_, v_s_145_, v_inst_146_, v_a_147_, v_b_148_);
lean_dec_ref(v_inst_144_);
v_r_150_ = lean_box(v_res_149_);
return v_r_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rightRel(lean_object* v_00_u03b1_151_, lean_object* v_inst_152_, lean_object* v_s_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lean_box(0);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rightRel___boxed(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_, lean_object* v_s_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_QuotientGroup_rightRel(v_00_u03b1_155_, v_inst_156_, v_s_157_);
lean_dec_ref(v_inst_156_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rightRel(lean_object* v_00_u03b1_159_, lean_object* v_inst_160_, lean_object* v_s_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_box(0);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rightRel___boxed(lean_object* v_00_u03b1_163_, lean_object* v_inst_164_, lean_object* v_s_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_QuotientAddGroup_rightRel(v_00_u03b1_163_, v_inst_164_, v_s_165_);
lean_dec_ref(v_inst_164_);
return v_res_166_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_rightRelDecidable___redArg(lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_x_169_, lean_object* v_y_170_){
_start:
{
lean_object* v_toMonoid_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v_toMul_174_; lean_object* v___x_175_; lean_object* v_toInv_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; uint8_t v___x_180_; 
v_toMonoid_171_ = lean_ctor_get(v_inst_167_, 0);
v___x_172_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_171_);
v___x_173_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_172_);
v_toMul_174_ = lean_ctor_get(v___x_173_, 1);
lean_inc(v_toMul_174_);
lean_dec_ref(v___x_173_);
v___x_175_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_167_);
v_toInv_176_ = lean_ctor_get(v___x_175_, 1);
lean_inc(v_toInv_176_);
lean_dec_ref(v___x_175_);
v___x_177_ = lean_apply_1(v_toInv_176_, v_x_169_);
v___x_178_ = lean_apply_2(v_toMul_174_, v_y_170_, v___x_177_);
v___x_179_ = lean_apply_1(v_inst_168_, v___x_178_);
v___x_180_ = lean_unbox(v___x_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rightRelDecidable___redArg___boxed(lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_x_183_, lean_object* v_y_184_){
_start:
{
uint8_t v_res_185_; lean_object* v_r_186_; 
v_res_185_ = lp_mathlib_QuotientGroup_rightRelDecidable___redArg(v_inst_181_, v_inst_182_, v_x_183_, v_y_184_);
lean_dec_ref(v_inst_181_);
v_r_186_ = lean_box(v_res_185_);
return v_r_186_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientGroup_rightRelDecidable(lean_object* v_00_u03b1_187_, lean_object* v_inst_188_, lean_object* v_s_189_, lean_object* v_inst_190_, lean_object* v_x_191_, lean_object* v_y_192_){
_start:
{
uint8_t v___x_193_; 
v___x_193_ = lp_mathlib_QuotientGroup_rightRelDecidable___redArg(v_inst_188_, v_inst_190_, v_x_191_, v_y_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rightRelDecidable___boxed(lean_object* v_00_u03b1_194_, lean_object* v_inst_195_, lean_object* v_s_196_, lean_object* v_inst_197_, lean_object* v_x_198_, lean_object* v_y_199_){
_start:
{
uint8_t v_res_200_; lean_object* v_r_201_; 
v_res_200_ = lp_mathlib_QuotientGroup_rightRelDecidable(v_00_u03b1_194_, v_inst_195_, v_s_196_, v_inst_197_, v_x_198_, v_y_199_);
lean_dec_ref(v_inst_195_);
v_r_201_ = lean_box(v_res_200_);
return v_r_201_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_rightRelDecidable___redArg(lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_x_204_, lean_object* v_y_205_){
_start:
{
lean_object* v_toAddMonoid_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v_toAdd_209_; lean_object* v___x_210_; lean_object* v_toNeg_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; uint8_t v___x_215_; 
v_toAddMonoid_206_ = lean_ctor_get(v_inst_202_, 0);
v___x_207_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_206_);
v___x_208_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_207_);
v_toAdd_209_ = lean_ctor_get(v___x_208_, 1);
lean_inc(v_toAdd_209_);
lean_dec_ref(v___x_208_);
v___x_210_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_202_);
v_toNeg_211_ = lean_ctor_get(v___x_210_, 1);
lean_inc(v_toNeg_211_);
lean_dec_ref(v___x_210_);
v___x_212_ = lean_apply_1(v_toNeg_211_, v_x_204_);
v___x_213_ = lean_apply_2(v_toAdd_209_, v_y_205_, v___x_212_);
v___x_214_ = lean_apply_1(v_inst_203_, v___x_213_);
v___x_215_ = lean_unbox(v___x_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rightRelDecidable___redArg___boxed(lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_x_218_, lean_object* v_y_219_){
_start:
{
uint8_t v_res_220_; lean_object* v_r_221_; 
v_res_220_ = lp_mathlib_QuotientAddGroup_rightRelDecidable___redArg(v_inst_216_, v_inst_217_, v_x_218_, v_y_219_);
lean_dec_ref(v_inst_216_);
v_r_221_ = lean_box(v_res_220_);
return v_r_221_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_QuotientAddGroup_rightRelDecidable(lean_object* v_00_u03b1_222_, lean_object* v_inst_223_, lean_object* v_s_224_, lean_object* v_inst_225_, lean_object* v_x_226_, lean_object* v_y_227_){
_start:
{
uint8_t v___x_228_; 
v___x_228_ = lp_mathlib_QuotientAddGroup_rightRelDecidable___redArg(v_inst_223_, v_inst_225_, v_x_226_, v_y_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rightRelDecidable___boxed(lean_object* v_00_u03b1_229_, lean_object* v_inst_230_, lean_object* v_s_231_, lean_object* v_inst_232_, lean_object* v_x_233_, lean_object* v_y_234_){
_start:
{
uint8_t v_res_235_; lean_object* v_r_236_; 
v_res_235_ = lp_mathlib_QuotientAddGroup_rightRelDecidable(v_00_u03b1_229_, v_inst_230_, v_s_231_, v_inst_232_, v_x_233_, v_y_234_);
lean_dec_ref(v_inst_230_);
v_r_236_ = lean_box(v_res_235_);
return v_r_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg___lam__0(lean_object* v_toInv_237_, lean_object* v_g_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lean_apply_1(v_toInv_237_, v_g_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg(lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v_toInv_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_252_; 
v___x_241_ = lean_box(0);
v___x_242_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_240_);
v_toInv_243_ = lean_ctor_get(v___x_242_, 1);
v_isSharedCheck_252_ = !lean_is_exclusive(v___x_242_);
if (v_isSharedCheck_252_ == 0)
{
lean_object* v_unused_253_; 
v_unused_253_ = lean_ctor_get(v___x_242_, 0);
lean_dec(v_unused_253_);
v___x_245_ = v___x_242_;
v_isShared_246_ = v_isSharedCheck_252_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_toInv_243_);
lean_dec(v___x_242_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_252_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___f_247_; lean_object* v___x_248_; lean_object* v___x_250_; 
v___f_247_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg___lam__0), 2, 1);
lean_closure_set(v___f_247_, 0, v_toInv_243_);
v___x_248_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_x27), 7, 6);
lean_closure_set(v___x_248_, 0, lean_box(0));
lean_closure_set(v___x_248_, 1, lean_box(0));
lean_closure_set(v___x_248_, 2, v___x_241_);
lean_closure_set(v___x_248_, 3, v___x_241_);
lean_closure_set(v___x_248_, 4, v___f_247_);
lean_closure_set(v___x_248_, 5, lean_box(0));
lean_inc_ref(v___x_248_);
if (v_isShared_246_ == 0)
{
lean_ctor_set(v___x_245_, 1, v___x_248_);
lean_ctor_set(v___x_245_, 0, v___x_248_);
v___x_250_ = v___x_245_;
goto v_reusejp_249_;
}
else
{
lean_object* v_reuseFailAlloc_251_; 
v_reuseFailAlloc_251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_251_, 0, v___x_248_);
lean_ctor_set(v_reuseFailAlloc_251_, 1, v___x_248_);
v___x_250_ = v_reuseFailAlloc_251_;
goto v_reusejp_249_;
}
v_reusejp_249_:
{
return v___x_250_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg___boxed(lean_object* v_inst_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg(v_inst_254_);
lean_dec_ref(v_inst_254_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel(lean_object* v_00_u03b1_256_, lean_object* v_inst_257_, lean_object* v_s_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___redArg(v_inst_257_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel___boxed(lean_object* v_00_u03b1_260_, lean_object* v_inst_261_, lean_object* v_s_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_QuotientGroup_quotientRightRelEquivQuotientLeftRel(v_00_u03b1_260_, v_inst_261_, v_s_262_);
lean_dec_ref(v_inst_261_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg___lam__0(lean_object* v_toNeg_264_, lean_object* v_g_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lean_apply_1(v_toNeg_264_, v_g_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg(lean_object* v_inst_267_){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v_toNeg_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_279_; 
v___x_268_ = lean_box(0);
v___x_269_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_267_);
v_toNeg_270_ = lean_ctor_get(v___x_269_, 1);
v_isSharedCheck_279_ = !lean_is_exclusive(v___x_269_);
if (v_isSharedCheck_279_ == 0)
{
lean_object* v_unused_280_; 
v_unused_280_ = lean_ctor_get(v___x_269_, 0);
lean_dec(v_unused_280_);
v___x_272_ = v___x_269_;
v_isShared_273_ = v_isSharedCheck_279_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_toNeg_270_);
lean_dec(v___x_269_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_279_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___f_274_; lean_object* v___x_275_; lean_object* v___x_277_; 
v___f_274_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg___lam__0), 2, 1);
lean_closure_set(v___f_274_, 0, v_toNeg_270_);
v___x_275_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_x27), 7, 6);
lean_closure_set(v___x_275_, 0, lean_box(0));
lean_closure_set(v___x_275_, 1, lean_box(0));
lean_closure_set(v___x_275_, 2, v___x_268_);
lean_closure_set(v___x_275_, 3, v___x_268_);
lean_closure_set(v___x_275_, 4, v___f_274_);
lean_closure_set(v___x_275_, 5, lean_box(0));
lean_inc_ref(v___x_275_);
if (v_isShared_273_ == 0)
{
lean_ctor_set(v___x_272_, 1, v___x_275_);
lean_ctor_set(v___x_272_, 0, v___x_275_);
v___x_277_ = v___x_272_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v___x_275_);
lean_ctor_set(v_reuseFailAlloc_278_, 1, v___x_275_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg___boxed(lean_object* v_inst_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg(v_inst_281_);
lean_dec_ref(v_inst_281_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel(lean_object* v_00_u03b1_283_, lean_object* v_inst_284_, lean_object* v_s_285_){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___redArg(v_inst_284_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel___boxed(lean_object* v_00_u03b1_287_, lean_object* v_inst_288_, lean_object* v_s_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_QuotientAddGroup_quotientRightRelEquivQuotientLeftRel(v_00_u03b1_287_, v_inst_288_, v_s_289_);
lean_dec_ref(v_inst_288_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk___redArg(lean_object* v_a_291_){
_start:
{
lean_inc(v_a_291_);
return v_a_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk___redArg___boxed(lean_object* v_a_292_){
_start:
{
lean_object* v_res_293_; 
v_res_293_ = lp_mathlib_QuotientGroup_mk___redArg(v_a_292_);
lean_dec(v_a_292_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk(lean_object* v_00_u03b1_294_, lean_object* v_inst_295_, lean_object* v_s_296_, lean_object* v_a_297_){
_start:
{
lean_inc(v_a_297_);
return v_a_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_mk___boxed(lean_object* v_00_u03b1_298_, lean_object* v_inst_299_, lean_object* v_s_300_, lean_object* v_a_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_mathlib_QuotientGroup_mk(v_00_u03b1_298_, v_inst_299_, v_s_300_, v_a_301_);
lean_dec(v_a_301_);
lean_dec_ref(v_inst_299_);
return v_res_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk___redArg(lean_object* v_a_303_){
_start:
{
lean_inc(v_a_303_);
return v_a_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk___redArg___boxed(lean_object* v_a_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_QuotientAddGroup_mk___redArg(v_a_304_);
lean_dec(v_a_304_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk(lean_object* v_00_u03b1_306_, lean_object* v_inst_307_, lean_object* v_s_308_, lean_object* v_a_309_){
_start:
{
lean_inc(v_a_309_);
return v_a_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_mk___boxed(lean_object* v_00_u03b1_310_, lean_object* v_inst_311_, lean_object* v_s_312_, lean_object* v_a_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_QuotientAddGroup_mk(v_00_u03b1_310_, v_inst_311_, v_s_312_, v_a_313_);
lean_dec(v_a_313_);
lean_dec_ref(v_inst_311_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instCoeQuotientSubgroup___redArg(lean_object* v_inst_315_, lean_object* v_s_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_317_, 0, lean_box(0));
lean_closure_set(v___x_317_, 1, v_inst_315_);
lean_closure_set(v___x_317_, 2, v_s_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instCoeQuotientSubgroup(lean_object* v_00_u03b1_318_, lean_object* v_inst_319_, lean_object* v_s_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_321_, 0, lean_box(0));
lean_closure_set(v___x_321_, 1, v_inst_319_);
lean_closure_set(v___x_321_, 2, v_s_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instCoeQuotientAddSubgroup___redArg(lean_object* v_inst_322_, lean_object* v_s_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_324_, 0, lean_box(0));
lean_closure_set(v___x_324_, 1, v_inst_322_);
lean_closure_set(v___x_324_, 2, v_s_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instCoeQuotientAddSubgroup(lean_object* v_00_u03b1_325_, lean_object* v_inst_326_, lean_object* v_s_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_328_, 0, lean_box(0));
lean_closure_set(v___x_328_, 1, v_inst_326_);
lean_closure_set(v___x_328_, 2, v_s_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup___redArg(lean_object* v_inst_329_){
_start:
{
lean_object* v___x_330_; lean_object* v_toOne_331_; 
v___x_330_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_329_);
v_toOne_331_ = lean_ctor_get(v___x_330_, 0);
lean_inc(v_toOne_331_);
lean_dec_ref(v___x_330_);
return v_toOne_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup___redArg___boxed(lean_object* v_inst_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup___redArg(v_inst_332_);
lean_dec_ref(v_inst_332_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup(lean_object* v_00_u03b1_334_, lean_object* v_inst_335_, lean_object* v_s_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup___redArg(v_inst_335_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup___boxed(lean_object* v_00_u03b1_338_, lean_object* v_inst_339_, lean_object* v_s_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_QuotientGroup_instInhabitedQuotientSubgroup(v_00_u03b1_338_, v_inst_339_, v_s_340_);
lean_dec_ref(v_inst_339_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup___redArg(lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; lean_object* v_toZero_344_; 
v___x_343_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_342_);
v_toZero_344_ = lean_ctor_get(v___x_343_, 0);
lean_inc(v_toZero_344_);
lean_dec_ref(v___x_343_);
return v_toZero_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup___redArg___boxed(lean_object* v_inst_345_){
_start:
{
lean_object* v_res_346_; 
v_res_346_ = lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup___redArg(v_inst_345_);
lean_dec_ref(v_inst_345_);
return v_res_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup(lean_object* v_00_u03b1_347_, lean_object* v_inst_348_, lean_object* v_s_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup___redArg(v_inst_348_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup___boxed(lean_object* v_00_u03b1_351_, lean_object* v_inst_352_, lean_object* v_s_353_){
_start:
{
lean_object* v_res_354_; 
v_res_354_ = lp_mathlib_QuotientAddGroup_instInhabitedQuotientAddSubgroup(v_00_u03b1_351_, v_inst_352_, v_s_353_);
lean_dec_ref(v_inst_352_);
return v_res_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivOfEq(lean_object* v_00_u03b1_361_, lean_object* v_inst_362_, lean_object* v_s_363_, lean_object* v_t_364_, lean_object* v_h_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = ((lean_object*)(lp_mathlib_Subgroup_quotientEquivOfEq___closed__2));
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_quotientEquivOfEq___boxed(lean_object* v_00_u03b1_367_, lean_object* v_inst_368_, lean_object* v_s_369_, lean_object* v_t_370_, lean_object* v_h_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib_Subgroup_quotientEquivOfEq(v_00_u03b1_367_, v_inst_368_, v_s_369_, v_t_370_, v_h_371_);
lean_dec_ref(v_inst_368_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivOfEq(lean_object* v_00_u03b1_373_, lean_object* v_inst_374_, lean_object* v_s_375_, lean_object* v_t_376_, lean_object* v_h_377_){
_start:
{
lean_object* v___x_378_; 
v___x_378_ = ((lean_object*)(lp_mathlib_Subgroup_quotientEquivOfEq___closed__2));
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_quotientEquivOfEq___boxed(lean_object* v_00_u03b1_379_, lean_object* v_inst_380_, lean_object* v_s_381_, lean_object* v_t_382_, lean_object* v_h_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_mathlib_AddSubgroup_quotientEquivOfEq(v_00_u03b1_379_, v_inst_380_, v_s_381_, v_t_382_, v_h_383_);
lean_dec_ref(v_inst_380_);
return v_res_384_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Quotient(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Quotient(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Coset_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
