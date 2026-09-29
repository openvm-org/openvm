// Lean compiler output
// Module: Mathlib.RingTheory.OreLocalization.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.DistribMulAction public import Mathlib.GroupTheory.OreLocalization.Basic public import Mathlib.Algebra.GroupWithZero.Defs
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_oreDenom___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_oreNum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_zero___redArg(lean_object*, lean_object*);
lean_object* l_nsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_hsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_liftExpand___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_instMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMonoidWithZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommMonoidWithZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAdd___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivAddChar_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivAddChar_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivAddChar_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_nsmul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_nsmul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_nsmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_nsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulAction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulActionOfIsScalarTower___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulActionOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulActionOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddCommMonoidOreLocalization___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddCommMonoidOreLocalization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instNegOreLocalization___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instNegOreLocalization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zsmul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zsmul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zsmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddGroupOreLocalization___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddGroupOreLocalization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMonoidWithZero___redArg(lean_object* v_inst_1_, lean_object* v_S_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v_toMonoid_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v_toZero_8_; lean_object* v___x_10_; uint8_t v_isShared_11_; uint8_t v_isSharedCheck_16_; 
v_toMonoid_4_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref_n(v_toMonoid_4_, 2);
v___x_5_ = lp_mathlib_OreLocalization_instMonoid___redArg(v_toMonoid_4_, v_S_2_, v_inst_3_);
v___x_6_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_1_);
v___x_7_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_6_);
v_toZero_8_ = lean_ctor_get(v___x_7_, 1);
v_isSharedCheck_16_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_16_ == 0)
{
lean_object* v_unused_17_; 
v_unused_17_ = lean_ctor_get(v___x_7_, 0);
lean_dec(v_unused_17_);
v___x_10_ = v___x_7_;
v_isShared_11_ = v_isSharedCheck_16_;
goto v_resetjp_9_;
}
else
{
lean_inc(v_toZero_8_);
lean_dec(v___x_7_);
v___x_10_ = lean_box(0);
v_isShared_11_ = v_isSharedCheck_16_;
goto v_resetjp_9_;
}
v_resetjp_9_:
{
lean_object* v___x_12_; lean_object* v___x_14_; 
v___x_12_ = lp_mathlib_OreLocalization_zero___redArg(v_toMonoid_4_, v_toZero_8_);
lean_dec_ref(v_toMonoid_4_);
if (v_isShared_11_ == 0)
{
lean_ctor_set(v___x_10_, 1, v___x_12_);
lean_ctor_set(v___x_10_, 0, v___x_5_);
v___x_14_ = v___x_10_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v___x_5_);
lean_ctor_set(v_reuseFailAlloc_15_, 1, v___x_12_);
v___x_14_ = v_reuseFailAlloc_15_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
return v___x_14_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instMonoidWithZero(lean_object* v_R_18_, lean_object* v_inst_19_, lean_object* v_S_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_OreLocalization_instMonoidWithZero___redArg(v_inst_19_, v_S_20_, v_inst_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommMonoidWithZero___redArg(lean_object* v_inst_23_, lean_object* v_S_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v_toMonoid_28_; lean_object* v_toZero_29_; lean_object* v___x_31_; uint8_t v_isShared_32_; uint8_t v_isSharedCheck_36_; 
v___x_26_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_23_);
v___x_27_ = lp_mathlib_OreLocalization_instMonoidWithZero___redArg(v___x_26_, v_S_24_, v_inst_25_);
v_toMonoid_28_ = lean_ctor_get(v___x_27_, 0);
v_toZero_29_ = lean_ctor_get(v___x_27_, 1);
v_isSharedCheck_36_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_36_ == 0)
{
v___x_31_ = v___x_27_;
v_isShared_32_ = v_isSharedCheck_36_;
goto v_resetjp_30_;
}
else
{
lean_inc(v_toZero_29_);
lean_inc(v_toMonoid_28_);
lean_dec(v___x_27_);
v___x_31_ = lean_box(0);
v_isShared_32_ = v_isSharedCheck_36_;
goto v_resetjp_30_;
}
v_resetjp_30_:
{
lean_object* v___x_34_; 
if (v_isShared_32_ == 0)
{
v___x_34_ = v___x_31_;
goto v_reusejp_33_;
}
else
{
lean_object* v_reuseFailAlloc_35_; 
v_reuseFailAlloc_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_35_, 0, v_toMonoid_28_);
lean_ctor_set(v_reuseFailAlloc_35_, 1, v_toZero_29_);
v___x_34_ = v_reuseFailAlloc_35_;
goto v_reusejp_33_;
}
v_reusejp_33_:
{
return v___x_34_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommMonoidWithZero(lean_object* v_R_37_, lean_object* v_inst_38_, lean_object* v_S_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_OreLocalization_instCommMonoidWithZero___redArg(v_inst_38_, v_S_39_, v_inst_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___redArg(lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_r_u2081_46_, lean_object* v_s_u2081_47_, lean_object* v_r_u2082_48_, lean_object* v_s_u2082_49_){
_start:
{
lean_object* v_toAdd_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v_toMul_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_66_; 
v_toAdd_50_ = lean_ctor_get(v_inst_44_, 1);
lean_inc(v_toAdd_50_);
lean_dec_ref(v_inst_44_);
v___x_51_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_42_);
v___x_52_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_51_);
v_toMul_53_ = lean_ctor_get(v___x_52_, 1);
v_isSharedCheck_66_ = !lean_is_exclusive(v___x_52_);
if (v_isSharedCheck_66_ == 0)
{
lean_object* v_unused_67_; 
v_unused_67_ = lean_ctor_get(v___x_52_, 0);
lean_dec(v_unused_67_);
v___x_55_ = v___x_52_;
v_isShared_56_ = v_isSharedCheck_66_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_toMul_53_);
lean_dec(v___x_52_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_66_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_64_; 
lean_inc(v_s_u2082_49_);
lean_inc_n(v_s_u2081_47_, 2);
lean_inc_ref(v_inst_43_);
v___x_57_ = lp_mathlib_OreLocalization_oreDenom___redArg(v_inst_43_, v_s_u2081_47_, v_s_u2082_49_);
lean_inc(v_inst_45_);
lean_inc(v___x_57_);
v___x_58_ = lean_apply_2(v_inst_45_, v___x_57_, v_r_u2081_46_);
v___x_59_ = lp_mathlib_OreLocalization_oreNum___redArg(v_inst_43_, v_s_u2081_47_, v_s_u2082_49_);
v___x_60_ = lean_apply_2(v_inst_45_, v___x_59_, v_r_u2082_48_);
v___x_61_ = lean_apply_2(v_toAdd_50_, v___x_58_, v___x_60_);
v___x_62_ = lean_apply_2(v_toMul_53_, v___x_57_, v_s_u2081_47_);
if (v_isShared_56_ == 0)
{
lean_ctor_set(v___x_55_, 1, v___x_62_);
lean_ctor_set(v___x_55_, 0, v___x_61_);
v___x_64_ = v___x_55_;
goto v_reusejp_63_;
}
else
{
lean_object* v_reuseFailAlloc_65_; 
v_reuseFailAlloc_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_65_, 0, v___x_61_);
lean_ctor_set(v_reuseFailAlloc_65_, 1, v___x_62_);
v___x_64_ = v_reuseFailAlloc_65_;
goto v_reusejp_63_;
}
v_reusejp_63_:
{
return v___x_64_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___redArg___boxed(lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_r_u2081_72_, lean_object* v_s_u2081_73_, lean_object* v_r_u2082_74_, lean_object* v_s_u2082_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___redArg(v_inst_68_, v_inst_69_, v_inst_70_, v_inst_71_, v_r_u2081_72_, v_s_u2081_73_, v_r_u2082_74_, v_s_u2082_75_);
lean_dec_ref(v_inst_68_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27(lean_object* v_R_77_, lean_object* v_inst_78_, lean_object* v_S_79_, lean_object* v_inst_80_, lean_object* v_X_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_r_u2081_84_, lean_object* v_s_u2081_85_, lean_object* v_r_u2082_86_, lean_object* v_s_u2082_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___redArg(v_inst_78_, v_inst_80_, v_inst_82_, v_inst_83_, v_r_u2081_84_, v_s_u2081_85_, v_r_u2082_86_, v_s_u2082_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___boxed(lean_object* v_R_89_, lean_object* v_inst_90_, lean_object* v_S_91_, lean_object* v_inst_92_, lean_object* v_X_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_r_u2081_96_, lean_object* v_s_u2081_97_, lean_object* v_r_u2082_98_, lean_object* v_s_u2082_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27(v_R_89_, v_inst_90_, v_S_91_, v_inst_92_, v_X_93_, v_inst_94_, v_inst_95_, v_r_u2081_96_, v_s_u2081_97_, v_r_u2082_98_, v_s_u2082_99_);
lean_dec_ref(v_inst_90_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___redArg(lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_r_u2082_105_, lean_object* v_s_u2082_106_, lean_object* v_a_107_){
_start:
{
lean_object* v_fst_108_; lean_object* v_snd_109_; lean_object* v___x_110_; 
v_fst_108_ = lean_ctor_get(v_a_107_, 0);
lean_inc(v_fst_108_);
v_snd_109_ = lean_ctor_get(v_a_107_, 1);
lean_inc(v_snd_109_);
lean_dec(v_a_107_);
v___x_110_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27_x27___redArg(v_inst_101_, v_inst_102_, v_inst_103_, v_inst_104_, v_fst_108_, v_snd_109_, v_r_u2082_105_, v_s_u2082_106_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___redArg___boxed(lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_r_u2082_115_, lean_object* v_s_u2082_116_, lean_object* v_a_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___redArg(v_inst_111_, v_inst_112_, v_inst_113_, v_inst_114_, v_r_u2082_115_, v_s_u2082_116_, v_a_117_);
lean_dec_ref(v_inst_111_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27(lean_object* v_R_119_, lean_object* v_inst_120_, lean_object* v_S_121_, lean_object* v_inst_122_, lean_object* v_X_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_r_u2082_126_, lean_object* v_s_u2082_127_, lean_object* v_a_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___redArg(v_inst_120_, v_inst_122_, v_inst_124_, v_inst_125_, v_r_u2082_126_, v_s_u2082_127_, v_a_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___boxed(lean_object* v_R_130_, lean_object* v_inst_131_, lean_object* v_S_132_, lean_object* v_inst_133_, lean_object* v_X_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_r_u2082_137_, lean_object* v_s_u2082_138_, lean_object* v_a_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27(v_R_130_, v_inst_131_, v_S_132_, v_inst_133_, v_X_134_, v_inst_135_, v_inst_136_, v_r_u2082_137_, v_s_u2082_138_, v_a_139_);
lean_dec_ref(v_inst_131_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___redArg(lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_x_145_, lean_object* v_a_146_){
_start:
{
lean_object* v_fst_147_; lean_object* v_snd_148_; lean_object* v___x_149_; 
v_fst_147_ = lean_ctor_get(v_a_146_, 0);
lean_inc(v_fst_147_);
v_snd_148_ = lean_ctor_get(v_a_146_, 1);
lean_inc(v_snd_148_);
lean_dec(v_a_146_);
v___x_149_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add_x27___redArg(v_inst_141_, v_inst_142_, v_inst_143_, v_inst_144_, v_fst_147_, v_snd_148_, v_x_145_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___redArg___boxed(lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_x_154_, lean_object* v_a_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___redArg(v_inst_150_, v_inst_151_, v_inst_152_, v_inst_153_, v_x_154_, v_a_155_);
lean_dec_ref(v_inst_150_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add(lean_object* v_R_157_, lean_object* v_inst_158_, lean_object* v_S_159_, lean_object* v_inst_160_, lean_object* v_X_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_x_164_, lean_object* v_a_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___redArg(v_inst_158_, v_inst_160_, v_inst_162_, v_inst_163_, v_x_164_, v_a_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___boxed(lean_object* v_R_167_, lean_object* v_inst_168_, lean_object* v_S_169_, lean_object* v_inst_170_, lean_object* v_X_171_, lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_x_174_, lean_object* v_a_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add(v_R_167_, v_inst_168_, v_S_169_, v_inst_170_, v_X_171_, v_inst_172_, v_inst_173_, v_x_174_, v_a_175_);
lean_dec_ref(v_inst_168_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAdd___redArg(lean_object* v_inst_177_, lean_object* v_S_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___boxed), 9, 7);
lean_closure_set(v___x_182_, 0, lean_box(0));
lean_closure_set(v___x_182_, 1, v_inst_177_);
lean_closure_set(v___x_182_, 2, v_S_178_);
lean_closure_set(v___x_182_, 3, v_inst_179_);
lean_closure_set(v___x_182_, 4, lean_box(0));
lean_closure_set(v___x_182_, 5, v_inst_180_);
lean_closure_set(v___x_182_, 6, v_inst_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAdd(lean_object* v_R_183_, lean_object* v_inst_184_, lean_object* v_S_185_, lean_object* v_inst_186_, lean_object* v_X_187_, lean_object* v_inst_188_, lean_object* v_inst_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___boxed), 9, 7);
lean_closure_set(v___x_190_, 0, lean_box(0));
lean_closure_set(v___x_190_, 1, v_inst_184_);
lean_closure_set(v___x_190_, 2, v_S_185_);
lean_closure_set(v___x_190_, 3, v_inst_186_);
lean_closure_set(v___x_190_, 4, lean_box(0));
lean_closure_set(v___x_190_, 5, v_inst_188_);
lean_closure_set(v___x_190_, 6, v_inst_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivAddChar_x27___redArg(lean_object* v_inst_191_, lean_object* v_s_192_, lean_object* v_s_x27_193_){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
lean_inc(v_s_x27_193_);
lean_inc(v_s_192_);
lean_inc_ref(v_inst_191_);
v___x_194_ = lp_mathlib_OreLocalization_oreNum___redArg(v_inst_191_, v_s_192_, v_s_x27_193_);
v___x_195_ = lp_mathlib_OreLocalization_oreDenom___redArg(v_inst_191_, v_s_192_, v_s_x27_193_);
v___x_196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_195_);
lean_ctor_set(v___x_196_, 1, lean_box(0));
v___x_197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_194_);
lean_ctor_set(v___x_197_, 1, v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivAddChar_x27(lean_object* v_R_198_, lean_object* v_inst_199_, lean_object* v_S_200_, lean_object* v_inst_201_, lean_object* v_X_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_r_205_, lean_object* v_r_x27_206_, lean_object* v_s_207_, lean_object* v_s_x27_208_){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lp_mathlib_OreLocalization_oreDivAddChar_x27___redArg(v_inst_201_, v_s_207_, v_s_x27_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_oreDivAddChar_x27___boxed(lean_object* v_R_210_, lean_object* v_inst_211_, lean_object* v_S_212_, lean_object* v_inst_213_, lean_object* v_X_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_r_217_, lean_object* v_r_x27_218_, lean_object* v_s_219_, lean_object* v_s_x27_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_OreLocalization_oreDivAddChar_x27(v_R_210_, v_inst_211_, v_S_212_, v_inst_213_, v_X_214_, v_inst_215_, v_inst_216_, v_r_217_, v_r_x27_218_, v_s_219_, v_s_x27_220_);
lean_dec(v_r_x27_218_);
lean_dec(v_r_217_);
lean_dec(v_inst_216_);
lean_dec_ref(v_inst_215_);
lean_dec_ref(v_inst_211_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_nsmul___redArg(lean_object* v_inst_222_, lean_object* v_S_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_a_227_, lean_object* v_a_228_){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v_toZero_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_229_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_225_);
v___x_230_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_229_);
v_toZero_231_ = lean_ctor_get(v___x_230_, 0);
lean_inc(v_toZero_231_);
lean_dec_ref(v___x_230_);
v___x_232_ = lp_mathlib_OreLocalization_zero___redArg(v_inst_222_, v_toZero_231_);
v___x_233_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___boxed), 9, 7);
lean_closure_set(v___x_233_, 0, lean_box(0));
lean_closure_set(v___x_233_, 1, v_inst_222_);
lean_closure_set(v___x_233_, 2, v_S_223_);
lean_closure_set(v___x_233_, 3, v_inst_224_);
lean_closure_set(v___x_233_, 4, lean_box(0));
lean_closure_set(v___x_233_, 5, v_inst_225_);
lean_closure_set(v___x_233_, 6, v_inst_226_);
v___x_234_ = l_nsmulRec___redArg(v___x_232_, v___x_233_, v_a_227_, v_a_228_);
lean_dec_ref(v___x_232_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_nsmul___redArg___boxed(lean_object* v_inst_235_, lean_object* v_S_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_a_240_, lean_object* v_a_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_OreLocalization_nsmul___redArg(v_inst_235_, v_S_236_, v_inst_237_, v_inst_238_, v_inst_239_, v_a_240_, v_a_241_);
lean_dec(v_a_240_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_nsmul(lean_object* v_R_243_, lean_object* v_inst_244_, lean_object* v_S_245_, lean_object* v_inst_246_, lean_object* v_X_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_a_250_, lean_object* v_a_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_mathlib_OreLocalization_nsmul___redArg(v_inst_244_, v_S_245_, v_inst_246_, v_inst_248_, v_inst_249_, v_a_250_, v_a_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_nsmul___boxed(lean_object* v_R_253_, lean_object* v_inst_254_, lean_object* v_S_255_, lean_object* v_inst_256_, lean_object* v_X_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_a_260_, lean_object* v_a_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_OreLocalization_nsmul(v_R_253_, v_inst_254_, v_S_255_, v_inst_256_, v_X_257_, v_inst_258_, v_inst_259_, v_a_260_, v_a_261_);
lean_dec(v_a_260_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddMonoid___redArg(lean_object* v_inst_263_, lean_object* v_S_264_, lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_inst_267_){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v_toZero_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_268_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_266_);
v___x_269_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_268_);
v_toZero_270_ = lean_ctor_get(v___x_269_, 0);
lean_inc(v_toZero_270_);
lean_dec_ref(v___x_269_);
v___x_271_ = lp_mathlib_OreLocalization_zero___redArg(v_inst_263_, v_toZero_270_);
lean_inc(v_inst_267_);
lean_inc_ref(v_inst_266_);
lean_inc_ref(v_inst_265_);
lean_inc_ref(v_inst_263_);
v___x_272_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___boxed), 9, 7);
lean_closure_set(v___x_272_, 0, lean_box(0));
lean_closure_set(v___x_272_, 1, v_inst_263_);
lean_closure_set(v___x_272_, 2, v_S_264_);
lean_closure_set(v___x_272_, 3, v_inst_265_);
lean_closure_set(v___x_272_, 4, lean_box(0));
lean_closure_set(v___x_272_, 5, v_inst_266_);
lean_closure_set(v___x_272_, 6, v_inst_267_);
v___x_273_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_nsmul___boxed), 9, 7);
lean_closure_set(v___x_273_, 0, lean_box(0));
lean_closure_set(v___x_273_, 1, v_inst_263_);
lean_closure_set(v___x_273_, 2, v_S_264_);
lean_closure_set(v___x_273_, 3, v_inst_265_);
lean_closure_set(v___x_273_, 4, lean_box(0));
lean_closure_set(v___x_273_, 5, v_inst_266_);
lean_closure_set(v___x_273_, 6, v_inst_267_);
v___x_274_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_274_, 0, v___x_271_);
lean_ctor_set(v___x_274_, 1, v___x_272_);
lean_ctor_set(v___x_274_, 2, v___x_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddMonoid(lean_object* v_R_275_, lean_object* v_inst_276_, lean_object* v_S_277_, lean_object* v_inst_278_, lean_object* v_X_279_, lean_object* v_inst_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lp_mathlib_OreLocalization_instAddMonoid___redArg(v_inst_276_, v_S_277_, v_inst_278_, v_inst_280_, v_inst_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulAction___redArg(lean_object* v_inst_283_, lean_object* v_S_284_, lean_object* v_inst_285_, lean_object* v_inst_286_){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___boxed), 8, 6);
lean_closure_set(v___x_287_, 0, lean_box(0));
lean_closure_set(v___x_287_, 1, v_inst_283_);
lean_closure_set(v___x_287_, 2, v_S_284_);
lean_closure_set(v___x_287_, 3, v_inst_285_);
lean_closure_set(v___x_287_, 4, lean_box(0));
lean_closure_set(v___x_287_, 5, v_inst_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulAction(lean_object* v_R_288_, lean_object* v_inst_289_, lean_object* v_S_290_, lean_object* v_inst_291_, lean_object* v_X_292_, lean_object* v_inst_293_, lean_object* v_inst_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___boxed), 8, 6);
lean_closure_set(v___x_295_, 0, lean_box(0));
lean_closure_set(v___x_295_, 1, v_inst_289_);
lean_closure_set(v___x_295_, 2, v_S_290_);
lean_closure_set(v___x_295_, 3, v_inst_291_);
lean_closure_set(v___x_295_, 4, lean_box(0));
lean_closure_set(v___x_295_, 5, v_inst_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulAction___boxed(lean_object* v_R_296_, lean_object* v_inst_297_, lean_object* v_S_298_, lean_object* v_inst_299_, lean_object* v_X_300_, lean_object* v_inst_301_, lean_object* v_inst_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_OreLocalization_instDistribMulAction(v_R_296_, v_inst_297_, v_S_298_, v_inst_299_, v_X_300_, v_inst_301_, v_inst_302_);
lean_dec_ref(v_inst_301_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulActionOfIsScalarTower___redArg(lean_object* v_inst_304_, lean_object* v_S_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_inst_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_hsmul___boxed), 10, 8);
lean_closure_set(v___x_309_, 0, lean_box(0));
lean_closure_set(v___x_309_, 1, lean_box(0));
lean_closure_set(v___x_309_, 2, lean_box(0));
lean_closure_set(v___x_309_, 3, v_inst_304_);
lean_closure_set(v___x_309_, 4, v_S_305_);
lean_closure_set(v___x_309_, 5, v_inst_306_);
lean_closure_set(v___x_309_, 6, v_inst_307_);
lean_closure_set(v___x_309_, 7, v_inst_308_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulActionOfIsScalarTower(lean_object* v_R_310_, lean_object* v_inst_311_, lean_object* v_S_312_, lean_object* v_inst_313_, lean_object* v_X_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_R_u2080_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_hsmul___boxed), 10, 8);
lean_closure_set(v___x_323_, 0, lean_box(0));
lean_closure_set(v___x_323_, 1, lean_box(0));
lean_closure_set(v___x_323_, 2, lean_box(0));
lean_closure_set(v___x_323_, 3, v_inst_311_);
lean_closure_set(v___x_323_, 4, v_S_312_);
lean_closure_set(v___x_323_, 5, v_inst_313_);
lean_closure_set(v___x_323_, 6, v_inst_316_);
lean_closure_set(v___x_323_, 7, v_inst_320_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instDistribMulActionOfIsScalarTower___boxed(lean_object* v_R_324_, lean_object* v_inst_325_, lean_object* v_S_326_, lean_object* v_inst_327_, lean_object* v_X_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_R_u2080_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_OreLocalization_instDistribMulActionOfIsScalarTower(v_R_324_, v_inst_325_, v_S_326_, v_inst_327_, v_X_328_, v_inst_329_, v_inst_330_, v_R_u2080_331_, v_inst_332_, v_inst_333_, v_inst_334_, v_inst_335_, v_inst_336_);
lean_dec(v_inst_333_);
lean_dec_ref(v_inst_332_);
lean_dec_ref(v_inst_329_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddCommMonoidOreLocalization___redArg(lean_object* v_inst_338_, lean_object* v_S_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lp_mathlib_OreLocalization_instAddMonoid___redArg(v_inst_338_, v_S_339_, v_inst_340_, v_inst_341_, v_inst_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddCommMonoidOreLocalization(lean_object* v_R_344_, lean_object* v_inst_345_, lean_object* v_S_346_, lean_object* v_inst_347_, lean_object* v_X_348_, lean_object* v_inst_349_, lean_object* v_inst_350_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lp_mathlib_OreLocalization_instAddMonoid___redArg(v_inst_345_, v_S_346_, v_inst_347_, v_inst_349_, v_inst_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg___redArg___lam__0(lean_object* v_toNeg_352_, lean_object* v_r_353_, lean_object* v_s_354_){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_355_ = lean_apply_1(v_toNeg_352_, v_r_353_);
v___x_356_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_355_);
lean_ctor_set(v___x_356_, 1, v_s_354_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg___redArg(lean_object* v_inst_357_, lean_object* v_a_358_){
_start:
{
lean_object* v___x_359_; lean_object* v_toNeg_360_; lean_object* v___f_361_; lean_object* v___x_362_; 
v___x_359_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_357_);
v_toNeg_360_ = lean_ctor_get(v___x_359_, 1);
lean_inc(v_toNeg_360_);
lean_dec_ref(v___x_359_);
v___f_361_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_neg___redArg___lam__0), 3, 1);
lean_closure_set(v___f_361_, 0, v_toNeg_360_);
v___x_362_ = lp_mathlib_OreLocalization_liftExpand___redArg(v___f_361_, v_a_358_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg___redArg___boxed(lean_object* v_inst_363_, lean_object* v_a_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_OreLocalization_neg___redArg(v_inst_363_, v_a_364_);
lean_dec_ref(v_inst_363_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg(lean_object* v_R_366_, lean_object* v_inst_367_, lean_object* v_S_368_, lean_object* v_inst_369_, lean_object* v_X_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_a_373_){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lp_mathlib_OreLocalization_neg___redArg(v_inst_371_, v_a_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_neg___boxed(lean_object* v_R_375_, lean_object* v_inst_376_, lean_object* v_S_377_, lean_object* v_inst_378_, lean_object* v_X_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_a_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_OreLocalization_neg(v_R_375_, v_inst_376_, v_S_377_, v_inst_378_, v_X_379_, v_inst_380_, v_inst_381_, v_a_382_);
lean_dec(v_inst_381_);
lean_dec_ref(v_inst_380_);
lean_dec_ref(v_inst_378_);
lean_dec_ref(v_inst_376_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instNegOreLocalization___redArg(lean_object* v_inst_384_, lean_object* v_S_385_, lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_neg___boxed), 8, 7);
lean_closure_set(v___x_389_, 0, lean_box(0));
lean_closure_set(v___x_389_, 1, v_inst_384_);
lean_closure_set(v___x_389_, 2, v_S_385_);
lean_closure_set(v___x_389_, 3, v_inst_386_);
lean_closure_set(v___x_389_, 4, lean_box(0));
lean_closure_set(v___x_389_, 5, v_inst_387_);
lean_closure_set(v___x_389_, 6, v_inst_388_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instNegOreLocalization(lean_object* v_R_390_, lean_object* v_inst_391_, lean_object* v_S_392_, lean_object* v_inst_393_, lean_object* v_X_394_, lean_object* v_inst_395_, lean_object* v_inst_396_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_neg___boxed), 8, 7);
lean_closure_set(v___x_397_, 0, lean_box(0));
lean_closure_set(v___x_397_, 1, v_inst_391_);
lean_closure_set(v___x_397_, 2, v_S_392_);
lean_closure_set(v___x_397_, 3, v_inst_393_);
lean_closure_set(v___x_397_, 4, lean_box(0));
lean_closure_set(v___x_397_, 5, v_inst_395_);
lean_closure_set(v___x_397_, 6, v_inst_396_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zsmul___redArg(lean_object* v_inst_398_, lean_object* v_S_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_a_403_, lean_object* v_a_404_){
_start:
{
lean_object* v___x_405_; lean_object* v_toZero_406_; lean_object* v_toAddMonoid_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; 
v___x_405_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_401_);
v_toZero_406_ = lean_ctor_get(v___x_405_, 0);
lean_inc(v_toZero_406_);
lean_dec_ref(v___x_405_);
v_toAddMonoid_407_ = lean_ctor_get(v_inst_401_, 0);
v___x_408_ = lp_mathlib_OreLocalization_zero___redArg(v_inst_398_, v_toZero_406_);
lean_inc(v_inst_402_);
lean_inc_ref(v_toAddMonoid_407_);
lean_inc_ref(v_inst_400_);
lean_inc_ref(v_inst_398_);
v___x_409_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_RingTheory_OreLocalization_Basic_0__OreLocalization_add___boxed), 9, 7);
lean_closure_set(v___x_409_, 0, lean_box(0));
lean_closure_set(v___x_409_, 1, v_inst_398_);
lean_closure_set(v___x_409_, 2, v_S_399_);
lean_closure_set(v___x_409_, 3, v_inst_400_);
lean_closure_set(v___x_409_, 4, lean_box(0));
lean_closure_set(v___x_409_, 5, v_toAddMonoid_407_);
lean_closure_set(v___x_409_, 6, v_inst_402_);
v___x_410_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_neg___boxed), 8, 7);
lean_closure_set(v___x_410_, 0, lean_box(0));
lean_closure_set(v___x_410_, 1, v_inst_398_);
lean_closure_set(v___x_410_, 2, v_S_399_);
lean_closure_set(v___x_410_, 3, v_inst_400_);
lean_closure_set(v___x_410_, 4, lean_box(0));
lean_closure_set(v___x_410_, 5, v_inst_401_);
lean_closure_set(v___x_410_, 6, v_inst_402_);
v___x_411_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_411_, 0, lean_box(0));
lean_closure_set(v___x_411_, 1, v___x_408_);
lean_closure_set(v___x_411_, 2, v___x_409_);
v___x_412_ = lp_mathlib_zsmulRec___redArg(v___x_410_, v___x_411_, v_a_403_, v_a_404_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zsmul___redArg___boxed(lean_object* v_inst_413_, lean_object* v_S_414_, lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_a_418_, lean_object* v_a_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_OreLocalization_zsmul___redArg(v_inst_413_, v_S_414_, v_inst_415_, v_inst_416_, v_inst_417_, v_a_418_, v_a_419_);
lean_dec(v_a_418_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zsmul(lean_object* v_R_421_, lean_object* v_inst_422_, lean_object* v_S_423_, lean_object* v_inst_424_, lean_object* v_X_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_a_428_, lean_object* v_a_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_OreLocalization_zsmul___redArg(v_inst_422_, v_S_423_, v_inst_424_, v_inst_426_, v_inst_427_, v_a_428_, v_a_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_zsmul___boxed(lean_object* v_R_431_, lean_object* v_inst_432_, lean_object* v_S_433_, lean_object* v_inst_434_, lean_object* v_X_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_a_438_, lean_object* v_a_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_mathlib_OreLocalization_zsmul(v_R_431_, v_inst_432_, v_S_433_, v_inst_434_, v_X_435_, v_inst_436_, v_inst_437_, v_a_438_, v_a_439_);
lean_dec(v_a_438_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddGroupOreLocalization___redArg(lean_object* v_inst_441_, lean_object* v_S_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_inst_445_){
_start:
{
lean_object* v_toAddMonoid_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v_toAddMonoid_446_ = lean_ctor_get(v_inst_444_, 0);
lean_inc_n(v_inst_445_, 2);
lean_inc_ref(v_toAddMonoid_446_);
lean_inc_ref_n(v_inst_443_, 2);
lean_inc_ref_n(v_inst_441_, 2);
v___x_447_ = lp_mathlib_OreLocalization_instAddMonoid___redArg(v_inst_441_, v_S_442_, v_inst_443_, v_toAddMonoid_446_, v_inst_445_);
lean_inc_ref(v_inst_444_);
v___x_448_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_neg___boxed), 8, 7);
lean_closure_set(v___x_448_, 0, lean_box(0));
lean_closure_set(v___x_448_, 1, v_inst_441_);
lean_closure_set(v___x_448_, 2, v_S_442_);
lean_closure_set(v___x_448_, 3, v_inst_443_);
lean_closure_set(v___x_448_, 4, lean_box(0));
lean_closure_set(v___x_448_, 5, v_inst_444_);
lean_closure_set(v___x_448_, 6, v_inst_445_);
lean_inc_ref(v___x_448_);
lean_inc_ref(v___x_447_);
v___x_449_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_449_, 0, lean_box(0));
lean_closure_set(v___x_449_, 1, v___x_447_);
lean_closure_set(v___x_449_, 2, v___x_448_);
v___x_450_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_zsmul___boxed), 9, 7);
lean_closure_set(v___x_450_, 0, lean_box(0));
lean_closure_set(v___x_450_, 1, v_inst_441_);
lean_closure_set(v___x_450_, 2, v_S_442_);
lean_closure_set(v___x_450_, 3, v_inst_443_);
lean_closure_set(v___x_450_, 4, lean_box(0));
lean_closure_set(v___x_450_, 5, v_inst_444_);
lean_closure_set(v___x_450_, 6, v_inst_445_);
v___x_451_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_451_, 0, v___x_447_);
lean_ctor_set(v___x_451_, 1, v___x_448_);
lean_ctor_set(v___x_451_, 2, v___x_449_);
lean_ctor_set(v___x_451_, 3, v___x_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddGroupOreLocalization(lean_object* v_R_452_, lean_object* v_inst_453_, lean_object* v_S_454_, lean_object* v_inst_455_, lean_object* v_X_456_, lean_object* v_inst_457_, lean_object* v_inst_458_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = lp_mathlib_OreLocalization_instAddGroupOreLocalization___redArg(v_inst_453_, v_S_454_, v_inst_455_, v_inst_457_, v_inst_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddCommGroup___redArg(lean_object* v_inst_460_, lean_object* v_S_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = lp_mathlib_OreLocalization_instAddGroupOreLocalization___redArg(v_inst_460_, v_S_461_, v_inst_462_, v_inst_463_, v_inst_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAddCommGroup(lean_object* v_R_466_, lean_object* v_inst_467_, lean_object* v_S_468_, lean_object* v_inst_469_, lean_object* v_X_470_, lean_object* v_inst_471_, lean_object* v_inst_472_){
_start:
{
lean_object* v___x_473_; 
v___x_473_ = lp_mathlib_OreLocalization_instAddGroupOreLocalization___redArg(v_inst_467_, v_S_468_, v_inst_469_, v_inst_471_, v_inst_472_);
return v___x_473_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_DistribMulAction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_DistribMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_DistribMulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_DistribMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
