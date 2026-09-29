// Lean compiler output
// Module: Mathlib.Algebra.Ring.Subring.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Subsemiring.Defs public import Mathlib.RingTheory.NonUnitalSubring.Defs
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
lean_object* lp_mathlib_SubsemiringClass_toSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_SubsemiringClass_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_InvMemClass_inv___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_sub___redArg(lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toHasIntCast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toHasIntCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toHasIntCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toHasIntCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocCommRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toCommRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_SubringClass_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubringClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubringClass_subtype___closed__0 = (const lean_object*)&lp_mathlib_SubringClass_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toAddSubgroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toAddSubgroup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSetLike___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subring_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subring_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subring_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toNonUnitalSubring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toNonUnitalSubring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_toSubring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_toSubring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_toSubring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_toCommRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_subtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_subtype___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_toSubring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_toSubring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toHasIntCast___redArg___lam__0(lean_object* v_toIntCast_1_, lean_object* v_n_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_toIntCast_1_, v_n_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toHasIntCast___redArg(lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v_toIntCast_7_; lean_object* v___f_8_; 
v___x_5_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_4_);
v___x_6_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_5_);
lean_dec_ref(v___x_5_);
v_toIntCast_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc(v_toIntCast_7_);
lean_dec_ref(v___x_6_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_SubringClass_toHasIntCast___redArg___lam__0), 2, 1);
lean_closure_set(v___f_8_, 0, v_toIntCast_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toHasIntCast(lean_object* v_R_9_, lean_object* v_S_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_hSR_13_, lean_object* v_s_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_SubringClass_toHasIntCast___redArg(v_inst_11_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toHasIntCast___boxed(lean_object* v_R_16_, lean_object* v_S_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_hSR_20_, lean_object* v_s_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_SubringClass_toHasIntCast(v_R_16_, v_S_17_, v_inst_18_, v_inst_19_, v_hSR_20_, v_s_21_);
lean_dec(v_s_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocRing___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v_toAddMonoidWithOne_28_; lean_object* v_toOne_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v_toNatCast_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v_toNonUnitalNonAssocRing_24_ = lean_ctor_get(v_inst_23_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_24_);
v___x_25_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_toNonUnitalNonAssocRing_24_);
lean_inc_ref_n(v_inst_23_, 2);
v___x_26_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_23_);
v___x_27_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_26_);
lean_dec_ref(v___x_26_);
v_toAddMonoidWithOne_28_ = lean_ctor_get(v___x_27_, 1);
lean_inc_ref(v_toAddMonoidWithOne_28_);
lean_dec_ref(v___x_27_);
v_toOne_29_ = lean_ctor_get(v_toAddMonoidWithOne_28_, 2);
lean_inc(v_toOne_29_);
lean_dec_ref(v_toAddMonoidWithOne_28_);
v___x_30_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_23_);
v___x_31_ = lp_mathlib_SubsemiringClass_toNonAssocSemiring___redArg(v___x_30_);
v___x_32_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_31_);
v_toNatCast_33_ = lean_ctor_get(v___x_32_, 0);
lean_inc(v_toNatCast_33_);
lean_dec_ref(v___x_32_);
v___x_34_ = lp_mathlib_SubringClass_toHasIntCast___redArg(v_inst_23_);
v___x_35_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_35_, 0, v___x_25_);
lean_ctor_set(v___x_35_, 1, v_toOne_29_);
lean_ctor_set(v___x_35_, 2, v_toNatCast_33_);
lean_ctor_set(v___x_35_, 3, v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocRing(lean_object* v_R_36_, lean_object* v_S_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_hSR_40_, lean_object* v_s_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_SubringClass_toNonAssocRing___redArg(v_inst_38_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocRing___boxed(lean_object* v_R_43_, lean_object* v_S_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_hSR_47_, lean_object* v_s_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_SubringClass_toNonAssocRing(v_R_43_, v_S_44_, v_inst_45_, v_inst_46_, v_hSR_47_, v_s_48_);
lean_dec(v_s_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toRing___redArg(lean_object* v_inst_50_){
_start:
{
lean_object* v_toSemiring_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v_toNeg_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_60_; uint8_t v_isShared_61_; uint8_t v_isSharedCheck_73_; 
v_toSemiring_51_ = lean_ctor_get(v_inst_50_, 0);
lean_inc_ref(v_toSemiring_51_);
v___x_52_ = lp_mathlib_SubsemiringClass_toSemiring___redArg(v_toSemiring_51_);
v___x_53_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_50_);
v___x_54_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_53_);
lean_dec_ref(v___x_53_);
v_toNeg_55_ = lean_ctor_get(v___x_54_, 1);
lean_inc(v_toNeg_55_);
lean_dec_ref(v___x_54_);
lean_inc_ref(v_inst_50_);
v___x_56_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_50_);
v___x_57_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_56_);
lean_dec_ref(v___x_56_);
v___x_58_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_50_);
v_isSharedCheck_73_ = !lean_is_exclusive(v_inst_50_);
if (v_isSharedCheck_73_ == 0)
{
lean_object* v_unused_74_; lean_object* v_unused_75_; lean_object* v_unused_76_; lean_object* v_unused_77_; lean_object* v_unused_78_; 
v_unused_74_ = lean_ctor_get(v_inst_50_, 4);
lean_dec(v_unused_74_);
v_unused_75_ = lean_ctor_get(v_inst_50_, 3);
lean_dec(v_unused_75_);
v_unused_76_ = lean_ctor_get(v_inst_50_, 2);
lean_dec(v_unused_76_);
v_unused_77_ = lean_ctor_get(v_inst_50_, 1);
lean_dec(v_unused_77_);
v_unused_78_ = lean_ctor_get(v_inst_50_, 0);
lean_dec(v_unused_78_);
v___x_60_ = v_inst_50_;
v_isShared_61_ = v_isSharedCheck_73_;
goto v_resetjp_59_;
}
else
{
lean_dec(v_inst_50_);
v___x_60_ = lean_box(0);
v_isShared_61_ = v_isSharedCheck_73_;
goto v_resetjp_59_;
}
v_resetjp_59_:
{
lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v_toZSMul_66_; lean_object* v_toIntCast_67_; lean_object* v___f_68_; lean_object* v___x_69_; lean_object* v___x_71_; 
v___x_62_ = lp_mathlib_SubringClass_toNonAssocRing___redArg(v___x_58_);
v___x_63_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v___x_62_);
v___x_64_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_63_);
lean_dec_ref(v___x_63_);
v___x_65_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_64_);
v_toZSMul_66_ = lean_ctor_get(v___x_65_, 3);
lean_inc(v_toZSMul_66_);
lean_dec_ref(v___x_65_);
v_toIntCast_67_ = lean_ctor_get(v___x_64_, 0);
lean_inc(v_toIntCast_67_);
lean_dec_ref(v___x_64_);
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_68_, 0, v_toNeg_55_);
v___x_69_ = lp_mathlib_AddSubgroupClass_sub___redArg(v___x_57_);
if (v_isShared_61_ == 0)
{
lean_ctor_set(v___x_60_, 4, v_toIntCast_67_);
lean_ctor_set(v___x_60_, 3, v_toZSMul_66_);
lean_ctor_set(v___x_60_, 2, v___x_69_);
lean_ctor_set(v___x_60_, 1, v___f_68_);
lean_ctor_set(v___x_60_, 0, v___x_52_);
v___x_71_ = v___x_60_;
goto v_reusejp_70_;
}
else
{
lean_object* v_reuseFailAlloc_72_; 
v_reuseFailAlloc_72_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_72_, 0, v___x_52_);
lean_ctor_set(v_reuseFailAlloc_72_, 1, v___f_68_);
lean_ctor_set(v_reuseFailAlloc_72_, 2, v___x_69_);
lean_ctor_set(v_reuseFailAlloc_72_, 3, v_toZSMul_66_);
lean_ctor_set(v_reuseFailAlloc_72_, 4, v_toIntCast_67_);
v___x_71_ = v_reuseFailAlloc_72_;
goto v_reusejp_70_;
}
v_reusejp_70_:
{
return v___x_71_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toRing(lean_object* v_S_79_, lean_object* v_s_80_, lean_object* v_R_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_82_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toRing___boxed(lean_object* v_S_86_, lean_object* v_s_87_, lean_object* v_R_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_SubringClass_toRing(v_S_86_, v_s_87_, v_R_88_, v_inst_89_, v_inst_90_, v_inst_91_);
lean_dec(v_s_87_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocCommRing___redArg(lean_object* v_inst_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib_SubringClass_toNonAssocRing___redArg(v_inst_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocCommRing(lean_object* v_S_95_, lean_object* v_s_96_, lean_object* v_R_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_SubringClass_toNonAssocRing___redArg(v_inst_98_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toNonAssocCommRing___boxed(lean_object* v_S_102_, lean_object* v_s_103_, lean_object* v_R_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_SubringClass_toNonAssocCommRing(v_S_102_, v_s_103_, v_R_104_, v_inst_105_, v_inst_106_, v_inst_107_);
lean_dec(v_s_103_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toCommRing___redArg(lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toCommRing(lean_object* v_S_111_, lean_object* v_s_112_, lean_object* v_R_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_114_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_toCommRing___boxed(lean_object* v_S_118_, lean_object* v_s_119_, lean_object* v_R_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_SubringClass_toCommRing(v_S_118_, v_s_119_, v_R_120_, v_inst_121_, v_inst_122_, v_inst_123_);
lean_dec(v_s_119_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_subtype___lam__0(lean_object* v_self_125_){
_start:
{
lean_inc(v_self_125_);
return v_self_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_subtype___lam__0___boxed(lean_object* v_self_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_SubringClass_subtype___lam__0(v_self_126_);
lean_dec(v_self_126_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_subtype(lean_object* v_R_129_, lean_object* v_S_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_hSR_133_, lean_object* v_s_134_){
_start:
{
lean_object* v___f_135_; 
v___f_135_ = ((lean_object*)(lp_mathlib_SubringClass_subtype___closed__0));
return v___f_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubringClass_subtype___boxed(lean_object* v_R_136_, lean_object* v_S_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_hSR_140_, lean_object* v_s_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_SubringClass_subtype(v_R_136_, v_S_137_, v_inst_138_, v_inst_139_, v_hSR_140_, v_s_141_);
lean_dec(v_s_141_);
lean_dec_ref(v_inst_138_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toAddSubgroup(lean_object* v_R_143_, lean_object* v_inst_144_, lean_object* v_self_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lean_box(0);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toAddSubgroup___boxed(lean_object* v_R_147_, lean_object* v_inst_148_, lean_object* v_self_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Subring_toAddSubgroup(v_R_147_, v_inst_148_, v_self_149_);
lean_dec_ref(v_inst_148_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSetLike(lean_object* v_R_151_, lean_object* v_inst_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lean_box(0);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instSetLike___boxed(lean_object* v_R_154_, lean_object* v_inst_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_Subring_instSetLike(v_R_154_, v_inst_155_);
lean_dec_ref(v_inst_155_);
return v_res_156_;
}
}
static lean_object* _init_lp_mathlib_Subring_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = lean_box(0);
v___x_158_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instPartialOrder(lean_object* v_R_159_, lean_object* v_inst_160_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = lean_obj_once(&lp_mathlib_Subring_instPartialOrder___closed__0, &lp_mathlib_Subring_instPartialOrder___closed__0_once, _init_lp_mathlib_Subring_instPartialOrder___closed__0);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_instPartialOrder___boxed(lean_object* v_R_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Subring_instPartialOrder(v_R_162_, v_inst_163_);
lean_dec_ref(v_inst_163_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_ofClass(lean_object* v_S_165_, lean_object* v_R_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_s_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_box(0);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_ofClass___boxed(lean_object* v_S_172_, lean_object* v_R_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_s_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_Subring_ofClass(v_S_172_, v_R_173_, v_inst_174_, v_inst_175_, v_inst_176_, v_s_177_);
lean_dec(v_s_177_);
lean_dec_ref(v_inst_174_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toNonUnitalSubring(lean_object* v_R_179_, lean_object* v_inst_180_, lean_object* v_S_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lean_box(0);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toNonUnitalSubring___boxed(lean_object* v_R_183_, lean_object* v_inst_184_, lean_object* v_S_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Subring_toNonUnitalSubring(v_R_183_, v_inst_184_, v_S_185_);
lean_dec_ref(v_inst_184_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_copy(lean_object* v_R_187_, lean_object* v_inst_188_, lean_object* v_S_189_, lean_object* v_s_190_, lean_object* v_hs_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lean_box(0);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_copy___boxed(lean_object* v_R_193_, lean_object* v_inst_194_, lean_object* v_S_195_, lean_object* v_s_196_, lean_object* v_hs_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_Subring_copy(v_R_193_, v_inst_194_, v_S_195_, v_s_196_, v_hs_197_);
lean_dec_ref(v_inst_194_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_mk_x27(lean_object* v_R_199_, lean_object* v_inst_200_, lean_object* v_s_201_, lean_object* v_sm_202_, lean_object* v_sa_203_, lean_object* v_hm_204_, lean_object* v_ha_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lean_box(0);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_mk_x27___boxed(lean_object* v_R_207_, lean_object* v_inst_208_, lean_object* v_s_209_, lean_object* v_sm_210_, lean_object* v_sa_211_, lean_object* v_hm_212_, lean_object* v_ha_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib_Subring_mk_x27(v_R_207_, v_inst_208_, v_s_209_, v_sm_210_, v_sa_211_, v_hm_212_, v_ha_213_);
lean_dec_ref(v_inst_208_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_toSubring___redArg(lean_object* v_s_215_){
_start:
{
return v_s_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_toSubring(lean_object* v_R_216_, lean_object* v_inst_217_, lean_object* v_s_218_, lean_object* v_hneg_219_){
_start:
{
return v_s_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_toSubring___boxed(lean_object* v_R_220_, lean_object* v_inst_221_, lean_object* v_s_222_, lean_object* v_hneg_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_mathlib_Subsemiring_toSubring(v_R_220_, v_inst_221_, v_s_222_, v_hneg_223_);
lean_dec_ref(v_inst_221_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toRing___redArg(lean_object* v_inst_225_){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toRing(lean_object* v_R_227_, lean_object* v_inst_228_, lean_object* v_s_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_228_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toCommRing___redArg(lean_object* v_inst_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_toCommRing(lean_object* v_R_233_, lean_object* v_inst_234_, lean_object* v_s_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lp_mathlib_SubringClass_toRing___redArg(v_inst_234_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_subtype(lean_object* v_R_237_, lean_object* v_inst_238_, lean_object* v_s_239_){
_start:
{
lean_object* v___f_240_; 
v___f_240_ = ((lean_object*)(lp_mathlib_SubringClass_subtype___closed__0));
return v___f_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_subtype___boxed(lean_object* v_R_241_, lean_object* v_inst_242_, lean_object* v_s_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_Subring_subtype(v_R_241_, v_inst_242_, v_s_243_);
lean_dec_ref(v_inst_242_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_toSubring(lean_object* v_R_245_, lean_object* v_inst_246_, lean_object* v_S_247_, lean_object* v_h1_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lean_box(0);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_toSubring___boxed(lean_object* v_R_250_, lean_object* v_inst_251_, lean_object* v_S_252_, lean_object* v_h1_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_NonUnitalSubring_toSubring(v_R_250_, v_inst_251_, v_S_252_, v_h1_253_);
lean_dec_ref(v_inst_251_);
return v_res_254_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Subring_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
