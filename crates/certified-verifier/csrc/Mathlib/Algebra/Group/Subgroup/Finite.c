// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.Finite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Basic public import Mathlib.Algebra.Group.Submonoid.BigOperators public import Mathlib.Algebra.Group.Submonoid.Finite public import Mathlib.Data.Set.Finite.Range public import Mathlib.SetTheory.Cardinal.NatCard
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
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
uint8_t lp_mathlib_Finset_decidableExistsAndFinset___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instFintypeSubtypeMemOfDecidablePred___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instFintypeSubtypeMemOfDecidablePred(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instFintypeSubtypeMemOfDecidablePred___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instFintypeSubtypeMemOfDecidablePred___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instFintypeSubtypeMemOfDecidablePred(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instFintypeSubtypeMemOfDecidablePred___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_fintypeBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_fintypeBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_fintypeBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_fintypeBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_fintypeBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_fintypeBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_fintypeBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_fintypeBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemRange___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemRange___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemRange___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemRange___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemRange___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemRange___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeMrange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeMrange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeMrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeMrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeMrange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeMrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeMrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instFintypeSubtypeMemOfDecidablePred___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v_this_3_; 
v_this_3_ = lp_mathlib_Subtype_fintype___redArg(v_inst_1_, v_inst_2_);
return v_this_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instFintypeSubtypeMemOfDecidablePred(lean_object* v_G_4_, lean_object* v_inst_5_, lean_object* v_K_6_, lean_object* v_inst_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v_this_9_; 
v_this_9_ = lp_mathlib_Subtype_fintype___redArg(v_inst_7_, v_inst_8_);
return v_this_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instFintypeSubtypeMemOfDecidablePred___boxed(lean_object* v_G_10_, lean_object* v_inst_11_, lean_object* v_K_12_, lean_object* v_inst_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_Subgroup_instFintypeSubtypeMemOfDecidablePred(v_G_10_, v_inst_11_, v_K_12_, v_inst_13_, v_inst_14_);
lean_dec_ref(v_inst_11_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instFintypeSubtypeMemOfDecidablePred___redArg(lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v_this_18_; 
v_this_18_ = lp_mathlib_Subtype_fintype___redArg(v_inst_16_, v_inst_17_);
return v_this_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instFintypeSubtypeMemOfDecidablePred(lean_object* v_G_19_, lean_object* v_inst_20_, lean_object* v_K_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_this_24_; 
v_this_24_ = lp_mathlib_Subtype_fintype___redArg(v_inst_22_, v_inst_23_);
return v_this_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instFintypeSubtypeMemOfDecidablePred___boxed(lean_object* v_G_25_, lean_object* v_inst_26_, lean_object* v_K_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_AddSubgroup_instFintypeSubtypeMemOfDecidablePred(v_G_25_, v_inst_26_, v_K_27_, v_inst_28_, v_inst_29_);
lean_dec_ref(v_inst_26_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_fintypeBot___redArg(lean_object* v_inst_31_){
_start:
{
lean_object* v_toMonoid_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v_toOne_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_43_; 
v_toMonoid_32_ = lean_ctor_get(v_inst_31_, 0);
v___x_33_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_32_);
v___x_34_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_33_);
v_toOne_35_ = lean_ctor_get(v___x_34_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_34_);
if (v_isSharedCheck_43_ == 0)
{
lean_object* v_unused_44_; 
v_unused_44_ = lean_ctor_get(v___x_34_, 1);
lean_dec(v_unused_44_);
v___x_37_ = v___x_34_;
v_isShared_38_ = v_isSharedCheck_43_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_toOne_35_);
lean_dec(v___x_34_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_43_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_39_; lean_object* v___x_41_; 
v___x_39_ = lean_box(0);
if (v_isShared_38_ == 0)
{
lean_ctor_set_tag(v___x_37_, 1);
lean_ctor_set(v___x_37_, 1, v___x_39_);
v___x_41_ = v___x_37_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_toOne_35_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v___x_39_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_fintypeBot___redArg___boxed(lean_object* v_inst_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Subgroup_fintypeBot___redArg(v_inst_45_);
lean_dec_ref(v_inst_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_fintypeBot(lean_object* v_G_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Subgroup_fintypeBot___redArg(v_inst_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_fintypeBot___boxed(lean_object* v_G_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Subgroup_fintypeBot(v_G_50_, v_inst_51_);
lean_dec_ref(v_inst_51_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_fintypeBot___redArg(lean_object* v_inst_53_){
_start:
{
lean_object* v_toAddMonoid_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v_toZero_57_; lean_object* v___x_59_; uint8_t v_isShared_60_; uint8_t v_isSharedCheck_65_; 
v_toAddMonoid_54_ = lean_ctor_get(v_inst_53_, 0);
v___x_55_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_54_);
v___x_56_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_55_);
v_toZero_57_ = lean_ctor_get(v___x_56_, 0);
v_isSharedCheck_65_ = !lean_is_exclusive(v___x_56_);
if (v_isSharedCheck_65_ == 0)
{
lean_object* v_unused_66_; 
v_unused_66_ = lean_ctor_get(v___x_56_, 1);
lean_dec(v_unused_66_);
v___x_59_ = v___x_56_;
v_isShared_60_ = v_isSharedCheck_65_;
goto v_resetjp_58_;
}
else
{
lean_inc(v_toZero_57_);
lean_dec(v___x_56_);
v___x_59_ = lean_box(0);
v_isShared_60_ = v_isSharedCheck_65_;
goto v_resetjp_58_;
}
v_resetjp_58_:
{
lean_object* v___x_61_; lean_object* v___x_63_; 
v___x_61_ = lean_box(0);
if (v_isShared_60_ == 0)
{
lean_ctor_set_tag(v___x_59_, 1);
lean_ctor_set(v___x_59_, 1, v___x_61_);
v___x_63_ = v___x_59_;
goto v_reusejp_62_;
}
else
{
lean_object* v_reuseFailAlloc_64_; 
v_reuseFailAlloc_64_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_64_, 0, v_toZero_57_);
lean_ctor_set(v_reuseFailAlloc_64_, 1, v___x_61_);
v___x_63_ = v_reuseFailAlloc_64_;
goto v_reusejp_62_;
}
v_reusejp_62_:
{
return v___x_63_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_fintypeBot___redArg___boxed(lean_object* v_inst_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_AddSubgroup_fintypeBot___redArg(v_inst_67_);
lean_dec_ref(v_inst_67_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_fintypeBot(lean_object* v_G_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_AddSubgroup_fintypeBot___redArg(v_inst_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_fintypeBot___boxed(lean_object* v_G_72_, lean_object* v_inst_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_AddSubgroup_fintypeBot(v_G_72_, v_inst_73_);
lean_dec_ref(v_inst_73_);
return v_res_74_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemRange___redArg___lam__0(lean_object* v_f_75_, lean_object* v_inst_76_, lean_object* v_x_77_, lean_object* v_a_78_){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; uint8_t v___x_81_; 
v___x_79_ = lean_apply_1(v_f_75_, v_a_78_);
v___x_80_ = lean_apply_2(v_inst_76_, v___x_79_, v_x_77_);
v___x_81_ = lean_unbox(v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemRange___redArg___lam__0___boxed(lean_object* v_f_82_, lean_object* v_inst_83_, lean_object* v_x_84_, lean_object* v_a_85_){
_start:
{
uint8_t v_res_86_; lean_object* v_r_87_; 
v_res_86_ = lp_mathlib_MonoidHom_decidableMemRange___redArg___lam__0(v_f_82_, v_inst_83_, v_x_84_, v_a_85_);
v_r_87_ = lean_box(v_res_86_);
return v_r_87_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemRange___redArg(lean_object* v_f_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_x_91_){
_start:
{
lean_object* v___f_92_; uint8_t v___x_93_; 
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_decidableMemRange___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_92_, 0, v_f_88_);
lean_closure_set(v___f_92_, 1, v_inst_90_);
lean_closure_set(v___f_92_, 2, v_x_91_);
v___x_93_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_89_, v___f_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemRange___redArg___boxed(lean_object* v_f_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_x_97_){
_start:
{
uint8_t v_res_98_; lean_object* v_r_99_; 
v_res_98_ = lp_mathlib_MonoidHom_decidableMemRange___redArg(v_f_94_, v_inst_95_, v_inst_96_, v_x_97_);
v_r_99_ = lean_box(v_res_98_);
return v_r_99_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemRange(lean_object* v_G_100_, lean_object* v_inst_101_, lean_object* v_N_102_, lean_object* v_inst_103_, lean_object* v_f_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_x_107_){
_start:
{
uint8_t v___x_108_; 
v___x_108_ = lp_mathlib_MonoidHom_decidableMemRange___redArg(v_f_104_, v_inst_105_, v_inst_106_, v_x_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemRange___boxed(lean_object* v_G_109_, lean_object* v_inst_110_, lean_object* v_N_111_, lean_object* v_inst_112_, lean_object* v_f_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_x_116_){
_start:
{
uint8_t v_res_117_; lean_object* v_r_118_; 
v_res_117_ = lp_mathlib_MonoidHom_decidableMemRange(v_G_109_, v_inst_110_, v_N_111_, v_inst_112_, v_f_113_, v_inst_114_, v_inst_115_, v_x_116_);
lean_dec_ref(v_inst_112_);
lean_dec_ref(v_inst_110_);
v_r_118_ = lean_box(v_res_117_);
return v_r_118_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemRange___redArg(lean_object* v_f_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_x_122_){
_start:
{
lean_object* v___f_123_; uint8_t v___x_124_; 
v___f_123_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_decidableMemRange___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_123_, 0, v_f_119_);
lean_closure_set(v___f_123_, 1, v_inst_121_);
lean_closure_set(v___f_123_, 2, v_x_122_);
v___x_124_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_120_, v___f_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemRange___redArg___boxed(lean_object* v_f_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_x_128_){
_start:
{
uint8_t v_res_129_; lean_object* v_r_130_; 
v_res_129_ = lp_mathlib_AddMonoidHom_decidableMemRange___redArg(v_f_125_, v_inst_126_, v_inst_127_, v_x_128_);
v_r_130_ = lean_box(v_res_129_);
return v_r_130_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemRange(lean_object* v_G_131_, lean_object* v_inst_132_, lean_object* v_N_133_, lean_object* v_inst_134_, lean_object* v_f_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_x_138_){
_start:
{
uint8_t v___x_139_; 
v___x_139_ = lp_mathlib_AddMonoidHom_decidableMemRange___redArg(v_f_135_, v_inst_136_, v_inst_137_, v_x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemRange___boxed(lean_object* v_G_140_, lean_object* v_inst_141_, lean_object* v_N_142_, lean_object* v_inst_143_, lean_object* v_f_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_x_147_){
_start:
{
uint8_t v_res_148_; lean_object* v_r_149_; 
v_res_148_ = lp_mathlib_AddMonoidHom_decidableMemRange(v_G_140_, v_inst_141_, v_N_142_, v_inst_143_, v_f_144_, v_inst_145_, v_inst_146_, v_x_147_);
lean_dec_ref(v_inst_143_);
lean_dec_ref(v_inst_141_);
v_r_149_ = lean_box(v_res_148_);
return v_r_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeMrange___redArg___lam__0(lean_object* v_f_150_, lean_object* v___y_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lean_apply_1(v_f_150_, v___y_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeMrange___redArg(lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_f_155_){
_start:
{
lean_object* v___f_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_fintypeMrange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_156_, 0, v_f_155_);
v___x_157_ = lp_mathlib_PLift_fintype___redArg(v_inst_153_);
v___x_158_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_154_, v___f_156_, v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeMrange(lean_object* v_M_159_, lean_object* v_N_160_, lean_object* v_inst_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_f_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_MonoidHom_fintypeMrange___redArg(v_inst_163_, v_inst_164_, v_f_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeMrange___boxed(lean_object* v_M_167_, lean_object* v_N_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_f_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_MonoidHom_fintypeMrange(v_M_167_, v_N_168_, v_inst_169_, v_inst_170_, v_inst_171_, v_inst_172_, v_f_173_);
lean_dec_ref(v_inst_170_);
lean_dec_ref(v_inst_169_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeMrange___redArg(lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_f_177_){
_start:
{
lean_object* v___f_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___f_178_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_fintypeMrange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_178_, 0, v_f_177_);
v___x_179_ = lp_mathlib_PLift_fintype___redArg(v_inst_175_);
v___x_180_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_176_, v___f_178_, v___x_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeMrange(lean_object* v_M_181_, lean_object* v_N_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_f_187_){
_start:
{
lean_object* v___x_188_; 
v___x_188_ = lp_mathlib_AddMonoidHom_fintypeMrange___redArg(v_inst_185_, v_inst_186_, v_f_187_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeMrange___boxed(lean_object* v_M_189_, lean_object* v_N_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_f_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_AddMonoidHom_fintypeMrange(v_M_189_, v_N_190_, v_inst_191_, v_inst_192_, v_inst_193_, v_inst_194_, v_f_195_);
lean_dec_ref(v_inst_192_);
lean_dec_ref(v_inst_191_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeRange___redArg(lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_f_199_){
_start:
{
lean_object* v___f_200_; lean_object* v___x_201_; lean_object* v___x_202_; 
v___f_200_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_fintypeMrange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_200_, 0, v_f_199_);
v___x_201_ = lp_mathlib_PLift_fintype___redArg(v_inst_197_);
v___x_202_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_198_, v___f_200_, v___x_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeRange(lean_object* v_G_203_, lean_object* v_inst_204_, lean_object* v_N_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_f_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_MonoidHom_fintypeRange___redArg(v_inst_207_, v_inst_208_, v_f_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fintypeRange___boxed(lean_object* v_G_211_, lean_object* v_inst_212_, lean_object* v_N_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_f_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_MonoidHom_fintypeRange(v_G_211_, v_inst_212_, v_N_213_, v_inst_214_, v_inst_215_, v_inst_216_, v_f_217_);
lean_dec_ref(v_inst_214_);
lean_dec_ref(v_inst_212_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeRange___redArg(lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_f_221_){
_start:
{
lean_object* v___f_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___f_222_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_fintypeMrange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_222_, 0, v_f_221_);
v___x_223_ = lp_mathlib_PLift_fintype___redArg(v_inst_219_);
v___x_224_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_220_, v___f_222_, v___x_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeRange(lean_object* v_G_225_, lean_object* v_inst_226_, lean_object* v_N_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_f_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_AddMonoidHom_fintypeRange___redArg(v_inst_229_, v_inst_230_, v_f_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fintypeRange___boxed(lean_object* v_G_233_, lean_object* v_inst_234_, lean_object* v_N_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_f_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_AddMonoidHom_fintypeRange(v_G_233_, v_inst_234_, v_N_235_, v_inst_236_, v_inst_237_, v_inst_238_, v_f_239_);
lean_dec_ref(v_inst_236_);
lean_dec_ref(v_inst_234_);
return v_res_240_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Finite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Range(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_NatCard(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_NatCard(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_NatCard(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_NatCard(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Finite(builtin);
}
#ifdef __cplusplus
}
#endif
