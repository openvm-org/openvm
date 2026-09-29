// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.Hom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Pretransitive public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Tactic.ToDual
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
lean_object* lp_mathlib_AddMonoid_toAddAction___redArg(lean_object*);
lean_object* lp_mathlib_VAdd_comp_vadd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_SMul_comp_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SMul_comp_smul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulAction___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionLeft___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addActionLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addActionLeft___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addActionLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addActionLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_compHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_compHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_compHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_compHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_smulOneHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_smulOneHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_smulOneHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_smulOneHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_vaddZeroHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_vaddZeroHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_vaddZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_vaddZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass_match__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass_match__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionLeft___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionLeft___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Function_Surjective_mulActionLeft___redArg(v_inst_2_);
lean_dec(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionLeft(lean_object* v_R_4_, lean_object* v_S_5_, lean_object* v_M_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_f_11_, lean_object* v_hf_12_, lean_object* v_hsmul_13_){
_start:
{
lean_inc(v_inst_10_);
return v_inst_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulActionLeft___boxed(lean_object* v_R_14_, lean_object* v_S_15_, lean_object* v_M_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_f_21_, lean_object* v_hf_22_, lean_object* v_hsmul_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Function_Surjective_mulActionLeft(v_R_14_, v_S_15_, v_M_16_, v_inst_17_, v_inst_18_, v_inst_19_, v_inst_20_, v_f_21_, v_hf_22_, v_hsmul_23_);
lean_dec(v_f_21_);
lean_dec(v_inst_20_);
lean_dec_ref(v_inst_19_);
lean_dec(v_inst_18_);
lean_dec_ref(v_inst_17_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addActionLeft___redArg(lean_object* v_inst_25_){
_start:
{
lean_inc(v_inst_25_);
return v_inst_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addActionLeft___redArg___boxed(lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Function_Surjective_addActionLeft___redArg(v_inst_26_);
lean_dec(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addActionLeft(lean_object* v_R_28_, lean_object* v_S_29_, lean_object* v_M_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_f_35_, lean_object* v_hf_36_, lean_object* v_hsmul_37_){
_start:
{
lean_inc(v_inst_34_);
return v_inst_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addActionLeft___boxed(lean_object* v_R_38_, lean_object* v_S_39_, lean_object* v_M_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_f_45_, lean_object* v_hf_46_, lean_object* v_hsmul_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_Function_Surjective_addActionLeft(v_R_38_, v_S_39_, v_M_40_, v_inst_41_, v_inst_42_, v_inst_43_, v_inst_44_, v_f_45_, v_hf_46_, v_hsmul_47_);
lean_dec(v_f_45_);
lean_dec(v_inst_44_);
lean_dec_ref(v_inst_43_);
lean_dec(v_inst_42_);
lean_dec_ref(v_inst_41_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_compHom___redArg___lam__0(lean_object* v_g_49_, lean_object* v___y_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lean_apply_1(v_g_49_, v___y_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_compHom___redArg(lean_object* v_inst_52_, lean_object* v_g_53_){
_start:
{
lean_object* v___f_54_; lean_object* v___x_55_; 
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_compHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_54_, 0, v_g_53_);
v___x_55_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_55_, 0, lean_box(0));
lean_closure_set(v___x_55_, 1, lean_box(0));
lean_closure_set(v___x_55_, 2, lean_box(0));
lean_closure_set(v___x_55_, 3, v_inst_52_);
lean_closure_set(v___x_55_, 4, v___f_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_compHom(lean_object* v_M_56_, lean_object* v_N_57_, lean_object* v_00_u03b1_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_g_62_){
_start:
{
lean_object* v___f_63_; lean_object* v___x_64_; 
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_compHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_63_, 0, v_g_62_);
v___x_64_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_64_, 0, lean_box(0));
lean_closure_set(v___x_64_, 1, lean_box(0));
lean_closure_set(v___x_64_, 2, lean_box(0));
lean_closure_set(v___x_64_, 3, v_inst_60_);
lean_closure_set(v___x_64_, 4, v___f_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_compHom___boxed(lean_object* v_M_65_, lean_object* v_N_66_, lean_object* v_00_u03b1_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_g_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_MulAction_compHom(v_M_65_, v_N_66_, v_00_u03b1_67_, v_inst_68_, v_inst_69_, v_inst_70_, v_g_71_);
lean_dec_ref(v_inst_70_);
lean_dec_ref(v_inst_68_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___redArg(lean_object* v_inst_73_, lean_object* v_g_74_){
_start:
{
lean_object* v___f_75_; lean_object* v___x_76_; 
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_compHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_75_, 0, v_g_74_);
v___x_76_ = lean_alloc_closure((void*)(lp_mathlib_VAdd_comp_vadd), 7, 5);
lean_closure_set(v___x_76_, 0, lean_box(0));
lean_closure_set(v___x_76_, 1, lean_box(0));
lean_closure_set(v___x_76_, 2, lean_box(0));
lean_closure_set(v___x_76_, 3, v_inst_73_);
lean_closure_set(v___x_76_, 4, v___f_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom(lean_object* v_M_77_, lean_object* v_N_78_, lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_g_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_AddAction_compHom___redArg(v_inst_81_, v_g_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___boxed(lean_object* v_M_85_, lean_object* v_N_86_, lean_object* v_00_u03b1_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_g_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_AddAction_compHom(v_M_85_, v_N_86_, v_00_u03b1_87_, v_inst_88_, v_inst_89_, v_inst_90_, v_g_91_);
lean_dec_ref(v_inst_90_);
lean_dec_ref(v_inst_88_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_smulOneHom___redArg___lam__0(lean_object* v_inst_93_, lean_object* v_toOne_94_, lean_object* v_x_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lean_apply_2(v_inst_93_, v_x_95_, v_toOne_94_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_smulOneHom___redArg(lean_object* v_inst_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v___x_99_; lean_object* v_toOne_100_; lean_object* v___f_101_; 
v___x_99_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_97_);
v_toOne_100_ = lean_ctor_get(v___x_99_, 0);
lean_inc(v_toOne_100_);
lean_dec_ref(v___x_99_);
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_smulOneHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_101_, 0, v_inst_98_);
lean_closure_set(v___f_101_, 1, v_toOne_100_);
return v___f_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_smulOneHom(lean_object* v_M_102_, lean_object* v_N_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_MonoidHom_smulOneHom___redArg(v_inst_105_, v_inst_106_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_smulOneHom___boxed(lean_object* v_M_109_, lean_object* v_N_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_MonoidHom_smulOneHom(v_M_109_, v_N_110_, v_inst_111_, v_inst_112_, v_inst_113_, v_inst_114_);
lean_dec_ref(v_inst_111_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_vaddZeroHom___redArg___lam__0(lean_object* v_inst_116_, lean_object* v_toZero_117_, lean_object* v_x_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lean_apply_2(v_inst_116_, v_x_118_, v_toZero_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_vaddZeroHom___redArg(lean_object* v_inst_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v___x_122_; lean_object* v_toZero_123_; lean_object* v___f_124_; 
v___x_122_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_120_);
v_toZero_123_ = lean_ctor_get(v___x_122_, 0);
lean_inc(v_toZero_123_);
lean_dec_ref(v___x_122_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_vaddZeroHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_124_, 0, v_inst_121_);
lean_closure_set(v___f_124_, 1, v_toZero_123_);
return v___f_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_vaddZeroHom(lean_object* v_M_125_, lean_object* v_N_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_AddMonoidHom_vaddZeroHom___redArg(v_inst_128_, v_inst_129_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_vaddZeroHom___boxed(lean_object* v_M_132_, lean_object* v_N_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_AddMonoidHom_vaddZeroHom(v_M_132_, v_N_133_, v_inst_134_, v_inst_135_, v_inst_136_, v_inst_137_);
lean_dec_ref(v_inst_134_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__0(lean_object* v_f_139_, lean_object* v___y_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lean_apply_1(v_f_139_, v___y_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__1(lean_object* v___x_142_, lean_object* v_f_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v___f_146_; lean_object* v___x_147_; 
v___f_146_ = lean_alloc_closure((void*)(lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__0), 2, 1);
lean_closure_set(v___f_146_, 0, v_f_143_);
v___x_147_ = lp_mathlib_SMul_comp_smul___redArg(v___x_142_, v___f_146_, v___y_144_, v___y_145_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__2(lean_object* v___x_148_, lean_object* v_x_149_, lean_object* v___y_150_){
_start:
{
lean_object* v___x_53__overap_151_; lean_object* v___x_152_; 
v___x_53__overap_151_ = lp_mathlib_MonoidHom_smulOneHom___redArg(v___x_148_, v_x_149_);
v___x_152_ = lean_apply_1(v___x_53__overap_151_, v___y_150_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg(lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; lean_object* v___f_155_; lean_object* v___x_156_; lean_object* v___f_157_; lean_object* v___x_158_; 
v___x_154_ = lp_mathlib_Monoid_toMulAction___redArg(v_inst_153_);
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__1), 4, 1);
lean_closure_set(v___f_155_, 0, v___x_154_);
v___x_156_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_153_);
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___lam__2), 3, 1);
lean_closure_set(v___f_157_, 0, v___x_156_);
v___x_158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_158_, 0, v___f_155_);
lean_ctor_set(v___x_158_, 1, v___f_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg___boxed(lean_object* v_inst_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg(v_inst_159_);
lean_dec_ref(v_inst_159_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower(lean_object* v_M_161_, lean_object* v_N_162_, lean_object* v_inst_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_monoidHomEquivMulActionIsScalarTower___redArg(v_inst_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidHomEquivMulActionIsScalarTower___boxed(lean_object* v_M_166_, lean_object* v_N_167_, lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_monoidHomEquivMulActionIsScalarTower(v_M_166_, v_N_167_, v_inst_168_, v_inst_169_);
lean_dec_ref(v_inst_169_);
lean_dec_ref(v_inst_168_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass_match__1___redArg(lean_object* v_x_171_, lean_object* v_h__1_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lean_apply_2(v_h__1_172_, v_x_171_, lean_box(0));
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass_match__1(lean_object* v_M_174_, lean_object* v_N_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_motive_178_, lean_object* v_x_179_, lean_object* v_h__1_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lean_apply_2(v_h__1_180_, v_x_179_, lean_box(0));
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass_match__1___boxed(lean_object* v_M_182_, lean_object* v_N_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_motive_186_, lean_object* v_x_187_, lean_object* v_h__1_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass_match__1(v_M_182_, v_N_183_, v_inst_184_, v_inst_185_, v_motive_186_, v_x_187_, v_h__1_188_);
lean_dec_ref(v_inst_185_);
lean_dec_ref(v_inst_184_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg___lam__0(lean_object* v___x_190_, lean_object* v_f_191_, lean_object* v___y_192_, lean_object* v___y_193_){
_start:
{
lean_object* v___x_33__overap_194_; lean_object* v___x_195_; 
v___x_33__overap_194_ = lp_mathlib_AddAction_compHom___redArg(v___x_190_, v_f_191_);
v___x_195_ = lean_apply_2(v___x_33__overap_194_, v___y_192_, v___y_193_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg___lam__1(lean_object* v___x_196_, lean_object* v_x_197_, lean_object* v___y_198_){
_start:
{
lean_object* v___x_37__overap_199_; lean_object* v___x_200_; 
v___x_37__overap_199_ = lp_mathlib_AddMonoidHom_vaddZeroHom___redArg(v___x_196_, v_x_197_);
v___x_200_ = lean_apply_1(v___x_37__overap_199_, v___y_198_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg(lean_object* v_inst_201_){
_start:
{
lean_object* v___x_202_; lean_object* v___f_203_; lean_object* v___x_204_; lean_object* v___f_205_; lean_object* v___x_206_; 
v___x_202_ = lp_mathlib_AddMonoid_toAddAction___redArg(v_inst_201_);
v___f_203_ = lean_alloc_closure((void*)(lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_203_, 0, v___x_202_);
v___x_204_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_201_);
v___f_205_ = lean_alloc_closure((void*)(lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg___lam__1), 3, 1);
lean_closure_set(v___f_205_, 0, v___x_204_);
v___x_206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_206_, 0, v___f_203_);
lean_ctor_set(v___x_206_, 1, v___f_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg___boxed(lean_object* v_inst_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg(v_inst_207_);
lean_dec_ref(v_inst_207_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass(lean_object* v_M_209_, lean_object* v_N_210_, lean_object* v_inst_211_, lean_object* v_inst_212_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___redArg(v_inst_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass___boxed(lean_object* v_M_214_, lean_object* v_N_215_, lean_object* v_inst_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_addMonoidHomEquivAddActionVAddAssocClass(v_M_214_, v_N_215_, v_inst_216_, v_inst_217_);
lean_dec_ref(v_inst_217_);
lean_dec_ref(v_inst_216_);
return v_res_218_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pretransitive(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pretransitive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Pretransitive(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Pretransitive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
}
#ifdef __cplusplus
}
#endif
