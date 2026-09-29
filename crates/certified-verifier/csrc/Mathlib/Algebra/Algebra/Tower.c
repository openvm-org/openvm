// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Tower
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Equiv public import Mathlib.LinearAlgebra.Span.Basic
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
lean_object* lp_mathlib_DistribSMul_toLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_aut___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_applyMulSemiringAction___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_MulSemiringAction_toAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lsmul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lsmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_restrictScalars___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_restrictScalars___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_restrictScalars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_restrictScalars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgHom_restrictScalars___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsHomOfSurjective___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsHomOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalars___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalars___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_restrictScalarsHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_applyMulSemiringAction___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_restrictScalarsHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_restrictScalarsHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalarsHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalarsHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalarsHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsHomOfSurjective___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsHomOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__0 = (const lean_object*)&lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__0_value;
static const lean_closure_object lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__1 = (const lean_object*)&lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__1_value;
static const lean_ctor_object lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__1_value),((lean_object*)&lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__0_value)}};
static const lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__2 = (const lean_object*)&lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lsmul___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_5_, 0, lean_box(0));
lean_closure_set(v___x_5_, 1, lean_box(0));
lean_closure_set(v___x_5_, 2, lean_box(0));
lean_closure_set(v___x_5_, 3, v_inst_1_);
lean_closure_set(v___x_5_, 4, v_inst_2_);
lean_closure_set(v___x_5_, 5, v_inst_4_);
lean_closure_set(v___x_5_, 6, v_inst_3_);
lean_closure_set(v___x_5_, 7, lean_box(0));
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lsmul(lean_object* v_R_6_, lean_object* v_A_7_, lean_object* v_B_8_, lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_alloc_closure((void*)(lp_mathlib_DistribSMul_toLinearMap___boxed), 9, 8);
lean_closure_set(v___x_22_, 0, lean_box(0));
lean_closure_set(v___x_22_, 1, lean_box(0));
lean_closure_set(v___x_22_, 2, lean_box(0));
lean_closure_set(v___x_22_, 3, v_inst_12_);
lean_closure_set(v___x_22_, 4, v_inst_15_);
lean_closure_set(v___x_22_, 5, v_inst_18_);
lean_closure_set(v___x_22_, 6, v_inst_17_);
lean_closure_set(v___x_22_, 7, lean_box(0));
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_lsmul___boxed(lean_object* v_R_23_, lean_object* v_A_24_, lean_object* v_B_25_, lean_object* v_M_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Algebra_lsmul(v_R_23_, v_A_24_, v_B_25_, v_M_26_, v_inst_27_, v_inst_28_, v_inst_29_, v_inst_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_inst_35_, v_inst_36_, v_inst_37_, v_inst_38_);
lean_dec(v_inst_33_);
lean_dec_ref(v_inst_31_);
lean_dec_ref(v_inst_30_);
lean_dec_ref(v_inst_28_);
lean_dec_ref(v_inst_27_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_restrictScalars___redArg___lam__0(lean_object* v_f_40_, lean_object* v___y_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lean_apply_1(v_f_40_, v___y_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_restrictScalars___redArg(lean_object* v_f_43_){
_start:
{
lean_object* v___f_44_; 
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_restrictScalars___redArg___lam__0), 2, 1);
lean_closure_set(v___f_44_, 0, v_f_43_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_restrictScalars(lean_object* v_R_45_, lean_object* v_S_46_, lean_object* v_A_47_, lean_object* v_B_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_f_60_){
_start:
{
lean_object* v___f_61_; 
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_restrictScalars___redArg___lam__0), 2, 1);
lean_closure_set(v___f_61_, 0, v_f_60_);
return v___f_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_restrictScalars___boxed(lean_object* v_R_62_, lean_object* v_S_63_, lean_object* v_A_64_, lean_object* v_B_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_f_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_AlgHom_restrictScalars(v_R_62_, v_S_63_, v_A_64_, v_B_65_, v_inst_66_, v_inst_67_, v_inst_68_, v_inst_69_, v_inst_70_, v_inst_71_, v_inst_72_, v_inst_73_, v_inst_74_, v_inst_75_, v_inst_76_, v_f_77_);
lean_dec_ref(v_inst_74_);
lean_dec_ref(v_inst_73_);
lean_dec_ref(v_inst_72_);
lean_dec_ref(v_inst_71_);
lean_dec_ref(v_inst_70_);
lean_dec_ref(v_inst_69_);
lean_dec_ref(v_inst_68_);
lean_dec_ref(v_inst_67_);
lean_dec_ref(v_inst_66_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg(lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___f_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___f_89_ = ((lean_object*)(lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg___closed__0));
v___x_90_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_restrictScalars___boxed), 16, 15);
lean_closure_set(v___x_90_, 0, lean_box(0));
lean_closure_set(v___x_90_, 1, lean_box(0));
lean_closure_set(v___x_90_, 2, lean_box(0));
lean_closure_set(v___x_90_, 3, lean_box(0));
lean_closure_set(v___x_90_, 4, v_inst_80_);
lean_closure_set(v___x_90_, 5, v_inst_81_);
lean_closure_set(v___x_90_, 6, v_inst_82_);
lean_closure_set(v___x_90_, 7, v_inst_83_);
lean_closure_set(v___x_90_, 8, v_inst_84_);
lean_closure_set(v___x_90_, 9, v_inst_85_);
lean_closure_set(v___x_90_, 10, v_inst_86_);
lean_closure_set(v___x_90_, 11, v_inst_87_);
lean_closure_set(v___x_90_, 12, v_inst_88_);
lean_closure_set(v___x_90_, 13, lean_box(0));
lean_closure_set(v___x_90_, 14, lean_box(0));
v___x_91_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_91_, 0, v___f_89_);
lean_ctor_set(v___x_91_, 1, v___x_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfSurjective(lean_object* v_R_92_, lean_object* v_S_93_, lean_object* v_A_94_, lean_object* v_B_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_h_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg(v_inst_96_, v_inst_97_, v_inst_98_, v_inst_99_, v_inst_100_, v_inst_101_, v_inst_102_, v_inst_103_, v_inst_104_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsHomOfSurjective___redArg(lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; 
lean_inc_ref(v_inst_114_);
lean_inc_ref(v_inst_113_);
lean_inc_ref(v_inst_111_);
v___x_115_ = lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg(v_inst_109_, v_inst_110_, v_inst_111_, v_inst_111_, v_inst_112_, v_inst_113_, v_inst_113_, v_inst_114_, v_inst_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsHomOfSurjective(lean_object* v_R_116_, lean_object* v_S_117_, lean_object* v_A_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_h_126_){
_start:
{
lean_object* v___x_127_; 
lean_inc_ref(v_inst_124_);
lean_inc_ref(v_inst_123_);
lean_inc_ref(v_inst_121_);
v___x_127_ = lp_mathlib_AlgHom_extendScalarsOfSurjective___redArg(v_inst_119_, v_inst_120_, v_inst_121_, v_inst_121_, v_inst_122_, v_inst_123_, v_inst_123_, v_inst_124_, v_inst_124_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalars___redArg(lean_object* v_f_128_){
_start:
{
lean_inc_ref(v_f_128_);
return v_f_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalars___redArg___boxed(lean_object* v_f_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_AlgEquiv_restrictScalars___redArg(v_f_129_);
lean_dec_ref(v_f_129_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalars(lean_object* v_R_131_, lean_object* v_S_132_, lean_object* v_A_133_, lean_object* v_B_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_f_146_){
_start:
{
lean_inc_ref(v_f_146_);
return v_f_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalars___boxed(lean_object* v_R_147_, lean_object* v_S_148_, lean_object* v_A_149_, lean_object* v_B_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_f_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_AlgEquiv_restrictScalars(v_R_147_, v_S_148_, v_A_149_, v_B_150_, v_inst_151_, v_inst_152_, v_inst_153_, v_inst_154_, v_inst_155_, v_inst_156_, v_inst_157_, v_inst_158_, v_inst_159_, v_inst_160_, v_inst_161_, v_f_162_);
lean_dec_ref(v_f_162_);
lean_dec_ref(v_inst_159_);
lean_dec_ref(v_inst_158_);
lean_dec_ref(v_inst_157_);
lean_dec_ref(v_inst_156_);
lean_dec_ref(v_inst_155_);
lean_dec_ref(v_inst_154_);
lean_dec_ref(v_inst_153_);
lean_dec_ref(v_inst_152_);
lean_dec_ref(v_inst_151_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalarsHom___redArg(lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v___x_170_; lean_object* v___f_171_; lean_object* v___x_172_; 
lean_inc_ref(v_inst_167_);
v___x_170_ = lp_mathlib_AlgEquiv_aut___redArg(v_inst_166_, v_inst_167_, v_inst_168_);
v___f_171_ = ((lean_object*)(lp_mathlib_AlgEquiv_restrictScalarsHom___redArg___closed__0));
v___x_172_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_toAlgEquiv___boxed), 10, 9);
lean_closure_set(v___x_172_, 0, lean_box(0));
lean_closure_set(v___x_172_, 1, lean_box(0));
lean_closure_set(v___x_172_, 2, lean_box(0));
lean_closure_set(v___x_172_, 3, v_inst_165_);
lean_closure_set(v___x_172_, 4, v_inst_167_);
lean_closure_set(v___x_172_, 5, v_inst_169_);
lean_closure_set(v___x_172_, 6, v___x_170_);
lean_closure_set(v___x_172_, 7, v___f_171_);
lean_closure_set(v___x_172_, 8, lean_box(0));
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalarsHom(lean_object* v_R_173_, lean_object* v_S_174_, lean_object* v_A_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lp_mathlib_AlgEquiv_restrictScalarsHom___redArg(v_inst_176_, v_inst_177_, v_inst_178_, v_inst_180_, v_inst_181_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_restrictScalarsHom___boxed(lean_object* v_R_184_, lean_object* v_S_185_, lean_object* v_A_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_inst_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_AlgEquiv_restrictScalarsHom(v_R_184_, v_S_185_, v_A_186_, v_inst_187_, v_inst_188_, v_inst_189_, v_inst_190_, v_inst_191_, v_inst_192_, v_inst_193_);
lean_dec_ref(v_inst_190_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___lam__0(lean_object* v_f_195_){
_start:
{
lean_inc_ref(v_f_195_);
return v_f_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___lam__0___boxed(lean_object* v_f_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___lam__0(v_f_196_);
lean_dec_ref(v_f_196_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg(lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v___f_208_; lean_object* v___x_209_; lean_object* v___x_210_; 
v___f_208_ = ((lean_object*)(lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg___closed__0));
v___x_209_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_restrictScalars___boxed), 16, 15);
lean_closure_set(v___x_209_, 0, lean_box(0));
lean_closure_set(v___x_209_, 1, lean_box(0));
lean_closure_set(v___x_209_, 2, lean_box(0));
lean_closure_set(v___x_209_, 3, lean_box(0));
lean_closure_set(v___x_209_, 4, v_inst_199_);
lean_closure_set(v___x_209_, 5, v_inst_200_);
lean_closure_set(v___x_209_, 6, v_inst_201_);
lean_closure_set(v___x_209_, 7, v_inst_202_);
lean_closure_set(v___x_209_, 8, v_inst_203_);
lean_closure_set(v___x_209_, 9, v_inst_204_);
lean_closure_set(v___x_209_, 10, v_inst_205_);
lean_closure_set(v___x_209_, 11, v_inst_206_);
lean_closure_set(v___x_209_, 12, v_inst_207_);
lean_closure_set(v___x_209_, 13, lean_box(0));
lean_closure_set(v___x_209_, 14, lean_box(0));
v___x_210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_210_, 0, v___f_208_);
lean_ctor_set(v___x_210_, 1, v___x_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfSurjective(lean_object* v_R_211_, lean_object* v_S_212_, lean_object* v_A_213_, lean_object* v_B_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_h_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg(v_inst_215_, v_inst_216_, v_inst_217_, v_inst_218_, v_inst_219_, v_inst_220_, v_inst_221_, v_inst_222_, v_inst_223_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsHomOfSurjective___redArg(lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_inst_233_){
_start:
{
lean_object* v___x_234_; 
lean_inc_ref(v_inst_233_);
lean_inc_ref(v_inst_232_);
lean_inc_ref(v_inst_230_);
v___x_234_ = lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg(v_inst_228_, v_inst_229_, v_inst_230_, v_inst_230_, v_inst_231_, v_inst_232_, v_inst_232_, v_inst_233_, v_inst_233_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsHomOfSurjective(lean_object* v_R_235_, lean_object* v_S_236_, lean_object* v_A_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_h_245_){
_start:
{
lean_object* v___x_246_; 
lean_inc_ref(v_inst_243_);
lean_inc_ref(v_inst_242_);
lean_inc_ref(v_inst_240_);
v___x_246_ = lp_mathlib_AlgEquiv_extendScalarsOfSurjective___redArg(v_inst_238_, v_inst_239_, v_inst_240_, v_inst_240_, v_inst_241_, v_inst_242_, v_inst_242_, v_inst_243_, v_inst_243_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___lam__0(lean_object* v_N_247_){
_start:
{
return v_N_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___lam__1(lean_object* v_N_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lean_box(0);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective(lean_object* v_R_255_, lean_object* v_S_256_, lean_object* v_M_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_h_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = ((lean_object*)(lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___closed__2));
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective___boxed(lean_object* v_R_267_, lean_object* v_S_268_, lean_object* v_M_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_h_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_Submodule_orderIsoOfAlgebraMapSurjective(v_R_267_, v_S_268_, v_M_269_, v_inst_270_, v_inst_271_, v_inst_272_, v_inst_273_, v_inst_274_, v_inst_275_, v_inst_276_, v_h_277_);
lean_dec(v_inst_275_);
lean_dec(v_inst_274_);
lean_dec_ref(v_inst_273_);
lean_dec_ref(v_inst_272_);
lean_dec_ref(v_inst_271_);
lean_dec_ref(v_inst_270_);
return v_res_278_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
}
#ifdef __cplusplus
}
#endif
