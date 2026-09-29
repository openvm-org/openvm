// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Prod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Equiv public import Mathlib.Algebra.Algebra.Hom public import Mathlib.Algebra.Module.Prod
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
lean_object* lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_fst___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_snd___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_RingEquiv_prodCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_ofRingEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_algebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_algebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_algebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgHom_fst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgHom_fst___closed__0 = (const lean_object*)&lp_mathlib_AlgHom_fst___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgHom_snd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgHom_snd___closed__0 = (const lean_object*)&lp_mathlib_AlgHom_snd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_AlgHom_prodEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgHom_prodEquiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgHom_prodEquiv___closed__0 = (const lean_object*)&lp_mathlib_AlgHom_prodEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_AlgHom_prodEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgHom_prodEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgHom_prodEquiv___closed__1 = (const lean_object*)&lp_mathlib_AlgHom_prodEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_AlgHom_prodEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AlgHom_prodEquiv___closed__0_value),((lean_object*)&lp_mathlib_AlgHom_prodEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_AlgHom_prodEquiv___closed__2 = (const lean_object*)&lp_mathlib_AlgHom_prodEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_prodUnique___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_prodUnique___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_prodUnique___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgEquiv_uniqueProd___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgEquiv_uniqueProd___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_algebra___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v_toSMul_3_; lean_object* v_algebraMap_4_; lean_object* v_toSMul_5_; lean_object* v_algebraMap_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_15_; 
v_toSMul_3_ = lean_ctor_get(v_inst_1_, 0);
lean_inc(v_toSMul_3_);
v_algebraMap_4_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_algebraMap_4_);
lean_dec_ref(v_inst_1_);
v_toSMul_5_ = lean_ctor_get(v_inst_2_, 0);
v_algebraMap_6_ = lean_ctor_get(v_inst_2_, 1);
v_isSharedCheck_15_ = !lean_is_exclusive(v_inst_2_);
if (v_isSharedCheck_15_ == 0)
{
v___x_8_ = v_inst_2_;
v_isShared_9_ = v_isSharedCheck_15_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_algebraMap_6_);
lean_inc(v_toSMul_5_);
lean_dec(v_inst_2_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_15_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___f_10_; lean_object* v___f_11_; lean_object* v___x_13_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_10_, 0, v_toSMul_3_);
lean_closure_set(v___f_10_, 1, v_toSMul_5_);
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_11_, 0, v_algebraMap_4_);
lean_closure_set(v___f_11_, 1, v_algebraMap_6_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v___f_11_);
lean_ctor_set(v___x_8_, 0, v___f_10_);
v___x_13_ = v___x_8_;
goto v_reusejp_12_;
}
else
{
lean_object* v_reuseFailAlloc_14_; 
v_reuseFailAlloc_14_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_14_, 0, v___f_10_);
lean_ctor_set(v_reuseFailAlloc_14_, 1, v___f_11_);
v___x_13_ = v_reuseFailAlloc_14_;
goto v_reusejp_12_;
}
v_reusejp_12_:
{
return v___x_13_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_algebra(lean_object* v_R_16_, lean_object* v_A_17_, lean_object* v_B_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Prod_algebra___redArg(v_inst_21_, v_inst_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_algebra___boxed(lean_object* v_R_25_, lean_object* v_A_26_, lean_object* v_B_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Prod_algebra(v_R_25_, v_A_26_, v_B_27_, v_inst_28_, v_inst_29_, v_inst_30_, v_inst_31_, v_inst_32_);
lean_dec_ref(v_inst_31_);
lean_dec_ref(v_inst_29_);
lean_dec_ref(v_inst_28_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fst(lean_object* v_R_35_, lean_object* v_A_36_, lean_object* v_B_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v___f_43_; 
v___f_43_ = ((lean_object*)(lp_mathlib_AlgHom_fst___closed__0));
return v___f_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fst___boxed(lean_object* v_R_44_, lean_object* v_A_45_, lean_object* v_B_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_AlgHom_fst(v_R_44_, v_A_45_, v_B_46_, v_inst_47_, v_inst_48_, v_inst_49_, v_inst_50_, v_inst_51_);
lean_dec_ref(v_inst_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
lean_dec_ref(v_inst_48_);
lean_dec_ref(v_inst_47_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_snd(lean_object* v_R_54_, lean_object* v_A_55_, lean_object* v_B_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___f_62_; 
v___f_62_ = ((lean_object*)(lp_mathlib_AlgHom_snd___closed__0));
return v___f_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_snd___boxed(lean_object* v_R_63_, lean_object* v_A_64_, lean_object* v_B_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_AlgHom_snd(v_R_63_, v_A_64_, v_B_65_, v_inst_66_, v_inst_67_, v_inst_68_, v_inst_69_, v_inst_70_);
lean_dec_ref(v_inst_70_);
lean_dec_ref(v_inst_69_);
lean_dec_ref(v_inst_68_);
lean_dec_ref(v_inst_67_);
lean_dec_ref(v_inst_66_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prod___redArg(lean_object* v_f_72_, lean_object* v_g_73_){
_start:
{
lean_object* v___f_74_; 
v___f_74_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_74_, 0, v_f_72_);
lean_closure_set(v___f_74_, 1, v_g_73_);
return v___f_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prod(lean_object* v_R_75_, lean_object* v_A_76_, lean_object* v_B_77_, lean_object* v_C_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_f_86_, lean_object* v_g_87_){
_start:
{
lean_object* v___f_88_; 
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_88_, 0, v_f_86_);
lean_closure_set(v___f_88_, 1, v_g_87_);
return v___f_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prod___boxed(lean_object* v_R_89_, lean_object* v_A_90_, lean_object* v_B_91_, lean_object* v_C_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_f_100_, lean_object* v_g_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_AlgHom_prod(v_R_89_, v_A_90_, v_B_91_, v_C_92_, v_inst_93_, v_inst_94_, v_inst_95_, v_inst_96_, v_inst_97_, v_inst_98_, v_inst_99_, v_f_100_, v_g_101_);
lean_dec_ref(v_inst_99_);
lean_dec_ref(v_inst_98_);
lean_dec_ref(v_inst_97_);
lean_dec_ref(v_inst_96_);
lean_dec_ref(v_inst_95_);
lean_dec_ref(v_inst_94_);
lean_dec_ref(v_inst_93_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodEquiv___lam__0(lean_object* v_f_103_, lean_object* v___y_104_){
_start:
{
lean_object* v_fst_105_; lean_object* v_snd_106_; lean_object* v___x_107_; 
v_fst_105_ = lean_ctor_get(v_f_103_, 0);
lean_inc(v_fst_105_);
v_snd_106_ = lean_ctor_get(v_f_103_, 1);
lean_inc(v_snd_106_);
lean_dec_ref(v_f_103_);
v___x_107_ = lp_mathlib_NonUnitalRingHom_prod___redArg___lam__0(v_fst_105_, v_snd_106_, v___y_104_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodEquiv___lam__1(lean_object* v_f_108_){
_start:
{
lean_object* v___f_109_; lean_object* v___x_110_; lean_object* v___f_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___f_109_ = ((lean_object*)(lp_mathlib_AlgHom_fst___closed__0));
lean_inc_ref(v_f_108_);
v___x_110_ = lp_mathlib_AlgHom_comp___redArg(v___f_109_, v_f_108_);
v___f_111_ = ((lean_object*)(lp_mathlib_AlgHom_snd___closed__0));
v___x_112_ = lp_mathlib_AlgHom_comp___redArg(v___f_111_, v_f_108_);
v___x_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_110_);
lean_ctor_set(v___x_113_, 1, v___x_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodEquiv(lean_object* v_R_119_, lean_object* v_A_120_, lean_object* v_B_121_, lean_object* v_C_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = ((lean_object*)(lp_mathlib_AlgHom_prodEquiv___closed__2));
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodEquiv___boxed(lean_object* v_R_131_, lean_object* v_A_132_, lean_object* v_B_133_, lean_object* v_C_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_AlgHom_prodEquiv(v_R_131_, v_A_132_, v_B_133_, v_C_134_, v_inst_135_, v_inst_136_, v_inst_137_, v_inst_138_, v_inst_139_, v_inst_140_, v_inst_141_);
lean_dec_ref(v_inst_141_);
lean_dec_ref(v_inst_140_);
lean_dec_ref(v_inst_139_);
lean_dec_ref(v_inst_138_);
lean_dec_ref(v_inst_137_);
lean_dec_ref(v_inst_136_);
lean_dec_ref(v_inst_135_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodMap___redArg(lean_object* v_f_143_, lean_object* v_g_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib_RingHom_prodMap___redArg(v_f_143_, v_g_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodMap(lean_object* v_R_146_, lean_object* v_A_147_, lean_object* v_B_148_, lean_object* v_C_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_D_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_f_160_, lean_object* v_g_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_RingHom_prodMap___redArg(v_f_160_, v_g_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_prodMap___boxed(lean_object* v_R_163_, lean_object* v_A_164_, lean_object* v_B_165_, lean_object* v_C_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_D_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_f_177_, lean_object* v_g_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_AlgHom_prodMap(v_R_163_, v_A_164_, v_B_165_, v_C_166_, v_inst_167_, v_inst_168_, v_inst_169_, v_inst_170_, v_inst_171_, v_inst_172_, v_inst_173_, v_D_174_, v_inst_175_, v_inst_176_, v_f_177_, v_g_178_);
lean_dec_ref(v_inst_176_);
lean_dec_ref(v_inst_175_);
lean_dec_ref(v_inst_173_);
lean_dec_ref(v_inst_172_);
lean_dec_ref(v_inst_171_);
lean_dec_ref(v_inst_170_);
lean_dec_ref(v_inst_169_);
lean_dec_ref(v_inst_168_);
lean_dec_ref(v_inst_167_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodCongr___redArg(lean_object* v_l_180_, lean_object* v_r_181_){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; 
v___x_182_ = lp_mathlib_RingEquiv_prodCongr___redArg(v_l_180_, v_r_181_);
v___x_183_ = lp_mathlib_AlgEquiv_ofRingEquiv___redArg(v___x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodCongr(lean_object* v_R_184_, lean_object* v_inst_185_, lean_object* v_S_186_, lean_object* v_T_187_, lean_object* v_A_188_, lean_object* v_B_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_l_198_, lean_object* v_r_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lp_mathlib_AlgEquiv_prodCongr___redArg(v_l_198_, v_r_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodCongr___boxed(lean_object* v_R_201_, lean_object* v_inst_202_, lean_object* v_S_203_, lean_object* v_T_204_, lean_object* v_A_205_, lean_object* v_B_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_l_215_, lean_object* v_r_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_AlgEquiv_prodCongr(v_R_201_, v_inst_202_, v_S_203_, v_T_204_, v_A_205_, v_B_206_, v_inst_207_, v_inst_208_, v_inst_209_, v_inst_210_, v_inst_211_, v_inst_212_, v_inst_213_, v_inst_214_, v_l_215_, v_r_216_);
lean_dec_ref(v_inst_214_);
lean_dec_ref(v_inst_213_);
lean_dec_ref(v_inst_212_);
lean_dec_ref(v_inst_211_);
lean_dec_ref(v_inst_210_);
lean_dec_ref(v_inst_209_);
lean_dec_ref(v_inst_208_);
lean_dec_ref(v_inst_207_);
lean_dec_ref(v_inst_202_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg___lam__0(lean_object* v_self_218_){
_start:
{
lean_object* v_fst_219_; 
v_fst_219_ = lean_ctor_get(v_self_218_, 0);
lean_inc(v_fst_219_);
return v_fst_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg___lam__0___boxed(lean_object* v_self_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_AlgEquiv_prodUnique___redArg___lam__0(v_self_220_);
lean_dec_ref(v_self_220_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg___lam__1(lean_object* v_toZero_222_, lean_object* v_x_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_224_, 0, v_x_223_);
lean_ctor_set(v___x_224_, 1, v_toZero_222_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___redArg(lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; lean_object* v_toZero_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_237_; 
v___x_227_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_226_);
v_toZero_228_ = lean_ctor_get(v___x_227_, 1);
v_isSharedCheck_237_ = !lean_is_exclusive(v___x_227_);
if (v_isSharedCheck_237_ == 0)
{
lean_object* v_unused_238_; 
v_unused_238_ = lean_ctor_get(v___x_227_, 0);
lean_dec(v_unused_238_);
v___x_230_ = v___x_227_;
v_isShared_231_ = v_isSharedCheck_237_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_toZero_228_);
lean_dec(v___x_227_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_237_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___f_232_; lean_object* v___f_233_; lean_object* v___x_235_; 
v___f_232_ = ((lean_object*)(lp_mathlib_AlgEquiv_prodUnique___redArg___closed__0));
v___f_233_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_prodUnique___redArg___lam__1), 2, 1);
lean_closure_set(v___f_233_, 0, v_toZero_228_);
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 1, v___f_233_);
lean_ctor_set(v___x_230_, 0, v___f_232_);
v___x_235_ = v___x_230_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v___f_232_);
lean_ctor_set(v_reuseFailAlloc_236_, 1, v___f_233_);
v___x_235_ = v_reuseFailAlloc_236_;
goto v_reusejp_234_;
}
v_reusejp_234_:
{
return v___x_235_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique(lean_object* v_R_239_, lean_object* v_A_240_, lean_object* v_B_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lp_mathlib_AlgEquiv_prodUnique___redArg(v_inst_245_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_prodUnique___boxed(lean_object* v_R_249_, lean_object* v_A_250_, lean_object* v_B_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_inst_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_AlgEquiv_prodUnique(v_R_249_, v_A_250_, v_B_251_, v_inst_252_, v_inst_253_, v_inst_254_, v_inst_255_, v_inst_256_, v_inst_257_);
lean_dec(v_inst_257_);
lean_dec_ref(v_inst_256_);
lean_dec_ref(v_inst_254_);
lean_dec_ref(v_inst_253_);
lean_dec_ref(v_inst_252_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__0(lean_object* v_self_259_){
_start:
{
lean_object* v_snd_260_; 
v_snd_260_ = lean_ctor_get(v_self_259_, 1);
lean_inc(v_snd_260_);
return v_snd_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__0___boxed(lean_object* v_self_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__0(v_self_261_);
lean_dec_ref(v_self_261_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__1(lean_object* v_toZero_263_, lean_object* v_x_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_265_, 0, v_toZero_263_);
lean_ctor_set(v___x_265_, 1, v_x_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___redArg(lean_object* v_inst_267_){
_start:
{
lean_object* v___x_268_; lean_object* v_toZero_269_; lean_object* v___x_271_; uint8_t v_isShared_272_; uint8_t v_isSharedCheck_278_; 
v___x_268_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_267_);
v_toZero_269_ = lean_ctor_get(v___x_268_, 1);
v_isSharedCheck_278_ = !lean_is_exclusive(v___x_268_);
if (v_isSharedCheck_278_ == 0)
{
lean_object* v_unused_279_; 
v_unused_279_ = lean_ctor_get(v___x_268_, 0);
lean_dec(v_unused_279_);
v___x_271_ = v___x_268_;
v_isShared_272_ = v_isSharedCheck_278_;
goto v_resetjp_270_;
}
else
{
lean_inc(v_toZero_269_);
lean_dec(v___x_268_);
v___x_271_ = lean_box(0);
v_isShared_272_ = v_isSharedCheck_278_;
goto v_resetjp_270_;
}
v_resetjp_270_:
{
lean_object* v___f_273_; lean_object* v___f_274_; lean_object* v___x_276_; 
v___f_273_ = ((lean_object*)(lp_mathlib_AlgEquiv_uniqueProd___redArg___closed__0));
v___f_274_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_uniqueProd___redArg___lam__1), 2, 1);
lean_closure_set(v___f_274_, 0, v_toZero_269_);
if (v_isShared_272_ == 0)
{
lean_ctor_set(v___x_271_, 1, v___f_274_);
lean_ctor_set(v___x_271_, 0, v___f_273_);
v___x_276_ = v___x_271_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v___f_273_);
lean_ctor_set(v_reuseFailAlloc_277_, 1, v___f_274_);
v___x_276_ = v_reuseFailAlloc_277_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
return v___x_276_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd(lean_object* v_R_280_, lean_object* v_A_281_, lean_object* v_B_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lp_mathlib_AlgEquiv_uniqueProd___redArg(v_inst_286_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_uniqueProd___boxed(lean_object* v_R_290_, lean_object* v_A_291_, lean_object* v_B_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_AlgEquiv_uniqueProd(v_R_290_, v_A_291_, v_B_292_, v_inst_293_, v_inst_294_, v_inst_295_, v_inst_296_, v_inst_297_, v_inst_298_);
lean_dec(v_inst_298_);
lean_dec_ref(v_inst_297_);
lean_dec_ref(v_inst_295_);
lean_dec_ref(v_inst_294_);
lean_dec_ref(v_inst_293_);
return v_res_299_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Prod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Prod(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Prod(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
