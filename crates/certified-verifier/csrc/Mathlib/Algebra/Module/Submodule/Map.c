// Lean compiler output
// Module: Mathlib.Algebra.Module.Submodule.Map
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Map public import Mathlib.Algebra.Module.Submodule.Basic public import Mathlib.Algebra.Module.Submodule.Lattice public import Mathlib.Algebra.Module.Submodule.LinearMap
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
lean_object* lp_mathlib_LinearMap_restrict___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_domRestrict___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_submoduleOfEquivOfLe___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__0 = (const lean_object*)&lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__0_value;
static const lean_ctor_object lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__0_value),((lean_object*)&lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__0_value)}};
static const lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__1 = (const lean_object*)&lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_giMapComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_giMapComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gciMapComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gciMapComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComapOfBijective___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComapOfBijective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_comapSubtypeEquivOfLe___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__0 = (const lean_object*)&lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__0_value;
static const lean_ctor_object lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__0_value),((lean_object*)&lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__0_value)}};
static const lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__1 = (const lean_object*)&lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_compatibleMaps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_compatibleMaps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleComap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleComap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_submoduleMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_submoduleMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_submoduleMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_submoduleMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map(lean_object* v_R_1_, lean_object* v_R_u2082_2_, lean_object* v_M_3_, lean_object* v_M_u2082_4_, lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_00_u03c3_u2081_u2082_11_, lean_object* v_inst_12_, lean_object* v_f_13_, lean_object* v_p_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map___boxed(lean_object* v_R_16_, lean_object* v_R_u2082_17_, lean_object* v_M_18_, lean_object* v_M_u2082_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_00_u03c3_u2081_u2082_26_, lean_object* v_inst_27_, lean_object* v_f_28_, lean_object* v_p_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Submodule_map(v_R_16_, v_R_u2082_17_, v_M_18_, v_M_u2082_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_inst_23_, v_inst_24_, v_inst_25_, v_00_u03c3_u2081_u2082_26_, v_inst_27_, v_f_28_, v_p_29_);
lean_dec(v_f_28_);
lean_dec(v_00_u03c3_u2081_u2082_26_);
lean_dec(v_inst_25_);
lean_dec(v_inst_24_);
lean_dec_ref(v_inst_23_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
lean_dec_ref(v_inst_20_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comap(lean_object* v_R_31_, lean_object* v_R_u2082_32_, lean_object* v_M_33_, lean_object* v_M_u2082_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_00_u03c3_u2081_u2082_41_, lean_object* v_f_42_, lean_object* v_p_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lean_box(0);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comap___boxed(lean_object* v_R_45_, lean_object* v_R_u2082_46_, lean_object* v_M_47_, lean_object* v_M_u2082_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_00_u03c3_u2081_u2082_55_, lean_object* v_f_56_, lean_object* v_p_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_Submodule_comap(v_R_45_, v_R_u2082_46_, v_M_47_, v_M_u2082_48_, v_inst_49_, v_inst_50_, v_inst_51_, v_inst_52_, v_inst_53_, v_inst_54_, v_00_u03c3_u2081_u2082_55_, v_f_56_, v_p_57_);
lean_dec(v_f_56_);
lean_dec(v_00_u03c3_u2081_u2082_55_);
lean_dec(v_inst_54_);
lean_dec(v_inst_53_);
lean_dec_ref(v_inst_52_);
lean_dec_ref(v_inst_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOf(lean_object* v_R_59_, lean_object* v_M_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_p_64_, lean_object* v_q_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lean_box(0);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOf___boxed(lean_object* v_R_67_, lean_object* v_M_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_p_72_, lean_object* v_q_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_Submodule_submoduleOf(v_R_67_, v_M_68_, v_inst_69_, v_inst_70_, v_inst_71_, v_p_72_, v_q_73_);
lean_dec(v_inst_71_);
lean_dec_ref(v_inst_70_);
lean_dec_ref(v_inst_69_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe___lam__0(lean_object* v_m_75_){
_start:
{
lean_inc(v_m_75_);
return v_m_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe___lam__0___boxed(lean_object* v_m_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_Submodule_submoduleOfEquivOfLe___lam__0(v_m_76_);
lean_dec(v_m_76_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe(lean_object* v_R_81_, lean_object* v_M_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_p_86_, lean_object* v_q_87_, lean_object* v_h_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = ((lean_object*)(lp_mathlib_Submodule_submoduleOfEquivOfLe___closed__1));
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_submoduleOfEquivOfLe___boxed(lean_object* v_R_90_, lean_object* v_M_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_p_95_, lean_object* v_q_96_, lean_object* v_h_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_Submodule_submoduleOfEquivOfLe(v_R_90_, v_M_91_, v_inst_92_, v_inst_93_, v_inst_94_, v_p_95_, v_q_96_, v_h_97_);
lean_dec(v_inst_94_);
lean_dec_ref(v_inst_93_);
lean_dec_ref(v_inst_92_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_giMapComap___redArg(lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_00_u03c3_u2081_u2082_105_, lean_object* v_f_106_){
_start:
{
lean_object* v___x_107_; lean_object* v___f_108_; 
v___x_107_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_map___boxed), 14, 13);
lean_closure_set(v___x_107_, 0, lean_box(0));
lean_closure_set(v___x_107_, 1, lean_box(0));
lean_closure_set(v___x_107_, 2, lean_box(0));
lean_closure_set(v___x_107_, 3, lean_box(0));
lean_closure_set(v___x_107_, 4, v_inst_99_);
lean_closure_set(v___x_107_, 5, v_inst_100_);
lean_closure_set(v___x_107_, 6, v_inst_101_);
lean_closure_set(v___x_107_, 7, v_inst_102_);
lean_closure_set(v___x_107_, 8, v_inst_103_);
lean_closure_set(v___x_107_, 9, v_inst_104_);
lean_closure_set(v___x_107_, 10, v_00_u03c3_u2081_u2082_105_);
lean_closure_set(v___x_107_, 11, lean_box(0));
lean_closure_set(v___x_107_, 12, v_f_106_);
v___f_108_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_108_, 0, v___x_107_);
return v___f_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_giMapComap(lean_object* v_R_109_, lean_object* v_R_u2082_110_, lean_object* v_M_111_, lean_object* v_M_u2082_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_00_u03c3_u2081_u2082_119_, lean_object* v_inst_120_, lean_object* v_f_121_, lean_object* v_hf_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_Submodule_giMapComap___redArg(v_inst_113_, v_inst_114_, v_inst_115_, v_inst_116_, v_inst_117_, v_inst_118_, v_00_u03c3_u2081_u2082_119_, v_f_121_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gciMapComap___redArg(lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_00_u03c3_u2081_u2082_130_, lean_object* v_f_131_){
_start:
{
lean_object* v___x_132_; lean_object* v___f_133_; 
v___x_132_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_comap___boxed), 13, 12);
lean_closure_set(v___x_132_, 0, lean_box(0));
lean_closure_set(v___x_132_, 1, lean_box(0));
lean_closure_set(v___x_132_, 2, lean_box(0));
lean_closure_set(v___x_132_, 3, lean_box(0));
lean_closure_set(v___x_132_, 4, v_inst_124_);
lean_closure_set(v___x_132_, 5, v_inst_125_);
lean_closure_set(v___x_132_, 6, v_inst_126_);
lean_closure_set(v___x_132_, 7, v_inst_127_);
lean_closure_set(v___x_132_, 8, v_inst_128_);
lean_closure_set(v___x_132_, 9, v_inst_129_);
lean_closure_set(v___x_132_, 10, v_00_u03c3_u2081_u2082_130_);
lean_closure_set(v___x_132_, 11, v_f_131_);
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_133_, 0, v___x_132_);
return v___f_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gciMapComap(lean_object* v_R_134_, lean_object* v_R_u2082_135_, lean_object* v_M_136_, lean_object* v_M_u2082_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_00_u03c3_u2081_u2082_144_, lean_object* v_inst_145_, lean_object* v_f_146_, lean_object* v_hf_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lp_mathlib_Submodule_gciMapComap___redArg(v_inst_138_, v_inst_139_, v_inst_140_, v_inst_141_, v_inst_142_, v_inst_143_, v_00_u03c3_u2081_u2082_144_, v_f_146_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComapOfBijective___redArg(lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_00_u03c3_u2081_u2082_155_, lean_object* v_f_156_){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
lean_inc(v_f_156_);
lean_inc(v_00_u03c3_u2081_u2082_155_);
lean_inc(v_inst_154_);
lean_inc(v_inst_153_);
lean_inc_ref(v_inst_152_);
lean_inc_ref(v_inst_151_);
lean_inc_ref(v_inst_150_);
lean_inc_ref(v_inst_149_);
v___x_157_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_map___boxed), 14, 13);
lean_closure_set(v___x_157_, 0, lean_box(0));
lean_closure_set(v___x_157_, 1, lean_box(0));
lean_closure_set(v___x_157_, 2, lean_box(0));
lean_closure_set(v___x_157_, 3, lean_box(0));
lean_closure_set(v___x_157_, 4, v_inst_149_);
lean_closure_set(v___x_157_, 5, v_inst_150_);
lean_closure_set(v___x_157_, 6, v_inst_151_);
lean_closure_set(v___x_157_, 7, v_inst_152_);
lean_closure_set(v___x_157_, 8, v_inst_153_);
lean_closure_set(v___x_157_, 9, v_inst_154_);
lean_closure_set(v___x_157_, 10, v_00_u03c3_u2081_u2082_155_);
lean_closure_set(v___x_157_, 11, lean_box(0));
lean_closure_set(v___x_157_, 12, v_f_156_);
v___x_158_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_comap___boxed), 13, 12);
lean_closure_set(v___x_158_, 0, lean_box(0));
lean_closure_set(v___x_158_, 1, lean_box(0));
lean_closure_set(v___x_158_, 2, lean_box(0));
lean_closure_set(v___x_158_, 3, lean_box(0));
lean_closure_set(v___x_158_, 4, v_inst_149_);
lean_closure_set(v___x_158_, 5, v_inst_150_);
lean_closure_set(v___x_158_, 6, v_inst_151_);
lean_closure_set(v___x_158_, 7, v_inst_152_);
lean_closure_set(v___x_158_, 8, v_inst_153_);
lean_closure_set(v___x_158_, 9, v_inst_154_);
lean_closure_set(v___x_158_, 10, v_00_u03c3_u2081_u2082_155_);
lean_closure_set(v___x_158_, 11, v_f_156_);
v___x_159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_157_);
lean_ctor_set(v___x_159_, 1, v___x_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComapOfBijective(lean_object* v_R_160_, lean_object* v_R_u2082_161_, lean_object* v_M_162_, lean_object* v_M_u2082_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_00_u03c3_u2081_u2082_170_, lean_object* v_inst_171_, lean_object* v_f_172_, lean_object* v_hf_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_Submodule_orderIsoMapComapOfBijective___redArg(v_inst_164_, v_inst_165_, v_inst_166_, v_inst_167_, v_inst_168_, v_inst_169_, v_00_u03c3_u2081_u2082_170_, v_f_172_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComap___redArg(lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_00_u03c3_u2081_u2082_181_, lean_object* v_f_182_){
_start:
{
lean_object* v_toLinearMap_183_; lean_object* v___x_184_; 
v_toLinearMap_183_ = lean_ctor_get(v_f_182_, 0);
lean_inc(v_toLinearMap_183_);
lean_dec_ref(v_f_182_);
v___x_184_ = lp_mathlib_Submodule_orderIsoMapComapOfBijective___redArg(v_inst_175_, v_inst_176_, v_inst_177_, v_inst_178_, v_inst_179_, v_inst_180_, v_00_u03c3_u2081_u2082_181_, v_toLinearMap_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComap(lean_object* v_R_185_, lean_object* v_R_u2082_186_, lean_object* v_M_187_, lean_object* v_M_u2082_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_00_u03c3_u2081_u2082_195_, lean_object* v_inst_196_, lean_object* v_00_u03c3_u2082_u2081_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_f_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_Submodule_orderIsoMapComap___redArg(v_inst_189_, v_inst_190_, v_inst_191_, v_inst_192_, v_inst_193_, v_inst_194_, v_00_u03c3_u2081_u2082_195_, v_f_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_orderIsoMapComap___boxed(lean_object* v_R_202_, lean_object* v_R_u2082_203_, lean_object* v_M_204_, lean_object* v_M_u2082_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_00_u03c3_u2081_u2082_212_, lean_object* v_inst_213_, lean_object* v_00_u03c3_u2082_u2081_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_f_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_Submodule_orderIsoMapComap(v_R_202_, v_R_u2082_203_, v_M_204_, v_M_u2082_205_, v_inst_206_, v_inst_207_, v_inst_208_, v_inst_209_, v_inst_210_, v_inst_211_, v_00_u03c3_u2081_u2082_212_, v_inst_213_, v_00_u03c3_u2082_u2081_214_, v_inst_215_, v_inst_216_, v_f_217_);
lean_dec(v_00_u03c3_u2082_u2081_214_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe___lam__0(lean_object* v_x_219_){
_start:
{
lean_inc(v_x_219_);
return v_x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe___lam__0___boxed(lean_object* v_x_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_Submodule_comapSubtypeEquivOfLe___lam__0(v_x_220_);
lean_dec(v_x_220_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe(lean_object* v_R_225_, lean_object* v_M_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_p_230_, lean_object* v_q_231_, lean_object* v_hpq_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = ((lean_object*)(lp_mathlib_Submodule_comapSubtypeEquivOfLe___closed__1));
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapSubtypeEquivOfLe___boxed(lean_object* v_R_234_, lean_object* v_M_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_p_239_, lean_object* v_q_240_, lean_object* v_hpq_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_Submodule_comapSubtypeEquivOfLe(v_R_234_, v_M_235_, v_inst_236_, v_inst_237_, v_inst_238_, v_p_239_, v_q_240_, v_hpq_241_);
lean_dec(v_inst_238_);
lean_dec_ref(v_inst_237_);
lean_dec_ref(v_inst_236_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_compatibleMaps(lean_object* v_S_243_, lean_object* v_N_244_, lean_object* v_N_u2082_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_p_u2097_251_, lean_object* v_q_u2097_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lean_box(0);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_compatibleMaps___boxed(lean_object* v_S_254_, lean_object* v_N_255_, lean_object* v_N_u2082_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_p_u2097_262_, lean_object* v_q_u2097_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_Submodule_compatibleMaps(v_S_254_, v_N_255_, v_N_u2082_256_, v_inst_257_, v_inst_258_, v_inst_259_, v_inst_260_, v_inst_261_, v_p_u2097_262_, v_q_u2097_263_);
lean_dec(v_inst_261_);
lean_dec(v_inst_260_);
lean_dec_ref(v_inst_259_);
lean_dec_ref(v_inst_258_);
lean_dec_ref(v_inst_257_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleComap___redArg(lean_object* v_f_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lp_mathlib_LinearMap_restrict___redArg(v_f_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleComap(lean_object* v_R_267_, lean_object* v_R_u2082_268_, lean_object* v_M_269_, lean_object* v_M_u2082_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_00_u03c3_u2081_u2082_277_, lean_object* v_f_278_, lean_object* v_q_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lp_mathlib_LinearMap_restrict___redArg(v_f_278_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleComap___boxed(lean_object* v_R_281_, lean_object* v_R_u2082_282_, lean_object* v_M_283_, lean_object* v_M_u2082_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_00_u03c3_u2081_u2082_291_, lean_object* v_f_292_, lean_object* v_q_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_LinearMap_submoduleComap(v_R_281_, v_R_u2082_282_, v_M_283_, v_M_u2082_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_inst_288_, v_inst_289_, v_inst_290_, v_00_u03c3_u2081_u2082_291_, v_f_292_, v_q_293_);
lean_dec(v_00_u03c3_u2081_u2082_291_);
lean_dec(v_inst_290_);
lean_dec(v_inst_289_);
lean_dec_ref(v_inst_288_);
lean_dec_ref(v_inst_287_);
lean_dec_ref(v_inst_286_);
lean_dec_ref(v_inst_285_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleMap___redArg(lean_object* v_f_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_LinearMap_restrict___redArg(v_f_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleMap(lean_object* v_R_297_, lean_object* v_R_u2082_298_, lean_object* v_M_299_, lean_object* v_M_u2082_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_00_u03c3_u2081_u2082_307_, lean_object* v_inst_308_, lean_object* v_f_309_, lean_object* v_p_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_LinearMap_restrict___redArg(v_f_309_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_submoduleMap___boxed(lean_object* v_R_312_, lean_object* v_R_u2082_313_, lean_object* v_M_314_, lean_object* v_M_u2082_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_00_u03c3_u2081_u2082_322_, lean_object* v_inst_323_, lean_object* v_f_324_, lean_object* v_p_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_LinearMap_submoduleMap(v_R_312_, v_R_u2082_313_, v_M_314_, v_M_u2082_315_, v_inst_316_, v_inst_317_, v_inst_318_, v_inst_319_, v_inst_320_, v_inst_321_, v_00_u03c3_u2081_u2082_322_, v_inst_323_, v_f_324_, v_p_325_);
lean_dec(v_00_u03c3_u2081_u2082_322_);
lean_dec(v_inst_321_);
lean_dec(v_inst_320_);
lean_dec_ref(v_inst_319_);
lean_dec_ref(v_inst_318_);
lean_dec_ref(v_inst_317_);
lean_dec_ref(v_inst_316_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_submoduleMap___redArg___lam__0(lean_object* v_e_327_, lean_object* v_y_328_){
_start:
{
lean_object* v___x_329_; lean_object* v_toLinearMap_330_; lean_object* v___x_331_; 
v___x_329_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_327_);
v_toLinearMap_330_ = lean_ctor_get(v___x_329_, 0);
lean_inc(v_toLinearMap_330_);
lean_dec_ref(v___x_329_);
v___x_331_ = lean_apply_1(v_toLinearMap_330_, v_y_328_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_submoduleMap___redArg(lean_object* v_e_332_){
_start:
{
lean_object* v_toLinearMap_333_; lean_object* v___f_334_; lean_object* v___x_335_; lean_object* v___f_336_; lean_object* v___x_337_; 
v_toLinearMap_333_ = lean_ctor_get(v_e_332_, 0);
lean_inc(v_toLinearMap_333_);
v___f_334_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_submoduleMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_334_, 0, v_e_332_);
v___x_335_ = lp_mathlib_LinearMap_domRestrict___redArg(v_toLinearMap_333_);
v___f_336_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_336_, 0, v___x_335_);
v___x_337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_337_, 0, v___f_336_);
lean_ctor_set(v___x_337_, 1, v___f_334_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_submoduleMap(lean_object* v_R_338_, lean_object* v_R_u2082_339_, lean_object* v_M_340_, lean_object* v_M_u2082_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_module__M_346_, lean_object* v_module__M_u2082_347_, lean_object* v_00_u03c3_u2081_u2082_348_, lean_object* v_00_u03c3_u2082_u2081_349_, lean_object* v_re_u2081_u2082_350_, lean_object* v_re_u2082_u2081_351_, lean_object* v_e_352_, lean_object* v_p_353_){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = lp_mathlib_LinearEquiv_submoduleMap___redArg(v_e_352_);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_submoduleMap___boxed(lean_object* v_R_355_, lean_object* v_R_u2082_356_, lean_object* v_M_357_, lean_object* v_M_u2082_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_module__M_363_, lean_object* v_module__M_u2082_364_, lean_object* v_00_u03c3_u2081_u2082_365_, lean_object* v_00_u03c3_u2082_u2081_366_, lean_object* v_re_u2081_u2082_367_, lean_object* v_re_u2082_u2081_368_, lean_object* v_e_369_, lean_object* v_p_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_mathlib_LinearEquiv_submoduleMap(v_R_355_, v_R_u2082_356_, v_M_357_, v_M_u2082_358_, v_inst_359_, v_inst_360_, v_inst_361_, v_inst_362_, v_module__M_363_, v_module__M_u2082_364_, v_00_u03c3_u2081_u2082_365_, v_00_u03c3_u2082_u2081_366_, v_re_u2081_u2082_367_, v_re_u2082_u2081_368_, v_e_369_, v_p_370_);
lean_dec(v_00_u03c3_u2082_u2081_366_);
lean_dec(v_00_u03c3_u2081_u2082_365_);
lean_dec(v_module__M_u2082_364_);
lean_dec(v_module__M_363_);
lean_dec_ref(v_inst_362_);
lean_dec_ref(v_inst_361_);
lean_dec_ref(v_inst_360_);
lean_dec_ref(v_inst_359_);
return v_res_371_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_LinearMap(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Map(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_LinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Map(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_LinearMap(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Map(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_LinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Submodule_Map(builtin);
}
#ifdef __cplusplus
}
#endif
