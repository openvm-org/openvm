// Lean compiler output
// Module: Mathlib.Algebra.DirectSum.Module
// Imports: public import Init public meta import Init public import Mathlib.Algebra.DirectSum.Basic public import Mathlib.LinearAlgebra.DFinsupp public import Mathlib.LinearAlgebra.Basis.Defs
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
lean_object* lp_mathlib_DFinsupp_mapRange_linearMap___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_DirectSum_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_domLCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_lsum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_lsingle___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_instAddCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DirectSum_id___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SMulMemClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_DFinsupp_lapply___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_sigmaUncurry___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_linearEquivFunOnFintype___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_toAddEquiv___redArg(lean_object*);
lean_object* lp_mathlib_DirectSum_sigmaCurry___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_coeFnLinearMap___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_lmk___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModule___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DirectSum_coeFnLinearMap___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_coeFnLinearMap___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum_coeFnLinearMap___closed__0 = (const lean_object*)&lp_mathlib_DirectSum_coeFnLinearMap___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmk___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lof___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toModule___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_lsetToSet___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_linearEquivFunOnFintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_linearEquivFunOnFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_linearEquivFunOnFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_component___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_component(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_component___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lequivCongrLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lequivCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lequivCongrLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurry___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLuncurry___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLuncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLuncurry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurryEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurryEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurryEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DirectSum_coeLinearMap___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DirectSum_coeLinearMap___redArg___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___closed__0 = (const lean_object*)&lp_mathlib_DirectSum_coeLinearMap___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrLinearEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrLinearEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModule___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed), 8, 6);
lean_closure_set(v___x_4_, 0, lean_box(0));
lean_closure_set(v___x_4_, 1, lean_box(0));
lean_closure_set(v___x_4_, 2, lean_box(0));
lean_closure_set(v___x_4_, 3, v_inst_1_);
lean_closure_set(v___x_4_, 4, v_inst_2_);
lean_closure_set(v___x_4_, 5, v_inst_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModule(lean_object* v_R_5_, lean_object* v_inst_6_, lean_object* v_00_u03b9_7_, lean_object* v_M_8_, lean_object* v_inst_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed), 8, 6);
lean_closure_set(v___x_11_, 0, lean_box(0));
lean_closure_set(v___x_11_, 1, lean_box(0));
lean_closure_set(v___x_11_, 2, lean_box(0));
lean_closure_set(v___x_11_, 3, v_inst_6_);
lean_closure_set(v___x_11_, 4, v_inst_9_);
lean_closure_set(v___x_11_, 5, v_inst_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnLinearMap(lean_object* v_R_13_, lean_object* v_inst_14_, lean_object* v_00_u03b9_15_, lean_object* v_M_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___f_19_; 
v___f_19_ = ((lean_object*)(lp_mathlib_DirectSum_coeFnLinearMap___closed__0));
return v___f_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeFnLinearMap___boxed(lean_object* v_R_20_, lean_object* v_inst_21_, lean_object* v_00_u03b9_22_, lean_object* v_M_23_, lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_DirectSum_coeFnLinearMap(v_R_20_, v_inst_21_, v_00_u03b9_22_, v_M_23_, v_inst_24_, v_inst_25_);
lean_dec(v_inst_25_);
lean_dec_ref(v_inst_24_);
lean_dec_ref(v_inst_21_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmk___redArg(lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_s_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_DFinsupp_lmk___redArg(v_inst_27_, v_inst_28_, v_s_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmk(lean_object* v_R_31_, lean_object* v_inst_32_, lean_object* v_00_u03b9_33_, lean_object* v_M_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_s_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_DFinsupp_lmk___redArg(v_inst_35_, v_inst_37_, v_s_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmk___boxed(lean_object* v_R_40_, lean_object* v_inst_41_, lean_object* v_00_u03b9_42_, lean_object* v_M_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_s_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_DirectSum_lmk(v_R_40_, v_inst_41_, v_00_u03b9_42_, v_M_43_, v_inst_44_, v_inst_45_, v_inst_46_, v_s_47_);
lean_dec(v_inst_45_);
lean_dec_ref(v_inst_41_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lof___redArg(lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_i_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_DFinsupp_lsingle___redArg(v_inst_49_, v_inst_50_, v_i_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lof(lean_object* v_R_53_, lean_object* v_inst_54_, lean_object* v_00_u03b9_55_, lean_object* v_M_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_i_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_DFinsupp_lsingle___redArg(v_inst_57_, v_inst_59_, v_i_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lof___boxed(lean_object* v_R_62_, lean_object* v_inst_63_, lean_object* v_00_u03b9_64_, lean_object* v_M_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_i_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_DirectSum_lof(v_R_62_, v_inst_63_, v_00_u03b9_64_, v_M_65_, v_inst_66_, v_inst_67_, v_inst_68_, v_i_69_);
lean_dec(v_inst_67_);
lean_dec_ref(v_inst_63_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toModule___redArg(lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_00_u03c6_74_){
_start:
{
lean_object* v___x_75_; lean_object* v_toLinearMap_76_; lean_object* v___x_77_; 
v___x_75_ = lp_mathlib_DFinsupp_lsum___redArg(v_inst_71_, v_inst_73_, v_inst_72_);
v_toLinearMap_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_toLinearMap_76_);
lean_dec_ref(v___x_75_);
v___x_77_ = lean_apply_1(v_toLinearMap_76_, v_00_u03c6_74_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toModule(lean_object* v_R_78_, lean_object* v_inst_79_, lean_object* v_00_u03b9_80_, lean_object* v_M_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_N_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_00_u03c6_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_DirectSum_toModule___redArg(v_inst_82_, v_inst_84_, v_inst_86_, v_00_u03c6_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toModule___boxed(lean_object* v_R_90_, lean_object* v_inst_91_, lean_object* v_00_u03b9_92_, lean_object* v_M_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_N_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_00_u03c6_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_DirectSum_toModule(v_R_90_, v_inst_91_, v_00_u03b9_92_, v_M_93_, v_inst_94_, v_inst_95_, v_inst_96_, v_N_97_, v_inst_98_, v_inst_99_, v_00_u03c6_100_);
lean_dec(v_inst_99_);
lean_dec(v_inst_95_);
lean_dec_ref(v_inst_91_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___redArg___lam__0(lean_object* v_inst_102_, lean_object* v_i_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lean_apply_1(v_inst_102_, v_i_103_);
return v___x_104_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DirectSum_lsetToSet___redArg___lam__1(lean_object* v_inst_105_, lean_object* v_a_106_, lean_object* v_b_107_){
_start:
{
lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_108_ = lean_apply_2(v_inst_105_, v_a_106_, v_b_107_);
v___x_109_ = lean_unbox(v___x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___redArg___lam__1___boxed(lean_object* v_inst_110_, lean_object* v_a_111_, lean_object* v_b_112_){
_start:
{
uint8_t v_res_113_; lean_object* v_r_114_; 
v_res_113_ = lp_mathlib_DirectSum_lsetToSet___redArg___lam__1(v_inst_110_, v_a_111_, v_b_112_);
v_r_114_ = lean_box(v_res_113_);
return v_r_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___redArg___lam__4(lean_object* v___f_115_, lean_object* v___f_116_, lean_object* v_i_117_, lean_object* v___y_118_){
_start:
{
lean_object* v___x_57__overap_119_; lean_object* v___x_120_; 
v___x_57__overap_119_ = lp_mathlib_DFinsupp_lsingle___redArg(v___f_115_, v___f_116_, v_i_117_);
v___x_120_ = lean_apply_1(v___x_57__overap_119_, v___y_118_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___redArg(lean_object* v_inst_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v___f_123_; lean_object* v___f_124_; lean_object* v___f_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___f_123_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_lsetToSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_123_, 0, v_inst_121_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_lsetToSet___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_124_, 0, v_inst_122_);
lean_inc_ref(v___f_124_);
lean_inc_ref_n(v___f_123_, 2);
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_lsetToSet___redArg___lam__4), 4, 2);
lean_closure_set(v___f_125_, 0, v___f_123_);
lean_closure_set(v___f_125_, 1, v___f_124_);
v___x_126_ = lp_mathlib_DirectSum_instAddCommMonoid___redArg(v___f_123_);
v___x_127_ = lp_mathlib_DirectSum_toModule___redArg(v___f_123_, v___f_124_, v___x_126_, v___f_125_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet(lean_object* v_R_128_, lean_object* v_inst_129_, lean_object* v_00_u03b9_130_, lean_object* v_M_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_S_135_, lean_object* v_T_136_, lean_object* v_H_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_DirectSum_lsetToSet___redArg(v_inst_132_, v_inst_134_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lsetToSet___boxed(lean_object* v_R_139_, lean_object* v_inst_140_, lean_object* v_00_u03b9_141_, lean_object* v_M_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_S_146_, lean_object* v_T_147_, lean_object* v_H_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_DirectSum_lsetToSet(v_R_139_, v_inst_140_, v_00_u03b9_141_, v_M_142_, v_inst_143_, v_inst_144_, v_inst_145_, v_S_146_, v_T_147_, v_H_148_);
lean_dec(v_inst_144_);
lean_dec_ref(v_inst_140_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_linearEquivFunOnFintype___redArg(lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_DFinsupp_linearEquivFunOnFintype___redArg(v_inst_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_linearEquivFunOnFintype(lean_object* v_R_152_, lean_object* v_inst_153_, lean_object* v_00_u03b9_154_, lean_object* v_M_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_DFinsupp_linearEquivFunOnFintype___redArg(v_inst_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_linearEquivFunOnFintype___boxed(lean_object* v_R_160_, lean_object* v_inst_161_, lean_object* v_00_u03b9_162_, lean_object* v_M_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_DirectSum_linearEquivFunOnFintype(v_R_160_, v_inst_161_, v_00_u03b9_162_, v_M_163_, v_inst_164_, v_inst_165_, v_inst_166_);
lean_dec(v_inst_165_);
lean_dec_ref(v_inst_164_);
lean_dec_ref(v_inst_161_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lid___redArg(lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v___x_170_; lean_object* v_toFun_171_; lean_object* v_invFun_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_179_; 
v___x_170_ = lp_mathlib_DirectSum_id___redArg(v_inst_168_, v_inst_169_);
v_toFun_171_ = lean_ctor_get(v___x_170_, 0);
v_invFun_172_ = lean_ctor_get(v___x_170_, 1);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_170_);
if (v_isSharedCheck_179_ == 0)
{
v___x_174_ = v___x_170_;
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_invFun_172_);
lean_inc(v_toFun_171_);
lean_dec(v___x_170_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_177_; 
if (v_isShared_175_ == 0)
{
v___x_177_ = v___x_174_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v_toFun_171_);
lean_ctor_set(v_reuseFailAlloc_178_, 1, v_invFun_172_);
v___x_177_ = v_reuseFailAlloc_178_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
return v___x_177_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lid(lean_object* v_R_180_, lean_object* v_inst_181_, lean_object* v_M_182_, lean_object* v_00_u03b9_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_DirectSum_lid___redArg(v_inst_184_, v_inst_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lid___boxed(lean_object* v_R_188_, lean_object* v_inst_189_, lean_object* v_M_190_, lean_object* v_00_u03b9_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_inst_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_DirectSum_lid(v_R_188_, v_inst_189_, v_M_190_, v_00_u03b9_191_, v_inst_192_, v_inst_193_, v_inst_194_);
lean_dec(v_inst_193_);
lean_dec_ref(v_inst_189_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_component___redArg(lean_object* v_i_196_){
_start:
{
lean_object* v___f_197_; 
v___f_197_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lapply___redArg___lam__0), 2, 1);
lean_closure_set(v___f_197_, 0, v_i_196_);
return v___f_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_component(lean_object* v_R_198_, lean_object* v_inst_199_, lean_object* v_00_u03b9_200_, lean_object* v_M_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_i_204_){
_start:
{
lean_object* v___f_205_; 
v___f_205_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lapply___redArg___lam__0), 2, 1);
lean_closure_set(v___f_205_, 0, v_i_204_);
return v___f_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_component___boxed(lean_object* v_R_206_, lean_object* v_inst_207_, lean_object* v_00_u03b9_208_, lean_object* v_M_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_i_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_DirectSum_component(v_R_206_, v_inst_207_, v_00_u03b9_208_, v_M_209_, v_inst_210_, v_inst_211_, v_i_212_);
lean_dec(v_inst_211_);
lean_dec_ref(v_inst_210_);
lean_dec_ref(v_inst_207_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmap___redArg(lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_f_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lp_mathlib_DFinsupp_mapRange_linearMap___redArg(v_inst_214_, v_inst_215_, v_f_216_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmap(lean_object* v_R_218_, lean_object* v_inst_219_, lean_object* v_00_u03b9_220_, lean_object* v_M_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_N_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_f_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib_DFinsupp_mapRange_linearMap___redArg(v_inst_222_, v_inst_225_, v_f_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lmap___boxed(lean_object* v_R_229_, lean_object* v_inst_230_, lean_object* v_00_u03b9_231_, lean_object* v_M_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_N_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_f_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_DirectSum_lmap(v_R_229_, v_inst_230_, v_00_u03b9_231_, v_M_232_, v_inst_233_, v_inst_234_, v_N_235_, v_inst_236_, v_inst_237_, v_f_238_);
lean_dec(v_inst_237_);
lean_dec(v_inst_234_);
lean_dec_ref(v_inst_230_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lequivCongrLeft___redArg(lean_object* v_inst_240_, lean_object* v_h_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_DFinsupp_domLCongr___redArg(v_inst_240_, v_h_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lequivCongrLeft(lean_object* v_R_243_, lean_object* v_inst_244_, lean_object* v_00_u03b9_245_, lean_object* v_M_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_00_u03ba_249_, lean_object* v_h_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lp_mathlib_DFinsupp_domLCongr___redArg(v_inst_247_, v_h_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_lequivCongrLeft___boxed(lean_object* v_R_252_, lean_object* v_inst_253_, lean_object* v_00_u03b9_254_, lean_object* v_M_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_00_u03ba_258_, lean_object* v_h_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_DirectSum_lequivCongrLeft(v_R_252_, v_inst_253_, v_00_u03b9_254_, v_M_255_, v_inst_256_, v_inst_257_, v_00_u03ba_258_, v_h_259_);
lean_dec(v_inst_257_);
lean_dec_ref(v_inst_253_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurry___redArg(lean_object* v_inst_261_, lean_object* v_inst_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_mathlib_DirectSum_sigmaCurry___redArg(v_inst_261_, v_inst_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurry(lean_object* v_R_264_, lean_object* v_inst_265_, lean_object* v_00_u03b9_266_, lean_object* v_00_u03b1_267_, lean_object* v_00_u03b4_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_inst_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_mathlib_DirectSum_sigmaCurry___redArg(v_inst_269_, v_inst_270_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurry___boxed(lean_object* v_R_273_, lean_object* v_inst_274_, lean_object* v_00_u03b9_275_, lean_object* v_00_u03b1_276_, lean_object* v_00_u03b4_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_DirectSum_sigmaLcurry(v_R_273_, v_inst_274_, v_00_u03b9_275_, v_00_u03b1_276_, v_00_u03b4_277_, v_inst_278_, v_inst_279_, v_inst_280_);
lean_dec(v_inst_280_);
lean_dec_ref(v_inst_274_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLuncurry___redArg(lean_object* v_inst_282_, lean_object* v_inst_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lp_mathlib_DirectSum_sigmaUncurry___redArg(v_inst_282_, v_inst_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLuncurry(lean_object* v_R_285_, lean_object* v_inst_286_, lean_object* v_00_u03b9_287_, lean_object* v_00_u03b1_288_, lean_object* v_00_u03b4_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lp_mathlib_DirectSum_sigmaUncurry___redArg(v_inst_290_, v_inst_291_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLuncurry___boxed(lean_object* v_R_294_, lean_object* v_inst_295_, lean_object* v_00_u03b9_296_, lean_object* v_00_u03b1_297_, lean_object* v_00_u03b4_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_mathlib_DirectSum_sigmaLuncurry(v_R_294_, v_inst_295_, v_00_u03b9_296_, v_00_u03b1_297_, v_00_u03b4_298_, v_inst_299_, v_inst_300_, v_inst_301_);
lean_dec(v_inst_301_);
lean_dec_ref(v_inst_295_);
return v_res_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurryEquiv___redArg(lean_object* v_inst_303_, lean_object* v_inst_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg(v_inst_303_, v_inst_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurryEquiv(lean_object* v_R_306_, lean_object* v_inst_307_, lean_object* v_00_u03b9_308_, lean_object* v_00_u03b1_309_, lean_object* v_00_u03b4_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_inst_313_){
_start:
{
lean_object* v___x_314_; 
v___x_314_ = lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg(v_inst_311_, v_inst_312_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_sigmaLcurryEquiv___boxed(lean_object* v_R_315_, lean_object* v_inst_316_, lean_object* v_00_u03b9_317_, lean_object* v_00_u03b1_318_, lean_object* v_00_u03b4_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_DirectSum_sigmaLcurryEquiv(v_R_315_, v_inst_316_, v_00_u03b9_317_, v_00_u03b1_318_, v_00_u03b4_319_, v_inst_320_, v_inst_321_, v_inst_322_);
lean_dec(v_inst_322_);
lean_dec_ref(v_inst_316_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___lam__0(lean_object* v_inst_324_, lean_object* v_i_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_324_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___lam__0___boxed(lean_object* v_inst_327_, lean_object* v_i_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_DirectSum_coeLinearMap___redArg___lam__0(v_inst_327_, v_i_328_);
lean_dec(v_i_328_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___lam__1(lean_object* v_i_330_, lean_object* v___y_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lp_mathlib_SMulMemClass_subtype___lam__0(v___y_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg___lam__1___boxed(lean_object* v_i_333_, lean_object* v___y_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_DirectSum_coeLinearMap___redArg___lam__1(v_i_333_, v___y_334_);
lean_dec(v___y_334_);
lean_dec(v_i_333_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___redArg(lean_object* v_dec___u03b9_337_, lean_object* v_inst_338_){
_start:
{
lean_object* v___f_339_; lean_object* v___f_340_; lean_object* v___x_341_; 
lean_inc_ref(v_inst_338_);
v___f_339_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_coeLinearMap___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_339_, 0, v_inst_338_);
v___f_340_ = ((lean_object*)(lp_mathlib_DirectSum_coeLinearMap___redArg___closed__0));
v___x_341_ = lp_mathlib_DirectSum_toModule___redArg(v___f_339_, v_dec___u03b9_337_, v_inst_338_, v___f_340_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap(lean_object* v_R_342_, lean_object* v_inst_343_, lean_object* v_00_u03b9_344_, lean_object* v_dec___u03b9_345_, lean_object* v_M_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_A_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lp_mathlib_DirectSum_coeLinearMap___redArg(v_dec___u03b9_345_, v_inst_347_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_coeLinearMap___boxed(lean_object* v_R_351_, lean_object* v_inst_352_, lean_object* v_00_u03b9_353_, lean_object* v_dec___u03b9_354_, lean_object* v_M_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_A_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_mathlib_DirectSum_coeLinearMap(v_R_351_, v_inst_352_, v_00_u03b9_353_, v_dec___u03b9_354_, v_M_355_, v_inst_356_, v_inst_357_, v_A_358_);
lean_dec_ref(v_A_358_);
lean_dec(v_inst_357_);
lean_dec_ref(v_inst_352_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__0(lean_object* v_u_360_, lean_object* v_i_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v_toFun_365_; lean_object* v___x_366_; 
v___x_363_ = lean_apply_1(v_u_360_, v_i_361_);
v___x_364_ = lp_mathlib_Equiv_symm___redArg(v___x_363_);
v_toFun_365_ = lean_ctor_get(v___x_364_, 0);
lean_inc(v_toFun_365_);
lean_dec_ref(v___x_364_);
v___x_366_ = lean_apply_1(v_toFun_365_, v___y_362_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__1(lean_object* v_u_367_, lean_object* v_i_368_, lean_object* v___y_369_){
_start:
{
lean_object* v___x_370_; lean_object* v_toFun_371_; lean_object* v___x_372_; 
v___x_370_ = lean_apply_1(v_u_367_, v_i_368_);
v_toFun_371_ = lean_ctor_get(v___x_370_, 0);
lean_inc(v_toFun_371_);
lean_dec_ref(v___x_370_);
v___x_372_ = lean_apply_1(v_toFun_371_, v___y_369_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__2(lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v___f_375_, lean_object* v___y_376_){
_start:
{
lean_object* v___x_94__overap_377_; lean_object* v___x_378_; 
v___x_94__overap_377_ = lp_mathlib_DirectSum_map___redArg(v_inst_373_, v_inst_374_, v___f_375_);
v___x_378_ = lean_apply_1(v___x_94__overap_377_, v___y_376_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv___redArg(lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_u_381_){
_start:
{
lean_object* v___f_382_; lean_object* v___f_383_; lean_object* v___f_384_; lean_object* v___f_385_; lean_object* v___x_386_; 
lean_inc_ref(v_u_381_);
v___f_382_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_382_, 0, v_u_381_);
v___f_383_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__1), 3, 1);
lean_closure_set(v___f_383_, 0, v_u_381_);
lean_inc_ref(v_inst_380_);
lean_inc_ref(v_inst_379_);
v___f_384_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__2), 4, 3);
lean_closure_set(v___f_384_, 0, v_inst_379_);
lean_closure_set(v___f_384_, 1, v_inst_380_);
lean_closure_set(v___f_384_, 2, v___f_383_);
v___f_385_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_congrAddEquiv___redArg___lam__2), 4, 3);
lean_closure_set(v___f_385_, 0, v_inst_380_);
lean_closure_set(v___f_385_, 1, v_inst_379_);
lean_closure_set(v___f_385_, 2, v___f_382_);
v___x_386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_386_, 0, v___f_384_);
lean_ctor_set(v___x_386_, 1, v___f_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrAddEquiv(lean_object* v_00_u03b9_387_, lean_object* v_N_388_, lean_object* v_inst_389_, lean_object* v_P_390_, lean_object* v_inst_391_, lean_object* v_u_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_DirectSum_congrAddEquiv___redArg(v_inst_389_, v_inst_391_, v_u_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrLinearEquiv___redArg___lam__0(lean_object* v_u_394_, lean_object* v_i_395_){
_start:
{
lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_396_ = lean_apply_1(v_u_394_, v_i_395_);
v___x_397_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v___x_396_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrLinearEquiv___redArg(lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_u_400_){
_start:
{
lean_object* v___f_401_; lean_object* v___x_402_; lean_object* v_toFun_403_; lean_object* v_invFun_404_; lean_object* v___x_406_; uint8_t v_isShared_407_; uint8_t v_isSharedCheck_411_; 
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_congrLinearEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_401_, 0, v_u_400_);
v___x_402_ = lp_mathlib_DirectSum_congrAddEquiv___redArg(v_inst_398_, v_inst_399_, v___f_401_);
v_toFun_403_ = lean_ctor_get(v___x_402_, 0);
v_invFun_404_ = lean_ctor_get(v___x_402_, 1);
v_isSharedCheck_411_ = !lean_is_exclusive(v___x_402_);
if (v_isSharedCheck_411_ == 0)
{
v___x_406_ = v___x_402_;
v_isShared_407_ = v_isSharedCheck_411_;
goto v_resetjp_405_;
}
else
{
lean_inc(v_invFun_404_);
lean_inc(v_toFun_403_);
lean_dec(v___x_402_);
v___x_406_ = lean_box(0);
v_isShared_407_ = v_isSharedCheck_411_;
goto v_resetjp_405_;
}
v_resetjp_405_:
{
lean_object* v___x_409_; 
if (v_isShared_407_ == 0)
{
v___x_409_ = v___x_406_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_410_; 
v_reuseFailAlloc_410_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_410_, 0, v_toFun_403_);
lean_ctor_set(v_reuseFailAlloc_410_, 1, v_invFun_404_);
v___x_409_ = v_reuseFailAlloc_410_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
return v___x_409_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrLinearEquiv(lean_object* v_R_412_, lean_object* v_inst_413_, lean_object* v_00_u03b9_414_, lean_object* v_N_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_P_418_, lean_object* v_inst_419_, lean_object* v_inst_420_, lean_object* v_u_421_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lp_mathlib_DirectSum_congrLinearEquiv___redArg(v_inst_416_, v_inst_419_, v_u_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_congrLinearEquiv___boxed(lean_object* v_R_423_, lean_object* v_inst_424_, lean_object* v_00_u03b9_425_, lean_object* v_N_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_P_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_u_432_){
_start:
{
lean_object* v_res_433_; 
v_res_433_ = lp_mathlib_DirectSum_congrLinearEquiv(v_R_423_, v_inst_424_, v_00_u03b9_425_, v_N_426_, v_inst_427_, v_inst_428_, v_P_429_, v_inst_430_, v_inst_431_, v_u_432_);
lean_dec(v_inst_431_);
lean_dec(v_inst_428_);
lean_dec_ref(v_inst_424_);
return v_res_433_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
}
#ifdef __cplusplus
}
#endif
