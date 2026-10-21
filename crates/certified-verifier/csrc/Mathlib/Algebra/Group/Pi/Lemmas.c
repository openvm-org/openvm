// Lean compiler output
// Module: Mathlib.Algebra.Group.Pi.Lemmas
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Commute.Defs public import Mathlib.Algebra.Group.Hom.Instances public import Mathlib.Algebra.Group.Pi.Basic public import Mathlib.Algebra.Group.SelfInv public import Mathlib.Data.Set.Piecewise public import Mathlib.Logic.Pairwise import Mathlib.Util.Delaborators public import Mathlib.Util.Delaborators
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
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Pi_mulSingle___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_single___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_pi___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_piMap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_piMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_piMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_piMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_piMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_piMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_piMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Pi_constMulHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_const___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Pi_constMulHom___closed__0 = (const lean_object*)&lp_mathlib_Pi_constMulHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Pi_constMulHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constMulHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constAddHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constAddHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coeFn___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulHom_coeFn___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulHom_coeFn___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulHom_coeFn___closed__0 = (const lean_object*)&lp_mathlib_MulHom_coeFn___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coeFn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coeFn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coeFn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coeFn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_compLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_compLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_compLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_compLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_compLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_compLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_compLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_piMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_piMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_piMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_piMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_piMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_piMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_constAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeFn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeFn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeFn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeFn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_mulSingle___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_mulSingle(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_single___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mulSingle___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mulSingle___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mulSingle(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_single___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_single___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_pi___redArg___lam__0(lean_object* v_g_1_, lean_object* v_x_2_, lean_object* v_i_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_g_1_, v_i_3_, v_x_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_pi___redArg(lean_object* v_g_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_g_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_pi(lean_object* v_I_7_, lean_object* v_f_8_, lean_object* v_inst_9_, lean_object* v_00_u03b3_10_, lean_object* v_inst_11_, lean_object* v_g_12_){
_start:
{
lean_object* v___f_13_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_13_, 0, v_g_12_);
return v___f_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_pi___boxed(lean_object* v_I_14_, lean_object* v_f_15_, lean_object* v_inst_16_, lean_object* v_00_u03b3_17_, lean_object* v_inst_18_, lean_object* v_g_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_MulHom_pi(v_I_14_, v_f_15_, v_inst_16_, v_00_u03b3_17_, v_inst_18_, v_g_19_);
lean_dec(v_inst_18_);
lean_dec(v_inst_16_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_pi___redArg(lean_object* v_g_21_){
_start:
{
lean_object* v___f_22_; 
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_22_, 0, v_g_21_);
return v___f_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_pi(lean_object* v_I_23_, lean_object* v_f_24_, lean_object* v_inst_25_, lean_object* v_00_u03b3_26_, lean_object* v_inst_27_, lean_object* v_g_28_){
_start:
{
lean_object* v___f_29_; 
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_29_, 0, v_g_28_);
return v___f_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_pi___boxed(lean_object* v_I_30_, lean_object* v_f_31_, lean_object* v_inst_32_, lean_object* v_00_u03b3_33_, lean_object* v_inst_34_, lean_object* v_g_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_AddHom_pi(v_I_30_, v_f_31_, v_inst_32_, v_00_u03b3_33_, v_inst_34_, v_g_35_);
lean_dec(v_inst_34_);
lean_dec(v_inst_32_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulHom___redArg(lean_object* v_g_37_){
_start:
{
lean_object* v___f_38_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_38_, 0, v_g_37_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulHom(lean_object* v_I_39_, lean_object* v_f_40_, lean_object* v_inst_41_, lean_object* v_00_u03b3_42_, lean_object* v_inst_43_, lean_object* v_g_44_){
_start:
{
lean_object* v___f_45_; 
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_45_, 0, v_g_44_);
return v___f_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulHom___boxed(lean_object* v_I_46_, lean_object* v_f_47_, lean_object* v_inst_48_, lean_object* v_00_u03b3_49_, lean_object* v_inst_50_, lean_object* v_g_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Pi_mulHom(v_I_46_, v_f_47_, v_inst_48_, v_00_u03b3_49_, v_inst_50_, v_g_51_);
lean_dec(v_inst_50_);
lean_dec(v_inst_48_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addHom___redArg(lean_object* v_g_53_){
_start:
{
lean_object* v___f_54_; 
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_54_, 0, v_g_53_);
return v___f_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addHom(lean_object* v_I_55_, lean_object* v_f_56_, lean_object* v_inst_57_, lean_object* v_00_u03b3_58_, lean_object* v_inst_59_, lean_object* v_g_60_){
_start:
{
lean_object* v___f_61_; 
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_61_, 0, v_g_60_);
return v___f_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addHom___boxed(lean_object* v_I_62_, lean_object* v_f_63_, lean_object* v_inst_64_, lean_object* v_00_u03b3_65_, lean_object* v_inst_66_, lean_object* v_g_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Pi_addHom(v_I_62_, v_f_63_, v_inst_64_, v_00_u03b3_65_, v_inst_66_, v_g_67_);
lean_dec(v_inst_66_);
lean_dec(v_inst_64_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulHom___redArg___lam__0(lean_object* v_i_69_, lean_object* v_g_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lean_apply_1(v_g_70_, v_i_69_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulHom___redArg(lean_object* v_i_72_){
_start:
{
lean_object* v___f_73_; 
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_73_, 0, v_i_72_);
return v___f_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulHom(lean_object* v_I_74_, lean_object* v_f_75_, lean_object* v_inst_76_, lean_object* v_i_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_78_, 0, v_i_77_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMulHom___boxed(lean_object* v_I_79_, lean_object* v_f_80_, lean_object* v_inst_81_, lean_object* v_i_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Pi_evalMulHom(v_I_79_, v_f_80_, v_inst_81_, v_i_82_);
lean_dec(v_inst_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddHom___redArg(lean_object* v_i_84_){
_start:
{
lean_object* v___f_85_; 
v___f_85_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_85_, 0, v_i_84_);
return v___f_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddHom(lean_object* v_I_86_, lean_object* v_f_87_, lean_object* v_inst_88_, lean_object* v_i_89_){
_start:
{
lean_object* v___f_90_; 
v___f_90_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_90_, 0, v_i_89_);
return v___f_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddHom___boxed(lean_object* v_I_91_, lean_object* v_f_92_, lean_object* v_inst_93_, lean_object* v_i_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Pi_evalAddHom(v_I_91_, v_f_92_, v_inst_93_, v_i_94_);
lean_dec(v_inst_93_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_piMap___redArg___lam__0(lean_object* v_g_96_, lean_object* v_i_97_, lean_object* v___y_98_){
_start:
{
lean_object* v___x_99_; lean_object* v___f_100_; lean_object* v___x_101_; 
lean_inc(v_i_97_);
v___x_99_ = lean_apply_1(v_g_96_, v_i_97_);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_100_, 0, v_i_97_);
v___x_101_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_100_, v___x_99_, v___y_98_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_piMap___redArg(lean_object* v_g_102_){
_start:
{
lean_object* v___f_103_; lean_object* v___f_104_; 
v___f_103_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_piMap___redArg___lam__0), 3, 1);
lean_closure_set(v___f_103_, 0, v_g_102_);
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_104_, 0, v___f_103_);
return v___f_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_piMap(lean_object* v_00_u03b9_105_, lean_object* v_M_106_, lean_object* v_N_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_g_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lp_mathlib_MulHom_piMap___redArg(v_g_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_piMap___boxed(lean_object* v_00_u03b9_112_, lean_object* v_M_113_, lean_object* v_N_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_g_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_MulHom_piMap(v_00_u03b9_112_, v_M_113_, v_N_114_, v_inst_115_, v_inst_116_, v_g_117_);
lean_dec(v_inst_116_);
lean_dec(v_inst_115_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_piMap___redArg(lean_object* v_g_119_){
_start:
{
lean_object* v___f_120_; lean_object* v___f_121_; 
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_piMap___redArg___lam__0), 3, 1);
lean_closure_set(v___f_120_, 0, v_g_119_);
v___f_121_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_121_, 0, v___f_120_);
return v___f_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_piMap(lean_object* v_00_u03b9_122_, lean_object* v_M_123_, lean_object* v_N_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_g_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_AddHom_piMap___redArg(v_g_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_piMap___boxed(lean_object* v_00_u03b9_129_, lean_object* v_M_130_, lean_object* v_N_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_g_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_AddHom_piMap(v_00_u03b9_129_, v_M_130_, v_N_131_, v_inst_132_, v_inst_133_, v_g_134_);
lean_dec(v_inst_133_);
lean_dec(v_inst_132_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constMulHom(lean_object* v_00_u03b1_137_, lean_object* v_00_u03b2_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = ((lean_object*)(lp_mathlib_Pi_constMulHom___closed__0));
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constMulHom___boxed(lean_object* v_00_u03b1_141_, lean_object* v_00_u03b2_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Pi_constMulHom(v_00_u03b1_141_, v_00_u03b2_142_, v_inst_143_);
lean_dec(v_inst_143_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constAddHom(lean_object* v_00_u03b1_145_, lean_object* v_00_u03b2_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = ((lean_object*)(lp_mathlib_Pi_constMulHom___closed__0));
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constAddHom___boxed(lean_object* v_00_u03b1_149_, lean_object* v_00_u03b2_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_Pi_constAddHom(v_00_u03b1_149_, v_00_u03b2_150_, v_inst_151_);
lean_dec(v_inst_151_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coeFn___lam__0(lean_object* v_g_153_, lean_object* v___y_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lean_apply_1(v_g_153_, v___y_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coeFn(lean_object* v_00_u03b1_157_, lean_object* v_00_u03b2_158_, lean_object* v_inst_159_, lean_object* v_inst_160_){
_start:
{
lean_object* v___f_161_; 
v___f_161_ = ((lean_object*)(lp_mathlib_MulHom_coeFn___closed__0));
return v___f_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coeFn___boxed(lean_object* v_00_u03b1_162_, lean_object* v_00_u03b2_163_, lean_object* v_inst_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_MulHom_coeFn(v_00_u03b1_162_, v_00_u03b2_163_, v_inst_164_, v_inst_165_);
lean_dec(v_inst_165_);
lean_dec(v_inst_164_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coeFn(lean_object* v_00_u03b1_167_, lean_object* v_00_u03b2_168_, lean_object* v_inst_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___f_171_; 
v___f_171_ = ((lean_object*)(lp_mathlib_MulHom_coeFn___closed__0));
return v___f_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coeFn___boxed(lean_object* v_00_u03b1_172_, lean_object* v_00_u03b2_173_, lean_object* v_inst_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_AddHom_coeFn(v_00_u03b1_172_, v_00_u03b2_173_, v_inst_174_, v_inst_175_);
lean_dec(v_inst_175_);
lean_dec(v_inst_174_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_compLeft___redArg___lam__0(lean_object* v_f_177_, lean_object* v_h_178_, lean_object* v___y_179_){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_180_ = lean_apply_1(v_h_178_, v___y_179_);
v___x_181_ = lean_apply_1(v_f_177_, v___x_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_compLeft___redArg(lean_object* v_f_182_){
_start:
{
lean_object* v___f_183_; 
v___f_183_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_183_, 0, v_f_182_);
return v___f_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_compLeft(lean_object* v_00_u03b1_184_, lean_object* v_00_u03b2_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_f_188_, lean_object* v_I_189_){
_start:
{
lean_object* v___f_190_; 
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_190_, 0, v_f_188_);
return v___f_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_compLeft___boxed(lean_object* v_00_u03b1_191_, lean_object* v_00_u03b2_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_f_195_, lean_object* v_I_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_MulHom_compLeft(v_00_u03b1_191_, v_00_u03b2_192_, v_inst_193_, v_inst_194_, v_f_195_, v_I_196_);
lean_dec(v_inst_194_);
lean_dec(v_inst_193_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_compLeft___redArg(lean_object* v_f_198_){
_start:
{
lean_object* v___f_199_; 
v___f_199_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_199_, 0, v_f_198_);
return v___f_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_compLeft(lean_object* v_00_u03b1_200_, lean_object* v_00_u03b2_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_f_204_, lean_object* v_I_205_){
_start:
{
lean_object* v___f_206_; 
v___f_206_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_206_, 0, v_f_204_);
return v___f_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_compLeft___boxed(lean_object* v_00_u03b1_207_, lean_object* v_00_u03b2_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_f_211_, lean_object* v_I_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_AddHom_compLeft(v_00_u03b1_207_, v_00_u03b2_208_, v_inst_209_, v_inst_210_, v_f_211_, v_I_212_);
lean_dec(v_inst_210_);
lean_dec(v_inst_209_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_pi___redArg(lean_object* v_g_214_){
_start:
{
lean_object* v___f_215_; 
v___f_215_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_215_, 0, v_g_214_);
return v___f_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_pi(lean_object* v_I_216_, lean_object* v_f_217_, lean_object* v_inst_218_, lean_object* v_00_u03b3_219_, lean_object* v_inst_220_, lean_object* v_g_221_){
_start:
{
lean_object* v___f_222_; 
v___f_222_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_222_, 0, v_g_221_);
return v___f_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_pi___boxed(lean_object* v_I_223_, lean_object* v_f_224_, lean_object* v_inst_225_, lean_object* v_00_u03b3_226_, lean_object* v_inst_227_, lean_object* v_g_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_MonoidHom_pi(v_I_223_, v_f_224_, v_inst_225_, v_00_u03b3_226_, v_inst_227_, v_g_228_);
lean_dec_ref(v_inst_227_);
lean_dec_ref(v_inst_225_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_pi___redArg(lean_object* v_g_230_){
_start:
{
lean_object* v___f_231_; 
v___f_231_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_231_, 0, v_g_230_);
return v___f_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_pi(lean_object* v_I_232_, lean_object* v_f_233_, lean_object* v_inst_234_, lean_object* v_00_u03b3_235_, lean_object* v_inst_236_, lean_object* v_g_237_){
_start:
{
lean_object* v___f_238_; 
v___f_238_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_238_, 0, v_g_237_);
return v___f_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_pi___boxed(lean_object* v_I_239_, lean_object* v_f_240_, lean_object* v_inst_241_, lean_object* v_00_u03b3_242_, lean_object* v_inst_243_, lean_object* v_g_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_AddMonoidHom_pi(v_I_239_, v_f_240_, v_inst_241_, v_00_u03b3_242_, v_inst_243_, v_g_244_);
lean_dec_ref(v_inst_243_);
lean_dec_ref(v_inst_241_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHom___redArg(lean_object* v_g_246_){
_start:
{
lean_object* v___f_247_; 
v___f_247_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_247_, 0, v_g_246_);
return v___f_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHom(lean_object* v_I_248_, lean_object* v_f_249_, lean_object* v_inst_250_, lean_object* v_00_u03b3_251_, lean_object* v_inst_252_, lean_object* v_g_253_){
_start:
{
lean_object* v___f_254_; 
v___f_254_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_254_, 0, v_g_253_);
return v___f_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidHom___boxed(lean_object* v_I_255_, lean_object* v_f_256_, lean_object* v_inst_257_, lean_object* v_00_u03b3_258_, lean_object* v_inst_259_, lean_object* v_g_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_Pi_monoidHom(v_I_255_, v_f_256_, v_inst_257_, v_00_u03b3_258_, v_inst_259_, v_g_260_);
lean_dec_ref(v_inst_259_);
lean_dec_ref(v_inst_257_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHom___redArg(lean_object* v_g_262_){
_start:
{
lean_object* v___f_263_; 
v___f_263_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_263_, 0, v_g_262_);
return v___f_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHom(lean_object* v_I_264_, lean_object* v_f_265_, lean_object* v_inst_266_, lean_object* v_00_u03b3_267_, lean_object* v_inst_268_, lean_object* v_g_269_){
_start:
{
lean_object* v___f_270_; 
v___f_270_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_270_, 0, v_g_269_);
return v___f_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoidHom___boxed(lean_object* v_I_271_, lean_object* v_f_272_, lean_object* v_inst_273_, lean_object* v_00_u03b3_274_, lean_object* v_inst_275_, lean_object* v_g_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_Pi_addMonoidHom(v_I_271_, v_f_272_, v_inst_273_, v_00_u03b3_274_, v_inst_275_, v_g_276_);
lean_dec_ref(v_inst_275_);
lean_dec_ref(v_inst_273_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMonoidHom___redArg(lean_object* v_i_278_){
_start:
{
lean_object* v___f_279_; 
v___f_279_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_279_, 0, v_i_278_);
return v___f_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMonoidHom(lean_object* v_I_280_, lean_object* v_f_281_, lean_object* v_inst_282_, lean_object* v_i_283_){
_start:
{
lean_object* v___f_284_; 
v___f_284_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_284_, 0, v_i_283_);
return v___f_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalMonoidHom___boxed(lean_object* v_I_285_, lean_object* v_f_286_, lean_object* v_inst_287_, lean_object* v_i_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_Pi_evalMonoidHom(v_I_285_, v_f_286_, v_inst_287_, v_i_288_);
lean_dec_ref(v_inst_287_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddMonoidHom___redArg(lean_object* v_i_290_){
_start:
{
lean_object* v___f_291_; 
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_291_, 0, v_i_290_);
return v___f_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddMonoidHom(lean_object* v_I_292_, lean_object* v_f_293_, lean_object* v_inst_294_, lean_object* v_i_295_){
_start:
{
lean_object* v___f_296_; 
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_296_, 0, v_i_295_);
return v___f_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalAddMonoidHom___boxed(lean_object* v_I_297_, lean_object* v_f_298_, lean_object* v_inst_299_, lean_object* v_i_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_Pi_evalAddMonoidHom(v_I_297_, v_f_298_, v_inst_299_, v_i_300_);
lean_dec_ref(v_inst_299_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_piMap___redArg(lean_object* v_g_302_){
_start:
{
lean_object* v___f_303_; lean_object* v___f_304_; 
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_piMap___redArg___lam__0), 3, 1);
lean_closure_set(v___f_303_, 0, v_g_302_);
v___f_304_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_304_, 0, v___f_303_);
return v___f_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_piMap(lean_object* v_00_u03b9_305_, lean_object* v_M_306_, lean_object* v_N_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_g_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_MonoidHom_piMap___redArg(v_g_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_piMap___boxed(lean_object* v_00_u03b9_312_, lean_object* v_M_313_, lean_object* v_N_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_g_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_mathlib_MonoidHom_piMap(v_00_u03b9_312_, v_M_313_, v_N_314_, v_inst_315_, v_inst_316_, v_g_317_);
lean_dec_ref(v_inst_316_);
lean_dec_ref(v_inst_315_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_piMap___redArg(lean_object* v_g_319_){
_start:
{
lean_object* v___f_320_; lean_object* v___f_321_; 
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_piMap___redArg___lam__0), 3, 1);
lean_closure_set(v___f_320_, 0, v_g_319_);
v___f_321_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_321_, 0, v___f_320_);
return v___f_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_piMap(lean_object* v_00_u03b9_322_, lean_object* v_M_323_, lean_object* v_N_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_g_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lp_mathlib_AddMonoidHom_piMap___redArg(v_g_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_piMap___boxed(lean_object* v_00_u03b9_329_, lean_object* v_M_330_, lean_object* v_N_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_g_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_AddMonoidHom_piMap(v_00_u03b9_329_, v_M_330_, v_N_331_, v_inst_332_, v_inst_333_, v_g_334_);
lean_dec_ref(v_inst_333_);
lean_dec_ref(v_inst_332_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constMonoidHom(lean_object* v_00_u03b1_336_, lean_object* v_00_u03b2_337_, lean_object* v_inst_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = ((lean_object*)(lp_mathlib_Pi_constMulHom___closed__0));
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constMonoidHom___boxed(lean_object* v_00_u03b1_340_, lean_object* v_00_u03b2_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_Pi_constMonoidHom(v_00_u03b1_340_, v_00_u03b2_341_, v_inst_342_);
lean_dec_ref(v_inst_342_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constAddMonoidHom(lean_object* v_00_u03b1_344_, lean_object* v_00_u03b2_345_, lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = ((lean_object*)(lp_mathlib_Pi_constMulHom___closed__0));
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_constAddMonoidHom___boxed(lean_object* v_00_u03b1_348_, lean_object* v_00_u03b2_349_, lean_object* v_inst_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_Pi_constAddMonoidHom(v_00_u03b1_348_, v_00_u03b2_349_, v_inst_350_);
lean_dec_ref(v_inst_350_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeFn(lean_object* v_00_u03b1_352_, lean_object* v_00_u03b2_353_, lean_object* v_inst_354_, lean_object* v_inst_355_){
_start:
{
lean_object* v___f_356_; 
v___f_356_ = ((lean_object*)(lp_mathlib_MulHom_coeFn___closed__0));
return v___f_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coeFn___boxed(lean_object* v_00_u03b1_357_, lean_object* v_00_u03b2_358_, lean_object* v_inst_359_, lean_object* v_inst_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_mathlib_MonoidHom_coeFn(v_00_u03b1_357_, v_00_u03b2_358_, v_inst_359_, v_inst_360_);
lean_dec_ref(v_inst_360_);
lean_dec_ref(v_inst_359_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeFn(lean_object* v_00_u03b1_362_, lean_object* v_00_u03b2_363_, lean_object* v_inst_364_, lean_object* v_inst_365_){
_start:
{
lean_object* v___f_366_; 
v___f_366_ = ((lean_object*)(lp_mathlib_MulHom_coeFn___closed__0));
return v___f_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coeFn___boxed(lean_object* v_00_u03b1_367_, lean_object* v_00_u03b2_368_, lean_object* v_inst_369_, lean_object* v_inst_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_mathlib_AddMonoidHom_coeFn(v_00_u03b1_367_, v_00_u03b2_368_, v_inst_369_, v_inst_370_);
lean_dec_ref(v_inst_370_);
lean_dec_ref(v_inst_369_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeft___redArg(lean_object* v_f_372_){
_start:
{
lean_object* v___f_373_; 
v___f_373_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_373_, 0, v_f_372_);
return v___f_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeft(lean_object* v_00_u03b1_374_, lean_object* v_00_u03b2_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_f_378_, lean_object* v_I_379_){
_start:
{
lean_object* v___f_380_; 
v___f_380_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_380_, 0, v_f_378_);
return v___f_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compLeft___boxed(lean_object* v_00_u03b1_381_, lean_object* v_00_u03b2_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_f_385_, lean_object* v_I_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_mathlib_MonoidHom_compLeft(v_00_u03b1_381_, v_00_u03b2_382_, v_inst_383_, v_inst_384_, v_f_385_, v_I_386_);
lean_dec_ref(v_inst_384_);
lean_dec_ref(v_inst_383_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeft___redArg(lean_object* v_f_388_){
_start:
{
lean_object* v___f_389_; 
v___f_389_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_389_, 0, v_f_388_);
return v___f_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeft(lean_object* v_00_u03b1_390_, lean_object* v_00_u03b2_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_f_394_, lean_object* v_I_395_){
_start:
{
lean_object* v___f_396_; 
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_compLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_396_, 0, v_f_394_);
return v___f_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compLeft___boxed(lean_object* v_00_u03b1_397_, lean_object* v_00_u03b2_398_, lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_f_401_, lean_object* v_I_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_AddMonoidHom_compLeft(v_00_u03b1_397_, v_00_u03b2_398_, v_inst_399_, v_inst_400_, v_f_401_, v_I_402_);
lean_dec_ref(v_inst_400_);
lean_dec_ref(v_inst_399_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_mulSingle___redArg(lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_i_406_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulSingle___boxed), 7, 5);
lean_closure_set(v___x_407_, 0, lean_box(0));
lean_closure_set(v___x_407_, 1, lean_box(0));
lean_closure_set(v___x_407_, 2, v_inst_405_);
lean_closure_set(v___x_407_, 3, v_inst_404_);
lean_closure_set(v___x_407_, 4, v_i_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_mulSingle(lean_object* v_I_408_, lean_object* v_f_409_, lean_object* v_inst_410_, lean_object* v_inst_411_, lean_object* v_i_412_){
_start:
{
lean_object* v___x_413_; 
v___x_413_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulSingle___boxed), 7, 5);
lean_closure_set(v___x_413_, 0, lean_box(0));
lean_closure_set(v___x_413_, 1, lean_box(0));
lean_closure_set(v___x_413_, 2, v_inst_411_);
lean_closure_set(v___x_413_, 3, v_inst_410_);
lean_closure_set(v___x_413_, 4, v_i_412_);
return v___x_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_single___redArg(lean_object* v_inst_414_, lean_object* v_inst_415_, lean_object* v_i_416_){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 5);
lean_closure_set(v___x_417_, 0, lean_box(0));
lean_closure_set(v___x_417_, 1, lean_box(0));
lean_closure_set(v___x_417_, 2, v_inst_415_);
lean_closure_set(v___x_417_, 3, v_inst_414_);
lean_closure_set(v___x_417_, 4, v_i_416_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_single(lean_object* v_I_418_, lean_object* v_f_419_, lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_i_422_){
_start:
{
lean_object* v___x_423_; 
v___x_423_ = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 5);
lean_closure_set(v___x_423_, 0, lean_box(0));
lean_closure_set(v___x_423_, 1, lean_box(0));
lean_closure_set(v___x_423_, 2, v_inst_421_);
lean_closure_set(v___x_423_, 3, v_inst_420_);
lean_closure_set(v___x_423_, 4, v_i_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mulSingle___redArg___lam__0(lean_object* v_inst_424_, lean_object* v_i_425_){
_start:
{
lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v_toOne_428_; 
v___x_426_ = lean_apply_1(v_inst_424_, v_i_425_);
v___x_427_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_426_);
v_toOne_428_ = lean_ctor_get(v___x_427_, 0);
lean_inc(v_toOne_428_);
lean_dec_ref(v___x_427_);
return v_toOne_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mulSingle___redArg(lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_i_431_){
_start:
{
lean_object* v___f_432_; lean_object* v___x_433_; 
v___f_432_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_mulSingle___redArg___lam__0), 2, 1);
lean_closure_set(v___f_432_, 0, v_inst_430_);
v___x_433_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulSingle___boxed), 7, 5);
lean_closure_set(v___x_433_, 0, lean_box(0));
lean_closure_set(v___x_433_, 1, lean_box(0));
lean_closure_set(v___x_433_, 2, v___f_432_);
lean_closure_set(v___x_433_, 3, v_inst_429_);
lean_closure_set(v___x_433_, 4, v_i_431_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mulSingle(lean_object* v_I_434_, lean_object* v_f_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_i_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_MonoidHom_mulSingle___redArg(v_inst_436_, v_inst_437_, v_i_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_single___redArg___lam__0(lean_object* v_inst_440_, lean_object* v_i_441_){
_start:
{
lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v_toZero_444_; 
v___x_442_ = lean_apply_1(v_inst_440_, v_i_441_);
v___x_443_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_442_);
v_toZero_444_ = lean_ctor_get(v___x_443_, 0);
lean_inc(v_toZero_444_);
lean_dec_ref(v___x_443_);
return v_toZero_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_single___redArg(lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_i_447_){
_start:
{
lean_object* v___f_448_; lean_object* v___x_449_; 
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_single___redArg___lam__0), 2, 1);
lean_closure_set(v___f_448_, 0, v_inst_446_);
v___x_449_ = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 5);
lean_closure_set(v___x_449_, 0, lean_box(0));
lean_closure_set(v___x_449_, 1, lean_box(0));
lean_closure_set(v___x_449_, 2, v___f_448_);
lean_closure_set(v___x_449_, 3, v_inst_445_);
lean_closure_set(v___x_449_, 4, v_i_447_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_single(lean_object* v_I_450_, lean_object* v_f_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_i_454_){
_start:
{
lean_object* v___x_455_; 
v___x_455_ = lp_mathlib_AddMonoidHom_single___redArg(v_inst_452_, v_inst_453_, v_i_454_);
return v___x_455_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Piecewise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Pairwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Piecewise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Pairwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
}
#ifdef __cplusplus
}
#endif
