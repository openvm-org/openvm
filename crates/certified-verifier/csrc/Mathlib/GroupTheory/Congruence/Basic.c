// Lean compiler output
// Module: Mathlib.GroupTheory.Congruence.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.Data.Setoid.Basic public import Mathlib.GroupTheory.Congruence.Hom
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
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AddCon_toQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Con_toQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Quot_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_pi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_pi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Con_congr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Con_congr___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Con_congr___redArg___closed__0 = (const lean_object*)&lp_mathlib_Con_congr___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Con_congr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Con_congr___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Con_congr___redArg___closed__1 = (const lean_object*)&lp_mathlib_Con_congr___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Con_congr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Con_congr___redArg___closed__0_value),((lean_object*)&lp_mathlib_Con_congr___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Con_congr___redArg___closed__2 = (const lean_object*)&lp_mathlib_Con_congr___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_submonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_submonoid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubmonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubmonoid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_ofSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_ofSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ofAddSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ofAddSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Con_toSubmonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Con_toSubmonoid___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Con_toSubmonoid___closed__0 = (const lean_object*)&lp_mathlib_Con_toSubmonoid___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_AddCon_toAddSubmonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddCon_toAddSubmonoid___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddCon_toAddSubmonoid___closed__0 = (const lean_object*)&lp_mathlib_AddCon_toAddSubmonoid___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_prod(lean_object* v_M_1_, lean_object* v_N_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_c_5_, lean_object* v_d_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_prod___boxed(lean_object* v_M_8_, lean_object* v_N_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_c_12_, lean_object* v_d_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Con_prod(v_M_8_, v_N_9_, v_inst_10_, v_inst_11_, v_c_12_, v_d_13_);
lean_dec(v_inst_11_);
lean_dec(v_inst_10_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_prod(lean_object* v_M_15_, lean_object* v_N_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_c_19_, lean_object* v_d_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_box(0);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_prod___boxed(lean_object* v_M_22_, lean_object* v_N_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_c_26_, lean_object* v_d_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_AddCon_prod(v_M_22_, v_N_23_, v_inst_24_, v_inst_25_, v_c_26_, v_d_27_);
lean_dec(v_inst_25_);
lean_dec(v_inst_24_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_pi(lean_object* v_00_u03b9_29_, lean_object* v_f_30_, lean_object* v_inst_31_, lean_object* v_C_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_box(0);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_pi___boxed(lean_object* v_00_u03b9_34_, lean_object* v_f_35_, lean_object* v_inst_36_, lean_object* v_C_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Con_pi(v_00_u03b9_34_, v_f_35_, v_inst_36_, v_C_37_);
lean_dec_ref(v_C_37_);
lean_dec(v_inst_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_pi(lean_object* v_00_u03b9_39_, lean_object* v_f_40_, lean_object* v_inst_41_, lean_object* v_C_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lean_box(0);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_pi___boxed(lean_object* v_00_u03b9_44_, lean_object* v_f_45_, lean_object* v_inst_46_, lean_object* v_C_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_AddCon_pi(v_00_u03b9_44_, v_f_45_, v_inst_46_, v_C_47_);
lean_dec_ref(v_C_47_);
lean_dec(v_inst_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___redArg___lam__0(lean_object* v_f_49_, lean_object* v___y_50_){
_start:
{
lean_object* v_toFun_51_; lean_object* v___x_52_; 
v_toFun_51_ = lean_ctor_get(v_f_49_, 0);
lean_inc(v_toFun_51_);
lean_dec_ref(v_f_49_);
v___x_52_ = lean_apply_1(v_toFun_51_, v___y_50_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___redArg___lam__1(lean_object* v_f_53_, lean_object* v___y_54_){
_start:
{
lean_object* v_invFun_55_; lean_object* v___x_56_; 
v_invFun_55_ = lean_ctor_get(v_f_53_, 1);
lean_inc(v_invFun_55_);
lean_dec_ref(v_f_53_);
v___x_56_ = lean_apply_1(v_invFun_55_, v___y_54_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___redArg(lean_object* v_e_62_){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_63_ = ((lean_object*)(lp_mathlib_Con_congr___redArg___closed__2));
v___x_64_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_63_, v_e_62_);
v___x_65_ = lp_mathlib_Quot_congr___redArg(v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_congr(lean_object* v_M_66_, lean_object* v_N_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_c_70_, lean_object* v_d_71_, lean_object* v_e_72_, lean_object* v_h_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_Con_congr___redArg(v_e_72_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___boxed(lean_object* v_M_75_, lean_object* v_N_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_c_79_, lean_object* v_d_80_, lean_object* v_e_81_, lean_object* v_h_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Con_congr(v_M_75_, v_N_76_, v_inst_77_, v_inst_78_, v_c_79_, v_d_80_, v_e_81_, v_h_82_);
lean_dec(v_inst_78_);
lean_dec(v_inst_77_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr___redArg(lean_object* v_e_84_){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_85_ = ((lean_object*)(lp_mathlib_Con_congr___redArg___closed__2));
v___x_86_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_85_, v_e_84_);
v___x_87_ = lp_mathlib_Quot_congr___redArg(v___x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr(lean_object* v_M_88_, lean_object* v_N_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_c_92_, lean_object* v_d_93_, lean_object* v_e_94_, lean_object* v_h_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_AddCon_congr___redArg(v_e_94_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr___boxed(lean_object* v_M_97_, lean_object* v_N_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_c_101_, lean_object* v_d_102_, lean_object* v_e_103_, lean_object* v_h_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_AddCon_congr(v_M_97_, v_N_98_, v_inst_99_, v_inst_100_, v_c_101_, v_d_102_, v_e_103_, v_h_104_);
lean_dec(v_inst_100_);
lean_dec(v_inst_99_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_submonoid(lean_object* v_M_106_, lean_object* v_inst_107_, lean_object* v_c_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lean_box(0);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_submonoid___boxed(lean_object* v_M_110_, lean_object* v_inst_111_, lean_object* v_c_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Con_submonoid(v_M_110_, v_inst_111_, v_c_112_);
lean_dec_ref(v_inst_111_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubmonoid(lean_object* v_M_114_, lean_object* v_inst_115_, lean_object* v_c_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lean_box(0);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubmonoid___boxed(lean_object* v_M_118_, lean_object* v_inst_119_, lean_object* v_c_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_AddCon_addSubmonoid(v_M_118_, v_inst_119_, v_c_120_);
lean_dec_ref(v_inst_119_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_ofSubmonoid(lean_object* v_M_122_, lean_object* v_inst_123_, lean_object* v_N_124_, lean_object* v_H_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lean_box(0);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_ofSubmonoid___boxed(lean_object* v_M_127_, lean_object* v_inst_128_, lean_object* v_N_129_, lean_object* v_H_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Con_ofSubmonoid(v_M_127_, v_inst_128_, v_N_129_, v_H_130_);
lean_dec_ref(v_inst_128_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ofAddSubmonoid(lean_object* v_M_132_, lean_object* v_inst_133_, lean_object* v_N_134_, lean_object* v_H_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_box(0);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ofAddSubmonoid___boxed(lean_object* v_M_137_, lean_object* v_inst_138_, lean_object* v_N_139_, lean_object* v_H_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_AddCon_ofAddSubmonoid(v_M_137_, v_inst_138_, v_N_139_, v_H_140_);
lean_dec_ref(v_inst_138_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid___lam__0(lean_object* v_c_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lean_box(0);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid(lean_object* v_M_145_, lean_object* v_inst_146_){
_start:
{
lean_object* v___f_147_; 
v___f_147_ = ((lean_object*)(lp_mathlib_Con_toSubmonoid___closed__0));
return v___f_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid___boxed(lean_object* v_M_148_, lean_object* v_inst_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Con_toSubmonoid(v_M_148_, v_inst_149_);
lean_dec_ref(v_inst_149_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid___lam__0(lean_object* v_c_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lean_box(0);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid(lean_object* v_M_154_, lean_object* v_inst_155_){
_start:
{
lean_object* v___f_156_; 
v___f_156_ = ((lean_object*)(lp_mathlib_AddCon_toAddSubmonoid___closed__0));
return v___f_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid___boxed(lean_object* v_M_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_AddCon_toAddSubmonoid(v_M_157_, v_inst_158_);
lean_dec_ref(v_inst_158_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0(lean_object* v_f_160_, lean_object* v___y_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_Con_lift___redArg___lam__0(v_f_160_, v___y_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg(lean_object* v_inst_163_, lean_object* v_f_164_, lean_object* v_g_165_){
_start:
{
lean_object* v___x_166_; lean_object* v_toMul_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_178_; 
v___x_166_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_163_);
v_toMul_167_ = lean_ctor_get(v___x_166_, 1);
v_isSharedCheck_178_ = !lean_is_exclusive(v___x_166_);
if (v_isSharedCheck_178_ == 0)
{
lean_object* v_unused_179_; 
v_unused_179_ = lean_ctor_get(v___x_166_, 0);
lean_dec(v_unused_179_);
v___x_169_ = v___x_166_;
v_isShared_170_ = v_isSharedCheck_178_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_toMul_167_);
lean_dec(v___x_166_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_178_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___f_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_176_; 
v___f_171_ = lean_alloc_closure((void*)(lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_171_, 0, v_f_164_);
v___x_172_ = lean_box(0);
v___x_173_ = lean_alloc_closure((void*)(lp_mathlib_Con_toQuotient___boxed), 4, 3);
lean_closure_set(v___x_173_, 0, lean_box(0));
lean_closure_set(v___x_173_, 1, v_toMul_167_);
lean_closure_set(v___x_173_, 2, v___x_172_);
v___x_174_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_174_, 0, lean_box(0));
lean_closure_set(v___x_174_, 1, lean_box(0));
lean_closure_set(v___x_174_, 2, lean_box(0));
lean_closure_set(v___x_174_, 3, v___x_173_);
lean_closure_set(v___x_174_, 4, v_g_165_);
if (v_isShared_170_ == 0)
{
lean_ctor_set(v___x_169_, 1, v___x_174_);
lean_ctor_set(v___x_169_, 0, v___f_171_);
v___x_176_ = v___x_169_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_177_; 
v_reuseFailAlloc_177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_177_, 0, v___f_171_);
lean_ctor_set(v_reuseFailAlloc_177_, 1, v___x_174_);
v___x_176_ = v_reuseFailAlloc_177_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
return v___x_176_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse(lean_object* v_M_180_, lean_object* v_P_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_f_184_, lean_object* v_g_185_, lean_object* v_hf_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg(v_inst_182_, v_f_184_, v_g_185_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___boxed(lean_object* v_M_188_, lean_object* v_P_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_f_192_, lean_object* v_g_193_, lean_object* v_hf_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_Con_quotientKerEquivOfRightInverse(v_M_188_, v_P_189_, v_inst_190_, v_inst_191_, v_f_192_, v_g_193_, v_hf_194_);
lean_dec_ref(v_inst_191_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse___redArg(lean_object* v_inst_196_, lean_object* v_f_197_, lean_object* v_g_198_){
_start:
{
lean_object* v___x_199_; lean_object* v_toAdd_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_211_; 
v___x_199_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_196_);
v_toAdd_200_ = lean_ctor_get(v___x_199_, 1);
v_isSharedCheck_211_ = !lean_is_exclusive(v___x_199_);
if (v_isSharedCheck_211_ == 0)
{
lean_object* v_unused_212_; 
v_unused_212_ = lean_ctor_get(v___x_199_, 0);
lean_dec(v_unused_212_);
v___x_202_ = v___x_199_;
v_isShared_203_ = v_isSharedCheck_211_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_toAdd_200_);
lean_dec(v___x_199_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_211_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___f_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_209_; 
v___f_204_ = lean_alloc_closure((void*)(lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_204_, 0, v_f_197_);
v___x_205_ = lean_box(0);
v___x_206_ = lean_alloc_closure((void*)(lp_mathlib_AddCon_toQuotient___boxed), 4, 3);
lean_closure_set(v___x_206_, 0, lean_box(0));
lean_closure_set(v___x_206_, 1, v_toAdd_200_);
lean_closure_set(v___x_206_, 2, v___x_205_);
v___x_207_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_207_, 0, lean_box(0));
lean_closure_set(v___x_207_, 1, lean_box(0));
lean_closure_set(v___x_207_, 2, lean_box(0));
lean_closure_set(v___x_207_, 3, v___x_206_);
lean_closure_set(v___x_207_, 4, v_g_198_);
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 1, v___x_207_);
lean_ctor_set(v___x_202_, 0, v___f_204_);
v___x_209_ = v___x_202_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___f_204_);
lean_ctor_set(v_reuseFailAlloc_210_, 1, v___x_207_);
v___x_209_ = v_reuseFailAlloc_210_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
return v___x_209_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse(lean_object* v_M_213_, lean_object* v_P_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_f_217_, lean_object* v_g_218_, lean_object* v_hf_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lp_mathlib_AddCon_quotientKerEquivOfRightInverse___redArg(v_inst_215_, v_f_217_, v_g_218_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse___boxed(lean_object* v_M_221_, lean_object* v_P_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_f_225_, lean_object* v_g_226_, lean_object* v_hf_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_AddCon_quotientKerEquivOfRightInverse(v_M_221_, v_P_222_, v_inst_223_, v_inst_224_, v_f_225_, v_g_226_, v_hf_227_);
lean_dec_ref(v_inst_224_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient___redArg(lean_object* v_c_229_, lean_object* v_d_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(v_c_229_, v_d_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient(lean_object* v_M_232_, lean_object* v_inst_233_, lean_object* v_c_234_, lean_object* v_d_235_, lean_object* v_h_236_){
_start:
{
lean_object* v___x_237_; 
v___x_237_ = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(v_c_234_, v_d_235_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient___boxed(lean_object* v_M_238_, lean_object* v_inst_239_, lean_object* v_c_240_, lean_object* v_d_241_, lean_object* v_h_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_Con_quotientQuotientEquivQuotient(v_M_238_, v_inst_239_, v_c_240_, v_d_241_, v_h_242_);
lean_dec_ref(v_inst_239_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient___redArg(lean_object* v_c_244_, lean_object* v_d_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(v_c_244_, v_d_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient(lean_object* v_M_247_, lean_object* v_inst_248_, lean_object* v_c_249_, lean_object* v_d_250_, lean_object* v_h_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(v_c_249_, v_d_250_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient___boxed(lean_object* v_M_253_, lean_object* v_inst_254_, lean_object* v_c_255_, lean_object* v_d_256_, lean_object* v_h_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_AddCon_quotientQuotientEquivQuotient(v_M_253_, v_inst_254_, v_c_255_, v_d_256_, v_h_257_);
lean_dec_ref(v_inst_254_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___redArg___lam__0(lean_object* v_inst_259_, lean_object* v_a_260_, lean_object* v___y_261_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lean_apply_2(v_inst_259_, v_a_260_, v___y_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___redArg(lean_object* v_inst_263_){
_start:
{
lean_object* v___f_264_; 
v___f_264_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_264_, 0, v_inst_263_);
return v___f_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul(lean_object* v_00_u03b1_265_, lean_object* v_M_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_c_270_){
_start:
{
lean_object* v___f_271_; 
v___f_271_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_271_, 0, v_inst_268_);
return v___f_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___boxed(lean_object* v_00_u03b1_272_, lean_object* v_M_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_c_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_Con_instSMul(v_00_u03b1_272_, v_M_273_, v_inst_274_, v_inst_275_, v_inst_276_, v_c_277_);
lean_dec_ref(v_inst_274_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd___redArg(lean_object* v_inst_279_){
_start:
{
lean_object* v___f_280_; 
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_280_, 0, v_inst_279_);
return v___f_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd(lean_object* v_00_u03b1_281_, lean_object* v_M_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_c_286_){
_start:
{
lean_object* v___f_287_; 
v___f_287_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_287_, 0, v_inst_284_);
return v___f_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd___boxed(lean_object* v_00_u03b1_288_, lean_object* v_M_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_c_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_AddCon_instVAdd(v_00_u03b1_288_, v_M_289_, v_inst_290_, v_inst_291_, v_inst_292_, v_c_293_);
lean_dec_ref(v_inst_290_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction___redArg(lean_object* v_inst_295_){
_start:
{
lean_object* v___f_296_; 
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_296_, 0, v_inst_295_);
return v___f_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction(lean_object* v_00_u03b1_297_, lean_object* v_M_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_c_303_){
_start:
{
lean_object* v___f_304_; 
v___f_304_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_304_, 0, v_inst_301_);
return v___f_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction___boxed(lean_object* v_00_u03b1_305_, lean_object* v_M_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_c_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_Con_mulAction(v_00_u03b1_305_, v_M_306_, v_inst_307_, v_inst_308_, v_inst_309_, v_inst_310_, v_c_311_);
lean_dec_ref(v_inst_308_);
lean_dec_ref(v_inst_307_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction___redArg(lean_object* v_inst_313_){
_start:
{
lean_object* v___f_314_; 
v___f_314_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_314_, 0, v_inst_313_);
return v___f_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction(lean_object* v_00_u03b1_315_, lean_object* v_M_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_c_321_){
_start:
{
lean_object* v___f_322_; 
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_322_, 0, v_inst_319_);
return v___f_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction___boxed(lean_object* v_00_u03b1_323_, lean_object* v_M_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_c_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_AddCon_addAction(v_00_u03b1_323_, v_M_324_, v_inst_325_, v_inst_326_, v_inst_327_, v_inst_328_, v_c_329_);
lean_dec_ref(v_inst_326_);
lean_dec_ref(v_inst_325_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction___redArg(lean_object* v_inst_331_){
_start:
{
lean_object* v___f_332_; 
v___f_332_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_332_, 0, v_inst_331_);
return v___f_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction(lean_object* v_00_u03b1_333_, lean_object* v_M_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_c_339_){
_start:
{
lean_object* v___f_340_; 
v___f_340_ = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_340_, 0, v_inst_337_);
return v___f_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction___boxed(lean_object* v_00_u03b1_341_, lean_object* v_M_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_c_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_mathlib_Con_mulDistribMulAction(v_00_u03b1_341_, v_M_342_, v_inst_343_, v_inst_344_, v_inst_345_, v_inst_346_, v_c_347_);
lean_dec_ref(v_inst_344_);
lean_dec_ref(v_inst_343_);
return v_res_348_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
