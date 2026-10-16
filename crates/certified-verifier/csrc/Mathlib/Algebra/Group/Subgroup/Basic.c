// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Conj public import Mathlib.Algebra.Group.Pi.Lemmas public import Mathlib.Algebra.Group.Subgroup.Ker public import Mathlib.Algebra.Group.Torsion
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
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subgroup_prodEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subgroup_prodEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic___lam__2(lean_object*);
static const lean_closure_object lp_mathlib_MulAut_characteristic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulAut_characteristic___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulAut_characteristic___closed__0 = (const lean_object*)&lp_mathlib_MulAut_characteristic___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_characteristic(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_characteristic___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalClosure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalClosure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalClosure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalClosure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalCore(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalCore___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalCore(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalCore___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverseAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverseAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverseAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverseAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverseAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_inertia(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_inertia___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_prod(lean_object* v_G_1_, lean_object* v_inst_2_, lean_object* v_N_3_, lean_object* v_inst_4_, lean_object* v_H_5_, lean_object* v_K_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_prod___boxed(lean_object* v_G_8_, lean_object* v_inst_9_, lean_object* v_N_10_, lean_object* v_inst_11_, lean_object* v_H_12_, lean_object* v_K_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Subgroup_prod(v_G_8_, v_inst_9_, v_N_10_, v_inst_11_, v_H_12_, v_K_13_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_9_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_prod(lean_object* v_G_15_, lean_object* v_inst_16_, lean_object* v_N_17_, lean_object* v_inst_18_, lean_object* v_H_19_, lean_object* v_K_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_box(0);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_prod___boxed(lean_object* v_G_22_, lean_object* v_inst_23_, lean_object* v_N_24_, lean_object* v_inst_25_, lean_object* v_H_26_, lean_object* v_K_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_AddSubgroup_prod(v_G_22_, v_inst_23_, v_N_24_, v_inst_25_, v_H_26_, v_K_27_);
lean_dec_ref(v_inst_25_);
lean_dec_ref(v_inst_23_);
return v_res_28_;
}
}
static lean_object* _init_lp_mathlib_Subgroup_prodEquiv___closed__0(void){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_prodEquiv(lean_object* v_G_30_, lean_object* v_inst_31_, lean_object* v_N_32_, lean_object* v_inst_33_, lean_object* v_H_34_, lean_object* v_K_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lean_obj_once(&lp_mathlib_Subgroup_prodEquiv___closed__0, &lp_mathlib_Subgroup_prodEquiv___closed__0_once, _init_lp_mathlib_Subgroup_prodEquiv___closed__0);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_prodEquiv___boxed(lean_object* v_G_37_, lean_object* v_inst_38_, lean_object* v_N_39_, lean_object* v_inst_40_, lean_object* v_H_41_, lean_object* v_K_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Subgroup_prodEquiv(v_G_37_, v_inst_38_, v_N_39_, v_inst_40_, v_H_41_, v_K_42_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_38_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_prodEquiv(lean_object* v_G_44_, lean_object* v_inst_45_, lean_object* v_N_46_, lean_object* v_inst_47_, lean_object* v_H_48_, lean_object* v_K_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_obj_once(&lp_mathlib_Subgroup_prodEquiv___closed__0, &lp_mathlib_Subgroup_prodEquiv___closed__0_once, _init_lp_mathlib_Subgroup_prodEquiv___closed__0);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_prodEquiv___boxed(lean_object* v_G_51_, lean_object* v_inst_52_, lean_object* v_N_53_, lean_object* v_inst_54_, lean_object* v_H_55_, lean_object* v_K_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_AddSubgroup_prodEquiv(v_G_51_, v_inst_52_, v_N_53_, v_inst_54_, v_H_55_, v_K_56_);
lean_dec_ref(v_inst_54_);
lean_dec_ref(v_inst_52_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pi(lean_object* v_00_u03b7_58_, lean_object* v_f_59_, lean_object* v_inst_60_, lean_object* v_I_61_, lean_object* v_H_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lean_box(0);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pi___boxed(lean_object* v_00_u03b7_64_, lean_object* v_f_65_, lean_object* v_inst_66_, lean_object* v_I_67_, lean_object* v_H_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Subgroup_pi(v_00_u03b7_64_, v_f_65_, v_inst_66_, v_I_67_, v_H_68_);
lean_dec_ref(v_H_68_);
lean_dec_ref(v_inst_66_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_pi(lean_object* v_00_u03b7_70_, lean_object* v_f_71_, lean_object* v_inst_72_, lean_object* v_I_73_, lean_object* v_H_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lean_box(0);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_pi___boxed(lean_object* v_00_u03b7_76_, lean_object* v_f_77_, lean_object* v_inst_78_, lean_object* v_I_79_, lean_object* v_H_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_AddSubgroup_pi(v_00_u03b7_76_, v_f_77_, v_inst_78_, v_I_79_, v_H_80_);
lean_dec_ref(v_H_80_);
lean_dec_ref(v_inst_78_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic___lam__0(lean_object* v_00_u03c6_82_, lean_object* v_h_83_){
_start:
{
lean_object* v_toFun_84_; lean_object* v___x_85_; 
v_toFun_84_ = lean_ctor_get(v_00_u03c6_82_, 0);
lean_inc(v_toFun_84_);
lean_dec_ref(v_00_u03c6_82_);
v___x_85_ = lean_apply_1(v_toFun_84_, v_h_83_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic___lam__1(lean_object* v_00_u03c6_86_, lean_object* v_h_87_){
_start:
{
lean_object* v___x_88_; lean_object* v_toFun_89_; lean_object* v___x_90_; 
v___x_88_ = lp_mathlib_Equiv_symm___redArg(v_00_u03c6_86_);
v_toFun_89_ = lean_ctor_get(v___x_88_, 0);
lean_inc(v_toFun_89_);
lean_dec_ref(v___x_88_);
v___x_90_ = lean_apply_1(v_toFun_89_, v_h_87_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic___lam__2(lean_object* v_00_u03c6_91_){
_start:
{
lean_object* v___f_92_; lean_object* v___f_93_; lean_object* v___x_94_; 
lean_inc_ref(v_00_u03c6_91_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_characteristic___lam__0), 2, 1);
lean_closure_set(v___f_92_, 0, v_00_u03c6_91_);
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_MulAut_characteristic___lam__1), 2, 1);
lean_closure_set(v___f_93_, 0, v_00_u03c6_91_);
v___x_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_94_, 0, v___f_92_);
lean_ctor_set(v___x_94_, 1, v___f_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic(lean_object* v_G_96_, lean_object* v_inst_97_, lean_object* v_H_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v___f_100_; 
v___f_100_ = ((lean_object*)(lp_mathlib_MulAut_characteristic___closed__0));
return v___f_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_characteristic___boxed(lean_object* v_G_101_, lean_object* v_inst_102_, lean_object* v_H_103_, lean_object* v_inst_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_MulAut_characteristic(v_G_101_, v_inst_102_, v_H_103_, v_inst_104_);
lean_dec_ref(v_inst_102_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_characteristic(lean_object* v_G_106_, lean_object* v_inst_107_, lean_object* v_H_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___f_110_; 
v___f_110_ = ((lean_object*)(lp_mathlib_MulAut_characteristic___closed__0));
return v___f_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_characteristic___boxed(lean_object* v_G_111_, lean_object* v_inst_112_, lean_object* v_H_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_AddAut_characteristic(v_G_111_, v_inst_112_, v_H_113_, v_inst_114_);
lean_dec_ref(v_inst_112_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalClosure(lean_object* v_G_116_, lean_object* v_inst_117_, lean_object* v_s_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lean_box(0);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalClosure___boxed(lean_object* v_G_120_, lean_object* v_inst_121_, lean_object* v_s_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_Subgroup_normalClosure(v_G_120_, v_inst_121_, v_s_122_);
lean_dec_ref(v_inst_121_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalClosure(lean_object* v_G_124_, lean_object* v_inst_125_, lean_object* v_s_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lean_box(0);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalClosure___boxed(lean_object* v_G_128_, lean_object* v_inst_129_, lean_object* v_s_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_AddSubgroup_normalClosure(v_G_128_, v_inst_129_, v_s_130_);
lean_dec_ref(v_inst_129_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalCore(lean_object* v_G_132_, lean_object* v_inst_133_, lean_object* v_H_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lean_box(0);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalCore___boxed(lean_object* v_G_136_, lean_object* v_inst_137_, lean_object* v_H_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Subgroup_normalCore(v_G_136_, v_inst_137_, v_H_138_);
lean_dec_ref(v_inst_137_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalCore(lean_object* v_G_140_, lean_object* v_inst_141_, lean_object* v_H_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lean_box(0);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalCore___boxed(lean_object* v_G_144_, lean_object* v_inst_145_, lean_object* v_H_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_AddSubgroup_normalCore(v_G_144_, v_inst_145_, v_H_146_);
lean_dec_ref(v_inst_145_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg___lam__0(lean_object* v_f__inv_148_, lean_object* v_g_149_, lean_object* v_b_150_){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_151_ = lean_apply_1(v_f__inv_148_, v_b_150_);
v___x_152_ = lean_apply_1(v_g_149_, v___x_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg(lean_object* v_f__inv_153_, lean_object* v_g_154_){
_start:
{
lean_object* v___f_155_; 
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_155_, 0, v_f__inv_153_);
lean_closure_set(v___f_155_, 1, v_g_154_);
return v___f_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverseAux(lean_object* v_G_u2081_156_, lean_object* v_G_u2082_157_, lean_object* v_G_u2083_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_f_162_, lean_object* v_f__inv_163_, lean_object* v_hf_164_, lean_object* v_g_165_, lean_object* v_hg_166_){
_start:
{
lean_object* v___f_167_; 
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_167_, 0, v_f__inv_163_);
lean_closure_set(v___f_167_, 1, v_g_165_);
return v___f_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverseAux___boxed(lean_object* v_G_u2081_168_, lean_object* v_G_u2082_169_, lean_object* v_G_u2083_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_f_174_, lean_object* v_f__inv_175_, lean_object* v_hf_176_, lean_object* v_g_177_, lean_object* v_hg_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_MonoidHom_liftOfRightInverseAux(v_G_u2081_168_, v_G_u2082_169_, v_G_u2083_170_, v_inst_171_, v_inst_172_, v_inst_173_, v_f_174_, v_f__inv_175_, v_hf_176_, v_g_177_, v_hg_178_);
lean_dec(v_f_174_);
lean_dec_ref(v_inst_173_);
lean_dec_ref(v_inst_172_);
lean_dec_ref(v_inst_171_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverseAux___redArg(lean_object* v_f__inv_180_, lean_object* v_g_181_){
_start:
{
lean_object* v___f_182_; 
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_182_, 0, v_f__inv_180_);
lean_closure_set(v___f_182_, 1, v_g_181_);
return v___f_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverseAux(lean_object* v_G_u2081_183_, lean_object* v_G_u2082_184_, lean_object* v_G_u2083_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_f_189_, lean_object* v_f__inv_190_, lean_object* v_hf_191_, lean_object* v_g_192_, lean_object* v_hg_193_){
_start:
{
lean_object* v___f_194_; 
v___f_194_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_194_, 0, v_f__inv_190_);
lean_closure_set(v___f_194_, 1, v_g_192_);
return v___f_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverseAux___boxed(lean_object* v_G_u2081_195_, lean_object* v_G_u2082_196_, lean_object* v_G_u2083_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_f_201_, lean_object* v_f__inv_202_, lean_object* v_hf_203_, lean_object* v_g_204_, lean_object* v_hg_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_AddMonoidHom_liftOfRightInverseAux(v_G_u2081_195_, v_G_u2082_196_, v_G_u2083_197_, v_inst_198_, v_inst_199_, v_inst_200_, v_f_201_, v_f__inv_202_, v_hf_203_, v_g_204_, v_hg_205_);
lean_dec(v_f_201_);
lean_dec_ref(v_inst_200_);
lean_dec_ref(v_inst_199_);
lean_dec_ref(v_inst_198_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse___redArg___lam__0(lean_object* v_f__inv_207_, lean_object* v_g_208_, lean_object* v___y_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_MonoidHom_liftOfRightInverseAux___redArg___lam__0(v_f__inv_207_, v_g_208_, v___y_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse___redArg___lam__1(lean_object* v_f_211_, lean_object* v_00_u03c6_212_, lean_object* v___y_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lp_mathlib_OneHom_comp___redArg___lam__0(v_f_211_, v_00_u03c6_212_, v___y_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse___redArg(lean_object* v_f_215_, lean_object* v_f__inv_216_){
_start:
{
lean_object* v___f_217_; lean_object* v___f_218_; lean_object* v___x_219_; 
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_liftOfRightInverse___redArg___lam__0), 3, 1);
lean_closure_set(v___f_217_, 0, v_f__inv_216_);
v___f_218_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_liftOfRightInverse___redArg___lam__1), 3, 1);
lean_closure_set(v___f_218_, 0, v_f_215_);
v___x_219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_219_, 0, v___f_217_);
lean_ctor_set(v___x_219_, 1, v___f_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse(lean_object* v_G_u2081_220_, lean_object* v_G_u2082_221_, lean_object* v_G_u2083_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_f_226_, lean_object* v_f__inv_227_, lean_object* v_hf_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_MonoidHom_liftOfRightInverse___redArg(v_f_226_, v_f__inv_227_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_liftOfRightInverse___boxed(lean_object* v_G_u2081_230_, lean_object* v_G_u2082_231_, lean_object* v_G_u2083_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_f_236_, lean_object* v_f__inv_237_, lean_object* v_hf_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_MonoidHom_liftOfRightInverse(v_G_u2081_230_, v_G_u2082_231_, v_G_u2083_232_, v_inst_233_, v_inst_234_, v_inst_235_, v_f_236_, v_f__inv_237_, v_hf_238_);
lean_dec_ref(v_inst_235_);
lean_dec_ref(v_inst_234_);
lean_dec_ref(v_inst_233_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverse___redArg(lean_object* v_f_240_, lean_object* v_f__inv_241_){
_start:
{
lean_object* v___f_242_; lean_object* v___f_243_; lean_object* v___x_244_; 
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_liftOfRightInverse___redArg___lam__0), 3, 1);
lean_closure_set(v___f_242_, 0, v_f__inv_241_);
v___f_243_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_liftOfRightInverse___redArg___lam__1), 3, 1);
lean_closure_set(v___f_243_, 0, v_f_240_);
v___x_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_244_, 0, v___f_242_);
lean_ctor_set(v___x_244_, 1, v___f_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverse(lean_object* v_G_u2081_245_, lean_object* v_G_u2082_246_, lean_object* v_G_u2083_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_f_251_, lean_object* v_f__inv_252_, lean_object* v_hf_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lp_mathlib_AddMonoidHom_liftOfRightInverse___redArg(v_f_251_, v_f__inv_252_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_liftOfRightInverse___boxed(lean_object* v_G_u2081_255_, lean_object* v_G_u2082_256_, lean_object* v_G_u2083_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_f_261_, lean_object* v_f__inv_262_, lean_object* v_hf_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_AddMonoidHom_liftOfRightInverse(v_G_u2081_255_, v_G_u2082_256_, v_G_u2083_257_, v_inst_258_, v_inst_259_, v_inst_260_, v_f_261_, v_f__inv_262_, v_hf_263_);
lean_dec_ref(v_inst_260_);
lean_dec_ref(v_inst_259_);
lean_dec_ref(v_inst_258_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_inertia(lean_object* v_M_265_, lean_object* v_inst_266_, lean_object* v_I_267_, lean_object* v_G_268_, lean_object* v_inst_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lean_box(0);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_inertia___boxed(lean_object* v_M_272_, lean_object* v_inst_273_, lean_object* v_I_274_, lean_object* v_G_275_, lean_object* v_inst_276_, lean_object* v_inst_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_AddSubgroup_inertia(v_M_272_, v_inst_273_, v_I_274_, v_G_275_, v_inst_276_, v_inst_277_);
lean_dec(v_inst_277_);
lean_dec_ref(v_inst_276_);
lean_dec_ref(v_inst_273_);
return v_res_278_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Conj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Conj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Conj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Conj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
