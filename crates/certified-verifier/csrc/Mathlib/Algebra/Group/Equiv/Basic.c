// Lean compiler output
// Module: Mathlib.Algebra.Group.Equiv.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.Hom.Basic public import Mathlib.Logic.Equiv.Basic public import Mathlib.Tactic.Spread
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
lean_object* lp_mathlib_Equiv_ofUnique___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piUnique___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_arrowCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_arrowCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_arrowCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRightEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRight___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRight___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piCongrRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piCongrRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_neg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofUnique___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_1_, v_inst_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofUnique(lean_object* v_M_4_, lean_object* v_N_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_6_, v_inst_7_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofUnique___boxed(lean_object* v_M_11_, lean_object* v_N_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_MulEquiv_ofUnique(v_M_11_, v_N_12_, v_inst_13_, v_inst_14_, v_inst_15_, v_inst_16_);
lean_dec(v_inst_16_);
lean_dec(v_inst_15_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofUnique___redArg(lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_18_, v_inst_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofUnique(lean_object* v_M_21_, lean_object* v_N_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_23_, v_inst_24_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofUnique___boxed(lean_object* v_M_28_, lean_object* v_N_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_AddEquiv_ofUnique(v_M_28_, v_N_29_, v_inst_30_, v_inst_31_, v_inst_32_, v_inst_33_);
lean_dec(v_inst_33_);
lean_dec(v_inst_32_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instUnique___redArg(lean_object* v_inst_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_35_, v_inst_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instUnique(lean_object* v_M_38_, lean_object* v_N_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_40_, v_inst_41_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instUnique___boxed(lean_object* v_M_45_, lean_object* v_N_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_MulEquiv_instUnique(v_M_45_, v_N_46_, v_inst_47_, v_inst_48_, v_inst_49_, v_inst_50_);
lean_dec(v_inst_50_);
lean_dec(v_inst_49_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instUnique___redArg(lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_52_, v_inst_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instUnique(lean_object* v_M_55_, lean_object* v_N_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_57_, v_inst_58_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instUnique___boxed(lean_object* v_M_62_, lean_object* v_N_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_AddEquiv_instUnique(v_M_62_, v_N_63_, v_inst_64_, v_inst_65_, v_inst_66_, v_inst_67_);
lean_dec(v_inst_67_);
lean_dec(v_inst_66_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funUnique___redArg(lean_object* v_inst_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funUnique(lean_object* v_00_u03b1_71_, lean_object* v_M_72_, lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funUnique___boxed(lean_object* v_00_u03b1_76_, lean_object* v_M_77_, lean_object* v_inst_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_MulEquiv_funUnique(v_00_u03b1_76_, v_M_77_, v_inst_78_, v_inst_79_);
lean_dec(v_inst_78_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funUnique___redArg(lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funUnique(lean_object* v_00_u03b1_83_, lean_object* v_M_84_, lean_object* v_inst_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funUnique___boxed(lean_object* v_00_u03b1_88_, lean_object* v_M_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_AddEquiv_funUnique(v_00_u03b1_88_, v_M_89_, v_inst_90_, v_inst_91_);
lean_dec(v_inst_90_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr___redArg___lam__0(lean_object* v_f_93_, lean_object* v_g_94_, lean_object* v_h_95_, lean_object* v_n_96_){
_start:
{
lean_object* v___x_97_; lean_object* v_toFun_98_; lean_object* v_toFun_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_97_ = lp_mathlib_Equiv_symm___redArg(v_f_93_);
v_toFun_98_ = lean_ctor_get(v___x_97_, 0);
lean_inc(v_toFun_98_);
lean_dec_ref(v___x_97_);
v_toFun_99_ = lean_ctor_get(v_g_94_, 0);
lean_inc(v_toFun_99_);
lean_dec_ref(v_g_94_);
v___x_100_ = lean_apply_1(v_toFun_98_, v_n_96_);
v___x_101_ = lean_apply_1(v_h_95_, v___x_100_);
v___x_102_ = lean_apply_1(v_toFun_99_, v___x_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr___redArg___lam__1(lean_object* v_f_103_, lean_object* v_g_104_, lean_object* v_k_105_, lean_object* v_m_106_){
_start:
{
lean_object* v_toFun_107_; lean_object* v___x_108_; lean_object* v_toFun_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v_toFun_107_ = lean_ctor_get(v_f_103_, 0);
lean_inc(v_toFun_107_);
lean_dec_ref(v_f_103_);
v___x_108_ = lp_mathlib_Equiv_symm___redArg(v_g_104_);
v_toFun_109_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_toFun_109_);
lean_dec_ref(v___x_108_);
v___x_110_ = lean_apply_1(v_toFun_107_, v_m_106_);
v___x_111_ = lean_apply_1(v_k_105_, v___x_110_);
v___x_112_ = lean_apply_1(v_toFun_109_, v___x_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr___redArg(lean_object* v_f_113_, lean_object* v_g_114_){
_start:
{
lean_object* v___f_115_; lean_object* v___f_116_; lean_object* v___x_117_; 
lean_inc_ref(v_g_114_);
lean_inc_ref(v_f_113_);
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_arrowCongr___redArg___lam__0), 4, 2);
lean_closure_set(v___f_115_, 0, v_f_113_);
lean_closure_set(v___f_115_, 1, v_g_114_);
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_arrowCongr___redArg___lam__1), 4, 2);
lean_closure_set(v___f_116_, 0, v_f_113_);
lean_closure_set(v___f_116_, 1, v_g_114_);
v___x_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_117_, 0, v___f_115_);
lean_ctor_set(v___x_117_, 1, v___f_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr(lean_object* v_M_118_, lean_object* v_N_119_, lean_object* v_P_120_, lean_object* v_Q_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_f_124_, lean_object* v_g_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_MulEquiv_arrowCongr___redArg(v_f_124_, v_g_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_arrowCongr___boxed(lean_object* v_M_127_, lean_object* v_N_128_, lean_object* v_P_129_, lean_object* v_Q_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_f_133_, lean_object* v_g_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_MulEquiv_arrowCongr(v_M_127_, v_N_128_, v_P_129_, v_Q_130_, v_inst_131_, v_inst_132_, v_f_133_, v_g_134_);
lean_dec(v_inst_132_);
lean_dec(v_inst_131_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_arrowCongr___redArg(lean_object* v_f_136_, lean_object* v_g_137_){
_start:
{
lean_object* v___f_138_; lean_object* v___f_139_; lean_object* v___x_140_; 
lean_inc_ref(v_g_137_);
lean_inc_ref(v_f_136_);
v___f_138_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_arrowCongr___redArg___lam__0), 4, 2);
lean_closure_set(v___f_138_, 0, v_f_136_);
lean_closure_set(v___f_138_, 1, v_g_137_);
v___f_139_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_arrowCongr___redArg___lam__1), 4, 2);
lean_closure_set(v___f_139_, 0, v_f_136_);
lean_closure_set(v___f_139_, 1, v_g_137_);
v___x_140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_140_, 0, v___f_138_);
lean_ctor_set(v___x_140_, 1, v___f_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_arrowCongr(lean_object* v_M_141_, lean_object* v_N_142_, lean_object* v_P_143_, lean_object* v_Q_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_f_147_, lean_object* v_g_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_AddEquiv_arrowCongr___redArg(v_f_147_, v_g_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_arrowCongr___boxed(lean_object* v_M_150_, lean_object* v_N_151_, lean_object* v_P_152_, lean_object* v_Q_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_f_156_, lean_object* v_g_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_AddEquiv_arrowCongr(v_M_150_, v_N_151_, v_P_152_, v_Q_153_, v_inst_154_, v_inst_155_, v_f_156_, v_g_157_);
lean_dec(v_inst_155_);
lean_dec(v_inst_154_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg___lam__0(lean_object* v_e_159_, lean_object* v_f_160_, lean_object* v___y_161_){
_start:
{
lean_object* v_toFun_162_; lean_object* v___x_163_; 
v_toFun_162_ = lean_ctor_get(v_e_159_, 0);
lean_inc(v_toFun_162_);
lean_dec_ref(v_e_159_);
v___x_163_ = lp_mathlib_OneHom_comp___redArg___lam__0(v_toFun_162_, v_f_160_, v___y_161_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg___lam__1(lean_object* v_e_164_, lean_object* v_f_165_, lean_object* v___y_166_){
_start:
{
lean_object* v___x_167_; lean_object* v_toFun_168_; lean_object* v___x_169_; 
v___x_167_ = lp_mathlib_Equiv_symm___redArg(v_e_164_);
v_toFun_168_ = lean_ctor_get(v___x_167_, 0);
lean_inc(v_toFun_168_);
lean_dec_ref(v___x_167_);
v___x_169_ = lp_mathlib_OneHom_comp___redArg___lam__0(v_toFun_168_, v_f_165_, v___y_166_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg(lean_object* v_e_170_){
_start:
{
lean_object* v___f_171_; lean_object* v___f_172_; lean_object* v___x_173_; 
lean_inc_ref(v_e_170_);
v___f_171_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_171_, 0, v_e_170_);
v___f_172_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg___lam__1), 3, 1);
lean_closure_set(v___f_172_, 0, v_e_170_);
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v___f_172_);
lean_ctor_set(v___x_173_, 1, v___f_171_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv(lean_object* v_M_u2081_174_, lean_object* v_M_u2082_175_, lean_object* v_N_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_e_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg(v_e_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___boxed(lean_object* v_M_u2081_182_, lean_object* v_M_u2082_183_, lean_object* v_N_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_e_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv(v_M_u2081_182_, v_M_u2082_183_, v_N_184_, v_inst_185_, v_inst_186_, v_inst_187_, v_e_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
lean_dec_ref(v_inst_185_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv___redArg(lean_object* v_e_190_){
_start:
{
lean_object* v___f_191_; lean_object* v___f_192_; lean_object* v___x_193_; 
lean_inc_ref(v_e_190_);
v___f_191_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_191_, 0, v_e_190_);
v___f_192_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg___lam__1), 3, 1);
lean_closure_set(v___f_192_, 0, v_e_190_);
v___x_193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_193_, 0, v___f_192_);
lean_ctor_set(v___x_193_, 1, v___f_191_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv(lean_object* v_M_u2081_194_, lean_object* v_M_u2082_195_, lean_object* v_N_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_e_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv___redArg(v_e_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv___boxed(lean_object* v_M_u2081_202_, lean_object* v_M_u2082_203_, lean_object* v_N_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_e_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv(v_M_u2081_202_, v_M_u2082_203_, v_N_204_, v_inst_205_, v_inst_206_, v_inst_207_, v_e_208_);
lean_dec_ref(v_inst_207_);
lean_dec_ref(v_inst_206_);
lean_dec_ref(v_inst_205_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___redArg(lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_e_213_){
_start:
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v_toFun_219_; lean_object* v___x_220_; lean_object* v_toFun_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_230_; 
v___x_214_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_210_);
v___x_215_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_211_);
v___x_216_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_215_);
v___x_217_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_212_);
v___x_218_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_217_);
v_toFun_219_ = lean_ctor_get(v_e_213_, 0);
lean_inc(v_toFun_219_);
v___x_220_ = lp_mathlib_Equiv_symm___redArg(v_e_213_);
v_toFun_221_ = lean_ctor_get(v___x_220_, 0);
v_isSharedCheck_230_ = !lean_is_exclusive(v___x_220_);
if (v_isSharedCheck_230_ == 0)
{
lean_object* v_unused_231_; 
v_unused_231_ = lean_ctor_get(v___x_220_, 1);
lean_dec(v_unused_231_);
v___x_223_ = v___x_220_;
v_isShared_224_ = v_isSharedCheck_230_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_toFun_221_);
lean_dec(v___x_220_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_230_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_228_; 
lean_inc_ref(v___x_218_);
lean_inc_ref(v___x_216_);
lean_inc_ref(v___x_214_);
v___x_225_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___boxed), 8, 7);
lean_closure_set(v___x_225_, 0, lean_box(0));
lean_closure_set(v___x_225_, 1, lean_box(0));
lean_closure_set(v___x_225_, 2, lean_box(0));
lean_closure_set(v___x_225_, 3, v___x_214_);
lean_closure_set(v___x_225_, 4, v___x_216_);
lean_closure_set(v___x_225_, 5, v___x_218_);
lean_closure_set(v___x_225_, 6, v_toFun_219_);
v___x_226_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___boxed), 8, 7);
lean_closure_set(v___x_226_, 0, lean_box(0));
lean_closure_set(v___x_226_, 1, lean_box(0));
lean_closure_set(v___x_226_, 2, lean_box(0));
lean_closure_set(v___x_226_, 3, v___x_214_);
lean_closure_set(v___x_226_, 4, v___x_218_);
lean_closure_set(v___x_226_, 5, v___x_216_);
lean_closure_set(v___x_226_, 6, v_toFun_221_);
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 1, v___x_226_);
lean_ctor_set(v___x_223_, 0, v___x_225_);
v___x_228_ = v___x_223_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v___x_225_);
lean_ctor_set(v_reuseFailAlloc_229_, 1, v___x_226_);
v___x_228_ = v_reuseFailAlloc_229_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
return v___x_228_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___redArg___boxed(lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_e_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___redArg(v_inst_232_, v_inst_233_, v_inst_234_, v_e_235_);
lean_dec_ref(v_inst_234_);
lean_dec_ref(v_inst_233_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRightEquiv(lean_object* v_M_237_, lean_object* v_N_u2081_238_, lean_object* v_N_u2082_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_e_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___redArg(v_inst_240_, v_inst_241_, v_inst_242_, v_e_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___boxed(lean_object* v_M_245_, lean_object* v_N_u2081_246_, lean_object* v_N_u2082_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_e_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib_MulEquiv_monoidHomCongrRightEquiv(v_M_245_, v_N_u2081_246_, v_N_u2082_247_, v_inst_248_, v_inst_249_, v_inst_250_, v_e_251_);
lean_dec_ref(v_inst_250_);
lean_dec_ref(v_inst_249_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___redArg(lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_e_256_){
_start:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v_toFun_262_; lean_object* v___x_263_; lean_object* v_toFun_264_; lean_object* v___x_266_; uint8_t v_isShared_267_; uint8_t v_isSharedCheck_273_; 
v___x_257_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_253_);
v___x_258_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_254_);
v___x_259_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_258_);
v___x_260_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_255_);
v___x_261_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_260_);
v_toFun_262_ = lean_ctor_get(v_e_256_, 0);
lean_inc(v_toFun_262_);
v___x_263_ = lp_mathlib_Equiv_symm___redArg(v_e_256_);
v_toFun_264_ = lean_ctor_get(v___x_263_, 0);
v_isSharedCheck_273_ = !lean_is_exclusive(v___x_263_);
if (v_isSharedCheck_273_ == 0)
{
lean_object* v_unused_274_; 
v_unused_274_ = lean_ctor_get(v___x_263_, 1);
lean_dec(v_unused_274_);
v___x_266_ = v___x_263_;
v_isShared_267_ = v_isSharedCheck_273_;
goto v_resetjp_265_;
}
else
{
lean_inc(v_toFun_264_);
lean_dec(v___x_263_);
v___x_266_ = lean_box(0);
v_isShared_267_ = v_isSharedCheck_273_;
goto v_resetjp_265_;
}
v_resetjp_265_:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_271_; 
lean_inc_ref(v___x_261_);
lean_inc_ref(v___x_259_);
lean_inc_ref(v___x_257_);
v___x_268_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_comp___boxed), 8, 7);
lean_closure_set(v___x_268_, 0, lean_box(0));
lean_closure_set(v___x_268_, 1, lean_box(0));
lean_closure_set(v___x_268_, 2, lean_box(0));
lean_closure_set(v___x_268_, 3, v___x_257_);
lean_closure_set(v___x_268_, 4, v___x_259_);
lean_closure_set(v___x_268_, 5, v___x_261_);
lean_closure_set(v___x_268_, 6, v_toFun_262_);
v___x_269_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_comp___boxed), 8, 7);
lean_closure_set(v___x_269_, 0, lean_box(0));
lean_closure_set(v___x_269_, 1, lean_box(0));
lean_closure_set(v___x_269_, 2, lean_box(0));
lean_closure_set(v___x_269_, 3, v___x_257_);
lean_closure_set(v___x_269_, 4, v___x_261_);
lean_closure_set(v___x_269_, 5, v___x_259_);
lean_closure_set(v___x_269_, 6, v_toFun_264_);
if (v_isShared_267_ == 0)
{
lean_ctor_set(v___x_266_, 1, v___x_269_);
lean_ctor_set(v___x_266_, 0, v___x_268_);
v___x_271_ = v___x_266_;
goto v_reusejp_270_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v___x_268_);
lean_ctor_set(v_reuseFailAlloc_272_, 1, v___x_269_);
v___x_271_ = v_reuseFailAlloc_272_;
goto v_reusejp_270_;
}
v_reusejp_270_:
{
return v___x_271_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___redArg___boxed(lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_e_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___redArg(v_inst_275_, v_inst_276_, v_inst_277_, v_e_278_);
lean_dec_ref(v_inst_277_);
lean_dec_ref(v_inst_276_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv(lean_object* v_M_280_, lean_object* v_N_u2081_281_, lean_object* v_N_u2082_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_e_286_){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___redArg(v_inst_283_, v_inst_284_, v_inst_285_, v_e_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___boxed(lean_object* v_M_288_, lean_object* v_N_u2081_289_, lean_object* v_N_u2082_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_e_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv(v_M_288_, v_N_u2081_289_, v_N_u2082_290_, v_inst_291_, v_inst_292_, v_inst_293_, v_e_294_);
lean_dec_ref(v_inst_293_);
lean_dec_ref(v_inst_292_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeft___redArg(lean_object* v_e_296_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg(v_e_296_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeft(lean_object* v_M_u2081_298_, lean_object* v_M_u2082_299_, lean_object* v_N_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_e_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_mathlib_MulEquiv_monoidHomCongrLeftEquiv___redArg(v_e_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrLeft___boxed(lean_object* v_M_u2081_306_, lean_object* v_M_u2082_307_, lean_object* v_N_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_e_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_mathlib_MulEquiv_monoidHomCongrLeft(v_M_u2081_306_, v_M_u2082_307_, v_N_308_, v_inst_309_, v_inst_310_, v_inst_311_, v_e_312_);
lean_dec_ref(v_inst_311_);
lean_dec_ref(v_inst_310_);
lean_dec_ref(v_inst_309_);
return v_res_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeft___redArg(lean_object* v_e_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv___redArg(v_e_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeft(lean_object* v_M_u2081_316_, lean_object* v_M_u2082_317_, lean_object* v_N_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_e_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lp_mathlib_AddEquiv_addMonoidHomCongrLeftEquiv___redArg(v_e_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrLeft___boxed(lean_object* v_M_u2081_324_, lean_object* v_M_u2082_325_, lean_object* v_N_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_e_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_mathlib_AddEquiv_addMonoidHomCongrLeft(v_M_u2081_324_, v_M_u2082_325_, v_N_326_, v_inst_327_, v_inst_328_, v_inst_329_, v_e_330_);
lean_dec_ref(v_inst_329_);
lean_dec_ref(v_inst_328_);
lean_dec_ref(v_inst_327_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRight___redArg(lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_e_335_){
_start:
{
lean_object* v___x_336_; 
v___x_336_ = lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___redArg(v_inst_332_, v_inst_333_, v_inst_334_, v_e_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRight___redArg___boxed(lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_e_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_MulEquiv_monoidHomCongrRight___redArg(v_inst_337_, v_inst_338_, v_inst_339_, v_e_340_);
lean_dec_ref(v_inst_339_);
lean_dec_ref(v_inst_338_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRight(lean_object* v_M_342_, lean_object* v_N_u2081_343_, lean_object* v_N_u2082_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_e_348_){
_start:
{
lean_object* v___x_349_; 
v___x_349_ = lp_mathlib_MulEquiv_monoidHomCongrRightEquiv___redArg(v_inst_345_, v_inst_346_, v_inst_347_, v_e_348_);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidHomCongrRight___boxed(lean_object* v_M_350_, lean_object* v_N_u2081_351_, lean_object* v_N_u2082_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_e_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_MulEquiv_monoidHomCongrRight(v_M_350_, v_N_u2081_351_, v_N_u2082_352_, v_inst_353_, v_inst_354_, v_inst_355_, v_e_356_);
lean_dec_ref(v_inst_355_);
lean_dec_ref(v_inst_354_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRight___redArg(lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_e_361_){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___redArg(v_inst_358_, v_inst_359_, v_inst_360_, v_e_361_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRight___redArg___boxed(lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_e_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_AddEquiv_addMonoidHomCongrRight___redArg(v_inst_363_, v_inst_364_, v_inst_365_, v_e_366_);
lean_dec_ref(v_inst_365_);
lean_dec_ref(v_inst_364_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRight(lean_object* v_M_368_, lean_object* v_N_u2081_369_, lean_object* v_N_u2082_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_e_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lp_mathlib_AddEquiv_addMonoidHomCongrRightEquiv___redArg(v_inst_371_, v_inst_372_, v_inst_373_, v_e_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addMonoidHomCongrRight___boxed(lean_object* v_M_376_, lean_object* v_N_u2081_377_, lean_object* v_N_u2082_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_e_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_AddEquiv_addMonoidHomCongrRight(v_M_376_, v_N_u2081_377_, v_N_u2082_378_, v_inst_379_, v_inst_380_, v_inst_381_, v_e_382_);
lean_dec_ref(v_inst_381_);
lean_dec_ref(v_inst_380_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight___redArg___lam__0(lean_object* v_es_384_, lean_object* v_x_385_, lean_object* v_j_386_){
_start:
{
lean_object* v___x_387_; lean_object* v_toFun_388_; lean_object* v___x_389_; lean_object* v___x_390_; 
lean_inc(v_j_386_);
v___x_387_ = lean_apply_1(v_es_384_, v_j_386_);
v_toFun_388_ = lean_ctor_get(v___x_387_, 0);
lean_inc(v_toFun_388_);
lean_dec_ref(v___x_387_);
v___x_389_ = lean_apply_1(v_x_385_, v_j_386_);
v___x_390_ = lean_apply_1(v_toFun_388_, v___x_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight___redArg___lam__1(lean_object* v_es_391_, lean_object* v_x_392_, lean_object* v_j_393_){
_start:
{
lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v_toFun_396_; lean_object* v___x_397_; lean_object* v___x_398_; 
lean_inc(v_j_393_);
v___x_394_ = lean_apply_1(v_es_391_, v_j_393_);
v___x_395_ = lp_mathlib_Equiv_symm___redArg(v___x_394_);
v_toFun_396_ = lean_ctor_get(v___x_395_, 0);
lean_inc(v_toFun_396_);
lean_dec_ref(v___x_395_);
v___x_397_ = lean_apply_1(v_x_392_, v_j_393_);
v___x_398_ = lean_apply_1(v_toFun_396_, v___x_397_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight___redArg(lean_object* v_es_399_){
_start:
{
lean_object* v___f_400_; lean_object* v___f_401_; lean_object* v___x_402_; 
lean_inc_ref(v_es_399_);
v___f_400_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_piCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(v___f_400_, 0, v_es_399_);
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_piCongrRight___redArg___lam__1), 3, 1);
lean_closure_set(v___f_401_, 0, v_es_399_);
v___x_402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_402_, 0, v___f_400_);
lean_ctor_set(v___x_402_, 1, v___f_401_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight(lean_object* v_00_u03b7_403_, lean_object* v_Ms_404_, lean_object* v_Ns_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_es_408_){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lp_mathlib_MulEquiv_piCongrRight___redArg(v_es_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piCongrRight___boxed(lean_object* v_00_u03b7_410_, lean_object* v_Ms_411_, lean_object* v_Ns_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_es_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_MulEquiv_piCongrRight(v_00_u03b7_410_, v_Ms_411_, v_Ns_412_, v_inst_413_, v_inst_414_, v_es_415_);
lean_dec(v_inst_414_);
lean_dec(v_inst_413_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piCongrRight___redArg(lean_object* v_es_417_){
_start:
{
lean_object* v___f_418_; lean_object* v___f_419_; lean_object* v___x_420_; 
lean_inc_ref(v_es_417_);
v___f_418_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_piCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(v___f_418_, 0, v_es_417_);
v___f_419_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_piCongrRight___redArg___lam__1), 3, 1);
lean_closure_set(v___f_419_, 0, v_es_417_);
v___x_420_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_420_, 0, v___f_418_);
lean_ctor_set(v___x_420_, 1, v___f_419_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piCongrRight(lean_object* v_00_u03b7_421_, lean_object* v_Ms_422_, lean_object* v_Ns_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_es_426_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = lp_mathlib_AddEquiv_piCongrRight___redArg(v_es_426_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piCongrRight___boxed(lean_object* v_00_u03b7_428_, lean_object* v_Ms_429_, lean_object* v_Ns_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_es_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_AddEquiv_piCongrRight(v_00_u03b7_428_, v_Ms_429_, v_Ns_430_, v_inst_431_, v_inst_432_, v_es_433_);
lean_dec(v_inst_432_);
lean_dec(v_inst_431_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piUnique___redArg(lean_object* v_inst_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piUnique(lean_object* v_00_u03b9_437_, lean_object* v_M_438_, lean_object* v_inst_439_, lean_object* v_inst_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piUnique___boxed(lean_object* v_00_u03b9_442_, lean_object* v_M_443_, lean_object* v_inst_444_, lean_object* v_inst_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_mathlib_MulEquiv_piUnique(v_00_u03b9_442_, v_M_443_, v_inst_444_, v_inst_445_);
lean_dec(v_inst_444_);
return v_res_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piUnique___redArg(lean_object* v_inst_447_){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piUnique(lean_object* v_00_u03b9_449_, lean_object* v_M_450_, lean_object* v_inst_451_, lean_object* v_inst_452_){
_start:
{
lean_object* v___x_453_; 
v___x_453_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_452_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piUnique___boxed(lean_object* v_00_u03b9_454_, lean_object* v_M_455_, lean_object* v_inst_456_, lean_object* v_inst_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_AddEquiv_piUnique(v_00_u03b9_454_, v_M_455_, v_inst_456_, v_inst_457_);
lean_dec(v_inst_456_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inv___redArg(lean_object* v_inst_459_){
_start:
{
lean_object* v___x_460_; 
lean_inc(v_inst_459_);
v___x_460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_460_, 0, v_inst_459_);
lean_ctor_set(v___x_460_, 1, v_inst_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_inv(lean_object* v_G_461_, lean_object* v_inst_462_){
_start:
{
lean_object* v___x_463_; 
lean_inc(v_inst_462_);
v___x_463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_463_, 0, v_inst_462_);
lean_ctor_set(v___x_463_, 1, v_inst_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_neg___redArg(lean_object* v_inst_464_){
_start:
{
lean_object* v___x_465_; 
lean_inc(v_inst_464_);
v___x_465_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_465_, 0, v_inst_464_);
lean_ctor_set(v___x_465_, 1, v_inst_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_neg(lean_object* v_G_466_, lean_object* v_inst_467_){
_start:
{
lean_object* v___x_468_; 
lean_inc(v_inst_467_);
v___x_468_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_468_, 0, v_inst_467_);
lean_ctor_set(v___x_468_, 1, v_inst_467_);
return v___x_468_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
