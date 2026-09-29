// Lean compiler output
// Module: Mathlib.Data.DFinsupp.BigOperators
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.GroupWithZero.Action public import Mathlib.Data.DFinsupp.Ext public import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
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
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_coeFnAddMonoidHom___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_support___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_prod___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_evalMulHom___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_singleAddHom___redArg(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_evalAddMonoidHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_coeFnAddMonoidHom___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_evalAddMonoidHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_evalAddMonoidHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_evalAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_evalAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_evalAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sum___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sum___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumAddHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumAddHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_liftAddHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_liftAddHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_liftAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_evalAddMonoidHom___redArg(lean_object* v_i_2_){
_start:
{
lean_object* v___f_3_; lean_object* v___f_4_; lean_object* v___f_5_; 
v___f_3_ = lean_alloc_closure((void*)(lp_mathlib_Pi_evalMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_3_, 0, v_i_2_);
v___f_4_ = ((lean_object*)(lp_mathlib_DFinsupp_evalAddMonoidHom___redArg___closed__0));
v___f_5_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_5_, 0, v___f_4_);
lean_closure_set(v___f_5_, 1, v___f_3_);
return v___f_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_evalAddMonoidHom(lean_object* v_00_u03b9_6_, lean_object* v_00_u03b2_7_, lean_object* v_inst_8_, lean_object* v_i_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_DFinsupp_evalAddMonoidHom___redArg(v_i_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_evalAddMonoidHom___boxed(lean_object* v_00_u03b9_11_, lean_object* v_00_u03b2_12_, lean_object* v_inst_13_, lean_object* v_i_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_DFinsupp_evalAddMonoidHom(v_00_u03b9_11_, v_00_u03b2_12_, v_inst_13_, v_i_14_);
lean_dec_ref(v_inst_13_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod___redArg___lam__0(lean_object* v_f_16_, lean_object* v_g_17_, lean_object* v_i_18_){
_start:
{
lean_object* v_toFun_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v_toFun_19_ = lean_ctor_get(v_f_16_, 0);
lean_inc(v_toFun_19_);
lean_dec_ref(v_f_16_);
lean_inc(v_i_18_);
v___x_20_ = lean_apply_1(v_toFun_19_, v_i_18_);
v___x_21_ = lean_apply_2(v_g_17_, v_i_18_, v___x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod___redArg(lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_f_25_, lean_object* v_g_26_){
_start:
{
lean_object* v___f_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
lean_inc_ref(v_f_25_);
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_27_, 0, v_f_25_);
lean_closure_set(v___f_27_, 1, v_g_26_);
v___x_28_ = lp_mathlib_DFinsupp_support___redArg(v_inst_22_, v_inst_23_, v_f_25_);
v___x_29_ = lp_mathlib_Finset_prod___redArg(v_inst_24_, v___x_28_, v___f_27_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod___redArg___boxed(lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_f_33_, lean_object* v_g_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_DFinsupp_prod___redArg(v_inst_30_, v_inst_31_, v_inst_32_, v_f_33_, v_g_34_);
lean_dec_ref(v_inst_32_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod(lean_object* v_00_u03b9_36_, lean_object* v_00_u03b3_37_, lean_object* v_00_u03b2_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_f_43_, lean_object* v_g_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_DFinsupp_prod___redArg(v_inst_39_, v_inst_41_, v_inst_42_, v_f_43_, v_g_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_prod___boxed(lean_object* v_00_u03b9_46_, lean_object* v_00_u03b3_47_, lean_object* v_00_u03b2_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_f_53_, lean_object* v_g_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_DFinsupp_prod(v_00_u03b9_46_, v_00_u03b3_47_, v_00_u03b2_48_, v_inst_49_, v_inst_50_, v_inst_51_, v_inst_52_, v_f_53_, v_g_54_);
lean_dec_ref(v_inst_52_);
lean_dec(v_inst_50_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sum___redArg(lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_f_59_, lean_object* v_g_60_){
_start:
{
lean_object* v___f_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
lean_inc_ref(v_f_59_);
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_61_, 0, v_f_59_);
lean_closure_set(v___f_61_, 1, v_g_60_);
v___x_62_ = lp_mathlib_DFinsupp_support___redArg(v_inst_56_, v_inst_57_, v_f_59_);
v___x_63_ = lp_mathlib_Finset_sum___redArg(v_inst_58_, v___x_62_, v___f_61_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sum___redArg___boxed(lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_f_67_, lean_object* v_g_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_DFinsupp_sum___redArg(v_inst_64_, v_inst_65_, v_inst_66_, v_f_67_, v_g_68_);
lean_dec_ref(v_inst_66_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sum(lean_object* v_00_u03b9_70_, lean_object* v_00_u03b3_71_, lean_object* v_00_u03b2_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_f_77_, lean_object* v_g_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_DFinsupp_sum___redArg(v_inst_73_, v_inst_75_, v_inst_76_, v_f_77_, v_g_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sum___boxed(lean_object* v_00_u03b9_80_, lean_object* v_00_u03b3_81_, lean_object* v_00_u03b2_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_f_87_, lean_object* v_g_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_DFinsupp_sum(v_00_u03b9_80_, v_00_u03b3_81_, v_00_u03b2_82_, v_inst_83_, v_inst_84_, v_inst_85_, v_inst_86_, v_f_87_, v_g_88_);
lean_dec_ref(v_inst_86_);
lean_dec(v_inst_84_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__0(lean_object* v_toFun_90_, lean_object* v_00_u03c6_91_, lean_object* v_i_92_){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; 
lean_inc(v_i_92_);
v___x_93_ = lean_apply_1(v_toFun_90_, v_i_92_);
v___x_94_ = lean_apply_2(v_00_u03c6_91_, v_i_92_, v___x_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__1(lean_object* v_00_u03c6_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_f_98_){
_start:
{
lean_object* v_toFun_99_; lean_object* v_support_x27_100_; lean_object* v___f_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v_toFun_99_ = lean_ctor_get(v_f_98_, 0);
lean_inc(v_toFun_99_);
v_support_x27_100_ = lean_ctor_get(v_f_98_, 1);
lean_inc(v_support_x27_100_);
lean_dec_ref(v_f_98_);
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_101_, 0, v_toFun_99_);
lean_closure_set(v___f_101_, 1, v_00_u03c6_95_);
v___x_102_ = lp_mathlib_List_dedup___redArg(v_inst_96_, v_support_x27_100_);
v___x_103_ = lp_mathlib_Finset_sum___redArg(v_inst_97_, v___x_102_, v___f_101_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__1___boxed(lean_object* v_00_u03c6_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_f_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__1(v_00_u03c6_104_, v_inst_105_, v_inst_106_, v_f_107_);
lean_dec_ref(v_inst_106_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___redArg(lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_00_u03c6_111_){
_start:
{
lean_object* v___f_112_; 
v___f_112_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_112_, 0, v_00_u03c6_111_);
lean_closure_set(v___f_112_, 1, v_inst_109_);
lean_closure_set(v___f_112_, 2, v_inst_110_);
return v___f_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom(lean_object* v_00_u03b9_113_, lean_object* v_00_u03b3_114_, lean_object* v_00_u03b2_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_00_u03c6_119_){
_start:
{
lean_object* v___f_120_; 
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_120_, 0, v_00_u03c6_119_);
lean_closure_set(v___f_120_, 1, v_inst_116_);
lean_closure_set(v___f_120_, 2, v_inst_118_);
return v___f_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumZeroHom___boxed(lean_object* v_00_u03b9_121_, lean_object* v_00_u03b3_122_, lean_object* v_00_u03b2_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_00_u03c6_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_DFinsupp_sumZeroHom(v_00_u03b9_121_, v_00_u03b3_122_, v_00_u03b2_123_, v_inst_124_, v_inst_125_, v_inst_126_, v_00_u03c6_127_);
lean_dec(v_inst_125_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumAddHom___redArg___lam__0(lean_object* v_00_u03c6_129_, lean_object* v_i_130_, lean_object* v___y_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lean_apply_2(v_00_u03c6_129_, v_i_130_, v___y_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumAddHom___redArg(lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_00_u03c6_135_){
_start:
{
lean_object* v___f_136_; lean_object* v___f_137_; 
v___f_136_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sumAddHom___redArg___lam__0), 3, 1);
lean_closure_set(v___f_136_, 0, v_00_u03c6_135_);
v___f_137_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sumZeroHom___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_137_, 0, v___f_136_);
lean_closure_set(v___f_137_, 1, v_inst_133_);
lean_closure_set(v___f_137_, 2, v_inst_134_);
return v___f_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumAddHom(lean_object* v_00_u03b9_138_, lean_object* v_00_u03b3_139_, lean_object* v_00_u03b2_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_00_u03c6_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib_DFinsupp_sumAddHom___redArg(v_inst_141_, v_inst_143_, v_00_u03c6_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sumAddHom___boxed(lean_object* v_00_u03b9_146_, lean_object* v_00_u03b3_147_, lean_object* v_00_u03b2_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_00_u03c6_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_DFinsupp_sumAddHom(v_00_u03b9_146_, v_00_u03b3_147_, v_00_u03b2_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_00_u03c6_152_);
lean_dec_ref(v_inst_150_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_liftAddHom___redArg___lam__0(lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_F_156_, lean_object* v_i_157_, lean_object* v___y_158_){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_159_ = lp_mathlib_DFinsupp_singleAddHom___redArg(v_inst_154_, v_inst_155_, v_i_157_);
v___x_160_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_159_, v_F_156_, v___y_158_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_liftAddHom___redArg(lean_object* v_inst_161_, lean_object* v_inst_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v___f_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
lean_inc_ref(v_inst_162_);
lean_inc_ref(v_inst_161_);
v___f_164_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_liftAddHom___redArg___lam__0), 5, 2);
lean_closure_set(v___f_164_, 0, v_inst_161_);
lean_closure_set(v___f_164_, 1, v_inst_162_);
v___x_165_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sumAddHom___boxed), 7, 6);
lean_closure_set(v___x_165_, 0, lean_box(0));
lean_closure_set(v___x_165_, 1, lean_box(0));
lean_closure_set(v___x_165_, 2, lean_box(0));
lean_closure_set(v___x_165_, 3, v_inst_161_);
lean_closure_set(v___x_165_, 4, v_inst_162_);
lean_closure_set(v___x_165_, 5, v_inst_163_);
v___x_166_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_166_, 0, v___x_165_);
lean_ctor_set(v___x_166_, 1, v___f_164_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_liftAddHom(lean_object* v_00_u03b9_167_, lean_object* v_00_u03b3_168_, lean_object* v_00_u03b2_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_mathlib_DFinsupp_liftAddHom___redArg(v_inst_170_, v_inst_171_, v_inst_172_);
return v___x_173_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Ext(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_BigOperators(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_DFinsupp_BigOperators(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Ext(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_BigOperators(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_GroupWithZero_Action(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_DFinsupp_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_DFinsupp_BigOperators(builtin);
}
#ifdef __cplusplus
}
#endif
