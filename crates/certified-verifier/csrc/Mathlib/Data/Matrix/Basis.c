// Lean compiler output
// Module: Mathlib.Data.Matrix.Basis
// Imports: public import Init public meta import Init public import Mathlib.Data.Matrix.Basic
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
lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_lsum___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_piCongrRight___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_ofLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_congrLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_single___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_single___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Matrix_single___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_single___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_single___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleAddMonoidHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleAddMonoidHom___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleLinearMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleLinearMap___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_single___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_j_5_, lean_object* v_a_6_, lean_object* v_i_x27_7_, lean_object* v_j_x27_8_){
_start:
{
lean_object* v___x_9_; uint8_t v___x_10_; 
v___x_9_ = lean_apply_2(v_inst_1_, v_i_2_, v_i_x27_7_);
v___x_10_ = lean_unbox(v___x_9_);
if (v___x_10_ == 0)
{
lean_dec(v_j_x27_8_);
lean_dec(v_j_5_);
lean_dec_ref(v_inst_4_);
lean_inc(v_inst_3_);
return v_inst_3_;
}
else
{
lean_object* v___x_11_; uint8_t v___x_12_; 
v___x_11_ = lean_apply_2(v_inst_4_, v_j_5_, v_j_x27_8_);
v___x_12_ = lean_unbox(v___x_11_);
if (v___x_12_ == 0)
{
lean_inc(v_inst_3_);
return v_inst_3_;
}
else
{
lean_inc(v_a_6_);
return v_a_6_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_single___redArg___lam__0___boxed(lean_object* v_inst_13_, lean_object* v_i_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_j_17_, lean_object* v_a_18_, lean_object* v_i_x27_19_, lean_object* v_j_x27_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Matrix_single___redArg___lam__0(v_inst_13_, v_i_14_, v_inst_15_, v_inst_16_, v_j_17_, v_a_18_, v_i_x27_19_, v_j_x27_20_);
lean_dec(v_a_18_);
lean_dec(v_inst_15_);
return v_res_21_;
}
}
static lean_object* _init_lp_mathlib_Matrix_single___redArg___closed__0(void){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_single___redArg(lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_i_26_, lean_object* v_j_27_, lean_object* v_a_28_, lean_object* v_a_29_, lean_object* v_a_30_){
_start:
{
lean_object* v___x_31_; lean_object* v_toFun_32_; lean_object* v___f_33_; lean_object* v___x_34_; 
v___x_31_ = lean_obj_once(&lp_mathlib_Matrix_single___redArg___closed__0, &lp_mathlib_Matrix_single___redArg___closed__0_once, _init_lp_mathlib_Matrix_single___redArg___closed__0);
v_toFun_32_ = lean_ctor_get(v___x_31_, 0);
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_single___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_33_, 0, v_inst_23_);
lean_closure_set(v___f_33_, 1, v_i_26_);
lean_closure_set(v___f_33_, 2, v_inst_25_);
lean_closure_set(v___f_33_, 3, v_inst_24_);
lean_closure_set(v___f_33_, 4, v_j_27_);
lean_closure_set(v___f_33_, 5, v_a_28_);
lean_inc(v_toFun_32_);
v___x_34_ = lean_apply_3(v_toFun_32_, v___f_33_, v_a_29_, v_a_30_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_single(lean_object* v_m_35_, lean_object* v_n_36_, lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_i_41_, lean_object* v_j_42_, lean_object* v_a_43_, lean_object* v_a_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_Matrix_single___redArg(v_inst_38_, v_inst_39_, v_inst_40_, v_i_41_, v_j_42_, v_a_43_, v_a_44_, v_a_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleAddMonoidHom___redArg(lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_i_50_, lean_object* v_j_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v_toZero_54_; lean_object* v___x_55_; 
v___x_52_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_49_);
v___x_53_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_52_);
v_toZero_54_ = lean_ctor_get(v___x_53_, 0);
lean_inc(v_toZero_54_);
lean_dec_ref(v___x_53_);
v___x_55_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_single), 11, 8);
lean_closure_set(v___x_55_, 0, lean_box(0));
lean_closure_set(v___x_55_, 1, lean_box(0));
lean_closure_set(v___x_55_, 2, lean_box(0));
lean_closure_set(v___x_55_, 3, v_inst_47_);
lean_closure_set(v___x_55_, 4, v_inst_48_);
lean_closure_set(v___x_55_, 5, v_toZero_54_);
lean_closure_set(v___x_55_, 6, v_i_50_);
lean_closure_set(v___x_55_, 7, v_j_51_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleAddMonoidHom___redArg___boxed(lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_i_59_, lean_object* v_j_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Matrix_singleAddMonoidHom___redArg(v_inst_56_, v_inst_57_, v_inst_58_, v_i_59_, v_j_60_);
lean_dec_ref(v_inst_58_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleAddMonoidHom(lean_object* v_m_62_, lean_object* v_n_63_, lean_object* v_00_u03b1_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_i_68_, lean_object* v_j_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Matrix_singleAddMonoidHom___redArg(v_inst_65_, v_inst_66_, v_inst_67_, v_i_68_, v_j_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleAddMonoidHom___boxed(lean_object* v_m_71_, lean_object* v_n_72_, lean_object* v_00_u03b1_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_i_77_, lean_object* v_j_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_Matrix_singleAddMonoidHom(v_m_71_, v_n_72_, v_00_u03b1_73_, v_inst_74_, v_inst_75_, v_inst_76_, v_i_77_, v_j_78_);
lean_dec_ref(v_inst_76_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleLinearMap___redArg(lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_i_83_, lean_object* v_j_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_Matrix_singleAddMonoidHom___redArg(v_inst_80_, v_inst_81_, v_inst_82_, v_i_83_, v_j_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleLinearMap___redArg___boxed(lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_i_89_, lean_object* v_j_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_Matrix_singleLinearMap___redArg(v_inst_86_, v_inst_87_, v_inst_88_, v_i_89_, v_j_90_);
lean_dec_ref(v_inst_88_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleLinearMap(lean_object* v_m_92_, lean_object* v_n_93_, lean_object* v_R_94_, lean_object* v_00_u03b1_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_i_101_, lean_object* v_j_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_mathlib_Matrix_singleAddMonoidHom___redArg(v_inst_96_, v_inst_97_, v_inst_99_, v_i_101_, v_j_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_singleLinearMap___boxed(lean_object* v_m_104_, lean_object* v_n_105_, lean_object* v_R_106_, lean_object* v_00_u03b1_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_i_113_, lean_object* v_j_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_Matrix_singleLinearMap(v_m_104_, v_n_105_, v_R_106_, v_00_u03b1_107_, v_inst_108_, v_inst_109_, v_inst_110_, v_inst_111_, v_inst_112_, v_i_113_, v_j_114_);
lean_dec(v_inst_112_);
lean_dec_ref(v_inst_111_);
lean_dec_ref(v_inst_110_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__0(lean_object* v_inst_116_, lean_object* v_i_117_){
_start:
{
lean_inc_ref(v_inst_116_);
return v_inst_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__0___boxed(lean_object* v_inst_118_, lean_object* v_i_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_Matrix_liftLinear___redArg___lam__0(v_inst_118_, v_i_119_);
lean_dec(v_i_119_);
lean_dec_ref(v_inst_118_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__1(lean_object* v___f_121_, lean_object* v_i_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_Pi_addCommMonoid___redArg(v___f_121_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__1___boxed(lean_object* v___f_124_, lean_object* v_i_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Matrix_liftLinear___redArg___lam__1(v___f_124_, v_i_125_);
lean_dec(v_i_125_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__3(lean_object* v___f_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_x_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_LinearMap_lsum___redArg(v___f_127_, v_inst_128_, v_inst_129_, v_inst_130_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___lam__3___boxed(lean_object* v___f_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_x_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Matrix_liftLinear___redArg___lam__3(v___f_133_, v_inst_134_, v_inst_135_, v_inst_136_, v_x_137_);
lean_dec(v_x_137_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg(lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___f_148_; lean_object* v___f_149_; lean_object* v___f_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
lean_inc_ref(v_inst_144_);
v___f_148_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_liftLinear___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_148_, 0, v_inst_144_);
lean_inc_ref(v___f_148_);
v___f_149_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_liftLinear___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_149_, 0, v___f_148_);
lean_inc_ref_n(v_inst_145_, 2);
v___f_150_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_liftLinear___redArg___lam__3___boxed), 5, 4);
lean_closure_set(v___f_150_, 0, v___f_148_);
lean_closure_set(v___f_150_, 1, v_inst_140_);
lean_closure_set(v___f_150_, 2, v_inst_145_);
lean_closure_set(v___f_150_, 3, v_inst_142_);
v___x_151_ = lp_mathlib_LinearEquiv_piCongrRight___redArg(v___f_150_);
v___x_152_ = lp_mathlib_LinearMap_lsum___redArg(v___f_149_, v_inst_139_, v_inst_145_, v_inst_141_);
v___x_153_ = lp_mathlib_LinearEquiv_trans___redArg(v___x_151_, v___x_152_);
v___x_154_ = lp_mathlib_Matrix_ofLinearEquiv(lean_box(0), lean_box(0), lean_box(0), lean_box(0), v_inst_143_, v_inst_144_, v_inst_146_);
lean_dec_ref(v_inst_144_);
v___x_155_ = lp_mathlib_LinearEquiv_congrLeft___redArg(v_inst_145_, v_inst_143_, v_inst_147_, v___x_154_);
lean_dec_ref(v_inst_145_);
v___x_156_ = lp_mathlib_LinearEquiv_trans___redArg(v___x_153_, v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___redArg___boxed(lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_Matrix_liftLinear___redArg(v_inst_157_, v_inst_158_, v_inst_159_, v_inst_160_, v_inst_161_, v_inst_162_, v_inst_163_, v_inst_164_, v_inst_165_);
lean_dec(v_inst_165_);
lean_dec(v_inst_164_);
lean_dec_ref(v_inst_161_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear(lean_object* v_m_167_, lean_object* v_n_168_, lean_object* v_R_169_, lean_object* v_S_170_, lean_object* v_00_u03b1_171_, lean_object* v_00_u03b2_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_Matrix_liftLinear___redArg(v_inst_173_, v_inst_174_, v_inst_175_, v_inst_176_, v_inst_177_, v_inst_179_, v_inst_180_, v_inst_181_, v_inst_182_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_liftLinear___boxed(lean_object** _args){
lean_object* v_m_186_ = _args[0];
lean_object* v_n_187_ = _args[1];
lean_object* v_R_188_ = _args[2];
lean_object* v_S_189_ = _args[3];
lean_object* v_00_u03b1_190_ = _args[4];
lean_object* v_00_u03b2_191_ = _args[5];
lean_object* v_inst_192_ = _args[6];
lean_object* v_inst_193_ = _args[7];
lean_object* v_inst_194_ = _args[8];
lean_object* v_inst_195_ = _args[9];
lean_object* v_inst_196_ = _args[10];
lean_object* v_inst_197_ = _args[11];
lean_object* v_inst_198_ = _args[12];
lean_object* v_inst_199_ = _args[13];
lean_object* v_inst_200_ = _args[14];
lean_object* v_inst_201_ = _args[15];
lean_object* v_inst_202_ = _args[16];
lean_object* v_inst_203_ = _args[17];
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Matrix_liftLinear(v_m_186_, v_n_187_, v_R_188_, v_S_189_, v_00_u03b1_190_, v_00_u03b2_191_, v_inst_192_, v_inst_193_, v_inst_194_, v_inst_195_, v_inst_196_, v_inst_197_, v_inst_198_, v_inst_199_, v_inst_200_, v_inst_201_, v_inst_202_, v_inst_203_);
lean_dec(v_inst_202_);
lean_dec(v_inst_201_);
lean_dec(v_inst_200_);
lean_dec_ref(v_inst_197_);
lean_dec_ref(v_inst_196_);
return v_res_204_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Matrix_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Matrix_Basis(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Matrix_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Matrix_Basis(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Matrix_Basis(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Matrix_Basis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Matrix_Basis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Matrix_Basis(builtin);
}
#ifdef __cplusplus
}
#endif
