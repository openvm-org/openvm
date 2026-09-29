// Lean compiler output
// Module: Mathlib.Order.SuccPred.Archimedean
// Imports: public import Init public meta import Init public import Mathlib.Order.SuccPred.Basic
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
lean_object* lp_mathlib_OrderDual_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___aux__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___aux__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
lean_inc(v_b_3_);
lean_inc(v_a_2_);
v___x_4_ = lean_apply_2(v_inst_1_, v_a_2_, v_b_3_);
v___x_5_ = lean_unbox(v___x_4_);
if (v___x_5_ == 0)
{
lean_dec(v_a_2_);
return v_b_3_;
}
else
{
lean_dec(v_b_3_);
return v_a_2_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__1(lean_object* v_inst_6_, lean_object* v_a_7_, lean_object* v_b_8_){
_start:
{
lean_object* v___x_9_; uint8_t v___x_10_; 
lean_inc(v_b_8_);
lean_inc(v_a_7_);
v___x_9_ = lean_apply_2(v_inst_6_, v_a_7_, v_b_8_);
v___x_10_ = lean_unbox(v___x_9_);
if (v___x_10_ == 0)
{
lean_dec(v_b_8_);
return v_a_7_;
}
else
{
lean_dec(v_a_7_);
return v_b_8_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2(lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_a_13_, lean_object* v_b_14_){
_start:
{
lean_object* v___x_15_; uint8_t v___x_16_; 
lean_inc(v_b_14_);
lean_inc(v_a_13_);
v___x_15_ = lean_apply_2(v_inst_11_, v_a_13_, v_b_14_);
v___x_16_ = lean_unbox(v___x_15_);
if (v___x_16_ == 0)
{
lean_object* v___x_17_; uint8_t v___x_18_; 
v___x_17_ = lean_apply_2(v_inst_12_, v_a_13_, v_b_14_);
v___x_18_ = lean_unbox(v___x_17_);
if (v___x_18_ == 0)
{
uint8_t v___x_19_; 
v___x_19_ = 2;
return v___x_19_;
}
else
{
uint8_t v___x_20_; 
v___x_20_ = 1;
return v___x_20_;
}
}
else
{
uint8_t v___x_21_; 
lean_dec(v_b_14_);
lean_dec(v_a_13_);
lean_dec_ref(v_inst_12_);
v___x_21_ = 0;
return v___x_21_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2___boxed(lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_a_24_, lean_object* v_b_25_){
_start:
{
uint8_t v_res_26_; lean_object* v_r_27_; 
v_res_26_ = lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2(v_inst_22_, v_inst_23_, v_a_24_, v_b_25_);
v_r_27_ = lean_box(v_res_26_);
return v_r_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___redArg(lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___f_32_; lean_object* v___f_33_; lean_object* v___f_34_; lean_object* v___x_35_; 
lean_inc_ref_n(v_inst_30_, 2);
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__0), 3, 1);
lean_closure_set(v___f_32_, 0, v_inst_30_);
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__1), 3, 1);
lean_closure_set(v___f_33_, 0, v_inst_30_);
lean_inc_ref(v_inst_29_);
lean_inc_ref(v_inst_31_);
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_34_, 0, v_inst_31_);
lean_closure_set(v___f_34_, 1, v_inst_29_);
v___x_35_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_35_, 0, v_inst_28_);
lean_ctor_set(v___x_35_, 1, v___f_32_);
lean_ctor_set(v___x_35_, 2, v___f_33_);
lean_ctor_set(v___x_35_, 3, v___f_34_);
lean_ctor_set(v___x_35_, 4, v_inst_30_);
lean_ctor_set(v___x_35_, 5, v_inst_29_);
lean_ctor_set(v___x_35_, 6, v_inst_31_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder(lean_object* v_00_u03b1_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___f_44_; lean_object* v___f_45_; lean_object* v___f_46_; lean_object* v___x_47_; 
lean_inc_ref_n(v_inst_41_, 2);
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__0), 3, 1);
lean_closure_set(v___f_44_, 0, v_inst_41_);
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__1), 3, 1);
lean_closure_set(v___f_45_, 0, v_inst_41_);
lean_inc_ref(v_inst_40_);
lean_inc_ref(v_inst_42_);
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_IsSuccArchimedean_linearOrder___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_46_, 0, v_inst_42_);
lean_closure_set(v___f_46_, 1, v_inst_40_);
v___x_47_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_47_, 0, v_inst_37_);
lean_ctor_set(v___x_47_, 1, v___f_44_);
lean_ctor_set(v___x_47_, 2, v___f_45_);
lean_ctor_set(v___x_47_, 3, v___f_46_);
lean_ctor_set(v___x_47_, 4, v_inst_41_);
lean_ctor_set(v___x_47_, 5, v_inst_40_);
lean_ctor_set(v___x_47_, 6, v_inst_42_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSuccArchimedean_linearOrder___boxed(lean_object* v_00_u03b1_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_IsSuccArchimedean_linearOrder(v_00_u03b1_48_, v_inst_49_, v_inst_50_, v_inst_51_, v_inst_52_, v_inst_53_, v_inst_54_, v_inst_55_);
lean_dec(v_inst_50_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__1___redArg(lean_object* v_this_57_, lean_object* v_a_58_, lean_object* v_b_59_){
_start:
{
lean_object* v_toMax_60_; lean_object* v___x_61_; 
v_toMax_60_ = lean_ctor_get(v_this_57_, 2);
lean_inc(v_toMax_60_);
lean_dec_ref(v_this_57_);
v___x_61_ = lean_apply_2(v_toMax_60_, v_a_58_, v_b_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__1(lean_object* v_00_u03b1_62_, lean_object* v_this_63_, lean_object* v_a_64_, lean_object* v_b_65_){
_start:
{
lean_object* v_toMax_66_; lean_object* v___x_67_; 
v_toMax_66_ = lean_ctor_get(v_this_63_, 2);
lean_inc(v_toMax_66_);
lean_dec_ref(v_this_63_);
v___x_67_ = lean_apply_2(v_toMax_66_, v_a_64_, v_b_65_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__3___redArg(lean_object* v_this_68_, lean_object* v_a_69_, lean_object* v_b_70_){
_start:
{
lean_object* v_toMin_71_; lean_object* v___x_72_; 
v_toMin_71_ = lean_ctor_get(v_this_68_, 1);
lean_inc(v_toMin_71_);
lean_dec_ref(v_this_68_);
v___x_72_ = lean_apply_2(v_toMin_71_, v_a_69_, v_b_70_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__3(lean_object* v_00_u03b1_73_, lean_object* v_this_74_, lean_object* v_a_75_, lean_object* v_b_76_){
_start:
{
lean_object* v_toMin_77_; lean_object* v___x_78_; 
v_toMin_77_ = lean_ctor_get(v_this_74_, 1);
lean_inc(v_toMin_77_);
lean_dec_ref(v_this_74_);
v___x_78_ = lean_apply_2(v_toMin_77_, v_a_75_, v_b_76_);
return v___x_78_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___aux__5___redArg(lean_object* v_this_79_, lean_object* v_a_80_, lean_object* v_b_81_){
_start:
{
lean_object* v_toOrd_82_; lean_object* v___x_83_; uint8_t v___x_84_; 
v_toOrd_82_ = lean_ctor_get(v_this_79_, 3);
lean_inc_ref(v_toOrd_82_);
lean_dec_ref(v_this_79_);
v___x_83_ = lean_apply_2(v_toOrd_82_, v_b_81_, v_a_80_);
v___x_84_ = lean_unbox(v___x_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__5___redArg___boxed(lean_object* v_this_85_, lean_object* v_a_86_, lean_object* v_b_87_){
_start:
{
uint8_t v_res_88_; lean_object* v_r_89_; 
v_res_88_ = lp_mathlib_IsPredArchimedean_linearOrder___aux__5___redArg(v_this_85_, v_a_86_, v_b_87_);
v_r_89_ = lean_box(v_res_88_);
return v_r_89_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___aux__5(lean_object* v_00_u03b1_90_, lean_object* v_this_91_, lean_object* v_a_92_, lean_object* v_b_93_){
_start:
{
lean_object* v_toOrd_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v_toOrd_94_ = lean_ctor_get(v_this_91_, 3);
lean_inc_ref(v_toOrd_94_);
lean_dec_ref(v_this_91_);
v___x_95_ = lean_apply_2(v_toOrd_94_, v_b_93_, v_a_92_);
v___x_96_ = lean_unbox(v___x_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___aux__5___boxed(lean_object* v_00_u03b1_97_, lean_object* v_this_98_, lean_object* v_a_99_, lean_object* v_b_100_){
_start:
{
uint8_t v_res_101_; lean_object* v_r_102_; 
v_res_101_ = lp_mathlib_IsPredArchimedean_linearOrder___aux__5(v_00_u03b1_97_, v_this_98_, v_a_99_, v_b_100_);
v_r_102_ = lean_box(v_res_101_);
return v_r_102_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0(lean_object* v_inst_103_, lean_object* v_a_104_, lean_object* v_b_105_){
_start:
{
lean_object* v___x_106_; uint8_t v___x_107_; 
v___x_106_ = lean_apply_2(v_inst_103_, v_a_104_, v_b_105_);
v___x_107_ = lean_unbox(v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed(lean_object* v_inst_108_, lean_object* v_a_109_, lean_object* v_b_110_){
_start:
{
uint8_t v_res_111_; lean_object* v_r_112_; 
v_res_111_ = lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0(v_inst_108_, v_a_109_, v_b_110_);
v_r_112_ = lean_box(v_res_111_);
return v_r_112_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1(lean_object* v_inst_113_, lean_object* v_a_114_, lean_object* v_b_115_){
_start:
{
lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_116_ = lean_apply_2(v_inst_113_, v_b_115_, v_a_114_);
v___x_117_ = lean_unbox(v___x_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1___boxed(lean_object* v_inst_118_, lean_object* v_a_119_, lean_object* v_b_120_){
_start:
{
uint8_t v_res_121_; lean_object* v_r_122_; 
v_res_121_ = lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1(v_inst_118_, v_a_119_, v_b_120_);
v_r_122_ = lean_box(v_res_121_);
return v_r_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__2(lean_object* v_inst_123_, lean_object* v_a_124_, lean_object* v_b_125_){
_start:
{
lean_object* v___x_126_; uint8_t v___x_127_; 
lean_inc(v_a_124_);
lean_inc(v_b_125_);
v___x_126_ = lean_apply_2(v_inst_123_, v_b_125_, v_a_124_);
v___x_127_ = lean_unbox(v___x_126_);
if (v___x_127_ == 0)
{
lean_dec(v_a_124_);
return v_b_125_;
}
else
{
lean_dec(v_b_125_);
return v_a_124_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3(lean_object* v_inst_128_, lean_object* v_a_129_, lean_object* v_b_130_){
_start:
{
lean_object* v___x_131_; uint8_t v___x_132_; 
lean_inc(v_a_129_);
lean_inc(v_b_130_);
v___x_131_ = lean_apply_2(v_inst_128_, v_b_130_, v_a_129_);
v___x_132_ = lean_unbox(v___x_131_);
if (v___x_132_ == 0)
{
lean_dec(v_b_130_);
return v_a_129_;
}
else
{
lean_dec(v_a_129_);
return v_b_130_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__4(lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_a_135_, lean_object* v_b_136_){
_start:
{
lean_object* v___x_137_; uint8_t v___x_138_; 
lean_inc(v_a_135_);
lean_inc(v_b_136_);
v___x_137_ = lean_apply_2(v_inst_133_, v_b_136_, v_a_135_);
v___x_138_ = lean_unbox(v___x_137_);
if (v___x_138_ == 0)
{
lean_object* v___x_139_; uint8_t v___x_140_; 
v___x_139_ = lean_apply_2(v_inst_134_, v_a_135_, v_b_136_);
v___x_140_ = lean_unbox(v___x_139_);
if (v___x_140_ == 0)
{
uint8_t v___x_141_; 
v___x_141_ = 2;
return v___x_141_;
}
else
{
uint8_t v___x_142_; 
v___x_142_ = 1;
return v___x_142_;
}
}
else
{
uint8_t v___x_143_; 
lean_dec(v_b_136_);
lean_dec(v_a_135_);
lean_dec_ref(v_inst_134_);
v___x_143_ = 0;
return v___x_143_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__4___boxed(lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_a_146_, lean_object* v_b_147_){
_start:
{
uint8_t v_res_148_; lean_object* v_r_149_; 
v_res_148_ = lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__4(v_inst_144_, v_inst_145_, v_a_146_, v_b_147_);
v_r_149_ = lean_box(v_res_148_);
return v_r_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___redArg(lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v___f_154_; lean_object* v___f_155_; lean_object* v___f_156_; lean_object* v___f_157_; lean_object* v___f_158_; lean_object* v___f_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
lean_inc_ref_n(v_inst_151_, 2);
v___f_154_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_154_, 0, v_inst_151_);
lean_inc_ref_n(v_inst_152_, 3);
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_155_, 0, v_inst_152_);
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__2), 3, 1);
lean_closure_set(v___f_156_, 0, v_inst_152_);
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3), 3, 1);
lean_closure_set(v___f_157_, 0, v_inst_152_);
lean_inc_ref_n(v_inst_153_, 2);
v___f_158_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__4___boxed), 4, 2);
lean_closure_set(v___f_158_, 0, v_inst_153_);
lean_closure_set(v___f_158_, 1, v_inst_151_);
v___f_159_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_159_, 0, v_inst_153_);
v___x_160_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_inst_150_);
v___x_161_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v___f_156_);
lean_ctor_set(v___x_161_, 2, v___f_157_);
lean_ctor_set(v___x_161_, 3, v___f_158_);
lean_ctor_set(v___x_161_, 4, v___f_155_);
lean_ctor_set(v___x_161_, 5, v___f_154_);
lean_ctor_set(v___x_161_, 6, v___f_159_);
lean_inc_ref_n(v___x_161_, 2);
v___x_162_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___aux__1), 4, 2);
lean_closure_set(v___x_162_, 0, lean_box(0));
lean_closure_set(v___x_162_, 1, v___x_161_);
v___x_163_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___aux__3), 4, 2);
lean_closure_set(v___x_163_, 0, lean_box(0));
lean_closure_set(v___x_163_, 1, v___x_161_);
v___x_164_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___aux__5___boxed), 4, 2);
lean_closure_set(v___x_164_, 0, lean_box(0));
lean_closure_set(v___x_164_, 1, v___x_161_);
v___x_165_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_165_, 0, v_inst_150_);
lean_ctor_set(v___x_165_, 1, v___x_162_);
lean_ctor_set(v___x_165_, 2, v___x_163_);
lean_ctor_set(v___x_165_, 3, v___x_164_);
lean_ctor_set(v___x_165_, 4, v_inst_152_);
lean_ctor_set(v___x_165_, 5, v_inst_151_);
lean_ctor_set(v___x_165_, 6, v_inst_153_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder(lean_object* v_00_u03b1_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___f_174_; lean_object* v___f_175_; lean_object* v___f_176_; lean_object* v___f_177_; lean_object* v___f_178_; lean_object* v___f_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
lean_inc_ref_n(v_inst_170_, 2);
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_174_, 0, v_inst_170_);
lean_inc_ref_n(v_inst_171_, 3);
v___f_175_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_175_, 0, v_inst_171_);
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__2), 3, 1);
lean_closure_set(v___f_176_, 0, v_inst_171_);
v___f_177_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__3), 3, 1);
lean_closure_set(v___f_177_, 0, v_inst_171_);
lean_inc_ref_n(v_inst_172_, 2);
v___f_178_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__4___boxed), 4, 2);
lean_closure_set(v___f_178_, 0, v_inst_172_);
lean_closure_set(v___f_178_, 1, v_inst_170_);
v___f_179_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_179_, 0, v_inst_172_);
v___x_180_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_inst_167_);
v___x_181_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_181_, 0, v___x_180_);
lean_ctor_set(v___x_181_, 1, v___f_176_);
lean_ctor_set(v___x_181_, 2, v___f_177_);
lean_ctor_set(v___x_181_, 3, v___f_178_);
lean_ctor_set(v___x_181_, 4, v___f_175_);
lean_ctor_set(v___x_181_, 5, v___f_174_);
lean_ctor_set(v___x_181_, 6, v___f_179_);
lean_inc_ref_n(v___x_181_, 2);
v___x_182_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___aux__1), 4, 2);
lean_closure_set(v___x_182_, 0, lean_box(0));
lean_closure_set(v___x_182_, 1, v___x_181_);
v___x_183_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___aux__3), 4, 2);
lean_closure_set(v___x_183_, 0, lean_box(0));
lean_closure_set(v___x_183_, 1, v___x_181_);
v___x_184_ = lean_alloc_closure((void*)(lp_mathlib_IsPredArchimedean_linearOrder___aux__5___boxed), 4, 2);
lean_closure_set(v___x_184_, 0, lean_box(0));
lean_closure_set(v___x_184_, 1, v___x_181_);
v___x_185_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_185_, 0, v_inst_167_);
lean_ctor_set(v___x_185_, 1, v___x_182_);
lean_ctor_set(v___x_185_, 2, v___x_183_);
lean_ctor_set(v___x_185_, 3, v___x_184_);
lean_ctor_set(v___x_185_, 4, v_inst_171_);
lean_ctor_set(v___x_185_, 5, v_inst_170_);
lean_ctor_set(v___x_185_, 6, v_inst_172_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsPredArchimedean_linearOrder___boxed(lean_object* v_00_u03b1_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_inst_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_IsPredArchimedean_linearOrder(v_00_u03b1_186_, v_inst_187_, v_inst_188_, v_inst_189_, v_inst_190_, v_inst_191_, v_inst_192_, v_inst_193_);
lean_dec(v_inst_188_);
return v_res_194_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
}
#ifdef __cplusplus
}
#endif
