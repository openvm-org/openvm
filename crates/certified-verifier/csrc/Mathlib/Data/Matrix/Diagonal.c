// Lean compiler output
// Module: Mathlib.Data.Matrix.Diagonal
// Imports: public import Init public meta import Init public import Mathlib.Data.Int.Cast.Basic public import Mathlib.Data.Int.Cast.Pi public import Mathlib.Data.Nat.Cast.Basic public import Mathlib.LinearAlgebra.Matrix.Defs public import Mathlib.Logic.Embedding.Basic
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
lean_object* lp_mathlib_Matrix_addGroup___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Matrix_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Matrix_diagonal___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_diagonal___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddMonoidWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diag___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diag(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_d_3_, lean_object* v_i_4_, lean_object* v_j_5_){
_start:
{
lean_object* v___x_6_; uint8_t v___x_7_; 
lean_inc(v_i_4_);
v___x_6_ = lean_apply_2(v_inst_1_, v_i_4_, v_j_5_);
v___x_7_ = lean_unbox(v___x_6_);
if (v___x_7_ == 0)
{
lean_dec(v_i_4_);
lean_dec(v_d_3_);
lean_inc(v_inst_2_);
return v_inst_2_;
}
else
{
lean_object* v___x_8_; 
v___x_8_ = lean_apply_1(v_d_3_, v_i_4_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg___lam__0___boxed(lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_d_11_, lean_object* v_i_12_, lean_object* v_j_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Matrix_diagonal___redArg___lam__0(v_inst_9_, v_inst_10_, v_d_11_, v_i_12_, v_j_13_);
lean_dec(v_inst_10_);
return v_res_14_;
}
}
static lean_object* _init_lp_mathlib_Matrix_diagonal___redArg___closed__0(void){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal___redArg(lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_d_18_, lean_object* v_a_19_, lean_object* v_a_20_){
_start:
{
lean_object* v___x_21_; lean_object* v_toFun_22_; lean_object* v___f_23_; lean_object* v___x_24_; 
v___x_21_ = lean_obj_once(&lp_mathlib_Matrix_diagonal___redArg___closed__0, &lp_mathlib_Matrix_diagonal___redArg___closed__0_once, _init_lp_mathlib_Matrix_diagonal___redArg___closed__0);
v_toFun_22_ = lean_ctor_get(v___x_21_, 0);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_23_, 0, v_inst_16_);
lean_closure_set(v___f_23_, 1, v_inst_17_);
lean_closure_set(v___f_23_, 2, v_d_18_);
lean_inc(v_toFun_22_);
v___x_24_ = lean_apply_3(v_toFun_22_, v___f_23_, v_a_19_, v_a_20_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonal(lean_object* v_n_25_, lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_d_29_, lean_object* v_a_30_, lean_object* v_a_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_Matrix_diagonal___redArg(v_inst_27_, v_inst_28_, v_d_29_, v_a_30_, v_a_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0(lean_object* v_inst_33_, lean_object* v_m_34_, lean_object* v_x_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lean_apply_1(v_inst_33_, v_m_34_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0___boxed(lean_object* v_inst_37_, lean_object* v_m_38_, lean_object* v_x_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0(v_inst_37_, v_m_38_, v_x_39_);
lean_dec(v_x_39_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1(lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_m_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v___f_47_; lean_object* v___x_48_; 
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_47_, 0, v_inst_41_);
lean_closure_set(v___f_47_, 1, v_m_44_);
v___x_48_ = lp_mathlib_Matrix_diagonal___redArg(v_inst_42_, v_inst_43_, v___f_47_, v___y_45_, v___y_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero___redArg(lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v___f_52_; 
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(v___f_52_, 0, v_inst_51_);
lean_closure_set(v___f_52_, 1, v_inst_49_);
lean_closure_set(v___f_52_, 2, v_inst_50_);
return v___f_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instNatCastOfZero(lean_object* v_n_53_, lean_object* v_00_u03b1_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___f_58_; 
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(v___f_58_, 0, v_inst_57_);
lean_closure_set(v___f_58_, 1, v_inst_55_);
lean_closure_set(v___f_58_, 2, v_inst_56_);
return v___f_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0(lean_object* v_inst_59_, lean_object* v_m_60_, lean_object* v_x_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lean_apply_1(v_inst_59_, v_m_60_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0___boxed(lean_object* v_inst_63_, lean_object* v_m_64_, lean_object* v_x_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0(v_inst_63_, v_m_64_, v_x_65_);
lean_dec(v_x_65_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1(lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_m_70_, lean_object* v___y_71_, lean_object* v___y_72_){
_start:
{
lean_object* v___f_73_; lean_object* v___x_74_; 
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_73_, 0, v_inst_67_);
lean_closure_set(v___f_73_, 1, v_m_70_);
v___x_74_ = lp_mathlib_Matrix_diagonal___redArg(v_inst_68_, v_inst_69_, v___f_73_, v___y_71_, v___y_72_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero___redArg(lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(v___f_78_, 0, v_inst_77_);
lean_closure_set(v___f_78_, 1, v_inst_75_);
lean_closure_set(v___f_78_, 2, v_inst_76_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instIntCastOfZero(lean_object* v_n_79_, lean_object* v_00_u03b1_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_){
_start:
{
lean_object* v___f_84_; 
v___f_84_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(v___f_84_, 0, v_inst_83_);
lean_closure_set(v___f_84_, 1, v_inst_81_);
lean_closure_set(v___f_84_, 2, v_inst_82_);
return v___f_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg___lam__0(lean_object* v_inst_85_, lean_object* v_x_86_){
_start:
{
lean_inc(v_inst_85_);
return v_inst_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg___lam__0___boxed(lean_object* v_inst_87_, lean_object* v_x_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Matrix_one___redArg___lam__0(v_inst_87_, v_x_88_);
lean_dec(v_x_88_);
lean_dec(v_inst_87_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one___redArg(lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v___f_93_; lean_object* v___x_94_; 
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_one___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_93_, 0, v_inst_92_);
v___x_94_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal), 7, 5);
lean_closure_set(v___x_94_, 0, lean_box(0));
lean_closure_set(v___x_94_, 1, lean_box(0));
lean_closure_set(v___x_94_, 2, v_inst_90_);
lean_closure_set(v___x_94_, 3, v_inst_91_);
lean_closure_set(v___x_94_, 4, v___f_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_one(lean_object* v_n_95_, lean_object* v_00_u03b1_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lp_mathlib_Matrix_one___redArg(v_inst_97_, v_inst_98_, v_inst_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddMonoidWithOne___redArg(lean_object* v_inst_101_, lean_object* v_inst_102_){
_start:
{
lean_object* v_toNatCast_103_; lean_object* v_toAddMonoid_104_; lean_object* v_toOne_105_; lean_object* v___x_107_; uint8_t v_isShared_108_; uint8_t v_isSharedCheck_118_; 
v_toNatCast_103_ = lean_ctor_get(v_inst_102_, 0);
v_toAddMonoid_104_ = lean_ctor_get(v_inst_102_, 1);
v_toOne_105_ = lean_ctor_get(v_inst_102_, 2);
v_isSharedCheck_118_ = !lean_is_exclusive(v_inst_102_);
if (v_isSharedCheck_118_ == 0)
{
v___x_107_ = v_inst_102_;
v_isShared_108_ = v_isSharedCheck_118_;
goto v_resetjp_106_;
}
else
{
lean_inc(v_toOne_105_);
lean_inc(v_toAddMonoid_104_);
lean_inc(v_toNatCast_103_);
lean_dec(v_inst_102_);
v___x_107_ = lean_box(0);
v_isShared_108_ = v_isSharedCheck_118_;
goto v_resetjp_106_;
}
v_resetjp_106_:
{
lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v_toZero_111_; lean_object* v___f_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_116_; 
v___x_109_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_104_);
v___x_110_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_109_);
v_toZero_111_ = lean_ctor_get(v___x_110_, 0);
lean_inc_n(v_toZero_111_, 2);
lean_dec_ref(v___x_110_);
lean_inc_ref(v_inst_101_);
v___f_112_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instNatCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(v___f_112_, 0, v_toNatCast_103_);
lean_closure_set(v___f_112_, 1, v_inst_101_);
lean_closure_set(v___f_112_, 2, v_toZero_111_);
v___x_113_ = lp_mathlib_Matrix_addMonoid___redArg(v_toAddMonoid_104_);
v___x_114_ = lp_mathlib_Matrix_one___redArg(v_inst_101_, v_toZero_111_, v_toOne_105_);
if (v_isShared_108_ == 0)
{
lean_ctor_set(v___x_107_, 2, v___x_114_);
lean_ctor_set(v___x_107_, 1, v___x_113_);
lean_ctor_set(v___x_107_, 0, v___f_112_);
v___x_116_ = v___x_107_;
goto v_reusejp_115_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v___f_112_);
lean_ctor_set(v_reuseFailAlloc_117_, 1, v___x_113_);
lean_ctor_set(v_reuseFailAlloc_117_, 2, v___x_114_);
v___x_116_ = v_reuseFailAlloc_117_;
goto v_reusejp_115_;
}
v_reusejp_115_:
{
return v___x_116_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddMonoidWithOne(lean_object* v_n_119_, lean_object* v_00_u03b1_120_, lean_object* v_inst_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_Matrix_instAddMonoidWithOne___redArg(v_inst_121_, v_inst_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne___redArg(lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v_toIntCast_128_; lean_object* v_toAddMonoidWithOne_129_; lean_object* v___x_131_; uint8_t v_isShared_132_; uint8_t v_isSharedCheck_154_; 
v___x_126_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v_inst_125_);
lean_inc_ref(v___x_126_);
v___x_127_ = lp_mathlib_Matrix_addGroup___redArg(v___x_126_);
v_toIntCast_128_ = lean_ctor_get(v_inst_125_, 0);
v_toAddMonoidWithOne_129_ = lean_ctor_get(v_inst_125_, 1);
v_isSharedCheck_154_ = !lean_is_exclusive(v_inst_125_);
if (v_isSharedCheck_154_ == 0)
{
lean_object* v_unused_155_; lean_object* v_unused_156_; lean_object* v_unused_157_; 
v_unused_155_ = lean_ctor_get(v_inst_125_, 4);
lean_dec(v_unused_155_);
v_unused_156_ = lean_ctor_get(v_inst_125_, 3);
lean_dec(v_unused_156_);
v_unused_157_ = lean_ctor_get(v_inst_125_, 2);
lean_dec(v_unused_157_);
v___x_131_ = v_inst_125_;
v_isShared_132_ = v_isSharedCheck_154_;
goto v_resetjp_130_;
}
else
{
lean_inc(v_toAddMonoidWithOne_129_);
lean_inc(v_toIntCast_128_);
lean_dec(v_inst_125_);
v___x_131_ = lean_box(0);
v_isShared_132_ = v_isSharedCheck_154_;
goto v_resetjp_130_;
}
v_resetjp_130_:
{
lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v_toZero_135_; lean_object* v_toNatCast_136_; lean_object* v_toOne_137_; lean_object* v___x_139_; uint8_t v_isShared_140_; uint8_t v_isSharedCheck_152_; 
lean_inc_ref(v_inst_124_);
v___x_133_ = lp_mathlib_Matrix_instAddMonoidWithOne___redArg(v_inst_124_, v_toAddMonoidWithOne_129_);
v___x_134_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_126_);
lean_dec_ref(v___x_126_);
v_toZero_135_ = lean_ctor_get(v___x_134_, 0);
lean_inc(v_toZero_135_);
lean_dec_ref(v___x_134_);
v_toNatCast_136_ = lean_ctor_get(v___x_133_, 0);
v_toOne_137_ = lean_ctor_get(v___x_133_, 2);
v_isSharedCheck_152_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_152_ == 0)
{
lean_object* v_unused_153_; 
v_unused_153_ = lean_ctor_get(v___x_133_, 1);
lean_dec(v_unused_153_);
v___x_139_ = v___x_133_;
v_isShared_140_ = v_isSharedCheck_152_;
goto v_resetjp_138_;
}
else
{
lean_inc(v_toOne_137_);
lean_inc(v_toNatCast_136_);
lean_dec(v___x_133_);
v___x_139_ = lean_box(0);
v_isShared_140_ = v_isSharedCheck_152_;
goto v_resetjp_138_;
}
v_resetjp_138_:
{
lean_object* v_toAddMonoid_141_; lean_object* v_toNeg_142_; lean_object* v_toSub_143_; lean_object* v_toZSMul_144_; lean_object* v___f_145_; lean_object* v___x_147_; 
v_toAddMonoid_141_ = lean_ctor_get(v___x_127_, 0);
lean_inc_ref(v_toAddMonoid_141_);
v_toNeg_142_ = lean_ctor_get(v___x_127_, 1);
lean_inc(v_toNeg_142_);
v_toSub_143_ = lean_ctor_get(v___x_127_, 2);
lean_inc(v_toSub_143_);
v_toZSMul_144_ = lean_ctor_get(v___x_127_, 3);
lean_inc(v_toZSMul_144_);
lean_dec_ref(v___x_127_);
v___f_145_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instIntCastOfZero___redArg___lam__1), 6, 3);
lean_closure_set(v___f_145_, 0, v_toIntCast_128_);
lean_closure_set(v___f_145_, 1, v_inst_124_);
lean_closure_set(v___f_145_, 2, v_toZero_135_);
if (v_isShared_140_ == 0)
{
lean_ctor_set(v___x_139_, 1, v_toAddMonoid_141_);
v___x_147_ = v___x_139_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v_toNatCast_136_);
lean_ctor_set(v_reuseFailAlloc_151_, 1, v_toAddMonoid_141_);
lean_ctor_set(v_reuseFailAlloc_151_, 2, v_toOne_137_);
v___x_147_ = v_reuseFailAlloc_151_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
lean_object* v___x_149_; 
if (v_isShared_132_ == 0)
{
lean_ctor_set(v___x_131_, 4, v_toZSMul_144_);
lean_ctor_set(v___x_131_, 3, v_toSub_143_);
lean_ctor_set(v___x_131_, 2, v_toNeg_142_);
lean_ctor_set(v___x_131_, 1, v___x_147_);
lean_ctor_set(v___x_131_, 0, v___f_145_);
v___x_149_ = v___x_131_;
goto v_reusejp_148_;
}
else
{
lean_object* v_reuseFailAlloc_150_; 
v_reuseFailAlloc_150_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_150_, 0, v___f_145_);
lean_ctor_set(v_reuseFailAlloc_150_, 1, v___x_147_);
lean_ctor_set(v_reuseFailAlloc_150_, 2, v_toNeg_142_);
lean_ctor_set(v_reuseFailAlloc_150_, 3, v_toSub_143_);
lean_ctor_set(v_reuseFailAlloc_150_, 4, v_toZSMul_144_);
v___x_149_ = v_reuseFailAlloc_150_;
goto v_reusejp_148_;
}
v_reusejp_148_:
{
return v___x_149_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddGroupWithOne(lean_object* v_n_158_, lean_object* v_00_u03b1_159_, lean_object* v_inst_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_Matrix_instAddGroupWithOne___redArg(v_inst_160_, v_inst_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne___redArg(lean_object* v_inst_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v_toAddMonoid_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v_toNatCast_168_; lean_object* v_toOne_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_176_; 
v_toAddMonoid_165_ = lean_ctor_get(v_inst_164_, 1);
lean_inc_ref(v_toAddMonoid_165_);
v___x_166_ = lp_mathlib_Matrix_addMonoid___redArg(v_toAddMonoid_165_);
v___x_167_ = lp_mathlib_Matrix_instAddMonoidWithOne___redArg(v_inst_163_, v_inst_164_);
v_toNatCast_168_ = lean_ctor_get(v___x_167_, 0);
v_toOne_169_ = lean_ctor_get(v___x_167_, 2);
v_isSharedCheck_176_ = !lean_is_exclusive(v___x_167_);
if (v_isSharedCheck_176_ == 0)
{
lean_object* v_unused_177_; 
v_unused_177_ = lean_ctor_get(v___x_167_, 1);
lean_dec(v_unused_177_);
v___x_171_ = v___x_167_;
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_toOne_169_);
lean_inc(v_toNatCast_168_);
lean_dec(v___x_167_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_174_; 
if (v_isShared_172_ == 0)
{
lean_ctor_set(v___x_171_, 1, v___x_166_);
v___x_174_ = v___x_171_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_toNatCast_168_);
lean_ctor_set(v_reuseFailAlloc_175_, 1, v___x_166_);
lean_ctor_set(v_reuseFailAlloc_175_, 2, v_toOne_169_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommMonoidWithOne(lean_object* v_n_178_, lean_object* v_00_u03b1_179_, lean_object* v_inst_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_mathlib_Matrix_instAddCommMonoidWithOne___redArg(v_inst_180_, v_inst_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(lean_object* v_inst_183_, lean_object* v_inst_184_){
_start:
{
lean_object* v_toAddCommGroup_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_189_; uint8_t v_isShared_190_; uint8_t v_isSharedCheck_199_; 
v_toAddCommGroup_185_ = lean_ctor_get(v_inst_184_, 0);
lean_inc_ref(v_toAddCommGroup_185_);
v___x_186_ = lp_mathlib_Matrix_addGroup___redArg(v_toAddCommGroup_185_);
v___x_187_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v_inst_184_);
v_isSharedCheck_199_ = !lean_is_exclusive(v_inst_184_);
if (v_isSharedCheck_199_ == 0)
{
lean_object* v_unused_200_; lean_object* v_unused_201_; lean_object* v_unused_202_; lean_object* v_unused_203_; 
v_unused_200_ = lean_ctor_get(v_inst_184_, 3);
lean_dec(v_unused_200_);
v_unused_201_ = lean_ctor_get(v_inst_184_, 2);
lean_dec(v_unused_201_);
v_unused_202_ = lean_ctor_get(v_inst_184_, 1);
lean_dec(v_unused_202_);
v_unused_203_ = lean_ctor_get(v_inst_184_, 0);
lean_dec(v_unused_203_);
v___x_189_ = v_inst_184_;
v_isShared_190_ = v_isSharedCheck_199_;
goto v_resetjp_188_;
}
else
{
lean_dec(v_inst_184_);
v___x_189_ = lean_box(0);
v_isShared_190_ = v_isSharedCheck_199_;
goto v_resetjp_188_;
}
v_resetjp_188_:
{
lean_object* v___x_191_; lean_object* v_toAddMonoidWithOne_192_; lean_object* v_toIntCast_193_; lean_object* v_toNatCast_194_; lean_object* v_toOne_195_; lean_object* v___x_197_; 
v___x_191_ = lp_mathlib_Matrix_instAddGroupWithOne___redArg(v_inst_183_, v___x_187_);
v_toAddMonoidWithOne_192_ = lean_ctor_get(v___x_191_, 1);
lean_inc_ref(v_toAddMonoidWithOne_192_);
v_toIntCast_193_ = lean_ctor_get(v___x_191_, 0);
lean_inc(v_toIntCast_193_);
lean_dec_ref(v___x_191_);
v_toNatCast_194_ = lean_ctor_get(v_toAddMonoidWithOne_192_, 0);
lean_inc(v_toNatCast_194_);
v_toOne_195_ = lean_ctor_get(v_toAddMonoidWithOne_192_, 2);
lean_inc(v_toOne_195_);
lean_dec_ref(v_toAddMonoidWithOne_192_);
if (v_isShared_190_ == 0)
{
lean_ctor_set(v___x_189_, 3, v_toOne_195_);
lean_ctor_set(v___x_189_, 2, v_toNatCast_194_);
lean_ctor_set(v___x_189_, 1, v_toIntCast_193_);
lean_ctor_set(v___x_189_, 0, v___x_186_);
v___x_197_ = v___x_189_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v___x_186_);
lean_ctor_set(v_reuseFailAlloc_198_, 1, v_toIntCast_193_);
lean_ctor_set(v_reuseFailAlloc_198_, 2, v_toNatCast_194_);
lean_ctor_set(v_reuseFailAlloc_198_, 3, v_toOne_195_);
v___x_197_ = v_reuseFailAlloc_198_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
return v___x_197_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAddCommGroupWithOne(lean_object* v_n_204_, lean_object* v_00_u03b1_205_, lean_object* v_inst_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_Matrix_instAddCommGroupWithOne___redArg(v_inst_206_, v_inst_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diag___redArg(lean_object* v_A_209_, lean_object* v_i_210_){
_start:
{
lean_object* v___x_211_; 
lean_inc(v_i_210_);
v___x_211_ = lean_apply_2(v_A_209_, v_i_210_, v_i_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diag(lean_object* v_n_212_, lean_object* v_00_u03b1_213_, lean_object* v_A_214_, lean_object* v_i_215_){
_start:
{
lean_object* v___x_216_; 
lean_inc(v_i_215_);
v___x_216_ = lean_apply_2(v_A_214_, v_i_215_, v_i_215_);
return v___x_216_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Matrix_Diagonal(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Matrix_Diagonal(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Matrix_Diagonal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Matrix_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Matrix_Diagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Matrix_Diagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Matrix_Diagonal(builtin);
}
#ifdef __cplusplus
}
#endif
