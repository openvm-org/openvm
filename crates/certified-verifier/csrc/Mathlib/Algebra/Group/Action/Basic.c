// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Units public import Mathlib.Algebra.Group.Invertible.Basic public import Mathlib.Algebra.Group.Pi.Basic public import Mathlib.Logic.Embedding.Basic
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAction___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowMulDistribMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowMulDistribMulAction___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowMulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_arrowMulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toFun___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulDistribMulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulDistribMulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidEnd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidEnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_a_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___redArg___lam__1(lean_object* v_toInv_5_, lean_object* v_a_6_, lean_object* v_inst_7_, lean_object* v_x_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_apply_1(v_toInv_5_, v_a_6_);
v___x_10_ = lean_apply_2(v_inst_7_, v___x_9_, v_x_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___redArg(lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_a_13_){
_start:
{
lean_object* v___x_14_; lean_object* v_toInv_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v___x_14_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_11_);
v_toInv_15_ = lean_ctor_get(v___x_14_, 1);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_14_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v___x_14_, 0);
lean_dec(v_unused_25_);
v___x_17_ = v___x_14_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_toInv_15_);
lean_dec(v___x_14_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___f_19_; lean_object* v___f_20_; lean_object* v___x_22_; 
lean_inc(v_a_13_);
lean_inc(v_inst_12_);
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toPerm___redArg___lam__0), 3, 2);
lean_closure_set(v___f_19_, 0, v_inst_12_);
lean_closure_set(v___f_19_, 1, v_a_13_);
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toPerm___redArg___lam__1), 4, 3);
lean_closure_set(v___f_20_, 0, v_toInv_15_);
lean_closure_set(v___f_20_, 1, v_a_13_);
lean_closure_set(v___f_20_, 2, v_inst_12_);
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 1, v___f_20_);
lean_ctor_set(v___x_17_, 0, v___f_19_);
v___x_22_ = v___x_17_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v___f_19_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v___f_20_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___redArg___boxed(lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_a_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_MulAction_toPerm___redArg(v_inst_26_, v_inst_27_, v_a_28_);
lean_dec_ref(v_inst_26_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm(lean_object* v_00_u03b1_30_, lean_object* v_00_u03b2_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_a_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_MulAction_toPerm___redArg(v_inst_32_, v_inst_33_, v_a_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toPerm___boxed(lean_object* v_00_u03b1_36_, lean_object* v_00_u03b2_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_a_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_MulAction_toPerm(v_00_u03b1_36_, v_00_u03b2_37_, v_inst_38_, v_inst_39_, v_a_40_);
lean_dec_ref(v_inst_38_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm___redArg___lam__1(lean_object* v_toNeg_42_, lean_object* v_a_43_, lean_object* v_inst_44_, lean_object* v_x_45_){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lean_apply_1(v_toNeg_42_, v_a_43_);
v___x_47_ = lean_apply_2(v_inst_44_, v___x_46_, v_x_45_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm___redArg(lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_a_50_){
_start:
{
lean_object* v___x_51_; lean_object* v_toNeg_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_61_; 
v___x_51_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_48_);
v_toNeg_52_ = lean_ctor_get(v___x_51_, 1);
v_isSharedCheck_61_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_61_ == 0)
{
lean_object* v_unused_62_; 
v_unused_62_ = lean_ctor_get(v___x_51_, 0);
lean_dec(v_unused_62_);
v___x_54_ = v___x_51_;
v_isShared_55_ = v_isSharedCheck_61_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_toNeg_52_);
lean_dec(v___x_51_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_61_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___f_56_; lean_object* v___f_57_; lean_object* v___x_59_; 
lean_inc(v_a_50_);
lean_inc(v_inst_49_);
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toPerm___redArg___lam__0), 3, 2);
lean_closure_set(v___f_56_, 0, v_inst_49_);
lean_closure_set(v___f_56_, 1, v_a_50_);
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_AddAction_toPerm___redArg___lam__1), 4, 3);
lean_closure_set(v___f_57_, 0, v_toNeg_52_);
lean_closure_set(v___f_57_, 1, v_a_50_);
lean_closure_set(v___f_57_, 2, v_inst_49_);
if (v_isShared_55_ == 0)
{
lean_ctor_set(v___x_54_, 1, v___f_57_);
lean_ctor_set(v___x_54_, 0, v___f_56_);
v___x_59_ = v___x_54_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v___f_56_);
lean_ctor_set(v_reuseFailAlloc_60_, 1, v___f_57_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
return v___x_59_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm___redArg___boxed(lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_a_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_AddAction_toPerm___redArg(v_inst_63_, v_inst_64_, v_a_65_);
lean_dec_ref(v_inst_63_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm(lean_object* v_00_u03b1_67_, lean_object* v_00_u03b2_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_a_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_AddAction_toPerm___redArg(v_inst_69_, v_inst_70_, v_a_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toPerm___boxed(lean_object* v_00_u03b1_73_, lean_object* v_00_u03b2_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_AddAction_toPerm(v_00_u03b1_73_, v_00_u03b2_74_, v_inst_75_, v_inst_76_, v_a_77_);
lean_dec_ref(v_inst_75_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAction___redArg___lam__0(lean_object* v_toInv_79_, lean_object* v_inst_80_, lean_object* v_g_81_, lean_object* v_F_82_, lean_object* v_a_83_){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_84_ = lean_apply_1(v_toInv_79_, v_g_81_);
v___x_85_ = lean_apply_2(v_inst_80_, v___x_84_, v_a_83_);
v___x_86_ = lean_apply_1(v_F_82_, v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAction___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; lean_object* v_toInv_90_; lean_object* v___f_91_; 
v___x_89_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_87_);
v_toInv_90_ = lean_ctor_get(v___x_89_, 1);
lean_inc(v_toInv_90_);
lean_dec_ref(v___x_89_);
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_arrowAction___redArg___lam__0), 5, 2);
lean_closure_set(v___f_91_, 0, v_toInv_90_);
lean_closure_set(v___f_91_, 1, v_inst_88_);
return v___f_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAction___redArg___boxed(lean_object* v_inst_92_, lean_object* v_inst_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_arrowAction___redArg(v_inst_92_, v_inst_93_);
lean_dec_ref(v_inst_92_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAction(lean_object* v_G_95_, lean_object* v_A_96_, lean_object* v_B_97_, lean_object* v_inst_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lp_mathlib_arrowAction___redArg(v_inst_98_, v_inst_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAction___boxed(lean_object* v_G_101_, lean_object* v_A_102_, lean_object* v_B_103_, lean_object* v_inst_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_arrowAction(v_G_101_, v_A_102_, v_B_103_, v_inst_104_, v_inst_105_);
lean_dec_ref(v_inst_104_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction___redArg___lam__0(lean_object* v_toNeg_107_, lean_object* v_inst_108_, lean_object* v_g_109_, lean_object* v_F_110_, lean_object* v_a_111_){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_112_ = lean_apply_1(v_toNeg_107_, v_g_109_);
v___x_113_ = lean_apply_2(v_inst_108_, v___x_112_, v_a_111_);
v___x_114_ = lean_apply_1(v_F_110_, v___x_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction___redArg(lean_object* v_inst_115_, lean_object* v_inst_116_){
_start:
{
lean_object* v___x_117_; lean_object* v_toNeg_118_; lean_object* v___f_119_; 
v___x_117_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_115_);
v_toNeg_118_ = lean_ctor_get(v___x_117_, 1);
lean_inc(v_toNeg_118_);
lean_dec_ref(v___x_117_);
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_arrowAddAction___redArg___lam__0), 5, 2);
lean_closure_set(v___f_119_, 0, v_toNeg_118_);
lean_closure_set(v___f_119_, 1, v_inst_116_);
return v___f_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction___redArg___boxed(lean_object* v_inst_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_arrowAddAction___redArg(v_inst_120_, v_inst_121_);
lean_dec_ref(v_inst_120_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction(lean_object* v_G_123_, lean_object* v_A_124_, lean_object* v_B_125_, lean_object* v_inst_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_arrowAddAction___redArg(v_inst_126_, v_inst_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowAddAction___boxed(lean_object* v_G_129_, lean_object* v_A_130_, lean_object* v_B_131_, lean_object* v_inst_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_arrowAddAction(v_G_129_, v_A_130_, v_B_131_, v_inst_132_, v_inst_133_);
lean_dec_ref(v_inst_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowMulDistribMulAction___redArg(lean_object* v_inst_135_, lean_object* v_inst_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_mathlib_arrowAction___redArg(v_inst_135_, v_inst_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowMulDistribMulAction___redArg___boxed(lean_object* v_inst_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_arrowMulDistribMulAction___redArg(v_inst_138_, v_inst_139_);
lean_dec_ref(v_inst_138_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowMulDistribMulAction(lean_object* v_M_141_, lean_object* v_G_142_, lean_object* v_A_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_mathlib_arrowAction___redArg(v_inst_144_, v_inst_145_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_arrowMulDistribMulAction___boxed(lean_object* v_M_148_, lean_object* v_G_149_, lean_object* v_A_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_arrowMulDistribMulAction(v_M_148_, v_G_149_, v_A_150_, v_inst_151_, v_inst_152_, v_inst_153_);
lean_dec_ref(v_inst_153_);
lean_dec_ref(v_inst_151_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toFun___redArg___lam__0(lean_object* v_inst_155_, lean_object* v_y_156_, lean_object* v_x_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lean_apply_2(v_inst_155_, v_x_157_, v_y_156_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toFun___redArg(lean_object* v_inst_159_){
_start:
{
lean_object* v___f_160_; 
v___f_160_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toFun___redArg___lam__0), 3, 1);
lean_closure_set(v___f_160_, 0, v_inst_159_);
return v___f_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toFun(lean_object* v_M_161_, lean_object* v_00_u03b1_162_, lean_object* v_inst_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___f_165_; 
v___f_165_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toFun___redArg___lam__0), 3, 1);
lean_closure_set(v___f_165_, 0, v_inst_164_);
return v___f_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_toFun___boxed(lean_object* v_M_166_, lean_object* v_00_u03b1_167_, lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_MulAction_toFun(v_M_166_, v_00_u03b1_167_, v_inst_168_, v_inst_169_);
lean_dec_ref(v_inst_168_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toFun___redArg(lean_object* v_inst_171_){
_start:
{
lean_object* v___f_172_; 
v___f_172_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toFun___redArg___lam__0), 3, 1);
lean_closure_set(v___f_172_, 0, v_inst_171_);
return v___f_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toFun(lean_object* v_M_173_, lean_object* v_00_u03b1_174_, lean_object* v_inst_175_, lean_object* v_inst_176_){
_start:
{
lean_object* v___f_177_; 
v___f_177_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_toFun___redArg___lam__0), 3, 1);
lean_closure_set(v___f_177_, 0, v_inst_176_);
return v___f_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_toFun___boxed(lean_object* v_M_178_, lean_object* v_00_u03b1_179_, lean_object* v_inst_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_AddAction_toFun(v_M_178_, v_00_u03b1_179_, v_inst_180_, v_inst_181_);
lean_dec_ref(v_inst_180_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulDistribMulAction___redArg(lean_object* v_inst_183_){
_start:
{
lean_inc(v_inst_183_);
return v_inst_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulDistribMulAction___redArg___boxed(lean_object* v_inst_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_Function_Injective_mulDistribMulAction___redArg(v_inst_184_);
lean_dec(v_inst_184_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulDistribMulAction(lean_object* v_M_186_, lean_object* v_A_187_, lean_object* v_B_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_f_194_, lean_object* v_hf_195_, lean_object* v_smul_196_){
_start:
{
lean_inc(v_inst_193_);
return v_inst_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulDistribMulAction___boxed(lean_object* v_M_197_, lean_object* v_A_198_, lean_object* v_B_199_, lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_f_205_, lean_object* v_hf_206_, lean_object* v_smul_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_Function_Injective_mulDistribMulAction(v_M_197_, v_A_198_, v_B_199_, v_inst_200_, v_inst_201_, v_inst_202_, v_inst_203_, v_inst_204_, v_f_205_, v_hf_206_, v_smul_207_);
lean_dec(v_f_205_);
lean_dec(v_inst_204_);
lean_dec_ref(v_inst_203_);
lean_dec(v_inst_202_);
lean_dec_ref(v_inst_201_);
lean_dec_ref(v_inst_200_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulDistribMulAction___redArg(lean_object* v_inst_209_){
_start:
{
lean_inc(v_inst_209_);
return v_inst_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulDistribMulAction___redArg___boxed(lean_object* v_inst_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_Function_Surjective_mulDistribMulAction___redArg(v_inst_210_);
lean_dec(v_inst_210_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulDistribMulAction(lean_object* v_M_212_, lean_object* v_A_213_, lean_object* v_B_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_f_220_, lean_object* v_hf_221_, lean_object* v_smul_222_){
_start:
{
lean_inc(v_inst_219_);
return v_inst_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulDistribMulAction___boxed(lean_object* v_M_223_, lean_object* v_A_224_, lean_object* v_B_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_f_231_, lean_object* v_hf_232_, lean_object* v_smul_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_Function_Surjective_mulDistribMulAction(v_M_223_, v_A_224_, v_B_225_, v_inst_226_, v_inst_227_, v_inst_228_, v_inst_229_, v_inst_230_, v_f_231_, v_hf_232_, v_smul_233_);
lean_dec(v_f_231_);
lean_dec(v_inst_230_);
lean_dec_ref(v_inst_229_);
lean_dec(v_inst_228_);
lean_dec_ref(v_inst_227_);
lean_dec_ref(v_inst_226_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0(lean_object* v_inst_235_, lean_object* v_r_236_, lean_object* v_x_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lean_apply_2(v_inst_235_, v_r_236_, v_x_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___redArg(lean_object* v_inst_239_, lean_object* v_r_240_){
_start:
{
lean_object* v___f_241_; 
v___f_241_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_241_, 0, v_inst_239_);
lean_closure_set(v___f_241_, 1, v_r_240_);
return v___f_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom(lean_object* v_M_242_, lean_object* v_A_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_r_247_){
_start:
{
lean_object* v___f_248_; 
v___f_248_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_248_, 0, v_inst_246_);
lean_closure_set(v___f_248_, 1, v_r_247_);
return v___f_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___boxed(lean_object* v_M_249_, lean_object* v_A_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_r_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_MulDistribMulAction_toMonoidHom(v_M_249_, v_A_250_, v_inst_251_, v_inst_252_, v_inst_253_, v_r_254_);
lean_dec_ref(v_inst_252_);
lean_dec_ref(v_inst_251_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidEnd___redArg(lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMonoidHom___boxed), 6, 5);
lean_closure_set(v___x_259_, 0, lean_box(0));
lean_closure_set(v___x_259_, 1, lean_box(0));
lean_closure_set(v___x_259_, 2, v_inst_256_);
lean_closure_set(v___x_259_, 3, v_inst_257_);
lean_closure_set(v___x_259_, 4, v_inst_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulDistribMulAction_toMonoidEnd(lean_object* v_M_260_, lean_object* v_A_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMonoidHom___boxed), 6, 5);
lean_closure_set(v___x_265_, 0, lean_box(0));
lean_closure_set(v___x_265_, 1, lean_box(0));
lean_closure_set(v___x_265_, 2, v_inst_262_);
lean_closure_set(v___x_265_, 3, v_inst_263_);
lean_closure_set(v___x_265_, 4, v_inst_264_);
return v___x_265_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
