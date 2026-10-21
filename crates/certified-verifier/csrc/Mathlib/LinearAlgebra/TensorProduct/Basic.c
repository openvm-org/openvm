// Lean compiler output
// Module: Mathlib.LinearAlgebra.TensorProduct.Basic
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.TensorProduct.Defs public import Mathlib.Algebra.Module.Equiv.Basic public import Mathlib.Tactic.Abel
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
lean_object* lp_mathlib_TensorProduct_addMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_TensorProduct_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_FreeAddMonoid_lift___redArg(lean_object*);
lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_compr_u2082_u209b_u2097___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_flip___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofLinearMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddCommGroup_toIntModule___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAddHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAddHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uncurry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift_equiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift_equiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lcurry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lcurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_curry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_curry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_comm___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_comm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_equivOfCompatibleSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_equivOfCompatibleSMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_equivOfCompatibleSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_equivOfCompatibleSMul___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_Neg_aux___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_Neg_aux___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_Neg_aux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_Neg_aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_neg___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_neg___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAddHom___redArg___lam__0(lean_object* v_f_1_, lean_object* v_mn_2_){
_start:
{
lean_object* v_fst_3_; lean_object* v_snd_4_; lean_object* v___x_5_; 
v_fst_3_ = lean_ctor_get(v_mn_2_, 0);
lean_inc(v_fst_3_);
v_snd_4_ = lean_ctor_get(v_mn_2_, 1);
lean_inc(v_snd_4_);
lean_dec_ref(v_mn_2_);
v___x_5_ = lean_apply_2(v_f_1_, v_fst_3_, v_snd_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAddHom___redArg(lean_object* v_inst_6_, lean_object* v_f_7_){
_start:
{
lean_object* v___x_8_; lean_object* v_toFun_9_; lean_object* v___f_10_; lean_object* v___x_11_; lean_object* v___f_12_; 
v___x_8_ = lp_mathlib_FreeAddMonoid_lift___redArg(v_inst_6_);
v_toFun_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_toFun_9_);
lean_dec_ref(v___x_8_);
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_liftAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_10_, 0, v_f_7_);
v___x_11_ = lean_apply_1(v_toFun_9_, v___f_10_);
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_12_, 0, v___x_11_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAddHom(lean_object* v_R_13_, lean_object* v_inst_14_, lean_object* v_M_15_, lean_object* v_N_16_, lean_object* v_P_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_f_23_, lean_object* v_hf_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_TensorProduct_liftAddHom___redArg(v_inst_20_, v_f_23_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAddHom___boxed(lean_object* v_R_26_, lean_object* v_inst_27_, lean_object* v_M_28_, lean_object* v_N_29_, lean_object* v_P_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_f_36_, lean_object* v_hf_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_TensorProduct_liftAddHom(v_R_26_, v_inst_27_, v_M_28_, v_N_29_, v_P_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_inst_35_, v_f_36_, v_hf_37_);
lean_dec(v_inst_35_);
lean_dec(v_inst_34_);
lean_dec_ref(v_inst_32_);
lean_dec_ref(v_inst_31_);
lean_dec_ref(v_inst_27_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAux___redArg(lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_00_u03c3_u2081_u2082_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_f_x27_46_){
_start:
{
lean_object* v___x_47_; lean_object* v___f_48_; lean_object* v___f_49_; lean_object* v___x_50_; 
lean_inc_ref(v_inst_43_);
v___x_47_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___boxed), 12, 11);
lean_closure_set(v___x_47_, 0, lean_box(0));
lean_closure_set(v___x_47_, 1, lean_box(0));
lean_closure_set(v___x_47_, 2, lean_box(0));
lean_closure_set(v___x_47_, 3, lean_box(0));
lean_closure_set(v___x_47_, 4, v_inst_39_);
lean_closure_set(v___x_47_, 5, v_inst_40_);
lean_closure_set(v___x_47_, 6, v_inst_42_);
lean_closure_set(v___x_47_, 7, v_inst_43_);
lean_closure_set(v___x_47_, 8, v_inst_44_);
lean_closure_set(v___x_47_, 9, v_inst_45_);
lean_closure_set(v___x_47_, 10, v_00_u03c3_u2081_u2082_41_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_48_, 0, v_f_x27_46_);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_49_, 0, v___f_48_);
lean_closure_set(v___f_49_, 1, v___x_47_);
v___x_50_ = lp_mathlib_TensorProduct_liftAddHom___redArg(v_inst_43_, v___f_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAux(lean_object* v_R_51_, lean_object* v_R_u2082_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_00_u03c3_u2081_u2082_55_, lean_object* v_M_56_, lean_object* v_N_57_, lean_object* v_P_u2082_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_f_x27_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_53_, v_inst_54_, v_00_u03c3_u2081_u2082_55_, v_inst_60_, v_inst_61_, v_inst_63_, v_inst_64_, v_f_x27_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_liftAux___boxed(lean_object* v_R_67_, lean_object* v_R_u2082_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_00_u03c3_u2081_u2082_71_, lean_object* v_M_72_, lean_object* v_N_73_, lean_object* v_P_u2082_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_f_x27_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_TensorProduct_liftAux(v_R_67_, v_R_u2082_68_, v_inst_69_, v_inst_70_, v_00_u03c3_u2081_u2082_71_, v_M_72_, v_N_73_, v_P_u2082_74_, v_inst_75_, v_inst_76_, v_inst_77_, v_inst_78_, v_inst_79_, v_inst_80_, v_f_x27_81_);
lean_dec(v_inst_78_);
lean_dec_ref(v_inst_75_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift___redArg(lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_00_u03c3_u2081_u2082_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_f_x27_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_83_, v_inst_84_, v_00_u03c3_u2081_u2082_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_inst_89_, v_f_x27_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift(lean_object* v_R_92_, lean_object* v_R_u2082_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_00_u03c3_u2081_u2082_96_, lean_object* v_M_97_, lean_object* v_N_98_, lean_object* v_P_u2082_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_f_x27_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_94_, v_inst_95_, v_00_u03c3_u2081_u2082_96_, v_inst_101_, v_inst_102_, v_inst_104_, v_inst_105_, v_f_x27_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift___boxed(lean_object* v_R_108_, lean_object* v_R_u2082_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_00_u03c3_u2081_u2082_112_, lean_object* v_M_113_, lean_object* v_N_114_, lean_object* v_P_u2082_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_f_x27_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_TensorProduct_lift(v_R_108_, v_R_u2082_109_, v_inst_110_, v_inst_111_, v_00_u03c3_u2081_u2082_112_, v_M_113_, v_N_114_, v_P_u2082_115_, v_inst_116_, v_inst_117_, v_inst_118_, v_inst_119_, v_inst_120_, v_inst_121_, v_f_x27_122_);
lean_dec(v_inst_119_);
lean_dec_ref(v_inst_116_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uncurry___redArg(lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_00_u03c3_u2081_u2082_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_lift___boxed), 15, 14);
lean_closure_set(v___x_133_, 0, lean_box(0));
lean_closure_set(v___x_133_, 1, lean_box(0));
lean_closure_set(v___x_133_, 2, v_inst_124_);
lean_closure_set(v___x_133_, 3, v_inst_125_);
lean_closure_set(v___x_133_, 4, v_00_u03c3_u2081_u2082_126_);
lean_closure_set(v___x_133_, 5, lean_box(0));
lean_closure_set(v___x_133_, 6, lean_box(0));
lean_closure_set(v___x_133_, 7, lean_box(0));
lean_closure_set(v___x_133_, 8, v_inst_127_);
lean_closure_set(v___x_133_, 9, v_inst_128_);
lean_closure_set(v___x_133_, 10, v_inst_129_);
lean_closure_set(v___x_133_, 11, v_inst_130_);
lean_closure_set(v___x_133_, 12, v_inst_131_);
lean_closure_set(v___x_133_, 13, v_inst_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_uncurry(lean_object* v_R_134_, lean_object* v_R_u2082_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_00_u03c3_u2081_u2082_138_, lean_object* v_M_139_, lean_object* v_N_140_, lean_object* v_P_u2082_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_lift___boxed), 15, 14);
lean_closure_set(v___x_148_, 0, lean_box(0));
lean_closure_set(v___x_148_, 1, lean_box(0));
lean_closure_set(v___x_148_, 2, v_inst_136_);
lean_closure_set(v___x_148_, 3, v_inst_137_);
lean_closure_set(v___x_148_, 4, v_00_u03c3_u2081_u2082_138_);
lean_closure_set(v___x_148_, 5, lean_box(0));
lean_closure_set(v___x_148_, 6, lean_box(0));
lean_closure_set(v___x_148_, 7, lean_box(0));
lean_closure_set(v___x_148_, 8, v_inst_142_);
lean_closure_set(v___x_148_, 9, v_inst_143_);
lean_closure_set(v___x_148_, 10, v_inst_144_);
lean_closure_set(v___x_148_, 11, v_inst_145_);
lean_closure_set(v___x_148_, 12, v_inst_146_);
lean_closure_set(v___x_148_, 13, v_inst_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0(lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v___x_156_, lean_object* v_inst_157_, lean_object* v___f_158_, lean_object* v_inst_159_, lean_object* v_00_u03c3_u2081_u2082_160_, lean_object* v_f_161_, lean_object* v___y_162_, lean_object* v___y_163_){
_start:
{
lean_object* v___f_164_; lean_object* v___x_165_; lean_object* v___x_70__overap_166_; lean_object* v___x_167_; 
v___f_164_ = ((lean_object*)(lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___closed__0));
v___x_165_ = lp_mathlib_TensorProduct_mk(lean_box(0), v_inst_150_, lean_box(0), lean_box(0), v_inst_151_, v_inst_152_, v_inst_153_, v_inst_154_);
lean_inc(v_00_u03c3_u2081_u2082_160_);
lean_inc_ref(v_inst_150_);
v___x_70__overap_166_ = lp_mathlib_LinearMap_compr_u2082_u209b_u2097___redArg(v_inst_150_, v_inst_150_, v_inst_155_, v_inst_152_, v___x_156_, v_inst_157_, v_inst_154_, v___f_158_, v_inst_159_, v___f_164_, v_00_u03c3_u2081_u2082_160_, v_00_u03c3_u2081_u2082_160_, v___x_165_, v_f_161_);
v___x_167_ = lean_apply_2(v___x_70__overap_166_, v___y_162_, v___y_163_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___boxed(lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v___x_174_, lean_object* v_inst_175_, lean_object* v___f_176_, lean_object* v_inst_177_, lean_object* v_00_u03c3_u2081_u2082_178_, lean_object* v_f_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0(v_inst_168_, v_inst_169_, v_inst_170_, v_inst_171_, v_inst_172_, v_inst_173_, v___x_174_, v_inst_175_, v___f_176_, v_inst_177_, v_00_u03c3_u2081_u2082_178_, v_f_179_, v___y_180_, v___y_181_);
lean_dec(v_inst_171_);
lean_dec_ref(v_inst_169_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift_equiv___redArg(lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_00_u03c3_u2081_u2082_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_){
_start:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___f_194_; lean_object* v___f_195_; lean_object* v___x_196_; 
lean_inc(v_inst_191_);
lean_inc_n(v_inst_190_, 3);
lean_inc_n(v_inst_189_, 4);
lean_inc_ref(v_inst_188_);
lean_inc_ref_n(v_inst_187_, 3);
lean_inc_ref_n(v_inst_186_, 3);
lean_inc(v_00_u03c3_u2081_u2082_185_);
lean_inc_ref(v_inst_184_);
lean_inc_ref_n(v_inst_183_, 3);
v___x_192_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_lift___boxed), 15, 14);
lean_closure_set(v___x_192_, 0, lean_box(0));
lean_closure_set(v___x_192_, 1, lean_box(0));
lean_closure_set(v___x_192_, 2, v_inst_183_);
lean_closure_set(v___x_192_, 3, v_inst_184_);
lean_closure_set(v___x_192_, 4, v_00_u03c3_u2081_u2082_185_);
lean_closure_set(v___x_192_, 5, lean_box(0));
lean_closure_set(v___x_192_, 6, lean_box(0));
lean_closure_set(v___x_192_, 7, lean_box(0));
lean_closure_set(v___x_192_, 8, v_inst_186_);
lean_closure_set(v___x_192_, 9, v_inst_187_);
lean_closure_set(v___x_192_, 10, v_inst_188_);
lean_closure_set(v___x_192_, 11, v_inst_189_);
lean_closure_set(v___x_192_, 12, v_inst_190_);
lean_closure_set(v___x_192_, 13, v_inst_191_);
v___x_193_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_183_, v_inst_186_, v_inst_187_, v_inst_189_, v_inst_190_);
v___f_194_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_194_, 0, v_inst_183_);
lean_closure_set(v___f_194_, 1, v_inst_186_);
lean_closure_set(v___f_194_, 2, v_inst_187_);
lean_closure_set(v___f_194_, 3, v_inst_189_);
lean_closure_set(v___f_194_, 4, v_inst_190_);
lean_closure_set(v___f_194_, 5, v_inst_189_);
v___f_195_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___boxed), 14, 11);
lean_closure_set(v___f_195_, 0, v_inst_183_);
lean_closure_set(v___f_195_, 1, v_inst_186_);
lean_closure_set(v___f_195_, 2, v_inst_187_);
lean_closure_set(v___f_195_, 3, v_inst_189_);
lean_closure_set(v___f_195_, 4, v_inst_190_);
lean_closure_set(v___f_195_, 5, v_inst_184_);
lean_closure_set(v___f_195_, 6, v___x_193_);
lean_closure_set(v___f_195_, 7, v_inst_188_);
lean_closure_set(v___f_195_, 8, v___f_194_);
lean_closure_set(v___f_195_, 9, v_inst_191_);
lean_closure_set(v___f_195_, 10, v_00_u03c3_u2081_u2082_185_);
v___x_196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_192_);
lean_ctor_set(v___x_196_, 1, v___f_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lift_equiv(lean_object* v_R_197_, lean_object* v_R_u2082_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_00_u03c3_u2081_u2082_201_, lean_object* v_M_202_, lean_object* v_N_203_, lean_object* v_P_u2082_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_TensorProduct_lift_equiv___redArg(v_inst_199_, v_inst_200_, v_00_u03c3_u2081_u2082_201_, v_inst_205_, v_inst_206_, v_inst_207_, v_inst_208_, v_inst_209_, v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lcurry___redArg(lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_00_u03c3_u2081_u2082_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v_toLinearMap_223_; 
v___x_221_ = lp_mathlib_TensorProduct_lift_equiv___redArg(v_inst_212_, v_inst_213_, v_00_u03c3_u2081_u2082_214_, v_inst_215_, v_inst_216_, v_inst_217_, v_inst_218_, v_inst_219_, v_inst_220_);
v___x_222_ = lp_mathlib_LinearEquiv_symm___redArg(v___x_221_);
v_toLinearMap_223_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_toLinearMap_223_);
lean_dec_ref(v___x_222_);
return v_toLinearMap_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_lcurry(lean_object* v_R_224_, lean_object* v_R_u2082_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_00_u03c3_u2081_u2082_228_, lean_object* v_M_229_, lean_object* v_N_230_, lean_object* v_P_u2082_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lp_mathlib_TensorProduct_lcurry___redArg(v_inst_226_, v_inst_227_, v_00_u03c3_u2081_u2082_228_, v_inst_232_, v_inst_233_, v_inst_234_, v_inst_235_, v_inst_236_, v_inst_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_curry___redArg(lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_00_u03c3_u2081_u2082_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_f_248_){
_start:
{
lean_object* v___x_21__overap_249_; lean_object* v___x_250_; 
v___x_21__overap_249_ = lp_mathlib_TensorProduct_lcurry___redArg(v_inst_239_, v_inst_240_, v_00_u03c3_u2081_u2082_241_, v_inst_242_, v_inst_243_, v_inst_244_, v_inst_245_, v_inst_246_, v_inst_247_);
v___x_250_ = lean_apply_1(v___x_21__overap_249_, v_f_248_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_curry(lean_object* v_R_251_, lean_object* v_R_u2082_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_00_u03c3_u2081_u2082_255_, lean_object* v_M_256_, lean_object* v_N_257_, lean_object* v_P_u2082_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_f_265_){
_start:
{
lean_object* v___x_31__overap_266_; lean_object* v___x_267_; 
v___x_31__overap_266_ = lp_mathlib_TensorProduct_lcurry___redArg(v_inst_253_, v_inst_254_, v_00_u03c3_u2081_u2082_255_, v_inst_259_, v_inst_260_, v_inst_261_, v_inst_262_, v_inst_263_, v_inst_264_);
v___x_267_ = lean_apply_1(v___x_31__overap_266_, v_f_265_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_comm___redArg(lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___f_275_; lean_object* v___f_276_; lean_object* v___f_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
lean_inc_n(v_inst_272_, 6);
lean_inc_n(v_inst_271_, 5);
lean_inc_ref_n(v_inst_270_, 5);
lean_inc_ref_n(v_inst_269_, 4);
lean_inc_ref_n(v_inst_268_, 7);
v___x_273_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_268_, v_inst_269_, v_inst_270_, v_inst_271_, v_inst_272_);
v___x_274_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_268_, v_inst_270_, v_inst_269_, v_inst_272_, v_inst_271_);
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_275_, 0, v_inst_268_);
lean_closure_set(v___f_275_, 1, v_inst_269_);
lean_closure_set(v___f_275_, 2, v_inst_270_);
lean_closure_set(v___f_275_, 3, v_inst_271_);
lean_closure_set(v___f_275_, 4, v_inst_272_);
lean_closure_set(v___f_275_, 5, v_inst_271_);
v___f_276_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_276_, 0, v_inst_268_);
lean_closure_set(v___f_276_, 1, v_inst_270_);
lean_closure_set(v___f_276_, 2, v_inst_269_);
lean_closure_set(v___f_276_, 3, v_inst_272_);
lean_closure_set(v___f_276_, 4, v_inst_271_);
lean_closure_set(v___f_276_, 5, v_inst_272_);
v___f_277_ = ((lean_object*)(lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___closed__0));
v___x_278_ = lp_mathlib_TensorProduct_mk(lean_box(0), v_inst_268_, lean_box(0), lean_box(0), v_inst_270_, v_inst_269_, v_inst_272_, v_inst_271_);
v___x_279_ = lp_mathlib_LinearMap_flip___redArg(v___x_278_);
v___x_280_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_268_, v_inst_268_, v___f_277_, v_inst_270_, v___x_274_, v_inst_272_, v___f_276_, v___x_279_);
v___x_281_ = lp_mathlib_TensorProduct_mk(lean_box(0), v_inst_268_, lean_box(0), lean_box(0), v_inst_269_, v_inst_270_, v_inst_271_, v_inst_272_);
lean_dec(v_inst_272_);
lean_dec_ref(v_inst_270_);
v___x_282_ = lp_mathlib_LinearMap_flip___redArg(v___x_281_);
v___x_283_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_268_, v_inst_268_, v___f_277_, v_inst_269_, v___x_273_, v_inst_271_, v___f_275_, v___x_282_);
v___x_284_ = lp_mathlib_LinearEquiv_ofLinearMap___redArg(v___x_280_, v___x_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_comm(lean_object* v_R_285_, lean_object* v_inst_286_, lean_object* v_M_287_, lean_object* v_N_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lp_mathlib_TensorProduct_comm___redArg(v_inst_286_, v_inst_289_, v_inst_290_, v_inst_291_, v_inst_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg___lam__0(lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_m_299_, lean_object* v___y_300_){
_start:
{
lean_object* v___x_95__overap_301_; lean_object* v___x_302_; 
v___x_95__overap_301_ = lp_mathlib_TensorProduct_mk(lean_box(0), v_inst_294_, lean_box(0), lean_box(0), v_inst_295_, v_inst_296_, v_inst_297_, v_inst_298_);
v___x_302_ = lean_apply_2(v___x_95__overap_301_, v_m_299_, v___y_300_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg___lam__0___boxed(lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_m_308_, lean_object* v___y_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg___lam__0(v_inst_303_, v_inst_304_, v_inst_305_, v_inst_306_, v_inst_307_, v_m_308_, v___y_309_);
lean_dec(v_inst_307_);
lean_dec(v_inst_306_);
lean_dec_ref(v_inst_305_);
lean_dec_ref(v_inst_304_);
lean_dec_ref(v_inst_303_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg(lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v___f_319_; lean_object* v___f_320_; lean_object* v___x_321_; lean_object* v___f_322_; lean_object* v___x_323_; 
lean_inc_n(v_inst_315_, 2);
lean_inc_n(v_inst_314_, 2);
lean_inc_ref_n(v_inst_313_, 3);
lean_inc_ref_n(v_inst_312_, 2);
lean_inc_ref_n(v_inst_311_, 2);
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg___lam__0___boxed), 7, 5);
lean_closure_set(v___f_319_, 0, v_inst_311_);
lean_closure_set(v___f_319_, 1, v_inst_312_);
lean_closure_set(v___f_319_, 2, v_inst_313_);
lean_closure_set(v___f_319_, 3, v_inst_314_);
lean_closure_set(v___f_319_, 4, v_inst_315_);
v___f_320_ = ((lean_object*)(lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___closed__0));
v___x_321_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_311_, v_inst_312_, v_inst_313_, v_inst_314_, v_inst_315_);
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_322_, 0, v_inst_311_);
lean_closure_set(v___f_322_, 1, v_inst_312_);
lean_closure_set(v___f_322_, 2, v_inst_313_);
lean_closure_set(v___f_322_, 3, v_inst_314_);
lean_closure_set(v___f_322_, 4, v_inst_315_);
lean_closure_set(v___f_322_, 5, v_inst_317_);
lean_inc_ref(v_inst_316_);
v___x_323_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_316_, v_inst_316_, v___f_320_, v_inst_313_, v___x_321_, v_inst_318_, v___f_322_, v___f_319_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul(lean_object* v_R_324_, lean_object* v_inst_325_, lean_object* v_A_326_, lean_object* v_S_327_, lean_object* v_M_328_, lean_object* v_N_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg(v_inst_325_, v_inst_330_, v_inst_331_, v_inst_332_, v_inst_333_, v_inst_334_, v_inst_335_, v_inst_336_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_mapOfCompatibleSMul___boxed(lean_object** _args){
lean_object* v_R_344_ = _args[0];
lean_object* v_inst_345_ = _args[1];
lean_object* v_A_346_ = _args[2];
lean_object* v_S_347_ = _args[3];
lean_object* v_M_348_ = _args[4];
lean_object* v_N_349_ = _args[5];
lean_object* v_inst_350_ = _args[6];
lean_object* v_inst_351_ = _args[7];
lean_object* v_inst_352_ = _args[8];
lean_object* v_inst_353_ = _args[9];
lean_object* v_inst_354_ = _args[10];
lean_object* v_inst_355_ = _args[11];
lean_object* v_inst_356_ = _args[12];
lean_object* v_inst_357_ = _args[13];
lean_object* v_inst_358_ = _args[14];
lean_object* v_inst_359_ = _args[15];
lean_object* v_inst_360_ = _args[16];
lean_object* v_inst_361_ = _args[17];
lean_object* v_inst_362_ = _args[18];
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_TensorProduct_mapOfCompatibleSMul(v_R_344_, v_inst_345_, v_A_346_, v_S_347_, v_M_348_, v_N_349_, v_inst_350_, v_inst_351_, v_inst_352_, v_inst_353_, v_inst_354_, v_inst_355_, v_inst_356_, v_inst_357_, v_inst_358_, v_inst_359_, v_inst_360_, v_inst_361_, v_inst_362_);
lean_dec(v_inst_359_);
lean_dec_ref(v_inst_358_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_equivOfCompatibleSMul___redArg___lam__0(lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v___y_372_){
_start:
{
lean_object* v___x_71__overap_373_; lean_object* v___x_374_; 
v___x_71__overap_373_ = lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg(v_inst_364_, v_inst_365_, v_inst_366_, v_inst_367_, v_inst_368_, v_inst_369_, v_inst_370_, v_inst_371_);
v___x_374_ = lean_apply_1(v___x_71__overap_373_, v___y_372_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_equivOfCompatibleSMul___redArg(lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_inst_382_){
_start:
{
lean_object* v___f_383_; lean_object* v___x_384_; lean_object* v___x_385_; 
lean_inc(v_inst_379_);
lean_inc(v_inst_378_);
lean_inc_ref(v_inst_375_);
lean_inc(v_inst_382_);
lean_inc(v_inst_381_);
lean_inc_ref(v_inst_377_);
lean_inc_ref(v_inst_376_);
lean_inc_ref(v_inst_380_);
v___f_383_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_equivOfCompatibleSMul___redArg___lam__0), 9, 8);
lean_closure_set(v___f_383_, 0, v_inst_380_);
lean_closure_set(v___f_383_, 1, v_inst_376_);
lean_closure_set(v___f_383_, 2, v_inst_377_);
lean_closure_set(v___f_383_, 3, v_inst_381_);
lean_closure_set(v___f_383_, 4, v_inst_382_);
lean_closure_set(v___f_383_, 5, v_inst_375_);
lean_closure_set(v___f_383_, 6, v_inst_378_);
lean_closure_set(v___f_383_, 7, v_inst_379_);
v___x_384_ = lp_mathlib_TensorProduct_mapOfCompatibleSMul___redArg(v_inst_375_, v_inst_376_, v_inst_377_, v_inst_378_, v_inst_379_, v_inst_380_, v_inst_381_, v_inst_382_);
v___x_385_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_385_, 0, v___x_384_);
lean_ctor_set(v___x_385_, 1, v___f_383_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_equivOfCompatibleSMul(lean_object* v_R_386_, lean_object* v_inst_387_, lean_object* v_A_388_, lean_object* v_S_389_, lean_object* v_M_390_, lean_object* v_N_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_TensorProduct_equivOfCompatibleSMul___redArg(v_inst_387_, v_inst_392_, v_inst_393_, v_inst_394_, v_inst_395_, v_inst_396_, v_inst_397_, v_inst_398_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_equivOfCompatibleSMul___boxed(lean_object** _args){
lean_object* v_R_407_ = _args[0];
lean_object* v_inst_408_ = _args[1];
lean_object* v_A_409_ = _args[2];
lean_object* v_S_410_ = _args[3];
lean_object* v_M_411_ = _args[4];
lean_object* v_N_412_ = _args[5];
lean_object* v_inst_413_ = _args[6];
lean_object* v_inst_414_ = _args[7];
lean_object* v_inst_415_ = _args[8];
lean_object* v_inst_416_ = _args[9];
lean_object* v_inst_417_ = _args[10];
lean_object* v_inst_418_ = _args[11];
lean_object* v_inst_419_ = _args[12];
lean_object* v_inst_420_ = _args[13];
lean_object* v_inst_421_ = _args[14];
lean_object* v_inst_422_ = _args[15];
lean_object* v_inst_423_ = _args[16];
lean_object* v_inst_424_ = _args[17];
lean_object* v_inst_425_ = _args[18];
lean_object* v_inst_426_ = _args[19];
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_TensorProduct_equivOfCompatibleSMul(v_R_407_, v_inst_408_, v_A_409_, v_S_410_, v_M_411_, v_N_412_, v_inst_413_, v_inst_414_, v_inst_415_, v_inst_416_, v_inst_417_, v_inst_418_, v_inst_419_, v_inst_420_, v_inst_421_, v_inst_422_, v_inst_423_, v_inst_424_, v_inst_425_, v_inst_426_);
lean_dec(v_inst_422_);
lean_dec_ref(v_inst_421_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_Neg_aux___redArg___lam__0(lean_object* v_toNeg_428_, lean_object* v___y_429_){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_430_ = lp_mathlib_LinearMap_id___lam__0(v___y_429_);
v___x_431_ = lean_apply_1(v_toNeg_428_, v___x_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_Neg_aux___redArg___lam__0___boxed(lean_object* v_toNeg_432_, lean_object* v___y_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_TensorProduct_Neg_aux___redArg___lam__0(v_toNeg_432_, v___y_433_);
lean_dec(v___y_433_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_Neg_aux___redArg(lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_){
_start:
{
lean_object* v_toAddMonoid_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v_toNeg_443_; lean_object* v___f_444_; lean_object* v___f_445_; lean_object* v___x_446_; lean_object* v___f_447_; lean_object* v___f_448_; lean_object* v___x_449_; 
v_toAddMonoid_440_ = lean_ctor_get(v_inst_436_, 0);
lean_inc_ref_n(v_toAddMonoid_440_, 3);
lean_inc_n(v_inst_439_, 2);
lean_inc_n(v_inst_438_, 3);
lean_inc_ref_n(v_inst_437_, 2);
lean_inc_ref_n(v_inst_435_, 3);
v___x_441_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_435_, v_toAddMonoid_440_, v_inst_437_, v_inst_438_, v_inst_439_);
v___x_442_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_436_);
lean_dec_ref(v_inst_436_);
v_toNeg_443_ = lean_ctor_get(v___x_442_, 1);
lean_inc(v_toNeg_443_);
lean_dec_ref(v___x_442_);
v___f_444_ = ((lean_object*)(lp_mathlib_TensorProduct_lift_equiv___redArg___lam__0___closed__0));
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_445_, 0, v_inst_435_);
lean_closure_set(v___f_445_, 1, v_toAddMonoid_440_);
lean_closure_set(v___f_445_, 2, v_inst_437_);
lean_closure_set(v___f_445_, 3, v_inst_438_);
lean_closure_set(v___f_445_, 4, v_inst_439_);
lean_closure_set(v___f_445_, 5, v_inst_438_);
v___x_446_ = lp_mathlib_TensorProduct_mk(lean_box(0), v_inst_435_, lean_box(0), lean_box(0), v_toAddMonoid_440_, v_inst_437_, v_inst_438_, v_inst_439_);
lean_dec(v_inst_438_);
lean_dec_ref(v_toAddMonoid_440_);
v___f_447_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_Neg_aux___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_447_, 0, v_toNeg_443_);
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_448_, 0, v___f_447_);
lean_closure_set(v___f_448_, 1, v___x_446_);
v___x_449_ = lp_mathlib_TensorProduct_liftAux___redArg(v_inst_435_, v_inst_435_, v___f_444_, v_inst_437_, v___x_441_, v_inst_439_, v___f_445_, v___f_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_Neg_aux(lean_object* v_R_450_, lean_object* v_inst_451_, lean_object* v_M_452_, lean_object* v_N_453_, lean_object* v_inst_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_inst_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lp_mathlib_TensorProduct_Neg_aux___redArg(v_inst_451_, v_inst_454_, v_inst_455_, v_inst_456_, v_inst_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_neg___redArg___lam__0(lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v___y_464_){
_start:
{
lean_object* v___x_34__overap_465_; lean_object* v___x_466_; 
v___x_34__overap_465_ = lp_mathlib_TensorProduct_Neg_aux___redArg(v_inst_459_, v_inst_460_, v_inst_461_, v_inst_462_, v_inst_463_);
v___x_466_ = lean_apply_1(v___x_34__overap_465_, v___y_464_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_neg___redArg(lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_inst_471_){
_start:
{
lean_object* v___f_472_; 
v___f_472_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_neg___redArg___lam__0), 6, 5);
lean_closure_set(v___f_472_, 0, v_inst_467_);
lean_closure_set(v___f_472_, 1, v_inst_468_);
lean_closure_set(v___f_472_, 2, v_inst_469_);
lean_closure_set(v___f_472_, 3, v_inst_470_);
lean_closure_set(v___f_472_, 4, v_inst_471_);
return v___f_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_neg(lean_object* v_R_473_, lean_object* v_inst_474_, lean_object* v_M_475_, lean_object* v_N_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_){
_start:
{
lean_object* v___f_481_; 
v___f_481_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_neg___redArg___lam__0), 6, 5);
lean_closure_set(v___f_481_, 0, v_inst_474_);
lean_closure_set(v___f_481_, 1, v_inst_477_);
lean_closure_set(v___f_481_, 2, v_inst_478_);
lean_closure_set(v___f_481_, 3, v_inst_479_);
lean_closure_set(v___f_481_, 4, v_inst_480_);
return v___f_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommGroup___redArg(lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v_toAddMonoid_487_; lean_object* v___x_488_; lean_object* v___f_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___f_492_; lean_object* v___f_493_; lean_object* v___x_494_; 
v_toAddMonoid_487_ = lean_ctor_get(v_inst_483_, 0);
lean_inc_ref_n(v_toAddMonoid_487_, 2);
lean_inc_n(v_inst_486_, 2);
lean_inc_n(v_inst_485_, 2);
lean_inc_ref_n(v_inst_484_, 2);
lean_inc_ref_n(v_inst_482_, 2);
v___x_488_ = lp_mathlib_TensorProduct_addMonoid___redArg(v_inst_482_, v_toAddMonoid_487_, v_inst_484_, v_inst_485_, v_inst_486_);
lean_inc_ref(v_inst_483_);
v___f_489_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_neg___redArg___lam__0), 6, 5);
lean_closure_set(v___f_489_, 0, v_inst_482_);
lean_closure_set(v___f_489_, 1, v_inst_483_);
lean_closure_set(v___f_489_, 2, v_inst_484_);
lean_closure_set(v___f_489_, 3, v_inst_485_);
lean_closure_set(v___f_489_, 4, v_inst_486_);
lean_inc_ref(v___f_489_);
lean_inc_ref(v___x_488_);
v___x_490_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_490_, 0, lean_box(0));
lean_closure_set(v___x_490_, 1, v___x_488_);
lean_closure_set(v___x_490_, 2, v___f_489_);
v___x_491_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v_inst_483_);
v___f_492_ = lean_alloc_closure((void*)(lp_mathlib_TensorProduct_leftHasSMul___redArg___lam__0___boxed), 8, 6);
lean_closure_set(v___f_492_, 0, v_inst_482_);
lean_closure_set(v___f_492_, 1, v_toAddMonoid_487_);
lean_closure_set(v___f_492_, 2, v_inst_484_);
lean_closure_set(v___f_492_, 3, v_inst_485_);
lean_closure_set(v___f_492_, 4, v_inst_486_);
lean_closure_set(v___f_492_, 5, v___x_491_);
v___f_493_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_493_, 0, v___f_492_);
v___x_494_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_494_, 0, v___x_488_);
lean_ctor_set(v___x_494_, 1, v___f_489_);
lean_ctor_set(v___x_494_, 2, v___x_490_);
lean_ctor_set(v___x_494_, 3, v___f_493_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TensorProduct_addCommGroup(lean_object* v_R_495_, lean_object* v_inst_496_, lean_object* v_M_497_, lean_object* v_N_498_, lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_inst_501_, lean_object* v_inst_502_){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = lp_mathlib_TensorProduct_addCommGroup___redArg(v_inst_496_, v_inst_499_, v_inst_500_, v_inst_501_, v_inst_502_);
return v___x_503_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_TensorProduct_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
