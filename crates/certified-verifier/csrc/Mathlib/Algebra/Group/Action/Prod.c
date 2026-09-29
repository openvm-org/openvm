// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.Prod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Faithful public import Mathlib.Algebra.Group.Action.Hom public import Mathlib.Algebra.Group.Prod
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instAddMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_addAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_addAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_addAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMulHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodOfSMulCommClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodOfSMulCommClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodOfVAddCommClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodOfVAddCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodOfVAddCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulAction_prodEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulAction_prodEquiv___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulAction_prodEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_MulAction_prodEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___elam__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___elam__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddAction_prodEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddAction_prodEquiv___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddAction_prodEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_AddAction_prodEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulAction___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___f_3_; 
v___f_3_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_3_, 0, v_inst_1_);
lean_closure_set(v___f_3_, 1, v_inst_2_);
return v___f_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulAction(lean_object* v_M_4_, lean_object* v_00_u03b1_5_, lean_object* v_00_u03b2_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_10_, 0, v_inst_8_);
lean_closure_set(v___f_10_, 1, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulAction___boxed(lean_object* v_M_11_, lean_object* v_00_u03b1_12_, lean_object* v_00_u03b2_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Prod_mulAction(v_M_11_, v_00_u03b1_12_, v_00_u03b2_13_, v_inst_14_, v_inst_15_, v_inst_16_);
lean_dec_ref(v_inst_14_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_addAction___redArg(lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_20_, 0, v_inst_18_);
lean_closure_set(v___f_20_, 1, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_addAction(lean_object* v_M_21_, lean_object* v_00_u03b1_22_, lean_object* v_00_u03b2_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___f_27_; 
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_27_, 0, v_inst_25_);
lean_closure_set(v___f_27_, 1, v_inst_26_);
return v___f_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_addAction___boxed(lean_object* v_M_28_, lean_object* v_00_u03b1_29_, lean_object* v_00_u03b2_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_Prod_addAction(v_M_28_, v_00_u03b1_29_, v_00_u03b2_30_, v_inst_31_, v_inst_32_, v_inst_33_);
lean_dec_ref(v_inst_31_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMulHom___redArg___lam__0(lean_object* v_inst_35_, lean_object* v_a_36_){
_start:
{
lean_object* v_fst_37_; lean_object* v_snd_38_; lean_object* v___x_39_; 
v_fst_37_ = lean_ctor_get(v_a_36_, 0);
lean_inc(v_fst_37_);
v_snd_38_ = lean_ctor_get(v_a_36_, 1);
lean_inc(v_snd_38_);
lean_dec_ref(v_a_36_);
v___x_39_ = lean_apply_2(v_inst_35_, v_fst_37_, v_snd_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMulHom___redArg(lean_object* v_inst_40_){
_start:
{
lean_object* v___f_41_; 
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_smulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_41_, 0, v_inst_40_);
return v___f_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMulHom(lean_object* v_00_u03b1_42_, lean_object* v_00_u03b2_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___f_49_; 
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_smulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_49_, 0, v_inst_46_);
return v___f_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMulHom___boxed(lean_object* v_00_u03b1_50_, lean_object* v_00_u03b2_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_smulMulHom(v_00_u03b1_50_, v_00_u03b2_51_, v_inst_52_, v_inst_53_, v_inst_54_, v_inst_55_, v_inst_56_);
lean_dec(v_inst_53_);
lean_dec_ref(v_inst_52_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidHom___redArg(lean_object* v_inst_58_){
_start:
{
lean_object* v___f_59_; 
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_smulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_59_, 0, v_inst_58_);
return v___f_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidHom(lean_object* v_00_u03b1_60_, lean_object* v_00_u03b2_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___f_67_; 
v___f_67_ = lean_alloc_closure((void*)(lp_mathlib_smulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_67_, 0, v_inst_64_);
return v___f_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidHom___boxed(lean_object* v_00_u03b1_68_, lean_object* v_00_u03b2_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_smulMonoidHom(v_00_u03b1_68_, v_00_u03b2_69_, v_inst_70_, v_inst_71_, v_inst_72_, v_inst_73_, v_inst_74_);
lean_dec_ref(v_inst_71_);
lean_dec_ref(v_inst_70_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodOfSMulCommClass___redArg___lam__0(lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_mn_78_, lean_object* v_a_79_){
_start:
{
lean_object* v_fst_80_; lean_object* v_snd_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v_fst_80_ = lean_ctor_get(v_mn_78_, 0);
lean_inc(v_fst_80_);
v_snd_81_ = lean_ctor_get(v_mn_78_, 1);
lean_inc(v_snd_81_);
lean_dec_ref(v_mn_78_);
v___x_82_ = lean_apply_2(v_inst_76_, v_snd_81_, v_a_79_);
v___x_83_ = lean_apply_2(v_inst_77_, v_fst_80_, v___x_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodOfSMulCommClass___redArg(lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___f_86_; 
v___f_86_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_prodOfSMulCommClass___redArg___lam__0), 4, 2);
lean_closure_set(v___f_86_, 0, v_inst_85_);
lean_closure_set(v___f_86_, 1, v_inst_84_);
return v___f_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodOfSMulCommClass(lean_object* v_M_87_, lean_object* v_N_88_, lean_object* v_00_u03b1_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___f_95_; 
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_prodOfSMulCommClass___redArg___lam__0), 4, 2);
lean_closure_set(v___f_95_, 0, v_inst_93_);
lean_closure_set(v___f_95_, 1, v_inst_92_);
return v___f_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodOfSMulCommClass___boxed(lean_object* v_M_96_, lean_object* v_N_97_, lean_object* v_00_u03b1_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_MulAction_prodOfSMulCommClass(v_M_96_, v_N_97_, v_00_u03b1_98_, v_inst_99_, v_inst_100_, v_inst_101_, v_inst_102_, v_inst_103_);
lean_dec_ref(v_inst_100_);
lean_dec_ref(v_inst_99_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodOfVAddCommClass___redArg(lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___f_107_; 
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_prodOfSMulCommClass___redArg___lam__0), 4, 2);
lean_closure_set(v___f_107_, 0, v_inst_106_);
lean_closure_set(v___f_107_, 1, v_inst_105_);
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodOfVAddCommClass(lean_object* v_M_108_, lean_object* v_N_109_, lean_object* v_00_u03b1_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___f_116_; 
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_prodOfSMulCommClass___redArg___lam__0), 4, 2);
lean_closure_set(v___f_116_, 0, v_inst_114_);
lean_closure_set(v___f_116_, 1, v_inst_113_);
return v___f_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodOfVAddCommClass___boxed(lean_object* v_M_117_, lean_object* v_N_118_, lean_object* v_00_u03b1_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_AddAction_prodOfVAddCommClass(v_M_117_, v_N_118_, v_00_u03b1_119_, v_inst_120_, v_inst_121_, v_inst_122_, v_inst_123_, v_inst_124_);
lean_dec_ref(v_inst_121_);
lean_dec_ref(v_inst_120_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___redArg___lam__0(lean_object* v___insts_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v_snd_129_; lean_object* v_fst_130_; lean_object* v_fst_131_; lean_object* v_fst_132_; lean_object* v_snd_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v_snd_129_ = lean_ctor_get(v___insts_126_, 1);
lean_inc(v_snd_129_);
v_fst_130_ = lean_ctor_get(v___insts_126_, 0);
lean_inc(v_fst_130_);
lean_dec_ref(v___insts_126_);
v_fst_131_ = lean_ctor_get(v_snd_129_, 0);
lean_inc(v_fst_131_);
lean_dec(v_snd_129_);
v_fst_132_ = lean_ctor_get(v___y_127_, 0);
lean_inc(v_fst_132_);
v_snd_133_ = lean_ctor_get(v___y_127_, 1);
lean_inc(v_snd_133_);
lean_dec_ref(v___y_127_);
v___x_134_ = lean_apply_2(v_fst_131_, v_snd_133_, v___y_128_);
v___x_135_ = lean_apply_2(v_fst_130_, v_fst_132_, v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___redArg___lam__0(lean_object* v_toOne_136_, lean_object* v_x_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_138_, 0, v_x_137_);
lean_ctor_set(v___x_138_, 1, v_toOne_136_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___redArg(lean_object* v___x_139_){
_start:
{
lean_object* v___x_140_; lean_object* v_toOne_141_; lean_object* v___f_142_; 
v___x_140_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_139_);
v_toOne_141_ = lean_ctor_get(v___x_140_, 0);
lean_inc(v_toOne_141_);
lean_dec_ref(v___x_140_);
v___f_142_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_142_, 0, v_toOne_141_);
return v___f_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0___redArg___lam__0(lean_object* v___x_143_, lean_object* v___y_144_){
_start:
{
lean_object* v___x_226__overap_145_; lean_object* v___x_146_; 
v___x_226__overap_145_ = lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___redArg(v___x_143_);
v___x_146_ = lean_apply_1(v___x_226__overap_145_, v___y_144_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___redArg___lam__0(lean_object* v_toOne_147_, lean_object* v_y_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_149_, 0, v_toOne_147_);
lean_ctor_set(v___x_149_, 1, v_y_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___redArg(lean_object* v___x_150_){
_start:
{
lean_object* v___x_151_; lean_object* v_toOne_152_; lean_object* v___f_153_; 
v___x_151_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_150_);
v_toOne_152_ = lean_ctor_get(v___x_151_, 0);
lean_inc(v_toOne_152_);
lean_dec_ref(v___x_151_);
v___f_153_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___redArg___lam__0), 2, 1);
lean_closure_set(v___f_153_, 0, v_toOne_152_);
return v___f_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0___redArg___lam__1(lean_object* v___x_154_, lean_object* v___y_155_){
_start:
{
lean_object* v___x_229__overap_156_; lean_object* v___x_157_; 
v___x_229__overap_156_ = lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___redArg(v___x_154_);
v___x_157_ = lean_apply_1(v___x_229__overap_156_, v___y_155_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__3___redArg(lean_object* v_x_158_, lean_object* v_g_159_, lean_object* v_n_160_, lean_object* v_a_161_){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_162_ = lean_apply_1(v_g_159_, v_n_160_);
v___x_163_ = lean_apply_2(v_x_158_, v___x_162_, v_a_161_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__3(lean_object* v_M_164_, lean_object* v_N_165_, lean_object* v_00_u03b1_166_, lean_object* v_x_167_, lean_object* v_N_168_, lean_object* v_g_169_, lean_object* v_n_170_, lean_object* v_a_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__3___redArg(v_x_167_, v_g_169_, v_n_170_, v_a_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__1___redArg(lean_object* v_x_173_, lean_object* v_g_174_, lean_object* v_n_175_, lean_object* v_a_176_){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_177_ = lean_apply_1(v_g_174_, v_n_175_);
v___x_178_ = lean_apply_2(v_x_173_, v___x_177_, v_a_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__1(lean_object* v_M_179_, lean_object* v_N_180_, lean_object* v_00_u03b1_181_, lean_object* v_x_182_, lean_object* v_N_183_, lean_object* v_g_184_, lean_object* v_n_185_, lean_object* v_a_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__1___redArg(v_x_182_, v_g_184_, v_n_185_, v_a_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0___redArg(lean_object* v___x_188_, lean_object* v___x_189_, lean_object* v_x_190_){
_start:
{
lean_object* v___f_191_; lean_object* v___f_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
v___f_191_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_prodEquiv___elam__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_191_, 0, v___x_189_);
v___f_192_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_prodEquiv___elam__0___redArg___lam__1), 2, 1);
lean_closure_set(v___f_192_, 0, v___x_188_);
lean_inc(v_x_190_);
v___x_193_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__1), 8, 6);
lean_closure_set(v___x_193_, 0, lean_box(0));
lean_closure_set(v___x_193_, 1, lean_box(0));
lean_closure_set(v___x_193_, 2, lean_box(0));
lean_closure_set(v___x_193_, 3, v_x_190_);
lean_closure_set(v___x_193_, 4, lean_box(0));
lean_closure_set(v___x_193_, 5, v___f_191_);
v___x_194_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul___at___00MulAction_prodEquiv___elam__0_spec__3), 8, 6);
lean_closure_set(v___x_194_, 0, lean_box(0));
lean_closure_set(v___x_194_, 1, lean_box(0));
lean_closure_set(v___x_194_, 2, lean_box(0));
lean_closure_set(v___x_194_, 3, v_x_190_);
lean_closure_set(v___x_194_, 4, lean_box(0));
lean_closure_set(v___x_194_, 5, v___f_192_);
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_194_);
lean_ctor_set(v___x_195_, 1, lean_box(0));
v___x_196_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_193_);
lean_ctor_set(v___x_196_, 1, v___x_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0(lean_object* v_M_197_, lean_object* v_N_198_, lean_object* v_00_u03b1_199_, lean_object* v___x_200_, lean_object* v___x_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_x_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_MulAction_prodEquiv___elam__0___redArg(v___x_200_, v___x_201_, v_x_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___elam__0___boxed(lean_object* v_M_206_, lean_object* v_N_207_, lean_object* v_00_u03b1_208_, lean_object* v___x_209_, lean_object* v___x_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_x_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib_MulAction_prodEquiv___elam__0(v_M_206_, v_N_207_, v_00_u03b1_208_, v___x_209_, v___x_210_, v_inst_211_, v_inst_212_, v_x_213_);
lean_dec_ref(v_inst_212_);
lean_dec_ref(v_inst_211_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv___redArg(lean_object* v_inst_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v___f_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___f_221_; lean_object* v___x_222_; 
v___f_218_ = ((lean_object*)(lp_mathlib_MulAction_prodEquiv___redArg___closed__0));
v___x_219_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_216_);
v___x_220_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_217_);
v___f_221_ = lean_alloc_closure((void*)(lp_mathlib_MulAction_prodEquiv___elam__0___boxed), 8, 7);
lean_closure_set(v___f_221_, 0, lean_box(0));
lean_closure_set(v___f_221_, 1, lean_box(0));
lean_closure_set(v___f_221_, 2, lean_box(0));
lean_closure_set(v___f_221_, 3, v___x_219_);
lean_closure_set(v___f_221_, 4, v___x_220_);
lean_closure_set(v___f_221_, 5, v_inst_216_);
lean_closure_set(v___f_221_, 6, v_inst_217_);
v___x_222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_222_, 0, v___f_221_);
lean_ctor_set(v___x_222_, 1, v___f_218_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAction_prodEquiv(lean_object* v_M_223_, lean_object* v_N_224_, lean_object* v_00_u03b1_225_, lean_object* v_inst_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib_MulAction_prodEquiv___redArg(v_inst_226_, v_inst_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0(lean_object* v_M_229_, lean_object* v_N_230_, lean_object* v___x_231_, lean_object* v___x_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___redArg(v___x_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0___boxed(lean_object* v_M_234_, lean_object* v_N_235_, lean_object* v___x_236_, lean_object* v___x_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_MonoidHom_inl___at___00MulAction_prodEquiv___elam__0_spec__0(v_M_234_, v_N_235_, v___x_236_, v___x_237_);
lean_dec_ref(v___x_236_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2(lean_object* v_M_239_, lean_object* v_N_240_, lean_object* v___x_241_, lean_object* v___x_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___redArg(v___x_241_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2___boxed(lean_object* v_M_244_, lean_object* v_N_245_, lean_object* v___x_246_, lean_object* v___x_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_MonoidHom_inr___at___00MulAction_prodEquiv___elam__0_spec__2(v_M_244_, v_N_245_, v___x_246_, v___x_247_);
lean_dec_ref(v___x_247_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___redArg___lam__0(lean_object* v___insts_249_, lean_object* v___y_250_, lean_object* v___y_251_){
_start:
{
lean_object* v_snd_252_; lean_object* v_fst_253_; lean_object* v_fst_254_; lean_object* v___x_255_; 
v_snd_252_ = lean_ctor_get(v___insts_249_, 1);
lean_inc(v_snd_252_);
v_fst_253_ = lean_ctor_get(v___insts_249_, 0);
lean_inc(v_fst_253_);
lean_dec_ref(v___insts_249_);
v_fst_254_ = lean_ctor_get(v_snd_252_, 0);
lean_inc(v_fst_254_);
lean_dec(v_snd_252_);
v___x_255_ = lp_mathlib_MulAction_prodOfSMulCommClass___redArg___lam__0(v_fst_254_, v_fst_253_, v___y_250_, v___y_251_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1_spec__2___redArg(lean_object* v_x_256_, lean_object* v_g_257_, lean_object* v_n_258_, lean_object* v_a_259_){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_260_ = lean_apply_1(v_g_257_, v_n_258_);
v___x_261_ = lean_apply_2(v_x_256_, v___x_260_, v_a_259_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg___lam__1(lean_object* v_x_262_, lean_object* v___f_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1_spec__2___redArg(v_x_262_, v___f_263_, v___y_264_, v___y_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg___lam__0(lean_object* v_g_267_, lean_object* v___y_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lean_apply_1(v_g_267_, v___y_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg(lean_object* v_x_270_, lean_object* v_g_271_){
_start:
{
lean_object* v___f_272_; lean_object* v___f_273_; 
v___f_272_ = lean_alloc_closure((void*)(lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_272_, 0, v_g_271_);
v___f_273_ = lean_alloc_closure((void*)(lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg___lam__1), 4, 2);
lean_closure_set(v___f_273_, 0, v_x_270_);
lean_closure_set(v___f_273_, 1, v___f_272_);
return v___f_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___redArg___lam__0(lean_object* v_toZero_274_, lean_object* v_y_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_276_, 0, v_toZero_274_);
lean_ctor_set(v___x_276_, 1, v_y_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___redArg(lean_object* v___x_277_){
_start:
{
lean_object* v___x_278_; lean_object* v_toZero_279_; lean_object* v___f_280_; 
v___x_278_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_277_);
v_toZero_279_ = lean_ctor_get(v___x_278_, 0);
lean_inc(v_toZero_279_);
lean_dec_ref(v___x_278_);
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___redArg___lam__0), 2, 1);
lean_closure_set(v___f_280_, 0, v_toZero_279_);
return v___f_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___redArg___lam__0(lean_object* v_toZero_281_, lean_object* v_x_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_283_, 0, v_x_282_);
lean_ctor_set(v___x_283_, 1, v_toZero_281_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___redArg(lean_object* v___x_284_){
_start:
{
lean_object* v___x_285_; lean_object* v_toZero_286_; lean_object* v___f_287_; 
v___x_285_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_284_);
v_toZero_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_toZero_286_);
lean_dec_ref(v___x_285_);
v___f_287_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_287_, 0, v_toZero_286_);
return v___f_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3_spec__5___redArg(lean_object* v_x_288_, lean_object* v_g_289_, lean_object* v_n_290_, lean_object* v_a_291_){
_start:
{
lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_292_ = lean_apply_1(v_g_289_, v_n_290_);
v___x_293_ = lean_apply_2(v_x_288_, v___x_292_, v_a_291_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___redArg___lam__1(lean_object* v_x_294_, lean_object* v___f_295_, lean_object* v___y_296_, lean_object* v___y_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3_spec__5___redArg(v_x_294_, v___f_295_, v___y_296_, v___y_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___redArg(lean_object* v_x_299_, lean_object* v_g_300_){
_start:
{
lean_object* v___f_301_; lean_object* v___f_302_; 
v___f_301_ = lean_alloc_closure((void*)(lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_301_, 0, v_g_300_);
v___f_302_ = lean_alloc_closure((void*)(lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___redArg___lam__1), 4, 2);
lean_closure_set(v___f_302_, 0, v_x_299_);
lean_closure_set(v___f_302_, 1, v___f_301_);
return v___f_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___elam__0___redArg(lean_object* v___x_303_, lean_object* v___x_304_, lean_object* v___x_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_x_308_){
_start:
{
lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; 
v___x_309_ = lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___redArg(v___x_304_);
lean_inc(v_x_308_);
v___x_310_ = lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg(v_x_308_, v___x_309_);
v___x_311_ = lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___redArg(v___x_303_);
v___x_312_ = lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___redArg(v_x_308_, v___x_311_);
v___x_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_312_);
lean_ctor_set(v___x_313_, 1, lean_box(0));
v___x_314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_314_, 0, v___x_310_);
lean_ctor_set(v___x_314_, 1, v___x_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___elam__0___redArg___boxed(lean_object* v___x_315_, lean_object* v___x_316_, lean_object* v___x_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_x_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_AddAction_prodEquiv___elam__0___redArg(v___x_315_, v___x_316_, v___x_317_, v_inst_318_, v_inst_319_, v_x_320_);
lean_dec_ref(v_inst_319_);
lean_dec_ref(v_inst_318_);
lean_dec_ref(v___x_317_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___elam__0(lean_object* v_M_322_, lean_object* v_N_323_, lean_object* v_00_u03b1_324_, lean_object* v___x_325_, lean_object* v___x_326_, lean_object* v___x_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_x_330_){
_start:
{
lean_object* v___x_331_; 
v___x_331_ = lp_mathlib_AddAction_prodEquiv___elam__0___redArg(v___x_325_, v___x_326_, v___x_327_, v_inst_328_, v_inst_329_, v_x_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___elam__0___boxed(lean_object* v_M_332_, lean_object* v_N_333_, lean_object* v_00_u03b1_334_, lean_object* v___x_335_, lean_object* v___x_336_, lean_object* v___x_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_x_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_AddAction_prodEquiv___elam__0(v_M_332_, v_N_333_, v_00_u03b1_334_, v___x_335_, v___x_336_, v___x_337_, v_inst_338_, v_inst_339_, v_x_340_);
lean_dec_ref(v_inst_339_);
lean_dec_ref(v_inst_338_);
lean_dec_ref(v___x_337_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv___redArg(lean_object* v_inst_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v___f_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___f_349_; lean_object* v___x_350_; 
v___f_345_ = ((lean_object*)(lp_mathlib_AddAction_prodEquiv___redArg___closed__0));
lean_inc_ref(v_inst_344_);
lean_inc_ref(v_inst_343_);
v___x_346_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_343_, v_inst_344_);
v___x_347_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_343_);
v___x_348_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_344_);
v___f_349_ = lean_alloc_closure((void*)(lp_mathlib_AddAction_prodEquiv___elam__0___boxed), 9, 8);
lean_closure_set(v___f_349_, 0, lean_box(0));
lean_closure_set(v___f_349_, 1, lean_box(0));
lean_closure_set(v___f_349_, 2, lean_box(0));
lean_closure_set(v___f_349_, 3, v___x_347_);
lean_closure_set(v___f_349_, 4, v___x_348_);
lean_closure_set(v___f_349_, 5, v___x_346_);
lean_closure_set(v___f_349_, 6, v_inst_343_);
lean_closure_set(v___f_349_, 7, v_inst_344_);
v___x_350_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_350_, 0, v___f_349_);
lean_ctor_set(v___x_350_, 1, v___f_345_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_prodEquiv(lean_object* v_M_351_, lean_object* v_N_352_, lean_object* v_00_u03b1_353_, lean_object* v_inst_354_, lean_object* v_inst_355_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = lp_mathlib_AddAction_prodEquiv___redArg(v_inst_354_, v_inst_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0(lean_object* v_M_357_, lean_object* v_N_358_, lean_object* v___x_359_, lean_object* v___x_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___redArg(v___x_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0___boxed(lean_object* v_M_362_, lean_object* v_N_363_, lean_object* v___x_364_, lean_object* v___x_365_){
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_mathlib_AddMonoidHom_inl___at___00AddAction_prodEquiv___elam__0_spec__0(v_M_362_, v_N_363_, v___x_364_, v___x_365_);
lean_dec_ref(v___x_364_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2(lean_object* v_M_367_, lean_object* v_N_368_, lean_object* v___x_369_, lean_object* v___x_370_){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___redArg(v___x_369_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2___boxed(lean_object* v_M_372_, lean_object* v_N_373_, lean_object* v___x_374_, lean_object* v___x_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib_AddMonoidHom_inr___at___00AddAction_prodEquiv___elam__0_spec__2(v_M_372_, v_N_373_, v___x_374_, v___x_375_);
lean_dec_ref(v___x_375_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1_spec__2(lean_object* v_M_377_, lean_object* v_N_378_, lean_object* v_00_u03b1_379_, lean_object* v_x_380_, lean_object* v_N_381_, lean_object* v_g_382_, lean_object* v_n_383_, lean_object* v_a_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1_spec__2___redArg(v_x_380_, v_g_382_, v_n_383_, v_a_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1(lean_object* v_M_386_, lean_object* v_N_387_, lean_object* v_00_u03b1_388_, lean_object* v___x_389_, lean_object* v_x_390_, lean_object* v_inst_391_, lean_object* v_g_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___redArg(v_x_390_, v_g_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1___boxed(lean_object* v_M_394_, lean_object* v_N_395_, lean_object* v_00_u03b1_396_, lean_object* v___x_397_, lean_object* v_x_398_, lean_object* v_inst_399_, lean_object* v_g_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__1(v_M_394_, v_N_395_, v_00_u03b1_396_, v___x_397_, v_x_398_, v_inst_399_, v_g_400_);
lean_dec_ref(v_inst_399_);
lean_dec_ref(v___x_397_);
return v_res_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3_spec__5(lean_object* v_M_402_, lean_object* v_N_403_, lean_object* v_00_u03b1_404_, lean_object* v_x_405_, lean_object* v_N_406_, lean_object* v_g_407_, lean_object* v_n_408_, lean_object* v_a_409_){
_start:
{
lean_object* v___x_410_; 
v___x_410_ = lp_mathlib_VAdd_comp_vadd___at___00AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3_spec__5___redArg(v_x_405_, v_g_407_, v_n_408_, v_a_409_);
return v___x_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3(lean_object* v_M_411_, lean_object* v_N_412_, lean_object* v_00_u03b1_413_, lean_object* v___x_414_, lean_object* v_x_415_, lean_object* v_inst_416_, lean_object* v_g_417_){
_start:
{
lean_object* v___x_418_; 
v___x_418_ = lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___redArg(v_x_415_, v_g_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3___boxed(lean_object* v_M_419_, lean_object* v_N_420_, lean_object* v_00_u03b1_421_, lean_object* v___x_422_, lean_object* v_x_423_, lean_object* v_inst_424_, lean_object* v_g_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_mathlib_AddAction_compHom___at___00AddAction_prodEquiv___elam__0_spec__3(v_M_419_, v_N_420_, v_00_u03b1_421_, v___x_422_, v_x_423_, v_inst_424_, v_g_425_);
lean_dec_ref(v_inst_424_);
lean_dec_ref(v___x_422_);
return v_res_426_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
