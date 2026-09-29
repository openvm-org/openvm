// Lean compiler output
// Module: Mathlib.Algebra.Group.Pointwise.Finset.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Pointwise.Set.Finite public import Mathlib.Algebra.Group.Pointwise.Set.ListOfFn public import Mathlib.Algebra.Order.Monoid.Unbundled.WithTop public import Mathlib.Data.Finset.Max public import Mathlib.Data.Finset.NAry public import Mathlib.Data.Finset.Preimage
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
lean_object* lp_mathlib_Finset_image_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_image(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* l_nsmulRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_fintypeSingleton___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonOneHom___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Finset_singletonOneHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_singletonOneHom___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_singletonOneHom___closed__0 = (const lean_object*)&lp_mathlib_Finset_singletonOneHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonOneHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonOneHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonZeroHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonZeroHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageOneHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageOneHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageOneHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageZeroHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_neg___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_neg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_mul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_mul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_add___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_add(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonMulHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonMulHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonAddHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonAddHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMulHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_div___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_div(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sub___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sub(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_npow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_npow___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_npow___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_npow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_nsmul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_nsmul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_nsmul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_nsmul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zpow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zpow___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zpow___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zpow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zsmul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zsmul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zsmul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_zsmul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_semigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_semigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_commSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_commSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addCommSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addCommSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_mulOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_mulOneClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMonoidHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddMonoidHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_monoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_monoid___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_monoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_monoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_commMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_commMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_addCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionMonoid___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtractionMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtractionMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtractionCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtractionCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_one___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_box(0);
v___x_3_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3_, 0, v_inst_1_);
lean_ctor_set(v___x_3_, 1, v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_one(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_mathlib_Finset_one___redArg(v_inst_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zero___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lean_box(0);
v___x_9_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_9_, 0, v_inst_7_);
lean_ctor_set(v___x_9_, 1, v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zero(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Finset_zero___redArg(v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonOneHom___lam__0(lean_object* v_a_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_box(0);
v___x_15_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_15_, 0, v_a_13_);
lean_ctor_set(v___x_15_, 1, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonOneHom(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___f_19_; 
v___f_19_ = ((lean_object*)(lp_mathlib_Finset_singletonOneHom___closed__0));
return v___f_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonOneHom___boxed(lean_object* v_00_u03b1_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Finset_singletonOneHom(v_00_u03b1_20_, v_inst_21_);
lean_dec(v_inst_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonZeroHom(lean_object* v_00_u03b1_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = ((lean_object*)(lp_mathlib_Finset_singletonOneHom___closed__0));
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonZeroHom___boxed(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Finset_singletonZeroHom(v_00_u03b1_26_, v_inst_27_);
lean_dec(v_inst_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageOneHom___redArg(lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_f_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_apply_1(v_inst_30_, v_f_31_);
v___x_33_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_33_, 0, lean_box(0));
lean_closure_set(v___x_33_, 1, lean_box(0));
lean_closure_set(v___x_33_, 2, v_inst_29_);
lean_closure_set(v___x_33_, 3, v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageOneHom(lean_object* v_F_34_, lean_object* v_00_u03b1_35_, lean_object* v_00_u03b2_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_f_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Finset_imageOneHom___redArg(v_inst_38_, v_inst_40_, v_f_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageOneHom___boxed(lean_object* v_F_44_, lean_object* v_00_u03b1_45_, lean_object* v_00_u03b2_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_f_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_Finset_imageOneHom(v_F_44_, v_00_u03b1_45_, v_00_u03b2_46_, v_inst_47_, v_inst_48_, v_inst_49_, v_inst_50_, v_inst_51_, v_f_52_);
lean_dec(v_inst_49_);
lean_dec(v_inst_47_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageZeroHom___redArg(lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_f_56_){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = lean_apply_1(v_inst_55_, v_f_56_);
v___x_58_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_58_, 0, lean_box(0));
lean_closure_set(v___x_58_, 1, lean_box(0));
lean_closure_set(v___x_58_, 2, v_inst_54_);
lean_closure_set(v___x_58_, 3, v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageZeroHom(lean_object* v_F_59_, lean_object* v_00_u03b1_60_, lean_object* v_00_u03b2_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_f_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_mathlib_Finset_imageZeroHom___redArg(v_inst_63_, v_inst_65_, v_f_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageZeroHom___boxed(lean_object* v_F_69_, lean_object* v_00_u03b1_70_, lean_object* v_00_u03b2_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_f_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_Finset_imageZeroHom(v_F_69_, v_00_u03b1_70_, v_00_u03b2_71_, v_inst_72_, v_inst_73_, v_inst_74_, v_inst_75_, v_inst_76_, v_f_77_);
lean_dec(v_inst_74_);
lean_dec(v_inst_72_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inv___redArg(lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_81_, 0, lean_box(0));
lean_closure_set(v___x_81_, 1, lean_box(0));
lean_closure_set(v___x_81_, 2, v_inst_79_);
lean_closure_set(v___x_81_, 3, v_inst_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inv(lean_object* v_00_u03b1_82_, lean_object* v_inst_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_85_, 0, lean_box(0));
lean_closure_set(v___x_85_, 1, lean_box(0));
lean_closure_set(v___x_85_, 2, v_inst_83_);
lean_closure_set(v___x_85_, 3, v_inst_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_neg___redArg(lean_object* v_inst_86_, lean_object* v_inst_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_88_, 0, lean_box(0));
lean_closure_set(v___x_88_, 1, lean_box(0));
lean_closure_set(v___x_88_, 2, v_inst_86_);
lean_closure_set(v___x_88_, 3, v_inst_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_neg(lean_object* v_00_u03b1_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_92_, 0, lean_box(0));
lean_closure_set(v___x_92_, 1, lean_box(0));
lean_closure_set(v___x_92_, 2, v_inst_90_);
lean_closure_set(v___x_92_, 3, v_inst_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mul___redArg___lam__0(lean_object* v_inst_93_, lean_object* v_x1_94_, lean_object* v_x2_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lean_apply_2(v_inst_93_, v_x1_94_, v_x2_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mul___redArg(lean_object* v_inst_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v___f_99_; lean_object* v___x_100_; 
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_Finset_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_99_, 0, v_inst_98_);
v___x_100_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image_u2082), 7, 5);
lean_closure_set(v___x_100_, 0, lean_box(0));
lean_closure_set(v___x_100_, 1, lean_box(0));
lean_closure_set(v___x_100_, 2, lean_box(0));
lean_closure_set(v___x_100_, 3, v_inst_97_);
lean_closure_set(v___x_100_, 4, v___f_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mul(lean_object* v_00_u03b1_101_, lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lp_mathlib_Finset_mul___redArg(v_inst_102_, v_inst_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_add___redArg(lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___f_107_; lean_object* v___x_108_; 
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_Finset_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_107_, 0, v_inst_106_);
v___x_108_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image_u2082), 7, 5);
lean_closure_set(v___x_108_, 0, lean_box(0));
lean_closure_set(v___x_108_, 1, lean_box(0));
lean_closure_set(v___x_108_, 2, lean_box(0));
lean_closure_set(v___x_108_, 3, v_inst_105_);
lean_closure_set(v___x_108_, 4, v___f_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_add(lean_object* v_00_u03b1_109_, lean_object* v_inst_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_Finset_add___redArg(v_inst_110_, v_inst_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonMulHom(lean_object* v_00_u03b1_113_, lean_object* v_inst_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___f_116_; 
v___f_116_ = ((lean_object*)(lp_mathlib_Finset_singletonOneHom___closed__0));
return v___f_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonMulHom___boxed(lean_object* v_00_u03b1_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_Finset_singletonMulHom(v_00_u03b1_117_, v_inst_118_, v_inst_119_);
lean_dec(v_inst_119_);
lean_dec_ref(v_inst_118_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonAddHom(lean_object* v_00_u03b1_121_, lean_object* v_inst_122_, lean_object* v_inst_123_){
_start:
{
lean_object* v___f_124_; 
v___f_124_ = ((lean_object*)(lp_mathlib_Finset_singletonOneHom___closed__0));
return v___f_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonAddHom___boxed(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_Finset_singletonAddHom(v_00_u03b1_125_, v_inst_126_, v_inst_127_);
lean_dec(v_inst_127_);
lean_dec_ref(v_inst_126_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMulHom___redArg(lean_object* v_inst_129_, lean_object* v_f_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_132_ = lean_apply_1(v_inst_129_, v_f_130_);
v___x_133_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_133_, 0, lean_box(0));
lean_closure_set(v___x_133_, 1, lean_box(0));
lean_closure_set(v___x_133_, 2, v_inst_131_);
lean_closure_set(v___x_133_, 3, v___x_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMulHom(lean_object* v_F_134_, lean_object* v_00_u03b1_135_, lean_object* v_00_u03b2_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_f_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_Finset_imageMulHom___redArg(v_inst_140_, v_f_142_, v_inst_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMulHom___boxed(lean_object* v_F_145_, lean_object* v_00_u03b1_146_, lean_object* v_00_u03b2_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_f_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_Finset_imageMulHom(v_F_145_, v_00_u03b1_146_, v_00_u03b2_147_, v_inst_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_inst_152_, v_f_153_, v_inst_154_);
lean_dec(v_inst_150_);
lean_dec(v_inst_149_);
lean_dec_ref(v_inst_148_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddHom___redArg(lean_object* v_inst_156_, lean_object* v_f_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_159_ = lean_apply_1(v_inst_156_, v_f_157_);
v___x_160_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_160_, 0, lean_box(0));
lean_closure_set(v___x_160_, 1, lean_box(0));
lean_closure_set(v___x_160_, 2, v_inst_158_);
lean_closure_set(v___x_160_, 3, v___x_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddHom(lean_object* v_F_161_, lean_object* v_00_u03b1_162_, lean_object* v_00_u03b2_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_f_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_mathlib_Finset_imageAddHom___redArg(v_inst_167_, v_f_169_, v_inst_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddHom___boxed(lean_object* v_F_172_, lean_object* v_00_u03b1_173_, lean_object* v_00_u03b2_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_f_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_Finset_imageAddHom(v_F_172_, v_00_u03b1_173_, v_00_u03b2_174_, v_inst_175_, v_inst_176_, v_inst_177_, v_inst_178_, v_inst_179_, v_f_180_, v_inst_181_);
lean_dec(v_inst_177_);
lean_dec(v_inst_176_);
lean_dec_ref(v_inst_175_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_div___redArg(lean_object* v_inst_183_, lean_object* v_inst_184_){
_start:
{
lean_object* v___f_185_; lean_object* v___x_186_; 
v___f_185_ = lean_alloc_closure((void*)(lp_mathlib_Finset_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_185_, 0, v_inst_184_);
v___x_186_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image_u2082), 7, 5);
lean_closure_set(v___x_186_, 0, lean_box(0));
lean_closure_set(v___x_186_, 1, lean_box(0));
lean_closure_set(v___x_186_, 2, lean_box(0));
lean_closure_set(v___x_186_, 3, v_inst_183_);
lean_closure_set(v___x_186_, 4, v___f_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_div(lean_object* v_00_u03b1_187_, lean_object* v_inst_188_, lean_object* v_inst_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lp_mathlib_Finset_div___redArg(v_inst_188_, v_inst_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sub___redArg(lean_object* v_inst_191_, lean_object* v_inst_192_){
_start:
{
lean_object* v___f_193_; lean_object* v___x_194_; 
v___f_193_ = lean_alloc_closure((void*)(lp_mathlib_Finset_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_193_, 0, v_inst_192_);
v___x_194_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image_u2082), 7, 5);
lean_closure_set(v___x_194_, 0, lean_box(0));
lean_closure_set(v___x_194_, 1, lean_box(0));
lean_closure_set(v___x_194_, 2, lean_box(0));
lean_closure_set(v___x_194_, 3, v_inst_191_);
lean_closure_set(v___x_194_, 4, v___f_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sub(lean_object* v_00_u03b1_195_, lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_Finset_sub___redArg(v_inst_196_, v_inst_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_npow___redArg___lam__0(lean_object* v___x_199_, lean_object* v___x_200_, lean_object* v_s_201_, lean_object* v_n_202_){
_start:
{
lean_object* v___x_203_; 
v___x_203_ = l_npowRec___redArg(v___x_199_, v___x_200_, v_n_202_, v_s_201_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_npow___redArg___lam__0___boxed(lean_object* v___x_204_, lean_object* v___x_205_, lean_object* v_s_206_, lean_object* v_n_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_Finset_npow___redArg___lam__0(v___x_204_, v___x_205_, v_s_206_, v_n_207_);
lean_dec(v_n_207_);
lean_dec(v___x_204_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_npow___redArg(lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_){
_start:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___f_214_; 
v___x_212_ = lp_mathlib_Finset_one___redArg(v_inst_210_);
v___x_213_ = lp_mathlib_Finset_mul___redArg(v_inst_209_, v_inst_211_);
v___f_214_ = lean_alloc_closure((void*)(lp_mathlib_Finset_npow___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_214_, 0, v___x_212_);
lean_closure_set(v___f_214_, 1, v___x_213_);
return v___f_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_npow(lean_object* v_00_u03b1_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_Finset_npow___redArg(v_inst_216_, v_inst_217_, v_inst_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_nsmul___redArg___lam__0(lean_object* v___x_220_, lean_object* v___x_221_, lean_object* v_n_222_, lean_object* v_s_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = l_nsmulRec___redArg(v___x_220_, v___x_221_, v_n_222_, v_s_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_nsmul___redArg___lam__0___boxed(lean_object* v___x_225_, lean_object* v___x_226_, lean_object* v_n_227_, lean_object* v_s_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_Finset_nsmul___redArg___lam__0(v___x_225_, v___x_226_, v_n_227_, v_s_228_);
lean_dec(v_n_227_);
lean_dec(v___x_225_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_nsmul___redArg(lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___f_235_; 
v___x_233_ = lp_mathlib_Finset_zero___redArg(v_inst_231_);
v___x_234_ = lp_mathlib_Finset_add___redArg(v_inst_230_, v_inst_232_);
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_Finset_nsmul___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_235_, 0, v___x_233_);
lean_closure_set(v___f_235_, 1, v___x_234_);
return v___f_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_nsmul(lean_object* v_00_u03b1_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_){
_start:
{
lean_object* v___x_240_; 
v___x_240_ = lp_mathlib_Finset_nsmul___redArg(v_inst_237_, v_inst_238_, v_inst_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zpow___redArg___lam__0(lean_object* v___x_241_, lean_object* v___x_242_, lean_object* v___x_243_, lean_object* v_s_244_, lean_object* v_n_245_){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_246_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_246_, 0, lean_box(0));
lean_closure_set(v___x_246_, 1, v___x_241_);
lean_closure_set(v___x_246_, 2, v___x_242_);
v___x_247_ = lp_mathlib_zpowRec___redArg(v___x_243_, v___x_246_, v_n_245_, v_s_244_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zpow___redArg___lam__0___boxed(lean_object* v___x_248_, lean_object* v___x_249_, lean_object* v___x_250_, lean_object* v_s_251_, lean_object* v_n_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_mathlib_Finset_zpow___redArg___lam__0(v___x_248_, v___x_249_, v___x_250_, v_s_251_, v_n_252_);
lean_dec(v_n_252_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zpow___redArg(lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_inst_257_){
_start:
{
lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___f_261_; 
v___x_258_ = lp_mathlib_Finset_one___redArg(v_inst_255_);
lean_inc_ref(v_inst_254_);
v___x_259_ = lp_mathlib_Finset_mul___redArg(v_inst_254_, v_inst_256_);
v___x_260_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_260_, 0, lean_box(0));
lean_closure_set(v___x_260_, 1, lean_box(0));
lean_closure_set(v___x_260_, 2, v_inst_254_);
lean_closure_set(v___x_260_, 3, v_inst_257_);
v___f_261_ = lean_alloc_closure((void*)(lp_mathlib_Finset_zpow___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_261_, 0, v___x_258_);
lean_closure_set(v___f_261_, 1, v___x_259_);
lean_closure_set(v___f_261_, 2, v___x_260_);
return v___f_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zpow(lean_object* v_00_u03b1_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_inst_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_mathlib_Finset_zpow___redArg(v_inst_263_, v_inst_264_, v_inst_265_, v_inst_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zsmul___redArg___lam__0(lean_object* v___x_268_, lean_object* v___x_269_, lean_object* v___x_270_, lean_object* v_n_271_, lean_object* v_s_272_){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_273_, 0, lean_box(0));
lean_closure_set(v___x_273_, 1, v___x_268_);
lean_closure_set(v___x_273_, 2, v___x_269_);
v___x_274_ = lp_mathlib_zsmulRec___redArg(v___x_270_, v___x_273_, v_n_271_, v_s_272_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zsmul___redArg___lam__0___boxed(lean_object* v___x_275_, lean_object* v___x_276_, lean_object* v___x_277_, lean_object* v_n_278_, lean_object* v_s_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_Finset_zsmul___redArg___lam__0(v___x_275_, v___x_276_, v___x_277_, v_n_278_, v_s_279_);
lean_dec(v_n_278_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zsmul___redArg(lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___f_288_; 
v___x_285_ = lp_mathlib_Finset_zero___redArg(v_inst_282_);
lean_inc_ref(v_inst_281_);
v___x_286_ = lp_mathlib_Finset_add___redArg(v_inst_281_, v_inst_283_);
v___x_287_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_287_, 0, lean_box(0));
lean_closure_set(v___x_287_, 1, lean_box(0));
lean_closure_set(v___x_287_, 2, v_inst_281_);
lean_closure_set(v___x_287_, 3, v_inst_284_);
v___f_288_ = lean_alloc_closure((void*)(lp_mathlib_Finset_zsmul___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_288_, 0, v___x_285_);
lean_closure_set(v___f_288_, 1, v___x_286_);
lean_closure_set(v___f_288_, 2, v___x_287_);
return v___f_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_zsmul(lean_object* v_00_u03b1_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lp_mathlib_Finset_zsmul___redArg(v_inst_290_, v_inst_291_, v_inst_292_, v_inst_293_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_semigroup___redArg(lean_object* v_inst_295_, lean_object* v_inst_296_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lp_mathlib_Finset_mul___redArg(v_inst_295_, v_inst_296_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_semigroup(lean_object* v_00_u03b1_298_, lean_object* v_inst_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lp_mathlib_Finset_mul___redArg(v_inst_299_, v_inst_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addSemigroup___redArg(lean_object* v_inst_302_, lean_object* v_inst_303_){
_start:
{
lean_object* v___x_304_; 
v___x_304_ = lp_mathlib_Finset_add___redArg(v_inst_302_, v_inst_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addSemigroup(lean_object* v_00_u03b1_305_, lean_object* v_inst_306_, lean_object* v_inst_307_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = lp_mathlib_Finset_add___redArg(v_inst_306_, v_inst_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_commSemigroup___redArg(lean_object* v_inst_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_Finset_mul___redArg(v_inst_309_, v_inst_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_commSemigroup(lean_object* v_00_u03b1_312_, lean_object* v_inst_313_, lean_object* v_inst_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_Finset_mul___redArg(v_inst_313_, v_inst_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addCommSemigroup___redArg(lean_object* v_inst_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Finset_add___redArg(v_inst_316_, v_inst_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addCommSemigroup(lean_object* v_00_u03b1_319_, lean_object* v_inst_320_, lean_object* v_inst_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_Finset_add___redArg(v_inst_320_, v_inst_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mulOneClass___redArg(lean_object* v_inst_323_, lean_object* v_inst_324_){
_start:
{
lean_object* v___x_325_; lean_object* v_toOne_326_; lean_object* v_toMul_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_336_; 
v___x_325_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_324_);
v_toOne_326_ = lean_ctor_get(v___x_325_, 0);
v_toMul_327_ = lean_ctor_get(v___x_325_, 1);
v_isSharedCheck_336_ = !lean_is_exclusive(v___x_325_);
if (v_isSharedCheck_336_ == 0)
{
v___x_329_ = v___x_325_;
v_isShared_330_ = v_isSharedCheck_336_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_toMul_327_);
lean_inc(v_toOne_326_);
lean_dec(v___x_325_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_336_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_334_; 
v___x_331_ = lp_mathlib_Finset_mul___redArg(v_inst_323_, v_toMul_327_);
v___x_332_ = lp_mathlib_Finset_one___redArg(v_toOne_326_);
if (v_isShared_330_ == 0)
{
lean_ctor_set(v___x_329_, 1, v___x_331_);
lean_ctor_set(v___x_329_, 0, v___x_332_);
v___x_334_ = v___x_329_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v___x_332_);
lean_ctor_set(v_reuseFailAlloc_335_, 1, v___x_331_);
v___x_334_ = v_reuseFailAlloc_335_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
return v___x_334_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mulOneClass(lean_object* v_00_u03b1_337_, lean_object* v_inst_338_, lean_object* v_inst_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_mathlib_Finset_mulOneClass___redArg(v_inst_338_, v_inst_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addZeroClass___redArg(lean_object* v_inst_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; lean_object* v_toZero_344_; lean_object* v_toAdd_345_; lean_object* v___x_347_; uint8_t v_isShared_348_; uint8_t v_isSharedCheck_354_; 
v___x_343_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_342_);
v_toZero_344_ = lean_ctor_get(v___x_343_, 0);
v_toAdd_345_ = lean_ctor_get(v___x_343_, 1);
v_isSharedCheck_354_ = !lean_is_exclusive(v___x_343_);
if (v_isSharedCheck_354_ == 0)
{
v___x_347_ = v___x_343_;
v_isShared_348_ = v_isSharedCheck_354_;
goto v_resetjp_346_;
}
else
{
lean_inc(v_toAdd_345_);
lean_inc(v_toZero_344_);
lean_dec(v___x_343_);
v___x_347_ = lean_box(0);
v_isShared_348_ = v_isSharedCheck_354_;
goto v_resetjp_346_;
}
v_resetjp_346_:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_352_; 
v___x_349_ = lp_mathlib_Finset_add___redArg(v_inst_341_, v_toAdd_345_);
v___x_350_ = lp_mathlib_Finset_zero___redArg(v_toZero_344_);
if (v_isShared_348_ == 0)
{
lean_ctor_set(v___x_347_, 1, v___x_349_);
lean_ctor_set(v___x_347_, 0, v___x_350_);
v___x_352_ = v___x_347_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v___x_350_);
lean_ctor_set(v_reuseFailAlloc_353_, 1, v___x_349_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addZeroClass(lean_object* v_00_u03b1_355_, lean_object* v_inst_356_, lean_object* v_inst_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lp_mathlib_Finset_addZeroClass___redArg(v_inst_356_, v_inst_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonMonoidHom(lean_object* v_00_u03b1_359_, lean_object* v_inst_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v___f_362_; 
v___f_362_ = ((lean_object*)(lp_mathlib_Finset_singletonOneHom___closed__0));
return v___f_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonMonoidHom___boxed(lean_object* v_00_u03b1_363_, lean_object* v_inst_364_, lean_object* v_inst_365_){
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_mathlib_Finset_singletonMonoidHom(v_00_u03b1_363_, v_inst_364_, v_inst_365_);
lean_dec_ref(v_inst_365_);
lean_dec_ref(v_inst_364_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonAddMonoidHom(lean_object* v_00_u03b1_367_, lean_object* v_inst_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v___f_370_; 
v___f_370_ = ((lean_object*)(lp_mathlib_Finset_singletonOneHom___closed__0));
return v___f_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_singletonAddMonoidHom___boxed(lean_object* v_00_u03b1_371_, lean_object* v_inst_372_, lean_object* v_inst_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_mathlib_Finset_singletonAddMonoidHom(v_00_u03b1_371_, v_inst_372_, v_inst_373_);
lean_dec_ref(v_inst_373_);
lean_dec_ref(v_inst_372_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeMonoidHom(lean_object* v_00_u03b1_375_, lean_object* v_inst_376_, lean_object* v_inst_377_){
_start:
{
lean_object* v___x_378_; 
v___x_378_ = lean_box(0);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeMonoidHom___boxed(lean_object* v_00_u03b1_379_, lean_object* v_inst_380_, lean_object* v_inst_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_Finset_coeMonoidHom(v_00_u03b1_379_, v_inst_380_, v_inst_381_);
lean_dec_ref(v_inst_381_);
lean_dec_ref(v_inst_380_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeAddMonoidHom(lean_object* v_00_u03b1_383_, lean_object* v_inst_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lean_box(0);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeAddMonoidHom___boxed(lean_object* v_00_u03b1_387_, lean_object* v_inst_388_, lean_object* v_inst_389_){
_start:
{
lean_object* v_res_390_; 
v_res_390_ = lp_mathlib_Finset_coeAddMonoidHom(v_00_u03b1_387_, v_inst_388_, v_inst_389_);
lean_dec_ref(v_inst_389_);
lean_dec_ref(v_inst_388_);
return v_res_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMonoidHom___redArg(lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_f_393_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lp_mathlib_Finset_imageMulHom___redArg(v_inst_392_, v_f_393_, v_inst_391_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMonoidHom(lean_object* v_F_395_, lean_object* v_00_u03b1_396_, lean_object* v_00_u03b2_397_, lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_f_404_){
_start:
{
lean_object* v___x_405_; 
v___x_405_ = lp_mathlib_Finset_imageMulHom___redArg(v_inst_402_, v_f_404_, v_inst_399_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageMonoidHom___boxed(lean_object* v_F_406_, lean_object* v_00_u03b1_407_, lean_object* v_00_u03b2_408_, lean_object* v_inst_409_, lean_object* v_inst_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_f_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_Finset_imageMonoidHom(v_F_406_, v_00_u03b1_407_, v_00_u03b2_408_, v_inst_409_, v_inst_410_, v_inst_411_, v_inst_412_, v_inst_413_, v_inst_414_, v_f_415_);
lean_dec_ref(v_inst_412_);
lean_dec_ref(v_inst_411_);
lean_dec_ref(v_inst_409_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddMonoidHom___redArg(lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_f_419_){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = lp_mathlib_Finset_imageAddHom___redArg(v_inst_418_, v_f_419_, v_inst_417_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddMonoidHom(lean_object* v_F_421_, lean_object* v_00_u03b1_422_, lean_object* v_00_u03b2_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_f_430_){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = lp_mathlib_Finset_imageAddHom___redArg(v_inst_428_, v_f_430_, v_inst_425_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_imageAddMonoidHom___boxed(lean_object* v_F_432_, lean_object* v_00_u03b1_433_, lean_object* v_00_u03b2_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_f_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib_Finset_imageAddMonoidHom(v_F_432_, v_00_u03b1_433_, v_00_u03b2_434_, v_inst_435_, v_inst_436_, v_inst_437_, v_inst_438_, v_inst_439_, v_inst_440_, v_f_441_);
lean_dec_ref(v_inst_438_);
lean_dec_ref(v_inst_437_);
lean_dec_ref(v_inst_435_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_monoid___redArg___lam__0(lean_object* v_inst_443_, lean_object* v_toMul_444_, lean_object* v___x_445_, lean_object* v_n_446_, lean_object* v_x_447_){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_448_ = lp_mathlib_Finset_mul___redArg(v_inst_443_, v_toMul_444_);
v___x_449_ = l_npowRec___redArg(v___x_445_, v___x_448_, v_n_446_, v_x_447_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_monoid___redArg___lam__0___boxed(lean_object* v_inst_450_, lean_object* v_toMul_451_, lean_object* v___x_452_, lean_object* v_n_453_, lean_object* v_x_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_Finset_monoid___redArg___lam__0(v_inst_450_, v_toMul_451_, v___x_452_, v_n_453_, v_x_454_);
lean_dec(v_n_453_);
lean_dec(v___x_452_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_monoid___redArg(lean_object* v_inst_456_, lean_object* v_inst_457_){
_start:
{
lean_object* v_toMul_458_; lean_object* v___x_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_472_; 
v_toMul_458_ = lean_ctor_get(v_inst_457_, 1);
lean_inc(v_toMul_458_);
v___x_459_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_457_);
v_isSharedCheck_472_ = !lean_is_exclusive(v_inst_457_);
if (v_isSharedCheck_472_ == 0)
{
lean_object* v_unused_473_; lean_object* v_unused_474_; lean_object* v_unused_475_; 
v_unused_473_ = lean_ctor_get(v_inst_457_, 2);
lean_dec(v_unused_473_);
v_unused_474_ = lean_ctor_get(v_inst_457_, 1);
lean_dec(v_unused_474_);
v_unused_475_ = lean_ctor_get(v_inst_457_, 0);
lean_dec(v_unused_475_);
v___x_461_ = v_inst_457_;
v_isShared_462_ = v_isSharedCheck_472_;
goto v_resetjp_460_;
}
else
{
lean_dec(v_inst_457_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_472_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
lean_object* v___x_463_; lean_object* v_toOne_464_; lean_object* v_toMul_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___f_468_; lean_object* v___x_470_; 
v___x_463_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_459_);
v_toOne_464_ = lean_ctor_get(v___x_463_, 0);
lean_inc(v_toOne_464_);
v_toMul_465_ = lean_ctor_get(v___x_463_, 1);
lean_inc(v_toMul_465_);
lean_dec_ref(v___x_463_);
lean_inc_ref(v_inst_456_);
v___x_466_ = lp_mathlib_Finset_mul___redArg(v_inst_456_, v_toMul_458_);
v___x_467_ = lp_mathlib_Finset_one___redArg(v_toOne_464_);
lean_inc(v___x_467_);
v___f_468_ = lean_alloc_closure((void*)(lp_mathlib_Finset_monoid___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_468_, 0, v_inst_456_);
lean_closure_set(v___f_468_, 1, v_toMul_465_);
lean_closure_set(v___f_468_, 2, v___x_467_);
if (v_isShared_462_ == 0)
{
lean_ctor_set(v___x_461_, 2, v___f_468_);
lean_ctor_set(v___x_461_, 1, v___x_466_);
lean_ctor_set(v___x_461_, 0, v___x_467_);
v___x_470_ = v___x_461_;
goto v_reusejp_469_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v___x_467_);
lean_ctor_set(v_reuseFailAlloc_471_, 1, v___x_466_);
lean_ctor_set(v_reuseFailAlloc_471_, 2, v___f_468_);
v___x_470_ = v_reuseFailAlloc_471_;
goto v_reusejp_469_;
}
v_reusejp_469_:
{
return v___x_470_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_monoid(lean_object* v_00_u03b1_476_, lean_object* v_inst_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_mathlib_Finset_monoid___redArg(v_inst_477_, v_inst_478_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addMonoid___redArg(lean_object* v_inst_480_, lean_object* v_inst_481_){
_start:
{
lean_object* v_toAdd_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v_toZero_485_; lean_object* v_toAdd_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; 
v_toAdd_482_ = lean_ctor_get(v_inst_481_, 1);
lean_inc(v_toAdd_482_);
v___x_483_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_481_);
lean_dec_ref(v_inst_481_);
v___x_484_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_483_);
v_toZero_485_ = lean_ctor_get(v___x_484_, 0);
lean_inc_n(v_toZero_485_, 2);
v_toAdd_486_ = lean_ctor_get(v___x_484_, 1);
lean_inc(v_toAdd_486_);
lean_dec_ref(v___x_484_);
lean_inc_ref(v_inst_480_);
v___x_487_ = lp_mathlib_Finset_add___redArg(v_inst_480_, v_toAdd_482_);
v___x_488_ = lp_mathlib_Finset_zero___redArg(v_toZero_485_);
v___x_489_ = lp_mathlib_Finset_nsmul___redArg(v_inst_480_, v_toZero_485_, v_toAdd_486_);
v___x_490_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___x_487_, v___x_488_, v___x_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addMonoid(lean_object* v_00_u03b1_491_, lean_object* v_inst_492_, lean_object* v_inst_493_){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lp_mathlib_Finset_addMonoid___redArg(v_inst_492_, v_inst_493_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_commMonoid___redArg(lean_object* v_inst_495_, lean_object* v_inst_496_){
_start:
{
lean_object* v_toMul_497_; lean_object* v___x_498_; lean_object* v___x_500_; uint8_t v_isShared_501_; uint8_t v_isSharedCheck_511_; 
v_toMul_497_ = lean_ctor_get(v_inst_496_, 1);
lean_inc(v_toMul_497_);
v___x_498_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_496_);
v_isSharedCheck_511_ = !lean_is_exclusive(v_inst_496_);
if (v_isSharedCheck_511_ == 0)
{
lean_object* v_unused_512_; lean_object* v_unused_513_; lean_object* v_unused_514_; 
v_unused_512_ = lean_ctor_get(v_inst_496_, 2);
lean_dec(v_unused_512_);
v_unused_513_ = lean_ctor_get(v_inst_496_, 1);
lean_dec(v_unused_513_);
v_unused_514_ = lean_ctor_get(v_inst_496_, 0);
lean_dec(v_unused_514_);
v___x_500_ = v_inst_496_;
v_isShared_501_ = v_isSharedCheck_511_;
goto v_resetjp_499_;
}
else
{
lean_dec(v_inst_496_);
v___x_500_ = lean_box(0);
v_isShared_501_ = v_isSharedCheck_511_;
goto v_resetjp_499_;
}
v_resetjp_499_:
{
lean_object* v___x_502_; lean_object* v_toOne_503_; lean_object* v_toMul_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___f_507_; lean_object* v___x_509_; 
v___x_502_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_498_);
v_toOne_503_ = lean_ctor_get(v___x_502_, 0);
lean_inc(v_toOne_503_);
v_toMul_504_ = lean_ctor_get(v___x_502_, 1);
lean_inc(v_toMul_504_);
lean_dec_ref(v___x_502_);
lean_inc_ref(v_inst_495_);
v___x_505_ = lp_mathlib_Finset_mul___redArg(v_inst_495_, v_toMul_497_);
v___x_506_ = lp_mathlib_Finset_one___redArg(v_toOne_503_);
lean_inc(v___x_506_);
v___f_507_ = lean_alloc_closure((void*)(lp_mathlib_Finset_monoid___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_507_, 0, v_inst_495_);
lean_closure_set(v___f_507_, 1, v_toMul_504_);
lean_closure_set(v___f_507_, 2, v___x_506_);
if (v_isShared_501_ == 0)
{
lean_ctor_set(v___x_500_, 2, v___f_507_);
lean_ctor_set(v___x_500_, 1, v___x_505_);
lean_ctor_set(v___x_500_, 0, v___x_506_);
v___x_509_ = v___x_500_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v___x_506_);
lean_ctor_set(v_reuseFailAlloc_510_, 1, v___x_505_);
lean_ctor_set(v_reuseFailAlloc_510_, 2, v___f_507_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_commMonoid(lean_object* v_00_u03b1_515_, lean_object* v_inst_516_, lean_object* v_inst_517_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_mathlib_Finset_commMonoid___redArg(v_inst_516_, v_inst_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addCommMonoid___redArg(lean_object* v_inst_519_, lean_object* v_inst_520_){
_start:
{
lean_object* v_toAdd_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v_toZero_524_; lean_object* v_toAdd_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; 
v_toAdd_521_ = lean_ctor_get(v_inst_520_, 1);
lean_inc(v_toAdd_521_);
v___x_522_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_520_);
lean_dec_ref(v_inst_520_);
v___x_523_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_522_);
v_toZero_524_ = lean_ctor_get(v___x_523_, 0);
lean_inc_n(v_toZero_524_, 2);
v_toAdd_525_ = lean_ctor_get(v___x_523_, 1);
lean_inc(v_toAdd_525_);
lean_dec_ref(v___x_523_);
lean_inc_ref(v_inst_519_);
v___x_526_ = lp_mathlib_Finset_add___redArg(v_inst_519_, v_toAdd_521_);
v___x_527_ = lp_mathlib_Finset_zero___redArg(v_toZero_524_);
v___x_528_ = lp_mathlib_Finset_nsmul___redArg(v_inst_519_, v_toZero_524_, v_toAdd_525_);
v___x_529_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___x_526_, v___x_527_, v___x_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_addCommMonoid(lean_object* v_00_u03b1_530_, lean_object* v_inst_531_, lean_object* v_inst_532_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_mathlib_Finset_addCommMonoid___redArg(v_inst_531_, v_inst_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionMonoid___redArg___lam__1(lean_object* v_toOne_534_, lean_object* v_inst_535_, lean_object* v_toMul_536_, lean_object* v_toInv_537_, lean_object* v_n_538_, lean_object* v_x_539_){
_start:
{
lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_540_ = lp_mathlib_Finset_one___redArg(v_toOne_534_);
lean_inc_ref(v_inst_535_);
v___x_541_ = lp_mathlib_Finset_mul___redArg(v_inst_535_, v_toMul_536_);
v___x_542_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_542_, 0, lean_box(0));
lean_closure_set(v___x_542_, 1, lean_box(0));
lean_closure_set(v___x_542_, 2, v_inst_535_);
lean_closure_set(v___x_542_, 3, v_toInv_537_);
v___x_543_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_543_, 0, lean_box(0));
lean_closure_set(v___x_543_, 1, v___x_540_);
lean_closure_set(v___x_543_, 2, v___x_541_);
v___x_544_ = lp_mathlib_zpowRec___redArg(v___x_542_, v___x_543_, v_n_538_, v_x_539_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionMonoid___redArg___lam__1___boxed(lean_object* v_toOne_545_, lean_object* v_inst_546_, lean_object* v_toMul_547_, lean_object* v_toInv_548_, lean_object* v_n_549_, lean_object* v_x_550_){
_start:
{
lean_object* v_res_551_; 
v_res_551_ = lp_mathlib_Finset_divisionMonoid___redArg___lam__1(v_toOne_545_, v_inst_546_, v_toMul_547_, v_toInv_548_, v_n_549_, v_x_550_);
lean_dec(v_n_549_);
return v_res_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionMonoid___redArg(lean_object* v_inst_552_, lean_object* v_inst_553_){
_start:
{
lean_object* v_toMonoid_554_; lean_object* v_toInv_555_; lean_object* v_toDiv_556_; lean_object* v_toMul_557_; lean_object* v___x_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_588_; 
v_toMonoid_554_ = lean_ctor_get(v_inst_553_, 0);
lean_inc_ref(v_toMonoid_554_);
v_toInv_555_ = lean_ctor_get(v_inst_553_, 1);
lean_inc(v_toInv_555_);
v_toDiv_556_ = lean_ctor_get(v_inst_553_, 2);
lean_inc(v_toDiv_556_);
v_toMul_557_ = lean_ctor_get(v_toMonoid_554_, 1);
lean_inc(v_toMul_557_);
v___x_558_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_554_);
v_isSharedCheck_588_ = !lean_is_exclusive(v_toMonoid_554_);
if (v_isSharedCheck_588_ == 0)
{
lean_object* v_unused_589_; lean_object* v_unused_590_; lean_object* v_unused_591_; 
v_unused_589_ = lean_ctor_get(v_toMonoid_554_, 2);
lean_dec(v_unused_589_);
v_unused_590_ = lean_ctor_get(v_toMonoid_554_, 1);
lean_dec(v_unused_590_);
v_unused_591_ = lean_ctor_get(v_toMonoid_554_, 0);
lean_dec(v_unused_591_);
v___x_560_ = v_toMonoid_554_;
v_isShared_561_ = v_isSharedCheck_588_;
goto v_resetjp_559_;
}
else
{
lean_dec(v_toMonoid_554_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_588_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
lean_object* v___x_562_; lean_object* v_toOne_563_; lean_object* v_toMul_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_583_; 
v___x_562_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_558_);
v_toOne_563_ = lean_ctor_get(v___x_562_, 0);
lean_inc(v_toOne_563_);
v_toMul_564_ = lean_ctor_get(v___x_562_, 1);
lean_inc(v_toMul_564_);
lean_dec_ref(v___x_562_);
lean_inc_ref(v_inst_552_);
v___x_565_ = lp_mathlib_Finset_mul___redArg(v_inst_552_, v_toMul_557_);
v___x_566_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_553_);
v_isSharedCheck_583_ = !lean_is_exclusive(v_inst_553_);
if (v_isSharedCheck_583_ == 0)
{
lean_object* v_unused_584_; lean_object* v_unused_585_; lean_object* v_unused_586_; lean_object* v_unused_587_; 
v_unused_584_ = lean_ctor_get(v_inst_553_, 3);
lean_dec(v_unused_584_);
v_unused_585_ = lean_ctor_get(v_inst_553_, 2);
lean_dec(v_unused_585_);
v_unused_586_ = lean_ctor_get(v_inst_553_, 1);
lean_dec(v_unused_586_);
v_unused_587_ = lean_ctor_get(v_inst_553_, 0);
lean_dec(v_unused_587_);
v___x_568_ = v_inst_553_;
v_isShared_569_ = v_isSharedCheck_583_;
goto v_resetjp_567_;
}
else
{
lean_dec(v_inst_553_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_583_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v_toOne_570_; lean_object* v_toInv_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___f_574_; lean_object* v___x_575_; lean_object* v___f_576_; lean_object* v___x_578_; 
v_toOne_570_ = lean_ctor_get(v___x_566_, 0);
lean_inc(v_toOne_570_);
v_toInv_571_ = lean_ctor_get(v___x_566_, 1);
lean_inc(v_toInv_571_);
lean_dec_ref(v___x_566_);
lean_inc_ref_n(v_inst_552_, 3);
v___x_572_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_572_, 0, lean_box(0));
lean_closure_set(v___x_572_, 1, lean_box(0));
lean_closure_set(v___x_572_, 2, v_inst_552_);
lean_closure_set(v___x_572_, 3, v_toInv_555_);
v___x_573_ = lp_mathlib_Finset_one___redArg(v_toOne_563_);
lean_inc(v___x_573_);
lean_inc(v_toMul_564_);
v___f_574_ = lean_alloc_closure((void*)(lp_mathlib_Finset_monoid___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_574_, 0, v_inst_552_);
lean_closure_set(v___f_574_, 1, v_toMul_564_);
lean_closure_set(v___f_574_, 2, v___x_573_);
v___x_575_ = lp_mathlib_Finset_div___redArg(v_inst_552_, v_toDiv_556_);
v___f_576_ = lean_alloc_closure((void*)(lp_mathlib_Finset_divisionMonoid___redArg___lam__1___boxed), 6, 4);
lean_closure_set(v___f_576_, 0, v_toOne_570_);
lean_closure_set(v___f_576_, 1, v_inst_552_);
lean_closure_set(v___f_576_, 2, v_toMul_564_);
lean_closure_set(v___f_576_, 3, v_toInv_571_);
if (v_isShared_561_ == 0)
{
lean_ctor_set(v___x_560_, 2, v___f_574_);
lean_ctor_set(v___x_560_, 1, v___x_565_);
lean_ctor_set(v___x_560_, 0, v___x_573_);
v___x_578_ = v___x_560_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v___x_573_);
lean_ctor_set(v_reuseFailAlloc_582_, 1, v___x_565_);
lean_ctor_set(v_reuseFailAlloc_582_, 2, v___f_574_);
v___x_578_ = v_reuseFailAlloc_582_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
lean_object* v___x_580_; 
if (v_isShared_569_ == 0)
{
lean_ctor_set(v___x_568_, 3, v___f_576_);
lean_ctor_set(v___x_568_, 2, v___x_575_);
lean_ctor_set(v___x_568_, 1, v___x_572_);
lean_ctor_set(v___x_568_, 0, v___x_578_);
v___x_580_ = v___x_568_;
goto v_reusejp_579_;
}
else
{
lean_object* v_reuseFailAlloc_581_; 
v_reuseFailAlloc_581_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_581_, 0, v___x_578_);
lean_ctor_set(v_reuseFailAlloc_581_, 1, v___x_572_);
lean_ctor_set(v_reuseFailAlloc_581_, 2, v___x_575_);
lean_ctor_set(v_reuseFailAlloc_581_, 3, v___f_576_);
v___x_580_ = v_reuseFailAlloc_581_;
goto v_reusejp_579_;
}
v_reusejp_579_:
{
return v___x_580_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionMonoid(lean_object* v_00_u03b1_592_, lean_object* v_inst_593_, lean_object* v_inst_594_){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lp_mathlib_Finset_divisionMonoid___redArg(v_inst_593_, v_inst_594_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtractionMonoid___redArg(lean_object* v_inst_596_, lean_object* v_inst_597_){
_start:
{
lean_object* v_toAddMonoid_598_; lean_object* v_toNeg_599_; lean_object* v_toSub_600_; lean_object* v_toAdd_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v_toZero_604_; lean_object* v_toAdd_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v_toZero_610_; lean_object* v_toNeg_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; 
v_toAddMonoid_598_ = lean_ctor_get(v_inst_597_, 0);
v_toNeg_599_ = lean_ctor_get(v_inst_597_, 1);
lean_inc(v_toNeg_599_);
v_toSub_600_ = lean_ctor_get(v_inst_597_, 2);
lean_inc(v_toSub_600_);
v_toAdd_601_ = lean_ctor_get(v_toAddMonoid_598_, 1);
v___x_602_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_598_);
v___x_603_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_602_);
v_toZero_604_ = lean_ctor_get(v___x_603_, 0);
lean_inc_n(v_toZero_604_, 2);
v_toAdd_605_ = lean_ctor_get(v___x_603_, 1);
lean_inc_n(v_toAdd_605_, 2);
lean_dec_ref(v___x_603_);
lean_inc(v_toAdd_601_);
lean_inc_ref_n(v_inst_596_, 4);
v___x_606_ = lp_mathlib_Finset_add___redArg(v_inst_596_, v_toAdd_601_);
v___x_607_ = lp_mathlib_Finset_zero___redArg(v_toZero_604_);
v___x_608_ = lp_mathlib_Finset_nsmul___redArg(v_inst_596_, v_toZero_604_, v_toAdd_605_);
v___x_609_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_597_);
lean_dec_ref(v_inst_597_);
v_toZero_610_ = lean_ctor_get(v___x_609_, 0);
lean_inc(v_toZero_610_);
v_toNeg_611_ = lean_ctor_get(v___x_609_, 1);
lean_inc(v_toNeg_611_);
lean_dec_ref(v___x_609_);
v___x_612_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_612_, 0, lean_box(0));
lean_closure_set(v___x_612_, 1, lean_box(0));
lean_closure_set(v___x_612_, 2, v_inst_596_);
lean_closure_set(v___x_612_, 3, v_toNeg_599_);
v___x_613_ = lp_mathlib_Finset_sub___redArg(v_inst_596_, v_toSub_600_);
v___x_614_ = lp_mathlib_Finset_zsmul___redArg(v_inst_596_, v_toZero_610_, v_toAdd_605_, v_toNeg_611_);
v___x_615_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___x_606_, v___x_607_, v___x_608_, v___x_612_, v___x_613_, v___x_614_);
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtractionMonoid(lean_object* v_00_u03b1_616_, lean_object* v_inst_617_, lean_object* v_inst_618_){
_start:
{
lean_object* v___x_619_; 
v___x_619_ = lp_mathlib_Finset_subtractionMonoid___redArg(v_inst_617_, v_inst_618_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionCommMonoid___redArg(lean_object* v_inst_620_, lean_object* v_inst_621_){
_start:
{
lean_object* v_toMonoid_622_; lean_object* v_toInv_623_; lean_object* v_toDiv_624_; lean_object* v_toMul_625_; lean_object* v___x_626_; lean_object* v___x_628_; uint8_t v_isShared_629_; uint8_t v_isSharedCheck_656_; 
v_toMonoid_622_ = lean_ctor_get(v_inst_621_, 0);
lean_inc_ref(v_toMonoid_622_);
v_toInv_623_ = lean_ctor_get(v_inst_621_, 1);
lean_inc(v_toInv_623_);
v_toDiv_624_ = lean_ctor_get(v_inst_621_, 2);
lean_inc(v_toDiv_624_);
v_toMul_625_ = lean_ctor_get(v_toMonoid_622_, 1);
lean_inc(v_toMul_625_);
v___x_626_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_622_);
v_isSharedCheck_656_ = !lean_is_exclusive(v_toMonoid_622_);
if (v_isSharedCheck_656_ == 0)
{
lean_object* v_unused_657_; lean_object* v_unused_658_; lean_object* v_unused_659_; 
v_unused_657_ = lean_ctor_get(v_toMonoid_622_, 2);
lean_dec(v_unused_657_);
v_unused_658_ = lean_ctor_get(v_toMonoid_622_, 1);
lean_dec(v_unused_658_);
v_unused_659_ = lean_ctor_get(v_toMonoid_622_, 0);
lean_dec(v_unused_659_);
v___x_628_ = v_toMonoid_622_;
v_isShared_629_ = v_isSharedCheck_656_;
goto v_resetjp_627_;
}
else
{
lean_dec(v_toMonoid_622_);
v___x_628_ = lean_box(0);
v_isShared_629_ = v_isSharedCheck_656_;
goto v_resetjp_627_;
}
v_resetjp_627_:
{
lean_object* v___x_630_; lean_object* v_toOne_631_; lean_object* v_toMul_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_636_; uint8_t v_isShared_637_; uint8_t v_isSharedCheck_651_; 
v___x_630_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_626_);
v_toOne_631_ = lean_ctor_get(v___x_630_, 0);
lean_inc(v_toOne_631_);
v_toMul_632_ = lean_ctor_get(v___x_630_, 1);
lean_inc(v_toMul_632_);
lean_dec_ref(v___x_630_);
lean_inc_ref(v_inst_620_);
v___x_633_ = lp_mathlib_Finset_mul___redArg(v_inst_620_, v_toMul_625_);
v___x_634_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_621_);
v_isSharedCheck_651_ = !lean_is_exclusive(v_inst_621_);
if (v_isSharedCheck_651_ == 0)
{
lean_object* v_unused_652_; lean_object* v_unused_653_; lean_object* v_unused_654_; lean_object* v_unused_655_; 
v_unused_652_ = lean_ctor_get(v_inst_621_, 3);
lean_dec(v_unused_652_);
v_unused_653_ = lean_ctor_get(v_inst_621_, 2);
lean_dec(v_unused_653_);
v_unused_654_ = lean_ctor_get(v_inst_621_, 1);
lean_dec(v_unused_654_);
v_unused_655_ = lean_ctor_get(v_inst_621_, 0);
lean_dec(v_unused_655_);
v___x_636_ = v_inst_621_;
v_isShared_637_ = v_isSharedCheck_651_;
goto v_resetjp_635_;
}
else
{
lean_dec(v_inst_621_);
v___x_636_ = lean_box(0);
v_isShared_637_ = v_isSharedCheck_651_;
goto v_resetjp_635_;
}
v_resetjp_635_:
{
lean_object* v_toOne_638_; lean_object* v_toInv_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___f_642_; lean_object* v___x_643_; lean_object* v___f_644_; lean_object* v___x_646_; 
v_toOne_638_ = lean_ctor_get(v___x_634_, 0);
lean_inc(v_toOne_638_);
v_toInv_639_ = lean_ctor_get(v___x_634_, 1);
lean_inc(v_toInv_639_);
lean_dec_ref(v___x_634_);
lean_inc_ref_n(v_inst_620_, 3);
v___x_640_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_640_, 0, lean_box(0));
lean_closure_set(v___x_640_, 1, lean_box(0));
lean_closure_set(v___x_640_, 2, v_inst_620_);
lean_closure_set(v___x_640_, 3, v_toInv_623_);
v___x_641_ = lp_mathlib_Finset_one___redArg(v_toOne_631_);
lean_inc(v___x_641_);
lean_inc(v_toMul_632_);
v___f_642_ = lean_alloc_closure((void*)(lp_mathlib_Finset_monoid___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_642_, 0, v_inst_620_);
lean_closure_set(v___f_642_, 1, v_toMul_632_);
lean_closure_set(v___f_642_, 2, v___x_641_);
v___x_643_ = lp_mathlib_Finset_div___redArg(v_inst_620_, v_toDiv_624_);
v___f_644_ = lean_alloc_closure((void*)(lp_mathlib_Finset_divisionMonoid___redArg___lam__1___boxed), 6, 4);
lean_closure_set(v___f_644_, 0, v_toOne_638_);
lean_closure_set(v___f_644_, 1, v_inst_620_);
lean_closure_set(v___f_644_, 2, v_toMul_632_);
lean_closure_set(v___f_644_, 3, v_toInv_639_);
if (v_isShared_629_ == 0)
{
lean_ctor_set(v___x_628_, 2, v___f_642_);
lean_ctor_set(v___x_628_, 1, v___x_633_);
lean_ctor_set(v___x_628_, 0, v___x_641_);
v___x_646_ = v___x_628_;
goto v_reusejp_645_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v___x_641_);
lean_ctor_set(v_reuseFailAlloc_650_, 1, v___x_633_);
lean_ctor_set(v_reuseFailAlloc_650_, 2, v___f_642_);
v___x_646_ = v_reuseFailAlloc_650_;
goto v_reusejp_645_;
}
v_reusejp_645_:
{
lean_object* v___x_648_; 
if (v_isShared_637_ == 0)
{
lean_ctor_set(v___x_636_, 3, v___f_644_);
lean_ctor_set(v___x_636_, 2, v___x_643_);
lean_ctor_set(v___x_636_, 1, v___x_640_);
lean_ctor_set(v___x_636_, 0, v___x_646_);
v___x_648_ = v___x_636_;
goto v_reusejp_647_;
}
else
{
lean_object* v_reuseFailAlloc_649_; 
v_reuseFailAlloc_649_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_649_, 0, v___x_646_);
lean_ctor_set(v_reuseFailAlloc_649_, 1, v___x_640_);
lean_ctor_set(v_reuseFailAlloc_649_, 2, v___x_643_);
lean_ctor_set(v_reuseFailAlloc_649_, 3, v___f_644_);
v___x_648_ = v_reuseFailAlloc_649_;
goto v_reusejp_647_;
}
v_reusejp_647_:
{
return v___x_648_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_divisionCommMonoid(lean_object* v_00_u03b1_660_, lean_object* v_inst_661_, lean_object* v_inst_662_){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = lp_mathlib_Finset_divisionCommMonoid___redArg(v_inst_661_, v_inst_662_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtractionCommMonoid___redArg(lean_object* v_inst_664_, lean_object* v_inst_665_){
_start:
{
lean_object* v_toAddMonoid_666_; lean_object* v_toNeg_667_; lean_object* v_toSub_668_; lean_object* v_toAdd_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v_toZero_672_; lean_object* v_toAdd_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v_toZero_678_; lean_object* v_toNeg_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; 
v_toAddMonoid_666_ = lean_ctor_get(v_inst_665_, 0);
v_toNeg_667_ = lean_ctor_get(v_inst_665_, 1);
lean_inc(v_toNeg_667_);
v_toSub_668_ = lean_ctor_get(v_inst_665_, 2);
lean_inc(v_toSub_668_);
v_toAdd_669_ = lean_ctor_get(v_toAddMonoid_666_, 1);
v___x_670_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_666_);
v___x_671_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_670_);
v_toZero_672_ = lean_ctor_get(v___x_671_, 0);
lean_inc_n(v_toZero_672_, 2);
v_toAdd_673_ = lean_ctor_get(v___x_671_, 1);
lean_inc_n(v_toAdd_673_, 2);
lean_dec_ref(v___x_671_);
lean_inc(v_toAdd_669_);
lean_inc_ref_n(v_inst_664_, 4);
v___x_674_ = lp_mathlib_Finset_add___redArg(v_inst_664_, v_toAdd_669_);
v___x_675_ = lp_mathlib_Finset_zero___redArg(v_toZero_672_);
v___x_676_ = lp_mathlib_Finset_nsmul___redArg(v_inst_664_, v_toZero_672_, v_toAdd_673_);
v___x_677_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_665_);
lean_dec_ref(v_inst_665_);
v_toZero_678_ = lean_ctor_get(v___x_677_, 0);
lean_inc(v_toZero_678_);
v_toNeg_679_ = lean_ctor_get(v___x_677_, 1);
lean_inc(v_toNeg_679_);
lean_dec_ref(v___x_677_);
v___x_680_ = lean_alloc_closure((void*)(lp_mathlib_Finset_image), 5, 4);
lean_closure_set(v___x_680_, 0, lean_box(0));
lean_closure_set(v___x_680_, 1, lean_box(0));
lean_closure_set(v___x_680_, 2, v_inst_664_);
lean_closure_set(v___x_680_, 3, v_toNeg_667_);
v___x_681_ = lp_mathlib_Finset_sub___redArg(v_inst_664_, v_toSub_668_);
v___x_682_ = lp_mathlib_Finset_zsmul___redArg(v_inst_664_, v_toZero_678_, v_toAdd_673_, v_toNeg_679_);
v___x_683_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___x_674_, v___x_675_, v___x_676_, v___x_680_, v___x_681_, v___x_682_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtractionCommMonoid(lean_object* v_00_u03b1_684_, lean_object* v_inst_685_, lean_object* v_inst_686_){
_start:
{
lean_object* v___x_687_; 
v___x_687_ = lp_mathlib_Finset_subtractionCommMonoid___redArg(v_inst_685_, v_inst_686_);
return v___x_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeOne___redArg(lean_object* v_inst_688_){
_start:
{
lean_object* v___x_689_; 
v___x_689_ = lp_mathlib_Set_fintypeSingleton___redArg(v_inst_688_);
return v___x_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeOne(lean_object* v_00_u03b1_690_, lean_object* v_inst_691_){
_start:
{
lean_object* v___x_692_; 
v___x_692_ = lp_mathlib_Set_fintypeSingleton___redArg(v_inst_691_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeZero___redArg(lean_object* v_inst_693_){
_start:
{
lean_object* v___x_694_; 
v___x_694_ = lp_mathlib_Set_fintypeSingleton___redArg(v_inst_693_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeZero(lean_object* v_00_u03b1_695_, lean_object* v_inst_696_){
_start:
{
lean_object* v___x_697_; 
v___x_697_ = lp_mathlib_Set_fintypeSingleton___redArg(v_inst_696_);
return v___x_697_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_ListOfFn(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_NAry(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Preimage(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_ListOfFn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_ListOfFn(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_NAry(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Preimage(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_ListOfFn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Finset_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
