// Lean compiler output
// Module: Mathlib.Algebra.Group.ULift
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.InjSurj
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
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_div___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_sub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_neg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_vadd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_vadd(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_ulift___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_ulift___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_semigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_subNegAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_subNegAddMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_group___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_group(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_ULift_one___redArg(v_inst_2_);
lean_dec(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_one(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_inc(v_inst_5_);
return v_inst_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_one___boxed(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_ULift_one(v_00_u03b1_6_, v_inst_7_);
lean_dec(v_inst_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___redArg(lean_object* v_inst_9_){
_start:
{
lean_inc(v_inst_9_);
return v_inst_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___redArg___boxed(lean_object* v_inst_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_ULift_zero___redArg(v_inst_10_);
lean_dec(v_inst_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_){
_start:
{
lean_inc(v_inst_13_);
return v_inst_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_zero___boxed(lean_object* v_00_u03b1_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_ULift_zero(v_00_u03b1_14_, v_inst_15_);
lean_dec(v_inst_15_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul___redArg___lam__0(lean_object* v_inst_17_, lean_object* v_f_18_, lean_object* v_g_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_apply_2(v_inst_17_, v_f_18_, v_g_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul___redArg(lean_object* v_inst_21_){
_start:
{
lean_object* v___f_22_; 
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_22_, 0, v_inst_21_);
return v___f_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mul(lean_object* v_00_u03b1_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_25_, 0, v_inst_24_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_add___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v___f_27_; 
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_27_, 0, v_inst_26_);
return v___f_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_add(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___f_30_; 
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_30_, 0, v_inst_29_);
return v___f_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_div___redArg(lean_object* v_inst_31_){
_start:
{
lean_object* v___f_32_; 
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_32_, 0, v_inst_31_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_div(lean_object* v_00_u03b1_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_35_, 0, v_inst_34_);
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_sub___redArg(lean_object* v_inst_36_){
_start:
{
lean_object* v___f_37_; 
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_37_, 0, v_inst_36_);
return v___f_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_sub(lean_object* v_00_u03b1_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___f_40_; 
v___f_40_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_40_, 0, v_inst_39_);
return v___f_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv___redArg___lam__0(lean_object* v_inst_41_, lean_object* v_f_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lean_apply_1(v_inst_41_, v_f_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv___redArg(lean_object* v_inst_44_){
_start:
{
lean_object* v___f_45_; 
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_45_, 0, v_inst_44_);
return v___f_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_inv(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___f_48_; 
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_48_, 0, v_inst_47_);
return v___f_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_neg___redArg(lean_object* v_inst_49_){
_start:
{
lean_object* v___f_50_; 
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_50_, 0, v_inst_49_);
return v___f_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_neg(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___f_53_; 
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_53_, 0, v_inst_52_);
return v___f_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow___redArg___lam__0(lean_object* v_inst_54_, lean_object* v_x_55_, lean_object* v_n_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lean_apply_2(v_inst_54_, v_x_55_, v_n_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow___redArg(lean_object* v_inst_58_){
_start:
{
lean_object* v___f_59_; 
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_ULift_pow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_59_, 0, v_inst_58_);
return v___f_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_pow(lean_object* v_00_u03b1_60_, lean_object* v_00_u03b2_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v___f_63_; 
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_ULift_pow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_63_, 0, v_inst_62_);
return v___f_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul___redArg___lam__0(lean_object* v_inst_64_, lean_object* v_n_65_, lean_object* v_x_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lean_apply_2(v_inst_64_, v_n_65_, v_x_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul___redArg(lean_object* v_inst_68_){
_start:
{
lean_object* v___f_69_; 
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_69_, 0, v_inst_68_);
return v___f_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smul(lean_object* v_00_u03b1_70_, lean_object* v_00_u03b2_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v___f_73_; 
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_73_, 0, v_inst_72_);
return v___f_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_vadd___redArg(lean_object* v_inst_74_){
_start:
{
lean_object* v___f_75_; 
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_75_, 0, v_inst_74_);
return v___f_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_vadd(lean_object* v_00_u03b1_76_, lean_object* v_00_u03b2_77_, lean_object* v_inst_78_){
_start:
{
lean_object* v___f_79_; 
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_79_, 0, v_inst_78_);
return v___f_79_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_ulift___closed__0(void){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lp_mathlib_Equiv_ulift(lean_box(0));
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift(lean_object* v_00_u03b1_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_obj_once(&lp_mathlib_MulEquiv_ulift___closed__0, &lp_mathlib_MulEquiv_ulift___closed__0_once, _init_lp_mathlib_MulEquiv_ulift___closed__0);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___boxed(lean_object* v_00_u03b1_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_MulEquiv_ulift(v_00_u03b1_84_, v_inst_85_);
lean_dec(v_inst_85_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift(lean_object* v_00_u03b1_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_obj_once(&lp_mathlib_MulEquiv_ulift___closed__0, &lp_mathlib_MulEquiv_ulift___closed__0_once, _init_lp_mathlib_MulEquiv_ulift___closed__0);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___boxed(lean_object* v_00_u03b1_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_AddEquiv_ulift(v_00_u03b1_90_, v_inst_91_);
lean_dec(v_inst_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semigroup___redArg(lean_object* v_inst_93_){
_start:
{
lean_object* v___f_94_; 
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_94_, 0, v_inst_93_);
return v___f_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_semigroup(lean_object* v_00_u03b1_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___f_97_; 
v___f_97_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_97_, 0, v_inst_96_);
return v___f_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addSemigroup___redArg(lean_object* v_inst_98_){
_start:
{
lean_object* v___f_99_; 
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_99_, 0, v_inst_98_);
return v___f_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addSemigroup(lean_object* v_00_u03b1_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v___f_102_; 
v___f_102_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_102_, 0, v_inst_101_);
return v___f_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemigroup___redArg(lean_object* v_inst_103_){
_start:
{
lean_object* v___f_104_; 
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_104_, 0, v_inst_103_);
return v___f_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commSemigroup(lean_object* v_00_u03b1_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___f_107_; 
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_107_, 0, v_inst_106_);
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommSemigroup___redArg(lean_object* v_inst_108_){
_start:
{
lean_object* v___f_109_; 
v___f_109_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_109_, 0, v_inst_108_);
return v___f_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommSemigroup(lean_object* v_00_u03b1_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___f_112_; 
v___f_112_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_112_, 0, v_inst_111_);
return v___f_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulOneClass___redArg(lean_object* v_inst_113_){
_start:
{
lean_object* v___x_114_; lean_object* v_toOne_115_; lean_object* v_toMul_116_; lean_object* v___x_118_; uint8_t v_isShared_119_; uint8_t v_isSharedCheck_124_; 
v___x_114_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_113_);
v_toOne_115_ = lean_ctor_get(v___x_114_, 0);
v_toMul_116_ = lean_ctor_get(v___x_114_, 1);
v_isSharedCheck_124_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_124_ == 0)
{
v___x_118_ = v___x_114_;
v_isShared_119_ = v_isSharedCheck_124_;
goto v_resetjp_117_;
}
else
{
lean_inc(v_toMul_116_);
lean_inc(v_toOne_115_);
lean_dec(v___x_114_);
v___x_118_ = lean_box(0);
v_isShared_119_ = v_isSharedCheck_124_;
goto v_resetjp_117_;
}
v_resetjp_117_:
{
lean_object* v___f_120_; lean_object* v___x_122_; 
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_120_, 0, v_toMul_116_);
if (v_isShared_119_ == 0)
{
lean_ctor_set(v___x_118_, 1, v___f_120_);
v___x_122_ = v___x_118_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v_toOne_115_);
lean_ctor_set(v_reuseFailAlloc_123_, 1, v___f_120_);
v___x_122_ = v_reuseFailAlloc_123_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
return v___x_122_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulOneClass(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_ULift_mulOneClass___redArg(v_inst_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addZeroClass___redArg(lean_object* v_inst_128_){
_start:
{
lean_object* v___x_129_; lean_object* v_toZero_130_; lean_object* v_toAdd_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_139_; 
v___x_129_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_128_);
v_toZero_130_ = lean_ctor_get(v___x_129_, 0);
v_toAdd_131_ = lean_ctor_get(v___x_129_, 1);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_139_ == 0)
{
v___x_133_ = v___x_129_;
v_isShared_134_ = v_isSharedCheck_139_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_toAdd_131_);
lean_inc(v_toZero_130_);
lean_dec(v___x_129_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_139_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___f_135_; lean_object* v___x_137_; 
v___f_135_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_135_, 0, v_toAdd_131_);
if (v_isShared_134_ == 0)
{
lean_ctor_set(v___x_133_, 1, v___f_135_);
v___x_137_ = v___x_133_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v_toZero_130_);
lean_ctor_set(v_reuseFailAlloc_138_, 1, v___f_135_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addZeroClass(lean_object* v_00_u03b1_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_ULift_addZeroClass___redArg(v_inst_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid___redArg___lam__0(lean_object* v_toNPow_143_, lean_object* v_n_144_, lean_object* v_x_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lean_apply_2(v_toNPow_143_, v_n_144_, v_x_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid___redArg(lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v_toOne_150_; lean_object* v_toMul_151_; lean_object* v_toNPow_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_161_; 
v___x_148_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_147_);
v___x_149_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_148_);
v_toOne_150_ = lean_ctor_get(v___x_149_, 0);
lean_inc(v_toOne_150_);
v_toMul_151_ = lean_ctor_get(v___x_149_, 1);
lean_inc(v_toMul_151_);
lean_dec_ref(v___x_149_);
v_toNPow_152_ = lean_ctor_get(v_inst_147_, 2);
v_isSharedCheck_161_ = !lean_is_exclusive(v_inst_147_);
if (v_isSharedCheck_161_ == 0)
{
lean_object* v_unused_162_; lean_object* v_unused_163_; 
v_unused_162_ = lean_ctor_get(v_inst_147_, 1);
lean_dec(v_unused_162_);
v_unused_163_ = lean_ctor_get(v_inst_147_, 0);
lean_dec(v_unused_163_);
v___x_154_ = v_inst_147_;
v_isShared_155_ = v_isSharedCheck_161_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_toNPow_152_);
lean_dec(v_inst_147_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_161_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___f_156_; lean_object* v___f_157_; lean_object* v___x_159_; 
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_156_, 0, v_toMul_151_);
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_157_, 0, v_toNPow_152_);
if (v_isShared_155_ == 0)
{
lean_ctor_set(v___x_154_, 2, v___f_157_);
lean_ctor_set(v___x_154_, 1, v___f_156_);
lean_ctor_set(v___x_154_, 0, v_toOne_150_);
v___x_159_ = v___x_154_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v_toOne_150_);
lean_ctor_set(v_reuseFailAlloc_160_, 1, v___f_156_);
lean_ctor_set(v_reuseFailAlloc_160_, 2, v___f_157_);
v___x_159_ = v_reuseFailAlloc_160_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
return v___x_159_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoid(lean_object* v_00_u03b1_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_ULift_monoid___redArg(v_inst_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoid___redArg(lean_object* v_inst_167_){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v_toZero_170_; lean_object* v_toAdd_171_; lean_object* v_toNSMul_172_; lean_object* v___f_173_; lean_object* v___f_174_; lean_object* v___f_175_; lean_object* v___x_176_; 
v___x_168_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_167_);
v___x_169_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_168_);
v_toZero_170_ = lean_ctor_get(v___x_169_, 0);
lean_inc(v_toZero_170_);
v_toAdd_171_ = lean_ctor_get(v___x_169_, 1);
lean_inc(v_toAdd_171_);
lean_dec_ref(v___x_169_);
v_toNSMul_172_ = lean_ctor_get(v_inst_167_, 2);
lean_inc(v_toNSMul_172_);
lean_dec_ref(v_inst_167_);
v___f_173_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_173_, 0, v_toAdd_171_);
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_174_, 0, v_toNSMul_172_);
v___f_175_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_175_, 0, v___f_174_);
v___x_176_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___f_173_, v_toZero_170_, v___f_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addMonoid(lean_object* v_00_u03b1_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lp_mathlib_ULift_addMonoid___redArg(v_inst_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoid___redArg(lean_object* v_inst_180_){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v_toOne_183_; lean_object* v_toMul_184_; lean_object* v_toNPow_185_; lean_object* v___x_187_; uint8_t v_isShared_188_; uint8_t v_isSharedCheck_194_; 
v___x_181_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_180_);
v___x_182_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_181_);
v_toOne_183_ = lean_ctor_get(v___x_182_, 0);
lean_inc(v_toOne_183_);
v_toMul_184_ = lean_ctor_get(v___x_182_, 1);
lean_inc(v_toMul_184_);
lean_dec_ref(v___x_182_);
v_toNPow_185_ = lean_ctor_get(v_inst_180_, 2);
v_isSharedCheck_194_ = !lean_is_exclusive(v_inst_180_);
if (v_isSharedCheck_194_ == 0)
{
lean_object* v_unused_195_; lean_object* v_unused_196_; 
v_unused_195_ = lean_ctor_get(v_inst_180_, 1);
lean_dec(v_unused_195_);
v_unused_196_ = lean_ctor_get(v_inst_180_, 0);
lean_dec(v_unused_196_);
v___x_187_ = v_inst_180_;
v_isShared_188_ = v_isSharedCheck_194_;
goto v_resetjp_186_;
}
else
{
lean_inc(v_toNPow_185_);
lean_dec(v_inst_180_);
v___x_187_ = lean_box(0);
v_isShared_188_ = v_isSharedCheck_194_;
goto v_resetjp_186_;
}
v_resetjp_186_:
{
lean_object* v___f_189_; lean_object* v___f_190_; lean_object* v___x_192_; 
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_189_, 0, v_toMul_184_);
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_190_, 0, v_toNPow_185_);
if (v_isShared_188_ == 0)
{
lean_ctor_set(v___x_187_, 2, v___f_190_);
lean_ctor_set(v___x_187_, 1, v___f_189_);
lean_ctor_set(v___x_187_, 0, v_toOne_183_);
v___x_192_ = v___x_187_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v_toOne_183_);
lean_ctor_set(v_reuseFailAlloc_193_, 1, v___f_189_);
lean_ctor_set(v_reuseFailAlloc_193_, 2, v___f_190_);
v___x_192_ = v_reuseFailAlloc_193_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
return v___x_192_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoid(lean_object* v_00_u03b1_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_ULift_commMonoid___redArg(v_inst_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoid___redArg(lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v_toZero_203_; lean_object* v_toAdd_204_; lean_object* v_toNSMul_205_; lean_object* v___f_206_; lean_object* v___f_207_; lean_object* v___f_208_; lean_object* v___x_209_; 
v___x_201_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_200_);
v___x_202_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_201_);
v_toZero_203_ = lean_ctor_get(v___x_202_, 0);
lean_inc(v_toZero_203_);
v_toAdd_204_ = lean_ctor_get(v___x_202_, 1);
lean_inc(v_toAdd_204_);
lean_dec_ref(v___x_202_);
v_toNSMul_205_ = lean_ctor_get(v_inst_200_, 2);
lean_inc(v_toNSMul_205_);
lean_dec_ref(v_inst_200_);
v___f_206_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_206_, 0, v_toAdd_204_);
v___f_207_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_207_, 0, v_toNSMul_205_);
v___f_208_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_208_, 0, v___f_207_);
v___x_209_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___f_206_, v_toZero_203_, v___f_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommMonoid(lean_object* v_00_u03b1_210_, lean_object* v_inst_211_){
_start:
{
lean_object* v___x_212_; 
v___x_212_ = lp_mathlib_ULift_addCommMonoid___redArg(v_inst_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg___lam__0(lean_object* v_toZPow_213_, lean_object* v_n_214_, lean_object* v_x_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lean_apply_2(v_toZPow_213_, v_n_214_, v_x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid___redArg(lean_object* v_inst_217_){
_start:
{
lean_object* v_toMonoid_218_; lean_object* v_toInv_219_; lean_object* v_toDiv_220_; lean_object* v_toZPow_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_247_; 
v_toMonoid_218_ = lean_ctor_get(v_inst_217_, 0);
v_toInv_219_ = lean_ctor_get(v_inst_217_, 1);
v_toDiv_220_ = lean_ctor_get(v_inst_217_, 2);
v_toZPow_221_ = lean_ctor_get(v_inst_217_, 3);
v_isSharedCheck_247_ = !lean_is_exclusive(v_inst_217_);
if (v_isSharedCheck_247_ == 0)
{
v___x_223_ = v_inst_217_;
v_isShared_224_ = v_isSharedCheck_247_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_toZPow_221_);
lean_inc(v_toDiv_220_);
lean_inc(v_toInv_219_);
lean_inc(v_toMonoid_218_);
lean_dec(v_inst_217_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_247_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v_toOne_227_; lean_object* v_toMul_228_; lean_object* v_toNPow_229_; lean_object* v___x_231_; uint8_t v_isShared_232_; uint8_t v_isSharedCheck_244_; 
v___x_225_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_218_);
v___x_226_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_225_);
v_toOne_227_ = lean_ctor_get(v___x_226_, 0);
lean_inc(v_toOne_227_);
v_toMul_228_ = lean_ctor_get(v___x_226_, 1);
lean_inc(v_toMul_228_);
lean_dec_ref(v___x_226_);
v_toNPow_229_ = lean_ctor_get(v_toMonoid_218_, 2);
v_isSharedCheck_244_ = !lean_is_exclusive(v_toMonoid_218_);
if (v_isSharedCheck_244_ == 0)
{
lean_object* v_unused_245_; lean_object* v_unused_246_; 
v_unused_245_ = lean_ctor_get(v_toMonoid_218_, 1);
lean_dec(v_unused_245_);
v_unused_246_ = lean_ctor_get(v_toMonoid_218_, 0);
lean_dec(v_unused_246_);
v___x_231_ = v_toMonoid_218_;
v_isShared_232_ = v_isSharedCheck_244_;
goto v_resetjp_230_;
}
else
{
lean_inc(v_toNPow_229_);
lean_dec(v_toMonoid_218_);
v___x_231_ = lean_box(0);
v_isShared_232_ = v_isSharedCheck_244_;
goto v_resetjp_230_;
}
v_resetjp_230_:
{
lean_object* v___f_233_; lean_object* v___f_234_; lean_object* v___f_235_; lean_object* v___f_236_; lean_object* v___f_237_; lean_object* v___x_239_; 
v___f_233_ = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_233_, 0, v_toZPow_221_);
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_234_, 0, v_toMul_228_);
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_235_, 0, v_toNPow_229_);
v___f_236_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_236_, 0, v_toInv_219_);
v___f_237_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_237_, 0, v_toDiv_220_);
if (v_isShared_232_ == 0)
{
lean_ctor_set(v___x_231_, 2, v___f_235_);
lean_ctor_set(v___x_231_, 1, v___f_234_);
lean_ctor_set(v___x_231_, 0, v_toOne_227_);
v___x_239_ = v___x_231_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_toOne_227_);
lean_ctor_set(v_reuseFailAlloc_243_, 1, v___f_234_);
lean_ctor_set(v_reuseFailAlloc_243_, 2, v___f_235_);
v___x_239_ = v_reuseFailAlloc_243_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
lean_object* v___x_241_; 
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 3, v___f_233_);
lean_ctor_set(v___x_223_, 2, v___f_237_);
lean_ctor_set(v___x_223_, 1, v___f_236_);
lean_ctor_set(v___x_223_, 0, v___x_239_);
v___x_241_ = v___x_223_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v___x_239_);
lean_ctor_set(v_reuseFailAlloc_242_, 1, v___f_236_);
lean_ctor_set(v_reuseFailAlloc_242_, 2, v___f_237_);
lean_ctor_set(v_reuseFailAlloc_242_, 3, v___f_233_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_divInvMonoid(lean_object* v_00_u03b1_248_, lean_object* v_inst_249_){
_start:
{
lean_object* v___x_250_; 
v___x_250_ = lp_mathlib_ULift_divInvMonoid___redArg(v_inst_249_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_subNegAddMonoid___redArg(lean_object* v_inst_251_){
_start:
{
lean_object* v_toAddMonoid_252_; lean_object* v_toNeg_253_; lean_object* v_toSub_254_; lean_object* v_toZSMul_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v_toZero_258_; lean_object* v_toAdd_259_; lean_object* v_toNSMul_260_; lean_object* v___f_261_; lean_object* v___f_262_; lean_object* v___f_263_; lean_object* v___f_264_; lean_object* v___f_265_; lean_object* v___f_266_; lean_object* v___f_267_; lean_object* v___x_268_; 
v_toAddMonoid_252_ = lean_ctor_get(v_inst_251_, 0);
lean_inc_ref(v_toAddMonoid_252_);
v_toNeg_253_ = lean_ctor_get(v_inst_251_, 1);
lean_inc(v_toNeg_253_);
v_toSub_254_ = lean_ctor_get(v_inst_251_, 2);
lean_inc(v_toSub_254_);
v_toZSMul_255_ = lean_ctor_get(v_inst_251_, 3);
lean_inc(v_toZSMul_255_);
lean_dec_ref(v_inst_251_);
v___x_256_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_252_);
v___x_257_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_256_);
v_toZero_258_ = lean_ctor_get(v___x_257_, 0);
lean_inc(v_toZero_258_);
v_toAdd_259_ = lean_ctor_get(v___x_257_, 1);
lean_inc(v_toAdd_259_);
lean_dec_ref(v___x_257_);
v_toNSMul_260_ = lean_ctor_get(v_toAddMonoid_252_, 2);
lean_inc(v_toNSMul_260_);
lean_dec_ref(v_toAddMonoid_252_);
v___f_261_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_261_, 0, v_toAdd_259_);
v___f_262_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_262_, 0, v_toNSMul_260_);
v___f_263_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_263_, 0, v___f_262_);
v___f_264_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_264_, 0, v_toNeg_253_);
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_265_, 0, v_toSub_254_);
v___f_266_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_266_, 0, v_toZSMul_255_);
v___f_267_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_267_, 0, v___f_266_);
v___x_268_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___f_261_, v_toZero_258_, v___f_263_, v___f_264_, v___f_265_, v___f_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_subNegAddMonoid(lean_object* v_00_u03b1_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_mathlib_ULift_subNegAddMonoid___redArg(v_inst_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_group___redArg(lean_object* v_inst_272_){
_start:
{
lean_object* v_toMonoid_273_; lean_object* v_toInv_274_; lean_object* v_toDiv_275_; lean_object* v_toZPow_276_; lean_object* v___x_278_; uint8_t v_isShared_279_; uint8_t v_isSharedCheck_302_; 
v_toMonoid_273_ = lean_ctor_get(v_inst_272_, 0);
v_toInv_274_ = lean_ctor_get(v_inst_272_, 1);
v_toDiv_275_ = lean_ctor_get(v_inst_272_, 2);
v_toZPow_276_ = lean_ctor_get(v_inst_272_, 3);
v_isSharedCheck_302_ = !lean_is_exclusive(v_inst_272_);
if (v_isSharedCheck_302_ == 0)
{
v___x_278_ = v_inst_272_;
v_isShared_279_ = v_isSharedCheck_302_;
goto v_resetjp_277_;
}
else
{
lean_inc(v_toZPow_276_);
lean_inc(v_toDiv_275_);
lean_inc(v_toInv_274_);
lean_inc(v_toMonoid_273_);
lean_dec(v_inst_272_);
v___x_278_ = lean_box(0);
v_isShared_279_ = v_isSharedCheck_302_;
goto v_resetjp_277_;
}
v_resetjp_277_:
{
lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v_toOne_282_; lean_object* v_toMul_283_; lean_object* v_toNPow_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_299_; 
v___x_280_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_273_);
v___x_281_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_280_);
v_toOne_282_ = lean_ctor_get(v___x_281_, 0);
lean_inc(v_toOne_282_);
v_toMul_283_ = lean_ctor_get(v___x_281_, 1);
lean_inc(v_toMul_283_);
lean_dec_ref(v___x_281_);
v_toNPow_284_ = lean_ctor_get(v_toMonoid_273_, 2);
v_isSharedCheck_299_ = !lean_is_exclusive(v_toMonoid_273_);
if (v_isSharedCheck_299_ == 0)
{
lean_object* v_unused_300_; lean_object* v_unused_301_; 
v_unused_300_ = lean_ctor_get(v_toMonoid_273_, 1);
lean_dec(v_unused_300_);
v_unused_301_ = lean_ctor_get(v_toMonoid_273_, 0);
lean_dec(v_unused_301_);
v___x_286_ = v_toMonoid_273_;
v_isShared_287_ = v_isSharedCheck_299_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_toNPow_284_);
lean_dec(v_toMonoid_273_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_299_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___f_288_; lean_object* v___f_289_; lean_object* v___f_290_; lean_object* v___f_291_; lean_object* v___f_292_; lean_object* v___x_294_; 
v___f_288_ = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_288_, 0, v_toZPow_276_);
v___f_289_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_289_, 0, v_toMul_283_);
v___f_290_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_290_, 0, v_toNPow_284_);
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_291_, 0, v_toInv_274_);
v___f_292_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_292_, 0, v_toDiv_275_);
if (v_isShared_287_ == 0)
{
lean_ctor_set(v___x_286_, 2, v___f_290_);
lean_ctor_set(v___x_286_, 1, v___f_289_);
lean_ctor_set(v___x_286_, 0, v_toOne_282_);
v___x_294_ = v___x_286_;
goto v_reusejp_293_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v_toOne_282_);
lean_ctor_set(v_reuseFailAlloc_298_, 1, v___f_289_);
lean_ctor_set(v_reuseFailAlloc_298_, 2, v___f_290_);
v___x_294_ = v_reuseFailAlloc_298_;
goto v_reusejp_293_;
}
v_reusejp_293_:
{
lean_object* v___x_296_; 
if (v_isShared_279_ == 0)
{
lean_ctor_set(v___x_278_, 3, v___f_288_);
lean_ctor_set(v___x_278_, 2, v___f_292_);
lean_ctor_set(v___x_278_, 1, v___f_291_);
lean_ctor_set(v___x_278_, 0, v___x_294_);
v___x_296_ = v___x_278_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v___x_294_);
lean_ctor_set(v_reuseFailAlloc_297_, 1, v___f_291_);
lean_ctor_set(v_reuseFailAlloc_297_, 2, v___f_292_);
lean_ctor_set(v_reuseFailAlloc_297_, 3, v___f_288_);
v___x_296_ = v_reuseFailAlloc_297_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
return v___x_296_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_group(lean_object* v_00_u03b1_303_, lean_object* v_inst_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_mathlib_ULift_group___redArg(v_inst_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroup___redArg(lean_object* v_inst_306_){
_start:
{
lean_object* v_toAddMonoid_307_; lean_object* v_toNeg_308_; lean_object* v_toSub_309_; lean_object* v_toZSMul_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v_toZero_313_; lean_object* v_toAdd_314_; lean_object* v_toNSMul_315_; lean_object* v___f_316_; lean_object* v___f_317_; lean_object* v___f_318_; lean_object* v___f_319_; lean_object* v___f_320_; lean_object* v___f_321_; lean_object* v___f_322_; lean_object* v___x_323_; 
v_toAddMonoid_307_ = lean_ctor_get(v_inst_306_, 0);
lean_inc_ref(v_toAddMonoid_307_);
v_toNeg_308_ = lean_ctor_get(v_inst_306_, 1);
lean_inc(v_toNeg_308_);
v_toSub_309_ = lean_ctor_get(v_inst_306_, 2);
lean_inc(v_toSub_309_);
v_toZSMul_310_ = lean_ctor_get(v_inst_306_, 3);
lean_inc(v_toZSMul_310_);
lean_dec_ref(v_inst_306_);
v___x_311_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_307_);
v___x_312_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_311_);
v_toZero_313_ = lean_ctor_get(v___x_312_, 0);
lean_inc(v_toZero_313_);
v_toAdd_314_ = lean_ctor_get(v___x_312_, 1);
lean_inc(v_toAdd_314_);
lean_dec_ref(v___x_312_);
v_toNSMul_315_ = lean_ctor_get(v_toAddMonoid_307_, 2);
lean_inc(v_toNSMul_315_);
lean_dec_ref(v_toAddMonoid_307_);
v___f_316_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_316_, 0, v_toAdd_314_);
v___f_317_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_317_, 0, v_toNSMul_315_);
v___f_318_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_318_, 0, v___f_317_);
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_319_, 0, v_toNeg_308_);
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_320_, 0, v_toSub_309_);
v___f_321_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_321_, 0, v_toZSMul_310_);
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_322_, 0, v___f_321_);
v___x_323_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___f_316_, v_toZero_313_, v___f_318_, v___f_319_, v___f_320_, v___f_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addGroup(lean_object* v_00_u03b1_324_, lean_object* v_inst_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lp_mathlib_ULift_addGroup___redArg(v_inst_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroup___redArg(lean_object* v_inst_327_){
_start:
{
lean_object* v_toMonoid_328_; lean_object* v_toInv_329_; lean_object* v_toDiv_330_; lean_object* v_toZPow_331_; lean_object* v___x_333_; uint8_t v_isShared_334_; uint8_t v_isSharedCheck_357_; 
v_toMonoid_328_ = lean_ctor_get(v_inst_327_, 0);
v_toInv_329_ = lean_ctor_get(v_inst_327_, 1);
v_toDiv_330_ = lean_ctor_get(v_inst_327_, 2);
v_toZPow_331_ = lean_ctor_get(v_inst_327_, 3);
v_isSharedCheck_357_ = !lean_is_exclusive(v_inst_327_);
if (v_isSharedCheck_357_ == 0)
{
v___x_333_ = v_inst_327_;
v_isShared_334_ = v_isSharedCheck_357_;
goto v_resetjp_332_;
}
else
{
lean_inc(v_toZPow_331_);
lean_inc(v_toDiv_330_);
lean_inc(v_toInv_329_);
lean_inc(v_toMonoid_328_);
lean_dec(v_inst_327_);
v___x_333_ = lean_box(0);
v_isShared_334_ = v_isSharedCheck_357_;
goto v_resetjp_332_;
}
v_resetjp_332_:
{
lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v_toOne_337_; lean_object* v_toMul_338_; lean_object* v_toNPow_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_354_; 
v___x_335_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_328_);
v___x_336_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_335_);
v_toOne_337_ = lean_ctor_get(v___x_336_, 0);
lean_inc(v_toOne_337_);
v_toMul_338_ = lean_ctor_get(v___x_336_, 1);
lean_inc(v_toMul_338_);
lean_dec_ref(v___x_336_);
v_toNPow_339_ = lean_ctor_get(v_toMonoid_328_, 2);
v_isSharedCheck_354_ = !lean_is_exclusive(v_toMonoid_328_);
if (v_isSharedCheck_354_ == 0)
{
lean_object* v_unused_355_; lean_object* v_unused_356_; 
v_unused_355_ = lean_ctor_get(v_toMonoid_328_, 1);
lean_dec(v_unused_355_);
v_unused_356_ = lean_ctor_get(v_toMonoid_328_, 0);
lean_dec(v_unused_356_);
v___x_341_ = v_toMonoid_328_;
v_isShared_342_ = v_isSharedCheck_354_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_toNPow_339_);
lean_dec(v_toMonoid_328_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_354_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v___f_343_; lean_object* v___f_344_; lean_object* v___f_345_; lean_object* v___f_346_; lean_object* v___f_347_; lean_object* v___x_349_; 
v___f_343_ = lean_alloc_closure((void*)(lp_mathlib_ULift_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_343_, 0, v_toZPow_331_);
v___f_344_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_344_, 0, v_toMul_338_);
v___f_345_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_345_, 0, v_toNPow_339_);
v___f_346_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_346_, 0, v_toInv_329_);
v___f_347_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_347_, 0, v_toDiv_330_);
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 2, v___f_345_);
lean_ctor_set(v___x_341_, 1, v___f_344_);
lean_ctor_set(v___x_341_, 0, v_toOne_337_);
v___x_349_ = v___x_341_;
goto v_reusejp_348_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v_toOne_337_);
lean_ctor_set(v_reuseFailAlloc_353_, 1, v___f_344_);
lean_ctor_set(v_reuseFailAlloc_353_, 2, v___f_345_);
v___x_349_ = v_reuseFailAlloc_353_;
goto v_reusejp_348_;
}
v_reusejp_348_:
{
lean_object* v___x_351_; 
if (v_isShared_334_ == 0)
{
lean_ctor_set(v___x_333_, 3, v___f_343_);
lean_ctor_set(v___x_333_, 2, v___f_347_);
lean_ctor_set(v___x_333_, 1, v___f_346_);
lean_ctor_set(v___x_333_, 0, v___x_349_);
v___x_351_ = v___x_333_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v___x_349_);
lean_ctor_set(v_reuseFailAlloc_352_, 1, v___f_346_);
lean_ctor_set(v_reuseFailAlloc_352_, 2, v___f_347_);
lean_ctor_set(v_reuseFailAlloc_352_, 3, v___f_343_);
v___x_351_ = v_reuseFailAlloc_352_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
return v___x_351_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroup(lean_object* v_00_u03b1_358_, lean_object* v_inst_359_){
_start:
{
lean_object* v___x_360_; 
v___x_360_ = lp_mathlib_ULift_commGroup___redArg(v_inst_359_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroup___redArg(lean_object* v_inst_361_){
_start:
{
lean_object* v_toAddMonoid_362_; lean_object* v_toNeg_363_; lean_object* v_toSub_364_; lean_object* v_toZSMul_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v_toZero_368_; lean_object* v_toAdd_369_; lean_object* v_toNSMul_370_; lean_object* v___f_371_; lean_object* v___f_372_; lean_object* v___f_373_; lean_object* v___f_374_; lean_object* v___f_375_; lean_object* v___f_376_; lean_object* v___f_377_; lean_object* v___x_378_; 
v_toAddMonoid_362_ = lean_ctor_get(v_inst_361_, 0);
lean_inc_ref(v_toAddMonoid_362_);
v_toNeg_363_ = lean_ctor_get(v_inst_361_, 1);
lean_inc(v_toNeg_363_);
v_toSub_364_ = lean_ctor_get(v_inst_361_, 2);
lean_inc(v_toSub_364_);
v_toZSMul_365_ = lean_ctor_get(v_inst_361_, 3);
lean_inc(v_toZSMul_365_);
lean_dec_ref(v_inst_361_);
v___x_366_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_362_);
v___x_367_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_366_);
v_toZero_368_ = lean_ctor_get(v___x_367_, 0);
lean_inc(v_toZero_368_);
v_toAdd_369_ = lean_ctor_get(v___x_367_, 1);
lean_inc(v_toAdd_369_);
lean_dec_ref(v___x_367_);
v_toNSMul_370_ = lean_ctor_get(v_toAddMonoid_362_, 2);
lean_inc(v_toNSMul_370_);
lean_dec_ref(v_toAddMonoid_362_);
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_371_, 0, v_toAdd_369_);
v___f_372_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_372_, 0, v_toNSMul_370_);
v___f_373_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_373_, 0, v___f_372_);
v___f_374_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_374_, 0, v_toNeg_363_);
v___f_375_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_375_, 0, v_toSub_364_);
v___f_376_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_376_, 0, v_toZSMul_365_);
v___f_377_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_377_, 0, v___f_376_);
v___x_378_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___f_371_, v_toZero_368_, v___f_373_, v___f_374_, v___f_375_, v___f_377_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCommGroup(lean_object* v_00_u03b1_379_, lean_object* v_inst_380_){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lp_mathlib_ULift_addCommGroup___redArg(v_inst_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelSemigroup___redArg(lean_object* v_inst_382_){
_start:
{
lean_object* v___f_383_; 
v___f_383_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_383_, 0, v_inst_382_);
return v___f_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelSemigroup(lean_object* v_00_u03b1_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v___f_386_; 
v___f_386_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_386_, 0, v_inst_385_);
return v___f_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelSemigroup___redArg(lean_object* v_inst_387_){
_start:
{
lean_object* v___f_388_; 
v___f_388_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_388_, 0, v_inst_387_);
return v___f_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelSemigroup(lean_object* v_00_u03b1_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v___f_391_; 
v___f_391_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_391_, 0, v_inst_390_);
return v___f_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelSemigroup___redArg(lean_object* v_inst_392_){
_start:
{
lean_object* v___f_393_; 
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_393_, 0, v_inst_392_);
return v___f_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelSemigroup(lean_object* v_00_u03b1_394_, lean_object* v_inst_395_){
_start:
{
lean_object* v___f_396_; 
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_396_, 0, v_inst_395_);
return v___f_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelSemigroup___redArg(lean_object* v_inst_397_){
_start:
{
lean_object* v___f_398_; 
v___f_398_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_398_, 0, v_inst_397_);
return v___f_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelSemigroup(lean_object* v_00_u03b1_399_, lean_object* v_inst_400_){
_start:
{
lean_object* v___f_401_; 
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_401_, 0, v_inst_400_);
return v___f_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelMonoid___redArg(lean_object* v_inst_402_){
_start:
{
lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v_toOne_405_; lean_object* v_toMul_406_; lean_object* v_toNPow_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_416_; 
v___x_403_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_402_);
v___x_404_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_403_);
v_toOne_405_ = lean_ctor_get(v___x_404_, 0);
lean_inc(v_toOne_405_);
v_toMul_406_ = lean_ctor_get(v___x_404_, 1);
lean_inc(v_toMul_406_);
lean_dec_ref(v___x_404_);
v_toNPow_407_ = lean_ctor_get(v_inst_402_, 2);
v_isSharedCheck_416_ = !lean_is_exclusive(v_inst_402_);
if (v_isSharedCheck_416_ == 0)
{
lean_object* v_unused_417_; lean_object* v_unused_418_; 
v_unused_417_ = lean_ctor_get(v_inst_402_, 1);
lean_dec(v_unused_417_);
v_unused_418_ = lean_ctor_get(v_inst_402_, 0);
lean_dec(v_unused_418_);
v___x_409_ = v_inst_402_;
v_isShared_410_ = v_isSharedCheck_416_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_toNPow_407_);
lean_dec(v_inst_402_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_416_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
lean_object* v___f_411_; lean_object* v___f_412_; lean_object* v___x_414_; 
v___f_411_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_411_, 0, v_toMul_406_);
v___f_412_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_412_, 0, v_toNPow_407_);
if (v_isShared_410_ == 0)
{
lean_ctor_set(v___x_409_, 2, v___f_412_);
lean_ctor_set(v___x_409_, 1, v___f_411_);
lean_ctor_set(v___x_409_, 0, v_toOne_405_);
v___x_414_ = v___x_409_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v_toOne_405_);
lean_ctor_set(v_reuseFailAlloc_415_, 1, v___f_411_);
lean_ctor_set(v_reuseFailAlloc_415_, 2, v___f_412_);
v___x_414_ = v_reuseFailAlloc_415_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
return v___x_414_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_leftCancelMonoid(lean_object* v_00_u03b1_419_, lean_object* v_inst_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lp_mathlib_ULift_leftCancelMonoid___redArg(v_inst_420_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelMonoid___redArg(lean_object* v_inst_422_){
_start:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v_toZero_425_; lean_object* v_toAdd_426_; lean_object* v_toNSMul_427_; lean_object* v___f_428_; lean_object* v___f_429_; lean_object* v___f_430_; lean_object* v___x_431_; 
v___x_423_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_422_);
v___x_424_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_423_);
v_toZero_425_ = lean_ctor_get(v___x_424_, 0);
lean_inc(v_toZero_425_);
v_toAdd_426_ = lean_ctor_get(v___x_424_, 1);
lean_inc(v_toAdd_426_);
lean_dec_ref(v___x_424_);
v_toNSMul_427_ = lean_ctor_get(v_inst_422_, 2);
lean_inc(v_toNSMul_427_);
lean_dec_ref(v_inst_422_);
v___f_428_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_428_, 0, v_toAdd_426_);
v___f_429_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_429_, 0, v_toNSMul_427_);
v___f_430_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_430_, 0, v___f_429_);
v___x_431_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___f_428_, v_toZero_425_, v___f_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addLeftCancelMonoid(lean_object* v_00_u03b1_432_, lean_object* v_inst_433_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lp_mathlib_ULift_addLeftCancelMonoid___redArg(v_inst_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelMonoid___redArg(lean_object* v_inst_435_){
_start:
{
lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v_toOne_438_; lean_object* v_toMul_439_; lean_object* v_toNPow_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_449_; 
v___x_436_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_435_);
v___x_437_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_436_);
v_toOne_438_ = lean_ctor_get(v___x_437_, 0);
lean_inc(v_toOne_438_);
v_toMul_439_ = lean_ctor_get(v___x_437_, 1);
lean_inc(v_toMul_439_);
lean_dec_ref(v___x_437_);
v_toNPow_440_ = lean_ctor_get(v_inst_435_, 2);
v_isSharedCheck_449_ = !lean_is_exclusive(v_inst_435_);
if (v_isSharedCheck_449_ == 0)
{
lean_object* v_unused_450_; lean_object* v_unused_451_; 
v_unused_450_ = lean_ctor_get(v_inst_435_, 1);
lean_dec(v_unused_450_);
v_unused_451_ = lean_ctor_get(v_inst_435_, 0);
lean_dec(v_unused_451_);
v___x_442_ = v_inst_435_;
v_isShared_443_ = v_isSharedCheck_449_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_toNPow_440_);
lean_dec(v_inst_435_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_449_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___f_444_; lean_object* v___f_445_; lean_object* v___x_447_; 
v___f_444_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_444_, 0, v_toMul_439_);
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_445_, 0, v_toNPow_440_);
if (v_isShared_443_ == 0)
{
lean_ctor_set(v___x_442_, 2, v___f_445_);
lean_ctor_set(v___x_442_, 1, v___f_444_);
lean_ctor_set(v___x_442_, 0, v_toOne_438_);
v___x_447_ = v___x_442_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v_toOne_438_);
lean_ctor_set(v_reuseFailAlloc_448_, 1, v___f_444_);
lean_ctor_set(v_reuseFailAlloc_448_, 2, v___f_445_);
v___x_447_ = v_reuseFailAlloc_448_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
return v___x_447_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_rightCancelMonoid(lean_object* v_00_u03b1_452_, lean_object* v_inst_453_){
_start:
{
lean_object* v___x_454_; 
v___x_454_ = lp_mathlib_ULift_rightCancelMonoid___redArg(v_inst_453_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelMonoid___redArg(lean_object* v_inst_455_){
_start:
{
lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v_toZero_458_; lean_object* v_toAdd_459_; lean_object* v_toNSMul_460_; lean_object* v___f_461_; lean_object* v___f_462_; lean_object* v___f_463_; lean_object* v___x_464_; 
v___x_456_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_455_);
v___x_457_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_456_);
v_toZero_458_ = lean_ctor_get(v___x_457_, 0);
lean_inc(v_toZero_458_);
v_toAdd_459_ = lean_ctor_get(v___x_457_, 1);
lean_inc(v_toAdd_459_);
lean_dec_ref(v___x_457_);
v_toNSMul_460_ = lean_ctor_get(v_inst_455_, 2);
lean_inc(v_toNSMul_460_);
lean_dec_ref(v_inst_455_);
v___f_461_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_461_, 0, v_toAdd_459_);
v___f_462_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_462_, 0, v_toNSMul_460_);
v___f_463_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_463_, 0, v___f_462_);
v___x_464_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___f_461_, v_toZero_458_, v___f_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addRightCancelMonoid(lean_object* v_00_u03b1_465_, lean_object* v_inst_466_){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lp_mathlib_ULift_addRightCancelMonoid___redArg(v_inst_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelMonoid___redArg(lean_object* v_inst_468_){
_start:
{
lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v_toOne_471_; lean_object* v_toMul_472_; lean_object* v_toNPow_473_; lean_object* v___x_475_; uint8_t v_isShared_476_; uint8_t v_isSharedCheck_482_; 
v___x_469_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_468_);
v___x_470_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_469_);
v_toOne_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc(v_toOne_471_);
v_toMul_472_ = lean_ctor_get(v___x_470_, 1);
lean_inc(v_toMul_472_);
lean_dec_ref(v___x_470_);
v_toNPow_473_ = lean_ctor_get(v_inst_468_, 2);
v_isSharedCheck_482_ = !lean_is_exclusive(v_inst_468_);
if (v_isSharedCheck_482_ == 0)
{
lean_object* v_unused_483_; lean_object* v_unused_484_; 
v_unused_483_ = lean_ctor_get(v_inst_468_, 1);
lean_dec(v_unused_483_);
v_unused_484_ = lean_ctor_get(v_inst_468_, 0);
lean_dec(v_unused_484_);
v___x_475_ = v_inst_468_;
v_isShared_476_ = v_isSharedCheck_482_;
goto v_resetjp_474_;
}
else
{
lean_inc(v_toNPow_473_);
lean_dec(v_inst_468_);
v___x_475_ = lean_box(0);
v_isShared_476_ = v_isSharedCheck_482_;
goto v_resetjp_474_;
}
v_resetjp_474_:
{
lean_object* v___f_477_; lean_object* v___f_478_; lean_object* v___x_480_; 
v___f_477_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_477_, 0, v_toMul_472_);
v___f_478_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_478_, 0, v_toNPow_473_);
if (v_isShared_476_ == 0)
{
lean_ctor_set(v___x_475_, 2, v___f_478_);
lean_ctor_set(v___x_475_, 1, v___f_477_);
lean_ctor_set(v___x_475_, 0, v_toOne_471_);
v___x_480_ = v___x_475_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v_toOne_471_);
lean_ctor_set(v_reuseFailAlloc_481_, 1, v___f_477_);
lean_ctor_set(v_reuseFailAlloc_481_, 2, v___f_478_);
v___x_480_ = v_reuseFailAlloc_481_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
return v___x_480_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelMonoid(lean_object* v_00_u03b1_485_, lean_object* v_inst_486_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_mathlib_ULift_cancelMonoid___redArg(v_inst_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelMonoid___redArg(lean_object* v_inst_488_){
_start:
{
lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v_toZero_491_; lean_object* v_toAdd_492_; lean_object* v_toNSMul_493_; lean_object* v___f_494_; lean_object* v___f_495_; lean_object* v___f_496_; lean_object* v___x_497_; 
v___x_489_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_488_);
v___x_490_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_489_);
v_toZero_491_ = lean_ctor_get(v___x_490_, 0);
lean_inc(v_toZero_491_);
v_toAdd_492_ = lean_ctor_get(v___x_490_, 1);
lean_inc(v_toAdd_492_);
lean_dec_ref(v___x_490_);
v_toNSMul_493_ = lean_ctor_get(v_inst_488_, 2);
lean_inc(v_toNSMul_493_);
lean_dec_ref(v_inst_488_);
v___f_494_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_494_, 0, v_toAdd_492_);
v___f_495_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_495_, 0, v_toNSMul_493_);
v___f_496_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_496_, 0, v___f_495_);
v___x_497_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___f_494_, v_toZero_491_, v___f_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelMonoid(lean_object* v_00_u03b1_498_, lean_object* v_inst_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lp_mathlib_ULift_addCancelMonoid___redArg(v_inst_499_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelCommMonoid___redArg(lean_object* v_inst_501_){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v_toOne_504_; lean_object* v_toMul_505_; lean_object* v_toNPow_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_515_; 
v___x_502_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_501_);
v___x_503_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_502_);
v_toOne_504_ = lean_ctor_get(v___x_503_, 0);
lean_inc(v_toOne_504_);
v_toMul_505_ = lean_ctor_get(v___x_503_, 1);
lean_inc(v_toMul_505_);
lean_dec_ref(v___x_503_);
v_toNPow_506_ = lean_ctor_get(v_inst_501_, 2);
v_isSharedCheck_515_ = !lean_is_exclusive(v_inst_501_);
if (v_isSharedCheck_515_ == 0)
{
lean_object* v_unused_516_; lean_object* v_unused_517_; 
v_unused_516_ = lean_ctor_get(v_inst_501_, 1);
lean_dec(v_unused_516_);
v_unused_517_ = lean_ctor_get(v_inst_501_, 0);
lean_dec(v_unused_517_);
v___x_508_ = v_inst_501_;
v_isShared_509_ = v_isSharedCheck_515_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_toNPow_506_);
lean_dec(v_inst_501_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_515_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v___f_510_; lean_object* v___f_511_; lean_object* v___x_513_; 
v___f_510_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_510_, 0, v_toMul_505_);
v___f_511_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_511_, 0, v_toNPow_506_);
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 2, v___f_511_);
lean_ctor_set(v___x_508_, 1, v___f_510_);
lean_ctor_set(v___x_508_, 0, v_toOne_504_);
v___x_513_ = v___x_508_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v_toOne_504_);
lean_ctor_set(v_reuseFailAlloc_514_, 1, v___f_510_);
lean_ctor_set(v_reuseFailAlloc_514_, 2, v___f_511_);
v___x_513_ = v_reuseFailAlloc_514_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
return v___x_513_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_cancelCommMonoid(lean_object* v_00_u03b1_518_, lean_object* v_inst_519_){
_start:
{
lean_object* v___x_520_; 
v___x_520_ = lp_mathlib_ULift_cancelCommMonoid___redArg(v_inst_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelCommMonoid___redArg(lean_object* v_inst_521_){
_start:
{
lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v_toZero_524_; lean_object* v_toAdd_525_; lean_object* v_toNSMul_526_; lean_object* v___f_527_; lean_object* v___f_528_; lean_object* v___f_529_; lean_object* v___x_530_; 
v___x_522_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_521_);
v___x_523_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_522_);
v_toZero_524_ = lean_ctor_get(v___x_523_, 0);
lean_inc(v_toZero_524_);
v_toAdd_525_ = lean_ctor_get(v___x_523_, 1);
lean_inc(v_toAdd_525_);
lean_dec_ref(v___x_523_);
v_toNSMul_526_ = lean_ctor_get(v_inst_521_, 2);
lean_inc(v_toNSMul_526_);
lean_dec_ref(v_inst_521_);
v___f_527_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_527_, 0, v_toAdd_525_);
v___f_528_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_528_, 0, v_toNSMul_526_);
v___f_529_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_529_, 0, v___f_528_);
v___x_530_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___f_527_, v_toZero_524_, v___f_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addCancelCommMonoid(lean_object* v_00_u03b1_531_, lean_object* v_inst_532_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_mathlib_ULift_addCancelCommMonoid___redArg(v_inst_532_);
return v___x_533_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_ULift(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_ULift(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_ULift(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_ULift(builtin);
}
#ifdef __cplusplus
}
#endif
