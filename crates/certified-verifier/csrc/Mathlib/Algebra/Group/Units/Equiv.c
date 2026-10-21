// Lean compiler output
// Module: Mathlib.Algebra.Group.Units.Equiv
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Basic public import Mathlib.Algebra.Group.Units.Hom
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Units_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_toUnits___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_toUnits___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_toUnits___redArg___closed__0 = (const lean_object*)&lp_mathlib_toUnits___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_toAddUnits___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_toAddUnits___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_toAddUnits___redArg___closed__0 = (const lean_object*)&lp_mathlib_toAddUnits___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mapEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mapEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mapEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mapEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addLeft___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addRight___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subLeft___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivProdSubtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivProdSubtype___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_unitsEquivProdSubtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_unitsEquivProdSubtype___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_unitsEquivProdSubtype___closed__0 = (const lean_object*)&lp_mathlib_unitsEquivProdSubtype___closed__0_value;
static const lean_closure_object lp_mathlib_unitsEquivProdSubtype___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_unitsEquivProdSubtype___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_unitsEquivProdSubtype___closed__1 = (const lean_object*)&lp_mathlib_unitsEquivProdSubtype___closed__1_value;
static const lean_ctor_object lp_mathlib_unitsEquivProdSubtype___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_unitsEquivProdSubtype___closed__0_value),((lean_object*)&lp_mathlib_unitsEquivProdSubtype___closed__1_value)}};
static const lean_object* lp_mathlib_unitsEquivProdSubtype___closed__2 = (const lean_object*)&lp_mathlib_unitsEquivProdSubtype___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivProdSubtype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivProdSubtype___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg___lam__0(lean_object* v_x_1_){
_start:
{
lean_object* v_val_2_; 
v_val_2_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_val_2_);
return v_val_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg___lam__0___boxed(lean_object* v_x_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_toUnits___redArg___lam__0(v_x_3_);
lean_dec_ref(v_x_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg___lam__1(lean_object* v_toInv_5_, lean_object* v_x_6_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
lean_inc(v_x_6_);
v___x_7_ = lean_apply_1(v_toInv_5_, v_x_6_);
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v_x_6_);
lean_ctor_set(v___x_8_, 1, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; lean_object* v_toInv_12_; lean_object* v___x_14_; uint8_t v_isShared_15_; uint8_t v_isSharedCheck_21_; 
v___x_11_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_10_);
v_toInv_12_ = lean_ctor_get(v___x_11_, 1);
v_isSharedCheck_21_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_21_ == 0)
{
lean_object* v_unused_22_; 
v_unused_22_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_22_);
v___x_14_ = v___x_11_;
v_isShared_15_ = v_isSharedCheck_21_;
goto v_resetjp_13_;
}
else
{
lean_inc(v_toInv_12_);
lean_dec(v___x_11_);
v___x_14_ = lean_box(0);
v_isShared_15_ = v_isSharedCheck_21_;
goto v_resetjp_13_;
}
v_resetjp_13_:
{
lean_object* v___f_16_; lean_object* v___f_17_; lean_object* v___x_19_; 
v___f_16_ = ((lean_object*)(lp_mathlib_toUnits___redArg___closed__0));
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_toUnits___redArg___lam__1), 2, 1);
lean_closure_set(v___f_17_, 0, v_toInv_12_);
if (v_isShared_15_ == 0)
{
lean_ctor_set(v___x_14_, 1, v___f_16_);
lean_ctor_set(v___x_14_, 0, v___f_17_);
v___x_19_ = v___x_14_;
goto v_reusejp_18_;
}
else
{
lean_object* v_reuseFailAlloc_20_; 
v_reuseFailAlloc_20_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_20_, 0, v___f_17_);
lean_ctor_set(v_reuseFailAlloc_20_, 1, v___f_16_);
v___x_19_ = v_reuseFailAlloc_20_;
goto v_reusejp_18_;
}
v_reusejp_18_:
{
return v___x_19_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits___redArg___boxed(lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_toUnits___redArg(v_inst_23_);
lean_dec_ref(v_inst_23_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits(lean_object* v_G_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_toUnits___redArg(v_inst_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toUnits___boxed(lean_object* v_G_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_toUnits(v_G_28_, v_inst_29_);
lean_dec_ref(v_inst_29_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg___lam__0(lean_object* v_x_31_){
_start:
{
lean_object* v_val_32_; 
v_val_32_ = lean_ctor_get(v_x_31_, 0);
lean_inc(v_val_32_);
return v_val_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg___lam__0___boxed(lean_object* v_x_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_toAddUnits___redArg___lam__0(v_x_33_);
lean_dec_ref(v_x_33_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg___lam__1(lean_object* v_toNeg_35_, lean_object* v_x_36_){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; 
lean_inc(v_x_36_);
v___x_37_ = lean_apply_1(v_toNeg_35_, v_x_36_);
v___x_38_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_38_, 0, v_x_36_);
lean_ctor_set(v___x_38_, 1, v___x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg(lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; lean_object* v_toNeg_42_; lean_object* v___x_44_; uint8_t v_isShared_45_; uint8_t v_isSharedCheck_51_; 
v___x_41_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_40_);
v_toNeg_42_ = lean_ctor_get(v___x_41_, 1);
v_isSharedCheck_51_ = !lean_is_exclusive(v___x_41_);
if (v_isSharedCheck_51_ == 0)
{
lean_object* v_unused_52_; 
v_unused_52_ = lean_ctor_get(v___x_41_, 0);
lean_dec(v_unused_52_);
v___x_44_ = v___x_41_;
v_isShared_45_ = v_isSharedCheck_51_;
goto v_resetjp_43_;
}
else
{
lean_inc(v_toNeg_42_);
lean_dec(v___x_41_);
v___x_44_ = lean_box(0);
v_isShared_45_ = v_isSharedCheck_51_;
goto v_resetjp_43_;
}
v_resetjp_43_:
{
lean_object* v___f_46_; lean_object* v___f_47_; lean_object* v___x_49_; 
v___f_46_ = ((lean_object*)(lp_mathlib_toAddUnits___redArg___closed__0));
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_toAddUnits___redArg___lam__1), 2, 1);
lean_closure_set(v___f_47_, 0, v_toNeg_42_);
if (v_isShared_45_ == 0)
{
lean_ctor_set(v___x_44_, 1, v___f_46_);
lean_ctor_set(v___x_44_, 0, v___f_47_);
v___x_49_ = v___x_44_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v___f_47_);
lean_ctor_set(v_reuseFailAlloc_50_, 1, v___f_46_);
v___x_49_ = v_reuseFailAlloc_50_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
return v___x_49_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___redArg___boxed(lean_object* v_inst_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_toAddUnits___redArg(v_inst_53_);
lean_dec_ref(v_inst_53_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits(lean_object* v_G_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_toAddUnits___redArg(v_inst_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAddUnits___boxed(lean_object* v_G_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_toAddUnits(v_G_58_, v_inst_59_);
lean_dec_ref(v_inst_59_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mapEquiv___redArg___lam__0(lean_object* v_toFun_61_, lean_object* v___y_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_Units_map___redArg___lam__0(v_toFun_61_, v___y_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mapEquiv___redArg(lean_object* v_h_64_){
_start:
{
lean_object* v_toFun_65_; lean_object* v___x_66_; lean_object* v_toFun_67_; lean_object* v___x_69_; uint8_t v_isShared_70_; uint8_t v_isSharedCheck_76_; 
v_toFun_65_ = lean_ctor_get(v_h_64_, 0);
lean_inc(v_toFun_65_);
v___x_66_ = lp_mathlib_Equiv_symm___redArg(v_h_64_);
v_toFun_67_ = lean_ctor_get(v___x_66_, 0);
v_isSharedCheck_76_ = !lean_is_exclusive(v___x_66_);
if (v_isSharedCheck_76_ == 0)
{
lean_object* v_unused_77_; 
v_unused_77_ = lean_ctor_get(v___x_66_, 1);
lean_dec(v_unused_77_);
v___x_69_ = v___x_66_;
v_isShared_70_ = v_isSharedCheck_76_;
goto v_resetjp_68_;
}
else
{
lean_inc(v_toFun_67_);
lean_dec(v___x_66_);
v___x_69_ = lean_box(0);
v_isShared_70_ = v_isSharedCheck_76_;
goto v_resetjp_68_;
}
v_resetjp_68_:
{
lean_object* v___f_71_; lean_object* v___f_72_; lean_object* v___x_74_; 
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Units_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_71_, 0, v_toFun_65_);
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_Units_mapEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_72_, 0, v_toFun_67_);
if (v_isShared_70_ == 0)
{
lean_ctor_set(v___x_69_, 1, v___f_72_);
lean_ctor_set(v___x_69_, 0, v___f_71_);
v___x_74_ = v___x_69_;
goto v_reusejp_73_;
}
else
{
lean_object* v_reuseFailAlloc_75_; 
v_reuseFailAlloc_75_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_75_, 0, v___f_71_);
lean_ctor_set(v_reuseFailAlloc_75_, 1, v___f_72_);
v___x_74_ = v_reuseFailAlloc_75_;
goto v_reusejp_73_;
}
v_reusejp_73_:
{
return v___x_74_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mapEquiv(lean_object* v_M_78_, lean_object* v_N_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_h_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_mathlib_Units_mapEquiv___redArg(v_h_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mapEquiv___boxed(lean_object* v_M_84_, lean_object* v_N_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_h_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Units_mapEquiv(v_M_84_, v_N_85_, v_inst_86_, v_inst_87_, v_h_88_);
lean_dec_ref(v_inst_87_);
lean_dec_ref(v_inst_86_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___redArg___lam__0(lean_object* v_u_90_, lean_object* v_toMul_91_, lean_object* v_x_92_){
_start:
{
lean_object* v_val_93_; lean_object* v___x_94_; 
v_val_93_ = lean_ctor_get(v_u_90_, 0);
lean_inc(v_val_93_);
lean_dec_ref(v_u_90_);
v___x_94_ = lean_apply_2(v_toMul_91_, v_val_93_, v_x_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___redArg___lam__1(lean_object* v_u_95_, lean_object* v_toMul_96_, lean_object* v_x_97_){
_start:
{
lean_object* v_inv_98_; lean_object* v___x_99_; 
v_inv_98_ = lean_ctor_get(v_u_95_, 1);
lean_inc(v_inv_98_);
lean_dec_ref(v_u_95_);
v___x_99_ = lean_apply_2(v_toMul_96_, v_inv_98_, v_x_97_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___redArg(lean_object* v_inst_100_, lean_object* v_u_101_){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v_toMul_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_113_; 
v___x_102_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_100_);
v___x_103_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_102_);
v_toMul_104_ = lean_ctor_get(v___x_103_, 1);
v_isSharedCheck_113_ = !lean_is_exclusive(v___x_103_);
if (v_isSharedCheck_113_ == 0)
{
lean_object* v_unused_114_; 
v_unused_114_ = lean_ctor_get(v___x_103_, 0);
lean_dec(v_unused_114_);
v___x_106_ = v___x_103_;
v_isShared_107_ = v_isSharedCheck_113_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_toMul_104_);
lean_dec(v___x_103_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_113_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___f_108_; lean_object* v___f_109_; lean_object* v___x_111_; 
lean_inc(v_toMul_104_);
lean_inc_ref(v_u_101_);
v___f_108_ = lean_alloc_closure((void*)(lp_mathlib_Units_mulLeft___redArg___lam__0), 3, 2);
lean_closure_set(v___f_108_, 0, v_u_101_);
lean_closure_set(v___f_108_, 1, v_toMul_104_);
v___f_109_ = lean_alloc_closure((void*)(lp_mathlib_Units_mulLeft___redArg___lam__1), 3, 2);
lean_closure_set(v___f_109_, 0, v_u_101_);
lean_closure_set(v___f_109_, 1, v_toMul_104_);
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 1, v___f_109_);
lean_ctor_set(v___x_106_, 0, v___f_108_);
v___x_111_ = v___x_106_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v___f_108_);
lean_ctor_set(v_reuseFailAlloc_112_, 1, v___f_109_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
return v___x_111_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___redArg___boxed(lean_object* v_inst_115_, lean_object* v_u_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Units_mulLeft___redArg(v_inst_115_, v_u_116_);
lean_dec_ref(v_inst_115_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft(lean_object* v_M_118_, lean_object* v_inst_119_, lean_object* v_u_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lp_mathlib_Units_mulLeft___redArg(v_inst_119_, v_u_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulLeft___boxed(lean_object* v_M_122_, lean_object* v_inst_123_, lean_object* v_u_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Units_mulLeft(v_M_122_, v_inst_123_, v_u_124_);
lean_dec_ref(v_inst_123_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___redArg___lam__0(lean_object* v_u_126_, lean_object* v_toAdd_127_, lean_object* v_x_128_){
_start:
{
lean_object* v_val_129_; lean_object* v___x_130_; 
v_val_129_ = lean_ctor_get(v_u_126_, 0);
lean_inc(v_val_129_);
lean_dec_ref(v_u_126_);
v___x_130_ = lean_apply_2(v_toAdd_127_, v_val_129_, v_x_128_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___redArg___lam__1(lean_object* v_u_131_, lean_object* v_toAdd_132_, lean_object* v_x_133_){
_start:
{
lean_object* v_neg_134_; lean_object* v___x_135_; 
v_neg_134_ = lean_ctor_get(v_u_131_, 1);
lean_inc(v_neg_134_);
lean_dec_ref(v_u_131_);
v___x_135_ = lean_apply_2(v_toAdd_132_, v_neg_134_, v_x_133_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___redArg(lean_object* v_inst_136_, lean_object* v_u_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v_toAdd_140_; lean_object* v___x_142_; uint8_t v_isShared_143_; uint8_t v_isSharedCheck_149_; 
v___x_138_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_136_);
v___x_139_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_138_);
v_toAdd_140_ = lean_ctor_get(v___x_139_, 1);
v_isSharedCheck_149_ = !lean_is_exclusive(v___x_139_);
if (v_isSharedCheck_149_ == 0)
{
lean_object* v_unused_150_; 
v_unused_150_ = lean_ctor_get(v___x_139_, 0);
lean_dec(v_unused_150_);
v___x_142_ = v___x_139_;
v_isShared_143_ = v_isSharedCheck_149_;
goto v_resetjp_141_;
}
else
{
lean_inc(v_toAdd_140_);
lean_dec(v___x_139_);
v___x_142_ = lean_box(0);
v_isShared_143_ = v_isSharedCheck_149_;
goto v_resetjp_141_;
}
v_resetjp_141_:
{
lean_object* v___f_144_; lean_object* v___f_145_; lean_object* v___x_147_; 
lean_inc(v_toAdd_140_);
lean_inc_ref(v_u_137_);
v___f_144_ = lean_alloc_closure((void*)(lp_mathlib_AddUnits_addLeft___redArg___lam__0), 3, 2);
lean_closure_set(v___f_144_, 0, v_u_137_);
lean_closure_set(v___f_144_, 1, v_toAdd_140_);
v___f_145_ = lean_alloc_closure((void*)(lp_mathlib_AddUnits_addLeft___redArg___lam__1), 3, 2);
lean_closure_set(v___f_145_, 0, v_u_137_);
lean_closure_set(v___f_145_, 1, v_toAdd_140_);
if (v_isShared_143_ == 0)
{
lean_ctor_set(v___x_142_, 1, v___f_145_);
lean_ctor_set(v___x_142_, 0, v___f_144_);
v___x_147_ = v___x_142_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___f_144_);
lean_ctor_set(v_reuseFailAlloc_148_, 1, v___f_145_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
return v___x_147_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___redArg___boxed(lean_object* v_inst_151_, lean_object* v_u_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_AddUnits_addLeft___redArg(v_inst_151_, v_u_152_);
lean_dec_ref(v_inst_151_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft(lean_object* v_M_154_, lean_object* v_inst_155_, lean_object* v_u_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lp_mathlib_AddUnits_addLeft___redArg(v_inst_155_, v_u_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addLeft___boxed(lean_object* v_M_158_, lean_object* v_inst_159_, lean_object* v_u_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_AddUnits_addLeft(v_M_158_, v_inst_159_, v_u_160_);
lean_dec_ref(v_inst_159_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___redArg___lam__0(lean_object* v_u_162_, lean_object* v_toMul_163_, lean_object* v_x_164_){
_start:
{
lean_object* v_val_165_; lean_object* v___x_166_; 
v_val_165_ = lean_ctor_get(v_u_162_, 0);
lean_inc(v_val_165_);
lean_dec_ref(v_u_162_);
v___x_166_ = lean_apply_2(v_toMul_163_, v_x_164_, v_val_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___redArg___lam__1(lean_object* v_u_167_, lean_object* v_toMul_168_, lean_object* v_x_169_){
_start:
{
lean_object* v_inv_170_; lean_object* v___x_171_; 
v_inv_170_ = lean_ctor_get(v_u_167_, 1);
lean_inc(v_inv_170_);
lean_dec_ref(v_u_167_);
v___x_171_ = lean_apply_2(v_toMul_168_, v_x_169_, v_inv_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___redArg(lean_object* v_inst_172_, lean_object* v_u_173_){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v_toMul_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_185_; 
v___x_174_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_172_);
v___x_175_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_174_);
v_toMul_176_ = lean_ctor_get(v___x_175_, 1);
v_isSharedCheck_185_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_185_ == 0)
{
lean_object* v_unused_186_; 
v_unused_186_ = lean_ctor_get(v___x_175_, 0);
lean_dec(v_unused_186_);
v___x_178_ = v___x_175_;
v_isShared_179_ = v_isSharedCheck_185_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_toMul_176_);
lean_dec(v___x_175_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_185_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v___f_180_; lean_object* v___f_181_; lean_object* v___x_183_; 
lean_inc(v_toMul_176_);
lean_inc_ref(v_u_173_);
v___f_180_ = lean_alloc_closure((void*)(lp_mathlib_Units_mulRight___redArg___lam__0), 3, 2);
lean_closure_set(v___f_180_, 0, v_u_173_);
lean_closure_set(v___f_180_, 1, v_toMul_176_);
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_Units_mulRight___redArg___lam__1), 3, 2);
lean_closure_set(v___f_181_, 0, v_u_173_);
lean_closure_set(v___f_181_, 1, v_toMul_176_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 1, v___f_181_);
lean_ctor_set(v___x_178_, 0, v___f_180_);
v___x_183_ = v___x_178_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v___f_180_);
lean_ctor_set(v_reuseFailAlloc_184_, 1, v___f_181_);
v___x_183_ = v_reuseFailAlloc_184_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
return v___x_183_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___redArg___boxed(lean_object* v_inst_187_, lean_object* v_u_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_Units_mulRight___redArg(v_inst_187_, v_u_188_);
lean_dec_ref(v_inst_187_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight(lean_object* v_M_190_, lean_object* v_inst_191_, lean_object* v_u_192_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_mathlib_Units_mulRight___redArg(v_inst_191_, v_u_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_mulRight___boxed(lean_object* v_M_194_, lean_object* v_inst_195_, lean_object* v_u_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_Units_mulRight(v_M_194_, v_inst_195_, v_u_196_);
lean_dec_ref(v_inst_195_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___redArg___lam__0(lean_object* v_u_198_, lean_object* v_toAdd_199_, lean_object* v_x_200_){
_start:
{
lean_object* v_val_201_; lean_object* v___x_202_; 
v_val_201_ = lean_ctor_get(v_u_198_, 0);
lean_inc(v_val_201_);
lean_dec_ref(v_u_198_);
v___x_202_ = lean_apply_2(v_toAdd_199_, v_x_200_, v_val_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___redArg___lam__1(lean_object* v_u_203_, lean_object* v_toAdd_204_, lean_object* v_x_205_){
_start:
{
lean_object* v_neg_206_; lean_object* v___x_207_; 
v_neg_206_ = lean_ctor_get(v_u_203_, 1);
lean_inc(v_neg_206_);
lean_dec_ref(v_u_203_);
v___x_207_ = lean_apply_2(v_toAdd_204_, v_x_205_, v_neg_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___redArg(lean_object* v_inst_208_, lean_object* v_u_209_){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v_toAdd_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_221_; 
v___x_210_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_208_);
v___x_211_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_210_);
v_toAdd_212_ = lean_ctor_get(v___x_211_, 1);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_211_);
if (v_isSharedCheck_221_ == 0)
{
lean_object* v_unused_222_; 
v_unused_222_ = lean_ctor_get(v___x_211_, 0);
lean_dec(v_unused_222_);
v___x_214_ = v___x_211_;
v_isShared_215_ = v_isSharedCheck_221_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_toAdd_212_);
lean_dec(v___x_211_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_221_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___f_216_; lean_object* v___f_217_; lean_object* v___x_219_; 
lean_inc(v_toAdd_212_);
lean_inc_ref(v_u_209_);
v___f_216_ = lean_alloc_closure((void*)(lp_mathlib_AddUnits_addRight___redArg___lam__0), 3, 2);
lean_closure_set(v___f_216_, 0, v_u_209_);
lean_closure_set(v___f_216_, 1, v_toAdd_212_);
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_AddUnits_addRight___redArg___lam__1), 3, 2);
lean_closure_set(v___f_217_, 0, v_u_209_);
lean_closure_set(v___f_217_, 1, v_toAdd_212_);
if (v_isShared_215_ == 0)
{
lean_ctor_set(v___x_214_, 1, v___f_217_);
lean_ctor_set(v___x_214_, 0, v___f_216_);
v___x_219_ = v___x_214_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v___f_216_);
lean_ctor_set(v_reuseFailAlloc_220_, 1, v___f_217_);
v___x_219_ = v_reuseFailAlloc_220_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
return v___x_219_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___redArg___boxed(lean_object* v_inst_223_, lean_object* v_u_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_AddUnits_addRight___redArg(v_inst_223_, v_u_224_);
lean_dec_ref(v_inst_223_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight(lean_object* v_M_226_, lean_object* v_inst_227_, lean_object* v_u_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_AddUnits_addRight___redArg(v_inst_227_, v_u_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_addRight___boxed(lean_object* v_M_230_, lean_object* v_inst_231_, lean_object* v_u_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_AddUnits_addRight(v_M_230_, v_inst_231_, v_u_232_);
lean_dec_ref(v_inst_231_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft___redArg(lean_object* v_inst_234_, lean_object* v_a_235_){
_start:
{
lean_object* v_toMonoid_236_; lean_object* v___x_237_; lean_object* v_toFun_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v_toMonoid_236_ = lean_ctor_get(v_inst_234_, 0);
v___x_237_ = lp_mathlib_toUnits___redArg(v_inst_234_);
v_toFun_238_ = lean_ctor_get(v___x_237_, 0);
lean_inc(v_toFun_238_);
lean_dec_ref(v___x_237_);
v___x_239_ = lean_apply_1(v_toFun_238_, v_a_235_);
v___x_240_ = lp_mathlib_Units_mulLeft___redArg(v_toMonoid_236_, v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft___redArg___boxed(lean_object* v_inst_241_, lean_object* v_a_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_Equiv_mulLeft___redArg(v_inst_241_, v_a_242_);
lean_dec_ref(v_inst_241_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft(lean_object* v_G_244_, lean_object* v_inst_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_Equiv_mulLeft___redArg(v_inst_245_, v_a_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulLeft___boxed(lean_object* v_G_248_, lean_object* v_inst_249_, lean_object* v_a_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_Equiv_mulLeft(v_G_248_, v_inst_249_, v_a_250_);
lean_dec_ref(v_inst_249_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addLeft___redArg(lean_object* v_inst_252_, lean_object* v_a_253_){
_start:
{
lean_object* v_toAddMonoid_254_; lean_object* v___x_255_; lean_object* v_toFun_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v_toAddMonoid_254_ = lean_ctor_get(v_inst_252_, 0);
v___x_255_ = lp_mathlib_toAddUnits___redArg(v_inst_252_);
v_toFun_256_ = lean_ctor_get(v___x_255_, 0);
lean_inc(v_toFun_256_);
lean_dec_ref(v___x_255_);
v___x_257_ = lean_apply_1(v_toFun_256_, v_a_253_);
v___x_258_ = lp_mathlib_AddUnits_addLeft___redArg(v_toAddMonoid_254_, v___x_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addLeft___redArg___boxed(lean_object* v_inst_259_, lean_object* v_a_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_Equiv_addLeft___redArg(v_inst_259_, v_a_260_);
lean_dec_ref(v_inst_259_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addLeft(lean_object* v_G_262_, lean_object* v_inst_263_, lean_object* v_a_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_Equiv_addLeft___redArg(v_inst_263_, v_a_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addLeft___boxed(lean_object* v_G_266_, lean_object* v_inst_267_, lean_object* v_a_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_Equiv_addLeft(v_G_266_, v_inst_267_, v_a_268_);
lean_dec_ref(v_inst_267_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight___redArg(lean_object* v_inst_270_, lean_object* v_a_271_){
_start:
{
lean_object* v_toMonoid_272_; lean_object* v___x_273_; lean_object* v_toFun_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v_toMonoid_272_ = lean_ctor_get(v_inst_270_, 0);
v___x_273_ = lp_mathlib_toUnits___redArg(v_inst_270_);
v_toFun_274_ = lean_ctor_get(v___x_273_, 0);
lean_inc(v_toFun_274_);
lean_dec_ref(v___x_273_);
v___x_275_ = lean_apply_1(v_toFun_274_, v_a_271_);
v___x_276_ = lp_mathlib_Units_mulRight___redArg(v_toMonoid_272_, v___x_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight___redArg___boxed(lean_object* v_inst_277_, lean_object* v_a_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_Equiv_mulRight___redArg(v_inst_277_, v_a_278_);
lean_dec_ref(v_inst_277_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight(lean_object* v_G_280_, lean_object* v_inst_281_, lean_object* v_a_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_mathlib_Equiv_mulRight___redArg(v_inst_281_, v_a_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mulRight___boxed(lean_object* v_G_284_, lean_object* v_inst_285_, lean_object* v_a_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_Equiv_mulRight(v_G_284_, v_inst_285_, v_a_286_);
lean_dec_ref(v_inst_285_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addRight___redArg(lean_object* v_inst_288_, lean_object* v_a_289_){
_start:
{
lean_object* v_toAddMonoid_290_; lean_object* v___x_291_; lean_object* v_toFun_292_; lean_object* v___x_293_; lean_object* v___x_294_; 
v_toAddMonoid_290_ = lean_ctor_get(v_inst_288_, 0);
v___x_291_ = lp_mathlib_toAddUnits___redArg(v_inst_288_);
v_toFun_292_ = lean_ctor_get(v___x_291_, 0);
lean_inc(v_toFun_292_);
lean_dec_ref(v___x_291_);
v___x_293_ = lean_apply_1(v_toFun_292_, v_a_289_);
v___x_294_ = lp_mathlib_AddUnits_addRight___redArg(v_toAddMonoid_290_, v___x_293_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addRight___redArg___boxed(lean_object* v_inst_295_, lean_object* v_a_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_Equiv_addRight___redArg(v_inst_295_, v_a_296_);
lean_dec_ref(v_inst_295_);
return v_res_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addRight(lean_object* v_G_298_, lean_object* v_inst_299_, lean_object* v_a_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lp_mathlib_Equiv_addRight___redArg(v_inst_299_, v_a_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_addRight___boxed(lean_object* v_G_302_, lean_object* v_inst_303_, lean_object* v_a_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_Equiv_addRight(v_G_302_, v_inst_303_, v_a_304_);
lean_dec_ref(v_inst_303_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft___redArg___lam__0(lean_object* v_toDiv_306_, lean_object* v_a_307_, lean_object* v_b_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lean_apply_2(v_toDiv_306_, v_a_307_, v_b_308_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft___redArg___lam__1(lean_object* v_toInv_310_, lean_object* v_toMul_311_, lean_object* v_a_312_, lean_object* v_b_313_){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; 
v___x_314_ = lean_apply_1(v_toInv_310_, v_b_313_);
v___x_315_ = lean_apply_2(v_toMul_311_, v___x_314_, v_a_312_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft___redArg(lean_object* v_inst_316_, lean_object* v_a_317_){
_start:
{
lean_object* v_toMonoid_318_; lean_object* v_toDiv_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v_toMul_322_; lean_object* v___x_323_; lean_object* v_toInv_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_333_; 
v_toMonoid_318_ = lean_ctor_get(v_inst_316_, 0);
v_toDiv_319_ = lean_ctor_get(v_inst_316_, 2);
lean_inc(v_toDiv_319_);
v___x_320_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_318_);
v___x_321_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_320_);
v_toMul_322_ = lean_ctor_get(v___x_321_, 1);
lean_inc(v_toMul_322_);
lean_dec_ref(v___x_321_);
v___x_323_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_316_);
lean_dec_ref(v_inst_316_);
v_toInv_324_ = lean_ctor_get(v___x_323_, 1);
v_isSharedCheck_333_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_333_ == 0)
{
lean_object* v_unused_334_; 
v_unused_334_ = lean_ctor_get(v___x_323_, 0);
lean_dec(v_unused_334_);
v___x_326_ = v___x_323_;
v_isShared_327_ = v_isSharedCheck_333_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_toInv_324_);
lean_dec(v___x_323_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_333_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v___f_328_; lean_object* v___f_329_; lean_object* v___x_331_; 
lean_inc(v_a_317_);
v___f_328_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_divLeft___redArg___lam__0), 3, 2);
lean_closure_set(v___f_328_, 0, v_toDiv_319_);
lean_closure_set(v___f_328_, 1, v_a_317_);
v___f_329_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_divLeft___redArg___lam__1), 4, 3);
lean_closure_set(v___f_329_, 0, v_toInv_324_);
lean_closure_set(v___f_329_, 1, v_toMul_322_);
lean_closure_set(v___f_329_, 2, v_a_317_);
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 1, v___f_329_);
lean_ctor_set(v___x_326_, 0, v___f_328_);
v___x_331_ = v___x_326_;
goto v_reusejp_330_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v___f_328_);
lean_ctor_set(v_reuseFailAlloc_332_, 1, v___f_329_);
v___x_331_ = v_reuseFailAlloc_332_;
goto v_reusejp_330_;
}
v_reusejp_330_:
{
return v___x_331_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divLeft(lean_object* v_G_335_, lean_object* v_inst_336_, lean_object* v_a_337_){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lp_mathlib_Equiv_divLeft___redArg(v_inst_336_, v_a_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subLeft___redArg___lam__0(lean_object* v_toSub_339_, lean_object* v_a_340_, lean_object* v_b_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lean_apply_2(v_toSub_339_, v_a_340_, v_b_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subLeft___redArg___lam__1(lean_object* v_toNeg_343_, lean_object* v_toAdd_344_, lean_object* v_a_345_, lean_object* v_b_346_){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_347_ = lean_apply_1(v_toNeg_343_, v_b_346_);
v___x_348_ = lean_apply_2(v_toAdd_344_, v___x_347_, v_a_345_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subLeft___redArg(lean_object* v_inst_349_, lean_object* v_a_350_){
_start:
{
lean_object* v_toAddMonoid_351_; lean_object* v_toSub_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v_toAdd_355_; lean_object* v___x_356_; lean_object* v_toNeg_357_; lean_object* v___x_359_; uint8_t v_isShared_360_; uint8_t v_isSharedCheck_366_; 
v_toAddMonoid_351_ = lean_ctor_get(v_inst_349_, 0);
v_toSub_352_ = lean_ctor_get(v_inst_349_, 2);
lean_inc(v_toSub_352_);
v___x_353_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_351_);
v___x_354_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_353_);
v_toAdd_355_ = lean_ctor_get(v___x_354_, 1);
lean_inc(v_toAdd_355_);
lean_dec_ref(v___x_354_);
v___x_356_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_349_);
lean_dec_ref(v_inst_349_);
v_toNeg_357_ = lean_ctor_get(v___x_356_, 1);
v_isSharedCheck_366_ = !lean_is_exclusive(v___x_356_);
if (v_isSharedCheck_366_ == 0)
{
lean_object* v_unused_367_; 
v_unused_367_ = lean_ctor_get(v___x_356_, 0);
lean_dec(v_unused_367_);
v___x_359_ = v___x_356_;
v_isShared_360_ = v_isSharedCheck_366_;
goto v_resetjp_358_;
}
else
{
lean_inc(v_toNeg_357_);
lean_dec(v___x_356_);
v___x_359_ = lean_box(0);
v_isShared_360_ = v_isSharedCheck_366_;
goto v_resetjp_358_;
}
v_resetjp_358_:
{
lean_object* v___f_361_; lean_object* v___f_362_; lean_object* v___x_364_; 
lean_inc(v_a_350_);
v___f_361_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_subLeft___redArg___lam__0), 3, 2);
lean_closure_set(v___f_361_, 0, v_toSub_352_);
lean_closure_set(v___f_361_, 1, v_a_350_);
v___f_362_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_subLeft___redArg___lam__1), 4, 3);
lean_closure_set(v___f_362_, 0, v_toNeg_357_);
lean_closure_set(v___f_362_, 1, v_toAdd_355_);
lean_closure_set(v___f_362_, 2, v_a_350_);
if (v_isShared_360_ == 0)
{
lean_ctor_set(v___x_359_, 1, v___f_362_);
lean_ctor_set(v___x_359_, 0, v___f_361_);
v___x_364_ = v___x_359_;
goto v_reusejp_363_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v___f_361_);
lean_ctor_set(v_reuseFailAlloc_365_, 1, v___f_362_);
v___x_364_ = v_reuseFailAlloc_365_;
goto v_reusejp_363_;
}
v_reusejp_363_:
{
return v___x_364_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subLeft(lean_object* v_G_368_, lean_object* v_inst_369_, lean_object* v_a_370_){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lp_mathlib_Equiv_subLeft___redArg(v_inst_369_, v_a_370_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight___redArg___lam__0(lean_object* v_toDiv_372_, lean_object* v_a_373_, lean_object* v_b_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lean_apply_2(v_toDiv_372_, v_b_374_, v_a_373_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight___redArg___lam__1(lean_object* v_toMul_376_, lean_object* v_a_377_, lean_object* v_b_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lean_apply_2(v_toMul_376_, v_b_378_, v_a_377_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight___redArg(lean_object* v_inst_380_, lean_object* v_a_381_){
_start:
{
lean_object* v_toMonoid_382_; lean_object* v_toDiv_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v_toMul_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_395_; 
v_toMonoid_382_ = lean_ctor_get(v_inst_380_, 0);
lean_inc_ref(v_toMonoid_382_);
v_toDiv_383_ = lean_ctor_get(v_inst_380_, 2);
lean_inc(v_toDiv_383_);
lean_dec_ref(v_inst_380_);
v___x_384_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_382_);
lean_dec_ref(v_toMonoid_382_);
v___x_385_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_384_);
v_toMul_386_ = lean_ctor_get(v___x_385_, 1);
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_385_);
if (v_isSharedCheck_395_ == 0)
{
lean_object* v_unused_396_; 
v_unused_396_ = lean_ctor_get(v___x_385_, 0);
lean_dec(v_unused_396_);
v___x_388_ = v___x_385_;
v_isShared_389_ = v_isSharedCheck_395_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_toMul_386_);
lean_dec(v___x_385_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_395_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___f_390_; lean_object* v___f_391_; lean_object* v___x_393_; 
lean_inc(v_a_381_);
v___f_390_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_divRight___redArg___lam__0), 3, 2);
lean_closure_set(v___f_390_, 0, v_toDiv_383_);
lean_closure_set(v___f_390_, 1, v_a_381_);
v___f_391_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_divRight___redArg___lam__1), 3, 2);
lean_closure_set(v___f_391_, 0, v_toMul_386_);
lean_closure_set(v___f_391_, 1, v_a_381_);
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 1, v___f_391_);
lean_ctor_set(v___x_388_, 0, v___f_390_);
v___x_393_ = v___x_388_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v___f_390_);
lean_ctor_set(v_reuseFailAlloc_394_, 1, v___f_391_);
v___x_393_ = v_reuseFailAlloc_394_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
return v___x_393_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_divRight(lean_object* v_G_397_, lean_object* v_inst_398_, lean_object* v_a_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lp_mathlib_Equiv_divRight___redArg(v_inst_398_, v_a_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subRight___redArg___lam__0(lean_object* v_toSub_401_, lean_object* v_a_402_, lean_object* v_b_403_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lean_apply_2(v_toSub_401_, v_b_403_, v_a_402_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subRight___redArg___lam__1(lean_object* v_toAdd_405_, lean_object* v_a_406_, lean_object* v_b_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lean_apply_2(v_toAdd_405_, v_b_407_, v_a_406_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subRight___redArg(lean_object* v_inst_409_, lean_object* v_a_410_){
_start:
{
lean_object* v_toAddMonoid_411_; lean_object* v_toSub_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v_toAdd_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_424_; 
v_toAddMonoid_411_ = lean_ctor_get(v_inst_409_, 0);
lean_inc_ref(v_toAddMonoid_411_);
v_toSub_412_ = lean_ctor_get(v_inst_409_, 2);
lean_inc(v_toSub_412_);
lean_dec_ref(v_inst_409_);
v___x_413_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_411_);
lean_dec_ref(v_toAddMonoid_411_);
v___x_414_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_413_);
v_toAdd_415_ = lean_ctor_get(v___x_414_, 1);
v_isSharedCheck_424_ = !lean_is_exclusive(v___x_414_);
if (v_isSharedCheck_424_ == 0)
{
lean_object* v_unused_425_; 
v_unused_425_ = lean_ctor_get(v___x_414_, 0);
lean_dec(v_unused_425_);
v___x_417_ = v___x_414_;
v_isShared_418_ = v_isSharedCheck_424_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_toAdd_415_);
lean_dec(v___x_414_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_424_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v___f_419_; lean_object* v___f_420_; lean_object* v___x_422_; 
lean_inc(v_a_410_);
v___f_419_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_subRight___redArg___lam__0), 3, 2);
lean_closure_set(v___f_419_, 0, v_toSub_412_);
lean_closure_set(v___f_419_, 1, v_a_410_);
v___f_420_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_subRight___redArg___lam__1), 3, 2);
lean_closure_set(v___f_420_, 0, v_toAdd_415_);
lean_closure_set(v___f_420_, 1, v_a_410_);
if (v_isShared_418_ == 0)
{
lean_ctor_set(v___x_417_, 1, v___f_420_);
lean_ctor_set(v___x_417_, 0, v___f_419_);
v___x_422_ = v___x_417_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v___f_419_);
lean_ctor_set(v_reuseFailAlloc_423_, 1, v___f_420_);
v___x_422_ = v_reuseFailAlloc_423_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
return v___x_422_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subRight(lean_object* v_G_426_, lean_object* v_inst_427_, lean_object* v_a_428_){
_start:
{
lean_object* v___x_429_; 
v___x_429_ = lp_mathlib_Equiv_subRight___redArg(v_inst_427_, v_a_428_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivProdSubtype___lam__0(lean_object* v_u_430_){
_start:
{
lean_object* v_val_431_; lean_object* v_inv_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_439_; 
v_val_431_ = lean_ctor_get(v_u_430_, 0);
v_inv_432_ = lean_ctor_get(v_u_430_, 1);
v_isSharedCheck_439_ = !lean_is_exclusive(v_u_430_);
if (v_isSharedCheck_439_ == 0)
{
v___x_434_ = v_u_430_;
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_inv_432_);
lean_inc(v_val_431_);
lean_dec(v_u_430_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_437_; 
if (v_isShared_435_ == 0)
{
v___x_437_ = v___x_434_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_val_431_);
lean_ctor_set(v_reuseFailAlloc_438_, 1, v_inv_432_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivProdSubtype___lam__1(lean_object* v_p_440_){
_start:
{
lean_object* v_fst_441_; lean_object* v_snd_442_; lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_449_; 
v_fst_441_ = lean_ctor_get(v_p_440_, 0);
v_snd_442_ = lean_ctor_get(v_p_440_, 1);
v_isSharedCheck_449_ = !lean_is_exclusive(v_p_440_);
if (v_isSharedCheck_449_ == 0)
{
v___x_444_ = v_p_440_;
v_isShared_445_ = v_isSharedCheck_449_;
goto v_resetjp_443_;
}
else
{
lean_inc(v_snd_442_);
lean_inc(v_fst_441_);
lean_dec(v_p_440_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_449_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
lean_object* v___x_447_; 
if (v_isShared_445_ == 0)
{
v___x_447_ = v___x_444_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v_fst_441_);
lean_ctor_set(v_reuseFailAlloc_448_, 1, v_snd_442_);
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
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivProdSubtype(lean_object* v_00_u03b1_455_, lean_object* v_inst_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = ((lean_object*)(lp_mathlib_unitsEquivProdSubtype___closed__2));
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsEquivProdSubtype___boxed(lean_object* v_00_u03b1_458_, lean_object* v_inst_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_mathlib_unitsEquivProdSubtype(v_00_u03b1_458_, v_inst_459_);
lean_dec_ref(v_inst_459_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv___redArg(lean_object* v_inst_461_){
_start:
{
lean_object* v___x_462_; lean_object* v_toInv_463_; lean_object* v___x_465_; uint8_t v_isShared_466_; uint8_t v_isSharedCheck_470_; 
v___x_462_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_461_);
v_toInv_463_ = lean_ctor_get(v___x_462_, 1);
v_isSharedCheck_470_ = !lean_is_exclusive(v___x_462_);
if (v_isSharedCheck_470_ == 0)
{
lean_object* v_unused_471_; 
v_unused_471_ = lean_ctor_get(v___x_462_, 0);
lean_dec(v_unused_471_);
v___x_465_ = v___x_462_;
v_isShared_466_ = v_isSharedCheck_470_;
goto v_resetjp_464_;
}
else
{
lean_inc(v_toInv_463_);
lean_dec(v___x_462_);
v___x_465_ = lean_box(0);
v_isShared_466_ = v_isSharedCheck_470_;
goto v_resetjp_464_;
}
v_resetjp_464_:
{
lean_object* v___x_468_; 
lean_inc(v_toInv_463_);
if (v_isShared_466_ == 0)
{
lean_ctor_set(v___x_465_, 0, v_toInv_463_);
v___x_468_ = v___x_465_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v_toInv_463_);
lean_ctor_set(v_reuseFailAlloc_469_, 1, v_toInv_463_);
v___x_468_ = v_reuseFailAlloc_469_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
return v___x_468_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv___redArg___boxed(lean_object* v_inst_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib_MulEquiv_inv___redArg(v_inst_472_);
lean_dec_ref(v_inst_472_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv(lean_object* v_G_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_MulEquiv_inv___redArg(v_inst_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv___boxed(lean_object* v_G_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v_res_479_; 
v_res_479_ = lp_mathlib_MulEquiv_inv(v_G_477_, v_inst_478_);
lean_dec_ref(v_inst_478_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg___redArg(lean_object* v_inst_480_){
_start:
{
lean_object* v___x_481_; lean_object* v_toNeg_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_489_; 
v___x_481_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_480_);
v_toNeg_482_ = lean_ctor_get(v___x_481_, 1);
v_isSharedCheck_489_ = !lean_is_exclusive(v___x_481_);
if (v_isSharedCheck_489_ == 0)
{
lean_object* v_unused_490_; 
v_unused_490_ = lean_ctor_get(v___x_481_, 0);
lean_dec(v_unused_490_);
v___x_484_ = v___x_481_;
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_toNeg_482_);
lean_dec(v___x_481_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_487_; 
lean_inc(v_toNeg_482_);
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 0, v_toNeg_482_);
v___x_487_ = v___x_484_;
goto v_reusejp_486_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v_toNeg_482_);
lean_ctor_set(v_reuseFailAlloc_488_, 1, v_toNeg_482_);
v___x_487_ = v_reuseFailAlloc_488_;
goto v_reusejp_486_;
}
v_reusejp_486_:
{
return v___x_487_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg___redArg___boxed(lean_object* v_inst_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib_AddEquiv_neg___redArg(v_inst_491_);
lean_dec_ref(v_inst_491_);
return v_res_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg(lean_object* v_G_493_, lean_object* v_inst_494_){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lp_mathlib_AddEquiv_neg___redArg(v_inst_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg___boxed(lean_object* v_G_496_, lean_object* v_inst_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib_AddEquiv_neg(v_G_496_, v_inst_497_);
lean_dec_ref(v_inst_497_);
return v_res_498_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Units_Equiv(builtin);
}
#ifdef __cplusplus
}
#endif
