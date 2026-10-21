// Lean compiler output
// Module: Mathlib.Algebra.Group.TypeTags.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Torsion public import Mathlib.Algebra.Notation.Pi.Basic public import Mathlib.Data.FunLike.Basic public import Mathlib.Logic.Function.Iterate public import Mathlib.Logic.Equiv.Defs
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Additive_ofMul___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Additive_ofMul___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Additive_ofMul___closed__0 = (const lean_object*)&lp_mathlib_Additive_ofMul___closed__0_value;
static const lean_ctor_object lp_mathlib_Additive_ofMul___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Additive_ofMul___closed__0_value),((lean_object*)&lp_mathlib_Additive_ofMul___closed__0_value)}};
static const lean_object* lp_mathlib_Additive_ofMul___closed__1 = (const lean_object*)&lp_mathlib_Additive_ofMul___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul(lean_object*);
static lean_once_cell_t lp_mathlib_Additive_toMul___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Additive_toMul___closed__0;
static lean_once_cell_t lp_mathlib_Additive_toMul___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Additive_toMul___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Additive_toMul(lean_object*);
static lean_once_cell_t lp_mathlib_Additive_rec___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Additive_rec___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Additive_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_rec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
static lean_once_cell_t lp_mathlib_Multiplicative_toAdd___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Multiplicative_toAdd___closed__0;
static lean_once_cell_t lp_mathlib_Multiplicative_toAdd___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Multiplicative_toAdd___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
static lean_once_cell_t lp_mathlib_Multiplicative_rec___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Multiplicative_rec___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAdditive(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMultiplicative(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instUniqueAdditive___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instUniqueAdditive___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAdditive(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instUniqueMultiplicative___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instUniqueMultiplicative___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instUniqueMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueMultiplicative(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqMultiplicative___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqMultiplicative___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqMultiplicative(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqMultiplicative___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqAdditive___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqAdditive___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqAdditive(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqAdditive___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Additive_add___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Additive_add___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Additive_add___redArg___closed__0 = (const lean_object*)&lp_mathlib_Additive_add___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_semigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAdditiveOfOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instZeroAdditiveOfOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMultiplicativeOfZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOneMultiplicativeOfZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_instAddCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_instAddCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_instCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_instCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_sub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_div___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_involutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_involutiveNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_involutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_involutiveInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_group___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_group(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul___lam__0(lean_object* v_x_1_){
_start:
{
lean_inc(v_x_1_);
return v_x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul___lam__0___boxed(lean_object* v_x_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Additive_ofMul___lam__0(v_x_2_);
lean_dec(v_x_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_ofMul(lean_object* v_00_u03b1_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = ((lean_object*)(lp_mathlib_Additive_ofMul___closed__1));
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_Additive_toMul___closed__0(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_9_;
}
}
static lean_object* _init_lp_mathlib_Additive_toMul___closed__1(void){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lean_obj_once(&lp_mathlib_Additive_toMul___closed__0, &lp_mathlib_Additive_toMul___closed__0_once, _init_lp_mathlib_Additive_toMul___closed__0);
v___x_11_ = lp_mathlib_Equiv_symm___redArg(v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_toMul(lean_object* v_00_u03b1_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_obj_once(&lp_mathlib_Additive_toMul___closed__1, &lp_mathlib_Additive_toMul___closed__1_once, _init_lp_mathlib_Additive_toMul___closed__1);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_Additive_rec___redArg___closed__0(void){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_rec___redArg(lean_object* v_ofMul_15_, lean_object* v_a_16_){
_start:
{
lean_object* v___x_17_; lean_object* v_toFun_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_17_ = lean_obj_once(&lp_mathlib_Additive_rec___redArg___closed__0, &lp_mathlib_Additive_rec___redArg___closed__0_once, _init_lp_mathlib_Additive_rec___redArg___closed__0);
v_toFun_18_ = lean_ctor_get(v___x_17_, 0);
lean_inc(v_toFun_18_);
v___x_19_ = lean_apply_1(v_toFun_18_, v_a_16_);
v___x_20_ = lean_apply_1(v_ofMul_15_, v___x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_rec(lean_object* v_00_u03b1_21_, lean_object* v_motive_22_, lean_object* v_ofMul_23_, lean_object* v_a_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_Additive_rec___redArg(v_ofMul_23_, v_a_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object* v_00_u03b1_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = ((lean_object*)(lp_mathlib_Additive_ofMul___closed__1));
return v___x_27_;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_toAdd___closed__0(void){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_toAdd___closed__1(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Multiplicative_toAdd___closed__0, &lp_mathlib_Multiplicative_toAdd___closed__0_once, _init_lp_mathlib_Multiplicative_toAdd___closed__0);
v___x_30_ = lp_mathlib_Equiv_symm___redArg(v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_toAdd(lean_object* v_00_u03b1_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_obj_once(&lp_mathlib_Multiplicative_toAdd___closed__1, &lp_mathlib_Multiplicative_toAdd___closed__1_once, _init_lp_mathlib_Multiplicative_toAdd___closed__1);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_rec___redArg___closed__0(void){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rec___redArg(lean_object* v_ofAdd_34_, lean_object* v_a_35_){
_start:
{
lean_object* v___x_36_; lean_object* v_toFun_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_36_ = lean_obj_once(&lp_mathlib_Multiplicative_rec___redArg___closed__0, &lp_mathlib_Multiplicative_rec___redArg___closed__0_once, _init_lp_mathlib_Multiplicative_rec___redArg___closed__0);
v_toFun_37_ = lean_ctor_get(v___x_36_, 0);
lean_inc(v_toFun_37_);
v___x_38_ = lean_apply_1(v_toFun_37_, v_a_35_);
v___x_39_ = lean_apply_1(v_ofAdd_34_, v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rec(lean_object* v_00_u03b1_40_, lean_object* v_motive_41_, lean_object* v_ofAdd_42_, lean_object* v_a_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_Multiplicative_rec___redArg(v_ofAdd_42_, v_a_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAdditive___redArg(lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; lean_object* v_toFun_47_; lean_object* v___x_48_; 
v___x_46_ = lean_obj_once(&lp_mathlib_Additive_toMul___closed__0, &lp_mathlib_Additive_toMul___closed__0_once, _init_lp_mathlib_Additive_toMul___closed__0);
v_toFun_47_ = lean_ctor_get(v___x_46_, 0);
lean_inc(v_toFun_47_);
v___x_48_ = lean_apply_1(v_toFun_47_, v_inst_45_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAdditive(lean_object* v_00_u03b1_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_instInhabitedAdditive___redArg(v_inst_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMultiplicative___redArg(lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; lean_object* v_toFun_54_; lean_object* v___x_55_; 
v___x_53_ = lean_obj_once(&lp_mathlib_Multiplicative_toAdd___closed__0, &lp_mathlib_Multiplicative_toAdd___closed__0_once, _init_lp_mathlib_Multiplicative_toAdd___closed__0);
v_toFun_54_ = lean_ctor_get(v___x_53_, 0);
lean_inc(v_toFun_54_);
v___x_55_ = lean_apply_1(v_toFun_54_, v_inst_52_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedMultiplicative(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_instInhabitedMultiplicative___redArg(v_inst_57_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib_instUniqueAdditive___redArg___closed__0(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_obj_once(&lp_mathlib_Additive_rec___redArg___closed__0, &lp_mathlib_Additive_rec___redArg___closed__0_once, _init_lp_mathlib_Additive_rec___redArg___closed__0);
v___x_60_ = lp_mathlib_Equiv_symm___redArg(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAdditive___redArg(lean_object* v_inst_61_){
_start:
{
lean_object* v___x_62_; lean_object* v_toFun_63_; lean_object* v___x_64_; 
v___x_62_ = lean_obj_once(&lp_mathlib_instUniqueAdditive___redArg___closed__0, &lp_mathlib_instUniqueAdditive___redArg___closed__0_once, _init_lp_mathlib_instUniqueAdditive___redArg___closed__0);
v_toFun_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc(v_toFun_63_);
v___x_64_ = lean_apply_1(v_toFun_63_, v_inst_61_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueAdditive(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_instUniqueAdditive___redArg(v_inst_66_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_instUniqueMultiplicative___redArg___closed__0(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = lean_obj_once(&lp_mathlib_Multiplicative_rec___redArg___closed__0, &lp_mathlib_Multiplicative_rec___redArg___closed__0_once, _init_lp_mathlib_Multiplicative_rec___redArg___closed__0);
v___x_69_ = lp_mathlib_Equiv_symm___redArg(v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueMultiplicative___redArg(lean_object* v_inst_70_){
_start:
{
lean_object* v___x_71_; lean_object* v_toFun_72_; lean_object* v___x_73_; 
v___x_71_ = lean_obj_once(&lp_mathlib_instUniqueMultiplicative___redArg___closed__0, &lp_mathlib_instUniqueMultiplicative___redArg___closed__0_once, _init_lp_mathlib_instUniqueMultiplicative___redArg___closed__0);
v_toFun_72_ = lean_ctor_get(v___x_71_, 0);
lean_inc(v_toFun_72_);
v___x_73_ = lean_apply_1(v_toFun_72_, v_inst_70_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueMultiplicative(lean_object* v_00_u03b1_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lp_mathlib_instUniqueMultiplicative___redArg(v_inst_75_);
return v___x_76_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqMultiplicative___redArg(lean_object* v_h_77_, lean_object* v_a_78_, lean_object* v_b_79_){
_start:
{
lean_object* v___x_80_; uint8_t v___x_81_; 
v___x_80_ = lean_apply_2(v_h_77_, v_a_78_, v_b_79_);
v___x_81_ = lean_unbox(v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqMultiplicative___redArg___boxed(lean_object* v_h_82_, lean_object* v_a_83_, lean_object* v_b_84_){
_start:
{
uint8_t v_res_85_; lean_object* v_r_86_; 
v_res_85_ = lp_mathlib_instDecidableEqMultiplicative___redArg(v_h_82_, v_a_83_, v_b_84_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqMultiplicative(lean_object* v_00_u03b1_87_, lean_object* v_h_88_, lean_object* v_a_89_, lean_object* v_b_90_){
_start:
{
lean_object* v___x_91_; uint8_t v___x_92_; 
v___x_91_ = lean_apply_2(v_h_88_, v_a_89_, v_b_90_);
v___x_92_ = lean_unbox(v___x_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqMultiplicative___boxed(lean_object* v_00_u03b1_93_, lean_object* v_h_94_, lean_object* v_a_95_, lean_object* v_b_96_){
_start:
{
uint8_t v_res_97_; lean_object* v_r_98_; 
v_res_97_ = lp_mathlib_instDecidableEqMultiplicative(v_00_u03b1_93_, v_h_94_, v_a_95_, v_b_96_);
v_r_98_ = lean_box(v_res_97_);
return v_r_98_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqAdditive___redArg(lean_object* v_h_99_, lean_object* v_a_100_, lean_object* v_b_101_){
_start:
{
lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_102_ = lean_apply_2(v_h_99_, v_a_100_, v_b_101_);
v___x_103_ = lean_unbox(v___x_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqAdditive___redArg___boxed(lean_object* v_h_104_, lean_object* v_a_105_, lean_object* v_b_106_){
_start:
{
uint8_t v_res_107_; lean_object* v_r_108_; 
v_res_107_ = lp_mathlib_instDecidableEqAdditive___redArg(v_h_104_, v_a_105_, v_b_106_);
v_r_108_ = lean_box(v_res_107_);
return v_r_108_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqAdditive(lean_object* v_00_u03b1_109_, lean_object* v_h_110_, lean_object* v_a_111_, lean_object* v_b_112_){
_start:
{
lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_113_ = lean_apply_2(v_h_110_, v_a_111_, v_b_112_);
v___x_114_ = lean_unbox(v___x_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqAdditive___boxed(lean_object* v_00_u03b1_115_, lean_object* v_h_116_, lean_object* v_a_117_, lean_object* v_b_118_){
_start:
{
uint8_t v_res_119_; lean_object* v_r_120_; 
v_res_119_ = lp_mathlib_instDecidableEqAdditive(v_00_u03b1_115_, v_h_116_, v_a_117_, v_b_118_);
v_r_120_ = lean_box(v_res_119_);
return v_r_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg___lam__0(lean_object* v_self_121_, lean_object* v___y_122_){
_start:
{
lean_object* v_toFun_123_; lean_object* v___x_124_; 
v_toFun_123_ = lean_ctor_get(v_self_121_, 0);
lean_inc(v_toFun_123_);
lean_dec_ref(v_self_121_);
v___x_124_ = lean_apply_1(v_toFun_123_, v___y_122_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg___lam__1(lean_object* v___f_125_, lean_object* v_inst_126_, lean_object* v_x_127_, lean_object* v_y_128_){
_start:
{
lean_object* v___x_129_; lean_object* v_toFun_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_129_ = lean_obj_once(&lp_mathlib_Additive_toMul___closed__0, &lp_mathlib_Additive_toMul___closed__0_once, _init_lp_mathlib_Additive_toMul___closed__0);
v_toFun_130_ = lean_ctor_get(v___x_129_, 0);
v___x_131_ = lean_obj_once(&lp_mathlib_Additive_rec___redArg___closed__0, &lp_mathlib_Additive_rec___redArg___closed__0_once, _init_lp_mathlib_Additive_rec___redArg___closed__0);
lean_inc(v___f_125_);
v___x_132_ = lean_apply_2(v___f_125_, v___x_131_, v_x_127_);
v___x_133_ = lean_apply_2(v___f_125_, v___x_131_, v_y_128_);
v___x_134_ = lean_apply_2(v_inst_126_, v___x_132_, v___x_133_);
lean_inc(v_toFun_130_);
v___x_135_ = lean_apply_1(v_toFun_130_, v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_add___redArg(lean_object* v_inst_137_){
_start:
{
lean_object* v___f_138_; lean_object* v___f_139_; 
v___f_138_ = ((lean_object*)(lp_mathlib_Additive_add___redArg___closed__0));
v___f_139_ = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__1), 4, 2);
lean_closure_set(v___f_139_, 0, v___f_138_);
lean_closure_set(v___f_139_, 1, v_inst_137_);
return v___f_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_add(lean_object* v_00_u03b1_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_Additive_add___redArg(v_inst_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul___redArg___lam__1(lean_object* v___f_143_, lean_object* v_inst_144_, lean_object* v_x_145_, lean_object* v_y_146_){
_start:
{
lean_object* v___x_147_; lean_object* v_toFun_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_147_ = lean_obj_once(&lp_mathlib_Multiplicative_toAdd___closed__0, &lp_mathlib_Multiplicative_toAdd___closed__0_once, _init_lp_mathlib_Multiplicative_toAdd___closed__0);
v_toFun_148_ = lean_ctor_get(v___x_147_, 0);
v___x_149_ = lean_obj_once(&lp_mathlib_Multiplicative_rec___redArg___closed__0, &lp_mathlib_Multiplicative_rec___redArg___closed__0_once, _init_lp_mathlib_Multiplicative_rec___redArg___closed__0);
lean_inc(v___f_143_);
v___x_150_ = lean_apply_2(v___f_143_, v___x_149_, v_x_145_);
v___x_151_ = lean_apply_2(v___f_143_, v___x_149_, v_y_146_);
v___x_152_ = lean_apply_2(v_inst_144_, v___x_150_, v___x_151_);
lean_inc(v_toFun_148_);
v___x_153_ = lean_apply_1(v_toFun_148_, v___x_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul___redArg(lean_object* v_inst_154_){
_start:
{
lean_object* v___f_155_; lean_object* v___f_156_; 
v___f_155_ = ((lean_object*)(lp_mathlib_Additive_add___redArg___closed__0));
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__1), 4, 2);
lean_closure_set(v___f_156_, 0, v___f_155_);
lean_closure_set(v___f_156_, 1, v_inst_154_);
return v___f_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mul(lean_object* v_00_u03b1_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addSemigroup___redArg(lean_object* v_inst_160_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = lp_mathlib_Additive_add___redArg(v_inst_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addSemigroup(lean_object* v_00_u03b1_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_Additive_add___redArg(v_inst_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_semigroup___redArg(lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_semigroup(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommSemigroup___redArg(lean_object* v_inst_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_mathlib_Additive_add___redArg(v_inst_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommSemigroup(lean_object* v_00_u03b1_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_Additive_add___redArg(v_inst_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commSemigroup___redArg(lean_object* v_inst_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commSemigroup(lean_object* v_00_u03b1_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelSemigroup___redArg(lean_object* v_inst_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Additive_add___redArg(v_inst_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelSemigroup(lean_object* v_00_u03b1_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_Additive_add___redArg(v_inst_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelSemigroup___redArg(lean_object* v_inst_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelSemigroup(lean_object* v_00_u03b1_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelSemigroup___redArg(lean_object* v_inst_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_mathlib_Additive_add___redArg(v_inst_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelSemigroup(lean_object* v_00_u03b1_192_, lean_object* v_inst_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_Additive_add___redArg(v_inst_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelSemigroup___redArg(lean_object* v_inst_195_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelSemigroup(lean_object* v_00_u03b1_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAdditiveOfOne___redArg(lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; lean_object* v_toFun_202_; lean_object* v___x_203_; 
v___x_201_ = lean_obj_once(&lp_mathlib_Additive_toMul___closed__0, &lp_mathlib_Additive_toMul___closed__0_once, _init_lp_mathlib_Additive_toMul___closed__0);
v_toFun_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc(v_toFun_202_);
v___x_203_ = lean_apply_1(v_toFun_202_, v_inst_200_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instZeroAdditiveOfOne(lean_object* v_00_u03b1_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lp_mathlib_instZeroAdditiveOfOne___redArg(v_inst_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMultiplicativeOfZero___redArg(lean_object* v_inst_207_){
_start:
{
lean_object* v___x_208_; lean_object* v_toFun_209_; lean_object* v___x_210_; 
v___x_208_ = lean_obj_once(&lp_mathlib_Multiplicative_toAdd___closed__0, &lp_mathlib_Multiplicative_toAdd___closed__0_once, _init_lp_mathlib_Multiplicative_toAdd___closed__0);
v_toFun_209_ = lean_ctor_get(v___x_208_, 0);
lean_inc(v_toFun_209_);
v___x_210_ = lean_apply_1(v_toFun_209_, v_inst_207_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOneMultiplicativeOfZero(lean_object* v_00_u03b1_211_, lean_object* v_inst_212_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lp_mathlib_instOneMultiplicativeOfZero___redArg(v_inst_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addZeroClass___redArg(lean_object* v_inst_214_){
_start:
{
lean_object* v___x_215_; lean_object* v_toOne_216_; lean_object* v_toMul_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_226_; 
v___x_215_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_214_);
v_toOne_216_ = lean_ctor_get(v___x_215_, 0);
v_toMul_217_ = lean_ctor_get(v___x_215_, 1);
v_isSharedCheck_226_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_226_ == 0)
{
v___x_219_ = v___x_215_;
v_isShared_220_ = v_isSharedCheck_226_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_toMul_217_);
lean_inc(v_toOne_216_);
lean_dec(v___x_215_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_226_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_224_; 
v___x_221_ = lp_mathlib_instZeroAdditiveOfOne___redArg(v_toOne_216_);
v___x_222_ = lp_mathlib_Additive_add___redArg(v_toMul_217_);
if (v_isShared_220_ == 0)
{
lean_ctor_set(v___x_219_, 1, v___x_222_);
lean_ctor_set(v___x_219_, 0, v___x_221_);
v___x_224_ = v___x_219_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v___x_221_);
lean_ctor_set(v_reuseFailAlloc_225_, 1, v___x_222_);
v___x_224_ = v_reuseFailAlloc_225_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
return v___x_224_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addZeroClass(lean_object* v_00_u03b1_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_Additive_addZeroClass___redArg(v_inst_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulOneClass___redArg(lean_object* v_inst_230_){
_start:
{
lean_object* v___x_231_; lean_object* v_toZero_232_; lean_object* v_toAdd_233_; lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_242_; 
v___x_231_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_230_);
v_toZero_232_ = lean_ctor_get(v___x_231_, 0);
v_toAdd_233_ = lean_ctor_get(v___x_231_, 1);
v_isSharedCheck_242_ = !lean_is_exclusive(v___x_231_);
if (v_isSharedCheck_242_ == 0)
{
v___x_235_ = v___x_231_;
v_isShared_236_ = v_isSharedCheck_242_;
goto v_resetjp_234_;
}
else
{
lean_inc(v_toAdd_233_);
lean_inc(v_toZero_232_);
lean_dec(v___x_231_);
v___x_235_ = lean_box(0);
v_isShared_236_ = v_isSharedCheck_242_;
goto v_resetjp_234_;
}
v_resetjp_234_:
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_240_; 
v___x_237_ = lp_mathlib_instOneMultiplicativeOfZero___redArg(v_toZero_232_);
v___x_238_ = lp_mathlib_Multiplicative_mul___redArg(v_toAdd_233_);
if (v_isShared_236_ == 0)
{
lean_ctor_set(v___x_235_, 1, v___x_238_);
lean_ctor_set(v___x_235_, 0, v___x_237_);
v___x_240_ = v___x_235_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v___x_237_);
lean_ctor_set(v_reuseFailAlloc_241_, 1, v___x_238_);
v___x_240_ = v_reuseFailAlloc_241_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
return v___x_240_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_mulOneClass(lean_object* v_00_u03b1_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lp_mathlib_Multiplicative_mulOneClass___redArg(v_inst_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid___redArg___lam__0(lean_object* v_toNPow_246_, lean_object* v_n_247_, lean_object* v_a_248_){
_start:
{
lean_object* v___x_249_; lean_object* v_toFun_250_; lean_object* v___x_251_; lean_object* v_toFun_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_249_ = lean_obj_once(&lp_mathlib_Additive_rec___redArg___closed__0, &lp_mathlib_Additive_rec___redArg___closed__0_once, _init_lp_mathlib_Additive_rec___redArg___closed__0);
v_toFun_250_ = lean_ctor_get(v___x_249_, 0);
v___x_251_ = lean_obj_once(&lp_mathlib_Additive_toMul___closed__0, &lp_mathlib_Additive_toMul___closed__0_once, _init_lp_mathlib_Additive_toMul___closed__0);
v_toFun_252_ = lean_ctor_get(v___x_251_, 0);
lean_inc(v_toFun_250_);
v___x_253_ = lean_apply_1(v_toFun_250_, v_a_248_);
v___x_254_ = lean_apply_2(v_toNPow_246_, v_n_247_, v___x_253_);
lean_inc(v_toFun_252_);
v___x_255_ = lean_apply_1(v_toFun_252_, v___x_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid___redArg(lean_object* v_h_256_){
_start:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v_toOne_259_; lean_object* v_toMul_260_; lean_object* v_toNPow_261_; lean_object* v___x_263_; uint8_t v_isShared_264_; uint8_t v_isSharedCheck_271_; 
v___x_257_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_h_256_);
v___x_258_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_257_);
v_toOne_259_ = lean_ctor_get(v___x_258_, 0);
lean_inc(v_toOne_259_);
v_toMul_260_ = lean_ctor_get(v___x_258_, 1);
lean_inc(v_toMul_260_);
lean_dec_ref(v___x_258_);
v_toNPow_261_ = lean_ctor_get(v_h_256_, 2);
v_isSharedCheck_271_ = !lean_is_exclusive(v_h_256_);
if (v_isSharedCheck_271_ == 0)
{
lean_object* v_unused_272_; lean_object* v_unused_273_; 
v_unused_272_ = lean_ctor_get(v_h_256_, 1);
lean_dec(v_unused_272_);
v_unused_273_ = lean_ctor_get(v_h_256_, 0);
lean_dec(v_unused_273_);
v___x_263_ = v_h_256_;
v_isShared_264_ = v_isSharedCheck_271_;
goto v_resetjp_262_;
}
else
{
lean_inc(v_toNPow_261_);
lean_dec(v_h_256_);
v___x_263_ = lean_box(0);
v_isShared_264_ = v_isSharedCheck_271_;
goto v_resetjp_262_;
}
v_resetjp_262_:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___f_267_; lean_object* v___x_269_; 
v___x_265_ = lp_mathlib_instZeroAdditiveOfOne___redArg(v_toOne_259_);
v___x_266_ = lp_mathlib_Additive_add___redArg(v_toMul_260_);
v___f_267_ = lean_alloc_closure((void*)(lp_mathlib_Additive_addMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_267_, 0, v_toNPow_261_);
if (v_isShared_264_ == 0)
{
lean_ctor_set(v___x_263_, 2, v___f_267_);
lean_ctor_set(v___x_263_, 1, v___x_266_);
lean_ctor_set(v___x_263_, 0, v___x_265_);
v___x_269_ = v___x_263_;
goto v_reusejp_268_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v___x_265_);
lean_ctor_set(v_reuseFailAlloc_270_, 1, v___x_266_);
lean_ctor_set(v_reuseFailAlloc_270_, 2, v___f_267_);
v___x_269_ = v_reuseFailAlloc_270_;
goto v_reusejp_268_;
}
v_reusejp_268_:
{
return v___x_269_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addMonoid(lean_object* v_00_u03b1_274_, lean_object* v_h_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_mathlib_Additive_addMonoid___redArg(v_h_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid___redArg___lam__0(lean_object* v_toNSMul_277_, lean_object* v_n_278_, lean_object* v_a_279_){
_start:
{
lean_object* v___x_280_; lean_object* v_toFun_281_; lean_object* v___x_282_; lean_object* v_toFun_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_280_ = lean_obj_once(&lp_mathlib_Multiplicative_rec___redArg___closed__0, &lp_mathlib_Multiplicative_rec___redArg___closed__0_once, _init_lp_mathlib_Multiplicative_rec___redArg___closed__0);
v_toFun_281_ = lean_ctor_get(v___x_280_, 0);
v___x_282_ = lean_obj_once(&lp_mathlib_Multiplicative_toAdd___closed__0, &lp_mathlib_Multiplicative_toAdd___closed__0_once, _init_lp_mathlib_Multiplicative_toAdd___closed__0);
v_toFun_283_ = lean_ctor_get(v___x_282_, 0);
lean_inc(v_toFun_281_);
v___x_284_ = lean_apply_1(v_toFun_281_, v_a_279_);
v___x_285_ = lean_apply_2(v_toNSMul_277_, v_n_278_, v___x_284_);
lean_inc(v_toFun_283_);
v___x_286_ = lean_apply_1(v_toFun_283_, v___x_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid___redArg(lean_object* v_h_287_){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v_toZero_290_; lean_object* v_toAdd_291_; lean_object* v_toNSMul_292_; lean_object* v___x_294_; uint8_t v_isShared_295_; uint8_t v_isSharedCheck_302_; 
v___x_288_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_h_287_);
v___x_289_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_288_);
v_toZero_290_ = lean_ctor_get(v___x_289_, 0);
lean_inc(v_toZero_290_);
lean_dec_ref(v___x_289_);
v_toAdd_291_ = lean_ctor_get(v_h_287_, 1);
v_toNSMul_292_ = lean_ctor_get(v_h_287_, 2);
v_isSharedCheck_302_ = !lean_is_exclusive(v_h_287_);
if (v_isSharedCheck_302_ == 0)
{
lean_object* v_unused_303_; 
v_unused_303_ = lean_ctor_get(v_h_287_, 0);
lean_dec(v_unused_303_);
v___x_294_ = v_h_287_;
v_isShared_295_ = v_isSharedCheck_302_;
goto v_resetjp_293_;
}
else
{
lean_inc(v_toNSMul_292_);
lean_inc(v_toAdd_291_);
lean_dec(v_h_287_);
v___x_294_ = lean_box(0);
v_isShared_295_ = v_isSharedCheck_302_;
goto v_resetjp_293_;
}
v_resetjp_293_:
{
lean_object* v___x_296_; lean_object* v___f_297_; lean_object* v___x_298_; lean_object* v___x_300_; 
v___x_296_ = lp_mathlib_instOneMultiplicativeOfZero___redArg(v_toZero_290_);
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_297_, 0, v_toNSMul_292_);
v___x_298_ = lp_mathlib_Multiplicative_mul___redArg(v_toAdd_291_);
if (v_isShared_295_ == 0)
{
lean_ctor_set(v___x_294_, 2, v___f_297_);
lean_ctor_set(v___x_294_, 1, v___x_298_);
lean_ctor_set(v___x_294_, 0, v___x_296_);
v___x_300_ = v___x_294_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v___x_296_);
lean_ctor_set(v_reuseFailAlloc_301_, 1, v___x_298_);
lean_ctor_set(v_reuseFailAlloc_301_, 2, v___f_297_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_monoid(lean_object* v_00_u03b1_304_, lean_object* v_h_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_mathlib_Multiplicative_monoid___redArg(v_h_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelMonoid___redArg(lean_object* v_inst_307_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addLeftCancelMonoid(lean_object* v_00_u03b1_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelMonoid___redArg(lean_object* v_inst_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lp_mathlib_Multiplicative_monoid___redArg(v_inst_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_leftCancelMonoid(lean_object* v_00_u03b1_314_, lean_object* v_inst_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_Multiplicative_monoid___redArg(v_inst_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelMonoid___redArg(lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addRightCancelMonoid(lean_object* v_00_u03b1_319_, lean_object* v_inst_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelMonoid___redArg(lean_object* v_inst_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lp_mathlib_Multiplicative_monoid___redArg(v_inst_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_rightCancelMonoid(lean_object* v_00_u03b1_324_, lean_object* v_inst_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lp_mathlib_Multiplicative_monoid___redArg(v_inst_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommMonoid___redArg(lean_object* v_inst_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommMonoid(lean_object* v_00_u03b1_329_, lean_object* v_inst_330_){
_start:
{
lean_object* v___x_331_; 
v___x_331_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commMonoid___redArg(lean_object* v_inst_332_){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = lp_mathlib_Multiplicative_monoid___redArg(v_inst_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commMonoid(lean_object* v_00_u03b1_334_, lean_object* v_inst_335_){
_start:
{
lean_object* v___x_336_; 
v___x_336_ = lp_mathlib_Multiplicative_monoid___redArg(v_inst_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_instAddCancelCommMonoid___redArg(lean_object* v_inst_337_){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_instAddCancelCommMonoid(lean_object* v_00_u03b1_339_, lean_object* v_inst_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_340_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_instCancelCommMonoid___redArg(lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lp_mathlib_Multiplicative_monoid___redArg(v_inst_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_instCancelCommMonoid(lean_object* v_00_u03b1_344_, lean_object* v_inst_345_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_mathlib_Multiplicative_monoid___redArg(v_inst_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg___redArg___lam__0(lean_object* v_inst_347_, lean_object* v_x_348_){
_start:
{
lean_object* v___x_349_; lean_object* v_toFun_350_; lean_object* v___x_351_; lean_object* v_toFun_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_349_ = lean_obj_once(&lp_mathlib_Additive_rec___redArg___closed__0, &lp_mathlib_Additive_rec___redArg___closed__0_once, _init_lp_mathlib_Additive_rec___redArg___closed__0);
v_toFun_350_ = lean_ctor_get(v___x_349_, 0);
v___x_351_ = lean_obj_once(&lp_mathlib_Multiplicative_toAdd___closed__0, &lp_mathlib_Multiplicative_toAdd___closed__0_once, _init_lp_mathlib_Multiplicative_toAdd___closed__0);
v_toFun_352_ = lean_ctor_get(v___x_351_, 0);
lean_inc(v_toFun_350_);
v___x_353_ = lean_apply_1(v_toFun_350_, v_x_348_);
v___x_354_ = lean_apply_1(v_inst_347_, v___x_353_);
lean_inc(v_toFun_352_);
v___x_355_ = lean_apply_1(v_toFun_352_, v___x_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg___redArg(lean_object* v_inst_356_){
_start:
{
lean_object* v___f_357_; 
v___f_357_ = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_357_, 0, v_inst_356_);
return v___f_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_neg(lean_object* v_00_u03b1_358_, lean_object* v_inst_359_){
_start:
{
lean_object* v___f_360_; 
v___f_360_ = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_360_, 0, v_inst_359_);
return v___f_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv___redArg___lam__0(lean_object* v_inst_361_, lean_object* v_x_362_){
_start:
{
lean_object* v___x_363_; lean_object* v_toFun_364_; lean_object* v___x_365_; lean_object* v_toFun_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_363_ = lean_obj_once(&lp_mathlib_Multiplicative_rec___redArg___closed__0, &lp_mathlib_Multiplicative_rec___redArg___closed__0_once, _init_lp_mathlib_Multiplicative_rec___redArg___closed__0);
v_toFun_364_ = lean_ctor_get(v___x_363_, 0);
v___x_365_ = lean_obj_once(&lp_mathlib_Additive_toMul___closed__0, &lp_mathlib_Additive_toMul___closed__0_once, _init_lp_mathlib_Additive_toMul___closed__0);
v_toFun_366_ = lean_ctor_get(v___x_365_, 0);
lean_inc(v_toFun_364_);
v___x_367_ = lean_apply_1(v_toFun_364_, v_x_362_);
v___x_368_ = lean_apply_1(v_inst_361_, v___x_367_);
lean_inc(v_toFun_366_);
v___x_369_ = lean_apply_1(v_toFun_366_, v___x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv___redArg(lean_object* v_inst_370_){
_start:
{
lean_object* v___f_371_; 
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_371_, 0, v_inst_370_);
return v___f_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_inv(lean_object* v_00_u03b1_372_, lean_object* v_inst_373_){
_start:
{
lean_object* v___f_374_; 
v___f_374_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_374_, 0, v_inst_373_);
return v___f_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_sub___redArg(lean_object* v_inst_375_){
_start:
{
lean_object* v___f_376_; lean_object* v___f_377_; 
v___f_376_ = ((lean_object*)(lp_mathlib_Additive_add___redArg___closed__0));
v___f_377_ = lean_alloc_closure((void*)(lp_mathlib_Additive_add___redArg___lam__1), 4, 2);
lean_closure_set(v___f_377_, 0, v___f_376_);
lean_closure_set(v___f_377_, 1, v_inst_375_);
return v___f_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_sub(lean_object* v_00_u03b1_378_, lean_object* v_inst_379_){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lp_mathlib_Additive_sub___redArg(v_inst_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_div___redArg(lean_object* v_inst_381_){
_start:
{
lean_object* v___f_382_; lean_object* v___f_383_; 
v___f_382_ = ((lean_object*)(lp_mathlib_Additive_add___redArg___closed__0));
v___f_383_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_mul___redArg___lam__1), 4, 2);
lean_closure_set(v___f_383_, 0, v___f_382_);
lean_closure_set(v___f_383_, 1, v_inst_381_);
return v___f_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_div(lean_object* v_00_u03b1_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lp_mathlib_Multiplicative_div___redArg(v_inst_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_involutiveNeg___redArg(lean_object* v_inst_387_){
_start:
{
lean_object* v___f_388_; 
v___f_388_ = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_388_, 0, v_inst_387_);
return v___f_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_involutiveNeg(lean_object* v_00_u03b1_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v___f_391_; 
v___f_391_ = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_391_, 0, v_inst_390_);
return v___f_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_involutiveInv___redArg(lean_object* v_inst_392_){
_start:
{
lean_object* v___f_393_; 
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_393_, 0, v_inst_392_);
return v___f_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_involutiveInv(lean_object* v_00_u03b1_394_, lean_object* v_inst_395_){
_start:
{
lean_object* v___f_396_; 
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_396_, 0, v_inst_395_);
return v___f_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid___redArg___lam__0(lean_object* v_toZPow_397_, lean_object* v_n_398_, lean_object* v_a_399_){
_start:
{
lean_object* v___x_400_; lean_object* v_toFun_401_; lean_object* v___x_402_; lean_object* v_toFun_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_400_ = lean_obj_once(&lp_mathlib_Additive_rec___redArg___closed__0, &lp_mathlib_Additive_rec___redArg___closed__0_once, _init_lp_mathlib_Additive_rec___redArg___closed__0);
v_toFun_401_ = lean_ctor_get(v___x_400_, 0);
v___x_402_ = lean_obj_once(&lp_mathlib_Additive_toMul___closed__0, &lp_mathlib_Additive_toMul___closed__0_once, _init_lp_mathlib_Additive_toMul___closed__0);
v_toFun_403_ = lean_ctor_get(v___x_402_, 0);
lean_inc(v_toFun_401_);
v___x_404_ = lean_apply_1(v_toFun_401_, v_a_399_);
v___x_405_ = lean_apply_2(v_toZPow_397_, v_n_398_, v___x_404_);
lean_inc(v_toFun_403_);
v___x_406_ = lean_apply_1(v_toFun_403_, v___x_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid___redArg(lean_object* v_h_407_){
_start:
{
lean_object* v_toMonoid_408_; lean_object* v_toInv_409_; lean_object* v_toDiv_410_; lean_object* v_toZPow_411_; lean_object* v___x_413_; uint8_t v_isShared_414_; uint8_t v_isSharedCheck_422_; 
v_toMonoid_408_ = lean_ctor_get(v_h_407_, 0);
v_toInv_409_ = lean_ctor_get(v_h_407_, 1);
v_toDiv_410_ = lean_ctor_get(v_h_407_, 2);
v_toZPow_411_ = lean_ctor_get(v_h_407_, 3);
v_isSharedCheck_422_ = !lean_is_exclusive(v_h_407_);
if (v_isSharedCheck_422_ == 0)
{
v___x_413_ = v_h_407_;
v_isShared_414_ = v_isSharedCheck_422_;
goto v_resetjp_412_;
}
else
{
lean_inc(v_toZPow_411_);
lean_inc(v_toDiv_410_);
lean_inc(v_toInv_409_);
lean_inc(v_toMonoid_408_);
lean_dec(v_h_407_);
v___x_413_ = lean_box(0);
v_isShared_414_ = v_isSharedCheck_422_;
goto v_resetjp_412_;
}
v_resetjp_412_:
{
lean_object* v___f_415_; lean_object* v___x_416_; lean_object* v___f_417_; lean_object* v___x_418_; lean_object* v___x_420_; 
v___f_415_ = lean_alloc_closure((void*)(lp_mathlib_Additive_subNegMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_415_, 0, v_toZPow_411_);
v___x_416_ = lp_mathlib_Additive_addMonoid___redArg(v_toMonoid_408_);
v___f_417_ = lean_alloc_closure((void*)(lp_mathlib_Additive_neg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_417_, 0, v_toInv_409_);
v___x_418_ = lp_mathlib_Additive_sub___redArg(v_toDiv_410_);
if (v_isShared_414_ == 0)
{
lean_ctor_set(v___x_413_, 3, v___f_415_);
lean_ctor_set(v___x_413_, 2, v___x_418_);
lean_ctor_set(v___x_413_, 1, v___f_417_);
lean_ctor_set(v___x_413_, 0, v___x_416_);
v___x_420_ = v___x_413_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v___x_416_);
lean_ctor_set(v_reuseFailAlloc_421_, 1, v___f_417_);
lean_ctor_set(v_reuseFailAlloc_421_, 2, v___x_418_);
lean_ctor_set(v_reuseFailAlloc_421_, 3, v___f_415_);
v___x_420_ = v_reuseFailAlloc_421_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
return v___x_420_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subNegMonoid(lean_object* v_00_u03b1_423_, lean_object* v_h_424_){
_start:
{
lean_object* v___x_425_; 
v___x_425_ = lp_mathlib_Additive_subNegMonoid___redArg(v_h_424_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg___lam__0(lean_object* v_toZSMul_426_, lean_object* v_n_427_, lean_object* v_a_428_){
_start:
{
lean_object* v___x_429_; lean_object* v_toFun_430_; lean_object* v___x_431_; lean_object* v_toFun_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; 
v___x_429_ = lean_obj_once(&lp_mathlib_Multiplicative_rec___redArg___closed__0, &lp_mathlib_Multiplicative_rec___redArg___closed__0_once, _init_lp_mathlib_Multiplicative_rec___redArg___closed__0);
v_toFun_430_ = lean_ctor_get(v___x_429_, 0);
v___x_431_ = lean_obj_once(&lp_mathlib_Multiplicative_toAdd___closed__0, &lp_mathlib_Multiplicative_toAdd___closed__0_once, _init_lp_mathlib_Multiplicative_toAdd___closed__0);
v_toFun_432_ = lean_ctor_get(v___x_431_, 0);
lean_inc(v_toFun_430_);
v___x_433_ = lean_apply_1(v_toFun_430_, v_a_428_);
v___x_434_ = lean_apply_2(v_toZSMul_426_, v_n_427_, v___x_433_);
lean_inc(v_toFun_432_);
v___x_435_ = lean_apply_1(v_toFun_432_, v___x_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object* v_h_436_){
_start:
{
lean_object* v_toAddMonoid_437_; lean_object* v_toNeg_438_; lean_object* v_toSub_439_; lean_object* v_toZSMul_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_451_; 
v_toAddMonoid_437_ = lean_ctor_get(v_h_436_, 0);
v_toNeg_438_ = lean_ctor_get(v_h_436_, 1);
v_toSub_439_ = lean_ctor_get(v_h_436_, 2);
v_toZSMul_440_ = lean_ctor_get(v_h_436_, 3);
v_isSharedCheck_451_ = !lean_is_exclusive(v_h_436_);
if (v_isSharedCheck_451_ == 0)
{
v___x_442_ = v_h_436_;
v_isShared_443_ = v_isSharedCheck_451_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_toZSMul_440_);
lean_inc(v_toSub_439_);
lean_inc(v_toNeg_438_);
lean_inc(v_toAddMonoid_437_);
lean_dec(v_h_436_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_451_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___f_444_; lean_object* v___x_445_; lean_object* v___f_446_; lean_object* v___x_447_; lean_object* v___x_449_; 
v___f_444_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_divInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_444_, 0, v_toZSMul_440_);
v___x_445_ = lp_mathlib_Multiplicative_monoid___redArg(v_toAddMonoid_437_);
v___f_446_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_446_, 0, v_toNeg_438_);
v___x_447_ = lp_mathlib_Multiplicative_div___redArg(v_toSub_439_);
if (v_isShared_443_ == 0)
{
lean_ctor_set(v___x_442_, 3, v___f_444_);
lean_ctor_set(v___x_442_, 2, v___x_447_);
lean_ctor_set(v___x_442_, 1, v___f_446_);
lean_ctor_set(v___x_442_, 0, v___x_445_);
v___x_449_ = v___x_442_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v___x_445_);
lean_ctor_set(v_reuseFailAlloc_450_, 1, v___f_446_);
lean_ctor_set(v_reuseFailAlloc_450_, 2, v___x_447_);
lean_ctor_set(v_reuseFailAlloc_450_, 3, v___f_444_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divInvMonoid(lean_object* v_00_u03b1_452_, lean_object* v_h_453_){
_start:
{
lean_object* v___x_454_; 
v___x_454_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_h_453_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionMonoid___redArg(lean_object* v_inst_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionMonoid(lean_object* v_00_u03b1_457_, lean_object* v_inst_458_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionMonoid___redArg(lean_object* v_inst_460_){
_start:
{
lean_object* v___x_461_; 
v___x_461_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionMonoid(lean_object* v_00_u03b1_462_, lean_object* v_inst_463_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionCommMonoid___redArg(lean_object* v_inst_465_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_465_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_subtractionCommMonoid(lean_object* v_00_u03b1_467_, lean_object* v_inst_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionCommMonoid___redArg(lean_object* v_inst_470_){
_start:
{
lean_object* v___x_471_; 
v___x_471_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_470_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_divisionCommMonoid(lean_object* v_00_u03b1_472_, lean_object* v_inst_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addGroup___redArg(lean_object* v_inst_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addGroup(lean_object* v_00_u03b1_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_478_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_group___redArg(lean_object* v_inst_480_){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_480_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_group(lean_object* v_00_u03b1_482_, lean_object* v_inst_483_){
_start:
{
lean_object* v___x_484_; 
v___x_484_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_483_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommGroup___redArg(lean_object* v_inst_485_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_485_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_addCommGroup(lean_object* v_00_u03b1_487_, lean_object* v_inst_488_){
_start:
{
lean_object* v___x_489_; 
v___x_489_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_488_);
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commGroup___redArg(lean_object* v_inst_490_){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_490_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_commGroup(lean_object* v_00_u03b1_492_, lean_object* v_inst_493_){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_493_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun___redArg___lam__0(lean_object* v_inst_495_, lean_object* v_a_496_){
_start:
{
lean_object* v___x_497_; lean_object* v_toFun_498_; lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_497_ = lean_obj_once(&lp_mathlib_Additive_rec___redArg___closed__0, &lp_mathlib_Additive_rec___redArg___closed__0_once, _init_lp_mathlib_Additive_rec___redArg___closed__0);
v_toFun_498_ = lean_ctor_get(v___x_497_, 0);
lean_inc(v_toFun_498_);
v___x_499_ = lean_apply_1(v_toFun_498_, v_a_496_);
v___x_500_ = lean_apply_1(v_inst_495_, v___x_499_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun___redArg(lean_object* v_inst_501_){
_start:
{
lean_object* v___f_502_; 
v___f_502_ = lean_alloc_closure((void*)(lp_mathlib_Additive_coeToFun___redArg___lam__0), 2, 1);
lean_closure_set(v___f_502_, 0, v_inst_501_);
return v___f_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_coeToFun(lean_object* v_00_u03b1_503_, lean_object* v_00_u03b2_504_, lean_object* v_inst_505_){
_start:
{
lean_object* v___f_506_; 
v___f_506_ = lean_alloc_closure((void*)(lp_mathlib_Additive_coeToFun___redArg___lam__0), 2, 1);
lean_closure_set(v___f_506_, 0, v_inst_505_);
return v___f_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun___redArg___lam__0(lean_object* v_inst_507_, lean_object* v_a_508_){
_start:
{
lean_object* v___x_509_; lean_object* v_toFun_510_; lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_509_ = lean_obj_once(&lp_mathlib_Multiplicative_rec___redArg___closed__0, &lp_mathlib_Multiplicative_rec___redArg___closed__0_once, _init_lp_mathlib_Multiplicative_rec___redArg___closed__0);
v_toFun_510_ = lean_ctor_get(v___x_509_, 0);
lean_inc(v_toFun_510_);
v___x_511_ = lean_apply_1(v_toFun_510_, v_a_508_);
v___x_512_ = lean_apply_1(v_inst_507_, v___x_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun___redArg(lean_object* v_inst_513_){
_start:
{
lean_object* v___f_514_; 
v___f_514_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_coeToFun___redArg___lam__0), 2, 1);
lean_closure_set(v___f_514_, 0, v_inst_513_);
return v___f_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_coeToFun(lean_object* v_00_u03b1_515_, lean_object* v_00_u03b2_516_, lean_object* v_inst_517_){
_start:
{
lean_object* v___f_518_; 
v___f_518_ = lean_alloc_closure((void*)(lp_mathlib_Multiplicative_coeToFun___redArg___lam__0), 2, 1);
lean_closure_set(v___f_518_, 0, v_inst_517_);
return v___f_518_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_FunLike_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_FunLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_FunLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_FunLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
