// Lean compiler output
// Module: Mathlib.Algebra.Group.TypeTags.Hom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.Hom.Basic public import Mathlib.Algebra.Group.TypeTags.Basic
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
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
lean_object* lp_mathlib_Additive_toMul(lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddMonoidHom_toMultiplicative___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddMonoidHom_toMultiplicative___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___closed__0 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicative___closed__0_value;
static const lean_closure_object lp_mathlib_AddMonoidHom_toMultiplicative___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddMonoidHom_toMultiplicative___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___closed__1 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicative___closed__1_value;
static const lean_ctor_object lp_mathlib_AddMonoidHom_toMultiplicative___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicative___closed__0_value),((lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicative___closed__1_value)}};
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___closed__2 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicative___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicative(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_toAdditive___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_toAdditive___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_toAdditive___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_toAdditive___closed__0_value;
static const lean_closure_object lp_mathlib_MonoidHom_toAdditive___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_toAdditive___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_toAdditive___closed__1 = (const lean_object*)&lp_mathlib_MonoidHom_toAdditive___closed__1_value;
static const lean_ctor_object lp_mathlib_MonoidHom_toAdditive___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MonoidHom_toAdditive___closed__0_value),((lean_object*)&lp_mathlib_MonoidHom_toAdditive___closed__1_value)}};
static const lean_object* lp_mathlib_MonoidHom_toAdditive___closed__2 = (const lean_object*)&lp_mathlib_MonoidHom_toAdditive___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddMonoidHom_toMultiplicativeRight___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__0 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__0_value;
static const lean_closure_object lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddMonoidHom_toMultiplicativeRight___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__1 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__1_value;
static const lean_ctor_object lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__0_value),((lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__1_value)}};
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__2 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddMonoidHom_toMultiplicativeLeft___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__0 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__0_value;
static const lean_closure_object lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddMonoidHom_toMultiplicativeLeft___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__1 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__1_value;
static const lean_ctor_object lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__0_value),((lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__1_value)}};
static const lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__2 = (const lean_object*)&lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRightMulEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRightMulEquiv___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRightMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRightMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___lam__0(lean_object* v_f_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_toFun_6_; lean_object* v___x_7_; lean_object* v_toFun_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_5_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0);
v_toFun_6_ = lean_ctor_get(v___x_5_, 0);
v___x_7_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1);
v_toFun_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_toFun_6_);
v___x_9_ = lean_apply_1(v_toFun_6_, v___y_4_);
v___x_10_ = lean_apply_1(v_f_3_, v___x_9_);
lean_inc(v_toFun_8_);
v___x_11_ = lean_apply_1(v_toFun_8_, v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___lam__1(lean_object* v_f_12_, lean_object* v___y_13_){
_start:
{
lean_object* v___x_14_; lean_object* v_toFun_15_; lean_object* v___x_16_; lean_object* v_toFun_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_14_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1);
v_toFun_15_ = lean_ctor_get(v___x_14_, 0);
v___x_16_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0);
v_toFun_17_ = lean_ctor_get(v___x_16_, 0);
lean_inc(v_toFun_15_);
v___x_18_ = lean_apply_1(v_toFun_15_, v___y_13_);
v___x_19_ = lean_apply_1(v_f_12_, v___x_18_);
lean_inc(v_toFun_17_);
v___x_20_ = lean_apply_1(v_toFun_17_, v___x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicative(lean_object* v_00_u03b1_26_, lean_object* v_00_u03b2_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = ((lean_object*)(lp_mathlib_AddMonoidHom_toMultiplicative___closed__2));
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicative___boxed(lean_object* v_00_u03b1_31_, lean_object* v_00_u03b2_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_AddMonoidHom_toMultiplicative(v_00_u03b1_31_, v_00_u03b2_32_, v_inst_33_, v_inst_34_);
lean_dec_ref(v_inst_34_);
lean_dec_ref(v_inst_33_);
return v_res_35_;
}
}
static lean_object* _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1(void){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___lam__0(lean_object* v_f_38_, lean_object* v___y_39_){
_start:
{
lean_object* v___x_40_; lean_object* v_toFun_41_; lean_object* v___x_42_; lean_object* v_toFun_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_40_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0, &lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0);
v_toFun_41_ = lean_ctor_get(v___x_40_, 0);
v___x_42_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1, &lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1);
v_toFun_43_ = lean_ctor_get(v___x_42_, 0);
lean_inc(v_toFun_41_);
v___x_44_ = lean_apply_1(v_toFun_41_, v___y_39_);
v___x_45_ = lean_apply_1(v_f_38_, v___x_44_);
lean_inc(v_toFun_43_);
v___x_46_ = lean_apply_1(v_toFun_43_, v___x_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___lam__1(lean_object* v_f_47_, lean_object* v___y_48_){
_start:
{
lean_object* v___x_49_; lean_object* v_toFun_50_; lean_object* v___x_51_; lean_object* v_toFun_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_49_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1, &lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1);
v_toFun_50_ = lean_ctor_get(v___x_49_, 0);
v___x_51_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0, &lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0);
v_toFun_52_ = lean_ctor_get(v___x_51_, 0);
lean_inc(v_toFun_50_);
v___x_53_ = lean_apply_1(v_toFun_50_, v___y_48_);
v___x_54_ = lean_apply_1(v_f_47_, v___x_53_);
lean_inc(v_toFun_52_);
v___x_55_ = lean_apply_1(v_toFun_52_, v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive(lean_object* v_00_u03b1_61_, lean_object* v_00_u03b2_62_, lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = ((lean_object*)(lp_mathlib_MonoidHom_toAdditive___closed__2));
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___boxed(lean_object* v_00_u03b1_66_, lean_object* v_00_u03b2_67_, lean_object* v_inst_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_MonoidHom_toAdditive(v_00_u03b1_66_, v_00_u03b2_67_, v_inst_68_, v_inst_69_);
lean_dec_ref(v_inst_69_);
lean_dec_ref(v_inst_68_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___lam__0(lean_object* v_f_71_, lean_object* v___y_72_){
_start:
{
lean_object* v___x_73_; lean_object* v_toFun_74_; lean_object* v___x_75_; lean_object* v_toFun_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_73_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1, &lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1);
v_toFun_74_ = lean_ctor_get(v___x_73_, 0);
v___x_75_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1);
v_toFun_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_toFun_74_);
v___x_77_ = lean_apply_1(v_toFun_74_, v___y_72_);
v___x_78_ = lean_apply_1(v_f_71_, v___x_77_);
lean_inc(v_toFun_76_);
v___x_79_ = lean_apply_1(v_toFun_76_, v___x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___lam__1(lean_object* v_f_80_, lean_object* v___y_81_){
_start:
{
lean_object* v___x_82_; lean_object* v_toFun_83_; lean_object* v___x_84_; lean_object* v_toFun_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_82_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0, &lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0);
v_toFun_83_ = lean_ctor_get(v___x_82_, 0);
v___x_84_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0);
v_toFun_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc(v_toFun_83_);
v___x_86_ = lean_apply_1(v_toFun_83_, v___y_81_);
v___x_87_ = lean_apply_1(v_f_80_, v___x_86_);
lean_inc(v_toFun_85_);
v___x_88_ = lean_apply_1(v_toFun_85_, v___x_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight(lean_object* v_00_u03b1_94_, lean_object* v_00_u03b2_95_, lean_object* v_inst_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = ((lean_object*)(lp_mathlib_AddMonoidHom_toMultiplicativeRight___closed__2));
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRight___boxed(lean_object* v_00_u03b1_99_, lean_object* v_00_u03b2_100_, lean_object* v_inst_101_, lean_object* v_inst_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_AddMonoidHom_toMultiplicativeRight(v_00_u03b1_99_, v_00_u03b2_100_, v_inst_101_, v_inst_102_);
lean_dec_ref(v_inst_102_);
lean_dec_ref(v_inst_101_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeft___redArg(lean_object* v_inst_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = lp_mathlib_AddMonoidHom_toMultiplicativeRight(lean_box(0), lean_box(0), v_inst_104_, v_inst_105_);
v___x_107_ = lp_mathlib_Equiv_symm___redArg(v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeft___redArg___boxed(lean_object* v_inst_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_MonoidHom_toAdditiveLeft___redArg(v_inst_108_, v_inst_109_);
lean_dec_ref(v_inst_109_);
lean_dec_ref(v_inst_108_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeft(lean_object* v_00_u03b1_111_, lean_object* v_00_u03b2_112_, lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lp_mathlib_MonoidHom_toAdditiveLeft___redArg(v_inst_113_, v_inst_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeft___boxed(lean_object* v_00_u03b1_116_, lean_object* v_00_u03b2_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_MonoidHom_toAdditiveLeft(v_00_u03b1_116_, v_00_u03b2_117_, v_inst_118_, v_inst_119_);
lean_dec_ref(v_inst_119_);
lean_dec_ref(v_inst_118_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___lam__0(lean_object* v_f_121_, lean_object* v___y_122_){
_start:
{
lean_object* v___x_123_; lean_object* v_toFun_124_; lean_object* v___x_125_; lean_object* v_toFun_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_123_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__0);
v_toFun_124_ = lean_ctor_get(v___x_123_, 0);
v___x_125_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0, &lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__0);
v_toFun_126_ = lean_ctor_get(v___x_125_, 0);
lean_inc(v_toFun_124_);
v___x_127_ = lean_apply_1(v_toFun_124_, v___y_122_);
v___x_128_ = lean_apply_1(v_f_121_, v___x_127_);
lean_inc(v_toFun_126_);
v___x_129_ = lean_apply_1(v_toFun_126_, v___x_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___lam__1(lean_object* v_f_130_, lean_object* v___y_131_){
_start:
{
lean_object* v___x_132_; lean_object* v_toFun_133_; lean_object* v___x_134_; lean_object* v_toFun_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_132_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddMonoidHom_toMultiplicative___lam__0___closed__1);
v_toFun_133_ = lean_ctor_get(v___x_132_, 0);
v___x_134_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1, &lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MonoidHom_toAdditive___lam__0___closed__1);
v_toFun_135_ = lean_ctor_get(v___x_134_, 0);
lean_inc(v_toFun_133_);
v___x_136_ = lean_apply_1(v_toFun_133_, v___y_131_);
v___x_137_ = lean_apply_1(v_f_130_, v___x_136_);
lean_inc(v_toFun_135_);
v___x_138_ = lean_apply_1(v_toFun_135_, v___x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_object* v_00_u03b1_144_, lean_object* v_00_u03b2_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = ((lean_object*)(lp_mathlib_AddMonoidHom_toMultiplicativeLeft___closed__2));
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft___boxed(lean_object* v_00_u03b1_149_, lean_object* v_00_u03b2_150_, lean_object* v_inst_151_, lean_object* v_inst_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_AddMonoidHom_toMultiplicativeLeft(v_00_u03b1_149_, v_00_u03b2_150_, v_inst_151_, v_inst_152_);
lean_dec_ref(v_inst_152_);
lean_dec_ref(v_inst_151_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRight___redArg(lean_object* v_inst_154_, lean_object* v_inst_155_){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_box(0), lean_box(0), v_inst_154_, v_inst_155_);
v___x_157_ = lp_mathlib_Equiv_symm___redArg(v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRight___redArg___boxed(lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_MonoidHom_toAdditiveRight___redArg(v_inst_158_, v_inst_159_);
lean_dec_ref(v_inst_159_);
lean_dec_ref(v_inst_158_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRight(lean_object* v_00_u03b1_161_, lean_object* v_00_u03b2_162_, lean_object* v_inst_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_MonoidHom_toAdditiveRight___redArg(v_inst_163_, v_inst_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRight___boxed(lean_object* v_00_u03b1_166_, lean_object* v_00_u03b2_167_, lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_MonoidHom_toAdditiveRight(v_00_u03b1_166_, v_00_u03b2_167_, v_inst_168_, v_inst_169_);
lean_dec_ref(v_inst_169_);
lean_dec_ref(v_inst_168_);
return v_res_170_;
}
}
static lean_object* _init_lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0(void){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg(lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_174_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_172_);
v___x_175_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_173_);
v___x_176_ = lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_box(0), lean_box(0), v___x_174_, v___x_175_);
lean_dec_ref(v___x_175_);
lean_dec_ref(v___x_174_);
v___x_177_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0, &lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0_once, _init_lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0);
v___x_178_ = lp_mathlib_Equiv_trans___redArg(v___x_176_, v___x_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___boxed(lean_object* v_inst_179_, lean_object* v_inst_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg(v_inst_179_, v_inst_180_);
lean_dec_ref(v_inst_180_);
lean_dec_ref(v_inst_179_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv(lean_object* v_M_182_, lean_object* v_N_183_, lean_object* v_inst_184_, lean_object* v_inst_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg(v_inst_184_, v_inst_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___boxed(lean_object* v_M_187_, lean_object* v_N_188_, lean_object* v_inst_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv(v_M_187_, v_N_188_, v_inst_189_, v_inst_190_);
lean_dec_ref(v_inst_190_);
lean_dec_ref(v_inst_189_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv___redArg(lean_object* v_inst_192_, lean_object* v_inst_193_){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_194_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_192_);
v___x_195_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_193_);
v___x_196_ = lp_mathlib_AddMonoidHom_toMultiplicativeRight(lean_box(0), lean_box(0), v___x_194_, v___x_195_);
lean_dec_ref(v___x_195_);
lean_dec_ref(v___x_194_);
v___x_197_ = lean_obj_once(&lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0, &lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0_once, _init_lp_mathlib_AddMonoidHom_toMultiplicativeLeftAddEquiv___redArg___closed__0);
v___x_198_ = lp_mathlib_Equiv_trans___redArg(v___x_196_, v___x_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv___redArg___boxed(lean_object* v_inst_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv___redArg(v_inst_199_, v_inst_200_);
lean_dec_ref(v_inst_200_);
lean_dec_ref(v_inst_199_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv(lean_object* v_M_202_, lean_object* v_N_203_, lean_object* v_inst_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv___redArg(v_inst_204_, v_inst_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv___boxed(lean_object* v_M_207_, lean_object* v_N_208_, lean_object* v_inst_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_AddMonoidHom_toMultiplicativeRightAddEquiv(v_M_207_, v_N_208_, v_inst_209_, v_inst_210_);
lean_dec_ref(v_inst_210_);
lean_dec_ref(v_inst_209_);
return v_res_211_;
}
}
static lean_object* _init_lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0(void){
_start:
{
lean_object* v___x_212_; 
v___x_212_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg(lean_object* v_inst_213_, lean_object* v_inst_214_){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_215_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_213_);
v___x_216_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_214_);
v___x_217_ = lp_mathlib_MonoidHom_toAdditiveLeft___redArg(v___x_215_, v___x_216_);
lean_dec_ref(v___x_216_);
lean_dec_ref(v___x_215_);
v___x_218_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0, &lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0_once, _init_lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0);
v___x_219_ = lp_mathlib_Equiv_trans___redArg(v___x_217_, v___x_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___boxed(lean_object* v_inst_220_, lean_object* v_inst_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg(v_inst_220_, v_inst_221_);
lean_dec_ref(v_inst_221_);
lean_dec_ref(v_inst_220_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv(lean_object* v_M_223_, lean_object* v_N_224_, lean_object* v_inst_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg(v_inst_225_, v_inst_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___boxed(lean_object* v_M_228_, lean_object* v_N_229_, lean_object* v_inst_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv(v_M_228_, v_N_229_, v_inst_230_, v_inst_231_);
lean_dec_ref(v_inst_231_);
lean_dec_ref(v_inst_230_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRightMulEquiv___redArg(lean_object* v_inst_233_, lean_object* v_inst_234_){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_235_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_233_);
v___x_236_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_234_);
v___x_237_ = lp_mathlib_MonoidHom_toAdditiveRight___redArg(v___x_235_, v___x_236_);
lean_dec_ref(v___x_236_);
lean_dec_ref(v___x_235_);
v___x_238_ = lean_obj_once(&lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0, &lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0_once, _init_lp_mathlib_MonoidHom_toAdditiveLeftMulEquiv___redArg___closed__0);
v___x_239_ = lp_mathlib_Equiv_trans___redArg(v___x_237_, v___x_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRightMulEquiv___redArg___boxed(lean_object* v_inst_240_, lean_object* v_inst_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_MonoidHom_toAdditiveRightMulEquiv___redArg(v_inst_240_, v_inst_241_);
lean_dec_ref(v_inst_241_);
lean_dec_ref(v_inst_240_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRightMulEquiv(lean_object* v_M_243_, lean_object* v_N_244_, lean_object* v_inst_245_, lean_object* v_inst_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_MonoidHom_toAdditiveRightMulEquiv___redArg(v_inst_245_, v_inst_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditiveRightMulEquiv___boxed(lean_object* v_M_248_, lean_object* v_N_249_, lean_object* v_inst_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib_MonoidHom_toAdditiveRightMulEquiv(v_M_248_, v_N_249_, v_inst_250_, v_inst_251_);
lean_dec_ref(v_inst_251_);
lean_dec_ref(v_inst_250_);
return v_res_252_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
}
#ifdef __cplusplus
}
#endif
