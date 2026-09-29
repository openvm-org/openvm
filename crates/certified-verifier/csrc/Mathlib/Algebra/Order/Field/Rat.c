// Lean compiler output
// Module: Mathlib.Algebra.Order.Field.Rat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Rat public import Mathlib.Algebra.Order.Nonneg.Field public import Mathlib.Algebra.Order.Ring.Rat
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
extern lean_object* lp_mathlib_NNRat_instSemifield;
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_NNRat_instInv___lam__0___boxed(lean_object*);
extern lean_object* lp_mathlib_instLinearOrderNNRat;
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
extern lean_object* lp_mathlib_NNRat_instOrderBot;
lean_object* lp_mathlib_NNRat_instDiv___lam__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0;
static lean_once_cell_t lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__1;
static lean_once_cell_t lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__2;
static lean_once_cell_t lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__3;
static const lean_closure_object lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_instInv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__4 = (const lean_object*)&lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__4_value;
static const lean_closure_object lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_instDiv___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__5 = (const lean_object*)&lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat;
static lean_object* _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lp_mathlib_NNRat_instSemifield;
v___x_2_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lp_mathlib_NNRat_instSemifield;
v___x_4_ = lp_mathlib_Semifield_toDivisionSemiring___redArg(v___x_3_);
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__2(void){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_obj_once(&lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__1, &lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__1_once, _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__1);
v___x_6_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_5_);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__3(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_obj_once(&lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__2, &lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__2_once, _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__2);
v___x_8_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat(void){
_start:
{
lean_object* v___x_11_; lean_object* v_toCommMonoidWithZero_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v_toZPow_15_; lean_object* v___x_16_; lean_object* v___f_17_; lean_object* v___f_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_11_ = lean_obj_once(&lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0, &lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0_once, _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__0);
v_toCommMonoidWithZero_12_ = lean_ctor_get(v___x_11_, 0);
v___x_13_ = lp_mathlib_instLinearOrderNNRat;
v___x_14_ = lean_obj_once(&lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__3, &lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__3_once, _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__3);
v_toZPow_15_ = lean_ctor_get(v___x_14_, 3);
v___x_16_ = lp_mathlib_NNRat_instOrderBot;
v___f_17_ = ((lean_object*)(lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__4));
v___f_18_ = ((lean_object*)(lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat___closed__5));
lean_inc_ref(v_toCommMonoidWithZero_12_);
v___x_19_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_19_, 0, v_toCommMonoidWithZero_12_);
lean_ctor_set(v___x_19_, 1, v___x_13_);
lean_ctor_set(v___x_19_, 2, v___x_16_);
lean_inc(v_toZPow_15_);
v___x_20_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
lean_ctor_set(v___x_20_, 1, v___f_17_);
lean_ctor_set(v___x_20_, 2, v___f_18_);
lean_ctor_set(v___x_20_, 3, v_toZPow_15_);
return v___x_20_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Rat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Rat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat = _init_lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat();
lean_mark_persistent(lp_mathlib_instLinearOrderedCommGroupWithZeroNNRat);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Rat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Field_Rat(builtin);
}
#ifdef __cplusplus
}
#endif
