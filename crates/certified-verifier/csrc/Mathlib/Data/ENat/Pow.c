// Lean compiler output
// Module: Mathlib.Data.ENat.Pow
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Monoid.Unbundled.Pow public import Mathlib.Data.ENat.SuccOrder import Mathlib.Data.Nat.Cast.Order.Basic
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
extern lean_object* lp_mathlib_instCommSemiringENat;
extern lean_object* lp_mathlib_instLinearOrderENat;
extern lean_object* lp_mathlib_instZeroENat;
extern lean_object* lp_mathlib_instOneENat;
LEAN_EXPORT lean_object* lp_mathlib_ENat_instPow___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_instPow;
LEAN_EXPORT lean_object* lp_mathlib_ENat_instPow___lam__0(lean_object* v___x_1_, lean_object* v___x_2_, lean_object* v_toNPow_3_, lean_object* v_x_4_, lean_object* v_x_5_){
_start:
{
if (lean_obj_tag(v_x_5_) == 0)
{
lean_object* v_toDecidableEq_6_; lean_object* v___x_7_; uint8_t v___x_8_; 
lean_dec(v_toNPow_3_);
v_toDecidableEq_6_ = lean_ctor_get(v___x_1_, 5);
lean_inc_ref_n(v_toDecidableEq_6_, 2);
lean_dec_ref(v___x_1_);
lean_inc(v___x_2_);
lean_inc(v_x_4_);
v___x_7_ = lean_apply_2(v_toDecidableEq_6_, v_x_4_, v___x_2_);
v___x_8_ = lean_unbox(v___x_7_);
if (v___x_8_ == 0)
{
lean_object* v___x_9_; lean_object* v___x_10_; uint8_t v___x_11_; 
lean_dec(v___x_2_);
v___x_9_ = lp_mathlib_instOneENat;
v___x_10_ = lean_apply_2(v_toDecidableEq_6_, v_x_4_, v___x_9_);
v___x_11_ = lean_unbox(v___x_10_);
if (v___x_11_ == 0)
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
else
{
return v___x_9_;
}
}
else
{
lean_dec_ref(v_toDecidableEq_6_);
lean_dec(v_x_4_);
return v___x_2_;
}
}
else
{
lean_object* v_val_13_; lean_object* v___x_14_; 
lean_dec(v___x_2_);
lean_dec_ref(v___x_1_);
v_val_13_ = lean_ctor_get(v_x_5_, 0);
lean_inc(v_val_13_);
lean_dec_ref_known(v_x_5_, 1);
v___x_14_ = lean_apply_2(v_toNPow_3_, v_val_13_, v_x_4_);
return v___x_14_;
}
}
}
static lean_object* _init_lp_mathlib_ENat_instPow(void){
_start:
{
lean_object* v___x_15_; lean_object* v_toMonoid_16_; lean_object* v_toNPow_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___f_20_; 
v___x_15_ = lp_mathlib_instCommSemiringENat;
v_toMonoid_16_ = lean_ctor_get(v___x_15_, 1);
v_toNPow_17_ = lean_ctor_get(v_toMonoid_16_, 2);
v___x_18_ = lp_mathlib_instLinearOrderENat;
v___x_19_ = lp_mathlib_instZeroENat;
lean_inc(v_toNPow_17_);
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_ENat_instPow___lam__0), 5, 3);
lean_closure_set(v___f_20_, 0, v___x_18_);
lean_closure_set(v___f_20_, 1, v___x_19_);
lean_closure_set(v___f_20_, 2, v_toNPow_17_);
return v___f_20_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_SuccOrder(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_Pow(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_SuccOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ENat_instPow = _init_lp_mathlib_ENat_instPow();
lean_mark_persistent(lp_mathlib_ENat_instPow);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ENat_Pow(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENat_SuccOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ENat_Pow(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_SuccOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ENat_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ENat_Pow(builtin);
}
#ifdef __cplusplus
}
#endif
