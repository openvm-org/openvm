// Lean compiler output
// Module: Mathlib.Order.Filter.Pi
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Piecewise public import Mathlib.Order.Filter.Tendsto public import Mathlib.Order.Filter.Bases.Finite
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
LEAN_EXPORT lean_object* lp_mathlib_iSup___at___00Filter_coprod_u1d62_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup___at___00Filter_coprod_u1d62_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_coprod_u1d62(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_coprod_u1d62___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup___at___00Filter_coprod_u1d62_spec__0(lean_object* v_00_u03b9_1_, lean_object* v_00_u03b1_2_, lean_object* v_00_u03b9_3_, lean_object* v_s_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup___at___00Filter_coprod_u1d62_spec__0___boxed(lean_object* v_00_u03b9_6_, lean_object* v_00_u03b1_7_, lean_object* v_00_u03b9_8_, lean_object* v_s_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_iSup___at___00Filter_coprod_u1d62_spec__0(v_00_u03b9_6_, v_00_u03b1_7_, v_00_u03b9_8_, v_s_9_);
lean_dec_ref(v_s_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_coprod_u1d62(lean_object* v_00_u03b9_11_, lean_object* v_00_u03b1_12_, lean_object* v_f_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_box(0);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_coprod_u1d62___boxed(lean_object* v_00_u03b9_15_, lean_object* v_00_u03b1_16_, lean_object* v_f_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Filter_coprod_u1d62(v_00_u03b9_15_, v_00_u03b1_16_, v_f_17_);
lean_dec_ref(v_f_17_);
return v_res_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Piecewise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Tendsto(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Bases_Finite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Tendsto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Bases_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Filter_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Piecewise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Tendsto(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Bases_Finite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Filter_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Tendsto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Bases_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Filter_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Filter_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
