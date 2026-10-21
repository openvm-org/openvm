// Lean compiler output
// Module: Mathlib.Data.Prod.PProd
// Imports: public import Init public meta import Init public import Batteries.Logic public import Mathlib.Init
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
LEAN_EXPORT lean_object* lp_mathlib_PProd_mk_injArrow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PProd_mk_injArrow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PProd_mk_injArrow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PProd_mk_injArrow___redArg(lean_object* v_w_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_apply_2(v_w_1_, lean_box(0), lean_box(0));
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PProd_mk_injArrow(lean_object* v_00_u03b1_3_, lean_object* v_00_u03b2_4_, lean_object* v_x_u2081_5_, lean_object* v_y_u2081_6_, lean_object* v_x_u2082_7_, lean_object* v_y_u2082_8_, lean_object* v_h_9_, lean_object* v_P_10_, lean_object* v_w_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_apply_2(v_w_11_, lean_box(0), lean_box(0));
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PProd_mk_injArrow___boxed(lean_object* v_00_u03b1_13_, lean_object* v_00_u03b2_14_, lean_object* v_x_u2081_15_, lean_object* v_y_u2081_16_, lean_object* v_x_u2082_17_, lean_object* v_y_u2082_18_, lean_object* v_h_19_, lean_object* v_P_20_, lean_object* v_w_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_PProd_mk_injArrow(v_00_u03b1_13_, v_00_u03b2_14_, v_x_u2081_15_, v_y_u2081_16_, v_x_u2082_17_, v_y_u2082_18_, v_h_19_, v_P_20_, v_w_21_);
lean_dec(v_y_u2082_18_);
lean_dec(v_x_u2082_17_);
lean_dec(v_y_u2081_16_);
lean_dec(v_x_u2081_15_);
return v_res_22_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Prod_PProd(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Prod_PProd(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Prod_PProd(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Prod_PProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Prod_PProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Prod_PProd(builtin);
}
#ifdef __cplusplus
}
#endif
