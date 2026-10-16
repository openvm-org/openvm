// Lean compiler output
// Module: Mathlib.Data.Finset.Piecewise
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.BooleanAlgebra public import Mathlib.Data.Set.Piecewise public import Mathlib.Order.Interval.Set.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Finset_piecewise___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_piecewise(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_piecewise___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_piecewise___redArg(lean_object* v_f_1_, lean_object* v_g_2_, lean_object* v_inst_3_, lean_object* v_i_4_){
_start:
{
lean_object* v___x_5_; uint8_t v___x_6_; 
lean_inc(v_i_4_);
v___x_5_ = lean_apply_1(v_inst_3_, v_i_4_);
v___x_6_ = lean_unbox(v___x_5_);
if (v___x_6_ == 0)
{
lean_object* v___x_7_; 
lean_dec(v_f_1_);
v___x_7_ = lean_apply_1(v_g_2_, v_i_4_);
return v___x_7_;
}
else
{
lean_object* v___x_8_; 
lean_dec(v_g_2_);
v___x_8_ = lean_apply_1(v_f_1_, v_i_4_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_piecewise(lean_object* v_00_u03b9_9_, lean_object* v_00_u03c0_10_, lean_object* v_s_11_, lean_object* v_f_12_, lean_object* v_g_13_, lean_object* v_inst_14_, lean_object* v_i_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Finset_piecewise___redArg(v_f_12_, v_g_13_, v_inst_14_, v_i_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_piecewise___boxed(lean_object* v_00_u03b9_17_, lean_object* v_00_u03c0_18_, lean_object* v_s_19_, lean_object* v_f_20_, lean_object* v_g_21_, lean_object* v_inst_22_, lean_object* v_i_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Finset_piecewise(v_00_u03b9_17_, v_00_u03c0_18_, v_s_19_, v_f_20_, v_g_21_, v_inst_22_, v_i_23_);
lean_dec(v_s_19_);
return v_res_24_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Piecewise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Piecewise(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Piecewise(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Piecewise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Piecewise(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Piecewise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Piecewise(builtin);
}
#ifdef __cplusplus
}
#endif
