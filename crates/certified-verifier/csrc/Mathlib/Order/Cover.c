// Lean compiler output
// Module: Mathlib.Order.Cover
// Imports: public import Init public meta import Init public import Mathlib.Order.Antisymmetrization public import Mathlib.Order.Interval.Set.OrdConnected public import Mathlib.Order.Interval.Set.WithBotTop
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
uint8_t l_Bool_instDecidableLe(uint8_t, uint8_t);
uint8_t l_Bool_instDecidableLt(uint8_t, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib_Bool_instDecidableRelWCovBy(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDecidableRelWCovBy___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Bool_instDecidableRelCovBy(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDecidableRelCovBy___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Bool_instDecidableRelWCovBy(uint8_t v_x_1_, uint8_t v_x_2_){
_start:
{
uint8_t v___x_3_; 
v___x_3_ = l_Bool_instDecidableLe(v_x_1_, v_x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDecidableRelWCovBy___boxed(lean_object* v_x_4_, lean_object* v_x_5_){
_start:
{
uint8_t v_x_8__boxed_6_; uint8_t v_x_9__boxed_7_; uint8_t v_res_8_; lean_object* v_r_9_; 
v_x_8__boxed_6_ = lean_unbox(v_x_4_);
v_x_9__boxed_7_ = lean_unbox(v_x_5_);
v_res_8_ = lp_mathlib_Bool_instDecidableRelWCovBy(v_x_8__boxed_6_, v_x_9__boxed_7_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_instDecidableRelCovBy(uint8_t v_x_10_, uint8_t v_x_11_){
_start:
{
uint8_t v___x_12_; 
v___x_12_ = l_Bool_instDecidableLt(v_x_10_, v_x_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_instDecidableRelCovBy___boxed(lean_object* v_x_13_, lean_object* v_x_14_){
_start:
{
uint8_t v_x_8__boxed_15_; uint8_t v_x_9__boxed_16_; uint8_t v_res_17_; lean_object* v_r_18_; 
v_x_8__boxed_15_ = lean_unbox(v_x_13_);
v_x_9__boxed_16_ = lean_unbox(v_x_14_);
v_res_17_ = lp_mathlib_Bool_instDecidableRelCovBy(v_x_8__boxed_15_, v_x_9__boxed_16_);
v_r_18_ = lean_box(v_res_17_);
return v_r_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Antisymmetrization(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_WithBotTop(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Antisymmetrization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_WithBotTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Antisymmetrization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_WithBotTop(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Antisymmetrization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_OrdConnected(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_WithBotTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Cover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Cover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Cover(builtin);
}
#ifdef __cplusplus
}
#endif
