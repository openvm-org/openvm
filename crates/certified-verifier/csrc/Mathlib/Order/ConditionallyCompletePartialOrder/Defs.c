// Lean compiler output
// Module: Mathlib.Order.ConditionallyCompletePartialOrder.Defs
// Imports: public import Init public meta import Init public import Mathlib.Order.Bounds.Defs public import Mathlib.Order.Directed public import Mathlib.Order.SetNotation
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
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toConditionallyCompletePartialOrderSup_2_; lean_object* v_toInfSet_3_; lean_object* v_toPartialOrder_4_; lean_object* v___x_6_; uint8_t v_isShared_7_; uint8_t v_isSharedCheck_11_; 
v_toConditionallyCompletePartialOrderSup_2_ = lean_ctor_get(v_self_1_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_2_);
v_toInfSet_3_ = lean_ctor_get(v_self_1_, 1);
lean_inc(v_toInfSet_3_);
lean_dec_ref(v_self_1_);
v_toPartialOrder_4_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_2_, 0);
v_isSharedCheck_11_ = !lean_is_exclusive(v_toConditionallyCompletePartialOrderSup_2_);
if (v_isSharedCheck_11_ == 0)
{
lean_object* v_unused_12_; 
v_unused_12_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_2_, 1);
lean_dec(v_unused_12_);
v___x_6_ = v_toConditionallyCompletePartialOrderSup_2_;
v_isShared_7_ = v_isSharedCheck_11_;
goto v_resetjp_5_;
}
else
{
lean_inc(v_toPartialOrder_4_);
lean_dec(v_toConditionallyCompletePartialOrderSup_2_);
v___x_6_ = lean_box(0);
v_isShared_7_ = v_isSharedCheck_11_;
goto v_resetjp_5_;
}
v_resetjp_5_:
{
lean_object* v___x_9_; 
if (v_isShared_7_ == 0)
{
lean_ctor_set(v___x_6_, 1, v_toInfSet_3_);
v___x_9_ = v___x_6_;
goto v_reusejp_8_;
}
else
{
lean_object* v_reuseFailAlloc_10_; 
v_reuseFailAlloc_10_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_10_, 0, v_toPartialOrder_4_);
lean_ctor_set(v_reuseFailAlloc_10_, 1, v_toInfSet_3_);
v___x_9_ = v_reuseFailAlloc_10_;
goto v_reusejp_8_;
}
v_reusejp_8_:
{
return v___x_9_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf(lean_object* v_00_u03b1_13_, lean_object* v_self_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_ConditionallyCompletePartialOrder_toConditionallyCompletePartialOrderInf___redArg(v_self_14_);
return v___x_15_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
