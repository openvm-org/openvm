// Lean compiler output
// Module: Mathlib.Order.SuccPred.WithBot
// Imports: public import Init public meta import Init public import Mathlib.Order.SuccPred.Basic
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
lean_object* lp_mathlib_Order_succ___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_recBotCoe___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Order_pred___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_recTopCoe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_succ___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_succ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_succ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_succ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_pred___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_pred___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_pred(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_pred___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_succ___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_a_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_alloc_closure((void*)(lp_mathlib_Order_succ___boxed), 4, 3);
lean_closure_set(v___x_5_, 0, lean_box(0));
lean_closure_set(v___x_5_, 1, v_inst_1_);
lean_closure_set(v___x_5_, 2, v_inst_3_);
v___x_6_ = lp_mathlib_WithBot_recBotCoe___redArg(v_inst_2_, v___x_5_, v_a_4_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_succ___redArg___boxed(lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_a_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_WithBot_succ___redArg(v_inst_7_, v_inst_8_, v_inst_9_, v_a_10_);
lean_dec(v_inst_8_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_succ(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_a_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_WithBot_succ___redArg(v_inst_13_, v_inst_14_, v_inst_15_, v_a_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_succ___boxed(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_a_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_WithBot_succ(v_00_u03b1_18_, v_inst_19_, v_inst_20_, v_inst_21_, v_a_22_);
lean_dec(v_inst_20_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_pred___redArg(lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_a_27_){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_28_ = lean_alloc_closure((void*)(lp_mathlib_Order_pred___boxed), 4, 3);
lean_closure_set(v___x_28_, 0, lean_box(0));
lean_closure_set(v___x_28_, 1, v_inst_24_);
lean_closure_set(v___x_28_, 2, v_inst_26_);
v___x_29_ = lp_mathlib_WithTop_recTopCoe___redArg(v_inst_25_, v___x_28_, v_a_27_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_pred___redArg___boxed(lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_a_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_WithTop_pred___redArg(v_inst_30_, v_inst_31_, v_inst_32_, v_a_33_);
lean_dec(v_inst_31_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_pred(lean_object* v_00_u03b1_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_a_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_WithTop_pred___redArg(v_inst_36_, v_inst_37_, v_inst_38_, v_a_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_pred___boxed(lean_object* v_00_u03b1_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_a_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_WithTop_pred(v_00_u03b1_41_, v_inst_42_, v_inst_43_, v_inst_44_, v_a_45_);
lean_dec(v_inst_43_);
return v_res_46_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_WithBot(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_SuccPred_WithBot(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SuccPred_WithBot(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_SuccPred_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_SuccPred_WithBot(builtin);
}
#ifdef __cplusplus
}
#endif
