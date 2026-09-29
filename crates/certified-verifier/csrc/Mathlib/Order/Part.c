// Lean compiler output
// Module: Mathlib.Order.Part
// Imports: public import Init public meta import Init public import Mathlib.Data.Part public import Mathlib.Order.Hom.Basic public import Mathlib.Tactic.Common public import Mathlib.Tactic.Attr.Core
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
lean_object* lp_mathlib_Part_bind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind___redArg___lam__0(lean_object* v_g_1_, lean_object* v_x_2_, lean_object* v___y_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_apply_3(v_g_1_, v_x_2_, v___y_3_, lean_box(0));
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind___redArg___lam__1(lean_object* v_g_6_, lean_object* v_f_7_, lean_object* v_x_8_, lean_object* v___y_9_){
_start:
{
lean_object* v___f_10_; lean_object* v___x_11_; lean_object* v___x_41__overap_12_; lean_object* v___x_13_; 
lean_inc(v_x_8_);
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_partBind___redArg___lam__0), 4, 2);
lean_closure_set(v___f_10_, 0, v_g_6_);
lean_closure_set(v___f_10_, 1, v_x_8_);
v___x_11_ = lean_apply_1(v_f_7_, v_x_8_);
v___x_41__overap_12_ = lp_mathlib_Part_bind___redArg(v___x_11_, v___f_10_);
v___x_13_ = lean_apply_1(v___x_41__overap_12_, lean_box(0));
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind___redArg(lean_object* v_f_14_, lean_object* v_g_15_){
_start:
{
lean_object* v___f_16_; 
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_partBind___redArg___lam__1), 4, 2);
lean_closure_set(v___f_16_, 0, v_g_15_);
lean_closure_set(v___f_16_, 1, v_f_14_);
return v___f_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind(lean_object* v_00_u03b1_17_, lean_object* v_00_u03b2_18_, lean_object* v_00_u03b3_19_, lean_object* v_inst_20_, lean_object* v_f_21_, lean_object* v_g_22_){
_start:
{
lean_object* v___f_23_; 
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_partBind___redArg___lam__1), 4, 2);
lean_closure_set(v___f_23_, 0, v_g_22_);
lean_closure_set(v___f_23_, 1, v_f_21_);
return v___f_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_partBind___boxed(lean_object* v_00_u03b1_24_, lean_object* v_00_u03b2_25_, lean_object* v_00_u03b3_26_, lean_object* v_inst_27_, lean_object* v_f_28_, lean_object* v_g_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_OrderHom_partBind(v_00_u03b1_24_, v_00_u03b2_25_, v_00_u03b3_26_, v_inst_27_, v_f_28_, v_g_29_);
lean_dec_ref(v_inst_27_);
return v_res_30_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Part(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Part(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Part(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Part(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Part(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Part(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Part(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Part(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Part(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Part(builtin);
}
#ifdef __cplusplus
}
#endif
