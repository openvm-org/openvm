// Lean compiler output
// Module: Mathlib.Dynamics.PeriodicPts.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Defs public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Algebra.Order.Sub.Basic public import Mathlib.Data.List.Cycle public import Mathlib.Data.PNat.Notation public import Mathlib.Dynamics.FixedPoints.Basic
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
lean_object* lp_mathlib_Nat_iterate(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Function_IsFixedPt_decidable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq___redArg(lean_object* v_inst_1_, lean_object* v_f_2_, lean_object* v_n_3_, lean_object* v_x_4_){
_start:
{
lean_object* v___x_5_; uint8_t v___x_6_; 
v___x_5_ = lean_alloc_closure((void*)(lp_mathlib_Nat_iterate), 4, 3);
lean_closure_set(v___x_5_, 0, lean_box(0));
lean_closure_set(v___x_5_, 1, v_f_2_);
lean_closure_set(v___x_5_, 2, v_n_3_);
v___x_6_ = lp_mathlib_Function_IsFixedPt_decidable___redArg(v_inst_1_, v___x_5_, v_x_4_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq___redArg___boxed(lean_object* v_inst_7_, lean_object* v_f_8_, lean_object* v_n_9_, lean_object* v_x_10_){
_start:
{
uint8_t v_res_11_; lean_object* v_r_12_; 
v_res_11_ = lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq___redArg(v_inst_7_, v_f_8_, v_n_9_, v_x_10_);
v_r_12_ = lean_box(v_res_11_);
return v_r_12_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_, lean_object* v_f_15_, lean_object* v_n_16_, lean_object* v_x_17_){
_start:
{
uint8_t v___x_18_; 
v___x_18_ = lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq___redArg(v_inst_14_, v_f_15_, v_n_16_, v_x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq___boxed(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_f_21_, lean_object* v_n_22_, lean_object* v_x_23_){
_start:
{
uint8_t v_res_24_; lean_object* v_r_25_; 
v_res_24_ = lp_mathlib_Function_IsPeriodicPt_instDecidableOfDecidableEq(v_00_u03b1_19_, v_inst_20_, v_f_21_, v_n_22_, v_x_23_);
v_r_25_ = lean_box(v_res_24_);
return v_r_25_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Cycle(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_PNat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Dynamics_FixedPoints_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Cycle(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_PNat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Dynamics_FixedPoints_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Cycle(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_PNat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Dynamics_FixedPoints_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Cycle(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_PNat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Dynamics_FixedPoints_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Dynamics_PeriodicPts_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
