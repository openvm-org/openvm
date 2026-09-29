// Lean compiler output
// Module: Mathlib.Data.List.Lattice
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Basic
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Lattice_0__List_bagInter_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Lattice_0__List_bagInter_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Lattice_0__List_bagInter_match__1_splitter___redArg(lean_object* v_x_1_, lean_object* v_x_2_, lean_object* v_h__1_3_, lean_object* v_h__2_4_, lean_object* v_h__3_5_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_6_; 
lean_dec(v_h__3_5_);
lean_dec(v_h__2_4_);
v___x_6_ = lean_apply_1(v_h__1_3_, v_x_2_);
return v___x_6_;
}
else
{
lean_dec(v_h__1_3_);
if (lean_obj_tag(v_x_2_) == 0)
{
lean_object* v___x_7_; 
lean_dec(v_h__3_5_);
v___x_7_ = lean_apply_2(v_h__2_4_, v_x_1_, lean_box(0));
return v___x_7_;
}
else
{
lean_object* v_head_8_; lean_object* v_tail_9_; lean_object* v___x_10_; 
lean_dec(v_h__2_4_);
v_head_8_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_head_8_);
v_tail_9_ = lean_ctor_get(v_x_1_, 1);
lean_inc(v_tail_9_);
lean_dec_ref_known(v_x_1_, 2);
v___x_10_ = lean_apply_4(v_h__3_5_, v_head_8_, v_tail_9_, v_x_2_, lean_box(0));
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Lattice_0__List_bagInter_match__1_splitter(lean_object* v_00_u03b1_11_, lean_object* v_motive_12_, lean_object* v_x_13_, lean_object* v_x_14_, lean_object* v_h__1_15_, lean_object* v_h__2_16_, lean_object* v_h__3_17_){
_start:
{
if (lean_obj_tag(v_x_13_) == 0)
{
lean_object* v___x_18_; 
lean_dec(v_h__3_17_);
lean_dec(v_h__2_16_);
v___x_18_ = lean_apply_1(v_h__1_15_, v_x_14_);
return v___x_18_;
}
else
{
lean_dec(v_h__1_15_);
if (lean_obj_tag(v_x_14_) == 0)
{
lean_object* v___x_19_; 
lean_dec(v_h__3_17_);
v___x_19_ = lean_apply_2(v_h__2_16_, v_x_13_, lean_box(0));
return v___x_19_;
}
else
{
lean_object* v_head_20_; lean_object* v_tail_21_; lean_object* v___x_22_; 
lean_dec(v_h__2_16_);
v_head_20_ = lean_ctor_get(v_x_13_, 0);
lean_inc(v_head_20_);
v_tail_21_ = lean_ctor_get(v_x_13_, 1);
lean_inc(v_tail_21_);
lean_dec_ref_known(v_x_13_, 2);
v___x_22_ = lean_apply_4(v_h__3_17_, v_head_20_, v_tail_21_, v_x_14_, lean_box(0));
return v___x_22_;
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Lattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Lattice(builtin);
}
#ifdef __cplusplus
}
#endif
