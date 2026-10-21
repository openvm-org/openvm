// Lean compiler output
// Module: Mathlib.Data.Option.NAry
// Imports: public import Init public meta import Init public import Mathlib.Init
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
LEAN_EXPORT lean_object* lp_mathlib_Option_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_map_u2082___redArg(lean_object* v_f_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_4_; 
lean_dec(v_b_3_);
lean_dec(v_f_1_);
v___x_4_ = lean_box(0);
return v___x_4_;
}
else
{
if (lean_obj_tag(v_b_3_) == 0)
{
lean_object* v___x_5_; 
lean_dec_ref_known(v_a_2_, 1);
lean_dec(v_f_1_);
v___x_5_ = lean_box(0);
return v___x_5_;
}
else
{
lean_object* v_val_6_; lean_object* v_val_7_; lean_object* v___x_9_; uint8_t v_isShared_10_; uint8_t v_isSharedCheck_15_; 
v_val_6_ = lean_ctor_get(v_a_2_, 0);
lean_inc(v_val_6_);
lean_dec_ref_known(v_a_2_, 1);
v_val_7_ = lean_ctor_get(v_b_3_, 0);
v_isSharedCheck_15_ = !lean_is_exclusive(v_b_3_);
if (v_isSharedCheck_15_ == 0)
{
v___x_9_ = v_b_3_;
v_isShared_10_ = v_isSharedCheck_15_;
goto v_resetjp_8_;
}
else
{
lean_inc(v_val_7_);
lean_dec(v_b_3_);
v___x_9_ = lean_box(0);
v_isShared_10_ = v_isSharedCheck_15_;
goto v_resetjp_8_;
}
v_resetjp_8_:
{
lean_object* v___x_11_; lean_object* v___x_13_; 
v___x_11_ = lean_apply_2(v_f_1_, v_val_6_, v_val_7_);
if (v_isShared_10_ == 0)
{
lean_ctor_set(v___x_9_, 0, v___x_11_);
v___x_13_ = v___x_9_;
goto v_reusejp_12_;
}
else
{
lean_object* v_reuseFailAlloc_14_; 
v_reuseFailAlloc_14_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_14_, 0, v___x_11_);
v___x_13_ = v_reuseFailAlloc_14_;
goto v_reusejp_12_;
}
v_reusejp_12_:
{
return v___x_13_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_map_u2082(lean_object* v_00_u03b1_16_, lean_object* v_00_u03b2_17_, lean_object* v_00_u03b3_18_, lean_object* v_f_19_, lean_object* v_a_20_, lean_object* v_b_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Option_map_u2082___redArg(v_f_19_, v_a_20_, v_b_21_);
return v___x_22_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
}
#ifdef __cplusplus
}
#endif
