// Lean compiler output
// Module: Mathlib.Algebra.Group.WithOne.Map
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.WithOne.Defs public import Mathlib.Data.Option.NAry
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
lean_object* lp_mathlib_Option_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_map___redArg(lean_object* v_f_1_, lean_object* v_a_2_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_3_; 
lean_dec(v_f_1_);
v___x_3_ = lean_box(0);
return v___x_3_;
}
else
{
lean_object* v_val_4_; lean_object* v___x_6_; uint8_t v_isShared_7_; uint8_t v_isSharedCheck_12_; 
v_val_4_ = lean_ctor_get(v_a_2_, 0);
v_isSharedCheck_12_ = !lean_is_exclusive(v_a_2_);
if (v_isSharedCheck_12_ == 0)
{
v___x_6_ = v_a_2_;
v_isShared_7_ = v_isSharedCheck_12_;
goto v_resetjp_5_;
}
else
{
lean_inc(v_val_4_);
lean_dec(v_a_2_);
v___x_6_ = lean_box(0);
v_isShared_7_ = v_isSharedCheck_12_;
goto v_resetjp_5_;
}
v_resetjp_5_:
{
lean_object* v___x_8_; lean_object* v___x_10_; 
v___x_8_ = lean_apply_1(v_f_1_, v_val_4_);
if (v_isShared_7_ == 0)
{
lean_ctor_set(v___x_6_, 0, v___x_8_);
v___x_10_ = v___x_6_;
goto v_reusejp_9_;
}
else
{
lean_object* v_reuseFailAlloc_11_; 
v_reuseFailAlloc_11_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_11_, 0, v___x_8_);
v___x_10_ = v_reuseFailAlloc_11_;
goto v_reusejp_9_;
}
v_reusejp_9_:
{
return v___x_10_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_map(lean_object* v_00_u03b1_13_, lean_object* v_00_u03b2_14_, lean_object* v_f_15_, lean_object* v_a_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_mathlib_WithOne_map___redArg(v_f_15_, v_a_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map___redArg(lean_object* v_f_18_, lean_object* v_a_19_){
_start:
{
if (lean_obj_tag(v_a_19_) == 0)
{
lean_object* v___x_20_; 
lean_dec(v_f_18_);
v___x_20_ = lean_box(0);
return v___x_20_;
}
else
{
lean_object* v_val_21_; lean_object* v___x_23_; uint8_t v_isShared_24_; uint8_t v_isSharedCheck_29_; 
v_val_21_ = lean_ctor_get(v_a_19_, 0);
v_isSharedCheck_29_ = !lean_is_exclusive(v_a_19_);
if (v_isSharedCheck_29_ == 0)
{
v___x_23_ = v_a_19_;
v_isShared_24_ = v_isSharedCheck_29_;
goto v_resetjp_22_;
}
else
{
lean_inc(v_val_21_);
lean_dec(v_a_19_);
v___x_23_ = lean_box(0);
v_isShared_24_ = v_isSharedCheck_29_;
goto v_resetjp_22_;
}
v_resetjp_22_:
{
lean_object* v___x_25_; lean_object* v___x_27_; 
v___x_25_ = lean_apply_1(v_f_18_, v_val_21_);
if (v_isShared_24_ == 0)
{
lean_ctor_set(v___x_23_, 0, v___x_25_);
v___x_27_ = v___x_23_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_28_; 
v_reuseFailAlloc_28_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_28_, 0, v___x_25_);
v___x_27_ = v_reuseFailAlloc_28_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
return v___x_27_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map(lean_object* v_00_u03b1_30_, lean_object* v_00_u03b2_31_, lean_object* v_f_32_, lean_object* v_a_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_WithZero_map___redArg(v_f_32_, v_a_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_map_u2082___redArg(lean_object* v_f_35_, lean_object* v_a_36_, lean_object* v_b_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Option_map_u2082___redArg(v_f_35_, v_a_36_, v_b_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_map_u2082(lean_object* v_00_u03b1_39_, lean_object* v_00_u03b2_40_, lean_object* v_00_u03b3_41_, lean_object* v_f_42_, lean_object* v_a_43_, lean_object* v_b_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_Option_map_u2082___redArg(v_f_42_, v_a_43_, v_b_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_u2082___redArg(lean_object* v_f_46_, lean_object* v_a_47_, lean_object* v_b_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Option_map_u2082___redArg(v_f_46_, v_a_47_, v_b_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_map_u2082(lean_object* v_00_u03b1_50_, lean_object* v_00_u03b2_51_, lean_object* v_00_u03b3_52_, lean_object* v_f_53_, lean_object* v_a_54_, lean_object* v_b_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Option_map_u2082___redArg(v_f_53_, v_a_54_, v_b_55_);
return v___x_56_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_WithOne_Map(builtin);
}
#ifdef __cplusplus
}
#endif
