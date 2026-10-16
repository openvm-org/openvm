// Lean compiler output
// Module: Mathlib.Data.Subtype
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.Basic public import Mathlib.Tactic.AdaptationNote public import Mathlib.Tactic.Simps
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
LEAN_EXPORT lean_object* lp_mathlib_Subtype_restrict___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_coind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_coind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instHasEquiv__mathlib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instSetoid__mathlib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_restrict___redArg(lean_object* v_f_1_, lean_object* v_x_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_f_1_, v_x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_restrict(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_p_6_, lean_object* v_f_7_, lean_object* v_x_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_apply_1(v_f_7_, v_x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_coind___redArg(lean_object* v_f_10_, lean_object* v_a_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_apply_1(v_f_10_, v_a_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_coind(lean_object* v_00_u03b1_13_, lean_object* v_00_u03b2_14_, lean_object* v_f_15_, lean_object* v_p_16_, lean_object* v_h_17_, lean_object* v_a_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_apply_1(v_f_15_, v_a_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_map___redArg(lean_object* v_f_20_, lean_object* v_x_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_apply_1(v_f_20_, v_x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_map(lean_object* v_00_u03b1_23_, lean_object* v_00_u03b2_24_, lean_object* v_p_25_, lean_object* v_q_26_, lean_object* v_f_27_, lean_object* v_h_28_, lean_object* v_x_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_apply_1(v_f_27_, v_x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instHasEquiv__mathlib(lean_object* v_00_u03b1_31_, lean_object* v_inst_32_, lean_object* v_p_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lean_box(0);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instSetoid__mathlib(lean_object* v_00_u03b1_35_, lean_object* v_inst_36_, lean_object* v_p_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lean_box(0);
return v___x_38_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_AdaptationNote(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_AdaptationNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_AdaptationNote(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_AdaptationNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Subtype(builtin);
}
#ifdef __cplusplus
}
#endif
