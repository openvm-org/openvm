// Lean compiler output
// Module: Mathlib.Data.Ordering.Basic
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
LEAN_EXPORT uint8_t lp_mathlib_cmpUsing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cmpUsing___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_cmpUsing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cmpUsing___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_cmp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cmp___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_cmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_cmp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_cmpUsing___redArg(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
lean_inc_ref(v_inst_1_);
lean_inc(v_b_3_);
lean_inc(v_a_2_);
v___x_4_ = lean_apply_2(v_inst_1_, v_a_2_, v_b_3_);
v___x_5_ = lean_unbox(v___x_4_);
if (v___x_5_ == 0)
{
lean_object* v___x_6_; uint8_t v___x_7_; 
v___x_6_ = lean_apply_2(v_inst_1_, v_b_3_, v_a_2_);
v___x_7_ = lean_unbox(v___x_6_);
if (v___x_7_ == 0)
{
uint8_t v___x_8_; 
v___x_8_ = 1;
return v___x_8_;
}
else
{
uint8_t v___x_9_; 
v___x_9_ = 2;
return v___x_9_;
}
}
else
{
uint8_t v___x_10_; 
lean_dec(v_b_3_);
lean_dec(v_a_2_);
lean_dec_ref(v_inst_1_);
v___x_10_ = 0;
return v___x_10_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_cmpUsing___redArg___boxed(lean_object* v_inst_11_, lean_object* v_a_12_, lean_object* v_b_13_){
_start:
{
uint8_t v_res_14_; lean_object* v_r_15_; 
v_res_14_ = lp_mathlib_cmpUsing___redArg(v_inst_11_, v_a_12_, v_b_13_);
v_r_15_ = lean_box(v_res_14_);
return v_r_15_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_cmpUsing(lean_object* v_00_u03b1_16_, lean_object* v_lt_17_, lean_object* v_inst_18_, lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
uint8_t v___x_21_; 
v___x_21_ = lp_mathlib_cmpUsing___redArg(v_inst_18_, v_a_19_, v_b_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cmpUsing___boxed(lean_object* v_00_u03b1_22_, lean_object* v_lt_23_, lean_object* v_inst_24_, lean_object* v_a_25_, lean_object* v_b_26_){
_start:
{
uint8_t v_res_27_; lean_object* v_r_28_; 
v_res_27_ = lp_mathlib_cmpUsing(v_00_u03b1_22_, v_lt_23_, v_inst_24_, v_a_25_, v_b_26_);
v_r_28_ = lean_box(v_res_27_);
return v_r_28_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_cmp___redArg(lean_object* v_inst_29_, lean_object* v_a_30_, lean_object* v_b_31_){
_start:
{
uint8_t v___x_32_; 
v___x_32_ = lp_mathlib_cmpUsing___redArg(v_inst_29_, v_a_30_, v_b_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cmp___redArg___boxed(lean_object* v_inst_33_, lean_object* v_a_34_, lean_object* v_b_35_){
_start:
{
uint8_t v_res_36_; lean_object* v_r_37_; 
v_res_36_ = lp_mathlib_cmp___redArg(v_inst_33_, v_a_34_, v_b_35_);
v_r_37_ = lean_box(v_res_36_);
return v_r_37_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_cmp(lean_object* v_00_u03b1_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_a_41_, lean_object* v_b_42_){
_start:
{
uint8_t v___x_43_; 
v___x_43_ = lp_mathlib_cmpUsing___redArg(v_inst_40_, v_a_41_, v_b_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_cmp___boxed(lean_object* v_00_u03b1_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_a_47_, lean_object* v_b_48_){
_start:
{
uint8_t v_res_49_; lean_object* v_r_50_; 
v_res_49_ = lp_mathlib_cmp(v_00_u03b1_44_, v_inst_45_, v_inst_46_, v_a_47_, v_b_48_);
v_r_50_ = lean_box(v_res_49_);
return v_r_50_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Ordering_Basic(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Ordering_Basic(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Ordering_Basic(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Data_Ordering_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Ordering_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Ordering_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
