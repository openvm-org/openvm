// Lean compiler output
// Module: Mathlib.GroupTheory.Congruence.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Opposites public import Mathlib.GroupTheory.Congruence.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Con_op(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_op___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_op(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_op___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_unop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_unop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_unop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_unop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_orderIsoOp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_orderIsoOp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_orderIsoOp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_orderIsoOp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_op(lean_object* v_M_1_, lean_object* v_inst_2_, lean_object* v_c_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_op___boxed(lean_object* v_M_5_, lean_object* v_inst_6_, lean_object* v_c_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Con_op(v_M_5_, v_inst_6_, v_c_7_);
lean_dec(v_inst_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_op(lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_c_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_op___boxed(lean_object* v_M_13_, lean_object* v_inst_14_, lean_object* v_c_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_AddCon_op(v_M_13_, v_inst_14_, v_c_15_);
lean_dec(v_inst_14_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_unop(lean_object* v_M_17_, lean_object* v_inst_18_, lean_object* v_c_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_box(0);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_unop___boxed(lean_object* v_M_21_, lean_object* v_inst_22_, lean_object* v_c_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Con_unop(v_M_21_, v_inst_22_, v_c_23_);
lean_dec(v_inst_22_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_unop(lean_object* v_M_25_, lean_object* v_inst_26_, lean_object* v_c_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_box(0);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_unop___boxed(lean_object* v_M_29_, lean_object* v_inst_30_, lean_object* v_c_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_AddCon_unop(v_M_29_, v_inst_30_, v_c_31_);
lean_dec(v_inst_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_orderIsoOp___redArg(lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
lean_inc(v_inst_33_);
v___x_34_ = lean_alloc_closure((void*)(lp_mathlib_Con_op___boxed), 3, 2);
lean_closure_set(v___x_34_, 0, lean_box(0));
lean_closure_set(v___x_34_, 1, v_inst_33_);
v___x_35_ = lean_alloc_closure((void*)(lp_mathlib_Con_unop___boxed), 3, 2);
lean_closure_set(v___x_35_, 0, lean_box(0));
lean_closure_set(v___x_35_, 1, v_inst_33_);
v___x_36_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_36_, 0, v___x_34_);
lean_ctor_set(v___x_36_, 1, v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_orderIsoOp(lean_object* v_M_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Con_orderIsoOp___redArg(v_inst_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_orderIsoOp___redArg(lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
lean_inc(v_inst_40_);
v___x_41_ = lean_alloc_closure((void*)(lp_mathlib_AddCon_op___boxed), 3, 2);
lean_closure_set(v___x_41_, 0, lean_box(0));
lean_closure_set(v___x_41_, 1, v_inst_40_);
v___x_42_ = lean_alloc_closure((void*)(lp_mathlib_AddCon_unop___boxed), 3, 2);
lean_closure_set(v___x_42_, 0, lean_box(0));
lean_closure_set(v___x_42_, 1, v_inst_40_);
v___x_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_43_, 0, v___x_41_);
lean_ctor_set(v___x_43_, 1, v___x_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_orderIsoOp(lean_object* v_M_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_AddCon_orderIsoOp___redArg(v_inst_45_);
return v___x_46_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Congruence_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Congruence_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Congruence_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
