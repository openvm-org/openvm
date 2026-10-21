// Lean compiler output
// Module: Mathlib.Algebra.Group.Graph
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Ker
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
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mgraph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mgraph___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mgraph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mgraph___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_graph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_graph___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_graph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_graph___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mgraph(lean_object* v_G_1_, lean_object* v_H_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_f_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mgraph___boxed(lean_object* v_G_7_, lean_object* v_H_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_f_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_MonoidHom_mgraph(v_G_7_, v_H_8_, v_inst_9_, v_inst_10_, v_f_11_);
lean_dec(v_f_11_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mgraph(lean_object* v_G_13_, lean_object* v_H_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_f_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_box(0);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mgraph___boxed(lean_object* v_G_19_, lean_object* v_H_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_f_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_AddMonoidHom_mgraph(v_G_19_, v_H_20_, v_inst_21_, v_inst_22_, v_f_23_);
lean_dec(v_f_23_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_graph(lean_object* v_G_25_, lean_object* v_H_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_f_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_box(0);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_graph___boxed(lean_object* v_G_31_, lean_object* v_H_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_f_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_MonoidHom_graph(v_G_31_, v_H_32_, v_inst_33_, v_inst_34_, v_f_35_);
lean_dec(v_f_35_);
lean_dec_ref(v_inst_34_);
lean_dec_ref(v_inst_33_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_graph(lean_object* v_G_37_, lean_object* v_H_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_f_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lean_box(0);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_graph___boxed(lean_object* v_G_43_, lean_object* v_H_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_f_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_AddMonoidHom_graph(v_G_43_, v_H_44_, v_inst_45_, v_inst_46_, v_f_47_);
lean_dec(v_f_47_);
lean_dec_ref(v_inst_46_);
lean_dec_ref(v_inst_45_);
return v_res_48_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Graph(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Graph(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Graph(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Graph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Graph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Graph(builtin);
}
#ifdef __cplusplus
}
#endif
