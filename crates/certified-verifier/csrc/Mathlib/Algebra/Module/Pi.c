// Lean compiler output
// Module: Mathlib.Algebra.Module.Pi
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Action.Pi public import Mathlib.Algebra.Module.Defs public import Mathlib.Algebra.Regular.SMul public import Mathlib.Algebra.Ring.Pi
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
lean_object* lp_mathlib_Pi_distribMulAction___redArg(lean_object*);
lean_object* lp_mathlib_Pi_distribMulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_module___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_module_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_module_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_module_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_module___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_, lean_object* v___y_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_apply_3(v_inst_1_, v_i_2_, v___y_3_, v___y_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_module___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; lean_object* v___x_8_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_Pi_module___redArg___lam__0), 4, 1);
lean_closure_set(v___f_7_, 0, v_inst_6_);
v___x_8_ = lp_mathlib_Pi_distribMulAction___redArg(v___f_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_module(lean_object* v_I_9_, lean_object* v_f_10_, lean_object* v_00_u03b1_11_, lean_object* v_r_12_, lean_object* v_m_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Pi_module___redArg(v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_module___boxed(lean_object* v_I_16_, lean_object* v_f_17_, lean_object* v_00_u03b1_18_, lean_object* v_r_19_, lean_object* v_m_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Pi_module(v_I_16_, v_f_17_, v_00_u03b1_18_, v_r_19_, v_m_20_, v_inst_21_);
lean_dec_ref(v_m_20_);
lean_dec_ref(v_r_19_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module___redArg___lam__0(lean_object* v_inst_23_, lean_object* v_i_24_, lean_object* v___y_25_, lean_object* v___y_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_apply_2(v_inst_23_, v___y_25_, v___y_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module___redArg___lam__0___boxed(lean_object* v_inst_28_, lean_object* v_i_29_, lean_object* v___y_30_, lean_object* v___y_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Pi_Function_module___redArg___lam__0(v_inst_28_, v_i_29_, v___y_30_, v___y_31_);
lean_dec(v_i_29_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module___redArg(lean_object* v_inst_33_){
_start:
{
lean_object* v___f_34_; lean_object* v___x_35_; 
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_Pi_Function_module___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_34_, 0, v_inst_33_);
v___x_35_ = lp_mathlib_Pi_module___redArg(v___f_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module(lean_object* v_I_36_, lean_object* v_00_u03b1_37_, lean_object* v_00_u03b2_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_Pi_Function_module___redArg(v_inst_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_Function_module___boxed(lean_object* v_I_43_, lean_object* v_00_u03b1_44_, lean_object* v_00_u03b2_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Pi_Function_module(v_I_43_, v_00_u03b1_44_, v_00_u03b2_45_, v_inst_46_, v_inst_47_, v_inst_48_);
lean_dec_ref(v_inst_47_);
lean_dec_ref(v_inst_46_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_module_x27___redArg(lean_object* v_inst_50_){
_start:
{
lean_object* v___f_51_; lean_object* v___x_52_; 
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_Pi_module___redArg___lam__0), 4, 1);
lean_closure_set(v___f_51_, 0, v_inst_50_);
v___x_52_ = lp_mathlib_Pi_distribMulAction_x27___redArg(v___f_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_module_x27(lean_object* v_I_53_, lean_object* v_f_54_, lean_object* v_g_55_, lean_object* v_r_56_, lean_object* v_m_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Pi_module_x27___redArg(v_inst_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_module_x27___boxed(lean_object* v_I_60_, lean_object* v_f_61_, lean_object* v_g_62_, lean_object* v_r_63_, lean_object* v_m_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_Pi_module_x27(v_I_60_, v_f_61_, v_g_62_, v_r_63_, v_m_64_, v_inst_65_);
lean_dec_ref(v_m_64_);
lean_dec_ref(v_r_63_);
return v_res_66_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_SMul(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_SMul(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Pi(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
