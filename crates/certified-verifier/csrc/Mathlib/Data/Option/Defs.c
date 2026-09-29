// Lean compiler output
// Module: Mathlib.Data.Option.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Option_traverse___redArg___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Option_traverse___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Option_traverse___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Option_traverse___redArg___closed__0 = (const lean_object*)&lp_mathlib_Option_traverse___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Option_traverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_traverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_elim_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_elim_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_elim_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_elim_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_traverse___redArg___lam__0(lean_object* v_val_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2_, 0, v_val_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_traverse___redArg(lean_object* v_inst_4_, lean_object* v_f_5_, lean_object* v_a_6_){
_start:
{
if (lean_obj_tag(v_a_6_) == 0)
{
lean_object* v_toPure_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
lean_dec(v_f_5_);
v_toPure_7_ = lean_ctor_get(v_inst_4_, 1);
lean_inc(v_toPure_7_);
lean_dec_ref(v_inst_4_);
v___x_8_ = lean_box(0);
v___x_9_ = lean_apply_2(v_toPure_7_, lean_box(0), v___x_8_);
return v___x_9_;
}
else
{
lean_object* v_toFunctor_10_; lean_object* v_val_11_; lean_object* v_map_12_; lean_object* v___f_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v_toFunctor_10_ = lean_ctor_get(v_inst_4_, 0);
lean_inc_ref(v_toFunctor_10_);
lean_dec_ref(v_inst_4_);
v_val_11_ = lean_ctor_get(v_a_6_, 0);
lean_inc(v_val_11_);
lean_dec_ref_known(v_a_6_, 1);
v_map_12_ = lean_ctor_get(v_toFunctor_10_, 0);
lean_inc(v_map_12_);
lean_dec_ref(v_toFunctor_10_);
v___f_13_ = ((lean_object*)(lp_mathlib_Option_traverse___redArg___closed__0));
v___x_14_ = lean_apply_1(v_f_5_, v_val_11_);
v___x_15_ = lean_apply_4(v_map_12_, lean_box(0), lean_box(0), v___f_13_, v___x_14_);
return v___x_15_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_traverse(lean_object* v_F_16_, lean_object* v_inst_17_, lean_object* v_00_u03b1_18_, lean_object* v_00_u03b2_19_, lean_object* v_f_20_, lean_object* v_a_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Option_traverse___redArg(v_inst_17_, v_f_20_, v_a_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_elim_x27___redArg(lean_object* v_b_23_, lean_object* v_f_24_, lean_object* v_x_25_){
_start:
{
if (lean_obj_tag(v_x_25_) == 0)
{
lean_dec(v_f_24_);
lean_inc(v_b_23_);
return v_b_23_;
}
else
{
lean_object* v_val_26_; lean_object* v___x_27_; 
v_val_26_ = lean_ctor_get(v_x_25_, 0);
lean_inc(v_val_26_);
lean_dec_ref_known(v_x_25_, 1);
v___x_27_ = lean_apply_1(v_f_24_, v_val_26_);
return v___x_27_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_elim_x27___redArg___boxed(lean_object* v_b_28_, lean_object* v_f_29_, lean_object* v_x_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Option_elim_x27___redArg(v_b_28_, v_f_29_, v_x_30_);
lean_dec(v_b_28_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_elim_x27(lean_object* v_00_u03b1_32_, lean_object* v_00_u03b2_33_, lean_object* v_b_34_, lean_object* v_f_35_, lean_object* v_x_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Option_elim_x27___redArg(v_b_34_, v_f_35_, v_x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_elim_x27___boxed(lean_object* v_00_u03b1_38_, lean_object* v_00_u03b2_39_, lean_object* v_b_40_, lean_object* v_f_41_, lean_object* v_x_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Option_elim_x27(v_00_u03b1_38_, v_00_u03b2_39_, v_b_40_, v_f_41_, v_x_42_);
lean_dec(v_b_40_);
return v_res_43_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
