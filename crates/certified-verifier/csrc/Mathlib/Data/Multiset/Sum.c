// Lean compiler output
// Module: Mathlib.Data.Multiset.Sum
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Multiset
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
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_disjSum___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_disjSum___redArg___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Multiset_disjSum___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_disjSum___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_disjSum___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_disjSum___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Multiset_disjSum___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_disjSum___redArg___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_disjSum___redArg___closed__1 = (const lean_object*)&lp_mathlib_Multiset_disjSum___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_disjSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_disjSum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_disjSum___redArg___lam__0(lean_object* v_val_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2_, 0, v_val_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_disjSum___redArg___lam__1(lean_object* v_val_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4_, 0, v_val_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_disjSum___redArg(lean_object* v_s_7_, lean_object* v_t_8_){
_start:
{
lean_object* v___f_9_; lean_object* v___f_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___f_9_ = ((lean_object*)(lp_mathlib_Multiset_disjSum___redArg___closed__0));
v___f_10_ = ((lean_object*)(lp_mathlib_Multiset_disjSum___redArg___closed__1));
v___x_11_ = lp_mathlib_Multiset_map___redArg(v___f_9_, v_s_7_);
v___x_12_ = lp_mathlib_Multiset_map___redArg(v___f_10_, v_t_8_);
v___x_13_ = l_List_appendTR___redArg(v___x_11_, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_disjSum(lean_object* v_00_u03b1_14_, lean_object* v_00_u03b2_15_, lean_object* v_s_16_, lean_object* v_t_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Multiset_disjSum___redArg(v_s_16_, v_t_17_);
return v___x_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Sum(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Sum(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Sum(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Sum(builtin);
}
#ifdef __cplusplus
}
#endif
