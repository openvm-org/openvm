// Lean compiler output
// Module: Mathlib.Data.Multiset.Sections
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.Bind
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
lean_object* lp_mathlib_Multiset_cons(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_bind___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_rec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_Sections___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_Sections___redArg___lam__1___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_Sections___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_Sections___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Multiset_Sections___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Multiset_Sections___redArg___closed__1 = (const lean_object*)&lp_mathlib_Multiset_Sections___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections___redArg___lam__0(lean_object* v_c_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_cons), 3, 2);
lean_closure_set(v___x_3_, 0, lean_box(0));
lean_closure_set(v___x_3_, 1, v_a_2_);
v___x_4_ = lp_mathlib_Multiset_map___redArg(v___x_3_, v_c_1_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections___redArg___lam__1(lean_object* v_s_5_, lean_object* v_x_6_, lean_object* v_c_7_){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; 
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_Sections___redArg___lam__0), 2, 1);
lean_closure_set(v___f_8_, 0, v_c_7_);
v___x_9_ = lp_mathlib_Multiset_bind___redArg(v_s_5_, v___f_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections___redArg___lam__1___boxed(lean_object* v_s_10_, lean_object* v_x_11_, lean_object* v_c_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Multiset_Sections___redArg___lam__1(v_s_10_, v_x_11_, v_c_12_);
lean_dec(v_x_11_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections___redArg(lean_object* v_s_17_){
_start:
{
lean_object* v___f_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___f_18_ = ((lean_object*)(lp_mathlib_Multiset_Sections___redArg___closed__0));
v___x_19_ = ((lean_object*)(lp_mathlib_Multiset_Sections___redArg___closed__1));
v___x_20_ = lp_mathlib_Multiset_rec___redArg(v___x_19_, v___f_18_, v_s_17_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Sections(lean_object* v_00_u03b1_21_, lean_object* v_s_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_Multiset_Sections___redArg(v_s_22_);
return v___x_23_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Sections(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Sections(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Sections(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Sections(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Sections(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Sections(builtin);
}
#ifdef __cplusplus
}
#endif
