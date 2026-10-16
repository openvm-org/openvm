// Lean compiler output
// Module: Mathlib.Data.Finset.Fin
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Fin.Embedding
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
lean_object* lp_mathlib_Multiset_pmap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_attachFin___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_attachFin___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_attachFin___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_attachFin___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin___redArg___lam__0(lean_object* v_a_1_, lean_object* v_ha_2_){
_start:
{
lean_inc(v_a_1_);
return v_a_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin___redArg___lam__0___boxed(lean_object* v_a_3_, lean_object* v_ha_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_Finset_attachFin___redArg___lam__0(v_a_3_, v_ha_4_);
lean_dec(v_a_3_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin___redArg(lean_object* v_s_7_){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; 
v___f_8_ = ((lean_object*)(lp_mathlib_Finset_attachFin___redArg___closed__0));
v___x_9_ = lp_mathlib_Multiset_pmap___redArg(v___f_8_, v_s_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin(lean_object* v_s_10_, lean_object* v_n_11_, lean_object* v_h_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Finset_attachFin___redArg(v_s_10_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_attachFin___boxed(lean_object* v_s_14_, lean_object* v_n_15_, lean_object* v_h_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Finset_attachFin(v_s_14_, v_n_15_, v_h_16_);
lean_dec(v_n_15_);
return v_res_17_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Embedding(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Fin(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Fin(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Embedding(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Fin(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Fin(builtin);
}
#ifdef __cplusplus
}
#endif
