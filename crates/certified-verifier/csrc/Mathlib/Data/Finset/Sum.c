// Lean compiler output
// Module: Mathlib.Data.Finset.Sum
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Finset.Fold public import Mathlib.Data.Multiset.Sum
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
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_disjSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjSum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Finset_toLeft___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_toLeft___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_toLeft___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_toLeft___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Finset_toLeft___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_toLeft___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_toLeft___redArg___closed__1 = (const lean_object*)&lp_mathlib_Finset_toLeft___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Finset_toLeft___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_toLeft___redArg___closed__0_value),((lean_object*)&lp_mathlib_Finset_toLeft___redArg___closed__1_value)} };
static const lean_object* lp_mathlib_Finset_toLeft___redArg___closed__2 = (const lean_object*)&lp_mathlib_Finset_toLeft___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_toRight___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_toLeft___redArg___closed__1_value),((lean_object*)&lp_mathlib_Finset_toLeft___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_Finset_toRight___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_toRight___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_toRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_toRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sumEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sumEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Finset_sumEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_sumEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_sumEquiv___closed__0 = (const lean_object*)&lp_mathlib_Finset_sumEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Finset_sumEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_sumEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_sumEquiv___closed__1 = (const lean_object*)&lp_mathlib_Finset_sumEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Finset_sumEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sumEquiv___closed__0_value),((lean_object*)&lp_mathlib_Finset_sumEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_Finset_sumEquiv___closed__2 = (const lean_object*)&lp_mathlib_Finset_sumEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_sumEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjSum___redArg(lean_object* v_s_1_, lean_object* v_t_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_Multiset_disjSum___redArg(v_s_1_, v_t_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjSum(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_s_6_, lean_object* v_t_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Multiset_disjSum___redArg(v_s_6_, v_t_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft___redArg___lam__0(lean_object* v_val_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_10_, 0, v_val_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft___redArg___lam__1(lean_object* v_x_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft___redArg___lam__1___boxed(lean_object* v_x_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Finset_toLeft___redArg___lam__1(v_x_13_);
lean_dec(v_x_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft___redArg(lean_object* v_u_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = ((lean_object*)(lp_mathlib_Finset_toLeft___redArg___closed__2));
v___x_22_ = lp_mathlib_Multiset_filterMap___redArg(v___x_21_, v_u_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_toLeft(lean_object* v_00_u03b1_23_, lean_object* v_00_u03b2_24_, lean_object* v_u_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Finset_toLeft___redArg(v_u_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_toRight___redArg(lean_object* v_u_30_){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_31_ = ((lean_object*)(lp_mathlib_Finset_toRight___redArg___closed__0));
v___x_32_ = lp_mathlib_Multiset_filterMap___redArg(v___x_31_, v_u_30_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_toRight(lean_object* v_00_u03b1_33_, lean_object* v_00_u03b2_34_, lean_object* v_u_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Finset_toRight___redArg(v_u_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sumEquiv___lam__0(lean_object* v_s_37_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
lean_inc(v_s_37_);
v___x_38_ = lp_mathlib_Finset_toLeft___redArg(v_s_37_);
v___x_39_ = lp_mathlib_Finset_toRight___redArg(v_s_37_);
v___x_40_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_40_, 0, v___x_38_);
lean_ctor_set(v___x_40_, 1, v___x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sumEquiv___lam__1(lean_object* v_s_41_){
_start:
{
lean_object* v_fst_42_; lean_object* v_snd_43_; lean_object* v___x_44_; 
v_fst_42_ = lean_ctor_get(v_s_41_, 0);
lean_inc(v_fst_42_);
v_snd_43_ = lean_ctor_get(v_s_41_, 1);
lean_inc(v_snd_43_);
lean_dec_ref(v_s_41_);
v___x_44_ = lp_mathlib_Multiset_disjSum___redArg(v_fst_42_, v_snd_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sumEquiv(lean_object* v_00_u03b1_50_, lean_object* v_00_u03b2_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = ((lean_object*)(lp_mathlib_Finset_sumEquiv___closed__2));
return v___x_52_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Sum(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sum(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Sum(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Sum(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Sum(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Sum(builtin);
}
#ifdef __cplusplus
}
#endif
