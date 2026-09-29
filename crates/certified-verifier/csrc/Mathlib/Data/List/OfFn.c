// Lean compiler output
// Module: Mathlib.Data.List.OfFn
// Imports: public import Init public meta import Init public import Mathlib.Data.Fin.Tuple.Basic
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
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_get___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_ofFn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_equivSigmaTuple___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_equivSigmaTuple___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_List_equivSigmaTuple___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_equivSigmaTuple___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_equivSigmaTuple___closed__0 = (const lean_object*)&lp_mathlib_List_equivSigmaTuple___closed__0_value;
static const lean_closure_object lp_mathlib_List_equivSigmaTuple___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_equivSigmaTuple___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_equivSigmaTuple___closed__1 = (const lean_object*)&lp_mathlib_List_equivSigmaTuple___closed__1_value;
static const lean_ctor_object lp_mathlib_List_equivSigmaTuple___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_List_equivSigmaTuple___closed__0_value),((lean_object*)&lp_mathlib_List_equivSigmaTuple___closed__1_value)}};
static const lean_object* lp_mathlib_List_equivSigmaTuple___closed__2 = (const lean_object*)&lp_mathlib_List_equivSigmaTuple___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_List_equivSigmaTuple(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_ofFnRec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_ofFnRec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_equivSigmaTuple___lam__0(lean_object* v_l_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_2_ = l_List_lengthTR___redArg(v_l_1_);
v___x_3_ = lean_alloc_closure((void*)(l_List_get___boxed), 3, 2);
lean_closure_set(v___x_3_, 0, lean_box(0));
lean_closure_set(v___x_3_, 1, v_l_1_);
v___x_4_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4_, 0, v___x_2_);
lean_ctor_set(v___x_4_, 1, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_equivSigmaTuple___lam__1(lean_object* v_f_5_){
_start:
{
lean_object* v_fst_6_; lean_object* v_snd_7_; lean_object* v___x_8_; 
v_fst_6_ = lean_ctor_get(v_f_5_, 0);
lean_inc(v_fst_6_);
v_snd_7_ = lean_ctor_get(v_f_5_, 1);
lean_inc(v_snd_7_);
lean_dec_ref(v_f_5_);
v___x_8_ = l_List_ofFn___redArg(v_fst_6_, v_snd_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_equivSigmaTuple(lean_object* v_00_u03b1_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = ((lean_object*)(lp_mathlib_List_equivSigmaTuple___closed__2));
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_ofFnRec___redArg(lean_object* v_h_16_, lean_object* v_l_17_){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_18_ = l_List_lengthTR___redArg(v_l_17_);
v___x_19_ = lean_alloc_closure((void*)(l_List_get___boxed), 3, 2);
lean_closure_set(v___x_19_, 0, lean_box(0));
lean_closure_set(v___x_19_, 1, v_l_17_);
v___x_20_ = lean_apply_2(v_h_16_, v___x_18_, v___x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_ofFnRec(lean_object* v_00_u03b1_21_, lean_object* v_C_22_, lean_object* v_h_23_, lean_object* v_l_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_List_ofFnRec___redArg(v_h_23_, v_l_24_);
return v___x_25_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_OfFn(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_OfFn(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_OfFn(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_OfFn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_OfFn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_OfFn(builtin);
}
#ifdef __cplusplus
}
#endif
