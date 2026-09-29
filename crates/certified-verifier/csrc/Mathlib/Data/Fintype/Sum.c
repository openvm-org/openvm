// Lean compiler output
// Module: Mathlib.Data.Fintype.Sum
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Sum public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Logic.Embedding.Set
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
lean_object* lp_mathlib_Multiset_disjSum___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_subtypeEq___redArg(lean_object*);
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfFintypeNe___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfFintypeNe___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_fintypeOfFintypeNe___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_fintypeOfFintypeNe___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_fintypeOfFintypeNe___redArg___closed__0 = (const lean_object*)&lp_mathlib_fintypeOfFintypeNe___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_fintypeOfFintypeNe___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_fintypeOfFintypeNe___redArg___closed__0_value),((lean_object*)&lp_mathlib_fintypeOfFintypeNe___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_fintypeOfFintypeNe___redArg___closed__1 = (const lean_object*)&lp_mathlib_fintypeOfFintypeNe___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfFintypeNe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfFintypeNe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSum___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_Multiset_disjSum___redArg(v_inst_1_, v_inst_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeSum(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_inst_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Multiset_disjSum___redArg(v_inst_6_, v_inst_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfFintypeNe___redArg___lam__0(lean_object* v_self_9_){
_start:
{
lean_inc(v_self_9_);
return v_self_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfFintypeNe___redArg___lam__0___boxed(lean_object* v_self_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_fintypeOfFintypeNe___redArg___lam__0(v_self_10_);
lean_dec(v_self_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfFintypeNe___redArg(lean_object* v_a_15_, lean_object* v_x_16_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_17_ = lp_mathlib_Fintype_subtypeEq___redArg(v_a_15_);
v___x_18_ = lp_mathlib_Multiset_disjSum___redArg(v___x_17_, v_x_16_);
v___x_19_ = ((lean_object*)(lp_mathlib_fintypeOfFintypeNe___redArg___closed__1));
v___x_20_ = lp_mathlib_Finset_map___redArg(v___x_19_, v___x_18_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfFintypeNe(lean_object* v_00_u03b1_21_, lean_object* v_a_22_, lean_object* v_x_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_fintypeOfFintypeNe___redArg(v_a_22_, v_x_23_);
return v___x_24_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Embedding_Set(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Sum(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Embedding_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Sum(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Set(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sum(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Sum(builtin);
}
#ifdef __cplusplus
}
#endif
