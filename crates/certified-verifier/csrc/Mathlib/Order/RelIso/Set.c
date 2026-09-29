// Lean compiler output
// Module: Mathlib.Order.RelIso.Set
// Imports: public import Init public meta import Init public import Mathlib.Order.Directed public import Mathlib.Order.RelIso.Basic public import Mathlib.Logic.Embedding.Set public import Mathlib.Logic.Equiv.Set
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
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeUnivEquiv(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_codRestrict___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subrel_relEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subrel_relEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Subrel_relEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subrel_relEmbedding(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subrel_inclusionEmbedding___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subrel_inclusionEmbedding___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Subrel_inclusionEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subrel_inclusionEmbedding___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subrel_inclusionEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Subrel_inclusionEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subrel_inclusionEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RelIso_subrelUnivIso___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RelIso_subrelUnivIso___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_subrelUnivIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subrel_relEmbedding(lean_object* v_00_u03b1_2_, lean_object* v_r_3_, lean_object* v_p_4_){
_start:
{
lean_object* v___f_5_; 
v___f_5_ = ((lean_object*)(lp_mathlib_Subrel_relEmbedding___closed__0));
return v___f_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subrel_inclusionEmbedding___lam__0(lean_object* v___y_6_){
_start:
{
lean_inc(v___y_6_);
return v___y_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subrel_inclusionEmbedding___lam__0___boxed(lean_object* v___y_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Subrel_inclusionEmbedding___lam__0(v___y_7_);
lean_dec(v___y_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subrel_inclusionEmbedding(lean_object* v_00_u03b1_10_, lean_object* v_r_11_, lean_object* v_s_12_, lean_object* v_t_13_, lean_object* v_h_14_){
_start:
{
lean_object* v___f_15_; 
v___f_15_ = ((lean_object*)(lp_mathlib_Subrel_inclusionEmbedding___closed__0));
return v___f_15_;
}
}
static lean_object* _init_lp_mathlib_RelIso_subrelUnivIso___closed__0(void){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Equiv_subtypeUnivEquiv(lean_box(0), lean_box(0), lean_box(0));
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_subrelUnivIso(lean_object* v_00_u03b1_17_, lean_object* v_r_18_, lean_object* v_p_19_, lean_object* v_h_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_obj_once(&lp_mathlib_RelIso_subrelUnivIso___closed__0, &lp_mathlib_RelIso_subrelUnivIso___closed__0_once, _init_lp_mathlib_RelIso_subrelUnivIso___closed__0);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_codRestrict___redArg(lean_object* v_f_22_){
_start:
{
lean_object* v___f_23_; 
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_23_, 0, v_f_22_);
return v___f_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_codRestrict(lean_object* v_00_u03b1_24_, lean_object* v_00_u03b2_25_, lean_object* v_r_26_, lean_object* v_s_27_, lean_object* v_p_28_, lean_object* v_f_29_, lean_object* v_H_30_){
_start:
{
lean_object* v___f_31_; 
v___f_31_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_31_, 0, v_f_29_);
return v___f_31_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Embedding_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_RelIso_Set(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Embedding_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_RelIso_Set(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_RelIso_Set(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelIso_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_RelIso_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_RelIso_Set(builtin);
}
#ifdef __cplusplus
}
#endif
