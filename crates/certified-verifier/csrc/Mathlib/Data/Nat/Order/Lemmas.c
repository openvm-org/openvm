// Lean compiler output
// Module: Mathlib.Data.Nat.Order.Lemmas
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Find public import Mathlib.Data.Set.Basic public import Mathlib.Tactic.ByContra
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
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
extern lean_object* lp_mathlib_Nat_instLinearOrder;
lean_object* lp_mathlib_Subtype_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_orderBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_semilatticeSup___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_semilatticeSup___lam__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Nat_Subtype_semilatticeSup___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_Subtype_semilatticeSup___closed__0;
static const lean_closure_object lp_mathlib_Nat_Subtype_semilatticeSup___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_Subtype_semilatticeSup___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_Subtype_semilatticeSup___closed__1 = (const lean_object*)&lp_mathlib_Nat_Subtype_semilatticeSup___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_semilatticeSup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_orderBot___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lp_mathlib_Nat_findX___redArg(v_inst_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_orderBot(lean_object* v_s_3_, lean_object* v_inst_4_, lean_object* v_h_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_mathlib_Nat_findX___redArg(v_inst_4_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_semilatticeSup___lam__0(lean_object* v_a_7_, lean_object* v_a_8_){
_start:
{
uint8_t v___x_9_; 
v___x_9_ = lean_nat_dec_le(v_a_7_, v_a_8_);
if (v___x_9_ == 0)
{
lean_inc(v_a_7_);
return v_a_7_;
}
else
{
lean_inc(v_a_8_);
return v_a_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_semilatticeSup___lam__0___boxed(lean_object* v_a_10_, lean_object* v_a_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Nat_Subtype_semilatticeSup___lam__0(v_a_10_, v_a_11_);
lean_dec(v_a_11_);
lean_dec(v_a_10_);
return v_res_12_;
}
}
static lean_object* _init_lp_mathlib_Nat_Subtype_semilatticeSup___closed__0(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lp_mathlib_Nat_instLinearOrder;
v___x_14_ = lp_mathlib_Subtype_instLinearOrder___redArg(v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Subtype_semilatticeSup(lean_object* v_p_16_){
_start:
{
lean_object* v___x_17_; lean_object* v_toPartialOrder_18_; lean_object* v___f_19_; lean_object* v___x_20_; 
v___x_17_ = lean_obj_once(&lp_mathlib_Nat_Subtype_semilatticeSup___closed__0, &lp_mathlib_Nat_Subtype_semilatticeSup___closed__0_once, _init_lp_mathlib_Nat_Subtype_semilatticeSup___closed__0);
v_toPartialOrder_18_ = lean_ctor_get(v___x_17_, 0);
v___f_19_ = ((lean_object*)(lp_mathlib_Nat_Subtype_semilatticeSup___closed__1));
lean_inc_ref(v_toPartialOrder_18_);
v___x_20_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_20_, 0, v_toPartialOrder_18_);
lean_ctor_set(v___x_20_, 1, v___f_19_);
return v___x_20_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Order_Lemmas(builtin);
}
#ifdef __cplusplus
}
#endif
