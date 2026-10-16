// Lean compiler output
// Module: Mathlib.Data.Int.Order.Basic
// Imports: public import Init public meta import Init public import Mathlib.Basic.Logic.Basic public import Mathlib.Data.Int.Notation public import Mathlib.Data.Nat.Notation public import Mathlib.Order.Defs.LinearOrder
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
lean_object* l_Int_instMin___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Int_decLe___boxed(lean_object*, lean_object*);
lean_object* l_Int_decLt___boxed(lean_object*, lean_object*);
lean_object* l_Int_instDecidableEq___boxed(lean_object*, lean_object*);
lean_object* l_instOrdInt___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Int_instMax___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Int_instLinearOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_instMin___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instLinearOrder___closed__0 = (const lean_object*)&lp_mathlib_Int_instLinearOrder___closed__0_value;
static const lean_closure_object lp_mathlib_Int_instLinearOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_instMax___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instLinearOrder___closed__1 = (const lean_object*)&lp_mathlib_Int_instLinearOrder___closed__1_value;
static const lean_closure_object lp_mathlib_Int_instLinearOrder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdInt___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instLinearOrder___closed__2 = (const lean_object*)&lp_mathlib_Int_instLinearOrder___closed__2_value;
static const lean_ctor_object lp_mathlib_Int_instLinearOrder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Int_instLinearOrder___closed__3 = (const lean_object*)&lp_mathlib_Int_instLinearOrder___closed__3_value;
static const lean_closure_object lp_mathlib_Int_instLinearOrder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_decLe___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instLinearOrder___closed__4 = (const lean_object*)&lp_mathlib_Int_instLinearOrder___closed__4_value;
static const lean_closure_object lp_mathlib_Int_instLinearOrder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_decLt___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instLinearOrder___closed__5 = (const lean_object*)&lp_mathlib_Int_instLinearOrder___closed__5_value;
static lean_once_cell_t lp_mathlib_Int_instLinearOrder___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_instLinearOrder___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Int_instLinearOrder;
LEAN_EXPORT lean_object* lp_mathlib_Int_instPreorder;
LEAN_EXPORT lean_object* lp_mathlib_Int_instPartialOrder;
static lean_object* _init_lp_mathlib_Int_instLinearOrder___closed__6(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___f_12_; lean_object* v___f_13_; lean_object* v___f_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_9_ = ((lean_object*)(lp_mathlib_Int_instLinearOrder___closed__5));
v___x_10_ = lean_alloc_closure((void*)(l_Int_instDecidableEq___boxed), 2, 0);
v___x_11_ = ((lean_object*)(lp_mathlib_Int_instLinearOrder___closed__4));
v___f_12_ = ((lean_object*)(lp_mathlib_Int_instLinearOrder___closed__2));
v___f_13_ = ((lean_object*)(lp_mathlib_Int_instLinearOrder___closed__1));
v___f_14_ = ((lean_object*)(lp_mathlib_Int_instLinearOrder___closed__0));
v___x_15_ = ((lean_object*)(lp_mathlib_Int_instLinearOrder___closed__3));
v___x_16_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_16_, 0, v___x_15_);
lean_ctor_set(v___x_16_, 1, v___f_14_);
lean_ctor_set(v___x_16_, 2, v___f_13_);
lean_ctor_set(v___x_16_, 3, v___f_12_);
lean_ctor_set(v___x_16_, 4, v___x_11_);
lean_ctor_set(v___x_16_, 5, v___x_10_);
lean_ctor_set(v___x_16_, 6, v___x_9_);
return v___x_16_;
}
}
static lean_object* _init_lp_mathlib_Int_instLinearOrder(void){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_obj_once(&lp_mathlib_Int_instLinearOrder___closed__6, &lp_mathlib_Int_instLinearOrder___closed__6_once, _init_lp_mathlib_Int_instLinearOrder___closed__6);
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib_Int_instPreorder(void){
_start:
{
lean_object* v___x_18_; lean_object* v_toPartialOrder_19_; 
v___x_18_ = lp_mathlib_Int_instLinearOrder;
v_toPartialOrder_19_ = lean_ctor_get(v___x_18_, 0);
lean_inc_ref(v_toPartialOrder_19_);
return v_toPartialOrder_19_;
}
}
static lean_object* _init_lp_mathlib_Int_instPartialOrder(void){
_start:
{
lean_object* v___x_20_; lean_object* v_toPartialOrder_21_; 
v___x_20_ = lp_mathlib_Int_instLinearOrder;
v_toPartialOrder_21_ = lean_ctor_get(v___x_20_, 0);
lean_inc_ref(v_toPartialOrder_21_);
return v_toPartialOrder_21_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instLinearOrder = _init_lp_mathlib_Int_instLinearOrder();
lean_mark_persistent(lp_mathlib_Int_instLinearOrder);
lp_mathlib_Int_instPreorder = _init_lp_mathlib_Int_instPreorder();
lean_mark_persistent(lp_mathlib_Int_instPreorder);
lp_mathlib_Int_instPartialOrder = _init_lp_mathlib_Int_instPartialOrder();
lean_mark_persistent(lp_mathlib_Int_instPartialOrder);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_Order_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Int_Order_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
