// Lean compiler output
// Module: Mathlib.Data.Bool.Basic
// Imports: public import Init public meta import Init public import Mathlib.Basic.Logic.Basic public import Mathlib.Order.Defs.LinearOrder
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
lean_object* l_Bool_instDecidableLt___boxed(lean_object*, lean_object*);
lean_object* l_instDecidableEqBool___boxed(lean_object*, lean_object*);
lean_object* l_Bool_instDecidableLe___boxed(lean_object*, lean_object*);
lean_object* l_instOrdBool___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Bool_instMax___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Bool_instMin___lam__0___boxed(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Bool_linearOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Bool_instMin___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool_linearOrder___closed__0 = (const lean_object*)&lp_mathlib_Bool_linearOrder___closed__0_value;
static const lean_closure_object lp_mathlib_Bool_linearOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Bool_instMax___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool_linearOrder___closed__1 = (const lean_object*)&lp_mathlib_Bool_linearOrder___closed__1_value;
static const lean_closure_object lp_mathlib_Bool_linearOrder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdBool___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool_linearOrder___closed__2 = (const lean_object*)&lp_mathlib_Bool_linearOrder___closed__2_value;
static const lean_ctor_object lp_mathlib_Bool_linearOrder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Bool_linearOrder___closed__3 = (const lean_object*)&lp_mathlib_Bool_linearOrder___closed__3_value;
static const lean_closure_object lp_mathlib_Bool_linearOrder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Bool_instDecidableLe___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool_linearOrder___closed__4 = (const lean_object*)&lp_mathlib_Bool_linearOrder___closed__4_value;
static const lean_closure_object lp_mathlib_Bool_linearOrder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Bool_instDecidableLt___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool_linearOrder___closed__5 = (const lean_object*)&lp_mathlib_Bool_linearOrder___closed__5_value;
static lean_once_cell_t lp_mathlib_Bool_linearOrder___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Bool_linearOrder___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Bool_linearOrder;
LEAN_EXPORT uint8_t lp_mathlib_Bool_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Bool_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Bool_xor3(uint8_t, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_xor3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Bool_carry(uint8_t, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_carry___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Bool_linearOrder___closed__6(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___f_12_; lean_object* v___f_13_; lean_object* v___f_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_9_ = ((lean_object*)(lp_mathlib_Bool_linearOrder___closed__5));
v___x_10_ = lean_alloc_closure((void*)(l_instDecidableEqBool___boxed), 2, 0);
v___x_11_ = ((lean_object*)(lp_mathlib_Bool_linearOrder___closed__4));
v___f_12_ = ((lean_object*)(lp_mathlib_Bool_linearOrder___closed__2));
v___f_13_ = ((lean_object*)(lp_mathlib_Bool_linearOrder___closed__1));
v___f_14_ = ((lean_object*)(lp_mathlib_Bool_linearOrder___closed__0));
v___x_15_ = ((lean_object*)(lp_mathlib_Bool_linearOrder___closed__3));
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
static lean_object* _init_lp_mathlib_Bool_linearOrder(void){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_obj_once(&lp_mathlib_Bool_linearOrder___closed__6, &lp_mathlib_Bool_linearOrder___closed__6_once, _init_lp_mathlib_Bool_linearOrder___closed__6);
return v___x_17_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_ofNat(lean_object* v_n_18_){
_start:
{
lean_object* v___x_19_; uint8_t v___x_20_; 
v___x_19_ = lean_unsigned_to_nat(0u);
v___x_20_ = lean_nat_dec_eq(v_n_18_, v___x_19_);
if (v___x_20_ == 0)
{
uint8_t v___x_21_; 
v___x_21_ = 1;
return v___x_21_;
}
else
{
uint8_t v___x_22_; 
v___x_22_ = 0;
return v___x_22_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_ofNat___boxed(lean_object* v_n_23_){
_start:
{
uint8_t v_res_24_; lean_object* v_r_25_; 
v_res_24_ = lp_mathlib_Bool_ofNat(v_n_23_);
lean_dec(v_n_23_);
v_r_25_ = lean_box(v_res_24_);
return v_r_25_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_xor3(uint8_t v_x_26_, uint8_t v_y_27_, uint8_t v_c_28_){
_start:
{
if (v_x_26_ == 0)
{
if (v_y_27_ == 0)
{
return v_c_28_;
}
else
{
goto v___jp_29_;
}
}
else
{
if (v_y_27_ == 0)
{
goto v___jp_29_;
}
else
{
return v_c_28_;
}
}
v___jp_29_:
{
if (v_c_28_ == 0)
{
uint8_t v___x_30_; 
v___x_30_ = 1;
return v___x_30_;
}
else
{
uint8_t v___x_31_; 
v___x_31_ = 0;
return v___x_31_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_xor3___boxed(lean_object* v_x_32_, lean_object* v_y_33_, lean_object* v_c_34_){
_start:
{
uint8_t v_x_boxed_35_; uint8_t v_y_boxed_36_; uint8_t v_c_boxed_37_; uint8_t v_res_38_; lean_object* v_r_39_; 
v_x_boxed_35_ = lean_unbox(v_x_32_);
v_y_boxed_36_ = lean_unbox(v_y_33_);
v_c_boxed_37_ = lean_unbox(v_c_34_);
v_res_38_ = lp_mathlib_Bool_xor3(v_x_boxed_35_, v_y_boxed_36_, v_c_boxed_37_);
v_r_39_ = lean_box(v_res_38_);
return v_r_39_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_carry(uint8_t v_x_40_, uint8_t v_y_41_, uint8_t v_c_42_){
_start:
{
if (v_x_40_ == 0)
{
goto v___jp_43_;
}
else
{
if (v_y_41_ == 0)
{
goto v___jp_43_;
}
else
{
return v_y_41_;
}
}
v___jp_43_:
{
if (v_x_40_ == 0)
{
if (v_y_41_ == 0)
{
return v_y_41_;
}
else
{
return v_c_42_;
}
}
else
{
if (v_c_42_ == 0)
{
if (v_y_41_ == 0)
{
return v_y_41_;
}
else
{
return v_c_42_;
}
}
else
{
return v_c_42_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_carry___boxed(lean_object* v_x_44_, lean_object* v_y_45_, lean_object* v_c_46_){
_start:
{
uint8_t v_x_boxed_47_; uint8_t v_y_boxed_48_; uint8_t v_c_boxed_49_; uint8_t v_res_50_; lean_object* v_r_51_; 
v_x_boxed_47_ = lean_unbox(v_x_44_);
v_y_boxed_48_ = lean_unbox(v_y_45_);
v_c_boxed_49_ = lean_unbox(v_c_46_);
v_res_50_ = lp_mathlib_Bool_carry(v_x_boxed_47_, v_y_boxed_48_, v_c_boxed_49_);
v_r_51_ = lean_box(v_res_50_);
return v_r_51_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Bool_Basic(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Bool_linearOrder = _init_lp_mathlib_Bool_linearOrder();
lean_mark_persistent(lp_mathlib_Bool_linearOrder);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Bool_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Bool_Basic(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Bool_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Bool_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Bool_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
