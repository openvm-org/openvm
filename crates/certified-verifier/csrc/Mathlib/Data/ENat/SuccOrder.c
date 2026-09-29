// Lean compiler output
// Module: Mathlib.Data.ENat.SuccOrder
// Imports: public import Init public meta import Init public import Mathlib.Data.ENat.Monoid public import Mathlib.Data.Nat.SuccPred
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
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_instSuccOrder___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderENat___aux__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_succ___at___00instSuccOrderENat_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Order_succ___at___00instSuccOrderENat_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderENat___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_instSuccOrderENat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSuccOrderENat___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSuccOrderENat___closed__0 = (const lean_object*)&lp_mathlib_instSuccOrderENat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instSuccOrderENat = (const lean_object*)&lp_mathlib_instSuccOrderENat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_ENat_instSuccAddOrder = (const lean_object*)&lp_mathlib_instSuccOrderENat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderENat___aux__1(lean_object* v_x_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
if (lean_obj_tag(v_x_1_) == 0)
{
return v___x_2_;
}
else
{
lean_object* v_val_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_12_; 
v_val_3_ = lean_ctor_get(v_x_1_, 0);
v_isSharedCheck_12_ = !lean_is_exclusive(v_x_1_);
if (v_isSharedCheck_12_ == 0)
{
v___x_5_ = v_x_1_;
v_isShared_6_ = v_isSharedCheck_12_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_val_3_);
lean_dec(v_x_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_12_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_7_; uint8_t v___x_8_; 
v___x_7_ = lp_mathlib_Nat_instSuccOrder___lam__0(v_val_3_);
v___x_8_ = lean_nat_dec_eq(v___x_7_, v_val_3_);
lean_dec(v_val_3_);
if (v___x_8_ == 0)
{
lean_object* v___x_10_; 
if (v_isShared_6_ == 0)
{
lean_ctor_set(v___x_5_, 0, v___x_7_);
v___x_10_ = v___x_5_;
goto v_reusejp_9_;
}
else
{
lean_object* v_reuseFailAlloc_11_; 
v_reuseFailAlloc_11_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_11_, 0, v___x_7_);
v___x_10_ = v_reuseFailAlloc_11_;
goto v_reusejp_9_;
}
v_reusejp_9_:
{
return v___x_10_;
}
}
else
{
lean_dec(v___x_7_);
lean_del_object(v___x_5_);
return v___x_2_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_succ___at___00instSuccOrderENat_spec__0(lean_object* v_a_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_unsigned_to_nat(1u);
v___x_15_ = lean_nat_add(v_a_13_, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Order_succ___at___00instSuccOrderENat_spec__0___boxed(lean_object* v_a_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Order_succ___at___00instSuccOrderENat_spec__0(v_a_16_);
lean_dec(v_a_16_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSuccOrderENat___lam__0(lean_object* v___y_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_box(0);
if (lean_obj_tag(v___y_18_) == 0)
{
return v___x_19_;
}
else
{
lean_object* v_val_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_29_; 
v_val_20_ = lean_ctor_get(v___y_18_, 0);
v_isSharedCheck_29_ = !lean_is_exclusive(v___y_18_);
if (v_isSharedCheck_29_ == 0)
{
v___x_22_ = v___y_18_;
v_isShared_23_ = v_isSharedCheck_29_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_val_20_);
lean_dec(v___y_18_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_29_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v___x_24_; uint8_t v___x_25_; 
v___x_24_ = lp_mathlib_Order_succ___at___00instSuccOrderENat_spec__0(v_val_20_);
v___x_25_ = lean_nat_dec_eq(v___x_24_, v_val_20_);
lean_dec(v_val_20_);
if (v___x_25_ == 0)
{
lean_object* v___x_27_; 
if (v_isShared_23_ == 0)
{
lean_ctor_set(v___x_22_, 0, v___x_24_);
v___x_27_ = v___x_22_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_28_; 
v_reuseFailAlloc_28_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_28_, 0, v___x_24_);
v___x_27_ = v_reuseFailAlloc_28_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
return v___x_27_;
}
}
else
{
lean_dec(v___x_24_);
lean_del_object(v___x_22_);
return v___x_19_;
}
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_Monoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_SuccOrder(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ENat_SuccOrder(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_ENat_Monoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ENat_SuccOrder(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_SuccOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ENat_SuccOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ENat_SuccOrder(builtin);
}
#ifdef __cplusplus
}
#endif
