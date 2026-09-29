// Lean compiler output
// Module: Mathlib.Algebra.Ring.PUnit
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.PUnit public import Mathlib.Algebra.Ring.Defs
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
extern lean_object* lp_mathlib_PUnit_commGroup;
extern lean_object* lp_mathlib_PUnit_addCommGroup;
lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_PUnit_commRing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PUnit_commRing___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PUnit_commRing___closed__0 = (const lean_object*)&lp_mathlib_PUnit_commRing___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__0(lean_object* v_x_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_commRing___lam__0___boxed(lean_object* v_x_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_PUnit_commRing___lam__0(v_x_3_);
lean_dec(v_x_3_);
return v_res_4_;
}
}
static lean_object* _init_lp_mathlib_PUnit_commRing(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v_toAddMonoid_8_; lean_object* v_toNeg_9_; lean_object* v_toSub_10_; lean_object* v_toZSMul_11_; lean_object* v_toMonoid_12_; lean_object* v___f_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_6_ = lp_mathlib_PUnit_commGroup;
v___x_7_ = lp_mathlib_PUnit_addCommGroup;
v_toAddMonoid_8_ = lean_ctor_get(v___x_7_, 0);
v_toNeg_9_ = lean_ctor_get(v___x_7_, 1);
v_toSub_10_ = lean_ctor_get(v___x_7_, 2);
v_toZSMul_11_ = lean_ctor_get(v___x_7_, 3);
v_toMonoid_12_ = lean_ctor_get(v___x_6_, 0);
v___f_13_ = ((lean_object*)(lp_mathlib_PUnit_commRing___closed__0));
lean_inc_ref(v_toMonoid_12_);
lean_inc_ref(v_toAddMonoid_8_);
v___x_14_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_14_, 0, v_toAddMonoid_8_);
lean_ctor_set(v___x_14_, 1, v_toMonoid_12_);
lean_ctor_set(v___x_14_, 2, v___f_13_);
lean_inc_n(v_toNeg_9_, 2);
v___x_15_ = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(v___x_15_, 0, lean_box(0));
lean_closure_set(v___x_15_, 1, v___f_13_);
lean_closure_set(v___x_15_, 2, v_toNeg_9_);
lean_inc(v_toZSMul_11_);
lean_inc(v_toSub_10_);
v___x_16_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_16_, 0, v___x_14_);
lean_ctor_set(v___x_16_, 1, v_toNeg_9_);
lean_ctor_set(v___x_16_, 2, v_toSub_10_);
lean_ctor_set(v___x_16_, 3, v_toZSMul_11_);
lean_ctor_set(v___x_16_, 4, v___x_15_);
return v___x_16_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_PUnit(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_PUnit(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PUnit_commRing = _init_lp_mathlib_PUnit_commRing();
lean_mark_persistent(lp_mathlib_PUnit_commRing);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_PUnit(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_PUnit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_PUnit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_PUnit(builtin);
}
#ifdef __cplusplus
}
#endif
