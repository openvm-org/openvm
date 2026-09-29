// Lean compiler output
// Module: Mathlib.Algebra.Group.Nat.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Monoid
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
lean_object* l_Nat_mul___boxed(lean_object*, lean_object*);
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
static const lean_closure_object lp_mathlib_Nat_instMulOneClass___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instMulOneClass___closed__0 = (const lean_object*)&lp_mathlib_Nat_instMulOneClass___closed__0_value;
static const lean_ctor_object lp_mathlib_Nat_instMulOneClass___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_instMulOneClass___closed__0_value)}};
static const lean_object* lp_mathlib_Nat_instMulOneClass___closed__1 = (const lean_object*)&lp_mathlib_Nat_instMulOneClass___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instMulOneClass = (const lean_object*)&lp_mathlib_Nat_instMulOneClass___closed__1_value;
static const lean_closure_object lp_mathlib_Nat_instAddCancelCommMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instAddCancelCommMonoid___closed__0 = (const lean_object*)&lp_mathlib_Nat_instAddCancelCommMonoid___closed__0_value;
static const lean_closure_object lp_mathlib_Nat_instAddCancelCommMonoid___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_add___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instAddCancelCommMonoid___closed__1 = (const lean_object*)&lp_mathlib_Nat_instAddCancelCommMonoid___closed__1_value;
static const lean_ctor_object lp_mathlib_Nat_instAddCancelCommMonoid___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_instAddCancelCommMonoid___closed__1_value),((lean_object*)&lp_mathlib_Nat_instAddCancelCommMonoid___closed__0_value)}};
static const lean_object* lp_mathlib_Nat_instAddCancelCommMonoid___closed__2 = (const lean_object*)&lp_mathlib_Nat_instAddCancelCommMonoid___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instAddCancelCommMonoid = (const lean_object*)&lp_mathlib_Nat_instAddCancelCommMonoid___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_instCommMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_instCommMonoid___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instCommMonoid___closed__0 = (const lean_object*)&lp_mathlib_Nat_instCommMonoid___closed__0_value;
static const lean_ctor_object lp_mathlib_Nat_instCommMonoid___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib_Nat_instMulOneClass___closed__0_value),((lean_object*)&lp_mathlib_Nat_instCommMonoid___closed__0_value)}};
static const lean_object* lp_mathlib_Nat_instCommMonoid___closed__1 = (const lean_object*)&lp_mathlib_Nat_instCommMonoid___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instCommMonoid = (const lean_object*)&lp_mathlib_Nat_instCommMonoid___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instAddCommMonoid = (const lean_object*)&lp_mathlib_Nat_instAddCancelCommMonoid___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instAddMonoid = (const lean_object*)&lp_mathlib_Nat_instAddCancelCommMonoid___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instMonoid = (const lean_object*)&lp_mathlib_Nat_instCommMonoid___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddCommSemigroup;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddSemigroup;
static lean_once_cell_t lp_mathlib_Nat_instOne___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_instOne___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instOne;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid___lam__0(lean_object* v_m_13_, lean_object* v_n_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_nat_pow(v_n_14_, v_m_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoid___lam__0___boxed(lean_object* v_m_16_, lean_object* v_n_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Nat_instCommMonoid___lam__0(v_m_16_, v_n_17_);
lean_dec(v_n_17_);
lean_dec(v_m_16_);
return v_res_18_;
}
}
static lean_object* _init_lp_mathlib_Nat_instCommSemigroup(void){
_start:
{
lean_object* v___x_28_; lean_object* v_toMul_29_; 
v___x_28_ = ((lean_object*)(lp_mathlib_Nat_instCommMonoid));
v_toMul_29_ = lean_ctor_get(v___x_28_, 1);
lean_inc(v_toMul_29_);
return v_toMul_29_;
}
}
static lean_object* _init_lp_mathlib_Nat_instSemigroup(void){
_start:
{
lean_object* v___x_30_; lean_object* v_toMul_31_; 
v___x_30_ = ((lean_object*)(lp_mathlib_Nat_instCommMonoid));
v_toMul_31_ = lean_ctor_get(v___x_30_, 1);
lean_inc(v_toMul_31_);
return v_toMul_31_;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCommSemigroup(void){
_start:
{
lean_object* v___x_32_; lean_object* v_toAdd_33_; 
v___x_32_ = ((lean_object*)(lp_mathlib_Nat_instAddCancelCommMonoid));
v_toAdd_33_ = lean_ctor_get(v___x_32_, 1);
lean_inc(v_toAdd_33_);
return v_toAdd_33_;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddSemigroup(void){
_start:
{
lean_object* v___x_34_; lean_object* v_toAdd_35_; 
v___x_34_ = ((lean_object*)(lp_mathlib_Nat_instAddCancelCommMonoid));
v_toAdd_35_ = lean_ctor_get(v___x_34_, 1);
lean_inc(v_toAdd_35_);
return v_toAdd_35_;
}
}
static lean_object* _init_lp_mathlib_Nat_instOne___closed__0(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_mathlib_Nat_instMulOneClass));
v___x_37_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_36_);
return v___x_37_;
}
}
static lean_object* _init_lp_mathlib_Nat_instOne(void){
_start:
{
lean_object* v___x_38_; lean_object* v_toOne_39_; 
v___x_38_ = lean_obj_once(&lp_mathlib_Nat_instOne___closed__0, &lp_mathlib_Nat_instOne___closed__0_once, _init_lp_mathlib_Nat_instOne___closed__0);
v_toOne_39_ = lean_ctor_get(v___x_38_, 0);
lean_inc(v_toOne_39_);
return v_toOne_39_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instCommSemigroup = _init_lp_mathlib_Nat_instCommSemigroup();
lean_mark_persistent(lp_mathlib_Nat_instCommSemigroup);
lp_mathlib_Nat_instSemigroup = _init_lp_mathlib_Nat_instSemigroup();
lean_mark_persistent(lp_mathlib_Nat_instSemigroup);
lp_mathlib_Nat_instAddCommSemigroup = _init_lp_mathlib_Nat_instAddCommSemigroup();
lean_mark_persistent(lp_mathlib_Nat_instAddCommSemigroup);
lp_mathlib_Nat_instAddSemigroup = _init_lp_mathlib_Nat_instAddSemigroup();
lean_mark_persistent(lp_mathlib_Nat_instAddSemigroup);
lp_mathlib_Nat_instOne = _init_lp_mathlib_Nat_instOne();
lean_mark_persistent(lp_mathlib_Nat_instOne);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
