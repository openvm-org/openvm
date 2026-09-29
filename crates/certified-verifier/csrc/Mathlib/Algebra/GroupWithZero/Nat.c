// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Nat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Algebra.GroupWithZero.Defs public import Mathlib.Tactic.Spread
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
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* l_Nat_mul___boxed(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instSemigroup;
extern lean_object* lp_mathlib_Nat_instCommMonoid;
static lean_once_cell_t lp_mathlib_Nat_instMulZeroClass___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_instMulZeroClass___closed__0;
static lean_once_cell_t lp_mathlib_Nat_instMulZeroClass___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_instMulZeroClass___closed__1;
static const lean_closure_object lp_mathlib_Nat_instMulZeroClass___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instMulZeroClass___closed__2 = (const lean_object*)&lp_mathlib_Nat_instMulZeroClass___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instMulZeroClass;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemigroupWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instMonoidWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommMonoidWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instMulZeroOneClass;
static lean_object* _init_lp_mathlib_Nat_instMulZeroClass___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lp_mathlib_Nat_instAddCancelCommMonoid;
v___x_2_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_Nat_instMulZeroClass___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_obj_once(&lp_mathlib_Nat_instMulZeroClass___closed__0, &lp_mathlib_Nat_instMulZeroClass___closed__0_once, _init_lp_mathlib_Nat_instMulZeroClass___closed__0);
v___x_4_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_3_);
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib_Nat_instMulZeroClass(void){
_start:
{
lean_object* v___x_6_; lean_object* v_toZero_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_6_ = lean_obj_once(&lp_mathlib_Nat_instMulZeroClass___closed__1, &lp_mathlib_Nat_instMulZeroClass___closed__1_once, _init_lp_mathlib_Nat_instMulZeroClass___closed__1);
v_toZero_7_ = lean_ctor_get(v___x_6_, 0);
v___x_8_ = ((lean_object*)(lp_mathlib_Nat_instMulZeroClass___closed__2));
lean_inc(v_toZero_7_);
v___x_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
lean_ctor_set(v___x_9_, 1, v_toZero_7_);
return v___x_9_;
}
}
static lean_object* _init_lp_mathlib_Nat_instSemigroupWithZero(void){
_start:
{
lean_object* v___x_10_; lean_object* v_toZero_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_10_ = lp_mathlib_Nat_instMulZeroClass;
v_toZero_11_ = lean_ctor_get(v___x_10_, 1);
v___x_12_ = lp_mathlib_Nat_instSemigroup;
lean_inc(v_toZero_11_);
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
lean_ctor_set(v___x_13_, 1, v_toZero_11_);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_Nat_instMonoidWithZero(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v_toZero_16_; lean_object* v___x_17_; 
v___x_14_ = lp_mathlib_Nat_instCommMonoid;
v___x_15_ = lp_mathlib_Nat_instMulZeroClass;
v_toZero_16_ = lean_ctor_get(v___x_15_, 1);
lean_inc(v_toZero_16_);
v___x_17_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_17_, 0, v___x_14_);
lean_ctor_set(v___x_17_, 1, v_toZero_16_);
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib_Nat_instCommMonoidWithZero(void){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v_toZero_20_; lean_object* v___x_21_; 
v___x_18_ = lp_mathlib_Nat_instCommMonoid;
v___x_19_ = lp_mathlib_Nat_instMonoidWithZero;
v_toZero_20_ = lean_ctor_get(v___x_19_, 1);
lean_inc(v_toZero_20_);
v___x_21_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_21_, 0, v___x_18_);
lean_ctor_set(v___x_21_, 1, v_toZero_20_);
return v___x_21_;
}
}
static lean_object* _init_lp_mathlib_Nat_instMulZeroOneClass(void){
_start:
{
lean_object* v___x_22_; lean_object* v_toMul_23_; lean_object* v_toZero_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_22_ = lp_mathlib_Nat_instMulZeroClass;
v_toMul_23_ = lean_ctor_get(v___x_22_, 0);
v_toZero_24_ = lean_ctor_get(v___x_22_, 1);
v___x_25_ = lean_unsigned_to_nat(1u);
lean_inc(v_toMul_23_);
v___x_26_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_26_, 0, v___x_25_);
lean_ctor_set(v___x_26_, 1, v_toMul_23_);
lean_inc(v_toZero_24_);
v___x_27_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
lean_ctor_set(v___x_27_, 1, v_toZero_24_);
return v___x_27_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instMulZeroClass = _init_lp_mathlib_Nat_instMulZeroClass();
lean_mark_persistent(lp_mathlib_Nat_instMulZeroClass);
lp_mathlib_Nat_instSemigroupWithZero = _init_lp_mathlib_Nat_instSemigroupWithZero();
lean_mark_persistent(lp_mathlib_Nat_instSemigroupWithZero);
lp_mathlib_Nat_instMonoidWithZero = _init_lp_mathlib_Nat_instMonoidWithZero();
lean_mark_persistent(lp_mathlib_Nat_instMonoidWithZero);
lp_mathlib_Nat_instCommMonoidWithZero = _init_lp_mathlib_Nat_instCommMonoidWithZero();
lean_mark_persistent(lp_mathlib_Nat_instCommMonoidWithZero);
lp_mathlib_Nat_instMulZeroOneClass = _init_lp_mathlib_Nat_instMulZeroOneClass();
lean_mark_persistent(lp_mathlib_Nat_instMulZeroOneClass);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
}
#ifdef __cplusplus
}
#endif
