// Lean compiler output
// Module: Mathlib.Algebra.Ring.Nat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.CharZero.Defs public import Mathlib.Algebra.GroupWithZero.Nat public import Mathlib.Algebra.Ring.Defs public import Mathlib.Data.Nat.Basic
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
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
extern lean_object* lp_mathlib_Nat_instMulZeroOneClass;
extern lean_object* lp_mathlib_Nat_instMonoidWithZero;
extern lean_object* lp_mathlib_Nat_instOne;
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Nat_instAddMonoidWithOne___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_instAddMonoidWithOne___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instAddMonoidWithOne___closed__0 = (const lean_object*)&lp_mathlib_Nat_instAddMonoidWithOne___closed__0_value;
static lean_once_cell_t lp_mathlib_Nat_instAddMonoidWithOne___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_instAddMonoidWithOne___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddCommMonoidWithOne;
static const lean_closure_object lp_mathlib_Nat_instDistrib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instDistrib___closed__0 = (const lean_object*)&lp_mathlib_Nat_instDistrib___closed__0_value;
static const lean_closure_object lp_mathlib_Nat_instDistrib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_add___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instDistrib___closed__1 = (const lean_object*)&lp_mathlib_Nat_instDistrib___closed__1_value;
static const lean_ctor_object lp_mathlib_Nat_instDistrib___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Nat_instDistrib___closed__0_value),((lean_object*)&lp_mathlib_Nat_instDistrib___closed__1_value)}};
static const lean_object* lp_mathlib_Nat_instDistrib___closed__2 = (const lean_object*)&lp_mathlib_Nat_instDistrib___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instDistrib = (const lean_object*)&lp_mathlib_Nat_instDistrib___closed__2_value;
static lean_once_cell_t lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instNonUnitalNonAssocSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instNonUnitalSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instNonAssocSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instCommSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne___lam__0(lean_object* v_n_1_){
_start:
{
lean_inc(v_n_1_);
return v_n_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instAddMonoidWithOne___lam__0___boxed(lean_object* v_n_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Nat_instAddMonoidWithOne___lam__0(v_n_2_);
lean_dec(v_n_2_);
return v_res_3_;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddMonoidWithOne___closed__1(void){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___f_7_; lean_object* v___x_8_; 
v___x_5_ = lp_mathlib_Nat_instOne;
v___x_6_ = lp_mathlib_Nat_instAddCancelCommMonoid;
v___f_7_ = ((lean_object*)(lp_mathlib_Nat_instAddMonoidWithOne___closed__0));
v___x_8_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_8_, 0, v___f_7_);
lean_ctor_set(v___x_8_, 1, v___x_6_);
lean_ctor_set(v___x_8_, 2, v___x_5_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddMonoidWithOne(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_obj_once(&lp_mathlib_Nat_instAddMonoidWithOne___closed__1, &lp_mathlib_Nat_instAddMonoidWithOne___closed__1_once, _init_lp_mathlib_Nat_instAddMonoidWithOne___closed__1);
return v___x_9_;
}
}
static lean_object* _init_lp_mathlib_Nat_instAddCommMonoidWithOne(void){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_Nat_instAddMonoidWithOne;
return v___x_10_;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_17_ = ((lean_object*)(lp_mathlib_Nat_instDistrib___closed__0));
v___x_18_ = lp_mathlib_Nat_instAddCancelCommMonoid;
v___x_19_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_19_, 0, v___x_18_);
lean_ctor_set(v___x_19_, 1, v___x_17_);
return v___x_19_;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonUnitalNonAssocSemiring(void){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_obj_once(&lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0, &lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0_once, _init_lp_mathlib_Nat_instNonUnitalNonAssocSemiring___closed__0);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonUnitalSemiring(void){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Nat_instNonUnitalNonAssocSemiring;
return v___x_21_;
}
}
static lean_object* _init_lp_mathlib_Nat_instNonAssocSemiring(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v_toMulOneClass_24_; lean_object* v_toOne_25_; lean_object* v___f_26_; lean_object* v___x_27_; 
v___x_22_ = lp_mathlib_Nat_instNonUnitalNonAssocSemiring;
v___x_23_ = lp_mathlib_Nat_instMulZeroOneClass;
v_toMulOneClass_24_ = lean_ctor_get(v___x_23_, 0);
v_toOne_25_ = lean_ctor_get(v_toMulOneClass_24_, 0);
v___f_26_ = ((lean_object*)(lp_mathlib_Nat_instAddMonoidWithOne___closed__0));
lean_inc(v_toOne_25_);
v___x_27_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_27_, 0, v___x_22_);
lean_ctor_set(v___x_27_, 1, v_toOne_25_);
lean_ctor_set(v___x_27_, 2, v___f_26_);
return v___x_27_;
}
}
static lean_object* _init_lp_mathlib_Nat_instSemiring(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v_toMonoid_31_; lean_object* v_toAddCommMonoid_32_; lean_object* v_toOne_33_; lean_object* v_toNatCast_34_; lean_object* v_toNPow_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_28_ = lp_mathlib_Nat_instNonUnitalNonAssocSemiring;
v___x_29_ = lp_mathlib_Nat_instNonAssocSemiring;
v___x_30_ = lp_mathlib_Nat_instMonoidWithZero;
v_toMonoid_31_ = lean_ctor_get(v___x_30_, 0);
v_toAddCommMonoid_32_ = lean_ctor_get(v___x_28_, 0);
v_toOne_33_ = lean_ctor_get(v___x_29_, 1);
v_toNatCast_34_ = lean_ctor_get(v___x_29_, 2);
v_toNPow_35_ = lean_ctor_get(v_toMonoid_31_, 2);
v___x_36_ = ((lean_object*)(lp_mathlib_Nat_instDistrib___closed__0));
lean_inc(v_toNPow_35_);
lean_inc(v_toOne_33_);
v___x_37_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_37_, 0, v_toOne_33_);
lean_ctor_set(v___x_37_, 1, v___x_36_);
lean_ctor_set(v___x_37_, 2, v_toNPow_35_);
lean_inc(v_toNatCast_34_);
lean_inc_ref(v_toAddCommMonoid_32_);
v___x_38_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_38_, 0, v_toAddCommMonoid_32_);
lean_ctor_set(v___x_38_, 1, v___x_37_);
lean_ctor_set(v___x_38_, 2, v_toNatCast_34_);
return v___x_38_;
}
}
static lean_object* _init_lp_mathlib_Nat_instCommSemiring(void){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Nat_instSemiring;
return v___x_39_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instAddMonoidWithOne = _init_lp_mathlib_Nat_instAddMonoidWithOne();
lean_mark_persistent(lp_mathlib_Nat_instAddMonoidWithOne);
lp_mathlib_Nat_instAddCommMonoidWithOne = _init_lp_mathlib_Nat_instAddCommMonoidWithOne();
lean_mark_persistent(lp_mathlib_Nat_instAddCommMonoidWithOne);
lp_mathlib_Nat_instNonUnitalNonAssocSemiring = _init_lp_mathlib_Nat_instNonUnitalNonAssocSemiring();
lean_mark_persistent(lp_mathlib_Nat_instNonUnitalNonAssocSemiring);
lp_mathlib_Nat_instNonUnitalSemiring = _init_lp_mathlib_Nat_instNonUnitalSemiring();
lean_mark_persistent(lp_mathlib_Nat_instNonUnitalSemiring);
lp_mathlib_Nat_instNonAssocSemiring = _init_lp_mathlib_Nat_instNonAssocSemiring();
lean_mark_persistent(lp_mathlib_Nat_instNonAssocSemiring);
lp_mathlib_Nat_instSemiring = _init_lp_mathlib_Nat_instSemiring();
lean_mark_persistent(lp_mathlib_Nat_instSemiring);
lp_mathlib_Nat_instCommSemiring = _init_lp_mathlib_Nat_instCommSemiring();
lean_mark_persistent(lp_mathlib_Nat_instCommSemiring);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
}
#ifdef __cplusplus
}
#endif
