// Lean compiler output
// Module: Mathlib.Algebra.Ring.Rat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Units.Basic public import Mathlib.Algebra.Ring.Basic public import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Data.Rat.Defs public import Mathlib.Algebra.Group.Nat.Defs
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
lean_object* l_Rat_zpow(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Rat_commMonoid;
extern lean_object* lp_mathlib_Rat_addCommGroup;
lean_object* l_Rat_ofInt(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Rat_neg(lean_object*);
lean_object* l_Rat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_addCommGroup___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* l_Rat_inv(lean_object*);
lean_object* l_Rat_div___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Rat_commRing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_ofInt, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commRing___closed__0 = (const lean_object*)&lp_mathlib_Rat_commRing___closed__0_value;
static const lean_closure_object lp_mathlib_Rat_commRing___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_commRing___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commRing___closed__1 = (const lean_object*)&lp_mathlib_Rat_commRing___closed__1_value;
static const lean_closure_object lp_mathlib_Rat_commRing___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_neg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commRing___closed__2 = (const lean_object*)&lp_mathlib_Rat_commRing___closed__2_value;
static const lean_closure_object lp_mathlib_Rat_commRing___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_sub, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commRing___closed__3 = (const lean_object*)&lp_mathlib_Rat_commRing___closed__3_value;
static const lean_closure_object lp_mathlib_Rat_commRing___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_addCommGroup___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commRing___closed__4 = (const lean_object*)&lp_mathlib_Rat_commRing___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Rat_commGroupWithZero___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_commGroupWithZero___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commGroupWithZero___closed__0 = (const lean_object*)&lp_mathlib_Rat_commGroupWithZero___closed__0_value;
static const lean_closure_object lp_mathlib_Rat_commGroupWithZero___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_inv, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commGroupWithZero___closed__1 = (const lean_object*)&lp_mathlib_Rat_commGroupWithZero___closed__1_value;
static const lean_closure_object lp_mathlib_Rat_commGroupWithZero___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_div___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_commGroupWithZero___closed__2 = (const lean_object*)&lp_mathlib_Rat_commGroupWithZero___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Rat_semiring;
LEAN_EXPORT lean_object* lp_mathlib_Rat_commRing___lam__0(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_nat_to_int(v_n_1_);
v___x_3_ = l_Rat_ofInt(v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_Rat_commRing(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v_toAddMonoid_11_; lean_object* v___f_12_; lean_object* v___f_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___f_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_9_ = lp_mathlib_Rat_addCommGroup;
v___x_10_ = lp_mathlib_Rat_commMonoid;
v_toAddMonoid_11_ = lean_ctor_get(v___x_9_, 0);
v___f_12_ = ((lean_object*)(lp_mathlib_Rat_commRing___closed__0));
v___f_13_ = ((lean_object*)(lp_mathlib_Rat_commRing___closed__1));
v___x_14_ = ((lean_object*)(lp_mathlib_Rat_commRing___closed__2));
v___x_15_ = ((lean_object*)(lp_mathlib_Rat_commRing___closed__3));
v___f_16_ = ((lean_object*)(lp_mathlib_Rat_commRing___closed__4));
lean_inc_ref(v_toAddMonoid_11_);
v___x_17_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_17_, 0, v_toAddMonoid_11_);
lean_ctor_set(v___x_17_, 1, v___x_10_);
lean_ctor_set(v___x_17_, 2, v___f_13_);
v___x_18_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_18_, 0, v___x_17_);
lean_ctor_set(v___x_18_, 1, v___x_14_);
lean_ctor_set(v___x_18_, 2, v___x_15_);
lean_ctor_set(v___x_18_, 3, v___f_16_);
lean_ctor_set(v___x_18_, 4, v___f_12_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero___lam__0(lean_object* v_z_19_, lean_object* v_q_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = l_Rat_zpow(v_q_20_, v_z_19_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_commGroupWithZero___lam__0___boxed(lean_object* v_z_22_, lean_object* v_q_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Rat_commGroupWithZero___lam__0(v_z_22_, v_q_23_);
lean_dec(v_z_22_);
return v_res_24_;
}
}
static lean_object* _init_lp_mathlib_Rat_commGroupWithZero(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v_toSemiring_30_; lean_object* v___x_31_; lean_object* v_toZero_32_; lean_object* v___x_34_; uint8_t v_isShared_35_; uint8_t v_isSharedCheck_43_; 
v___x_28_ = lp_mathlib_Rat_commMonoid;
v___x_29_ = lp_mathlib_Rat_commRing;
v_toSemiring_30_ = lean_ctor_get(v___x_29_, 0);
lean_inc_ref(v_toSemiring_30_);
v___x_31_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_30_);
v_toZero_32_ = lean_ctor_get(v___x_31_, 1);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_31_);
if (v_isSharedCheck_43_ == 0)
{
lean_object* v_unused_44_; 
v_unused_44_ = lean_ctor_get(v___x_31_, 0);
lean_dec(v_unused_44_);
v___x_34_ = v___x_31_;
v_isShared_35_ = v_isSharedCheck_43_;
goto v_resetjp_33_;
}
else
{
lean_inc(v_toZero_32_);
lean_dec(v___x_31_);
v___x_34_ = lean_box(0);
v_isShared_35_ = v_isSharedCheck_43_;
goto v_resetjp_33_;
}
v_resetjp_33_:
{
lean_object* v___f_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_40_; 
v___f_36_ = ((lean_object*)(lp_mathlib_Rat_commGroupWithZero___closed__0));
v___x_37_ = ((lean_object*)(lp_mathlib_Rat_commGroupWithZero___closed__1));
v___x_38_ = ((lean_object*)(lp_mathlib_Rat_commGroupWithZero___closed__2));
if (v_isShared_35_ == 0)
{
lean_ctor_set(v___x_34_, 0, v___x_28_);
v___x_40_ = v___x_34_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v___x_28_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v_toZero_32_);
v___x_40_ = v_reuseFailAlloc_42_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
lean_object* v___x_41_; 
v___x_41_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_41_, 0, v___x_40_);
lean_ctor_set(v___x_41_, 1, v___x_37_);
lean_ctor_set(v___x_41_, 2, v___x_38_);
lean_ctor_set(v___x_41_, 3, v___f_36_);
return v___x_41_;
}
}
}
}
static lean_object* _init_lp_mathlib_Rat_commSemiring(void){
_start:
{
lean_object* v___x_45_; lean_object* v_toSemiring_46_; 
v___x_45_ = lp_mathlib_Rat_commRing;
v_toSemiring_46_ = lean_ctor_get(v___x_45_, 0);
lean_inc_ref(v_toSemiring_46_);
return v_toSemiring_46_;
}
}
static lean_object* _init_lp_mathlib_Rat_semiring(void){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_Rat_commSemiring;
return v___x_47_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Rat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_commRing = _init_lp_mathlib_Rat_commRing();
lean_mark_persistent(lp_mathlib_Rat_commRing);
lp_mathlib_Rat_commGroupWithZero = _init_lp_mathlib_Rat_commGroupWithZero();
lean_mark_persistent(lp_mathlib_Rat_commGroupWithZero);
lp_mathlib_Rat_commSemiring = _init_lp_mathlib_Rat_commSemiring();
lean_mark_persistent(lp_mathlib_Rat_commSemiring);
lp_mathlib_Rat_semiring = _init_lp_mathlib_Rat_semiring();
lean_mark_persistent(lp_mathlib_Rat_semiring);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Rat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Rat(builtin);
}
#ifdef __cplusplus
}
#endif
