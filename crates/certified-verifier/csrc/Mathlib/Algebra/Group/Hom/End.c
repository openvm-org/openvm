// Lean compiler output
// Module: Mathlib.Algebra.Group.Hom.End
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Instances public import Mathlib.Algebra.Ring.Defs
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
lean_object* lp_mathlib_OneHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_OneHom_id___lam__0(lean_object*);
lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_End_instMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OneHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0 = (const lean_object*)&lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_n_2_, lean_object* v___y_3_){
_start:
{
lean_object* v_toNSMul_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v_toNSMul_4_ = lean_ctor_get(v_inst_1_, 2);
lean_inc(v_toNSMul_4_);
lean_dec_ref(v_inst_1_);
v___x_5_ = lp_mathlib_OneHom_id___lam__0(v___y_3_);
v___x_6_ = lean_apply_2(v_toNSMul_4_, v_n_2_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0___boxed(lean_object* v_inst_7_, lean_object* v_n_8_, lean_object* v___y_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0(v_inst_7_, v_n_8_, v___y_9_);
lean_dec(v___y_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg(lean_object* v_inst_12_){
_start:
{
lean_object* v___f_13_; lean_object* v___x_14_; lean_object* v___f_15_; lean_object* v___x_16_; 
lean_inc_ref(v_inst_12_);
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_13_, 0, v_inst_12_);
v___x_14_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___redArg(v_inst_12_);
v___f_15_ = ((lean_object*)(lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___closed__0));
v___x_16_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_16_, 0, v___f_13_);
lean_ctor_set(v___x_16_, 1, v___x_14_);
lean_ctor_set(v___x_16_, 2, v___f_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddMonoidWithOne(lean_object* v_M_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg(v_inst_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring___redArg(lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___f_25_; lean_object* v___x_26_; 
lean_inc_ref(v_inst_20_);
v___x_21_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___redArg(v_inst_20_);
v___x_22_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_20_);
v___x_23_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_22_);
v___x_24_ = lp_mathlib_AddMonoid_End_instMonoid___redArg(v___x_23_);
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddMonoidWithOne___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_25_, 0, v_inst_20_);
v___x_26_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_26_, 0, v___x_21_);
lean_ctor_set(v___x_26_, 1, v___x_24_);
lean_ctor_set(v___x_26_, 2, v___f_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instSemiring(lean_object* v_M_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_AddMonoid_End_instSemiring___redArg(v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing___redArg(lean_object* v_inst_30_){
_start:
{
lean_object* v_toAddMonoid_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v_toNeg_35_; lean_object* v_toSub_36_; lean_object* v_toZSMul_37_; lean_object* v___f_38_; lean_object* v___x_39_; 
v_toAddMonoid_31_ = lean_ctor_get(v_inst_30_, 0);
lean_inc_ref(v_toAddMonoid_31_);
v___x_32_ = lp_mathlib_AddMonoid_End_instSemiring___redArg(v_toAddMonoid_31_);
lean_inc_ref(v_inst_30_);
v___x_33_ = lp_mathlib_AddMonoid_End_instAddCommGroup___redArg(v_inst_30_);
v___x_34_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_33_);
v_toNeg_35_ = lean_ctor_get(v___x_34_, 1);
lean_inc(v_toNeg_35_);
lean_dec_ref(v___x_34_);
v_toSub_36_ = lean_ctor_get(v___x_33_, 2);
lean_inc(v_toSub_36_);
v_toZSMul_37_ = lean_ctor_get(v___x_33_, 3);
lean_inc(v_toZSMul_37_);
lean_dec_ref(v___x_33_);
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_38_, 0, v_inst_30_);
v___x_39_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_39_, 0, v___x_32_);
lean_ctor_set(v___x_39_, 1, v_toNeg_35_);
lean_ctor_set(v___x_39_, 2, v_toSub_36_);
lean_ctor_set(v___x_39_, 3, v_toZSMul_37_);
lean_ctor_set(v___x_39_, 4, v___f_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instRing(lean_object* v_M_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_AddMonoid_End_instRing___redArg(v_inst_41_);
return v___x_42_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_End(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Hom_End(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_End(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Hom_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Hom_End(builtin);
}
#ifdef __cplusplus
}
#endif
