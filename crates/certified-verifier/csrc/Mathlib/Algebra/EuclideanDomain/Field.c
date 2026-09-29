// Lean compiler output
// Module: Mathlib.Algebra.EuclideanDomain.Field
// Imports: public import Init public meta import Init public import Mathlib.Algebra.EuclideanDomain.Defs public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.GroupWithZero.Units.Basic
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
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toEuclideanDomain___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toEuclideanDomain___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toEuclideanDomain___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toEuclideanDomain(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toEuclideanDomain___redArg___lam__0(lean_object* v_toDiv_1_, lean_object* v_x1_2_, lean_object* v_x2_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toDiv_1_, v_x1_2_, v_x2_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toEuclideanDomain___redArg___lam__1(lean_object* v_toMul_5_, lean_object* v_toDiv_6_, lean_object* v_toSub_7_, lean_object* v_a_8_, lean_object* v_b_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
lean_inc(v_b_9_);
lean_inc(v_a_8_);
v___x_10_ = lean_apply_2(v_toMul_5_, v_a_8_, v_b_9_);
v___x_11_ = lean_apply_2(v_toDiv_6_, v___x_10_, v_b_9_);
v___x_12_ = lean_apply_2(v_toSub_7_, v_a_8_, v___x_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toEuclideanDomain___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v_toCommRing_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v_toDiv_17_; lean_object* v_toRing_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v_toSub_21_; lean_object* v___x_22_; lean_object* v_toCommSemiring_23_; lean_object* v___x_24_; lean_object* v_toMul_25_; lean_object* v___f_26_; lean_object* v___f_27_; lean_object* v___x_28_; 
v_toCommRing_14_ = lean_ctor_get(v_inst_13_, 0);
lean_inc_ref(v_toCommRing_14_);
lean_inc_ref(v_inst_13_);
v___x_15_ = lp_mathlib_Field_toDivisionRing___redArg(v_inst_13_);
v___x_16_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v___x_15_);
v_toDiv_17_ = lean_ctor_get(v___x_16_, 2);
lean_inc_n(v_toDiv_17_, 2);
lean_dec_ref(v___x_16_);
v_toRing_18_ = lean_ctor_get(v___x_15_, 0);
lean_inc_ref(v_toRing_18_);
lean_dec_ref(v___x_15_);
v___x_19_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toRing_18_);
v___x_20_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_19_);
lean_dec_ref(v___x_19_);
v_toSub_21_ = lean_ctor_get(v___x_20_, 2);
lean_inc(v_toSub_21_);
lean_dec_ref(v___x_20_);
v___x_22_ = lp_mathlib_Field_toSemifield___redArg(v_inst_13_);
lean_dec_ref(v_inst_13_);
v_toCommSemiring_23_ = lean_ctor_get(v___x_22_, 0);
lean_inc_ref(v_toCommSemiring_23_);
lean_dec_ref(v___x_22_);
v___x_24_ = lp_mathlib_instDistribOfSemiring___redArg(v_toCommSemiring_23_);
v_toMul_25_ = lean_ctor_get(v___x_24_, 0);
lean_inc(v_toMul_25_);
lean_dec_ref(v___x_24_);
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_Field_toEuclideanDomain___redArg___lam__0), 3, 1);
lean_closure_set(v___f_26_, 0, v_toDiv_17_);
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_Field_toEuclideanDomain___redArg___lam__1), 5, 3);
lean_closure_set(v___f_27_, 0, v_toMul_25_);
lean_closure_set(v___f_27_, 1, v_toDiv_17_);
lean_closure_set(v___f_27_, 2, v_toSub_21_);
v___x_28_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_28_, 0, v_toCommRing_14_);
lean_ctor_set(v___x_28_, 1, v___f_26_);
lean_ctor_set(v___x_28_, 2, v___f_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toEuclideanDomain(lean_object* v_K_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_Field_toEuclideanDomain___redArg(v_inst_30_);
return v___x_31_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Field(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Field(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Field(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_EuclideanDomain_Field(builtin);
}
#ifdef __cplusplus
}
#endif
