// Lean compiler output
// Module: Mathlib.Data.Nat.Cast.Prod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Prod public import Mathlib.Data.Nat.Cast.Defs
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
lean_object* lp_mathlib_Prod_instAddMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoidWithOne___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoidWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoidWithOne___redArg___lam__0(lean_object* v_toNatCast_1_, lean_object* v_toNatCast_2_, lean_object* v_n_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
lean_inc(v_n_3_);
v___x_4_ = lean_apply_1(v_toNatCast_1_, v_n_3_);
v___x_5_ = lean_apply_1(v_toNatCast_2_, v_n_3_);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_4_);
lean_ctor_set(v___x_6_, 1, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoidWithOne___redArg(lean_object* v_inst_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v_toNatCast_9_; lean_object* v_toAddMonoid_10_; lean_object* v_toOne_11_; lean_object* v_toNatCast_12_; lean_object* v_toAddMonoid_13_; lean_object* v_toOne_14_; lean_object* v___x_16_; uint8_t v_isShared_17_; uint8_t v_isSharedCheck_24_; 
v_toNatCast_9_ = lean_ctor_get(v_inst_7_, 0);
lean_inc(v_toNatCast_9_);
v_toAddMonoid_10_ = lean_ctor_get(v_inst_7_, 1);
lean_inc_ref(v_toAddMonoid_10_);
v_toOne_11_ = lean_ctor_get(v_inst_7_, 2);
lean_inc(v_toOne_11_);
lean_dec_ref(v_inst_7_);
v_toNatCast_12_ = lean_ctor_get(v_inst_8_, 0);
v_toAddMonoid_13_ = lean_ctor_get(v_inst_8_, 1);
v_toOne_14_ = lean_ctor_get(v_inst_8_, 2);
v_isSharedCheck_24_ = !lean_is_exclusive(v_inst_8_);
if (v_isSharedCheck_24_ == 0)
{
v___x_16_ = v_inst_8_;
v_isShared_17_ = v_isSharedCheck_24_;
goto v_resetjp_15_;
}
else
{
lean_inc(v_toOne_14_);
lean_inc(v_toAddMonoid_13_);
lean_inc(v_toNatCast_12_);
lean_dec(v_inst_8_);
v___x_16_ = lean_box(0);
v_isShared_17_ = v_isSharedCheck_24_;
goto v_resetjp_15_;
}
v_resetjp_15_:
{
lean_object* v___f_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_22_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instAddMonoidWithOne___redArg___lam__0), 3, 2);
lean_closure_set(v___f_18_, 0, v_toNatCast_9_);
lean_closure_set(v___f_18_, 1, v_toNatCast_12_);
v___x_19_ = lp_mathlib_Prod_instAddMonoid___redArg(v_toAddMonoid_10_, v_toAddMonoid_13_);
v___x_20_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_20_, 0, v_toOne_11_);
lean_ctor_set(v___x_20_, 1, v_toOne_14_);
if (v_isShared_17_ == 0)
{
lean_ctor_set(v___x_16_, 2, v___x_20_);
lean_ctor_set(v___x_16_, 1, v___x_19_);
lean_ctor_set(v___x_16_, 0, v___f_18_);
v___x_22_ = v___x_16_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v___f_18_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v___x_19_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v___x_20_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoidWithOne(lean_object* v_00_u03b1_25_, lean_object* v_00_u03b2_26_, lean_object* v_inst_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Prod_instAddMonoidWithOne___redArg(v_inst_27_, v_inst_28_);
return v___x_29_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
