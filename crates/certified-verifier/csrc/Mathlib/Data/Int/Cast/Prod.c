// Lean compiler output
// Module: Mathlib.Data.Int.Cast.Prod
// Imports: public import Init public meta import Init public import Mathlib.Data.Int.Cast.Basic public import Mathlib.Data.Nat.Cast.Prod
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
lean_object* lp_mathlib_Prod_instAddMonoidWithOne___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_Prod_subNegMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__0(lean_object* v_toIntCast_1_, lean_object* v_toIntCast_2_, lean_object* v_n_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
lean_inc(v_n_3_);
v___x_4_ = lean_apply_1(v_toIntCast_1_, v_n_3_);
v___x_5_ = lean_apply_1(v_toIntCast_2_, v_n_3_);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_4_);
lean_ctor_set(v___x_6_, 1, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne___redArg(lean_object* v_inst_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v_toIntCast_9_; lean_object* v_toAddMonoidWithOne_10_; lean_object* v_toIntCast_11_; lean_object* v_toAddMonoidWithOne_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_27_; 
v_toIntCast_9_ = lean_ctor_get(v_inst_7_, 0);
lean_inc(v_toIntCast_9_);
v_toAddMonoidWithOne_10_ = lean_ctor_get(v_inst_7_, 1);
v_toIntCast_11_ = lean_ctor_get(v_inst_8_, 0);
lean_inc(v_toIntCast_11_);
v_toAddMonoidWithOne_12_ = lean_ctor_get(v_inst_8_, 1);
lean_inc_ref(v_toAddMonoidWithOne_12_);
lean_inc_ref(v_toAddMonoidWithOne_10_);
v___x_13_ = lp_mathlib_Prod_instAddMonoidWithOne___redArg(v_toAddMonoidWithOne_10_, v_toAddMonoidWithOne_12_);
v___x_14_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v_inst_7_);
lean_dec_ref(v_inst_7_);
v___x_15_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v_inst_8_);
v_isSharedCheck_27_ = !lean_is_exclusive(v_inst_8_);
if (v_isSharedCheck_27_ == 0)
{
lean_object* v_unused_28_; lean_object* v_unused_29_; lean_object* v_unused_30_; lean_object* v_unused_31_; lean_object* v_unused_32_; 
v_unused_28_ = lean_ctor_get(v_inst_8_, 4);
lean_dec(v_unused_28_);
v_unused_29_ = lean_ctor_get(v_inst_8_, 3);
lean_dec(v_unused_29_);
v_unused_30_ = lean_ctor_get(v_inst_8_, 2);
lean_dec(v_unused_30_);
v_unused_31_ = lean_ctor_get(v_inst_8_, 1);
lean_dec(v_unused_31_);
v_unused_32_ = lean_ctor_get(v_inst_8_, 0);
lean_dec(v_unused_32_);
v___x_17_ = v_inst_8_;
v_isShared_18_ = v_isSharedCheck_27_;
goto v_resetjp_16_;
}
else
{
lean_dec(v_inst_8_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_27_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_19_; lean_object* v_toNeg_20_; lean_object* v_toSub_21_; lean_object* v_toZSMul_22_; lean_object* v___f_23_; lean_object* v___x_25_; 
v___x_19_ = lp_mathlib_Prod_subNegMonoid___redArg(v___x_14_, v___x_15_);
v_toNeg_20_ = lean_ctor_get(v___x_19_, 1);
lean_inc(v_toNeg_20_);
v_toSub_21_ = lean_ctor_get(v___x_19_, 2);
lean_inc(v_toSub_21_);
v_toZSMul_22_ = lean_ctor_get(v___x_19_, 3);
lean_inc(v_toZSMul_22_);
lean_dec_ref(v___x_19_);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instAddGroupWithOne___redArg___lam__0), 3, 2);
lean_closure_set(v___f_23_, 0, v_toIntCast_9_);
lean_closure_set(v___f_23_, 1, v_toIntCast_11_);
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 4, v_toZSMul_22_);
lean_ctor_set(v___x_17_, 3, v_toSub_21_);
lean_ctor_set(v___x_17_, 2, v_toNeg_20_);
lean_ctor_set(v___x_17_, 1, v___x_13_);
lean_ctor_set(v___x_17_, 0, v___f_23_);
v___x_25_ = v___x_17_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___f_23_);
lean_ctor_set(v_reuseFailAlloc_26_, 1, v___x_13_);
lean_ctor_set(v_reuseFailAlloc_26_, 2, v_toNeg_20_);
lean_ctor_set(v_reuseFailAlloc_26_, 3, v_toSub_21_);
lean_ctor_set(v_reuseFailAlloc_26_, 4, v_toZSMul_22_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroupWithOne(lean_object* v_00_u03b1_33_, lean_object* v_00_u03b2_34_, lean_object* v_inst_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Prod_instAddGroupWithOne___redArg(v_inst_35_, v_inst_36_);
return v___x_37_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Int_Cast_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Int_Cast_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Int_Cast_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
