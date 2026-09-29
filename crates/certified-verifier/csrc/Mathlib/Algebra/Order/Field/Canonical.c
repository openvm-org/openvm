// Lean compiler output
// Module: Mathlib.Algebra.Order.Field.Canonical
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Order.GroupWithZero.Canonical public import Mathlib.Algebra.Order.Ring.Canonical
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
lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CanonicallyOrderedAdd_toLinearOrderedCommGroupWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CanonicallyOrderedAdd_toLinearOrderedCommGroupWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CanonicallyOrderedAdd_toLinearOrderedCommGroupWithZero___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; lean_object* v_toCommMonoidWithZero_4_; lean_object* v_toInv_5_; lean_object* v_toDiv_6_; lean_object* v_toZPow_7_; lean_object* v___x_9_; uint8_t v_isShared_10_; uint8_t v_isSharedCheck_18_; 
v___x_3_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v_inst_1_);
v_toCommMonoidWithZero_4_ = lean_ctor_get(v___x_3_, 0);
v_toInv_5_ = lean_ctor_get(v___x_3_, 1);
v_toDiv_6_ = lean_ctor_get(v___x_3_, 2);
v_toZPow_7_ = lean_ctor_get(v___x_3_, 3);
v_isSharedCheck_18_ = !lean_is_exclusive(v___x_3_);
if (v_isSharedCheck_18_ == 0)
{
v___x_9_ = v___x_3_;
v_isShared_10_ = v_isSharedCheck_18_;
goto v_resetjp_8_;
}
else
{
lean_inc(v_toZPow_7_);
lean_inc(v_toDiv_6_);
lean_inc(v_toInv_5_);
lean_inc(v_toCommMonoidWithZero_4_);
lean_dec(v___x_3_);
v___x_9_ = lean_box(0);
v_isShared_10_ = v_isSharedCheck_18_;
goto v_resetjp_8_;
}
v_resetjp_8_:
{
lean_object* v_toCommSemiring_11_; lean_object* v___x_12_; lean_object* v_toZero_13_; lean_object* v___x_14_; lean_object* v___x_16_; 
v_toCommSemiring_11_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref(v_toCommSemiring_11_);
lean_dec_ref(v_inst_1_);
v___x_12_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_11_);
v_toZero_13_ = lean_ctor_get(v___x_12_, 1);
lean_inc(v_toZero_13_);
lean_dec_ref(v___x_12_);
v___x_14_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_14_, 0, v_toCommMonoidWithZero_4_);
lean_ctor_set(v___x_14_, 1, v_inst_2_);
lean_ctor_set(v___x_14_, 2, v_toZero_13_);
if (v_isShared_10_ == 0)
{
lean_ctor_set(v___x_9_, 0, v___x_14_);
v___x_16_ = v___x_9_;
goto v_reusejp_15_;
}
else
{
lean_object* v_reuseFailAlloc_17_; 
v_reuseFailAlloc_17_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_17_, 0, v___x_14_);
lean_ctor_set(v_reuseFailAlloc_17_, 1, v_toInv_5_);
lean_ctor_set(v_reuseFailAlloc_17_, 2, v_toDiv_6_);
lean_ctor_set(v_reuseFailAlloc_17_, 3, v_toZPow_7_);
v___x_16_ = v_reuseFailAlloc_17_;
goto v_reusejp_15_;
}
v_reusejp_15_:
{
return v___x_16_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CanonicallyOrderedAdd_toLinearOrderedCommGroupWithZero(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___x_23_; lean_object* v_toCommMonoidWithZero_24_; lean_object* v_toInv_25_; lean_object* v_toDiv_26_; lean_object* v_toZPow_27_; lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_38_; 
v___x_23_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v_inst_20_);
v_toCommMonoidWithZero_24_ = lean_ctor_get(v___x_23_, 0);
v_toInv_25_ = lean_ctor_get(v___x_23_, 1);
v_toDiv_26_ = lean_ctor_get(v___x_23_, 2);
v_toZPow_27_ = lean_ctor_get(v___x_23_, 3);
v_isSharedCheck_38_ = !lean_is_exclusive(v___x_23_);
if (v_isSharedCheck_38_ == 0)
{
v___x_29_ = v___x_23_;
v_isShared_30_ = v_isSharedCheck_38_;
goto v_resetjp_28_;
}
else
{
lean_inc(v_toZPow_27_);
lean_inc(v_toDiv_26_);
lean_inc(v_toInv_25_);
lean_inc(v_toCommMonoidWithZero_24_);
lean_dec(v___x_23_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_38_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v_toCommSemiring_31_; lean_object* v___x_32_; lean_object* v_toZero_33_; lean_object* v___x_34_; lean_object* v___x_36_; 
v_toCommSemiring_31_ = lean_ctor_get(v_inst_20_, 0);
lean_inc_ref(v_toCommSemiring_31_);
lean_dec_ref(v_inst_20_);
v___x_32_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_31_);
v_toZero_33_ = lean_ctor_get(v___x_32_, 1);
lean_inc(v_toZero_33_);
lean_dec_ref(v___x_32_);
v___x_34_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_34_, 0, v_toCommMonoidWithZero_24_);
lean_ctor_set(v___x_34_, 1, v_inst_21_);
lean_ctor_set(v___x_34_, 2, v_toZero_33_);
if (v_isShared_30_ == 0)
{
lean_ctor_set(v___x_29_, 0, v___x_34_);
v___x_36_ = v___x_29_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_37_; 
v_reuseFailAlloc_37_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_37_, 0, v___x_34_);
lean_ctor_set(v_reuseFailAlloc_37_, 1, v_toInv_25_);
lean_ctor_set(v_reuseFailAlloc_37_, 2, v_toDiv_26_);
lean_ctor_set(v_reuseFailAlloc_37_, 3, v_toZPow_27_);
v___x_36_ = v_reuseFailAlloc_37_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
return v___x_36_;
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Field_Canonical(builtin);
}
#ifdef __cplusplus
}
#endif
