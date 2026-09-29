// Lean compiler output
// Module: Mathlib.Algebra.Ring.NegOnePow
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Int.Parity public import Mathlib.Algebra.Ring.Int.Units public import Mathlib.Data.ZMod.IntUnitsPower
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
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lp_mathlib_Additive_toMul(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* l_Int_pow(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Int_negOnePow___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_negOnePow___closed__0;
static lean_once_cell_t lp_mathlib_Int_negOnePow___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_negOnePow___closed__1;
static lean_once_cell_t lp_mathlib_Int_negOnePow___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_negOnePow___closed__2;
static lean_once_cell_t lp_mathlib_Int_negOnePow___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_negOnePow___closed__3;
static lean_once_cell_t lp_mathlib_Int_negOnePow___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_negOnePow___closed__4;
static lean_once_cell_t lp_mathlib_Int_negOnePow___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_negOnePow___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Int_negOnePow(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_negOnePow___boxed(lean_object*);
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_unsigned_to_nat(1u);
v___x_2_ = lean_nat_to_int(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_obj_once(&lp_mathlib_Int_negOnePow___closed__0, &lp_mathlib_Int_negOnePow___closed__0_once, _init_lp_mathlib_Int_negOnePow___closed__0);
v___x_4_ = lean_int_neg(v___x_3_);
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__2(void){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_obj_once(&lp_mathlib_Int_negOnePow___closed__1, &lp_mathlib_Int_negOnePow___closed__1_once, _init_lp_mathlib_Int_negOnePow___closed__1);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_5_);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__3(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_7_;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__4(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_Int_negOnePow___closed__5(void){
_start:
{
lean_object* v_natZero_9_; lean_object* v_intZero_10_; 
v_natZero_9_ = lean_unsigned_to_nat(0u);
v_intZero_10_ = lean_nat_to_int(v_natZero_9_);
return v_intZero_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_negOnePow(lean_object* v_n_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v_toFun_15_; lean_object* v___x_16_; lean_object* v___y_18_; lean_object* v_toFun_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v_intZero_25_; uint8_t v_isNeg_26_; 
v___x_12_ = lean_unsigned_to_nat(1u);
v___x_13_ = lean_obj_once(&lp_mathlib_Int_negOnePow___closed__2, &lp_mathlib_Int_negOnePow___closed__2_once, _init_lp_mathlib_Int_negOnePow___closed__2);
v___x_14_ = lean_obj_once(&lp_mathlib_Int_negOnePow___closed__3, &lp_mathlib_Int_negOnePow___closed__3_once, _init_lp_mathlib_Int_negOnePow___closed__3);
v_toFun_15_ = lean_ctor_get(v___x_14_, 0);
v___x_16_ = lean_obj_once(&lp_mathlib_Int_negOnePow___closed__4, &lp_mathlib_Int_negOnePow___closed__4_once, _init_lp_mathlib_Int_negOnePow___closed__4);
v_toFun_22_ = lean_ctor_get(v___x_16_, 0);
lean_inc(v_toFun_15_);
v___x_23_ = lean_apply_1(v_toFun_15_, v___x_13_);
lean_inc(v_toFun_22_);
v___x_24_ = lean_apply_1(v_toFun_22_, v___x_23_);
v_intZero_25_ = lean_obj_once(&lp_mathlib_Int_negOnePow___closed__5, &lp_mathlib_Int_negOnePow___closed__5_once, _init_lp_mathlib_Int_negOnePow___closed__5);
v_isNeg_26_ = lean_int_dec_lt(v_n_11_, v_intZero_25_);
if (v_isNeg_26_ == 0)
{
lean_object* v_val_27_; lean_object* v_inv_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_38_; 
v_val_27_ = lean_ctor_get(v___x_24_, 0);
v_inv_28_ = lean_ctor_get(v___x_24_, 1);
v_isSharedCheck_38_ = !lean_is_exclusive(v___x_24_);
if (v_isSharedCheck_38_ == 0)
{
v___x_30_ = v___x_24_;
v_isShared_31_ = v_isSharedCheck_38_;
goto v_resetjp_29_;
}
else
{
lean_inc(v_inv_28_);
lean_inc(v_val_27_);
lean_dec(v___x_24_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_38_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v_a_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_36_; 
v_a_32_ = lean_nat_abs(v_n_11_);
v___x_33_ = l_Int_pow(v_val_27_, v_a_32_);
lean_dec(v_val_27_);
v___x_34_ = l_Int_pow(v_inv_28_, v_a_32_);
lean_dec(v_a_32_);
lean_dec(v_inv_28_);
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 1, v___x_34_);
lean_ctor_set(v___x_30_, 0, v___x_33_);
v___x_36_ = v___x_30_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_37_; 
v_reuseFailAlloc_37_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_37_, 0, v___x_33_);
lean_ctor_set(v_reuseFailAlloc_37_, 1, v___x_34_);
v___x_36_ = v_reuseFailAlloc_37_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
v___y_18_ = v___x_36_;
goto v___jp_17_;
}
}
}
else
{
lean_object* v_val_39_; lean_object* v_inv_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_52_; 
v_val_39_ = lean_ctor_get(v___x_24_, 0);
v_inv_40_ = lean_ctor_get(v___x_24_, 1);
v_isSharedCheck_52_ = !lean_is_exclusive(v___x_24_);
if (v_isSharedCheck_52_ == 0)
{
v___x_42_ = v___x_24_;
v_isShared_43_ = v_isSharedCheck_52_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_inv_40_);
lean_inc(v_val_39_);
lean_dec(v___x_24_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_52_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v_abs_44_; lean_object* v_a_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_50_; 
v_abs_44_ = lean_nat_abs(v_n_11_);
v_a_45_ = lean_nat_sub(v_abs_44_, v___x_12_);
lean_dec(v_abs_44_);
v___x_46_ = lean_nat_add(v_a_45_, v___x_12_);
lean_dec(v_a_45_);
v___x_47_ = l_Int_pow(v_val_39_, v___x_46_);
lean_dec(v_val_39_);
v___x_48_ = l_Int_pow(v_inv_40_, v___x_46_);
lean_dec(v___x_46_);
lean_dec(v_inv_40_);
if (v_isShared_43_ == 0)
{
lean_ctor_set(v___x_42_, 1, v___x_47_);
lean_ctor_set(v___x_42_, 0, v___x_48_);
v___x_50_ = v___x_42_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v___x_48_);
lean_ctor_set(v_reuseFailAlloc_51_, 1, v___x_47_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
v___y_18_ = v___x_50_;
goto v___jp_17_;
}
}
}
v___jp_17_:
{
lean_object* v_toFun_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v_toFun_19_ = lean_ctor_get(v___x_16_, 0);
lean_inc(v_toFun_15_);
v___x_20_ = lean_apply_1(v_toFun_15_, v___y_18_);
lean_inc(v_toFun_19_);
v___x_21_ = lean_apply_1(v_toFun_19_, v___x_20_);
return v___x_21_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_negOnePow___boxed(lean_object* v_n_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Int_negOnePow(v_n_53_);
lean_dec(v_n_53_);
return v_res_54_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(builtin);
}
#ifdef __cplusplus
}
#endif
