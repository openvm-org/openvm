// Lean compiler output
// Module: Mathlib.Data.Nat.Choose.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Factorial.Basic public import Mathlib.Order.Monotone.Defs
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
lean_object* lp_mathlib_Nat_descFactorialBinary(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_factorialBinarySplitting(lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_choose(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_choose___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_fast__choose(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_fast__choose___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_multichoose(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_multichoose___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_choose(lean_object* v_x_1_, lean_object* v_x_2_){
_start:
{
lean_object* v_zero_3_; uint8_t v_isZero_4_; 
v_zero_3_ = lean_unsigned_to_nat(0u);
v_isZero_4_ = lean_nat_dec_eq(v_x_2_, v_zero_3_);
if (v_isZero_4_ == 1)
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(1u);
return v___x_5_;
}
else
{
uint8_t v_isZero_6_; 
v_isZero_6_ = lean_nat_dec_eq(v_x_1_, v_zero_3_);
if (v_isZero_6_ == 1)
{
return v_zero_3_;
}
else
{
lean_object* v_one_7_; lean_object* v_n_8_; lean_object* v_n_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v_one_7_ = lean_unsigned_to_nat(1u);
v_n_8_ = lean_nat_sub(v_x_2_, v_one_7_);
v_n_9_ = lean_nat_sub(v_x_1_, v_one_7_);
v___x_10_ = lp_mathlib_Nat_choose(v_n_9_, v_n_8_);
v___x_11_ = lean_nat_add(v_n_8_, v_one_7_);
lean_dec(v_n_8_);
v___x_12_ = lp_mathlib_Nat_choose(v_n_9_, v___x_11_);
lean_dec(v___x_11_);
lean_dec(v_n_9_);
v___x_13_ = lean_nat_add(v___x_10_, v___x_12_);
lean_dec(v___x_12_);
lean_dec(v___x_10_);
return v___x_13_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_choose___boxed(lean_object* v_x_14_, lean_object* v_x_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Nat_choose(v_x_14_, v_x_15_);
lean_dec(v_x_15_);
lean_dec(v_x_14_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg(lean_object* v_x_17_, lean_object* v_x_18_, lean_object* v_h__1_19_, lean_object* v_h__2_20_, lean_object* v_h__3_21_){
_start:
{
lean_object* v_zero_22_; uint8_t v_isZero_23_; 
v_zero_22_ = lean_unsigned_to_nat(0u);
v_isZero_23_ = lean_nat_dec_eq(v_x_18_, v_zero_22_);
if (v_isZero_23_ == 1)
{
lean_object* v___x_24_; 
lean_dec(v_h__3_21_);
lean_dec(v_h__2_20_);
v___x_24_ = lean_apply_1(v_h__1_19_, v_x_17_);
return v___x_24_;
}
else
{
lean_object* v_one_25_; lean_object* v_n_26_; uint8_t v_isZero_27_; 
lean_dec(v_h__1_19_);
v_one_25_ = lean_unsigned_to_nat(1u);
v_n_26_ = lean_nat_sub(v_x_18_, v_one_25_);
v_isZero_27_ = lean_nat_dec_eq(v_x_17_, v_zero_22_);
if (v_isZero_27_ == 1)
{
lean_object* v___x_28_; 
lean_dec(v_h__3_21_);
lean_dec(v_x_17_);
v___x_28_ = lean_apply_1(v_h__2_20_, v_n_26_);
return v___x_28_;
}
else
{
lean_object* v_n_29_; lean_object* v___x_30_; 
lean_dec(v_h__2_20_);
v_n_29_ = lean_nat_sub(v_x_17_, v_one_25_);
lean_dec(v_x_17_);
v___x_30_ = lean_apply_2(v_h__3_21_, v_n_29_, v_n_26_);
return v___x_30_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg___boxed(lean_object* v_x_31_, lean_object* v_x_32_, lean_object* v_h__1_33_, lean_object* v_h__2_34_, lean_object* v_h__3_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___redArg(v_x_31_, v_x_32_, v_h__1_33_, v_h__2_34_, v_h__3_35_);
lean_dec(v_x_32_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter(lean_object* v_motive_37_, lean_object* v_x_38_, lean_object* v_x_39_, lean_object* v_h__1_40_, lean_object* v_h__2_41_, lean_object* v_h__3_42_){
_start:
{
lean_object* v_zero_43_; uint8_t v_isZero_44_; 
v_zero_43_ = lean_unsigned_to_nat(0u);
v_isZero_44_ = lean_nat_dec_eq(v_x_39_, v_zero_43_);
if (v_isZero_44_ == 1)
{
lean_object* v___x_45_; 
lean_dec(v_h__3_42_);
lean_dec(v_h__2_41_);
v___x_45_ = lean_apply_1(v_h__1_40_, v_x_38_);
return v___x_45_;
}
else
{
lean_object* v_one_46_; lean_object* v_n_47_; uint8_t v_isZero_48_; 
lean_dec(v_h__1_40_);
v_one_46_ = lean_unsigned_to_nat(1u);
v_n_47_ = lean_nat_sub(v_x_39_, v_one_46_);
v_isZero_48_ = lean_nat_dec_eq(v_x_38_, v_zero_43_);
if (v_isZero_48_ == 1)
{
lean_object* v___x_49_; 
lean_dec(v_h__3_42_);
lean_dec(v_x_38_);
v___x_49_ = lean_apply_1(v_h__2_41_, v_n_47_);
return v___x_49_;
}
else
{
lean_object* v_n_50_; lean_object* v___x_51_; 
lean_dec(v_h__2_41_);
v_n_50_ = lean_nat_sub(v_x_38_, v_one_46_);
lean_dec(v_x_38_);
v___x_51_ = lean_apply_2(v_h__3_42_, v_n_50_, v_n_47_);
return v___x_51_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter___boxed(lean_object* v_motive_52_, lean_object* v_x_53_, lean_object* v_x_54_, lean_object* v_h__1_55_, lean_object* v_h__2_56_, lean_object* v_h__3_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib___private_Mathlib_Data_Nat_Choose_Basic_0__Nat_choose_match__1_splitter(v_motive_52_, v_x_53_, v_x_54_, v_h__1_55_, v_h__2_56_, v_h__3_57_);
lean_dec(v_x_54_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_fast__choose(lean_object* v_n_59_, lean_object* v_k_60_){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lp_mathlib_Nat_descFactorialBinary(v_n_59_, v_k_60_);
v___x_62_ = lp_mathlib_Nat_factorialBinarySplitting(v_k_60_);
v___x_63_ = lean_nat_div(v___x_61_, v___x_62_);
lean_dec(v___x_62_);
lean_dec(v___x_61_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_fast__choose___boxed(lean_object* v_n_64_, lean_object* v_k_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_Nat_fast__choose(v_n_64_, v_k_65_);
lean_dec(v_k_65_);
lean_dec(v_n_64_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_multichoose(lean_object* v_x_67_, lean_object* v_x_68_){
_start:
{
lean_object* v_zero_69_; uint8_t v_isZero_70_; 
v_zero_69_ = lean_unsigned_to_nat(0u);
v_isZero_70_ = lean_nat_dec_eq(v_x_68_, v_zero_69_);
if (v_isZero_70_ == 1)
{
lean_object* v___x_71_; 
v___x_71_ = lean_unsigned_to_nat(1u);
return v___x_71_;
}
else
{
uint8_t v_isZero_72_; 
v_isZero_72_ = lean_nat_dec_eq(v_x_67_, v_zero_69_);
if (v_isZero_72_ == 1)
{
return v_zero_69_;
}
else
{
lean_object* v_one_73_; lean_object* v_n_74_; lean_object* v_n_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v_one_73_ = lean_unsigned_to_nat(1u);
v_n_74_ = lean_nat_sub(v_x_68_, v_one_73_);
v_n_75_ = lean_nat_sub(v_x_67_, v_one_73_);
v___x_76_ = lean_nat_add(v_n_74_, v_one_73_);
v___x_77_ = lp_mathlib_Nat_multichoose(v_n_75_, v___x_76_);
lean_dec(v___x_76_);
v___x_78_ = lean_nat_add(v_n_75_, v_one_73_);
lean_dec(v_n_75_);
v___x_79_ = lp_mathlib_Nat_multichoose(v___x_78_, v_n_74_);
lean_dec(v_n_74_);
lean_dec(v___x_78_);
v___x_80_ = lean_nat_add(v___x_77_, v___x_79_);
lean_dec(v___x_79_);
lean_dec(v___x_77_);
return v___x_80_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_multichoose___boxed(lean_object* v_x_81_, lean_object* v_x_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Nat_multichoose(v_x_81_, v_x_82_);
lean_dec(v_x_82_);
lean_dec(v_x_81_);
return v_res_83_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Monotone_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Monotone_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Monotone_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Monotone_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Choose_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
