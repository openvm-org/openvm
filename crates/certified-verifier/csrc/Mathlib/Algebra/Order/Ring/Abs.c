// Lean compiler output
// Module: Mathlib.Algebra.Order.Ring.Abs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Abs public import Mathlib.Algebra.Order.Ring.Basic public import Mathlib.Algebra.Order.Ring.Int public import Mathlib.Algebra.Ring.Divisibility.Basic public import Mathlib.Algebra.Ring.Int.Units public import Mathlib.Data.Nat.Cast.Order.Ring
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
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_abs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_absHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_absHom___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_absHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_absHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_absHom___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_3_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_2_);
v___x_4_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_1_);
v___x_5_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_4_);
lean_dec_ref(v___x_4_);
v___x_6_ = lean_alloc_closure((void*)(lp_mathlib_abs___boxed), 4, 3);
lean_closure_set(v___x_6_, 0, lean_box(0));
lean_closure_set(v___x_6_, 1, v___x_3_);
lean_closure_set(v___x_6_, 2, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_absHom___redArg___boxed(lean_object* v_inst_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_absHom___redArg(v_inst_7_, v_inst_8_);
lean_dec_ref(v_inst_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_absHom(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_absHom___redArg(v_inst_11_, v_inst_12_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_absHom___boxed(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_absHom(v_00_u03b1_15_, v_inst_16_, v_inst_17_, v_inst_18_);
lean_dec_ref(v_inst_17_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___redArg(lean_object* v_inst_20_, lean_object* v_a_21_, lean_object* v_b_22_, lean_object* v_x_23_){
_start:
{
lean_object* v___x_24_; lean_object* v_toAddMonoidWithOne_25_; lean_object* v_toOne_26_; lean_object* v_toSemiring_27_; lean_object* v___x_28_; lean_object* v_toMonoid_29_; lean_object* v_toMul_30_; lean_object* v_toAdd_31_; lean_object* v_toNPow_32_; lean_object* v_zero_33_; uint8_t v_isZero_34_; 
lean_inc_ref(v_inst_20_);
v___x_24_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_20_);
v_toAddMonoidWithOne_25_ = lean_ctor_get(v___x_24_, 1);
lean_inc_ref(v_toAddMonoidWithOne_25_);
lean_dec_ref(v___x_24_);
v_toOne_26_ = lean_ctor_get(v_toAddMonoidWithOne_25_, 2);
lean_inc(v_toOne_26_);
lean_dec_ref(v_toAddMonoidWithOne_25_);
v_toSemiring_27_ = lean_ctor_get(v_inst_20_, 0);
lean_inc_ref(v_toSemiring_27_);
v___x_28_ = lp_mathlib_instDistribOfSemiring___redArg(v_toSemiring_27_);
v_toMonoid_29_ = lean_ctor_get(v_toSemiring_27_, 1);
v_toMul_30_ = lean_ctor_get(v___x_28_, 0);
lean_inc(v_toMul_30_);
v_toAdd_31_ = lean_ctor_get(v___x_28_, 1);
lean_inc(v_toAdd_31_);
lean_dec_ref(v___x_28_);
v_toNPow_32_ = lean_ctor_get(v_toMonoid_29_, 2);
lean_inc(v_toNPow_32_);
v_zero_33_ = lean_unsigned_to_nat(0u);
v_isZero_34_ = lean_nat_dec_eq(v_x_23_, v_zero_33_);
if (v_isZero_34_ == 1)
{
lean_dec(v_toNPow_32_);
lean_dec(v_toAdd_31_);
lean_dec(v_toMul_30_);
lean_dec(v_b_22_);
lean_dec(v_a_21_);
lean_dec_ref(v_inst_20_);
return v_toOne_26_;
}
else
{
lean_object* v_one_35_; lean_object* v_n_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
lean_dec(v_toOne_26_);
v_one_35_ = lean_unsigned_to_nat(1u);
v_n_36_ = lean_nat_sub(v_x_23_, v_one_35_);
lean_inc(v_b_22_);
lean_inc(v_a_21_);
v___x_37_ = lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___redArg(v_inst_20_, v_a_21_, v_b_22_, v_n_36_);
v___x_38_ = lean_apply_2(v_toMul_30_, v_a_21_, v___x_37_);
v___x_39_ = lean_nat_add(v_n_36_, v_one_35_);
lean_dec(v_n_36_);
v___x_40_ = lean_apply_2(v_toNPow_32_, v___x_39_, v_b_22_);
v___x_41_ = lean_apply_2(v_toAdd_31_, v___x_38_, v___x_40_);
return v___x_41_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___redArg___boxed(lean_object* v_inst_42_, lean_object* v_a_43_, lean_object* v_b_44_, lean_object* v_x_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___redArg(v_inst_42_, v_a_43_, v_b_44_, v_x_45_);
lean_dec(v_x_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_, lean_object* v_a_49_, lean_object* v_b_50_, lean_object* v_x_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___redArg(v_inst_48_, v_a_49_, v_b_50_, v_x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum___boxed(lean_object* v_00_u03b1_53_, lean_object* v_inst_54_, lean_object* v_a_55_, lean_object* v_b_56_, lean_object* v_x_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum(v_00_u03b1_53_, v_inst_54_, v_a_55_, v_b_56_, v_x_57_);
lean_dec(v_x_57_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter___redArg(lean_object* v_x_59_, lean_object* v_h__1_60_, lean_object* v_h__2_61_){
_start:
{
lean_object* v_zero_62_; uint8_t v_isZero_63_; 
v_zero_62_ = lean_unsigned_to_nat(0u);
v_isZero_63_ = lean_nat_dec_eq(v_x_59_, v_zero_62_);
if (v_isZero_63_ == 1)
{
lean_object* v___x_64_; lean_object* v___x_65_; 
lean_dec(v_h__2_61_);
v___x_64_ = lean_box(0);
v___x_65_ = lean_apply_1(v_h__1_60_, v___x_64_);
return v___x_65_;
}
else
{
lean_object* v_one_66_; lean_object* v_n_67_; lean_object* v___x_68_; 
lean_dec(v_h__1_60_);
v_one_66_ = lean_unsigned_to_nat(1u);
v_n_67_ = lean_nat_sub(v_x_59_, v_one_66_);
v___x_68_ = lean_apply_1(v_h__2_61_, v_n_67_);
return v___x_68_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter___redArg___boxed(lean_object* v_x_69_, lean_object* v_h__1_70_, lean_object* v_h__2_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter___redArg(v_x_69_, v_h__1_70_, v_h__2_71_);
lean_dec(v_x_69_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter(lean_object* v_motive_73_, lean_object* v_x_74_, lean_object* v_h__1_75_, lean_object* v_h__2_76_){
_start:
{
lean_object* v_zero_77_; uint8_t v_isZero_78_; 
v_zero_77_ = lean_unsigned_to_nat(0u);
v_isZero_78_ = lean_nat_dec_eq(v_x_74_, v_zero_77_);
if (v_isZero_78_ == 1)
{
lean_object* v___x_79_; lean_object* v___x_80_; 
lean_dec(v_h__2_76_);
v___x_79_ = lean_box(0);
v___x_80_ = lean_apply_1(v_h__1_75_, v___x_79_);
return v___x_80_;
}
else
{
lean_object* v_one_81_; lean_object* v_n_82_; lean_object* v___x_83_; 
lean_dec(v_h__1_75_);
v_one_81_ = lean_unsigned_to_nat(1u);
v_n_82_ = lean_nat_sub(v_x_74_, v_one_81_);
v___x_83_ = lean_apply_1(v_h__2_76_, v_n_82_);
return v___x_83_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter___boxed(lean_object* v_motive_84_, lean_object* v_x_85_, lean_object* v_h__1_86_, lean_object* v_h__2_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib___private_Mathlib_Algebra_Order_Ring_Abs_0__geomSum_match__1_splitter(v_motive_84_, v_x_85_, v_h__1_86_, v_h__2_87_);
lean_dec(v_x_85_);
return v_res_88_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Abs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Divisibility_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Abs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Divisibility_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Ring_Abs(builtin);
}
#ifdef __cplusplus
}
#endif
