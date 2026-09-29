// Lean compiler output
// Module: Mathlib.Algebra.Ring.GrindInstances
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Defs public import Mathlib.Data.Int.Cast.Basic
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
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toGrindCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toGrindCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toGrindRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toGrindRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toGrindCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toGrindCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0(lean_object* v_toNPow_1_, lean_object* v_a_2_, lean_object* v_n_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toNPow_1_, v_n_3_, v_a_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1(lean_object* v_toZero_5_, lean_object* v_toOne_6_, lean_object* v_toNatCast_7_, lean_object* v_x_8_){
_start:
{
lean_object* v_zero_9_; uint8_t v_isZero_10_; 
v_zero_9_ = lean_unsigned_to_nat(0u);
v_isZero_10_ = lean_nat_dec_eq(v_x_8_, v_zero_9_);
if (v_isZero_10_ == 1)
{
lean_dec(v_toNatCast_7_);
lean_inc(v_toZero_5_);
return v_toZero_5_;
}
else
{
lean_object* v_one_11_; lean_object* v_n_12_; uint8_t v_isZero_13_; 
v_one_11_ = lean_unsigned_to_nat(1u);
v_n_12_ = lean_nat_sub(v_x_8_, v_one_11_);
v_isZero_13_ = lean_nat_dec_eq(v_n_12_, v_zero_9_);
if (v_isZero_13_ == 1)
{
lean_dec(v_n_12_);
lean_dec(v_toNatCast_7_);
lean_inc(v_toOne_6_);
return v_toOne_6_;
}
else
{
lean_object* v_n_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v_n_14_ = lean_nat_sub(v_n_12_, v_one_11_);
lean_dec(v_n_12_);
v___x_15_ = lean_unsigned_to_nat(2u);
v___x_16_ = lean_nat_add(v_n_14_, v___x_15_);
lean_dec(v_n_14_);
v___x_17_ = lean_apply_1(v_toNatCast_7_, v___x_16_);
return v___x_17_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed(lean_object* v_toZero_18_, lean_object* v_toOne_19_, lean_object* v_toNatCast_20_, lean_object* v_x_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1(v_toZero_18_, v_toOne_19_, v_toNatCast_20_, v_x_21_);
lean_dec(v_x_21_);
lean_dec(v_toOne_19_);
lean_dec(v_toZero_18_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring___redArg(lean_object* v_s_23_){
_start:
{
lean_object* v_toAddCommMonoid_24_; lean_object* v_toMonoid_25_; lean_object* v_toAdd_26_; lean_object* v_toNSMul_27_; lean_object* v_toMul_28_; lean_object* v_toNPow_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v_toNatCast_32_; lean_object* v_toOne_33_; lean_object* v___x_34_; lean_object* v_toZero_35_; lean_object* v___f_36_; lean_object* v___f_37_; lean_object* v___x_38_; 
v_toAddCommMonoid_24_ = lean_ctor_get(v_s_23_, 0);
v_toMonoid_25_ = lean_ctor_get(v_s_23_, 1);
v_toAdd_26_ = lean_ctor_get(v_toAddCommMonoid_24_, 1);
lean_inc(v_toAdd_26_);
v_toNSMul_27_ = lean_ctor_get(v_toAddCommMonoid_24_, 2);
lean_inc(v_toNSMul_27_);
v_toMul_28_ = lean_ctor_get(v_toMonoid_25_, 1);
lean_inc(v_toMul_28_);
v_toNPow_29_ = lean_ctor_get(v_toMonoid_25_, 2);
lean_inc(v_toNPow_29_);
lean_inc_ref(v_s_23_);
v___x_30_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_s_23_);
v___x_31_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_30_);
v_toNatCast_32_ = lean_ctor_get(v___x_31_, 0);
lean_inc_n(v_toNatCast_32_, 2);
v_toOne_33_ = lean_ctor_get(v___x_31_, 2);
lean_inc(v_toOne_33_);
lean_dec_ref(v___x_31_);
v___x_34_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_s_23_);
v_toZero_35_ = lean_ctor_get(v___x_34_, 1);
lean_inc(v_toZero_35_);
lean_dec_ref(v___x_34_);
v___f_36_ = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_36_, 0, v_toNPow_29_);
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_37_, 0, v_toZero_35_);
lean_closure_set(v___f_37_, 1, v_toOne_33_);
lean_closure_set(v___f_37_, 2, v_toNatCast_32_);
v___x_38_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_38_, 0, v_toAdd_26_);
lean_ctor_set(v___x_38_, 1, v_toMul_28_);
lean_ctor_set(v___x_38_, 2, v_toNatCast_32_);
lean_ctor_set(v___x_38_, 3, v___f_37_);
lean_ctor_set(v___x_38_, 4, v_toNSMul_27_);
lean_ctor_set(v___x_38_, 5, v___f_36_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toGrindSemiring(lean_object* v_00_u03b1_39_, lean_object* v_s_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Semiring_toGrindSemiring___redArg(v_s_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toGrindCommSemiring___redArg(lean_object* v_s_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Semiring_toGrindSemiring___redArg(v_s_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toGrindCommSemiring(lean_object* v_00_u03b1_44_, lean_object* v_s_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_Semiring_toGrindSemiring___redArg(v_s_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toGrindRing___redArg(lean_object* v_s_47_){
_start:
{
lean_object* v_toSemiring_48_; lean_object* v_toAddCommMonoid_49_; lean_object* v_toMonoid_50_; lean_object* v_toNeg_51_; lean_object* v_toSub_52_; lean_object* v_toZSMul_53_; lean_object* v_toAdd_54_; lean_object* v_toNSMul_55_; lean_object* v_toMul_56_; lean_object* v_toNPow_57_; lean_object* v___x_58_; lean_object* v_toAddMonoidWithOne_59_; lean_object* v_toIntCast_60_; lean_object* v___x_62_; uint8_t v_isShared_63_; uint8_t v_isSharedCheck_77_; 
v_toSemiring_48_ = lean_ctor_get(v_s_47_, 0);
lean_inc_ref(v_toSemiring_48_);
v_toAddCommMonoid_49_ = lean_ctor_get(v_toSemiring_48_, 0);
v_toMonoid_50_ = lean_ctor_get(v_toSemiring_48_, 1);
v_toNeg_51_ = lean_ctor_get(v_s_47_, 1);
lean_inc(v_toNeg_51_);
v_toSub_52_ = lean_ctor_get(v_s_47_, 2);
lean_inc(v_toSub_52_);
v_toZSMul_53_ = lean_ctor_get(v_s_47_, 3);
lean_inc(v_toZSMul_53_);
v_toAdd_54_ = lean_ctor_get(v_toAddCommMonoid_49_, 1);
lean_inc(v_toAdd_54_);
v_toNSMul_55_ = lean_ctor_get(v_toAddCommMonoid_49_, 2);
lean_inc(v_toNSMul_55_);
v_toMul_56_ = lean_ctor_get(v_toMonoid_50_, 1);
lean_inc(v_toMul_56_);
v_toNPow_57_ = lean_ctor_get(v_toMonoid_50_, 2);
lean_inc(v_toNPow_57_);
v___x_58_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_s_47_);
v_toAddMonoidWithOne_59_ = lean_ctor_get(v___x_58_, 1);
v_toIntCast_60_ = lean_ctor_get(v___x_58_, 0);
v_isSharedCheck_77_ = !lean_is_exclusive(v___x_58_);
if (v_isSharedCheck_77_ == 0)
{
lean_object* v_unused_78_; lean_object* v_unused_79_; lean_object* v_unused_80_; 
v_unused_78_ = lean_ctor_get(v___x_58_, 4);
lean_dec(v_unused_78_);
v_unused_79_ = lean_ctor_get(v___x_58_, 3);
lean_dec(v_unused_79_);
v_unused_80_ = lean_ctor_get(v___x_58_, 2);
lean_dec(v_unused_80_);
v___x_62_ = v___x_58_;
v_isShared_63_ = v_isSharedCheck_77_;
goto v_resetjp_61_;
}
else
{
lean_inc(v_toAddMonoidWithOne_59_);
lean_inc(v_toIntCast_60_);
lean_dec(v___x_58_);
v___x_62_ = lean_box(0);
v_isShared_63_ = v_isSharedCheck_77_;
goto v_resetjp_61_;
}
v_resetjp_61_:
{
lean_object* v_toNatCast_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v_toNatCast_67_; lean_object* v_toOne_68_; lean_object* v___x_69_; lean_object* v_toZero_70_; lean_object* v___f_71_; lean_object* v___f_72_; lean_object* v___x_73_; lean_object* v___x_75_; 
v_toNatCast_64_ = lean_ctor_get(v_toAddMonoidWithOne_59_, 0);
lean_inc(v_toNatCast_64_);
lean_dec_ref(v_toAddMonoidWithOne_59_);
lean_inc_ref(v_toSemiring_48_);
v___x_65_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_toSemiring_48_);
v___x_66_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_65_);
v_toNatCast_67_ = lean_ctor_get(v___x_66_, 0);
lean_inc(v_toNatCast_67_);
v_toOne_68_ = lean_ctor_get(v___x_66_, 2);
lean_inc(v_toOne_68_);
lean_dec_ref(v___x_66_);
v___x_69_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_48_);
v_toZero_70_ = lean_ctor_get(v___x_69_, 1);
lean_inc(v_toZero_70_);
lean_dec_ref(v___x_69_);
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_71_, 0, v_toNPow_57_);
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_Semiring_toGrindSemiring___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_72_, 0, v_toZero_70_);
lean_closure_set(v___f_72_, 1, v_toOne_68_);
lean_closure_set(v___f_72_, 2, v_toNatCast_67_);
v___x_73_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_73_, 0, v_toAdd_54_);
lean_ctor_set(v___x_73_, 1, v_toMul_56_);
lean_ctor_set(v___x_73_, 2, v_toNatCast_64_);
lean_ctor_set(v___x_73_, 3, v___f_72_);
lean_ctor_set(v___x_73_, 4, v_toNSMul_55_);
lean_ctor_set(v___x_73_, 5, v___f_71_);
if (v_isShared_63_ == 0)
{
lean_ctor_set(v___x_62_, 4, v_toZSMul_53_);
lean_ctor_set(v___x_62_, 3, v_toIntCast_60_);
lean_ctor_set(v___x_62_, 2, v_toSub_52_);
lean_ctor_set(v___x_62_, 1, v_toNeg_51_);
lean_ctor_set(v___x_62_, 0, v___x_73_);
v___x_75_ = v___x_62_;
goto v_reusejp_74_;
}
else
{
lean_object* v_reuseFailAlloc_76_; 
v_reuseFailAlloc_76_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_76_, 0, v___x_73_);
lean_ctor_set(v_reuseFailAlloc_76_, 1, v_toNeg_51_);
lean_ctor_set(v_reuseFailAlloc_76_, 2, v_toSub_52_);
lean_ctor_set(v_reuseFailAlloc_76_, 3, v_toIntCast_60_);
lean_ctor_set(v_reuseFailAlloc_76_, 4, v_toZSMul_53_);
v___x_75_ = v_reuseFailAlloc_76_;
goto v_reusejp_74_;
}
v_reusejp_74_:
{
return v___x_75_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toGrindRing(lean_object* v_00_u03b1_81_, lean_object* v_s_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_mathlib_Ring_toGrindRing___redArg(v_s_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toGrindCommRing___redArg(lean_object* v_s_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_Ring_toGrindRing___redArg(v_s_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toGrindCommRing(lean_object* v_00_u03b1_86_, lean_object* v_s_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_Ring_toGrindRing___redArg(v_s_87_);
return v___x_88_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(builtin);
}
#ifdef __cplusplus
}
#endif
