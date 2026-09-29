// Lean compiler output
// Module: Mathlib.RingTheory.IntegralDomain
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Polynomial.Roots public import Mathlib.Algebra.Ring.GeomSum public import Mathlib.Data.Fintype.Inv public import Mathlib.GroupTheory.SpecificGroups.Cyclic public import Mathlib.Tactic.FieldSimp
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
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Fintype_bijInv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_fieldOfDomain___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_fieldOfDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__0(lean_object* v_toMul_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toMul_1_, v_a_2_, v_b_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1(lean_object* v_inst_5_, lean_object* v_toZero_6_, lean_object* v___x_7_, lean_object* v_toMul_8_, lean_object* v_inst_9_, lean_object* v_a_10_){
_start:
{
lean_object* v___x_11_; uint8_t v___x_12_; 
lean_inc_ref(v_inst_5_);
lean_inc(v_toZero_6_);
lean_inc(v_a_10_);
v___x_11_ = lean_apply_2(v_inst_5_, v_a_10_, v_toZero_6_);
v___x_12_ = lean_unbox(v___x_11_);
if (v___x_12_ == 0)
{
lean_object* v_toMulOneClass_13_; lean_object* v___x_14_; lean_object* v_toOne_15_; lean_object* v___f_16_; lean_object* v___x_17_; 
lean_dec(v_toZero_6_);
v_toMulOneClass_13_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_toMulOneClass_13_);
lean_dec_ref(v___x_7_);
v___x_14_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_13_);
v_toOne_15_ = lean_ctor_get(v___x_14_, 0);
lean_inc(v_toOne_15_);
lean_dec_ref(v___x_14_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__0), 3, 2);
lean_closure_set(v___f_16_, 0, v_toMul_8_);
lean_closure_set(v___f_16_, 1, v_a_10_);
v___x_17_ = lp_mathlib_Fintype_bijInv___redArg(v_inst_9_, v_inst_5_, v___f_16_, v_toOne_15_);
return v___x_17_;
}
else
{
lean_dec(v_a_10_);
lean_dec(v_inst_9_);
lean_dec(v_toMul_8_);
lean_dec_ref(v___x_7_);
lean_dec_ref(v_inst_5_);
return v_toZero_6_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v_toMonoid_23_; lean_object* v_toMul_24_; lean_object* v_toZero_25_; lean_object* v_toOne_26_; lean_object* v_toMul_27_; lean_object* v___f_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
lean_inc_ref(v_inst_18_);
v___x_21_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_18_);
lean_inc_ref(v___x_21_);
v___x_22_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_21_);
v_toMonoid_23_ = lean_ctor_get(v_inst_18_, 0);
v_toMul_24_ = lean_ctor_get(v___x_22_, 0);
lean_inc(v_toMul_24_);
v_toZero_25_ = lean_ctor_get(v___x_22_, 1);
lean_inc(v_toZero_25_);
lean_dec_ref(v___x_22_);
v_toOne_26_ = lean_ctor_get(v_toMonoid_23_, 0);
v_toMul_27_ = lean_ctor_get(v_toMonoid_23_, 1);
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_groupWithZeroOfCancel___redArg___lam__1), 6, 5);
lean_closure_set(v___f_28_, 0, v_inst_19_);
lean_closure_set(v___f_28_, 1, v_toZero_25_);
lean_closure_set(v___f_28_, 2, v___x_21_);
lean_closure_set(v___f_28_, 3, v_toMul_24_);
lean_closure_set(v___f_28_, 4, v_inst_20_);
lean_inc_ref_n(v___f_28_, 2);
lean_inc_ref(v_toMonoid_23_);
v___x_29_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_29_, 0, lean_box(0));
lean_closure_set(v___x_29_, 1, v_toMonoid_23_);
lean_closure_set(v___x_29_, 2, v___f_28_);
lean_inc_n(v_toMul_27_, 2);
lean_inc_n(v_toOne_26_, 2);
v___x_30_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_30_, 0, lean_box(0));
lean_closure_set(v___x_30_, 1, v_toOne_26_);
lean_closure_set(v___x_30_, 2, v_toMul_27_);
v___x_31_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_31_, 0, lean_box(0));
lean_closure_set(v___x_31_, 1, v_toOne_26_);
lean_closure_set(v___x_31_, 2, v_toMul_27_);
lean_closure_set(v___x_31_, 3, v___f_28_);
lean_closure_set(v___x_31_, 4, v___x_30_);
v___x_32_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_32_, 0, v_inst_18_);
lean_ctor_set(v___x_32_, 1, v___f_28_);
lean_ctor_set(v___x_32_, 2, v___x_29_);
lean_ctor_set(v___x_32_, 3, v___x_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_groupWithZeroOfCancel(lean_object* v_M_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(v_inst_34_, v_inst_36_, v_inst_37_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0(lean_object* v_toNatCast_40_, lean_object* v_toDiv_41_, lean_object* v_toMul_42_, lean_object* v_x_43_, lean_object* v___y_44_){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_45_ = lp_mathlib_NNRat_castRec___redArg(v_toNatCast_40_, v_toDiv_41_, v_x_43_);
v___x_46_ = lean_apply_2(v_toMul_42_, v___x_45_, v___y_44_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__1(lean_object* v_toNatCast_47_, lean_object* v_toIntCast_48_, lean_object* v_toDiv_49_, lean_object* v_toMul_50_, lean_object* v_x_51_, lean_object* v___y_52_){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_53_ = lp_mathlib_Rat_castRec___redArg(v_toNatCast_47_, v_toIntCast_48_, v_toDiv_49_, v_x_51_);
v___x_54_ = lean_apply_2(v_toMul_50_, v___x_53_, v___y_52_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain___redArg(lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v_toSemiring_58_; lean_object* v_toIntCast_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v_toMonoid_62_; lean_object* v_toInv_63_; lean_object* v_toDiv_64_; lean_object* v_toZPow_65_; lean_object* v_toNatCast_66_; lean_object* v_toMul_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___f_70_; lean_object* v___f_71_; lean_object* v___x_72_; 
v_toSemiring_58_ = lean_ctor_get(v_inst_55_, 0);
v_toIntCast_59_ = lean_ctor_get(v_inst_55_, 4);
v___x_60_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_toSemiring_58_);
v___x_61_ = lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(v___x_60_, v_inst_56_, v_inst_57_);
v_toMonoid_62_ = lean_ctor_get(v_toSemiring_58_, 1);
v_toInv_63_ = lean_ctor_get(v___x_61_, 1);
lean_inc(v_toInv_63_);
v_toDiv_64_ = lean_ctor_get(v___x_61_, 2);
lean_inc_n(v_toDiv_64_, 5);
v_toZPow_65_ = lean_ctor_get(v___x_61_, 3);
lean_inc(v_toZPow_65_);
lean_dec_ref(v___x_61_);
v_toNatCast_66_ = lean_ctor_get(v_toSemiring_58_, 2);
v_toMul_67_ = lean_ctor_get(v_toMonoid_62_, 1);
lean_inc_n(v_toNatCast_66_, 4);
v___x_68_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(v___x_68_, 0, lean_box(0));
lean_closure_set(v___x_68_, 1, v_toNatCast_66_);
lean_closure_set(v___x_68_, 2, v_toDiv_64_);
lean_inc_n(v_toIntCast_59_, 2);
v___x_69_ = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(v___x_69_, 0, lean_box(0));
lean_closure_set(v___x_69_, 1, v_toNatCast_66_);
lean_closure_set(v___x_69_, 2, v_toIntCast_59_);
lean_closure_set(v___x_69_, 3, v_toDiv_64_);
lean_inc_n(v_toMul_67_, 2);
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0), 5, 3);
lean_closure_set(v___f_70_, 0, v_toNatCast_66_);
lean_closure_set(v___f_70_, 1, v_toDiv_64_);
lean_closure_set(v___f_70_, 2, v_toMul_67_);
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__1), 6, 4);
lean_closure_set(v___f_71_, 0, v_toNatCast_66_);
lean_closure_set(v___f_71_, 1, v_toIntCast_59_);
lean_closure_set(v___f_71_, 2, v_toDiv_64_);
lean_closure_set(v___f_71_, 3, v_toMul_67_);
v___x_72_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_72_, 0, v_inst_55_);
lean_ctor_set(v___x_72_, 1, v_toInv_63_);
lean_ctor_set(v___x_72_, 2, v_toDiv_64_);
lean_ctor_set(v___x_72_, 3, v_toZPow_65_);
lean_ctor_set(v___x_72_, 4, v___x_68_);
lean_ctor_set(v___x_72_, 5, v___x_69_);
lean_ctor_set(v___x_72_, 6, v___f_70_);
lean_ctor_set(v___x_72_, 7, v___f_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_divisionRingOfIsDomain(lean_object* v_R_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Fintype_divisionRingOfIsDomain___redArg(v_inst_74_, v_inst_76_, v_inst_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_fieldOfDomain___redArg(lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; lean_object* v_toRing_83_; lean_object* v_toInv_84_; lean_object* v_toDiv_85_; lean_object* v_toZPow_86_; lean_object* v_toNNRatCast_87_; lean_object* v_toRatCast_88_; lean_object* v___x_90_; uint8_t v_isShared_91_; uint8_t v_isSharedCheck_105_; 
lean_inc(v_inst_81_);
lean_inc_ref(v_inst_80_);
lean_inc_ref(v_inst_79_);
v___x_82_ = lp_mathlib_Fintype_divisionRingOfIsDomain___redArg(v_inst_79_, v_inst_80_, v_inst_81_);
v_toRing_83_ = lean_ctor_get(v___x_82_, 0);
v_toInv_84_ = lean_ctor_get(v___x_82_, 1);
v_toDiv_85_ = lean_ctor_get(v___x_82_, 2);
v_toZPow_86_ = lean_ctor_get(v___x_82_, 3);
v_toNNRatCast_87_ = lean_ctor_get(v___x_82_, 4);
v_toRatCast_88_ = lean_ctor_get(v___x_82_, 5);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_82_);
if (v_isSharedCheck_105_ == 0)
{
lean_object* v_unused_106_; lean_object* v_unused_107_; 
v_unused_106_ = lean_ctor_get(v___x_82_, 7);
lean_dec(v_unused_106_);
v_unused_107_ = lean_ctor_get(v___x_82_, 6);
lean_dec(v_unused_107_);
v___x_90_ = v___x_82_;
v_isShared_91_ = v_isSharedCheck_105_;
goto v_resetjp_89_;
}
else
{
lean_inc(v_toRatCast_88_);
lean_inc(v_toNNRatCast_87_);
lean_inc(v_toZPow_86_);
lean_inc(v_toDiv_85_);
lean_inc(v_toInv_84_);
lean_inc(v_toRing_83_);
lean_dec(v___x_82_);
v___x_90_ = lean_box(0);
v_isShared_91_ = v_isSharedCheck_105_;
goto v_resetjp_89_;
}
v_resetjp_89_:
{
lean_object* v_toSemiring_92_; lean_object* v_toIntCast_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v_toMonoid_96_; lean_object* v_toDiv_97_; lean_object* v_toNatCast_98_; lean_object* v_toMul_99_; lean_object* v___f_100_; lean_object* v___f_101_; lean_object* v___x_103_; 
v_toSemiring_92_ = lean_ctor_get(v_inst_79_, 0);
lean_inc_ref(v_toSemiring_92_);
v_toIntCast_93_ = lean_ctor_get(v_inst_79_, 4);
lean_inc(v_toIntCast_93_);
lean_dec_ref(v_inst_79_);
v___x_94_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_toSemiring_92_);
v___x_95_ = lp_mathlib_Fintype_groupWithZeroOfCancel___redArg(v___x_94_, v_inst_80_, v_inst_81_);
v_toMonoid_96_ = lean_ctor_get(v_toSemiring_92_, 1);
lean_inc_ref(v_toMonoid_96_);
v_toDiv_97_ = lean_ctor_get(v___x_95_, 2);
lean_inc_n(v_toDiv_97_, 2);
lean_dec_ref(v___x_95_);
v_toNatCast_98_ = lean_ctor_get(v_toSemiring_92_, 2);
lean_inc_n(v_toNatCast_98_, 2);
lean_dec_ref(v_toSemiring_92_);
v_toMul_99_ = lean_ctor_get(v_toMonoid_96_, 1);
lean_inc_n(v_toMul_99_, 2);
lean_dec_ref(v_toMonoid_96_);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__1), 6, 4);
lean_closure_set(v___f_100_, 0, v_toNatCast_98_);
lean_closure_set(v___f_100_, 1, v_toIntCast_93_);
lean_closure_set(v___f_100_, 2, v_toDiv_97_);
lean_closure_set(v___f_100_, 3, v_toMul_99_);
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_divisionRingOfIsDomain___redArg___lam__0), 5, 3);
lean_closure_set(v___f_101_, 0, v_toNatCast_98_);
lean_closure_set(v___f_101_, 1, v_toDiv_97_);
lean_closure_set(v___f_101_, 2, v_toMul_99_);
if (v_isShared_91_ == 0)
{
lean_ctor_set(v___x_90_, 7, v___f_100_);
lean_ctor_set(v___x_90_, 6, v___f_101_);
v___x_103_ = v___x_90_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_toRing_83_);
lean_ctor_set(v_reuseFailAlloc_104_, 1, v_toInv_84_);
lean_ctor_set(v_reuseFailAlloc_104_, 2, v_toDiv_85_);
lean_ctor_set(v_reuseFailAlloc_104_, 3, v_toZPow_86_);
lean_ctor_set(v_reuseFailAlloc_104_, 4, v_toNNRatCast_87_);
lean_ctor_set(v_reuseFailAlloc_104_, 5, v_toRatCast_88_);
lean_ctor_set(v_reuseFailAlloc_104_, 6, v___f_101_);
lean_ctor_set(v_reuseFailAlloc_104_, 7, v___f_100_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_fieldOfDomain(lean_object* v_R_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_mathlib_Fintype_fieldOfDomain___redArg(v_inst_109_, v_inst_111_, v_inst_112_);
return v___x_113_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_GeomSum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Inv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FieldSimp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_IntegralDomain(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_GeomSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FieldSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_IntegralDomain(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_GeomSum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Inv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FieldSimp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_IntegralDomain(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_GeomSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_SpecificGroups_Cyclic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FieldSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_IntegralDomain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_IntegralDomain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_IntegralDomain(builtin);
}
#ifdef __cplusplus
}
#endif
