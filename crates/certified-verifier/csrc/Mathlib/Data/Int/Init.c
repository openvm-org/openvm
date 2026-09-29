// Lean compiler output
// Module: Mathlib.Data.Int.Init
// Imports: public import Init public meta import Init public import Batteries.Logic public import Mathlib.Data.Int.Notation public import Mathlib.Data.Nat.Notation public import Mathlib.Tactic.DepRewrite
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lean_int_neg_succ_of_nat(lean_object*);
lean_object* lean_int_emod(lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Int_succ___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_succ___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_succ(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_succ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_pred(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_pred___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Int_inductionOn_x27_pos_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_pos___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_pos___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_pos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_pos___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_neg___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_neg___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Int_inductionOn_x27___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_inductionOn_x27___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_negInduction___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_negInduction___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_negInduction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_negInduction___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Int_leInduction___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Int_leInduction___redArg___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_leInduction___redArg___closed__0 = (const lean_object*)&lp_mathlib_Int_leInduction___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction__down___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction__down___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction__down(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction__down___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Int_strongRec___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Int_strongRec___redArg___lam__2___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_strongRec___redArg___closed__0 = (const lean_object*)&lp_mathlib_Int_strongRec___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_natMod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_natMod___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Int_succ___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_unsigned_to_nat(1u);
v___x_2_ = lean_nat_to_int(v___x_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_succ(lean_object* v_a_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_obj_once(&lp_mathlib_Int_succ___closed__0, &lp_mathlib_Int_succ___closed__0_once, _init_lp_mathlib_Int_succ___closed__0);
v___x_5_ = lean_int_add(v_a_3_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_succ___boxed(lean_object* v_a_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Int_succ(v_a_6_);
lean_dec(v_a_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_pred(lean_object* v_a_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_obj_once(&lp_mathlib_Int_succ___closed__0, &lp_mathlib_Int_succ___closed__0_once, _init_lp_mathlib_Int_succ___closed__0);
v___x_10_ = lean_int_sub(v_a_8_, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_pred___boxed(lean_object* v_a_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Int_pred(v_a_11_);
lean_dec(v_a_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Int_inductionOn_x27_pos_spec__0(lean_object* v_a_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_nat_to_int(v_a_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_pos___redArg(lean_object* v_b_15_, lean_object* v_zero_16_, lean_object* v_succ_17_, lean_object* v_n_18_){
_start:
{
lean_object* v_zero_19_; uint8_t v_isZero_20_; 
v_zero_19_ = lean_unsigned_to_nat(0u);
v_isZero_20_ = lean_nat_dec_eq(v_n_18_, v_zero_19_);
if (v_isZero_20_ == 1)
{
lean_dec(v_succ_17_);
lean_inc(v_zero_16_);
return v_zero_16_;
}
else
{
lean_object* v_one_21_; lean_object* v_n_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; 
v_one_21_ = lean_unsigned_to_nat(1u);
v_n_22_ = lean_nat_sub(v_n_18_, v_one_21_);
lean_inc(v_n_22_);
v___x_23_ = lean_nat_to_int(v_n_22_);
v___x_24_ = lean_int_add(v_b_15_, v___x_23_);
lean_dec(v___x_23_);
lean_inc(v_succ_17_);
v___x_25_ = lp_mathlib_Int_inductionOn_x27_pos___redArg(v_b_15_, v_zero_16_, v_succ_17_, v_n_22_);
lean_dec(v_n_22_);
v___x_26_ = lean_apply_3(v_succ_17_, v___x_24_, lean_box(0), v___x_25_);
return v___x_26_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_pos___redArg___boxed(lean_object* v_b_27_, lean_object* v_zero_28_, lean_object* v_succ_29_, lean_object* v_n_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Int_inductionOn_x27_pos___redArg(v_b_27_, v_zero_28_, v_succ_29_, v_n_30_);
lean_dec(v_n_30_);
lean_dec(v_zero_28_);
lean_dec(v_b_27_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_pos(lean_object* v_motive_32_, lean_object* v_b_33_, lean_object* v_zero_34_, lean_object* v_succ_35_, lean_object* v_n_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Int_inductionOn_x27_pos___redArg(v_b_33_, v_zero_34_, v_succ_35_, v_n_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_pos___boxed(lean_object* v_motive_38_, lean_object* v_b_39_, lean_object* v_zero_40_, lean_object* v_succ_41_, lean_object* v_n_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Int_inductionOn_x27_pos(v_motive_38_, v_b_39_, v_zero_40_, v_succ_41_, v_n_42_);
lean_dec(v_n_42_);
lean_dec(v_zero_40_);
lean_dec(v_b_39_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_neg___redArg(lean_object* v_b_44_, lean_object* v_zero_45_, lean_object* v_pred_46_, lean_object* v_n_47_){
_start:
{
lean_object* v_zero_48_; uint8_t v_isZero_49_; 
v_zero_48_ = lean_unsigned_to_nat(0u);
v_isZero_49_ = lean_nat_dec_eq(v_n_47_, v_zero_48_);
if (v_isZero_49_ == 1)
{
lean_object* v___x_50_; 
v___x_50_ = lean_apply_3(v_pred_46_, v_b_44_, lean_box(0), v_zero_45_);
return v___x_50_;
}
else
{
lean_object* v_one_51_; lean_object* v_n_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v_one_51_ = lean_unsigned_to_nat(1u);
v_n_52_ = lean_nat_sub(v_n_47_, v_one_51_);
lean_inc(v_n_52_);
v___x_53_ = lean_int_neg_succ_of_nat(v_n_52_);
v___x_54_ = lean_int_add(v_b_44_, v___x_53_);
lean_dec(v___x_53_);
lean_inc(v_pred_46_);
v___x_55_ = lp_mathlib_Int_inductionOn_x27_neg___redArg(v_b_44_, v_zero_45_, v_pred_46_, v_n_52_);
lean_dec(v_n_52_);
v___x_56_ = lean_apply_3(v_pred_46_, v___x_54_, lean_box(0), v___x_55_);
return v___x_56_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_neg___redArg___boxed(lean_object* v_b_57_, lean_object* v_zero_58_, lean_object* v_pred_59_, lean_object* v_n_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Int_inductionOn_x27_neg___redArg(v_b_57_, v_zero_58_, v_pred_59_, v_n_60_);
lean_dec(v_n_60_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_neg(lean_object* v_motive_62_, lean_object* v_b_63_, lean_object* v_zero_64_, lean_object* v_pred_65_, lean_object* v_n_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_Int_inductionOn_x27_neg___redArg(v_b_63_, v_zero_64_, v_pred_65_, v_n_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27_neg___boxed(lean_object* v_motive_68_, lean_object* v_b_69_, lean_object* v_zero_70_, lean_object* v_pred_71_, lean_object* v_n_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_Int_inductionOn_x27_neg(v_motive_68_, v_b_69_, v_zero_70_, v_pred_71_, v_n_72_);
lean_dec(v_n_72_);
return v_res_73_;
}
}
static lean_object* _init_lp_mathlib_Int_inductionOn_x27___redArg___closed__0(void){
_start:
{
lean_object* v_natZero_74_; lean_object* v_intZero_75_; 
v_natZero_74_ = lean_unsigned_to_nat(0u);
v_intZero_75_ = lean_nat_to_int(v_natZero_74_);
return v_intZero_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27___redArg(lean_object* v_z_76_, lean_object* v_b_77_, lean_object* v_zero_78_, lean_object* v_succ_79_, lean_object* v_pred_80_){
_start:
{
lean_object* v___x_81_; lean_object* v_intZero_82_; uint8_t v_isNeg_83_; 
v___x_81_ = lean_int_sub(v_z_76_, v_b_77_);
v_intZero_82_ = lean_obj_once(&lp_mathlib_Int_inductionOn_x27___redArg___closed__0, &lp_mathlib_Int_inductionOn_x27___redArg___closed__0_once, _init_lp_mathlib_Int_inductionOn_x27___redArg___closed__0);
v_isNeg_83_ = lean_int_dec_lt(v___x_81_, v_intZero_82_);
if (v_isNeg_83_ == 0)
{
lean_object* v_a_84_; lean_object* v___x_85_; 
lean_dec(v_pred_80_);
v_a_84_ = lean_nat_abs(v___x_81_);
lean_dec(v___x_81_);
v___x_85_ = lp_mathlib_Int_inductionOn_x27_pos___redArg(v_b_77_, v_zero_78_, v_succ_79_, v_a_84_);
lean_dec(v_a_84_);
lean_dec(v_zero_78_);
lean_dec(v_b_77_);
return v___x_85_;
}
else
{
lean_object* v_abs_86_; lean_object* v_one_87_; lean_object* v_a_88_; lean_object* v___x_89_; 
lean_dec(v_succ_79_);
v_abs_86_ = lean_nat_abs(v___x_81_);
lean_dec(v___x_81_);
v_one_87_ = lean_unsigned_to_nat(1u);
v_a_88_ = lean_nat_sub(v_abs_86_, v_one_87_);
lean_dec(v_abs_86_);
v___x_89_ = lp_mathlib_Int_inductionOn_x27_neg___redArg(v_b_77_, v_zero_78_, v_pred_80_, v_a_88_);
lean_dec(v_a_88_);
return v___x_89_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27___redArg___boxed(lean_object* v_z_90_, lean_object* v_b_91_, lean_object* v_zero_92_, lean_object* v_succ_93_, lean_object* v_pred_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Int_inductionOn_x27___redArg(v_z_90_, v_b_91_, v_zero_92_, v_succ_93_, v_pred_94_);
lean_dec(v_z_90_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27(lean_object* v_motive_96_, lean_object* v_z_97_, lean_object* v_b_98_, lean_object* v_zero_99_, lean_object* v_succ_100_, lean_object* v_pred_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_Int_inductionOn_x27___redArg(v_z_97_, v_b_98_, v_zero_99_, v_succ_100_, v_pred_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_inductionOn_x27___boxed(lean_object* v_motive_103_, lean_object* v_z_104_, lean_object* v_b_105_, lean_object* v_zero_106_, lean_object* v_succ_107_, lean_object* v_pred_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_Int_inductionOn_x27(v_motive_103_, v_z_104_, v_b_105_, v_zero_106_, v_succ_107_, v_pred_108_);
lean_dec(v_z_104_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter___redArg(lean_object* v_x_110_, lean_object* v_h__1_111_, lean_object* v_h__2_112_){
_start:
{
lean_object* v_zero_113_; uint8_t v_isZero_114_; 
v_zero_113_ = lean_unsigned_to_nat(0u);
v_isZero_114_ = lean_nat_dec_eq(v_x_110_, v_zero_113_);
if (v_isZero_114_ == 1)
{
lean_object* v___x_115_; lean_object* v___x_116_; 
lean_dec(v_h__2_112_);
v___x_115_ = lean_box(0);
v___x_116_ = lean_apply_1(v_h__1_111_, v___x_115_);
return v___x_116_;
}
else
{
lean_object* v_one_117_; lean_object* v_n_118_; lean_object* v___x_119_; 
lean_dec(v_h__1_111_);
v_one_117_ = lean_unsigned_to_nat(1u);
v_n_118_ = lean_nat_sub(v_x_110_, v_one_117_);
v___x_119_ = lean_apply_1(v_h__2_112_, v_n_118_);
return v___x_119_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter___redArg___boxed(lean_object* v_x_120_, lean_object* v_h__1_121_, lean_object* v_h__2_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter___redArg(v_x_120_, v_h__1_121_, v_h__2_122_);
lean_dec(v_x_120_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter(lean_object* v_motive_124_, lean_object* v_x_125_, lean_object* v_h__1_126_, lean_object* v_h__2_127_){
_start:
{
lean_object* v_zero_128_; uint8_t v_isZero_129_; 
v_zero_128_ = lean_unsigned_to_nat(0u);
v_isZero_129_ = lean_nat_dec_eq(v_x_125_, v_zero_128_);
if (v_isZero_129_ == 1)
{
lean_object* v___x_130_; lean_object* v___x_131_; 
lean_dec(v_h__2_127_);
v___x_130_ = lean_box(0);
v___x_131_ = lean_apply_1(v_h__1_126_, v___x_130_);
return v___x_131_;
}
else
{
lean_object* v_one_132_; lean_object* v_n_133_; lean_object* v___x_134_; 
lean_dec(v_h__1_126_);
v_one_132_ = lean_unsigned_to_nat(1u);
v_n_133_ = lean_nat_sub(v_x_125_, v_one_132_);
v___x_134_ = lean_apply_1(v_h__2_127_, v_n_133_);
return v___x_134_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter___boxed(lean_object* v_motive_135_, lean_object* v_x_136_, lean_object* v_h__1_137_, lean_object* v_h__2_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib___private_Mathlib_Data_Int_Init_0__Int_inductionOn_x27_pos_match__1_splitter(v_motive_135_, v_x_136_, v_h__1_137_, v_h__2_138_);
lean_dec(v_x_136_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_negInduction___redArg(lean_object* v_nat_140_, lean_object* v_neg_141_, lean_object* v_x_142_){
_start:
{
lean_object* v_intZero_143_; uint8_t v_isNeg_144_; 
v_intZero_143_ = lean_obj_once(&lp_mathlib_Int_inductionOn_x27___redArg___closed__0, &lp_mathlib_Int_inductionOn_x27___redArg___closed__0_once, _init_lp_mathlib_Int_inductionOn_x27___redArg___closed__0);
v_isNeg_144_ = lean_int_dec_lt(v_x_142_, v_intZero_143_);
if (v_isNeg_144_ == 0)
{
lean_object* v_a_145_; lean_object* v___x_146_; 
lean_dec(v_neg_141_);
v_a_145_ = lean_nat_abs(v_x_142_);
v___x_146_ = lean_apply_1(v_nat_140_, v_a_145_);
return v___x_146_;
}
else
{
lean_object* v_abs_147_; lean_object* v_one_148_; lean_object* v_a_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v_abs_147_ = lean_nat_abs(v_x_142_);
v_one_148_ = lean_unsigned_to_nat(1u);
v_a_149_ = lean_nat_sub(v_abs_147_, v_one_148_);
lean_dec(v_abs_147_);
v___x_150_ = lean_nat_add(v_a_149_, v_one_148_);
lean_dec(v_a_149_);
v___x_151_ = lean_apply_2(v_neg_141_, v_nat_140_, v___x_150_);
return v___x_151_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_negInduction___redArg___boxed(lean_object* v_nat_152_, lean_object* v_neg_153_, lean_object* v_x_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_Int_negInduction___redArg(v_nat_152_, v_neg_153_, v_x_154_);
lean_dec(v_x_154_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_negInduction(lean_object* v_motive_156_, lean_object* v_nat_157_, lean_object* v_neg_158_, lean_object* v_x_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_Int_negInduction___redArg(v_nat_157_, v_neg_158_, v_x_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_negInduction___boxed(lean_object* v_motive_161_, lean_object* v_nat_162_, lean_object* v_neg_163_, lean_object* v_x_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_Int_negInduction(v_motive_161_, v_nat_162_, v_neg_163_, v_x_164_);
lean_dec(v_x_164_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__0(lean_object* v_base_166_, lean_object* v_x_167_){
_start:
{
lean_inc(v_base_166_);
return v_base_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__0___boxed(lean_object* v_base_168_, lean_object* v_x_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_Int_leInduction___redArg___lam__0(v_base_168_, v_x_169_);
lean_dec(v_base_168_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__1(lean_object* v_succ_171_, lean_object* v_k_172_, lean_object* v_hle_173_, lean_object* v_ih_174_, lean_object* v_x_175_){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = lean_apply_1(v_ih_174_, lean_box(0));
v___x_177_ = lean_apply_3(v_succ_171_, v_k_172_, lean_box(0), v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__2(lean_object* v_x_178_, lean_object* v_x_179_, lean_object* v_x_180_, lean_object* v_x_181_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___lam__2___boxed(lean_object* v_x_182_, lean_object* v_x_183_, lean_object* v_x_184_, lean_object* v_x_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Int_leInduction___redArg___lam__2(v_x_182_, v_x_183_, v_x_184_, v_x_185_);
lean_dec(v_x_184_);
lean_dec(v_x_182_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg(lean_object* v_m_188_, lean_object* v_base_189_, lean_object* v_succ_190_, lean_object* v_n_191_){
_start:
{
lean_object* v___f_192_; lean_object* v___f_193_; lean_object* v___f_194_; lean_object* v___x_13__overap_195_; lean_object* v___x_196_; 
v___f_192_ = lean_alloc_closure((void*)(lp_mathlib_Int_leInduction___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_192_, 0, v_base_189_);
v___f_193_ = lean_alloc_closure((void*)(lp_mathlib_Int_leInduction___redArg___lam__1), 5, 1);
lean_closure_set(v___f_193_, 0, v_succ_190_);
v___f_194_ = ((lean_object*)(lp_mathlib_Int_leInduction___redArg___closed__0));
v___x_13__overap_195_ = lp_mathlib_Int_inductionOn_x27___redArg(v_n_191_, v_m_188_, v___f_192_, v___f_193_, v___f_194_);
v___x_196_ = lean_apply_1(v___x_13__overap_195_, lean_box(0));
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___redArg___boxed(lean_object* v_m_197_, lean_object* v_base_198_, lean_object* v_succ_199_, lean_object* v_n_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Int_leInduction___redArg(v_m_197_, v_base_198_, v_succ_199_, v_n_200_);
lean_dec(v_n_200_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction(lean_object* v_m_202_, lean_object* v_motive_203_, lean_object* v_base_204_, lean_object* v_succ_205_, lean_object* v_n_206_, lean_object* v_hmn_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_Int_leInduction___redArg(v_m_202_, v_base_204_, v_succ_205_, v_n_206_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInduction___boxed(lean_object* v_m_209_, lean_object* v_motive_210_, lean_object* v_base_211_, lean_object* v_succ_212_, lean_object* v_n_213_, lean_object* v_hmn_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_Int_leInduction(v_m_209_, v_motive_210_, v_base_211_, v_succ_212_, v_n_213_, v_hmn_214_);
lean_dec(v_n_213_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction___redArg(lean_object* v_m_216_, lean_object* v_base_217_, lean_object* v_succ_218_, lean_object* v_n_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lp_mathlib_Int_leInduction___redArg(v_m_216_, v_base_217_, v_succ_218_, v_n_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction___redArg___boxed(lean_object* v_m_221_, lean_object* v_base_222_, lean_object* v_succ_223_, lean_object* v_n_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_Int_le__induction___redArg(v_m_221_, v_base_222_, v_succ_223_, v_n_224_);
lean_dec(v_n_224_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction(lean_object* v_m_226_, lean_object* v_motive_227_, lean_object* v_base_228_, lean_object* v_succ_229_, lean_object* v_n_230_, lean_object* v_hmn_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_Int_leInduction___redArg(v_m_226_, v_base_228_, v_succ_229_, v_n_230_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction___boxed(lean_object* v_m_233_, lean_object* v_motive_234_, lean_object* v_base_235_, lean_object* v_succ_236_, lean_object* v_n_237_, lean_object* v_hmn_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_Int_le__induction(v_m_233_, v_motive_234_, v_base_235_, v_succ_236_, v_n_237_, v_hmn_238_);
lean_dec(v_n_237_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown___redArg___lam__2(lean_object* v_pred_240_, lean_object* v_k_241_, lean_object* v_hle_242_, lean_object* v_ih_243_, lean_object* v_x_244_){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_245_ = lean_apply_1(v_ih_243_, lean_box(0));
v___x_246_ = lean_apply_3(v_pred_240_, v_k_241_, lean_box(0), v___x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown___redArg(lean_object* v_m_247_, lean_object* v_base_248_, lean_object* v_pred_249_, lean_object* v_n_250_){
_start:
{
lean_object* v___f_251_; lean_object* v___f_252_; lean_object* v___f_253_; lean_object* v___x_13__overap_254_; lean_object* v___x_255_; 
v___f_251_ = lean_alloc_closure((void*)(lp_mathlib_Int_leInduction___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_251_, 0, v_base_248_);
v___f_252_ = ((lean_object*)(lp_mathlib_Int_leInduction___redArg___closed__0));
v___f_253_ = lean_alloc_closure((void*)(lp_mathlib_Int_leInductionDown___redArg___lam__2), 5, 1);
lean_closure_set(v___f_253_, 0, v_pred_249_);
v___x_13__overap_254_ = lp_mathlib_Int_inductionOn_x27___redArg(v_n_250_, v_m_247_, v___f_251_, v___f_252_, v___f_253_);
v___x_255_ = lean_apply_1(v___x_13__overap_254_, lean_box(0));
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown___redArg___boxed(lean_object* v_m_256_, lean_object* v_base_257_, lean_object* v_pred_258_, lean_object* v_n_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_Int_leInductionDown___redArg(v_m_256_, v_base_257_, v_pred_258_, v_n_259_);
lean_dec(v_n_259_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown(lean_object* v_m_261_, lean_object* v_motive_262_, lean_object* v_base_263_, lean_object* v_pred_264_, lean_object* v_n_265_, lean_object* v_hnm_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_mathlib_Int_leInductionDown___redArg(v_m_261_, v_base_263_, v_pred_264_, v_n_265_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_leInductionDown___boxed(lean_object* v_m_268_, lean_object* v_motive_269_, lean_object* v_base_270_, lean_object* v_pred_271_, lean_object* v_n_272_, lean_object* v_hnm_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_Int_leInductionDown(v_m_268_, v_motive_269_, v_base_270_, v_pred_271_, v_n_272_, v_hnm_273_);
lean_dec(v_n_272_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction__down___redArg(lean_object* v_m_275_, lean_object* v_base_276_, lean_object* v_pred_277_, lean_object* v_n_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_mathlib_Int_leInductionDown___redArg(v_m_275_, v_base_276_, v_pred_277_, v_n_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction__down___redArg___boxed(lean_object* v_m_280_, lean_object* v_base_281_, lean_object* v_pred_282_, lean_object* v_n_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_mathlib_Int_le__induction__down___redArg(v_m_280_, v_base_281_, v_pred_282_, v_n_283_);
lean_dec(v_n_283_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction__down(lean_object* v_m_285_, lean_object* v_motive_286_, lean_object* v_base_287_, lean_object* v_pred_288_, lean_object* v_n_289_, lean_object* v_hnm_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_Int_leInductionDown___redArg(v_m_285_, v_base_287_, v_pred_288_, v_n_289_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_le__induction__down___boxed(lean_object* v_m_292_, lean_object* v_motive_293_, lean_object* v_base_294_, lean_object* v_pred_295_, lean_object* v_n_296_, lean_object* v_hnm_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_mathlib_Int_le__induction__down(v_m_292_, v_motive_293_, v_base_294_, v_pred_295_, v_n_296_, v_hnm_297_);
lean_dec(v_n_296_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__0(lean_object* v_ih_299_, lean_object* v_k_300_, lean_object* v_x_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lean_apply_2(v_ih_299_, v_k_300_, lean_box(0));
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__1(lean_object* v_m_303_, lean_object* v_ge_304_, lean_object* v_lt_305_, lean_object* v___n_306_, lean_object* v_a_307_, lean_object* v_ih_308_, lean_object* v_l_309_, lean_object* v_a_310_){
_start:
{
uint8_t v___x_311_; 
v___x_311_ = lean_int_dec_lt(v_l_309_, v_m_303_);
if (v___x_311_ == 0)
{
lean_object* v___f_312_; lean_object* v___x_313_; 
lean_dec(v_lt_305_);
v___f_312_ = lean_alloc_closure((void*)(lp_mathlib_Int_strongRec___redArg___lam__0), 3, 1);
lean_closure_set(v___f_312_, 0, v_ih_308_);
v___x_313_ = lean_apply_3(v_ge_304_, v_l_309_, lean_box(0), v___f_312_);
return v___x_313_;
}
else
{
lean_object* v___x_314_; 
lean_dec(v_ih_308_);
lean_dec(v_ge_304_);
v___x_314_ = lean_apply_2(v_lt_305_, v_l_309_, lean_box(0));
return v___x_314_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__1___boxed(lean_object* v_m_315_, lean_object* v_ge_316_, lean_object* v_lt_317_, lean_object* v___n_318_, lean_object* v_a_319_, lean_object* v_ih_320_, lean_object* v_l_321_, lean_object* v_a_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_Int_strongRec___redArg___lam__1(v_m_315_, v_ge_316_, v_lt_317_, v___n_318_, v_a_319_, v_ih_320_, v_l_321_, v_a_322_);
lean_dec(v___n_318_);
lean_dec(v_m_315_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__2(lean_object* v_n_324_, lean_object* v_x_325_, lean_object* v_hn_326_, lean_object* v_l_327_, lean_object* v_x_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lean_apply_2(v_hn_326_, v_l_327_, lean_box(0));
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg___lam__2___boxed(lean_object* v_n_330_, lean_object* v_x_331_, lean_object* v_hn_332_, lean_object* v_l_333_, lean_object* v_x_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_Int_strongRec___redArg___lam__2(v_n_330_, v_x_331_, v_hn_332_, v_l_333_, v_x_334_);
lean_dec(v_n_330_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec___redArg(lean_object* v_m_337_, lean_object* v_lt_338_, lean_object* v_ge_339_, lean_object* v_n_340_){
_start:
{
uint8_t v___x_341_; 
v___x_341_ = lean_int_dec_lt(v_n_340_, v_m_337_);
if (v___x_341_ == 0)
{
lean_object* v___f_342_; lean_object* v___f_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
lean_inc(v_lt_338_);
lean_inc(v_ge_339_);
lean_inc(v_m_337_);
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_Int_strongRec___redArg___lam__1___boxed), 8, 3);
lean_closure_set(v___f_342_, 0, v_m_337_);
lean_closure_set(v___f_342_, 1, v_ge_339_);
lean_closure_set(v___f_342_, 2, v_lt_338_);
v___f_343_ = ((lean_object*)(lp_mathlib_Int_strongRec___redArg___closed__0));
v___x_344_ = lp_mathlib_Int_inductionOn_x27___redArg(v_n_340_, v_m_337_, v_lt_338_, v___f_342_, v___f_343_);
v___x_345_ = lean_apply_3(v_ge_339_, v_n_340_, lean_box(0), v___x_344_);
return v___x_345_;
}
else
{
lean_object* v___x_346_; 
lean_dec(v_ge_339_);
lean_dec(v_m_337_);
v___x_346_ = lean_apply_2(v_lt_338_, v_n_340_, lean_box(0));
return v___x_346_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongRec(lean_object* v_m_347_, lean_object* v_motive_348_, lean_object* v_lt_349_, lean_object* v_ge_350_, lean_object* v_n_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lp_mathlib_Int_strongRec___redArg(v_m_347_, v_lt_349_, v_ge_350_, v_n_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_natMod(lean_object* v_m_353_, lean_object* v_n_354_){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_355_ = lean_int_emod(v_m_353_, v_n_354_);
v___x_356_ = l_Int_toNat(v___x_355_);
lean_dec(v___x_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_natMod___boxed(lean_object* v_m_357_, lean_object* v_n_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_mathlib_Int_natMod(v_m_357_, v_n_358_);
lean_dec(v_n_358_);
lean_dec(v_m_357_);
return v_res_359_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DepRewrite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DepRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Logic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DepRewrite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DepRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Int_Init(builtin);
}
#ifdef __cplusplus
}
#endif
