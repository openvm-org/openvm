// Lean compiler output
// Module: Mathlib.Data.Nat.Init
// Imports: public import Init public meta import Init public import Batteries.Data.Nat.Lemmas public import Batteries.Util.LibraryNote public import Mathlib.Data.Int.Notation public import Mathlib.Data.Nat.Notation
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
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_batteries_Nat_strongRec___redArg(lean_object*, lean_object*);
uint8_t l_Nat_decidableBallLTTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_foundational__algebra__order__theory;
LEAN_EXPORT lean_object* lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongRec_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongRec_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongRecOn_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongRecOn_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_twoStepInduction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_twoStepInduction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_twoStepInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_twoStepInduction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_stepInduction___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_stepInduction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_stepInduction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_stepInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_decreasingInduction___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_decreasingInduction___redArg___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_decreasingInduction___redArg___closed__0 = (const lean_object*)&lp_mathlib_Nat_decreasingInduction___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Nat_decreasingInduction___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_decreasingInduction___redArg___lam__2, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_decreasingInduction___redArg___closed__1 = (const lean_object*)&lp_mathlib_Nat_decreasingInduction___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongSubRecursion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongSubRecursion___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongSubRecursion(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_strongSubRecursion_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_strongSubRecursion_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pincerRecursion___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pincerRecursion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_pincerRecursion_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_pincerRecursion_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_decreasingInduction_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___closed__0 = (const lean_object*)&lp_mathlib_Nat_decreasingInduction_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHi___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHi___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHi___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHi___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHiLe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHiLe___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHiLe(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHiLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_LibraryNote_foundational__algebra__order__theory(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___redArg(lean_object* v_n_2_, lean_object* v_n_3_, lean_object* v_h_u2081_4_, lean_object* v_h_u2082_5_){
_start:
{
uint8_t v___x_6_; 
v___x_6_ = lean_nat_dec_le(v_n_2_, v_n_3_);
if (v___x_6_ == 0)
{
lean_object* v___x_7_; 
lean_dec(v_h_u2081_4_);
v___x_7_ = lean_apply_1(v_h_u2082_5_, lean_box(0));
return v___x_7_;
}
else
{
lean_object* v___x_8_; 
lean_dec(v_h_u2082_5_);
v___x_8_ = lean_apply_1(v_h_u2081_4_, lean_box(0));
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___redArg___boxed(lean_object* v_n_9_, lean_object* v_n_10_, lean_object* v_h_u2081_11_, lean_object* v_h_u2082_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___redArg(v_n_9_, v_n_10_, v_h_u2081_11_, v_h_u2082_12_);
lean_dec(v_n_10_);
lean_dec(v_n_9_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0(lean_object* v_n_14_, lean_object* v_n_15_, lean_object* v_p_16_, lean_object* v_q_17_, lean_object* v_00_u03b1_18_, lean_object* v_h_19_, lean_object* v_h_u2081_20_, lean_object* v_h_u2082_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___redArg(v_n_14_, v_n_15_, v_h_u2081_20_, v_h_u2082_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___boxed(lean_object* v_n_23_, lean_object* v_n_24_, lean_object* v_p_25_, lean_object* v_q_26_, lean_object* v_00_u03b1_27_, lean_object* v_h_28_, lean_object* v_h_u2081_29_, lean_object* v_h_u2082_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0(v_n_23_, v_n_24_, v_p_25_, v_q_26_, v_00_u03b1_27_, v_h_28_, v_h_u2081_29_, v_h_u2082_30_);
lean_dec(v_n_24_);
lean_dec(v_n_23_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg___lam__0(lean_object* v_refl_32_, lean_object* v_h_33_){
_start:
{
lean_inc(v_refl_32_);
return v_refl_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg___lam__0___boxed(lean_object* v_refl_34_, lean_object* v_h_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Nat_leRec___redArg___lam__0(v_refl_34_, v_h_35_);
lean_dec(v_refl_34_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg(lean_object* v_n_37_, lean_object* v_refl_38_, lean_object* v_le__succ__of__le_39_, lean_object* v_x_40_){
_start:
{
lean_object* v_zero_41_; uint8_t v_isZero_42_; 
v_zero_41_ = lean_unsigned_to_nat(0u);
v_isZero_42_ = lean_nat_dec_eq(v_x_40_, v_zero_41_);
if (v_isZero_42_ == 1)
{
lean_dec(v_le__succ__of__le_39_);
lean_dec(v_n_37_);
return v_refl_38_;
}
else
{
lean_object* v___f_43_; lean_object* v_one_44_; lean_object* v_n_45_; lean_object* v___f_46_; lean_object* v___x_47_; 
lean_inc(v_refl_38_);
v___f_43_ = lean_alloc_closure((void*)(lp_mathlib_Nat_leRec___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_43_, 0, v_refl_38_);
v_one_44_ = lean_unsigned_to_nat(1u);
v_n_45_ = lean_nat_sub(v_x_40_, v_one_44_);
lean_inc(v_n_45_);
lean_inc(v_n_37_);
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_Nat_leRec___redArg___lam__1), 5, 4);
lean_closure_set(v___f_46_, 0, v_n_37_);
lean_closure_set(v___f_46_, 1, v_refl_38_);
lean_closure_set(v___f_46_, 2, v_le__succ__of__le_39_);
lean_closure_set(v___f_46_, 3, v_n_45_);
v___x_47_ = lp_mathlib_Or_by__cases___at___00Nat_leRec_spec__0___redArg(v_n_37_, v_n_45_, v___f_46_, v___f_43_);
lean_dec(v_n_45_);
lean_dec(v_n_37_);
return v___x_47_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg___lam__1(lean_object* v_n_48_, lean_object* v_refl_49_, lean_object* v_le__succ__of__le_50_, lean_object* v_n_51_, lean_object* v_h_52_){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
lean_inc(v_le__succ__of__le_50_);
v___x_53_ = lp_mathlib_Nat_leRec___redArg(v_n_48_, v_refl_49_, v_le__succ__of__le_50_, v_n_51_);
v___x_54_ = lean_apply_3(v_le__succ__of__le_50_, v_n_51_, lean_box(0), v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___redArg___boxed(lean_object* v_n_55_, lean_object* v_refl_56_, lean_object* v_le__succ__of__le_57_, lean_object* v_x_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_Nat_leRec___redArg(v_n_55_, v_refl_56_, v_le__succ__of__le_57_, v_x_58_);
lean_dec(v_x_58_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec(lean_object* v_n_60_, lean_object* v_motive_61_, lean_object* v_refl_62_, lean_object* v_le__succ__of__le_63_, lean_object* v_x_64_, lean_object* v_x_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Nat_leRec___redArg(v_n_60_, v_refl_62_, v_le__succ__of__le_63_, v_x_64_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRec___boxed(lean_object* v_n_67_, lean_object* v_motive_68_, lean_object* v_refl_69_, lean_object* v_le__succ__of__le_70_, lean_object* v_x_71_, lean_object* v_x_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_Nat_leRec(v_n_67_, v_motive_68_, v_refl_69_, v_le__succ__of__le_70_, v_x_71_, v_x_72_);
lean_dec(v_x_71_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter___redArg(lean_object* v_x_74_, lean_object* v_h__1_75_, lean_object* v_h__2_76_){
_start:
{
lean_object* v_zero_77_; uint8_t v_isZero_78_; 
v_zero_77_ = lean_unsigned_to_nat(0u);
v_isZero_78_ = lean_nat_dec_eq(v_x_74_, v_zero_77_);
if (v_isZero_78_ == 1)
{
lean_object* v___x_79_; 
lean_dec(v_h__2_76_);
v___x_79_ = lean_apply_1(v_h__1_75_, lean_box(0));
return v___x_79_;
}
else
{
lean_object* v_one_80_; lean_object* v_n_81_; lean_object* v___x_82_; 
lean_dec(v_h__1_75_);
v_one_80_ = lean_unsigned_to_nat(1u);
v_n_81_ = lean_nat_sub(v_x_74_, v_one_80_);
v___x_82_ = lean_apply_2(v_h__2_76_, v_n_81_, lean_box(0));
return v___x_82_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter___redArg___boxed(lean_object* v_x_83_, lean_object* v_h__1_84_, lean_object* v_h__2_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter___redArg(v_x_83_, v_h__1_84_, v_h__2_85_);
lean_dec(v_x_83_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter(lean_object* v_n_87_, lean_object* v_motive_88_, lean_object* v_x_89_, lean_object* v_x_90_, lean_object* v_h__1_91_, lean_object* v_h__2_92_){
_start:
{
lean_object* v_zero_93_; uint8_t v_isZero_94_; 
v_zero_93_ = lean_unsigned_to_nat(0u);
v_isZero_94_ = lean_nat_dec_eq(v_x_89_, v_zero_93_);
if (v_isZero_94_ == 1)
{
lean_object* v___x_95_; 
lean_dec(v_h__2_92_);
v___x_95_ = lean_apply_1(v_h__1_91_, lean_box(0));
return v___x_95_;
}
else
{
lean_object* v_one_96_; lean_object* v_n_97_; lean_object* v___x_98_; 
lean_dec(v_h__1_91_);
v_one_96_ = lean_unsigned_to_nat(1u);
v_n_97_ = lean_nat_sub(v_x_89_, v_one_96_);
v___x_98_ = lean_apply_2(v_h__2_92_, v_n_97_, lean_box(0));
return v___x_98_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter___boxed(lean_object* v_n_99_, lean_object* v_motive_100_, lean_object* v_x_101_, lean_object* v_x_102_, lean_object* v_h__1_103_, lean_object* v_h__2_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_leRec_match__1_splitter(v_n_99_, v_motive_100_, v_x_101_, v_x_102_, v_h__1_103_, v_h__2_104_);
lean_dec(v_x_101_);
lean_dec(v_n_99_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn___redArg___lam__0(lean_object* v_of__succ_106_, lean_object* v_x_107_, lean_object* v_x_108_, lean_object* v___y_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_apply_2(v_of__succ_106_, v_x_107_, v___y_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn___redArg(lean_object* v_n_111_, lean_object* v_m_112_, lean_object* v_of__succ_113_, lean_object* v_self_114_){
_start:
{
lean_object* v___f_115_; lean_object* v___x_116_; 
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_Nat_leRecOn___redArg___lam__0), 4, 1);
lean_closure_set(v___f_115_, 0, v_of__succ_113_);
v___x_116_ = lp_mathlib_Nat_leRec___redArg(v_n_111_, v_self_114_, v___f_115_, v_m_112_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn___redArg___boxed(lean_object* v_n_117_, lean_object* v_m_118_, lean_object* v_of__succ_119_, lean_object* v_self_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_Nat_leRecOn___redArg(v_n_117_, v_m_118_, v_of__succ_119_, v_self_120_);
lean_dec(v_m_118_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn(lean_object* v_C_122_, lean_object* v_n_123_, lean_object* v_m_124_, lean_object* v_h_125_, lean_object* v_of__succ_126_, lean_object* v_self_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_Nat_leRecOn___redArg(v_n_123_, v_m_124_, v_of__succ_126_, v_self_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_leRecOn___boxed(lean_object* v_C_129_, lean_object* v_n_130_, lean_object* v_m_131_, lean_object* v_h_132_, lean_object* v_of__succ_133_, lean_object* v_self_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_Nat_leRecOn(v_C_129_, v_n_130_, v_m_131_, v_h_132_, v_of__succ_133_, v_self_134_);
lean_dec(v_m_131_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongRec_x27___redArg(lean_object* v_ind_136_, lean_object* v_t_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_batteries_Nat_strongRec___redArg(v_ind_136_, v_t_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongRec_x27(lean_object* v_motive_139_, lean_object* v_ind_140_, lean_object* v_t_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_batteries_Nat_strongRec___redArg(v_ind_140_, v_t_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongRecOn_x27___redArg(lean_object* v_ind_143_, lean_object* v_t_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_batteries_Nat_strongRec___redArg(v_ind_143_, v_t_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongRecOn_x27(lean_object* v_motive_146_, lean_object* v_ind_147_, lean_object* v_t_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_batteries_Nat_strongRec___redArg(v_ind_147_, v_t_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_twoStepInduction___redArg(lean_object* v_zero_150_, lean_object* v_one_151_, lean_object* v_more_152_, lean_object* v_x_153_){
_start:
{
lean_object* v_zero_154_; uint8_t v_isZero_155_; 
v_zero_154_ = lean_unsigned_to_nat(0u);
v_isZero_155_ = lean_nat_dec_eq(v_x_153_, v_zero_154_);
if (v_isZero_155_ == 1)
{
lean_dec(v_more_152_);
lean_inc(v_zero_150_);
return v_zero_150_;
}
else
{
lean_object* v_one_156_; lean_object* v_n_157_; uint8_t v_isZero_158_; 
v_one_156_ = lean_unsigned_to_nat(1u);
v_n_157_ = lean_nat_sub(v_x_153_, v_one_156_);
v_isZero_158_ = lean_nat_dec_eq(v_n_157_, v_zero_154_);
if (v_isZero_158_ == 1)
{
lean_dec(v_n_157_);
lean_dec(v_more_152_);
lean_inc(v_one_151_);
return v_one_151_;
}
else
{
lean_object* v_n_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v_n_159_ = lean_nat_sub(v_n_157_, v_one_156_);
lean_dec(v_n_157_);
lean_inc_n(v_more_152_, 2);
v___x_160_ = lp_mathlib_Nat_twoStepInduction___redArg(v_zero_150_, v_one_151_, v_more_152_, v_n_159_);
v___x_161_ = lean_nat_add(v_n_159_, v_one_156_);
v___x_162_ = lp_mathlib_Nat_twoStepInduction___redArg(v_zero_150_, v_one_151_, v_more_152_, v___x_161_);
lean_dec(v___x_161_);
v___x_163_ = lean_apply_3(v_more_152_, v_n_159_, v___x_160_, v___x_162_);
return v___x_163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_twoStepInduction___redArg___boxed(lean_object* v_zero_164_, lean_object* v_one_165_, lean_object* v_more_166_, lean_object* v_x_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Nat_twoStepInduction___redArg(v_zero_164_, v_one_165_, v_more_166_, v_x_167_);
lean_dec(v_x_167_);
lean_dec(v_one_165_);
lean_dec(v_zero_164_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_twoStepInduction(lean_object* v_motive_169_, lean_object* v_zero_170_, lean_object* v_one_171_, lean_object* v_more_172_, lean_object* v_x_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_Nat_twoStepInduction___redArg(v_zero_170_, v_one_171_, v_more_172_, v_x_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_twoStepInduction___boxed(lean_object* v_motive_175_, lean_object* v_zero_176_, lean_object* v_one_177_, lean_object* v_more_178_, lean_object* v_x_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Nat_twoStepInduction(v_motive_175_, v_zero_176_, v_one_177_, v_more_178_, v_x_179_);
lean_dec(v_x_179_);
lean_dec(v_one_177_);
lean_dec(v_zero_176_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_stepInduction___redArg___lam__0___boxed(lean_object* v___x_181_, lean_object* v_k_182_, lean_object* v_base_183_, lean_object* v_step_184_, lean_object* v_x_185_, lean_object* v_x_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_Nat_stepInduction___redArg___lam__0(v___x_181_, v_k_182_, v_base_183_, v_step_184_, v_x_185_, v_x_186_);
lean_dec(v_x_185_);
lean_dec(v___x_181_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_stepInduction___redArg(lean_object* v_k_188_, lean_object* v_base_189_, lean_object* v_step_190_, lean_object* v_a_191_){
_start:
{
uint8_t v___x_192_; 
v___x_192_ = lean_nat_dec_lt(v_a_191_, v_k_188_);
if (v___x_192_ == 0)
{
lean_object* v___x_193_; lean_object* v___f_194_; lean_object* v___x_195_; 
v___x_193_ = lean_nat_sub(v_a_191_, v_k_188_);
lean_dec(v_a_191_);
lean_inc(v_step_190_);
lean_inc(v___x_193_);
v___f_194_ = lean_alloc_closure((void*)(lp_mathlib_Nat_stepInduction___redArg___lam__0___boxed), 6, 4);
lean_closure_set(v___f_194_, 0, v___x_193_);
lean_closure_set(v___f_194_, 1, v_k_188_);
lean_closure_set(v___f_194_, 2, v_base_189_);
lean_closure_set(v___f_194_, 3, v_step_190_);
v___x_195_ = lean_apply_2(v_step_190_, v___x_193_, v___f_194_);
return v___x_195_;
}
else
{
lean_object* v___x_196_; 
lean_dec(v_step_190_);
lean_dec(v_k_188_);
v___x_196_ = lean_apply_2(v_base_189_, v_a_191_, lean_box(0));
return v___x_196_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_stepInduction___redArg___lam__0(lean_object* v___x_197_, lean_object* v_k_198_, lean_object* v_base_199_, lean_object* v_step_200_, lean_object* v_x_201_, lean_object* v_x_202_){
_start:
{
lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_203_ = lean_nat_add(v___x_197_, v_x_201_);
v___x_204_ = lp_mathlib_Nat_stepInduction___redArg(v_k_198_, v_base_199_, v_step_200_, v___x_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_stepInduction(lean_object* v_motive_205_, lean_object* v_k_206_, lean_object* v_base_207_, lean_object* v_step_208_, lean_object* v_a_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_Nat_stepInduction___redArg(v_k_206_, v_base_207_, v_step_208_, v_a_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___lam__0(lean_object* v_motive_211_, lean_object* v_of__succ_212_, lean_object* v_self_213_){
_start:
{
lean_inc(v_self_213_);
return v_self_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___lam__0___boxed(lean_object* v_motive_214_, lean_object* v_of__succ_215_, lean_object* v_self_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Nat_decreasingInduction___redArg___lam__0(v_motive_214_, v_of__succ_215_, v_self_216_);
lean_dec(v_self_216_);
lean_dec(v_of__succ_215_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___lam__1(lean_object* v_of__succ_218_, lean_object* v_i_219_, lean_object* v_hi_220_, lean_object* v___y_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lean_apply_3(v_of__succ_218_, v_i_219_, lean_box(0), v___y_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___lam__2(lean_object* v_k_223_, lean_object* v_h_224_, lean_object* v_ih_225_, lean_object* v_motive_226_, lean_object* v_of__succ_227_, lean_object* v_self_228_){
_start:
{
lean_object* v___f_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
lean_inc(v_of__succ_227_);
v___f_229_ = lean_alloc_closure((void*)(lp_mathlib_Nat_decreasingInduction___redArg___lam__1), 4, 1);
lean_closure_set(v___f_229_, 0, v_of__succ_227_);
v___x_230_ = lean_apply_3(v_of__succ_227_, v_k_223_, lean_box(0), v_self_228_);
v___x_231_ = lean_apply_3(v_ih_225_, lean_box(0), v___f_229_, v___x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg(lean_object* v_n_234_, lean_object* v_of__succ_235_, lean_object* v_self_236_, lean_object* v_m_237_){
_start:
{
lean_object* v___f_238_; lean_object* v___f_239_; lean_object* v___x_8__overap_240_; lean_object* v___x_241_; 
v___f_238_ = ((lean_object*)(lp_mathlib_Nat_decreasingInduction___redArg___closed__0));
v___f_239_ = ((lean_object*)(lp_mathlib_Nat_decreasingInduction___redArg___closed__1));
v___x_8__overap_240_ = lp_mathlib_Nat_leRec___redArg(v_m_237_, v___f_238_, v___f_239_, v_n_234_);
v___x_241_ = lean_apply_3(v___x_8__overap_240_, lean_box(0), v_of__succ_235_, v_self_236_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___redArg___boxed(lean_object* v_n_242_, lean_object* v_of__succ_243_, lean_object* v_self_244_, lean_object* v_m_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_Nat_decreasingInduction___redArg(v_n_242_, v_of__succ_243_, v_self_244_, v_m_245_);
lean_dec(v_n_242_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction(lean_object* v_n_247_, lean_object* v_motive_248_, lean_object* v_of__succ_249_, lean_object* v_self_250_, lean_object* v_m_251_, lean_object* v_mn_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_Nat_decreasingInduction___redArg(v_n_247_, v_of__succ_249_, v_self_250_, v_m_251_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction___boxed(lean_object* v_n_254_, lean_object* v_motive_255_, lean_object* v_of__succ_256_, lean_object* v_self_257_, lean_object* v_m_258_, lean_object* v_mn_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_Nat_decreasingInduction(v_n_254_, v_motive_255_, v_of__succ_256_, v_self_257_, v_m_258_, v_mn_259_);
lean_dec(v_n_254_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongSubRecursion___redArg(lean_object* v_H_261_, lean_object* v_x_262_, lean_object* v_x_263_){
_start:
{
lean_object* v___f_264_; lean_object* v___x_265_; 
lean_inc(v_H_261_);
v___f_264_ = lean_alloc_closure((void*)(lp_mathlib_Nat_strongSubRecursion___redArg___lam__0), 5, 1);
lean_closure_set(v___f_264_, 0, v_H_261_);
v___x_265_ = lean_apply_3(v_H_261_, v_x_262_, v_x_263_, v___f_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongSubRecursion___redArg___lam__0(lean_object* v_H_266_, lean_object* v_x_267_, lean_object* v_y_268_, lean_object* v_x_269_, lean_object* v_x_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_mathlib_Nat_strongSubRecursion___redArg(v_H_266_, v_x_267_, v_y_268_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_strongSubRecursion(lean_object* v_P_272_, lean_object* v_H_273_, lean_object* v_x_274_, lean_object* v_x_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_mathlib_Nat_strongSubRecursion___redArg(v_H_273_, v_x_274_, v_x_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_strongSubRecursion_match__1_splitter___redArg(lean_object* v_x_277_, lean_object* v_x_278_, lean_object* v_h__1_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lean_apply_2(v_h__1_279_, v_x_277_, v_x_278_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_strongSubRecursion_match__1_splitter(lean_object* v_motive_281_, lean_object* v_x_282_, lean_object* v_x_283_, lean_object* v_h__1_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lean_apply_2(v_h__1_284_, v_x_282_, v_x_283_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_pincerRecursion___redArg(lean_object* v_Ha0_286_, lean_object* v_H0b_287_, lean_object* v_H_288_, lean_object* v_x_289_, lean_object* v_x_290_){
_start:
{
lean_object* v_zero_291_; uint8_t v_isZero_292_; 
v_zero_291_ = lean_unsigned_to_nat(0u);
v_isZero_292_ = lean_nat_dec_eq(v_x_290_, v_zero_291_);
if (v_isZero_292_ == 1)
{
lean_object* v___x_293_; 
lean_dec(v_x_290_);
lean_dec(v_H_288_);
lean_dec(v_H0b_287_);
v___x_293_ = lean_apply_1(v_Ha0_286_, v_x_289_);
return v___x_293_;
}
else
{
uint8_t v_isZero_294_; 
v_isZero_294_ = lean_nat_dec_eq(v_x_289_, v_zero_291_);
if (v_isZero_294_ == 1)
{
lean_object* v___x_295_; 
lean_dec(v_x_289_);
lean_dec(v_H_288_);
lean_dec(v_Ha0_286_);
v___x_295_ = lean_apply_1(v_H0b_287_, v_x_290_);
return v___x_295_;
}
else
{
lean_object* v_one_296_; lean_object* v_n_297_; lean_object* v_n_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v_one_296_ = lean_unsigned_to_nat(1u);
v_n_297_ = lean_nat_sub(v_x_290_, v_one_296_);
v_n_298_ = lean_nat_sub(v_x_289_, v_one_296_);
lean_inc(v_n_298_);
lean_inc_n(v_H_288_, 2);
lean_inc(v_H0b_287_);
lean_inc(v_Ha0_286_);
v___x_299_ = lp_mathlib_Nat_pincerRecursion___redArg(v_Ha0_286_, v_H0b_287_, v_H_288_, v_n_298_, v_x_290_);
lean_inc(v_n_297_);
v___x_300_ = lp_mathlib_Nat_pincerRecursion___redArg(v_Ha0_286_, v_H0b_287_, v_H_288_, v_x_289_, v_n_297_);
v___x_301_ = lean_apply_4(v_H_288_, v_n_298_, v_n_297_, v___x_299_, v___x_300_);
return v___x_301_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_pincerRecursion(lean_object* v_P_302_, lean_object* v_Ha0_303_, lean_object* v_H0b_304_, lean_object* v_H_305_, lean_object* v_x_306_, lean_object* v_x_307_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = lp_mathlib_Nat_pincerRecursion___redArg(v_Ha0_303_, v_H0b_304_, v_H_305_, v_x_306_, v_x_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_pincerRecursion_match__1_splitter___redArg(lean_object* v_x_309_, lean_object* v_x_310_, lean_object* v_h__1_311_, lean_object* v_h__2_312_, lean_object* v_h__3_313_){
_start:
{
lean_object* v_zero_314_; uint8_t v_isZero_315_; 
v_zero_314_ = lean_unsigned_to_nat(0u);
v_isZero_315_ = lean_nat_dec_eq(v_x_310_, v_zero_314_);
if (v_isZero_315_ == 1)
{
lean_object* v___x_316_; 
lean_dec(v_h__3_313_);
lean_dec(v_h__2_312_);
lean_dec(v_x_310_);
v___x_316_ = lean_apply_1(v_h__1_311_, v_x_309_);
return v___x_316_;
}
else
{
uint8_t v_isZero_317_; 
lean_dec(v_h__1_311_);
v_isZero_317_ = lean_nat_dec_eq(v_x_309_, v_zero_314_);
if (v_isZero_317_ == 1)
{
lean_object* v___x_318_; 
lean_dec(v_h__3_313_);
lean_dec(v_x_309_);
v___x_318_ = lean_apply_2(v_h__2_312_, v_x_310_, lean_box(0));
return v___x_318_;
}
else
{
lean_object* v_one_319_; lean_object* v_n_320_; lean_object* v_n_321_; lean_object* v___x_322_; 
lean_dec(v_h__2_312_);
v_one_319_ = lean_unsigned_to_nat(1u);
v_n_320_ = lean_nat_sub(v_x_310_, v_one_319_);
lean_dec(v_x_310_);
v_n_321_ = lean_nat_sub(v_x_309_, v_one_319_);
lean_dec(v_x_309_);
v___x_322_ = lean_apply_2(v_h__3_313_, v_n_321_, v_n_320_);
return v___x_322_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Init_0__Nat_pincerRecursion_match__1_splitter(lean_object* v_motive_323_, lean_object* v_x_324_, lean_object* v_x_325_, lean_object* v_h__1_326_, lean_object* v_h__2_327_, lean_object* v_h__3_328_){
_start:
{
lean_object* v_zero_329_; uint8_t v_isZero_330_; 
v_zero_329_ = lean_unsigned_to_nat(0u);
v_isZero_330_ = lean_nat_dec_eq(v_x_325_, v_zero_329_);
if (v_isZero_330_ == 1)
{
lean_object* v___x_331_; 
lean_dec(v_h__3_328_);
lean_dec(v_h__2_327_);
lean_dec(v_x_325_);
v___x_331_ = lean_apply_1(v_h__1_326_, v_x_324_);
return v___x_331_;
}
else
{
uint8_t v_isZero_332_; 
lean_dec(v_h__1_326_);
v_isZero_332_ = lean_nat_dec_eq(v_x_324_, v_zero_329_);
if (v_isZero_332_ == 1)
{
lean_object* v___x_333_; 
lean_dec(v_h__3_328_);
lean_dec(v_x_324_);
v___x_333_ = lean_apply_2(v_h__2_327_, v_x_325_, lean_box(0));
return v___x_333_;
}
else
{
lean_object* v_one_334_; lean_object* v_n_335_; lean_object* v_n_336_; lean_object* v___x_337_; 
lean_dec(v_h__2_327_);
v_one_334_ = lean_unsigned_to_nat(1u);
v_n_335_ = lean_nat_sub(v_x_325_, v_one_334_);
lean_dec(v_x_325_);
v_n_336_ = lean_nat_sub(v_x_324_, v_one_334_);
lean_dec(v_x_324_);
v___x_337_ = lean_apply_2(v_h__3_328_, v_n_336_, v_n_335_);
return v___x_337_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__0(lean_object* v_h_338_, lean_object* v_k_x27_339_, lean_object* v_hk_x27_340_, lean_object* v_h_x27_x27_341_, lean_object* v___y_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lean_apply_4(v_h_338_, v_k_x27_339_, lean_box(0), lean_box(0), v___y_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__1(lean_object* v_k_344_, lean_object* v_hk_345_, lean_object* v_ih_346_, lean_object* v_h_347_){
_start:
{
lean_object* v___f_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
lean_inc(v_h_347_);
v___f_348_ = lean_alloc_closure((void*)(lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__0), 5, 1);
lean_closure_set(v___f_348_, 0, v_h_347_);
v___x_349_ = lean_apply_1(v_ih_346_, v___f_348_);
v___x_350_ = lean_apply_4(v_h_347_, v_k_344_, lean_box(0), lean_box(0), v___x_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__2(lean_object* v_hP_351_, lean_object* v_h_352_){
_start:
{
lean_inc(v_hP_351_);
return v_hP_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__2___boxed(lean_object* v_hP_353_, lean_object* v_h_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__2(v_hP_353_, v_h_354_);
lean_dec(v_h_354_);
lean_dec(v_hP_353_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg(lean_object* v_m_357_, lean_object* v_n_358_, lean_object* v_h_359_, lean_object* v_hP_360_){
_start:
{
lean_object* v___f_361_; lean_object* v___f_362_; lean_object* v___x_7__overap_363_; lean_object* v___x_364_; 
v___f_361_ = ((lean_object*)(lp_mathlib_Nat_decreasingInduction_x27___redArg___closed__0));
v___f_362_ = lean_alloc_closure((void*)(lp_mathlib_Nat_decreasingInduction_x27___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_362_, 0, v_hP_360_);
v___x_7__overap_363_ = lp_mathlib_Nat_decreasingInduction___redArg(v_n_358_, v___f_361_, v___f_362_, v_m_357_);
v___x_364_ = lean_apply_1(v___x_7__overap_363_, v_h_359_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___redArg___boxed(lean_object* v_m_365_, lean_object* v_n_366_, lean_object* v_h_367_, lean_object* v_hP_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_mathlib_Nat_decreasingInduction_x27___redArg(v_m_365_, v_n_366_, v_h_367_, v_hP_368_);
lean_dec(v_n_366_);
return v_res_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27(lean_object* v_m_370_, lean_object* v_n_371_, lean_object* v_P_372_, lean_object* v_h_373_, lean_object* v_mn_374_, lean_object* v_hP_375_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lp_mathlib_Nat_decreasingInduction_x27___redArg(v_m_370_, v_n_371_, v_h_373_, v_hP_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decreasingInduction_x27___boxed(lean_object* v_m_377_, lean_object* v_n_378_, lean_object* v_P_379_, lean_object* v_h_380_, lean_object* v_mn_381_, lean_object* v_hP_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_Nat_decreasingInduction_x27(v_m_377_, v_n_378_, v_P_379_, v_h_380_, v_mn_381_, v_hP_382_);
lean_dec(v_n_378_);
return v_res_383_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHi___redArg___lam__0(lean_object* v_lo_384_, lean_object* v_inst_385_, lean_object* v_n_386_, lean_object* v_h_387_){
_start:
{
lean_object* v___x_388_; lean_object* v___x_389_; uint8_t v___x_390_; 
v___x_388_ = lean_nat_add(v_lo_384_, v_n_386_);
v___x_389_ = lean_apply_1(v_inst_385_, v___x_388_);
v___x_390_ = lean_unbox(v___x_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHi___redArg___lam__0___boxed(lean_object* v_lo_391_, lean_object* v_inst_392_, lean_object* v_n_393_, lean_object* v_h_394_){
_start:
{
uint8_t v_res_395_; lean_object* v_r_396_; 
v_res_395_ = lp_mathlib_Nat_decidableLoHi___redArg___lam__0(v_lo_391_, v_inst_392_, v_n_393_, v_h_394_);
lean_dec(v_n_393_);
lean_dec(v_lo_391_);
v_r_396_ = lean_box(v_res_395_);
return v_r_396_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHi___redArg(lean_object* v_lo_397_, lean_object* v_hi_398_, lean_object* v_inst_399_){
_start:
{
lean_object* v___f_400_; lean_object* v___x_401_; uint8_t v___x_402_; 
lean_inc(v_lo_397_);
v___f_400_ = lean_alloc_closure((void*)(lp_mathlib_Nat_decidableLoHi___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_400_, 0, v_lo_397_);
lean_closure_set(v___f_400_, 1, v_inst_399_);
v___x_401_ = lean_nat_sub(v_hi_398_, v_lo_397_);
lean_dec(v_lo_397_);
v___x_402_ = l_Nat_decidableBallLTTR___redArg(v___x_401_, v___f_400_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHi___redArg___boxed(lean_object* v_lo_403_, lean_object* v_hi_404_, lean_object* v_inst_405_){
_start:
{
uint8_t v_res_406_; lean_object* v_r_407_; 
v_res_406_ = lp_mathlib_Nat_decidableLoHi___redArg(v_lo_403_, v_hi_404_, v_inst_405_);
lean_dec(v_hi_404_);
v_r_407_ = lean_box(v_res_406_);
return v_r_407_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHi(lean_object* v_lo_408_, lean_object* v_hi_409_, lean_object* v_P_410_, lean_object* v_inst_411_){
_start:
{
uint8_t v___x_412_; 
v___x_412_ = lp_mathlib_Nat_decidableLoHi___redArg(v_lo_408_, v_hi_409_, v_inst_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHi___boxed(lean_object* v_lo_413_, lean_object* v_hi_414_, lean_object* v_P_415_, lean_object* v_inst_416_){
_start:
{
uint8_t v_res_417_; lean_object* v_r_418_; 
v_res_417_ = lp_mathlib_Nat_decidableLoHi(v_lo_413_, v_hi_414_, v_P_415_, v_inst_416_);
lean_dec(v_hi_414_);
v_r_418_ = lean_box(v_res_417_);
return v_r_418_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHiLe___redArg(lean_object* v_lo_419_, lean_object* v_hi_420_, lean_object* v_inst_421_){
_start:
{
lean_object* v___x_422_; lean_object* v___x_423_; uint8_t v___x_424_; 
v___x_422_ = lean_unsigned_to_nat(1u);
v___x_423_ = lean_nat_add(v_hi_420_, v___x_422_);
v___x_424_ = lp_mathlib_Nat_decidableLoHi___redArg(v_lo_419_, v___x_423_, v_inst_421_);
lean_dec(v___x_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHiLe___redArg___boxed(lean_object* v_lo_425_, lean_object* v_hi_426_, lean_object* v_inst_427_){
_start:
{
uint8_t v_res_428_; lean_object* v_r_429_; 
v_res_428_ = lp_mathlib_Nat_decidableLoHiLe___redArg(v_lo_425_, v_hi_426_, v_inst_427_);
lean_dec(v_hi_426_);
v_r_429_ = lean_box(v_res_428_);
return v_r_429_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidableLoHiLe(lean_object* v_lo_430_, lean_object* v_hi_431_, lean_object* v_P_432_, lean_object* v_inst_433_){
_start:
{
uint8_t v___x_434_; 
v___x_434_ = lp_mathlib_Nat_decidableLoHiLe___redArg(v_lo_430_, v_hi_431_, v_inst_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidableLoHiLe___boxed(lean_object* v_lo_435_, lean_object* v_hi_436_, lean_object* v_P_437_, lean_object* v_inst_438_){
_start:
{
uint8_t v_res_439_; lean_object* v_r_440_; 
v_res_439_ = lp_mathlib_Nat_decidableLoHiLe(v_lo_435_, v_hi_436_, v_P_437_, v_inst_438_);
lean_dec(v_hi_436_);
v_r_440_ = lean_box(v_res_439_);
return v_r_440_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Nat_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Init(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Nat_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Init(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_foundational__algebra__order__theory = _init_lp_mathlib_LibraryNote_foundational__algebra__order__theory();
lean_mark_persistent(lp_mathlib_LibraryNote_foundational__algebra__order__theory);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Nat_Lemmas(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Init(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Nat_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Init(builtin);
}
#ifdef __cplusplus
}
#endif
