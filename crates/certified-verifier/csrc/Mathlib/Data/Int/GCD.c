// Lean compiler output
// Module: Mathlib.Data.Int.GCD
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Divisibility.Basic public import Mathlib.Algebra.Group.Commute.Units public import Mathlib.Algebra.Group.Int.Defs public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Algebra.GroupWithZero.Semiconj public import Mathlib.Data.Set.Operations public import Mathlib.Order.Basic public import Mathlib.Order.Bounds.Defs
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* lp_batteries_Nat_strongRec___redArg(lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_int_neg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_xgcdAux_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_xgcdAux___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_xgcdAux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_xgcdAux___lam__0, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_xgcdAux___closed__0 = (const lean_object*)&lp_mathlib_Nat_xgcdAux___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_xgcdAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Nat_xgcd___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_xgcd___closed__0;
static lean_once_cell_t lp_mathlib_Nat_xgcd___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Nat_xgcd___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Nat_xgcd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_gcdA(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_gcdB(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_P_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_P_match__1_splitter(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_gcdA(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_gcdA___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_gcdB(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_gcdB___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_xgcdAux_spec__0(lean_object* v_a_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_nat_to_int(v_a_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_xgcdAux___lam__0(lean_object* v_n_3_, lean_object* v_ih_4_, lean_object* v_s_5_, lean_object* v_t_6_, lean_object* v_r_x27_7_, lean_object* v_s_x27_8_, lean_object* v_t_x27_9_){
_start:
{
lean_object* v_zero_10_; uint8_t v_isZero_11_; 
v_zero_10_ = lean_unsigned_to_nat(0u);
v_isZero_11_ = lean_nat_dec_eq(v_n_3_, v_zero_10_);
if (v_isZero_11_ == 1)
{
lean_object* v___x_12_; lean_object* v___x_13_; 
lean_dec(v_t_6_);
lean_dec(v_s_5_);
lean_dec_ref(v_ih_4_);
lean_dec(v_n_3_);
v___x_12_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_12_, 0, v_s_x27_8_);
lean_ctor_set(v___x_12_, 1, v_t_x27_9_);
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v_r_x27_7_);
lean_ctor_set(v___x_13_, 1, v___x_12_);
return v___x_13_;
}
else
{
lean_object* v_q_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v_q_14_ = lean_nat_div(v_r_x27_7_, v_n_3_);
v___x_15_ = lean_nat_mod(v_r_x27_7_, v_n_3_);
lean_dec(v_r_x27_7_);
v___x_16_ = lean_nat_to_int(v_q_14_);
v___x_17_ = lean_int_mul(v___x_16_, v_s_5_);
v___x_18_ = lean_int_sub(v_s_x27_8_, v___x_17_);
lean_dec(v___x_17_);
lean_dec(v_s_x27_8_);
v___x_19_ = lean_int_mul(v___x_16_, v_t_6_);
lean_dec(v___x_16_);
v___x_20_ = lean_int_sub(v_t_x27_9_, v___x_19_);
lean_dec(v___x_19_);
lean_dec(v_t_x27_9_);
v___x_21_ = lean_apply_7(v_ih_4_, v___x_15_, lean_box(0), v___x_18_, v___x_20_, v_n_3_, v_s_5_, v_t_6_);
return v___x_21_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_xgcdAux(lean_object* v_t_23_, lean_object* v_a_24_, lean_object* v_a_25_, lean_object* v_a_26_, lean_object* v_a_27_, lean_object* v_a_28_){
_start:
{
lean_object* v___f_29_; lean_object* v___x_45__overap_30_; lean_object* v___x_31_; 
v___f_29_ = ((lean_object*)(lp_mathlib_Nat_xgcdAux___closed__0));
v___x_45__overap_30_ = lp_batteries_Nat_strongRec___redArg(v___f_29_, v_t_23_);
v___x_31_ = lean_apply_5(v___x_45__overap_30_, v_a_24_, v_a_25_, v_a_26_, v_a_27_, v_a_28_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_Nat_xgcd___closed__0(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_unsigned_to_nat(1u);
v___x_33_ = lean_nat_to_int(v___x_32_);
return v___x_33_;
}
}
static lean_object* _init_lp_mathlib_Nat_xgcd___closed__1(void){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_34_ = lean_unsigned_to_nat(0u);
v___x_35_ = lean_nat_to_int(v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_xgcd(lean_object* v_x_36_, lean_object* v_y_37_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v_snd_41_; 
v___x_38_ = lean_obj_once(&lp_mathlib_Nat_xgcd___closed__0, &lp_mathlib_Nat_xgcd___closed__0_once, _init_lp_mathlib_Nat_xgcd___closed__0);
v___x_39_ = lean_obj_once(&lp_mathlib_Nat_xgcd___closed__1, &lp_mathlib_Nat_xgcd___closed__1_once, _init_lp_mathlib_Nat_xgcd___closed__1);
v___x_40_ = lp_mathlib_Nat_xgcdAux(v_x_36_, v___x_38_, v___x_39_, v_y_37_, v___x_39_, v___x_38_);
v_snd_41_ = lean_ctor_get(v___x_40_, 1);
lean_inc(v_snd_41_);
lean_dec_ref(v___x_40_);
return v_snd_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_gcdA(lean_object* v_x_42_, lean_object* v_y_43_){
_start:
{
lean_object* v___x_44_; lean_object* v_fst_45_; 
v___x_44_ = lp_mathlib_Nat_xgcd(v_x_42_, v_y_43_);
v_fst_45_ = lean_ctor_get(v___x_44_, 0);
lean_inc(v_fst_45_);
lean_dec_ref(v___x_44_);
return v_fst_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_gcdB(lean_object* v_x_46_, lean_object* v_y_47_){
_start:
{
lean_object* v___x_48_; lean_object* v_snd_49_; 
v___x_48_ = lp_mathlib_Nat_xgcd(v_x_46_, v_y_47_);
v_snd_49_ = lean_ctor_get(v___x_48_, 1);
lean_inc(v_snd_49_);
lean_dec_ref(v___x_48_);
return v_snd_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter___redArg(lean_object* v_n_50_, lean_object* v_ih_51_, lean_object* v_h__1_52_, lean_object* v_h__2_53_){
_start:
{
lean_object* v_zero_54_; uint8_t v_isZero_55_; 
v_zero_54_ = lean_unsigned_to_nat(0u);
v_isZero_55_ = lean_nat_dec_eq(v_n_50_, v_zero_54_);
if (v_isZero_55_ == 1)
{
lean_object* v___x_56_; 
lean_dec(v_h__2_53_);
v___x_56_ = lean_apply_1(v_h__1_52_, v_ih_51_);
return v___x_56_;
}
else
{
lean_object* v_one_57_; lean_object* v_n_58_; lean_object* v___x_59_; 
lean_dec(v_h__1_52_);
v_one_57_ = lean_unsigned_to_nat(1u);
v_n_58_ = lean_nat_sub(v_n_50_, v_one_57_);
v___x_59_ = lean_apply_2(v_h__2_53_, v_n_58_, v_ih_51_);
return v___x_59_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter___redArg___boxed(lean_object* v_n_60_, lean_object* v_ih_61_, lean_object* v_h__1_62_, lean_object* v_h__2_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter___redArg(v_n_60_, v_ih_61_, v_h__1_62_, v_h__2_63_);
lean_dec(v_n_60_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter(lean_object* v_motive_65_, lean_object* v_n_66_, lean_object* v_ih_67_, lean_object* v_h__1_68_, lean_object* v_h__2_69_){
_start:
{
lean_object* v_zero_70_; uint8_t v_isZero_71_; 
v_zero_70_ = lean_unsigned_to_nat(0u);
v_isZero_71_ = lean_nat_dec_eq(v_n_66_, v_zero_70_);
if (v_isZero_71_ == 1)
{
lean_object* v___x_72_; 
lean_dec(v_h__2_69_);
v___x_72_ = lean_apply_1(v_h__1_68_, v_ih_67_);
return v___x_72_;
}
else
{
lean_object* v_one_73_; lean_object* v_n_74_; lean_object* v___x_75_; 
lean_dec(v_h__1_68_);
v_one_73_ = lean_unsigned_to_nat(1u);
v_n_74_ = lean_nat_sub(v_n_66_, v_one_73_);
v___x_75_ = lean_apply_2(v_h__2_69_, v_n_74_, v_ih_67_);
return v___x_75_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter___boxed(lean_object* v_motive_76_, lean_object* v_n_77_, lean_object* v_ih_78_, lean_object* v_h__1_79_, lean_object* v_h__2_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_xgcdAux_match__1_splitter(v_motive_76_, v_n_77_, v_ih_78_, v_h__1_79_, v_h__2_80_);
lean_dec(v_n_77_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_P_match__1_splitter___redArg(lean_object* v_x_82_, lean_object* v_h__1_83_){
_start:
{
lean_object* v_snd_84_; lean_object* v_fst_85_; lean_object* v_fst_86_; lean_object* v_snd_87_; lean_object* v___x_88_; 
v_snd_84_ = lean_ctor_get(v_x_82_, 1);
lean_inc(v_snd_84_);
v_fst_85_ = lean_ctor_get(v_x_82_, 0);
lean_inc(v_fst_85_);
lean_dec_ref(v_x_82_);
v_fst_86_ = lean_ctor_get(v_snd_84_, 0);
lean_inc(v_fst_86_);
v_snd_87_ = lean_ctor_get(v_snd_84_, 1);
lean_inc(v_snd_87_);
lean_dec(v_snd_84_);
v___x_88_ = lean_apply_3(v_h__1_83_, v_fst_85_, v_fst_86_, v_snd_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Int_GCD_0__Nat_P_match__1_splitter(lean_object* v_motive_89_, lean_object* v_x_90_, lean_object* v_h__1_91_){
_start:
{
lean_object* v_snd_92_; lean_object* v_fst_93_; lean_object* v_fst_94_; lean_object* v_snd_95_; lean_object* v___x_96_; 
v_snd_92_ = lean_ctor_get(v_x_90_, 1);
lean_inc(v_snd_92_);
v_fst_93_ = lean_ctor_get(v_x_90_, 0);
lean_inc(v_fst_93_);
lean_dec_ref(v_x_90_);
v_fst_94_ = lean_ctor_get(v_snd_92_, 0);
lean_inc(v_fst_94_);
v_snd_95_ = lean_ctor_get(v_snd_92_, 1);
lean_inc(v_snd_95_);
lean_dec(v_snd_92_);
v___x_96_ = lean_apply_3(v_h__1_91_, v_fst_93_, v_fst_94_, v_snd_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_gcdA(lean_object* v_x_97_, lean_object* v_x_98_){
_start:
{
lean_object* v_intZero_99_; uint8_t v_isNeg_100_; 
v_intZero_99_ = lean_obj_once(&lp_mathlib_Nat_xgcd___closed__1, &lp_mathlib_Nat_xgcd___closed__1_once, _init_lp_mathlib_Nat_xgcd___closed__1);
v_isNeg_100_ = lean_int_dec_lt(v_x_97_, v_intZero_99_);
if (v_isNeg_100_ == 0)
{
lean_object* v_a_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v_a_101_ = lean_nat_abs(v_x_97_);
v___x_102_ = lean_nat_abs(v_x_98_);
v___x_103_ = lp_mathlib_Nat_gcdA(v_a_101_, v___x_102_);
return v___x_103_;
}
else
{
lean_object* v_abs_104_; lean_object* v_one_105_; lean_object* v_a_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v_abs_104_ = lean_nat_abs(v_x_97_);
v_one_105_ = lean_unsigned_to_nat(1u);
v_a_106_ = lean_nat_sub(v_abs_104_, v_one_105_);
lean_dec(v_abs_104_);
v___x_107_ = lean_nat_add(v_a_106_, v_one_105_);
lean_dec(v_a_106_);
v___x_108_ = lean_nat_abs(v_x_98_);
v___x_109_ = lp_mathlib_Nat_gcdA(v___x_107_, v___x_108_);
v___x_110_ = lean_int_neg(v___x_109_);
lean_dec(v___x_109_);
return v___x_110_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_gcdA___boxed(lean_object* v_x_111_, lean_object* v_x_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Int_gcdA(v_x_111_, v_x_112_);
lean_dec(v_x_112_);
lean_dec(v_x_111_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_gcdB(lean_object* v_x_114_, lean_object* v_x_115_){
_start:
{
lean_object* v_intZero_116_; uint8_t v_isNeg_117_; 
v_intZero_116_ = lean_obj_once(&lp_mathlib_Nat_xgcd___closed__1, &lp_mathlib_Nat_xgcd___closed__1_once, _init_lp_mathlib_Nat_xgcd___closed__1);
v_isNeg_117_ = lean_int_dec_lt(v_x_115_, v_intZero_116_);
if (v_isNeg_117_ == 0)
{
lean_object* v_a_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v_a_118_ = lean_nat_abs(v_x_115_);
v___x_119_ = lean_nat_abs(v_x_114_);
v___x_120_ = lp_mathlib_Nat_gcdB(v___x_119_, v_a_118_);
return v___x_120_;
}
else
{
lean_object* v_abs_121_; lean_object* v_one_122_; lean_object* v_a_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v_abs_121_ = lean_nat_abs(v_x_115_);
v_one_122_ = lean_unsigned_to_nat(1u);
v_a_123_ = lean_nat_sub(v_abs_121_, v_one_122_);
lean_dec(v_abs_121_);
v___x_124_ = lean_nat_abs(v_x_114_);
v___x_125_ = lean_nat_add(v_a_123_, v_one_122_);
lean_dec(v_a_123_);
v___x_126_ = lp_mathlib_Nat_gcdB(v___x_124_, v___x_125_);
v___x_127_ = lean_int_neg(v___x_126_);
lean_dec(v___x_126_);
return v___x_127_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_gcdB___boxed(lean_object* v_x_128_, lean_object* v_x_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_Int_gcdB(v_x_128_, v_x_129_);
lean_dec(v_x_129_);
lean_dec(v_x_128_);
return v_res_130_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Semiconj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_GCD(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Semiconj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Int_GCD(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Semiconj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_GCD(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Semiconj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_GCD(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Int_GCD(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Int_GCD(builtin);
}
#ifdef __cplusplus
}
#endif
