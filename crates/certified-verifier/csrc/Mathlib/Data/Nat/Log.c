// Lean compiler output
// Module: Mathlib.Data.Nat.Log
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.BinaryRec public import Mathlib.Order.Interval.Set.Defs public import Mathlib.Order.Monotone.Basic public import Mathlib.Tactic.Bound.Attribute public import Mathlib.Tactic.Contrapose public import Mathlib.Tactic.Monotonicity.Attr
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
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_log_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_log_go___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_log(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_log___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__1_splitter(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_clog_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_clog_go___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_clog(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_clog___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_log_go(lean_object* v_n_1_, lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
lean_object* v_zero_4_; uint8_t v_isZero_5_; 
v_zero_4_ = lean_unsigned_to_nat(0u);
v_isZero_5_ = lean_nat_dec_eq(v_a_3_, v_zero_4_);
if (v_isZero_5_ == 1)
{
lean_object* v___x_6_; 
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v_n_1_);
lean_ctor_set(v___x_6_, 1, v_zero_4_);
return v___x_6_;
}
else
{
uint8_t v___x_7_; 
v___x_7_ = lean_nat_dec_lt(v_n_1_, v_a_2_);
if (v___x_7_ == 0)
{
lean_object* v_one_8_; lean_object* v_n_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v_fst_12_; lean_object* v_snd_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_30_; 
v_one_8_ = lean_unsigned_to_nat(1u);
v_n_9_ = lean_nat_sub(v_a_3_, v_one_8_);
v___x_10_ = lean_nat_mul(v_a_2_, v_a_2_);
v___x_11_ = lp_mathlib_Nat_log_go(v_n_1_, v___x_10_, v_n_9_);
lean_dec(v_n_9_);
lean_dec(v___x_10_);
v_fst_12_ = lean_ctor_get(v___x_11_, 0);
v_snd_13_ = lean_ctor_get(v___x_11_, 1);
v_isSharedCheck_30_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_30_ == 0)
{
v___x_15_ = v___x_11_;
v_isShared_16_ = v_isSharedCheck_30_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_snd_13_);
lean_inc(v_fst_12_);
lean_dec(v___x_11_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_30_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
uint8_t v___x_17_; 
v___x_17_ = lean_nat_dec_lt(v_fst_12_, v_a_2_);
if (v___x_17_ == 0)
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_23_; 
v___x_18_ = lean_nat_div(v_fst_12_, v_a_2_);
lean_dec(v_fst_12_);
v___x_19_ = lean_unsigned_to_nat(2u);
v___x_20_ = lean_nat_mul(v___x_19_, v_snd_13_);
lean_dec(v_snd_13_);
v___x_21_ = lean_nat_add(v___x_20_, v_one_8_);
lean_dec(v___x_20_);
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 1, v___x_21_);
lean_ctor_set(v___x_15_, 0, v___x_18_);
v___x_23_ = v___x_15_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v___x_18_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v___x_21_);
v___x_23_ = v_reuseFailAlloc_24_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
return v___x_23_;
}
}
else
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_28_; 
v___x_25_ = lean_unsigned_to_nat(2u);
v___x_26_ = lean_nat_mul(v___x_25_, v_snd_13_);
lean_dec(v_snd_13_);
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 1, v___x_26_);
v___x_28_ = v___x_15_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v_fst_12_);
lean_ctor_set(v_reuseFailAlloc_29_, 1, v___x_26_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
else
{
lean_object* v___x_31_; 
v___x_31_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_31_, 0, v_n_1_);
lean_ctor_set(v___x_31_, 1, v_zero_4_);
return v___x_31_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_log_go___boxed(lean_object* v_n_32_, lean_object* v_a_33_, lean_object* v_a_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Nat_log_go(v_n_32_, v_a_33_, v_a_34_);
lean_dec(v_a_34_);
lean_dec(v_a_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_log(lean_object* v_b_36_, lean_object* v_n_37_){
_start:
{
lean_object* v___x_38_; uint8_t v___x_39_; 
v___x_38_ = lean_unsigned_to_nat(1u);
v___x_39_ = lean_nat_dec_le(v_b_36_, v___x_38_);
if (v___x_39_ == 0)
{
lean_object* v___x_40_; lean_object* v_snd_41_; 
lean_inc(v_n_37_);
v___x_40_ = lp_mathlib_Nat_log_go(v_n_37_, v_b_36_, v_n_37_);
lean_dec(v_n_37_);
v_snd_41_ = lean_ctor_get(v___x_40_, 1);
lean_inc(v_snd_41_);
lean_dec_ref(v___x_40_);
return v_snd_41_;
}
else
{
lean_object* v___x_42_; 
lean_dec(v_n_37_);
v___x_42_ = lean_unsigned_to_nat(0u);
return v___x_42_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_log___boxed(lean_object* v_b_43_, lean_object* v_n_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_Nat_log(v_b_43_, v_n_44_);
lean_dec(v_b_43_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter___redArg(lean_object* v_x_46_, lean_object* v_x_47_, lean_object* v_h__1_48_, lean_object* v_h__2_49_){
_start:
{
lean_object* v_zero_50_; uint8_t v_isZero_51_; 
v_zero_50_ = lean_unsigned_to_nat(0u);
v_isZero_51_ = lean_nat_dec_eq(v_x_47_, v_zero_50_);
if (v_isZero_51_ == 1)
{
lean_object* v___x_52_; 
lean_dec(v_h__2_49_);
v___x_52_ = lean_apply_1(v_h__1_48_, v_x_46_);
return v___x_52_;
}
else
{
lean_object* v_one_53_; lean_object* v_n_54_; lean_object* v___x_55_; 
lean_dec(v_h__1_48_);
v_one_53_ = lean_unsigned_to_nat(1u);
v_n_54_ = lean_nat_sub(v_x_47_, v_one_53_);
v___x_55_ = lean_apply_2(v_h__2_49_, v_x_46_, v_n_54_);
return v___x_55_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter___redArg___boxed(lean_object* v_x_56_, lean_object* v_x_57_, lean_object* v_h__1_58_, lean_object* v_h__2_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter___redArg(v_x_56_, v_x_57_, v_h__1_58_, v_h__2_59_);
lean_dec(v_x_57_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter(lean_object* v_motive_61_, lean_object* v_x_62_, lean_object* v_x_63_, lean_object* v_h__1_64_, lean_object* v_h__2_65_){
_start:
{
lean_object* v_zero_66_; uint8_t v_isZero_67_; 
v_zero_66_ = lean_unsigned_to_nat(0u);
v_isZero_67_ = lean_nat_dec_eq(v_x_63_, v_zero_66_);
if (v_isZero_67_ == 1)
{
lean_object* v___x_68_; 
lean_dec(v_h__2_65_);
v___x_68_ = lean_apply_1(v_h__1_64_, v_x_62_);
return v___x_68_;
}
else
{
lean_object* v_one_69_; lean_object* v_n_70_; lean_object* v___x_71_; 
lean_dec(v_h__1_64_);
v_one_69_ = lean_unsigned_to_nat(1u);
v_n_70_ = lean_nat_sub(v_x_63_, v_one_69_);
v___x_71_ = lean_apply_2(v_h__2_65_, v_x_62_, v_n_70_);
return v___x_71_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter___boxed(lean_object* v_motive_72_, lean_object* v_x_73_, lean_object* v_x_74_, lean_object* v_h__1_75_, lean_object* v_h__2_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__3_splitter(v_motive_72_, v_x_73_, v_x_74_, v_h__1_75_, v_h__2_76_);
lean_dec(v_x_74_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__1_splitter___redArg(lean_object* v_x_78_, lean_object* v_h__1_79_){
_start:
{
lean_object* v_fst_80_; lean_object* v_snd_81_; lean_object* v___x_82_; 
v_fst_80_ = lean_ctor_get(v_x_78_, 0);
lean_inc(v_fst_80_);
v_snd_81_ = lean_ctor_get(v_x_78_, 1);
lean_inc(v_snd_81_);
lean_dec_ref(v_x_78_);
v___x_82_ = lean_apply_2(v_h__1_79_, v_fst_80_, v_snd_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Log_0__Nat_log_go_match__1_splitter(lean_object* v_motive_83_, lean_object* v_x_84_, lean_object* v_h__1_85_){
_start:
{
lean_object* v_fst_86_; lean_object* v_snd_87_; lean_object* v___x_88_; 
v_fst_86_ = lean_ctor_get(v_x_84_, 0);
lean_inc(v_fst_86_);
v_snd_87_ = lean_ctor_get(v_x_84_, 1);
lean_inc(v_snd_87_);
lean_dec_ref(v_x_84_);
v___x_88_ = lean_apply_2(v_h__1_85_, v_fst_86_, v_snd_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_clog_go(lean_object* v_n_89_, lean_object* v_a_90_, lean_object* v_a_91_){
_start:
{
lean_object* v_zero_92_; uint8_t v_isZero_93_; 
v_zero_92_ = lean_unsigned_to_nat(0u);
v_isZero_93_ = lean_nat_dec_eq(v_a_91_, v_zero_92_);
if (v_isZero_93_ == 1)
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = lean_nat_div(v_a_90_, v_n_89_);
v___x_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v_zero_92_);
return v___x_95_;
}
else
{
uint8_t v___x_96_; 
v___x_96_ = lean_nat_dec_le(v_n_89_, v_a_90_);
if (v___x_96_ == 0)
{
lean_object* v_one_97_; lean_object* v_n_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v_fst_101_; lean_object* v_snd_102_; lean_object* v___x_104_; uint8_t v_isShared_105_; uint8_t v_isSharedCheck_119_; 
v_one_97_ = lean_unsigned_to_nat(1u);
v_n_98_ = lean_nat_sub(v_a_91_, v_one_97_);
v___x_99_ = lean_nat_mul(v_a_90_, v_a_90_);
v___x_100_ = lp_mathlib_Nat_clog_go(v_n_89_, v___x_99_, v_n_98_);
lean_dec(v_n_98_);
lean_dec(v___x_99_);
v_fst_101_ = lean_ctor_get(v___x_100_, 0);
v_snd_102_ = lean_ctor_get(v___x_100_, 1);
v_isSharedCheck_119_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_119_ == 0)
{
v___x_104_ = v___x_100_;
v_isShared_105_ = v_isSharedCheck_119_;
goto v_resetjp_103_;
}
else
{
lean_inc(v_snd_102_);
lean_inc(v_fst_101_);
lean_dec(v___x_100_);
v___x_104_ = lean_box(0);
v_isShared_105_ = v_isSharedCheck_119_;
goto v_resetjp_103_;
}
v_resetjp_103_:
{
uint8_t v___x_106_; 
v___x_106_ = lean_nat_dec_lt(v_fst_101_, v_a_90_);
if (v___x_106_ == 0)
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_111_; 
v___x_107_ = lean_nat_div(v_fst_101_, v_a_90_);
lean_dec(v_fst_101_);
v___x_108_ = lean_unsigned_to_nat(2u);
v___x_109_ = lean_nat_mul(v___x_108_, v_snd_102_);
lean_dec(v_snd_102_);
if (v_isShared_105_ == 0)
{
lean_ctor_set(v___x_104_, 1, v___x_109_);
lean_ctor_set(v___x_104_, 0, v___x_107_);
v___x_111_ = v___x_104_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v___x_107_);
lean_ctor_set(v_reuseFailAlloc_112_, 1, v___x_109_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
return v___x_111_;
}
}
else
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_117_; 
v___x_113_ = lean_unsigned_to_nat(2u);
v___x_114_ = lean_nat_mul(v___x_113_, v_snd_102_);
lean_dec(v_snd_102_);
v___x_115_ = lean_nat_add(v___x_114_, v_one_97_);
lean_dec(v___x_114_);
if (v_isShared_105_ == 0)
{
lean_ctor_set(v___x_104_, 1, v___x_115_);
v___x_117_ = v___x_104_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v_fst_101_);
lean_ctor_set(v_reuseFailAlloc_118_, 1, v___x_115_);
v___x_117_ = v_reuseFailAlloc_118_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
return v___x_117_;
}
}
}
}
else
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = lean_nat_div(v_a_90_, v_n_89_);
v___x_121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v_zero_92_);
return v___x_121_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_clog_go___boxed(lean_object* v_n_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Nat_clog_go(v_n_122_, v_a_123_, v_a_124_);
lean_dec(v_a_124_);
lean_dec(v_a_123_);
lean_dec(v_n_122_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_clog(lean_object* v_b_126_, lean_object* v_n_127_){
_start:
{
lean_object* v___x_128_; uint8_t v___x_129_; 
v___x_128_ = lean_unsigned_to_nat(1u);
v___x_129_ = lean_nat_dec_lt(v___x_128_, v_b_126_);
if (v___x_129_ == 0)
{
lean_object* v___x_130_; 
v___x_130_ = lean_unsigned_to_nat(0u);
return v___x_130_;
}
else
{
uint8_t v___x_131_; 
v___x_131_ = lean_nat_dec_lt(v___x_128_, v_n_127_);
if (v___x_131_ == 0)
{
lean_object* v___x_132_; 
v___x_132_ = lean_unsigned_to_nat(0u);
return v___x_132_;
}
else
{
lean_object* v___x_133_; lean_object* v_snd_134_; lean_object* v___x_135_; 
v___x_133_ = lp_mathlib_Nat_clog_go(v_n_127_, v_b_126_, v_n_127_);
v_snd_134_ = lean_ctor_get(v___x_133_, 1);
lean_inc(v_snd_134_);
lean_dec_ref(v___x_133_);
v___x_135_ = lean_nat_add(v_snd_134_, v___x_128_);
lean_dec(v_snd_134_);
return v___x_135_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_clog___boxed(lean_object* v_b_136_, lean_object* v_n_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Nat_clog(v_b_136_, v_n_137_);
lean_dec(v_n_137_);
lean_dec(v_b_136_);
return v_res_138_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Monotone_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Bound_Attribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Log(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Monotone_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Bound_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Log(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Monotone_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Bound_Attribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Log(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Monotone_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Bound_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Log(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Log(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Log(builtin);
}
#ifdef __cplusplus
}
#endif
