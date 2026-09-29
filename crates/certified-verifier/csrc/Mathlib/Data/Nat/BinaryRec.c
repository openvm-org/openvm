// Lean compiler output
// Module: Mathlib.Data.Nat.BinaryRec
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bit(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bit___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bitCasesOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bitCasesOn___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bitCasesOn(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bitCasesOn___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_bit(uint8_t v_b_1_, lean_object* v_n_2_){
_start:
{
if (v_b_1_ == 0)
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_unsigned_to_nat(2u);
v___x_4_ = lean_nat_mul(v___x_3_, v_n_2_);
return v___x_4_;
}
else
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_5_ = lean_unsigned_to_nat(2u);
v___x_6_ = lean_nat_mul(v___x_5_, v_n_2_);
v___x_7_ = lean_unsigned_to_nat(1u);
v___x_8_ = lean_nat_add(v___x_6_, v___x_7_);
lean_dec(v___x_6_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_bit___boxed(lean_object* v_b_9_, lean_object* v_n_10_){
_start:
{
uint8_t v_b_boxed_11_; lean_object* v_res_12_; 
v_b_boxed_11_ = lean_unbox(v_b_9_);
v_res_12_ = lp_mathlib_Nat_bit(v_b_boxed_11_, v_n_10_);
lean_dec(v_n_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_bitCasesOn___redArg(lean_object* v_n_13_, lean_object* v_bit_14_){
_start:
{
lean_object* v___x_15_; uint8_t v___y_17_; lean_object* v___x_21_; lean_object* v___x_22_; uint8_t v___x_23_; 
v___x_15_ = lean_unsigned_to_nat(1u);
v___x_21_ = lean_nat_land(v___x_15_, v_n_13_);
v___x_22_ = lean_unsigned_to_nat(0u);
v___x_23_ = lean_nat_dec_eq(v___x_21_, v___x_22_);
lean_dec(v___x_21_);
if (v___x_23_ == 0)
{
uint8_t v___x_24_; 
v___x_24_ = 1;
v___y_17_ = v___x_24_;
goto v___jp_16_;
}
else
{
uint8_t v___x_25_; 
v___x_25_ = 0;
v___y_17_ = v___x_25_;
goto v___jp_16_;
}
v___jp_16_:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v_x_20_; 
v___x_18_ = lean_nat_shiftr(v_n_13_, v___x_15_);
v___x_19_ = lean_box(v___y_17_);
v_x_20_ = lean_apply_2(v_bit_14_, v___x_19_, v___x_18_);
return v_x_20_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_bitCasesOn___redArg___boxed(lean_object* v_n_26_, lean_object* v_bit_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Nat_bitCasesOn___redArg(v_n_26_, v_bit_27_);
lean_dec(v_n_26_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_bitCasesOn(lean_object* v_motive_29_, lean_object* v_n_30_, lean_object* v_bit_31_){
_start:
{
lean_object* v___x_32_; uint8_t v___y_34_; lean_object* v___x_38_; lean_object* v___x_39_; uint8_t v___x_40_; 
v___x_32_ = lean_unsigned_to_nat(1u);
v___x_38_ = lean_nat_land(v___x_32_, v_n_30_);
v___x_39_ = lean_unsigned_to_nat(0u);
v___x_40_ = lean_nat_dec_eq(v___x_38_, v___x_39_);
lean_dec(v___x_38_);
if (v___x_40_ == 0)
{
uint8_t v___x_41_; 
v___x_41_ = 1;
v___y_34_ = v___x_41_;
goto v___jp_33_;
}
else
{
uint8_t v___x_42_; 
v___x_42_ = 0;
v___y_34_ = v___x_42_;
goto v___jp_33_;
}
v___jp_33_:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v_x_37_; 
v___x_35_ = lean_nat_shiftr(v_n_30_, v___x_32_);
v___x_36_ = lean_box(v___y_34_);
v_x_37_ = lean_apply_2(v_bit_31_, v___x_36_, v___x_35_);
return v_x_37_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_bitCasesOn___boxed(lean_object* v_motive_43_, lean_object* v_n_44_, lean_object* v_bit_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Nat_bitCasesOn(v_motive_43_, v_n_44_, v_bit_45_);
lean_dec(v_n_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___redArg(lean_object* v_zero_47_, lean_object* v_bit_48_, lean_object* v_n_49_){
_start:
{
lean_object* v___x_50_; uint8_t v___x_51_; 
v___x_50_ = lean_unsigned_to_nat(0u);
v___x_51_ = lean_nat_dec_eq(v_n_49_, v___x_50_);
if (v___x_51_ == 0)
{
lean_object* v___x_52_; uint8_t v___y_54_; lean_object* v___x_59_; uint8_t v___x_60_; 
v___x_52_ = lean_unsigned_to_nat(1u);
v___x_59_ = lean_nat_land(v___x_52_, v_n_49_);
v___x_60_ = lean_nat_dec_eq(v___x_59_, v___x_50_);
lean_dec(v___x_59_);
if (v___x_60_ == 0)
{
uint8_t v___x_61_; 
v___x_61_ = 1;
v___y_54_ = v___x_61_;
goto v___jp_53_;
}
else
{
v___y_54_ = v___x_51_;
goto v___jp_53_;
}
v___jp_53_:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v_x_58_; 
v___x_55_ = lean_nat_shiftr(v_n_49_, v___x_52_);
lean_inc(v_bit_48_);
v___x_56_ = lp_mathlib_Nat_binaryRec___redArg(v_zero_47_, v_bit_48_, v___x_55_);
v___x_57_ = lean_box(v___y_54_);
v_x_58_ = lean_apply_3(v_bit_48_, v___x_57_, v___x_55_, v___x_56_);
return v_x_58_;
}
}
else
{
lean_dec(v_bit_48_);
lean_inc(v_zero_47_);
return v_zero_47_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___redArg___boxed(lean_object* v_zero_62_, lean_object* v_bit_63_, lean_object* v_n_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_Nat_binaryRec___redArg(v_zero_62_, v_bit_63_, v_n_64_);
lean_dec(v_n_64_);
lean_dec(v_zero_62_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec(lean_object* v_motive_66_, lean_object* v_zero_67_, lean_object* v_bit_68_, lean_object* v_n_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Nat_binaryRec___redArg(v_zero_67_, v_bit_68_, v_n_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___boxed(lean_object* v_motive_71_, lean_object* v_zero_72_, lean_object* v_bit_73_, lean_object* v_n_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_Nat_binaryRec(v_motive_71_, v_zero_72_, v_bit_73_, v_n_74_);
lean_dec(v_n_74_);
lean_dec(v_zero_72_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___redArg___lam__0(lean_object* v_bit_76_, lean_object* v_zero_77_, uint8_t v_b_78_, lean_object* v_n_79_, lean_object* v_ih_80_){
_start:
{
lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_81_ = lean_unsigned_to_nat(0u);
v___x_82_ = lean_nat_dec_eq(v_n_79_, v___x_81_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_83_ = lean_box(v_b_78_);
v___x_84_ = lean_apply_4(v_bit_76_, v___x_83_, v_n_79_, lean_box(0), v_ih_80_);
return v___x_84_;
}
else
{
if (v_b_78_ == 0)
{
lean_dec(v_ih_80_);
lean_dec(v_n_79_);
lean_dec(v_bit_76_);
lean_inc(v_zero_77_);
return v_zero_77_;
}
else
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = lean_box(v_b_78_);
v___x_86_ = lean_apply_4(v_bit_76_, v___x_85_, v_n_79_, lean_box(0), v_ih_80_);
return v___x_86_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___redArg___lam__0___boxed(lean_object* v_bit_87_, lean_object* v_zero_88_, lean_object* v_b_89_, lean_object* v_n_90_, lean_object* v_ih_91_){
_start:
{
uint8_t v_b_boxed_92_; lean_object* v_res_93_; 
v_b_boxed_92_ = lean_unbox(v_b_89_);
v_res_93_ = lp_mathlib_Nat_binaryRec_x27___redArg___lam__0(v_bit_87_, v_zero_88_, v_b_boxed_92_, v_n_90_, v_ih_91_);
lean_dec(v_zero_88_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___redArg(lean_object* v_zero_94_, lean_object* v_bit_95_, lean_object* v_n_96_){
_start:
{
lean_object* v___f_97_; lean_object* v___x_98_; 
lean_inc(v_zero_94_);
v___f_97_ = lean_alloc_closure((void*)(lp_mathlib_Nat_binaryRec_x27___redArg___lam__0___boxed), 5, 2);
lean_closure_set(v___f_97_, 0, v_bit_95_);
lean_closure_set(v___f_97_, 1, v_zero_94_);
v___x_98_ = lp_mathlib_Nat_binaryRec___redArg(v_zero_94_, v___f_97_, v_n_96_);
lean_dec(v_zero_94_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___redArg___boxed(lean_object* v_zero_99_, lean_object* v_bit_100_, lean_object* v_n_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Nat_binaryRec_x27___redArg(v_zero_99_, v_bit_100_, v_n_101_);
lean_dec(v_n_101_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27(lean_object* v_motive_103_, lean_object* v_zero_104_, lean_object* v_bit_105_, lean_object* v_n_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_Nat_binaryRec_x27___redArg(v_zero_104_, v_bit_105_, v_n_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec_x27___boxed(lean_object* v_motive_108_, lean_object* v_zero_109_, lean_object* v_bit_110_, lean_object* v_n_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_Nat_binaryRec_x27(v_motive_108_, v_zero_109_, v_bit_110_, v_n_111_);
lean_dec(v_n_111_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___redArg___lam__0(lean_object* v_bit_113_, lean_object* v_one_114_, uint8_t v_b_115_, lean_object* v_n_116_, lean_object* v_h_117_, lean_object* v_ih_118_){
_start:
{
lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_119_ = lean_unsigned_to_nat(0u);
v___x_120_ = lean_nat_dec_eq(v_n_116_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_121_ = lean_box(v_b_115_);
v___x_122_ = lean_apply_4(v_bit_113_, v___x_121_, v_n_116_, lean_box(0), v_ih_118_);
return v___x_122_;
}
else
{
lean_dec(v_ih_118_);
lean_dec(v_n_116_);
lean_dec(v_bit_113_);
lean_inc(v_one_114_);
return v_one_114_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___redArg___lam__0___boxed(lean_object* v_bit_123_, lean_object* v_one_124_, lean_object* v_b_125_, lean_object* v_n_126_, lean_object* v_h_127_, lean_object* v_ih_128_){
_start:
{
uint8_t v_b_boxed_129_; lean_object* v_res_130_; 
v_b_boxed_129_ = lean_unbox(v_b_125_);
v_res_130_ = lp_mathlib_Nat_binaryRecFromOne___redArg___lam__0(v_bit_123_, v_one_124_, v_b_boxed_129_, v_n_126_, v_h_127_, v_ih_128_);
lean_dec(v_one_124_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___redArg(lean_object* v_zero_131_, lean_object* v_one_132_, lean_object* v_bit_133_, lean_object* v_n_134_){
_start:
{
lean_object* v___f_135_; lean_object* v___x_136_; 
v___f_135_ = lean_alloc_closure((void*)(lp_mathlib_Nat_binaryRecFromOne___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_135_, 0, v_bit_133_);
lean_closure_set(v___f_135_, 1, v_one_132_);
v___x_136_ = lp_mathlib_Nat_binaryRec_x27___redArg(v_zero_131_, v___f_135_, v_n_134_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___redArg___boxed(lean_object* v_zero_137_, lean_object* v_one_138_, lean_object* v_bit_139_, lean_object* v_n_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Nat_binaryRecFromOne___redArg(v_zero_137_, v_one_138_, v_bit_139_, v_n_140_);
lean_dec(v_n_140_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne(lean_object* v_motive_142_, lean_object* v_zero_143_, lean_object* v_one_144_, lean_object* v_bit_145_, lean_object* v_n_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_mathlib_Nat_binaryRecFromOne___redArg(v_zero_143_, v_one_144_, v_bit_145_, v_n_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRecFromOne___boxed(lean_object* v_motive_148_, lean_object* v_zero_149_, lean_object* v_one_150_, lean_object* v_bit_151_, lean_object* v_n_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_Nat_binaryRecFromOne(v_motive_148_, v_zero_149_, v_one_150_, v_bit_151_, v_n_152_);
lean_dec(v_n_152_);
return v_res_153_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_BinaryRec(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_BinaryRec(builtin);
}
#ifdef __cplusplus
}
#endif
