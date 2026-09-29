// Lean compiler output
// Module: Mathlib.Data.Sigma.Lex
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.Defs public import Mathlib.Order.Defs.Unbundled public import Batteries.Logic
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
LEAN_EXPORT uint8_t lp_mathlib_Sigma_Lex_decidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_Lex_decidable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sigma_Lex_decidable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_Lex_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_PSigma_Lex_decidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_Lex_decidable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_PSigma_Lex_decidable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_Lex_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sigma_Lex_decidable___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_x_4_, lean_object* v_x_5_){
_start:
{
lean_object* v_fst_6_; lean_object* v_snd_7_; lean_object* v_fst_8_; lean_object* v_snd_9_; lean_object* v___x_10_; uint8_t v___x_11_; 
v_fst_6_ = lean_ctor_get(v_x_4_, 0);
lean_inc_n(v_fst_6_, 2);
v_snd_7_ = lean_ctor_get(v_x_4_, 1);
lean_inc(v_snd_7_);
lean_dec_ref(v_x_4_);
v_fst_8_ = lean_ctor_get(v_x_5_, 0);
lean_inc_n(v_fst_8_, 2);
v_snd_9_ = lean_ctor_get(v_x_5_, 1);
lean_inc(v_snd_9_);
lean_dec_ref(v_x_5_);
v___x_10_ = lean_apply_2(v_inst_2_, v_fst_6_, v_fst_8_);
v___x_11_ = lean_unbox(v___x_10_);
if (v___x_11_ == 0)
{
lean_object* v___x_12_; uint8_t v___x_13_; 
lean_inc(v_fst_8_);
v___x_12_ = lean_apply_2(v_inst_1_, v_fst_6_, v_fst_8_);
v___x_13_ = lean_unbox(v___x_12_);
if (v___x_13_ == 0)
{
uint8_t v___x_14_; 
lean_dec(v_snd_9_);
lean_dec(v_fst_8_);
lean_dec(v_snd_7_);
lean_dec_ref(v_inst_3_);
v___x_14_ = lean_unbox(v___x_12_);
return v___x_14_;
}
else
{
lean_object* v___x_15_; uint8_t v___x_16_; 
v___x_15_ = lean_apply_3(v_inst_3_, v_fst_8_, v_snd_7_, v_snd_9_);
v___x_16_ = lean_unbox(v___x_15_);
return v___x_16_;
}
}
else
{
uint8_t v___x_17_; 
lean_dec(v_snd_9_);
lean_dec(v_fst_8_);
lean_dec(v_snd_7_);
lean_dec(v_fst_6_);
lean_dec_ref(v_inst_3_);
lean_dec_ref(v_inst_1_);
v___x_17_ = lean_unbox(v___x_10_);
return v___x_17_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_Lex_decidable___redArg___boxed(lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_x_21_, lean_object* v_x_22_){
_start:
{
uint8_t v_res_23_; lean_object* v_r_24_; 
v_res_23_ = lp_mathlib_Sigma_Lex_decidable___redArg(v_inst_18_, v_inst_19_, v_inst_20_, v_x_21_, v_x_22_);
v_r_24_ = lean_box(v_res_23_);
return v_r_24_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sigma_Lex_decidable(lean_object* v_00_u03b9_25_, lean_object* v_00_u03b1_26_, lean_object* v_r_27_, lean_object* v_s_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
uint8_t v___x_34_; 
v___x_34_ = lp_mathlib_Sigma_Lex_decidable___redArg(v_inst_29_, v_inst_30_, v_inst_31_, v_x_32_, v_x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_Lex_decidable___boxed(lean_object* v_00_u03b9_35_, lean_object* v_00_u03b1_36_, lean_object* v_r_37_, lean_object* v_s_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_x_42_, lean_object* v_x_43_){
_start:
{
uint8_t v_res_44_; lean_object* v_r_45_; 
v_res_44_ = lp_mathlib_Sigma_Lex_decidable(v_00_u03b9_35_, v_00_u03b1_36_, v_r_37_, v_s_38_, v_inst_39_, v_inst_40_, v_inst_41_, v_x_42_, v_x_43_);
v_r_45_ = lean_box(v_res_44_);
return v_r_45_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PSigma_Lex_decidable___redArg(lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_x_49_, lean_object* v_x_50_){
_start:
{
lean_object* v_fst_51_; lean_object* v_snd_52_; lean_object* v_fst_53_; lean_object* v_snd_54_; lean_object* v___x_55_; uint8_t v___x_56_; 
v_fst_51_ = lean_ctor_get(v_x_49_, 0);
lean_inc_n(v_fst_51_, 2);
v_snd_52_ = lean_ctor_get(v_x_49_, 1);
lean_inc(v_snd_52_);
lean_dec_ref(v_x_49_);
v_fst_53_ = lean_ctor_get(v_x_50_, 0);
lean_inc_n(v_fst_53_, 2);
v_snd_54_ = lean_ctor_get(v_x_50_, 1);
lean_inc(v_snd_54_);
lean_dec_ref(v_x_50_);
v___x_55_ = lean_apply_2(v_inst_47_, v_fst_51_, v_fst_53_);
v___x_56_ = lean_unbox(v___x_55_);
if (v___x_56_ == 0)
{
lean_object* v___x_57_; uint8_t v___x_58_; 
lean_inc(v_fst_53_);
v___x_57_ = lean_apply_2(v_inst_46_, v_fst_51_, v_fst_53_);
v___x_58_ = lean_unbox(v___x_57_);
if (v___x_58_ == 0)
{
uint8_t v___x_59_; 
lean_dec(v_snd_54_);
lean_dec(v_fst_53_);
lean_dec(v_snd_52_);
lean_dec_ref(v_inst_48_);
v___x_59_ = lean_unbox(v___x_57_);
return v___x_59_;
}
else
{
lean_object* v___x_60_; uint8_t v___x_61_; 
v___x_60_ = lean_apply_3(v_inst_48_, v_fst_53_, v_snd_52_, v_snd_54_);
v___x_61_ = lean_unbox(v___x_60_);
return v___x_61_;
}
}
else
{
uint8_t v___x_62_; 
lean_dec(v_snd_54_);
lean_dec(v_fst_53_);
lean_dec(v_snd_52_);
lean_dec(v_fst_51_);
lean_dec_ref(v_inst_48_);
lean_dec_ref(v_inst_46_);
v___x_62_ = lean_unbox(v___x_55_);
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_Lex_decidable___redArg___boxed(lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_x_66_, lean_object* v_x_67_){
_start:
{
uint8_t v_res_68_; lean_object* v_r_69_; 
v_res_68_ = lp_mathlib_PSigma_Lex_decidable___redArg(v_inst_63_, v_inst_64_, v_inst_65_, v_x_66_, v_x_67_);
v_r_69_ = lean_box(v_res_68_);
return v_r_69_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PSigma_Lex_decidable(lean_object* v_00_u03b9_70_, lean_object* v_00_u03b1_71_, lean_object* v_r_72_, lean_object* v_s_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_x_77_, lean_object* v_x_78_){
_start:
{
uint8_t v___x_79_; 
v___x_79_ = lp_mathlib_PSigma_Lex_decidable___redArg(v_inst_74_, v_inst_75_, v_inst_76_, v_x_77_, v_x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_Lex_decidable___boxed(lean_object* v_00_u03b9_80_, lean_object* v_00_u03b1_81_, lean_object* v_r_82_, lean_object* v_s_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_x_87_, lean_object* v_x_88_){
_start:
{
uint8_t v_res_89_; lean_object* v_r_90_; 
v_res_89_ = lp_mathlib_PSigma_Lex_decidable(v_00_u03b9_80_, v_00_u03b1_81_, v_r_82_, v_s_83_, v_inst_84_, v_inst_85_, v_inst_86_, v_x_87_, v_x_88_);
v_r_90_ = lean_box(v_res_89_);
return v_r_90_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_Unbundled(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Logic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Sigma_Lex(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_Unbundled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Sigma_Lex(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_Unbundled(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Logic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Sigma_Lex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_Unbundled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Logic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sigma_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Sigma_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Sigma_Lex(builtin);
}
#ifdef __cplusplus
}
#endif
