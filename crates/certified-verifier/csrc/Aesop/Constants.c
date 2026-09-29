// Lean compiler output
// Module: Aesop.Constants
// Imports: public import Init public meta import Init public import Aesop.Percent
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
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_unificationGoalPenalty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_unificationGoalPenalty___closed__0;
LEAN_EXPORT double lp_aesop_Aesop_unificationGoalPenalty;
static lean_once_cell_t lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0;
LEAN_EXPORT double lp_aesop_Aesop_postponedSafeRuleSuccessProbability;
static double _init_lp_aesop_Aesop_unificationGoalPenalty___closed__0(void){
_start:
{
lean_object* v___x_1_; uint8_t v___x_2_; lean_object* v___x_3_; double v___x_4_; 
v___x_1_ = lean_unsigned_to_nat(1u);
v___x_2_ = 1;
v___x_3_ = lean_unsigned_to_nat(8u);
v___x_4_ = l_Float_ofScientific(v___x_3_, v___x_2_, v___x_1_);
return v___x_4_;
}
}
static double _init_lp_aesop_Aesop_unificationGoalPenalty(void){
_start:
{
double v___x_5_; 
v___x_5_ = lean_float_once(&lp_aesop_Aesop_unificationGoalPenalty___closed__0, &lp_aesop_Aesop_unificationGoalPenalty___closed__0_once, _init_lp_aesop_Aesop_unificationGoalPenalty___closed__0);
return v___x_5_;
}
}
static double _init_lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0(void){
_start:
{
lean_object* v___x_6_; uint8_t v___x_7_; lean_object* v___x_8_; double v___x_9_; 
v___x_6_ = lean_unsigned_to_nat(1u);
v___x_7_ = 1;
v___x_8_ = lean_unsigned_to_nat(9u);
v___x_9_ = l_Float_ofScientific(v___x_8_, v___x_7_, v___x_6_);
return v___x_9_;
}
}
static double _init_lp_aesop_Aesop_postponedSafeRuleSuccessProbability(void){
_start:
{
double v___x_10_; 
v___x_10_ = lean_float_once(&lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0, &lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0_once, _init_lp_aesop_Aesop_postponedSafeRuleSuccessProbability___closed__0);
return v___x_10_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Percent(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Constants(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_unificationGoalPenalty = _init_lp_aesop_Aesop_unificationGoalPenalty();
lp_aesop_Aesop_postponedSafeRuleSuccessProbability = _init_lp_aesop_Aesop_postponedSafeRuleSuccessProbability();
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Constants(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Percent(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Constants(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Constants(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Constants(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Constants(builtin);
}
#ifdef __cplusplus
}
#endif
