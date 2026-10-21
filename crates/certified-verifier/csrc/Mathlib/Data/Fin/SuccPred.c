// Lean compiler output
// Module: Mathlib.Data.Fin.SuccPred
// Imports: public import Init public meta import Init public import Mathlib.Data.Fin.Basic public import Mathlib.Data.Set.Operations
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCongr___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCongr___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_finCongr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finCongr___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finCongr___closed__0 = (const lean_object*)&lp_mathlib_finCongr___closed__0_value;
static const lean_ctor_object lp_mathlib_finCongr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_finCongr___closed__0_value),((lean_object*)&lp_mathlib_finCongr___closed__0_value)}};
static const lean_object* lp_mathlib_finCongr___closed__1 = (const lean_object*)&lp_mathlib_finCongr___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_finCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCongr___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castPred___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castPred___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castPred(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castPred___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAbove___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAbove___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAbove(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAbove___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAbove___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAbove___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAbove(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAbove___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finCongr___lam__0(lean_object* v___y_1_){
_start:
{
lean_inc(v___y_1_);
return v___y_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finCongr___lam__0___boxed(lean_object* v___y_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_finCongr___lam__0(v___y_2_);
lean_dec(v___y_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finCongr(lean_object* v_n_7_, lean_object* v_m_8_, lean_object* v_eq_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = ((lean_object*)(lp_mathlib_finCongr___closed__1));
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finCongr___boxed(lean_object* v_n_11_, lean_object* v_m_12_, lean_object* v_eq_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_finCongr(v_n_11_, v_m_12_, v_eq_13_);
lean_dec(v_m_12_);
lean_dec(v_n_11_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castPred___redArg(lean_object* v_i_15_){
_start:
{
lean_inc(v_i_15_);
return v_i_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castPred___redArg___boxed(lean_object* v_i_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Fin_castPred___redArg(v_i_16_);
lean_dec(v_i_16_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castPred(lean_object* v_n_18_, lean_object* v_i_19_, lean_object* v_h_20_){
_start:
{
lean_inc(v_i_19_);
return v_i_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castPred___boxed(lean_object* v_n_21_, lean_object* v_i_22_, lean_object* v_h_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Fin_castPred(v_n_21_, v_i_22_, v_h_23_);
lean_dec(v_i_22_);
lean_dec(v_n_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAbove___redArg(lean_object* v_p_25_, lean_object* v_i_26_){
_start:
{
uint8_t v___x_27_; 
v___x_27_ = lean_nat_dec_lt(v_i_26_, v_p_25_);
if (v___x_27_ == 0)
{
lean_object* v___x_28_; 
v___x_28_ = l_Fin_succ___redArg(v_i_26_);
return v___x_28_;
}
else
{
lean_inc(v_i_26_);
return v_i_26_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAbove___redArg___boxed(lean_object* v_p_29_, lean_object* v_i_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Fin_succAbove___redArg(v_p_29_, v_i_30_);
lean_dec(v_i_30_);
lean_dec(v_p_29_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAbove(lean_object* v_n_32_, lean_object* v_p_33_, lean_object* v_i_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_Fin_succAbove___redArg(v_p_33_, v_i_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAbove___boxed(lean_object* v_n_36_, lean_object* v_p_37_, lean_object* v_i_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Fin_succAbove(v_n_36_, v_p_37_, v_i_38_);
lean_dec(v_i_38_);
lean_dec(v_p_37_);
lean_dec(v_n_36_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAbove___redArg(lean_object* v_p_40_, lean_object* v_i_41_){
_start:
{
uint8_t v___x_42_; 
v___x_42_ = lean_nat_dec_lt(v_p_40_, v_i_41_);
if (v___x_42_ == 0)
{
lean_inc(v_i_41_);
return v_i_41_;
}
else
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = lean_unsigned_to_nat(1u);
v___x_44_ = lean_nat_sub(v_i_41_, v___x_43_);
return v___x_44_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAbove___redArg___boxed(lean_object* v_p_45_, lean_object* v_i_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Fin_predAbove___redArg(v_p_45_, v_i_46_);
lean_dec(v_i_46_);
lean_dec(v_p_45_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAbove(lean_object* v_n_48_, lean_object* v_p_49_, lean_object* v_i_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_Fin_predAbove___redArg(v_p_49_, v_i_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAbove___boxed(lean_object* v_n_52_, lean_object* v_p_53_, lean_object* v_i_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_Fin_predAbove(v_n_52_, v_p_53_, v_i_54_);
lean_dec(v_i_54_);
lean_dec(v_p_53_);
lean_dec(v_n_52_);
return v_res_55_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_SuccPred(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fin_SuccPred(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fin_SuccPred(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fin_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fin_SuccPred(builtin);
}
#ifdef __cplusplus
}
#endif
