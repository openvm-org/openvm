// Lean compiler output
// Module: Mathlib.Data.Fin.Embedding
// Imports: public import Init public meta import Init public import Mathlib.Data.Fin.SuccPred public import Mathlib.Logic.Embedding.Basic
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
lean_object* l_Fin_succ___boxed(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_finCongr(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_natAdd___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_succAbove___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fin_valEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_valEmbedding___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_valEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Fin_valEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succEmb(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fin_castLEEmb___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_castLEEmb___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_castLEEmb___closed__0 = (const lean_object*)&lp_mathlib_Fin_castLEEmb___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddEmb___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccEmb(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccEmb___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAddEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___lam__0(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___lam__0___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Fin_valEmbedding___lam__0(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding(lean_object* v_n_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = ((lean_object*)(lp_mathlib_Fin_valEmbedding___closed__0));
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_valEmbedding___boxed(lean_object* v_n_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Fin_valEmbedding(v_n_7_);
lean_dec(v_n_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succEmb(lean_object* v_n_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_alloc_closure((void*)(l_Fin_succ___boxed), 2, 1);
lean_closure_set(v___x_10_, 0, v_n_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___lam__0(lean_object* v___y_11_){
_start:
{
lean_inc(v___y_11_);
return v___y_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___lam__0___boxed(lean_object* v___y_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Fin_castLEEmb___lam__0(v___y_12_);
lean_dec(v___y_12_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb(lean_object* v_n_15_, lean_object* v_m_16_, lean_object* v_h_17_){
_start:
{
lean_object* v___f_18_; 
v___f_18_ = ((lean_object*)(lp_mathlib_Fin_castLEEmb___closed__0));
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEEmb___boxed(lean_object* v_n_19_, lean_object* v_m_20_, lean_object* v_h_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Fin_castLEEmb(v_n_19_, v_m_20_, v_h_21_);
lean_dec(v_m_20_);
lean_dec(v_n_19_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddEmb(lean_object* v_n_23_, lean_object* v_m_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = ((lean_object*)(lp_mathlib_Fin_castLEEmb___closed__0));
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddEmb___boxed(lean_object* v_n_26_, lean_object* v_m_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Fin_castAddEmb(v_n_26_, v_m_27_);
lean_dec(v_m_27_);
lean_dec(v_n_26_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccEmb(lean_object* v_n_29_){
_start:
{
lean_object* v___f_30_; 
v___f_30_ = ((lean_object*)(lp_mathlib_Fin_castLEEmb___closed__0));
return v___f_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccEmb___boxed(lean_object* v_n_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Fin_castSuccEmb(v_n_31_);
lean_dec(v_n_31_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg___lam__0(lean_object* v_m_33_, lean_object* v_x_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_nat_add(v_x_34_, v_m_33_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed(lean_object* v_m_36_, lean_object* v_x_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Fin_addNatEmb___redArg___lam__0(v_m_36_, v_x_37_);
lean_dec(v_x_37_);
lean_dec(v_m_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___redArg(lean_object* v_m_39_){
_start:
{
lean_object* v___f_40_; 
v___f_40_ = lean_alloc_closure((void*)(lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_40_, 0, v_m_39_);
return v___f_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb(lean_object* v_n_41_, lean_object* v_m_42_){
_start:
{
lean_object* v___f_43_; 
v___f_43_ = lean_alloc_closure((void*)(lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_43_, 0, v_m_42_);
return v___f_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatEmb___boxed(lean_object* v_n_44_, lean_object* v_m_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Fin_addNatEmb(v_n_44_, v_m_45_);
lean_dec(v_n_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAddEmb(lean_object* v_n_47_, lean_object* v_m_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_49_, 0, v_m_48_);
lean_closure_set(v___x_49_, 1, v_n_47_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveEmb(lean_object* v_n_50_, lean_object* v_p_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_alloc_closure((void*)(lp_mathlib_Fin_succAbove___boxed), 3, 2);
lean_closure_set(v___x_52_, 0, v_n_50_);
lean_closure_set(v___x_52_, 1, v_p_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___redArg(lean_object* v_n_53_, lean_object* v_m_54_){
_start:
{
lean_object* v___x_55_; lean_object* v___f_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___f_59_; lean_object* v___f_60_; 
v___x_55_ = lean_nat_sub(v_m_54_, v_n_53_);
lean_inc(v___x_55_);
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_Fin_addNatEmb___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_56_, 0, v___x_55_);
v___x_57_ = lean_nat_add(v_n_53_, v___x_55_);
lean_dec(v___x_55_);
v___x_58_ = lp_mathlib_finCongr(v___x_57_, v_m_54_, lean_box(0));
lean_dec(v___x_57_);
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_59_, 0, v___x_58_);
v___f_60_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_60_, 0, v___f_56_);
lean_closure_set(v___f_60_, 1, v___f_59_);
return v___f_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___redArg___boxed(lean_object* v_n_61_, lean_object* v_m_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Fin_natAdd__castLEEmb___redArg(v_n_61_, v_m_62_);
lean_dec(v_m_62_);
lean_dec(v_n_61_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb(lean_object* v_n_64_, lean_object* v_m_65_, lean_object* v_hmn_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_Fin_natAdd__castLEEmb___redArg(v_n_64_, v_m_65_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAdd__castLEEmb___boxed(lean_object* v_n_68_, lean_object* v_m_69_, lean_object* v_hmn_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_Fin_natAdd__castLEEmb(v_n_68_, v_m_69_, v_hmn_70_);
lean_dec(v_m_69_);
lean_dec(v_n_68_);
return v_res_71_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_SuccPred(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Embedding(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fin_Embedding(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fin_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fin_Embedding(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fin_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fin_Embedding(builtin);
}
#ifdef __cplusplus
}
#endif
