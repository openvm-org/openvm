// Lean compiler output
// Module: Mathlib.Data.Fintype.EquivFin
// Imports: public import Init public meta import Init public import Mathlib.Data.Fintype.Card public import Mathlib.Data.List.NodupEquivFin
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
lean_object* lp_mathlib_Fin_castLEEmb___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_finCongr(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFin(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncFinBijection___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncFinBijection(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFinOfCardEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFinOfCardEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFinOfCardEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFinOfCardEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivOfCardEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivOfCardEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverseOfCardLE___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverseOfCardLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverseOfCardLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofRightInverseOfCardLE___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofRightInverseOfCardLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofRightInverseOfCardLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_truncOfCardLE___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_castLEEmb___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_truncOfCardLE___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_truncOfCardLE___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_truncOfCardLE___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_truncOfCardLE___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_truncOfCardLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFin___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lp_mathlib_List_Nodup_getEquivOfForallMemList___redArg(v_inst_1_, v_inst_2_);
v___x_4_ = lp_mathlib_Equiv_symm___redArg(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFin(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Fintype_truncEquivFin___redArg(v_inst_6_, v_inst_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncFinBijection___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncFinBijection(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___f_13_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_List_Nodup_getBijectionOfForallMemList___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_13_, 0, v_inst_12_);
return v___f_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFinOfCardEq___redArg(lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_n_16_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
lean_inc(v_inst_14_);
v___x_17_ = lp_mathlib_Fintype_truncEquivFin___redArg(v_inst_15_, v_inst_14_);
v___x_18_ = l_List_lengthTR___redArg(v_inst_14_);
lean_dec(v_inst_14_);
v___x_19_ = lp_mathlib_finCongr(v___x_18_, v_n_16_, lean_box(0));
lean_dec(v___x_18_);
v___x_20_ = lp_mathlib_Equiv_trans___redArg(v___x_17_, v___x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFinOfCardEq___redArg___boxed(lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_n_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Fintype_truncEquivFinOfCardEq___redArg(v_inst_21_, v_inst_22_, v_n_23_);
lean_dec(v_n_23_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFinOfCardEq(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_n_28_, lean_object* v_h_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_Fintype_truncEquivFinOfCardEq___redArg(v_inst_26_, v_inst_27_, v_n_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivFinOfCardEq___boxed(lean_object* v_00_u03b1_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_n_34_, lean_object* v_h_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Fintype_truncEquivFinOfCardEq(v_00_u03b1_31_, v_inst_32_, v_inst_33_, v_n_34_, v_h_35_);
lean_dec(v_n_34_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivOfCardEq___redArg(lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_41_ = l_List_lengthTR___redArg(v_inst_38_);
v___x_42_ = lp_mathlib_Fintype_truncEquivFinOfCardEq___redArg(v_inst_37_, v_inst_39_, v___x_41_);
lean_dec(v___x_41_);
v___x_43_ = lp_mathlib_Fintype_truncEquivFin___redArg(v_inst_40_, v_inst_38_);
v___x_44_ = lp_mathlib_Equiv_symm___redArg(v___x_43_);
v___x_45_ = lp_mathlib_Equiv_trans___redArg(v___x_42_, v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncEquivOfCardEq(lean_object* v_00_u03b1_46_, lean_object* v_00_u03b2_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_h_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_Fintype_truncEquivOfCardEq___redArg(v_inst_48_, v_inst_49_, v_inst_50_, v_inst_51_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverseOfCardLE___redArg(lean_object* v_f_54_, lean_object* v_g_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_56_, 0, v_f_54_);
lean_ctor_set(v___x_56_, 1, v_g_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverseOfCardLE(lean_object* v_00_u03b1_57_, lean_object* v_00_u03b2_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_h_u03b2_u03b1_61_, lean_object* v_f_62_, lean_object* v_g_63_, lean_object* v_h_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_65_, 0, v_f_62_);
lean_ctor_set(v___x_65_, 1, v_g_63_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofLeftInverseOfCardLE___boxed(lean_object* v_00_u03b1_66_, lean_object* v_00_u03b2_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_h_u03b2_u03b1_70_, lean_object* v_f_71_, lean_object* v_g_72_, lean_object* v_h_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_Equiv_ofLeftInverseOfCardLE(v_00_u03b1_66_, v_00_u03b2_67_, v_inst_68_, v_inst_69_, v_h_u03b2_u03b1_70_, v_f_71_, v_g_72_, v_h_73_);
lean_dec(v_inst_69_);
lean_dec(v_inst_68_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofRightInverseOfCardLE___redArg(lean_object* v_f_75_, lean_object* v_g_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_77_, 0, v_f_75_);
lean_ctor_set(v___x_77_, 1, v_g_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofRightInverseOfCardLE(lean_object* v_00_u03b1_78_, lean_object* v_00_u03b2_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_h_u03b1_u03b2_82_, lean_object* v_f_83_, lean_object* v_g_84_, lean_object* v_h_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_86_, 0, v_f_83_);
lean_ctor_set(v___x_86_, 1, v_g_84_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofRightInverseOfCardLE___boxed(lean_object* v_00_u03b1_87_, lean_object* v_00_u03b2_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_h_u03b1_u03b2_91_, lean_object* v_f_92_, lean_object* v_g_93_, lean_object* v_h_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Equiv_ofRightInverseOfCardLE(v_00_u03b1_87_, v_00_u03b2_88_, v_inst_89_, v_inst_90_, v_h_u03b1_u03b2_91_, v_f_92_, v_g_93_, v_h_94_);
lean_dec(v_inst_90_);
lean_dec(v_inst_89_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_truncOfCardLE___redArg___lam__0(lean_object* v___x_97_, lean_object* v___x_98_, lean_object* v___y_99_){
_start:
{
lean_object* v___f_100_; lean_object* v___f_101_; lean_object* v___x_102_; lean_object* v___f_103_; lean_object* v___f_104_; lean_object* v___x_105_; 
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_100_, 0, v___x_97_);
v___f_101_ = ((lean_object*)(lp_mathlib_Function_Embedding_truncOfCardLE___redArg___lam__0___closed__0));
v___x_102_ = lp_mathlib_Equiv_symm___redArg(v___x_98_);
v___f_103_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_103_, 0, v___x_102_);
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_104_, 0, v___f_101_);
lean_closure_set(v___f_104_, 1, v___f_103_);
v___x_105_ = lp_mathlib_Function_Embedding_trans___redArg___lam__0(v___f_100_, v___f_104_, v___y_99_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_truncOfCardLE___redArg(lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___f_112_; 
v___x_110_ = lp_mathlib_Fintype_truncEquivFin___redArg(v_inst_108_, v_inst_106_);
v___x_111_ = lp_mathlib_Fintype_truncEquivFin___redArg(v_inst_109_, v_inst_107_);
v___f_112_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_truncOfCardLE___redArg___lam__0), 3, 2);
lean_closure_set(v___f_112_, 0, v___x_110_);
lean_closure_set(v___f_112_, 1, v___x_111_);
return v___f_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_truncOfCardLE(lean_object* v_00_u03b1_113_, lean_object* v_00_u03b2_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_h_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Function_Embedding_truncOfCardLE___redArg(v_inst_115_, v_inst_116_, v_inst_117_, v_inst_118_);
return v___x_120_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_NodupEquivFin(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_NodupEquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_NodupEquivFin(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_NodupEquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
}
#ifdef __cplusplus
}
#endif
