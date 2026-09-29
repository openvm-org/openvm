// Lean compiler output
// Module: Mathlib.Data.Fintype.Sigma
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Sigma public import Mathlib.Data.Fintype.OfMap
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
lean_object* lp_mathlib_Equiv_psigmaEquivSigma(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sigma___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instFintype___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instFintype(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_PSigma_instFintype___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PSigma_instFintype___redArg___closed__0;
static lean_once_cell_t lp_mathlib_PSigma_instFintype___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PSigma_instFintype___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_PSigma_instFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_instFintype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instFintype___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_x_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instFintype___redArg(lean_object* v_inst_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; lean_object* v___x_7_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Sigma_instFintype___redArg___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_inst_4_);
v___x_7_ = lp_mathlib_Finset_sigma___redArg(v_inst_5_, v___f_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instFintype(lean_object* v_00_u03b9_8_, lean_object* v_00_u03ba_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Sigma_instFintype___redArg(v_inst_10_, v_inst_11_);
return v___x_12_;
}
}
static lean_object* _init_lp_mathlib_PSigma_instFintype___redArg___closed__0(void){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Equiv_psigmaEquivSigma(lean_box(0), lean_box(0));
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_PSigma_instFintype___redArg___closed__1(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_obj_once(&lp_mathlib_PSigma_instFintype___redArg___closed__0, &lp_mathlib_PSigma_instFintype___redArg___closed__0_once, _init_lp_mathlib_PSigma_instFintype___redArg___closed__0);
v___x_15_ = lp_mathlib_Equiv_symm___redArg(v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_instFintype___redArg(lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_18_ = lp_mathlib_Sigma_instFintype___redArg(v_inst_16_, v_inst_17_);
v___x_19_ = lean_obj_once(&lp_mathlib_PSigma_instFintype___redArg___closed__1, &lp_mathlib_PSigma_instFintype___redArg___closed__1_once, _init_lp_mathlib_PSigma_instFintype___redArg___closed__1);
v___x_20_ = lp_mathlib_Fintype_ofEquiv___redArg(v___x_18_, v___x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_instFintype(lean_object* v_00_u03b9_21_, lean_object* v_00_u03ba_22_, lean_object* v_inst_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_PSigma_instFintype___redArg(v_inst_23_, v_inst_24_);
return v___x_25_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Sigma(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Sigma(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sigma(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Sigma(builtin);
}
#ifdef __cplusplus
}
#endif
