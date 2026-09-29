// Lean compiler output
// Module: Mathlib.Data.Fintype.Card
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Fintype.Basic
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
lean_object* lp_mathlib_Equiv_equivEmptyEquiv(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0;
static lean_once_cell_t lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1;
static lean_once_cell_t lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = l_List_lengthTR___redArg(v_inst_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___redArg___boxed(lean_object* v_inst_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_Fintype_card___redArg(v_inst_3_);
lean_dec(v_inst_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_List_lengthTR___redArg(v_inst_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_card___boxed(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Fintype_card(v_00_u03b1_8_, v_inst_9_);
lean_dec(v_inst_9_);
return v_res_10_;
}
}
static lean_object* _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0(void){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_Equiv_equivEmptyEquiv(lean_box(0));
return v___x_11_;
}
}
static lean_object* _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1(void){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_obj_once(&lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0, &lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0_once, _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__0);
v___x_13_ = lp_mathlib_Equiv_symm___redArg(v___x_12_);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_14_ = lean_obj_once(&lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1, &lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1_once, _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__1);
v___x_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_15_, 0, lean_box(0));
lean_ctor_set(v___x_15_, 1, lean_box(0));
v___x_16_ = lp_mathlib_Equiv_trans___redArg(v___x_15_, v___x_14_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_obj_once(&lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2, &lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2_once, _init_lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___closed__2);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty___boxed(lean_object* v_00_u03b1_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Fintype_cardEqZeroEquivEquivEmpty(v_00_u03b1_20_, v_inst_21_);
lean_dec(v_inst_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v_head_24_; 
v_head_24_ = lean_ctor_get(v_inst_23_, 0);
lean_inc(v_head_24_);
return v_head_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___redArg___boxed(lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_truncOfCardPos___redArg(v_inst_25_);
lean_dec(v_inst_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos(lean_object* v_00_u03b1_27_, lean_object* v_inst_28_, lean_object* v_h_29_){
_start:
{
lean_object* v_head_30_; 
v_head_30_ = lean_ctor_get(v_inst_28_, 0);
lean_inc(v_head_30_);
return v_head_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_truncOfCardPos___boxed(lean_object* v_00_u03b1_31_, lean_object* v_inst_32_, lean_object* v_h_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_truncOfCardPos(v_00_u03b1_31_, v_inst_32_, v_h_33_);
lean_dec(v_inst_32_);
return v_res_34_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
}
#ifdef __cplusplus
}
#endif
