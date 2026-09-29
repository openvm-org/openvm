// Lean compiler output
// Module: Mathlib.Data.Multiset.Powerset
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Sublists public import Mathlib.Data.List.Zip public import Mathlib.Data.Multiset.Bind public import Mathlib.Data.Multiset.Range
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
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_batteries_List_sublistsFast___redArg(lean_object*);
lean_object* lp_mathlib_List_sublistsLenAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_sublists_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powerset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powerset(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Multiset_powersetCardAux___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_powersetCardAux___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_powersetCardAux___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCard___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCard(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(lean_object* v_a_1_, lean_object* v_a_2_){
_start:
{
if (lean_obj_tag(v_a_1_) == 0)
{
lean_object* v___x_3_; 
v___x_3_ = l_List_reverse___redArg(v_a_2_);
return v___x_3_;
}
else
{
lean_object* v_head_4_; lean_object* v_tail_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_13_; 
v_head_4_ = lean_ctor_get(v_a_1_, 0);
v_tail_5_ = lean_ctor_get(v_a_1_, 1);
v_isSharedCheck_13_ = !lean_is_exclusive(v_a_1_);
if (v_isSharedCheck_13_ == 0)
{
v___x_7_ = v_a_1_;
v_isShared_8_ = v_isSharedCheck_13_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_tail_5_);
lean_inc(v_head_4_);
lean_dec(v_a_1_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_13_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
lean_object* v___x_10_; 
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 1, v_a_2_);
v___x_10_ = v___x_7_;
goto v_reusejp_9_;
}
else
{
lean_object* v_reuseFailAlloc_12_; 
v_reuseFailAlloc_12_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_12_, 0, v_head_4_);
lean_ctor_set(v_reuseFailAlloc_12_, 1, v_a_2_);
v___x_10_ = v_reuseFailAlloc_12_;
goto v_reusejp_9_;
}
v_reusejp_9_:
{
v_a_1_ = v_tail_5_;
v_a_2_ = v___x_10_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux___redArg(lean_object* v_l_14_){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_15_ = lp_batteries_List_sublistsFast___redArg(v_l_14_);
v___x_16_ = lean_box(0);
v___x_17_ = lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(v___x_15_, v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux(lean_object* v_00_u03b1_18_, lean_object* v_l_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_Multiset_powersetAux___redArg(v_l_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0(lean_object* v_00_u03b1_21_, lean_object* v_a_22_, lean_object* v_a_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(v_a_22_, v_a_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux_x27___redArg(lean_object* v_l_25_){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_26_ = lp_batteries_List_sublists_x27___redArg(v_l_25_);
v___x_27_ = lean_box(0);
v___x_28_ = lp_mathlib_List_mapTR_loop___at___00Multiset_powersetAux_spec__0___redArg(v___x_26_, v___x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetAux_x27(lean_object* v_00_u03b1_29_, lean_object* v_l_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_Multiset_powersetAux_x27___redArg(v_l_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powerset___redArg(lean_object* v_s_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Multiset_powersetAux___redArg(v_s_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powerset(lean_object* v_00_u03b1_34_, lean_object* v_s_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Multiset_powersetAux___redArg(v_s_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___lam__0(lean_object* v___y_37_){
_start:
{
lean_inc(v___y_37_);
return v___y_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg___lam__0___boxed(lean_object* v___y_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Multiset_powersetCardAux___redArg___lam__0(v___y_38_);
lean_dec(v___y_38_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux___redArg(lean_object* v_n_41_, lean_object* v_l_42_){
_start:
{
lean_object* v___f_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___f_43_ = ((lean_object*)(lp_mathlib_Multiset_powersetCardAux___redArg___closed__0));
v___x_44_ = lean_box(0);
v___x_45_ = lp_mathlib_List_sublistsLenAux___redArg(v_n_41_, v_l_42_, v___f_43_, v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCardAux(lean_object* v_00_u03b1_46_, lean_object* v_n_47_, lean_object* v_l_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Multiset_powersetCardAux___redArg(v_n_47_, v_l_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCard___redArg(lean_object* v_n_50_, lean_object* v_s_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_Multiset_powersetCardAux___redArg(v_n_50_, v_s_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_powersetCard(lean_object* v_00_u03b1_53_, lean_object* v_n_54_, lean_object* v_s_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Multiset_powersetCardAux___redArg(v_n_54_, v_s_55_);
return v___x_56_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Sublists(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Zip(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Range(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Powerset(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Sublists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Zip(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Powerset(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Sublists(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Zip(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Range(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Powerset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Sublists(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Zip(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Powerset(builtin);
}
#ifdef __cplusplus
}
#endif
