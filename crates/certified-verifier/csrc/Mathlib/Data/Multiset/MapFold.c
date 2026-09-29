// Lean compiler output
// Module: Mathlib.Data.Multiset.MapFold
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Perm.Basic public import Mathlib.Data.Multiset.Replicate public import Mathlib.Data.Set.List
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
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_map_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_map_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_foldl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_foldl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_foldr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_foldr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_map_spec__0___redArg(lean_object* v_f_1_, lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_4_; 
lean_dec(v_f_1_);
v___x_4_ = l_List_reverse___redArg(v_a_3_);
return v___x_4_;
}
else
{
lean_object* v_head_5_; lean_object* v_tail_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_15_; 
v_head_5_ = lean_ctor_get(v_a_2_, 0);
v_tail_6_ = lean_ctor_get(v_a_2_, 1);
v_isSharedCheck_15_ = !lean_is_exclusive(v_a_2_);
if (v_isSharedCheck_15_ == 0)
{
v___x_8_ = v_a_2_;
v_isShared_9_ = v_isSharedCheck_15_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_tail_6_);
lean_inc(v_head_5_);
lean_dec(v_a_2_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_15_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___x_10_; lean_object* v___x_12_; 
lean_inc(v_f_1_);
v___x_10_ = lean_apply_1(v_f_1_, v_head_5_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v_a_3_);
lean_ctor_set(v___x_8_, 0, v___x_10_);
v___x_12_ = v___x_8_;
goto v_reusejp_11_;
}
else
{
lean_object* v_reuseFailAlloc_14_; 
v_reuseFailAlloc_14_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_14_, 0, v___x_10_);
lean_ctor_set(v_reuseFailAlloc_14_, 1, v_a_3_);
v___x_12_ = v_reuseFailAlloc_14_;
goto v_reusejp_11_;
}
v_reusejp_11_:
{
v_a_2_ = v_tail_6_;
v_a_3_ = v___x_12_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_map___redArg(lean_object* v_f_16_, lean_object* v_s_17_){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_18_ = lean_box(0);
v___x_19_ = lp_mathlib_List_mapTR_loop___at___00Multiset_map_spec__0___redArg(v_f_16_, v_s_17_, v___x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_map(lean_object* v_00_u03b1_20_, lean_object* v_00_u03b2_21_, lean_object* v_f_22_, lean_object* v_s_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Multiset_map___redArg(v_f_22_, v_s_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_map_spec__0(lean_object* v_00_u03b1_25_, lean_object* v_00_u03b2_26_, lean_object* v_f_27_, lean_object* v_a_28_, lean_object* v_a_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_List_mapTR_loop___at___00Multiset_map_spec__0___redArg(v_f_27_, v_a_28_, v_a_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_foldl___redArg(lean_object* v_f_31_, lean_object* v_b_32_, lean_object* v_s_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = l_List_foldl___redArg(v_f_31_, v_b_32_, v_s_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_foldl(lean_object* v_00_u03b1_35_, lean_object* v_00_u03b2_36_, lean_object* v_f_37_, lean_object* v_inst_38_, lean_object* v_b_39_, lean_object* v_s_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = l_List_foldl___redArg(v_f_37_, v_b_39_, v_s_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_foldr___redArg(lean_object* v_f_42_, lean_object* v_b_43_, lean_object* v_s_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = l_List_foldrTR___redArg(v_f_42_, v_b_43_, v_s_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_foldr(lean_object* v_00_u03b1_46_, lean_object* v_00_u03b2_47_, lean_object* v_f_48_, lean_object* v_inst_49_, lean_object* v_b_50_, lean_object* v_s_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = l_List_foldrTR___redArg(v_f_48_, v_b_50_, v_s_51_);
return v___x_52_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Replicate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_List(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_MapFold(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Replicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_MapFold(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Replicate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_List(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_MapFold(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Replicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_MapFold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_MapFold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_MapFold(builtin);
}
#ifdef __cplusplus
}
#endif
