// Lean compiler output
// Module: Mathlib.Data.List.Range
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Chain
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
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_ranges(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Range_0__List_ranges_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Range_0__List_ranges_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__0(lean_object* v_head_1_, lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_4_; 
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
v___x_10_ = lean_nat_add(v_head_1_, v_head_5_);
lean_dec(v_head_5_);
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
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__0___boxed(lean_object* v_head_16_, lean_object* v_a_17_, lean_object* v_a_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__0(v_head_16_, v_a_17_, v_a_18_);
lean_dec(v_head_16_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__1(lean_object* v_head_20_, lean_object* v_a_21_, lean_object* v_a_22_){
_start:
{
if (lean_obj_tag(v_a_21_) == 0)
{
lean_object* v___x_23_; 
v___x_23_ = l_List_reverse___redArg(v_a_22_);
return v___x_23_;
}
else
{
lean_object* v_head_24_; lean_object* v_tail_25_; lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_35_; 
v_head_24_ = lean_ctor_get(v_a_21_, 0);
v_tail_25_ = lean_ctor_get(v_a_21_, 1);
v_isSharedCheck_35_ = !lean_is_exclusive(v_a_21_);
if (v_isSharedCheck_35_ == 0)
{
v___x_27_ = v_a_21_;
v_isShared_28_ = v_isSharedCheck_35_;
goto v_resetjp_26_;
}
else
{
lean_inc(v_tail_25_);
lean_inc(v_head_24_);
lean_dec(v_a_21_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_35_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_32_; 
v___x_29_ = lean_box(0);
v___x_30_ = lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__0(v_head_20_, v_head_24_, v___x_29_);
if (v_isShared_28_ == 0)
{
lean_ctor_set(v___x_27_, 1, v_a_22_);
lean_ctor_set(v___x_27_, 0, v___x_30_);
v___x_32_ = v___x_27_;
goto v_reusejp_31_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v___x_30_);
lean_ctor_set(v_reuseFailAlloc_34_, 1, v_a_22_);
v___x_32_ = v_reuseFailAlloc_34_;
goto v_reusejp_31_;
}
v_reusejp_31_:
{
v_a_21_ = v_tail_25_;
v_a_22_ = v___x_32_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__1___boxed(lean_object* v_head_36_, lean_object* v_a_37_, lean_object* v_a_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__1(v_head_36_, v_a_37_, v_a_38_);
lean_dec(v_head_36_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_ranges(lean_object* v_x_40_){
_start:
{
if (lean_obj_tag(v_x_40_) == 0)
{
lean_object* v___x_41_; 
v___x_41_ = lean_box(0);
return v___x_41_;
}
else
{
lean_object* v_head_42_; lean_object* v_tail_43_; lean_object* v___x_45_; uint8_t v_isShared_46_; uint8_t v_isSharedCheck_54_; 
v_head_42_ = lean_ctor_get(v_x_40_, 0);
v_tail_43_ = lean_ctor_get(v_x_40_, 1);
v_isSharedCheck_54_ = !lean_is_exclusive(v_x_40_);
if (v_isSharedCheck_54_ == 0)
{
v___x_45_ = v_x_40_;
v_isShared_46_ = v_isSharedCheck_54_;
goto v_resetjp_44_;
}
else
{
lean_inc(v_tail_43_);
lean_inc(v_head_42_);
lean_dec(v_x_40_);
v___x_45_ = lean_box(0);
v_isShared_46_ = v_isSharedCheck_54_;
goto v_resetjp_44_;
}
v_resetjp_44_:
{
lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_52_; 
lean_inc(v_head_42_);
v___x_47_ = l_List_range(v_head_42_);
v___x_48_ = lp_mathlib_List_ranges(v_tail_43_);
v___x_49_ = lean_box(0);
v___x_50_ = lp_mathlib_List_mapTR_loop___at___00List_ranges_spec__1(v_head_42_, v___x_48_, v___x_49_);
lean_dec(v_head_42_);
if (v_isShared_46_ == 0)
{
lean_ctor_set(v___x_45_, 1, v___x_50_);
lean_ctor_set(v___x_45_, 0, v___x_47_);
v___x_52_ = v___x_45_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_53_; 
v_reuseFailAlloc_53_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_53_, 0, v___x_47_);
lean_ctor_set(v_reuseFailAlloc_53_, 1, v___x_50_);
v___x_52_ = v_reuseFailAlloc_53_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
return v___x_52_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Range_0__List_ranges_match__1_splitter___redArg(lean_object* v_x_55_, lean_object* v_h__1_56_, lean_object* v_h__2_57_){
_start:
{
if (lean_obj_tag(v_x_55_) == 0)
{
lean_object* v___x_58_; lean_object* v___x_59_; 
lean_dec(v_h__2_57_);
v___x_58_ = lean_box(0);
v___x_59_ = lean_apply_1(v_h__1_56_, v___x_58_);
return v___x_59_;
}
else
{
lean_object* v_head_60_; lean_object* v_tail_61_; lean_object* v___x_62_; 
lean_dec(v_h__1_56_);
v_head_60_ = lean_ctor_get(v_x_55_, 0);
lean_inc(v_head_60_);
v_tail_61_ = lean_ctor_get(v_x_55_, 1);
lean_inc(v_tail_61_);
lean_dec_ref_known(v_x_55_, 2);
v___x_62_ = lean_apply_2(v_h__2_57_, v_head_60_, v_tail_61_);
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Range_0__List_ranges_match__1_splitter(lean_object* v_motive_63_, lean_object* v_x_64_, lean_object* v_h__1_65_, lean_object* v_h__2_66_){
_start:
{
if (lean_obj_tag(v_x_64_) == 0)
{
lean_object* v___x_67_; lean_object* v___x_68_; 
lean_dec(v_h__2_66_);
v___x_67_ = lean_box(0);
v___x_68_ = lean_apply_1(v_h__1_65_, v___x_67_);
return v___x_68_;
}
else
{
lean_object* v_head_69_; lean_object* v_tail_70_; lean_object* v___x_71_; 
lean_dec(v_h__1_65_);
v_head_69_ = lean_ctor_get(v_x_64_, 0);
lean_inc(v_head_69_);
v_tail_70_ = lean_ctor_get(v_x_64_, 1);
lean_inc(v_tail_70_);
lean_dec_ref_known(v_x_64_, 2);
v___x_71_ = lean_apply_2(v_h__2_66_, v_head_69_, v_tail_70_);
return v___x_71_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Chain(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Range(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Range(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Chain(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Range(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Range(builtin);
}
#ifdef __cplusplus
}
#endif
