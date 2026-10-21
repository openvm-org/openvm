// Lean compiler output
// Module: Mathlib.Data.List.OffDiag
// Imports: public import Init public meta import Init import Mathlib.Data.List.Count import Mathlib.Data.List.Enum import Mathlib.Data.List.Nodup import Mathlib.Data.List.Perm.Basic public import Mathlib.Data.Nat.Notation
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_List_zipIdxTR___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_eraseIdxTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_offDiag_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_offDiag___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_offDiag___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_offDiag___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_offDiag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_offDiag(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_offDiag_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_offDiag_spec__0___redArg(lean_object* v_fst_1_, lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_4_; 
lean_dec(v_fst_1_);
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
lean_inc(v_fst_1_);
v___x_10_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_10_, 0, v_fst_1_);
lean_ctor_set(v___x_10_, 1, v_head_5_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1___redArg(lean_object* v_l_18_, lean_object* v_a_19_, lean_object* v_a_20_){
_start:
{
if (lean_obj_tag(v_a_19_) == 0)
{
lean_object* v___x_21_; 
lean_dec(v_l_18_);
v___x_21_ = lean_array_to_list(v_a_20_);
return v___x_21_;
}
else
{
lean_object* v_head_22_; lean_object* v_tail_23_; lean_object* v_fst_24_; lean_object* v_snd_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v_head_22_ = lean_ctor_get(v_a_19_, 0);
lean_inc(v_head_22_);
v_tail_23_ = lean_ctor_get(v_a_19_, 1);
lean_inc(v_tail_23_);
lean_dec_ref_known(v_a_19_, 2);
v_fst_24_ = lean_ctor_get(v_head_22_, 0);
lean_inc(v_fst_24_);
v_snd_25_ = lean_ctor_get(v_head_22_, 1);
lean_inc(v_snd_25_);
lean_dec(v_head_22_);
v___x_26_ = ((lean_object*)(lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1___redArg___closed__0));
lean_inc(v_l_18_);
v___x_27_ = l___private_Init_Data_List_Impl_0__List_eraseIdxTR_go(lean_box(0), v_l_18_, v_l_18_, v_snd_25_, v___x_26_);
v___x_28_ = lean_box(0);
v___x_29_ = lp_mathlib_List_mapTR_loop___at___00List_offDiag_spec__0___redArg(v_fst_24_, v___x_27_, v___x_28_);
v___x_30_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_20_, v___x_29_);
v_a_19_ = v_tail_23_;
v_a_20_ = v___x_30_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_offDiag___redArg(lean_object* v_l_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_35_ = lean_unsigned_to_nat(0u);
lean_inc(v_l_34_);
v___x_36_ = l_List_zipIdxTR___redArg(v_l_34_, v___x_35_);
v___x_37_ = ((lean_object*)(lp_mathlib_List_offDiag___redArg___closed__0));
v___x_38_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1___redArg(v_l_34_, v___x_36_, v___x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_offDiag(lean_object* v_00_u03b1_39_, lean_object* v_l_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_List_offDiag___redArg(v_l_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_offDiag_spec__0(lean_object* v_00_u03b1_42_, lean_object* v_fst_43_, lean_object* v_a_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_List_mapTR_loop___at___00List_offDiag_spec__0___redArg(v_fst_43_, v_a_44_, v_a_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1(lean_object* v_00_u03b1_47_, lean_object* v_l_48_, lean_object* v_a_49_, lean_object* v_a_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_offDiag_spec__1___redArg(v_l_48_, v_a_49_, v_a_50_);
return v___x_51_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Count(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Enum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_OffDiag(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Enum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_OffDiag(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Count(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Enum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_OffDiag(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Enum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Perm_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_OffDiag(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_OffDiag(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_OffDiag(builtin);
}
#ifdef __cplusplus
}
#endif
