// Lean compiler output
// Module: Aesop.Tree.Stats
// Imports: public import Init public meta import Init public import Aesop.Tree.TreeM import Aesop.Tree.Traversal
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
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getRootMetaState___redArg(lean_object*);
lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getDecl(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lp_aesop_Aesop_ForwardState_stats(lean_object*);
uint8_t lp_aesop_Aesop_Goal_isNormal(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1___closed__0;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stats___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stats___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stats(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stats___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__1(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__2(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__0(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_collectGoalStatsIfEnabled___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_collectGoalStatsIfEnabled___closed__0 = (const lean_object*)&lp_aesop_Aesop_collectGoalStatsIfEnabled___closed__0_value;
static const lean_string_object lp_aesop_Aesop_collectGoalStatsIfEnabled___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_collectGoalStatsIfEnabled___closed__1 = (const lean_object*)&lp_aesop_Aesop_collectGoalStatsIfEnabled___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_collectGoalStatsIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_collectGoalStatsIfEnabled___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(lean_object* v_as_1_, size_t v_i_2_, size_t v_stop_3_, lean_object* v_b_4_){
_start:
{
lean_object* v___y_6_; uint8_t v___x_10_; 
v___x_10_ = lean_usize_dec_eq(v_i_2_, v_stop_3_);
if (v___x_10_ == 0)
{
lean_object* v___x_11_; 
v___x_11_ = lean_array_uget_borrowed(v_as_1_, v_i_2_);
if (lean_obj_tag(v___x_11_) == 0)
{
v___y_6_ = v_b_4_;
goto v___jp_5_;
}
else
{
lean_object* v_val_12_; uint8_t v___x_13_; 
v_val_12_ = lean_ctor_get(v___x_11_, 0);
v___x_13_ = l_Lean_LocalDecl_isImplementationDetail(v_val_12_);
if (v___x_13_ == 0)
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_unsigned_to_nat(1u);
v___x_15_ = lean_nat_add(v_b_4_, v___x_14_);
lean_dec(v_b_4_);
v___y_6_ = v___x_15_;
goto v___jp_5_;
}
else
{
v___y_6_ = v_b_4_;
goto v___jp_5_;
}
}
}
else
{
return v_b_4_;
}
v___jp_5_:
{
size_t v___x_7_; size_t v___x_8_; 
v___x_7_ = ((size_t)1ULL);
v___x_8_ = lean_usize_add(v_i_2_, v___x_7_);
v_i_2_ = v___x_8_;
v_b_4_ = v___y_6_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2___boxed(lean_object* v_as_16_, lean_object* v_i_17_, lean_object* v_stop_18_, lean_object* v_b_19_){
_start:
{
size_t v_i_boxed_20_; size_t v_stop_boxed_21_; lean_object* v_res_22_; 
v_i_boxed_20_ = lean_unbox_usize(v_i_17_);
lean_dec(v_i_17_);
v_stop_boxed_21_ = lean_unbox_usize(v_stop_18_);
lean_dec(v_stop_18_);
v_res_22_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_as_16_, v_i_boxed_20_, v_stop_boxed_21_, v_b_19_);
lean_dec_ref(v_as_16_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__3(lean_object* v_x_23_, lean_object* v_x_24_){
_start:
{
if (lean_obj_tag(v_x_23_) == 0)
{
lean_object* v_cs_25_; lean_object* v___x_26_; lean_object* v___x_27_; uint8_t v___x_28_; 
v_cs_25_ = lean_ctor_get(v_x_23_, 0);
v___x_26_ = lean_unsigned_to_nat(0u);
v___x_27_ = lean_array_get_size(v_cs_25_);
v___x_28_ = lean_nat_dec_lt(v___x_26_, v___x_27_);
if (v___x_28_ == 0)
{
return v_x_24_;
}
else
{
uint8_t v___x_29_; 
v___x_29_ = lean_nat_dec_le(v___x_27_, v___x_27_);
if (v___x_29_ == 0)
{
if (v___x_28_ == 0)
{
return v_x_24_;
}
else
{
size_t v___x_30_; size_t v___x_31_; lean_object* v___x_32_; 
v___x_30_ = ((size_t)0ULL);
v___x_31_ = lean_usize_of_nat(v___x_27_);
v___x_32_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2(v_cs_25_, v___x_30_, v___x_31_, v_x_24_);
return v___x_32_;
}
}
else
{
size_t v___x_33_; size_t v___x_34_; lean_object* v___x_35_; 
v___x_33_ = ((size_t)0ULL);
v___x_34_ = lean_usize_of_nat(v___x_27_);
v___x_35_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2(v_cs_25_, v___x_33_, v___x_34_, v_x_24_);
return v___x_35_;
}
}
}
else
{
lean_object* v_vs_36_; lean_object* v___x_37_; lean_object* v___x_38_; uint8_t v___x_39_; 
v_vs_36_ = lean_ctor_get(v_x_23_, 0);
v___x_37_ = lean_unsigned_to_nat(0u);
v___x_38_ = lean_array_get_size(v_vs_36_);
v___x_39_ = lean_nat_dec_lt(v___x_37_, v___x_38_);
if (v___x_39_ == 0)
{
return v_x_24_;
}
else
{
uint8_t v___x_40_; 
v___x_40_ = lean_nat_dec_le(v___x_38_, v___x_38_);
if (v___x_40_ == 0)
{
if (v___x_39_ == 0)
{
return v_x_24_;
}
else
{
size_t v___x_41_; size_t v___x_42_; lean_object* v___x_43_; 
v___x_41_ = ((size_t)0ULL);
v___x_42_ = lean_usize_of_nat(v___x_38_);
v___x_43_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_vs_36_, v___x_41_, v___x_42_, v_x_24_);
return v___x_43_;
}
}
else
{
size_t v___x_44_; size_t v___x_45_; lean_object* v___x_46_; 
v___x_44_ = ((size_t)0ULL);
v___x_45_ = lean_usize_of_nat(v___x_38_);
v___x_46_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_vs_36_, v___x_44_, v___x_45_, v_x_24_);
return v___x_46_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2(lean_object* v_as_47_, size_t v_i_48_, size_t v_stop_49_, lean_object* v_b_50_){
_start:
{
uint8_t v___x_51_; 
v___x_51_ = lean_usize_dec_eq(v_i_48_, v_stop_49_);
if (v___x_51_ == 0)
{
lean_object* v___x_52_; lean_object* v___x_53_; size_t v___x_54_; size_t v___x_55_; 
v___x_52_ = lean_array_uget_borrowed(v_as_47_, v_i_48_);
v___x_53_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__3(v___x_52_, v_b_50_);
v___x_54_ = ((size_t)1ULL);
v___x_55_ = lean_usize_add(v_i_48_, v___x_54_);
v_i_48_ = v___x_55_;
v_b_50_ = v___x_53_;
goto _start;
}
else
{
return v_b_50_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_as_57_, lean_object* v_i_58_, lean_object* v_stop_59_, lean_object* v_b_60_){
_start:
{
size_t v_i_boxed_61_; size_t v_stop_boxed_62_; lean_object* v_res_63_; 
v_i_boxed_61_ = lean_unbox_usize(v_i_58_);
lean_dec(v_i_58_);
v_stop_boxed_62_ = lean_unbox_usize(v_stop_59_);
lean_dec(v_stop_59_);
v_res_63_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2(v_as_57_, v_i_boxed_61_, v_stop_boxed_62_, v_b_60_);
lean_dec_ref(v_as_57_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__3___boxed(lean_object* v_x_64_, lean_object* v_x_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__3(v_x_64_, v_x_65_);
lean_dec_ref(v_x_64_);
return v_res_66_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1___closed__0(void){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1(lean_object* v_x_68_, size_t v_x_69_, size_t v_x_70_, lean_object* v_x_71_){
_start:
{
if (lean_obj_tag(v_x_68_) == 0)
{
lean_object* v_cs_72_; lean_object* v___x_73_; size_t v___x_74_; lean_object* v_j_75_; lean_object* v___x_76_; size_t v___x_77_; size_t v___x_78_; size_t v___x_79_; size_t v___x_80_; size_t v___x_81_; size_t v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; uint8_t v___x_87_; 
v_cs_72_ = lean_ctor_get(v_x_68_, 0);
v___x_73_ = lean_obj_once(&lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1___closed__0, &lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1___closed__0_once, _init_lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1___closed__0);
v___x_74_ = lean_usize_shift_right(v_x_69_, v_x_70_);
v_j_75_ = lean_usize_to_nat(v___x_74_);
v___x_76_ = lean_array_get_borrowed(v___x_73_, v_cs_72_, v_j_75_);
v___x_77_ = ((size_t)1ULL);
v___x_78_ = lean_usize_shift_left(v___x_77_, v_x_70_);
v___x_79_ = lean_usize_sub(v___x_78_, v___x_77_);
v___x_80_ = lean_usize_land(v_x_69_, v___x_79_);
v___x_81_ = ((size_t)5ULL);
v___x_82_ = lean_usize_sub(v_x_70_, v___x_81_);
v___x_83_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1(v___x_76_, v___x_80_, v___x_82_, v_x_71_);
v___x_84_ = lean_unsigned_to_nat(1u);
v___x_85_ = lean_nat_add(v_j_75_, v___x_84_);
lean_dec(v_j_75_);
v___x_86_ = lean_array_get_size(v_cs_72_);
v___x_87_ = lean_nat_dec_lt(v___x_85_, v___x_86_);
if (v___x_87_ == 0)
{
lean_dec(v___x_85_);
return v___x_83_;
}
else
{
uint8_t v___x_88_; 
v___x_88_ = lean_nat_dec_le(v___x_86_, v___x_86_);
if (v___x_88_ == 0)
{
if (v___x_87_ == 0)
{
lean_dec(v___x_85_);
return v___x_83_;
}
else
{
size_t v___x_89_; size_t v___x_90_; lean_object* v___x_91_; 
v___x_89_ = lean_usize_of_nat(v___x_85_);
lean_dec(v___x_85_);
v___x_90_ = lean_usize_of_nat(v___x_86_);
v___x_91_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2(v_cs_72_, v___x_89_, v___x_90_, v___x_83_);
return v___x_91_;
}
}
else
{
size_t v___x_92_; size_t v___x_93_; lean_object* v___x_94_; 
v___x_92_ = lean_usize_of_nat(v___x_85_);
lean_dec(v___x_85_);
v___x_93_ = lean_usize_of_nat(v___x_86_);
v___x_94_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1_spec__2(v_cs_72_, v___x_92_, v___x_93_, v___x_83_);
return v___x_94_;
}
}
}
else
{
lean_object* v_vs_95_; lean_object* v___x_96_; lean_object* v___x_97_; uint8_t v___x_98_; 
v_vs_95_ = lean_ctor_get(v_x_68_, 0);
v___x_96_ = lean_usize_to_nat(v_x_69_);
v___x_97_ = lean_array_get_size(v_vs_95_);
v___x_98_ = lean_nat_dec_lt(v___x_96_, v___x_97_);
if (v___x_98_ == 0)
{
lean_dec(v___x_96_);
return v_x_71_;
}
else
{
uint8_t v___x_99_; 
v___x_99_ = lean_nat_dec_le(v___x_97_, v___x_97_);
if (v___x_99_ == 0)
{
if (v___x_98_ == 0)
{
lean_dec(v___x_96_);
return v_x_71_;
}
else
{
size_t v___x_100_; size_t v___x_101_; lean_object* v___x_102_; 
v___x_100_ = lean_usize_of_nat(v___x_96_);
lean_dec(v___x_96_);
v___x_101_ = lean_usize_of_nat(v___x_97_);
v___x_102_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_vs_95_, v___x_100_, v___x_101_, v_x_71_);
return v___x_102_;
}
}
else
{
size_t v___x_103_; size_t v___x_104_; lean_object* v___x_105_; 
v___x_103_ = lean_usize_of_nat(v___x_96_);
lean_dec(v___x_96_);
v___x_104_ = lean_usize_of_nat(v___x_97_);
v___x_105_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_vs_95_, v___x_103_, v___x_104_, v_x_71_);
return v___x_105_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1___boxed(lean_object* v_x_106_, lean_object* v_x_107_, lean_object* v_x_108_, lean_object* v_x_109_){
_start:
{
size_t v_x_3552__boxed_110_; size_t v_x_3553__boxed_111_; lean_object* v_res_112_; 
v_x_3552__boxed_110_ = lean_unbox_usize(v_x_107_);
lean_dec(v_x_107_);
v_x_3553__boxed_111_ = lean_unbox_usize(v_x_108_);
lean_dec(v_x_108_);
v_res_112_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1(v_x_106_, v_x_3552__boxed_110_, v_x_3553__boxed_111_, v_x_109_);
lean_dec_ref(v_x_106_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0(lean_object* v_t_113_, lean_object* v_init_114_, lean_object* v_start_115_){
_start:
{
lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_116_ = lean_unsigned_to_nat(0u);
v___x_117_ = lean_nat_dec_eq(v_start_115_, v___x_116_);
if (v___x_117_ == 0)
{
lean_object* v_root_118_; lean_object* v_tail_119_; size_t v_shift_120_; lean_object* v_tailOff_121_; uint8_t v___x_122_; 
v_root_118_ = lean_ctor_get(v_t_113_, 0);
v_tail_119_ = lean_ctor_get(v_t_113_, 1);
v_shift_120_ = lean_ctor_get_usize(v_t_113_, 4);
v_tailOff_121_ = lean_ctor_get(v_t_113_, 3);
v___x_122_ = lean_nat_dec_le(v_tailOff_121_, v_start_115_);
if (v___x_122_ == 0)
{
size_t v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_123_ = lean_usize_of_nat(v_start_115_);
v___x_124_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__1(v_root_118_, v___x_123_, v_shift_120_, v_init_114_);
v___x_125_ = lean_array_get_size(v_tail_119_);
v___x_126_ = lean_nat_dec_lt(v___x_116_, v___x_125_);
if (v___x_126_ == 0)
{
return v___x_124_;
}
else
{
uint8_t v___x_127_; 
v___x_127_ = lean_nat_dec_le(v___x_125_, v___x_125_);
if (v___x_127_ == 0)
{
if (v___x_126_ == 0)
{
return v___x_124_;
}
else
{
size_t v___x_128_; size_t v___x_129_; lean_object* v___x_130_; 
v___x_128_ = ((size_t)0ULL);
v___x_129_ = lean_usize_of_nat(v___x_125_);
v___x_130_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_tail_119_, v___x_128_, v___x_129_, v___x_124_);
return v___x_130_;
}
}
else
{
size_t v___x_131_; size_t v___x_132_; lean_object* v___x_133_; 
v___x_131_ = ((size_t)0ULL);
v___x_132_ = lean_usize_of_nat(v___x_125_);
v___x_133_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_tail_119_, v___x_131_, v___x_132_, v___x_124_);
return v___x_133_;
}
}
}
else
{
lean_object* v___x_134_; lean_object* v___x_135_; uint8_t v___x_136_; 
v___x_134_ = lean_nat_sub(v_start_115_, v_tailOff_121_);
v___x_135_ = lean_array_get_size(v_tail_119_);
v___x_136_ = lean_nat_dec_lt(v___x_134_, v___x_135_);
if (v___x_136_ == 0)
{
lean_dec(v___x_134_);
return v_init_114_;
}
else
{
uint8_t v___x_137_; 
v___x_137_ = lean_nat_dec_le(v___x_135_, v___x_135_);
if (v___x_137_ == 0)
{
if (v___x_136_ == 0)
{
lean_dec(v___x_134_);
return v_init_114_;
}
else
{
size_t v___x_138_; size_t v___x_139_; lean_object* v___x_140_; 
v___x_138_ = lean_usize_of_nat(v___x_134_);
lean_dec(v___x_134_);
v___x_139_ = lean_usize_of_nat(v___x_135_);
v___x_140_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_tail_119_, v___x_138_, v___x_139_, v_init_114_);
return v___x_140_;
}
}
else
{
size_t v___x_141_; size_t v___x_142_; lean_object* v___x_143_; 
v___x_141_ = lean_usize_of_nat(v___x_134_);
lean_dec(v___x_134_);
v___x_142_ = lean_usize_of_nat(v___x_135_);
v___x_143_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_tail_119_, v___x_141_, v___x_142_, v_init_114_);
return v___x_143_;
}
}
}
}
else
{
lean_object* v_root_144_; lean_object* v_tail_145_; lean_object* v___x_146_; lean_object* v___x_147_; uint8_t v___x_148_; 
v_root_144_ = lean_ctor_get(v_t_113_, 0);
v_tail_145_ = lean_ctor_get(v_t_113_, 1);
v___x_146_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__3(v_root_144_, v_init_114_);
v___x_147_ = lean_array_get_size(v_tail_145_);
v___x_148_ = lean_nat_dec_lt(v___x_116_, v___x_147_);
if (v___x_148_ == 0)
{
return v___x_146_;
}
else
{
uint8_t v___x_149_; 
v___x_149_ = lean_nat_dec_le(v___x_147_, v___x_147_);
if (v___x_149_ == 0)
{
if (v___x_148_ == 0)
{
return v___x_146_;
}
else
{
size_t v___x_150_; size_t v___x_151_; lean_object* v___x_152_; 
v___x_150_ = ((size_t)0ULL);
v___x_151_ = lean_usize_of_nat(v___x_147_);
v___x_152_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_tail_145_, v___x_150_, v___x_151_, v___x_146_);
return v___x_152_;
}
}
else
{
size_t v___x_153_; size_t v___x_154_; lean_object* v___x_155_; 
v___x_153_ = ((size_t)0ULL);
v___x_154_ = lean_usize_of_nat(v___x_147_);
v___x_155_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0_spec__2(v_tail_145_, v___x_153_, v___x_154_, v___x_146_);
return v___x_155_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0___boxed(lean_object* v_t_156_, lean_object* v_init_157_, lean_object* v_start_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0(v_t_156_, v_init_157_, v_start_158_);
lean_dec(v_start_158_);
lean_dec_ref(v_t_156_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0(lean_object* v_lctx_160_, lean_object* v_init_161_, lean_object* v_start_162_){
_start:
{
lean_object* v_decls_163_; lean_object* v___x_164_; 
v_decls_163_ = lean_ctor_get(v_lctx_160_, 1);
v___x_164_ = lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0_spec__0(v_decls_163_, v_init_161_, v_start_162_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0___boxed(lean_object* v_lctx_165_, lean_object* v_init_166_, lean_object* v_start_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_aesop_Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0(v_lctx_165_, v_init_166_, v_start_167_);
lean_dec(v_start_167_);
lean_dec_ref(v_lctx_165_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stats___redArg(lean_object* v_g_169_, lean_object* v_a_170_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_aesop_Aesop_getRootMetaState___redArg(v_a_170_);
if (lean_obj_tag(v___x_172_) == 0)
{
lean_object* v_a_173_; lean_object* v___x_174_; 
v_a_173_ = lean_ctor_get(v___x_172_, 0);
lean_inc(v_a_173_);
lean_dec_ref_known(v___x_172_, 1);
lean_inc(v_g_169_);
v___x_174_ = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(v_g_169_, v_a_173_);
lean_dec(v_a_173_);
if (lean_obj_tag(v___x_174_) == 0)
{
lean_object* v_a_175_; lean_object* v___x_177_; uint8_t v_isShared_178_; uint8_t v_isSharedCheck_203_; 
v_a_175_ = lean_ctor_get(v___x_174_, 0);
v_isSharedCheck_203_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_203_ == 0)
{
v___x_177_ = v___x_174_;
v_isShared_178_ = v_isSharedCheck_203_;
goto v_resetjp_176_;
}
else
{
lean_inc(v_a_175_);
lean_dec(v___x_174_);
v___x_177_ = lean_box(0);
v_isShared_178_ = v_isSharedCheck_203_;
goto v_resetjp_176_;
}
v_resetjp_176_:
{
lean_object* v_snd_179_; lean_object* v_meta_180_; lean_object* v_fst_181_; lean_object* v_mctx_182_; lean_object* v___x_183_; lean_object* v_lctx_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v_elimGoal_188_; lean_object* v___x_189_; lean_object* v_id_190_; lean_object* v_depth_191_; lean_object* v_forwardState_192_; uint8_t v___y_194_; uint8_t v___x_200_; 
v_snd_179_ = lean_ctor_get(v_a_175_, 1);
v_meta_180_ = lean_ctor_get(v_snd_179_, 1);
lean_inc_ref(v_meta_180_);
v_fst_181_ = lean_ctor_get(v_a_175_, 0);
lean_inc(v_fst_181_);
lean_dec(v_a_175_);
v_mctx_182_ = lean_ctor_get(v_meta_180_, 0);
lean_inc_ref(v_mctx_182_);
lean_dec_ref(v_meta_180_);
v___x_183_ = l_Lean_MetavarContext_getDecl(v_mctx_182_, v_fst_181_);
lean_dec_ref(v_mctx_182_);
v_lctx_184_ = lean_ctor_get(v___x_183_, 1);
lean_inc_ref(v_lctx_184_);
lean_dec_ref(v___x_183_);
v___x_185_ = lean_unsigned_to_nat(0u);
v___x_186_ = lp_aesop_Lean_LocalContext_foldlM___at___00Aesop_Goal_stats_spec__0(v_lctx_184_, v___x_185_, v___x_185_);
lean_dec_ref(v_lctx_184_);
v___x_187_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_188_ = lean_ctor_get(v___x_187_, 1);
lean_inc_ref(v_elimGoal_188_);
lean_inc(v_g_169_);
v___x_189_ = lean_apply_1(v_elimGoal_188_, v_g_169_);
v_id_190_ = lean_ctor_get(v___x_189_, 0);
lean_inc(v_id_190_);
v_depth_191_ = lean_ctor_get(v___x_189_, 4);
lean_inc(v_depth_191_);
v_forwardState_192_ = lean_ctor_get(v___x_189_, 8);
lean_inc_ref(v_forwardState_192_);
lean_dec_ref(v___x_189_);
v___x_200_ = lp_aesop_Aesop_Goal_isNormal(v_g_169_);
if (v___x_200_ == 0)
{
uint8_t v___x_201_; 
v___x_201_ = 0;
v___y_194_ = v___x_201_;
goto v___jp_193_;
}
else
{
uint8_t v___x_202_; 
v___x_202_ = 1;
v___y_194_ = v___x_202_;
goto v___jp_193_;
}
v___jp_193_:
{
lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_198_; 
v___x_195_ = lp_aesop_Aesop_ForwardState_stats(v_forwardState_192_);
lean_dec_ref(v_forwardState_192_);
v___x_196_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_196_, 0, v_id_190_);
lean_ctor_set(v___x_196_, 1, v___x_186_);
lean_ctor_set(v___x_196_, 2, v_depth_191_);
lean_ctor_set(v___x_196_, 3, v___x_195_);
lean_ctor_set_uint8(v___x_196_, sizeof(void*)*4, v___y_194_);
if (v_isShared_178_ == 0)
{
lean_ctor_set(v___x_177_, 0, v___x_196_);
v___x_198_ = v___x_177_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v___x_196_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
}
else
{
lean_object* v_a_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_211_; 
lean_dec(v_g_169_);
v_a_204_ = lean_ctor_get(v___x_174_, 0);
v_isSharedCheck_211_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_211_ == 0)
{
v___x_206_ = v___x_174_;
v_isShared_207_ = v_isSharedCheck_211_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_a_204_);
lean_dec(v___x_174_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_211_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_209_; 
if (v_isShared_207_ == 0)
{
v___x_209_ = v___x_206_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v_a_204_);
v___x_209_ = v_reuseFailAlloc_210_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
return v___x_209_;
}
}
}
}
else
{
lean_object* v_a_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_219_; 
lean_dec(v_g_169_);
v_a_212_ = lean_ctor_get(v___x_172_, 0);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_172_);
if (v_isSharedCheck_219_ == 0)
{
v___x_214_ = v___x_172_;
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_a_212_);
lean_dec(v___x_172_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___x_217_; 
if (v_isShared_215_ == 0)
{
v___x_217_ = v___x_214_;
goto v_reusejp_216_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_a_212_);
v___x_217_ = v_reuseFailAlloc_218_;
goto v_reusejp_216_;
}
v_reusejp_216_:
{
return v___x_217_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stats___redArg___boxed(lean_object* v_g_220_, lean_object* v_a_221_, lean_object* v_a_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_aesop_Aesop_Goal_stats___redArg(v_g_220_, v_a_221_);
lean_dec(v_a_221_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stats(lean_object* v_g_224_, lean_object* v_a_225_, lean_object* v_a_226_, lean_object* v_a_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_aesop_Aesop_Goal_stats___redArg(v_g_224_, v_a_226_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stats___boxed(lean_object* v_g_234_, lean_object* v_a_235_, lean_object* v_a_236_, lean_object* v_a_237_, lean_object* v_a_238_, lean_object* v_a_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_aesop_Aesop_Goal_stats(v_g_234_, v_a_235_, v_a_236_, v_a_237_, v_a_238_, v_a_239_, v_a_240_, v_a_241_);
lean_dec(v_a_241_);
lean_dec_ref(v_a_240_);
lean_dec(v_a_239_);
lean_dec_ref(v_a_238_);
lean_dec(v_a_237_);
lean_dec(v_a_236_);
lean_dec_ref(v_a_235_);
return v_res_243_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__1(lean_object* v_opts_244_, lean_object* v_opt_245_){
_start:
{
lean_object* v_name_246_; lean_object* v_defValue_247_; lean_object* v_map_248_; lean_object* v___x_249_; 
v_name_246_ = lean_ctor_get(v_opt_245_, 0);
v_defValue_247_ = lean_ctor_get(v_opt_245_, 1);
v_map_248_ = lean_ctor_get(v_opts_244_, 0);
v___x_249_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_248_, v_name_246_);
if (lean_obj_tag(v___x_249_) == 0)
{
uint8_t v___x_250_; 
v___x_250_ = lean_unbox(v_defValue_247_);
return v___x_250_;
}
else
{
lean_object* v_val_251_; 
v_val_251_ = lean_ctor_get(v___x_249_, 0);
lean_inc(v_val_251_);
lean_dec_ref_known(v___x_249_, 1);
if (lean_obj_tag(v_val_251_) == 1)
{
uint8_t v_v_252_; 
v_v_252_ = lean_ctor_get_uint8(v_val_251_, 0);
lean_dec_ref_known(v_val_251_, 0);
return v_v_252_;
}
else
{
uint8_t v___x_253_; 
lean_dec(v_val_251_);
v___x_253_ = lean_unbox(v_defValue_247_);
return v___x_253_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__1___boxed(lean_object* v_opts_254_, lean_object* v_opt_255_){
_start:
{
uint8_t v_res_256_; lean_object* v_r_257_; 
v_res_256_ = lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__1(v_opts_254_, v_opt_255_);
lean_dec_ref(v_opt_255_);
lean_dec_ref(v_opts_254_);
v_r_257_ = lean_box(v_res_256_);
return v_r_257_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__3(lean_object* v_opts_258_, lean_object* v_opt_259_){
_start:
{
lean_object* v_name_260_; lean_object* v_defValue_261_; lean_object* v_map_262_; lean_object* v___x_263_; 
v_name_260_ = lean_ctor_get(v_opt_259_, 0);
v_defValue_261_ = lean_ctor_get(v_opt_259_, 1);
v_map_262_ = lean_ctor_get(v_opts_258_, 0);
v___x_263_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_262_, v_name_260_);
if (lean_obj_tag(v___x_263_) == 0)
{
lean_inc(v_defValue_261_);
return v_defValue_261_;
}
else
{
lean_object* v_val_264_; 
v_val_264_ = lean_ctor_get(v___x_263_, 0);
lean_inc(v_val_264_);
lean_dec_ref_known(v___x_263_, 1);
if (lean_obj_tag(v_val_264_) == 0)
{
lean_object* v_v_265_; 
v_v_265_ = lean_ctor_get(v_val_264_, 0);
lean_inc_ref(v_v_265_);
lean_dec_ref_known(v_val_264_, 1);
return v_v_265_;
}
else
{
lean_dec(v_val_264_);
lean_inc(v_defValue_261_);
return v_defValue_261_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__3___boxed(lean_object* v_opts_266_, lean_object* v_opt_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__3(v_opts_266_, v_opt_267_);
lean_dec_ref(v_opt_267_);
lean_dec_ref(v_opts_266_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___redArg(lean_object* v_opt_269_, lean_object* v___y_270_){
_start:
{
lean_object* v_options_272_; lean_object* v_option_273_; uint8_t v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v_options_272_ = lean_ctor_get(v___y_270_, 2);
v_option_273_ = lean_ctor_get(v_opt_269_, 1);
v___x_274_ = lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__1(v_options_272_, v_option_273_);
v___x_275_ = lean_box(v___x_274_);
v___x_276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_276_, 0, v___x_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___redArg___boxed(lean_object* v_opt_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___redArg(v_opt_277_, v___y_278_);
lean_dec_ref(v___y_278_);
lean_dec_ref(v_opt_277_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__1(uint8_t v_a_281_, lean_object* v_as_282_, size_t v_i_283_, size_t v_stop_284_, lean_object* v_b_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_){
_start:
{
uint8_t v___x_295_; 
v___x_295_ = lean_usize_dec_eq(v_i_283_, v_stop_284_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_296_ = lean_array_uget_borrowed(v_as_282_, v_i_283_);
lean_inc(v___x_296_);
v___x_297_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
v___x_298_ = lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0(v_a_281_, v___x_297_, v___y_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_);
if (lean_obj_tag(v___x_298_) == 0)
{
lean_object* v_a_299_; size_t v___x_300_; size_t v___x_301_; 
v_a_299_ = lean_ctor_get(v___x_298_, 0);
lean_inc(v_a_299_);
lean_dec_ref_known(v___x_298_, 1);
v___x_300_ = ((size_t)1ULL);
v___x_301_ = lean_usize_add(v_i_283_, v___x_300_);
v_i_283_ = v___x_301_;
v_b_285_ = v_a_299_;
goto _start;
}
else
{
return v___x_298_;
}
}
else
{
lean_object* v___x_303_; 
v___x_303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_303_, 0, v_b_285_);
return v___x_303_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__2(uint8_t v_a_304_, lean_object* v_as_305_, size_t v_i_306_, size_t v_stop_307_, lean_object* v_b_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
uint8_t v___x_318_; 
v___x_318_ = lean_usize_dec_eq(v_i_306_, v_stop_307_);
if (v___x_318_ == 0)
{
lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_319_ = lean_array_uget_borrowed(v_as_305_, v_i_306_);
lean_inc(v___x_319_);
v___x_320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_320_, 0, v___x_319_);
v___x_321_ = lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0(v_a_304_, v___x_320_, v___y_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_);
if (lean_obj_tag(v___x_321_) == 0)
{
lean_object* v_a_322_; size_t v___x_323_; size_t v___x_324_; 
v_a_322_ = lean_ctor_get(v___x_321_, 0);
lean_inc(v_a_322_);
lean_dec_ref_known(v___x_321_, 1);
v___x_323_ = ((size_t)1ULL);
v___x_324_ = lean_usize_add(v_i_306_, v___x_323_);
v_i_306_ = v___x_324_;
v_b_308_ = v_a_322_;
goto _start;
}
else
{
return v___x_321_;
}
}
else
{
lean_object* v___x_326_; 
v___x_326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_326_, 0, v_b_308_);
return v___x_326_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0(uint8_t v_a_327_, lean_object* v_x_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v_gref_339_; lean_object* v___y_340_; lean_object* v___y_341_; 
switch(lean_obj_tag(v_x_328_))
{
case 0:
{
if (v_a_327_ == 0)
{
lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_377_; 
v_isSharedCheck_377_ = !lean_is_exclusive(v_x_328_);
if (v_isSharedCheck_377_ == 0)
{
lean_object* v_unused_378_; 
v_unused_378_ = lean_ctor_get(v_x_328_, 0);
lean_dec(v_unused_378_);
v___x_371_ = v_x_328_;
v_isShared_372_ = v_isSharedCheck_377_;
goto v_resetjp_370_;
}
else
{
lean_dec(v_x_328_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_377_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v___x_373_; lean_object* v___x_375_; 
v___x_373_ = lean_box(0);
if (v_isShared_372_ == 0)
{
lean_ctor_set(v___x_371_, 0, v___x_373_);
v___x_375_ = v___x_371_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v___x_373_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
else
{
lean_object* v_gref_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v_elimGoal_382_; lean_object* v___x_383_; lean_object* v_children_384_; lean_object* v___x_385_; lean_object* v___x_386_; uint8_t v___x_387_; 
v_gref_379_ = lean_ctor_get(v_x_328_, 0);
lean_inc(v_gref_379_);
lean_dec_ref_known(v_x_328_, 1);
v___x_380_ = lean_st_ref_get(v_gref_379_);
v___x_381_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_382_ = lean_ctor_get(v___x_381_, 1);
lean_inc_ref(v_elimGoal_382_);
v___x_383_ = lean_apply_1(v_elimGoal_382_, v___x_380_);
v_children_384_ = lean_ctor_get(v___x_383_, 2);
lean_inc_ref(v_children_384_);
lean_dec_ref(v___x_383_);
v___x_385_ = lean_unsigned_to_nat(0u);
v___x_386_ = lean_array_get_size(v_children_384_);
v___x_387_ = lean_nat_dec_lt(v___x_385_, v___x_386_);
if (v___x_387_ == 0)
{
lean_dec_ref(v_children_384_);
v_gref_339_ = v_gref_379_;
v___y_340_ = v___y_329_;
v___y_341_ = v___y_331_;
goto v___jp_338_;
}
else
{
lean_object* v___x_388_; uint8_t v___x_389_; 
v___x_388_ = lean_box(0);
v___x_389_ = lean_nat_dec_le(v___x_386_, v___x_386_);
if (v___x_389_ == 0)
{
if (v___x_387_ == 0)
{
lean_dec_ref(v_children_384_);
v_gref_339_ = v_gref_379_;
v___y_340_ = v___y_329_;
v___y_341_ = v___y_331_;
goto v___jp_338_;
}
else
{
size_t v___x_390_; size_t v___x_391_; lean_object* v___x_392_; 
v___x_390_ = ((size_t)0ULL);
v___x_391_ = lean_usize_of_nat(v___x_386_);
v___x_392_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__0(v_a_327_, v_children_384_, v___x_390_, v___x_391_, v___x_388_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec_ref(v_children_384_);
if (lean_obj_tag(v___x_392_) == 0)
{
lean_dec_ref_known(v___x_392_, 1);
v_gref_339_ = v_gref_379_;
v___y_340_ = v___y_329_;
v___y_341_ = v___y_331_;
goto v___jp_338_;
}
else
{
lean_dec(v_gref_379_);
return v___x_392_;
}
}
}
else
{
size_t v___x_393_; size_t v___x_394_; lean_object* v___x_395_; 
v___x_393_ = ((size_t)0ULL);
v___x_394_ = lean_usize_of_nat(v___x_386_);
v___x_395_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__0(v_a_327_, v_children_384_, v___x_393_, v___x_394_, v___x_388_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec_ref(v_children_384_);
if (lean_obj_tag(v___x_395_) == 0)
{
lean_dec_ref_known(v___x_395_, 1);
v_gref_339_ = v_gref_379_;
v___y_340_ = v___y_329_;
v___y_341_ = v___y_331_;
goto v___jp_338_;
}
else
{
lean_dec(v_gref_379_);
return v___x_395_;
}
}
}
}
}
case 1:
{
if (v_a_327_ == 0)
{
lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_403_; 
v_isSharedCheck_403_ = !lean_is_exclusive(v_x_328_);
if (v_isSharedCheck_403_ == 0)
{
lean_object* v_unused_404_; 
v_unused_404_ = lean_ctor_get(v_x_328_, 0);
lean_dec(v_unused_404_);
v___x_397_ = v_x_328_;
v_isShared_398_ = v_isSharedCheck_403_;
goto v_resetjp_396_;
}
else
{
lean_dec(v_x_328_);
v___x_397_ = lean_box(0);
v_isShared_398_ = v_isSharedCheck_403_;
goto v_resetjp_396_;
}
v_resetjp_396_:
{
lean_object* v___x_399_; lean_object* v___x_401_; 
v___x_399_ = lean_box(0);
if (v_isShared_398_ == 0)
{
lean_ctor_set_tag(v___x_397_, 0);
lean_ctor_set(v___x_397_, 0, v___x_399_);
v___x_401_ = v___x_397_;
goto v_reusejp_400_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v___x_399_);
v___x_401_ = v_reuseFailAlloc_402_;
goto v_reusejp_400_;
}
v_reusejp_400_:
{
return v___x_401_;
}
}
}
else
{
lean_object* v_rref_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v_elimRapp_408_; lean_object* v___x_409_; lean_object* v_children_410_; lean_object* v___x_411_; lean_object* v___x_412_; uint8_t v___x_413_; 
v_rref_405_ = lean_ctor_get(v_x_328_, 0);
lean_inc(v_rref_405_);
lean_dec_ref_known(v_x_328_, 1);
v___x_406_ = lean_st_ref_get(v_rref_405_);
lean_dec(v_rref_405_);
v___x_407_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_408_ = lean_ctor_get(v___x_407_, 3);
lean_inc_ref(v_elimRapp_408_);
v___x_409_ = lean_apply_1(v_elimRapp_408_, v___x_406_);
v_children_410_ = lean_ctor_get(v___x_409_, 2);
lean_inc_ref(v_children_410_);
lean_dec_ref(v___x_409_);
v___x_411_ = lean_unsigned_to_nat(0u);
v___x_412_ = lean_array_get_size(v_children_410_);
v___x_413_ = lean_nat_dec_lt(v___x_411_, v___x_412_);
if (v___x_413_ == 0)
{
lean_dec_ref(v_children_410_);
goto v___jp_364_;
}
else
{
lean_object* v___x_414_; uint8_t v___x_415_; 
v___x_414_ = lean_box(0);
v___x_415_ = lean_nat_dec_le(v___x_412_, v___x_412_);
if (v___x_415_ == 0)
{
if (v___x_413_ == 0)
{
lean_dec_ref(v_children_410_);
goto v___jp_364_;
}
else
{
size_t v___x_416_; size_t v___x_417_; lean_object* v___x_418_; 
v___x_416_ = ((size_t)0ULL);
v___x_417_ = lean_usize_of_nat(v___x_412_);
v___x_418_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__1(v_a_327_, v_children_410_, v___x_416_, v___x_417_, v___x_414_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec_ref(v_children_410_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_dec_ref_known(v___x_418_, 1);
goto v___jp_364_;
}
else
{
return v___x_418_;
}
}
}
else
{
size_t v___x_419_; size_t v___x_420_; lean_object* v___x_421_; 
v___x_419_ = ((size_t)0ULL);
v___x_420_ = lean_usize_of_nat(v___x_412_);
v___x_421_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__1(v_a_327_, v_children_410_, v___x_419_, v___x_420_, v___x_414_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec_ref(v_children_410_);
if (lean_obj_tag(v___x_421_) == 0)
{
lean_dec_ref_known(v___x_421_, 1);
goto v___jp_364_;
}
else
{
return v___x_421_;
}
}
}
}
}
default: 
{
if (v_a_327_ == 0)
{
lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_429_; 
v_isSharedCheck_429_ = !lean_is_exclusive(v_x_328_);
if (v_isSharedCheck_429_ == 0)
{
lean_object* v_unused_430_; 
v_unused_430_ = lean_ctor_get(v_x_328_, 0);
lean_dec(v_unused_430_);
v___x_423_ = v_x_328_;
v_isShared_424_ = v_isSharedCheck_429_;
goto v_resetjp_422_;
}
else
{
lean_dec(v_x_328_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_429_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v___x_425_; lean_object* v___x_427_; 
v___x_425_ = lean_box(0);
if (v_isShared_424_ == 0)
{
lean_ctor_set_tag(v___x_423_, 0);
lean_ctor_set(v___x_423_, 0, v___x_425_);
v___x_427_ = v___x_423_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_425_);
v___x_427_ = v_reuseFailAlloc_428_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
return v___x_427_;
}
}
}
else
{
lean_object* v_cref_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v_elimMVarCluster_434_; lean_object* v___x_435_; lean_object* v_goals_436_; lean_object* v___x_437_; lean_object* v___x_438_; uint8_t v___x_439_; 
v_cref_431_ = lean_ctor_get(v_x_328_, 0);
lean_inc(v_cref_431_);
lean_dec_ref_known(v_x_328_, 1);
v___x_432_ = lean_st_ref_get(v_cref_431_);
lean_dec(v_cref_431_);
v___x_433_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_434_ = lean_ctor_get(v___x_433_, 5);
lean_inc_ref(v_elimMVarCluster_434_);
v___x_435_ = lean_apply_1(v_elimMVarCluster_434_, v___x_432_);
v_goals_436_ = lean_ctor_get(v___x_435_, 1);
lean_inc_ref(v_goals_436_);
lean_dec_ref(v___x_435_);
v___x_437_ = lean_unsigned_to_nat(0u);
v___x_438_ = lean_array_get_size(v_goals_436_);
v___x_439_ = lean_nat_dec_lt(v___x_437_, v___x_438_);
if (v___x_439_ == 0)
{
lean_dec_ref(v_goals_436_);
goto v___jp_367_;
}
else
{
lean_object* v___x_440_; uint8_t v___x_441_; 
v___x_440_ = lean_box(0);
v___x_441_ = lean_nat_dec_le(v___x_438_, v___x_438_);
if (v___x_441_ == 0)
{
if (v___x_439_ == 0)
{
lean_dec_ref(v_goals_436_);
goto v___jp_367_;
}
else
{
size_t v___x_442_; size_t v___x_443_; lean_object* v___x_444_; 
v___x_442_ = ((size_t)0ULL);
v___x_443_ = lean_usize_of_nat(v___x_438_);
v___x_444_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__2(v_a_327_, v_goals_436_, v___x_442_, v___x_443_, v___x_440_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec_ref(v_goals_436_);
if (lean_obj_tag(v___x_444_) == 0)
{
lean_dec_ref_known(v___x_444_, 1);
goto v___jp_367_;
}
else
{
return v___x_444_;
}
}
}
else
{
size_t v___x_445_; size_t v___x_446_; lean_object* v___x_447_; 
v___x_445_ = ((size_t)0ULL);
v___x_446_ = lean_usize_of_nat(v___x_438_);
v___x_447_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__2(v_a_327_, v_goals_436_, v___x_445_, v___x_446_, v___x_440_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec_ref(v_goals_436_);
if (lean_obj_tag(v___x_447_) == 0)
{
lean_dec_ref_known(v___x_447_, 1);
goto v___jp_367_;
}
else
{
return v___x_447_;
}
}
}
}
}
}
v___jp_338_:
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = lean_st_ref_get(v_gref_339_);
lean_dec(v_gref_339_);
v___x_343_ = lp_aesop_Aesop_Goal_stats___redArg(v___x_342_, v___y_341_);
if (lean_obj_tag(v___x_343_) == 0)
{
lean_object* v_a_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_355_; 
v_a_344_ = lean_ctor_get(v___x_343_, 0);
v_isSharedCheck_355_ = !lean_is_exclusive(v___x_343_);
if (v_isSharedCheck_355_ == 0)
{
v___x_346_ = v___x_343_;
v_isShared_347_ = v_isSharedCheck_355_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_a_344_);
lean_dec(v___x_343_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_355_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_353_; 
v___x_348_ = lean_st_ref_take(v___y_340_);
v___x_349_ = lean_array_push(v___x_348_, v_a_344_);
v___x_350_ = lean_st_ref_set(v___y_340_, v___x_349_);
v___x_351_ = lean_box(0);
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 0, v___x_351_);
v___x_353_ = v___x_346_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v___x_351_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
}
}
}
else
{
lean_object* v_a_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_363_; 
v_a_356_ = lean_ctor_get(v___x_343_, 0);
v_isSharedCheck_363_ = !lean_is_exclusive(v___x_343_);
if (v_isSharedCheck_363_ == 0)
{
v___x_358_ = v___x_343_;
v_isShared_359_ = v_isSharedCheck_363_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_a_356_);
lean_dec(v___x_343_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_363_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_361_; 
if (v_isShared_359_ == 0)
{
v___x_361_ = v___x_358_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v_a_356_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
return v___x_361_;
}
}
}
}
v___jp_364_:
{
lean_object* v___x_365_; lean_object* v___x_366_; 
v___x_365_ = lean_box(0);
v___x_366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_366_, 0, v___x_365_);
return v___x_366_;
}
v___jp_367_:
{
lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_368_ = lean_box(0);
v___x_369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
return v___x_369_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__0(uint8_t v_a_448_, lean_object* v_as_449_, size_t v_i_450_, size_t v_stop_451_, lean_object* v_b_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_){
_start:
{
uint8_t v___x_462_; 
v___x_462_ = lean_usize_dec_eq(v_i_450_, v_stop_451_);
if (v___x_462_ == 0)
{
lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v___x_463_ = lean_array_uget_borrowed(v_as_449_, v_i_450_);
lean_inc(v___x_463_);
v___x_464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_464_, 0, v___x_463_);
v___x_465_ = lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0(v_a_448_, v___x_464_, v___y_453_, v___y_454_, v___y_455_, v___y_456_, v___y_457_, v___y_458_, v___y_459_, v___y_460_);
if (lean_obj_tag(v___x_465_) == 0)
{
lean_object* v_a_466_; size_t v___x_467_; size_t v___x_468_; 
v_a_466_ = lean_ctor_get(v___x_465_, 0);
lean_inc(v_a_466_);
lean_dec_ref_known(v___x_465_, 1);
v___x_467_ = ((size_t)1ULL);
v___x_468_ = lean_usize_add(v_i_450_, v___x_467_);
v_i_450_ = v___x_468_;
v_b_452_ = v_a_466_;
goto _start;
}
else
{
return v___x_465_;
}
}
else
{
lean_object* v___x_470_; 
v___x_470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_470_, 0, v_b_452_);
return v___x_470_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__0___boxed(lean_object* v_a_471_, lean_object* v_as_472_, lean_object* v_i_473_, lean_object* v_stop_474_, lean_object* v_b_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_){
_start:
{
uint8_t v_a_27393__boxed_485_; size_t v_i_boxed_486_; size_t v_stop_boxed_487_; lean_object* v_res_488_; 
v_a_27393__boxed_485_ = lean_unbox(v_a_471_);
v_i_boxed_486_ = lean_unbox_usize(v_i_473_);
lean_dec(v_i_473_);
v_stop_boxed_487_ = lean_unbox_usize(v_stop_474_);
lean_dec(v_stop_474_);
v_res_488_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__0(v_a_27393__boxed_485_, v_as_472_, v_i_boxed_486_, v_stop_boxed_487_, v_b_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_, v___y_482_, v___y_483_);
lean_dec(v___y_483_);
lean_dec_ref(v___y_482_);
lean_dec(v___y_481_);
lean_dec_ref(v___y_480_);
lean_dec(v___y_479_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_477_);
lean_dec(v___y_476_);
lean_dec_ref(v_as_472_);
return v_res_488_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__1___boxed(lean_object* v_a_489_, lean_object* v_as_490_, lean_object* v_i_491_, lean_object* v_stop_492_, lean_object* v_b_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_){
_start:
{
uint8_t v_a_27413__boxed_503_; size_t v_i_boxed_504_; size_t v_stop_boxed_505_; lean_object* v_res_506_; 
v_a_27413__boxed_503_ = lean_unbox(v_a_489_);
v_i_boxed_504_ = lean_unbox_usize(v_i_491_);
lean_dec(v_i_491_);
v_stop_boxed_505_ = lean_unbox_usize(v_stop_492_);
lean_dec(v_stop_492_);
v_res_506_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__1(v_a_27413__boxed_503_, v_as_490_, v_i_boxed_504_, v_stop_boxed_505_, v_b_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_, v___y_499_, v___y_500_, v___y_501_);
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
lean_dec(v___y_499_);
lean_dec_ref(v___y_498_);
lean_dec(v___y_497_);
lean_dec(v___y_496_);
lean_dec_ref(v___y_495_);
lean_dec(v___y_494_);
lean_dec_ref(v_as_490_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__2___boxed(lean_object* v_a_507_, lean_object* v_as_508_, lean_object* v_i_509_, lean_object* v_stop_510_, lean_object* v_b_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_){
_start:
{
uint8_t v_a_27433__boxed_521_; size_t v_i_boxed_522_; size_t v_stop_boxed_523_; lean_object* v_res_524_; 
v_a_27433__boxed_521_ = lean_unbox(v_a_507_);
v_i_boxed_522_ = lean_unbox_usize(v_i_509_);
lean_dec(v_i_509_);
v_stop_boxed_523_ = lean_unbox_usize(v_stop_510_);
lean_dec(v_stop_510_);
v_res_524_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0_spec__2(v_a_27433__boxed_521_, v_as_508_, v_i_boxed_522_, v_stop_boxed_523_, v_b_511_, v___y_512_, v___y_513_, v___y_514_, v___y_515_, v___y_516_, v___y_517_, v___y_518_, v___y_519_);
lean_dec(v___y_519_);
lean_dec_ref(v___y_518_);
lean_dec(v___y_517_);
lean_dec_ref(v___y_516_);
lean_dec(v___y_515_);
lean_dec(v___y_514_);
lean_dec_ref(v___y_513_);
lean_dec(v___y_512_);
lean_dec_ref(v_as_508_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0___boxed(lean_object* v_a_525_, lean_object* v_x_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_){
_start:
{
uint8_t v_a_27453__boxed_536_; lean_object* v_res_537_; 
v_a_27453__boxed_536_ = lean_unbox(v_a_525_);
v_res_537_ = lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0(v_a_27453__boxed_536_, v_x_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_, v___y_531_, v___y_532_, v___y_533_, v___y_534_);
lean_dec(v___y_534_);
lean_dec_ref(v___y_533_);
lean_dec(v___y_532_);
lean_dec_ref(v___y_531_);
lean_dec(v___y_530_);
lean_dec(v___y_529_);
lean_dec_ref(v___y_528_);
lean_dec(v___y_527_);
return v_res_537_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_collectGoalStatsIfEnabled(lean_object* v_a_541_, lean_object* v_a_542_, lean_object* v_a_543_, lean_object* v_a_544_, lean_object* v_a_545_, lean_object* v_a_546_, lean_object* v_a_547_){
_start:
{
uint8_t v_a_553_; uint8_t v_a_599_; lean_object* v___y_601_; lean_object* v_options_604_; lean_object* v___x_605_; uint8_t v___x_606_; 
v_options_604_ = lean_ctor_get(v_a_546_, 2);
v___x_605_ = lp_aesop_Aesop_aesop_collectStats;
v___x_606_ = lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__1(v_options_604_, v___x_605_);
if (v___x_606_ == 0)
{
lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v_a_609_; uint8_t v___x_610_; 
v___x_607_ = lp_aesop_Aesop_TraceOption_stats;
v___x_608_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___redArg(v___x_607_, v_a_546_);
v_a_609_ = lean_ctor_get(v___x_608_, 0);
lean_inc(v_a_609_);
v___x_610_ = lean_unbox(v_a_609_);
lean_dec(v_a_609_);
if (v___x_610_ == 0)
{
lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; uint8_t v___x_614_; 
lean_dec_ref(v___x_608_);
v___x_611_ = lp_aesop_Aesop_aesop_stats_file;
v___x_612_ = lp_aesop_Lean_Option_get___at___00Aesop_collectGoalStatsIfEnabled_spec__3(v_options_604_, v___x_611_);
v___x_613_ = ((lean_object*)(lp_aesop_Aesop_collectGoalStatsIfEnabled___closed__1));
v___x_614_ = lean_string_dec_eq(v___x_612_, v___x_613_);
lean_dec_ref(v___x_612_);
if (v___x_614_ == 0)
{
uint8_t v___x_615_; 
v___x_615_ = 1;
v_a_553_ = v___x_615_;
goto v___jp_552_;
}
else
{
goto v___jp_549_;
}
}
else
{
v___y_601_ = v___x_608_;
goto v___jp_600_;
}
}
else
{
v_a_599_ = v___x_606_;
goto v___jp_598_;
}
v___jp_549_:
{
lean_object* v___x_550_; lean_object* v___x_551_; 
v___x_550_ = lean_box(0);
v___x_551_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_551_, 0, v___x_550_);
return v___x_551_;
}
v___jp_552_:
{
lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v_root_557_; lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_554_ = lean_st_ref_get(v_a_542_);
v___x_555_ = ((lean_object*)(lp_aesop_Aesop_collectGoalStatsIfEnabled___closed__0));
v___x_556_ = lean_st_mk_ref(v___x_555_);
v_root_557_ = lean_ctor_get(v___x_554_, 0);
lean_inc(v_root_557_);
lean_dec(v___x_554_);
v___x_558_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_558_, 0, v_root_557_);
v___x_559_ = lp_aesop_Aesop_traverseDown___at___00Aesop_collectGoalStatsIfEnabled_spec__0(v_a_553_, v___x_558_, v___x_556_, v_a_541_, v_a_542_, v_a_543_, v_a_544_, v_a_545_, v_a_546_, v_a_547_);
if (lean_obj_tag(v___x_559_) == 0)
{
lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_596_; 
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_596_ == 0)
{
lean_object* v_unused_597_; 
v_unused_597_ = lean_ctor_get(v___x_559_, 0);
lean_dec(v_unused_597_);
v___x_561_ = v___x_559_;
v_isShared_562_ = v_isSharedCheck_596_;
goto v_resetjp_560_;
}
else
{
lean_dec(v___x_559_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_596_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v_stats_565_; lean_object* v_rulePatternCache_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_595_; 
v___x_563_ = lean_st_ref_get(v___x_556_);
lean_dec(v___x_556_);
v___x_564_ = lean_st_ref_take(v_a_543_);
v_stats_565_ = lean_ctor_get(v___x_564_, 1);
v_rulePatternCache_566_ = lean_ctor_get(v___x_564_, 0);
v_isSharedCheck_595_ = !lean_is_exclusive(v___x_564_);
if (v_isSharedCheck_595_ == 0)
{
v___x_568_ = v___x_564_;
v_isShared_569_ = v_isSharedCheck_595_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_stats_565_);
lean_inc(v_rulePatternCache_566_);
lean_dec(v___x_564_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_595_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v_total_570_; lean_object* v_configParsing_571_; lean_object* v_ruleSetConstruction_572_; lean_object* v_search_573_; lean_object* v_ruleSelection_574_; lean_object* v_script_575_; lean_object* v_forwardState_576_; lean_object* v_scriptGenerated_577_; lean_object* v_ruleStats_578_; lean_object* v___x_580_; uint8_t v_isShared_581_; uint8_t v_isSharedCheck_593_; 
v_total_570_ = lean_ctor_get(v_stats_565_, 0);
v_configParsing_571_ = lean_ctor_get(v_stats_565_, 1);
v_ruleSetConstruction_572_ = lean_ctor_get(v_stats_565_, 2);
v_search_573_ = lean_ctor_get(v_stats_565_, 3);
v_ruleSelection_574_ = lean_ctor_get(v_stats_565_, 4);
v_script_575_ = lean_ctor_get(v_stats_565_, 5);
v_forwardState_576_ = lean_ctor_get(v_stats_565_, 6);
v_scriptGenerated_577_ = lean_ctor_get(v_stats_565_, 7);
v_ruleStats_578_ = lean_ctor_get(v_stats_565_, 8);
v_isSharedCheck_593_ = !lean_is_exclusive(v_stats_565_);
if (v_isSharedCheck_593_ == 0)
{
lean_object* v_unused_594_; 
v_unused_594_ = lean_ctor_get(v_stats_565_, 9);
lean_dec(v_unused_594_);
v___x_580_ = v_stats_565_;
v_isShared_581_ = v_isSharedCheck_593_;
goto v_resetjp_579_;
}
else
{
lean_inc(v_ruleStats_578_);
lean_inc(v_scriptGenerated_577_);
lean_inc(v_forwardState_576_);
lean_inc(v_script_575_);
lean_inc(v_ruleSelection_574_);
lean_inc(v_search_573_);
lean_inc(v_ruleSetConstruction_572_);
lean_inc(v_configParsing_571_);
lean_inc(v_total_570_);
lean_dec(v_stats_565_);
v___x_580_ = lean_box(0);
v_isShared_581_ = v_isSharedCheck_593_;
goto v_resetjp_579_;
}
v_resetjp_579_:
{
lean_object* v___x_583_; 
if (v_isShared_581_ == 0)
{
lean_ctor_set(v___x_580_, 9, v___x_563_);
v___x_583_ = v___x_580_;
goto v_reusejp_582_;
}
else
{
lean_object* v_reuseFailAlloc_592_; 
v_reuseFailAlloc_592_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_592_, 0, v_total_570_);
lean_ctor_set(v_reuseFailAlloc_592_, 1, v_configParsing_571_);
lean_ctor_set(v_reuseFailAlloc_592_, 2, v_ruleSetConstruction_572_);
lean_ctor_set(v_reuseFailAlloc_592_, 3, v_search_573_);
lean_ctor_set(v_reuseFailAlloc_592_, 4, v_ruleSelection_574_);
lean_ctor_set(v_reuseFailAlloc_592_, 5, v_script_575_);
lean_ctor_set(v_reuseFailAlloc_592_, 6, v_forwardState_576_);
lean_ctor_set(v_reuseFailAlloc_592_, 7, v_scriptGenerated_577_);
lean_ctor_set(v_reuseFailAlloc_592_, 8, v_ruleStats_578_);
lean_ctor_set(v_reuseFailAlloc_592_, 9, v___x_563_);
v___x_583_ = v_reuseFailAlloc_592_;
goto v_reusejp_582_;
}
v_reusejp_582_:
{
lean_object* v___x_585_; 
if (v_isShared_569_ == 0)
{
lean_ctor_set(v___x_568_, 1, v___x_583_);
v___x_585_ = v___x_568_;
goto v_reusejp_584_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v_rulePatternCache_566_);
lean_ctor_set(v_reuseFailAlloc_591_, 1, v___x_583_);
v___x_585_ = v_reuseFailAlloc_591_;
goto v_reusejp_584_;
}
v_reusejp_584_:
{
lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_589_; 
v___x_586_ = lean_st_ref_set(v_a_543_, v___x_585_);
v___x_587_ = lean_box(0);
if (v_isShared_562_ == 0)
{
lean_ctor_set(v___x_561_, 0, v___x_587_);
v___x_589_ = v___x_561_;
goto v_reusejp_588_;
}
else
{
lean_object* v_reuseFailAlloc_590_; 
v_reuseFailAlloc_590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_590_, 0, v___x_587_);
v___x_589_ = v_reuseFailAlloc_590_;
goto v_reusejp_588_;
}
v_reusejp_588_:
{
return v___x_589_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_556_);
return v___x_559_;
}
}
v___jp_598_:
{
if (v_a_599_ == 0)
{
goto v___jp_549_;
}
else
{
v_a_553_ = v_a_599_;
goto v___jp_552_;
}
}
v___jp_600_:
{
lean_object* v_a_602_; uint8_t v___x_603_; 
v_a_602_ = lean_ctor_get(v___y_601_, 0);
lean_inc(v_a_602_);
lean_dec_ref(v___y_601_);
v___x_603_ = lean_unbox(v_a_602_);
lean_dec(v_a_602_);
v_a_599_ = v___x_603_;
goto v___jp_598_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_collectGoalStatsIfEnabled___boxed(lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_, lean_object* v_a_623_){
_start:
{
lean_object* v_res_624_; 
v_res_624_ = lp_aesop_Aesop_collectGoalStatsIfEnabled(v_a_616_, v_a_617_, v_a_618_, v_a_619_, v_a_620_, v_a_621_, v_a_622_);
lean_dec(v_a_622_);
lean_dec_ref(v_a_621_);
lean_dec(v_a_620_);
lean_dec_ref(v_a_619_);
lean_dec(v_a_618_);
lean_dec(v_a_617_);
lean_dec_ref(v_a_616_);
return v_res_624_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2(lean_object* v_opt_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_){
_start:
{
lean_object* v___x_634_; 
v___x_634_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___redArg(v_opt_625_, v___y_631_);
return v___x_634_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2___boxed(lean_object* v_opt_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_collectGoalStatsIfEnabled_spec__2(v_opt_635_, v___y_636_, v___y_637_, v___y_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_);
lean_dec(v___y_642_);
lean_dec_ref(v___y_641_);
lean_dec(v___y_640_);
lean_dec_ref(v___y_639_);
lean_dec(v___y_638_);
lean_dec(v___y_637_);
lean_dec_ref(v___y_636_);
lean_dec_ref(v_opt_635_);
return v_res_644_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_Stats(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_Stats(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_Stats(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Stats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_Stats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_Stats(builtin);
}
#ifdef __cplusplus
}
#endif
