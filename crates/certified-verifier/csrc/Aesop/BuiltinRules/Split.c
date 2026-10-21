// Lean compiler output
// Module: Aesop.BuiltinRules.Split
// Imports: public import Init public meta import Init public import Aesop.Frontend.Attribute
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
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mvarIdToSubgoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_splitTargetS_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_splitTarget_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_splitTarget_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_BuiltinRules_splitTarget___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "nothing to split in target"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_splitTarget___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_splitTarget___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_BuiltinRules_splitTarget___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_BuiltinRules_splitTarget___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__0_value;
static const lean_string_object lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__2_value_aux_0),((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__2 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__3;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__4;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_BuiltinRules_splitHypothesesCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_BuiltinRules_splitHypothesesCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_splitHypothesesCore___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitHypothesesCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitHypothesesCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "no splittable hypothesis found"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitHypotheses(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitHypotheses___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg(lean_object* v_x_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_, lean_object* v___y_8_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_10_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg___closed__0));
v___x_11_ = lean_st_mk_ref(v___x_10_);
lean_inc(v___y_8_);
lean_inc_ref(v___y_7_);
lean_inc(v___y_6_);
lean_inc_ref(v___y_5_);
lean_inc(v___y_4_);
lean_inc(v___x_11_);
v___x_12_ = lean_apply_7(v_x_3_, v___x_11_, v___y_4_, v___y_5_, v___y_6_, v___y_7_, v___y_8_, lean_box(0));
if (lean_obj_tag(v___x_12_) == 0)
{
lean_object* v_a_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_22_; 
v_a_13_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_22_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_22_ == 0)
{
v___x_15_ = v___x_12_;
v_isShared_16_ = v_isSharedCheck_22_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_a_13_);
lean_dec(v___x_12_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_22_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_20_; 
v___x_17_ = lean_st_ref_get(v___x_11_);
lean_dec(v___x_11_);
v___x_18_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_18_, 0, v_a_13_);
lean_ctor_set(v___x_18_, 1, v___x_17_);
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 0, v___x_18_);
v___x_20_ = v___x_15_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v___x_18_);
v___x_20_ = v_reuseFailAlloc_21_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
return v___x_20_;
}
}
}
else
{
lean_object* v_a_23_; lean_object* v___x_25_; uint8_t v_isShared_26_; uint8_t v_isSharedCheck_30_; 
lean_dec(v___x_11_);
v_a_23_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_30_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_30_ == 0)
{
v___x_25_ = v___x_12_;
v_isShared_26_ = v_isSharedCheck_30_;
goto v_resetjp_24_;
}
else
{
lean_inc(v_a_23_);
lean_dec(v___x_12_);
v___x_25_ = lean_box(0);
v_isShared_26_ = v_isSharedCheck_30_;
goto v_resetjp_24_;
}
v_resetjp_24_:
{
lean_object* v___x_28_; 
if (v_isShared_26_ == 0)
{
v___x_28_ = v___x_25_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v_a_23_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg___boxed(lean_object* v_x_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg(v_x_31_, v___y_32_, v___y_33_, v___y_34_, v___y_35_, v___y_36_);
lean_dec(v___y_36_);
lean_dec_ref(v___y_35_);
lean_dec(v___y_34_);
lean_dec_ref(v___y_33_);
lean_dec(v___y_32_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0(lean_object* v_00_u03b1_39_, lean_object* v_x_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg(v_x_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___boxed(lean_object* v_00_u03b1_48_, lean_object* v_x_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0(v_00_u03b1_48_, v_x_49_, v___y_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_);
lean_dec(v___y_54_);
lean_dec_ref(v___y_53_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec(v___y_50_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2_spec__2(lean_object* v_msgData_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_){
_start:
{
lean_object* v___x_63_; lean_object* v_env_64_; lean_object* v___x_65_; lean_object* v_mctx_66_; lean_object* v_lctx_67_; lean_object* v_options_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_63_ = lean_st_ref_get(v___y_61_);
v_env_64_ = lean_ctor_get(v___x_63_, 0);
lean_inc_ref(v_env_64_);
lean_dec(v___x_63_);
v___x_65_ = lean_st_ref_get(v___y_59_);
v_mctx_66_ = lean_ctor_get(v___x_65_, 0);
lean_inc_ref(v_mctx_66_);
lean_dec(v___x_65_);
v_lctx_67_ = lean_ctor_get(v___y_58_, 2);
v_options_68_ = lean_ctor_get(v___y_60_, 2);
lean_inc_ref(v_options_68_);
lean_inc_ref(v_lctx_67_);
v___x_69_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_69_, 0, v_env_64_);
lean_ctor_set(v___x_69_, 1, v_mctx_66_);
lean_ctor_set(v___x_69_, 2, v_lctx_67_);
lean_ctor_set(v___x_69_, 3, v_options_68_);
v___x_70_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v_msgData_57_);
v___x_71_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2_spec__2___boxed(lean_object* v_msgData_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2_spec__2(v_msgData_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_);
lean_dec(v___y_76_);
lean_dec_ref(v___y_75_);
lean_dec(v___y_74_);
lean_dec_ref(v___y_73_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___redArg(lean_object* v_msg_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_ref_85_; lean_object* v___x_86_; lean_object* v_a_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_95_; 
v_ref_85_ = lean_ctor_get(v___y_82_, 5);
v___x_86_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2_spec__2(v_msg_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
v_a_87_ = lean_ctor_get(v___x_86_, 0);
v_isSharedCheck_95_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_95_ == 0)
{
v___x_89_ = v___x_86_;
v_isShared_90_ = v_isSharedCheck_95_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_a_87_);
lean_dec(v___x_86_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_95_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_91_; lean_object* v___x_93_; 
lean_inc(v_ref_85_);
v___x_91_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_91_, 0, v_ref_85_);
lean_ctor_set(v___x_91_, 1, v_a_87_);
if (v_isShared_90_ == 0)
{
lean_ctor_set_tag(v___x_89_, 1);
lean_ctor_set(v___x_89_, 0, v___x_91_);
v___x_93_ = v___x_89_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v___x_91_);
v___x_93_ = v_reuseFailAlloc_94_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
return v___x_93_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___redArg___boxed(lean_object* v_msg_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___redArg(v_msg_96_, v___y_97_, v___y_98_, v___y_99_, v___y_100_);
lean_dec(v___y_100_);
lean_dec_ref(v___y_99_);
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_splitTarget_spec__1(lean_object* v___x_103_, size_t v_sz_104_, size_t v_i_105_, lean_object* v_bs_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_){
_start:
{
uint8_t v___x_113_; 
v___x_113_ = lean_usize_dec_lt(v_i_105_, v_sz_104_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; 
lean_dec(v___x_103_);
v___x_114_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_114_, 0, v_bs_106_);
return v___x_114_;
}
else
{
lean_object* v_v_115_; lean_object* v___x_116_; 
v_v_115_ = lean_array_uget_borrowed(v_bs_106_, v_i_105_);
lean_inc(v_v_115_);
lean_inc(v___x_103_);
v___x_116_ = lp_aesop_Aesop_mvarIdToSubgoal(v___x_103_, v_v_115_, v___y_107_, v___y_108_, v___y_109_, v___y_110_, v___y_111_);
if (lean_obj_tag(v___x_116_) == 0)
{
lean_object* v_a_117_; lean_object* v___x_118_; lean_object* v_bs_x27_119_; size_t v___x_120_; size_t v___x_121_; lean_object* v___x_122_; 
v_a_117_ = lean_ctor_get(v___x_116_, 0);
lean_inc(v_a_117_);
lean_dec_ref_known(v___x_116_, 1);
v___x_118_ = lean_unsigned_to_nat(0u);
v_bs_x27_119_ = lean_array_uset(v_bs_106_, v_i_105_, v___x_118_);
v___x_120_ = ((size_t)1ULL);
v___x_121_ = lean_usize_add(v_i_105_, v___x_120_);
v___x_122_ = lean_array_uset(v_bs_x27_119_, v_i_105_, v_a_117_);
v_i_105_ = v___x_121_;
v_bs_106_ = v___x_122_;
goto _start;
}
else
{
lean_object* v_a_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_131_; 
lean_dec_ref(v_bs_106_);
lean_dec(v___x_103_);
v_a_124_ = lean_ctor_get(v___x_116_, 0);
v_isSharedCheck_131_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_131_ == 0)
{
v___x_126_ = v___x_116_;
v_isShared_127_ = v_isSharedCheck_131_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_a_124_);
lean_dec(v___x_116_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_131_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
lean_object* v___x_129_; 
if (v_isShared_127_ == 0)
{
v___x_129_ = v___x_126_;
goto v_reusejp_128_;
}
else
{
lean_object* v_reuseFailAlloc_130_; 
v_reuseFailAlloc_130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_130_, 0, v_a_124_);
v___x_129_ = v_reuseFailAlloc_130_;
goto v_reusejp_128_;
}
v_reusejp_128_:
{
return v___x_129_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_splitTarget_spec__1___boxed(lean_object* v___x_132_, lean_object* v_sz_133_, lean_object* v_i_134_, lean_object* v_bs_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_){
_start:
{
size_t v_sz_boxed_142_; size_t v_i_boxed_143_; lean_object* v_res_144_; 
v_sz_boxed_142_ = lean_unbox_usize(v_sz_133_);
lean_dec(v_sz_133_);
v_i_boxed_143_ = lean_unbox_usize(v_i_134_);
lean_dec(v_i_134_);
v_res_144_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_splitTarget_spec__1(v___x_132_, v_sz_boxed_142_, v_i_boxed_143_, v_bs_135_, v___y_136_, v___y_137_, v___y_138_, v___y_139_, v___y_140_);
lean_dec(v___y_140_);
lean_dec_ref(v___y_139_);
lean_dec(v___y_138_);
lean_dec_ref(v___y_137_);
lean_dec(v___y_136_);
return v_res_144_;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_splitTarget___closed__1(void){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_146_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_splitTarget___closed__0));
v___x_147_ = l_Lean_stringToMessageData(v___x_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitTarget(lean_object* v_a_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_, lean_object* v_a_153_){
_start:
{
lean_object* v_fst_156_; lean_object* v_fst_157_; lean_object* v_snd_158_; lean_object* v_goal_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v_goal_180_ = lean_ctor_get(v_a_148_, 0);
lean_inc_n(v_goal_180_, 2);
lean_dec_ref(v_a_148_);
v___x_181_ = lean_alloc_closure((void*)(lp_aesop_Aesop_splitTargetS_x3f___boxed), 8, 1);
lean_closure_set(v___x_181_, 0, v_goal_180_);
v___x_182_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg(v___x_181_, v_a_149_, v_a_150_, v_a_151_, v_a_152_, v_a_153_);
if (lean_obj_tag(v___x_182_) == 0)
{
lean_object* v_a_183_; lean_object* v_fst_184_; 
v_a_183_ = lean_ctor_get(v___x_182_, 0);
lean_inc(v_a_183_);
lean_dec_ref_known(v___x_182_, 1);
v_fst_184_ = lean_ctor_get(v_a_183_, 0);
lean_inc(v_fst_184_);
if (lean_obj_tag(v_fst_184_) == 1)
{
lean_object* v_snd_185_; lean_object* v_val_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_206_; 
v_snd_185_ = lean_ctor_get(v_a_183_, 1);
lean_inc(v_snd_185_);
lean_dec(v_a_183_);
v_val_186_ = lean_ctor_get(v_fst_184_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v_fst_184_);
if (v_isSharedCheck_206_ == 0)
{
v___x_188_ = v_fst_184_;
v_isShared_189_ = v_isSharedCheck_206_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_val_186_);
lean_dec(v_fst_184_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_206_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
size_t v_sz_190_; size_t v___x_191_; lean_object* v___x_192_; 
v_sz_190_ = lean_array_size(v_val_186_);
v___x_191_ = ((size_t)0ULL);
v___x_192_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_splitTarget_spec__1(v_goal_180_, v_sz_190_, v___x_191_, v_val_186_, v_a_149_, v_a_150_, v_a_151_, v_a_152_, v_a_153_);
if (lean_obj_tag(v___x_192_) == 0)
{
lean_object* v_a_193_; lean_object* v___x_195_; 
v_a_193_ = lean_ctor_get(v___x_192_, 0);
lean_inc(v_a_193_);
lean_dec_ref_known(v___x_192_, 1);
if (v_isShared_189_ == 0)
{
lean_ctor_set(v___x_188_, 0, v_snd_185_);
v___x_195_ = v___x_188_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v_snd_185_);
v___x_195_ = v_reuseFailAlloc_197_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
lean_object* v___x_196_; 
v___x_196_ = lean_box(0);
v_fst_156_ = v_a_193_;
v_fst_157_ = v___x_195_;
v_snd_158_ = v___x_196_;
goto v___jp_155_;
}
}
else
{
lean_object* v_a_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_205_; 
lean_del_object(v___x_188_);
lean_dec(v_snd_185_);
v_a_198_ = lean_ctor_get(v___x_192_, 0);
v_isSharedCheck_205_ = !lean_is_exclusive(v___x_192_);
if (v_isSharedCheck_205_ == 0)
{
v___x_200_ = v___x_192_;
v_isShared_201_ = v_isSharedCheck_205_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_a_198_);
lean_dec(v___x_192_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_205_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_203_; 
if (v_isShared_201_ == 0)
{
v___x_203_ = v___x_200_;
goto v_reusejp_202_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v_a_198_);
v___x_203_ = v_reuseFailAlloc_204_;
goto v_reusejp_202_;
}
v_reusejp_202_:
{
return v___x_203_;
}
}
}
}
}
else
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v_a_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_216_; 
lean_dec(v_fst_184_);
lean_dec(v_a_183_);
lean_dec(v_goal_180_);
v___x_207_ = lean_obj_once(&lp_aesop_Aesop_BuiltinRules_splitTarget___closed__1, &lp_aesop_Aesop_BuiltinRules_splitTarget___closed__1_once, _init_lp_aesop_Aesop_BuiltinRules_splitTarget___closed__1);
v___x_208_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___redArg(v___x_207_, v_a_150_, v_a_151_, v_a_152_, v_a_153_);
v_a_209_ = lean_ctor_get(v___x_208_, 0);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_208_);
if (v_isSharedCheck_216_ == 0)
{
v___x_211_ = v___x_208_;
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_a_209_);
lean_dec(v___x_208_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_214_; 
if (v_isShared_212_ == 0)
{
v___x_214_ = v___x_211_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v_a_209_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
}
else
{
lean_object* v_a_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_224_; 
lean_dec(v_goal_180_);
v_a_217_ = lean_ctor_get(v___x_182_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_224_ == 0)
{
v___x_219_ = v___x_182_;
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_a_217_);
lean_dec(v___x_182_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_222_; 
if (v_isShared_220_ == 0)
{
v___x_222_ = v___x_219_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_a_217_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
v___jp_155_:
{
lean_object* v___x_159_; 
v___x_159_ = l_Lean_Meta_saveState___redArg(v_a_151_, v_a_153_);
if (lean_obj_tag(v___x_159_) == 0)
{
lean_object* v_a_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_171_; 
v_a_160_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_171_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_171_ == 0)
{
v___x_162_ = v___x_159_;
v_isShared_163_ = v_isSharedCheck_171_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_a_160_);
lean_dec(v___x_159_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_171_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_169_; 
lean_inc(v_snd_158_);
v___x_164_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_164_, 0, v_fst_156_);
lean_ctor_set(v___x_164_, 1, v_a_160_);
lean_ctor_set(v___x_164_, 2, v_fst_157_);
lean_ctor_set(v___x_164_, 3, v_snd_158_);
v___x_165_ = lean_unsigned_to_nat(1u);
v___x_166_ = lean_mk_empty_array_with_capacity(v___x_165_);
v___x_167_ = lean_array_push(v___x_166_, v___x_164_);
if (v_isShared_163_ == 0)
{
lean_ctor_set(v___x_162_, 0, v___x_167_);
v___x_169_ = v___x_162_;
goto v_reusejp_168_;
}
else
{
lean_object* v_reuseFailAlloc_170_; 
v_reuseFailAlloc_170_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_170_, 0, v___x_167_);
v___x_169_ = v_reuseFailAlloc_170_;
goto v_reusejp_168_;
}
v_reusejp_168_:
{
return v___x_169_;
}
}
}
else
{
lean_object* v_a_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_179_; 
lean_dec(v_fst_157_);
lean_dec_ref(v_fst_156_);
v_a_172_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_179_ == 0)
{
v___x_174_ = v___x_159_;
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_a_172_);
lean_dec(v___x_159_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_177_; 
if (v_isShared_175_ == 0)
{
v___x_177_ = v___x_174_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v_a_172_);
v___x_177_ = v_reuseFailAlloc_178_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
return v___x_177_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitTarget___boxed(lean_object* v_a_225_, lean_object* v_a_226_, lean_object* v_a_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_aesop_Aesop_BuiltinRules_splitTarget(v_a_225_, v_a_226_, v_a_227_, v_a_228_, v_a_229_, v_a_230_);
lean_dec(v_a_230_);
lean_dec_ref(v_a_229_);
lean_dec(v_a_228_);
lean_dec_ref(v_a_227_);
lean_dec(v_a_226_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2(lean_object* v_00_u03b1_233_, lean_object* v_msg_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___redArg(v_msg_234_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___boxed(lean_object* v_00_u03b1_242_, lean_object* v_msg_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2(v_00_u03b1_242_, v_msg_243_, v___y_244_, v___y_245_, v___y_246_, v___y_247_, v___y_248_);
lean_dec(v___y_248_);
lean_dec_ref(v___y_247_);
lean_dec(v___y_246_);
lean_dec_ref(v___y_245_);
lean_dec(v___y_244_);
return v_res_250_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_256_ = l_Lean_maxRecDepthErrorMessage;
v___x_257_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
return v___x_257_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__4(void){
_start:
{
lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_258_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__3, &lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__3_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__3);
v___x_259_ = l_Lean_MessageData_ofFormat(v___x_258_);
return v___x_259_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__5(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_260_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__4, &lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__4_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__4);
v___x_261_ = ((lean_object*)(lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__2));
v___x_262_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v___x_260_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg(lean_object* v_ref_263_){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_265_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__5, &lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__5_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___closed__5);
v___x_266_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_266_, 0, v_ref_263_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_267_, 0, v___x_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg___boxed(lean_object* v_ref_268_, lean_object* v___y_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg(v_ref_268_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1(lean_object* v_00_u03b1_271_, lean_object* v_ref_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg(v_ref_272_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___boxed(lean_object* v_00_u03b1_281_, lean_object* v_ref_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1(v_00_u03b1_281_, v_ref_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
lean_dec(v___y_284_);
lean_dec(v___y_283_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitHypothesesCore(lean_object* v_goal_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_, lean_object* v_a_299_){
_start:
{
lean_object* v_fileName_301_; lean_object* v_fileMap_302_; lean_object* v_options_303_; lean_object* v_currRecDepth_304_; lean_object* v_maxRecDepth_305_; lean_object* v_ref_306_; lean_object* v_currNamespace_307_; lean_object* v_openDecls_308_; lean_object* v_initHeartbeats_309_; lean_object* v_maxHeartbeats_310_; lean_object* v_quotContext_311_; lean_object* v_currMacroScope_312_; uint8_t v_diag_313_; lean_object* v_cancelTk_x3f_314_; uint8_t v_suppressElabErrors_315_; lean_object* v_inheritedTraceOptions_316_; lean_object* v___x_359_; uint8_t v___x_360_; 
v_fileName_301_ = lean_ctor_get(v_a_298_, 0);
v_fileMap_302_ = lean_ctor_get(v_a_298_, 1);
v_options_303_ = lean_ctor_get(v_a_298_, 2);
v_currRecDepth_304_ = lean_ctor_get(v_a_298_, 3);
v_maxRecDepth_305_ = lean_ctor_get(v_a_298_, 4);
v_ref_306_ = lean_ctor_get(v_a_298_, 5);
v_currNamespace_307_ = lean_ctor_get(v_a_298_, 6);
v_openDecls_308_ = lean_ctor_get(v_a_298_, 7);
v_initHeartbeats_309_ = lean_ctor_get(v_a_298_, 8);
v_maxHeartbeats_310_ = lean_ctor_get(v_a_298_, 9);
v_quotContext_311_ = lean_ctor_get(v_a_298_, 10);
v_currMacroScope_312_ = lean_ctor_get(v_a_298_, 11);
v_diag_313_ = lean_ctor_get_uint8(v_a_298_, sizeof(void*)*14);
v_cancelTk_x3f_314_ = lean_ctor_get(v_a_298_, 12);
v_suppressElabErrors_315_ = lean_ctor_get_uint8(v_a_298_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_316_ = lean_ctor_get(v_a_298_, 13);
v___x_359_ = lean_unsigned_to_nat(0u);
v___x_360_ = lean_nat_dec_eq(v_maxRecDepth_305_, v___x_359_);
if (v___x_360_ == 0)
{
uint8_t v___x_361_; 
v___x_361_ = lean_nat_dec_eq(v_currRecDepth_304_, v_maxRecDepth_305_);
if (v___x_361_ == 0)
{
goto v___jp_317_;
}
else
{
lean_object* v___x_362_; 
lean_dec(v_goal_293_);
lean_inc(v_ref_306_);
v___x_362_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__1___redArg(v_ref_306_);
return v___x_362_;
}
}
else
{
goto v___jp_317_;
}
v___jp_317_:
{
lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_318_ = lean_unsigned_to_nat(1u);
v___x_319_ = lean_nat_add(v_currRecDepth_304_, v___x_318_);
lean_inc_ref(v_inheritedTraceOptions_316_);
lean_inc(v_cancelTk_x3f_314_);
lean_inc(v_currMacroScope_312_);
lean_inc(v_quotContext_311_);
lean_inc(v_maxHeartbeats_310_);
lean_inc(v_initHeartbeats_309_);
lean_inc(v_openDecls_308_);
lean_inc(v_currNamespace_307_);
lean_inc(v_ref_306_);
lean_inc(v_maxRecDepth_305_);
lean_inc_ref(v_options_303_);
lean_inc_ref(v_fileMap_302_);
lean_inc_ref(v_fileName_301_);
v___x_320_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_320_, 0, v_fileName_301_);
lean_ctor_set(v___x_320_, 1, v_fileMap_302_);
lean_ctor_set(v___x_320_, 2, v_options_303_);
lean_ctor_set(v___x_320_, 3, v___x_319_);
lean_ctor_set(v___x_320_, 4, v_maxRecDepth_305_);
lean_ctor_set(v___x_320_, 5, v_ref_306_);
lean_ctor_set(v___x_320_, 6, v_currNamespace_307_);
lean_ctor_set(v___x_320_, 7, v_openDecls_308_);
lean_ctor_set(v___x_320_, 8, v_initHeartbeats_309_);
lean_ctor_set(v___x_320_, 9, v_maxHeartbeats_310_);
lean_ctor_set(v___x_320_, 10, v_quotContext_311_);
lean_ctor_set(v___x_320_, 11, v_currMacroScope_312_);
lean_ctor_set(v___x_320_, 12, v_cancelTk_x3f_314_);
lean_ctor_set(v___x_320_, 13, v_inheritedTraceOptions_316_);
lean_ctor_set_uint8(v___x_320_, sizeof(void*)*14, v_diag_313_);
lean_ctor_set_uint8(v___x_320_, sizeof(void*)*14 + 1, v_suppressElabErrors_315_);
v___x_321_ = lp_aesop_Aesop_splitFirstHypothesisS_x3f(v_goal_293_, v_a_294_, v_a_295_, v_a_296_, v_a_297_, v___x_320_, v_a_299_);
if (lean_obj_tag(v___x_321_) == 0)
{
lean_object* v_a_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_358_; 
v_a_322_ = lean_ctor_get(v___x_321_, 0);
v_isSharedCheck_358_ = !lean_is_exclusive(v___x_321_);
if (v_isSharedCheck_358_ == 0)
{
v___x_324_ = v___x_321_;
v_isShared_325_ = v_isSharedCheck_358_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_a_322_);
lean_dec(v___x_321_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_358_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
if (lean_obj_tag(v_a_322_) == 1)
{
lean_object* v_val_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_353_; 
lean_del_object(v___x_324_);
v_val_326_ = lean_ctor_get(v_a_322_, 0);
v_isSharedCheck_353_ = !lean_is_exclusive(v_a_322_);
if (v_isSharedCheck_353_ == 0)
{
v___x_328_ = v_a_322_;
v_isShared_329_ = v_isSharedCheck_353_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_val_326_);
lean_dec(v_a_322_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_353_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___x_330_; size_t v_sz_331_; size_t v___x_332_; lean_object* v___x_333_; 
v___x_330_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_splitHypothesesCore___closed__0));
v_sz_331_ = lean_array_size(v_val_326_);
v___x_332_ = ((size_t)0ULL);
v___x_333_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__0(v_val_326_, v_sz_331_, v___x_332_, v___x_330_, v_a_294_, v_a_295_, v_a_296_, v_a_297_, v___x_320_, v_a_299_);
lean_dec_ref_known(v___x_320_, 14);
lean_dec(v_val_326_);
if (lean_obj_tag(v___x_333_) == 0)
{
lean_object* v_a_334_; lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_344_; 
v_a_334_ = lean_ctor_get(v___x_333_, 0);
v_isSharedCheck_344_ = !lean_is_exclusive(v___x_333_);
if (v_isSharedCheck_344_ == 0)
{
v___x_336_ = v___x_333_;
v_isShared_337_ = v_isSharedCheck_344_;
goto v_resetjp_335_;
}
else
{
lean_inc(v_a_334_);
lean_dec(v___x_333_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_344_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v___x_339_; 
if (v_isShared_329_ == 0)
{
lean_ctor_set(v___x_328_, 0, v_a_334_);
v___x_339_ = v___x_328_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v_a_334_);
v___x_339_ = v_reuseFailAlloc_343_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
lean_object* v___x_341_; 
if (v_isShared_337_ == 0)
{
lean_ctor_set(v___x_336_, 0, v___x_339_);
v___x_341_ = v___x_336_;
goto v_reusejp_340_;
}
else
{
lean_object* v_reuseFailAlloc_342_; 
v_reuseFailAlloc_342_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_342_, 0, v___x_339_);
v___x_341_ = v_reuseFailAlloc_342_;
goto v_reusejp_340_;
}
v_reusejp_340_:
{
return v___x_341_;
}
}
}
}
else
{
lean_object* v_a_345_; lean_object* v___x_347_; uint8_t v_isShared_348_; uint8_t v_isSharedCheck_352_; 
lean_del_object(v___x_328_);
v_a_345_ = lean_ctor_get(v___x_333_, 0);
v_isSharedCheck_352_ = !lean_is_exclusive(v___x_333_);
if (v_isSharedCheck_352_ == 0)
{
v___x_347_ = v___x_333_;
v_isShared_348_ = v_isSharedCheck_352_;
goto v_resetjp_346_;
}
else
{
lean_inc(v_a_345_);
lean_dec(v___x_333_);
v___x_347_ = lean_box(0);
v_isShared_348_ = v_isSharedCheck_352_;
goto v_resetjp_346_;
}
v_resetjp_346_:
{
lean_object* v___x_350_; 
if (v_isShared_348_ == 0)
{
v___x_350_ = v___x_347_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_351_; 
v_reuseFailAlloc_351_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_351_, 0, v_a_345_);
v___x_350_ = v_reuseFailAlloc_351_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
return v___x_350_;
}
}
}
}
}
else
{
lean_object* v___x_354_; lean_object* v___x_356_; 
lean_dec(v_a_322_);
lean_dec_ref_known(v___x_320_, 14);
v___x_354_ = lean_box(0);
if (v_isShared_325_ == 0)
{
lean_ctor_set(v___x_324_, 0, v___x_354_);
v___x_356_ = v___x_324_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v___x_354_);
v___x_356_ = v_reuseFailAlloc_357_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
return v___x_356_;
}
}
}
}
else
{
lean_dec_ref_known(v___x_320_, 14);
return v___x_321_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__0(lean_object* v_as_363_, size_t v_sz_364_, size_t v_i_365_, lean_object* v_b_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_){
_start:
{
uint8_t v___x_374_; 
v___x_374_ = lean_usize_dec_lt(v_i_365_, v_sz_364_);
if (v___x_374_ == 0)
{
lean_object* v___x_375_; 
v___x_375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_375_, 0, v_b_366_);
return v___x_375_;
}
else
{
lean_object* v_a_376_; lean_object* v___x_377_; 
v_a_376_ = lean_array_uget_borrowed(v_as_363_, v_i_365_);
lean_inc(v_a_376_);
v___x_377_ = lp_aesop_Aesop_BuiltinRules_splitHypothesesCore(v_a_376_, v___y_367_, v___y_368_, v___y_369_, v___y_370_, v___y_371_, v___y_372_);
if (lean_obj_tag(v___x_377_) == 0)
{
lean_object* v_a_378_; lean_object* v_a_380_; 
v_a_378_ = lean_ctor_get(v___x_377_, 0);
lean_inc(v_a_378_);
lean_dec_ref_known(v___x_377_, 1);
if (lean_obj_tag(v_a_378_) == 1)
{
lean_object* v_val_384_; lean_object* v___x_385_; 
v_val_384_ = lean_ctor_get(v_a_378_, 0);
lean_inc(v_val_384_);
lean_dec_ref_known(v_a_378_, 1);
v___x_385_ = l_Array_append___redArg(v_b_366_, v_val_384_);
lean_dec(v_val_384_);
v_a_380_ = v___x_385_;
goto v___jp_379_;
}
else
{
lean_object* v___x_386_; 
lean_dec(v_a_378_);
lean_inc(v_a_376_);
v___x_386_ = lean_array_push(v_b_366_, v_a_376_);
v_a_380_ = v___x_386_;
goto v___jp_379_;
}
v___jp_379_:
{
size_t v___x_381_; size_t v___x_382_; 
v___x_381_ = ((size_t)1ULL);
v___x_382_ = lean_usize_add(v_i_365_, v___x_381_);
v_i_365_ = v___x_382_;
v_b_366_ = v_a_380_;
goto _start;
}
}
else
{
lean_object* v_a_387_; lean_object* v___x_389_; uint8_t v_isShared_390_; uint8_t v_isSharedCheck_394_; 
lean_dec_ref(v_b_366_);
v_a_387_ = lean_ctor_get(v___x_377_, 0);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_377_);
if (v_isSharedCheck_394_ == 0)
{
v___x_389_ = v___x_377_;
v_isShared_390_ = v_isSharedCheck_394_;
goto v_resetjp_388_;
}
else
{
lean_inc(v_a_387_);
lean_dec(v___x_377_);
v___x_389_ = lean_box(0);
v_isShared_390_ = v_isSharedCheck_394_;
goto v_resetjp_388_;
}
v_resetjp_388_:
{
lean_object* v___x_392_; 
if (v_isShared_390_ == 0)
{
v___x_392_ = v___x_389_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v_a_387_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__0___boxed(lean_object* v_as_395_, lean_object* v_sz_396_, lean_object* v_i_397_, lean_object* v_b_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
size_t v_sz_boxed_406_; size_t v_i_boxed_407_; lean_object* v_res_408_; 
v_sz_boxed_406_ = lean_unbox_usize(v_sz_396_);
lean_dec(v_sz_396_);
v_i_boxed_407_ = lean_unbox_usize(v_i_397_);
lean_dec(v_i_397_);
v_res_408_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_BuiltinRules_splitHypothesesCore_spec__0(v_as_395_, v_sz_boxed_406_, v_i_boxed_407_, v_b_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
lean_dec(v___y_404_);
lean_dec_ref(v___y_403_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
lean_dec(v___y_400_);
lean_dec(v___y_399_);
lean_dec_ref(v_as_395_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitHypothesesCore___boxed(lean_object* v_goal_409_, lean_object* v_a_410_, lean_object* v_a_411_, lean_object* v_a_412_, lean_object* v_a_413_, lean_object* v_a_414_, lean_object* v_a_415_, lean_object* v_a_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_aesop_Aesop_BuiltinRules_splitHypothesesCore(v_goal_409_, v_a_410_, v_a_411_, v_a_412_, v_a_413_, v_a_414_, v_a_415_);
lean_dec(v_a_415_);
lean_dec_ref(v_a_414_);
lean_dec(v_a_413_);
lean_dec_ref(v_a_412_);
lean_dec(v_a_411_);
lean_dec(v_a_410_);
return v_res_417_;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__1(void){
_start:
{
lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_419_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__0));
v___x_420_ = l_Lean_stringToMessageData(v___x_419_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitHypotheses(lean_object* v_a_421_, lean_object* v_a_422_, lean_object* v_a_423_, lean_object* v_a_424_, lean_object* v_a_425_, lean_object* v_a_426_){
_start:
{
lean_object* v_fst_429_; lean_object* v_fst_430_; lean_object* v_snd_431_; lean_object* v_goal_453_; lean_object* v___x_454_; lean_object* v___x_455_; 
v_goal_453_ = lean_ctor_get(v_a_421_, 0);
lean_inc_n(v_goal_453_, 2);
lean_dec_ref(v_a_421_);
v___x_454_ = lean_alloc_closure((void*)(lp_aesop_Aesop_BuiltinRules_splitHypothesesCore___boxed), 8, 1);
lean_closure_set(v___x_454_, 0, v_goal_453_);
v___x_455_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_splitTarget_spec__0___redArg(v___x_454_, v_a_422_, v_a_423_, v_a_424_, v_a_425_, v_a_426_);
if (lean_obj_tag(v___x_455_) == 0)
{
lean_object* v_a_456_; lean_object* v_fst_457_; 
v_a_456_ = lean_ctor_get(v___x_455_, 0);
lean_inc(v_a_456_);
lean_dec_ref_known(v___x_455_, 1);
v_fst_457_ = lean_ctor_get(v_a_456_, 0);
lean_inc(v_fst_457_);
if (lean_obj_tag(v_fst_457_) == 1)
{
lean_object* v_snd_458_; lean_object* v_val_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_479_; 
v_snd_458_ = lean_ctor_get(v_a_456_, 1);
lean_inc(v_snd_458_);
lean_dec(v_a_456_);
v_val_459_ = lean_ctor_get(v_fst_457_, 0);
v_isSharedCheck_479_ = !lean_is_exclusive(v_fst_457_);
if (v_isSharedCheck_479_ == 0)
{
v___x_461_ = v_fst_457_;
v_isShared_462_ = v_isSharedCheck_479_;
goto v_resetjp_460_;
}
else
{
lean_inc(v_val_459_);
lean_dec(v_fst_457_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_479_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
size_t v_sz_463_; size_t v___x_464_; lean_object* v___x_465_; 
v_sz_463_ = lean_array_size(v_val_459_);
v___x_464_ = ((size_t)0ULL);
v___x_465_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_splitTarget_spec__1(v_goal_453_, v_sz_463_, v___x_464_, v_val_459_, v_a_422_, v_a_423_, v_a_424_, v_a_425_, v_a_426_);
if (lean_obj_tag(v___x_465_) == 0)
{
lean_object* v_a_466_; lean_object* v___x_468_; 
v_a_466_ = lean_ctor_get(v___x_465_, 0);
lean_inc(v_a_466_);
lean_dec_ref_known(v___x_465_, 1);
if (v_isShared_462_ == 0)
{
lean_ctor_set(v___x_461_, 0, v_snd_458_);
v___x_468_ = v___x_461_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v_snd_458_);
v___x_468_ = v_reuseFailAlloc_470_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
lean_object* v___x_469_; 
v___x_469_ = lean_box(0);
v_fst_429_ = v_a_466_;
v_fst_430_ = v___x_468_;
v_snd_431_ = v___x_469_;
goto v___jp_428_;
}
}
else
{
lean_object* v_a_471_; lean_object* v___x_473_; uint8_t v_isShared_474_; uint8_t v_isSharedCheck_478_; 
lean_del_object(v___x_461_);
lean_dec(v_snd_458_);
v_a_471_ = lean_ctor_get(v___x_465_, 0);
v_isSharedCheck_478_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_478_ == 0)
{
v___x_473_ = v___x_465_;
v_isShared_474_ = v_isSharedCheck_478_;
goto v_resetjp_472_;
}
else
{
lean_inc(v_a_471_);
lean_dec(v___x_465_);
v___x_473_ = lean_box(0);
v_isShared_474_ = v_isSharedCheck_478_;
goto v_resetjp_472_;
}
v_resetjp_472_:
{
lean_object* v___x_476_; 
if (v_isShared_474_ == 0)
{
v___x_476_ = v___x_473_;
goto v_reusejp_475_;
}
else
{
lean_object* v_reuseFailAlloc_477_; 
v_reuseFailAlloc_477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_477_, 0, v_a_471_);
v___x_476_ = v_reuseFailAlloc_477_;
goto v_reusejp_475_;
}
v_reusejp_475_:
{
return v___x_476_;
}
}
}
}
}
else
{
lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v_a_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_489_; 
lean_dec(v_fst_457_);
lean_dec(v_a_456_);
lean_dec(v_goal_453_);
v___x_480_ = lean_obj_once(&lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__1, &lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__1_once, _init_lp_aesop_Aesop_BuiltinRules_splitHypotheses___closed__1);
v___x_481_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_splitTarget_spec__2___redArg(v___x_480_, v_a_423_, v_a_424_, v_a_425_, v_a_426_);
v_a_482_ = lean_ctor_get(v___x_481_, 0);
v_isSharedCheck_489_ = !lean_is_exclusive(v___x_481_);
if (v_isSharedCheck_489_ == 0)
{
v___x_484_ = v___x_481_;
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_a_482_);
lean_dec(v___x_481_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_487_; 
if (v_isShared_485_ == 0)
{
v___x_487_ = v___x_484_;
goto v_reusejp_486_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v_a_482_);
v___x_487_ = v_reuseFailAlloc_488_;
goto v_reusejp_486_;
}
v_reusejp_486_:
{
return v___x_487_;
}
}
}
}
else
{
lean_object* v_a_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_497_; 
lean_dec(v_goal_453_);
v_a_490_ = lean_ctor_get(v___x_455_, 0);
v_isSharedCheck_497_ = !lean_is_exclusive(v___x_455_);
if (v_isSharedCheck_497_ == 0)
{
v___x_492_ = v___x_455_;
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_a_490_);
lean_dec(v___x_455_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v___x_495_; 
if (v_isShared_493_ == 0)
{
v___x_495_ = v___x_492_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v_a_490_);
v___x_495_ = v_reuseFailAlloc_496_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
return v___x_495_;
}
}
}
v___jp_428_:
{
lean_object* v___x_432_; 
v___x_432_ = l_Lean_Meta_saveState___redArg(v_a_424_, v_a_426_);
if (lean_obj_tag(v___x_432_) == 0)
{
lean_object* v_a_433_; lean_object* v___x_435_; uint8_t v_isShared_436_; uint8_t v_isSharedCheck_444_; 
v_a_433_ = lean_ctor_get(v___x_432_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_432_);
if (v_isSharedCheck_444_ == 0)
{
v___x_435_ = v___x_432_;
v_isShared_436_ = v_isSharedCheck_444_;
goto v_resetjp_434_;
}
else
{
lean_inc(v_a_433_);
lean_dec(v___x_432_);
v___x_435_ = lean_box(0);
v_isShared_436_ = v_isSharedCheck_444_;
goto v_resetjp_434_;
}
v_resetjp_434_:
{
lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_442_; 
lean_inc(v_snd_431_);
v___x_437_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_437_, 0, v_fst_429_);
lean_ctor_set(v___x_437_, 1, v_a_433_);
lean_ctor_set(v___x_437_, 2, v_fst_430_);
lean_ctor_set(v___x_437_, 3, v_snd_431_);
v___x_438_ = lean_unsigned_to_nat(1u);
v___x_439_ = lean_mk_empty_array_with_capacity(v___x_438_);
v___x_440_ = lean_array_push(v___x_439_, v___x_437_);
if (v_isShared_436_ == 0)
{
lean_ctor_set(v___x_435_, 0, v___x_440_);
v___x_442_ = v___x_435_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v___x_440_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
else
{
lean_object* v_a_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_452_; 
lean_dec(v_fst_430_);
lean_dec_ref(v_fst_429_);
v_a_445_ = lean_ctor_get(v___x_432_, 0);
v_isSharedCheck_452_ = !lean_is_exclusive(v___x_432_);
if (v_isSharedCheck_452_ == 0)
{
v___x_447_ = v___x_432_;
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_a_445_);
lean_dec(v___x_432_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v___x_450_; 
if (v_isShared_448_ == 0)
{
v___x_450_ = v___x_447_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_a_445_);
v___x_450_ = v_reuseFailAlloc_451_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
return v___x_450_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_splitHypotheses___boxed(lean_object* v_a_498_, lean_object* v_a_499_, lean_object* v_a_500_, lean_object* v_a_501_, lean_object* v_a_502_, lean_object* v_a_503_, lean_object* v_a_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_aesop_Aesop_BuiltinRules_splitHypotheses(v_a_498_, v_a_499_, v_a_500_, v_a_501_, v_a_502_, v_a_503_);
lean_dec(v_a_503_);
lean_dec_ref(v_a_502_);
lean_dec(v_a_501_);
lean_dec_ref(v_a_500_);
lean_dec(v_a_499_);
return v_res_505_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Split(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_BuiltinRules_Split(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_BuiltinRules_Split(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Split(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_BuiltinRules_Split(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_BuiltinRules_Split(builtin);
}
#ifdef __cplusplus
}
#endif
