// Lean compiler output
// Module: Aesop.BuiltinRules.Ext
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lp_aesop_Aesop_straightLineExtS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mvarIdToSubgoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_extCore_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_extCore_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_extCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_extCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__0_value;
static const lean_string_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__2_value_aux_0),((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__2 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__3;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__4;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1___closed__0 = (const lean_object*)&lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_BuiltinRules_extCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BuiltinRules_extCore___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BuiltinRules_extCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_extCore___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_extCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_extCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_ext_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_ext_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_BuiltinRules_ext___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "found no applicable ext lemma"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_ext___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_ext___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_BuiltinRules_ext___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_BuiltinRules_ext___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_ext(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_ext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_extCore_spec__0(size_t v_sz_1_, size_t v_i_2_, lean_object* v_bs_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = lean_usize_dec_lt(v_i_2_, v_sz_1_);
if (v___x_4_ == 0)
{
return v_bs_3_;
}
else
{
lean_object* v_v_5_; lean_object* v_fst_6_; lean_object* v___x_7_; lean_object* v_bs_x27_8_; size_t v___x_9_; size_t v___x_10_; lean_object* v___x_11_; 
v_v_5_ = lean_array_uget_borrowed(v_bs_3_, v_i_2_);
v_fst_6_ = lean_ctor_get(v_v_5_, 0);
lean_inc(v_fst_6_);
v___x_7_ = lean_unsigned_to_nat(0u);
v_bs_x27_8_ = lean_array_uset(v_bs_3_, v_i_2_, v___x_7_);
v___x_9_ = ((size_t)1ULL);
v___x_10_ = lean_usize_add(v_i_2_, v___x_9_);
v___x_11_ = lean_array_uset(v_bs_x27_8_, v_i_2_, v_fst_6_);
v_i_2_ = v___x_10_;
v_bs_3_ = v___x_11_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_extCore_spec__0___boxed(lean_object* v_sz_13_, lean_object* v_i_14_, lean_object* v_bs_15_){
_start:
{
size_t v_sz_boxed_16_; size_t v_i_boxed_17_; lean_object* v_res_18_; 
v_sz_boxed_16_ = lean_unbox_usize(v_sz_13_);
lean_dec(v_sz_13_);
v_i_boxed_17_ = lean_unbox_usize(v_i_14_);
lean_dec(v_i_14_);
v_res_18_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_extCore_spec__0(v_sz_boxed_16_, v_i_boxed_17_, v_bs_15_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_extCore___lam__0(lean_object* v_goal_19_, lean_object* v___y_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_aesop_Aesop_straightLineExtS(v_goal_19_, v___y_20_, v___y_21_, v___y_22_, v___y_23_, v___y_24_, v___y_25_);
if (lean_obj_tag(v___x_27_) == 0)
{
lean_object* v_a_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_47_; 
v_a_28_ = lean_ctor_get(v___x_27_, 0);
v_isSharedCheck_47_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_47_ == 0)
{
v___x_30_ = v___x_27_;
v_isShared_31_ = v_isSharedCheck_47_;
goto v_resetjp_29_;
}
else
{
lean_inc(v_a_28_);
lean_dec(v___x_27_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_47_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v_depth_32_; lean_object* v_goals_33_; lean_object* v___x_34_; uint8_t v___x_35_; 
v_depth_32_ = lean_ctor_get(v_a_28_, 0);
lean_inc(v_depth_32_);
v_goals_33_ = lean_ctor_get(v_a_28_, 2);
lean_inc_ref(v_goals_33_);
lean_dec(v_a_28_);
v___x_34_ = lean_unsigned_to_nat(0u);
v___x_35_ = lean_nat_dec_eq(v_depth_32_, v___x_34_);
lean_dec(v_depth_32_);
if (v___x_35_ == 0)
{
size_t v_sz_36_; size_t v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_41_; 
v_sz_36_ = lean_array_size(v_goals_33_);
v___x_37_ = ((size_t)0ULL);
v___x_38_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_extCore_spec__0(v_sz_36_, v___x_37_, v_goals_33_);
v___x_39_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_39_, 0, v___x_38_);
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 0, v___x_39_);
v___x_41_ = v___x_30_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v___x_39_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
else
{
lean_object* v___x_43_; lean_object* v___x_45_; 
lean_dec_ref(v_goals_33_);
v___x_43_ = lean_box(0);
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 0, v___x_43_);
v___x_45_ = v___x_30_;
goto v_reusejp_44_;
}
else
{
lean_object* v_reuseFailAlloc_46_; 
v_reuseFailAlloc_46_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_46_, 0, v___x_43_);
v___x_45_ = v_reuseFailAlloc_46_;
goto v_reusejp_44_;
}
v_reusejp_44_:
{
return v___x_45_;
}
}
}
}
else
{
lean_object* v_a_48_; lean_object* v___x_50_; uint8_t v_isShared_51_; uint8_t v_isSharedCheck_55_; 
v_a_48_ = lean_ctor_get(v___x_27_, 0);
v_isSharedCheck_55_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_55_ == 0)
{
v___x_50_ = v___x_27_;
v_isShared_51_ = v_isSharedCheck_55_;
goto v_resetjp_49_;
}
else
{
lean_inc(v_a_48_);
lean_dec(v___x_27_);
v___x_50_ = lean_box(0);
v_isShared_51_ = v_isSharedCheck_55_;
goto v_resetjp_49_;
}
v_resetjp_49_:
{
lean_object* v___x_53_; 
if (v_isShared_51_ == 0)
{
v___x_53_ = v___x_50_;
goto v_reusejp_52_;
}
else
{
lean_object* v_reuseFailAlloc_54_; 
v_reuseFailAlloc_54_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_54_, 0, v_a_48_);
v___x_53_ = v_reuseFailAlloc_54_;
goto v_reusejp_52_;
}
v_reusejp_52_:
{
return v___x_53_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_extCore___lam__0___boxed(lean_object* v_goal_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_aesop_Aesop_BuiltinRules_extCore___lam__0(v_goal_56_, v___y_57_, v___y_58_, v___y_59_, v___y_60_, v___y_61_, v___y_62_);
lean_dec(v___y_62_);
lean_dec_ref(v___y_61_);
lean_dec(v___y_60_);
lean_dec_ref(v___y_59_);
lean_dec(v___y_58_);
lean_dec(v___y_57_);
return v_res_64_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = l_Lean_maxRecDepthErrorMessage;
v___x_71_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
return v___x_71_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__4(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_72_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__3, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__3_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__3);
v___x_73_ = l_Lean_MessageData_ofFormat(v___x_72_);
return v___x_73_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__5(void){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_74_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__4, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__4_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__4);
v___x_75_ = ((lean_object*)(lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__2));
v___x_76_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v___x_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg(lean_object* v_ref_77_){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_79_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__5, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__5_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___closed__5);
v___x_80_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_80_, 0, v_ref_77_);
lean_ctor_set(v___x_80_, 1, v___x_79_);
v___x_81_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_ref_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg(v_ref_82_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1(lean_object* v_tac_85_, lean_object* v_acc_86_, lean_object* v_goal_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_){
_start:
{
lean_object* v_fileName_95_; lean_object* v_fileMap_96_; lean_object* v_options_97_; lean_object* v_currRecDepth_98_; lean_object* v_maxRecDepth_99_; lean_object* v_ref_100_; lean_object* v_currNamespace_101_; lean_object* v_openDecls_102_; lean_object* v_initHeartbeats_103_; lean_object* v_maxHeartbeats_104_; lean_object* v_quotContext_105_; lean_object* v_currMacroScope_106_; uint8_t v_diag_107_; lean_object* v_cancelTk_x3f_108_; uint8_t v_suppressElabErrors_109_; lean_object* v_inheritedTraceOptions_110_; lean_object* v___x_153_; uint8_t v___x_154_; 
v_fileName_95_ = lean_ctor_get(v___y_92_, 0);
v_fileMap_96_ = lean_ctor_get(v___y_92_, 1);
v_options_97_ = lean_ctor_get(v___y_92_, 2);
v_currRecDepth_98_ = lean_ctor_get(v___y_92_, 3);
v_maxRecDepth_99_ = lean_ctor_get(v___y_92_, 4);
v_ref_100_ = lean_ctor_get(v___y_92_, 5);
v_currNamespace_101_ = lean_ctor_get(v___y_92_, 6);
v_openDecls_102_ = lean_ctor_get(v___y_92_, 7);
v_initHeartbeats_103_ = lean_ctor_get(v___y_92_, 8);
v_maxHeartbeats_104_ = lean_ctor_get(v___y_92_, 9);
v_quotContext_105_ = lean_ctor_get(v___y_92_, 10);
v_currMacroScope_106_ = lean_ctor_get(v___y_92_, 11);
v_diag_107_ = lean_ctor_get_uint8(v___y_92_, sizeof(void*)*14);
v_cancelTk_x3f_108_ = lean_ctor_get(v___y_92_, 12);
v_suppressElabErrors_109_ = lean_ctor_get_uint8(v___y_92_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_110_ = lean_ctor_get(v___y_92_, 13);
v___x_153_ = lean_unsigned_to_nat(0u);
v___x_154_ = lean_nat_dec_eq(v_maxRecDepth_99_, v___x_153_);
if (v___x_154_ == 0)
{
uint8_t v___x_155_; 
v___x_155_ = lean_nat_dec_eq(v_currRecDepth_98_, v_maxRecDepth_99_);
if (v___x_155_ == 0)
{
goto v___jp_111_;
}
else
{
lean_object* v___x_156_; 
lean_dec(v_goal_87_);
lean_dec_ref(v_tac_85_);
lean_inc(v_ref_100_);
v___x_156_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg(v_ref_100_);
return v___x_156_;
}
}
else
{
goto v___jp_111_;
}
v___jp_111_:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_112_ = lean_unsigned_to_nat(1u);
v___x_113_ = lean_nat_add(v_currRecDepth_98_, v___x_112_);
lean_inc_ref(v_inheritedTraceOptions_110_);
lean_inc(v_cancelTk_x3f_108_);
lean_inc(v_currMacroScope_106_);
lean_inc(v_quotContext_105_);
lean_inc(v_maxHeartbeats_104_);
lean_inc(v_initHeartbeats_103_);
lean_inc(v_openDecls_102_);
lean_inc(v_currNamespace_101_);
lean_inc(v_ref_100_);
lean_inc(v_maxRecDepth_99_);
lean_inc_ref(v_options_97_);
lean_inc_ref(v_fileMap_96_);
lean_inc_ref(v_fileName_95_);
v___x_114_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_114_, 0, v_fileName_95_);
lean_ctor_set(v___x_114_, 1, v_fileMap_96_);
lean_ctor_set(v___x_114_, 2, v_options_97_);
lean_ctor_set(v___x_114_, 3, v___x_113_);
lean_ctor_set(v___x_114_, 4, v_maxRecDepth_99_);
lean_ctor_set(v___x_114_, 5, v_ref_100_);
lean_ctor_set(v___x_114_, 6, v_currNamespace_101_);
lean_ctor_set(v___x_114_, 7, v_openDecls_102_);
lean_ctor_set(v___x_114_, 8, v_initHeartbeats_103_);
lean_ctor_set(v___x_114_, 9, v_maxHeartbeats_104_);
lean_ctor_set(v___x_114_, 10, v_quotContext_105_);
lean_ctor_set(v___x_114_, 11, v_currMacroScope_106_);
lean_ctor_set(v___x_114_, 12, v_cancelTk_x3f_108_);
lean_ctor_set(v___x_114_, 13, v_inheritedTraceOptions_110_);
lean_ctor_set_uint8(v___x_114_, sizeof(void*)*14, v_diag_107_);
lean_ctor_set_uint8(v___x_114_, sizeof(void*)*14 + 1, v_suppressElabErrors_109_);
lean_inc_ref(v_tac_85_);
lean_inc(v___y_93_);
lean_inc_ref(v___x_114_);
lean_inc(v___y_91_);
lean_inc_ref(v___y_90_);
lean_inc(v___y_89_);
lean_inc(v___y_88_);
lean_inc(v_goal_87_);
v___x_115_ = lean_apply_8(v_tac_85_, v_goal_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_, v___x_114_, v___y_93_, lean_box(0));
if (lean_obj_tag(v___x_115_) == 0)
{
lean_object* v_a_116_; lean_object* v___x_118_; uint8_t v_isShared_119_; uint8_t v_isSharedCheck_144_; 
v_a_116_ = lean_ctor_get(v___x_115_, 0);
v_isSharedCheck_144_ = !lean_is_exclusive(v___x_115_);
if (v_isSharedCheck_144_ == 0)
{
v___x_118_ = v___x_115_;
v_isShared_119_ = v_isSharedCheck_144_;
goto v_resetjp_117_;
}
else
{
lean_inc(v_a_116_);
lean_dec(v___x_115_);
v___x_118_ = lean_box(0);
v_isShared_119_ = v_isSharedCheck_144_;
goto v_resetjp_117_;
}
v_resetjp_117_:
{
if (lean_obj_tag(v_a_116_) == 0)
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_124_; 
lean_dec_ref_known(v___x_114_, 14);
lean_dec_ref(v_tac_85_);
v___x_120_ = lean_st_ref_take(v_acc_86_);
v___x_121_ = lean_array_push(v___x_120_, v_goal_87_);
v___x_122_ = lean_st_ref_set(v_acc_86_, v___x_121_);
if (v_isShared_119_ == 0)
{
lean_ctor_set(v___x_118_, 0, v___x_122_);
v___x_124_ = v___x_118_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v___x_122_);
v___x_124_ = v_reuseFailAlloc_125_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
return v___x_124_;
}
}
else
{
lean_object* v_val_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; uint8_t v___x_130_; 
lean_dec(v_goal_87_);
v_val_126_ = lean_ctor_get(v_a_116_, 0);
lean_inc(v_val_126_);
lean_dec_ref_known(v_a_116_, 1);
v___x_127_ = lean_unsigned_to_nat(0u);
v___x_128_ = lean_array_get_size(v_val_126_);
v___x_129_ = lean_box(0);
v___x_130_ = lean_nat_dec_lt(v___x_127_, v___x_128_);
if (v___x_130_ == 0)
{
lean_object* v___x_132_; 
lean_dec(v_val_126_);
lean_dec_ref_known(v___x_114_, 14);
lean_dec_ref(v_tac_85_);
if (v_isShared_119_ == 0)
{
lean_ctor_set(v___x_118_, 0, v___x_129_);
v___x_132_ = v___x_118_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v___x_129_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
else
{
uint8_t v___x_134_; 
v___x_134_ = lean_nat_dec_le(v___x_128_, v___x_128_);
if (v___x_134_ == 0)
{
if (v___x_130_ == 0)
{
lean_object* v___x_136_; 
lean_dec(v_val_126_);
lean_dec_ref_known(v___x_114_, 14);
lean_dec_ref(v_tac_85_);
if (v_isShared_119_ == 0)
{
lean_ctor_set(v___x_118_, 0, v___x_129_);
v___x_136_ = v___x_118_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v___x_129_);
v___x_136_ = v_reuseFailAlloc_137_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
return v___x_136_;
}
}
else
{
size_t v___x_138_; size_t v___x_139_; lean_object* v___x_140_; 
lean_del_object(v___x_118_);
v___x_138_ = ((size_t)0ULL);
v___x_139_ = lean_usize_of_nat(v___x_128_);
v___x_140_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2(v_tac_85_, v_acc_86_, v_val_126_, v___x_138_, v___x_139_, v___x_129_, v___y_88_, v___y_89_, v___y_90_, v___y_91_, v___x_114_, v___y_93_);
lean_dec_ref_known(v___x_114_, 14);
lean_dec(v_val_126_);
return v___x_140_;
}
}
else
{
size_t v___x_141_; size_t v___x_142_; lean_object* v___x_143_; 
lean_del_object(v___x_118_);
v___x_141_ = ((size_t)0ULL);
v___x_142_ = lean_usize_of_nat(v___x_128_);
v___x_143_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2(v_tac_85_, v_acc_86_, v_val_126_, v___x_141_, v___x_142_, v___x_129_, v___y_88_, v___y_89_, v___y_90_, v___y_91_, v___x_114_, v___y_93_);
lean_dec_ref_known(v___x_114_, 14);
lean_dec(v_val_126_);
return v___x_143_;
}
}
}
}
}
else
{
lean_object* v_a_145_; lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_152_; 
lean_dec_ref_known(v___x_114_, 14);
lean_dec(v_goal_87_);
lean_dec_ref(v_tac_85_);
v_a_145_ = lean_ctor_get(v___x_115_, 0);
v_isSharedCheck_152_ = !lean_is_exclusive(v___x_115_);
if (v_isSharedCheck_152_ == 0)
{
v___x_147_ = v___x_115_;
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
else
{
lean_inc(v_a_145_);
lean_dec(v___x_115_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v___x_150_; 
if (v_isShared_148_ == 0)
{
v___x_150_ = v___x_147_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v_a_145_);
v___x_150_ = v_reuseFailAlloc_151_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
return v___x_150_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2(lean_object* v_tac_157_, lean_object* v_val_158_, lean_object* v_as_159_, size_t v_i_160_, size_t v_stop_161_, lean_object* v_b_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_){
_start:
{
uint8_t v___x_170_; 
v___x_170_ = lean_usize_dec_eq(v_i_160_, v_stop_161_);
if (v___x_170_ == 0)
{
lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_171_ = lean_array_uget_borrowed(v_as_159_, v_i_160_);
lean_inc(v___x_171_);
lean_inc_ref(v_tac_157_);
v___x_172_ = lp_aesop___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1(v_tac_157_, v_val_158_, v___x_171_, v___y_163_, v___y_164_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
if (lean_obj_tag(v___x_172_) == 0)
{
lean_object* v_a_173_; size_t v___x_174_; size_t v___x_175_; 
v_a_173_ = lean_ctor_get(v___x_172_, 0);
lean_inc(v_a_173_);
lean_dec_ref_known(v___x_172_, 1);
v___x_174_ = ((size_t)1ULL);
v___x_175_ = lean_usize_add(v_i_160_, v___x_174_);
v_i_160_ = v___x_175_;
v_b_162_ = v_a_173_;
goto _start;
}
else
{
lean_dec_ref(v_tac_157_);
return v___x_172_;
}
}
else
{
lean_object* v___x_177_; 
lean_dec_ref(v_tac_157_);
v___x_177_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_177_, 0, v_b_162_);
return v___x_177_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2___boxed(lean_object* v_tac_178_, lean_object* v_val_179_, lean_object* v_as_180_, lean_object* v_i_181_, lean_object* v_stop_182_, lean_object* v_b_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_){
_start:
{
size_t v_i_boxed_191_; size_t v_stop_boxed_192_; lean_object* v_res_193_; 
v_i_boxed_191_ = lean_unbox_usize(v_i_181_);
lean_dec(v_i_181_);
v_stop_boxed_192_ = lean_unbox_usize(v_stop_182_);
lean_dec(v_stop_182_);
v_res_193_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2(v_tac_178_, v_val_179_, v_as_180_, v_i_boxed_191_, v_stop_boxed_192_, v_b_183_, v___y_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_);
lean_dec(v___y_189_);
lean_dec_ref(v___y_188_);
lean_dec(v___y_187_);
lean_dec_ref(v___y_186_);
lean_dec(v___y_185_);
lean_dec(v___y_184_);
lean_dec_ref(v_as_180_);
lean_dec(v_val_179_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1___boxed(lean_object* v_tac_194_, lean_object* v_acc_195_, lean_object* v_goal_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_aesop___private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1(v_tac_194_, v_acc_195_, v_goal_196_, v___y_197_, v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_);
lean_dec(v___y_202_);
lean_dec_ref(v___y_201_);
lean_dec(v___y_200_);
lean_dec_ref(v___y_199_);
lean_dec(v___y_198_);
lean_dec(v___y_197_);
lean_dec(v_acc_195_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1(lean_object* v_goal_207_, lean_object* v_tac_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
lean_object* v___x_216_; 
lean_inc_ref(v_tac_208_);
lean_inc(v___y_214_);
lean_inc_ref(v___y_213_);
lean_inc(v___y_212_);
lean_inc_ref(v___y_211_);
lean_inc(v___y_210_);
lean_inc(v___y_209_);
v___x_216_ = lean_apply_8(v_tac_208_, v_goal_207_, v___y_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_, lean_box(0));
if (lean_obj_tag(v___x_216_) == 0)
{
lean_object* v_a_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_270_; 
v_a_217_ = lean_ctor_get(v___x_216_, 0);
v_isSharedCheck_270_ = !lean_is_exclusive(v___x_216_);
if (v_isSharedCheck_270_ == 0)
{
v___x_219_ = v___x_216_;
v_isShared_220_ = v_isSharedCheck_270_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_a_217_);
lean_dec(v___x_216_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_270_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
if (lean_obj_tag(v_a_217_) == 1)
{
lean_object* v_val_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_265_; 
v_val_221_ = lean_ctor_get(v_a_217_, 0);
v_isSharedCheck_265_ = !lean_is_exclusive(v_a_217_);
if (v_isSharedCheck_265_ == 0)
{
v___x_223_ = v_a_217_;
v_isShared_224_ = v_isSharedCheck_265_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_val_221_);
lean_dec(v_a_217_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_265_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_236_; uint8_t v___x_237_; 
v___x_225_ = lean_unsigned_to_nat(0u);
v___x_226_ = ((lean_object*)(lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1___closed__0));
v___x_227_ = lean_st_mk_ref(v___x_226_);
v___x_236_ = lean_array_get_size(v_val_221_);
v___x_237_ = lean_nat_dec_lt(v___x_225_, v___x_236_);
if (v___x_237_ == 0)
{
lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
lean_del_object(v___x_223_);
lean_dec(v_val_221_);
lean_del_object(v___x_219_);
lean_dec_ref(v_tac_208_);
v___x_238_ = lean_st_ref_get(v___x_227_);
lean_dec(v___x_227_);
v___x_239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_239_, 0, v___x_238_);
v___x_240_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_240_, 0, v___x_239_);
return v___x_240_;
}
else
{
lean_object* v___x_241_; uint8_t v___x_242_; 
v___x_241_ = lean_box(0);
v___x_242_ = lean_nat_dec_le(v___x_236_, v___x_236_);
if (v___x_242_ == 0)
{
if (v___x_237_ == 0)
{
lean_dec(v_val_221_);
lean_dec_ref(v_tac_208_);
goto v___jp_228_;
}
else
{
size_t v___x_243_; size_t v___x_244_; lean_object* v___x_245_; 
v___x_243_ = ((size_t)0ULL);
v___x_244_ = lean_usize_of_nat(v___x_236_);
v___x_245_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2(v_tac_208_, v___x_227_, v_val_221_, v___x_243_, v___x_244_, v___x_241_, v___y_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec(v_val_221_);
if (lean_obj_tag(v___x_245_) == 0)
{
lean_dec_ref_known(v___x_245_, 1);
goto v___jp_228_;
}
else
{
lean_object* v_a_246_; lean_object* v___x_248_; uint8_t v_isShared_249_; uint8_t v_isSharedCheck_253_; 
lean_dec(v___x_227_);
lean_del_object(v___x_223_);
lean_del_object(v___x_219_);
v_a_246_ = lean_ctor_get(v___x_245_, 0);
v_isSharedCheck_253_ = !lean_is_exclusive(v___x_245_);
if (v_isSharedCheck_253_ == 0)
{
v___x_248_ = v___x_245_;
v_isShared_249_ = v_isSharedCheck_253_;
goto v_resetjp_247_;
}
else
{
lean_inc(v_a_246_);
lean_dec(v___x_245_);
v___x_248_ = lean_box(0);
v_isShared_249_ = v_isSharedCheck_253_;
goto v_resetjp_247_;
}
v_resetjp_247_:
{
lean_object* v___x_251_; 
if (v_isShared_249_ == 0)
{
v___x_251_ = v___x_248_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_252_; 
v_reuseFailAlloc_252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_252_, 0, v_a_246_);
v___x_251_ = v_reuseFailAlloc_252_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
return v___x_251_;
}
}
}
}
}
else
{
size_t v___x_254_; size_t v___x_255_; lean_object* v___x_256_; 
v___x_254_ = ((size_t)0ULL);
v___x_255_ = lean_usize_of_nat(v___x_236_);
v___x_256_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__2(v_tac_208_, v___x_227_, v_val_221_, v___x_254_, v___x_255_, v___x_241_, v___y_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec(v_val_221_);
if (lean_obj_tag(v___x_256_) == 0)
{
lean_dec_ref_known(v___x_256_, 1);
goto v___jp_228_;
}
else
{
lean_object* v_a_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_264_; 
lean_dec(v___x_227_);
lean_del_object(v___x_223_);
lean_del_object(v___x_219_);
v_a_257_ = lean_ctor_get(v___x_256_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_256_);
if (v_isSharedCheck_264_ == 0)
{
v___x_259_ = v___x_256_;
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_a_257_);
lean_dec(v___x_256_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
if (v_isShared_260_ == 0)
{
v___x_262_ = v___x_259_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_a_257_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
}
}
v___jp_228_:
{
lean_object* v___x_229_; lean_object* v___x_231_; 
v___x_229_ = lean_st_ref_get(v___x_227_);
lean_dec(v___x_227_);
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 0, v___x_229_);
v___x_231_ = v___x_223_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v___x_229_);
v___x_231_ = v_reuseFailAlloc_235_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
lean_object* v___x_233_; 
if (v_isShared_220_ == 0)
{
lean_ctor_set(v___x_219_, 0, v___x_231_);
v___x_233_ = v___x_219_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v___x_231_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
}
}
else
{
lean_object* v___x_266_; lean_object* v___x_268_; 
lean_dec(v_a_217_);
lean_dec_ref(v_tac_208_);
v___x_266_ = lean_box(0);
if (v_isShared_220_ == 0)
{
lean_ctor_set(v___x_219_, 0, v___x_266_);
v___x_268_ = v___x_219_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v___x_266_);
v___x_268_ = v_reuseFailAlloc_269_;
goto v_reusejp_267_;
}
v_reusejp_267_:
{
return v___x_268_;
}
}
}
}
else
{
lean_dec_ref(v_tac_208_);
return v___x_216_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1___boxed(lean_object* v_goal_271_, lean_object* v_tac_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1(v_goal_271_, v_tac_272_, v___y_273_, v___y_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_);
lean_dec(v___y_278_);
lean_dec_ref(v___y_277_);
lean_dec(v___y_276_);
lean_dec_ref(v___y_275_);
lean_dec(v___y_274_);
lean_dec(v___y_273_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_extCore(lean_object* v_goal_282_, lean_object* v_a_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_){
_start:
{
lean_object* v___f_290_; lean_object* v___x_291_; 
v___f_290_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_extCore___closed__0));
v___x_291_ = lp_aesop_Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1(v_goal_282_, v___f_290_, v_a_283_, v_a_284_, v_a_285_, v_a_286_, v_a_287_, v_a_288_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_extCore___boxed(lean_object* v_goal_292_, lean_object* v_a_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_, lean_object* v_a_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_aesop_Aesop_BuiltinRules_extCore(v_goal_292_, v_a_293_, v_a_294_, v_a_295_, v_a_296_, v_a_297_, v_a_298_);
lean_dec(v_a_298_);
lean_dec_ref(v_a_297_);
lean_dec(v_a_296_);
lean_dec_ref(v_a_295_);
lean_dec(v_a_294_);
lean_dec(v_a_293_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2(lean_object* v_00_u03b1_301_, lean_object* v_ref_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___redArg(v_ref_302_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b1_311_, lean_object* v_ref_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Batteries_Lean_Meta_Basic_0__Lean_Meta_saturate1_go___at___00Lean_Meta_saturate1___at___00Aesop_BuiltinRules_extCore_spec__1_spec__1_spec__2(v_00_u03b1_311_, v_ref_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_);
lean_dec(v___y_318_);
lean_dec_ref(v___y_317_);
lean_dec(v___y_316_);
lean_dec_ref(v___y_315_);
lean_dec(v___y_314_);
lean_dec(v___y_313_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg(lean_object* v_x_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_330_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg___closed__0));
v___x_331_ = lean_st_mk_ref(v___x_330_);
lean_inc(v___y_328_);
lean_inc_ref(v___y_327_);
lean_inc(v___y_326_);
lean_inc_ref(v___y_325_);
lean_inc(v___y_324_);
lean_inc(v___x_331_);
v___x_332_ = lean_apply_7(v_x_323_, v___x_331_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_, lean_box(0));
if (lean_obj_tag(v___x_332_) == 0)
{
lean_object* v_a_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_342_; 
v_a_333_ = lean_ctor_get(v___x_332_, 0);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_342_ == 0)
{
v___x_335_ = v___x_332_;
v_isShared_336_ = v_isSharedCheck_342_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_a_333_);
lean_dec(v___x_332_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_342_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_340_; 
v___x_337_ = lean_st_ref_get(v___x_331_);
lean_dec(v___x_331_);
v___x_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_338_, 0, v_a_333_);
lean_ctor_set(v___x_338_, 1, v___x_337_);
if (v_isShared_336_ == 0)
{
lean_ctor_set(v___x_335_, 0, v___x_338_);
v___x_340_ = v___x_335_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v___x_338_);
v___x_340_ = v_reuseFailAlloc_341_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
return v___x_340_;
}
}
}
else
{
lean_object* v_a_343_; lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_350_; 
lean_dec(v___x_331_);
v_a_343_ = lean_ctor_get(v___x_332_, 0);
v_isSharedCheck_350_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_350_ == 0)
{
v___x_345_ = v___x_332_;
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
else
{
lean_inc(v_a_343_);
lean_dec(v___x_332_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v___x_348_; 
if (v_isShared_346_ == 0)
{
v___x_348_ = v___x_345_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_349_; 
v_reuseFailAlloc_349_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_349_, 0, v_a_343_);
v___x_348_ = v_reuseFailAlloc_349_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
return v___x_348_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg___boxed(lean_object* v_x_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg(v_x_351_, v___y_352_, v___y_353_, v___y_354_, v___y_355_, v___y_356_);
lean_dec(v___y_356_);
lean_dec_ref(v___y_355_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
lean_dec(v___y_352_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0(lean_object* v_00_u03b1_359_, lean_object* v_x_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_){
_start:
{
lean_object* v___x_367_; 
v___x_367_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg(v_x_360_, v___y_361_, v___y_362_, v___y_363_, v___y_364_, v___y_365_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___boxed(lean_object* v_00_u03b1_368_, lean_object* v_x_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0(v_00_u03b1_368_, v_x_369_, v___y_370_, v___y_371_, v___y_372_, v___y_373_, v___y_374_);
lean_dec(v___y_374_);
lean_dec_ref(v___y_373_);
lean_dec(v___y_372_);
lean_dec_ref(v___y_371_);
lean_dec(v___y_370_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_ext_spec__1(lean_object* v___x_377_, size_t v_sz_378_, size_t v_i_379_, lean_object* v_bs_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_){
_start:
{
uint8_t v___x_387_; 
v___x_387_ = lean_usize_dec_lt(v_i_379_, v_sz_378_);
if (v___x_387_ == 0)
{
lean_object* v___x_388_; 
lean_dec(v___x_377_);
v___x_388_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_388_, 0, v_bs_380_);
return v___x_388_;
}
else
{
lean_object* v_v_389_; lean_object* v___x_390_; 
v_v_389_ = lean_array_uget_borrowed(v_bs_380_, v_i_379_);
lean_inc(v_v_389_);
lean_inc(v___x_377_);
v___x_390_ = lp_aesop_Aesop_mvarIdToSubgoal(v___x_377_, v_v_389_, v___y_381_, v___y_382_, v___y_383_, v___y_384_, v___y_385_);
if (lean_obj_tag(v___x_390_) == 0)
{
lean_object* v_a_391_; lean_object* v___x_392_; lean_object* v_bs_x27_393_; size_t v___x_394_; size_t v___x_395_; lean_object* v___x_396_; 
v_a_391_ = lean_ctor_get(v___x_390_, 0);
lean_inc(v_a_391_);
lean_dec_ref_known(v___x_390_, 1);
v___x_392_ = lean_unsigned_to_nat(0u);
v_bs_x27_393_ = lean_array_uset(v_bs_380_, v_i_379_, v___x_392_);
v___x_394_ = ((size_t)1ULL);
v___x_395_ = lean_usize_add(v_i_379_, v___x_394_);
v___x_396_ = lean_array_uset(v_bs_x27_393_, v_i_379_, v_a_391_);
v_i_379_ = v___x_395_;
v_bs_380_ = v___x_396_;
goto _start;
}
else
{
lean_object* v_a_398_; lean_object* v___x_400_; uint8_t v_isShared_401_; uint8_t v_isSharedCheck_405_; 
lean_dec_ref(v_bs_380_);
lean_dec(v___x_377_);
v_a_398_ = lean_ctor_get(v___x_390_, 0);
v_isSharedCheck_405_ = !lean_is_exclusive(v___x_390_);
if (v_isSharedCheck_405_ == 0)
{
v___x_400_ = v___x_390_;
v_isShared_401_ = v_isSharedCheck_405_;
goto v_resetjp_399_;
}
else
{
lean_inc(v_a_398_);
lean_dec(v___x_390_);
v___x_400_ = lean_box(0);
v_isShared_401_ = v_isSharedCheck_405_;
goto v_resetjp_399_;
}
v_resetjp_399_:
{
lean_object* v___x_403_; 
if (v_isShared_401_ == 0)
{
v___x_403_ = v___x_400_;
goto v_reusejp_402_;
}
else
{
lean_object* v_reuseFailAlloc_404_; 
v_reuseFailAlloc_404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_404_, 0, v_a_398_);
v___x_403_ = v_reuseFailAlloc_404_;
goto v_reusejp_402_;
}
v_reusejp_402_:
{
return v___x_403_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_ext_spec__1___boxed(lean_object* v___x_406_, lean_object* v_sz_407_, lean_object* v_i_408_, lean_object* v_bs_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
size_t v_sz_boxed_416_; size_t v_i_boxed_417_; lean_object* v_res_418_; 
v_sz_boxed_416_ = lean_unbox_usize(v_sz_407_);
lean_dec(v_sz_407_);
v_i_boxed_417_ = lean_unbox_usize(v_i_408_);
lean_dec(v_i_408_);
v_res_418_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_ext_spec__1(v___x_406_, v_sz_boxed_416_, v_i_boxed_417_, v_bs_409_, v___y_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
lean_dec(v___y_412_);
lean_dec_ref(v___y_411_);
lean_dec(v___y_410_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2_spec__2(lean_object* v_msgData_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v___x_425_; lean_object* v_env_426_; lean_object* v___x_427_; lean_object* v_mctx_428_; lean_object* v_lctx_429_; lean_object* v_options_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_425_ = lean_st_ref_get(v___y_423_);
v_env_426_ = lean_ctor_get(v___x_425_, 0);
lean_inc_ref(v_env_426_);
lean_dec(v___x_425_);
v___x_427_ = lean_st_ref_get(v___y_421_);
v_mctx_428_ = lean_ctor_get(v___x_427_, 0);
lean_inc_ref(v_mctx_428_);
lean_dec(v___x_427_);
v_lctx_429_ = lean_ctor_get(v___y_420_, 2);
v_options_430_ = lean_ctor_get(v___y_422_, 2);
lean_inc_ref(v_options_430_);
lean_inc_ref(v_lctx_429_);
v___x_431_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_431_, 0, v_env_426_);
lean_ctor_set(v___x_431_, 1, v_mctx_428_);
lean_ctor_set(v___x_431_, 2, v_lctx_429_);
lean_ctor_set(v___x_431_, 3, v_options_430_);
v___x_432_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_432_, 0, v___x_431_);
lean_ctor_set(v___x_432_, 1, v_msgData_419_);
v___x_433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_433_, 0, v___x_432_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2_spec__2___boxed(lean_object* v_msgData_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2_spec__2(v_msgData_434_, v___y_435_, v___y_436_, v___y_437_, v___y_438_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
lean_dec(v___y_436_);
lean_dec_ref(v___y_435_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___redArg(lean_object* v_msg_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
lean_object* v_ref_447_; lean_object* v___x_448_; lean_object* v_a_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_457_; 
v_ref_447_ = lean_ctor_get(v___y_444_, 5);
v___x_448_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2_spec__2(v_msg_441_, v___y_442_, v___y_443_, v___y_444_, v___y_445_);
v_a_449_ = lean_ctor_get(v___x_448_, 0);
v_isSharedCheck_457_ = !lean_is_exclusive(v___x_448_);
if (v_isSharedCheck_457_ == 0)
{
v___x_451_ = v___x_448_;
v_isShared_452_ = v_isSharedCheck_457_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_a_449_);
lean_dec(v___x_448_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_457_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v___x_453_; lean_object* v___x_455_; 
lean_inc(v_ref_447_);
v___x_453_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_453_, 0, v_ref_447_);
lean_ctor_set(v___x_453_, 1, v_a_449_);
if (v_isShared_452_ == 0)
{
lean_ctor_set_tag(v___x_451_, 1);
lean_ctor_set(v___x_451_, 0, v___x_453_);
v___x_455_ = v___x_451_;
goto v_reusejp_454_;
}
else
{
lean_object* v_reuseFailAlloc_456_; 
v_reuseFailAlloc_456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_456_, 0, v___x_453_);
v___x_455_ = v_reuseFailAlloc_456_;
goto v_reusejp_454_;
}
v_reusejp_454_:
{
return v___x_455_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___redArg___boxed(lean_object* v_msg_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_){
_start:
{
lean_object* v_res_464_; 
v_res_464_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___redArg(v_msg_458_, v___y_459_, v___y_460_, v___y_461_, v___y_462_);
lean_dec(v___y_462_);
lean_dec_ref(v___y_461_);
lean_dec(v___y_460_);
lean_dec_ref(v___y_459_);
return v_res_464_;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_ext___closed__1(void){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_466_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_ext___closed__0));
v___x_467_ = l_Lean_stringToMessageData(v___x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_ext(lean_object* v_a_468_, lean_object* v_a_469_, lean_object* v_a_470_, lean_object* v_a_471_, lean_object* v_a_472_, lean_object* v_a_473_){
_start:
{
lean_object* v_fst_476_; lean_object* v_fst_477_; lean_object* v_snd_478_; lean_object* v_goal_500_; lean_object* v___x_501_; lean_object* v___x_502_; 
v_goal_500_ = lean_ctor_get(v_a_468_, 0);
lean_inc_n(v_goal_500_, 2);
lean_dec_ref(v_a_468_);
v___x_501_ = lean_alloc_closure((void*)(lp_aesop_Aesop_BuiltinRules_extCore___boxed), 8, 1);
lean_closure_set(v___x_501_, 0, v_goal_500_);
v___x_502_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_ext_spec__0___redArg(v___x_501_, v_a_469_, v_a_470_, v_a_471_, v_a_472_, v_a_473_);
if (lean_obj_tag(v___x_502_) == 0)
{
lean_object* v_a_503_; lean_object* v_fst_504_; 
v_a_503_ = lean_ctor_get(v___x_502_, 0);
lean_inc(v_a_503_);
lean_dec_ref_known(v___x_502_, 1);
v_fst_504_ = lean_ctor_get(v_a_503_, 0);
lean_inc(v_fst_504_);
if (lean_obj_tag(v_fst_504_) == 1)
{
lean_object* v_snd_505_; lean_object* v_val_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_526_; 
v_snd_505_ = lean_ctor_get(v_a_503_, 1);
lean_inc(v_snd_505_);
lean_dec(v_a_503_);
v_val_506_ = lean_ctor_get(v_fst_504_, 0);
v_isSharedCheck_526_ = !lean_is_exclusive(v_fst_504_);
if (v_isSharedCheck_526_ == 0)
{
v___x_508_ = v_fst_504_;
v_isShared_509_ = v_isSharedCheck_526_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_val_506_);
lean_dec(v_fst_504_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_526_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
size_t v_sz_510_; size_t v___x_511_; lean_object* v___x_512_; 
v_sz_510_ = lean_array_size(v_val_506_);
v___x_511_ = ((size_t)0ULL);
v___x_512_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_BuiltinRules_ext_spec__1(v_goal_500_, v_sz_510_, v___x_511_, v_val_506_, v_a_469_, v_a_470_, v_a_471_, v_a_472_, v_a_473_);
if (lean_obj_tag(v___x_512_) == 0)
{
lean_object* v_a_513_; lean_object* v___x_515_; 
v_a_513_ = lean_ctor_get(v___x_512_, 0);
lean_inc(v_a_513_);
lean_dec_ref_known(v___x_512_, 1);
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 0, v_snd_505_);
v___x_515_ = v___x_508_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_snd_505_);
v___x_515_ = v_reuseFailAlloc_517_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
lean_object* v___x_516_; 
v___x_516_ = lean_box(0);
v_fst_476_ = v_a_513_;
v_fst_477_ = v___x_515_;
v_snd_478_ = v___x_516_;
goto v___jp_475_;
}
}
else
{
lean_object* v_a_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_525_; 
lean_del_object(v___x_508_);
lean_dec(v_snd_505_);
v_a_518_ = lean_ctor_get(v___x_512_, 0);
v_isSharedCheck_525_ = !lean_is_exclusive(v___x_512_);
if (v_isSharedCheck_525_ == 0)
{
v___x_520_ = v___x_512_;
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_a_518_);
lean_dec(v___x_512_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___x_523_; 
if (v_isShared_521_ == 0)
{
v___x_523_ = v___x_520_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_524_; 
v_reuseFailAlloc_524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_524_, 0, v_a_518_);
v___x_523_ = v_reuseFailAlloc_524_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
return v___x_523_;
}
}
}
}
}
else
{
lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v_a_529_; lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_536_; 
lean_dec(v_fst_504_);
lean_dec(v_a_503_);
lean_dec(v_goal_500_);
v___x_527_ = lean_obj_once(&lp_aesop_Aesop_BuiltinRules_ext___closed__1, &lp_aesop_Aesop_BuiltinRules_ext___closed__1_once, _init_lp_aesop_Aesop_BuiltinRules_ext___closed__1);
v___x_528_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___redArg(v___x_527_, v_a_470_, v_a_471_, v_a_472_, v_a_473_);
v_a_529_ = lean_ctor_get(v___x_528_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v___x_528_);
if (v_isSharedCheck_536_ == 0)
{
v___x_531_ = v___x_528_;
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
else
{
lean_inc(v_a_529_);
lean_dec(v___x_528_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
lean_object* v___x_534_; 
if (v_isShared_532_ == 0)
{
v___x_534_ = v___x_531_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v_a_529_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
}
else
{
lean_object* v_a_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_544_; 
lean_dec(v_goal_500_);
v_a_537_ = lean_ctor_get(v___x_502_, 0);
v_isSharedCheck_544_ = !lean_is_exclusive(v___x_502_);
if (v_isSharedCheck_544_ == 0)
{
v___x_539_ = v___x_502_;
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_a_537_);
lean_dec(v___x_502_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
lean_object* v___x_542_; 
if (v_isShared_540_ == 0)
{
v___x_542_ = v___x_539_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v_a_537_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
return v___x_542_;
}
}
}
v___jp_475_:
{
lean_object* v___x_479_; 
v___x_479_ = l_Lean_Meta_saveState___redArg(v_a_471_, v_a_473_);
if (lean_obj_tag(v___x_479_) == 0)
{
lean_object* v_a_480_; lean_object* v___x_482_; uint8_t v_isShared_483_; uint8_t v_isSharedCheck_491_; 
v_a_480_ = lean_ctor_get(v___x_479_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_491_ == 0)
{
v___x_482_ = v___x_479_;
v_isShared_483_ = v_isSharedCheck_491_;
goto v_resetjp_481_;
}
else
{
lean_inc(v_a_480_);
lean_dec(v___x_479_);
v___x_482_ = lean_box(0);
v_isShared_483_ = v_isSharedCheck_491_;
goto v_resetjp_481_;
}
v_resetjp_481_:
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_489_; 
lean_inc(v_snd_478_);
v___x_484_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_484_, 0, v_fst_476_);
lean_ctor_set(v___x_484_, 1, v_a_480_);
lean_ctor_set(v___x_484_, 2, v_fst_477_);
lean_ctor_set(v___x_484_, 3, v_snd_478_);
v___x_485_ = lean_unsigned_to_nat(1u);
v___x_486_ = lean_mk_empty_array_with_capacity(v___x_485_);
v___x_487_ = lean_array_push(v___x_486_, v___x_484_);
if (v_isShared_483_ == 0)
{
lean_ctor_set(v___x_482_, 0, v___x_487_);
v___x_489_ = v___x_482_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v___x_487_);
v___x_489_ = v_reuseFailAlloc_490_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
return v___x_489_;
}
}
}
else
{
lean_object* v_a_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_499_; 
lean_dec(v_fst_477_);
lean_dec_ref(v_fst_476_);
v_a_492_ = lean_ctor_get(v___x_479_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_499_ == 0)
{
v___x_494_ = v___x_479_;
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_a_492_);
lean_dec(v___x_479_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_497_; 
if (v_isShared_495_ == 0)
{
v___x_497_ = v___x_494_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v_a_492_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_ext___boxed(lean_object* v_a_545_, lean_object* v_a_546_, lean_object* v_a_547_, lean_object* v_a_548_, lean_object* v_a_549_, lean_object* v_a_550_, lean_object* v_a_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_aesop_Aesop_BuiltinRules_ext(v_a_545_, v_a_546_, v_a_547_, v_a_548_, v_a_549_, v_a_550_);
lean_dec(v_a_550_);
lean_dec_ref(v_a_549_);
lean_dec(v_a_548_);
lean_dec_ref(v_a_547_);
lean_dec(v_a_546_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2(lean_object* v_00_u03b1_553_, lean_object* v_msg_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_){
_start:
{
lean_object* v___x_561_; 
v___x_561_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___redArg(v_msg_554_, v___y_556_, v___y_557_, v___y_558_, v___y_559_);
return v___x_561_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2___boxed(lean_object* v_00_u03b1_562_, lean_object* v_msg_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_ext_spec__2(v_00_u03b1_562_, v_msg_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
lean_dec(v___y_564_);
return v_res_570_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Ext(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_BuiltinRules_Ext(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_aesop_Aesop_BuiltinRules_Ext(uint8_t builtin) {
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
res = runtime_initialize_aesop_Aesop_BuiltinRules_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_BuiltinRules_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_BuiltinRules_Ext(builtin);
}
#ifdef __cplusplus
}
#endif
