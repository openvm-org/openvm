// Lean compiler output
// Module: Aesop.Search.Expansion.Basic
// Imports: public import Init public meta import Init public import Aesop.RuleTac.Basic
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
extern lean_object* lp_aesop_Aesop_Check_rules;
lean_object* lp_aesop_Aesop_Check_name(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Check_get(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_aesop_Aesop_RuleApplication_check(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRuleTac_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRuleTac_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__0;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = ": while applying rule "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__2_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__3;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__4;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__5_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__6;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__7_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__8_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__9_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__10_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__11 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__11_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__12 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__12_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__13 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__13_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__14 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__14_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__15 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__15_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__16 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__16_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__17 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__17_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__18 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__18_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__19 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__19_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__20 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__20_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRuleTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRuleTac___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___redArg(lean_object* v_s_1_, lean_object* v_x_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = l_Lean_Meta_saveState___redArg(v___y_5_, v___y_7_);
if (lean_obj_tag(v___x_9_) == 0)
{
lean_object* v_a_10_; lean_object* v_a_12_; lean_object* v___x_30_; 
v_a_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc(v_a_10_);
lean_dec_ref_known(v___x_9_, 1);
v___x_30_ = l_Lean_Meta_SavedState_restore___redArg(v_s_1_, v___y_5_, v___y_7_);
if (lean_obj_tag(v___x_30_) == 0)
{
lean_object* v___x_31_; 
lean_dec_ref_known(v___x_30_, 1);
lean_inc(v___y_7_);
lean_inc_ref(v___y_6_);
lean_inc(v___y_5_);
lean_inc_ref(v___y_4_);
lean_inc(v___y_3_);
v___x_31_ = lean_apply_6(v_x_2_, v___y_3_, v___y_4_, v___y_5_, v___y_6_, v___y_7_, lean_box(0));
if (lean_obj_tag(v___x_31_) == 0)
{
lean_object* v_a_32_; lean_object* v___x_33_; 
v_a_32_ = lean_ctor_get(v___x_31_, 0);
lean_inc(v_a_32_);
lean_dec_ref_known(v___x_31_, 1);
v___x_33_ = l_Lean_Meta_SavedState_restore___redArg(v_a_10_, v___y_5_, v___y_7_);
lean_dec(v_a_10_);
if (lean_obj_tag(v___x_33_) == 0)
{
lean_object* v___x_35_; uint8_t v_isShared_36_; uint8_t v_isSharedCheck_40_; 
v_isSharedCheck_40_ = !lean_is_exclusive(v___x_33_);
if (v_isSharedCheck_40_ == 0)
{
lean_object* v_unused_41_; 
v_unused_41_ = lean_ctor_get(v___x_33_, 0);
lean_dec(v_unused_41_);
v___x_35_ = v___x_33_;
v_isShared_36_ = v_isSharedCheck_40_;
goto v_resetjp_34_;
}
else
{
lean_dec(v___x_33_);
v___x_35_ = lean_box(0);
v_isShared_36_ = v_isSharedCheck_40_;
goto v_resetjp_34_;
}
v_resetjp_34_:
{
lean_object* v___x_38_; 
if (v_isShared_36_ == 0)
{
lean_ctor_set(v___x_35_, 0, v_a_32_);
v___x_38_ = v___x_35_;
goto v_reusejp_37_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v_a_32_);
v___x_38_ = v_reuseFailAlloc_39_;
goto v_reusejp_37_;
}
v_reusejp_37_:
{
return v___x_38_;
}
}
}
else
{
lean_object* v_a_42_; lean_object* v___x_44_; uint8_t v_isShared_45_; uint8_t v_isSharedCheck_49_; 
lean_dec(v_a_32_);
v_a_42_ = lean_ctor_get(v___x_33_, 0);
v_isSharedCheck_49_ = !lean_is_exclusive(v___x_33_);
if (v_isSharedCheck_49_ == 0)
{
v___x_44_ = v___x_33_;
v_isShared_45_ = v_isSharedCheck_49_;
goto v_resetjp_43_;
}
else
{
lean_inc(v_a_42_);
lean_dec(v___x_33_);
v___x_44_ = lean_box(0);
v_isShared_45_ = v_isSharedCheck_49_;
goto v_resetjp_43_;
}
v_resetjp_43_:
{
lean_object* v___x_47_; 
if (v_isShared_45_ == 0)
{
v___x_47_ = v___x_44_;
goto v_reusejp_46_;
}
else
{
lean_object* v_reuseFailAlloc_48_; 
v_reuseFailAlloc_48_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_48_, 0, v_a_42_);
v___x_47_ = v_reuseFailAlloc_48_;
goto v_reusejp_46_;
}
v_reusejp_46_:
{
return v___x_47_;
}
}
}
}
else
{
lean_object* v_a_50_; 
v_a_50_ = lean_ctor_get(v___x_31_, 0);
lean_inc(v_a_50_);
lean_dec_ref_known(v___x_31_, 1);
v_a_12_ = v_a_50_;
goto v___jp_11_;
}
}
else
{
lean_object* v_a_51_; 
lean_dec_ref(v_x_2_);
v_a_51_ = lean_ctor_get(v___x_30_, 0);
lean_inc(v_a_51_);
lean_dec_ref_known(v___x_30_, 1);
v_a_12_ = v_a_51_;
goto v___jp_11_;
}
v___jp_11_:
{
lean_object* v___x_13_; 
v___x_13_ = l_Lean_Meta_SavedState_restore___redArg(v_a_10_, v___y_5_, v___y_7_);
lean_dec(v_a_10_);
if (lean_obj_tag(v___x_13_) == 0)
{
lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_20_; 
v_isSharedCheck_20_ = !lean_is_exclusive(v___x_13_);
if (v_isSharedCheck_20_ == 0)
{
lean_object* v_unused_21_; 
v_unused_21_ = lean_ctor_get(v___x_13_, 0);
lean_dec(v_unused_21_);
v___x_15_ = v___x_13_;
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
else
{
lean_dec(v___x_13_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_18_; 
if (v_isShared_16_ == 0)
{
lean_ctor_set_tag(v___x_15_, 1);
lean_ctor_set(v___x_15_, 0, v_a_12_);
v___x_18_ = v___x_15_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_19_; 
v_reuseFailAlloc_19_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_19_, 0, v_a_12_);
v___x_18_ = v_reuseFailAlloc_19_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
return v___x_18_;
}
}
}
else
{
lean_object* v_a_22_; lean_object* v___x_24_; uint8_t v_isShared_25_; uint8_t v_isSharedCheck_29_; 
lean_dec_ref(v_a_12_);
v_a_22_ = lean_ctor_get(v___x_13_, 0);
v_isSharedCheck_29_ = !lean_is_exclusive(v___x_13_);
if (v_isSharedCheck_29_ == 0)
{
v___x_24_ = v___x_13_;
v_isShared_25_ = v_isSharedCheck_29_;
goto v_resetjp_23_;
}
else
{
lean_inc(v_a_22_);
lean_dec(v___x_13_);
v___x_24_ = lean_box(0);
v_isShared_25_ = v_isSharedCheck_29_;
goto v_resetjp_23_;
}
v_resetjp_23_:
{
lean_object* v___x_27_; 
if (v_isShared_25_ == 0)
{
v___x_27_ = v___x_24_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_28_; 
v_reuseFailAlloc_28_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_28_, 0, v_a_22_);
v___x_27_ = v_reuseFailAlloc_28_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
return v___x_27_;
}
}
}
}
}
else
{
lean_object* v_a_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_59_; 
lean_dec_ref(v_x_2_);
v_a_52_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_59_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_59_ == 0)
{
v___x_54_ = v___x_9_;
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_a_52_);
lean_dec(v___x_9_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_57_; 
if (v_isShared_55_ == 0)
{
v___x_57_ = v___x_54_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_a_52_);
v___x_57_ = v_reuseFailAlloc_58_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
return v___x_57_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___redArg___boxed(lean_object* v_s_60_, lean_object* v_x_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___redArg(v_s_60_, v_x_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_);
lean_dec(v___y_66_);
lean_dec_ref(v___y_65_);
lean_dec(v___y_64_);
lean_dec_ref(v___y_63_);
lean_dec(v___y_62_);
lean_dec_ref(v_s_60_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0(lean_object* v_00_u03b1_69_, lean_object* v_s_70_, lean_object* v_x_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___redArg(v_s_70_, v_x_71_, v___y_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___boxed(lean_object* v_00_u03b1_79_, lean_object* v_s_80_, lean_object* v_x_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0(v_00_u03b1_79_, v_s_80_, v_x_81_, v___y_82_, v___y_83_, v___y_84_, v___y_85_, v___y_86_);
lean_dec(v___y_86_);
lean_dec_ref(v___y_85_);
lean_dec(v___y_84_);
lean_dec_ref(v___y_83_);
lean_dec(v___y_82_);
lean_dec_ref(v_s_80_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___redArg(lean_object* v_opt_89_, lean_object* v___y_90_){
_start:
{
lean_object* v_options_92_; uint8_t v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v_options_92_ = lean_ctor_get(v___y_90_, 2);
v___x_93_ = lp_aesop_Aesop_Check_get(v_options_92_, v_opt_89_);
v___x_94_ = lean_box(v___x_93_);
v___x_95_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___redArg___boxed(lean_object* v_opt_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___redArg(v_opt_96_, v___y_97_);
lean_dec_ref(v___y_97_);
lean_dec_ref(v_opt_96_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1(lean_object* v_opt_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___redArg(v_opt_100_, v___y_104_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___boxed(lean_object* v_opt_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1(v_opt_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec(v___y_109_);
lean_dec_ref(v_opt_108_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRuleTac_spec__2_spec__2(lean_object* v_msgData_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v___x_122_; lean_object* v_env_123_; lean_object* v___x_124_; lean_object* v_mctx_125_; lean_object* v_lctx_126_; lean_object* v_options_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_122_ = lean_st_ref_get(v___y_120_);
v_env_123_ = lean_ctor_get(v___x_122_, 0);
lean_inc_ref(v_env_123_);
lean_dec(v___x_122_);
v___x_124_ = lean_st_ref_get(v___y_118_);
v_mctx_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc_ref(v_mctx_125_);
lean_dec(v___x_124_);
v_lctx_126_ = lean_ctor_get(v___y_117_, 2);
v_options_127_ = lean_ctor_get(v___y_119_, 2);
lean_inc_ref(v_options_127_);
lean_inc_ref(v_lctx_126_);
v___x_128_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_128_, 0, v_env_123_);
lean_ctor_set(v___x_128_, 1, v_mctx_125_);
lean_ctor_set(v___x_128_, 2, v_lctx_126_);
lean_ctor_set(v___x_128_, 3, v_options_127_);
v___x_129_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v_msgData_116_);
v___x_130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRuleTac_spec__2_spec__2___boxed(lean_object* v_msgData_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRuleTac_spec__2_spec__2(v_msgData_131_, v___y_132_, v___y_133_, v___y_134_, v___y_135_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___redArg(lean_object* v_msg_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_){
_start:
{
lean_object* v_ref_144_; lean_object* v___x_145_; lean_object* v_a_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_154_; 
v_ref_144_ = lean_ctor_get(v___y_141_, 5);
v___x_145_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRuleTac_spec__2_spec__2(v_msg_138_, v___y_139_, v___y_140_, v___y_141_, v___y_142_);
v_a_146_ = lean_ctor_get(v___x_145_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_145_);
if (v_isSharedCheck_154_ == 0)
{
v___x_148_ = v___x_145_;
v_isShared_149_ = v_isSharedCheck_154_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_a_146_);
lean_dec(v___x_145_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_154_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___x_150_; lean_object* v___x_152_; 
lean_inc(v_ref_144_);
v___x_150_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_150_, 0, v_ref_144_);
lean_ctor_set(v___x_150_, 1, v_a_146_);
if (v_isShared_149_ == 0)
{
lean_ctor_set_tag(v___x_148_, 1);
lean_ctor_set(v___x_148_, 0, v___x_150_);
v___x_152_ = v___x_148_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v___x_150_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___redArg___boxed(lean_object* v_msg_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___redArg(v_msg_155_, v___y_156_, v___y_157_, v___y_158_, v___y_159_);
lean_dec(v___y_159_);
lean_dec_ref(v___y_158_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
return v_res_161_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__0(void){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_162_ = lp_aesop_Aesop_Check_rules;
v___x_163_ = lp_aesop_Aesop_Check_name(v___x_162_);
return v___x_163_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__1(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__0, &lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__0_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__0);
v___x_165_ = l_Lean_MessageData_ofName(v___x_164_);
return v___x_165_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__3(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_167_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__2));
v___x_168_ = l_Lean_stringToMessageData(v___x_167_);
return v___x_168_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__4(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_169_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__3);
v___x_170_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__1);
v___x_171_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_171_, 0, v___x_170_);
lean_ctor_set(v___x_171_, 1, v___x_169_);
return v___x_171_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__6(void){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_173_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__5));
v___x_174_ = l_Lean_stringToMessageData(v___x_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3(lean_object* v_input_189_, lean_object* v_ruleName_190_, uint8_t v_a_191_, lean_object* v_as_192_, size_t v_i_193_, size_t v_stop_194_, lean_object* v_b_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_){
_start:
{
lean_object* v_a_203_; uint8_t v___x_207_; 
v___x_207_ = lean_usize_dec_eq(v_i_193_, v_stop_194_);
if (v___x_207_ == 0)
{
lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_208_ = lean_array_uget_borrowed(v_as_192_, v_i_193_);
lean_inc_ref(v_input_189_);
lean_inc(v___x_208_);
v___x_209_ = lp_aesop_Aesop_RuleApplication_check(v___x_208_, v_input_189_, v___y_196_, v___y_197_, v___y_198_, v___y_199_, v___y_200_);
if (lean_obj_tag(v___x_209_) == 0)
{
lean_object* v_a_210_; 
v_a_210_ = lean_ctor_get(v___x_209_, 0);
lean_inc(v_a_210_);
lean_dec_ref_known(v___x_209_, 1);
if (lean_obj_tag(v_a_210_) == 1)
{
lean_object* v_val_211_; lean_object* v___x_213_; uint8_t v_isShared_214_; uint8_t v_isSharedCheck_261_; 
v_val_211_ = lean_ctor_get(v_a_210_, 0);
v_isSharedCheck_261_ = !lean_is_exclusive(v_a_210_);
if (v_isSharedCheck_261_ == 0)
{
v___x_213_ = v_a_210_;
v_isShared_214_ = v_isSharedCheck_261_;
goto v_resetjp_212_;
}
else
{
lean_inc(v_val_211_);
lean_dec(v_a_210_);
v___x_213_ = lean_box(0);
v_isShared_214_ = v_isSharedCheck_261_;
goto v_resetjp_212_;
}
v_resetjp_212_:
{
lean_object* v_name_215_; uint8_t v_builder_216_; uint8_t v_phase_217_; uint8_t v_scope_218_; lean_object* v___x_219_; lean_object* v___y_221_; lean_object* v___y_222_; lean_object* v___y_223_; lean_object* v___y_239_; lean_object* v___y_240_; lean_object* v___y_241_; lean_object* v___y_247_; 
v_name_215_ = lean_ctor_get(v_ruleName_190_, 0);
v_builder_216_ = lean_ctor_get_uint8(v_ruleName_190_, sizeof(void*)*1 + 8);
v_phase_217_ = lean_ctor_get_uint8(v_ruleName_190_, sizeof(void*)*1 + 9);
v_scope_218_ = lean_ctor_get_uint8(v_ruleName_190_, sizeof(void*)*1 + 10);
v___x_219_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__4, &lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__4_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__4);
switch(v_phase_217_)
{
case 0:
{
lean_object* v___x_258_; 
v___x_258_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__18));
v___y_247_ = v___x_258_;
goto v___jp_246_;
}
case 1:
{
lean_object* v___x_259_; 
v___x_259_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__19));
v___y_247_ = v___x_259_;
goto v___jp_246_;
}
default: 
{
lean_object* v___x_260_; 
v___x_260_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__20));
v___y_247_ = v___x_260_;
goto v___jp_246_;
}
}
v___jp_220_:
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_229_; 
v___x_224_ = lean_string_append(v___y_221_, v___y_223_);
v___x_225_ = lean_string_append(v___x_224_, v___y_222_);
lean_inc(v_name_215_);
v___x_226_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_215_, v_a_191_);
v___x_227_ = lean_string_append(v___x_225_, v___x_226_);
lean_dec_ref(v___x_226_);
if (v_isShared_214_ == 0)
{
lean_ctor_set_tag(v___x_213_, 3);
lean_ctor_set(v___x_213_, 0, v___x_227_);
v___x_229_ = v___x_213_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v___x_227_);
v___x_229_ = v_reuseFailAlloc_237_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_230_ = l_Lean_MessageData_ofFormat(v___x_229_);
v___x_231_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_219_);
lean_ctor_set(v___x_231_, 1, v___x_230_);
v___x_232_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__6, &lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__6_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__6);
v___x_233_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_231_);
lean_ctor_set(v___x_233_, 1, v___x_232_);
v___x_234_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v_val_211_);
v___x_235_ = lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___redArg(v___x_234_, v___y_197_, v___y_198_, v___y_199_, v___y_200_);
if (lean_obj_tag(v___x_235_) == 0)
{
lean_object* v_a_236_; 
v_a_236_ = lean_ctor_get(v___x_235_, 0);
lean_inc(v_a_236_);
lean_dec_ref_known(v___x_235_, 1);
v_a_203_ = v_a_236_;
goto v___jp_202_;
}
else
{
lean_dec_ref(v_ruleName_190_);
lean_dec_ref(v_input_189_);
return v___x_235_;
}
}
}
v___jp_238_:
{
lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_242_ = lean_string_append(v___y_239_, v___y_241_);
v___x_243_ = lean_string_append(v___x_242_, v___y_240_);
if (v_scope_218_ == 0)
{
lean_object* v___x_244_; 
v___x_244_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__7));
v___y_221_ = v___x_243_;
v___y_222_ = v___y_240_;
v___y_223_ = v___x_244_;
goto v___jp_220_;
}
else
{
lean_object* v___x_245_; 
v___x_245_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__8));
v___y_221_ = v___x_243_;
v___y_222_ = v___y_240_;
v___y_223_ = v___x_245_;
goto v___jp_220_;
}
}
v___jp_246_:
{
lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_248_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__9));
lean_inc_ref(v___y_247_);
v___x_249_ = lean_string_append(v___y_247_, v___x_248_);
switch(v_builder_216_)
{
case 0:
{
lean_object* v___x_250_; 
v___x_250_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__10));
v___y_239_ = v___x_249_;
v___y_240_ = v___x_248_;
v___y_241_ = v___x_250_;
goto v___jp_238_;
}
case 1:
{
lean_object* v___x_251_; 
v___x_251_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__11));
v___y_239_ = v___x_249_;
v___y_240_ = v___x_248_;
v___y_241_ = v___x_251_;
goto v___jp_238_;
}
case 2:
{
lean_object* v___x_252_; 
v___x_252_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__12));
v___y_239_ = v___x_249_;
v___y_240_ = v___x_248_;
v___y_241_ = v___x_252_;
goto v___jp_238_;
}
case 3:
{
lean_object* v___x_253_; 
v___x_253_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__13));
v___y_239_ = v___x_249_;
v___y_240_ = v___x_248_;
v___y_241_ = v___x_253_;
goto v___jp_238_;
}
case 4:
{
lean_object* v___x_254_; 
v___x_254_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__14));
v___y_239_ = v___x_249_;
v___y_240_ = v___x_248_;
v___y_241_ = v___x_254_;
goto v___jp_238_;
}
case 5:
{
lean_object* v___x_255_; 
v___x_255_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__15));
v___y_239_ = v___x_249_;
v___y_240_ = v___x_248_;
v___y_241_ = v___x_255_;
goto v___jp_238_;
}
case 6:
{
lean_object* v___x_256_; 
v___x_256_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__16));
v___y_239_ = v___x_249_;
v___y_240_ = v___x_248_;
v___y_241_ = v___x_256_;
goto v___jp_238_;
}
default: 
{
lean_object* v___x_257_; 
v___x_257_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___closed__17));
v___y_239_ = v___x_249_;
v___y_240_ = v___x_248_;
v___y_241_ = v___x_257_;
goto v___jp_238_;
}
}
}
}
}
else
{
lean_object* v___x_262_; 
lean_dec(v_a_210_);
v___x_262_ = lean_box(0);
v_a_203_ = v___x_262_;
goto v___jp_202_;
}
}
else
{
lean_object* v_a_263_; lean_object* v___x_265_; uint8_t v_isShared_266_; uint8_t v_isSharedCheck_270_; 
lean_dec_ref(v_ruleName_190_);
lean_dec_ref(v_input_189_);
v_a_263_ = lean_ctor_get(v___x_209_, 0);
v_isSharedCheck_270_ = !lean_is_exclusive(v___x_209_);
if (v_isSharedCheck_270_ == 0)
{
v___x_265_ = v___x_209_;
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
else
{
lean_inc(v_a_263_);
lean_dec(v___x_209_);
v___x_265_ = lean_box(0);
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
v_resetjp_264_:
{
lean_object* v___x_268_; 
if (v_isShared_266_ == 0)
{
v___x_268_ = v___x_265_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v_a_263_);
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
lean_object* v___x_271_; 
lean_dec_ref(v_ruleName_190_);
lean_dec_ref(v_input_189_);
v___x_271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_271_, 0, v_b_195_);
return v___x_271_;
}
v___jp_202_:
{
size_t v___x_204_; size_t v___x_205_; 
v___x_204_ = ((size_t)1ULL);
v___x_205_ = lean_usize_add(v_i_193_, v___x_204_);
v_i_193_ = v___x_205_;
v_b_195_ = v_a_203_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3___boxed(lean_object* v_input_272_, lean_object* v_ruleName_273_, lean_object* v_a_274_, lean_object* v_as_275_, lean_object* v_i_276_, lean_object* v_stop_277_, lean_object* v_b_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_){
_start:
{
uint8_t v_a_8472__boxed_285_; size_t v_i_boxed_286_; size_t v_stop_boxed_287_; lean_object* v_res_288_; 
v_a_8472__boxed_285_ = lean_unbox(v_a_274_);
v_i_boxed_286_ = lean_unbox_usize(v_i_276_);
lean_dec(v_i_276_);
v_stop_boxed_287_ = lean_unbox_usize(v_stop_277_);
lean_dec(v_stop_277_);
v_res_288_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3(v_input_272_, v_ruleName_273_, v_a_8472__boxed_285_, v_as_275_, v_i_boxed_286_, v_stop_boxed_287_, v_b_278_, v___y_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_);
lean_dec(v___y_283_);
lean_dec_ref(v___y_282_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec(v___y_279_);
lean_dec_ref(v_as_275_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRuleTac(lean_object* v_tac_289_, lean_object* v_ruleName_290_, lean_object* v_preState_291_, lean_object* v_input_292_, lean_object* v_a_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; 
lean_inc_ref(v_input_292_);
v___x_299_ = lean_apply_1(v_tac_289_, v_input_292_);
v___x_300_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_runRuleTac_spec__0___redArg(v_preState_291_, v___x_299_, v_a_293_, v_a_294_, v_a_295_, v_a_296_, v_a_297_);
if (lean_obj_tag(v___x_300_) == 0)
{
lean_object* v_a_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v_a_304_; lean_object* v___x_306_; uint8_t v_isShared_307_; uint8_t v_isSharedCheck_350_; 
v_a_301_ = lean_ctor_get(v___x_300_, 0);
lean_inc(v_a_301_);
lean_dec_ref_known(v___x_300_, 1);
v___x_302_ = lp_aesop_Aesop_Check_rules;
v___x_303_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_runRuleTac_spec__1___redArg(v___x_302_, v_a_296_);
v_a_304_ = lean_ctor_get(v___x_303_, 0);
v_isSharedCheck_350_ = !lean_is_exclusive(v___x_303_);
if (v_isSharedCheck_350_ == 0)
{
v___x_306_ = v___x_303_;
v_isShared_307_ = v_isSharedCheck_350_;
goto v_resetjp_305_;
}
else
{
lean_inc(v_a_304_);
lean_dec(v___x_303_);
v___x_306_ = lean_box(0);
v_isShared_307_ = v_isSharedCheck_350_;
goto v_resetjp_305_;
}
v_resetjp_305_:
{
lean_object* v___x_308_; lean_object* v___y_310_; uint8_t v___x_327_; 
lean_inc(v_a_301_);
v___x_308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_308_, 0, v_a_301_);
v___x_327_ = lean_unbox(v_a_304_);
if (v___x_327_ == 0)
{
lean_object* v___x_329_; 
lean_dec(v_a_304_);
lean_dec(v_a_301_);
lean_dec_ref(v_input_292_);
lean_dec_ref(v_ruleName_290_);
if (v_isShared_307_ == 0)
{
lean_ctor_set(v___x_306_, 0, v___x_308_);
v___x_329_ = v___x_306_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v___x_308_);
v___x_329_ = v_reuseFailAlloc_330_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
return v___x_329_;
}
}
else
{
lean_object* v___x_331_; lean_object* v___x_332_; uint8_t v___x_333_; 
v___x_331_ = lean_unsigned_to_nat(0u);
v___x_332_ = lean_array_get_size(v_a_301_);
v___x_333_ = lean_nat_dec_lt(v___x_331_, v___x_332_);
if (v___x_333_ == 0)
{
lean_object* v___x_335_; 
lean_dec(v_a_304_);
lean_dec(v_a_301_);
lean_dec_ref(v_input_292_);
lean_dec_ref(v_ruleName_290_);
if (v_isShared_307_ == 0)
{
lean_ctor_set(v___x_306_, 0, v___x_308_);
v___x_335_ = v___x_306_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v___x_308_);
v___x_335_ = v_reuseFailAlloc_336_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
return v___x_335_;
}
}
else
{
lean_object* v___x_337_; uint8_t v___x_338_; 
v___x_337_ = lean_box(0);
v___x_338_ = lean_nat_dec_le(v___x_332_, v___x_332_);
if (v___x_338_ == 0)
{
if (v___x_333_ == 0)
{
lean_object* v___x_340_; 
lean_dec(v_a_304_);
lean_dec(v_a_301_);
lean_dec_ref(v_input_292_);
lean_dec_ref(v_ruleName_290_);
if (v_isShared_307_ == 0)
{
lean_ctor_set(v___x_306_, 0, v___x_308_);
v___x_340_ = v___x_306_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v___x_308_);
v___x_340_ = v_reuseFailAlloc_341_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
return v___x_340_;
}
}
else
{
size_t v___x_342_; size_t v___x_343_; uint8_t v___x_344_; lean_object* v___x_345_; 
lean_del_object(v___x_306_);
v___x_342_ = ((size_t)0ULL);
v___x_343_ = lean_usize_of_nat(v___x_332_);
v___x_344_ = lean_unbox(v_a_304_);
lean_dec(v_a_304_);
v___x_345_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3(v_input_292_, v_ruleName_290_, v___x_344_, v_a_301_, v___x_342_, v___x_343_, v___x_337_, v_a_293_, v_a_294_, v_a_295_, v_a_296_, v_a_297_);
lean_dec(v_a_301_);
v___y_310_ = v___x_345_;
goto v___jp_309_;
}
}
else
{
size_t v___x_346_; size_t v___x_347_; uint8_t v___x_348_; lean_object* v___x_349_; 
lean_del_object(v___x_306_);
v___x_346_ = ((size_t)0ULL);
v___x_347_ = lean_usize_of_nat(v___x_332_);
v___x_348_ = lean_unbox(v_a_304_);
lean_dec(v_a_304_);
v___x_349_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_runRuleTac_spec__3(v_input_292_, v_ruleName_290_, v___x_348_, v_a_301_, v___x_346_, v___x_347_, v___x_337_, v_a_293_, v_a_294_, v_a_295_, v_a_296_, v_a_297_);
lean_dec(v_a_301_);
v___y_310_ = v___x_349_;
goto v___jp_309_;
}
}
}
v___jp_309_:
{
if (lean_obj_tag(v___y_310_) == 0)
{
lean_object* v___x_312_; uint8_t v_isShared_313_; uint8_t v_isSharedCheck_317_; 
v_isSharedCheck_317_ = !lean_is_exclusive(v___y_310_);
if (v_isSharedCheck_317_ == 0)
{
lean_object* v_unused_318_; 
v_unused_318_ = lean_ctor_get(v___y_310_, 0);
lean_dec(v_unused_318_);
v___x_312_ = v___y_310_;
v_isShared_313_ = v_isSharedCheck_317_;
goto v_resetjp_311_;
}
else
{
lean_dec(v___y_310_);
v___x_312_ = lean_box(0);
v_isShared_313_ = v_isSharedCheck_317_;
goto v_resetjp_311_;
}
v_resetjp_311_:
{
lean_object* v___x_315_; 
if (v_isShared_313_ == 0)
{
lean_ctor_set(v___x_312_, 0, v___x_308_);
v___x_315_ = v___x_312_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_316_; 
v_reuseFailAlloc_316_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_316_, 0, v___x_308_);
v___x_315_ = v_reuseFailAlloc_316_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
return v___x_315_;
}
}
}
else
{
lean_object* v_a_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_326_; 
lean_dec_ref_known(v___x_308_, 1);
v_a_319_ = lean_ctor_get(v___y_310_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___y_310_);
if (v_isSharedCheck_326_ == 0)
{
v___x_321_ = v___y_310_;
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_a_319_);
lean_dec(v___y_310_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v___x_324_; 
if (v_isShared_322_ == 0)
{
v___x_324_ = v___x_321_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v_a_319_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
}
}
}
else
{
lean_object* v_a_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_366_; 
lean_dec_ref(v_input_292_);
lean_dec_ref(v_ruleName_290_);
v_a_351_ = lean_ctor_get(v___x_300_, 0);
v_isSharedCheck_366_ = !lean_is_exclusive(v___x_300_);
if (v_isSharedCheck_366_ == 0)
{
v___x_353_ = v___x_300_;
v_isShared_354_ = v_isSharedCheck_366_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_a_351_);
lean_dec(v___x_300_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_366_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
uint8_t v___y_356_; uint8_t v___x_364_; 
v___x_364_ = l_Lean_Exception_isInterrupt(v_a_351_);
if (v___x_364_ == 0)
{
uint8_t v___x_365_; 
lean_inc(v_a_351_);
v___x_365_ = l_Lean_Exception_isRuntime(v_a_351_);
v___y_356_ = v___x_365_;
goto v___jp_355_;
}
else
{
v___y_356_ = v___x_364_;
goto v___jp_355_;
}
v___jp_355_:
{
if (v___y_356_ == 0)
{
lean_object* v___x_357_; lean_object* v___x_359_; 
v___x_357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_357_, 0, v_a_351_);
if (v_isShared_354_ == 0)
{
lean_ctor_set_tag(v___x_353_, 0);
lean_ctor_set(v___x_353_, 0, v___x_357_);
v___x_359_ = v___x_353_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v___x_357_);
v___x_359_ = v_reuseFailAlloc_360_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
return v___x_359_;
}
}
else
{
lean_object* v___x_362_; 
if (v_isShared_354_ == 0)
{
v___x_362_ = v___x_353_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_363_; 
v_reuseFailAlloc_363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_363_, 0, v_a_351_);
v___x_362_ = v_reuseFailAlloc_363_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
return v___x_362_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRuleTac___boxed(lean_object* v_tac_367_, lean_object* v_ruleName_368_, lean_object* v_preState_369_, lean_object* v_input_370_, lean_object* v_a_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_aesop_Aesop_runRuleTac(v_tac_367_, v_ruleName_368_, v_preState_369_, v_input_370_, v_a_371_, v_a_372_, v_a_373_, v_a_374_, v_a_375_);
lean_dec(v_a_375_);
lean_dec_ref(v_a_374_);
lean_dec(v_a_373_);
lean_dec_ref(v_a_372_);
lean_dec(v_a_371_);
lean_dec_ref(v_preState_369_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2(lean_object* v_00_u03b1_378_, lean_object* v_msg_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___redArg(v_msg_379_, v___y_381_, v___y_382_, v___y_383_, v___y_384_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2___boxed(lean_object* v_00_u03b1_387_, lean_object* v_msg_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_aesop_Lean_throwError___at___00Aesop_runRuleTac_spec__2(v_00_u03b1_387_, v_msg_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_, v___y_393_);
lean_dec(v___y_393_);
lean_dec_ref(v___y_392_);
lean_dec(v___y_391_);
lean_dec_ref(v___y_390_);
lean_dec(v___y_389_);
return v_res_395_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_Expansion_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_Expansion_Basic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Expansion_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Expansion_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_Expansion_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_Expansion_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
