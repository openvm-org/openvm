// Lean compiler output
// Module: Aesop.Forward.State.UpdateGoal
// Imports: public import Init public meta import Init public import Aesop.Tree.TreeM import Aesop.Tree.RunMetaM
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
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lp_aesop_Aesop_ForwardState_update(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleMatches_update(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Goal_isNormal(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "aesop: internal error: expected goal "};
static const lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__1;
static const lean_string_object lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = " to be normalised (but not proven by normalisation)."};
static const lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__0;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__1;
static const lean_array_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__2 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__2_value;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "aesop: internal error: "};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__3 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__4;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__5 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__5_value;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "GoalRef"};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__6 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__6_value;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "updateForwardState"};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__7 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__8_value_aux_0),((lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__6_value),LEAN_SCALAR_PTR_LITERAL(144, 218, 39, 213, 2, 247, 209, 131)}};
static const lean_ctor_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__8_value_aux_1),((lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__7_value),LEAN_SCALAR_PTR_LITERAL(128, 63, 180, 129, 13, 202, 171, 28)}};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__8 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__9;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__10;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = ": attempt to update forward state of non-normal goal "};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__11 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__11_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__12;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__13;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " to phase "};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__14 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__14_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__15;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__16 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__16_value;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__17 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__17_value;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__18 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__18_value;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = ": at goal "};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__19 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__19_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__20;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__21;
static const lean_string_object lp_aesop_Aesop_GoalRef_updateForwardState___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = ": norm phase not supported"};
static const lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__22 = (const lean_object*)&lp_aesop_Aesop_GoalRef_updateForwardState___closed__22_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_updateForwardState___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___closed__23;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_updateForwardState(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___lam__0(lean_object* v_val_1_, uint8_t v_phase_2_, lean_object* v_goal_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_, lean_object* v___y_8_, lean_object* v___y_9_, lean_object* v___y_10_){
_start:
{
lean_object* v___x_12_; lean_object* v_elimGoal_13_; lean_object* v___x_14_; lean_object* v_forwardState_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_12_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_13_ = lean_ctor_get(v___x_12_, 1);
lean_inc_ref(v_elimGoal_13_);
v___x_14_ = lean_apply_1(v_elimGoal_13_, v_val_1_);
v_forwardState_15_ = lean_ctor_get(v___x_14_, 8);
lean_inc_ref(v_forwardState_15_);
lean_dec_ref(v___x_14_);
v___x_16_ = lean_box(v_phase_2_);
v___x_17_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
v___x_18_ = lp_aesop_Aesop_ForwardState_update(v_goal_3_, v_forwardState_15_, v___x_17_, v___y_6_, v___y_7_, v___y_8_, v___y_9_, v___y_10_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___lam__0___boxed(lean_object* v_val_19_, lean_object* v_phase_20_, lean_object* v_goal_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_){
_start:
{
uint8_t v_phase_boxed_30_; lean_object* v_res_31_; 
v_phase_boxed_30_ = lean_unbox(v_phase_20_);
v_res_31_ = lp_aesop_Aesop_GoalRef_updateForwardState___lam__0(v_val_19_, v_phase_boxed_30_, v_goal_21_, v___y_22_, v___y_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_, v___y_28_);
lean_dec(v___y_28_);
lean_dec_ref(v___y_27_);
lean_dec(v___y_26_);
lean_dec_ref(v___y_25_);
lean_dec(v___y_24_);
lean_dec(v___y_23_);
lean_dec_ref(v___y_22_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___redArg(lean_object* v_s_32_, lean_object* v_x_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = l_Lean_Meta_saveState___redArg(v___y_38_, v___y_40_);
if (lean_obj_tag(v___x_42_) == 0)
{
lean_object* v_a_43_; lean_object* v_a_45_; lean_object* v___x_63_; 
v_a_43_ = lean_ctor_get(v___x_42_, 0);
lean_inc(v_a_43_);
lean_dec_ref_known(v___x_42_, 1);
v___x_63_ = l_Lean_Meta_SavedState_restore___redArg(v_s_32_, v___y_38_, v___y_40_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_object* v___x_64_; 
lean_dec_ref_known(v___x_63_, 1);
lean_inc(v___y_40_);
lean_inc_ref(v___y_39_);
lean_inc(v___y_38_);
lean_inc_ref(v___y_37_);
lean_inc(v___y_36_);
lean_inc(v___y_35_);
lean_inc_ref(v___y_34_);
v___x_64_ = lean_apply_8(v_x_33_, v___y_34_, v___y_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_, v___y_40_, lean_box(0));
if (lean_obj_tag(v___x_64_) == 0)
{
lean_object* v_a_65_; lean_object* v___x_66_; 
v_a_65_ = lean_ctor_get(v___x_64_, 0);
lean_inc(v_a_65_);
lean_dec_ref_known(v___x_64_, 1);
v___x_66_ = l_Lean_Meta_SavedState_restore___redArg(v_a_43_, v___y_38_, v___y_40_);
lean_dec(v_a_43_);
if (lean_obj_tag(v___x_66_) == 0)
{
lean_object* v___x_68_; uint8_t v_isShared_69_; uint8_t v_isSharedCheck_73_; 
v_isSharedCheck_73_ = !lean_is_exclusive(v___x_66_);
if (v_isSharedCheck_73_ == 0)
{
lean_object* v_unused_74_; 
v_unused_74_ = lean_ctor_get(v___x_66_, 0);
lean_dec(v_unused_74_);
v___x_68_ = v___x_66_;
v_isShared_69_ = v_isSharedCheck_73_;
goto v_resetjp_67_;
}
else
{
lean_dec(v___x_66_);
v___x_68_ = lean_box(0);
v_isShared_69_ = v_isSharedCheck_73_;
goto v_resetjp_67_;
}
v_resetjp_67_:
{
lean_object* v___x_71_; 
if (v_isShared_69_ == 0)
{
lean_ctor_set(v___x_68_, 0, v_a_65_);
v___x_71_ = v___x_68_;
goto v_reusejp_70_;
}
else
{
lean_object* v_reuseFailAlloc_72_; 
v_reuseFailAlloc_72_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_72_, 0, v_a_65_);
v___x_71_ = v_reuseFailAlloc_72_;
goto v_reusejp_70_;
}
v_reusejp_70_:
{
return v___x_71_;
}
}
}
else
{
lean_object* v_a_75_; lean_object* v___x_77_; uint8_t v_isShared_78_; uint8_t v_isSharedCheck_82_; 
lean_dec(v_a_65_);
v_a_75_ = lean_ctor_get(v___x_66_, 0);
v_isSharedCheck_82_ = !lean_is_exclusive(v___x_66_);
if (v_isSharedCheck_82_ == 0)
{
v___x_77_ = v___x_66_;
v_isShared_78_ = v_isSharedCheck_82_;
goto v_resetjp_76_;
}
else
{
lean_inc(v_a_75_);
lean_dec(v___x_66_);
v___x_77_ = lean_box(0);
v_isShared_78_ = v_isSharedCheck_82_;
goto v_resetjp_76_;
}
v_resetjp_76_:
{
lean_object* v___x_80_; 
if (v_isShared_78_ == 0)
{
v___x_80_ = v___x_77_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_81_; 
v_reuseFailAlloc_81_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_81_, 0, v_a_75_);
v___x_80_ = v_reuseFailAlloc_81_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
return v___x_80_;
}
}
}
}
else
{
lean_object* v_a_83_; 
v_a_83_ = lean_ctor_get(v___x_64_, 0);
lean_inc(v_a_83_);
lean_dec_ref_known(v___x_64_, 1);
v_a_45_ = v_a_83_;
goto v___jp_44_;
}
}
else
{
lean_object* v_a_84_; 
lean_dec_ref(v_x_33_);
v_a_84_ = lean_ctor_get(v___x_63_, 0);
lean_inc(v_a_84_);
lean_dec_ref_known(v___x_63_, 1);
v_a_45_ = v_a_84_;
goto v___jp_44_;
}
v___jp_44_:
{
lean_object* v___x_46_; 
v___x_46_ = l_Lean_Meta_SavedState_restore___redArg(v_a_43_, v___y_38_, v___y_40_);
lean_dec(v_a_43_);
if (lean_obj_tag(v___x_46_) == 0)
{
lean_object* v___x_48_; uint8_t v_isShared_49_; uint8_t v_isSharedCheck_53_; 
v_isSharedCheck_53_ = !lean_is_exclusive(v___x_46_);
if (v_isSharedCheck_53_ == 0)
{
lean_object* v_unused_54_; 
v_unused_54_ = lean_ctor_get(v___x_46_, 0);
lean_dec(v_unused_54_);
v___x_48_ = v___x_46_;
v_isShared_49_ = v_isSharedCheck_53_;
goto v_resetjp_47_;
}
else
{
lean_dec(v___x_46_);
v___x_48_ = lean_box(0);
v_isShared_49_ = v_isSharedCheck_53_;
goto v_resetjp_47_;
}
v_resetjp_47_:
{
lean_object* v___x_51_; 
if (v_isShared_49_ == 0)
{
lean_ctor_set_tag(v___x_48_, 1);
lean_ctor_set(v___x_48_, 0, v_a_45_);
v___x_51_ = v___x_48_;
goto v_reusejp_50_;
}
else
{
lean_object* v_reuseFailAlloc_52_; 
v_reuseFailAlloc_52_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_52_, 0, v_a_45_);
v___x_51_ = v_reuseFailAlloc_52_;
goto v_reusejp_50_;
}
v_reusejp_50_:
{
return v___x_51_;
}
}
}
else
{
lean_object* v_a_55_; lean_object* v___x_57_; uint8_t v_isShared_58_; uint8_t v_isSharedCheck_62_; 
lean_dec_ref(v_a_45_);
v_a_55_ = lean_ctor_get(v___x_46_, 0);
v_isSharedCheck_62_ = !lean_is_exclusive(v___x_46_);
if (v_isSharedCheck_62_ == 0)
{
v___x_57_ = v___x_46_;
v_isShared_58_ = v_isSharedCheck_62_;
goto v_resetjp_56_;
}
else
{
lean_inc(v_a_55_);
lean_dec(v___x_46_);
v___x_57_ = lean_box(0);
v_isShared_58_ = v_isSharedCheck_62_;
goto v_resetjp_56_;
}
v_resetjp_56_:
{
lean_object* v___x_60_; 
if (v_isShared_58_ == 0)
{
v___x_60_ = v___x_57_;
goto v_reusejp_59_;
}
else
{
lean_object* v_reuseFailAlloc_61_; 
v_reuseFailAlloc_61_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_61_, 0, v_a_55_);
v___x_60_ = v_reuseFailAlloc_61_;
goto v_reusejp_59_;
}
v_reusejp_59_:
{
return v___x_60_;
}
}
}
}
}
else
{
lean_object* v_a_85_; lean_object* v___x_87_; uint8_t v_isShared_88_; uint8_t v_isSharedCheck_92_; 
lean_dec_ref(v_x_33_);
v_a_85_ = lean_ctor_get(v___x_42_, 0);
v_isSharedCheck_92_ = !lean_is_exclusive(v___x_42_);
if (v_isSharedCheck_92_ == 0)
{
v___x_87_ = v___x_42_;
v_isShared_88_ = v_isSharedCheck_92_;
goto v_resetjp_86_;
}
else
{
lean_inc(v_a_85_);
lean_dec(v___x_42_);
v___x_87_ = lean_box(0);
v_isShared_88_ = v_isSharedCheck_92_;
goto v_resetjp_86_;
}
v_resetjp_86_:
{
lean_object* v___x_90_; 
if (v_isShared_88_ == 0)
{
v___x_90_ = v___x_87_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_91_; 
v_reuseFailAlloc_91_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_91_, 0, v_a_85_);
v___x_90_ = v_reuseFailAlloc_91_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
return v___x_90_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___redArg___boxed(lean_object* v_s_93_, lean_object* v_x_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___redArg(v_s_93_, v_x_94_, v___y_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_, v___y_100_, v___y_101_);
lean_dec(v___y_101_);
lean_dec_ref(v___y_100_);
lean_dec(v___y_99_);
lean_dec_ref(v___y_98_);
lean_dec(v___y_97_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
lean_dec_ref(v_s_93_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1_spec__2(lean_object* v_msgData_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_){
_start:
{
lean_object* v___x_110_; lean_object* v_env_111_; lean_object* v___x_112_; lean_object* v_mctx_113_; lean_object* v_lctx_114_; lean_object* v_options_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_110_ = lean_st_ref_get(v___y_108_);
v_env_111_ = lean_ctor_get(v___x_110_, 0);
lean_inc_ref(v_env_111_);
lean_dec(v___x_110_);
v___x_112_ = lean_st_ref_get(v___y_106_);
v_mctx_113_ = lean_ctor_get(v___x_112_, 0);
lean_inc_ref(v_mctx_113_);
lean_dec(v___x_112_);
v_lctx_114_ = lean_ctor_get(v___y_105_, 2);
v_options_115_ = lean_ctor_get(v___y_107_, 2);
lean_inc_ref(v_options_115_);
lean_inc_ref(v_lctx_114_);
v___x_116_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_116_, 0, v_env_111_);
lean_ctor_set(v___x_116_, 1, v_mctx_113_);
lean_ctor_set(v___x_116_, 2, v_lctx_114_);
lean_ctor_set(v___x_116_, 3, v_options_115_);
v___x_117_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v_msgData_104_);
v___x_118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1_spec__2___boxed(lean_object* v_msgData_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1_spec__2(v_msgData_119_, v___y_120_, v___y_121_, v___y_122_, v___y_123_);
lean_dec(v___y_123_);
lean_dec_ref(v___y_122_);
lean_dec(v___y_121_);
lean_dec_ref(v___y_120_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg(lean_object* v_msg_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_){
_start:
{
lean_object* v_ref_132_; lean_object* v___x_133_; lean_object* v_a_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_142_; 
v_ref_132_ = lean_ctor_get(v___y_129_, 5);
v___x_133_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1_spec__2(v_msg_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_);
v_a_134_ = lean_ctor_get(v___x_133_, 0);
v_isSharedCheck_142_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_142_ == 0)
{
v___x_136_ = v___x_133_;
v_isShared_137_ = v_isSharedCheck_142_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_a_134_);
lean_dec(v___x_133_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_142_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v___x_138_; lean_object* v___x_140_; 
lean_inc(v_ref_132_);
v___x_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_138_, 0, v_ref_132_);
lean_ctor_set(v___x_138_, 1, v_a_134_);
if (v_isShared_137_ == 0)
{
lean_ctor_set_tag(v___x_136_, 1);
lean_ctor_set(v___x_136_, 0, v___x_138_);
v___x_140_ = v___x_136_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v___x_138_);
v___x_140_ = v_reuseFailAlloc_141_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
return v___x_140_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg___boxed(lean_object* v_msg_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg(v_msg_143_, v___y_144_, v___y_145_, v___y_146_, v___y_147_);
lean_dec(v___y_147_);
lean_dec_ref(v___y_146_);
lean_dec(v___y_145_);
lean_dec_ref(v___y_144_);
return v_res_149_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_151_ = ((lean_object*)(lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__0));
v___x_152_ = l_Lean_stringToMessageData(v___x_151_);
return v___x_152_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_154_ = ((lean_object*)(lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__2));
v___x_155_ = l_Lean_stringToMessageData(v___x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg(lean_object* v_x_156_, lean_object* v_g_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_){
_start:
{
lean_object* v___x_166_; lean_object* v_elimGoal_167_; lean_object* v___x_168_; lean_object* v_normalizationState_169_; 
v___x_166_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_167_ = lean_ctor_get(v___x_166_, 1);
lean_inc_ref(v_elimGoal_167_);
v___x_168_ = lean_apply_1(v_elimGoal_167_, v_g_157_);
v_normalizationState_169_ = lean_ctor_get(v___x_168_, 6);
lean_inc(v_normalizationState_169_);
if (lean_obj_tag(v_normalizationState_169_) == 1)
{
lean_object* v_postGoal_170_; lean_object* v_postState_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
lean_dec_ref(v___x_168_);
v_postGoal_170_ = lean_ctor_get(v_normalizationState_169_, 0);
lean_inc(v_postGoal_170_);
v_postState_171_ = lean_ctor_get(v_normalizationState_169_, 1);
lean_inc_ref(v_postState_171_);
lean_dec_ref_known(v_normalizationState_169_, 3);
v___x_172_ = lean_apply_1(v_x_156_, v_postGoal_170_);
v___x_173_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___redArg(v_postState_171_, v___x_172_, v___y_158_, v___y_159_, v___y_160_, v___y_161_, v___y_162_, v___y_163_, v___y_164_);
lean_dec_ref(v_postState_171_);
return v___x_173_;
}
else
{
lean_object* v_id_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
lean_dec(v_normalizationState_169_);
lean_dec_ref(v_x_156_);
v_id_174_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_id_174_);
lean_dec_ref(v___x_168_);
v___x_175_ = lean_obj_once(&lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__1, &lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__1_once, _init_lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__1);
v___x_176_ = l_Nat_reprFast(v_id_174_);
v___x_177_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
v___x_178_ = l_Lean_MessageData_ofFormat(v___x_177_);
v___x_179_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_175_);
lean_ctor_set(v___x_179_, 1, v___x_178_);
v___x_180_ = lean_obj_once(&lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__3, &lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__3_once, _init_lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___closed__3);
v___x_181_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_179_);
lean_ctor_set(v___x_181_, 1, v___x_180_);
v___x_182_ = lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg(v___x_181_, v___y_161_, v___y_162_, v___y_163_, v___y_164_);
return v___x_182_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg___boxed(lean_object* v_x_183_, lean_object* v_g_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg(v_x_183_, v_g_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_, v___y_191_);
lean_dec(v___y_191_);
lean_dec_ref(v___y_190_);
lean_dec(v___y_189_);
lean_dec_ref(v___y_188_);
lean_dec(v___y_187_);
lean_dec(v___y_186_);
lean_dec_ref(v___y_185_);
return v_res_193_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__0(void){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
v___x_194_ = lean_box(0);
v___x_195_ = lean_unsigned_to_nat(16u);
v___x_196_ = lean_mk_array(v___x_195_, v___x_194_);
return v___x_196_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__1(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_197_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__0, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__0_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__0);
v___x_198_ = lean_unsigned_to_nat(0u);
v___x_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_198_);
lean_ctor_set(v___x_199_, 1, v___x_197_);
return v___x_199_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__4(void){
_start:
{
lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_203_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__3));
v___x_204_ = l_Lean_stringToMessageData(v___x_203_);
return v___x_204_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__9(void){
_start:
{
lean_object* v___x_212_; lean_object* v___x_213_; 
v___x_212_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__8));
v___x_213_ = l_Lean_MessageData_ofName(v___x_212_);
return v___x_213_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__10(void){
_start:
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_214_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__9, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__9_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__9);
v___x_215_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__4, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__4_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__4);
v___x_216_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_216_, 0, v___x_215_);
lean_ctor_set(v___x_216_, 1, v___x_214_);
return v___x_216_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__12(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_218_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__11));
v___x_219_ = l_Lean_stringToMessageData(v___x_218_);
return v___x_219_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__13(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_220_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__12, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__12_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__12);
v___x_221_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__10, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__10_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__10);
v___x_222_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
lean_ctor_set(v___x_222_, 1, v___x_220_);
return v___x_222_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__15(void){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; 
v___x_224_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__14));
v___x_225_ = l_Lean_stringToMessageData(v___x_224_);
return v___x_225_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__20(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_230_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__19));
v___x_231_ = l_Lean_stringToMessageData(v___x_230_);
return v___x_231_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__21(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_232_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__20, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__20_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__20);
v___x_233_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__10, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__10_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__10);
v___x_234_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v___x_232_);
return v___x_234_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__23(void){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_236_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__22));
v___x_237_ = l_Lean_stringToMessageData(v___x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_updateForwardState(uint8_t v_phase_238_, lean_object* v_gref_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_, lean_object* v_a_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___f_250_; lean_object* v___y_252_; lean_object* v___y_253_; lean_object* v___y_254_; lean_object* v___y_255_; lean_object* v___y_256_; lean_object* v___y_257_; lean_object* v___y_258_; lean_object* v___y_342_; lean_object* v___y_343_; lean_object* v___y_344_; lean_object* v___y_345_; lean_object* v___y_346_; lean_object* v___y_347_; lean_object* v___y_348_; lean_object* v___y_349_; lean_object* v___y_350_; lean_object* v___y_356_; lean_object* v___y_357_; lean_object* v___y_358_; lean_object* v___y_359_; lean_object* v___y_360_; lean_object* v___y_361_; lean_object* v___y_362_; uint8_t v___x_378_; uint8_t v___x_379_; 
v___x_248_ = lean_st_ref_get(v_gref_239_);
v___x_249_ = lean_box(v_phase_238_);
lean_inc(v___x_248_);
v___f_250_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_updateForwardState___lam__0___boxed), 11, 2);
lean_closure_set(v___f_250_, 0, v___x_248_);
lean_closure_set(v___f_250_, 1, v___x_249_);
v___x_378_ = 0;
v___x_379_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_238_, v___x_378_);
if (v___x_379_ == 0)
{
v___y_356_ = v_a_240_;
v___y_357_ = v_a_241_;
v___y_358_ = v_a_242_;
v___y_359_ = v_a_243_;
v___y_360_ = v_a_244_;
v___y_361_ = v_a_245_;
v___y_362_ = v_a_246_;
goto v___jp_355_;
}
else
{
lean_object* v___x_380_; lean_object* v_elimGoal_381_; lean_object* v___x_382_; lean_object* v_id_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
lean_dec_ref(v___f_250_);
v___x_380_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_381_ = lean_ctor_get(v___x_380_, 1);
lean_inc_ref(v_elimGoal_381_);
v___x_382_ = lean_apply_1(v_elimGoal_381_, v___x_248_);
v_id_383_ = lean_ctor_get(v___x_382_, 0);
lean_inc(v_id_383_);
lean_dec_ref(v___x_382_);
v___x_384_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__21, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__21_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__21);
v___x_385_ = l_Nat_reprFast(v_id_383_);
v___x_386_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
v___x_387_ = l_Lean_MessageData_ofFormat(v___x_386_);
v___x_388_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_388_, 0, v___x_384_);
lean_ctor_set(v___x_388_, 1, v___x_387_);
v___x_389_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__23, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__23_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__23);
v___x_390_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_390_, 0, v___x_388_);
lean_ctor_set(v___x_390_, 1, v___x_389_);
v___x_391_ = lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg(v___x_390_, v_a_243_, v_a_244_, v_a_245_, v_a_246_);
return v___x_391_;
}
v___jp_251_:
{
lean_object* v___x_259_; 
lean_inc(v___x_248_);
v___x_259_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg(v___f_250_, v___x_248_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_, v___y_258_);
if (lean_obj_tag(v___x_259_) == 0)
{
lean_object* v_a_260_; lean_object* v___x_262_; uint8_t v_isShared_263_; uint8_t v_isSharedCheck_332_; 
v_a_260_ = lean_ctor_get(v___x_259_, 0);
v_isSharedCheck_332_ = !lean_is_exclusive(v___x_259_);
if (v_isSharedCheck_332_ == 0)
{
v___x_262_ = v___x_259_;
v_isShared_263_ = v_isSharedCheck_332_;
goto v_resetjp_261_;
}
else
{
lean_inc(v_a_260_);
lean_dec(v___x_259_);
v___x_262_ = lean_box(0);
v_isShared_263_ = v_isSharedCheck_332_;
goto v_resetjp_261_;
}
v_resetjp_261_:
{
lean_object* v_fst_264_; lean_object* v_snd_265_; lean_object* v___x_266_; lean_object* v_introGoal_267_; lean_object* v_elimGoal_268_; lean_object* v___x_269_; lean_object* v_id_270_; lean_object* v_parent_271_; lean_object* v_children_272_; lean_object* v_origin_273_; lean_object* v_depth_274_; uint8_t v_state_275_; uint8_t v_isIrrelevant_276_; uint8_t v_isForcedUnprovable_277_; lean_object* v_preNormGoal_278_; lean_object* v_normalizationState_279_; lean_object* v_mvars_280_; lean_object* v_forwardRuleMatches_281_; double v_successProbability_282_; lean_object* v_addedInIteration_283_; lean_object* v_lastExpandedInIteration_284_; uint8_t v_unsafeRulesSelected_285_; lean_object* v_unsafeQueue_286_; lean_object* v_failedRapps_287_; lean_object* v___x_289_; uint8_t v_isShared_290_; uint8_t v_isSharedCheck_330_; 
v_fst_264_ = lean_ctor_get(v_a_260_, 0);
lean_inc(v_fst_264_);
v_snd_265_ = lean_ctor_get(v_a_260_, 1);
lean_inc(v_snd_265_);
lean_dec(v_a_260_);
v___x_266_ = lp_aesop_Aesop_treeImpl;
v_introGoal_267_ = lean_ctor_get(v___x_266_, 0);
v_elimGoal_268_ = lean_ctor_get(v___x_266_, 1);
lean_inc_ref(v_elimGoal_268_);
v___x_269_ = lean_apply_1(v_elimGoal_268_, v___x_248_);
v_id_270_ = lean_ctor_get(v___x_269_, 0);
v_parent_271_ = lean_ctor_get(v___x_269_, 1);
v_children_272_ = lean_ctor_get(v___x_269_, 2);
v_origin_273_ = lean_ctor_get(v___x_269_, 3);
v_depth_274_ = lean_ctor_get(v___x_269_, 4);
v_state_275_ = lean_ctor_get_uint8(v___x_269_, sizeof(void*)*14 + 8);
v_isIrrelevant_276_ = lean_ctor_get_uint8(v___x_269_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_277_ = lean_ctor_get_uint8(v___x_269_, sizeof(void*)*14 + 10);
v_preNormGoal_278_ = lean_ctor_get(v___x_269_, 5);
v_normalizationState_279_ = lean_ctor_get(v___x_269_, 6);
v_mvars_280_ = lean_ctor_get(v___x_269_, 7);
v_forwardRuleMatches_281_ = lean_ctor_get(v___x_269_, 9);
v_successProbability_282_ = lean_ctor_get_float(v___x_269_, sizeof(void*)*14);
v_addedInIteration_283_ = lean_ctor_get(v___x_269_, 10);
v_lastExpandedInIteration_284_ = lean_ctor_get(v___x_269_, 11);
v_unsafeRulesSelected_285_ = lean_ctor_get_uint8(v___x_269_, sizeof(void*)*14 + 11);
v_unsafeQueue_286_ = lean_ctor_get(v___x_269_, 12);
v_failedRapps_287_ = lean_ctor_get(v___x_269_, 13);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_269_);
if (v_isSharedCheck_330_ == 0)
{
lean_object* v_unused_331_; 
v_unused_331_ = lean_ctor_get(v___x_269_, 8);
lean_dec(v_unused_331_);
v___x_289_ = v___x_269_;
v_isShared_290_ = v_isSharedCheck_330_;
goto v_resetjp_288_;
}
else
{
lean_inc(v_failedRapps_287_);
lean_inc(v_unsafeQueue_286_);
lean_inc(v_lastExpandedInIteration_284_);
lean_inc(v_addedInIteration_283_);
lean_inc(v_forwardRuleMatches_281_);
lean_inc(v_mvars_280_);
lean_inc(v_normalizationState_279_);
lean_inc(v_preNormGoal_278_);
lean_inc(v_depth_274_);
lean_inc(v_origin_273_);
lean_inc(v_children_272_);
lean_inc(v_parent_271_);
lean_inc(v_id_270_);
lean_dec(v___x_269_);
v___x_289_ = lean_box(0);
v_isShared_290_ = v_isSharedCheck_330_;
goto v_resetjp_288_;
}
v_resetjp_288_:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_295_; 
v___x_291_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__1, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__1_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__1);
v___x_292_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__2));
lean_inc_ref(v_forwardRuleMatches_281_);
v___x_293_ = lp_aesop_Aesop_ForwardRuleMatches_update(v_snd_265_, v___x_291_, v___x_292_, v_forwardRuleMatches_281_);
lean_dec(v_snd_265_);
if (v_isShared_290_ == 0)
{
lean_ctor_set(v___x_289_, 8, v_fst_264_);
v___x_295_ = v___x_289_;
goto v_reusejp_294_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_id_270_);
lean_ctor_set(v_reuseFailAlloc_329_, 1, v_parent_271_);
lean_ctor_set(v_reuseFailAlloc_329_, 2, v_children_272_);
lean_ctor_set(v_reuseFailAlloc_329_, 3, v_origin_273_);
lean_ctor_set(v_reuseFailAlloc_329_, 4, v_depth_274_);
lean_ctor_set(v_reuseFailAlloc_329_, 5, v_preNormGoal_278_);
lean_ctor_set(v_reuseFailAlloc_329_, 6, v_normalizationState_279_);
lean_ctor_set(v_reuseFailAlloc_329_, 7, v_mvars_280_);
lean_ctor_set(v_reuseFailAlloc_329_, 8, v_fst_264_);
lean_ctor_set(v_reuseFailAlloc_329_, 9, v_forwardRuleMatches_281_);
lean_ctor_set(v_reuseFailAlloc_329_, 10, v_addedInIteration_283_);
lean_ctor_set(v_reuseFailAlloc_329_, 11, v_lastExpandedInIteration_284_);
lean_ctor_set(v_reuseFailAlloc_329_, 12, v_unsafeQueue_286_);
lean_ctor_set(v_reuseFailAlloc_329_, 13, v_failedRapps_287_);
lean_ctor_set_uint8(v_reuseFailAlloc_329_, sizeof(void*)*14 + 8, v_state_275_);
lean_ctor_set_uint8(v_reuseFailAlloc_329_, sizeof(void*)*14 + 9, v_isIrrelevant_276_);
lean_ctor_set_uint8(v_reuseFailAlloc_329_, sizeof(void*)*14 + 10, v_isForcedUnprovable_277_);
lean_ctor_set_float(v_reuseFailAlloc_329_, sizeof(void*)*14, v_successProbability_282_);
lean_ctor_set_uint8(v_reuseFailAlloc_329_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_285_);
v___x_295_ = v_reuseFailAlloc_329_;
goto v_reusejp_294_;
}
v_reusejp_294_:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v_id_298_; lean_object* v_parent_299_; lean_object* v_children_300_; lean_object* v_origin_301_; lean_object* v_depth_302_; uint8_t v_state_303_; uint8_t v_isIrrelevant_304_; uint8_t v_isForcedUnprovable_305_; lean_object* v_preNormGoal_306_; lean_object* v_normalizationState_307_; lean_object* v_mvars_308_; lean_object* v_forwardState_309_; double v_successProbability_310_; lean_object* v_addedInIteration_311_; lean_object* v_lastExpandedInIteration_312_; uint8_t v_unsafeRulesSelected_313_; lean_object* v_unsafeQueue_314_; lean_object* v_failedRapps_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_327_; 
lean_inc(v_introGoal_267_);
v___x_296_ = lean_apply_1(v_introGoal_267_, v___x_295_);
lean_inc_ref(v_elimGoal_268_);
v___x_297_ = lean_apply_1(v_elimGoal_268_, v___x_296_);
v_id_298_ = lean_ctor_get(v___x_297_, 0);
v_parent_299_ = lean_ctor_get(v___x_297_, 1);
v_children_300_ = lean_ctor_get(v___x_297_, 2);
v_origin_301_ = lean_ctor_get(v___x_297_, 3);
v_depth_302_ = lean_ctor_get(v___x_297_, 4);
v_state_303_ = lean_ctor_get_uint8(v___x_297_, sizeof(void*)*14 + 8);
v_isIrrelevant_304_ = lean_ctor_get_uint8(v___x_297_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_305_ = lean_ctor_get_uint8(v___x_297_, sizeof(void*)*14 + 10);
v_preNormGoal_306_ = lean_ctor_get(v___x_297_, 5);
v_normalizationState_307_ = lean_ctor_get(v___x_297_, 6);
v_mvars_308_ = lean_ctor_get(v___x_297_, 7);
v_forwardState_309_ = lean_ctor_get(v___x_297_, 8);
v_successProbability_310_ = lean_ctor_get_float(v___x_297_, sizeof(void*)*14);
v_addedInIteration_311_ = lean_ctor_get(v___x_297_, 10);
v_lastExpandedInIteration_312_ = lean_ctor_get(v___x_297_, 11);
v_unsafeRulesSelected_313_ = lean_ctor_get_uint8(v___x_297_, sizeof(void*)*14 + 11);
v_unsafeQueue_314_ = lean_ctor_get(v___x_297_, 12);
v_failedRapps_315_ = lean_ctor_get(v___x_297_, 13);
v_isSharedCheck_327_ = !lean_is_exclusive(v___x_297_);
if (v_isSharedCheck_327_ == 0)
{
lean_object* v_unused_328_; 
v_unused_328_ = lean_ctor_get(v___x_297_, 9);
lean_dec(v_unused_328_);
v___x_317_ = v___x_297_;
v_isShared_318_ = v_isSharedCheck_327_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_failedRapps_315_);
lean_inc(v_unsafeQueue_314_);
lean_inc(v_lastExpandedInIteration_312_);
lean_inc(v_addedInIteration_311_);
lean_inc(v_forwardState_309_);
lean_inc(v_mvars_308_);
lean_inc(v_normalizationState_307_);
lean_inc(v_preNormGoal_306_);
lean_inc(v_depth_302_);
lean_inc(v_origin_301_);
lean_inc(v_children_300_);
lean_inc(v_parent_299_);
lean_inc(v_id_298_);
lean_dec(v___x_297_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_327_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v___x_320_; 
if (v_isShared_318_ == 0)
{
lean_ctor_set(v___x_317_, 9, v___x_293_);
v___x_320_ = v___x_317_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v_id_298_);
lean_ctor_set(v_reuseFailAlloc_326_, 1, v_parent_299_);
lean_ctor_set(v_reuseFailAlloc_326_, 2, v_children_300_);
lean_ctor_set(v_reuseFailAlloc_326_, 3, v_origin_301_);
lean_ctor_set(v_reuseFailAlloc_326_, 4, v_depth_302_);
lean_ctor_set(v_reuseFailAlloc_326_, 5, v_preNormGoal_306_);
lean_ctor_set(v_reuseFailAlloc_326_, 6, v_normalizationState_307_);
lean_ctor_set(v_reuseFailAlloc_326_, 7, v_mvars_308_);
lean_ctor_set(v_reuseFailAlloc_326_, 8, v_forwardState_309_);
lean_ctor_set(v_reuseFailAlloc_326_, 9, v___x_293_);
lean_ctor_set(v_reuseFailAlloc_326_, 10, v_addedInIteration_311_);
lean_ctor_set(v_reuseFailAlloc_326_, 11, v_lastExpandedInIteration_312_);
lean_ctor_set(v_reuseFailAlloc_326_, 12, v_unsafeQueue_314_);
lean_ctor_set(v_reuseFailAlloc_326_, 13, v_failedRapps_315_);
lean_ctor_set_uint8(v_reuseFailAlloc_326_, sizeof(void*)*14 + 8, v_state_303_);
lean_ctor_set_uint8(v_reuseFailAlloc_326_, sizeof(void*)*14 + 9, v_isIrrelevant_304_);
lean_ctor_set_uint8(v_reuseFailAlloc_326_, sizeof(void*)*14 + 10, v_isForcedUnprovable_305_);
lean_ctor_set_float(v_reuseFailAlloc_326_, sizeof(void*)*14, v_successProbability_310_);
lean_ctor_set_uint8(v_reuseFailAlloc_326_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_313_);
v___x_320_ = v_reuseFailAlloc_326_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_324_; 
lean_inc(v_introGoal_267_);
v___x_321_ = lean_apply_1(v_introGoal_267_, v___x_320_);
v___x_322_ = lean_st_ref_set(v_gref_239_, v___x_321_);
if (v_isShared_263_ == 0)
{
lean_ctor_set(v___x_262_, 0, v___x_322_);
v___x_324_ = v___x_262_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v___x_322_);
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
}
else
{
lean_object* v_a_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_340_; 
lean_dec(v___x_248_);
v_a_333_ = lean_ctor_get(v___x_259_, 0);
v_isSharedCheck_340_ = !lean_is_exclusive(v___x_259_);
if (v_isSharedCheck_340_ == 0)
{
v___x_335_ = v___x_259_;
v_isShared_336_ = v_isSharedCheck_340_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_a_333_);
lean_dec(v___x_259_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_340_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_338_; 
if (v_isShared_336_ == 0)
{
v___x_338_ = v___x_335_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v_a_333_);
v___x_338_ = v_reuseFailAlloc_339_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
return v___x_338_;
}
}
}
}
v___jp_341_:
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; 
lean_inc_ref(v___y_350_);
v___x_351_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_351_, 0, v___y_350_);
v___x_352_ = l_Lean_MessageData_ofFormat(v___x_351_);
v___x_353_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_353_, 0, v___y_343_);
lean_ctor_set(v___x_353_, 1, v___x_352_);
v___x_354_ = lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg(v___x_353_, v___y_349_, v___y_348_, v___y_347_, v___y_342_);
return v___x_354_;
}
v___jp_355_:
{
uint8_t v___x_363_; 
lean_inc(v___x_248_);
v___x_363_ = lp_aesop_Aesop_Goal_isNormal(v___x_248_);
if (v___x_363_ == 0)
{
lean_object* v___x_364_; lean_object* v_elimGoal_365_; lean_object* v___x_366_; lean_object* v_id_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; 
lean_dec_ref(v___f_250_);
v___x_364_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_365_ = lean_ctor_get(v___x_364_, 1);
lean_inc_ref(v_elimGoal_365_);
v___x_366_ = lean_apply_1(v_elimGoal_365_, v___x_248_);
v_id_367_ = lean_ctor_get(v___x_366_, 0);
lean_inc(v_id_367_);
lean_dec_ref(v___x_366_);
v___x_368_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__13, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__13_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__13);
v___x_369_ = l_Nat_reprFast(v_id_367_);
v___x_370_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_370_, 0, v___x_369_);
v___x_371_ = l_Lean_MessageData_ofFormat(v___x_370_);
v___x_372_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_368_);
lean_ctor_set(v___x_372_, 1, v___x_371_);
v___x_373_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_updateForwardState___closed__15, &lp_aesop_Aesop_GoalRef_updateForwardState___closed__15_once, _init_lp_aesop_Aesop_GoalRef_updateForwardState___closed__15);
v___x_374_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_374_, 0, v___x_372_);
lean_ctor_set(v___x_374_, 1, v___x_373_);
switch(v_phase_238_)
{
case 0:
{
lean_object* v___x_375_; 
v___x_375_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__16));
v___y_342_ = v___y_362_;
v___y_343_ = v___x_374_;
v___y_344_ = v___y_358_;
v___y_345_ = v___y_356_;
v___y_346_ = v___y_357_;
v___y_347_ = v___y_361_;
v___y_348_ = v___y_360_;
v___y_349_ = v___y_359_;
v___y_350_ = v___x_375_;
goto v___jp_341_;
}
case 1:
{
lean_object* v___x_376_; 
v___x_376_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__17));
v___y_342_ = v___y_362_;
v___y_343_ = v___x_374_;
v___y_344_ = v___y_358_;
v___y_345_ = v___y_356_;
v___y_346_ = v___y_357_;
v___y_347_ = v___y_361_;
v___y_348_ = v___y_360_;
v___y_349_ = v___y_359_;
v___y_350_ = v___x_376_;
goto v___jp_341_;
}
default: 
{
lean_object* v___x_377_; 
v___x_377_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_updateForwardState___closed__18));
v___y_342_ = v___y_362_;
v___y_343_ = v___x_374_;
v___y_344_ = v___y_358_;
v___y_345_ = v___y_356_;
v___y_346_ = v___y_357_;
v___y_347_ = v___y_361_;
v___y_348_ = v___y_360_;
v___y_349_ = v___y_359_;
v___y_350_ = v___x_377_;
goto v___jp_341_;
}
}
}
else
{
v___y_252_ = v___y_356_;
v___y_253_ = v___y_357_;
v___y_254_ = v___y_358_;
v___y_255_ = v___y_359_;
v___y_256_ = v___y_360_;
v___y_257_ = v___y_361_;
v___y_258_ = v___y_362_;
goto v___jp_251_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_updateForwardState___boxed(lean_object* v_phase_392_, lean_object* v_gref_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_, lean_object* v_a_397_, lean_object* v_a_398_, lean_object* v_a_399_, lean_object* v_a_400_, lean_object* v_a_401_){
_start:
{
uint8_t v_phase_boxed_402_; lean_object* v_res_403_; 
v_phase_boxed_402_ = lean_unbox(v_phase_392_);
v_res_403_ = lp_aesop_Aesop_GoalRef_updateForwardState(v_phase_boxed_402_, v_gref_393_, v_a_394_, v_a_395_, v_a_396_, v_a_397_, v_a_398_, v_a_399_, v_a_400_);
lean_dec(v_a_400_);
lean_dec_ref(v_a_399_);
lean_dec(v_a_398_);
lean_dec_ref(v_a_397_);
lean_dec(v_a_396_);
lean_dec(v_a_395_);
lean_dec_ref(v_a_394_);
lean_dec(v_gref_393_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0(lean_object* v_00_u03b1_404_, lean_object* v_s_405_, lean_object* v_x_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_){
_start:
{
lean_object* v___x_415_; 
v___x_415_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___redArg(v_s_405_, v_x_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_, v___y_411_, v___y_412_, v___y_413_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0___boxed(lean_object* v_00_u03b1_416_, lean_object* v_s_417_, lean_object* v_x_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0_spec__0(v_00_u03b1_416_, v_s_417_, v_x_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_);
lean_dec(v___y_425_);
lean_dec_ref(v___y_424_);
lean_dec(v___y_423_);
lean_dec_ref(v___y_422_);
lean_dec(v___y_421_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
lean_dec_ref(v_s_417_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0(lean_object* v_00_u03b1_428_, lean_object* v_x_429_, lean_object* v_g_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___redArg(v_x_429_, v_g_430_, v___y_431_, v___y_432_, v___y_433_, v___y_434_, v___y_435_, v___y_436_, v___y_437_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0___boxed(lean_object* v_00_u03b1_440_, lean_object* v_x_441_, lean_object* v_g_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___at___00Aesop_GoalRef_updateForwardState_spec__0(v_00_u03b1_440_, v_x_441_, v_g_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
lean_dec(v___y_445_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1(lean_object* v_00_u03b1_452_, lean_object* v_msg_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___redArg(v_msg_453_, v___y_457_, v___y_458_, v___y_459_, v___y_460_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1___boxed(lean_object* v_00_u03b1_463_, lean_object* v_msg_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_aesop_Lean_throwError___at___00Aesop_GoalRef_updateForwardState_spec__1(v_00_u03b1_463_, v_msg_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_, v___y_471_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
lean_dec(v___y_467_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
return v_res_473_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Forward_State_UpdateGoal(uint8_t builtin) {
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
res = runtime_initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Forward_State_UpdateGoal(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_State_UpdateGoal(uint8_t builtin) {
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
res = initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State_UpdateGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Forward_State_UpdateGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Forward_State_UpdateGoal(builtin);
}
#ifdef __cplusplus
}
#endif
