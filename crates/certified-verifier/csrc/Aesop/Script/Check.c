// Lean compiler output
// Module: Aesop.Script.Check
// Imports: public import Init public meta import Init public import Aesop.Script.UScript import Aesop.Check
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
extern lean_object* lp_aesop_Aesop_Check_script;
uint8_t lp_aesop_Aesop_Check_get(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_Tactic_setGoals___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getUnsolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Check_name(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
extern lean_object* lp_aesop_Aesop_Check_script_steps;
lean_object* lp_aesop_Aesop_Script_UScript_validate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__0;
static lean_once_cell_t lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__1;
static const lean_string_object lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__3;
static lean_once_cell_t lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "script executed successfully but did not solve the main goal"};
static const lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__0 = (const lean_object*)&lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__1;
static lean_once_cell_t lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__2;
static const lean_string_object lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = ": error while executing generated script:"};
static const lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__3 = (const lean_object*)&lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__4;
static lean_once_cell_t lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___redArg(lean_object* v_opt_1_, lean_object* v___y_2_){
_start:
{
lean_object* v_options_4_; uint8_t v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v_options_4_ = lean_ctor_get(v___y_2_, 2);
v___x_5_ = lp_aesop_Aesop_Check_get(v_options_4_, v_opt_1_);
v___x_6_ = lean_box(v___x_5_);
v___x_7_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_7_, 0, v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___redArg___boxed(lean_object* v_opt_8_, lean_object* v___y_9_, lean_object* v___y_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___redArg(v_opt_8_, v___y_9_);
lean_dec_ref(v___y_9_);
lean_dec_ref(v_opt_8_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0(lean_object* v_opt_12_, lean_object* v___y_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___redArg(v_opt_12_, v___y_15_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___boxed(lean_object* v_opt_19_, lean_object* v___y_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0(v_opt_19_, v___y_20_, v___y_21_, v___y_22_, v___y_23_);
lean_dec(v___y_23_);
lean_dec_ref(v___y_22_);
lean_dec(v___y_21_);
lean_dec_ref(v___y_20_);
lean_dec_ref(v_opt_19_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1_spec__1(lean_object* v_msgData_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_){
_start:
{
lean_object* v___x_32_; lean_object* v_env_33_; lean_object* v___x_34_; lean_object* v_mctx_35_; lean_object* v_lctx_36_; lean_object* v_options_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_32_ = lean_st_ref_get(v___y_30_);
v_env_33_ = lean_ctor_get(v___x_32_, 0);
lean_inc_ref(v_env_33_);
lean_dec(v___x_32_);
v___x_34_ = lean_st_ref_get(v___y_28_);
v_mctx_35_ = lean_ctor_get(v___x_34_, 0);
lean_inc_ref(v_mctx_35_);
lean_dec(v___x_34_);
v_lctx_36_ = lean_ctor_get(v___y_27_, 2);
v_options_37_ = lean_ctor_get(v___y_29_, 2);
lean_inc_ref(v_options_37_);
lean_inc_ref(v_lctx_36_);
v___x_38_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_38_, 0, v_env_33_);
lean_ctor_set(v___x_38_, 1, v_mctx_35_);
lean_ctor_set(v___x_38_, 2, v_lctx_36_);
lean_ctor_set(v___x_38_, 3, v_options_37_);
v___x_39_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_39_, 0, v___x_38_);
lean_ctor_set(v___x_39_, 1, v_msgData_26_);
v___x_40_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_40_, 0, v___x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1_spec__1___boxed(lean_object* v_msgData_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1_spec__1(v_msgData_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___redArg(lean_object* v_msg_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_){
_start:
{
lean_object* v_ref_54_; lean_object* v___x_55_; lean_object* v_a_56_; lean_object* v___x_58_; uint8_t v_isShared_59_; uint8_t v_isSharedCheck_64_; 
v_ref_54_ = lean_ctor_get(v___y_51_, 5);
v___x_55_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1_spec__1(v_msg_48_, v___y_49_, v___y_50_, v___y_51_, v___y_52_);
v_a_56_ = lean_ctor_get(v___x_55_, 0);
v_isSharedCheck_64_ = !lean_is_exclusive(v___x_55_);
if (v_isSharedCheck_64_ == 0)
{
v___x_58_ = v___x_55_;
v_isShared_59_ = v_isSharedCheck_64_;
goto v_resetjp_57_;
}
else
{
lean_inc(v_a_56_);
lean_dec(v___x_55_);
v___x_58_ = lean_box(0);
v_isShared_59_ = v_isSharedCheck_64_;
goto v_resetjp_57_;
}
v_resetjp_57_:
{
lean_object* v___x_60_; lean_object* v___x_62_; 
lean_inc(v_ref_54_);
v___x_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_60_, 0, v_ref_54_);
lean_ctor_set(v___x_60_, 1, v_a_56_);
if (v_isShared_59_ == 0)
{
lean_ctor_set_tag(v___x_58_, 1);
lean_ctor_set(v___x_58_, 0, v___x_60_);
v___x_62_ = v___x_58_;
goto v_reusejp_61_;
}
else
{
lean_object* v_reuseFailAlloc_63_; 
v_reuseFailAlloc_63_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_63_, 0, v___x_60_);
v___x_62_ = v_reuseFailAlloc_63_;
goto v_reusejp_61_;
}
v_reusejp_61_:
{
return v___x_62_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___redArg___boxed(lean_object* v_msg_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___redArg(v_msg_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_);
lean_dec(v___y_69_);
lean_dec_ref(v___y_68_);
lean_dec(v___y_67_);
lean_dec_ref(v___y_66_);
return v_res_71_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__0(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_72_ = lp_aesop_Aesop_Check_script_steps;
v___x_73_ = lp_aesop_Aesop_Check_name(v___x_72_);
return v___x_73_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__1(void){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_74_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__0, &lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__0_once, _init_lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__0);
v___x_75_ = l_Lean_MessageData_ofName(v___x_74_);
return v___x_75_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__3(void){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_77_ = ((lean_object*)(lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__2));
v___x_78_ = l_Lean_stringToMessageData(v___x_77_);
return v___x_78_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__4(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_79_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__3, &lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__3_once, _init_lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__3);
v___x_80_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__1, &lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__1_once, _init_lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__1);
v___x_81_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___x_79_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled(lean_object* v_uscript_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v_a_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_109_; 
v___x_88_ = lp_aesop_Aesop_Check_script_steps;
v___x_89_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___redArg(v___x_88_, v_a_85_);
v_a_90_ = lean_ctor_get(v___x_89_, 0);
v_isSharedCheck_109_ = !lean_is_exclusive(v___x_89_);
if (v_isSharedCheck_109_ == 0)
{
v___x_92_ = v___x_89_;
v_isShared_93_ = v_isSharedCheck_109_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_a_90_);
lean_dec(v___x_89_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_109_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
uint8_t v___x_94_; 
v___x_94_ = lean_unbox(v_a_90_);
lean_dec(v_a_90_);
if (v___x_94_ == 0)
{
lean_object* v___x_95_; lean_object* v___x_97_; 
v___x_95_ = lean_box(0);
if (v_isShared_93_ == 0)
{
lean_ctor_set(v___x_92_, 0, v___x_95_);
v___x_97_ = v___x_92_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v___x_95_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
return v___x_97_;
}
}
else
{
lean_object* v___x_99_; 
lean_del_object(v___x_92_);
v___x_99_ = lp_aesop_Aesop_Script_UScript_validate(v_uscript_82_, v_a_83_, v_a_84_, v_a_85_, v_a_86_);
if (lean_obj_tag(v___x_99_) == 0)
{
return v___x_99_;
}
else
{
lean_object* v_a_100_; uint8_t v___y_102_; uint8_t v___x_107_; 
v_a_100_ = lean_ctor_get(v___x_99_, 0);
lean_inc(v_a_100_);
v___x_107_ = l_Lean_Exception_isInterrupt(v_a_100_);
if (v___x_107_ == 0)
{
uint8_t v___x_108_; 
lean_inc(v_a_100_);
v___x_108_ = l_Lean_Exception_isRuntime(v_a_100_);
v___y_102_ = v___x_108_;
goto v___jp_101_;
}
else
{
v___y_102_ = v___x_107_;
goto v___jp_101_;
}
v___jp_101_:
{
if (v___y_102_ == 0)
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
lean_dec_ref_known(v___x_99_, 1);
v___x_103_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__4, &lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__4_once, _init_lp_aesop_Aesop_Script_UScript_checkIfEnabled___closed__4);
v___x_104_ = l_Lean_Exception_toMessageData(v_a_100_);
v___x_105_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_103_);
lean_ctor_set(v___x_105_, 1, v___x_104_);
v___x_106_ = lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___redArg(v___x_105_, v_a_83_, v_a_84_, v_a_85_, v_a_86_);
return v___x_106_;
}
else
{
lean_dec(v_a_100_);
return v___x_99_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled___boxed(lean_object* v_uscript_110_, lean_object* v_a_111_, lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_aesop_Aesop_Script_UScript_checkIfEnabled(v_uscript_110_, v_a_111_, v_a_112_, v_a_113_, v_a_114_);
lean_dec(v_a_114_);
lean_dec_ref(v_a_113_);
lean_dec(v_a_112_);
lean_dec_ref(v_a_111_);
lean_dec_ref(v_uscript_110_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1(lean_object* v_00_u03b1_117_, lean_object* v_msg_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___redArg(v_msg_118_, v___y_119_, v___y_120_, v___y_121_, v___y_122_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___boxed(lean_object* v_00_u03b1_125_, lean_object* v_msg_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1(v_00_u03b1_125_, v_msg_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_);
lean_dec(v___y_130_);
lean_dec_ref(v___y_129_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___redArg(lean_object* v_x_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = l_Lean_Meta_saveState___redArg(v___y_135_, v___y_137_);
if (lean_obj_tag(v___x_139_) == 0)
{
lean_object* v_a_140_; lean_object* v_r_141_; 
v_a_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc(v_a_140_);
lean_dec_ref_known(v___x_139_, 1);
lean_inc(v___y_137_);
lean_inc_ref(v___y_136_);
lean_inc(v___y_135_);
lean_inc_ref(v___y_134_);
v_r_141_ = lean_apply_5(v_x_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_, lean_box(0));
if (lean_obj_tag(v_r_141_) == 0)
{
lean_object* v_a_142_; lean_object* v___x_143_; 
v_a_142_ = lean_ctor_get(v_r_141_, 0);
lean_inc(v_a_142_);
lean_dec_ref_known(v_r_141_, 1);
v___x_143_ = l_Lean_Meta_SavedState_restore___redArg(v_a_140_, v___y_135_, v___y_137_);
lean_dec(v_a_140_);
if (lean_obj_tag(v___x_143_) == 0)
{
lean_object* v___x_145_; uint8_t v_isShared_146_; uint8_t v_isSharedCheck_150_; 
v_isSharedCheck_150_ = !lean_is_exclusive(v___x_143_);
if (v_isSharedCheck_150_ == 0)
{
lean_object* v_unused_151_; 
v_unused_151_ = lean_ctor_get(v___x_143_, 0);
lean_dec(v_unused_151_);
v___x_145_ = v___x_143_;
v_isShared_146_ = v_isSharedCheck_150_;
goto v_resetjp_144_;
}
else
{
lean_dec(v___x_143_);
v___x_145_ = lean_box(0);
v_isShared_146_ = v_isSharedCheck_150_;
goto v_resetjp_144_;
}
v_resetjp_144_:
{
lean_object* v___x_148_; 
if (v_isShared_146_ == 0)
{
lean_ctor_set(v___x_145_, 0, v_a_142_);
v___x_148_ = v___x_145_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_149_; 
v_reuseFailAlloc_149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_149_, 0, v_a_142_);
v___x_148_ = v_reuseFailAlloc_149_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
return v___x_148_;
}
}
}
else
{
lean_object* v_a_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_159_; 
lean_dec(v_a_142_);
v_a_152_ = lean_ctor_get(v___x_143_, 0);
v_isSharedCheck_159_ = !lean_is_exclusive(v___x_143_);
if (v_isSharedCheck_159_ == 0)
{
v___x_154_ = v___x_143_;
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_a_152_);
lean_dec(v___x_143_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_157_; 
if (v_isShared_155_ == 0)
{
v___x_157_ = v___x_154_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v_a_152_);
v___x_157_ = v_reuseFailAlloc_158_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
return v___x_157_;
}
}
}
}
else
{
lean_object* v_a_160_; lean_object* v___x_161_; 
v_a_160_ = lean_ctor_get(v_r_141_, 0);
lean_inc(v_a_160_);
lean_dec_ref_known(v_r_141_, 1);
v___x_161_ = l_Lean_Meta_SavedState_restore___redArg(v_a_140_, v___y_135_, v___y_137_);
lean_dec(v_a_140_);
if (lean_obj_tag(v___x_161_) == 0)
{
lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_168_; 
v_isSharedCheck_168_ = !lean_is_exclusive(v___x_161_);
if (v_isSharedCheck_168_ == 0)
{
lean_object* v_unused_169_; 
v_unused_169_ = lean_ctor_get(v___x_161_, 0);
lean_dec(v_unused_169_);
v___x_163_ = v___x_161_;
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
else
{
lean_dec(v___x_161_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
lean_object* v___x_166_; 
if (v_isShared_164_ == 0)
{
lean_ctor_set_tag(v___x_163_, 1);
lean_ctor_set(v___x_163_, 0, v_a_160_);
v___x_166_ = v___x_163_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v_a_160_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
else
{
lean_object* v_a_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_177_; 
lean_dec(v_a_160_);
v_a_170_ = lean_ctor_get(v___x_161_, 0);
v_isSharedCheck_177_ = !lean_is_exclusive(v___x_161_);
if (v_isSharedCheck_177_ == 0)
{
v___x_172_ = v___x_161_;
v_isShared_173_ = v_isSharedCheck_177_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_a_170_);
lean_dec(v___x_161_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_177_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v___x_175_; 
if (v_isShared_173_ == 0)
{
v___x_175_ = v___x_172_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v_a_170_);
v___x_175_ = v_reuseFailAlloc_176_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
return v___x_175_;
}
}
}
}
}
else
{
lean_object* v_a_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_185_; 
lean_dec_ref(v_x_133_);
v_a_178_ = lean_ctor_get(v___x_139_, 0);
v_isSharedCheck_185_ = !lean_is_exclusive(v___x_139_);
if (v_isSharedCheck_185_ == 0)
{
v___x_180_ = v___x_139_;
v_isShared_181_ = v_isSharedCheck_185_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_a_178_);
lean_dec(v___x_139_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_185_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___x_183_; 
if (v_isShared_181_ == 0)
{
v___x_183_ = v___x_180_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v_a_178_);
v___x_183_ = v_reuseFailAlloc_184_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
return v___x_183_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___redArg___boxed(lean_object* v_x_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___redArg(v_x_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_);
lean_dec(v___y_190_);
lean_dec_ref(v___y_189_);
lean_dec(v___y_188_);
lean_dec_ref(v___y_187_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1(lean_object* v_00_u03b1_193_, lean_object* v_x_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___redArg(v_x_194_, v___y_195_, v___y_196_, v___y_197_, v___y_198_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___boxed(lean_object* v_00_u03b1_201_, lean_object* v_x_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1(v_00_u03b1_201_, v_x_202_, v___y_203_, v___y_204_, v___y_205_, v___y_206_);
lean_dec(v___y_206_);
lean_dec_ref(v___y_205_);
lean_dec(v___y_204_);
lean_dec_ref(v___y_203_);
return v_res_208_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__0(uint8_t v___x_209_, lean_object* v_x_210_){
_start:
{
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__0___boxed(lean_object* v___x_211_, lean_object* v_x_212_){
_start:
{
uint8_t v___x_5782__boxed_213_; uint8_t v_res_214_; lean_object* v_r_215_; 
v___x_5782__boxed_213_ = lean_unbox(v___x_211_);
v_res_214_ = lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__0(v___x_5782__boxed_213_, v_x_212_);
lean_dec(v_x_212_);
v_r_215_ = lean_box(v_res_214_);
return v_r_215_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___redArg(lean_object* v_msg_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_){
_start:
{
lean_object* v_ref_222_; lean_object* v___x_223_; lean_object* v_a_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_232_; 
v_ref_222_ = lean_ctor_get(v___y_219_, 5);
v___x_223_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1_spec__1(v_msg_216_, v___y_217_, v___y_218_, v___y_219_, v___y_220_);
v_a_224_ = lean_ctor_get(v___x_223_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_232_ == 0)
{
v___x_226_ = v___x_223_;
v_isShared_227_ = v_isSharedCheck_232_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_a_224_);
lean_dec(v___x_223_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_232_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
lean_object* v___x_228_; lean_object* v___x_230_; 
lean_inc(v_ref_222_);
v___x_228_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_228_, 0, v_ref_222_);
lean_ctor_set(v___x_228_, 1, v_a_224_);
if (v_isShared_227_ == 0)
{
lean_ctor_set_tag(v___x_226_, 1);
lean_ctor_set(v___x_226_, 0, v___x_228_);
v___x_230_ = v___x_226_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v___x_228_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___redArg___boxed(lean_object* v_msg_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___redArg(v_msg_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_);
lean_dec(v___y_237_);
lean_dec_ref(v___y_236_);
lean_dec(v___y_235_);
lean_dec_ref(v___y_234_);
return v_res_239_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__1(void){
_start:
{
lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_241_ = ((lean_object*)(lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__0));
v___x_242_ = l_Lean_stringToMessageData(v___x_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1(lean_object* v___x_243_, lean_object* v_script_244_, lean_object* v___x_245_, uint8_t v_expectCompleteProof_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_){
_start:
{
lean_object* v___x_254_; lean_object* v_a_256_; lean_object* v___y_260_; lean_object* v___x_262_; 
lean_inc(v___x_243_);
v___x_254_ = lean_st_mk_ref(v___x_243_);
v___x_262_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_243_, v___x_254_);
if (lean_obj_tag(v___x_262_) == 0)
{
lean_object* v___x_263_; 
lean_dec_ref_known(v___x_262_, 1);
v___x_263_ = l_Lean_Elab_Tactic_evalTactic(v_script_244_, v___x_245_, v___x_254_, v___y_247_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_);
if (lean_obj_tag(v___x_263_) == 0)
{
lean_object* v___x_264_; 
lean_dec_ref_known(v___x_263_, 1);
v___x_264_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v___x_245_, v___x_254_, v___y_247_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_);
if (lean_obj_tag(v___x_264_) == 0)
{
lean_object* v_a_265_; 
v_a_265_ = lean_ctor_get(v___x_264_, 0);
lean_inc(v_a_265_);
lean_dec_ref_known(v___x_264_, 1);
if (v_expectCompleteProof_246_ == 0)
{
lean_dec(v_a_265_);
goto v___jp_266_;
}
else
{
uint8_t v___x_268_; 
v___x_268_ = l_List_isEmpty___redArg(v_a_265_);
lean_dec(v_a_265_);
if (v___x_268_ == 0)
{
lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_269_ = lean_obj_once(&lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__1, &lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__1_once, _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___closed__1);
v___x_270_ = lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___redArg(v___x_269_, v___y_249_, v___y_250_, v___y_251_, v___y_252_);
v___y_260_ = v___x_270_;
goto v___jp_259_;
}
else
{
goto v___jp_266_;
}
}
v___jp_266_:
{
lean_object* v___x_267_; 
v___x_267_ = lean_box(0);
v_a_256_ = v___x_267_;
goto v___jp_255_;
}
}
else
{
lean_object* v_a_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_278_; 
lean_dec(v___x_254_);
v_a_271_ = lean_ctor_get(v___x_264_, 0);
v_isSharedCheck_278_ = !lean_is_exclusive(v___x_264_);
if (v_isSharedCheck_278_ == 0)
{
v___x_273_ = v___x_264_;
v_isShared_274_ = v_isSharedCheck_278_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_a_271_);
lean_dec(v___x_264_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_278_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
lean_object* v___x_276_; 
if (v_isShared_274_ == 0)
{
v___x_276_ = v___x_273_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v_a_271_);
v___x_276_ = v_reuseFailAlloc_277_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
return v___x_276_;
}
}
}
}
else
{
v___y_260_ = v___x_263_;
goto v___jp_259_;
}
}
else
{
lean_dec(v_script_244_);
v___y_260_ = v___x_262_;
goto v___jp_259_;
}
v___jp_255_:
{
lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_257_ = lean_st_ref_get(v___x_254_);
lean_dec(v___x_254_);
lean_dec(v___x_257_);
v___x_258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_258_, 0, v_a_256_);
return v___x_258_;
}
v___jp_259_:
{
if (lean_obj_tag(v___y_260_) == 0)
{
lean_object* v_a_261_; 
v_a_261_ = lean_ctor_get(v___y_260_, 0);
lean_inc(v_a_261_);
lean_dec_ref_known(v___y_260_, 1);
v_a_256_ = v_a_261_;
goto v___jp_255_;
}
else
{
lean_dec(v___x_254_);
return v___y_260_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___boxed(lean_object* v___x_279_, lean_object* v_script_280_, lean_object* v___x_281_, lean_object* v_expectCompleteProof_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_){
_start:
{
uint8_t v_expectCompleteProof_boxed_290_; lean_object* v_res_291_; 
v_expectCompleteProof_boxed_290_ = lean_unbox(v_expectCompleteProof_282_);
v_res_291_ = lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1(v___x_279_, v_script_280_, v___x_281_, v_expectCompleteProof_boxed_290_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
lean_dec_ref(v___x_281_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2(lean_object* v_preState_294_, uint8_t v___x_295_, lean_object* v___x_296_, lean_object* v_script_297_, uint8_t v_expectCompleteProof_298_, uint8_t v_a_299_, lean_object* v___f_300_, lean_object* v___x_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = l_Lean_Meta_SavedState_restore___redArg(v_preState_294_, v___y_303_, v___y_305_);
if (lean_obj_tag(v___x_307_) == 0)
{
lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___f_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; 
lean_dec_ref_known(v___x_307_, 1);
v___x_308_ = lean_box(0);
v___x_309_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set_uint8(v___x_309_, sizeof(void*)*1, v___x_295_);
v___x_310_ = lean_box(v_expectCompleteProof_298_);
v___f_311_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__1___boxed), 11, 4);
lean_closure_set(v___f_311_, 0, v___x_296_);
lean_closure_set(v___f_311_, 1, v_script_297_);
lean_closure_set(v___f_311_, 2, v___x_309_);
lean_closure_set(v___f_311_, 3, v___x_310_);
v___x_312_ = lean_box(0);
v___x_313_ = lean_box(0);
v___x_314_ = lean_box(1);
v___x_315_ = ((lean_object*)(lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2___closed__0));
v___x_316_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_316_, 0, v___x_312_);
lean_ctor_set(v___x_316_, 1, v___x_313_);
lean_ctor_set(v___x_316_, 2, v___x_312_);
lean_ctor_set(v___x_316_, 3, v___f_300_);
lean_ctor_set(v___x_316_, 4, v___x_314_);
lean_ctor_set(v___x_316_, 5, v___x_314_);
lean_ctor_set(v___x_316_, 6, v___x_312_);
lean_ctor_set(v___x_316_, 7, v___x_315_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8, v_a_299_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 1, v_a_299_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 2, v_a_299_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 3, v_a_299_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 4, v___x_295_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 5, v___x_295_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 6, v___x_295_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 7, v___x_295_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 8, v_a_299_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 9, v___x_295_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*8 + 10, v_a_299_);
v___x_317_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_317_, 0, v___x_313_);
lean_ctor_set(v___x_317_, 1, v___x_314_);
lean_ctor_set(v___x_317_, 2, v___x_301_);
lean_ctor_set(v___x_317_, 3, v___x_313_);
lean_ctor_set(v___x_317_, 4, v___x_313_);
lean_ctor_set(v___x_317_, 5, v___x_314_);
lean_ctor_set(v___x_317_, 6, v___x_313_);
v___x_318_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___f_311_, v___x_316_, v___x_317_, v___y_302_, v___y_303_, v___y_304_, v___y_305_);
if (lean_obj_tag(v___x_318_) == 0)
{
lean_object* v_a_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_327_; 
v_a_319_ = lean_ctor_get(v___x_318_, 0);
v_isSharedCheck_327_ = !lean_is_exclusive(v___x_318_);
if (v_isSharedCheck_327_ == 0)
{
v___x_321_ = v___x_318_;
v_isShared_322_ = v_isSharedCheck_327_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_a_319_);
lean_dec(v___x_318_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_327_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v_fst_323_; lean_object* v___x_325_; 
v_fst_323_ = lean_ctor_get(v_a_319_, 0);
lean_inc(v_fst_323_);
lean_dec(v_a_319_);
if (v_isShared_322_ == 0)
{
lean_ctor_set(v___x_321_, 0, v_fst_323_);
v___x_325_ = v___x_321_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v_fst_323_);
v___x_325_ = v_reuseFailAlloc_326_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
return v___x_325_;
}
}
}
else
{
lean_object* v_a_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_335_; 
v_a_328_ = lean_ctor_get(v___x_318_, 0);
v_isSharedCheck_335_ = !lean_is_exclusive(v___x_318_);
if (v_isSharedCheck_335_ == 0)
{
v___x_330_ = v___x_318_;
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_a_328_);
lean_dec(v___x_318_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
lean_object* v___x_333_; 
if (v_isShared_331_ == 0)
{
v___x_333_ = v___x_330_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v_a_328_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
else
{
lean_dec(v___x_301_);
lean_dec_ref(v___f_300_);
lean_dec(v_script_297_);
lean_dec(v___x_296_);
return v___x_307_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2___boxed(lean_object* v_preState_336_, lean_object* v___x_337_, lean_object* v___x_338_, lean_object* v_script_339_, lean_object* v_expectCompleteProof_340_, lean_object* v_a_341_, lean_object* v___f_342_, lean_object* v___x_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_){
_start:
{
uint8_t v___x_5919__boxed_349_; uint8_t v_expectCompleteProof_boxed_350_; uint8_t v_a_5921__boxed_351_; lean_object* v_res_352_; 
v___x_5919__boxed_349_ = lean_unbox(v___x_337_);
v_expectCompleteProof_boxed_350_ = lean_unbox(v_expectCompleteProof_340_);
v_a_5921__boxed_351_ = lean_unbox(v_a_341_);
v_res_352_ = lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2(v_preState_336_, v___x_5919__boxed_349_, v___x_338_, v_script_339_, v_expectCompleteProof_boxed_350_, v_a_5921__boxed_351_, v___f_342_, v___x_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
lean_dec(v___y_345_);
lean_dec_ref(v___y_344_);
lean_dec_ref(v_preState_336_);
return v_res_352_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__1(void){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_356_ = lp_aesop_Aesop_Check_script;
v___x_357_ = lp_aesop_Aesop_Check_name(v___x_356_);
return v___x_357_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__2(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_358_ = lean_obj_once(&lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__1, &lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__1_once, _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__1);
v___x_359_ = l_Lean_MessageData_ofName(v___x_358_);
return v___x_359_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__4(void){
_start:
{
lean_object* v___x_361_; lean_object* v___x_362_; 
v___x_361_ = ((lean_object*)(lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__3));
v___x_362_ = l_Lean_stringToMessageData(v___x_361_);
return v___x_362_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__5(void){
_start:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_363_ = lean_obj_once(&lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__4, &lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__4_once, _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__4);
v___x_364_ = lean_obj_once(&lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__2, &lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__2_once, _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__2);
v___x_365_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_364_);
lean_ctor_set(v___x_365_, 1, v___x_363_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled(lean_object* v_script_366_, lean_object* v_preState_367_, lean_object* v_goal_368_, uint8_t v_expectCompleteProof_369_, lean_object* v_a_370_, lean_object* v_a_371_, lean_object* v_a_372_, lean_object* v_a_373_){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v_a_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_404_; 
v___x_375_ = lp_aesop_Aesop_Check_script;
v___x_376_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Script_UScript_checkIfEnabled_spec__0___redArg(v___x_375_, v_a_372_);
v_a_377_ = lean_ctor_get(v___x_376_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_404_ == 0)
{
v___x_379_ = v___x_376_;
v_isShared_380_ = v_isSharedCheck_404_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_a_377_);
lean_dec(v___x_376_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_404_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
uint8_t v___x_381_; 
v___x_381_ = lean_unbox(v_a_377_);
if (v___x_381_ == 0)
{
lean_object* v___x_382_; lean_object* v___x_384_; 
lean_dec(v_a_377_);
lean_dec(v_goal_368_);
lean_dec_ref(v_preState_367_);
lean_dec(v_script_366_);
v___x_382_ = lean_box(0);
if (v_isShared_380_ == 0)
{
lean_ctor_set(v___x_379_, 0, v___x_382_);
v___x_384_ = v___x_379_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v___x_382_);
v___x_384_ = v_reuseFailAlloc_385_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
return v___x_384_;
}
}
else
{
uint8_t v___x_386_; lean_object* v___f_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___f_392_; lean_object* v___x_393_; 
lean_del_object(v___x_379_);
v___x_386_ = 0;
v___f_387_ = ((lean_object*)(lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__0));
v___x_388_ = lean_box(0);
v___x_389_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_389_, 0, v_goal_368_);
lean_ctor_set(v___x_389_, 1, v___x_388_);
v___x_390_ = lean_box(v___x_386_);
v___x_391_ = lean_box(v_expectCompleteProof_369_);
v___f_392_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkRenderedScriptIfEnabled___lam__2___boxed), 13, 8);
lean_closure_set(v___f_392_, 0, v_preState_367_);
lean_closure_set(v___f_392_, 1, v___x_390_);
lean_closure_set(v___f_392_, 2, v___x_389_);
lean_closure_set(v___f_392_, 3, v_script_366_);
lean_closure_set(v___f_392_, 4, v___x_391_);
lean_closure_set(v___f_392_, 5, v_a_377_);
lean_closure_set(v___f_392_, 6, v___f_387_);
lean_closure_set(v___f_392_, 7, v___x_388_);
v___x_393_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_checkRenderedScriptIfEnabled_spec__1___redArg(v___f_392_, v_a_370_, v_a_371_, v_a_372_, v_a_373_);
if (lean_obj_tag(v___x_393_) == 0)
{
return v___x_393_;
}
else
{
lean_object* v_a_394_; uint8_t v___y_396_; uint8_t v___x_402_; 
v_a_394_ = lean_ctor_get(v___x_393_, 0);
lean_inc(v_a_394_);
v___x_402_ = l_Lean_Exception_isInterrupt(v_a_394_);
if (v___x_402_ == 0)
{
uint8_t v___x_403_; 
lean_inc(v_a_394_);
v___x_403_ = l_Lean_Exception_isRuntime(v_a_394_);
v___y_396_ = v___x_403_;
goto v___jp_395_;
}
else
{
v___y_396_ = v___x_402_;
goto v___jp_395_;
}
v___jp_395_:
{
if (v___y_396_ == 0)
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; 
lean_dec_ref_known(v___x_393_, 1);
v___x_397_ = lean_obj_once(&lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__5, &lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__5_once, _init_lp_aesop_Aesop_checkRenderedScriptIfEnabled___closed__5);
v___x_398_ = l_Lean_Exception_toMessageData(v_a_394_);
v___x_399_ = l_Lean_indentD(v___x_398_);
v___x_400_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_400_, 0, v___x_397_);
lean_ctor_set(v___x_400_, 1, v___x_399_);
v___x_401_ = lp_aesop_Lean_throwError___at___00Aesop_Script_UScript_checkIfEnabled_spec__1___redArg(v___x_400_, v_a_370_, v_a_371_, v_a_372_, v_a_373_);
return v___x_401_;
}
else
{
lean_dec(v_a_394_);
return v___x_393_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___boxed(lean_object* v_script_405_, lean_object* v_preState_406_, lean_object* v_goal_407_, lean_object* v_expectCompleteProof_408_, lean_object* v_a_409_, lean_object* v_a_410_, lean_object* v_a_411_, lean_object* v_a_412_, lean_object* v_a_413_){
_start:
{
uint8_t v_expectCompleteProof_boxed_414_; lean_object* v_res_415_; 
v_expectCompleteProof_boxed_414_ = lean_unbox(v_expectCompleteProof_408_);
v_res_415_ = lp_aesop_Aesop_checkRenderedScriptIfEnabled(v_script_405_, v_preState_406_, v_goal_407_, v_expectCompleteProof_boxed_414_, v_a_409_, v_a_410_, v_a_411_, v_a_412_);
lean_dec(v_a_412_);
lean_dec_ref(v_a_411_);
lean_dec(v_a_410_);
lean_dec_ref(v_a_409_);
return v_res_415_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0(lean_object* v_00_u03b1_416_, lean_object* v_msg_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___redArg(v_msg_417_, v___y_422_, v___y_423_, v___y_424_, v___y_425_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0___boxed(lean_object* v_00_u03b1_428_, lean_object* v_msg_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_aesop_Lean_throwError___at___00Aesop_checkRenderedScriptIfEnabled_spec__0(v_00_u03b1_428_, v_msg_429_, v___y_430_, v___y_431_, v___y_432_, v___y_433_, v___y_434_, v___y_435_, v___y_436_, v___y_437_);
lean_dec(v___y_437_);
lean_dec_ref(v___y_436_);
lean_dec(v___y_435_);
lean_dec_ref(v___y_434_);
lean_dec(v___y_433_);
lean_dec_ref(v___y_432_);
lean_dec(v___y_431_);
lean_dec_ref(v___y_430_);
return v_res_439_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_UScript(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Check(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_Check(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_UScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_Check(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Script_UScript(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Check(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_Check(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_UScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_Check(builtin);
}
#ifdef __cplusplus
}
#endif
