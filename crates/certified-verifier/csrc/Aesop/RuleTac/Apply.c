// Lean compiler output
// Module: Aesop.RuleTac.Apply
// Imports: public import Init public meta import Init public import Aesop.RuleTac.Basic public import Aesop.RuleTac.RuleTerm public import Aesop.RuleTac.ElabRuleTerm public import Aesop.Script.SpecificTactics
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lp_aesop_Aesop_Substitution_specializeRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_applyS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lp_aesop_Aesop_isGoalDiffDefeqTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "aesop: internal error in applyExpr': multiple steps"};
static const lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleTac_applyExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "failed to apply '"};
static const lean_object* lp_aesop_Aesop_RuleTac_applyExpr___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_applyExpr___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_applyExpr___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_applyExpr___closed__1;
static const lean_string_object lp_aesop_Aesop_RuleTac_applyExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "' with any of the matched instances of the rule pattern"};
static const lean_object* lp_aesop_Aesop_RuleTac_applyExpr___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleTac_applyExpr___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_applyExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_applyExpr___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyConst(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyConst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyTerm___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyTerm___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyTerm(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_apply(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0___closed__0 = (const lean_object*)&lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_applyConsts_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleTac_applyConsts___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "failed to apply any of these declarations: "};
static const lean_object* lp_aesop_Aesop_RuleTac_applyConsts___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_applyConsts___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_applyConsts___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_applyConsts___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyConsts(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyConsts___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg(lean_object* v_x_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_, lean_object* v___y_8_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_10_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg___closed__0));
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg___boxed(lean_object* v_x_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg(v_x_31_, v___y_32_, v___y_33_, v___y_34_, v___y_35_, v___y_36_);
lean_dec(v___y_36_);
lean_dec_ref(v___y_35_);
lean_dec(v___y_34_);
lean_dec_ref(v___y_33_);
lean_dec(v___y_32_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0(lean_object* v_00_u03b1_39_, lean_object* v_x_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg(v_x_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___boxed(lean_object* v_00_u03b1_48_, lean_object* v_x_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0(v_00_u03b1_48_, v_x_49_, v___y_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_);
lean_dec(v___y_54_);
lean_dec_ref(v___y_53_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec(v___y_50_);
return v_res_56_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_57_ = lean_box(0);
v___x_58_ = lean_unsigned_to_nat(16u);
v___x_59_ = lean_mk_array(v___x_58_, v___x_57_);
return v___x_59_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__0, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__0_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__0);
v___x_61_ = lean_unsigned_to_nat(0u);
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v___x_60_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg(lean_object* v_goal_63_, size_t v_sz_64_, size_t v_i_65_, lean_object* v_bs_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_){
_start:
{
uint8_t v___x_72_; 
v___x_72_ = lean_usize_dec_lt(v_i_65_, v_sz_64_);
if (v___x_72_ == 0)
{
lean_object* v___x_73_; 
lean_dec(v_goal_63_);
v___x_73_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_73_, 0, v_bs_66_);
return v___x_73_;
}
else
{
lean_object* v_v_74_; lean_object* v___x_75_; 
v_v_74_ = lean_array_uget(v_bs_66_, v_i_65_);
lean_inc(v_v_74_);
lean_inc(v_goal_63_);
v___x_75_ = lp_aesop_Aesop_isGoalDiffDefeqTarget(v_goal_63_, v_v_74_, v___y_67_, v___y_68_, v___y_69_, v___y_70_);
if (lean_obj_tag(v___x_75_) == 0)
{
lean_object* v_a_76_; lean_object* v___x_77_; lean_object* v_bs_x27_78_; lean_object* v___x_79_; uint8_t v___y_81_; uint8_t v___x_87_; 
v_a_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_a_76_);
lean_dec_ref_known(v___x_75_, 1);
v___x_77_ = lean_unsigned_to_nat(0u);
v_bs_x27_78_ = lean_array_uset(v_bs_66_, v_i_65_, v___x_77_);
v___x_79_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___closed__1);
v___x_87_ = lean_unbox(v_a_76_);
lean_dec(v_a_76_);
if (v___x_87_ == 0)
{
v___y_81_ = v___x_72_;
goto v___jp_80_;
}
else
{
uint8_t v___x_88_; 
v___x_88_ = 0;
v___y_81_ = v___x_88_;
goto v___jp_80_;
}
v___jp_80_:
{
lean_object* v___x_82_; size_t v___x_83_; size_t v___x_84_; lean_object* v___x_85_; 
lean_inc(v_goal_63_);
v___x_82_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_82_, 0, v_goal_63_);
lean_ctor_set(v___x_82_, 1, v_v_74_);
lean_ctor_set(v___x_82_, 2, v___x_79_);
lean_ctor_set(v___x_82_, 3, v___x_79_);
lean_ctor_set_uint8(v___x_82_, sizeof(void*)*4, v___y_81_);
v___x_83_ = ((size_t)1ULL);
v___x_84_ = lean_usize_add(v_i_65_, v___x_83_);
v___x_85_ = lean_array_uset(v_bs_x27_78_, v_i_65_, v___x_82_);
v_i_65_ = v___x_84_;
v_bs_66_ = v___x_85_;
goto _start;
}
}
else
{
lean_object* v_a_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_96_; 
lean_dec(v_v_74_);
lean_dec_ref(v_bs_66_);
lean_dec(v_goal_63_);
v_a_89_ = lean_ctor_get(v___x_75_, 0);
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_75_);
if (v_isSharedCheck_96_ == 0)
{
v___x_91_ = v___x_75_;
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_a_89_);
lean_dec(v___x_75_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_94_; 
if (v_isShared_92_ == 0)
{
v___x_94_ = v___x_91_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v_a_89_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg___boxed(lean_object* v_goal_97_, lean_object* v_sz_98_, lean_object* v_i_99_, lean_object* v_bs_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
size_t v_sz_boxed_106_; size_t v_i_boxed_107_; lean_object* v_res_108_; 
v_sz_boxed_106_ = lean_unbox_usize(v_sz_98_);
lean_dec(v_sz_98_);
v_i_boxed_107_ = lean_unbox_usize(v_i_99_);
lean_dec(v_i_99_);
v_res_108_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg(v_goal_97_, v_sz_boxed_106_, v_i_boxed_107_, v_bs_100_, v___y_101_, v___y_102_, v___y_103_, v___y_104_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
lean_dec(v___y_102_);
lean_dec_ref(v___y_101_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1_spec__1(lean_object* v_msgData_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_){
_start:
{
lean_object* v___x_115_; lean_object* v_env_116_; lean_object* v___x_117_; lean_object* v_mctx_118_; lean_object* v_lctx_119_; lean_object* v_options_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_115_ = lean_st_ref_get(v___y_113_);
v_env_116_ = lean_ctor_get(v___x_115_, 0);
lean_inc_ref(v_env_116_);
lean_dec(v___x_115_);
v___x_117_ = lean_st_ref_get(v___y_111_);
v_mctx_118_ = lean_ctor_get(v___x_117_, 0);
lean_inc_ref(v_mctx_118_);
lean_dec(v___x_117_);
v_lctx_119_ = lean_ctor_get(v___y_110_, 2);
v_options_120_ = lean_ctor_get(v___y_112_, 2);
lean_inc_ref(v_options_120_);
lean_inc_ref(v_lctx_119_);
v___x_121_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_121_, 0, v_env_116_);
lean_ctor_set(v___x_121_, 1, v_mctx_118_);
lean_ctor_set(v___x_121_, 2, v_lctx_119_);
lean_ctor_set(v___x_121_, 3, v_options_120_);
v___x_122_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_msgData_109_);
v___x_123_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1_spec__1___boxed(lean_object* v_msgData_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1_spec__1(v_msgData_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg(lean_object* v_msg_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_){
_start:
{
lean_object* v_ref_137_; lean_object* v___x_138_; lean_object* v_a_139_; lean_object* v___x_141_; uint8_t v_isShared_142_; uint8_t v_isSharedCheck_147_; 
v_ref_137_ = lean_ctor_get(v___y_134_, 5);
v___x_138_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1_spec__1(v_msg_131_, v___y_132_, v___y_133_, v___y_134_, v___y_135_);
v_a_139_ = lean_ctor_get(v___x_138_, 0);
v_isSharedCheck_147_ = !lean_is_exclusive(v___x_138_);
if (v_isSharedCheck_147_ == 0)
{
v___x_141_ = v___x_138_;
v_isShared_142_ = v_isSharedCheck_147_;
goto v_resetjp_140_;
}
else
{
lean_inc(v_a_139_);
lean_dec(v___x_138_);
v___x_141_ = lean_box(0);
v_isShared_142_ = v_isSharedCheck_147_;
goto v_resetjp_140_;
}
v_resetjp_140_:
{
lean_object* v___x_143_; lean_object* v___x_145_; 
lean_inc(v_ref_137_);
v___x_143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_143_, 0, v_ref_137_);
lean_ctor_set(v___x_143_, 1, v_a_139_);
if (v_isShared_142_ == 0)
{
lean_ctor_set_tag(v___x_141_, 1);
lean_ctor_set(v___x_141_, 0, v___x_143_);
v___x_145_ = v___x_141_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v___x_143_);
v___x_145_ = v_reuseFailAlloc_146_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
return v___x_145_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg___boxed(lean_object* v_msg_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg(v_msg_148_, v___y_149_, v___y_150_, v___y_151_, v___y_152_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
return v_res_154_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__1(void){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__0));
v___x_157_ = l_Lean_stringToMessageData(v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0(lean_object* v_eStx_158_, lean_object* v_goal_159_, uint8_t v_md_160_, lean_object* v_e_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_168_, 0, v_eStx_158_);
v___x_169_ = lean_box(v_md_160_);
lean_inc(v_goal_159_);
v___x_170_ = lean_alloc_closure((void*)(lp_aesop_Aesop_applyS___boxed), 11, 4);
lean_closure_set(v___x_170_, 0, v_goal_159_);
lean_closure_set(v___x_170_, 1, v_e_161_);
lean_closure_set(v___x_170_, 2, v___x_168_);
lean_closure_set(v___x_170_, 3, v___x_169_);
v___x_171_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_applyExpr_x27_spec__0___redArg(v___x_170_, v___y_162_, v___y_163_, v___y_164_, v___y_165_, v___y_166_);
if (lean_obj_tag(v___x_171_) == 0)
{
lean_object* v_a_172_; lean_object* v_fst_173_; lean_object* v_snd_174_; lean_object* v___x_175_; lean_object* v___x_176_; uint8_t v___x_177_; 
v_a_172_ = lean_ctor_get(v___x_171_, 0);
lean_inc(v_a_172_);
lean_dec_ref_known(v___x_171_, 1);
v_fst_173_ = lean_ctor_get(v_a_172_, 0);
lean_inc(v_fst_173_);
v_snd_174_ = lean_ctor_get(v_a_172_, 1);
lean_inc(v_snd_174_);
lean_dec(v_a_172_);
v___x_175_ = lean_array_get_size(v_snd_174_);
v___x_176_ = lean_unsigned_to_nat(1u);
v___x_177_ = lean_nat_dec_eq(v___x_175_, v___x_176_);
if (v___x_177_ == 0)
{
lean_object* v___x_178_; lean_object* v___x_179_; 
lean_dec(v_snd_174_);
lean_dec(v_fst_173_);
lean_dec(v_goal_159_);
v___x_178_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__1, &lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__1_once, _init_lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___closed__1);
v___x_179_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg(v___x_178_, v___y_163_, v___y_164_, v___y_165_, v___y_166_);
return v___x_179_;
}
else
{
size_t v_sz_180_; size_t v___x_181_; lean_object* v___x_182_; 
v_sz_180_ = lean_array_size(v_fst_173_);
v___x_181_ = ((size_t)0ULL);
v___x_182_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg(v_goal_159_, v_sz_180_, v___x_181_, v_fst_173_, v___y_163_, v___y_164_, v___y_165_, v___y_166_);
if (lean_obj_tag(v___x_182_) == 0)
{
lean_object* v_a_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_198_; 
v_a_183_ = lean_ctor_get(v___x_182_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_198_ == 0)
{
v___x_185_ = v___x_182_;
v_isShared_186_ = v_isSharedCheck_198_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_a_183_);
lean_dec(v___x_182_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_198_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v_postState_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_196_; 
v___x_187_ = lean_unsigned_to_nat(0u);
v___x_188_ = lean_array_fget(v_snd_174_, v___x_187_);
lean_dec(v_snd_174_);
v_postState_189_ = lean_ctor_get(v___x_188_, 3);
lean_inc_ref(v_postState_189_);
v___x_190_ = lean_mk_empty_array_with_capacity(v___x_176_);
v___x_191_ = lean_array_push(v___x_190_, v___x_188_);
v___x_192_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_192_, 0, v___x_191_);
v___x_193_ = lean_box(0);
v___x_194_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_194_, 0, v_a_183_);
lean_ctor_set(v___x_194_, 1, v_postState_189_);
lean_ctor_set(v___x_194_, 2, v___x_192_);
lean_ctor_set(v___x_194_, 3, v___x_193_);
if (v_isShared_186_ == 0)
{
lean_ctor_set(v___x_185_, 0, v___x_194_);
v___x_196_ = v___x_185_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v___x_194_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
else
{
lean_object* v_a_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_206_; 
lean_dec(v_snd_174_);
v_a_199_ = lean_ctor_get(v___x_182_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_206_ == 0)
{
v___x_201_ = v___x_182_;
v_isShared_202_ = v_isSharedCheck_206_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_a_199_);
lean_dec(v___x_182_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_206_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___x_204_; 
if (v_isShared_202_ == 0)
{
v___x_204_ = v___x_201_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v_a_199_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
return v___x_204_;
}
}
}
}
}
else
{
lean_object* v_a_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_214_; 
lean_dec(v_goal_159_);
v_a_207_ = lean_ctor_get(v___x_171_, 0);
v_isSharedCheck_214_ = !lean_is_exclusive(v___x_171_);
if (v_isSharedCheck_214_ == 0)
{
v___x_209_ = v___x_171_;
v_isShared_210_ = v_isSharedCheck_214_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_a_207_);
lean_dec(v___x_171_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_214_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v___x_212_; 
if (v_isShared_210_ == 0)
{
v___x_212_ = v___x_209_;
goto v_reusejp_211_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v_a_207_);
v___x_212_ = v_reuseFailAlloc_213_;
goto v_reusejp_211_;
}
v_reusejp_211_:
{
return v___x_212_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0___boxed(lean_object* v_eStx_215_, lean_object* v_goal_216_, lean_object* v_md_217_, lean_object* v_e_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_){
_start:
{
uint8_t v_md_boxed_225_; lean_object* v_res_226_; 
v_md_boxed_225_ = lean_unbox(v_md_217_);
v_res_226_ = lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0(v_eStx_215_, v_goal_216_, v_md_boxed_225_, v_e_218_, v___y_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_);
lean_dec(v___y_223_);
lean_dec_ref(v___y_222_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
lean_dec(v___y_219_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27(lean_object* v_goal_227_, lean_object* v_e_228_, lean_object* v_eStx_229_, lean_object* v_patSubst_x3f_230_, uint8_t v_md_231_, lean_object* v_a_232_, lean_object* v_a_233_, lean_object* v_a_234_, lean_object* v_a_235_, lean_object* v_a_236_){
_start:
{
lean_object* v___y_239_; lean_object* v_keyedConfig_248_; uint8_t v_trackZetaDelta_249_; lean_object* v_zetaDeltaSet_250_; lean_object* v_lctx_251_; lean_object* v_localInstances_252_; lean_object* v_defEqCtx_x3f_253_; lean_object* v_synthPendingDepth_254_; lean_object* v_customCanUnfoldPredicate_x3f_255_; uint8_t v_univApprox_256_; uint8_t v_inTypeClassResolution_257_; uint8_t v_cacheInferType_258_; lean_object* v___x_259_; lean_object* v___x_260_; 
v_keyedConfig_248_ = lean_ctor_get(v_a_233_, 0);
v_trackZetaDelta_249_ = lean_ctor_get_uint8(v_a_233_, sizeof(void*)*7);
v_zetaDeltaSet_250_ = lean_ctor_get(v_a_233_, 1);
v_lctx_251_ = lean_ctor_get(v_a_233_, 2);
v_localInstances_252_ = lean_ctor_get(v_a_233_, 3);
v_defEqCtx_x3f_253_ = lean_ctor_get(v_a_233_, 4);
v_synthPendingDepth_254_ = lean_ctor_get(v_a_233_, 5);
v_customCanUnfoldPredicate_x3f_255_ = lean_ctor_get(v_a_233_, 6);
v_univApprox_256_ = lean_ctor_get_uint8(v_a_233_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_257_ = lean_ctor_get_uint8(v_a_233_, sizeof(void*)*7 + 2);
v_cacheInferType_258_ = lean_ctor_get_uint8(v_a_233_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_248_);
v___x_259_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_md_231_, v_keyedConfig_248_);
lean_inc(v_customCanUnfoldPredicate_x3f_255_);
lean_inc(v_synthPendingDepth_254_);
lean_inc(v_defEqCtx_x3f_253_);
lean_inc_ref(v_localInstances_252_);
lean_inc_ref(v_lctx_251_);
lean_inc(v_zetaDeltaSet_250_);
v___x_260_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_260_, 0, v___x_259_);
lean_ctor_set(v___x_260_, 1, v_zetaDeltaSet_250_);
lean_ctor_set(v___x_260_, 2, v_lctx_251_);
lean_ctor_set(v___x_260_, 3, v_localInstances_252_);
lean_ctor_set(v___x_260_, 4, v_defEqCtx_x3f_253_);
lean_ctor_set(v___x_260_, 5, v_synthPendingDepth_254_);
lean_ctor_set(v___x_260_, 6, v_customCanUnfoldPredicate_x3f_255_);
lean_ctor_set_uint8(v___x_260_, sizeof(void*)*7, v_trackZetaDelta_249_);
lean_ctor_set_uint8(v___x_260_, sizeof(void*)*7 + 1, v_univApprox_256_);
lean_ctor_set_uint8(v___x_260_, sizeof(void*)*7 + 2, v_inTypeClassResolution_257_);
lean_ctor_set_uint8(v___x_260_, sizeof(void*)*7 + 3, v_cacheInferType_258_);
if (lean_obj_tag(v_patSubst_x3f_230_) == 1)
{
lean_object* v_val_261_; lean_object* v___x_262_; 
v_val_261_ = lean_ctor_get(v_patSubst_x3f_230_, 0);
lean_inc(v_val_261_);
lean_dec_ref_known(v_patSubst_x3f_230_, 1);
v___x_262_ = lp_aesop_Aesop_Substitution_specializeRule(v_e_228_, v_val_261_, v___x_260_, v_a_234_, v_a_235_, v_a_236_);
if (lean_obj_tag(v___x_262_) == 0)
{
lean_object* v_a_263_; lean_object* v___x_264_; 
v_a_263_ = lean_ctor_get(v___x_262_, 0);
lean_inc(v_a_263_);
lean_dec_ref_known(v___x_262_, 1);
v___x_264_ = lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0(v_eStx_229_, v_goal_227_, v_md_231_, v_a_263_, v_a_232_, v___x_260_, v_a_234_, v_a_235_, v_a_236_);
lean_dec_ref_known(v___x_260_, 7);
v___y_239_ = v___x_264_;
goto v___jp_238_;
}
else
{
lean_object* v_a_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_272_; 
lean_dec_ref_known(v___x_260_, 7);
lean_dec(v_eStx_229_);
lean_dec(v_goal_227_);
v_a_265_ = lean_ctor_get(v___x_262_, 0);
v_isSharedCheck_272_ = !lean_is_exclusive(v___x_262_);
if (v_isSharedCheck_272_ == 0)
{
v___x_267_ = v___x_262_;
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_a_265_);
lean_dec(v___x_262_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_270_; 
if (v_isShared_268_ == 0)
{
v___x_270_ = v___x_267_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_a_265_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
return v___x_270_;
}
}
}
}
else
{
lean_object* v___x_273_; 
lean_dec(v_patSubst_x3f_230_);
v___x_273_ = lp_aesop_Aesop_RuleTac_applyExpr_x27___lam__0(v_eStx_229_, v_goal_227_, v_md_231_, v_e_228_, v_a_232_, v___x_260_, v_a_234_, v_a_235_, v_a_236_);
lean_dec_ref_known(v___x_260_, 7);
v___y_239_ = v___x_273_;
goto v___jp_238_;
}
v___jp_238_:
{
if (lean_obj_tag(v___y_239_) == 0)
{
lean_object* v_a_240_; lean_object* v___x_242_; uint8_t v_isShared_243_; uint8_t v_isSharedCheck_247_; 
v_a_240_ = lean_ctor_get(v___y_239_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v___y_239_);
if (v_isSharedCheck_247_ == 0)
{
v___x_242_ = v___y_239_;
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
else
{
lean_inc(v_a_240_);
lean_dec(v___y_239_);
v___x_242_ = lean_box(0);
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
v_resetjp_241_:
{
lean_object* v___x_245_; 
if (v_isShared_243_ == 0)
{
v___x_245_ = v___x_242_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v_a_240_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
else
{
return v___y_239_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr_x27___boxed(lean_object* v_goal_274_, lean_object* v_e_275_, lean_object* v_eStx_276_, lean_object* v_patSubst_x3f_277_, lean_object* v_md_278_, lean_object* v_a_279_, lean_object* v_a_280_, lean_object* v_a_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_){
_start:
{
uint8_t v_md_boxed_285_; lean_object* v_res_286_; 
v_md_boxed_285_ = lean_unbox(v_md_278_);
v_res_286_ = lp_aesop_Aesop_RuleTac_applyExpr_x27(v_goal_274_, v_e_275_, v_eStx_276_, v_patSubst_x3f_277_, v_md_boxed_285_, v_a_279_, v_a_280_, v_a_281_, v_a_282_, v_a_283_);
lean_dec(v_a_283_);
lean_dec_ref(v_a_282_);
lean_dec(v_a_281_);
lean_dec_ref(v_a_280_);
lean_dec(v_a_279_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1(lean_object* v_00_u03b1_287_, lean_object* v_msg_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg(v_msg_288_, v___y_290_, v___y_291_, v___y_292_, v___y_293_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___boxed(lean_object* v_00_u03b1_296_, lean_object* v_msg_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1(v_00_u03b1_296_, v_msg_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
lean_dec(v___y_298_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2(lean_object* v_goal_305_, size_t v_sz_306_, size_t v_i_307_, lean_object* v_bs_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___redArg(v_goal_305_, v_sz_306_, v_i_307_, v_bs_308_, v___y_310_, v___y_311_, v___y_312_, v___y_313_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2___boxed(lean_object* v_goal_316_, lean_object* v_sz_317_, lean_object* v_i_318_, lean_object* v_bs_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_){
_start:
{
size_t v_sz_boxed_326_; size_t v_i_boxed_327_; lean_object* v_res_328_; 
v_sz_boxed_326_ = lean_unbox_usize(v_sz_317_);
lean_dec(v_sz_317_);
v_i_boxed_327_ = lean_unbox_usize(v_i_318_);
lean_dec(v_i_318_);
v_res_328_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_applyExpr_x27_spec__2(v_goal_316_, v_sz_boxed_326_, v_i_boxed_327_, v_bs_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
lean_dec(v___y_320_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0(lean_object* v_a_331_, lean_object* v_goal_332_, lean_object* v_e_333_, lean_object* v_eStx_334_, uint8_t v_md_335_, lean_object* v_as_336_, size_t v_sz_337_, size_t v_i_338_, lean_object* v_b_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v_a_347_; uint8_t v___x_361_; 
v___x_361_ = lean_usize_dec_lt(v_i_338_, v_sz_337_);
if (v___x_361_ == 0)
{
lean_object* v___x_362_; 
lean_dec(v_eStx_334_);
lean_dec_ref(v_e_333_);
lean_dec(v_goal_332_);
v___x_362_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_362_, 0, v_b_339_);
return v___x_362_;
}
else
{
lean_object* v_a_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v_a_363_ = lean_array_uget_borrowed(v_as_336_, v_i_338_);
lean_inc(v_a_363_);
v___x_364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_364_, 0, v_a_363_);
lean_inc(v_eStx_334_);
lean_inc_ref(v_e_333_);
lean_inc(v_goal_332_);
v___x_365_ = lp_aesop_Aesop_RuleTac_applyExpr_x27(v_goal_332_, v_e_333_, v_eStx_334_, v___x_364_, v_md_335_, v___y_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_);
if (lean_obj_tag(v___x_365_) == 0)
{
lean_object* v_a_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
v_a_366_ = lean_ctor_get(v___x_365_, 0);
lean_inc(v_a_366_);
lean_dec_ref_known(v___x_365_, 1);
v___x_367_ = lean_array_push(v_b_339_, v_a_366_);
v___x_368_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0___closed__0));
v___x_369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
lean_ctor_set(v___x_369_, 1, v___x_367_);
v_a_347_ = v___x_369_;
goto v___jp_346_;
}
else
{
lean_object* v_a_370_; uint8_t v___y_372_; uint8_t v___x_392_; 
v_a_370_ = lean_ctor_get(v___x_365_, 0);
lean_inc(v_a_370_);
lean_dec_ref_known(v___x_365_, 1);
v___x_392_ = l_Lean_Exception_isInterrupt(v_a_370_);
if (v___x_392_ == 0)
{
uint8_t v___x_393_; 
lean_inc(v_a_370_);
v___x_393_ = l_Lean_Exception_isRuntime(v_a_370_);
v___y_372_ = v___x_393_;
goto v___jp_371_;
}
else
{
v___y_372_ = v___x_392_;
goto v___jp_371_;
}
v___jp_371_:
{
if (v___y_372_ == 0)
{
lean_object* v___x_373_; lean_object* v___x_374_; 
lean_dec(v_a_370_);
v___x_373_ = lean_box(0);
v___x_374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_374_, 0, v___x_373_);
lean_ctor_set(v___x_374_, 1, v_b_339_);
v_a_347_ = v___x_374_;
goto v___jp_346_;
}
else
{
lean_object* v___x_375_; 
lean_dec_ref(v_b_339_);
lean_dec(v_eStx_334_);
lean_dec_ref(v_e_333_);
lean_dec(v_goal_332_);
v___x_375_ = l_Lean_Meta_SavedState_restore___redArg(v_a_331_, v___y_342_, v___y_344_);
if (lean_obj_tag(v___x_375_) == 0)
{
lean_object* v___x_377_; uint8_t v_isShared_378_; uint8_t v_isSharedCheck_382_; 
v_isSharedCheck_382_ = !lean_is_exclusive(v___x_375_);
if (v_isSharedCheck_382_ == 0)
{
lean_object* v_unused_383_; 
v_unused_383_ = lean_ctor_get(v___x_375_, 0);
lean_dec(v_unused_383_);
v___x_377_ = v___x_375_;
v_isShared_378_ = v_isSharedCheck_382_;
goto v_resetjp_376_;
}
else
{
lean_dec(v___x_375_);
v___x_377_ = lean_box(0);
v_isShared_378_ = v_isSharedCheck_382_;
goto v_resetjp_376_;
}
v_resetjp_376_:
{
lean_object* v___x_380_; 
if (v_isShared_378_ == 0)
{
lean_ctor_set_tag(v___x_377_, 1);
lean_ctor_set(v___x_377_, 0, v_a_370_);
v___x_380_ = v___x_377_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v_a_370_);
v___x_380_ = v_reuseFailAlloc_381_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
return v___x_380_;
}
}
}
else
{
lean_object* v_a_384_; lean_object* v___x_386_; uint8_t v_isShared_387_; uint8_t v_isSharedCheck_391_; 
lean_dec(v_a_370_);
v_a_384_ = lean_ctor_get(v___x_375_, 0);
v_isSharedCheck_391_ = !lean_is_exclusive(v___x_375_);
if (v_isSharedCheck_391_ == 0)
{
v___x_386_ = v___x_375_;
v_isShared_387_ = v_isSharedCheck_391_;
goto v_resetjp_385_;
}
else
{
lean_inc(v_a_384_);
lean_dec(v___x_375_);
v___x_386_ = lean_box(0);
v_isShared_387_ = v_isSharedCheck_391_;
goto v_resetjp_385_;
}
v_resetjp_385_:
{
lean_object* v___x_389_; 
if (v_isShared_387_ == 0)
{
v___x_389_ = v___x_386_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v_a_384_);
v___x_389_ = v_reuseFailAlloc_390_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
return v___x_389_;
}
}
}
}
}
}
}
v___jp_346_:
{
lean_object* v___x_348_; 
v___x_348_ = l_Lean_Meta_SavedState_restore___redArg(v_a_331_, v___y_342_, v___y_344_);
if (lean_obj_tag(v___x_348_) == 0)
{
lean_object* v_snd_349_; size_t v___x_350_; size_t v___x_351_; 
lean_dec_ref_known(v___x_348_, 1);
v_snd_349_ = lean_ctor_get(v_a_347_, 1);
lean_inc(v_snd_349_);
lean_dec_ref(v_a_347_);
v___x_350_ = ((size_t)1ULL);
v___x_351_ = lean_usize_add(v_i_338_, v___x_350_);
v_i_338_ = v___x_351_;
v_b_339_ = v_snd_349_;
goto _start;
}
else
{
lean_object* v_a_353_; lean_object* v___x_355_; uint8_t v_isShared_356_; uint8_t v_isSharedCheck_360_; 
lean_dec_ref(v_a_347_);
lean_dec(v_eStx_334_);
lean_dec_ref(v_e_333_);
lean_dec(v_goal_332_);
v_a_353_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_360_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_360_ == 0)
{
v___x_355_ = v___x_348_;
v_isShared_356_ = v_isSharedCheck_360_;
goto v_resetjp_354_;
}
else
{
lean_inc(v_a_353_);
lean_dec(v___x_348_);
v___x_355_ = lean_box(0);
v_isShared_356_ = v_isSharedCheck_360_;
goto v_resetjp_354_;
}
v_resetjp_354_:
{
lean_object* v___x_358_; 
if (v_isShared_356_ == 0)
{
v___x_358_ = v___x_355_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v_a_353_);
v___x_358_ = v_reuseFailAlloc_359_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
return v___x_358_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0___boxed(lean_object* v_a_394_, lean_object* v_goal_395_, lean_object* v_e_396_, lean_object* v_eStx_397_, lean_object* v_md_398_, lean_object* v_as_399_, lean_object* v_sz_400_, lean_object* v_i_401_, lean_object* v_b_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_){
_start:
{
uint8_t v_md_boxed_409_; size_t v_sz_boxed_410_; size_t v_i_boxed_411_; lean_object* v_res_412_; 
v_md_boxed_409_ = lean_unbox(v_md_398_);
v_sz_boxed_410_ = lean_unbox_usize(v_sz_400_);
lean_dec(v_sz_400_);
v_i_boxed_411_ = lean_unbox_usize(v_i_401_);
lean_dec(v_i_401_);
v_res_412_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0(v_a_394_, v_goal_395_, v_e_396_, v_eStx_397_, v_md_boxed_409_, v_as_399_, v_sz_boxed_410_, v_i_boxed_411_, v_b_402_, v___y_403_, v___y_404_, v___y_405_, v___y_406_, v___y_407_);
lean_dec(v___y_407_);
lean_dec_ref(v___y_406_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v_as_399_);
lean_dec_ref(v_a_394_);
return v_res_412_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_applyExpr___closed__1(void){
_start:
{
lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_414_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_applyExpr___closed__0));
v___x_415_ = l_Lean_stringToMessageData(v___x_414_);
return v___x_415_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_applyExpr___closed__3(void){
_start:
{
lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_417_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_applyExpr___closed__2));
v___x_418_ = l_Lean_stringToMessageData(v___x_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr(lean_object* v_goal_419_, lean_object* v_e_420_, lean_object* v_eStx_421_, lean_object* v_patSubsts_x3f_422_, uint8_t v_md_423_, lean_object* v_a_424_, lean_object* v_a_425_, lean_object* v_a_426_, lean_object* v_a_427_, lean_object* v_a_428_){
_start:
{
if (lean_obj_tag(v_patSubsts_x3f_422_) == 1)
{
lean_object* v_val_430_; lean_object* v___x_431_; 
v_val_430_ = lean_ctor_get(v_patSubsts_x3f_422_, 0);
v___x_431_ = l_Lean_Meta_saveState___redArg(v_a_426_, v_a_428_);
if (lean_obj_tag(v___x_431_) == 0)
{
lean_object* v_a_432_; lean_object* v___x_433_; lean_object* v_rapps_434_; size_t v_sz_435_; size_t v___x_436_; lean_object* v___x_437_; 
v_a_432_ = lean_ctor_get(v___x_431_, 0);
lean_inc(v_a_432_);
lean_dec_ref_known(v___x_431_, 1);
v___x_433_ = lean_array_get_size(v_val_430_);
v_rapps_434_ = lean_mk_empty_array_with_capacity(v___x_433_);
v_sz_435_ = lean_array_size(v_val_430_);
v___x_436_ = ((size_t)0ULL);
lean_inc_ref(v_e_420_);
v___x_437_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyExpr_spec__0(v_a_432_, v_goal_419_, v_e_420_, v_eStx_421_, v_md_423_, v_val_430_, v_sz_435_, v___x_436_, v_rapps_434_, v_a_424_, v_a_425_, v_a_426_, v_a_427_, v_a_428_);
lean_dec(v_a_432_);
if (lean_obj_tag(v___x_437_) == 0)
{
lean_object* v_a_438_; lean_object* v___x_440_; uint8_t v_isShared_441_; uint8_t v_isSharedCheck_462_; 
v_a_438_ = lean_ctor_get(v___x_437_, 0);
v_isSharedCheck_462_ = !lean_is_exclusive(v___x_437_);
if (v_isSharedCheck_462_ == 0)
{
v___x_440_ = v___x_437_;
v_isShared_441_ = v_isSharedCheck_462_;
goto v_resetjp_439_;
}
else
{
lean_inc(v_a_438_);
lean_dec(v___x_437_);
v___x_440_ = lean_box(0);
v_isShared_441_ = v_isSharedCheck_462_;
goto v_resetjp_439_;
}
v_resetjp_439_:
{
lean_object* v___x_442_; lean_object* v___x_443_; uint8_t v___x_444_; 
v___x_442_ = lean_array_get_size(v_a_438_);
v___x_443_ = lean_unsigned_to_nat(0u);
v___x_444_ = lean_nat_dec_eq(v___x_442_, v___x_443_);
if (v___x_444_ == 0)
{
lean_object* v___x_446_; 
lean_dec_ref(v_e_420_);
if (v_isShared_441_ == 0)
{
v___x_446_ = v___x_440_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_447_; 
v_reuseFailAlloc_447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_447_, 0, v_a_438_);
v___x_446_ = v_reuseFailAlloc_447_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
return v___x_446_;
}
}
else
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v_a_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_461_; 
lean_del_object(v___x_440_);
lean_dec(v_a_438_);
v___x_448_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_applyExpr___closed__1, &lp_aesop_Aesop_RuleTac_applyExpr___closed__1_once, _init_lp_aesop_Aesop_RuleTac_applyExpr___closed__1);
v___x_449_ = l_Lean_MessageData_ofExpr(v_e_420_);
v___x_450_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_450_, 0, v___x_448_);
lean_ctor_set(v___x_450_, 1, v___x_449_);
v___x_451_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_applyExpr___closed__3, &lp_aesop_Aesop_RuleTac_applyExpr___closed__3_once, _init_lp_aesop_Aesop_RuleTac_applyExpr___closed__3);
v___x_452_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_452_, 0, v___x_450_);
lean_ctor_set(v___x_452_, 1, v___x_451_);
v___x_453_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg(v___x_452_, v_a_425_, v_a_426_, v_a_427_, v_a_428_);
v_a_454_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_461_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_461_ == 0)
{
v___x_456_ = v___x_453_;
v_isShared_457_ = v_isSharedCheck_461_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_a_454_);
lean_dec(v___x_453_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_461_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v___x_459_; 
if (v_isShared_457_ == 0)
{
v___x_459_ = v___x_456_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v_a_454_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
}
}
else
{
lean_object* v_a_463_; lean_object* v___x_465_; uint8_t v_isShared_466_; uint8_t v_isSharedCheck_470_; 
lean_dec_ref(v_e_420_);
v_a_463_ = lean_ctor_get(v___x_437_, 0);
v_isSharedCheck_470_ = !lean_is_exclusive(v___x_437_);
if (v_isSharedCheck_470_ == 0)
{
v___x_465_ = v___x_437_;
v_isShared_466_ = v_isSharedCheck_470_;
goto v_resetjp_464_;
}
else
{
lean_inc(v_a_463_);
lean_dec(v___x_437_);
v___x_465_ = lean_box(0);
v_isShared_466_ = v_isSharedCheck_470_;
goto v_resetjp_464_;
}
v_resetjp_464_:
{
lean_object* v___x_468_; 
if (v_isShared_466_ == 0)
{
v___x_468_ = v___x_465_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v_a_463_);
v___x_468_ = v_reuseFailAlloc_469_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
return v___x_468_;
}
}
}
}
else
{
lean_object* v_a_471_; lean_object* v___x_473_; uint8_t v_isShared_474_; uint8_t v_isSharedCheck_478_; 
lean_dec(v_eStx_421_);
lean_dec_ref(v_e_420_);
lean_dec(v_goal_419_);
v_a_471_ = lean_ctor_get(v___x_431_, 0);
v_isSharedCheck_478_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_478_ == 0)
{
v___x_473_ = v___x_431_;
v_isShared_474_ = v_isSharedCheck_478_;
goto v_resetjp_472_;
}
else
{
lean_inc(v_a_471_);
lean_dec(v___x_431_);
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
else
{
lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_479_ = lean_box(0);
v___x_480_ = lp_aesop_Aesop_RuleTac_applyExpr_x27(v_goal_419_, v_e_420_, v_eStx_421_, v___x_479_, v_md_423_, v_a_424_, v_a_425_, v_a_426_, v_a_427_, v_a_428_);
if (lean_obj_tag(v___x_480_) == 0)
{
lean_object* v_a_481_; lean_object* v___x_483_; uint8_t v_isShared_484_; uint8_t v_isSharedCheck_491_; 
v_a_481_ = lean_ctor_get(v___x_480_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_491_ == 0)
{
v___x_483_ = v___x_480_;
v_isShared_484_ = v_isSharedCheck_491_;
goto v_resetjp_482_;
}
else
{
lean_inc(v_a_481_);
lean_dec(v___x_480_);
v___x_483_ = lean_box(0);
v_isShared_484_ = v_isSharedCheck_491_;
goto v_resetjp_482_;
}
v_resetjp_482_:
{
lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_489_; 
v___x_485_ = lean_unsigned_to_nat(1u);
v___x_486_ = lean_mk_empty_array_with_capacity(v___x_485_);
v___x_487_ = lean_array_push(v___x_486_, v_a_481_);
if (v_isShared_484_ == 0)
{
lean_ctor_set(v___x_483_, 0, v___x_487_);
v___x_489_ = v___x_483_;
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
v_a_492_ = lean_ctor_get(v___x_480_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_499_ == 0)
{
v___x_494_ = v___x_480_;
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_a_492_);
lean_dec(v___x_480_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyExpr___boxed(lean_object* v_goal_500_, lean_object* v_e_501_, lean_object* v_eStx_502_, lean_object* v_patSubsts_x3f_503_, lean_object* v_md_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_, lean_object* v_a_509_, lean_object* v_a_510_){
_start:
{
uint8_t v_md_boxed_511_; lean_object* v_res_512_; 
v_md_boxed_511_ = lean_unbox(v_md_504_);
v_res_512_ = lp_aesop_Aesop_RuleTac_applyExpr(v_goal_500_, v_e_501_, v_eStx_502_, v_patSubsts_x3f_503_, v_md_boxed_511_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_);
lean_dec(v_a_509_);
lean_dec_ref(v_a_508_);
lean_dec(v_a_507_);
lean_dec_ref(v_a_506_);
lean_dec(v_a_505_);
lean_dec(v_patSubsts_x3f_503_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyConst(lean_object* v_decl_513_, uint8_t v_md_514_, lean_object* v_input_515_, lean_object* v_a_516_, lean_object* v_a_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_){
_start:
{
lean_object* v___x_522_; 
lean_inc(v_decl_513_);
v___x_522_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_decl_513_, v_a_517_, v_a_518_, v_a_519_, v_a_520_);
if (lean_obj_tag(v___x_522_) == 0)
{
lean_object* v_a_523_; lean_object* v_goal_524_; lean_object* v_patternSubsts_x3f_525_; lean_object* v___x_526_; lean_object* v___x_527_; 
v_a_523_ = lean_ctor_get(v___x_522_, 0);
lean_inc(v_a_523_);
lean_dec_ref_known(v___x_522_, 1);
v_goal_524_ = lean_ctor_get(v_input_515_, 0);
lean_inc(v_goal_524_);
v_patternSubsts_x3f_525_ = lean_ctor_get(v_input_515_, 3);
lean_inc(v_patternSubsts_x3f_525_);
lean_dec_ref(v_input_515_);
v___x_526_ = l_Lean_mkIdent(v_decl_513_);
v___x_527_ = lp_aesop_Aesop_RuleTac_applyExpr(v_goal_524_, v_a_523_, v___x_526_, v_patternSubsts_x3f_525_, v_md_514_, v_a_516_, v_a_517_, v_a_518_, v_a_519_, v_a_520_);
lean_dec(v_patternSubsts_x3f_525_);
return v___x_527_;
}
else
{
lean_object* v_a_528_; lean_object* v___x_530_; uint8_t v_isShared_531_; uint8_t v_isSharedCheck_535_; 
lean_dec_ref(v_input_515_);
lean_dec(v_decl_513_);
v_a_528_ = lean_ctor_get(v___x_522_, 0);
v_isSharedCheck_535_ = !lean_is_exclusive(v___x_522_);
if (v_isSharedCheck_535_ == 0)
{
v___x_530_ = v___x_522_;
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
else
{
lean_inc(v_a_528_);
lean_dec(v___x_522_);
v___x_530_ = lean_box(0);
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
v_resetjp_529_:
{
lean_object* v___x_533_; 
if (v_isShared_531_ == 0)
{
v___x_533_ = v___x_530_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v_a_528_);
v___x_533_ = v_reuseFailAlloc_534_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
return v___x_533_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyConst___boxed(lean_object* v_decl_536_, lean_object* v_md_537_, lean_object* v_input_538_, lean_object* v_a_539_, lean_object* v_a_540_, lean_object* v_a_541_, lean_object* v_a_542_, lean_object* v_a_543_, lean_object* v_a_544_){
_start:
{
uint8_t v_md_boxed_545_; lean_object* v_res_546_; 
v_md_boxed_545_ = lean_unbox(v_md_537_);
v_res_546_ = lp_aesop_Aesop_RuleTac_applyConst(v_decl_536_, v_md_boxed_545_, v_input_538_, v_a_539_, v_a_540_, v_a_541_, v_a_542_, v_a_543_);
lean_dec(v_a_543_);
lean_dec_ref(v_a_542_);
lean_dec(v_a_541_);
lean_dec_ref(v_a_540_);
lean_dec(v_a_539_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg___lam__0(lean_object* v_x_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_){
_start:
{
lean_object* v___x_554_; 
lean_inc(v___y_548_);
v___x_554_ = lean_apply_6(v_x_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_, v___y_552_, lean_box(0));
return v___x_554_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg___lam__0___boxed(lean_object* v_x_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_){
_start:
{
lean_object* v_res_562_; 
v_res_562_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg___lam__0(v_x_555_, v___y_556_, v___y_557_, v___y_558_, v___y_559_, v___y_560_);
lean_dec(v___y_556_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg(lean_object* v_mvarId_563_, lean_object* v_x_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v___f_571_; lean_object* v___x_572_; 
lean_inc(v___y_565_);
v___f_571_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_571_, 0, v_x_564_);
lean_closure_set(v___f_571_, 1, v___y_565_);
v___x_572_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_563_, v___f_571_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
if (lean_obj_tag(v___x_572_) == 0)
{
return v___x_572_;
}
else
{
lean_object* v_a_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_580_; 
v_a_573_ = lean_ctor_get(v___x_572_, 0);
v_isSharedCheck_580_ = !lean_is_exclusive(v___x_572_);
if (v_isSharedCheck_580_ == 0)
{
v___x_575_ = v___x_572_;
v_isShared_576_ = v_isSharedCheck_580_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_a_573_);
lean_dec(v___x_572_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_580_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
lean_object* v___x_578_; 
if (v_isShared_576_ == 0)
{
v___x_578_ = v___x_575_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v_a_573_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
return v___x_578_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg___boxed(lean_object* v_mvarId_581_, lean_object* v_x_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_){
_start:
{
lean_object* v_res_589_; 
v_res_589_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg(v_mvarId_581_, v_x_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_);
lean_dec(v___y_587_);
lean_dec_ref(v___y_586_);
lean_dec(v___y_585_);
lean_dec_ref(v___y_584_);
lean_dec(v___y_583_);
return v_res_589_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0(lean_object* v_00_u03b1_590_, lean_object* v_mvarId_591_, lean_object* v_x_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg(v_mvarId_591_, v_x_592_, v___y_593_, v___y_594_, v___y_595_, v___y_596_, v___y_597_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___boxed(lean_object* v_00_u03b1_600_, lean_object* v_mvarId_601_, lean_object* v_x_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0(v_00_u03b1_600_, v_mvarId_601_, v_x_602_, v___y_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_);
lean_dec(v___y_607_);
lean_dec_ref(v___y_606_);
lean_dec(v___y_605_);
lean_dec_ref(v___y_604_);
lean_dec(v___y_603_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyTerm___lam__0(lean_object* v_goal_610_, lean_object* v_stx_611_, lean_object* v_patternSubsts_x3f_612_, uint8_t v_md_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_){
_start:
{
lean_object* v___x_620_; 
lean_inc(v_stx_611_);
lean_inc(v_goal_610_);
v___x_620_ = lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(v_goal_610_, v_stx_611_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
if (lean_obj_tag(v___x_620_) == 0)
{
lean_object* v_a_621_; lean_object* v___x_622_; 
v_a_621_ = lean_ctor_get(v___x_620_, 0);
lean_inc(v_a_621_);
lean_dec_ref_known(v___x_620_, 1);
v___x_622_ = lp_aesop_Aesop_RuleTac_applyExpr(v_goal_610_, v_a_621_, v_stx_611_, v_patternSubsts_x3f_612_, v_md_613_, v___y_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
return v___x_622_;
}
else
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_630_; 
lean_dec(v_stx_611_);
lean_dec(v_goal_610_);
v_a_623_ = lean_ctor_get(v___x_620_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_620_);
if (v_isSharedCheck_630_ == 0)
{
v___x_625_ = v___x_620_;
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_620_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_a_623_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyTerm___lam__0___boxed(lean_object* v_goal_631_, lean_object* v_stx_632_, lean_object* v_patternSubsts_x3f_633_, lean_object* v_md_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_){
_start:
{
uint8_t v_md_boxed_641_; lean_object* v_res_642_; 
v_md_boxed_641_ = lean_unbox(v_md_634_);
v_res_642_ = lp_aesop_Aesop_RuleTac_applyTerm___lam__0(v_goal_631_, v_stx_632_, v_patternSubsts_x3f_633_, v_md_boxed_641_, v___y_635_, v___y_636_, v___y_637_, v___y_638_, v___y_639_);
lean_dec(v___y_639_);
lean_dec_ref(v___y_638_);
lean_dec(v___y_637_);
lean_dec_ref(v___y_636_);
lean_dec(v___y_635_);
lean_dec(v_patternSubsts_x3f_633_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyTerm(lean_object* v_stx_643_, uint8_t v_md_644_, lean_object* v_input_645_, lean_object* v_a_646_, lean_object* v_a_647_, lean_object* v_a_648_, lean_object* v_a_649_, lean_object* v_a_650_){
_start:
{
lean_object* v_goal_652_; lean_object* v_patternSubsts_x3f_653_; lean_object* v___x_654_; lean_object* v___f_655_; lean_object* v___x_656_; 
v_goal_652_ = lean_ctor_get(v_input_645_, 0);
lean_inc_n(v_goal_652_, 2);
v_patternSubsts_x3f_653_ = lean_ctor_get(v_input_645_, 3);
lean_inc(v_patternSubsts_x3f_653_);
lean_dec_ref(v_input_645_);
v___x_654_ = lean_box(v_md_644_);
v___f_655_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_applyTerm___lam__0___boxed), 10, 4);
lean_closure_set(v___f_655_, 0, v_goal_652_);
lean_closure_set(v___f_655_, 1, v_stx_643_);
lean_closure_set(v___f_655_, 2, v_patternSubsts_x3f_653_);
lean_closure_set(v___f_655_, 3, v___x_654_);
v___x_656_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyTerm_spec__0___redArg(v_goal_652_, v___f_655_, v_a_646_, v_a_647_, v_a_648_, v_a_649_, v_a_650_);
return v___x_656_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyTerm___boxed(lean_object* v_stx_657_, lean_object* v_md_658_, lean_object* v_input_659_, lean_object* v_a_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_){
_start:
{
uint8_t v_md_boxed_666_; lean_object* v_res_667_; 
v_md_boxed_666_ = lean_unbox(v_md_658_);
v_res_667_ = lp_aesop_Aesop_RuleTac_applyTerm(v_stx_657_, v_md_boxed_666_, v_input_659_, v_a_660_, v_a_661_, v_a_662_, v_a_663_, v_a_664_);
lean_dec(v_a_664_);
lean_dec_ref(v_a_663_);
lean_dec(v_a_662_);
lean_dec_ref(v_a_661_);
lean_dec(v_a_660_);
return v_res_667_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_apply(lean_object* v_t_668_, uint8_t v_md_669_, lean_object* v_a_670_, lean_object* v_a_671_, lean_object* v_a_672_, lean_object* v_a_673_, lean_object* v_a_674_, lean_object* v_a_675_){
_start:
{
if (lean_obj_tag(v_t_668_) == 0)
{
lean_object* v_decl_677_; lean_object* v___x_678_; 
v_decl_677_ = lean_ctor_get(v_t_668_, 0);
lean_inc(v_decl_677_);
lean_dec_ref_known(v_t_668_, 1);
v___x_678_ = lp_aesop_Aesop_RuleTac_applyConst(v_decl_677_, v_md_669_, v_a_670_, v_a_671_, v_a_672_, v_a_673_, v_a_674_, v_a_675_);
return v___x_678_;
}
else
{
lean_object* v_term_679_; lean_object* v___x_680_; 
v_term_679_ = lean_ctor_get(v_t_668_, 0);
lean_inc(v_term_679_);
lean_dec_ref_known(v_t_668_, 1);
v___x_680_ = lp_aesop_Aesop_RuleTac_applyTerm(v_term_679_, v_md_669_, v_a_670_, v_a_671_, v_a_672_, v_a_673_, v_a_674_, v_a_675_);
return v___x_680_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_apply___boxed(lean_object* v_t_681_, lean_object* v_md_682_, lean_object* v_a_683_, lean_object* v_a_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_){
_start:
{
uint8_t v_md_boxed_690_; lean_object* v_res_691_; 
v_md_boxed_690_ = lean_unbox(v_md_682_);
v_res_691_ = lp_aesop_Aesop_RuleTac_apply(v_t_681_, v_md_boxed_690_, v_a_683_, v_a_684_, v_a_685_, v_a_686_, v_a_687_, v_a_688_);
lean_dec(v_a_688_);
lean_dec_ref(v_a_687_);
lean_dec(v_a_686_);
lean_dec_ref(v_a_685_);
lean_dec(v_a_684_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0(lean_object* v_a_694_, lean_object* v_input_695_, uint8_t v_md_696_, lean_object* v_as_697_, size_t v_i_698_, size_t v_stop_699_, lean_object* v_b_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_){
_start:
{
lean_object* v_a_708_; lean_object* v_a_713_; lean_object* v_a_717_; lean_object* v___y_729_; uint8_t v___y_730_; lean_object* v_a_750_; uint8_t v___x_753_; 
v___x_753_ = lean_usize_dec_eq(v_i_698_, v_stop_699_);
if (v___x_753_ == 0)
{
lean_object* v___x_754_; lean_object* v___x_755_; 
v___x_754_ = lean_array_uget_borrowed(v_as_697_, v_i_698_);
lean_inc(v___x_754_);
v___x_755_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v___x_754_, v___y_702_, v___y_703_, v___y_704_, v___y_705_);
if (lean_obj_tag(v___x_755_) == 0)
{
lean_object* v_a_756_; lean_object* v___x_758_; uint8_t v_isShared_759_; uint8_t v_isSharedCheck_776_; 
v_a_756_ = lean_ctor_get(v___x_755_, 0);
v_isSharedCheck_776_ = !lean_is_exclusive(v___x_755_);
if (v_isSharedCheck_776_ == 0)
{
v___x_758_ = v___x_755_;
v_isShared_759_ = v_isSharedCheck_776_;
goto v_resetjp_757_;
}
else
{
lean_inc(v_a_756_);
lean_dec(v___x_755_);
v___x_758_ = lean_box(0);
v_isShared_759_ = v_isSharedCheck_776_;
goto v_resetjp_757_;
}
v_resetjp_757_:
{
lean_object* v_goal_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; 
v_goal_760_ = lean_ctor_get(v_input_695_, 0);
lean_inc(v___x_754_);
v___x_761_ = l_Lean_mkIdent(v___x_754_);
v___x_762_ = lean_box(0);
lean_inc(v_goal_760_);
v___x_763_ = lp_aesop_Aesop_RuleTac_applyExpr_x27(v_goal_760_, v_a_756_, v___x_761_, v___x_762_, v_md_696_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_);
if (lean_obj_tag(v___x_763_) == 0)
{
lean_object* v_a_764_; lean_object* v___x_766_; uint8_t v_isShared_767_; uint8_t v_isSharedCheck_774_; 
v_a_764_ = lean_ctor_get(v___x_763_, 0);
v_isSharedCheck_774_ = !lean_is_exclusive(v___x_763_);
if (v_isSharedCheck_774_ == 0)
{
v___x_766_ = v___x_763_;
v_isShared_767_ = v_isSharedCheck_774_;
goto v_resetjp_765_;
}
else
{
lean_inc(v_a_764_);
lean_dec(v___x_763_);
v___x_766_ = lean_box(0);
v_isShared_767_ = v_isSharedCheck_774_;
goto v_resetjp_765_;
}
v_resetjp_765_:
{
lean_object* v___x_769_; 
if (v_isShared_767_ == 0)
{
lean_ctor_set_tag(v___x_766_, 1);
v___x_769_ = v___x_766_;
goto v_reusejp_768_;
}
else
{
lean_object* v_reuseFailAlloc_773_; 
v_reuseFailAlloc_773_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_773_, 0, v_a_764_);
v___x_769_ = v_reuseFailAlloc_773_;
goto v_reusejp_768_;
}
v_reusejp_768_:
{
lean_object* v___x_771_; 
if (v_isShared_759_ == 0)
{
lean_ctor_set_tag(v___x_758_, 1);
lean_ctor_set(v___x_758_, 0, v___x_769_);
v___x_771_ = v___x_758_;
goto v_reusejp_770_;
}
else
{
lean_object* v_reuseFailAlloc_772_; 
v_reuseFailAlloc_772_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_772_, 0, v___x_769_);
v___x_771_ = v_reuseFailAlloc_772_;
goto v_reusejp_770_;
}
v_reusejp_770_:
{
v_a_717_ = v___x_771_;
goto v___jp_716_;
}
}
}
}
else
{
lean_object* v_a_775_; 
lean_del_object(v___x_758_);
v_a_775_ = lean_ctor_get(v___x_763_, 0);
lean_inc(v_a_775_);
lean_dec_ref_known(v___x_763_, 1);
v_a_750_ = v_a_775_;
goto v___jp_749_;
}
}
}
else
{
lean_object* v_a_777_; 
v_a_777_ = lean_ctor_get(v___x_755_, 0);
lean_inc(v_a_777_);
lean_dec_ref_known(v___x_755_, 1);
v_a_750_ = v_a_777_;
goto v___jp_749_;
}
}
else
{
lean_object* v___x_778_; 
lean_dec_ref(v_input_695_);
v___x_778_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_778_, 0, v_b_700_);
return v___x_778_;
}
v___jp_707_:
{
size_t v___x_709_; size_t v___x_710_; 
v___x_709_ = ((size_t)1ULL);
v___x_710_ = lean_usize_add(v_i_698_, v___x_709_);
v_i_698_ = v___x_710_;
v_b_700_ = v_a_708_;
goto _start;
}
v___jp_712_:
{
if (lean_obj_tag(v_a_713_) == 0)
{
v_a_708_ = v_b_700_;
goto v___jp_707_;
}
else
{
lean_object* v_val_714_; lean_object* v___x_715_; 
v_val_714_ = lean_ctor_get(v_a_713_, 0);
lean_inc(v_val_714_);
lean_dec_ref_known(v_a_713_, 1);
v___x_715_ = lean_array_push(v_b_700_, v_val_714_);
v_a_708_ = v___x_715_;
goto v___jp_707_;
}
}
v___jp_716_:
{
lean_object* v___x_718_; 
v___x_718_ = l_Lean_Meta_SavedState_restore___redArg(v_a_694_, v___y_703_, v___y_705_);
if (lean_obj_tag(v___x_718_) == 0)
{
lean_object* v_a_719_; 
lean_dec_ref_known(v___x_718_, 1);
v_a_719_ = lean_ctor_get(v_a_717_, 0);
lean_inc(v_a_719_);
lean_dec_ref(v_a_717_);
v_a_713_ = v_a_719_;
goto v___jp_712_;
}
else
{
lean_object* v_a_720_; lean_object* v___x_722_; uint8_t v_isShared_723_; uint8_t v_isSharedCheck_727_; 
lean_dec_ref(v_a_717_);
lean_dec_ref(v_b_700_);
lean_dec_ref(v_input_695_);
v_a_720_ = lean_ctor_get(v___x_718_, 0);
v_isSharedCheck_727_ = !lean_is_exclusive(v___x_718_);
if (v_isSharedCheck_727_ == 0)
{
v___x_722_ = v___x_718_;
v_isShared_723_ = v_isSharedCheck_727_;
goto v_resetjp_721_;
}
else
{
lean_inc(v_a_720_);
lean_dec(v___x_718_);
v___x_722_ = lean_box(0);
v_isShared_723_ = v_isSharedCheck_727_;
goto v_resetjp_721_;
}
v_resetjp_721_:
{
lean_object* v___x_725_; 
if (v_isShared_723_ == 0)
{
v___x_725_ = v___x_722_;
goto v_reusejp_724_;
}
else
{
lean_object* v_reuseFailAlloc_726_; 
v_reuseFailAlloc_726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_726_, 0, v_a_720_);
v___x_725_ = v_reuseFailAlloc_726_;
goto v_reusejp_724_;
}
v_reusejp_724_:
{
return v___x_725_;
}
}
}
}
v___jp_728_:
{
if (v___y_730_ == 0)
{
lean_object* v___x_731_; 
lean_dec_ref(v___y_729_);
v___x_731_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0___closed__0));
v_a_717_ = v___x_731_;
goto v___jp_716_;
}
else
{
lean_object* v___x_732_; 
lean_dec_ref(v_b_700_);
lean_dec_ref(v_input_695_);
v___x_732_ = l_Lean_Meta_SavedState_restore___redArg(v_a_694_, v___y_703_, v___y_705_);
if (lean_obj_tag(v___x_732_) == 0)
{
lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_739_; 
v_isSharedCheck_739_ = !lean_is_exclusive(v___x_732_);
if (v_isSharedCheck_739_ == 0)
{
lean_object* v_unused_740_; 
v_unused_740_ = lean_ctor_get(v___x_732_, 0);
lean_dec(v_unused_740_);
v___x_734_ = v___x_732_;
v_isShared_735_ = v_isSharedCheck_739_;
goto v_resetjp_733_;
}
else
{
lean_dec(v___x_732_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_739_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_737_; 
if (v_isShared_735_ == 0)
{
lean_ctor_set_tag(v___x_734_, 1);
lean_ctor_set(v___x_734_, 0, v___y_729_);
v___x_737_ = v___x_734_;
goto v_reusejp_736_;
}
else
{
lean_object* v_reuseFailAlloc_738_; 
v_reuseFailAlloc_738_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_738_, 0, v___y_729_);
v___x_737_ = v_reuseFailAlloc_738_;
goto v_reusejp_736_;
}
v_reusejp_736_:
{
return v___x_737_;
}
}
}
else
{
lean_object* v_a_741_; lean_object* v___x_743_; uint8_t v_isShared_744_; uint8_t v_isSharedCheck_748_; 
lean_dec_ref(v___y_729_);
v_a_741_ = lean_ctor_get(v___x_732_, 0);
v_isSharedCheck_748_ = !lean_is_exclusive(v___x_732_);
if (v_isSharedCheck_748_ == 0)
{
v___x_743_ = v___x_732_;
v_isShared_744_ = v_isSharedCheck_748_;
goto v_resetjp_742_;
}
else
{
lean_inc(v_a_741_);
lean_dec(v___x_732_);
v___x_743_ = lean_box(0);
v_isShared_744_ = v_isSharedCheck_748_;
goto v_resetjp_742_;
}
v_resetjp_742_:
{
lean_object* v___x_746_; 
if (v_isShared_744_ == 0)
{
v___x_746_ = v___x_743_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_747_; 
v_reuseFailAlloc_747_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_747_, 0, v_a_741_);
v___x_746_ = v_reuseFailAlloc_747_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
return v___x_746_;
}
}
}
}
}
v___jp_749_:
{
uint8_t v___x_751_; 
v___x_751_ = l_Lean_Exception_isInterrupt(v_a_750_);
if (v___x_751_ == 0)
{
uint8_t v___x_752_; 
lean_inc_ref(v_a_750_);
v___x_752_ = l_Lean_Exception_isRuntime(v_a_750_);
v___y_729_ = v_a_750_;
v___y_730_ = v___x_752_;
goto v___jp_728_;
}
else
{
v___y_729_ = v_a_750_;
v___y_730_ = v___x_751_;
goto v___jp_728_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0___boxed(lean_object* v_a_779_, lean_object* v_input_780_, lean_object* v_md_781_, lean_object* v_as_782_, lean_object* v_i_783_, lean_object* v_stop_784_, lean_object* v_b_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_){
_start:
{
uint8_t v_md_boxed_792_; size_t v_i_boxed_793_; size_t v_stop_boxed_794_; lean_object* v_res_795_; 
v_md_boxed_792_ = lean_unbox(v_md_781_);
v_i_boxed_793_ = lean_unbox_usize(v_i_783_);
lean_dec(v_i_783_);
v_stop_boxed_794_ = lean_unbox_usize(v_stop_784_);
lean_dec(v_stop_784_);
v_res_795_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0(v_a_779_, v_input_780_, v_md_boxed_792_, v_as_782_, v_i_boxed_793_, v_stop_boxed_794_, v_b_785_, v___y_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_);
lean_dec(v___y_790_);
lean_dec_ref(v___y_789_);
lean_dec(v___y_788_);
lean_dec_ref(v___y_787_);
lean_dec(v___y_786_);
lean_dec_ref(v_as_782_);
lean_dec_ref(v_a_779_);
return v_res_795_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0(lean_object* v_a_798_, lean_object* v_input_799_, uint8_t v_md_800_, lean_object* v_as_801_, lean_object* v_start_802_, lean_object* v_stop_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_){
_start:
{
lean_object* v___x_810_; uint8_t v___x_811_; 
v___x_810_ = ((lean_object*)(lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0___closed__0));
v___x_811_ = lean_nat_dec_lt(v_start_802_, v_stop_803_);
if (v___x_811_ == 0)
{
lean_object* v___x_812_; 
lean_dec_ref(v_input_799_);
v___x_812_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_812_, 0, v___x_810_);
return v___x_812_;
}
else
{
lean_object* v___x_813_; uint8_t v___x_814_; 
v___x_813_ = lean_array_get_size(v_as_801_);
v___x_814_ = lean_nat_dec_le(v_stop_803_, v___x_813_);
if (v___x_814_ == 0)
{
uint8_t v___x_815_; 
v___x_815_ = lean_nat_dec_lt(v_start_802_, v___x_813_);
if (v___x_815_ == 0)
{
lean_object* v___x_816_; 
lean_dec_ref(v_input_799_);
v___x_816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_816_, 0, v___x_810_);
return v___x_816_;
}
else
{
size_t v___x_817_; size_t v___x_818_; lean_object* v___x_819_; 
v___x_817_ = lean_usize_of_nat(v_start_802_);
v___x_818_ = lean_usize_of_nat(v___x_813_);
v___x_819_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0(v_a_798_, v_input_799_, v_md_800_, v_as_801_, v___x_817_, v___x_818_, v___x_810_, v___y_804_, v___y_805_, v___y_806_, v___y_807_, v___y_808_);
return v___x_819_;
}
}
else
{
size_t v___x_820_; size_t v___x_821_; lean_object* v___x_822_; 
v___x_820_ = lean_usize_of_nat(v_start_802_);
v___x_821_ = lean_usize_of_nat(v_stop_803_);
v___x_822_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0_spec__0(v_a_798_, v_input_799_, v_md_800_, v_as_801_, v___x_820_, v___x_821_, v___x_810_, v___y_804_, v___y_805_, v___y_806_, v___y_807_, v___y_808_);
return v___x_822_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0___boxed(lean_object* v_a_823_, lean_object* v_input_824_, lean_object* v_md_825_, lean_object* v_as_826_, lean_object* v_start_827_, lean_object* v_stop_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_){
_start:
{
uint8_t v_md_boxed_835_; lean_object* v_res_836_; 
v_md_boxed_835_ = lean_unbox(v_md_825_);
v_res_836_ = lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0(v_a_823_, v_input_824_, v_md_boxed_835_, v_as_826_, v_start_827_, v_stop_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_, v___y_833_);
lean_dec(v___y_833_);
lean_dec_ref(v___y_832_);
lean_dec(v___y_831_);
lean_dec_ref(v___y_830_);
lean_dec(v___y_829_);
lean_dec(v_stop_828_);
lean_dec(v_start_827_);
lean_dec_ref(v_as_826_);
lean_dec_ref(v_a_823_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_applyConsts_spec__1(lean_object* v_a_837_, lean_object* v_a_838_){
_start:
{
if (lean_obj_tag(v_a_837_) == 0)
{
lean_object* v___x_839_; 
v___x_839_ = l_List_reverse___redArg(v_a_838_);
return v___x_839_;
}
else
{
lean_object* v_head_840_; lean_object* v_tail_841_; lean_object* v___x_843_; uint8_t v_isShared_844_; uint8_t v_isSharedCheck_850_; 
v_head_840_ = lean_ctor_get(v_a_837_, 0);
v_tail_841_ = lean_ctor_get(v_a_837_, 1);
v_isSharedCheck_850_ = !lean_is_exclusive(v_a_837_);
if (v_isSharedCheck_850_ == 0)
{
v___x_843_ = v_a_837_;
v_isShared_844_ = v_isSharedCheck_850_;
goto v_resetjp_842_;
}
else
{
lean_inc(v_tail_841_);
lean_inc(v_head_840_);
lean_dec(v_a_837_);
v___x_843_ = lean_box(0);
v_isShared_844_ = v_isSharedCheck_850_;
goto v_resetjp_842_;
}
v_resetjp_842_:
{
lean_object* v___x_845_; lean_object* v___x_847_; 
v___x_845_ = l_Lean_MessageData_ofName(v_head_840_);
if (v_isShared_844_ == 0)
{
lean_ctor_set(v___x_843_, 1, v_a_838_);
lean_ctor_set(v___x_843_, 0, v___x_845_);
v___x_847_ = v___x_843_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_849_; 
v_reuseFailAlloc_849_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_849_, 0, v___x_845_);
lean_ctor_set(v_reuseFailAlloc_849_, 1, v_a_838_);
v___x_847_ = v_reuseFailAlloc_849_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
v_a_837_ = v_tail_841_;
v_a_838_ = v___x_847_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_applyConsts___closed__1(void){
_start:
{
lean_object* v___x_852_; lean_object* v___x_853_; 
v___x_852_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_applyConsts___closed__0));
v___x_853_ = l_Lean_stringToMessageData(v___x_852_);
return v___x_853_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyConsts(lean_object* v_decls_854_, uint8_t v_md_855_, lean_object* v_input_856_, lean_object* v_a_857_, lean_object* v_a_858_, lean_object* v_a_859_, lean_object* v_a_860_, lean_object* v_a_861_){
_start:
{
lean_object* v___x_863_; 
v___x_863_ = l_Lean_Meta_saveState___redArg(v_a_859_, v_a_861_);
if (lean_obj_tag(v___x_863_) == 0)
{
lean_object* v_a_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; 
v_a_864_ = lean_ctor_get(v___x_863_, 0);
lean_inc(v_a_864_);
lean_dec_ref_known(v___x_863_, 1);
v___x_865_ = lean_unsigned_to_nat(0u);
v___x_866_ = lean_array_get_size(v_decls_854_);
v___x_867_ = lp_aesop_Array_filterMapM___at___00Aesop_RuleTac_applyConsts_spec__0(v_a_864_, v_input_856_, v_md_855_, v_decls_854_, v___x_865_, v___x_866_, v_a_857_, v_a_858_, v_a_859_, v_a_860_, v_a_861_);
lean_dec(v_a_864_);
if (lean_obj_tag(v___x_867_) == 0)
{
lean_object* v_a_868_; lean_object* v___x_870_; uint8_t v_isShared_871_; uint8_t v_isSharedCheck_892_; 
v_a_868_ = lean_ctor_get(v___x_867_, 0);
v_isSharedCheck_892_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_892_ == 0)
{
v___x_870_ = v___x_867_;
v_isShared_871_ = v_isSharedCheck_892_;
goto v_resetjp_869_;
}
else
{
lean_inc(v_a_868_);
lean_dec(v___x_867_);
v___x_870_ = lean_box(0);
v_isShared_871_ = v_isSharedCheck_892_;
goto v_resetjp_869_;
}
v_resetjp_869_:
{
lean_object* v___x_872_; uint8_t v___x_873_; 
v___x_872_ = lean_array_get_size(v_a_868_);
v___x_873_ = lean_nat_dec_eq(v___x_872_, v___x_865_);
if (v___x_873_ == 0)
{
lean_object* v___x_875_; 
lean_dec_ref(v_decls_854_);
if (v_isShared_871_ == 0)
{
v___x_875_ = v___x_870_;
goto v_reusejp_874_;
}
else
{
lean_object* v_reuseFailAlloc_876_; 
v_reuseFailAlloc_876_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_876_, 0, v_a_868_);
v___x_875_ = v_reuseFailAlloc_876_;
goto v_reusejp_874_;
}
v_reusejp_874_:
{
return v___x_875_;
}
}
else
{
lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v_a_884_; lean_object* v___x_886_; uint8_t v_isShared_887_; uint8_t v_isSharedCheck_891_; 
lean_del_object(v___x_870_);
lean_dec(v_a_868_);
v___x_877_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_applyConsts___closed__1, &lp_aesop_Aesop_RuleTac_applyConsts___closed__1_once, _init_lp_aesop_Aesop_RuleTac_applyConsts___closed__1);
v___x_878_ = lean_array_to_list(v_decls_854_);
v___x_879_ = lean_box(0);
v___x_880_ = lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_applyConsts_spec__1(v___x_878_, v___x_879_);
v___x_881_ = l_Lean_MessageData_ofList(v___x_880_);
v___x_882_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_882_, 0, v___x_877_);
lean_ctor_set(v___x_882_, 1, v___x_881_);
v___x_883_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyExpr_x27_spec__1___redArg(v___x_882_, v_a_858_, v_a_859_, v_a_860_, v_a_861_);
v_a_884_ = lean_ctor_get(v___x_883_, 0);
v_isSharedCheck_891_ = !lean_is_exclusive(v___x_883_);
if (v_isSharedCheck_891_ == 0)
{
v___x_886_ = v___x_883_;
v_isShared_887_ = v_isSharedCheck_891_;
goto v_resetjp_885_;
}
else
{
lean_inc(v_a_884_);
lean_dec(v___x_883_);
v___x_886_ = lean_box(0);
v_isShared_887_ = v_isSharedCheck_891_;
goto v_resetjp_885_;
}
v_resetjp_885_:
{
lean_object* v___x_889_; 
if (v_isShared_887_ == 0)
{
v___x_889_ = v___x_886_;
goto v_reusejp_888_;
}
else
{
lean_object* v_reuseFailAlloc_890_; 
v_reuseFailAlloc_890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_890_, 0, v_a_884_);
v___x_889_ = v_reuseFailAlloc_890_;
goto v_reusejp_888_;
}
v_reusejp_888_:
{
return v___x_889_;
}
}
}
}
}
else
{
lean_object* v_a_893_; lean_object* v___x_895_; uint8_t v_isShared_896_; uint8_t v_isSharedCheck_900_; 
lean_dec_ref(v_decls_854_);
v_a_893_ = lean_ctor_get(v___x_867_, 0);
v_isSharedCheck_900_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_900_ == 0)
{
v___x_895_ = v___x_867_;
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
else
{
lean_inc(v_a_893_);
lean_dec(v___x_867_);
v___x_895_ = lean_box(0);
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
v_resetjp_894_:
{
lean_object* v___x_898_; 
if (v_isShared_896_ == 0)
{
v___x_898_ = v___x_895_;
goto v_reusejp_897_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v_a_893_);
v___x_898_ = v_reuseFailAlloc_899_;
goto v_reusejp_897_;
}
v_reusejp_897_:
{
return v___x_898_;
}
}
}
}
else
{
lean_object* v_a_901_; lean_object* v___x_903_; uint8_t v_isShared_904_; uint8_t v_isSharedCheck_908_; 
lean_dec_ref(v_input_856_);
lean_dec_ref(v_decls_854_);
v_a_901_ = lean_ctor_get(v___x_863_, 0);
v_isSharedCheck_908_ = !lean_is_exclusive(v___x_863_);
if (v_isSharedCheck_908_ == 0)
{
v___x_903_ = v___x_863_;
v_isShared_904_ = v_isSharedCheck_908_;
goto v_resetjp_902_;
}
else
{
lean_inc(v_a_901_);
lean_dec(v___x_863_);
v___x_903_ = lean_box(0);
v_isShared_904_ = v_isSharedCheck_908_;
goto v_resetjp_902_;
}
v_resetjp_902_:
{
lean_object* v___x_906_; 
if (v_isShared_904_ == 0)
{
v___x_906_ = v___x_903_;
goto v_reusejp_905_;
}
else
{
lean_object* v_reuseFailAlloc_907_; 
v_reuseFailAlloc_907_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_907_, 0, v_a_901_);
v___x_906_ = v_reuseFailAlloc_907_;
goto v_reusejp_905_;
}
v_reusejp_905_:
{
return v___x_906_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyConsts___boxed(lean_object* v_decls_909_, lean_object* v_md_910_, lean_object* v_input_911_, lean_object* v_a_912_, lean_object* v_a_913_, lean_object* v_a_914_, lean_object* v_a_915_, lean_object* v_a_916_, lean_object* v_a_917_){
_start:
{
uint8_t v_md_boxed_918_; lean_object* v_res_919_; 
v_md_boxed_918_ = lean_unbox(v_md_910_);
v_res_919_ = lp_aesop_Aesop_RuleTac_applyConsts(v_decls_909_, v_md_boxed_918_, v_input_911_, v_a_912_, v_a_913_, v_a_914_, v_a_915_, v_a_916_);
lean_dec(v_a_916_);
lean_dec_ref(v_a_915_);
lean_dec(v_a_914_);
lean_dec_ref(v_a_913_);
lean_dec(v_a_912_);
return v_res_919_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_RuleTerm(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac_Apply(uint8_t builtin) {
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
res = runtime_initialize_aesop_Aesop_RuleTac_RuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac_Apply(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleTac_RuleTerm(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_Apply(uint8_t builtin) {
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
res = initialize_aesop_Aesop_RuleTac_RuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac_Apply(builtin);
}
#ifdef __cplusplus
}
#endif
