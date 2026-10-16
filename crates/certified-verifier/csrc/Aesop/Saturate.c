// Lean compiler output
// Module: Aesop.Saturate
// Imports: public import Init public meta import Init public import Aesop.RuleSet public import Aesop.Script.ScriptM import Aesop.Forward.State.Initial import Aesop.RuleTac import Aesop.Search.Expansion.Basic import Aesop.Script.Check import Batteries.Data.BinomialHeap.Basic
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_aesop_Aesop_RuleTacDescr_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_runRuleTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
double lean_float_div(double, double);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_mkOnGoal(lean_object*, lean_object*);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
extern lean_object* lp_aesop_Aesop_Stats_empty;
lean_object* lp_aesop_Aesop_BaseM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
lean_object* lp_aesop_Aesop_ForwardState_update(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_ForwardRuleMatch_le(lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_eraseHyp(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_ForwardRuleMatch_anyHyp(lean_object*, lean_object*);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleMatch_le___boxed(lean_object*, lean_object*);
lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_ForwardRuleMatches_empty;
lean_object* lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getMVarDependencies(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_LazyStep_toStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_dev_statefulForward;
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedOptions_x27_default;
lean_object* lp_aesop_Aesop_Script_TacticState_mkInitial(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_Script_instInhabitedTacticState_default;
lean_object* lp_aesop_Aesop_Stats_trace(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_isForwardOrDestructRuleName(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isForwardOrDestructRuleName___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_instInhabitedContext_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_instInhabitedContext;
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_getSingleGoal___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "rule produced more than one rule application"};
static const lean_object* lp_aesop_Aesop_getSingleGoal___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_getSingleGoal___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_getSingleGoal___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_getSingleGoal___redArg___closed__1;
static const lean_string_object lp_aesop_Aesop_getSingleGoal___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "rule did not produce exactly one subgoal"};
static const lean_object* lp_aesop_Aesop_getSingleGoal___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_getSingleGoal___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_getSingleGoal___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_getSingleGoal___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__0_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "saturate"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__0_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__0_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__0_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(54, 189, 4, 8, 167, 212, 113, 154)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__2_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__2_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__2_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__3_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__2_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__3_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__3_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__4_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__4_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__4_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__5_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__3_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__4_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(213, 96, 250, 13, 195, 1, 48, 100)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__5_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__5_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__6_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Saturate"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__6_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__6_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__7_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__5_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__6_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(205, 11, 47, 37, 88, 238, 188, 42)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__7_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__7_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__8_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__7_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(56, 213, 51, 248, 26, 211, 247, 92)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__8_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__8_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__9_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__8_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__4_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 220, 181, 153, 176, 211, 146, 195)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__9_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__9_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__10_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__10_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__10_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__11_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__9_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__10_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(91, 56, 43, 23, 149, 206, 111, 116)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__11_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__11_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__12_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__12_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__12_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__13_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__11_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__12_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(190, 163, 219, 113, 221, 168, 244, 177)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__13_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__13_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__14_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__13_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__4_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(120, 91, 23, 108, 247, 105, 12, 67)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__14_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__14_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__15_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__14_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__6_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(4, 34, 57, 61, 254, 163, 99, 141)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__15_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__15_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__16_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__16_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__17_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__17_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__17_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__18_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__18_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__19_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__19_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__19_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__20_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__20_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__21_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__21_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0;
static const lean_array_object lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___boxed(lean_object**);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "running rule "};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__4_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__6_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__7_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__8_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__9_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__10_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__11_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__12_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__13_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__14 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__14_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__15 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__10(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__10___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8_spec__9(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__0 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__0_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rule '"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "' does not support script generation (saturate\?)"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__2_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__3;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "selecting safe rules"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__3___boxed(lean_object**);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "trying safe rules"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1_spec__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__1_value;
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__2_value;
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__3_value;
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "selecting normalisation rules"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__3___boxed(lean_object**);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "trying normalisation rules"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__1_value;
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__0_value;
static const lean_string_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__2_value_aux_0),((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__2 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__3;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__4;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0___boxed(lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goal "};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_saturateCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "saturate: internal error: "};
static const lean_object* lp_aesop_Aesop_saturateCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_saturateCore___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_saturateCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_saturateCore___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_saturateCore___closed__1 = (const lean_object*)&lp_aesop_Aesop_saturateCore___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_saturateCore___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_saturateCore___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__3(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__6(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__4(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRuleMatch_le___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "added hyp (depth "};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ") "};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___boxed(lean_object**);
static const lean_string_object lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goal:"};
static const lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__0;
static lean_once_cell_t lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stateful_saturateCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stateful_saturateCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stateful_saturateCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stateful_saturateCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9_spec__10(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "internal error: "};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = ": unknown goal '\?"};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "applyTactic"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "getVisibleGoalIndex"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7;
static const lean_string_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__8 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9_value_aux_0),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9_value_aux_1),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9_value_aux_2),((lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9 = (const lean_object*)&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_isForwardOrDestructRuleName(lean_object* v_n_1_){
_start:
{
uint8_t v_builder_2_; uint8_t v___x_3_; uint8_t v___x_4_; 
v_builder_2_ = lean_ctor_get_uint8(v_n_1_, sizeof(void*)*1 + 8);
v___x_3_ = 4;
v___x_4_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_2_, v___x_3_);
if (v___x_4_ == 0)
{
uint8_t v___x_5_; uint8_t v___x_6_; 
v___x_5_ = 3;
v___x_6_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_2_, v___x_5_);
return v___x_6_;
}
else
{
return v___x_4_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isForwardOrDestructRuleName___boxed(lean_object* v_n_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_aesop_Aesop_isForwardOrDestructRuleName(v_n_7_);
lean_dec_ref(v_n_7_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
static lean_object* _init_lp_aesop_Aesop_SaturateM_instInhabitedContext_default(void){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_aesop_Aesop_instInhabitedOptions_x27_default;
return v___x_10_;
}
}
static lean_object* _init_lp_aesop_Aesop_SaturateM_instInhabitedContext(void){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_aesop_Aesop_instInhabitedOptions_x27_default;
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg(lean_object* v_x_14_, lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_21_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg___closed__0));
v___x_22_ = lean_st_mk_ref(v___x_21_);
lean_inc(v___y_19_);
lean_inc_ref(v___y_18_);
lean_inc(v___y_17_);
lean_inc_ref(v___y_16_);
lean_inc(v___y_15_);
lean_inc(v___x_22_);
v___x_23_ = lean_apply_7(v_x_14_, v___x_22_, v___y_15_, v___y_16_, v___y_17_, v___y_18_, v___y_19_, lean_box(0));
if (lean_obj_tag(v___x_23_) == 0)
{
lean_object* v_a_24_; lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_33_; 
v_a_24_ = lean_ctor_get(v___x_23_, 0);
v_isSharedCheck_33_ = !lean_is_exclusive(v___x_23_);
if (v_isSharedCheck_33_ == 0)
{
v___x_26_ = v___x_23_;
v_isShared_27_ = v_isSharedCheck_33_;
goto v_resetjp_25_;
}
else
{
lean_inc(v_a_24_);
lean_dec(v___x_23_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_33_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_31_; 
v___x_28_ = lean_st_ref_get(v___x_22_);
lean_dec(v___x_22_);
v___x_29_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_29_, 0, v_a_24_);
lean_ctor_set(v___x_29_, 1, v___x_28_);
if (v_isShared_27_ == 0)
{
lean_ctor_set(v___x_26_, 0, v___x_29_);
v___x_31_ = v___x_26_;
goto v_reusejp_30_;
}
else
{
lean_object* v_reuseFailAlloc_32_; 
v_reuseFailAlloc_32_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_32_, 0, v___x_29_);
v___x_31_ = v_reuseFailAlloc_32_;
goto v_reusejp_30_;
}
v_reusejp_30_:
{
return v___x_31_;
}
}
}
else
{
lean_object* v_a_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_41_; 
lean_dec(v___x_22_);
v_a_34_ = lean_ctor_get(v___x_23_, 0);
v_isSharedCheck_41_ = !lean_is_exclusive(v___x_23_);
if (v_isSharedCheck_41_ == 0)
{
v___x_36_ = v___x_23_;
v_isShared_37_ = v_isSharedCheck_41_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_a_34_);
lean_dec(v___x_23_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_41_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_39_; 
if (v_isShared_37_ == 0)
{
v___x_39_ = v___x_36_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v_a_34_);
v___x_39_ = v_reuseFailAlloc_40_;
goto v_reusejp_38_;
}
v_reusejp_38_:
{
return v___x_39_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg___boxed(lean_object* v_x_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg(v_x_42_, v___y_43_, v___y_44_, v___y_45_, v___y_46_, v___y_47_);
lean_dec(v___y_47_);
lean_dec_ref(v___y_46_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0(lean_object* v_00_u03b1_50_, lean_object* v_x_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___redArg(v_x_51_, v___y_52_, v___y_53_, v___y_54_, v___y_55_, v___y_56_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___boxed(lean_object* v_00_u03b1_59_, lean_object* v_x_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0(v_00_u03b1_59_, v_x_60_, v___y_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
lean_dec(v___y_63_);
lean_dec_ref(v___y_62_);
lean_dec(v___y_61_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_run___redArg(lean_object* v_options_68_, lean_object* v_x_69_, lean_object* v_a_70_, lean_object* v_a_71_, lean_object* v_a_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_75_ = lean_apply_1(v_x_69_, v_options_68_);
v___x_76_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_SaturateM_run_spec__0___boxed), 8, 2);
lean_closure_set(v___x_76_, 0, lean_box(0));
lean_closure_set(v___x_76_, 1, v___x_75_);
v___x_77_ = lp_aesop_Aesop_Stats_empty;
v___x_78_ = lp_aesop_Aesop_BaseM_run___redArg(v___x_76_, v___x_77_, v_a_70_, v_a_71_, v_a_72_, v_a_73_);
if (lean_obj_tag(v___x_78_) == 0)
{
lean_object* v_a_79_; lean_object* v___x_81_; uint8_t v_isShared_82_; uint8_t v_isSharedCheck_97_; 
v_a_79_ = lean_ctor_get(v___x_78_, 0);
v_isSharedCheck_97_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_97_ == 0)
{
v___x_81_ = v___x_78_;
v_isShared_82_ = v_isSharedCheck_97_;
goto v_resetjp_80_;
}
else
{
lean_inc(v_a_79_);
lean_dec(v___x_78_);
v___x_81_ = lean_box(0);
v_isShared_82_ = v_isSharedCheck_97_;
goto v_resetjp_80_;
}
v_resetjp_80_:
{
lean_object* v_fst_83_; lean_object* v_snd_84_; lean_object* v_fst_85_; lean_object* v___x_87_; uint8_t v_isShared_88_; uint8_t v_isSharedCheck_95_; 
v_fst_83_ = lean_ctor_get(v_a_79_, 0);
lean_inc(v_fst_83_);
v_snd_84_ = lean_ctor_get(v_a_79_, 1);
lean_inc(v_snd_84_);
lean_dec(v_a_79_);
v_fst_85_ = lean_ctor_get(v_fst_83_, 0);
v_isSharedCheck_95_ = !lean_is_exclusive(v_fst_83_);
if (v_isSharedCheck_95_ == 0)
{
lean_object* v_unused_96_; 
v_unused_96_ = lean_ctor_get(v_fst_83_, 1);
lean_dec(v_unused_96_);
v___x_87_ = v_fst_83_;
v_isShared_88_ = v_isSharedCheck_95_;
goto v_resetjp_86_;
}
else
{
lean_inc(v_fst_85_);
lean_dec(v_fst_83_);
v___x_87_ = lean_box(0);
v_isShared_88_ = v_isSharedCheck_95_;
goto v_resetjp_86_;
}
v_resetjp_86_:
{
lean_object* v___x_90_; 
if (v_isShared_88_ == 0)
{
lean_ctor_set(v___x_87_, 1, v_snd_84_);
v___x_90_ = v___x_87_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v_fst_85_);
lean_ctor_set(v_reuseFailAlloc_94_, 1, v_snd_84_);
v___x_90_ = v_reuseFailAlloc_94_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
lean_object* v___x_92_; 
if (v_isShared_82_ == 0)
{
lean_ctor_set(v___x_81_, 0, v___x_90_);
v___x_92_ = v___x_81_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v___x_90_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
}
}
else
{
lean_object* v_a_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_105_; 
v_a_98_ = lean_ctor_get(v___x_78_, 0);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_105_ == 0)
{
v___x_100_ = v___x_78_;
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_a_98_);
lean_dec(v___x_78_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_103_; 
if (v_isShared_101_ == 0)
{
v___x_103_ = v___x_100_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_a_98_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_run___redArg___boxed(lean_object* v_options_106_, lean_object* v_x_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_aesop_Aesop_SaturateM_run___redArg(v_options_106_, v_x_107_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
lean_dec(v_a_111_);
lean_dec_ref(v_a_110_);
lean_dec(v_a_109_);
lean_dec_ref(v_a_108_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_run(lean_object* v_00_u03b1_114_, lean_object* v_options_115_, lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_, lean_object* v_a_119_, lean_object* v_a_120_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_aesop_Aesop_SaturateM_run___redArg(v_options_115_, v_x_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SaturateM_run___boxed(lean_object* v_00_u03b1_123_, lean_object* v_options_124_, lean_object* v_x_125_, lean_object* v_a_126_, lean_object* v_a_127_, lean_object* v_a_128_, lean_object* v_a_129_, lean_object* v_a_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_aesop_Aesop_SaturateM_run(v_00_u03b1_123_, v_options_124_, v_x_125_, v_a_126_, v_a_127_, v_a_128_, v_a_129_);
lean_dec(v_a_129_);
lean_dec_ref(v_a_128_);
lean_dec(v_a_127_);
lean_dec_ref(v_a_126_);
return v_res_131_;
}
}
static lean_object* _init_lp_aesop_Aesop_getSingleGoal___redArg___closed__1(void){
_start:
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = ((lean_object*)(lp_aesop_Aesop_getSingleGoal___redArg___closed__0));
v___x_134_ = l_Lean_stringToMessageData(v___x_133_);
return v___x_134_;
}
}
static lean_object* _init_lp_aesop_Aesop_getSingleGoal___redArg___closed__3(void){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_136_ = ((lean_object*)(lp_aesop_Aesop_getSingleGoal___redArg___closed__2));
v___x_137_ = l_Lean_stringToMessageData(v___x_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___redArg(lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_o_140_){
_start:
{
lean_object* v_toApplicative_141_; lean_object* v_toPure_142_; lean_object* v___x_143_; lean_object* v___x_144_; uint8_t v___x_145_; 
v_toApplicative_141_ = lean_ctor_get(v_inst_138_, 0);
v_toPure_142_ = lean_ctor_get(v_toApplicative_141_, 1);
v___x_143_ = lean_array_get_size(v_o_140_);
v___x_144_ = lean_unsigned_to_nat(1u);
v___x_145_ = lean_nat_dec_eq(v___x_143_, v___x_144_);
if (v___x_145_ == 0)
{
lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_146_ = lean_obj_once(&lp_aesop_Aesop_getSingleGoal___redArg___closed__1, &lp_aesop_Aesop_getSingleGoal___redArg___closed__1_once, _init_lp_aesop_Aesop_getSingleGoal___redArg___closed__1);
v___x_147_ = l_Lean_throwError___redArg(v_inst_138_, v_inst_139_, v___x_146_);
return v___x_147_;
}
else
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v_goals_150_; lean_object* v_postState_151_; lean_object* v_scriptSteps_x3f_152_; lean_object* v___x_153_; uint8_t v___x_154_; 
v___x_148_ = lean_unsigned_to_nat(0u);
v___x_149_ = lean_array_fget_borrowed(v_o_140_, v___x_148_);
v_goals_150_ = lean_ctor_get(v___x_149_, 0);
v_postState_151_ = lean_ctor_get(v___x_149_, 1);
v_scriptSteps_x3f_152_ = lean_ctor_get(v___x_149_, 2);
v___x_153_ = lean_array_get_size(v_goals_150_);
v___x_154_ = lean_nat_dec_eq(v___x_153_, v___x_144_);
if (v___x_154_ == 0)
{
lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_155_ = lean_obj_once(&lp_aesop_Aesop_getSingleGoal___redArg___closed__3, &lp_aesop_Aesop_getSingleGoal___redArg___closed__3_once, _init_lp_aesop_Aesop_getSingleGoal___redArg___closed__3);
v___x_156_ = l_Lean_throwError___redArg(v_inst_138_, v_inst_139_, v___x_155_);
return v___x_156_;
}
else
{
lean_object* v___x_158_; uint8_t v_isShared_159_; uint8_t v_isSharedCheck_166_; 
lean_inc(v_toPure_142_);
lean_dec_ref(v_inst_139_);
v_isSharedCheck_166_ = !lean_is_exclusive(v_inst_138_);
if (v_isSharedCheck_166_ == 0)
{
lean_object* v_unused_167_; lean_object* v_unused_168_; 
v_unused_167_ = lean_ctor_get(v_inst_138_, 1);
lean_dec(v_unused_167_);
v_unused_168_ = lean_ctor_get(v_inst_138_, 0);
lean_dec(v_unused_168_);
v___x_158_ = v_inst_138_;
v_isShared_159_ = v_isSharedCheck_166_;
goto v_resetjp_157_;
}
else
{
lean_dec(v_inst_138_);
v___x_158_ = lean_box(0);
v_isShared_159_ = v_isSharedCheck_166_;
goto v_resetjp_157_;
}
v_resetjp_157_:
{
lean_object* v___x_160_; lean_object* v___x_162_; 
v___x_160_ = lean_array_fget_borrowed(v_goals_150_, v___x_148_);
lean_inc(v_scriptSteps_x3f_152_);
lean_inc_ref(v_postState_151_);
if (v_isShared_159_ == 0)
{
lean_ctor_set(v___x_158_, 1, v_scriptSteps_x3f_152_);
lean_ctor_set(v___x_158_, 0, v_postState_151_);
v___x_162_ = v___x_158_;
goto v_reusejp_161_;
}
else
{
lean_object* v_reuseFailAlloc_165_; 
v_reuseFailAlloc_165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_165_, 0, v_postState_151_);
lean_ctor_set(v_reuseFailAlloc_165_, 1, v_scriptSteps_x3f_152_);
v___x_162_ = v_reuseFailAlloc_165_;
goto v_reusejp_161_;
}
v_reusejp_161_:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
lean_inc(v___x_160_);
v___x_163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_163_, 0, v___x_160_);
lean_ctor_set(v___x_163_, 1, v___x_162_);
v___x_164_ = lean_apply_2(v_toPure_142_, lean_box(0), v___x_163_);
return v___x_164_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___redArg___boxed(lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_o_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_aesop_Aesop_getSingleGoal___redArg(v_inst_169_, v_inst_170_, v_o_171_);
lean_dec_ref(v_o_171_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal(lean_object* v_m_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_o_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lp_aesop_Aesop_getSingleGoal___redArg(v_inst_174_, v_inst_175_, v_o_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___boxed(lean_object* v_m_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_o_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_aesop_Aesop_getSingleGoal(v_m_178_, v_inst_179_, v_inst_180_, v_o_181_);
lean_dec_ref(v_o_181_);
return v_res_182_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__16_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v___x_218_ = lean_unsigned_to_nat(2642905901u);
v___x_219_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__15_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_220_ = l_Lean_Name_num___override(v___x_219_, v___x_218_);
return v___x_220_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__18_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_222_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__17_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_223_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__16_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__16_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__16_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_);
v___x_224_ = l_Lean_Name_str___override(v___x_223_, v___x_222_);
return v___x_224_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__20_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_226_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__19_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_227_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__18_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__18_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__18_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_);
v___x_228_ = l_Lean_Name_str___override(v___x_227_, v___x_226_);
return v___x_228_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__21_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_229_ = lean_unsigned_to_nat(2u);
v___x_230_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__20_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__20_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__20_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_);
v___x_231_ = l_Lean_Name_num___override(v___x_230_, v___x_229_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_233_; uint8_t v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_233_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_234_ = 0;
v___x_235_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__21_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__21_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__21_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_);
v___x_236_ = l_Lean_registerTraceClass(v___x_233_, v___x_234_, v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_initFn_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2____boxed(lean_object* v_a_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_aesop___private_Aesop_Saturate_0__Aesop_initFn_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_();
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(lean_object* v_opts_239_, lean_object* v_opt_240_){
_start:
{
lean_object* v_name_241_; lean_object* v_defValue_242_; lean_object* v_map_243_; lean_object* v___x_244_; 
v_name_241_ = lean_ctor_get(v_opt_240_, 0);
v_defValue_242_ = lean_ctor_get(v_opt_240_, 1);
v_map_243_ = lean_ctor_get(v_opts_239_, 0);
v___x_244_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_243_, v_name_241_);
if (lean_obj_tag(v___x_244_) == 0)
{
lean_inc(v_defValue_242_);
return v_defValue_242_;
}
else
{
lean_object* v_val_245_; 
v_val_245_ = lean_ctor_get(v___x_244_, 0);
lean_inc(v_val_245_);
lean_dec_ref_known(v___x_244_, 1);
if (lean_obj_tag(v_val_245_) == 0)
{
lean_object* v_v_246_; 
v_v_246_ = lean_ctor_get(v_val_245_, 0);
lean_inc_ref(v_v_246_);
lean_dec_ref_known(v_val_245_, 1);
return v_v_246_;
}
else
{
lean_dec(v_val_245_);
lean_inc(v_defValue_242_);
return v_defValue_242_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3___boxed(lean_object* v_opts_247_, lean_object* v_opt_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_opts_247_, v_opt_248_);
lean_dec_ref(v_opt_248_);
lean_dec_ref(v_opts_247_);
return v_res_249_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(lean_object* v_opts_250_, lean_object* v_opt_251_){
_start:
{
lean_object* v_name_252_; lean_object* v_defValue_253_; lean_object* v_map_254_; lean_object* v___x_255_; 
v_name_252_ = lean_ctor_get(v_opt_251_, 0);
v_defValue_253_ = lean_ctor_get(v_opt_251_, 1);
v_map_254_ = lean_ctor_get(v_opts_250_, 0);
v___x_255_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_254_, v_name_252_);
if (lean_obj_tag(v___x_255_) == 0)
{
uint8_t v___x_256_; 
v___x_256_ = lean_unbox(v_defValue_253_);
return v___x_256_;
}
else
{
lean_object* v_val_257_; 
v_val_257_ = lean_ctor_get(v___x_255_, 0);
lean_inc(v_val_257_);
lean_dec_ref_known(v___x_255_, 1);
if (lean_obj_tag(v_val_257_) == 1)
{
uint8_t v_v_258_; 
v_v_258_ = lean_ctor_get_uint8(v_val_257_, 0);
lean_dec_ref_known(v_val_257_, 0);
return v_v_258_;
}
else
{
uint8_t v___x_259_; 
lean_dec(v_val_257_);
v___x_259_ = lean_unbox(v_defValue_253_);
return v___x_259_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4___boxed(lean_object* v_opts_260_, lean_object* v_opt_261_){
_start:
{
uint8_t v_res_262_; lean_object* v_r_263_; 
v_res_262_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_260_, v_opt_261_);
lean_dec_ref(v_opt_261_);
lean_dec_ref(v_opts_260_);
v_r_263_ = lean_box(v_res_262_);
return v_r_263_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_264_ = lean_unsigned_to_nat(32u);
v___x_265_ = lean_mk_empty_array_with_capacity(v___x_264_);
v___x_266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_266_, 0, v___x_265_);
return v___x_266_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__1(void){
_start:
{
size_t v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_267_ = ((size_t)5ULL);
v___x_268_ = lean_unsigned_to_nat(0u);
v___x_269_ = lean_unsigned_to_nat(32u);
v___x_270_ = lean_mk_empty_array_with_capacity(v___x_269_);
v___x_271_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__0);
v___x_272_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_272_, 0, v___x_271_);
lean_ctor_set(v___x_272_, 1, v___x_270_);
lean_ctor_set(v___x_272_, 2, v___x_268_);
lean_ctor_set(v___x_272_, 3, v___x_268_);
lean_ctor_set_usize(v___x_272_, 4, v___x_267_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(lean_object* v___y_273_){
_start:
{
lean_object* v___x_275_; lean_object* v_traceState_276_; lean_object* v_traces_277_; lean_object* v___x_278_; lean_object* v_traceState_279_; lean_object* v_env_280_; lean_object* v_nextMacroScope_281_; lean_object* v_ngen_282_; lean_object* v_auxDeclNGen_283_; lean_object* v_cache_284_; lean_object* v_messages_285_; lean_object* v_infoState_286_; lean_object* v_snapshotTasks_287_; lean_object* v___x_289_; uint8_t v_isShared_290_; uint8_t v_isSharedCheck_306_; 
v___x_275_ = lean_st_ref_get(v___y_273_);
v_traceState_276_ = lean_ctor_get(v___x_275_, 4);
lean_inc_ref(v_traceState_276_);
lean_dec(v___x_275_);
v_traces_277_ = lean_ctor_get(v_traceState_276_, 0);
lean_inc_ref(v_traces_277_);
lean_dec_ref(v_traceState_276_);
v___x_278_ = lean_st_ref_take(v___y_273_);
v_traceState_279_ = lean_ctor_get(v___x_278_, 4);
v_env_280_ = lean_ctor_get(v___x_278_, 0);
v_nextMacroScope_281_ = lean_ctor_get(v___x_278_, 1);
v_ngen_282_ = lean_ctor_get(v___x_278_, 2);
v_auxDeclNGen_283_ = lean_ctor_get(v___x_278_, 3);
v_cache_284_ = lean_ctor_get(v___x_278_, 5);
v_messages_285_ = lean_ctor_get(v___x_278_, 6);
v_infoState_286_ = lean_ctor_get(v___x_278_, 7);
v_snapshotTasks_287_ = lean_ctor_get(v___x_278_, 8);
v_isSharedCheck_306_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_306_ == 0)
{
v___x_289_ = v___x_278_;
v_isShared_290_ = v_isSharedCheck_306_;
goto v_resetjp_288_;
}
else
{
lean_inc(v_snapshotTasks_287_);
lean_inc(v_infoState_286_);
lean_inc(v_messages_285_);
lean_inc(v_cache_284_);
lean_inc(v_traceState_279_);
lean_inc(v_auxDeclNGen_283_);
lean_inc(v_ngen_282_);
lean_inc(v_nextMacroScope_281_);
lean_inc(v_env_280_);
lean_dec(v___x_278_);
v___x_289_ = lean_box(0);
v_isShared_290_ = v_isSharedCheck_306_;
goto v_resetjp_288_;
}
v_resetjp_288_:
{
uint64_t v_tid_291_; lean_object* v___x_293_; uint8_t v_isShared_294_; uint8_t v_isSharedCheck_304_; 
v_tid_291_ = lean_ctor_get_uint64(v_traceState_279_, sizeof(void*)*1);
v_isSharedCheck_304_ = !lean_is_exclusive(v_traceState_279_);
if (v_isSharedCheck_304_ == 0)
{
lean_object* v_unused_305_; 
v_unused_305_ = lean_ctor_get(v_traceState_279_, 0);
lean_dec(v_unused_305_);
v___x_293_ = v_traceState_279_;
v_isShared_294_ = v_isSharedCheck_304_;
goto v_resetjp_292_;
}
else
{
lean_dec(v_traceState_279_);
v___x_293_ = lean_box(0);
v_isShared_294_ = v_isSharedCheck_304_;
goto v_resetjp_292_;
}
v_resetjp_292_:
{
lean_object* v___x_295_; lean_object* v___x_297_; 
v___x_295_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___closed__1);
if (v_isShared_294_ == 0)
{
lean_ctor_set(v___x_293_, 0, v___x_295_);
v___x_297_ = v___x_293_;
goto v_reusejp_296_;
}
else
{
lean_object* v_reuseFailAlloc_303_; 
v_reuseFailAlloc_303_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_303_, 0, v___x_295_);
lean_ctor_set_uint64(v_reuseFailAlloc_303_, sizeof(void*)*1, v_tid_291_);
v___x_297_ = v_reuseFailAlloc_303_;
goto v_reusejp_296_;
}
v_reusejp_296_:
{
lean_object* v___x_299_; 
if (v_isShared_290_ == 0)
{
lean_ctor_set(v___x_289_, 4, v___x_297_);
v___x_299_ = v___x_289_;
goto v_reusejp_298_;
}
else
{
lean_object* v_reuseFailAlloc_302_; 
v_reuseFailAlloc_302_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_302_, 0, v_env_280_);
lean_ctor_set(v_reuseFailAlloc_302_, 1, v_nextMacroScope_281_);
lean_ctor_set(v_reuseFailAlloc_302_, 2, v_ngen_282_);
lean_ctor_set(v_reuseFailAlloc_302_, 3, v_auxDeclNGen_283_);
lean_ctor_set(v_reuseFailAlloc_302_, 4, v___x_297_);
lean_ctor_set(v_reuseFailAlloc_302_, 5, v_cache_284_);
lean_ctor_set(v_reuseFailAlloc_302_, 6, v_messages_285_);
lean_ctor_set(v_reuseFailAlloc_302_, 7, v_infoState_286_);
lean_ctor_set(v_reuseFailAlloc_302_, 8, v_snapshotTasks_287_);
v___x_299_ = v_reuseFailAlloc_302_;
goto v_reusejp_298_;
}
v_reusejp_298_:
{
lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_300_ = lean_st_ref_set(v___y_273_, v___x_299_);
v___x_301_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_301_, 0, v_traces_277_);
return v___x_301_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg___boxed(lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v___y_307_);
lean_dec(v___y_307_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5(lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v___y_316_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___boxed(lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5(v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_);
lean_dec(v___y_325_);
lean_dec_ref(v___y_324_);
lean_dec(v___y_323_);
lean_dec_ref(v___y_322_);
lean_dec(v___y_321_);
lean_dec(v___y_320_);
lean_dec_ref(v___y_319_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_){
_start:
{
lean_object* v_options_336_; lean_object* v___x_337_; uint8_t v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v_options_336_ = lean_ctor_get(v___y_333_, 2);
v___x_337_ = lp_aesop_Aesop_aesop_collectStats;
v___x_338_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_336_, v___x_337_);
v___x_339_ = lean_box(v___x_338_);
v___x_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0___boxed(lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(v___y_341_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
lean_dec(v___y_345_);
lean_dec_ref(v___y_344_);
lean_dec(v___y_343_);
lean_dec(v___y_342_);
lean_dec_ref(v___y_341_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(lean_object* v_opt_350_, lean_object* v___y_351_){
_start:
{
lean_object* v_options_353_; lean_object* v_option_354_; uint8_t v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v_options_353_ = lean_ctor_get(v___y_351_, 2);
v_option_354_ = lean_ctor_get(v_opt_350_, 1);
v___x_355_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_353_, v_option_354_);
v___x_356_ = lean_box(v___x_355_);
v___x_357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_357_, 0, v___x_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg___boxed(lean_object* v_opt_358_, lean_object* v___y_359_, lean_object* v___y_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v_opt_358_, v___y_359_);
lean_dec_ref(v___y_359_);
lean_dec_ref(v_opt_358_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(uint8_t v_b_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_){
_start:
{
if (v_b_363_ == 0)
{
lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v_a_374_; uint8_t v___x_375_; 
v___x_372_ = lp_aesop_Aesop_TraceOption_stats;
v___x_373_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_372_, v___y_369_);
v_a_374_ = lean_ctor_get(v___x_373_, 0);
lean_inc(v_a_374_);
v___x_375_ = lean_unbox(v_a_374_);
lean_dec(v_a_374_);
if (v___x_375_ == 0)
{
lean_object* v_options_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; uint8_t v___x_380_; 
v_options_376_ = lean_ctor_get(v___y_369_, 2);
v___x_377_ = lp_aesop_Aesop_aesop_stats_file;
v___x_378_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_376_, v___x_377_);
v___x_379_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_380_ = lean_string_dec_eq(v___x_378_, v___x_379_);
lean_dec_ref(v___x_378_);
if (v___x_380_ == 0)
{
lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_389_; 
v_isSharedCheck_389_ = !lean_is_exclusive(v___x_373_);
if (v_isSharedCheck_389_ == 0)
{
lean_object* v_unused_390_; 
v_unused_390_ = lean_ctor_get(v___x_373_, 0);
lean_dec(v_unused_390_);
v___x_382_ = v___x_373_;
v_isShared_383_ = v_isSharedCheck_389_;
goto v_resetjp_381_;
}
else
{
lean_dec(v___x_373_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_389_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
uint8_t v___x_384_; lean_object* v___x_385_; lean_object* v___x_387_; 
v___x_384_ = 1;
v___x_385_ = lean_box(v___x_384_);
if (v_isShared_383_ == 0)
{
lean_ctor_set(v___x_382_, 0, v___x_385_);
v___x_387_ = v___x_382_;
goto v_reusejp_386_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v___x_385_);
v___x_387_ = v_reuseFailAlloc_388_;
goto v_reusejp_386_;
}
v_reusejp_386_:
{
return v___x_387_;
}
}
}
else
{
return v___x_373_;
}
}
else
{
return v___x_373_;
}
}
else
{
lean_object* v___x_391_; lean_object* v___x_392_; 
v___x_391_ = lean_box(v_b_363_);
v___x_392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
return v___x_392_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___boxed(lean_object* v_b_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_){
_start:
{
uint8_t v_b_boxed_402_; lean_object* v_res_403_; 
v_b_boxed_402_ = lean_unbox(v_b_393_);
v_res_403_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(v_b_boxed_402_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_);
lean_dec(v___y_400_);
lean_dec_ref(v___y_399_);
lean_dec(v___y_398_);
lean_dec_ref(v___y_397_);
lean_dec(v___y_396_);
lean_dec(v___y_395_);
lean_dec_ref(v___y_394_);
return v_res_403_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__2(lean_object* v_x_404_){
_start:
{
if (lean_obj_tag(v_x_404_) == 0)
{
uint8_t v___x_405_; 
v___x_405_ = 0;
return v___x_405_;
}
else
{
uint8_t v___x_406_; 
v___x_406_ = 1;
return v___x_406_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__2___boxed(lean_object* v_x_407_){
_start:
{
uint8_t v_res_408_; lean_object* v_r_409_; 
v_res_408_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__2(v_x_407_);
lean_dec(v_x_407_);
v_r_409_ = lean_box(v_res_408_);
return v_r_409_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0_spec__0(lean_object* v_msgData_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_){
_start:
{
lean_object* v___x_416_; lean_object* v_env_417_; lean_object* v___x_418_; lean_object* v_mctx_419_; lean_object* v_lctx_420_; lean_object* v_options_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_416_ = lean_st_ref_get(v___y_414_);
v_env_417_ = lean_ctor_get(v___x_416_, 0);
lean_inc_ref(v_env_417_);
lean_dec(v___x_416_);
v___x_418_ = lean_st_ref_get(v___y_412_);
v_mctx_419_ = lean_ctor_get(v___x_418_, 0);
lean_inc_ref(v_mctx_419_);
lean_dec(v___x_418_);
v_lctx_420_ = lean_ctor_get(v___y_411_, 2);
v_options_421_ = lean_ctor_get(v___y_413_, 2);
lean_inc_ref(v_options_421_);
lean_inc_ref(v_lctx_420_);
v___x_422_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_422_, 0, v_env_417_);
lean_ctor_set(v___x_422_, 1, v_mctx_419_);
lean_ctor_set(v___x_422_, 2, v_lctx_420_);
lean_ctor_set(v___x_422_, 3, v_options_421_);
v___x_423_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
lean_ctor_set(v___x_423_, 1, v_msgData_410_);
v___x_424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_424_, 0, v___x_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0_spec__0___boxed(lean_object* v_msgData_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_){
_start:
{
lean_object* v_res_431_; 
v_res_431_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0_spec__0(v_msgData_425_, v___y_426_, v___y_427_, v___y_428_, v___y_429_);
lean_dec(v___y_429_);
lean_dec_ref(v___y_428_);
lean_dec(v___y_427_);
lean_dec_ref(v___y_426_);
return v_res_431_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_432_; double v___x_433_; 
v___x_432_ = lean_unsigned_to_nat(0u);
v___x_433_ = lean_float_of_nat(v___x_432_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(lean_object* v_cls_436_, lean_object* v_msg_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
lean_object* v_ref_443_; lean_object* v___x_444_; lean_object* v_a_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_489_; 
v_ref_443_ = lean_ctor_get(v___y_440_, 5);
v___x_444_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0_spec__0(v_msg_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_);
v_a_445_ = lean_ctor_get(v___x_444_, 0);
v_isSharedCheck_489_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_489_ == 0)
{
v___x_447_ = v___x_444_;
v_isShared_448_ = v_isSharedCheck_489_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_a_445_);
lean_dec(v___x_444_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_489_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v___x_449_; lean_object* v_traceState_450_; lean_object* v_env_451_; lean_object* v_nextMacroScope_452_; lean_object* v_ngen_453_; lean_object* v_auxDeclNGen_454_; lean_object* v_cache_455_; lean_object* v_messages_456_; lean_object* v_infoState_457_; lean_object* v_snapshotTasks_458_; lean_object* v___x_460_; uint8_t v_isShared_461_; uint8_t v_isSharedCheck_488_; 
v___x_449_ = lean_st_ref_take(v___y_441_);
v_traceState_450_ = lean_ctor_get(v___x_449_, 4);
v_env_451_ = lean_ctor_get(v___x_449_, 0);
v_nextMacroScope_452_ = lean_ctor_get(v___x_449_, 1);
v_ngen_453_ = lean_ctor_get(v___x_449_, 2);
v_auxDeclNGen_454_ = lean_ctor_get(v___x_449_, 3);
v_cache_455_ = lean_ctor_get(v___x_449_, 5);
v_messages_456_ = lean_ctor_get(v___x_449_, 6);
v_infoState_457_ = lean_ctor_get(v___x_449_, 7);
v_snapshotTasks_458_ = lean_ctor_get(v___x_449_, 8);
v_isSharedCheck_488_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_488_ == 0)
{
v___x_460_ = v___x_449_;
v_isShared_461_ = v_isSharedCheck_488_;
goto v_resetjp_459_;
}
else
{
lean_inc(v_snapshotTasks_458_);
lean_inc(v_infoState_457_);
lean_inc(v_messages_456_);
lean_inc(v_cache_455_);
lean_inc(v_traceState_450_);
lean_inc(v_auxDeclNGen_454_);
lean_inc(v_ngen_453_);
lean_inc(v_nextMacroScope_452_);
lean_inc(v_env_451_);
lean_dec(v___x_449_);
v___x_460_ = lean_box(0);
v_isShared_461_ = v_isSharedCheck_488_;
goto v_resetjp_459_;
}
v_resetjp_459_:
{
uint64_t v_tid_462_; lean_object* v_traces_463_; lean_object* v___x_465_; uint8_t v_isShared_466_; uint8_t v_isSharedCheck_487_; 
v_tid_462_ = lean_ctor_get_uint64(v_traceState_450_, sizeof(void*)*1);
v_traces_463_ = lean_ctor_get(v_traceState_450_, 0);
v_isSharedCheck_487_ = !lean_is_exclusive(v_traceState_450_);
if (v_isSharedCheck_487_ == 0)
{
v___x_465_ = v_traceState_450_;
v_isShared_466_ = v_isSharedCheck_487_;
goto v_resetjp_464_;
}
else
{
lean_inc(v_traces_463_);
lean_dec(v_traceState_450_);
v___x_465_ = lean_box(0);
v_isShared_466_ = v_isSharedCheck_487_;
goto v_resetjp_464_;
}
v_resetjp_464_:
{
lean_object* v___x_467_; double v___x_468_; uint8_t v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_477_; 
v___x_467_ = lean_box(0);
v___x_468_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0);
v___x_469_ = 0;
v___x_470_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_471_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_471_, 0, v_cls_436_);
lean_ctor_set(v___x_471_, 1, v___x_467_);
lean_ctor_set(v___x_471_, 2, v___x_470_);
lean_ctor_set_float(v___x_471_, sizeof(void*)*3, v___x_468_);
lean_ctor_set_float(v___x_471_, sizeof(void*)*3 + 8, v___x_468_);
lean_ctor_set_uint8(v___x_471_, sizeof(void*)*3 + 16, v___x_469_);
v___x_472_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__1));
v___x_473_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_473_, 0, v___x_471_);
lean_ctor_set(v___x_473_, 1, v_a_445_);
lean_ctor_set(v___x_473_, 2, v___x_472_);
lean_inc(v_ref_443_);
v___x_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_474_, 0, v_ref_443_);
lean_ctor_set(v___x_474_, 1, v___x_473_);
v___x_475_ = l_Lean_PersistentArray_push___redArg(v_traces_463_, v___x_474_);
if (v_isShared_466_ == 0)
{
lean_ctor_set(v___x_465_, 0, v___x_475_);
v___x_477_ = v___x_465_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v___x_475_);
lean_ctor_set_uint64(v_reuseFailAlloc_486_, sizeof(void*)*1, v_tid_462_);
v___x_477_ = v_reuseFailAlloc_486_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
lean_object* v___x_479_; 
if (v_isShared_461_ == 0)
{
lean_ctor_set(v___x_460_, 4, v___x_477_);
v___x_479_ = v___x_460_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_485_; 
v_reuseFailAlloc_485_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_485_, 0, v_env_451_);
lean_ctor_set(v_reuseFailAlloc_485_, 1, v_nextMacroScope_452_);
lean_ctor_set(v_reuseFailAlloc_485_, 2, v_ngen_453_);
lean_ctor_set(v_reuseFailAlloc_485_, 3, v_auxDeclNGen_454_);
lean_ctor_set(v_reuseFailAlloc_485_, 4, v___x_477_);
lean_ctor_set(v_reuseFailAlloc_485_, 5, v_cache_455_);
lean_ctor_set(v_reuseFailAlloc_485_, 6, v_messages_456_);
lean_ctor_set(v_reuseFailAlloc_485_, 7, v_infoState_457_);
lean_ctor_set(v_reuseFailAlloc_485_, 8, v_snapshotTasks_458_);
v___x_479_ = v_reuseFailAlloc_485_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_483_; 
v___x_480_ = lean_st_ref_set(v___y_441_, v___x_479_);
v___x_481_ = lean_box(0);
if (v_isShared_448_ == 0)
{
lean_ctor_set(v___x_447_, 0, v___x_481_);
v___x_483_ = v___x_447_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v___x_481_);
v___x_483_ = v_reuseFailAlloc_484_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
return v___x_483_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___boxed(lean_object* v_cls_490_, lean_object* v_msg_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v_cls_490_, v_msg_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
lean_dec(v___y_493_);
lean_dec_ref(v___y_492_);
return v_res_497_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg(lean_object* v_msg_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_){
_start:
{
lean_object* v_ref_504_; lean_object* v___x_505_; lean_object* v_a_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_514_; 
v_ref_504_ = lean_ctor_get(v___y_501_, 5);
v___x_505_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0_spec__0(v_msg_498_, v___y_499_, v___y_500_, v___y_501_, v___y_502_);
v_a_506_ = lean_ctor_get(v___x_505_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_505_);
if (v_isSharedCheck_514_ == 0)
{
v___x_508_ = v___x_505_;
v_isShared_509_ = v_isSharedCheck_514_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_a_506_);
lean_dec(v___x_505_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_514_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v___x_510_; lean_object* v___x_512_; 
lean_inc(v_ref_504_);
v___x_510_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_510_, 0, v_ref_504_);
lean_ctor_set(v___x_510_, 1, v_a_506_);
if (v_isShared_509_ == 0)
{
lean_ctor_set_tag(v___x_508_, 1);
lean_ctor_set(v___x_508_, 0, v___x_510_);
v___x_512_ = v___x_508_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v___x_510_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
return v___x_512_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg___boxed(lean_object* v_msg_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg(v_msg_515_, v___y_516_, v___y_517_, v___y_518_, v___y_519_);
lean_dec(v___y_519_);
lean_dec_ref(v___y_518_);
lean_dec(v___y_517_);
lean_dec_ref(v___y_516_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1(lean_object* v_o_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_){
_start:
{
lean_object* v___x_531_; lean_object* v___x_532_; uint8_t v___x_533_; 
v___x_531_ = lean_array_get_size(v_o_522_);
v___x_532_ = lean_unsigned_to_nat(1u);
v___x_533_ = lean_nat_dec_eq(v___x_531_, v___x_532_);
if (v___x_533_ == 0)
{
lean_object* v___x_534_; lean_object* v___x_535_; 
v___x_534_ = lean_obj_once(&lp_aesop_Aesop_getSingleGoal___redArg___closed__1, &lp_aesop_Aesop_getSingleGoal___redArg___closed__1_once, _init_lp_aesop_Aesop_getSingleGoal___redArg___closed__1);
v___x_535_ = lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg(v___x_534_, v___y_526_, v___y_527_, v___y_528_, v___y_529_);
return v___x_535_;
}
else
{
lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v_goals_538_; lean_object* v_postState_539_; lean_object* v_scriptSteps_x3f_540_; lean_object* v___x_541_; uint8_t v___x_542_; 
v___x_536_ = lean_unsigned_to_nat(0u);
v___x_537_ = lean_array_fget_borrowed(v_o_522_, v___x_536_);
v_goals_538_ = lean_ctor_get(v___x_537_, 0);
v_postState_539_ = lean_ctor_get(v___x_537_, 1);
v_scriptSteps_x3f_540_ = lean_ctor_get(v___x_537_, 2);
v___x_541_ = lean_array_get_size(v_goals_538_);
v___x_542_ = lean_nat_dec_eq(v___x_541_, v___x_532_);
if (v___x_542_ == 0)
{
lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_543_ = lean_obj_once(&lp_aesop_Aesop_getSingleGoal___redArg___closed__3, &lp_aesop_Aesop_getSingleGoal___redArg___closed__3_once, _init_lp_aesop_Aesop_getSingleGoal___redArg___closed__3);
v___x_544_ = lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg(v___x_543_, v___y_526_, v___y_527_, v___y_528_, v___y_529_);
return v___x_544_;
}
else
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v___x_545_ = lean_array_fget_borrowed(v_goals_538_, v___x_536_);
lean_inc(v_scriptSteps_x3f_540_);
lean_inc_ref(v_postState_539_);
v___x_546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_546_, 0, v_postState_539_);
lean_ctor_set(v___x_546_, 1, v_scriptSteps_x3f_540_);
lean_inc(v___x_545_);
v___x_547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_547_, 0, v___x_545_);
lean_ctor_set(v___x_547_, 1, v___x_546_);
v___x_548_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_548_, 0, v___x_547_);
return v___x_548_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1___boxed(lean_object* v_o_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_){
_start:
{
lean_object* v_res_558_; 
v_res_558_ = lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1(v_o_549_, v___y_550_, v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_);
lean_dec(v___y_556_);
lean_dec_ref(v___y_555_);
lean_dec(v___y_554_);
lean_dec_ref(v___y_553_);
lean_dec(v___y_552_);
lean_dec(v___y_551_);
lean_dec_ref(v___y_550_);
lean_dec_ref(v_o_549_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3(lean_object* v_goal_562_, lean_object* v_mvars_563_, lean_object* v_locations_564_, lean_object* v_patternSubsts_x3f_565_, lean_object* v_tac_566_, lean_object* v_name_567_, lean_object* v_preState_568_, lean_object* v_cls_569_, lean_object* v_____do__lift_570_, lean_object* v___y_571_, lean_object* v___y_572_, lean_object* v___y_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_){
_start:
{
lean_object* v_input_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v_input_582_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_input_582_, 0, v_goal_562_);
lean_ctor_set(v_input_582_, 1, v_mvars_563_);
lean_ctor_set(v_input_582_, 2, v_locations_564_);
lean_ctor_set(v_input_582_, 3, v_patternSubsts_x3f_565_);
lean_ctor_set(v_input_582_, 4, v_____do__lift_570_);
v___x_583_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTacDescr_run___boxed), 8, 1);
lean_closure_set(v___x_583_, 0, v_tac_566_);
v___x_584_ = lp_aesop_Aesop_runRuleTac(v___x_583_, v_name_567_, v_preState_568_, v_input_582_, v___y_573_, v___y_574_, v___y_575_, v___y_576_, v___y_577_);
if (lean_obj_tag(v___x_584_) == 0)
{
lean_object* v_a_585_; 
v_a_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_a_585_);
lean_dec_ref_known(v___x_584_, 1);
if (lean_obj_tag(v_a_585_) == 0)
{
lean_object* v_options_586_; uint8_t v_hasTrace_587_; 
v_options_586_ = lean_ctor_get(v___y_576_, 2);
v_hasTrace_587_ = lean_ctor_get_uint8(v_options_586_, sizeof(void*)*1);
if (v_hasTrace_587_ == 0)
{
lean_dec_ref_known(v_a_585_, 1);
lean_dec(v_cls_569_);
goto v___jp_579_;
}
else
{
lean_object* v_a_588_; lean_object* v_inheritedTraceOptions_589_; lean_object* v___x_590_; lean_object* v___x_591_; uint8_t v___x_592_; 
v_a_588_ = lean_ctor_get(v_a_585_, 0);
lean_inc(v_a_588_);
lean_dec_ref_known(v_a_585_, 1);
v_inheritedTraceOptions_589_ = lean_ctor_get(v___y_576_, 13);
v___x_590_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1));
lean_inc(v_cls_569_);
v___x_591_ = l_Lean_Name_append(v___x_590_, v_cls_569_);
v___x_592_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_589_, v_options_586_, v___x_591_);
lean_dec(v___x_591_);
if (v___x_592_ == 0)
{
lean_dec(v_a_588_);
lean_dec(v_cls_569_);
goto v___jp_579_;
}
else
{
lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_593_ = l_Lean_Exception_toMessageData(v_a_588_);
v___x_594_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v_cls_569_, v___x_593_, v___y_574_, v___y_575_, v___y_576_, v___y_577_);
if (lean_obj_tag(v___x_594_) == 0)
{
lean_dec_ref_known(v___x_594_, 1);
goto v___jp_579_;
}
else
{
lean_object* v_a_595_; lean_object* v___x_597_; uint8_t v_isShared_598_; uint8_t v_isSharedCheck_602_; 
v_a_595_ = lean_ctor_get(v___x_594_, 0);
v_isSharedCheck_602_ = !lean_is_exclusive(v___x_594_);
if (v_isSharedCheck_602_ == 0)
{
v___x_597_ = v___x_594_;
v_isShared_598_ = v_isSharedCheck_602_;
goto v_resetjp_596_;
}
else
{
lean_inc(v_a_595_);
lean_dec(v___x_594_);
v___x_597_ = lean_box(0);
v_isShared_598_ = v_isSharedCheck_602_;
goto v_resetjp_596_;
}
v_resetjp_596_:
{
lean_object* v___x_600_; 
if (v_isShared_598_ == 0)
{
v___x_600_ = v___x_597_;
goto v_reusejp_599_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v_a_595_);
v___x_600_ = v_reuseFailAlloc_601_;
goto v_reusejp_599_;
}
v_reusejp_599_:
{
return v___x_600_;
}
}
}
}
}
}
else
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_648_; 
lean_dec(v_cls_569_);
v_a_603_ = lean_ctor_get(v_a_585_, 0);
v_isSharedCheck_648_ = !lean_is_exclusive(v_a_585_);
if (v_isSharedCheck_648_ == 0)
{
v___x_605_ = v_a_585_;
v_isShared_606_ = v_isSharedCheck_648_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v_a_585_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_648_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_607_; 
v___x_607_ = lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1(v_a_603_, v___y_571_, v___y_572_, v___y_573_, v___y_574_, v___y_575_, v___y_576_, v___y_577_);
lean_dec(v_a_603_);
if (lean_obj_tag(v___x_607_) == 0)
{
lean_object* v_a_608_; lean_object* v_snd_609_; lean_object* v_fst_610_; lean_object* v_fst_611_; lean_object* v_snd_612_; lean_object* v___x_614_; uint8_t v_isShared_615_; uint8_t v_isSharedCheck_639_; 
v_a_608_ = lean_ctor_get(v___x_607_, 0);
lean_inc(v_a_608_);
lean_dec_ref_known(v___x_607_, 1);
v_snd_609_ = lean_ctor_get(v_a_608_, 1);
lean_inc(v_snd_609_);
v_fst_610_ = lean_ctor_get(v_a_608_, 0);
lean_inc(v_fst_610_);
lean_dec(v_a_608_);
v_fst_611_ = lean_ctor_get(v_snd_609_, 0);
v_snd_612_ = lean_ctor_get(v_snd_609_, 1);
v_isSharedCheck_639_ = !lean_is_exclusive(v_snd_609_);
if (v_isSharedCheck_639_ == 0)
{
v___x_614_ = v_snd_609_;
v_isShared_615_ = v_isSharedCheck_639_;
goto v_resetjp_613_;
}
else
{
lean_inc(v_snd_612_);
lean_inc(v_fst_611_);
lean_dec(v_snd_609_);
v___x_614_ = lean_box(0);
v_isShared_615_ = v_isSharedCheck_639_;
goto v_resetjp_613_;
}
v_resetjp_613_:
{
lean_object* v___x_616_; 
v___x_616_ = l_Lean_Meta_SavedState_restore___redArg(v_fst_611_, v___y_575_, v___y_577_);
lean_dec(v_fst_611_);
if (lean_obj_tag(v___x_616_) == 0)
{
lean_object* v___x_618_; uint8_t v_isShared_619_; uint8_t v_isSharedCheck_629_; 
v_isSharedCheck_629_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_629_ == 0)
{
lean_object* v_unused_630_; 
v_unused_630_ = lean_ctor_get(v___x_616_, 0);
lean_dec(v_unused_630_);
v___x_618_ = v___x_616_;
v_isShared_619_ = v_isSharedCheck_629_;
goto v_resetjp_617_;
}
else
{
lean_dec(v___x_616_);
v___x_618_ = lean_box(0);
v_isShared_619_ = v_isSharedCheck_629_;
goto v_resetjp_617_;
}
v_resetjp_617_:
{
lean_object* v___x_621_; 
if (v_isShared_615_ == 0)
{
lean_ctor_set(v___x_614_, 0, v_fst_610_);
v___x_621_ = v___x_614_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_628_; 
v_reuseFailAlloc_628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_628_, 0, v_fst_610_);
lean_ctor_set(v_reuseFailAlloc_628_, 1, v_snd_612_);
v___x_621_ = v_reuseFailAlloc_628_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
lean_object* v___x_623_; 
if (v_isShared_606_ == 0)
{
lean_ctor_set(v___x_605_, 0, v___x_621_);
v___x_623_ = v___x_605_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_627_; 
v_reuseFailAlloc_627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_627_, 0, v___x_621_);
v___x_623_ = v_reuseFailAlloc_627_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
lean_object* v___x_625_; 
if (v_isShared_619_ == 0)
{
lean_ctor_set(v___x_618_, 0, v___x_623_);
v___x_625_ = v___x_618_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v___x_623_);
v___x_625_ = v_reuseFailAlloc_626_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
return v___x_625_;
}
}
}
}
}
else
{
lean_object* v_a_631_; lean_object* v___x_633_; uint8_t v_isShared_634_; uint8_t v_isSharedCheck_638_; 
lean_del_object(v___x_614_);
lean_dec(v_snd_612_);
lean_dec(v_fst_610_);
lean_del_object(v___x_605_);
v_a_631_ = lean_ctor_get(v___x_616_, 0);
v_isSharedCheck_638_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_638_ == 0)
{
v___x_633_ = v___x_616_;
v_isShared_634_ = v_isSharedCheck_638_;
goto v_resetjp_632_;
}
else
{
lean_inc(v_a_631_);
lean_dec(v___x_616_);
v___x_633_ = lean_box(0);
v_isShared_634_ = v_isSharedCheck_638_;
goto v_resetjp_632_;
}
v_resetjp_632_:
{
lean_object* v___x_636_; 
if (v_isShared_634_ == 0)
{
v___x_636_ = v___x_633_;
goto v_reusejp_635_;
}
else
{
lean_object* v_reuseFailAlloc_637_; 
v_reuseFailAlloc_637_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_637_, 0, v_a_631_);
v___x_636_ = v_reuseFailAlloc_637_;
goto v_reusejp_635_;
}
v_reusejp_635_:
{
return v___x_636_;
}
}
}
}
}
else
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
lean_del_object(v___x_605_);
v_a_640_ = lean_ctor_get(v___x_607_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_607_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_607_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_607_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_645_; 
if (v_isShared_643_ == 0)
{
v___x_645_ = v___x_642_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v_a_640_);
v___x_645_ = v_reuseFailAlloc_646_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
return v___x_645_;
}
}
}
}
}
}
else
{
lean_object* v_a_649_; lean_object* v___x_651_; uint8_t v_isShared_652_; uint8_t v_isSharedCheck_656_; 
lean_dec(v_cls_569_);
v_a_649_ = lean_ctor_get(v___x_584_, 0);
v_isSharedCheck_656_ = !lean_is_exclusive(v___x_584_);
if (v_isSharedCheck_656_ == 0)
{
v___x_651_ = v___x_584_;
v_isShared_652_ = v_isSharedCheck_656_;
goto v_resetjp_650_;
}
else
{
lean_inc(v_a_649_);
lean_dec(v___x_584_);
v___x_651_ = lean_box(0);
v_isShared_652_ = v_isSharedCheck_656_;
goto v_resetjp_650_;
}
v_resetjp_650_:
{
lean_object* v___x_654_; 
if (v_isShared_652_ == 0)
{
v___x_654_ = v___x_651_;
goto v_reusejp_653_;
}
else
{
lean_object* v_reuseFailAlloc_655_; 
v_reuseFailAlloc_655_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_655_, 0, v_a_649_);
v___x_654_ = v_reuseFailAlloc_655_;
goto v_reusejp_653_;
}
v_reusejp_653_:
{
return v___x_654_;
}
}
}
v___jp_579_:
{
lean_object* v___x_580_; lean_object* v___x_581_; 
v___x_580_ = lean_box(0);
v___x_581_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_581_, 0, v___x_580_);
return v___x_581_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___boxed(lean_object** _args){
lean_object* v_goal_657_ = _args[0];
lean_object* v_mvars_658_ = _args[1];
lean_object* v_locations_659_ = _args[2];
lean_object* v_patternSubsts_x3f_660_ = _args[3];
lean_object* v_tac_661_ = _args[4];
lean_object* v_name_662_ = _args[5];
lean_object* v_preState_663_ = _args[6];
lean_object* v_cls_664_ = _args[7];
lean_object* v_____do__lift_665_ = _args[8];
lean_object* v___y_666_ = _args[9];
lean_object* v___y_667_ = _args[10];
lean_object* v___y_668_ = _args[11];
lean_object* v___y_669_ = _args[12];
lean_object* v___y_670_ = _args[13];
lean_object* v___y_671_ = _args[14];
lean_object* v___y_672_ = _args[15];
lean_object* v___y_673_ = _args[16];
_start:
{
lean_object* v_res_674_; 
v_res_674_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3(v_goal_657_, v_mvars_658_, v_locations_659_, v_patternSubsts_x3f_660_, v_tac_661_, v_name_662_, v_preState_663_, v_cls_664_, v_____do__lift_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_);
lean_dec(v___y_672_);
lean_dec_ref(v___y_671_);
lean_dec(v___y_670_);
lean_dec_ref(v___y_669_);
lean_dec(v___y_668_);
lean_dec(v___y_667_);
lean_dec_ref(v___y_666_);
lean_dec_ref(v_preState_663_);
return v_res_674_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__1(void){
_start:
{
lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_676_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__0));
v___x_677_ = l_Lean_stringToMessageData(v___x_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4(lean_object* v_name_692_, uint8_t v_hasTrace_693_, lean_object* v_x_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_){
_start:
{
lean_object* v_name_703_; uint8_t v_builder_704_; uint8_t v_phase_705_; uint8_t v_scope_706_; lean_object* v___x_707_; lean_object* v___y_709_; lean_object* v___y_710_; lean_object* v___y_711_; lean_object* v___y_721_; lean_object* v___y_722_; lean_object* v___y_723_; lean_object* v___y_729_; 
v_name_703_ = lean_ctor_get(v_name_692_, 0);
lean_inc(v_name_703_);
v_builder_704_ = lean_ctor_get_uint8(v_name_692_, sizeof(void*)*1 + 8);
v_phase_705_ = lean_ctor_get_uint8(v_name_692_, sizeof(void*)*1 + 9);
v_scope_706_ = lean_ctor_get_uint8(v_name_692_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_692_);
v___x_707_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__1);
switch(v_phase_705_)
{
case 0:
{
lean_object* v___x_740_; 
v___x_740_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__13));
v___y_729_ = v___x_740_;
goto v___jp_728_;
}
case 1:
{
lean_object* v___x_741_; 
v___x_741_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__14));
v___y_729_ = v___x_741_;
goto v___jp_728_;
}
default: 
{
lean_object* v___x_742_; 
v___x_742_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__15));
v___y_729_ = v___x_742_;
goto v___jp_728_;
}
}
v___jp_708_:
{
lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_712_ = lean_string_append(v___y_710_, v___y_711_);
v___x_713_ = lean_string_append(v___x_712_, v___y_709_);
v___x_714_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_703_, v_hasTrace_693_);
v___x_715_ = lean_string_append(v___x_713_, v___x_714_);
lean_dec_ref(v___x_714_);
v___x_716_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_716_, 0, v___x_715_);
v___x_717_ = l_Lean_MessageData_ofFormat(v___x_716_);
v___x_718_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_718_, 0, v___x_707_);
lean_ctor_set(v___x_718_, 1, v___x_717_);
v___x_719_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
return v___x_719_;
}
v___jp_720_:
{
lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_724_ = lean_string_append(v___y_722_, v___y_723_);
v___x_725_ = lean_string_append(v___x_724_, v___y_721_);
if (v_scope_706_ == 0)
{
lean_object* v___x_726_; 
v___x_726_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__2));
v___y_709_ = v___y_721_;
v___y_710_ = v___x_725_;
v___y_711_ = v___x_726_;
goto v___jp_708_;
}
else
{
lean_object* v___x_727_; 
v___x_727_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__3));
v___y_709_ = v___y_721_;
v___y_710_ = v___x_725_;
v___y_711_ = v___x_727_;
goto v___jp_708_;
}
}
v___jp_728_:
{
lean_object* v___x_730_; lean_object* v___x_731_; 
v___x_730_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__4));
lean_inc_ref(v___y_729_);
v___x_731_ = lean_string_append(v___y_729_, v___x_730_);
switch(v_builder_704_)
{
case 0:
{
lean_object* v___x_732_; 
v___x_732_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__5));
v___y_721_ = v___x_730_;
v___y_722_ = v___x_731_;
v___y_723_ = v___x_732_;
goto v___jp_720_;
}
case 1:
{
lean_object* v___x_733_; 
v___x_733_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__6));
v___y_721_ = v___x_730_;
v___y_722_ = v___x_731_;
v___y_723_ = v___x_733_;
goto v___jp_720_;
}
case 2:
{
lean_object* v___x_734_; 
v___x_734_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__7));
v___y_721_ = v___x_730_;
v___y_722_ = v___x_731_;
v___y_723_ = v___x_734_;
goto v___jp_720_;
}
case 3:
{
lean_object* v___x_735_; 
v___x_735_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__8));
v___y_721_ = v___x_730_;
v___y_722_ = v___x_731_;
v___y_723_ = v___x_735_;
goto v___jp_720_;
}
case 4:
{
lean_object* v___x_736_; 
v___x_736_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__9));
v___y_721_ = v___x_730_;
v___y_722_ = v___x_731_;
v___y_723_ = v___x_736_;
goto v___jp_720_;
}
case 5:
{
lean_object* v___x_737_; 
v___x_737_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__10));
v___y_721_ = v___x_730_;
v___y_722_ = v___x_731_;
v___y_723_ = v___x_737_;
goto v___jp_720_;
}
case 6:
{
lean_object* v___x_738_; 
v___x_738_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__11));
v___y_721_ = v___x_730_;
v___y_722_ = v___x_731_;
v___y_723_ = v___x_738_;
goto v___jp_720_;
}
default: 
{
lean_object* v___x_739_; 
v___x_739_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__12));
v___y_721_ = v___x_730_;
v___y_722_ = v___x_731_;
v___y_723_ = v___x_739_;
goto v___jp_720_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___boxed(lean_object* v_name_743_, lean_object* v_hasTrace_744_, lean_object* v_x_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_){
_start:
{
uint8_t v_hasTrace_boxed_754_; lean_object* v_res_755_; 
v_hasTrace_boxed_754_ = lean_unbox(v_hasTrace_744_);
v_res_755_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4(v_name_743_, v_hasTrace_boxed_754_, v_x_745_, v___y_746_, v___y_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_748_);
lean_dec(v___y_747_);
lean_dec_ref(v___y_746_);
lean_dec_ref(v_x_745_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5(lean_object* v_____r_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_){
_start:
{
lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_765_ = lean_box(0);
v___x_766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_766_, 0, v___x_765_);
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5___boxed(lean_object* v_____r_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_){
_start:
{
lean_object* v_res_776_; 
v_res_776_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5(v_____r_767_, v___y_768_, v___y_769_, v___y_770_, v___y_771_, v___y_772_, v___y_773_, v___y_774_);
lean_dec(v___y_774_);
lean_dec_ref(v___y_773_);
lean_dec(v___y_772_);
lean_dec_ref(v___y_771_);
lean_dec(v___y_770_);
lean_dec(v___y_769_);
lean_dec_ref(v___y_768_);
return v_res_776_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(lean_object* v_x_777_){
_start:
{
if (lean_obj_tag(v_x_777_) == 0)
{
lean_object* v_a_779_; lean_object* v___x_781_; uint8_t v_isShared_782_; uint8_t v_isSharedCheck_786_; 
v_a_779_ = lean_ctor_get(v_x_777_, 0);
v_isSharedCheck_786_ = !lean_is_exclusive(v_x_777_);
if (v_isSharedCheck_786_ == 0)
{
v___x_781_ = v_x_777_;
v_isShared_782_ = v_isSharedCheck_786_;
goto v_resetjp_780_;
}
else
{
lean_inc(v_a_779_);
lean_dec(v_x_777_);
v___x_781_ = lean_box(0);
v_isShared_782_ = v_isSharedCheck_786_;
goto v_resetjp_780_;
}
v_resetjp_780_:
{
lean_object* v___x_784_; 
if (v_isShared_782_ == 0)
{
lean_ctor_set_tag(v___x_781_, 1);
v___x_784_ = v___x_781_;
goto v_reusejp_783_;
}
else
{
lean_object* v_reuseFailAlloc_785_; 
v_reuseFailAlloc_785_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_785_, 0, v_a_779_);
v___x_784_ = v_reuseFailAlloc_785_;
goto v_reusejp_783_;
}
v_reusejp_783_:
{
return v___x_784_;
}
}
}
else
{
lean_object* v_a_787_; lean_object* v___x_789_; uint8_t v_isShared_790_; uint8_t v_isSharedCheck_794_; 
v_a_787_ = lean_ctor_get(v_x_777_, 0);
v_isSharedCheck_794_ = !lean_is_exclusive(v_x_777_);
if (v_isSharedCheck_794_ == 0)
{
v___x_789_ = v_x_777_;
v_isShared_790_ = v_isSharedCheck_794_;
goto v_resetjp_788_;
}
else
{
lean_inc(v_a_787_);
lean_dec(v_x_777_);
v___x_789_ = lean_box(0);
v_isShared_790_ = v_isSharedCheck_794_;
goto v_resetjp_788_;
}
v_resetjp_788_:
{
lean_object* v___x_792_; 
if (v_isShared_790_ == 0)
{
lean_ctor_set_tag(v___x_789_, 0);
v___x_792_ = v___x_789_;
goto v_reusejp_791_;
}
else
{
lean_object* v_reuseFailAlloc_793_; 
v_reuseFailAlloc_793_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_793_, 0, v_a_787_);
v___x_792_ = v_reuseFailAlloc_793_;
goto v_reusejp_791_;
}
v_reusejp_791_:
{
return v___x_792_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg___boxed(lean_object* v_x_795_, lean_object* v___y_796_){
_start:
{
lean_object* v_res_797_; 
v_res_797_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_x_795_);
return v_res_797_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__10(lean_object* v_e_798_){
_start:
{
if (lean_obj_tag(v_e_798_) == 0)
{
uint8_t v___x_799_; 
v___x_799_ = 2;
return v___x_799_;
}
else
{
lean_object* v_a_800_; 
v_a_800_ = lean_ctor_get(v_e_798_, 0);
if (lean_obj_tag(v_a_800_) == 0)
{
uint8_t v___x_801_; 
v___x_801_ = 1;
return v___x_801_;
}
else
{
uint8_t v___x_802_; 
v___x_802_ = 0;
return v___x_802_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__10___boxed(lean_object* v_e_803_){
_start:
{
uint8_t v_res_804_; lean_object* v_r_805_; 
v_res_804_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__10(v_e_803_);
lean_dec_ref(v_e_803_);
v_r_805_ = lean_box(v_res_804_);
return v_r_805_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(lean_object* v_opts_806_, lean_object* v_opt_807_){
_start:
{
lean_object* v_name_808_; lean_object* v_defValue_809_; lean_object* v_map_810_; lean_object* v___x_811_; 
v_name_808_ = lean_ctor_get(v_opt_807_, 0);
v_defValue_809_ = lean_ctor_get(v_opt_807_, 1);
v_map_810_ = lean_ctor_get(v_opts_806_, 0);
v___x_811_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_810_, v_name_808_);
if (lean_obj_tag(v___x_811_) == 0)
{
lean_inc(v_defValue_809_);
return v_defValue_809_;
}
else
{
lean_object* v_val_812_; 
v_val_812_ = lean_ctor_get(v___x_811_, 0);
lean_inc(v_val_812_);
lean_dec_ref_known(v___x_811_, 1);
if (lean_obj_tag(v_val_812_) == 3)
{
lean_object* v_v_813_; 
v_v_813_ = lean_ctor_get(v_val_812_, 0);
lean_inc(v_v_813_);
lean_dec_ref_known(v_val_812_, 1);
return v_v_813_;
}
else
{
lean_dec(v_val_812_);
lean_inc(v_defValue_809_);
return v_defValue_809_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11___boxed(lean_object* v_opts_814_, lean_object* v_opt_815_){
_start:
{
lean_object* v_res_816_; 
v_res_816_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_814_, v_opt_815_);
lean_dec_ref(v_opt_815_);
lean_dec_ref(v_opts_814_);
return v_res_816_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8_spec__9(size_t v_sz_817_, size_t v_i_818_, lean_object* v_bs_819_){
_start:
{
uint8_t v___x_820_; 
v___x_820_ = lean_usize_dec_lt(v_i_818_, v_sz_817_);
if (v___x_820_ == 0)
{
return v_bs_819_;
}
else
{
lean_object* v_v_821_; lean_object* v_msg_822_; lean_object* v___x_823_; lean_object* v_bs_x27_824_; size_t v___x_825_; size_t v___x_826_; lean_object* v___x_827_; 
v_v_821_ = lean_array_uget_borrowed(v_bs_819_, v_i_818_);
v_msg_822_ = lean_ctor_get(v_v_821_, 1);
lean_inc_ref(v_msg_822_);
v___x_823_ = lean_unsigned_to_nat(0u);
v_bs_x27_824_ = lean_array_uset(v_bs_819_, v_i_818_, v___x_823_);
v___x_825_ = ((size_t)1ULL);
v___x_826_ = lean_usize_add(v_i_818_, v___x_825_);
v___x_827_ = lean_array_uset(v_bs_x27_824_, v_i_818_, v_msg_822_);
v_i_818_ = v___x_826_;
v_bs_819_ = v___x_827_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8_spec__9___boxed(lean_object* v_sz_829_, lean_object* v_i_830_, lean_object* v_bs_831_){
_start:
{
size_t v_sz_boxed_832_; size_t v_i_boxed_833_; lean_object* v_res_834_; 
v_sz_boxed_832_ = lean_unbox_usize(v_sz_829_);
lean_dec(v_sz_829_);
v_i_boxed_833_ = lean_unbox_usize(v_i_830_);
lean_dec(v_i_830_);
v_res_834_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8_spec__9(v_sz_boxed_832_, v_i_boxed_833_, v_bs_831_);
return v_res_834_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg(lean_object* v_oldTraces_835_, lean_object* v_data_836_, lean_object* v_ref_837_, lean_object* v_msg_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_){
_start:
{
lean_object* v_fileName_844_; lean_object* v_fileMap_845_; lean_object* v_options_846_; lean_object* v_currRecDepth_847_; lean_object* v_maxRecDepth_848_; lean_object* v_ref_849_; lean_object* v_currNamespace_850_; lean_object* v_openDecls_851_; lean_object* v_initHeartbeats_852_; lean_object* v_maxHeartbeats_853_; lean_object* v_quotContext_854_; lean_object* v_currMacroScope_855_; uint8_t v_diag_856_; lean_object* v_cancelTk_x3f_857_; uint8_t v_suppressElabErrors_858_; lean_object* v_inheritedTraceOptions_859_; lean_object* v___x_860_; lean_object* v_traceState_861_; lean_object* v_traces_862_; lean_object* v_ref_863_; lean_object* v___x_864_; lean_object* v___x_865_; size_t v_sz_866_; size_t v___x_867_; lean_object* v___x_868_; lean_object* v_msg_869_; lean_object* v___x_870_; lean_object* v_a_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_908_; 
v_fileName_844_ = lean_ctor_get(v___y_841_, 0);
v_fileMap_845_ = lean_ctor_get(v___y_841_, 1);
v_options_846_ = lean_ctor_get(v___y_841_, 2);
v_currRecDepth_847_ = lean_ctor_get(v___y_841_, 3);
v_maxRecDepth_848_ = lean_ctor_get(v___y_841_, 4);
v_ref_849_ = lean_ctor_get(v___y_841_, 5);
v_currNamespace_850_ = lean_ctor_get(v___y_841_, 6);
v_openDecls_851_ = lean_ctor_get(v___y_841_, 7);
v_initHeartbeats_852_ = lean_ctor_get(v___y_841_, 8);
v_maxHeartbeats_853_ = lean_ctor_get(v___y_841_, 9);
v_quotContext_854_ = lean_ctor_get(v___y_841_, 10);
v_currMacroScope_855_ = lean_ctor_get(v___y_841_, 11);
v_diag_856_ = lean_ctor_get_uint8(v___y_841_, sizeof(void*)*14);
v_cancelTk_x3f_857_ = lean_ctor_get(v___y_841_, 12);
v_suppressElabErrors_858_ = lean_ctor_get_uint8(v___y_841_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_859_ = lean_ctor_get(v___y_841_, 13);
v___x_860_ = lean_st_ref_get(v___y_842_);
v_traceState_861_ = lean_ctor_get(v___x_860_, 4);
lean_inc_ref(v_traceState_861_);
lean_dec(v___x_860_);
v_traces_862_ = lean_ctor_get(v_traceState_861_, 0);
lean_inc_ref(v_traces_862_);
lean_dec_ref(v_traceState_861_);
v_ref_863_ = l_Lean_replaceRef(v_ref_837_, v_ref_849_);
lean_inc_ref(v_inheritedTraceOptions_859_);
lean_inc(v_cancelTk_x3f_857_);
lean_inc(v_currMacroScope_855_);
lean_inc(v_quotContext_854_);
lean_inc(v_maxHeartbeats_853_);
lean_inc(v_initHeartbeats_852_);
lean_inc(v_openDecls_851_);
lean_inc(v_currNamespace_850_);
lean_inc(v_maxRecDepth_848_);
lean_inc(v_currRecDepth_847_);
lean_inc_ref(v_options_846_);
lean_inc_ref(v_fileMap_845_);
lean_inc_ref(v_fileName_844_);
v___x_864_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_864_, 0, v_fileName_844_);
lean_ctor_set(v___x_864_, 1, v_fileMap_845_);
lean_ctor_set(v___x_864_, 2, v_options_846_);
lean_ctor_set(v___x_864_, 3, v_currRecDepth_847_);
lean_ctor_set(v___x_864_, 4, v_maxRecDepth_848_);
lean_ctor_set(v___x_864_, 5, v_ref_863_);
lean_ctor_set(v___x_864_, 6, v_currNamespace_850_);
lean_ctor_set(v___x_864_, 7, v_openDecls_851_);
lean_ctor_set(v___x_864_, 8, v_initHeartbeats_852_);
lean_ctor_set(v___x_864_, 9, v_maxHeartbeats_853_);
lean_ctor_set(v___x_864_, 10, v_quotContext_854_);
lean_ctor_set(v___x_864_, 11, v_currMacroScope_855_);
lean_ctor_set(v___x_864_, 12, v_cancelTk_x3f_857_);
lean_ctor_set(v___x_864_, 13, v_inheritedTraceOptions_859_);
lean_ctor_set_uint8(v___x_864_, sizeof(void*)*14, v_diag_856_);
lean_ctor_set_uint8(v___x_864_, sizeof(void*)*14 + 1, v_suppressElabErrors_858_);
v___x_865_ = l_Lean_PersistentArray_toArray___redArg(v_traces_862_);
lean_dec_ref(v_traces_862_);
v_sz_866_ = lean_array_size(v___x_865_);
v___x_867_ = ((size_t)0ULL);
v___x_868_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8_spec__9(v_sz_866_, v___x_867_, v___x_865_);
v_msg_869_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_869_, 0, v_data_836_);
lean_ctor_set(v_msg_869_, 1, v_msg_838_);
lean_ctor_set(v_msg_869_, 2, v___x_868_);
v___x_870_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0_spec__0(v_msg_869_, v___y_839_, v___y_840_, v___x_864_, v___y_842_);
lean_dec_ref_known(v___x_864_, 14);
v_a_871_ = lean_ctor_get(v___x_870_, 0);
v_isSharedCheck_908_ = !lean_is_exclusive(v___x_870_);
if (v_isSharedCheck_908_ == 0)
{
v___x_873_ = v___x_870_;
v_isShared_874_ = v_isSharedCheck_908_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_a_871_);
lean_dec(v___x_870_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_908_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___x_875_; lean_object* v_traceState_876_; lean_object* v_env_877_; lean_object* v_nextMacroScope_878_; lean_object* v_ngen_879_; lean_object* v_auxDeclNGen_880_; lean_object* v_cache_881_; lean_object* v_messages_882_; lean_object* v_infoState_883_; lean_object* v_snapshotTasks_884_; lean_object* v___x_886_; uint8_t v_isShared_887_; uint8_t v_isSharedCheck_907_; 
v___x_875_ = lean_st_ref_take(v___y_842_);
v_traceState_876_ = lean_ctor_get(v___x_875_, 4);
v_env_877_ = lean_ctor_get(v___x_875_, 0);
v_nextMacroScope_878_ = lean_ctor_get(v___x_875_, 1);
v_ngen_879_ = lean_ctor_get(v___x_875_, 2);
v_auxDeclNGen_880_ = lean_ctor_get(v___x_875_, 3);
v_cache_881_ = lean_ctor_get(v___x_875_, 5);
v_messages_882_ = lean_ctor_get(v___x_875_, 6);
v_infoState_883_ = lean_ctor_get(v___x_875_, 7);
v_snapshotTasks_884_ = lean_ctor_get(v___x_875_, 8);
v_isSharedCheck_907_ = !lean_is_exclusive(v___x_875_);
if (v_isSharedCheck_907_ == 0)
{
v___x_886_ = v___x_875_;
v_isShared_887_ = v_isSharedCheck_907_;
goto v_resetjp_885_;
}
else
{
lean_inc(v_snapshotTasks_884_);
lean_inc(v_infoState_883_);
lean_inc(v_messages_882_);
lean_inc(v_cache_881_);
lean_inc(v_traceState_876_);
lean_inc(v_auxDeclNGen_880_);
lean_inc(v_ngen_879_);
lean_inc(v_nextMacroScope_878_);
lean_inc(v_env_877_);
lean_dec(v___x_875_);
v___x_886_ = lean_box(0);
v_isShared_887_ = v_isSharedCheck_907_;
goto v_resetjp_885_;
}
v_resetjp_885_:
{
uint64_t v_tid_888_; lean_object* v___x_890_; uint8_t v_isShared_891_; uint8_t v_isSharedCheck_905_; 
v_tid_888_ = lean_ctor_get_uint64(v_traceState_876_, sizeof(void*)*1);
v_isSharedCheck_905_ = !lean_is_exclusive(v_traceState_876_);
if (v_isSharedCheck_905_ == 0)
{
lean_object* v_unused_906_; 
v_unused_906_ = lean_ctor_get(v_traceState_876_, 0);
lean_dec(v_unused_906_);
v___x_890_ = v_traceState_876_;
v_isShared_891_ = v_isSharedCheck_905_;
goto v_resetjp_889_;
}
else
{
lean_dec(v_traceState_876_);
v___x_890_ = lean_box(0);
v_isShared_891_ = v_isSharedCheck_905_;
goto v_resetjp_889_;
}
v_resetjp_889_:
{
lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_895_; 
v___x_892_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_892_, 0, v_ref_837_);
lean_ctor_set(v___x_892_, 1, v_a_871_);
v___x_893_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_835_, v___x_892_);
if (v_isShared_891_ == 0)
{
lean_ctor_set(v___x_890_, 0, v___x_893_);
v___x_895_ = v___x_890_;
goto v_reusejp_894_;
}
else
{
lean_object* v_reuseFailAlloc_904_; 
v_reuseFailAlloc_904_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_904_, 0, v___x_893_);
lean_ctor_set_uint64(v_reuseFailAlloc_904_, sizeof(void*)*1, v_tid_888_);
v___x_895_ = v_reuseFailAlloc_904_;
goto v_reusejp_894_;
}
v_reusejp_894_:
{
lean_object* v___x_897_; 
if (v_isShared_887_ == 0)
{
lean_ctor_set(v___x_886_, 4, v___x_895_);
v___x_897_ = v___x_886_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_903_; 
v_reuseFailAlloc_903_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_903_, 0, v_env_877_);
lean_ctor_set(v_reuseFailAlloc_903_, 1, v_nextMacroScope_878_);
lean_ctor_set(v_reuseFailAlloc_903_, 2, v_ngen_879_);
lean_ctor_set(v_reuseFailAlloc_903_, 3, v_auxDeclNGen_880_);
lean_ctor_set(v_reuseFailAlloc_903_, 4, v___x_895_);
lean_ctor_set(v_reuseFailAlloc_903_, 5, v_cache_881_);
lean_ctor_set(v_reuseFailAlloc_903_, 6, v_messages_882_);
lean_ctor_set(v_reuseFailAlloc_903_, 7, v_infoState_883_);
lean_ctor_set(v_reuseFailAlloc_903_, 8, v_snapshotTasks_884_);
v___x_897_ = v_reuseFailAlloc_903_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_901_; 
v___x_898_ = lean_st_ref_set(v___y_842_, v___x_897_);
v___x_899_ = lean_box(0);
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 0, v___x_899_);
v___x_901_ = v___x_873_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_902_; 
v_reuseFailAlloc_902_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_902_, 0, v___x_899_);
v___x_901_ = v_reuseFailAlloc_902_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
return v___x_901_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg___boxed(lean_object* v_oldTraces_909_, lean_object* v_data_910_, lean_object* v_ref_911_, lean_object* v_msg_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_){
_start:
{
lean_object* v_res_918_; 
v_res_918_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg(v_oldTraces_909_, v_data_910_, v_ref_911_, v_msg_912_, v___y_913_, v___y_914_, v___y_915_, v___y_916_);
lean_dec(v___y_916_);
lean_dec_ref(v___y_915_);
lean_dec(v___y_914_);
lean_dec_ref(v___y_913_);
return v_res_918_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1(void){
_start:
{
lean_object* v___x_920_; lean_object* v___x_921_; 
v___x_920_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__0));
v___x_921_ = l_Lean_stringToMessageData(v___x_920_);
return v___x_921_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2(void){
_start:
{
lean_object* v___x_922_; double v___x_923_; 
v___x_922_ = lean_unsigned_to_nat(1000u);
v___x_923_ = lean_float_of_nat(v___x_922_);
return v___x_923_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6(lean_object* v_cls_924_, uint8_t v_collapsed_925_, lean_object* v_tag_926_, lean_object* v_opts_927_, uint8_t v_clsEnabled_928_, lean_object* v_oldTraces_929_, lean_object* v_msg_930_, lean_object* v_resStartStop_931_, lean_object* v___y_932_, lean_object* v___y_933_, lean_object* v___y_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_){
_start:
{
lean_object* v_fst_940_; lean_object* v_snd_941_; lean_object* v___y_943_; lean_object* v___y_944_; lean_object* v_data_945_; lean_object* v_fst_956_; lean_object* v_snd_957_; lean_object* v___x_958_; uint8_t v___x_959_; lean_object* v___y_961_; lean_object* v_a_962_; uint8_t v___y_977_; double v___y_1008_; 
v_fst_940_ = lean_ctor_get(v_resStartStop_931_, 0);
lean_inc(v_fst_940_);
v_snd_941_ = lean_ctor_get(v_resStartStop_931_, 1);
lean_inc(v_snd_941_);
lean_dec_ref(v_resStartStop_931_);
v_fst_956_ = lean_ctor_get(v_snd_941_, 0);
lean_inc(v_fst_956_);
v_snd_957_ = lean_ctor_get(v_snd_941_, 1);
lean_inc(v_snd_957_);
lean_dec(v_snd_941_);
v___x_958_ = l_Lean_trace_profiler;
v___x_959_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_927_, v___x_958_);
if (v___x_959_ == 0)
{
v___y_977_ = v___x_959_;
goto v___jp_976_;
}
else
{
lean_object* v___x_1013_; uint8_t v___x_1014_; 
v___x_1013_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1014_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_927_, v___x_1013_);
if (v___x_1014_ == 0)
{
lean_object* v___x_1015_; lean_object* v___x_1016_; double v___x_1017_; double v___x_1018_; double v___x_1019_; 
v___x_1015_ = l_Lean_trace_profiler_threshold;
v___x_1016_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_927_, v___x_1015_);
v___x_1017_ = lean_float_of_nat(v___x_1016_);
v___x_1018_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2);
v___x_1019_ = lean_float_div(v___x_1017_, v___x_1018_);
v___y_1008_ = v___x_1019_;
goto v___jp_1007_;
}
else
{
lean_object* v___x_1020_; lean_object* v___x_1021_; double v___x_1022_; 
v___x_1020_ = l_Lean_trace_profiler_threshold;
v___x_1021_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_927_, v___x_1020_);
v___x_1022_ = lean_float_of_nat(v___x_1021_);
v___y_1008_ = v___x_1022_;
goto v___jp_1007_;
}
}
v___jp_942_:
{
lean_object* v___x_946_; 
lean_inc(v___y_943_);
v___x_946_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg(v_oldTraces_929_, v_data_945_, v___y_943_, v___y_944_, v___y_935_, v___y_936_, v___y_937_, v___y_938_);
if (lean_obj_tag(v___x_946_) == 0)
{
lean_object* v___x_947_; 
lean_dec_ref_known(v___x_946_, 1);
v___x_947_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_fst_940_);
return v___x_947_;
}
else
{
lean_object* v_a_948_; lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_955_; 
lean_dec(v_fst_940_);
v_a_948_ = lean_ctor_get(v___x_946_, 0);
v_isSharedCheck_955_ = !lean_is_exclusive(v___x_946_);
if (v_isSharedCheck_955_ == 0)
{
v___x_950_ = v___x_946_;
v_isShared_951_ = v_isSharedCheck_955_;
goto v_resetjp_949_;
}
else
{
lean_inc(v_a_948_);
lean_dec(v___x_946_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_955_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v___x_953_; 
if (v_isShared_951_ == 0)
{
v___x_953_ = v___x_950_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v_a_948_);
v___x_953_ = v_reuseFailAlloc_954_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
return v___x_953_;
}
}
}
}
v___jp_960_:
{
uint8_t v_result_963_; lean_object* v___x_964_; lean_object* v___x_965_; double v___x_966_; lean_object* v_data_967_; 
v_result_963_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__10(v_fst_940_);
v___x_964_ = lean_box(v_result_963_);
v___x_965_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_965_, 0, v___x_964_);
v___x_966_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0);
lean_inc_ref(v_tag_926_);
lean_inc_ref(v___x_965_);
lean_inc(v_cls_924_);
v_data_967_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_967_, 0, v_cls_924_);
lean_ctor_set(v_data_967_, 1, v___x_965_);
lean_ctor_set(v_data_967_, 2, v_tag_926_);
lean_ctor_set_float(v_data_967_, sizeof(void*)*3, v___x_966_);
lean_ctor_set_float(v_data_967_, sizeof(void*)*3 + 8, v___x_966_);
lean_ctor_set_uint8(v_data_967_, sizeof(void*)*3 + 16, v_collapsed_925_);
if (v___x_959_ == 0)
{
lean_dec_ref_known(v___x_965_, 1);
lean_dec(v_snd_957_);
lean_dec(v_fst_956_);
lean_dec_ref(v_tag_926_);
lean_dec(v_cls_924_);
v___y_943_ = v___y_961_;
v___y_944_ = v_a_962_;
v_data_945_ = v_data_967_;
goto v___jp_942_;
}
else
{
lean_object* v_data_968_; double v___x_969_; double v___x_970_; 
lean_dec_ref_known(v_data_967_, 3);
v_data_968_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_968_, 0, v_cls_924_);
lean_ctor_set(v_data_968_, 1, v___x_965_);
lean_ctor_set(v_data_968_, 2, v_tag_926_);
v___x_969_ = lean_unbox_float(v_fst_956_);
lean_dec(v_fst_956_);
lean_ctor_set_float(v_data_968_, sizeof(void*)*3, v___x_969_);
v___x_970_ = lean_unbox_float(v_snd_957_);
lean_dec(v_snd_957_);
lean_ctor_set_float(v_data_968_, sizeof(void*)*3 + 8, v___x_970_);
lean_ctor_set_uint8(v_data_968_, sizeof(void*)*3 + 16, v_collapsed_925_);
v___y_943_ = v___y_961_;
v___y_944_ = v_a_962_;
v_data_945_ = v_data_968_;
goto v___jp_942_;
}
}
v___jp_971_:
{
lean_object* v_ref_972_; lean_object* v___x_973_; 
v_ref_972_ = lean_ctor_get(v___y_937_, 5);
lean_inc(v___y_938_);
lean_inc_ref(v___y_937_);
lean_inc(v___y_936_);
lean_inc_ref(v___y_935_);
lean_inc(v___y_934_);
lean_inc(v___y_933_);
lean_inc_ref(v___y_932_);
lean_inc(v_fst_940_);
v___x_973_ = lean_apply_9(v_msg_930_, v_fst_940_, v___y_932_, v___y_933_, v___y_934_, v___y_935_, v___y_936_, v___y_937_, v___y_938_, lean_box(0));
if (lean_obj_tag(v___x_973_) == 0)
{
lean_object* v_a_974_; 
v_a_974_ = lean_ctor_get(v___x_973_, 0);
lean_inc(v_a_974_);
lean_dec_ref_known(v___x_973_, 1);
v___y_961_ = v_ref_972_;
v_a_962_ = v_a_974_;
goto v___jp_960_;
}
else
{
lean_object* v___x_975_; 
lean_dec_ref_known(v___x_973_, 1);
v___x_975_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1);
v___y_961_ = v_ref_972_;
v_a_962_ = v___x_975_;
goto v___jp_960_;
}
}
v___jp_976_:
{
if (v_clsEnabled_928_ == 0)
{
if (v___y_977_ == 0)
{
lean_object* v___x_978_; lean_object* v_traceState_979_; lean_object* v_env_980_; lean_object* v_nextMacroScope_981_; lean_object* v_ngen_982_; lean_object* v_auxDeclNGen_983_; lean_object* v_cache_984_; lean_object* v_messages_985_; lean_object* v_infoState_986_; lean_object* v_snapshotTasks_987_; lean_object* v___x_989_; uint8_t v_isShared_990_; uint8_t v_isSharedCheck_1006_; 
lean_dec(v_snd_957_);
lean_dec(v_fst_956_);
lean_dec_ref(v_msg_930_);
lean_dec_ref(v_tag_926_);
lean_dec(v_cls_924_);
v___x_978_ = lean_st_ref_take(v___y_938_);
v_traceState_979_ = lean_ctor_get(v___x_978_, 4);
v_env_980_ = lean_ctor_get(v___x_978_, 0);
v_nextMacroScope_981_ = lean_ctor_get(v___x_978_, 1);
v_ngen_982_ = lean_ctor_get(v___x_978_, 2);
v_auxDeclNGen_983_ = lean_ctor_get(v___x_978_, 3);
v_cache_984_ = lean_ctor_get(v___x_978_, 5);
v_messages_985_ = lean_ctor_get(v___x_978_, 6);
v_infoState_986_ = lean_ctor_get(v___x_978_, 7);
v_snapshotTasks_987_ = lean_ctor_get(v___x_978_, 8);
v_isSharedCheck_1006_ = !lean_is_exclusive(v___x_978_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_989_ = v___x_978_;
v_isShared_990_ = v_isSharedCheck_1006_;
goto v_resetjp_988_;
}
else
{
lean_inc(v_snapshotTasks_987_);
lean_inc(v_infoState_986_);
lean_inc(v_messages_985_);
lean_inc(v_cache_984_);
lean_inc(v_traceState_979_);
lean_inc(v_auxDeclNGen_983_);
lean_inc(v_ngen_982_);
lean_inc(v_nextMacroScope_981_);
lean_inc(v_env_980_);
lean_dec(v___x_978_);
v___x_989_ = lean_box(0);
v_isShared_990_ = v_isSharedCheck_1006_;
goto v_resetjp_988_;
}
v_resetjp_988_:
{
uint64_t v_tid_991_; lean_object* v_traces_992_; lean_object* v___x_994_; uint8_t v_isShared_995_; uint8_t v_isSharedCheck_1005_; 
v_tid_991_ = lean_ctor_get_uint64(v_traceState_979_, sizeof(void*)*1);
v_traces_992_ = lean_ctor_get(v_traceState_979_, 0);
v_isSharedCheck_1005_ = !lean_is_exclusive(v_traceState_979_);
if (v_isSharedCheck_1005_ == 0)
{
v___x_994_ = v_traceState_979_;
v_isShared_995_ = v_isSharedCheck_1005_;
goto v_resetjp_993_;
}
else
{
lean_inc(v_traces_992_);
lean_dec(v_traceState_979_);
v___x_994_ = lean_box(0);
v_isShared_995_ = v_isSharedCheck_1005_;
goto v_resetjp_993_;
}
v_resetjp_993_:
{
lean_object* v___x_996_; lean_object* v___x_998_; 
v___x_996_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_929_, v_traces_992_);
lean_dec_ref(v_traces_992_);
if (v_isShared_995_ == 0)
{
lean_ctor_set(v___x_994_, 0, v___x_996_);
v___x_998_ = v___x_994_;
goto v_reusejp_997_;
}
else
{
lean_object* v_reuseFailAlloc_1004_; 
v_reuseFailAlloc_1004_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1004_, 0, v___x_996_);
lean_ctor_set_uint64(v_reuseFailAlloc_1004_, sizeof(void*)*1, v_tid_991_);
v___x_998_ = v_reuseFailAlloc_1004_;
goto v_reusejp_997_;
}
v_reusejp_997_:
{
lean_object* v___x_1000_; 
if (v_isShared_990_ == 0)
{
lean_ctor_set(v___x_989_, 4, v___x_998_);
v___x_1000_ = v___x_989_;
goto v_reusejp_999_;
}
else
{
lean_object* v_reuseFailAlloc_1003_; 
v_reuseFailAlloc_1003_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1003_, 0, v_env_980_);
lean_ctor_set(v_reuseFailAlloc_1003_, 1, v_nextMacroScope_981_);
lean_ctor_set(v_reuseFailAlloc_1003_, 2, v_ngen_982_);
lean_ctor_set(v_reuseFailAlloc_1003_, 3, v_auxDeclNGen_983_);
lean_ctor_set(v_reuseFailAlloc_1003_, 4, v___x_998_);
lean_ctor_set(v_reuseFailAlloc_1003_, 5, v_cache_984_);
lean_ctor_set(v_reuseFailAlloc_1003_, 6, v_messages_985_);
lean_ctor_set(v_reuseFailAlloc_1003_, 7, v_infoState_986_);
lean_ctor_set(v_reuseFailAlloc_1003_, 8, v_snapshotTasks_987_);
v___x_1000_ = v_reuseFailAlloc_1003_;
goto v_reusejp_999_;
}
v_reusejp_999_:
{
lean_object* v___x_1001_; lean_object* v___x_1002_; 
v___x_1001_ = lean_st_ref_set(v___y_938_, v___x_1000_);
v___x_1002_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_fst_940_);
return v___x_1002_;
}
}
}
}
}
else
{
goto v___jp_971_;
}
}
else
{
goto v___jp_971_;
}
}
v___jp_1007_:
{
double v___x_1009_; double v___x_1010_; double v___x_1011_; uint8_t v___x_1012_; 
v___x_1009_ = lean_unbox_float(v_snd_957_);
v___x_1010_ = lean_unbox_float(v_fst_956_);
v___x_1011_ = lean_float_sub(v___x_1009_, v___x_1010_);
v___x_1012_ = lean_float_decLt(v___y_1008_, v___x_1011_);
v___y_977_ = v___x_1012_;
goto v___jp_976_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___boxed(lean_object* v_cls_1023_, lean_object* v_collapsed_1024_, lean_object* v_tag_1025_, lean_object* v_opts_1026_, lean_object* v_clsEnabled_1027_, lean_object* v_oldTraces_1028_, lean_object* v_msg_1029_, lean_object* v_resStartStop_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_){
_start:
{
uint8_t v_collapsed_boxed_1039_; uint8_t v_clsEnabled_boxed_1040_; lean_object* v_res_1041_; 
v_collapsed_boxed_1039_ = lean_unbox(v_collapsed_1024_);
v_clsEnabled_boxed_1040_ = lean_unbox(v_clsEnabled_1027_);
v_res_1041_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6(v_cls_1023_, v_collapsed_boxed_1039_, v_tag_1025_, v_opts_1026_, v_clsEnabled_boxed_1040_, v_oldTraces_1028_, v_msg_1029_, v_resStartStop_1030_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_, v___y_1035_, v___y_1036_, v___y_1037_);
lean_dec(v___y_1037_);
lean_dec_ref(v___y_1036_);
lean_dec(v___y_1035_);
lean_dec_ref(v___y_1034_);
lean_dec(v___y_1033_);
lean_dec(v___y_1032_);
lean_dec_ref(v___y_1031_);
lean_dec_ref(v_opts_1026_);
return v_res_1041_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0(void){
_start:
{
lean_object* v_cls_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; 
v_cls_1042_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_1043_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1));
v___x_1044_ = l_Lean_Name_append(v___x_1043_, v_cls_1042_);
return v___x_1044_;
}
}
static double _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1(void){
_start:
{
lean_object* v___x_1045_; double v___x_1046_; 
v___x_1045_ = lean_unsigned_to_nat(1000000000u);
v___x_1046_ = lean_float_of_nat(v___x_1045_);
return v___x_1046_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg(lean_object* v_goal_1047_, lean_object* v_mvars_1048_, lean_object* v_preState_1049_, lean_object* v_matchResult_1050_, lean_object* v_a_1051_, lean_object* v_a_1052_, lean_object* v_a_1053_, lean_object* v_a_1054_, lean_object* v_a_1055_, lean_object* v_a_1056_, lean_object* v_a_1057_){
_start:
{
lean_object* v_rule_1062_; lean_object* v_options_1063_; lean_object* v_locations_1064_; lean_object* v_patternSubsts_x3f_1065_; lean_object* v_name_1066_; lean_object* v_tac_1067_; lean_object* v___x_1069_; uint8_t v_isShared_1070_; uint8_t v_isSharedCheck_1482_; 
v_rule_1062_ = lean_ctor_get(v_matchResult_1050_, 0);
lean_inc(v_rule_1062_);
v_options_1063_ = lean_ctor_get(v_a_1056_, 2);
v_locations_1064_ = lean_ctor_get(v_matchResult_1050_, 1);
lean_inc_ref(v_locations_1064_);
v_patternSubsts_x3f_1065_ = lean_ctor_get(v_matchResult_1050_, 2);
lean_inc(v_patternSubsts_x3f_1065_);
lean_dec_ref(v_matchResult_1050_);
v_name_1066_ = lean_ctor_get(v_rule_1062_, 0);
v_tac_1067_ = lean_ctor_get(v_rule_1062_, 4);
v_isSharedCheck_1482_ = !lean_is_exclusive(v_rule_1062_);
if (v_isSharedCheck_1482_ == 0)
{
lean_object* v_unused_1483_; lean_object* v_unused_1484_; lean_object* v_unused_1485_; 
v_unused_1483_ = lean_ctor_get(v_rule_1062_, 3);
lean_dec(v_unused_1483_);
v_unused_1484_ = lean_ctor_get(v_rule_1062_, 2);
lean_dec(v_unused_1484_);
v_unused_1485_ = lean_ctor_get(v_rule_1062_, 1);
lean_dec(v_unused_1485_);
v___x_1069_ = v_rule_1062_;
v_isShared_1070_ = v_isSharedCheck_1482_;
goto v_resetjp_1068_;
}
else
{
lean_inc(v_tac_1067_);
lean_inc(v_name_1066_);
lean_dec(v_rule_1062_);
v___x_1069_ = lean_box(0);
v_isShared_1070_ = v_isSharedCheck_1482_;
goto v_resetjp_1068_;
}
v___jp_1059_:
{
lean_object* v___x_1060_; lean_object* v___x_1061_; 
v___x_1060_ = lean_box(0);
v___x_1061_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1061_, 0, v___x_1060_);
return v___x_1061_;
}
v_resetjp_1068_:
{
lean_object* v_inheritedTraceOptions_1071_; uint8_t v_hasTrace_1072_; lean_object* v_cls_1073_; lean_object* v___x_1074_; uint8_t v_____do__lift_1076_; lean_object* v___y_1077_; lean_object* v___y_1078_; lean_object* v___y_1079_; lean_object* v___y_1080_; lean_object* v___y_1081_; lean_object* v___y_1082_; lean_object* v___y_1083_; 
v_inheritedTraceOptions_1071_ = lean_ctor_get(v_a_1056_, 13);
v_hasTrace_1072_ = lean_ctor_get_uint8(v_options_1063_, sizeof(void*)*1);
v_cls_1073_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
lean_inc_ref(v_name_1066_);
v___x_1074_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1074_, 0, v_name_1066_);
if (v_hasTrace_1072_ == 0)
{
lean_object* v___x_1203_; lean_object* v_a_1204_; uint8_t v___x_1205_; lean_object* v___x_1206_; lean_object* v_a_1207_; uint8_t v___x_1208_; 
v___x_1203_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
v_a_1204_ = lean_ctor_get(v___x_1203_, 0);
lean_inc(v_a_1204_);
lean_dec_ref(v___x_1203_);
v___x_1205_ = lean_unbox(v_a_1204_);
lean_dec(v_a_1204_);
v___x_1206_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(v___x_1205_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
v_a_1207_ = lean_ctor_get(v___x_1206_, 0);
lean_inc(v_a_1207_);
lean_dec_ref(v___x_1206_);
v___x_1208_ = lean_unbox(v_a_1207_);
lean_dec(v_a_1207_);
v_____do__lift_1076_ = v___x_1208_;
v___y_1077_ = v_a_1051_;
v___y_1078_ = v_a_1052_;
v___y_1079_ = v_a_1053_;
v___y_1080_ = v_a_1054_;
v___y_1081_ = v_a_1055_;
v___y_1082_ = v_a_1056_;
v___y_1083_ = v_a_1057_;
goto v___jp_1075_;
}
else
{
lean_object* v___x_1209_; lean_object* v___f_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; uint8_t v___x_1213_; lean_object* v___y_1215_; lean_object* v___y_1216_; lean_object* v_a_1217_; lean_object* v___y_1230_; lean_object* v___y_1231_; lean_object* v_a_1232_; lean_object* v___y_1235_; lean_object* v___y_1236_; lean_object* v_a_1237_; lean_object* v___y_1240_; lean_object* v___y_1241_; lean_object* v___y_1280_; lean_object* v___y_1281_; lean_object* v___y_1282_; lean_object* v___y_1285_; lean_object* v___y_1286_; lean_object* v___y_1326_; lean_object* v___y_1327_; lean_object* v___y_1328_; lean_object* v___y_1332_; lean_object* v___y_1333_; lean_object* v_a_1334_; lean_object* v___y_1344_; lean_object* v___y_1345_; lean_object* v_a_1346_; lean_object* v___y_1349_; lean_object* v___y_1350_; lean_object* v_a_1351_; lean_object* v___y_1354_; lean_object* v___y_1355_; lean_object* v___y_1394_; lean_object* v___y_1395_; lean_object* v___y_1396_; lean_object* v___y_1399_; lean_object* v___y_1400_; lean_object* v___y_1440_; lean_object* v___y_1441_; uint8_t v_a_1442_; lean_object* v___y_1444_; lean_object* v___y_1445_; lean_object* v___y_1446_; 
v___x_1209_ = lean_box(v_hasTrace_1072_);
lean_inc_ref(v_name_1066_);
v___f_1210_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___boxed), 11, 2);
lean_closure_set(v___f_1210_, 0, v_name_1066_);
lean_closure_set(v___f_1210_, 1, v___x_1209_);
v___x_1211_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_1212_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0);
v___x_1213_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1071_, v_options_1063_, v___x_1212_);
if (v___x_1213_ == 0)
{
lean_object* v___x_1474_; uint8_t v___x_1475_; 
v___x_1474_ = l_Lean_trace_profiler;
v___x_1475_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_1063_, v___x_1474_);
if (v___x_1475_ == 0)
{
lean_object* v___x_1476_; lean_object* v_a_1477_; uint8_t v___x_1478_; lean_object* v___x_1479_; lean_object* v_a_1480_; uint8_t v___x_1481_; 
lean_dec_ref(v___f_1210_);
v___x_1476_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
v_a_1477_ = lean_ctor_get(v___x_1476_, 0);
lean_inc(v_a_1477_);
lean_dec_ref(v___x_1476_);
v___x_1478_ = lean_unbox(v_a_1477_);
lean_dec(v_a_1477_);
v___x_1479_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(v___x_1478_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
v_a_1480_ = lean_ctor_get(v___x_1479_, 0);
lean_inc(v_a_1480_);
lean_dec_ref(v___x_1479_);
v___x_1481_ = lean_unbox(v_a_1480_);
lean_dec(v_a_1480_);
v_____do__lift_1076_ = v___x_1481_;
v___y_1077_ = v_a_1051_;
v___y_1078_ = v_a_1052_;
v___y_1079_ = v_a_1053_;
v___y_1080_ = v_a_1054_;
v___y_1081_ = v_a_1055_;
v___y_1082_ = v_a_1056_;
v___y_1083_ = v_a_1057_;
goto v___jp_1075_;
}
else
{
lean_del_object(v___x_1069_);
goto v___jp_1449_;
}
}
else
{
lean_del_object(v___x_1069_);
goto v___jp_1449_;
}
v___jp_1214_:
{
lean_object* v___x_1218_; double v___x_1219_; double v___x_1220_; double v___x_1221_; double v___x_1222_; double v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; 
v___x_1218_ = lean_io_mono_nanos_now();
v___x_1219_ = lean_float_of_nat(v___y_1216_);
v___x_1220_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_1221_ = lean_float_div(v___x_1219_, v___x_1220_);
v___x_1222_ = lean_float_of_nat(v___x_1218_);
v___x_1223_ = lean_float_div(v___x_1222_, v___x_1220_);
v___x_1224_ = lean_box_float(v___x_1221_);
v___x_1225_ = lean_box_float(v___x_1223_);
v___x_1226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1226_, 0, v___x_1224_);
lean_ctor_set(v___x_1226_, 1, v___x_1225_);
v___x_1227_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1227_, 0, v_a_1217_);
lean_ctor_set(v___x_1227_, 1, v___x_1226_);
v___x_1228_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6(v_cls_1073_, v_hasTrace_1072_, v___x_1211_, v_options_1063_, v___x_1213_, v___y_1215_, v___f_1210_, v___x_1227_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
return v___x_1228_;
}
v___jp_1229_:
{
lean_object* v___x_1233_; 
v___x_1233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1233_, 0, v_a_1232_);
v___y_1215_ = v___y_1230_;
v___y_1216_ = v___y_1231_;
v_a_1217_ = v___x_1233_;
goto v___jp_1214_;
}
v___jp_1234_:
{
lean_object* v___x_1238_; 
v___x_1238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1238_, 0, v_a_1237_);
v___y_1215_ = v___y_1235_;
v___y_1216_ = v___y_1236_;
v_a_1217_ = v___x_1238_;
goto v___jp_1214_;
}
v___jp_1239_:
{
lean_object* v___x_1242_; lean_object* v___x_1243_; 
v___x_1242_ = lean_io_mono_nanos_now();
lean_inc_ref(v_a_1051_);
v___x_1243_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3(v_goal_1047_, v_mvars_1048_, v_locations_1064_, v_patternSubsts_x3f_1065_, v_tac_1067_, v_name_1066_, v_preState_1049_, v_cls_1073_, v_a_1051_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
if (lean_obj_tag(v___x_1243_) == 0)
{
lean_object* v_a_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v_stats_1247_; lean_object* v_rulePatternCache_1248_; lean_object* v___x_1250_; uint8_t v_isShared_1251_; uint8_t v_isSharedCheck_1277_; 
v_a_1244_ = lean_ctor_get(v___x_1243_, 0);
lean_inc(v_a_1244_);
lean_dec_ref_known(v___x_1243_, 1);
v___x_1245_ = lean_io_mono_nanos_now();
v___x_1246_ = lean_st_ref_take(v_a_1053_);
v_stats_1247_ = lean_ctor_get(v___x_1246_, 1);
v_rulePatternCache_1248_ = lean_ctor_get(v___x_1246_, 0);
v_isSharedCheck_1277_ = !lean_is_exclusive(v___x_1246_);
if (v_isSharedCheck_1277_ == 0)
{
v___x_1250_ = v___x_1246_;
v_isShared_1251_ = v_isSharedCheck_1277_;
goto v_resetjp_1249_;
}
else
{
lean_inc(v_stats_1247_);
lean_inc(v_rulePatternCache_1248_);
lean_dec(v___x_1246_);
v___x_1250_ = lean_box(0);
v_isShared_1251_ = v_isSharedCheck_1277_;
goto v_resetjp_1249_;
}
v_resetjp_1249_:
{
lean_object* v_total_1252_; lean_object* v_configParsing_1253_; lean_object* v_ruleSetConstruction_1254_; lean_object* v_search_1255_; lean_object* v_ruleSelection_1256_; lean_object* v_script_1257_; lean_object* v_forwardState_1258_; lean_object* v_scriptGenerated_1259_; lean_object* v_ruleStats_1260_; lean_object* v_goalStats_1261_; lean_object* v___x_1263_; uint8_t v_isShared_1264_; uint8_t v_isSharedCheck_1276_; 
v_total_1252_ = lean_ctor_get(v_stats_1247_, 0);
v_configParsing_1253_ = lean_ctor_get(v_stats_1247_, 1);
v_ruleSetConstruction_1254_ = lean_ctor_get(v_stats_1247_, 2);
v_search_1255_ = lean_ctor_get(v_stats_1247_, 3);
v_ruleSelection_1256_ = lean_ctor_get(v_stats_1247_, 4);
v_script_1257_ = lean_ctor_get(v_stats_1247_, 5);
v_forwardState_1258_ = lean_ctor_get(v_stats_1247_, 6);
v_scriptGenerated_1259_ = lean_ctor_get(v_stats_1247_, 7);
v_ruleStats_1260_ = lean_ctor_get(v_stats_1247_, 8);
v_goalStats_1261_ = lean_ctor_get(v_stats_1247_, 9);
v_isSharedCheck_1276_ = !lean_is_exclusive(v_stats_1247_);
if (v_isSharedCheck_1276_ == 0)
{
v___x_1263_ = v_stats_1247_;
v_isShared_1264_ = v_isSharedCheck_1276_;
goto v_resetjp_1262_;
}
else
{
lean_inc(v_goalStats_1261_);
lean_inc(v_ruleStats_1260_);
lean_inc(v_scriptGenerated_1259_);
lean_inc(v_forwardState_1258_);
lean_inc(v_script_1257_);
lean_inc(v_ruleSelection_1256_);
lean_inc(v_search_1255_);
lean_inc(v_ruleSetConstruction_1254_);
lean_inc(v_configParsing_1253_);
lean_inc(v_total_1252_);
lean_dec(v_stats_1247_);
v___x_1263_ = lean_box(0);
v_isShared_1264_ = v_isSharedCheck_1276_;
goto v_resetjp_1262_;
}
v_resetjp_1262_:
{
lean_object* v___x_1265_; uint8_t v___x_1266_; lean_object* v_rp_1267_; lean_object* v___x_1268_; lean_object* v___x_1270_; 
v___x_1265_ = lean_nat_sub(v___x_1245_, v___x_1242_);
lean_dec(v___x_1242_);
lean_dec(v___x_1245_);
v___x_1266_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__2(v_a_1244_);
v_rp_1267_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_rp_1267_, 0, v___x_1074_);
lean_ctor_set(v_rp_1267_, 1, v___x_1265_);
lean_ctor_set_uint8(v_rp_1267_, sizeof(void*)*2, v___x_1266_);
v___x_1268_ = lean_array_push(v_ruleStats_1260_, v_rp_1267_);
if (v_isShared_1264_ == 0)
{
lean_ctor_set(v___x_1263_, 8, v___x_1268_);
v___x_1270_ = v___x_1263_;
goto v_reusejp_1269_;
}
else
{
lean_object* v_reuseFailAlloc_1275_; 
v_reuseFailAlloc_1275_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1275_, 0, v_total_1252_);
lean_ctor_set(v_reuseFailAlloc_1275_, 1, v_configParsing_1253_);
lean_ctor_set(v_reuseFailAlloc_1275_, 2, v_ruleSetConstruction_1254_);
lean_ctor_set(v_reuseFailAlloc_1275_, 3, v_search_1255_);
lean_ctor_set(v_reuseFailAlloc_1275_, 4, v_ruleSelection_1256_);
lean_ctor_set(v_reuseFailAlloc_1275_, 5, v_script_1257_);
lean_ctor_set(v_reuseFailAlloc_1275_, 6, v_forwardState_1258_);
lean_ctor_set(v_reuseFailAlloc_1275_, 7, v_scriptGenerated_1259_);
lean_ctor_set(v_reuseFailAlloc_1275_, 8, v___x_1268_);
lean_ctor_set(v_reuseFailAlloc_1275_, 9, v_goalStats_1261_);
v___x_1270_ = v_reuseFailAlloc_1275_;
goto v_reusejp_1269_;
}
v_reusejp_1269_:
{
lean_object* v___x_1272_; 
if (v_isShared_1251_ == 0)
{
lean_ctor_set(v___x_1250_, 1, v___x_1270_);
v___x_1272_ = v___x_1250_;
goto v_reusejp_1271_;
}
else
{
lean_object* v_reuseFailAlloc_1274_; 
v_reuseFailAlloc_1274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1274_, 0, v_rulePatternCache_1248_);
lean_ctor_set(v_reuseFailAlloc_1274_, 1, v___x_1270_);
v___x_1272_ = v_reuseFailAlloc_1274_;
goto v_reusejp_1271_;
}
v_reusejp_1271_:
{
lean_object* v___x_1273_; 
v___x_1273_ = lean_st_ref_set(v_a_1053_, v___x_1272_);
v___y_1230_ = v___y_1240_;
v___y_1231_ = v___y_1241_;
v_a_1232_ = v_a_1244_;
goto v___jp_1229_;
}
}
}
}
}
else
{
lean_object* v_a_1278_; 
lean_dec(v___x_1242_);
lean_dec_ref_known(v___x_1074_, 1);
v_a_1278_ = lean_ctor_get(v___x_1243_, 0);
lean_inc(v_a_1278_);
lean_dec_ref_known(v___x_1243_, 1);
v___y_1235_ = v___y_1240_;
v___y_1236_ = v___y_1241_;
v_a_1237_ = v_a_1278_;
goto v___jp_1234_;
}
}
v___jp_1279_:
{
lean_object* v_a_1283_; 
v_a_1283_ = lean_ctor_get(v___y_1282_, 0);
lean_inc(v_a_1283_);
lean_dec_ref(v___y_1282_);
v___y_1230_ = v___y_1280_;
v___y_1231_ = v___y_1281_;
v_a_1232_ = v_a_1283_;
goto v___jp_1229_;
}
v___jp_1284_:
{
lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; 
lean_inc_ref(v_a_1051_);
v___x_1287_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1287_, 0, v_goal_1047_);
lean_ctor_set(v___x_1287_, 1, v_mvars_1048_);
lean_ctor_set(v___x_1287_, 2, v_locations_1064_);
lean_ctor_set(v___x_1287_, 3, v_patternSubsts_x3f_1065_);
lean_ctor_set(v___x_1287_, 4, v_a_1051_);
v___x_1288_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTacDescr_run___boxed), 8, 1);
lean_closure_set(v___x_1288_, 0, v_tac_1067_);
v___x_1289_ = lp_aesop_Aesop_runRuleTac(v___x_1288_, v_name_1066_, v_preState_1049_, v___x_1287_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
if (lean_obj_tag(v___x_1289_) == 0)
{
lean_object* v_a_1290_; 
v_a_1290_ = lean_ctor_get(v___x_1289_, 0);
lean_inc(v_a_1290_);
lean_dec_ref_known(v___x_1289_, 1);
if (lean_obj_tag(v_a_1290_) == 0)
{
if (v___x_1213_ == 0)
{
lean_object* v___x_1291_; lean_object* v___x_1292_; 
lean_dec_ref_known(v_a_1290_, 1);
v___x_1291_ = lean_box(0);
v___x_1292_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5(v___x_1291_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
v___y_1280_ = v___y_1285_;
v___y_1281_ = v___y_1286_;
v___y_1282_ = v___x_1292_;
goto v___jp_1279_;
}
else
{
lean_object* v_a_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; 
v_a_1293_ = lean_ctor_get(v_a_1290_, 0);
lean_inc(v_a_1293_);
lean_dec_ref_known(v_a_1290_, 1);
v___x_1294_ = l_Lean_Exception_toMessageData(v_a_1293_);
v___x_1295_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v_cls_1073_, v___x_1294_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
if (lean_obj_tag(v___x_1295_) == 0)
{
lean_object* v_a_1296_; lean_object* v___x_1297_; 
v_a_1296_ = lean_ctor_get(v___x_1295_, 0);
lean_inc(v_a_1296_);
lean_dec_ref_known(v___x_1295_, 1);
v___x_1297_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5(v_a_1296_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
v___y_1280_ = v___y_1285_;
v___y_1281_ = v___y_1286_;
v___y_1282_ = v___x_1297_;
goto v___jp_1279_;
}
else
{
lean_object* v_a_1298_; 
v_a_1298_ = lean_ctor_get(v___x_1295_, 0);
lean_inc(v_a_1298_);
lean_dec_ref_known(v___x_1295_, 1);
v___y_1235_ = v___y_1285_;
v___y_1236_ = v___y_1286_;
v_a_1237_ = v_a_1298_;
goto v___jp_1234_;
}
}
}
else
{
lean_object* v_a_1299_; lean_object* v___x_1300_; 
v_a_1299_ = lean_ctor_get(v_a_1290_, 0);
lean_inc(v_a_1299_);
lean_dec_ref_known(v_a_1290_, 1);
v___x_1300_ = lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1(v_a_1299_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
lean_dec(v_a_1299_);
if (lean_obj_tag(v___x_1300_) == 0)
{
lean_object* v_a_1301_; lean_object* v_snd_1302_; lean_object* v_fst_1303_; lean_object* v_fst_1304_; lean_object* v_snd_1305_; lean_object* v___x_1307_; uint8_t v_isShared_1308_; uint8_t v_isSharedCheck_1322_; 
v_a_1301_ = lean_ctor_get(v___x_1300_, 0);
lean_inc(v_a_1301_);
lean_dec_ref_known(v___x_1300_, 1);
v_snd_1302_ = lean_ctor_get(v_a_1301_, 1);
lean_inc(v_snd_1302_);
v_fst_1303_ = lean_ctor_get(v_a_1301_, 0);
lean_inc(v_fst_1303_);
lean_dec(v_a_1301_);
v_fst_1304_ = lean_ctor_get(v_snd_1302_, 0);
v_snd_1305_ = lean_ctor_get(v_snd_1302_, 1);
v_isSharedCheck_1322_ = !lean_is_exclusive(v_snd_1302_);
if (v_isSharedCheck_1322_ == 0)
{
v___x_1307_ = v_snd_1302_;
v_isShared_1308_ = v_isSharedCheck_1322_;
goto v_resetjp_1306_;
}
else
{
lean_inc(v_snd_1305_);
lean_inc(v_fst_1304_);
lean_dec(v_snd_1302_);
v___x_1307_ = lean_box(0);
v_isShared_1308_ = v_isSharedCheck_1322_;
goto v_resetjp_1306_;
}
v_resetjp_1306_:
{
lean_object* v___x_1309_; 
v___x_1309_ = l_Lean_Meta_SavedState_restore___redArg(v_fst_1304_, v_a_1055_, v_a_1057_);
lean_dec(v_fst_1304_);
if (lean_obj_tag(v___x_1309_) == 0)
{
lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1319_; 
v_isSharedCheck_1319_ = !lean_is_exclusive(v___x_1309_);
if (v_isSharedCheck_1319_ == 0)
{
lean_object* v_unused_1320_; 
v_unused_1320_ = lean_ctor_get(v___x_1309_, 0);
lean_dec(v_unused_1320_);
v___x_1311_ = v___x_1309_;
v_isShared_1312_ = v_isSharedCheck_1319_;
goto v_resetjp_1310_;
}
else
{
lean_dec(v___x_1309_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1319_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
lean_object* v___x_1314_; 
if (v_isShared_1308_ == 0)
{
lean_ctor_set(v___x_1307_, 0, v_fst_1303_);
v___x_1314_ = v___x_1307_;
goto v_reusejp_1313_;
}
else
{
lean_object* v_reuseFailAlloc_1318_; 
v_reuseFailAlloc_1318_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1318_, 0, v_fst_1303_);
lean_ctor_set(v_reuseFailAlloc_1318_, 1, v_snd_1305_);
v___x_1314_ = v_reuseFailAlloc_1318_;
goto v_reusejp_1313_;
}
v_reusejp_1313_:
{
lean_object* v___x_1316_; 
if (v_isShared_1312_ == 0)
{
lean_ctor_set_tag(v___x_1311_, 1);
lean_ctor_set(v___x_1311_, 0, v___x_1314_);
v___x_1316_ = v___x_1311_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v___x_1314_);
v___x_1316_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1315_;
}
v_reusejp_1315_:
{
v___y_1230_ = v___y_1285_;
v___y_1231_ = v___y_1286_;
v_a_1232_ = v___x_1316_;
goto v___jp_1229_;
}
}
}
}
else
{
lean_object* v_a_1321_; 
lean_del_object(v___x_1307_);
lean_dec(v_snd_1305_);
lean_dec(v_fst_1303_);
v_a_1321_ = lean_ctor_get(v___x_1309_, 0);
lean_inc(v_a_1321_);
lean_dec_ref_known(v___x_1309_, 1);
v___y_1235_ = v___y_1285_;
v___y_1236_ = v___y_1286_;
v_a_1237_ = v_a_1321_;
goto v___jp_1234_;
}
}
}
else
{
lean_object* v_a_1323_; 
v_a_1323_ = lean_ctor_get(v___x_1300_, 0);
lean_inc(v_a_1323_);
lean_dec_ref_known(v___x_1300_, 1);
v___y_1235_ = v___y_1285_;
v___y_1236_ = v___y_1286_;
v_a_1237_ = v_a_1323_;
goto v___jp_1234_;
}
}
}
else
{
lean_object* v_a_1324_; 
v_a_1324_ = lean_ctor_get(v___x_1289_, 0);
lean_inc(v_a_1324_);
lean_dec_ref_known(v___x_1289_, 1);
v___y_1235_ = v___y_1285_;
v___y_1236_ = v___y_1286_;
v_a_1237_ = v_a_1324_;
goto v___jp_1234_;
}
}
v___jp_1325_:
{
lean_object* v_a_1329_; uint8_t v___x_1330_; 
v_a_1329_ = lean_ctor_get(v___y_1328_, 0);
lean_inc(v_a_1329_);
lean_dec_ref(v___y_1328_);
v___x_1330_ = lean_unbox(v_a_1329_);
lean_dec(v_a_1329_);
if (v___x_1330_ == 0)
{
lean_dec_ref_known(v___x_1074_, 1);
v___y_1285_ = v___y_1326_;
v___y_1286_ = v___y_1327_;
goto v___jp_1284_;
}
else
{
v___y_1240_ = v___y_1326_;
v___y_1241_ = v___y_1327_;
goto v___jp_1239_;
}
}
v___jp_1331_:
{
lean_object* v___x_1335_; double v___x_1336_; double v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; 
v___x_1335_ = lean_io_get_num_heartbeats();
v___x_1336_ = lean_float_of_nat(v___y_1333_);
v___x_1337_ = lean_float_of_nat(v___x_1335_);
v___x_1338_ = lean_box_float(v___x_1336_);
v___x_1339_ = lean_box_float(v___x_1337_);
v___x_1340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1340_, 0, v___x_1338_);
lean_ctor_set(v___x_1340_, 1, v___x_1339_);
v___x_1341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1341_, 0, v_a_1334_);
lean_ctor_set(v___x_1341_, 1, v___x_1340_);
v___x_1342_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6(v_cls_1073_, v_hasTrace_1072_, v___x_1211_, v_options_1063_, v___x_1213_, v___y_1332_, v___f_1210_, v___x_1341_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
return v___x_1342_;
}
v___jp_1343_:
{
lean_object* v___x_1347_; 
v___x_1347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1347_, 0, v_a_1346_);
v___y_1332_ = v___y_1345_;
v___y_1333_ = v___y_1344_;
v_a_1334_ = v___x_1347_;
goto v___jp_1331_;
}
v___jp_1348_:
{
lean_object* v___x_1352_; 
v___x_1352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1352_, 0, v_a_1351_);
v___y_1332_ = v___y_1350_;
v___y_1333_ = v___y_1349_;
v_a_1334_ = v___x_1352_;
goto v___jp_1331_;
}
v___jp_1353_:
{
lean_object* v___x_1356_; lean_object* v___x_1357_; 
v___x_1356_ = lean_io_mono_nanos_now();
lean_inc_ref(v_a_1051_);
v___x_1357_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3(v_goal_1047_, v_mvars_1048_, v_locations_1064_, v_patternSubsts_x3f_1065_, v_tac_1067_, v_name_1066_, v_preState_1049_, v_cls_1073_, v_a_1051_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
if (lean_obj_tag(v___x_1357_) == 0)
{
lean_object* v_a_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v_stats_1361_; lean_object* v_rulePatternCache_1362_; lean_object* v___x_1364_; uint8_t v_isShared_1365_; uint8_t v_isSharedCheck_1391_; 
v_a_1358_ = lean_ctor_get(v___x_1357_, 0);
lean_inc(v_a_1358_);
lean_dec_ref_known(v___x_1357_, 1);
v___x_1359_ = lean_io_mono_nanos_now();
v___x_1360_ = lean_st_ref_take(v_a_1053_);
v_stats_1361_ = lean_ctor_get(v___x_1360_, 1);
v_rulePatternCache_1362_ = lean_ctor_get(v___x_1360_, 0);
v_isSharedCheck_1391_ = !lean_is_exclusive(v___x_1360_);
if (v_isSharedCheck_1391_ == 0)
{
v___x_1364_ = v___x_1360_;
v_isShared_1365_ = v_isSharedCheck_1391_;
goto v_resetjp_1363_;
}
else
{
lean_inc(v_stats_1361_);
lean_inc(v_rulePatternCache_1362_);
lean_dec(v___x_1360_);
v___x_1364_ = lean_box(0);
v_isShared_1365_ = v_isSharedCheck_1391_;
goto v_resetjp_1363_;
}
v_resetjp_1363_:
{
lean_object* v_total_1366_; lean_object* v_configParsing_1367_; lean_object* v_ruleSetConstruction_1368_; lean_object* v_search_1369_; lean_object* v_ruleSelection_1370_; lean_object* v_script_1371_; lean_object* v_forwardState_1372_; lean_object* v_scriptGenerated_1373_; lean_object* v_ruleStats_1374_; lean_object* v_goalStats_1375_; lean_object* v___x_1377_; uint8_t v_isShared_1378_; uint8_t v_isSharedCheck_1390_; 
v_total_1366_ = lean_ctor_get(v_stats_1361_, 0);
v_configParsing_1367_ = lean_ctor_get(v_stats_1361_, 1);
v_ruleSetConstruction_1368_ = lean_ctor_get(v_stats_1361_, 2);
v_search_1369_ = lean_ctor_get(v_stats_1361_, 3);
v_ruleSelection_1370_ = lean_ctor_get(v_stats_1361_, 4);
v_script_1371_ = lean_ctor_get(v_stats_1361_, 5);
v_forwardState_1372_ = lean_ctor_get(v_stats_1361_, 6);
v_scriptGenerated_1373_ = lean_ctor_get(v_stats_1361_, 7);
v_ruleStats_1374_ = lean_ctor_get(v_stats_1361_, 8);
v_goalStats_1375_ = lean_ctor_get(v_stats_1361_, 9);
v_isSharedCheck_1390_ = !lean_is_exclusive(v_stats_1361_);
if (v_isSharedCheck_1390_ == 0)
{
v___x_1377_ = v_stats_1361_;
v_isShared_1378_ = v_isSharedCheck_1390_;
goto v_resetjp_1376_;
}
else
{
lean_inc(v_goalStats_1375_);
lean_inc(v_ruleStats_1374_);
lean_inc(v_scriptGenerated_1373_);
lean_inc(v_forwardState_1372_);
lean_inc(v_script_1371_);
lean_inc(v_ruleSelection_1370_);
lean_inc(v_search_1369_);
lean_inc(v_ruleSetConstruction_1368_);
lean_inc(v_configParsing_1367_);
lean_inc(v_total_1366_);
lean_dec(v_stats_1361_);
v___x_1377_ = lean_box(0);
v_isShared_1378_ = v_isSharedCheck_1390_;
goto v_resetjp_1376_;
}
v_resetjp_1376_:
{
lean_object* v___x_1379_; uint8_t v___x_1380_; lean_object* v_rp_1381_; lean_object* v___x_1382_; lean_object* v___x_1384_; 
v___x_1379_ = lean_nat_sub(v___x_1359_, v___x_1356_);
lean_dec(v___x_1356_);
lean_dec(v___x_1359_);
v___x_1380_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__2(v_a_1358_);
v_rp_1381_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_rp_1381_, 0, v___x_1074_);
lean_ctor_set(v_rp_1381_, 1, v___x_1379_);
lean_ctor_set_uint8(v_rp_1381_, sizeof(void*)*2, v___x_1380_);
v___x_1382_ = lean_array_push(v_ruleStats_1374_, v_rp_1381_);
if (v_isShared_1378_ == 0)
{
lean_ctor_set(v___x_1377_, 8, v___x_1382_);
v___x_1384_ = v___x_1377_;
goto v_reusejp_1383_;
}
else
{
lean_object* v_reuseFailAlloc_1389_; 
v_reuseFailAlloc_1389_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1389_, 0, v_total_1366_);
lean_ctor_set(v_reuseFailAlloc_1389_, 1, v_configParsing_1367_);
lean_ctor_set(v_reuseFailAlloc_1389_, 2, v_ruleSetConstruction_1368_);
lean_ctor_set(v_reuseFailAlloc_1389_, 3, v_search_1369_);
lean_ctor_set(v_reuseFailAlloc_1389_, 4, v_ruleSelection_1370_);
lean_ctor_set(v_reuseFailAlloc_1389_, 5, v_script_1371_);
lean_ctor_set(v_reuseFailAlloc_1389_, 6, v_forwardState_1372_);
lean_ctor_set(v_reuseFailAlloc_1389_, 7, v_scriptGenerated_1373_);
lean_ctor_set(v_reuseFailAlloc_1389_, 8, v___x_1382_);
lean_ctor_set(v_reuseFailAlloc_1389_, 9, v_goalStats_1375_);
v___x_1384_ = v_reuseFailAlloc_1389_;
goto v_reusejp_1383_;
}
v_reusejp_1383_:
{
lean_object* v___x_1386_; 
if (v_isShared_1365_ == 0)
{
lean_ctor_set(v___x_1364_, 1, v___x_1384_);
v___x_1386_ = v___x_1364_;
goto v_reusejp_1385_;
}
else
{
lean_object* v_reuseFailAlloc_1388_; 
v_reuseFailAlloc_1388_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1388_, 0, v_rulePatternCache_1362_);
lean_ctor_set(v_reuseFailAlloc_1388_, 1, v___x_1384_);
v___x_1386_ = v_reuseFailAlloc_1388_;
goto v_reusejp_1385_;
}
v_reusejp_1385_:
{
lean_object* v___x_1387_; 
v___x_1387_ = lean_st_ref_set(v_a_1053_, v___x_1386_);
v___y_1344_ = v___y_1355_;
v___y_1345_ = v___y_1354_;
v_a_1346_ = v_a_1358_;
goto v___jp_1343_;
}
}
}
}
}
else
{
lean_object* v_a_1392_; 
lean_dec(v___x_1356_);
lean_dec_ref_known(v___x_1074_, 1);
v_a_1392_ = lean_ctor_get(v___x_1357_, 0);
lean_inc(v_a_1392_);
lean_dec_ref_known(v___x_1357_, 1);
v___y_1349_ = v___y_1355_;
v___y_1350_ = v___y_1354_;
v_a_1351_ = v_a_1392_;
goto v___jp_1348_;
}
}
v___jp_1393_:
{
lean_object* v_a_1397_; 
v_a_1397_ = lean_ctor_get(v___y_1396_, 0);
lean_inc(v_a_1397_);
lean_dec_ref(v___y_1396_);
v___y_1344_ = v___y_1395_;
v___y_1345_ = v___y_1394_;
v_a_1346_ = v_a_1397_;
goto v___jp_1343_;
}
v___jp_1398_:
{
lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; 
lean_inc_ref(v_a_1051_);
v___x_1401_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1401_, 0, v_goal_1047_);
lean_ctor_set(v___x_1401_, 1, v_mvars_1048_);
lean_ctor_set(v___x_1401_, 2, v_locations_1064_);
lean_ctor_set(v___x_1401_, 3, v_patternSubsts_x3f_1065_);
lean_ctor_set(v___x_1401_, 4, v_a_1051_);
v___x_1402_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTacDescr_run___boxed), 8, 1);
lean_closure_set(v___x_1402_, 0, v_tac_1067_);
v___x_1403_ = lp_aesop_Aesop_runRuleTac(v___x_1402_, v_name_1066_, v_preState_1049_, v___x_1401_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
if (lean_obj_tag(v___x_1403_) == 0)
{
lean_object* v_a_1404_; 
v_a_1404_ = lean_ctor_get(v___x_1403_, 0);
lean_inc(v_a_1404_);
lean_dec_ref_known(v___x_1403_, 1);
if (lean_obj_tag(v_a_1404_) == 0)
{
if (v___x_1213_ == 0)
{
lean_object* v___x_1405_; lean_object* v___x_1406_; 
lean_dec_ref_known(v_a_1404_, 1);
v___x_1405_ = lean_box(0);
v___x_1406_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5(v___x_1405_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
v___y_1394_ = v___y_1400_;
v___y_1395_ = v___y_1399_;
v___y_1396_ = v___x_1406_;
goto v___jp_1393_;
}
else
{
lean_object* v_a_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; 
v_a_1407_ = lean_ctor_get(v_a_1404_, 0);
lean_inc(v_a_1407_);
lean_dec_ref_known(v_a_1404_, 1);
v___x_1408_ = l_Lean_Exception_toMessageData(v_a_1407_);
v___x_1409_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v_cls_1073_, v___x_1408_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
if (lean_obj_tag(v___x_1409_) == 0)
{
lean_object* v_a_1410_; lean_object* v___x_1411_; 
v_a_1410_ = lean_ctor_get(v___x_1409_, 0);
lean_inc(v_a_1410_);
lean_dec_ref_known(v___x_1409_, 1);
v___x_1411_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__5(v_a_1410_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
v___y_1394_ = v___y_1400_;
v___y_1395_ = v___y_1399_;
v___y_1396_ = v___x_1411_;
goto v___jp_1393_;
}
else
{
lean_object* v_a_1412_; 
v_a_1412_ = lean_ctor_get(v___x_1409_, 0);
lean_inc(v_a_1412_);
lean_dec_ref_known(v___x_1409_, 1);
v___y_1349_ = v___y_1399_;
v___y_1350_ = v___y_1400_;
v_a_1351_ = v_a_1412_;
goto v___jp_1348_;
}
}
}
else
{
lean_object* v_a_1413_; lean_object* v___x_1414_; 
v_a_1413_ = lean_ctor_get(v_a_1404_, 0);
lean_inc(v_a_1413_);
lean_dec_ref_known(v_a_1404_, 1);
v___x_1414_ = lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1(v_a_1413_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_);
lean_dec(v_a_1413_);
if (lean_obj_tag(v___x_1414_) == 0)
{
lean_object* v_a_1415_; lean_object* v_snd_1416_; lean_object* v_fst_1417_; lean_object* v_fst_1418_; lean_object* v_snd_1419_; lean_object* v___x_1421_; uint8_t v_isShared_1422_; uint8_t v_isSharedCheck_1436_; 
v_a_1415_ = lean_ctor_get(v___x_1414_, 0);
lean_inc(v_a_1415_);
lean_dec_ref_known(v___x_1414_, 1);
v_snd_1416_ = lean_ctor_get(v_a_1415_, 1);
lean_inc(v_snd_1416_);
v_fst_1417_ = lean_ctor_get(v_a_1415_, 0);
lean_inc(v_fst_1417_);
lean_dec(v_a_1415_);
v_fst_1418_ = lean_ctor_get(v_snd_1416_, 0);
v_snd_1419_ = lean_ctor_get(v_snd_1416_, 1);
v_isSharedCheck_1436_ = !lean_is_exclusive(v_snd_1416_);
if (v_isSharedCheck_1436_ == 0)
{
v___x_1421_ = v_snd_1416_;
v_isShared_1422_ = v_isSharedCheck_1436_;
goto v_resetjp_1420_;
}
else
{
lean_inc(v_snd_1419_);
lean_inc(v_fst_1418_);
lean_dec(v_snd_1416_);
v___x_1421_ = lean_box(0);
v_isShared_1422_ = v_isSharedCheck_1436_;
goto v_resetjp_1420_;
}
v_resetjp_1420_:
{
lean_object* v___x_1423_; 
v___x_1423_ = l_Lean_Meta_SavedState_restore___redArg(v_fst_1418_, v_a_1055_, v_a_1057_);
lean_dec(v_fst_1418_);
if (lean_obj_tag(v___x_1423_) == 0)
{
lean_object* v___x_1425_; uint8_t v_isShared_1426_; uint8_t v_isSharedCheck_1433_; 
v_isSharedCheck_1433_ = !lean_is_exclusive(v___x_1423_);
if (v_isSharedCheck_1433_ == 0)
{
lean_object* v_unused_1434_; 
v_unused_1434_ = lean_ctor_get(v___x_1423_, 0);
lean_dec(v_unused_1434_);
v___x_1425_ = v___x_1423_;
v_isShared_1426_ = v_isSharedCheck_1433_;
goto v_resetjp_1424_;
}
else
{
lean_dec(v___x_1423_);
v___x_1425_ = lean_box(0);
v_isShared_1426_ = v_isSharedCheck_1433_;
goto v_resetjp_1424_;
}
v_resetjp_1424_:
{
lean_object* v___x_1428_; 
if (v_isShared_1422_ == 0)
{
lean_ctor_set(v___x_1421_, 0, v_fst_1417_);
v___x_1428_ = v___x_1421_;
goto v_reusejp_1427_;
}
else
{
lean_object* v_reuseFailAlloc_1432_; 
v_reuseFailAlloc_1432_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1432_, 0, v_fst_1417_);
lean_ctor_set(v_reuseFailAlloc_1432_, 1, v_snd_1419_);
v___x_1428_ = v_reuseFailAlloc_1432_;
goto v_reusejp_1427_;
}
v_reusejp_1427_:
{
lean_object* v___x_1430_; 
if (v_isShared_1426_ == 0)
{
lean_ctor_set_tag(v___x_1425_, 1);
lean_ctor_set(v___x_1425_, 0, v___x_1428_);
v___x_1430_ = v___x_1425_;
goto v_reusejp_1429_;
}
else
{
lean_object* v_reuseFailAlloc_1431_; 
v_reuseFailAlloc_1431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1431_, 0, v___x_1428_);
v___x_1430_ = v_reuseFailAlloc_1431_;
goto v_reusejp_1429_;
}
v_reusejp_1429_:
{
v___y_1344_ = v___y_1399_;
v___y_1345_ = v___y_1400_;
v_a_1346_ = v___x_1430_;
goto v___jp_1343_;
}
}
}
}
else
{
lean_object* v_a_1435_; 
lean_del_object(v___x_1421_);
lean_dec(v_snd_1419_);
lean_dec(v_fst_1417_);
v_a_1435_ = lean_ctor_get(v___x_1423_, 0);
lean_inc(v_a_1435_);
lean_dec_ref_known(v___x_1423_, 1);
v___y_1349_ = v___y_1399_;
v___y_1350_ = v___y_1400_;
v_a_1351_ = v_a_1435_;
goto v___jp_1348_;
}
}
}
else
{
lean_object* v_a_1437_; 
v_a_1437_ = lean_ctor_get(v___x_1414_, 0);
lean_inc(v_a_1437_);
lean_dec_ref_known(v___x_1414_, 1);
v___y_1349_ = v___y_1399_;
v___y_1350_ = v___y_1400_;
v_a_1351_ = v_a_1437_;
goto v___jp_1348_;
}
}
}
else
{
lean_object* v_a_1438_; 
v_a_1438_ = lean_ctor_get(v___x_1403_, 0);
lean_inc(v_a_1438_);
lean_dec_ref_known(v___x_1403_, 1);
v___y_1349_ = v___y_1399_;
v___y_1350_ = v___y_1400_;
v_a_1351_ = v_a_1438_;
goto v___jp_1348_;
}
}
v___jp_1439_:
{
if (v_a_1442_ == 0)
{
lean_dec_ref_known(v___x_1074_, 1);
v___y_1399_ = v___y_1441_;
v___y_1400_ = v___y_1440_;
goto v___jp_1398_;
}
else
{
v___y_1354_ = v___y_1440_;
v___y_1355_ = v___y_1441_;
goto v___jp_1353_;
}
}
v___jp_1443_:
{
lean_object* v_a_1447_; uint8_t v___x_1448_; 
v_a_1447_ = lean_ctor_get(v___y_1446_, 0);
lean_inc(v_a_1447_);
lean_dec_ref(v___y_1446_);
v___x_1448_ = lean_unbox(v_a_1447_);
lean_dec(v_a_1447_);
v___y_1440_ = v___y_1445_;
v___y_1441_ = v___y_1444_;
v_a_1442_ = v___x_1448_;
goto v___jp_1439_;
}
v___jp_1449_:
{
lean_object* v___x_1450_; lean_object* v_a_1451_; lean_object* v___x_1452_; uint8_t v___x_1453_; 
v___x_1450_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v_a_1057_);
v_a_1451_ = lean_ctor_get(v___x_1450_, 0);
lean_inc(v_a_1451_);
lean_dec_ref(v___x_1450_);
v___x_1452_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1453_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_1063_, v___x_1452_);
if (v___x_1453_ == 0)
{
lean_object* v___x_1454_; lean_object* v___x_1455_; uint8_t v___x_1456_; 
v___x_1454_ = lean_io_mono_nanos_now();
v___x_1455_ = lp_aesop_Aesop_aesop_collectStats;
v___x_1456_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_1063_, v___x_1455_);
if (v___x_1456_ == 0)
{
lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v_a_1459_; uint8_t v___x_1460_; 
v___x_1457_ = lp_aesop_Aesop_TraceOption_stats;
v___x_1458_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_1457_, v_a_1056_);
v_a_1459_ = lean_ctor_get(v___x_1458_, 0);
lean_inc(v_a_1459_);
v___x_1460_ = lean_unbox(v_a_1459_);
lean_dec(v_a_1459_);
if (v___x_1460_ == 0)
{
lean_object* v___x_1461_; lean_object* v___x_1462_; uint8_t v___x_1463_; 
lean_dec_ref(v___x_1458_);
v___x_1461_ = lp_aesop_Aesop_aesop_stats_file;
v___x_1462_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_1063_, v___x_1461_);
v___x_1463_ = lean_string_dec_eq(v___x_1462_, v___x_1211_);
lean_dec_ref(v___x_1462_);
if (v___x_1463_ == 0)
{
v___y_1240_ = v_a_1451_;
v___y_1241_ = v___x_1454_;
goto v___jp_1239_;
}
else
{
lean_dec_ref_known(v___x_1074_, 1);
v___y_1285_ = v_a_1451_;
v___y_1286_ = v___x_1454_;
goto v___jp_1284_;
}
}
else
{
v___y_1326_ = v_a_1451_;
v___y_1327_ = v___x_1454_;
v___y_1328_ = v___x_1458_;
goto v___jp_1325_;
}
}
else
{
v___y_1240_ = v_a_1451_;
v___y_1241_ = v___x_1454_;
goto v___jp_1239_;
}
}
else
{
lean_object* v___x_1464_; lean_object* v___x_1465_; uint8_t v___x_1466_; 
v___x_1464_ = lean_io_get_num_heartbeats();
v___x_1465_ = lp_aesop_Aesop_aesop_collectStats;
v___x_1466_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_1063_, v___x_1465_);
if (v___x_1466_ == 0)
{
lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v_a_1469_; uint8_t v___x_1470_; 
v___x_1467_ = lp_aesop_Aesop_TraceOption_stats;
v___x_1468_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_1467_, v_a_1056_);
v_a_1469_ = lean_ctor_get(v___x_1468_, 0);
lean_inc(v_a_1469_);
v___x_1470_ = lean_unbox(v_a_1469_);
lean_dec(v_a_1469_);
if (v___x_1470_ == 0)
{
lean_object* v___x_1471_; lean_object* v___x_1472_; uint8_t v___x_1473_; 
lean_dec_ref(v___x_1468_);
v___x_1471_ = lp_aesop_Aesop_aesop_stats_file;
v___x_1472_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_1063_, v___x_1471_);
v___x_1473_ = lean_string_dec_eq(v___x_1472_, v___x_1211_);
lean_dec_ref(v___x_1472_);
if (v___x_1473_ == 0)
{
v___y_1440_ = v_a_1451_;
v___y_1441_ = v___x_1464_;
v_a_1442_ = v___x_1453_;
goto v___jp_1439_;
}
else
{
lean_dec_ref_known(v___x_1074_, 1);
v___y_1399_ = v___x_1464_;
v___y_1400_ = v_a_1451_;
goto v___jp_1398_;
}
}
else
{
v___y_1444_ = v___x_1464_;
v___y_1445_ = v_a_1451_;
v___y_1446_ = v___x_1468_;
goto v___jp_1443_;
}
}
else
{
v___y_1354_ = v_a_1451_;
v___y_1355_ = v___x_1464_;
goto v___jp_1353_;
}
}
}
}
v___jp_1075_:
{
if (v_____do__lift_1076_ == 0)
{
lean_object* v___x_1085_; 
lean_dec_ref_known(v___x_1074_, 1);
lean_inc_ref(v___y_1077_);
if (v_isShared_1070_ == 0)
{
lean_ctor_set(v___x_1069_, 4, v___y_1077_);
lean_ctor_set(v___x_1069_, 3, v_patternSubsts_x3f_1065_);
lean_ctor_set(v___x_1069_, 2, v_locations_1064_);
lean_ctor_set(v___x_1069_, 1, v_mvars_1048_);
lean_ctor_set(v___x_1069_, 0, v_goal_1047_);
v___x_1085_ = v___x_1069_;
goto v_reusejp_1084_;
}
else
{
lean_object* v_reuseFailAlloc_1159_; 
v_reuseFailAlloc_1159_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1159_, 0, v_goal_1047_);
lean_ctor_set(v_reuseFailAlloc_1159_, 1, v_mvars_1048_);
lean_ctor_set(v_reuseFailAlloc_1159_, 2, v_locations_1064_);
lean_ctor_set(v_reuseFailAlloc_1159_, 3, v_patternSubsts_x3f_1065_);
lean_ctor_set(v_reuseFailAlloc_1159_, 4, v___y_1077_);
v___x_1085_ = v_reuseFailAlloc_1159_;
goto v_reusejp_1084_;
}
v_reusejp_1084_:
{
lean_object* v___x_1086_; lean_object* v___x_1087_; 
v___x_1086_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTacDescr_run___boxed), 8, 1);
lean_closure_set(v___x_1086_, 0, v_tac_1067_);
v___x_1087_ = lp_aesop_Aesop_runRuleTac(v___x_1086_, v_name_1066_, v_preState_1049_, v___x_1085_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_);
if (lean_obj_tag(v___x_1087_) == 0)
{
lean_object* v_a_1088_; 
v_a_1088_ = lean_ctor_get(v___x_1087_, 0);
lean_inc(v_a_1088_);
lean_dec_ref_known(v___x_1087_, 1);
if (lean_obj_tag(v_a_1088_) == 0)
{
lean_object* v_options_1089_; uint8_t v_hasTrace_1090_; 
v_options_1089_ = lean_ctor_get(v___y_1082_, 2);
v_hasTrace_1090_ = lean_ctor_get_uint8(v_options_1089_, sizeof(void*)*1);
if (v_hasTrace_1090_ == 0)
{
lean_dec_ref_known(v_a_1088_, 1);
goto v___jp_1059_;
}
else
{
lean_object* v_a_1091_; lean_object* v_inheritedTraceOptions_1092_; lean_object* v___x_1093_; uint8_t v___x_1094_; 
v_a_1091_ = lean_ctor_get(v_a_1088_, 0);
lean_inc(v_a_1091_);
lean_dec_ref_known(v_a_1088_, 1);
v_inheritedTraceOptions_1092_ = lean_ctor_get(v___y_1082_, 13);
v___x_1093_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0);
v___x_1094_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1092_, v_options_1089_, v___x_1093_);
if (v___x_1094_ == 0)
{
lean_dec(v_a_1091_);
goto v___jp_1059_;
}
else
{
lean_object* v___x_1095_; lean_object* v___x_1096_; 
v___x_1095_ = l_Lean_Exception_toMessageData(v_a_1091_);
v___x_1096_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v_cls_1073_, v___x_1095_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_);
if (lean_obj_tag(v___x_1096_) == 0)
{
lean_dec_ref_known(v___x_1096_, 1);
goto v___jp_1059_;
}
else
{
lean_object* v_a_1097_; lean_object* v___x_1099_; uint8_t v_isShared_1100_; uint8_t v_isSharedCheck_1104_; 
v_a_1097_ = lean_ctor_get(v___x_1096_, 0);
v_isSharedCheck_1104_ = !lean_is_exclusive(v___x_1096_);
if (v_isSharedCheck_1104_ == 0)
{
v___x_1099_ = v___x_1096_;
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
else
{
lean_inc(v_a_1097_);
lean_dec(v___x_1096_);
v___x_1099_ = lean_box(0);
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
v_resetjp_1098_:
{
lean_object* v___x_1102_; 
if (v_isShared_1100_ == 0)
{
v___x_1102_ = v___x_1099_;
goto v_reusejp_1101_;
}
else
{
lean_object* v_reuseFailAlloc_1103_; 
v_reuseFailAlloc_1103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1103_, 0, v_a_1097_);
v___x_1102_ = v_reuseFailAlloc_1103_;
goto v_reusejp_1101_;
}
v_reusejp_1101_:
{
return v___x_1102_;
}
}
}
}
}
}
else
{
lean_object* v_a_1105_; lean_object* v___x_1107_; uint8_t v_isShared_1108_; uint8_t v_isSharedCheck_1150_; 
v_a_1105_ = lean_ctor_get(v_a_1088_, 0);
v_isSharedCheck_1150_ = !lean_is_exclusive(v_a_1088_);
if (v_isSharedCheck_1150_ == 0)
{
v___x_1107_ = v_a_1088_;
v_isShared_1108_ = v_isSharedCheck_1150_;
goto v_resetjp_1106_;
}
else
{
lean_inc(v_a_1105_);
lean_dec(v_a_1088_);
v___x_1107_ = lean_box(0);
v_isShared_1108_ = v_isSharedCheck_1150_;
goto v_resetjp_1106_;
}
v_resetjp_1106_:
{
lean_object* v___x_1109_; 
v___x_1109_ = lp_aesop_Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1(v_a_1105_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_);
lean_dec(v_a_1105_);
if (lean_obj_tag(v___x_1109_) == 0)
{
lean_object* v_a_1110_; lean_object* v_snd_1111_; lean_object* v_fst_1112_; lean_object* v_fst_1113_; lean_object* v_snd_1114_; lean_object* v___x_1116_; uint8_t v_isShared_1117_; uint8_t v_isSharedCheck_1141_; 
v_a_1110_ = lean_ctor_get(v___x_1109_, 0);
lean_inc(v_a_1110_);
lean_dec_ref_known(v___x_1109_, 1);
v_snd_1111_ = lean_ctor_get(v_a_1110_, 1);
lean_inc(v_snd_1111_);
v_fst_1112_ = lean_ctor_get(v_a_1110_, 0);
lean_inc(v_fst_1112_);
lean_dec(v_a_1110_);
v_fst_1113_ = lean_ctor_get(v_snd_1111_, 0);
v_snd_1114_ = lean_ctor_get(v_snd_1111_, 1);
v_isSharedCheck_1141_ = !lean_is_exclusive(v_snd_1111_);
if (v_isSharedCheck_1141_ == 0)
{
v___x_1116_ = v_snd_1111_;
v_isShared_1117_ = v_isSharedCheck_1141_;
goto v_resetjp_1115_;
}
else
{
lean_inc(v_snd_1114_);
lean_inc(v_fst_1113_);
lean_dec(v_snd_1111_);
v___x_1116_ = lean_box(0);
v_isShared_1117_ = v_isSharedCheck_1141_;
goto v_resetjp_1115_;
}
v_resetjp_1115_:
{
lean_object* v___x_1118_; 
v___x_1118_ = l_Lean_Meta_SavedState_restore___redArg(v_fst_1113_, v___y_1081_, v___y_1083_);
lean_dec(v_fst_1113_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v___x_1120_; uint8_t v_isShared_1121_; uint8_t v_isSharedCheck_1131_; 
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1131_ == 0)
{
lean_object* v_unused_1132_; 
v_unused_1132_ = lean_ctor_get(v___x_1118_, 0);
lean_dec(v_unused_1132_);
v___x_1120_ = v___x_1118_;
v_isShared_1121_ = v_isSharedCheck_1131_;
goto v_resetjp_1119_;
}
else
{
lean_dec(v___x_1118_);
v___x_1120_ = lean_box(0);
v_isShared_1121_ = v_isSharedCheck_1131_;
goto v_resetjp_1119_;
}
v_resetjp_1119_:
{
lean_object* v___x_1123_; 
if (v_isShared_1117_ == 0)
{
lean_ctor_set(v___x_1116_, 0, v_fst_1112_);
v___x_1123_ = v___x_1116_;
goto v_reusejp_1122_;
}
else
{
lean_object* v_reuseFailAlloc_1130_; 
v_reuseFailAlloc_1130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1130_, 0, v_fst_1112_);
lean_ctor_set(v_reuseFailAlloc_1130_, 1, v_snd_1114_);
v___x_1123_ = v_reuseFailAlloc_1130_;
goto v_reusejp_1122_;
}
v_reusejp_1122_:
{
lean_object* v___x_1125_; 
if (v_isShared_1108_ == 0)
{
lean_ctor_set(v___x_1107_, 0, v___x_1123_);
v___x_1125_ = v___x_1107_;
goto v_reusejp_1124_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v___x_1123_);
v___x_1125_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1124_;
}
v_reusejp_1124_:
{
lean_object* v___x_1127_; 
if (v_isShared_1121_ == 0)
{
lean_ctor_set(v___x_1120_, 0, v___x_1125_);
v___x_1127_ = v___x_1120_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1128_; 
v_reuseFailAlloc_1128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1128_, 0, v___x_1125_);
v___x_1127_ = v_reuseFailAlloc_1128_;
goto v_reusejp_1126_;
}
v_reusejp_1126_:
{
return v___x_1127_;
}
}
}
}
}
else
{
lean_object* v_a_1133_; lean_object* v___x_1135_; uint8_t v_isShared_1136_; uint8_t v_isSharedCheck_1140_; 
lean_del_object(v___x_1116_);
lean_dec(v_snd_1114_);
lean_dec(v_fst_1112_);
lean_del_object(v___x_1107_);
v_a_1133_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1140_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1140_ == 0)
{
v___x_1135_ = v___x_1118_;
v_isShared_1136_ = v_isSharedCheck_1140_;
goto v_resetjp_1134_;
}
else
{
lean_inc(v_a_1133_);
lean_dec(v___x_1118_);
v___x_1135_ = lean_box(0);
v_isShared_1136_ = v_isSharedCheck_1140_;
goto v_resetjp_1134_;
}
v_resetjp_1134_:
{
lean_object* v___x_1138_; 
if (v_isShared_1136_ == 0)
{
v___x_1138_ = v___x_1135_;
goto v_reusejp_1137_;
}
else
{
lean_object* v_reuseFailAlloc_1139_; 
v_reuseFailAlloc_1139_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1139_, 0, v_a_1133_);
v___x_1138_ = v_reuseFailAlloc_1139_;
goto v_reusejp_1137_;
}
v_reusejp_1137_:
{
return v___x_1138_;
}
}
}
}
}
else
{
lean_object* v_a_1142_; lean_object* v___x_1144_; uint8_t v_isShared_1145_; uint8_t v_isSharedCheck_1149_; 
lean_del_object(v___x_1107_);
v_a_1142_ = lean_ctor_get(v___x_1109_, 0);
v_isSharedCheck_1149_ = !lean_is_exclusive(v___x_1109_);
if (v_isSharedCheck_1149_ == 0)
{
v___x_1144_ = v___x_1109_;
v_isShared_1145_ = v_isSharedCheck_1149_;
goto v_resetjp_1143_;
}
else
{
lean_inc(v_a_1142_);
lean_dec(v___x_1109_);
v___x_1144_ = lean_box(0);
v_isShared_1145_ = v_isSharedCheck_1149_;
goto v_resetjp_1143_;
}
v_resetjp_1143_:
{
lean_object* v___x_1147_; 
if (v_isShared_1145_ == 0)
{
v___x_1147_ = v___x_1144_;
goto v_reusejp_1146_;
}
else
{
lean_object* v_reuseFailAlloc_1148_; 
v_reuseFailAlloc_1148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1148_, 0, v_a_1142_);
v___x_1147_ = v_reuseFailAlloc_1148_;
goto v_reusejp_1146_;
}
v_reusejp_1146_:
{
return v___x_1147_;
}
}
}
}
}
}
else
{
lean_object* v_a_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1158_; 
v_a_1151_ = lean_ctor_get(v___x_1087_, 0);
v_isSharedCheck_1158_ = !lean_is_exclusive(v___x_1087_);
if (v_isSharedCheck_1158_ == 0)
{
v___x_1153_ = v___x_1087_;
v_isShared_1154_ = v_isSharedCheck_1158_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_a_1151_);
lean_dec(v___x_1087_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1158_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v___x_1156_; 
if (v_isShared_1154_ == 0)
{
v___x_1156_ = v___x_1153_;
goto v_reusejp_1155_;
}
else
{
lean_object* v_reuseFailAlloc_1157_; 
v_reuseFailAlloc_1157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1157_, 0, v_a_1151_);
v___x_1156_ = v_reuseFailAlloc_1157_;
goto v_reusejp_1155_;
}
v_reusejp_1155_:
{
return v___x_1156_;
}
}
}
}
}
else
{
lean_object* v___x_1160_; lean_object* v___x_1161_; 
lean_del_object(v___x_1069_);
v___x_1160_ = lean_io_mono_nanos_now();
lean_inc_ref(v___y_1077_);
v___x_1161_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3(v_goal_1047_, v_mvars_1048_, v_locations_1064_, v_patternSubsts_x3f_1065_, v_tac_1067_, v_name_1066_, v_preState_1049_, v_cls_1073_, v___y_1077_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_);
if (lean_obj_tag(v___x_1161_) == 0)
{
lean_object* v_a_1162_; lean_object* v___x_1164_; uint8_t v_isShared_1165_; uint8_t v_isSharedCheck_1202_; 
v_a_1162_ = lean_ctor_get(v___x_1161_, 0);
v_isSharedCheck_1202_ = !lean_is_exclusive(v___x_1161_);
if (v_isSharedCheck_1202_ == 0)
{
v___x_1164_ = v___x_1161_;
v_isShared_1165_ = v_isSharedCheck_1202_;
goto v_resetjp_1163_;
}
else
{
lean_inc(v_a_1162_);
lean_dec(v___x_1161_);
v___x_1164_ = lean_box(0);
v_isShared_1165_ = v_isSharedCheck_1202_;
goto v_resetjp_1163_;
}
v_resetjp_1163_:
{
lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v_stats_1168_; lean_object* v_rulePatternCache_1169_; lean_object* v___x_1171_; uint8_t v_isShared_1172_; uint8_t v_isSharedCheck_1201_; 
v___x_1166_ = lean_io_mono_nanos_now();
v___x_1167_ = lean_st_ref_take(v___y_1079_);
v_stats_1168_ = lean_ctor_get(v___x_1167_, 1);
v_rulePatternCache_1169_ = lean_ctor_get(v___x_1167_, 0);
v_isSharedCheck_1201_ = !lean_is_exclusive(v___x_1167_);
if (v_isSharedCheck_1201_ == 0)
{
v___x_1171_ = v___x_1167_;
v_isShared_1172_ = v_isSharedCheck_1201_;
goto v_resetjp_1170_;
}
else
{
lean_inc(v_stats_1168_);
lean_inc(v_rulePatternCache_1169_);
lean_dec(v___x_1167_);
v___x_1171_ = lean_box(0);
v_isShared_1172_ = v_isSharedCheck_1201_;
goto v_resetjp_1170_;
}
v_resetjp_1170_:
{
lean_object* v_total_1173_; lean_object* v_configParsing_1174_; lean_object* v_ruleSetConstruction_1175_; lean_object* v_search_1176_; lean_object* v_ruleSelection_1177_; lean_object* v_script_1178_; lean_object* v_forwardState_1179_; lean_object* v_scriptGenerated_1180_; lean_object* v_ruleStats_1181_; lean_object* v_goalStats_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1200_; 
v_total_1173_ = lean_ctor_get(v_stats_1168_, 0);
v_configParsing_1174_ = lean_ctor_get(v_stats_1168_, 1);
v_ruleSetConstruction_1175_ = lean_ctor_get(v_stats_1168_, 2);
v_search_1176_ = lean_ctor_get(v_stats_1168_, 3);
v_ruleSelection_1177_ = lean_ctor_get(v_stats_1168_, 4);
v_script_1178_ = lean_ctor_get(v_stats_1168_, 5);
v_forwardState_1179_ = lean_ctor_get(v_stats_1168_, 6);
v_scriptGenerated_1180_ = lean_ctor_get(v_stats_1168_, 7);
v_ruleStats_1181_ = lean_ctor_get(v_stats_1168_, 8);
v_goalStats_1182_ = lean_ctor_get(v_stats_1168_, 9);
v_isSharedCheck_1200_ = !lean_is_exclusive(v_stats_1168_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_1184_ = v_stats_1168_;
v_isShared_1185_ = v_isSharedCheck_1200_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_goalStats_1182_);
lean_inc(v_ruleStats_1181_);
lean_inc(v_scriptGenerated_1180_);
lean_inc(v_forwardState_1179_);
lean_inc(v_script_1178_);
lean_inc(v_ruleSelection_1177_);
lean_inc(v_search_1176_);
lean_inc(v_ruleSetConstruction_1175_);
lean_inc(v_configParsing_1174_);
lean_inc(v_total_1173_);
lean_dec(v_stats_1168_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1200_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
lean_object* v___x_1186_; uint8_t v___x_1187_; lean_object* v_rp_1188_; lean_object* v___x_1189_; lean_object* v___x_1191_; 
v___x_1186_ = lean_nat_sub(v___x_1166_, v___x_1160_);
lean_dec(v___x_1160_);
lean_dec(v___x_1166_);
v___x_1187_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__2(v_a_1162_);
v_rp_1188_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_rp_1188_, 0, v___x_1074_);
lean_ctor_set(v_rp_1188_, 1, v___x_1186_);
lean_ctor_set_uint8(v_rp_1188_, sizeof(void*)*2, v___x_1187_);
v___x_1189_ = lean_array_push(v_ruleStats_1181_, v_rp_1188_);
if (v_isShared_1185_ == 0)
{
lean_ctor_set(v___x_1184_, 8, v___x_1189_);
v___x_1191_ = v___x_1184_;
goto v_reusejp_1190_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v_total_1173_);
lean_ctor_set(v_reuseFailAlloc_1199_, 1, v_configParsing_1174_);
lean_ctor_set(v_reuseFailAlloc_1199_, 2, v_ruleSetConstruction_1175_);
lean_ctor_set(v_reuseFailAlloc_1199_, 3, v_search_1176_);
lean_ctor_set(v_reuseFailAlloc_1199_, 4, v_ruleSelection_1177_);
lean_ctor_set(v_reuseFailAlloc_1199_, 5, v_script_1178_);
lean_ctor_set(v_reuseFailAlloc_1199_, 6, v_forwardState_1179_);
lean_ctor_set(v_reuseFailAlloc_1199_, 7, v_scriptGenerated_1180_);
lean_ctor_set(v_reuseFailAlloc_1199_, 8, v___x_1189_);
lean_ctor_set(v_reuseFailAlloc_1199_, 9, v_goalStats_1182_);
v___x_1191_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1190_;
}
v_reusejp_1190_:
{
lean_object* v___x_1193_; 
if (v_isShared_1172_ == 0)
{
lean_ctor_set(v___x_1171_, 1, v___x_1191_);
v___x_1193_ = v___x_1171_;
goto v_reusejp_1192_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v_rulePatternCache_1169_);
lean_ctor_set(v_reuseFailAlloc_1198_, 1, v___x_1191_);
v___x_1193_ = v_reuseFailAlloc_1198_;
goto v_reusejp_1192_;
}
v_reusejp_1192_:
{
lean_object* v___x_1194_; lean_object* v___x_1196_; 
v___x_1194_ = lean_st_ref_set(v___y_1079_, v___x_1193_);
if (v_isShared_1165_ == 0)
{
v___x_1196_ = v___x_1164_;
goto v_reusejp_1195_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v_a_1162_);
v___x_1196_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1195_;
}
v_reusejp_1195_:
{
return v___x_1196_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_1160_);
lean_dec_ref_known(v___x_1074_, 1);
return v___x_1161_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___boxed(lean_object* v_goal_1486_, lean_object* v_mvars_1487_, lean_object* v_preState_1488_, lean_object* v_matchResult_1489_, lean_object* v_a_1490_, lean_object* v_a_1491_, lean_object* v_a_1492_, lean_object* v_a_1493_, lean_object* v_a_1494_, lean_object* v_a_1495_, lean_object* v_a_1496_, lean_object* v_a_1497_){
_start:
{
lean_object* v_res_1498_; 
v_res_1498_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg(v_goal_1486_, v_mvars_1487_, v_preState_1488_, v_matchResult_1489_, v_a_1490_, v_a_1491_, v_a_1492_, v_a_1493_, v_a_1494_, v_a_1495_, v_a_1496_);
lean_dec(v_a_1496_);
lean_dec_ref(v_a_1495_);
lean_dec(v_a_1494_);
lean_dec_ref(v_a_1493_);
lean_dec(v_a_1492_);
lean_dec(v_a_1491_);
lean_dec_ref(v_a_1490_);
lean_dec_ref(v_preState_1488_);
return v_res_1498_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule(lean_object* v_00_u03b1_1499_, lean_object* v_goal_1500_, lean_object* v_mvars_1501_, lean_object* v_preState_1502_, lean_object* v_matchResult_1503_, lean_object* v_a_1504_, lean_object* v_a_1505_, lean_object* v_a_1506_, lean_object* v_a_1507_, lean_object* v_a_1508_, lean_object* v_a_1509_, lean_object* v_a_1510_){
_start:
{
lean_object* v___x_1512_; 
v___x_1512_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg(v_goal_1500_, v_mvars_1501_, v_preState_1502_, v_matchResult_1503_, v_a_1504_, v_a_1505_, v_a_1506_, v_a_1507_, v_a_1508_, v_a_1509_, v_a_1510_);
return v___x_1512_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___boxed(lean_object* v_00_u03b1_1513_, lean_object* v_goal_1514_, lean_object* v_mvars_1515_, lean_object* v_preState_1516_, lean_object* v_matchResult_1517_, lean_object* v_a_1518_, lean_object* v_a_1519_, lean_object* v_a_1520_, lean_object* v_a_1521_, lean_object* v_a_1522_, lean_object* v_a_1523_, lean_object* v_a_1524_, lean_object* v_a_1525_){
_start:
{
lean_object* v_res_1526_; 
v_res_1526_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule(v_00_u03b1_1513_, v_goal_1514_, v_mvars_1515_, v_preState_1516_, v_matchResult_1517_, v_a_1518_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_, v_a_1524_);
lean_dec(v_a_1524_);
lean_dec_ref(v_a_1523_);
lean_dec(v_a_1522_);
lean_dec_ref(v_a_1521_);
lean_dec(v_a_1520_);
lean_dec(v_a_1519_);
lean_dec_ref(v_a_1518_);
lean_dec_ref(v_preState_1516_);
return v_res_1526_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0(lean_object* v_cls_1527_, lean_object* v_msg_1528_, lean_object* v___y_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_, lean_object* v___y_1535_){
_start:
{
lean_object* v___x_1537_; 
v___x_1537_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v_cls_1527_, v_msg_1528_, v___y_1532_, v___y_1533_, v___y_1534_, v___y_1535_);
return v___x_1537_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___boxed(lean_object* v_cls_1538_, lean_object* v_msg_1539_, lean_object* v___y_1540_, lean_object* v___y_1541_, lean_object* v___y_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_){
_start:
{
lean_object* v_res_1548_; 
v_res_1548_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0(v_cls_1538_, v_msg_1539_, v___y_1540_, v___y_1541_, v___y_1542_, v___y_1543_, v___y_1544_, v___y_1545_, v___y_1546_);
lean_dec(v___y_1546_);
lean_dec_ref(v___y_1545_);
lean_dec(v___y_1544_);
lean_dec_ref(v___y_1543_);
lean_dec(v___y_1542_);
lean_dec(v___y_1541_);
lean_dec_ref(v___y_1540_);
return v_res_1548_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2(lean_object* v_opt_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_){
_start:
{
lean_object* v___x_1558_; 
v___x_1558_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v_opt_1549_, v___y_1555_);
return v___x_1558_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___boxed(lean_object* v_opt_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_){
_start:
{
lean_object* v_res_1568_; 
v_res_1568_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2(v_opt_1559_, v___y_1560_, v___y_1561_, v___y_1562_, v___y_1563_, v___y_1564_, v___y_1565_, v___y_1566_);
lean_dec(v___y_1566_);
lean_dec_ref(v___y_1565_);
lean_dec(v___y_1564_);
lean_dec_ref(v___y_1563_);
lean_dec(v___y_1562_);
lean_dec(v___y_1561_);
lean_dec_ref(v___y_1560_);
lean_dec_ref(v_opt_1559_);
return v_res_1568_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9(lean_object* v_00_u03b1_1569_, lean_object* v_x_1570_, lean_object* v___y_1571_, lean_object* v___y_1572_, lean_object* v___y_1573_, lean_object* v___y_1574_, lean_object* v___y_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_){
_start:
{
lean_object* v___x_1579_; 
v___x_1579_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_x_1570_);
return v___x_1579_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___boxed(lean_object* v_00_u03b1_1580_, lean_object* v_x_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_, lean_object* v___y_1589_){
_start:
{
lean_object* v_res_1590_; 
v_res_1590_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9(v_00_u03b1_1580_, v_x_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec(v___y_1583_);
lean_dec_ref(v___y_1582_);
return v_res_1590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2(lean_object* v_00_u03b1_1591_, lean_object* v_msg_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_){
_start:
{
lean_object* v___x_1601_; 
v___x_1601_ = lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg(v_msg_1592_, v___y_1596_, v___y_1597_, v___y_1598_, v___y_1599_);
return v___x_1601_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___boxed(lean_object* v_00_u03b1_1602_, lean_object* v_msg_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_){
_start:
{
lean_object* v_res_1612_; 
v_res_1612_ = lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2(v_00_u03b1_1602_, v_msg_1603_, v___y_1604_, v___y_1605_, v___y_1606_, v___y_1607_, v___y_1608_, v___y_1609_, v___y_1610_);
lean_dec(v___y_1610_);
lean_dec_ref(v___y_1609_);
lean_dec(v___y_1608_);
lean_dec_ref(v___y_1607_);
lean_dec(v___y_1606_);
lean_dec(v___y_1605_);
lean_dec_ref(v___y_1604_);
return v_res_1612_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8(lean_object* v_oldTraces_1613_, lean_object* v_data_1614_, lean_object* v_ref_1615_, lean_object* v_msg_1616_, lean_object* v___y_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v___x_1625_; 
v___x_1625_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg(v_oldTraces_1613_, v_data_1614_, v_ref_1615_, v_msg_1616_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_);
return v___x_1625_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___boxed(lean_object* v_oldTraces_1626_, lean_object* v_data_1627_, lean_object* v_ref_1628_, lean_object* v_msg_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_){
_start:
{
lean_object* v_res_1638_; 
v_res_1638_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8(v_oldTraces_1626_, v_data_1627_, v_ref_1628_, v_msg_1629_, v___y_1630_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_, v___y_1636_);
lean_dec(v___y_1636_);
lean_dec_ref(v___y_1635_);
lean_dec(v___y_1634_);
lean_dec_ref(v___y_1633_);
lean_dec(v___y_1632_);
lean_dec(v___y_1631_);
lean_dec_ref(v___y_1630_);
return v_res_1638_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___redArg(lean_object* v_steps_1639_, lean_object* v___y_1640_){
_start:
{
lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; 
v___x_1642_ = lean_st_ref_take(v___y_1640_);
v___x_1643_ = l_Array_append___redArg(v___x_1642_, v_steps_1639_);
v___x_1644_ = lean_st_ref_set(v___y_1640_, v___x_1643_);
v___x_1645_ = lean_box(0);
v___x_1646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1646_, 0, v___x_1645_);
return v___x_1646_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___redArg___boxed(lean_object* v_steps_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_){
_start:
{
lean_object* v_res_1650_; 
v_res_1650_ = lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___redArg(v_steps_1647_, v___y_1648_);
lean_dec(v___y_1648_);
lean_dec_ref(v_steps_1647_);
return v_res_1650_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0(lean_object* v_steps_1651_, lean_object* v___y_1652_, lean_object* v___y_1653_, lean_object* v___y_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_){
_start:
{
lean_object* v___x_1660_; 
v___x_1660_ = lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___redArg(v_steps_1651_, v___y_1653_);
return v___x_1660_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___boxed(lean_object* v_steps_1661_, lean_object* v___y_1662_, lean_object* v___y_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_){
_start:
{
lean_object* v_res_1670_; 
v_res_1670_ = lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0(v_steps_1661_, v___y_1662_, v___y_1663_, v___y_1664_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
lean_dec(v___y_1668_);
lean_dec_ref(v___y_1667_);
lean_dec(v___y_1666_);
lean_dec_ref(v___y_1665_);
lean_dec(v___y_1664_);
lean_dec(v___y_1663_);
lean_dec_ref(v___y_1662_);
lean_dec_ref(v_steps_1661_);
return v_res_1670_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_1672_; lean_object* v___x_1673_; 
v___x_1672_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__0));
v___x_1673_ = l_Lean_stringToMessageData(v___x_1672_);
return v___x_1673_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_1675_; lean_object* v___x_1676_; 
v___x_1675_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__2));
v___x_1676_ = l_Lean_stringToMessageData(v___x_1675_);
return v___x_1676_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg(lean_object* v_goal_1680_, lean_object* v_mvars_1681_, lean_object* v_preState_1682_, lean_object* v_as_1683_, size_t v_sz_1684_, size_t v_i_1685_, lean_object* v_b_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_){
_start:
{
uint8_t v___x_1695_; 
v___x_1695_ = lean_usize_dec_lt(v_i_1685_, v_sz_1684_);
if (v___x_1695_ == 0)
{
lean_object* v___x_1696_; 
lean_dec_ref(v_mvars_1681_);
lean_dec(v_goal_1680_);
v___x_1696_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1696_, 0, v_b_1686_);
return v___x_1696_;
}
else
{
lean_object* v_a_1697_; lean_object* v___x_1698_; 
lean_dec_ref(v_b_1686_);
v_a_1697_ = lean_array_uget_borrowed(v_as_1683_, v_i_1685_);
lean_inc(v_a_1697_);
lean_inc_ref(v_mvars_1681_);
lean_inc(v_goal_1680_);
v___x_1698_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg(v_goal_1680_, v_mvars_1681_, v_preState_1682_, v_a_1697_, v___y_1687_, v___y_1688_, v___y_1689_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_);
if (lean_obj_tag(v___x_1698_) == 0)
{
lean_object* v_a_1699_; lean_object* v___x_1701_; uint8_t v_isShared_1702_; uint8_t v_isSharedCheck_1793_; 
v_a_1699_ = lean_ctor_get(v___x_1698_, 0);
v_isSharedCheck_1793_ = !lean_is_exclusive(v___x_1698_);
if (v_isSharedCheck_1793_ == 0)
{
v___x_1701_ = v___x_1698_;
v_isShared_1702_ = v_isSharedCheck_1793_;
goto v_resetjp_1700_;
}
else
{
lean_inc(v_a_1699_);
lean_dec(v___x_1698_);
v___x_1701_ = lean_box(0);
v_isShared_1702_ = v_isSharedCheck_1793_;
goto v_resetjp_1700_;
}
v_resetjp_1700_:
{
lean_object* v___x_1703_; 
v___x_1703_ = lean_box(0);
if (lean_obj_tag(v_a_1699_) == 1)
{
lean_object* v_val_1704_; lean_object* v___x_1706_; uint8_t v_isShared_1707_; uint8_t v_isSharedCheck_1788_; 
lean_dec_ref(v_mvars_1681_);
lean_dec(v_goal_1680_);
v_val_1704_ = lean_ctor_get(v_a_1699_, 0);
v_isSharedCheck_1788_ = !lean_is_exclusive(v_a_1699_);
if (v_isSharedCheck_1788_ == 0)
{
v___x_1706_ = v_a_1699_;
v_isShared_1707_ = v_isSharedCheck_1788_;
goto v_resetjp_1705_;
}
else
{
lean_inc(v_val_1704_);
lean_dec(v_a_1699_);
v___x_1706_ = lean_box(0);
v_isShared_1707_ = v_isSharedCheck_1788_;
goto v_resetjp_1705_;
}
v_resetjp_1705_:
{
lean_object* v_fst_1708_; lean_object* v_snd_1709_; lean_object* v___x_1711_; uint8_t v_isShared_1712_; uint8_t v_isSharedCheck_1787_; 
v_fst_1708_ = lean_ctor_get(v_val_1704_, 0);
v_snd_1709_ = lean_ctor_get(v_val_1704_, 1);
v_isSharedCheck_1787_ = !lean_is_exclusive(v_val_1704_);
if (v_isSharedCheck_1787_ == 0)
{
v___x_1711_ = v_val_1704_;
v_isShared_1712_ = v_isSharedCheck_1787_;
goto v_resetjp_1710_;
}
else
{
lean_inc(v_snd_1709_);
lean_inc(v_fst_1708_);
lean_dec(v_val_1704_);
v___x_1711_ = lean_box(0);
v_isShared_1712_ = v_isSharedCheck_1787_;
goto v_resetjp_1710_;
}
v_resetjp_1710_:
{
uint8_t v_generateScript_1724_; 
v_generateScript_1724_ = lean_ctor_get_uint8(v___y_1687_, sizeof(void*)*2);
if (v_generateScript_1724_ == 0)
{
lean_dec(v_snd_1709_);
goto v___jp_1713_;
}
else
{
if (lean_obj_tag(v_snd_1709_) == 1)
{
lean_object* v_val_1725_; lean_object* v___x_1726_; 
v_val_1725_ = lean_ctor_get(v_snd_1709_, 0);
lean_inc(v_val_1725_);
lean_dec_ref_known(v_snd_1709_, 1);
v___x_1726_ = lp_aesop_Aesop_recordScriptSteps___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__0___redArg(v_val_1725_, v___y_1688_);
lean_dec(v_val_1725_);
if (lean_obj_tag(v___x_1726_) == 0)
{
lean_dec_ref_known(v___x_1726_, 1);
goto v___jp_1713_;
}
else
{
lean_object* v_a_1727_; lean_object* v___x_1729_; uint8_t v_isShared_1730_; uint8_t v_isSharedCheck_1734_; 
lean_del_object(v___x_1711_);
lean_dec(v_fst_1708_);
lean_del_object(v___x_1706_);
lean_del_object(v___x_1701_);
v_a_1727_ = lean_ctor_get(v___x_1726_, 0);
v_isSharedCheck_1734_ = !lean_is_exclusive(v___x_1726_);
if (v_isSharedCheck_1734_ == 0)
{
v___x_1729_ = v___x_1726_;
v_isShared_1730_ = v_isSharedCheck_1734_;
goto v_resetjp_1728_;
}
else
{
lean_inc(v_a_1727_);
lean_dec(v___x_1726_);
v___x_1729_ = lean_box(0);
v_isShared_1730_ = v_isSharedCheck_1734_;
goto v_resetjp_1728_;
}
v_resetjp_1728_:
{
lean_object* v___x_1732_; 
if (v_isShared_1730_ == 0)
{
v___x_1732_ = v___x_1729_;
goto v_reusejp_1731_;
}
else
{
lean_object* v_reuseFailAlloc_1733_; 
v_reuseFailAlloc_1733_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1733_, 0, v_a_1727_);
v___x_1732_ = v_reuseFailAlloc_1733_;
goto v_reusejp_1731_;
}
v_reusejp_1731_:
{
return v___x_1732_;
}
}
}
}
else
{
lean_object* v_rule_1735_; lean_object* v_name_1736_; lean_object* v_name_1737_; uint8_t v_builder_1738_; uint8_t v_phase_1739_; uint8_t v_scope_1740_; lean_object* v___x_1741_; lean_object* v___y_1743_; lean_object* v___y_1744_; lean_object* v___y_1745_; lean_object* v___y_1765_; lean_object* v___y_1766_; lean_object* v___y_1767_; lean_object* v___y_1773_; 
lean_dec(v_snd_1709_);
v_rule_1735_ = lean_ctor_get(v_a_1697_, 0);
v_name_1736_ = lean_ctor_get(v_rule_1735_, 0);
v_name_1737_ = lean_ctor_get(v_name_1736_, 0);
v_builder_1738_ = lean_ctor_get_uint8(v_name_1736_, sizeof(void*)*1 + 8);
v_phase_1739_ = lean_ctor_get_uint8(v_name_1736_, sizeof(void*)*1 + 9);
v_scope_1740_ = lean_ctor_get_uint8(v_name_1736_, sizeof(void*)*1 + 10);
v___x_1741_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__1);
switch(v_phase_1739_)
{
case 0:
{
lean_object* v___x_1784_; 
v___x_1784_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__13));
v___y_1773_ = v___x_1784_;
goto v___jp_1772_;
}
case 1:
{
lean_object* v___x_1785_; 
v___x_1785_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__14));
v___y_1773_ = v___x_1785_;
goto v___jp_1772_;
}
default: 
{
lean_object* v___x_1786_; 
v___x_1786_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__15));
v___y_1773_ = v___x_1786_;
goto v___jp_1772_;
}
}
v___jp_1742_:
{
lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; 
v___x_1746_ = lean_string_append(v___y_1743_, v___y_1745_);
v___x_1747_ = lean_string_append(v___x_1746_, v___y_1744_);
lean_inc(v_name_1737_);
v___x_1748_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1737_, v_generateScript_1724_);
v___x_1749_ = lean_string_append(v___x_1747_, v___x_1748_);
lean_dec_ref(v___x_1748_);
v___x_1750_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1750_, 0, v___x_1749_);
v___x_1751_ = l_Lean_MessageData_ofFormat(v___x_1750_);
v___x_1752_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1752_, 0, v___x_1741_);
lean_ctor_set(v___x_1752_, 1, v___x_1751_);
v___x_1753_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__3);
v___x_1754_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1754_, 0, v___x_1752_);
lean_ctor_set(v___x_1754_, 1, v___x_1753_);
v___x_1755_ = lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg(v___x_1754_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_);
if (lean_obj_tag(v___x_1755_) == 0)
{
lean_dec_ref_known(v___x_1755_, 1);
goto v___jp_1713_;
}
else
{
lean_object* v_a_1756_; lean_object* v___x_1758_; uint8_t v_isShared_1759_; uint8_t v_isSharedCheck_1763_; 
lean_del_object(v___x_1711_);
lean_dec(v_fst_1708_);
lean_del_object(v___x_1706_);
lean_del_object(v___x_1701_);
v_a_1756_ = lean_ctor_get(v___x_1755_, 0);
v_isSharedCheck_1763_ = !lean_is_exclusive(v___x_1755_);
if (v_isSharedCheck_1763_ == 0)
{
v___x_1758_ = v___x_1755_;
v_isShared_1759_ = v_isSharedCheck_1763_;
goto v_resetjp_1757_;
}
else
{
lean_inc(v_a_1756_);
lean_dec(v___x_1755_);
v___x_1758_ = lean_box(0);
v_isShared_1759_ = v_isSharedCheck_1763_;
goto v_resetjp_1757_;
}
v_resetjp_1757_:
{
lean_object* v___x_1761_; 
if (v_isShared_1759_ == 0)
{
v___x_1761_ = v___x_1758_;
goto v_reusejp_1760_;
}
else
{
lean_object* v_reuseFailAlloc_1762_; 
v_reuseFailAlloc_1762_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1762_, 0, v_a_1756_);
v___x_1761_ = v_reuseFailAlloc_1762_;
goto v_reusejp_1760_;
}
v_reusejp_1760_:
{
return v___x_1761_;
}
}
}
}
v___jp_1764_:
{
lean_object* v___x_1768_; lean_object* v___x_1769_; 
v___x_1768_ = lean_string_append(v___y_1766_, v___y_1767_);
v___x_1769_ = lean_string_append(v___x_1768_, v___y_1765_);
if (v_scope_1740_ == 0)
{
lean_object* v___x_1770_; 
v___x_1770_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__2));
v___y_1743_ = v___x_1769_;
v___y_1744_ = v___y_1765_;
v___y_1745_ = v___x_1770_;
goto v___jp_1742_;
}
else
{
lean_object* v___x_1771_; 
v___x_1771_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__3));
v___y_1743_ = v___x_1769_;
v___y_1744_ = v___y_1765_;
v___y_1745_ = v___x_1771_;
goto v___jp_1742_;
}
}
v___jp_1772_:
{
lean_object* v___x_1774_; lean_object* v___x_1775_; 
v___x_1774_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__4));
lean_inc_ref(v___y_1773_);
v___x_1775_ = lean_string_append(v___y_1773_, v___x_1774_);
switch(v_builder_1738_)
{
case 0:
{
lean_object* v___x_1776_; 
v___x_1776_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__5));
v___y_1765_ = v___x_1774_;
v___y_1766_ = v___x_1775_;
v___y_1767_ = v___x_1776_;
goto v___jp_1764_;
}
case 1:
{
lean_object* v___x_1777_; 
v___x_1777_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__6));
v___y_1765_ = v___x_1774_;
v___y_1766_ = v___x_1775_;
v___y_1767_ = v___x_1777_;
goto v___jp_1764_;
}
case 2:
{
lean_object* v___x_1778_; 
v___x_1778_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__7));
v___y_1765_ = v___x_1774_;
v___y_1766_ = v___x_1775_;
v___y_1767_ = v___x_1778_;
goto v___jp_1764_;
}
case 3:
{
lean_object* v___x_1779_; 
v___x_1779_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__8));
v___y_1765_ = v___x_1774_;
v___y_1766_ = v___x_1775_;
v___y_1767_ = v___x_1779_;
goto v___jp_1764_;
}
case 4:
{
lean_object* v___x_1780_; 
v___x_1780_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__9));
v___y_1765_ = v___x_1774_;
v___y_1766_ = v___x_1775_;
v___y_1767_ = v___x_1780_;
goto v___jp_1764_;
}
case 5:
{
lean_object* v___x_1781_; 
v___x_1781_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__10));
v___y_1765_ = v___x_1774_;
v___y_1766_ = v___x_1775_;
v___y_1767_ = v___x_1781_;
goto v___jp_1764_;
}
case 6:
{
lean_object* v___x_1782_; 
v___x_1782_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__11));
v___y_1765_ = v___x_1774_;
v___y_1766_ = v___x_1775_;
v___y_1767_ = v___x_1782_;
goto v___jp_1764_;
}
default: 
{
lean_object* v___x_1783_; 
v___x_1783_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__4___closed__12));
v___y_1765_ = v___x_1774_;
v___y_1766_ = v___x_1775_;
v___y_1767_ = v___x_1783_;
goto v___jp_1764_;
}
}
}
}
}
v___jp_1713_:
{
lean_object* v___x_1715_; 
if (v_isShared_1707_ == 0)
{
lean_ctor_set(v___x_1706_, 0, v_fst_1708_);
v___x_1715_ = v___x_1706_;
goto v_reusejp_1714_;
}
else
{
lean_object* v_reuseFailAlloc_1723_; 
v_reuseFailAlloc_1723_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1723_, 0, v_fst_1708_);
v___x_1715_ = v_reuseFailAlloc_1723_;
goto v_reusejp_1714_;
}
v_reusejp_1714_:
{
lean_object* v___x_1716_; lean_object* v___x_1718_; 
v___x_1716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1716_, 0, v___x_1715_);
if (v_isShared_1712_ == 0)
{
lean_ctor_set(v___x_1711_, 1, v___x_1703_);
lean_ctor_set(v___x_1711_, 0, v___x_1716_);
v___x_1718_ = v___x_1711_;
goto v_reusejp_1717_;
}
else
{
lean_object* v_reuseFailAlloc_1722_; 
v_reuseFailAlloc_1722_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1722_, 0, v___x_1716_);
lean_ctor_set(v_reuseFailAlloc_1722_, 1, v___x_1703_);
v___x_1718_ = v_reuseFailAlloc_1722_;
goto v_reusejp_1717_;
}
v_reusejp_1717_:
{
lean_object* v___x_1720_; 
if (v_isShared_1702_ == 0)
{
lean_ctor_set(v___x_1701_, 0, v___x_1718_);
v___x_1720_ = v___x_1701_;
goto v_reusejp_1719_;
}
else
{
lean_object* v_reuseFailAlloc_1721_; 
v_reuseFailAlloc_1721_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1721_, 0, v___x_1718_);
v___x_1720_ = v_reuseFailAlloc_1721_;
goto v_reusejp_1719_;
}
v_reusejp_1719_:
{
return v___x_1720_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1789_; size_t v___x_1790_; size_t v___x_1791_; 
lean_del_object(v___x_1701_);
lean_dec(v_a_1699_);
v___x_1789_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__4));
v___x_1790_ = ((size_t)1ULL);
v___x_1791_ = lean_usize_add(v_i_1685_, v___x_1790_);
v_i_1685_ = v___x_1791_;
v_b_1686_ = v___x_1789_;
goto _start;
}
}
}
else
{
lean_object* v_a_1794_; lean_object* v___x_1796_; uint8_t v_isShared_1797_; uint8_t v_isSharedCheck_1801_; 
lean_dec_ref(v_mvars_1681_);
lean_dec(v_goal_1680_);
v_a_1794_ = lean_ctor_get(v___x_1698_, 0);
v_isSharedCheck_1801_ = !lean_is_exclusive(v___x_1698_);
if (v_isSharedCheck_1801_ == 0)
{
v___x_1796_ = v___x_1698_;
v_isShared_1797_ = v_isSharedCheck_1801_;
goto v_resetjp_1795_;
}
else
{
lean_inc(v_a_1794_);
lean_dec(v___x_1698_);
v___x_1796_ = lean_box(0);
v_isShared_1797_ = v_isSharedCheck_1801_;
goto v_resetjp_1795_;
}
v_resetjp_1795_:
{
lean_object* v___x_1799_; 
if (v_isShared_1797_ == 0)
{
v___x_1799_ = v___x_1796_;
goto v_reusejp_1798_;
}
else
{
lean_object* v_reuseFailAlloc_1800_; 
v_reuseFailAlloc_1800_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1800_, 0, v_a_1794_);
v___x_1799_ = v_reuseFailAlloc_1800_;
goto v_reusejp_1798_;
}
v_reusejp_1798_:
{
return v___x_1799_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___boxed(lean_object* v_goal_1802_, lean_object* v_mvars_1803_, lean_object* v_preState_1804_, lean_object* v_as_1805_, lean_object* v_sz_1806_, lean_object* v_i_1807_, lean_object* v_b_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_){
_start:
{
size_t v_sz_boxed_1817_; size_t v_i_boxed_1818_; lean_object* v_res_1819_; 
v_sz_boxed_1817_ = lean_unbox_usize(v_sz_1806_);
lean_dec(v_sz_1806_);
v_i_boxed_1818_ = lean_unbox_usize(v_i_1807_);
lean_dec(v_i_1807_);
v_res_1819_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg(v_goal_1802_, v_mvars_1803_, v_preState_1804_, v_as_1805_, v_sz_boxed_1817_, v_i_boxed_1818_, v_b_1808_, v___y_1809_, v___y_1810_, v___y_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
lean_dec(v___y_1815_);
lean_dec_ref(v___y_1814_);
lean_dec(v___y_1813_);
lean_dec_ref(v___y_1812_);
lean_dec(v___y_1811_);
lean_dec(v___y_1810_);
lean_dec_ref(v___y_1809_);
lean_dec_ref(v_as_1805_);
lean_dec_ref(v_preState_1804_);
return v_res_1819_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(lean_object* v_goal_1820_, lean_object* v_mvars_1821_, lean_object* v_preState_1822_, lean_object* v_matchResults_1823_, lean_object* v_a_1824_, lean_object* v_a_1825_, lean_object* v_a_1826_, lean_object* v_a_1827_, lean_object* v_a_1828_, lean_object* v_a_1829_, lean_object* v_a_1830_){
_start:
{
lean_object* v___x_1832_; lean_object* v___x_1833_; size_t v_sz_1834_; size_t v___x_1835_; lean_object* v___x_1836_; 
v___x_1832_ = lean_box(0);
v___x_1833_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg___closed__4));
v_sz_1834_ = lean_array_size(v_matchResults_1823_);
v___x_1835_ = ((size_t)0ULL);
v___x_1836_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg(v_goal_1820_, v_mvars_1821_, v_preState_1822_, v_matchResults_1823_, v_sz_1834_, v___x_1835_, v___x_1833_, v_a_1824_, v_a_1825_, v_a_1826_, v_a_1827_, v_a_1828_, v_a_1829_, v_a_1830_);
if (lean_obj_tag(v___x_1836_) == 0)
{
lean_object* v_a_1837_; lean_object* v___x_1839_; uint8_t v_isShared_1840_; uint8_t v_isSharedCheck_1849_; 
v_a_1837_ = lean_ctor_get(v___x_1836_, 0);
v_isSharedCheck_1849_ = !lean_is_exclusive(v___x_1836_);
if (v_isSharedCheck_1849_ == 0)
{
v___x_1839_ = v___x_1836_;
v_isShared_1840_ = v_isSharedCheck_1849_;
goto v_resetjp_1838_;
}
else
{
lean_inc(v_a_1837_);
lean_dec(v___x_1836_);
v___x_1839_ = lean_box(0);
v_isShared_1840_ = v_isSharedCheck_1849_;
goto v_resetjp_1838_;
}
v_resetjp_1838_:
{
lean_object* v_fst_1841_; 
v_fst_1841_ = lean_ctor_get(v_a_1837_, 0);
lean_inc(v_fst_1841_);
lean_dec(v_a_1837_);
if (lean_obj_tag(v_fst_1841_) == 0)
{
lean_object* v___x_1843_; 
if (v_isShared_1840_ == 0)
{
lean_ctor_set(v___x_1839_, 0, v___x_1832_);
v___x_1843_ = v___x_1839_;
goto v_reusejp_1842_;
}
else
{
lean_object* v_reuseFailAlloc_1844_; 
v_reuseFailAlloc_1844_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1844_, 0, v___x_1832_);
v___x_1843_ = v_reuseFailAlloc_1844_;
goto v_reusejp_1842_;
}
v_reusejp_1842_:
{
return v___x_1843_;
}
}
else
{
lean_object* v_val_1845_; lean_object* v___x_1847_; 
v_val_1845_ = lean_ctor_get(v_fst_1841_, 0);
lean_inc(v_val_1845_);
lean_dec_ref_known(v_fst_1841_, 1);
if (v_isShared_1840_ == 0)
{
lean_ctor_set(v___x_1839_, 0, v_val_1845_);
v___x_1847_ = v___x_1839_;
goto v_reusejp_1846_;
}
else
{
lean_object* v_reuseFailAlloc_1848_; 
v_reuseFailAlloc_1848_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1848_, 0, v_val_1845_);
v___x_1847_ = v_reuseFailAlloc_1848_;
goto v_reusejp_1846_;
}
v_reusejp_1846_:
{
return v___x_1847_;
}
}
}
}
else
{
lean_object* v_a_1850_; lean_object* v___x_1852_; uint8_t v_isShared_1853_; uint8_t v_isSharedCheck_1857_; 
v_a_1850_ = lean_ctor_get(v___x_1836_, 0);
v_isSharedCheck_1857_ = !lean_is_exclusive(v___x_1836_);
if (v_isSharedCheck_1857_ == 0)
{
v___x_1852_ = v___x_1836_;
v_isShared_1853_ = v_isSharedCheck_1857_;
goto v_resetjp_1851_;
}
else
{
lean_inc(v_a_1850_);
lean_dec(v___x_1836_);
v___x_1852_ = lean_box(0);
v_isShared_1853_ = v_isSharedCheck_1857_;
goto v_resetjp_1851_;
}
v_resetjp_1851_:
{
lean_object* v___x_1855_; 
if (v_isShared_1853_ == 0)
{
v___x_1855_ = v___x_1852_;
goto v_reusejp_1854_;
}
else
{
lean_object* v_reuseFailAlloc_1856_; 
v_reuseFailAlloc_1856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1856_, 0, v_a_1850_);
v___x_1855_ = v_reuseFailAlloc_1856_;
goto v_reusejp_1854_;
}
v_reusejp_1854_:
{
return v___x_1855_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg___boxed(lean_object* v_goal_1858_, lean_object* v_mvars_1859_, lean_object* v_preState_1860_, lean_object* v_matchResults_1861_, lean_object* v_a_1862_, lean_object* v_a_1863_, lean_object* v_a_1864_, lean_object* v_a_1865_, lean_object* v_a_1866_, lean_object* v_a_1867_, lean_object* v_a_1868_, lean_object* v_a_1869_){
_start:
{
lean_object* v_res_1870_; 
v_res_1870_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_1858_, v_mvars_1859_, v_preState_1860_, v_matchResults_1861_, v_a_1862_, v_a_1863_, v_a_1864_, v_a_1865_, v_a_1866_, v_a_1867_, v_a_1868_);
lean_dec(v_a_1868_);
lean_dec_ref(v_a_1867_);
lean_dec(v_a_1866_);
lean_dec_ref(v_a_1865_);
lean_dec(v_a_1864_);
lean_dec(v_a_1863_);
lean_dec_ref(v_a_1862_);
lean_dec_ref(v_matchResults_1861_);
lean_dec_ref(v_preState_1860_);
return v_res_1870_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule(lean_object* v_00_u03b1_1871_, lean_object* v_goal_1872_, lean_object* v_mvars_1873_, lean_object* v_preState_1874_, lean_object* v_matchResults_1875_, lean_object* v_a_1876_, lean_object* v_a_1877_, lean_object* v_a_1878_, lean_object* v_a_1879_, lean_object* v_a_1880_, lean_object* v_a_1881_, lean_object* v_a_1882_){
_start:
{
lean_object* v___x_1884_; 
v___x_1884_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_1872_, v_mvars_1873_, v_preState_1874_, v_matchResults_1875_, v_a_1876_, v_a_1877_, v_a_1878_, v_a_1879_, v_a_1880_, v_a_1881_, v_a_1882_);
return v___x_1884_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___boxed(lean_object* v_00_u03b1_1885_, lean_object* v_goal_1886_, lean_object* v_mvars_1887_, lean_object* v_preState_1888_, lean_object* v_matchResults_1889_, lean_object* v_a_1890_, lean_object* v_a_1891_, lean_object* v_a_1892_, lean_object* v_a_1893_, lean_object* v_a_1894_, lean_object* v_a_1895_, lean_object* v_a_1896_, lean_object* v_a_1897_){
_start:
{
lean_object* v_res_1898_; 
v_res_1898_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule(v_00_u03b1_1885_, v_goal_1886_, v_mvars_1887_, v_preState_1888_, v_matchResults_1889_, v_a_1890_, v_a_1891_, v_a_1892_, v_a_1893_, v_a_1894_, v_a_1895_, v_a_1896_);
lean_dec(v_a_1896_);
lean_dec_ref(v_a_1895_);
lean_dec(v_a_1894_);
lean_dec_ref(v_a_1893_);
lean_dec(v_a_1892_);
lean_dec(v_a_1891_);
lean_dec_ref(v_a_1890_);
lean_dec_ref(v_matchResults_1889_);
lean_dec_ref(v_preState_1888_);
return v_res_1898_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1(lean_object* v_00_u03b1_1899_, lean_object* v_goal_1900_, lean_object* v_mvars_1901_, lean_object* v_preState_1902_, lean_object* v_as_1903_, size_t v_sz_1904_, size_t v_i_1905_, lean_object* v_b_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_){
_start:
{
lean_object* v___x_1915_; 
v___x_1915_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___redArg(v_goal_1900_, v_mvars_1901_, v_preState_1902_, v_as_1903_, v_sz_1904_, v_i_1905_, v_b_1906_, v___y_1907_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_);
return v___x_1915_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1___boxed(lean_object* v_00_u03b1_1916_, lean_object* v_goal_1917_, lean_object* v_mvars_1918_, lean_object* v_preState_1919_, lean_object* v_as_1920_, lean_object* v_sz_1921_, lean_object* v_i_1922_, lean_object* v_b_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_){
_start:
{
size_t v_sz_boxed_1932_; size_t v_i_boxed_1933_; lean_object* v_res_1934_; 
v_sz_boxed_1932_ = lean_unbox_usize(v_sz_1921_);
lean_dec(v_sz_1921_);
v_i_boxed_1933_ = lean_unbox_usize(v_i_1922_);
lean_dec(v_i_1922_);
v_res_1934_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule_spec__1(v_00_u03b1_1916_, v_goal_1917_, v_mvars_1918_, v_preState_1919_, v_as_1920_, v_sz_boxed_1932_, v_i_boxed_1933_, v_b_1923_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_, v___y_1928_, v___y_1929_, v___y_1930_);
lean_dec(v___y_1930_);
lean_dec_ref(v___y_1929_);
lean_dec(v___y_1928_);
lean_dec_ref(v___y_1927_);
lean_dec(v___y_1926_);
lean_dec(v___y_1925_);
lean_dec_ref(v___y_1924_);
lean_dec_ref(v_as_1920_);
lean_dec_ref(v_preState_1919_);
return v_res_1934_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__1(void){
_start:
{
lean_object* v___x_1936_; lean_object* v___x_1937_; 
v___x_1936_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__0));
v___x_1937_ = l_Lean_stringToMessageData(v___x_1936_);
return v___x_1937_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2(lean_object* v_x_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_){
_start:
{
lean_object* v___x_1947_; lean_object* v___x_1948_; 
v___x_1947_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___closed__1);
v___x_1948_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1948_, 0, v___x_1947_);
return v___x_1948_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2___boxed(lean_object* v_x_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_){
_start:
{
lean_object* v_res_1958_; 
v_res_1958_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__2(v_x_1949_, v___y_1950_, v___y_1951_, v___y_1952_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_);
lean_dec(v___y_1956_);
lean_dec_ref(v___y_1955_);
lean_dec(v___y_1954_);
lean_dec_ref(v___y_1953_);
lean_dec(v___y_1952_);
lean_dec(v___y_1951_);
lean_dec_ref(v___y_1950_);
lean_dec_ref(v_x_1949_);
return v_res_1958_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__0(lean_object* v_x_1959_){
_start:
{
lean_object* v_name_1960_; uint8_t v___x_1961_; 
v_name_1960_ = lean_ctor_get(v_x_1959_, 0);
v___x_1961_ = lp_aesop_Aesop_isForwardOrDestructRuleName(v_name_1960_);
return v___x_1961_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__0___boxed(lean_object* v_x_1962_){
_start:
{
uint8_t v_res_1963_; lean_object* v_r_1964_; 
v_res_1963_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__0(v_x_1962_);
lean_dec_ref(v_x_1962_);
v_r_1964_ = lean_box(v_res_1963_);
return v_r_1964_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__1(lean_object* v_rs_1965_, lean_object* v___x_1966_, lean_object* v_goal_1967_, lean_object* v___f_1968_, uint8_t v_____do__lift_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_){
_start:
{
if (v_____do__lift_1969_ == 0)
{
lean_object* v___x_1978_; 
v___x_1978_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_1965_, v___x_1966_, v_goal_1967_, v___f_1968_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_);
return v___x_1978_;
}
else
{
lean_object* v___x_1979_; lean_object* v___x_1980_; 
v___x_1979_ = lean_io_mono_nanos_now();
v___x_1980_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_1965_, v___x_1966_, v_goal_1967_, v___f_1968_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_);
if (lean_obj_tag(v___x_1980_) == 0)
{
lean_object* v_a_1981_; lean_object* v___x_1983_; uint8_t v_isShared_1984_; uint8_t v_isSharedCheck_2019_; 
v_a_1981_ = lean_ctor_get(v___x_1980_, 0);
v_isSharedCheck_2019_ = !lean_is_exclusive(v___x_1980_);
if (v_isSharedCheck_2019_ == 0)
{
v___x_1983_ = v___x_1980_;
v_isShared_1984_ = v_isSharedCheck_2019_;
goto v_resetjp_1982_;
}
else
{
lean_inc(v_a_1981_);
lean_dec(v___x_1980_);
v___x_1983_ = lean_box(0);
v_isShared_1984_ = v_isSharedCheck_2019_;
goto v_resetjp_1982_;
}
v_resetjp_1982_:
{
lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v_stats_1987_; lean_object* v_rulePatternCache_1988_; lean_object* v___x_1990_; uint8_t v_isShared_1991_; uint8_t v_isSharedCheck_2018_; 
v___x_1985_ = lean_io_mono_nanos_now();
v___x_1986_ = lean_st_ref_take(v___y_1972_);
v_stats_1987_ = lean_ctor_get(v___x_1986_, 1);
v_rulePatternCache_1988_ = lean_ctor_get(v___x_1986_, 0);
v_isSharedCheck_2018_ = !lean_is_exclusive(v___x_1986_);
if (v_isSharedCheck_2018_ == 0)
{
v___x_1990_ = v___x_1986_;
v_isShared_1991_ = v_isSharedCheck_2018_;
goto v_resetjp_1989_;
}
else
{
lean_inc(v_stats_1987_);
lean_inc(v_rulePatternCache_1988_);
lean_dec(v___x_1986_);
v___x_1990_ = lean_box(0);
v_isShared_1991_ = v_isSharedCheck_2018_;
goto v_resetjp_1989_;
}
v_resetjp_1989_:
{
lean_object* v_total_1992_; lean_object* v_configParsing_1993_; lean_object* v_ruleSetConstruction_1994_; lean_object* v_search_1995_; lean_object* v_ruleSelection_1996_; lean_object* v_script_1997_; lean_object* v_forwardState_1998_; lean_object* v_scriptGenerated_1999_; lean_object* v_ruleStats_2000_; lean_object* v_goalStats_2001_; lean_object* v___x_2003_; uint8_t v_isShared_2004_; uint8_t v_isSharedCheck_2017_; 
v_total_1992_ = lean_ctor_get(v_stats_1987_, 0);
v_configParsing_1993_ = lean_ctor_get(v_stats_1987_, 1);
v_ruleSetConstruction_1994_ = lean_ctor_get(v_stats_1987_, 2);
v_search_1995_ = lean_ctor_get(v_stats_1987_, 3);
v_ruleSelection_1996_ = lean_ctor_get(v_stats_1987_, 4);
v_script_1997_ = lean_ctor_get(v_stats_1987_, 5);
v_forwardState_1998_ = lean_ctor_get(v_stats_1987_, 6);
v_scriptGenerated_1999_ = lean_ctor_get(v_stats_1987_, 7);
v_ruleStats_2000_ = lean_ctor_get(v_stats_1987_, 8);
v_goalStats_2001_ = lean_ctor_get(v_stats_1987_, 9);
v_isSharedCheck_2017_ = !lean_is_exclusive(v_stats_1987_);
if (v_isSharedCheck_2017_ == 0)
{
v___x_2003_ = v_stats_1987_;
v_isShared_2004_ = v_isSharedCheck_2017_;
goto v_resetjp_2002_;
}
else
{
lean_inc(v_goalStats_2001_);
lean_inc(v_ruleStats_2000_);
lean_inc(v_scriptGenerated_1999_);
lean_inc(v_forwardState_1998_);
lean_inc(v_script_1997_);
lean_inc(v_ruleSelection_1996_);
lean_inc(v_search_1995_);
lean_inc(v_ruleSetConstruction_1994_);
lean_inc(v_configParsing_1993_);
lean_inc(v_total_1992_);
lean_dec(v_stats_1987_);
v___x_2003_ = lean_box(0);
v_isShared_2004_ = v_isSharedCheck_2017_;
goto v_resetjp_2002_;
}
v_resetjp_2002_:
{
lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2008_; 
v___x_2005_ = lean_nat_sub(v___x_1985_, v___x_1979_);
lean_dec(v___x_1979_);
lean_dec(v___x_1985_);
v___x_2006_ = lean_nat_add(v_ruleSelection_1996_, v___x_2005_);
lean_dec(v___x_2005_);
lean_dec(v_ruleSelection_1996_);
if (v_isShared_2004_ == 0)
{
lean_ctor_set(v___x_2003_, 4, v___x_2006_);
v___x_2008_ = v___x_2003_;
goto v_reusejp_2007_;
}
else
{
lean_object* v_reuseFailAlloc_2016_; 
v_reuseFailAlloc_2016_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2016_, 0, v_total_1992_);
lean_ctor_set(v_reuseFailAlloc_2016_, 1, v_configParsing_1993_);
lean_ctor_set(v_reuseFailAlloc_2016_, 2, v_ruleSetConstruction_1994_);
lean_ctor_set(v_reuseFailAlloc_2016_, 3, v_search_1995_);
lean_ctor_set(v_reuseFailAlloc_2016_, 4, v___x_2006_);
lean_ctor_set(v_reuseFailAlloc_2016_, 5, v_script_1997_);
lean_ctor_set(v_reuseFailAlloc_2016_, 6, v_forwardState_1998_);
lean_ctor_set(v_reuseFailAlloc_2016_, 7, v_scriptGenerated_1999_);
lean_ctor_set(v_reuseFailAlloc_2016_, 8, v_ruleStats_2000_);
lean_ctor_set(v_reuseFailAlloc_2016_, 9, v_goalStats_2001_);
v___x_2008_ = v_reuseFailAlloc_2016_;
goto v_reusejp_2007_;
}
v_reusejp_2007_:
{
lean_object* v___x_2010_; 
if (v_isShared_1991_ == 0)
{
lean_ctor_set(v___x_1990_, 1, v___x_2008_);
v___x_2010_ = v___x_1990_;
goto v_reusejp_2009_;
}
else
{
lean_object* v_reuseFailAlloc_2015_; 
v_reuseFailAlloc_2015_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2015_, 0, v_rulePatternCache_1988_);
lean_ctor_set(v_reuseFailAlloc_2015_, 1, v___x_2008_);
v___x_2010_ = v_reuseFailAlloc_2015_;
goto v_reusejp_2009_;
}
v_reusejp_2009_:
{
lean_object* v___x_2011_; lean_object* v___x_2013_; 
v___x_2011_ = lean_st_ref_set(v___y_1972_, v___x_2010_);
if (v_isShared_1984_ == 0)
{
v___x_2013_ = v___x_1983_;
goto v_reusejp_2012_;
}
else
{
lean_object* v_reuseFailAlloc_2014_; 
v_reuseFailAlloc_2014_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2014_, 0, v_a_1981_);
v___x_2013_ = v_reuseFailAlloc_2014_;
goto v_reusejp_2012_;
}
v_reusejp_2012_:
{
return v___x_2013_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_1979_);
return v___x_1980_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__1___boxed(lean_object* v_rs_2020_, lean_object* v___x_2021_, lean_object* v_goal_2022_, lean_object* v___f_2023_, lean_object* v_____do__lift_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_, lean_object* v___y_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_){
_start:
{
uint8_t v_____do__lift_120849__boxed_2033_; lean_object* v_res_2034_; 
v_____do__lift_120849__boxed_2033_ = lean_unbox(v_____do__lift_2024_);
v_res_2034_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__1(v_rs_2020_, v___x_2021_, v_goal_2022_, v___f_2023_, v_____do__lift_120849__boxed_2033_, v___y_2025_, v___y_2026_, v___y_2027_, v___y_2028_, v___y_2029_, v___y_2030_, v___y_2031_);
lean_dec(v___y_2031_);
lean_dec_ref(v___y_2030_);
lean_dec(v___y_2029_);
lean_dec_ref(v___y_2028_);
lean_dec(v___y_2027_);
lean_dec(v___y_2026_);
lean_dec_ref(v___y_2025_);
lean_dec_ref(v___x_2021_);
return v_res_2034_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0_spec__0(lean_object* v_e_2035_){
_start:
{
if (lean_obj_tag(v_e_2035_) == 0)
{
uint8_t v___x_2036_; 
v___x_2036_ = 2;
return v___x_2036_;
}
else
{
uint8_t v___x_2037_; 
v___x_2037_ = 0;
return v___x_2037_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0_spec__0___boxed(lean_object* v_e_2038_){
_start:
{
uint8_t v_res_2039_; lean_object* v_r_2040_; 
v_res_2039_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0_spec__0(v_e_2038_);
lean_dec_ref(v_e_2038_);
v_r_2040_ = lean_box(v_res_2039_);
return v_r_2040_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(lean_object* v_cls_2041_, uint8_t v_collapsed_2042_, lean_object* v_tag_2043_, lean_object* v_opts_2044_, uint8_t v_clsEnabled_2045_, lean_object* v_oldTraces_2046_, lean_object* v_msg_2047_, lean_object* v_resStartStop_2048_, lean_object* v___y_2049_, lean_object* v___y_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_){
_start:
{
lean_object* v_fst_2057_; lean_object* v_snd_2058_; lean_object* v___y_2060_; lean_object* v___y_2061_; lean_object* v_data_2062_; lean_object* v_fst_2073_; lean_object* v_snd_2074_; lean_object* v___x_2075_; uint8_t v___x_2076_; lean_object* v___y_2078_; lean_object* v_a_2079_; uint8_t v___y_2094_; double v___y_2125_; 
v_fst_2057_ = lean_ctor_get(v_resStartStop_2048_, 0);
lean_inc(v_fst_2057_);
v_snd_2058_ = lean_ctor_get(v_resStartStop_2048_, 1);
lean_inc(v_snd_2058_);
lean_dec_ref(v_resStartStop_2048_);
v_fst_2073_ = lean_ctor_get(v_snd_2058_, 0);
lean_inc(v_fst_2073_);
v_snd_2074_ = lean_ctor_get(v_snd_2058_, 1);
lean_inc(v_snd_2074_);
lean_dec(v_snd_2058_);
v___x_2075_ = l_Lean_trace_profiler;
v___x_2076_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_2044_, v___x_2075_);
if (v___x_2076_ == 0)
{
v___y_2094_ = v___x_2076_;
goto v___jp_2093_;
}
else
{
lean_object* v___x_2130_; uint8_t v___x_2131_; 
v___x_2130_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2131_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_2044_, v___x_2130_);
if (v___x_2131_ == 0)
{
lean_object* v___x_2132_; lean_object* v___x_2133_; double v___x_2134_; double v___x_2135_; double v___x_2136_; 
v___x_2132_ = l_Lean_trace_profiler_threshold;
v___x_2133_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_2044_, v___x_2132_);
v___x_2134_ = lean_float_of_nat(v___x_2133_);
v___x_2135_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2);
v___x_2136_ = lean_float_div(v___x_2134_, v___x_2135_);
v___y_2125_ = v___x_2136_;
goto v___jp_2124_;
}
else
{
lean_object* v___x_2137_; lean_object* v___x_2138_; double v___x_2139_; 
v___x_2137_ = l_Lean_trace_profiler_threshold;
v___x_2138_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_2044_, v___x_2137_);
v___x_2139_ = lean_float_of_nat(v___x_2138_);
v___y_2125_ = v___x_2139_;
goto v___jp_2124_;
}
}
v___jp_2059_:
{
lean_object* v___x_2063_; 
lean_inc(v___y_2061_);
v___x_2063_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg(v_oldTraces_2046_, v_data_2062_, v___y_2061_, v___y_2060_, v___y_2052_, v___y_2053_, v___y_2054_, v___y_2055_);
if (lean_obj_tag(v___x_2063_) == 0)
{
lean_object* v___x_2064_; 
lean_dec_ref_known(v___x_2063_, 1);
v___x_2064_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_fst_2057_);
return v___x_2064_;
}
else
{
lean_object* v_a_2065_; lean_object* v___x_2067_; uint8_t v_isShared_2068_; uint8_t v_isSharedCheck_2072_; 
lean_dec(v_fst_2057_);
v_a_2065_ = lean_ctor_get(v___x_2063_, 0);
v_isSharedCheck_2072_ = !lean_is_exclusive(v___x_2063_);
if (v_isSharedCheck_2072_ == 0)
{
v___x_2067_ = v___x_2063_;
v_isShared_2068_ = v_isSharedCheck_2072_;
goto v_resetjp_2066_;
}
else
{
lean_inc(v_a_2065_);
lean_dec(v___x_2063_);
v___x_2067_ = lean_box(0);
v_isShared_2068_ = v_isSharedCheck_2072_;
goto v_resetjp_2066_;
}
v_resetjp_2066_:
{
lean_object* v___x_2070_; 
if (v_isShared_2068_ == 0)
{
v___x_2070_ = v___x_2067_;
goto v_reusejp_2069_;
}
else
{
lean_object* v_reuseFailAlloc_2071_; 
v_reuseFailAlloc_2071_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2071_, 0, v_a_2065_);
v___x_2070_ = v_reuseFailAlloc_2071_;
goto v_reusejp_2069_;
}
v_reusejp_2069_:
{
return v___x_2070_;
}
}
}
}
v___jp_2077_:
{
uint8_t v_result_2080_; lean_object* v___x_2081_; lean_object* v___x_2082_; double v___x_2083_; lean_object* v_data_2084_; 
v_result_2080_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0_spec__0(v_fst_2057_);
v___x_2081_ = lean_box(v_result_2080_);
v___x_2082_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2082_, 0, v___x_2081_);
v___x_2083_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0);
lean_inc_ref(v_tag_2043_);
lean_inc_ref(v___x_2082_);
lean_inc(v_cls_2041_);
v_data_2084_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2084_, 0, v_cls_2041_);
lean_ctor_set(v_data_2084_, 1, v___x_2082_);
lean_ctor_set(v_data_2084_, 2, v_tag_2043_);
lean_ctor_set_float(v_data_2084_, sizeof(void*)*3, v___x_2083_);
lean_ctor_set_float(v_data_2084_, sizeof(void*)*3 + 8, v___x_2083_);
lean_ctor_set_uint8(v_data_2084_, sizeof(void*)*3 + 16, v_collapsed_2042_);
if (v___x_2076_ == 0)
{
lean_dec_ref_known(v___x_2082_, 1);
lean_dec(v_snd_2074_);
lean_dec(v_fst_2073_);
lean_dec_ref(v_tag_2043_);
lean_dec(v_cls_2041_);
v___y_2060_ = v_a_2079_;
v___y_2061_ = v___y_2078_;
v_data_2062_ = v_data_2084_;
goto v___jp_2059_;
}
else
{
lean_object* v_data_2085_; double v___x_2086_; double v___x_2087_; 
lean_dec_ref_known(v_data_2084_, 3);
v_data_2085_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2085_, 0, v_cls_2041_);
lean_ctor_set(v_data_2085_, 1, v___x_2082_);
lean_ctor_set(v_data_2085_, 2, v_tag_2043_);
v___x_2086_ = lean_unbox_float(v_fst_2073_);
lean_dec(v_fst_2073_);
lean_ctor_set_float(v_data_2085_, sizeof(void*)*3, v___x_2086_);
v___x_2087_ = lean_unbox_float(v_snd_2074_);
lean_dec(v_snd_2074_);
lean_ctor_set_float(v_data_2085_, sizeof(void*)*3 + 8, v___x_2087_);
lean_ctor_set_uint8(v_data_2085_, sizeof(void*)*3 + 16, v_collapsed_2042_);
v___y_2060_ = v_a_2079_;
v___y_2061_ = v___y_2078_;
v_data_2062_ = v_data_2085_;
goto v___jp_2059_;
}
}
v___jp_2088_:
{
lean_object* v_ref_2089_; lean_object* v___x_2090_; 
v_ref_2089_ = lean_ctor_get(v___y_2054_, 5);
lean_inc(v___y_2055_);
lean_inc_ref(v___y_2054_);
lean_inc(v___y_2053_);
lean_inc_ref(v___y_2052_);
lean_inc(v___y_2051_);
lean_inc(v___y_2050_);
lean_inc_ref(v___y_2049_);
lean_inc(v_fst_2057_);
v___x_2090_ = lean_apply_9(v_msg_2047_, v_fst_2057_, v___y_2049_, v___y_2050_, v___y_2051_, v___y_2052_, v___y_2053_, v___y_2054_, v___y_2055_, lean_box(0));
if (lean_obj_tag(v___x_2090_) == 0)
{
lean_object* v_a_2091_; 
v_a_2091_ = lean_ctor_get(v___x_2090_, 0);
lean_inc(v_a_2091_);
lean_dec_ref_known(v___x_2090_, 1);
v___y_2078_ = v_ref_2089_;
v_a_2079_ = v_a_2091_;
goto v___jp_2077_;
}
else
{
lean_object* v___x_2092_; 
lean_dec_ref_known(v___x_2090_, 1);
v___x_2092_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1);
v___y_2078_ = v_ref_2089_;
v_a_2079_ = v___x_2092_;
goto v___jp_2077_;
}
}
v___jp_2093_:
{
if (v_clsEnabled_2045_ == 0)
{
if (v___y_2094_ == 0)
{
lean_object* v___x_2095_; lean_object* v_traceState_2096_; lean_object* v_env_2097_; lean_object* v_nextMacroScope_2098_; lean_object* v_ngen_2099_; lean_object* v_auxDeclNGen_2100_; lean_object* v_cache_2101_; lean_object* v_messages_2102_; lean_object* v_infoState_2103_; lean_object* v_snapshotTasks_2104_; lean_object* v___x_2106_; uint8_t v_isShared_2107_; uint8_t v_isSharedCheck_2123_; 
lean_dec(v_snd_2074_);
lean_dec(v_fst_2073_);
lean_dec_ref(v_msg_2047_);
lean_dec_ref(v_tag_2043_);
lean_dec(v_cls_2041_);
v___x_2095_ = lean_st_ref_take(v___y_2055_);
v_traceState_2096_ = lean_ctor_get(v___x_2095_, 4);
v_env_2097_ = lean_ctor_get(v___x_2095_, 0);
v_nextMacroScope_2098_ = lean_ctor_get(v___x_2095_, 1);
v_ngen_2099_ = lean_ctor_get(v___x_2095_, 2);
v_auxDeclNGen_2100_ = lean_ctor_get(v___x_2095_, 3);
v_cache_2101_ = lean_ctor_get(v___x_2095_, 5);
v_messages_2102_ = lean_ctor_get(v___x_2095_, 6);
v_infoState_2103_ = lean_ctor_get(v___x_2095_, 7);
v_snapshotTasks_2104_ = lean_ctor_get(v___x_2095_, 8);
v_isSharedCheck_2123_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2123_ == 0)
{
v___x_2106_ = v___x_2095_;
v_isShared_2107_ = v_isSharedCheck_2123_;
goto v_resetjp_2105_;
}
else
{
lean_inc(v_snapshotTasks_2104_);
lean_inc(v_infoState_2103_);
lean_inc(v_messages_2102_);
lean_inc(v_cache_2101_);
lean_inc(v_traceState_2096_);
lean_inc(v_auxDeclNGen_2100_);
lean_inc(v_ngen_2099_);
lean_inc(v_nextMacroScope_2098_);
lean_inc(v_env_2097_);
lean_dec(v___x_2095_);
v___x_2106_ = lean_box(0);
v_isShared_2107_ = v_isSharedCheck_2123_;
goto v_resetjp_2105_;
}
v_resetjp_2105_:
{
uint64_t v_tid_2108_; lean_object* v_traces_2109_; lean_object* v___x_2111_; uint8_t v_isShared_2112_; uint8_t v_isSharedCheck_2122_; 
v_tid_2108_ = lean_ctor_get_uint64(v_traceState_2096_, sizeof(void*)*1);
v_traces_2109_ = lean_ctor_get(v_traceState_2096_, 0);
v_isSharedCheck_2122_ = !lean_is_exclusive(v_traceState_2096_);
if (v_isSharedCheck_2122_ == 0)
{
v___x_2111_ = v_traceState_2096_;
v_isShared_2112_ = v_isSharedCheck_2122_;
goto v_resetjp_2110_;
}
else
{
lean_inc(v_traces_2109_);
lean_dec(v_traceState_2096_);
v___x_2111_ = lean_box(0);
v_isShared_2112_ = v_isSharedCheck_2122_;
goto v_resetjp_2110_;
}
v_resetjp_2110_:
{
lean_object* v___x_2113_; lean_object* v___x_2115_; 
v___x_2113_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2046_, v_traces_2109_);
lean_dec_ref(v_traces_2109_);
if (v_isShared_2112_ == 0)
{
lean_ctor_set(v___x_2111_, 0, v___x_2113_);
v___x_2115_ = v___x_2111_;
goto v_reusejp_2114_;
}
else
{
lean_object* v_reuseFailAlloc_2121_; 
v_reuseFailAlloc_2121_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2121_, 0, v___x_2113_);
lean_ctor_set_uint64(v_reuseFailAlloc_2121_, sizeof(void*)*1, v_tid_2108_);
v___x_2115_ = v_reuseFailAlloc_2121_;
goto v_reusejp_2114_;
}
v_reusejp_2114_:
{
lean_object* v___x_2117_; 
if (v_isShared_2107_ == 0)
{
lean_ctor_set(v___x_2106_, 4, v___x_2115_);
v___x_2117_ = v___x_2106_;
goto v_reusejp_2116_;
}
else
{
lean_object* v_reuseFailAlloc_2120_; 
v_reuseFailAlloc_2120_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2120_, 0, v_env_2097_);
lean_ctor_set(v_reuseFailAlloc_2120_, 1, v_nextMacroScope_2098_);
lean_ctor_set(v_reuseFailAlloc_2120_, 2, v_ngen_2099_);
lean_ctor_set(v_reuseFailAlloc_2120_, 3, v_auxDeclNGen_2100_);
lean_ctor_set(v_reuseFailAlloc_2120_, 4, v___x_2115_);
lean_ctor_set(v_reuseFailAlloc_2120_, 5, v_cache_2101_);
lean_ctor_set(v_reuseFailAlloc_2120_, 6, v_messages_2102_);
lean_ctor_set(v_reuseFailAlloc_2120_, 7, v_infoState_2103_);
lean_ctor_set(v_reuseFailAlloc_2120_, 8, v_snapshotTasks_2104_);
v___x_2117_ = v_reuseFailAlloc_2120_;
goto v_reusejp_2116_;
}
v_reusejp_2116_:
{
lean_object* v___x_2118_; lean_object* v___x_2119_; 
v___x_2118_ = lean_st_ref_set(v___y_2055_, v___x_2117_);
v___x_2119_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_fst_2057_);
return v___x_2119_;
}
}
}
}
}
else
{
goto v___jp_2088_;
}
}
else
{
goto v___jp_2088_;
}
}
v___jp_2124_:
{
double v___x_2126_; double v___x_2127_; double v___x_2128_; uint8_t v___x_2129_; 
v___x_2126_ = lean_unbox_float(v_snd_2074_);
v___x_2127_ = lean_unbox_float(v_fst_2073_);
v___x_2128_ = lean_float_sub(v___x_2126_, v___x_2127_);
v___x_2129_ = lean_float_decLt(v___y_2125_, v___x_2128_);
v___y_2094_ = v___x_2129_;
goto v___jp_2093_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0___boxed(lean_object* v_cls_2140_, lean_object* v_collapsed_2141_, lean_object* v_tag_2142_, lean_object* v_opts_2143_, lean_object* v_clsEnabled_2144_, lean_object* v_oldTraces_2145_, lean_object* v_msg_2146_, lean_object* v_resStartStop_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_, lean_object* v___y_2150_, lean_object* v___y_2151_, lean_object* v___y_2152_, lean_object* v___y_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_){
_start:
{
uint8_t v_collapsed_boxed_2156_; uint8_t v_clsEnabled_boxed_2157_; lean_object* v_res_2158_; 
v_collapsed_boxed_2156_ = lean_unbox(v_collapsed_2141_);
v_clsEnabled_boxed_2157_ = lean_unbox(v_clsEnabled_2144_);
v_res_2158_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(v_cls_2140_, v_collapsed_boxed_2156_, v_tag_2142_, v_opts_2143_, v_clsEnabled_boxed_2157_, v_oldTraces_2145_, v_msg_2146_, v_resStartStop_2147_, v___y_2148_, v___y_2149_, v___y_2150_, v___y_2151_, v___y_2152_, v___y_2153_, v___y_2154_);
lean_dec(v___y_2154_);
lean_dec_ref(v___y_2153_);
lean_dec(v___y_2152_);
lean_dec_ref(v___y_2151_);
lean_dec(v___y_2150_);
lean_dec(v___y_2149_);
lean_dec_ref(v___y_2148_);
lean_dec_ref(v_opts_2143_);
return v_res_2158_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__3(lean_object* v___f_2159_, lean_object* v___f_2160_, lean_object* v___f_2161_, lean_object* v___x_2162_, uint8_t v___x_2163_, lean_object* v___x_2164_, lean_object* v___f_2165_, lean_object* v_rs_2166_, lean_object* v___x_2167_, lean_object* v_goal_2168_, lean_object* v___f_2169_, lean_object* v_opts_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_){
_start:
{
uint8_t v___y_2180_; lean_object* v___y_2181_; lean_object* v___y_2182_; lean_object* v_a_2183_; uint8_t v___y_2193_; lean_object* v___y_2194_; lean_object* v___y_2195_; lean_object* v_a_2196_; uint8_t v___y_2199_; lean_object* v___y_2200_; lean_object* v___y_2201_; lean_object* v_a_2202_; uint8_t v___y_2205_; lean_object* v___y_2206_; lean_object* v___y_2207_; uint8_t v___y_2212_; lean_object* v___y_2213_; lean_object* v___y_2214_; uint8_t v___y_2251_; lean_object* v___y_2252_; lean_object* v___y_2253_; uint8_t v_a_2254_; uint8_t v___y_2256_; lean_object* v___y_2257_; lean_object* v___y_2258_; lean_object* v___y_2259_; uint8_t v___y_2263_; lean_object* v___y_2264_; lean_object* v___y_2265_; lean_object* v_a_2266_; uint8_t v___y_2279_; lean_object* v___y_2280_; lean_object* v___y_2281_; lean_object* v_a_2282_; uint8_t v___y_2285_; lean_object* v___y_2286_; lean_object* v___y_2287_; lean_object* v_a_2288_; uint8_t v___y_2291_; lean_object* v___y_2292_; lean_object* v___y_2293_; uint8_t v___y_2330_; lean_object* v___y_2331_; lean_object* v___y_2332_; uint8_t v___y_2337_; lean_object* v___y_2338_; lean_object* v___y_2339_; lean_object* v___y_2340_; uint8_t v_hasTrace_2343_; 
v_hasTrace_2343_ = lean_ctor_get_uint8(v_opts_2170_, sizeof(void*)*1);
if (v_hasTrace_2343_ == 0)
{
lean_object* v___x_2344_; 
lean_dec_ref(v___f_2169_);
lean_dec(v_goal_2168_);
lean_dec_ref(v_rs_2166_);
lean_dec_ref(v___f_2165_);
lean_dec_ref(v___x_2164_);
lean_dec(v___x_2162_);
lean_inc(v___y_2177_);
lean_inc_ref(v___y_2176_);
lean_inc(v___y_2175_);
lean_inc_ref(v___y_2174_);
lean_inc(v___y_2173_);
lean_inc(v___y_2172_);
lean_inc_ref(v___y_2171_);
v___x_2344_ = lean_apply_8(v___f_2159_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_, lean_box(0));
if (lean_obj_tag(v___x_2344_) == 0)
{
lean_object* v_a_2345_; lean_object* v___x_2346_; 
v_a_2345_ = lean_ctor_get(v___x_2344_, 0);
lean_inc(v_a_2345_);
lean_dec_ref_known(v___x_2344_, 1);
lean_inc(v___y_2177_);
lean_inc_ref(v___y_2176_);
lean_inc(v___y_2175_);
lean_inc_ref(v___y_2174_);
lean_inc(v___y_2173_);
lean_inc(v___y_2172_);
lean_inc_ref(v___y_2171_);
v___x_2346_ = lean_apply_9(v___f_2160_, v_a_2345_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_, lean_box(0));
if (lean_obj_tag(v___x_2346_) == 0)
{
lean_object* v_a_2347_; lean_object* v___x_2348_; 
v_a_2347_ = lean_ctor_get(v___x_2346_, 0);
lean_inc(v_a_2347_);
lean_dec_ref_known(v___x_2346_, 1);
lean_inc(v___y_2177_);
lean_inc_ref(v___y_2176_);
lean_inc(v___y_2175_);
lean_inc_ref(v___y_2174_);
lean_inc(v___y_2173_);
lean_inc(v___y_2172_);
lean_inc_ref(v___y_2171_);
v___x_2348_ = lean_apply_9(v___f_2161_, v_a_2347_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_, lean_box(0));
return v___x_2348_;
}
else
{
lean_object* v_a_2349_; lean_object* v___x_2351_; uint8_t v_isShared_2352_; uint8_t v_isSharedCheck_2356_; 
lean_dec_ref(v___f_2161_);
v_a_2349_ = lean_ctor_get(v___x_2346_, 0);
v_isSharedCheck_2356_ = !lean_is_exclusive(v___x_2346_);
if (v_isSharedCheck_2356_ == 0)
{
v___x_2351_ = v___x_2346_;
v_isShared_2352_ = v_isSharedCheck_2356_;
goto v_resetjp_2350_;
}
else
{
lean_inc(v_a_2349_);
lean_dec(v___x_2346_);
v___x_2351_ = lean_box(0);
v_isShared_2352_ = v_isSharedCheck_2356_;
goto v_resetjp_2350_;
}
v_resetjp_2350_:
{
lean_object* v___x_2354_; 
if (v_isShared_2352_ == 0)
{
v___x_2354_ = v___x_2351_;
goto v_reusejp_2353_;
}
else
{
lean_object* v_reuseFailAlloc_2355_; 
v_reuseFailAlloc_2355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2355_, 0, v_a_2349_);
v___x_2354_ = v_reuseFailAlloc_2355_;
goto v_reusejp_2353_;
}
v_reusejp_2353_:
{
return v___x_2354_;
}
}
}
}
else
{
lean_object* v_a_2357_; lean_object* v___x_2359_; uint8_t v_isShared_2360_; uint8_t v_isSharedCheck_2364_; 
lean_dec_ref(v___f_2161_);
lean_dec_ref(v___f_2160_);
v_a_2357_ = lean_ctor_get(v___x_2344_, 0);
v_isSharedCheck_2364_ = !lean_is_exclusive(v___x_2344_);
if (v_isSharedCheck_2364_ == 0)
{
v___x_2359_ = v___x_2344_;
v_isShared_2360_ = v_isSharedCheck_2364_;
goto v_resetjp_2358_;
}
else
{
lean_inc(v_a_2357_);
lean_dec(v___x_2344_);
v___x_2359_ = lean_box(0);
v_isShared_2360_ = v_isSharedCheck_2364_;
goto v_resetjp_2358_;
}
v_resetjp_2358_:
{
lean_object* v___x_2362_; 
if (v_isShared_2360_ == 0)
{
v___x_2362_ = v___x_2359_;
goto v_reusejp_2361_;
}
else
{
lean_object* v_reuseFailAlloc_2363_; 
v_reuseFailAlloc_2363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2363_, 0, v_a_2357_);
v___x_2362_ = v_reuseFailAlloc_2363_;
goto v_reusejp_2361_;
}
v_reusejp_2361_:
{
return v___x_2362_;
}
}
}
}
else
{
lean_object* v_options_2365_; lean_object* v_inheritedTraceOptions_2366_; uint8_t v___y_2368_; uint8_t v_a_2394_; uint8_t v_hasTrace_2418_; 
v_options_2365_ = lean_ctor_get(v___y_2176_, 2);
v_inheritedTraceOptions_2366_ = lean_ctor_get(v___y_2176_, 13);
v_hasTrace_2418_ = lean_ctor_get_uint8(v_options_2365_, sizeof(void*)*1);
if (v_hasTrace_2418_ == 0)
{
v_a_2394_ = v_hasTrace_2418_;
goto v___jp_2393_;
}
else
{
lean_object* v___x_2419_; lean_object* v___x_2420_; uint8_t v___x_2421_; 
v___x_2419_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1));
lean_inc(v___x_2162_);
v___x_2420_ = l_Lean_Name_append(v___x_2419_, v___x_2162_);
v___x_2421_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2366_, v_options_2365_, v___x_2420_);
lean_dec(v___x_2420_);
if (v___x_2421_ == 0)
{
v_a_2394_ = v___x_2421_;
goto v___jp_2393_;
}
else
{
lean_dec_ref(v___f_2161_);
lean_dec_ref(v___f_2160_);
lean_dec_ref(v___f_2159_);
v___y_2368_ = v___x_2421_;
goto v___jp_2367_;
}
}
v___jp_2367_:
{
lean_object* v___x_2369_; lean_object* v_a_2370_; lean_object* v___x_2371_; uint8_t v___x_2372_; 
v___x_2369_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v___y_2177_);
v_a_2370_ = lean_ctor_get(v___x_2369_, 0);
lean_inc(v_a_2370_);
lean_dec_ref(v___x_2369_);
v___x_2371_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2372_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_2170_, v___x_2371_);
if (v___x_2372_ == 0)
{
lean_object* v___x_2373_; lean_object* v___x_2374_; uint8_t v___x_2375_; 
v___x_2373_ = lean_io_mono_nanos_now();
v___x_2374_ = lp_aesop_Aesop_aesop_collectStats;
v___x_2375_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2365_, v___x_2374_);
if (v___x_2375_ == 0)
{
lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v_a_2378_; uint8_t v___x_2379_; 
v___x_2376_ = lp_aesop_Aesop_TraceOption_stats;
v___x_2377_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_2376_, v___y_2176_);
v_a_2378_ = lean_ctor_get(v___x_2377_, 0);
lean_inc(v_a_2378_);
v___x_2379_ = lean_unbox(v_a_2378_);
lean_dec(v_a_2378_);
if (v___x_2379_ == 0)
{
lean_object* v___x_2380_; lean_object* v___x_2381_; uint8_t v___x_2382_; 
lean_dec_ref(v___x_2377_);
v___x_2380_ = lp_aesop_Aesop_aesop_stats_file;
v___x_2381_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_2365_, v___x_2380_);
v___x_2382_ = lean_string_dec_eq(v___x_2381_, v___x_2164_);
lean_dec_ref(v___x_2381_);
if (v___x_2382_ == 0)
{
v___y_2291_ = v___y_2368_;
v___y_2292_ = v_a_2370_;
v___y_2293_ = v___x_2373_;
goto v___jp_2290_;
}
else
{
v___y_2330_ = v___y_2368_;
v___y_2331_ = v_a_2370_;
v___y_2332_ = v___x_2373_;
goto v___jp_2329_;
}
}
else
{
v___y_2337_ = v___y_2368_;
v___y_2338_ = v_a_2370_;
v___y_2339_ = v___x_2373_;
v___y_2340_ = v___x_2377_;
goto v___jp_2336_;
}
}
else
{
v___y_2291_ = v___y_2368_;
v___y_2292_ = v_a_2370_;
v___y_2293_ = v___x_2373_;
goto v___jp_2290_;
}
}
else
{
lean_object* v___x_2383_; lean_object* v___x_2384_; uint8_t v___x_2385_; 
v___x_2383_ = lean_io_get_num_heartbeats();
v___x_2384_ = lp_aesop_Aesop_aesop_collectStats;
v___x_2385_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2365_, v___x_2384_);
if (v___x_2385_ == 0)
{
lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v_a_2388_; uint8_t v___x_2389_; 
v___x_2386_ = lp_aesop_Aesop_TraceOption_stats;
v___x_2387_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_2386_, v___y_2176_);
v_a_2388_ = lean_ctor_get(v___x_2387_, 0);
lean_inc(v_a_2388_);
v___x_2389_ = lean_unbox(v_a_2388_);
lean_dec(v_a_2388_);
if (v___x_2389_ == 0)
{
lean_object* v___x_2390_; lean_object* v___x_2391_; uint8_t v___x_2392_; 
lean_dec_ref(v___x_2387_);
v___x_2390_ = lp_aesop_Aesop_aesop_stats_file;
v___x_2391_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_2365_, v___x_2390_);
v___x_2392_ = lean_string_dec_eq(v___x_2391_, v___x_2164_);
lean_dec_ref(v___x_2391_);
if (v___x_2392_ == 0)
{
v___y_2251_ = v___y_2368_;
v___y_2252_ = v___x_2383_;
v___y_2253_ = v_a_2370_;
v_a_2254_ = v___x_2372_;
goto v___jp_2250_;
}
else
{
v___y_2205_ = v___y_2368_;
v___y_2206_ = v___x_2383_;
v___y_2207_ = v_a_2370_;
goto v___jp_2204_;
}
}
else
{
v___y_2256_ = v___y_2368_;
v___y_2257_ = v___x_2383_;
v___y_2258_ = v_a_2370_;
v___y_2259_ = v___x_2387_;
goto v___jp_2255_;
}
}
else
{
v___y_2212_ = v___y_2368_;
v___y_2213_ = v___x_2383_;
v___y_2214_ = v_a_2370_;
goto v___jp_2211_;
}
}
}
v___jp_2393_:
{
lean_object* v___x_2395_; uint8_t v___x_2396_; 
v___x_2395_ = l_Lean_trace_profiler;
v___x_2396_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_2170_, v___x_2395_);
if (v___x_2396_ == 0)
{
lean_object* v___x_2397_; 
lean_dec_ref(v___f_2169_);
lean_dec(v_goal_2168_);
lean_dec_ref(v_rs_2166_);
lean_dec_ref(v___f_2165_);
lean_dec_ref(v___x_2164_);
lean_dec(v___x_2162_);
lean_inc(v___y_2177_);
lean_inc_ref(v___y_2176_);
lean_inc(v___y_2175_);
lean_inc_ref(v___y_2174_);
lean_inc(v___y_2173_);
lean_inc(v___y_2172_);
lean_inc_ref(v___y_2171_);
v___x_2397_ = lean_apply_8(v___f_2159_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_, lean_box(0));
if (lean_obj_tag(v___x_2397_) == 0)
{
lean_object* v_a_2398_; lean_object* v___x_2399_; 
v_a_2398_ = lean_ctor_get(v___x_2397_, 0);
lean_inc(v_a_2398_);
lean_dec_ref_known(v___x_2397_, 1);
lean_inc(v___y_2177_);
lean_inc_ref(v___y_2176_);
lean_inc(v___y_2175_);
lean_inc_ref(v___y_2174_);
lean_inc(v___y_2173_);
lean_inc(v___y_2172_);
lean_inc_ref(v___y_2171_);
v___x_2399_ = lean_apply_9(v___f_2160_, v_a_2398_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_, lean_box(0));
if (lean_obj_tag(v___x_2399_) == 0)
{
lean_object* v_a_2400_; lean_object* v___x_2401_; 
v_a_2400_ = lean_ctor_get(v___x_2399_, 0);
lean_inc(v_a_2400_);
lean_dec_ref_known(v___x_2399_, 1);
lean_inc(v___y_2177_);
lean_inc_ref(v___y_2176_);
lean_inc(v___y_2175_);
lean_inc_ref(v___y_2174_);
lean_inc(v___y_2173_);
lean_inc(v___y_2172_);
lean_inc_ref(v___y_2171_);
v___x_2401_ = lean_apply_9(v___f_2161_, v_a_2400_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_, lean_box(0));
return v___x_2401_;
}
else
{
lean_object* v_a_2402_; lean_object* v___x_2404_; uint8_t v_isShared_2405_; uint8_t v_isSharedCheck_2409_; 
lean_dec_ref(v___f_2161_);
v_a_2402_ = lean_ctor_get(v___x_2399_, 0);
v_isSharedCheck_2409_ = !lean_is_exclusive(v___x_2399_);
if (v_isSharedCheck_2409_ == 0)
{
v___x_2404_ = v___x_2399_;
v_isShared_2405_ = v_isSharedCheck_2409_;
goto v_resetjp_2403_;
}
else
{
lean_inc(v_a_2402_);
lean_dec(v___x_2399_);
v___x_2404_ = lean_box(0);
v_isShared_2405_ = v_isSharedCheck_2409_;
goto v_resetjp_2403_;
}
v_resetjp_2403_:
{
lean_object* v___x_2407_; 
if (v_isShared_2405_ == 0)
{
v___x_2407_ = v___x_2404_;
goto v_reusejp_2406_;
}
else
{
lean_object* v_reuseFailAlloc_2408_; 
v_reuseFailAlloc_2408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2408_, 0, v_a_2402_);
v___x_2407_ = v_reuseFailAlloc_2408_;
goto v_reusejp_2406_;
}
v_reusejp_2406_:
{
return v___x_2407_;
}
}
}
}
else
{
lean_object* v_a_2410_; lean_object* v___x_2412_; uint8_t v_isShared_2413_; uint8_t v_isSharedCheck_2417_; 
lean_dec_ref(v___f_2161_);
lean_dec_ref(v___f_2160_);
v_a_2410_ = lean_ctor_get(v___x_2397_, 0);
v_isSharedCheck_2417_ = !lean_is_exclusive(v___x_2397_);
if (v_isSharedCheck_2417_ == 0)
{
v___x_2412_ = v___x_2397_;
v_isShared_2413_ = v_isSharedCheck_2417_;
goto v_resetjp_2411_;
}
else
{
lean_inc(v_a_2410_);
lean_dec(v___x_2397_);
v___x_2412_ = lean_box(0);
v_isShared_2413_ = v_isSharedCheck_2417_;
goto v_resetjp_2411_;
}
v_resetjp_2411_:
{
lean_object* v___x_2415_; 
if (v_isShared_2413_ == 0)
{
v___x_2415_ = v___x_2412_;
goto v_reusejp_2414_;
}
else
{
lean_object* v_reuseFailAlloc_2416_; 
v_reuseFailAlloc_2416_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2416_, 0, v_a_2410_);
v___x_2415_ = v_reuseFailAlloc_2416_;
goto v_reusejp_2414_;
}
v_reusejp_2414_:
{
return v___x_2415_;
}
}
}
}
else
{
lean_dec_ref(v___f_2161_);
lean_dec_ref(v___f_2160_);
lean_dec_ref(v___f_2159_);
v___y_2368_ = v_a_2394_;
goto v___jp_2367_;
}
}
}
v___jp_2179_:
{
lean_object* v___x_2184_; double v___x_2185_; double v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2191_; 
v___x_2184_ = lean_io_get_num_heartbeats();
v___x_2185_ = lean_float_of_nat(v___y_2181_);
v___x_2186_ = lean_float_of_nat(v___x_2184_);
v___x_2187_ = lean_box_float(v___x_2185_);
v___x_2188_ = lean_box_float(v___x_2186_);
v___x_2189_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2189_, 0, v___x_2187_);
lean_ctor_set(v___x_2189_, 1, v___x_2188_);
v___x_2190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2190_, 0, v_a_2183_);
lean_ctor_set(v___x_2190_, 1, v___x_2189_);
v___x_2191_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(v___x_2162_, v___x_2163_, v___x_2164_, v_opts_2170_, v___y_2180_, v___y_2182_, v___f_2165_, v___x_2190_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_);
return v___x_2191_;
}
v___jp_2192_:
{
lean_object* v___x_2197_; 
v___x_2197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2197_, 0, v_a_2196_);
v___y_2180_ = v___y_2193_;
v___y_2181_ = v___y_2194_;
v___y_2182_ = v___y_2195_;
v_a_2183_ = v___x_2197_;
goto v___jp_2179_;
}
v___jp_2198_:
{
lean_object* v___x_2203_; 
v___x_2203_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2203_, 0, v_a_2202_);
v___y_2180_ = v___y_2199_;
v___y_2181_ = v___y_2200_;
v___y_2182_ = v___y_2201_;
v_a_2183_ = v___x_2203_;
goto v___jp_2179_;
}
v___jp_2204_:
{
lean_object* v___x_2208_; 
v___x_2208_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2166_, v___x_2167_, v_goal_2168_, v___f_2169_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_);
if (lean_obj_tag(v___x_2208_) == 0)
{
lean_object* v_a_2209_; 
v_a_2209_ = lean_ctor_get(v___x_2208_, 0);
lean_inc(v_a_2209_);
lean_dec_ref_known(v___x_2208_, 1);
v___y_2193_ = v___y_2205_;
v___y_2194_ = v___y_2206_;
v___y_2195_ = v___y_2207_;
v_a_2196_ = v_a_2209_;
goto v___jp_2192_;
}
else
{
lean_object* v_a_2210_; 
v_a_2210_ = lean_ctor_get(v___x_2208_, 0);
lean_inc(v_a_2210_);
lean_dec_ref_known(v___x_2208_, 1);
v___y_2199_ = v___y_2205_;
v___y_2200_ = v___y_2206_;
v___y_2201_ = v___y_2207_;
v_a_2202_ = v_a_2210_;
goto v___jp_2198_;
}
}
v___jp_2211_:
{
lean_object* v___x_2215_; lean_object* v___x_2216_; 
v___x_2215_ = lean_io_mono_nanos_now();
v___x_2216_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2166_, v___x_2167_, v_goal_2168_, v___f_2169_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_);
if (lean_obj_tag(v___x_2216_) == 0)
{
lean_object* v_a_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v_stats_2220_; lean_object* v_rulePatternCache_2221_; lean_object* v___x_2223_; uint8_t v_isShared_2224_; uint8_t v_isSharedCheck_2248_; 
v_a_2217_ = lean_ctor_get(v___x_2216_, 0);
lean_inc(v_a_2217_);
lean_dec_ref_known(v___x_2216_, 1);
v___x_2218_ = lean_io_mono_nanos_now();
v___x_2219_ = lean_st_ref_take(v___y_2173_);
v_stats_2220_ = lean_ctor_get(v___x_2219_, 1);
v_rulePatternCache_2221_ = lean_ctor_get(v___x_2219_, 0);
v_isSharedCheck_2248_ = !lean_is_exclusive(v___x_2219_);
if (v_isSharedCheck_2248_ == 0)
{
v___x_2223_ = v___x_2219_;
v_isShared_2224_ = v_isSharedCheck_2248_;
goto v_resetjp_2222_;
}
else
{
lean_inc(v_stats_2220_);
lean_inc(v_rulePatternCache_2221_);
lean_dec(v___x_2219_);
v___x_2223_ = lean_box(0);
v_isShared_2224_ = v_isSharedCheck_2248_;
goto v_resetjp_2222_;
}
v_resetjp_2222_:
{
lean_object* v_total_2225_; lean_object* v_configParsing_2226_; lean_object* v_ruleSetConstruction_2227_; lean_object* v_search_2228_; lean_object* v_ruleSelection_2229_; lean_object* v_script_2230_; lean_object* v_forwardState_2231_; lean_object* v_scriptGenerated_2232_; lean_object* v_ruleStats_2233_; lean_object* v_goalStats_2234_; lean_object* v___x_2236_; uint8_t v_isShared_2237_; uint8_t v_isSharedCheck_2247_; 
v_total_2225_ = lean_ctor_get(v_stats_2220_, 0);
v_configParsing_2226_ = lean_ctor_get(v_stats_2220_, 1);
v_ruleSetConstruction_2227_ = lean_ctor_get(v_stats_2220_, 2);
v_search_2228_ = lean_ctor_get(v_stats_2220_, 3);
v_ruleSelection_2229_ = lean_ctor_get(v_stats_2220_, 4);
v_script_2230_ = lean_ctor_get(v_stats_2220_, 5);
v_forwardState_2231_ = lean_ctor_get(v_stats_2220_, 6);
v_scriptGenerated_2232_ = lean_ctor_get(v_stats_2220_, 7);
v_ruleStats_2233_ = lean_ctor_get(v_stats_2220_, 8);
v_goalStats_2234_ = lean_ctor_get(v_stats_2220_, 9);
v_isSharedCheck_2247_ = !lean_is_exclusive(v_stats_2220_);
if (v_isSharedCheck_2247_ == 0)
{
v___x_2236_ = v_stats_2220_;
v_isShared_2237_ = v_isSharedCheck_2247_;
goto v_resetjp_2235_;
}
else
{
lean_inc(v_goalStats_2234_);
lean_inc(v_ruleStats_2233_);
lean_inc(v_scriptGenerated_2232_);
lean_inc(v_forwardState_2231_);
lean_inc(v_script_2230_);
lean_inc(v_ruleSelection_2229_);
lean_inc(v_search_2228_);
lean_inc(v_ruleSetConstruction_2227_);
lean_inc(v_configParsing_2226_);
lean_inc(v_total_2225_);
lean_dec(v_stats_2220_);
v___x_2236_ = lean_box(0);
v_isShared_2237_ = v_isSharedCheck_2247_;
goto v_resetjp_2235_;
}
v_resetjp_2235_:
{
lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2241_; 
v___x_2238_ = lean_nat_sub(v___x_2218_, v___x_2215_);
lean_dec(v___x_2215_);
lean_dec(v___x_2218_);
v___x_2239_ = lean_nat_add(v_ruleSelection_2229_, v___x_2238_);
lean_dec(v___x_2238_);
lean_dec(v_ruleSelection_2229_);
if (v_isShared_2237_ == 0)
{
lean_ctor_set(v___x_2236_, 4, v___x_2239_);
v___x_2241_ = v___x_2236_;
goto v_reusejp_2240_;
}
else
{
lean_object* v_reuseFailAlloc_2246_; 
v_reuseFailAlloc_2246_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2246_, 0, v_total_2225_);
lean_ctor_set(v_reuseFailAlloc_2246_, 1, v_configParsing_2226_);
lean_ctor_set(v_reuseFailAlloc_2246_, 2, v_ruleSetConstruction_2227_);
lean_ctor_set(v_reuseFailAlloc_2246_, 3, v_search_2228_);
lean_ctor_set(v_reuseFailAlloc_2246_, 4, v___x_2239_);
lean_ctor_set(v_reuseFailAlloc_2246_, 5, v_script_2230_);
lean_ctor_set(v_reuseFailAlloc_2246_, 6, v_forwardState_2231_);
lean_ctor_set(v_reuseFailAlloc_2246_, 7, v_scriptGenerated_2232_);
lean_ctor_set(v_reuseFailAlloc_2246_, 8, v_ruleStats_2233_);
lean_ctor_set(v_reuseFailAlloc_2246_, 9, v_goalStats_2234_);
v___x_2241_ = v_reuseFailAlloc_2246_;
goto v_reusejp_2240_;
}
v_reusejp_2240_:
{
lean_object* v___x_2243_; 
if (v_isShared_2224_ == 0)
{
lean_ctor_set(v___x_2223_, 1, v___x_2241_);
v___x_2243_ = v___x_2223_;
goto v_reusejp_2242_;
}
else
{
lean_object* v_reuseFailAlloc_2245_; 
v_reuseFailAlloc_2245_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2245_, 0, v_rulePatternCache_2221_);
lean_ctor_set(v_reuseFailAlloc_2245_, 1, v___x_2241_);
v___x_2243_ = v_reuseFailAlloc_2245_;
goto v_reusejp_2242_;
}
v_reusejp_2242_:
{
lean_object* v___x_2244_; 
v___x_2244_ = lean_st_ref_set(v___y_2173_, v___x_2243_);
v___y_2193_ = v___y_2212_;
v___y_2194_ = v___y_2213_;
v___y_2195_ = v___y_2214_;
v_a_2196_ = v_a_2217_;
goto v___jp_2192_;
}
}
}
}
}
else
{
lean_object* v_a_2249_; 
lean_dec(v___x_2215_);
v_a_2249_ = lean_ctor_get(v___x_2216_, 0);
lean_inc(v_a_2249_);
lean_dec_ref_known(v___x_2216_, 1);
v___y_2199_ = v___y_2212_;
v___y_2200_ = v___y_2213_;
v___y_2201_ = v___y_2214_;
v_a_2202_ = v_a_2249_;
goto v___jp_2198_;
}
}
v___jp_2250_:
{
if (v_a_2254_ == 0)
{
v___y_2205_ = v___y_2251_;
v___y_2206_ = v___y_2252_;
v___y_2207_ = v___y_2253_;
goto v___jp_2204_;
}
else
{
v___y_2212_ = v___y_2251_;
v___y_2213_ = v___y_2252_;
v___y_2214_ = v___y_2253_;
goto v___jp_2211_;
}
}
v___jp_2255_:
{
lean_object* v_a_2260_; uint8_t v___x_2261_; 
v_a_2260_ = lean_ctor_get(v___y_2259_, 0);
lean_inc(v_a_2260_);
lean_dec_ref(v___y_2259_);
v___x_2261_ = lean_unbox(v_a_2260_);
lean_dec(v_a_2260_);
v___y_2251_ = v___y_2256_;
v___y_2252_ = v___y_2257_;
v___y_2253_ = v___y_2258_;
v_a_2254_ = v___x_2261_;
goto v___jp_2250_;
}
v___jp_2262_:
{
lean_object* v___x_2267_; double v___x_2268_; double v___x_2269_; double v___x_2270_; double v___x_2271_; double v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; 
v___x_2267_ = lean_io_mono_nanos_now();
v___x_2268_ = lean_float_of_nat(v___y_2265_);
v___x_2269_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_2270_ = lean_float_div(v___x_2268_, v___x_2269_);
v___x_2271_ = lean_float_of_nat(v___x_2267_);
v___x_2272_ = lean_float_div(v___x_2271_, v___x_2269_);
v___x_2273_ = lean_box_float(v___x_2270_);
v___x_2274_ = lean_box_float(v___x_2272_);
v___x_2275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2275_, 0, v___x_2273_);
lean_ctor_set(v___x_2275_, 1, v___x_2274_);
v___x_2276_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2276_, 0, v_a_2266_);
lean_ctor_set(v___x_2276_, 1, v___x_2275_);
v___x_2277_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(v___x_2162_, v___x_2163_, v___x_2164_, v_opts_2170_, v___y_2263_, v___y_2264_, v___f_2165_, v___x_2276_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_);
return v___x_2277_;
}
v___jp_2278_:
{
lean_object* v___x_2283_; 
v___x_2283_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2283_, 0, v_a_2282_);
v___y_2263_ = v___y_2279_;
v___y_2264_ = v___y_2280_;
v___y_2265_ = v___y_2281_;
v_a_2266_ = v___x_2283_;
goto v___jp_2262_;
}
v___jp_2284_:
{
lean_object* v___x_2289_; 
v___x_2289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2289_, 0, v_a_2288_);
v___y_2263_ = v___y_2285_;
v___y_2264_ = v___y_2286_;
v___y_2265_ = v___y_2287_;
v_a_2266_ = v___x_2289_;
goto v___jp_2262_;
}
v___jp_2290_:
{
lean_object* v___x_2294_; lean_object* v___x_2295_; 
v___x_2294_ = lean_io_mono_nanos_now();
v___x_2295_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2166_, v___x_2167_, v_goal_2168_, v___f_2169_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_);
if (lean_obj_tag(v___x_2295_) == 0)
{
lean_object* v_a_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v_stats_2299_; lean_object* v_rulePatternCache_2300_; lean_object* v___x_2302_; uint8_t v_isShared_2303_; uint8_t v_isSharedCheck_2327_; 
v_a_2296_ = lean_ctor_get(v___x_2295_, 0);
lean_inc(v_a_2296_);
lean_dec_ref_known(v___x_2295_, 1);
v___x_2297_ = lean_io_mono_nanos_now();
v___x_2298_ = lean_st_ref_take(v___y_2173_);
v_stats_2299_ = lean_ctor_get(v___x_2298_, 1);
v_rulePatternCache_2300_ = lean_ctor_get(v___x_2298_, 0);
v_isSharedCheck_2327_ = !lean_is_exclusive(v___x_2298_);
if (v_isSharedCheck_2327_ == 0)
{
v___x_2302_ = v___x_2298_;
v_isShared_2303_ = v_isSharedCheck_2327_;
goto v_resetjp_2301_;
}
else
{
lean_inc(v_stats_2299_);
lean_inc(v_rulePatternCache_2300_);
lean_dec(v___x_2298_);
v___x_2302_ = lean_box(0);
v_isShared_2303_ = v_isSharedCheck_2327_;
goto v_resetjp_2301_;
}
v_resetjp_2301_:
{
lean_object* v_total_2304_; lean_object* v_configParsing_2305_; lean_object* v_ruleSetConstruction_2306_; lean_object* v_search_2307_; lean_object* v_ruleSelection_2308_; lean_object* v_script_2309_; lean_object* v_forwardState_2310_; lean_object* v_scriptGenerated_2311_; lean_object* v_ruleStats_2312_; lean_object* v_goalStats_2313_; lean_object* v___x_2315_; uint8_t v_isShared_2316_; uint8_t v_isSharedCheck_2326_; 
v_total_2304_ = lean_ctor_get(v_stats_2299_, 0);
v_configParsing_2305_ = lean_ctor_get(v_stats_2299_, 1);
v_ruleSetConstruction_2306_ = lean_ctor_get(v_stats_2299_, 2);
v_search_2307_ = lean_ctor_get(v_stats_2299_, 3);
v_ruleSelection_2308_ = lean_ctor_get(v_stats_2299_, 4);
v_script_2309_ = lean_ctor_get(v_stats_2299_, 5);
v_forwardState_2310_ = lean_ctor_get(v_stats_2299_, 6);
v_scriptGenerated_2311_ = lean_ctor_get(v_stats_2299_, 7);
v_ruleStats_2312_ = lean_ctor_get(v_stats_2299_, 8);
v_goalStats_2313_ = lean_ctor_get(v_stats_2299_, 9);
v_isSharedCheck_2326_ = !lean_is_exclusive(v_stats_2299_);
if (v_isSharedCheck_2326_ == 0)
{
v___x_2315_ = v_stats_2299_;
v_isShared_2316_ = v_isSharedCheck_2326_;
goto v_resetjp_2314_;
}
else
{
lean_inc(v_goalStats_2313_);
lean_inc(v_ruleStats_2312_);
lean_inc(v_scriptGenerated_2311_);
lean_inc(v_forwardState_2310_);
lean_inc(v_script_2309_);
lean_inc(v_ruleSelection_2308_);
lean_inc(v_search_2307_);
lean_inc(v_ruleSetConstruction_2306_);
lean_inc(v_configParsing_2305_);
lean_inc(v_total_2304_);
lean_dec(v_stats_2299_);
v___x_2315_ = lean_box(0);
v_isShared_2316_ = v_isSharedCheck_2326_;
goto v_resetjp_2314_;
}
v_resetjp_2314_:
{
lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2320_; 
v___x_2317_ = lean_nat_sub(v___x_2297_, v___x_2294_);
lean_dec(v___x_2294_);
lean_dec(v___x_2297_);
v___x_2318_ = lean_nat_add(v_ruleSelection_2308_, v___x_2317_);
lean_dec(v___x_2317_);
lean_dec(v_ruleSelection_2308_);
if (v_isShared_2316_ == 0)
{
lean_ctor_set(v___x_2315_, 4, v___x_2318_);
v___x_2320_ = v___x_2315_;
goto v_reusejp_2319_;
}
else
{
lean_object* v_reuseFailAlloc_2325_; 
v_reuseFailAlloc_2325_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2325_, 0, v_total_2304_);
lean_ctor_set(v_reuseFailAlloc_2325_, 1, v_configParsing_2305_);
lean_ctor_set(v_reuseFailAlloc_2325_, 2, v_ruleSetConstruction_2306_);
lean_ctor_set(v_reuseFailAlloc_2325_, 3, v_search_2307_);
lean_ctor_set(v_reuseFailAlloc_2325_, 4, v___x_2318_);
lean_ctor_set(v_reuseFailAlloc_2325_, 5, v_script_2309_);
lean_ctor_set(v_reuseFailAlloc_2325_, 6, v_forwardState_2310_);
lean_ctor_set(v_reuseFailAlloc_2325_, 7, v_scriptGenerated_2311_);
lean_ctor_set(v_reuseFailAlloc_2325_, 8, v_ruleStats_2312_);
lean_ctor_set(v_reuseFailAlloc_2325_, 9, v_goalStats_2313_);
v___x_2320_ = v_reuseFailAlloc_2325_;
goto v_reusejp_2319_;
}
v_reusejp_2319_:
{
lean_object* v___x_2322_; 
if (v_isShared_2303_ == 0)
{
lean_ctor_set(v___x_2302_, 1, v___x_2320_);
v___x_2322_ = v___x_2302_;
goto v_reusejp_2321_;
}
else
{
lean_object* v_reuseFailAlloc_2324_; 
v_reuseFailAlloc_2324_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2324_, 0, v_rulePatternCache_2300_);
lean_ctor_set(v_reuseFailAlloc_2324_, 1, v___x_2320_);
v___x_2322_ = v_reuseFailAlloc_2324_;
goto v_reusejp_2321_;
}
v_reusejp_2321_:
{
lean_object* v___x_2323_; 
v___x_2323_ = lean_st_ref_set(v___y_2173_, v___x_2322_);
v___y_2285_ = v___y_2291_;
v___y_2286_ = v___y_2292_;
v___y_2287_ = v___y_2293_;
v_a_2288_ = v_a_2296_;
goto v___jp_2284_;
}
}
}
}
}
else
{
lean_object* v_a_2328_; 
lean_dec(v___x_2294_);
v_a_2328_ = lean_ctor_get(v___x_2295_, 0);
lean_inc(v_a_2328_);
lean_dec_ref_known(v___x_2295_, 1);
v___y_2279_ = v___y_2291_;
v___y_2280_ = v___y_2292_;
v___y_2281_ = v___y_2293_;
v_a_2282_ = v_a_2328_;
goto v___jp_2278_;
}
}
v___jp_2329_:
{
lean_object* v___x_2333_; 
v___x_2333_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2166_, v___x_2167_, v_goal_2168_, v___f_2169_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_);
if (lean_obj_tag(v___x_2333_) == 0)
{
lean_object* v_a_2334_; 
v_a_2334_ = lean_ctor_get(v___x_2333_, 0);
lean_inc(v_a_2334_);
lean_dec_ref_known(v___x_2333_, 1);
v___y_2285_ = v___y_2330_;
v___y_2286_ = v___y_2331_;
v___y_2287_ = v___y_2332_;
v_a_2288_ = v_a_2334_;
goto v___jp_2284_;
}
else
{
lean_object* v_a_2335_; 
v_a_2335_ = lean_ctor_get(v___x_2333_, 0);
lean_inc(v_a_2335_);
lean_dec_ref_known(v___x_2333_, 1);
v___y_2279_ = v___y_2330_;
v___y_2280_ = v___y_2331_;
v___y_2281_ = v___y_2332_;
v_a_2282_ = v_a_2335_;
goto v___jp_2278_;
}
}
v___jp_2336_:
{
lean_object* v_a_2341_; uint8_t v___x_2342_; 
v_a_2341_ = lean_ctor_get(v___y_2340_, 0);
lean_inc(v_a_2341_);
lean_dec_ref(v___y_2340_);
v___x_2342_ = lean_unbox(v_a_2341_);
lean_dec(v_a_2341_);
if (v___x_2342_ == 0)
{
v___y_2330_ = v___y_2337_;
v___y_2331_ = v___y_2338_;
v___y_2332_ = v___y_2339_;
goto v___jp_2329_;
}
else
{
v___y_2291_ = v___y_2337_;
v___y_2292_ = v___y_2338_;
v___y_2293_ = v___y_2339_;
goto v___jp_2290_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__3___boxed(lean_object** _args){
lean_object* v___f_2422_ = _args[0];
lean_object* v___f_2423_ = _args[1];
lean_object* v___f_2424_ = _args[2];
lean_object* v___x_2425_ = _args[3];
lean_object* v___x_2426_ = _args[4];
lean_object* v___x_2427_ = _args[5];
lean_object* v___f_2428_ = _args[6];
lean_object* v_rs_2429_ = _args[7];
lean_object* v___x_2430_ = _args[8];
lean_object* v_goal_2431_ = _args[9];
lean_object* v___f_2432_ = _args[10];
lean_object* v_opts_2433_ = _args[11];
lean_object* v___y_2434_ = _args[12];
lean_object* v___y_2435_ = _args[13];
lean_object* v___y_2436_ = _args[14];
lean_object* v___y_2437_ = _args[15];
lean_object* v___y_2438_ = _args[16];
lean_object* v___y_2439_ = _args[17];
lean_object* v___y_2440_ = _args[18];
lean_object* v___y_2441_ = _args[19];
_start:
{
uint8_t v___x_121126__boxed_2442_; lean_object* v_res_2443_; 
v___x_121126__boxed_2442_ = lean_unbox(v___x_2426_);
v_res_2443_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__3(v___f_2422_, v___f_2423_, v___f_2424_, v___x_2425_, v___x_121126__boxed_2442_, v___x_2427_, v___f_2428_, v_rs_2429_, v___x_2430_, v_goal_2431_, v___f_2432_, v_opts_2433_, v___y_2434_, v___y_2435_, v___y_2436_, v___y_2437_, v___y_2438_, v___y_2439_, v___y_2440_);
lean_dec(v___y_2440_);
lean_dec_ref(v___y_2439_);
lean_dec(v___y_2438_);
lean_dec_ref(v___y_2437_);
lean_dec(v___y_2436_);
lean_dec(v___y_2435_);
lean_dec_ref(v___y_2434_);
lean_dec_ref(v_opts_2433_);
lean_dec_ref(v___x_2430_);
return v_res_2443_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__1(void){
_start:
{
lean_object* v___x_2445_; lean_object* v___x_2446_; 
v___x_2445_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__0));
v___x_2446_ = l_Lean_stringToMessageData(v___x_2445_);
return v___x_2446_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4(lean_object* v_x_2447_, lean_object* v___y_2448_, lean_object* v___y_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_){
_start:
{
lean_object* v___x_2456_; lean_object* v___x_2457_; 
v___x_2456_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___closed__1);
v___x_2457_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2457_, 0, v___x_2456_);
return v___x_2457_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4___boxed(lean_object* v_x_2458_, lean_object* v___y_2459_, lean_object* v___y_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_, lean_object* v___y_2465_, lean_object* v___y_2466_){
_start:
{
lean_object* v_res_2467_; 
v_res_2467_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__4(v_x_2458_, v___y_2459_, v___y_2460_, v___y_2461_, v___y_2462_, v___y_2463_, v___y_2464_, v___y_2465_);
lean_dec(v___y_2465_);
lean_dec_ref(v___y_2464_);
lean_dec(v___y_2463_);
lean_dec_ref(v___y_2462_);
lean_dec(v___y_2461_);
lean_dec(v___y_2460_);
lean_dec_ref(v___y_2459_);
lean_dec_ref(v_x_2458_);
return v_res_2467_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1_spec__2(lean_object* v_e_2468_){
_start:
{
if (lean_obj_tag(v_e_2468_) == 0)
{
uint8_t v___x_2469_; 
v___x_2469_ = 2;
return v___x_2469_;
}
else
{
lean_object* v_a_2470_; 
v_a_2470_ = lean_ctor_get(v_e_2468_, 0);
if (lean_obj_tag(v_a_2470_) == 0)
{
uint8_t v___x_2471_; 
v___x_2471_ = 1;
return v___x_2471_;
}
else
{
uint8_t v___x_2472_; 
v___x_2472_ = 0;
return v___x_2472_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1_spec__2___boxed(lean_object* v_e_2473_){
_start:
{
uint8_t v_res_2474_; lean_object* v_r_2475_; 
v_res_2474_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1_spec__2(v_e_2473_);
lean_dec_ref(v_e_2473_);
v_r_2475_ = lean_box(v_res_2474_);
return v_r_2475_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1(lean_object* v_cls_2476_, uint8_t v_collapsed_2477_, lean_object* v_tag_2478_, lean_object* v_opts_2479_, uint8_t v_clsEnabled_2480_, lean_object* v_oldTraces_2481_, lean_object* v_msg_2482_, lean_object* v_resStartStop_2483_, lean_object* v___y_2484_, lean_object* v___y_2485_, lean_object* v___y_2486_, lean_object* v___y_2487_, lean_object* v___y_2488_, lean_object* v___y_2489_, lean_object* v___y_2490_){
_start:
{
lean_object* v_fst_2492_; lean_object* v_snd_2493_; lean_object* v___y_2495_; lean_object* v___y_2496_; lean_object* v_data_2497_; lean_object* v_fst_2508_; lean_object* v_snd_2509_; lean_object* v___x_2510_; uint8_t v___x_2511_; lean_object* v___y_2513_; lean_object* v_a_2514_; uint8_t v___y_2529_; double v___y_2560_; 
v_fst_2492_ = lean_ctor_get(v_resStartStop_2483_, 0);
lean_inc(v_fst_2492_);
v_snd_2493_ = lean_ctor_get(v_resStartStop_2483_, 1);
lean_inc(v_snd_2493_);
lean_dec_ref(v_resStartStop_2483_);
v_fst_2508_ = lean_ctor_get(v_snd_2493_, 0);
lean_inc(v_fst_2508_);
v_snd_2509_ = lean_ctor_get(v_snd_2493_, 1);
lean_inc(v_snd_2509_);
lean_dec(v_snd_2493_);
v___x_2510_ = l_Lean_trace_profiler;
v___x_2511_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_2479_, v___x_2510_);
if (v___x_2511_ == 0)
{
v___y_2529_ = v___x_2511_;
goto v___jp_2528_;
}
else
{
lean_object* v___x_2565_; uint8_t v___x_2566_; 
v___x_2565_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2566_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_2479_, v___x_2565_);
if (v___x_2566_ == 0)
{
lean_object* v___x_2567_; lean_object* v___x_2568_; double v___x_2569_; double v___x_2570_; double v___x_2571_; 
v___x_2567_ = l_Lean_trace_profiler_threshold;
v___x_2568_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_2479_, v___x_2567_);
v___x_2569_ = lean_float_of_nat(v___x_2568_);
v___x_2570_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2);
v___x_2571_ = lean_float_div(v___x_2569_, v___x_2570_);
v___y_2560_ = v___x_2571_;
goto v___jp_2559_;
}
else
{
lean_object* v___x_2572_; lean_object* v___x_2573_; double v___x_2574_; 
v___x_2572_ = l_Lean_trace_profiler_threshold;
v___x_2573_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_2479_, v___x_2572_);
v___x_2574_ = lean_float_of_nat(v___x_2573_);
v___y_2560_ = v___x_2574_;
goto v___jp_2559_;
}
}
v___jp_2494_:
{
lean_object* v___x_2498_; 
lean_inc(v___y_2495_);
v___x_2498_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg(v_oldTraces_2481_, v_data_2497_, v___y_2495_, v___y_2496_, v___y_2487_, v___y_2488_, v___y_2489_, v___y_2490_);
if (lean_obj_tag(v___x_2498_) == 0)
{
lean_object* v___x_2499_; 
lean_dec_ref_known(v___x_2498_, 1);
v___x_2499_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_fst_2492_);
return v___x_2499_;
}
else
{
lean_object* v_a_2500_; lean_object* v___x_2502_; uint8_t v_isShared_2503_; uint8_t v_isSharedCheck_2507_; 
lean_dec(v_fst_2492_);
v_a_2500_ = lean_ctor_get(v___x_2498_, 0);
v_isSharedCheck_2507_ = !lean_is_exclusive(v___x_2498_);
if (v_isSharedCheck_2507_ == 0)
{
v___x_2502_ = v___x_2498_;
v_isShared_2503_ = v_isSharedCheck_2507_;
goto v_resetjp_2501_;
}
else
{
lean_inc(v_a_2500_);
lean_dec(v___x_2498_);
v___x_2502_ = lean_box(0);
v_isShared_2503_ = v_isSharedCheck_2507_;
goto v_resetjp_2501_;
}
v_resetjp_2501_:
{
lean_object* v___x_2505_; 
if (v_isShared_2503_ == 0)
{
v___x_2505_ = v___x_2502_;
goto v_reusejp_2504_;
}
else
{
lean_object* v_reuseFailAlloc_2506_; 
v_reuseFailAlloc_2506_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2506_, 0, v_a_2500_);
v___x_2505_ = v_reuseFailAlloc_2506_;
goto v_reusejp_2504_;
}
v_reusejp_2504_:
{
return v___x_2505_;
}
}
}
}
v___jp_2512_:
{
uint8_t v_result_2515_; lean_object* v___x_2516_; lean_object* v___x_2517_; double v___x_2518_; lean_object* v_data_2519_; 
v_result_2515_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1_spec__2(v_fst_2492_);
v___x_2516_ = lean_box(v_result_2515_);
v___x_2517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2517_, 0, v___x_2516_);
v___x_2518_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0);
lean_inc_ref(v_tag_2478_);
lean_inc_ref(v___x_2517_);
lean_inc(v_cls_2476_);
v_data_2519_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2519_, 0, v_cls_2476_);
lean_ctor_set(v_data_2519_, 1, v___x_2517_);
lean_ctor_set(v_data_2519_, 2, v_tag_2478_);
lean_ctor_set_float(v_data_2519_, sizeof(void*)*3, v___x_2518_);
lean_ctor_set_float(v_data_2519_, sizeof(void*)*3 + 8, v___x_2518_);
lean_ctor_set_uint8(v_data_2519_, sizeof(void*)*3 + 16, v_collapsed_2477_);
if (v___x_2511_ == 0)
{
lean_dec_ref_known(v___x_2517_, 1);
lean_dec(v_snd_2509_);
lean_dec(v_fst_2508_);
lean_dec_ref(v_tag_2478_);
lean_dec(v_cls_2476_);
v___y_2495_ = v___y_2513_;
v___y_2496_ = v_a_2514_;
v_data_2497_ = v_data_2519_;
goto v___jp_2494_;
}
else
{
lean_object* v_data_2520_; double v___x_2521_; double v___x_2522_; 
lean_dec_ref_known(v_data_2519_, 3);
v_data_2520_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2520_, 0, v_cls_2476_);
lean_ctor_set(v_data_2520_, 1, v___x_2517_);
lean_ctor_set(v_data_2520_, 2, v_tag_2478_);
v___x_2521_ = lean_unbox_float(v_fst_2508_);
lean_dec(v_fst_2508_);
lean_ctor_set_float(v_data_2520_, sizeof(void*)*3, v___x_2521_);
v___x_2522_ = lean_unbox_float(v_snd_2509_);
lean_dec(v_snd_2509_);
lean_ctor_set_float(v_data_2520_, sizeof(void*)*3 + 8, v___x_2522_);
lean_ctor_set_uint8(v_data_2520_, sizeof(void*)*3 + 16, v_collapsed_2477_);
v___y_2495_ = v___y_2513_;
v___y_2496_ = v_a_2514_;
v_data_2497_ = v_data_2520_;
goto v___jp_2494_;
}
}
v___jp_2523_:
{
lean_object* v_ref_2524_; lean_object* v___x_2525_; 
v_ref_2524_ = lean_ctor_get(v___y_2489_, 5);
lean_inc(v___y_2490_);
lean_inc_ref(v___y_2489_);
lean_inc(v___y_2488_);
lean_inc_ref(v___y_2487_);
lean_inc(v___y_2486_);
lean_inc(v___y_2485_);
lean_inc_ref(v___y_2484_);
lean_inc(v_fst_2492_);
v___x_2525_ = lean_apply_9(v_msg_2482_, v_fst_2492_, v___y_2484_, v___y_2485_, v___y_2486_, v___y_2487_, v___y_2488_, v___y_2489_, v___y_2490_, lean_box(0));
if (lean_obj_tag(v___x_2525_) == 0)
{
lean_object* v_a_2526_; 
v_a_2526_ = lean_ctor_get(v___x_2525_, 0);
lean_inc(v_a_2526_);
lean_dec_ref_known(v___x_2525_, 1);
v___y_2513_ = v_ref_2524_;
v_a_2514_ = v_a_2526_;
goto v___jp_2512_;
}
else
{
lean_object* v___x_2527_; 
lean_dec_ref_known(v___x_2525_, 1);
v___x_2527_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1);
v___y_2513_ = v_ref_2524_;
v_a_2514_ = v___x_2527_;
goto v___jp_2512_;
}
}
v___jp_2528_:
{
if (v_clsEnabled_2480_ == 0)
{
if (v___y_2529_ == 0)
{
lean_object* v___x_2530_; lean_object* v_traceState_2531_; lean_object* v_env_2532_; lean_object* v_nextMacroScope_2533_; lean_object* v_ngen_2534_; lean_object* v_auxDeclNGen_2535_; lean_object* v_cache_2536_; lean_object* v_messages_2537_; lean_object* v_infoState_2538_; lean_object* v_snapshotTasks_2539_; lean_object* v___x_2541_; uint8_t v_isShared_2542_; uint8_t v_isSharedCheck_2558_; 
lean_dec(v_snd_2509_);
lean_dec(v_fst_2508_);
lean_dec_ref(v_msg_2482_);
lean_dec_ref(v_tag_2478_);
lean_dec(v_cls_2476_);
v___x_2530_ = lean_st_ref_take(v___y_2490_);
v_traceState_2531_ = lean_ctor_get(v___x_2530_, 4);
v_env_2532_ = lean_ctor_get(v___x_2530_, 0);
v_nextMacroScope_2533_ = lean_ctor_get(v___x_2530_, 1);
v_ngen_2534_ = lean_ctor_get(v___x_2530_, 2);
v_auxDeclNGen_2535_ = lean_ctor_get(v___x_2530_, 3);
v_cache_2536_ = lean_ctor_get(v___x_2530_, 5);
v_messages_2537_ = lean_ctor_get(v___x_2530_, 6);
v_infoState_2538_ = lean_ctor_get(v___x_2530_, 7);
v_snapshotTasks_2539_ = lean_ctor_get(v___x_2530_, 8);
v_isSharedCheck_2558_ = !lean_is_exclusive(v___x_2530_);
if (v_isSharedCheck_2558_ == 0)
{
v___x_2541_ = v___x_2530_;
v_isShared_2542_ = v_isSharedCheck_2558_;
goto v_resetjp_2540_;
}
else
{
lean_inc(v_snapshotTasks_2539_);
lean_inc(v_infoState_2538_);
lean_inc(v_messages_2537_);
lean_inc(v_cache_2536_);
lean_inc(v_traceState_2531_);
lean_inc(v_auxDeclNGen_2535_);
lean_inc(v_ngen_2534_);
lean_inc(v_nextMacroScope_2533_);
lean_inc(v_env_2532_);
lean_dec(v___x_2530_);
v___x_2541_ = lean_box(0);
v_isShared_2542_ = v_isSharedCheck_2558_;
goto v_resetjp_2540_;
}
v_resetjp_2540_:
{
uint64_t v_tid_2543_; lean_object* v_traces_2544_; lean_object* v___x_2546_; uint8_t v_isShared_2547_; uint8_t v_isSharedCheck_2557_; 
v_tid_2543_ = lean_ctor_get_uint64(v_traceState_2531_, sizeof(void*)*1);
v_traces_2544_ = lean_ctor_get(v_traceState_2531_, 0);
v_isSharedCheck_2557_ = !lean_is_exclusive(v_traceState_2531_);
if (v_isSharedCheck_2557_ == 0)
{
v___x_2546_ = v_traceState_2531_;
v_isShared_2547_ = v_isSharedCheck_2557_;
goto v_resetjp_2545_;
}
else
{
lean_inc(v_traces_2544_);
lean_dec(v_traceState_2531_);
v___x_2546_ = lean_box(0);
v_isShared_2547_ = v_isSharedCheck_2557_;
goto v_resetjp_2545_;
}
v_resetjp_2545_:
{
lean_object* v___x_2548_; lean_object* v___x_2550_; 
v___x_2548_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2481_, v_traces_2544_);
lean_dec_ref(v_traces_2544_);
if (v_isShared_2547_ == 0)
{
lean_ctor_set(v___x_2546_, 0, v___x_2548_);
v___x_2550_ = v___x_2546_;
goto v_reusejp_2549_;
}
else
{
lean_object* v_reuseFailAlloc_2556_; 
v_reuseFailAlloc_2556_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2556_, 0, v___x_2548_);
lean_ctor_set_uint64(v_reuseFailAlloc_2556_, sizeof(void*)*1, v_tid_2543_);
v___x_2550_ = v_reuseFailAlloc_2556_;
goto v_reusejp_2549_;
}
v_reusejp_2549_:
{
lean_object* v___x_2552_; 
if (v_isShared_2542_ == 0)
{
lean_ctor_set(v___x_2541_, 4, v___x_2550_);
v___x_2552_ = v___x_2541_;
goto v_reusejp_2551_;
}
else
{
lean_object* v_reuseFailAlloc_2555_; 
v_reuseFailAlloc_2555_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2555_, 0, v_env_2532_);
lean_ctor_set(v_reuseFailAlloc_2555_, 1, v_nextMacroScope_2533_);
lean_ctor_set(v_reuseFailAlloc_2555_, 2, v_ngen_2534_);
lean_ctor_set(v_reuseFailAlloc_2555_, 3, v_auxDeclNGen_2535_);
lean_ctor_set(v_reuseFailAlloc_2555_, 4, v___x_2550_);
lean_ctor_set(v_reuseFailAlloc_2555_, 5, v_cache_2536_);
lean_ctor_set(v_reuseFailAlloc_2555_, 6, v_messages_2537_);
lean_ctor_set(v_reuseFailAlloc_2555_, 7, v_infoState_2538_);
lean_ctor_set(v_reuseFailAlloc_2555_, 8, v_snapshotTasks_2539_);
v___x_2552_ = v_reuseFailAlloc_2555_;
goto v_reusejp_2551_;
}
v_reusejp_2551_:
{
lean_object* v___x_2553_; lean_object* v___x_2554_; 
v___x_2553_ = lean_st_ref_set(v___y_2490_, v___x_2552_);
v___x_2554_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_fst_2492_);
return v___x_2554_;
}
}
}
}
}
else
{
goto v___jp_2523_;
}
}
else
{
goto v___jp_2523_;
}
}
v___jp_2559_:
{
double v___x_2561_; double v___x_2562_; double v___x_2563_; uint8_t v___x_2564_; 
v___x_2561_ = lean_unbox_float(v_snd_2509_);
v___x_2562_ = lean_unbox_float(v_fst_2508_);
v___x_2563_ = lean_float_sub(v___x_2561_, v___x_2562_);
v___x_2564_ = lean_float_decLt(v___y_2560_, v___x_2563_);
v___y_2529_ = v___x_2564_;
goto v___jp_2528_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1___boxed(lean_object* v_cls_2575_, lean_object* v_collapsed_2576_, lean_object* v_tag_2577_, lean_object* v_opts_2578_, lean_object* v_clsEnabled_2579_, lean_object* v_oldTraces_2580_, lean_object* v_msg_2581_, lean_object* v_resStartStop_2582_, lean_object* v___y_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_, lean_object* v___y_2589_, lean_object* v___y_2590_){
_start:
{
uint8_t v_collapsed_boxed_2591_; uint8_t v_clsEnabled_boxed_2592_; lean_object* v_res_2593_; 
v_collapsed_boxed_2591_ = lean_unbox(v_collapsed_2576_);
v_clsEnabled_boxed_2592_ = lean_unbox(v_clsEnabled_2579_);
v_res_2593_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1(v_cls_2575_, v_collapsed_boxed_2591_, v_tag_2577_, v_opts_2578_, v_clsEnabled_boxed_2592_, v_oldTraces_2580_, v_msg_2581_, v_resStartStop_2582_, v___y_2583_, v___y_2584_, v___y_2585_, v___y_2586_, v___y_2587_, v___y_2588_, v___y_2589_);
lean_dec(v___y_2589_);
lean_dec_ref(v___y_2588_);
lean_dec(v___y_2587_);
lean_dec_ref(v___y_2586_);
lean_dec(v___y_2585_);
lean_dec(v___y_2584_);
lean_dec_ref(v___y_2583_);
lean_dec_ref(v_opts_2578_);
return v_res_2593_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules(lean_object* v_rs_2599_, lean_object* v_goal_2600_, lean_object* v_mvars_2601_, lean_object* v_preState_2602_, lean_object* v_a_2603_, lean_object* v_a_2604_, lean_object* v_a_2605_, lean_object* v_a_2606_, lean_object* v_a_2607_, lean_object* v_a_2608_, lean_object* v_a_2609_){
_start:
{
lean_object* v_options_2611_; lean_object* v_inheritedTraceOptions_2612_; uint8_t v_hasTrace_2613_; lean_object* v___f_2614_; lean_object* v___f_2615_; lean_object* v___f_2616_; lean_object* v___f_2617_; lean_object* v___x_2618_; lean_object* v___x_2619_; lean_object* v___f_2620_; uint8_t v___x_2621_; lean_object* v___x_2622_; 
v_options_2611_ = lean_ctor_get(v_a_2608_, 2);
v_inheritedTraceOptions_2612_ = lean_ctor_get(v_a_2608_, 13);
v_hasTrace_2613_ = lean_ctor_get_uint8(v_options_2611_, sizeof(void*)*1);
v___f_2614_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__0));
v___f_2615_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__1));
v___f_2616_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__2));
v___f_2617_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__3));
v___x_2618_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_2619_ = lp_aesop_Aesop_ForwardRuleMatches_empty;
lean_inc(v_goal_2600_);
lean_inc_ref(v_rs_2599_);
v___f_2620_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__1___boxed), 13, 4);
lean_closure_set(v___f_2620_, 0, v_rs_2599_);
lean_closure_set(v___f_2620_, 1, v___x_2619_);
lean_closure_set(v___f_2620_, 2, v_goal_2600_);
lean_closure_set(v___f_2620_, 3, v___f_2617_);
v___x_2621_ = 1;
v___x_2622_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
if (v_hasTrace_2613_ == 0)
{
lean_object* v___x_2623_; 
lean_inc(v_goal_2600_);
v___x_2623_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__3(v___f_2614_, v___f_2615_, v___f_2620_, v___x_2618_, v___x_2621_, v___x_2622_, v___f_2616_, v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_options_2611_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_2623_) == 0)
{
lean_object* v_a_2624_; lean_object* v___x_2625_; 
v_a_2624_ = lean_ctor_get(v___x_2623_, 0);
lean_inc(v_a_2624_);
lean_dec_ref_known(v___x_2623_, 1);
v___x_2625_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_2600_, v_mvars_2601_, v_preState_2602_, v_a_2624_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
lean_dec(v_a_2624_);
return v___x_2625_;
}
else
{
lean_object* v_a_2626_; lean_object* v___x_2628_; uint8_t v_isShared_2629_; uint8_t v_isSharedCheck_2633_; 
lean_dec_ref(v_mvars_2601_);
lean_dec(v_goal_2600_);
v_a_2626_ = lean_ctor_get(v___x_2623_, 0);
v_isSharedCheck_2633_ = !lean_is_exclusive(v___x_2623_);
if (v_isSharedCheck_2633_ == 0)
{
v___x_2628_ = v___x_2623_;
v_isShared_2629_ = v_isSharedCheck_2633_;
goto v_resetjp_2627_;
}
else
{
lean_inc(v_a_2626_);
lean_dec(v___x_2623_);
v___x_2628_ = lean_box(0);
v_isShared_2629_ = v_isSharedCheck_2633_;
goto v_resetjp_2627_;
}
v_resetjp_2627_:
{
lean_object* v___x_2631_; 
if (v_isShared_2629_ == 0)
{
v___x_2631_ = v___x_2628_;
goto v_reusejp_2630_;
}
else
{
lean_object* v_reuseFailAlloc_2632_; 
v_reuseFailAlloc_2632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2632_, 0, v_a_2626_);
v___x_2631_ = v_reuseFailAlloc_2632_;
goto v_reusejp_2630_;
}
v_reusejp_2630_:
{
return v___x_2631_;
}
}
}
}
else
{
lean_object* v___f_2634_; lean_object* v___x_2635_; uint8_t v___x_2636_; lean_object* v___y_2638_; lean_object* v___y_2639_; lean_object* v_a_2640_; lean_object* v___y_2653_; lean_object* v___y_2654_; lean_object* v_a_2655_; lean_object* v___y_2658_; lean_object* v___y_2659_; lean_object* v___y_2660_; lean_object* v___y_2674_; lean_object* v___y_2675_; lean_object* v___y_2676_; lean_object* v___y_2677_; uint8_t v___y_2678_; lean_object* v_a_2679_; lean_object* v___y_2692_; lean_object* v___y_2693_; lean_object* v___y_2694_; lean_object* v___y_2695_; uint8_t v___y_2696_; lean_object* v_a_2697_; lean_object* v___y_2700_; lean_object* v___y_2701_; lean_object* v___y_2702_; lean_object* v___y_2703_; uint8_t v___y_2704_; lean_object* v_a_2705_; lean_object* v___y_2708_; lean_object* v___y_2709_; lean_object* v___y_2710_; lean_object* v___y_2711_; uint8_t v___y_2712_; lean_object* v___y_2717_; lean_object* v___y_2718_; lean_object* v___y_2719_; lean_object* v___y_2720_; uint8_t v___y_2721_; lean_object* v___y_2758_; lean_object* v___y_2759_; lean_object* v___y_2760_; lean_object* v___y_2761_; uint8_t v___y_2762_; lean_object* v___y_2763_; lean_object* v___y_2767_; lean_object* v___y_2768_; lean_object* v___y_2769_; lean_object* v___y_2770_; uint8_t v___y_2771_; lean_object* v_a_2772_; lean_object* v___y_2782_; lean_object* v___y_2783_; lean_object* v___y_2784_; lean_object* v___y_2785_; uint8_t v___y_2786_; lean_object* v_a_2787_; lean_object* v___y_2790_; lean_object* v___y_2791_; lean_object* v___y_2792_; lean_object* v___y_2793_; uint8_t v___y_2794_; lean_object* v_a_2795_; lean_object* v___y_2798_; lean_object* v___y_2799_; lean_object* v___y_2800_; lean_object* v___y_2801_; uint8_t v___y_2802_; lean_object* v___y_2839_; lean_object* v___y_2840_; lean_object* v___y_2841_; lean_object* v___y_2842_; uint8_t v___y_2843_; lean_object* v___y_2848_; lean_object* v___y_2849_; lean_object* v___y_2850_; lean_object* v___y_2851_; uint8_t v___y_2852_; lean_object* v___y_2853_; lean_object* v___y_2857_; lean_object* v___y_2858_; uint8_t v___y_2859_; uint8_t v___y_2860_; lean_object* v___y_2885_; lean_object* v___y_2886_; lean_object* v_a_2887_; lean_object* v___y_2897_; lean_object* v___y_2898_; lean_object* v_a_2899_; lean_object* v___y_2902_; lean_object* v___y_2903_; lean_object* v___y_2904_; lean_object* v___y_2918_; lean_object* v___y_2919_; uint8_t v___y_2920_; lean_object* v___y_2921_; lean_object* v___y_2922_; lean_object* v_a_2923_; lean_object* v___y_2936_; lean_object* v___y_2937_; uint8_t v___y_2938_; lean_object* v___y_2939_; lean_object* v___y_2940_; lean_object* v_a_2941_; lean_object* v___y_2944_; lean_object* v___y_2945_; uint8_t v___y_2946_; lean_object* v___y_2947_; lean_object* v___y_2948_; lean_object* v_a_2949_; lean_object* v___y_2952_; lean_object* v___y_2953_; uint8_t v___y_2954_; lean_object* v___y_2955_; lean_object* v___y_2956_; lean_object* v___y_2993_; lean_object* v___y_2994_; uint8_t v___y_2995_; lean_object* v___y_2996_; lean_object* v___y_2997_; lean_object* v___y_3002_; lean_object* v___y_3003_; uint8_t v___y_3004_; lean_object* v___y_3005_; lean_object* v___y_3006_; lean_object* v___y_3007_; lean_object* v___y_3011_; uint8_t v___y_3012_; lean_object* v___y_3013_; lean_object* v___y_3014_; lean_object* v___y_3015_; lean_object* v_a_3016_; lean_object* v___y_3026_; uint8_t v___y_3027_; lean_object* v___y_3028_; lean_object* v___y_3029_; lean_object* v___y_3030_; lean_object* v_a_3031_; lean_object* v___y_3034_; uint8_t v___y_3035_; lean_object* v___y_3036_; lean_object* v___y_3037_; lean_object* v___y_3038_; lean_object* v_a_3039_; lean_object* v___y_3042_; uint8_t v___y_3043_; lean_object* v___y_3044_; lean_object* v___y_3045_; lean_object* v___y_3046_; lean_object* v___y_3083_; uint8_t v___y_3084_; lean_object* v___y_3085_; lean_object* v___y_3086_; lean_object* v___y_3087_; lean_object* v___y_3092_; uint8_t v___y_3093_; lean_object* v___y_3094_; lean_object* v___y_3095_; lean_object* v___y_3096_; lean_object* v___y_3097_; lean_object* v___y_3101_; uint8_t v___y_3102_; lean_object* v___y_3103_; uint8_t v___y_3104_; 
v___f_2634_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__4));
v___x_2635_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0);
v___x_2636_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2612_, v_options_2611_, v___x_2635_);
if (v___x_2636_ == 0)
{
lean_object* v___x_3153_; uint8_t v___x_3154_; 
v___x_3153_ = l_Lean_trace_profiler;
v___x_3154_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2611_, v___x_3153_);
if (v___x_3154_ == 0)
{
lean_object* v___x_3155_; 
lean_inc(v_goal_2600_);
v___x_3155_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__3(v___f_2614_, v___f_2615_, v___f_2620_, v___x_2618_, v___x_2621_, v___x_2622_, v___f_2616_, v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_options_2611_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_3155_) == 0)
{
lean_object* v_a_3156_; lean_object* v___x_3157_; 
v_a_3156_ = lean_ctor_get(v___x_3155_, 0);
lean_inc(v_a_3156_);
lean_dec_ref_known(v___x_3155_, 1);
v___x_3157_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_2600_, v_mvars_2601_, v_preState_2602_, v_a_3156_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
lean_dec(v_a_3156_);
return v___x_3157_;
}
else
{
lean_object* v_a_3158_; lean_object* v___x_3160_; uint8_t v_isShared_3161_; uint8_t v_isSharedCheck_3165_; 
lean_dec_ref(v_mvars_2601_);
lean_dec(v_goal_2600_);
v_a_3158_ = lean_ctor_get(v___x_3155_, 0);
v_isSharedCheck_3165_ = !lean_is_exclusive(v___x_3155_);
if (v_isSharedCheck_3165_ == 0)
{
v___x_3160_ = v___x_3155_;
v_isShared_3161_ = v_isSharedCheck_3165_;
goto v_resetjp_3159_;
}
else
{
lean_inc(v_a_3158_);
lean_dec(v___x_3155_);
v___x_3160_ = lean_box(0);
v_isShared_3161_ = v_isSharedCheck_3165_;
goto v_resetjp_3159_;
}
v_resetjp_3159_:
{
lean_object* v___x_3163_; 
if (v_isShared_3161_ == 0)
{
v___x_3163_ = v___x_3160_;
goto v_reusejp_3162_;
}
else
{
lean_object* v_reuseFailAlloc_3164_; 
v_reuseFailAlloc_3164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3164_, 0, v_a_3158_);
v___x_3163_ = v_reuseFailAlloc_3164_;
goto v_reusejp_3162_;
}
v_reusejp_3162_:
{
return v___x_3163_;
}
}
}
}
else
{
lean_dec_ref(v___f_2620_);
goto v___jp_3128_;
}
}
else
{
lean_dec_ref(v___f_2620_);
goto v___jp_3128_;
}
v___jp_2637_:
{
lean_object* v___x_2641_; double v___x_2642_; double v___x_2643_; double v___x_2644_; double v___x_2645_; double v___x_2646_; lean_object* v___x_2647_; lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; 
v___x_2641_ = lean_io_mono_nanos_now();
v___x_2642_ = lean_float_of_nat(v___y_2639_);
v___x_2643_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_2644_ = lean_float_div(v___x_2642_, v___x_2643_);
v___x_2645_ = lean_float_of_nat(v___x_2641_);
v___x_2646_ = lean_float_div(v___x_2645_, v___x_2643_);
v___x_2647_ = lean_box_float(v___x_2644_);
v___x_2648_ = lean_box_float(v___x_2646_);
v___x_2649_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2649_, 0, v___x_2647_);
lean_ctor_set(v___x_2649_, 1, v___x_2648_);
v___x_2650_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2650_, 0, v_a_2640_);
lean_ctor_set(v___x_2650_, 1, v___x_2649_);
v___x_2651_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1(v___x_2618_, v___x_2621_, v___x_2622_, v_options_2611_, v___x_2636_, v___y_2638_, v___f_2634_, v___x_2650_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
return v___x_2651_;
}
v___jp_2652_:
{
lean_object* v___x_2656_; 
v___x_2656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2656_, 0, v_a_2655_);
v___y_2638_ = v___y_2653_;
v___y_2639_ = v___y_2654_;
v_a_2640_ = v___x_2656_;
goto v___jp_2637_;
}
v___jp_2657_:
{
if (lean_obj_tag(v___y_2660_) == 0)
{
lean_object* v_a_2661_; lean_object* v___x_2662_; 
v_a_2661_ = lean_ctor_get(v___y_2660_, 0);
lean_inc(v_a_2661_);
lean_dec_ref_known(v___y_2660_, 1);
v___x_2662_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_2600_, v_mvars_2601_, v_preState_2602_, v_a_2661_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
lean_dec(v_a_2661_);
if (lean_obj_tag(v___x_2662_) == 0)
{
lean_object* v_a_2663_; lean_object* v___x_2665_; uint8_t v_isShared_2666_; uint8_t v_isSharedCheck_2670_; 
v_a_2663_ = lean_ctor_get(v___x_2662_, 0);
v_isSharedCheck_2670_ = !lean_is_exclusive(v___x_2662_);
if (v_isSharedCheck_2670_ == 0)
{
v___x_2665_ = v___x_2662_;
v_isShared_2666_ = v_isSharedCheck_2670_;
goto v_resetjp_2664_;
}
else
{
lean_inc(v_a_2663_);
lean_dec(v___x_2662_);
v___x_2665_ = lean_box(0);
v_isShared_2666_ = v_isSharedCheck_2670_;
goto v_resetjp_2664_;
}
v_resetjp_2664_:
{
lean_object* v___x_2668_; 
if (v_isShared_2666_ == 0)
{
lean_ctor_set_tag(v___x_2665_, 1);
v___x_2668_ = v___x_2665_;
goto v_reusejp_2667_;
}
else
{
lean_object* v_reuseFailAlloc_2669_; 
v_reuseFailAlloc_2669_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2669_, 0, v_a_2663_);
v___x_2668_ = v_reuseFailAlloc_2669_;
goto v_reusejp_2667_;
}
v_reusejp_2667_:
{
v___y_2638_ = v___y_2658_;
v___y_2639_ = v___y_2659_;
v_a_2640_ = v___x_2668_;
goto v___jp_2637_;
}
}
}
else
{
lean_object* v_a_2671_; 
v_a_2671_ = lean_ctor_get(v___x_2662_, 0);
lean_inc(v_a_2671_);
lean_dec_ref_known(v___x_2662_, 1);
v___y_2653_ = v___y_2658_;
v___y_2654_ = v___y_2659_;
v_a_2655_ = v_a_2671_;
goto v___jp_2652_;
}
}
else
{
lean_object* v_a_2672_; 
lean_dec_ref(v_mvars_2601_);
lean_dec(v_goal_2600_);
v_a_2672_ = lean_ctor_get(v___y_2660_, 0);
lean_inc(v_a_2672_);
lean_dec_ref_known(v___y_2660_, 1);
v___y_2653_ = v___y_2658_;
v___y_2654_ = v___y_2659_;
v_a_2655_ = v_a_2672_;
goto v___jp_2652_;
}
}
v___jp_2673_:
{
lean_object* v___x_2680_; double v___x_2681_; double v___x_2682_; double v___x_2683_; double v___x_2684_; double v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; 
v___x_2680_ = lean_io_mono_nanos_now();
v___x_2681_ = lean_float_of_nat(v___y_2674_);
v___x_2682_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_2683_ = lean_float_div(v___x_2681_, v___x_2682_);
v___x_2684_ = lean_float_of_nat(v___x_2680_);
v___x_2685_ = lean_float_div(v___x_2684_, v___x_2682_);
v___x_2686_ = lean_box_float(v___x_2683_);
v___x_2687_ = lean_box_float(v___x_2685_);
v___x_2688_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2688_, 0, v___x_2686_);
lean_ctor_set(v___x_2688_, 1, v___x_2687_);
v___x_2689_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2689_, 0, v_a_2679_);
lean_ctor_set(v___x_2689_, 1, v___x_2688_);
v___x_2690_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(v___x_2618_, v___x_2621_, v___x_2622_, v_options_2611_, v___y_2678_, v___y_2676_, v___f_2616_, v___x_2689_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v___y_2658_ = v___y_2675_;
v___y_2659_ = v___y_2677_;
v___y_2660_ = v___x_2690_;
goto v___jp_2657_;
}
v___jp_2691_:
{
lean_object* v___x_2698_; 
v___x_2698_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2698_, 0, v_a_2697_);
v___y_2674_ = v___y_2692_;
v___y_2675_ = v___y_2694_;
v___y_2676_ = v___y_2693_;
v___y_2677_ = v___y_2695_;
v___y_2678_ = v___y_2696_;
v_a_2679_ = v___x_2698_;
goto v___jp_2673_;
}
v___jp_2699_:
{
lean_object* v___x_2706_; 
v___x_2706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2706_, 0, v_a_2705_);
v___y_2674_ = v___y_2700_;
v___y_2675_ = v___y_2702_;
v___y_2676_ = v___y_2701_;
v___y_2677_ = v___y_2703_;
v___y_2678_ = v___y_2704_;
v_a_2679_ = v___x_2706_;
goto v___jp_2673_;
}
v___jp_2707_:
{
lean_object* v___x_2713_; 
lean_inc(v_goal_2600_);
v___x_2713_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_2713_) == 0)
{
lean_object* v_a_2714_; 
v_a_2714_ = lean_ctor_get(v___x_2713_, 0);
lean_inc(v_a_2714_);
lean_dec_ref_known(v___x_2713_, 1);
v___y_2700_ = v___y_2708_;
v___y_2701_ = v___y_2710_;
v___y_2702_ = v___y_2709_;
v___y_2703_ = v___y_2711_;
v___y_2704_ = v___y_2712_;
v_a_2705_ = v_a_2714_;
goto v___jp_2699_;
}
else
{
lean_object* v_a_2715_; 
v_a_2715_ = lean_ctor_get(v___x_2713_, 0);
lean_inc(v_a_2715_);
lean_dec_ref_known(v___x_2713_, 1);
v___y_2692_ = v___y_2708_;
v___y_2693_ = v___y_2710_;
v___y_2694_ = v___y_2709_;
v___y_2695_ = v___y_2711_;
v___y_2696_ = v___y_2712_;
v_a_2697_ = v_a_2715_;
goto v___jp_2691_;
}
}
v___jp_2716_:
{
lean_object* v___x_2722_; lean_object* v___x_2723_; 
v___x_2722_ = lean_io_mono_nanos_now();
lean_inc(v_goal_2600_);
v___x_2723_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_2723_) == 0)
{
lean_object* v_a_2724_; lean_object* v___x_2725_; lean_object* v___x_2726_; lean_object* v_stats_2727_; lean_object* v_rulePatternCache_2728_; lean_object* v___x_2730_; uint8_t v_isShared_2731_; uint8_t v_isSharedCheck_2755_; 
v_a_2724_ = lean_ctor_get(v___x_2723_, 0);
lean_inc(v_a_2724_);
lean_dec_ref_known(v___x_2723_, 1);
v___x_2725_ = lean_io_mono_nanos_now();
v___x_2726_ = lean_st_ref_take(v_a_2605_);
v_stats_2727_ = lean_ctor_get(v___x_2726_, 1);
v_rulePatternCache_2728_ = lean_ctor_get(v___x_2726_, 0);
v_isSharedCheck_2755_ = !lean_is_exclusive(v___x_2726_);
if (v_isSharedCheck_2755_ == 0)
{
v___x_2730_ = v___x_2726_;
v_isShared_2731_ = v_isSharedCheck_2755_;
goto v_resetjp_2729_;
}
else
{
lean_inc(v_stats_2727_);
lean_inc(v_rulePatternCache_2728_);
lean_dec(v___x_2726_);
v___x_2730_ = lean_box(0);
v_isShared_2731_ = v_isSharedCheck_2755_;
goto v_resetjp_2729_;
}
v_resetjp_2729_:
{
lean_object* v_total_2732_; lean_object* v_configParsing_2733_; lean_object* v_ruleSetConstruction_2734_; lean_object* v_search_2735_; lean_object* v_ruleSelection_2736_; lean_object* v_script_2737_; lean_object* v_forwardState_2738_; lean_object* v_scriptGenerated_2739_; lean_object* v_ruleStats_2740_; lean_object* v_goalStats_2741_; lean_object* v___x_2743_; uint8_t v_isShared_2744_; uint8_t v_isSharedCheck_2754_; 
v_total_2732_ = lean_ctor_get(v_stats_2727_, 0);
v_configParsing_2733_ = lean_ctor_get(v_stats_2727_, 1);
v_ruleSetConstruction_2734_ = lean_ctor_get(v_stats_2727_, 2);
v_search_2735_ = lean_ctor_get(v_stats_2727_, 3);
v_ruleSelection_2736_ = lean_ctor_get(v_stats_2727_, 4);
v_script_2737_ = lean_ctor_get(v_stats_2727_, 5);
v_forwardState_2738_ = lean_ctor_get(v_stats_2727_, 6);
v_scriptGenerated_2739_ = lean_ctor_get(v_stats_2727_, 7);
v_ruleStats_2740_ = lean_ctor_get(v_stats_2727_, 8);
v_goalStats_2741_ = lean_ctor_get(v_stats_2727_, 9);
v_isSharedCheck_2754_ = !lean_is_exclusive(v_stats_2727_);
if (v_isSharedCheck_2754_ == 0)
{
v___x_2743_ = v_stats_2727_;
v_isShared_2744_ = v_isSharedCheck_2754_;
goto v_resetjp_2742_;
}
else
{
lean_inc(v_goalStats_2741_);
lean_inc(v_ruleStats_2740_);
lean_inc(v_scriptGenerated_2739_);
lean_inc(v_forwardState_2738_);
lean_inc(v_script_2737_);
lean_inc(v_ruleSelection_2736_);
lean_inc(v_search_2735_);
lean_inc(v_ruleSetConstruction_2734_);
lean_inc(v_configParsing_2733_);
lean_inc(v_total_2732_);
lean_dec(v_stats_2727_);
v___x_2743_ = lean_box(0);
v_isShared_2744_ = v_isSharedCheck_2754_;
goto v_resetjp_2742_;
}
v_resetjp_2742_:
{
lean_object* v___x_2745_; lean_object* v___x_2746_; lean_object* v___x_2748_; 
v___x_2745_ = lean_nat_sub(v___x_2725_, v___x_2722_);
lean_dec(v___x_2722_);
lean_dec(v___x_2725_);
v___x_2746_ = lean_nat_add(v_ruleSelection_2736_, v___x_2745_);
lean_dec(v___x_2745_);
lean_dec(v_ruleSelection_2736_);
if (v_isShared_2744_ == 0)
{
lean_ctor_set(v___x_2743_, 4, v___x_2746_);
v___x_2748_ = v___x_2743_;
goto v_reusejp_2747_;
}
else
{
lean_object* v_reuseFailAlloc_2753_; 
v_reuseFailAlloc_2753_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2753_, 0, v_total_2732_);
lean_ctor_set(v_reuseFailAlloc_2753_, 1, v_configParsing_2733_);
lean_ctor_set(v_reuseFailAlloc_2753_, 2, v_ruleSetConstruction_2734_);
lean_ctor_set(v_reuseFailAlloc_2753_, 3, v_search_2735_);
lean_ctor_set(v_reuseFailAlloc_2753_, 4, v___x_2746_);
lean_ctor_set(v_reuseFailAlloc_2753_, 5, v_script_2737_);
lean_ctor_set(v_reuseFailAlloc_2753_, 6, v_forwardState_2738_);
lean_ctor_set(v_reuseFailAlloc_2753_, 7, v_scriptGenerated_2739_);
lean_ctor_set(v_reuseFailAlloc_2753_, 8, v_ruleStats_2740_);
lean_ctor_set(v_reuseFailAlloc_2753_, 9, v_goalStats_2741_);
v___x_2748_ = v_reuseFailAlloc_2753_;
goto v_reusejp_2747_;
}
v_reusejp_2747_:
{
lean_object* v___x_2750_; 
if (v_isShared_2731_ == 0)
{
lean_ctor_set(v___x_2730_, 1, v___x_2748_);
v___x_2750_ = v___x_2730_;
goto v_reusejp_2749_;
}
else
{
lean_object* v_reuseFailAlloc_2752_; 
v_reuseFailAlloc_2752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2752_, 0, v_rulePatternCache_2728_);
lean_ctor_set(v_reuseFailAlloc_2752_, 1, v___x_2748_);
v___x_2750_ = v_reuseFailAlloc_2752_;
goto v_reusejp_2749_;
}
v_reusejp_2749_:
{
lean_object* v___x_2751_; 
v___x_2751_ = lean_st_ref_set(v_a_2605_, v___x_2750_);
v___y_2700_ = v___y_2717_;
v___y_2701_ = v___y_2719_;
v___y_2702_ = v___y_2718_;
v___y_2703_ = v___y_2720_;
v___y_2704_ = v___y_2721_;
v_a_2705_ = v_a_2724_;
goto v___jp_2699_;
}
}
}
}
}
else
{
lean_object* v_a_2756_; 
lean_dec(v___x_2722_);
v_a_2756_ = lean_ctor_get(v___x_2723_, 0);
lean_inc(v_a_2756_);
lean_dec_ref_known(v___x_2723_, 1);
v___y_2692_ = v___y_2717_;
v___y_2693_ = v___y_2719_;
v___y_2694_ = v___y_2718_;
v___y_2695_ = v___y_2720_;
v___y_2696_ = v___y_2721_;
v_a_2697_ = v_a_2756_;
goto v___jp_2691_;
}
}
v___jp_2757_:
{
lean_object* v_a_2764_; uint8_t v___x_2765_; 
v_a_2764_ = lean_ctor_get(v___y_2763_, 0);
lean_inc(v_a_2764_);
lean_dec_ref(v___y_2763_);
v___x_2765_ = lean_unbox(v_a_2764_);
lean_dec(v_a_2764_);
if (v___x_2765_ == 0)
{
v___y_2708_ = v___y_2758_;
v___y_2709_ = v___y_2760_;
v___y_2710_ = v___y_2759_;
v___y_2711_ = v___y_2761_;
v___y_2712_ = v___y_2762_;
goto v___jp_2707_;
}
else
{
v___y_2717_ = v___y_2758_;
v___y_2718_ = v___y_2760_;
v___y_2719_ = v___y_2759_;
v___y_2720_ = v___y_2761_;
v___y_2721_ = v___y_2762_;
goto v___jp_2716_;
}
}
v___jp_2766_:
{
lean_object* v___x_2773_; double v___x_2774_; double v___x_2775_; lean_object* v___x_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2779_; lean_object* v___x_2780_; 
v___x_2773_ = lean_io_get_num_heartbeats();
v___x_2774_ = lean_float_of_nat(v___y_2770_);
v___x_2775_ = lean_float_of_nat(v___x_2773_);
v___x_2776_ = lean_box_float(v___x_2774_);
v___x_2777_ = lean_box_float(v___x_2775_);
v___x_2778_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2778_, 0, v___x_2776_);
lean_ctor_set(v___x_2778_, 1, v___x_2777_);
v___x_2779_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2779_, 0, v_a_2772_);
lean_ctor_set(v___x_2779_, 1, v___x_2778_);
v___x_2780_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(v___x_2618_, v___x_2621_, v___x_2622_, v_options_2611_, v___y_2771_, v___y_2768_, v___f_2616_, v___x_2779_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v___y_2658_ = v___y_2767_;
v___y_2659_ = v___y_2769_;
v___y_2660_ = v___x_2780_;
goto v___jp_2657_;
}
v___jp_2781_:
{
lean_object* v___x_2788_; 
v___x_2788_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2788_, 0, v_a_2787_);
v___y_2767_ = v___y_2783_;
v___y_2768_ = v___y_2782_;
v___y_2769_ = v___y_2784_;
v___y_2770_ = v___y_2785_;
v___y_2771_ = v___y_2786_;
v_a_2772_ = v___x_2788_;
goto v___jp_2766_;
}
v___jp_2789_:
{
lean_object* v___x_2796_; 
v___x_2796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2796_, 0, v_a_2795_);
v___y_2767_ = v___y_2791_;
v___y_2768_ = v___y_2790_;
v___y_2769_ = v___y_2792_;
v___y_2770_ = v___y_2793_;
v___y_2771_ = v___y_2794_;
v_a_2772_ = v___x_2796_;
goto v___jp_2766_;
}
v___jp_2797_:
{
lean_object* v___x_2803_; lean_object* v___x_2804_; 
v___x_2803_ = lean_io_mono_nanos_now();
lean_inc(v_goal_2600_);
v___x_2804_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_2804_) == 0)
{
lean_object* v_a_2805_; lean_object* v___x_2806_; lean_object* v___x_2807_; lean_object* v_stats_2808_; lean_object* v_rulePatternCache_2809_; lean_object* v___x_2811_; uint8_t v_isShared_2812_; uint8_t v_isSharedCheck_2836_; 
v_a_2805_ = lean_ctor_get(v___x_2804_, 0);
lean_inc(v_a_2805_);
lean_dec_ref_known(v___x_2804_, 1);
v___x_2806_ = lean_io_mono_nanos_now();
v___x_2807_ = lean_st_ref_take(v_a_2605_);
v_stats_2808_ = lean_ctor_get(v___x_2807_, 1);
v_rulePatternCache_2809_ = lean_ctor_get(v___x_2807_, 0);
v_isSharedCheck_2836_ = !lean_is_exclusive(v___x_2807_);
if (v_isSharedCheck_2836_ == 0)
{
v___x_2811_ = v___x_2807_;
v_isShared_2812_ = v_isSharedCheck_2836_;
goto v_resetjp_2810_;
}
else
{
lean_inc(v_stats_2808_);
lean_inc(v_rulePatternCache_2809_);
lean_dec(v___x_2807_);
v___x_2811_ = lean_box(0);
v_isShared_2812_ = v_isSharedCheck_2836_;
goto v_resetjp_2810_;
}
v_resetjp_2810_:
{
lean_object* v_total_2813_; lean_object* v_configParsing_2814_; lean_object* v_ruleSetConstruction_2815_; lean_object* v_search_2816_; lean_object* v_ruleSelection_2817_; lean_object* v_script_2818_; lean_object* v_forwardState_2819_; lean_object* v_scriptGenerated_2820_; lean_object* v_ruleStats_2821_; lean_object* v_goalStats_2822_; lean_object* v___x_2824_; uint8_t v_isShared_2825_; uint8_t v_isSharedCheck_2835_; 
v_total_2813_ = lean_ctor_get(v_stats_2808_, 0);
v_configParsing_2814_ = lean_ctor_get(v_stats_2808_, 1);
v_ruleSetConstruction_2815_ = lean_ctor_get(v_stats_2808_, 2);
v_search_2816_ = lean_ctor_get(v_stats_2808_, 3);
v_ruleSelection_2817_ = lean_ctor_get(v_stats_2808_, 4);
v_script_2818_ = lean_ctor_get(v_stats_2808_, 5);
v_forwardState_2819_ = lean_ctor_get(v_stats_2808_, 6);
v_scriptGenerated_2820_ = lean_ctor_get(v_stats_2808_, 7);
v_ruleStats_2821_ = lean_ctor_get(v_stats_2808_, 8);
v_goalStats_2822_ = lean_ctor_get(v_stats_2808_, 9);
v_isSharedCheck_2835_ = !lean_is_exclusive(v_stats_2808_);
if (v_isSharedCheck_2835_ == 0)
{
v___x_2824_ = v_stats_2808_;
v_isShared_2825_ = v_isSharedCheck_2835_;
goto v_resetjp_2823_;
}
else
{
lean_inc(v_goalStats_2822_);
lean_inc(v_ruleStats_2821_);
lean_inc(v_scriptGenerated_2820_);
lean_inc(v_forwardState_2819_);
lean_inc(v_script_2818_);
lean_inc(v_ruleSelection_2817_);
lean_inc(v_search_2816_);
lean_inc(v_ruleSetConstruction_2815_);
lean_inc(v_configParsing_2814_);
lean_inc(v_total_2813_);
lean_dec(v_stats_2808_);
v___x_2824_ = lean_box(0);
v_isShared_2825_ = v_isSharedCheck_2835_;
goto v_resetjp_2823_;
}
v_resetjp_2823_:
{
lean_object* v___x_2826_; lean_object* v___x_2827_; lean_object* v___x_2829_; 
v___x_2826_ = lean_nat_sub(v___x_2806_, v___x_2803_);
lean_dec(v___x_2803_);
lean_dec(v___x_2806_);
v___x_2827_ = lean_nat_add(v_ruleSelection_2817_, v___x_2826_);
lean_dec(v___x_2826_);
lean_dec(v_ruleSelection_2817_);
if (v_isShared_2825_ == 0)
{
lean_ctor_set(v___x_2824_, 4, v___x_2827_);
v___x_2829_ = v___x_2824_;
goto v_reusejp_2828_;
}
else
{
lean_object* v_reuseFailAlloc_2834_; 
v_reuseFailAlloc_2834_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2834_, 0, v_total_2813_);
lean_ctor_set(v_reuseFailAlloc_2834_, 1, v_configParsing_2814_);
lean_ctor_set(v_reuseFailAlloc_2834_, 2, v_ruleSetConstruction_2815_);
lean_ctor_set(v_reuseFailAlloc_2834_, 3, v_search_2816_);
lean_ctor_set(v_reuseFailAlloc_2834_, 4, v___x_2827_);
lean_ctor_set(v_reuseFailAlloc_2834_, 5, v_script_2818_);
lean_ctor_set(v_reuseFailAlloc_2834_, 6, v_forwardState_2819_);
lean_ctor_set(v_reuseFailAlloc_2834_, 7, v_scriptGenerated_2820_);
lean_ctor_set(v_reuseFailAlloc_2834_, 8, v_ruleStats_2821_);
lean_ctor_set(v_reuseFailAlloc_2834_, 9, v_goalStats_2822_);
v___x_2829_ = v_reuseFailAlloc_2834_;
goto v_reusejp_2828_;
}
v_reusejp_2828_:
{
lean_object* v___x_2831_; 
if (v_isShared_2812_ == 0)
{
lean_ctor_set(v___x_2811_, 1, v___x_2829_);
v___x_2831_ = v___x_2811_;
goto v_reusejp_2830_;
}
else
{
lean_object* v_reuseFailAlloc_2833_; 
v_reuseFailAlloc_2833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2833_, 0, v_rulePatternCache_2809_);
lean_ctor_set(v_reuseFailAlloc_2833_, 1, v___x_2829_);
v___x_2831_ = v_reuseFailAlloc_2833_;
goto v_reusejp_2830_;
}
v_reusejp_2830_:
{
lean_object* v___x_2832_; 
v___x_2832_ = lean_st_ref_set(v_a_2605_, v___x_2831_);
v___y_2782_ = v___y_2799_;
v___y_2783_ = v___y_2798_;
v___y_2784_ = v___y_2800_;
v___y_2785_ = v___y_2801_;
v___y_2786_ = v___y_2802_;
v_a_2787_ = v_a_2805_;
goto v___jp_2781_;
}
}
}
}
}
else
{
lean_object* v_a_2837_; 
lean_dec(v___x_2803_);
v_a_2837_ = lean_ctor_get(v___x_2804_, 0);
lean_inc(v_a_2837_);
lean_dec_ref_known(v___x_2804_, 1);
v___y_2790_ = v___y_2799_;
v___y_2791_ = v___y_2798_;
v___y_2792_ = v___y_2800_;
v___y_2793_ = v___y_2801_;
v___y_2794_ = v___y_2802_;
v_a_2795_ = v_a_2837_;
goto v___jp_2789_;
}
}
v___jp_2838_:
{
lean_object* v___x_2844_; 
lean_inc(v_goal_2600_);
v___x_2844_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_2844_) == 0)
{
lean_object* v_a_2845_; 
v_a_2845_ = lean_ctor_get(v___x_2844_, 0);
lean_inc(v_a_2845_);
lean_dec_ref_known(v___x_2844_, 1);
v___y_2782_ = v___y_2840_;
v___y_2783_ = v___y_2839_;
v___y_2784_ = v___y_2841_;
v___y_2785_ = v___y_2842_;
v___y_2786_ = v___y_2843_;
v_a_2787_ = v_a_2845_;
goto v___jp_2781_;
}
else
{
lean_object* v_a_2846_; 
v_a_2846_ = lean_ctor_get(v___x_2844_, 0);
lean_inc(v_a_2846_);
lean_dec_ref_known(v___x_2844_, 1);
v___y_2790_ = v___y_2840_;
v___y_2791_ = v___y_2839_;
v___y_2792_ = v___y_2841_;
v___y_2793_ = v___y_2842_;
v___y_2794_ = v___y_2843_;
v_a_2795_ = v_a_2846_;
goto v___jp_2789_;
}
}
v___jp_2847_:
{
lean_object* v_a_2854_; uint8_t v___x_2855_; 
v_a_2854_ = lean_ctor_get(v___y_2853_, 0);
lean_inc(v_a_2854_);
lean_dec_ref(v___y_2853_);
v___x_2855_ = lean_unbox(v_a_2854_);
lean_dec(v_a_2854_);
if (v___x_2855_ == 0)
{
v___y_2839_ = v___y_2848_;
v___y_2840_ = v___y_2849_;
v___y_2841_ = v___y_2850_;
v___y_2842_ = v___y_2851_;
v___y_2843_ = v___y_2852_;
goto v___jp_2838_;
}
else
{
v___y_2798_ = v___y_2848_;
v___y_2799_ = v___y_2849_;
v___y_2800_ = v___y_2850_;
v___y_2801_ = v___y_2851_;
v___y_2802_ = v___y_2852_;
goto v___jp_2797_;
}
}
v___jp_2856_:
{
lean_object* v___x_2861_; 
v___x_2861_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v_a_2609_);
if (v___y_2860_ == 0)
{
lean_object* v_a_2862_; lean_object* v___x_2863_; lean_object* v___x_2864_; uint8_t v___x_2865_; 
v_a_2862_ = lean_ctor_get(v___x_2861_, 0);
lean_inc(v_a_2862_);
lean_dec_ref(v___x_2861_);
v___x_2863_ = lean_io_mono_nanos_now();
v___x_2864_ = lp_aesop_Aesop_aesop_collectStats;
v___x_2865_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2611_, v___x_2864_);
if (v___x_2865_ == 0)
{
lean_object* v___x_2866_; lean_object* v___x_2867_; lean_object* v_a_2868_; uint8_t v___x_2869_; 
v___x_2866_ = lp_aesop_Aesop_TraceOption_stats;
v___x_2867_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_2866_, v_a_2608_);
v_a_2868_ = lean_ctor_get(v___x_2867_, 0);
lean_inc(v_a_2868_);
v___x_2869_ = lean_unbox(v_a_2868_);
lean_dec(v_a_2868_);
if (v___x_2869_ == 0)
{
lean_object* v___x_2870_; lean_object* v___x_2871_; uint8_t v___x_2872_; 
lean_dec_ref(v___x_2867_);
v___x_2870_ = lp_aesop_Aesop_aesop_stats_file;
v___x_2871_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_2611_, v___x_2870_);
v___x_2872_ = lean_string_dec_eq(v___x_2871_, v___x_2622_);
lean_dec_ref(v___x_2871_);
if (v___x_2872_ == 0)
{
v___y_2717_ = v___x_2863_;
v___y_2718_ = v___y_2857_;
v___y_2719_ = v_a_2862_;
v___y_2720_ = v___y_2858_;
v___y_2721_ = v___y_2859_;
goto v___jp_2716_;
}
else
{
v___y_2708_ = v___x_2863_;
v___y_2709_ = v___y_2857_;
v___y_2710_ = v_a_2862_;
v___y_2711_ = v___y_2858_;
v___y_2712_ = v___y_2859_;
goto v___jp_2707_;
}
}
else
{
v___y_2758_ = v___x_2863_;
v___y_2759_ = v_a_2862_;
v___y_2760_ = v___y_2857_;
v___y_2761_ = v___y_2858_;
v___y_2762_ = v___y_2859_;
v___y_2763_ = v___x_2867_;
goto v___jp_2757_;
}
}
else
{
v___y_2717_ = v___x_2863_;
v___y_2718_ = v___y_2857_;
v___y_2719_ = v_a_2862_;
v___y_2720_ = v___y_2858_;
v___y_2721_ = v___y_2859_;
goto v___jp_2716_;
}
}
else
{
lean_object* v_a_2873_; lean_object* v___x_2874_; lean_object* v___x_2875_; uint8_t v___x_2876_; 
v_a_2873_ = lean_ctor_get(v___x_2861_, 0);
lean_inc(v_a_2873_);
lean_dec_ref(v___x_2861_);
v___x_2874_ = lean_io_get_num_heartbeats();
v___x_2875_ = lp_aesop_Aesop_aesop_collectStats;
v___x_2876_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2611_, v___x_2875_);
if (v___x_2876_ == 0)
{
lean_object* v___x_2877_; lean_object* v___x_2878_; lean_object* v_a_2879_; uint8_t v___x_2880_; 
v___x_2877_ = lp_aesop_Aesop_TraceOption_stats;
v___x_2878_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_2877_, v_a_2608_);
v_a_2879_ = lean_ctor_get(v___x_2878_, 0);
lean_inc(v_a_2879_);
v___x_2880_ = lean_unbox(v_a_2879_);
lean_dec(v_a_2879_);
if (v___x_2880_ == 0)
{
lean_object* v___x_2881_; lean_object* v___x_2882_; uint8_t v___x_2883_; 
lean_dec_ref(v___x_2878_);
v___x_2881_ = lp_aesop_Aesop_aesop_stats_file;
v___x_2882_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_2611_, v___x_2881_);
v___x_2883_ = lean_string_dec_eq(v___x_2882_, v___x_2622_);
lean_dec_ref(v___x_2882_);
if (v___x_2883_ == 0)
{
v___y_2798_ = v___y_2857_;
v___y_2799_ = v_a_2873_;
v___y_2800_ = v___y_2858_;
v___y_2801_ = v___x_2874_;
v___y_2802_ = v___y_2859_;
goto v___jp_2797_;
}
else
{
v___y_2839_ = v___y_2857_;
v___y_2840_ = v_a_2873_;
v___y_2841_ = v___y_2858_;
v___y_2842_ = v___x_2874_;
v___y_2843_ = v___y_2859_;
goto v___jp_2838_;
}
}
else
{
v___y_2848_ = v___y_2857_;
v___y_2849_ = v_a_2873_;
v___y_2850_ = v___y_2858_;
v___y_2851_ = v___x_2874_;
v___y_2852_ = v___y_2859_;
v___y_2853_ = v___x_2878_;
goto v___jp_2847_;
}
}
else
{
v___y_2798_ = v___y_2857_;
v___y_2799_ = v_a_2873_;
v___y_2800_ = v___y_2858_;
v___y_2801_ = v___x_2874_;
v___y_2802_ = v___y_2859_;
goto v___jp_2797_;
}
}
}
v___jp_2884_:
{
lean_object* v___x_2888_; double v___x_2889_; double v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; 
v___x_2888_ = lean_io_get_num_heartbeats();
v___x_2889_ = lean_float_of_nat(v___y_2885_);
v___x_2890_ = lean_float_of_nat(v___x_2888_);
v___x_2891_ = lean_box_float(v___x_2889_);
v___x_2892_ = lean_box_float(v___x_2890_);
v___x_2893_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2893_, 0, v___x_2891_);
lean_ctor_set(v___x_2893_, 1, v___x_2892_);
v___x_2894_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2894_, 0, v_a_2887_);
lean_ctor_set(v___x_2894_, 1, v___x_2893_);
v___x_2895_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1(v___x_2618_, v___x_2621_, v___x_2622_, v_options_2611_, v___x_2636_, v___y_2886_, v___f_2634_, v___x_2894_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
return v___x_2895_;
}
v___jp_2896_:
{
lean_object* v___x_2900_; 
v___x_2900_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2900_, 0, v_a_2899_);
v___y_2885_ = v___y_2897_;
v___y_2886_ = v___y_2898_;
v_a_2887_ = v___x_2900_;
goto v___jp_2884_;
}
v___jp_2901_:
{
if (lean_obj_tag(v___y_2904_) == 0)
{
lean_object* v_a_2905_; lean_object* v___x_2906_; 
v_a_2905_ = lean_ctor_get(v___y_2904_, 0);
lean_inc(v_a_2905_);
lean_dec_ref_known(v___y_2904_, 1);
v___x_2906_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_2600_, v_mvars_2601_, v_preState_2602_, v_a_2905_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
lean_dec(v_a_2905_);
if (lean_obj_tag(v___x_2906_) == 0)
{
lean_object* v_a_2907_; lean_object* v___x_2909_; uint8_t v_isShared_2910_; uint8_t v_isSharedCheck_2914_; 
v_a_2907_ = lean_ctor_get(v___x_2906_, 0);
v_isSharedCheck_2914_ = !lean_is_exclusive(v___x_2906_);
if (v_isSharedCheck_2914_ == 0)
{
v___x_2909_ = v___x_2906_;
v_isShared_2910_ = v_isSharedCheck_2914_;
goto v_resetjp_2908_;
}
else
{
lean_inc(v_a_2907_);
lean_dec(v___x_2906_);
v___x_2909_ = lean_box(0);
v_isShared_2910_ = v_isSharedCheck_2914_;
goto v_resetjp_2908_;
}
v_resetjp_2908_:
{
lean_object* v___x_2912_; 
if (v_isShared_2910_ == 0)
{
lean_ctor_set_tag(v___x_2909_, 1);
v___x_2912_ = v___x_2909_;
goto v_reusejp_2911_;
}
else
{
lean_object* v_reuseFailAlloc_2913_; 
v_reuseFailAlloc_2913_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2913_, 0, v_a_2907_);
v___x_2912_ = v_reuseFailAlloc_2913_;
goto v_reusejp_2911_;
}
v_reusejp_2911_:
{
v___y_2885_ = v___y_2902_;
v___y_2886_ = v___y_2903_;
v_a_2887_ = v___x_2912_;
goto v___jp_2884_;
}
}
}
else
{
lean_object* v_a_2915_; 
v_a_2915_ = lean_ctor_get(v___x_2906_, 0);
lean_inc(v_a_2915_);
lean_dec_ref_known(v___x_2906_, 1);
v___y_2897_ = v___y_2902_;
v___y_2898_ = v___y_2903_;
v_a_2899_ = v_a_2915_;
goto v___jp_2896_;
}
}
else
{
lean_object* v_a_2916_; 
lean_dec_ref(v_mvars_2601_);
lean_dec(v_goal_2600_);
v_a_2916_ = lean_ctor_get(v___y_2904_, 0);
lean_inc(v_a_2916_);
lean_dec_ref_known(v___y_2904_, 1);
v___y_2897_ = v___y_2902_;
v___y_2898_ = v___y_2903_;
v_a_2899_ = v_a_2916_;
goto v___jp_2896_;
}
}
v___jp_2917_:
{
lean_object* v___x_2924_; double v___x_2925_; double v___x_2926_; double v___x_2927_; double v___x_2928_; double v___x_2929_; lean_object* v___x_2930_; lean_object* v___x_2931_; lean_object* v___x_2932_; lean_object* v___x_2933_; lean_object* v___x_2934_; 
v___x_2924_ = lean_io_mono_nanos_now();
v___x_2925_ = lean_float_of_nat(v___y_2918_);
v___x_2926_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_2927_ = lean_float_div(v___x_2925_, v___x_2926_);
v___x_2928_ = lean_float_of_nat(v___x_2924_);
v___x_2929_ = lean_float_div(v___x_2928_, v___x_2926_);
v___x_2930_ = lean_box_float(v___x_2927_);
v___x_2931_ = lean_box_float(v___x_2929_);
v___x_2932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2932_, 0, v___x_2930_);
lean_ctor_set(v___x_2932_, 1, v___x_2931_);
v___x_2933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2933_, 0, v_a_2923_);
lean_ctor_set(v___x_2933_, 1, v___x_2932_);
v___x_2934_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(v___x_2618_, v___x_2621_, v___x_2622_, v_options_2611_, v___y_2920_, v___y_2922_, v___f_2616_, v___x_2933_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v___y_2902_ = v___y_2919_;
v___y_2903_ = v___y_2921_;
v___y_2904_ = v___x_2934_;
goto v___jp_2901_;
}
v___jp_2935_:
{
lean_object* v___x_2942_; 
v___x_2942_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2942_, 0, v_a_2941_);
v___y_2918_ = v___y_2936_;
v___y_2919_ = v___y_2937_;
v___y_2920_ = v___y_2938_;
v___y_2921_ = v___y_2939_;
v___y_2922_ = v___y_2940_;
v_a_2923_ = v___x_2942_;
goto v___jp_2917_;
}
v___jp_2943_:
{
lean_object* v___x_2950_; 
v___x_2950_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2950_, 0, v_a_2949_);
v___y_2918_ = v___y_2944_;
v___y_2919_ = v___y_2945_;
v___y_2920_ = v___y_2946_;
v___y_2921_ = v___y_2947_;
v___y_2922_ = v___y_2948_;
v_a_2923_ = v___x_2950_;
goto v___jp_2917_;
}
v___jp_2951_:
{
lean_object* v___x_2957_; lean_object* v___x_2958_; 
v___x_2957_ = lean_io_mono_nanos_now();
lean_inc(v_goal_2600_);
v___x_2958_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_2958_) == 0)
{
lean_object* v_a_2959_; lean_object* v___x_2960_; lean_object* v___x_2961_; lean_object* v_stats_2962_; lean_object* v_rulePatternCache_2963_; lean_object* v___x_2965_; uint8_t v_isShared_2966_; uint8_t v_isSharedCheck_2990_; 
v_a_2959_ = lean_ctor_get(v___x_2958_, 0);
lean_inc(v_a_2959_);
lean_dec_ref_known(v___x_2958_, 1);
v___x_2960_ = lean_io_mono_nanos_now();
v___x_2961_ = lean_st_ref_take(v_a_2605_);
v_stats_2962_ = lean_ctor_get(v___x_2961_, 1);
v_rulePatternCache_2963_ = lean_ctor_get(v___x_2961_, 0);
v_isSharedCheck_2990_ = !lean_is_exclusive(v___x_2961_);
if (v_isSharedCheck_2990_ == 0)
{
v___x_2965_ = v___x_2961_;
v_isShared_2966_ = v_isSharedCheck_2990_;
goto v_resetjp_2964_;
}
else
{
lean_inc(v_stats_2962_);
lean_inc(v_rulePatternCache_2963_);
lean_dec(v___x_2961_);
v___x_2965_ = lean_box(0);
v_isShared_2966_ = v_isSharedCheck_2990_;
goto v_resetjp_2964_;
}
v_resetjp_2964_:
{
lean_object* v_total_2967_; lean_object* v_configParsing_2968_; lean_object* v_ruleSetConstruction_2969_; lean_object* v_search_2970_; lean_object* v_ruleSelection_2971_; lean_object* v_script_2972_; lean_object* v_forwardState_2973_; lean_object* v_scriptGenerated_2974_; lean_object* v_ruleStats_2975_; lean_object* v_goalStats_2976_; lean_object* v___x_2978_; uint8_t v_isShared_2979_; uint8_t v_isSharedCheck_2989_; 
v_total_2967_ = lean_ctor_get(v_stats_2962_, 0);
v_configParsing_2968_ = lean_ctor_get(v_stats_2962_, 1);
v_ruleSetConstruction_2969_ = lean_ctor_get(v_stats_2962_, 2);
v_search_2970_ = lean_ctor_get(v_stats_2962_, 3);
v_ruleSelection_2971_ = lean_ctor_get(v_stats_2962_, 4);
v_script_2972_ = lean_ctor_get(v_stats_2962_, 5);
v_forwardState_2973_ = lean_ctor_get(v_stats_2962_, 6);
v_scriptGenerated_2974_ = lean_ctor_get(v_stats_2962_, 7);
v_ruleStats_2975_ = lean_ctor_get(v_stats_2962_, 8);
v_goalStats_2976_ = lean_ctor_get(v_stats_2962_, 9);
v_isSharedCheck_2989_ = !lean_is_exclusive(v_stats_2962_);
if (v_isSharedCheck_2989_ == 0)
{
v___x_2978_ = v_stats_2962_;
v_isShared_2979_ = v_isSharedCheck_2989_;
goto v_resetjp_2977_;
}
else
{
lean_inc(v_goalStats_2976_);
lean_inc(v_ruleStats_2975_);
lean_inc(v_scriptGenerated_2974_);
lean_inc(v_forwardState_2973_);
lean_inc(v_script_2972_);
lean_inc(v_ruleSelection_2971_);
lean_inc(v_search_2970_);
lean_inc(v_ruleSetConstruction_2969_);
lean_inc(v_configParsing_2968_);
lean_inc(v_total_2967_);
lean_dec(v_stats_2962_);
v___x_2978_ = lean_box(0);
v_isShared_2979_ = v_isSharedCheck_2989_;
goto v_resetjp_2977_;
}
v_resetjp_2977_:
{
lean_object* v___x_2980_; lean_object* v___x_2981_; lean_object* v___x_2983_; 
v___x_2980_ = lean_nat_sub(v___x_2960_, v___x_2957_);
lean_dec(v___x_2957_);
lean_dec(v___x_2960_);
v___x_2981_ = lean_nat_add(v_ruleSelection_2971_, v___x_2980_);
lean_dec(v___x_2980_);
lean_dec(v_ruleSelection_2971_);
if (v_isShared_2979_ == 0)
{
lean_ctor_set(v___x_2978_, 4, v___x_2981_);
v___x_2983_ = v___x_2978_;
goto v_reusejp_2982_;
}
else
{
lean_object* v_reuseFailAlloc_2988_; 
v_reuseFailAlloc_2988_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2988_, 0, v_total_2967_);
lean_ctor_set(v_reuseFailAlloc_2988_, 1, v_configParsing_2968_);
lean_ctor_set(v_reuseFailAlloc_2988_, 2, v_ruleSetConstruction_2969_);
lean_ctor_set(v_reuseFailAlloc_2988_, 3, v_search_2970_);
lean_ctor_set(v_reuseFailAlloc_2988_, 4, v___x_2981_);
lean_ctor_set(v_reuseFailAlloc_2988_, 5, v_script_2972_);
lean_ctor_set(v_reuseFailAlloc_2988_, 6, v_forwardState_2973_);
lean_ctor_set(v_reuseFailAlloc_2988_, 7, v_scriptGenerated_2974_);
lean_ctor_set(v_reuseFailAlloc_2988_, 8, v_ruleStats_2975_);
lean_ctor_set(v_reuseFailAlloc_2988_, 9, v_goalStats_2976_);
v___x_2983_ = v_reuseFailAlloc_2988_;
goto v_reusejp_2982_;
}
v_reusejp_2982_:
{
lean_object* v___x_2985_; 
if (v_isShared_2966_ == 0)
{
lean_ctor_set(v___x_2965_, 1, v___x_2983_);
v___x_2985_ = v___x_2965_;
goto v_reusejp_2984_;
}
else
{
lean_object* v_reuseFailAlloc_2987_; 
v_reuseFailAlloc_2987_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2987_, 0, v_rulePatternCache_2963_);
lean_ctor_set(v_reuseFailAlloc_2987_, 1, v___x_2983_);
v___x_2985_ = v_reuseFailAlloc_2987_;
goto v_reusejp_2984_;
}
v_reusejp_2984_:
{
lean_object* v___x_2986_; 
v___x_2986_ = lean_st_ref_set(v_a_2605_, v___x_2985_);
v___y_2936_ = v___y_2952_;
v___y_2937_ = v___y_2953_;
v___y_2938_ = v___y_2954_;
v___y_2939_ = v___y_2955_;
v___y_2940_ = v___y_2956_;
v_a_2941_ = v_a_2959_;
goto v___jp_2935_;
}
}
}
}
}
else
{
lean_object* v_a_2991_; 
lean_dec(v___x_2957_);
v_a_2991_ = lean_ctor_get(v___x_2958_, 0);
lean_inc(v_a_2991_);
lean_dec_ref_known(v___x_2958_, 1);
v___y_2944_ = v___y_2952_;
v___y_2945_ = v___y_2953_;
v___y_2946_ = v___y_2954_;
v___y_2947_ = v___y_2955_;
v___y_2948_ = v___y_2956_;
v_a_2949_ = v_a_2991_;
goto v___jp_2943_;
}
}
v___jp_2992_:
{
lean_object* v___x_2998_; 
lean_inc(v_goal_2600_);
v___x_2998_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_2998_) == 0)
{
lean_object* v_a_2999_; 
v_a_2999_ = lean_ctor_get(v___x_2998_, 0);
lean_inc(v_a_2999_);
lean_dec_ref_known(v___x_2998_, 1);
v___y_2936_ = v___y_2993_;
v___y_2937_ = v___y_2994_;
v___y_2938_ = v___y_2995_;
v___y_2939_ = v___y_2996_;
v___y_2940_ = v___y_2997_;
v_a_2941_ = v_a_2999_;
goto v___jp_2935_;
}
else
{
lean_object* v_a_3000_; 
v_a_3000_ = lean_ctor_get(v___x_2998_, 0);
lean_inc(v_a_3000_);
lean_dec_ref_known(v___x_2998_, 1);
v___y_2944_ = v___y_2993_;
v___y_2945_ = v___y_2994_;
v___y_2946_ = v___y_2995_;
v___y_2947_ = v___y_2996_;
v___y_2948_ = v___y_2997_;
v_a_2949_ = v_a_3000_;
goto v___jp_2943_;
}
}
v___jp_3001_:
{
lean_object* v_a_3008_; uint8_t v___x_3009_; 
v_a_3008_ = lean_ctor_get(v___y_3007_, 0);
lean_inc(v_a_3008_);
lean_dec_ref(v___y_3007_);
v___x_3009_ = lean_unbox(v_a_3008_);
lean_dec(v_a_3008_);
if (v___x_3009_ == 0)
{
v___y_2993_ = v___y_3002_;
v___y_2994_ = v___y_3003_;
v___y_2995_ = v___y_3004_;
v___y_2996_ = v___y_3005_;
v___y_2997_ = v___y_3006_;
goto v___jp_2992_;
}
else
{
v___y_2952_ = v___y_3002_;
v___y_2953_ = v___y_3003_;
v___y_2954_ = v___y_3004_;
v___y_2955_ = v___y_3005_;
v___y_2956_ = v___y_3006_;
goto v___jp_2951_;
}
}
v___jp_3010_:
{
lean_object* v___x_3017_; double v___x_3018_; double v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; lean_object* v___x_3024_; 
v___x_3017_ = lean_io_get_num_heartbeats();
v___x_3018_ = lean_float_of_nat(v___y_3014_);
v___x_3019_ = lean_float_of_nat(v___x_3017_);
v___x_3020_ = lean_box_float(v___x_3018_);
v___x_3021_ = lean_box_float(v___x_3019_);
v___x_3022_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3022_, 0, v___x_3020_);
lean_ctor_set(v___x_3022_, 1, v___x_3021_);
v___x_3023_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3023_, 0, v_a_3016_);
lean_ctor_set(v___x_3023_, 1, v___x_3022_);
v___x_3024_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__0(v___x_2618_, v___x_2621_, v___x_2622_, v_options_2611_, v___y_3012_, v___y_3015_, v___f_2616_, v___x_3023_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v___y_2902_ = v___y_3011_;
v___y_2903_ = v___y_3013_;
v___y_2904_ = v___x_3024_;
goto v___jp_2901_;
}
v___jp_3025_:
{
lean_object* v___x_3032_; 
v___x_3032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3032_, 0, v_a_3031_);
v___y_3011_ = v___y_3026_;
v___y_3012_ = v___y_3027_;
v___y_3013_ = v___y_3029_;
v___y_3014_ = v___y_3028_;
v___y_3015_ = v___y_3030_;
v_a_3016_ = v___x_3032_;
goto v___jp_3010_;
}
v___jp_3033_:
{
lean_object* v___x_3040_; 
v___x_3040_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3040_, 0, v_a_3039_);
v___y_3011_ = v___y_3034_;
v___y_3012_ = v___y_3035_;
v___y_3013_ = v___y_3037_;
v___y_3014_ = v___y_3036_;
v___y_3015_ = v___y_3038_;
v_a_3016_ = v___x_3040_;
goto v___jp_3010_;
}
v___jp_3041_:
{
lean_object* v___x_3047_; lean_object* v___x_3048_; 
v___x_3047_ = lean_io_mono_nanos_now();
lean_inc(v_goal_2600_);
v___x_3048_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_3048_) == 0)
{
lean_object* v_a_3049_; lean_object* v___x_3050_; lean_object* v___x_3051_; lean_object* v_stats_3052_; lean_object* v_rulePatternCache_3053_; lean_object* v___x_3055_; uint8_t v_isShared_3056_; uint8_t v_isSharedCheck_3080_; 
v_a_3049_ = lean_ctor_get(v___x_3048_, 0);
lean_inc(v_a_3049_);
lean_dec_ref_known(v___x_3048_, 1);
v___x_3050_ = lean_io_mono_nanos_now();
v___x_3051_ = lean_st_ref_take(v_a_2605_);
v_stats_3052_ = lean_ctor_get(v___x_3051_, 1);
v_rulePatternCache_3053_ = lean_ctor_get(v___x_3051_, 0);
v_isSharedCheck_3080_ = !lean_is_exclusive(v___x_3051_);
if (v_isSharedCheck_3080_ == 0)
{
v___x_3055_ = v___x_3051_;
v_isShared_3056_ = v_isSharedCheck_3080_;
goto v_resetjp_3054_;
}
else
{
lean_inc(v_stats_3052_);
lean_inc(v_rulePatternCache_3053_);
lean_dec(v___x_3051_);
v___x_3055_ = lean_box(0);
v_isShared_3056_ = v_isSharedCheck_3080_;
goto v_resetjp_3054_;
}
v_resetjp_3054_:
{
lean_object* v_total_3057_; lean_object* v_configParsing_3058_; lean_object* v_ruleSetConstruction_3059_; lean_object* v_search_3060_; lean_object* v_ruleSelection_3061_; lean_object* v_script_3062_; lean_object* v_forwardState_3063_; lean_object* v_scriptGenerated_3064_; lean_object* v_ruleStats_3065_; lean_object* v_goalStats_3066_; lean_object* v___x_3068_; uint8_t v_isShared_3069_; uint8_t v_isSharedCheck_3079_; 
v_total_3057_ = lean_ctor_get(v_stats_3052_, 0);
v_configParsing_3058_ = lean_ctor_get(v_stats_3052_, 1);
v_ruleSetConstruction_3059_ = lean_ctor_get(v_stats_3052_, 2);
v_search_3060_ = lean_ctor_get(v_stats_3052_, 3);
v_ruleSelection_3061_ = lean_ctor_get(v_stats_3052_, 4);
v_script_3062_ = lean_ctor_get(v_stats_3052_, 5);
v_forwardState_3063_ = lean_ctor_get(v_stats_3052_, 6);
v_scriptGenerated_3064_ = lean_ctor_get(v_stats_3052_, 7);
v_ruleStats_3065_ = lean_ctor_get(v_stats_3052_, 8);
v_goalStats_3066_ = lean_ctor_get(v_stats_3052_, 9);
v_isSharedCheck_3079_ = !lean_is_exclusive(v_stats_3052_);
if (v_isSharedCheck_3079_ == 0)
{
v___x_3068_ = v_stats_3052_;
v_isShared_3069_ = v_isSharedCheck_3079_;
goto v_resetjp_3067_;
}
else
{
lean_inc(v_goalStats_3066_);
lean_inc(v_ruleStats_3065_);
lean_inc(v_scriptGenerated_3064_);
lean_inc(v_forwardState_3063_);
lean_inc(v_script_3062_);
lean_inc(v_ruleSelection_3061_);
lean_inc(v_search_3060_);
lean_inc(v_ruleSetConstruction_3059_);
lean_inc(v_configParsing_3058_);
lean_inc(v_total_3057_);
lean_dec(v_stats_3052_);
v___x_3068_ = lean_box(0);
v_isShared_3069_ = v_isSharedCheck_3079_;
goto v_resetjp_3067_;
}
v_resetjp_3067_:
{
lean_object* v___x_3070_; lean_object* v___x_3071_; lean_object* v___x_3073_; 
v___x_3070_ = lean_nat_sub(v___x_3050_, v___x_3047_);
lean_dec(v___x_3047_);
lean_dec(v___x_3050_);
v___x_3071_ = lean_nat_add(v_ruleSelection_3061_, v___x_3070_);
lean_dec(v___x_3070_);
lean_dec(v_ruleSelection_3061_);
if (v_isShared_3069_ == 0)
{
lean_ctor_set(v___x_3068_, 4, v___x_3071_);
v___x_3073_ = v___x_3068_;
goto v_reusejp_3072_;
}
else
{
lean_object* v_reuseFailAlloc_3078_; 
v_reuseFailAlloc_3078_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3078_, 0, v_total_3057_);
lean_ctor_set(v_reuseFailAlloc_3078_, 1, v_configParsing_3058_);
lean_ctor_set(v_reuseFailAlloc_3078_, 2, v_ruleSetConstruction_3059_);
lean_ctor_set(v_reuseFailAlloc_3078_, 3, v_search_3060_);
lean_ctor_set(v_reuseFailAlloc_3078_, 4, v___x_3071_);
lean_ctor_set(v_reuseFailAlloc_3078_, 5, v_script_3062_);
lean_ctor_set(v_reuseFailAlloc_3078_, 6, v_forwardState_3063_);
lean_ctor_set(v_reuseFailAlloc_3078_, 7, v_scriptGenerated_3064_);
lean_ctor_set(v_reuseFailAlloc_3078_, 8, v_ruleStats_3065_);
lean_ctor_set(v_reuseFailAlloc_3078_, 9, v_goalStats_3066_);
v___x_3073_ = v_reuseFailAlloc_3078_;
goto v_reusejp_3072_;
}
v_reusejp_3072_:
{
lean_object* v___x_3075_; 
if (v_isShared_3056_ == 0)
{
lean_ctor_set(v___x_3055_, 1, v___x_3073_);
v___x_3075_ = v___x_3055_;
goto v_reusejp_3074_;
}
else
{
lean_object* v_reuseFailAlloc_3077_; 
v_reuseFailAlloc_3077_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3077_, 0, v_rulePatternCache_3053_);
lean_ctor_set(v_reuseFailAlloc_3077_, 1, v___x_3073_);
v___x_3075_ = v_reuseFailAlloc_3077_;
goto v_reusejp_3074_;
}
v_reusejp_3074_:
{
lean_object* v___x_3076_; 
v___x_3076_ = lean_st_ref_set(v_a_2605_, v___x_3075_);
v___y_3026_ = v___y_3042_;
v___y_3027_ = v___y_3043_;
v___y_3028_ = v___y_3045_;
v___y_3029_ = v___y_3044_;
v___y_3030_ = v___y_3046_;
v_a_3031_ = v_a_3049_;
goto v___jp_3025_;
}
}
}
}
}
else
{
lean_object* v_a_3081_; 
lean_dec(v___x_3047_);
v_a_3081_ = lean_ctor_get(v___x_3048_, 0);
lean_inc(v_a_3081_);
lean_dec_ref_known(v___x_3048_, 1);
v___y_3034_ = v___y_3042_;
v___y_3035_ = v___y_3043_;
v___y_3036_ = v___y_3045_;
v___y_3037_ = v___y_3044_;
v___y_3038_ = v___y_3046_;
v_a_3039_ = v_a_3081_;
goto v___jp_3033_;
}
}
v___jp_3082_:
{
lean_object* v___x_3088_; 
lean_inc(v_goal_2600_);
v___x_3088_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
if (lean_obj_tag(v___x_3088_) == 0)
{
lean_object* v_a_3089_; 
v_a_3089_ = lean_ctor_get(v___x_3088_, 0);
lean_inc(v_a_3089_);
lean_dec_ref_known(v___x_3088_, 1);
v___y_3026_ = v___y_3083_;
v___y_3027_ = v___y_3084_;
v___y_3028_ = v___y_3086_;
v___y_3029_ = v___y_3085_;
v___y_3030_ = v___y_3087_;
v_a_3031_ = v_a_3089_;
goto v___jp_3025_;
}
else
{
lean_object* v_a_3090_; 
v_a_3090_ = lean_ctor_get(v___x_3088_, 0);
lean_inc(v_a_3090_);
lean_dec_ref_known(v___x_3088_, 1);
v___y_3034_ = v___y_3083_;
v___y_3035_ = v___y_3084_;
v___y_3036_ = v___y_3086_;
v___y_3037_ = v___y_3085_;
v___y_3038_ = v___y_3087_;
v_a_3039_ = v_a_3090_;
goto v___jp_3033_;
}
}
v___jp_3091_:
{
lean_object* v_a_3098_; uint8_t v___x_3099_; 
v_a_3098_ = lean_ctor_get(v___y_3097_, 0);
lean_inc(v_a_3098_);
lean_dec_ref(v___y_3097_);
v___x_3099_ = lean_unbox(v_a_3098_);
lean_dec(v_a_3098_);
if (v___x_3099_ == 0)
{
v___y_3083_ = v___y_3092_;
v___y_3084_ = v___y_3093_;
v___y_3085_ = v___y_3094_;
v___y_3086_ = v___y_3095_;
v___y_3087_ = v___y_3096_;
goto v___jp_3082_;
}
else
{
v___y_3042_ = v___y_3092_;
v___y_3043_ = v___y_3093_;
v___y_3044_ = v___y_3094_;
v___y_3045_ = v___y_3095_;
v___y_3046_ = v___y_3096_;
goto v___jp_3041_;
}
}
v___jp_3100_:
{
lean_object* v___x_3105_; 
v___x_3105_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v_a_2609_);
if (v___y_3104_ == 0)
{
lean_object* v_a_3106_; lean_object* v___x_3107_; lean_object* v___x_3108_; uint8_t v___x_3109_; 
v_a_3106_ = lean_ctor_get(v___x_3105_, 0);
lean_inc(v_a_3106_);
lean_dec_ref(v___x_3105_);
v___x_3107_ = lean_io_mono_nanos_now();
v___x_3108_ = lp_aesop_Aesop_aesop_collectStats;
v___x_3109_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2611_, v___x_3108_);
if (v___x_3109_ == 0)
{
lean_object* v___x_3110_; lean_object* v___x_3111_; lean_object* v_a_3112_; uint8_t v___x_3113_; 
v___x_3110_ = lp_aesop_Aesop_TraceOption_stats;
v___x_3111_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_3110_, v_a_2608_);
v_a_3112_ = lean_ctor_get(v___x_3111_, 0);
lean_inc(v_a_3112_);
v___x_3113_ = lean_unbox(v_a_3112_);
lean_dec(v_a_3112_);
if (v___x_3113_ == 0)
{
lean_object* v___x_3114_; lean_object* v___x_3115_; uint8_t v___x_3116_; 
lean_dec_ref(v___x_3111_);
v___x_3114_ = lp_aesop_Aesop_aesop_stats_file;
v___x_3115_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_2611_, v___x_3114_);
v___x_3116_ = lean_string_dec_eq(v___x_3115_, v___x_2622_);
lean_dec_ref(v___x_3115_);
if (v___x_3116_ == 0)
{
v___y_2952_ = v___x_3107_;
v___y_2953_ = v___y_3101_;
v___y_2954_ = v___y_3102_;
v___y_2955_ = v___y_3103_;
v___y_2956_ = v_a_3106_;
goto v___jp_2951_;
}
else
{
v___y_2993_ = v___x_3107_;
v___y_2994_ = v___y_3101_;
v___y_2995_ = v___y_3102_;
v___y_2996_ = v___y_3103_;
v___y_2997_ = v_a_3106_;
goto v___jp_2992_;
}
}
else
{
v___y_3002_ = v___x_3107_;
v___y_3003_ = v___y_3101_;
v___y_3004_ = v___y_3102_;
v___y_3005_ = v___y_3103_;
v___y_3006_ = v_a_3106_;
v___y_3007_ = v___x_3111_;
goto v___jp_3001_;
}
}
else
{
v___y_2952_ = v___x_3107_;
v___y_2953_ = v___y_3101_;
v___y_2954_ = v___y_3102_;
v___y_2955_ = v___y_3103_;
v___y_2956_ = v_a_3106_;
goto v___jp_2951_;
}
}
else
{
lean_object* v_a_3117_; lean_object* v___x_3118_; lean_object* v___x_3119_; uint8_t v___x_3120_; 
v_a_3117_ = lean_ctor_get(v___x_3105_, 0);
lean_inc(v_a_3117_);
lean_dec_ref(v___x_3105_);
v___x_3118_ = lean_io_get_num_heartbeats();
v___x_3119_ = lp_aesop_Aesop_aesop_collectStats;
v___x_3120_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2611_, v___x_3119_);
if (v___x_3120_ == 0)
{
lean_object* v___x_3121_; lean_object* v___x_3122_; lean_object* v_a_3123_; uint8_t v___x_3124_; 
v___x_3121_ = lp_aesop_Aesop_TraceOption_stats;
v___x_3122_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_3121_, v_a_2608_);
v_a_3123_ = lean_ctor_get(v___x_3122_, 0);
lean_inc(v_a_3123_);
v___x_3124_ = lean_unbox(v_a_3123_);
lean_dec(v_a_3123_);
if (v___x_3124_ == 0)
{
lean_object* v___x_3125_; lean_object* v___x_3126_; uint8_t v___x_3127_; 
lean_dec_ref(v___x_3122_);
v___x_3125_ = lp_aesop_Aesop_aesop_stats_file;
v___x_3126_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_2611_, v___x_3125_);
v___x_3127_ = lean_string_dec_eq(v___x_3126_, v___x_2622_);
lean_dec_ref(v___x_3126_);
if (v___x_3127_ == 0)
{
v___y_3042_ = v___y_3101_;
v___y_3043_ = v___y_3102_;
v___y_3044_ = v___y_3103_;
v___y_3045_ = v___x_3118_;
v___y_3046_ = v_a_3117_;
goto v___jp_3041_;
}
else
{
v___y_3083_ = v___y_3101_;
v___y_3084_ = v___y_3102_;
v___y_3085_ = v___y_3103_;
v___y_3086_ = v___x_3118_;
v___y_3087_ = v_a_3117_;
goto v___jp_3082_;
}
}
else
{
v___y_3092_ = v___y_3101_;
v___y_3093_ = v___y_3102_;
v___y_3094_ = v___y_3103_;
v___y_3095_ = v___x_3118_;
v___y_3096_ = v_a_3117_;
v___y_3097_ = v___x_3122_;
goto v___jp_3091_;
}
}
else
{
v___y_3042_ = v___y_3101_;
v___y_3043_ = v___y_3102_;
v___y_3044_ = v___y_3103_;
v___y_3045_ = v___x_3118_;
v___y_3046_ = v_a_3117_;
goto v___jp_3041_;
}
}
}
v___jp_3128_:
{
lean_object* v___x_3129_; lean_object* v_a_3130_; lean_object* v___x_3131_; uint8_t v___x_3132_; 
v___x_3129_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v_a_2609_);
v_a_3130_ = lean_ctor_get(v___x_3129_, 0);
lean_inc(v_a_3130_);
lean_dec_ref(v___x_3129_);
v___x_3131_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3132_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2611_, v___x_3131_);
if (v___x_3132_ == 0)
{
lean_object* v___x_3133_; 
v___x_3133_ = lean_io_mono_nanos_now();
if (v___x_2636_ == 0)
{
lean_object* v___x_3134_; uint8_t v___x_3135_; 
v___x_3134_ = l_Lean_trace_profiler;
v___x_3135_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2611_, v___x_3134_);
if (v___x_3135_ == 0)
{
lean_object* v___x_3136_; lean_object* v_a_3137_; uint8_t v___x_3138_; lean_object* v___x_3139_; lean_object* v_a_3140_; uint8_t v___x_3141_; lean_object* v___x_3142_; 
v___x_3136_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v_a_3137_ = lean_ctor_get(v___x_3136_, 0);
lean_inc(v_a_3137_);
lean_dec_ref(v___x_3136_);
v___x_3138_ = lean_unbox(v_a_3137_);
lean_dec(v_a_3137_);
v___x_3139_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(v___x_3138_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v_a_3140_ = lean_ctor_get(v___x_3139_, 0);
lean_inc(v_a_3140_);
lean_dec_ref(v___x_3139_);
v___x_3141_ = lean_unbox(v_a_3140_);
lean_dec(v_a_3140_);
lean_inc(v_goal_2600_);
v___x_3142_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__1(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v___x_3141_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v___y_2658_ = v_a_3130_;
v___y_2659_ = v___x_3133_;
v___y_2660_ = v___x_3142_;
goto v___jp_2657_;
}
else
{
v___y_2857_ = v_a_3130_;
v___y_2858_ = v___x_3133_;
v___y_2859_ = v___x_2636_;
v___y_2860_ = v___x_3132_;
goto v___jp_2856_;
}
}
else
{
v___y_2857_ = v_a_3130_;
v___y_2858_ = v___x_3133_;
v___y_2859_ = v___x_2636_;
v___y_2860_ = v___x_3132_;
goto v___jp_2856_;
}
}
else
{
lean_object* v___x_3143_; 
v___x_3143_ = lean_io_get_num_heartbeats();
if (v___x_2636_ == 0)
{
lean_object* v___x_3144_; uint8_t v___x_3145_; 
v___x_3144_ = l_Lean_trace_profiler;
v___x_3145_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_2611_, v___x_3144_);
if (v___x_3145_ == 0)
{
lean_object* v___x_3146_; lean_object* v_a_3147_; uint8_t v___x_3148_; lean_object* v___x_3149_; lean_object* v_a_3150_; uint8_t v___x_3151_; lean_object* v___x_3152_; 
v___x_3146_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v_a_3147_ = lean_ctor_get(v___x_3146_, 0);
lean_inc(v_a_3147_);
lean_dec_ref(v___x_3146_);
v___x_3148_ = lean_unbox(v_a_3147_);
lean_dec(v_a_3147_);
v___x_3149_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(v___x_3148_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v_a_3150_ = lean_ctor_get(v___x_3149_, 0);
lean_inc(v_a_3150_);
lean_dec_ref(v___x_3149_);
v___x_3151_ = lean_unbox(v_a_3150_);
lean_dec(v_a_3150_);
lean_inc(v_goal_2600_);
v___x_3152_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___lam__1(v_rs_2599_, v___x_2619_, v_goal_2600_, v___f_2617_, v___x_3151_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_, v_a_2608_, v_a_2609_);
v___y_2902_ = v___x_3143_;
v___y_2903_ = v_a_3130_;
v___y_2904_ = v___x_3152_;
goto v___jp_2901_;
}
else
{
v___y_3101_ = v___x_3143_;
v___y_3102_ = v___x_2636_;
v___y_3103_ = v_a_3130_;
v___y_3104_ = v___x_3132_;
goto v___jp_3100_;
}
}
else
{
v___y_3101_ = v___x_3143_;
v___y_3102_ = v___x_2636_;
v___y_3103_ = v_a_3130_;
v___y_3104_ = v___x_3132_;
goto v___jp_3100_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___boxed(lean_object* v_rs_3166_, lean_object* v_goal_3167_, lean_object* v_mvars_3168_, lean_object* v_preState_3169_, lean_object* v_a_3170_, lean_object* v_a_3171_, lean_object* v_a_3172_, lean_object* v_a_3173_, lean_object* v_a_3174_, lean_object* v_a_3175_, lean_object* v_a_3176_, lean_object* v_a_3177_){
_start:
{
lean_object* v_res_3178_; 
v_res_3178_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules(v_rs_3166_, v_goal_3167_, v_mvars_3168_, v_preState_3169_, v_a_3170_, v_a_3171_, v_a_3172_, v_a_3173_, v_a_3174_, v_a_3175_, v_a_3176_);
lean_dec(v_a_3176_);
lean_dec_ref(v_a_3175_);
lean_dec(v_a_3174_);
lean_dec_ref(v_a_3173_);
lean_dec(v_a_3172_);
lean_dec(v_a_3171_);
lean_dec_ref(v_a_3170_);
lean_dec_ref(v_preState_3169_);
return v_res_3178_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__1(void){
_start:
{
lean_object* v___x_3180_; lean_object* v___x_3181_; 
v___x_3180_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__0));
v___x_3181_ = l_Lean_stringToMessageData(v___x_3180_);
return v___x_3181_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2(lean_object* v_x_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_, lean_object* v___y_3188_, lean_object* v___y_3189_){
_start:
{
lean_object* v___x_3191_; lean_object* v___x_3192_; 
v___x_3191_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___closed__1);
v___x_3192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3192_, 0, v___x_3191_);
return v___x_3192_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2___boxed(lean_object* v_x_3193_, lean_object* v___y_3194_, lean_object* v___y_3195_, lean_object* v___y_3196_, lean_object* v___y_3197_, lean_object* v___y_3198_, lean_object* v___y_3199_, lean_object* v___y_3200_, lean_object* v___y_3201_){
_start:
{
lean_object* v_res_3202_; 
v_res_3202_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__2(v_x_3193_, v___y_3194_, v___y_3195_, v___y_3196_, v___y_3197_, v___y_3198_, v___y_3199_, v___y_3200_);
lean_dec(v___y_3200_);
lean_dec_ref(v___y_3199_);
lean_dec(v___y_3198_);
lean_dec_ref(v___y_3197_);
lean_dec(v___y_3196_);
lean_dec(v___y_3195_);
lean_dec_ref(v___y_3194_);
lean_dec_ref(v_x_3193_);
return v_res_3202_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__0(lean_object* v_x_3203_){
_start:
{
lean_object* v_name_3204_; uint8_t v___x_3205_; 
v_name_3204_ = lean_ctor_get(v_x_3203_, 0);
v___x_3205_ = lp_aesop_Aesop_isForwardOrDestructRuleName(v_name_3204_);
return v___x_3205_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__0___boxed(lean_object* v_x_3206_){
_start:
{
uint8_t v_res_3207_; lean_object* v_r_3208_; 
v_res_3207_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__0(v_x_3206_);
lean_dec_ref(v_x_3206_);
v_r_3208_ = lean_box(v_res_3207_);
return v_r_3208_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__1(lean_object* v_rs_3209_, lean_object* v___x_3210_, lean_object* v_goal_3211_, lean_object* v___f_3212_, uint8_t v_____do__lift_3213_, lean_object* v___y_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_, lean_object* v___y_3218_, lean_object* v___y_3219_, lean_object* v___y_3220_){
_start:
{
if (v_____do__lift_3213_ == 0)
{
lean_object* v___x_3222_; 
v___x_3222_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3209_, v___x_3210_, v_goal_3211_, v___f_3212_, v___y_3216_, v___y_3217_, v___y_3218_, v___y_3219_, v___y_3220_);
return v___x_3222_;
}
else
{
lean_object* v___x_3223_; lean_object* v___x_3224_; 
v___x_3223_ = lean_io_mono_nanos_now();
v___x_3224_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3209_, v___x_3210_, v_goal_3211_, v___f_3212_, v___y_3216_, v___y_3217_, v___y_3218_, v___y_3219_, v___y_3220_);
if (lean_obj_tag(v___x_3224_) == 0)
{
lean_object* v_a_3225_; lean_object* v___x_3227_; uint8_t v_isShared_3228_; uint8_t v_isSharedCheck_3263_; 
v_a_3225_ = lean_ctor_get(v___x_3224_, 0);
v_isSharedCheck_3263_ = !lean_is_exclusive(v___x_3224_);
if (v_isSharedCheck_3263_ == 0)
{
v___x_3227_ = v___x_3224_;
v_isShared_3228_ = v_isSharedCheck_3263_;
goto v_resetjp_3226_;
}
else
{
lean_inc(v_a_3225_);
lean_dec(v___x_3224_);
v___x_3227_ = lean_box(0);
v_isShared_3228_ = v_isSharedCheck_3263_;
goto v_resetjp_3226_;
}
v_resetjp_3226_:
{
lean_object* v___x_3229_; lean_object* v___x_3230_; lean_object* v_stats_3231_; lean_object* v_rulePatternCache_3232_; lean_object* v___x_3234_; uint8_t v_isShared_3235_; uint8_t v_isSharedCheck_3262_; 
v___x_3229_ = lean_io_mono_nanos_now();
v___x_3230_ = lean_st_ref_take(v___y_3216_);
v_stats_3231_ = lean_ctor_get(v___x_3230_, 1);
v_rulePatternCache_3232_ = lean_ctor_get(v___x_3230_, 0);
v_isSharedCheck_3262_ = !lean_is_exclusive(v___x_3230_);
if (v_isSharedCheck_3262_ == 0)
{
v___x_3234_ = v___x_3230_;
v_isShared_3235_ = v_isSharedCheck_3262_;
goto v_resetjp_3233_;
}
else
{
lean_inc(v_stats_3231_);
lean_inc(v_rulePatternCache_3232_);
lean_dec(v___x_3230_);
v___x_3234_ = lean_box(0);
v_isShared_3235_ = v_isSharedCheck_3262_;
goto v_resetjp_3233_;
}
v_resetjp_3233_:
{
lean_object* v_total_3236_; lean_object* v_configParsing_3237_; lean_object* v_ruleSetConstruction_3238_; lean_object* v_search_3239_; lean_object* v_ruleSelection_3240_; lean_object* v_script_3241_; lean_object* v_forwardState_3242_; lean_object* v_scriptGenerated_3243_; lean_object* v_ruleStats_3244_; lean_object* v_goalStats_3245_; lean_object* v___x_3247_; uint8_t v_isShared_3248_; uint8_t v_isSharedCheck_3261_; 
v_total_3236_ = lean_ctor_get(v_stats_3231_, 0);
v_configParsing_3237_ = lean_ctor_get(v_stats_3231_, 1);
v_ruleSetConstruction_3238_ = lean_ctor_get(v_stats_3231_, 2);
v_search_3239_ = lean_ctor_get(v_stats_3231_, 3);
v_ruleSelection_3240_ = lean_ctor_get(v_stats_3231_, 4);
v_script_3241_ = lean_ctor_get(v_stats_3231_, 5);
v_forwardState_3242_ = lean_ctor_get(v_stats_3231_, 6);
v_scriptGenerated_3243_ = lean_ctor_get(v_stats_3231_, 7);
v_ruleStats_3244_ = lean_ctor_get(v_stats_3231_, 8);
v_goalStats_3245_ = lean_ctor_get(v_stats_3231_, 9);
v_isSharedCheck_3261_ = !lean_is_exclusive(v_stats_3231_);
if (v_isSharedCheck_3261_ == 0)
{
v___x_3247_ = v_stats_3231_;
v_isShared_3248_ = v_isSharedCheck_3261_;
goto v_resetjp_3246_;
}
else
{
lean_inc(v_goalStats_3245_);
lean_inc(v_ruleStats_3244_);
lean_inc(v_scriptGenerated_3243_);
lean_inc(v_forwardState_3242_);
lean_inc(v_script_3241_);
lean_inc(v_ruleSelection_3240_);
lean_inc(v_search_3239_);
lean_inc(v_ruleSetConstruction_3238_);
lean_inc(v_configParsing_3237_);
lean_inc(v_total_3236_);
lean_dec(v_stats_3231_);
v___x_3247_ = lean_box(0);
v_isShared_3248_ = v_isSharedCheck_3261_;
goto v_resetjp_3246_;
}
v_resetjp_3246_:
{
lean_object* v___x_3249_; lean_object* v___x_3250_; lean_object* v___x_3252_; 
v___x_3249_ = lean_nat_sub(v___x_3229_, v___x_3223_);
lean_dec(v___x_3223_);
lean_dec(v___x_3229_);
v___x_3250_ = lean_nat_add(v_ruleSelection_3240_, v___x_3249_);
lean_dec(v___x_3249_);
lean_dec(v_ruleSelection_3240_);
if (v_isShared_3248_ == 0)
{
lean_ctor_set(v___x_3247_, 4, v___x_3250_);
v___x_3252_ = v___x_3247_;
goto v_reusejp_3251_;
}
else
{
lean_object* v_reuseFailAlloc_3260_; 
v_reuseFailAlloc_3260_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3260_, 0, v_total_3236_);
lean_ctor_set(v_reuseFailAlloc_3260_, 1, v_configParsing_3237_);
lean_ctor_set(v_reuseFailAlloc_3260_, 2, v_ruleSetConstruction_3238_);
lean_ctor_set(v_reuseFailAlloc_3260_, 3, v_search_3239_);
lean_ctor_set(v_reuseFailAlloc_3260_, 4, v___x_3250_);
lean_ctor_set(v_reuseFailAlloc_3260_, 5, v_script_3241_);
lean_ctor_set(v_reuseFailAlloc_3260_, 6, v_forwardState_3242_);
lean_ctor_set(v_reuseFailAlloc_3260_, 7, v_scriptGenerated_3243_);
lean_ctor_set(v_reuseFailAlloc_3260_, 8, v_ruleStats_3244_);
lean_ctor_set(v_reuseFailAlloc_3260_, 9, v_goalStats_3245_);
v___x_3252_ = v_reuseFailAlloc_3260_;
goto v_reusejp_3251_;
}
v_reusejp_3251_:
{
lean_object* v___x_3254_; 
if (v_isShared_3235_ == 0)
{
lean_ctor_set(v___x_3234_, 1, v___x_3252_);
v___x_3254_ = v___x_3234_;
goto v_reusejp_3253_;
}
else
{
lean_object* v_reuseFailAlloc_3259_; 
v_reuseFailAlloc_3259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3259_, 0, v_rulePatternCache_3232_);
lean_ctor_set(v_reuseFailAlloc_3259_, 1, v___x_3252_);
v___x_3254_ = v_reuseFailAlloc_3259_;
goto v_reusejp_3253_;
}
v_reusejp_3253_:
{
lean_object* v___x_3255_; lean_object* v___x_3257_; 
v___x_3255_ = lean_st_ref_set(v___y_3216_, v___x_3254_);
if (v_isShared_3228_ == 0)
{
v___x_3257_ = v___x_3227_;
goto v_reusejp_3256_;
}
else
{
lean_object* v_reuseFailAlloc_3258_; 
v_reuseFailAlloc_3258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3258_, 0, v_a_3225_);
v___x_3257_ = v_reuseFailAlloc_3258_;
goto v_reusejp_3256_;
}
v_reusejp_3256_:
{
return v___x_3257_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_3223_);
return v___x_3224_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__1___boxed(lean_object* v_rs_3264_, lean_object* v___x_3265_, lean_object* v_goal_3266_, lean_object* v___f_3267_, lean_object* v_____do__lift_3268_, lean_object* v___y_3269_, lean_object* v___y_3270_, lean_object* v___y_3271_, lean_object* v___y_3272_, lean_object* v___y_3273_, lean_object* v___y_3274_, lean_object* v___y_3275_, lean_object* v___y_3276_){
_start:
{
uint8_t v_____do__lift_118031__boxed_3277_; lean_object* v_res_3278_; 
v_____do__lift_118031__boxed_3277_ = lean_unbox(v_____do__lift_3268_);
v_res_3278_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__1(v_rs_3264_, v___x_3265_, v_goal_3266_, v___f_3267_, v_____do__lift_118031__boxed_3277_, v___y_3269_, v___y_3270_, v___y_3271_, v___y_3272_, v___y_3273_, v___y_3274_, v___y_3275_);
lean_dec(v___y_3275_);
lean_dec_ref(v___y_3274_);
lean_dec(v___y_3273_);
lean_dec_ref(v___y_3272_);
lean_dec(v___y_3271_);
lean_dec(v___y_3270_);
lean_dec_ref(v___y_3269_);
lean_dec_ref(v___x_3265_);
return v_res_3278_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0_spec__0(lean_object* v_e_3279_){
_start:
{
if (lean_obj_tag(v_e_3279_) == 0)
{
uint8_t v___x_3280_; 
v___x_3280_ = 2;
return v___x_3280_;
}
else
{
uint8_t v___x_3281_; 
v___x_3281_ = 0;
return v___x_3281_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0_spec__0___boxed(lean_object* v_e_3282_){
_start:
{
uint8_t v_res_3283_; lean_object* v_r_3284_; 
v_res_3283_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0_spec__0(v_e_3282_);
lean_dec_ref(v_e_3282_);
v_r_3284_ = lean_box(v_res_3283_);
return v_r_3284_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(lean_object* v_cls_3285_, uint8_t v_collapsed_3286_, lean_object* v_tag_3287_, lean_object* v_opts_3288_, uint8_t v_clsEnabled_3289_, lean_object* v_oldTraces_3290_, lean_object* v_msg_3291_, lean_object* v_resStartStop_3292_, lean_object* v___y_3293_, lean_object* v___y_3294_, lean_object* v___y_3295_, lean_object* v___y_3296_, lean_object* v___y_3297_, lean_object* v___y_3298_, lean_object* v___y_3299_){
_start:
{
lean_object* v_fst_3301_; lean_object* v_snd_3302_; lean_object* v___y_3304_; lean_object* v___y_3305_; lean_object* v_data_3306_; lean_object* v_fst_3317_; lean_object* v_snd_3318_; lean_object* v___x_3319_; uint8_t v___x_3320_; lean_object* v___y_3322_; lean_object* v_a_3323_; uint8_t v___y_3338_; double v___y_3369_; 
v_fst_3301_ = lean_ctor_get(v_resStartStop_3292_, 0);
lean_inc(v_fst_3301_);
v_snd_3302_ = lean_ctor_get(v_resStartStop_3292_, 1);
lean_inc(v_snd_3302_);
lean_dec_ref(v_resStartStop_3292_);
v_fst_3317_ = lean_ctor_get(v_snd_3302_, 0);
lean_inc(v_fst_3317_);
v_snd_3318_ = lean_ctor_get(v_snd_3302_, 1);
lean_inc(v_snd_3318_);
lean_dec(v_snd_3302_);
v___x_3319_ = l_Lean_trace_profiler;
v___x_3320_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_3288_, v___x_3319_);
if (v___x_3320_ == 0)
{
v___y_3338_ = v___x_3320_;
goto v___jp_3337_;
}
else
{
lean_object* v___x_3374_; uint8_t v___x_3375_; 
v___x_3374_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3375_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_3288_, v___x_3374_);
if (v___x_3375_ == 0)
{
lean_object* v___x_3376_; lean_object* v___x_3377_; double v___x_3378_; double v___x_3379_; double v___x_3380_; 
v___x_3376_ = l_Lean_trace_profiler_threshold;
v___x_3377_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_3288_, v___x_3376_);
v___x_3378_ = lean_float_of_nat(v___x_3377_);
v___x_3379_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__2);
v___x_3380_ = lean_float_div(v___x_3378_, v___x_3379_);
v___y_3369_ = v___x_3380_;
goto v___jp_3368_;
}
else
{
lean_object* v___x_3381_; lean_object* v___x_3382_; double v___x_3383_; 
v___x_3381_ = l_Lean_trace_profiler_threshold;
v___x_3382_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__11(v_opts_3288_, v___x_3381_);
v___x_3383_ = lean_float_of_nat(v___x_3382_);
v___y_3369_ = v___x_3383_;
goto v___jp_3368_;
}
}
v___jp_3303_:
{
lean_object* v___x_3307_; 
lean_inc(v___y_3305_);
v___x_3307_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__8___redArg(v_oldTraces_3290_, v_data_3306_, v___y_3305_, v___y_3304_, v___y_3296_, v___y_3297_, v___y_3298_, v___y_3299_);
if (lean_obj_tag(v___x_3307_) == 0)
{
lean_object* v___x_3308_; 
lean_dec_ref_known(v___x_3307_, 1);
v___x_3308_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_fst_3301_);
return v___x_3308_;
}
else
{
lean_object* v_a_3309_; lean_object* v___x_3311_; uint8_t v_isShared_3312_; uint8_t v_isSharedCheck_3316_; 
lean_dec(v_fst_3301_);
v_a_3309_ = lean_ctor_get(v___x_3307_, 0);
v_isSharedCheck_3316_ = !lean_is_exclusive(v___x_3307_);
if (v_isSharedCheck_3316_ == 0)
{
v___x_3311_ = v___x_3307_;
v_isShared_3312_ = v_isSharedCheck_3316_;
goto v_resetjp_3310_;
}
else
{
lean_inc(v_a_3309_);
lean_dec(v___x_3307_);
v___x_3311_ = lean_box(0);
v_isShared_3312_ = v_isSharedCheck_3316_;
goto v_resetjp_3310_;
}
v_resetjp_3310_:
{
lean_object* v___x_3314_; 
if (v_isShared_3312_ == 0)
{
v___x_3314_ = v___x_3311_;
goto v_reusejp_3313_;
}
else
{
lean_object* v_reuseFailAlloc_3315_; 
v_reuseFailAlloc_3315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3315_, 0, v_a_3309_);
v___x_3314_ = v_reuseFailAlloc_3315_;
goto v_reusejp_3313_;
}
v_reusejp_3313_:
{
return v___x_3314_;
}
}
}
}
v___jp_3321_:
{
uint8_t v_result_3324_; lean_object* v___x_3325_; lean_object* v___x_3326_; double v___x_3327_; lean_object* v_data_3328_; 
v_result_3324_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0_spec__0(v_fst_3301_);
v___x_3325_ = lean_box(v_result_3324_);
v___x_3326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3326_, 0, v___x_3325_);
v___x_3327_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg___closed__0);
lean_inc_ref(v_tag_3287_);
lean_inc_ref(v___x_3326_);
lean_inc(v_cls_3285_);
v_data_3328_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_3328_, 0, v_cls_3285_);
lean_ctor_set(v_data_3328_, 1, v___x_3326_);
lean_ctor_set(v_data_3328_, 2, v_tag_3287_);
lean_ctor_set_float(v_data_3328_, sizeof(void*)*3, v___x_3327_);
lean_ctor_set_float(v_data_3328_, sizeof(void*)*3 + 8, v___x_3327_);
lean_ctor_set_uint8(v_data_3328_, sizeof(void*)*3 + 16, v_collapsed_3286_);
if (v___x_3320_ == 0)
{
lean_dec_ref_known(v___x_3326_, 1);
lean_dec(v_snd_3318_);
lean_dec(v_fst_3317_);
lean_dec_ref(v_tag_3287_);
lean_dec(v_cls_3285_);
v___y_3304_ = v_a_3323_;
v___y_3305_ = v___y_3322_;
v_data_3306_ = v_data_3328_;
goto v___jp_3303_;
}
else
{
lean_object* v_data_3329_; double v___x_3330_; double v___x_3331_; 
lean_dec_ref_known(v_data_3328_, 3);
v_data_3329_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_3329_, 0, v_cls_3285_);
lean_ctor_set(v_data_3329_, 1, v___x_3326_);
lean_ctor_set(v_data_3329_, 2, v_tag_3287_);
v___x_3330_ = lean_unbox_float(v_fst_3317_);
lean_dec(v_fst_3317_);
lean_ctor_set_float(v_data_3329_, sizeof(void*)*3, v___x_3330_);
v___x_3331_ = lean_unbox_float(v_snd_3318_);
lean_dec(v_snd_3318_);
lean_ctor_set_float(v_data_3329_, sizeof(void*)*3 + 8, v___x_3331_);
lean_ctor_set_uint8(v_data_3329_, sizeof(void*)*3 + 16, v_collapsed_3286_);
v___y_3304_ = v_a_3323_;
v___y_3305_ = v___y_3322_;
v_data_3306_ = v_data_3329_;
goto v___jp_3303_;
}
}
v___jp_3332_:
{
lean_object* v_ref_3333_; lean_object* v___x_3334_; 
v_ref_3333_ = lean_ctor_get(v___y_3298_, 5);
lean_inc(v___y_3299_);
lean_inc_ref(v___y_3298_);
lean_inc(v___y_3297_);
lean_inc_ref(v___y_3296_);
lean_inc(v___y_3295_);
lean_inc(v___y_3294_);
lean_inc_ref(v___y_3293_);
lean_inc(v_fst_3301_);
v___x_3334_ = lean_apply_9(v_msg_3291_, v_fst_3301_, v___y_3293_, v___y_3294_, v___y_3295_, v___y_3296_, v___y_3297_, v___y_3298_, v___y_3299_, lean_box(0));
if (lean_obj_tag(v___x_3334_) == 0)
{
lean_object* v_a_3335_; 
v_a_3335_ = lean_ctor_get(v___x_3334_, 0);
lean_inc(v_a_3335_);
lean_dec_ref_known(v___x_3334_, 1);
v___y_3322_ = v_ref_3333_;
v_a_3323_ = v_a_3335_;
goto v___jp_3321_;
}
else
{
lean_object* v___x_3336_; 
lean_dec_ref_known(v___x_3334_, 1);
v___x_3336_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6___closed__1);
v___y_3322_ = v_ref_3333_;
v_a_3323_ = v___x_3336_;
goto v___jp_3321_;
}
}
v___jp_3337_:
{
if (v_clsEnabled_3289_ == 0)
{
if (v___y_3338_ == 0)
{
lean_object* v___x_3339_; lean_object* v_traceState_3340_; lean_object* v_env_3341_; lean_object* v_nextMacroScope_3342_; lean_object* v_ngen_3343_; lean_object* v_auxDeclNGen_3344_; lean_object* v_cache_3345_; lean_object* v_messages_3346_; lean_object* v_infoState_3347_; lean_object* v_snapshotTasks_3348_; lean_object* v___x_3350_; uint8_t v_isShared_3351_; uint8_t v_isSharedCheck_3367_; 
lean_dec(v_snd_3318_);
lean_dec(v_fst_3317_);
lean_dec_ref(v_msg_3291_);
lean_dec_ref(v_tag_3287_);
lean_dec(v_cls_3285_);
v___x_3339_ = lean_st_ref_take(v___y_3299_);
v_traceState_3340_ = lean_ctor_get(v___x_3339_, 4);
v_env_3341_ = lean_ctor_get(v___x_3339_, 0);
v_nextMacroScope_3342_ = lean_ctor_get(v___x_3339_, 1);
v_ngen_3343_ = lean_ctor_get(v___x_3339_, 2);
v_auxDeclNGen_3344_ = lean_ctor_get(v___x_3339_, 3);
v_cache_3345_ = lean_ctor_get(v___x_3339_, 5);
v_messages_3346_ = lean_ctor_get(v___x_3339_, 6);
v_infoState_3347_ = lean_ctor_get(v___x_3339_, 7);
v_snapshotTasks_3348_ = lean_ctor_get(v___x_3339_, 8);
v_isSharedCheck_3367_ = !lean_is_exclusive(v___x_3339_);
if (v_isSharedCheck_3367_ == 0)
{
v___x_3350_ = v___x_3339_;
v_isShared_3351_ = v_isSharedCheck_3367_;
goto v_resetjp_3349_;
}
else
{
lean_inc(v_snapshotTasks_3348_);
lean_inc(v_infoState_3347_);
lean_inc(v_messages_3346_);
lean_inc(v_cache_3345_);
lean_inc(v_traceState_3340_);
lean_inc(v_auxDeclNGen_3344_);
lean_inc(v_ngen_3343_);
lean_inc(v_nextMacroScope_3342_);
lean_inc(v_env_3341_);
lean_dec(v___x_3339_);
v___x_3350_ = lean_box(0);
v_isShared_3351_ = v_isSharedCheck_3367_;
goto v_resetjp_3349_;
}
v_resetjp_3349_:
{
uint64_t v_tid_3352_; lean_object* v_traces_3353_; lean_object* v___x_3355_; uint8_t v_isShared_3356_; uint8_t v_isSharedCheck_3366_; 
v_tid_3352_ = lean_ctor_get_uint64(v_traceState_3340_, sizeof(void*)*1);
v_traces_3353_ = lean_ctor_get(v_traceState_3340_, 0);
v_isSharedCheck_3366_ = !lean_is_exclusive(v_traceState_3340_);
if (v_isSharedCheck_3366_ == 0)
{
v___x_3355_ = v_traceState_3340_;
v_isShared_3356_ = v_isSharedCheck_3366_;
goto v_resetjp_3354_;
}
else
{
lean_inc(v_traces_3353_);
lean_dec(v_traceState_3340_);
v___x_3355_ = lean_box(0);
v_isShared_3356_ = v_isSharedCheck_3366_;
goto v_resetjp_3354_;
}
v_resetjp_3354_:
{
lean_object* v___x_3357_; lean_object* v___x_3359_; 
v___x_3357_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_3290_, v_traces_3353_);
lean_dec_ref(v_traces_3353_);
if (v_isShared_3356_ == 0)
{
lean_ctor_set(v___x_3355_, 0, v___x_3357_);
v___x_3359_ = v___x_3355_;
goto v_reusejp_3358_;
}
else
{
lean_object* v_reuseFailAlloc_3365_; 
v_reuseFailAlloc_3365_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3365_, 0, v___x_3357_);
lean_ctor_set_uint64(v_reuseFailAlloc_3365_, sizeof(void*)*1, v_tid_3352_);
v___x_3359_ = v_reuseFailAlloc_3365_;
goto v_reusejp_3358_;
}
v_reusejp_3358_:
{
lean_object* v___x_3361_; 
if (v_isShared_3351_ == 0)
{
lean_ctor_set(v___x_3350_, 4, v___x_3359_);
v___x_3361_ = v___x_3350_;
goto v_reusejp_3360_;
}
else
{
lean_object* v_reuseFailAlloc_3364_; 
v_reuseFailAlloc_3364_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3364_, 0, v_env_3341_);
lean_ctor_set(v_reuseFailAlloc_3364_, 1, v_nextMacroScope_3342_);
lean_ctor_set(v_reuseFailAlloc_3364_, 2, v_ngen_3343_);
lean_ctor_set(v_reuseFailAlloc_3364_, 3, v_auxDeclNGen_3344_);
lean_ctor_set(v_reuseFailAlloc_3364_, 4, v___x_3359_);
lean_ctor_set(v_reuseFailAlloc_3364_, 5, v_cache_3345_);
lean_ctor_set(v_reuseFailAlloc_3364_, 6, v_messages_3346_);
lean_ctor_set(v_reuseFailAlloc_3364_, 7, v_infoState_3347_);
lean_ctor_set(v_reuseFailAlloc_3364_, 8, v_snapshotTasks_3348_);
v___x_3361_ = v_reuseFailAlloc_3364_;
goto v_reusejp_3360_;
}
v_reusejp_3360_:
{
lean_object* v___x_3362_; lean_object* v___x_3363_; 
v___x_3362_ = lean_st_ref_set(v___y_3299_, v___x_3361_);
v___x_3363_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__6_spec__9___redArg(v_fst_3301_);
return v___x_3363_;
}
}
}
}
}
else
{
goto v___jp_3332_;
}
}
else
{
goto v___jp_3332_;
}
}
v___jp_3368_:
{
double v___x_3370_; double v___x_3371_; double v___x_3372_; uint8_t v___x_3373_; 
v___x_3370_ = lean_unbox_float(v_snd_3318_);
v___x_3371_ = lean_unbox_float(v_fst_3317_);
v___x_3372_ = lean_float_sub(v___x_3370_, v___x_3371_);
v___x_3373_ = lean_float_decLt(v___y_3369_, v___x_3372_);
v___y_3338_ = v___x_3373_;
goto v___jp_3337_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0___boxed(lean_object* v_cls_3384_, lean_object* v_collapsed_3385_, lean_object* v_tag_3386_, lean_object* v_opts_3387_, lean_object* v_clsEnabled_3388_, lean_object* v_oldTraces_3389_, lean_object* v_msg_3390_, lean_object* v_resStartStop_3391_, lean_object* v___y_3392_, lean_object* v___y_3393_, lean_object* v___y_3394_, lean_object* v___y_3395_, lean_object* v___y_3396_, lean_object* v___y_3397_, lean_object* v___y_3398_, lean_object* v___y_3399_){
_start:
{
uint8_t v_collapsed_boxed_3400_; uint8_t v_clsEnabled_boxed_3401_; lean_object* v_res_3402_; 
v_collapsed_boxed_3400_ = lean_unbox(v_collapsed_3385_);
v_clsEnabled_boxed_3401_ = lean_unbox(v_clsEnabled_3388_);
v_res_3402_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(v_cls_3384_, v_collapsed_boxed_3400_, v_tag_3386_, v_opts_3387_, v_clsEnabled_boxed_3401_, v_oldTraces_3389_, v_msg_3390_, v_resStartStop_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_, v___y_3397_, v___y_3398_);
lean_dec(v___y_3398_);
lean_dec_ref(v___y_3397_);
lean_dec(v___y_3396_);
lean_dec_ref(v___y_3395_);
lean_dec(v___y_3394_);
lean_dec(v___y_3393_);
lean_dec_ref(v___y_3392_);
lean_dec_ref(v_opts_3387_);
return v_res_3402_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__3(lean_object* v___f_3403_, lean_object* v___f_3404_, lean_object* v___f_3405_, lean_object* v___x_3406_, uint8_t v___x_3407_, lean_object* v___x_3408_, lean_object* v___f_3409_, lean_object* v_rs_3410_, lean_object* v___x_3411_, lean_object* v_goal_3412_, lean_object* v___f_3413_, lean_object* v_opts_3414_, lean_object* v___y_3415_, lean_object* v___y_3416_, lean_object* v___y_3417_, lean_object* v___y_3418_, lean_object* v___y_3419_, lean_object* v___y_3420_, lean_object* v___y_3421_){
_start:
{
lean_object* v___y_3424_; lean_object* v___y_3425_; uint8_t v___y_3426_; lean_object* v_a_3427_; lean_object* v___y_3437_; lean_object* v___y_3438_; uint8_t v___y_3439_; lean_object* v_a_3440_; lean_object* v___y_3443_; lean_object* v___y_3444_; uint8_t v___y_3445_; lean_object* v_a_3446_; lean_object* v___y_3449_; lean_object* v___y_3450_; uint8_t v___y_3451_; lean_object* v___y_3456_; lean_object* v___y_3457_; uint8_t v___y_3458_; lean_object* v___y_3495_; lean_object* v___y_3496_; uint8_t v___y_3497_; uint8_t v_a_3498_; lean_object* v___y_3500_; lean_object* v___y_3501_; uint8_t v___y_3502_; lean_object* v___y_3503_; lean_object* v___y_3507_; lean_object* v___y_3508_; uint8_t v___y_3509_; lean_object* v_a_3510_; lean_object* v___y_3523_; lean_object* v___y_3524_; uint8_t v___y_3525_; lean_object* v_a_3526_; lean_object* v___y_3529_; lean_object* v___y_3530_; uint8_t v___y_3531_; lean_object* v_a_3532_; lean_object* v___y_3535_; lean_object* v___y_3536_; uint8_t v___y_3537_; lean_object* v___y_3574_; lean_object* v___y_3575_; uint8_t v___y_3576_; lean_object* v___y_3581_; lean_object* v___y_3582_; uint8_t v___y_3583_; lean_object* v___y_3584_; uint8_t v_hasTrace_3587_; 
v_hasTrace_3587_ = lean_ctor_get_uint8(v_opts_3414_, sizeof(void*)*1);
if (v_hasTrace_3587_ == 0)
{
lean_object* v___x_3588_; 
lean_dec_ref(v___f_3413_);
lean_dec(v_goal_3412_);
lean_dec_ref(v_rs_3410_);
lean_dec_ref(v___f_3409_);
lean_dec_ref(v___x_3408_);
lean_dec(v___x_3406_);
lean_inc(v___y_3421_);
lean_inc_ref(v___y_3420_);
lean_inc(v___y_3419_);
lean_inc_ref(v___y_3418_);
lean_inc(v___y_3417_);
lean_inc(v___y_3416_);
lean_inc_ref(v___y_3415_);
v___x_3588_ = lean_apply_8(v___f_3403_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_, lean_box(0));
if (lean_obj_tag(v___x_3588_) == 0)
{
lean_object* v_a_3589_; lean_object* v___x_3590_; 
v_a_3589_ = lean_ctor_get(v___x_3588_, 0);
lean_inc(v_a_3589_);
lean_dec_ref_known(v___x_3588_, 1);
lean_inc(v___y_3421_);
lean_inc_ref(v___y_3420_);
lean_inc(v___y_3419_);
lean_inc_ref(v___y_3418_);
lean_inc(v___y_3417_);
lean_inc(v___y_3416_);
lean_inc_ref(v___y_3415_);
v___x_3590_ = lean_apply_9(v___f_3404_, v_a_3589_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_, lean_box(0));
if (lean_obj_tag(v___x_3590_) == 0)
{
lean_object* v_a_3591_; lean_object* v___x_3592_; 
v_a_3591_ = lean_ctor_get(v___x_3590_, 0);
lean_inc(v_a_3591_);
lean_dec_ref_known(v___x_3590_, 1);
lean_inc(v___y_3421_);
lean_inc_ref(v___y_3420_);
lean_inc(v___y_3419_);
lean_inc_ref(v___y_3418_);
lean_inc(v___y_3417_);
lean_inc(v___y_3416_);
lean_inc_ref(v___y_3415_);
v___x_3592_ = lean_apply_9(v___f_3405_, v_a_3591_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_, lean_box(0));
return v___x_3592_;
}
else
{
lean_object* v_a_3593_; lean_object* v___x_3595_; uint8_t v_isShared_3596_; uint8_t v_isSharedCheck_3600_; 
lean_dec_ref(v___f_3405_);
v_a_3593_ = lean_ctor_get(v___x_3590_, 0);
v_isSharedCheck_3600_ = !lean_is_exclusive(v___x_3590_);
if (v_isSharedCheck_3600_ == 0)
{
v___x_3595_ = v___x_3590_;
v_isShared_3596_ = v_isSharedCheck_3600_;
goto v_resetjp_3594_;
}
else
{
lean_inc(v_a_3593_);
lean_dec(v___x_3590_);
v___x_3595_ = lean_box(0);
v_isShared_3596_ = v_isSharedCheck_3600_;
goto v_resetjp_3594_;
}
v_resetjp_3594_:
{
lean_object* v___x_3598_; 
if (v_isShared_3596_ == 0)
{
v___x_3598_ = v___x_3595_;
goto v_reusejp_3597_;
}
else
{
lean_object* v_reuseFailAlloc_3599_; 
v_reuseFailAlloc_3599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3599_, 0, v_a_3593_);
v___x_3598_ = v_reuseFailAlloc_3599_;
goto v_reusejp_3597_;
}
v_reusejp_3597_:
{
return v___x_3598_;
}
}
}
}
else
{
lean_object* v_a_3601_; lean_object* v___x_3603_; uint8_t v_isShared_3604_; uint8_t v_isSharedCheck_3608_; 
lean_dec_ref(v___f_3405_);
lean_dec_ref(v___f_3404_);
v_a_3601_ = lean_ctor_get(v___x_3588_, 0);
v_isSharedCheck_3608_ = !lean_is_exclusive(v___x_3588_);
if (v_isSharedCheck_3608_ == 0)
{
v___x_3603_ = v___x_3588_;
v_isShared_3604_ = v_isSharedCheck_3608_;
goto v_resetjp_3602_;
}
else
{
lean_inc(v_a_3601_);
lean_dec(v___x_3588_);
v___x_3603_ = lean_box(0);
v_isShared_3604_ = v_isSharedCheck_3608_;
goto v_resetjp_3602_;
}
v_resetjp_3602_:
{
lean_object* v___x_3606_; 
if (v_isShared_3604_ == 0)
{
v___x_3606_ = v___x_3603_;
goto v_reusejp_3605_;
}
else
{
lean_object* v_reuseFailAlloc_3607_; 
v_reuseFailAlloc_3607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3607_, 0, v_a_3601_);
v___x_3606_ = v_reuseFailAlloc_3607_;
goto v_reusejp_3605_;
}
v_reusejp_3605_:
{
return v___x_3606_;
}
}
}
}
else
{
lean_object* v_options_3609_; lean_object* v_inheritedTraceOptions_3610_; uint8_t v___y_3612_; uint8_t v_a_3638_; uint8_t v_hasTrace_3662_; 
v_options_3609_ = lean_ctor_get(v___y_3420_, 2);
v_inheritedTraceOptions_3610_ = lean_ctor_get(v___y_3420_, 13);
v_hasTrace_3662_ = lean_ctor_get_uint8(v_options_3609_, sizeof(void*)*1);
if (v_hasTrace_3662_ == 0)
{
v_a_3638_ = v_hasTrace_3662_;
goto v___jp_3637_;
}
else
{
lean_object* v___x_3663_; lean_object* v___x_3664_; uint8_t v___x_3665_; 
v___x_3663_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1));
lean_inc(v___x_3406_);
v___x_3664_ = l_Lean_Name_append(v___x_3663_, v___x_3406_);
v___x_3665_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3610_, v_options_3609_, v___x_3664_);
lean_dec(v___x_3664_);
if (v___x_3665_ == 0)
{
v_a_3638_ = v___x_3665_;
goto v___jp_3637_;
}
else
{
lean_dec_ref(v___f_3405_);
lean_dec_ref(v___f_3404_);
lean_dec_ref(v___f_3403_);
v___y_3612_ = v___x_3665_;
goto v___jp_3611_;
}
}
v___jp_3611_:
{
lean_object* v___x_3613_; lean_object* v_a_3614_; lean_object* v___x_3615_; uint8_t v___x_3616_; 
v___x_3613_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v___y_3421_);
v_a_3614_ = lean_ctor_get(v___x_3613_, 0);
lean_inc(v_a_3614_);
lean_dec_ref(v___x_3613_);
v___x_3615_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3616_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_3414_, v___x_3615_);
if (v___x_3616_ == 0)
{
lean_object* v___x_3617_; lean_object* v___x_3618_; uint8_t v___x_3619_; 
v___x_3617_ = lean_io_mono_nanos_now();
v___x_3618_ = lp_aesop_Aesop_aesop_collectStats;
v___x_3619_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3609_, v___x_3618_);
if (v___x_3619_ == 0)
{
lean_object* v___x_3620_; lean_object* v___x_3621_; lean_object* v_a_3622_; uint8_t v___x_3623_; 
v___x_3620_ = lp_aesop_Aesop_TraceOption_stats;
v___x_3621_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_3620_, v___y_3420_);
v_a_3622_ = lean_ctor_get(v___x_3621_, 0);
lean_inc(v_a_3622_);
v___x_3623_ = lean_unbox(v_a_3622_);
lean_dec(v_a_3622_);
if (v___x_3623_ == 0)
{
lean_object* v___x_3624_; lean_object* v___x_3625_; uint8_t v___x_3626_; 
lean_dec_ref(v___x_3621_);
v___x_3624_ = lp_aesop_Aesop_aesop_stats_file;
v___x_3625_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_3609_, v___x_3624_);
v___x_3626_ = lean_string_dec_eq(v___x_3625_, v___x_3408_);
lean_dec_ref(v___x_3625_);
if (v___x_3626_ == 0)
{
v___y_3535_ = v_a_3614_;
v___y_3536_ = v___x_3617_;
v___y_3537_ = v___y_3612_;
goto v___jp_3534_;
}
else
{
v___y_3574_ = v_a_3614_;
v___y_3575_ = v___x_3617_;
v___y_3576_ = v___y_3612_;
goto v___jp_3573_;
}
}
else
{
v___y_3581_ = v_a_3614_;
v___y_3582_ = v___x_3617_;
v___y_3583_ = v___y_3612_;
v___y_3584_ = v___x_3621_;
goto v___jp_3580_;
}
}
else
{
v___y_3535_ = v_a_3614_;
v___y_3536_ = v___x_3617_;
v___y_3537_ = v___y_3612_;
goto v___jp_3534_;
}
}
else
{
lean_object* v___x_3627_; lean_object* v___x_3628_; uint8_t v___x_3629_; 
v___x_3627_ = lean_io_get_num_heartbeats();
v___x_3628_ = lp_aesop_Aesop_aesop_collectStats;
v___x_3629_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3609_, v___x_3628_);
if (v___x_3629_ == 0)
{
lean_object* v___x_3630_; lean_object* v___x_3631_; lean_object* v_a_3632_; uint8_t v___x_3633_; 
v___x_3630_ = lp_aesop_Aesop_TraceOption_stats;
v___x_3631_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_3630_, v___y_3420_);
v_a_3632_ = lean_ctor_get(v___x_3631_, 0);
lean_inc(v_a_3632_);
v___x_3633_ = lean_unbox(v_a_3632_);
lean_dec(v_a_3632_);
if (v___x_3633_ == 0)
{
lean_object* v___x_3634_; lean_object* v___x_3635_; uint8_t v___x_3636_; 
lean_dec_ref(v___x_3631_);
v___x_3634_ = lp_aesop_Aesop_aesop_stats_file;
v___x_3635_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_3609_, v___x_3634_);
v___x_3636_ = lean_string_dec_eq(v___x_3635_, v___x_3408_);
lean_dec_ref(v___x_3635_);
if (v___x_3636_ == 0)
{
v___y_3495_ = v_a_3614_;
v___y_3496_ = v___x_3627_;
v___y_3497_ = v___y_3612_;
v_a_3498_ = v___x_3616_;
goto v___jp_3494_;
}
else
{
v___y_3449_ = v_a_3614_;
v___y_3450_ = v___x_3627_;
v___y_3451_ = v___y_3612_;
goto v___jp_3448_;
}
}
else
{
v___y_3500_ = v_a_3614_;
v___y_3501_ = v___x_3627_;
v___y_3502_ = v___y_3612_;
v___y_3503_ = v___x_3631_;
goto v___jp_3499_;
}
}
else
{
v___y_3456_ = v_a_3614_;
v___y_3457_ = v___x_3627_;
v___y_3458_ = v___y_3612_;
goto v___jp_3455_;
}
}
}
v___jp_3637_:
{
lean_object* v___x_3639_; uint8_t v___x_3640_; 
v___x_3639_ = l_Lean_trace_profiler;
v___x_3640_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_opts_3414_, v___x_3639_);
if (v___x_3640_ == 0)
{
lean_object* v___x_3641_; 
lean_dec_ref(v___f_3413_);
lean_dec(v_goal_3412_);
lean_dec_ref(v_rs_3410_);
lean_dec_ref(v___f_3409_);
lean_dec_ref(v___x_3408_);
lean_dec(v___x_3406_);
lean_inc(v___y_3421_);
lean_inc_ref(v___y_3420_);
lean_inc(v___y_3419_);
lean_inc_ref(v___y_3418_);
lean_inc(v___y_3417_);
lean_inc(v___y_3416_);
lean_inc_ref(v___y_3415_);
v___x_3641_ = lean_apply_8(v___f_3403_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_, lean_box(0));
if (lean_obj_tag(v___x_3641_) == 0)
{
lean_object* v_a_3642_; lean_object* v___x_3643_; 
v_a_3642_ = lean_ctor_get(v___x_3641_, 0);
lean_inc(v_a_3642_);
lean_dec_ref_known(v___x_3641_, 1);
lean_inc(v___y_3421_);
lean_inc_ref(v___y_3420_);
lean_inc(v___y_3419_);
lean_inc_ref(v___y_3418_);
lean_inc(v___y_3417_);
lean_inc(v___y_3416_);
lean_inc_ref(v___y_3415_);
v___x_3643_ = lean_apply_9(v___f_3404_, v_a_3642_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_, lean_box(0));
if (lean_obj_tag(v___x_3643_) == 0)
{
lean_object* v_a_3644_; lean_object* v___x_3645_; 
v_a_3644_ = lean_ctor_get(v___x_3643_, 0);
lean_inc(v_a_3644_);
lean_dec_ref_known(v___x_3643_, 1);
lean_inc(v___y_3421_);
lean_inc_ref(v___y_3420_);
lean_inc(v___y_3419_);
lean_inc_ref(v___y_3418_);
lean_inc(v___y_3417_);
lean_inc(v___y_3416_);
lean_inc_ref(v___y_3415_);
v___x_3645_ = lean_apply_9(v___f_3405_, v_a_3644_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_, lean_box(0));
return v___x_3645_;
}
else
{
lean_object* v_a_3646_; lean_object* v___x_3648_; uint8_t v_isShared_3649_; uint8_t v_isSharedCheck_3653_; 
lean_dec_ref(v___f_3405_);
v_a_3646_ = lean_ctor_get(v___x_3643_, 0);
v_isSharedCheck_3653_ = !lean_is_exclusive(v___x_3643_);
if (v_isSharedCheck_3653_ == 0)
{
v___x_3648_ = v___x_3643_;
v_isShared_3649_ = v_isSharedCheck_3653_;
goto v_resetjp_3647_;
}
else
{
lean_inc(v_a_3646_);
lean_dec(v___x_3643_);
v___x_3648_ = lean_box(0);
v_isShared_3649_ = v_isSharedCheck_3653_;
goto v_resetjp_3647_;
}
v_resetjp_3647_:
{
lean_object* v___x_3651_; 
if (v_isShared_3649_ == 0)
{
v___x_3651_ = v___x_3648_;
goto v_reusejp_3650_;
}
else
{
lean_object* v_reuseFailAlloc_3652_; 
v_reuseFailAlloc_3652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3652_, 0, v_a_3646_);
v___x_3651_ = v_reuseFailAlloc_3652_;
goto v_reusejp_3650_;
}
v_reusejp_3650_:
{
return v___x_3651_;
}
}
}
}
else
{
lean_object* v_a_3654_; lean_object* v___x_3656_; uint8_t v_isShared_3657_; uint8_t v_isSharedCheck_3661_; 
lean_dec_ref(v___f_3405_);
lean_dec_ref(v___f_3404_);
v_a_3654_ = lean_ctor_get(v___x_3641_, 0);
v_isSharedCheck_3661_ = !lean_is_exclusive(v___x_3641_);
if (v_isSharedCheck_3661_ == 0)
{
v___x_3656_ = v___x_3641_;
v_isShared_3657_ = v_isSharedCheck_3661_;
goto v_resetjp_3655_;
}
else
{
lean_inc(v_a_3654_);
lean_dec(v___x_3641_);
v___x_3656_ = lean_box(0);
v_isShared_3657_ = v_isSharedCheck_3661_;
goto v_resetjp_3655_;
}
v_resetjp_3655_:
{
lean_object* v___x_3659_; 
if (v_isShared_3657_ == 0)
{
v___x_3659_ = v___x_3656_;
goto v_reusejp_3658_;
}
else
{
lean_object* v_reuseFailAlloc_3660_; 
v_reuseFailAlloc_3660_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3660_, 0, v_a_3654_);
v___x_3659_ = v_reuseFailAlloc_3660_;
goto v_reusejp_3658_;
}
v_reusejp_3658_:
{
return v___x_3659_;
}
}
}
}
else
{
lean_dec_ref(v___f_3405_);
lean_dec_ref(v___f_3404_);
lean_dec_ref(v___f_3403_);
v___y_3612_ = v_a_3638_;
goto v___jp_3611_;
}
}
}
v___jp_3423_:
{
lean_object* v___x_3428_; double v___x_3429_; double v___x_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; lean_object* v___x_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; 
v___x_3428_ = lean_io_get_num_heartbeats();
v___x_3429_ = lean_float_of_nat(v___y_3425_);
v___x_3430_ = lean_float_of_nat(v___x_3428_);
v___x_3431_ = lean_box_float(v___x_3429_);
v___x_3432_ = lean_box_float(v___x_3430_);
v___x_3433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3433_, 0, v___x_3431_);
lean_ctor_set(v___x_3433_, 1, v___x_3432_);
v___x_3434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3434_, 0, v_a_3427_);
lean_ctor_set(v___x_3434_, 1, v___x_3433_);
v___x_3435_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(v___x_3406_, v___x_3407_, v___x_3408_, v_opts_3414_, v___y_3426_, v___y_3424_, v___f_3409_, v___x_3434_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
return v___x_3435_;
}
v___jp_3436_:
{
lean_object* v___x_3441_; 
v___x_3441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3441_, 0, v_a_3440_);
v___y_3424_ = v___y_3437_;
v___y_3425_ = v___y_3438_;
v___y_3426_ = v___y_3439_;
v_a_3427_ = v___x_3441_;
goto v___jp_3423_;
}
v___jp_3442_:
{
lean_object* v___x_3447_; 
v___x_3447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3447_, 0, v_a_3446_);
v___y_3424_ = v___y_3443_;
v___y_3425_ = v___y_3444_;
v___y_3426_ = v___y_3445_;
v_a_3427_ = v___x_3447_;
goto v___jp_3423_;
}
v___jp_3448_:
{
lean_object* v___x_3452_; 
v___x_3452_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3410_, v___x_3411_, v_goal_3412_, v___f_3413_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
if (lean_obj_tag(v___x_3452_) == 0)
{
lean_object* v_a_3453_; 
v_a_3453_ = lean_ctor_get(v___x_3452_, 0);
lean_inc(v_a_3453_);
lean_dec_ref_known(v___x_3452_, 1);
v___y_3437_ = v___y_3449_;
v___y_3438_ = v___y_3450_;
v___y_3439_ = v___y_3451_;
v_a_3440_ = v_a_3453_;
goto v___jp_3436_;
}
else
{
lean_object* v_a_3454_; 
v_a_3454_ = lean_ctor_get(v___x_3452_, 0);
lean_inc(v_a_3454_);
lean_dec_ref_known(v___x_3452_, 1);
v___y_3443_ = v___y_3449_;
v___y_3444_ = v___y_3450_;
v___y_3445_ = v___y_3451_;
v_a_3446_ = v_a_3454_;
goto v___jp_3442_;
}
}
v___jp_3455_:
{
lean_object* v___x_3459_; lean_object* v___x_3460_; 
v___x_3459_ = lean_io_mono_nanos_now();
v___x_3460_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3410_, v___x_3411_, v_goal_3412_, v___f_3413_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
if (lean_obj_tag(v___x_3460_) == 0)
{
lean_object* v_a_3461_; lean_object* v___x_3462_; lean_object* v___x_3463_; lean_object* v_stats_3464_; lean_object* v_rulePatternCache_3465_; lean_object* v___x_3467_; uint8_t v_isShared_3468_; uint8_t v_isSharedCheck_3492_; 
v_a_3461_ = lean_ctor_get(v___x_3460_, 0);
lean_inc(v_a_3461_);
lean_dec_ref_known(v___x_3460_, 1);
v___x_3462_ = lean_io_mono_nanos_now();
v___x_3463_ = lean_st_ref_take(v___y_3417_);
v_stats_3464_ = lean_ctor_get(v___x_3463_, 1);
v_rulePatternCache_3465_ = lean_ctor_get(v___x_3463_, 0);
v_isSharedCheck_3492_ = !lean_is_exclusive(v___x_3463_);
if (v_isSharedCheck_3492_ == 0)
{
v___x_3467_ = v___x_3463_;
v_isShared_3468_ = v_isSharedCheck_3492_;
goto v_resetjp_3466_;
}
else
{
lean_inc(v_stats_3464_);
lean_inc(v_rulePatternCache_3465_);
lean_dec(v___x_3463_);
v___x_3467_ = lean_box(0);
v_isShared_3468_ = v_isSharedCheck_3492_;
goto v_resetjp_3466_;
}
v_resetjp_3466_:
{
lean_object* v_total_3469_; lean_object* v_configParsing_3470_; lean_object* v_ruleSetConstruction_3471_; lean_object* v_search_3472_; lean_object* v_ruleSelection_3473_; lean_object* v_script_3474_; lean_object* v_forwardState_3475_; lean_object* v_scriptGenerated_3476_; lean_object* v_ruleStats_3477_; lean_object* v_goalStats_3478_; lean_object* v___x_3480_; uint8_t v_isShared_3481_; uint8_t v_isSharedCheck_3491_; 
v_total_3469_ = lean_ctor_get(v_stats_3464_, 0);
v_configParsing_3470_ = lean_ctor_get(v_stats_3464_, 1);
v_ruleSetConstruction_3471_ = lean_ctor_get(v_stats_3464_, 2);
v_search_3472_ = lean_ctor_get(v_stats_3464_, 3);
v_ruleSelection_3473_ = lean_ctor_get(v_stats_3464_, 4);
v_script_3474_ = lean_ctor_get(v_stats_3464_, 5);
v_forwardState_3475_ = lean_ctor_get(v_stats_3464_, 6);
v_scriptGenerated_3476_ = lean_ctor_get(v_stats_3464_, 7);
v_ruleStats_3477_ = lean_ctor_get(v_stats_3464_, 8);
v_goalStats_3478_ = lean_ctor_get(v_stats_3464_, 9);
v_isSharedCheck_3491_ = !lean_is_exclusive(v_stats_3464_);
if (v_isSharedCheck_3491_ == 0)
{
v___x_3480_ = v_stats_3464_;
v_isShared_3481_ = v_isSharedCheck_3491_;
goto v_resetjp_3479_;
}
else
{
lean_inc(v_goalStats_3478_);
lean_inc(v_ruleStats_3477_);
lean_inc(v_scriptGenerated_3476_);
lean_inc(v_forwardState_3475_);
lean_inc(v_script_3474_);
lean_inc(v_ruleSelection_3473_);
lean_inc(v_search_3472_);
lean_inc(v_ruleSetConstruction_3471_);
lean_inc(v_configParsing_3470_);
lean_inc(v_total_3469_);
lean_dec(v_stats_3464_);
v___x_3480_ = lean_box(0);
v_isShared_3481_ = v_isSharedCheck_3491_;
goto v_resetjp_3479_;
}
v_resetjp_3479_:
{
lean_object* v___x_3482_; lean_object* v___x_3483_; lean_object* v___x_3485_; 
v___x_3482_ = lean_nat_sub(v___x_3462_, v___x_3459_);
lean_dec(v___x_3459_);
lean_dec(v___x_3462_);
v___x_3483_ = lean_nat_add(v_ruleSelection_3473_, v___x_3482_);
lean_dec(v___x_3482_);
lean_dec(v_ruleSelection_3473_);
if (v_isShared_3481_ == 0)
{
lean_ctor_set(v___x_3480_, 4, v___x_3483_);
v___x_3485_ = v___x_3480_;
goto v_reusejp_3484_;
}
else
{
lean_object* v_reuseFailAlloc_3490_; 
v_reuseFailAlloc_3490_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3490_, 0, v_total_3469_);
lean_ctor_set(v_reuseFailAlloc_3490_, 1, v_configParsing_3470_);
lean_ctor_set(v_reuseFailAlloc_3490_, 2, v_ruleSetConstruction_3471_);
lean_ctor_set(v_reuseFailAlloc_3490_, 3, v_search_3472_);
lean_ctor_set(v_reuseFailAlloc_3490_, 4, v___x_3483_);
lean_ctor_set(v_reuseFailAlloc_3490_, 5, v_script_3474_);
lean_ctor_set(v_reuseFailAlloc_3490_, 6, v_forwardState_3475_);
lean_ctor_set(v_reuseFailAlloc_3490_, 7, v_scriptGenerated_3476_);
lean_ctor_set(v_reuseFailAlloc_3490_, 8, v_ruleStats_3477_);
lean_ctor_set(v_reuseFailAlloc_3490_, 9, v_goalStats_3478_);
v___x_3485_ = v_reuseFailAlloc_3490_;
goto v_reusejp_3484_;
}
v_reusejp_3484_:
{
lean_object* v___x_3487_; 
if (v_isShared_3468_ == 0)
{
lean_ctor_set(v___x_3467_, 1, v___x_3485_);
v___x_3487_ = v___x_3467_;
goto v_reusejp_3486_;
}
else
{
lean_object* v_reuseFailAlloc_3489_; 
v_reuseFailAlloc_3489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3489_, 0, v_rulePatternCache_3465_);
lean_ctor_set(v_reuseFailAlloc_3489_, 1, v___x_3485_);
v___x_3487_ = v_reuseFailAlloc_3489_;
goto v_reusejp_3486_;
}
v_reusejp_3486_:
{
lean_object* v___x_3488_; 
v___x_3488_ = lean_st_ref_set(v___y_3417_, v___x_3487_);
v___y_3437_ = v___y_3456_;
v___y_3438_ = v___y_3457_;
v___y_3439_ = v___y_3458_;
v_a_3440_ = v_a_3461_;
goto v___jp_3436_;
}
}
}
}
}
else
{
lean_object* v_a_3493_; 
lean_dec(v___x_3459_);
v_a_3493_ = lean_ctor_get(v___x_3460_, 0);
lean_inc(v_a_3493_);
lean_dec_ref_known(v___x_3460_, 1);
v___y_3443_ = v___y_3456_;
v___y_3444_ = v___y_3457_;
v___y_3445_ = v___y_3458_;
v_a_3446_ = v_a_3493_;
goto v___jp_3442_;
}
}
v___jp_3494_:
{
if (v_a_3498_ == 0)
{
v___y_3449_ = v___y_3495_;
v___y_3450_ = v___y_3496_;
v___y_3451_ = v___y_3497_;
goto v___jp_3448_;
}
else
{
v___y_3456_ = v___y_3495_;
v___y_3457_ = v___y_3496_;
v___y_3458_ = v___y_3497_;
goto v___jp_3455_;
}
}
v___jp_3499_:
{
lean_object* v_a_3504_; uint8_t v___x_3505_; 
v_a_3504_ = lean_ctor_get(v___y_3503_, 0);
lean_inc(v_a_3504_);
lean_dec_ref(v___y_3503_);
v___x_3505_ = lean_unbox(v_a_3504_);
lean_dec(v_a_3504_);
v___y_3495_ = v___y_3500_;
v___y_3496_ = v___y_3501_;
v___y_3497_ = v___y_3502_;
v_a_3498_ = v___x_3505_;
goto v___jp_3494_;
}
v___jp_3506_:
{
lean_object* v___x_3511_; double v___x_3512_; double v___x_3513_; double v___x_3514_; double v___x_3515_; double v___x_3516_; lean_object* v___x_3517_; lean_object* v___x_3518_; lean_object* v___x_3519_; lean_object* v___x_3520_; lean_object* v___x_3521_; 
v___x_3511_ = lean_io_mono_nanos_now();
v___x_3512_ = lean_float_of_nat(v___y_3508_);
v___x_3513_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_3514_ = lean_float_div(v___x_3512_, v___x_3513_);
v___x_3515_ = lean_float_of_nat(v___x_3511_);
v___x_3516_ = lean_float_div(v___x_3515_, v___x_3513_);
v___x_3517_ = lean_box_float(v___x_3514_);
v___x_3518_ = lean_box_float(v___x_3516_);
v___x_3519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3519_, 0, v___x_3517_);
lean_ctor_set(v___x_3519_, 1, v___x_3518_);
v___x_3520_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3520_, 0, v_a_3510_);
lean_ctor_set(v___x_3520_, 1, v___x_3519_);
v___x_3521_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(v___x_3406_, v___x_3407_, v___x_3408_, v_opts_3414_, v___y_3509_, v___y_3507_, v___f_3409_, v___x_3520_, v___y_3415_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
return v___x_3521_;
}
v___jp_3522_:
{
lean_object* v___x_3527_; 
v___x_3527_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3527_, 0, v_a_3526_);
v___y_3507_ = v___y_3523_;
v___y_3508_ = v___y_3524_;
v___y_3509_ = v___y_3525_;
v_a_3510_ = v___x_3527_;
goto v___jp_3506_;
}
v___jp_3528_:
{
lean_object* v___x_3533_; 
v___x_3533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3533_, 0, v_a_3532_);
v___y_3507_ = v___y_3529_;
v___y_3508_ = v___y_3530_;
v___y_3509_ = v___y_3531_;
v_a_3510_ = v___x_3533_;
goto v___jp_3506_;
}
v___jp_3534_:
{
lean_object* v___x_3538_; lean_object* v___x_3539_; 
v___x_3538_ = lean_io_mono_nanos_now();
v___x_3539_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3410_, v___x_3411_, v_goal_3412_, v___f_3413_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
if (lean_obj_tag(v___x_3539_) == 0)
{
lean_object* v_a_3540_; lean_object* v___x_3541_; lean_object* v___x_3542_; lean_object* v_stats_3543_; lean_object* v_rulePatternCache_3544_; lean_object* v___x_3546_; uint8_t v_isShared_3547_; uint8_t v_isSharedCheck_3571_; 
v_a_3540_ = lean_ctor_get(v___x_3539_, 0);
lean_inc(v_a_3540_);
lean_dec_ref_known(v___x_3539_, 1);
v___x_3541_ = lean_io_mono_nanos_now();
v___x_3542_ = lean_st_ref_take(v___y_3417_);
v_stats_3543_ = lean_ctor_get(v___x_3542_, 1);
v_rulePatternCache_3544_ = lean_ctor_get(v___x_3542_, 0);
v_isSharedCheck_3571_ = !lean_is_exclusive(v___x_3542_);
if (v_isSharedCheck_3571_ == 0)
{
v___x_3546_ = v___x_3542_;
v_isShared_3547_ = v_isSharedCheck_3571_;
goto v_resetjp_3545_;
}
else
{
lean_inc(v_stats_3543_);
lean_inc(v_rulePatternCache_3544_);
lean_dec(v___x_3542_);
v___x_3546_ = lean_box(0);
v_isShared_3547_ = v_isSharedCheck_3571_;
goto v_resetjp_3545_;
}
v_resetjp_3545_:
{
lean_object* v_total_3548_; lean_object* v_configParsing_3549_; lean_object* v_ruleSetConstruction_3550_; lean_object* v_search_3551_; lean_object* v_ruleSelection_3552_; lean_object* v_script_3553_; lean_object* v_forwardState_3554_; lean_object* v_scriptGenerated_3555_; lean_object* v_ruleStats_3556_; lean_object* v_goalStats_3557_; lean_object* v___x_3559_; uint8_t v_isShared_3560_; uint8_t v_isSharedCheck_3570_; 
v_total_3548_ = lean_ctor_get(v_stats_3543_, 0);
v_configParsing_3549_ = lean_ctor_get(v_stats_3543_, 1);
v_ruleSetConstruction_3550_ = lean_ctor_get(v_stats_3543_, 2);
v_search_3551_ = lean_ctor_get(v_stats_3543_, 3);
v_ruleSelection_3552_ = lean_ctor_get(v_stats_3543_, 4);
v_script_3553_ = lean_ctor_get(v_stats_3543_, 5);
v_forwardState_3554_ = lean_ctor_get(v_stats_3543_, 6);
v_scriptGenerated_3555_ = lean_ctor_get(v_stats_3543_, 7);
v_ruleStats_3556_ = lean_ctor_get(v_stats_3543_, 8);
v_goalStats_3557_ = lean_ctor_get(v_stats_3543_, 9);
v_isSharedCheck_3570_ = !lean_is_exclusive(v_stats_3543_);
if (v_isSharedCheck_3570_ == 0)
{
v___x_3559_ = v_stats_3543_;
v_isShared_3560_ = v_isSharedCheck_3570_;
goto v_resetjp_3558_;
}
else
{
lean_inc(v_goalStats_3557_);
lean_inc(v_ruleStats_3556_);
lean_inc(v_scriptGenerated_3555_);
lean_inc(v_forwardState_3554_);
lean_inc(v_script_3553_);
lean_inc(v_ruleSelection_3552_);
lean_inc(v_search_3551_);
lean_inc(v_ruleSetConstruction_3550_);
lean_inc(v_configParsing_3549_);
lean_inc(v_total_3548_);
lean_dec(v_stats_3543_);
v___x_3559_ = lean_box(0);
v_isShared_3560_ = v_isSharedCheck_3570_;
goto v_resetjp_3558_;
}
v_resetjp_3558_:
{
lean_object* v___x_3561_; lean_object* v___x_3562_; lean_object* v___x_3564_; 
v___x_3561_ = lean_nat_sub(v___x_3541_, v___x_3538_);
lean_dec(v___x_3538_);
lean_dec(v___x_3541_);
v___x_3562_ = lean_nat_add(v_ruleSelection_3552_, v___x_3561_);
lean_dec(v___x_3561_);
lean_dec(v_ruleSelection_3552_);
if (v_isShared_3560_ == 0)
{
lean_ctor_set(v___x_3559_, 4, v___x_3562_);
v___x_3564_ = v___x_3559_;
goto v_reusejp_3563_;
}
else
{
lean_object* v_reuseFailAlloc_3569_; 
v_reuseFailAlloc_3569_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3569_, 0, v_total_3548_);
lean_ctor_set(v_reuseFailAlloc_3569_, 1, v_configParsing_3549_);
lean_ctor_set(v_reuseFailAlloc_3569_, 2, v_ruleSetConstruction_3550_);
lean_ctor_set(v_reuseFailAlloc_3569_, 3, v_search_3551_);
lean_ctor_set(v_reuseFailAlloc_3569_, 4, v___x_3562_);
lean_ctor_set(v_reuseFailAlloc_3569_, 5, v_script_3553_);
lean_ctor_set(v_reuseFailAlloc_3569_, 6, v_forwardState_3554_);
lean_ctor_set(v_reuseFailAlloc_3569_, 7, v_scriptGenerated_3555_);
lean_ctor_set(v_reuseFailAlloc_3569_, 8, v_ruleStats_3556_);
lean_ctor_set(v_reuseFailAlloc_3569_, 9, v_goalStats_3557_);
v___x_3564_ = v_reuseFailAlloc_3569_;
goto v_reusejp_3563_;
}
v_reusejp_3563_:
{
lean_object* v___x_3566_; 
if (v_isShared_3547_ == 0)
{
lean_ctor_set(v___x_3546_, 1, v___x_3564_);
v___x_3566_ = v___x_3546_;
goto v_reusejp_3565_;
}
else
{
lean_object* v_reuseFailAlloc_3568_; 
v_reuseFailAlloc_3568_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3568_, 0, v_rulePatternCache_3544_);
lean_ctor_set(v_reuseFailAlloc_3568_, 1, v___x_3564_);
v___x_3566_ = v_reuseFailAlloc_3568_;
goto v_reusejp_3565_;
}
v_reusejp_3565_:
{
lean_object* v___x_3567_; 
v___x_3567_ = lean_st_ref_set(v___y_3417_, v___x_3566_);
v___y_3529_ = v___y_3535_;
v___y_3530_ = v___y_3536_;
v___y_3531_ = v___y_3537_;
v_a_3532_ = v_a_3540_;
goto v___jp_3528_;
}
}
}
}
}
else
{
lean_object* v_a_3572_; 
lean_dec(v___x_3538_);
v_a_3572_ = lean_ctor_get(v___x_3539_, 0);
lean_inc(v_a_3572_);
lean_dec_ref_known(v___x_3539_, 1);
v___y_3523_ = v___y_3535_;
v___y_3524_ = v___y_3536_;
v___y_3525_ = v___y_3537_;
v_a_3526_ = v_a_3572_;
goto v___jp_3522_;
}
}
v___jp_3573_:
{
lean_object* v___x_3577_; 
v___x_3577_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3410_, v___x_3411_, v_goal_3412_, v___f_3413_, v___y_3417_, v___y_3418_, v___y_3419_, v___y_3420_, v___y_3421_);
if (lean_obj_tag(v___x_3577_) == 0)
{
lean_object* v_a_3578_; 
v_a_3578_ = lean_ctor_get(v___x_3577_, 0);
lean_inc(v_a_3578_);
lean_dec_ref_known(v___x_3577_, 1);
v___y_3529_ = v___y_3574_;
v___y_3530_ = v___y_3575_;
v___y_3531_ = v___y_3576_;
v_a_3532_ = v_a_3578_;
goto v___jp_3528_;
}
else
{
lean_object* v_a_3579_; 
v_a_3579_ = lean_ctor_get(v___x_3577_, 0);
lean_inc(v_a_3579_);
lean_dec_ref_known(v___x_3577_, 1);
v___y_3523_ = v___y_3574_;
v___y_3524_ = v___y_3575_;
v___y_3525_ = v___y_3576_;
v_a_3526_ = v_a_3579_;
goto v___jp_3522_;
}
}
v___jp_3580_:
{
lean_object* v_a_3585_; uint8_t v___x_3586_; 
v_a_3585_ = lean_ctor_get(v___y_3584_, 0);
lean_inc(v_a_3585_);
lean_dec_ref(v___y_3584_);
v___x_3586_ = lean_unbox(v_a_3585_);
lean_dec(v_a_3585_);
if (v___x_3586_ == 0)
{
v___y_3574_ = v___y_3581_;
v___y_3575_ = v___y_3582_;
v___y_3576_ = v___y_3583_;
goto v___jp_3573_;
}
else
{
v___y_3535_ = v___y_3581_;
v___y_3536_ = v___y_3582_;
v___y_3537_ = v___y_3583_;
goto v___jp_3534_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__3___boxed(lean_object** _args){
lean_object* v___f_3666_ = _args[0];
lean_object* v___f_3667_ = _args[1];
lean_object* v___f_3668_ = _args[2];
lean_object* v___x_3669_ = _args[3];
lean_object* v___x_3670_ = _args[4];
lean_object* v___x_3671_ = _args[5];
lean_object* v___f_3672_ = _args[6];
lean_object* v_rs_3673_ = _args[7];
lean_object* v___x_3674_ = _args[8];
lean_object* v_goal_3675_ = _args[9];
lean_object* v___f_3676_ = _args[10];
lean_object* v_opts_3677_ = _args[11];
lean_object* v___y_3678_ = _args[12];
lean_object* v___y_3679_ = _args[13];
lean_object* v___y_3680_ = _args[14];
lean_object* v___y_3681_ = _args[15];
lean_object* v___y_3682_ = _args[16];
lean_object* v___y_3683_ = _args[17];
lean_object* v___y_3684_ = _args[18];
lean_object* v___y_3685_ = _args[19];
_start:
{
uint8_t v___x_118308__boxed_3686_; lean_object* v_res_3687_; 
v___x_118308__boxed_3686_ = lean_unbox(v___x_3670_);
v_res_3687_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__3(v___f_3666_, v___f_3667_, v___f_3668_, v___x_3669_, v___x_118308__boxed_3686_, v___x_3671_, v___f_3672_, v_rs_3673_, v___x_3674_, v_goal_3675_, v___f_3676_, v_opts_3677_, v___y_3678_, v___y_3679_, v___y_3680_, v___y_3681_, v___y_3682_, v___y_3683_, v___y_3684_);
lean_dec(v___y_3684_);
lean_dec_ref(v___y_3683_);
lean_dec(v___y_3682_);
lean_dec_ref(v___y_3681_);
lean_dec(v___y_3680_);
lean_dec(v___y_3679_);
lean_dec_ref(v___y_3678_);
lean_dec_ref(v_opts_3677_);
lean_dec_ref(v___x_3674_);
return v_res_3687_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__1(void){
_start:
{
lean_object* v___x_3689_; lean_object* v___x_3690_; 
v___x_3689_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__0));
v___x_3690_ = l_Lean_stringToMessageData(v___x_3689_);
return v___x_3690_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4(lean_object* v_x_3691_, lean_object* v___y_3692_, lean_object* v___y_3693_, lean_object* v___y_3694_, lean_object* v___y_3695_, lean_object* v___y_3696_, lean_object* v___y_3697_, lean_object* v___y_3698_){
_start:
{
lean_object* v___x_3700_; lean_object* v___x_3701_; 
v___x_3700_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___closed__1);
v___x_3701_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3701_, 0, v___x_3700_);
return v___x_3701_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4___boxed(lean_object* v_x_3702_, lean_object* v___y_3703_, lean_object* v___y_3704_, lean_object* v___y_3705_, lean_object* v___y_3706_, lean_object* v___y_3707_, lean_object* v___y_3708_, lean_object* v___y_3709_, lean_object* v___y_3710_){
_start:
{
lean_object* v_res_3711_; 
v_res_3711_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__4(v_x_3702_, v___y_3703_, v___y_3704_, v___y_3705_, v___y_3706_, v___y_3707_, v___y_3708_, v___y_3709_);
lean_dec(v___y_3709_);
lean_dec_ref(v___y_3708_);
lean_dec(v___y_3707_);
lean_dec_ref(v___y_3706_);
lean_dec(v___y_3705_);
lean_dec(v___y_3704_);
lean_dec_ref(v___y_3703_);
lean_dec_ref(v_x_3702_);
return v_res_3711_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules(lean_object* v_rs_3715_, lean_object* v_goal_3716_, lean_object* v_mvars_3717_, lean_object* v_preState_3718_, lean_object* v_a_3719_, lean_object* v_a_3720_, lean_object* v_a_3721_, lean_object* v_a_3722_, lean_object* v_a_3723_, lean_object* v_a_3724_, lean_object* v_a_3725_){
_start:
{
lean_object* v_options_3727_; lean_object* v_inheritedTraceOptions_3728_; uint8_t v_hasTrace_3729_; lean_object* v___f_3730_; lean_object* v___f_3731_; lean_object* v___f_3732_; lean_object* v___f_3733_; lean_object* v___x_3734_; lean_object* v___x_3735_; lean_object* v___f_3736_; uint8_t v___x_3737_; lean_object* v___x_3738_; 
v_options_3727_ = lean_ctor_get(v_a_3724_, 2);
v_inheritedTraceOptions_3728_ = lean_ctor_get(v_a_3724_, 13);
v_hasTrace_3729_ = lean_ctor_get_uint8(v_options_3727_, sizeof(void*)*1);
v___f_3730_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__0));
v___f_3731_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules___closed__1));
v___f_3732_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__0));
v___f_3733_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__1));
v___x_3734_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_3735_ = lp_aesop_Aesop_ForwardRuleMatches_empty;
lean_inc(v_goal_3716_);
lean_inc_ref(v_rs_3715_);
v___f_3736_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__1___boxed), 13, 4);
lean_closure_set(v___f_3736_, 0, v_rs_3715_);
lean_closure_set(v___f_3736_, 1, v___x_3735_);
lean_closure_set(v___f_3736_, 2, v_goal_3716_);
lean_closure_set(v___f_3736_, 3, v___f_3733_);
v___x_3737_ = 1;
v___x_3738_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
if (v_hasTrace_3729_ == 0)
{
lean_object* v___x_3739_; 
lean_inc(v_goal_3716_);
v___x_3739_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__3(v___f_3730_, v___f_3731_, v___f_3736_, v___x_3734_, v___x_3737_, v___x_3738_, v___f_3732_, v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_options_3727_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_3739_) == 0)
{
lean_object* v_a_3740_; lean_object* v___x_3741_; 
v_a_3740_ = lean_ctor_get(v___x_3739_, 0);
lean_inc(v_a_3740_);
lean_dec_ref_known(v___x_3739_, 1);
v___x_3741_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_3716_, v_mvars_3717_, v_preState_3718_, v_a_3740_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
lean_dec(v_a_3740_);
return v___x_3741_;
}
else
{
lean_object* v_a_3742_; lean_object* v___x_3744_; uint8_t v_isShared_3745_; uint8_t v_isSharedCheck_3749_; 
lean_dec_ref(v_mvars_3717_);
lean_dec(v_goal_3716_);
v_a_3742_ = lean_ctor_get(v___x_3739_, 0);
v_isSharedCheck_3749_ = !lean_is_exclusive(v___x_3739_);
if (v_isSharedCheck_3749_ == 0)
{
v___x_3744_ = v___x_3739_;
v_isShared_3745_ = v_isSharedCheck_3749_;
goto v_resetjp_3743_;
}
else
{
lean_inc(v_a_3742_);
lean_dec(v___x_3739_);
v___x_3744_ = lean_box(0);
v_isShared_3745_ = v_isSharedCheck_3749_;
goto v_resetjp_3743_;
}
v_resetjp_3743_:
{
lean_object* v___x_3747_; 
if (v_isShared_3745_ == 0)
{
v___x_3747_ = v___x_3744_;
goto v_reusejp_3746_;
}
else
{
lean_object* v_reuseFailAlloc_3748_; 
v_reuseFailAlloc_3748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3748_, 0, v_a_3742_);
v___x_3747_ = v_reuseFailAlloc_3748_;
goto v_reusejp_3746_;
}
v_reusejp_3746_:
{
return v___x_3747_;
}
}
}
}
else
{
lean_object* v___f_3750_; lean_object* v___x_3751_; uint8_t v___x_3752_; lean_object* v___y_3754_; lean_object* v___y_3755_; lean_object* v_a_3756_; lean_object* v___y_3769_; lean_object* v___y_3770_; lean_object* v_a_3771_; lean_object* v___y_3774_; lean_object* v___y_3775_; lean_object* v___y_3776_; lean_object* v___y_3790_; lean_object* v___y_3791_; lean_object* v___y_3792_; lean_object* v___y_3793_; uint8_t v___y_3794_; lean_object* v_a_3795_; lean_object* v___y_3808_; lean_object* v___y_3809_; lean_object* v___y_3810_; lean_object* v___y_3811_; uint8_t v___y_3812_; lean_object* v_a_3813_; lean_object* v___y_3816_; lean_object* v___y_3817_; lean_object* v___y_3818_; lean_object* v___y_3819_; uint8_t v___y_3820_; lean_object* v_a_3821_; lean_object* v___y_3824_; lean_object* v___y_3825_; lean_object* v___y_3826_; lean_object* v___y_3827_; uint8_t v___y_3828_; lean_object* v___y_3833_; lean_object* v___y_3834_; lean_object* v___y_3835_; lean_object* v___y_3836_; uint8_t v___y_3837_; lean_object* v___y_3874_; lean_object* v___y_3875_; lean_object* v___y_3876_; lean_object* v___y_3877_; uint8_t v___y_3878_; lean_object* v___y_3879_; lean_object* v___y_3883_; lean_object* v___y_3884_; lean_object* v___y_3885_; lean_object* v___y_3886_; uint8_t v___y_3887_; lean_object* v_a_3888_; lean_object* v___y_3898_; lean_object* v___y_3899_; lean_object* v___y_3900_; lean_object* v___y_3901_; uint8_t v___y_3902_; lean_object* v_a_3903_; lean_object* v___y_3906_; lean_object* v___y_3907_; lean_object* v___y_3908_; lean_object* v___y_3909_; uint8_t v___y_3910_; lean_object* v_a_3911_; lean_object* v___y_3914_; lean_object* v___y_3915_; lean_object* v___y_3916_; lean_object* v___y_3917_; uint8_t v___y_3918_; lean_object* v___y_3955_; lean_object* v___y_3956_; lean_object* v___y_3957_; lean_object* v___y_3958_; uint8_t v___y_3959_; lean_object* v___y_3964_; lean_object* v___y_3965_; lean_object* v___y_3966_; lean_object* v___y_3967_; uint8_t v___y_3968_; lean_object* v___y_3969_; lean_object* v___y_3973_; lean_object* v___y_3974_; uint8_t v___y_3975_; uint8_t v___y_3976_; lean_object* v___y_4001_; lean_object* v___y_4002_; lean_object* v_a_4003_; lean_object* v___y_4013_; lean_object* v___y_4014_; lean_object* v_a_4015_; lean_object* v___y_4018_; lean_object* v___y_4019_; lean_object* v___y_4020_; lean_object* v___y_4034_; uint8_t v___y_4035_; lean_object* v___y_4036_; lean_object* v___y_4037_; lean_object* v___y_4038_; lean_object* v_a_4039_; lean_object* v___y_4052_; lean_object* v___y_4053_; uint8_t v___y_4054_; lean_object* v___y_4055_; lean_object* v___y_4056_; lean_object* v_a_4057_; lean_object* v___y_4060_; lean_object* v___y_4061_; uint8_t v___y_4062_; lean_object* v___y_4063_; lean_object* v___y_4064_; lean_object* v_a_4065_; lean_object* v___y_4068_; uint8_t v___y_4069_; lean_object* v___y_4070_; lean_object* v___y_4071_; lean_object* v___y_4072_; lean_object* v___y_4109_; uint8_t v___y_4110_; lean_object* v___y_4111_; lean_object* v___y_4112_; lean_object* v___y_4113_; lean_object* v___y_4118_; lean_object* v___y_4119_; uint8_t v___y_4120_; lean_object* v___y_4121_; lean_object* v___y_4122_; lean_object* v___y_4123_; lean_object* v___y_4127_; uint8_t v___y_4128_; lean_object* v___y_4129_; lean_object* v___y_4130_; lean_object* v___y_4131_; lean_object* v_a_4132_; lean_object* v___y_4142_; uint8_t v___y_4143_; lean_object* v___y_4144_; lean_object* v___y_4145_; lean_object* v___y_4146_; lean_object* v_a_4147_; lean_object* v___y_4150_; uint8_t v___y_4151_; lean_object* v___y_4152_; lean_object* v___y_4153_; lean_object* v___y_4154_; lean_object* v_a_4155_; lean_object* v___y_4158_; uint8_t v___y_4159_; lean_object* v___y_4160_; lean_object* v___y_4161_; lean_object* v___y_4162_; lean_object* v___y_4199_; uint8_t v___y_4200_; lean_object* v___y_4201_; lean_object* v___y_4202_; lean_object* v___y_4203_; lean_object* v___y_4208_; uint8_t v___y_4209_; lean_object* v___y_4210_; lean_object* v___y_4211_; lean_object* v___y_4212_; lean_object* v___y_4213_; lean_object* v___y_4217_; uint8_t v___y_4218_; lean_object* v___y_4219_; uint8_t v___y_4220_; 
v___f_3750_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___closed__2));
v___x_3751_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0);
v___x_3752_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3728_, v_options_3727_, v___x_3751_);
if (v___x_3752_ == 0)
{
lean_object* v___x_4269_; uint8_t v___x_4270_; 
v___x_4269_ = l_Lean_trace_profiler;
v___x_4270_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3727_, v___x_4269_);
if (v___x_4270_ == 0)
{
lean_object* v___x_4271_; 
lean_inc(v_goal_3716_);
v___x_4271_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__3(v___f_3730_, v___f_3731_, v___f_3736_, v___x_3734_, v___x_3737_, v___x_3738_, v___f_3732_, v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_options_3727_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_4271_) == 0)
{
lean_object* v_a_4272_; lean_object* v___x_4273_; 
v_a_4272_ = lean_ctor_get(v___x_4271_, 0);
lean_inc(v_a_4272_);
lean_dec_ref_known(v___x_4271_, 1);
v___x_4273_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_3716_, v_mvars_3717_, v_preState_3718_, v_a_4272_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
lean_dec(v_a_4272_);
return v___x_4273_;
}
else
{
lean_object* v_a_4274_; lean_object* v___x_4276_; uint8_t v_isShared_4277_; uint8_t v_isSharedCheck_4281_; 
lean_dec_ref(v_mvars_3717_);
lean_dec(v_goal_3716_);
v_a_4274_ = lean_ctor_get(v___x_4271_, 0);
v_isSharedCheck_4281_ = !lean_is_exclusive(v___x_4271_);
if (v_isSharedCheck_4281_ == 0)
{
v___x_4276_ = v___x_4271_;
v_isShared_4277_ = v_isSharedCheck_4281_;
goto v_resetjp_4275_;
}
else
{
lean_inc(v_a_4274_);
lean_dec(v___x_4271_);
v___x_4276_ = lean_box(0);
v_isShared_4277_ = v_isSharedCheck_4281_;
goto v_resetjp_4275_;
}
v_resetjp_4275_:
{
lean_object* v___x_4279_; 
if (v_isShared_4277_ == 0)
{
v___x_4279_ = v___x_4276_;
goto v_reusejp_4278_;
}
else
{
lean_object* v_reuseFailAlloc_4280_; 
v_reuseFailAlloc_4280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4280_, 0, v_a_4274_);
v___x_4279_ = v_reuseFailAlloc_4280_;
goto v_reusejp_4278_;
}
v_reusejp_4278_:
{
return v___x_4279_;
}
}
}
}
else
{
lean_dec_ref(v___f_3736_);
goto v___jp_4244_;
}
}
else
{
lean_dec_ref(v___f_3736_);
goto v___jp_4244_;
}
v___jp_3753_:
{
lean_object* v___x_3757_; double v___x_3758_; double v___x_3759_; double v___x_3760_; double v___x_3761_; double v___x_3762_; lean_object* v___x_3763_; lean_object* v___x_3764_; lean_object* v___x_3765_; lean_object* v___x_3766_; lean_object* v___x_3767_; 
v___x_3757_ = lean_io_mono_nanos_now();
v___x_3758_ = lean_float_of_nat(v___y_3755_);
v___x_3759_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_3760_ = lean_float_div(v___x_3758_, v___x_3759_);
v___x_3761_ = lean_float_of_nat(v___x_3757_);
v___x_3762_ = lean_float_div(v___x_3761_, v___x_3759_);
v___x_3763_ = lean_box_float(v___x_3760_);
v___x_3764_ = lean_box_float(v___x_3762_);
v___x_3765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3765_, 0, v___x_3763_);
lean_ctor_set(v___x_3765_, 1, v___x_3764_);
v___x_3766_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3766_, 0, v_a_3756_);
lean_ctor_set(v___x_3766_, 1, v___x_3765_);
v___x_3767_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1(v___x_3734_, v___x_3737_, v___x_3738_, v_options_3727_, v___x_3752_, v___y_3754_, v___f_3750_, v___x_3766_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
return v___x_3767_;
}
v___jp_3768_:
{
lean_object* v___x_3772_; 
v___x_3772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3772_, 0, v_a_3771_);
v___y_3754_ = v___y_3770_;
v___y_3755_ = v___y_3769_;
v_a_3756_ = v___x_3772_;
goto v___jp_3753_;
}
v___jp_3773_:
{
if (lean_obj_tag(v___y_3776_) == 0)
{
lean_object* v_a_3777_; lean_object* v___x_3778_; 
v_a_3777_ = lean_ctor_get(v___y_3776_, 0);
lean_inc(v_a_3777_);
lean_dec_ref_known(v___y_3776_, 1);
v___x_3778_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_3716_, v_mvars_3717_, v_preState_3718_, v_a_3777_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
lean_dec(v_a_3777_);
if (lean_obj_tag(v___x_3778_) == 0)
{
lean_object* v_a_3779_; lean_object* v___x_3781_; uint8_t v_isShared_3782_; uint8_t v_isSharedCheck_3786_; 
v_a_3779_ = lean_ctor_get(v___x_3778_, 0);
v_isSharedCheck_3786_ = !lean_is_exclusive(v___x_3778_);
if (v_isSharedCheck_3786_ == 0)
{
v___x_3781_ = v___x_3778_;
v_isShared_3782_ = v_isSharedCheck_3786_;
goto v_resetjp_3780_;
}
else
{
lean_inc(v_a_3779_);
lean_dec(v___x_3778_);
v___x_3781_ = lean_box(0);
v_isShared_3782_ = v_isSharedCheck_3786_;
goto v_resetjp_3780_;
}
v_resetjp_3780_:
{
lean_object* v___x_3784_; 
if (v_isShared_3782_ == 0)
{
lean_ctor_set_tag(v___x_3781_, 1);
v___x_3784_ = v___x_3781_;
goto v_reusejp_3783_;
}
else
{
lean_object* v_reuseFailAlloc_3785_; 
v_reuseFailAlloc_3785_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3785_, 0, v_a_3779_);
v___x_3784_ = v_reuseFailAlloc_3785_;
goto v_reusejp_3783_;
}
v_reusejp_3783_:
{
v___y_3754_ = v___y_3775_;
v___y_3755_ = v___y_3774_;
v_a_3756_ = v___x_3784_;
goto v___jp_3753_;
}
}
}
else
{
lean_object* v_a_3787_; 
v_a_3787_ = lean_ctor_get(v___x_3778_, 0);
lean_inc(v_a_3787_);
lean_dec_ref_known(v___x_3778_, 1);
v___y_3769_ = v___y_3774_;
v___y_3770_ = v___y_3775_;
v_a_3771_ = v_a_3787_;
goto v___jp_3768_;
}
}
else
{
lean_object* v_a_3788_; 
lean_dec_ref(v_mvars_3717_);
lean_dec(v_goal_3716_);
v_a_3788_ = lean_ctor_get(v___y_3776_, 0);
lean_inc(v_a_3788_);
lean_dec_ref_known(v___y_3776_, 1);
v___y_3769_ = v___y_3774_;
v___y_3770_ = v___y_3775_;
v_a_3771_ = v_a_3788_;
goto v___jp_3768_;
}
}
v___jp_3789_:
{
lean_object* v___x_3796_; double v___x_3797_; double v___x_3798_; double v___x_3799_; double v___x_3800_; double v___x_3801_; lean_object* v___x_3802_; lean_object* v___x_3803_; lean_object* v___x_3804_; lean_object* v___x_3805_; lean_object* v___x_3806_; 
v___x_3796_ = lean_io_mono_nanos_now();
v___x_3797_ = lean_float_of_nat(v___y_3792_);
v___x_3798_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_3799_ = lean_float_div(v___x_3797_, v___x_3798_);
v___x_3800_ = lean_float_of_nat(v___x_3796_);
v___x_3801_ = lean_float_div(v___x_3800_, v___x_3798_);
v___x_3802_ = lean_box_float(v___x_3799_);
v___x_3803_ = lean_box_float(v___x_3801_);
v___x_3804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3804_, 0, v___x_3802_);
lean_ctor_set(v___x_3804_, 1, v___x_3803_);
v___x_3805_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3805_, 0, v_a_3795_);
lean_ctor_set(v___x_3805_, 1, v___x_3804_);
v___x_3806_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(v___x_3734_, v___x_3737_, v___x_3738_, v_options_3727_, v___y_3794_, v___y_3793_, v___f_3732_, v___x_3805_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v___y_3774_ = v___y_3791_;
v___y_3775_ = v___y_3790_;
v___y_3776_ = v___x_3806_;
goto v___jp_3773_;
}
v___jp_3807_:
{
lean_object* v___x_3814_; 
v___x_3814_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3814_, 0, v_a_3813_);
v___y_3790_ = v___y_3809_;
v___y_3791_ = v___y_3808_;
v___y_3792_ = v___y_3810_;
v___y_3793_ = v___y_3811_;
v___y_3794_ = v___y_3812_;
v_a_3795_ = v___x_3814_;
goto v___jp_3789_;
}
v___jp_3815_:
{
lean_object* v___x_3822_; 
v___x_3822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3822_, 0, v_a_3821_);
v___y_3790_ = v___y_3817_;
v___y_3791_ = v___y_3816_;
v___y_3792_ = v___y_3818_;
v___y_3793_ = v___y_3819_;
v___y_3794_ = v___y_3820_;
v_a_3795_ = v___x_3822_;
goto v___jp_3789_;
}
v___jp_3823_:
{
lean_object* v___x_3829_; 
lean_inc(v_goal_3716_);
v___x_3829_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_3829_) == 0)
{
lean_object* v_a_3830_; 
v_a_3830_ = lean_ctor_get(v___x_3829_, 0);
lean_inc(v_a_3830_);
lean_dec_ref_known(v___x_3829_, 1);
v___y_3816_ = v___y_3825_;
v___y_3817_ = v___y_3824_;
v___y_3818_ = v___y_3826_;
v___y_3819_ = v___y_3827_;
v___y_3820_ = v___y_3828_;
v_a_3821_ = v_a_3830_;
goto v___jp_3815_;
}
else
{
lean_object* v_a_3831_; 
v_a_3831_ = lean_ctor_get(v___x_3829_, 0);
lean_inc(v_a_3831_);
lean_dec_ref_known(v___x_3829_, 1);
v___y_3808_ = v___y_3825_;
v___y_3809_ = v___y_3824_;
v___y_3810_ = v___y_3826_;
v___y_3811_ = v___y_3827_;
v___y_3812_ = v___y_3828_;
v_a_3813_ = v_a_3831_;
goto v___jp_3807_;
}
}
v___jp_3832_:
{
lean_object* v___x_3838_; lean_object* v___x_3839_; 
v___x_3838_ = lean_io_mono_nanos_now();
lean_inc(v_goal_3716_);
v___x_3839_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_3839_) == 0)
{
lean_object* v_a_3840_; lean_object* v___x_3841_; lean_object* v___x_3842_; lean_object* v_stats_3843_; lean_object* v_rulePatternCache_3844_; lean_object* v___x_3846_; uint8_t v_isShared_3847_; uint8_t v_isSharedCheck_3871_; 
v_a_3840_ = lean_ctor_get(v___x_3839_, 0);
lean_inc(v_a_3840_);
lean_dec_ref_known(v___x_3839_, 1);
v___x_3841_ = lean_io_mono_nanos_now();
v___x_3842_ = lean_st_ref_take(v_a_3721_);
v_stats_3843_ = lean_ctor_get(v___x_3842_, 1);
v_rulePatternCache_3844_ = lean_ctor_get(v___x_3842_, 0);
v_isSharedCheck_3871_ = !lean_is_exclusive(v___x_3842_);
if (v_isSharedCheck_3871_ == 0)
{
v___x_3846_ = v___x_3842_;
v_isShared_3847_ = v_isSharedCheck_3871_;
goto v_resetjp_3845_;
}
else
{
lean_inc(v_stats_3843_);
lean_inc(v_rulePatternCache_3844_);
lean_dec(v___x_3842_);
v___x_3846_ = lean_box(0);
v_isShared_3847_ = v_isSharedCheck_3871_;
goto v_resetjp_3845_;
}
v_resetjp_3845_:
{
lean_object* v_total_3848_; lean_object* v_configParsing_3849_; lean_object* v_ruleSetConstruction_3850_; lean_object* v_search_3851_; lean_object* v_ruleSelection_3852_; lean_object* v_script_3853_; lean_object* v_forwardState_3854_; lean_object* v_scriptGenerated_3855_; lean_object* v_ruleStats_3856_; lean_object* v_goalStats_3857_; lean_object* v___x_3859_; uint8_t v_isShared_3860_; uint8_t v_isSharedCheck_3870_; 
v_total_3848_ = lean_ctor_get(v_stats_3843_, 0);
v_configParsing_3849_ = lean_ctor_get(v_stats_3843_, 1);
v_ruleSetConstruction_3850_ = lean_ctor_get(v_stats_3843_, 2);
v_search_3851_ = lean_ctor_get(v_stats_3843_, 3);
v_ruleSelection_3852_ = lean_ctor_get(v_stats_3843_, 4);
v_script_3853_ = lean_ctor_get(v_stats_3843_, 5);
v_forwardState_3854_ = lean_ctor_get(v_stats_3843_, 6);
v_scriptGenerated_3855_ = lean_ctor_get(v_stats_3843_, 7);
v_ruleStats_3856_ = lean_ctor_get(v_stats_3843_, 8);
v_goalStats_3857_ = lean_ctor_get(v_stats_3843_, 9);
v_isSharedCheck_3870_ = !lean_is_exclusive(v_stats_3843_);
if (v_isSharedCheck_3870_ == 0)
{
v___x_3859_ = v_stats_3843_;
v_isShared_3860_ = v_isSharedCheck_3870_;
goto v_resetjp_3858_;
}
else
{
lean_inc(v_goalStats_3857_);
lean_inc(v_ruleStats_3856_);
lean_inc(v_scriptGenerated_3855_);
lean_inc(v_forwardState_3854_);
lean_inc(v_script_3853_);
lean_inc(v_ruleSelection_3852_);
lean_inc(v_search_3851_);
lean_inc(v_ruleSetConstruction_3850_);
lean_inc(v_configParsing_3849_);
lean_inc(v_total_3848_);
lean_dec(v_stats_3843_);
v___x_3859_ = lean_box(0);
v_isShared_3860_ = v_isSharedCheck_3870_;
goto v_resetjp_3858_;
}
v_resetjp_3858_:
{
lean_object* v___x_3861_; lean_object* v___x_3862_; lean_object* v___x_3864_; 
v___x_3861_ = lean_nat_sub(v___x_3841_, v___x_3838_);
lean_dec(v___x_3838_);
lean_dec(v___x_3841_);
v___x_3862_ = lean_nat_add(v_ruleSelection_3852_, v___x_3861_);
lean_dec(v___x_3861_);
lean_dec(v_ruleSelection_3852_);
if (v_isShared_3860_ == 0)
{
lean_ctor_set(v___x_3859_, 4, v___x_3862_);
v___x_3864_ = v___x_3859_;
goto v_reusejp_3863_;
}
else
{
lean_object* v_reuseFailAlloc_3869_; 
v_reuseFailAlloc_3869_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3869_, 0, v_total_3848_);
lean_ctor_set(v_reuseFailAlloc_3869_, 1, v_configParsing_3849_);
lean_ctor_set(v_reuseFailAlloc_3869_, 2, v_ruleSetConstruction_3850_);
lean_ctor_set(v_reuseFailAlloc_3869_, 3, v_search_3851_);
lean_ctor_set(v_reuseFailAlloc_3869_, 4, v___x_3862_);
lean_ctor_set(v_reuseFailAlloc_3869_, 5, v_script_3853_);
lean_ctor_set(v_reuseFailAlloc_3869_, 6, v_forwardState_3854_);
lean_ctor_set(v_reuseFailAlloc_3869_, 7, v_scriptGenerated_3855_);
lean_ctor_set(v_reuseFailAlloc_3869_, 8, v_ruleStats_3856_);
lean_ctor_set(v_reuseFailAlloc_3869_, 9, v_goalStats_3857_);
v___x_3864_ = v_reuseFailAlloc_3869_;
goto v_reusejp_3863_;
}
v_reusejp_3863_:
{
lean_object* v___x_3866_; 
if (v_isShared_3847_ == 0)
{
lean_ctor_set(v___x_3846_, 1, v___x_3864_);
v___x_3866_ = v___x_3846_;
goto v_reusejp_3865_;
}
else
{
lean_object* v_reuseFailAlloc_3868_; 
v_reuseFailAlloc_3868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3868_, 0, v_rulePatternCache_3844_);
lean_ctor_set(v_reuseFailAlloc_3868_, 1, v___x_3864_);
v___x_3866_ = v_reuseFailAlloc_3868_;
goto v_reusejp_3865_;
}
v_reusejp_3865_:
{
lean_object* v___x_3867_; 
v___x_3867_ = lean_st_ref_set(v_a_3721_, v___x_3866_);
v___y_3816_ = v___y_3834_;
v___y_3817_ = v___y_3833_;
v___y_3818_ = v___y_3835_;
v___y_3819_ = v___y_3836_;
v___y_3820_ = v___y_3837_;
v_a_3821_ = v_a_3840_;
goto v___jp_3815_;
}
}
}
}
}
else
{
lean_object* v_a_3872_; 
lean_dec(v___x_3838_);
v_a_3872_ = lean_ctor_get(v___x_3839_, 0);
lean_inc(v_a_3872_);
lean_dec_ref_known(v___x_3839_, 1);
v___y_3808_ = v___y_3834_;
v___y_3809_ = v___y_3833_;
v___y_3810_ = v___y_3835_;
v___y_3811_ = v___y_3836_;
v___y_3812_ = v___y_3837_;
v_a_3813_ = v_a_3872_;
goto v___jp_3807_;
}
}
v___jp_3873_:
{
lean_object* v_a_3880_; uint8_t v___x_3881_; 
v_a_3880_ = lean_ctor_get(v___y_3879_, 0);
lean_inc(v_a_3880_);
lean_dec_ref(v___y_3879_);
v___x_3881_ = lean_unbox(v_a_3880_);
lean_dec(v_a_3880_);
if (v___x_3881_ == 0)
{
v___y_3824_ = v___y_3875_;
v___y_3825_ = v___y_3874_;
v___y_3826_ = v___y_3876_;
v___y_3827_ = v___y_3877_;
v___y_3828_ = v___y_3878_;
goto v___jp_3823_;
}
else
{
v___y_3833_ = v___y_3875_;
v___y_3834_ = v___y_3874_;
v___y_3835_ = v___y_3876_;
v___y_3836_ = v___y_3877_;
v___y_3837_ = v___y_3878_;
goto v___jp_3832_;
}
}
v___jp_3882_:
{
lean_object* v___x_3889_; double v___x_3890_; double v___x_3891_; lean_object* v___x_3892_; lean_object* v___x_3893_; lean_object* v___x_3894_; lean_object* v___x_3895_; lean_object* v___x_3896_; 
v___x_3889_ = lean_io_get_num_heartbeats();
v___x_3890_ = lean_float_of_nat(v___y_3885_);
v___x_3891_ = lean_float_of_nat(v___x_3889_);
v___x_3892_ = lean_box_float(v___x_3890_);
v___x_3893_ = lean_box_float(v___x_3891_);
v___x_3894_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3894_, 0, v___x_3892_);
lean_ctor_set(v___x_3894_, 1, v___x_3893_);
v___x_3895_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3895_, 0, v_a_3888_);
lean_ctor_set(v___x_3895_, 1, v___x_3894_);
v___x_3896_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(v___x_3734_, v___x_3737_, v___x_3738_, v_options_3727_, v___y_3887_, v___y_3886_, v___f_3732_, v___x_3895_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v___y_3774_ = v___y_3884_;
v___y_3775_ = v___y_3883_;
v___y_3776_ = v___x_3896_;
goto v___jp_3773_;
}
v___jp_3897_:
{
lean_object* v___x_3904_; 
v___x_3904_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3904_, 0, v_a_3903_);
v___y_3883_ = v___y_3899_;
v___y_3884_ = v___y_3898_;
v___y_3885_ = v___y_3900_;
v___y_3886_ = v___y_3901_;
v___y_3887_ = v___y_3902_;
v_a_3888_ = v___x_3904_;
goto v___jp_3882_;
}
v___jp_3905_:
{
lean_object* v___x_3912_; 
v___x_3912_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3912_, 0, v_a_3911_);
v___y_3883_ = v___y_3907_;
v___y_3884_ = v___y_3906_;
v___y_3885_ = v___y_3908_;
v___y_3886_ = v___y_3909_;
v___y_3887_ = v___y_3910_;
v_a_3888_ = v___x_3912_;
goto v___jp_3882_;
}
v___jp_3913_:
{
lean_object* v___x_3919_; lean_object* v___x_3920_; 
v___x_3919_ = lean_io_mono_nanos_now();
lean_inc(v_goal_3716_);
v___x_3920_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_3920_) == 0)
{
lean_object* v_a_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; lean_object* v_stats_3924_; lean_object* v_rulePatternCache_3925_; lean_object* v___x_3927_; uint8_t v_isShared_3928_; uint8_t v_isSharedCheck_3952_; 
v_a_3921_ = lean_ctor_get(v___x_3920_, 0);
lean_inc(v_a_3921_);
lean_dec_ref_known(v___x_3920_, 1);
v___x_3922_ = lean_io_mono_nanos_now();
v___x_3923_ = lean_st_ref_take(v_a_3721_);
v_stats_3924_ = lean_ctor_get(v___x_3923_, 1);
v_rulePatternCache_3925_ = lean_ctor_get(v___x_3923_, 0);
v_isSharedCheck_3952_ = !lean_is_exclusive(v___x_3923_);
if (v_isSharedCheck_3952_ == 0)
{
v___x_3927_ = v___x_3923_;
v_isShared_3928_ = v_isSharedCheck_3952_;
goto v_resetjp_3926_;
}
else
{
lean_inc(v_stats_3924_);
lean_inc(v_rulePatternCache_3925_);
lean_dec(v___x_3923_);
v___x_3927_ = lean_box(0);
v_isShared_3928_ = v_isSharedCheck_3952_;
goto v_resetjp_3926_;
}
v_resetjp_3926_:
{
lean_object* v_total_3929_; lean_object* v_configParsing_3930_; lean_object* v_ruleSetConstruction_3931_; lean_object* v_search_3932_; lean_object* v_ruleSelection_3933_; lean_object* v_script_3934_; lean_object* v_forwardState_3935_; lean_object* v_scriptGenerated_3936_; lean_object* v_ruleStats_3937_; lean_object* v_goalStats_3938_; lean_object* v___x_3940_; uint8_t v_isShared_3941_; uint8_t v_isSharedCheck_3951_; 
v_total_3929_ = lean_ctor_get(v_stats_3924_, 0);
v_configParsing_3930_ = lean_ctor_get(v_stats_3924_, 1);
v_ruleSetConstruction_3931_ = lean_ctor_get(v_stats_3924_, 2);
v_search_3932_ = lean_ctor_get(v_stats_3924_, 3);
v_ruleSelection_3933_ = lean_ctor_get(v_stats_3924_, 4);
v_script_3934_ = lean_ctor_get(v_stats_3924_, 5);
v_forwardState_3935_ = lean_ctor_get(v_stats_3924_, 6);
v_scriptGenerated_3936_ = lean_ctor_get(v_stats_3924_, 7);
v_ruleStats_3937_ = lean_ctor_get(v_stats_3924_, 8);
v_goalStats_3938_ = lean_ctor_get(v_stats_3924_, 9);
v_isSharedCheck_3951_ = !lean_is_exclusive(v_stats_3924_);
if (v_isSharedCheck_3951_ == 0)
{
v___x_3940_ = v_stats_3924_;
v_isShared_3941_ = v_isSharedCheck_3951_;
goto v_resetjp_3939_;
}
else
{
lean_inc(v_goalStats_3938_);
lean_inc(v_ruleStats_3937_);
lean_inc(v_scriptGenerated_3936_);
lean_inc(v_forwardState_3935_);
lean_inc(v_script_3934_);
lean_inc(v_ruleSelection_3933_);
lean_inc(v_search_3932_);
lean_inc(v_ruleSetConstruction_3931_);
lean_inc(v_configParsing_3930_);
lean_inc(v_total_3929_);
lean_dec(v_stats_3924_);
v___x_3940_ = lean_box(0);
v_isShared_3941_ = v_isSharedCheck_3951_;
goto v_resetjp_3939_;
}
v_resetjp_3939_:
{
lean_object* v___x_3942_; lean_object* v___x_3943_; lean_object* v___x_3945_; 
v___x_3942_ = lean_nat_sub(v___x_3922_, v___x_3919_);
lean_dec(v___x_3919_);
lean_dec(v___x_3922_);
v___x_3943_ = lean_nat_add(v_ruleSelection_3933_, v___x_3942_);
lean_dec(v___x_3942_);
lean_dec(v_ruleSelection_3933_);
if (v_isShared_3941_ == 0)
{
lean_ctor_set(v___x_3940_, 4, v___x_3943_);
v___x_3945_ = v___x_3940_;
goto v_reusejp_3944_;
}
else
{
lean_object* v_reuseFailAlloc_3950_; 
v_reuseFailAlloc_3950_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3950_, 0, v_total_3929_);
lean_ctor_set(v_reuseFailAlloc_3950_, 1, v_configParsing_3930_);
lean_ctor_set(v_reuseFailAlloc_3950_, 2, v_ruleSetConstruction_3931_);
lean_ctor_set(v_reuseFailAlloc_3950_, 3, v_search_3932_);
lean_ctor_set(v_reuseFailAlloc_3950_, 4, v___x_3943_);
lean_ctor_set(v_reuseFailAlloc_3950_, 5, v_script_3934_);
lean_ctor_set(v_reuseFailAlloc_3950_, 6, v_forwardState_3935_);
lean_ctor_set(v_reuseFailAlloc_3950_, 7, v_scriptGenerated_3936_);
lean_ctor_set(v_reuseFailAlloc_3950_, 8, v_ruleStats_3937_);
lean_ctor_set(v_reuseFailAlloc_3950_, 9, v_goalStats_3938_);
v___x_3945_ = v_reuseFailAlloc_3950_;
goto v_reusejp_3944_;
}
v_reusejp_3944_:
{
lean_object* v___x_3947_; 
if (v_isShared_3928_ == 0)
{
lean_ctor_set(v___x_3927_, 1, v___x_3945_);
v___x_3947_ = v___x_3927_;
goto v_reusejp_3946_;
}
else
{
lean_object* v_reuseFailAlloc_3949_; 
v_reuseFailAlloc_3949_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3949_, 0, v_rulePatternCache_3925_);
lean_ctor_set(v_reuseFailAlloc_3949_, 1, v___x_3945_);
v___x_3947_ = v_reuseFailAlloc_3949_;
goto v_reusejp_3946_;
}
v_reusejp_3946_:
{
lean_object* v___x_3948_; 
v___x_3948_ = lean_st_ref_set(v_a_3721_, v___x_3947_);
v___y_3898_ = v___y_3915_;
v___y_3899_ = v___y_3914_;
v___y_3900_ = v___y_3916_;
v___y_3901_ = v___y_3917_;
v___y_3902_ = v___y_3918_;
v_a_3903_ = v_a_3921_;
goto v___jp_3897_;
}
}
}
}
}
else
{
lean_object* v_a_3953_; 
lean_dec(v___x_3919_);
v_a_3953_ = lean_ctor_get(v___x_3920_, 0);
lean_inc(v_a_3953_);
lean_dec_ref_known(v___x_3920_, 1);
v___y_3906_ = v___y_3915_;
v___y_3907_ = v___y_3914_;
v___y_3908_ = v___y_3916_;
v___y_3909_ = v___y_3917_;
v___y_3910_ = v___y_3918_;
v_a_3911_ = v_a_3953_;
goto v___jp_3905_;
}
}
v___jp_3954_:
{
lean_object* v___x_3960_; 
lean_inc(v_goal_3716_);
v___x_3960_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_3960_) == 0)
{
lean_object* v_a_3961_; 
v_a_3961_ = lean_ctor_get(v___x_3960_, 0);
lean_inc(v_a_3961_);
lean_dec_ref_known(v___x_3960_, 1);
v___y_3898_ = v___y_3956_;
v___y_3899_ = v___y_3955_;
v___y_3900_ = v___y_3957_;
v___y_3901_ = v___y_3958_;
v___y_3902_ = v___y_3959_;
v_a_3903_ = v_a_3961_;
goto v___jp_3897_;
}
else
{
lean_object* v_a_3962_; 
v_a_3962_ = lean_ctor_get(v___x_3960_, 0);
lean_inc(v_a_3962_);
lean_dec_ref_known(v___x_3960_, 1);
v___y_3906_ = v___y_3956_;
v___y_3907_ = v___y_3955_;
v___y_3908_ = v___y_3957_;
v___y_3909_ = v___y_3958_;
v___y_3910_ = v___y_3959_;
v_a_3911_ = v_a_3962_;
goto v___jp_3905_;
}
}
v___jp_3963_:
{
lean_object* v_a_3970_; uint8_t v___x_3971_; 
v_a_3970_ = lean_ctor_get(v___y_3969_, 0);
lean_inc(v_a_3970_);
lean_dec_ref(v___y_3969_);
v___x_3971_ = lean_unbox(v_a_3970_);
lean_dec(v_a_3970_);
if (v___x_3971_ == 0)
{
v___y_3955_ = v___y_3964_;
v___y_3956_ = v___y_3965_;
v___y_3957_ = v___y_3966_;
v___y_3958_ = v___y_3967_;
v___y_3959_ = v___y_3968_;
goto v___jp_3954_;
}
else
{
v___y_3914_ = v___y_3964_;
v___y_3915_ = v___y_3965_;
v___y_3916_ = v___y_3966_;
v___y_3917_ = v___y_3967_;
v___y_3918_ = v___y_3968_;
goto v___jp_3913_;
}
}
v___jp_3972_:
{
lean_object* v___x_3977_; 
v___x_3977_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v_a_3725_);
if (v___y_3975_ == 0)
{
lean_object* v_a_3978_; lean_object* v___x_3979_; lean_object* v___x_3980_; uint8_t v___x_3981_; 
v_a_3978_ = lean_ctor_get(v___x_3977_, 0);
lean_inc(v_a_3978_);
lean_dec_ref(v___x_3977_);
v___x_3979_ = lean_io_mono_nanos_now();
v___x_3980_ = lp_aesop_Aesop_aesop_collectStats;
v___x_3981_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3727_, v___x_3980_);
if (v___x_3981_ == 0)
{
lean_object* v___x_3982_; lean_object* v___x_3983_; lean_object* v_a_3984_; uint8_t v___x_3985_; 
v___x_3982_ = lp_aesop_Aesop_TraceOption_stats;
v___x_3983_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_3982_, v_a_3724_);
v_a_3984_ = lean_ctor_get(v___x_3983_, 0);
lean_inc(v_a_3984_);
v___x_3985_ = lean_unbox(v_a_3984_);
lean_dec(v_a_3984_);
if (v___x_3985_ == 0)
{
lean_object* v___x_3986_; lean_object* v___x_3987_; uint8_t v___x_3988_; 
lean_dec_ref(v___x_3983_);
v___x_3986_ = lp_aesop_Aesop_aesop_stats_file;
v___x_3987_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_3727_, v___x_3986_);
v___x_3988_ = lean_string_dec_eq(v___x_3987_, v___x_3738_);
lean_dec_ref(v___x_3987_);
if (v___x_3988_ == 0)
{
v___y_3833_ = v___y_3974_;
v___y_3834_ = v___y_3973_;
v___y_3835_ = v___x_3979_;
v___y_3836_ = v_a_3978_;
v___y_3837_ = v___y_3976_;
goto v___jp_3832_;
}
else
{
v___y_3824_ = v___y_3974_;
v___y_3825_ = v___y_3973_;
v___y_3826_ = v___x_3979_;
v___y_3827_ = v_a_3978_;
v___y_3828_ = v___y_3976_;
goto v___jp_3823_;
}
}
else
{
v___y_3874_ = v___y_3973_;
v___y_3875_ = v___y_3974_;
v___y_3876_ = v___x_3979_;
v___y_3877_ = v_a_3978_;
v___y_3878_ = v___y_3976_;
v___y_3879_ = v___x_3983_;
goto v___jp_3873_;
}
}
else
{
v___y_3833_ = v___y_3974_;
v___y_3834_ = v___y_3973_;
v___y_3835_ = v___x_3979_;
v___y_3836_ = v_a_3978_;
v___y_3837_ = v___y_3976_;
goto v___jp_3832_;
}
}
else
{
lean_object* v_a_3989_; lean_object* v___x_3990_; lean_object* v___x_3991_; uint8_t v___x_3992_; 
v_a_3989_ = lean_ctor_get(v___x_3977_, 0);
lean_inc(v_a_3989_);
lean_dec_ref(v___x_3977_);
v___x_3990_ = lean_io_get_num_heartbeats();
v___x_3991_ = lp_aesop_Aesop_aesop_collectStats;
v___x_3992_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3727_, v___x_3991_);
if (v___x_3992_ == 0)
{
lean_object* v___x_3993_; lean_object* v___x_3994_; lean_object* v_a_3995_; uint8_t v___x_3996_; 
v___x_3993_ = lp_aesop_Aesop_TraceOption_stats;
v___x_3994_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_3993_, v_a_3724_);
v_a_3995_ = lean_ctor_get(v___x_3994_, 0);
lean_inc(v_a_3995_);
v___x_3996_ = lean_unbox(v_a_3995_);
lean_dec(v_a_3995_);
if (v___x_3996_ == 0)
{
lean_object* v___x_3997_; lean_object* v___x_3998_; uint8_t v___x_3999_; 
lean_dec_ref(v___x_3994_);
v___x_3997_ = lp_aesop_Aesop_aesop_stats_file;
v___x_3998_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_3727_, v___x_3997_);
v___x_3999_ = lean_string_dec_eq(v___x_3998_, v___x_3738_);
lean_dec_ref(v___x_3998_);
if (v___x_3999_ == 0)
{
v___y_3914_ = v___y_3974_;
v___y_3915_ = v___y_3973_;
v___y_3916_ = v___x_3990_;
v___y_3917_ = v_a_3989_;
v___y_3918_ = v___y_3976_;
goto v___jp_3913_;
}
else
{
v___y_3955_ = v___y_3974_;
v___y_3956_ = v___y_3973_;
v___y_3957_ = v___x_3990_;
v___y_3958_ = v_a_3989_;
v___y_3959_ = v___y_3976_;
goto v___jp_3954_;
}
}
else
{
v___y_3964_ = v___y_3974_;
v___y_3965_ = v___y_3973_;
v___y_3966_ = v___x_3990_;
v___y_3967_ = v_a_3989_;
v___y_3968_ = v___y_3976_;
v___y_3969_ = v___x_3994_;
goto v___jp_3963_;
}
}
else
{
v___y_3914_ = v___y_3974_;
v___y_3915_ = v___y_3973_;
v___y_3916_ = v___x_3990_;
v___y_3917_ = v_a_3989_;
v___y_3918_ = v___y_3976_;
goto v___jp_3913_;
}
}
}
v___jp_4000_:
{
lean_object* v___x_4004_; double v___x_4005_; double v___x_4006_; lean_object* v___x_4007_; lean_object* v___x_4008_; lean_object* v___x_4009_; lean_object* v___x_4010_; lean_object* v___x_4011_; 
v___x_4004_ = lean_io_get_num_heartbeats();
v___x_4005_ = lean_float_of_nat(v___y_4002_);
v___x_4006_ = lean_float_of_nat(v___x_4004_);
v___x_4007_ = lean_box_float(v___x_4005_);
v___x_4008_ = lean_box_float(v___x_4006_);
v___x_4009_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4009_, 0, v___x_4007_);
lean_ctor_set(v___x_4009_, 1, v___x_4008_);
v___x_4010_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4010_, 0, v_a_4003_);
lean_ctor_set(v___x_4010_, 1, v___x_4009_);
v___x_4011_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules_spec__1(v___x_3734_, v___x_3737_, v___x_3738_, v_options_3727_, v___x_3752_, v___y_4001_, v___f_3750_, v___x_4010_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
return v___x_4011_;
}
v___jp_4012_:
{
lean_object* v___x_4016_; 
v___x_4016_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4016_, 0, v_a_4015_);
v___y_4001_ = v___y_4013_;
v___y_4002_ = v___y_4014_;
v_a_4003_ = v___x_4016_;
goto v___jp_4000_;
}
v___jp_4017_:
{
if (lean_obj_tag(v___y_4020_) == 0)
{
lean_object* v_a_4021_; lean_object* v___x_4022_; 
v_a_4021_ = lean_ctor_get(v___y_4020_, 0);
lean_inc(v_a_4021_);
lean_dec_ref_known(v___y_4020_, 1);
v___x_4022_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runFirstRule___redArg(v_goal_3716_, v_mvars_3717_, v_preState_3718_, v_a_4021_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
lean_dec(v_a_4021_);
if (lean_obj_tag(v___x_4022_) == 0)
{
lean_object* v_a_4023_; lean_object* v___x_4025_; uint8_t v_isShared_4026_; uint8_t v_isSharedCheck_4030_; 
v_a_4023_ = lean_ctor_get(v___x_4022_, 0);
v_isSharedCheck_4030_ = !lean_is_exclusive(v___x_4022_);
if (v_isSharedCheck_4030_ == 0)
{
v___x_4025_ = v___x_4022_;
v_isShared_4026_ = v_isSharedCheck_4030_;
goto v_resetjp_4024_;
}
else
{
lean_inc(v_a_4023_);
lean_dec(v___x_4022_);
v___x_4025_ = lean_box(0);
v_isShared_4026_ = v_isSharedCheck_4030_;
goto v_resetjp_4024_;
}
v_resetjp_4024_:
{
lean_object* v___x_4028_; 
if (v_isShared_4026_ == 0)
{
lean_ctor_set_tag(v___x_4025_, 1);
v___x_4028_ = v___x_4025_;
goto v_reusejp_4027_;
}
else
{
lean_object* v_reuseFailAlloc_4029_; 
v_reuseFailAlloc_4029_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4029_, 0, v_a_4023_);
v___x_4028_ = v_reuseFailAlloc_4029_;
goto v_reusejp_4027_;
}
v_reusejp_4027_:
{
v___y_4001_ = v___y_4018_;
v___y_4002_ = v___y_4019_;
v_a_4003_ = v___x_4028_;
goto v___jp_4000_;
}
}
}
else
{
lean_object* v_a_4031_; 
v_a_4031_ = lean_ctor_get(v___x_4022_, 0);
lean_inc(v_a_4031_);
lean_dec_ref_known(v___x_4022_, 1);
v___y_4013_ = v___y_4018_;
v___y_4014_ = v___y_4019_;
v_a_4015_ = v_a_4031_;
goto v___jp_4012_;
}
}
else
{
lean_object* v_a_4032_; 
lean_dec_ref(v_mvars_3717_);
lean_dec(v_goal_3716_);
v_a_4032_ = lean_ctor_get(v___y_4020_, 0);
lean_inc(v_a_4032_);
lean_dec_ref_known(v___y_4020_, 1);
v___y_4013_ = v___y_4018_;
v___y_4014_ = v___y_4019_;
v_a_4015_ = v_a_4032_;
goto v___jp_4012_;
}
}
v___jp_4033_:
{
lean_object* v___x_4040_; double v___x_4041_; double v___x_4042_; double v___x_4043_; double v___x_4044_; double v___x_4045_; lean_object* v___x_4046_; lean_object* v___x_4047_; lean_object* v___x_4048_; lean_object* v___x_4049_; lean_object* v___x_4050_; 
v___x_4040_ = lean_io_mono_nanos_now();
v___x_4041_ = lean_float_of_nat(v___y_4036_);
v___x_4042_ = lean_float_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__1);
v___x_4043_ = lean_float_div(v___x_4041_, v___x_4042_);
v___x_4044_ = lean_float_of_nat(v___x_4040_);
v___x_4045_ = lean_float_div(v___x_4044_, v___x_4042_);
v___x_4046_ = lean_box_float(v___x_4043_);
v___x_4047_ = lean_box_float(v___x_4045_);
v___x_4048_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4048_, 0, v___x_4046_);
lean_ctor_set(v___x_4048_, 1, v___x_4047_);
v___x_4049_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4049_, 0, v_a_4039_);
lean_ctor_set(v___x_4049_, 1, v___x_4048_);
v___x_4050_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(v___x_3734_, v___x_3737_, v___x_3738_, v_options_3727_, v___y_4035_, v___y_4038_, v___f_3732_, v___x_4049_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v___y_4018_ = v___y_4034_;
v___y_4019_ = v___y_4037_;
v___y_4020_ = v___x_4050_;
goto v___jp_4017_;
}
v___jp_4051_:
{
lean_object* v___x_4058_; 
v___x_4058_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4058_, 0, v_a_4057_);
v___y_4034_ = v___y_4052_;
v___y_4035_ = v___y_4054_;
v___y_4036_ = v___y_4053_;
v___y_4037_ = v___y_4055_;
v___y_4038_ = v___y_4056_;
v_a_4039_ = v___x_4058_;
goto v___jp_4033_;
}
v___jp_4059_:
{
lean_object* v___x_4066_; 
v___x_4066_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4066_, 0, v_a_4065_);
v___y_4034_ = v___y_4060_;
v___y_4035_ = v___y_4062_;
v___y_4036_ = v___y_4061_;
v___y_4037_ = v___y_4063_;
v___y_4038_ = v___y_4064_;
v_a_4039_ = v___x_4066_;
goto v___jp_4033_;
}
v___jp_4067_:
{
lean_object* v___x_4073_; lean_object* v___x_4074_; 
v___x_4073_ = lean_io_mono_nanos_now();
lean_inc(v_goal_3716_);
v___x_4074_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_4074_) == 0)
{
lean_object* v_a_4075_; lean_object* v___x_4076_; lean_object* v___x_4077_; lean_object* v_stats_4078_; lean_object* v_rulePatternCache_4079_; lean_object* v___x_4081_; uint8_t v_isShared_4082_; uint8_t v_isSharedCheck_4106_; 
v_a_4075_ = lean_ctor_get(v___x_4074_, 0);
lean_inc(v_a_4075_);
lean_dec_ref_known(v___x_4074_, 1);
v___x_4076_ = lean_io_mono_nanos_now();
v___x_4077_ = lean_st_ref_take(v_a_3721_);
v_stats_4078_ = lean_ctor_get(v___x_4077_, 1);
v_rulePatternCache_4079_ = lean_ctor_get(v___x_4077_, 0);
v_isSharedCheck_4106_ = !lean_is_exclusive(v___x_4077_);
if (v_isSharedCheck_4106_ == 0)
{
v___x_4081_ = v___x_4077_;
v_isShared_4082_ = v_isSharedCheck_4106_;
goto v_resetjp_4080_;
}
else
{
lean_inc(v_stats_4078_);
lean_inc(v_rulePatternCache_4079_);
lean_dec(v___x_4077_);
v___x_4081_ = lean_box(0);
v_isShared_4082_ = v_isSharedCheck_4106_;
goto v_resetjp_4080_;
}
v_resetjp_4080_:
{
lean_object* v_total_4083_; lean_object* v_configParsing_4084_; lean_object* v_ruleSetConstruction_4085_; lean_object* v_search_4086_; lean_object* v_ruleSelection_4087_; lean_object* v_script_4088_; lean_object* v_forwardState_4089_; lean_object* v_scriptGenerated_4090_; lean_object* v_ruleStats_4091_; lean_object* v_goalStats_4092_; lean_object* v___x_4094_; uint8_t v_isShared_4095_; uint8_t v_isSharedCheck_4105_; 
v_total_4083_ = lean_ctor_get(v_stats_4078_, 0);
v_configParsing_4084_ = lean_ctor_get(v_stats_4078_, 1);
v_ruleSetConstruction_4085_ = lean_ctor_get(v_stats_4078_, 2);
v_search_4086_ = lean_ctor_get(v_stats_4078_, 3);
v_ruleSelection_4087_ = lean_ctor_get(v_stats_4078_, 4);
v_script_4088_ = lean_ctor_get(v_stats_4078_, 5);
v_forwardState_4089_ = lean_ctor_get(v_stats_4078_, 6);
v_scriptGenerated_4090_ = lean_ctor_get(v_stats_4078_, 7);
v_ruleStats_4091_ = lean_ctor_get(v_stats_4078_, 8);
v_goalStats_4092_ = lean_ctor_get(v_stats_4078_, 9);
v_isSharedCheck_4105_ = !lean_is_exclusive(v_stats_4078_);
if (v_isSharedCheck_4105_ == 0)
{
v___x_4094_ = v_stats_4078_;
v_isShared_4095_ = v_isSharedCheck_4105_;
goto v_resetjp_4093_;
}
else
{
lean_inc(v_goalStats_4092_);
lean_inc(v_ruleStats_4091_);
lean_inc(v_scriptGenerated_4090_);
lean_inc(v_forwardState_4089_);
lean_inc(v_script_4088_);
lean_inc(v_ruleSelection_4087_);
lean_inc(v_search_4086_);
lean_inc(v_ruleSetConstruction_4085_);
lean_inc(v_configParsing_4084_);
lean_inc(v_total_4083_);
lean_dec(v_stats_4078_);
v___x_4094_ = lean_box(0);
v_isShared_4095_ = v_isSharedCheck_4105_;
goto v_resetjp_4093_;
}
v_resetjp_4093_:
{
lean_object* v___x_4096_; lean_object* v___x_4097_; lean_object* v___x_4099_; 
v___x_4096_ = lean_nat_sub(v___x_4076_, v___x_4073_);
lean_dec(v___x_4073_);
lean_dec(v___x_4076_);
v___x_4097_ = lean_nat_add(v_ruleSelection_4087_, v___x_4096_);
lean_dec(v___x_4096_);
lean_dec(v_ruleSelection_4087_);
if (v_isShared_4095_ == 0)
{
lean_ctor_set(v___x_4094_, 4, v___x_4097_);
v___x_4099_ = v___x_4094_;
goto v_reusejp_4098_;
}
else
{
lean_object* v_reuseFailAlloc_4104_; 
v_reuseFailAlloc_4104_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_4104_, 0, v_total_4083_);
lean_ctor_set(v_reuseFailAlloc_4104_, 1, v_configParsing_4084_);
lean_ctor_set(v_reuseFailAlloc_4104_, 2, v_ruleSetConstruction_4085_);
lean_ctor_set(v_reuseFailAlloc_4104_, 3, v_search_4086_);
lean_ctor_set(v_reuseFailAlloc_4104_, 4, v___x_4097_);
lean_ctor_set(v_reuseFailAlloc_4104_, 5, v_script_4088_);
lean_ctor_set(v_reuseFailAlloc_4104_, 6, v_forwardState_4089_);
lean_ctor_set(v_reuseFailAlloc_4104_, 7, v_scriptGenerated_4090_);
lean_ctor_set(v_reuseFailAlloc_4104_, 8, v_ruleStats_4091_);
lean_ctor_set(v_reuseFailAlloc_4104_, 9, v_goalStats_4092_);
v___x_4099_ = v_reuseFailAlloc_4104_;
goto v_reusejp_4098_;
}
v_reusejp_4098_:
{
lean_object* v___x_4101_; 
if (v_isShared_4082_ == 0)
{
lean_ctor_set(v___x_4081_, 1, v___x_4099_);
v___x_4101_ = v___x_4081_;
goto v_reusejp_4100_;
}
else
{
lean_object* v_reuseFailAlloc_4103_; 
v_reuseFailAlloc_4103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4103_, 0, v_rulePatternCache_4079_);
lean_ctor_set(v_reuseFailAlloc_4103_, 1, v___x_4099_);
v___x_4101_ = v_reuseFailAlloc_4103_;
goto v_reusejp_4100_;
}
v_reusejp_4100_:
{
lean_object* v___x_4102_; 
v___x_4102_ = lean_st_ref_set(v_a_3721_, v___x_4101_);
v___y_4052_ = v___y_4068_;
v___y_4053_ = v___y_4070_;
v___y_4054_ = v___y_4069_;
v___y_4055_ = v___y_4071_;
v___y_4056_ = v___y_4072_;
v_a_4057_ = v_a_4075_;
goto v___jp_4051_;
}
}
}
}
}
else
{
lean_object* v_a_4107_; 
lean_dec(v___x_4073_);
v_a_4107_ = lean_ctor_get(v___x_4074_, 0);
lean_inc(v_a_4107_);
lean_dec_ref_known(v___x_4074_, 1);
v___y_4060_ = v___y_4068_;
v___y_4061_ = v___y_4070_;
v___y_4062_ = v___y_4069_;
v___y_4063_ = v___y_4071_;
v___y_4064_ = v___y_4072_;
v_a_4065_ = v_a_4107_;
goto v___jp_4059_;
}
}
v___jp_4108_:
{
lean_object* v___x_4114_; 
lean_inc(v_goal_3716_);
v___x_4114_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_4114_) == 0)
{
lean_object* v_a_4115_; 
v_a_4115_ = lean_ctor_get(v___x_4114_, 0);
lean_inc(v_a_4115_);
lean_dec_ref_known(v___x_4114_, 1);
v___y_4052_ = v___y_4109_;
v___y_4053_ = v___y_4111_;
v___y_4054_ = v___y_4110_;
v___y_4055_ = v___y_4112_;
v___y_4056_ = v___y_4113_;
v_a_4057_ = v_a_4115_;
goto v___jp_4051_;
}
else
{
lean_object* v_a_4116_; 
v_a_4116_ = lean_ctor_get(v___x_4114_, 0);
lean_inc(v_a_4116_);
lean_dec_ref_known(v___x_4114_, 1);
v___y_4060_ = v___y_4109_;
v___y_4061_ = v___y_4111_;
v___y_4062_ = v___y_4110_;
v___y_4063_ = v___y_4112_;
v___y_4064_ = v___y_4113_;
v_a_4065_ = v_a_4116_;
goto v___jp_4059_;
}
}
v___jp_4117_:
{
lean_object* v_a_4124_; uint8_t v___x_4125_; 
v_a_4124_ = lean_ctor_get(v___y_4123_, 0);
lean_inc(v_a_4124_);
lean_dec_ref(v___y_4123_);
v___x_4125_ = lean_unbox(v_a_4124_);
lean_dec(v_a_4124_);
if (v___x_4125_ == 0)
{
v___y_4109_ = v___y_4118_;
v___y_4110_ = v___y_4120_;
v___y_4111_ = v___y_4119_;
v___y_4112_ = v___y_4121_;
v___y_4113_ = v___y_4122_;
goto v___jp_4108_;
}
else
{
v___y_4068_ = v___y_4118_;
v___y_4069_ = v___y_4120_;
v___y_4070_ = v___y_4119_;
v___y_4071_ = v___y_4121_;
v___y_4072_ = v___y_4122_;
goto v___jp_4067_;
}
}
v___jp_4126_:
{
lean_object* v___x_4133_; double v___x_4134_; double v___x_4135_; lean_object* v___x_4136_; lean_object* v___x_4137_; lean_object* v___x_4138_; lean_object* v___x_4139_; lean_object* v___x_4140_; 
v___x_4133_ = lean_io_get_num_heartbeats();
v___x_4134_ = lean_float_of_nat(v___y_4131_);
v___x_4135_ = lean_float_of_nat(v___x_4133_);
v___x_4136_ = lean_box_float(v___x_4134_);
v___x_4137_ = lean_box_float(v___x_4135_);
v___x_4138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4138_, 0, v___x_4136_);
lean_ctor_set(v___x_4138_, 1, v___x_4137_);
v___x_4139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4139_, 0, v_a_4132_);
lean_ctor_set(v___x_4139_, 1, v___x_4138_);
v___x_4140_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules_spec__0(v___x_3734_, v___x_3737_, v___x_3738_, v_options_3727_, v___y_4128_, v___y_4130_, v___f_3732_, v___x_4139_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v___y_4018_ = v___y_4127_;
v___y_4019_ = v___y_4129_;
v___y_4020_ = v___x_4140_;
goto v___jp_4017_;
}
v___jp_4141_:
{
lean_object* v___x_4148_; 
v___x_4148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4148_, 0, v_a_4147_);
v___y_4127_ = v___y_4142_;
v___y_4128_ = v___y_4143_;
v___y_4129_ = v___y_4144_;
v___y_4130_ = v___y_4146_;
v___y_4131_ = v___y_4145_;
v_a_4132_ = v___x_4148_;
goto v___jp_4126_;
}
v___jp_4149_:
{
lean_object* v___x_4156_; 
v___x_4156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4156_, 0, v_a_4155_);
v___y_4127_ = v___y_4150_;
v___y_4128_ = v___y_4151_;
v___y_4129_ = v___y_4152_;
v___y_4130_ = v___y_4154_;
v___y_4131_ = v___y_4153_;
v_a_4132_ = v___x_4156_;
goto v___jp_4126_;
}
v___jp_4157_:
{
lean_object* v___x_4163_; lean_object* v___x_4164_; 
v___x_4163_ = lean_io_mono_nanos_now();
lean_inc(v_goal_3716_);
v___x_4164_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_4164_) == 0)
{
lean_object* v_a_4165_; lean_object* v___x_4166_; lean_object* v___x_4167_; lean_object* v_stats_4168_; lean_object* v_rulePatternCache_4169_; lean_object* v___x_4171_; uint8_t v_isShared_4172_; uint8_t v_isSharedCheck_4196_; 
v_a_4165_ = lean_ctor_get(v___x_4164_, 0);
lean_inc(v_a_4165_);
lean_dec_ref_known(v___x_4164_, 1);
v___x_4166_ = lean_io_mono_nanos_now();
v___x_4167_ = lean_st_ref_take(v_a_3721_);
v_stats_4168_ = lean_ctor_get(v___x_4167_, 1);
v_rulePatternCache_4169_ = lean_ctor_get(v___x_4167_, 0);
v_isSharedCheck_4196_ = !lean_is_exclusive(v___x_4167_);
if (v_isSharedCheck_4196_ == 0)
{
v___x_4171_ = v___x_4167_;
v_isShared_4172_ = v_isSharedCheck_4196_;
goto v_resetjp_4170_;
}
else
{
lean_inc(v_stats_4168_);
lean_inc(v_rulePatternCache_4169_);
lean_dec(v___x_4167_);
v___x_4171_ = lean_box(0);
v_isShared_4172_ = v_isSharedCheck_4196_;
goto v_resetjp_4170_;
}
v_resetjp_4170_:
{
lean_object* v_total_4173_; lean_object* v_configParsing_4174_; lean_object* v_ruleSetConstruction_4175_; lean_object* v_search_4176_; lean_object* v_ruleSelection_4177_; lean_object* v_script_4178_; lean_object* v_forwardState_4179_; lean_object* v_scriptGenerated_4180_; lean_object* v_ruleStats_4181_; lean_object* v_goalStats_4182_; lean_object* v___x_4184_; uint8_t v_isShared_4185_; uint8_t v_isSharedCheck_4195_; 
v_total_4173_ = lean_ctor_get(v_stats_4168_, 0);
v_configParsing_4174_ = lean_ctor_get(v_stats_4168_, 1);
v_ruleSetConstruction_4175_ = lean_ctor_get(v_stats_4168_, 2);
v_search_4176_ = lean_ctor_get(v_stats_4168_, 3);
v_ruleSelection_4177_ = lean_ctor_get(v_stats_4168_, 4);
v_script_4178_ = lean_ctor_get(v_stats_4168_, 5);
v_forwardState_4179_ = lean_ctor_get(v_stats_4168_, 6);
v_scriptGenerated_4180_ = lean_ctor_get(v_stats_4168_, 7);
v_ruleStats_4181_ = lean_ctor_get(v_stats_4168_, 8);
v_goalStats_4182_ = lean_ctor_get(v_stats_4168_, 9);
v_isSharedCheck_4195_ = !lean_is_exclusive(v_stats_4168_);
if (v_isSharedCheck_4195_ == 0)
{
v___x_4184_ = v_stats_4168_;
v_isShared_4185_ = v_isSharedCheck_4195_;
goto v_resetjp_4183_;
}
else
{
lean_inc(v_goalStats_4182_);
lean_inc(v_ruleStats_4181_);
lean_inc(v_scriptGenerated_4180_);
lean_inc(v_forwardState_4179_);
lean_inc(v_script_4178_);
lean_inc(v_ruleSelection_4177_);
lean_inc(v_search_4176_);
lean_inc(v_ruleSetConstruction_4175_);
lean_inc(v_configParsing_4174_);
lean_inc(v_total_4173_);
lean_dec(v_stats_4168_);
v___x_4184_ = lean_box(0);
v_isShared_4185_ = v_isSharedCheck_4195_;
goto v_resetjp_4183_;
}
v_resetjp_4183_:
{
lean_object* v___x_4186_; lean_object* v___x_4187_; lean_object* v___x_4189_; 
v___x_4186_ = lean_nat_sub(v___x_4166_, v___x_4163_);
lean_dec(v___x_4163_);
lean_dec(v___x_4166_);
v___x_4187_ = lean_nat_add(v_ruleSelection_4177_, v___x_4186_);
lean_dec(v___x_4186_);
lean_dec(v_ruleSelection_4177_);
if (v_isShared_4185_ == 0)
{
lean_ctor_set(v___x_4184_, 4, v___x_4187_);
v___x_4189_ = v___x_4184_;
goto v_reusejp_4188_;
}
else
{
lean_object* v_reuseFailAlloc_4194_; 
v_reuseFailAlloc_4194_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_4194_, 0, v_total_4173_);
lean_ctor_set(v_reuseFailAlloc_4194_, 1, v_configParsing_4174_);
lean_ctor_set(v_reuseFailAlloc_4194_, 2, v_ruleSetConstruction_4175_);
lean_ctor_set(v_reuseFailAlloc_4194_, 3, v_search_4176_);
lean_ctor_set(v_reuseFailAlloc_4194_, 4, v___x_4187_);
lean_ctor_set(v_reuseFailAlloc_4194_, 5, v_script_4178_);
lean_ctor_set(v_reuseFailAlloc_4194_, 6, v_forwardState_4179_);
lean_ctor_set(v_reuseFailAlloc_4194_, 7, v_scriptGenerated_4180_);
lean_ctor_set(v_reuseFailAlloc_4194_, 8, v_ruleStats_4181_);
lean_ctor_set(v_reuseFailAlloc_4194_, 9, v_goalStats_4182_);
v___x_4189_ = v_reuseFailAlloc_4194_;
goto v_reusejp_4188_;
}
v_reusejp_4188_:
{
lean_object* v___x_4191_; 
if (v_isShared_4172_ == 0)
{
lean_ctor_set(v___x_4171_, 1, v___x_4189_);
v___x_4191_ = v___x_4171_;
goto v_reusejp_4190_;
}
else
{
lean_object* v_reuseFailAlloc_4193_; 
v_reuseFailAlloc_4193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4193_, 0, v_rulePatternCache_4169_);
lean_ctor_set(v_reuseFailAlloc_4193_, 1, v___x_4189_);
v___x_4191_ = v_reuseFailAlloc_4193_;
goto v_reusejp_4190_;
}
v_reusejp_4190_:
{
lean_object* v___x_4192_; 
v___x_4192_ = lean_st_ref_set(v_a_3721_, v___x_4191_);
v___y_4142_ = v___y_4158_;
v___y_4143_ = v___y_4159_;
v___y_4144_ = v___y_4160_;
v___y_4145_ = v___y_4162_;
v___y_4146_ = v___y_4161_;
v_a_4147_ = v_a_4165_;
goto v___jp_4141_;
}
}
}
}
}
else
{
lean_object* v_a_4197_; 
lean_dec(v___x_4163_);
v_a_4197_ = lean_ctor_get(v___x_4164_, 0);
lean_inc(v_a_4197_);
lean_dec_ref_known(v___x_4164_, 1);
v___y_4150_ = v___y_4158_;
v___y_4151_ = v___y_4159_;
v___y_4152_ = v___y_4160_;
v___y_4153_ = v___y_4162_;
v___y_4154_ = v___y_4161_;
v_a_4155_ = v_a_4197_;
goto v___jp_4149_;
}
}
v___jp_4198_:
{
lean_object* v___x_4204_; 
lean_inc(v_goal_3716_);
v___x_4204_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
if (lean_obj_tag(v___x_4204_) == 0)
{
lean_object* v_a_4205_; 
v_a_4205_ = lean_ctor_get(v___x_4204_, 0);
lean_inc(v_a_4205_);
lean_dec_ref_known(v___x_4204_, 1);
v___y_4142_ = v___y_4199_;
v___y_4143_ = v___y_4200_;
v___y_4144_ = v___y_4201_;
v___y_4145_ = v___y_4203_;
v___y_4146_ = v___y_4202_;
v_a_4147_ = v_a_4205_;
goto v___jp_4141_;
}
else
{
lean_object* v_a_4206_; 
v_a_4206_ = lean_ctor_get(v___x_4204_, 0);
lean_inc(v_a_4206_);
lean_dec_ref_known(v___x_4204_, 1);
v___y_4150_ = v___y_4199_;
v___y_4151_ = v___y_4200_;
v___y_4152_ = v___y_4201_;
v___y_4153_ = v___y_4203_;
v___y_4154_ = v___y_4202_;
v_a_4155_ = v_a_4206_;
goto v___jp_4149_;
}
}
v___jp_4207_:
{
lean_object* v_a_4214_; uint8_t v___x_4215_; 
v_a_4214_ = lean_ctor_get(v___y_4213_, 0);
lean_inc(v_a_4214_);
lean_dec_ref(v___y_4213_);
v___x_4215_ = lean_unbox(v_a_4214_);
lean_dec(v_a_4214_);
if (v___x_4215_ == 0)
{
v___y_4199_ = v___y_4208_;
v___y_4200_ = v___y_4209_;
v___y_4201_ = v___y_4210_;
v___y_4202_ = v___y_4211_;
v___y_4203_ = v___y_4212_;
goto v___jp_4198_;
}
else
{
v___y_4158_ = v___y_4208_;
v___y_4159_ = v___y_4209_;
v___y_4160_ = v___y_4210_;
v___y_4161_ = v___y_4211_;
v___y_4162_ = v___y_4212_;
goto v___jp_4157_;
}
}
v___jp_4216_:
{
lean_object* v___x_4221_; 
v___x_4221_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v_a_3725_);
if (v___y_4220_ == 0)
{
lean_object* v_a_4222_; lean_object* v___x_4223_; lean_object* v___x_4224_; uint8_t v___x_4225_; 
v_a_4222_ = lean_ctor_get(v___x_4221_, 0);
lean_inc(v_a_4222_);
lean_dec_ref(v___x_4221_);
v___x_4223_ = lean_io_mono_nanos_now();
v___x_4224_ = lp_aesop_Aesop_aesop_collectStats;
v___x_4225_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3727_, v___x_4224_);
if (v___x_4225_ == 0)
{
lean_object* v___x_4226_; lean_object* v___x_4227_; lean_object* v_a_4228_; uint8_t v___x_4229_; 
v___x_4226_ = lp_aesop_Aesop_TraceOption_stats;
v___x_4227_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_4226_, v_a_3724_);
v_a_4228_ = lean_ctor_get(v___x_4227_, 0);
lean_inc(v_a_4228_);
v___x_4229_ = lean_unbox(v_a_4228_);
lean_dec(v_a_4228_);
if (v___x_4229_ == 0)
{
lean_object* v___x_4230_; lean_object* v___x_4231_; uint8_t v___x_4232_; 
lean_dec_ref(v___x_4227_);
v___x_4230_ = lp_aesop_Aesop_aesop_stats_file;
v___x_4231_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_3727_, v___x_4230_);
v___x_4232_ = lean_string_dec_eq(v___x_4231_, v___x_3738_);
lean_dec_ref(v___x_4231_);
if (v___x_4232_ == 0)
{
v___y_4068_ = v___y_4217_;
v___y_4069_ = v___y_4218_;
v___y_4070_ = v___x_4223_;
v___y_4071_ = v___y_4219_;
v___y_4072_ = v_a_4222_;
goto v___jp_4067_;
}
else
{
v___y_4109_ = v___y_4217_;
v___y_4110_ = v___y_4218_;
v___y_4111_ = v___x_4223_;
v___y_4112_ = v___y_4219_;
v___y_4113_ = v_a_4222_;
goto v___jp_4108_;
}
}
else
{
v___y_4118_ = v___y_4217_;
v___y_4119_ = v___x_4223_;
v___y_4120_ = v___y_4218_;
v___y_4121_ = v___y_4219_;
v___y_4122_ = v_a_4222_;
v___y_4123_ = v___x_4227_;
goto v___jp_4117_;
}
}
else
{
v___y_4068_ = v___y_4217_;
v___y_4069_ = v___y_4218_;
v___y_4070_ = v___x_4223_;
v___y_4071_ = v___y_4219_;
v___y_4072_ = v_a_4222_;
goto v___jp_4067_;
}
}
else
{
lean_object* v_a_4233_; lean_object* v___x_4234_; lean_object* v___x_4235_; uint8_t v___x_4236_; 
v_a_4233_ = lean_ctor_get(v___x_4221_, 0);
lean_inc(v_a_4233_);
lean_dec_ref(v___x_4221_);
v___x_4234_ = lean_io_get_num_heartbeats();
v___x_4235_ = lp_aesop_Aesop_aesop_collectStats;
v___x_4236_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3727_, v___x_4235_);
if (v___x_4236_ == 0)
{
lean_object* v___x_4237_; lean_object* v___x_4238_; lean_object* v_a_4239_; uint8_t v___x_4240_; 
v___x_4237_ = lp_aesop_Aesop_TraceOption_stats;
v___x_4238_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_4237_, v_a_3724_);
v_a_4239_ = lean_ctor_get(v___x_4238_, 0);
lean_inc(v_a_4239_);
v___x_4240_ = lean_unbox(v_a_4239_);
lean_dec(v_a_4239_);
if (v___x_4240_ == 0)
{
lean_object* v___x_4241_; lean_object* v___x_4242_; uint8_t v___x_4243_; 
lean_dec_ref(v___x_4238_);
v___x_4241_ = lp_aesop_Aesop_aesop_stats_file;
v___x_4242_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_3727_, v___x_4241_);
v___x_4243_ = lean_string_dec_eq(v___x_4242_, v___x_3738_);
lean_dec_ref(v___x_4242_);
if (v___x_4243_ == 0)
{
v___y_4158_ = v___y_4217_;
v___y_4159_ = v___y_4218_;
v___y_4160_ = v___y_4219_;
v___y_4161_ = v_a_4233_;
v___y_4162_ = v___x_4234_;
goto v___jp_4157_;
}
else
{
v___y_4199_ = v___y_4217_;
v___y_4200_ = v___y_4218_;
v___y_4201_ = v___y_4219_;
v___y_4202_ = v_a_4233_;
v___y_4203_ = v___x_4234_;
goto v___jp_4198_;
}
}
else
{
v___y_4208_ = v___y_4217_;
v___y_4209_ = v___y_4218_;
v___y_4210_ = v___y_4219_;
v___y_4211_ = v_a_4233_;
v___y_4212_ = v___x_4234_;
v___y_4213_ = v___x_4238_;
goto v___jp_4207_;
}
}
else
{
v___y_4158_ = v___y_4217_;
v___y_4159_ = v___y_4218_;
v___y_4160_ = v___y_4219_;
v___y_4161_ = v_a_4233_;
v___y_4162_ = v___x_4234_;
goto v___jp_4157_;
}
}
}
v___jp_4244_:
{
lean_object* v___x_4245_; lean_object* v_a_4246_; lean_object* v___x_4247_; uint8_t v___x_4248_; 
v___x_4245_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__5___redArg(v_a_3725_);
v_a_4246_ = lean_ctor_get(v___x_4245_, 0);
lean_inc(v_a_4246_);
lean_dec_ref(v___x_4245_);
v___x_4247_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4248_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3727_, v___x_4247_);
if (v___x_4248_ == 0)
{
lean_object* v___x_4249_; 
v___x_4249_ = lean_io_mono_nanos_now();
if (v___x_3752_ == 0)
{
lean_object* v___x_4250_; uint8_t v___x_4251_; 
v___x_4250_ = l_Lean_trace_profiler;
v___x_4251_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3727_, v___x_4250_);
if (v___x_4251_ == 0)
{
lean_object* v___x_4252_; lean_object* v_a_4253_; uint8_t v___x_4254_; lean_object* v___x_4255_; lean_object* v_a_4256_; uint8_t v___x_4257_; lean_object* v___x_4258_; 
v___x_4252_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v_a_4253_ = lean_ctor_get(v___x_4252_, 0);
lean_inc(v_a_4253_);
lean_dec_ref(v___x_4252_);
v___x_4254_ = lean_unbox(v_a_4253_);
lean_dec(v_a_4253_);
v___x_4255_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(v___x_4254_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v_a_4256_ = lean_ctor_get(v___x_4255_, 0);
lean_inc(v_a_4256_);
lean_dec_ref(v___x_4255_);
v___x_4257_ = lean_unbox(v_a_4256_);
lean_dec(v_a_4256_);
lean_inc(v_goal_3716_);
v___x_4258_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__1(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v___x_4257_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v___y_3774_ = v___x_4249_;
v___y_3775_ = v_a_4246_;
v___y_3776_ = v___x_4258_;
goto v___jp_3773_;
}
else
{
v___y_3973_ = v___x_4249_;
v___y_3974_ = v_a_4246_;
v___y_3975_ = v___x_4248_;
v___y_3976_ = v___x_3752_;
goto v___jp_3972_;
}
}
else
{
v___y_3973_ = v___x_4249_;
v___y_3974_ = v_a_4246_;
v___y_3975_ = v___x_4248_;
v___y_3976_ = v___x_3752_;
goto v___jp_3972_;
}
}
else
{
lean_object* v___x_4259_; 
v___x_4259_ = lean_io_get_num_heartbeats();
if (v___x_3752_ == 0)
{
lean_object* v___x_4260_; uint8_t v___x_4261_; 
v___x_4260_ = l_Lean_trace_profiler;
v___x_4261_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_3727_, v___x_4260_);
if (v___x_4261_ == 0)
{
lean_object* v___x_4262_; lean_object* v_a_4263_; uint8_t v___x_4264_; lean_object* v___x_4265_; lean_object* v_a_4266_; uint8_t v___x_4267_; lean_object* v___x_4268_; 
v___x_4262_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__0(v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v_a_4263_ = lean_ctor_get(v___x_4262_, 0);
lean_inc(v_a_4263_);
lean_dec_ref(v___x_4262_);
v___x_4264_ = lean_unbox(v_a_4263_);
lean_dec(v_a_4263_);
v___x_4265_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1(v___x_4264_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v_a_4266_ = lean_ctor_get(v___x_4265_, 0);
lean_inc(v_a_4266_);
lean_dec_ref(v___x_4265_);
v___x_4267_ = lean_unbox(v_a_4266_);
lean_dec(v_a_4266_);
lean_inc(v_goal_3716_);
v___x_4268_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___lam__1(v_rs_3715_, v___x_3735_, v_goal_3716_, v___f_3733_, v___x_4267_, v_a_3719_, v_a_3720_, v_a_3721_, v_a_3722_, v_a_3723_, v_a_3724_, v_a_3725_);
v___y_4018_ = v_a_4246_;
v___y_4019_ = v___x_4259_;
v___y_4020_ = v___x_4268_;
goto v___jp_4017_;
}
else
{
v___y_4217_ = v_a_4246_;
v___y_4218_ = v___x_3752_;
v___y_4219_ = v___x_4259_;
v___y_4220_ = v___x_4248_;
goto v___jp_4216_;
}
}
else
{
v___y_4217_ = v_a_4246_;
v___y_4218_ = v___x_3752_;
v___y_4219_ = v___x_4259_;
v___y_4220_ = v___x_4248_;
goto v___jp_4216_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules___boxed(lean_object* v_rs_4282_, lean_object* v_goal_4283_, lean_object* v_mvars_4284_, lean_object* v_preState_4285_, lean_object* v_a_4286_, lean_object* v_a_4287_, lean_object* v_a_4288_, lean_object* v_a_4289_, lean_object* v_a_4290_, lean_object* v_a_4291_, lean_object* v_a_4292_, lean_object* v_a_4293_){
_start:
{
lean_object* v_res_4294_; 
v_res_4294_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules(v_rs_4282_, v_goal_4283_, v_mvars_4284_, v_preState_4285_, v_a_4286_, v_a_4287_, v_a_4288_, v_a_4289_, v_a_4290_, v_a_4291_, v_a_4292_);
lean_dec(v_a_4292_);
lean_dec_ref(v_a_4291_);
lean_dec(v_a_4290_);
lean_dec_ref(v_a_4289_);
lean_dec(v_a_4288_);
lean_dec(v_a_4287_);
lean_dec_ref(v_a_4286_);
lean_dec_ref(v_preState_4285_);
return v_res_4294_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_4300_; lean_object* v___x_4301_; 
v___x_4300_ = l_Lean_maxRecDepthErrorMessage;
v___x_4301_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4301_, 0, v___x_4300_);
return v___x_4301_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__4(void){
_start:
{
lean_object* v___x_4302_; lean_object* v___x_4303_; 
v___x_4302_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__3, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__3_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__3);
v___x_4303_ = l_Lean_MessageData_ofFormat(v___x_4302_);
return v___x_4303_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__5(void){
_start:
{
lean_object* v___x_4304_; lean_object* v___x_4305_; lean_object* v___x_4306_; 
v___x_4304_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__4, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__4_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__4);
v___x_4305_ = ((lean_object*)(lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__2));
v___x_4306_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_4306_, 0, v___x_4305_);
lean_ctor_set(v___x_4306_, 1, v___x_4304_);
return v___x_4306_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg(lean_object* v_ref_4307_){
_start:
{
lean_object* v___x_4309_; lean_object* v___x_4310_; lean_object* v___x_4311_; 
v___x_4309_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__5, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__5_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___closed__5);
v___x_4310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4310_, 0, v_ref_4307_);
lean_ctor_set(v___x_4310_, 1, v___x_4309_);
v___x_4311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4311_, 0, v___x_4310_);
return v___x_4311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg___boxed(lean_object* v_ref_4312_, lean_object* v___y_4313_){
_start:
{
lean_object* v_res_4314_; 
v_res_4314_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg(v_ref_4312_);
return v_res_4314_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1(lean_object* v_00_u03b1_4315_, lean_object* v_ref_4316_, lean_object* v___y_4317_, lean_object* v___y_4318_, lean_object* v___y_4319_, lean_object* v___y_4320_, lean_object* v___y_4321_, lean_object* v___y_4322_, lean_object* v___y_4323_){
_start:
{
lean_object* v___x_4325_; 
v___x_4325_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg(v_ref_4316_);
return v___x_4325_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___boxed(lean_object* v_00_u03b1_4326_, lean_object* v_ref_4327_, lean_object* v___y_4328_, lean_object* v___y_4329_, lean_object* v___y_4330_, lean_object* v___y_4331_, lean_object* v___y_4332_, lean_object* v___y_4333_, lean_object* v___y_4334_, lean_object* v___y_4335_){
_start:
{
lean_object* v_res_4336_; 
v_res_4336_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1(v_00_u03b1_4326_, v_ref_4327_, v___y_4328_, v___y_4329_, v___y_4330_, v___y_4331_, v___y_4332_, v___y_4333_, v___y_4334_);
lean_dec(v___y_4334_);
lean_dec_ref(v___y_4333_);
lean_dec(v___y_4332_);
lean_dec_ref(v___y_4331_);
lean_dec(v___y_4330_);
lean_dec(v___y_4329_);
lean_dec_ref(v___y_4328_);
return v_res_4336_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__0(lean_object* v_x_4337_, lean_object* v_x_4338_){
_start:
{
if (lean_obj_tag(v_x_4338_) == 0)
{
return v_x_4337_;
}
else
{
lean_object* v_key_4339_; lean_object* v_tail_4340_; lean_object* v___x_4341_; 
v_key_4339_ = lean_ctor_get(v_x_4338_, 0);
lean_inc(v_key_4339_);
v_tail_4340_ = lean_ctor_get(v_x_4338_, 2);
lean_inc(v_tail_4340_);
lean_dec_ref_known(v_x_4338_, 3);
v___x_4341_ = lean_array_push(v_x_4337_, v_key_4339_);
v_x_4337_ = v___x_4341_;
v_x_4338_ = v_tail_4340_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__1(lean_object* v_as_4343_, size_t v_i_4344_, size_t v_stop_4345_, lean_object* v_b_4346_){
_start:
{
uint8_t v___x_4347_; 
v___x_4347_ = lean_usize_dec_eq(v_i_4344_, v_stop_4345_);
if (v___x_4347_ == 0)
{
lean_object* v___x_4348_; lean_object* v___x_4349_; size_t v___x_4350_; size_t v___x_4351_; 
v___x_4348_ = lean_array_uget_borrowed(v_as_4343_, v_i_4344_);
lean_inc(v___x_4348_);
v___x_4349_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__0(v_b_4346_, v___x_4348_);
v___x_4350_ = ((size_t)1ULL);
v___x_4351_ = lean_usize_add(v_i_4344_, v___x_4350_);
v_i_4344_ = v___x_4351_;
v_b_4346_ = v___x_4349_;
goto _start;
}
else
{
return v_b_4346_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__1___boxed(lean_object* v_as_4353_, lean_object* v_i_4354_, lean_object* v_stop_4355_, lean_object* v_b_4356_){
_start:
{
size_t v_i_boxed_4357_; size_t v_stop_boxed_4358_; lean_object* v_res_4359_; 
v_i_boxed_4357_ = lean_unbox_usize(v_i_4354_);
lean_dec(v_i_4354_);
v_stop_boxed_4358_ = lean_unbox_usize(v_stop_4355_);
lean_dec(v_stop_4355_);
v_res_4359_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__1(v_as_4353_, v_i_boxed_4357_, v_stop_boxed_4358_, v_b_4356_);
lean_dec_ref(v_as_4353_);
return v_res_4359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0(lean_object* v_xs_4360_){
_start:
{
lean_object* v_size_4361_; lean_object* v_buckets_4362_; lean_object* v___x_4363_; lean_object* v___x_4364_; lean_object* v___x_4365_; uint8_t v___x_4366_; 
v_size_4361_ = lean_ctor_get(v_xs_4360_, 0);
v_buckets_4362_ = lean_ctor_get(v_xs_4360_, 1);
v___x_4363_ = lean_mk_empty_array_with_capacity(v_size_4361_);
v___x_4364_ = lean_unsigned_to_nat(0u);
v___x_4365_ = lean_array_get_size(v_buckets_4362_);
v___x_4366_ = lean_nat_dec_lt(v___x_4364_, v___x_4365_);
if (v___x_4366_ == 0)
{
return v___x_4363_;
}
else
{
uint8_t v___x_4367_; 
v___x_4367_ = lean_nat_dec_le(v___x_4365_, v___x_4365_);
if (v___x_4367_ == 0)
{
if (v___x_4366_ == 0)
{
return v___x_4363_;
}
else
{
size_t v___x_4368_; size_t v___x_4369_; lean_object* v___x_4370_; 
v___x_4368_ = ((size_t)0ULL);
v___x_4369_ = lean_usize_of_nat(v___x_4365_);
v___x_4370_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__1(v_buckets_4362_, v___x_4368_, v___x_4369_, v___x_4363_);
return v___x_4370_;
}
}
else
{
size_t v___x_4371_; size_t v___x_4372_; lean_object* v___x_4373_; 
v___x_4371_ = ((size_t)0ULL);
v___x_4372_ = lean_usize_of_nat(v___x_4365_);
v___x_4373_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0_spec__1(v_buckets_4362_, v___x_4371_, v___x_4372_, v___x_4363_);
return v___x_4373_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0___boxed(lean_object* v_xs_4374_){
_start:
{
lean_object* v_res_4375_; 
v_res_4375_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0(v_xs_4374_);
lean_dec_ref(v_xs_4374_);
return v_res_4375_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__1(void){
_start:
{
lean_object* v___x_4377_; lean_object* v___x_4378_; 
v___x_4377_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__0));
v___x_4378_ = l_Lean_stringToMessageData(v___x_4377_);
return v___x_4378_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__3(void){
_start:
{
lean_object* v___x_4380_; lean_object* v___x_4381_; 
v___x_4380_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__2));
v___x_4381_ = l_Lean_stringToMessageData(v___x_4380_);
return v___x_4381_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go(lean_object* v_rs_4382_, lean_object* v_goal_4383_, lean_object* v_a_4384_, lean_object* v_a_4385_, lean_object* v_a_4386_, lean_object* v_a_4387_, lean_object* v_a_4388_, lean_object* v_a_4389_, lean_object* v_a_4390_){
_start:
{
lean_object* v_diff_4393_; lean_object* v___y_4394_; lean_object* v___y_4395_; lean_object* v___y_4396_; lean_object* v___y_4397_; lean_object* v___y_4398_; lean_object* v___y_4399_; lean_object* v___y_4400_; lean_object* v___y_4404_; lean_object* v___y_4405_; lean_object* v___y_4406_; lean_object* v___y_4407_; lean_object* v___y_4408_; lean_object* v___y_4409_; lean_object* v___y_4410_; lean_object* v_fileName_4456_; lean_object* v_fileMap_4457_; lean_object* v_options_4458_; lean_object* v_currRecDepth_4459_; lean_object* v_maxRecDepth_4460_; lean_object* v_ref_4461_; lean_object* v_currNamespace_4462_; lean_object* v_openDecls_4463_; lean_object* v_initHeartbeats_4464_; lean_object* v_maxHeartbeats_4465_; lean_object* v_quotContext_4466_; lean_object* v_currMacroScope_4467_; uint8_t v_diag_4468_; lean_object* v_cancelTk_x3f_4469_; uint8_t v_suppressElabErrors_4470_; lean_object* v_inheritedTraceOptions_4471_; lean_object* v___x_4472_; lean_object* v___x_4507_; uint8_t v___x_4508_; 
v_fileName_4456_ = lean_ctor_get(v_a_4389_, 0);
lean_inc_ref(v_fileName_4456_);
v_fileMap_4457_ = lean_ctor_get(v_a_4389_, 1);
lean_inc_ref(v_fileMap_4457_);
v_options_4458_ = lean_ctor_get(v_a_4389_, 2);
lean_inc_ref(v_options_4458_);
v_currRecDepth_4459_ = lean_ctor_get(v_a_4389_, 3);
lean_inc(v_currRecDepth_4459_);
v_maxRecDepth_4460_ = lean_ctor_get(v_a_4389_, 4);
lean_inc(v_maxRecDepth_4460_);
v_ref_4461_ = lean_ctor_get(v_a_4389_, 5);
lean_inc(v_ref_4461_);
v_currNamespace_4462_ = lean_ctor_get(v_a_4389_, 6);
lean_inc(v_currNamespace_4462_);
v_openDecls_4463_ = lean_ctor_get(v_a_4389_, 7);
lean_inc(v_openDecls_4463_);
v_initHeartbeats_4464_ = lean_ctor_get(v_a_4389_, 8);
lean_inc(v_initHeartbeats_4464_);
v_maxHeartbeats_4465_ = lean_ctor_get(v_a_4389_, 9);
lean_inc(v_maxHeartbeats_4465_);
v_quotContext_4466_ = lean_ctor_get(v_a_4389_, 10);
lean_inc(v_quotContext_4466_);
v_currMacroScope_4467_ = lean_ctor_get(v_a_4389_, 11);
lean_inc(v_currMacroScope_4467_);
v_diag_4468_ = lean_ctor_get_uint8(v_a_4389_, sizeof(void*)*14);
v_cancelTk_x3f_4469_ = lean_ctor_get(v_a_4389_, 12);
lean_inc(v_cancelTk_x3f_4469_);
v_suppressElabErrors_4470_ = lean_ctor_get_uint8(v_a_4389_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4471_ = lean_ctor_get(v_a_4389_, 13);
lean_inc_ref(v_inheritedTraceOptions_4471_);
lean_dec_ref(v_a_4389_);
v___x_4472_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__0_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_4507_ = lean_unsigned_to_nat(0u);
v___x_4508_ = lean_nat_dec_eq(v_maxRecDepth_4460_, v___x_4507_);
if (v___x_4508_ == 0)
{
uint8_t v___x_4509_; 
v___x_4509_ = lean_nat_dec_eq(v_currRecDepth_4459_, v_maxRecDepth_4460_);
if (v___x_4509_ == 0)
{
goto v___jp_4473_;
}
else
{
lean_object* v___x_4510_; 
lean_dec_ref(v_inheritedTraceOptions_4471_);
lean_dec(v_cancelTk_x3f_4469_);
lean_dec(v_currMacroScope_4467_);
lean_dec(v_quotContext_4466_);
lean_dec(v_maxHeartbeats_4465_);
lean_dec(v_initHeartbeats_4464_);
lean_dec(v_openDecls_4463_);
lean_dec(v_currNamespace_4462_);
lean_dec(v_maxRecDepth_4460_);
lean_dec(v_currRecDepth_4459_);
lean_dec_ref(v_options_4458_);
lean_dec_ref(v_fileMap_4457_);
lean_dec_ref(v_fileName_4456_);
lean_dec(v_goal_4383_);
lean_dec_ref(v_rs_4382_);
v___x_4510_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg(v_ref_4461_);
return v___x_4510_;
}
}
else
{
goto v___jp_4473_;
}
v___jp_4392_:
{
lean_object* v_newGoal_4401_; 
v_newGoal_4401_ = lean_ctor_get(v_diff_4393_, 1);
lean_inc(v_newGoal_4401_);
lean_dec_ref(v_diff_4393_);
v_goal_4383_ = v_newGoal_4401_;
v_a_4384_ = v___y_4394_;
v_a_4385_ = v___y_4395_;
v_a_4386_ = v___y_4396_;
v_a_4387_ = v___y_4397_;
v_a_4388_ = v___y_4398_;
v_a_4389_ = v___y_4399_;
v_a_4390_ = v___y_4400_;
goto _start;
}
v___jp_4403_:
{
uint8_t v___x_4411_; lean_object* v___x_4412_; 
v___x_4411_ = 0;
lean_inc(v_goal_4383_);
v___x_4412_ = l_Lean_MVarId_getMVarDependencies(v_goal_4383_, v___x_4411_, v___y_4407_, v___y_4408_, v___y_4409_, v___y_4410_);
if (lean_obj_tag(v___x_4412_) == 0)
{
lean_object* v_a_4413_; lean_object* v___x_4414_; 
v_a_4413_ = lean_ctor_get(v___x_4412_, 0);
lean_inc(v_a_4413_);
lean_dec_ref_known(v___x_4412_, 1);
v___x_4414_ = l_Lean_Meta_saveState___redArg(v___y_4408_, v___y_4410_);
if (lean_obj_tag(v___x_4414_) == 0)
{
lean_object* v_a_4415_; lean_object* v___x_4416_; lean_object* v___x_4417_; 
v_a_4415_ = lean_ctor_get(v___x_4414_, 0);
lean_inc(v_a_4415_);
lean_dec_ref_known(v___x_4414_, 1);
v___x_4416_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__0(v_a_4413_);
lean_dec(v_a_4413_);
lean_inc_ref(v___x_4416_);
lean_inc(v_goal_4383_);
lean_inc_ref(v_rs_4382_);
v___x_4417_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_tryNormRules(v_rs_4382_, v_goal_4383_, v___x_4416_, v_a_4415_, v___y_4404_, v___y_4405_, v___y_4406_, v___y_4407_, v___y_4408_, v___y_4409_, v___y_4410_);
if (lean_obj_tag(v___x_4417_) == 0)
{
lean_object* v_a_4418_; 
v_a_4418_ = lean_ctor_get(v___x_4417_, 0);
lean_inc(v_a_4418_);
lean_dec_ref_known(v___x_4417_, 1);
if (lean_obj_tag(v_a_4418_) == 1)
{
lean_object* v_val_4419_; 
lean_dec_ref(v___x_4416_);
lean_dec(v_a_4415_);
lean_dec(v_goal_4383_);
v_val_4419_ = lean_ctor_get(v_a_4418_, 0);
lean_inc(v_val_4419_);
lean_dec_ref_known(v_a_4418_, 1);
v_diff_4393_ = v_val_4419_;
v___y_4394_ = v___y_4404_;
v___y_4395_ = v___y_4405_;
v___y_4396_ = v___y_4406_;
v___y_4397_ = v___y_4407_;
v___y_4398_ = v___y_4408_;
v___y_4399_ = v___y_4409_;
v___y_4400_ = v___y_4410_;
goto v___jp_4392_;
}
else
{
lean_object* v___x_4420_; 
lean_dec(v_a_4418_);
lean_inc(v_goal_4383_);
lean_inc_ref(v_rs_4382_);
v___x_4420_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_trySafeRules(v_rs_4382_, v_goal_4383_, v___x_4416_, v_a_4415_, v___y_4404_, v___y_4405_, v___y_4406_, v___y_4407_, v___y_4408_, v___y_4409_, v___y_4410_);
lean_dec(v_a_4415_);
if (lean_obj_tag(v___x_4420_) == 0)
{
lean_object* v_a_4421_; 
v_a_4421_ = lean_ctor_get(v___x_4420_, 0);
lean_inc(v_a_4421_);
lean_dec_ref_known(v___x_4420_, 1);
if (lean_obj_tag(v_a_4421_) == 1)
{
lean_object* v_val_4422_; 
lean_dec(v_goal_4383_);
v_val_4422_ = lean_ctor_get(v_a_4421_, 0);
lean_inc(v_val_4422_);
lean_dec_ref_known(v_a_4421_, 1);
v_diff_4393_ = v_val_4422_;
v___y_4394_ = v___y_4404_;
v___y_4395_ = v___y_4405_;
v___y_4396_ = v___y_4406_;
v___y_4397_ = v___y_4407_;
v___y_4398_ = v___y_4408_;
v___y_4399_ = v___y_4409_;
v___y_4400_ = v___y_4410_;
goto v___jp_4392_;
}
else
{
lean_object* v___x_4423_; 
lean_dec(v_a_4421_);
lean_dec_ref(v_rs_4382_);
v___x_4423_ = lp_aesop_Aesop_clearForwardImplDetailHyps(v_goal_4383_, v___y_4407_, v___y_4408_, v___y_4409_, v___y_4410_);
lean_dec_ref(v___y_4409_);
return v___x_4423_;
}
}
else
{
lean_object* v_a_4424_; lean_object* v___x_4426_; uint8_t v_isShared_4427_; uint8_t v_isSharedCheck_4431_; 
lean_dec_ref(v___y_4409_);
lean_dec(v_goal_4383_);
lean_dec_ref(v_rs_4382_);
v_a_4424_ = lean_ctor_get(v___x_4420_, 0);
v_isSharedCheck_4431_ = !lean_is_exclusive(v___x_4420_);
if (v_isSharedCheck_4431_ == 0)
{
v___x_4426_ = v___x_4420_;
v_isShared_4427_ = v_isSharedCheck_4431_;
goto v_resetjp_4425_;
}
else
{
lean_inc(v_a_4424_);
lean_dec(v___x_4420_);
v___x_4426_ = lean_box(0);
v_isShared_4427_ = v_isSharedCheck_4431_;
goto v_resetjp_4425_;
}
v_resetjp_4425_:
{
lean_object* v___x_4429_; 
if (v_isShared_4427_ == 0)
{
v___x_4429_ = v___x_4426_;
goto v_reusejp_4428_;
}
else
{
lean_object* v_reuseFailAlloc_4430_; 
v_reuseFailAlloc_4430_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4430_, 0, v_a_4424_);
v___x_4429_ = v_reuseFailAlloc_4430_;
goto v_reusejp_4428_;
}
v_reusejp_4428_:
{
return v___x_4429_;
}
}
}
}
}
else
{
lean_object* v_a_4432_; lean_object* v___x_4434_; uint8_t v_isShared_4435_; uint8_t v_isSharedCheck_4439_; 
lean_dec_ref(v___x_4416_);
lean_dec(v_a_4415_);
lean_dec_ref(v___y_4409_);
lean_dec(v_goal_4383_);
lean_dec_ref(v_rs_4382_);
v_a_4432_ = lean_ctor_get(v___x_4417_, 0);
v_isSharedCheck_4439_ = !lean_is_exclusive(v___x_4417_);
if (v_isSharedCheck_4439_ == 0)
{
v___x_4434_ = v___x_4417_;
v_isShared_4435_ = v_isSharedCheck_4439_;
goto v_resetjp_4433_;
}
else
{
lean_inc(v_a_4432_);
lean_dec(v___x_4417_);
v___x_4434_ = lean_box(0);
v_isShared_4435_ = v_isSharedCheck_4439_;
goto v_resetjp_4433_;
}
v_resetjp_4433_:
{
lean_object* v___x_4437_; 
if (v_isShared_4435_ == 0)
{
v___x_4437_ = v___x_4434_;
goto v_reusejp_4436_;
}
else
{
lean_object* v_reuseFailAlloc_4438_; 
v_reuseFailAlloc_4438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4438_, 0, v_a_4432_);
v___x_4437_ = v_reuseFailAlloc_4438_;
goto v_reusejp_4436_;
}
v_reusejp_4436_:
{
return v___x_4437_;
}
}
}
}
else
{
lean_object* v_a_4440_; lean_object* v___x_4442_; uint8_t v_isShared_4443_; uint8_t v_isSharedCheck_4447_; 
lean_dec(v_a_4413_);
lean_dec_ref(v___y_4409_);
lean_dec(v_goal_4383_);
lean_dec_ref(v_rs_4382_);
v_a_4440_ = lean_ctor_get(v___x_4414_, 0);
v_isSharedCheck_4447_ = !lean_is_exclusive(v___x_4414_);
if (v_isSharedCheck_4447_ == 0)
{
v___x_4442_ = v___x_4414_;
v_isShared_4443_ = v_isSharedCheck_4447_;
goto v_resetjp_4441_;
}
else
{
lean_inc(v_a_4440_);
lean_dec(v___x_4414_);
v___x_4442_ = lean_box(0);
v_isShared_4443_ = v_isSharedCheck_4447_;
goto v_resetjp_4441_;
}
v_resetjp_4441_:
{
lean_object* v___x_4445_; 
if (v_isShared_4443_ == 0)
{
v___x_4445_ = v___x_4442_;
goto v_reusejp_4444_;
}
else
{
lean_object* v_reuseFailAlloc_4446_; 
v_reuseFailAlloc_4446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4446_, 0, v_a_4440_);
v___x_4445_ = v_reuseFailAlloc_4446_;
goto v_reusejp_4444_;
}
v_reusejp_4444_:
{
return v___x_4445_;
}
}
}
}
else
{
lean_object* v_a_4448_; lean_object* v___x_4450_; uint8_t v_isShared_4451_; uint8_t v_isSharedCheck_4455_; 
lean_dec_ref(v___y_4409_);
lean_dec(v_goal_4383_);
lean_dec_ref(v_rs_4382_);
v_a_4448_ = lean_ctor_get(v___x_4412_, 0);
v_isSharedCheck_4455_ = !lean_is_exclusive(v___x_4412_);
if (v_isSharedCheck_4455_ == 0)
{
v___x_4450_ = v___x_4412_;
v_isShared_4451_ = v_isSharedCheck_4455_;
goto v_resetjp_4449_;
}
else
{
lean_inc(v_a_4448_);
lean_dec(v___x_4412_);
v___x_4450_ = lean_box(0);
v_isShared_4451_ = v_isSharedCheck_4455_;
goto v_resetjp_4449_;
}
v_resetjp_4449_:
{
lean_object* v___x_4453_; 
if (v_isShared_4451_ == 0)
{
v___x_4453_ = v___x_4450_;
goto v_reusejp_4452_;
}
else
{
lean_object* v_reuseFailAlloc_4454_; 
v_reuseFailAlloc_4454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4454_, 0, v_a_4448_);
v___x_4453_ = v_reuseFailAlloc_4454_;
goto v_reusejp_4452_;
}
v_reusejp_4452_:
{
return v___x_4453_;
}
}
}
}
v___jp_4473_:
{
lean_object* v___x_4474_; lean_object* v___x_4475_; lean_object* v___x_4476_; lean_object* v___x_4477_; 
v___x_4474_ = lean_unsigned_to_nat(1u);
v___x_4475_ = lean_nat_add(v_currRecDepth_4459_, v___x_4474_);
lean_dec(v_currRecDepth_4459_);
lean_inc_ref(v_inheritedTraceOptions_4471_);
lean_inc_ref(v_options_4458_);
v___x_4476_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4476_, 0, v_fileName_4456_);
lean_ctor_set(v___x_4476_, 1, v_fileMap_4457_);
lean_ctor_set(v___x_4476_, 2, v_options_4458_);
lean_ctor_set(v___x_4476_, 3, v___x_4475_);
lean_ctor_set(v___x_4476_, 4, v_maxRecDepth_4460_);
lean_ctor_set(v___x_4476_, 5, v_ref_4461_);
lean_ctor_set(v___x_4476_, 6, v_currNamespace_4462_);
lean_ctor_set(v___x_4476_, 7, v_openDecls_4463_);
lean_ctor_set(v___x_4476_, 8, v_initHeartbeats_4464_);
lean_ctor_set(v___x_4476_, 9, v_maxHeartbeats_4465_);
lean_ctor_set(v___x_4476_, 10, v_quotContext_4466_);
lean_ctor_set(v___x_4476_, 11, v_currMacroScope_4467_);
lean_ctor_set(v___x_4476_, 12, v_cancelTk_x3f_4469_);
lean_ctor_set(v___x_4476_, 13, v_inheritedTraceOptions_4471_);
lean_ctor_set_uint8(v___x_4476_, sizeof(void*)*14, v_diag_4468_);
lean_ctor_set_uint8(v___x_4476_, sizeof(void*)*14 + 1, v_suppressElabErrors_4470_);
v___x_4477_ = l_Lean_Core_checkSystem(v___x_4472_, v___x_4476_, v_a_4390_);
if (lean_obj_tag(v___x_4477_) == 0)
{
uint8_t v_hasTrace_4478_; 
lean_dec_ref_known(v___x_4477_, 1);
v_hasTrace_4478_ = lean_ctor_get_uint8(v_options_4458_, sizeof(void*)*1);
if (v_hasTrace_4478_ == 0)
{
lean_dec_ref(v_inheritedTraceOptions_4471_);
lean_dec_ref(v_options_4458_);
v___y_4404_ = v_a_4384_;
v___y_4405_ = v_a_4385_;
v___y_4406_ = v_a_4386_;
v___y_4407_ = v_a_4387_;
v___y_4408_ = v_a_4388_;
v___y_4409_ = v___x_4476_;
v___y_4410_ = v_a_4390_;
goto v___jp_4403_;
}
else
{
lean_object* v___x_4479_; lean_object* v___x_4480_; uint8_t v___x_4481_; 
v___x_4479_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_4480_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___closed__0);
v___x_4481_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4471_, v_options_4458_, v___x_4480_);
lean_dec_ref(v_options_4458_);
lean_dec_ref(v_inheritedTraceOptions_4471_);
if (v___x_4481_ == 0)
{
v___y_4404_ = v_a_4384_;
v___y_4405_ = v_a_4385_;
v___y_4406_ = v_a_4386_;
v___y_4407_ = v_a_4387_;
v___y_4408_ = v_a_4388_;
v___y_4409_ = v___x_4476_;
v___y_4410_ = v_a_4390_;
goto v___jp_4403_;
}
else
{
lean_object* v___x_4482_; lean_object* v___x_4483_; lean_object* v___x_4484_; lean_object* v___x_4485_; lean_object* v___x_4486_; lean_object* v___x_4487_; lean_object* v___x_4488_; lean_object* v___x_4489_; lean_object* v___x_4490_; 
v___x_4482_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__1);
lean_inc_n(v_goal_4383_, 2);
v___x_4483_ = l_Lean_MessageData_ofName(v_goal_4383_);
v___x_4484_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4484_, 0, v___x_4482_);
lean_ctor_set(v___x_4484_, 1, v___x_4483_);
v___x_4485_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__3, &lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__3_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___closed__3);
v___x_4486_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4486_, 0, v___x_4484_);
lean_ctor_set(v___x_4486_, 1, v___x_4485_);
v___x_4487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4487_, 0, v_goal_4383_);
v___x_4488_ = l_Lean_indentD(v___x_4487_);
v___x_4489_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4489_, 0, v___x_4486_);
lean_ctor_set(v___x_4489_, 1, v___x_4488_);
v___x_4490_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v___x_4479_, v___x_4489_, v_a_4387_, v_a_4388_, v___x_4476_, v_a_4390_);
if (lean_obj_tag(v___x_4490_) == 0)
{
lean_dec_ref_known(v___x_4490_, 1);
v___y_4404_ = v_a_4384_;
v___y_4405_ = v_a_4385_;
v___y_4406_ = v_a_4386_;
v___y_4407_ = v_a_4387_;
v___y_4408_ = v_a_4388_;
v___y_4409_ = v___x_4476_;
v___y_4410_ = v_a_4390_;
goto v___jp_4403_;
}
else
{
lean_object* v_a_4491_; lean_object* v___x_4493_; uint8_t v_isShared_4494_; uint8_t v_isSharedCheck_4498_; 
lean_dec_ref_known(v___x_4476_, 14);
lean_dec(v_goal_4383_);
lean_dec_ref(v_rs_4382_);
v_a_4491_ = lean_ctor_get(v___x_4490_, 0);
v_isSharedCheck_4498_ = !lean_is_exclusive(v___x_4490_);
if (v_isSharedCheck_4498_ == 0)
{
v___x_4493_ = v___x_4490_;
v_isShared_4494_ = v_isSharedCheck_4498_;
goto v_resetjp_4492_;
}
else
{
lean_inc(v_a_4491_);
lean_dec(v___x_4490_);
v___x_4493_ = lean_box(0);
v_isShared_4494_ = v_isSharedCheck_4498_;
goto v_resetjp_4492_;
}
v_resetjp_4492_:
{
lean_object* v___x_4496_; 
if (v_isShared_4494_ == 0)
{
v___x_4496_ = v___x_4493_;
goto v_reusejp_4495_;
}
else
{
lean_object* v_reuseFailAlloc_4497_; 
v_reuseFailAlloc_4497_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4497_, 0, v_a_4491_);
v___x_4496_ = v_reuseFailAlloc_4497_;
goto v_reusejp_4495_;
}
v_reusejp_4495_:
{
return v___x_4496_;
}
}
}
}
}
}
else
{
lean_object* v_a_4499_; lean_object* v___x_4501_; uint8_t v_isShared_4502_; uint8_t v_isSharedCheck_4506_; 
lean_dec_ref_known(v___x_4476_, 14);
lean_dec_ref(v_inheritedTraceOptions_4471_);
lean_dec_ref(v_options_4458_);
lean_dec(v_goal_4383_);
lean_dec_ref(v_rs_4382_);
v_a_4499_ = lean_ctor_get(v___x_4477_, 0);
v_isSharedCheck_4506_ = !lean_is_exclusive(v___x_4477_);
if (v_isSharedCheck_4506_ == 0)
{
v___x_4501_ = v___x_4477_;
v_isShared_4502_ = v_isSharedCheck_4506_;
goto v_resetjp_4500_;
}
else
{
lean_inc(v_a_4499_);
lean_dec(v___x_4477_);
v___x_4501_ = lean_box(0);
v_isShared_4502_ = v_isSharedCheck_4506_;
goto v_resetjp_4500_;
}
v_resetjp_4500_:
{
lean_object* v___x_4504_; 
if (v_isShared_4502_ == 0)
{
v___x_4504_ = v___x_4501_;
goto v_reusejp_4503_;
}
else
{
lean_object* v_reuseFailAlloc_4505_; 
v_reuseFailAlloc_4505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4505_, 0, v_a_4499_);
v___x_4504_ = v_reuseFailAlloc_4505_;
goto v_reusejp_4503_;
}
v_reusejp_4503_:
{
return v___x_4504_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go___boxed(lean_object* v_rs_4511_, lean_object* v_goal_4512_, lean_object* v_a_4513_, lean_object* v_a_4514_, lean_object* v_a_4515_, lean_object* v_a_4516_, lean_object* v_a_4517_, lean_object* v_a_4518_, lean_object* v_a_4519_, lean_object* v_a_4520_){
_start:
{
lean_object* v_res_4521_; 
v_res_4521_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go(v_rs_4511_, v_goal_4512_, v_a_4513_, v_a_4514_, v_a_4515_, v_a_4516_, v_a_4517_, v_a_4518_, v_a_4519_);
lean_dec(v_a_4519_);
lean_dec(v_a_4517_);
lean_dec_ref(v_a_4516_);
lean_dec(v_a_4515_);
lean_dec(v_a_4514_);
lean_dec_ref(v_a_4513_);
return v_res_4521_;
}
}
static lean_object* _init_lp_aesop_Aesop_saturateCore___closed__2(void){
_start:
{
lean_object* v___x_4525_; lean_object* v___x_4526_; 
v___x_4525_ = ((lean_object*)(lp_aesop_Aesop_saturateCore___closed__1));
v___x_4526_ = l_Lean_MessageData_ofFormat(v___x_4525_);
return v___x_4526_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateCore(lean_object* v_rs_4527_, lean_object* v_goal_4528_, lean_object* v_a_4529_, lean_object* v_a_4530_, lean_object* v_a_4531_, lean_object* v_a_4532_, lean_object* v_a_4533_, lean_object* v_a_4534_, lean_object* v_a_4535_){
_start:
{
lean_object* v___y_4538_; lean_object* v___y_4539_; uint8_t v___y_4540_; lean_object* v___y_4554_; lean_object* v_a_4555_; lean_object* v___x_4558_; lean_object* v___x_4559_; 
v___x_4558_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
lean_inc(v_goal_4528_);
v___x_4559_ = l_Lean_MVarId_checkNotAssigned(v_goal_4528_, v___x_4558_, v_a_4532_, v_a_4533_, v_a_4534_, v_a_4535_);
if (lean_obj_tag(v___x_4559_) == 0)
{
lean_object* v___x_4560_; 
lean_dec_ref_known(v___x_4559_, 1);
lean_inc_ref(v_a_4534_);
v___x_4560_ = lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_go(v_rs_4527_, v_goal_4528_, v_a_4529_, v_a_4530_, v_a_4531_, v_a_4532_, v_a_4533_, v_a_4534_, v_a_4535_);
if (lean_obj_tag(v___x_4560_) == 0)
{
return v___x_4560_;
}
else
{
lean_object* v_a_4561_; 
v_a_4561_ = lean_ctor_get(v___x_4560_, 0);
lean_inc(v_a_4561_);
v___y_4554_ = v___x_4560_;
v_a_4555_ = v_a_4561_;
goto v___jp_4553_;
}
}
else
{
lean_object* v_a_4562_; lean_object* v___x_4564_; uint8_t v_isShared_4565_; uint8_t v_isSharedCheck_4569_; 
lean_dec(v_goal_4528_);
lean_dec_ref(v_rs_4527_);
v_a_4562_ = lean_ctor_get(v___x_4559_, 0);
v_isSharedCheck_4569_ = !lean_is_exclusive(v___x_4559_);
if (v_isSharedCheck_4569_ == 0)
{
v___x_4564_ = v___x_4559_;
v_isShared_4565_ = v_isSharedCheck_4569_;
goto v_resetjp_4563_;
}
else
{
lean_inc(v_a_4562_);
lean_dec(v___x_4559_);
v___x_4564_ = lean_box(0);
v_isShared_4565_ = v_isSharedCheck_4569_;
goto v_resetjp_4563_;
}
v_resetjp_4563_:
{
lean_object* v___x_4567_; 
lean_inc(v_a_4562_);
if (v_isShared_4565_ == 0)
{
v___x_4567_ = v___x_4564_;
goto v_reusejp_4566_;
}
else
{
lean_object* v_reuseFailAlloc_4568_; 
v_reuseFailAlloc_4568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4568_, 0, v_a_4562_);
v___x_4567_ = v_reuseFailAlloc_4568_;
goto v_reusejp_4566_;
}
v_reusejp_4566_:
{
v___y_4554_ = v___x_4567_;
v_a_4555_ = v_a_4562_;
goto v___jp_4553_;
}
}
}
v___jp_4537_:
{
if (v___y_4540_ == 0)
{
if (lean_obj_tag(v___y_4539_) == 0)
{
lean_object* v_ref_4541_; lean_object* v_msg_4542_; lean_object* v___x_4544_; uint8_t v_isShared_4545_; uint8_t v_isSharedCheck_4552_; 
lean_dec_ref(v___y_4538_);
v_ref_4541_ = lean_ctor_get(v___y_4539_, 0);
v_msg_4542_ = lean_ctor_get(v___y_4539_, 1);
v_isSharedCheck_4552_ = !lean_is_exclusive(v___y_4539_);
if (v_isSharedCheck_4552_ == 0)
{
v___x_4544_ = v___y_4539_;
v_isShared_4545_ = v_isSharedCheck_4552_;
goto v_resetjp_4543_;
}
else
{
lean_inc(v_msg_4542_);
lean_inc(v_ref_4541_);
lean_dec(v___y_4539_);
v___x_4544_ = lean_box(0);
v_isShared_4545_ = v_isSharedCheck_4552_;
goto v_resetjp_4543_;
}
v_resetjp_4543_:
{
lean_object* v___x_4546_; lean_object* v___x_4547_; lean_object* v___x_4549_; 
v___x_4546_ = lean_obj_once(&lp_aesop_Aesop_saturateCore___closed__2, &lp_aesop_Aesop_saturateCore___closed__2_once, _init_lp_aesop_Aesop_saturateCore___closed__2);
v___x_4547_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4547_, 0, v___x_4546_);
lean_ctor_set(v___x_4547_, 1, v_msg_4542_);
if (v_isShared_4545_ == 0)
{
lean_ctor_set(v___x_4544_, 1, v___x_4547_);
v___x_4549_ = v___x_4544_;
goto v_reusejp_4548_;
}
else
{
lean_object* v_reuseFailAlloc_4551_; 
v_reuseFailAlloc_4551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4551_, 0, v_ref_4541_);
lean_ctor_set(v_reuseFailAlloc_4551_, 1, v___x_4547_);
v___x_4549_ = v_reuseFailAlloc_4551_;
goto v_reusejp_4548_;
}
v_reusejp_4548_:
{
lean_object* v___x_4550_; 
v___x_4550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4550_, 0, v___x_4549_);
return v___x_4550_;
}
}
}
else
{
lean_dec_ref_known(v___y_4539_, 2);
return v___y_4538_;
}
}
else
{
lean_dec_ref(v___y_4539_);
return v___y_4538_;
}
}
v___jp_4553_:
{
uint8_t v___x_4556_; 
v___x_4556_ = l_Lean_Exception_isInterrupt(v_a_4555_);
if (v___x_4556_ == 0)
{
uint8_t v___x_4557_; 
lean_inc_ref(v_a_4555_);
v___x_4557_ = l_Lean_Exception_isRuntime(v_a_4555_);
v___y_4538_ = v___y_4554_;
v___y_4539_ = v_a_4555_;
v___y_4540_ = v___x_4557_;
goto v___jp_4537_;
}
else
{
v___y_4538_ = v___y_4554_;
v___y_4539_ = v_a_4555_;
v___y_4540_ = v___x_4556_;
goto v___jp_4537_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateCore___boxed(lean_object* v_rs_4570_, lean_object* v_goal_4571_, lean_object* v_a_4572_, lean_object* v_a_4573_, lean_object* v_a_4574_, lean_object* v_a_4575_, lean_object* v_a_4576_, lean_object* v_a_4577_, lean_object* v_a_4578_, lean_object* v_a_4579_){
_start:
{
lean_object* v_res_4580_; 
v_res_4580_ = lp_aesop_Aesop_saturateCore(v_rs_4570_, v_goal_4571_, v_a_4572_, v_a_4573_, v_a_4574_, v_a_4575_, v_a_4576_, v_a_4577_, v_a_4578_);
lean_dec(v_a_4578_);
lean_dec_ref(v_a_4577_);
lean_dec(v_a_4576_);
lean_dec_ref(v_a_4575_);
lean_dec(v_a_4574_);
lean_dec(v_a_4573_);
lean_dec_ref(v_a_4572_);
return v_res_4580_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg___lam__0(lean_object* v_x_4581_, lean_object* v___y_4582_, lean_object* v___y_4583_, lean_object* v___y_4584_, lean_object* v___y_4585_, lean_object* v___y_4586_, lean_object* v___y_4587_, lean_object* v___y_4588_){
_start:
{
lean_object* v___x_4590_; 
lean_inc(v___y_4584_);
lean_inc(v___y_4583_);
lean_inc_ref(v___y_4582_);
v___x_4590_ = lean_apply_8(v_x_4581_, v___y_4582_, v___y_4583_, v___y_4584_, v___y_4585_, v___y_4586_, v___y_4587_, v___y_4588_, lean_box(0));
return v___x_4590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg___lam__0___boxed(lean_object* v_x_4591_, lean_object* v___y_4592_, lean_object* v___y_4593_, lean_object* v___y_4594_, lean_object* v___y_4595_, lean_object* v___y_4596_, lean_object* v___y_4597_, lean_object* v___y_4598_, lean_object* v___y_4599_){
_start:
{
lean_object* v_res_4600_; 
v_res_4600_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg___lam__0(v_x_4591_, v___y_4592_, v___y_4593_, v___y_4594_, v___y_4595_, v___y_4596_, v___y_4597_, v___y_4598_);
lean_dec(v___y_4594_);
lean_dec(v___y_4593_);
lean_dec_ref(v___y_4592_);
return v_res_4600_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg(lean_object* v_mvarId_4601_, lean_object* v_x_4602_, lean_object* v___y_4603_, lean_object* v___y_4604_, lean_object* v___y_4605_, lean_object* v___y_4606_, lean_object* v___y_4607_, lean_object* v___y_4608_, lean_object* v___y_4609_){
_start:
{
lean_object* v___f_4611_; lean_object* v___x_4612_; 
lean_inc(v___y_4605_);
lean_inc(v___y_4604_);
lean_inc_ref(v___y_4603_);
v___f_4611_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg___lam__0___boxed), 9, 4);
lean_closure_set(v___f_4611_, 0, v_x_4602_);
lean_closure_set(v___f_4611_, 1, v___y_4603_);
lean_closure_set(v___f_4611_, 2, v___y_4604_);
lean_closure_set(v___f_4611_, 3, v___y_4605_);
v___x_4612_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_4601_, v___f_4611_, v___y_4606_, v___y_4607_, v___y_4608_, v___y_4609_);
if (lean_obj_tag(v___x_4612_) == 0)
{
return v___x_4612_;
}
else
{
lean_object* v_a_4613_; lean_object* v___x_4615_; uint8_t v_isShared_4616_; uint8_t v_isSharedCheck_4620_; 
v_a_4613_ = lean_ctor_get(v___x_4612_, 0);
v_isSharedCheck_4620_ = !lean_is_exclusive(v___x_4612_);
if (v_isSharedCheck_4620_ == 0)
{
v___x_4615_ = v___x_4612_;
v_isShared_4616_ = v_isSharedCheck_4620_;
goto v_resetjp_4614_;
}
else
{
lean_inc(v_a_4613_);
lean_dec(v___x_4612_);
v___x_4615_ = lean_box(0);
v_isShared_4616_ = v_isSharedCheck_4620_;
goto v_resetjp_4614_;
}
v_resetjp_4614_:
{
lean_object* v___x_4618_; 
if (v_isShared_4616_ == 0)
{
v___x_4618_ = v___x_4615_;
goto v_reusejp_4617_;
}
else
{
lean_object* v_reuseFailAlloc_4619_; 
v_reuseFailAlloc_4619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4619_, 0, v_a_4613_);
v___x_4618_ = v_reuseFailAlloc_4619_;
goto v_reusejp_4617_;
}
v_reusejp_4617_:
{
return v___x_4618_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg___boxed(lean_object* v_mvarId_4621_, lean_object* v_x_4622_, lean_object* v___y_4623_, lean_object* v___y_4624_, lean_object* v___y_4625_, lean_object* v___y_4626_, lean_object* v___y_4627_, lean_object* v___y_4628_, lean_object* v___y_4629_, lean_object* v___y_4630_){
_start:
{
lean_object* v_res_4631_; 
v_res_4631_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg(v_mvarId_4621_, v_x_4622_, v___y_4623_, v___y_4624_, v___y_4625_, v___y_4626_, v___y_4627_, v___y_4628_, v___y_4629_);
lean_dec(v___y_4629_);
lean_dec_ref(v___y_4628_);
lean_dec(v___y_4627_);
lean_dec_ref(v___y_4626_);
lean_dec(v___y_4625_);
lean_dec(v___y_4624_);
lean_dec_ref(v___y_4623_);
return v_res_4631_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5(lean_object* v_00_u03b1_4632_, lean_object* v_mvarId_4633_, lean_object* v_x_4634_, lean_object* v___y_4635_, lean_object* v___y_4636_, lean_object* v___y_4637_, lean_object* v___y_4638_, lean_object* v___y_4639_, lean_object* v___y_4640_, lean_object* v___y_4641_){
_start:
{
lean_object* v___x_4643_; 
v___x_4643_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg(v_mvarId_4633_, v_x_4634_, v___y_4635_, v___y_4636_, v___y_4637_, v___y_4638_, v___y_4639_, v___y_4640_, v___y_4641_);
return v___x_4643_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___boxed(lean_object* v_00_u03b1_4644_, lean_object* v_mvarId_4645_, lean_object* v_x_4646_, lean_object* v___y_4647_, lean_object* v___y_4648_, lean_object* v___y_4649_, lean_object* v___y_4650_, lean_object* v___y_4651_, lean_object* v___y_4652_, lean_object* v___y_4653_, lean_object* v___y_4654_){
_start:
{
lean_object* v_res_4655_; 
v_res_4655_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5(v_00_u03b1_4644_, v_mvarId_4645_, v_x_4646_, v___y_4647_, v___y_4648_, v___y_4649_, v___y_4650_, v___y_4651_, v___y_4652_, v___y_4653_);
lean_dec(v___y_4653_);
lean_dec_ref(v___y_4652_);
lean_dec(v___y_4651_);
lean_dec_ref(v___y_4650_);
lean_dec(v___y_4649_);
lean_dec(v___y_4648_);
lean_dec_ref(v___y_4647_);
return v_res_4655_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg(lean_object* v_a_4656_, lean_object* v_x_4657_){
_start:
{
if (lean_obj_tag(v_x_4657_) == 0)
{
uint8_t v___x_4658_; 
v___x_4658_ = 0;
return v___x_4658_;
}
else
{
lean_object* v_key_4659_; lean_object* v_tail_4660_; uint8_t v___x_4661_; 
v_key_4659_ = lean_ctor_get(v_x_4657_, 0);
v_tail_4660_ = lean_ctor_get(v_x_4657_, 2);
v___x_4661_ = l_Lean_instBEqFVarId_beq(v_key_4659_, v_a_4656_);
if (v___x_4661_ == 0)
{
v_x_4657_ = v_tail_4660_;
goto _start;
}
else
{
return v___x_4661_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg___boxed(lean_object* v_a_4663_, lean_object* v_x_4664_){
_start:
{
uint8_t v_res_4665_; lean_object* v_r_4666_; 
v_res_4665_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg(v_a_4663_, v_x_4664_);
lean_dec(v_x_4664_);
lean_dec(v_a_4663_);
v_r_4666_ = lean_box(v_res_4665_);
return v_r_4666_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___redArg(lean_object* v_m_4667_, lean_object* v_a_4668_){
_start:
{
lean_object* v_buckets_4669_; lean_object* v___x_4670_; uint64_t v___x_4671_; uint64_t v___x_4672_; uint64_t v___x_4673_; uint64_t v_fold_4674_; uint64_t v___x_4675_; uint64_t v___x_4676_; uint64_t v___x_4677_; size_t v___x_4678_; size_t v___x_4679_; size_t v___x_4680_; size_t v___x_4681_; size_t v___x_4682_; lean_object* v___x_4683_; uint8_t v___x_4684_; 
v_buckets_4669_ = lean_ctor_get(v_m_4667_, 1);
v___x_4670_ = lean_array_get_size(v_buckets_4669_);
v___x_4671_ = l_Lean_instHashableFVarId_hash(v_a_4668_);
v___x_4672_ = 32ULL;
v___x_4673_ = lean_uint64_shift_right(v___x_4671_, v___x_4672_);
v_fold_4674_ = lean_uint64_xor(v___x_4671_, v___x_4673_);
v___x_4675_ = 16ULL;
v___x_4676_ = lean_uint64_shift_right(v_fold_4674_, v___x_4675_);
v___x_4677_ = lean_uint64_xor(v_fold_4674_, v___x_4676_);
v___x_4678_ = lean_uint64_to_usize(v___x_4677_);
v___x_4679_ = lean_usize_of_nat(v___x_4670_);
v___x_4680_ = ((size_t)1ULL);
v___x_4681_ = lean_usize_sub(v___x_4679_, v___x_4680_);
v___x_4682_ = lean_usize_land(v___x_4678_, v___x_4681_);
v___x_4683_ = lean_array_uget_borrowed(v_buckets_4669_, v___x_4682_);
v___x_4684_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg(v_a_4668_, v___x_4683_);
return v___x_4684_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___redArg___boxed(lean_object* v_m_4685_, lean_object* v_a_4686_){
_start:
{
uint8_t v_res_4687_; lean_object* v_r_4688_; 
v_res_4687_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___redArg(v_m_4685_, v_a_4686_);
lean_dec(v_a_4686_);
lean_dec_ref(v_m_4685_);
v_r_4688_ = lean_box(v_res_4687_);
return v_r_4688_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__0(lean_object* v_erasedHyps_4689_, lean_object* v___y_4690_){
_start:
{
uint8_t v___x_4691_; 
v___x_4691_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___redArg(v_erasedHyps_4689_, v___y_4690_);
return v___x_4691_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__0___boxed(lean_object* v_erasedHyps_4692_, lean_object* v___y_4693_){
_start:
{
uint8_t v_res_4694_; lean_object* v_r_4695_; 
v_res_4694_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__0(v_erasedHyps_4692_, v___y_4693_);
lean_dec(v___y_4693_);
lean_dec_ref(v_erasedHyps_4692_);
v_r_4695_ = lean_box(v_res_4694_);
return v_r_4695_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__1(lean_object* v_x_4696_){
_start:
{
uint8_t v___x_4697_; 
v___x_4697_ = 1;
return v___x_4697_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__1___boxed(lean_object* v_x_4698_){
_start:
{
uint8_t v_res_4699_; lean_object* v_r_4700_; 
v_res_4699_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__1(v_x_4698_);
lean_dec_ref(v_x_4698_);
v_r_4700_ = lean_box(v_res_4699_);
return v_r_4700_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___redArg(lean_object* v_a_4701_, lean_object* v_x_4702_){
_start:
{
if (lean_obj_tag(v_x_4702_) == 0)
{
lean_object* v___x_4703_; 
v___x_4703_ = lean_box(0);
return v___x_4703_;
}
else
{
lean_object* v_key_4704_; lean_object* v_value_4705_; lean_object* v_tail_4706_; uint8_t v___x_4707_; 
v_key_4704_ = lean_ctor_get(v_x_4702_, 0);
v_value_4705_ = lean_ctor_get(v_x_4702_, 1);
v_tail_4706_ = lean_ctor_get(v_x_4702_, 2);
v___x_4707_ = l_Lean_instBEqFVarId_beq(v_key_4704_, v_a_4701_);
if (v___x_4707_ == 0)
{
v_x_4702_ = v_tail_4706_;
goto _start;
}
else
{
lean_object* v___x_4709_; 
lean_inc(v_value_4705_);
v___x_4709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4709_, 0, v_value_4705_);
return v___x_4709_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___redArg___boxed(lean_object* v_a_4710_, lean_object* v_x_4711_){
_start:
{
lean_object* v_res_4712_; 
v_res_4712_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___redArg(v_a_4710_, v_x_4711_);
lean_dec(v_x_4711_);
lean_dec(v_a_4710_);
return v_res_4712_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___redArg(lean_object* v_m_4713_, lean_object* v_a_4714_){
_start:
{
lean_object* v_buckets_4715_; lean_object* v___x_4716_; uint64_t v___x_4717_; uint64_t v___x_4718_; uint64_t v___x_4719_; uint64_t v_fold_4720_; uint64_t v___x_4721_; uint64_t v___x_4722_; uint64_t v___x_4723_; size_t v___x_4724_; size_t v___x_4725_; size_t v___x_4726_; size_t v___x_4727_; size_t v___x_4728_; lean_object* v___x_4729_; lean_object* v___x_4730_; 
v_buckets_4715_ = lean_ctor_get(v_m_4713_, 1);
v___x_4716_ = lean_array_get_size(v_buckets_4715_);
v___x_4717_ = l_Lean_instHashableFVarId_hash(v_a_4714_);
v___x_4718_ = 32ULL;
v___x_4719_ = lean_uint64_shift_right(v___x_4717_, v___x_4718_);
v_fold_4720_ = lean_uint64_xor(v___x_4717_, v___x_4719_);
v___x_4721_ = 16ULL;
v___x_4722_ = lean_uint64_shift_right(v_fold_4720_, v___x_4721_);
v___x_4723_ = lean_uint64_xor(v_fold_4720_, v___x_4722_);
v___x_4724_ = lean_uint64_to_usize(v___x_4723_);
v___x_4725_ = lean_usize_of_nat(v___x_4716_);
v___x_4726_ = ((size_t)1ULL);
v___x_4727_ = lean_usize_sub(v___x_4725_, v___x_4726_);
v___x_4728_ = lean_usize_land(v___x_4724_, v___x_4727_);
v___x_4729_ = lean_array_uget_borrowed(v_buckets_4715_, v___x_4728_);
v___x_4730_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___redArg(v_a_4714_, v___x_4729_);
return v___x_4730_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___redArg___boxed(lean_object* v_m_4731_, lean_object* v_a_4732_){
_start:
{
lean_object* v_res_4733_; 
v_res_4733_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___redArg(v_m_4731_, v_a_4732_);
lean_dec(v_a_4732_);
lean_dec_ref(v_m_4731_);
return v_res_4733_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__2(lean_object* v_hypDepths_4734_, lean_object* v___x_4735_, lean_object* v_depth_4736_, lean_object* v_h_4737_){
_start:
{
lean_object* v___y_4739_; lean_object* v___x_4741_; 
v___x_4741_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___redArg(v_hypDepths_4734_, v_h_4737_);
if (lean_obj_tag(v___x_4741_) == 0)
{
v___y_4739_ = v___x_4735_;
goto v___jp_4738_;
}
else
{
lean_object* v_val_4742_; 
lean_dec(v___x_4735_);
v_val_4742_ = lean_ctor_get(v___x_4741_, 0);
lean_inc(v_val_4742_);
lean_dec_ref_known(v___x_4741_, 1);
v___y_4739_ = v_val_4742_;
goto v___jp_4738_;
}
v___jp_4738_:
{
uint8_t v___x_4740_; 
v___x_4740_ = lean_nat_dec_le(v_depth_4736_, v___y_4739_);
if (v___x_4740_ == 0)
{
lean_dec(v___y_4739_);
lean_inc(v_depth_4736_);
return v_depth_4736_;
}
else
{
return v___y_4739_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__2___boxed(lean_object* v_hypDepths_4743_, lean_object* v___x_4744_, lean_object* v_depth_4745_, lean_object* v_h_4746_){
_start:
{
lean_object* v_res_4747_; 
v_res_4747_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__2(v_hypDepths_4743_, v___x_4744_, v_depth_4745_, v_h_4746_);
lean_dec(v_h_4746_);
lean_dec(v_depth_4745_);
lean_dec_ref(v_hypDepths_4743_);
return v_res_4747_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__3(uint8_t v_a_4748_, lean_object* v_x_4749_){
_start:
{
return v_a_4748_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__3___boxed(lean_object* v_a_4750_, lean_object* v_x_4751_){
_start:
{
uint8_t v_a_133803__boxed_4752_; uint8_t v_res_4753_; lean_object* v_r_4754_; 
v_a_133803__boxed_4752_ = lean_unbox(v_a_4750_);
v_res_4753_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__3(v_a_133803__boxed_4752_, v_x_4751_);
lean_dec_ref(v_x_4751_);
v_r_4754_ = lean_box(v_res_4753_);
return v_r_4754_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10_spec__13___redArg(lean_object* v_x_4755_, lean_object* v_x_4756_){
_start:
{
if (lean_obj_tag(v_x_4756_) == 0)
{
return v_x_4755_;
}
else
{
lean_object* v_key_4757_; lean_object* v_value_4758_; lean_object* v_tail_4759_; lean_object* v___x_4761_; uint8_t v_isShared_4762_; uint8_t v_isSharedCheck_4782_; 
v_key_4757_ = lean_ctor_get(v_x_4756_, 0);
v_value_4758_ = lean_ctor_get(v_x_4756_, 1);
v_tail_4759_ = lean_ctor_get(v_x_4756_, 2);
v_isSharedCheck_4782_ = !lean_is_exclusive(v_x_4756_);
if (v_isSharedCheck_4782_ == 0)
{
v___x_4761_ = v_x_4756_;
v_isShared_4762_ = v_isSharedCheck_4782_;
goto v_resetjp_4760_;
}
else
{
lean_inc(v_tail_4759_);
lean_inc(v_value_4758_);
lean_inc(v_key_4757_);
lean_dec(v_x_4756_);
v___x_4761_ = lean_box(0);
v_isShared_4762_ = v_isSharedCheck_4782_;
goto v_resetjp_4760_;
}
v_resetjp_4760_:
{
lean_object* v___x_4763_; uint64_t v___x_4764_; uint64_t v___x_4765_; uint64_t v___x_4766_; uint64_t v_fold_4767_; uint64_t v___x_4768_; uint64_t v___x_4769_; uint64_t v___x_4770_; size_t v___x_4771_; size_t v___x_4772_; size_t v___x_4773_; size_t v___x_4774_; size_t v___x_4775_; lean_object* v___x_4776_; lean_object* v___x_4778_; 
v___x_4763_ = lean_array_get_size(v_x_4755_);
v___x_4764_ = l_Lean_instHashableFVarId_hash(v_key_4757_);
v___x_4765_ = 32ULL;
v___x_4766_ = lean_uint64_shift_right(v___x_4764_, v___x_4765_);
v_fold_4767_ = lean_uint64_xor(v___x_4764_, v___x_4766_);
v___x_4768_ = 16ULL;
v___x_4769_ = lean_uint64_shift_right(v_fold_4767_, v___x_4768_);
v___x_4770_ = lean_uint64_xor(v_fold_4767_, v___x_4769_);
v___x_4771_ = lean_uint64_to_usize(v___x_4770_);
v___x_4772_ = lean_usize_of_nat(v___x_4763_);
v___x_4773_ = ((size_t)1ULL);
v___x_4774_ = lean_usize_sub(v___x_4772_, v___x_4773_);
v___x_4775_ = lean_usize_land(v___x_4771_, v___x_4774_);
v___x_4776_ = lean_array_uget_borrowed(v_x_4755_, v___x_4775_);
lean_inc(v___x_4776_);
if (v_isShared_4762_ == 0)
{
lean_ctor_set(v___x_4761_, 2, v___x_4776_);
v___x_4778_ = v___x_4761_;
goto v_reusejp_4777_;
}
else
{
lean_object* v_reuseFailAlloc_4781_; 
v_reuseFailAlloc_4781_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4781_, 0, v_key_4757_);
lean_ctor_set(v_reuseFailAlloc_4781_, 1, v_value_4758_);
lean_ctor_set(v_reuseFailAlloc_4781_, 2, v___x_4776_);
v___x_4778_ = v_reuseFailAlloc_4781_;
goto v_reusejp_4777_;
}
v_reusejp_4777_:
{
lean_object* v___x_4779_; 
v___x_4779_ = lean_array_uset(v_x_4755_, v___x_4775_, v___x_4778_);
v_x_4755_ = v___x_4779_;
v_x_4756_ = v_tail_4759_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10___redArg(lean_object* v_i_4783_, lean_object* v_source_4784_, lean_object* v_target_4785_){
_start:
{
lean_object* v___x_4786_; uint8_t v___x_4787_; 
v___x_4786_ = lean_array_get_size(v_source_4784_);
v___x_4787_ = lean_nat_dec_lt(v_i_4783_, v___x_4786_);
if (v___x_4787_ == 0)
{
lean_dec_ref(v_source_4784_);
lean_dec(v_i_4783_);
return v_target_4785_;
}
else
{
lean_object* v_es_4788_; lean_object* v___x_4789_; lean_object* v_source_4790_; lean_object* v_target_4791_; lean_object* v___x_4792_; lean_object* v___x_4793_; 
v_es_4788_ = lean_array_fget(v_source_4784_, v_i_4783_);
v___x_4789_ = lean_box(0);
v_source_4790_ = lean_array_fset(v_source_4784_, v_i_4783_, v___x_4789_);
v_target_4791_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10_spec__13___redArg(v_target_4785_, v_es_4788_);
v___x_4792_ = lean_unsigned_to_nat(1u);
v___x_4793_ = lean_nat_add(v_i_4783_, v___x_4792_);
lean_dec(v_i_4783_);
v_i_4783_ = v___x_4793_;
v_source_4784_ = v_source_4790_;
v_target_4785_ = v_target_4791_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8___redArg(lean_object* v_data_4795_){
_start:
{
lean_object* v___x_4796_; lean_object* v___x_4797_; lean_object* v_nbuckets_4798_; lean_object* v___x_4799_; lean_object* v___x_4800_; lean_object* v___x_4801_; lean_object* v___x_4802_; 
v___x_4796_ = lean_array_get_size(v_data_4795_);
v___x_4797_ = lean_unsigned_to_nat(2u);
v_nbuckets_4798_ = lean_nat_mul(v___x_4796_, v___x_4797_);
v___x_4799_ = lean_unsigned_to_nat(0u);
v___x_4800_ = lean_box(0);
v___x_4801_ = lean_mk_array(v_nbuckets_4798_, v___x_4800_);
v___x_4802_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10___redArg(v___x_4799_, v_data_4795_, v___x_4801_);
return v___x_4802_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__9___redArg(lean_object* v_a_4803_, lean_object* v_b_4804_, lean_object* v_x_4805_){
_start:
{
if (lean_obj_tag(v_x_4805_) == 0)
{
lean_dec(v_b_4804_);
lean_dec(v_a_4803_);
return v_x_4805_;
}
else
{
lean_object* v_key_4806_; lean_object* v_value_4807_; lean_object* v_tail_4808_; lean_object* v___x_4810_; uint8_t v_isShared_4811_; uint8_t v_isSharedCheck_4820_; 
v_key_4806_ = lean_ctor_get(v_x_4805_, 0);
v_value_4807_ = lean_ctor_get(v_x_4805_, 1);
v_tail_4808_ = lean_ctor_get(v_x_4805_, 2);
v_isSharedCheck_4820_ = !lean_is_exclusive(v_x_4805_);
if (v_isSharedCheck_4820_ == 0)
{
v___x_4810_ = v_x_4805_;
v_isShared_4811_ = v_isSharedCheck_4820_;
goto v_resetjp_4809_;
}
else
{
lean_inc(v_tail_4808_);
lean_inc(v_value_4807_);
lean_inc(v_key_4806_);
lean_dec(v_x_4805_);
v___x_4810_ = lean_box(0);
v_isShared_4811_ = v_isSharedCheck_4820_;
goto v_resetjp_4809_;
}
v_resetjp_4809_:
{
uint8_t v___x_4812_; 
v___x_4812_ = l_Lean_instBEqFVarId_beq(v_key_4806_, v_a_4803_);
if (v___x_4812_ == 0)
{
lean_object* v___x_4813_; lean_object* v___x_4815_; 
v___x_4813_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__9___redArg(v_a_4803_, v_b_4804_, v_tail_4808_);
if (v_isShared_4811_ == 0)
{
lean_ctor_set(v___x_4810_, 2, v___x_4813_);
v___x_4815_ = v___x_4810_;
goto v_reusejp_4814_;
}
else
{
lean_object* v_reuseFailAlloc_4816_; 
v_reuseFailAlloc_4816_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4816_, 0, v_key_4806_);
lean_ctor_set(v_reuseFailAlloc_4816_, 1, v_value_4807_);
lean_ctor_set(v_reuseFailAlloc_4816_, 2, v___x_4813_);
v___x_4815_ = v_reuseFailAlloc_4816_;
goto v_reusejp_4814_;
}
v_reusejp_4814_:
{
return v___x_4815_;
}
}
else
{
lean_object* v___x_4818_; 
lean_dec(v_value_4807_);
lean_dec(v_key_4806_);
if (v_isShared_4811_ == 0)
{
lean_ctor_set(v___x_4810_, 1, v_b_4804_);
lean_ctor_set(v___x_4810_, 0, v_a_4803_);
v___x_4818_ = v___x_4810_;
goto v_reusejp_4817_;
}
else
{
lean_object* v_reuseFailAlloc_4819_; 
v_reuseFailAlloc_4819_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4819_, 0, v_a_4803_);
lean_ctor_set(v_reuseFailAlloc_4819_, 1, v_b_4804_);
lean_ctor_set(v_reuseFailAlloc_4819_, 2, v_tail_4808_);
v___x_4818_ = v_reuseFailAlloc_4819_;
goto v_reusejp_4817_;
}
v_reusejp_4817_:
{
return v___x_4818_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4___redArg(lean_object* v_m_4821_, lean_object* v_a_4822_, lean_object* v_b_4823_){
_start:
{
lean_object* v_size_4824_; lean_object* v_buckets_4825_; lean_object* v___x_4827_; uint8_t v_isShared_4828_; uint8_t v_isSharedCheck_4868_; 
v_size_4824_ = lean_ctor_get(v_m_4821_, 0);
v_buckets_4825_ = lean_ctor_get(v_m_4821_, 1);
v_isSharedCheck_4868_ = !lean_is_exclusive(v_m_4821_);
if (v_isSharedCheck_4868_ == 0)
{
v___x_4827_ = v_m_4821_;
v_isShared_4828_ = v_isSharedCheck_4868_;
goto v_resetjp_4826_;
}
else
{
lean_inc(v_buckets_4825_);
lean_inc(v_size_4824_);
lean_dec(v_m_4821_);
v___x_4827_ = lean_box(0);
v_isShared_4828_ = v_isSharedCheck_4868_;
goto v_resetjp_4826_;
}
v_resetjp_4826_:
{
lean_object* v___x_4829_; uint64_t v___x_4830_; uint64_t v___x_4831_; uint64_t v___x_4832_; uint64_t v_fold_4833_; uint64_t v___x_4834_; uint64_t v___x_4835_; uint64_t v___x_4836_; size_t v___x_4837_; size_t v___x_4838_; size_t v___x_4839_; size_t v___x_4840_; size_t v___x_4841_; lean_object* v_bkt_4842_; uint8_t v___x_4843_; 
v___x_4829_ = lean_array_get_size(v_buckets_4825_);
v___x_4830_ = l_Lean_instHashableFVarId_hash(v_a_4822_);
v___x_4831_ = 32ULL;
v___x_4832_ = lean_uint64_shift_right(v___x_4830_, v___x_4831_);
v_fold_4833_ = lean_uint64_xor(v___x_4830_, v___x_4832_);
v___x_4834_ = 16ULL;
v___x_4835_ = lean_uint64_shift_right(v_fold_4833_, v___x_4834_);
v___x_4836_ = lean_uint64_xor(v_fold_4833_, v___x_4835_);
v___x_4837_ = lean_uint64_to_usize(v___x_4836_);
v___x_4838_ = lean_usize_of_nat(v___x_4829_);
v___x_4839_ = ((size_t)1ULL);
v___x_4840_ = lean_usize_sub(v___x_4838_, v___x_4839_);
v___x_4841_ = lean_usize_land(v___x_4837_, v___x_4840_);
v_bkt_4842_ = lean_array_uget_borrowed(v_buckets_4825_, v___x_4841_);
v___x_4843_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg(v_a_4822_, v_bkt_4842_);
if (v___x_4843_ == 0)
{
lean_object* v___x_4844_; lean_object* v_size_x27_4845_; lean_object* v___x_4846_; lean_object* v_buckets_x27_4847_; lean_object* v___x_4848_; lean_object* v___x_4849_; lean_object* v___x_4850_; lean_object* v___x_4851_; lean_object* v___x_4852_; uint8_t v___x_4853_; 
v___x_4844_ = lean_unsigned_to_nat(1u);
v_size_x27_4845_ = lean_nat_add(v_size_4824_, v___x_4844_);
lean_dec(v_size_4824_);
lean_inc(v_bkt_4842_);
v___x_4846_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4846_, 0, v_a_4822_);
lean_ctor_set(v___x_4846_, 1, v_b_4823_);
lean_ctor_set(v___x_4846_, 2, v_bkt_4842_);
v_buckets_x27_4847_ = lean_array_uset(v_buckets_4825_, v___x_4841_, v___x_4846_);
v___x_4848_ = lean_unsigned_to_nat(4u);
v___x_4849_ = lean_nat_mul(v_size_x27_4845_, v___x_4848_);
v___x_4850_ = lean_unsigned_to_nat(3u);
v___x_4851_ = lean_nat_div(v___x_4849_, v___x_4850_);
lean_dec(v___x_4849_);
v___x_4852_ = lean_array_get_size(v_buckets_x27_4847_);
v___x_4853_ = lean_nat_dec_le(v___x_4851_, v___x_4852_);
lean_dec(v___x_4851_);
if (v___x_4853_ == 0)
{
lean_object* v_val_4854_; lean_object* v___x_4856_; 
v_val_4854_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8___redArg(v_buckets_x27_4847_);
if (v_isShared_4828_ == 0)
{
lean_ctor_set(v___x_4827_, 1, v_val_4854_);
lean_ctor_set(v___x_4827_, 0, v_size_x27_4845_);
v___x_4856_ = v___x_4827_;
goto v_reusejp_4855_;
}
else
{
lean_object* v_reuseFailAlloc_4857_; 
v_reuseFailAlloc_4857_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4857_, 0, v_size_x27_4845_);
lean_ctor_set(v_reuseFailAlloc_4857_, 1, v_val_4854_);
v___x_4856_ = v_reuseFailAlloc_4857_;
goto v_reusejp_4855_;
}
v_reusejp_4855_:
{
return v___x_4856_;
}
}
else
{
lean_object* v___x_4859_; 
if (v_isShared_4828_ == 0)
{
lean_ctor_set(v___x_4827_, 1, v_buckets_x27_4847_);
lean_ctor_set(v___x_4827_, 0, v_size_x27_4845_);
v___x_4859_ = v___x_4827_;
goto v_reusejp_4858_;
}
else
{
lean_object* v_reuseFailAlloc_4860_; 
v_reuseFailAlloc_4860_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4860_, 0, v_size_x27_4845_);
lean_ctor_set(v_reuseFailAlloc_4860_, 1, v_buckets_x27_4847_);
v___x_4859_ = v_reuseFailAlloc_4860_;
goto v_reusejp_4858_;
}
v_reusejp_4858_:
{
return v___x_4859_;
}
}
}
else
{
lean_object* v___x_4861_; lean_object* v_buckets_x27_4862_; lean_object* v___x_4863_; lean_object* v___x_4864_; lean_object* v___x_4866_; 
lean_inc(v_bkt_4842_);
v___x_4861_ = lean_box(0);
v_buckets_x27_4862_ = lean_array_uset(v_buckets_4825_, v___x_4841_, v___x_4861_);
v___x_4863_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__9___redArg(v_a_4822_, v_b_4823_, v_bkt_4842_);
v___x_4864_ = lean_array_uset(v_buckets_x27_4862_, v___x_4841_, v___x_4863_);
if (v_isShared_4828_ == 0)
{
lean_ctor_set(v___x_4827_, 1, v___x_4864_);
v___x_4866_ = v___x_4827_;
goto v_reusejp_4865_;
}
else
{
lean_object* v_reuseFailAlloc_4867_; 
v_reuseFailAlloc_4867_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4867_, 0, v_size_4824_);
lean_ctor_set(v_reuseFailAlloc_4867_, 1, v___x_4864_);
v___x_4866_ = v_reuseFailAlloc_4867_;
goto v_reusejp_4865_;
}
v_reusejp_4865_:
{
return v___x_4866_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__0(lean_object* v_x_4869_, lean_object* v_x_4870_){
_start:
{
if (lean_obj_tag(v_x_4869_) == 0)
{
return v_x_4870_;
}
else
{
if (lean_obj_tag(v_x_4870_) == 0)
{
return v_x_4869_;
}
else
{
lean_object* v_rank_4871_; lean_object* v_val_4872_; lean_object* v_node_4873_; lean_object* v_next_4874_; lean_object* v_rank_4875_; lean_object* v_val_4876_; lean_object* v_node_4877_; lean_object* v_next_4878_; lean_object* v_fst_4880_; lean_object* v_snd_4881_; uint8_t v___x_4895_; 
v_rank_4871_ = lean_ctor_get(v_x_4869_, 0);
v_val_4872_ = lean_ctor_get(v_x_4869_, 1);
v_node_4873_ = lean_ctor_get(v_x_4869_, 2);
v_next_4874_ = lean_ctor_get(v_x_4869_, 3);
v_rank_4875_ = lean_ctor_get(v_x_4870_, 0);
v_val_4876_ = lean_ctor_get(v_x_4870_, 1);
v_node_4877_ = lean_ctor_get(v_x_4870_, 2);
v_next_4878_ = lean_ctor_get(v_x_4870_, 3);
v___x_4895_ = lean_nat_dec_lt(v_rank_4871_, v_rank_4875_);
if (v___x_4895_ == 0)
{
lean_object* v___x_4897_; uint8_t v_isShared_4898_; uint8_t v_isSharedCheck_4907_; 
lean_inc(v_next_4878_);
lean_inc(v_node_4877_);
lean_inc(v_val_4876_);
lean_inc(v_rank_4875_);
v_isSharedCheck_4907_ = !lean_is_exclusive(v_x_4870_);
if (v_isSharedCheck_4907_ == 0)
{
lean_object* v_unused_4908_; lean_object* v_unused_4909_; lean_object* v_unused_4910_; lean_object* v_unused_4911_; 
v_unused_4908_ = lean_ctor_get(v_x_4870_, 3);
lean_dec(v_unused_4908_);
v_unused_4909_ = lean_ctor_get(v_x_4870_, 2);
lean_dec(v_unused_4909_);
v_unused_4910_ = lean_ctor_get(v_x_4870_, 1);
lean_dec(v_unused_4910_);
v_unused_4911_ = lean_ctor_get(v_x_4870_, 0);
lean_dec(v_unused_4911_);
v___x_4897_ = v_x_4870_;
v_isShared_4898_ = v_isSharedCheck_4907_;
goto v_resetjp_4896_;
}
else
{
lean_dec(v_x_4870_);
v___x_4897_ = lean_box(0);
v_isShared_4898_ = v_isSharedCheck_4907_;
goto v_resetjp_4896_;
}
v_resetjp_4896_:
{
uint8_t v___x_4899_; 
v___x_4899_ = lean_nat_dec_lt(v_rank_4875_, v_rank_4871_);
if (v___x_4899_ == 0)
{
uint8_t v___x_4900_; 
lean_inc(v_next_4874_);
lean_inc(v_node_4873_);
lean_inc(v_val_4872_);
lean_inc(v_rank_4871_);
lean_del_object(v___x_4897_);
lean_dec(v_rank_4875_);
lean_dec_ref_known(v_x_4869_, 4);
v___x_4900_ = lp_aesop_Aesop_ForwardRuleMatch_le(v_val_4872_, v_val_4876_);
if (v___x_4900_ == 0)
{
lean_object* v___x_4901_; 
v___x_4901_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4901_, 0, v_val_4872_);
lean_ctor_set(v___x_4901_, 1, v_node_4873_);
lean_ctor_set(v___x_4901_, 2, v_node_4877_);
v_fst_4880_ = v_val_4876_;
v_snd_4881_ = v___x_4901_;
goto v___jp_4879_;
}
else
{
lean_object* v___x_4902_; 
v___x_4902_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4902_, 0, v_val_4876_);
lean_ctor_set(v___x_4902_, 1, v_node_4877_);
lean_ctor_set(v___x_4902_, 2, v_node_4873_);
v_fst_4880_ = v_val_4872_;
v_snd_4881_ = v___x_4902_;
goto v___jp_4879_;
}
}
else
{
lean_object* v___x_4903_; lean_object* v___x_4905_; 
v___x_4903_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__0(v_x_4869_, v_next_4878_);
if (v_isShared_4898_ == 0)
{
lean_ctor_set(v___x_4897_, 3, v___x_4903_);
v___x_4905_ = v___x_4897_;
goto v_reusejp_4904_;
}
else
{
lean_object* v_reuseFailAlloc_4906_; 
v_reuseFailAlloc_4906_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_4906_, 0, v_rank_4875_);
lean_ctor_set(v_reuseFailAlloc_4906_, 1, v_val_4876_);
lean_ctor_set(v_reuseFailAlloc_4906_, 2, v_node_4877_);
lean_ctor_set(v_reuseFailAlloc_4906_, 3, v___x_4903_);
v___x_4905_ = v_reuseFailAlloc_4906_;
goto v_reusejp_4904_;
}
v_reusejp_4904_:
{
return v___x_4905_;
}
}
}
}
else
{
lean_object* v___x_4913_; uint8_t v_isShared_4914_; uint8_t v_isSharedCheck_4919_; 
lean_inc(v_next_4874_);
lean_inc(v_node_4873_);
lean_inc(v_val_4872_);
lean_inc(v_rank_4871_);
v_isSharedCheck_4919_ = !lean_is_exclusive(v_x_4869_);
if (v_isSharedCheck_4919_ == 0)
{
lean_object* v_unused_4920_; lean_object* v_unused_4921_; lean_object* v_unused_4922_; lean_object* v_unused_4923_; 
v_unused_4920_ = lean_ctor_get(v_x_4869_, 3);
lean_dec(v_unused_4920_);
v_unused_4921_ = lean_ctor_get(v_x_4869_, 2);
lean_dec(v_unused_4921_);
v_unused_4922_ = lean_ctor_get(v_x_4869_, 1);
lean_dec(v_unused_4922_);
v_unused_4923_ = lean_ctor_get(v_x_4869_, 0);
lean_dec(v_unused_4923_);
v___x_4913_ = v_x_4869_;
v_isShared_4914_ = v_isSharedCheck_4919_;
goto v_resetjp_4912_;
}
else
{
lean_dec(v_x_4869_);
v___x_4913_ = lean_box(0);
v_isShared_4914_ = v_isSharedCheck_4919_;
goto v_resetjp_4912_;
}
v_resetjp_4912_:
{
lean_object* v___x_4915_; lean_object* v___x_4917_; 
v___x_4915_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__0(v_next_4874_, v_x_4870_);
if (v_isShared_4914_ == 0)
{
lean_ctor_set(v___x_4913_, 3, v___x_4915_);
v___x_4917_ = v___x_4913_;
goto v_reusejp_4916_;
}
else
{
lean_object* v_reuseFailAlloc_4918_; 
v_reuseFailAlloc_4918_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v_reuseFailAlloc_4918_, 0, v_rank_4871_);
lean_ctor_set(v_reuseFailAlloc_4918_, 1, v_val_4872_);
lean_ctor_set(v_reuseFailAlloc_4918_, 2, v_node_4873_);
lean_ctor_set(v_reuseFailAlloc_4918_, 3, v___x_4915_);
v___x_4917_ = v_reuseFailAlloc_4918_;
goto v_reusejp_4916_;
}
v_reusejp_4916_:
{
return v___x_4917_;
}
}
}
v___jp_4879_:
{
lean_object* v___x_4882_; lean_object* v_r_4883_; uint8_t v___x_4884_; 
v___x_4882_ = lean_unsigned_to_nat(1u);
v_r_4883_ = lean_nat_add(v_rank_4871_, v___x_4882_);
lean_dec(v_rank_4871_);
v___x_4884_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_4874_, v_r_4883_);
if (v___x_4884_ == 0)
{
uint8_t v___x_4885_; 
v___x_4885_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_4878_, v_r_4883_);
if (v___x_4885_ == 0)
{
lean_object* v___x_4886_; lean_object* v___x_4887_; 
v___x_4886_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__0(v_next_4874_, v_next_4878_);
v___x_4887_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_4887_, 0, v_r_4883_);
lean_ctor_set(v___x_4887_, 1, v_fst_4880_);
lean_ctor_set(v___x_4887_, 2, v_snd_4881_);
lean_ctor_set(v___x_4887_, 3, v___x_4886_);
return v___x_4887_;
}
else
{
lean_object* v___x_4888_; 
v___x_4888_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_4888_, 0, v_r_4883_);
lean_ctor_set(v___x_4888_, 1, v_fst_4880_);
lean_ctor_set(v___x_4888_, 2, v_snd_4881_);
lean_ctor_set(v___x_4888_, 3, v_next_4878_);
v_x_4869_ = v_next_4874_;
v_x_4870_ = v___x_4888_;
goto _start;
}
}
else
{
uint8_t v___x_4890_; 
v___x_4890_ = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(v_next_4878_, v_r_4883_);
if (v___x_4890_ == 0)
{
lean_object* v___x_4891_; 
v___x_4891_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_4891_, 0, v_r_4883_);
lean_ctor_set(v___x_4891_, 1, v_fst_4880_);
lean_ctor_set(v___x_4891_, 2, v_snd_4881_);
lean_ctor_set(v___x_4891_, 3, v_next_4874_);
v_x_4869_ = v___x_4891_;
v_x_4870_ = v_next_4878_;
goto _start;
}
else
{
lean_object* v___x_4893_; lean_object* v___x_4894_; 
v___x_4893_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__0(v_next_4874_, v_next_4878_);
v___x_4894_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_4894_, 0, v_r_4883_);
lean_ctor_set(v___x_4894_, 1, v_fst_4880_);
lean_ctor_set(v___x_4894_, 2, v_snd_4881_);
lean_ctor_set(v___x_4894_, 3, v___x_4893_);
return v___x_4894_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__3(lean_object* v_as_4924_, size_t v_i_4925_, size_t v_stop_4926_, lean_object* v_b_4927_){
_start:
{
uint8_t v___x_4928_; 
v___x_4928_ = lean_usize_dec_eq(v_i_4925_, v_stop_4926_);
if (v___x_4928_ == 0)
{
lean_object* v___x_4929_; lean_object* v___x_4930_; lean_object* v___x_4931_; lean_object* v___x_4932_; lean_object* v___x_4933_; lean_object* v___x_4934_; size_t v___x_4935_; size_t v___x_4936_; 
v___x_4929_ = lean_array_uget_borrowed(v_as_4924_, v_i_4925_);
v___x_4930_ = lean_unsigned_to_nat(0u);
v___x_4931_ = lean_box(0);
v___x_4932_ = lean_box(0);
lean_inc(v___x_4929_);
v___x_4933_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_4933_, 0, v___x_4930_);
lean_ctor_set(v___x_4933_, 1, v___x_4929_);
lean_ctor_set(v___x_4933_, 2, v___x_4931_);
lean_ctor_set(v___x_4933_, 3, v___x_4932_);
v___x_4934_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__0(v___x_4933_, v_b_4927_);
v___x_4935_ = ((size_t)1ULL);
v___x_4936_ = lean_usize_add(v_i_4925_, v___x_4935_);
v_i_4925_ = v___x_4936_;
v_b_4927_ = v___x_4934_;
goto _start;
}
else
{
return v_b_4927_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__3___boxed(lean_object* v_as_4938_, lean_object* v_i_4939_, lean_object* v_stop_4940_, lean_object* v_b_4941_){
_start:
{
size_t v_i_boxed_4942_; size_t v_stop_boxed_4943_; lean_object* v_res_4944_; 
v_i_boxed_4942_ = lean_unbox_usize(v_i_4939_);
lean_dec(v_i_4939_);
v_stop_boxed_4943_ = lean_unbox_usize(v_stop_4940_);
lean_dec(v_stop_4940_);
v_res_4944_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__3(v_as_4938_, v_i_boxed_4942_, v_stop_boxed_4943_, v_b_4941_);
lean_dec_ref(v_as_4938_);
return v_res_4944_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__6(lean_object* v_as_4945_, size_t v_i_4946_, size_t v_stop_4947_, lean_object* v_b_4948_){
_start:
{
uint8_t v___x_4949_; 
v___x_4949_ = lean_usize_dec_eq(v_i_4946_, v_stop_4947_);
if (v___x_4949_ == 0)
{
lean_object* v___x_4950_; lean_object* v___x_4951_; size_t v___x_4952_; size_t v___x_4953_; 
v___x_4950_ = lean_array_uget_borrowed(v_as_4945_, v_i_4946_);
lean_inc(v___x_4950_);
v___x_4951_ = lp_aesop_Aesop_ForwardState_eraseHyp(v___x_4950_, v_b_4948_);
v___x_4952_ = ((size_t)1ULL);
v___x_4953_ = lean_usize_add(v_i_4946_, v___x_4952_);
v_i_4946_ = v___x_4953_;
v_b_4948_ = v___x_4951_;
goto _start;
}
else
{
return v_b_4948_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__6___boxed(lean_object* v_as_4955_, lean_object* v_i_4956_, lean_object* v_stop_4957_, lean_object* v_b_4958_){
_start:
{
size_t v_i_boxed_4959_; size_t v_stop_boxed_4960_; lean_object* v_res_4961_; 
v_i_boxed_4959_ = lean_unbox_usize(v_i_4956_);
lean_dec(v_i_4956_);
v_stop_boxed_4960_ = lean_unbox_usize(v_stop_4957_);
lean_dec(v_stop_4957_);
v_res_4961_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__6(v_as_4955_, v_i_boxed_4959_, v_stop_boxed_4960_, v_b_4958_);
lean_dec_ref(v_as_4955_);
return v_res_4961_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__3___redArg(lean_object* v_m_4962_, lean_object* v_a_4963_, lean_object* v_b_4964_){
_start:
{
lean_object* v_size_4965_; lean_object* v_buckets_4966_; lean_object* v___x_4967_; uint64_t v___x_4968_; uint64_t v___x_4969_; uint64_t v___x_4970_; uint64_t v_fold_4971_; uint64_t v___x_4972_; uint64_t v___x_4973_; uint64_t v___x_4974_; size_t v___x_4975_; size_t v___x_4976_; size_t v___x_4977_; size_t v___x_4978_; size_t v___x_4979_; lean_object* v_bkt_4980_; uint8_t v___x_4981_; 
v_size_4965_ = lean_ctor_get(v_m_4962_, 0);
v_buckets_4966_ = lean_ctor_get(v_m_4962_, 1);
v___x_4967_ = lean_array_get_size(v_buckets_4966_);
v___x_4968_ = l_Lean_instHashableFVarId_hash(v_a_4963_);
v___x_4969_ = 32ULL;
v___x_4970_ = lean_uint64_shift_right(v___x_4968_, v___x_4969_);
v_fold_4971_ = lean_uint64_xor(v___x_4968_, v___x_4970_);
v___x_4972_ = 16ULL;
v___x_4973_ = lean_uint64_shift_right(v_fold_4971_, v___x_4972_);
v___x_4974_ = lean_uint64_xor(v_fold_4971_, v___x_4973_);
v___x_4975_ = lean_uint64_to_usize(v___x_4974_);
v___x_4976_ = lean_usize_of_nat(v___x_4967_);
v___x_4977_ = ((size_t)1ULL);
v___x_4978_ = lean_usize_sub(v___x_4976_, v___x_4977_);
v___x_4979_ = lean_usize_land(v___x_4975_, v___x_4978_);
v_bkt_4980_ = lean_array_uget_borrowed(v_buckets_4966_, v___x_4979_);
v___x_4981_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg(v_a_4963_, v_bkt_4980_);
if (v___x_4981_ == 0)
{
lean_object* v___x_4983_; uint8_t v_isShared_4984_; uint8_t v_isSharedCheck_5002_; 
lean_inc_ref(v_buckets_4966_);
lean_inc(v_size_4965_);
v_isSharedCheck_5002_ = !lean_is_exclusive(v_m_4962_);
if (v_isSharedCheck_5002_ == 0)
{
lean_object* v_unused_5003_; lean_object* v_unused_5004_; 
v_unused_5003_ = lean_ctor_get(v_m_4962_, 1);
lean_dec(v_unused_5003_);
v_unused_5004_ = lean_ctor_get(v_m_4962_, 0);
lean_dec(v_unused_5004_);
v___x_4983_ = v_m_4962_;
v_isShared_4984_ = v_isSharedCheck_5002_;
goto v_resetjp_4982_;
}
else
{
lean_dec(v_m_4962_);
v___x_4983_ = lean_box(0);
v_isShared_4984_ = v_isSharedCheck_5002_;
goto v_resetjp_4982_;
}
v_resetjp_4982_:
{
lean_object* v___x_4985_; lean_object* v_size_x27_4986_; lean_object* v___x_4987_; lean_object* v_buckets_x27_4988_; lean_object* v___x_4989_; lean_object* v___x_4990_; lean_object* v___x_4991_; lean_object* v___x_4992_; lean_object* v___x_4993_; uint8_t v___x_4994_; 
v___x_4985_ = lean_unsigned_to_nat(1u);
v_size_x27_4986_ = lean_nat_add(v_size_4965_, v___x_4985_);
lean_dec(v_size_4965_);
lean_inc(v_bkt_4980_);
v___x_4987_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4987_, 0, v_a_4963_);
lean_ctor_set(v___x_4987_, 1, v_b_4964_);
lean_ctor_set(v___x_4987_, 2, v_bkt_4980_);
v_buckets_x27_4988_ = lean_array_uset(v_buckets_4966_, v___x_4979_, v___x_4987_);
v___x_4989_ = lean_unsigned_to_nat(4u);
v___x_4990_ = lean_nat_mul(v_size_x27_4986_, v___x_4989_);
v___x_4991_ = lean_unsigned_to_nat(3u);
v___x_4992_ = lean_nat_div(v___x_4990_, v___x_4991_);
lean_dec(v___x_4990_);
v___x_4993_ = lean_array_get_size(v_buckets_x27_4988_);
v___x_4994_ = lean_nat_dec_le(v___x_4992_, v___x_4993_);
lean_dec(v___x_4992_);
if (v___x_4994_ == 0)
{
lean_object* v_val_4995_; lean_object* v___x_4997_; 
v_val_4995_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8___redArg(v_buckets_x27_4988_);
if (v_isShared_4984_ == 0)
{
lean_ctor_set(v___x_4983_, 1, v_val_4995_);
lean_ctor_set(v___x_4983_, 0, v_size_x27_4986_);
v___x_4997_ = v___x_4983_;
goto v_reusejp_4996_;
}
else
{
lean_object* v_reuseFailAlloc_4998_; 
v_reuseFailAlloc_4998_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4998_, 0, v_size_x27_4986_);
lean_ctor_set(v_reuseFailAlloc_4998_, 1, v_val_4995_);
v___x_4997_ = v_reuseFailAlloc_4998_;
goto v_reusejp_4996_;
}
v_reusejp_4996_:
{
return v___x_4997_;
}
}
else
{
lean_object* v___x_5000_; 
if (v_isShared_4984_ == 0)
{
lean_ctor_set(v___x_4983_, 1, v_buckets_x27_4988_);
lean_ctor_set(v___x_4983_, 0, v_size_x27_4986_);
v___x_5000_ = v___x_4983_;
goto v_reusejp_4999_;
}
else
{
lean_object* v_reuseFailAlloc_5001_; 
v_reuseFailAlloc_5001_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5001_, 0, v_size_x27_4986_);
lean_ctor_set(v_reuseFailAlloc_5001_, 1, v_buckets_x27_4988_);
v___x_5000_ = v_reuseFailAlloc_5001_;
goto v_reusejp_4999_;
}
v_reusejp_4999_:
{
return v___x_5000_;
}
}
}
}
else
{
lean_dec(v_b_4964_);
lean_dec(v_a_4963_);
return v_m_4962_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__4(lean_object* v_as_5005_, size_t v_sz_5006_, size_t v_i_5007_, lean_object* v_b_5008_){
_start:
{
uint8_t v___x_5009_; 
v___x_5009_ = lean_usize_dec_lt(v_i_5007_, v_sz_5006_);
if (v___x_5009_ == 0)
{
return v_b_5008_;
}
else
{
lean_object* v_a_5010_; lean_object* v___x_5011_; lean_object* v_r_5012_; size_t v___x_5013_; size_t v___x_5014_; 
v_a_5010_ = lean_array_uget_borrowed(v_as_5005_, v_i_5007_);
v___x_5011_ = lean_box(0);
lean_inc(v_a_5010_);
v_r_5012_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__3___redArg(v_b_5008_, v_a_5010_, v___x_5011_);
v___x_5013_ = ((size_t)1ULL);
v___x_5014_ = lean_usize_add(v_i_5007_, v___x_5013_);
v_i_5007_ = v___x_5014_;
v_b_5008_ = v_r_5012_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__4___boxed(lean_object* v_as_5016_, lean_object* v_sz_5017_, lean_object* v_i_5018_, lean_object* v_b_5019_){
_start:
{
size_t v_sz_boxed_5020_; size_t v_i_boxed_5021_; lean_object* v_res_5022_; 
v_sz_boxed_5020_ = lean_unbox_usize(v_sz_5017_);
lean_dec(v_sz_5017_);
v_i_boxed_5021_ = lean_unbox_usize(v_i_5018_);
lean_dec(v_i_5018_);
v_res_5022_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__4(v_as_5016_, v_sz_boxed_5020_, v_i_boxed_5021_, v_b_5019_);
lean_dec_ref(v_as_5016_);
return v_res_5022_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2(lean_object* v_m_5023_, lean_object* v_l_5024_){
_start:
{
size_t v_sz_5025_; size_t v___x_5026_; lean_object* v___x_5027_; 
v_sz_5025_ = lean_array_size(v_l_5024_);
v___x_5026_ = ((size_t)0ULL);
v___x_5027_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__4(v_l_5024_, v_sz_5025_, v___x_5026_, v_m_5023_);
return v___x_5027_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2___boxed(lean_object* v_m_5028_, lean_object* v_l_5029_){
_start:
{
lean_object* v_res_5030_; 
v_res_5030_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2(v_m_5028_, v_l_5029_);
lean_dec_ref(v_l_5029_);
return v_res_5030_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__1(void){
_start:
{
lean_object* v___x_5034_; lean_object* v___x_5035_; 
v___x_5034_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__0));
v___x_5035_ = l_Lean_stringToMessageData(v___x_5034_);
return v___x_5035_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__3(void){
_start:
{
lean_object* v___x_5037_; lean_object* v___x_5038_; 
v___x_5037_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__2));
v___x_5038_ = l_Lean_stringToMessageData(v___x_5037_);
return v___x_5038_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__5(void){
_start:
{
lean_object* v___x_5040_; lean_object* v___x_5041_; 
v___x_5040_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__4));
v___x_5041_ = l_Lean_stringToMessageData(v___x_5040_);
return v___x_5041_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4(lean_object* v_fst_5042_, lean_object* v___x_5043_, lean_object* v_rs_5044_, lean_object* v_snd_5045_, lean_object* v_fst_5046_, lean_object* v___f_5047_, uint8_t v___x_5048_, lean_object* v___x_5049_, lean_object* v_erasedHyps_5050_, lean_object* v_snd_5051_, lean_object* v_hypDepths_5052_, lean_object* v___f_5053_, lean_object* v_fst_5054_, lean_object* v___x_5055_, lean_object* v___y_5056_, lean_object* v___y_5057_, lean_object* v___y_5058_, lean_object* v___y_5059_, lean_object* v___y_5060_, lean_object* v___y_5061_, lean_object* v___y_5062_, lean_object* v___y_5063_){
_start:
{
lean_object* v___y_5066_; lean_object* v___y_5067_; lean_object* v___y_5068_; lean_object* v___y_5069_; lean_object* v___y_5070_; lean_object* v___y_5071_; lean_object* v___y_5072_; lean_object* v___y_5073_; lean_object* v___y_5074_; lean_object* v_a_5075_; lean_object* v___y_5103_; lean_object* v___y_5104_; lean_object* v___y_5105_; lean_object* v___y_5106_; lean_object* v___y_5107_; lean_object* v___y_5108_; lean_object* v___y_5109_; lean_object* v___y_5110_; lean_object* v___y_5111_; lean_object* v___y_5112_; lean_object* v___y_5113_; lean_object* v___y_5168_; lean_object* v___y_5169_; lean_object* v___y_5170_; lean_object* v___y_5171_; lean_object* v___y_5172_; lean_object* v___y_5173_; lean_object* v___y_5174_; lean_object* v___y_5175_; lean_object* v___y_5176_; lean_object* v___y_5177_; lean_object* v___y_5178_; uint8_t v_a_5179_; lean_object* v___y_5202_; lean_object* v___y_5203_; lean_object* v___y_5204_; lean_object* v___y_5205_; lean_object* v___y_5206_; lean_object* v___y_5207_; lean_object* v___y_5208_; lean_object* v___y_5209_; lean_object* v___y_5210_; lean_object* v___y_5211_; lean_object* v___y_5212_; lean_object* v___y_5213_; lean_object* v_options_5224_; lean_object* v_inheritedTraceOptions_5225_; lean_object* v___x_5226_; lean_object* v___y_5228_; lean_object* v___y_5229_; lean_object* v___y_5230_; lean_object* v___y_5231_; lean_object* v___y_5232_; lean_object* v___y_5233_; lean_object* v___y_5234_; lean_object* v___y_5235_; lean_object* v___y_5236_; lean_object* v___y_5237_; lean_object* v_a_5238_; lean_object* v___y_5250_; lean_object* v___y_5251_; lean_object* v___y_5252_; lean_object* v___y_5253_; lean_object* v___y_5254_; lean_object* v___y_5255_; lean_object* v___y_5256_; lean_object* v___y_5257_; lean_object* v___y_5258_; lean_object* v___y_5259_; lean_object* v___y_5260_; uint8_t v_a_5261_; lean_object* v___y_5307_; lean_object* v___y_5308_; lean_object* v___y_5309_; lean_object* v___y_5310_; lean_object* v___y_5311_; lean_object* v___y_5312_; lean_object* v___y_5313_; lean_object* v___y_5314_; lean_object* v___y_5315_; lean_object* v___y_5316_; lean_object* v___y_5317_; uint8_t v_a_5318_; lean_object* v___y_5330_; lean_object* v___y_5331_; lean_object* v___y_5332_; lean_object* v___y_5333_; lean_object* v___y_5334_; lean_object* v___y_5335_; lean_object* v___y_5336_; lean_object* v___y_5337_; lean_object* v___y_5338_; lean_object* v___y_5339_; lean_object* v___y_5340_; lean_object* v___y_5341_; lean_object* v___y_5353_; lean_object* v___y_5354_; lean_object* v___y_5355_; lean_object* v___y_5356_; lean_object* v___y_5357_; lean_object* v___y_5358_; lean_object* v___y_5359_; lean_object* v___y_5360_; lean_object* v___y_5361_; lean_object* v___y_5362_; lean_object* v___y_5363_; lean_object* v___y_5375_; lean_object* v___y_5376_; lean_object* v___y_5377_; lean_object* v___y_5378_; lean_object* v___y_5379_; lean_object* v___y_5380_; lean_object* v___y_5381_; lean_object* v___y_5382_; lean_object* v___y_5383_; lean_object* v___y_5384_; lean_object* v___y_5385_; lean_object* v___y_5386_; lean_object* v___y_5387_; lean_object* v___y_5392_; lean_object* v___y_5393_; lean_object* v___y_5394_; lean_object* v___y_5395_; lean_object* v_depth_5396_; lean_object* v_hypDepths_5397_; lean_object* v___y_5398_; lean_object* v___y_5399_; lean_object* v___y_5400_; lean_object* v___y_5401_; lean_object* v___y_5402_; lean_object* v___y_5403_; lean_object* v_options_5404_; lean_object* v_inheritedTraceOptions_5405_; lean_object* v___y_5406_; lean_object* v_a_5435_; uint8_t v_a_5485_; lean_object* v___y_5487_; uint8_t v___x_5498_; 
v_options_5224_ = lean_ctor_get(v___y_5062_, 2);
v_inheritedTraceOptions_5225_ = lean_ctor_get(v___y_5062_, 13);
v___x_5226_ = lp_aesop_Aesop_aesop_collectStats;
v___x_5498_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_5224_, v___x_5226_);
if (v___x_5498_ == 0)
{
lean_object* v___x_5499_; lean_object* v___x_5500_; 
v___x_5499_ = lp_aesop_Aesop_TraceOption_stats;
v___x_5500_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_5499_, v___y_5062_);
if (lean_obj_tag(v___x_5500_) == 0)
{
lean_object* v_a_5501_; uint8_t v___x_5502_; 
v_a_5501_ = lean_ctor_get(v___x_5500_, 0);
lean_inc(v_a_5501_);
v___x_5502_ = lean_unbox(v_a_5501_);
lean_dec(v_a_5501_);
if (v___x_5502_ == 0)
{
lean_object* v___x_5503_; lean_object* v___x_5504_; lean_object* v___x_5505_; uint8_t v___x_5506_; 
v___x_5503_ = lp_aesop_Aesop_aesop_stats_file;
v___x_5504_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_5224_, v___x_5503_);
v___x_5505_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_5506_ = lean_string_dec_eq(v___x_5504_, v___x_5505_);
lean_dec_ref(v___x_5504_);
if (v___x_5506_ == 0)
{
lean_dec_ref_known(v___x_5500_, 1);
goto v___jp_5451_;
}
else
{
v___y_5487_ = v___x_5500_;
goto v___jp_5486_;
}
}
else
{
v___y_5487_ = v___x_5500_;
goto v___jp_5486_;
}
}
else
{
v___y_5487_ = v___x_5500_;
goto v___jp_5486_;
}
}
else
{
v_a_5485_ = v___x_5498_;
goto v___jp_5484_;
}
v___jp_5065_:
{
lean_object* v___x_5076_; lean_object* v___x_5077_; 
v___x_5076_ = lean_box(0);
lean_inc(v_fst_5042_);
v___x_5077_ = lp_aesop_Aesop_ForwardState_update(v_fst_5042_, v_a_5075_, v___x_5076_, v___y_5074_, v___y_5072_, v___y_5071_, v___y_5067_, v___y_5069_);
if (lean_obj_tag(v___x_5077_) == 0)
{
lean_object* v_a_5078_; lean_object* v_fst_5079_; lean_object* v_snd_5080_; lean_object* v___x_5081_; uint8_t v___x_5082_; 
v_a_5078_ = lean_ctor_get(v___x_5077_, 0);
lean_inc(v_a_5078_);
lean_dec_ref_known(v___x_5077_, 1);
v_fst_5079_ = lean_ctor_get(v_a_5078_, 0);
lean_inc(v_fst_5079_);
v_snd_5080_ = lean_ctor_get(v_a_5078_, 1);
lean_inc(v_snd_5080_);
lean_dec(v_a_5078_);
v___x_5081_ = lean_array_get_size(v_snd_5080_);
v___x_5082_ = lean_nat_dec_lt(v___x_5043_, v___x_5081_);
lean_dec(v___x_5043_);
if (v___x_5082_ == 0)
{
lean_object* v___x_5083_; 
lean_dec(v_snd_5080_);
v___x_5083_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5044_, v___y_5068_, v_fst_5079_, v_snd_5045_, v___y_5066_, v_fst_5042_, v___y_5070_, v___y_5073_, v___y_5074_, v___y_5072_, v___y_5071_, v___y_5067_, v___y_5069_);
return v___x_5083_;
}
else
{
uint8_t v___x_5084_; 
v___x_5084_ = lean_nat_dec_le(v___x_5081_, v___x_5081_);
if (v___x_5084_ == 0)
{
if (v___x_5082_ == 0)
{
lean_object* v___x_5085_; 
lean_dec(v_snd_5080_);
v___x_5085_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5044_, v___y_5068_, v_fst_5079_, v_snd_5045_, v___y_5066_, v_fst_5042_, v___y_5070_, v___y_5073_, v___y_5074_, v___y_5072_, v___y_5071_, v___y_5067_, v___y_5069_);
return v___x_5085_;
}
else
{
size_t v___x_5086_; size_t v___x_5087_; lean_object* v___x_5088_; lean_object* v___x_5089_; 
v___x_5086_ = ((size_t)0ULL);
v___x_5087_ = lean_usize_of_nat(v___x_5081_);
v___x_5088_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__3(v_snd_5080_, v___x_5086_, v___x_5087_, v_snd_5045_);
lean_dec(v_snd_5080_);
v___x_5089_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5044_, v___y_5068_, v_fst_5079_, v___x_5088_, v___y_5066_, v_fst_5042_, v___y_5070_, v___y_5073_, v___y_5074_, v___y_5072_, v___y_5071_, v___y_5067_, v___y_5069_);
return v___x_5089_;
}
}
else
{
size_t v___x_5090_; size_t v___x_5091_; lean_object* v___x_5092_; lean_object* v___x_5093_; 
v___x_5090_ = ((size_t)0ULL);
v___x_5091_ = lean_usize_of_nat(v___x_5081_);
v___x_5092_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__3(v_snd_5080_, v___x_5090_, v___x_5091_, v_snd_5045_);
lean_dec(v_snd_5080_);
v___x_5093_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5044_, v___y_5068_, v_fst_5079_, v___x_5092_, v___y_5066_, v_fst_5042_, v___y_5070_, v___y_5073_, v___y_5074_, v___y_5072_, v___y_5071_, v___y_5067_, v___y_5069_);
return v___x_5093_;
}
}
}
else
{
lean_object* v_a_5094_; lean_object* v___x_5096_; uint8_t v_isShared_5097_; uint8_t v_isSharedCheck_5101_; 
lean_dec_ref(v___y_5068_);
lean_dec_ref(v___y_5066_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5094_ = lean_ctor_get(v___x_5077_, 0);
v_isSharedCheck_5101_ = !lean_is_exclusive(v___x_5077_);
if (v_isSharedCheck_5101_ == 0)
{
v___x_5096_ = v___x_5077_;
v_isShared_5097_ = v_isSharedCheck_5101_;
goto v_resetjp_5095_;
}
else
{
lean_inc(v_a_5094_);
lean_dec(v___x_5077_);
v___x_5096_ = lean_box(0);
v_isShared_5097_ = v_isSharedCheck_5101_;
goto v_resetjp_5095_;
}
v_resetjp_5095_:
{
lean_object* v___x_5099_; 
if (v_isShared_5097_ == 0)
{
v___x_5099_ = v___x_5096_;
goto v_reusejp_5098_;
}
else
{
lean_object* v_reuseFailAlloc_5100_; 
v_reuseFailAlloc_5100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5100_, 0, v_a_5094_);
v___x_5099_ = v_reuseFailAlloc_5100_;
goto v_reusejp_5098_;
}
v_reusejp_5098_:
{
return v___x_5099_;
}
}
}
}
v___jp_5102_:
{
lean_object* v___x_5114_; lean_object* v___x_5115_; 
v___x_5114_ = lean_io_mono_nanos_now();
lean_inc(v_fst_5046_);
v___x_5115_ = l_Lean_FVarId_getDecl___redArg(v_fst_5046_, v___y_5111_, v___y_5105_, v___y_5108_);
if (lean_obj_tag(v___x_5115_) == 0)
{
lean_object* v_a_5116_; lean_object* v___x_5117_; 
v_a_5116_ = lean_ctor_get(v___x_5115_, 0);
lean_inc(v_a_5116_);
lean_dec_ref_known(v___x_5115_, 1);
lean_inc_ref(v_rs_5044_);
v___x_5117_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(v_rs_5044_, v_a_5116_, v___y_5113_, v___y_5111_, v___y_5110_, v___y_5105_, v___y_5108_);
if (lean_obj_tag(v___x_5117_) == 0)
{
lean_object* v_a_5118_; lean_object* v___x_5119_; lean_object* v___x_5120_; lean_object* v_stats_5121_; lean_object* v_rulePatternCache_5122_; lean_object* v___x_5124_; uint8_t v_isShared_5125_; uint8_t v_isSharedCheck_5150_; 
v_a_5118_ = lean_ctor_get(v___x_5117_, 0);
lean_inc(v_a_5118_);
lean_dec_ref_known(v___x_5117_, 1);
v___x_5119_ = lean_io_mono_nanos_now();
v___x_5120_ = lean_st_ref_take(v___y_5113_);
v_stats_5121_ = lean_ctor_get(v___x_5120_, 1);
v_rulePatternCache_5122_ = lean_ctor_get(v___x_5120_, 0);
v_isSharedCheck_5150_ = !lean_is_exclusive(v___x_5120_);
if (v_isSharedCheck_5150_ == 0)
{
v___x_5124_ = v___x_5120_;
v_isShared_5125_ = v_isSharedCheck_5150_;
goto v_resetjp_5123_;
}
else
{
lean_inc(v_stats_5121_);
lean_inc(v_rulePatternCache_5122_);
lean_dec(v___x_5120_);
v___x_5124_ = lean_box(0);
v_isShared_5125_ = v_isSharedCheck_5150_;
goto v_resetjp_5123_;
}
v_resetjp_5123_:
{
lean_object* v_total_5126_; lean_object* v_configParsing_5127_; lean_object* v_ruleSetConstruction_5128_; lean_object* v_search_5129_; lean_object* v_ruleSelection_5130_; lean_object* v_script_5131_; lean_object* v_forwardState_5132_; lean_object* v_scriptGenerated_5133_; lean_object* v_ruleStats_5134_; lean_object* v_goalStats_5135_; lean_object* v___x_5137_; uint8_t v_isShared_5138_; uint8_t v_isSharedCheck_5149_; 
v_total_5126_ = lean_ctor_get(v_stats_5121_, 0);
v_configParsing_5127_ = lean_ctor_get(v_stats_5121_, 1);
v_ruleSetConstruction_5128_ = lean_ctor_get(v_stats_5121_, 2);
v_search_5129_ = lean_ctor_get(v_stats_5121_, 3);
v_ruleSelection_5130_ = lean_ctor_get(v_stats_5121_, 4);
v_script_5131_ = lean_ctor_get(v_stats_5121_, 5);
v_forwardState_5132_ = lean_ctor_get(v_stats_5121_, 6);
v_scriptGenerated_5133_ = lean_ctor_get(v_stats_5121_, 7);
v_ruleStats_5134_ = lean_ctor_get(v_stats_5121_, 8);
v_goalStats_5135_ = lean_ctor_get(v_stats_5121_, 9);
v_isSharedCheck_5149_ = !lean_is_exclusive(v_stats_5121_);
if (v_isSharedCheck_5149_ == 0)
{
v___x_5137_ = v_stats_5121_;
v_isShared_5138_ = v_isSharedCheck_5149_;
goto v_resetjp_5136_;
}
else
{
lean_inc(v_goalStats_5135_);
lean_inc(v_ruleStats_5134_);
lean_inc(v_scriptGenerated_5133_);
lean_inc(v_forwardState_5132_);
lean_inc(v_script_5131_);
lean_inc(v_ruleSelection_5130_);
lean_inc(v_search_5129_);
lean_inc(v_ruleSetConstruction_5128_);
lean_inc(v_configParsing_5127_);
lean_inc(v_total_5126_);
lean_dec(v_stats_5121_);
v___x_5137_ = lean_box(0);
v_isShared_5138_ = v_isSharedCheck_5149_;
goto v_resetjp_5136_;
}
v_resetjp_5136_:
{
lean_object* v___x_5139_; lean_object* v___x_5140_; lean_object* v___x_5142_; 
v___x_5139_ = lean_nat_sub(v___x_5119_, v___x_5114_);
lean_dec(v___x_5114_);
lean_dec(v___x_5119_);
v___x_5140_ = lean_nat_add(v_forwardState_5132_, v___x_5139_);
lean_dec(v___x_5139_);
lean_dec(v_forwardState_5132_);
if (v_isShared_5138_ == 0)
{
lean_ctor_set(v___x_5137_, 6, v___x_5140_);
v___x_5142_ = v___x_5137_;
goto v_reusejp_5141_;
}
else
{
lean_object* v_reuseFailAlloc_5148_; 
v_reuseFailAlloc_5148_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_5148_, 0, v_total_5126_);
lean_ctor_set(v_reuseFailAlloc_5148_, 1, v_configParsing_5127_);
lean_ctor_set(v_reuseFailAlloc_5148_, 2, v_ruleSetConstruction_5128_);
lean_ctor_set(v_reuseFailAlloc_5148_, 3, v_search_5129_);
lean_ctor_set(v_reuseFailAlloc_5148_, 4, v_ruleSelection_5130_);
lean_ctor_set(v_reuseFailAlloc_5148_, 5, v_script_5131_);
lean_ctor_set(v_reuseFailAlloc_5148_, 6, v___x_5140_);
lean_ctor_set(v_reuseFailAlloc_5148_, 7, v_scriptGenerated_5133_);
lean_ctor_set(v_reuseFailAlloc_5148_, 8, v_ruleStats_5134_);
lean_ctor_set(v_reuseFailAlloc_5148_, 9, v_goalStats_5135_);
v___x_5142_ = v_reuseFailAlloc_5148_;
goto v_reusejp_5141_;
}
v_reusejp_5141_:
{
lean_object* v___x_5144_; 
if (v_isShared_5125_ == 0)
{
lean_ctor_set(v___x_5124_, 1, v___x_5142_);
v___x_5144_ = v___x_5124_;
goto v_reusejp_5143_;
}
else
{
lean_object* v_reuseFailAlloc_5147_; 
v_reuseFailAlloc_5147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5147_, 0, v_rulePatternCache_5122_);
lean_ctor_set(v_reuseFailAlloc_5147_, 1, v___x_5142_);
v___x_5144_ = v_reuseFailAlloc_5147_;
goto v_reusejp_5143_;
}
v_reusejp_5143_:
{
lean_object* v___x_5145_; lean_object* v___x_5146_; 
v___x_5145_ = lean_st_ref_set(v___y_5113_, v___x_5144_);
v___x_5146_ = lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(v_fst_5046_, v___y_5106_, v_a_5118_, v___y_5103_);
lean_dec(v_a_5118_);
v___y_5066_ = v___y_5104_;
v___y_5067_ = v___y_5105_;
v___y_5068_ = v___y_5107_;
v___y_5069_ = v___y_5108_;
v___y_5070_ = v___y_5109_;
v___y_5071_ = v___y_5110_;
v___y_5072_ = v___y_5111_;
v___y_5073_ = v___y_5112_;
v___y_5074_ = v___y_5113_;
v_a_5075_ = v___x_5146_;
goto v___jp_5065_;
}
}
}
}
}
else
{
lean_object* v_a_5151_; lean_object* v___x_5153_; uint8_t v_isShared_5154_; uint8_t v_isSharedCheck_5158_; 
lean_dec(v___x_5114_);
lean_dec_ref(v___y_5107_);
lean_dec_ref(v___y_5106_);
lean_dec_ref(v___y_5104_);
lean_dec_ref(v___y_5103_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5151_ = lean_ctor_get(v___x_5117_, 0);
v_isSharedCheck_5158_ = !lean_is_exclusive(v___x_5117_);
if (v_isSharedCheck_5158_ == 0)
{
v___x_5153_ = v___x_5117_;
v_isShared_5154_ = v_isSharedCheck_5158_;
goto v_resetjp_5152_;
}
else
{
lean_inc(v_a_5151_);
lean_dec(v___x_5117_);
v___x_5153_ = lean_box(0);
v_isShared_5154_ = v_isSharedCheck_5158_;
goto v_resetjp_5152_;
}
v_resetjp_5152_:
{
lean_object* v___x_5156_; 
if (v_isShared_5154_ == 0)
{
v___x_5156_ = v___x_5153_;
goto v_reusejp_5155_;
}
else
{
lean_object* v_reuseFailAlloc_5157_; 
v_reuseFailAlloc_5157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5157_, 0, v_a_5151_);
v___x_5156_ = v_reuseFailAlloc_5157_;
goto v_reusejp_5155_;
}
v_reusejp_5155_:
{
return v___x_5156_;
}
}
}
}
else
{
lean_object* v_a_5159_; lean_object* v___x_5161_; uint8_t v_isShared_5162_; uint8_t v_isSharedCheck_5166_; 
lean_dec(v___x_5114_);
lean_dec_ref(v___y_5107_);
lean_dec_ref(v___y_5106_);
lean_dec_ref(v___y_5104_);
lean_dec_ref(v___y_5103_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5159_ = lean_ctor_get(v___x_5115_, 0);
v_isSharedCheck_5166_ = !lean_is_exclusive(v___x_5115_);
if (v_isSharedCheck_5166_ == 0)
{
v___x_5161_ = v___x_5115_;
v_isShared_5162_ = v_isSharedCheck_5166_;
goto v_resetjp_5160_;
}
else
{
lean_inc(v_a_5159_);
lean_dec(v___x_5115_);
v___x_5161_ = lean_box(0);
v_isShared_5162_ = v_isSharedCheck_5166_;
goto v_resetjp_5160_;
}
v_resetjp_5160_:
{
lean_object* v___x_5164_; 
if (v_isShared_5162_ == 0)
{
v___x_5164_ = v___x_5161_;
goto v_reusejp_5163_;
}
else
{
lean_object* v_reuseFailAlloc_5165_; 
v_reuseFailAlloc_5165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5165_, 0, v_a_5159_);
v___x_5164_ = v_reuseFailAlloc_5165_;
goto v_reusejp_5163_;
}
v_reusejp_5163_:
{
return v___x_5164_;
}
}
}
}
v___jp_5167_:
{
if (v_a_5179_ == 0)
{
lean_object* v___x_5180_; 
lean_inc(v_fst_5046_);
v___x_5180_ = l_Lean_FVarId_getDecl___redArg(v_fst_5046_, v___y_5176_, v___y_5170_, v___y_5173_);
if (lean_obj_tag(v___x_5180_) == 0)
{
lean_object* v_a_5181_; lean_object* v___x_5182_; 
v_a_5181_ = lean_ctor_get(v___x_5180_, 0);
lean_inc(v_a_5181_);
lean_dec_ref_known(v___x_5180_, 1);
lean_inc_ref(v_rs_5044_);
v___x_5182_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(v_rs_5044_, v_a_5181_, v___y_5178_, v___y_5176_, v___y_5175_, v___y_5170_, v___y_5173_);
if (lean_obj_tag(v___x_5182_) == 0)
{
lean_object* v_a_5183_; lean_object* v___x_5184_; 
v_a_5183_ = lean_ctor_get(v___x_5182_, 0);
lean_inc(v_a_5183_);
lean_dec_ref_known(v___x_5182_, 1);
v___x_5184_ = lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(v_fst_5046_, v___y_5171_, v_a_5183_, v___y_5168_);
lean_dec(v_a_5183_);
v___y_5066_ = v___y_5169_;
v___y_5067_ = v___y_5170_;
v___y_5068_ = v___y_5172_;
v___y_5069_ = v___y_5173_;
v___y_5070_ = v___y_5174_;
v___y_5071_ = v___y_5175_;
v___y_5072_ = v___y_5176_;
v___y_5073_ = v___y_5177_;
v___y_5074_ = v___y_5178_;
v_a_5075_ = v___x_5184_;
goto v___jp_5065_;
}
else
{
lean_object* v_a_5185_; lean_object* v___x_5187_; uint8_t v_isShared_5188_; uint8_t v_isSharedCheck_5192_; 
lean_dec_ref(v___y_5172_);
lean_dec_ref(v___y_5171_);
lean_dec_ref(v___y_5169_);
lean_dec_ref(v___y_5168_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5185_ = lean_ctor_get(v___x_5182_, 0);
v_isSharedCheck_5192_ = !lean_is_exclusive(v___x_5182_);
if (v_isSharedCheck_5192_ == 0)
{
v___x_5187_ = v___x_5182_;
v_isShared_5188_ = v_isSharedCheck_5192_;
goto v_resetjp_5186_;
}
else
{
lean_inc(v_a_5185_);
lean_dec(v___x_5182_);
v___x_5187_ = lean_box(0);
v_isShared_5188_ = v_isSharedCheck_5192_;
goto v_resetjp_5186_;
}
v_resetjp_5186_:
{
lean_object* v___x_5190_; 
if (v_isShared_5188_ == 0)
{
v___x_5190_ = v___x_5187_;
goto v_reusejp_5189_;
}
else
{
lean_object* v_reuseFailAlloc_5191_; 
v_reuseFailAlloc_5191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5191_, 0, v_a_5185_);
v___x_5190_ = v_reuseFailAlloc_5191_;
goto v_reusejp_5189_;
}
v_reusejp_5189_:
{
return v___x_5190_;
}
}
}
}
else
{
lean_object* v_a_5193_; lean_object* v___x_5195_; uint8_t v_isShared_5196_; uint8_t v_isSharedCheck_5200_; 
lean_dec_ref(v___y_5172_);
lean_dec_ref(v___y_5171_);
lean_dec_ref(v___y_5169_);
lean_dec_ref(v___y_5168_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5193_ = lean_ctor_get(v___x_5180_, 0);
v_isSharedCheck_5200_ = !lean_is_exclusive(v___x_5180_);
if (v_isSharedCheck_5200_ == 0)
{
v___x_5195_ = v___x_5180_;
v_isShared_5196_ = v_isSharedCheck_5200_;
goto v_resetjp_5194_;
}
else
{
lean_inc(v_a_5193_);
lean_dec(v___x_5180_);
v___x_5195_ = lean_box(0);
v_isShared_5196_ = v_isSharedCheck_5200_;
goto v_resetjp_5194_;
}
v_resetjp_5194_:
{
lean_object* v___x_5198_; 
if (v_isShared_5196_ == 0)
{
v___x_5198_ = v___x_5195_;
goto v_reusejp_5197_;
}
else
{
lean_object* v_reuseFailAlloc_5199_; 
v_reuseFailAlloc_5199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5199_, 0, v_a_5193_);
v___x_5198_ = v_reuseFailAlloc_5199_;
goto v_reusejp_5197_;
}
v_reusejp_5197_:
{
return v___x_5198_;
}
}
}
}
else
{
v___y_5103_ = v___y_5168_;
v___y_5104_ = v___y_5169_;
v___y_5105_ = v___y_5170_;
v___y_5106_ = v___y_5171_;
v___y_5107_ = v___y_5172_;
v___y_5108_ = v___y_5173_;
v___y_5109_ = v___y_5174_;
v___y_5110_ = v___y_5175_;
v___y_5111_ = v___y_5176_;
v___y_5112_ = v___y_5177_;
v___y_5113_ = v___y_5178_;
goto v___jp_5102_;
}
}
v___jp_5201_:
{
if (lean_obj_tag(v___y_5213_) == 0)
{
lean_object* v_a_5214_; uint8_t v___x_5215_; 
v_a_5214_ = lean_ctor_get(v___y_5213_, 0);
lean_inc(v_a_5214_);
lean_dec_ref_known(v___y_5213_, 1);
v___x_5215_ = lean_unbox(v_a_5214_);
lean_dec(v_a_5214_);
v___y_5168_ = v___y_5202_;
v___y_5169_ = v___y_5204_;
v___y_5170_ = v___y_5203_;
v___y_5171_ = v___y_5205_;
v___y_5172_ = v___y_5206_;
v___y_5173_ = v___y_5207_;
v___y_5174_ = v___y_5208_;
v___y_5175_ = v___y_5210_;
v___y_5176_ = v___y_5209_;
v___y_5177_ = v___y_5212_;
v___y_5178_ = v___y_5211_;
v_a_5179_ = v___x_5215_;
goto v___jp_5167_;
}
else
{
lean_object* v_a_5216_; lean_object* v___x_5218_; uint8_t v_isShared_5219_; uint8_t v_isSharedCheck_5223_; 
lean_dec_ref(v___y_5206_);
lean_dec_ref(v___y_5205_);
lean_dec_ref(v___y_5204_);
lean_dec_ref(v___y_5202_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5216_ = lean_ctor_get(v___y_5213_, 0);
v_isSharedCheck_5223_ = !lean_is_exclusive(v___y_5213_);
if (v_isSharedCheck_5223_ == 0)
{
v___x_5218_ = v___y_5213_;
v_isShared_5219_ = v_isSharedCheck_5223_;
goto v_resetjp_5217_;
}
else
{
lean_inc(v_a_5216_);
lean_dec(v___y_5213_);
v___x_5218_ = lean_box(0);
v_isShared_5219_ = v_isSharedCheck_5223_;
goto v_resetjp_5217_;
}
v_resetjp_5217_:
{
lean_object* v___x_5221_; 
if (v_isShared_5219_ == 0)
{
v___x_5221_ = v___x_5218_;
goto v_reusejp_5220_;
}
else
{
lean_object* v_reuseFailAlloc_5222_; 
v_reuseFailAlloc_5222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5222_, 0, v_a_5216_);
v___x_5221_ = v_reuseFailAlloc_5222_;
goto v_reusejp_5220_;
}
v_reusejp_5220_:
{
return v___x_5221_;
}
}
}
}
v___jp_5227_:
{
lean_object* v_options_5239_; uint8_t v___x_5240_; 
v_options_5239_ = lean_ctor_get(v___y_5229_, 2);
v___x_5240_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_5239_, v___x_5226_);
if (v___x_5240_ == 0)
{
lean_object* v___x_5241_; lean_object* v___x_5242_; 
v___x_5241_ = lp_aesop_Aesop_TraceOption_stats;
v___x_5242_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_5241_, v___y_5229_);
if (lean_obj_tag(v___x_5242_) == 0)
{
lean_object* v_a_5243_; uint8_t v___x_5244_; 
v_a_5243_ = lean_ctor_get(v___x_5242_, 0);
lean_inc(v_a_5243_);
v___x_5244_ = lean_unbox(v_a_5243_);
lean_dec(v_a_5243_);
if (v___x_5244_ == 0)
{
lean_object* v___x_5245_; lean_object* v___x_5246_; lean_object* v___x_5247_; uint8_t v___x_5248_; 
v___x_5245_ = lp_aesop_Aesop_aesop_stats_file;
v___x_5246_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_5239_, v___x_5245_);
v___x_5247_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_5248_ = lean_string_dec_eq(v___x_5246_, v___x_5247_);
lean_dec_ref(v___x_5246_);
if (v___x_5248_ == 0)
{
lean_dec_ref_known(v___x_5242_, 1);
v___y_5103_ = v___y_5228_;
v___y_5104_ = v___y_5230_;
v___y_5105_ = v___y_5229_;
v___y_5106_ = v_a_5238_;
v___y_5107_ = v___y_5231_;
v___y_5108_ = v___y_5232_;
v___y_5109_ = v___y_5233_;
v___y_5110_ = v___y_5235_;
v___y_5111_ = v___y_5234_;
v___y_5112_ = v___y_5237_;
v___y_5113_ = v___y_5236_;
goto v___jp_5102_;
}
else
{
v___y_5202_ = v___y_5228_;
v___y_5203_ = v___y_5229_;
v___y_5204_ = v___y_5230_;
v___y_5205_ = v_a_5238_;
v___y_5206_ = v___y_5231_;
v___y_5207_ = v___y_5232_;
v___y_5208_ = v___y_5233_;
v___y_5209_ = v___y_5234_;
v___y_5210_ = v___y_5235_;
v___y_5211_ = v___y_5236_;
v___y_5212_ = v___y_5237_;
v___y_5213_ = v___x_5242_;
goto v___jp_5201_;
}
}
else
{
v___y_5202_ = v___y_5228_;
v___y_5203_ = v___y_5229_;
v___y_5204_ = v___y_5230_;
v___y_5205_ = v_a_5238_;
v___y_5206_ = v___y_5231_;
v___y_5207_ = v___y_5232_;
v___y_5208_ = v___y_5233_;
v___y_5209_ = v___y_5234_;
v___y_5210_ = v___y_5235_;
v___y_5211_ = v___y_5236_;
v___y_5212_ = v___y_5237_;
v___y_5213_ = v___x_5242_;
goto v___jp_5201_;
}
}
else
{
v___y_5202_ = v___y_5228_;
v___y_5203_ = v___y_5229_;
v___y_5204_ = v___y_5230_;
v___y_5205_ = v_a_5238_;
v___y_5206_ = v___y_5231_;
v___y_5207_ = v___y_5232_;
v___y_5208_ = v___y_5233_;
v___y_5209_ = v___y_5234_;
v___y_5210_ = v___y_5235_;
v___y_5211_ = v___y_5236_;
v___y_5212_ = v___y_5237_;
v___y_5213_ = v___x_5242_;
goto v___jp_5201_;
}
}
else
{
v___y_5168_ = v___y_5228_;
v___y_5169_ = v___y_5230_;
v___y_5170_ = v___y_5229_;
v___y_5171_ = v_a_5238_;
v___y_5172_ = v___y_5231_;
v___y_5173_ = v___y_5232_;
v___y_5174_ = v___y_5233_;
v___y_5175_ = v___y_5235_;
v___y_5176_ = v___y_5234_;
v___y_5177_ = v___y_5237_;
v___y_5178_ = v___y_5236_;
v_a_5179_ = v___x_5240_;
goto v___jp_5167_;
}
}
v___jp_5249_:
{
lean_object* v___x_5262_; lean_object* v___x_5263_; lean_object* v___f_5264_; lean_object* v___x_5265_; 
v___x_5262_ = lean_io_mono_nanos_now();
v___x_5263_ = lean_box(v_a_5261_);
v___f_5264_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__3___boxed), 2, 1);
lean_closure_set(v___f_5264_, 0, v___x_5263_);
lean_inc_ref(v_rs_5044_);
v___x_5265_ = lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(v_rs_5044_, v___y_5251_, v___f_5264_, v___y_5258_, v___y_5257_, v___y_5253_, v___y_5255_);
if (lean_obj_tag(v___x_5265_) == 0)
{
lean_object* v_a_5266_; lean_object* v___x_5267_; lean_object* v___x_5268_; lean_object* v_stats_5269_; lean_object* v_rulePatternCache_5270_; lean_object* v___x_5272_; uint8_t v_isShared_5273_; uint8_t v_isSharedCheck_5297_; 
v_a_5266_ = lean_ctor_get(v___x_5265_, 0);
lean_inc(v_a_5266_);
lean_dec_ref_known(v___x_5265_, 1);
v___x_5267_ = lean_io_mono_nanos_now();
v___x_5268_ = lean_st_ref_take(v___y_5260_);
v_stats_5269_ = lean_ctor_get(v___x_5268_, 1);
v_rulePatternCache_5270_ = lean_ctor_get(v___x_5268_, 0);
v_isSharedCheck_5297_ = !lean_is_exclusive(v___x_5268_);
if (v_isSharedCheck_5297_ == 0)
{
v___x_5272_ = v___x_5268_;
v_isShared_5273_ = v_isSharedCheck_5297_;
goto v_resetjp_5271_;
}
else
{
lean_inc(v_stats_5269_);
lean_inc(v_rulePatternCache_5270_);
lean_dec(v___x_5268_);
v___x_5272_ = lean_box(0);
v_isShared_5273_ = v_isSharedCheck_5297_;
goto v_resetjp_5271_;
}
v_resetjp_5271_:
{
lean_object* v_total_5274_; lean_object* v_configParsing_5275_; lean_object* v_ruleSetConstruction_5276_; lean_object* v_search_5277_; lean_object* v_ruleSelection_5278_; lean_object* v_script_5279_; lean_object* v_forwardState_5280_; lean_object* v_scriptGenerated_5281_; lean_object* v_ruleStats_5282_; lean_object* v_goalStats_5283_; lean_object* v___x_5285_; uint8_t v_isShared_5286_; uint8_t v_isSharedCheck_5296_; 
v_total_5274_ = lean_ctor_get(v_stats_5269_, 0);
v_configParsing_5275_ = lean_ctor_get(v_stats_5269_, 1);
v_ruleSetConstruction_5276_ = lean_ctor_get(v_stats_5269_, 2);
v_search_5277_ = lean_ctor_get(v_stats_5269_, 3);
v_ruleSelection_5278_ = lean_ctor_get(v_stats_5269_, 4);
v_script_5279_ = lean_ctor_get(v_stats_5269_, 5);
v_forwardState_5280_ = lean_ctor_get(v_stats_5269_, 6);
v_scriptGenerated_5281_ = lean_ctor_get(v_stats_5269_, 7);
v_ruleStats_5282_ = lean_ctor_get(v_stats_5269_, 8);
v_goalStats_5283_ = lean_ctor_get(v_stats_5269_, 9);
v_isSharedCheck_5296_ = !lean_is_exclusive(v_stats_5269_);
if (v_isSharedCheck_5296_ == 0)
{
v___x_5285_ = v_stats_5269_;
v_isShared_5286_ = v_isSharedCheck_5296_;
goto v_resetjp_5284_;
}
else
{
lean_inc(v_goalStats_5283_);
lean_inc(v_ruleStats_5282_);
lean_inc(v_scriptGenerated_5281_);
lean_inc(v_forwardState_5280_);
lean_inc(v_script_5279_);
lean_inc(v_ruleSelection_5278_);
lean_inc(v_search_5277_);
lean_inc(v_ruleSetConstruction_5276_);
lean_inc(v_configParsing_5275_);
lean_inc(v_total_5274_);
lean_dec(v_stats_5269_);
v___x_5285_ = lean_box(0);
v_isShared_5286_ = v_isSharedCheck_5296_;
goto v_resetjp_5284_;
}
v_resetjp_5284_:
{
lean_object* v___x_5287_; lean_object* v___x_5288_; lean_object* v___x_5290_; 
v___x_5287_ = lean_nat_sub(v___x_5267_, v___x_5262_);
lean_dec(v___x_5262_);
lean_dec(v___x_5267_);
v___x_5288_ = lean_nat_add(v_ruleSelection_5278_, v___x_5287_);
lean_dec(v___x_5287_);
lean_dec(v_ruleSelection_5278_);
if (v_isShared_5286_ == 0)
{
lean_ctor_set(v___x_5285_, 4, v___x_5288_);
v___x_5290_ = v___x_5285_;
goto v_reusejp_5289_;
}
else
{
lean_object* v_reuseFailAlloc_5295_; 
v_reuseFailAlloc_5295_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_5295_, 0, v_total_5274_);
lean_ctor_set(v_reuseFailAlloc_5295_, 1, v_configParsing_5275_);
lean_ctor_set(v_reuseFailAlloc_5295_, 2, v_ruleSetConstruction_5276_);
lean_ctor_set(v_reuseFailAlloc_5295_, 3, v_search_5277_);
lean_ctor_set(v_reuseFailAlloc_5295_, 4, v___x_5288_);
lean_ctor_set(v_reuseFailAlloc_5295_, 5, v_script_5279_);
lean_ctor_set(v_reuseFailAlloc_5295_, 6, v_forwardState_5280_);
lean_ctor_set(v_reuseFailAlloc_5295_, 7, v_scriptGenerated_5281_);
lean_ctor_set(v_reuseFailAlloc_5295_, 8, v_ruleStats_5282_);
lean_ctor_set(v_reuseFailAlloc_5295_, 9, v_goalStats_5283_);
v___x_5290_ = v_reuseFailAlloc_5295_;
goto v_reusejp_5289_;
}
v_reusejp_5289_:
{
lean_object* v___x_5292_; 
if (v_isShared_5273_ == 0)
{
lean_ctor_set(v___x_5272_, 1, v___x_5290_);
v___x_5292_ = v___x_5272_;
goto v_reusejp_5291_;
}
else
{
lean_object* v_reuseFailAlloc_5294_; 
v_reuseFailAlloc_5294_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5294_, 0, v_rulePatternCache_5270_);
lean_ctor_set(v_reuseFailAlloc_5294_, 1, v___x_5290_);
v___x_5292_ = v_reuseFailAlloc_5294_;
goto v_reusejp_5291_;
}
v_reusejp_5291_:
{
lean_object* v___x_5293_; 
v___x_5293_ = lean_st_ref_set(v___y_5260_, v___x_5292_);
v___y_5228_ = v___y_5250_;
v___y_5229_ = v___y_5253_;
v___y_5230_ = v___y_5252_;
v___y_5231_ = v___y_5254_;
v___y_5232_ = v___y_5255_;
v___y_5233_ = v___y_5256_;
v___y_5234_ = v___y_5258_;
v___y_5235_ = v___y_5257_;
v___y_5236_ = v___y_5260_;
v___y_5237_ = v___y_5259_;
v_a_5238_ = v_a_5266_;
goto v___jp_5227_;
}
}
}
}
}
else
{
lean_object* v_a_5298_; lean_object* v___x_5300_; uint8_t v_isShared_5301_; uint8_t v_isSharedCheck_5305_; 
lean_dec(v___x_5262_);
lean_dec_ref(v___y_5254_);
lean_dec_ref(v___y_5252_);
lean_dec_ref(v___y_5250_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5298_ = lean_ctor_get(v___x_5265_, 0);
v_isSharedCheck_5305_ = !lean_is_exclusive(v___x_5265_);
if (v_isSharedCheck_5305_ == 0)
{
v___x_5300_ = v___x_5265_;
v_isShared_5301_ = v_isSharedCheck_5305_;
goto v_resetjp_5299_;
}
else
{
lean_inc(v_a_5298_);
lean_dec(v___x_5265_);
v___x_5300_ = lean_box(0);
v_isShared_5301_ = v_isSharedCheck_5305_;
goto v_resetjp_5299_;
}
v_resetjp_5299_:
{
lean_object* v___x_5303_; 
if (v_isShared_5301_ == 0)
{
v___x_5303_ = v___x_5300_;
goto v_reusejp_5302_;
}
else
{
lean_object* v_reuseFailAlloc_5304_; 
v_reuseFailAlloc_5304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5304_, 0, v_a_5298_);
v___x_5303_ = v_reuseFailAlloc_5304_;
goto v_reusejp_5302_;
}
v_reusejp_5302_:
{
return v___x_5303_;
}
}
}
}
v___jp_5306_:
{
if (v_a_5318_ == 0)
{
lean_object* v___x_5319_; 
lean_inc_ref(v_rs_5044_);
v___x_5319_ = lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(v_rs_5044_, v___y_5308_, v___f_5047_, v___y_5315_, v___y_5314_, v___y_5310_, v___y_5312_);
if (lean_obj_tag(v___x_5319_) == 0)
{
lean_object* v_a_5320_; 
v_a_5320_ = lean_ctor_get(v___x_5319_, 0);
lean_inc(v_a_5320_);
lean_dec_ref_known(v___x_5319_, 1);
v___y_5228_ = v___y_5307_;
v___y_5229_ = v___y_5310_;
v___y_5230_ = v___y_5309_;
v___y_5231_ = v___y_5311_;
v___y_5232_ = v___y_5312_;
v___y_5233_ = v___y_5313_;
v___y_5234_ = v___y_5315_;
v___y_5235_ = v___y_5314_;
v___y_5236_ = v___y_5317_;
v___y_5237_ = v___y_5316_;
v_a_5238_ = v_a_5320_;
goto v___jp_5227_;
}
else
{
lean_object* v_a_5321_; lean_object* v___x_5323_; uint8_t v_isShared_5324_; uint8_t v_isSharedCheck_5328_; 
lean_dec_ref(v___y_5311_);
lean_dec_ref(v___y_5309_);
lean_dec_ref(v___y_5307_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5321_ = lean_ctor_get(v___x_5319_, 0);
v_isSharedCheck_5328_ = !lean_is_exclusive(v___x_5319_);
if (v_isSharedCheck_5328_ == 0)
{
v___x_5323_ = v___x_5319_;
v_isShared_5324_ = v_isSharedCheck_5328_;
goto v_resetjp_5322_;
}
else
{
lean_inc(v_a_5321_);
lean_dec(v___x_5319_);
v___x_5323_ = lean_box(0);
v_isShared_5324_ = v_isSharedCheck_5328_;
goto v_resetjp_5322_;
}
v_resetjp_5322_:
{
lean_object* v___x_5326_; 
if (v_isShared_5324_ == 0)
{
v___x_5326_ = v___x_5323_;
goto v_reusejp_5325_;
}
else
{
lean_object* v_reuseFailAlloc_5327_; 
v_reuseFailAlloc_5327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5327_, 0, v_a_5321_);
v___x_5326_ = v_reuseFailAlloc_5327_;
goto v_reusejp_5325_;
}
v_reusejp_5325_:
{
return v___x_5326_;
}
}
}
}
else
{
lean_dec_ref(v___f_5047_);
v___y_5250_ = v___y_5307_;
v___y_5251_ = v___y_5308_;
v___y_5252_ = v___y_5309_;
v___y_5253_ = v___y_5310_;
v___y_5254_ = v___y_5311_;
v___y_5255_ = v___y_5312_;
v___y_5256_ = v___y_5313_;
v___y_5257_ = v___y_5314_;
v___y_5258_ = v___y_5315_;
v___y_5259_ = v___y_5316_;
v___y_5260_ = v___y_5317_;
v_a_5261_ = v_a_5318_;
goto v___jp_5249_;
}
}
v___jp_5329_:
{
if (lean_obj_tag(v___y_5341_) == 0)
{
lean_object* v_a_5342_; uint8_t v___x_5343_; 
v_a_5342_ = lean_ctor_get(v___y_5341_, 0);
lean_inc(v_a_5342_);
lean_dec_ref_known(v___y_5341_, 1);
v___x_5343_ = lean_unbox(v_a_5342_);
lean_dec(v_a_5342_);
v___y_5307_ = v___y_5331_;
v___y_5308_ = v___y_5330_;
v___y_5309_ = v___y_5333_;
v___y_5310_ = v___y_5332_;
v___y_5311_ = v___y_5334_;
v___y_5312_ = v___y_5335_;
v___y_5313_ = v___y_5336_;
v___y_5314_ = v___y_5338_;
v___y_5315_ = v___y_5337_;
v___y_5316_ = v___y_5340_;
v___y_5317_ = v___y_5339_;
v_a_5318_ = v___x_5343_;
goto v___jp_5306_;
}
else
{
lean_object* v_a_5344_; lean_object* v___x_5346_; uint8_t v_isShared_5347_; uint8_t v_isSharedCheck_5351_; 
lean_dec_ref(v___y_5334_);
lean_dec_ref(v___y_5333_);
lean_dec_ref(v___y_5331_);
lean_dec_ref(v___y_5330_);
lean_dec_ref(v___f_5047_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5344_ = lean_ctor_get(v___y_5341_, 0);
v_isSharedCheck_5351_ = !lean_is_exclusive(v___y_5341_);
if (v_isSharedCheck_5351_ == 0)
{
v___x_5346_ = v___y_5341_;
v_isShared_5347_ = v_isSharedCheck_5351_;
goto v_resetjp_5345_;
}
else
{
lean_inc(v_a_5344_);
lean_dec(v___y_5341_);
v___x_5346_ = lean_box(0);
v_isShared_5347_ = v_isSharedCheck_5351_;
goto v_resetjp_5345_;
}
v_resetjp_5345_:
{
lean_object* v___x_5349_; 
if (v_isShared_5347_ == 0)
{
v___x_5349_ = v___x_5346_;
goto v_reusejp_5348_;
}
else
{
lean_object* v_reuseFailAlloc_5350_; 
v_reuseFailAlloc_5350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5350_, 0, v_a_5344_);
v___x_5349_ = v_reuseFailAlloc_5350_;
goto v_reusejp_5348_;
}
v_reusejp_5348_:
{
return v___x_5349_;
}
}
}
}
v___jp_5352_:
{
lean_object* v_options_5364_; uint8_t v___x_5365_; 
v_options_5364_ = lean_ctor_get(v___y_5356_, 2);
v___x_5365_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_5364_, v___x_5226_);
if (v___x_5365_ == 0)
{
lean_object* v___x_5366_; lean_object* v___x_5367_; 
v___x_5366_ = lp_aesop_Aesop_TraceOption_stats;
v___x_5367_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_5366_, v___y_5356_);
if (lean_obj_tag(v___x_5367_) == 0)
{
lean_object* v_a_5368_; uint8_t v___x_5369_; 
v_a_5368_ = lean_ctor_get(v___x_5367_, 0);
lean_inc(v_a_5368_);
v___x_5369_ = lean_unbox(v_a_5368_);
lean_dec(v_a_5368_);
if (v___x_5369_ == 0)
{
lean_object* v___x_5370_; lean_object* v___x_5371_; lean_object* v___x_5372_; uint8_t v___x_5373_; 
v___x_5370_ = lp_aesop_Aesop_aesop_stats_file;
v___x_5371_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_5364_, v___x_5370_);
v___x_5372_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_5373_ = lean_string_dec_eq(v___x_5371_, v___x_5372_);
lean_dec_ref(v___x_5371_);
if (v___x_5373_ == 0)
{
lean_dec_ref_known(v___x_5367_, 1);
lean_dec_ref(v___f_5047_);
v___y_5250_ = v___y_5354_;
v___y_5251_ = v___y_5353_;
v___y_5252_ = v___y_5355_;
v___y_5253_ = v___y_5356_;
v___y_5254_ = v___y_5357_;
v___y_5255_ = v___y_5358_;
v___y_5256_ = v___y_5359_;
v___y_5257_ = v___y_5361_;
v___y_5258_ = v___y_5360_;
v___y_5259_ = v___y_5363_;
v___y_5260_ = v___y_5362_;
v_a_5261_ = v___x_5048_;
goto v___jp_5249_;
}
else
{
v___y_5330_ = v___y_5353_;
v___y_5331_ = v___y_5354_;
v___y_5332_ = v___y_5356_;
v___y_5333_ = v___y_5355_;
v___y_5334_ = v___y_5357_;
v___y_5335_ = v___y_5358_;
v___y_5336_ = v___y_5359_;
v___y_5337_ = v___y_5360_;
v___y_5338_ = v___y_5361_;
v___y_5339_ = v___y_5362_;
v___y_5340_ = v___y_5363_;
v___y_5341_ = v___x_5367_;
goto v___jp_5329_;
}
}
else
{
v___y_5330_ = v___y_5353_;
v___y_5331_ = v___y_5354_;
v___y_5332_ = v___y_5356_;
v___y_5333_ = v___y_5355_;
v___y_5334_ = v___y_5357_;
v___y_5335_ = v___y_5358_;
v___y_5336_ = v___y_5359_;
v___y_5337_ = v___y_5360_;
v___y_5338_ = v___y_5361_;
v___y_5339_ = v___y_5362_;
v___y_5340_ = v___y_5363_;
v___y_5341_ = v___x_5367_;
goto v___jp_5329_;
}
}
else
{
v___y_5330_ = v___y_5353_;
v___y_5331_ = v___y_5354_;
v___y_5332_ = v___y_5356_;
v___y_5333_ = v___y_5355_;
v___y_5334_ = v___y_5357_;
v___y_5335_ = v___y_5358_;
v___y_5336_ = v___y_5359_;
v___y_5337_ = v___y_5360_;
v___y_5338_ = v___y_5361_;
v___y_5339_ = v___y_5362_;
v___y_5340_ = v___y_5363_;
v___y_5341_ = v___x_5367_;
goto v___jp_5329_;
}
}
else
{
v___y_5307_ = v___y_5354_;
v___y_5308_ = v___y_5353_;
v___y_5309_ = v___y_5355_;
v___y_5310_ = v___y_5356_;
v___y_5311_ = v___y_5357_;
v___y_5312_ = v___y_5358_;
v___y_5313_ = v___y_5359_;
v___y_5314_ = v___y_5361_;
v___y_5315_ = v___y_5360_;
v___y_5316_ = v___y_5363_;
v___y_5317_ = v___y_5362_;
v_a_5318_ = v___x_5365_;
goto v___jp_5306_;
}
}
v___jp_5374_:
{
if (lean_obj_tag(v___y_5378_) == 0)
{
lean_dec(v___y_5380_);
v___y_5353_ = v___y_5375_;
v___y_5354_ = v___y_5376_;
v___y_5355_ = v___y_5377_;
v___y_5356_ = v___y_5386_;
v___y_5357_ = v___y_5379_;
v___y_5358_ = v___y_5387_;
v___y_5359_ = v___y_5381_;
v___y_5360_ = v___y_5384_;
v___y_5361_ = v___y_5385_;
v___y_5362_ = v___y_5383_;
v___y_5363_ = v___y_5382_;
goto v___jp_5352_;
}
else
{
lean_object* v_val_5388_; uint8_t v___x_5389_; 
v_val_5388_ = lean_ctor_get(v___y_5378_, 0);
v___x_5389_ = lean_nat_dec_le(v_val_5388_, v___y_5380_);
lean_dec(v___y_5380_);
if (v___x_5389_ == 0)
{
v___y_5353_ = v___y_5375_;
v___y_5354_ = v___y_5376_;
v___y_5355_ = v___y_5377_;
v___y_5356_ = v___y_5386_;
v___y_5357_ = v___y_5379_;
v___y_5358_ = v___y_5387_;
v___y_5359_ = v___y_5381_;
v___y_5360_ = v___y_5384_;
v___y_5361_ = v___y_5385_;
v___y_5362_ = v___y_5383_;
v___y_5363_ = v___y_5382_;
goto v___jp_5352_;
}
else
{
lean_object* v___x_5390_; 
lean_dec_ref(v___y_5375_);
lean_dec_ref(v___f_5047_);
lean_dec(v_fst_5046_);
lean_dec(v___x_5043_);
v___x_5390_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5044_, v___y_5379_, v___y_5376_, v_snd_5045_, v___y_5377_, v_fst_5042_, v___y_5381_, v___y_5382_, v___y_5383_, v___y_5384_, v___y_5385_, v___y_5386_, v___y_5387_);
return v___x_5390_;
}
}
}
v___jp_5391_:
{
uint8_t v_hasTrace_5407_; 
v_hasTrace_5407_ = lean_ctor_get_uint8(v_options_5404_, sizeof(void*)*1);
if (v_hasTrace_5407_ == 0)
{
lean_dec(v___x_5049_);
v___y_5375_ = v___y_5393_;
v___y_5376_ = v___y_5392_;
v___y_5377_ = v___y_5394_;
v___y_5378_ = v___y_5395_;
v___y_5379_ = v_hypDepths_5397_;
v___y_5380_ = v_depth_5396_;
v___y_5381_ = v___y_5398_;
v___y_5382_ = v___y_5399_;
v___y_5383_ = v___y_5400_;
v___y_5384_ = v___y_5401_;
v___y_5385_ = v___y_5402_;
v___y_5386_ = v___y_5403_;
v___y_5387_ = v___y_5406_;
goto v___jp_5374_;
}
else
{
lean_object* v___x_5408_; lean_object* v___x_5409_; uint8_t v___x_5410_; 
v___x_5408_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1));
lean_inc(v___x_5049_);
v___x_5409_ = l_Lean_Name_append(v___x_5408_, v___x_5049_);
v___x_5410_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_5405_, v_options_5404_, v___x_5409_);
lean_dec(v___x_5409_);
if (v___x_5410_ == 0)
{
lean_dec(v___x_5049_);
v___y_5375_ = v___y_5393_;
v___y_5376_ = v___y_5392_;
v___y_5377_ = v___y_5394_;
v___y_5378_ = v___y_5395_;
v___y_5379_ = v_hypDepths_5397_;
v___y_5380_ = v_depth_5396_;
v___y_5381_ = v___y_5398_;
v___y_5382_ = v___y_5399_;
v___y_5383_ = v___y_5400_;
v___y_5384_ = v___y_5401_;
v___y_5385_ = v___y_5402_;
v___y_5386_ = v___y_5403_;
v___y_5387_ = v___y_5406_;
goto v___jp_5374_;
}
else
{
lean_object* v___x_5411_; lean_object* v___x_5412_; lean_object* v___x_5413_; lean_object* v___x_5414_; lean_object* v___x_5415_; lean_object* v___x_5416_; lean_object* v___x_5417_; lean_object* v___x_5418_; lean_object* v___x_5419_; lean_object* v___x_5420_; lean_object* v___x_5421_; lean_object* v___x_5422_; lean_object* v___x_5423_; lean_object* v___x_5424_; lean_object* v___x_5425_; 
v___x_5411_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__1);
lean_inc(v_depth_5396_);
v___x_5412_ = l_Nat_reprFast(v_depth_5396_);
v___x_5413_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_5413_, 0, v___x_5412_);
v___x_5414_ = l_Lean_MessageData_ofFormat(v___x_5413_);
v___x_5415_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5415_, 0, v___x_5411_);
lean_ctor_set(v___x_5415_, 1, v___x_5414_);
v___x_5416_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__3, &lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__3_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__3);
v___x_5417_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5417_, 0, v___x_5415_);
lean_ctor_set(v___x_5417_, 1, v___x_5416_);
lean_inc(v_fst_5046_);
v___x_5418_ = l_Lean_Expr_fvar___override(v_fst_5046_);
v___x_5419_ = l_Lean_MessageData_ofExpr(v___x_5418_);
v___x_5420_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5420_, 0, v___x_5417_);
lean_ctor_set(v___x_5420_, 1, v___x_5419_);
v___x_5421_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__5, &lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__5_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___closed__5);
v___x_5422_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5422_, 0, v___x_5420_);
lean_ctor_set(v___x_5422_, 1, v___x_5421_);
lean_inc_ref(v___y_5393_);
v___x_5423_ = l_Lean_MessageData_ofExpr(v___y_5393_);
v___x_5424_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5424_, 0, v___x_5422_);
lean_ctor_set(v___x_5424_, 1, v___x_5423_);
v___x_5425_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v___x_5049_, v___x_5424_, v___y_5401_, v___y_5402_, v___y_5403_, v___y_5406_);
if (lean_obj_tag(v___x_5425_) == 0)
{
lean_dec_ref_known(v___x_5425_, 1);
v___y_5375_ = v___y_5393_;
v___y_5376_ = v___y_5392_;
v___y_5377_ = v___y_5394_;
v___y_5378_ = v___y_5395_;
v___y_5379_ = v_hypDepths_5397_;
v___y_5380_ = v_depth_5396_;
v___y_5381_ = v___y_5398_;
v___y_5382_ = v___y_5399_;
v___y_5383_ = v___y_5400_;
v___y_5384_ = v___y_5401_;
v___y_5385_ = v___y_5402_;
v___y_5386_ = v___y_5403_;
v___y_5387_ = v___y_5406_;
goto v___jp_5374_;
}
else
{
lean_object* v_a_5426_; lean_object* v___x_5428_; uint8_t v_isShared_5429_; uint8_t v_isSharedCheck_5433_; 
lean_dec_ref(v_hypDepths_5397_);
lean_dec(v_depth_5396_);
lean_dec_ref(v___y_5394_);
lean_dec_ref(v___y_5393_);
lean_dec_ref(v___y_5392_);
lean_dec_ref(v___f_5047_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5426_ = lean_ctor_get(v___x_5425_, 0);
v_isSharedCheck_5433_ = !lean_is_exclusive(v___x_5425_);
if (v_isSharedCheck_5433_ == 0)
{
v___x_5428_ = v___x_5425_;
v_isShared_5429_ = v_isSharedCheck_5433_;
goto v_resetjp_5427_;
}
else
{
lean_inc(v_a_5426_);
lean_dec(v___x_5425_);
v___x_5428_ = lean_box(0);
v_isShared_5429_ = v_isSharedCheck_5433_;
goto v_resetjp_5427_;
}
v_resetjp_5427_:
{
lean_object* v___x_5431_; 
if (v_isShared_5429_ == 0)
{
v___x_5431_ = v___x_5428_;
goto v_reusejp_5430_;
}
else
{
lean_object* v_reuseFailAlloc_5432_; 
v_reuseFailAlloc_5432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5432_, 0, v_a_5426_);
v___x_5431_ = v_reuseFailAlloc_5432_;
goto v_reusejp_5430_;
}
v_reusejp_5430_:
{
return v___x_5431_;
}
}
}
}
}
}
v___jp_5434_:
{
lean_object* v___x_5436_; 
lean_inc(v_fst_5046_);
v___x_5436_ = l_Lean_FVarId_getType___redArg(v_fst_5046_, v___y_5060_, v___y_5062_, v___y_5063_);
if (lean_obj_tag(v___x_5436_) == 0)
{
lean_object* v_a_5437_; lean_object* v_forwardMaxDepth_x3f_5438_; lean_object* v___x_5439_; 
v_a_5437_ = lean_ctor_get(v___x_5436_, 0);
lean_inc(v_a_5437_);
lean_dec_ref_known(v___x_5436_, 1);
v_forwardMaxDepth_x3f_5438_ = lean_ctor_get(v___y_5057_, 1);
v___x_5439_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2(v_erasedHyps_5050_, v_snd_5051_);
if (lean_obj_tag(v_forwardMaxDepth_x3f_5438_) == 0)
{
lean_dec_ref(v___f_5053_);
lean_inc(v___x_5043_);
v___y_5392_ = v_a_5435_;
v___y_5393_ = v_a_5437_;
v___y_5394_ = v___x_5439_;
v___y_5395_ = v_forwardMaxDepth_x3f_5438_;
v_depth_5396_ = v___x_5043_;
v_hypDepths_5397_ = v_hypDepths_5052_;
v___y_5398_ = v___y_5057_;
v___y_5399_ = v___y_5058_;
v___y_5400_ = v___y_5059_;
v___y_5401_ = v___y_5060_;
v___y_5402_ = v___y_5061_;
v___y_5403_ = v___y_5062_;
v_options_5404_ = v_options_5224_;
v_inheritedTraceOptions_5405_ = v_inheritedTraceOptions_5225_;
v___y_5406_ = v___y_5063_;
goto v___jp_5391_;
}
else
{
lean_object* v___x_5440_; lean_object* v___x_5441_; lean_object* v___x_5442_; 
lean_inc(v___x_5043_);
v___x_5440_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg(v___f_5053_, v___x_5043_, v_fst_5054_);
v___x_5441_ = lean_nat_add(v___x_5055_, v___x_5440_);
lean_dec(v___x_5440_);
lean_inc(v___x_5441_);
lean_inc(v_fst_5046_);
v___x_5442_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4___redArg(v_hypDepths_5052_, v_fst_5046_, v___x_5441_);
v___y_5392_ = v_a_5435_;
v___y_5393_ = v_a_5437_;
v___y_5394_ = v___x_5439_;
v___y_5395_ = v_forwardMaxDepth_x3f_5438_;
v_depth_5396_ = v___x_5441_;
v_hypDepths_5397_ = v___x_5442_;
v___y_5398_ = v___y_5057_;
v___y_5399_ = v___y_5058_;
v___y_5400_ = v___y_5059_;
v___y_5401_ = v___y_5060_;
v___y_5402_ = v___y_5061_;
v___y_5403_ = v___y_5062_;
v_options_5404_ = v_options_5224_;
v_inheritedTraceOptions_5405_ = v_inheritedTraceOptions_5225_;
v___y_5406_ = v___y_5063_;
goto v___jp_5391_;
}
}
else
{
lean_object* v_a_5443_; lean_object* v___x_5445_; uint8_t v_isShared_5446_; uint8_t v_isSharedCheck_5450_; 
lean_dec_ref(v_a_5435_);
lean_dec_ref(v___f_5053_);
lean_dec_ref(v_hypDepths_5052_);
lean_dec_ref(v_erasedHyps_5050_);
lean_dec(v___x_5049_);
lean_dec_ref(v___f_5047_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5443_ = lean_ctor_get(v___x_5436_, 0);
v_isSharedCheck_5450_ = !lean_is_exclusive(v___x_5436_);
if (v_isSharedCheck_5450_ == 0)
{
v___x_5445_ = v___x_5436_;
v_isShared_5446_ = v_isSharedCheck_5450_;
goto v_resetjp_5444_;
}
else
{
lean_inc(v_a_5443_);
lean_dec(v___x_5436_);
v___x_5445_ = lean_box(0);
v_isShared_5446_ = v_isSharedCheck_5450_;
goto v_resetjp_5444_;
}
v_resetjp_5444_:
{
lean_object* v___x_5448_; 
if (v_isShared_5446_ == 0)
{
v___x_5448_ = v___x_5445_;
goto v_reusejp_5447_;
}
else
{
lean_object* v_reuseFailAlloc_5449_; 
v_reuseFailAlloc_5449_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5449_, 0, v_a_5443_);
v___x_5448_ = v_reuseFailAlloc_5449_;
goto v_reusejp_5447_;
}
v_reusejp_5447_:
{
return v___x_5448_;
}
}
}
}
v___jp_5451_:
{
lean_object* v___x_5452_; lean_object* v___x_5453_; lean_object* v___x_5454_; lean_object* v_stats_5455_; lean_object* v_rulePatternCache_5456_; lean_object* v___x_5458_; uint8_t v_isShared_5459_; uint8_t v_isSharedCheck_5483_; 
v___x_5452_ = lean_io_mono_nanos_now();
v___x_5453_ = lean_io_mono_nanos_now();
v___x_5454_ = lean_st_ref_take(v___y_5059_);
v_stats_5455_ = lean_ctor_get(v___x_5454_, 1);
v_rulePatternCache_5456_ = lean_ctor_get(v___x_5454_, 0);
v_isSharedCheck_5483_ = !lean_is_exclusive(v___x_5454_);
if (v_isSharedCheck_5483_ == 0)
{
v___x_5458_ = v___x_5454_;
v_isShared_5459_ = v_isSharedCheck_5483_;
goto v_resetjp_5457_;
}
else
{
lean_inc(v_stats_5455_);
lean_inc(v_rulePatternCache_5456_);
lean_dec(v___x_5454_);
v___x_5458_ = lean_box(0);
v_isShared_5459_ = v_isSharedCheck_5483_;
goto v_resetjp_5457_;
}
v_resetjp_5457_:
{
lean_object* v_total_5460_; lean_object* v_configParsing_5461_; lean_object* v_ruleSetConstruction_5462_; lean_object* v_search_5463_; lean_object* v_ruleSelection_5464_; lean_object* v_script_5465_; lean_object* v_forwardState_5466_; lean_object* v_scriptGenerated_5467_; lean_object* v_ruleStats_5468_; lean_object* v_goalStats_5469_; lean_object* v___x_5471_; uint8_t v_isShared_5472_; uint8_t v_isSharedCheck_5482_; 
v_total_5460_ = lean_ctor_get(v_stats_5455_, 0);
v_configParsing_5461_ = lean_ctor_get(v_stats_5455_, 1);
v_ruleSetConstruction_5462_ = lean_ctor_get(v_stats_5455_, 2);
v_search_5463_ = lean_ctor_get(v_stats_5455_, 3);
v_ruleSelection_5464_ = lean_ctor_get(v_stats_5455_, 4);
v_script_5465_ = lean_ctor_get(v_stats_5455_, 5);
v_forwardState_5466_ = lean_ctor_get(v_stats_5455_, 6);
v_scriptGenerated_5467_ = lean_ctor_get(v_stats_5455_, 7);
v_ruleStats_5468_ = lean_ctor_get(v_stats_5455_, 8);
v_goalStats_5469_ = lean_ctor_get(v_stats_5455_, 9);
v_isSharedCheck_5482_ = !lean_is_exclusive(v_stats_5455_);
if (v_isSharedCheck_5482_ == 0)
{
v___x_5471_ = v_stats_5455_;
v_isShared_5472_ = v_isSharedCheck_5482_;
goto v_resetjp_5470_;
}
else
{
lean_inc(v_goalStats_5469_);
lean_inc(v_ruleStats_5468_);
lean_inc(v_scriptGenerated_5467_);
lean_inc(v_forwardState_5466_);
lean_inc(v_script_5465_);
lean_inc(v_ruleSelection_5464_);
lean_inc(v_search_5463_);
lean_inc(v_ruleSetConstruction_5462_);
lean_inc(v_configParsing_5461_);
lean_inc(v_total_5460_);
lean_dec(v_stats_5455_);
v___x_5471_ = lean_box(0);
v_isShared_5472_ = v_isSharedCheck_5482_;
goto v_resetjp_5470_;
}
v_resetjp_5470_:
{
lean_object* v___x_5473_; lean_object* v___x_5474_; lean_object* v___x_5476_; 
v___x_5473_ = lean_nat_sub(v___x_5453_, v___x_5452_);
lean_dec(v___x_5452_);
lean_dec(v___x_5453_);
v___x_5474_ = lean_nat_add(v_forwardState_5466_, v___x_5473_);
lean_dec(v___x_5473_);
lean_dec(v_forwardState_5466_);
if (v_isShared_5472_ == 0)
{
lean_ctor_set(v___x_5471_, 6, v___x_5474_);
v___x_5476_ = v___x_5471_;
goto v_reusejp_5475_;
}
else
{
lean_object* v_reuseFailAlloc_5481_; 
v_reuseFailAlloc_5481_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_5481_, 0, v_total_5460_);
lean_ctor_set(v_reuseFailAlloc_5481_, 1, v_configParsing_5461_);
lean_ctor_set(v_reuseFailAlloc_5481_, 2, v_ruleSetConstruction_5462_);
lean_ctor_set(v_reuseFailAlloc_5481_, 3, v_search_5463_);
lean_ctor_set(v_reuseFailAlloc_5481_, 4, v_ruleSelection_5464_);
lean_ctor_set(v_reuseFailAlloc_5481_, 5, v_script_5465_);
lean_ctor_set(v_reuseFailAlloc_5481_, 6, v___x_5474_);
lean_ctor_set(v_reuseFailAlloc_5481_, 7, v_scriptGenerated_5467_);
lean_ctor_set(v_reuseFailAlloc_5481_, 8, v_ruleStats_5468_);
lean_ctor_set(v_reuseFailAlloc_5481_, 9, v_goalStats_5469_);
v___x_5476_ = v_reuseFailAlloc_5481_;
goto v_reusejp_5475_;
}
v_reusejp_5475_:
{
lean_object* v___x_5478_; 
if (v_isShared_5459_ == 0)
{
lean_ctor_set(v___x_5458_, 1, v___x_5476_);
v___x_5478_ = v___x_5458_;
goto v_reusejp_5477_;
}
else
{
lean_object* v_reuseFailAlloc_5480_; 
v_reuseFailAlloc_5480_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5480_, 0, v_rulePatternCache_5456_);
lean_ctor_set(v_reuseFailAlloc_5480_, 1, v___x_5476_);
v___x_5478_ = v_reuseFailAlloc_5480_;
goto v_reusejp_5477_;
}
v_reusejp_5477_:
{
lean_object* v___x_5479_; 
v___x_5479_ = lean_st_ref_set(v___y_5059_, v___x_5478_);
v_a_5435_ = v___y_5056_;
goto v___jp_5434_;
}
}
}
}
}
v___jp_5484_:
{
if (v_a_5485_ == 0)
{
v_a_5435_ = v___y_5056_;
goto v___jp_5434_;
}
else
{
goto v___jp_5451_;
}
}
v___jp_5486_:
{
if (lean_obj_tag(v___y_5487_) == 0)
{
lean_object* v_a_5488_; uint8_t v___x_5489_; 
v_a_5488_ = lean_ctor_get(v___y_5487_, 0);
lean_inc(v_a_5488_);
lean_dec_ref_known(v___y_5487_, 1);
v___x_5489_ = lean_unbox(v_a_5488_);
lean_dec(v_a_5488_);
v_a_5485_ = v___x_5489_;
goto v___jp_5484_;
}
else
{
lean_object* v_a_5490_; lean_object* v___x_5492_; uint8_t v_isShared_5493_; uint8_t v_isSharedCheck_5497_; 
lean_dec_ref(v___y_5056_);
lean_dec_ref(v___f_5053_);
lean_dec_ref(v_hypDepths_5052_);
lean_dec_ref(v_erasedHyps_5050_);
lean_dec(v___x_5049_);
lean_dec_ref(v___f_5047_);
lean_dec(v_fst_5046_);
lean_dec(v_snd_5045_);
lean_dec_ref(v_rs_5044_);
lean_dec(v___x_5043_);
lean_dec(v_fst_5042_);
v_a_5490_ = lean_ctor_get(v___y_5487_, 0);
v_isSharedCheck_5497_ = !lean_is_exclusive(v___y_5487_);
if (v_isSharedCheck_5497_ == 0)
{
v___x_5492_ = v___y_5487_;
v_isShared_5493_ = v_isSharedCheck_5497_;
goto v_resetjp_5491_;
}
else
{
lean_inc(v_a_5490_);
lean_dec(v___y_5487_);
v___x_5492_ = lean_box(0);
v_isShared_5493_ = v_isSharedCheck_5497_;
goto v_resetjp_5491_;
}
v_resetjp_5491_:
{
lean_object* v___x_5495_; 
if (v_isShared_5493_ == 0)
{
v___x_5495_ = v___x_5492_;
goto v_reusejp_5494_;
}
else
{
lean_object* v_reuseFailAlloc_5496_; 
v_reuseFailAlloc_5496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5496_, 0, v_a_5490_);
v___x_5495_ = v_reuseFailAlloc_5496_;
goto v_reusejp_5494_;
}
v_reusejp_5494_:
{
return v___x_5495_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___boxed(lean_object** _args){
lean_object* v_fst_5507_ = _args[0];
lean_object* v___x_5508_ = _args[1];
lean_object* v_rs_5509_ = _args[2];
lean_object* v_snd_5510_ = _args[3];
lean_object* v_fst_5511_ = _args[4];
lean_object* v___f_5512_ = _args[5];
lean_object* v___x_5513_ = _args[6];
lean_object* v___x_5514_ = _args[7];
lean_object* v_erasedHyps_5515_ = _args[8];
lean_object* v_snd_5516_ = _args[9];
lean_object* v_hypDepths_5517_ = _args[10];
lean_object* v___f_5518_ = _args[11];
lean_object* v_fst_5519_ = _args[12];
lean_object* v___x_5520_ = _args[13];
lean_object* v___y_5521_ = _args[14];
lean_object* v___y_5522_ = _args[15];
lean_object* v___y_5523_ = _args[16];
lean_object* v___y_5524_ = _args[17];
lean_object* v___y_5525_ = _args[18];
lean_object* v___y_5526_ = _args[19];
lean_object* v___y_5527_ = _args[20];
lean_object* v___y_5528_ = _args[21];
lean_object* v___y_5529_ = _args[22];
_start:
{
uint8_t v___x_134503__boxed_5530_; lean_object* v_res_5531_; 
v___x_134503__boxed_5530_ = lean_unbox(v___x_5513_);
v_res_5531_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4(v_fst_5507_, v___x_5508_, v_rs_5509_, v_snd_5510_, v_fst_5511_, v___f_5512_, v___x_134503__boxed_5530_, v___x_5514_, v_erasedHyps_5515_, v_snd_5516_, v_hypDepths_5517_, v___f_5518_, v_fst_5519_, v___x_5520_, v___y_5521_, v___y_5522_, v___y_5523_, v___y_5524_, v___y_5525_, v___y_5526_, v___y_5527_, v___y_5528_);
lean_dec(v___y_5528_);
lean_dec_ref(v___y_5527_);
lean_dec(v___y_5526_);
lean_dec_ref(v___y_5525_);
lean_dec(v___y_5524_);
lean_dec(v___y_5523_);
lean_dec_ref(v___y_5522_);
lean_dec(v___x_5520_);
lean_dec_ref(v_fst_5519_);
lean_dec_ref(v_snd_5516_);
return v_res_5531_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__1(void){
_start:
{
lean_object* v___x_5533_; lean_object* v___x_5534_; 
v___x_5533_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__0));
v___x_5534_ = l_Lean_stringToMessageData(v___x_5533_);
return v___x_5534_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5(lean_object* v___y_5535_, lean_object* v___x_5536_, lean_object* v_hypDepths_5537_, lean_object* v_rs_5538_, lean_object* v___f_5539_, lean_object* v_erasedHyps_5540_, lean_object* v___x_5541_, lean_object* v_fs_5542_, lean_object* v_goal_5543_, lean_object* v___f_5544_, lean_object* v___y_5545_, lean_object* v___y_5546_, lean_object* v___y_5547_, lean_object* v___y_5548_, lean_object* v___y_5549_, lean_object* v___y_5550_, lean_object* v___y_5551_){
_start:
{
if (lean_obj_tag(v___y_5535_) == 1)
{
lean_object* v_val_5553_; lean_object* v___x_5555_; uint8_t v_isShared_5556_; uint8_t v_isSharedCheck_5783_; 
v_val_5553_ = lean_ctor_get(v___y_5535_, 0);
v_isSharedCheck_5783_ = !lean_is_exclusive(v___y_5535_);
if (v_isSharedCheck_5783_ == 0)
{
v___x_5555_ = v___y_5535_;
v_isShared_5556_ = v_isSharedCheck_5783_;
goto v_resetjp_5554_;
}
else
{
lean_inc(v_val_5553_);
lean_dec(v___y_5535_);
v___x_5555_ = lean_box(0);
v_isShared_5556_ = v_isSharedCheck_5783_;
goto v_resetjp_5554_;
}
v_resetjp_5554_:
{
lean_object* v_fst_5557_; lean_object* v_snd_5558_; lean_object* v___x_5560_; uint8_t v_isShared_5561_; uint8_t v_isSharedCheck_5782_; 
v_fst_5557_ = lean_ctor_get(v_val_5553_, 0);
v_snd_5558_ = lean_ctor_get(v_val_5553_, 1);
v_isSharedCheck_5782_ = !lean_is_exclusive(v_val_5553_);
if (v_isSharedCheck_5782_ == 0)
{
v___x_5560_ = v_val_5553_;
v_isShared_5561_ = v_isSharedCheck_5782_;
goto v_resetjp_5559_;
}
else
{
lean_inc(v_snd_5558_);
lean_inc(v_fst_5557_);
lean_dec(v_val_5553_);
v___x_5560_ = lean_box(0);
v_isShared_5561_ = v_isSharedCheck_5782_;
goto v_resetjp_5559_;
}
v_resetjp_5559_:
{
uint8_t v___y_5563_; lean_object* v___y_5564_; lean_object* v___y_5565_; lean_object* v___y_5566_; lean_object* v___y_5567_; lean_object* v___y_5568_; lean_object* v___y_5569_; lean_object* v___y_5570_; lean_object* v___y_5571_; lean_object* v___y_5572_; lean_object* v___y_5573_; lean_object* v___y_5574_; lean_object* v___y_5575_; lean_object* v___y_5576_; lean_object* v___y_5577_; lean_object* v___y_5578_; uint8_t v___y_5583_; lean_object* v___y_5584_; lean_object* v___y_5585_; lean_object* v___y_5586_; lean_object* v___y_5587_; lean_object* v___y_5588_; lean_object* v___y_5589_; lean_object* v___y_5590_; lean_object* v___y_5591_; lean_object* v_a_5592_; uint8_t v___y_5611_; lean_object* v___y_5612_; lean_object* v___y_5613_; lean_object* v___y_5614_; lean_object* v___y_5615_; lean_object* v___y_5616_; lean_object* v___y_5617_; lean_object* v___y_5618_; lean_object* v___y_5619_; lean_object* v___y_5620_; lean_object* v___y_5621_; lean_object* v___y_5622_; lean_object* v___y_5623_; lean_object* v___y_5624_; uint8_t v___y_5625_; uint8_t v___y_5650_; lean_object* v___y_5651_; lean_object* v___y_5652_; lean_object* v___y_5653_; lean_object* v___y_5654_; lean_object* v___y_5655_; lean_object* v___y_5656_; uint8_t v___y_5657_; lean_object* v___y_5658_; lean_object* v___y_5659_; uint8_t v___y_5660_; lean_object* v___y_5661_; uint8_t v_a_5662_; uint8_t v___y_5680_; lean_object* v___y_5681_; lean_object* v___y_5682_; lean_object* v___y_5683_; lean_object* v___y_5684_; lean_object* v___y_5685_; lean_object* v___y_5686_; uint8_t v___y_5687_; lean_object* v___y_5688_; lean_object* v___y_5689_; uint8_t v___y_5690_; lean_object* v___y_5691_; uint8_t v_a_5692_; uint8_t v___y_5704_; lean_object* v___y_5705_; lean_object* v___y_5706_; lean_object* v___y_5707_; lean_object* v___y_5708_; lean_object* v___y_5709_; lean_object* v___y_5710_; uint8_t v___y_5711_; lean_object* v___y_5712_; lean_object* v___y_5713_; lean_object* v___y_5714_; uint8_t v___y_5715_; lean_object* v___y_5716_; lean_object* v_rule_5727_; lean_object* v_name_5728_; uint8_t v___y_5730_; lean_object* v___y_5731_; uint8_t v___y_5732_; lean_object* v___y_5733_; lean_object* v___y_5734_; lean_object* v___y_5735_; lean_object* v___y_5736_; lean_object* v___y_5737_; lean_object* v___y_5738_; lean_object* v_options_5739_; lean_object* v___y_5740_; uint8_t v___y_5755_; uint8_t v_phase_5778_; uint8_t v___x_5779_; uint8_t v___x_5780_; 
v_rule_5727_ = lean_ctor_get(v_fst_5557_, 0);
v_name_5728_ = lean_ctor_get(v_rule_5727_, 1);
v_phase_5778_ = lean_ctor_get_uint8(v_name_5728_, sizeof(void*)*1 + 9);
v___x_5779_ = 2;
v___x_5780_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_5778_, v___x_5779_);
if (v___x_5780_ == 0)
{
uint8_t v___x_5781_; 
v___x_5781_ = lp_aesop_Aesop_ForwardRuleMatch_anyHyp(v_fst_5557_, v___f_5544_);
v___y_5755_ = v___x_5781_;
goto v___jp_5754_;
}
else
{
lean_dec_ref(v___f_5544_);
v___y_5755_ = v___x_5780_;
goto v___jp_5754_;
}
v___jp_5562_:
{
lean_object* v___x_5579_; lean_object* v___f_5580_; lean_object* v___x_5581_; 
v___x_5579_ = lean_box(v___y_5563_);
v___f_5580_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__4___boxed), 23, 15);
lean_closure_set(v___f_5580_, 0, v___y_5566_);
lean_closure_set(v___f_5580_, 1, v___y_5568_);
lean_closure_set(v___f_5580_, 2, v_rs_5538_);
lean_closure_set(v___f_5580_, 3, v_snd_5558_);
lean_closure_set(v___f_5580_, 4, v___y_5564_);
lean_closure_set(v___f_5580_, 5, v___f_5539_);
lean_closure_set(v___f_5580_, 6, v___x_5579_);
lean_closure_set(v___f_5580_, 7, v___y_5569_);
lean_closure_set(v___f_5580_, 8, v_erasedHyps_5540_);
lean_closure_set(v___f_5580_, 9, v___y_5565_);
lean_closure_set(v___f_5580_, 10, v_hypDepths_5537_);
lean_closure_set(v___f_5580_, 11, v___y_5567_);
lean_closure_set(v___f_5580_, 12, v_fst_5557_);
lean_closure_set(v___f_5580_, 13, v___x_5541_);
lean_closure_set(v___f_5580_, 14, v___y_5578_);
v___x_5581_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg(v___y_5571_, v___f_5580_, v___y_5576_, v___y_5574_, v___y_5575_, v___y_5570_, v___y_5577_, v___y_5573_, v___y_5572_);
return v___x_5581_;
}
v___jp_5582_:
{
if (lean_obj_tag(v_a_5592_) == 1)
{
lean_object* v_val_5593_; lean_object* v_snd_5594_; lean_object* v_fst_5595_; lean_object* v_fst_5596_; lean_object* v_snd_5597_; lean_object* v___x_5598_; lean_object* v___f_5599_; lean_object* v___x_5600_; uint8_t v___x_5601_; 
lean_dec(v_goal_5543_);
v_val_5593_ = lean_ctor_get(v_a_5592_, 0);
lean_inc(v_val_5593_);
lean_dec_ref_known(v_a_5592_, 1);
v_snd_5594_ = lean_ctor_get(v_val_5593_, 1);
lean_inc(v_snd_5594_);
v_fst_5595_ = lean_ctor_get(v_val_5593_, 0);
lean_inc(v_fst_5595_);
lean_dec(v_val_5593_);
v_fst_5596_ = lean_ctor_get(v_snd_5594_, 0);
lean_inc(v_fst_5596_);
v_snd_5597_ = lean_ctor_get(v_snd_5594_, 1);
lean_inc(v_snd_5597_);
lean_dec(v_snd_5594_);
v___x_5598_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_hypDepths_5537_);
v___f_5599_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__2___boxed), 4, 2);
lean_closure_set(v___f_5599_, 0, v_hypDepths_5537_);
lean_closure_set(v___f_5599_, 1, v___x_5598_);
v___x_5600_ = lean_array_get_size(v_snd_5597_);
v___x_5601_ = lean_nat_dec_lt(v___x_5598_, v___x_5600_);
if (v___x_5601_ == 0)
{
lean_inc(v_fst_5595_);
v___y_5563_ = v___y_5583_;
v___y_5564_ = v_fst_5596_;
v___y_5565_ = v_snd_5597_;
v___y_5566_ = v_fst_5595_;
v___y_5567_ = v___f_5599_;
v___y_5568_ = v___x_5598_;
v___y_5569_ = v___y_5584_;
v___y_5570_ = v___y_5585_;
v___y_5571_ = v_fst_5595_;
v___y_5572_ = v___y_5586_;
v___y_5573_ = v___y_5588_;
v___y_5574_ = v___y_5587_;
v___y_5575_ = v___y_5589_;
v___y_5576_ = v___y_5590_;
v___y_5577_ = v___y_5591_;
v___y_5578_ = v_fs_5542_;
goto v___jp_5562_;
}
else
{
uint8_t v___x_5602_; 
v___x_5602_ = lean_nat_dec_le(v___x_5600_, v___x_5600_);
if (v___x_5602_ == 0)
{
if (v___x_5601_ == 0)
{
lean_inc(v_fst_5595_);
v___y_5563_ = v___y_5583_;
v___y_5564_ = v_fst_5596_;
v___y_5565_ = v_snd_5597_;
v___y_5566_ = v_fst_5595_;
v___y_5567_ = v___f_5599_;
v___y_5568_ = v___x_5598_;
v___y_5569_ = v___y_5584_;
v___y_5570_ = v___y_5585_;
v___y_5571_ = v_fst_5595_;
v___y_5572_ = v___y_5586_;
v___y_5573_ = v___y_5588_;
v___y_5574_ = v___y_5587_;
v___y_5575_ = v___y_5589_;
v___y_5576_ = v___y_5590_;
v___y_5577_ = v___y_5591_;
v___y_5578_ = v_fs_5542_;
goto v___jp_5562_;
}
else
{
size_t v___x_5603_; size_t v___x_5604_; lean_object* v___x_5605_; 
v___x_5603_ = ((size_t)0ULL);
v___x_5604_ = lean_usize_of_nat(v___x_5600_);
v___x_5605_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__6(v_snd_5597_, v___x_5603_, v___x_5604_, v_fs_5542_);
lean_inc(v_fst_5595_);
v___y_5563_ = v___y_5583_;
v___y_5564_ = v_fst_5596_;
v___y_5565_ = v_snd_5597_;
v___y_5566_ = v_fst_5595_;
v___y_5567_ = v___f_5599_;
v___y_5568_ = v___x_5598_;
v___y_5569_ = v___y_5584_;
v___y_5570_ = v___y_5585_;
v___y_5571_ = v_fst_5595_;
v___y_5572_ = v___y_5586_;
v___y_5573_ = v___y_5588_;
v___y_5574_ = v___y_5587_;
v___y_5575_ = v___y_5589_;
v___y_5576_ = v___y_5590_;
v___y_5577_ = v___y_5591_;
v___y_5578_ = v___x_5605_;
goto v___jp_5562_;
}
}
else
{
size_t v___x_5606_; size_t v___x_5607_; lean_object* v___x_5608_; 
v___x_5606_ = ((size_t)0ULL);
v___x_5607_ = lean_usize_of_nat(v___x_5600_);
v___x_5608_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__6(v_snd_5597_, v___x_5606_, v___x_5607_, v_fs_5542_);
lean_inc(v_fst_5595_);
v___y_5563_ = v___y_5583_;
v___y_5564_ = v_fst_5596_;
v___y_5565_ = v_snd_5597_;
v___y_5566_ = v_fst_5595_;
v___y_5567_ = v___f_5599_;
v___y_5568_ = v___x_5598_;
v___y_5569_ = v___y_5584_;
v___y_5570_ = v___y_5585_;
v___y_5571_ = v_fst_5595_;
v___y_5572_ = v___y_5586_;
v___y_5573_ = v___y_5588_;
v___y_5574_ = v___y_5587_;
v___y_5575_ = v___y_5589_;
v___y_5576_ = v___y_5590_;
v___y_5577_ = v___y_5591_;
v___y_5578_ = v___x_5608_;
goto v___jp_5562_;
}
}
}
else
{
lean_object* v___x_5609_; 
lean_dec(v_a_5592_);
lean_dec(v___y_5584_);
lean_dec(v_fst_5557_);
lean_dec(v___x_5541_);
lean_dec_ref(v___f_5539_);
v___x_5609_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5538_, v_hypDepths_5537_, v_fs_5542_, v_snd_5558_, v_erasedHyps_5540_, v_goal_5543_, v___y_5590_, v___y_5587_, v___y_5589_, v___y_5585_, v___y_5591_, v___y_5588_, v___y_5586_);
return v___x_5609_;
}
}
v___jp_5610_:
{
lean_object* v_total_5626_; lean_object* v_configParsing_5627_; lean_object* v_ruleSetConstruction_5628_; lean_object* v_search_5629_; lean_object* v_ruleSelection_5630_; lean_object* v_script_5631_; lean_object* v_forwardState_5632_; lean_object* v_scriptGenerated_5633_; lean_object* v_ruleStats_5634_; lean_object* v_goalStats_5635_; lean_object* v___x_5637_; uint8_t v_isShared_5638_; uint8_t v_isSharedCheck_5648_; 
v_total_5626_ = lean_ctor_get(v___y_5618_, 0);
v_configParsing_5627_ = lean_ctor_get(v___y_5618_, 1);
v_ruleSetConstruction_5628_ = lean_ctor_get(v___y_5618_, 2);
v_search_5629_ = lean_ctor_get(v___y_5618_, 3);
v_ruleSelection_5630_ = lean_ctor_get(v___y_5618_, 4);
v_script_5631_ = lean_ctor_get(v___y_5618_, 5);
v_forwardState_5632_ = lean_ctor_get(v___y_5618_, 6);
v_scriptGenerated_5633_ = lean_ctor_get(v___y_5618_, 7);
v_ruleStats_5634_ = lean_ctor_get(v___y_5618_, 8);
v_goalStats_5635_ = lean_ctor_get(v___y_5618_, 9);
v_isSharedCheck_5648_ = !lean_is_exclusive(v___y_5618_);
if (v_isSharedCheck_5648_ == 0)
{
v___x_5637_ = v___y_5618_;
v_isShared_5638_ = v_isSharedCheck_5648_;
goto v_resetjp_5636_;
}
else
{
lean_inc(v_goalStats_5635_);
lean_inc(v_ruleStats_5634_);
lean_inc(v_scriptGenerated_5633_);
lean_inc(v_forwardState_5632_);
lean_inc(v_script_5631_);
lean_inc(v_ruleSelection_5630_);
lean_inc(v_search_5629_);
lean_inc(v_ruleSetConstruction_5628_);
lean_inc(v_configParsing_5627_);
lean_inc(v_total_5626_);
lean_dec(v___y_5618_);
v___x_5637_ = lean_box(0);
v_isShared_5638_ = v_isSharedCheck_5648_;
goto v_resetjp_5636_;
}
v_resetjp_5636_:
{
lean_object* v_rp_5639_; lean_object* v___x_5640_; lean_object* v___x_5642_; 
v_rp_5639_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_rp_5639_, 0, v___y_5613_);
lean_ctor_set(v_rp_5639_, 1, v___y_5620_);
lean_ctor_set_uint8(v_rp_5639_, sizeof(void*)*2, v___y_5625_);
v___x_5640_ = lean_array_push(v_ruleStats_5634_, v_rp_5639_);
if (v_isShared_5638_ == 0)
{
lean_ctor_set(v___x_5637_, 8, v___x_5640_);
v___x_5642_ = v___x_5637_;
goto v_reusejp_5641_;
}
else
{
lean_object* v_reuseFailAlloc_5647_; 
v_reuseFailAlloc_5647_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_5647_, 0, v_total_5626_);
lean_ctor_set(v_reuseFailAlloc_5647_, 1, v_configParsing_5627_);
lean_ctor_set(v_reuseFailAlloc_5647_, 2, v_ruleSetConstruction_5628_);
lean_ctor_set(v_reuseFailAlloc_5647_, 3, v_search_5629_);
lean_ctor_set(v_reuseFailAlloc_5647_, 4, v_ruleSelection_5630_);
lean_ctor_set(v_reuseFailAlloc_5647_, 5, v_script_5631_);
lean_ctor_set(v_reuseFailAlloc_5647_, 6, v_forwardState_5632_);
lean_ctor_set(v_reuseFailAlloc_5647_, 7, v_scriptGenerated_5633_);
lean_ctor_set(v_reuseFailAlloc_5647_, 8, v___x_5640_);
lean_ctor_set(v_reuseFailAlloc_5647_, 9, v_goalStats_5635_);
v___x_5642_ = v_reuseFailAlloc_5647_;
goto v_reusejp_5641_;
}
v_reusejp_5641_:
{
lean_object* v___x_5644_; 
if (v_isShared_5561_ == 0)
{
lean_ctor_set(v___x_5560_, 1, v___x_5642_);
lean_ctor_set(v___x_5560_, 0, v___y_5616_);
v___x_5644_ = v___x_5560_;
goto v_reusejp_5643_;
}
else
{
lean_object* v_reuseFailAlloc_5646_; 
v_reuseFailAlloc_5646_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5646_, 0, v___y_5616_);
lean_ctor_set(v_reuseFailAlloc_5646_, 1, v___x_5642_);
v___x_5644_ = v_reuseFailAlloc_5646_;
goto v_reusejp_5643_;
}
v_reusejp_5643_:
{
lean_object* v___x_5645_; 
v___x_5645_ = lean_st_ref_set(v___y_5623_, v___x_5644_);
v___y_5583_ = v___y_5611_;
v___y_5584_ = v___y_5612_;
v___y_5585_ = v___y_5614_;
v___y_5586_ = v___y_5617_;
v___y_5587_ = v___y_5621_;
v___y_5588_ = v___y_5622_;
v___y_5589_ = v___y_5623_;
v___y_5590_ = v___y_5624_;
v___y_5591_ = v___y_5619_;
v_a_5592_ = v___y_5615_;
goto v___jp_5582_;
}
}
}
}
v___jp_5649_:
{
lean_object* v___x_5663_; lean_object* v___x_5664_; 
v___x_5663_ = lean_io_mono_nanos_now();
lean_inc(v_fst_5557_);
lean_inc(v_goal_5543_);
v___x_5664_ = lp_aesop_Aesop_ForwardRuleMatch_apply(v_goal_5543_, v_fst_5557_, v___y_5657_, v___y_5656_, v___y_5658_, v___y_5653_, v___y_5661_, v___y_5655_, v___y_5654_);
if (lean_obj_tag(v___x_5664_) == 0)
{
lean_object* v_a_5665_; lean_object* v___x_5666_; lean_object* v___x_5667_; lean_object* v_rulePatternCache_5668_; lean_object* v_stats_5669_; lean_object* v___x_5670_; 
v_a_5665_ = lean_ctor_get(v___x_5664_, 0);
lean_inc(v_a_5665_);
lean_dec_ref_known(v___x_5664_, 1);
v___x_5666_ = lean_io_mono_nanos_now();
v___x_5667_ = lean_st_ref_take(v___y_5658_);
v_rulePatternCache_5668_ = lean_ctor_get(v___x_5667_, 0);
lean_inc_ref(v_rulePatternCache_5668_);
v_stats_5669_ = lean_ctor_get(v___x_5667_, 1);
lean_inc_ref(v_stats_5669_);
lean_dec(v___x_5667_);
v___x_5670_ = lean_nat_sub(v___x_5666_, v___x_5663_);
lean_dec(v___x_5663_);
lean_dec(v___x_5666_);
if (lean_obj_tag(v_a_5665_) == 0)
{
v___y_5611_ = v___y_5650_;
v___y_5612_ = v___y_5651_;
v___y_5613_ = v___y_5652_;
v___y_5614_ = v___y_5653_;
v___y_5615_ = v_a_5665_;
v___y_5616_ = v_rulePatternCache_5668_;
v___y_5617_ = v___y_5654_;
v___y_5618_ = v_stats_5669_;
v___y_5619_ = v___y_5661_;
v___y_5620_ = v___x_5670_;
v___y_5621_ = v___y_5656_;
v___y_5622_ = v___y_5655_;
v___y_5623_ = v___y_5658_;
v___y_5624_ = v___y_5659_;
v___y_5625_ = v___y_5660_;
goto v___jp_5610_;
}
else
{
v___y_5611_ = v___y_5650_;
v___y_5612_ = v___y_5651_;
v___y_5613_ = v___y_5652_;
v___y_5614_ = v___y_5653_;
v___y_5615_ = v_a_5665_;
v___y_5616_ = v_rulePatternCache_5668_;
v___y_5617_ = v___y_5654_;
v___y_5618_ = v_stats_5669_;
v___y_5619_ = v___y_5661_;
v___y_5620_ = v___x_5670_;
v___y_5621_ = v___y_5656_;
v___y_5622_ = v___y_5655_;
v___y_5623_ = v___y_5658_;
v___y_5624_ = v___y_5659_;
v___y_5625_ = v_a_5662_;
goto v___jp_5610_;
}
}
else
{
lean_object* v_a_5671_; lean_object* v___x_5673_; uint8_t v_isShared_5674_; uint8_t v_isSharedCheck_5678_; 
lean_dec(v___x_5663_);
lean_dec(v___y_5652_);
lean_dec(v___y_5651_);
lean_del_object(v___x_5560_);
lean_dec(v_snd_5558_);
lean_dec(v_fst_5557_);
lean_dec(v_goal_5543_);
lean_dec_ref(v_fs_5542_);
lean_dec(v___x_5541_);
lean_dec_ref(v_erasedHyps_5540_);
lean_dec_ref(v___f_5539_);
lean_dec_ref(v_rs_5538_);
lean_dec_ref(v_hypDepths_5537_);
v_a_5671_ = lean_ctor_get(v___x_5664_, 0);
v_isSharedCheck_5678_ = !lean_is_exclusive(v___x_5664_);
if (v_isSharedCheck_5678_ == 0)
{
v___x_5673_ = v___x_5664_;
v_isShared_5674_ = v_isSharedCheck_5678_;
goto v_resetjp_5672_;
}
else
{
lean_inc(v_a_5671_);
lean_dec(v___x_5664_);
v___x_5673_ = lean_box(0);
v_isShared_5674_ = v_isSharedCheck_5678_;
goto v_resetjp_5672_;
}
v_resetjp_5672_:
{
lean_object* v___x_5676_; 
if (v_isShared_5674_ == 0)
{
v___x_5676_ = v___x_5673_;
goto v_reusejp_5675_;
}
else
{
lean_object* v_reuseFailAlloc_5677_; 
v_reuseFailAlloc_5677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5677_, 0, v_a_5671_);
v___x_5676_ = v_reuseFailAlloc_5677_;
goto v_reusejp_5675_;
}
v_reusejp_5675_:
{
return v___x_5676_;
}
}
}
}
v___jp_5679_:
{
if (v_a_5692_ == 0)
{
lean_object* v___x_5693_; 
lean_dec(v___y_5682_);
lean_del_object(v___x_5560_);
lean_inc(v_fst_5557_);
lean_inc(v_goal_5543_);
v___x_5693_ = lp_aesop_Aesop_ForwardRuleMatch_apply(v_goal_5543_, v_fst_5557_, v___y_5687_, v___y_5686_, v___y_5688_, v___y_5683_, v___y_5691_, v___y_5685_, v___y_5684_);
if (lean_obj_tag(v___x_5693_) == 0)
{
lean_object* v_a_5694_; 
v_a_5694_ = lean_ctor_get(v___x_5693_, 0);
lean_inc(v_a_5694_);
lean_dec_ref_known(v___x_5693_, 1);
v___y_5583_ = v___y_5680_;
v___y_5584_ = v___y_5681_;
v___y_5585_ = v___y_5683_;
v___y_5586_ = v___y_5684_;
v___y_5587_ = v___y_5686_;
v___y_5588_ = v___y_5685_;
v___y_5589_ = v___y_5688_;
v___y_5590_ = v___y_5689_;
v___y_5591_ = v___y_5691_;
v_a_5592_ = v_a_5694_;
goto v___jp_5582_;
}
else
{
lean_object* v_a_5695_; lean_object* v___x_5697_; uint8_t v_isShared_5698_; uint8_t v_isSharedCheck_5702_; 
lean_dec(v___y_5681_);
lean_dec(v_snd_5558_);
lean_dec(v_fst_5557_);
lean_dec(v_goal_5543_);
lean_dec_ref(v_fs_5542_);
lean_dec(v___x_5541_);
lean_dec_ref(v_erasedHyps_5540_);
lean_dec_ref(v___f_5539_);
lean_dec_ref(v_rs_5538_);
lean_dec_ref(v_hypDepths_5537_);
v_a_5695_ = lean_ctor_get(v___x_5693_, 0);
v_isSharedCheck_5702_ = !lean_is_exclusive(v___x_5693_);
if (v_isSharedCheck_5702_ == 0)
{
v___x_5697_ = v___x_5693_;
v_isShared_5698_ = v_isSharedCheck_5702_;
goto v_resetjp_5696_;
}
else
{
lean_inc(v_a_5695_);
lean_dec(v___x_5693_);
v___x_5697_ = lean_box(0);
v_isShared_5698_ = v_isSharedCheck_5702_;
goto v_resetjp_5696_;
}
v_resetjp_5696_:
{
lean_object* v___x_5700_; 
if (v_isShared_5698_ == 0)
{
v___x_5700_ = v___x_5697_;
goto v_reusejp_5699_;
}
else
{
lean_object* v_reuseFailAlloc_5701_; 
v_reuseFailAlloc_5701_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5701_, 0, v_a_5695_);
v___x_5700_ = v_reuseFailAlloc_5701_;
goto v_reusejp_5699_;
}
v_reusejp_5699_:
{
return v___x_5700_;
}
}
}
}
else
{
v___y_5650_ = v___y_5680_;
v___y_5651_ = v___y_5681_;
v___y_5652_ = v___y_5682_;
v___y_5653_ = v___y_5683_;
v___y_5654_ = v___y_5684_;
v___y_5655_ = v___y_5685_;
v___y_5656_ = v___y_5686_;
v___y_5657_ = v___y_5687_;
v___y_5658_ = v___y_5688_;
v___y_5659_ = v___y_5689_;
v___y_5660_ = v___y_5690_;
v___y_5661_ = v___y_5691_;
v_a_5662_ = v_a_5692_;
goto v___jp_5649_;
}
}
v___jp_5703_:
{
if (lean_obj_tag(v___y_5716_) == 0)
{
lean_object* v_a_5717_; uint8_t v___x_5718_; 
v_a_5717_ = lean_ctor_get(v___y_5716_, 0);
lean_inc(v_a_5717_);
lean_dec_ref_known(v___y_5716_, 1);
v___x_5718_ = lean_unbox(v_a_5717_);
lean_dec(v_a_5717_);
v___y_5680_ = v___y_5704_;
v___y_5681_ = v___y_5705_;
v___y_5682_ = v___y_5707_;
v___y_5683_ = v___y_5706_;
v___y_5684_ = v___y_5708_;
v___y_5685_ = v___y_5710_;
v___y_5686_ = v___y_5709_;
v___y_5687_ = v___y_5711_;
v___y_5688_ = v___y_5712_;
v___y_5689_ = v___y_5713_;
v___y_5690_ = v___y_5715_;
v___y_5691_ = v___y_5714_;
v_a_5692_ = v___x_5718_;
goto v___jp_5679_;
}
else
{
lean_object* v_a_5719_; lean_object* v___x_5721_; uint8_t v_isShared_5722_; uint8_t v_isSharedCheck_5726_; 
lean_dec(v___y_5707_);
lean_dec(v___y_5705_);
lean_del_object(v___x_5560_);
lean_dec(v_snd_5558_);
lean_dec(v_fst_5557_);
lean_dec(v_goal_5543_);
lean_dec_ref(v_fs_5542_);
lean_dec(v___x_5541_);
lean_dec_ref(v_erasedHyps_5540_);
lean_dec_ref(v___f_5539_);
lean_dec_ref(v_rs_5538_);
lean_dec_ref(v_hypDepths_5537_);
v_a_5719_ = lean_ctor_get(v___y_5716_, 0);
v_isSharedCheck_5726_ = !lean_is_exclusive(v___y_5716_);
if (v_isSharedCheck_5726_ == 0)
{
v___x_5721_ = v___y_5716_;
v_isShared_5722_ = v_isSharedCheck_5726_;
goto v_resetjp_5720_;
}
else
{
lean_inc(v_a_5719_);
lean_dec(v___y_5716_);
v___x_5721_ = lean_box(0);
v_isShared_5722_ = v_isSharedCheck_5726_;
goto v_resetjp_5720_;
}
v_resetjp_5720_:
{
lean_object* v___x_5724_; 
if (v_isShared_5722_ == 0)
{
v___x_5724_ = v___x_5721_;
goto v_reusejp_5723_;
}
else
{
lean_object* v_reuseFailAlloc_5725_; 
v_reuseFailAlloc_5725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5725_, 0, v_a_5719_);
v___x_5724_ = v_reuseFailAlloc_5725_;
goto v_reusejp_5723_;
}
v_reusejp_5723_:
{
return v___x_5724_;
}
}
}
}
v___jp_5729_:
{
lean_object* v___x_5742_; 
lean_inc_ref(v_name_5728_);
if (v_isShared_5556_ == 0)
{
lean_ctor_set_tag(v___x_5555_, 0);
lean_ctor_set(v___x_5555_, 0, v_name_5728_);
v___x_5742_ = v___x_5555_;
goto v_reusejp_5741_;
}
else
{
lean_object* v_reuseFailAlloc_5753_; 
v_reuseFailAlloc_5753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5753_, 0, v_name_5728_);
v___x_5742_ = v_reuseFailAlloc_5753_;
goto v_reusejp_5741_;
}
v_reusejp_5741_:
{
lean_object* v___x_5743_; uint8_t v___x_5744_; 
v___x_5743_ = lp_aesop_Aesop_aesop_collectStats;
v___x_5744_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_5739_, v___x_5743_);
if (v___x_5744_ == 0)
{
lean_object* v___x_5745_; lean_object* v___x_5746_; 
v___x_5745_ = lp_aesop_Aesop_TraceOption_stats;
v___x_5746_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_5745_, v___y_5738_);
if (lean_obj_tag(v___x_5746_) == 0)
{
lean_object* v_a_5747_; uint8_t v___x_5748_; 
v_a_5747_ = lean_ctor_get(v___x_5746_, 0);
lean_inc(v_a_5747_);
v___x_5748_ = lean_unbox(v_a_5747_);
lean_dec(v_a_5747_);
if (v___x_5748_ == 0)
{
lean_object* v___x_5749_; lean_object* v___x_5750_; lean_object* v___x_5751_; uint8_t v___x_5752_; 
v___x_5749_ = lp_aesop_Aesop_aesop_stats_file;
v___x_5750_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_5739_, v___x_5749_);
v___x_5751_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_5752_ = lean_string_dec_eq(v___x_5750_, v___x_5751_);
lean_dec_ref(v___x_5750_);
if (v___x_5752_ == 0)
{
lean_dec_ref_known(v___x_5746_, 1);
v___y_5650_ = v___y_5730_;
v___y_5651_ = v___y_5731_;
v___y_5652_ = v___x_5742_;
v___y_5653_ = v___y_5736_;
v___y_5654_ = v___y_5740_;
v___y_5655_ = v___y_5738_;
v___y_5656_ = v___y_5734_;
v___y_5657_ = v___y_5730_;
v___y_5658_ = v___y_5735_;
v___y_5659_ = v___y_5733_;
v___y_5660_ = v___y_5732_;
v___y_5661_ = v___y_5737_;
v_a_5662_ = v___y_5730_;
goto v___jp_5649_;
}
else
{
v___y_5704_ = v___y_5730_;
v___y_5705_ = v___y_5731_;
v___y_5706_ = v___y_5736_;
v___y_5707_ = v___x_5742_;
v___y_5708_ = v___y_5740_;
v___y_5709_ = v___y_5734_;
v___y_5710_ = v___y_5738_;
v___y_5711_ = v___y_5730_;
v___y_5712_ = v___y_5735_;
v___y_5713_ = v___y_5733_;
v___y_5714_ = v___y_5737_;
v___y_5715_ = v___y_5732_;
v___y_5716_ = v___x_5746_;
goto v___jp_5703_;
}
}
else
{
v___y_5704_ = v___y_5730_;
v___y_5705_ = v___y_5731_;
v___y_5706_ = v___y_5736_;
v___y_5707_ = v___x_5742_;
v___y_5708_ = v___y_5740_;
v___y_5709_ = v___y_5734_;
v___y_5710_ = v___y_5738_;
v___y_5711_ = v___y_5730_;
v___y_5712_ = v___y_5735_;
v___y_5713_ = v___y_5733_;
v___y_5714_ = v___y_5737_;
v___y_5715_ = v___y_5732_;
v___y_5716_ = v___x_5746_;
goto v___jp_5703_;
}
}
else
{
v___y_5704_ = v___y_5730_;
v___y_5705_ = v___y_5731_;
v___y_5706_ = v___y_5736_;
v___y_5707_ = v___x_5742_;
v___y_5708_ = v___y_5740_;
v___y_5709_ = v___y_5734_;
v___y_5710_ = v___y_5738_;
v___y_5711_ = v___y_5730_;
v___y_5712_ = v___y_5735_;
v___y_5713_ = v___y_5733_;
v___y_5714_ = v___y_5737_;
v___y_5715_ = v___y_5732_;
v___y_5716_ = v___x_5746_;
goto v___jp_5703_;
}
}
else
{
v___y_5680_ = v___y_5730_;
v___y_5681_ = v___y_5731_;
v___y_5682_ = v___x_5742_;
v___y_5683_ = v___y_5736_;
v___y_5684_ = v___y_5740_;
v___y_5685_ = v___y_5738_;
v___y_5686_ = v___y_5734_;
v___y_5687_ = v___y_5730_;
v___y_5688_ = v___y_5735_;
v___y_5689_ = v___y_5733_;
v___y_5690_ = v___y_5732_;
v___y_5691_ = v___y_5737_;
v_a_5692_ = v___x_5744_;
goto v___jp_5679_;
}
}
}
v___jp_5754_:
{
if (v___y_5755_ == 0)
{
lean_object* v_options_5756_; lean_object* v_inheritedTraceOptions_5757_; uint8_t v_hasTrace_5758_; uint8_t v___x_5759_; lean_object* v___x_5760_; 
v_options_5756_ = lean_ctor_get(v___y_5550_, 2);
v_inheritedTraceOptions_5757_ = lean_ctor_get(v___y_5550_, 13);
v_hasTrace_5758_ = lean_ctor_get_uint8(v_options_5756_, sizeof(void*)*1);
v___x_5759_ = 1;
v___x_5760_ = l_Lean_Name_mkStr1(v___x_5536_);
if (v_hasTrace_5758_ == 0)
{
v___y_5730_ = v___x_5759_;
v___y_5731_ = v___x_5760_;
v___y_5732_ = v___y_5755_;
v___y_5733_ = v___y_5545_;
v___y_5734_ = v___y_5546_;
v___y_5735_ = v___y_5547_;
v___y_5736_ = v___y_5548_;
v___y_5737_ = v___y_5549_;
v___y_5738_ = v___y_5550_;
v_options_5739_ = v_options_5756_;
v___y_5740_ = v___y_5551_;
goto v___jp_5729_;
}
else
{
lean_object* v___x_5761_; lean_object* v___x_5762_; uint8_t v___x_5763_; 
v___x_5761_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__3___closed__1));
lean_inc(v___x_5760_);
v___x_5762_ = l_Lean_Name_append(v___x_5761_, v___x_5760_);
v___x_5763_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_5757_, v_options_5756_, v___x_5762_);
lean_dec(v___x_5762_);
if (v___x_5763_ == 0)
{
v___y_5730_ = v___x_5759_;
v___y_5731_ = v___x_5760_;
v___y_5732_ = v___y_5755_;
v___y_5733_ = v___y_5545_;
v___y_5734_ = v___y_5546_;
v___y_5735_ = v___y_5547_;
v___y_5736_ = v___y_5548_;
v___y_5737_ = v___y_5549_;
v___y_5738_ = v___y_5550_;
v_options_5739_ = v_options_5756_;
v___y_5740_ = v___y_5551_;
goto v___jp_5729_;
}
else
{
lean_object* v___x_5764_; lean_object* v___x_5765_; lean_object* v___x_5766_; lean_object* v___x_5767_; lean_object* v___x_5768_; 
v___x_5764_ = lean_obj_once(&lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__1, &lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__1_once, _init_lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___closed__1);
lean_inc(v_goal_5543_);
v___x_5765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5765_, 0, v_goal_5543_);
v___x_5766_ = l_Lean_indentD(v___x_5765_);
v___x_5767_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5767_, 0, v___x_5764_);
lean_ctor_set(v___x_5767_, 1, v___x_5766_);
lean_inc(v___x_5760_);
v___x_5768_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__0___redArg(v___x_5760_, v___x_5767_, v___y_5548_, v___y_5549_, v___y_5550_, v___y_5551_);
if (lean_obj_tag(v___x_5768_) == 0)
{
lean_dec_ref_known(v___x_5768_, 1);
v___y_5730_ = v___x_5759_;
v___y_5731_ = v___x_5760_;
v___y_5732_ = v___y_5755_;
v___y_5733_ = v___y_5545_;
v___y_5734_ = v___y_5546_;
v___y_5735_ = v___y_5547_;
v___y_5736_ = v___y_5548_;
v___y_5737_ = v___y_5549_;
v___y_5738_ = v___y_5550_;
v_options_5739_ = v_options_5756_;
v___y_5740_ = v___y_5551_;
goto v___jp_5729_;
}
else
{
lean_object* v_a_5769_; lean_object* v___x_5771_; uint8_t v_isShared_5772_; uint8_t v_isSharedCheck_5776_; 
lean_dec(v___x_5760_);
lean_del_object(v___x_5560_);
lean_dec(v_snd_5558_);
lean_dec(v_fst_5557_);
lean_del_object(v___x_5555_);
lean_dec(v_goal_5543_);
lean_dec_ref(v_fs_5542_);
lean_dec(v___x_5541_);
lean_dec_ref(v_erasedHyps_5540_);
lean_dec_ref(v___f_5539_);
lean_dec_ref(v_rs_5538_);
lean_dec_ref(v_hypDepths_5537_);
v_a_5769_ = lean_ctor_get(v___x_5768_, 0);
v_isSharedCheck_5776_ = !lean_is_exclusive(v___x_5768_);
if (v_isSharedCheck_5776_ == 0)
{
v___x_5771_ = v___x_5768_;
v_isShared_5772_ = v_isSharedCheck_5776_;
goto v_resetjp_5770_;
}
else
{
lean_inc(v_a_5769_);
lean_dec(v___x_5768_);
v___x_5771_ = lean_box(0);
v_isShared_5772_ = v_isSharedCheck_5776_;
goto v_resetjp_5770_;
}
v_resetjp_5770_:
{
lean_object* v___x_5774_; 
if (v_isShared_5772_ == 0)
{
v___x_5774_ = v___x_5771_;
goto v_reusejp_5773_;
}
else
{
lean_object* v_reuseFailAlloc_5775_; 
v_reuseFailAlloc_5775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5775_, 0, v_a_5769_);
v___x_5774_ = v_reuseFailAlloc_5775_;
goto v_reusejp_5773_;
}
v_reusejp_5773_:
{
return v___x_5774_;
}
}
}
}
}
}
else
{
lean_object* v___x_5777_; 
lean_del_object(v___x_5560_);
lean_dec(v_fst_5557_);
lean_del_object(v___x_5555_);
lean_dec(v___x_5541_);
lean_dec_ref(v___f_5539_);
lean_dec_ref(v___x_5536_);
v___x_5777_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5538_, v_hypDepths_5537_, v_fs_5542_, v_snd_5558_, v_erasedHyps_5540_, v_goal_5543_, v___y_5545_, v___y_5546_, v___y_5547_, v___y_5548_, v___y_5549_, v___y_5550_, v___y_5551_);
return v___x_5777_;
}
}
}
}
}
else
{
lean_object* v___x_5784_; 
lean_dec_ref(v___f_5544_);
lean_dec_ref(v_fs_5542_);
lean_dec(v___x_5541_);
lean_dec_ref(v_erasedHyps_5540_);
lean_dec_ref(v___f_5539_);
lean_dec_ref(v_rs_5538_);
lean_dec_ref(v_hypDepths_5537_);
lean_dec_ref(v___x_5536_);
lean_dec(v___y_5535_);
v___x_5784_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5784_, 0, v_goal_5543_);
return v___x_5784_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___boxed(lean_object** _args){
lean_object* v___y_5785_ = _args[0];
lean_object* v___x_5786_ = _args[1];
lean_object* v_hypDepths_5787_ = _args[2];
lean_object* v_rs_5788_ = _args[3];
lean_object* v___f_5789_ = _args[4];
lean_object* v_erasedHyps_5790_ = _args[5];
lean_object* v___x_5791_ = _args[6];
lean_object* v_fs_5792_ = _args[7];
lean_object* v_goal_5793_ = _args[8];
lean_object* v___f_5794_ = _args[9];
lean_object* v___y_5795_ = _args[10];
lean_object* v___y_5796_ = _args[11];
lean_object* v___y_5797_ = _args[12];
lean_object* v___y_5798_ = _args[13];
lean_object* v___y_5799_ = _args[14];
lean_object* v___y_5800_ = _args[15];
lean_object* v___y_5801_ = _args[16];
lean_object* v___y_5802_ = _args[17];
_start:
{
lean_object* v_res_5803_; 
v_res_5803_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5(v___y_5785_, v___x_5786_, v_hypDepths_5787_, v_rs_5788_, v___f_5789_, v_erasedHyps_5790_, v___x_5791_, v_fs_5792_, v_goal_5793_, v___f_5794_, v___y_5795_, v___y_5796_, v___y_5797_, v___y_5798_, v___y_5799_, v___y_5800_, v___y_5801_);
lean_dec(v___y_5801_);
lean_dec_ref(v___y_5800_);
lean_dec(v___y_5799_);
lean_dec_ref(v___y_5798_);
lean_dec(v___y_5797_);
lean_dec(v___y_5796_);
lean_dec_ref(v___y_5795_);
return v_res_5803_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(lean_object* v_rs_5804_, lean_object* v_hypDepths_5805_, lean_object* v_fs_5806_, lean_object* v_queue_5807_, lean_object* v_erasedHyps_5808_, lean_object* v_goal_5809_, lean_object* v_a_5810_, lean_object* v_a_5811_, lean_object* v_a_5812_, lean_object* v_a_5813_, lean_object* v_a_5814_, lean_object* v_a_5815_, lean_object* v_a_5816_){
_start:
{
lean_object* v_fileName_5818_; lean_object* v_fileMap_5819_; lean_object* v_options_5820_; lean_object* v_currRecDepth_5821_; lean_object* v_maxRecDepth_5822_; lean_object* v_ref_5823_; lean_object* v_currNamespace_5824_; lean_object* v_openDecls_5825_; lean_object* v_initHeartbeats_5826_; lean_object* v_maxHeartbeats_5827_; lean_object* v_quotContext_5828_; lean_object* v_currMacroScope_5829_; uint8_t v_diag_5830_; lean_object* v_cancelTk_x3f_5831_; uint8_t v_suppressElabErrors_5832_; lean_object* v_inheritedTraceOptions_5833_; lean_object* v___f_5834_; lean_object* v___f_5835_; lean_object* v___x_5836_; lean_object* v___y_5838_; lean_object* v___y_5839_; lean_object* v___y_5840_; lean_object* v___x_5876_; uint8_t v___x_5877_; 
v_fileName_5818_ = lean_ctor_get(v_a_5815_, 0);
v_fileMap_5819_ = lean_ctor_get(v_a_5815_, 1);
v_options_5820_ = lean_ctor_get(v_a_5815_, 2);
v_currRecDepth_5821_ = lean_ctor_get(v_a_5815_, 3);
v_maxRecDepth_5822_ = lean_ctor_get(v_a_5815_, 4);
v_ref_5823_ = lean_ctor_get(v_a_5815_, 5);
v_currNamespace_5824_ = lean_ctor_get(v_a_5815_, 6);
v_openDecls_5825_ = lean_ctor_get(v_a_5815_, 7);
v_initHeartbeats_5826_ = lean_ctor_get(v_a_5815_, 8);
v_maxHeartbeats_5827_ = lean_ctor_get(v_a_5815_, 9);
v_quotContext_5828_ = lean_ctor_get(v_a_5815_, 10);
v_currMacroScope_5829_ = lean_ctor_get(v_a_5815_, 11);
v_diag_5830_ = lean_ctor_get_uint8(v_a_5815_, sizeof(void*)*14);
v_cancelTk_x3f_5831_ = lean_ctor_get(v_a_5815_, 12);
v_suppressElabErrors_5832_ = lean_ctor_get_uint8(v_a_5815_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_5833_ = lean_ctor_get(v_a_5815_, 13);
lean_inc_ref(v_erasedHyps_5808_);
v___f_5834_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__0___boxed), 2, 1);
lean_closure_set(v___f_5834_, 0, v_erasedHyps_5808_);
v___f_5835_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___closed__0));
v___x_5836_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__0_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
v___x_5876_ = lean_unsigned_to_nat(0u);
v___x_5877_ = lean_nat_dec_eq(v_maxRecDepth_5822_, v___x_5876_);
if (v___x_5877_ == 0)
{
uint8_t v___x_5878_; 
v___x_5878_ = lean_nat_dec_eq(v_currRecDepth_5821_, v_maxRecDepth_5822_);
if (v___x_5878_ == 0)
{
goto v___jp_5843_;
}
else
{
lean_object* v___x_5879_; 
lean_dec_ref(v___f_5834_);
lean_dec(v_goal_5809_);
lean_dec_ref(v_erasedHyps_5808_);
lean_dec(v_queue_5807_);
lean_dec_ref(v_fs_5806_);
lean_dec_ref(v_hypDepths_5805_);
lean_dec_ref(v_rs_5804_);
lean_inc(v_ref_5823_);
v___x_5879_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_go_spec__1___redArg(v_ref_5823_);
return v___x_5879_;
}
}
else
{
goto v___jp_5843_;
}
v___jp_5837_:
{
lean_object* v___y_5841_; lean_object* v___x_5842_; 
lean_inc(v_goal_5809_);
v___y_5841_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___lam__5___boxed), 18, 10);
lean_closure_set(v___y_5841_, 0, v___y_5840_);
lean_closure_set(v___y_5841_, 1, v___x_5836_);
lean_closure_set(v___y_5841_, 2, v_hypDepths_5805_);
lean_closure_set(v___y_5841_, 3, v_rs_5804_);
lean_closure_set(v___y_5841_, 4, v___f_5835_);
lean_closure_set(v___y_5841_, 5, v_erasedHyps_5808_);
lean_closure_set(v___y_5841_, 6, v___y_5838_);
lean_closure_set(v___y_5841_, 7, v_fs_5806_);
lean_closure_set(v___y_5841_, 8, v_goal_5809_);
lean_closure_set(v___y_5841_, 9, v___f_5834_);
v___x_5842_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg(v_goal_5809_, v___y_5841_, v_a_5810_, v_a_5811_, v_a_5812_, v_a_5813_, v_a_5814_, v___y_5839_, v_a_5816_);
lean_dec_ref(v___y_5839_);
return v___x_5842_;
}
v___jp_5843_:
{
lean_object* v___x_5844_; lean_object* v___x_5845_; lean_object* v___x_5846_; lean_object* v___x_5847_; 
v___x_5844_ = lean_unsigned_to_nat(1u);
v___x_5845_ = lean_nat_add(v_currRecDepth_5821_, v___x_5844_);
lean_inc_ref(v_inheritedTraceOptions_5833_);
lean_inc(v_cancelTk_x3f_5831_);
lean_inc(v_currMacroScope_5829_);
lean_inc(v_quotContext_5828_);
lean_inc(v_maxHeartbeats_5827_);
lean_inc(v_initHeartbeats_5826_);
lean_inc(v_openDecls_5825_);
lean_inc(v_currNamespace_5824_);
lean_inc(v_ref_5823_);
lean_inc(v_maxRecDepth_5822_);
lean_inc_ref(v_options_5820_);
lean_inc_ref(v_fileMap_5819_);
lean_inc_ref(v_fileName_5818_);
v___x_5846_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_5846_, 0, v_fileName_5818_);
lean_ctor_set(v___x_5846_, 1, v_fileMap_5819_);
lean_ctor_set(v___x_5846_, 2, v_options_5820_);
lean_ctor_set(v___x_5846_, 3, v___x_5845_);
lean_ctor_set(v___x_5846_, 4, v_maxRecDepth_5822_);
lean_ctor_set(v___x_5846_, 5, v_ref_5823_);
lean_ctor_set(v___x_5846_, 6, v_currNamespace_5824_);
lean_ctor_set(v___x_5846_, 7, v_openDecls_5825_);
lean_ctor_set(v___x_5846_, 8, v_initHeartbeats_5826_);
lean_ctor_set(v___x_5846_, 9, v_maxHeartbeats_5827_);
lean_ctor_set(v___x_5846_, 10, v_quotContext_5828_);
lean_ctor_set(v___x_5846_, 11, v_currMacroScope_5829_);
lean_ctor_set(v___x_5846_, 12, v_cancelTk_x3f_5831_);
lean_ctor_set(v___x_5846_, 13, v_inheritedTraceOptions_5833_);
lean_ctor_set_uint8(v___x_5846_, sizeof(void*)*14, v_diag_5830_);
lean_ctor_set_uint8(v___x_5846_, sizeof(void*)*14 + 1, v_suppressElabErrors_5832_);
v___x_5847_ = l_Lean_Core_checkSystem(v___x_5836_, v___x_5846_, v_a_5816_);
if (lean_obj_tag(v___x_5847_) == 0)
{
lean_object* v___x_5848_; lean_object* v___x_5849_; 
lean_dec_ref_known(v___x_5847_, 1);
v___x_5848_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___closed__1));
v___x_5849_ = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(v___x_5848_, v_queue_5807_);
if (lean_obj_tag(v___x_5849_) == 0)
{
lean_object* v___x_5850_; 
v___x_5850_ = lean_box(0);
v___y_5838_ = v___x_5844_;
v___y_5839_ = v___x_5846_;
v___y_5840_ = v___x_5850_;
goto v___jp_5837_;
}
else
{
lean_object* v_val_5851_; lean_object* v___x_5853_; uint8_t v_isShared_5854_; uint8_t v_isSharedCheck_5867_; 
v_val_5851_ = lean_ctor_get(v___x_5849_, 0);
v_isSharedCheck_5867_ = !lean_is_exclusive(v___x_5849_);
if (v_isSharedCheck_5867_ == 0)
{
v___x_5853_ = v___x_5849_;
v_isShared_5854_ = v_isSharedCheck_5867_;
goto v_resetjp_5852_;
}
else
{
lean_inc(v_val_5851_);
lean_dec(v___x_5849_);
v___x_5853_ = lean_box(0);
v_isShared_5854_ = v_isSharedCheck_5867_;
goto v_resetjp_5852_;
}
v_resetjp_5852_:
{
lean_object* v_fst_5855_; lean_object* v_snd_5856_; lean_object* v___x_5858_; uint8_t v_isShared_5859_; uint8_t v_isSharedCheck_5866_; 
v_fst_5855_ = lean_ctor_get(v_val_5851_, 0);
v_snd_5856_ = lean_ctor_get(v_val_5851_, 1);
v_isSharedCheck_5866_ = !lean_is_exclusive(v_val_5851_);
if (v_isSharedCheck_5866_ == 0)
{
v___x_5858_ = v_val_5851_;
v_isShared_5859_ = v_isSharedCheck_5866_;
goto v_resetjp_5857_;
}
else
{
lean_inc(v_snd_5856_);
lean_inc(v_fst_5855_);
lean_dec(v_val_5851_);
v___x_5858_ = lean_box(0);
v_isShared_5859_ = v_isSharedCheck_5866_;
goto v_resetjp_5857_;
}
v_resetjp_5857_:
{
lean_object* v___x_5861_; 
if (v_isShared_5859_ == 0)
{
v___x_5861_ = v___x_5858_;
goto v_reusejp_5860_;
}
else
{
lean_object* v_reuseFailAlloc_5865_; 
v_reuseFailAlloc_5865_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5865_, 0, v_fst_5855_);
lean_ctor_set(v_reuseFailAlloc_5865_, 1, v_snd_5856_);
v___x_5861_ = v_reuseFailAlloc_5865_;
goto v_reusejp_5860_;
}
v_reusejp_5860_:
{
lean_object* v___x_5863_; 
if (v_isShared_5854_ == 0)
{
lean_ctor_set(v___x_5853_, 0, v___x_5861_);
v___x_5863_ = v___x_5853_;
goto v_reusejp_5862_;
}
else
{
lean_object* v_reuseFailAlloc_5864_; 
v_reuseFailAlloc_5864_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5864_, 0, v___x_5861_);
v___x_5863_ = v_reuseFailAlloc_5864_;
goto v_reusejp_5862_;
}
v_reusejp_5862_:
{
v___y_5838_ = v___x_5844_;
v___y_5839_ = v___x_5846_;
v___y_5840_ = v___x_5863_;
goto v___jp_5837_;
}
}
}
}
}
}
else
{
lean_object* v_a_5868_; lean_object* v___x_5870_; uint8_t v_isShared_5871_; uint8_t v_isSharedCheck_5875_; 
lean_dec_ref_known(v___x_5846_, 14);
lean_dec_ref(v___f_5834_);
lean_dec(v_goal_5809_);
lean_dec_ref(v_erasedHyps_5808_);
lean_dec(v_queue_5807_);
lean_dec_ref(v_fs_5806_);
lean_dec_ref(v_hypDepths_5805_);
lean_dec_ref(v_rs_5804_);
v_a_5868_ = lean_ctor_get(v___x_5847_, 0);
v_isSharedCheck_5875_ = !lean_is_exclusive(v___x_5847_);
if (v_isSharedCheck_5875_ == 0)
{
v___x_5870_ = v___x_5847_;
v_isShared_5871_ = v_isSharedCheck_5875_;
goto v_resetjp_5869_;
}
else
{
lean_inc(v_a_5868_);
lean_dec(v___x_5847_);
v___x_5870_ = lean_box(0);
v_isShared_5871_ = v_isSharedCheck_5875_;
goto v_resetjp_5869_;
}
v_resetjp_5869_:
{
lean_object* v___x_5873_; 
if (v_isShared_5871_ == 0)
{
v___x_5873_ = v___x_5870_;
goto v_reusejp_5872_;
}
else
{
lean_object* v_reuseFailAlloc_5874_; 
v_reuseFailAlloc_5874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5874_, 0, v_a_5868_);
v___x_5873_ = v_reuseFailAlloc_5874_;
goto v_reusejp_5872_;
}
v_reusejp_5872_:
{
return v___x_5873_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go___boxed(lean_object* v_rs_5880_, lean_object* v_hypDepths_5881_, lean_object* v_fs_5882_, lean_object* v_queue_5883_, lean_object* v_erasedHyps_5884_, lean_object* v_goal_5885_, lean_object* v_a_5886_, lean_object* v_a_5887_, lean_object* v_a_5888_, lean_object* v_a_5889_, lean_object* v_a_5890_, lean_object* v_a_5891_, lean_object* v_a_5892_, lean_object* v_a_5893_){
_start:
{
lean_object* v_res_5894_; 
v_res_5894_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5880_, v_hypDepths_5881_, v_fs_5882_, v_queue_5883_, v_erasedHyps_5884_, v_goal_5885_, v_a_5886_, v_a_5887_, v_a_5888_, v_a_5889_, v_a_5890_, v_a_5891_, v_a_5892_);
lean_dec(v_a_5892_);
lean_dec_ref(v_a_5891_);
lean_dec(v_a_5890_);
lean_dec_ref(v_a_5889_);
lean_dec(v_a_5888_);
lean_dec(v_a_5887_);
lean_dec_ref(v_a_5886_);
return v_res_5894_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1(lean_object* v_00_u03b2_5895_, lean_object* v_m_5896_, lean_object* v_a_5897_){
_start:
{
lean_object* v___x_5898_; 
v___x_5898_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___redArg(v_m_5896_, v_a_5897_);
return v___x_5898_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1___boxed(lean_object* v_00_u03b2_5899_, lean_object* v_m_5900_, lean_object* v_a_5901_){
_start:
{
lean_object* v_res_5902_; 
v_res_5902_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1(v_00_u03b2_5899_, v_m_5900_, v_a_5901_);
lean_dec(v_a_5901_);
lean_dec_ref(v_m_5900_);
return v_res_5902_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4(lean_object* v_00_u03b2_5903_, lean_object* v_m_5904_, lean_object* v_a_5905_, lean_object* v_b_5906_){
_start:
{
lean_object* v___x_5907_; 
v___x_5907_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4___redArg(v_m_5904_, v_a_5905_, v_b_5906_);
return v___x_5907_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7(lean_object* v_00_u03b2_5908_, lean_object* v_m_5909_, lean_object* v_a_5910_){
_start:
{
uint8_t v___x_5911_; 
v___x_5911_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___redArg(v_m_5909_, v_a_5910_);
return v___x_5911_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7___boxed(lean_object* v_00_u03b2_5912_, lean_object* v_m_5913_, lean_object* v_a_5914_){
_start:
{
uint8_t v_res_5915_; lean_object* v_r_5916_; 
v_res_5915_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__7(v_00_u03b2_5912_, v_m_5913_, v_a_5914_);
lean_dec(v_a_5914_);
lean_dec_ref(v_m_5913_);
v_r_5916_ = lean_box(v_res_5915_);
return v_r_5916_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1(lean_object* v_00_u03b2_5917_, lean_object* v_a_5918_, lean_object* v_x_5919_){
_start:
{
lean_object* v___x_5920_; 
v___x_5920_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___redArg(v_a_5918_, v_x_5919_);
return v___x_5920_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1___boxed(lean_object* v_00_u03b2_5921_, lean_object* v_a_5922_, lean_object* v_x_5923_){
_start:
{
lean_object* v_res_5924_; 
v_res_5924_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__1_spec__1(v_00_u03b2_5921_, v_a_5922_, v_x_5923_);
lean_dec(v_x_5923_);
lean_dec(v_a_5922_);
return v_res_5924_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__3(lean_object* v_00_u03b2_5925_, lean_object* v_m_5926_, lean_object* v_a_5927_, lean_object* v_b_5928_){
_start:
{
lean_object* v___x_5929_; 
v___x_5929_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__2_spec__3___redArg(v_m_5926_, v_a_5927_, v_b_5928_);
return v___x_5929_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7(lean_object* v_00_u03b2_5930_, lean_object* v_a_5931_, lean_object* v_x_5932_){
_start:
{
uint8_t v___x_5933_; 
v___x_5933_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___redArg(v_a_5931_, v_x_5932_);
return v___x_5933_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7___boxed(lean_object* v_00_u03b2_5934_, lean_object* v_a_5935_, lean_object* v_x_5936_){
_start:
{
uint8_t v_res_5937_; lean_object* v_r_5938_; 
v_res_5937_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__7(v_00_u03b2_5934_, v_a_5935_, v_x_5936_);
lean_dec(v_x_5936_);
lean_dec(v_a_5935_);
v_r_5938_ = lean_box(v_res_5937_);
return v_r_5938_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8(lean_object* v_00_u03b2_5939_, lean_object* v_data_5940_){
_start:
{
lean_object* v___x_5941_; 
v___x_5941_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8___redArg(v_data_5940_);
return v___x_5941_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__9(lean_object* v_00_u03b2_5942_, lean_object* v_a_5943_, lean_object* v_b_5944_, lean_object* v_x_5945_){
_start:
{
lean_object* v___x_5946_; 
v___x_5946_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__9___redArg(v_a_5943_, v_b_5944_, v_x_5945_);
return v___x_5946_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10(lean_object* v_00_u03b2_5947_, lean_object* v_i_5948_, lean_object* v_source_5949_, lean_object* v_target_5950_){
_start:
{
lean_object* v___x_5951_; 
v___x_5951_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10___redArg(v_i_5948_, v_source_5949_, v_target_5950_);
return v___x_5951_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10_spec__13(lean_object* v_00_u03b2_5952_, lean_object* v_x_5953_, lean_object* v_x_5954_){
_start:
{
lean_object* v___x_5955_; 
v___x_5955_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__4_spec__8_spec__10_spec__13___redArg(v_x_5953_, v_x_5954_);
return v___x_5955_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___redArg(lean_object* v_as_5956_, size_t v_sz_5957_, size_t v_i_5958_, lean_object* v_b_5959_){
_start:
{
uint8_t v___x_5961_; 
v___x_5961_ = lean_usize_dec_lt(v_i_5958_, v_sz_5957_);
if (v___x_5961_ == 0)
{
lean_object* v___x_5962_; 
v___x_5962_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5962_, 0, v_b_5959_);
return v___x_5962_;
}
else
{
lean_object* v_a_5963_; lean_object* v___x_5964_; lean_object* v___x_5965_; lean_object* v___x_5966_; lean_object* v___x_5967_; lean_object* v___x_5968_; size_t v___x_5969_; size_t v___x_5970_; 
v_a_5963_ = lean_array_uget_borrowed(v_as_5956_, v_i_5958_);
v___x_5964_ = lean_unsigned_to_nat(0u);
v___x_5965_ = lean_box(0);
v___x_5966_ = lean_box(0);
lean_inc(v_a_5963_);
v___x_5967_ = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(v___x_5967_, 0, v___x_5964_);
lean_ctor_set(v___x_5967_, 1, v_a_5963_);
lean_ctor_set(v___x_5967_, 2, v___x_5965_);
lean_ctor_set(v___x_5967_, 3, v___x_5966_);
v___x_5968_ = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__0(v___x_5967_, v_b_5959_);
v___x_5969_ = ((size_t)1ULL);
v___x_5970_ = lean_usize_add(v_i_5958_, v___x_5969_);
v_i_5958_ = v___x_5970_;
v_b_5959_ = v___x_5968_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___redArg___boxed(lean_object* v_as_5972_, lean_object* v_sz_5973_, lean_object* v_i_5974_, lean_object* v_b_5975_, lean_object* v___y_5976_){
_start:
{
size_t v_sz_boxed_5977_; size_t v_i_boxed_5978_; lean_object* v_res_5979_; 
v_sz_boxed_5977_ = lean_unbox_usize(v_sz_5973_);
lean_dec(v_sz_5973_);
v_i_boxed_5978_ = lean_unbox_usize(v_i_5974_);
lean_dec(v_i_5974_);
v_res_5979_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___redArg(v_as_5972_, v_sz_boxed_5977_, v_i_boxed_5978_, v_b_5975_);
lean_dec_ref(v_as_5972_);
return v_res_5979_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__0(void){
_start:
{
lean_object* v___x_5980_; lean_object* v___x_5981_; lean_object* v___x_5982_; 
v___x_5980_ = lean_box(0);
v___x_5981_ = lean_unsigned_to_nat(16u);
v___x_5982_ = lean_mk_array(v___x_5981_, v___x_5980_);
return v___x_5982_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1(void){
_start:
{
lean_object* v___x_5983_; lean_object* v___x_5984_; lean_object* v___x_5985_; 
v___x_5983_ = lean_obj_once(&lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__0, &lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__0_once, _init_lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__0);
v___x_5984_ = lean_unsigned_to_nat(0u);
v___x_5985_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5985_, 0, v___x_5984_);
lean_ctor_set(v___x_5985_, 1, v___x_5983_);
return v___x_5985_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stateful_saturateCore___lam__0(lean_object* v_goal_5986_, lean_object* v___x_5987_, lean_object* v_rs_5988_, lean_object* v___y_5989_, lean_object* v___y_5990_, lean_object* v___y_5991_, lean_object* v___y_5992_, lean_object* v___y_5993_, lean_object* v___y_5994_, lean_object* v___y_5995_){
_start:
{
lean_object* v___x_5997_; 
lean_inc(v_goal_5986_);
v___x_5997_ = l_Lean_MVarId_checkNotAssigned(v_goal_5986_, v___x_5987_, v___y_5992_, v___y_5993_, v___y_5994_, v___y_5995_);
if (lean_obj_tag(v___x_5997_) == 0)
{
lean_object* v___x_5998_; 
lean_dec_ref_known(v___x_5997_, 1);
lean_inc_ref(v_rs_5988_);
lean_inc(v_goal_5986_);
v___x_5998_ = lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(v_goal_5986_, v_rs_5988_, v___y_5991_, v___y_5992_, v___y_5993_, v___y_5994_, v___y_5995_);
if (lean_obj_tag(v___x_5998_) == 0)
{
lean_object* v_a_5999_; lean_object* v_fst_6000_; lean_object* v_snd_6001_; lean_object* v_queue_6002_; size_t v_sz_6003_; size_t v___x_6004_; lean_object* v___x_6005_; 
v_a_5999_ = lean_ctor_get(v___x_5998_, 0);
lean_inc(v_a_5999_);
lean_dec_ref_known(v___x_5998_, 1);
v_fst_6000_ = lean_ctor_get(v_a_5999_, 0);
lean_inc(v_fst_6000_);
v_snd_6001_ = lean_ctor_get(v_a_5999_, 1);
lean_inc(v_snd_6001_);
lean_dec(v_a_5999_);
v_queue_6002_ = lean_box(0);
v_sz_6003_ = lean_array_size(v_snd_6001_);
v___x_6004_ = ((size_t)0ULL);
v___x_6005_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___redArg(v_snd_6001_, v_sz_6003_, v___x_6004_, v_queue_6002_);
lean_dec(v_snd_6001_);
if (lean_obj_tag(v___x_6005_) == 0)
{
lean_object* v_a_6006_; lean_object* v___x_6007_; lean_object* v___x_6008_; 
v_a_6006_ = lean_ctor_get(v___x_6005_, 0);
lean_inc(v_a_6006_);
lean_dec_ref_known(v___x_6005_, 1);
v___x_6007_ = lean_box(0);
lean_inc(v_goal_5986_);
v___x_6008_ = lp_aesop_Aesop_ForwardState_update(v_goal_5986_, v_fst_6000_, v___x_6007_, v___y_5991_, v___y_5992_, v___y_5993_, v___y_5994_, v___y_5995_);
if (lean_obj_tag(v___x_6008_) == 0)
{
lean_object* v_a_6009_; lean_object* v_fst_6010_; lean_object* v_snd_6011_; size_t v_sz_6012_; lean_object* v___x_6013_; 
v_a_6009_ = lean_ctor_get(v___x_6008_, 0);
lean_inc(v_a_6009_);
lean_dec_ref_known(v___x_6008_, 1);
v_fst_6010_ = lean_ctor_get(v_a_6009_, 0);
lean_inc(v_fst_6010_);
v_snd_6011_ = lean_ctor_get(v_a_6009_, 1);
lean_inc(v_snd_6011_);
lean_dec(v_a_6009_);
v_sz_6012_ = lean_array_size(v_snd_6011_);
v___x_6013_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___redArg(v_snd_6011_, v_sz_6012_, v___x_6004_, v_a_6006_);
lean_dec(v_snd_6011_);
if (lean_obj_tag(v___x_6013_) == 0)
{
lean_object* v_a_6014_; lean_object* v___x_6015_; lean_object* v___x_6016_; 
v_a_6014_ = lean_ctor_get(v___x_6013_, 0);
lean_inc(v_a_6014_);
lean_dec_ref_known(v___x_6013_, 1);
v___x_6015_ = lean_obj_once(&lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1, &lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1_once, _init_lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1);
v___x_6016_ = lp_aesop___private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go(v_rs_5988_, v___x_6015_, v_fst_6010_, v_a_6014_, v___x_6015_, v_goal_5986_, v___y_5989_, v___y_5990_, v___y_5991_, v___y_5992_, v___y_5993_, v___y_5994_, v___y_5995_);
return v___x_6016_;
}
else
{
lean_object* v_a_6017_; lean_object* v___x_6019_; uint8_t v_isShared_6020_; uint8_t v_isSharedCheck_6024_; 
lean_dec(v_fst_6010_);
lean_dec_ref(v_rs_5988_);
lean_dec(v_goal_5986_);
v_a_6017_ = lean_ctor_get(v___x_6013_, 0);
v_isSharedCheck_6024_ = !lean_is_exclusive(v___x_6013_);
if (v_isSharedCheck_6024_ == 0)
{
v___x_6019_ = v___x_6013_;
v_isShared_6020_ = v_isSharedCheck_6024_;
goto v_resetjp_6018_;
}
else
{
lean_inc(v_a_6017_);
lean_dec(v___x_6013_);
v___x_6019_ = lean_box(0);
v_isShared_6020_ = v_isSharedCheck_6024_;
goto v_resetjp_6018_;
}
v_resetjp_6018_:
{
lean_object* v___x_6022_; 
if (v_isShared_6020_ == 0)
{
v___x_6022_ = v___x_6019_;
goto v_reusejp_6021_;
}
else
{
lean_object* v_reuseFailAlloc_6023_; 
v_reuseFailAlloc_6023_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6023_, 0, v_a_6017_);
v___x_6022_ = v_reuseFailAlloc_6023_;
goto v_reusejp_6021_;
}
v_reusejp_6021_:
{
return v___x_6022_;
}
}
}
}
else
{
lean_object* v_a_6025_; lean_object* v___x_6027_; uint8_t v_isShared_6028_; uint8_t v_isSharedCheck_6032_; 
lean_dec(v_a_6006_);
lean_dec_ref(v_rs_5988_);
lean_dec(v_goal_5986_);
v_a_6025_ = lean_ctor_get(v___x_6008_, 0);
v_isSharedCheck_6032_ = !lean_is_exclusive(v___x_6008_);
if (v_isSharedCheck_6032_ == 0)
{
v___x_6027_ = v___x_6008_;
v_isShared_6028_ = v_isSharedCheck_6032_;
goto v_resetjp_6026_;
}
else
{
lean_inc(v_a_6025_);
lean_dec(v___x_6008_);
v___x_6027_ = lean_box(0);
v_isShared_6028_ = v_isSharedCheck_6032_;
goto v_resetjp_6026_;
}
v_resetjp_6026_:
{
lean_object* v___x_6030_; 
if (v_isShared_6028_ == 0)
{
v___x_6030_ = v___x_6027_;
goto v_reusejp_6029_;
}
else
{
lean_object* v_reuseFailAlloc_6031_; 
v_reuseFailAlloc_6031_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6031_, 0, v_a_6025_);
v___x_6030_ = v_reuseFailAlloc_6031_;
goto v_reusejp_6029_;
}
v_reusejp_6029_:
{
return v___x_6030_;
}
}
}
}
else
{
lean_object* v_a_6033_; lean_object* v___x_6035_; uint8_t v_isShared_6036_; uint8_t v_isSharedCheck_6040_; 
lean_dec(v_fst_6000_);
lean_dec_ref(v_rs_5988_);
lean_dec(v_goal_5986_);
v_a_6033_ = lean_ctor_get(v___x_6005_, 0);
v_isSharedCheck_6040_ = !lean_is_exclusive(v___x_6005_);
if (v_isSharedCheck_6040_ == 0)
{
v___x_6035_ = v___x_6005_;
v_isShared_6036_ = v_isSharedCheck_6040_;
goto v_resetjp_6034_;
}
else
{
lean_inc(v_a_6033_);
lean_dec(v___x_6005_);
v___x_6035_ = lean_box(0);
v_isShared_6036_ = v_isSharedCheck_6040_;
goto v_resetjp_6034_;
}
v_resetjp_6034_:
{
lean_object* v___x_6038_; 
if (v_isShared_6036_ == 0)
{
v___x_6038_ = v___x_6035_;
goto v_reusejp_6037_;
}
else
{
lean_object* v_reuseFailAlloc_6039_; 
v_reuseFailAlloc_6039_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6039_, 0, v_a_6033_);
v___x_6038_ = v_reuseFailAlloc_6039_;
goto v_reusejp_6037_;
}
v_reusejp_6037_:
{
return v___x_6038_;
}
}
}
}
else
{
lean_object* v_a_6041_; lean_object* v___x_6043_; uint8_t v_isShared_6044_; uint8_t v_isSharedCheck_6048_; 
lean_dec_ref(v_rs_5988_);
lean_dec(v_goal_5986_);
v_a_6041_ = lean_ctor_get(v___x_5998_, 0);
v_isSharedCheck_6048_ = !lean_is_exclusive(v___x_5998_);
if (v_isSharedCheck_6048_ == 0)
{
v___x_6043_ = v___x_5998_;
v_isShared_6044_ = v_isSharedCheck_6048_;
goto v_resetjp_6042_;
}
else
{
lean_inc(v_a_6041_);
lean_dec(v___x_5998_);
v___x_6043_ = lean_box(0);
v_isShared_6044_ = v_isSharedCheck_6048_;
goto v_resetjp_6042_;
}
v_resetjp_6042_:
{
lean_object* v___x_6046_; 
if (v_isShared_6044_ == 0)
{
v___x_6046_ = v___x_6043_;
goto v_reusejp_6045_;
}
else
{
lean_object* v_reuseFailAlloc_6047_; 
v_reuseFailAlloc_6047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6047_, 0, v_a_6041_);
v___x_6046_ = v_reuseFailAlloc_6047_;
goto v_reusejp_6045_;
}
v_reusejp_6045_:
{
return v___x_6046_;
}
}
}
}
else
{
lean_object* v_a_6049_; lean_object* v___x_6051_; uint8_t v_isShared_6052_; uint8_t v_isSharedCheck_6056_; 
lean_dec_ref(v_rs_5988_);
lean_dec(v_goal_5986_);
v_a_6049_ = lean_ctor_get(v___x_5997_, 0);
v_isSharedCheck_6056_ = !lean_is_exclusive(v___x_5997_);
if (v_isSharedCheck_6056_ == 0)
{
v___x_6051_ = v___x_5997_;
v_isShared_6052_ = v_isSharedCheck_6056_;
goto v_resetjp_6050_;
}
else
{
lean_inc(v_a_6049_);
lean_dec(v___x_5997_);
v___x_6051_ = lean_box(0);
v_isShared_6052_ = v_isSharedCheck_6056_;
goto v_resetjp_6050_;
}
v_resetjp_6050_:
{
lean_object* v___x_6054_; 
if (v_isShared_6052_ == 0)
{
v___x_6054_ = v___x_6051_;
goto v_reusejp_6053_;
}
else
{
lean_object* v_reuseFailAlloc_6055_; 
v_reuseFailAlloc_6055_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6055_, 0, v_a_6049_);
v___x_6054_ = v_reuseFailAlloc_6055_;
goto v_reusejp_6053_;
}
v_reusejp_6053_:
{
return v___x_6054_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stateful_saturateCore___lam__0___boxed(lean_object* v_goal_6057_, lean_object* v___x_6058_, lean_object* v_rs_6059_, lean_object* v___y_6060_, lean_object* v___y_6061_, lean_object* v___y_6062_, lean_object* v___y_6063_, lean_object* v___y_6064_, lean_object* v___y_6065_, lean_object* v___y_6066_, lean_object* v___y_6067_){
_start:
{
lean_object* v_res_6068_; 
v_res_6068_ = lp_aesop_Aesop_Stateful_saturateCore___lam__0(v_goal_6057_, v___x_6058_, v_rs_6059_, v___y_6060_, v___y_6061_, v___y_6062_, v___y_6063_, v___y_6064_, v___y_6065_, v___y_6066_);
lean_dec(v___y_6066_);
lean_dec_ref(v___y_6065_);
lean_dec(v___y_6064_);
lean_dec_ref(v___y_6063_);
lean_dec(v___y_6062_);
lean_dec(v___y_6061_);
lean_dec_ref(v___y_6060_);
return v_res_6068_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stateful_saturateCore(lean_object* v_rs_6069_, lean_object* v_goal_6070_, lean_object* v_a_6071_, lean_object* v_a_6072_, lean_object* v_a_6073_, lean_object* v_a_6074_, lean_object* v_a_6075_, lean_object* v_a_6076_, lean_object* v_a_6077_){
_start:
{
lean_object* v___x_6079_; lean_object* v___f_6080_; lean_object* v___x_6081_; 
v___x_6079_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_initFn___closed__1_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_));
lean_inc(v_goal_6070_);
v___f_6080_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Stateful_saturateCore___lam__0___boxed), 11, 3);
lean_closure_set(v___f_6080_, 0, v_goal_6070_);
lean_closure_set(v___f_6080_, 1, v___x_6079_);
lean_closure_set(v___f_6080_, 2, v_rs_6069_);
v___x_6081_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_Saturate_0__Aesop_Stateful_saturateCore_go_spec__5___redArg(v_goal_6070_, v___f_6080_, v_a_6071_, v_a_6072_, v_a_6073_, v_a_6074_, v_a_6075_, v_a_6076_, v_a_6077_);
if (lean_obj_tag(v___x_6081_) == 0)
{
return v___x_6081_;
}
else
{
lean_object* v_a_6082_; uint8_t v___y_6084_; uint8_t v___x_6104_; 
v_a_6082_ = lean_ctor_get(v___x_6081_, 0);
lean_inc(v_a_6082_);
v___x_6104_ = l_Lean_Exception_isInterrupt(v_a_6082_);
if (v___x_6104_ == 0)
{
uint8_t v___x_6105_; 
lean_inc(v_a_6082_);
v___x_6105_ = l_Lean_Exception_isRuntime(v_a_6082_);
v___y_6084_ = v___x_6105_;
goto v___jp_6083_;
}
else
{
v___y_6084_ = v___x_6104_;
goto v___jp_6083_;
}
v___jp_6083_:
{
if (v___y_6084_ == 0)
{
if (lean_obj_tag(v_a_6082_) == 0)
{
lean_object* v___x_6086_; uint8_t v_isShared_6087_; uint8_t v_isSharedCheck_6102_; 
v_isSharedCheck_6102_ = !lean_is_exclusive(v___x_6081_);
if (v_isSharedCheck_6102_ == 0)
{
lean_object* v_unused_6103_; 
v_unused_6103_ = lean_ctor_get(v___x_6081_, 0);
lean_dec(v_unused_6103_);
v___x_6086_ = v___x_6081_;
v_isShared_6087_ = v_isSharedCheck_6102_;
goto v_resetjp_6085_;
}
else
{
lean_dec(v___x_6081_);
v___x_6086_ = lean_box(0);
v_isShared_6087_ = v_isSharedCheck_6102_;
goto v_resetjp_6085_;
}
v_resetjp_6085_:
{
lean_object* v_ref_6088_; lean_object* v_msg_6089_; lean_object* v___x_6091_; uint8_t v_isShared_6092_; uint8_t v_isSharedCheck_6101_; 
v_ref_6088_ = lean_ctor_get(v_a_6082_, 0);
v_msg_6089_ = lean_ctor_get(v_a_6082_, 1);
v_isSharedCheck_6101_ = !lean_is_exclusive(v_a_6082_);
if (v_isSharedCheck_6101_ == 0)
{
v___x_6091_ = v_a_6082_;
v_isShared_6092_ = v_isSharedCheck_6101_;
goto v_resetjp_6090_;
}
else
{
lean_inc(v_msg_6089_);
lean_inc(v_ref_6088_);
lean_dec(v_a_6082_);
v___x_6091_ = lean_box(0);
v_isShared_6092_ = v_isSharedCheck_6101_;
goto v_resetjp_6090_;
}
v_resetjp_6090_:
{
lean_object* v___x_6093_; lean_object* v___x_6094_; lean_object* v___x_6096_; 
v___x_6093_ = lean_obj_once(&lp_aesop_Aesop_saturateCore___closed__2, &lp_aesop_Aesop_saturateCore___closed__2_once, _init_lp_aesop_Aesop_saturateCore___closed__2);
v___x_6094_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6094_, 0, v___x_6093_);
lean_ctor_set(v___x_6094_, 1, v_msg_6089_);
if (v_isShared_6092_ == 0)
{
lean_ctor_set(v___x_6091_, 1, v___x_6094_);
v___x_6096_ = v___x_6091_;
goto v_reusejp_6095_;
}
else
{
lean_object* v_reuseFailAlloc_6100_; 
v_reuseFailAlloc_6100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6100_, 0, v_ref_6088_);
lean_ctor_set(v_reuseFailAlloc_6100_, 1, v___x_6094_);
v___x_6096_ = v_reuseFailAlloc_6100_;
goto v_reusejp_6095_;
}
v_reusejp_6095_:
{
lean_object* v___x_6098_; 
if (v_isShared_6087_ == 0)
{
lean_ctor_set(v___x_6086_, 0, v___x_6096_);
v___x_6098_ = v___x_6086_;
goto v_reusejp_6097_;
}
else
{
lean_object* v_reuseFailAlloc_6099_; 
v_reuseFailAlloc_6099_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6099_, 0, v___x_6096_);
v___x_6098_ = v_reuseFailAlloc_6099_;
goto v_reusejp_6097_;
}
v_reusejp_6097_:
{
return v___x_6098_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_a_6082_, 2);
return v___x_6081_;
}
}
else
{
lean_dec(v_a_6082_);
return v___x_6081_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stateful_saturateCore___boxed(lean_object* v_rs_6106_, lean_object* v_goal_6107_, lean_object* v_a_6108_, lean_object* v_a_6109_, lean_object* v_a_6110_, lean_object* v_a_6111_, lean_object* v_a_6112_, lean_object* v_a_6113_, lean_object* v_a_6114_, lean_object* v_a_6115_){
_start:
{
lean_object* v_res_6116_; 
v_res_6116_ = lp_aesop_Aesop_Stateful_saturateCore(v_rs_6106_, v_goal_6107_, v_a_6108_, v_a_6109_, v_a_6110_, v_a_6111_, v_a_6112_, v_a_6113_, v_a_6114_);
lean_dec(v_a_6114_);
lean_dec_ref(v_a_6113_);
lean_dec(v_a_6112_);
lean_dec_ref(v_a_6111_);
lean_dec(v_a_6110_);
lean_dec(v_a_6109_);
lean_dec_ref(v_a_6108_);
return v_res_6116_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0(lean_object* v_as_6117_, size_t v_sz_6118_, size_t v_i_6119_, lean_object* v_b_6120_, lean_object* v___y_6121_, lean_object* v___y_6122_, lean_object* v___y_6123_, lean_object* v___y_6124_, lean_object* v___y_6125_, lean_object* v___y_6126_, lean_object* v___y_6127_){
_start:
{
lean_object* v___x_6129_; 
v___x_6129_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___redArg(v_as_6117_, v_sz_6118_, v_i_6119_, v_b_6120_);
return v___x_6129_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0___boxed(lean_object* v_as_6130_, lean_object* v_sz_6131_, lean_object* v_i_6132_, lean_object* v_b_6133_, lean_object* v___y_6134_, lean_object* v___y_6135_, lean_object* v___y_6136_, lean_object* v___y_6137_, lean_object* v___y_6138_, lean_object* v___y_6139_, lean_object* v___y_6140_, lean_object* v___y_6141_){
_start:
{
size_t v_sz_boxed_6142_; size_t v_i_boxed_6143_; lean_object* v_res_6144_; 
v_sz_boxed_6142_ = lean_unbox_usize(v_sz_6131_);
lean_dec(v_sz_6131_);
v_i_boxed_6143_ = lean_unbox_usize(v_i_6132_);
lean_dec(v_i_6132_);
v_res_6144_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stateful_saturateCore_spec__0(v_as_6130_, v_sz_boxed_6142_, v_i_boxed_6143_, v_b_6133_, v___y_6134_, v___y_6135_, v___y_6136_, v___y_6137_, v___y_6138_, v___y_6139_, v___y_6140_);
lean_dec(v___y_6140_);
lean_dec_ref(v___y_6139_);
lean_dec(v___y_6138_);
lean_dec_ref(v___y_6137_);
lean_dec(v___y_6136_);
lean_dec(v___y_6135_);
lean_dec_ref(v___y_6134_);
lean_dec_ref(v_as_6130_);
return v_res_6144_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg(size_t v_sz_6145_, size_t v_i_6146_, lean_object* v_bs_6147_, lean_object* v___y_6148_, lean_object* v___y_6149_, lean_object* v___y_6150_, lean_object* v___y_6151_){
_start:
{
uint8_t v___x_6153_; 
v___x_6153_ = lean_usize_dec_lt(v_i_6146_, v_sz_6145_);
if (v___x_6153_ == 0)
{
lean_object* v___x_6154_; 
v___x_6154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6154_, 0, v_bs_6147_);
return v___x_6154_;
}
else
{
lean_object* v_v_6155_; lean_object* v___x_6156_; 
v_v_6155_ = lean_array_uget_borrowed(v_bs_6147_, v_i_6146_);
lean_inc(v_v_6155_);
v___x_6156_ = lp_aesop_Aesop_Script_LazyStep_toStep(v_v_6155_, v___y_6148_, v___y_6149_, v___y_6150_, v___y_6151_);
if (lean_obj_tag(v___x_6156_) == 0)
{
lean_object* v_a_6157_; lean_object* v___x_6158_; lean_object* v_bs_x27_6159_; size_t v___x_6160_; size_t v___x_6161_; lean_object* v___x_6162_; 
v_a_6157_ = lean_ctor_get(v___x_6156_, 0);
lean_inc(v_a_6157_);
lean_dec_ref_known(v___x_6156_, 1);
v___x_6158_ = lean_unsigned_to_nat(0u);
v_bs_x27_6159_ = lean_array_uset(v_bs_6147_, v_i_6146_, v___x_6158_);
v___x_6160_ = ((size_t)1ULL);
v___x_6161_ = lean_usize_add(v_i_6146_, v___x_6160_);
v___x_6162_ = lean_array_uset(v_bs_x27_6159_, v_i_6146_, v_a_6157_);
v_i_6146_ = v___x_6161_;
v_bs_6147_ = v___x_6162_;
goto _start;
}
else
{
lean_object* v_a_6164_; lean_object* v___x_6166_; uint8_t v_isShared_6167_; uint8_t v_isSharedCheck_6171_; 
lean_dec_ref(v_bs_6147_);
v_a_6164_ = lean_ctor_get(v___x_6156_, 0);
v_isSharedCheck_6171_ = !lean_is_exclusive(v___x_6156_);
if (v_isSharedCheck_6171_ == 0)
{
v___x_6166_ = v___x_6156_;
v_isShared_6167_ = v_isSharedCheck_6171_;
goto v_resetjp_6165_;
}
else
{
lean_inc(v_a_6164_);
lean_dec(v___x_6156_);
v___x_6166_ = lean_box(0);
v_isShared_6167_ = v_isSharedCheck_6171_;
goto v_resetjp_6165_;
}
v_resetjp_6165_:
{
lean_object* v___x_6169_; 
if (v_isShared_6167_ == 0)
{
v___x_6169_ = v___x_6166_;
goto v_reusejp_6168_;
}
else
{
lean_object* v_reuseFailAlloc_6170_; 
v_reuseFailAlloc_6170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6170_, 0, v_a_6164_);
v___x_6169_ = v_reuseFailAlloc_6170_;
goto v_reusejp_6168_;
}
v_reusejp_6168_:
{
return v___x_6169_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg___boxed(lean_object* v_sz_6172_, lean_object* v_i_6173_, lean_object* v_bs_6174_, lean_object* v___y_6175_, lean_object* v___y_6176_, lean_object* v___y_6177_, lean_object* v___y_6178_, lean_object* v___y_6179_){
_start:
{
size_t v_sz_boxed_6180_; size_t v_i_boxed_6181_; lean_object* v_res_6182_; 
v_sz_boxed_6180_ = lean_unbox_usize(v_sz_6172_);
lean_dec(v_sz_6172_);
v_i_boxed_6181_ = lean_unbox_usize(v_i_6173_);
lean_dec(v_i_6173_);
v_res_6182_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg(v_sz_boxed_6180_, v_i_boxed_6181_, v_bs_6174_, v___y_6175_, v___y_6176_, v___y_6177_, v___y_6178_);
lean_dec(v___y_6178_);
lean_dec_ref(v___y_6177_);
lean_dec(v___y_6176_);
lean_dec_ref(v___y_6175_);
return v_res_6182_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0___lam__0(lean_object* v_x_6183_, lean_object* v_x_6184_){
_start:
{
lean_object* v_total_6185_; lean_object* v_configParsing_6186_; lean_object* v_ruleSetConstruction_6187_; lean_object* v_search_6188_; lean_object* v_ruleSelection_6189_; lean_object* v_script_6190_; lean_object* v_forwardState_6191_; lean_object* v_ruleStats_6192_; lean_object* v_goalStats_6193_; lean_object* v___x_6195_; uint8_t v_isShared_6196_; uint8_t v_isSharedCheck_6201_; 
v_total_6185_ = lean_ctor_get(v_x_6184_, 0);
v_configParsing_6186_ = lean_ctor_get(v_x_6184_, 1);
v_ruleSetConstruction_6187_ = lean_ctor_get(v_x_6184_, 2);
v_search_6188_ = lean_ctor_get(v_x_6184_, 3);
v_ruleSelection_6189_ = lean_ctor_get(v_x_6184_, 4);
v_script_6190_ = lean_ctor_get(v_x_6184_, 5);
v_forwardState_6191_ = lean_ctor_get(v_x_6184_, 6);
v_ruleStats_6192_ = lean_ctor_get(v_x_6184_, 8);
v_goalStats_6193_ = lean_ctor_get(v_x_6184_, 9);
v_isSharedCheck_6201_ = !lean_is_exclusive(v_x_6184_);
if (v_isSharedCheck_6201_ == 0)
{
lean_object* v_unused_6202_; 
v_unused_6202_ = lean_ctor_get(v_x_6184_, 7);
lean_dec(v_unused_6202_);
v___x_6195_ = v_x_6184_;
v_isShared_6196_ = v_isSharedCheck_6201_;
goto v_resetjp_6194_;
}
else
{
lean_inc(v_goalStats_6193_);
lean_inc(v_ruleStats_6192_);
lean_inc(v_forwardState_6191_);
lean_inc(v_script_6190_);
lean_inc(v_ruleSelection_6189_);
lean_inc(v_search_6188_);
lean_inc(v_ruleSetConstruction_6187_);
lean_inc(v_configParsing_6186_);
lean_inc(v_total_6185_);
lean_dec(v_x_6184_);
v___x_6195_ = lean_box(0);
v_isShared_6196_ = v_isSharedCheck_6201_;
goto v_resetjp_6194_;
}
v_resetjp_6194_:
{
lean_object* v___x_6197_; lean_object* v___x_6199_; 
v___x_6197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6197_, 0, v_x_6183_);
if (v_isShared_6196_ == 0)
{
lean_ctor_set(v___x_6195_, 7, v___x_6197_);
v___x_6199_ = v___x_6195_;
goto v_reusejp_6198_;
}
else
{
lean_object* v_reuseFailAlloc_6200_; 
v_reuseFailAlloc_6200_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_6200_, 0, v_total_6185_);
lean_ctor_set(v_reuseFailAlloc_6200_, 1, v_configParsing_6186_);
lean_ctor_set(v_reuseFailAlloc_6200_, 2, v_ruleSetConstruction_6187_);
lean_ctor_set(v_reuseFailAlloc_6200_, 3, v_search_6188_);
lean_ctor_set(v_reuseFailAlloc_6200_, 4, v_ruleSelection_6189_);
lean_ctor_set(v_reuseFailAlloc_6200_, 5, v_script_6190_);
lean_ctor_set(v_reuseFailAlloc_6200_, 6, v_forwardState_6191_);
lean_ctor_set(v_reuseFailAlloc_6200_, 7, v___x_6197_);
lean_ctor_set(v_reuseFailAlloc_6200_, 8, v_ruleStats_6192_);
lean_ctor_set(v_reuseFailAlloc_6200_, 9, v_goalStats_6193_);
v___x_6199_ = v_reuseFailAlloc_6200_;
goto v_reusejp_6198_;
}
v_reusejp_6198_:
{
return v___x_6199_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___redArg(lean_object* v_f_6203_, lean_object* v___y_6204_, lean_object* v___y_6205_){
_start:
{
lean_object* v___y_6226_; lean_object* v_options_6229_; lean_object* v___x_6230_; uint8_t v___x_6231_; 
v_options_6229_ = lean_ctor_get(v___y_6205_, 2);
v___x_6230_ = lp_aesop_Aesop_aesop_collectStats;
v___x_6231_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_6229_, v___x_6230_);
if (v___x_6231_ == 0)
{
lean_object* v___x_6232_; lean_object* v___x_6233_; lean_object* v_a_6234_; uint8_t v___x_6235_; 
v___x_6232_ = lp_aesop_Aesop_TraceOption_stats;
v___x_6233_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_6232_, v___y_6205_);
v_a_6234_ = lean_ctor_get(v___x_6233_, 0);
lean_inc(v_a_6234_);
v___x_6235_ = lean_unbox(v_a_6234_);
lean_dec(v_a_6234_);
if (v___x_6235_ == 0)
{
lean_object* v___x_6236_; lean_object* v___x_6237_; lean_object* v___x_6238_; uint8_t v___x_6239_; 
lean_dec_ref(v___x_6233_);
v___x_6236_ = lp_aesop_Aesop_aesop_stats_file;
v___x_6237_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_6229_, v___x_6236_);
v___x_6238_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_6239_ = lean_string_dec_eq(v___x_6237_, v___x_6238_);
lean_dec_ref(v___x_6237_);
if (v___x_6239_ == 0)
{
goto v___jp_6207_;
}
else
{
lean_dec_ref(v_f_6203_);
goto v___jp_6222_;
}
}
else
{
v___y_6226_ = v___x_6233_;
goto v___jp_6225_;
}
}
else
{
goto v___jp_6207_;
}
v___jp_6207_:
{
lean_object* v___x_6208_; lean_object* v_rulePatternCache_6209_; lean_object* v_stats_6210_; lean_object* v___x_6212_; uint8_t v_isShared_6213_; uint8_t v_isSharedCheck_6221_; 
v___x_6208_ = lean_st_ref_take(v___y_6204_);
v_rulePatternCache_6209_ = lean_ctor_get(v___x_6208_, 0);
v_stats_6210_ = lean_ctor_get(v___x_6208_, 1);
v_isSharedCheck_6221_ = !lean_is_exclusive(v___x_6208_);
if (v_isSharedCheck_6221_ == 0)
{
v___x_6212_ = v___x_6208_;
v_isShared_6213_ = v_isSharedCheck_6221_;
goto v_resetjp_6211_;
}
else
{
lean_inc(v_stats_6210_);
lean_inc(v_rulePatternCache_6209_);
lean_dec(v___x_6208_);
v___x_6212_ = lean_box(0);
v_isShared_6213_ = v_isSharedCheck_6221_;
goto v_resetjp_6211_;
}
v_resetjp_6211_:
{
lean_object* v___x_6214_; lean_object* v___x_6216_; 
v___x_6214_ = lean_apply_1(v_f_6203_, v_stats_6210_);
if (v_isShared_6213_ == 0)
{
lean_ctor_set(v___x_6212_, 1, v___x_6214_);
v___x_6216_ = v___x_6212_;
goto v_reusejp_6215_;
}
else
{
lean_object* v_reuseFailAlloc_6220_; 
v_reuseFailAlloc_6220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6220_, 0, v_rulePatternCache_6209_);
lean_ctor_set(v_reuseFailAlloc_6220_, 1, v___x_6214_);
v___x_6216_ = v_reuseFailAlloc_6220_;
goto v_reusejp_6215_;
}
v_reusejp_6215_:
{
lean_object* v___x_6217_; lean_object* v___x_6218_; lean_object* v___x_6219_; 
v___x_6217_ = lean_st_ref_set(v___y_6204_, v___x_6216_);
v___x_6218_ = lean_box(0);
v___x_6219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6219_, 0, v___x_6218_);
return v___x_6219_;
}
}
}
v___jp_6222_:
{
lean_object* v___x_6223_; lean_object* v___x_6224_; 
v___x_6223_ = lean_box(0);
v___x_6224_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6224_, 0, v___x_6223_);
return v___x_6224_;
}
v___jp_6225_:
{
lean_object* v_a_6227_; uint8_t v___x_6228_; 
v_a_6227_ = lean_ctor_get(v___y_6226_, 0);
lean_inc(v_a_6227_);
lean_dec_ref(v___y_6226_);
v___x_6228_ = lean_unbox(v_a_6227_);
lean_dec(v_a_6227_);
if (v___x_6228_ == 0)
{
lean_dec_ref(v_f_6203_);
goto v___jp_6222_;
}
else
{
goto v___jp_6207_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___redArg___boxed(lean_object* v_f_6240_, lean_object* v___y_6241_, lean_object* v___y_6242_, lean_object* v___y_6243_){
_start:
{
lean_object* v_res_6244_; 
v_res_6244_ = lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___redArg(v_f_6240_, v___y_6241_, v___y_6242_);
lean_dec_ref(v___y_6242_);
lean_dec(v___y_6241_);
return v_res_6244_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0(lean_object* v_x_6245_, lean_object* v___y_6246_, lean_object* v___y_6247_, lean_object* v___y_6248_, lean_object* v___y_6249_, lean_object* v___y_6250_, lean_object* v___y_6251_, lean_object* v___y_6252_){
_start:
{
lean_object* v___f_6254_; lean_object* v___x_6255_; 
v___f_6254_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0___lam__0), 2, 1);
lean_closure_set(v___f_6254_, 0, v_x_6245_);
v___x_6255_ = lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___redArg(v___f_6254_, v___y_6248_, v___y_6251_);
return v___x_6255_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0___boxed(lean_object* v_x_6256_, lean_object* v___y_6257_, lean_object* v___y_6258_, lean_object* v___y_6259_, lean_object* v___y_6260_, lean_object* v___y_6261_, lean_object* v___y_6262_, lean_object* v___y_6263_, lean_object* v___y_6264_){
_start:
{
lean_object* v_res_6265_; 
v_res_6265_ = lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0(v_x_6256_, v___y_6257_, v___y_6258_, v___y_6259_, v___y_6260_, v___y_6261_, v___y_6262_, v___y_6263_);
lean_dec(v___y_6263_);
lean_dec_ref(v___y_6262_);
lean_dec(v___y_6261_);
lean_dec_ref(v___y_6260_);
lean_dec(v___y_6259_);
lean_dec(v___y_6258_);
lean_dec_ref(v___y_6257_);
return v_res_6265_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9_spec__10(lean_object* v_x_6266_, lean_object* v_r_6267_, lean_object* v_as_6268_, size_t v_sz_6269_, size_t v_i_6270_, lean_object* v_b_6271_){
_start:
{
lean_object* v_a_6273_; uint8_t v___x_6277_; 
v___x_6277_ = lean_usize_dec_lt(v_i_6270_, v_sz_6269_);
if (v___x_6277_ == 0)
{
return v_b_6271_;
}
else
{
lean_object* v_fst_6278_; lean_object* v_snd_6279_; lean_object* v___x_6281_; uint8_t v_isShared_6282_; uint8_t v_isSharedCheck_6296_; 
v_fst_6278_ = lean_ctor_get(v_b_6271_, 0);
v_snd_6279_ = lean_ctor_get(v_b_6271_, 1);
v_isSharedCheck_6296_ = !lean_is_exclusive(v_b_6271_);
if (v_isSharedCheck_6296_ == 0)
{
v___x_6281_ = v_b_6271_;
v_isShared_6282_ = v_isSharedCheck_6296_;
goto v_resetjp_6280_;
}
else
{
lean_inc(v_snd_6279_);
lean_inc(v_fst_6278_);
lean_dec(v_b_6271_);
v___x_6281_ = lean_box(0);
v_isShared_6282_ = v_isSharedCheck_6296_;
goto v_resetjp_6280_;
}
v_resetjp_6280_:
{
lean_object* v_a_6283_; lean_object* v_goal_6284_; lean_object* v_goal_6285_; uint8_t v___x_6286_; 
v_a_6283_ = lean_array_uget_borrowed(v_as_6268_, v_i_6270_);
v_goal_6284_ = lean_ctor_get(v_a_6283_, 0);
v_goal_6285_ = lean_ctor_get(v_x_6266_, 0);
v___x_6286_ = l_Lean_instBEqMVarId_beq(v_goal_6284_, v_goal_6285_);
if (v___x_6286_ == 0)
{
lean_object* v___x_6287_; lean_object* v___x_6289_; 
lean_inc(v_a_6283_);
v___x_6287_ = lean_array_push(v_snd_6279_, v_a_6283_);
if (v_isShared_6282_ == 0)
{
lean_ctor_set(v___x_6281_, 1, v___x_6287_);
v___x_6289_ = v___x_6281_;
goto v_reusejp_6288_;
}
else
{
lean_object* v_reuseFailAlloc_6290_; 
v_reuseFailAlloc_6290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6290_, 0, v_fst_6278_);
lean_ctor_set(v_reuseFailAlloc_6290_, 1, v___x_6287_);
v___x_6289_ = v_reuseFailAlloc_6290_;
goto v_reusejp_6288_;
}
v_reusejp_6288_:
{
v_a_6273_ = v___x_6289_;
goto v___jp_6272_;
}
}
else
{
lean_object* v___x_6291_; lean_object* v___x_6292_; lean_object* v___x_6294_; 
lean_dec(v_fst_6278_);
v___x_6291_ = l_Array_append___redArg(v_snd_6279_, v_r_6267_);
v___x_6292_ = lean_box(v___x_6286_);
if (v_isShared_6282_ == 0)
{
lean_ctor_set(v___x_6281_, 1, v___x_6291_);
lean_ctor_set(v___x_6281_, 0, v___x_6292_);
v___x_6294_ = v___x_6281_;
goto v_reusejp_6293_;
}
else
{
lean_object* v_reuseFailAlloc_6295_; 
v_reuseFailAlloc_6295_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6295_, 0, v___x_6292_);
lean_ctor_set(v_reuseFailAlloc_6295_, 1, v___x_6291_);
v___x_6294_ = v_reuseFailAlloc_6295_;
goto v_reusejp_6293_;
}
v_reusejp_6293_:
{
v_a_6273_ = v___x_6294_;
goto v___jp_6272_;
}
}
}
}
v___jp_6272_:
{
size_t v___x_6274_; size_t v___x_6275_; 
v___x_6274_ = ((size_t)1ULL);
v___x_6275_ = lean_usize_add(v_i_6270_, v___x_6274_);
v_i_6270_ = v___x_6275_;
v_b_6271_ = v_a_6273_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9_spec__10___boxed(lean_object* v_x_6297_, lean_object* v_r_6298_, lean_object* v_as_6299_, lean_object* v_sz_6300_, lean_object* v_i_6301_, lean_object* v_b_6302_){
_start:
{
size_t v_sz_boxed_6303_; size_t v_i_boxed_6304_; lean_object* v_res_6305_; 
v_sz_boxed_6303_ = lean_unbox_usize(v_sz_6300_);
lean_dec(v_sz_6300_);
v_i_boxed_6304_ = lean_unbox_usize(v_i_6301_);
lean_dec(v_i_6301_);
v_res_6305_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9_spec__10(v_x_6297_, v_r_6298_, v_as_6299_, v_sz_boxed_6303_, v_i_boxed_6304_, v_b_6302_);
lean_dec_ref(v_as_6299_);
lean_dec_ref(v_r_6298_);
lean_dec_ref(v_x_6297_);
return v_res_6305_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9(lean_object* v_xs_6306_, lean_object* v_x_6307_, lean_object* v_r_6308_){
_start:
{
uint8_t v_found_6309_; lean_object* v___x_6310_; lean_object* v___x_6311_; lean_object* v___x_6312_; lean_object* v___x_6313_; lean_object* v___x_6314_; lean_object* v_ys_6315_; lean_object* v___x_6316_; lean_object* v___x_6317_; size_t v_sz_6318_; size_t v___x_6319_; lean_object* v___x_6320_; lean_object* v_fst_6321_; uint8_t v___x_6322_; 
v_found_6309_ = 0;
v___x_6310_ = lean_array_get_size(v_xs_6306_);
v___x_6311_ = lean_unsigned_to_nat(1u);
v___x_6312_ = lean_nat_sub(v___x_6310_, v___x_6311_);
v___x_6313_ = lean_array_get_size(v_r_6308_);
v___x_6314_ = lean_nat_add(v___x_6312_, v___x_6313_);
lean_dec(v___x_6312_);
v_ys_6315_ = lean_mk_empty_array_with_capacity(v___x_6314_);
lean_dec(v___x_6314_);
v___x_6316_ = lean_box(v_found_6309_);
v___x_6317_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6317_, 0, v___x_6316_);
lean_ctor_set(v___x_6317_, 1, v_ys_6315_);
v_sz_6318_ = lean_array_size(v_xs_6306_);
v___x_6319_ = ((size_t)0ULL);
v___x_6320_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9_spec__10(v_x_6307_, v_r_6308_, v_xs_6306_, v_sz_6318_, v___x_6319_, v___x_6317_);
v_fst_6321_ = lean_ctor_get(v___x_6320_, 0);
lean_inc(v_fst_6321_);
v___x_6322_ = lean_unbox(v_fst_6321_);
lean_dec(v_fst_6321_);
if (v___x_6322_ == 0)
{
lean_object* v___x_6323_; 
lean_dec_ref(v___x_6320_);
v___x_6323_ = lean_box(0);
return v___x_6323_;
}
else
{
lean_object* v_snd_6324_; lean_object* v___x_6325_; 
v_snd_6324_ = lean_ctor_get(v___x_6320_, 1);
lean_inc(v_snd_6324_);
lean_dec_ref(v___x_6320_);
v___x_6325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6325_, 0, v_snd_6324_);
return v___x_6325_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9___boxed(lean_object* v_xs_6326_, lean_object* v_x_6327_, lean_object* v_r_6328_){
_start:
{
lean_object* v_res_6329_; 
v_res_6329_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9(v_xs_6326_, v_x_6327_, v_r_6328_);
lean_dec_ref(v_r_6328_);
lean_dec_ref(v_x_6327_);
lean_dec_ref(v_xs_6326_);
return v_res_6329_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_6331_; lean_object* v___x_6332_; 
v___x_6331_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__0));
v___x_6332_ = l_Lean_stringToMessageData(v___x_6331_);
return v___x_6332_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_6334_; lean_object* v___x_6335_; 
v___x_6334_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__2));
v___x_6335_ = l_Lean_stringToMessageData(v___x_6334_);
return v___x_6335_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__5(void){
_start:
{
lean_object* v___x_6337_; lean_object* v___x_6338_; 
v___x_6337_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__4));
v___x_6338_ = l_Lean_stringToMessageData(v___x_6337_);
return v___x_6338_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg(lean_object* v_goal_6339_, lean_object* v_pre_6340_, lean_object* v___y_6341_, lean_object* v___y_6342_, lean_object* v___y_6343_, lean_object* v___y_6344_){
_start:
{
lean_object* v___x_6346_; lean_object* v___x_6347_; lean_object* v___x_6348_; lean_object* v___x_6349_; lean_object* v___x_6350_; lean_object* v___x_6351_; lean_object* v___x_6352_; lean_object* v___x_6353_; lean_object* v___x_6354_; 
v___x_6346_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__1, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_6347_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6347_, 0, v___x_6346_);
lean_ctor_set(v___x_6347_, 1, v_pre_6340_);
v___x_6348_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__3, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__3_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__3);
v___x_6349_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6349_, 0, v___x_6347_);
lean_ctor_set(v___x_6349_, 1, v___x_6348_);
v___x_6350_ = l_Lean_MessageData_ofName(v_goal_6339_);
v___x_6351_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6351_, 0, v___x_6349_);
lean_ctor_set(v___x_6351_, 1, v___x_6350_);
v___x_6352_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__5, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___closed__5);
v___x_6353_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_6353_, 0, v___x_6351_);
lean_ctor_set(v___x_6353_, 1, v___x_6352_);
v___x_6354_ = lp_aesop_Lean_throwError___at___00Aesop_getSingleGoal___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__1_spec__2___redArg(v___x_6353_, v___y_6341_, v___y_6342_, v___y_6343_, v___y_6344_);
return v___x_6354_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg___boxed(lean_object* v_goal_6355_, lean_object* v_pre_6356_, lean_object* v___y_6357_, lean_object* v___y_6358_, lean_object* v___y_6359_, lean_object* v___y_6360_, lean_object* v___y_6361_){
_start:
{
lean_object* v_res_6362_; 
v_res_6362_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg(v_goal_6355_, v_pre_6356_, v___y_6357_, v___y_6358_, v___y_6359_, v___y_6360_);
lean_dec(v___y_6360_);
lean_dec_ref(v___y_6359_);
lean_dec(v___y_6358_);
lean_dec_ref(v___y_6357_);
return v_res_6362_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__2(void){
_start:
{
lean_object* v___x_6366_; lean_object* v___x_6367_; 
v___x_6366_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__1));
v___x_6367_ = l_Lean_MessageData_ofFormat(v___x_6366_);
return v___x_6367_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7(lean_object* v_ts_6368_, lean_object* v_inGoal_6369_, lean_object* v_outGoals_6370_, lean_object* v_preMCtx_6371_, lean_object* v_postMCtx_6372_, lean_object* v___y_6373_, lean_object* v___y_6374_, lean_object* v___y_6375_, lean_object* v___y_6376_, lean_object* v___y_6377_, lean_object* v___y_6378_, lean_object* v___y_6379_){
_start:
{
lean_object* v_visibleGoals_6381_; lean_object* v_invisibleGoals_6382_; lean_object* v___x_6384_; uint8_t v_isShared_6385_; uint8_t v_isSharedCheck_6403_; 
v_visibleGoals_6381_ = lean_ctor_get(v_ts_6368_, 0);
v_invisibleGoals_6382_ = lean_ctor_get(v_ts_6368_, 1);
v_isSharedCheck_6403_ = !lean_is_exclusive(v_ts_6368_);
if (v_isSharedCheck_6403_ == 0)
{
v___x_6384_ = v_ts_6368_;
v_isShared_6385_ = v_isSharedCheck_6403_;
goto v_resetjp_6383_;
}
else
{
lean_inc(v_invisibleGoals_6382_);
lean_inc(v_visibleGoals_6381_);
lean_dec(v_ts_6368_);
v___x_6384_ = lean_box(0);
v_isShared_6385_ = v_isSharedCheck_6403_;
goto v_resetjp_6383_;
}
v_resetjp_6383_:
{
lean_object* v___x_6386_; lean_object* v___x_6387_; lean_object* v___x_6388_; 
v___x_6386_ = lean_obj_once(&lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1, &lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1_once, _init_lp_aesop_Aesop_Stateful_saturateCore___lam__0___closed__1);
lean_inc(v_inGoal_6369_);
v___x_6387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6387_, 0, v_inGoal_6369_);
lean_ctor_set(v___x_6387_, 1, v___x_6386_);
v___x_6388_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7_spec__9(v_visibleGoals_6381_, v___x_6387_, v_outGoals_6370_);
lean_dec_ref_known(v___x_6387_, 2);
lean_dec_ref(v_visibleGoals_6381_);
if (lean_obj_tag(v___x_6388_) == 1)
{
lean_object* v_val_6389_; lean_object* v___x_6391_; uint8_t v_isShared_6392_; uint8_t v_isSharedCheck_6400_; 
lean_dec(v_inGoal_6369_);
v_val_6389_ = lean_ctor_get(v___x_6388_, 0);
v_isSharedCheck_6400_ = !lean_is_exclusive(v___x_6388_);
if (v_isSharedCheck_6400_ == 0)
{
v___x_6391_ = v___x_6388_;
v_isShared_6392_ = v_isSharedCheck_6400_;
goto v_resetjp_6390_;
}
else
{
lean_inc(v_val_6389_);
lean_dec(v___x_6388_);
v___x_6391_ = lean_box(0);
v_isShared_6392_ = v_isSharedCheck_6400_;
goto v_resetjp_6390_;
}
v_resetjp_6390_:
{
lean_object* v_ts_6394_; 
if (v_isShared_6385_ == 0)
{
lean_ctor_set(v___x_6384_, 0, v_val_6389_);
v_ts_6394_ = v___x_6384_;
goto v_reusejp_6393_;
}
else
{
lean_object* v_reuseFailAlloc_6399_; 
v_reuseFailAlloc_6399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6399_, 0, v_val_6389_);
lean_ctor_set(v_reuseFailAlloc_6399_, 1, v_invisibleGoals_6382_);
v_ts_6394_ = v_reuseFailAlloc_6399_;
goto v_reusejp_6393_;
}
v_reusejp_6393_:
{
lean_object* v___x_6395_; lean_object* v___x_6397_; 
v___x_6395_ = lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(v_ts_6394_, v_preMCtx_6371_, v_postMCtx_6372_);
if (v_isShared_6392_ == 0)
{
lean_ctor_set_tag(v___x_6391_, 0);
lean_ctor_set(v___x_6391_, 0, v___x_6395_);
v___x_6397_ = v___x_6391_;
goto v_reusejp_6396_;
}
else
{
lean_object* v_reuseFailAlloc_6398_; 
v_reuseFailAlloc_6398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6398_, 0, v___x_6395_);
v___x_6397_ = v_reuseFailAlloc_6398_;
goto v_reusejp_6396_;
}
v_reusejp_6396_:
{
return v___x_6397_;
}
}
}
}
else
{
lean_object* v___x_6401_; lean_object* v___x_6402_; 
lean_dec(v___x_6388_);
lean_del_object(v___x_6384_);
lean_dec_ref(v_invisibleGoals_6382_);
v___x_6401_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__2, &lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__2_once, _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___closed__2);
v___x_6402_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg(v_inGoal_6369_, v___x_6401_, v___y_6376_, v___y_6377_, v___y_6378_, v___y_6379_);
return v___x_6402_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7___boxed(lean_object* v_ts_6404_, lean_object* v_inGoal_6405_, lean_object* v_outGoals_6406_, lean_object* v_preMCtx_6407_, lean_object* v_postMCtx_6408_, lean_object* v___y_6409_, lean_object* v___y_6410_, lean_object* v___y_6411_, lean_object* v___y_6412_, lean_object* v___y_6413_, lean_object* v___y_6414_, lean_object* v___y_6415_, lean_object* v___y_6416_){
_start:
{
lean_object* v_res_6417_; 
v_res_6417_ = lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7(v_ts_6404_, v_inGoal_6405_, v_outGoals_6406_, v_preMCtx_6407_, v_postMCtx_6408_, v___y_6409_, v___y_6410_, v___y_6411_, v___y_6412_, v___y_6413_, v___y_6414_, v___y_6415_);
lean_dec(v___y_6415_);
lean_dec_ref(v___y_6414_);
lean_dec(v___y_6413_);
lean_dec_ref(v___y_6412_);
lean_dec(v___y_6411_);
lean_dec(v___y_6410_);
lean_dec_ref(v___y_6409_);
lean_dec_ref(v_postMCtx_6408_);
lean_dec_ref(v_preMCtx_6407_);
lean_dec_ref(v_outGoals_6406_);
return v_res_6417_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5(lean_object* v_tacticState_6418_, lean_object* v_step_6419_, lean_object* v___y_6420_, lean_object* v___y_6421_, lean_object* v___y_6422_, lean_object* v___y_6423_, lean_object* v___y_6424_, lean_object* v___y_6425_, lean_object* v___y_6426_){
_start:
{
lean_object* v_preState_6428_; lean_object* v_meta_6429_; lean_object* v_postState_6430_; lean_object* v_meta_6431_; lean_object* v_preGoal_6432_; lean_object* v_postGoals_6433_; lean_object* v_mctx_6434_; lean_object* v_mctx_6435_; lean_object* v___x_6436_; 
v_preState_6428_ = lean_ctor_get(v_step_6419_, 0);
v_meta_6429_ = lean_ctor_get(v_preState_6428_, 1);
lean_inc_ref(v_meta_6429_);
v_postState_6430_ = lean_ctor_get(v_step_6419_, 3);
v_meta_6431_ = lean_ctor_get(v_postState_6430_, 1);
lean_inc_ref(v_meta_6431_);
v_preGoal_6432_ = lean_ctor_get(v_step_6419_, 1);
lean_inc(v_preGoal_6432_);
v_postGoals_6433_ = lean_ctor_get(v_step_6419_, 4);
lean_inc_ref(v_postGoals_6433_);
lean_dec_ref(v_step_6419_);
v_mctx_6434_ = lean_ctor_get(v_meta_6429_, 0);
lean_inc_ref(v_mctx_6434_);
lean_dec_ref(v_meta_6429_);
v_mctx_6435_ = lean_ctor_get(v_meta_6431_, 0);
lean_inc_ref(v_mctx_6435_);
lean_dec_ref(v_meta_6431_);
v___x_6436_ = lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5_spec__7(v_tacticState_6418_, v_preGoal_6432_, v_postGoals_6433_, v_mctx_6434_, v_mctx_6435_, v___y_6420_, v___y_6421_, v___y_6422_, v___y_6423_, v___y_6424_, v___y_6425_, v___y_6426_);
lean_dec_ref(v_mctx_6435_);
lean_dec_ref(v_mctx_6434_);
lean_dec_ref(v_postGoals_6433_);
return v___x_6436_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5___boxed(lean_object* v_tacticState_6437_, lean_object* v_step_6438_, lean_object* v___y_6439_, lean_object* v___y_6440_, lean_object* v___y_6441_, lean_object* v___y_6442_, lean_object* v___y_6443_, lean_object* v___y_6444_, lean_object* v___y_6445_, lean_object* v___y_6446_){
_start:
{
lean_object* v_res_6447_; 
v_res_6447_ = lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5(v_tacticState_6437_, v_step_6438_, v___y_6439_, v___y_6440_, v___y_6441_, v___y_6442_, v___y_6443_, v___y_6444_, v___y_6445_);
lean_dec(v___y_6445_);
lean_dec_ref(v___y_6444_);
lean_dec(v___y_6443_);
lean_dec_ref(v___y_6442_);
lean_dec(v___y_6441_);
lean_dec(v___y_6440_);
lean_dec_ref(v___y_6439_);
return v_res_6447_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__2(void){
_start:
{
lean_object* v___x_6451_; lean_object* v___x_6452_; 
v___x_6451_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__1));
v___x_6452_ = l_Lean_MessageData_ofFormat(v___x_6451_);
return v___x_6452_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4(lean_object* v_ts_6453_, lean_object* v_goal_6454_, lean_object* v___y_6455_, lean_object* v___y_6456_, lean_object* v___y_6457_, lean_object* v___y_6458_, lean_object* v___y_6459_, lean_object* v___y_6460_, lean_object* v___y_6461_){
_start:
{
lean_object* v___x_6463_; 
v___x_6463_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(v_ts_6453_, v_goal_6454_);
if (lean_obj_tag(v___x_6463_) == 1)
{
lean_object* v_val_6464_; lean_object* v___x_6466_; uint8_t v_isShared_6467_; uint8_t v_isSharedCheck_6471_; 
lean_dec(v_goal_6454_);
v_val_6464_ = lean_ctor_get(v___x_6463_, 0);
v_isSharedCheck_6471_ = !lean_is_exclusive(v___x_6463_);
if (v_isSharedCheck_6471_ == 0)
{
v___x_6466_ = v___x_6463_;
v_isShared_6467_ = v_isSharedCheck_6471_;
goto v_resetjp_6465_;
}
else
{
lean_inc(v_val_6464_);
lean_dec(v___x_6463_);
v___x_6466_ = lean_box(0);
v_isShared_6467_ = v_isSharedCheck_6471_;
goto v_resetjp_6465_;
}
v_resetjp_6465_:
{
lean_object* v___x_6469_; 
if (v_isShared_6467_ == 0)
{
lean_ctor_set_tag(v___x_6466_, 0);
v___x_6469_ = v___x_6466_;
goto v_reusejp_6468_;
}
else
{
lean_object* v_reuseFailAlloc_6470_; 
v_reuseFailAlloc_6470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6470_, 0, v_val_6464_);
v___x_6469_ = v_reuseFailAlloc_6470_;
goto v_reusejp_6468_;
}
v_reusejp_6468_:
{
return v___x_6469_;
}
}
}
else
{
lean_object* v___x_6472_; lean_object* v___x_6473_; 
lean_dec(v___x_6463_);
v___x_6472_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__2, &lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__2_once, _init_lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___closed__2);
v___x_6473_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg(v_goal_6454_, v___x_6472_, v___y_6458_, v___y_6459_, v___y_6460_, v___y_6461_);
return v___x_6473_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4___boxed(lean_object* v_ts_6474_, lean_object* v_goal_6475_, lean_object* v___y_6476_, lean_object* v___y_6477_, lean_object* v___y_6478_, lean_object* v___y_6479_, lean_object* v___y_6480_, lean_object* v___y_6481_, lean_object* v___y_6482_, lean_object* v___y_6483_){
_start:
{
lean_object* v_res_6484_; 
v_res_6484_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4(v_ts_6474_, v_goal_6475_, v___y_6476_, v___y_6477_, v___y_6478_, v___y_6479_, v___y_6480_, v___y_6481_, v___y_6482_);
lean_dec(v___y_6482_);
lean_dec_ref(v___y_6481_);
lean_dec(v___y_6480_);
lean_dec_ref(v___y_6479_);
lean_dec(v___y_6478_);
lean_dec(v___y_6477_);
lean_dec_ref(v___y_6476_);
lean_dec_ref(v_ts_6474_);
return v_res_6484_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3(lean_object* v_acc_6485_, lean_object* v_step_6486_, lean_object* v_tacticState_6487_, lean_object* v___y_6488_, lean_object* v___y_6489_, lean_object* v___y_6490_, lean_object* v___y_6491_, lean_object* v___y_6492_, lean_object* v___y_6493_, lean_object* v___y_6494_){
_start:
{
lean_object* v_preGoal_6496_; lean_object* v_tactic_6497_; lean_object* v___x_6498_; 
v_preGoal_6496_ = lean_ctor_get(v_step_6486_, 1);
v_tactic_6497_ = lean_ctor_get(v_step_6486_, 2);
lean_inc_ref(v_tactic_6497_);
lean_inc(v_preGoal_6496_);
v___x_6498_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4(v_tacticState_6487_, v_preGoal_6496_, v___y_6488_, v___y_6489_, v___y_6490_, v___y_6491_, v___y_6492_, v___y_6493_, v___y_6494_);
if (lean_obj_tag(v___x_6498_) == 0)
{
lean_object* v_a_6499_; lean_object* v___x_6500_; 
v_a_6499_ = lean_ctor_get(v___x_6498_, 0);
lean_inc(v_a_6499_);
lean_dec_ref_known(v___x_6498_, 1);
v___x_6500_ = lp_aesop_Aesop_Script_TacticState_applyStep___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__5(v_tacticState_6487_, v_step_6486_, v___y_6488_, v___y_6489_, v___y_6490_, v___y_6491_, v___y_6492_, v___y_6493_, v___y_6494_);
if (lean_obj_tag(v___x_6500_) == 0)
{
lean_object* v_a_6501_; lean_object* v___x_6503_; uint8_t v_isShared_6504_; uint8_t v_isSharedCheck_6519_; 
v_a_6501_ = lean_ctor_get(v___x_6500_, 0);
v_isSharedCheck_6519_ = !lean_is_exclusive(v___x_6500_);
if (v_isSharedCheck_6519_ == 0)
{
v___x_6503_ = v___x_6500_;
v_isShared_6504_ = v_isSharedCheck_6519_;
goto v_resetjp_6502_;
}
else
{
lean_inc(v_a_6501_);
lean_dec(v___x_6500_);
v___x_6503_ = lean_box(0);
v_isShared_6504_ = v_isSharedCheck_6519_;
goto v_resetjp_6502_;
}
v_resetjp_6502_:
{
lean_object* v_uTactic_6505_; lean_object* v___x_6507_; uint8_t v_isShared_6508_; uint8_t v_isSharedCheck_6517_; 
v_uTactic_6505_ = lean_ctor_get(v_tactic_6497_, 0);
v_isSharedCheck_6517_ = !lean_is_exclusive(v_tactic_6497_);
if (v_isSharedCheck_6517_ == 0)
{
lean_object* v_unused_6518_; 
v_unused_6518_ = lean_ctor_get(v_tactic_6497_, 1);
lean_dec(v_unused_6518_);
v___x_6507_ = v_tactic_6497_;
v_isShared_6508_ = v_isSharedCheck_6517_;
goto v_resetjp_6506_;
}
else
{
lean_inc(v_uTactic_6505_);
lean_dec(v_tactic_6497_);
v___x_6507_ = lean_box(0);
v_isShared_6508_ = v_isSharedCheck_6517_;
goto v_resetjp_6506_;
}
v_resetjp_6506_:
{
lean_object* v___x_6509_; lean_object* v_acc_6510_; lean_object* v___x_6512_; 
v___x_6509_ = lp_aesop_Aesop_Script_mkOnGoal(v_a_6499_, v_uTactic_6505_);
lean_dec(v_a_6499_);
v_acc_6510_ = lean_array_push(v_acc_6485_, v___x_6509_);
if (v_isShared_6508_ == 0)
{
lean_ctor_set(v___x_6507_, 1, v_a_6501_);
lean_ctor_set(v___x_6507_, 0, v_acc_6510_);
v___x_6512_ = v___x_6507_;
goto v_reusejp_6511_;
}
else
{
lean_object* v_reuseFailAlloc_6516_; 
v_reuseFailAlloc_6516_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6516_, 0, v_acc_6510_);
lean_ctor_set(v_reuseFailAlloc_6516_, 1, v_a_6501_);
v___x_6512_ = v_reuseFailAlloc_6516_;
goto v_reusejp_6511_;
}
v_reusejp_6511_:
{
lean_object* v___x_6514_; 
if (v_isShared_6504_ == 0)
{
lean_ctor_set(v___x_6503_, 0, v___x_6512_);
v___x_6514_ = v___x_6503_;
goto v_reusejp_6513_;
}
else
{
lean_object* v_reuseFailAlloc_6515_; 
v_reuseFailAlloc_6515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6515_, 0, v___x_6512_);
v___x_6514_ = v_reuseFailAlloc_6515_;
goto v_reusejp_6513_;
}
v_reusejp_6513_:
{
return v___x_6514_;
}
}
}
}
}
else
{
lean_object* v_a_6520_; lean_object* v___x_6522_; uint8_t v_isShared_6523_; uint8_t v_isSharedCheck_6527_; 
lean_dec(v_a_6499_);
lean_dec_ref(v_tactic_6497_);
lean_dec_ref(v_acc_6485_);
v_a_6520_ = lean_ctor_get(v___x_6500_, 0);
v_isSharedCheck_6527_ = !lean_is_exclusive(v___x_6500_);
if (v_isSharedCheck_6527_ == 0)
{
v___x_6522_ = v___x_6500_;
v_isShared_6523_ = v_isSharedCheck_6527_;
goto v_resetjp_6521_;
}
else
{
lean_inc(v_a_6520_);
lean_dec(v___x_6500_);
v___x_6522_ = lean_box(0);
v_isShared_6523_ = v_isSharedCheck_6527_;
goto v_resetjp_6521_;
}
v_resetjp_6521_:
{
lean_object* v___x_6525_; 
if (v_isShared_6523_ == 0)
{
v___x_6525_ = v___x_6522_;
goto v_reusejp_6524_;
}
else
{
lean_object* v_reuseFailAlloc_6526_; 
v_reuseFailAlloc_6526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6526_, 0, v_a_6520_);
v___x_6525_ = v_reuseFailAlloc_6526_;
goto v_reusejp_6524_;
}
v_reusejp_6524_:
{
return v___x_6525_;
}
}
}
}
else
{
lean_object* v_a_6528_; lean_object* v___x_6530_; uint8_t v_isShared_6531_; uint8_t v_isSharedCheck_6535_; 
lean_dec_ref(v_tactic_6497_);
lean_dec_ref(v_tacticState_6487_);
lean_dec_ref(v_step_6486_);
lean_dec_ref(v_acc_6485_);
v_a_6528_ = lean_ctor_get(v___x_6498_, 0);
v_isSharedCheck_6535_ = !lean_is_exclusive(v___x_6498_);
if (v_isSharedCheck_6535_ == 0)
{
v___x_6530_ = v___x_6498_;
v_isShared_6531_ = v_isSharedCheck_6535_;
goto v_resetjp_6529_;
}
else
{
lean_inc(v_a_6528_);
lean_dec(v___x_6498_);
v___x_6530_ = lean_box(0);
v_isShared_6531_ = v_isSharedCheck_6535_;
goto v_resetjp_6529_;
}
v_resetjp_6529_:
{
lean_object* v___x_6533_; 
if (v_isShared_6531_ == 0)
{
v___x_6533_ = v___x_6530_;
goto v_reusejp_6532_;
}
else
{
lean_object* v_reuseFailAlloc_6534_; 
v_reuseFailAlloc_6534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6534_, 0, v_a_6528_);
v___x_6533_ = v_reuseFailAlloc_6534_;
goto v_reusejp_6532_;
}
v_reusejp_6532_:
{
return v___x_6533_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3___boxed(lean_object* v_acc_6536_, lean_object* v_step_6537_, lean_object* v_tacticState_6538_, lean_object* v___y_6539_, lean_object* v___y_6540_, lean_object* v___y_6541_, lean_object* v___y_6542_, lean_object* v___y_6543_, lean_object* v___y_6544_, lean_object* v___y_6545_, lean_object* v___y_6546_){
_start:
{
lean_object* v_res_6547_; 
v_res_6547_ = lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3(v_acc_6536_, v_step_6537_, v_tacticState_6538_, v___y_6539_, v___y_6540_, v___y_6541_, v___y_6542_, v___y_6543_, v___y_6544_, v___y_6545_);
lean_dec(v___y_6545_);
lean_dec_ref(v___y_6544_);
lean_dec(v___y_6543_);
lean_dec_ref(v___y_6542_);
lean_dec(v___y_6541_);
lean_dec(v___y_6540_);
lean_dec_ref(v___y_6539_);
return v_res_6547_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__4(lean_object* v_as_6548_, size_t v_sz_6549_, size_t v_i_6550_, lean_object* v_b_6551_, lean_object* v___y_6552_, lean_object* v___y_6553_, lean_object* v___y_6554_, lean_object* v___y_6555_, lean_object* v___y_6556_, lean_object* v___y_6557_, lean_object* v___y_6558_){
_start:
{
uint8_t v___x_6560_; 
v___x_6560_ = lean_usize_dec_lt(v_i_6550_, v_sz_6549_);
if (v___x_6560_ == 0)
{
lean_object* v___x_6561_; 
v___x_6561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6561_, 0, v_b_6551_);
return v___x_6561_;
}
else
{
lean_object* v_fst_6562_; lean_object* v_snd_6563_; lean_object* v_a_6564_; lean_object* v___x_6565_; 
v_fst_6562_ = lean_ctor_get(v_b_6551_, 0);
lean_inc(v_fst_6562_);
v_snd_6563_ = lean_ctor_get(v_b_6551_, 1);
lean_inc(v_snd_6563_);
lean_dec_ref(v_b_6551_);
v_a_6564_ = lean_array_uget_borrowed(v_as_6548_, v_i_6550_);
lean_inc(v_a_6564_);
v___x_6565_ = lp_aesop_Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3(v_fst_6562_, v_a_6564_, v_snd_6563_, v___y_6552_, v___y_6553_, v___y_6554_, v___y_6555_, v___y_6556_, v___y_6557_, v___y_6558_);
if (lean_obj_tag(v___x_6565_) == 0)
{
lean_object* v_a_6566_; size_t v___x_6567_; size_t v___x_6568_; 
v_a_6566_ = lean_ctor_get(v___x_6565_, 0);
lean_inc(v_a_6566_);
lean_dec_ref_known(v___x_6565_, 1);
v___x_6567_ = ((size_t)1ULL);
v___x_6568_ = lean_usize_add(v_i_6550_, v___x_6567_);
v_i_6550_ = v___x_6568_;
v_b_6551_ = v_a_6566_;
goto _start;
}
else
{
return v___x_6565_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__4___boxed(lean_object* v_as_6570_, lean_object* v_sz_6571_, lean_object* v_i_6572_, lean_object* v_b_6573_, lean_object* v___y_6574_, lean_object* v___y_6575_, lean_object* v___y_6576_, lean_object* v___y_6577_, lean_object* v___y_6578_, lean_object* v___y_6579_, lean_object* v___y_6580_, lean_object* v___y_6581_){
_start:
{
size_t v_sz_boxed_6582_; size_t v_i_boxed_6583_; lean_object* v_res_6584_; 
v_sz_boxed_6582_ = lean_unbox_usize(v_sz_6571_);
lean_dec(v_sz_6571_);
v_i_boxed_6583_ = lean_unbox_usize(v_i_6572_);
lean_dec(v_i_6572_);
v_res_6584_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__4(v_as_6570_, v_sz_boxed_6582_, v_i_boxed_6583_, v_b_6573_, v___y_6574_, v___y_6575_, v___y_6576_, v___y_6577_, v___y_6578_, v___y_6579_, v___y_6580_);
lean_dec(v___y_6580_);
lean_dec_ref(v___y_6579_);
lean_dec(v___y_6578_);
lean_dec_ref(v___y_6577_);
lean_dec(v___y_6576_);
lean_dec(v___y_6575_);
lean_dec_ref(v___y_6574_);
lean_dec_ref(v_as_6570_);
return v_res_6584_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2(lean_object* v_tacticState_6585_, lean_object* v_s_6586_, lean_object* v___y_6587_, lean_object* v___y_6588_, lean_object* v___y_6589_, lean_object* v___y_6590_, lean_object* v___y_6591_, lean_object* v___y_6592_, lean_object* v___y_6593_){
_start:
{
lean_object* v___x_6595_; lean_object* v_script_6596_; lean_object* v___x_6597_; size_t v_sz_6598_; size_t v___x_6599_; lean_object* v___x_6600_; 
v___x_6595_ = lean_array_get_size(v_s_6586_);
v_script_6596_ = lean_mk_empty_array_with_capacity(v___x_6595_);
v___x_6597_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6597_, 0, v_script_6596_);
lean_ctor_set(v___x_6597_, 1, v_tacticState_6585_);
v_sz_6598_ = lean_array_size(v_s_6586_);
v___x_6599_ = ((size_t)0ULL);
v___x_6600_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__4(v_s_6586_, v_sz_6598_, v___x_6599_, v___x_6597_, v___y_6587_, v___y_6588_, v___y_6589_, v___y_6590_, v___y_6591_, v___y_6592_, v___y_6593_);
if (lean_obj_tag(v___x_6600_) == 0)
{
lean_object* v_a_6601_; lean_object* v___x_6603_; uint8_t v_isShared_6604_; uint8_t v_isSharedCheck_6609_; 
v_a_6601_ = lean_ctor_get(v___x_6600_, 0);
v_isSharedCheck_6609_ = !lean_is_exclusive(v___x_6600_);
if (v_isSharedCheck_6609_ == 0)
{
v___x_6603_ = v___x_6600_;
v_isShared_6604_ = v_isSharedCheck_6609_;
goto v_resetjp_6602_;
}
else
{
lean_inc(v_a_6601_);
lean_dec(v___x_6600_);
v___x_6603_ = lean_box(0);
v_isShared_6604_ = v_isSharedCheck_6609_;
goto v_resetjp_6602_;
}
v_resetjp_6602_:
{
lean_object* v_fst_6605_; lean_object* v___x_6607_; 
v_fst_6605_ = lean_ctor_get(v_a_6601_, 0);
lean_inc(v_fst_6605_);
lean_dec(v_a_6601_);
if (v_isShared_6604_ == 0)
{
lean_ctor_set(v___x_6603_, 0, v_fst_6605_);
v___x_6607_ = v___x_6603_;
goto v_reusejp_6606_;
}
else
{
lean_object* v_reuseFailAlloc_6608_; 
v_reuseFailAlloc_6608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6608_, 0, v_fst_6605_);
v___x_6607_ = v_reuseFailAlloc_6608_;
goto v_reusejp_6606_;
}
v_reusejp_6606_:
{
return v___x_6607_;
}
}
}
else
{
lean_object* v_a_6610_; lean_object* v___x_6612_; uint8_t v_isShared_6613_; uint8_t v_isSharedCheck_6617_; 
v_a_6610_ = lean_ctor_get(v___x_6600_, 0);
v_isSharedCheck_6617_ = !lean_is_exclusive(v___x_6600_);
if (v_isSharedCheck_6617_ == 0)
{
v___x_6612_ = v___x_6600_;
v_isShared_6613_ = v_isSharedCheck_6617_;
goto v_resetjp_6611_;
}
else
{
lean_inc(v_a_6610_);
lean_dec(v___x_6600_);
v___x_6612_ = lean_box(0);
v_isShared_6613_ = v_isSharedCheck_6617_;
goto v_resetjp_6611_;
}
v_resetjp_6611_:
{
lean_object* v___x_6615_; 
if (v_isShared_6613_ == 0)
{
v___x_6615_ = v___x_6612_;
goto v_reusejp_6614_;
}
else
{
lean_object* v_reuseFailAlloc_6616_; 
v_reuseFailAlloc_6616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6616_, 0, v_a_6610_);
v___x_6615_ = v_reuseFailAlloc_6616_;
goto v_reusejp_6614_;
}
v_reusejp_6614_:
{
return v___x_6615_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2___boxed(lean_object* v_tacticState_6618_, lean_object* v_s_6619_, lean_object* v___y_6620_, lean_object* v___y_6621_, lean_object* v___y_6622_, lean_object* v___y_6623_, lean_object* v___y_6624_, lean_object* v___y_6625_, lean_object* v___y_6626_, lean_object* v___y_6627_){
_start:
{
lean_object* v_res_6628_; 
v_res_6628_ = lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2(v_tacticState_6618_, v_s_6619_, v___y_6620_, v___y_6621_, v___y_6622_, v___y_6623_, v___y_6624_, v___y_6625_, v___y_6626_);
lean_dec(v___y_6626_);
lean_dec_ref(v___y_6625_);
lean_dec(v___y_6624_);
lean_dec_ref(v___y_6623_);
lean_dec(v___y_6622_);
lean_dec(v___y_6621_);
lean_dec_ref(v___y_6620_);
lean_dec_ref(v_s_6619_);
return v_res_6628_;
}
}
static lean_object* _init_lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7(void){
_start:
{
lean_object* v___x_6641_; 
v___x_6641_ = l_Array_mkArray0(lean_box(0));
return v___x_6641_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0(lean_object* v_a_6648_, lean_object* v_goal_6649_, lean_object* v_rs_6650_, lean_object* v_tacticState_6651_, lean_object* v___y_6652_, lean_object* v___y_6653_, lean_object* v___y_6654_, lean_object* v___y_6655_, lean_object* v___y_6656_, lean_object* v___y_6657_, lean_object* v___y_6658_){
_start:
{
lean_object* v_options_6660_; lean_object* v_ref_6661_; lean_object* v___y_6663_; lean_object* v___y_6664_; uint8_t v___y_6665_; lean_object* v_a_6666_; lean_object* v___y_6716_; lean_object* v___y_6717_; uint8_t v___y_6718_; lean_object* v___y_6719_; lean_object* v___y_6787_; lean_object* v___y_6788_; uint8_t v___y_6789_; lean_object* v___y_6790_; uint8_t v_a_6791_; lean_object* v___y_6826_; lean_object* v___y_6827_; lean_object* v___y_6828_; uint8_t v___y_6829_; lean_object* v___y_6830_; lean_object* v_a_6834_; lean_object* v___y_6850_; lean_object* v___y_6851_; lean_object* v___y_6890_; lean_object* v___y_6898_; lean_object* v___x_6901_; uint8_t v___x_6902_; 
v_options_6660_ = lean_ctor_get(v___y_6657_, 2);
v_ref_6661_ = lean_ctor_get(v___y_6657_, 5);
v___x_6901_ = lp_aesop_Aesop_aesop_collectStats;
v___x_6902_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_6660_, v___x_6901_);
if (v___x_6902_ == 0)
{
lean_object* v___x_6903_; lean_object* v___x_6904_; lean_object* v_a_6905_; uint8_t v___x_6906_; 
v___x_6903_ = lp_aesop_Aesop_TraceOption_stats;
v___x_6904_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_6903_, v___y_6657_);
v_a_6905_ = lean_ctor_get(v___x_6904_, 0);
lean_inc(v_a_6905_);
v___x_6906_ = lean_unbox(v_a_6905_);
lean_dec(v_a_6905_);
if (v___x_6906_ == 0)
{
lean_object* v___x_6907_; lean_object* v___x_6908_; lean_object* v___x_6909_; uint8_t v___x_6910_; 
lean_dec_ref(v___x_6904_);
v___x_6907_ = lp_aesop_Aesop_aesop_stats_file;
v___x_6908_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_6660_, v___x_6907_);
v___x_6909_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_6910_ = lean_string_dec_eq(v___x_6908_, v___x_6909_);
lean_dec_ref(v___x_6908_);
if (v___x_6910_ == 0)
{
goto v___jp_6883_;
}
else
{
goto v___jp_6892_;
}
}
else
{
v___y_6898_ = v___x_6904_;
goto v___jp_6897_;
}
}
else
{
goto v___jp_6883_;
}
v___jp_6662_:
{
uint8_t v___x_6667_; uint8_t v___x_6668_; lean_object* v___x_6669_; lean_object* v___x_6670_; 
v___x_6667_ = 0;
v___x_6668_ = 0;
v___x_6669_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v___x_6669_, 0, v___x_6667_);
lean_ctor_set_uint8(v___x_6669_, 1, v___y_6665_);
lean_ctor_set_uint8(v___x_6669_, 2, v___x_6668_);
v___x_6670_ = lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0(v___x_6669_, v___y_6652_, v___y_6653_, v___y_6654_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
if (lean_obj_tag(v___x_6670_) == 0)
{
lean_object* v___x_6671_; 
lean_dec_ref_known(v___x_6670_, 1);
lean_inc(v_a_6666_);
v___x_6671_ = lp_aesop_Aesop_checkRenderedScriptIfEnabled(v_a_6666_, v_a_6648_, v_goal_6649_, v___x_6668_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
if (lean_obj_tag(v___x_6671_) == 0)
{
lean_object* v___x_6673_; uint8_t v_isShared_6674_; uint8_t v_isSharedCheck_6697_; 
v_isSharedCheck_6697_ = !lean_is_exclusive(v___x_6671_);
if (v_isSharedCheck_6697_ == 0)
{
lean_object* v_unused_6698_; 
v_unused_6698_ = lean_ctor_get(v___x_6671_, 0);
lean_dec(v_unused_6698_);
v___x_6673_ = v___x_6671_;
v_isShared_6674_ = v_isSharedCheck_6697_;
goto v_resetjp_6672_;
}
else
{
lean_dec(v___x_6671_);
v___x_6673_ = lean_box(0);
v_isShared_6674_ = v_isSharedCheck_6697_;
goto v_resetjp_6672_;
}
v_resetjp_6672_:
{
uint8_t v_traceScript_6675_; 
v_traceScript_6675_ = lean_ctor_get_uint8(v___y_6663_, sizeof(void*)*6 + 6);
if (v_traceScript_6675_ == 0)
{
lean_object* v___x_6677_; 
lean_dec(v_a_6666_);
if (v_isShared_6674_ == 0)
{
lean_ctor_set(v___x_6673_, 0, v___y_6664_);
v___x_6677_ = v___x_6673_;
goto v_reusejp_6676_;
}
else
{
lean_object* v_reuseFailAlloc_6678_; 
v_reuseFailAlloc_6678_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6678_, 0, v___y_6664_);
v___x_6677_ = v_reuseFailAlloc_6678_;
goto v_reusejp_6676_;
}
v_reusejp_6676_:
{
return v___x_6677_;
}
}
else
{
lean_object* v___x_6679_; lean_object* v___x_6680_; 
lean_del_object(v___x_6673_);
v___x_6679_ = lean_box(0);
lean_inc(v_ref_6661_);
v___x_6680_ = lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg(v_ref_6661_, v_a_6666_, v___x_6679_, v___y_6657_, v___y_6658_);
if (lean_obj_tag(v___x_6680_) == 0)
{
lean_object* v___x_6682_; uint8_t v_isShared_6683_; uint8_t v_isSharedCheck_6687_; 
v_isSharedCheck_6687_ = !lean_is_exclusive(v___x_6680_);
if (v_isSharedCheck_6687_ == 0)
{
lean_object* v_unused_6688_; 
v_unused_6688_ = lean_ctor_get(v___x_6680_, 0);
lean_dec(v_unused_6688_);
v___x_6682_ = v___x_6680_;
v_isShared_6683_ = v_isSharedCheck_6687_;
goto v_resetjp_6681_;
}
else
{
lean_dec(v___x_6680_);
v___x_6682_ = lean_box(0);
v_isShared_6683_ = v_isSharedCheck_6687_;
goto v_resetjp_6681_;
}
v_resetjp_6681_:
{
lean_object* v___x_6685_; 
if (v_isShared_6683_ == 0)
{
lean_ctor_set(v___x_6682_, 0, v___y_6664_);
v___x_6685_ = v___x_6682_;
goto v_reusejp_6684_;
}
else
{
lean_object* v_reuseFailAlloc_6686_; 
v_reuseFailAlloc_6686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6686_, 0, v___y_6664_);
v___x_6685_ = v_reuseFailAlloc_6686_;
goto v_reusejp_6684_;
}
v_reusejp_6684_:
{
return v___x_6685_;
}
}
}
else
{
lean_object* v_a_6689_; lean_object* v___x_6691_; uint8_t v_isShared_6692_; uint8_t v_isSharedCheck_6696_; 
lean_dec(v___y_6664_);
v_a_6689_ = lean_ctor_get(v___x_6680_, 0);
v_isSharedCheck_6696_ = !lean_is_exclusive(v___x_6680_);
if (v_isSharedCheck_6696_ == 0)
{
v___x_6691_ = v___x_6680_;
v_isShared_6692_ = v_isSharedCheck_6696_;
goto v_resetjp_6690_;
}
else
{
lean_inc(v_a_6689_);
lean_dec(v___x_6680_);
v___x_6691_ = lean_box(0);
v_isShared_6692_ = v_isSharedCheck_6696_;
goto v_resetjp_6690_;
}
v_resetjp_6690_:
{
lean_object* v___x_6694_; 
if (v_isShared_6692_ == 0)
{
v___x_6694_ = v___x_6691_;
goto v_reusejp_6693_;
}
else
{
lean_object* v_reuseFailAlloc_6695_; 
v_reuseFailAlloc_6695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6695_, 0, v_a_6689_);
v___x_6694_ = v_reuseFailAlloc_6695_;
goto v_reusejp_6693_;
}
v_reusejp_6693_:
{
return v___x_6694_;
}
}
}
}
}
}
else
{
lean_object* v_a_6699_; lean_object* v___x_6701_; uint8_t v_isShared_6702_; uint8_t v_isSharedCheck_6706_; 
lean_dec(v_a_6666_);
lean_dec(v___y_6664_);
v_a_6699_ = lean_ctor_get(v___x_6671_, 0);
v_isSharedCheck_6706_ = !lean_is_exclusive(v___x_6671_);
if (v_isSharedCheck_6706_ == 0)
{
v___x_6701_ = v___x_6671_;
v_isShared_6702_ = v_isSharedCheck_6706_;
goto v_resetjp_6700_;
}
else
{
lean_inc(v_a_6699_);
lean_dec(v___x_6671_);
v___x_6701_ = lean_box(0);
v_isShared_6702_ = v_isSharedCheck_6706_;
goto v_resetjp_6700_;
}
v_resetjp_6700_:
{
lean_object* v___x_6704_; 
if (v_isShared_6702_ == 0)
{
v___x_6704_ = v___x_6701_;
goto v_reusejp_6703_;
}
else
{
lean_object* v_reuseFailAlloc_6705_; 
v_reuseFailAlloc_6705_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6705_, 0, v_a_6699_);
v___x_6704_ = v_reuseFailAlloc_6705_;
goto v_reusejp_6703_;
}
v_reusejp_6703_:
{
return v___x_6704_;
}
}
}
}
else
{
lean_object* v_a_6707_; lean_object* v___x_6709_; uint8_t v_isShared_6710_; uint8_t v_isSharedCheck_6714_; 
lean_dec(v_a_6666_);
lean_dec(v___y_6664_);
lean_dec(v_goal_6649_);
lean_dec_ref(v_a_6648_);
v_a_6707_ = lean_ctor_get(v___x_6670_, 0);
v_isSharedCheck_6714_ = !lean_is_exclusive(v___x_6670_);
if (v_isSharedCheck_6714_ == 0)
{
v___x_6709_ = v___x_6670_;
v_isShared_6710_ = v_isSharedCheck_6714_;
goto v_resetjp_6708_;
}
else
{
lean_inc(v_a_6707_);
lean_dec(v___x_6670_);
v___x_6709_ = lean_box(0);
v_isShared_6710_ = v_isSharedCheck_6714_;
goto v_resetjp_6708_;
}
v_resetjp_6708_:
{
lean_object* v___x_6712_; 
if (v_isShared_6710_ == 0)
{
v___x_6712_ = v___x_6709_;
goto v_reusejp_6711_;
}
else
{
lean_object* v_reuseFailAlloc_6713_; 
v_reuseFailAlloc_6713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6713_, 0, v_a_6707_);
v___x_6712_ = v_reuseFailAlloc_6713_;
goto v_reusejp_6711_;
}
v_reusejp_6711_:
{
return v___x_6712_;
}
}
}
}
v___jp_6715_:
{
lean_object* v___x_6720_; size_t v_sz_6721_; size_t v___x_6722_; lean_object* v___x_6723_; 
v___x_6720_ = lean_io_mono_nanos_now();
v_sz_6721_ = lean_array_size(v___y_6717_);
v___x_6722_ = ((size_t)0ULL);
v___x_6723_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg(v_sz_6721_, v___x_6722_, v___y_6717_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
if (lean_obj_tag(v___x_6723_) == 0)
{
lean_object* v_a_6724_; lean_object* v___x_6725_; 
v_a_6724_ = lean_ctor_get(v___x_6723_, 0);
lean_inc(v_a_6724_);
lean_dec_ref_known(v___x_6723_, 1);
v___x_6725_ = lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2(v_tacticState_6651_, v_a_6724_, v___y_6652_, v___y_6653_, v___y_6654_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
lean_dec(v_a_6724_);
if (lean_obj_tag(v___x_6725_) == 0)
{
lean_object* v_a_6726_; lean_object* v___x_6727_; lean_object* v___x_6728_; lean_object* v_stats_6729_; lean_object* v_rulePatternCache_6730_; lean_object* v___x_6732_; uint8_t v_isShared_6733_; uint8_t v_isSharedCheck_6769_; 
v_a_6726_ = lean_ctor_get(v___x_6725_, 0);
lean_inc(v_a_6726_);
lean_dec_ref_known(v___x_6725_, 1);
v___x_6727_ = lean_io_mono_nanos_now();
v___x_6728_ = lean_st_ref_take(v___y_6654_);
v_stats_6729_ = lean_ctor_get(v___x_6728_, 1);
v_rulePatternCache_6730_ = lean_ctor_get(v___x_6728_, 0);
v_isSharedCheck_6769_ = !lean_is_exclusive(v___x_6728_);
if (v_isSharedCheck_6769_ == 0)
{
v___x_6732_ = v___x_6728_;
v_isShared_6733_ = v_isSharedCheck_6769_;
goto v_resetjp_6731_;
}
else
{
lean_inc(v_stats_6729_);
lean_inc(v_rulePatternCache_6730_);
lean_dec(v___x_6728_);
v___x_6732_ = lean_box(0);
v_isShared_6733_ = v_isSharedCheck_6769_;
goto v_resetjp_6731_;
}
v_resetjp_6731_:
{
lean_object* v_total_6734_; lean_object* v_configParsing_6735_; lean_object* v_ruleSetConstruction_6736_; lean_object* v_search_6737_; lean_object* v_ruleSelection_6738_; lean_object* v_script_6739_; lean_object* v_forwardState_6740_; lean_object* v_scriptGenerated_6741_; lean_object* v_ruleStats_6742_; lean_object* v_goalStats_6743_; lean_object* v___x_6745_; uint8_t v_isShared_6746_; uint8_t v_isSharedCheck_6768_; 
v_total_6734_ = lean_ctor_get(v_stats_6729_, 0);
v_configParsing_6735_ = lean_ctor_get(v_stats_6729_, 1);
v_ruleSetConstruction_6736_ = lean_ctor_get(v_stats_6729_, 2);
v_search_6737_ = lean_ctor_get(v_stats_6729_, 3);
v_ruleSelection_6738_ = lean_ctor_get(v_stats_6729_, 4);
v_script_6739_ = lean_ctor_get(v_stats_6729_, 5);
v_forwardState_6740_ = lean_ctor_get(v_stats_6729_, 6);
v_scriptGenerated_6741_ = lean_ctor_get(v_stats_6729_, 7);
v_ruleStats_6742_ = lean_ctor_get(v_stats_6729_, 8);
v_goalStats_6743_ = lean_ctor_get(v_stats_6729_, 9);
v_isSharedCheck_6768_ = !lean_is_exclusive(v_stats_6729_);
if (v_isSharedCheck_6768_ == 0)
{
v___x_6745_ = v_stats_6729_;
v_isShared_6746_ = v_isSharedCheck_6768_;
goto v_resetjp_6744_;
}
else
{
lean_inc(v_goalStats_6743_);
lean_inc(v_ruleStats_6742_);
lean_inc(v_scriptGenerated_6741_);
lean_inc(v_forwardState_6740_);
lean_inc(v_script_6739_);
lean_inc(v_ruleSelection_6738_);
lean_inc(v_search_6737_);
lean_inc(v_ruleSetConstruction_6736_);
lean_inc(v_configParsing_6735_);
lean_inc(v_total_6734_);
lean_dec(v_stats_6729_);
v___x_6745_ = lean_box(0);
v_isShared_6746_ = v_isSharedCheck_6768_;
goto v_resetjp_6744_;
}
v_resetjp_6744_:
{
lean_object* v___x_6747_; lean_object* v___x_6748_; lean_object* v___x_6750_; 
v___x_6747_ = lean_nat_sub(v___x_6727_, v___x_6720_);
lean_dec(v___x_6720_);
lean_dec(v___x_6727_);
v___x_6748_ = lean_nat_add(v_script_6739_, v___x_6747_);
lean_dec(v___x_6747_);
lean_dec(v_script_6739_);
if (v_isShared_6746_ == 0)
{
lean_ctor_set(v___x_6745_, 5, v___x_6748_);
v___x_6750_ = v___x_6745_;
goto v_reusejp_6749_;
}
else
{
lean_object* v_reuseFailAlloc_6767_; 
v_reuseFailAlloc_6767_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_6767_, 0, v_total_6734_);
lean_ctor_set(v_reuseFailAlloc_6767_, 1, v_configParsing_6735_);
lean_ctor_set(v_reuseFailAlloc_6767_, 2, v_ruleSetConstruction_6736_);
lean_ctor_set(v_reuseFailAlloc_6767_, 3, v_search_6737_);
lean_ctor_set(v_reuseFailAlloc_6767_, 4, v_ruleSelection_6738_);
lean_ctor_set(v_reuseFailAlloc_6767_, 5, v___x_6748_);
lean_ctor_set(v_reuseFailAlloc_6767_, 6, v_forwardState_6740_);
lean_ctor_set(v_reuseFailAlloc_6767_, 7, v_scriptGenerated_6741_);
lean_ctor_set(v_reuseFailAlloc_6767_, 8, v_ruleStats_6742_);
lean_ctor_set(v_reuseFailAlloc_6767_, 9, v_goalStats_6743_);
v___x_6750_ = v_reuseFailAlloc_6767_;
goto v_reusejp_6749_;
}
v_reusejp_6749_:
{
lean_object* v___x_6752_; 
if (v_isShared_6733_ == 0)
{
lean_ctor_set(v___x_6732_, 1, v___x_6750_);
v___x_6752_ = v___x_6732_;
goto v_reusejp_6751_;
}
else
{
lean_object* v_reuseFailAlloc_6766_; 
v_reuseFailAlloc_6766_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6766_, 0, v_rulePatternCache_6730_);
lean_ctor_set(v_reuseFailAlloc_6766_, 1, v___x_6750_);
v___x_6752_ = v_reuseFailAlloc_6766_;
goto v_reusejp_6751_;
}
v_reusejp_6751_:
{
lean_object* v___x_6753_; uint8_t v___x_6754_; lean_object* v___x_6755_; lean_object* v___x_6756_; lean_object* v___x_6757_; lean_object* v___x_6758_; lean_object* v___x_6759_; lean_object* v___x_6760_; lean_object* v___x_6761_; lean_object* v___x_6762_; lean_object* v___x_6763_; lean_object* v___x_6764_; lean_object* v___x_6765_; 
v___x_6753_ = lean_st_ref_set(v___y_6654_, v___x_6752_);
v___x_6754_ = 0;
v___x_6755_ = l_Lean_SourceInfo_fromRef(v_ref_6661_, v___x_6754_);
v___x_6756_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4));
v___x_6757_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__6));
v___x_6758_ = lean_obj_once(&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7, &lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7_once, _init_lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7);
v___x_6759_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_6760_ = l_Lean_Syntax_SepArray_ofElems(v___x_6759_, v_a_6726_);
lean_dec(v_a_6726_);
v___x_6761_ = l_Array_append___redArg(v___x_6758_, v___x_6760_);
lean_dec_ref(v___x_6760_);
lean_inc_n(v___x_6755_, 2);
v___x_6762_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_6762_, 0, v___x_6755_);
lean_ctor_set(v___x_6762_, 1, v___x_6757_);
lean_ctor_set(v___x_6762_, 2, v___x_6761_);
v___x_6763_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9));
v___x_6764_ = l_Lean_Syntax_node1(v___x_6755_, v___x_6763_, v___x_6762_);
v___x_6765_ = l_Lean_Syntax_node1(v___x_6755_, v___x_6756_, v___x_6764_);
v___y_6663_ = v___y_6716_;
v___y_6664_ = v___y_6719_;
v___y_6665_ = v___y_6718_;
v_a_6666_ = v___x_6765_;
goto v___jp_6662_;
}
}
}
}
}
else
{
lean_object* v_a_6770_; lean_object* v___x_6772_; uint8_t v_isShared_6773_; uint8_t v_isSharedCheck_6777_; 
lean_dec(v___x_6720_);
lean_dec(v___y_6719_);
lean_dec(v_goal_6649_);
lean_dec_ref(v_a_6648_);
v_a_6770_ = lean_ctor_get(v___x_6725_, 0);
v_isSharedCheck_6777_ = !lean_is_exclusive(v___x_6725_);
if (v_isSharedCheck_6777_ == 0)
{
v___x_6772_ = v___x_6725_;
v_isShared_6773_ = v_isSharedCheck_6777_;
goto v_resetjp_6771_;
}
else
{
lean_inc(v_a_6770_);
lean_dec(v___x_6725_);
v___x_6772_ = lean_box(0);
v_isShared_6773_ = v_isSharedCheck_6777_;
goto v_resetjp_6771_;
}
v_resetjp_6771_:
{
lean_object* v___x_6775_; 
if (v_isShared_6773_ == 0)
{
v___x_6775_ = v___x_6772_;
goto v_reusejp_6774_;
}
else
{
lean_object* v_reuseFailAlloc_6776_; 
v_reuseFailAlloc_6776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6776_, 0, v_a_6770_);
v___x_6775_ = v_reuseFailAlloc_6776_;
goto v_reusejp_6774_;
}
v_reusejp_6774_:
{
return v___x_6775_;
}
}
}
}
else
{
lean_object* v_a_6778_; lean_object* v___x_6780_; uint8_t v_isShared_6781_; uint8_t v_isSharedCheck_6785_; 
lean_dec(v___x_6720_);
lean_dec(v___y_6719_);
lean_dec_ref(v_tacticState_6651_);
lean_dec(v_goal_6649_);
lean_dec_ref(v_a_6648_);
v_a_6778_ = lean_ctor_get(v___x_6723_, 0);
v_isSharedCheck_6785_ = !lean_is_exclusive(v___x_6723_);
if (v_isSharedCheck_6785_ == 0)
{
v___x_6780_ = v___x_6723_;
v_isShared_6781_ = v_isSharedCheck_6785_;
goto v_resetjp_6779_;
}
else
{
lean_inc(v_a_6778_);
lean_dec(v___x_6723_);
v___x_6780_ = lean_box(0);
v_isShared_6781_ = v_isSharedCheck_6785_;
goto v_resetjp_6779_;
}
v_resetjp_6779_:
{
lean_object* v___x_6783_; 
if (v_isShared_6781_ == 0)
{
v___x_6783_ = v___x_6780_;
goto v_reusejp_6782_;
}
else
{
lean_object* v_reuseFailAlloc_6784_; 
v_reuseFailAlloc_6784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6784_, 0, v_a_6778_);
v___x_6783_ = v_reuseFailAlloc_6784_;
goto v_reusejp_6782_;
}
v_reusejp_6782_:
{
return v___x_6783_;
}
}
}
}
v___jp_6786_:
{
if (v_a_6791_ == 0)
{
size_t v_sz_6792_; size_t v___x_6793_; lean_object* v___x_6794_; 
v_sz_6792_ = lean_array_size(v___y_6788_);
v___x_6793_ = ((size_t)0ULL);
v___x_6794_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg(v_sz_6792_, v___x_6793_, v___y_6788_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
if (lean_obj_tag(v___x_6794_) == 0)
{
lean_object* v_a_6795_; lean_object* v___x_6796_; 
v_a_6795_ = lean_ctor_get(v___x_6794_, 0);
lean_inc(v_a_6795_);
lean_dec_ref_known(v___x_6794_, 1);
v___x_6796_ = lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2(v_tacticState_6651_, v_a_6795_, v___y_6652_, v___y_6653_, v___y_6654_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
lean_dec(v_a_6795_);
if (lean_obj_tag(v___x_6796_) == 0)
{
lean_object* v_a_6797_; lean_object* v___x_6798_; lean_object* v___x_6799_; lean_object* v___x_6800_; lean_object* v___x_6801_; lean_object* v___x_6802_; lean_object* v___x_6803_; lean_object* v___x_6804_; lean_object* v___x_6805_; lean_object* v___x_6806_; lean_object* v___x_6807_; lean_object* v___x_6808_; 
v_a_6797_ = lean_ctor_get(v___x_6796_, 0);
lean_inc(v_a_6797_);
lean_dec_ref_known(v___x_6796_, 1);
v___x_6798_ = l_Lean_SourceInfo_fromRef(v_ref_6661_, v_a_6791_);
v___x_6799_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4));
v___x_6800_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9));
v___x_6801_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__6));
v___x_6802_ = lean_obj_once(&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7, &lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7_once, _init_lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7);
v___x_6803_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_6804_ = l_Lean_Syntax_SepArray_ofElems(v___x_6803_, v_a_6797_);
lean_dec(v_a_6797_);
v___x_6805_ = l_Array_append___redArg(v___x_6802_, v___x_6804_);
lean_dec_ref(v___x_6804_);
lean_inc_n(v___x_6798_, 2);
v___x_6806_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_6806_, 0, v___x_6798_);
lean_ctor_set(v___x_6806_, 1, v___x_6801_);
lean_ctor_set(v___x_6806_, 2, v___x_6805_);
v___x_6807_ = l_Lean_Syntax_node1(v___x_6798_, v___x_6800_, v___x_6806_);
v___x_6808_ = l_Lean_Syntax_node1(v___x_6798_, v___x_6799_, v___x_6807_);
v___y_6663_ = v___y_6787_;
v___y_6664_ = v___y_6790_;
v___y_6665_ = v___y_6789_;
v_a_6666_ = v___x_6808_;
goto v___jp_6662_;
}
else
{
lean_object* v_a_6809_; lean_object* v___x_6811_; uint8_t v_isShared_6812_; uint8_t v_isSharedCheck_6816_; 
lean_dec(v___y_6790_);
lean_dec(v_goal_6649_);
lean_dec_ref(v_a_6648_);
v_a_6809_ = lean_ctor_get(v___x_6796_, 0);
v_isSharedCheck_6816_ = !lean_is_exclusive(v___x_6796_);
if (v_isSharedCheck_6816_ == 0)
{
v___x_6811_ = v___x_6796_;
v_isShared_6812_ = v_isSharedCheck_6816_;
goto v_resetjp_6810_;
}
else
{
lean_inc(v_a_6809_);
lean_dec(v___x_6796_);
v___x_6811_ = lean_box(0);
v_isShared_6812_ = v_isSharedCheck_6816_;
goto v_resetjp_6810_;
}
v_resetjp_6810_:
{
lean_object* v___x_6814_; 
if (v_isShared_6812_ == 0)
{
v___x_6814_ = v___x_6811_;
goto v_reusejp_6813_;
}
else
{
lean_object* v_reuseFailAlloc_6815_; 
v_reuseFailAlloc_6815_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6815_, 0, v_a_6809_);
v___x_6814_ = v_reuseFailAlloc_6815_;
goto v_reusejp_6813_;
}
v_reusejp_6813_:
{
return v___x_6814_;
}
}
}
}
else
{
lean_object* v_a_6817_; lean_object* v___x_6819_; uint8_t v_isShared_6820_; uint8_t v_isSharedCheck_6824_; 
lean_dec(v___y_6790_);
lean_dec_ref(v_tacticState_6651_);
lean_dec(v_goal_6649_);
lean_dec_ref(v_a_6648_);
v_a_6817_ = lean_ctor_get(v___x_6794_, 0);
v_isSharedCheck_6824_ = !lean_is_exclusive(v___x_6794_);
if (v_isSharedCheck_6824_ == 0)
{
v___x_6819_ = v___x_6794_;
v_isShared_6820_ = v_isSharedCheck_6824_;
goto v_resetjp_6818_;
}
else
{
lean_inc(v_a_6817_);
lean_dec(v___x_6794_);
v___x_6819_ = lean_box(0);
v_isShared_6820_ = v_isSharedCheck_6824_;
goto v_resetjp_6818_;
}
v_resetjp_6818_:
{
lean_object* v___x_6822_; 
if (v_isShared_6820_ == 0)
{
v___x_6822_ = v___x_6819_;
goto v_reusejp_6821_;
}
else
{
lean_object* v_reuseFailAlloc_6823_; 
v_reuseFailAlloc_6823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6823_, 0, v_a_6817_);
v___x_6822_ = v_reuseFailAlloc_6823_;
goto v_reusejp_6821_;
}
v_reusejp_6821_:
{
return v___x_6822_;
}
}
}
}
else
{
v___y_6716_ = v___y_6787_;
v___y_6717_ = v___y_6788_;
v___y_6718_ = v___y_6789_;
v___y_6719_ = v___y_6790_;
goto v___jp_6715_;
}
}
v___jp_6825_:
{
lean_object* v_a_6831_; uint8_t v___x_6832_; 
v_a_6831_ = lean_ctor_get(v___y_6830_, 0);
lean_inc(v_a_6831_);
lean_dec_ref(v___y_6830_);
v___x_6832_ = lean_unbox(v_a_6831_);
lean_dec(v_a_6831_);
v___y_6787_ = v___y_6826_;
v___y_6788_ = v___y_6827_;
v___y_6789_ = v___y_6829_;
v___y_6790_ = v___y_6828_;
v_a_6791_ = v___x_6832_;
goto v___jp_6786_;
}
v___jp_6833_:
{
uint8_t v_generateScript_6835_; 
v_generateScript_6835_ = lean_ctor_get_uint8(v___y_6652_, sizeof(void*)*2);
if (v_generateScript_6835_ == 0)
{
lean_object* v___x_6836_; 
lean_dec_ref(v_tacticState_6651_);
lean_dec(v_goal_6649_);
lean_dec_ref(v_a_6648_);
v___x_6836_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6836_, 0, v_a_6834_);
return v___x_6836_;
}
else
{
lean_object* v_toOptions_6837_; lean_object* v___x_6838_; lean_object* v___x_6839_; uint8_t v___x_6840_; 
v_toOptions_6837_ = lean_ctor_get(v___y_6652_, 0);
v___x_6838_ = lean_st_ref_get(v___y_6653_);
v___x_6839_ = lp_aesop_Aesop_aesop_collectStats;
v___x_6840_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_6660_, v___x_6839_);
if (v___x_6840_ == 0)
{
lean_object* v___x_6841_; lean_object* v___x_6842_; lean_object* v_a_6843_; uint8_t v___x_6844_; 
v___x_6841_ = lp_aesop_Aesop_TraceOption_stats;
v___x_6842_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_6841_, v___y_6657_);
v_a_6843_ = lean_ctor_get(v___x_6842_, 0);
lean_inc(v_a_6843_);
v___x_6844_ = lean_unbox(v_a_6843_);
lean_dec(v_a_6843_);
if (v___x_6844_ == 0)
{
lean_object* v___x_6845_; lean_object* v___x_6846_; lean_object* v___x_6847_; uint8_t v___x_6848_; 
v___x_6845_ = lp_aesop_Aesop_aesop_stats_file;
v___x_6846_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_6660_, v___x_6845_);
v___x_6847_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_6848_ = lean_string_dec_eq(v___x_6846_, v___x_6847_);
lean_dec_ref(v___x_6846_);
if (v___x_6848_ == 0)
{
lean_dec_ref(v___x_6842_);
v___y_6716_ = v_toOptions_6837_;
v___y_6717_ = v___x_6838_;
v___y_6718_ = v_generateScript_6835_;
v___y_6719_ = v_a_6834_;
goto v___jp_6715_;
}
else
{
v___y_6826_ = v_toOptions_6837_;
v___y_6827_ = v___x_6838_;
v___y_6828_ = v_a_6834_;
v___y_6829_ = v_generateScript_6835_;
v___y_6830_ = v___x_6842_;
goto v___jp_6825_;
}
}
else
{
v___y_6826_ = v_toOptions_6837_;
v___y_6827_ = v___x_6838_;
v___y_6828_ = v_a_6834_;
v___y_6829_ = v_generateScript_6835_;
v___y_6830_ = v___x_6842_;
goto v___jp_6825_;
}
}
else
{
v___y_6787_ = v_toOptions_6837_;
v___y_6788_ = v___x_6838_;
v___y_6789_ = v_generateScript_6835_;
v___y_6790_ = v_a_6834_;
v_a_6791_ = v___x_6840_;
goto v___jp_6786_;
}
}
}
v___jp_6849_:
{
if (lean_obj_tag(v___y_6851_) == 0)
{
lean_object* v_a_6852_; lean_object* v___x_6853_; lean_object* v___x_6854_; lean_object* v_stats_6855_; lean_object* v_rulePatternCache_6856_; lean_object* v___x_6858_; uint8_t v_isShared_6859_; uint8_t v_isSharedCheck_6882_; 
v_a_6852_ = lean_ctor_get(v___y_6851_, 0);
lean_inc(v_a_6852_);
lean_dec_ref_known(v___y_6851_, 1);
v___x_6853_ = lean_io_mono_nanos_now();
v___x_6854_ = lean_st_ref_take(v___y_6654_);
v_stats_6855_ = lean_ctor_get(v___x_6854_, 1);
v_rulePatternCache_6856_ = lean_ctor_get(v___x_6854_, 0);
v_isSharedCheck_6882_ = !lean_is_exclusive(v___x_6854_);
if (v_isSharedCheck_6882_ == 0)
{
v___x_6858_ = v___x_6854_;
v_isShared_6859_ = v_isSharedCheck_6882_;
goto v_resetjp_6857_;
}
else
{
lean_inc(v_stats_6855_);
lean_inc(v_rulePatternCache_6856_);
lean_dec(v___x_6854_);
v___x_6858_ = lean_box(0);
v_isShared_6859_ = v_isSharedCheck_6882_;
goto v_resetjp_6857_;
}
v_resetjp_6857_:
{
lean_object* v_total_6860_; lean_object* v_configParsing_6861_; lean_object* v_ruleSetConstruction_6862_; lean_object* v_ruleSelection_6863_; lean_object* v_script_6864_; lean_object* v_forwardState_6865_; lean_object* v_scriptGenerated_6866_; lean_object* v_ruleStats_6867_; lean_object* v_goalStats_6868_; lean_object* v___x_6870_; uint8_t v_isShared_6871_; uint8_t v_isSharedCheck_6880_; 
v_total_6860_ = lean_ctor_get(v_stats_6855_, 0);
v_configParsing_6861_ = lean_ctor_get(v_stats_6855_, 1);
v_ruleSetConstruction_6862_ = lean_ctor_get(v_stats_6855_, 2);
v_ruleSelection_6863_ = lean_ctor_get(v_stats_6855_, 4);
v_script_6864_ = lean_ctor_get(v_stats_6855_, 5);
v_forwardState_6865_ = lean_ctor_get(v_stats_6855_, 6);
v_scriptGenerated_6866_ = lean_ctor_get(v_stats_6855_, 7);
v_ruleStats_6867_ = lean_ctor_get(v_stats_6855_, 8);
v_goalStats_6868_ = lean_ctor_get(v_stats_6855_, 9);
v_isSharedCheck_6880_ = !lean_is_exclusive(v_stats_6855_);
if (v_isSharedCheck_6880_ == 0)
{
lean_object* v_unused_6881_; 
v_unused_6881_ = lean_ctor_get(v_stats_6855_, 3);
lean_dec(v_unused_6881_);
v___x_6870_ = v_stats_6855_;
v_isShared_6871_ = v_isSharedCheck_6880_;
goto v_resetjp_6869_;
}
else
{
lean_inc(v_goalStats_6868_);
lean_inc(v_ruleStats_6867_);
lean_inc(v_scriptGenerated_6866_);
lean_inc(v_forwardState_6865_);
lean_inc(v_script_6864_);
lean_inc(v_ruleSelection_6863_);
lean_inc(v_ruleSetConstruction_6862_);
lean_inc(v_configParsing_6861_);
lean_inc(v_total_6860_);
lean_dec(v_stats_6855_);
v___x_6870_ = lean_box(0);
v_isShared_6871_ = v_isSharedCheck_6880_;
goto v_resetjp_6869_;
}
v_resetjp_6869_:
{
lean_object* v___x_6872_; lean_object* v___x_6874_; 
v___x_6872_ = lean_nat_sub(v___x_6853_, v___y_6850_);
lean_dec(v___y_6850_);
lean_dec(v___x_6853_);
if (v_isShared_6871_ == 0)
{
lean_ctor_set(v___x_6870_, 3, v___x_6872_);
v___x_6874_ = v___x_6870_;
goto v_reusejp_6873_;
}
else
{
lean_object* v_reuseFailAlloc_6879_; 
v_reuseFailAlloc_6879_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_6879_, 0, v_total_6860_);
lean_ctor_set(v_reuseFailAlloc_6879_, 1, v_configParsing_6861_);
lean_ctor_set(v_reuseFailAlloc_6879_, 2, v_ruleSetConstruction_6862_);
lean_ctor_set(v_reuseFailAlloc_6879_, 3, v___x_6872_);
lean_ctor_set(v_reuseFailAlloc_6879_, 4, v_ruleSelection_6863_);
lean_ctor_set(v_reuseFailAlloc_6879_, 5, v_script_6864_);
lean_ctor_set(v_reuseFailAlloc_6879_, 6, v_forwardState_6865_);
lean_ctor_set(v_reuseFailAlloc_6879_, 7, v_scriptGenerated_6866_);
lean_ctor_set(v_reuseFailAlloc_6879_, 8, v_ruleStats_6867_);
lean_ctor_set(v_reuseFailAlloc_6879_, 9, v_goalStats_6868_);
v___x_6874_ = v_reuseFailAlloc_6879_;
goto v_reusejp_6873_;
}
v_reusejp_6873_:
{
lean_object* v___x_6876_; 
if (v_isShared_6859_ == 0)
{
lean_ctor_set(v___x_6858_, 1, v___x_6874_);
v___x_6876_ = v___x_6858_;
goto v_reusejp_6875_;
}
else
{
lean_object* v_reuseFailAlloc_6878_; 
v_reuseFailAlloc_6878_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6878_, 0, v_rulePatternCache_6856_);
lean_ctor_set(v_reuseFailAlloc_6878_, 1, v___x_6874_);
v___x_6876_ = v_reuseFailAlloc_6878_;
goto v_reusejp_6875_;
}
v_reusejp_6875_:
{
lean_object* v___x_6877_; 
v___x_6877_ = lean_st_ref_set(v___y_6654_, v___x_6876_);
v_a_6834_ = v_a_6852_;
goto v___jp_6833_;
}
}
}
}
}
else
{
lean_dec(v___y_6850_);
lean_dec_ref(v_tacticState_6651_);
lean_dec(v_goal_6649_);
lean_dec_ref(v_a_6648_);
return v___y_6851_;
}
}
v___jp_6883_:
{
lean_object* v___x_6884_; lean_object* v___x_6885_; uint8_t v___x_6886_; 
v___x_6884_ = lean_io_mono_nanos_now();
v___x_6885_ = lp_aesop_Aesop_aesop_dev_statefulForward;
v___x_6886_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_6660_, v___x_6885_);
if (v___x_6886_ == 0)
{
lean_object* v___x_6887_; 
lean_inc(v_goal_6649_);
v___x_6887_ = lp_aesop_Aesop_saturateCore(v_rs_6650_, v_goal_6649_, v___y_6652_, v___y_6653_, v___y_6654_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
v___y_6850_ = v___x_6884_;
v___y_6851_ = v___x_6887_;
goto v___jp_6849_;
}
else
{
lean_object* v___x_6888_; 
lean_inc(v_goal_6649_);
v___x_6888_ = lp_aesop_Aesop_Stateful_saturateCore(v_rs_6650_, v_goal_6649_, v___y_6652_, v___y_6653_, v___y_6654_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
v___y_6850_ = v___x_6884_;
v___y_6851_ = v___x_6888_;
goto v___jp_6849_;
}
}
v___jp_6889_:
{
if (lean_obj_tag(v___y_6890_) == 0)
{
lean_object* v_a_6891_; 
v_a_6891_ = lean_ctor_get(v___y_6890_, 0);
lean_inc(v_a_6891_);
lean_dec_ref_known(v___y_6890_, 1);
v_a_6834_ = v_a_6891_;
goto v___jp_6833_;
}
else
{
lean_dec_ref(v_tacticState_6651_);
lean_dec(v_goal_6649_);
lean_dec_ref(v_a_6648_);
return v___y_6890_;
}
}
v___jp_6892_:
{
lean_object* v___x_6893_; uint8_t v___x_6894_; 
v___x_6893_ = lp_aesop_Aesop_aesop_dev_statefulForward;
v___x_6894_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_6660_, v___x_6893_);
if (v___x_6894_ == 0)
{
lean_object* v___x_6895_; 
lean_inc(v_goal_6649_);
v___x_6895_ = lp_aesop_Aesop_saturateCore(v_rs_6650_, v_goal_6649_, v___y_6652_, v___y_6653_, v___y_6654_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
v___y_6890_ = v___x_6895_;
goto v___jp_6889_;
}
else
{
lean_object* v___x_6896_; 
lean_inc(v_goal_6649_);
v___x_6896_ = lp_aesop_Aesop_Stateful_saturateCore(v_rs_6650_, v_goal_6649_, v___y_6652_, v___y_6653_, v___y_6654_, v___y_6655_, v___y_6656_, v___y_6657_, v___y_6658_);
v___y_6890_ = v___x_6896_;
goto v___jp_6889_;
}
}
v___jp_6897_:
{
lean_object* v_a_6899_; uint8_t v___x_6900_; 
v_a_6899_ = lean_ctor_get(v___y_6898_, 0);
lean_inc(v_a_6899_);
lean_dec_ref(v___y_6898_);
v___x_6900_ = lean_unbox(v_a_6899_);
lean_dec(v_a_6899_);
if (v___x_6900_ == 0)
{
goto v___jp_6892_;
}
else
{
goto v___jp_6883_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain_x27___lam__0___boxed(lean_object* v_a_6911_, lean_object* v_goal_6912_, lean_object* v_rs_6913_, lean_object* v_tacticState_6914_, lean_object* v___y_6915_, lean_object* v___y_6916_, lean_object* v___y_6917_, lean_object* v___y_6918_, lean_object* v___y_6919_, lean_object* v___y_6920_, lean_object* v___y_6921_, lean_object* v___y_6922_){
_start:
{
lean_object* v_res_6923_; 
v_res_6923_ = lp_aesop_Aesop_saturateMain_x27___lam__0(v_a_6911_, v_goal_6912_, v_rs_6913_, v_tacticState_6914_, v___y_6915_, v___y_6916_, v___y_6917_, v___y_6918_, v___y_6919_, v___y_6920_, v___y_6921_);
lean_dec(v___y_6921_);
lean_dec_ref(v___y_6920_);
lean_dec(v___y_6919_);
lean_dec_ref(v___y_6918_);
lean_dec(v___y_6917_);
lean_dec(v___y_6916_);
lean_dec_ref(v___y_6915_);
return v_res_6923_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain_x27(lean_object* v_rs_6924_, lean_object* v_goal_6925_, lean_object* v_a_6926_, lean_object* v_a_6927_, lean_object* v_a_6928_, lean_object* v_a_6929_, lean_object* v_a_6930_, lean_object* v_a_6931_, lean_object* v_a_6932_){
_start:
{
lean_object* v___y_6935_; lean_object* v___y_6936_; lean_object* v___y_6937_; uint8_t v___y_6938_; lean_object* v___y_6939_; uint8_t v___y_6940_; lean_object* v___y_6941_; lean_object* v___y_6942_; lean_object* v___y_6943_; lean_object* v___y_6944_; lean_object* v___y_6945_; lean_object* v___y_6946_; lean_object* v_a_6947_; uint8_t v___y_6997_; uint8_t v___y_6998_; lean_object* v___y_6999_; lean_object* v___y_7000_; lean_object* v___y_7001_; lean_object* v___y_7002_; lean_object* v___y_7003_; lean_object* v___y_7004_; lean_object* v___y_7005_; lean_object* v___y_7006_; lean_object* v___y_7007_; lean_object* v___y_7008_; lean_object* v___y_7009_; lean_object* v___y_7010_; uint8_t v___y_7078_; uint8_t v___y_7079_; lean_object* v___y_7080_; lean_object* v___y_7081_; lean_object* v___y_7082_; lean_object* v___y_7083_; lean_object* v___y_7084_; lean_object* v___y_7085_; lean_object* v___y_7086_; lean_object* v___y_7087_; lean_object* v___y_7088_; lean_object* v___y_7089_; lean_object* v___y_7090_; lean_object* v___y_7091_; uint8_t v_a_7092_; uint8_t v___y_7128_; uint8_t v___y_7129_; lean_object* v___y_7130_; lean_object* v___y_7131_; lean_object* v___y_7132_; lean_object* v___y_7133_; lean_object* v___y_7134_; lean_object* v___y_7135_; lean_object* v___y_7136_; lean_object* v___y_7137_; lean_object* v___y_7138_; lean_object* v___y_7139_; lean_object* v___y_7140_; lean_object* v___y_7141_; lean_object* v___y_7142_; lean_object* v___y_7146_; lean_object* v___y_7147_; lean_object* v___y_7148_; uint8_t v___y_7149_; lean_object* v___y_7150_; lean_object* v___y_7151_; lean_object* v___y_7152_; lean_object* v___y_7153_; lean_object* v___y_7154_; lean_object* v___y_7155_; lean_object* v___y_7156_; lean_object* v_a_7157_; lean_object* v___y_7173_; lean_object* v___y_7174_; lean_object* v___y_7175_; lean_object* v___y_7176_; uint8_t v___y_7177_; lean_object* v___y_7178_; lean_object* v___y_7179_; lean_object* v___y_7180_; lean_object* v___y_7181_; lean_object* v___y_7182_; lean_object* v___y_7183_; lean_object* v___y_7184_; lean_object* v___y_7224_; lean_object* v___y_7225_; lean_object* v___y_7226_; uint8_t v___y_7227_; lean_object* v___y_7228_; lean_object* v___y_7229_; lean_object* v___y_7230_; lean_object* v___y_7231_; lean_object* v___y_7232_; lean_object* v___y_7233_; lean_object* v___y_7241_; lean_object* v___y_7242_; lean_object* v___y_7243_; lean_object* v___y_7244_; uint8_t v___y_7245_; lean_object* v___y_7246_; lean_object* v___y_7247_; lean_object* v___y_7248_; lean_object* v___y_7249_; lean_object* v___y_7250_; lean_object* v___y_7251_; lean_object* v___y_7254_; lean_object* v___y_7255_; lean_object* v___y_7256_; uint8_t v___y_7257_; lean_object* v___y_7258_; lean_object* v___y_7259_; lean_object* v___y_7260_; lean_object* v___y_7261_; lean_object* v___y_7262_; lean_object* v___y_7263_; uint8_t v_a_7264_; lean_object* v___y_7271_; lean_object* v___y_7272_; lean_object* v___y_7273_; lean_object* v___y_7274_; uint8_t v___y_7275_; lean_object* v___y_7276_; lean_object* v___y_7277_; lean_object* v___y_7278_; lean_object* v___y_7279_; lean_object* v___y_7280_; lean_object* v___y_7281_; lean_object* v___y_7285_; uint8_t v___y_7286_; lean_object* v_tacticState_7287_; lean_object* v___y_7288_; lean_object* v___y_7289_; lean_object* v___y_7290_; lean_object* v___y_7291_; lean_object* v___y_7292_; lean_object* v___y_7293_; lean_object* v_options_7294_; lean_object* v___y_7295_; lean_object* v___y_7307_; lean_object* v___y_7308_; lean_object* v___y_7348_; lean_object* v___y_7349_; lean_object* v___y_7394_; lean_object* v___y_7395_; lean_object* v___y_7408_; lean_object* v___y_7409_; lean_object* v___y_7410_; lean_object* v_options_7413_; lean_object* v___y_7415_; uint8_t v___y_7416_; lean_object* v___y_7460_; uint8_t v___y_7461_; uint8_t v_a_7462_; lean_object* v___y_7474_; uint8_t v___y_7475_; lean_object* v___y_7476_; uint8_t v_a_7480_; lean_object* v___y_7531_; lean_object* v___x_7535_; uint8_t v___x_7536_; 
v_options_7413_ = lean_ctor_get(v_a_6931_, 2);
v___x_7535_ = lp_aesop_Aesop_aesop_collectStats;
v___x_7536_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_7413_, v___x_7535_);
if (v___x_7536_ == 0)
{
lean_object* v___x_7537_; lean_object* v___x_7538_; lean_object* v_a_7539_; uint8_t v___x_7540_; 
v___x_7537_ = lp_aesop_Aesop_TraceOption_stats;
v___x_7538_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_7537_, v_a_6931_);
v_a_7539_ = lean_ctor_get(v___x_7538_, 0);
lean_inc(v_a_7539_);
v___x_7540_ = lean_unbox(v_a_7539_);
if (v___x_7540_ == 0)
{
lean_object* v___x_7541_; lean_object* v___x_7542_; lean_object* v___x_7543_; uint8_t v___x_7544_; 
lean_dec_ref(v___x_7538_);
v___x_7541_ = lp_aesop_Aesop_aesop_stats_file;
v___x_7542_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_7413_, v___x_7541_);
v___x_7543_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_7544_ = lean_string_dec_eq(v___x_7542_, v___x_7543_);
lean_dec_ref(v___x_7542_);
if (v___x_7544_ == 0)
{
lean_dec(v_a_7539_);
goto v___jp_7504_;
}
else
{
uint8_t v___x_7545_; 
v___x_7545_ = lean_unbox(v_a_7539_);
lean_dec(v_a_7539_);
v_a_7480_ = v___x_7545_;
goto v___jp_7479_;
}
}
else
{
lean_dec(v_a_7539_);
v___y_7531_ = v___x_7538_;
goto v___jp_7530_;
}
}
else
{
goto v___jp_7504_;
}
v___jp_6934_:
{
uint8_t v___x_6948_; lean_object* v___x_6949_; lean_object* v___x_6950_; 
v___x_6948_ = 0;
v___x_6949_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v___x_6949_, 0, v___x_6948_);
lean_ctor_set_uint8(v___x_6949_, 1, v___y_6938_);
lean_ctor_set_uint8(v___x_6949_, 2, v___y_6940_);
v___x_6950_ = lp_aesop_Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0(v___x_6949_, v___y_6941_, v___y_6939_, v___y_6943_, v___y_6945_, v___y_6942_, v___y_6936_, v___y_6946_);
if (lean_obj_tag(v___x_6950_) == 0)
{
lean_object* v___x_6951_; 
lean_dec_ref_known(v___x_6950_, 1);
lean_inc(v_a_6947_);
v___x_6951_ = lp_aesop_Aesop_checkRenderedScriptIfEnabled(v_a_6947_, v___y_6937_, v_goal_6925_, v___y_6940_, v___y_6945_, v___y_6942_, v___y_6936_, v___y_6946_);
if (lean_obj_tag(v___x_6951_) == 0)
{
lean_object* v___x_6953_; uint8_t v_isShared_6954_; uint8_t v_isSharedCheck_6978_; 
v_isSharedCheck_6978_ = !lean_is_exclusive(v___x_6951_);
if (v_isSharedCheck_6978_ == 0)
{
lean_object* v_unused_6979_; 
v_unused_6979_ = lean_ctor_get(v___x_6951_, 0);
lean_dec(v_unused_6979_);
v___x_6953_ = v___x_6951_;
v_isShared_6954_ = v_isSharedCheck_6978_;
goto v_resetjp_6952_;
}
else
{
lean_dec(v___x_6951_);
v___x_6953_ = lean_box(0);
v_isShared_6954_ = v_isSharedCheck_6978_;
goto v_resetjp_6952_;
}
v_resetjp_6952_:
{
uint8_t v_traceScript_6955_; 
v_traceScript_6955_ = lean_ctor_get_uint8(v___y_6935_, sizeof(void*)*6 + 6);
if (v_traceScript_6955_ == 0)
{
lean_object* v___x_6957_; 
lean_dec(v_a_6947_);
if (v_isShared_6954_ == 0)
{
lean_ctor_set(v___x_6953_, 0, v___y_6944_);
v___x_6957_ = v___x_6953_;
goto v_reusejp_6956_;
}
else
{
lean_object* v_reuseFailAlloc_6958_; 
v_reuseFailAlloc_6958_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6958_, 0, v___y_6944_);
v___x_6957_ = v_reuseFailAlloc_6958_;
goto v_reusejp_6956_;
}
v_reusejp_6956_:
{
return v___x_6957_;
}
}
else
{
lean_object* v_ref_6959_; lean_object* v___x_6960_; lean_object* v___x_6961_; 
lean_del_object(v___x_6953_);
v_ref_6959_ = lean_ctor_get(v___y_6936_, 5);
v___x_6960_ = lean_box(0);
lean_inc(v_ref_6959_);
v___x_6961_ = lp_aesop_Aesop_addTryThisTacticSeqSuggestion___redArg(v_ref_6959_, v_a_6947_, v___x_6960_, v___y_6936_, v___y_6946_);
if (lean_obj_tag(v___x_6961_) == 0)
{
lean_object* v___x_6963_; uint8_t v_isShared_6964_; uint8_t v_isSharedCheck_6968_; 
v_isSharedCheck_6968_ = !lean_is_exclusive(v___x_6961_);
if (v_isSharedCheck_6968_ == 0)
{
lean_object* v_unused_6969_; 
v_unused_6969_ = lean_ctor_get(v___x_6961_, 0);
lean_dec(v_unused_6969_);
v___x_6963_ = v___x_6961_;
v_isShared_6964_ = v_isSharedCheck_6968_;
goto v_resetjp_6962_;
}
else
{
lean_dec(v___x_6961_);
v___x_6963_ = lean_box(0);
v_isShared_6964_ = v_isSharedCheck_6968_;
goto v_resetjp_6962_;
}
v_resetjp_6962_:
{
lean_object* v___x_6966_; 
if (v_isShared_6964_ == 0)
{
lean_ctor_set(v___x_6963_, 0, v___y_6944_);
v___x_6966_ = v___x_6963_;
goto v_reusejp_6965_;
}
else
{
lean_object* v_reuseFailAlloc_6967_; 
v_reuseFailAlloc_6967_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6967_, 0, v___y_6944_);
v___x_6966_ = v_reuseFailAlloc_6967_;
goto v_reusejp_6965_;
}
v_reusejp_6965_:
{
return v___x_6966_;
}
}
}
else
{
lean_object* v_a_6970_; lean_object* v___x_6972_; uint8_t v_isShared_6973_; uint8_t v_isSharedCheck_6977_; 
lean_dec(v___y_6944_);
v_a_6970_ = lean_ctor_get(v___x_6961_, 0);
v_isSharedCheck_6977_ = !lean_is_exclusive(v___x_6961_);
if (v_isSharedCheck_6977_ == 0)
{
v___x_6972_ = v___x_6961_;
v_isShared_6973_ = v_isSharedCheck_6977_;
goto v_resetjp_6971_;
}
else
{
lean_inc(v_a_6970_);
lean_dec(v___x_6961_);
v___x_6972_ = lean_box(0);
v_isShared_6973_ = v_isSharedCheck_6977_;
goto v_resetjp_6971_;
}
v_resetjp_6971_:
{
lean_object* v___x_6975_; 
if (v_isShared_6973_ == 0)
{
v___x_6975_ = v___x_6972_;
goto v_reusejp_6974_;
}
else
{
lean_object* v_reuseFailAlloc_6976_; 
v_reuseFailAlloc_6976_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6976_, 0, v_a_6970_);
v___x_6975_ = v_reuseFailAlloc_6976_;
goto v_reusejp_6974_;
}
v_reusejp_6974_:
{
return v___x_6975_;
}
}
}
}
}
}
else
{
lean_object* v_a_6980_; lean_object* v___x_6982_; uint8_t v_isShared_6983_; uint8_t v_isSharedCheck_6987_; 
lean_dec(v_a_6947_);
lean_dec(v___y_6944_);
v_a_6980_ = lean_ctor_get(v___x_6951_, 0);
v_isSharedCheck_6987_ = !lean_is_exclusive(v___x_6951_);
if (v_isSharedCheck_6987_ == 0)
{
v___x_6982_ = v___x_6951_;
v_isShared_6983_ = v_isSharedCheck_6987_;
goto v_resetjp_6981_;
}
else
{
lean_inc(v_a_6980_);
lean_dec(v___x_6951_);
v___x_6982_ = lean_box(0);
v_isShared_6983_ = v_isSharedCheck_6987_;
goto v_resetjp_6981_;
}
v_resetjp_6981_:
{
lean_object* v___x_6985_; 
if (v_isShared_6983_ == 0)
{
v___x_6985_ = v___x_6982_;
goto v_reusejp_6984_;
}
else
{
lean_object* v_reuseFailAlloc_6986_; 
v_reuseFailAlloc_6986_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6986_, 0, v_a_6980_);
v___x_6985_ = v_reuseFailAlloc_6986_;
goto v_reusejp_6984_;
}
v_reusejp_6984_:
{
return v___x_6985_;
}
}
}
}
else
{
lean_object* v_a_6988_; lean_object* v___x_6990_; uint8_t v_isShared_6991_; uint8_t v_isSharedCheck_6995_; 
lean_dec(v_a_6947_);
lean_dec(v___y_6944_);
lean_dec_ref(v___y_6937_);
lean_dec(v_goal_6925_);
v_a_6988_ = lean_ctor_get(v___x_6950_, 0);
v_isSharedCheck_6995_ = !lean_is_exclusive(v___x_6950_);
if (v_isSharedCheck_6995_ == 0)
{
v___x_6990_ = v___x_6950_;
v_isShared_6991_ = v_isSharedCheck_6995_;
goto v_resetjp_6989_;
}
else
{
lean_inc(v_a_6988_);
lean_dec(v___x_6950_);
v___x_6990_ = lean_box(0);
v_isShared_6991_ = v_isSharedCheck_6995_;
goto v_resetjp_6989_;
}
v_resetjp_6989_:
{
lean_object* v___x_6993_; 
if (v_isShared_6991_ == 0)
{
v___x_6993_ = v___x_6990_;
goto v_reusejp_6992_;
}
else
{
lean_object* v_reuseFailAlloc_6994_; 
v_reuseFailAlloc_6994_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6994_, 0, v_a_6988_);
v___x_6993_ = v_reuseFailAlloc_6994_;
goto v_reusejp_6992_;
}
v_reusejp_6992_:
{
return v___x_6993_;
}
}
}
}
v___jp_6996_:
{
lean_object* v___x_7011_; size_t v_sz_7012_; size_t v___x_7013_; lean_object* v___x_7014_; 
v___x_7011_ = lean_io_mono_nanos_now();
v_sz_7012_ = lean_array_size(v___y_7005_);
v___x_7013_ = ((size_t)0ULL);
v___x_7014_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg(v_sz_7012_, v___x_7013_, v___y_7005_, v___y_7009_, v___y_6999_, v___y_7003_, v___y_7010_);
if (lean_obj_tag(v___x_7014_) == 0)
{
lean_object* v_a_7015_; lean_object* v___x_7016_; 
v_a_7015_ = lean_ctor_get(v___x_7014_, 0);
lean_inc(v_a_7015_);
lean_dec_ref_known(v___x_7014_, 1);
v___x_7016_ = lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2(v___y_7001_, v_a_7015_, v___y_7007_, v___y_7006_, v___y_7008_, v___y_7009_, v___y_6999_, v___y_7003_, v___y_7010_);
lean_dec(v_a_7015_);
if (lean_obj_tag(v___x_7016_) == 0)
{
lean_object* v_a_7017_; lean_object* v_ref_7018_; lean_object* v___x_7019_; lean_object* v___x_7020_; lean_object* v_stats_7021_; lean_object* v_rulePatternCache_7022_; lean_object* v___x_7024_; uint8_t v_isShared_7025_; uint8_t v_isSharedCheck_7060_; 
v_a_7017_ = lean_ctor_get(v___x_7016_, 0);
lean_inc(v_a_7017_);
lean_dec_ref_known(v___x_7016_, 1);
v_ref_7018_ = lean_ctor_get(v___y_7003_, 5);
v___x_7019_ = lean_io_mono_nanos_now();
v___x_7020_ = lean_st_ref_take(v___y_7008_);
v_stats_7021_ = lean_ctor_get(v___x_7020_, 1);
v_rulePatternCache_7022_ = lean_ctor_get(v___x_7020_, 0);
v_isSharedCheck_7060_ = !lean_is_exclusive(v___x_7020_);
if (v_isSharedCheck_7060_ == 0)
{
v___x_7024_ = v___x_7020_;
v_isShared_7025_ = v_isSharedCheck_7060_;
goto v_resetjp_7023_;
}
else
{
lean_inc(v_stats_7021_);
lean_inc(v_rulePatternCache_7022_);
lean_dec(v___x_7020_);
v___x_7024_ = lean_box(0);
v_isShared_7025_ = v_isSharedCheck_7060_;
goto v_resetjp_7023_;
}
v_resetjp_7023_:
{
lean_object* v_total_7026_; lean_object* v_configParsing_7027_; lean_object* v_ruleSetConstruction_7028_; lean_object* v_search_7029_; lean_object* v_ruleSelection_7030_; lean_object* v_script_7031_; lean_object* v_forwardState_7032_; lean_object* v_scriptGenerated_7033_; lean_object* v_ruleStats_7034_; lean_object* v_goalStats_7035_; lean_object* v___x_7037_; uint8_t v_isShared_7038_; uint8_t v_isSharedCheck_7059_; 
v_total_7026_ = lean_ctor_get(v_stats_7021_, 0);
v_configParsing_7027_ = lean_ctor_get(v_stats_7021_, 1);
v_ruleSetConstruction_7028_ = lean_ctor_get(v_stats_7021_, 2);
v_search_7029_ = lean_ctor_get(v_stats_7021_, 3);
v_ruleSelection_7030_ = lean_ctor_get(v_stats_7021_, 4);
v_script_7031_ = lean_ctor_get(v_stats_7021_, 5);
v_forwardState_7032_ = lean_ctor_get(v_stats_7021_, 6);
v_scriptGenerated_7033_ = lean_ctor_get(v_stats_7021_, 7);
v_ruleStats_7034_ = lean_ctor_get(v_stats_7021_, 8);
v_goalStats_7035_ = lean_ctor_get(v_stats_7021_, 9);
v_isSharedCheck_7059_ = !lean_is_exclusive(v_stats_7021_);
if (v_isSharedCheck_7059_ == 0)
{
v___x_7037_ = v_stats_7021_;
v_isShared_7038_ = v_isSharedCheck_7059_;
goto v_resetjp_7036_;
}
else
{
lean_inc(v_goalStats_7035_);
lean_inc(v_ruleStats_7034_);
lean_inc(v_scriptGenerated_7033_);
lean_inc(v_forwardState_7032_);
lean_inc(v_script_7031_);
lean_inc(v_ruleSelection_7030_);
lean_inc(v_search_7029_);
lean_inc(v_ruleSetConstruction_7028_);
lean_inc(v_configParsing_7027_);
lean_inc(v_total_7026_);
lean_dec(v_stats_7021_);
v___x_7037_ = lean_box(0);
v_isShared_7038_ = v_isSharedCheck_7059_;
goto v_resetjp_7036_;
}
v_resetjp_7036_:
{
lean_object* v___x_7039_; lean_object* v___x_7040_; lean_object* v___x_7042_; 
v___x_7039_ = lean_nat_sub(v___x_7019_, v___x_7011_);
lean_dec(v___x_7011_);
lean_dec(v___x_7019_);
v___x_7040_ = lean_nat_add(v_script_7031_, v___x_7039_);
lean_dec(v___x_7039_);
lean_dec(v_script_7031_);
if (v_isShared_7038_ == 0)
{
lean_ctor_set(v___x_7037_, 5, v___x_7040_);
v___x_7042_ = v___x_7037_;
goto v_reusejp_7041_;
}
else
{
lean_object* v_reuseFailAlloc_7058_; 
v_reuseFailAlloc_7058_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_7058_, 0, v_total_7026_);
lean_ctor_set(v_reuseFailAlloc_7058_, 1, v_configParsing_7027_);
lean_ctor_set(v_reuseFailAlloc_7058_, 2, v_ruleSetConstruction_7028_);
lean_ctor_set(v_reuseFailAlloc_7058_, 3, v_search_7029_);
lean_ctor_set(v_reuseFailAlloc_7058_, 4, v_ruleSelection_7030_);
lean_ctor_set(v_reuseFailAlloc_7058_, 5, v___x_7040_);
lean_ctor_set(v_reuseFailAlloc_7058_, 6, v_forwardState_7032_);
lean_ctor_set(v_reuseFailAlloc_7058_, 7, v_scriptGenerated_7033_);
lean_ctor_set(v_reuseFailAlloc_7058_, 8, v_ruleStats_7034_);
lean_ctor_set(v_reuseFailAlloc_7058_, 9, v_goalStats_7035_);
v___x_7042_ = v_reuseFailAlloc_7058_;
goto v_reusejp_7041_;
}
v_reusejp_7041_:
{
lean_object* v___x_7044_; 
if (v_isShared_7025_ == 0)
{
lean_ctor_set(v___x_7024_, 1, v___x_7042_);
v___x_7044_ = v___x_7024_;
goto v_reusejp_7043_;
}
else
{
lean_object* v_reuseFailAlloc_7057_; 
v_reuseFailAlloc_7057_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_7057_, 0, v_rulePatternCache_7022_);
lean_ctor_set(v_reuseFailAlloc_7057_, 1, v___x_7042_);
v___x_7044_ = v_reuseFailAlloc_7057_;
goto v_reusejp_7043_;
}
v_reusejp_7043_:
{
lean_object* v___x_7045_; lean_object* v___x_7046_; lean_object* v___x_7047_; lean_object* v___x_7048_; lean_object* v___x_7049_; lean_object* v___x_7050_; lean_object* v___x_7051_; lean_object* v___x_7052_; lean_object* v___x_7053_; lean_object* v___x_7054_; lean_object* v___x_7055_; lean_object* v___x_7056_; 
v___x_7045_ = lean_st_ref_set(v___y_7008_, v___x_7044_);
v___x_7046_ = l_Lean_SourceInfo_fromRef(v_ref_7018_, v___y_6998_);
v___x_7047_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4));
v___x_7048_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__6));
v___x_7049_ = lean_obj_once(&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7, &lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7_once, _init_lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7);
v___x_7050_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_7051_ = l_Lean_Syntax_SepArray_ofElems(v___x_7050_, v_a_7017_);
lean_dec(v_a_7017_);
v___x_7052_ = l_Array_append___redArg(v___x_7049_, v___x_7051_);
lean_dec_ref(v___x_7051_);
lean_inc_n(v___x_7046_, 2);
v___x_7053_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_7053_, 0, v___x_7046_);
lean_ctor_set(v___x_7053_, 1, v___x_7048_);
lean_ctor_set(v___x_7053_, 2, v___x_7052_);
v___x_7054_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9));
v___x_7055_ = l_Lean_Syntax_node1(v___x_7046_, v___x_7054_, v___x_7053_);
v___x_7056_ = l_Lean_Syntax_node1(v___x_7046_, v___x_7047_, v___x_7055_);
v___y_6935_ = v___y_7002_;
v___y_6936_ = v___y_7003_;
v___y_6937_ = v___y_7004_;
v___y_6938_ = v___y_6997_;
v___y_6939_ = v___y_7006_;
v___y_6940_ = v___y_6998_;
v___y_6941_ = v___y_7007_;
v___y_6942_ = v___y_6999_;
v___y_6943_ = v___y_7008_;
v___y_6944_ = v___y_7000_;
v___y_6945_ = v___y_7009_;
v___y_6946_ = v___y_7010_;
v_a_6947_ = v___x_7056_;
goto v___jp_6934_;
}
}
}
}
}
else
{
lean_object* v_a_7061_; lean_object* v___x_7063_; uint8_t v_isShared_7064_; uint8_t v_isSharedCheck_7068_; 
lean_dec(v___x_7011_);
lean_dec_ref(v___y_7004_);
lean_dec(v___y_7000_);
lean_dec(v_goal_6925_);
v_a_7061_ = lean_ctor_get(v___x_7016_, 0);
v_isSharedCheck_7068_ = !lean_is_exclusive(v___x_7016_);
if (v_isSharedCheck_7068_ == 0)
{
v___x_7063_ = v___x_7016_;
v_isShared_7064_ = v_isSharedCheck_7068_;
goto v_resetjp_7062_;
}
else
{
lean_inc(v_a_7061_);
lean_dec(v___x_7016_);
v___x_7063_ = lean_box(0);
v_isShared_7064_ = v_isSharedCheck_7068_;
goto v_resetjp_7062_;
}
v_resetjp_7062_:
{
lean_object* v___x_7066_; 
if (v_isShared_7064_ == 0)
{
v___x_7066_ = v___x_7063_;
goto v_reusejp_7065_;
}
else
{
lean_object* v_reuseFailAlloc_7067_; 
v_reuseFailAlloc_7067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7067_, 0, v_a_7061_);
v___x_7066_ = v_reuseFailAlloc_7067_;
goto v_reusejp_7065_;
}
v_reusejp_7065_:
{
return v___x_7066_;
}
}
}
}
else
{
lean_object* v_a_7069_; lean_object* v___x_7071_; uint8_t v_isShared_7072_; uint8_t v_isSharedCheck_7076_; 
lean_dec(v___x_7011_);
lean_dec_ref(v___y_7004_);
lean_dec_ref(v___y_7001_);
lean_dec(v___y_7000_);
lean_dec(v_goal_6925_);
v_a_7069_ = lean_ctor_get(v___x_7014_, 0);
v_isSharedCheck_7076_ = !lean_is_exclusive(v___x_7014_);
if (v_isSharedCheck_7076_ == 0)
{
v___x_7071_ = v___x_7014_;
v_isShared_7072_ = v_isSharedCheck_7076_;
goto v_resetjp_7070_;
}
else
{
lean_inc(v_a_7069_);
lean_dec(v___x_7014_);
v___x_7071_ = lean_box(0);
v_isShared_7072_ = v_isSharedCheck_7076_;
goto v_resetjp_7070_;
}
v_resetjp_7070_:
{
lean_object* v___x_7074_; 
if (v_isShared_7072_ == 0)
{
v___x_7074_ = v___x_7071_;
goto v_reusejp_7073_;
}
else
{
lean_object* v_reuseFailAlloc_7075_; 
v_reuseFailAlloc_7075_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7075_, 0, v_a_7069_);
v___x_7074_ = v_reuseFailAlloc_7075_;
goto v_reusejp_7073_;
}
v_reusejp_7073_:
{
return v___x_7074_;
}
}
}
}
v___jp_7077_:
{
if (v_a_7092_ == 0)
{
size_t v_sz_7093_; size_t v___x_7094_; lean_object* v___x_7095_; 
v_sz_7093_ = lean_array_size(v___y_7086_);
v___x_7094_ = ((size_t)0ULL);
v___x_7095_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg(v_sz_7093_, v___x_7094_, v___y_7086_, v___y_7090_, v___y_7080_, v___y_7084_, v___y_7091_);
if (lean_obj_tag(v___x_7095_) == 0)
{
lean_object* v_a_7096_; lean_object* v___x_7097_; 
v_a_7096_ = lean_ctor_get(v___x_7095_, 0);
lean_inc(v_a_7096_);
lean_dec_ref_known(v___x_7095_, 1);
v___x_7097_ = lp_aesop_Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2(v___y_7082_, v_a_7096_, v___y_7088_, v___y_7087_, v___y_7089_, v___y_7090_, v___y_7080_, v___y_7084_, v___y_7091_);
lean_dec(v_a_7096_);
if (lean_obj_tag(v___x_7097_) == 0)
{
lean_object* v_a_7098_; lean_object* v_ref_7099_; lean_object* v___x_7100_; lean_object* v___x_7101_; lean_object* v___x_7102_; lean_object* v___x_7103_; lean_object* v___x_7104_; lean_object* v___x_7105_; lean_object* v___x_7106_; lean_object* v___x_7107_; lean_object* v___x_7108_; lean_object* v___x_7109_; lean_object* v___x_7110_; 
v_a_7098_ = lean_ctor_get(v___x_7097_, 0);
lean_inc(v_a_7098_);
lean_dec_ref_known(v___x_7097_, 1);
v_ref_7099_ = lean_ctor_get(v___y_7084_, 5);
v___x_7100_ = l_Lean_SourceInfo_fromRef(v_ref_7099_, v_a_7092_);
v___x_7101_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__4));
v___x_7102_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__9));
v___x_7103_ = ((lean_object*)(lp_aesop_Aesop_saturateMain_x27___lam__0___closed__6));
v___x_7104_ = lean_obj_once(&lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7, &lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7_once, _init_lp_aesop_Aesop_saturateMain_x27___lam__0___closed__7);
v___x_7105_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_7106_ = l_Lean_Syntax_SepArray_ofElems(v___x_7105_, v_a_7098_);
lean_dec(v_a_7098_);
v___x_7107_ = l_Array_append___redArg(v___x_7104_, v___x_7106_);
lean_dec_ref(v___x_7106_);
lean_inc_n(v___x_7100_, 2);
v___x_7108_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_7108_, 0, v___x_7100_);
lean_ctor_set(v___x_7108_, 1, v___x_7103_);
lean_ctor_set(v___x_7108_, 2, v___x_7107_);
v___x_7109_ = l_Lean_Syntax_node1(v___x_7100_, v___x_7102_, v___x_7108_);
v___x_7110_ = l_Lean_Syntax_node1(v___x_7100_, v___x_7101_, v___x_7109_);
v___y_6935_ = v___y_7083_;
v___y_6936_ = v___y_7084_;
v___y_6937_ = v___y_7085_;
v___y_6938_ = v___y_7078_;
v___y_6939_ = v___y_7087_;
v___y_6940_ = v___y_7079_;
v___y_6941_ = v___y_7088_;
v___y_6942_ = v___y_7080_;
v___y_6943_ = v___y_7089_;
v___y_6944_ = v___y_7081_;
v___y_6945_ = v___y_7090_;
v___y_6946_ = v___y_7091_;
v_a_6947_ = v___x_7110_;
goto v___jp_6934_;
}
else
{
lean_object* v_a_7111_; lean_object* v___x_7113_; uint8_t v_isShared_7114_; uint8_t v_isSharedCheck_7118_; 
lean_dec_ref(v___y_7085_);
lean_dec(v___y_7081_);
lean_dec(v_goal_6925_);
v_a_7111_ = lean_ctor_get(v___x_7097_, 0);
v_isSharedCheck_7118_ = !lean_is_exclusive(v___x_7097_);
if (v_isSharedCheck_7118_ == 0)
{
v___x_7113_ = v___x_7097_;
v_isShared_7114_ = v_isSharedCheck_7118_;
goto v_resetjp_7112_;
}
else
{
lean_inc(v_a_7111_);
lean_dec(v___x_7097_);
v___x_7113_ = lean_box(0);
v_isShared_7114_ = v_isSharedCheck_7118_;
goto v_resetjp_7112_;
}
v_resetjp_7112_:
{
lean_object* v___x_7116_; 
if (v_isShared_7114_ == 0)
{
v___x_7116_ = v___x_7113_;
goto v_reusejp_7115_;
}
else
{
lean_object* v_reuseFailAlloc_7117_; 
v_reuseFailAlloc_7117_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7117_, 0, v_a_7111_);
v___x_7116_ = v_reuseFailAlloc_7117_;
goto v_reusejp_7115_;
}
v_reusejp_7115_:
{
return v___x_7116_;
}
}
}
}
else
{
lean_object* v_a_7119_; lean_object* v___x_7121_; uint8_t v_isShared_7122_; uint8_t v_isSharedCheck_7126_; 
lean_dec_ref(v___y_7085_);
lean_dec_ref(v___y_7082_);
lean_dec(v___y_7081_);
lean_dec(v_goal_6925_);
v_a_7119_ = lean_ctor_get(v___x_7095_, 0);
v_isSharedCheck_7126_ = !lean_is_exclusive(v___x_7095_);
if (v_isSharedCheck_7126_ == 0)
{
v___x_7121_ = v___x_7095_;
v_isShared_7122_ = v_isSharedCheck_7126_;
goto v_resetjp_7120_;
}
else
{
lean_inc(v_a_7119_);
lean_dec(v___x_7095_);
v___x_7121_ = lean_box(0);
v_isShared_7122_ = v_isSharedCheck_7126_;
goto v_resetjp_7120_;
}
v_resetjp_7120_:
{
lean_object* v___x_7124_; 
if (v_isShared_7122_ == 0)
{
v___x_7124_ = v___x_7121_;
goto v_reusejp_7123_;
}
else
{
lean_object* v_reuseFailAlloc_7125_; 
v_reuseFailAlloc_7125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7125_, 0, v_a_7119_);
v___x_7124_ = v_reuseFailAlloc_7125_;
goto v_reusejp_7123_;
}
v_reusejp_7123_:
{
return v___x_7124_;
}
}
}
}
else
{
v___y_6997_ = v___y_7078_;
v___y_6998_ = v___y_7079_;
v___y_6999_ = v___y_7080_;
v___y_7000_ = v___y_7081_;
v___y_7001_ = v___y_7082_;
v___y_7002_ = v___y_7083_;
v___y_7003_ = v___y_7084_;
v___y_7004_ = v___y_7085_;
v___y_7005_ = v___y_7086_;
v___y_7006_ = v___y_7087_;
v___y_7007_ = v___y_7088_;
v___y_7008_ = v___y_7089_;
v___y_7009_ = v___y_7090_;
v___y_7010_ = v___y_7091_;
goto v___jp_6996_;
}
}
v___jp_7127_:
{
lean_object* v_a_7143_; uint8_t v___x_7144_; 
v_a_7143_ = lean_ctor_get(v___y_7142_, 0);
lean_inc(v_a_7143_);
lean_dec_ref(v___y_7142_);
v___x_7144_ = lean_unbox(v_a_7143_);
lean_dec(v_a_7143_);
v___y_7078_ = v___y_7128_;
v___y_7079_ = v___y_7129_;
v___y_7080_ = v___y_7130_;
v___y_7081_ = v___y_7131_;
v___y_7082_ = v___y_7132_;
v___y_7083_ = v___y_7133_;
v___y_7084_ = v___y_7134_;
v___y_7085_ = v___y_7135_;
v___y_7086_ = v___y_7136_;
v___y_7087_ = v___y_7137_;
v___y_7088_ = v___y_7138_;
v___y_7089_ = v___y_7139_;
v___y_7090_ = v___y_7140_;
v___y_7091_ = v___y_7141_;
v_a_7092_ = v___x_7144_;
goto v___jp_7077_;
}
v___jp_7145_:
{
uint8_t v_generateScript_7158_; 
v_generateScript_7158_ = lean_ctor_get_uint8(v___y_7151_, sizeof(void*)*2);
if (v_generateScript_7158_ == 0)
{
lean_dec(v_a_7157_);
lean_dec_ref(v___y_7148_);
lean_dec_ref(v___y_7146_);
lean_dec(v_goal_6925_);
return v___y_7156_;
}
else
{
lean_object* v_toOptions_7159_; lean_object* v___x_7160_; lean_object* v_options_7161_; lean_object* v___x_7162_; uint8_t v___x_7163_; 
lean_dec_ref(v___y_7156_);
v_toOptions_7159_ = lean_ctor_get(v___y_7151_, 0);
v___x_7160_ = lean_st_ref_get(v___y_7150_);
v_options_7161_ = lean_ctor_get(v___y_7147_, 2);
v___x_7162_ = lp_aesop_Aesop_aesop_collectStats;
v___x_7163_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_7161_, v___x_7162_);
if (v___x_7163_ == 0)
{
lean_object* v___x_7164_; lean_object* v___x_7165_; lean_object* v_a_7166_; uint8_t v___x_7167_; 
v___x_7164_ = lp_aesop_Aesop_TraceOption_stats;
v___x_7165_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_7164_, v___y_7147_);
v_a_7166_ = lean_ctor_get(v___x_7165_, 0);
lean_inc(v_a_7166_);
v___x_7167_ = lean_unbox(v_a_7166_);
lean_dec(v_a_7166_);
if (v___x_7167_ == 0)
{
lean_object* v___x_7168_; lean_object* v___x_7169_; lean_object* v___x_7170_; uint8_t v___x_7171_; 
v___x_7168_ = lp_aesop_Aesop_aesop_stats_file;
v___x_7169_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_7161_, v___x_7168_);
v___x_7170_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_7171_ = lean_string_dec_eq(v___x_7169_, v___x_7170_);
lean_dec_ref(v___x_7169_);
if (v___x_7171_ == 0)
{
lean_dec_ref(v___x_7165_);
v___y_6997_ = v_generateScript_7158_;
v___y_6998_ = v___y_7149_;
v___y_6999_ = v___y_7153_;
v___y_7000_ = v_a_7157_;
v___y_7001_ = v___y_7146_;
v___y_7002_ = v_toOptions_7159_;
v___y_7003_ = v___y_7147_;
v___y_7004_ = v___y_7148_;
v___y_7005_ = v___x_7160_;
v___y_7006_ = v___y_7150_;
v___y_7007_ = v___y_7151_;
v___y_7008_ = v___y_7152_;
v___y_7009_ = v___y_7154_;
v___y_7010_ = v___y_7155_;
goto v___jp_6996_;
}
else
{
v___y_7128_ = v_generateScript_7158_;
v___y_7129_ = v___y_7149_;
v___y_7130_ = v___y_7153_;
v___y_7131_ = v_a_7157_;
v___y_7132_ = v___y_7146_;
v___y_7133_ = v_toOptions_7159_;
v___y_7134_ = v___y_7147_;
v___y_7135_ = v___y_7148_;
v___y_7136_ = v___x_7160_;
v___y_7137_ = v___y_7150_;
v___y_7138_ = v___y_7151_;
v___y_7139_ = v___y_7152_;
v___y_7140_ = v___y_7154_;
v___y_7141_ = v___y_7155_;
v___y_7142_ = v___x_7165_;
goto v___jp_7127_;
}
}
else
{
v___y_7128_ = v_generateScript_7158_;
v___y_7129_ = v___y_7149_;
v___y_7130_ = v___y_7153_;
v___y_7131_ = v_a_7157_;
v___y_7132_ = v___y_7146_;
v___y_7133_ = v_toOptions_7159_;
v___y_7134_ = v___y_7147_;
v___y_7135_ = v___y_7148_;
v___y_7136_ = v___x_7160_;
v___y_7137_ = v___y_7150_;
v___y_7138_ = v___y_7151_;
v___y_7139_ = v___y_7152_;
v___y_7140_ = v___y_7154_;
v___y_7141_ = v___y_7155_;
v___y_7142_ = v___x_7165_;
goto v___jp_7127_;
}
}
else
{
v___y_7078_ = v_generateScript_7158_;
v___y_7079_ = v___y_7149_;
v___y_7080_ = v___y_7153_;
v___y_7081_ = v_a_7157_;
v___y_7082_ = v___y_7146_;
v___y_7083_ = v_toOptions_7159_;
v___y_7084_ = v___y_7147_;
v___y_7085_ = v___y_7148_;
v___y_7086_ = v___x_7160_;
v___y_7087_ = v___y_7150_;
v___y_7088_ = v___y_7151_;
v___y_7089_ = v___y_7152_;
v___y_7090_ = v___y_7154_;
v___y_7091_ = v___y_7155_;
v_a_7092_ = v___x_7163_;
goto v___jp_7077_;
}
}
}
v___jp_7172_:
{
if (lean_obj_tag(v___y_7184_) == 0)
{
lean_object* v_a_7185_; lean_object* v___x_7187_; uint8_t v_isShared_7188_; uint8_t v_isSharedCheck_7222_; 
v_a_7185_ = lean_ctor_get(v___y_7184_, 0);
v_isSharedCheck_7222_ = !lean_is_exclusive(v___y_7184_);
if (v_isSharedCheck_7222_ == 0)
{
v___x_7187_ = v___y_7184_;
v_isShared_7188_ = v_isSharedCheck_7222_;
goto v_resetjp_7186_;
}
else
{
lean_inc(v_a_7185_);
lean_dec(v___y_7184_);
v___x_7187_ = lean_box(0);
v_isShared_7188_ = v_isSharedCheck_7222_;
goto v_resetjp_7186_;
}
v_resetjp_7186_:
{
lean_object* v___x_7189_; lean_object* v___x_7190_; lean_object* v_stats_7191_; lean_object* v_rulePatternCache_7192_; lean_object* v___x_7194_; uint8_t v_isShared_7195_; uint8_t v_isSharedCheck_7221_; 
v___x_7189_ = lean_io_mono_nanos_now();
v___x_7190_ = lean_st_ref_take(v___y_7180_);
v_stats_7191_ = lean_ctor_get(v___x_7190_, 1);
v_rulePatternCache_7192_ = lean_ctor_get(v___x_7190_, 0);
v_isSharedCheck_7221_ = !lean_is_exclusive(v___x_7190_);
if (v_isSharedCheck_7221_ == 0)
{
v___x_7194_ = v___x_7190_;
v_isShared_7195_ = v_isSharedCheck_7221_;
goto v_resetjp_7193_;
}
else
{
lean_inc(v_stats_7191_);
lean_inc(v_rulePatternCache_7192_);
lean_dec(v___x_7190_);
v___x_7194_ = lean_box(0);
v_isShared_7195_ = v_isSharedCheck_7221_;
goto v_resetjp_7193_;
}
v_resetjp_7193_:
{
lean_object* v_total_7196_; lean_object* v_configParsing_7197_; lean_object* v_ruleSetConstruction_7198_; lean_object* v_ruleSelection_7199_; lean_object* v_script_7200_; lean_object* v_forwardState_7201_; lean_object* v_scriptGenerated_7202_; lean_object* v_ruleStats_7203_; lean_object* v_goalStats_7204_; lean_object* v___x_7206_; uint8_t v_isShared_7207_; uint8_t v_isSharedCheck_7219_; 
v_total_7196_ = lean_ctor_get(v_stats_7191_, 0);
v_configParsing_7197_ = lean_ctor_get(v_stats_7191_, 1);
v_ruleSetConstruction_7198_ = lean_ctor_get(v_stats_7191_, 2);
v_ruleSelection_7199_ = lean_ctor_get(v_stats_7191_, 4);
v_script_7200_ = lean_ctor_get(v_stats_7191_, 5);
v_forwardState_7201_ = lean_ctor_get(v_stats_7191_, 6);
v_scriptGenerated_7202_ = lean_ctor_get(v_stats_7191_, 7);
v_ruleStats_7203_ = lean_ctor_get(v_stats_7191_, 8);
v_goalStats_7204_ = lean_ctor_get(v_stats_7191_, 9);
v_isSharedCheck_7219_ = !lean_is_exclusive(v_stats_7191_);
if (v_isSharedCheck_7219_ == 0)
{
lean_object* v_unused_7220_; 
v_unused_7220_ = lean_ctor_get(v_stats_7191_, 3);
lean_dec(v_unused_7220_);
v___x_7206_ = v_stats_7191_;
v_isShared_7207_ = v_isSharedCheck_7219_;
goto v_resetjp_7205_;
}
else
{
lean_inc(v_goalStats_7204_);
lean_inc(v_ruleStats_7203_);
lean_inc(v_scriptGenerated_7202_);
lean_inc(v_forwardState_7201_);
lean_inc(v_script_7200_);
lean_inc(v_ruleSelection_7199_);
lean_inc(v_ruleSetConstruction_7198_);
lean_inc(v_configParsing_7197_);
lean_inc(v_total_7196_);
lean_dec(v_stats_7191_);
v___x_7206_ = lean_box(0);
v_isShared_7207_ = v_isSharedCheck_7219_;
goto v_resetjp_7205_;
}
v_resetjp_7205_:
{
lean_object* v___x_7208_; lean_object* v___x_7210_; 
v___x_7208_ = lean_nat_sub(v___x_7189_, v___y_7183_);
lean_dec(v___y_7183_);
lean_dec(v___x_7189_);
if (v_isShared_7207_ == 0)
{
lean_ctor_set(v___x_7206_, 3, v___x_7208_);
v___x_7210_ = v___x_7206_;
goto v_reusejp_7209_;
}
else
{
lean_object* v_reuseFailAlloc_7218_; 
v_reuseFailAlloc_7218_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_7218_, 0, v_total_7196_);
lean_ctor_set(v_reuseFailAlloc_7218_, 1, v_configParsing_7197_);
lean_ctor_set(v_reuseFailAlloc_7218_, 2, v_ruleSetConstruction_7198_);
lean_ctor_set(v_reuseFailAlloc_7218_, 3, v___x_7208_);
lean_ctor_set(v_reuseFailAlloc_7218_, 4, v_ruleSelection_7199_);
lean_ctor_set(v_reuseFailAlloc_7218_, 5, v_script_7200_);
lean_ctor_set(v_reuseFailAlloc_7218_, 6, v_forwardState_7201_);
lean_ctor_set(v_reuseFailAlloc_7218_, 7, v_scriptGenerated_7202_);
lean_ctor_set(v_reuseFailAlloc_7218_, 8, v_ruleStats_7203_);
lean_ctor_set(v_reuseFailAlloc_7218_, 9, v_goalStats_7204_);
v___x_7210_ = v_reuseFailAlloc_7218_;
goto v_reusejp_7209_;
}
v_reusejp_7209_:
{
lean_object* v___x_7212_; 
if (v_isShared_7195_ == 0)
{
lean_ctor_set(v___x_7194_, 1, v___x_7210_);
v___x_7212_ = v___x_7194_;
goto v_reusejp_7211_;
}
else
{
lean_object* v_reuseFailAlloc_7217_; 
v_reuseFailAlloc_7217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_7217_, 0, v_rulePatternCache_7192_);
lean_ctor_set(v_reuseFailAlloc_7217_, 1, v___x_7210_);
v___x_7212_ = v_reuseFailAlloc_7217_;
goto v_reusejp_7211_;
}
v_reusejp_7211_:
{
lean_object* v___x_7213_; lean_object* v___x_7215_; 
v___x_7213_ = lean_st_ref_set(v___y_7180_, v___x_7212_);
lean_inc(v_a_7185_);
if (v_isShared_7188_ == 0)
{
v___x_7215_ = v___x_7187_;
goto v_reusejp_7214_;
}
else
{
lean_object* v_reuseFailAlloc_7216_; 
v_reuseFailAlloc_7216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7216_, 0, v_a_7185_);
v___x_7215_ = v_reuseFailAlloc_7216_;
goto v_reusejp_7214_;
}
v_reusejp_7214_:
{
v___y_7146_ = v___y_7173_;
v___y_7147_ = v___y_7174_;
v___y_7148_ = v___y_7175_;
v___y_7149_ = v___y_7177_;
v___y_7150_ = v___y_7176_;
v___y_7151_ = v___y_7178_;
v___y_7152_ = v___y_7180_;
v___y_7153_ = v___y_7179_;
v___y_7154_ = v___y_7181_;
v___y_7155_ = v___y_7182_;
v___y_7156_ = v___x_7215_;
v_a_7157_ = v_a_7185_;
goto v___jp_7145_;
}
}
}
}
}
}
}
else
{
lean_dec(v___y_7183_);
lean_dec_ref(v___y_7175_);
lean_dec_ref(v___y_7173_);
lean_dec(v_goal_6925_);
return v___y_7184_;
}
}
v___jp_7223_:
{
lean_object* v___x_7234_; lean_object* v_options_7235_; lean_object* v___x_7236_; uint8_t v___x_7237_; 
v___x_7234_ = lean_io_mono_nanos_now();
v_options_7235_ = lean_ctor_get(v___y_7225_, 2);
v___x_7236_ = lp_aesop_Aesop_aesop_dev_statefulForward;
v___x_7237_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_7235_, v___x_7236_);
if (v___x_7237_ == 0)
{
lean_object* v___x_7238_; 
lean_inc(v_goal_6925_);
v___x_7238_ = lp_aesop_Aesop_saturateCore(v_rs_6924_, v_goal_6925_, v___y_7229_, v___y_7228_, v___y_7230_, v___y_7232_, v___y_7231_, v___y_7225_, v___y_7233_);
v___y_7173_ = v___y_7224_;
v___y_7174_ = v___y_7225_;
v___y_7175_ = v___y_7226_;
v___y_7176_ = v___y_7228_;
v___y_7177_ = v___y_7227_;
v___y_7178_ = v___y_7229_;
v___y_7179_ = v___y_7231_;
v___y_7180_ = v___y_7230_;
v___y_7181_ = v___y_7232_;
v___y_7182_ = v___y_7233_;
v___y_7183_ = v___x_7234_;
v___y_7184_ = v___x_7238_;
goto v___jp_7172_;
}
else
{
lean_object* v___x_7239_; 
lean_inc(v_goal_6925_);
v___x_7239_ = lp_aesop_Aesop_Stateful_saturateCore(v_rs_6924_, v_goal_6925_, v___y_7229_, v___y_7228_, v___y_7230_, v___y_7232_, v___y_7231_, v___y_7225_, v___y_7233_);
v___y_7173_ = v___y_7224_;
v___y_7174_ = v___y_7225_;
v___y_7175_ = v___y_7226_;
v___y_7176_ = v___y_7228_;
v___y_7177_ = v___y_7227_;
v___y_7178_ = v___y_7229_;
v___y_7179_ = v___y_7231_;
v___y_7180_ = v___y_7230_;
v___y_7181_ = v___y_7232_;
v___y_7182_ = v___y_7233_;
v___y_7183_ = v___x_7234_;
v___y_7184_ = v___x_7239_;
goto v___jp_7172_;
}
}
v___jp_7240_:
{
if (lean_obj_tag(v___y_7251_) == 0)
{
lean_object* v_a_7252_; 
v_a_7252_ = lean_ctor_get(v___y_7251_, 0);
lean_inc(v_a_7252_);
v___y_7146_ = v___y_7241_;
v___y_7147_ = v___y_7242_;
v___y_7148_ = v___y_7243_;
v___y_7149_ = v___y_7245_;
v___y_7150_ = v___y_7244_;
v___y_7151_ = v___y_7246_;
v___y_7152_ = v___y_7248_;
v___y_7153_ = v___y_7247_;
v___y_7154_ = v___y_7249_;
v___y_7155_ = v___y_7250_;
v___y_7156_ = v___y_7251_;
v_a_7157_ = v_a_7252_;
goto v___jp_7145_;
}
else
{
lean_dec_ref(v___y_7243_);
lean_dec_ref(v___y_7241_);
lean_dec(v_goal_6925_);
return v___y_7251_;
}
}
v___jp_7253_:
{
if (v_a_7264_ == 0)
{
lean_object* v_options_7265_; lean_object* v___x_7266_; uint8_t v___x_7267_; 
v_options_7265_ = lean_ctor_get(v___y_7255_, 2);
v___x_7266_ = lp_aesop_Aesop_aesop_dev_statefulForward;
v___x_7267_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_7265_, v___x_7266_);
if (v___x_7267_ == 0)
{
lean_object* v___x_7268_; 
lean_inc(v_goal_6925_);
v___x_7268_ = lp_aesop_Aesop_saturateCore(v_rs_6924_, v_goal_6925_, v___y_7259_, v___y_7258_, v___y_7261_, v___y_7262_, v___y_7260_, v___y_7255_, v___y_7263_);
v___y_7241_ = v___y_7254_;
v___y_7242_ = v___y_7255_;
v___y_7243_ = v___y_7256_;
v___y_7244_ = v___y_7258_;
v___y_7245_ = v___y_7257_;
v___y_7246_ = v___y_7259_;
v___y_7247_ = v___y_7260_;
v___y_7248_ = v___y_7261_;
v___y_7249_ = v___y_7262_;
v___y_7250_ = v___y_7263_;
v___y_7251_ = v___x_7268_;
goto v___jp_7240_;
}
else
{
lean_object* v___x_7269_; 
lean_inc(v_goal_6925_);
v___x_7269_ = lp_aesop_Aesop_Stateful_saturateCore(v_rs_6924_, v_goal_6925_, v___y_7259_, v___y_7258_, v___y_7261_, v___y_7262_, v___y_7260_, v___y_7255_, v___y_7263_);
v___y_7241_ = v___y_7254_;
v___y_7242_ = v___y_7255_;
v___y_7243_ = v___y_7256_;
v___y_7244_ = v___y_7258_;
v___y_7245_ = v___y_7257_;
v___y_7246_ = v___y_7259_;
v___y_7247_ = v___y_7260_;
v___y_7248_ = v___y_7261_;
v___y_7249_ = v___y_7262_;
v___y_7250_ = v___y_7263_;
v___y_7251_ = v___x_7269_;
goto v___jp_7240_;
}
}
else
{
v___y_7224_ = v___y_7254_;
v___y_7225_ = v___y_7255_;
v___y_7226_ = v___y_7256_;
v___y_7227_ = v___y_7257_;
v___y_7228_ = v___y_7258_;
v___y_7229_ = v___y_7259_;
v___y_7230_ = v___y_7261_;
v___y_7231_ = v___y_7260_;
v___y_7232_ = v___y_7262_;
v___y_7233_ = v___y_7263_;
goto v___jp_7223_;
}
}
v___jp_7270_:
{
lean_object* v_a_7282_; uint8_t v___x_7283_; 
v_a_7282_ = lean_ctor_get(v___y_7281_, 0);
lean_inc(v_a_7282_);
lean_dec_ref(v___y_7281_);
v___x_7283_ = lean_unbox(v_a_7282_);
lean_dec(v_a_7282_);
v___y_7254_ = v___y_7271_;
v___y_7255_ = v___y_7272_;
v___y_7256_ = v___y_7273_;
v___y_7257_ = v___y_7275_;
v___y_7258_ = v___y_7274_;
v___y_7259_ = v___y_7276_;
v___y_7260_ = v___y_7278_;
v___y_7261_ = v___y_7277_;
v___y_7262_ = v___y_7279_;
v___y_7263_ = v___y_7280_;
v_a_7264_ = v___x_7283_;
goto v___jp_7253_;
}
v___jp_7284_:
{
lean_object* v___x_7296_; uint8_t v___x_7297_; 
v___x_7296_ = lp_aesop_Aesop_aesop_collectStats;
v___x_7297_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_7294_, v___x_7296_);
if (v___x_7297_ == 0)
{
lean_object* v___x_7298_; lean_object* v___x_7299_; lean_object* v_a_7300_; uint8_t v___x_7301_; 
v___x_7298_ = lp_aesop_Aesop_TraceOption_stats;
v___x_7299_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_7298_, v___y_7293_);
v_a_7300_ = lean_ctor_get(v___x_7299_, 0);
lean_inc(v_a_7300_);
v___x_7301_ = lean_unbox(v_a_7300_);
lean_dec(v_a_7300_);
if (v___x_7301_ == 0)
{
lean_object* v___x_7302_; lean_object* v___x_7303_; lean_object* v___x_7304_; uint8_t v___x_7305_; 
v___x_7302_ = lp_aesop_Aesop_aesop_stats_file;
v___x_7303_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_7294_, v___x_7302_);
v___x_7304_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_7305_ = lean_string_dec_eq(v___x_7303_, v___x_7304_);
lean_dec_ref(v___x_7303_);
if (v___x_7305_ == 0)
{
lean_dec_ref(v___x_7299_);
v___y_7224_ = v_tacticState_7287_;
v___y_7225_ = v___y_7293_;
v___y_7226_ = v___y_7285_;
v___y_7227_ = v___y_7286_;
v___y_7228_ = v___y_7289_;
v___y_7229_ = v___y_7288_;
v___y_7230_ = v___y_7290_;
v___y_7231_ = v___y_7292_;
v___y_7232_ = v___y_7291_;
v___y_7233_ = v___y_7295_;
goto v___jp_7223_;
}
else
{
v___y_7271_ = v_tacticState_7287_;
v___y_7272_ = v___y_7293_;
v___y_7273_ = v___y_7285_;
v___y_7274_ = v___y_7289_;
v___y_7275_ = v___y_7286_;
v___y_7276_ = v___y_7288_;
v___y_7277_ = v___y_7290_;
v___y_7278_ = v___y_7292_;
v___y_7279_ = v___y_7291_;
v___y_7280_ = v___y_7295_;
v___y_7281_ = v___x_7299_;
goto v___jp_7270_;
}
}
else
{
v___y_7271_ = v_tacticState_7287_;
v___y_7272_ = v___y_7293_;
v___y_7273_ = v___y_7285_;
v___y_7274_ = v___y_7289_;
v___y_7275_ = v___y_7286_;
v___y_7276_ = v___y_7288_;
v___y_7277_ = v___y_7290_;
v___y_7278_ = v___y_7292_;
v___y_7279_ = v___y_7291_;
v___y_7280_ = v___y_7295_;
v___y_7281_ = v___x_7299_;
goto v___jp_7270_;
}
}
else
{
v___y_7254_ = v_tacticState_7287_;
v___y_7255_ = v___y_7293_;
v___y_7256_ = v___y_7285_;
v___y_7257_ = v___y_7286_;
v___y_7258_ = v___y_7289_;
v___y_7259_ = v___y_7288_;
v___y_7260_ = v___y_7292_;
v___y_7261_ = v___y_7290_;
v___y_7262_ = v___y_7291_;
v___y_7263_ = v___y_7295_;
v_a_7264_ = v___x_7297_;
goto v___jp_7253_;
}
}
v___jp_7306_:
{
if (lean_obj_tag(v___y_7308_) == 0)
{
lean_object* v_a_7309_; lean_object* v___x_7311_; uint8_t v_isShared_7312_; uint8_t v_isSharedCheck_7346_; 
v_a_7309_ = lean_ctor_get(v___y_7308_, 0);
v_isSharedCheck_7346_ = !lean_is_exclusive(v___y_7308_);
if (v_isSharedCheck_7346_ == 0)
{
v___x_7311_ = v___y_7308_;
v_isShared_7312_ = v_isSharedCheck_7346_;
goto v_resetjp_7310_;
}
else
{
lean_inc(v_a_7309_);
lean_dec(v___y_7308_);
v___x_7311_ = lean_box(0);
v_isShared_7312_ = v_isSharedCheck_7346_;
goto v_resetjp_7310_;
}
v_resetjp_7310_:
{
lean_object* v___x_7313_; lean_object* v___x_7314_; lean_object* v_stats_7315_; lean_object* v_rulePatternCache_7316_; lean_object* v___x_7318_; uint8_t v_isShared_7319_; uint8_t v_isSharedCheck_7345_; 
v___x_7313_ = lean_io_mono_nanos_now();
v___x_7314_ = lean_st_ref_take(v_a_6928_);
v_stats_7315_ = lean_ctor_get(v___x_7314_, 1);
v_rulePatternCache_7316_ = lean_ctor_get(v___x_7314_, 0);
v_isSharedCheck_7345_ = !lean_is_exclusive(v___x_7314_);
if (v_isSharedCheck_7345_ == 0)
{
v___x_7318_ = v___x_7314_;
v_isShared_7319_ = v_isSharedCheck_7345_;
goto v_resetjp_7317_;
}
else
{
lean_inc(v_stats_7315_);
lean_inc(v_rulePatternCache_7316_);
lean_dec(v___x_7314_);
v___x_7318_ = lean_box(0);
v_isShared_7319_ = v_isSharedCheck_7345_;
goto v_resetjp_7317_;
}
v_resetjp_7317_:
{
lean_object* v_configParsing_7320_; lean_object* v_ruleSetConstruction_7321_; lean_object* v_search_7322_; lean_object* v_ruleSelection_7323_; lean_object* v_script_7324_; lean_object* v_forwardState_7325_; lean_object* v_scriptGenerated_7326_; lean_object* v_ruleStats_7327_; lean_object* v_goalStats_7328_; lean_object* v___x_7330_; uint8_t v_isShared_7331_; uint8_t v_isSharedCheck_7343_; 
v_configParsing_7320_ = lean_ctor_get(v_stats_7315_, 1);
v_ruleSetConstruction_7321_ = lean_ctor_get(v_stats_7315_, 2);
v_search_7322_ = lean_ctor_get(v_stats_7315_, 3);
v_ruleSelection_7323_ = lean_ctor_get(v_stats_7315_, 4);
v_script_7324_ = lean_ctor_get(v_stats_7315_, 5);
v_forwardState_7325_ = lean_ctor_get(v_stats_7315_, 6);
v_scriptGenerated_7326_ = lean_ctor_get(v_stats_7315_, 7);
v_ruleStats_7327_ = lean_ctor_get(v_stats_7315_, 8);
v_goalStats_7328_ = lean_ctor_get(v_stats_7315_, 9);
v_isSharedCheck_7343_ = !lean_is_exclusive(v_stats_7315_);
if (v_isSharedCheck_7343_ == 0)
{
lean_object* v_unused_7344_; 
v_unused_7344_ = lean_ctor_get(v_stats_7315_, 0);
lean_dec(v_unused_7344_);
v___x_7330_ = v_stats_7315_;
v_isShared_7331_ = v_isSharedCheck_7343_;
goto v_resetjp_7329_;
}
else
{
lean_inc(v_goalStats_7328_);
lean_inc(v_ruleStats_7327_);
lean_inc(v_scriptGenerated_7326_);
lean_inc(v_forwardState_7325_);
lean_inc(v_script_7324_);
lean_inc(v_ruleSelection_7323_);
lean_inc(v_search_7322_);
lean_inc(v_ruleSetConstruction_7321_);
lean_inc(v_configParsing_7320_);
lean_dec(v_stats_7315_);
v___x_7330_ = lean_box(0);
v_isShared_7331_ = v_isSharedCheck_7343_;
goto v_resetjp_7329_;
}
v_resetjp_7329_:
{
lean_object* v___x_7332_; lean_object* v___x_7334_; 
v___x_7332_ = lean_nat_sub(v___x_7313_, v___y_7307_);
lean_dec(v___y_7307_);
lean_dec(v___x_7313_);
if (v_isShared_7331_ == 0)
{
lean_ctor_set(v___x_7330_, 0, v___x_7332_);
v___x_7334_ = v___x_7330_;
goto v_reusejp_7333_;
}
else
{
lean_object* v_reuseFailAlloc_7342_; 
v_reuseFailAlloc_7342_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_7342_, 0, v___x_7332_);
lean_ctor_set(v_reuseFailAlloc_7342_, 1, v_configParsing_7320_);
lean_ctor_set(v_reuseFailAlloc_7342_, 2, v_ruleSetConstruction_7321_);
lean_ctor_set(v_reuseFailAlloc_7342_, 3, v_search_7322_);
lean_ctor_set(v_reuseFailAlloc_7342_, 4, v_ruleSelection_7323_);
lean_ctor_set(v_reuseFailAlloc_7342_, 5, v_script_7324_);
lean_ctor_set(v_reuseFailAlloc_7342_, 6, v_forwardState_7325_);
lean_ctor_set(v_reuseFailAlloc_7342_, 7, v_scriptGenerated_7326_);
lean_ctor_set(v_reuseFailAlloc_7342_, 8, v_ruleStats_7327_);
lean_ctor_set(v_reuseFailAlloc_7342_, 9, v_goalStats_7328_);
v___x_7334_ = v_reuseFailAlloc_7342_;
goto v_reusejp_7333_;
}
v_reusejp_7333_:
{
lean_object* v___x_7336_; 
if (v_isShared_7319_ == 0)
{
lean_ctor_set(v___x_7318_, 1, v___x_7334_);
v___x_7336_ = v___x_7318_;
goto v_reusejp_7335_;
}
else
{
lean_object* v_reuseFailAlloc_7341_; 
v_reuseFailAlloc_7341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_7341_, 0, v_rulePatternCache_7316_);
lean_ctor_set(v_reuseFailAlloc_7341_, 1, v___x_7334_);
v___x_7336_ = v_reuseFailAlloc_7341_;
goto v_reusejp_7335_;
}
v_reusejp_7335_:
{
lean_object* v___x_7337_; lean_object* v___x_7339_; 
v___x_7337_ = lean_st_ref_set(v_a_6928_, v___x_7336_);
if (v_isShared_7312_ == 0)
{
v___x_7339_ = v___x_7311_;
goto v_reusejp_7338_;
}
else
{
lean_object* v_reuseFailAlloc_7340_; 
v_reuseFailAlloc_7340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7340_, 0, v_a_7309_);
v___x_7339_ = v_reuseFailAlloc_7340_;
goto v_reusejp_7338_;
}
v_reusejp_7338_:
{
return v___x_7339_;
}
}
}
}
}
}
}
else
{
lean_dec(v___y_7307_);
return v___y_7308_;
}
}
v___jp_7347_:
{
lean_object* v___x_7350_; lean_object* v___x_7351_; 
v___x_7350_ = lean_io_mono_nanos_now();
v___x_7351_ = lp_aesop_Aesop_Script_TacticState_mkInitial(v_goal_6925_, v_a_6929_, v_a_6930_, v_a_6931_, v_a_6932_);
if (lean_obj_tag(v___x_7351_) == 0)
{
lean_object* v_a_7352_; lean_object* v___x_7353_; lean_object* v___x_7354_; lean_object* v_stats_7355_; lean_object* v_rulePatternCache_7356_; lean_object* v___x_7358_; uint8_t v_isShared_7359_; uint8_t v_isSharedCheck_7384_; 
v_a_7352_ = lean_ctor_get(v___x_7351_, 0);
lean_inc(v_a_7352_);
lean_dec_ref_known(v___x_7351_, 1);
v___x_7353_ = lean_io_mono_nanos_now();
v___x_7354_ = lean_st_ref_take(v_a_6928_);
v_stats_7355_ = lean_ctor_get(v___x_7354_, 1);
v_rulePatternCache_7356_ = lean_ctor_get(v___x_7354_, 0);
v_isSharedCheck_7384_ = !lean_is_exclusive(v___x_7354_);
if (v_isSharedCheck_7384_ == 0)
{
v___x_7358_ = v___x_7354_;
v_isShared_7359_ = v_isSharedCheck_7384_;
goto v_resetjp_7357_;
}
else
{
lean_inc(v_stats_7355_);
lean_inc(v_rulePatternCache_7356_);
lean_dec(v___x_7354_);
v___x_7358_ = lean_box(0);
v_isShared_7359_ = v_isSharedCheck_7384_;
goto v_resetjp_7357_;
}
v_resetjp_7357_:
{
lean_object* v_total_7360_; lean_object* v_configParsing_7361_; lean_object* v_ruleSetConstruction_7362_; lean_object* v_search_7363_; lean_object* v_ruleSelection_7364_; lean_object* v_script_7365_; lean_object* v_forwardState_7366_; lean_object* v_scriptGenerated_7367_; lean_object* v_ruleStats_7368_; lean_object* v_goalStats_7369_; lean_object* v___x_7371_; uint8_t v_isShared_7372_; uint8_t v_isSharedCheck_7383_; 
v_total_7360_ = lean_ctor_get(v_stats_7355_, 0);
v_configParsing_7361_ = lean_ctor_get(v_stats_7355_, 1);
v_ruleSetConstruction_7362_ = lean_ctor_get(v_stats_7355_, 2);
v_search_7363_ = lean_ctor_get(v_stats_7355_, 3);
v_ruleSelection_7364_ = lean_ctor_get(v_stats_7355_, 4);
v_script_7365_ = lean_ctor_get(v_stats_7355_, 5);
v_forwardState_7366_ = lean_ctor_get(v_stats_7355_, 6);
v_scriptGenerated_7367_ = lean_ctor_get(v_stats_7355_, 7);
v_ruleStats_7368_ = lean_ctor_get(v_stats_7355_, 8);
v_goalStats_7369_ = lean_ctor_get(v_stats_7355_, 9);
v_isSharedCheck_7383_ = !lean_is_exclusive(v_stats_7355_);
if (v_isSharedCheck_7383_ == 0)
{
v___x_7371_ = v_stats_7355_;
v_isShared_7372_ = v_isSharedCheck_7383_;
goto v_resetjp_7370_;
}
else
{
lean_inc(v_goalStats_7369_);
lean_inc(v_ruleStats_7368_);
lean_inc(v_scriptGenerated_7367_);
lean_inc(v_forwardState_7366_);
lean_inc(v_script_7365_);
lean_inc(v_ruleSelection_7364_);
lean_inc(v_search_7363_);
lean_inc(v_ruleSetConstruction_7362_);
lean_inc(v_configParsing_7361_);
lean_inc(v_total_7360_);
lean_dec(v_stats_7355_);
v___x_7371_ = lean_box(0);
v_isShared_7372_ = v_isSharedCheck_7383_;
goto v_resetjp_7370_;
}
v_resetjp_7370_:
{
lean_object* v___x_7373_; lean_object* v___x_7374_; lean_object* v___x_7376_; 
v___x_7373_ = lean_nat_sub(v___x_7353_, v___x_7350_);
lean_dec(v___x_7350_);
lean_dec(v___x_7353_);
v___x_7374_ = lean_nat_add(v_script_7365_, v___x_7373_);
lean_dec(v___x_7373_);
lean_dec(v_script_7365_);
if (v_isShared_7372_ == 0)
{
lean_ctor_set(v___x_7371_, 5, v___x_7374_);
v___x_7376_ = v___x_7371_;
goto v_reusejp_7375_;
}
else
{
lean_object* v_reuseFailAlloc_7382_; 
v_reuseFailAlloc_7382_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_7382_, 0, v_total_7360_);
lean_ctor_set(v_reuseFailAlloc_7382_, 1, v_configParsing_7361_);
lean_ctor_set(v_reuseFailAlloc_7382_, 2, v_ruleSetConstruction_7362_);
lean_ctor_set(v_reuseFailAlloc_7382_, 3, v_search_7363_);
lean_ctor_set(v_reuseFailAlloc_7382_, 4, v_ruleSelection_7364_);
lean_ctor_set(v_reuseFailAlloc_7382_, 5, v___x_7374_);
lean_ctor_set(v_reuseFailAlloc_7382_, 6, v_forwardState_7366_);
lean_ctor_set(v_reuseFailAlloc_7382_, 7, v_scriptGenerated_7367_);
lean_ctor_set(v_reuseFailAlloc_7382_, 8, v_ruleStats_7368_);
lean_ctor_set(v_reuseFailAlloc_7382_, 9, v_goalStats_7369_);
v___x_7376_ = v_reuseFailAlloc_7382_;
goto v_reusejp_7375_;
}
v_reusejp_7375_:
{
lean_object* v___x_7378_; 
if (v_isShared_7359_ == 0)
{
lean_ctor_set(v___x_7358_, 1, v___x_7376_);
v___x_7378_ = v___x_7358_;
goto v_reusejp_7377_;
}
else
{
lean_object* v_reuseFailAlloc_7381_; 
v_reuseFailAlloc_7381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_7381_, 0, v_rulePatternCache_7356_);
lean_ctor_set(v_reuseFailAlloc_7381_, 1, v___x_7376_);
v___x_7378_ = v_reuseFailAlloc_7381_;
goto v_reusejp_7377_;
}
v_reusejp_7377_:
{
lean_object* v___x_7379_; lean_object* v___x_7380_; 
v___x_7379_ = lean_st_ref_set(v_a_6928_, v___x_7378_);
lean_inc(v_a_6932_);
lean_inc_ref(v_a_6931_);
lean_inc(v_a_6930_);
lean_inc_ref(v_a_6929_);
lean_inc(v_a_6928_);
lean_inc(v_a_6927_);
lean_inc_ref(v_a_6926_);
v___x_7380_ = lean_apply_9(v___y_7348_, v_a_7352_, v_a_6926_, v_a_6927_, v_a_6928_, v_a_6929_, v_a_6930_, v_a_6931_, v_a_6932_, lean_box(0));
v___y_7307_ = v___y_7349_;
v___y_7308_ = v___x_7380_;
goto v___jp_7306_;
}
}
}
}
}
else
{
lean_object* v_a_7385_; lean_object* v___x_7387_; uint8_t v_isShared_7388_; uint8_t v_isSharedCheck_7392_; 
lean_dec(v___x_7350_);
lean_dec(v___y_7349_);
lean_dec_ref(v___y_7348_);
v_a_7385_ = lean_ctor_get(v___x_7351_, 0);
v_isSharedCheck_7392_ = !lean_is_exclusive(v___x_7351_);
if (v_isSharedCheck_7392_ == 0)
{
v___x_7387_ = v___x_7351_;
v_isShared_7388_ = v_isSharedCheck_7392_;
goto v_resetjp_7386_;
}
else
{
lean_inc(v_a_7385_);
lean_dec(v___x_7351_);
v___x_7387_ = lean_box(0);
v_isShared_7388_ = v_isSharedCheck_7392_;
goto v_resetjp_7386_;
}
v_resetjp_7386_:
{
lean_object* v___x_7390_; 
if (v_isShared_7388_ == 0)
{
v___x_7390_ = v___x_7387_;
goto v_reusejp_7389_;
}
else
{
lean_object* v_reuseFailAlloc_7391_; 
v_reuseFailAlloc_7391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7391_, 0, v_a_7385_);
v___x_7390_ = v_reuseFailAlloc_7391_;
goto v_reusejp_7389_;
}
v_reusejp_7389_:
{
return v___x_7390_;
}
}
}
}
v___jp_7393_:
{
lean_object* v___x_7396_; 
v___x_7396_ = lp_aesop_Aesop_Script_TacticState_mkInitial(v_goal_6925_, v_a_6929_, v_a_6930_, v_a_6931_, v_a_6932_);
if (lean_obj_tag(v___x_7396_) == 0)
{
lean_object* v_a_7397_; lean_object* v___x_7398_; 
v_a_7397_ = lean_ctor_get(v___x_7396_, 0);
lean_inc(v_a_7397_);
lean_dec_ref_known(v___x_7396_, 1);
lean_inc(v_a_6932_);
lean_inc_ref(v_a_6931_);
lean_inc(v_a_6930_);
lean_inc_ref(v_a_6929_);
lean_inc(v_a_6928_);
lean_inc(v_a_6927_);
lean_inc_ref(v_a_6926_);
v___x_7398_ = lean_apply_9(v___y_7394_, v_a_7397_, v_a_6926_, v_a_6927_, v_a_6928_, v_a_6929_, v_a_6930_, v_a_6931_, v_a_6932_, lean_box(0));
v___y_7307_ = v___y_7395_;
v___y_7308_ = v___x_7398_;
goto v___jp_7306_;
}
else
{
lean_object* v_a_7399_; lean_object* v___x_7401_; uint8_t v_isShared_7402_; uint8_t v_isSharedCheck_7406_; 
lean_dec(v___y_7395_);
lean_dec_ref(v___y_7394_);
v_a_7399_ = lean_ctor_get(v___x_7396_, 0);
v_isSharedCheck_7406_ = !lean_is_exclusive(v___x_7396_);
if (v_isSharedCheck_7406_ == 0)
{
v___x_7401_ = v___x_7396_;
v_isShared_7402_ = v_isSharedCheck_7406_;
goto v_resetjp_7400_;
}
else
{
lean_inc(v_a_7399_);
lean_dec(v___x_7396_);
v___x_7401_ = lean_box(0);
v_isShared_7402_ = v_isSharedCheck_7406_;
goto v_resetjp_7400_;
}
v_resetjp_7400_:
{
lean_object* v___x_7404_; 
if (v_isShared_7402_ == 0)
{
v___x_7404_ = v___x_7401_;
goto v_reusejp_7403_;
}
else
{
lean_object* v_reuseFailAlloc_7405_; 
v_reuseFailAlloc_7405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7405_, 0, v_a_7399_);
v___x_7404_ = v_reuseFailAlloc_7405_;
goto v_reusejp_7403_;
}
v_reusejp_7403_:
{
return v___x_7404_;
}
}
}
}
v___jp_7407_:
{
lean_object* v_a_7411_; uint8_t v___x_7412_; 
v_a_7411_ = lean_ctor_get(v___y_7410_, 0);
lean_inc(v_a_7411_);
lean_dec_ref(v___y_7410_);
v___x_7412_ = lean_unbox(v_a_7411_);
lean_dec(v_a_7411_);
if (v___x_7412_ == 0)
{
v___y_7394_ = v___y_7408_;
v___y_7395_ = v___y_7409_;
goto v___jp_7393_;
}
else
{
v___y_7348_ = v___y_7408_;
v___y_7349_ = v___y_7409_;
goto v___jp_7347_;
}
}
v___jp_7414_:
{
lean_object* v___x_7417_; lean_object* v___x_7418_; 
v___x_7417_ = lean_io_mono_nanos_now();
lean_inc(v_goal_6925_);
v___x_7418_ = lp_aesop_Aesop_Script_TacticState_mkInitial(v_goal_6925_, v_a_6929_, v_a_6930_, v_a_6931_, v_a_6932_);
if (lean_obj_tag(v___x_7418_) == 0)
{
lean_object* v_a_7419_; lean_object* v___x_7420_; lean_object* v___x_7421_; lean_object* v_stats_7422_; lean_object* v_rulePatternCache_7423_; lean_object* v___x_7425_; uint8_t v_isShared_7426_; uint8_t v_isSharedCheck_7450_; 
v_a_7419_ = lean_ctor_get(v___x_7418_, 0);
lean_inc(v_a_7419_);
lean_dec_ref_known(v___x_7418_, 1);
v___x_7420_ = lean_io_mono_nanos_now();
v___x_7421_ = lean_st_ref_take(v_a_6928_);
v_stats_7422_ = lean_ctor_get(v___x_7421_, 1);
v_rulePatternCache_7423_ = lean_ctor_get(v___x_7421_, 0);
v_isSharedCheck_7450_ = !lean_is_exclusive(v___x_7421_);
if (v_isSharedCheck_7450_ == 0)
{
v___x_7425_ = v___x_7421_;
v_isShared_7426_ = v_isSharedCheck_7450_;
goto v_resetjp_7424_;
}
else
{
lean_inc(v_stats_7422_);
lean_inc(v_rulePatternCache_7423_);
lean_dec(v___x_7421_);
v___x_7425_ = lean_box(0);
v_isShared_7426_ = v_isSharedCheck_7450_;
goto v_resetjp_7424_;
}
v_resetjp_7424_:
{
lean_object* v_total_7427_; lean_object* v_configParsing_7428_; lean_object* v_ruleSetConstruction_7429_; lean_object* v_search_7430_; lean_object* v_ruleSelection_7431_; lean_object* v_script_7432_; lean_object* v_forwardState_7433_; lean_object* v_scriptGenerated_7434_; lean_object* v_ruleStats_7435_; lean_object* v_goalStats_7436_; lean_object* v___x_7438_; uint8_t v_isShared_7439_; uint8_t v_isSharedCheck_7449_; 
v_total_7427_ = lean_ctor_get(v_stats_7422_, 0);
v_configParsing_7428_ = lean_ctor_get(v_stats_7422_, 1);
v_ruleSetConstruction_7429_ = lean_ctor_get(v_stats_7422_, 2);
v_search_7430_ = lean_ctor_get(v_stats_7422_, 3);
v_ruleSelection_7431_ = lean_ctor_get(v_stats_7422_, 4);
v_script_7432_ = lean_ctor_get(v_stats_7422_, 5);
v_forwardState_7433_ = lean_ctor_get(v_stats_7422_, 6);
v_scriptGenerated_7434_ = lean_ctor_get(v_stats_7422_, 7);
v_ruleStats_7435_ = lean_ctor_get(v_stats_7422_, 8);
v_goalStats_7436_ = lean_ctor_get(v_stats_7422_, 9);
v_isSharedCheck_7449_ = !lean_is_exclusive(v_stats_7422_);
if (v_isSharedCheck_7449_ == 0)
{
v___x_7438_ = v_stats_7422_;
v_isShared_7439_ = v_isSharedCheck_7449_;
goto v_resetjp_7437_;
}
else
{
lean_inc(v_goalStats_7436_);
lean_inc(v_ruleStats_7435_);
lean_inc(v_scriptGenerated_7434_);
lean_inc(v_forwardState_7433_);
lean_inc(v_script_7432_);
lean_inc(v_ruleSelection_7431_);
lean_inc(v_search_7430_);
lean_inc(v_ruleSetConstruction_7429_);
lean_inc(v_configParsing_7428_);
lean_inc(v_total_7427_);
lean_dec(v_stats_7422_);
v___x_7438_ = lean_box(0);
v_isShared_7439_ = v_isSharedCheck_7449_;
goto v_resetjp_7437_;
}
v_resetjp_7437_:
{
lean_object* v___x_7440_; lean_object* v___x_7441_; lean_object* v___x_7443_; 
v___x_7440_ = lean_nat_sub(v___x_7420_, v___x_7417_);
lean_dec(v___x_7417_);
lean_dec(v___x_7420_);
v___x_7441_ = lean_nat_add(v_script_7432_, v___x_7440_);
lean_dec(v___x_7440_);
lean_dec(v_script_7432_);
if (v_isShared_7439_ == 0)
{
lean_ctor_set(v___x_7438_, 5, v___x_7441_);
v___x_7443_ = v___x_7438_;
goto v_reusejp_7442_;
}
else
{
lean_object* v_reuseFailAlloc_7448_; 
v_reuseFailAlloc_7448_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_7448_, 0, v_total_7427_);
lean_ctor_set(v_reuseFailAlloc_7448_, 1, v_configParsing_7428_);
lean_ctor_set(v_reuseFailAlloc_7448_, 2, v_ruleSetConstruction_7429_);
lean_ctor_set(v_reuseFailAlloc_7448_, 3, v_search_7430_);
lean_ctor_set(v_reuseFailAlloc_7448_, 4, v_ruleSelection_7431_);
lean_ctor_set(v_reuseFailAlloc_7448_, 5, v___x_7441_);
lean_ctor_set(v_reuseFailAlloc_7448_, 6, v_forwardState_7433_);
lean_ctor_set(v_reuseFailAlloc_7448_, 7, v_scriptGenerated_7434_);
lean_ctor_set(v_reuseFailAlloc_7448_, 8, v_ruleStats_7435_);
lean_ctor_set(v_reuseFailAlloc_7448_, 9, v_goalStats_7436_);
v___x_7443_ = v_reuseFailAlloc_7448_;
goto v_reusejp_7442_;
}
v_reusejp_7442_:
{
lean_object* v___x_7445_; 
if (v_isShared_7426_ == 0)
{
lean_ctor_set(v___x_7425_, 1, v___x_7443_);
v___x_7445_ = v___x_7425_;
goto v_reusejp_7444_;
}
else
{
lean_object* v_reuseFailAlloc_7447_; 
v_reuseFailAlloc_7447_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_7447_, 0, v_rulePatternCache_7423_);
lean_ctor_set(v_reuseFailAlloc_7447_, 1, v___x_7443_);
v___x_7445_ = v_reuseFailAlloc_7447_;
goto v_reusejp_7444_;
}
v_reusejp_7444_:
{
lean_object* v___x_7446_; 
v___x_7446_ = lean_st_ref_set(v_a_6928_, v___x_7445_);
v___y_7285_ = v___y_7415_;
v___y_7286_ = v___y_7416_;
v_tacticState_7287_ = v_a_7419_;
v___y_7288_ = v_a_6926_;
v___y_7289_ = v_a_6927_;
v___y_7290_ = v_a_6928_;
v___y_7291_ = v_a_6929_;
v___y_7292_ = v_a_6930_;
v___y_7293_ = v_a_6931_;
v_options_7294_ = v_options_7413_;
v___y_7295_ = v_a_6932_;
goto v___jp_7284_;
}
}
}
}
}
else
{
lean_object* v_a_7451_; lean_object* v___x_7453_; uint8_t v_isShared_7454_; uint8_t v_isSharedCheck_7458_; 
lean_dec(v___x_7417_);
lean_dec_ref(v___y_7415_);
lean_dec(v_goal_6925_);
lean_dec_ref(v_rs_6924_);
v_a_7451_ = lean_ctor_get(v___x_7418_, 0);
v_isSharedCheck_7458_ = !lean_is_exclusive(v___x_7418_);
if (v_isSharedCheck_7458_ == 0)
{
v___x_7453_ = v___x_7418_;
v_isShared_7454_ = v_isSharedCheck_7458_;
goto v_resetjp_7452_;
}
else
{
lean_inc(v_a_7451_);
lean_dec(v___x_7418_);
v___x_7453_ = lean_box(0);
v_isShared_7454_ = v_isSharedCheck_7458_;
goto v_resetjp_7452_;
}
v_resetjp_7452_:
{
lean_object* v___x_7456_; 
if (v_isShared_7454_ == 0)
{
v___x_7456_ = v___x_7453_;
goto v_reusejp_7455_;
}
else
{
lean_object* v_reuseFailAlloc_7457_; 
v_reuseFailAlloc_7457_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7457_, 0, v_a_7451_);
v___x_7456_ = v_reuseFailAlloc_7457_;
goto v_reusejp_7455_;
}
v_reusejp_7455_:
{
return v___x_7456_;
}
}
}
}
v___jp_7459_:
{
if (v_a_7462_ == 0)
{
lean_object* v___x_7463_; 
lean_inc(v_goal_6925_);
v___x_7463_ = lp_aesop_Aesop_Script_TacticState_mkInitial(v_goal_6925_, v_a_6929_, v_a_6930_, v_a_6931_, v_a_6932_);
if (lean_obj_tag(v___x_7463_) == 0)
{
lean_object* v_a_7464_; 
v_a_7464_ = lean_ctor_get(v___x_7463_, 0);
lean_inc(v_a_7464_);
lean_dec_ref_known(v___x_7463_, 1);
v___y_7285_ = v___y_7460_;
v___y_7286_ = v___y_7461_;
v_tacticState_7287_ = v_a_7464_;
v___y_7288_ = v_a_6926_;
v___y_7289_ = v_a_6927_;
v___y_7290_ = v_a_6928_;
v___y_7291_ = v_a_6929_;
v___y_7292_ = v_a_6930_;
v___y_7293_ = v_a_6931_;
v_options_7294_ = v_options_7413_;
v___y_7295_ = v_a_6932_;
goto v___jp_7284_;
}
else
{
lean_object* v_a_7465_; lean_object* v___x_7467_; uint8_t v_isShared_7468_; uint8_t v_isSharedCheck_7472_; 
lean_dec_ref(v___y_7460_);
lean_dec(v_goal_6925_);
lean_dec_ref(v_rs_6924_);
v_a_7465_ = lean_ctor_get(v___x_7463_, 0);
v_isSharedCheck_7472_ = !lean_is_exclusive(v___x_7463_);
if (v_isSharedCheck_7472_ == 0)
{
v___x_7467_ = v___x_7463_;
v_isShared_7468_ = v_isSharedCheck_7472_;
goto v_resetjp_7466_;
}
else
{
lean_inc(v_a_7465_);
lean_dec(v___x_7463_);
v___x_7467_ = lean_box(0);
v_isShared_7468_ = v_isSharedCheck_7472_;
goto v_resetjp_7466_;
}
v_resetjp_7466_:
{
lean_object* v___x_7470_; 
if (v_isShared_7468_ == 0)
{
v___x_7470_ = v___x_7467_;
goto v_reusejp_7469_;
}
else
{
lean_object* v_reuseFailAlloc_7471_; 
v_reuseFailAlloc_7471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7471_, 0, v_a_7465_);
v___x_7470_ = v_reuseFailAlloc_7471_;
goto v_reusejp_7469_;
}
v_reusejp_7469_:
{
return v___x_7470_;
}
}
}
}
else
{
v___y_7415_ = v___y_7460_;
v___y_7416_ = v___y_7461_;
goto v___jp_7414_;
}
}
v___jp_7473_:
{
lean_object* v_a_7477_; uint8_t v___x_7478_; 
v_a_7477_ = lean_ctor_get(v___y_7476_, 0);
lean_inc(v_a_7477_);
lean_dec_ref(v___y_7476_);
v___x_7478_ = lean_unbox(v_a_7477_);
lean_dec(v_a_7477_);
v___y_7460_ = v___y_7474_;
v___y_7461_ = v___y_7475_;
v_a_7462_ = v___x_7478_;
goto v___jp_7459_;
}
v___jp_7479_:
{
lean_object* v___x_7481_; 
v___x_7481_ = l_Lean_Meta_saveState___redArg(v_a_6930_, v_a_6932_);
if (lean_obj_tag(v___x_7481_) == 0)
{
uint8_t v_generateScript_7482_; 
v_generateScript_7482_ = lean_ctor_get_uint8(v_a_6926_, sizeof(void*)*2);
if (v_generateScript_7482_ == 0)
{
lean_object* v_a_7483_; lean_object* v___x_7484_; 
v_a_7483_ = lean_ctor_get(v___x_7481_, 0);
lean_inc(v_a_7483_);
lean_dec_ref_known(v___x_7481_, 1);
v___x_7484_ = lp_aesop_Aesop_Script_instInhabitedTacticState_default;
v___y_7285_ = v_a_7483_;
v___y_7286_ = v_a_7480_;
v_tacticState_7287_ = v___x_7484_;
v___y_7288_ = v_a_6926_;
v___y_7289_ = v_a_6927_;
v___y_7290_ = v_a_6928_;
v___y_7291_ = v_a_6929_;
v___y_7292_ = v_a_6930_;
v___y_7293_ = v_a_6931_;
v_options_7294_ = v_options_7413_;
v___y_7295_ = v_a_6932_;
goto v___jp_7284_;
}
else
{
lean_object* v_a_7485_; lean_object* v___x_7486_; uint8_t v___x_7487_; 
v_a_7485_ = lean_ctor_get(v___x_7481_, 0);
lean_inc(v_a_7485_);
lean_dec_ref_known(v___x_7481_, 1);
v___x_7486_ = lp_aesop_Aesop_aesop_collectStats;
v___x_7487_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_7413_, v___x_7486_);
if (v___x_7487_ == 0)
{
lean_object* v___x_7488_; lean_object* v___x_7489_; lean_object* v_a_7490_; uint8_t v___x_7491_; 
v___x_7488_ = lp_aesop_Aesop_TraceOption_stats;
v___x_7489_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_7488_, v_a_6931_);
v_a_7490_ = lean_ctor_get(v___x_7489_, 0);
lean_inc(v_a_7490_);
v___x_7491_ = lean_unbox(v_a_7490_);
lean_dec(v_a_7490_);
if (v___x_7491_ == 0)
{
lean_object* v___x_7492_; lean_object* v___x_7493_; lean_object* v___x_7494_; uint8_t v___x_7495_; 
v___x_7492_ = lp_aesop_Aesop_aesop_stats_file;
v___x_7493_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_7413_, v___x_7492_);
v___x_7494_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_7495_ = lean_string_dec_eq(v___x_7493_, v___x_7494_);
lean_dec_ref(v___x_7493_);
if (v___x_7495_ == 0)
{
lean_dec_ref(v___x_7489_);
v___y_7415_ = v_a_7485_;
v___y_7416_ = v_a_7480_;
goto v___jp_7414_;
}
else
{
v___y_7474_ = v_a_7485_;
v___y_7475_ = v_a_7480_;
v___y_7476_ = v___x_7489_;
goto v___jp_7473_;
}
}
else
{
v___y_7474_ = v_a_7485_;
v___y_7475_ = v_a_7480_;
v___y_7476_ = v___x_7489_;
goto v___jp_7473_;
}
}
else
{
v___y_7460_ = v_a_7485_;
v___y_7461_ = v_a_7480_;
v_a_7462_ = v___x_7487_;
goto v___jp_7459_;
}
}
}
else
{
lean_object* v_a_7496_; lean_object* v___x_7498_; uint8_t v_isShared_7499_; uint8_t v_isSharedCheck_7503_; 
lean_dec(v_goal_6925_);
lean_dec_ref(v_rs_6924_);
v_a_7496_ = lean_ctor_get(v___x_7481_, 0);
v_isSharedCheck_7503_ = !lean_is_exclusive(v___x_7481_);
if (v_isSharedCheck_7503_ == 0)
{
v___x_7498_ = v___x_7481_;
v_isShared_7499_ = v_isSharedCheck_7503_;
goto v_resetjp_7497_;
}
else
{
lean_inc(v_a_7496_);
lean_dec(v___x_7481_);
v___x_7498_ = lean_box(0);
v_isShared_7499_ = v_isSharedCheck_7503_;
goto v_resetjp_7497_;
}
v_resetjp_7497_:
{
lean_object* v___x_7501_; 
if (v_isShared_7499_ == 0)
{
v___x_7501_ = v___x_7498_;
goto v_reusejp_7500_;
}
else
{
lean_object* v_reuseFailAlloc_7502_; 
v_reuseFailAlloc_7502_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7502_, 0, v_a_7496_);
v___x_7501_ = v_reuseFailAlloc_7502_;
goto v_reusejp_7500_;
}
v_reusejp_7500_:
{
return v___x_7501_;
}
}
}
}
v___jp_7504_:
{
lean_object* v___x_7505_; lean_object* v___x_7506_; 
v___x_7505_ = lean_io_mono_nanos_now();
v___x_7506_ = l_Lean_Meta_saveState___redArg(v_a_6930_, v_a_6932_);
if (lean_obj_tag(v___x_7506_) == 0)
{
lean_object* v_a_7507_; uint8_t v_generateScript_7508_; lean_object* v___f_7509_; 
v_a_7507_ = lean_ctor_get(v___x_7506_, 0);
lean_inc_n(v_a_7507_, 2);
lean_dec_ref_known(v___x_7506_, 1);
v_generateScript_7508_ = lean_ctor_get_uint8(v_a_6926_, sizeof(void*)*2);
lean_inc_ref(v_rs_6924_);
lean_inc(v_goal_6925_);
v___f_7509_ = lean_alloc_closure((void*)(lp_aesop_Aesop_saturateMain_x27___lam__0___boxed), 12, 3);
lean_closure_set(v___f_7509_, 0, v_a_7507_);
lean_closure_set(v___f_7509_, 1, v_goal_6925_);
lean_closure_set(v___f_7509_, 2, v_rs_6924_);
if (v_generateScript_7508_ == 0)
{
lean_object* v___x_7510_; lean_object* v___x_7511_; 
lean_dec_ref(v___f_7509_);
v___x_7510_ = lp_aesop_Aesop_Script_instInhabitedTacticState_default;
v___x_7511_ = lp_aesop_Aesop_saturateMain_x27___lam__0(v_a_7507_, v_goal_6925_, v_rs_6924_, v___x_7510_, v_a_6926_, v_a_6927_, v_a_6928_, v_a_6929_, v_a_6930_, v_a_6931_, v_a_6932_);
v___y_7307_ = v___x_7505_;
v___y_7308_ = v___x_7511_;
goto v___jp_7306_;
}
else
{
lean_object* v___x_7512_; uint8_t v___x_7513_; 
lean_dec(v_a_7507_);
lean_dec_ref(v_rs_6924_);
v___x_7512_ = lp_aesop_Aesop_aesop_collectStats;
v___x_7513_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__4(v_options_7413_, v___x_7512_);
if (v___x_7513_ == 0)
{
lean_object* v___x_7514_; lean_object* v___x_7515_; lean_object* v_a_7516_; uint8_t v___x_7517_; 
v___x_7514_ = lp_aesop_Aesop_TraceOption_stats;
v___x_7515_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__2___redArg(v___x_7514_, v_a_6931_);
v_a_7516_ = lean_ctor_get(v___x_7515_, 0);
lean_inc(v_a_7516_);
v___x_7517_ = lean_unbox(v_a_7516_);
lean_dec(v_a_7516_);
if (v___x_7517_ == 0)
{
lean_object* v___x_7518_; lean_object* v___x_7519_; lean_object* v___x_7520_; uint8_t v___x_7521_; 
lean_dec_ref(v___x_7515_);
v___x_7518_ = lp_aesop_Aesop_aesop_stats_file;
v___x_7519_ = lp_aesop_Lean_Option_get___at___00__private_Aesop_Saturate_0__Aesop_saturateCore_runRule_spec__3(v_options_7413_, v___x_7518_);
v___x_7520_ = ((lean_object*)(lp_aesop___private_Aesop_Saturate_0__Aesop_saturateCore_runRule___redArg___lam__1___closed__0));
v___x_7521_ = lean_string_dec_eq(v___x_7519_, v___x_7520_);
lean_dec_ref(v___x_7519_);
if (v___x_7521_ == 0)
{
v___y_7348_ = v___f_7509_;
v___y_7349_ = v___x_7505_;
goto v___jp_7347_;
}
else
{
v___y_7394_ = v___f_7509_;
v___y_7395_ = v___x_7505_;
goto v___jp_7393_;
}
}
else
{
v___y_7408_ = v___f_7509_;
v___y_7409_ = v___x_7505_;
v___y_7410_ = v___x_7515_;
goto v___jp_7407_;
}
}
else
{
v___y_7348_ = v___f_7509_;
v___y_7349_ = v___x_7505_;
goto v___jp_7347_;
}
}
}
else
{
lean_object* v_a_7522_; lean_object* v___x_7524_; uint8_t v_isShared_7525_; uint8_t v_isSharedCheck_7529_; 
lean_dec(v___x_7505_);
lean_dec(v_goal_6925_);
lean_dec_ref(v_rs_6924_);
v_a_7522_ = lean_ctor_get(v___x_7506_, 0);
v_isSharedCheck_7529_ = !lean_is_exclusive(v___x_7506_);
if (v_isSharedCheck_7529_ == 0)
{
v___x_7524_ = v___x_7506_;
v_isShared_7525_ = v_isSharedCheck_7529_;
goto v_resetjp_7523_;
}
else
{
lean_inc(v_a_7522_);
lean_dec(v___x_7506_);
v___x_7524_ = lean_box(0);
v_isShared_7525_ = v_isSharedCheck_7529_;
goto v_resetjp_7523_;
}
v_resetjp_7523_:
{
lean_object* v___x_7527_; 
if (v_isShared_7525_ == 0)
{
v___x_7527_ = v___x_7524_;
goto v_reusejp_7526_;
}
else
{
lean_object* v_reuseFailAlloc_7528_; 
v_reuseFailAlloc_7528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7528_, 0, v_a_7522_);
v___x_7527_ = v_reuseFailAlloc_7528_;
goto v_reusejp_7526_;
}
v_reusejp_7526_:
{
return v___x_7527_;
}
}
}
}
v___jp_7530_:
{
lean_object* v_a_7532_; uint8_t v___x_7533_; 
v_a_7532_ = lean_ctor_get(v___y_7531_, 0);
lean_inc(v_a_7532_);
lean_dec_ref(v___y_7531_);
v___x_7533_ = lean_unbox(v_a_7532_);
if (v___x_7533_ == 0)
{
uint8_t v___x_7534_; 
v___x_7534_ = lean_unbox(v_a_7532_);
lean_dec(v_a_7532_);
v_a_7480_ = v___x_7534_;
goto v___jp_7479_;
}
else
{
lean_dec(v_a_7532_);
goto v___jp_7504_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain_x27___boxed(lean_object* v_rs_7546_, lean_object* v_goal_7547_, lean_object* v_a_7548_, lean_object* v_a_7549_, lean_object* v_a_7550_, lean_object* v_a_7551_, lean_object* v_a_7552_, lean_object* v_a_7553_, lean_object* v_a_7554_, lean_object* v_a_7555_){
_start:
{
lean_object* v_res_7556_; 
v_res_7556_ = lp_aesop_Aesop_saturateMain_x27(v_rs_7546_, v_goal_7547_, v_a_7548_, v_a_7549_, v_a_7550_, v_a_7551_, v_a_7552_, v_a_7553_, v_a_7554_);
lean_dec(v_a_7554_);
lean_dec_ref(v_a_7553_);
lean_dec(v_a_7552_);
lean_dec_ref(v_a_7551_);
lean_dec(v_a_7550_);
lean_dec(v_a_7549_);
lean_dec_ref(v_a_7548_);
return v_res_7556_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1(size_t v_sz_7557_, size_t v_i_7558_, lean_object* v_bs_7559_, lean_object* v___y_7560_, lean_object* v___y_7561_, lean_object* v___y_7562_, lean_object* v___y_7563_, lean_object* v___y_7564_, lean_object* v___y_7565_, lean_object* v___y_7566_){
_start:
{
lean_object* v___x_7568_; 
v___x_7568_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___redArg(v_sz_7557_, v_i_7558_, v_bs_7559_, v___y_7563_, v___y_7564_, v___y_7565_, v___y_7566_);
return v___x_7568_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1___boxed(lean_object* v_sz_7569_, lean_object* v_i_7570_, lean_object* v_bs_7571_, lean_object* v___y_7572_, lean_object* v___y_7573_, lean_object* v___y_7574_, lean_object* v___y_7575_, lean_object* v___y_7576_, lean_object* v___y_7577_, lean_object* v___y_7578_, lean_object* v___y_7579_){
_start:
{
size_t v_sz_boxed_7580_; size_t v_i_boxed_7581_; lean_object* v_res_7582_; 
v_sz_boxed_7580_ = lean_unbox_usize(v_sz_7569_);
lean_dec(v_sz_7569_);
v_i_boxed_7581_ = lean_unbox_usize(v_i_7570_);
lean_dec(v_i_7570_);
v_res_7582_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_saturateMain_x27_spec__1(v_sz_boxed_7580_, v_i_boxed_7581_, v_bs_7571_, v___y_7572_, v___y_7573_, v___y_7574_, v___y_7575_, v___y_7576_, v___y_7577_, v___y_7578_);
lean_dec(v___y_7578_);
lean_dec_ref(v___y_7577_);
lean_dec(v___y_7576_);
lean_dec_ref(v___y_7575_);
lean_dec(v___y_7574_);
lean_dec(v___y_7573_);
lean_dec_ref(v___y_7572_);
return v_res_7582_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0(lean_object* v_f_7583_, lean_object* v___y_7584_, lean_object* v___y_7585_, lean_object* v___y_7586_, lean_object* v___y_7587_, lean_object* v___y_7588_, lean_object* v___y_7589_, lean_object* v___y_7590_){
_start:
{
lean_object* v___x_7592_; 
v___x_7592_ = lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___redArg(v_f_7583_, v___y_7586_, v___y_7589_);
return v___x_7592_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0___boxed(lean_object* v_f_7593_, lean_object* v___y_7594_, lean_object* v___y_7595_, lean_object* v___y_7596_, lean_object* v___y_7597_, lean_object* v___y_7598_, lean_object* v___y_7599_, lean_object* v___y_7600_, lean_object* v___y_7601_){
_start:
{
lean_object* v_res_7602_; 
v_res_7602_ = lp_aesop_Aesop_modifyStatsIfEnabled___at___00Aesop_recordScriptGenerated___at___00Aesop_saturateMain_x27_spec__0_spec__0(v_f_7593_, v___y_7594_, v___y_7595_, v___y_7596_, v___y_7597_, v___y_7598_, v___y_7599_, v___y_7600_);
lean_dec(v___y_7600_);
lean_dec_ref(v___y_7599_);
lean_dec(v___y_7598_);
lean_dec_ref(v___y_7597_);
lean_dec(v___y_7596_);
lean_dec(v___y_7595_);
lean_dec_ref(v___y_7594_);
return v_res_7602_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5(lean_object* v_00_u03b1_7603_, lean_object* v_goal_7604_, lean_object* v_pre_7605_, lean_object* v___y_7606_, lean_object* v___y_7607_, lean_object* v___y_7608_, lean_object* v___y_7609_, lean_object* v___y_7610_, lean_object* v___y_7611_, lean_object* v___y_7612_){
_start:
{
lean_object* v___x_7614_; 
v___x_7614_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___redArg(v_goal_7604_, v_pre_7605_, v___y_7609_, v___y_7610_, v___y_7611_, v___y_7612_);
return v___x_7614_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5___boxed(lean_object* v_00_u03b1_7615_, lean_object* v_goal_7616_, lean_object* v_pre_7617_, lean_object* v___y_7618_, lean_object* v___y_7619_, lean_object* v___y_7620_, lean_object* v___y_7621_, lean_object* v___y_7622_, lean_object* v___y_7623_, lean_object* v___y_7624_, lean_object* v___y_7625_){
_start:
{
lean_object* v_res_7626_; 
v_res_7626_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_getVisibleGoalIndex___at___00Aesop_Script_Step_render___at___00Aesop_Script_UScript_render___at___00Aesop_saturateMain_x27_spec__2_spec__3_spec__4_spec__5(v_00_u03b1_7615_, v_goal_7616_, v_pre_7617_, v___y_7618_, v___y_7619_, v___y_7620_, v___y_7621_, v___y_7622_, v___y_7623_, v___y_7624_);
lean_dec(v___y_7624_);
lean_dec_ref(v___y_7623_);
lean_dec(v___y_7622_);
lean_dec_ref(v___y_7621_);
lean_dec(v___y_7620_);
lean_dec(v___y_7619_);
lean_dec_ref(v___y_7618_);
return v_res_7626_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain(lean_object* v_rs_7627_, lean_object* v_goal_7628_, lean_object* v_a_7629_, lean_object* v_a_7630_, lean_object* v_a_7631_, lean_object* v_a_7632_, lean_object* v_a_7633_, lean_object* v_a_7634_, lean_object* v_a_7635_){
_start:
{
lean_object* v___x_7637_; 
v___x_7637_ = lp_aesop_Aesop_saturateMain_x27(v_rs_7627_, v_goal_7628_, v_a_7629_, v_a_7630_, v_a_7631_, v_a_7632_, v_a_7633_, v_a_7634_, v_a_7635_);
if (lean_obj_tag(v___x_7637_) == 0)
{
lean_object* v_a_7638_; lean_object* v___x_7639_; lean_object* v_stats_7640_; lean_object* v___x_7641_; lean_object* v___x_7642_; 
v_a_7638_ = lean_ctor_get(v___x_7637_, 0);
lean_inc(v_a_7638_);
lean_dec_ref_known(v___x_7637_, 1);
v___x_7639_ = lean_st_ref_get(v_a_7631_);
v_stats_7640_ = lean_ctor_get(v___x_7639_, 1);
lean_inc_ref(v_stats_7640_);
lean_dec(v___x_7639_);
v___x_7641_ = lp_aesop_Aesop_TraceOption_stats;
v___x_7642_ = lp_aesop_Aesop_Stats_trace(v_stats_7640_, v___x_7641_, v_a_7634_, v_a_7635_);
if (lean_obj_tag(v___x_7642_) == 0)
{
lean_object* v___x_7644_; uint8_t v_isShared_7645_; uint8_t v_isSharedCheck_7649_; 
v_isSharedCheck_7649_ = !lean_is_exclusive(v___x_7642_);
if (v_isSharedCheck_7649_ == 0)
{
lean_object* v_unused_7650_; 
v_unused_7650_ = lean_ctor_get(v___x_7642_, 0);
lean_dec(v_unused_7650_);
v___x_7644_ = v___x_7642_;
v_isShared_7645_ = v_isSharedCheck_7649_;
goto v_resetjp_7643_;
}
else
{
lean_dec(v___x_7642_);
v___x_7644_ = lean_box(0);
v_isShared_7645_ = v_isSharedCheck_7649_;
goto v_resetjp_7643_;
}
v_resetjp_7643_:
{
lean_object* v___x_7647_; 
if (v_isShared_7645_ == 0)
{
lean_ctor_set(v___x_7644_, 0, v_a_7638_);
v___x_7647_ = v___x_7644_;
goto v_reusejp_7646_;
}
else
{
lean_object* v_reuseFailAlloc_7648_; 
v_reuseFailAlloc_7648_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7648_, 0, v_a_7638_);
v___x_7647_ = v_reuseFailAlloc_7648_;
goto v_reusejp_7646_;
}
v_reusejp_7646_:
{
return v___x_7647_;
}
}
}
else
{
lean_object* v_a_7651_; lean_object* v___x_7653_; uint8_t v_isShared_7654_; uint8_t v_isSharedCheck_7658_; 
lean_dec(v_a_7638_);
v_a_7651_ = lean_ctor_get(v___x_7642_, 0);
v_isSharedCheck_7658_ = !lean_is_exclusive(v___x_7642_);
if (v_isSharedCheck_7658_ == 0)
{
v___x_7653_ = v___x_7642_;
v_isShared_7654_ = v_isSharedCheck_7658_;
goto v_resetjp_7652_;
}
else
{
lean_inc(v_a_7651_);
lean_dec(v___x_7642_);
v___x_7653_ = lean_box(0);
v_isShared_7654_ = v_isSharedCheck_7658_;
goto v_resetjp_7652_;
}
v_resetjp_7652_:
{
lean_object* v___x_7656_; 
if (v_isShared_7654_ == 0)
{
v___x_7656_ = v___x_7653_;
goto v_reusejp_7655_;
}
else
{
lean_object* v_reuseFailAlloc_7657_; 
v_reuseFailAlloc_7657_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_7657_, 0, v_a_7651_);
v___x_7656_ = v_reuseFailAlloc_7657_;
goto v_reusejp_7655_;
}
v_reusejp_7655_:
{
return v___x_7656_;
}
}
}
}
else
{
return v___x_7637_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturateMain___boxed(lean_object* v_rs_7659_, lean_object* v_goal_7660_, lean_object* v_a_7661_, lean_object* v_a_7662_, lean_object* v_a_7663_, lean_object* v_a_7664_, lean_object* v_a_7665_, lean_object* v_a_7666_, lean_object* v_a_7667_, lean_object* v_a_7668_){
_start:
{
lean_object* v_res_7669_; 
v_res_7669_ = lp_aesop_Aesop_saturateMain(v_rs_7659_, v_goal_7660_, v_a_7661_, v_a_7662_, v_a_7663_, v_a_7664_, v_a_7665_, v_a_7666_, v_a_7667_);
lean_dec(v_a_7667_);
lean_dec_ref(v_a_7666_);
lean_dec(v_a_7665_);
lean_dec_ref(v_a_7664_);
lean_dec(v_a_7663_);
lean_dec(v_a_7662_);
lean_dec_ref(v_a_7661_);
return v_res_7669_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturate(lean_object* v_rs_7670_, lean_object* v_goal_7671_, lean_object* v_options_7672_, lean_object* v_a_7673_, lean_object* v_a_7674_, lean_object* v_a_7675_, lean_object* v_a_7676_){
_start:
{
lean_object* v___x_7678_; lean_object* v___x_7679_; 
v___x_7678_ = lean_alloc_closure((void*)(lp_aesop_Aesop_saturateMain___boxed), 10, 2);
lean_closure_set(v___x_7678_, 0, v_rs_7670_);
lean_closure_set(v___x_7678_, 1, v_goal_7671_);
v___x_7679_ = lp_aesop_Aesop_SaturateM_run___redArg(v_options_7672_, v___x_7678_, v_a_7673_, v_a_7674_, v_a_7675_, v_a_7676_);
return v___x_7679_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_saturate___boxed(lean_object* v_rs_7680_, lean_object* v_goal_7681_, lean_object* v_options_7682_, lean_object* v_a_7683_, lean_object* v_a_7684_, lean_object* v_a_7685_, lean_object* v_a_7686_, lean_object* v_a_7687_){
_start:
{
lean_object* v_res_7688_; 
v_res_7688_ = lp_aesop_Aesop_saturate(v_rs_7680_, v_goal_7681_, v_options_7682_, v_a_7683_, v_a_7684_, v_a_7685_, v_a_7686_);
lean_dec(v_a_7686_);
lean_dec_ref(v_a_7685_);
lean_dec(v_a_7684_);
lean_dec_ref(v_a_7683_);
return v_res_7688_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleSet(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_ScriptM(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_State_Initial(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_Expansion_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_Check(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Saturate(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_ScriptM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State_Initial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Expansion_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_BinomialHeap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_SaturateM_instInhabitedContext_default = _init_lp_aesop_Aesop_SaturateM_instInhabitedContext_default();
lean_mark_persistent(lp_aesop_Aesop_SaturateM_instInhabitedContext_default);
lp_aesop_Aesop_SaturateM_instInhabitedContext = _init_lp_aesop_Aesop_SaturateM_instInhabitedContext();
lean_mark_persistent(lp_aesop_Aesop_SaturateM_instInhabitedContext);
res = lp_aesop___private_Aesop_Saturate_0__Aesop_initFn_00___x40_Aesop_Saturate_2642905901____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Saturate(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleSet(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_ScriptM(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_State_Initial(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Expansion_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_Check(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Saturate(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_ScriptM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State_Initial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Expansion_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_BinomialHeap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Saturate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Saturate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Saturate(builtin);
}
#ifdef __cplusplus
}
#endif
