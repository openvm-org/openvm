// Lean compiler output
// Module: Aesop.Search.Expansion
// Imports: public import Init public meta import Init public import Aesop.Search.Expansion.Norm public import Aesop.Tree.AddRapp public import Aesop.Forward.State.UpdateGoal public import Aesop.RuleTac public import Aesop.Search.RuleSelection public import Batteries.Data.Array.Basic import Aesop.Search.Expansion.Basic
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
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_BaseM_instMonadStats;
lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(lean_object*);
lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* lp_aesop_Aesop_ruleProvedEmoji;
extern lean_object* lp_aesop_Aesop_ruleSuccessEmoji;
extern lean_object* lp_aesop_Aesop_ruleFailureEmoji;
extern lean_object* lp_aesop_Aesop_rulePostponedEmoji;
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instMonadMCtxMetaM;
lean_object* l_Lean_MVarId_isAssignedOrDelayedAssigned___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
double lean_float_of_nat(lean_object*);
lean_object* l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
double lean_float_div(double, double);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
double lean_float_mul(double, double);
lean_object* lp_aesop_Aesop_addRappUnsafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
double lp_aesop_Aesop_RegularRule_successProbability(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t);
lean_object* lp_aesop_Aesop_enqueueGoals___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RappRef_markProven(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RegularRule_tac(lean_object*);
lean_object* lp_aesop_Aesop_RuleTacDescr_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RegularRule_name(lean_object*);
lean_object* lp_aesop_Aesop_runRuleTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_steps;
lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_addTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t);
lean_object* l_instMonadExceptOfEIO(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptReaderT___redArg(lean_object*);
lean_object* l_Lean_instExceptToTraceResult___lam__0___boxed(lean_object*);
lean_object* lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
extern lean_object* l_Lean_KVMap_instValueString;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lp_batteries_Subarray_popHead_x3f___redArg(lean_object*);
lean_object* lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule(lean_object*);
extern lean_object* lp_aesop_Aesop_ruleErrorEmoji;
lean_object* lp_aesop_Aesop_GoalRef_updateForwardState(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_selectSafeRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* lp_aesop_Aesop_ruleSkippedEmoji;
lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_GoalState_isProven(uint8_t);
uint8_t lp_aesop_Aesop_Goal_isUnprovableNoCache(lean_object*);
lean_object* lp_aesop_Aesop_GoalRef_markUnprovable(lean_object*);
lean_object* lp_aesop_Aesop_getRootMetaState___redArg(lean_object*);
lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instExceptToTraceResultBool___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_proved_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_proved_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_succeeded_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_succeeded_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_failed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_failed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_toEmoji(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_toEmoji___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleResult_isSuccessful(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_isSuccessful___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_regular_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_regular_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_postponed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_postponed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_toEmoji(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_toEmoji___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_SafeRuleResult_isSuccessfulOrPostponed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_isSuccessfulOrPostponed___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_runRegularRuleTac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "aesop: internal error during expansion: expected goal "};
static const lean_object* lp_aesop_Aesop_runRegularRuleTac___closed__0 = (const lean_object*)&lp_aesop_Aesop_runRegularRuleTac___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_runRegularRuleTac___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runRegularRuleTac___closed__1;
static const lean_string_object lp_aesop_Aesop_runRegularRuleTac___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = " to be normalised (but not proven by normalisation)."};
static const lean_object* lp_aesop_Aesop_runRegularRuleTac___closed__2 = (const lean_object*)&lp_aesop_Aesop_runRegularRuleTac___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_runRegularRuleTac___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runRegularRuleTac___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleTac___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_addRapps___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_addRapps___redArg___lam__0___boxed, .m_arity = 12, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_addRapps___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_addRapps___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_addRapps___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_addRapps___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_addRapps___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_addRapps___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_addRapps___redArg___lam__3___boxed, .m_arity = 14, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_addRapps___redArg___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_addRapps___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_addRapps___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_addRapps___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_addRapps___redArg___lam__4___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_addRapps___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_addRapps___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__4_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__6_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__7_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__8_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__9_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__10_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__11_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__12_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__13_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__14 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__14_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__15 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_withRuleTraceNode___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__4;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__6;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__7;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__8;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__9;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__10;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__11;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__12;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__13;
static const lean_closure_object lp_aesop_Aesop_withRuleTraceNode___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__14 = (const lean_object*)&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__14_value;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__16;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__18;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__19;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__20;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21;
static const lean_closure_object lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instExceptToTraceResult___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22 = (const lean_object*)&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22_value;
static const lean_string_object lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23 = (const lean_object*)&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23_value;
static const lean_string_object lp_aesop_Aesop_withRuleTraceNode___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__24 = (const lean_object*)&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__24_value;
static const lean_ctor_object lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__24_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25 = (const lean_object*)&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25_value;
static lean_once_cell_t lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26;
LEAN_EXPORT lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withRuleTraceNode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withRuleTraceNode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_runRegularRuleCore___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_runRegularRuleCore___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2;
static const lean_string_object lp_aesop_Aesop_runRegularRuleCore___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Rule returned no rule applications"};
static const lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_runRegularRuleCore___redArg___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_runRegularRuleCore___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__0_value;
static const lean_string_object lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "Safe rule did not produce exactly one rule application"};
static const lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__1 = (const lean_object*)&lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2;
static const lean_string_object lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Safe rule assigned metavariables, so we postpone it"};
static const lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__3 = (const lean_object*)&lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2___boxed(lean_object**);
static lean_once_cell_t lp_aesop_Aesop_runSafeRule___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runSafeRule___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_runSafeRule___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_runSafeRule___redArg___closed__1;
static const lean_closure_object lp_aesop_Aesop_runSafeRule___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runSafeRule___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_runSafeRule___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_runSafeRule___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runSafeRule___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_runSafeRule___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_runSafeRule___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runSafeRule___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_runSafeRule___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_runSafeRule___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runSafeRule___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_runSafeRule___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_runSafeRule___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SafeRuleResult_toEmoji___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runSafeRule___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_runSafeRule___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_runUnsafeRule___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleResult_toEmoji___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_runUnsafeRule___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_runUnsafeRule___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_proved_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_proved_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_succeeded_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_succeeded_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_failed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_failed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_skipped_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_skipped_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_toEmoji(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_toEmoji___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_runFirstSafeRule___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_runFirstSafeRule___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_runFirstSafeRule___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_runFirstSafeRule___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_runFirstSafeRule___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_applyPostponedSafeRule___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " (postponed)"};
static const lean_object* lp_aesop_Aesop_applyPostponedSafeRule___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_applyPostponedSafeRule___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyPostponedSafeRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyPostponedSafeRule___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyPostponedSafeRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyPostponedSafeRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstUnsafeRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstUnsafeRule___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstUnsafeRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstUnsafeRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = " Unsafe rules"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = " Normalisation"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__1;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__2;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__3;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__4;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__5;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__6;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SafeRulesResult_toEmoji___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " Safe rules"};
static const lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_expandGoal___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_expandGoal___redArg___closed__0_value;
static const lean_array_object lp_aesop_Aesop_expandGoal___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_expandGoal___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_expandGoal___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_expandGoal___redArg___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_expandGoal___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_expandGoal___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_expandGoal___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_expandGoal___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_expandGoal___redArg___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_expandGoal___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_expandGoal___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__6;
static const lean_string_object lp_aesop_Aesop_expandGoal___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Goal after normalisation:"};
static const lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_expandGoal___redArg___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_expandGoal___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__8;
static const lean_closure_object lp_aesop_Aesop_expandGoal___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instExceptToTraceResultBool___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_expandGoal___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_expandGoal___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandGoal___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
default: 
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_aesop_Aesop_RuleResult_ctorIdx(v_x_5_);
lean_dec(v_x_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorElim___redArg(lean_object* v_t_7_, lean_object* v_k_8_){
_start:
{
if (lean_obj_tag(v_t_7_) == 2)
{
return v_k_8_;
}
else
{
lean_object* v_newRapps_9_; lean_object* v___x_10_; 
v_newRapps_9_ = lean_ctor_get(v_t_7_, 0);
lean_inc_ref(v_newRapps_9_);
lean_dec(v_t_7_);
v___x_10_ = lean_apply_1(v_k_8_, v_newRapps_9_);
return v___x_10_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorElim(lean_object* v_motive_11_, lean_object* v_ctorIdx_12_, lean_object* v_t_13_, lean_object* v_h_14_, lean_object* v_k_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_aesop_Aesop_RuleResult_ctorElim___redArg(v_t_13_, v_k_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_ctorElim___boxed(lean_object* v_motive_17_, lean_object* v_ctorIdx_18_, lean_object* v_t_19_, lean_object* v_h_20_, lean_object* v_k_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_aesop_Aesop_RuleResult_ctorElim(v_motive_17_, v_ctorIdx_18_, v_t_19_, v_h_20_, v_k_21_);
lean_dec(v_ctorIdx_18_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_proved_elim___redArg(lean_object* v_t_23_, lean_object* v_proved_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_aesop_Aesop_RuleResult_ctorElim___redArg(v_t_23_, v_proved_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_proved_elim(lean_object* v_motive_26_, lean_object* v_t_27_, lean_object* v_h_28_, lean_object* v_proved_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_aesop_Aesop_RuleResult_ctorElim___redArg(v_t_27_, v_proved_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_succeeded_elim___redArg(lean_object* v_t_31_, lean_object* v_succeeded_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_aesop_Aesop_RuleResult_ctorElim___redArg(v_t_31_, v_succeeded_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_succeeded_elim(lean_object* v_motive_34_, lean_object* v_t_35_, lean_object* v_h_36_, lean_object* v_succeeded_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_aesop_Aesop_RuleResult_ctorElim___redArg(v_t_35_, v_succeeded_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_failed_elim___redArg(lean_object* v_t_39_, lean_object* v_failed_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_aesop_Aesop_RuleResult_ctorElim___redArg(v_t_39_, v_failed_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_failed_elim(lean_object* v_motive_42_, lean_object* v_t_43_, lean_object* v_h_44_, lean_object* v_failed_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_aesop_Aesop_RuleResult_ctorElim___redArg(v_t_43_, v_failed_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_toEmoji(lean_object* v_x_47_){
_start:
{
switch(lean_obj_tag(v_x_47_))
{
case 0:
{
lean_object* v___x_48_; 
v___x_48_ = lp_aesop_Aesop_ruleProvedEmoji;
return v___x_48_;
}
case 1:
{
lean_object* v___x_49_; 
v___x_49_ = lp_aesop_Aesop_ruleSuccessEmoji;
return v___x_49_;
}
default: 
{
lean_object* v___x_50_; 
v___x_50_ = lp_aesop_Aesop_ruleFailureEmoji;
return v___x_50_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_toEmoji___boxed(lean_object* v_x_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_aesop_Aesop_RuleResult_toEmoji(v_x_51_);
lean_dec(v_x_51_);
return v_res_52_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleResult_isSuccessful(lean_object* v_x_53_){
_start:
{
if (lean_obj_tag(v_x_53_) == 2)
{
uint8_t v___x_54_; 
v___x_54_ = 0;
return v___x_54_;
}
else
{
uint8_t v___x_55_; 
v___x_55_ = 1;
return v___x_55_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleResult_isSuccessful___boxed(lean_object* v_x_56_){
_start:
{
uint8_t v_res_57_; lean_object* v_r_58_; 
v_res_57_ = lp_aesop_Aesop_RuleResult_isSuccessful(v_x_56_);
lean_dec(v_x_56_);
v_r_58_ = lean_box(v_res_57_);
return v_r_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorIdx(lean_object* v_x_59_){
_start:
{
if (lean_obj_tag(v_x_59_) == 0)
{
lean_object* v___x_60_; 
v___x_60_ = lean_unsigned_to_nat(0u);
return v___x_60_;
}
else
{
lean_object* v___x_61_; 
v___x_61_ = lean_unsigned_to_nat(1u);
return v___x_61_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorIdx___boxed(lean_object* v_x_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_aesop_Aesop_SafeRuleResult_ctorIdx(v_x_62_);
lean_dec_ref(v_x_62_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorElim___redArg(lean_object* v_t_64_, lean_object* v_k_65_){
_start:
{
if (lean_obj_tag(v_t_64_) == 0)
{
lean_object* v_result_66_; lean_object* v___x_67_; 
v_result_66_ = lean_ctor_get(v_t_64_, 0);
lean_inc(v_result_66_);
lean_dec_ref_known(v_t_64_, 1);
v___x_67_ = lean_apply_1(v_k_65_, v_result_66_);
return v___x_67_;
}
else
{
lean_object* v_result_68_; lean_object* v___x_69_; 
v_result_68_ = lean_ctor_get(v_t_64_, 0);
lean_inc_ref(v_result_68_);
lean_dec_ref_known(v_t_64_, 1);
v___x_69_ = lean_apply_1(v_k_65_, v_result_68_);
return v___x_69_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorElim(lean_object* v_motive_70_, lean_object* v_ctorIdx_71_, lean_object* v_t_72_, lean_object* v_h_73_, lean_object* v_k_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_aesop_Aesop_SafeRuleResult_ctorElim___redArg(v_t_72_, v_k_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_ctorElim___boxed(lean_object* v_motive_76_, lean_object* v_ctorIdx_77_, lean_object* v_t_78_, lean_object* v_h_79_, lean_object* v_k_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_aesop_Aesop_SafeRuleResult_ctorElim(v_motive_76_, v_ctorIdx_77_, v_t_78_, v_h_79_, v_k_80_);
lean_dec(v_ctorIdx_77_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_regular_elim___redArg(lean_object* v_t_82_, lean_object* v_regular_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_aesop_Aesop_SafeRuleResult_ctorElim___redArg(v_t_82_, v_regular_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_regular_elim(lean_object* v_motive_85_, lean_object* v_t_86_, lean_object* v_h_87_, lean_object* v_regular_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_aesop_Aesop_SafeRuleResult_ctorElim___redArg(v_t_86_, v_regular_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_postponed_elim___redArg(lean_object* v_t_90_, lean_object* v_postponed_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_aesop_Aesop_SafeRuleResult_ctorElim___redArg(v_t_90_, v_postponed_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_postponed_elim(lean_object* v_motive_93_, lean_object* v_t_94_, lean_object* v_h_95_, lean_object* v_postponed_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lp_aesop_Aesop_SafeRuleResult_ctorElim___redArg(v_t_94_, v_postponed_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_toEmoji(lean_object* v_x_98_){
_start:
{
if (lean_obj_tag(v_x_98_) == 0)
{
lean_object* v_result_99_; lean_object* v___x_100_; 
v_result_99_ = lean_ctor_get(v_x_98_, 0);
v___x_100_ = lp_aesop_Aesop_RuleResult_toEmoji(v_result_99_);
return v___x_100_;
}
else
{
lean_object* v___x_101_; 
v___x_101_ = lp_aesop_Aesop_rulePostponedEmoji;
return v___x_101_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_toEmoji___boxed(lean_object* v_x_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_aesop_Aesop_SafeRuleResult_toEmoji(v_x_102_);
lean_dec_ref(v_x_102_);
return v_res_103_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_SafeRuleResult_isSuccessfulOrPostponed(lean_object* v_x_104_){
_start:
{
if (lean_obj_tag(v_x_104_) == 0)
{
lean_object* v_result_105_; uint8_t v___x_106_; 
v_result_105_ = lean_ctor_get(v_x_104_, 0);
v___x_106_ = lp_aesop_Aesop_RuleResult_isSuccessful(v_result_105_);
return v___x_106_;
}
else
{
uint8_t v___x_107_; 
v___x_107_ = 1;
return v___x_107_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRuleResult_isSuccessfulOrPostponed___boxed(lean_object* v_x_108_){
_start:
{
uint8_t v_res_109_; lean_object* v_r_110_; 
v_res_109_ = lp_aesop_Aesop_SafeRuleResult_isSuccessfulOrPostponed(v_x_108_);
lean_dec_ref(v_x_108_);
v_r_110_ = lean_box(v_res_109_);
return v_r_110_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0_spec__0(lean_object* v_msgData_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_){
_start:
{
lean_object* v___x_117_; lean_object* v_env_118_; lean_object* v___x_119_; lean_object* v_mctx_120_; lean_object* v_lctx_121_; lean_object* v_options_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_117_ = lean_st_ref_get(v___y_115_);
v_env_118_ = lean_ctor_get(v___x_117_, 0);
lean_inc_ref(v_env_118_);
lean_dec(v___x_117_);
v___x_119_ = lean_st_ref_get(v___y_113_);
v_mctx_120_ = lean_ctor_get(v___x_119_, 0);
lean_inc_ref(v_mctx_120_);
lean_dec(v___x_119_);
v_lctx_121_ = lean_ctor_get(v___y_112_, 2);
v_options_122_ = lean_ctor_get(v___y_114_, 2);
lean_inc_ref(v_options_122_);
lean_inc_ref(v_lctx_121_);
v___x_123_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_123_, 0, v_env_118_);
lean_ctor_set(v___x_123_, 1, v_mctx_120_);
lean_ctor_set(v___x_123_, 2, v_lctx_121_);
lean_ctor_set(v___x_123_, 3, v_options_122_);
v___x_124_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
lean_ctor_set(v___x_124_, 1, v_msgData_111_);
v___x_125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0_spec__0___boxed(lean_object* v_msgData_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0_spec__0(v_msgData_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_);
lean_dec(v___y_130_);
lean_dec_ref(v___y_129_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___redArg(lean_object* v_msg_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v_ref_139_; lean_object* v___x_140_; lean_object* v_a_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_149_; 
v_ref_139_ = lean_ctor_get(v___y_136_, 5);
v___x_140_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0_spec__0(v_msg_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
v_a_141_ = lean_ctor_get(v___x_140_, 0);
v_isSharedCheck_149_ = !lean_is_exclusive(v___x_140_);
if (v_isSharedCheck_149_ == 0)
{
v___x_143_ = v___x_140_;
v_isShared_144_ = v_isSharedCheck_149_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_a_141_);
lean_dec(v___x_140_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_149_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v___x_145_; lean_object* v___x_147_; 
lean_inc(v_ref_139_);
v___x_145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_145_, 0, v_ref_139_);
lean_ctor_set(v___x_145_, 1, v_a_141_);
if (v_isShared_144_ == 0)
{
lean_ctor_set_tag(v___x_143_, 1);
lean_ctor_set(v___x_143_, 0, v___x_145_);
v___x_147_ = v___x_143_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_145_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
return v___x_147_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___redArg___boxed(lean_object* v_msg_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___redArg(v_msg_150_, v___y_151_, v___y_152_, v___y_153_, v___y_154_);
lean_dec(v___y_154_);
lean_dec_ref(v___y_153_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
return v_res_156_;
}
}
static lean_object* _init_lp_aesop_Aesop_runRegularRuleTac___closed__1(void){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_158_ = ((lean_object*)(lp_aesop_Aesop_runRegularRuleTac___closed__0));
v___x_159_ = l_Lean_stringToMessageData(v___x_158_);
return v___x_159_;
}
}
static lean_object* _init_lp_aesop_Aesop_runRegularRuleTac___closed__3(void){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_161_ = ((lean_object*)(lp_aesop_Aesop_runRegularRuleTac___closed__2));
v___x_162_ = l_Lean_stringToMessageData(v___x_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleTac(lean_object* v_goal_163_, lean_object* v_tac_164_, lean_object* v_ruleName_165_, lean_object* v_indexMatchLocations_166_, lean_object* v_patternSubsts_x3f_167_, lean_object* v_options_168_, lean_object* v_a_169_, lean_object* v_a_170_, lean_object* v_a_171_, lean_object* v_a_172_, lean_object* v_a_173_){
_start:
{
lean_object* v___x_175_; lean_object* v_elimGoal_176_; lean_object* v___x_177_; lean_object* v_normalizationState_178_; 
v___x_175_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_176_ = lean_ctor_get(v___x_175_, 1);
lean_inc_ref(v_elimGoal_176_);
v___x_177_ = lean_apply_1(v_elimGoal_176_, v_goal_163_);
v_normalizationState_178_ = lean_ctor_get(v___x_177_, 6);
lean_inc(v_normalizationState_178_);
if (lean_obj_tag(v_normalizationState_178_) == 1)
{
lean_object* v_mvars_179_; lean_object* v_postGoal_180_; lean_object* v_postState_181_; lean_object* v_input_182_; lean_object* v___x_183_; 
v_mvars_179_ = lean_ctor_get(v___x_177_, 7);
lean_inc_ref(v_mvars_179_);
lean_dec_ref(v___x_177_);
v_postGoal_180_ = lean_ctor_get(v_normalizationState_178_, 0);
lean_inc(v_postGoal_180_);
v_postState_181_ = lean_ctor_get(v_normalizationState_178_, 1);
lean_inc_ref(v_postState_181_);
lean_dec_ref_known(v_normalizationState_178_, 3);
v_input_182_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_input_182_, 0, v_postGoal_180_);
lean_ctor_set(v_input_182_, 1, v_mvars_179_);
lean_ctor_set(v_input_182_, 2, v_indexMatchLocations_166_);
lean_ctor_set(v_input_182_, 3, v_patternSubsts_x3f_167_);
lean_ctor_set(v_input_182_, 4, v_options_168_);
v___x_183_ = lp_aesop_Aesop_runRuleTac(v_tac_164_, v_ruleName_165_, v_postState_181_, v_input_182_, v_a_169_, v_a_170_, v_a_171_, v_a_172_, v_a_173_);
lean_dec_ref(v_postState_181_);
return v___x_183_;
}
else
{
lean_object* v_id_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
lean_dec(v_normalizationState_178_);
lean_dec_ref(v_options_168_);
lean_dec(v_patternSubsts_x3f_167_);
lean_dec_ref(v_indexMatchLocations_166_);
lean_dec_ref(v_ruleName_165_);
lean_dec_ref(v_tac_164_);
v_id_184_ = lean_ctor_get(v___x_177_, 0);
lean_inc(v_id_184_);
lean_dec_ref(v___x_177_);
v___x_185_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleTac___closed__1, &lp_aesop_Aesop_runRegularRuleTac___closed__1_once, _init_lp_aesop_Aesop_runRegularRuleTac___closed__1);
v___x_186_ = l_Nat_reprFast(v_id_184_);
v___x_187_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_187_, 0, v___x_186_);
v___x_188_ = l_Lean_MessageData_ofFormat(v___x_187_);
v___x_189_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_185_);
lean_ctor_set(v___x_189_, 1, v___x_188_);
v___x_190_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleTac___closed__3, &lp_aesop_Aesop_runRegularRuleTac___closed__3_once, _init_lp_aesop_Aesop_runRegularRuleTac___closed__3);
v___x_191_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_189_);
lean_ctor_set(v___x_191_, 1, v___x_190_);
v___x_192_ = lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___redArg(v___x_191_, v_a_170_, v_a_171_, v_a_172_, v_a_173_);
return v___x_192_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleTac___boxed(lean_object* v_goal_193_, lean_object* v_tac_194_, lean_object* v_ruleName_195_, lean_object* v_indexMatchLocations_196_, lean_object* v_patternSubsts_x3f_197_, lean_object* v_options_198_, lean_object* v_a_199_, lean_object* v_a_200_, lean_object* v_a_201_, lean_object* v_a_202_, lean_object* v_a_203_, lean_object* v_a_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_aesop_Aesop_runRegularRuleTac(v_goal_193_, v_tac_194_, v_ruleName_195_, v_indexMatchLocations_196_, v_patternSubsts_x3f_197_, v_options_198_, v_a_199_, v_a_200_, v_a_201_, v_a_202_, v_a_203_);
lean_dec(v_a_203_);
lean_dec_ref(v_a_202_);
lean_dec(v_a_201_);
lean_dec_ref(v_a_200_);
lean_dec(v_a_199_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0(lean_object* v_00_u03b1_206_, lean_object* v_msg_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___redArg(v_msg_207_, v___y_209_, v___y_210_, v___y_211_, v___y_212_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0___boxed(lean_object* v_00_u03b1_215_, lean_object* v_msg_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_aesop_Lean_throwError___at___00Aesop_runRegularRuleTac_spec__0(v_00_u03b1_215_, v_msg_216_, v___y_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
lean_dec(v___y_219_);
lean_dec_ref(v___y_218_);
lean_dec(v___y_217_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__0(lean_object* v_a_224_, lean_object* v_x_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_236_ = lean_array_push(v___y_226_, v_a_224_);
v___x_237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
v___x_238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_238_, 0, v___x_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__0___boxed(lean_object* v_a_239_, lean_object* v_x_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_aesop_Aesop_addRapps___redArg___lam__0(v_a_239_, v_x_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_, v___y_247_, v___y_248_, v___y_249_);
lean_dec(v___y_249_);
lean_dec_ref(v___y_248_);
lean_dec(v___y_247_);
lean_dec_ref(v___y_246_);
lean_dec(v___y_245_);
lean_dec(v___y_244_);
lean_dec(v___y_243_);
lean_dec_ref(v___y_242_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__1(lean_object* v___x_252_, lean_object* v___f_253_, lean_object* v_a_254_, lean_object* v_x_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_){
_start:
{
lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v_elimMVarCluster_269_; lean_object* v___x_270_; lean_object* v_goals_271_; size_t v_sz_272_; size_t v___x_273_; lean_object* v___x_19925__overap_274_; lean_object* v___x_275_; 
v___x_266_ = lean_st_ref_get(v___y_258_);
lean_dec(v___x_266_);
v___x_267_ = lean_st_ref_get(v_a_254_);
v___x_268_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_269_ = lean_ctor_get(v___x_268_, 5);
lean_inc_ref(v_elimMVarCluster_269_);
v___x_270_ = lean_apply_1(v_elimMVarCluster_269_, v___x_267_);
v_goals_271_ = lean_ctor_get(v___x_270_, 1);
lean_inc_ref(v_goals_271_);
lean_dec_ref(v___x_270_);
v_sz_272_ = lean_array_size(v_goals_271_);
v___x_273_ = ((size_t)0ULL);
v___x_19925__overap_274_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_252_, v_goals_271_, v___f_253_, v_sz_272_, v___x_273_, v___y_256_);
lean_inc(v___y_264_);
lean_inc_ref(v___y_263_);
lean_inc(v___y_262_);
lean_inc_ref(v___y_261_);
lean_inc(v___y_260_);
lean_inc(v___y_259_);
lean_inc(v___y_258_);
lean_inc_ref(v___y_257_);
v___x_275_ = lean_apply_9(v___x_19925__overap_274_, v___y_257_, v___y_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_, lean_box(0));
if (lean_obj_tag(v___x_275_) == 0)
{
lean_object* v_a_276_; lean_object* v___x_278_; uint8_t v_isShared_279_; uint8_t v_isSharedCheck_284_; 
v_a_276_ = lean_ctor_get(v___x_275_, 0);
v_isSharedCheck_284_ = !lean_is_exclusive(v___x_275_);
if (v_isSharedCheck_284_ == 0)
{
v___x_278_ = v___x_275_;
v_isShared_279_ = v_isSharedCheck_284_;
goto v_resetjp_277_;
}
else
{
lean_inc(v_a_276_);
lean_dec(v___x_275_);
v___x_278_ = lean_box(0);
v_isShared_279_ = v_isSharedCheck_284_;
goto v_resetjp_277_;
}
v_resetjp_277_:
{
lean_object* v___x_280_; lean_object* v___x_282_; 
v___x_280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_280_, 0, v_a_276_);
if (v_isShared_279_ == 0)
{
lean_ctor_set(v___x_278_, 0, v___x_280_);
v___x_282_ = v___x_278_;
goto v_reusejp_281_;
}
else
{
lean_object* v_reuseFailAlloc_283_; 
v_reuseFailAlloc_283_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_283_, 0, v___x_280_);
v___x_282_ = v_reuseFailAlloc_283_;
goto v_reusejp_281_;
}
v_reusejp_281_:
{
return v___x_282_;
}
}
}
else
{
lean_object* v_a_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_292_; 
v_a_285_ = lean_ctor_get(v___x_275_, 0);
v_isSharedCheck_292_ = !lean_is_exclusive(v___x_275_);
if (v_isSharedCheck_292_ == 0)
{
v___x_287_ = v___x_275_;
v_isShared_288_ = v_isSharedCheck_292_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_a_285_);
lean_dec(v___x_275_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_292_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v___x_290_; 
if (v_isShared_288_ == 0)
{
v___x_290_ = v___x_287_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v_a_285_);
v___x_290_ = v_reuseFailAlloc_291_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
return v___x_290_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__1___boxed(lean_object* v___x_293_, lean_object* v___f_294_, lean_object* v_a_295_, lean_object* v_x_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_aesop_Aesop_addRapps___redArg___lam__1(v___x_293_, v___f_294_, v_a_295_, v_x_296_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
lean_dec(v___y_301_);
lean_dec(v___y_300_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
lean_dec(v_a_295_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__2(lean_object* v_val_308_, lean_object* v_rapps_309_, lean_object* v_parentRef_310_, lean_object* v_rule_311_, lean_object* v___x_312_, lean_object* v___f_313_, lean_object* v_i_314_, lean_object* v_h_315_, lean_object* v_____s_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_){
_start:
{
lean_object* v_fst_326_; lean_object* v_snd_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_388_; 
v_fst_326_ = lean_ctor_get(v_____s_316_, 0);
v_snd_327_ = lean_ctor_get(v_____s_316_, 1);
v_isSharedCheck_388_ = !lean_is_exclusive(v_____s_316_);
if (v_isSharedCheck_388_ == 0)
{
v___x_329_ = v_____s_316_;
v_isShared_330_ = v_isSharedCheck_388_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_snd_327_);
lean_inc(v_fst_326_);
lean_dec(v_____s_316_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_388_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v___x_331_; lean_object* v_elimGoal_332_; lean_object* v_elimRapp_333_; lean_object* v___x_334_; double v_successProbability_335_; lean_object* v___x_336_; double v___y_338_; lean_object* v_successProbability_x3f_384_; 
v___x_331_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_332_ = lean_ctor_get(v___x_331_, 1);
v_elimRapp_333_ = lean_ctor_get(v___x_331_, 3);
lean_inc_ref(v_elimGoal_332_);
v___x_334_ = lean_apply_1(v_elimGoal_332_, v_val_308_);
v_successProbability_335_ = lean_ctor_get_float(v___x_334_, sizeof(void*)*14);
lean_dec_ref(v___x_334_);
v___x_336_ = lean_array_fget_borrowed(v_rapps_309_, v_i_314_);
v_successProbability_x3f_384_ = lean_ctor_get(v___x_336_, 3);
if (lean_obj_tag(v_successProbability_x3f_384_) == 0)
{
double v___x_385_; 
v___x_385_ = lp_aesop_Aesop_RegularRule_successProbability(v_rule_311_);
v___y_338_ = v___x_385_;
goto v___jp_337_;
}
else
{
lean_object* v_val_386_; double v___x_387_; 
v_val_386_ = lean_ctor_get(v_successProbability_x3f_384_, 0);
v___x_387_ = lean_unbox_float(v_val_386_);
v___y_338_ = v___x_387_;
goto v___jp_337_;
}
v___jp_337_:
{
lean_object* v___x_339_; lean_object* v_iteration_340_; lean_object* v_ruleSet_341_; double v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_339_ = lean_st_ref_get(v___y_318_);
v_iteration_340_ = lean_ctor_get(v___x_339_, 0);
lean_inc(v_iteration_340_);
lean_dec(v___x_339_);
v_ruleSet_341_ = lean_ctor_get(v___y_317_, 0);
v___x_342_ = lean_float_mul(v_successProbability_335_, v___y_338_);
lean_inc(v___x_336_);
v___x_343_ = lean_alloc_ctor(0, 3, 8);
lean_ctor_set(v___x_343_, 0, v___x_336_);
lean_ctor_set(v___x_343_, 1, v_parentRef_310_);
lean_ctor_set(v___x_343_, 2, v_rule_311_);
lean_ctor_set_float(v___x_343_, sizeof(void*)*3, v___x_342_);
lean_inc_ref(v_ruleSet_341_);
v___x_344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_344_, 0, v_iteration_340_);
lean_ctor_set(v___x_344_, 1, v_ruleSet_341_);
v___x_345_ = lp_aesop_Aesop_addRappUnsafe(v___x_343_, v___x_344_, v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
lean_dec_ref_known(v___x_344_, 2);
if (lean_obj_tag(v___x_345_) == 0)
{
lean_object* v_a_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v_children_350_; size_t v_sz_351_; size_t v___x_352_; lean_object* v___x_19968__overap_353_; lean_object* v___x_354_; 
v_a_346_ = lean_ctor_get(v___x_345_, 0);
lean_inc(v_a_346_);
lean_dec_ref_known(v___x_345_, 1);
v___x_347_ = lean_st_ref_get(v___y_318_);
lean_dec(v___x_347_);
v___x_348_ = lean_st_ref_get(v_a_346_);
lean_inc_ref(v_elimRapp_333_);
v___x_349_ = lean_apply_1(v_elimRapp_333_, v___x_348_);
v_children_350_ = lean_ctor_get(v___x_349_, 2);
lean_inc_ref(v_children_350_);
lean_dec_ref(v___x_349_);
v_sz_351_ = lean_array_size(v_children_350_);
v___x_352_ = ((size_t)0ULL);
v___x_19968__overap_353_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_312_, v_children_350_, v___f_313_, v_sz_351_, v___x_352_, v_snd_327_);
lean_inc(v___y_324_);
lean_inc_ref(v___y_323_);
lean_inc(v___y_322_);
lean_inc_ref(v___y_321_);
lean_inc(v___y_320_);
lean_inc(v___y_319_);
lean_inc(v___y_318_);
lean_inc_ref(v___y_317_);
v___x_354_ = lean_apply_9(v___x_19968__overap_353_, v___y_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, lean_box(0));
if (lean_obj_tag(v___x_354_) == 0)
{
lean_object* v_a_355_; lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_367_; 
v_a_355_ = lean_ctor_get(v___x_354_, 0);
v_isSharedCheck_367_ = !lean_is_exclusive(v___x_354_);
if (v_isSharedCheck_367_ == 0)
{
v___x_357_ = v___x_354_;
v_isShared_358_ = v_isSharedCheck_367_;
goto v_resetjp_356_;
}
else
{
lean_inc(v_a_355_);
lean_dec(v___x_354_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_367_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v___x_359_; lean_object* v___x_361_; 
v___x_359_ = lean_array_push(v_fst_326_, v_a_346_);
if (v_isShared_330_ == 0)
{
lean_ctor_set(v___x_329_, 1, v_a_355_);
lean_ctor_set(v___x_329_, 0, v___x_359_);
v___x_361_ = v___x_329_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v___x_359_);
lean_ctor_set(v_reuseFailAlloc_366_, 1, v_a_355_);
v___x_361_ = v_reuseFailAlloc_366_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
lean_object* v___x_362_; lean_object* v___x_364_; 
v___x_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_362_, 0, v___x_361_);
if (v_isShared_358_ == 0)
{
lean_ctor_set(v___x_357_, 0, v___x_362_);
v___x_364_ = v___x_357_;
goto v_reusejp_363_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v___x_362_);
v___x_364_ = v_reuseFailAlloc_365_;
goto v_reusejp_363_;
}
v_reusejp_363_:
{
return v___x_364_;
}
}
}
}
else
{
lean_object* v_a_368_; lean_object* v___x_370_; uint8_t v_isShared_371_; uint8_t v_isSharedCheck_375_; 
lean_dec(v_a_346_);
lean_del_object(v___x_329_);
lean_dec(v_fst_326_);
v_a_368_ = lean_ctor_get(v___x_354_, 0);
v_isSharedCheck_375_ = !lean_is_exclusive(v___x_354_);
if (v_isSharedCheck_375_ == 0)
{
v___x_370_ = v___x_354_;
v_isShared_371_ = v_isSharedCheck_375_;
goto v_resetjp_369_;
}
else
{
lean_inc(v_a_368_);
lean_dec(v___x_354_);
v___x_370_ = lean_box(0);
v_isShared_371_ = v_isSharedCheck_375_;
goto v_resetjp_369_;
}
v_resetjp_369_:
{
lean_object* v___x_373_; 
if (v_isShared_371_ == 0)
{
v___x_373_ = v___x_370_;
goto v_reusejp_372_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v_a_368_);
v___x_373_ = v_reuseFailAlloc_374_;
goto v_reusejp_372_;
}
v_reusejp_372_:
{
return v___x_373_;
}
}
}
}
else
{
lean_object* v_a_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_383_; 
lean_del_object(v___x_329_);
lean_dec(v_snd_327_);
lean_dec(v_fst_326_);
lean_dec_ref(v___f_313_);
lean_dec_ref(v___x_312_);
v_a_376_ = lean_ctor_get(v___x_345_, 0);
v_isSharedCheck_383_ = !lean_is_exclusive(v___x_345_);
if (v_isSharedCheck_383_ == 0)
{
v___x_378_ = v___x_345_;
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_a_376_);
lean_dec(v___x_345_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_381_; 
if (v_isShared_379_ == 0)
{
v___x_381_ = v___x_378_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_a_376_);
v___x_381_ = v_reuseFailAlloc_382_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
return v___x_381_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__2___boxed(lean_object** _args){
lean_object* v_val_389_ = _args[0];
lean_object* v_rapps_390_ = _args[1];
lean_object* v_parentRef_391_ = _args[2];
lean_object* v_rule_392_ = _args[3];
lean_object* v___x_393_ = _args[4];
lean_object* v___f_394_ = _args[5];
lean_object* v_i_395_ = _args[6];
lean_object* v_h_396_ = _args[7];
lean_object* v_____s_397_ = _args[8];
lean_object* v___y_398_ = _args[9];
lean_object* v___y_399_ = _args[10];
lean_object* v___y_400_ = _args[11];
lean_object* v___y_401_ = _args[12];
lean_object* v___y_402_ = _args[13];
lean_object* v___y_403_ = _args[14];
lean_object* v___y_404_ = _args[15];
lean_object* v___y_405_ = _args[16];
lean_object* v___y_406_ = _args[17];
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_aesop_Aesop_addRapps___redArg___lam__2(v_val_389_, v_rapps_390_, v_parentRef_391_, v_rule_392_, v___x_393_, v___f_394_, v_i_395_, v_h_396_, v_____s_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_, v___y_405_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
lean_dec(v___y_401_);
lean_dec(v___y_400_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
lean_dec(v_i_395_);
lean_dec_ref(v_rapps_390_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__3(lean_object* v___x_408_, lean_object* v___x_409_, lean_object* v_a_410_, lean_object* v_x_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_){
_start:
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v_elimRapp_425_; lean_object* v___x_426_; uint8_t v_state_427_; uint8_t v___x_428_; 
v___x_422_ = lean_st_ref_get(v___y_414_);
lean_dec(v___x_422_);
v___x_423_ = lean_st_ref_get(v_a_410_);
v___x_424_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_425_ = lean_ctor_get(v___x_424_, 3);
lean_inc_ref(v_elimRapp_425_);
v___x_426_ = lean_apply_1(v_elimRapp_425_, v___x_423_);
v_state_427_ = lean_ctor_get_uint8(v___x_426_, sizeof(void*)*9 + 8);
lean_dec_ref(v___x_426_);
v___x_428_ = lp_aesop_Aesop_NodeState_isProven(v_state_427_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; lean_object* v___x_430_; 
lean_dec(v_a_410_);
v___x_429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_429_, 0, v___x_408_);
v___x_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_430_, 0, v___x_429_);
return v___x_430_;
}
else
{
lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; 
lean_dec_ref(v___x_408_);
v___x_431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_431_, 0, v_a_410_);
v___x_432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_432_, 0, v___x_431_);
v___x_433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_433_, 0, v___x_432_);
lean_ctor_set(v___x_433_, 1, v___x_409_);
v___x_434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_434_, 0, v___x_433_);
v___x_435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_435_, 0, v___x_434_);
return v___x_435_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__3___boxed(lean_object* v___x_436_, lean_object* v___x_437_, lean_object* v_a_438_, lean_object* v_x_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_aesop_Aesop_addRapps___redArg___lam__3(v___x_436_, v___x_437_, v_a_438_, v_x_439_, v___y_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_, v___y_448_);
lean_dec(v___y_448_);
lean_dec_ref(v___y_447_);
lean_dec(v___y_446_);
lean_dec_ref(v___y_445_);
lean_dec(v___y_444_);
lean_dec(v___y_443_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
lean_dec_ref(v___y_440_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__4(lean_object* v_x_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; 
v___x_462_ = lean_st_ref_get(v___y_454_);
lean_dec(v___x_462_);
v___x_463_ = lp_aesop_Aesop_RappRef_markProven(v___y_452_);
v___x_464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_464_, 0, v___x_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___lam__4___boxed(lean_object* v_x_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_aesop_Aesop_addRapps___redArg___lam__4(v_x_465_, v___y_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_, v___y_471_, v___y_472_, v___y_473_, v___y_474_);
lean_dec(v___y_474_);
lean_dec_ref(v___y_473_);
lean_dec(v___y_472_);
lean_dec_ref(v___y_471_);
lean_dec(v___y_470_);
lean_dec(v___y_469_);
lean_dec(v___y_468_);
lean_dec_ref(v___y_467_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg(lean_object* v_inst_485_, lean_object* v_parentRef_486_, lean_object* v_rule_487_, lean_object* v_rapps_488_, lean_object* v_a_489_, lean_object* v_a_490_, lean_object* v_a_491_, lean_object* v_a_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___f_501_; lean_object* v___f_502_; lean_object* v___f_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_19752__overap_513_; lean_object* v___x_514_; 
v___x_498_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_485_);
v___x_499_ = lean_st_ref_get(v_a_490_);
lean_dec(v___x_499_);
v___x_500_ = lean_st_ref_get(v_parentRef_486_);
v___f_501_ = ((lean_object*)(lp_aesop_Aesop_addRapps___redArg___closed__0));
lean_inc_ref_n(v___x_498_, 3);
v___f_502_ = lean_alloc_closure((void*)(lp_aesop_Aesop_addRapps___redArg___lam__1___boxed), 14, 2);
lean_closure_set(v___f_502_, 0, v___x_498_);
lean_closure_set(v___f_502_, 1, v___f_501_);
lean_inc_ref(v_rapps_488_);
v___f_503_ = lean_alloc_closure((void*)(lp_aesop_Aesop_addRapps___redArg___lam__2___boxed), 18, 6);
lean_closure_set(v___f_503_, 0, v___x_500_);
lean_closure_set(v___f_503_, 1, v_rapps_488_);
lean_closure_set(v___f_503_, 2, v_parentRef_486_);
lean_closure_set(v___f_503_, 3, v_rule_487_);
lean_closure_set(v___f_503_, 4, v___x_498_);
lean_closure_set(v___f_503_, 5, v___f_502_);
v___x_504_ = lean_array_get_size(v_rapps_488_);
lean_dec_ref(v_rapps_488_);
v___x_505_ = lean_mk_empty_array_with_capacity(v___x_504_);
v___x_506_ = lean_unsigned_to_nat(3u);
v___x_507_ = lean_nat_mul(v___x_504_, v___x_506_);
v___x_508_ = lean_mk_empty_array_with_capacity(v___x_507_);
lean_dec(v___x_507_);
v___x_509_ = lean_unsigned_to_nat(0u);
v___x_510_ = lean_unsigned_to_nat(1u);
v___x_511_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_511_, 0, v___x_509_);
lean_ctor_set(v___x_511_, 1, v___x_504_);
lean_ctor_set(v___x_511_, 2, v___x_510_);
v___x_512_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_512_, 0, v___x_505_);
lean_ctor_set(v___x_512_, 1, v___x_508_);
v___x_19752__overap_513_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_498_, v___x_511_, v___f_503_, v___x_512_, v___x_509_, lean_box(0), lean_box(0));
lean_inc(v_a_496_);
lean_inc_ref(v_a_495_);
lean_inc(v_a_494_);
lean_inc_ref(v_a_493_);
lean_inc(v_a_492_);
lean_inc(v_a_491_);
lean_inc(v_a_490_);
lean_inc_ref(v_a_489_);
v___x_514_ = lean_apply_9(v___x_19752__overap_513_, v_a_489_, v_a_490_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_, lean_box(0));
if (lean_obj_tag(v___x_514_) == 0)
{
lean_object* v_a_515_; lean_object* v___x_517_; uint8_t v_isShared_518_; uint8_t v_isSharedCheck_591_; 
v_a_515_ = lean_ctor_get(v___x_514_, 0);
v_isSharedCheck_591_ = !lean_is_exclusive(v___x_514_);
if (v_isSharedCheck_591_ == 0)
{
v___x_517_ = v___x_514_;
v_isShared_518_ = v_isSharedCheck_591_;
goto v_resetjp_516_;
}
else
{
lean_inc(v_a_515_);
lean_dec(v___x_514_);
v___x_517_ = lean_box(0);
v_isShared_518_ = v_isSharedCheck_591_;
goto v_resetjp_516_;
}
v_resetjp_516_:
{
lean_object* v_fst_519_; lean_object* v_snd_520_; lean_object* v___y_560_; lean_object* v___x_569_; 
v_fst_519_ = lean_ctor_get(v_a_515_, 0);
lean_inc(v_fst_519_);
v_snd_520_ = lean_ctor_get(v_a_515_, 1);
lean_inc(v_snd_520_);
lean_dec(v_a_515_);
v___x_569_ = lp_aesop_Aesop_enqueueGoals___redArg(v_inst_485_, v_snd_520_, v_a_490_);
if (lean_obj_tag(v___x_569_) == 0)
{
lean_object* v___x_570_; uint8_t v___x_571_; 
lean_dec_ref_known(v___x_569_, 1);
v___x_570_ = lean_array_get_size(v_fst_519_);
v___x_571_ = lean_nat_dec_lt(v___x_509_, v___x_570_);
if (v___x_571_ == 0)
{
goto v___jp_526_;
}
else
{
lean_object* v___f_572_; lean_object* v___x_573_; uint8_t v___x_574_; 
v___f_572_ = ((lean_object*)(lp_aesop_Aesop_addRapps___redArg___closed__3));
v___x_573_ = lean_box(0);
v___x_574_ = lean_nat_dec_le(v___x_570_, v___x_570_);
if (v___x_574_ == 0)
{
if (v___x_571_ == 0)
{
goto v___jp_526_;
}
else
{
size_t v___x_575_; size_t v___x_576_; lean_object* v___x_19866__overap_577_; lean_object* v___x_578_; 
v___x_575_ = ((size_t)0ULL);
v___x_576_ = lean_usize_of_nat(v___x_570_);
lean_inc(v_fst_519_);
lean_inc_ref(v___x_498_);
v___x_19866__overap_577_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_498_, v___f_572_, v_fst_519_, v___x_575_, v___x_576_, v___x_573_);
lean_inc(v_a_496_);
lean_inc_ref(v_a_495_);
lean_inc(v_a_494_);
lean_inc_ref(v_a_493_);
lean_inc(v_a_492_);
lean_inc(v_a_491_);
lean_inc(v_a_490_);
lean_inc_ref(v_a_489_);
v___x_578_ = lean_apply_9(v___x_19866__overap_577_, v_a_489_, v_a_490_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_, lean_box(0));
v___y_560_ = v___x_578_;
goto v___jp_559_;
}
}
else
{
size_t v___x_579_; size_t v___x_580_; lean_object* v___x_19870__overap_581_; lean_object* v___x_582_; 
v___x_579_ = ((size_t)0ULL);
v___x_580_ = lean_usize_of_nat(v___x_570_);
lean_inc(v_fst_519_);
lean_inc_ref(v___x_498_);
v___x_19870__overap_581_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_498_, v___f_572_, v_fst_519_, v___x_579_, v___x_580_, v___x_573_);
lean_inc(v_a_496_);
lean_inc_ref(v_a_495_);
lean_inc(v_a_494_);
lean_inc_ref(v_a_493_);
lean_inc(v_a_492_);
lean_inc(v_a_491_);
lean_inc(v_a_490_);
lean_inc_ref(v_a_489_);
v___x_582_ = lean_apply_9(v___x_19870__overap_581_, v_a_489_, v_a_490_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_, lean_box(0));
v___y_560_ = v___x_582_;
goto v___jp_559_;
}
}
}
else
{
lean_object* v_a_583_; lean_object* v___x_585_; uint8_t v_isShared_586_; uint8_t v_isSharedCheck_590_; 
lean_dec(v_fst_519_);
lean_del_object(v___x_517_);
lean_dec_ref(v___x_498_);
v_a_583_ = lean_ctor_get(v___x_569_, 0);
v_isSharedCheck_590_ = !lean_is_exclusive(v___x_569_);
if (v_isSharedCheck_590_ == 0)
{
v___x_585_ = v___x_569_;
v_isShared_586_ = v_isSharedCheck_590_;
goto v_resetjp_584_;
}
else
{
lean_inc(v_a_583_);
lean_dec(v___x_569_);
v___x_585_ = lean_box(0);
v_isShared_586_ = v_isSharedCheck_590_;
goto v_resetjp_584_;
}
v_resetjp_584_:
{
lean_object* v___x_588_; 
if (v_isShared_586_ == 0)
{
v___x_588_ = v___x_585_;
goto v_reusejp_587_;
}
else
{
lean_object* v_reuseFailAlloc_589_; 
v_reuseFailAlloc_589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_589_, 0, v_a_583_);
v___x_588_ = v_reuseFailAlloc_589_;
goto v_reusejp_587_;
}
v_reusejp_587_:
{
return v___x_588_;
}
}
}
v___jp_521_:
{
lean_object* v___x_522_; lean_object* v___x_524_; 
v___x_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_522_, 0, v_fst_519_);
if (v_isShared_518_ == 0)
{
lean_ctor_set(v___x_517_, 0, v___x_522_);
v___x_524_ = v___x_517_;
goto v_reusejp_523_;
}
else
{
lean_object* v_reuseFailAlloc_525_; 
v_reuseFailAlloc_525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_525_, 0, v___x_522_);
v___x_524_ = v_reuseFailAlloc_525_;
goto v_reusejp_523_;
}
v_reusejp_523_:
{
return v___x_524_;
}
}
v___jp_526_:
{
lean_object* v___x_527_; lean_object* v___f_528_; size_t v_sz_529_; size_t v___x_530_; lean_object* v___x_19815__overap_531_; lean_object* v___x_532_; 
v___x_527_ = ((lean_object*)(lp_aesop_Aesop_addRapps___redArg___closed__1));
v___f_528_ = ((lean_object*)(lp_aesop_Aesop_addRapps___redArg___closed__2));
v_sz_529_ = lean_array_size(v_fst_519_);
v___x_530_ = ((size_t)0ULL);
lean_inc(v_fst_519_);
v___x_19815__overap_531_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_498_, v_fst_519_, v___f_528_, v_sz_529_, v___x_530_, v___x_527_);
lean_inc(v_a_496_);
lean_inc_ref(v_a_495_);
lean_inc(v_a_494_);
lean_inc_ref(v_a_493_);
lean_inc(v_a_492_);
lean_inc(v_a_491_);
lean_inc(v_a_490_);
lean_inc_ref(v_a_489_);
v___x_532_ = lean_apply_9(v___x_19815__overap_531_, v_a_489_, v_a_490_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_, lean_box(0));
if (lean_obj_tag(v___x_532_) == 0)
{
lean_object* v_a_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_550_; 
v_a_533_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_550_ == 0)
{
v___x_535_ = v___x_532_;
v_isShared_536_ = v_isSharedCheck_550_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_a_533_);
lean_dec(v___x_532_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_550_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v_fst_537_; 
v_fst_537_ = lean_ctor_get(v_a_533_, 0);
lean_inc(v_fst_537_);
lean_dec(v_a_533_);
if (lean_obj_tag(v_fst_537_) == 0)
{
lean_del_object(v___x_535_);
goto v___jp_521_;
}
else
{
lean_object* v_val_538_; 
v_val_538_ = lean_ctor_get(v_fst_537_, 0);
lean_inc(v_val_538_);
lean_dec_ref_known(v_fst_537_, 1);
if (lean_obj_tag(v_val_538_) == 1)
{
lean_object* v___x_540_; uint8_t v_isShared_541_; uint8_t v_isSharedCheck_548_; 
lean_del_object(v___x_517_);
v_isSharedCheck_548_ = !lean_is_exclusive(v_val_538_);
if (v_isSharedCheck_548_ == 0)
{
lean_object* v_unused_549_; 
v_unused_549_ = lean_ctor_get(v_val_538_, 0);
lean_dec(v_unused_549_);
v___x_540_ = v_val_538_;
v_isShared_541_ = v_isSharedCheck_548_;
goto v_resetjp_539_;
}
else
{
lean_dec(v_val_538_);
v___x_540_ = lean_box(0);
v_isShared_541_ = v_isSharedCheck_548_;
goto v_resetjp_539_;
}
v_resetjp_539_:
{
lean_object* v___x_543_; 
if (v_isShared_541_ == 0)
{
lean_ctor_set_tag(v___x_540_, 0);
lean_ctor_set(v___x_540_, 0, v_fst_519_);
v___x_543_ = v___x_540_;
goto v_reusejp_542_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v_fst_519_);
v___x_543_ = v_reuseFailAlloc_547_;
goto v_reusejp_542_;
}
v_reusejp_542_:
{
lean_object* v___x_545_; 
if (v_isShared_536_ == 0)
{
lean_ctor_set(v___x_535_, 0, v___x_543_);
v___x_545_ = v___x_535_;
goto v_reusejp_544_;
}
else
{
lean_object* v_reuseFailAlloc_546_; 
v_reuseFailAlloc_546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_546_, 0, v___x_543_);
v___x_545_ = v_reuseFailAlloc_546_;
goto v_reusejp_544_;
}
v_reusejp_544_:
{
return v___x_545_;
}
}
}
}
else
{
lean_dec(v_val_538_);
lean_del_object(v___x_535_);
goto v___jp_521_;
}
}
}
}
else
{
lean_object* v_a_551_; lean_object* v___x_553_; uint8_t v_isShared_554_; uint8_t v_isSharedCheck_558_; 
lean_dec(v_fst_519_);
lean_del_object(v___x_517_);
v_a_551_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_558_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_558_ == 0)
{
v___x_553_ = v___x_532_;
v_isShared_554_ = v_isSharedCheck_558_;
goto v_resetjp_552_;
}
else
{
lean_inc(v_a_551_);
lean_dec(v___x_532_);
v___x_553_ = lean_box(0);
v_isShared_554_ = v_isSharedCheck_558_;
goto v_resetjp_552_;
}
v_resetjp_552_:
{
lean_object* v___x_556_; 
if (v_isShared_554_ == 0)
{
v___x_556_ = v___x_553_;
goto v_reusejp_555_;
}
else
{
lean_object* v_reuseFailAlloc_557_; 
v_reuseFailAlloc_557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_557_, 0, v_a_551_);
v___x_556_ = v_reuseFailAlloc_557_;
goto v_reusejp_555_;
}
v_reusejp_555_:
{
return v___x_556_;
}
}
}
}
v___jp_559_:
{
if (lean_obj_tag(v___y_560_) == 0)
{
lean_dec_ref_known(v___y_560_, 1);
goto v___jp_526_;
}
else
{
lean_object* v_a_561_; lean_object* v___x_563_; uint8_t v_isShared_564_; uint8_t v_isSharedCheck_568_; 
lean_dec(v_fst_519_);
lean_del_object(v___x_517_);
lean_dec_ref(v___x_498_);
v_a_561_ = lean_ctor_get(v___y_560_, 0);
v_isSharedCheck_568_ = !lean_is_exclusive(v___y_560_);
if (v_isSharedCheck_568_ == 0)
{
v___x_563_ = v___y_560_;
v_isShared_564_ = v_isSharedCheck_568_;
goto v_resetjp_562_;
}
else
{
lean_inc(v_a_561_);
lean_dec(v___y_560_);
v___x_563_ = lean_box(0);
v_isShared_564_ = v_isSharedCheck_568_;
goto v_resetjp_562_;
}
v_resetjp_562_:
{
lean_object* v___x_566_; 
if (v_isShared_564_ == 0)
{
v___x_566_ = v___x_563_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_567_; 
v_reuseFailAlloc_567_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_567_, 0, v_a_561_);
v___x_566_ = v_reuseFailAlloc_567_;
goto v_reusejp_565_;
}
v_reusejp_565_:
{
return v___x_566_;
}
}
}
}
}
}
else
{
lean_object* v_a_592_; lean_object* v___x_594_; uint8_t v_isShared_595_; uint8_t v_isSharedCheck_599_; 
lean_dec_ref(v___x_498_);
lean_dec_ref(v_inst_485_);
v_a_592_ = lean_ctor_get(v___x_514_, 0);
v_isSharedCheck_599_ = !lean_is_exclusive(v___x_514_);
if (v_isSharedCheck_599_ == 0)
{
v___x_594_ = v___x_514_;
v_isShared_595_ = v_isSharedCheck_599_;
goto v_resetjp_593_;
}
else
{
lean_inc(v_a_592_);
lean_dec(v___x_514_);
v___x_594_ = lean_box(0);
v_isShared_595_ = v_isSharedCheck_599_;
goto v_resetjp_593_;
}
v_resetjp_593_:
{
lean_object* v___x_597_; 
if (v_isShared_595_ == 0)
{
v___x_597_ = v___x_594_;
goto v_reusejp_596_;
}
else
{
lean_object* v_reuseFailAlloc_598_; 
v_reuseFailAlloc_598_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_598_, 0, v_a_592_);
v___x_597_ = v_reuseFailAlloc_598_;
goto v_reusejp_596_;
}
v_reusejp_596_:
{
return v___x_597_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___redArg___boxed(lean_object* v_inst_600_, lean_object* v_parentRef_601_, lean_object* v_rule_602_, lean_object* v_rapps_603_, lean_object* v_a_604_, lean_object* v_a_605_, lean_object* v_a_606_, lean_object* v_a_607_, lean_object* v_a_608_, lean_object* v_a_609_, lean_object* v_a_610_, lean_object* v_a_611_, lean_object* v_a_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_aesop_Aesop_addRapps___redArg(v_inst_600_, v_parentRef_601_, v_rule_602_, v_rapps_603_, v_a_604_, v_a_605_, v_a_606_, v_a_607_, v_a_608_, v_a_609_, v_a_610_, v_a_611_);
lean_dec(v_a_611_);
lean_dec_ref(v_a_610_);
lean_dec(v_a_609_);
lean_dec_ref(v_a_608_);
lean_dec(v_a_607_);
lean_dec(v_a_606_);
lean_dec(v_a_605_);
lean_dec_ref(v_a_604_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps(lean_object* v_Q_614_, lean_object* v_inst_615_, lean_object* v_parentRef_616_, lean_object* v_rule_617_, lean_object* v_rapps_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_, lean_object* v_a_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = lp_aesop_Aesop_addRapps___redArg(v_inst_615_, v_parentRef_616_, v_rule_617_, v_rapps_618_, v_a_619_, v_a_620_, v_a_621_, v_a_622_, v_a_623_, v_a_624_, v_a_625_, v_a_626_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRapps___boxed(lean_object* v_Q_629_, lean_object* v_inst_630_, lean_object* v_parentRef_631_, lean_object* v_rule_632_, lean_object* v_rapps_633_, lean_object* v_a_634_, lean_object* v_a_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_aesop_Aesop_addRapps(v_Q_629_, v_inst_630_, v_parentRef_631_, v_rule_632_, v_rapps_633_, v_a_634_, v_a_635_, v_a_636_, v_a_637_, v_a_638_, v_a_639_, v_a_640_, v_a_641_);
lean_dec(v_a_641_);
lean_dec_ref(v_a_640_);
lean_dec(v_a_639_);
lean_dec_ref(v_a_638_);
lean_dec(v_a_637_);
lean_dec(v_a_636_);
lean_dec(v_a_635_);
lean_dec_ref(v_a_634_);
return v_res_643_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(lean_object* v_rule_644_, lean_object* v_parentRef_645_, lean_object* v_a_646_){
_start:
{
lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v_introGoal_651_; lean_object* v_elimGoal_652_; lean_object* v___x_653_; lean_object* v_id_654_; lean_object* v_parent_655_; lean_object* v_children_656_; lean_object* v_origin_657_; lean_object* v_depth_658_; uint8_t v_state_659_; uint8_t v_isIrrelevant_660_; uint8_t v_isForcedUnprovable_661_; lean_object* v_preNormGoal_662_; lean_object* v_normalizationState_663_; lean_object* v_mvars_664_; lean_object* v_forwardState_665_; lean_object* v_forwardRuleMatches_666_; double v_successProbability_667_; lean_object* v_addedInIteration_668_; lean_object* v_lastExpandedInIteration_669_; uint8_t v_unsafeRulesSelected_670_; lean_object* v_unsafeQueue_671_; lean_object* v_failedRapps_672_; lean_object* v___x_674_; uint8_t v_isShared_675_; uint8_t v_isSharedCheck_683_; 
v___x_648_ = lean_st_ref_get(v_a_646_);
lean_dec(v___x_648_);
v___x_649_ = lean_st_ref_take(v_parentRef_645_);
v___x_650_ = lp_aesop_Aesop_treeImpl;
v_introGoal_651_ = lean_ctor_get(v___x_650_, 0);
v_elimGoal_652_ = lean_ctor_get(v___x_650_, 1);
lean_inc_ref(v_elimGoal_652_);
v___x_653_ = lean_apply_1(v_elimGoal_652_, v___x_649_);
v_id_654_ = lean_ctor_get(v___x_653_, 0);
v_parent_655_ = lean_ctor_get(v___x_653_, 1);
v_children_656_ = lean_ctor_get(v___x_653_, 2);
v_origin_657_ = lean_ctor_get(v___x_653_, 3);
v_depth_658_ = lean_ctor_get(v___x_653_, 4);
v_state_659_ = lean_ctor_get_uint8(v___x_653_, sizeof(void*)*14 + 8);
v_isIrrelevant_660_ = lean_ctor_get_uint8(v___x_653_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_661_ = lean_ctor_get_uint8(v___x_653_, sizeof(void*)*14 + 10);
v_preNormGoal_662_ = lean_ctor_get(v___x_653_, 5);
v_normalizationState_663_ = lean_ctor_get(v___x_653_, 6);
v_mvars_664_ = lean_ctor_get(v___x_653_, 7);
v_forwardState_665_ = lean_ctor_get(v___x_653_, 8);
v_forwardRuleMatches_666_ = lean_ctor_get(v___x_653_, 9);
v_successProbability_667_ = lean_ctor_get_float(v___x_653_, sizeof(void*)*14);
v_addedInIteration_668_ = lean_ctor_get(v___x_653_, 10);
v_lastExpandedInIteration_669_ = lean_ctor_get(v___x_653_, 11);
v_unsafeRulesSelected_670_ = lean_ctor_get_uint8(v___x_653_, sizeof(void*)*14 + 11);
v_unsafeQueue_671_ = lean_ctor_get(v___x_653_, 12);
v_failedRapps_672_ = lean_ctor_get(v___x_653_, 13);
v_isSharedCheck_683_ = !lean_is_exclusive(v___x_653_);
if (v_isSharedCheck_683_ == 0)
{
v___x_674_ = v___x_653_;
v_isShared_675_ = v_isSharedCheck_683_;
goto v_resetjp_673_;
}
else
{
lean_inc(v_failedRapps_672_);
lean_inc(v_unsafeQueue_671_);
lean_inc(v_lastExpandedInIteration_669_);
lean_inc(v_addedInIteration_668_);
lean_inc(v_forwardRuleMatches_666_);
lean_inc(v_forwardState_665_);
lean_inc(v_mvars_664_);
lean_inc(v_normalizationState_663_);
lean_inc(v_preNormGoal_662_);
lean_inc(v_depth_658_);
lean_inc(v_origin_657_);
lean_inc(v_children_656_);
lean_inc(v_parent_655_);
lean_inc(v_id_654_);
lean_dec(v___x_653_);
v___x_674_ = lean_box(0);
v_isShared_675_ = v_isSharedCheck_683_;
goto v_resetjp_673_;
}
v_resetjp_673_:
{
lean_object* v___x_676_; lean_object* v___x_678_; 
v___x_676_ = lean_array_push(v_failedRapps_672_, v_rule_644_);
if (v_isShared_675_ == 0)
{
lean_ctor_set(v___x_674_, 13, v___x_676_);
v___x_678_ = v___x_674_;
goto v_reusejp_677_;
}
else
{
lean_object* v_reuseFailAlloc_682_; 
v_reuseFailAlloc_682_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_682_, 0, v_id_654_);
lean_ctor_set(v_reuseFailAlloc_682_, 1, v_parent_655_);
lean_ctor_set(v_reuseFailAlloc_682_, 2, v_children_656_);
lean_ctor_set(v_reuseFailAlloc_682_, 3, v_origin_657_);
lean_ctor_set(v_reuseFailAlloc_682_, 4, v_depth_658_);
lean_ctor_set(v_reuseFailAlloc_682_, 5, v_preNormGoal_662_);
lean_ctor_set(v_reuseFailAlloc_682_, 6, v_normalizationState_663_);
lean_ctor_set(v_reuseFailAlloc_682_, 7, v_mvars_664_);
lean_ctor_set(v_reuseFailAlloc_682_, 8, v_forwardState_665_);
lean_ctor_set(v_reuseFailAlloc_682_, 9, v_forwardRuleMatches_666_);
lean_ctor_set(v_reuseFailAlloc_682_, 10, v_addedInIteration_668_);
lean_ctor_set(v_reuseFailAlloc_682_, 11, v_lastExpandedInIteration_669_);
lean_ctor_set(v_reuseFailAlloc_682_, 12, v_unsafeQueue_671_);
lean_ctor_set(v_reuseFailAlloc_682_, 13, v___x_676_);
lean_ctor_set_uint8(v_reuseFailAlloc_682_, sizeof(void*)*14 + 8, v_state_659_);
lean_ctor_set_uint8(v_reuseFailAlloc_682_, sizeof(void*)*14 + 9, v_isIrrelevant_660_);
lean_ctor_set_uint8(v_reuseFailAlloc_682_, sizeof(void*)*14 + 10, v_isForcedUnprovable_661_);
lean_ctor_set_float(v_reuseFailAlloc_682_, sizeof(void*)*14, v_successProbability_667_);
lean_ctor_set_uint8(v_reuseFailAlloc_682_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_670_);
v___x_678_ = v_reuseFailAlloc_682_;
goto v_reusejp_677_;
}
v_reusejp_677_:
{
lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; 
lean_inc(v_introGoal_651_);
v___x_679_ = lean_apply_1(v_introGoal_651_, v___x_678_);
v___x_680_ = lean_st_ref_set(v_parentRef_645_, v___x_679_);
v___x_681_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_681_, 0, v___x_680_);
return v___x_681_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg___boxed(lean_object* v_rule_684_, lean_object* v_parentRef_685_, lean_object* v_a_686_, lean_object* v_a_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(v_rule_684_, v_parentRef_685_, v_a_686_);
lean_dec(v_a_686_);
lean_dec(v_parentRef_685_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure(lean_object* v_Q_689_, lean_object* v_inst_690_, lean_object* v_rule_691_, lean_object* v_parentRef_692_, lean_object* v_a_693_, lean_object* v_a_694_, lean_object* v_a_695_, lean_object* v_a_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_){
_start:
{
lean_object* v___x_702_; 
v___x_702_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(v_rule_691_, v_parentRef_692_, v_a_694_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___boxed(lean_object* v_Q_703_, lean_object* v_inst_704_, lean_object* v_rule_705_, lean_object* v_parentRef_706_, lean_object* v_a_707_, lean_object* v_a_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_, lean_object* v_a_714_, lean_object* v_a_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure(v_Q_703_, v_inst_704_, v_rule_705_, v_parentRef_706_, v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_, v_a_713_, v_a_714_);
lean_dec(v_a_714_);
lean_dec_ref(v_a_713_);
lean_dec(v_a_712_);
lean_dec_ref(v_a_711_);
lean_dec(v_a_710_);
lean_dec(v_a_709_);
lean_dec(v_a_708_);
lean_dec_ref(v_a_707_);
lean_dec(v_parentRef_706_);
lean_dec_ref(v_inst_704_);
return v_res_716_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__1(void){
_start:
{
lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_718_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__0));
v___x_719_ = l_Lean_stringToMessageData(v___x_718_);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg(lean_object* v_ruleName_734_, lean_object* v_toEmoji_735_, lean_object* v_suffix_736_, lean_object* v_result_737_){
_start:
{
lean_object* v_name_739_; uint8_t v_builder_740_; uint8_t v_phase_741_; uint8_t v_scope_742_; lean_object* v_emoji_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___y_748_; lean_object* v___y_749_; lean_object* v___y_750_; lean_object* v___y_763_; lean_object* v___y_764_; lean_object* v___y_765_; lean_object* v___y_771_; 
v_name_739_ = lean_ctor_get(v_ruleName_734_, 0);
lean_inc(v_name_739_);
v_builder_740_ = lean_ctor_get_uint8(v_ruleName_734_, sizeof(void*)*1 + 8);
v_phase_741_ = lean_ctor_get_uint8(v_ruleName_734_, sizeof(void*)*1 + 9);
v_scope_742_ = lean_ctor_get_uint8(v_ruleName_734_, sizeof(void*)*1 + 10);
lean_dec_ref(v_ruleName_734_);
v_emoji_743_ = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(v_toEmoji_735_, v_result_737_);
v___x_744_ = l_Lean_stringToMessageData(v_emoji_743_);
v___x_745_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__1, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__1);
v___x_746_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_746_, 0, v___x_744_);
lean_ctor_set(v___x_746_, 1, v___x_745_);
switch(v_phase_741_)
{
case 0:
{
lean_object* v___x_782_; 
v___x_782_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__13));
v___y_771_ = v___x_782_;
goto v___jp_770_;
}
case 1:
{
lean_object* v___x_783_; 
v___x_783_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__14));
v___y_771_ = v___x_783_;
goto v___jp_770_;
}
default: 
{
lean_object* v___x_784_; 
v___x_784_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__15));
v___y_771_ = v___x_784_;
goto v___jp_770_;
}
}
v___jp_747_:
{
lean_object* v___x_751_; lean_object* v___x_752_; uint8_t v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; 
v___x_751_ = lean_string_append(v___y_749_, v___y_750_);
v___x_752_ = lean_string_append(v___x_751_, v___y_748_);
v___x_753_ = 1;
v___x_754_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_739_, v___x_753_);
v___x_755_ = lean_string_append(v___x_752_, v___x_754_);
lean_dec_ref(v___x_754_);
v___x_756_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_756_, 0, v___x_755_);
v___x_757_ = l_Lean_MessageData_ofFormat(v___x_756_);
v___x_758_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_758_, 0, v___x_746_);
lean_ctor_set(v___x_758_, 1, v___x_757_);
v___x_759_ = l_Lean_stringToMessageData(v_suffix_736_);
v___x_760_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_760_, 0, v___x_758_);
lean_ctor_set(v___x_760_, 1, v___x_759_);
v___x_761_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_761_, 0, v___x_760_);
return v___x_761_;
}
v___jp_762_:
{
lean_object* v___x_766_; lean_object* v___x_767_; 
v___x_766_ = lean_string_append(v___y_764_, v___y_765_);
v___x_767_ = lean_string_append(v___x_766_, v___y_763_);
if (v_scope_742_ == 0)
{
lean_object* v___x_768_; 
v___x_768_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__2));
v___y_748_ = v___y_763_;
v___y_749_ = v___x_767_;
v___y_750_ = v___x_768_;
goto v___jp_747_;
}
else
{
lean_object* v___x_769_; 
v___x_769_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__3));
v___y_748_ = v___y_763_;
v___y_749_ = v___x_767_;
v___y_750_ = v___x_769_;
goto v___jp_747_;
}
}
v___jp_770_:
{
lean_object* v___x_772_; lean_object* v___x_773_; 
v___x_772_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__4));
lean_inc_ref(v___y_771_);
v___x_773_ = lean_string_append(v___y_771_, v___x_772_);
switch(v_builder_740_)
{
case 0:
{
lean_object* v___x_774_; 
v___x_774_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__5));
v___y_763_ = v___x_772_;
v___y_764_ = v___x_773_;
v___y_765_ = v___x_774_;
goto v___jp_762_;
}
case 1:
{
lean_object* v___x_775_; 
v___x_775_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__6));
v___y_763_ = v___x_772_;
v___y_764_ = v___x_773_;
v___y_765_ = v___x_775_;
goto v___jp_762_;
}
case 2:
{
lean_object* v___x_776_; 
v___x_776_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__7));
v___y_763_ = v___x_772_;
v___y_764_ = v___x_773_;
v___y_765_ = v___x_776_;
goto v___jp_762_;
}
case 3:
{
lean_object* v___x_777_; 
v___x_777_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__8));
v___y_763_ = v___x_772_;
v___y_764_ = v___x_773_;
v___y_765_ = v___x_777_;
goto v___jp_762_;
}
case 4:
{
lean_object* v___x_778_; 
v___x_778_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__9));
v___y_763_ = v___x_772_;
v___y_764_ = v___x_773_;
v___y_765_ = v___x_778_;
goto v___jp_762_;
}
case 5:
{
lean_object* v___x_779_; 
v___x_779_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__10));
v___y_763_ = v___x_772_;
v___y_764_ = v___x_773_;
v___y_765_ = v___x_779_;
goto v___jp_762_;
}
case 6:
{
lean_object* v___x_780_; 
v___x_780_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__11));
v___y_763_ = v___x_772_;
v___y_764_ = v___x_773_;
v___y_765_ = v___x_780_;
goto v___jp_762_;
}
default: 
{
lean_object* v___x_781_; 
v___x_781_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___closed__12));
v___y_763_ = v___x_772_;
v___y_764_ = v___x_773_;
v___y_765_ = v___x_781_;
goto v___jp_762_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg___boxed(lean_object* v_ruleName_785_, lean_object* v_toEmoji_786_, lean_object* v_suffix_787_, lean_object* v_result_788_, lean_object* v_a_789_){
_start:
{
lean_object* v_res_790_; 
v_res_790_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg(v_ruleName_785_, v_toEmoji_786_, v_suffix_787_, v_result_788_);
return v_res_790_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt(lean_object* v_Q_791_, lean_object* v_inst_792_, lean_object* v_00_u03b1_793_, lean_object* v_ruleName_794_, lean_object* v_toEmoji_795_, lean_object* v_suffix_796_, lean_object* v_result_797_, lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_, lean_object* v_a_802_, lean_object* v_a_803_, lean_object* v_a_804_, lean_object* v_a_805_){
_start:
{
lean_object* v___x_807_; 
v___x_807_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___redArg(v_ruleName_794_, v_toEmoji_795_, v_suffix_796_, v_result_797_);
return v___x_807_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed(lean_object* v_Q_808_, lean_object* v_inst_809_, lean_object* v_00_u03b1_810_, lean_object* v_ruleName_811_, lean_object* v_toEmoji_812_, lean_object* v_suffix_813_, lean_object* v_result_814_, lean_object* v_a_815_, lean_object* v_a_816_, lean_object* v_a_817_, lean_object* v_a_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_, lean_object* v_a_822_, lean_object* v_a_823_){
_start:
{
lean_object* v_res_824_; 
v_res_824_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt(v_Q_808_, v_inst_809_, v_00_u03b1_810_, v_ruleName_811_, v_toEmoji_812_, v_suffix_813_, v_result_814_, v_a_815_, v_a_816_, v_a_817_, v_a_818_, v_a_819_, v_a_820_, v_a_821_, v_a_822_);
lean_dec(v_a_822_);
lean_dec_ref(v_a_821_);
lean_dec(v_a_820_);
lean_dec_ref(v_a_819_);
lean_dec(v_a_818_);
lean_dec(v_a_817_);
lean_dec(v_a_816_);
lean_dec_ref(v_a_815_);
lean_dec_ref(v_inst_809_);
return v_res_824_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__2(void){
_start:
{
lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_827_ = l_Lean_Core_instMonadTraceCoreM;
v___x_828_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1));
v___x_829_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_828_, v___x_827_);
return v___x_829_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3(void){
_start:
{
lean_object* v___x_830_; lean_object* v___f_831_; lean_object* v___x_832_; 
v___x_830_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__2, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__2_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__2);
v___f_831_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__0));
v___x_832_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_831_, v___x_830_);
return v___x_832_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__4(void){
_start:
{
lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_833_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3);
v___x_834_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1));
v___x_835_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_834_, v___x_833_);
return v___x_835_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__5(void){
_start:
{
lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; 
v___x_836_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__4, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__4_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__4);
v___x_837_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1));
v___x_838_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_837_, v___x_836_);
return v___x_838_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__6(void){
_start:
{
lean_object* v___x_839_; lean_object* v___f_840_; lean_object* v___x_841_; 
v___x_839_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__5, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__5_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__5);
v___f_840_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__0));
v___x_841_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_840_, v___x_839_);
return v___x_841_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__7(void){
_start:
{
lean_object* v___x_842_; 
v___x_842_ = l_instMonadExceptOfEIO(lean_box(0));
return v___x_842_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__8(void){
_start:
{
lean_object* v___x_843_; lean_object* v___x_844_; 
v___x_843_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__7, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__7_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__7);
v___x_844_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_843_);
return v___x_844_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__9(void){
_start:
{
lean_object* v___x_845_; lean_object* v___x_846_; 
v___x_845_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__8, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__8_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__8);
v___x_846_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_845_);
return v___x_846_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__10(void){
_start:
{
lean_object* v___x_847_; lean_object* v___x_848_; 
v___x_847_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__9, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__9_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__9);
v___x_848_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_847_);
return v___x_848_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__11(void){
_start:
{
lean_object* v___x_849_; lean_object* v___x_850_; 
v___x_849_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__10, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__10_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__10);
v___x_850_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_849_);
return v___x_850_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__12(void){
_start:
{
lean_object* v___x_851_; lean_object* v___x_852_; 
v___x_851_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__11, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__11_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__11);
v___x_852_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_851_);
return v___x_852_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__13(void){
_start:
{
lean_object* v___x_853_; lean_object* v___x_854_; 
v___x_853_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__12, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__12_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__12);
v___x_854_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_853_);
return v___x_854_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15(void){
_start:
{
lean_object* v___x_856_; lean_object* v___f_857_; lean_object* v___x_858_; 
v___x_856_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__6, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__6_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__6);
v___f_857_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__14));
v___x_858_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_857_, v___x_856_);
return v___x_858_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__16(void){
_start:
{
lean_object* v___x_859_; lean_object* v___x_860_; 
v___x_859_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__13, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__13_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__13);
v___x_860_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_859_);
return v___x_860_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17(void){
_start:
{
lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_861_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__16, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__16_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__16);
v___x_862_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_861_);
return v___x_862_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__18(void){
_start:
{
lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___f_865_; 
v___x_863_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1));
v___x_864_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_865_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_865_, 0, v___x_864_);
lean_closure_set(v___f_865_, 1, v___x_863_);
return v___f_865_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__19(void){
_start:
{
lean_object* v___x_866_; lean_object* v___f_867_; lean_object* v___f_868_; 
v___x_866_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1));
v___f_867_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__18, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__18_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__18);
v___f_868_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_868_, 0, v___f_867_);
lean_closure_set(v___f_868_, 1, v___x_866_);
return v___f_868_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__20(void){
_start:
{
lean_object* v___f_869_; lean_object* v___f_870_; lean_object* v___f_871_; 
v___f_869_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__0));
v___f_870_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__19, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__19_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__19);
v___f_871_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_871_, 0, v___f_870_);
lean_closure_set(v___f_871_, 1, v___f_869_);
return v___f_871_;
}
}
static lean_object* _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21(void){
_start:
{
lean_object* v___f_872_; lean_object* v___f_873_; lean_object* v___f_874_; 
v___f_872_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__14));
v___f_873_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__20, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__20_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__20);
v___f_874_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_874_, 0, v___f_873_);
lean_closure_set(v___f_874_, 1, v___f_872_);
return v___f_874_;
}
}
static double _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26(void){
_start:
{
lean_object* v___x_880_; double v___x_881_; 
v___x_880_ = lean_unsigned_to_nat(1000000000u);
v___x_881_ = lean_float_of_nat(v___x_880_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg(lean_object* v_inst_882_, lean_object* v_ruleName_883_, lean_object* v_toEmoji_884_, lean_object* v_suffix_885_, lean_object* v_k_886_, lean_object* v_a_887_, lean_object* v_a_888_, lean_object* v_a_889_, lean_object* v_a_890_, lean_object* v_a_891_, lean_object* v_a_892_, lean_object* v_a_893_, lean_object* v_a_894_){
_start:
{
lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v_options_900_; uint8_t v_hasTrace_901_; 
v___x_896_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_882_);
v___x_897_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_898_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_882_);
v___x_899_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_options_900_ = lean_ctor_get(v_a_893_, 2);
v_hasTrace_901_ = lean_ctor_get_uint8(v_options_900_, sizeof(void*)*1);
if (v_hasTrace_901_ == 0)
{
lean_object* v___x_902_; 
lean_dec_ref(v___x_898_);
lean_dec_ref(v___x_896_);
lean_dec_ref(v_suffix_885_);
lean_dec_ref(v_toEmoji_884_);
lean_dec_ref(v_ruleName_883_);
lean_dec_ref(v_inst_882_);
lean_inc(v_a_894_);
lean_inc_ref(v_a_893_);
lean_inc(v_a_892_);
lean_inc_ref(v_a_891_);
lean_inc(v_a_890_);
lean_inc(v_a_889_);
lean_inc(v_a_888_);
lean_inc_ref(v_a_887_);
v___x_902_ = lean_apply_9(v_k_886_, v_a_887_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, lean_box(0));
return v___x_902_;
}
else
{
lean_object* v_inheritedTraceOptions_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v_traceClass_906_; lean_object* v___f_907_; lean_object* v___f_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; uint8_t v___x_913_; lean_object* v___y_915_; lean_object* v___y_916_; lean_object* v_a_917_; lean_object* v___y_932_; lean_object* v___y_933_; lean_object* v_a_934_; 
v_inheritedTraceOptions_903_ = lean_ctor_get(v_a_893_, 13);
v___x_904_ = lean_st_ref_get(v_a_888_);
lean_dec(v___x_904_);
v___x_905_ = lp_aesop_Aesop_TraceOption_steps;
v_traceClass_906_ = lean_ctor_get(v___x_905_, 0);
v___f_907_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___f_908_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
v___x_909_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed), 16, 6);
lean_closure_set(v___x_909_, 0, lean_box(0));
lean_closure_set(v___x_909_, 1, v_inst_882_);
lean_closure_set(v___x_909_, 2, lean_box(0));
lean_closure_set(v___x_909_, 3, v_ruleName_883_);
lean_closure_set(v___x_909_, 4, v_toEmoji_884_);
lean_closure_set(v___x_909_, 5, v_suffix_885_);
v___x_910_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23));
v___x_911_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_906_);
v___x_912_ = l_Lean_Name_append(v___x_911_, v_traceClass_906_);
v___x_913_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_903_, v_options_900_, v___x_912_);
lean_dec(v___x_912_);
if (v___x_913_ == 0)
{
lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; uint8_t v___x_1002_; 
v___x_999_ = l_Lean_KVMap_instValueBool;
v___x_1000_ = l_Lean_trace_profiler;
v___x_1001_ = l_Lean_Option_get___redArg(v___x_999_, v_options_900_, v___x_1000_);
v___x_1002_ = lean_unbox(v___x_1001_);
lean_dec(v___x_1001_);
if (v___x_1002_ == 0)
{
lean_object* v___x_1003_; 
lean_dec_ref(v___x_909_);
lean_dec_ref(v___x_898_);
lean_dec_ref(v___x_896_);
lean_inc(v_a_894_);
lean_inc_ref(v_a_893_);
lean_inc(v_a_892_);
lean_inc_ref(v_a_891_);
lean_inc(v_a_890_);
lean_inc(v_a_889_);
lean_inc(v_a_888_);
lean_inc_ref(v_a_887_);
v___x_1003_ = lean_apply_9(v_k_886_, v_a_887_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, lean_box(0));
return v___x_1003_;
}
else
{
goto v___jp_945_;
}
}
else
{
goto v___jp_945_;
}
v___jp_914_:
{
lean_object* v___x_918_; lean_object* v___x_919_; double v___x_920_; double v___x_921_; double v___x_922_; double v___x_923_; double v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_18324__overap_929_; lean_object* v___x_930_; 
v___x_918_ = lean_st_ref_get(v_a_888_);
lean_dec(v___x_918_);
v___x_919_ = lean_io_mono_nanos_now();
v___x_920_ = lean_float_of_nat(v___y_916_);
v___x_921_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_922_ = lean_float_div(v___x_920_, v___x_921_);
v___x_923_ = lean_float_of_nat(v___x_919_);
v___x_924_ = lean_float_div(v___x_923_, v___x_921_);
v___x_925_ = lean_box_float(v___x_922_);
v___x_926_ = lean_box_float(v___x_924_);
v___x_927_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_927_, 0, v___x_925_);
lean_ctor_set(v___x_927_, 1, v___x_926_);
v___x_928_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_928_, 0, v_a_917_);
lean_ctor_set(v___x_928_, 1, v___x_927_);
lean_inc(v_traceClass_906_);
v___x_18324__overap_929_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_896_, v___x_897_, v___x_898_, v___f_907_, lean_box(0), v___x_899_, v___f_908_, v_traceClass_906_, v_hasTrace_901_, v___x_910_, v_options_900_, v___x_913_, v___y_915_, v___x_909_, v___x_928_);
lean_inc(v_a_894_);
lean_inc_ref(v_a_893_);
lean_inc(v_a_892_);
lean_inc_ref(v_a_891_);
lean_inc(v_a_890_);
lean_inc(v_a_889_);
lean_inc(v_a_888_);
lean_inc_ref(v_a_887_);
v___x_930_ = lean_apply_9(v___x_18324__overap_929_, v_a_887_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, lean_box(0));
return v___x_930_;
}
v___jp_931_:
{
lean_object* v___x_935_; lean_object* v___x_936_; double v___x_937_; double v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_18351__overap_943_; lean_object* v___x_944_; 
v___x_935_ = lean_st_ref_get(v_a_888_);
lean_dec(v___x_935_);
v___x_936_ = lean_io_get_num_heartbeats();
v___x_937_ = lean_float_of_nat(v___y_933_);
v___x_938_ = lean_float_of_nat(v___x_936_);
v___x_939_ = lean_box_float(v___x_937_);
v___x_940_ = lean_box_float(v___x_938_);
v___x_941_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_941_, 0, v___x_939_);
lean_ctor_set(v___x_941_, 1, v___x_940_);
v___x_942_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_942_, 0, v_a_934_);
lean_ctor_set(v___x_942_, 1, v___x_941_);
lean_inc(v_traceClass_906_);
v___x_18351__overap_943_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_896_, v___x_897_, v___x_898_, v___f_907_, lean_box(0), v___x_899_, v___f_908_, v_traceClass_906_, v_hasTrace_901_, v___x_910_, v_options_900_, v___x_913_, v___y_932_, v___x_909_, v___x_942_);
lean_inc(v_a_894_);
lean_inc_ref(v_a_893_);
lean_inc(v_a_892_);
lean_inc_ref(v_a_891_);
lean_inc(v_a_890_);
lean_inc(v_a_889_);
lean_inc(v_a_888_);
lean_inc_ref(v_a_887_);
v___x_944_ = lean_apply_9(v___x_18351__overap_943_, v_a_887_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, lean_box(0));
return v___x_944_;
}
v___jp_945_:
{
lean_object* v___x_18295__overap_946_; lean_object* v___x_947_; 
lean_inc_ref(v___x_896_);
v___x_18295__overap_946_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_896_, v___x_897_);
lean_inc(v_a_894_);
lean_inc_ref(v_a_893_);
lean_inc(v_a_892_);
lean_inc_ref(v_a_891_);
lean_inc(v_a_890_);
lean_inc(v_a_889_);
lean_inc(v_a_888_);
lean_inc_ref(v_a_887_);
v___x_947_ = lean_apply_9(v___x_18295__overap_946_, v_a_887_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, lean_box(0));
if (lean_obj_tag(v___x_947_) == 0)
{
lean_object* v_a_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; uint8_t v___x_952_; 
v_a_948_ = lean_ctor_get(v___x_947_, 0);
lean_inc(v_a_948_);
lean_dec_ref_known(v___x_947_, 1);
v___x_949_ = l_Lean_KVMap_instValueBool;
v___x_950_ = l_Lean_trace_profiler_useHeartbeats;
v___x_951_ = l_Lean_Option_get___redArg(v___x_949_, v_options_900_, v___x_950_);
v___x_952_ = lean_unbox(v___x_951_);
lean_dec(v___x_951_);
if (v___x_952_ == 0)
{
lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; 
v___x_953_ = lean_st_ref_get(v_a_888_);
lean_dec(v___x_953_);
v___x_954_ = lean_io_mono_nanos_now();
lean_inc(v_a_894_);
lean_inc_ref(v_a_893_);
lean_inc(v_a_892_);
lean_inc_ref(v_a_891_);
lean_inc(v_a_890_);
lean_inc(v_a_889_);
lean_inc(v_a_888_);
lean_inc_ref(v_a_887_);
v___x_955_ = lean_apply_9(v_k_886_, v_a_887_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, lean_box(0));
if (lean_obj_tag(v___x_955_) == 0)
{
lean_object* v_a_956_; lean_object* v___x_958_; uint8_t v_isShared_959_; uint8_t v_isSharedCheck_963_; 
v_a_956_ = lean_ctor_get(v___x_955_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_955_);
if (v_isSharedCheck_963_ == 0)
{
v___x_958_ = v___x_955_;
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
else
{
lean_inc(v_a_956_);
lean_dec(v___x_955_);
v___x_958_ = lean_box(0);
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
v_resetjp_957_:
{
lean_object* v___x_961_; 
if (v_isShared_959_ == 0)
{
lean_ctor_set_tag(v___x_958_, 1);
v___x_961_ = v___x_958_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v_a_956_);
v___x_961_ = v_reuseFailAlloc_962_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
v___y_915_ = v_a_948_;
v___y_916_ = v___x_954_;
v_a_917_ = v___x_961_;
goto v___jp_914_;
}
}
}
else
{
lean_object* v_a_964_; lean_object* v___x_966_; uint8_t v_isShared_967_; uint8_t v_isSharedCheck_971_; 
v_a_964_ = lean_ctor_get(v___x_955_, 0);
v_isSharedCheck_971_ = !lean_is_exclusive(v___x_955_);
if (v_isSharedCheck_971_ == 0)
{
v___x_966_ = v___x_955_;
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
else
{
lean_inc(v_a_964_);
lean_dec(v___x_955_);
v___x_966_ = lean_box(0);
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
v_resetjp_965_:
{
lean_object* v___x_969_; 
if (v_isShared_967_ == 0)
{
lean_ctor_set_tag(v___x_966_, 0);
v___x_969_ = v___x_966_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v_a_964_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
v___y_915_ = v_a_948_;
v___y_916_ = v___x_954_;
v_a_917_ = v___x_969_;
goto v___jp_914_;
}
}
}
}
else
{
lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; 
v___x_972_ = lean_st_ref_get(v_a_888_);
lean_dec(v___x_972_);
v___x_973_ = lean_io_get_num_heartbeats();
lean_inc(v_a_894_);
lean_inc_ref(v_a_893_);
lean_inc(v_a_892_);
lean_inc_ref(v_a_891_);
lean_inc(v_a_890_);
lean_inc(v_a_889_);
lean_inc(v_a_888_);
lean_inc_ref(v_a_887_);
v___x_974_ = lean_apply_9(v_k_886_, v_a_887_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, lean_box(0));
if (lean_obj_tag(v___x_974_) == 0)
{
lean_object* v_a_975_; lean_object* v___x_977_; uint8_t v_isShared_978_; uint8_t v_isSharedCheck_982_; 
v_a_975_ = lean_ctor_get(v___x_974_, 0);
v_isSharedCheck_982_ = !lean_is_exclusive(v___x_974_);
if (v_isSharedCheck_982_ == 0)
{
v___x_977_ = v___x_974_;
v_isShared_978_ = v_isSharedCheck_982_;
goto v_resetjp_976_;
}
else
{
lean_inc(v_a_975_);
lean_dec(v___x_974_);
v___x_977_ = lean_box(0);
v_isShared_978_ = v_isSharedCheck_982_;
goto v_resetjp_976_;
}
v_resetjp_976_:
{
lean_object* v___x_980_; 
if (v_isShared_978_ == 0)
{
lean_ctor_set_tag(v___x_977_, 1);
v___x_980_ = v___x_977_;
goto v_reusejp_979_;
}
else
{
lean_object* v_reuseFailAlloc_981_; 
v_reuseFailAlloc_981_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_981_, 0, v_a_975_);
v___x_980_ = v_reuseFailAlloc_981_;
goto v_reusejp_979_;
}
v_reusejp_979_:
{
v___y_932_ = v_a_948_;
v___y_933_ = v___x_973_;
v_a_934_ = v___x_980_;
goto v___jp_931_;
}
}
}
else
{
lean_object* v_a_983_; lean_object* v___x_985_; uint8_t v_isShared_986_; uint8_t v_isSharedCheck_990_; 
v_a_983_ = lean_ctor_get(v___x_974_, 0);
v_isSharedCheck_990_ = !lean_is_exclusive(v___x_974_);
if (v_isSharedCheck_990_ == 0)
{
v___x_985_ = v___x_974_;
v_isShared_986_ = v_isSharedCheck_990_;
goto v_resetjp_984_;
}
else
{
lean_inc(v_a_983_);
lean_dec(v___x_974_);
v___x_985_ = lean_box(0);
v_isShared_986_ = v_isSharedCheck_990_;
goto v_resetjp_984_;
}
v_resetjp_984_:
{
lean_object* v___x_988_; 
if (v_isShared_986_ == 0)
{
lean_ctor_set_tag(v___x_985_, 0);
v___x_988_ = v___x_985_;
goto v_reusejp_987_;
}
else
{
lean_object* v_reuseFailAlloc_989_; 
v_reuseFailAlloc_989_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_989_, 0, v_a_983_);
v___x_988_ = v_reuseFailAlloc_989_;
goto v_reusejp_987_;
}
v_reusejp_987_:
{
v___y_932_ = v_a_948_;
v___y_933_ = v___x_973_;
v_a_934_ = v___x_988_;
goto v___jp_931_;
}
}
}
}
}
else
{
lean_object* v_a_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_998_; 
lean_dec_ref(v___x_909_);
lean_dec_ref(v___x_898_);
lean_dec_ref(v___x_896_);
lean_dec_ref(v_k_886_);
v_a_991_ = lean_ctor_get(v___x_947_, 0);
v_isSharedCheck_998_ = !lean_is_exclusive(v___x_947_);
if (v_isSharedCheck_998_ == 0)
{
v___x_993_ = v___x_947_;
v_isShared_994_ = v_isSharedCheck_998_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_a_991_);
lean_dec(v___x_947_);
v___x_993_ = lean_box(0);
v_isShared_994_ = v_isSharedCheck_998_;
goto v_resetjp_992_;
}
v_resetjp_992_:
{
lean_object* v___x_996_; 
if (v_isShared_994_ == 0)
{
v___x_996_ = v___x_993_;
goto v_reusejp_995_;
}
else
{
lean_object* v_reuseFailAlloc_997_; 
v_reuseFailAlloc_997_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_997_, 0, v_a_991_);
v___x_996_ = v_reuseFailAlloc_997_;
goto v_reusejp_995_;
}
v_reusejp_995_:
{
return v___x_996_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withRuleTraceNode___redArg___boxed(lean_object* v_inst_1004_, lean_object* v_ruleName_1005_, lean_object* v_toEmoji_1006_, lean_object* v_suffix_1007_, lean_object* v_k_1008_, lean_object* v_a_1009_, lean_object* v_a_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_, lean_object* v_a_1013_, lean_object* v_a_1014_, lean_object* v_a_1015_, lean_object* v_a_1016_, lean_object* v_a_1017_){
_start:
{
lean_object* v_res_1018_; 
v_res_1018_ = lp_aesop_Aesop_withRuleTraceNode___redArg(v_inst_1004_, v_ruleName_1005_, v_toEmoji_1006_, v_suffix_1007_, v_k_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_, v_a_1014_, v_a_1015_, v_a_1016_);
lean_dec(v_a_1016_);
lean_dec_ref(v_a_1015_);
lean_dec(v_a_1014_);
lean_dec_ref(v_a_1013_);
lean_dec(v_a_1012_);
lean_dec(v_a_1011_);
lean_dec(v_a_1010_);
lean_dec_ref(v_a_1009_);
return v_res_1018_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withRuleTraceNode(lean_object* v_Q_1019_, lean_object* v_inst_1020_, lean_object* v_00_u03b1_1021_, lean_object* v_ruleName_1022_, lean_object* v_toEmoji_1023_, lean_object* v_suffix_1024_, lean_object* v_k_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_, lean_object* v_a_1031_, lean_object* v_a_1032_, lean_object* v_a_1033_){
_start:
{
lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v_options_1039_; uint8_t v_hasTrace_1040_; 
v___x_1035_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_1020_);
v___x_1036_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_1037_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_1020_);
v___x_1038_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_options_1039_ = lean_ctor_get(v_a_1032_, 2);
v_hasTrace_1040_ = lean_ctor_get_uint8(v_options_1039_, sizeof(void*)*1);
if (v_hasTrace_1040_ == 0)
{
lean_object* v___x_1041_; 
lean_dec_ref(v___x_1037_);
lean_dec_ref(v___x_1035_);
lean_dec_ref(v_suffix_1024_);
lean_dec_ref(v_toEmoji_1023_);
lean_dec_ref(v_ruleName_1022_);
lean_dec_ref(v_inst_1020_);
lean_inc(v_a_1033_);
lean_inc_ref(v_a_1032_);
lean_inc(v_a_1031_);
lean_inc_ref(v_a_1030_);
lean_inc(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
v___x_1041_ = lean_apply_9(v_k_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, lean_box(0));
return v___x_1041_;
}
else
{
lean_object* v_inheritedTraceOptions_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v_traceClass_1045_; lean_object* v___f_1046_; lean_object* v___f_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; uint8_t v___x_1052_; lean_object* v___y_1054_; lean_object* v___y_1055_; lean_object* v_a_1056_; lean_object* v___y_1071_; lean_object* v___y_1072_; lean_object* v_a_1073_; 
v_inheritedTraceOptions_1042_ = lean_ctor_get(v_a_1032_, 13);
v___x_1043_ = lean_st_ref_get(v_a_1027_);
lean_dec(v___x_1043_);
v___x_1044_ = lp_aesop_Aesop_TraceOption_steps;
v_traceClass_1045_ = lean_ctor_get(v___x_1044_, 0);
v___f_1046_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___f_1047_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
v___x_1048_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed), 16, 6);
lean_closure_set(v___x_1048_, 0, lean_box(0));
lean_closure_set(v___x_1048_, 1, v_inst_1020_);
lean_closure_set(v___x_1048_, 2, lean_box(0));
lean_closure_set(v___x_1048_, 3, v_ruleName_1022_);
lean_closure_set(v___x_1048_, 4, v_toEmoji_1023_);
lean_closure_set(v___x_1048_, 5, v_suffix_1024_);
v___x_1049_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23));
v___x_1050_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_1045_);
v___x_1051_ = l_Lean_Name_append(v___x_1050_, v_traceClass_1045_);
v___x_1052_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1042_, v_options_1039_, v___x_1051_);
lean_dec(v___x_1051_);
if (v___x_1052_ == 0)
{
lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; uint8_t v___x_1141_; 
v___x_1138_ = l_Lean_KVMap_instValueBool;
v___x_1139_ = l_Lean_trace_profiler;
v___x_1140_ = l_Lean_Option_get___redArg(v___x_1138_, v_options_1039_, v___x_1139_);
v___x_1141_ = lean_unbox(v___x_1140_);
lean_dec(v___x_1140_);
if (v___x_1141_ == 0)
{
lean_object* v___x_1142_; 
lean_dec_ref(v___x_1048_);
lean_dec_ref(v___x_1037_);
lean_dec_ref(v___x_1035_);
lean_inc(v_a_1033_);
lean_inc_ref(v_a_1032_);
lean_inc(v_a_1031_);
lean_inc_ref(v_a_1030_);
lean_inc(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
v___x_1142_ = lean_apply_9(v_k_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, lean_box(0));
return v___x_1142_;
}
else
{
goto v___jp_1084_;
}
}
else
{
goto v___jp_1084_;
}
v___jp_1053_:
{
lean_object* v___x_1057_; lean_object* v___x_1058_; double v___x_1059_; double v___x_1060_; double v___x_1061_; double v___x_1062_; double v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_18449__overap_1068_; lean_object* v___x_1069_; 
v___x_1057_ = lean_st_ref_get(v_a_1027_);
lean_dec(v___x_1057_);
v___x_1058_ = lean_io_mono_nanos_now();
v___x_1059_ = lean_float_of_nat(v___y_1055_);
v___x_1060_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_1061_ = lean_float_div(v___x_1059_, v___x_1060_);
v___x_1062_ = lean_float_of_nat(v___x_1058_);
v___x_1063_ = lean_float_div(v___x_1062_, v___x_1060_);
v___x_1064_ = lean_box_float(v___x_1061_);
v___x_1065_ = lean_box_float(v___x_1063_);
v___x_1066_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1066_, 0, v___x_1064_);
lean_ctor_set(v___x_1066_, 1, v___x_1065_);
v___x_1067_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1067_, 0, v_a_1056_);
lean_ctor_set(v___x_1067_, 1, v___x_1066_);
lean_inc(v_traceClass_1045_);
v___x_18449__overap_1068_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1035_, v___x_1036_, v___x_1037_, v___f_1046_, lean_box(0), v___x_1038_, v___f_1047_, v_traceClass_1045_, v_hasTrace_1040_, v___x_1049_, v_options_1039_, v___x_1052_, v___y_1054_, v___x_1048_, v___x_1067_);
lean_inc(v_a_1033_);
lean_inc_ref(v_a_1032_);
lean_inc(v_a_1031_);
lean_inc_ref(v_a_1030_);
lean_inc(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
v___x_1069_ = lean_apply_9(v___x_18449__overap_1068_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, lean_box(0));
return v___x_1069_;
}
v___jp_1070_:
{
lean_object* v___x_1074_; lean_object* v___x_1075_; double v___x_1076_; double v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_18429__overap_1082_; lean_object* v___x_1083_; 
v___x_1074_ = lean_st_ref_get(v_a_1027_);
lean_dec(v___x_1074_);
v___x_1075_ = lean_io_get_num_heartbeats();
v___x_1076_ = lean_float_of_nat(v___y_1072_);
v___x_1077_ = lean_float_of_nat(v___x_1075_);
v___x_1078_ = lean_box_float(v___x_1076_);
v___x_1079_ = lean_box_float(v___x_1077_);
v___x_1080_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1080_, 0, v___x_1078_);
lean_ctor_set(v___x_1080_, 1, v___x_1079_);
v___x_1081_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1081_, 0, v_a_1073_);
lean_ctor_set(v___x_1081_, 1, v___x_1080_);
lean_inc(v_traceClass_1045_);
v___x_18429__overap_1082_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1035_, v___x_1036_, v___x_1037_, v___f_1046_, lean_box(0), v___x_1038_, v___f_1047_, v_traceClass_1045_, v_hasTrace_1040_, v___x_1049_, v_options_1039_, v___x_1052_, v___y_1071_, v___x_1048_, v___x_1081_);
lean_inc(v_a_1033_);
lean_inc_ref(v_a_1032_);
lean_inc(v_a_1031_);
lean_inc_ref(v_a_1030_);
lean_inc(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
v___x_1083_ = lean_apply_9(v___x_18429__overap_1082_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, lean_box(0));
return v___x_1083_;
}
v___jp_1084_:
{
lean_object* v___x_18451__overap_1085_; lean_object* v___x_1086_; 
lean_inc_ref(v___x_1035_);
v___x_18451__overap_1085_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_1035_, v___x_1036_);
lean_inc(v_a_1033_);
lean_inc_ref(v_a_1032_);
lean_inc(v_a_1031_);
lean_inc_ref(v_a_1030_);
lean_inc(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
v___x_1086_ = lean_apply_9(v___x_18451__overap_1085_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, lean_box(0));
if (lean_obj_tag(v___x_1086_) == 0)
{
lean_object* v_a_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; uint8_t v___x_1091_; 
v_a_1087_ = lean_ctor_get(v___x_1086_, 0);
lean_inc(v_a_1087_);
lean_dec_ref_known(v___x_1086_, 1);
v___x_1088_ = l_Lean_KVMap_instValueBool;
v___x_1089_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1090_ = l_Lean_Option_get___redArg(v___x_1088_, v_options_1039_, v___x_1089_);
v___x_1091_ = lean_unbox(v___x_1090_);
lean_dec(v___x_1090_);
if (v___x_1091_ == 0)
{
lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; 
v___x_1092_ = lean_st_ref_get(v_a_1027_);
lean_dec(v___x_1092_);
v___x_1093_ = lean_io_mono_nanos_now();
lean_inc(v_a_1033_);
lean_inc_ref(v_a_1032_);
lean_inc(v_a_1031_);
lean_inc_ref(v_a_1030_);
lean_inc(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
v___x_1094_ = lean_apply_9(v_k_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, lean_box(0));
if (lean_obj_tag(v___x_1094_) == 0)
{
lean_object* v_a_1095_; lean_object* v___x_1097_; uint8_t v_isShared_1098_; uint8_t v_isSharedCheck_1102_; 
v_a_1095_ = lean_ctor_get(v___x_1094_, 0);
v_isSharedCheck_1102_ = !lean_is_exclusive(v___x_1094_);
if (v_isSharedCheck_1102_ == 0)
{
v___x_1097_ = v___x_1094_;
v_isShared_1098_ = v_isSharedCheck_1102_;
goto v_resetjp_1096_;
}
else
{
lean_inc(v_a_1095_);
lean_dec(v___x_1094_);
v___x_1097_ = lean_box(0);
v_isShared_1098_ = v_isSharedCheck_1102_;
goto v_resetjp_1096_;
}
v_resetjp_1096_:
{
lean_object* v___x_1100_; 
if (v_isShared_1098_ == 0)
{
lean_ctor_set_tag(v___x_1097_, 1);
v___x_1100_ = v___x_1097_;
goto v_reusejp_1099_;
}
else
{
lean_object* v_reuseFailAlloc_1101_; 
v_reuseFailAlloc_1101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1101_, 0, v_a_1095_);
v___x_1100_ = v_reuseFailAlloc_1101_;
goto v_reusejp_1099_;
}
v_reusejp_1099_:
{
v___y_1054_ = v_a_1087_;
v___y_1055_ = v___x_1093_;
v_a_1056_ = v___x_1100_;
goto v___jp_1053_;
}
}
}
else
{
lean_object* v_a_1103_; lean_object* v___x_1105_; uint8_t v_isShared_1106_; uint8_t v_isSharedCheck_1110_; 
v_a_1103_ = lean_ctor_get(v___x_1094_, 0);
v_isSharedCheck_1110_ = !lean_is_exclusive(v___x_1094_);
if (v_isSharedCheck_1110_ == 0)
{
v___x_1105_ = v___x_1094_;
v_isShared_1106_ = v_isSharedCheck_1110_;
goto v_resetjp_1104_;
}
else
{
lean_inc(v_a_1103_);
lean_dec(v___x_1094_);
v___x_1105_ = lean_box(0);
v_isShared_1106_ = v_isSharedCheck_1110_;
goto v_resetjp_1104_;
}
v_resetjp_1104_:
{
lean_object* v___x_1108_; 
if (v_isShared_1106_ == 0)
{
lean_ctor_set_tag(v___x_1105_, 0);
v___x_1108_ = v___x_1105_;
goto v_reusejp_1107_;
}
else
{
lean_object* v_reuseFailAlloc_1109_; 
v_reuseFailAlloc_1109_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1109_, 0, v_a_1103_);
v___x_1108_ = v_reuseFailAlloc_1109_;
goto v_reusejp_1107_;
}
v_reusejp_1107_:
{
v___y_1054_ = v_a_1087_;
v___y_1055_ = v___x_1093_;
v_a_1056_ = v___x_1108_;
goto v___jp_1053_;
}
}
}
}
else
{
lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; 
v___x_1111_ = lean_st_ref_get(v_a_1027_);
lean_dec(v___x_1111_);
v___x_1112_ = lean_io_get_num_heartbeats();
lean_inc(v_a_1033_);
lean_inc_ref(v_a_1032_);
lean_inc(v_a_1031_);
lean_inc_ref(v_a_1030_);
lean_inc(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
v___x_1113_ = lean_apply_9(v_k_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_, v_a_1033_, lean_box(0));
if (lean_obj_tag(v___x_1113_) == 0)
{
lean_object* v_a_1114_; lean_object* v___x_1116_; uint8_t v_isShared_1117_; uint8_t v_isSharedCheck_1121_; 
v_a_1114_ = lean_ctor_get(v___x_1113_, 0);
v_isSharedCheck_1121_ = !lean_is_exclusive(v___x_1113_);
if (v_isSharedCheck_1121_ == 0)
{
v___x_1116_ = v___x_1113_;
v_isShared_1117_ = v_isSharedCheck_1121_;
goto v_resetjp_1115_;
}
else
{
lean_inc(v_a_1114_);
lean_dec(v___x_1113_);
v___x_1116_ = lean_box(0);
v_isShared_1117_ = v_isSharedCheck_1121_;
goto v_resetjp_1115_;
}
v_resetjp_1115_:
{
lean_object* v___x_1119_; 
if (v_isShared_1117_ == 0)
{
lean_ctor_set_tag(v___x_1116_, 1);
v___x_1119_ = v___x_1116_;
goto v_reusejp_1118_;
}
else
{
lean_object* v_reuseFailAlloc_1120_; 
v_reuseFailAlloc_1120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1120_, 0, v_a_1114_);
v___x_1119_ = v_reuseFailAlloc_1120_;
goto v_reusejp_1118_;
}
v_reusejp_1118_:
{
v___y_1071_ = v_a_1087_;
v___y_1072_ = v___x_1112_;
v_a_1073_ = v___x_1119_;
goto v___jp_1070_;
}
}
}
else
{
lean_object* v_a_1122_; lean_object* v___x_1124_; uint8_t v_isShared_1125_; uint8_t v_isSharedCheck_1129_; 
v_a_1122_ = lean_ctor_get(v___x_1113_, 0);
v_isSharedCheck_1129_ = !lean_is_exclusive(v___x_1113_);
if (v_isSharedCheck_1129_ == 0)
{
v___x_1124_ = v___x_1113_;
v_isShared_1125_ = v_isSharedCheck_1129_;
goto v_resetjp_1123_;
}
else
{
lean_inc(v_a_1122_);
lean_dec(v___x_1113_);
v___x_1124_ = lean_box(0);
v_isShared_1125_ = v_isSharedCheck_1129_;
goto v_resetjp_1123_;
}
v_resetjp_1123_:
{
lean_object* v___x_1127_; 
if (v_isShared_1125_ == 0)
{
lean_ctor_set_tag(v___x_1124_, 0);
v___x_1127_ = v___x_1124_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1128_; 
v_reuseFailAlloc_1128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1128_, 0, v_a_1122_);
v___x_1127_ = v_reuseFailAlloc_1128_;
goto v_reusejp_1126_;
}
v_reusejp_1126_:
{
v___y_1071_ = v_a_1087_;
v___y_1072_ = v___x_1112_;
v_a_1073_ = v___x_1127_;
goto v___jp_1070_;
}
}
}
}
}
else
{
lean_object* v_a_1130_; lean_object* v___x_1132_; uint8_t v_isShared_1133_; uint8_t v_isSharedCheck_1137_; 
lean_dec_ref(v___x_1048_);
lean_dec_ref(v___x_1037_);
lean_dec_ref(v___x_1035_);
lean_dec_ref(v_k_1025_);
v_a_1130_ = lean_ctor_get(v___x_1086_, 0);
v_isSharedCheck_1137_ = !lean_is_exclusive(v___x_1086_);
if (v_isSharedCheck_1137_ == 0)
{
v___x_1132_ = v___x_1086_;
v_isShared_1133_ = v_isSharedCheck_1137_;
goto v_resetjp_1131_;
}
else
{
lean_inc(v_a_1130_);
lean_dec(v___x_1086_);
v___x_1132_ = lean_box(0);
v_isShared_1133_ = v_isSharedCheck_1137_;
goto v_resetjp_1131_;
}
v_resetjp_1131_:
{
lean_object* v___x_1135_; 
if (v_isShared_1133_ == 0)
{
v___x_1135_ = v___x_1132_;
goto v_reusejp_1134_;
}
else
{
lean_object* v_reuseFailAlloc_1136_; 
v_reuseFailAlloc_1136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1136_, 0, v_a_1130_);
v___x_1135_ = v_reuseFailAlloc_1136_;
goto v_reusejp_1134_;
}
v_reusejp_1134_:
{
return v___x_1135_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withRuleTraceNode___boxed(lean_object* v_Q_1143_, lean_object* v_inst_1144_, lean_object* v_00_u03b1_1145_, lean_object* v_ruleName_1146_, lean_object* v_toEmoji_1147_, lean_object* v_suffix_1148_, lean_object* v_k_1149_, lean_object* v_a_1150_, lean_object* v_a_1151_, lean_object* v_a_1152_, lean_object* v_a_1153_, lean_object* v_a_1154_, lean_object* v_a_1155_, lean_object* v_a_1156_, lean_object* v_a_1157_, lean_object* v_a_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_aesop_Aesop_withRuleTraceNode(v_Q_1143_, v_inst_1144_, v_00_u03b1_1145_, v_ruleName_1146_, v_toEmoji_1147_, v_suffix_1148_, v_k_1149_, v_a_1150_, v_a_1151_, v_a_1152_, v_a_1153_, v_a_1154_, v_a_1155_, v_a_1156_, v_a_1157_);
lean_dec(v_a_1157_);
lean_dec_ref(v_a_1156_);
lean_dec(v_a_1155_);
lean_dec_ref(v_a_1154_);
lean_dec(v_a_1153_);
lean_dec(v_a_1152_);
lean_dec(v_a_1151_);
lean_dec_ref(v_a_1150_);
return v_res_1159_;
}
}
static lean_object* _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__0(void){
_start:
{
lean_object* v___x_1160_; lean_object* v___x_1161_; 
v___x_1160_ = lp_aesop_Aesop_BaseM_instMonadStats;
v___x_1161_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_1160_);
return v___x_1161_;
}
}
static lean_object* _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__1(void){
_start:
{
lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1162_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleCore___redArg___closed__0, &lp_aesop_Aesop_runRegularRuleCore___redArg___closed__0_once, _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__0);
v___x_1163_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_1162_);
return v___x_1163_;
}
}
static lean_object* _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2(void){
_start:
{
lean_object* v___x_1164_; lean_object* v___x_1165_; 
v___x_1164_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleCore___redArg___closed__1, &lp_aesop_Aesop_runRegularRuleCore___redArg___closed__1_once, _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__1);
v___x_1165_ = lp_aesop_Aesop_instMonadStatsReaderT___redArg(v___x_1164_);
return v___x_1165_;
}
}
static lean_object* _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__4(void){
_start:
{
lean_object* v___x_1167_; lean_object* v___x_1168_; 
v___x_1167_ = ((lean_object*)(lp_aesop_Aesop_runRegularRuleCore___redArg___closed__3));
v___x_1168_ = l_Lean_stringToMessageData(v___x_1167_);
return v___x_1168_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg(lean_object* v_inst_1169_, lean_object* v_parentRef_1170_, lean_object* v_rule_1171_, lean_object* v_indexMatchLocations_1172_, lean_object* v_patternSubsts_x3f_1173_, lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_, lean_object* v_a_1181_){
_start:
{
lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v_toMonadOptions_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v_options_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; 
v___x_1189_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_1169_);
v___x_1190_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2, &lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2_once, _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2);
v_toMonadOptions_1191_ = lean_ctor_get(v___x_1190_, 0);
v___x_1192_ = lean_st_ref_get(v_a_1175_);
lean_dec(v___x_1192_);
v___x_1193_ = lean_st_ref_get(v_parentRef_1170_);
v___x_1194_ = lean_st_ref_get(v_a_1175_);
lean_dec(v___x_1194_);
v_options_1195_ = lean_ctor_get(v_a_1174_, 2);
v___x_1196_ = lp_aesop_Aesop_RegularRule_tac(v_rule_1171_);
v___x_1197_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTacDescr_run___boxed), 8, 1);
lean_closure_set(v___x_1197_, 0, v___x_1196_);
v___x_1198_ = lp_aesop_Aesop_RegularRule_name(v_rule_1171_);
lean_inc_ref(v_options_1195_);
v___x_1199_ = lp_aesop_Aesop_runRegularRuleTac(v___x_1193_, v___x_1197_, v___x_1198_, v_indexMatchLocations_1172_, v_patternSubsts_x3f_1173_, v_options_1195_, v_a_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_);
if (lean_obj_tag(v___x_1199_) == 0)
{
lean_object* v_a_1200_; lean_object* v___x_1202_; uint8_t v_isShared_1203_; uint8_t v_isSharedCheck_1275_; 
v_a_1200_ = lean_ctor_get(v___x_1199_, 0);
v_isSharedCheck_1275_ = !lean_is_exclusive(v___x_1199_);
if (v_isSharedCheck_1275_ == 0)
{
v___x_1202_ = v___x_1199_;
v_isShared_1203_ = v_isSharedCheck_1275_;
goto v_resetjp_1201_;
}
else
{
lean_inc(v_a_1200_);
lean_dec(v___x_1199_);
v___x_1202_ = lean_box(0);
v_isShared_1203_ = v_isSharedCheck_1275_;
goto v_resetjp_1201_;
}
v_resetjp_1201_:
{
if (lean_obj_tag(v_a_1200_) == 0)
{
lean_object* v_a_1204_; lean_object* v___x_1205_; lean_object* v___x_10611__overap_1206_; lean_object* v___x_1207_; 
lean_del_object(v___x_1202_);
v_a_1204_ = lean_ctor_get(v_a_1200_, 0);
lean_inc(v_a_1204_);
lean_dec_ref_known(v_a_1200_, 1);
v___x_1205_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_toMonadOptions_1191_);
lean_inc_ref(v___x_1189_);
v___x_10611__overap_1206_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1189_, v_toMonadOptions_1191_, v___x_1205_);
lean_inc(v_a_1181_);
lean_inc_ref(v_a_1180_);
lean_inc(v_a_1179_);
lean_inc_ref(v_a_1178_);
lean_inc(v_a_1177_);
lean_inc(v_a_1176_);
lean_inc(v_a_1175_);
lean_inc_ref(v_a_1174_);
v___x_1207_ = lean_apply_9(v___x_10611__overap_1206_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_, lean_box(0));
if (lean_obj_tag(v___x_1207_) == 0)
{
lean_object* v_a_1208_; uint8_t v___x_1209_; 
v_a_1208_ = lean_ctor_get(v___x_1207_, 0);
lean_inc(v_a_1208_);
lean_dec_ref_known(v___x_1207_, 1);
v___x_1209_ = lean_unbox(v_a_1208_);
lean_dec(v_a_1208_);
if (v___x_1209_ == 0)
{
lean_dec(v_a_1204_);
lean_dec_ref(v___x_1189_);
goto v___jp_1183_;
}
else
{
lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v_traceClass_1212_; lean_object* v___f_1213_; lean_object* v___x_1214_; lean_object* v___x_10743__overap_1215_; lean_object* v___x_1216_; 
v___x_1210_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_1211_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_1169_);
v_traceClass_1212_ = lean_ctor_get(v___x_1205_, 0);
v___f_1213_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___x_1214_ = l_Lean_Exception_toMessageData(v_a_1204_);
lean_inc(v_traceClass_1212_);
v___x_10743__overap_1215_ = l_Lean_addTrace___redArg(v___x_1189_, v___x_1210_, v___x_1211_, v___f_1213_, v_traceClass_1212_, v___x_1214_);
lean_inc(v_a_1181_);
lean_inc_ref(v_a_1180_);
lean_inc(v_a_1179_);
lean_inc_ref(v_a_1178_);
lean_inc(v_a_1177_);
lean_inc(v_a_1176_);
lean_inc(v_a_1175_);
lean_inc_ref(v_a_1174_);
v___x_1216_ = lean_apply_9(v___x_10743__overap_1215_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_, lean_box(0));
if (lean_obj_tag(v___x_1216_) == 0)
{
lean_dec_ref_known(v___x_1216_, 1);
goto v___jp_1183_;
}
else
{
lean_object* v_a_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1224_; 
v_a_1217_ = lean_ctor_get(v___x_1216_, 0);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1216_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1219_ = v___x_1216_;
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_a_1217_);
lean_dec(v___x_1216_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v___x_1222_; 
if (v_isShared_1220_ == 0)
{
v___x_1222_ = v___x_1219_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v_a_1217_);
v___x_1222_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
return v___x_1222_;
}
}
}
}
}
else
{
lean_object* v_a_1225_; lean_object* v___x_1227_; uint8_t v_isShared_1228_; uint8_t v_isSharedCheck_1232_; 
lean_dec(v_a_1204_);
lean_dec_ref(v___x_1189_);
v_a_1225_ = lean_ctor_get(v___x_1207_, 0);
v_isSharedCheck_1232_ = !lean_is_exclusive(v___x_1207_);
if (v_isSharedCheck_1232_ == 0)
{
v___x_1227_ = v___x_1207_;
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
else
{
lean_inc(v_a_1225_);
lean_dec(v___x_1207_);
v___x_1227_ = lean_box(0);
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
v_resetjp_1226_:
{
lean_object* v___x_1230_; 
if (v_isShared_1228_ == 0)
{
v___x_1230_ = v___x_1227_;
goto v_reusejp_1229_;
}
else
{
lean_object* v_reuseFailAlloc_1231_; 
v_reuseFailAlloc_1231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1231_, 0, v_a_1225_);
v___x_1230_ = v_reuseFailAlloc_1231_;
goto v_reusejp_1229_;
}
v_reusejp_1229_:
{
return v___x_1230_;
}
}
}
}
else
{
lean_object* v_a_1233_; lean_object* v___x_1235_; uint8_t v_isShared_1236_; uint8_t v_isSharedCheck_1274_; 
v_a_1233_ = lean_ctor_get(v_a_1200_, 0);
v_isSharedCheck_1274_ = !lean_is_exclusive(v_a_1200_);
if (v_isSharedCheck_1274_ == 0)
{
v___x_1235_ = v_a_1200_;
v_isShared_1236_ = v_isSharedCheck_1274_;
goto v_resetjp_1234_;
}
else
{
lean_inc(v_a_1233_);
lean_dec(v_a_1200_);
v___x_1235_ = lean_box(0);
v_isShared_1236_ = v_isSharedCheck_1274_;
goto v_resetjp_1234_;
}
v_resetjp_1234_:
{
lean_object* v___x_1237_; lean_object* v___x_1238_; uint8_t v___x_1239_; 
v___x_1237_ = lean_array_get_size(v_a_1233_);
v___x_1238_ = lean_unsigned_to_nat(0u);
v___x_1239_ = lean_nat_dec_eq(v___x_1237_, v___x_1238_);
if (v___x_1239_ == 0)
{
lean_object* v___x_1241_; 
lean_dec_ref(v___x_1189_);
if (v_isShared_1236_ == 0)
{
v___x_1241_ = v___x_1235_;
goto v_reusejp_1240_;
}
else
{
lean_object* v_reuseFailAlloc_1245_; 
v_reuseFailAlloc_1245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1245_, 0, v_a_1233_);
v___x_1241_ = v_reuseFailAlloc_1245_;
goto v_reusejp_1240_;
}
v_reusejp_1240_:
{
lean_object* v___x_1243_; 
if (v_isShared_1203_ == 0)
{
lean_ctor_set(v___x_1202_, 0, v___x_1241_);
v___x_1243_ = v___x_1202_;
goto v_reusejp_1242_;
}
else
{
lean_object* v_reuseFailAlloc_1244_; 
v_reuseFailAlloc_1244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1244_, 0, v___x_1241_);
v___x_1243_ = v_reuseFailAlloc_1244_;
goto v_reusejp_1242_;
}
v_reusejp_1242_:
{
return v___x_1243_;
}
}
}
else
{
lean_object* v___x_1246_; lean_object* v___x_10671__overap_1247_; lean_object* v___x_1248_; 
lean_del_object(v___x_1235_);
lean_dec(v_a_1233_);
lean_del_object(v___x_1202_);
v___x_1246_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_toMonadOptions_1191_);
lean_inc_ref(v___x_1189_);
v___x_10671__overap_1247_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1189_, v_toMonadOptions_1191_, v___x_1246_);
lean_inc(v_a_1181_);
lean_inc_ref(v_a_1180_);
lean_inc(v_a_1179_);
lean_inc_ref(v_a_1178_);
lean_inc(v_a_1177_);
lean_inc(v_a_1176_);
lean_inc(v_a_1175_);
lean_inc_ref(v_a_1174_);
v___x_1248_ = lean_apply_9(v___x_10671__overap_1247_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_, lean_box(0));
if (lean_obj_tag(v___x_1248_) == 0)
{
lean_object* v_a_1249_; uint8_t v___x_1250_; 
v_a_1249_ = lean_ctor_get(v___x_1248_, 0);
lean_inc(v_a_1249_);
lean_dec_ref_known(v___x_1248_, 1);
v___x_1250_ = lean_unbox(v_a_1249_);
lean_dec(v_a_1249_);
if (v___x_1250_ == 0)
{
lean_dec_ref(v___x_1189_);
goto v___jp_1186_;
}
else
{
lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v_traceClass_1253_; lean_object* v___f_1254_; lean_object* v___x_1255_; lean_object* v___x_10768__overap_1256_; lean_object* v___x_1257_; 
v___x_1251_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_1252_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_1169_);
v_traceClass_1253_ = lean_ctor_get(v___x_1246_, 0);
v___f_1254_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___x_1255_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleCore___redArg___closed__4, &lp_aesop_Aesop_runRegularRuleCore___redArg___closed__4_once, _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__4);
lean_inc(v_traceClass_1253_);
v___x_10768__overap_1256_ = l_Lean_addTrace___redArg(v___x_1189_, v___x_1251_, v___x_1252_, v___f_1254_, v_traceClass_1253_, v___x_1255_);
lean_inc(v_a_1181_);
lean_inc_ref(v_a_1180_);
lean_inc(v_a_1179_);
lean_inc_ref(v_a_1178_);
lean_inc(v_a_1177_);
lean_inc(v_a_1176_);
lean_inc(v_a_1175_);
lean_inc_ref(v_a_1174_);
v___x_1257_ = lean_apply_9(v___x_10768__overap_1256_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_, lean_box(0));
if (lean_obj_tag(v___x_1257_) == 0)
{
lean_dec_ref_known(v___x_1257_, 1);
goto v___jp_1186_;
}
else
{
lean_object* v_a_1258_; lean_object* v___x_1260_; uint8_t v_isShared_1261_; uint8_t v_isSharedCheck_1265_; 
v_a_1258_ = lean_ctor_get(v___x_1257_, 0);
v_isSharedCheck_1265_ = !lean_is_exclusive(v___x_1257_);
if (v_isSharedCheck_1265_ == 0)
{
v___x_1260_ = v___x_1257_;
v_isShared_1261_ = v_isSharedCheck_1265_;
goto v_resetjp_1259_;
}
else
{
lean_inc(v_a_1258_);
lean_dec(v___x_1257_);
v___x_1260_ = lean_box(0);
v_isShared_1261_ = v_isSharedCheck_1265_;
goto v_resetjp_1259_;
}
v_resetjp_1259_:
{
lean_object* v___x_1263_; 
if (v_isShared_1261_ == 0)
{
v___x_1263_ = v___x_1260_;
goto v_reusejp_1262_;
}
else
{
lean_object* v_reuseFailAlloc_1264_; 
v_reuseFailAlloc_1264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1264_, 0, v_a_1258_);
v___x_1263_ = v_reuseFailAlloc_1264_;
goto v_reusejp_1262_;
}
v_reusejp_1262_:
{
return v___x_1263_;
}
}
}
}
}
else
{
lean_object* v_a_1266_; lean_object* v___x_1268_; uint8_t v_isShared_1269_; uint8_t v_isSharedCheck_1273_; 
lean_dec_ref(v___x_1189_);
v_a_1266_ = lean_ctor_get(v___x_1248_, 0);
v_isSharedCheck_1273_ = !lean_is_exclusive(v___x_1248_);
if (v_isSharedCheck_1273_ == 0)
{
v___x_1268_ = v___x_1248_;
v_isShared_1269_ = v_isSharedCheck_1273_;
goto v_resetjp_1267_;
}
else
{
lean_inc(v_a_1266_);
lean_dec(v___x_1248_);
v___x_1268_ = lean_box(0);
v_isShared_1269_ = v_isSharedCheck_1273_;
goto v_resetjp_1267_;
}
v_resetjp_1267_:
{
lean_object* v___x_1271_; 
if (v_isShared_1269_ == 0)
{
v___x_1271_ = v___x_1268_;
goto v_reusejp_1270_;
}
else
{
lean_object* v_reuseFailAlloc_1272_; 
v_reuseFailAlloc_1272_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1272_, 0, v_a_1266_);
v___x_1271_ = v_reuseFailAlloc_1272_;
goto v_reusejp_1270_;
}
v_reusejp_1270_:
{
return v___x_1271_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_1276_; lean_object* v___x_1278_; uint8_t v_isShared_1279_; uint8_t v_isSharedCheck_1283_; 
lean_dec_ref(v___x_1189_);
v_a_1276_ = lean_ctor_get(v___x_1199_, 0);
v_isSharedCheck_1283_ = !lean_is_exclusive(v___x_1199_);
if (v_isSharedCheck_1283_ == 0)
{
v___x_1278_ = v___x_1199_;
v_isShared_1279_ = v_isSharedCheck_1283_;
goto v_resetjp_1277_;
}
else
{
lean_inc(v_a_1276_);
lean_dec(v___x_1199_);
v___x_1278_ = lean_box(0);
v_isShared_1279_ = v_isSharedCheck_1283_;
goto v_resetjp_1277_;
}
v_resetjp_1277_:
{
lean_object* v___x_1281_; 
if (v_isShared_1279_ == 0)
{
v___x_1281_ = v___x_1278_;
goto v_reusejp_1280_;
}
else
{
lean_object* v_reuseFailAlloc_1282_; 
v_reuseFailAlloc_1282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1282_, 0, v_a_1276_);
v___x_1281_ = v_reuseFailAlloc_1282_;
goto v_reusejp_1280_;
}
v_reusejp_1280_:
{
return v___x_1281_;
}
}
}
v___jp_1183_:
{
lean_object* v___x_1184_; lean_object* v___x_1185_; 
v___x_1184_ = lean_box(0);
v___x_1185_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1185_, 0, v___x_1184_);
return v___x_1185_;
}
v___jp_1186_:
{
lean_object* v___x_1187_; lean_object* v___x_1188_; 
v___x_1187_ = lean_box(0);
v___x_1188_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1188_, 0, v___x_1187_);
return v___x_1188_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleCore___redArg___boxed(lean_object* v_inst_1284_, lean_object* v_parentRef_1285_, lean_object* v_rule_1286_, lean_object* v_indexMatchLocations_1287_, lean_object* v_patternSubsts_x3f_1288_, lean_object* v_a_1289_, lean_object* v_a_1290_, lean_object* v_a_1291_, lean_object* v_a_1292_, lean_object* v_a_1293_, lean_object* v_a_1294_, lean_object* v_a_1295_, lean_object* v_a_1296_, lean_object* v_a_1297_){
_start:
{
lean_object* v_res_1298_; 
v_res_1298_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1284_, v_parentRef_1285_, v_rule_1286_, v_indexMatchLocations_1287_, v_patternSubsts_x3f_1288_, v_a_1289_, v_a_1290_, v_a_1291_, v_a_1292_, v_a_1293_, v_a_1294_, v_a_1295_, v_a_1296_);
lean_dec(v_a_1296_);
lean_dec_ref(v_a_1295_);
lean_dec(v_a_1294_);
lean_dec_ref(v_a_1293_);
lean_dec(v_a_1292_);
lean_dec(v_a_1291_);
lean_dec(v_a_1290_);
lean_dec_ref(v_a_1289_);
lean_dec_ref(v_rule_1286_);
lean_dec(v_parentRef_1285_);
lean_dec_ref(v_inst_1284_);
return v_res_1298_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleCore(lean_object* v_Q_1299_, lean_object* v_inst_1300_, lean_object* v_parentRef_1301_, lean_object* v_rule_1302_, lean_object* v_indexMatchLocations_1303_, lean_object* v_patternSubsts_x3f_1304_, lean_object* v_a_1305_, lean_object* v_a_1306_, lean_object* v_a_1307_, lean_object* v_a_1308_, lean_object* v_a_1309_, lean_object* v_a_1310_, lean_object* v_a_1311_, lean_object* v_a_1312_){
_start:
{
lean_object* v___x_1314_; 
v___x_1314_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1300_, v_parentRef_1301_, v_rule_1302_, v_indexMatchLocations_1303_, v_patternSubsts_x3f_1304_, v_a_1305_, v_a_1306_, v_a_1307_, v_a_1308_, v_a_1309_, v_a_1310_, v_a_1311_, v_a_1312_);
return v___x_1314_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runRegularRuleCore___boxed(lean_object* v_Q_1315_, lean_object* v_inst_1316_, lean_object* v_parentRef_1317_, lean_object* v_rule_1318_, lean_object* v_indexMatchLocations_1319_, lean_object* v_patternSubsts_x3f_1320_, lean_object* v_a_1321_, lean_object* v_a_1322_, lean_object* v_a_1323_, lean_object* v_a_1324_, lean_object* v_a_1325_, lean_object* v_a_1326_, lean_object* v_a_1327_, lean_object* v_a_1328_, lean_object* v_a_1329_){
_start:
{
lean_object* v_res_1330_; 
v_res_1330_ = lp_aesop_Aesop_runRegularRuleCore(v_Q_1315_, v_inst_1316_, v_parentRef_1317_, v_rule_1318_, v_indexMatchLocations_1319_, v_patternSubsts_x3f_1320_, v_a_1321_, v_a_1322_, v_a_1323_, v_a_1324_, v_a_1325_, v_a_1326_, v_a_1327_, v_a_1328_);
lean_dec(v_a_1328_);
lean_dec_ref(v_a_1327_);
lean_dec(v_a_1326_);
lean_dec_ref(v_a_1325_);
lean_dec(v_a_1324_);
lean_dec(v_a_1323_);
lean_dec(v_a_1322_);
lean_dec_ref(v_a_1321_);
lean_dec_ref(v_rule_1318_);
lean_dec(v_parentRef_1317_);
lean_dec_ref(v_inst_1316_);
return v_res_1330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__0(lean_object* v___x_1331_, lean_object* v_x_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_){
_start:
{
lean_object* v___x_1338_; lean_object* v___x_45013__overap_1339_; lean_object* v___x_1340_; 
v___x_1338_ = l_Lean_Meta_instMonadMCtxMetaM;
v___x_45013__overap_1339_ = l_Lean_MVarId_isAssignedOrDelayedAssigned___redArg(v___x_1331_, v___x_1338_, v_x_1332_);
lean_inc(v___y_1336_);
lean_inc_ref(v___y_1335_);
lean_inc(v___y_1334_);
lean_inc_ref(v___y_1333_);
v___x_1340_ = lean_apply_5(v___x_45013__overap_1339_, v___y_1333_, v___y_1334_, v___y_1335_, v___y_1336_, lean_box(0));
return v___x_1340_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__0___boxed(lean_object* v___x_1341_, lean_object* v_x_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_){
_start:
{
lean_object* v_res_1348_; 
v_res_1348_ = lp_aesop_Aesop_runSafeRule___redArg___lam__0(v___x_1341_, v_x_1342_, v___y_1343_, v___y_1344_, v___y_1345_, v___y_1346_);
lean_dec(v___y_1346_);
lean_dec_ref(v___y_1345_);
lean_dec(v___y_1344_);
lean_dec_ref(v___y_1343_);
return v_res_1348_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__1(lean_object* v_mvars_1349_, lean_object* v___x_1350_, lean_object* v___f_1351_, lean_object* v_rapp_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_, lean_object* v___y_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_){
_start:
{
lean_object* v___x_1362_; lean_object* v_postState_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; 
v___x_1362_ = lean_st_ref_get(v___y_1354_);
lean_dec(v___x_1362_);
v_postState_1363_ = lean_ctor_get(v_rapp_1352_, 1);
lean_inc_ref(v_postState_1363_);
lean_dec_ref(v_rapp_1352_);
v___x_1364_ = lean_array_get_size(v_mvars_1349_);
v___x_1365_ = lean_unsigned_to_nat(0u);
v___x_1366_ = lp_aesop_Aesop_UnorderedArraySet_anyM___redArg(v___x_1350_, v___f_1351_, v_mvars_1349_, v___x_1365_, v___x_1364_);
v___x_1367_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_postState_1363_, v___x_1366_, v___y_1357_, v___y_1358_, v___y_1359_, v___y_1360_);
return v___x_1367_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__1___boxed(lean_object* v_mvars_1368_, lean_object* v___x_1369_, lean_object* v___f_1370_, lean_object* v_rapp_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_){
_start:
{
lean_object* v_res_1381_; 
v_res_1381_ = lp_aesop_Aesop_runSafeRule___redArg___lam__1(v_mvars_1368_, v___x_1369_, v___f_1370_, v_rapp_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_, v___y_1378_, v___y_1379_);
lean_dec(v___y_1379_);
lean_dec_ref(v___y_1378_);
lean_dec(v___y_1377_);
lean_dec_ref(v___y_1376_);
lean_dec(v___y_1375_);
lean_dec(v___y_1374_);
lean_dec(v___y_1373_);
lean_dec_ref(v___y_1372_);
return v_res_1381_;
}
}
static lean_object* _init_lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2(void){
_start:
{
lean_object* v___x_1385_; lean_object* v___x_1386_; 
v___x_1385_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__1));
v___x_1386_ = l_Lean_stringToMessageData(v___x_1385_);
return v___x_1386_;
}
}
static lean_object* _init_lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4(void){
_start:
{
lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1388_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__3));
v___x_1389_ = l_Lean_stringToMessageData(v___x_1388_);
return v___x_1389_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2(lean_object* v_parentRef_1390_, lean_object* v___x_1391_, lean_object* v___x_1392_, lean_object* v_rule_1393_, lean_object* v___x_1394_, lean_object* v___f_1395_, lean_object* v___x_1396_, lean_object* v___x_1397_, lean_object* v___f_1398_, lean_object* v___f_1399_, lean_object* v_inst_1400_, lean_object* v___x_1401_, lean_object* v___f_1402_, lean_object* v_____x_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_){
_start:
{
lean_object* v___y_1414_; 
if (lean_obj_tag(v_____x_1403_) == 1)
{
lean_object* v_val_1426_; lean_object* v___x_1428_; uint8_t v_isShared_1429_; uint8_t v_isSharedCheck_1560_; 
v_val_1426_ = lean_ctor_get(v_____x_1403_, 0);
v_isSharedCheck_1560_ = !lean_is_exclusive(v_____x_1403_);
if (v_isSharedCheck_1560_ == 0)
{
v___x_1428_ = v_____x_1403_;
v_isShared_1429_ = v_isSharedCheck_1560_;
goto v_resetjp_1427_;
}
else
{
lean_inc(v_val_1426_);
lean_dec(v_____x_1403_);
v___x_1428_ = lean_box(0);
v_isShared_1429_ = v_isSharedCheck_1560_;
goto v_resetjp_1427_;
}
v_resetjp_1427_:
{
lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; uint8_t v___x_1460_; 
v___x_1456_ = lean_st_ref_get(v___y_1405_);
lean_dec(v___x_1456_);
v___x_1457_ = lean_st_ref_get(v_parentRef_1390_);
v___x_1458_ = lean_array_get_size(v_val_1426_);
v___x_1459_ = lean_unsigned_to_nat(1u);
v___x_1460_ = lean_nat_dec_eq(v___x_1458_, v___x_1459_);
if (v___x_1460_ == 0)
{
lean_object* v_toMonadOptions_1461_; lean_object* v___x_1462_; lean_object* v___x_45064__overap_1463_; lean_object* v___x_1464_; 
lean_dec(v___x_1457_);
lean_del_object(v___x_1428_);
lean_dec(v_val_1426_);
lean_dec_ref(v___f_1402_);
lean_dec_ref(v___x_1401_);
v_toMonadOptions_1461_ = lean_ctor_get(v___x_1391_, 0);
lean_inc(v_toMonadOptions_1461_);
lean_dec_ref(v___x_1391_);
v___x_1462_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc_ref(v___x_1392_);
v___x_45064__overap_1463_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1392_, v_toMonadOptions_1461_, v___x_1462_);
lean_inc(v___y_1411_);
lean_inc_ref(v___y_1410_);
lean_inc(v___y_1409_);
lean_inc_ref(v___y_1408_);
lean_inc(v___y_1407_);
lean_inc(v___y_1406_);
lean_inc(v___y_1405_);
lean_inc_ref(v___y_1404_);
v___x_1464_ = lean_apply_9(v___x_45064__overap_1463_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, lean_box(0));
if (lean_obj_tag(v___x_1464_) == 0)
{
lean_object* v_a_1465_; uint8_t v___x_1466_; 
v_a_1465_ = lean_ctor_get(v___x_1464_, 0);
lean_inc(v_a_1465_);
lean_dec_ref_known(v___x_1464_, 1);
v___x_1466_ = lean_unbox(v_a_1465_);
lean_dec(v_a_1465_);
if (v___x_1466_ == 0)
{
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec_ref(v___x_1392_);
v___y_1414_ = v___y_1405_;
goto v___jp_1413_;
}
else
{
lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v_traceClass_1475_; lean_object* v___x_1476_; lean_object* v___f_1477_; lean_object* v___f_1478_; lean_object* v___f_1479_; lean_object* v___f_1480_; lean_object* v___x_1481_; lean_object* v___x_45095__overap_1482_; lean_object* v___x_1483_; 
v___x_1467_ = l_Lean_Core_instMonadTraceCoreM;
v___x_1468_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1394_, v___x_1467_);
v___x_1469_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1395_, v___x_1468_);
lean_inc(v___x_1396_);
v___x_1470_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1396_, v___x_1469_);
lean_inc(v___x_1397_);
v___x_1471_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1397_, v___x_1470_);
lean_inc(v___f_1398_);
v___x_1472_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1398_, v___x_1471_);
lean_inc_ref(v___f_1399_);
v___x_1473_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1399_, v___x_1472_);
v___x_1474_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_1400_);
lean_dec_ref(v_inst_1400_);
v_traceClass_1475_ = lean_ctor_get(v___x_1462_, 0);
v___x_1476_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_1477_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1477_, 0, v___x_1476_);
lean_closure_set(v___f_1477_, 1, v___x_1396_);
v___f_1478_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1478_, 0, v___f_1477_);
lean_closure_set(v___f_1478_, 1, v___x_1397_);
v___f_1479_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1479_, 0, v___f_1478_);
lean_closure_set(v___f_1479_, 1, v___f_1398_);
v___f_1480_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1480_, 0, v___f_1479_);
lean_closure_set(v___f_1480_, 1, v___f_1399_);
v___x_1481_ = lean_obj_once(&lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2, &lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2_once, _init_lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2);
lean_inc(v_traceClass_1475_);
v___x_45095__overap_1482_ = l_Lean_addTrace___redArg(v___x_1392_, v___x_1473_, v___x_1474_, v___f_1480_, v_traceClass_1475_, v___x_1481_);
lean_inc(v___y_1411_);
lean_inc_ref(v___y_1410_);
lean_inc(v___y_1409_);
lean_inc_ref(v___y_1408_);
lean_inc(v___y_1407_);
lean_inc(v___y_1406_);
lean_inc(v___y_1405_);
lean_inc_ref(v___y_1404_);
v___x_1483_ = lean_apply_9(v___x_45095__overap_1482_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, lean_box(0));
if (lean_obj_tag(v___x_1483_) == 0)
{
lean_dec_ref_known(v___x_1483_, 1);
v___y_1414_ = v___y_1405_;
goto v___jp_1413_;
}
else
{
lean_object* v_a_1484_; lean_object* v___x_1486_; uint8_t v_isShared_1487_; uint8_t v_isSharedCheck_1491_; 
lean_dec(v_rule_1393_);
lean_dec(v_parentRef_1390_);
v_a_1484_ = lean_ctor_get(v___x_1483_, 0);
v_isSharedCheck_1491_ = !lean_is_exclusive(v___x_1483_);
if (v_isSharedCheck_1491_ == 0)
{
v___x_1486_ = v___x_1483_;
v_isShared_1487_ = v_isSharedCheck_1491_;
goto v_resetjp_1485_;
}
else
{
lean_inc(v_a_1484_);
lean_dec(v___x_1483_);
v___x_1486_ = lean_box(0);
v_isShared_1487_ = v_isSharedCheck_1491_;
goto v_resetjp_1485_;
}
v_resetjp_1485_:
{
lean_object* v___x_1489_; 
if (v_isShared_1487_ == 0)
{
v___x_1489_ = v___x_1486_;
goto v_reusejp_1488_;
}
else
{
lean_object* v_reuseFailAlloc_1490_; 
v_reuseFailAlloc_1490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1490_, 0, v_a_1484_);
v___x_1489_ = v_reuseFailAlloc_1490_;
goto v_reusejp_1488_;
}
v_reusejp_1488_:
{
return v___x_1489_;
}
}
}
}
}
else
{
lean_object* v_a_1492_; lean_object* v___x_1494_; uint8_t v_isShared_1495_; uint8_t v_isSharedCheck_1499_; 
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec(v_rule_1393_);
lean_dec_ref(v___x_1392_);
lean_dec(v_parentRef_1390_);
v_a_1492_ = lean_ctor_get(v___x_1464_, 0);
v_isSharedCheck_1499_ = !lean_is_exclusive(v___x_1464_);
if (v_isSharedCheck_1499_ == 0)
{
v___x_1494_ = v___x_1464_;
v_isShared_1495_ = v_isSharedCheck_1499_;
goto v_resetjp_1493_;
}
else
{
lean_inc(v_a_1492_);
lean_dec(v___x_1464_);
v___x_1494_ = lean_box(0);
v_isShared_1495_ = v_isSharedCheck_1499_;
goto v_resetjp_1493_;
}
v_resetjp_1493_:
{
lean_object* v___x_1497_; 
if (v_isShared_1495_ == 0)
{
v___x_1497_ = v___x_1494_;
goto v_reusejp_1496_;
}
else
{
lean_object* v_reuseFailAlloc_1498_; 
v_reuseFailAlloc_1498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1498_, 0, v_a_1492_);
v___x_1497_ = v_reuseFailAlloc_1498_;
goto v_reusejp_1496_;
}
v_reusejp_1496_:
{
return v___x_1497_;
}
}
}
}
else
{
lean_object* v___x_1500_; lean_object* v_elimGoal_1501_; lean_object* v___x_1502_; lean_object* v_mvars_1503_; lean_object* v___x_1504_; uint8_t v___x_1505_; 
v___x_1500_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1501_ = lean_ctor_get(v___x_1500_, 1);
lean_inc_ref(v_elimGoal_1501_);
v___x_1502_ = lean_apply_1(v_elimGoal_1501_, v___x_1457_);
v_mvars_1503_ = lean_ctor_get(v___x_1502_, 7);
lean_inc_ref(v_mvars_1503_);
lean_dec_ref(v___x_1502_);
v___x_1504_ = lean_unsigned_to_nat(0u);
v___x_1505_ = lean_nat_dec_lt(v___x_1504_, v___x_1458_);
if (v___x_1505_ == 0)
{
lean_dec_ref(v_mvars_1503_);
lean_del_object(v___x_1428_);
lean_dec_ref(v___f_1402_);
lean_dec_ref(v___x_1401_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec_ref(v___x_1392_);
lean_dec_ref(v___x_1391_);
goto v___jp_1436_;
}
else
{
if (v___x_1505_ == 0)
{
lean_dec_ref(v_mvars_1503_);
lean_del_object(v___x_1428_);
lean_dec_ref(v___f_1402_);
lean_dec_ref(v___x_1401_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec_ref(v___x_1392_);
lean_dec_ref(v___x_1391_);
goto v___jp_1436_;
}
else
{
lean_object* v___f_1506_; size_t v___x_1507_; size_t v___x_1508_; lean_object* v___x_45123__overap_1509_; lean_object* v___x_1510_; 
v___f_1506_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runSafeRule___redArg___lam__1___boxed), 13, 3);
lean_closure_set(v___f_1506_, 0, v_mvars_1503_);
lean_closure_set(v___f_1506_, 1, v___x_1401_);
lean_closure_set(v___f_1506_, 2, v___f_1402_);
v___x_1507_ = ((size_t)0ULL);
v___x_1508_ = lean_usize_of_nat(v___x_1458_);
lean_inc(v_val_1426_);
lean_inc_ref(v___x_1392_);
v___x_45123__overap_1509_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_1392_, v___f_1506_, v_val_1426_, v___x_1507_, v___x_1508_);
lean_inc(v___y_1411_);
lean_inc_ref(v___y_1410_);
lean_inc(v___y_1409_);
lean_inc_ref(v___y_1408_);
lean_inc(v___y_1407_);
lean_inc(v___y_1406_);
lean_inc(v___y_1405_);
lean_inc_ref(v___y_1404_);
v___x_1510_ = lean_apply_9(v___x_45123__overap_1509_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, lean_box(0));
if (lean_obj_tag(v___x_1510_) == 0)
{
lean_object* v_a_1511_; uint8_t v___x_1512_; 
v_a_1511_ = lean_ctor_get(v___x_1510_, 0);
lean_inc(v_a_1511_);
lean_dec_ref_known(v___x_1510_, 1);
v___x_1512_ = lean_unbox(v_a_1511_);
lean_dec(v_a_1511_);
if (v___x_1512_ == 0)
{
lean_del_object(v___x_1428_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec_ref(v___x_1392_);
lean_dec_ref(v___x_1391_);
goto v___jp_1436_;
}
else
{
lean_object* v_toMonadOptions_1513_; lean_object* v___x_1514_; lean_object* v___x_45127__overap_1515_; lean_object* v___x_1516_; 
lean_dec(v_parentRef_1390_);
v_toMonadOptions_1513_ = lean_ctor_get(v___x_1391_, 0);
lean_inc(v_toMonadOptions_1513_);
lean_dec_ref(v___x_1391_);
v___x_1514_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc_ref(v___x_1392_);
v___x_45127__overap_1515_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1392_, v_toMonadOptions_1513_, v___x_1514_);
lean_inc(v___y_1411_);
lean_inc_ref(v___y_1410_);
lean_inc(v___y_1409_);
lean_inc_ref(v___y_1408_);
lean_inc(v___y_1407_);
lean_inc(v___y_1406_);
lean_inc(v___y_1405_);
lean_inc_ref(v___y_1404_);
v___x_1516_ = lean_apply_9(v___x_45127__overap_1515_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, lean_box(0));
if (lean_obj_tag(v___x_1516_) == 0)
{
lean_object* v_a_1517_; uint8_t v___x_1518_; 
v_a_1517_ = lean_ctor_get(v___x_1516_, 0);
lean_inc(v_a_1517_);
lean_dec_ref_known(v___x_1516_, 1);
v___x_1518_ = lean_unbox(v_a_1517_);
lean_dec(v_a_1517_);
if (v___x_1518_ == 0)
{
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec_ref(v___x_1392_);
goto v___jp_1430_;
}
else
{
lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v_traceClass_1527_; lean_object* v___x_1528_; lean_object* v___f_1529_; lean_object* v___f_1530_; lean_object* v___f_1531_; lean_object* v___f_1532_; lean_object* v___x_1533_; lean_object* v___x_45150__overap_1534_; lean_object* v___x_1535_; 
v___x_1519_ = l_Lean_Core_instMonadTraceCoreM;
v___x_1520_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1394_, v___x_1519_);
v___x_1521_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1395_, v___x_1520_);
lean_inc(v___x_1396_);
v___x_1522_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1396_, v___x_1521_);
lean_inc(v___x_1397_);
v___x_1523_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1397_, v___x_1522_);
lean_inc(v___f_1398_);
v___x_1524_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1398_, v___x_1523_);
lean_inc_ref(v___f_1399_);
v___x_1525_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1399_, v___x_1524_);
v___x_1526_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_1400_);
lean_dec_ref(v_inst_1400_);
v_traceClass_1527_ = lean_ctor_get(v___x_1514_, 0);
v___x_1528_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_1529_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1529_, 0, v___x_1528_);
lean_closure_set(v___f_1529_, 1, v___x_1396_);
v___f_1530_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1530_, 0, v___f_1529_);
lean_closure_set(v___f_1530_, 1, v___x_1397_);
v___f_1531_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1531_, 0, v___f_1530_);
lean_closure_set(v___f_1531_, 1, v___f_1398_);
v___f_1532_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1532_, 0, v___f_1531_);
lean_closure_set(v___f_1532_, 1, v___f_1399_);
v___x_1533_ = lean_obj_once(&lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4, &lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4_once, _init_lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4);
lean_inc(v_traceClass_1527_);
v___x_45150__overap_1534_ = l_Lean_addTrace___redArg(v___x_1392_, v___x_1525_, v___x_1526_, v___f_1532_, v_traceClass_1527_, v___x_1533_);
lean_inc(v___y_1411_);
lean_inc_ref(v___y_1410_);
lean_inc(v___y_1409_);
lean_inc_ref(v___y_1408_);
lean_inc(v___y_1407_);
lean_inc(v___y_1406_);
lean_inc(v___y_1405_);
lean_inc_ref(v___y_1404_);
v___x_1535_ = lean_apply_9(v___x_45150__overap_1534_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, lean_box(0));
if (lean_obj_tag(v___x_1535_) == 0)
{
lean_dec_ref_known(v___x_1535_, 1);
goto v___jp_1430_;
}
else
{
lean_object* v_a_1536_; lean_object* v___x_1538_; uint8_t v_isShared_1539_; uint8_t v_isSharedCheck_1543_; 
lean_del_object(v___x_1428_);
lean_dec(v_val_1426_);
lean_dec(v_rule_1393_);
v_a_1536_ = lean_ctor_get(v___x_1535_, 0);
v_isSharedCheck_1543_ = !lean_is_exclusive(v___x_1535_);
if (v_isSharedCheck_1543_ == 0)
{
v___x_1538_ = v___x_1535_;
v_isShared_1539_ = v_isSharedCheck_1543_;
goto v_resetjp_1537_;
}
else
{
lean_inc(v_a_1536_);
lean_dec(v___x_1535_);
v___x_1538_ = lean_box(0);
v_isShared_1539_ = v_isSharedCheck_1543_;
goto v_resetjp_1537_;
}
v_resetjp_1537_:
{
lean_object* v___x_1541_; 
if (v_isShared_1539_ == 0)
{
v___x_1541_ = v___x_1538_;
goto v_reusejp_1540_;
}
else
{
lean_object* v_reuseFailAlloc_1542_; 
v_reuseFailAlloc_1542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1542_, 0, v_a_1536_);
v___x_1541_ = v_reuseFailAlloc_1542_;
goto v_reusejp_1540_;
}
v_reusejp_1540_:
{
return v___x_1541_;
}
}
}
}
}
else
{
lean_object* v_a_1544_; lean_object* v___x_1546_; uint8_t v_isShared_1547_; uint8_t v_isSharedCheck_1551_; 
lean_del_object(v___x_1428_);
lean_dec(v_val_1426_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec(v_rule_1393_);
lean_dec_ref(v___x_1392_);
v_a_1544_ = lean_ctor_get(v___x_1516_, 0);
v_isSharedCheck_1551_ = !lean_is_exclusive(v___x_1516_);
if (v_isSharedCheck_1551_ == 0)
{
v___x_1546_ = v___x_1516_;
v_isShared_1547_ = v_isSharedCheck_1551_;
goto v_resetjp_1545_;
}
else
{
lean_inc(v_a_1544_);
lean_dec(v___x_1516_);
v___x_1546_ = lean_box(0);
v_isShared_1547_ = v_isSharedCheck_1551_;
goto v_resetjp_1545_;
}
v_resetjp_1545_:
{
lean_object* v___x_1549_; 
if (v_isShared_1547_ == 0)
{
v___x_1549_ = v___x_1546_;
goto v_reusejp_1548_;
}
else
{
lean_object* v_reuseFailAlloc_1550_; 
v_reuseFailAlloc_1550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1550_, 0, v_a_1544_);
v___x_1549_ = v_reuseFailAlloc_1550_;
goto v_reusejp_1548_;
}
v_reusejp_1548_:
{
return v___x_1549_;
}
}
}
}
}
else
{
lean_object* v_a_1552_; lean_object* v___x_1554_; uint8_t v_isShared_1555_; uint8_t v_isSharedCheck_1559_; 
lean_del_object(v___x_1428_);
lean_dec(v_val_1426_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec(v_rule_1393_);
lean_dec_ref(v___x_1392_);
lean_dec_ref(v___x_1391_);
lean_dec(v_parentRef_1390_);
v_a_1552_ = lean_ctor_get(v___x_1510_, 0);
v_isSharedCheck_1559_ = !lean_is_exclusive(v___x_1510_);
if (v_isSharedCheck_1559_ == 0)
{
v___x_1554_ = v___x_1510_;
v_isShared_1555_ = v_isSharedCheck_1559_;
goto v_resetjp_1553_;
}
else
{
lean_inc(v_a_1552_);
lean_dec(v___x_1510_);
v___x_1554_ = lean_box(0);
v_isShared_1555_ = v_isSharedCheck_1559_;
goto v_resetjp_1553_;
}
v_resetjp_1553_:
{
lean_object* v___x_1557_; 
if (v_isShared_1555_ == 0)
{
v___x_1557_ = v___x_1554_;
goto v_reusejp_1556_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v_a_1552_);
v___x_1557_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1556_;
}
v_reusejp_1556_:
{
return v___x_1557_;
}
}
}
}
}
}
v___jp_1430_:
{
lean_object* v___x_1431_; lean_object* v___x_1433_; 
v___x_1431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1431_, 0, v_rule_1393_);
lean_ctor_set(v___x_1431_, 1, v_val_1426_);
if (v_isShared_1429_ == 0)
{
lean_ctor_set(v___x_1428_, 0, v___x_1431_);
v___x_1433_ = v___x_1428_;
goto v_reusejp_1432_;
}
else
{
lean_object* v_reuseFailAlloc_1435_; 
v_reuseFailAlloc_1435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1435_, 0, v___x_1431_);
v___x_1433_ = v_reuseFailAlloc_1435_;
goto v_reusejp_1432_;
}
v_reusejp_1432_:
{
lean_object* v___x_1434_; 
v___x_1434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1434_, 0, v___x_1433_);
return v___x_1434_;
}
}
v___jp_1436_:
{
lean_object* v___x_1437_; lean_object* v___x_1438_; 
v___x_1437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1437_, 0, v_rule_1393_);
v___x_1438_ = lp_aesop_Aesop_addRapps___redArg(v_inst_1400_, v_parentRef_1390_, v___x_1437_, v_val_1426_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_);
if (lean_obj_tag(v___x_1438_) == 0)
{
lean_object* v_a_1439_; lean_object* v___x_1441_; uint8_t v_isShared_1442_; uint8_t v_isSharedCheck_1447_; 
v_a_1439_ = lean_ctor_get(v___x_1438_, 0);
v_isSharedCheck_1447_ = !lean_is_exclusive(v___x_1438_);
if (v_isSharedCheck_1447_ == 0)
{
v___x_1441_ = v___x_1438_;
v_isShared_1442_ = v_isSharedCheck_1447_;
goto v_resetjp_1440_;
}
else
{
lean_inc(v_a_1439_);
lean_dec(v___x_1438_);
v___x_1441_ = lean_box(0);
v_isShared_1442_ = v_isSharedCheck_1447_;
goto v_resetjp_1440_;
}
v_resetjp_1440_:
{
lean_object* v___x_1443_; lean_object* v___x_1445_; 
v___x_1443_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1443_, 0, v_a_1439_);
if (v_isShared_1442_ == 0)
{
lean_ctor_set(v___x_1441_, 0, v___x_1443_);
v___x_1445_ = v___x_1441_;
goto v_reusejp_1444_;
}
else
{
lean_object* v_reuseFailAlloc_1446_; 
v_reuseFailAlloc_1446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1446_, 0, v___x_1443_);
v___x_1445_ = v_reuseFailAlloc_1446_;
goto v_reusejp_1444_;
}
v_reusejp_1444_:
{
return v___x_1445_;
}
}
}
else
{
lean_object* v_a_1448_; lean_object* v___x_1450_; uint8_t v_isShared_1451_; uint8_t v_isSharedCheck_1455_; 
v_a_1448_ = lean_ctor_get(v___x_1438_, 0);
v_isSharedCheck_1455_ = !lean_is_exclusive(v___x_1438_);
if (v_isSharedCheck_1455_ == 0)
{
v___x_1450_ = v___x_1438_;
v_isShared_1451_ = v_isSharedCheck_1455_;
goto v_resetjp_1449_;
}
else
{
lean_inc(v_a_1448_);
lean_dec(v___x_1438_);
v___x_1450_ = lean_box(0);
v_isShared_1451_ = v_isSharedCheck_1455_;
goto v_resetjp_1449_;
}
v_resetjp_1449_:
{
lean_object* v___x_1453_; 
if (v_isShared_1451_ == 0)
{
v___x_1453_ = v___x_1450_;
goto v_reusejp_1452_;
}
else
{
lean_object* v_reuseFailAlloc_1454_; 
v_reuseFailAlloc_1454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1454_, 0, v_a_1448_);
v___x_1453_ = v_reuseFailAlloc_1454_;
goto v_reusejp_1452_;
}
v_reusejp_1452_:
{
return v___x_1453_;
}
}
}
}
}
}
else
{
lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1564_; uint8_t v_isShared_1565_; uint8_t v_isSharedCheck_1570_; 
lean_dec(v_____x_1403_);
lean_dec_ref(v___f_1402_);
lean_dec_ref(v___x_1401_);
lean_dec_ref(v_inst_1400_);
lean_dec_ref(v___f_1399_);
lean_dec(v___f_1398_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
lean_dec(v___f_1395_);
lean_dec(v___x_1394_);
lean_dec_ref(v___x_1392_);
lean_dec_ref(v___x_1391_);
v___x_1561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1561_, 0, v_rule_1393_);
v___x_1562_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(v___x_1561_, v_parentRef_1390_, v___y_1405_);
lean_dec(v_parentRef_1390_);
v_isSharedCheck_1570_ = !lean_is_exclusive(v___x_1562_);
if (v_isSharedCheck_1570_ == 0)
{
lean_object* v_unused_1571_; 
v_unused_1571_ = lean_ctor_get(v___x_1562_, 0);
lean_dec(v_unused_1571_);
v___x_1564_ = v___x_1562_;
v_isShared_1565_ = v_isSharedCheck_1570_;
goto v_resetjp_1563_;
}
else
{
lean_dec(v___x_1562_);
v___x_1564_ = lean_box(0);
v_isShared_1565_ = v_isSharedCheck_1570_;
goto v_resetjp_1563_;
}
v_resetjp_1563_:
{
lean_object* v___x_1566_; lean_object* v___x_1568_; 
v___x_1566_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__0));
if (v_isShared_1565_ == 0)
{
lean_ctor_set(v___x_1564_, 0, v___x_1566_);
v___x_1568_ = v___x_1564_;
goto v_reusejp_1567_;
}
else
{
lean_object* v_reuseFailAlloc_1569_; 
v_reuseFailAlloc_1569_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1569_, 0, v___x_1566_);
v___x_1568_ = v_reuseFailAlloc_1569_;
goto v_reusejp_1567_;
}
v_reusejp_1567_:
{
return v___x_1568_;
}
}
}
v___jp_1413_:
{
lean_object* v___x_1415_; lean_object* v___x_1416_; lean_object* v___x_1418_; uint8_t v_isShared_1419_; uint8_t v_isSharedCheck_1424_; 
v___x_1415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1415_, 0, v_rule_1393_);
v___x_1416_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(v___x_1415_, v_parentRef_1390_, v___y_1414_);
lean_dec(v_parentRef_1390_);
v_isSharedCheck_1424_ = !lean_is_exclusive(v___x_1416_);
if (v_isSharedCheck_1424_ == 0)
{
lean_object* v_unused_1425_; 
v_unused_1425_ = lean_ctor_get(v___x_1416_, 0);
lean_dec(v_unused_1425_);
v___x_1418_ = v___x_1416_;
v_isShared_1419_ = v_isSharedCheck_1424_;
goto v_resetjp_1417_;
}
else
{
lean_dec(v___x_1416_);
v___x_1418_ = lean_box(0);
v_isShared_1419_ = v_isSharedCheck_1424_;
goto v_resetjp_1417_;
}
v_resetjp_1417_:
{
lean_object* v___x_1420_; lean_object* v___x_1422_; 
v___x_1420_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__0));
if (v_isShared_1419_ == 0)
{
lean_ctor_set(v___x_1418_, 0, v___x_1420_);
v___x_1422_ = v___x_1418_;
goto v_reusejp_1421_;
}
else
{
lean_object* v_reuseFailAlloc_1423_; 
v_reuseFailAlloc_1423_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1423_, 0, v___x_1420_);
v___x_1422_ = v_reuseFailAlloc_1423_;
goto v_reusejp_1421_;
}
v_reusejp_1421_:
{
return v___x_1422_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___lam__2___boxed(lean_object** _args){
lean_object* v_parentRef_1572_ = _args[0];
lean_object* v___x_1573_ = _args[1];
lean_object* v___x_1574_ = _args[2];
lean_object* v_rule_1575_ = _args[3];
lean_object* v___x_1576_ = _args[4];
lean_object* v___f_1577_ = _args[5];
lean_object* v___x_1578_ = _args[6];
lean_object* v___x_1579_ = _args[7];
lean_object* v___f_1580_ = _args[8];
lean_object* v___f_1581_ = _args[9];
lean_object* v_inst_1582_ = _args[10];
lean_object* v___x_1583_ = _args[11];
lean_object* v___f_1584_ = _args[12];
lean_object* v_____x_1585_ = _args[13];
lean_object* v___y_1586_ = _args[14];
lean_object* v___y_1587_ = _args[15];
lean_object* v___y_1588_ = _args[16];
lean_object* v___y_1589_ = _args[17];
lean_object* v___y_1590_ = _args[18];
lean_object* v___y_1591_ = _args[19];
lean_object* v___y_1592_ = _args[20];
lean_object* v___y_1593_ = _args[21];
lean_object* v___y_1594_ = _args[22];
_start:
{
lean_object* v_res_1595_; 
v_res_1595_ = lp_aesop_Aesop_runSafeRule___redArg___lam__2(v_parentRef_1572_, v___x_1573_, v___x_1574_, v_rule_1575_, v___x_1576_, v___f_1577_, v___x_1578_, v___x_1579_, v___f_1580_, v___f_1581_, v_inst_1582_, v___x_1583_, v___f_1584_, v_____x_1585_, v___y_1586_, v___y_1587_, v___y_1588_, v___y_1589_, v___y_1590_, v___y_1591_, v___y_1592_, v___y_1593_);
lean_dec(v___y_1593_);
lean_dec_ref(v___y_1592_);
lean_dec(v___y_1591_);
lean_dec_ref(v___y_1590_);
lean_dec(v___y_1589_);
lean_dec(v___y_1588_);
lean_dec(v___y_1587_);
lean_dec_ref(v___y_1586_);
return v_res_1595_;
}
}
static lean_object* _init_lp_aesop_Aesop_runSafeRule___redArg___closed__0(void){
_start:
{
lean_object* v___x_1596_; 
v___x_1596_ = l_instMonadEIO(lean_box(0));
return v___x_1596_;
}
}
static lean_object* _init_lp_aesop_Aesop_runSafeRule___redArg___closed__1(void){
_start:
{
lean_object* v___x_1597_; lean_object* v___x_1598_; 
v___x_1597_ = lean_obj_once(&lp_aesop_Aesop_runSafeRule___redArg___closed__0, &lp_aesop_Aesop_runSafeRule___redArg___closed__0_once, _init_lp_aesop_Aesop_runSafeRule___redArg___closed__0);
v___x_1598_ = l_StateRefT_x27_instMonad___redArg(v___x_1597_);
return v___x_1598_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg(lean_object* v_inst_1604_, lean_object* v_parentRef_1605_, lean_object* v_matchResult_1606_, lean_object* v_a_1607_, lean_object* v_a_1608_, lean_object* v_a_1609_, lean_object* v_a_1610_, lean_object* v_a_1611_, lean_object* v_a_1612_, lean_object* v_a_1613_, lean_object* v_a_1614_){
_start:
{
lean_object* v___x_1616_; lean_object* v_toApplicative_1617_; lean_object* v_toFunctor_1618_; lean_object* v_toSeq_1619_; lean_object* v_toSeqLeft_1620_; lean_object* v_toSeqRight_1621_; lean_object* v___f_1622_; lean_object* v___f_1623_; lean_object* v___f_1624_; lean_object* v___f_1625_; lean_object* v___x_1626_; lean_object* v___f_1627_; lean_object* v___f_1628_; lean_object* v___f_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v_toApplicative_1633_; lean_object* v___x_1635_; uint8_t v_isShared_1636_; uint8_t v_isSharedCheck_2230_; 
v___x_1616_ = lean_obj_once(&lp_aesop_Aesop_runSafeRule___redArg___closed__1, &lp_aesop_Aesop_runSafeRule___redArg___closed__1_once, _init_lp_aesop_Aesop_runSafeRule___redArg___closed__1);
v_toApplicative_1617_ = lean_ctor_get(v___x_1616_, 0);
v_toFunctor_1618_ = lean_ctor_get(v_toApplicative_1617_, 0);
v_toSeq_1619_ = lean_ctor_get(v_toApplicative_1617_, 2);
v_toSeqLeft_1620_ = lean_ctor_get(v_toApplicative_1617_, 3);
v_toSeqRight_1621_ = lean_ctor_get(v_toApplicative_1617_, 4);
v___f_1622_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__2));
v___f_1623_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_1618_, 2);
v___f_1624_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1624_, 0, v_toFunctor_1618_);
v___f_1625_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1625_, 0, v_toFunctor_1618_);
v___x_1626_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1626_, 0, v___f_1624_);
lean_ctor_set(v___x_1626_, 1, v___f_1625_);
lean_inc(v_toSeqRight_1621_);
v___f_1627_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1627_, 0, v_toSeqRight_1621_);
lean_inc(v_toSeqLeft_1620_);
v___f_1628_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1628_, 0, v_toSeqLeft_1620_);
lean_inc(v_toSeq_1619_);
v___f_1629_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1629_, 0, v_toSeq_1619_);
v___x_1630_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1630_, 0, v___x_1626_);
lean_ctor_set(v___x_1630_, 1, v___f_1622_);
lean_ctor_set(v___x_1630_, 2, v___f_1629_);
lean_ctor_set(v___x_1630_, 3, v___f_1628_);
lean_ctor_set(v___x_1630_, 4, v___f_1627_);
v___x_1631_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1631_, 0, v___x_1630_);
lean_ctor_set(v___x_1631_, 1, v___f_1623_);
v___x_1632_ = l_StateRefT_x27_instMonad___redArg(v___x_1631_);
v_toApplicative_1633_ = lean_ctor_get(v___x_1632_, 0);
v_isSharedCheck_2230_ = !lean_is_exclusive(v___x_1632_);
if (v_isSharedCheck_2230_ == 0)
{
lean_object* v_unused_2231_; 
v_unused_2231_ = lean_ctor_get(v___x_1632_, 1);
lean_dec(v_unused_2231_);
v___x_1635_ = v___x_1632_;
v_isShared_1636_ = v_isSharedCheck_2230_;
goto v_resetjp_1634_;
}
else
{
lean_inc(v_toApplicative_1633_);
lean_dec(v___x_1632_);
v___x_1635_ = lean_box(0);
v_isShared_1636_ = v_isSharedCheck_2230_;
goto v_resetjp_1634_;
}
v_resetjp_1634_:
{
lean_object* v_toFunctor_1637_; lean_object* v_toSeq_1638_; lean_object* v_toSeqLeft_1639_; lean_object* v_toSeqRight_1640_; lean_object* v___x_1642_; uint8_t v_isShared_1643_; uint8_t v_isSharedCheck_2228_; 
v_toFunctor_1637_ = lean_ctor_get(v_toApplicative_1633_, 0);
v_toSeq_1638_ = lean_ctor_get(v_toApplicative_1633_, 2);
v_toSeqLeft_1639_ = lean_ctor_get(v_toApplicative_1633_, 3);
v_toSeqRight_1640_ = lean_ctor_get(v_toApplicative_1633_, 4);
v_isSharedCheck_2228_ = !lean_is_exclusive(v_toApplicative_1633_);
if (v_isSharedCheck_2228_ == 0)
{
lean_object* v_unused_2229_; 
v_unused_2229_ = lean_ctor_get(v_toApplicative_1633_, 1);
lean_dec(v_unused_2229_);
v___x_1642_ = v_toApplicative_1633_;
v_isShared_1643_ = v_isSharedCheck_2228_;
goto v_resetjp_1641_;
}
else
{
lean_inc(v_toSeqRight_1640_);
lean_inc(v_toSeqLeft_1639_);
lean_inc(v_toSeq_1638_);
lean_inc(v_toFunctor_1637_);
lean_dec(v_toApplicative_1633_);
v___x_1642_ = lean_box(0);
v_isShared_1643_ = v_isSharedCheck_2228_;
goto v_resetjp_1641_;
}
v_resetjp_1641_:
{
lean_object* v___f_1644_; lean_object* v___f_1645_; lean_object* v___f_1646_; lean_object* v___f_1647_; lean_object* v___x_1648_; lean_object* v___f_1649_; lean_object* v___f_1650_; lean_object* v___f_1651_; lean_object* v___x_1653_; 
v___f_1644_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__4));
v___f_1645_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__5));
lean_inc_ref(v_toFunctor_1637_);
v___f_1646_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1646_, 0, v_toFunctor_1637_);
v___f_1647_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1647_, 0, v_toFunctor_1637_);
v___x_1648_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1648_, 0, v___f_1646_);
lean_ctor_set(v___x_1648_, 1, v___f_1647_);
v___f_1649_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1649_, 0, v_toSeqRight_1640_);
v___f_1650_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1650_, 0, v_toSeqLeft_1639_);
v___f_1651_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1651_, 0, v_toSeq_1638_);
if (v_isShared_1643_ == 0)
{
lean_ctor_set(v___x_1642_, 4, v___f_1649_);
lean_ctor_set(v___x_1642_, 3, v___f_1650_);
lean_ctor_set(v___x_1642_, 2, v___f_1651_);
lean_ctor_set(v___x_1642_, 1, v___f_1644_);
lean_ctor_set(v___x_1642_, 0, v___x_1648_);
v___x_1653_ = v___x_1642_;
goto v_reusejp_1652_;
}
else
{
lean_object* v_reuseFailAlloc_2227_; 
v_reuseFailAlloc_2227_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2227_, 0, v___x_1648_);
lean_ctor_set(v_reuseFailAlloc_2227_, 1, v___f_1644_);
lean_ctor_set(v_reuseFailAlloc_2227_, 2, v___f_1651_);
lean_ctor_set(v_reuseFailAlloc_2227_, 3, v___f_1650_);
lean_ctor_set(v_reuseFailAlloc_2227_, 4, v___f_1649_);
v___x_1653_ = v_reuseFailAlloc_2227_;
goto v_reusejp_1652_;
}
v_reusejp_1652_:
{
lean_object* v___x_1655_; 
if (v_isShared_1636_ == 0)
{
lean_ctor_set(v___x_1635_, 1, v___f_1645_);
lean_ctor_set(v___x_1635_, 0, v___x_1653_);
v___x_1655_ = v___x_1635_;
goto v_reusejp_1654_;
}
else
{
lean_object* v_reuseFailAlloc_2226_; 
v_reuseFailAlloc_2226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2226_, 0, v___x_1653_);
lean_ctor_set(v_reuseFailAlloc_2226_, 1, v___f_1645_);
v___x_1655_ = v_reuseFailAlloc_2226_;
goto v_reusejp_1654_;
}
v_reusejp_1654_:
{
lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v_rule_1658_; lean_object* v_locations_1659_; lean_object* v_patternSubsts_x3f_1660_; lean_object* v___y_1662_; lean_object* v___y_1667_; lean_object* v_name_1679_; lean_object* v_toMonadOptions_1680_; lean_object* v___x_1681_; lean_object* v_options_1682_; lean_object* v_inheritedTraceOptions_1683_; lean_object* v___f_1684_; lean_object* v___f_1685_; lean_object* v___x_1686_; lean_object* v___f_1687_; lean_object* v___f_1688_; lean_object* v___x_1689_; lean_object* v___y_1691_; lean_object* v___y_1692_; lean_object* v___x_1735_; uint8_t v___y_1737_; lean_object* v___y_1738_; lean_object* v___y_1739_; lean_object* v___y_1740_; uint8_t v___y_1741_; lean_object* v___y_1742_; lean_object* v___y_1743_; lean_object* v___y_1744_; lean_object* v___y_1745_; lean_object* v___y_1746_; lean_object* v___y_1747_; lean_object* v___y_1748_; lean_object* v___y_1749_; lean_object* v_a_1750_; uint8_t v___y_1762_; lean_object* v___y_1763_; lean_object* v___y_1764_; lean_object* v___y_1765_; uint8_t v___y_1766_; lean_object* v___y_1767_; lean_object* v___y_1768_; lean_object* v___y_1769_; lean_object* v___y_1770_; lean_object* v___y_1771_; lean_object* v___y_1772_; lean_object* v___y_1773_; lean_object* v___y_1774_; lean_object* v_a_1775_; uint8_t v___y_1778_; lean_object* v___y_1779_; lean_object* v___y_1780_; lean_object* v___y_1781_; uint8_t v___y_1782_; lean_object* v___y_1783_; lean_object* v___y_1784_; lean_object* v___y_1785_; lean_object* v___y_1786_; lean_object* v___y_1787_; lean_object* v___y_1788_; lean_object* v___y_1789_; lean_object* v___y_1790_; lean_object* v_a_1791_; uint8_t v___y_1806_; lean_object* v___y_1807_; lean_object* v___y_1808_; lean_object* v___y_1809_; uint8_t v___y_1810_; lean_object* v___y_1811_; lean_object* v___y_1812_; lean_object* v___y_1813_; lean_object* v___y_1814_; lean_object* v___y_1815_; lean_object* v___y_1816_; lean_object* v___y_1817_; lean_object* v___y_1818_; lean_object* v_a_1819_; lean_object* v___y_1822_; lean_object* v___y_1823_; lean_object* v___y_1824_; lean_object* v___y_1825_; lean_object* v___y_1826_; lean_object* v___y_1827_; lean_object* v___y_1828_; lean_object* v___y_1829_; uint8_t v___y_1830_; lean_object* v___y_1831_; uint8_t v___y_1832_; lean_object* v___y_1833_; lean_object* v_a_1834_; lean_object* v___y_1846_; lean_object* v___y_1847_; lean_object* v___y_1848_; lean_object* v___y_1849_; lean_object* v___y_1850_; lean_object* v___y_1851_; lean_object* v___y_1852_; uint8_t v___y_1853_; lean_object* v___y_1854_; lean_object* v___y_1855_; uint8_t v___y_1856_; lean_object* v___y_1857_; lean_object* v_a_1858_; lean_object* v___y_1861_; lean_object* v___y_1862_; lean_object* v___y_1863_; lean_object* v___y_1864_; lean_object* v___y_1865_; lean_object* v___y_1866_; lean_object* v___y_1867_; lean_object* v___y_1868_; lean_object* v___y_1869_; uint8_t v___y_1870_; lean_object* v___y_1871_; uint8_t v___y_1872_; lean_object* v_a_1873_; lean_object* v___y_1888_; lean_object* v___y_1889_; lean_object* v___y_1890_; lean_object* v___y_1891_; lean_object* v___y_1892_; lean_object* v___y_1893_; lean_object* v___y_1894_; lean_object* v___y_1895_; uint8_t v___y_1896_; lean_object* v___y_1897_; uint8_t v___y_1898_; lean_object* v___y_1899_; lean_object* v_a_1900_; lean_object* v___x_1902_; lean_object* v___y_1904_; lean_object* v___y_1924_; lean_object* v___y_1925_; lean_object* v___y_1926_; lean_object* v___y_1927_; lean_object* v___y_1928_; lean_object* v___y_1929_; lean_object* v___y_1930_; uint8_t v___y_1931_; lean_object* v___y_1932_; uint8_t v___y_1933_; uint8_t v___y_1979_; lean_object* v___y_1980_; lean_object* v___y_1981_; lean_object* v___y_1982_; lean_object* v___y_1983_; uint8_t v___y_1984_; lean_object* v___y_1985_; lean_object* v___y_1986_; lean_object* v___y_1987_; lean_object* v___y_1988_; lean_object* v___y_1989_; lean_object* v___x_2075_; lean_object* v___x_2076_; uint8_t v___x_2077_; 
v___x_1656_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_1604_);
v___x_1657_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2, &lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2_once, _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2);
v_rule_1658_ = lean_ctor_get(v_matchResult_1606_, 0);
lean_inc_n(v_rule_1658_, 2);
v_locations_1659_ = lean_ctor_get(v_matchResult_1606_, 1);
lean_inc_ref(v_locations_1659_);
v_patternSubsts_x3f_1660_ = lean_ctor_get(v_matchResult_1606_, 2);
lean_inc(v_patternSubsts_x3f_1660_);
lean_dec_ref(v_matchResult_1606_);
v_name_1679_ = lean_ctor_get(v_rule_1658_, 0);
v_toMonadOptions_1680_ = lean_ctor_get(v___x_1657_, 0);
v___x_1681_ = l_Lean_KVMap_instValueBool;
v_options_1682_ = lean_ctor_get(v_a_1613_, 2);
v_inheritedTraceOptions_1683_ = lean_ctor_get(v_a_1613_, 13);
v___f_1684_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__6));
v___f_1685_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__0));
v___x_1686_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1));
lean_inc_ref(v___x_1655_);
v___f_1687_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runSafeRule___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1687_, 0, v___x_1655_);
v___f_1688_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__14));
lean_inc_ref(v_name_1679_);
v___x_1689_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1689_, 0, v_name_1679_);
v___x_1735_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23));
v___x_1902_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1902_, 0, v_rule_1658_);
v___x_2075_ = lp_aesop_Aesop_aesop_collectStats;
v___x_2076_ = l_Lean_Option_get___redArg(v___x_1681_, v_options_1682_, v___x_2075_);
v___x_2077_ = lean_unbox(v___x_2076_);
lean_dec(v___x_2076_);
if (v___x_2077_ == 0)
{
lean_object* v___y_2206_; lean_object* v___x_2217_; lean_object* v___x_43051__overap_2218_; lean_object* v___x_2219_; 
v___x_2217_ = lp_aesop_Aesop_TraceOption_stats;
lean_inc(v_toMonadOptions_1680_);
lean_inc_ref(v___x_1656_);
v___x_43051__overap_2218_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1656_, v_toMonadOptions_1680_, v___x_2217_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_2219_ = lean_apply_9(v___x_43051__overap_2218_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
if (lean_obj_tag(v___x_2219_) == 0)
{
lean_object* v_a_2220_; uint8_t v___x_2221_; 
v_a_2220_ = lean_ctor_get(v___x_2219_, 0);
lean_inc(v_a_2220_);
v___x_2221_ = lean_unbox(v_a_2220_);
lean_dec(v_a_2220_);
if (v___x_2221_ == 0)
{
lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; uint8_t v___x_2225_; 
lean_dec_ref_known(v___x_2219_, 1);
v___x_2222_ = l_Lean_KVMap_instValueString;
v___x_2223_ = lp_aesop_Aesop_aesop_stats_file;
v___x_2224_ = l_Lean_Option_get___redArg(v___x_2222_, v_options_1682_, v___x_2223_);
v___x_2225_ = lean_string_dec_eq(v___x_2224_, v___x_1735_);
lean_dec(v___x_2224_);
if (v___x_2225_ == 0)
{
goto v___jp_2034_;
}
else
{
lean_dec_ref_known(v___x_1689_, 1);
goto v___jp_2078_;
}
}
else
{
v___y_2206_ = v___x_2219_;
goto v___jp_2205_;
}
}
else
{
v___y_2206_ = v___x_2219_;
goto v___jp_2205_;
}
v___jp_2078_:
{
lean_object* v___x_2079_; lean_object* v___x_2080_; lean_object* v___x_2081_; uint8_t v_hasTrace_2082_; lean_object* v___f_2083_; 
v___x_2079_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_2080_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_1604_);
v___x_2081_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_hasTrace_2082_ = lean_ctor_get_uint8(v_options_1682_, sizeof(void*)*1);
v___f_2083_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
if (v_hasTrace_2082_ == 0)
{
lean_object* v___x_2084_; 
v___x_2084_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v_locations_1659_, v_patternSubsts_x3f_1660_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
if (lean_obj_tag(v___x_2084_) == 0)
{
lean_object* v_a_2085_; 
v_a_2085_ = lean_ctor_get(v___x_2084_, 0);
lean_inc(v_a_2085_);
lean_dec_ref_known(v___x_2084_, 1);
if (lean_obj_tag(v_a_2085_) == 1)
{
lean_object* v_val_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; uint8_t v___x_2091_; 
v_val_2086_ = lean_ctor_get(v_a_2085_, 0);
lean_inc(v_val_2086_);
lean_dec_ref_known(v_a_2085_, 1);
v___x_2087_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_2087_);
v___x_2088_ = lean_st_ref_get(v_parentRef_1605_);
v___x_2089_ = lean_array_get_size(v_val_2086_);
v___x_2090_ = lean_unsigned_to_nat(1u);
v___x_2091_ = lean_nat_dec_eq(v___x_2089_, v___x_2090_);
if (v___x_2091_ == 0)
{
lean_object* v_toMonadOptions_2092_; lean_object* v___x_2093_; lean_object* v___x_43618__overap_2094_; lean_object* v___x_2095_; 
lean_dec(v___x_2088_);
lean_dec(v_val_2086_);
lean_dec_ref_known(v___x_1902_, 1);
lean_dec_ref(v___f_1687_);
lean_dec_ref(v___x_1655_);
lean_dec_ref(v_inst_1604_);
v_toMonadOptions_2092_ = lean_ctor_get(v___x_1657_, 0);
v___x_2093_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_toMonadOptions_2092_);
lean_inc_ref(v___x_1656_);
v___x_43618__overap_2094_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1656_, v_toMonadOptions_2092_, v___x_2093_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_2095_ = lean_apply_9(v___x_43618__overap_2094_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
if (lean_obj_tag(v___x_2095_) == 0)
{
lean_object* v_a_2096_; uint8_t v___x_2097_; 
v_a_2096_ = lean_ctor_get(v___x_2095_, 0);
lean_inc(v_a_2096_);
lean_dec_ref_known(v___x_2095_, 1);
v___x_2097_ = lean_unbox(v_a_2096_);
lean_dec(v_a_2096_);
if (v___x_2097_ == 0)
{
lean_dec_ref(v___x_2080_);
lean_dec_ref(v___x_1656_);
v___y_1667_ = v_a_1608_;
goto v___jp_1666_;
}
else
{
lean_object* v_traceClass_2098_; lean_object* v___x_2099_; lean_object* v___x_43660__overap_2100_; lean_object* v___x_2101_; 
v_traceClass_2098_ = lean_ctor_get(v___x_2093_, 0);
v___x_2099_ = lean_obj_once(&lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2, &lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2_once, _init_lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__2);
lean_inc(v_traceClass_2098_);
v___x_43660__overap_2100_ = l_Lean_addTrace___redArg(v___x_1656_, v___x_2079_, v___x_2080_, v___f_2083_, v_traceClass_2098_, v___x_2099_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_2101_ = lean_apply_9(v___x_43660__overap_2100_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
if (lean_obj_tag(v___x_2101_) == 0)
{
lean_dec_ref_known(v___x_2101_, 1);
v___y_1667_ = v_a_1608_;
goto v___jp_1666_;
}
else
{
lean_object* v_a_2102_; lean_object* v___x_2104_; uint8_t v_isShared_2105_; uint8_t v_isSharedCheck_2109_; 
lean_dec(v_rule_1658_);
lean_dec(v_parentRef_1605_);
v_a_2102_ = lean_ctor_get(v___x_2101_, 0);
v_isSharedCheck_2109_ = !lean_is_exclusive(v___x_2101_);
if (v_isSharedCheck_2109_ == 0)
{
v___x_2104_ = v___x_2101_;
v_isShared_2105_ = v_isSharedCheck_2109_;
goto v_resetjp_2103_;
}
else
{
lean_inc(v_a_2102_);
lean_dec(v___x_2101_);
v___x_2104_ = lean_box(0);
v_isShared_2105_ = v_isSharedCheck_2109_;
goto v_resetjp_2103_;
}
v_resetjp_2103_:
{
lean_object* v___x_2107_; 
if (v_isShared_2105_ == 0)
{
v___x_2107_ = v___x_2104_;
goto v_reusejp_2106_;
}
else
{
lean_object* v_reuseFailAlloc_2108_; 
v_reuseFailAlloc_2108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2108_, 0, v_a_2102_);
v___x_2107_ = v_reuseFailAlloc_2108_;
goto v_reusejp_2106_;
}
v_reusejp_2106_:
{
return v___x_2107_;
}
}
}
}
}
else
{
lean_object* v_a_2110_; lean_object* v___x_2112_; uint8_t v_isShared_2113_; uint8_t v_isSharedCheck_2117_; 
lean_dec_ref(v___x_2080_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec(v_parentRef_1605_);
v_a_2110_ = lean_ctor_get(v___x_2095_, 0);
v_isSharedCheck_2117_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2117_ == 0)
{
v___x_2112_ = v___x_2095_;
v_isShared_2113_ = v_isSharedCheck_2117_;
goto v_resetjp_2111_;
}
else
{
lean_inc(v_a_2110_);
lean_dec(v___x_2095_);
v___x_2112_ = lean_box(0);
v_isShared_2113_ = v_isSharedCheck_2117_;
goto v_resetjp_2111_;
}
v_resetjp_2111_:
{
lean_object* v___x_2115_; 
if (v_isShared_2113_ == 0)
{
v___x_2115_ = v___x_2112_;
goto v_reusejp_2114_;
}
else
{
lean_object* v_reuseFailAlloc_2116_; 
v_reuseFailAlloc_2116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2116_, 0, v_a_2110_);
v___x_2115_ = v_reuseFailAlloc_2116_;
goto v_reusejp_2114_;
}
v_reusejp_2114_:
{
return v___x_2115_;
}
}
}
}
else
{
lean_object* v___x_2118_; lean_object* v_elimGoal_2119_; lean_object* v___x_2120_; lean_object* v_mvars_2121_; lean_object* v___x_2122_; uint8_t v___x_2123_; 
v___x_2118_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2119_ = lean_ctor_get(v___x_2118_, 1);
lean_inc_ref(v_elimGoal_2119_);
v___x_2120_ = lean_apply_1(v_elimGoal_2119_, v___x_2088_);
v_mvars_2121_ = lean_ctor_get(v___x_2120_, 7);
lean_inc_ref(v_mvars_2121_);
lean_dec_ref(v___x_2120_);
v___x_2122_ = lean_unsigned_to_nat(0u);
v___x_2123_ = lean_nat_dec_lt(v___x_2122_, v___x_2089_);
if (v___x_2123_ == 0)
{
lean_dec_ref(v_mvars_2121_);
lean_dec_ref(v___x_2080_);
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
v___y_1904_ = v_val_2086_;
goto v___jp_1903_;
}
else
{
if (v___x_2123_ == 0)
{
lean_dec_ref(v_mvars_2121_);
lean_dec_ref(v___x_2080_);
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
v___y_1904_ = v_val_2086_;
goto v___jp_1903_;
}
else
{
lean_object* v___f_2124_; size_t v___x_2125_; size_t v___x_2126_; lean_object* v___x_44853__overap_2127_; lean_object* v___x_2128_; 
v___f_2124_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runSafeRule___redArg___lam__1___boxed), 13, 3);
lean_closure_set(v___f_2124_, 0, v_mvars_2121_);
lean_closure_set(v___f_2124_, 1, v___x_1655_);
lean_closure_set(v___f_2124_, 2, v___f_1687_);
v___x_2125_ = ((size_t)0ULL);
v___x_2126_ = lean_usize_of_nat(v___x_2089_);
lean_inc(v_val_2086_);
lean_inc_ref(v___x_1656_);
v___x_44853__overap_2127_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_1656_, v___f_2124_, v_val_2086_, v___x_2125_, v___x_2126_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_2128_ = lean_apply_9(v___x_44853__overap_2127_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
if (lean_obj_tag(v___x_2128_) == 0)
{
lean_object* v_a_2129_; uint8_t v___x_2130_; 
v_a_2129_ = lean_ctor_get(v___x_2128_, 0);
lean_inc(v_a_2129_);
lean_dec_ref_known(v___x_2128_, 1);
v___x_2130_ = lean_unbox(v_a_2129_);
lean_dec(v_a_2129_);
if (v___x_2130_ == 0)
{
lean_dec_ref(v___x_2080_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
v___y_1904_ = v_val_2086_;
goto v___jp_1903_;
}
else
{
lean_object* v_toMonadOptions_2131_; lean_object* v___x_2132_; lean_object* v___x_44858__overap_2133_; lean_object* v___x_2134_; 
lean_dec_ref_known(v___x_1902_, 1);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_toMonadOptions_2131_ = lean_ctor_get(v___x_1657_, 0);
v___x_2132_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_toMonadOptions_2131_);
lean_inc_ref(v___x_1656_);
v___x_44858__overap_2133_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1656_, v_toMonadOptions_2131_, v___x_2132_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_2134_ = lean_apply_9(v___x_44858__overap_2133_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
if (lean_obj_tag(v___x_2134_) == 0)
{
lean_object* v_a_2135_; uint8_t v___x_2136_; 
v_a_2135_ = lean_ctor_get(v___x_2134_, 0);
lean_inc(v_a_2135_);
lean_dec_ref_known(v___x_2134_, 1);
v___x_2136_ = lean_unbox(v_a_2135_);
lean_dec(v_a_2135_);
if (v___x_2136_ == 0)
{
lean_dec_ref(v___x_2080_);
lean_dec_ref(v___x_1656_);
v___y_1662_ = v_val_2086_;
goto v___jp_1661_;
}
else
{
lean_object* v_traceClass_2137_; lean_object* v___x_2138_; lean_object* v___x_44864__overap_2139_; lean_object* v___x_2140_; 
v_traceClass_2137_ = lean_ctor_get(v___x_2132_, 0);
v___x_2138_ = lean_obj_once(&lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4, &lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4_once, _init_lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__4);
lean_inc(v_traceClass_2137_);
v___x_44864__overap_2139_ = l_Lean_addTrace___redArg(v___x_1656_, v___x_2079_, v___x_2080_, v___f_2083_, v_traceClass_2137_, v___x_2138_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_2140_ = lean_apply_9(v___x_44864__overap_2139_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
if (lean_obj_tag(v___x_2140_) == 0)
{
lean_dec_ref_known(v___x_2140_, 1);
v___y_1662_ = v_val_2086_;
goto v___jp_1661_;
}
else
{
lean_object* v_a_2141_; lean_object* v___x_2143_; uint8_t v_isShared_2144_; uint8_t v_isSharedCheck_2148_; 
lean_dec(v_val_2086_);
lean_dec(v_rule_1658_);
v_a_2141_ = lean_ctor_get(v___x_2140_, 0);
v_isSharedCheck_2148_ = !lean_is_exclusive(v___x_2140_);
if (v_isSharedCheck_2148_ == 0)
{
v___x_2143_ = v___x_2140_;
v_isShared_2144_ = v_isSharedCheck_2148_;
goto v_resetjp_2142_;
}
else
{
lean_inc(v_a_2141_);
lean_dec(v___x_2140_);
v___x_2143_ = lean_box(0);
v_isShared_2144_ = v_isSharedCheck_2148_;
goto v_resetjp_2142_;
}
v_resetjp_2142_:
{
lean_object* v___x_2146_; 
if (v_isShared_2144_ == 0)
{
v___x_2146_ = v___x_2143_;
goto v_reusejp_2145_;
}
else
{
lean_object* v_reuseFailAlloc_2147_; 
v_reuseFailAlloc_2147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2147_, 0, v_a_2141_);
v___x_2146_ = v_reuseFailAlloc_2147_;
goto v_reusejp_2145_;
}
v_reusejp_2145_:
{
return v___x_2146_;
}
}
}
}
}
else
{
lean_object* v_a_2149_; lean_object* v___x_2151_; uint8_t v_isShared_2152_; uint8_t v_isSharedCheck_2156_; 
lean_dec(v_val_2086_);
lean_dec_ref(v___x_2080_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
v_a_2149_ = lean_ctor_get(v___x_2134_, 0);
v_isSharedCheck_2156_ = !lean_is_exclusive(v___x_2134_);
if (v_isSharedCheck_2156_ == 0)
{
v___x_2151_ = v___x_2134_;
v_isShared_2152_ = v_isSharedCheck_2156_;
goto v_resetjp_2150_;
}
else
{
lean_inc(v_a_2149_);
lean_dec(v___x_2134_);
v___x_2151_ = lean_box(0);
v_isShared_2152_ = v_isSharedCheck_2156_;
goto v_resetjp_2150_;
}
v_resetjp_2150_:
{
lean_object* v___x_2154_; 
if (v_isShared_2152_ == 0)
{
v___x_2154_ = v___x_2151_;
goto v_reusejp_2153_;
}
else
{
lean_object* v_reuseFailAlloc_2155_; 
v_reuseFailAlloc_2155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2155_, 0, v_a_2149_);
v___x_2154_ = v_reuseFailAlloc_2155_;
goto v_reusejp_2153_;
}
v_reusejp_2153_:
{
return v___x_2154_;
}
}
}
}
}
else
{
lean_object* v_a_2157_; lean_object* v___x_2159_; uint8_t v_isShared_2160_; uint8_t v_isSharedCheck_2164_; 
lean_dec(v_val_2086_);
lean_dec_ref(v___x_2080_);
lean_dec_ref_known(v___x_1902_, 1);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2157_ = lean_ctor_get(v___x_2128_, 0);
v_isSharedCheck_2164_ = !lean_is_exclusive(v___x_2128_);
if (v_isSharedCheck_2164_ == 0)
{
v___x_2159_ = v___x_2128_;
v_isShared_2160_ = v_isSharedCheck_2164_;
goto v_resetjp_2158_;
}
else
{
lean_inc(v_a_2157_);
lean_dec(v___x_2128_);
v___x_2159_ = lean_box(0);
v_isShared_2160_ = v_isSharedCheck_2164_;
goto v_resetjp_2158_;
}
v_resetjp_2158_:
{
lean_object* v___x_2162_; 
if (v_isShared_2160_ == 0)
{
v___x_2162_ = v___x_2159_;
goto v_reusejp_2161_;
}
else
{
lean_object* v_reuseFailAlloc_2163_; 
v_reuseFailAlloc_2163_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2163_, 0, v_a_2157_);
v___x_2162_ = v_reuseFailAlloc_2163_;
goto v_reusejp_2161_;
}
v_reusejp_2161_:
{
return v___x_2162_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2165_; lean_object* v___x_2167_; uint8_t v_isShared_2168_; uint8_t v_isSharedCheck_2173_; 
lean_dec(v_a_2085_);
lean_dec_ref(v___x_2080_);
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
lean_dec_ref(v_inst_1604_);
v___x_2165_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(v___x_1902_, v_parentRef_1605_, v_a_1608_);
lean_dec(v_parentRef_1605_);
v_isSharedCheck_2173_ = !lean_is_exclusive(v___x_2165_);
if (v_isSharedCheck_2173_ == 0)
{
lean_object* v_unused_2174_; 
v_unused_2174_ = lean_ctor_get(v___x_2165_, 0);
lean_dec(v_unused_2174_);
v___x_2167_ = v___x_2165_;
v_isShared_2168_ = v_isSharedCheck_2173_;
goto v_resetjp_2166_;
}
else
{
lean_dec(v___x_2165_);
v___x_2167_ = lean_box(0);
v_isShared_2168_ = v_isSharedCheck_2173_;
goto v_resetjp_2166_;
}
v_resetjp_2166_:
{
lean_object* v___x_2169_; lean_object* v___x_2171_; 
v___x_2169_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__0));
if (v_isShared_2168_ == 0)
{
lean_ctor_set(v___x_2167_, 0, v___x_2169_);
v___x_2171_ = v___x_2167_;
goto v_reusejp_2170_;
}
else
{
lean_object* v_reuseFailAlloc_2172_; 
v_reuseFailAlloc_2172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2172_, 0, v___x_2169_);
v___x_2171_ = v_reuseFailAlloc_2172_;
goto v_reusejp_2170_;
}
v_reusejp_2170_:
{
return v___x_2171_;
}
}
}
}
else
{
lean_object* v_a_2175_; lean_object* v___x_2177_; uint8_t v_isShared_2178_; uint8_t v_isSharedCheck_2182_; 
lean_dec_ref(v___x_2080_);
lean_dec_ref_known(v___x_1902_, 1);
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2175_ = lean_ctor_get(v___x_2084_, 0);
v_isSharedCheck_2182_ = !lean_is_exclusive(v___x_2084_);
if (v_isSharedCheck_2182_ == 0)
{
v___x_2177_ = v___x_2084_;
v_isShared_2178_ = v_isSharedCheck_2182_;
goto v_resetjp_2176_;
}
else
{
lean_inc(v_a_2175_);
lean_dec(v___x_2084_);
v___x_2177_ = lean_box(0);
v_isShared_2178_ = v_isSharedCheck_2182_;
goto v_resetjp_2176_;
}
v_resetjp_2176_:
{
lean_object* v___x_2180_; 
if (v_isShared_2178_ == 0)
{
v___x_2180_ = v___x_2177_;
goto v_reusejp_2179_;
}
else
{
lean_object* v_reuseFailAlloc_2181_; 
v_reuseFailAlloc_2181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2181_, 0, v_a_2175_);
v___x_2180_ = v_reuseFailAlloc_2181_;
goto v_reusejp_2179_;
}
v_reusejp_2179_:
{
return v___x_2180_;
}
}
}
}
else
{
lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v_traceClass_2185_; lean_object* v___f_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; uint8_t v___x_2190_; 
v___x_2183_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_2183_);
v___x_2184_ = lp_aesop_Aesop_TraceOption_steps;
v_traceClass_2185_ = lean_ctor_get(v___x_2184_, 0);
v___f_2186_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
lean_inc_ref(v_name_1679_);
lean_inc_ref(v_inst_1604_);
v___x_2187_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed), 16, 6);
lean_closure_set(v___x_2187_, 0, lean_box(0));
lean_closure_set(v___x_2187_, 1, v_inst_1604_);
lean_closure_set(v___x_2187_, 2, lean_box(0));
lean_closure_set(v___x_2187_, 3, v_name_1679_);
lean_closure_set(v___x_2187_, 4, v___f_1684_);
lean_closure_set(v___x_2187_, 5, v___x_1735_);
v___x_2188_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_2185_);
v___x_2189_ = l_Lean_Name_append(v___x_2188_, v_traceClass_2185_);
v___x_2190_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1683_, v_options_1682_, v___x_2189_);
lean_dec(v___x_2189_);
if (v___x_2190_ == 0)
{
lean_object* v___x_2191_; lean_object* v___x_2192_; uint8_t v___x_2193_; 
v___x_2191_ = l_Lean_trace_profiler;
v___x_2192_ = l_Lean_Option_get___redArg(v___x_1681_, v_options_1682_, v___x_2191_);
v___x_2193_ = lean_unbox(v___x_2192_);
lean_dec(v___x_2192_);
if (v___x_2193_ == 0)
{
lean_object* v___x_2194_; 
lean_dec_ref(v___x_2187_);
lean_dec_ref(v___x_2080_);
v___x_2194_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v_locations_1659_, v_patternSubsts_x3f_1660_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
lean_dec_ref_known(v___x_1902_, 1);
if (lean_obj_tag(v___x_2194_) == 0)
{
lean_object* v_a_2195_; lean_object* v___x_2196_; 
v_a_2195_ = lean_ctor_get(v___x_2194_, 0);
lean_inc(v_a_2195_);
lean_dec_ref_known(v___x_2194_, 1);
v___x_2196_ = lp_aesop_Aesop_runSafeRule___redArg___lam__2(v_parentRef_1605_, v___x_1657_, v___x_1656_, v_rule_1658_, v___x_1686_, v___f_1685_, v___x_1686_, v___x_1686_, v___f_1685_, v___f_1688_, v_inst_1604_, v___x_1655_, v___f_1687_, v_a_2195_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
return v___x_2196_;
}
else
{
lean_object* v_a_2197_; lean_object* v___x_2199_; uint8_t v_isShared_2200_; uint8_t v_isSharedCheck_2204_; 
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2197_ = lean_ctor_get(v___x_2194_, 0);
v_isSharedCheck_2204_ = !lean_is_exclusive(v___x_2194_);
if (v_isSharedCheck_2204_ == 0)
{
v___x_2199_ = v___x_2194_;
v_isShared_2200_ = v_isSharedCheck_2204_;
goto v_resetjp_2198_;
}
else
{
lean_inc(v_a_2197_);
lean_dec(v___x_2194_);
v___x_2199_ = lean_box(0);
v_isShared_2200_ = v_isSharedCheck_2204_;
goto v_resetjp_2198_;
}
v_resetjp_2198_:
{
lean_object* v___x_2202_; 
if (v_isShared_2200_ == 0)
{
v___x_2202_ = v___x_2199_;
goto v_reusejp_2201_;
}
else
{
lean_object* v_reuseFailAlloc_2203_; 
v_reuseFailAlloc_2203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2203_, 0, v_a_2197_);
v___x_2202_ = v_reuseFailAlloc_2203_;
goto v_reusejp_2201_;
}
v_reusejp_2201_:
{
return v___x_2202_;
}
}
}
}
else
{
lean_inc(v_traceClass_2185_);
v___y_1924_ = v___x_2079_;
v___y_1925_ = v___x_2080_;
v___y_1926_ = v___x_2081_;
v___y_1927_ = v_options_1682_;
v___y_1928_ = v_traceClass_2185_;
v___y_1929_ = v___f_2083_;
v___y_1930_ = v___f_2186_;
v___y_1931_ = v_hasTrace_2082_;
v___y_1932_ = v___x_2187_;
v___y_1933_ = v___x_2190_;
goto v___jp_1923_;
}
}
else
{
lean_inc(v_traceClass_2185_);
v___y_1924_ = v___x_2079_;
v___y_1925_ = v___x_2080_;
v___y_1926_ = v___x_2081_;
v___y_1927_ = v_options_1682_;
v___y_1928_ = v_traceClass_2185_;
v___y_1929_ = v___f_2083_;
v___y_1930_ = v___f_2186_;
v___y_1931_ = v_hasTrace_2082_;
v___y_1932_ = v___x_2187_;
v___y_1933_ = v___x_2190_;
goto v___jp_1923_;
}
}
}
v___jp_2205_:
{
if (lean_obj_tag(v___y_2206_) == 0)
{
lean_object* v_a_2207_; uint8_t v___x_2208_; 
v_a_2207_ = lean_ctor_get(v___y_2206_, 0);
lean_inc(v_a_2207_);
lean_dec_ref_known(v___y_2206_, 1);
v___x_2208_ = lean_unbox(v_a_2207_);
lean_dec(v_a_2207_);
if (v___x_2208_ == 0)
{
lean_dec_ref_known(v___x_1689_, 1);
goto v___jp_2078_;
}
else
{
goto v___jp_2034_;
}
}
else
{
lean_object* v_a_2209_; lean_object* v___x_2211_; uint8_t v_isShared_2212_; uint8_t v_isSharedCheck_2216_; 
lean_dec_ref_known(v___x_1902_, 1);
lean_dec_ref_known(v___x_1689_, 1);
lean_dec_ref(v___f_1687_);
lean_dec(v_patternSubsts_x3f_1660_);
lean_dec_ref(v_locations_1659_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2209_ = lean_ctor_get(v___y_2206_, 0);
v_isSharedCheck_2216_ = !lean_is_exclusive(v___y_2206_);
if (v_isSharedCheck_2216_ == 0)
{
v___x_2211_ = v___y_2206_;
v_isShared_2212_ = v_isSharedCheck_2216_;
goto v_resetjp_2210_;
}
else
{
lean_inc(v_a_2209_);
lean_dec(v___y_2206_);
v___x_2211_ = lean_box(0);
v_isShared_2212_ = v_isSharedCheck_2216_;
goto v_resetjp_2210_;
}
v_resetjp_2210_:
{
lean_object* v___x_2214_; 
if (v_isShared_2212_ == 0)
{
v___x_2214_ = v___x_2211_;
goto v_reusejp_2213_;
}
else
{
lean_object* v_reuseFailAlloc_2215_; 
v_reuseFailAlloc_2215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2215_, 0, v_a_2209_);
v___x_2214_ = v_reuseFailAlloc_2215_;
goto v_reusejp_2213_;
}
v_reusejp_2213_:
{
return v___x_2214_;
}
}
}
}
}
else
{
goto v___jp_2034_;
}
v___jp_1661_:
{
lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; 
v___x_1663_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1663_, 0, v_rule_1658_);
lean_ctor_set(v___x_1663_, 1, v___y_1662_);
v___x_1664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1664_, 0, v___x_1663_);
v___x_1665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1665_, 0, v___x_1664_);
return v___x_1665_;
}
v___jp_1666_:
{
lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1671_; uint8_t v_isShared_1672_; uint8_t v_isSharedCheck_1677_; 
v___x_1668_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1668_, 0, v_rule_1658_);
v___x_1669_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(v___x_1668_, v_parentRef_1605_, v___y_1667_);
lean_dec(v_parentRef_1605_);
v_isSharedCheck_1677_ = !lean_is_exclusive(v___x_1669_);
if (v_isSharedCheck_1677_ == 0)
{
lean_object* v_unused_1678_; 
v_unused_1678_ = lean_ctor_get(v___x_1669_, 0);
lean_dec(v_unused_1678_);
v___x_1671_ = v___x_1669_;
v_isShared_1672_ = v_isSharedCheck_1677_;
goto v_resetjp_1670_;
}
else
{
lean_dec(v___x_1669_);
v___x_1671_ = lean_box(0);
v_isShared_1672_ = v_isSharedCheck_1677_;
goto v_resetjp_1670_;
}
v_resetjp_1670_:
{
lean_object* v___x_1673_; lean_object* v___x_1675_; 
v___x_1673_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___lam__2___closed__0));
if (v_isShared_1672_ == 0)
{
lean_ctor_set(v___x_1671_, 0, v___x_1673_);
v___x_1675_ = v___x_1671_;
goto v_reusejp_1674_;
}
else
{
lean_object* v_reuseFailAlloc_1676_; 
v_reuseFailAlloc_1676_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1676_, 0, v___x_1673_);
v___x_1675_ = v_reuseFailAlloc_1676_;
goto v_reusejp_1674_;
}
v_reusejp_1674_:
{
return v___x_1675_;
}
}
}
v___jp_1690_:
{
if (lean_obj_tag(v___y_1692_) == 0)
{
lean_object* v_a_1693_; lean_object* v___x_1695_; uint8_t v_isShared_1696_; uint8_t v_isSharedCheck_1734_; 
v_a_1693_ = lean_ctor_get(v___y_1692_, 0);
v_isSharedCheck_1734_ = !lean_is_exclusive(v___y_1692_);
if (v_isSharedCheck_1734_ == 0)
{
v___x_1695_ = v___y_1692_;
v_isShared_1696_ = v_isSharedCheck_1734_;
goto v_resetjp_1694_;
}
else
{
lean_inc(v_a_1693_);
lean_dec(v___y_1692_);
v___x_1695_ = lean_box(0);
v_isShared_1696_ = v_isSharedCheck_1734_;
goto v_resetjp_1694_;
}
v_resetjp_1694_:
{
lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v_stats_1700_; lean_object* v_rulePatternCache_1701_; lean_object* v___x_1703_; uint8_t v_isShared_1704_; uint8_t v_isSharedCheck_1733_; 
v___x_1697_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_1697_);
v___x_1698_ = lean_io_mono_nanos_now();
v___x_1699_ = lean_st_ref_take(v_a_1610_);
v_stats_1700_ = lean_ctor_get(v___x_1699_, 1);
v_rulePatternCache_1701_ = lean_ctor_get(v___x_1699_, 0);
v_isSharedCheck_1733_ = !lean_is_exclusive(v___x_1699_);
if (v_isSharedCheck_1733_ == 0)
{
v___x_1703_ = v___x_1699_;
v_isShared_1704_ = v_isSharedCheck_1733_;
goto v_resetjp_1702_;
}
else
{
lean_inc(v_stats_1700_);
lean_inc(v_rulePatternCache_1701_);
lean_dec(v___x_1699_);
v___x_1703_ = lean_box(0);
v_isShared_1704_ = v_isSharedCheck_1733_;
goto v_resetjp_1702_;
}
v_resetjp_1702_:
{
lean_object* v_total_1705_; lean_object* v_configParsing_1706_; lean_object* v_ruleSetConstruction_1707_; lean_object* v_search_1708_; lean_object* v_ruleSelection_1709_; lean_object* v_script_1710_; lean_object* v_forwardState_1711_; lean_object* v_scriptGenerated_1712_; lean_object* v_ruleStats_1713_; lean_object* v_goalStats_1714_; lean_object* v___x_1716_; uint8_t v_isShared_1717_; uint8_t v_isSharedCheck_1732_; 
v_total_1705_ = lean_ctor_get(v_stats_1700_, 0);
v_configParsing_1706_ = lean_ctor_get(v_stats_1700_, 1);
v_ruleSetConstruction_1707_ = lean_ctor_get(v_stats_1700_, 2);
v_search_1708_ = lean_ctor_get(v_stats_1700_, 3);
v_ruleSelection_1709_ = lean_ctor_get(v_stats_1700_, 4);
v_script_1710_ = lean_ctor_get(v_stats_1700_, 5);
v_forwardState_1711_ = lean_ctor_get(v_stats_1700_, 6);
v_scriptGenerated_1712_ = lean_ctor_get(v_stats_1700_, 7);
v_ruleStats_1713_ = lean_ctor_get(v_stats_1700_, 8);
v_goalStats_1714_ = lean_ctor_get(v_stats_1700_, 9);
v_isSharedCheck_1732_ = !lean_is_exclusive(v_stats_1700_);
if (v_isSharedCheck_1732_ == 0)
{
v___x_1716_ = v_stats_1700_;
v_isShared_1717_ = v_isSharedCheck_1732_;
goto v_resetjp_1715_;
}
else
{
lean_inc(v_goalStats_1714_);
lean_inc(v_ruleStats_1713_);
lean_inc(v_scriptGenerated_1712_);
lean_inc(v_forwardState_1711_);
lean_inc(v_script_1710_);
lean_inc(v_ruleSelection_1709_);
lean_inc(v_search_1708_);
lean_inc(v_ruleSetConstruction_1707_);
lean_inc(v_configParsing_1706_);
lean_inc(v_total_1705_);
lean_dec(v_stats_1700_);
v___x_1716_ = lean_box(0);
v_isShared_1717_ = v_isSharedCheck_1732_;
goto v_resetjp_1715_;
}
v_resetjp_1715_:
{
lean_object* v___x_1718_; uint8_t v___x_1719_; lean_object* v_rp_1720_; lean_object* v___x_1721_; lean_object* v___x_1723_; 
v___x_1718_ = lean_nat_sub(v___x_1698_, v___y_1691_);
lean_dec(v___y_1691_);
lean_dec(v___x_1698_);
v___x_1719_ = lp_aesop_Aesop_SafeRuleResult_isSuccessfulOrPostponed(v_a_1693_);
v_rp_1720_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_rp_1720_, 0, v___x_1689_);
lean_ctor_set(v_rp_1720_, 1, v___x_1718_);
lean_ctor_set_uint8(v_rp_1720_, sizeof(void*)*2, v___x_1719_);
v___x_1721_ = lean_array_push(v_ruleStats_1713_, v_rp_1720_);
if (v_isShared_1717_ == 0)
{
lean_ctor_set(v___x_1716_, 8, v___x_1721_);
v___x_1723_ = v___x_1716_;
goto v_reusejp_1722_;
}
else
{
lean_object* v_reuseFailAlloc_1731_; 
v_reuseFailAlloc_1731_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1731_, 0, v_total_1705_);
lean_ctor_set(v_reuseFailAlloc_1731_, 1, v_configParsing_1706_);
lean_ctor_set(v_reuseFailAlloc_1731_, 2, v_ruleSetConstruction_1707_);
lean_ctor_set(v_reuseFailAlloc_1731_, 3, v_search_1708_);
lean_ctor_set(v_reuseFailAlloc_1731_, 4, v_ruleSelection_1709_);
lean_ctor_set(v_reuseFailAlloc_1731_, 5, v_script_1710_);
lean_ctor_set(v_reuseFailAlloc_1731_, 6, v_forwardState_1711_);
lean_ctor_set(v_reuseFailAlloc_1731_, 7, v_scriptGenerated_1712_);
lean_ctor_set(v_reuseFailAlloc_1731_, 8, v___x_1721_);
lean_ctor_set(v_reuseFailAlloc_1731_, 9, v_goalStats_1714_);
v___x_1723_ = v_reuseFailAlloc_1731_;
goto v_reusejp_1722_;
}
v_reusejp_1722_:
{
lean_object* v___x_1725_; 
if (v_isShared_1704_ == 0)
{
lean_ctor_set(v___x_1703_, 1, v___x_1723_);
v___x_1725_ = v___x_1703_;
goto v_reusejp_1724_;
}
else
{
lean_object* v_reuseFailAlloc_1730_; 
v_reuseFailAlloc_1730_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1730_, 0, v_rulePatternCache_1701_);
lean_ctor_set(v_reuseFailAlloc_1730_, 1, v___x_1723_);
v___x_1725_ = v_reuseFailAlloc_1730_;
goto v_reusejp_1724_;
}
v_reusejp_1724_:
{
lean_object* v___x_1726_; lean_object* v___x_1728_; 
v___x_1726_ = lean_st_ref_set(v_a_1610_, v___x_1725_);
if (v_isShared_1696_ == 0)
{
v___x_1728_ = v___x_1695_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v_a_1693_);
v___x_1728_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
return v___x_1728_;
}
}
}
}
}
}
}
else
{
lean_dec(v___y_1691_);
lean_dec_ref_known(v___x_1689_, 1);
return v___y_1692_;
}
}
v___jp_1736_:
{
lean_object* v___x_1751_; lean_object* v___x_1752_; double v___x_1753_; double v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_42470__overap_1759_; lean_object* v___x_1760_; 
v___x_1751_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_1751_);
v___x_1752_ = lean_io_get_num_heartbeats();
v___x_1753_ = lean_float_of_nat(v___y_1740_);
v___x_1754_ = lean_float_of_nat(v___x_1752_);
v___x_1755_ = lean_box_float(v___x_1753_);
v___x_1756_ = lean_box_float(v___x_1754_);
v___x_1757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1757_, 0, v___x_1755_);
lean_ctor_set(v___x_1757_, 1, v___x_1756_);
v___x_1758_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1758_, 0, v_a_1750_);
lean_ctor_set(v___x_1758_, 1, v___x_1757_);
lean_inc_ref(v___y_1743_);
lean_inc_ref(v___y_1739_);
lean_inc(v___y_1742_);
lean_inc_ref(v___y_1746_);
v___x_42470__overap_1759_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1656_, v___y_1746_, v___y_1747_, v___y_1742_, lean_box(0), v___y_1739_, v___y_1743_, v___y_1749_, v___y_1741_, v___x_1735_, v___y_1744_, v___y_1737_, v___y_1745_, v___y_1748_, v___x_1758_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_1760_ = lean_apply_9(v___x_42470__overap_1759_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
v___y_1691_ = v___y_1738_;
v___y_1692_ = v___x_1760_;
goto v___jp_1690_;
}
v___jp_1761_:
{
lean_object* v___x_1776_; 
v___x_1776_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1776_, 0, v_a_1775_);
v___y_1737_ = v___y_1762_;
v___y_1738_ = v___y_1763_;
v___y_1739_ = v___y_1764_;
v___y_1740_ = v___y_1765_;
v___y_1741_ = v___y_1766_;
v___y_1742_ = v___y_1767_;
v___y_1743_ = v___y_1768_;
v___y_1744_ = v___y_1769_;
v___y_1745_ = v___y_1770_;
v___y_1746_ = v___y_1771_;
v___y_1747_ = v___y_1772_;
v___y_1748_ = v___y_1773_;
v___y_1749_ = v___y_1774_;
v_a_1750_ = v___x_1776_;
goto v___jp_1736_;
}
v___jp_1777_:
{
lean_object* v___x_1792_; lean_object* v___x_1793_; double v___x_1794_; double v___x_1795_; double v___x_1796_; double v___x_1797_; double v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_42435__overap_1803_; lean_object* v___x_1804_; 
v___x_1792_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_1792_);
v___x_1793_ = lean_io_mono_nanos_now();
v___x_1794_ = lean_float_of_nat(v___y_1780_);
v___x_1795_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_1796_ = lean_float_div(v___x_1794_, v___x_1795_);
v___x_1797_ = lean_float_of_nat(v___x_1793_);
v___x_1798_ = lean_float_div(v___x_1797_, v___x_1795_);
v___x_1799_ = lean_box_float(v___x_1796_);
v___x_1800_ = lean_box_float(v___x_1798_);
v___x_1801_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1801_, 0, v___x_1799_);
lean_ctor_set(v___x_1801_, 1, v___x_1800_);
v___x_1802_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1802_, 0, v_a_1791_);
lean_ctor_set(v___x_1802_, 1, v___x_1801_);
lean_inc_ref(v___y_1784_);
lean_inc_ref(v___y_1781_);
lean_inc(v___y_1783_);
lean_inc_ref(v___y_1787_);
v___x_42435__overap_1803_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1656_, v___y_1787_, v___y_1788_, v___y_1783_, lean_box(0), v___y_1781_, v___y_1784_, v___y_1790_, v___y_1782_, v___x_1735_, v___y_1785_, v___y_1778_, v___y_1786_, v___y_1789_, v___x_1802_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_1804_ = lean_apply_9(v___x_42435__overap_1803_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
v___y_1691_ = v___y_1779_;
v___y_1692_ = v___x_1804_;
goto v___jp_1690_;
}
v___jp_1805_:
{
lean_object* v___x_1820_; 
v___x_1820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1820_, 0, v_a_1819_);
v___y_1778_ = v___y_1806_;
v___y_1779_ = v___y_1807_;
v___y_1780_ = v___y_1808_;
v___y_1781_ = v___y_1809_;
v___y_1782_ = v___y_1810_;
v___y_1783_ = v___y_1811_;
v___y_1784_ = v___y_1812_;
v___y_1785_ = v___y_1813_;
v___y_1786_ = v___y_1814_;
v___y_1787_ = v___y_1815_;
v___y_1788_ = v___y_1816_;
v___y_1789_ = v___y_1817_;
v___y_1790_ = v___y_1818_;
v_a_1791_ = v___x_1820_;
goto v___jp_1777_;
}
v___jp_1821_:
{
lean_object* v___x_1835_; lean_object* v___x_1836_; double v___x_1837_; double v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_42243__overap_1843_; lean_object* v___x_1844_; 
v___x_1835_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_1835_);
v___x_1836_ = lean_io_get_num_heartbeats();
v___x_1837_ = lean_float_of_nat(v___y_1833_);
v___x_1838_ = lean_float_of_nat(v___x_1836_);
v___x_1839_ = lean_box_float(v___x_1837_);
v___x_1840_ = lean_box_float(v___x_1838_);
v___x_1841_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1841_, 0, v___x_1839_);
lean_ctor_set(v___x_1841_, 1, v___x_1840_);
v___x_1842_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1842_, 0, v_a_1834_);
lean_ctor_set(v___x_1842_, 1, v___x_1841_);
lean_inc_ref(v___y_1828_);
lean_inc_ref(v___y_1825_);
v___x_42243__overap_1843_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1656_, v___y_1822_, v___y_1823_, v___y_1827_, lean_box(0), v___y_1825_, v___y_1828_, v___y_1826_, v___y_1830_, v___x_1735_, v___y_1824_, v___y_1832_, v___y_1831_, v___y_1829_, v___x_1842_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_1844_ = lean_apply_9(v___x_42243__overap_1843_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
return v___x_1844_;
}
v___jp_1845_:
{
lean_object* v___x_1859_; 
v___x_1859_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1859_, 0, v_a_1858_);
v___y_1822_ = v___y_1846_;
v___y_1823_ = v___y_1847_;
v___y_1824_ = v___y_1849_;
v___y_1825_ = v___y_1848_;
v___y_1826_ = v___y_1850_;
v___y_1827_ = v___y_1851_;
v___y_1828_ = v___y_1852_;
v___y_1829_ = v___y_1854_;
v___y_1830_ = v___y_1853_;
v___y_1831_ = v___y_1857_;
v___y_1832_ = v___y_1856_;
v___y_1833_ = v___y_1855_;
v_a_1834_ = v___x_1859_;
goto v___jp_1821_;
}
v___jp_1860_:
{
lean_object* v___x_1874_; lean_object* v___x_1875_; double v___x_1876_; double v___x_1877_; double v___x_1878_; double v___x_1879_; double v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_42208__overap_1885_; lean_object* v___x_1886_; 
v___x_1874_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_1874_);
v___x_1875_ = lean_io_mono_nanos_now();
v___x_1876_ = lean_float_of_nat(v___y_1867_);
v___x_1877_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_1878_ = lean_float_div(v___x_1876_, v___x_1877_);
v___x_1879_ = lean_float_of_nat(v___x_1875_);
v___x_1880_ = lean_float_div(v___x_1879_, v___x_1877_);
v___x_1881_ = lean_box_float(v___x_1878_);
v___x_1882_ = lean_box_float(v___x_1880_);
v___x_1883_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1883_, 0, v___x_1881_);
lean_ctor_set(v___x_1883_, 1, v___x_1882_);
v___x_1884_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1884_, 0, v_a_1873_);
lean_ctor_set(v___x_1884_, 1, v___x_1883_);
lean_inc_ref(v___y_1868_);
lean_inc_ref(v___y_1864_);
v___x_42208__overap_1885_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1656_, v___y_1861_, v___y_1862_, v___y_1866_, lean_box(0), v___y_1864_, v___y_1868_, v___y_1865_, v___y_1870_, v___x_1735_, v___y_1863_, v___y_1872_, v___y_1871_, v___y_1869_, v___x_1884_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_1886_ = lean_apply_9(v___x_42208__overap_1885_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
return v___x_1886_;
}
v___jp_1887_:
{
lean_object* v___x_1901_; 
v___x_1901_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1901_, 0, v_a_1900_);
v___y_1861_ = v___y_1888_;
v___y_1862_ = v___y_1889_;
v___y_1863_ = v___y_1891_;
v___y_1864_ = v___y_1890_;
v___y_1865_ = v___y_1892_;
v___y_1866_ = v___y_1894_;
v___y_1867_ = v___y_1893_;
v___y_1868_ = v___y_1895_;
v___y_1869_ = v___y_1897_;
v___y_1870_ = v___y_1896_;
v___y_1871_ = v___y_1899_;
v___y_1872_ = v___y_1898_;
v_a_1873_ = v___x_1901_;
goto v___jp_1860_;
}
v___jp_1903_:
{
lean_object* v___x_1905_; 
v___x_1905_ = lp_aesop_Aesop_addRapps___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v___y_1904_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
if (lean_obj_tag(v___x_1905_) == 0)
{
lean_object* v_a_1906_; lean_object* v___x_1908_; uint8_t v_isShared_1909_; uint8_t v_isSharedCheck_1914_; 
v_a_1906_ = lean_ctor_get(v___x_1905_, 0);
v_isSharedCheck_1914_ = !lean_is_exclusive(v___x_1905_);
if (v_isSharedCheck_1914_ == 0)
{
v___x_1908_ = v___x_1905_;
v_isShared_1909_ = v_isSharedCheck_1914_;
goto v_resetjp_1907_;
}
else
{
lean_inc(v_a_1906_);
lean_dec(v___x_1905_);
v___x_1908_ = lean_box(0);
v_isShared_1909_ = v_isSharedCheck_1914_;
goto v_resetjp_1907_;
}
v_resetjp_1907_:
{
lean_object* v___x_1910_; lean_object* v___x_1912_; 
v___x_1910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1910_, 0, v_a_1906_);
if (v_isShared_1909_ == 0)
{
lean_ctor_set(v___x_1908_, 0, v___x_1910_);
v___x_1912_ = v___x_1908_;
goto v_reusejp_1911_;
}
else
{
lean_object* v_reuseFailAlloc_1913_; 
v_reuseFailAlloc_1913_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1913_, 0, v___x_1910_);
v___x_1912_ = v_reuseFailAlloc_1913_;
goto v_reusejp_1911_;
}
v_reusejp_1911_:
{
return v___x_1912_;
}
}
}
else
{
lean_object* v_a_1915_; lean_object* v___x_1917_; uint8_t v_isShared_1918_; uint8_t v_isSharedCheck_1922_; 
v_a_1915_ = lean_ctor_get(v___x_1905_, 0);
v_isSharedCheck_1922_ = !lean_is_exclusive(v___x_1905_);
if (v_isSharedCheck_1922_ == 0)
{
v___x_1917_ = v___x_1905_;
v_isShared_1918_ = v_isSharedCheck_1922_;
goto v_resetjp_1916_;
}
else
{
lean_inc(v_a_1915_);
lean_dec(v___x_1905_);
v___x_1917_ = lean_box(0);
v_isShared_1918_ = v_isSharedCheck_1922_;
goto v_resetjp_1916_;
}
v_resetjp_1916_:
{
lean_object* v___x_1920_; 
if (v_isShared_1918_ == 0)
{
v___x_1920_ = v___x_1917_;
goto v_reusejp_1919_;
}
else
{
lean_object* v_reuseFailAlloc_1921_; 
v_reuseFailAlloc_1921_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1921_, 0, v_a_1915_);
v___x_1920_ = v_reuseFailAlloc_1921_;
goto v_reusejp_1919_;
}
v_reusejp_1919_:
{
return v___x_1920_;
}
}
}
}
v___jp_1923_:
{
lean_object* v___x_42179__overap_1934_; lean_object* v___x_1935_; 
lean_inc_ref(v___y_1924_);
lean_inc_ref(v___x_1656_);
v___x_42179__overap_1934_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_1656_, v___y_1924_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_1935_ = lean_apply_9(v___x_42179__overap_1934_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
if (lean_obj_tag(v___x_1935_) == 0)
{
lean_object* v_a_1936_; lean_object* v___x_1937_; lean_object* v___x_1938_; uint8_t v___x_1939_; 
v_a_1936_ = lean_ctor_get(v___x_1935_, 0);
lean_inc(v_a_1936_);
lean_dec_ref_known(v___x_1935_, 1);
v___x_1937_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1938_ = l_Lean_Option_get___redArg(v___x_1681_, v___y_1927_, v___x_1937_);
v___x_1939_ = lean_unbox(v___x_1938_);
lean_dec(v___x_1938_);
if (v___x_1939_ == 0)
{
lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; 
v___x_1940_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_1940_);
v___x_1941_ = lean_io_mono_nanos_now();
v___x_1942_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v_locations_1659_, v_patternSubsts_x3f_1660_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
lean_dec_ref_known(v___x_1902_, 1);
if (lean_obj_tag(v___x_1942_) == 0)
{
lean_object* v_a_1943_; lean_object* v___x_1944_; 
v_a_1943_ = lean_ctor_get(v___x_1942_, 0);
lean_inc(v_a_1943_);
lean_dec_ref_known(v___x_1942_, 1);
lean_inc_ref(v___x_1656_);
v___x_1944_ = lp_aesop_Aesop_runSafeRule___redArg___lam__2(v_parentRef_1605_, v___x_1657_, v___x_1656_, v_rule_1658_, v___x_1686_, v___f_1685_, v___x_1686_, v___x_1686_, v___f_1685_, v___f_1688_, v_inst_1604_, v___x_1655_, v___f_1687_, v_a_1943_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
if (lean_obj_tag(v___x_1944_) == 0)
{
lean_object* v_a_1945_; lean_object* v___x_1947_; uint8_t v_isShared_1948_; uint8_t v_isSharedCheck_1952_; 
v_a_1945_ = lean_ctor_get(v___x_1944_, 0);
v_isSharedCheck_1952_ = !lean_is_exclusive(v___x_1944_);
if (v_isSharedCheck_1952_ == 0)
{
v___x_1947_ = v___x_1944_;
v_isShared_1948_ = v_isSharedCheck_1952_;
goto v_resetjp_1946_;
}
else
{
lean_inc(v_a_1945_);
lean_dec(v___x_1944_);
v___x_1947_ = lean_box(0);
v_isShared_1948_ = v_isSharedCheck_1952_;
goto v_resetjp_1946_;
}
v_resetjp_1946_:
{
lean_object* v___x_1950_; 
if (v_isShared_1948_ == 0)
{
lean_ctor_set_tag(v___x_1947_, 1);
v___x_1950_ = v___x_1947_;
goto v_reusejp_1949_;
}
else
{
lean_object* v_reuseFailAlloc_1951_; 
v_reuseFailAlloc_1951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1951_, 0, v_a_1945_);
v___x_1950_ = v_reuseFailAlloc_1951_;
goto v_reusejp_1949_;
}
v_reusejp_1949_:
{
v___y_1861_ = v___y_1924_;
v___y_1862_ = v___y_1925_;
v___y_1863_ = v___y_1927_;
v___y_1864_ = v___y_1926_;
v___y_1865_ = v___y_1928_;
v___y_1866_ = v___y_1929_;
v___y_1867_ = v___x_1941_;
v___y_1868_ = v___y_1930_;
v___y_1869_ = v___y_1932_;
v___y_1870_ = v___y_1931_;
v___y_1871_ = v_a_1936_;
v___y_1872_ = v___y_1933_;
v_a_1873_ = v___x_1950_;
goto v___jp_1860_;
}
}
}
else
{
lean_object* v_a_1953_; 
v_a_1953_ = lean_ctor_get(v___x_1944_, 0);
lean_inc(v_a_1953_);
lean_dec_ref_known(v___x_1944_, 1);
v___y_1888_ = v___y_1924_;
v___y_1889_ = v___y_1925_;
v___y_1890_ = v___y_1926_;
v___y_1891_ = v___y_1927_;
v___y_1892_ = v___y_1928_;
v___y_1893_ = v___x_1941_;
v___y_1894_ = v___y_1929_;
v___y_1895_ = v___y_1930_;
v___y_1896_ = v___y_1931_;
v___y_1897_ = v___y_1932_;
v___y_1898_ = v___y_1933_;
v___y_1899_ = v_a_1936_;
v_a_1900_ = v_a_1953_;
goto v___jp_1887_;
}
}
else
{
lean_object* v_a_1954_; 
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_1954_ = lean_ctor_get(v___x_1942_, 0);
lean_inc(v_a_1954_);
lean_dec_ref_known(v___x_1942_, 1);
v___y_1888_ = v___y_1924_;
v___y_1889_ = v___y_1925_;
v___y_1890_ = v___y_1926_;
v___y_1891_ = v___y_1927_;
v___y_1892_ = v___y_1928_;
v___y_1893_ = v___x_1941_;
v___y_1894_ = v___y_1929_;
v___y_1895_ = v___y_1930_;
v___y_1896_ = v___y_1931_;
v___y_1897_ = v___y_1932_;
v___y_1898_ = v___y_1933_;
v___y_1899_ = v_a_1936_;
v_a_1900_ = v_a_1954_;
goto v___jp_1887_;
}
}
else
{
lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; 
v___x_1955_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_1955_);
v___x_1956_ = lean_io_get_num_heartbeats();
v___x_1957_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v_locations_1659_, v_patternSubsts_x3f_1660_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
lean_dec_ref_known(v___x_1902_, 1);
if (lean_obj_tag(v___x_1957_) == 0)
{
lean_object* v_a_1958_; lean_object* v___x_1959_; 
v_a_1958_ = lean_ctor_get(v___x_1957_, 0);
lean_inc(v_a_1958_);
lean_dec_ref_known(v___x_1957_, 1);
lean_inc_ref(v___x_1656_);
v___x_1959_ = lp_aesop_Aesop_runSafeRule___redArg___lam__2(v_parentRef_1605_, v___x_1657_, v___x_1656_, v_rule_1658_, v___x_1686_, v___f_1685_, v___x_1686_, v___x_1686_, v___f_1685_, v___f_1688_, v_inst_1604_, v___x_1655_, v___f_1687_, v_a_1958_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
if (lean_obj_tag(v___x_1959_) == 0)
{
lean_object* v_a_1960_; lean_object* v___x_1962_; uint8_t v_isShared_1963_; uint8_t v_isSharedCheck_1967_; 
v_a_1960_ = lean_ctor_get(v___x_1959_, 0);
v_isSharedCheck_1967_ = !lean_is_exclusive(v___x_1959_);
if (v_isSharedCheck_1967_ == 0)
{
v___x_1962_ = v___x_1959_;
v_isShared_1963_ = v_isSharedCheck_1967_;
goto v_resetjp_1961_;
}
else
{
lean_inc(v_a_1960_);
lean_dec(v___x_1959_);
v___x_1962_ = lean_box(0);
v_isShared_1963_ = v_isSharedCheck_1967_;
goto v_resetjp_1961_;
}
v_resetjp_1961_:
{
lean_object* v___x_1965_; 
if (v_isShared_1963_ == 0)
{
lean_ctor_set_tag(v___x_1962_, 1);
v___x_1965_ = v___x_1962_;
goto v_reusejp_1964_;
}
else
{
lean_object* v_reuseFailAlloc_1966_; 
v_reuseFailAlloc_1966_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1966_, 0, v_a_1960_);
v___x_1965_ = v_reuseFailAlloc_1966_;
goto v_reusejp_1964_;
}
v_reusejp_1964_:
{
v___y_1822_ = v___y_1924_;
v___y_1823_ = v___y_1925_;
v___y_1824_ = v___y_1927_;
v___y_1825_ = v___y_1926_;
v___y_1826_ = v___y_1928_;
v___y_1827_ = v___y_1929_;
v___y_1828_ = v___y_1930_;
v___y_1829_ = v___y_1932_;
v___y_1830_ = v___y_1931_;
v___y_1831_ = v_a_1936_;
v___y_1832_ = v___y_1933_;
v___y_1833_ = v___x_1956_;
v_a_1834_ = v___x_1965_;
goto v___jp_1821_;
}
}
}
else
{
lean_object* v_a_1968_; 
v_a_1968_ = lean_ctor_get(v___x_1959_, 0);
lean_inc(v_a_1968_);
lean_dec_ref_known(v___x_1959_, 1);
v___y_1846_ = v___y_1924_;
v___y_1847_ = v___y_1925_;
v___y_1848_ = v___y_1926_;
v___y_1849_ = v___y_1927_;
v___y_1850_ = v___y_1928_;
v___y_1851_ = v___y_1929_;
v___y_1852_ = v___y_1930_;
v___y_1853_ = v___y_1931_;
v___y_1854_ = v___y_1932_;
v___y_1855_ = v___x_1956_;
v___y_1856_ = v___y_1933_;
v___y_1857_ = v_a_1936_;
v_a_1858_ = v_a_1968_;
goto v___jp_1845_;
}
}
else
{
lean_object* v_a_1969_; 
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_1969_ = lean_ctor_get(v___x_1957_, 0);
lean_inc(v_a_1969_);
lean_dec_ref_known(v___x_1957_, 1);
v___y_1846_ = v___y_1924_;
v___y_1847_ = v___y_1925_;
v___y_1848_ = v___y_1926_;
v___y_1849_ = v___y_1927_;
v___y_1850_ = v___y_1928_;
v___y_1851_ = v___y_1929_;
v___y_1852_ = v___y_1930_;
v___y_1853_ = v___y_1931_;
v___y_1854_ = v___y_1932_;
v___y_1855_ = v___x_1956_;
v___y_1856_ = v___y_1933_;
v___y_1857_ = v_a_1936_;
v_a_1858_ = v_a_1969_;
goto v___jp_1845_;
}
}
}
else
{
lean_object* v_a_1970_; lean_object* v___x_1972_; uint8_t v_isShared_1973_; uint8_t v_isSharedCheck_1977_; 
lean_dec_ref(v___y_1932_);
lean_dec(v___y_1929_);
lean_dec(v___y_1928_);
lean_dec_ref(v___y_1925_);
lean_dec_ref(v___y_1924_);
lean_dec_ref_known(v___x_1902_, 1);
lean_dec_ref(v___f_1687_);
lean_dec(v_patternSubsts_x3f_1660_);
lean_dec_ref(v_locations_1659_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_1970_ = lean_ctor_get(v___x_1935_, 0);
v_isSharedCheck_1977_ = !lean_is_exclusive(v___x_1935_);
if (v_isSharedCheck_1977_ == 0)
{
v___x_1972_ = v___x_1935_;
v_isShared_1973_ = v_isSharedCheck_1977_;
goto v_resetjp_1971_;
}
else
{
lean_inc(v_a_1970_);
lean_dec(v___x_1935_);
v___x_1972_ = lean_box(0);
v_isShared_1973_ = v_isSharedCheck_1977_;
goto v_resetjp_1971_;
}
v_resetjp_1971_:
{
lean_object* v___x_1975_; 
if (v_isShared_1973_ == 0)
{
v___x_1975_ = v___x_1972_;
goto v_reusejp_1974_;
}
else
{
lean_object* v_reuseFailAlloc_1976_; 
v_reuseFailAlloc_1976_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1976_, 0, v_a_1970_);
v___x_1975_ = v_reuseFailAlloc_1976_;
goto v_reusejp_1974_;
}
v_reusejp_1974_:
{
return v___x_1975_;
}
}
}
}
v___jp_1978_:
{
lean_object* v___x_42406__overap_1990_; lean_object* v___x_1991_; 
lean_inc_ref(v___y_1985_);
lean_inc_ref(v___x_1656_);
v___x_42406__overap_1990_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_1656_, v___y_1985_);
lean_inc(v_a_1614_);
lean_inc_ref(v_a_1613_);
lean_inc(v_a_1612_);
lean_inc_ref(v_a_1611_);
lean_inc(v_a_1610_);
lean_inc(v_a_1609_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
v___x_1991_ = lean_apply_9(v___x_42406__overap_1990_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_, lean_box(0));
if (lean_obj_tag(v___x_1991_) == 0)
{
lean_object* v_a_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; uint8_t v___x_1995_; 
v_a_1992_ = lean_ctor_get(v___x_1991_, 0);
lean_inc(v_a_1992_);
lean_dec_ref_known(v___x_1991_, 1);
v___x_1993_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1994_ = l_Lean_Option_get___redArg(v___x_1681_, v___y_1982_, v___x_1993_);
v___x_1995_ = lean_unbox(v___x_1994_);
lean_dec(v___x_1994_);
if (v___x_1995_ == 0)
{
lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; 
v___x_1996_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_1996_);
v___x_1997_ = lean_io_mono_nanos_now();
v___x_1998_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v_locations_1659_, v_patternSubsts_x3f_1660_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
lean_dec_ref_known(v___x_1902_, 1);
if (lean_obj_tag(v___x_1998_) == 0)
{
lean_object* v_a_1999_; lean_object* v___x_2000_; 
v_a_1999_ = lean_ctor_get(v___x_1998_, 0);
lean_inc(v_a_1999_);
lean_dec_ref_known(v___x_1998_, 1);
lean_inc_ref(v___x_1656_);
v___x_2000_ = lp_aesop_Aesop_runSafeRule___redArg___lam__2(v_parentRef_1605_, v___x_1657_, v___x_1656_, v_rule_1658_, v___x_1686_, v___f_1685_, v___x_1686_, v___x_1686_, v___f_1685_, v___f_1688_, v_inst_1604_, v___x_1655_, v___f_1687_, v_a_1999_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
if (lean_obj_tag(v___x_2000_) == 0)
{
lean_object* v_a_2001_; lean_object* v___x_2003_; uint8_t v_isShared_2004_; uint8_t v_isSharedCheck_2008_; 
v_a_2001_ = lean_ctor_get(v___x_2000_, 0);
v_isSharedCheck_2008_ = !lean_is_exclusive(v___x_2000_);
if (v_isSharedCheck_2008_ == 0)
{
v___x_2003_ = v___x_2000_;
v_isShared_2004_ = v_isSharedCheck_2008_;
goto v_resetjp_2002_;
}
else
{
lean_inc(v_a_2001_);
lean_dec(v___x_2000_);
v___x_2003_ = lean_box(0);
v_isShared_2004_ = v_isSharedCheck_2008_;
goto v_resetjp_2002_;
}
v_resetjp_2002_:
{
lean_object* v___x_2006_; 
if (v_isShared_2004_ == 0)
{
lean_ctor_set_tag(v___x_2003_, 1);
v___x_2006_ = v___x_2003_;
goto v_reusejp_2005_;
}
else
{
lean_object* v_reuseFailAlloc_2007_; 
v_reuseFailAlloc_2007_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2007_, 0, v_a_2001_);
v___x_2006_ = v_reuseFailAlloc_2007_;
goto v_reusejp_2005_;
}
v_reusejp_2005_:
{
v___y_1778_ = v___y_1984_;
v___y_1779_ = v___y_1983_;
v___y_1780_ = v___x_1997_;
v___y_1781_ = v___y_1987_;
v___y_1782_ = v___y_1979_;
v___y_1783_ = v___y_1981_;
v___y_1784_ = v___y_1980_;
v___y_1785_ = v___y_1982_;
v___y_1786_ = v_a_1992_;
v___y_1787_ = v___y_1985_;
v___y_1788_ = v___y_1986_;
v___y_1789_ = v___y_1988_;
v___y_1790_ = v___y_1989_;
v_a_1791_ = v___x_2006_;
goto v___jp_1777_;
}
}
}
else
{
lean_object* v_a_2009_; 
v_a_2009_ = lean_ctor_get(v___x_2000_, 0);
lean_inc(v_a_2009_);
lean_dec_ref_known(v___x_2000_, 1);
v___y_1806_ = v___y_1984_;
v___y_1807_ = v___y_1983_;
v___y_1808_ = v___x_1997_;
v___y_1809_ = v___y_1987_;
v___y_1810_ = v___y_1979_;
v___y_1811_ = v___y_1981_;
v___y_1812_ = v___y_1980_;
v___y_1813_ = v___y_1982_;
v___y_1814_ = v_a_1992_;
v___y_1815_ = v___y_1985_;
v___y_1816_ = v___y_1986_;
v___y_1817_ = v___y_1988_;
v___y_1818_ = v___y_1989_;
v_a_1819_ = v_a_2009_;
goto v___jp_1805_;
}
}
else
{
lean_object* v_a_2010_; 
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2010_ = lean_ctor_get(v___x_1998_, 0);
lean_inc(v_a_2010_);
lean_dec_ref_known(v___x_1998_, 1);
v___y_1806_ = v___y_1984_;
v___y_1807_ = v___y_1983_;
v___y_1808_ = v___x_1997_;
v___y_1809_ = v___y_1987_;
v___y_1810_ = v___y_1979_;
v___y_1811_ = v___y_1981_;
v___y_1812_ = v___y_1980_;
v___y_1813_ = v___y_1982_;
v___y_1814_ = v_a_1992_;
v___y_1815_ = v___y_1985_;
v___y_1816_ = v___y_1986_;
v___y_1817_ = v___y_1988_;
v___y_1818_ = v___y_1989_;
v_a_1819_ = v_a_2010_;
goto v___jp_1805_;
}
}
else
{
lean_object* v___x_2011_; lean_object* v___x_2012_; lean_object* v___x_2013_; 
v___x_2011_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_2011_);
v___x_2012_ = lean_io_get_num_heartbeats();
v___x_2013_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v_locations_1659_, v_patternSubsts_x3f_1660_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
lean_dec_ref_known(v___x_1902_, 1);
if (lean_obj_tag(v___x_2013_) == 0)
{
lean_object* v_a_2014_; lean_object* v___x_2015_; 
v_a_2014_ = lean_ctor_get(v___x_2013_, 0);
lean_inc(v_a_2014_);
lean_dec_ref_known(v___x_2013_, 1);
lean_inc_ref(v___x_1656_);
v___x_2015_ = lp_aesop_Aesop_runSafeRule___redArg___lam__2(v_parentRef_1605_, v___x_1657_, v___x_1656_, v_rule_1658_, v___x_1686_, v___f_1685_, v___x_1686_, v___x_1686_, v___f_1685_, v___f_1688_, v_inst_1604_, v___x_1655_, v___f_1687_, v_a_2014_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
if (lean_obj_tag(v___x_2015_) == 0)
{
lean_object* v_a_2016_; lean_object* v___x_2018_; uint8_t v_isShared_2019_; uint8_t v_isSharedCheck_2023_; 
v_a_2016_ = lean_ctor_get(v___x_2015_, 0);
v_isSharedCheck_2023_ = !lean_is_exclusive(v___x_2015_);
if (v_isSharedCheck_2023_ == 0)
{
v___x_2018_ = v___x_2015_;
v_isShared_2019_ = v_isSharedCheck_2023_;
goto v_resetjp_2017_;
}
else
{
lean_inc(v_a_2016_);
lean_dec(v___x_2015_);
v___x_2018_ = lean_box(0);
v_isShared_2019_ = v_isSharedCheck_2023_;
goto v_resetjp_2017_;
}
v_resetjp_2017_:
{
lean_object* v___x_2021_; 
if (v_isShared_2019_ == 0)
{
lean_ctor_set_tag(v___x_2018_, 1);
v___x_2021_ = v___x_2018_;
goto v_reusejp_2020_;
}
else
{
lean_object* v_reuseFailAlloc_2022_; 
v_reuseFailAlloc_2022_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2022_, 0, v_a_2016_);
v___x_2021_ = v_reuseFailAlloc_2022_;
goto v_reusejp_2020_;
}
v_reusejp_2020_:
{
v___y_1737_ = v___y_1984_;
v___y_1738_ = v___y_1983_;
v___y_1739_ = v___y_1987_;
v___y_1740_ = v___x_2012_;
v___y_1741_ = v___y_1979_;
v___y_1742_ = v___y_1981_;
v___y_1743_ = v___y_1980_;
v___y_1744_ = v___y_1982_;
v___y_1745_ = v_a_1992_;
v___y_1746_ = v___y_1985_;
v___y_1747_ = v___y_1986_;
v___y_1748_ = v___y_1988_;
v___y_1749_ = v___y_1989_;
v_a_1750_ = v___x_2021_;
goto v___jp_1736_;
}
}
}
else
{
lean_object* v_a_2024_; 
v_a_2024_ = lean_ctor_get(v___x_2015_, 0);
lean_inc(v_a_2024_);
lean_dec_ref_known(v___x_2015_, 1);
v___y_1762_ = v___y_1984_;
v___y_1763_ = v___y_1983_;
v___y_1764_ = v___y_1987_;
v___y_1765_ = v___x_2012_;
v___y_1766_ = v___y_1979_;
v___y_1767_ = v___y_1981_;
v___y_1768_ = v___y_1980_;
v___y_1769_ = v___y_1982_;
v___y_1770_ = v_a_1992_;
v___y_1771_ = v___y_1985_;
v___y_1772_ = v___y_1986_;
v___y_1773_ = v___y_1988_;
v___y_1774_ = v___y_1989_;
v_a_1775_ = v_a_2024_;
goto v___jp_1761_;
}
}
else
{
lean_object* v_a_2025_; 
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2025_ = lean_ctor_get(v___x_2013_, 0);
lean_inc(v_a_2025_);
lean_dec_ref_known(v___x_2013_, 1);
v___y_1762_ = v___y_1984_;
v___y_1763_ = v___y_1983_;
v___y_1764_ = v___y_1987_;
v___y_1765_ = v___x_2012_;
v___y_1766_ = v___y_1979_;
v___y_1767_ = v___y_1981_;
v___y_1768_ = v___y_1980_;
v___y_1769_ = v___y_1982_;
v___y_1770_ = v_a_1992_;
v___y_1771_ = v___y_1985_;
v___y_1772_ = v___y_1986_;
v___y_1773_ = v___y_1988_;
v___y_1774_ = v___y_1989_;
v_a_1775_ = v_a_2025_;
goto v___jp_1761_;
}
}
}
else
{
lean_object* v_a_2026_; lean_object* v___x_2028_; uint8_t v_isShared_2029_; uint8_t v_isSharedCheck_2033_; 
lean_dec(v___y_1989_);
lean_dec_ref(v___y_1988_);
lean_dec_ref(v___y_1986_);
lean_dec(v___y_1983_);
lean_dec_ref_known(v___x_1902_, 1);
lean_dec_ref_known(v___x_1689_, 1);
lean_dec_ref(v___f_1687_);
lean_dec(v_patternSubsts_x3f_1660_);
lean_dec_ref(v_locations_1659_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2026_ = lean_ctor_get(v___x_1991_, 0);
v_isSharedCheck_2033_ = !lean_is_exclusive(v___x_1991_);
if (v_isSharedCheck_2033_ == 0)
{
v___x_2028_ = v___x_1991_;
v_isShared_2029_ = v_isSharedCheck_2033_;
goto v_resetjp_2027_;
}
else
{
lean_inc(v_a_2026_);
lean_dec(v___x_1991_);
v___x_2028_ = lean_box(0);
v_isShared_2029_ = v_isSharedCheck_2033_;
goto v_resetjp_2027_;
}
v_resetjp_2027_:
{
lean_object* v___x_2031_; 
if (v_isShared_2029_ == 0)
{
v___x_2031_ = v___x_2028_;
goto v_reusejp_2030_;
}
else
{
lean_object* v_reuseFailAlloc_2032_; 
v_reuseFailAlloc_2032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2032_, 0, v_a_2026_);
v___x_2031_ = v_reuseFailAlloc_2032_;
goto v_reusejp_2030_;
}
v_reusejp_2030_:
{
return v___x_2031_;
}
}
}
}
v___jp_2034_:
{
lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; uint8_t v_hasTrace_2040_; 
v___x_2035_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_2035_);
v___x_2036_ = lean_io_mono_nanos_now();
v___x_2037_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_2038_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_1604_);
v___x_2039_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_hasTrace_2040_ = lean_ctor_get_uint8(v_options_1682_, sizeof(void*)*1);
if (v_hasTrace_2040_ == 0)
{
lean_object* v___x_2041_; 
lean_dec_ref(v___x_2038_);
v___x_2041_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v_locations_1659_, v_patternSubsts_x3f_1660_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
lean_dec_ref_known(v___x_1902_, 1);
if (lean_obj_tag(v___x_2041_) == 0)
{
lean_object* v_a_2042_; lean_object* v___x_2043_; 
v_a_2042_ = lean_ctor_get(v___x_2041_, 0);
lean_inc(v_a_2042_);
lean_dec_ref_known(v___x_2041_, 1);
v___x_2043_ = lp_aesop_Aesop_runSafeRule___redArg___lam__2(v_parentRef_1605_, v___x_1657_, v___x_1656_, v_rule_1658_, v___x_1686_, v___f_1685_, v___x_1686_, v___x_1686_, v___f_1685_, v___f_1688_, v_inst_1604_, v___x_1655_, v___f_1687_, v_a_2042_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
v___y_1691_ = v___x_2036_;
v___y_1692_ = v___x_2043_;
goto v___jp_1690_;
}
else
{
lean_object* v_a_2044_; lean_object* v___x_2046_; uint8_t v_isShared_2047_; uint8_t v_isSharedCheck_2051_; 
lean_dec(v___x_2036_);
lean_dec_ref_known(v___x_1689_, 1);
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2044_ = lean_ctor_get(v___x_2041_, 0);
v_isSharedCheck_2051_ = !lean_is_exclusive(v___x_2041_);
if (v_isSharedCheck_2051_ == 0)
{
v___x_2046_ = v___x_2041_;
v_isShared_2047_ = v_isSharedCheck_2051_;
goto v_resetjp_2045_;
}
else
{
lean_inc(v_a_2044_);
lean_dec(v___x_2041_);
v___x_2046_ = lean_box(0);
v_isShared_2047_ = v_isSharedCheck_2051_;
goto v_resetjp_2045_;
}
v_resetjp_2045_:
{
lean_object* v___x_2049_; 
if (v_isShared_2047_ == 0)
{
v___x_2049_ = v___x_2046_;
goto v_reusejp_2048_;
}
else
{
lean_object* v_reuseFailAlloc_2050_; 
v_reuseFailAlloc_2050_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2050_, 0, v_a_2044_);
v___x_2049_ = v_reuseFailAlloc_2050_;
goto v_reusejp_2048_;
}
v_reusejp_2048_:
{
return v___x_2049_;
}
}
}
}
else
{
lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v_traceClass_2054_; lean_object* v___f_2055_; lean_object* v___f_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; uint8_t v___x_2060_; 
v___x_2052_ = lean_st_ref_get(v_a_1608_);
lean_dec(v___x_2052_);
v___x_2053_ = lp_aesop_Aesop_TraceOption_steps;
v_traceClass_2054_ = lean_ctor_get(v___x_2053_, 0);
v___f_2055_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___f_2056_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
lean_inc_ref(v_name_1679_);
lean_inc_ref(v_inst_1604_);
v___x_2057_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed), 16, 6);
lean_closure_set(v___x_2057_, 0, lean_box(0));
lean_closure_set(v___x_2057_, 1, v_inst_1604_);
lean_closure_set(v___x_2057_, 2, lean_box(0));
lean_closure_set(v___x_2057_, 3, v_name_1679_);
lean_closure_set(v___x_2057_, 4, v___f_1684_);
lean_closure_set(v___x_2057_, 5, v___x_1735_);
v___x_2058_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_2054_);
v___x_2059_ = l_Lean_Name_append(v___x_2058_, v_traceClass_2054_);
v___x_2060_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1683_, v_options_1682_, v___x_2059_);
lean_dec(v___x_2059_);
if (v___x_2060_ == 0)
{
lean_object* v___x_2061_; lean_object* v___x_2062_; uint8_t v___x_2063_; 
v___x_2061_ = l_Lean_trace_profiler;
v___x_2062_ = l_Lean_Option_get___redArg(v___x_1681_, v_options_1682_, v___x_2061_);
v___x_2063_ = lean_unbox(v___x_2062_);
lean_dec(v___x_2062_);
if (v___x_2063_ == 0)
{
lean_object* v___x_2064_; 
lean_dec_ref(v___x_2057_);
lean_dec_ref(v___x_2038_);
v___x_2064_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_1604_, v_parentRef_1605_, v___x_1902_, v_locations_1659_, v_patternSubsts_x3f_1660_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
lean_dec_ref_known(v___x_1902_, 1);
if (lean_obj_tag(v___x_2064_) == 0)
{
lean_object* v_a_2065_; lean_object* v___x_2066_; 
v_a_2065_ = lean_ctor_get(v___x_2064_, 0);
lean_inc(v_a_2065_);
lean_dec_ref_known(v___x_2064_, 1);
v___x_2066_ = lp_aesop_Aesop_runSafeRule___redArg___lam__2(v_parentRef_1605_, v___x_1657_, v___x_1656_, v_rule_1658_, v___x_1686_, v___f_1685_, v___x_1686_, v___x_1686_, v___f_1685_, v___f_1688_, v_inst_1604_, v___x_1655_, v___f_1687_, v_a_2065_, v_a_1607_, v_a_1608_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_, v_a_1613_, v_a_1614_);
v___y_1691_ = v___x_2036_;
v___y_1692_ = v___x_2066_;
goto v___jp_1690_;
}
else
{
lean_object* v_a_2067_; lean_object* v___x_2069_; uint8_t v_isShared_2070_; uint8_t v_isSharedCheck_2074_; 
lean_dec(v___x_2036_);
lean_dec_ref_known(v___x_1689_, 1);
lean_dec_ref(v___f_1687_);
lean_dec(v_rule_1658_);
lean_dec_ref(v___x_1656_);
lean_dec_ref(v___x_1655_);
lean_dec(v_parentRef_1605_);
lean_dec_ref(v_inst_1604_);
v_a_2067_ = lean_ctor_get(v___x_2064_, 0);
v_isSharedCheck_2074_ = !lean_is_exclusive(v___x_2064_);
if (v_isSharedCheck_2074_ == 0)
{
v___x_2069_ = v___x_2064_;
v_isShared_2070_ = v_isSharedCheck_2074_;
goto v_resetjp_2068_;
}
else
{
lean_inc(v_a_2067_);
lean_dec(v___x_2064_);
v___x_2069_ = lean_box(0);
v_isShared_2070_ = v_isSharedCheck_2074_;
goto v_resetjp_2068_;
}
v_resetjp_2068_:
{
lean_object* v___x_2072_; 
if (v_isShared_2070_ == 0)
{
v___x_2072_ = v___x_2069_;
goto v_reusejp_2071_;
}
else
{
lean_object* v_reuseFailAlloc_2073_; 
v_reuseFailAlloc_2073_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2073_, 0, v_a_2067_);
v___x_2072_ = v_reuseFailAlloc_2073_;
goto v_reusejp_2071_;
}
v_reusejp_2071_:
{
return v___x_2072_;
}
}
}
}
else
{
lean_inc(v_traceClass_2054_);
v___y_1979_ = v_hasTrace_2040_;
v___y_1980_ = v___f_2056_;
v___y_1981_ = v___f_2055_;
v___y_1982_ = v_options_1682_;
v___y_1983_ = v___x_2036_;
v___y_1984_ = v___x_2060_;
v___y_1985_ = v___x_2037_;
v___y_1986_ = v___x_2038_;
v___y_1987_ = v___x_2039_;
v___y_1988_ = v___x_2057_;
v___y_1989_ = v_traceClass_2054_;
goto v___jp_1978_;
}
}
else
{
lean_inc(v_traceClass_2054_);
v___y_1979_ = v_hasTrace_2040_;
v___y_1980_ = v___f_2056_;
v___y_1981_ = v___f_2055_;
v___y_1982_ = v_options_1682_;
v___y_1983_ = v___x_2036_;
v___y_1984_ = v___x_2060_;
v___y_1985_ = v___x_2037_;
v___y_1986_ = v___x_2038_;
v___y_1987_ = v___x_2039_;
v___y_1988_ = v___x_2057_;
v___y_1989_ = v_traceClass_2054_;
goto v___jp_1978_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___redArg___boxed(lean_object* v_inst_2232_, lean_object* v_parentRef_2233_, lean_object* v_matchResult_2234_, lean_object* v_a_2235_, lean_object* v_a_2236_, lean_object* v_a_2237_, lean_object* v_a_2238_, lean_object* v_a_2239_, lean_object* v_a_2240_, lean_object* v_a_2241_, lean_object* v_a_2242_, lean_object* v_a_2243_){
_start:
{
lean_object* v_res_2244_; 
v_res_2244_ = lp_aesop_Aesop_runSafeRule___redArg(v_inst_2232_, v_parentRef_2233_, v_matchResult_2234_, v_a_2235_, v_a_2236_, v_a_2237_, v_a_2238_, v_a_2239_, v_a_2240_, v_a_2241_, v_a_2242_);
lean_dec(v_a_2242_);
lean_dec_ref(v_a_2241_);
lean_dec(v_a_2240_);
lean_dec_ref(v_a_2239_);
lean_dec(v_a_2238_);
lean_dec(v_a_2237_);
lean_dec(v_a_2236_);
lean_dec_ref(v_a_2235_);
return v_res_2244_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule(lean_object* v_Q_2245_, lean_object* v_inst_2246_, lean_object* v_parentRef_2247_, lean_object* v_matchResult_2248_, lean_object* v_a_2249_, lean_object* v_a_2250_, lean_object* v_a_2251_, lean_object* v_a_2252_, lean_object* v_a_2253_, lean_object* v_a_2254_, lean_object* v_a_2255_, lean_object* v_a_2256_){
_start:
{
lean_object* v___x_2258_; 
v___x_2258_ = lp_aesop_Aesop_runSafeRule___redArg(v_inst_2246_, v_parentRef_2247_, v_matchResult_2248_, v_a_2249_, v_a_2250_, v_a_2251_, v_a_2252_, v_a_2253_, v_a_2254_, v_a_2255_, v_a_2256_);
return v___x_2258_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runSafeRule___boxed(lean_object* v_Q_2259_, lean_object* v_inst_2260_, lean_object* v_parentRef_2261_, lean_object* v_matchResult_2262_, lean_object* v_a_2263_, lean_object* v_a_2264_, lean_object* v_a_2265_, lean_object* v_a_2266_, lean_object* v_a_2267_, lean_object* v_a_2268_, lean_object* v_a_2269_, lean_object* v_a_2270_, lean_object* v_a_2271_){
_start:
{
lean_object* v_res_2272_; 
v_res_2272_ = lp_aesop_Aesop_runSafeRule(v_Q_2259_, v_inst_2260_, v_parentRef_2261_, v_matchResult_2262_, v_a_2263_, v_a_2264_, v_a_2265_, v_a_2266_, v_a_2267_, v_a_2268_, v_a_2269_, v_a_2270_);
lean_dec(v_a_2270_);
lean_dec_ref(v_a_2269_);
lean_dec(v_a_2268_);
lean_dec_ref(v_a_2267_);
lean_dec(v_a_2266_);
lean_dec(v_a_2265_);
lean_dec(v_a_2264_);
lean_dec_ref(v_a_2263_);
return v_res_2272_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(lean_object* v_inst_2273_, lean_object* v_parentRef_2274_, lean_object* v___x_2275_, lean_object* v_____x_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_, lean_object* v___y_2281_, lean_object* v___y_2282_, lean_object* v___y_2283_, lean_object* v___y_2284_){
_start:
{
if (lean_obj_tag(v_____x_2276_) == 1)
{
lean_object* v_val_2286_; lean_object* v___x_2287_; 
v_val_2286_ = lean_ctor_get(v_____x_2276_, 0);
lean_inc(v_val_2286_);
lean_dec_ref_known(v_____x_2276_, 1);
v___x_2287_ = lp_aesop_Aesop_addRapps___redArg(v_inst_2273_, v_parentRef_2274_, v___x_2275_, v_val_2286_, v___y_2277_, v___y_2278_, v___y_2279_, v___y_2280_, v___y_2281_, v___y_2282_, v___y_2283_, v___y_2284_);
return v___x_2287_;
}
else
{
lean_object* v___x_2288_; lean_object* v___x_2290_; uint8_t v_isShared_2291_; uint8_t v_isSharedCheck_2296_; 
lean_dec(v_____x_2276_);
lean_dec_ref(v_inst_2273_);
v___x_2288_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(v___x_2275_, v_parentRef_2274_, v___y_2278_);
lean_dec(v_parentRef_2274_);
v_isSharedCheck_2296_ = !lean_is_exclusive(v___x_2288_);
if (v_isSharedCheck_2296_ == 0)
{
lean_object* v_unused_2297_; 
v_unused_2297_ = lean_ctor_get(v___x_2288_, 0);
lean_dec(v_unused_2297_);
v___x_2290_ = v___x_2288_;
v_isShared_2291_ = v_isSharedCheck_2296_;
goto v_resetjp_2289_;
}
else
{
lean_dec(v___x_2288_);
v___x_2290_ = lean_box(0);
v_isShared_2291_ = v_isSharedCheck_2296_;
goto v_resetjp_2289_;
}
v_resetjp_2289_:
{
lean_object* v___x_2292_; lean_object* v___x_2294_; 
v___x_2292_ = lean_box(2);
if (v_isShared_2291_ == 0)
{
lean_ctor_set(v___x_2290_, 0, v___x_2292_);
v___x_2294_ = v___x_2290_;
goto v_reusejp_2293_;
}
else
{
lean_object* v_reuseFailAlloc_2295_; 
v_reuseFailAlloc_2295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2295_, 0, v___x_2292_);
v___x_2294_ = v_reuseFailAlloc_2295_;
goto v_reusejp_2293_;
}
v_reusejp_2293_:
{
return v___x_2294_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___redArg___lam__0___boxed(lean_object* v_inst_2298_, lean_object* v_parentRef_2299_, lean_object* v___x_2300_, lean_object* v_____x_2301_, lean_object* v___y_2302_, lean_object* v___y_2303_, lean_object* v___y_2304_, lean_object* v___y_2305_, lean_object* v___y_2306_, lean_object* v___y_2307_, lean_object* v___y_2308_, lean_object* v___y_2309_, lean_object* v___y_2310_){
_start:
{
lean_object* v_res_2311_; 
v_res_2311_ = lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(v_inst_2298_, v_parentRef_2299_, v___x_2300_, v_____x_2301_, v___y_2302_, v___y_2303_, v___y_2304_, v___y_2305_, v___y_2306_, v___y_2307_, v___y_2308_, v___y_2309_);
lean_dec(v___y_2309_);
lean_dec_ref(v___y_2308_);
lean_dec(v___y_2307_);
lean_dec_ref(v___y_2306_);
lean_dec(v___y_2305_);
lean_dec(v___y_2304_);
lean_dec(v___y_2303_);
lean_dec_ref(v___y_2302_);
return v_res_2311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___redArg(lean_object* v_inst_2313_, lean_object* v_parentRef_2314_, lean_object* v_matchResult_2315_, lean_object* v_a_2316_, lean_object* v_a_2317_, lean_object* v_a_2318_, lean_object* v_a_2319_, lean_object* v_a_2320_, lean_object* v_a_2321_, lean_object* v_a_2322_, lean_object* v_a_2323_){
_start:
{
lean_object* v_rule_2325_; lean_object* v_locations_2326_; lean_object* v_patternSubsts_x3f_2327_; lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v_name_2330_; lean_object* v_toMonadOptions_2331_; lean_object* v___x_2332_; lean_object* v_options_2333_; lean_object* v_inheritedTraceOptions_2334_; lean_object* v___f_2335_; lean_object* v___x_2336_; lean_object* v___y_2338_; lean_object* v___y_2339_; lean_object* v___x_2382_; lean_object* v___y_2384_; lean_object* v___y_2385_; lean_object* v___y_2386_; uint8_t v___y_2387_; lean_object* v___y_2388_; lean_object* v___y_2389_; lean_object* v___y_2390_; lean_object* v___y_2391_; lean_object* v___y_2392_; uint8_t v___y_2393_; lean_object* v___y_2394_; lean_object* v___y_2395_; lean_object* v_a_2396_; lean_object* v___y_2411_; lean_object* v___y_2412_; lean_object* v___y_2413_; lean_object* v___y_2414_; uint8_t v___y_2415_; lean_object* v___y_2416_; lean_object* v___y_2417_; lean_object* v___y_2418_; lean_object* v___y_2419_; lean_object* v___y_2420_; uint8_t v___y_2421_; lean_object* v___y_2422_; lean_object* v_a_2423_; lean_object* v___y_2426_; lean_object* v___y_2427_; lean_object* v___y_2428_; uint8_t v___y_2429_; lean_object* v___y_2430_; lean_object* v___y_2431_; lean_object* v___y_2432_; lean_object* v___y_2433_; lean_object* v___y_2434_; uint8_t v___y_2435_; lean_object* v___y_2436_; lean_object* v___y_2437_; lean_object* v_a_2438_; lean_object* v___y_2450_; lean_object* v___y_2451_; lean_object* v___y_2452_; lean_object* v___y_2453_; uint8_t v___y_2454_; lean_object* v___y_2455_; lean_object* v___y_2456_; lean_object* v___y_2457_; lean_object* v___y_2458_; lean_object* v___y_2459_; uint8_t v___y_2460_; lean_object* v___y_2461_; lean_object* v_a_2462_; uint8_t v___y_2465_; lean_object* v___y_2466_; lean_object* v___y_2467_; lean_object* v___y_2468_; lean_object* v___y_2469_; lean_object* v___y_2470_; lean_object* v___y_2471_; lean_object* v___y_2472_; lean_object* v___y_2473_; uint8_t v___y_2474_; lean_object* v___y_2475_; lean_object* v___y_2476_; lean_object* v___y_2477_; lean_object* v_a_2478_; uint8_t v___y_2493_; lean_object* v___y_2494_; lean_object* v___y_2495_; lean_object* v___y_2496_; lean_object* v___y_2497_; lean_object* v___y_2498_; lean_object* v___y_2499_; lean_object* v___y_2500_; lean_object* v___y_2501_; uint8_t v___y_2502_; lean_object* v___y_2503_; lean_object* v___y_2504_; lean_object* v___y_2505_; lean_object* v_a_2506_; lean_object* v___y_2509_; uint8_t v___y_2510_; lean_object* v___y_2511_; lean_object* v___y_2512_; lean_object* v___y_2513_; lean_object* v___y_2514_; lean_object* v___y_2515_; lean_object* v___y_2516_; lean_object* v___y_2517_; uint8_t v___y_2518_; lean_object* v___y_2519_; lean_object* v___y_2520_; lean_object* v___y_2521_; lean_object* v_a_2522_; lean_object* v___y_2534_; uint8_t v___y_2535_; lean_object* v___y_2536_; lean_object* v___y_2537_; lean_object* v___y_2538_; lean_object* v___y_2539_; lean_object* v___y_2540_; lean_object* v___y_2541_; lean_object* v___y_2542_; uint8_t v___y_2543_; lean_object* v___y_2544_; lean_object* v___y_2545_; lean_object* v___y_2546_; lean_object* v_a_2547_; lean_object* v___x_2549_; lean_object* v___y_2551_; lean_object* v___y_2552_; lean_object* v___y_2553_; lean_object* v___y_2554_; uint8_t v___y_2555_; lean_object* v___y_2556_; lean_object* v___y_2557_; uint8_t v___y_2558_; lean_object* v___y_2559_; lean_object* v___y_2560_; lean_object* v___y_2656_; lean_object* v___y_2657_; uint8_t v___y_2658_; lean_object* v___y_2659_; lean_object* v___y_2660_; uint8_t v___y_2661_; lean_object* v___y_2662_; lean_object* v___y_2663_; lean_object* v___y_2664_; lean_object* v___y_2665_; lean_object* v___y_2666_; lean_object* v___y_2753_; lean_object* v___x_2764_; lean_object* v___x_2765_; uint8_t v___x_2766_; 
v_rule_2325_ = lean_ctor_get(v_matchResult_2315_, 0);
lean_inc(v_rule_2325_);
v_locations_2326_ = lean_ctor_get(v_matchResult_2315_, 1);
lean_inc_ref(v_locations_2326_);
v_patternSubsts_x3f_2327_ = lean_ctor_get(v_matchResult_2315_, 2);
lean_inc(v_patternSubsts_x3f_2327_);
lean_dec_ref(v_matchResult_2315_);
v___x_2328_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_2313_);
v___x_2329_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2, &lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2_once, _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2);
v_name_2330_ = lean_ctor_get(v_rule_2325_, 0);
lean_inc_ref_n(v_name_2330_, 2);
v_toMonadOptions_2331_ = lean_ctor_get(v___x_2329_, 0);
v___x_2332_ = l_Lean_KVMap_instValueBool;
v_options_2333_ = lean_ctor_get(v_a_2322_, 2);
v_inheritedTraceOptions_2334_ = lean_ctor_get(v_a_2322_, 13);
v___f_2335_ = ((lean_object*)(lp_aesop_Aesop_runUnsafeRule___redArg___closed__0));
v___x_2336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2336_, 0, v_name_2330_);
v___x_2382_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23));
v___x_2549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2549_, 0, v_rule_2325_);
v___x_2764_ = lp_aesop_Aesop_aesop_collectStats;
v___x_2765_ = l_Lean_Option_get___redArg(v___x_2332_, v_options_2333_, v___x_2764_);
v___x_2766_ = lean_unbox(v___x_2765_);
lean_dec(v___x_2765_);
if (v___x_2766_ == 0)
{
lean_object* v___x_2767_; lean_object* v___x_20403__overap_2768_; lean_object* v___x_2769_; 
v___x_2767_ = lp_aesop_Aesop_TraceOption_stats;
lean_inc(v_toMonadOptions_2331_);
lean_inc_ref(v___x_2328_);
v___x_20403__overap_2768_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_2328_, v_toMonadOptions_2331_, v___x_2767_);
lean_inc(v_a_2323_);
lean_inc_ref(v_a_2322_);
lean_inc(v_a_2321_);
lean_inc_ref(v_a_2320_);
lean_inc(v_a_2319_);
lean_inc(v_a_2318_);
lean_inc(v_a_2317_);
lean_inc_ref(v_a_2316_);
v___x_2769_ = lean_apply_9(v___x_20403__overap_2768_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_, lean_box(0));
if (lean_obj_tag(v___x_2769_) == 0)
{
lean_object* v_a_2770_; uint8_t v___x_2771_; 
v_a_2770_ = lean_ctor_get(v___x_2769_, 0);
lean_inc(v_a_2770_);
v___x_2771_ = lean_unbox(v_a_2770_);
lean_dec(v_a_2770_);
if (v___x_2771_ == 0)
{
lean_object* v___x_2772_; lean_object* v___x_2773_; lean_object* v___x_2774_; uint8_t v___x_2775_; 
lean_dec_ref_known(v___x_2769_, 1);
v___x_2772_ = l_Lean_KVMap_instValueString;
v___x_2773_ = lp_aesop_Aesop_aesop_stats_file;
v___x_2774_ = l_Lean_Option_get___redArg(v___x_2772_, v_options_2333_, v___x_2773_);
v___x_2775_ = lean_string_dec_eq(v___x_2774_, v___x_2382_);
lean_dec(v___x_2774_);
if (v___x_2775_ == 0)
{
goto v___jp_2711_;
}
else
{
lean_dec_ref_known(v___x_2336_, 1);
goto v___jp_2605_;
}
}
else
{
v___y_2753_ = v___x_2769_;
goto v___jp_2752_;
}
}
else
{
v___y_2753_ = v___x_2769_;
goto v___jp_2752_;
}
}
else
{
goto v___jp_2711_;
}
v___jp_2337_:
{
if (lean_obj_tag(v___y_2339_) == 0)
{
lean_object* v_a_2340_; lean_object* v___x_2342_; uint8_t v_isShared_2343_; uint8_t v_isSharedCheck_2381_; 
v_a_2340_ = lean_ctor_get(v___y_2339_, 0);
v_isSharedCheck_2381_ = !lean_is_exclusive(v___y_2339_);
if (v_isSharedCheck_2381_ == 0)
{
v___x_2342_ = v___y_2339_;
v_isShared_2343_ = v_isSharedCheck_2381_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_a_2340_);
lean_dec(v___y_2339_);
v___x_2342_ = lean_box(0);
v_isShared_2343_ = v_isSharedCheck_2381_;
goto v_resetjp_2341_;
}
v_resetjp_2341_:
{
lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v_stats_2347_; lean_object* v_rulePatternCache_2348_; lean_object* v___x_2350_; uint8_t v_isShared_2351_; uint8_t v_isSharedCheck_2380_; 
v___x_2344_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2344_);
v___x_2345_ = lean_io_mono_nanos_now();
v___x_2346_ = lean_st_ref_take(v_a_2319_);
v_stats_2347_ = lean_ctor_get(v___x_2346_, 1);
v_rulePatternCache_2348_ = lean_ctor_get(v___x_2346_, 0);
v_isSharedCheck_2380_ = !lean_is_exclusive(v___x_2346_);
if (v_isSharedCheck_2380_ == 0)
{
v___x_2350_ = v___x_2346_;
v_isShared_2351_ = v_isSharedCheck_2380_;
goto v_resetjp_2349_;
}
else
{
lean_inc(v_stats_2347_);
lean_inc(v_rulePatternCache_2348_);
lean_dec(v___x_2346_);
v___x_2350_ = lean_box(0);
v_isShared_2351_ = v_isSharedCheck_2380_;
goto v_resetjp_2349_;
}
v_resetjp_2349_:
{
lean_object* v_total_2352_; lean_object* v_configParsing_2353_; lean_object* v_ruleSetConstruction_2354_; lean_object* v_search_2355_; lean_object* v_ruleSelection_2356_; lean_object* v_script_2357_; lean_object* v_forwardState_2358_; lean_object* v_scriptGenerated_2359_; lean_object* v_ruleStats_2360_; lean_object* v_goalStats_2361_; lean_object* v___x_2363_; uint8_t v_isShared_2364_; uint8_t v_isSharedCheck_2379_; 
v_total_2352_ = lean_ctor_get(v_stats_2347_, 0);
v_configParsing_2353_ = lean_ctor_get(v_stats_2347_, 1);
v_ruleSetConstruction_2354_ = lean_ctor_get(v_stats_2347_, 2);
v_search_2355_ = lean_ctor_get(v_stats_2347_, 3);
v_ruleSelection_2356_ = lean_ctor_get(v_stats_2347_, 4);
v_script_2357_ = lean_ctor_get(v_stats_2347_, 5);
v_forwardState_2358_ = lean_ctor_get(v_stats_2347_, 6);
v_scriptGenerated_2359_ = lean_ctor_get(v_stats_2347_, 7);
v_ruleStats_2360_ = lean_ctor_get(v_stats_2347_, 8);
v_goalStats_2361_ = lean_ctor_get(v_stats_2347_, 9);
v_isSharedCheck_2379_ = !lean_is_exclusive(v_stats_2347_);
if (v_isSharedCheck_2379_ == 0)
{
v___x_2363_ = v_stats_2347_;
v_isShared_2364_ = v_isSharedCheck_2379_;
goto v_resetjp_2362_;
}
else
{
lean_inc(v_goalStats_2361_);
lean_inc(v_ruleStats_2360_);
lean_inc(v_scriptGenerated_2359_);
lean_inc(v_forwardState_2358_);
lean_inc(v_script_2357_);
lean_inc(v_ruleSelection_2356_);
lean_inc(v_search_2355_);
lean_inc(v_ruleSetConstruction_2354_);
lean_inc(v_configParsing_2353_);
lean_inc(v_total_2352_);
lean_dec(v_stats_2347_);
v___x_2363_ = lean_box(0);
v_isShared_2364_ = v_isSharedCheck_2379_;
goto v_resetjp_2362_;
}
v_resetjp_2362_:
{
lean_object* v___x_2365_; uint8_t v___x_2366_; lean_object* v_rp_2367_; lean_object* v___x_2368_; lean_object* v___x_2370_; 
v___x_2365_ = lean_nat_sub(v___x_2345_, v___y_2338_);
lean_dec(v___y_2338_);
lean_dec(v___x_2345_);
v___x_2366_ = lp_aesop_Aesop_RuleResult_isSuccessful(v_a_2340_);
v_rp_2367_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_rp_2367_, 0, v___x_2336_);
lean_ctor_set(v_rp_2367_, 1, v___x_2365_);
lean_ctor_set_uint8(v_rp_2367_, sizeof(void*)*2, v___x_2366_);
v___x_2368_ = lean_array_push(v_ruleStats_2360_, v_rp_2367_);
if (v_isShared_2364_ == 0)
{
lean_ctor_set(v___x_2363_, 8, v___x_2368_);
v___x_2370_ = v___x_2363_;
goto v_reusejp_2369_;
}
else
{
lean_object* v_reuseFailAlloc_2378_; 
v_reuseFailAlloc_2378_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2378_, 0, v_total_2352_);
lean_ctor_set(v_reuseFailAlloc_2378_, 1, v_configParsing_2353_);
lean_ctor_set(v_reuseFailAlloc_2378_, 2, v_ruleSetConstruction_2354_);
lean_ctor_set(v_reuseFailAlloc_2378_, 3, v_search_2355_);
lean_ctor_set(v_reuseFailAlloc_2378_, 4, v_ruleSelection_2356_);
lean_ctor_set(v_reuseFailAlloc_2378_, 5, v_script_2357_);
lean_ctor_set(v_reuseFailAlloc_2378_, 6, v_forwardState_2358_);
lean_ctor_set(v_reuseFailAlloc_2378_, 7, v_scriptGenerated_2359_);
lean_ctor_set(v_reuseFailAlloc_2378_, 8, v___x_2368_);
lean_ctor_set(v_reuseFailAlloc_2378_, 9, v_goalStats_2361_);
v___x_2370_ = v_reuseFailAlloc_2378_;
goto v_reusejp_2369_;
}
v_reusejp_2369_:
{
lean_object* v___x_2372_; 
if (v_isShared_2351_ == 0)
{
lean_ctor_set(v___x_2350_, 1, v___x_2370_);
v___x_2372_ = v___x_2350_;
goto v_reusejp_2371_;
}
else
{
lean_object* v_reuseFailAlloc_2377_; 
v_reuseFailAlloc_2377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2377_, 0, v_rulePatternCache_2348_);
lean_ctor_set(v_reuseFailAlloc_2377_, 1, v___x_2370_);
v___x_2372_ = v_reuseFailAlloc_2377_;
goto v_reusejp_2371_;
}
v_reusejp_2371_:
{
lean_object* v___x_2373_; lean_object* v___x_2375_; 
v___x_2373_ = lean_st_ref_set(v_a_2319_, v___x_2372_);
if (v_isShared_2343_ == 0)
{
v___x_2375_ = v___x_2342_;
goto v_reusejp_2374_;
}
else
{
lean_object* v_reuseFailAlloc_2376_; 
v_reuseFailAlloc_2376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2376_, 0, v_a_2340_);
v___x_2375_ = v_reuseFailAlloc_2376_;
goto v_reusejp_2374_;
}
v_reusejp_2374_:
{
return v___x_2375_;
}
}
}
}
}
}
}
else
{
lean_dec(v___y_2338_);
lean_dec_ref_known(v___x_2336_, 1);
return v___y_2339_;
}
}
v___jp_2383_:
{
lean_object* v___x_2397_; lean_object* v___x_2398_; double v___x_2399_; double v___x_2400_; double v___x_2401_; double v___x_2402_; double v___x_2403_; lean_object* v___x_2404_; lean_object* v___x_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_20080__overap_2408_; lean_object* v___x_2409_; 
v___x_2397_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2397_);
v___x_2398_ = lean_io_mono_nanos_now();
v___x_2399_ = lean_float_of_nat(v___y_2389_);
v___x_2400_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_2401_ = lean_float_div(v___x_2399_, v___x_2400_);
v___x_2402_ = lean_float_of_nat(v___x_2398_);
v___x_2403_ = lean_float_div(v___x_2402_, v___x_2400_);
v___x_2404_ = lean_box_float(v___x_2401_);
v___x_2405_ = lean_box_float(v___x_2403_);
v___x_2406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2406_, 0, v___x_2404_);
lean_ctor_set(v___x_2406_, 1, v___x_2405_);
v___x_2407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2407_, 0, v_a_2396_);
lean_ctor_set(v___x_2407_, 1, v___x_2406_);
lean_inc_ref(v___y_2391_);
lean_inc_ref(v___y_2392_);
lean_inc(v___y_2386_);
lean_inc_ref(v___y_2394_);
v___x_20080__overap_2408_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_2328_, v___y_2394_, v___y_2385_, v___y_2386_, lean_box(0), v___y_2392_, v___y_2391_, v___y_2388_, v___y_2387_, v___x_2382_, v___y_2384_, v___y_2393_, v___y_2390_, v___y_2395_, v___x_2407_);
lean_inc(v_a_2323_);
lean_inc_ref(v_a_2322_);
lean_inc(v_a_2321_);
lean_inc_ref(v_a_2320_);
lean_inc(v_a_2319_);
lean_inc(v_a_2318_);
lean_inc(v_a_2317_);
lean_inc_ref(v_a_2316_);
v___x_2409_ = lean_apply_9(v___x_20080__overap_2408_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_, lean_box(0));
return v___x_2409_;
}
v___jp_2410_:
{
lean_object* v___x_2424_; 
v___x_2424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2424_, 0, v_a_2423_);
v___y_2384_ = v___y_2411_;
v___y_2385_ = v___y_2412_;
v___y_2386_ = v___y_2413_;
v___y_2387_ = v___y_2415_;
v___y_2388_ = v___y_2414_;
v___y_2389_ = v___y_2416_;
v___y_2390_ = v___y_2417_;
v___y_2391_ = v___y_2419_;
v___y_2392_ = v___y_2418_;
v___y_2393_ = v___y_2421_;
v___y_2394_ = v___y_2420_;
v___y_2395_ = v___y_2422_;
v_a_2396_ = v___x_2424_;
goto v___jp_2383_;
}
v___jp_2425_:
{
lean_object* v___x_2439_; lean_object* v___x_2440_; double v___x_2441_; double v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_20115__overap_2447_; lean_object* v___x_2448_; 
v___x_2439_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2439_);
v___x_2440_ = lean_io_get_num_heartbeats();
v___x_2441_ = lean_float_of_nat(v___y_2431_);
v___x_2442_ = lean_float_of_nat(v___x_2440_);
v___x_2443_ = lean_box_float(v___x_2441_);
v___x_2444_ = lean_box_float(v___x_2442_);
v___x_2445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2445_, 0, v___x_2443_);
lean_ctor_set(v___x_2445_, 1, v___x_2444_);
v___x_2446_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2446_, 0, v_a_2438_);
lean_ctor_set(v___x_2446_, 1, v___x_2445_);
lean_inc_ref(v___y_2433_);
lean_inc_ref(v___y_2434_);
lean_inc(v___y_2428_);
lean_inc_ref(v___y_2436_);
v___x_20115__overap_2447_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_2328_, v___y_2436_, v___y_2427_, v___y_2428_, lean_box(0), v___y_2434_, v___y_2433_, v___y_2430_, v___y_2429_, v___x_2382_, v___y_2426_, v___y_2435_, v___y_2432_, v___y_2437_, v___x_2446_);
lean_inc(v_a_2323_);
lean_inc_ref(v_a_2322_);
lean_inc(v_a_2321_);
lean_inc_ref(v_a_2320_);
lean_inc(v_a_2319_);
lean_inc(v_a_2318_);
lean_inc(v_a_2317_);
lean_inc_ref(v_a_2316_);
v___x_2448_ = lean_apply_9(v___x_20115__overap_2447_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_, lean_box(0));
return v___x_2448_;
}
v___jp_2449_:
{
lean_object* v___x_2463_; 
v___x_2463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2463_, 0, v_a_2462_);
v___y_2426_ = v___y_2450_;
v___y_2427_ = v___y_2451_;
v___y_2428_ = v___y_2452_;
v___y_2429_ = v___y_2454_;
v___y_2430_ = v___y_2453_;
v___y_2431_ = v___y_2455_;
v___y_2432_ = v___y_2456_;
v___y_2433_ = v___y_2458_;
v___y_2434_ = v___y_2457_;
v___y_2435_ = v___y_2460_;
v___y_2436_ = v___y_2459_;
v___y_2437_ = v___y_2461_;
v_a_2438_ = v___x_2463_;
goto v___jp_2425_;
}
v___jp_2464_:
{
lean_object* v___x_2479_; lean_object* v___x_2480_; double v___x_2481_; double v___x_2482_; double v___x_2483_; double v___x_2484_; double v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_20307__overap_2490_; lean_object* v___x_2491_; 
v___x_2479_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2479_);
v___x_2480_ = lean_io_mono_nanos_now();
v___x_2481_ = lean_float_of_nat(v___y_2472_);
v___x_2482_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_2483_ = lean_float_div(v___x_2481_, v___x_2482_);
v___x_2484_ = lean_float_of_nat(v___x_2480_);
v___x_2485_ = lean_float_div(v___x_2484_, v___x_2482_);
v___x_2486_ = lean_box_float(v___x_2483_);
v___x_2487_ = lean_box_float(v___x_2485_);
v___x_2488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2488_, 0, v___x_2486_);
lean_ctor_set(v___x_2488_, 1, v___x_2487_);
v___x_2489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2489_, 0, v_a_2478_);
lean_ctor_set(v___x_2489_, 1, v___x_2488_);
lean_inc_ref(v___y_2470_);
lean_inc_ref(v___y_2468_);
lean_inc(v___y_2467_);
lean_inc_ref(v___y_2469_);
v___x_20307__overap_2490_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_2328_, v___y_2469_, v___y_2476_, v___y_2467_, lean_box(0), v___y_2468_, v___y_2470_, v___y_2475_, v___y_2465_, v___x_2382_, v___y_2466_, v___y_2474_, v___y_2473_, v___y_2477_, v___x_2489_);
lean_inc(v_a_2323_);
lean_inc_ref(v_a_2322_);
lean_inc(v_a_2321_);
lean_inc_ref(v_a_2320_);
lean_inc(v_a_2319_);
lean_inc(v_a_2318_);
lean_inc(v_a_2317_);
lean_inc_ref(v_a_2316_);
v___x_2491_ = lean_apply_9(v___x_20307__overap_2490_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_, lean_box(0));
v___y_2338_ = v___y_2471_;
v___y_2339_ = v___x_2491_;
goto v___jp_2337_;
}
v___jp_2492_:
{
lean_object* v___x_2507_; 
v___x_2507_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2507_, 0, v_a_2506_);
v___y_2465_ = v___y_2493_;
v___y_2466_ = v___y_2494_;
v___y_2467_ = v___y_2495_;
v___y_2468_ = v___y_2496_;
v___y_2469_ = v___y_2497_;
v___y_2470_ = v___y_2498_;
v___y_2471_ = v___y_2499_;
v___y_2472_ = v___y_2500_;
v___y_2473_ = v___y_2501_;
v___y_2474_ = v___y_2502_;
v___y_2475_ = v___y_2503_;
v___y_2476_ = v___y_2504_;
v___y_2477_ = v___y_2505_;
v_a_2478_ = v___x_2507_;
goto v___jp_2464_;
}
v___jp_2508_:
{
lean_object* v___x_2523_; lean_object* v___x_2524_; double v___x_2525_; double v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_20342__overap_2531_; lean_object* v___x_2532_; 
v___x_2523_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2523_);
v___x_2524_ = lean_io_get_num_heartbeats();
v___x_2525_ = lean_float_of_nat(v___y_2509_);
v___x_2526_ = lean_float_of_nat(v___x_2524_);
v___x_2527_ = lean_box_float(v___x_2525_);
v___x_2528_ = lean_box_float(v___x_2526_);
v___x_2529_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2529_, 0, v___x_2527_);
lean_ctor_set(v___x_2529_, 1, v___x_2528_);
v___x_2530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2530_, 0, v_a_2522_);
lean_ctor_set(v___x_2530_, 1, v___x_2529_);
lean_inc_ref(v___y_2515_);
lean_inc_ref(v___y_2513_);
lean_inc(v___y_2512_);
lean_inc_ref(v___y_2514_);
v___x_20342__overap_2531_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_2328_, v___y_2514_, v___y_2520_, v___y_2512_, lean_box(0), v___y_2513_, v___y_2515_, v___y_2519_, v___y_2510_, v___x_2382_, v___y_2511_, v___y_2518_, v___y_2517_, v___y_2521_, v___x_2530_);
lean_inc(v_a_2323_);
lean_inc_ref(v_a_2322_);
lean_inc(v_a_2321_);
lean_inc_ref(v_a_2320_);
lean_inc(v_a_2319_);
lean_inc(v_a_2318_);
lean_inc(v_a_2317_);
lean_inc_ref(v_a_2316_);
v___x_2532_ = lean_apply_9(v___x_20342__overap_2531_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_, lean_box(0));
v___y_2338_ = v___y_2516_;
v___y_2339_ = v___x_2532_;
goto v___jp_2337_;
}
v___jp_2533_:
{
lean_object* v___x_2548_; 
v___x_2548_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2548_, 0, v_a_2547_);
v___y_2509_ = v___y_2534_;
v___y_2510_ = v___y_2535_;
v___y_2511_ = v___y_2536_;
v___y_2512_ = v___y_2537_;
v___y_2513_ = v___y_2538_;
v___y_2514_ = v___y_2539_;
v___y_2515_ = v___y_2540_;
v___y_2516_ = v___y_2541_;
v___y_2517_ = v___y_2542_;
v___y_2518_ = v___y_2543_;
v___y_2519_ = v___y_2544_;
v___y_2520_ = v___y_2545_;
v___y_2521_ = v___y_2546_;
v_a_2522_ = v___x_2548_;
goto v___jp_2508_;
}
v___jp_2550_:
{
lean_object* v___x_20051__overap_2561_; lean_object* v___x_2562_; 
lean_inc_ref(v___y_2559_);
lean_inc_ref(v___x_2328_);
v___x_20051__overap_2561_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_2328_, v___y_2559_);
lean_inc(v_a_2323_);
lean_inc_ref(v_a_2322_);
lean_inc(v_a_2321_);
lean_inc_ref(v_a_2320_);
lean_inc(v_a_2319_);
lean_inc(v_a_2318_);
lean_inc(v_a_2317_);
lean_inc_ref(v_a_2316_);
v___x_2562_ = lean_apply_9(v___x_20051__overap_2561_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_, lean_box(0));
if (lean_obj_tag(v___x_2562_) == 0)
{
lean_object* v_a_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; uint8_t v___x_2566_; 
v_a_2563_ = lean_ctor_get(v___x_2562_, 0);
lean_inc(v_a_2563_);
lean_dec_ref_known(v___x_2562_, 1);
v___x_2564_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2565_ = l_Lean_Option_get___redArg(v___x_2332_, v___y_2551_, v___x_2564_);
v___x_2566_ = lean_unbox(v___x_2565_);
lean_dec(v___x_2565_);
if (v___x_2566_ == 0)
{
lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; 
v___x_2567_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2567_);
v___x_2568_ = lean_io_mono_nanos_now();
v___x_2569_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_locations_2326_, v_patternSubsts_x3f_2327_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2569_) == 0)
{
lean_object* v_a_2570_; lean_object* v___x_2571_; 
v_a_2570_ = lean_ctor_get(v___x_2569_, 0);
lean_inc(v_a_2570_);
lean_dec_ref_known(v___x_2569_, 1);
v___x_2571_ = lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_a_2570_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2571_) == 0)
{
lean_object* v_a_2572_; lean_object* v___x_2574_; uint8_t v_isShared_2575_; uint8_t v_isSharedCheck_2579_; 
v_a_2572_ = lean_ctor_get(v___x_2571_, 0);
v_isSharedCheck_2579_ = !lean_is_exclusive(v___x_2571_);
if (v_isSharedCheck_2579_ == 0)
{
v___x_2574_ = v___x_2571_;
v_isShared_2575_ = v_isSharedCheck_2579_;
goto v_resetjp_2573_;
}
else
{
lean_inc(v_a_2572_);
lean_dec(v___x_2571_);
v___x_2574_ = lean_box(0);
v_isShared_2575_ = v_isSharedCheck_2579_;
goto v_resetjp_2573_;
}
v_resetjp_2573_:
{
lean_object* v___x_2577_; 
if (v_isShared_2575_ == 0)
{
lean_ctor_set_tag(v___x_2574_, 1);
v___x_2577_ = v___x_2574_;
goto v_reusejp_2576_;
}
else
{
lean_object* v_reuseFailAlloc_2578_; 
v_reuseFailAlloc_2578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2578_, 0, v_a_2572_);
v___x_2577_ = v_reuseFailAlloc_2578_;
goto v_reusejp_2576_;
}
v_reusejp_2576_:
{
v___y_2384_ = v___y_2551_;
v___y_2385_ = v___y_2552_;
v___y_2386_ = v___y_2553_;
v___y_2387_ = v___y_2555_;
v___y_2388_ = v___y_2554_;
v___y_2389_ = v___x_2568_;
v___y_2390_ = v_a_2563_;
v___y_2391_ = v___y_2557_;
v___y_2392_ = v___y_2556_;
v___y_2393_ = v___y_2558_;
v___y_2394_ = v___y_2559_;
v___y_2395_ = v___y_2560_;
v_a_2396_ = v___x_2577_;
goto v___jp_2383_;
}
}
}
else
{
lean_object* v_a_2580_; 
v_a_2580_ = lean_ctor_get(v___x_2571_, 0);
lean_inc(v_a_2580_);
lean_dec_ref_known(v___x_2571_, 1);
v___y_2411_ = v___y_2551_;
v___y_2412_ = v___y_2552_;
v___y_2413_ = v___y_2553_;
v___y_2414_ = v___y_2554_;
v___y_2415_ = v___y_2555_;
v___y_2416_ = v___x_2568_;
v___y_2417_ = v_a_2563_;
v___y_2418_ = v___y_2556_;
v___y_2419_ = v___y_2557_;
v___y_2420_ = v___y_2559_;
v___y_2421_ = v___y_2558_;
v___y_2422_ = v___y_2560_;
v_a_2423_ = v_a_2580_;
goto v___jp_2410_;
}
}
else
{
lean_object* v_a_2581_; 
lean_dec_ref_known(v___x_2549_, 1);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2581_ = lean_ctor_get(v___x_2569_, 0);
lean_inc(v_a_2581_);
lean_dec_ref_known(v___x_2569_, 1);
v___y_2411_ = v___y_2551_;
v___y_2412_ = v___y_2552_;
v___y_2413_ = v___y_2553_;
v___y_2414_ = v___y_2554_;
v___y_2415_ = v___y_2555_;
v___y_2416_ = v___x_2568_;
v___y_2417_ = v_a_2563_;
v___y_2418_ = v___y_2556_;
v___y_2419_ = v___y_2557_;
v___y_2420_ = v___y_2559_;
v___y_2421_ = v___y_2558_;
v___y_2422_ = v___y_2560_;
v_a_2423_ = v_a_2581_;
goto v___jp_2410_;
}
}
else
{
lean_object* v___x_2582_; lean_object* v___x_2583_; lean_object* v___x_2584_; 
v___x_2582_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2582_);
v___x_2583_ = lean_io_get_num_heartbeats();
v___x_2584_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_locations_2326_, v_patternSubsts_x3f_2327_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2584_) == 0)
{
lean_object* v_a_2585_; lean_object* v___x_2586_; 
v_a_2585_ = lean_ctor_get(v___x_2584_, 0);
lean_inc(v_a_2585_);
lean_dec_ref_known(v___x_2584_, 1);
v___x_2586_ = lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_a_2585_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2586_) == 0)
{
lean_object* v_a_2587_; lean_object* v___x_2589_; uint8_t v_isShared_2590_; uint8_t v_isSharedCheck_2594_; 
v_a_2587_ = lean_ctor_get(v___x_2586_, 0);
v_isSharedCheck_2594_ = !lean_is_exclusive(v___x_2586_);
if (v_isSharedCheck_2594_ == 0)
{
v___x_2589_ = v___x_2586_;
v_isShared_2590_ = v_isSharedCheck_2594_;
goto v_resetjp_2588_;
}
else
{
lean_inc(v_a_2587_);
lean_dec(v___x_2586_);
v___x_2589_ = lean_box(0);
v_isShared_2590_ = v_isSharedCheck_2594_;
goto v_resetjp_2588_;
}
v_resetjp_2588_:
{
lean_object* v___x_2592_; 
if (v_isShared_2590_ == 0)
{
lean_ctor_set_tag(v___x_2589_, 1);
v___x_2592_ = v___x_2589_;
goto v_reusejp_2591_;
}
else
{
lean_object* v_reuseFailAlloc_2593_; 
v_reuseFailAlloc_2593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2593_, 0, v_a_2587_);
v___x_2592_ = v_reuseFailAlloc_2593_;
goto v_reusejp_2591_;
}
v_reusejp_2591_:
{
v___y_2426_ = v___y_2551_;
v___y_2427_ = v___y_2552_;
v___y_2428_ = v___y_2553_;
v___y_2429_ = v___y_2555_;
v___y_2430_ = v___y_2554_;
v___y_2431_ = v___x_2583_;
v___y_2432_ = v_a_2563_;
v___y_2433_ = v___y_2557_;
v___y_2434_ = v___y_2556_;
v___y_2435_ = v___y_2558_;
v___y_2436_ = v___y_2559_;
v___y_2437_ = v___y_2560_;
v_a_2438_ = v___x_2592_;
goto v___jp_2425_;
}
}
}
else
{
lean_object* v_a_2595_; 
v_a_2595_ = lean_ctor_get(v___x_2586_, 0);
lean_inc(v_a_2595_);
lean_dec_ref_known(v___x_2586_, 1);
v___y_2450_ = v___y_2551_;
v___y_2451_ = v___y_2552_;
v___y_2452_ = v___y_2553_;
v___y_2453_ = v___y_2554_;
v___y_2454_ = v___y_2555_;
v___y_2455_ = v___x_2583_;
v___y_2456_ = v_a_2563_;
v___y_2457_ = v___y_2556_;
v___y_2458_ = v___y_2557_;
v___y_2459_ = v___y_2559_;
v___y_2460_ = v___y_2558_;
v___y_2461_ = v___y_2560_;
v_a_2462_ = v_a_2595_;
goto v___jp_2449_;
}
}
else
{
lean_object* v_a_2596_; 
lean_dec_ref_known(v___x_2549_, 1);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2596_ = lean_ctor_get(v___x_2584_, 0);
lean_inc(v_a_2596_);
lean_dec_ref_known(v___x_2584_, 1);
v___y_2450_ = v___y_2551_;
v___y_2451_ = v___y_2552_;
v___y_2452_ = v___y_2553_;
v___y_2453_ = v___y_2554_;
v___y_2454_ = v___y_2555_;
v___y_2455_ = v___x_2583_;
v___y_2456_ = v_a_2563_;
v___y_2457_ = v___y_2556_;
v___y_2458_ = v___y_2557_;
v___y_2459_ = v___y_2559_;
v___y_2460_ = v___y_2558_;
v___y_2461_ = v___y_2560_;
v_a_2462_ = v_a_2596_;
goto v___jp_2449_;
}
}
}
else
{
lean_object* v_a_2597_; lean_object* v___x_2599_; uint8_t v_isShared_2600_; uint8_t v_isSharedCheck_2604_; 
lean_dec_ref(v___y_2560_);
lean_dec(v___y_2554_);
lean_dec_ref(v___y_2552_);
lean_dec_ref_known(v___x_2549_, 1);
lean_dec_ref(v___x_2328_);
lean_dec(v_patternSubsts_x3f_2327_);
lean_dec_ref(v_locations_2326_);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2597_ = lean_ctor_get(v___x_2562_, 0);
v_isSharedCheck_2604_ = !lean_is_exclusive(v___x_2562_);
if (v_isSharedCheck_2604_ == 0)
{
v___x_2599_ = v___x_2562_;
v_isShared_2600_ = v_isSharedCheck_2604_;
goto v_resetjp_2598_;
}
else
{
lean_inc(v_a_2597_);
lean_dec(v___x_2562_);
v___x_2599_ = lean_box(0);
v_isShared_2600_ = v_isSharedCheck_2604_;
goto v_resetjp_2598_;
}
v_resetjp_2598_:
{
lean_object* v___x_2602_; 
if (v_isShared_2600_ == 0)
{
v___x_2602_ = v___x_2599_;
goto v_reusejp_2601_;
}
else
{
lean_object* v_reuseFailAlloc_2603_; 
v_reuseFailAlloc_2603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2603_, 0, v_a_2597_);
v___x_2602_ = v_reuseFailAlloc_2603_;
goto v_reusejp_2601_;
}
v_reusejp_2601_:
{
return v___x_2602_;
}
}
}
}
v___jp_2605_:
{
lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; uint8_t v_hasTrace_2609_; 
v___x_2606_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_2607_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_2313_);
v___x_2608_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_hasTrace_2609_ = lean_ctor_get_uint8(v_options_2333_, sizeof(void*)*1);
if (v_hasTrace_2609_ == 0)
{
lean_object* v___x_2610_; 
lean_dec_ref(v___x_2607_);
lean_dec_ref(v_name_2330_);
lean_dec_ref(v___x_2328_);
v___x_2610_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_locations_2326_, v_patternSubsts_x3f_2327_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2610_) == 0)
{
lean_object* v_a_2611_; 
v_a_2611_ = lean_ctor_get(v___x_2610_, 0);
lean_inc(v_a_2611_);
lean_dec_ref_known(v___x_2610_, 1);
if (lean_obj_tag(v_a_2611_) == 1)
{
lean_object* v_val_2612_; lean_object* v___x_2613_; 
v_val_2612_ = lean_ctor_get(v_a_2611_, 0);
lean_inc(v_val_2612_);
lean_dec_ref_known(v_a_2611_, 1);
v___x_2613_ = lp_aesop_Aesop_addRapps___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_val_2612_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
return v___x_2613_;
}
else
{
lean_object* v___x_2614_; lean_object* v___x_2616_; uint8_t v_isShared_2617_; uint8_t v_isSharedCheck_2622_; 
lean_dec(v_a_2611_);
lean_dec_ref(v_inst_2313_);
v___x_2614_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_addRuleFailure___redArg(v___x_2549_, v_parentRef_2314_, v_a_2317_);
lean_dec(v_parentRef_2314_);
v_isSharedCheck_2622_ = !lean_is_exclusive(v___x_2614_);
if (v_isSharedCheck_2622_ == 0)
{
lean_object* v_unused_2623_; 
v_unused_2623_ = lean_ctor_get(v___x_2614_, 0);
lean_dec(v_unused_2623_);
v___x_2616_ = v___x_2614_;
v_isShared_2617_ = v_isSharedCheck_2622_;
goto v_resetjp_2615_;
}
else
{
lean_dec(v___x_2614_);
v___x_2616_ = lean_box(0);
v_isShared_2617_ = v_isSharedCheck_2622_;
goto v_resetjp_2615_;
}
v_resetjp_2615_:
{
lean_object* v___x_2618_; lean_object* v___x_2620_; 
v___x_2618_ = lean_box(2);
if (v_isShared_2617_ == 0)
{
lean_ctor_set(v___x_2616_, 0, v___x_2618_);
v___x_2620_ = v___x_2616_;
goto v_reusejp_2619_;
}
else
{
lean_object* v_reuseFailAlloc_2621_; 
v_reuseFailAlloc_2621_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2621_, 0, v___x_2618_);
v___x_2620_ = v_reuseFailAlloc_2621_;
goto v_reusejp_2619_;
}
v_reusejp_2619_:
{
return v___x_2620_;
}
}
}
}
else
{
lean_object* v_a_2624_; lean_object* v___x_2626_; uint8_t v_isShared_2627_; uint8_t v_isSharedCheck_2631_; 
lean_dec_ref_known(v___x_2549_, 1);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2624_ = lean_ctor_get(v___x_2610_, 0);
v_isSharedCheck_2631_ = !lean_is_exclusive(v___x_2610_);
if (v_isSharedCheck_2631_ == 0)
{
v___x_2626_ = v___x_2610_;
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
else
{
lean_inc(v_a_2624_);
lean_dec(v___x_2610_);
v___x_2626_ = lean_box(0);
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
v_resetjp_2625_:
{
lean_object* v___x_2629_; 
if (v_isShared_2627_ == 0)
{
v___x_2629_ = v___x_2626_;
goto v_reusejp_2628_;
}
else
{
lean_object* v_reuseFailAlloc_2630_; 
v_reuseFailAlloc_2630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2630_, 0, v_a_2624_);
v___x_2629_ = v_reuseFailAlloc_2630_;
goto v_reusejp_2628_;
}
v_reusejp_2628_:
{
return v___x_2629_;
}
}
}
}
else
{
lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v_traceClass_2634_; lean_object* v___f_2635_; lean_object* v___f_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; uint8_t v___x_2640_; 
v___x_2632_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2632_);
v___x_2633_ = lp_aesop_Aesop_TraceOption_steps;
v_traceClass_2634_ = lean_ctor_get(v___x_2633_, 0);
v___f_2635_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___f_2636_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
lean_inc_ref(v_inst_2313_);
v___x_2637_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed), 16, 6);
lean_closure_set(v___x_2637_, 0, lean_box(0));
lean_closure_set(v___x_2637_, 1, v_inst_2313_);
lean_closure_set(v___x_2637_, 2, lean_box(0));
lean_closure_set(v___x_2637_, 3, v_name_2330_);
lean_closure_set(v___x_2637_, 4, v___f_2335_);
lean_closure_set(v___x_2637_, 5, v___x_2382_);
v___x_2638_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_2634_);
v___x_2639_ = l_Lean_Name_append(v___x_2638_, v_traceClass_2634_);
v___x_2640_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2334_, v_options_2333_, v___x_2639_);
lean_dec(v___x_2639_);
if (v___x_2640_ == 0)
{
lean_object* v___x_2641_; lean_object* v___x_2642_; uint8_t v___x_2643_; 
v___x_2641_ = l_Lean_trace_profiler;
v___x_2642_ = l_Lean_Option_get___redArg(v___x_2332_, v_options_2333_, v___x_2641_);
v___x_2643_ = lean_unbox(v___x_2642_);
lean_dec(v___x_2642_);
if (v___x_2643_ == 0)
{
lean_object* v___x_2644_; 
lean_dec_ref(v___x_2637_);
lean_dec_ref(v___x_2607_);
lean_dec_ref(v___x_2328_);
v___x_2644_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_locations_2326_, v_patternSubsts_x3f_2327_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2644_) == 0)
{
lean_object* v_a_2645_; lean_object* v___x_2646_; 
v_a_2645_ = lean_ctor_get(v___x_2644_, 0);
lean_inc(v_a_2645_);
lean_dec_ref_known(v___x_2644_, 1);
v___x_2646_ = lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_a_2645_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
return v___x_2646_;
}
else
{
lean_object* v_a_2647_; lean_object* v___x_2649_; uint8_t v_isShared_2650_; uint8_t v_isSharedCheck_2654_; 
lean_dec_ref_known(v___x_2549_, 1);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2647_ = lean_ctor_get(v___x_2644_, 0);
v_isSharedCheck_2654_ = !lean_is_exclusive(v___x_2644_);
if (v_isSharedCheck_2654_ == 0)
{
v___x_2649_ = v___x_2644_;
v_isShared_2650_ = v_isSharedCheck_2654_;
goto v_resetjp_2648_;
}
else
{
lean_inc(v_a_2647_);
lean_dec(v___x_2644_);
v___x_2649_ = lean_box(0);
v_isShared_2650_ = v_isSharedCheck_2654_;
goto v_resetjp_2648_;
}
v_resetjp_2648_:
{
lean_object* v___x_2652_; 
if (v_isShared_2650_ == 0)
{
v___x_2652_ = v___x_2649_;
goto v_reusejp_2651_;
}
else
{
lean_object* v_reuseFailAlloc_2653_; 
v_reuseFailAlloc_2653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2653_, 0, v_a_2647_);
v___x_2652_ = v_reuseFailAlloc_2653_;
goto v_reusejp_2651_;
}
v_reusejp_2651_:
{
return v___x_2652_;
}
}
}
}
else
{
lean_inc(v_traceClass_2634_);
v___y_2551_ = v_options_2333_;
v___y_2552_ = v___x_2607_;
v___y_2553_ = v___f_2635_;
v___y_2554_ = v_traceClass_2634_;
v___y_2555_ = v_hasTrace_2609_;
v___y_2556_ = v___x_2608_;
v___y_2557_ = v___f_2636_;
v___y_2558_ = v___x_2640_;
v___y_2559_ = v___x_2606_;
v___y_2560_ = v___x_2637_;
goto v___jp_2550_;
}
}
else
{
lean_inc(v_traceClass_2634_);
v___y_2551_ = v_options_2333_;
v___y_2552_ = v___x_2607_;
v___y_2553_ = v___f_2635_;
v___y_2554_ = v_traceClass_2634_;
v___y_2555_ = v_hasTrace_2609_;
v___y_2556_ = v___x_2608_;
v___y_2557_ = v___f_2636_;
v___y_2558_ = v___x_2640_;
v___y_2559_ = v___x_2606_;
v___y_2560_ = v___x_2637_;
goto v___jp_2550_;
}
}
}
v___jp_2655_:
{
lean_object* v___x_20278__overap_2667_; lean_object* v___x_2668_; 
lean_inc_ref(v___y_2665_);
lean_inc_ref(v___x_2328_);
v___x_20278__overap_2667_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_2328_, v___y_2665_);
lean_inc(v_a_2323_);
lean_inc_ref(v_a_2322_);
lean_inc(v_a_2321_);
lean_inc_ref(v_a_2320_);
lean_inc(v_a_2319_);
lean_inc(v_a_2318_);
lean_inc(v_a_2317_);
lean_inc_ref(v_a_2316_);
v___x_2668_ = lean_apply_9(v___x_20278__overap_2667_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_, lean_box(0));
if (lean_obj_tag(v___x_2668_) == 0)
{
lean_object* v_a_2669_; lean_object* v___x_2670_; lean_object* v___x_2671_; uint8_t v___x_2672_; 
v_a_2669_ = lean_ctor_get(v___x_2668_, 0);
lean_inc(v_a_2669_);
lean_dec_ref_known(v___x_2668_, 1);
v___x_2670_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2671_ = l_Lean_Option_get___redArg(v___x_2332_, v___y_2659_, v___x_2670_);
v___x_2672_ = lean_unbox(v___x_2671_);
lean_dec(v___x_2671_);
if (v___x_2672_ == 0)
{
lean_object* v___x_2673_; lean_object* v___x_2674_; lean_object* v___x_2675_; 
v___x_2673_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2673_);
v___x_2674_ = lean_io_mono_nanos_now();
v___x_2675_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_locations_2326_, v_patternSubsts_x3f_2327_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2675_) == 0)
{
lean_object* v_a_2676_; lean_object* v___x_2677_; 
v_a_2676_ = lean_ctor_get(v___x_2675_, 0);
lean_inc(v_a_2676_);
lean_dec_ref_known(v___x_2675_, 1);
v___x_2677_ = lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_a_2676_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2677_) == 0)
{
lean_object* v_a_2678_; lean_object* v___x_2680_; uint8_t v_isShared_2681_; uint8_t v_isSharedCheck_2685_; 
v_a_2678_ = lean_ctor_get(v___x_2677_, 0);
v_isSharedCheck_2685_ = !lean_is_exclusive(v___x_2677_);
if (v_isSharedCheck_2685_ == 0)
{
v___x_2680_ = v___x_2677_;
v_isShared_2681_ = v_isSharedCheck_2685_;
goto v_resetjp_2679_;
}
else
{
lean_inc(v_a_2678_);
lean_dec(v___x_2677_);
v___x_2680_ = lean_box(0);
v_isShared_2681_ = v_isSharedCheck_2685_;
goto v_resetjp_2679_;
}
v_resetjp_2679_:
{
lean_object* v___x_2683_; 
if (v_isShared_2681_ == 0)
{
lean_ctor_set_tag(v___x_2680_, 1);
v___x_2683_ = v___x_2680_;
goto v_reusejp_2682_;
}
else
{
lean_object* v_reuseFailAlloc_2684_; 
v_reuseFailAlloc_2684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2684_, 0, v_a_2678_);
v___x_2683_ = v_reuseFailAlloc_2684_;
goto v_reusejp_2682_;
}
v_reusejp_2682_:
{
v___y_2465_ = v___y_2658_;
v___y_2466_ = v___y_2659_;
v___y_2467_ = v___y_2660_;
v___y_2468_ = v___y_2662_;
v___y_2469_ = v___y_2665_;
v___y_2470_ = v___y_2656_;
v___y_2471_ = v___y_2657_;
v___y_2472_ = v___x_2674_;
v___y_2473_ = v_a_2669_;
v___y_2474_ = v___y_2661_;
v___y_2475_ = v___y_2663_;
v___y_2476_ = v___y_2664_;
v___y_2477_ = v___y_2666_;
v_a_2478_ = v___x_2683_;
goto v___jp_2464_;
}
}
}
else
{
lean_object* v_a_2686_; 
v_a_2686_ = lean_ctor_get(v___x_2677_, 0);
lean_inc(v_a_2686_);
lean_dec_ref_known(v___x_2677_, 1);
v___y_2493_ = v___y_2658_;
v___y_2494_ = v___y_2659_;
v___y_2495_ = v___y_2660_;
v___y_2496_ = v___y_2662_;
v___y_2497_ = v___y_2665_;
v___y_2498_ = v___y_2656_;
v___y_2499_ = v___y_2657_;
v___y_2500_ = v___x_2674_;
v___y_2501_ = v_a_2669_;
v___y_2502_ = v___y_2661_;
v___y_2503_ = v___y_2663_;
v___y_2504_ = v___y_2664_;
v___y_2505_ = v___y_2666_;
v_a_2506_ = v_a_2686_;
goto v___jp_2492_;
}
}
else
{
lean_object* v_a_2687_; 
lean_dec_ref_known(v___x_2549_, 1);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2687_ = lean_ctor_get(v___x_2675_, 0);
lean_inc(v_a_2687_);
lean_dec_ref_known(v___x_2675_, 1);
v___y_2493_ = v___y_2658_;
v___y_2494_ = v___y_2659_;
v___y_2495_ = v___y_2660_;
v___y_2496_ = v___y_2662_;
v___y_2497_ = v___y_2665_;
v___y_2498_ = v___y_2656_;
v___y_2499_ = v___y_2657_;
v___y_2500_ = v___x_2674_;
v___y_2501_ = v_a_2669_;
v___y_2502_ = v___y_2661_;
v___y_2503_ = v___y_2663_;
v___y_2504_ = v___y_2664_;
v___y_2505_ = v___y_2666_;
v_a_2506_ = v_a_2687_;
goto v___jp_2492_;
}
}
else
{
lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; 
v___x_2688_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2688_);
v___x_2689_ = lean_io_get_num_heartbeats();
v___x_2690_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_locations_2326_, v_patternSubsts_x3f_2327_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2690_) == 0)
{
lean_object* v_a_2691_; lean_object* v___x_2692_; 
v_a_2691_ = lean_ctor_get(v___x_2690_, 0);
lean_inc(v_a_2691_);
lean_dec_ref_known(v___x_2690_, 1);
v___x_2692_ = lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_a_2691_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2692_) == 0)
{
lean_object* v_a_2693_; lean_object* v___x_2695_; uint8_t v_isShared_2696_; uint8_t v_isSharedCheck_2700_; 
v_a_2693_ = lean_ctor_get(v___x_2692_, 0);
v_isSharedCheck_2700_ = !lean_is_exclusive(v___x_2692_);
if (v_isSharedCheck_2700_ == 0)
{
v___x_2695_ = v___x_2692_;
v_isShared_2696_ = v_isSharedCheck_2700_;
goto v_resetjp_2694_;
}
else
{
lean_inc(v_a_2693_);
lean_dec(v___x_2692_);
v___x_2695_ = lean_box(0);
v_isShared_2696_ = v_isSharedCheck_2700_;
goto v_resetjp_2694_;
}
v_resetjp_2694_:
{
lean_object* v___x_2698_; 
if (v_isShared_2696_ == 0)
{
lean_ctor_set_tag(v___x_2695_, 1);
v___x_2698_ = v___x_2695_;
goto v_reusejp_2697_;
}
else
{
lean_object* v_reuseFailAlloc_2699_; 
v_reuseFailAlloc_2699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2699_, 0, v_a_2693_);
v___x_2698_ = v_reuseFailAlloc_2699_;
goto v_reusejp_2697_;
}
v_reusejp_2697_:
{
v___y_2509_ = v___x_2689_;
v___y_2510_ = v___y_2658_;
v___y_2511_ = v___y_2659_;
v___y_2512_ = v___y_2660_;
v___y_2513_ = v___y_2662_;
v___y_2514_ = v___y_2665_;
v___y_2515_ = v___y_2656_;
v___y_2516_ = v___y_2657_;
v___y_2517_ = v_a_2669_;
v___y_2518_ = v___y_2661_;
v___y_2519_ = v___y_2663_;
v___y_2520_ = v___y_2664_;
v___y_2521_ = v___y_2666_;
v_a_2522_ = v___x_2698_;
goto v___jp_2508_;
}
}
}
else
{
lean_object* v_a_2701_; 
v_a_2701_ = lean_ctor_get(v___x_2692_, 0);
lean_inc(v_a_2701_);
lean_dec_ref_known(v___x_2692_, 1);
v___y_2534_ = v___x_2689_;
v___y_2535_ = v___y_2658_;
v___y_2536_ = v___y_2659_;
v___y_2537_ = v___y_2660_;
v___y_2538_ = v___y_2662_;
v___y_2539_ = v___y_2665_;
v___y_2540_ = v___y_2656_;
v___y_2541_ = v___y_2657_;
v___y_2542_ = v_a_2669_;
v___y_2543_ = v___y_2661_;
v___y_2544_ = v___y_2663_;
v___y_2545_ = v___y_2664_;
v___y_2546_ = v___y_2666_;
v_a_2547_ = v_a_2701_;
goto v___jp_2533_;
}
}
else
{
lean_object* v_a_2702_; 
lean_dec_ref_known(v___x_2549_, 1);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2702_ = lean_ctor_get(v___x_2690_, 0);
lean_inc(v_a_2702_);
lean_dec_ref_known(v___x_2690_, 1);
v___y_2534_ = v___x_2689_;
v___y_2535_ = v___y_2658_;
v___y_2536_ = v___y_2659_;
v___y_2537_ = v___y_2660_;
v___y_2538_ = v___y_2662_;
v___y_2539_ = v___y_2665_;
v___y_2540_ = v___y_2656_;
v___y_2541_ = v___y_2657_;
v___y_2542_ = v_a_2669_;
v___y_2543_ = v___y_2661_;
v___y_2544_ = v___y_2663_;
v___y_2545_ = v___y_2664_;
v___y_2546_ = v___y_2666_;
v_a_2547_ = v_a_2702_;
goto v___jp_2533_;
}
}
}
else
{
lean_object* v_a_2703_; lean_object* v___x_2705_; uint8_t v_isShared_2706_; uint8_t v_isSharedCheck_2710_; 
lean_dec_ref(v___y_2666_);
lean_dec_ref(v___y_2664_);
lean_dec(v___y_2663_);
lean_dec(v___y_2657_);
lean_dec_ref_known(v___x_2549_, 1);
lean_dec_ref_known(v___x_2336_, 1);
lean_dec_ref(v___x_2328_);
lean_dec(v_patternSubsts_x3f_2327_);
lean_dec_ref(v_locations_2326_);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2703_ = lean_ctor_get(v___x_2668_, 0);
v_isSharedCheck_2710_ = !lean_is_exclusive(v___x_2668_);
if (v_isSharedCheck_2710_ == 0)
{
v___x_2705_ = v___x_2668_;
v_isShared_2706_ = v_isSharedCheck_2710_;
goto v_resetjp_2704_;
}
else
{
lean_inc(v_a_2703_);
lean_dec(v___x_2668_);
v___x_2705_ = lean_box(0);
v_isShared_2706_ = v_isSharedCheck_2710_;
goto v_resetjp_2704_;
}
v_resetjp_2704_:
{
lean_object* v___x_2708_; 
if (v_isShared_2706_ == 0)
{
v___x_2708_ = v___x_2705_;
goto v_reusejp_2707_;
}
else
{
lean_object* v_reuseFailAlloc_2709_; 
v_reuseFailAlloc_2709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2709_, 0, v_a_2703_);
v___x_2708_ = v_reuseFailAlloc_2709_;
goto v_reusejp_2707_;
}
v_reusejp_2707_:
{
return v___x_2708_;
}
}
}
}
v___jp_2711_:
{
lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; uint8_t v_hasTrace_2717_; 
v___x_2712_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2712_);
v___x_2713_ = lean_io_mono_nanos_now();
v___x_2714_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_2715_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_2313_);
v___x_2716_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_hasTrace_2717_ = lean_ctor_get_uint8(v_options_2333_, sizeof(void*)*1);
if (v_hasTrace_2717_ == 0)
{
lean_object* v___x_2718_; 
lean_dec_ref(v___x_2715_);
lean_dec_ref(v_name_2330_);
lean_dec_ref(v___x_2328_);
v___x_2718_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_locations_2326_, v_patternSubsts_x3f_2327_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2718_) == 0)
{
lean_object* v_a_2719_; lean_object* v___x_2720_; 
v_a_2719_ = lean_ctor_get(v___x_2718_, 0);
lean_inc(v_a_2719_);
lean_dec_ref_known(v___x_2718_, 1);
v___x_2720_ = lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_a_2719_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
v___y_2338_ = v___x_2713_;
v___y_2339_ = v___x_2720_;
goto v___jp_2337_;
}
else
{
lean_object* v_a_2721_; lean_object* v___x_2723_; uint8_t v_isShared_2724_; uint8_t v_isSharedCheck_2728_; 
lean_dec(v___x_2713_);
lean_dec_ref_known(v___x_2549_, 1);
lean_dec_ref_known(v___x_2336_, 1);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2721_ = lean_ctor_get(v___x_2718_, 0);
v_isSharedCheck_2728_ = !lean_is_exclusive(v___x_2718_);
if (v_isSharedCheck_2728_ == 0)
{
v___x_2723_ = v___x_2718_;
v_isShared_2724_ = v_isSharedCheck_2728_;
goto v_resetjp_2722_;
}
else
{
lean_inc(v_a_2721_);
lean_dec(v___x_2718_);
v___x_2723_ = lean_box(0);
v_isShared_2724_ = v_isSharedCheck_2728_;
goto v_resetjp_2722_;
}
v_resetjp_2722_:
{
lean_object* v___x_2726_; 
if (v_isShared_2724_ == 0)
{
v___x_2726_ = v___x_2723_;
goto v_reusejp_2725_;
}
else
{
lean_object* v_reuseFailAlloc_2727_; 
v_reuseFailAlloc_2727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2727_, 0, v_a_2721_);
v___x_2726_ = v_reuseFailAlloc_2727_;
goto v_reusejp_2725_;
}
v_reusejp_2725_:
{
return v___x_2726_;
}
}
}
}
else
{
lean_object* v___x_2729_; lean_object* v___x_2730_; lean_object* v_traceClass_2731_; lean_object* v___f_2732_; lean_object* v___f_2733_; lean_object* v___x_2734_; lean_object* v___x_2735_; lean_object* v___x_2736_; uint8_t v___x_2737_; 
v___x_2729_ = lean_st_ref_get(v_a_2317_);
lean_dec(v___x_2729_);
v___x_2730_ = lp_aesop_Aesop_TraceOption_steps;
v_traceClass_2731_ = lean_ctor_get(v___x_2730_, 0);
v___f_2732_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___f_2733_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
lean_inc_ref(v_inst_2313_);
v___x_2734_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed), 16, 6);
lean_closure_set(v___x_2734_, 0, lean_box(0));
lean_closure_set(v___x_2734_, 1, v_inst_2313_);
lean_closure_set(v___x_2734_, 2, lean_box(0));
lean_closure_set(v___x_2734_, 3, v_name_2330_);
lean_closure_set(v___x_2734_, 4, v___f_2335_);
lean_closure_set(v___x_2734_, 5, v___x_2382_);
v___x_2735_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_2731_);
v___x_2736_ = l_Lean_Name_append(v___x_2735_, v_traceClass_2731_);
v___x_2737_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2334_, v_options_2333_, v___x_2736_);
lean_dec(v___x_2736_);
if (v___x_2737_ == 0)
{
lean_object* v___x_2738_; lean_object* v___x_2739_; uint8_t v___x_2740_; 
v___x_2738_ = l_Lean_trace_profiler;
v___x_2739_ = l_Lean_Option_get___redArg(v___x_2332_, v_options_2333_, v___x_2738_);
v___x_2740_ = lean_unbox(v___x_2739_);
lean_dec(v___x_2739_);
if (v___x_2740_ == 0)
{
lean_object* v___x_2741_; 
lean_dec_ref(v___x_2734_);
lean_dec_ref(v___x_2715_);
lean_dec_ref(v___x_2328_);
v___x_2741_ = lp_aesop_Aesop_runRegularRuleCore___redArg(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_locations_2326_, v_patternSubsts_x3f_2327_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
if (lean_obj_tag(v___x_2741_) == 0)
{
lean_object* v_a_2742_; lean_object* v___x_2743_; 
v_a_2742_ = lean_ctor_get(v___x_2741_, 0);
lean_inc(v_a_2742_);
lean_dec_ref_known(v___x_2741_, 1);
v___x_2743_ = lp_aesop_Aesop_runUnsafeRule___redArg___lam__0(v_inst_2313_, v_parentRef_2314_, v___x_2549_, v_a_2742_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_);
v___y_2338_ = v___x_2713_;
v___y_2339_ = v___x_2743_;
goto v___jp_2337_;
}
else
{
lean_object* v_a_2744_; lean_object* v___x_2746_; uint8_t v_isShared_2747_; uint8_t v_isSharedCheck_2751_; 
lean_dec(v___x_2713_);
lean_dec_ref_known(v___x_2549_, 1);
lean_dec_ref_known(v___x_2336_, 1);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2744_ = lean_ctor_get(v___x_2741_, 0);
v_isSharedCheck_2751_ = !lean_is_exclusive(v___x_2741_);
if (v_isSharedCheck_2751_ == 0)
{
v___x_2746_ = v___x_2741_;
v_isShared_2747_ = v_isSharedCheck_2751_;
goto v_resetjp_2745_;
}
else
{
lean_inc(v_a_2744_);
lean_dec(v___x_2741_);
v___x_2746_ = lean_box(0);
v_isShared_2747_ = v_isSharedCheck_2751_;
goto v_resetjp_2745_;
}
v_resetjp_2745_:
{
lean_object* v___x_2749_; 
if (v_isShared_2747_ == 0)
{
v___x_2749_ = v___x_2746_;
goto v_reusejp_2748_;
}
else
{
lean_object* v_reuseFailAlloc_2750_; 
v_reuseFailAlloc_2750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2750_, 0, v_a_2744_);
v___x_2749_ = v_reuseFailAlloc_2750_;
goto v_reusejp_2748_;
}
v_reusejp_2748_:
{
return v___x_2749_;
}
}
}
}
else
{
lean_inc(v_traceClass_2731_);
v___y_2656_ = v___f_2733_;
v___y_2657_ = v___x_2713_;
v___y_2658_ = v_hasTrace_2717_;
v___y_2659_ = v_options_2333_;
v___y_2660_ = v___f_2732_;
v___y_2661_ = v___x_2737_;
v___y_2662_ = v___x_2716_;
v___y_2663_ = v_traceClass_2731_;
v___y_2664_ = v___x_2715_;
v___y_2665_ = v___x_2714_;
v___y_2666_ = v___x_2734_;
goto v___jp_2655_;
}
}
else
{
lean_inc(v_traceClass_2731_);
v___y_2656_ = v___f_2733_;
v___y_2657_ = v___x_2713_;
v___y_2658_ = v_hasTrace_2717_;
v___y_2659_ = v_options_2333_;
v___y_2660_ = v___f_2732_;
v___y_2661_ = v___x_2737_;
v___y_2662_ = v___x_2716_;
v___y_2663_ = v_traceClass_2731_;
v___y_2664_ = v___x_2715_;
v___y_2665_ = v___x_2714_;
v___y_2666_ = v___x_2734_;
goto v___jp_2655_;
}
}
}
v___jp_2752_:
{
if (lean_obj_tag(v___y_2753_) == 0)
{
lean_object* v_a_2754_; uint8_t v___x_2755_; 
v_a_2754_ = lean_ctor_get(v___y_2753_, 0);
lean_inc(v_a_2754_);
lean_dec_ref_known(v___y_2753_, 1);
v___x_2755_ = lean_unbox(v_a_2754_);
lean_dec(v_a_2754_);
if (v___x_2755_ == 0)
{
lean_dec_ref_known(v___x_2336_, 1);
goto v___jp_2605_;
}
else
{
goto v___jp_2711_;
}
}
else
{
lean_object* v_a_2756_; lean_object* v___x_2758_; uint8_t v_isShared_2759_; uint8_t v_isSharedCheck_2763_; 
lean_dec_ref_known(v___x_2549_, 1);
lean_dec_ref_known(v___x_2336_, 1);
lean_dec_ref(v_name_2330_);
lean_dec_ref(v___x_2328_);
lean_dec(v_patternSubsts_x3f_2327_);
lean_dec_ref(v_locations_2326_);
lean_dec(v_parentRef_2314_);
lean_dec_ref(v_inst_2313_);
v_a_2756_ = lean_ctor_get(v___y_2753_, 0);
v_isSharedCheck_2763_ = !lean_is_exclusive(v___y_2753_);
if (v_isSharedCheck_2763_ == 0)
{
v___x_2758_ = v___y_2753_;
v_isShared_2759_ = v_isSharedCheck_2763_;
goto v_resetjp_2757_;
}
else
{
lean_inc(v_a_2756_);
lean_dec(v___y_2753_);
v___x_2758_ = lean_box(0);
v_isShared_2759_ = v_isSharedCheck_2763_;
goto v_resetjp_2757_;
}
v_resetjp_2757_:
{
lean_object* v___x_2761_; 
if (v_isShared_2759_ == 0)
{
v___x_2761_ = v___x_2758_;
goto v_reusejp_2760_;
}
else
{
lean_object* v_reuseFailAlloc_2762_; 
v_reuseFailAlloc_2762_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2762_, 0, v_a_2756_);
v___x_2761_ = v_reuseFailAlloc_2762_;
goto v_reusejp_2760_;
}
v_reusejp_2760_:
{
return v___x_2761_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___redArg___boxed(lean_object* v_inst_2776_, lean_object* v_parentRef_2777_, lean_object* v_matchResult_2778_, lean_object* v_a_2779_, lean_object* v_a_2780_, lean_object* v_a_2781_, lean_object* v_a_2782_, lean_object* v_a_2783_, lean_object* v_a_2784_, lean_object* v_a_2785_, lean_object* v_a_2786_, lean_object* v_a_2787_){
_start:
{
lean_object* v_res_2788_; 
v_res_2788_ = lp_aesop_Aesop_runUnsafeRule___redArg(v_inst_2776_, v_parentRef_2777_, v_matchResult_2778_, v_a_2779_, v_a_2780_, v_a_2781_, v_a_2782_, v_a_2783_, v_a_2784_, v_a_2785_, v_a_2786_);
lean_dec(v_a_2786_);
lean_dec_ref(v_a_2785_);
lean_dec(v_a_2784_);
lean_dec_ref(v_a_2783_);
lean_dec(v_a_2782_);
lean_dec(v_a_2781_);
lean_dec(v_a_2780_);
lean_dec_ref(v_a_2779_);
return v_res_2788_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule(lean_object* v_Q_2789_, lean_object* v_inst_2790_, lean_object* v_parentRef_2791_, lean_object* v_matchResult_2792_, lean_object* v_a_2793_, lean_object* v_a_2794_, lean_object* v_a_2795_, lean_object* v_a_2796_, lean_object* v_a_2797_, lean_object* v_a_2798_, lean_object* v_a_2799_, lean_object* v_a_2800_){
_start:
{
lean_object* v___x_2802_; 
v___x_2802_ = lp_aesop_Aesop_runUnsafeRule___redArg(v_inst_2790_, v_parentRef_2791_, v_matchResult_2792_, v_a_2793_, v_a_2794_, v_a_2795_, v_a_2796_, v_a_2797_, v_a_2798_, v_a_2799_, v_a_2800_);
return v___x_2802_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runUnsafeRule___boxed(lean_object* v_Q_2803_, lean_object* v_inst_2804_, lean_object* v_parentRef_2805_, lean_object* v_matchResult_2806_, lean_object* v_a_2807_, lean_object* v_a_2808_, lean_object* v_a_2809_, lean_object* v_a_2810_, lean_object* v_a_2811_, lean_object* v_a_2812_, lean_object* v_a_2813_, lean_object* v_a_2814_, lean_object* v_a_2815_){
_start:
{
lean_object* v_res_2816_; 
v_res_2816_ = lp_aesop_Aesop_runUnsafeRule(v_Q_2803_, v_inst_2804_, v_parentRef_2805_, v_matchResult_2806_, v_a_2807_, v_a_2808_, v_a_2809_, v_a_2810_, v_a_2811_, v_a_2812_, v_a_2813_, v_a_2814_);
lean_dec(v_a_2814_);
lean_dec_ref(v_a_2813_);
lean_dec(v_a_2812_);
lean_dec_ref(v_a_2811_);
lean_dec(v_a_2810_);
lean_dec(v_a_2809_);
lean_dec(v_a_2808_);
lean_dec_ref(v_a_2807_);
return v_res_2816_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorIdx(lean_object* v_x_2817_){
_start:
{
switch(lean_obj_tag(v_x_2817_))
{
case 0:
{
lean_object* v___x_2818_; 
v___x_2818_ = lean_unsigned_to_nat(0u);
return v___x_2818_;
}
case 1:
{
lean_object* v___x_2819_; 
v___x_2819_ = lean_unsigned_to_nat(1u);
return v___x_2819_;
}
case 2:
{
lean_object* v___x_2820_; 
v___x_2820_ = lean_unsigned_to_nat(2u);
return v___x_2820_;
}
default: 
{
lean_object* v___x_2821_; 
v___x_2821_ = lean_unsigned_to_nat(3u);
return v___x_2821_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorIdx___boxed(lean_object* v_x_2822_){
_start:
{
lean_object* v_res_2823_; 
v_res_2823_ = lp_aesop_Aesop_SafeRulesResult_ctorIdx(v_x_2822_);
lean_dec(v_x_2822_);
return v_res_2823_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(lean_object* v_t_2824_, lean_object* v_k_2825_){
_start:
{
if (lean_obj_tag(v_t_2824_) == 3)
{
return v_k_2825_;
}
else
{
lean_object* v_newRapps_2826_; lean_object* v___x_2827_; 
v_newRapps_2826_ = lean_ctor_get(v_t_2824_, 0);
lean_inc_ref(v_newRapps_2826_);
lean_dec(v_t_2824_);
v___x_2827_ = lean_apply_1(v_k_2825_, v_newRapps_2826_);
return v___x_2827_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorElim(lean_object* v_motive_2828_, lean_object* v_ctorIdx_2829_, lean_object* v_t_2830_, lean_object* v_h_2831_, lean_object* v_k_2832_){
_start:
{
lean_object* v___x_2833_; 
v___x_2833_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2830_, v_k_2832_);
return v___x_2833_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_ctorElim___boxed(lean_object* v_motive_2834_, lean_object* v_ctorIdx_2835_, lean_object* v_t_2836_, lean_object* v_h_2837_, lean_object* v_k_2838_){
_start:
{
lean_object* v_res_2839_; 
v_res_2839_ = lp_aesop_Aesop_SafeRulesResult_ctorElim(v_motive_2834_, v_ctorIdx_2835_, v_t_2836_, v_h_2837_, v_k_2838_);
lean_dec(v_ctorIdx_2835_);
return v_res_2839_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_proved_elim___redArg(lean_object* v_t_2840_, lean_object* v_proved_2841_){
_start:
{
lean_object* v___x_2842_; 
v___x_2842_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2840_, v_proved_2841_);
return v___x_2842_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_proved_elim(lean_object* v_motive_2843_, lean_object* v_t_2844_, lean_object* v_h_2845_, lean_object* v_proved_2846_){
_start:
{
lean_object* v___x_2847_; 
v___x_2847_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2844_, v_proved_2846_);
return v___x_2847_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_succeeded_elim___redArg(lean_object* v_t_2848_, lean_object* v_succeeded_2849_){
_start:
{
lean_object* v___x_2850_; 
v___x_2850_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2848_, v_succeeded_2849_);
return v___x_2850_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_succeeded_elim(lean_object* v_motive_2851_, lean_object* v_t_2852_, lean_object* v_h_2853_, lean_object* v_succeeded_2854_){
_start:
{
lean_object* v___x_2855_; 
v___x_2855_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2852_, v_succeeded_2854_);
return v___x_2855_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_failed_elim___redArg(lean_object* v_t_2856_, lean_object* v_failed_2857_){
_start:
{
lean_object* v___x_2858_; 
v___x_2858_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2856_, v_failed_2857_);
return v___x_2858_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_failed_elim(lean_object* v_motive_2859_, lean_object* v_t_2860_, lean_object* v_h_2861_, lean_object* v_failed_2862_){
_start:
{
lean_object* v___x_2863_; 
v___x_2863_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2860_, v_failed_2862_);
return v___x_2863_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_skipped_elim___redArg(lean_object* v_t_2864_, lean_object* v_skipped_2865_){
_start:
{
lean_object* v___x_2866_; 
v___x_2866_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2864_, v_skipped_2865_);
return v___x_2866_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_skipped_elim(lean_object* v_motive_2867_, lean_object* v_t_2868_, lean_object* v_h_2869_, lean_object* v_skipped_2870_){
_start:
{
lean_object* v___x_2871_; 
v___x_2871_ = lp_aesop_Aesop_SafeRulesResult_ctorElim___redArg(v_t_2868_, v_skipped_2870_);
return v___x_2871_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_toEmoji(lean_object* v_x_2872_){
_start:
{
switch(lean_obj_tag(v_x_2872_))
{
case 0:
{
lean_object* v___x_2873_; 
v___x_2873_ = lp_aesop_Aesop_ruleProvedEmoji;
return v___x_2873_;
}
case 1:
{
lean_object* v___x_2874_; 
v___x_2874_ = lp_aesop_Aesop_ruleSuccessEmoji;
return v___x_2874_;
}
case 2:
{
lean_object* v___x_2875_; 
v___x_2875_ = lp_aesop_Aesop_ruleFailureEmoji;
return v___x_2875_;
}
default: 
{
lean_object* v___x_2876_; 
v___x_2876_ = lp_aesop_Aesop_ruleSkippedEmoji;
return v___x_2876_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_SafeRulesResult_toEmoji___boxed(lean_object* v_x_2877_){
_start:
{
lean_object* v_res_2878_; 
v_res_2878_ = lp_aesop_Aesop_SafeRulesResult_toEmoji(v_x_2877_);
lean_dec(v_x_2877_);
return v_res_2878_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg___lam__0(lean_object* v_inst_2879_, lean_object* v_gref_2880_, lean_object* v___x_2881_, lean_object* v_a_2882_, lean_object* v_x_2883_, lean_object* v___y_2884_, lean_object* v___y_2885_, lean_object* v___y_2886_, lean_object* v___y_2887_, lean_object* v___y_2888_, lean_object* v___y_2889_, lean_object* v___y_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_){
_start:
{
lean_object* v___x_2894_; 
v___x_2894_ = lp_aesop_Aesop_runSafeRule___redArg(v_inst_2879_, v_gref_2880_, v_a_2882_, v___y_2885_, v___y_2886_, v___y_2887_, v___y_2888_, v___y_2889_, v___y_2890_, v___y_2891_, v___y_2892_);
if (lean_obj_tag(v___x_2894_) == 0)
{
lean_object* v_a_2895_; lean_object* v___x_2897_; uint8_t v_isShared_2898_; uint8_t v_isSharedCheck_2988_; 
v_a_2895_ = lean_ctor_get(v___x_2894_, 0);
v_isSharedCheck_2988_ = !lean_is_exclusive(v___x_2894_);
if (v_isSharedCheck_2988_ == 0)
{
v___x_2897_ = v___x_2894_;
v_isShared_2898_ = v_isSharedCheck_2988_;
goto v_resetjp_2896_;
}
else
{
lean_inc(v_a_2895_);
lean_dec(v___x_2894_);
v___x_2897_ = lean_box(0);
v_isShared_2898_ = v_isSharedCheck_2988_;
goto v_resetjp_2896_;
}
v_resetjp_2896_:
{
if (lean_obj_tag(v_a_2895_) == 0)
{
lean_object* v_result_2899_; lean_object* v___x_2901_; uint8_t v_isShared_2902_; uint8_t v_isSharedCheck_2966_; 
v_result_2899_ = lean_ctor_get(v_a_2895_, 0);
v_isSharedCheck_2966_ = !lean_is_exclusive(v_a_2895_);
if (v_isSharedCheck_2966_ == 0)
{
v___x_2901_ = v_a_2895_;
v_isShared_2902_ = v_isSharedCheck_2966_;
goto v_resetjp_2900_;
}
else
{
lean_inc(v_result_2899_);
lean_dec(v_a_2895_);
v___x_2901_ = lean_box(0);
v_isShared_2902_ = v_isSharedCheck_2966_;
goto v_resetjp_2900_;
}
v_resetjp_2900_:
{
switch(lean_obj_tag(v_result_2899_))
{
case 0:
{
lean_object* v_snd_2903_; lean_object* v___x_2905_; uint8_t v_isShared_2906_; uint8_t v_isSharedCheck_2925_; 
lean_dec(v___x_2881_);
v_snd_2903_ = lean_ctor_get(v___y_2884_, 1);
v_isSharedCheck_2925_ = !lean_is_exclusive(v___y_2884_);
if (v_isSharedCheck_2925_ == 0)
{
lean_object* v_unused_2926_; 
v_unused_2926_ = lean_ctor_get(v___y_2884_, 0);
lean_dec(v_unused_2926_);
v___x_2905_ = v___y_2884_;
v_isShared_2906_ = v_isSharedCheck_2925_;
goto v_resetjp_2904_;
}
else
{
lean_inc(v_snd_2903_);
lean_dec(v___y_2884_);
v___x_2905_ = lean_box(0);
v_isShared_2906_ = v_isSharedCheck_2925_;
goto v_resetjp_2904_;
}
v_resetjp_2904_:
{
lean_object* v_newRapps_2907_; lean_object* v___x_2909_; uint8_t v_isShared_2910_; uint8_t v_isSharedCheck_2924_; 
v_newRapps_2907_ = lean_ctor_get(v_result_2899_, 0);
v_isSharedCheck_2924_ = !lean_is_exclusive(v_result_2899_);
if (v_isSharedCheck_2924_ == 0)
{
v___x_2909_ = v_result_2899_;
v_isShared_2910_ = v_isSharedCheck_2924_;
goto v_resetjp_2908_;
}
else
{
lean_inc(v_newRapps_2907_);
lean_dec(v_result_2899_);
v___x_2909_ = lean_box(0);
v_isShared_2910_ = v_isSharedCheck_2924_;
goto v_resetjp_2908_;
}
v_resetjp_2908_:
{
lean_object* v___x_2912_; 
if (v_isShared_2910_ == 0)
{
v___x_2912_ = v___x_2909_;
goto v_reusejp_2911_;
}
else
{
lean_object* v_reuseFailAlloc_2923_; 
v_reuseFailAlloc_2923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2923_, 0, v_newRapps_2907_);
v___x_2912_ = v_reuseFailAlloc_2923_;
goto v_reusejp_2911_;
}
v_reusejp_2911_:
{
lean_object* v___x_2914_; 
if (v_isShared_2902_ == 0)
{
lean_ctor_set_tag(v___x_2901_, 1);
lean_ctor_set(v___x_2901_, 0, v___x_2912_);
v___x_2914_ = v___x_2901_;
goto v_reusejp_2913_;
}
else
{
lean_object* v_reuseFailAlloc_2922_; 
v_reuseFailAlloc_2922_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2922_, 0, v___x_2912_);
v___x_2914_ = v_reuseFailAlloc_2922_;
goto v_reusejp_2913_;
}
v_reusejp_2913_:
{
lean_object* v___x_2916_; 
if (v_isShared_2906_ == 0)
{
lean_ctor_set(v___x_2905_, 0, v___x_2914_);
v___x_2916_ = v___x_2905_;
goto v_reusejp_2915_;
}
else
{
lean_object* v_reuseFailAlloc_2921_; 
v_reuseFailAlloc_2921_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2921_, 0, v___x_2914_);
lean_ctor_set(v_reuseFailAlloc_2921_, 1, v_snd_2903_);
v___x_2916_ = v_reuseFailAlloc_2921_;
goto v_reusejp_2915_;
}
v_reusejp_2915_:
{
lean_object* v___x_2917_; lean_object* v___x_2919_; 
v___x_2917_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2917_, 0, v___x_2916_);
if (v_isShared_2898_ == 0)
{
lean_ctor_set(v___x_2897_, 0, v___x_2917_);
v___x_2919_ = v___x_2897_;
goto v_reusejp_2918_;
}
else
{
lean_object* v_reuseFailAlloc_2920_; 
v_reuseFailAlloc_2920_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2920_, 0, v___x_2917_);
v___x_2919_ = v_reuseFailAlloc_2920_;
goto v_reusejp_2918_;
}
v_reusejp_2918_:
{
return v___x_2919_;
}
}
}
}
}
}
}
case 1:
{
lean_object* v_snd_2927_; lean_object* v___x_2929_; uint8_t v_isShared_2930_; uint8_t v_isSharedCheck_2949_; 
lean_dec(v___x_2881_);
v_snd_2927_ = lean_ctor_get(v___y_2884_, 1);
v_isSharedCheck_2949_ = !lean_is_exclusive(v___y_2884_);
if (v_isSharedCheck_2949_ == 0)
{
lean_object* v_unused_2950_; 
v_unused_2950_ = lean_ctor_get(v___y_2884_, 0);
lean_dec(v_unused_2950_);
v___x_2929_ = v___y_2884_;
v_isShared_2930_ = v_isSharedCheck_2949_;
goto v_resetjp_2928_;
}
else
{
lean_inc(v_snd_2927_);
lean_dec(v___y_2884_);
v___x_2929_ = lean_box(0);
v_isShared_2930_ = v_isSharedCheck_2949_;
goto v_resetjp_2928_;
}
v_resetjp_2928_:
{
lean_object* v_newRapps_2931_; lean_object* v___x_2933_; uint8_t v_isShared_2934_; uint8_t v_isSharedCheck_2948_; 
v_newRapps_2931_ = lean_ctor_get(v_result_2899_, 0);
v_isSharedCheck_2948_ = !lean_is_exclusive(v_result_2899_);
if (v_isSharedCheck_2948_ == 0)
{
v___x_2933_ = v_result_2899_;
v_isShared_2934_ = v_isSharedCheck_2948_;
goto v_resetjp_2932_;
}
else
{
lean_inc(v_newRapps_2931_);
lean_dec(v_result_2899_);
v___x_2933_ = lean_box(0);
v_isShared_2934_ = v_isSharedCheck_2948_;
goto v_resetjp_2932_;
}
v_resetjp_2932_:
{
lean_object* v___x_2936_; 
if (v_isShared_2934_ == 0)
{
v___x_2936_ = v___x_2933_;
goto v_reusejp_2935_;
}
else
{
lean_object* v_reuseFailAlloc_2947_; 
v_reuseFailAlloc_2947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2947_, 0, v_newRapps_2931_);
v___x_2936_ = v_reuseFailAlloc_2947_;
goto v_reusejp_2935_;
}
v_reusejp_2935_:
{
lean_object* v___x_2938_; 
if (v_isShared_2902_ == 0)
{
lean_ctor_set_tag(v___x_2901_, 1);
lean_ctor_set(v___x_2901_, 0, v___x_2936_);
v___x_2938_ = v___x_2901_;
goto v_reusejp_2937_;
}
else
{
lean_object* v_reuseFailAlloc_2946_; 
v_reuseFailAlloc_2946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2946_, 0, v___x_2936_);
v___x_2938_ = v_reuseFailAlloc_2946_;
goto v_reusejp_2937_;
}
v_reusejp_2937_:
{
lean_object* v___x_2940_; 
if (v_isShared_2930_ == 0)
{
lean_ctor_set(v___x_2929_, 0, v___x_2938_);
v___x_2940_ = v___x_2929_;
goto v_reusejp_2939_;
}
else
{
lean_object* v_reuseFailAlloc_2945_; 
v_reuseFailAlloc_2945_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2945_, 0, v___x_2938_);
lean_ctor_set(v_reuseFailAlloc_2945_, 1, v_snd_2927_);
v___x_2940_ = v_reuseFailAlloc_2945_;
goto v_reusejp_2939_;
}
v_reusejp_2939_:
{
lean_object* v___x_2941_; lean_object* v___x_2943_; 
v___x_2941_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2941_, 0, v___x_2940_);
if (v_isShared_2898_ == 0)
{
lean_ctor_set(v___x_2897_, 0, v___x_2941_);
v___x_2943_ = v___x_2897_;
goto v_reusejp_2942_;
}
else
{
lean_object* v_reuseFailAlloc_2944_; 
v_reuseFailAlloc_2944_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2944_, 0, v___x_2941_);
v___x_2943_ = v_reuseFailAlloc_2944_;
goto v_reusejp_2942_;
}
v_reusejp_2942_:
{
return v___x_2943_;
}
}
}
}
}
}
}
default: 
{
lean_object* v_snd_2951_; lean_object* v___x_2953_; uint8_t v_isShared_2954_; uint8_t v_isSharedCheck_2964_; 
v_snd_2951_ = lean_ctor_get(v___y_2884_, 1);
v_isSharedCheck_2964_ = !lean_is_exclusive(v___y_2884_);
if (v_isSharedCheck_2964_ == 0)
{
lean_object* v_unused_2965_; 
v_unused_2965_ = lean_ctor_get(v___y_2884_, 0);
lean_dec(v_unused_2965_);
v___x_2953_ = v___y_2884_;
v_isShared_2954_ = v_isSharedCheck_2964_;
goto v_resetjp_2952_;
}
else
{
lean_inc(v_snd_2951_);
lean_dec(v___y_2884_);
v___x_2953_ = lean_box(0);
v_isShared_2954_ = v_isSharedCheck_2964_;
goto v_resetjp_2952_;
}
v_resetjp_2952_:
{
lean_object* v___x_2956_; 
if (v_isShared_2954_ == 0)
{
lean_ctor_set(v___x_2953_, 0, v___x_2881_);
v___x_2956_ = v___x_2953_;
goto v_reusejp_2955_;
}
else
{
lean_object* v_reuseFailAlloc_2963_; 
v_reuseFailAlloc_2963_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2963_, 0, v___x_2881_);
lean_ctor_set(v_reuseFailAlloc_2963_, 1, v_snd_2951_);
v___x_2956_ = v_reuseFailAlloc_2963_;
goto v_reusejp_2955_;
}
v_reusejp_2955_:
{
lean_object* v___x_2958_; 
if (v_isShared_2902_ == 0)
{
lean_ctor_set_tag(v___x_2901_, 1);
lean_ctor_set(v___x_2901_, 0, v___x_2956_);
v___x_2958_ = v___x_2901_;
goto v_reusejp_2957_;
}
else
{
lean_object* v_reuseFailAlloc_2962_; 
v_reuseFailAlloc_2962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2962_, 0, v___x_2956_);
v___x_2958_ = v_reuseFailAlloc_2962_;
goto v_reusejp_2957_;
}
v_reusejp_2957_:
{
lean_object* v___x_2960_; 
if (v_isShared_2898_ == 0)
{
lean_ctor_set(v___x_2897_, 0, v___x_2958_);
v___x_2960_ = v___x_2897_;
goto v_reusejp_2959_;
}
else
{
lean_object* v_reuseFailAlloc_2961_; 
v_reuseFailAlloc_2961_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2961_, 0, v___x_2958_);
v___x_2960_ = v_reuseFailAlloc_2961_;
goto v_reusejp_2959_;
}
v_reusejp_2959_:
{
return v___x_2960_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_snd_2967_; lean_object* v___x_2969_; uint8_t v_isShared_2970_; uint8_t v_isSharedCheck_2986_; 
v_snd_2967_ = lean_ctor_get(v___y_2884_, 1);
v_isSharedCheck_2986_ = !lean_is_exclusive(v___y_2884_);
if (v_isSharedCheck_2986_ == 0)
{
lean_object* v_unused_2987_; 
v_unused_2987_ = lean_ctor_get(v___y_2884_, 0);
lean_dec(v_unused_2987_);
v___x_2969_ = v___y_2884_;
v_isShared_2970_ = v_isSharedCheck_2986_;
goto v_resetjp_2968_;
}
else
{
lean_inc(v_snd_2967_);
lean_dec(v___y_2884_);
v___x_2969_ = lean_box(0);
v_isShared_2970_ = v_isSharedCheck_2986_;
goto v_resetjp_2968_;
}
v_resetjp_2968_:
{
lean_object* v_result_2971_; lean_object* v___x_2973_; uint8_t v_isShared_2974_; uint8_t v_isSharedCheck_2985_; 
v_result_2971_ = lean_ctor_get(v_a_2895_, 0);
v_isSharedCheck_2985_ = !lean_is_exclusive(v_a_2895_);
if (v_isSharedCheck_2985_ == 0)
{
v___x_2973_ = v_a_2895_;
v_isShared_2974_ = v_isSharedCheck_2985_;
goto v_resetjp_2972_;
}
else
{
lean_inc(v_result_2971_);
lean_dec(v_a_2895_);
v___x_2973_ = lean_box(0);
v_isShared_2974_ = v_isSharedCheck_2985_;
goto v_resetjp_2972_;
}
v_resetjp_2972_:
{
lean_object* v___x_2975_; lean_object* v___x_2977_; 
v___x_2975_ = lean_array_push(v_snd_2967_, v_result_2971_);
if (v_isShared_2970_ == 0)
{
lean_ctor_set(v___x_2969_, 1, v___x_2975_);
lean_ctor_set(v___x_2969_, 0, v___x_2881_);
v___x_2977_ = v___x_2969_;
goto v_reusejp_2976_;
}
else
{
lean_object* v_reuseFailAlloc_2984_; 
v_reuseFailAlloc_2984_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2984_, 0, v___x_2881_);
lean_ctor_set(v_reuseFailAlloc_2984_, 1, v___x_2975_);
v___x_2977_ = v_reuseFailAlloc_2984_;
goto v_reusejp_2976_;
}
v_reusejp_2976_:
{
lean_object* v___x_2979_; 
if (v_isShared_2974_ == 0)
{
lean_ctor_set(v___x_2973_, 0, v___x_2977_);
v___x_2979_ = v___x_2973_;
goto v_reusejp_2978_;
}
else
{
lean_object* v_reuseFailAlloc_2983_; 
v_reuseFailAlloc_2983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2983_, 0, v___x_2977_);
v___x_2979_ = v_reuseFailAlloc_2983_;
goto v_reusejp_2978_;
}
v_reusejp_2978_:
{
lean_object* v___x_2981_; 
if (v_isShared_2898_ == 0)
{
lean_ctor_set(v___x_2897_, 0, v___x_2979_);
v___x_2981_ = v___x_2897_;
goto v_reusejp_2980_;
}
else
{
lean_object* v_reuseFailAlloc_2982_; 
v_reuseFailAlloc_2982_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2982_, 0, v___x_2979_);
v___x_2981_ = v_reuseFailAlloc_2982_;
goto v_reusejp_2980_;
}
v_reusejp_2980_:
{
return v___x_2981_;
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_2989_; lean_object* v___x_2991_; uint8_t v_isShared_2992_; uint8_t v_isSharedCheck_2996_; 
lean_dec_ref(v___y_2884_);
lean_dec(v___x_2881_);
v_a_2989_ = lean_ctor_get(v___x_2894_, 0);
v_isSharedCheck_2996_ = !lean_is_exclusive(v___x_2894_);
if (v_isSharedCheck_2996_ == 0)
{
v___x_2991_ = v___x_2894_;
v_isShared_2992_ = v_isSharedCheck_2996_;
goto v_resetjp_2990_;
}
else
{
lean_inc(v_a_2989_);
lean_dec(v___x_2894_);
v___x_2991_ = lean_box(0);
v_isShared_2992_ = v_isSharedCheck_2996_;
goto v_resetjp_2990_;
}
v_resetjp_2990_:
{
lean_object* v___x_2994_; 
if (v_isShared_2992_ == 0)
{
v___x_2994_ = v___x_2991_;
goto v_reusejp_2993_;
}
else
{
lean_object* v_reuseFailAlloc_2995_; 
v_reuseFailAlloc_2995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2995_, 0, v_a_2989_);
v___x_2994_ = v_reuseFailAlloc_2995_;
goto v_reusejp_2993_;
}
v_reusejp_2993_:
{
return v___x_2994_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg___lam__0___boxed(lean_object* v_inst_2997_, lean_object* v_gref_2998_, lean_object* v___x_2999_, lean_object* v_a_3000_, lean_object* v_x_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_){
_start:
{
lean_object* v_res_3012_; 
v_res_3012_ = lp_aesop_Aesop_runFirstSafeRule___redArg___lam__0(v_inst_2997_, v_gref_2998_, v___x_2999_, v_a_3000_, v_x_3001_, v___y_3002_, v___y_3003_, v___y_3004_, v___y_3005_, v___y_3006_, v___y_3007_, v___y_3008_, v___y_3009_, v___y_3010_);
lean_dec(v___y_3010_);
lean_dec_ref(v___y_3009_);
lean_dec(v___y_3008_);
lean_dec_ref(v___y_3007_);
lean_dec(v___y_3006_);
lean_dec(v___y_3005_);
lean_dec(v___y_3004_);
lean_dec_ref(v___y_3003_);
return v_res_3012_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg(lean_object* v_inst_3018_, lean_object* v_gref_3019_, lean_object* v_a_3020_, lean_object* v_a_3021_, lean_object* v_a_3022_, lean_object* v_a_3023_, lean_object* v_a_3024_, lean_object* v_a_3025_, lean_object* v_a_3026_, lean_object* v_a_3027_){
_start:
{
lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; lean_object* v_elimGoal_3033_; lean_object* v___x_3034_; uint8_t v_unsafeRulesSelected_3035_; 
v___x_3029_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_3018_);
v___x_3030_ = lean_st_ref_get(v_a_3021_);
lean_dec(v___x_3030_);
v___x_3031_ = lean_st_ref_get(v_gref_3019_);
v___x_3032_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_3033_ = lean_ctor_get(v___x_3032_, 1);
lean_inc_ref(v_elimGoal_3033_);
v___x_3034_ = lean_apply_1(v_elimGoal_3033_, v___x_3031_);
v_unsafeRulesSelected_3035_ = lean_ctor_get_uint8(v___x_3034_, sizeof(void*)*14 + 11);
lean_dec_ref(v___x_3034_);
if (v_unsafeRulesSelected_3035_ == 0)
{
lean_object* v___x_3036_; lean_object* v_iteration_3037_; lean_object* v_ruleSet_3038_; uint8_t v___x_3039_; lean_object* v___x_3040_; lean_object* v___x_3041_; 
v___x_3036_ = lean_st_ref_get(v_a_3021_);
v_iteration_3037_ = lean_ctor_get(v___x_3036_, 0);
lean_inc(v_iteration_3037_);
lean_dec(v___x_3036_);
v_ruleSet_3038_ = lean_ctor_get(v_a_3020_, 0);
v___x_3039_ = 1;
lean_inc_ref(v_ruleSet_3038_);
v___x_3040_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3040_, 0, v_iteration_3037_);
lean_ctor_set(v___x_3040_, 1, v_ruleSet_3038_);
v___x_3041_ = lp_aesop_Aesop_GoalRef_updateForwardState(v___x_3039_, v_gref_3019_, v___x_3040_, v_a_3022_, v_a_3023_, v_a_3024_, v_a_3025_, v_a_3026_, v_a_3027_);
lean_dec_ref_known(v___x_3040_, 2);
if (lean_obj_tag(v___x_3041_) == 0)
{
lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; 
lean_dec_ref_known(v___x_3041_, 1);
v___x_3042_ = lean_st_ref_get(v_a_3021_);
lean_dec(v___x_3042_);
v___x_3043_ = lean_st_ref_get(v_gref_3019_);
v___x_3044_ = lp_aesop_Aesop_selectSafeRules___redArg(v_inst_3018_, v___x_3043_, v_a_3020_, v_a_3021_, v_a_3022_, v_a_3023_, v_a_3024_, v_a_3025_, v_a_3026_, v_a_3027_);
if (lean_obj_tag(v___x_3044_) == 0)
{
lean_object* v_a_3045_; lean_object* v___x_3046_; lean_object* v___f_3047_; lean_object* v___x_3048_; size_t v_sz_3049_; size_t v___x_3050_; lean_object* v___x_9043__overap_3051_; lean_object* v___x_3052_; 
v_a_3045_ = lean_ctor_get(v___x_3044_, 0);
lean_inc(v_a_3045_);
lean_dec_ref_known(v___x_3044_, 1);
v___x_3046_ = lean_box(0);
v___f_3047_ = lean_alloc_closure((void*)(lp_aesop_Aesop_runFirstSafeRule___redArg___lam__0___boxed), 15, 3);
lean_closure_set(v___f_3047_, 0, v_inst_3018_);
lean_closure_set(v___f_3047_, 1, v_gref_3019_);
lean_closure_set(v___f_3047_, 2, v___x_3046_);
v___x_3048_ = ((lean_object*)(lp_aesop_Aesop_runFirstSafeRule___redArg___closed__1));
v_sz_3049_ = lean_array_size(v_a_3045_);
v___x_3050_ = ((size_t)0ULL);
v___x_9043__overap_3051_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_3029_, v_a_3045_, v___f_3047_, v_sz_3049_, v___x_3050_, v___x_3048_);
lean_inc(v_a_3027_);
lean_inc_ref(v_a_3026_);
lean_inc(v_a_3025_);
lean_inc_ref(v_a_3024_);
lean_inc(v_a_3023_);
lean_inc(v_a_3022_);
lean_inc(v_a_3021_);
lean_inc_ref(v_a_3020_);
v___x_3052_ = lean_apply_9(v___x_9043__overap_3051_, v_a_3020_, v_a_3021_, v_a_3022_, v_a_3023_, v_a_3024_, v_a_3025_, v_a_3026_, v_a_3027_, lean_box(0));
if (lean_obj_tag(v___x_3052_) == 0)
{
lean_object* v_a_3053_; lean_object* v___x_3055_; uint8_t v_isShared_3056_; uint8_t v_isSharedCheck_3067_; 
v_a_3053_ = lean_ctor_get(v___x_3052_, 0);
v_isSharedCheck_3067_ = !lean_is_exclusive(v___x_3052_);
if (v_isSharedCheck_3067_ == 0)
{
v___x_3055_ = v___x_3052_;
v_isShared_3056_ = v_isSharedCheck_3067_;
goto v_resetjp_3054_;
}
else
{
lean_inc(v_a_3053_);
lean_dec(v___x_3052_);
v___x_3055_ = lean_box(0);
v_isShared_3056_ = v_isSharedCheck_3067_;
goto v_resetjp_3054_;
}
v_resetjp_3054_:
{
lean_object* v_fst_3057_; 
v_fst_3057_ = lean_ctor_get(v_a_3053_, 0);
if (lean_obj_tag(v_fst_3057_) == 0)
{
lean_object* v_snd_3058_; lean_object* v___x_3059_; lean_object* v___x_3061_; 
v_snd_3058_ = lean_ctor_get(v_a_3053_, 1);
lean_inc(v_snd_3058_);
lean_dec(v_a_3053_);
v___x_3059_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_3059_, 0, v_snd_3058_);
if (v_isShared_3056_ == 0)
{
lean_ctor_set(v___x_3055_, 0, v___x_3059_);
v___x_3061_ = v___x_3055_;
goto v_reusejp_3060_;
}
else
{
lean_object* v_reuseFailAlloc_3062_; 
v_reuseFailAlloc_3062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3062_, 0, v___x_3059_);
v___x_3061_ = v_reuseFailAlloc_3062_;
goto v_reusejp_3060_;
}
v_reusejp_3060_:
{
return v___x_3061_;
}
}
else
{
lean_object* v_val_3063_; lean_object* v___x_3065_; 
lean_inc_ref(v_fst_3057_);
lean_dec(v_a_3053_);
v_val_3063_ = lean_ctor_get(v_fst_3057_, 0);
lean_inc(v_val_3063_);
lean_dec_ref_known(v_fst_3057_, 1);
if (v_isShared_3056_ == 0)
{
lean_ctor_set(v___x_3055_, 0, v_val_3063_);
v___x_3065_ = v___x_3055_;
goto v_reusejp_3064_;
}
else
{
lean_object* v_reuseFailAlloc_3066_; 
v_reuseFailAlloc_3066_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3066_, 0, v_val_3063_);
v___x_3065_ = v_reuseFailAlloc_3066_;
goto v_reusejp_3064_;
}
v_reusejp_3064_:
{
return v___x_3065_;
}
}
}
}
else
{
lean_object* v_a_3068_; lean_object* v___x_3070_; uint8_t v_isShared_3071_; uint8_t v_isSharedCheck_3075_; 
v_a_3068_ = lean_ctor_get(v___x_3052_, 0);
v_isSharedCheck_3075_ = !lean_is_exclusive(v___x_3052_);
if (v_isSharedCheck_3075_ == 0)
{
v___x_3070_ = v___x_3052_;
v_isShared_3071_ = v_isSharedCheck_3075_;
goto v_resetjp_3069_;
}
else
{
lean_inc(v_a_3068_);
lean_dec(v___x_3052_);
v___x_3070_ = lean_box(0);
v_isShared_3071_ = v_isSharedCheck_3075_;
goto v_resetjp_3069_;
}
v_resetjp_3069_:
{
lean_object* v___x_3073_; 
if (v_isShared_3071_ == 0)
{
v___x_3073_ = v___x_3070_;
goto v_reusejp_3072_;
}
else
{
lean_object* v_reuseFailAlloc_3074_; 
v_reuseFailAlloc_3074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3074_, 0, v_a_3068_);
v___x_3073_ = v_reuseFailAlloc_3074_;
goto v_reusejp_3072_;
}
v_reusejp_3072_:
{
return v___x_3073_;
}
}
}
}
else
{
lean_object* v_a_3076_; lean_object* v___x_3078_; uint8_t v_isShared_3079_; uint8_t v_isSharedCheck_3083_; 
lean_dec_ref(v___x_3029_);
lean_dec(v_gref_3019_);
lean_dec_ref(v_inst_3018_);
v_a_3076_ = lean_ctor_get(v___x_3044_, 0);
v_isSharedCheck_3083_ = !lean_is_exclusive(v___x_3044_);
if (v_isSharedCheck_3083_ == 0)
{
v___x_3078_ = v___x_3044_;
v_isShared_3079_ = v_isSharedCheck_3083_;
goto v_resetjp_3077_;
}
else
{
lean_inc(v_a_3076_);
lean_dec(v___x_3044_);
v___x_3078_ = lean_box(0);
v_isShared_3079_ = v_isSharedCheck_3083_;
goto v_resetjp_3077_;
}
v_resetjp_3077_:
{
lean_object* v___x_3081_; 
if (v_isShared_3079_ == 0)
{
v___x_3081_ = v___x_3078_;
goto v_reusejp_3080_;
}
else
{
lean_object* v_reuseFailAlloc_3082_; 
v_reuseFailAlloc_3082_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3082_, 0, v_a_3076_);
v___x_3081_ = v_reuseFailAlloc_3082_;
goto v_reusejp_3080_;
}
v_reusejp_3080_:
{
return v___x_3081_;
}
}
}
}
else
{
lean_object* v_a_3084_; lean_object* v___x_3086_; uint8_t v_isShared_3087_; uint8_t v_isSharedCheck_3091_; 
lean_dec_ref(v___x_3029_);
lean_dec(v_gref_3019_);
lean_dec_ref(v_inst_3018_);
v_a_3084_ = lean_ctor_get(v___x_3041_, 0);
v_isSharedCheck_3091_ = !lean_is_exclusive(v___x_3041_);
if (v_isSharedCheck_3091_ == 0)
{
v___x_3086_ = v___x_3041_;
v_isShared_3087_ = v_isSharedCheck_3091_;
goto v_resetjp_3085_;
}
else
{
lean_inc(v_a_3084_);
lean_dec(v___x_3041_);
v___x_3086_ = lean_box(0);
v_isShared_3087_ = v_isSharedCheck_3091_;
goto v_resetjp_3085_;
}
v_resetjp_3085_:
{
lean_object* v___x_3089_; 
if (v_isShared_3087_ == 0)
{
v___x_3089_ = v___x_3086_;
goto v_reusejp_3088_;
}
else
{
lean_object* v_reuseFailAlloc_3090_; 
v_reuseFailAlloc_3090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3090_, 0, v_a_3084_);
v___x_3089_ = v_reuseFailAlloc_3090_;
goto v_reusejp_3088_;
}
v_reusejp_3088_:
{
return v___x_3089_;
}
}
}
}
else
{
lean_object* v___x_3092_; lean_object* v___x_3093_; 
lean_dec_ref(v___x_3029_);
lean_dec(v_gref_3019_);
lean_dec_ref(v_inst_3018_);
v___x_3092_ = lean_box(3);
v___x_3093_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3093_, 0, v___x_3092_);
return v___x_3093_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg___boxed(lean_object* v_inst_3094_, lean_object* v_gref_3095_, lean_object* v_a_3096_, lean_object* v_a_3097_, lean_object* v_a_3098_, lean_object* v_a_3099_, lean_object* v_a_3100_, lean_object* v_a_3101_, lean_object* v_a_3102_, lean_object* v_a_3103_, lean_object* v_a_3104_){
_start:
{
lean_object* v_res_3105_; 
v_res_3105_ = lp_aesop_Aesop_runFirstSafeRule___redArg(v_inst_3094_, v_gref_3095_, v_a_3096_, v_a_3097_, v_a_3098_, v_a_3099_, v_a_3100_, v_a_3101_, v_a_3102_, v_a_3103_);
lean_dec(v_a_3103_);
lean_dec_ref(v_a_3102_);
lean_dec(v_a_3101_);
lean_dec_ref(v_a_3100_);
lean_dec(v_a_3099_);
lean_dec(v_a_3098_);
lean_dec(v_a_3097_);
lean_dec_ref(v_a_3096_);
return v_res_3105_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule(lean_object* v_Q_3106_, lean_object* v_inst_3107_, lean_object* v_gref_3108_, lean_object* v_a_3109_, lean_object* v_a_3110_, lean_object* v_a_3111_, lean_object* v_a_3112_, lean_object* v_a_3113_, lean_object* v_a_3114_, lean_object* v_a_3115_, lean_object* v_a_3116_){
_start:
{
lean_object* v___x_3118_; 
v___x_3118_ = lp_aesop_Aesop_runFirstSafeRule___redArg(v_inst_3107_, v_gref_3108_, v_a_3109_, v_a_3110_, v_a_3111_, v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_, v_a_3116_);
return v___x_3118_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstSafeRule___boxed(lean_object* v_Q_3119_, lean_object* v_inst_3120_, lean_object* v_gref_3121_, lean_object* v_a_3122_, lean_object* v_a_3123_, lean_object* v_a_3124_, lean_object* v_a_3125_, lean_object* v_a_3126_, lean_object* v_a_3127_, lean_object* v_a_3128_, lean_object* v_a_3129_, lean_object* v_a_3130_){
_start:
{
lean_object* v_res_3131_; 
v_res_3131_ = lp_aesop_Aesop_runFirstSafeRule(v_Q_3119_, v_inst_3120_, v_gref_3121_, v_a_3122_, v_a_3123_, v_a_3124_, v_a_3125_, v_a_3126_, v_a_3127_, v_a_3128_, v_a_3129_);
lean_dec(v_a_3129_);
lean_dec_ref(v_a_3128_);
lean_dec(v_a_3127_);
lean_dec_ref(v_a_3126_);
lean_dec(v_a_3125_);
lean_dec(v_a_3124_);
lean_dec(v_a_3123_);
lean_dec_ref(v_a_3122_);
return v_res_3131_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyPostponedSafeRule___redArg(lean_object* v_inst_3133_, lean_object* v_r_3134_, lean_object* v_parentRef_3135_, lean_object* v_a_3136_, lean_object* v_a_3137_, lean_object* v_a_3138_, lean_object* v_a_3139_, lean_object* v_a_3140_, lean_object* v_a_3141_, lean_object* v_a_3142_, lean_object* v_a_3143_){
_start:
{
lean_object* v_rule_3145_; lean_object* v_output_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; lean_object* v___x_3149_; lean_object* v___x_3150_; lean_object* v_options_3151_; lean_object* v_inheritedTraceOptions_3152_; uint8_t v_hasTrace_3153_; lean_object* v___x_3154_; lean_object* v___x_3155_; 
v_rule_3145_ = lean_ctor_get(v_r_3134_, 0);
lean_inc_ref(v_rule_3145_);
v_output_3146_ = lean_ctor_get(v_r_3134_, 1);
lean_inc_ref(v_output_3146_);
v___x_3147_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_3133_);
v___x_3148_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_3149_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_3133_);
v___x_3150_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_options_3151_ = lean_ctor_get(v_a_3142_, 2);
v_inheritedTraceOptions_3152_ = lean_ctor_get(v_a_3142_, 13);
v_hasTrace_3153_ = lean_ctor_get_uint8(v_options_3151_, sizeof(void*)*1);
v___x_3154_ = lp_aesop_Aesop_PostponedSafeRule_toUnsafeRule(v_r_3134_);
v___x_3155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3155_, 0, v___x_3154_);
if (v_hasTrace_3153_ == 0)
{
lean_object* v___x_3156_; 
lean_dec_ref(v___x_3149_);
lean_dec_ref(v___x_3147_);
lean_dec_ref(v_rule_3145_);
v___x_3156_ = lp_aesop_Aesop_addRapps___redArg(v_inst_3133_, v_parentRef_3135_, v___x_3155_, v_output_3146_, v_a_3136_, v_a_3137_, v_a_3138_, v_a_3139_, v_a_3140_, v_a_3141_, v_a_3142_, v_a_3143_);
return v___x_3156_;
}
else
{
lean_object* v___x_3157_; lean_object* v_name_3158_; lean_object* v___x_3159_; lean_object* v_traceClass_3160_; lean_object* v___f_3161_; lean_object* v___f_3162_; lean_object* v___f_3163_; lean_object* v___x_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; lean_object* v___x_3167_; lean_object* v___x_3168_; uint8_t v___x_3169_; lean_object* v___y_3171_; lean_object* v___y_3172_; lean_object* v_a_3173_; lean_object* v___y_3188_; lean_object* v___y_3189_; lean_object* v_a_3190_; 
v___x_3157_ = lean_st_ref_get(v_a_3137_);
lean_dec(v___x_3157_);
v_name_3158_ = lean_ctor_get(v_rule_3145_, 0);
lean_inc_ref(v_name_3158_);
lean_dec_ref(v_rule_3145_);
v___x_3159_ = lp_aesop_Aesop_TraceOption_steps;
v_traceClass_3160_ = lean_ctor_get(v___x_3159_, 0);
v___f_3161_ = ((lean_object*)(lp_aesop_Aesop_runUnsafeRule___redArg___closed__0));
v___f_3162_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___f_3163_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
v___x_3164_ = ((lean_object*)(lp_aesop_Aesop_applyPostponedSafeRule___redArg___closed__0));
lean_inc_ref(v_inst_3133_);
v___x_3165_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_withRuleTraceNode_fmt___boxed), 16, 6);
lean_closure_set(v___x_3165_, 0, lean_box(0));
lean_closure_set(v___x_3165_, 1, v_inst_3133_);
lean_closure_set(v___x_3165_, 2, lean_box(0));
lean_closure_set(v___x_3165_, 3, v_name_3158_);
lean_closure_set(v___x_3165_, 4, v___f_3161_);
lean_closure_set(v___x_3165_, 5, v___x_3164_);
v___x_3166_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23));
v___x_3167_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_3160_);
v___x_3168_ = l_Lean_Name_append(v___x_3167_, v_traceClass_3160_);
v___x_3169_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3152_, v_options_3151_, v___x_3168_);
lean_dec(v___x_3168_);
if (v___x_3169_ == 0)
{
lean_object* v___x_3255_; lean_object* v___x_3256_; lean_object* v___x_3257_; uint8_t v___x_3258_; 
v___x_3255_ = l_Lean_KVMap_instValueBool;
v___x_3256_ = l_Lean_trace_profiler;
v___x_3257_ = l_Lean_Option_get___redArg(v___x_3255_, v_options_3151_, v___x_3256_);
v___x_3258_ = lean_unbox(v___x_3257_);
lean_dec(v___x_3257_);
if (v___x_3258_ == 0)
{
lean_object* v___x_3259_; 
lean_dec_ref(v___x_3165_);
lean_dec_ref(v___x_3149_);
lean_dec_ref(v___x_3147_);
v___x_3259_ = lp_aesop_Aesop_addRapps___redArg(v_inst_3133_, v_parentRef_3135_, v___x_3155_, v_output_3146_, v_a_3136_, v_a_3137_, v_a_3138_, v_a_3139_, v_a_3140_, v_a_3141_, v_a_3142_, v_a_3143_);
return v___x_3259_;
}
else
{
goto v___jp_3201_;
}
}
else
{
goto v___jp_3201_;
}
v___jp_3170_:
{
lean_object* v___x_3174_; lean_object* v___x_3175_; double v___x_3176_; double v___x_3177_; double v___x_3178_; double v___x_3179_; double v___x_3180_; lean_object* v___x_3181_; lean_object* v___x_3182_; lean_object* v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_166__overap_3185_; lean_object* v___x_3186_; 
v___x_3174_ = lean_st_ref_get(v_a_3137_);
lean_dec(v___x_3174_);
v___x_3175_ = lean_io_mono_nanos_now();
v___x_3176_ = lean_float_of_nat(v___y_3172_);
v___x_3177_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_3178_ = lean_float_div(v___x_3176_, v___x_3177_);
v___x_3179_ = lean_float_of_nat(v___x_3175_);
v___x_3180_ = lean_float_div(v___x_3179_, v___x_3177_);
v___x_3181_ = lean_box_float(v___x_3178_);
v___x_3182_ = lean_box_float(v___x_3180_);
v___x_3183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3183_, 0, v___x_3181_);
lean_ctor_set(v___x_3183_, 1, v___x_3182_);
v___x_3184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3184_, 0, v_a_3173_);
lean_ctor_set(v___x_3184_, 1, v___x_3183_);
lean_inc(v_traceClass_3160_);
v___x_166__overap_3185_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3147_, v___x_3148_, v___x_3149_, v___f_3162_, lean_box(0), v___x_3150_, v___f_3163_, v_traceClass_3160_, v_hasTrace_3153_, v___x_3166_, v_options_3151_, v___x_3169_, v___y_3171_, v___x_3165_, v___x_3184_);
lean_inc(v_a_3143_);
lean_inc_ref(v_a_3142_);
lean_inc(v_a_3141_);
lean_inc_ref(v_a_3140_);
lean_inc(v_a_3139_);
lean_inc(v_a_3138_);
lean_inc(v_a_3137_);
lean_inc_ref(v_a_3136_);
v___x_3186_ = lean_apply_9(v___x_166__overap_3185_, v_a_3136_, v_a_3137_, v_a_3138_, v_a_3139_, v_a_3140_, v_a_3141_, v_a_3142_, v_a_3143_, lean_box(0));
return v___x_3186_;
}
v___jp_3187_:
{
lean_object* v___x_3191_; lean_object* v___x_3192_; double v___x_3193_; double v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v___x_3198_; lean_object* v___x_193__overap_3199_; lean_object* v___x_3200_; 
v___x_3191_ = lean_st_ref_get(v_a_3137_);
lean_dec(v___x_3191_);
v___x_3192_ = lean_io_get_num_heartbeats();
v___x_3193_ = lean_float_of_nat(v___y_3189_);
v___x_3194_ = lean_float_of_nat(v___x_3192_);
v___x_3195_ = lean_box_float(v___x_3193_);
v___x_3196_ = lean_box_float(v___x_3194_);
v___x_3197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3197_, 0, v___x_3195_);
lean_ctor_set(v___x_3197_, 1, v___x_3196_);
v___x_3198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3198_, 0, v_a_3190_);
lean_ctor_set(v___x_3198_, 1, v___x_3197_);
lean_inc(v_traceClass_3160_);
v___x_193__overap_3199_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3147_, v___x_3148_, v___x_3149_, v___f_3162_, lean_box(0), v___x_3150_, v___f_3163_, v_traceClass_3160_, v_hasTrace_3153_, v___x_3166_, v_options_3151_, v___x_3169_, v___y_3188_, v___x_3165_, v___x_3198_);
lean_inc(v_a_3143_);
lean_inc_ref(v_a_3142_);
lean_inc(v_a_3141_);
lean_inc_ref(v_a_3140_);
lean_inc(v_a_3139_);
lean_inc(v_a_3138_);
lean_inc(v_a_3137_);
lean_inc_ref(v_a_3136_);
v___x_3200_ = lean_apply_9(v___x_193__overap_3199_, v_a_3136_, v_a_3137_, v_a_3138_, v_a_3139_, v_a_3140_, v_a_3141_, v_a_3142_, v_a_3143_, lean_box(0));
return v___x_3200_;
}
v___jp_3201_:
{
lean_object* v___x_137__overap_3202_; lean_object* v___x_3203_; 
lean_inc_ref(v___x_3147_);
v___x_137__overap_3202_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_3147_, v___x_3148_);
lean_inc(v_a_3143_);
lean_inc_ref(v_a_3142_);
lean_inc(v_a_3141_);
lean_inc_ref(v_a_3140_);
lean_inc(v_a_3139_);
lean_inc(v_a_3138_);
lean_inc(v_a_3137_);
lean_inc_ref(v_a_3136_);
v___x_3203_ = lean_apply_9(v___x_137__overap_3202_, v_a_3136_, v_a_3137_, v_a_3138_, v_a_3139_, v_a_3140_, v_a_3141_, v_a_3142_, v_a_3143_, lean_box(0));
if (lean_obj_tag(v___x_3203_) == 0)
{
lean_object* v_a_3204_; lean_object* v___x_3205_; lean_object* v___x_3206_; lean_object* v___x_3207_; uint8_t v___x_3208_; 
v_a_3204_ = lean_ctor_get(v___x_3203_, 0);
lean_inc(v_a_3204_);
lean_dec_ref_known(v___x_3203_, 1);
v___x_3205_ = l_Lean_KVMap_instValueBool;
v___x_3206_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3207_ = l_Lean_Option_get___redArg(v___x_3205_, v_options_3151_, v___x_3206_);
v___x_3208_ = lean_unbox(v___x_3207_);
lean_dec(v___x_3207_);
if (v___x_3208_ == 0)
{
lean_object* v___x_3209_; lean_object* v___x_3210_; lean_object* v___x_3211_; 
v___x_3209_ = lean_st_ref_get(v_a_3137_);
lean_dec(v___x_3209_);
v___x_3210_ = lean_io_mono_nanos_now();
v___x_3211_ = lp_aesop_Aesop_addRapps___redArg(v_inst_3133_, v_parentRef_3135_, v___x_3155_, v_output_3146_, v_a_3136_, v_a_3137_, v_a_3138_, v_a_3139_, v_a_3140_, v_a_3141_, v_a_3142_, v_a_3143_);
if (lean_obj_tag(v___x_3211_) == 0)
{
lean_object* v_a_3212_; lean_object* v___x_3214_; uint8_t v_isShared_3215_; uint8_t v_isSharedCheck_3219_; 
v_a_3212_ = lean_ctor_get(v___x_3211_, 0);
v_isSharedCheck_3219_ = !lean_is_exclusive(v___x_3211_);
if (v_isSharedCheck_3219_ == 0)
{
v___x_3214_ = v___x_3211_;
v_isShared_3215_ = v_isSharedCheck_3219_;
goto v_resetjp_3213_;
}
else
{
lean_inc(v_a_3212_);
lean_dec(v___x_3211_);
v___x_3214_ = lean_box(0);
v_isShared_3215_ = v_isSharedCheck_3219_;
goto v_resetjp_3213_;
}
v_resetjp_3213_:
{
lean_object* v___x_3217_; 
if (v_isShared_3215_ == 0)
{
lean_ctor_set_tag(v___x_3214_, 1);
v___x_3217_ = v___x_3214_;
goto v_reusejp_3216_;
}
else
{
lean_object* v_reuseFailAlloc_3218_; 
v_reuseFailAlloc_3218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3218_, 0, v_a_3212_);
v___x_3217_ = v_reuseFailAlloc_3218_;
goto v_reusejp_3216_;
}
v_reusejp_3216_:
{
v___y_3171_ = v_a_3204_;
v___y_3172_ = v___x_3210_;
v_a_3173_ = v___x_3217_;
goto v___jp_3170_;
}
}
}
else
{
lean_object* v_a_3220_; lean_object* v___x_3222_; uint8_t v_isShared_3223_; uint8_t v_isSharedCheck_3227_; 
v_a_3220_ = lean_ctor_get(v___x_3211_, 0);
v_isSharedCheck_3227_ = !lean_is_exclusive(v___x_3211_);
if (v_isSharedCheck_3227_ == 0)
{
v___x_3222_ = v___x_3211_;
v_isShared_3223_ = v_isSharedCheck_3227_;
goto v_resetjp_3221_;
}
else
{
lean_inc(v_a_3220_);
lean_dec(v___x_3211_);
v___x_3222_ = lean_box(0);
v_isShared_3223_ = v_isSharedCheck_3227_;
goto v_resetjp_3221_;
}
v_resetjp_3221_:
{
lean_object* v___x_3225_; 
if (v_isShared_3223_ == 0)
{
lean_ctor_set_tag(v___x_3222_, 0);
v___x_3225_ = v___x_3222_;
goto v_reusejp_3224_;
}
else
{
lean_object* v_reuseFailAlloc_3226_; 
v_reuseFailAlloc_3226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3226_, 0, v_a_3220_);
v___x_3225_ = v_reuseFailAlloc_3226_;
goto v_reusejp_3224_;
}
v_reusejp_3224_:
{
v___y_3171_ = v_a_3204_;
v___y_3172_ = v___x_3210_;
v_a_3173_ = v___x_3225_;
goto v___jp_3170_;
}
}
}
}
else
{
lean_object* v___x_3228_; lean_object* v___x_3229_; lean_object* v___x_3230_; 
v___x_3228_ = lean_st_ref_get(v_a_3137_);
lean_dec(v___x_3228_);
v___x_3229_ = lean_io_get_num_heartbeats();
v___x_3230_ = lp_aesop_Aesop_addRapps___redArg(v_inst_3133_, v_parentRef_3135_, v___x_3155_, v_output_3146_, v_a_3136_, v_a_3137_, v_a_3138_, v_a_3139_, v_a_3140_, v_a_3141_, v_a_3142_, v_a_3143_);
if (lean_obj_tag(v___x_3230_) == 0)
{
lean_object* v_a_3231_; lean_object* v___x_3233_; uint8_t v_isShared_3234_; uint8_t v_isSharedCheck_3238_; 
v_a_3231_ = lean_ctor_get(v___x_3230_, 0);
v_isSharedCheck_3238_ = !lean_is_exclusive(v___x_3230_);
if (v_isSharedCheck_3238_ == 0)
{
v___x_3233_ = v___x_3230_;
v_isShared_3234_ = v_isSharedCheck_3238_;
goto v_resetjp_3232_;
}
else
{
lean_inc(v_a_3231_);
lean_dec(v___x_3230_);
v___x_3233_ = lean_box(0);
v_isShared_3234_ = v_isSharedCheck_3238_;
goto v_resetjp_3232_;
}
v_resetjp_3232_:
{
lean_object* v___x_3236_; 
if (v_isShared_3234_ == 0)
{
lean_ctor_set_tag(v___x_3233_, 1);
v___x_3236_ = v___x_3233_;
goto v_reusejp_3235_;
}
else
{
lean_object* v_reuseFailAlloc_3237_; 
v_reuseFailAlloc_3237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3237_, 0, v_a_3231_);
v___x_3236_ = v_reuseFailAlloc_3237_;
goto v_reusejp_3235_;
}
v_reusejp_3235_:
{
v___y_3188_ = v_a_3204_;
v___y_3189_ = v___x_3229_;
v_a_3190_ = v___x_3236_;
goto v___jp_3187_;
}
}
}
else
{
lean_object* v_a_3239_; lean_object* v___x_3241_; uint8_t v_isShared_3242_; uint8_t v_isSharedCheck_3246_; 
v_a_3239_ = lean_ctor_get(v___x_3230_, 0);
v_isSharedCheck_3246_ = !lean_is_exclusive(v___x_3230_);
if (v_isSharedCheck_3246_ == 0)
{
v___x_3241_ = v___x_3230_;
v_isShared_3242_ = v_isSharedCheck_3246_;
goto v_resetjp_3240_;
}
else
{
lean_inc(v_a_3239_);
lean_dec(v___x_3230_);
v___x_3241_ = lean_box(0);
v_isShared_3242_ = v_isSharedCheck_3246_;
goto v_resetjp_3240_;
}
v_resetjp_3240_:
{
lean_object* v___x_3244_; 
if (v_isShared_3242_ == 0)
{
lean_ctor_set_tag(v___x_3241_, 0);
v___x_3244_ = v___x_3241_;
goto v_reusejp_3243_;
}
else
{
lean_object* v_reuseFailAlloc_3245_; 
v_reuseFailAlloc_3245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3245_, 0, v_a_3239_);
v___x_3244_ = v_reuseFailAlloc_3245_;
goto v_reusejp_3243_;
}
v_reusejp_3243_:
{
v___y_3188_ = v_a_3204_;
v___y_3189_ = v___x_3229_;
v_a_3190_ = v___x_3244_;
goto v___jp_3187_;
}
}
}
}
}
else
{
lean_object* v_a_3247_; lean_object* v___x_3249_; uint8_t v_isShared_3250_; uint8_t v_isSharedCheck_3254_; 
lean_dec_ref(v___x_3165_);
lean_dec_ref_known(v___x_3155_, 1);
lean_dec_ref(v___x_3149_);
lean_dec_ref(v___x_3147_);
lean_dec_ref(v_output_3146_);
lean_dec(v_parentRef_3135_);
lean_dec_ref(v_inst_3133_);
v_a_3247_ = lean_ctor_get(v___x_3203_, 0);
v_isSharedCheck_3254_ = !lean_is_exclusive(v___x_3203_);
if (v_isSharedCheck_3254_ == 0)
{
v___x_3249_ = v___x_3203_;
v_isShared_3250_ = v_isSharedCheck_3254_;
goto v_resetjp_3248_;
}
else
{
lean_inc(v_a_3247_);
lean_dec(v___x_3203_);
v___x_3249_ = lean_box(0);
v_isShared_3250_ = v_isSharedCheck_3254_;
goto v_resetjp_3248_;
}
v_resetjp_3248_:
{
lean_object* v___x_3252_; 
if (v_isShared_3250_ == 0)
{
v___x_3252_ = v___x_3249_;
goto v_reusejp_3251_;
}
else
{
lean_object* v_reuseFailAlloc_3253_; 
v_reuseFailAlloc_3253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3253_, 0, v_a_3247_);
v___x_3252_ = v_reuseFailAlloc_3253_;
goto v_reusejp_3251_;
}
v_reusejp_3251_:
{
return v___x_3252_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyPostponedSafeRule___redArg___boxed(lean_object* v_inst_3260_, lean_object* v_r_3261_, lean_object* v_parentRef_3262_, lean_object* v_a_3263_, lean_object* v_a_3264_, lean_object* v_a_3265_, lean_object* v_a_3266_, lean_object* v_a_3267_, lean_object* v_a_3268_, lean_object* v_a_3269_, lean_object* v_a_3270_, lean_object* v_a_3271_){
_start:
{
lean_object* v_res_3272_; 
v_res_3272_ = lp_aesop_Aesop_applyPostponedSafeRule___redArg(v_inst_3260_, v_r_3261_, v_parentRef_3262_, v_a_3263_, v_a_3264_, v_a_3265_, v_a_3266_, v_a_3267_, v_a_3268_, v_a_3269_, v_a_3270_);
lean_dec(v_a_3270_);
lean_dec_ref(v_a_3269_);
lean_dec(v_a_3268_);
lean_dec_ref(v_a_3267_);
lean_dec(v_a_3266_);
lean_dec(v_a_3265_);
lean_dec(v_a_3264_);
lean_dec_ref(v_a_3263_);
return v_res_3272_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyPostponedSafeRule(lean_object* v_Q_3273_, lean_object* v_inst_3274_, lean_object* v_r_3275_, lean_object* v_parentRef_3276_, lean_object* v_a_3277_, lean_object* v_a_3278_, lean_object* v_a_3279_, lean_object* v_a_3280_, lean_object* v_a_3281_, lean_object* v_a_3282_, lean_object* v_a_3283_, lean_object* v_a_3284_){
_start:
{
lean_object* v___x_3286_; 
v___x_3286_ = lp_aesop_Aesop_applyPostponedSafeRule___redArg(v_inst_3274_, v_r_3275_, v_parentRef_3276_, v_a_3277_, v_a_3278_, v_a_3279_, v_a_3280_, v_a_3281_, v_a_3282_, v_a_3283_, v_a_3284_);
return v___x_3286_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyPostponedSafeRule___boxed(lean_object* v_Q_3287_, lean_object* v_inst_3288_, lean_object* v_r_3289_, lean_object* v_parentRef_3290_, lean_object* v_a_3291_, lean_object* v_a_3292_, lean_object* v_a_3293_, lean_object* v_a_3294_, lean_object* v_a_3295_, lean_object* v_a_3296_, lean_object* v_a_3297_, lean_object* v_a_3298_, lean_object* v_a_3299_){
_start:
{
lean_object* v_res_3300_; 
v_res_3300_ = lp_aesop_Aesop_applyPostponedSafeRule(v_Q_3287_, v_inst_3288_, v_r_3289_, v_parentRef_3290_, v_a_3291_, v_a_3292_, v_a_3293_, v_a_3294_, v_a_3295_, v_a_3296_, v_a_3297_, v_a_3298_);
lean_dec(v_a_3298_);
lean_dec_ref(v_a_3297_);
lean_dec(v_a_3296_);
lean_dec_ref(v_a_3295_);
lean_dec(v_a_3294_);
lean_dec(v_a_3293_);
lean_dec(v_a_3292_);
lean_dec_ref(v_a_3291_);
return v_res_3300_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___redArg(lean_object* v_inst_3301_, lean_object* v_parentRef_3302_, lean_object* v_queue_3303_, lean_object* v_a_3304_, lean_object* v_a_3305_, lean_object* v_a_3306_, lean_object* v_a_3307_, lean_object* v_a_3308_, lean_object* v_a_3309_, lean_object* v_a_3310_, lean_object* v_a_3311_){
_start:
{
lean_object* v___x_3313_; 
lean_inc_ref(v_queue_3303_);
v___x_3313_ = lp_batteries_Subarray_popHead_x3f___redArg(v_queue_3303_);
if (lean_obj_tag(v___x_3313_) == 1)
{
lean_object* v_val_3314_; lean_object* v_fst_3315_; 
lean_dec_ref(v_queue_3303_);
v_val_3314_ = lean_ctor_get(v___x_3313_, 0);
lean_inc(v_val_3314_);
lean_dec_ref_known(v___x_3313_, 1);
v_fst_3315_ = lean_ctor_get(v_val_3314_, 0);
lean_inc(v_fst_3315_);
if (lean_obj_tag(v_fst_3315_) == 0)
{
lean_object* v_snd_3316_; lean_object* v___x_3318_; uint8_t v_isShared_3319_; uint8_t v_isSharedCheck_3342_; 
v_snd_3316_ = lean_ctor_get(v_val_3314_, 1);
v_isSharedCheck_3342_ = !lean_is_exclusive(v_val_3314_);
if (v_isSharedCheck_3342_ == 0)
{
lean_object* v_unused_3343_; 
v_unused_3343_ = lean_ctor_get(v_val_3314_, 0);
lean_dec(v_unused_3343_);
v___x_3318_ = v_val_3314_;
v_isShared_3319_ = v_isSharedCheck_3342_;
goto v_resetjp_3317_;
}
else
{
lean_inc(v_snd_3316_);
lean_dec(v_val_3314_);
v___x_3318_ = lean_box(0);
v_isShared_3319_ = v_isSharedCheck_3342_;
goto v_resetjp_3317_;
}
v_resetjp_3317_:
{
lean_object* v_r_3320_; lean_object* v___x_3321_; 
v_r_3320_ = lean_ctor_get(v_fst_3315_, 0);
lean_inc_ref(v_r_3320_);
lean_dec_ref_known(v_fst_3315_, 1);
lean_inc(v_parentRef_3302_);
lean_inc_ref(v_inst_3301_);
v___x_3321_ = lp_aesop_Aesop_runUnsafeRule___redArg(v_inst_3301_, v_parentRef_3302_, v_r_3320_, v_a_3304_, v_a_3305_, v_a_3306_, v_a_3307_, v_a_3308_, v_a_3309_, v_a_3310_, v_a_3311_);
if (lean_obj_tag(v___x_3321_) == 0)
{
lean_object* v_a_3322_; lean_object* v___x_3324_; uint8_t v_isShared_3325_; uint8_t v_isSharedCheck_3333_; 
v_a_3322_ = lean_ctor_get(v___x_3321_, 0);
v_isSharedCheck_3333_ = !lean_is_exclusive(v___x_3321_);
if (v_isSharedCheck_3333_ == 0)
{
v___x_3324_ = v___x_3321_;
v_isShared_3325_ = v_isSharedCheck_3333_;
goto v_resetjp_3323_;
}
else
{
lean_inc(v_a_3322_);
lean_dec(v___x_3321_);
v___x_3324_ = lean_box(0);
v_isShared_3325_ = v_isSharedCheck_3333_;
goto v_resetjp_3323_;
}
v_resetjp_3323_:
{
if (lean_obj_tag(v_a_3322_) == 2)
{
lean_del_object(v___x_3324_);
lean_del_object(v___x_3318_);
v_queue_3303_ = v_snd_3316_;
goto _start;
}
else
{
lean_object* v___x_3328_; 
lean_dec(v_parentRef_3302_);
lean_dec_ref(v_inst_3301_);
if (v_isShared_3319_ == 0)
{
lean_ctor_set(v___x_3318_, 1, v_a_3322_);
lean_ctor_set(v___x_3318_, 0, v_snd_3316_);
v___x_3328_ = v___x_3318_;
goto v_reusejp_3327_;
}
else
{
lean_object* v_reuseFailAlloc_3332_; 
v_reuseFailAlloc_3332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3332_, 0, v_snd_3316_);
lean_ctor_set(v_reuseFailAlloc_3332_, 1, v_a_3322_);
v___x_3328_ = v_reuseFailAlloc_3332_;
goto v_reusejp_3327_;
}
v_reusejp_3327_:
{
lean_object* v___x_3330_; 
if (v_isShared_3325_ == 0)
{
lean_ctor_set(v___x_3324_, 0, v___x_3328_);
v___x_3330_ = v___x_3324_;
goto v_reusejp_3329_;
}
else
{
lean_object* v_reuseFailAlloc_3331_; 
v_reuseFailAlloc_3331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3331_, 0, v___x_3328_);
v___x_3330_ = v_reuseFailAlloc_3331_;
goto v_reusejp_3329_;
}
v_reusejp_3329_:
{
return v___x_3330_;
}
}
}
}
}
else
{
lean_object* v_a_3334_; lean_object* v___x_3336_; uint8_t v_isShared_3337_; uint8_t v_isSharedCheck_3341_; 
lean_del_object(v___x_3318_);
lean_dec(v_snd_3316_);
lean_dec(v_parentRef_3302_);
lean_dec_ref(v_inst_3301_);
v_a_3334_ = lean_ctor_get(v___x_3321_, 0);
v_isSharedCheck_3341_ = !lean_is_exclusive(v___x_3321_);
if (v_isSharedCheck_3341_ == 0)
{
v___x_3336_ = v___x_3321_;
v_isShared_3337_ = v_isSharedCheck_3341_;
goto v_resetjp_3335_;
}
else
{
lean_inc(v_a_3334_);
lean_dec(v___x_3321_);
v___x_3336_ = lean_box(0);
v_isShared_3337_ = v_isSharedCheck_3341_;
goto v_resetjp_3335_;
}
v_resetjp_3335_:
{
lean_object* v___x_3339_; 
if (v_isShared_3337_ == 0)
{
v___x_3339_ = v___x_3336_;
goto v_reusejp_3338_;
}
else
{
lean_object* v_reuseFailAlloc_3340_; 
v_reuseFailAlloc_3340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3340_, 0, v_a_3334_);
v___x_3339_ = v_reuseFailAlloc_3340_;
goto v_reusejp_3338_;
}
v_reusejp_3338_:
{
return v___x_3339_;
}
}
}
}
}
else
{
lean_object* v_snd_3344_; lean_object* v___x_3346_; uint8_t v_isShared_3347_; uint8_t v_isSharedCheck_3369_; 
v_snd_3344_ = lean_ctor_get(v_val_3314_, 1);
v_isSharedCheck_3369_ = !lean_is_exclusive(v_val_3314_);
if (v_isSharedCheck_3369_ == 0)
{
lean_object* v_unused_3370_; 
v_unused_3370_ = lean_ctor_get(v_val_3314_, 0);
lean_dec(v_unused_3370_);
v___x_3346_ = v_val_3314_;
v_isShared_3347_ = v_isSharedCheck_3369_;
goto v_resetjp_3345_;
}
else
{
lean_inc(v_snd_3344_);
lean_dec(v_val_3314_);
v___x_3346_ = lean_box(0);
v_isShared_3347_ = v_isSharedCheck_3369_;
goto v_resetjp_3345_;
}
v_resetjp_3345_:
{
lean_object* v_r_3348_; lean_object* v___x_3349_; 
v_r_3348_ = lean_ctor_get(v_fst_3315_, 0);
lean_inc_ref(v_r_3348_);
lean_dec_ref_known(v_fst_3315_, 1);
v___x_3349_ = lp_aesop_Aesop_applyPostponedSafeRule___redArg(v_inst_3301_, v_r_3348_, v_parentRef_3302_, v_a_3304_, v_a_3305_, v_a_3306_, v_a_3307_, v_a_3308_, v_a_3309_, v_a_3310_, v_a_3311_);
if (lean_obj_tag(v___x_3349_) == 0)
{
lean_object* v_a_3350_; lean_object* v___x_3352_; uint8_t v_isShared_3353_; uint8_t v_isSharedCheck_3360_; 
v_a_3350_ = lean_ctor_get(v___x_3349_, 0);
v_isSharedCheck_3360_ = !lean_is_exclusive(v___x_3349_);
if (v_isSharedCheck_3360_ == 0)
{
v___x_3352_ = v___x_3349_;
v_isShared_3353_ = v_isSharedCheck_3360_;
goto v_resetjp_3351_;
}
else
{
lean_inc(v_a_3350_);
lean_dec(v___x_3349_);
v___x_3352_ = lean_box(0);
v_isShared_3353_ = v_isSharedCheck_3360_;
goto v_resetjp_3351_;
}
v_resetjp_3351_:
{
lean_object* v___x_3355_; 
if (v_isShared_3347_ == 0)
{
lean_ctor_set(v___x_3346_, 1, v_a_3350_);
lean_ctor_set(v___x_3346_, 0, v_snd_3344_);
v___x_3355_ = v___x_3346_;
goto v_reusejp_3354_;
}
else
{
lean_object* v_reuseFailAlloc_3359_; 
v_reuseFailAlloc_3359_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3359_, 0, v_snd_3344_);
lean_ctor_set(v_reuseFailAlloc_3359_, 1, v_a_3350_);
v___x_3355_ = v_reuseFailAlloc_3359_;
goto v_reusejp_3354_;
}
v_reusejp_3354_:
{
lean_object* v___x_3357_; 
if (v_isShared_3353_ == 0)
{
lean_ctor_set(v___x_3352_, 0, v___x_3355_);
v___x_3357_ = v___x_3352_;
goto v_reusejp_3356_;
}
else
{
lean_object* v_reuseFailAlloc_3358_; 
v_reuseFailAlloc_3358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3358_, 0, v___x_3355_);
v___x_3357_ = v_reuseFailAlloc_3358_;
goto v_reusejp_3356_;
}
v_reusejp_3356_:
{
return v___x_3357_;
}
}
}
}
else
{
lean_object* v_a_3361_; lean_object* v___x_3363_; uint8_t v_isShared_3364_; uint8_t v_isSharedCheck_3368_; 
lean_del_object(v___x_3346_);
lean_dec(v_snd_3344_);
v_a_3361_ = lean_ctor_get(v___x_3349_, 0);
v_isSharedCheck_3368_ = !lean_is_exclusive(v___x_3349_);
if (v_isSharedCheck_3368_ == 0)
{
v___x_3363_ = v___x_3349_;
v_isShared_3364_ = v_isSharedCheck_3368_;
goto v_resetjp_3362_;
}
else
{
lean_inc(v_a_3361_);
lean_dec(v___x_3349_);
v___x_3363_ = lean_box(0);
v_isShared_3364_ = v_isSharedCheck_3368_;
goto v_resetjp_3362_;
}
v_resetjp_3362_:
{
lean_object* v___x_3366_; 
if (v_isShared_3364_ == 0)
{
v___x_3366_ = v___x_3363_;
goto v_reusejp_3365_;
}
else
{
lean_object* v_reuseFailAlloc_3367_; 
v_reuseFailAlloc_3367_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3367_, 0, v_a_3361_);
v___x_3366_ = v_reuseFailAlloc_3367_;
goto v_reusejp_3365_;
}
v_reusejp_3365_:
{
return v___x_3366_;
}
}
}
}
}
}
else
{
lean_object* v___x_3371_; lean_object* v___x_3372_; lean_object* v___x_3373_; 
lean_dec(v___x_3313_);
lean_dec(v_parentRef_3302_);
lean_dec_ref(v_inst_3301_);
v___x_3371_ = lean_box(2);
v___x_3372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3372_, 0, v_queue_3303_);
lean_ctor_set(v___x_3372_, 1, v___x_3371_);
v___x_3373_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3373_, 0, v___x_3372_);
return v___x_3373_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___redArg___boxed(lean_object* v_inst_3374_, lean_object* v_parentRef_3375_, lean_object* v_queue_3376_, lean_object* v_a_3377_, lean_object* v_a_3378_, lean_object* v_a_3379_, lean_object* v_a_3380_, lean_object* v_a_3381_, lean_object* v_a_3382_, lean_object* v_a_3383_, lean_object* v_a_3384_, lean_object* v_a_3385_){
_start:
{
lean_object* v_res_3386_; 
v_res_3386_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___redArg(v_inst_3374_, v_parentRef_3375_, v_queue_3376_, v_a_3377_, v_a_3378_, v_a_3379_, v_a_3380_, v_a_3381_, v_a_3382_, v_a_3383_, v_a_3384_);
lean_dec(v_a_3384_);
lean_dec_ref(v_a_3383_);
lean_dec(v_a_3382_);
lean_dec_ref(v_a_3381_);
lean_dec(v_a_3380_);
lean_dec(v_a_3379_);
lean_dec(v_a_3378_);
lean_dec_ref(v_a_3377_);
return v_res_3386_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop(lean_object* v_Q_3387_, lean_object* v_inst_3388_, lean_object* v_parentRef_3389_, lean_object* v_queue_3390_, lean_object* v_a_3391_, lean_object* v_a_3392_, lean_object* v_a_3393_, lean_object* v_a_3394_, lean_object* v_a_3395_, lean_object* v_a_3396_, lean_object* v_a_3397_, lean_object* v_a_3398_){
_start:
{
lean_object* v___x_3400_; 
v___x_3400_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___redArg(v_inst_3388_, v_parentRef_3389_, v_queue_3390_, v_a_3391_, v_a_3392_, v_a_3393_, v_a_3394_, v_a_3395_, v_a_3396_, v_a_3397_, v_a_3398_);
return v___x_3400_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___boxed(lean_object* v_Q_3401_, lean_object* v_inst_3402_, lean_object* v_parentRef_3403_, lean_object* v_queue_3404_, lean_object* v_a_3405_, lean_object* v_a_3406_, lean_object* v_a_3407_, lean_object* v_a_3408_, lean_object* v_a_3409_, lean_object* v_a_3410_, lean_object* v_a_3411_, lean_object* v_a_3412_, lean_object* v_a_3413_){
_start:
{
lean_object* v_res_3414_; 
v_res_3414_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop(v_Q_3401_, v_inst_3402_, v_parentRef_3403_, v_queue_3404_, v_a_3405_, v_a_3406_, v_a_3407_, v_a_3408_, v_a_3409_, v_a_3410_, v_a_3411_, v_a_3412_);
lean_dec(v_a_3412_);
lean_dec_ref(v_a_3411_);
lean_dec(v_a_3410_);
lean_dec_ref(v_a_3409_);
lean_dec(v_a_3408_);
lean_dec(v_a_3407_);
lean_dec(v_a_3406_);
lean_dec_ref(v_a_3405_);
return v_res_3414_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstUnsafeRule___redArg(lean_object* v_inst_3415_, lean_object* v_postponedSafeRules_3416_, lean_object* v_parentRef_3417_, lean_object* v_a_3418_, lean_object* v_a_3419_, lean_object* v_a_3420_, lean_object* v_a_3421_, lean_object* v_a_3422_, lean_object* v_a_3423_, lean_object* v_a_3424_, lean_object* v_a_3425_){
_start:
{
lean_object* v___x_3427_; lean_object* v_iteration_3428_; lean_object* v_ruleSet_3429_; uint8_t v___x_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; 
v___x_3427_ = lean_st_ref_get(v_a_3419_);
v_iteration_3428_ = lean_ctor_get(v___x_3427_, 0);
lean_inc(v_iteration_3428_);
lean_dec(v___x_3427_);
v_ruleSet_3429_ = lean_ctor_get(v_a_3418_, 0);
v___x_3430_ = 2;
lean_inc_ref(v_ruleSet_3429_);
v___x_3431_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3431_, 0, v_iteration_3428_);
lean_ctor_set(v___x_3431_, 1, v_ruleSet_3429_);
v___x_3432_ = lp_aesop_Aesop_GoalRef_updateForwardState(v___x_3430_, v_parentRef_3417_, v___x_3431_, v_a_3420_, v_a_3421_, v_a_3422_, v_a_3423_, v_a_3424_, v_a_3425_);
lean_dec_ref_known(v___x_3431_, 2);
if (lean_obj_tag(v___x_3432_) == 0)
{
lean_object* v___x_3433_; 
lean_dec_ref_known(v___x_3432_, 1);
v___x_3433_ = lp_aesop_Aesop_selectUnsafeRules___redArg(v_inst_3415_, v_postponedSafeRules_3416_, v_parentRef_3417_, v_a_3418_, v_a_3419_, v_a_3420_, v_a_3421_, v_a_3422_, v_a_3423_, v_a_3424_, v_a_3425_);
if (lean_obj_tag(v___x_3433_) == 0)
{
lean_object* v_a_3434_; lean_object* v___x_3435_; 
v_a_3434_ = lean_ctor_get(v___x_3433_, 0);
lean_inc(v_a_3434_);
lean_dec_ref_known(v___x_3433_, 1);
lean_inc(v_parentRef_3417_);
v___x_3435_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_runFirstUnsafeRule_loop___redArg(v_inst_3415_, v_parentRef_3417_, v_a_3434_, v_a_3418_, v_a_3419_, v_a_3420_, v_a_3421_, v_a_3422_, v_a_3423_, v_a_3424_, v_a_3425_);
if (lean_obj_tag(v___x_3435_) == 0)
{
lean_object* v_a_3436_; lean_object* v___x_3438_; uint8_t v_isShared_3439_; uint8_t v_isSharedCheck_3503_; 
v_a_3436_ = lean_ctor_get(v___x_3435_, 0);
v_isSharedCheck_3503_ = !lean_is_exclusive(v___x_3435_);
if (v_isSharedCheck_3503_ == 0)
{
v___x_3438_ = v___x_3435_;
v_isShared_3439_ = v_isSharedCheck_3503_;
goto v_resetjp_3437_;
}
else
{
lean_inc(v_a_3436_);
lean_dec(v___x_3435_);
v___x_3438_ = lean_box(0);
v_isShared_3439_ = v_isSharedCheck_3503_;
goto v_resetjp_3437_;
}
v_resetjp_3437_:
{
lean_object* v_fst_3440_; lean_object* v_snd_3441_; lean_object* v___x_3442_; lean_object* v___x_3443_; lean_object* v___x_3444_; lean_object* v_introGoal_3445_; lean_object* v_elimGoal_3446_; lean_object* v___x_3447_; lean_object* v_id_3448_; lean_object* v_parent_3449_; lean_object* v_children_3450_; lean_object* v_origin_3451_; lean_object* v_depth_3452_; uint8_t v_state_3453_; uint8_t v_isIrrelevant_3454_; uint8_t v_isForcedUnprovable_3455_; lean_object* v_preNormGoal_3456_; lean_object* v_normalizationState_3457_; lean_object* v_mvars_3458_; lean_object* v_forwardState_3459_; lean_object* v_forwardRuleMatches_3460_; double v_successProbability_3461_; lean_object* v_addedInIteration_3462_; lean_object* v_lastExpandedInIteration_3463_; uint8_t v_unsafeRulesSelected_3464_; lean_object* v_failedRapps_3465_; lean_object* v___x_3467_; uint8_t v_isShared_3468_; uint8_t v_isSharedCheck_3501_; 
v_fst_3440_ = lean_ctor_get(v_a_3436_, 0);
lean_inc(v_fst_3440_);
v_snd_3441_ = lean_ctor_get(v_a_3436_, 1);
lean_inc(v_snd_3441_);
lean_dec(v_a_3436_);
v___x_3442_ = lean_st_ref_get(v_a_3419_);
lean_dec(v___x_3442_);
v___x_3443_ = lean_st_ref_take(v_parentRef_3417_);
v___x_3444_ = lp_aesop_Aesop_treeImpl;
v_introGoal_3445_ = lean_ctor_get(v___x_3444_, 0);
v_elimGoal_3446_ = lean_ctor_get(v___x_3444_, 1);
lean_inc_ref(v_elimGoal_3446_);
v___x_3447_ = lean_apply_1(v_elimGoal_3446_, v___x_3443_);
v_id_3448_ = lean_ctor_get(v___x_3447_, 0);
v_parent_3449_ = lean_ctor_get(v___x_3447_, 1);
v_children_3450_ = lean_ctor_get(v___x_3447_, 2);
v_origin_3451_ = lean_ctor_get(v___x_3447_, 3);
v_depth_3452_ = lean_ctor_get(v___x_3447_, 4);
v_state_3453_ = lean_ctor_get_uint8(v___x_3447_, sizeof(void*)*14 + 8);
v_isIrrelevant_3454_ = lean_ctor_get_uint8(v___x_3447_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_3455_ = lean_ctor_get_uint8(v___x_3447_, sizeof(void*)*14 + 10);
v_preNormGoal_3456_ = lean_ctor_get(v___x_3447_, 5);
v_normalizationState_3457_ = lean_ctor_get(v___x_3447_, 6);
v_mvars_3458_ = lean_ctor_get(v___x_3447_, 7);
v_forwardState_3459_ = lean_ctor_get(v___x_3447_, 8);
v_forwardRuleMatches_3460_ = lean_ctor_get(v___x_3447_, 9);
v_successProbability_3461_ = lean_ctor_get_float(v___x_3447_, sizeof(void*)*14);
v_addedInIteration_3462_ = lean_ctor_get(v___x_3447_, 10);
v_lastExpandedInIteration_3463_ = lean_ctor_get(v___x_3447_, 11);
v_unsafeRulesSelected_3464_ = lean_ctor_get_uint8(v___x_3447_, sizeof(void*)*14 + 11);
v_failedRapps_3465_ = lean_ctor_get(v___x_3447_, 13);
v_isSharedCheck_3501_ = !lean_is_exclusive(v___x_3447_);
if (v_isSharedCheck_3501_ == 0)
{
lean_object* v_unused_3502_; 
v_unused_3502_ = lean_ctor_get(v___x_3447_, 12);
lean_dec(v_unused_3502_);
v___x_3467_ = v___x_3447_;
v_isShared_3468_ = v_isSharedCheck_3501_;
goto v_resetjp_3466_;
}
else
{
lean_inc(v_failedRapps_3465_);
lean_inc(v_lastExpandedInIteration_3463_);
lean_inc(v_addedInIteration_3462_);
lean_inc(v_forwardRuleMatches_3460_);
lean_inc(v_forwardState_3459_);
lean_inc(v_mvars_3458_);
lean_inc(v_normalizationState_3457_);
lean_inc(v_preNormGoal_3456_);
lean_inc(v_depth_3452_);
lean_inc(v_origin_3451_);
lean_inc(v_children_3450_);
lean_inc(v_parent_3449_);
lean_inc(v_id_3448_);
lean_dec(v___x_3447_);
v___x_3467_ = lean_box(0);
v_isShared_3468_ = v_isSharedCheck_3501_;
goto v_resetjp_3466_;
}
v_resetjp_3466_:
{
lean_object* v___x_3470_; 
lean_inc(v_fst_3440_);
if (v_isShared_3468_ == 0)
{
lean_ctor_set(v___x_3467_, 12, v_fst_3440_);
v___x_3470_ = v___x_3467_;
goto v_reusejp_3469_;
}
else
{
lean_object* v_reuseFailAlloc_3500_; 
v_reuseFailAlloc_3500_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_3500_, 0, v_id_3448_);
lean_ctor_set(v_reuseFailAlloc_3500_, 1, v_parent_3449_);
lean_ctor_set(v_reuseFailAlloc_3500_, 2, v_children_3450_);
lean_ctor_set(v_reuseFailAlloc_3500_, 3, v_origin_3451_);
lean_ctor_set(v_reuseFailAlloc_3500_, 4, v_depth_3452_);
lean_ctor_set(v_reuseFailAlloc_3500_, 5, v_preNormGoal_3456_);
lean_ctor_set(v_reuseFailAlloc_3500_, 6, v_normalizationState_3457_);
lean_ctor_set(v_reuseFailAlloc_3500_, 7, v_mvars_3458_);
lean_ctor_set(v_reuseFailAlloc_3500_, 8, v_forwardState_3459_);
lean_ctor_set(v_reuseFailAlloc_3500_, 9, v_forwardRuleMatches_3460_);
lean_ctor_set(v_reuseFailAlloc_3500_, 10, v_addedInIteration_3462_);
lean_ctor_set(v_reuseFailAlloc_3500_, 11, v_lastExpandedInIteration_3463_);
lean_ctor_set(v_reuseFailAlloc_3500_, 12, v_fst_3440_);
lean_ctor_set(v_reuseFailAlloc_3500_, 13, v_failedRapps_3465_);
lean_ctor_set_uint8(v_reuseFailAlloc_3500_, sizeof(void*)*14 + 8, v_state_3453_);
lean_ctor_set_uint8(v_reuseFailAlloc_3500_, sizeof(void*)*14 + 9, v_isIrrelevant_3454_);
lean_ctor_set_uint8(v_reuseFailAlloc_3500_, sizeof(void*)*14 + 10, v_isForcedUnprovable_3455_);
lean_ctor_set_float(v_reuseFailAlloc_3500_, sizeof(void*)*14, v_successProbability_3461_);
lean_ctor_set_uint8(v_reuseFailAlloc_3500_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_3464_);
v___x_3470_ = v_reuseFailAlloc_3500_;
goto v_reusejp_3469_;
}
v_reusejp_3469_:
{
lean_object* v___x_3471_; lean_object* v___x_3472_; lean_object* v_start_3473_; lean_object* v_stop_3474_; uint8_t v___x_3475_; 
lean_inc(v_introGoal_3445_);
v___x_3471_ = lean_apply_1(v_introGoal_3445_, v___x_3470_);
v___x_3472_ = lean_st_ref_set(v_parentRef_3417_, v___x_3471_);
v_start_3473_ = lean_ctor_get(v_fst_3440_, 1);
lean_inc(v_start_3473_);
v_stop_3474_ = lean_ctor_get(v_fst_3440_, 2);
lean_inc(v_stop_3474_);
lean_dec(v_fst_3440_);
v___x_3475_ = lean_nat_dec_eq(v_start_3473_, v_stop_3474_);
lean_dec(v_stop_3474_);
lean_dec(v_start_3473_);
if (v___x_3475_ == 0)
{
lean_object* v___x_3477_; 
lean_dec(v_parentRef_3417_);
if (v_isShared_3439_ == 0)
{
lean_ctor_set(v___x_3438_, 0, v_snd_3441_);
v___x_3477_ = v___x_3438_;
goto v_reusejp_3476_;
}
else
{
lean_object* v_reuseFailAlloc_3478_; 
v_reuseFailAlloc_3478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3478_, 0, v_snd_3441_);
v___x_3477_ = v_reuseFailAlloc_3478_;
goto v_reusejp_3476_;
}
v_reusejp_3476_:
{
return v___x_3477_;
}
}
else
{
lean_object* v___x_3479_; lean_object* v___x_3480_; lean_object* v___x_3481_; uint8_t v_state_3482_; uint8_t v___x_3483_; 
v___x_3479_ = lean_st_ref_get(v_a_3419_);
lean_dec(v___x_3479_);
v___x_3480_ = lean_st_ref_get(v_parentRef_3417_);
lean_inc_ref(v_elimGoal_3446_);
lean_inc(v___x_3480_);
v___x_3481_ = lean_apply_1(v_elimGoal_3446_, v___x_3480_);
v_state_3482_ = lean_ctor_get_uint8(v___x_3481_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_3481_);
v___x_3483_ = lp_aesop_Aesop_GoalState_isProven(v_state_3482_);
if (v___x_3483_ == 0)
{
if (v___x_3475_ == 0)
{
lean_object* v___x_3485_; 
lean_dec(v___x_3480_);
lean_dec(v_parentRef_3417_);
if (v_isShared_3439_ == 0)
{
lean_ctor_set(v___x_3438_, 0, v_snd_3441_);
v___x_3485_ = v___x_3438_;
goto v_reusejp_3484_;
}
else
{
lean_object* v_reuseFailAlloc_3486_; 
v_reuseFailAlloc_3486_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3486_, 0, v_snd_3441_);
v___x_3485_ = v_reuseFailAlloc_3486_;
goto v_reusejp_3484_;
}
v_reusejp_3484_:
{
return v___x_3485_;
}
}
else
{
lean_object* v___x_3487_; uint8_t v___x_3488_; 
v___x_3487_ = lean_st_ref_get(v_a_3419_);
lean_dec(v___x_3487_);
v___x_3488_ = lp_aesop_Aesop_Goal_isUnprovableNoCache(v___x_3480_);
if (v___x_3488_ == 0)
{
lean_object* v___x_3490_; 
lean_dec(v_parentRef_3417_);
if (v_isShared_3439_ == 0)
{
lean_ctor_set(v___x_3438_, 0, v_snd_3441_);
v___x_3490_ = v___x_3438_;
goto v_reusejp_3489_;
}
else
{
lean_object* v_reuseFailAlloc_3491_; 
v_reuseFailAlloc_3491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3491_, 0, v_snd_3441_);
v___x_3490_ = v_reuseFailAlloc_3491_;
goto v_reusejp_3489_;
}
v_reusejp_3489_:
{
return v___x_3490_;
}
}
else
{
lean_object* v___x_3492_; lean_object* v___x_3493_; lean_object* v___x_3495_; 
v___x_3492_ = lean_st_ref_get(v_a_3419_);
lean_dec(v___x_3492_);
v___x_3493_ = lp_aesop_Aesop_GoalRef_markUnprovable(v_parentRef_3417_);
lean_dec(v_parentRef_3417_);
if (v_isShared_3439_ == 0)
{
lean_ctor_set(v___x_3438_, 0, v_snd_3441_);
v___x_3495_ = v___x_3438_;
goto v_reusejp_3494_;
}
else
{
lean_object* v_reuseFailAlloc_3496_; 
v_reuseFailAlloc_3496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3496_, 0, v_snd_3441_);
v___x_3495_ = v_reuseFailAlloc_3496_;
goto v_reusejp_3494_;
}
v_reusejp_3494_:
{
return v___x_3495_;
}
}
}
}
else
{
lean_object* v___x_3498_; 
lean_dec(v___x_3480_);
lean_dec(v_parentRef_3417_);
if (v_isShared_3439_ == 0)
{
lean_ctor_set(v___x_3438_, 0, v_snd_3441_);
v___x_3498_ = v___x_3438_;
goto v_reusejp_3497_;
}
else
{
lean_object* v_reuseFailAlloc_3499_; 
v_reuseFailAlloc_3499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3499_, 0, v_snd_3441_);
v___x_3498_ = v_reuseFailAlloc_3499_;
goto v_reusejp_3497_;
}
v_reusejp_3497_:
{
return v___x_3498_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3504_; lean_object* v___x_3506_; uint8_t v_isShared_3507_; uint8_t v_isSharedCheck_3511_; 
lean_dec(v_parentRef_3417_);
v_a_3504_ = lean_ctor_get(v___x_3435_, 0);
v_isSharedCheck_3511_ = !lean_is_exclusive(v___x_3435_);
if (v_isSharedCheck_3511_ == 0)
{
v___x_3506_ = v___x_3435_;
v_isShared_3507_ = v_isSharedCheck_3511_;
goto v_resetjp_3505_;
}
else
{
lean_inc(v_a_3504_);
lean_dec(v___x_3435_);
v___x_3506_ = lean_box(0);
v_isShared_3507_ = v_isSharedCheck_3511_;
goto v_resetjp_3505_;
}
v_resetjp_3505_:
{
lean_object* v___x_3509_; 
if (v_isShared_3507_ == 0)
{
v___x_3509_ = v___x_3506_;
goto v_reusejp_3508_;
}
else
{
lean_object* v_reuseFailAlloc_3510_; 
v_reuseFailAlloc_3510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3510_, 0, v_a_3504_);
v___x_3509_ = v_reuseFailAlloc_3510_;
goto v_reusejp_3508_;
}
v_reusejp_3508_:
{
return v___x_3509_;
}
}
}
}
else
{
lean_object* v_a_3512_; lean_object* v___x_3514_; uint8_t v_isShared_3515_; uint8_t v_isSharedCheck_3519_; 
lean_dec(v_parentRef_3417_);
lean_dec_ref(v_inst_3415_);
v_a_3512_ = lean_ctor_get(v___x_3433_, 0);
v_isSharedCheck_3519_ = !lean_is_exclusive(v___x_3433_);
if (v_isSharedCheck_3519_ == 0)
{
v___x_3514_ = v___x_3433_;
v_isShared_3515_ = v_isSharedCheck_3519_;
goto v_resetjp_3513_;
}
else
{
lean_inc(v_a_3512_);
lean_dec(v___x_3433_);
v___x_3514_ = lean_box(0);
v_isShared_3515_ = v_isSharedCheck_3519_;
goto v_resetjp_3513_;
}
v_resetjp_3513_:
{
lean_object* v___x_3517_; 
if (v_isShared_3515_ == 0)
{
v___x_3517_ = v___x_3514_;
goto v_reusejp_3516_;
}
else
{
lean_object* v_reuseFailAlloc_3518_; 
v_reuseFailAlloc_3518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3518_, 0, v_a_3512_);
v___x_3517_ = v_reuseFailAlloc_3518_;
goto v_reusejp_3516_;
}
v_reusejp_3516_:
{
return v___x_3517_;
}
}
}
}
else
{
lean_object* v_a_3520_; lean_object* v___x_3522_; uint8_t v_isShared_3523_; uint8_t v_isSharedCheck_3527_; 
lean_dec(v_parentRef_3417_);
lean_dec_ref(v_postponedSafeRules_3416_);
lean_dec_ref(v_inst_3415_);
v_a_3520_ = lean_ctor_get(v___x_3432_, 0);
v_isSharedCheck_3527_ = !lean_is_exclusive(v___x_3432_);
if (v_isSharedCheck_3527_ == 0)
{
v___x_3522_ = v___x_3432_;
v_isShared_3523_ = v_isSharedCheck_3527_;
goto v_resetjp_3521_;
}
else
{
lean_inc(v_a_3520_);
lean_dec(v___x_3432_);
v___x_3522_ = lean_box(0);
v_isShared_3523_ = v_isSharedCheck_3527_;
goto v_resetjp_3521_;
}
v_resetjp_3521_:
{
lean_object* v___x_3525_; 
if (v_isShared_3523_ == 0)
{
v___x_3525_ = v___x_3522_;
goto v_reusejp_3524_;
}
else
{
lean_object* v_reuseFailAlloc_3526_; 
v_reuseFailAlloc_3526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3526_, 0, v_a_3520_);
v___x_3525_ = v_reuseFailAlloc_3526_;
goto v_reusejp_3524_;
}
v_reusejp_3524_:
{
return v___x_3525_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstUnsafeRule___redArg___boxed(lean_object* v_inst_3528_, lean_object* v_postponedSafeRules_3529_, lean_object* v_parentRef_3530_, lean_object* v_a_3531_, lean_object* v_a_3532_, lean_object* v_a_3533_, lean_object* v_a_3534_, lean_object* v_a_3535_, lean_object* v_a_3536_, lean_object* v_a_3537_, lean_object* v_a_3538_, lean_object* v_a_3539_){
_start:
{
lean_object* v_res_3540_; 
v_res_3540_ = lp_aesop_Aesop_runFirstUnsafeRule___redArg(v_inst_3528_, v_postponedSafeRules_3529_, v_parentRef_3530_, v_a_3531_, v_a_3532_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, v_a_3537_, v_a_3538_);
lean_dec(v_a_3538_);
lean_dec_ref(v_a_3537_);
lean_dec(v_a_3536_);
lean_dec_ref(v_a_3535_);
lean_dec(v_a_3534_);
lean_dec(v_a_3533_);
lean_dec(v_a_3532_);
lean_dec_ref(v_a_3531_);
return v_res_3540_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstUnsafeRule(lean_object* v_Q_3541_, lean_object* v_inst_3542_, lean_object* v_postponedSafeRules_3543_, lean_object* v_parentRef_3544_, lean_object* v_a_3545_, lean_object* v_a_3546_, lean_object* v_a_3547_, lean_object* v_a_3548_, lean_object* v_a_3549_, lean_object* v_a_3550_, lean_object* v_a_3551_, lean_object* v_a_3552_){
_start:
{
lean_object* v___x_3554_; 
v___x_3554_ = lp_aesop_Aesop_runFirstUnsafeRule___redArg(v_inst_3542_, v_postponedSafeRules_3543_, v_parentRef_3544_, v_a_3545_, v_a_3546_, v_a_3547_, v_a_3548_, v_a_3549_, v_a_3550_, v_a_3551_, v_a_3552_);
return v___x_3554_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runFirstUnsafeRule___boxed(lean_object* v_Q_3555_, lean_object* v_inst_3556_, lean_object* v_postponedSafeRules_3557_, lean_object* v_parentRef_3558_, lean_object* v_a_3559_, lean_object* v_a_3560_, lean_object* v_a_3561_, lean_object* v_a_3562_, lean_object* v_a_3563_, lean_object* v_a_3564_, lean_object* v_a_3565_, lean_object* v_a_3566_, lean_object* v_a_3567_){
_start:
{
lean_object* v_res_3568_; 
v_res_3568_ = lp_aesop_Aesop_runFirstUnsafeRule(v_Q_3555_, v_inst_3556_, v_postponedSafeRules_3557_, v_parentRef_3558_, v_a_3559_, v_a_3560_, v_a_3561_, v_a_3562_, v_a_3563_, v_a_3564_, v_a_3565_, v_a_3566_);
lean_dec(v_a_3566_);
lean_dec_ref(v_a_3565_);
lean_dec(v_a_3564_);
lean_dec_ref(v_a_3563_);
lean_dec(v_a_3562_);
lean_dec(v_a_3561_);
lean_dec(v_a_3560_);
lean_dec_ref(v_a_3559_);
return v_res_3568_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__1(void){
_start:
{
lean_object* v___x_3570_; lean_object* v___x_3571_; 
v___x_3570_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__0));
v___x_3571_ = l_Lean_stringToMessageData(v___x_3570_);
return v___x_3571_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg(lean_object* v_result_3572_){
_start:
{
lean_object* v___f_3574_; lean_object* v___x_3575_; lean_object* v___x_3576_; lean_object* v___x_3577_; lean_object* v___x_3578_; lean_object* v___x_3579_; 
v___f_3574_ = ((lean_object*)(lp_aesop_Aesop_runUnsafeRule___redArg___closed__0));
v___x_3575_ = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(v___f_3574_, v_result_3572_);
v___x_3576_ = l_Lean_stringToMessageData(v___x_3575_);
v___x_3577_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__1, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___closed__1);
v___x_3578_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3578_, 0, v___x_3576_);
lean_ctor_set(v___x_3578_, 1, v___x_3577_);
v___x_3579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3579_, 0, v___x_3578_);
return v___x_3579_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg___boxed(lean_object* v_result_3580_, lean_object* v_a_3581_){
_start:
{
lean_object* v_res_3582_; 
v_res_3582_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg(v_result_3580_);
return v_res_3582_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe(lean_object* v_Q_3583_, lean_object* v_inst_3584_, lean_object* v_result_3585_, lean_object* v_a_3586_, lean_object* v_a_3587_, lean_object* v_a_3588_, lean_object* v_a_3589_, lean_object* v_a_3590_, lean_object* v_a_3591_, lean_object* v_a_3592_, lean_object* v_a_3593_){
_start:
{
lean_object* v___x_3595_; 
v___x_3595_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___redArg(v_result_3585_);
return v___x_3595_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___boxed(lean_object* v_Q_3596_, lean_object* v_inst_3597_, lean_object* v_result_3598_, lean_object* v_a_3599_, lean_object* v_a_3600_, lean_object* v_a_3601_, lean_object* v_a_3602_, lean_object* v_a_3603_, lean_object* v_a_3604_, lean_object* v_a_3605_, lean_object* v_a_3606_, lean_object* v_a_3607_){
_start:
{
lean_object* v_res_3608_; 
v_res_3608_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe(v_Q_3596_, v_inst_3597_, v_result_3598_, v_a_3599_, v_a_3600_, v_a_3601_, v_a_3602_, v_a_3603_, v_a_3604_, v_a_3605_, v_a_3606_);
lean_dec(v_a_3606_);
lean_dec_ref(v_a_3605_);
lean_dec(v_a_3604_);
lean_dec_ref(v_a_3603_);
lean_dec(v_a_3602_);
lean_dec(v_a_3601_);
lean_dec(v_a_3600_);
lean_dec_ref(v_a_3599_);
lean_dec_ref(v_inst_3597_);
return v_res_3608_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___redArg(lean_object* v_inst_3609_, lean_object* v_gref_3610_, lean_object* v_postponedSafeRules_3611_, lean_object* v_a_3612_, lean_object* v_a_3613_, lean_object* v_a_3614_, lean_object* v_a_3615_, lean_object* v_a_3616_, lean_object* v_a_3617_, lean_object* v_a_3618_, lean_object* v_a_3619_){
_start:
{
lean_object* v___x_3621_; lean_object* v___x_3622_; lean_object* v___x_3623_; lean_object* v___x_3624_; lean_object* v_options_3625_; uint8_t v_hasTrace_3626_; 
v___x_3621_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_3609_);
v___x_3622_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_3623_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_3609_);
v___x_3624_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_options_3625_ = lean_ctor_get(v_a_3618_, 2);
v_hasTrace_3626_ = lean_ctor_get_uint8(v_options_3625_, sizeof(void*)*1);
if (v_hasTrace_3626_ == 0)
{
lean_object* v___x_3627_; 
lean_dec_ref(v___x_3623_);
lean_dec_ref(v___x_3621_);
v___x_3627_ = lp_aesop_Aesop_runFirstUnsafeRule___redArg(v_inst_3609_, v_postponedSafeRules_3611_, v_gref_3610_, v_a_3612_, v_a_3613_, v_a_3614_, v_a_3615_, v_a_3616_, v_a_3617_, v_a_3618_, v_a_3619_);
return v___x_3627_;
}
else
{
lean_object* v_inheritedTraceOptions_3628_; lean_object* v___x_3629_; lean_object* v___x_3630_; lean_object* v_traceClass_3631_; lean_object* v___f_3632_; lean_object* v___f_3633_; lean_object* v___x_3634_; lean_object* v___x_3635_; lean_object* v___x_3636_; lean_object* v___x_3637_; uint8_t v___x_3638_; lean_object* v___y_3640_; lean_object* v___y_3641_; lean_object* v_a_3642_; lean_object* v___y_3657_; lean_object* v___y_3658_; lean_object* v_a_3659_; 
v_inheritedTraceOptions_3628_ = lean_ctor_get(v_a_3618_, 13);
v___x_3629_ = lean_st_ref_get(v_a_3613_);
lean_dec(v___x_3629_);
v___x_3630_ = lp_aesop_Aesop_TraceOption_steps;
v_traceClass_3631_ = lean_ctor_get(v___x_3630_, 0);
v___f_3632_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
v___f_3633_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
lean_inc_ref(v_inst_3609_);
v___x_3634_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtUnsafe___boxed), 12, 2);
lean_closure_set(v___x_3634_, 0, lean_box(0));
lean_closure_set(v___x_3634_, 1, v_inst_3609_);
v___x_3635_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23));
v___x_3636_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_3631_);
v___x_3637_ = l_Lean_Name_append(v___x_3636_, v_traceClass_3631_);
v___x_3638_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3628_, v_options_3625_, v___x_3637_);
lean_dec(v___x_3637_);
if (v___x_3638_ == 0)
{
lean_object* v___x_3724_; lean_object* v___x_3725_; lean_object* v___x_3726_; uint8_t v___x_3727_; 
v___x_3724_ = l_Lean_KVMap_instValueBool;
v___x_3725_ = l_Lean_trace_profiler;
v___x_3726_ = l_Lean_Option_get___redArg(v___x_3724_, v_options_3625_, v___x_3725_);
v___x_3727_ = lean_unbox(v___x_3726_);
lean_dec(v___x_3726_);
if (v___x_3727_ == 0)
{
lean_object* v___x_3728_; 
lean_dec_ref(v___x_3634_);
lean_dec_ref(v___x_3623_);
lean_dec_ref(v___x_3621_);
v___x_3728_ = lp_aesop_Aesop_runFirstUnsafeRule___redArg(v_inst_3609_, v_postponedSafeRules_3611_, v_gref_3610_, v_a_3612_, v_a_3613_, v_a_3614_, v_a_3615_, v_a_3616_, v_a_3617_, v_a_3618_, v_a_3619_);
return v___x_3728_;
}
else
{
goto v___jp_3670_;
}
}
else
{
goto v___jp_3670_;
}
v___jp_3639_:
{
lean_object* v___x_3643_; lean_object* v___x_3644_; double v___x_3645_; double v___x_3646_; double v___x_3647_; double v___x_3648_; double v___x_3649_; lean_object* v___x_3650_; lean_object* v___x_3651_; lean_object* v___x_3652_; lean_object* v___x_3653_; lean_object* v___x_18325__overap_3654_; lean_object* v___x_3655_; 
v___x_3643_ = lean_st_ref_get(v_a_3613_);
lean_dec(v___x_3643_);
v___x_3644_ = lean_io_mono_nanos_now();
v___x_3645_ = lean_float_of_nat(v___y_3641_);
v___x_3646_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_3647_ = lean_float_div(v___x_3645_, v___x_3646_);
v___x_3648_ = lean_float_of_nat(v___x_3644_);
v___x_3649_ = lean_float_div(v___x_3648_, v___x_3646_);
v___x_3650_ = lean_box_float(v___x_3647_);
v___x_3651_ = lean_box_float(v___x_3649_);
v___x_3652_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3652_, 0, v___x_3650_);
lean_ctor_set(v___x_3652_, 1, v___x_3651_);
v___x_3653_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3653_, 0, v_a_3642_);
lean_ctor_set(v___x_3653_, 1, v___x_3652_);
lean_inc(v_traceClass_3631_);
v___x_18325__overap_3654_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3621_, v___x_3622_, v___x_3623_, v___f_3633_, lean_box(0), v___x_3624_, v___f_3632_, v_traceClass_3631_, v_hasTrace_3626_, v___x_3635_, v_options_3625_, v___x_3638_, v___y_3640_, v___x_3634_, v___x_3653_);
lean_inc(v_a_3619_);
lean_inc_ref(v_a_3618_);
lean_inc(v_a_3617_);
lean_inc_ref(v_a_3616_);
lean_inc(v_a_3615_);
lean_inc(v_a_3614_);
lean_inc(v_a_3613_);
lean_inc_ref(v_a_3612_);
v___x_3655_ = lean_apply_9(v___x_18325__overap_3654_, v_a_3612_, v_a_3613_, v_a_3614_, v_a_3615_, v_a_3616_, v_a_3617_, v_a_3618_, v_a_3619_, lean_box(0));
return v___x_3655_;
}
v___jp_3656_:
{
lean_object* v___x_3660_; lean_object* v___x_3661_; double v___x_3662_; double v___x_3663_; lean_object* v___x_3664_; lean_object* v___x_3665_; lean_object* v___x_3666_; lean_object* v___x_3667_; lean_object* v___x_18352__overap_3668_; lean_object* v___x_3669_; 
v___x_3660_ = lean_st_ref_get(v_a_3613_);
lean_dec(v___x_3660_);
v___x_3661_ = lean_io_get_num_heartbeats();
v___x_3662_ = lean_float_of_nat(v___y_3658_);
v___x_3663_ = lean_float_of_nat(v___x_3661_);
v___x_3664_ = lean_box_float(v___x_3662_);
v___x_3665_ = lean_box_float(v___x_3663_);
v___x_3666_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3666_, 0, v___x_3664_);
lean_ctor_set(v___x_3666_, 1, v___x_3665_);
v___x_3667_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3667_, 0, v_a_3659_);
lean_ctor_set(v___x_3667_, 1, v___x_3666_);
lean_inc(v_traceClass_3631_);
v___x_18352__overap_3668_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3621_, v___x_3622_, v___x_3623_, v___f_3633_, lean_box(0), v___x_3624_, v___f_3632_, v_traceClass_3631_, v_hasTrace_3626_, v___x_3635_, v_options_3625_, v___x_3638_, v___y_3657_, v___x_3634_, v___x_3667_);
lean_inc(v_a_3619_);
lean_inc_ref(v_a_3618_);
lean_inc(v_a_3617_);
lean_inc_ref(v_a_3616_);
lean_inc(v_a_3615_);
lean_inc(v_a_3614_);
lean_inc(v_a_3613_);
lean_inc_ref(v_a_3612_);
v___x_3669_ = lean_apply_9(v___x_18352__overap_3668_, v_a_3612_, v_a_3613_, v_a_3614_, v_a_3615_, v_a_3616_, v_a_3617_, v_a_3618_, v_a_3619_, lean_box(0));
return v___x_3669_;
}
v___jp_3670_:
{
lean_object* v___x_18296__overap_3671_; lean_object* v___x_3672_; 
lean_inc_ref(v___x_3621_);
v___x_18296__overap_3671_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_3621_, v___x_3622_);
lean_inc(v_a_3619_);
lean_inc_ref(v_a_3618_);
lean_inc(v_a_3617_);
lean_inc_ref(v_a_3616_);
lean_inc(v_a_3615_);
lean_inc(v_a_3614_);
lean_inc(v_a_3613_);
lean_inc_ref(v_a_3612_);
v___x_3672_ = lean_apply_9(v___x_18296__overap_3671_, v_a_3612_, v_a_3613_, v_a_3614_, v_a_3615_, v_a_3616_, v_a_3617_, v_a_3618_, v_a_3619_, lean_box(0));
if (lean_obj_tag(v___x_3672_) == 0)
{
lean_object* v_a_3673_; lean_object* v___x_3674_; lean_object* v___x_3675_; lean_object* v___x_3676_; uint8_t v___x_3677_; 
v_a_3673_ = lean_ctor_get(v___x_3672_, 0);
lean_inc(v_a_3673_);
lean_dec_ref_known(v___x_3672_, 1);
v___x_3674_ = l_Lean_KVMap_instValueBool;
v___x_3675_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3676_ = l_Lean_Option_get___redArg(v___x_3674_, v_options_3625_, v___x_3675_);
v___x_3677_ = lean_unbox(v___x_3676_);
lean_dec(v___x_3676_);
if (v___x_3677_ == 0)
{
lean_object* v___x_3678_; lean_object* v___x_3679_; lean_object* v___x_3680_; 
v___x_3678_ = lean_st_ref_get(v_a_3613_);
lean_dec(v___x_3678_);
v___x_3679_ = lean_io_mono_nanos_now();
v___x_3680_ = lp_aesop_Aesop_runFirstUnsafeRule___redArg(v_inst_3609_, v_postponedSafeRules_3611_, v_gref_3610_, v_a_3612_, v_a_3613_, v_a_3614_, v_a_3615_, v_a_3616_, v_a_3617_, v_a_3618_, v_a_3619_);
if (lean_obj_tag(v___x_3680_) == 0)
{
lean_object* v_a_3681_; lean_object* v___x_3683_; uint8_t v_isShared_3684_; uint8_t v_isSharedCheck_3688_; 
v_a_3681_ = lean_ctor_get(v___x_3680_, 0);
v_isSharedCheck_3688_ = !lean_is_exclusive(v___x_3680_);
if (v_isSharedCheck_3688_ == 0)
{
v___x_3683_ = v___x_3680_;
v_isShared_3684_ = v_isSharedCheck_3688_;
goto v_resetjp_3682_;
}
else
{
lean_inc(v_a_3681_);
lean_dec(v___x_3680_);
v___x_3683_ = lean_box(0);
v_isShared_3684_ = v_isSharedCheck_3688_;
goto v_resetjp_3682_;
}
v_resetjp_3682_:
{
lean_object* v___x_3686_; 
if (v_isShared_3684_ == 0)
{
lean_ctor_set_tag(v___x_3683_, 1);
v___x_3686_ = v___x_3683_;
goto v_reusejp_3685_;
}
else
{
lean_object* v_reuseFailAlloc_3687_; 
v_reuseFailAlloc_3687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3687_, 0, v_a_3681_);
v___x_3686_ = v_reuseFailAlloc_3687_;
goto v_reusejp_3685_;
}
v_reusejp_3685_:
{
v___y_3640_ = v_a_3673_;
v___y_3641_ = v___x_3679_;
v_a_3642_ = v___x_3686_;
goto v___jp_3639_;
}
}
}
else
{
lean_object* v_a_3689_; lean_object* v___x_3691_; uint8_t v_isShared_3692_; uint8_t v_isSharedCheck_3696_; 
v_a_3689_ = lean_ctor_get(v___x_3680_, 0);
v_isSharedCheck_3696_ = !lean_is_exclusive(v___x_3680_);
if (v_isSharedCheck_3696_ == 0)
{
v___x_3691_ = v___x_3680_;
v_isShared_3692_ = v_isSharedCheck_3696_;
goto v_resetjp_3690_;
}
else
{
lean_inc(v_a_3689_);
lean_dec(v___x_3680_);
v___x_3691_ = lean_box(0);
v_isShared_3692_ = v_isSharedCheck_3696_;
goto v_resetjp_3690_;
}
v_resetjp_3690_:
{
lean_object* v___x_3694_; 
if (v_isShared_3692_ == 0)
{
lean_ctor_set_tag(v___x_3691_, 0);
v___x_3694_ = v___x_3691_;
goto v_reusejp_3693_;
}
else
{
lean_object* v_reuseFailAlloc_3695_; 
v_reuseFailAlloc_3695_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3695_, 0, v_a_3689_);
v___x_3694_ = v_reuseFailAlloc_3695_;
goto v_reusejp_3693_;
}
v_reusejp_3693_:
{
v___y_3640_ = v_a_3673_;
v___y_3641_ = v___x_3679_;
v_a_3642_ = v___x_3694_;
goto v___jp_3639_;
}
}
}
}
else
{
lean_object* v___x_3697_; lean_object* v___x_3698_; lean_object* v___x_3699_; 
v___x_3697_ = lean_st_ref_get(v_a_3613_);
lean_dec(v___x_3697_);
v___x_3698_ = lean_io_get_num_heartbeats();
v___x_3699_ = lp_aesop_Aesop_runFirstUnsafeRule___redArg(v_inst_3609_, v_postponedSafeRules_3611_, v_gref_3610_, v_a_3612_, v_a_3613_, v_a_3614_, v_a_3615_, v_a_3616_, v_a_3617_, v_a_3618_, v_a_3619_);
if (lean_obj_tag(v___x_3699_) == 0)
{
lean_object* v_a_3700_; lean_object* v___x_3702_; uint8_t v_isShared_3703_; uint8_t v_isSharedCheck_3707_; 
v_a_3700_ = lean_ctor_get(v___x_3699_, 0);
v_isSharedCheck_3707_ = !lean_is_exclusive(v___x_3699_);
if (v_isSharedCheck_3707_ == 0)
{
v___x_3702_ = v___x_3699_;
v_isShared_3703_ = v_isSharedCheck_3707_;
goto v_resetjp_3701_;
}
else
{
lean_inc(v_a_3700_);
lean_dec(v___x_3699_);
v___x_3702_ = lean_box(0);
v_isShared_3703_ = v_isSharedCheck_3707_;
goto v_resetjp_3701_;
}
v_resetjp_3701_:
{
lean_object* v___x_3705_; 
if (v_isShared_3703_ == 0)
{
lean_ctor_set_tag(v___x_3702_, 1);
v___x_3705_ = v___x_3702_;
goto v_reusejp_3704_;
}
else
{
lean_object* v_reuseFailAlloc_3706_; 
v_reuseFailAlloc_3706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3706_, 0, v_a_3700_);
v___x_3705_ = v_reuseFailAlloc_3706_;
goto v_reusejp_3704_;
}
v_reusejp_3704_:
{
v___y_3657_ = v_a_3673_;
v___y_3658_ = v___x_3698_;
v_a_3659_ = v___x_3705_;
goto v___jp_3656_;
}
}
}
else
{
lean_object* v_a_3708_; lean_object* v___x_3710_; uint8_t v_isShared_3711_; uint8_t v_isSharedCheck_3715_; 
v_a_3708_ = lean_ctor_get(v___x_3699_, 0);
v_isSharedCheck_3715_ = !lean_is_exclusive(v___x_3699_);
if (v_isSharedCheck_3715_ == 0)
{
v___x_3710_ = v___x_3699_;
v_isShared_3711_ = v_isSharedCheck_3715_;
goto v_resetjp_3709_;
}
else
{
lean_inc(v_a_3708_);
lean_dec(v___x_3699_);
v___x_3710_ = lean_box(0);
v_isShared_3711_ = v_isSharedCheck_3715_;
goto v_resetjp_3709_;
}
v_resetjp_3709_:
{
lean_object* v___x_3713_; 
if (v_isShared_3711_ == 0)
{
lean_ctor_set_tag(v___x_3710_, 0);
v___x_3713_ = v___x_3710_;
goto v_reusejp_3712_;
}
else
{
lean_object* v_reuseFailAlloc_3714_; 
v_reuseFailAlloc_3714_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3714_, 0, v_a_3708_);
v___x_3713_ = v_reuseFailAlloc_3714_;
goto v_reusejp_3712_;
}
v_reusejp_3712_:
{
v___y_3657_ = v_a_3673_;
v___y_3658_ = v___x_3698_;
v_a_3659_ = v___x_3713_;
goto v___jp_3656_;
}
}
}
}
}
else
{
lean_object* v_a_3716_; lean_object* v___x_3718_; uint8_t v_isShared_3719_; uint8_t v_isSharedCheck_3723_; 
lean_dec_ref(v___x_3634_);
lean_dec_ref(v___x_3623_);
lean_dec_ref(v___x_3621_);
lean_dec_ref(v_postponedSafeRules_3611_);
lean_dec(v_gref_3610_);
lean_dec_ref(v_inst_3609_);
v_a_3716_ = lean_ctor_get(v___x_3672_, 0);
v_isSharedCheck_3723_ = !lean_is_exclusive(v___x_3672_);
if (v_isSharedCheck_3723_ == 0)
{
v___x_3718_ = v___x_3672_;
v_isShared_3719_ = v_isSharedCheck_3723_;
goto v_resetjp_3717_;
}
else
{
lean_inc(v_a_3716_);
lean_dec(v___x_3672_);
v___x_3718_ = lean_box(0);
v_isShared_3719_ = v_isSharedCheck_3723_;
goto v_resetjp_3717_;
}
v_resetjp_3717_:
{
lean_object* v___x_3721_; 
if (v_isShared_3719_ == 0)
{
v___x_3721_ = v___x_3718_;
goto v_reusejp_3720_;
}
else
{
lean_object* v_reuseFailAlloc_3722_; 
v_reuseFailAlloc_3722_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3722_, 0, v_a_3716_);
v___x_3721_ = v_reuseFailAlloc_3722_;
goto v_reusejp_3720_;
}
v_reusejp_3720_:
{
return v___x_3721_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___redArg___boxed(lean_object* v_inst_3729_, lean_object* v_gref_3730_, lean_object* v_postponedSafeRules_3731_, lean_object* v_a_3732_, lean_object* v_a_3733_, lean_object* v_a_3734_, lean_object* v_a_3735_, lean_object* v_a_3736_, lean_object* v_a_3737_, lean_object* v_a_3738_, lean_object* v_a_3739_, lean_object* v_a_3740_){
_start:
{
lean_object* v_res_3741_; 
v_res_3741_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___redArg(v_inst_3729_, v_gref_3730_, v_postponedSafeRules_3731_, v_a_3732_, v_a_3733_, v_a_3734_, v_a_3735_, v_a_3736_, v_a_3737_, v_a_3738_, v_a_3739_);
lean_dec(v_a_3739_);
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_a_3736_);
lean_dec(v_a_3735_);
lean_dec(v_a_3734_);
lean_dec(v_a_3733_);
lean_dec_ref(v_a_3732_);
return v_res_3741_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe(lean_object* v_Q_3742_, lean_object* v_inst_3743_, lean_object* v_gref_3744_, lean_object* v_postponedSafeRules_3745_, lean_object* v_a_3746_, lean_object* v_a_3747_, lean_object* v_a_3748_, lean_object* v_a_3749_, lean_object* v_a_3750_, lean_object* v_a_3751_, lean_object* v_a_3752_, lean_object* v_a_3753_){
_start:
{
lean_object* v___x_3755_; 
v___x_3755_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___redArg(v_inst_3743_, v_gref_3744_, v_postponedSafeRules_3745_, v_a_3746_, v_a_3747_, v_a_3748_, v_a_3749_, v_a_3750_, v_a_3751_, v_a_3752_, v_a_3753_);
return v___x_3755_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___boxed(lean_object* v_Q_3756_, lean_object* v_inst_3757_, lean_object* v_gref_3758_, lean_object* v_postponedSafeRules_3759_, lean_object* v_a_3760_, lean_object* v_a_3761_, lean_object* v_a_3762_, lean_object* v_a_3763_, lean_object* v_a_3764_, lean_object* v_a_3765_, lean_object* v_a_3766_, lean_object* v_a_3767_, lean_object* v_a_3768_){
_start:
{
lean_object* v_res_3769_; 
v_res_3769_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe(v_Q_3756_, v_inst_3757_, v_gref_3758_, v_postponedSafeRules_3759_, v_a_3760_, v_a_3761_, v_a_3762_, v_a_3763_, v_a_3764_, v_a_3765_, v_a_3766_, v_a_3767_);
lean_dec(v_a_3767_);
lean_dec_ref(v_a_3766_);
lean_dec(v_a_3765_);
lean_dec_ref(v_a_3764_);
lean_dec(v_a_3763_);
lean_dec(v_a_3762_);
lean_dec(v_a_3761_);
lean_dec_ref(v_a_3760_);
return v_res_3769_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__1(void){
_start:
{
lean_object* v___x_3771_; lean_object* v___x_3772_; 
v___x_3771_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__0));
v___x_3772_ = l_Lean_stringToMessageData(v___x_3771_);
return v___x_3772_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__2(void){
_start:
{
lean_object* v___x_3773_; lean_object* v___x_3774_; 
v___x_3773_ = lp_aesop_Aesop_ruleErrorEmoji;
v___x_3774_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3774_, 0, v___x_3773_);
return v___x_3774_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__3(void){
_start:
{
lean_object* v___x_3775_; lean_object* v___x_3776_; 
v___x_3775_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__2, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__2_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__2);
v___x_3776_ = l_Lean_MessageData_ofFormat(v___x_3775_);
return v___x_3776_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__4(void){
_start:
{
lean_object* v___x_3777_; lean_object* v___x_3778_; 
v___x_3777_ = lp_aesop_Aesop_ruleSuccessEmoji;
v___x_3778_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3778_, 0, v___x_3777_);
return v___x_3778_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__5(void){
_start:
{
lean_object* v___x_3779_; lean_object* v___x_3780_; 
v___x_3779_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__4, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__4_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__4);
v___x_3780_ = l_Lean_MessageData_ofFormat(v___x_3779_);
return v___x_3780_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__6(void){
_start:
{
lean_object* v___x_3781_; lean_object* v___x_3782_; 
v___x_3781_ = lp_aesop_Aesop_ruleProvedEmoji;
v___x_3782_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3782_, 0, v___x_3781_);
return v___x_3782_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__7(void){
_start:
{
lean_object* v___x_3783_; lean_object* v___x_3784_; 
v___x_3783_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__6, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__6_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__6);
v___x_3784_ = l_Lean_MessageData_ofFormat(v___x_3783_);
return v___x_3784_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg(lean_object* v_result_3785_){
_start:
{
lean_object* v___y_3788_; 
if (lean_obj_tag(v_result_3785_) == 0)
{
lean_object* v___x_3792_; 
v___x_3792_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__3, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__3_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__3);
v___y_3788_ = v___x_3792_;
goto v___jp_3787_;
}
else
{
lean_object* v_a_3793_; uint8_t v___x_3794_; 
v_a_3793_ = lean_ctor_get(v_result_3785_, 0);
v___x_3794_ = lean_unbox(v_a_3793_);
if (v___x_3794_ == 0)
{
lean_object* v___x_3795_; 
v___x_3795_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__5, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__5);
v___y_3788_ = v___x_3795_;
goto v___jp_3787_;
}
else
{
lean_object* v___x_3796_; 
v___x_3796_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__7, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__7_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__7);
v___y_3788_ = v___x_3796_;
goto v___jp_3787_;
}
}
v___jp_3787_:
{
lean_object* v___x_3789_; lean_object* v___x_3790_; lean_object* v___x_3791_; 
v___x_3789_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__1, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___closed__1);
lean_inc_ref(v___y_3788_);
v___x_3790_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3790_, 0, v___y_3788_);
lean_ctor_set(v___x_3790_, 1, v___x_3789_);
v___x_3791_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3791_, 0, v___x_3790_);
return v___x_3791_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg___boxed(lean_object* v_result_3797_, lean_object* v_a_3798_){
_start:
{
lean_object* v_res_3799_; 
v_res_3799_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg(v_result_3797_);
lean_dec_ref(v_result_3797_);
return v_res_3799_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm(lean_object* v_Q_3800_, lean_object* v_inst_3801_, lean_object* v_result_3802_, lean_object* v_a_3803_, lean_object* v_a_3804_, lean_object* v_a_3805_, lean_object* v_a_3806_, lean_object* v_a_3807_, lean_object* v_a_3808_, lean_object* v_a_3809_, lean_object* v_a_3810_){
_start:
{
lean_object* v___x_3812_; 
v___x_3812_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___redArg(v_result_3802_);
return v___x_3812_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___boxed(lean_object* v_Q_3813_, lean_object* v_inst_3814_, lean_object* v_result_3815_, lean_object* v_a_3816_, lean_object* v_a_3817_, lean_object* v_a_3818_, lean_object* v_a_3819_, lean_object* v_a_3820_, lean_object* v_a_3821_, lean_object* v_a_3822_, lean_object* v_a_3823_, lean_object* v_a_3824_){
_start:
{
lean_object* v_res_3825_; 
v_res_3825_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm(v_Q_3813_, v_inst_3814_, v_result_3815_, v_a_3816_, v_a_3817_, v_a_3818_, v_a_3819_, v_a_3820_, v_a_3821_, v_a_3822_, v_a_3823_);
lean_dec(v_a_3823_);
lean_dec_ref(v_a_3822_);
lean_dec(v_a_3821_);
lean_dec_ref(v_a_3820_);
lean_dec(v_a_3819_);
lean_dec(v_a_3818_);
lean_dec(v_a_3817_);
lean_dec_ref(v_a_3816_);
lean_dec_ref(v_result_3815_);
lean_dec_ref(v_inst_3814_);
return v_res_3825_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__2(void){
_start:
{
lean_object* v___x_3828_; lean_object* v___x_3829_; 
v___x_3828_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__1));
v___x_3829_ = l_Lean_stringToMessageData(v___x_3828_);
return v___x_3829_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg(lean_object* v_result_3830_){
_start:
{
lean_object* v___f_3832_; lean_object* v___x_3833_; lean_object* v___x_3834_; lean_object* v___x_3835_; lean_object* v___x_3836_; lean_object* v___x_3837_; 
v___f_3832_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__0));
v___x_3833_ = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(v___f_3832_, v_result_3830_);
v___x_3834_ = l_Lean_stringToMessageData(v___x_3833_);
v___x_3835_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__2, &lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__2_once, _init_lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___closed__2);
v___x_3836_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3836_, 0, v___x_3834_);
lean_ctor_set(v___x_3836_, 1, v___x_3835_);
v___x_3837_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3837_, 0, v___x_3836_);
return v___x_3837_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg___boxed(lean_object* v_result_3838_, lean_object* v_a_3839_){
_start:
{
lean_object* v_res_3840_; 
v_res_3840_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg(v_result_3838_);
return v_res_3840_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe(lean_object* v_Q_3841_, lean_object* v_inst_3842_, lean_object* v_result_3843_, lean_object* v_a_3844_, lean_object* v_a_3845_, lean_object* v_a_3846_, lean_object* v_a_3847_, lean_object* v_a_3848_, lean_object* v_a_3849_, lean_object* v_a_3850_, lean_object* v_a_3851_){
_start:
{
lean_object* v___x_3853_; 
v___x_3853_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___redArg(v_result_3843_);
return v___x_3853_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___boxed(lean_object* v_Q_3854_, lean_object* v_inst_3855_, lean_object* v_result_3856_, lean_object* v_a_3857_, lean_object* v_a_3858_, lean_object* v_a_3859_, lean_object* v_a_3860_, lean_object* v_a_3861_, lean_object* v_a_3862_, lean_object* v_a_3863_, lean_object* v_a_3864_, lean_object* v_a_3865_){
_start:
{
lean_object* v_res_3866_; 
v_res_3866_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe(v_Q_3854_, v_inst_3855_, v_result_3856_, v_a_3857_, v_a_3858_, v_a_3859_, v_a_3860_, v_a_3861_, v_a_3862_, v_a_3863_, v_a_3864_);
lean_dec(v_a_3864_);
lean_dec_ref(v_a_3863_);
lean_dec(v_a_3862_);
lean_dec_ref(v_a_3861_);
lean_dec(v_a_3860_);
lean_dec(v_a_3859_);
lean_dec(v_a_3858_);
lean_dec_ref(v_a_3857_);
lean_dec_ref(v_inst_3855_);
return v_res_3866_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandGoal___redArg___closed__5(void){
_start:
{
lean_object* v___x_3875_; lean_object* v___x_3876_; lean_object* v___x_3877_; lean_object* v___x_3878_; 
v___x_3875_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_3876_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__1));
v___x_3877_ = ((lean_object*)(lp_aesop_Aesop_expandGoal___redArg___closed__4));
v___x_3878_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_3877_, v___x_3876_, v___x_3875_);
return v___x_3878_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandGoal___redArg___closed__6(void){
_start:
{
lean_object* v___x_3879_; lean_object* v___f_3880_; lean_object* v___f_3881_; lean_object* v___x_3882_; 
v___x_3879_ = lean_obj_once(&lp_aesop_Aesop_expandGoal___redArg___closed__5, &lp_aesop_Aesop_expandGoal___redArg___closed__5_once, _init_lp_aesop_Aesop_expandGoal___redArg___closed__5);
v___f_3880_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__0));
v___f_3881_ = ((lean_object*)(lp_aesop_Aesop_expandGoal___redArg___closed__3));
v___x_3882_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_3881_, v___f_3880_, v___x_3879_);
return v___x_3882_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandGoal___redArg___closed__8(void){
_start:
{
lean_object* v___x_3884_; lean_object* v___x_3885_; 
v___x_3884_ = ((lean_object*)(lp_aesop_Aesop_expandGoal___redArg___closed__7));
v___x_3885_ = l_Lean_stringToMessageData(v___x_3884_);
return v___x_3885_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandGoal___redArg(lean_object* v_inst_3887_, lean_object* v_gref_3888_, lean_object* v_a_3889_, lean_object* v_a_3890_, lean_object* v_a_3891_, lean_object* v_a_3892_, lean_object* v_a_3893_, lean_object* v_a_3894_, lean_object* v_a_3895_, lean_object* v_a_3896_){
_start:
{
lean_object* v___y_3899_; lean_object* v___y_3900_; lean_object* v___y_3901_; lean_object* v___y_3902_; lean_object* v___y_3903_; lean_object* v___y_3904_; lean_object* v___y_3905_; lean_object* v___y_3906_; lean_object* v___y_3907_; lean_object* v___x_3950_; lean_object* v___x_3951_; lean_object* v___x_3952_; lean_object* v___x_3953_; lean_object* v___x_3954_; lean_object* v_toMonadOptions_3955_; lean_object* v___x_3956_; lean_object* v_options_3957_; lean_object* v_inheritedTraceOptions_3958_; uint8_t v_hasTrace_3959_; lean_object* v___x_3960_; lean_object* v___f_3961_; lean_object* v___x_3962_; uint8_t v___x_3963_; lean_object* v___y_3965_; lean_object* v___y_3966_; lean_object* v___y_3967_; lean_object* v___y_3968_; lean_object* v___y_3969_; lean_object* v___y_3970_; lean_object* v___y_3971_; lean_object* v___y_3972_; lean_object* v___y_3973_; lean_object* v___y_3974_; lean_object* v___y_3975_; lean_object* v___y_3976_; lean_object* v___y_3977_; lean_object* v___y_3978_; uint8_t v___y_3979_; lean_object* v___y_3980_; lean_object* v_a_3981_; lean_object* v___y_3996_; lean_object* v___y_3997_; lean_object* v___y_3998_; lean_object* v___y_3999_; lean_object* v___y_4000_; lean_object* v___y_4001_; lean_object* v___y_4002_; lean_object* v___y_4003_; lean_object* v___y_4004_; lean_object* v___y_4005_; lean_object* v___y_4006_; lean_object* v___y_4007_; lean_object* v___y_4008_; lean_object* v___y_4009_; uint8_t v___y_4010_; lean_object* v___y_4011_; lean_object* v_a_4012_; lean_object* v___y_4024_; lean_object* v___y_4025_; lean_object* v___y_4026_; lean_object* v___y_4027_; lean_object* v___y_4028_; lean_object* v___y_4029_; lean_object* v___y_4030_; lean_object* v___y_4031_; lean_object* v___y_4032_; lean_object* v___y_4033_; lean_object* v___y_4034_; lean_object* v___y_4035_; uint8_t v___y_4036_; lean_object* v___y_4037_; lean_object* v___y_4092_; lean_object* v___y_4093_; lean_object* v___y_4094_; lean_object* v___y_4095_; lean_object* v___y_4096_; lean_object* v___y_4097_; lean_object* v___y_4098_; lean_object* v_options_4099_; uint8_t v_hasTrace_4100_; lean_object* v_inheritedTraceOptions_4101_; lean_object* v___y_4102_; lean_object* v___y_4118_; 
v___x_3950_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__3);
v___x_3951_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_3887_);
v___x_3952_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__15);
v___x_3953_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_3887_);
v___x_3954_ = lean_obj_once(&lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2, &lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2_once, _init_lp_aesop_Aesop_runRegularRuleCore___redArg___closed__2);
v_toMonadOptions_3955_ = lean_ctor_get(v___x_3954_, 0);
v___x_3956_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__17);
v_options_3957_ = lean_ctor_get(v_a_3895_, 2);
v_inheritedTraceOptions_3958_ = lean_ctor_get(v_a_3895_, 13);
v_hasTrace_3959_ = lean_ctor_get_uint8(v_options_3957_, sizeof(void*)*1);
v___x_3960_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_3961_ = lean_obj_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__21);
v___x_3962_ = lp_aesop_Aesop_TraceOption_steps;
v___x_3963_ = 1;
if (v_hasTrace_3959_ == 0)
{
lean_object* v___x_4238_; 
v___x_4238_ = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(v_gref_3888_, v_inst_3887_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_);
v___y_4118_ = v___x_4238_;
goto v___jp_4117_;
}
else
{
lean_object* v___x_4239_; lean_object* v_traceClass_4240_; lean_object* v___f_4241_; lean_object* v___x_4242_; lean_object* v___x_4243_; lean_object* v___x_4244_; lean_object* v___x_4245_; uint8_t v___x_4246_; lean_object* v___y_4248_; lean_object* v___y_4249_; lean_object* v_a_4250_; lean_object* v___y_4265_; lean_object* v___y_4266_; lean_object* v_a_4267_; 
v___x_4239_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4239_);
v_traceClass_4240_ = lean_ctor_get(v___x_3962_, 0);
v___f_4241_ = ((lean_object*)(lp_aesop_Aesop_expandGoal___redArg___closed__9));
lean_inc_ref(v_inst_3887_);
v___x_4242_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtNorm___boxed), 12, 2);
lean_closure_set(v___x_4242_, 0, lean_box(0));
lean_closure_set(v___x_4242_, 1, v_inst_3887_);
v___x_4243_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23));
v___x_4244_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_4240_);
v___x_4245_ = l_Lean_Name_append(v___x_4244_, v_traceClass_4240_);
v___x_4246_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3958_, v_options_3957_, v___x_4245_);
lean_dec(v___x_4245_);
if (v___x_4246_ == 0)
{
lean_object* v___x_4332_; lean_object* v___x_4333_; lean_object* v___x_4334_; uint8_t v___x_4335_; 
v___x_4332_ = l_Lean_KVMap_instValueBool;
v___x_4333_ = l_Lean_trace_profiler;
v___x_4334_ = l_Lean_Option_get___redArg(v___x_4332_, v_options_3957_, v___x_4333_);
v___x_4335_ = lean_unbox(v___x_4334_);
lean_dec(v___x_4334_);
if (v___x_4335_ == 0)
{
lean_object* v___x_4336_; 
lean_dec_ref(v___x_4242_);
v___x_4336_ = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(v_gref_3888_, v_inst_3887_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_);
v___y_4118_ = v___x_4336_;
goto v___jp_4117_;
}
else
{
goto v___jp_4278_;
}
}
else
{
goto v___jp_4278_;
}
v___jp_4247_:
{
lean_object* v___x_4251_; lean_object* v___x_4252_; double v___x_4253_; double v___x_4254_; double v___x_4255_; double v___x_4256_; double v___x_4257_; lean_object* v___x_4258_; lean_object* v___x_4259_; lean_object* v___x_4260_; lean_object* v___x_4261_; lean_object* v___x_51621__overap_4262_; lean_object* v___x_4263_; 
v___x_4251_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4251_);
v___x_4252_ = lean_io_mono_nanos_now();
v___x_4253_ = lean_float_of_nat(v___y_4248_);
v___x_4254_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_4255_ = lean_float_div(v___x_4253_, v___x_4254_);
v___x_4256_ = lean_float_of_nat(v___x_4252_);
v___x_4257_ = lean_float_div(v___x_4256_, v___x_4254_);
v___x_4258_ = lean_box_float(v___x_4255_);
v___x_4259_ = lean_box_float(v___x_4257_);
v___x_4260_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4260_, 0, v___x_4258_);
lean_ctor_set(v___x_4260_, 1, v___x_4259_);
v___x_4261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4261_, 0, v_a_4250_);
lean_ctor_set(v___x_4261_, 1, v___x_4260_);
lean_inc(v_traceClass_4240_);
lean_inc_ref(v___x_3953_);
lean_inc_ref(v___x_3951_);
v___x_51621__overap_4262_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3951_, v___x_3952_, v___x_3953_, v___f_3961_, lean_box(0), v___x_3956_, v___f_4241_, v_traceClass_4240_, v___x_3963_, v___x_4243_, v_options_3957_, v___x_4246_, v___y_4249_, v___x_4242_, v___x_4261_);
lean_inc(v_a_3896_);
lean_inc_ref(v_a_3895_);
lean_inc(v_a_3894_);
lean_inc_ref(v_a_3893_);
lean_inc(v_a_3892_);
lean_inc(v_a_3891_);
lean_inc(v_a_3890_);
lean_inc_ref(v_a_3889_);
v___x_4263_ = lean_apply_9(v___x_51621__overap_4262_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_, lean_box(0));
v___y_4118_ = v___x_4263_;
goto v___jp_4117_;
}
v___jp_4264_:
{
lean_object* v___x_4268_; lean_object* v___x_4269_; double v___x_4270_; double v___x_4271_; lean_object* v___x_4272_; lean_object* v___x_4273_; lean_object* v___x_4274_; lean_object* v___x_4275_; lean_object* v___x_51648__overap_4276_; lean_object* v___x_4277_; 
v___x_4268_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4268_);
v___x_4269_ = lean_io_get_num_heartbeats();
v___x_4270_ = lean_float_of_nat(v___y_4265_);
v___x_4271_ = lean_float_of_nat(v___x_4269_);
v___x_4272_ = lean_box_float(v___x_4270_);
v___x_4273_ = lean_box_float(v___x_4271_);
v___x_4274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4274_, 0, v___x_4272_);
lean_ctor_set(v___x_4274_, 1, v___x_4273_);
v___x_4275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4275_, 0, v_a_4267_);
lean_ctor_set(v___x_4275_, 1, v___x_4274_);
lean_inc(v_traceClass_4240_);
lean_inc_ref(v___x_3953_);
lean_inc_ref(v___x_3951_);
v___x_51648__overap_4276_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3951_, v___x_3952_, v___x_3953_, v___f_3961_, lean_box(0), v___x_3956_, v___f_4241_, v_traceClass_4240_, v___x_3963_, v___x_4243_, v_options_3957_, v___x_4246_, v___y_4266_, v___x_4242_, v___x_4275_);
lean_inc(v_a_3896_);
lean_inc_ref(v_a_3895_);
lean_inc(v_a_3894_);
lean_inc_ref(v_a_3893_);
lean_inc(v_a_3892_);
lean_inc(v_a_3891_);
lean_inc(v_a_3890_);
lean_inc_ref(v_a_3889_);
v___x_4277_ = lean_apply_9(v___x_51648__overap_4276_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_, lean_box(0));
v___y_4118_ = v___x_4277_;
goto v___jp_4117_;
}
v___jp_4278_:
{
lean_object* v___x_51592__overap_4279_; lean_object* v___x_4280_; 
lean_inc_ref(v___x_3951_);
v___x_51592__overap_4279_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_3951_, v___x_3952_);
lean_inc(v_a_3896_);
lean_inc_ref(v_a_3895_);
lean_inc(v_a_3894_);
lean_inc_ref(v_a_3893_);
lean_inc(v_a_3892_);
lean_inc(v_a_3891_);
lean_inc(v_a_3890_);
lean_inc_ref(v_a_3889_);
v___x_4280_ = lean_apply_9(v___x_51592__overap_4279_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_, lean_box(0));
if (lean_obj_tag(v___x_4280_) == 0)
{
lean_object* v_a_4281_; lean_object* v___x_4282_; lean_object* v___x_4283_; lean_object* v___x_4284_; uint8_t v___x_4285_; 
v_a_4281_ = lean_ctor_get(v___x_4280_, 0);
lean_inc(v_a_4281_);
lean_dec_ref_known(v___x_4280_, 1);
v___x_4282_ = l_Lean_KVMap_instValueBool;
v___x_4283_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4284_ = l_Lean_Option_get___redArg(v___x_4282_, v_options_3957_, v___x_4283_);
v___x_4285_ = lean_unbox(v___x_4284_);
lean_dec(v___x_4284_);
if (v___x_4285_ == 0)
{
lean_object* v___x_4286_; lean_object* v___x_4287_; lean_object* v___x_4288_; 
v___x_4286_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4286_);
v___x_4287_ = lean_io_mono_nanos_now();
v___x_4288_ = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(v_gref_3888_, v_inst_3887_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_);
if (lean_obj_tag(v___x_4288_) == 0)
{
lean_object* v_a_4289_; lean_object* v___x_4291_; uint8_t v_isShared_4292_; uint8_t v_isSharedCheck_4296_; 
v_a_4289_ = lean_ctor_get(v___x_4288_, 0);
v_isSharedCheck_4296_ = !lean_is_exclusive(v___x_4288_);
if (v_isSharedCheck_4296_ == 0)
{
v___x_4291_ = v___x_4288_;
v_isShared_4292_ = v_isSharedCheck_4296_;
goto v_resetjp_4290_;
}
else
{
lean_inc(v_a_4289_);
lean_dec(v___x_4288_);
v___x_4291_ = lean_box(0);
v_isShared_4292_ = v_isSharedCheck_4296_;
goto v_resetjp_4290_;
}
v_resetjp_4290_:
{
lean_object* v___x_4294_; 
if (v_isShared_4292_ == 0)
{
lean_ctor_set_tag(v___x_4291_, 1);
v___x_4294_ = v___x_4291_;
goto v_reusejp_4293_;
}
else
{
lean_object* v_reuseFailAlloc_4295_; 
v_reuseFailAlloc_4295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4295_, 0, v_a_4289_);
v___x_4294_ = v_reuseFailAlloc_4295_;
goto v_reusejp_4293_;
}
v_reusejp_4293_:
{
v___y_4248_ = v___x_4287_;
v___y_4249_ = v_a_4281_;
v_a_4250_ = v___x_4294_;
goto v___jp_4247_;
}
}
}
else
{
lean_object* v_a_4297_; lean_object* v___x_4299_; uint8_t v_isShared_4300_; uint8_t v_isSharedCheck_4304_; 
v_a_4297_ = lean_ctor_get(v___x_4288_, 0);
v_isSharedCheck_4304_ = !lean_is_exclusive(v___x_4288_);
if (v_isSharedCheck_4304_ == 0)
{
v___x_4299_ = v___x_4288_;
v_isShared_4300_ = v_isSharedCheck_4304_;
goto v_resetjp_4298_;
}
else
{
lean_inc(v_a_4297_);
lean_dec(v___x_4288_);
v___x_4299_ = lean_box(0);
v_isShared_4300_ = v_isSharedCheck_4304_;
goto v_resetjp_4298_;
}
v_resetjp_4298_:
{
lean_object* v___x_4302_; 
if (v_isShared_4300_ == 0)
{
lean_ctor_set_tag(v___x_4299_, 0);
v___x_4302_ = v___x_4299_;
goto v_reusejp_4301_;
}
else
{
lean_object* v_reuseFailAlloc_4303_; 
v_reuseFailAlloc_4303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4303_, 0, v_a_4297_);
v___x_4302_ = v_reuseFailAlloc_4303_;
goto v_reusejp_4301_;
}
v_reusejp_4301_:
{
v___y_4248_ = v___x_4287_;
v___y_4249_ = v_a_4281_;
v_a_4250_ = v___x_4302_;
goto v___jp_4247_;
}
}
}
}
else
{
lean_object* v___x_4305_; lean_object* v___x_4306_; lean_object* v___x_4307_; 
v___x_4305_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4305_);
v___x_4306_ = lean_io_get_num_heartbeats();
v___x_4307_ = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(v_gref_3888_, v_inst_3887_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_);
if (lean_obj_tag(v___x_4307_) == 0)
{
lean_object* v_a_4308_; lean_object* v___x_4310_; uint8_t v_isShared_4311_; uint8_t v_isSharedCheck_4315_; 
v_a_4308_ = lean_ctor_get(v___x_4307_, 0);
v_isSharedCheck_4315_ = !lean_is_exclusive(v___x_4307_);
if (v_isSharedCheck_4315_ == 0)
{
v___x_4310_ = v___x_4307_;
v_isShared_4311_ = v_isSharedCheck_4315_;
goto v_resetjp_4309_;
}
else
{
lean_inc(v_a_4308_);
lean_dec(v___x_4307_);
v___x_4310_ = lean_box(0);
v_isShared_4311_ = v_isSharedCheck_4315_;
goto v_resetjp_4309_;
}
v_resetjp_4309_:
{
lean_object* v___x_4313_; 
if (v_isShared_4311_ == 0)
{
lean_ctor_set_tag(v___x_4310_, 1);
v___x_4313_ = v___x_4310_;
goto v_reusejp_4312_;
}
else
{
lean_object* v_reuseFailAlloc_4314_; 
v_reuseFailAlloc_4314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4314_, 0, v_a_4308_);
v___x_4313_ = v_reuseFailAlloc_4314_;
goto v_reusejp_4312_;
}
v_reusejp_4312_:
{
v___y_4265_ = v___x_4306_;
v___y_4266_ = v_a_4281_;
v_a_4267_ = v___x_4313_;
goto v___jp_4264_;
}
}
}
else
{
lean_object* v_a_4316_; lean_object* v___x_4318_; uint8_t v_isShared_4319_; uint8_t v_isSharedCheck_4323_; 
v_a_4316_ = lean_ctor_get(v___x_4307_, 0);
v_isSharedCheck_4323_ = !lean_is_exclusive(v___x_4307_);
if (v_isSharedCheck_4323_ == 0)
{
v___x_4318_ = v___x_4307_;
v_isShared_4319_ = v_isSharedCheck_4323_;
goto v_resetjp_4317_;
}
else
{
lean_inc(v_a_4316_);
lean_dec(v___x_4307_);
v___x_4318_ = lean_box(0);
v_isShared_4319_ = v_isSharedCheck_4323_;
goto v_resetjp_4317_;
}
v_resetjp_4317_:
{
lean_object* v___x_4321_; 
if (v_isShared_4319_ == 0)
{
lean_ctor_set_tag(v___x_4318_, 0);
v___x_4321_ = v___x_4318_;
goto v_reusejp_4320_;
}
else
{
lean_object* v_reuseFailAlloc_4322_; 
v_reuseFailAlloc_4322_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4322_, 0, v_a_4316_);
v___x_4321_ = v_reuseFailAlloc_4322_;
goto v_reusejp_4320_;
}
v_reusejp_4320_:
{
v___y_4265_ = v___x_4306_;
v___y_4266_ = v_a_4281_;
v_a_4267_ = v___x_4321_;
goto v___jp_4264_;
}
}
}
}
}
else
{
lean_object* v_a_4324_; lean_object* v___x_4326_; uint8_t v_isShared_4327_; uint8_t v_isSharedCheck_4331_; 
lean_dec_ref(v___x_4242_);
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_a_4324_ = lean_ctor_get(v___x_4280_, 0);
v_isSharedCheck_4331_ = !lean_is_exclusive(v___x_4280_);
if (v_isSharedCheck_4331_ == 0)
{
v___x_4326_ = v___x_4280_;
v_isShared_4327_ = v_isSharedCheck_4331_;
goto v_resetjp_4325_;
}
else
{
lean_inc(v_a_4324_);
lean_dec(v___x_4280_);
v___x_4326_ = lean_box(0);
v_isShared_4327_ = v_isSharedCheck_4331_;
goto v_resetjp_4325_;
}
v_resetjp_4325_:
{
lean_object* v___x_4329_; 
if (v_isShared_4327_ == 0)
{
v___x_4329_ = v___x_4326_;
goto v_reusejp_4328_;
}
else
{
lean_object* v_reuseFailAlloc_4330_; 
v_reuseFailAlloc_4330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4330_, 0, v_a_4324_);
v___x_4329_ = v_reuseFailAlloc_4330_;
goto v_reusejp_4328_;
}
v_reusejp_4328_:
{
return v___x_4329_;
}
}
}
}
}
v___jp_3898_:
{
if (lean_obj_tag(v___y_3907_) == 0)
{
lean_object* v_a_3908_; lean_object* v___x_3910_; uint8_t v_isShared_3911_; uint8_t v_isSharedCheck_3938_; 
v_a_3908_ = lean_ctor_get(v___y_3907_, 0);
v_isSharedCheck_3938_ = !lean_is_exclusive(v___y_3907_);
if (v_isSharedCheck_3938_ == 0)
{
v___x_3910_ = v___y_3907_;
v_isShared_3911_ = v_isSharedCheck_3938_;
goto v_resetjp_3909_;
}
else
{
lean_inc(v_a_3908_);
lean_dec(v___y_3907_);
v___x_3910_ = lean_box(0);
v_isShared_3911_ = v_isSharedCheck_3938_;
goto v_resetjp_3909_;
}
v_resetjp_3909_:
{
switch(lean_obj_tag(v_a_3908_))
{
case 0:
{
lean_object* v_newRapps_3912_; lean_object* v___x_3914_; uint8_t v_isShared_3915_; uint8_t v_isSharedCheck_3922_; 
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_newRapps_3912_ = lean_ctor_get(v_a_3908_, 0);
v_isSharedCheck_3922_ = !lean_is_exclusive(v_a_3908_);
if (v_isSharedCheck_3922_ == 0)
{
v___x_3914_ = v_a_3908_;
v_isShared_3915_ = v_isSharedCheck_3922_;
goto v_resetjp_3913_;
}
else
{
lean_inc(v_newRapps_3912_);
lean_dec(v_a_3908_);
v___x_3914_ = lean_box(0);
v_isShared_3915_ = v_isSharedCheck_3922_;
goto v_resetjp_3913_;
}
v_resetjp_3913_:
{
lean_object* v___x_3917_; 
if (v_isShared_3915_ == 0)
{
v___x_3917_ = v___x_3914_;
goto v_reusejp_3916_;
}
else
{
lean_object* v_reuseFailAlloc_3921_; 
v_reuseFailAlloc_3921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3921_, 0, v_newRapps_3912_);
v___x_3917_ = v_reuseFailAlloc_3921_;
goto v_reusejp_3916_;
}
v_reusejp_3916_:
{
lean_object* v___x_3919_; 
if (v_isShared_3911_ == 0)
{
lean_ctor_set(v___x_3910_, 0, v___x_3917_);
v___x_3919_ = v___x_3910_;
goto v_reusejp_3918_;
}
else
{
lean_object* v_reuseFailAlloc_3920_; 
v_reuseFailAlloc_3920_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3920_, 0, v___x_3917_);
v___x_3919_ = v_reuseFailAlloc_3920_;
goto v_reusejp_3918_;
}
v_reusejp_3918_:
{
return v___x_3919_;
}
}
}
}
case 1:
{
lean_object* v_newRapps_3923_; lean_object* v___x_3925_; uint8_t v_isShared_3926_; uint8_t v_isSharedCheck_3933_; 
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_newRapps_3923_ = lean_ctor_get(v_a_3908_, 0);
v_isSharedCheck_3933_ = !lean_is_exclusive(v_a_3908_);
if (v_isSharedCheck_3933_ == 0)
{
v___x_3925_ = v_a_3908_;
v_isShared_3926_ = v_isSharedCheck_3933_;
goto v_resetjp_3924_;
}
else
{
lean_inc(v_newRapps_3923_);
lean_dec(v_a_3908_);
v___x_3925_ = lean_box(0);
v_isShared_3926_ = v_isSharedCheck_3933_;
goto v_resetjp_3924_;
}
v_resetjp_3924_:
{
lean_object* v___x_3928_; 
if (v_isShared_3926_ == 0)
{
v___x_3928_ = v___x_3925_;
goto v_reusejp_3927_;
}
else
{
lean_object* v_reuseFailAlloc_3932_; 
v_reuseFailAlloc_3932_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3932_, 0, v_newRapps_3923_);
v___x_3928_ = v_reuseFailAlloc_3932_;
goto v_reusejp_3927_;
}
v_reusejp_3927_:
{
lean_object* v___x_3930_; 
if (v_isShared_3911_ == 0)
{
lean_ctor_set(v___x_3910_, 0, v___x_3928_);
v___x_3930_ = v___x_3910_;
goto v_reusejp_3929_;
}
else
{
lean_object* v_reuseFailAlloc_3931_; 
v_reuseFailAlloc_3931_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3931_, 0, v___x_3928_);
v___x_3930_ = v_reuseFailAlloc_3931_;
goto v_reusejp_3929_;
}
v_reusejp_3929_:
{
return v___x_3930_;
}
}
}
}
case 2:
{
lean_object* v_postponed_3934_; lean_object* v___x_3935_; 
lean_del_object(v___x_3910_);
v_postponed_3934_ = lean_ctor_get(v_a_3908_, 0);
lean_inc_ref(v_postponed_3934_);
lean_dec_ref_known(v_a_3908_, 1);
v___x_3935_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___redArg(v_inst_3887_, v_gref_3888_, v_postponed_3934_, v___y_3906_, v___y_3905_, v___y_3901_, v___y_3904_, v___y_3899_, v___y_3903_, v___y_3902_, v___y_3900_);
return v___x_3935_;
}
default: 
{
lean_object* v___x_3936_; lean_object* v___x_3937_; 
lean_del_object(v___x_3910_);
v___x_3936_ = ((lean_object*)(lp_aesop_Aesop_expandGoal___redArg___closed__0));
v___x_3937_ = lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_doUnsafe___redArg(v_inst_3887_, v_gref_3888_, v___x_3936_, v___y_3906_, v___y_3905_, v___y_3901_, v___y_3904_, v___y_3899_, v___y_3903_, v___y_3902_, v___y_3900_);
return v___x_3937_;
}
}
}
}
else
{
lean_object* v_a_3939_; lean_object* v___x_3941_; uint8_t v_isShared_3942_; uint8_t v_isSharedCheck_3946_; 
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_a_3939_ = lean_ctor_get(v___y_3907_, 0);
v_isSharedCheck_3946_ = !lean_is_exclusive(v___y_3907_);
if (v_isSharedCheck_3946_ == 0)
{
v___x_3941_ = v___y_3907_;
v_isShared_3942_ = v_isSharedCheck_3946_;
goto v_resetjp_3940_;
}
else
{
lean_inc(v_a_3939_);
lean_dec(v___y_3907_);
v___x_3941_ = lean_box(0);
v_isShared_3942_ = v_isSharedCheck_3946_;
goto v_resetjp_3940_;
}
v_resetjp_3940_:
{
lean_object* v___x_3944_; 
if (v_isShared_3942_ == 0)
{
v___x_3944_ = v___x_3941_;
goto v_reusejp_3943_;
}
else
{
lean_object* v_reuseFailAlloc_3945_; 
v_reuseFailAlloc_3945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3945_, 0, v_a_3939_);
v___x_3944_ = v_reuseFailAlloc_3945_;
goto v_reusejp_3943_;
}
v_reusejp_3943_:
{
return v___x_3944_;
}
}
}
}
v___jp_3947_:
{
lean_object* v___x_3948_; lean_object* v___x_3949_; 
v___x_3948_ = ((lean_object*)(lp_aesop_Aesop_expandGoal___redArg___closed__2));
v___x_3949_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3949_, 0, v___x_3948_);
return v___x_3949_;
}
v___jp_3964_:
{
lean_object* v___x_3982_; lean_object* v___x_3983_; double v___x_3984_; double v___x_3985_; double v___x_3986_; double v___x_3987_; double v___x_3988_; lean_object* v___x_3989_; lean_object* v___x_3990_; lean_object* v___x_3991_; lean_object* v___x_3992_; lean_object* v___x_51549__overap_3993_; lean_object* v___x_3994_; 
v___x_3982_ = lean_st_ref_get(v___y_3970_);
lean_dec(v___x_3982_);
v___x_3983_ = lean_io_mono_nanos_now();
v___x_3984_ = lean_float_of_nat(v___y_3969_);
v___x_3985_ = lean_float_once(&lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26, &lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26_once, _init_lp_aesop_Aesop_withRuleTraceNode___redArg___closed__26);
v___x_3986_ = lean_float_div(v___x_3984_, v___x_3985_);
v___x_3987_ = lean_float_of_nat(v___x_3983_);
v___x_3988_ = lean_float_div(v___x_3987_, v___x_3985_);
v___x_3989_ = lean_box_float(v___x_3986_);
v___x_3990_ = lean_box_float(v___x_3988_);
v___x_3991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3991_, 0, v___x_3989_);
lean_ctor_set(v___x_3991_, 1, v___x_3990_);
v___x_3992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3992_, 0, v_a_3981_);
lean_ctor_set(v___x_3992_, 1, v___x_3991_);
lean_inc_ref(v___y_3968_);
lean_inc_ref(v___y_3972_);
v___x_51549__overap_3993_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3951_, v___x_3952_, v___x_3953_, v___f_3961_, lean_box(0), v___x_3956_, v___y_3972_, v___y_3978_, v___x_3963_, v___y_3968_, v___y_3973_, v___y_3979_, v___y_3980_, v___y_3975_, v___x_3992_);
lean_inc(v___y_3966_);
lean_inc_ref(v___y_3967_);
lean_inc(v___y_3976_);
lean_inc_ref(v___y_3965_);
lean_inc(v___y_3977_);
lean_inc(v___y_3974_);
lean_inc(v___y_3970_);
lean_inc_ref(v___y_3971_);
v___x_3994_ = lean_apply_9(v___x_51549__overap_3993_, v___y_3971_, v___y_3970_, v___y_3974_, v___y_3977_, v___y_3965_, v___y_3976_, v___y_3967_, v___y_3966_, lean_box(0));
v___y_3899_ = v___y_3965_;
v___y_3900_ = v___y_3966_;
v___y_3901_ = v___y_3974_;
v___y_3902_ = v___y_3967_;
v___y_3903_ = v___y_3976_;
v___y_3904_ = v___y_3977_;
v___y_3905_ = v___y_3970_;
v___y_3906_ = v___y_3971_;
v___y_3907_ = v___x_3994_;
goto v___jp_3898_;
}
v___jp_3995_:
{
lean_object* v___x_4013_; lean_object* v___x_4014_; double v___x_4015_; double v___x_4016_; lean_object* v___x_4017_; lean_object* v___x_4018_; lean_object* v___x_4019_; lean_object* v___x_4020_; lean_object* v___x_51576__overap_4021_; lean_object* v___x_4022_; 
v___x_4013_ = lean_st_ref_get(v___y_4000_);
lean_dec(v___x_4013_);
v___x_4014_ = lean_io_get_num_heartbeats();
v___x_4015_ = lean_float_of_nat(v___y_4004_);
v___x_4016_ = lean_float_of_nat(v___x_4014_);
v___x_4017_ = lean_box_float(v___x_4015_);
v___x_4018_ = lean_box_float(v___x_4016_);
v___x_4019_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4019_, 0, v___x_4017_);
lean_ctor_set(v___x_4019_, 1, v___x_4018_);
v___x_4020_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4020_, 0, v_a_4012_);
lean_ctor_set(v___x_4020_, 1, v___x_4019_);
lean_inc_ref(v___y_3999_);
lean_inc_ref(v___y_4002_);
v___x_51576__overap_4021_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3951_, v___x_3952_, v___x_3953_, v___f_3961_, lean_box(0), v___x_3956_, v___y_4002_, v___y_4009_, v___x_3963_, v___y_3999_, v___y_4003_, v___y_4010_, v___y_4011_, v___y_4006_, v___x_4020_);
lean_inc(v___y_3997_);
lean_inc_ref(v___y_3998_);
lean_inc(v___y_4007_);
lean_inc_ref(v___y_3996_);
lean_inc(v___y_4008_);
lean_inc(v___y_4005_);
lean_inc(v___y_4000_);
lean_inc_ref(v___y_4001_);
v___x_4022_ = lean_apply_9(v___x_51576__overap_4021_, v___y_4001_, v___y_4000_, v___y_4005_, v___y_4008_, v___y_3996_, v___y_4007_, v___y_3998_, v___y_3997_, lean_box(0));
v___y_3899_ = v___y_3996_;
v___y_3900_ = v___y_3997_;
v___y_3901_ = v___y_4005_;
v___y_3902_ = v___y_3998_;
v___y_3903_ = v___y_4007_;
v___y_3904_ = v___y_4008_;
v___y_3905_ = v___y_4000_;
v___y_3906_ = v___y_4001_;
v___y_3907_ = v___x_4022_;
goto v___jp_3898_;
}
v___jp_4023_:
{
lean_object* v___x_51520__overap_4038_; lean_object* v___x_4039_; 
lean_inc_ref(v___x_3951_);
v___x_51520__overap_4038_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_3951_, v___x_3952_);
lean_inc(v___y_4025_);
lean_inc_ref(v___y_4026_);
lean_inc(v___y_4035_);
lean_inc_ref(v___y_4024_);
lean_inc(v___y_4034_);
lean_inc(v___y_4032_);
lean_inc(v___y_4028_);
lean_inc_ref(v___y_4029_);
v___x_4039_ = lean_apply_9(v___x_51520__overap_4038_, v___y_4029_, v___y_4028_, v___y_4032_, v___y_4034_, v___y_4024_, v___y_4035_, v___y_4026_, v___y_4025_, lean_box(0));
if (lean_obj_tag(v___x_4039_) == 0)
{
lean_object* v_a_4040_; lean_object* v___x_4041_; lean_object* v___x_4042_; lean_object* v___x_4043_; uint8_t v___x_4044_; 
v_a_4040_ = lean_ctor_get(v___x_4039_, 0);
lean_inc(v_a_4040_);
lean_dec_ref_known(v___x_4039_, 1);
v___x_4041_ = l_Lean_KVMap_instValueBool;
v___x_4042_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4043_ = l_Lean_Option_get___redArg(v___x_4041_, v___y_4030_, v___x_4042_);
v___x_4044_ = lean_unbox(v___x_4043_);
lean_dec(v___x_4043_);
if (v___x_4044_ == 0)
{
lean_object* v___x_4045_; lean_object* v___x_4046_; lean_object* v___x_4047_; 
v___x_4045_ = lean_st_ref_get(v___y_4028_);
lean_dec(v___x_4045_);
v___x_4046_ = lean_io_mono_nanos_now();
lean_inc(v_gref_3888_);
lean_inc_ref(v_inst_3887_);
v___x_4047_ = lp_aesop_Aesop_runFirstSafeRule___redArg(v_inst_3887_, v_gref_3888_, v___y_4029_, v___y_4028_, v___y_4032_, v___y_4034_, v___y_4024_, v___y_4035_, v___y_4026_, v___y_4025_);
if (lean_obj_tag(v___x_4047_) == 0)
{
lean_object* v_a_4048_; lean_object* v___x_4050_; uint8_t v_isShared_4051_; uint8_t v_isSharedCheck_4055_; 
v_a_4048_ = lean_ctor_get(v___x_4047_, 0);
v_isSharedCheck_4055_ = !lean_is_exclusive(v___x_4047_);
if (v_isSharedCheck_4055_ == 0)
{
v___x_4050_ = v___x_4047_;
v_isShared_4051_ = v_isSharedCheck_4055_;
goto v_resetjp_4049_;
}
else
{
lean_inc(v_a_4048_);
lean_dec(v___x_4047_);
v___x_4050_ = lean_box(0);
v_isShared_4051_ = v_isSharedCheck_4055_;
goto v_resetjp_4049_;
}
v_resetjp_4049_:
{
lean_object* v___x_4053_; 
if (v_isShared_4051_ == 0)
{
lean_ctor_set_tag(v___x_4050_, 1);
v___x_4053_ = v___x_4050_;
goto v_reusejp_4052_;
}
else
{
lean_object* v_reuseFailAlloc_4054_; 
v_reuseFailAlloc_4054_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4054_, 0, v_a_4048_);
v___x_4053_ = v_reuseFailAlloc_4054_;
goto v_reusejp_4052_;
}
v_reusejp_4052_:
{
v___y_3965_ = v___y_4024_;
v___y_3966_ = v___y_4025_;
v___y_3967_ = v___y_4026_;
v___y_3968_ = v___y_4027_;
v___y_3969_ = v___x_4046_;
v___y_3970_ = v___y_4028_;
v___y_3971_ = v___y_4029_;
v___y_3972_ = v___y_4031_;
v___y_3973_ = v___y_4030_;
v___y_3974_ = v___y_4032_;
v___y_3975_ = v___y_4033_;
v___y_3976_ = v___y_4035_;
v___y_3977_ = v___y_4034_;
v___y_3978_ = v___y_4037_;
v___y_3979_ = v___y_4036_;
v___y_3980_ = v_a_4040_;
v_a_3981_ = v___x_4053_;
goto v___jp_3964_;
}
}
}
else
{
lean_object* v_a_4056_; lean_object* v___x_4058_; uint8_t v_isShared_4059_; uint8_t v_isSharedCheck_4063_; 
v_a_4056_ = lean_ctor_get(v___x_4047_, 0);
v_isSharedCheck_4063_ = !lean_is_exclusive(v___x_4047_);
if (v_isSharedCheck_4063_ == 0)
{
v___x_4058_ = v___x_4047_;
v_isShared_4059_ = v_isSharedCheck_4063_;
goto v_resetjp_4057_;
}
else
{
lean_inc(v_a_4056_);
lean_dec(v___x_4047_);
v___x_4058_ = lean_box(0);
v_isShared_4059_ = v_isSharedCheck_4063_;
goto v_resetjp_4057_;
}
v_resetjp_4057_:
{
lean_object* v___x_4061_; 
if (v_isShared_4059_ == 0)
{
lean_ctor_set_tag(v___x_4058_, 0);
v___x_4061_ = v___x_4058_;
goto v_reusejp_4060_;
}
else
{
lean_object* v_reuseFailAlloc_4062_; 
v_reuseFailAlloc_4062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4062_, 0, v_a_4056_);
v___x_4061_ = v_reuseFailAlloc_4062_;
goto v_reusejp_4060_;
}
v_reusejp_4060_:
{
v___y_3965_ = v___y_4024_;
v___y_3966_ = v___y_4025_;
v___y_3967_ = v___y_4026_;
v___y_3968_ = v___y_4027_;
v___y_3969_ = v___x_4046_;
v___y_3970_ = v___y_4028_;
v___y_3971_ = v___y_4029_;
v___y_3972_ = v___y_4031_;
v___y_3973_ = v___y_4030_;
v___y_3974_ = v___y_4032_;
v___y_3975_ = v___y_4033_;
v___y_3976_ = v___y_4035_;
v___y_3977_ = v___y_4034_;
v___y_3978_ = v___y_4037_;
v___y_3979_ = v___y_4036_;
v___y_3980_ = v_a_4040_;
v_a_3981_ = v___x_4061_;
goto v___jp_3964_;
}
}
}
}
else
{
lean_object* v___x_4064_; lean_object* v___x_4065_; lean_object* v___x_4066_; 
v___x_4064_ = lean_st_ref_get(v___y_4028_);
lean_dec(v___x_4064_);
v___x_4065_ = lean_io_get_num_heartbeats();
lean_inc(v_gref_3888_);
lean_inc_ref(v_inst_3887_);
v___x_4066_ = lp_aesop_Aesop_runFirstSafeRule___redArg(v_inst_3887_, v_gref_3888_, v___y_4029_, v___y_4028_, v___y_4032_, v___y_4034_, v___y_4024_, v___y_4035_, v___y_4026_, v___y_4025_);
if (lean_obj_tag(v___x_4066_) == 0)
{
lean_object* v_a_4067_; lean_object* v___x_4069_; uint8_t v_isShared_4070_; uint8_t v_isSharedCheck_4074_; 
v_a_4067_ = lean_ctor_get(v___x_4066_, 0);
v_isSharedCheck_4074_ = !lean_is_exclusive(v___x_4066_);
if (v_isSharedCheck_4074_ == 0)
{
v___x_4069_ = v___x_4066_;
v_isShared_4070_ = v_isSharedCheck_4074_;
goto v_resetjp_4068_;
}
else
{
lean_inc(v_a_4067_);
lean_dec(v___x_4066_);
v___x_4069_ = lean_box(0);
v_isShared_4070_ = v_isSharedCheck_4074_;
goto v_resetjp_4068_;
}
v_resetjp_4068_:
{
lean_object* v___x_4072_; 
if (v_isShared_4070_ == 0)
{
lean_ctor_set_tag(v___x_4069_, 1);
v___x_4072_ = v___x_4069_;
goto v_reusejp_4071_;
}
else
{
lean_object* v_reuseFailAlloc_4073_; 
v_reuseFailAlloc_4073_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4073_, 0, v_a_4067_);
v___x_4072_ = v_reuseFailAlloc_4073_;
goto v_reusejp_4071_;
}
v_reusejp_4071_:
{
v___y_3996_ = v___y_4024_;
v___y_3997_ = v___y_4025_;
v___y_3998_ = v___y_4026_;
v___y_3999_ = v___y_4027_;
v___y_4000_ = v___y_4028_;
v___y_4001_ = v___y_4029_;
v___y_4002_ = v___y_4031_;
v___y_4003_ = v___y_4030_;
v___y_4004_ = v___x_4065_;
v___y_4005_ = v___y_4032_;
v___y_4006_ = v___y_4033_;
v___y_4007_ = v___y_4035_;
v___y_4008_ = v___y_4034_;
v___y_4009_ = v___y_4037_;
v___y_4010_ = v___y_4036_;
v___y_4011_ = v_a_4040_;
v_a_4012_ = v___x_4072_;
goto v___jp_3995_;
}
}
}
else
{
lean_object* v_a_4075_; lean_object* v___x_4077_; uint8_t v_isShared_4078_; uint8_t v_isSharedCheck_4082_; 
v_a_4075_ = lean_ctor_get(v___x_4066_, 0);
v_isSharedCheck_4082_ = !lean_is_exclusive(v___x_4066_);
if (v_isSharedCheck_4082_ == 0)
{
v___x_4077_ = v___x_4066_;
v_isShared_4078_ = v_isSharedCheck_4082_;
goto v_resetjp_4076_;
}
else
{
lean_inc(v_a_4075_);
lean_dec(v___x_4066_);
v___x_4077_ = lean_box(0);
v_isShared_4078_ = v_isSharedCheck_4082_;
goto v_resetjp_4076_;
}
v_resetjp_4076_:
{
lean_object* v___x_4080_; 
if (v_isShared_4078_ == 0)
{
lean_ctor_set_tag(v___x_4077_, 0);
v___x_4080_ = v___x_4077_;
goto v_reusejp_4079_;
}
else
{
lean_object* v_reuseFailAlloc_4081_; 
v_reuseFailAlloc_4081_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4081_, 0, v_a_4075_);
v___x_4080_ = v_reuseFailAlloc_4081_;
goto v_reusejp_4079_;
}
v_reusejp_4079_:
{
v___y_3996_ = v___y_4024_;
v___y_3997_ = v___y_4025_;
v___y_3998_ = v___y_4026_;
v___y_3999_ = v___y_4027_;
v___y_4000_ = v___y_4028_;
v___y_4001_ = v___y_4029_;
v___y_4002_ = v___y_4031_;
v___y_4003_ = v___y_4030_;
v___y_4004_ = v___x_4065_;
v___y_4005_ = v___y_4032_;
v___y_4006_ = v___y_4033_;
v___y_4007_ = v___y_4035_;
v___y_4008_ = v___y_4034_;
v___y_4009_ = v___y_4037_;
v___y_4010_ = v___y_4036_;
v___y_4011_ = v_a_4040_;
v_a_4012_ = v___x_4080_;
goto v___jp_3995_;
}
}
}
}
}
else
{
lean_object* v_a_4083_; lean_object* v___x_4085_; uint8_t v_isShared_4086_; uint8_t v_isSharedCheck_4090_; 
lean_dec(v___y_4037_);
lean_dec_ref(v___y_4033_);
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_a_4083_ = lean_ctor_get(v___x_4039_, 0);
v_isSharedCheck_4090_ = !lean_is_exclusive(v___x_4039_);
if (v_isSharedCheck_4090_ == 0)
{
v___x_4085_ = v___x_4039_;
v_isShared_4086_ = v_isSharedCheck_4090_;
goto v_resetjp_4084_;
}
else
{
lean_inc(v_a_4083_);
lean_dec(v___x_4039_);
v___x_4085_ = lean_box(0);
v_isShared_4086_ = v_isSharedCheck_4090_;
goto v_resetjp_4084_;
}
v_resetjp_4084_:
{
lean_object* v___x_4088_; 
if (v_isShared_4086_ == 0)
{
v___x_4088_ = v___x_4085_;
goto v_reusejp_4087_;
}
else
{
lean_object* v_reuseFailAlloc_4089_; 
v_reuseFailAlloc_4089_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4089_, 0, v_a_4083_);
v___x_4088_ = v_reuseFailAlloc_4089_;
goto v_reusejp_4087_;
}
v_reusejp_4087_:
{
return v___x_4088_;
}
}
}
}
v___jp_4091_:
{
if (v_hasTrace_4100_ == 0)
{
lean_object* v___x_4103_; 
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_inc(v_gref_3888_);
lean_inc_ref(v_inst_3887_);
v___x_4103_ = lp_aesop_Aesop_runFirstSafeRule___redArg(v_inst_3887_, v_gref_3888_, v___y_4092_, v___y_4093_, v___y_4094_, v___y_4095_, v___y_4096_, v___y_4097_, v___y_4098_, v___y_4102_);
v___y_3899_ = v___y_4096_;
v___y_3900_ = v___y_4102_;
v___y_3901_ = v___y_4094_;
v___y_3902_ = v___y_4098_;
v___y_3903_ = v___y_4097_;
v___y_3904_ = v___y_4095_;
v___y_3905_ = v___y_4093_;
v___y_3906_ = v___y_4092_;
v___y_3907_ = v___x_4103_;
goto v___jp_3898_;
}
else
{
lean_object* v___x_4104_; lean_object* v_traceClass_4105_; lean_object* v___f_4106_; lean_object* v___x_4107_; lean_object* v___x_4108_; lean_object* v___x_4109_; lean_object* v___x_4110_; uint8_t v___x_4111_; 
v___x_4104_ = lean_st_ref_get(v___y_4093_);
lean_dec(v___x_4104_);
v_traceClass_4105_ = lean_ctor_get(v___x_3962_, 0);
v___f_4106_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__22));
lean_inc_ref(v_inst_3887_);
v___x_4107_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Expansion_0__Aesop_expandGoal_fmtSafe___boxed), 12, 2);
lean_closure_set(v___x_4107_, 0, lean_box(0));
lean_closure_set(v___x_4107_, 1, v_inst_3887_);
v___x_4108_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__23));
v___x_4109_ = ((lean_object*)(lp_aesop_Aesop_withRuleTraceNode___redArg___closed__25));
lean_inc(v_traceClass_4105_);
v___x_4110_ = l_Lean_Name_append(v___x_4109_, v_traceClass_4105_);
v___x_4111_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4101_, v_options_4099_, v___x_4110_);
lean_dec(v___x_4110_);
if (v___x_4111_ == 0)
{
lean_object* v___x_4112_; lean_object* v___x_4113_; lean_object* v___x_4114_; uint8_t v___x_4115_; 
v___x_4112_ = l_Lean_KVMap_instValueBool;
v___x_4113_ = l_Lean_trace_profiler;
v___x_4114_ = l_Lean_Option_get___redArg(v___x_4112_, v_options_4099_, v___x_4113_);
v___x_4115_ = lean_unbox(v___x_4114_);
lean_dec(v___x_4114_);
if (v___x_4115_ == 0)
{
lean_object* v___x_4116_; 
lean_dec_ref(v___x_4107_);
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_inc(v_gref_3888_);
lean_inc_ref(v_inst_3887_);
v___x_4116_ = lp_aesop_Aesop_runFirstSafeRule___redArg(v_inst_3887_, v_gref_3888_, v___y_4092_, v___y_4093_, v___y_4094_, v___y_4095_, v___y_4096_, v___y_4097_, v___y_4098_, v___y_4102_);
v___y_3899_ = v___y_4096_;
v___y_3900_ = v___y_4102_;
v___y_3901_ = v___y_4094_;
v___y_3902_ = v___y_4098_;
v___y_3903_ = v___y_4097_;
v___y_3904_ = v___y_4095_;
v___y_3905_ = v___y_4093_;
v___y_3906_ = v___y_4092_;
v___y_3907_ = v___x_4116_;
goto v___jp_3898_;
}
else
{
lean_inc(v_traceClass_4105_);
v___y_4024_ = v___y_4096_;
v___y_4025_ = v___y_4102_;
v___y_4026_ = v___y_4098_;
v___y_4027_ = v___x_4108_;
v___y_4028_ = v___y_4093_;
v___y_4029_ = v___y_4092_;
v___y_4030_ = v_options_4099_;
v___y_4031_ = v___f_4106_;
v___y_4032_ = v___y_4094_;
v___y_4033_ = v___x_4107_;
v___y_4034_ = v___y_4095_;
v___y_4035_ = v___y_4097_;
v___y_4036_ = v___x_4111_;
v___y_4037_ = v_traceClass_4105_;
goto v___jp_4023_;
}
}
else
{
lean_inc(v_traceClass_4105_);
v___y_4024_ = v___y_4096_;
v___y_4025_ = v___y_4102_;
v___y_4026_ = v___y_4098_;
v___y_4027_ = v___x_4108_;
v___y_4028_ = v___y_4093_;
v___y_4029_ = v___y_4092_;
v___y_4030_ = v_options_4099_;
v___y_4031_ = v___f_4106_;
v___y_4032_ = v___y_4094_;
v___y_4033_ = v___x_4107_;
v___y_4034_ = v___y_4095_;
v___y_4035_ = v___y_4097_;
v___y_4036_ = v___x_4111_;
v___y_4037_ = v_traceClass_4105_;
goto v___jp_4023_;
}
}
}
v___jp_4117_:
{
if (lean_obj_tag(v___y_4118_) == 0)
{
lean_object* v_a_4119_; lean_object* v___x_51442__overap_4120_; lean_object* v___x_4121_; 
v_a_4119_ = lean_ctor_get(v___y_4118_, 0);
lean_inc(v_a_4119_);
lean_dec_ref_known(v___y_4118_, 1);
lean_inc(v_toMonadOptions_3955_);
lean_inc_ref(v___x_3951_);
v___x_51442__overap_4120_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_3951_, v_toMonadOptions_3955_, v___x_3962_);
lean_inc(v_a_3896_);
lean_inc_ref(v_a_3895_);
lean_inc(v_a_3894_);
lean_inc_ref(v_a_3893_);
lean_inc(v_a_3892_);
lean_inc(v_a_3891_);
lean_inc(v_a_3890_);
lean_inc_ref(v_a_3889_);
v___x_4121_ = lean_apply_9(v___x_51442__overap_4120_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_, lean_box(0));
if (lean_obj_tag(v___x_4121_) == 0)
{
lean_object* v_a_4122_; uint8_t v___x_4123_; 
v_a_4122_ = lean_ctor_get(v___x_4121_, 0);
lean_inc(v_a_4122_);
lean_dec_ref_known(v___x_4121_, 1);
v___x_4123_ = lean_unbox(v_a_4122_);
lean_dec(v_a_4122_);
if (v___x_4123_ == 0)
{
uint8_t v___x_4124_; 
v___x_4124_ = lean_unbox(v_a_4119_);
lean_dec(v_a_4119_);
if (v___x_4124_ == 0)
{
v___y_4092_ = v_a_3889_;
v___y_4093_ = v_a_3890_;
v___y_4094_ = v_a_3891_;
v___y_4095_ = v_a_3892_;
v___y_4096_ = v_a_3893_;
v___y_4097_ = v_a_3894_;
v___y_4098_ = v_a_3895_;
v_options_4099_ = v_options_3957_;
v_hasTrace_4100_ = v_hasTrace_3959_;
v_inheritedTraceOptions_4101_ = v_inheritedTraceOptions_3958_;
v___y_4102_ = v_a_3896_;
goto v___jp_4091_;
}
else
{
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
goto v___jp_3947_;
}
}
else
{
uint8_t v___x_4125_; 
v___x_4125_ = lean_unbox(v_a_4119_);
lean_dec(v_a_4119_);
if (v___x_4125_ == 0)
{
lean_object* v___x_4126_; lean_object* v___x_4127_; lean_object* v___x_4128_; lean_object* v___x_4129_; 
v___x_4126_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4126_);
v___x_4127_ = lean_st_ref_get(v_gref_3888_);
v___x_4128_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4128_);
v___x_4129_ = lp_aesop_Aesop_getRootMetaState___redArg(v_a_3891_);
if (lean_obj_tag(v___x_4129_) == 0)
{
lean_object* v_a_4130_; lean_object* v___x_4131_; lean_object* v___x_4132_; 
v_a_4130_ = lean_ctor_get(v___x_4129_, 0);
lean_inc(v_a_4130_);
lean_dec_ref_known(v___x_4129_, 1);
v___x_4131_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4131_);
v___x_4132_ = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(v___x_4127_, v_a_4130_);
lean_dec(v_a_4130_);
if (lean_obj_tag(v___x_4132_) == 0)
{
lean_object* v_a_4133_; lean_object* v_fst_4134_; lean_object* v_snd_4135_; lean_object* v___x_4137_; uint8_t v_isShared_4138_; uint8_t v_isSharedCheck_4205_; 
v_a_4133_ = lean_ctor_get(v___x_4132_, 0);
lean_inc(v_a_4133_);
lean_dec_ref_known(v___x_4132_, 1);
v_fst_4134_ = lean_ctor_get(v_a_4133_, 0);
v_snd_4135_ = lean_ctor_get(v_a_4133_, 1);
v_isSharedCheck_4205_ = !lean_is_exclusive(v_a_4133_);
if (v_isSharedCheck_4205_ == 0)
{
v___x_4137_ = v_a_4133_;
v_isShared_4138_ = v_isSharedCheck_4205_;
goto v_resetjp_4136_;
}
else
{
lean_inc(v_snd_4135_);
lean_inc(v_fst_4134_);
lean_dec(v_a_4133_);
v___x_4137_ = lean_box(0);
v_isShared_4138_ = v_isSharedCheck_4205_;
goto v_resetjp_4136_;
}
v_resetjp_4136_:
{
lean_object* v___x_4139_; lean_object* v_toApplicative_4140_; lean_object* v_toFunctor_4141_; lean_object* v_toSeq_4142_; lean_object* v_toSeqLeft_4143_; lean_object* v_toSeqRight_4144_; lean_object* v___f_4145_; lean_object* v___f_4146_; lean_object* v___f_4147_; lean_object* v___f_4148_; lean_object* v___x_4150_; 
v___x_4139_ = lean_obj_once(&lp_aesop_Aesop_runSafeRule___redArg___closed__1, &lp_aesop_Aesop_runSafeRule___redArg___closed__1_once, _init_lp_aesop_Aesop_runSafeRule___redArg___closed__1);
v_toApplicative_4140_ = lean_ctor_get(v___x_4139_, 0);
v_toFunctor_4141_ = lean_ctor_get(v_toApplicative_4140_, 0);
v_toSeq_4142_ = lean_ctor_get(v_toApplicative_4140_, 2);
v_toSeqLeft_4143_ = lean_ctor_get(v_toApplicative_4140_, 3);
v_toSeqRight_4144_ = lean_ctor_get(v_toApplicative_4140_, 4);
v___f_4145_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__2));
v___f_4146_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_4141_, 2);
v___f_4147_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4147_, 0, v_toFunctor_4141_);
v___f_4148_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4148_, 0, v_toFunctor_4141_);
if (v_isShared_4138_ == 0)
{
lean_ctor_set(v___x_4137_, 1, v___f_4148_);
lean_ctor_set(v___x_4137_, 0, v___f_4147_);
v___x_4150_ = v___x_4137_;
goto v_reusejp_4149_;
}
else
{
lean_object* v_reuseFailAlloc_4204_; 
v_reuseFailAlloc_4204_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4204_, 0, v___f_4147_);
lean_ctor_set(v_reuseFailAlloc_4204_, 1, v___f_4148_);
v___x_4150_ = v_reuseFailAlloc_4204_;
goto v_reusejp_4149_;
}
v_reusejp_4149_:
{
lean_object* v___f_4151_; lean_object* v___f_4152_; lean_object* v___f_4153_; lean_object* v___x_4154_; lean_object* v___x_4155_; lean_object* v___x_4156_; lean_object* v_toApplicative_4157_; lean_object* v___x_4159_; uint8_t v_isShared_4160_; uint8_t v_isSharedCheck_4202_; 
lean_inc(v_toSeqRight_4144_);
v___f_4151_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4151_, 0, v_toSeqRight_4144_);
lean_inc(v_toSeqLeft_4143_);
v___f_4152_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4152_, 0, v_toSeqLeft_4143_);
lean_inc(v_toSeq_4142_);
v___f_4153_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4153_, 0, v_toSeq_4142_);
v___x_4154_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4154_, 0, v___x_4150_);
lean_ctor_set(v___x_4154_, 1, v___f_4145_);
lean_ctor_set(v___x_4154_, 2, v___f_4153_);
lean_ctor_set(v___x_4154_, 3, v___f_4152_);
lean_ctor_set(v___x_4154_, 4, v___f_4151_);
v___x_4155_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4155_, 0, v___x_4154_);
lean_ctor_set(v___x_4155_, 1, v___f_4146_);
v___x_4156_ = l_StateRefT_x27_instMonad___redArg(v___x_4155_);
v_toApplicative_4157_ = lean_ctor_get(v___x_4156_, 0);
v_isSharedCheck_4202_ = !lean_is_exclusive(v___x_4156_);
if (v_isSharedCheck_4202_ == 0)
{
lean_object* v_unused_4203_; 
v_unused_4203_ = lean_ctor_get(v___x_4156_, 1);
lean_dec(v_unused_4203_);
v___x_4159_ = v___x_4156_;
v_isShared_4160_ = v_isSharedCheck_4202_;
goto v_resetjp_4158_;
}
else
{
lean_inc(v_toApplicative_4157_);
lean_dec(v___x_4156_);
v___x_4159_ = lean_box(0);
v_isShared_4160_ = v_isSharedCheck_4202_;
goto v_resetjp_4158_;
}
v_resetjp_4158_:
{
lean_object* v_toFunctor_4161_; lean_object* v_toSeq_4162_; lean_object* v_toSeqLeft_4163_; lean_object* v_toSeqRight_4164_; lean_object* v___x_4166_; uint8_t v_isShared_4167_; uint8_t v_isSharedCheck_4200_; 
v_toFunctor_4161_ = lean_ctor_get(v_toApplicative_4157_, 0);
v_toSeq_4162_ = lean_ctor_get(v_toApplicative_4157_, 2);
v_toSeqLeft_4163_ = lean_ctor_get(v_toApplicative_4157_, 3);
v_toSeqRight_4164_ = lean_ctor_get(v_toApplicative_4157_, 4);
v_isSharedCheck_4200_ = !lean_is_exclusive(v_toApplicative_4157_);
if (v_isSharedCheck_4200_ == 0)
{
lean_object* v_unused_4201_; 
v_unused_4201_ = lean_ctor_get(v_toApplicative_4157_, 1);
lean_dec(v_unused_4201_);
v___x_4166_ = v_toApplicative_4157_;
v_isShared_4167_ = v_isSharedCheck_4200_;
goto v_resetjp_4165_;
}
else
{
lean_inc(v_toSeqRight_4164_);
lean_inc(v_toSeqLeft_4163_);
lean_inc(v_toSeq_4162_);
lean_inc(v_toFunctor_4161_);
lean_dec(v_toApplicative_4157_);
v___x_4166_ = lean_box(0);
v_isShared_4167_ = v_isSharedCheck_4200_;
goto v_resetjp_4165_;
}
v_resetjp_4165_:
{
lean_object* v___f_4168_; lean_object* v___f_4169_; lean_object* v___f_4170_; lean_object* v___f_4171_; lean_object* v___x_4172_; lean_object* v___f_4173_; lean_object* v___f_4174_; lean_object* v___f_4175_; lean_object* v___x_4177_; 
v___f_4168_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__4));
v___f_4169_ = ((lean_object*)(lp_aesop_Aesop_runSafeRule___redArg___closed__5));
lean_inc_ref(v_toFunctor_4161_);
v___f_4170_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4170_, 0, v_toFunctor_4161_);
v___f_4171_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4171_, 0, v_toFunctor_4161_);
v___x_4172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4172_, 0, v___f_4170_);
lean_ctor_set(v___x_4172_, 1, v___f_4171_);
v___f_4173_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4173_, 0, v_toSeqRight_4164_);
v___f_4174_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4174_, 0, v_toSeqLeft_4163_);
v___f_4175_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4175_, 0, v_toSeq_4162_);
if (v_isShared_4167_ == 0)
{
lean_ctor_set(v___x_4166_, 4, v___f_4173_);
lean_ctor_set(v___x_4166_, 3, v___f_4174_);
lean_ctor_set(v___x_4166_, 2, v___f_4175_);
lean_ctor_set(v___x_4166_, 1, v___f_4168_);
lean_ctor_set(v___x_4166_, 0, v___x_4172_);
v___x_4177_ = v___x_4166_;
goto v_reusejp_4176_;
}
else
{
lean_object* v_reuseFailAlloc_4199_; 
v_reuseFailAlloc_4199_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4199_, 0, v___x_4172_);
lean_ctor_set(v_reuseFailAlloc_4199_, 1, v___f_4168_);
lean_ctor_set(v_reuseFailAlloc_4199_, 2, v___f_4175_);
lean_ctor_set(v_reuseFailAlloc_4199_, 3, v___f_4174_);
lean_ctor_set(v_reuseFailAlloc_4199_, 4, v___f_4173_);
v___x_4177_ = v_reuseFailAlloc_4199_;
goto v_reusejp_4176_;
}
v_reusejp_4176_:
{
lean_object* v___x_4179_; 
if (v_isShared_4160_ == 0)
{
lean_ctor_set(v___x_4159_, 1, v___f_4169_);
lean_ctor_set(v___x_4159_, 0, v___x_4177_);
v___x_4179_ = v___x_4159_;
goto v_reusejp_4178_;
}
else
{
lean_object* v_reuseFailAlloc_4198_; 
v_reuseFailAlloc_4198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4198_, 0, v___x_4177_);
lean_ctor_set(v_reuseFailAlloc_4198_, 1, v___f_4169_);
v___x_4179_ = v_reuseFailAlloc_4198_;
goto v_reusejp_4178_;
}
v_reusejp_4178_:
{
lean_object* v___x_4180_; lean_object* v_toMonadRef_4181_; lean_object* v___x_4182_; lean_object* v_traceClass_4183_; lean_object* v___x_4184_; lean_object* v___x_4185_; lean_object* v___x_4186_; lean_object* v___x_4187_; lean_object* v___x_4188_; lean_object* v___x_4189_; 
v___x_4180_ = lean_obj_once(&lp_aesop_Aesop_expandGoal___redArg___closed__6, &lp_aesop_Aesop_expandGoal___redArg___closed__6_once, _init_lp_aesop_Aesop_expandGoal___redArg___closed__6);
v_toMonadRef_4181_ = lean_ctor_get(v___x_4180_, 0);
v___x_4182_ = lean_st_ref_get(v_a_3890_);
lean_dec(v___x_4182_);
v_traceClass_4183_ = lean_ctor_get(v___x_3962_, 0);
v___x_4184_ = lean_obj_once(&lp_aesop_Aesop_expandGoal___redArg___closed__8, &lp_aesop_Aesop_expandGoal___redArg___closed__8_once, _init_lp_aesop_Aesop_expandGoal___redArg___closed__8);
v___x_4185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4185_, 0, v_fst_4134_);
v___x_4186_ = l_Lean_indentD(v___x_4185_);
v___x_4187_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4187_, 0, v___x_4184_);
lean_ctor_set(v___x_4187_, 1, v___x_4186_);
lean_inc(v_traceClass_4183_);
lean_inc_ref(v_toMonadRef_4181_);
v___x_4188_ = l_Lean_addTrace___redArg(v___x_4179_, v___x_3950_, v_toMonadRef_4181_, v___x_3960_, v_traceClass_4183_, v___x_4187_);
v___x_4189_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_snd_4135_, v___x_4188_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_);
if (lean_obj_tag(v___x_4189_) == 0)
{
lean_dec_ref_known(v___x_4189_, 1);
v___y_4092_ = v_a_3889_;
v___y_4093_ = v_a_3890_;
v___y_4094_ = v_a_3891_;
v___y_4095_ = v_a_3892_;
v___y_4096_ = v_a_3893_;
v___y_4097_ = v_a_3894_;
v___y_4098_ = v_a_3895_;
v_options_4099_ = v_options_3957_;
v_hasTrace_4100_ = v_hasTrace_3959_;
v_inheritedTraceOptions_4101_ = v_inheritedTraceOptions_3958_;
v___y_4102_ = v_a_3896_;
goto v___jp_4091_;
}
else
{
lean_object* v_a_4190_; lean_object* v___x_4192_; uint8_t v_isShared_4193_; uint8_t v_isSharedCheck_4197_; 
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_a_4190_ = lean_ctor_get(v___x_4189_, 0);
v_isSharedCheck_4197_ = !lean_is_exclusive(v___x_4189_);
if (v_isSharedCheck_4197_ == 0)
{
v___x_4192_ = v___x_4189_;
v_isShared_4193_ = v_isSharedCheck_4197_;
goto v_resetjp_4191_;
}
else
{
lean_inc(v_a_4190_);
lean_dec(v___x_4189_);
v___x_4192_ = lean_box(0);
v_isShared_4193_ = v_isSharedCheck_4197_;
goto v_resetjp_4191_;
}
v_resetjp_4191_:
{
lean_object* v___x_4195_; 
if (v_isShared_4193_ == 0)
{
v___x_4195_ = v___x_4192_;
goto v_reusejp_4194_;
}
else
{
lean_object* v_reuseFailAlloc_4196_; 
v_reuseFailAlloc_4196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4196_, 0, v_a_4190_);
v___x_4195_ = v_reuseFailAlloc_4196_;
goto v_reusejp_4194_;
}
v_reusejp_4194_:
{
return v___x_4195_;
}
}
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_4206_; lean_object* v___x_4208_; uint8_t v_isShared_4209_; uint8_t v_isSharedCheck_4213_; 
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_a_4206_ = lean_ctor_get(v___x_4132_, 0);
v_isSharedCheck_4213_ = !lean_is_exclusive(v___x_4132_);
if (v_isSharedCheck_4213_ == 0)
{
v___x_4208_ = v___x_4132_;
v_isShared_4209_ = v_isSharedCheck_4213_;
goto v_resetjp_4207_;
}
else
{
lean_inc(v_a_4206_);
lean_dec(v___x_4132_);
v___x_4208_ = lean_box(0);
v_isShared_4209_ = v_isSharedCheck_4213_;
goto v_resetjp_4207_;
}
v_resetjp_4207_:
{
lean_object* v___x_4211_; 
if (v_isShared_4209_ == 0)
{
v___x_4211_ = v___x_4208_;
goto v_reusejp_4210_;
}
else
{
lean_object* v_reuseFailAlloc_4212_; 
v_reuseFailAlloc_4212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4212_, 0, v_a_4206_);
v___x_4211_ = v_reuseFailAlloc_4212_;
goto v_reusejp_4210_;
}
v_reusejp_4210_:
{
return v___x_4211_;
}
}
}
}
else
{
lean_object* v_a_4214_; lean_object* v___x_4216_; uint8_t v_isShared_4217_; uint8_t v_isSharedCheck_4221_; 
lean_dec(v___x_4127_);
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_a_4214_ = lean_ctor_get(v___x_4129_, 0);
v_isSharedCheck_4221_ = !lean_is_exclusive(v___x_4129_);
if (v_isSharedCheck_4221_ == 0)
{
v___x_4216_ = v___x_4129_;
v_isShared_4217_ = v_isSharedCheck_4221_;
goto v_resetjp_4215_;
}
else
{
lean_inc(v_a_4214_);
lean_dec(v___x_4129_);
v___x_4216_ = lean_box(0);
v_isShared_4217_ = v_isSharedCheck_4221_;
goto v_resetjp_4215_;
}
v_resetjp_4215_:
{
lean_object* v___x_4219_; 
if (v_isShared_4217_ == 0)
{
v___x_4219_ = v___x_4216_;
goto v_reusejp_4218_;
}
else
{
lean_object* v_reuseFailAlloc_4220_; 
v_reuseFailAlloc_4220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4220_, 0, v_a_4214_);
v___x_4219_ = v_reuseFailAlloc_4220_;
goto v_reusejp_4218_;
}
v_reusejp_4218_:
{
return v___x_4219_;
}
}
}
}
else
{
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
goto v___jp_3947_;
}
}
}
else
{
lean_object* v_a_4222_; lean_object* v___x_4224_; uint8_t v_isShared_4225_; uint8_t v_isSharedCheck_4229_; 
lean_dec(v_a_4119_);
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_a_4222_ = lean_ctor_get(v___x_4121_, 0);
v_isSharedCheck_4229_ = !lean_is_exclusive(v___x_4121_);
if (v_isSharedCheck_4229_ == 0)
{
v___x_4224_ = v___x_4121_;
v_isShared_4225_ = v_isSharedCheck_4229_;
goto v_resetjp_4223_;
}
else
{
lean_inc(v_a_4222_);
lean_dec(v___x_4121_);
v___x_4224_ = lean_box(0);
v_isShared_4225_ = v_isSharedCheck_4229_;
goto v_resetjp_4223_;
}
v_resetjp_4223_:
{
lean_object* v___x_4227_; 
if (v_isShared_4225_ == 0)
{
v___x_4227_ = v___x_4224_;
goto v_reusejp_4226_;
}
else
{
lean_object* v_reuseFailAlloc_4228_; 
v_reuseFailAlloc_4228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4228_, 0, v_a_4222_);
v___x_4227_ = v_reuseFailAlloc_4228_;
goto v_reusejp_4226_;
}
v_reusejp_4226_:
{
return v___x_4227_;
}
}
}
}
else
{
lean_object* v_a_4230_; lean_object* v___x_4232_; uint8_t v_isShared_4233_; uint8_t v_isSharedCheck_4237_; 
lean_dec_ref(v___x_3953_);
lean_dec_ref(v___x_3951_);
lean_dec(v_gref_3888_);
lean_dec_ref(v_inst_3887_);
v_a_4230_ = lean_ctor_get(v___y_4118_, 0);
v_isSharedCheck_4237_ = !lean_is_exclusive(v___y_4118_);
if (v_isSharedCheck_4237_ == 0)
{
v___x_4232_ = v___y_4118_;
v_isShared_4233_ = v_isSharedCheck_4237_;
goto v_resetjp_4231_;
}
else
{
lean_inc(v_a_4230_);
lean_dec(v___y_4118_);
v___x_4232_ = lean_box(0);
v_isShared_4233_ = v_isSharedCheck_4237_;
goto v_resetjp_4231_;
}
v_resetjp_4231_:
{
lean_object* v___x_4235_; 
if (v_isShared_4233_ == 0)
{
v___x_4235_ = v___x_4232_;
goto v_reusejp_4234_;
}
else
{
lean_object* v_reuseFailAlloc_4236_; 
v_reuseFailAlloc_4236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4236_, 0, v_a_4230_);
v___x_4235_ = v_reuseFailAlloc_4236_;
goto v_reusejp_4234_;
}
v_reusejp_4234_:
{
return v___x_4235_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandGoal___redArg___boxed(lean_object* v_inst_4337_, lean_object* v_gref_4338_, lean_object* v_a_4339_, lean_object* v_a_4340_, lean_object* v_a_4341_, lean_object* v_a_4342_, lean_object* v_a_4343_, lean_object* v_a_4344_, lean_object* v_a_4345_, lean_object* v_a_4346_, lean_object* v_a_4347_){
_start:
{
lean_object* v_res_4348_; 
v_res_4348_ = lp_aesop_Aesop_expandGoal___redArg(v_inst_4337_, v_gref_4338_, v_a_4339_, v_a_4340_, v_a_4341_, v_a_4342_, v_a_4343_, v_a_4344_, v_a_4345_, v_a_4346_);
lean_dec(v_a_4346_);
lean_dec_ref(v_a_4345_);
lean_dec(v_a_4344_);
lean_dec_ref(v_a_4343_);
lean_dec(v_a_4342_);
lean_dec(v_a_4341_);
lean_dec(v_a_4340_);
lean_dec_ref(v_a_4339_);
return v_res_4348_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandGoal(lean_object* v_Q_4349_, lean_object* v_inst_4350_, lean_object* v_gref_4351_, lean_object* v_a_4352_, lean_object* v_a_4353_, lean_object* v_a_4354_, lean_object* v_a_4355_, lean_object* v_a_4356_, lean_object* v_a_4357_, lean_object* v_a_4358_, lean_object* v_a_4359_){
_start:
{
lean_object* v___x_4361_; 
v___x_4361_ = lp_aesop_Aesop_expandGoal___redArg(v_inst_4350_, v_gref_4351_, v_a_4352_, v_a_4353_, v_a_4354_, v_a_4355_, v_a_4356_, v_a_4357_, v_a_4358_, v_a_4359_);
return v___x_4361_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandGoal___boxed(lean_object* v_Q_4362_, lean_object* v_inst_4363_, lean_object* v_gref_4364_, lean_object* v_a_4365_, lean_object* v_a_4366_, lean_object* v_a_4367_, lean_object* v_a_4368_, lean_object* v_a_4369_, lean_object* v_a_4370_, lean_object* v_a_4371_, lean_object* v_a_4372_, lean_object* v_a_4373_){
_start:
{
lean_object* v_res_4374_; 
v_res_4374_ = lp_aesop_Aesop_expandGoal(v_Q_4362_, v_inst_4363_, v_gref_4364_, v_a_4365_, v_a_4366_, v_a_4367_, v_a_4368_, v_a_4369_, v_a_4370_, v_a_4371_, v_a_4372_);
lean_dec(v_a_4372_);
lean_dec_ref(v_a_4371_);
lean_dec(v_a_4370_);
lean_dec_ref(v_a_4369_);
lean_dec(v_a_4368_);
lean_dec(v_a_4367_);
lean_dec(v_a_4366_);
lean_dec_ref(v_a_4365_);
return v_res_4374_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_Expansion_Norm(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_AddRapp(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_State_UpdateGoal(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_RuleSelection(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_Expansion_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_Expansion(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Expansion_Norm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_AddRapp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State_UpdateGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_RuleSelection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Expansion_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_Expansion(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Search_Expansion_Norm(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_AddRapp(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_State_UpdateGoal(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_RuleSelection(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Expansion_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Expansion(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Expansion_Norm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_AddRapp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State_UpdateGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_RuleSelection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Expansion_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Expansion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_Expansion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_Expansion(builtin);
}
#ifdef __cplusplus
}
#endif
