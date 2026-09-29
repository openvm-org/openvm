// Lean compiler output
// Module: Aesop.Search.Main
// Imports: public import Init public meta import Init public import Aesop.Script.Main public import Aesop.Search.ExpandSafePrefix public import Aesop.Tree.Check public import Aesop.Tree.ExtractProof public import Aesop.Tree.ExtractScript public import Aesop.Tree.Tracing import Aesop.Frontend.Extension import Aesop.Search.Queue import Aesop.Tree.Free import Aesop.Tree.Stats
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
lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getTree___redArg(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
uint8_t lp_aesop_Aesop_NodeState_isUnprovable(uint8_t);
lean_object* lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* lp_aesop_Aesop_BaseM_instMonadStats;
lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(lean_object*);
lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg(lean_object*);
lean_object* lp_aesop_Aesop_expandSafePrefix___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_extractSafePrefix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_proof;
lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_clearForwardImplDetailHyps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
extern lean_object* l_Lean_Core_instMonadLogCoreM;
lean_object* l_Lean_instMonadLogOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* l_Lean_logWarning___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_warn_nonterminal;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_RegularRule_isUnsafe(lean_object*);
extern lean_object* lp_aesop_Aesop_preprocessRule;
lean_object* lp_aesop_Aesop_RegularRule_name(lean_object*);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
lean_object* lp_aesop_Aesop_getRootMVarId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_smallErrorMessages;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getRootGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_tree;
lean_object* lp_aesop_Aesop_Goal_traceTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_GoalRef_extractScriptCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ExtractScriptM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
double lean_float_div(double, double);
lean_object* l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
lean_object* lp_aesop_Aesop_Script_UScript_checkIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getRootMetaState___redArg(lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_script;
lean_object* l_instMonadLiftT___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instMonadLiftTOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_UScript_optimize(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_addTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_extractSafePrefixScript(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TreeM_instMonad;
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadExceptOfEIO(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptReaderT___redArg(lean_object*);
lean_object* l_Lean_instExceptToTraceResult___lam__0___boxed(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
extern lean_object* l_Lean_KVMap_instValueString;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instMonadMCtxMetaM;
lean_object* l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getExprMVarAssignment_x3f___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlReaderT(lean_object*, lean_object*);
lean_object* l_instMonadControlStateRefT_x27(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MVarId_withContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getRootMVarCluster___redArg(lean_object*);
uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_extractProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadWithOptionsCoreM;
lean_object* l_Lean_instMonadWithOptionsOfMonadFunctor___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_ReaderT_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_withPPAnalyze___redArg(lean_object*, lean_object*);
lean_object* l_Lean_instantiateMVars___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
lean_object* l_Lean_Meta_getMVarsNoDelayed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_steps;
extern lean_object* lp_aesop_Aesop_newNodeEmoji;
lean_object* lp_aesop_Aesop_Rapp_traceMetadata(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_liftIOCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadLiftBaseIOEIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_IO_instMonadLiftSTRealWorldBaseIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Goal_traceMetadata(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lp_aesop_Aesop_popGoal_x3f___redArg(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Goal_isActive(lean_object*);
lean_object* lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_GoalRef_markForcedUnprovable(lean_object*);
lean_object* lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(lean_object*);
lean_object* lp_aesop_Aesop_expandGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getIteration___redArg(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_enqueueGoals___redArg(lean_object*, lean_object*, lean_object*);
double lp_aesop_Aesop_Goal_priority(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instMonadEnvMetaM;
lean_object* l_Lean_Meta_instMonadLCtxMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMessageContextFull___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleResult_toEmoji___boxed(lean_object*);
lean_object* lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Percent_toHumanString(double);
lean_object* lp_aesop_Aesop_checkInvariantsIfEnabled___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_incrementIteration___redArg(lean_object*);
lean_object* l_Lean_throwMaxRecDepthAt___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_dev_generateScript;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_Check_script;
uint8_t lp_aesop_Aesop_Check_get(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_Check_script_steps;
lean_object* lp_aesop_Aesop_Options_queue(lean_object*);
lean_object* lp_aesop_Aesop_collectGoalStatsIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_freeTree___redArg(lean_object*);
lean_object* lp_aesop_Aesop_SearchM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_BaseM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkLocalRuleSet(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_nextActiveGoal___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_nextActiveGoal___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__3;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__4;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__6;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__7;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__8;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__9;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__10;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__11;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__12;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__13;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__14;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__15;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__16;
static const lean_closure_object lp_aesop_Aesop_nextActiveGoal___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__17 = (const lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__17_value;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__18;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__19;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__20;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__21;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__22;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__23;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__24;
static const lean_string_object lp_aesop_Aesop_nextActiveGoal___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 59, .m_capacity = 59, .m_length = 58, .m_data = "aesop/expandNextGoal: internal error: no active goals left"};
static const lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__25 = (const lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__25_value;
static lean_once_cell_t lp_aesop_Aesop_nextActiveGoal___redArg___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___closed__26;
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadLCtxMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__7_value)} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__8_value)} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleResult_toEmoji___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " (G"};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2;
static const lean_string_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ") ["};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4;
static const lean_string_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = "] ⋯ ⊢ "};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg(lean_object*, double, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt(lean_object*, lean_object*, lean_object*, double, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Metadata"};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__0_value)}};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0;
static const lean_string_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___boxed(lean_object**);
static const lean_string_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__0;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__2_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__4;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__6;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__7;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__8;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__9;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instExceptToTraceResult___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__11_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_liftIOCore___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__12_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftBaseIOEIO___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__13_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_IO_instMonadLiftSTRealWorldBaseIO___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__14 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__14_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftT___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__15 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__15_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__15_value),((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__14_value)} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__16 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__16_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__16_value),((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__13_value)} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__17 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__17_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__17_value),((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__12_value)} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__18 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__18_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__18_value),((lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__1_value)} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__19 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__19_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__19_value),((lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__0_value)} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__20 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__20_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___boxed(lean_object**);
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1;
static lean_once_cell_t lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Initial goal:"};
static const lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 88, .m_capacity = 88, .m_length = 87, .m_data = "Treating the goal as unprovable since it is beyond the maximum rule application depth ("};
static const lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0 = (const lean_object*)&lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1;
static const lean_string_object lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ")."};
static const lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2 = (const lean_object*)&lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__3;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__4;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__6;
static lean_once_cell_t lp_aesop_Aesop_expandNextGoal___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandNextGoal___redArg___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_checkGoalLimit___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "maximum number of goals ("};
static const lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_checkGoalLimit___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_checkGoalLimit___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___closed__1;
static const lean_string_object lp_aesop_Aesop_checkGoalLimit___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = ") reached. Set the 'maxGoals' option to increase the limit."};
static const lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_checkGoalLimit___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_checkGoalLimit___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_checkRappLimit___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "maximum number of rule applications ("};
static const lean_object* lp_aesop_Aesop_checkRappLimit___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_checkRappLimit___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_checkRappLimit___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRappLimit___redArg___closed__1;
static const lean_string_object lp_aesop_Aesop_checkRappLimit___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 71, .m_capacity = 71, .m_length = 70, .m_data = ") reached. Set the 'maxRuleApplications' option to increase the limit."};
static const lean_object* lp_aesop_Aesop_checkRappLimit___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_checkRappLimit___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_checkRappLimit___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRappLimit___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "failed to prove the goal after exhaustive search."};
static const lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1;
static const lean_string_object lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 100, .m_capacity = 100, .m_length = 99, .m_data = "failed to prove the goal. Some goals were not explored because the maximum rule application depth ("};
static const lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3;
static const lean_string_object lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 75, .m_capacity = 75, .m_length = 74, .m_data = ") was reached. Set option 'maxRuleApplicationDepth' to increase the limit."};
static const lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed(lean_object*);
static const lean_string_object lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Final proof:"};
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2___boxed(lean_object**);
static const lean_string_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Proof: "};
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1;
static const lean_string_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "\nUnassigned metavariables: "};
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8_value;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9_value;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__4_value),((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__11_value),((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__6_value),((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__7_value),((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__8_value),((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__12_value),((lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13_value;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_MessageData_ofName, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14_value;
static const lean_string_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "aesop: internal error: extracted proof has metavariables."};
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15_value;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16;
static const lean_string_object lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 79, .m_data = "aesop: internal error: root goal is proven but its metavariable is not assigned"};
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17_value;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18;
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed(lean_object**);
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__1;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_finalizeProof___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_finalizeProof___redArg___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__4;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__6;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__7;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__8;
static lean_once_cell_t lp_aesop_Aesop_finalizeProof___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_finalizeProof___redArg___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_traceScript___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Extract script"};
static const lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_traceScript___redArg___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0_value;
static const lean_string_object lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Unstructured script:"};
static const lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1 = (const lean_object*)&lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___boxed(lean_object**);
static const lean_closure_object lp_aesop_Aesop_traceScript___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_traceScript___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_traceScript___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_traceScript___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__3;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__4;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__6;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__7;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__8;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__9;
static lean_once_cell_t lp_aesop_Aesop_traceScript___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_traceScript___redArg___closed__10;
static const lean_closure_object lp_aesop_Aesop_traceScript___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__15_value),((lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__1_value)} };
static const lean_object* lp_aesop_Aesop_traceScript___redArg___closed__11 = (const lean_object*)&lp_aesop_Aesop_traceScript___redArg___closed__11_value;
static const lean_closure_object lp_aesop_Aesop_traceScript___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_traceScript___redArg___closed__11_value),((lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__1_value)} };
static const lean_object* lp_aesop_Aesop_traceScript___redArg___closed__12 = (const lean_object*)&lp_aesop_Aesop_traceScript___redArg___closed__12_value;
static const lean_closure_object lp_aesop_Aesop_traceScript___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_traceScript___redArg___closed__12_value),((lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_traceScript___redArg___closed__13 = (const lean_object*)&lp_aesop_Aesop_traceScript___redArg___closed__13_value;
static const lean_closure_object lp_aesop_Aesop_traceScript___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_traceScript___redArg___closed__13_value),((lean_object*)&lp_aesop_Aesop_nextActiveGoal___redArg___closed__17_value)} };
static const lean_object* lp_aesop_Aesop_traceScript___redArg___closed__14 = (const lean_object*)&lp_aesop_Aesop_traceScript___redArg___closed__14_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeHasProgress(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeHasProgress___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg___lam__0(lean_object*);
static const lean_string_object lp_aesop_Aesop_throwAesopEx___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "Tactic `aesop` failed\nInitial goal:"};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__1;
static const lean_string_object lp_aesop_Aesop_throwAesopEx___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Tactic `aesop` failed, "};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__3;
static const lean_string_object lp_aesop_Aesop_throwAesopEx___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "\nInitial goal:"};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__5;
static const lean_closure_object lp_aesop_Aesop_throwAesopEx___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_throwAesopEx___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__6_value;
static const lean_string_object lp_aesop_Aesop_throwAesopEx___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_throwAesopEx___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__9;
static const lean_string_object lp_aesop_Aesop_throwAesopEx___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "\nRemaining goals after safe rules:"};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__11;
static const lean_string_object lp_aesop_Aesop_throwAesopEx___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 90, .m_capacity = 90, .m_length = 89, .m_data = "\nThe safe prefix was not fully expanded because the maximum number of rule applications ("};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__12 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__12_value;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__13;
static const lean_string_object lp_aesop_Aesop_throwAesopEx___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = ") was reached."};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__14 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__14_value;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__15;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__16;
static const lean_string_object lp_aesop_Aesop_throwAesopEx___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Tactic `aesop` failed"};
static const lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__17 = (const lean_object*)&lp_aesop_Aesop_throwAesopEx___redArg___closed__17_value;
static lean_once_cell_t lp_aesop_Aesop_throwAesopEx___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_throwAesopEx___redArg___closed__18;
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_handleNonfatalError___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_handleNonfatalError___redArg___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_handleNonfatalError___redArg___closed__0_value;
static const lean_string_object lp_aesop_Aesop_handleNonfatalError___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 92, .m_capacity = 92, .m_length = 91, .m_data = "aesop: safe prefix was not fully expanded because the maximum number of rule applications ("};
static const lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_handleNonfatalError___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_handleNonfatalError___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__2;
static const lean_string_object lp_aesop_Aesop_handleNonfatalError___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "aesop: "};
static const lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_handleNonfatalError___redArg___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_handleNonfatalError___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__4;
static const lean_array_object lp_aesop_Aesop_handleNonfatalError___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_handleNonfatalError___redArg___closed__5_value;
static const lean_string_object lp_aesop_Aesop_handleNonfatalError___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "made no progress"};
static const lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_handleNonfatalError___redArg___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_handleNonfatalError___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_handleNonfatalError___redArg___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_handleNonfatalError___redArg___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_handleNonfatalError___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__8;
static lean_once_cell_t lp_aesop_Aesop_handleNonfatalError___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__9;
static const lean_string_object lp_aesop_Aesop_handleNonfatalError___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "<no proof>"};
static const lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_handleNonfatalError___redArg___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_handleNonfatalError___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___closed__11;
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_search___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_object* lp_aesop_Aesop_search___closed__0 = (const lean_object*)&lp_aesop_Aesop_search___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_search(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__2(void){
_start:
{
lean_object* v___x_3_; lean_object* v___f_4_; 
v___x_3_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_4_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_4_, 0, v___x_3_);
return v___f_4_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__3(void){
_start:
{
lean_object* v___x_5_; lean_object* v___f_6_; 
v___x_5_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_6_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_6_, 0, v___x_5_);
return v___f_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__4(void){
_start:
{
lean_object* v___f_7_; lean_object* v___f_8_; lean_object* v___x_9_; 
v___f_7_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__3, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__3_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__3);
v___f_8_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__2, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__2_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__2);
v___x_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_9_, 0, v___f_8_);
lean_ctor_set(v___x_9_, 1, v___f_7_);
return v___x_9_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__5(void){
_start:
{
lean_object* v___x_10_; lean_object* v___f_11_; 
v___x_10_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__4, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__4_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__4);
v___f_11_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_11_, 0, v___x_10_);
return v___f_11_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__6(void){
_start:
{
lean_object* v___x_12_; lean_object* v___f_13_; 
v___x_12_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__4, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__4_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__4);
v___f_13_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_13_, 0, v___x_12_);
return v___f_13_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__7(void){
_start:
{
lean_object* v___f_14_; lean_object* v___f_15_; lean_object* v___x_16_; 
v___f_14_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__6, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__6_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__6);
v___f_15_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__5, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__5_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__5);
v___x_16_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_16_, 0, v___f_15_);
lean_ctor_set(v___x_16_, 1, v___f_14_);
return v___x_16_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__8(void){
_start:
{
lean_object* v___x_17_; lean_object* v___f_18_; 
v___x_17_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__7, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__7_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__7);
v___f_18_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_18_, 0, v___x_17_);
return v___f_18_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__9(void){
_start:
{
lean_object* v___x_19_; lean_object* v___f_20_; 
v___x_19_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__7, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__7_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__7);
v___f_20_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_20_, 0, v___x_19_);
return v___f_20_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__10(void){
_start:
{
lean_object* v___f_21_; lean_object* v___f_22_; lean_object* v___x_23_; 
v___f_21_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__9, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__9_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__9);
v___f_22_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__8, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__8_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__8);
v___x_23_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_23_, 0, v___f_22_);
lean_ctor_set(v___x_23_, 1, v___f_21_);
return v___x_23_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__11(void){
_start:
{
lean_object* v___x_24_; lean_object* v___f_25_; 
v___x_24_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__10, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__10_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__10);
v___f_25_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_25_, 0, v___x_24_);
return v___f_25_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__12(void){
_start:
{
lean_object* v___x_26_; lean_object* v___f_27_; 
v___x_26_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__10, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__10_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__10);
v___f_27_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_27_, 0, v___x_26_);
return v___f_27_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__13(void){
_start:
{
lean_object* v___f_28_; lean_object* v___f_29_; lean_object* v___x_30_; 
v___f_28_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__12, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__12_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__12);
v___f_29_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__11, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__11_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__11);
v___x_30_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_30_, 0, v___f_29_);
lean_ctor_set(v___x_30_, 1, v___f_28_);
return v___x_30_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__14(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___f_33_; 
v___x_31_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_32_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_33_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_33_, 0, v___x_32_);
lean_closure_set(v___f_33_, 1, v___x_31_);
return v___f_33_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__15(void){
_start:
{
lean_object* v___x_34_; lean_object* v___f_35_; lean_object* v___f_36_; 
v___x_34_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___f_35_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__14, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__14_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__14);
v___f_36_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_36_, 0, v___f_35_);
lean_closure_set(v___f_36_, 1, v___x_34_);
return v___f_36_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__16(void){
_start:
{
lean_object* v___f_37_; lean_object* v___f_38_; lean_object* v___f_39_; 
v___f_37_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___f_38_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__15, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__15_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__15);
v___f_39_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_39_, 0, v___f_38_);
lean_closure_set(v___f_39_, 1, v___f_37_);
return v___f_39_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__18(void){
_start:
{
lean_object* v___x_41_; lean_object* v___f_42_; 
v___x_41_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__13, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__13_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__13);
v___f_42_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_42_, 0, v___x_41_);
return v___f_42_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__19(void){
_start:
{
lean_object* v___x_43_; lean_object* v___f_44_; 
v___x_43_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__13, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__13_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__13);
v___f_44_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_44_, 0, v___x_43_);
return v___f_44_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__20(void){
_start:
{
lean_object* v___f_45_; lean_object* v___f_46_; lean_object* v___x_47_; 
v___f_45_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__19, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__19_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__19);
v___f_46_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__18, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__18_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__18);
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v___f_46_);
lean_ctor_set(v___x_47_, 1, v___f_45_);
return v___x_47_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__21(void){
_start:
{
lean_object* v___x_48_; lean_object* v___f_49_; 
v___x_48_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__20, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__20_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__20);
v___f_49_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_49_, 0, v___x_48_);
return v___f_49_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__22(void){
_start:
{
lean_object* v___x_50_; lean_object* v___f_51_; 
v___x_50_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__20, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__20_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__20);
v___f_51_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_51_, 0, v___x_50_);
return v___f_51_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23(void){
_start:
{
lean_object* v___f_52_; lean_object* v___f_53_; lean_object* v___x_54_; 
v___f_52_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__22, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__22_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__22);
v___f_53_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__21, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__21_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__21);
v___x_54_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_54_, 0, v___f_53_);
lean_ctor_set(v___x_54_, 1, v___f_52_);
return v___x_54_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24(void){
_start:
{
lean_object* v___f_55_; lean_object* v___f_56_; lean_object* v___f_57_; 
v___f_55_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__17));
v___f_56_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__16, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__16_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__16);
v___f_57_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_57_, 0, v___f_56_);
lean_closure_set(v___f_57_, 1, v___f_55_);
return v___f_57_;
}
}
static lean_object* _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__26(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__25));
v___x_60_ = l_Lean_stringToMessageData(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___redArg(lean_object* v_inst_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___f_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_71_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_61_);
v___x_72_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___x_73_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_61_);
v___f_74_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
lean_inc_ref(v___x_71_);
v___x_75_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_74_, v___x_71_);
v___x_76_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_76_, 0, v___x_72_);
lean_ctor_set(v___x_76_, 1, v___x_73_);
lean_ctor_set(v___x_76_, 2, v___x_75_);
lean_inc_ref(v_inst_61_);
v___x_77_ = lp_aesop_Aesop_popGoal_x3f___redArg(v_inst_61_, v_a_63_);
if (lean_obj_tag(v___x_77_) == 0)
{
lean_object* v_a_78_; lean_object* v___x_80_; uint8_t v_isShared_81_; uint8_t v_isSharedCheck_94_; 
v_a_78_ = lean_ctor_get(v___x_77_, 0);
v_isSharedCheck_94_ = !lean_is_exclusive(v___x_77_);
if (v_isSharedCheck_94_ == 0)
{
v___x_80_ = v___x_77_;
v_isShared_81_ = v_isSharedCheck_94_;
goto v_resetjp_79_;
}
else
{
lean_inc(v_a_78_);
lean_dec(v___x_77_);
v___x_80_ = lean_box(0);
v_isShared_81_ = v_isSharedCheck_94_;
goto v_resetjp_79_;
}
v_resetjp_79_:
{
if (lean_obj_tag(v_a_78_) == 1)
{
lean_object* v_val_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; 
lean_dec_ref_known(v___x_76_, 3);
lean_dec_ref(v___x_71_);
v_val_82_ = lean_ctor_get(v_a_78_, 0);
lean_inc(v_val_82_);
lean_dec_ref_known(v_a_78_, 1);
v___x_83_ = lean_st_ref_get(v_a_63_);
lean_dec(v___x_83_);
v___x_84_ = lean_st_ref_get(v_val_82_);
v___x_85_ = lean_st_ref_get(v_a_63_);
lean_dec(v___x_85_);
v___x_86_ = lp_aesop_Aesop_Goal_isActive(v___x_84_);
if (v___x_86_ == 0)
{
lean_dec(v_val_82_);
lean_del_object(v___x_80_);
goto _start;
}
else
{
lean_object* v___x_89_; 
lean_dec_ref(v_inst_61_);
if (v_isShared_81_ == 0)
{
lean_ctor_set(v___x_80_, 0, v_val_82_);
v___x_89_ = v___x_80_;
goto v_reusejp_88_;
}
else
{
lean_object* v_reuseFailAlloc_90_; 
v_reuseFailAlloc_90_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_90_, 0, v_val_82_);
v___x_89_ = v_reuseFailAlloc_90_;
goto v_reusejp_88_;
}
v_reusejp_88_:
{
return v___x_89_;
}
}
}
else
{
lean_object* v___x_91_; lean_object* v___x_3144__overap_92_; lean_object* v___x_93_; 
lean_del_object(v___x_80_);
lean_dec(v_a_78_);
lean_dec_ref(v_inst_61_);
v___x_91_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__26, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__26_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__26);
v___x_3144__overap_92_ = l_Lean_throwError___redArg(v___x_71_, v___x_76_, v___x_91_);
lean_inc(v_a_69_);
lean_inc_ref(v_a_68_);
lean_inc(v_a_67_);
lean_inc_ref(v_a_66_);
lean_inc(v_a_65_);
lean_inc(v_a_64_);
lean_inc(v_a_63_);
lean_inc_ref(v_a_62_);
v___x_93_ = lean_apply_9(v___x_3144__overap_92_, v_a_62_, v_a_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_, v_a_68_, v_a_69_, lean_box(0));
return v___x_93_;
}
}
}
else
{
lean_object* v_a_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_102_; 
lean_dec_ref_known(v___x_76_, 3);
lean_dec_ref(v___x_71_);
lean_dec_ref(v_inst_61_);
v_a_95_ = lean_ctor_get(v___x_77_, 0);
v_isSharedCheck_102_ = !lean_is_exclusive(v___x_77_);
if (v_isSharedCheck_102_ == 0)
{
v___x_97_ = v___x_77_;
v_isShared_98_ = v_isSharedCheck_102_;
goto v_resetjp_96_;
}
else
{
lean_inc(v_a_95_);
lean_dec(v___x_77_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_102_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_100_; 
if (v_isShared_98_ == 0)
{
v___x_100_ = v___x_97_;
goto v_reusejp_99_;
}
else
{
lean_object* v_reuseFailAlloc_101_; 
v_reuseFailAlloc_101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_101_, 0, v_a_95_);
v___x_100_ = v_reuseFailAlloc_101_;
goto v_reusejp_99_;
}
v_reusejp_99_:
{
return v___x_100_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___redArg___boxed(lean_object* v_inst_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_, lean_object* v_a_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_aesop_Aesop_nextActiveGoal___redArg(v_inst_103_, v_a_104_, v_a_105_, v_a_106_, v_a_107_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
lean_dec(v_a_111_);
lean_dec_ref(v_a_110_);
lean_dec(v_a_109_);
lean_dec_ref(v_a_108_);
lean_dec(v_a_107_);
lean_dec(v_a_106_);
lean_dec(v_a_105_);
lean_dec_ref(v_a_104_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal(lean_object* v_Q_114_, lean_object* v_inst_115_, lean_object* v_a_116_, lean_object* v_a_117_, lean_object* v_a_118_, lean_object* v_a_119_, lean_object* v_a_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_aesop_Aesop_nextActiveGoal___redArg(v_inst_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_, v_a_121_, v_a_122_, v_a_123_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_nextActiveGoal___boxed(lean_object* v_Q_126_, lean_object* v_inst_127_, lean_object* v_a_128_, lean_object* v_a_129_, lean_object* v_a_130_, lean_object* v_a_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_aesop_Aesop_nextActiveGoal(v_Q_126_, v_inst_127_, v_a_128_, v_a_129_, v_a_130_, v_a_131_, v_a_132_, v_a_133_, v_a_134_, v_a_135_);
lean_dec(v_a_135_);
lean_dec_ref(v_a_134_);
lean_dec(v_a_133_);
lean_dec_ref(v_a_132_);
lean_dec(v_a_131_);
lean_dec(v_a_130_);
lean_dec(v_a_129_);
lean_dec_ref(v_a_128_);
return v_res_137_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = l_instMonadEIO(lean_box(0));
return v___x_138_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_139_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__0);
v___x_140_ = l_StateRefT_x27_instMonad___redArg(v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0(lean_object* v_initialGoal_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = l_Lean_MVarId_getType(v_initialGoal_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_);
if (lean_obj_tag(v___x_157_) == 0)
{
lean_object* v_a_158_; lean_object* v___x_159_; lean_object* v_toApplicative_160_; lean_object* v_toFunctor_161_; lean_object* v_toSeq_162_; lean_object* v_toSeqLeft_163_; lean_object* v_toSeqRight_164_; lean_object* v___f_165_; lean_object* v___f_166_; lean_object* v___f_167_; lean_object* v___f_168_; lean_object* v___x_169_; lean_object* v___f_170_; lean_object* v___f_171_; lean_object* v___f_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v_toApplicative_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_210_; 
v_a_158_ = lean_ctor_get(v___x_157_, 0);
lean_inc(v_a_158_);
lean_dec_ref_known(v___x_157_, 1);
v___x_159_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1);
v_toApplicative_160_ = lean_ctor_get(v___x_159_, 0);
v_toFunctor_161_ = lean_ctor_get(v_toApplicative_160_, 0);
v_toSeq_162_ = lean_ctor_get(v_toApplicative_160_, 2);
v_toSeqLeft_163_ = lean_ctor_get(v_toApplicative_160_, 3);
v_toSeqRight_164_ = lean_ctor_get(v_toApplicative_160_, 4);
v___f_165_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2));
v___f_166_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3));
lean_inc_ref_n(v_toFunctor_161_, 2);
v___f_167_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_167_, 0, v_toFunctor_161_);
v___f_168_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_168_, 0, v_toFunctor_161_);
v___x_169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_169_, 0, v___f_167_);
lean_ctor_set(v___x_169_, 1, v___f_168_);
lean_inc(v_toSeqRight_164_);
v___f_170_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_170_, 0, v_toSeqRight_164_);
lean_inc(v_toSeqLeft_163_);
v___f_171_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_171_, 0, v_toSeqLeft_163_);
lean_inc(v_toSeq_162_);
v___f_172_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_172_, 0, v_toSeq_162_);
v___x_173_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_173_, 0, v___x_169_);
lean_ctor_set(v___x_173_, 1, v___f_165_);
lean_ctor_set(v___x_173_, 2, v___f_172_);
lean_ctor_set(v___x_173_, 3, v___f_171_);
lean_ctor_set(v___x_173_, 4, v___f_170_);
v___x_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v___f_166_);
v___x_175_ = l_StateRefT_x27_instMonad___redArg(v___x_174_);
v_toApplicative_176_ = lean_ctor_get(v___x_175_, 0);
v_isSharedCheck_210_ = !lean_is_exclusive(v___x_175_);
if (v_isSharedCheck_210_ == 0)
{
lean_object* v_unused_211_; 
v_unused_211_ = lean_ctor_get(v___x_175_, 1);
lean_dec(v_unused_211_);
v___x_178_ = v___x_175_;
v_isShared_179_ = v_isSharedCheck_210_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_toApplicative_176_);
lean_dec(v___x_175_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_210_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v_toFunctor_180_; lean_object* v_toSeq_181_; lean_object* v_toSeqLeft_182_; lean_object* v_toSeqRight_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_208_; 
v_toFunctor_180_ = lean_ctor_get(v_toApplicative_176_, 0);
v_toSeq_181_ = lean_ctor_get(v_toApplicative_176_, 2);
v_toSeqLeft_182_ = lean_ctor_get(v_toApplicative_176_, 3);
v_toSeqRight_183_ = lean_ctor_get(v_toApplicative_176_, 4);
v_isSharedCheck_208_ = !lean_is_exclusive(v_toApplicative_176_);
if (v_isSharedCheck_208_ == 0)
{
lean_object* v_unused_209_; 
v_unused_209_ = lean_ctor_get(v_toApplicative_176_, 1);
lean_dec(v_unused_209_);
v___x_185_ = v_toApplicative_176_;
v_isShared_186_ = v_isSharedCheck_208_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_toSeqRight_183_);
lean_inc(v_toSeqLeft_182_);
lean_inc(v_toSeq_181_);
lean_inc(v_toFunctor_180_);
lean_dec(v_toApplicative_176_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_208_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v___f_187_; lean_object* v___f_188_; lean_object* v___f_189_; lean_object* v___f_190_; lean_object* v___x_191_; lean_object* v___f_192_; lean_object* v___f_193_; lean_object* v___f_194_; lean_object* v___x_196_; 
v___f_187_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4));
v___f_188_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5));
lean_inc_ref(v_toFunctor_180_);
v___f_189_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_189_, 0, v_toFunctor_180_);
v___f_190_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_190_, 0, v_toFunctor_180_);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v___f_189_);
lean_ctor_set(v___x_191_, 1, v___f_190_);
v___f_192_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_192_, 0, v_toSeqRight_183_);
v___f_193_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_193_, 0, v_toSeqLeft_182_);
v___f_194_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_194_, 0, v_toSeq_181_);
if (v_isShared_186_ == 0)
{
lean_ctor_set(v___x_185_, 4, v___f_192_);
lean_ctor_set(v___x_185_, 3, v___f_193_);
lean_ctor_set(v___x_185_, 2, v___f_194_);
lean_ctor_set(v___x_185_, 1, v___f_187_);
lean_ctor_set(v___x_185_, 0, v___x_191_);
v___x_196_ = v___x_185_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v___x_191_);
lean_ctor_set(v_reuseFailAlloc_207_, 1, v___f_187_);
lean_ctor_set(v_reuseFailAlloc_207_, 2, v___f_194_);
lean_ctor_set(v_reuseFailAlloc_207_, 3, v___f_193_);
lean_ctor_set(v_reuseFailAlloc_207_, 4, v___f_192_);
v___x_196_ = v_reuseFailAlloc_207_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
lean_object* v___x_198_; 
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 1, v___f_188_);
lean_ctor_set(v___x_178_, 0, v___x_196_);
v___x_198_ = v___x_178_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v___x_196_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v___f_188_);
v___x_198_ = v_reuseFailAlloc_206_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___f_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_1226__overap_204_; lean_object* v___x_205_; 
v___x_199_ = l_Lean_Meta_instMonadEnvMetaM;
v___x_200_ = l_Lean_Meta_instMonadMCtxMetaM;
v___f_201_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__6));
v___x_202_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9));
v___x_203_ = l_Lean_MessageData_ofExpr(v_a_158_);
v___x_1226__overap_204_ = l_Lean_addMessageContextFull___redArg(v___x_198_, v___x_199_, v___x_200_, v___f_201_, v___x_202_, v___x_203_);
v___x_205_ = lean_apply_5(v___x_1226__overap_204_, v___y_152_, v___y_153_, v___y_154_, v___y_155_, lean_box(0));
return v___x_205_;
}
}
}
}
}
else
{
lean_object* v_a_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_219_; 
lean_dec(v___y_155_);
lean_dec_ref(v___y_154_);
lean_dec(v___y_153_);
lean_dec_ref(v___y_152_);
v_a_212_ = lean_ctor_get(v___x_157_, 0);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_157_);
if (v_isSharedCheck_219_ == 0)
{
v___x_214_ = v___x_157_;
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_a_212_);
lean_dec(v___x_157_);
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
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed(lean_object* v_initialGoal_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0(v_initialGoal_220_, v___y_221_, v___y_222_, v___y_223_, v___y_224_);
return v_res_226_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_229_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__1));
v___x_230_ = l_Lean_stringToMessageData(v___x_229_);
return v___x_230_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__3));
v___x_233_ = l_Lean_stringToMessageData(v___x_232_);
return v___x_233_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6(void){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_235_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__5));
v___x_236_ = l_Lean_stringToMessageData(v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg(lean_object* v_id_237_, double v_priority_238_, lean_object* v_initialGoal_239_, lean_object* v_initialMetaState_240_, lean_object* v_result_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_, lean_object* v_a_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_248_; lean_object* v_toApplicative_249_; lean_object* v_toFunctor_250_; lean_object* v_toSeq_251_; lean_object* v_toSeqLeft_252_; lean_object* v_toSeqRight_253_; lean_object* v___f_254_; lean_object* v___f_255_; lean_object* v___f_256_; lean_object* v___f_257_; lean_object* v___x_258_; lean_object* v___f_259_; lean_object* v___f_260_; lean_object* v___f_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v_toApplicative_267_; lean_object* v_toFunctor_268_; lean_object* v_toSeq_269_; lean_object* v_toSeqLeft_270_; lean_object* v_toSeqRight_271_; lean_object* v___f_272_; lean_object* v___f_273_; lean_object* v___x_274_; lean_object* v___f_275_; lean_object* v___f_276_; lean_object* v___f_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v_toApplicative_281_; lean_object* v___x_283_; uint8_t v_isShared_284_; uint8_t v_isSharedCheck_337_; 
v___x_248_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1);
v_toApplicative_249_ = lean_ctor_get(v___x_248_, 0);
v_toFunctor_250_ = lean_ctor_get(v_toApplicative_249_, 0);
v_toSeq_251_ = lean_ctor_get(v_toApplicative_249_, 2);
v_toSeqLeft_252_ = lean_ctor_get(v_toApplicative_249_, 3);
v_toSeqRight_253_ = lean_ctor_get(v_toApplicative_249_, 4);
v___f_254_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2));
v___f_255_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3));
lean_inc_ref_n(v_toFunctor_250_, 2);
v___f_256_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_256_, 0, v_toFunctor_250_);
v___f_257_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_257_, 0, v_toFunctor_250_);
v___x_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_258_, 0, v___f_256_);
lean_ctor_set(v___x_258_, 1, v___f_257_);
lean_inc(v_toSeqRight_253_);
v___f_259_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_259_, 0, v_toSeqRight_253_);
lean_inc(v_toSeqLeft_252_);
v___f_260_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_260_, 0, v_toSeqLeft_252_);
lean_inc(v_toSeq_251_);
v___f_261_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_261_, 0, v_toSeq_251_);
v___x_262_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_262_, 0, v___x_258_);
lean_ctor_set(v___x_262_, 1, v___f_254_);
lean_ctor_set(v___x_262_, 2, v___f_261_);
lean_ctor_set(v___x_262_, 3, v___f_260_);
lean_ctor_set(v___x_262_, 4, v___f_259_);
v___x_263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v___f_255_);
v___x_264_ = l_StateRefT_x27_instMonad___redArg(v___x_263_);
v___x_265_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_265_, 0, lean_box(0));
lean_closure_set(v___x_265_, 1, lean_box(0));
lean_closure_set(v___x_265_, 2, v___x_264_);
v___x_266_ = l_instMonadControlTOfPure___redArg(v___x_265_);
v_toApplicative_267_ = lean_ctor_get(v___x_248_, 0);
v_toFunctor_268_ = lean_ctor_get(v_toApplicative_267_, 0);
v_toSeq_269_ = lean_ctor_get(v_toApplicative_267_, 2);
v_toSeqLeft_270_ = lean_ctor_get(v_toApplicative_267_, 3);
v_toSeqRight_271_ = lean_ctor_get(v_toApplicative_267_, 4);
lean_inc_ref_n(v_toFunctor_268_, 2);
v___f_272_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_272_, 0, v_toFunctor_268_);
v___f_273_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_273_, 0, v_toFunctor_268_);
v___x_274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_274_, 0, v___f_272_);
lean_ctor_set(v___x_274_, 1, v___f_273_);
lean_inc(v_toSeqRight_271_);
v___f_275_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_275_, 0, v_toSeqRight_271_);
lean_inc(v_toSeqLeft_270_);
v___f_276_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_276_, 0, v_toSeqLeft_270_);
lean_inc(v_toSeq_269_);
v___f_277_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_277_, 0, v_toSeq_269_);
v___x_278_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_278_, 0, v___x_274_);
lean_ctor_set(v___x_278_, 1, v___f_254_);
lean_ctor_set(v___x_278_, 2, v___f_277_);
lean_ctor_set(v___x_278_, 3, v___f_276_);
lean_ctor_set(v___x_278_, 4, v___f_275_);
v___x_279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_279_, 0, v___x_278_);
lean_ctor_set(v___x_279_, 1, v___f_255_);
v___x_280_ = l_StateRefT_x27_instMonad___redArg(v___x_279_);
v_toApplicative_281_ = lean_ctor_get(v___x_280_, 0);
v_isSharedCheck_337_ = !lean_is_exclusive(v___x_280_);
if (v_isSharedCheck_337_ == 0)
{
lean_object* v_unused_338_; 
v_unused_338_ = lean_ctor_get(v___x_280_, 1);
lean_dec(v_unused_338_);
v___x_283_ = v___x_280_;
v_isShared_284_ = v_isSharedCheck_337_;
goto v_resetjp_282_;
}
else
{
lean_inc(v_toApplicative_281_);
lean_dec(v___x_280_);
v___x_283_ = lean_box(0);
v_isShared_284_ = v_isSharedCheck_337_;
goto v_resetjp_282_;
}
v_resetjp_282_:
{
lean_object* v_toFunctor_285_; lean_object* v_toSeq_286_; lean_object* v_toSeqLeft_287_; lean_object* v_toSeqRight_288_; lean_object* v___x_290_; uint8_t v_isShared_291_; uint8_t v_isSharedCheck_335_; 
v_toFunctor_285_ = lean_ctor_get(v_toApplicative_281_, 0);
v_toSeq_286_ = lean_ctor_get(v_toApplicative_281_, 2);
v_toSeqLeft_287_ = lean_ctor_get(v_toApplicative_281_, 3);
v_toSeqRight_288_ = lean_ctor_get(v_toApplicative_281_, 4);
v_isSharedCheck_335_ = !lean_is_exclusive(v_toApplicative_281_);
if (v_isSharedCheck_335_ == 0)
{
lean_object* v_unused_336_; 
v_unused_336_ = lean_ctor_get(v_toApplicative_281_, 1);
lean_dec(v_unused_336_);
v___x_290_ = v_toApplicative_281_;
v_isShared_291_ = v_isSharedCheck_335_;
goto v_resetjp_289_;
}
else
{
lean_inc(v_toSeqRight_288_);
lean_inc(v_toSeqLeft_287_);
lean_inc(v_toSeq_286_);
lean_inc(v_toFunctor_285_);
lean_dec(v_toApplicative_281_);
v___x_290_ = lean_box(0);
v_isShared_291_ = v_isSharedCheck_335_;
goto v_resetjp_289_;
}
v_resetjp_289_:
{
lean_object* v___f_292_; lean_object* v___f_293_; lean_object* v___f_294_; lean_object* v___f_295_; lean_object* v___x_296_; lean_object* v___f_297_; lean_object* v___f_298_; lean_object* v___f_299_; lean_object* v___x_301_; 
v___f_292_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4));
v___f_293_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5));
lean_inc_ref(v_toFunctor_285_);
v___f_294_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_294_, 0, v_toFunctor_285_);
v___f_295_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_295_, 0, v_toFunctor_285_);
v___x_296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_296_, 0, v___f_294_);
lean_ctor_set(v___x_296_, 1, v___f_295_);
v___f_297_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_297_, 0, v_toSeqRight_288_);
v___f_298_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_298_, 0, v_toSeqLeft_287_);
v___f_299_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_299_, 0, v_toSeq_286_);
if (v_isShared_291_ == 0)
{
lean_ctor_set(v___x_290_, 4, v___f_297_);
lean_ctor_set(v___x_290_, 3, v___f_298_);
lean_ctor_set(v___x_290_, 2, v___f_299_);
lean_ctor_set(v___x_290_, 1, v___f_292_);
lean_ctor_set(v___x_290_, 0, v___x_296_);
v___x_301_ = v___x_290_;
goto v_reusejp_300_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_296_);
lean_ctor_set(v_reuseFailAlloc_334_, 1, v___f_292_);
lean_ctor_set(v_reuseFailAlloc_334_, 2, v___f_299_);
lean_ctor_set(v_reuseFailAlloc_334_, 3, v___f_298_);
lean_ctor_set(v_reuseFailAlloc_334_, 4, v___f_297_);
v___x_301_ = v_reuseFailAlloc_334_;
goto v_reusejp_300_;
}
v_reusejp_300_:
{
lean_object* v___x_303_; 
if (v_isShared_284_ == 0)
{
lean_ctor_set(v___x_283_, 1, v___f_293_);
lean_ctor_set(v___x_283_, 0, v___x_301_);
v___x_303_ = v___x_283_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v___x_301_);
lean_ctor_set(v_reuseFailAlloc_333_, 1, v___f_293_);
v___x_303_ = v_reuseFailAlloc_333_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
lean_object* v___x_304_; lean_object* v___f_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_304_ = lean_st_ref_get(v_a_242_);
lean_dec(v___x_304_);
lean_inc(v_initialGoal_239_);
v___f_305_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___boxed), 6, 1);
lean_closure_set(v___f_305_, 0, v_initialGoal_239_);
v___x_306_ = l_Lean_MVarId_withContext___redArg(v___x_266_, v___x_303_, v_initialGoal_239_, v___f_305_);
v___x_307_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_initialMetaState_240_, v___x_306_, v_a_243_, v_a_244_, v_a_245_, v_a_246_);
if (lean_obj_tag(v___x_307_) == 0)
{
lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_332_; 
v_a_308_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_332_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_332_ == 0)
{
v___x_310_ = v___x_307_;
v_isShared_311_ = v_isSharedCheck_332_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_307_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_332_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___f_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_330_; 
v___f_312_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__0));
v___x_313_ = lp_aesop_Aesop_exceptRuleResultToEmoji___redArg(v___f_312_, v_result_241_);
v___x_314_ = l_Lean_stringToMessageData(v___x_313_);
v___x_315_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__2);
v___x_316_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_314_);
lean_ctor_set(v___x_316_, 1, v___x_315_);
v___x_317_ = l_Nat_reprFast(v_id_237_);
v___x_318_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_318_, 0, v___x_317_);
v___x_319_ = l_Lean_MessageData_ofFormat(v___x_318_);
v___x_320_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_320_, 0, v___x_316_);
lean_ctor_set(v___x_320_, 1, v___x_319_);
v___x_321_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__4);
v___x_322_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_320_);
lean_ctor_set(v___x_322_, 1, v___x_321_);
v___x_323_ = lp_aesop_Aesop_Percent_toHumanString(v_priority_238_);
v___x_324_ = l_Lean_stringToMessageData(v___x_323_);
v___x_325_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_322_);
lean_ctor_set(v___x_325_, 1, v___x_324_);
v___x_326_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___closed__6);
v___x_327_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_327_, 0, v___x_325_);
lean_ctor_set(v___x_327_, 1, v___x_326_);
v___x_328_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_328_, 0, v___x_327_);
lean_ctor_set(v___x_328_, 1, v_a_308_);
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 0, v___x_328_);
v___x_330_ = v___x_310_;
goto v_reusejp_329_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v___x_328_);
v___x_330_ = v_reuseFailAlloc_331_;
goto v_reusejp_329_;
}
v_reusejp_329_:
{
return v___x_330_;
}
}
}
else
{
lean_dec_ref(v_result_241_);
lean_dec(v_id_237_);
return v___x_307_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___boxed(lean_object* v_id_339_, lean_object* v_priority_340_, lean_object* v_initialGoal_341_, lean_object* v_initialMetaState_342_, lean_object* v_result_343_, lean_object* v_a_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_, lean_object* v_a_348_, lean_object* v_a_349_){
_start:
{
double v_priority_boxed_350_; lean_object* v_res_351_; 
v_priority_boxed_350_ = lean_unbox_float(v_priority_340_);
lean_dec_ref(v_priority_340_);
v_res_351_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg(v_id_339_, v_priority_boxed_350_, v_initialGoal_341_, v_initialMetaState_342_, v_result_343_, v_a_344_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
lean_dec(v_a_348_);
lean_dec_ref(v_a_347_);
lean_dec(v_a_346_);
lean_dec_ref(v_a_345_);
lean_dec(v_a_344_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt(lean_object* v_Q_352_, lean_object* v_inst_353_, lean_object* v_id_354_, double v_priority_355_, lean_object* v_initialGoal_356_, lean_object* v_initialMetaState_357_, lean_object* v_result_358_, lean_object* v_a_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg(v_id_354_, v_priority_355_, v_initialGoal_356_, v_initialMetaState_357_, v_result_358_, v_a_360_, v_a_363_, v_a_364_, v_a_365_, v_a_366_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed(lean_object* v_Q_369_, lean_object* v_inst_370_, lean_object* v_id_371_, lean_object* v_priority_372_, lean_object* v_initialGoal_373_, lean_object* v_initialMetaState_374_, lean_object* v_result_375_, lean_object* v_a_376_, lean_object* v_a_377_, lean_object* v_a_378_, lean_object* v_a_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_, lean_object* v_a_384_){
_start:
{
double v_priority_boxed_385_; lean_object* v_res_386_; 
v_priority_boxed_385_ = lean_unbox_float(v_priority_372_);
lean_dec_ref(v_priority_372_);
v_res_386_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt(v_Q_369_, v_inst_370_, v_id_371_, v_priority_boxed_385_, v_initialGoal_373_, v_initialMetaState_374_, v_result_375_, v_a_376_, v_a_377_, v_a_378_, v_a_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_);
lean_dec(v_a_383_);
lean_dec_ref(v_a_382_);
lean_dec(v_a_381_);
lean_dec_ref(v_a_380_);
lean_dec(v_a_379_);
lean_dec(v_a_378_);
lean_dec(v_a_377_);
lean_dec_ref(v_a_376_);
lean_dec_ref(v_inst_370_);
return v_res_386_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2(void){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_390_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__1));
v___x_391_ = l_Lean_MessageData_ofFormat(v___x_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0(lean_object* v_x_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_398_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___closed__2);
v___x_399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_399_, 0, v___x_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0___boxed(lean_object* v_x_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__0(v_x_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
lean_dec(v___y_404_);
lean_dec_ref(v___y_403_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
lean_dec_ref(v_x_400_);
return v_res_406_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_407_; lean_object* v___x_408_; 
v___x_407_ = lp_aesop_Aesop_newNodeEmoji;
v___x_408_ = l_Lean_stringToMessageData(v___x_407_);
return v___x_408_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2(void){
_start:
{
lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_410_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__1));
v___x_411_ = l_Lean_stringToMessageData(v___x_410_);
return v___x_411_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__3(void){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_412_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__2);
v___x_413_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__0);
v___x_414_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_414_, 0, v___x_413_);
lean_ctor_set(v___x_414_, 1, v___x_412_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1(lean_object* v_msg_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_){
_start:
{
lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; 
v___x_421_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__3, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__3_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___closed__3);
v___x_422_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
lean_ctor_set(v___x_422_, 1, v_msg_415_);
v___x_423_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1___boxed(lean_object* v_msg_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__1(v_msg_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
return v_res_430_;
}
}
static double _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2(void){
_start:
{
lean_object* v___x_434_; double v___x_435_; 
v___x_434_ = lean_unsigned_to_nat(1000000000u);
v___x_435_ = lean_float_of_nat(v___x_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2(lean_object* v_val_436_, lean_object* v___x_437_, lean_object* v_traceClass_438_, lean_object* v___x_439_, lean_object* v___x_440_, lean_object* v_toMonadRef_441_, lean_object* v___x_442_, lean_object* v___x_443_, lean_object* v___f_444_, uint8_t v_a_445_, lean_object* v___x_446_, lean_object* v___f_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_){
_start:
{
lean_object* v_options_453_; uint8_t v_hasTrace_454_; 
v_options_453_ = lean_ctor_get(v___y_450_, 2);
v_hasTrace_454_ = lean_ctor_get_uint8(v_options_453_, sizeof(void*)*1);
if (v_hasTrace_454_ == 0)
{
lean_object* v___x_455_; 
lean_dec_ref(v___f_447_);
lean_dec_ref(v___x_446_);
lean_dec_ref(v___f_444_);
lean_dec_ref(v___x_443_);
lean_dec_ref(v___x_442_);
lean_dec_ref(v_toMonadRef_441_);
lean_dec_ref(v___x_440_);
lean_dec_ref(v___x_439_);
lean_dec(v_traceClass_438_);
v___x_455_ = lp_aesop_Aesop_Rapp_traceMetadata(v_val_436_, v___x_437_, v___y_448_, v___y_449_, v___y_450_, v___y_451_);
lean_dec(v___y_451_);
lean_dec_ref(v___y_450_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
return v___x_455_;
}
else
{
lean_object* v_inheritedTraceOptions_456_; lean_object* v___x_457_; lean_object* v___x_458_; uint8_t v___x_459_; lean_object* v___y_461_; lean_object* v___y_462_; lean_object* v_a_463_; lean_object* v___y_477_; lean_object* v___y_478_; lean_object* v_a_479_; 
v_inheritedTraceOptions_456_ = lean_ctor_get(v___y_450_, 13);
v___x_457_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__1));
lean_inc(v_traceClass_438_);
v___x_458_ = l_Lean_Name_append(v___x_457_, v_traceClass_438_);
v___x_459_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_456_, v_options_453_, v___x_458_);
lean_dec(v___x_458_);
if (v___x_459_ == 0)
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; uint8_t v___x_544_; 
v___x_541_ = l_Lean_KVMap_instValueBool;
v___x_542_ = l_Lean_trace_profiler;
v___x_543_ = l_Lean_Option_get___redArg(v___x_541_, v_options_453_, v___x_542_);
v___x_544_ = lean_unbox(v___x_543_);
lean_dec(v___x_543_);
if (v___x_544_ == 0)
{
lean_object* v___x_545_; 
lean_dec_ref(v___f_447_);
lean_dec_ref(v___x_446_);
lean_dec_ref(v___f_444_);
lean_dec_ref(v___x_443_);
lean_dec_ref(v___x_442_);
lean_dec_ref(v_toMonadRef_441_);
lean_dec_ref(v___x_440_);
lean_dec_ref(v___x_439_);
lean_dec(v_traceClass_438_);
v___x_545_ = lp_aesop_Aesop_Rapp_traceMetadata(v_val_436_, v___x_437_, v___y_448_, v___y_449_, v___y_450_, v___y_451_);
lean_dec(v___y_451_);
lean_dec_ref(v___y_450_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
return v___x_545_;
}
else
{
goto v___jp_489_;
}
}
else
{
goto v___jp_489_;
}
v___jp_460_:
{
lean_object* v___x_464_; double v___x_465_; double v___x_466_; double v___x_467_; double v___x_468_; double v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_27604__overap_474_; lean_object* v___x_475_; 
v___x_464_ = lean_io_mono_nanos_now();
v___x_465_ = lean_float_of_nat(v___y_461_);
v___x_466_ = lean_float_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2);
v___x_467_ = lean_float_div(v___x_465_, v___x_466_);
v___x_468_ = lean_float_of_nat(v___x_464_);
v___x_469_ = lean_float_div(v___x_468_, v___x_466_);
v___x_470_ = lean_box_float(v___x_467_);
v___x_471_ = lean_box_float(v___x_469_);
v___x_472_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_472_, 0, v___x_470_);
lean_ctor_set(v___x_472_, 1, v___x_471_);
v___x_473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_473_, 0, v_a_463_);
lean_ctor_set(v___x_473_, 1, v___x_472_);
v___x_27604__overap_474_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_439_, v___x_440_, v_toMonadRef_441_, v___x_442_, lean_box(0), v___x_443_, v___f_444_, v_traceClass_438_, v_a_445_, v___x_446_, v_options_453_, v___x_459_, v___y_462_, v___f_447_, v___x_473_);
v___x_475_ = lean_apply_5(v___x_27604__overap_474_, v___y_448_, v___y_449_, v___y_450_, v___y_451_, lean_box(0));
return v___x_475_;
}
v___jp_476_:
{
lean_object* v___x_480_; double v___x_481_; double v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_27587__overap_487_; lean_object* v___x_488_; 
v___x_480_ = lean_io_get_num_heartbeats();
v___x_481_ = lean_float_of_nat(v___y_478_);
v___x_482_ = lean_float_of_nat(v___x_480_);
v___x_483_ = lean_box_float(v___x_481_);
v___x_484_ = lean_box_float(v___x_482_);
v___x_485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_485_, 0, v___x_483_);
lean_ctor_set(v___x_485_, 1, v___x_484_);
v___x_486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_486_, 0, v_a_479_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
v___x_27587__overap_487_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_439_, v___x_440_, v_toMonadRef_441_, v___x_442_, lean_box(0), v___x_443_, v___f_444_, v_traceClass_438_, v_a_445_, v___x_446_, v_options_453_, v___x_459_, v___y_477_, v___f_447_, v___x_486_);
v___x_488_ = lean_apply_5(v___x_27587__overap_487_, v___y_448_, v___y_449_, v___y_450_, v___y_451_, lean_box(0));
return v___x_488_;
}
v___jp_489_:
{
lean_object* v___x_27606__overap_490_; lean_object* v___x_491_; 
lean_inc_ref(v___x_440_);
lean_inc_ref(v___x_439_);
v___x_27606__overap_490_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_439_, v___x_440_);
lean_inc(v___y_451_);
lean_inc_ref(v___y_450_);
lean_inc(v___y_449_);
lean_inc_ref(v___y_448_);
v___x_491_ = lean_apply_5(v___x_27606__overap_490_, v___y_448_, v___y_449_, v___y_450_, v___y_451_, lean_box(0));
if (lean_obj_tag(v___x_491_) == 0)
{
lean_object* v_a_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; uint8_t v___x_496_; 
v_a_492_ = lean_ctor_get(v___x_491_, 0);
lean_inc(v_a_492_);
lean_dec_ref_known(v___x_491_, 1);
v___x_493_ = l_Lean_KVMap_instValueBool;
v___x_494_ = l_Lean_trace_profiler_useHeartbeats;
v___x_495_ = l_Lean_Option_get___redArg(v___x_493_, v_options_453_, v___x_494_);
v___x_496_ = lean_unbox(v___x_495_);
lean_dec(v___x_495_);
if (v___x_496_ == 0)
{
lean_object* v___x_497_; lean_object* v___x_498_; 
v___x_497_ = lean_io_mono_nanos_now();
v___x_498_ = lp_aesop_Aesop_Rapp_traceMetadata(v_val_436_, v___x_437_, v___y_448_, v___y_449_, v___y_450_, v___y_451_);
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v_a_499_; lean_object* v___x_501_; uint8_t v_isShared_502_; uint8_t v_isSharedCheck_506_; 
v_a_499_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_506_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_506_ == 0)
{
v___x_501_ = v___x_498_;
v_isShared_502_ = v_isSharedCheck_506_;
goto v_resetjp_500_;
}
else
{
lean_inc(v_a_499_);
lean_dec(v___x_498_);
v___x_501_ = lean_box(0);
v_isShared_502_ = v_isSharedCheck_506_;
goto v_resetjp_500_;
}
v_resetjp_500_:
{
lean_object* v___x_504_; 
if (v_isShared_502_ == 0)
{
lean_ctor_set_tag(v___x_501_, 1);
v___x_504_ = v___x_501_;
goto v_reusejp_503_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v_a_499_);
v___x_504_ = v_reuseFailAlloc_505_;
goto v_reusejp_503_;
}
v_reusejp_503_:
{
v___y_461_ = v___x_497_;
v___y_462_ = v_a_492_;
v_a_463_ = v___x_504_;
goto v___jp_460_;
}
}
}
else
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_514_; 
v_a_507_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_514_ == 0)
{
v___x_509_ = v___x_498_;
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_498_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_512_; 
if (v_isShared_510_ == 0)
{
lean_ctor_set_tag(v___x_509_, 0);
v___x_512_ = v___x_509_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_a_507_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
v___y_461_ = v___x_497_;
v___y_462_ = v_a_492_;
v_a_463_ = v___x_512_;
goto v___jp_460_;
}
}
}
}
else
{
lean_object* v___x_515_; lean_object* v___x_516_; 
v___x_515_ = lean_io_get_num_heartbeats();
v___x_516_ = lp_aesop_Aesop_Rapp_traceMetadata(v_val_436_, v___x_437_, v___y_448_, v___y_449_, v___y_450_, v___y_451_);
if (lean_obj_tag(v___x_516_) == 0)
{
lean_object* v_a_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_524_; 
v_a_517_ = lean_ctor_get(v___x_516_, 0);
v_isSharedCheck_524_ = !lean_is_exclusive(v___x_516_);
if (v_isSharedCheck_524_ == 0)
{
v___x_519_ = v___x_516_;
v_isShared_520_ = v_isSharedCheck_524_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_a_517_);
lean_dec(v___x_516_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_524_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v___x_522_; 
if (v_isShared_520_ == 0)
{
lean_ctor_set_tag(v___x_519_, 1);
v___x_522_ = v___x_519_;
goto v_reusejp_521_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v_a_517_);
v___x_522_ = v_reuseFailAlloc_523_;
goto v_reusejp_521_;
}
v_reusejp_521_:
{
v___y_477_ = v_a_492_;
v___y_478_ = v___x_515_;
v_a_479_ = v___x_522_;
goto v___jp_476_;
}
}
}
else
{
lean_object* v_a_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_532_; 
v_a_525_ = lean_ctor_get(v___x_516_, 0);
v_isSharedCheck_532_ = !lean_is_exclusive(v___x_516_);
if (v_isSharedCheck_532_ == 0)
{
v___x_527_ = v___x_516_;
v_isShared_528_ = v_isSharedCheck_532_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_a_525_);
lean_dec(v___x_516_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_532_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
lean_object* v___x_530_; 
if (v_isShared_528_ == 0)
{
lean_ctor_set_tag(v___x_527_, 0);
v___x_530_ = v___x_527_;
goto v_reusejp_529_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v_a_525_);
v___x_530_ = v_reuseFailAlloc_531_;
goto v_reusejp_529_;
}
v_reusejp_529_:
{
v___y_477_ = v_a_492_;
v___y_478_ = v___x_515_;
v_a_479_ = v___x_530_;
goto v___jp_476_;
}
}
}
}
}
else
{
lean_object* v_a_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_540_; 
lean_dec(v___y_451_);
lean_dec_ref(v___y_450_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec_ref(v___f_447_);
lean_dec_ref(v___x_446_);
lean_dec_ref(v___f_444_);
lean_dec_ref(v___x_443_);
lean_dec_ref(v___x_442_);
lean_dec_ref(v_toMonadRef_441_);
lean_dec_ref(v___x_440_);
lean_dec_ref(v___x_439_);
lean_dec(v_traceClass_438_);
lean_dec_ref(v___x_437_);
lean_dec(v_val_436_);
v_a_533_ = lean_ctor_get(v___x_491_, 0);
v_isSharedCheck_540_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_540_ == 0)
{
v___x_535_ = v___x_491_;
v_isShared_536_ = v_isSharedCheck_540_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_a_533_);
lean_dec(v___x_491_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_540_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v___x_538_; 
if (v_isShared_536_ == 0)
{
v___x_538_ = v___x_535_;
goto v_reusejp_537_;
}
else
{
lean_object* v_reuseFailAlloc_539_; 
v_reuseFailAlloc_539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_539_, 0, v_a_533_);
v___x_538_ = v_reuseFailAlloc_539_;
goto v_reusejp_537_;
}
v_reusejp_537_:
{
return v___x_538_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___boxed(lean_object** _args){
lean_object* v_val_546_ = _args[0];
lean_object* v___x_547_ = _args[1];
lean_object* v_traceClass_548_ = _args[2];
lean_object* v___x_549_ = _args[3];
lean_object* v___x_550_ = _args[4];
lean_object* v_toMonadRef_551_ = _args[5];
lean_object* v___x_552_ = _args[6];
lean_object* v___x_553_ = _args[7];
lean_object* v___f_554_ = _args[8];
lean_object* v_a_555_ = _args[9];
lean_object* v___x_556_ = _args[10];
lean_object* v___f_557_ = _args[11];
lean_object* v___y_558_ = _args[12];
lean_object* v___y_559_ = _args[13];
lean_object* v___y_560_ = _args[14];
lean_object* v___y_561_ = _args[15];
lean_object* v___y_562_ = _args[16];
_start:
{
uint8_t v_a_27947__boxed_563_; lean_object* v_res_564_; 
v_a_27947__boxed_563_ = lean_unbox(v_a_555_);
v_res_564_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2(v_val_546_, v___x_547_, v_traceClass_548_, v___x_549_, v___x_550_, v_toMonadRef_551_, v___x_552_, v___x_553_, v___f_554_, v_a_27947__boxed_563_, v___x_556_, v___f_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3(lean_object* v___x_566_, lean_object* v___x_567_, lean_object* v_toMonadRef_568_, lean_object* v___x_569_, lean_object* v_traceClass_570_, lean_object* v___x_571_, lean_object* v_val_572_, lean_object* v___x_573_, lean_object* v___x_574_, lean_object* v___f_575_, uint8_t v_a_576_, lean_object* v___f_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_){
_start:
{
lean_object* v___x_27654__overap_583_; lean_object* v___x_584_; 
lean_inc(v_traceClass_570_);
lean_inc_ref(v___x_569_);
lean_inc_ref(v_toMonadRef_568_);
lean_inc_ref(v___x_567_);
lean_inc_ref(v___x_566_);
v___x_27654__overap_583_ = l_Lean_addTrace___redArg(v___x_566_, v___x_567_, v_toMonadRef_568_, v___x_569_, v_traceClass_570_, v___x_571_);
lean_inc(v___y_581_);
lean_inc_ref(v___y_580_);
lean_inc(v___y_579_);
lean_inc_ref(v___y_578_);
v___x_584_ = lean_apply_5(v___x_27654__overap_583_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, lean_box(0));
if (lean_obj_tag(v___x_584_) == 0)
{
lean_object* v_options_585_; uint8_t v_hasTrace_586_; 
lean_dec_ref_known(v___x_584_, 1);
v_options_585_ = lean_ctor_get(v___y_580_, 2);
v_hasTrace_586_ = lean_ctor_get_uint8(v_options_585_, sizeof(void*)*1);
if (v_hasTrace_586_ == 0)
{
lean_object* v___x_587_; 
lean_dec_ref(v___f_577_);
lean_dec_ref(v___f_575_);
lean_dec_ref(v___x_574_);
lean_dec(v_traceClass_570_);
lean_dec_ref(v___x_569_);
lean_dec_ref(v_toMonadRef_568_);
lean_dec_ref(v___x_567_);
lean_dec_ref(v___x_566_);
v___x_587_ = lp_aesop_Aesop_Goal_traceMetadata(v_val_572_, v___x_573_, v___y_578_, v___y_579_, v___y_580_, v___y_581_);
lean_dec(v___y_581_);
lean_dec_ref(v___y_580_);
lean_dec(v___y_579_);
lean_dec_ref(v___y_578_);
return v___x_587_;
}
else
{
lean_object* v_inheritedTraceOptions_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; uint8_t v___x_592_; lean_object* v___y_594_; lean_object* v___y_595_; lean_object* v_a_596_; lean_object* v___y_610_; lean_object* v___y_611_; lean_object* v_a_612_; 
v_inheritedTraceOptions_588_ = lean_ctor_get(v___y_580_, 13);
v___x_589_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0));
v___x_590_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__1));
lean_inc(v_traceClass_570_);
v___x_591_ = l_Lean_Name_append(v___x_590_, v_traceClass_570_);
v___x_592_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_588_, v_options_585_, v___x_591_);
lean_dec(v___x_591_);
if (v___x_592_ == 0)
{
lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; uint8_t v___x_677_; 
v___x_674_ = l_Lean_KVMap_instValueBool;
v___x_675_ = l_Lean_trace_profiler;
v___x_676_ = l_Lean_Option_get___redArg(v___x_674_, v_options_585_, v___x_675_);
v___x_677_ = lean_unbox(v___x_676_);
lean_dec(v___x_676_);
if (v___x_677_ == 0)
{
lean_object* v___x_678_; 
lean_dec_ref(v___f_577_);
lean_dec_ref(v___f_575_);
lean_dec_ref(v___x_574_);
lean_dec(v_traceClass_570_);
lean_dec_ref(v___x_569_);
lean_dec_ref(v_toMonadRef_568_);
lean_dec_ref(v___x_567_);
lean_dec_ref(v___x_566_);
v___x_678_ = lp_aesop_Aesop_Goal_traceMetadata(v_val_572_, v___x_573_, v___y_578_, v___y_579_, v___y_580_, v___y_581_);
lean_dec(v___y_581_);
lean_dec_ref(v___y_580_);
lean_dec(v___y_579_);
lean_dec_ref(v___y_578_);
return v___x_678_;
}
else
{
goto v___jp_622_;
}
}
else
{
goto v___jp_622_;
}
v___jp_593_:
{
lean_object* v___x_597_; double v___x_598_; double v___x_599_; double v___x_600_; double v___x_601_; double v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_27692__overap_607_; lean_object* v___x_608_; 
v___x_597_ = lean_io_mono_nanos_now();
v___x_598_ = lean_float_of_nat(v___y_594_);
v___x_599_ = lean_float_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2);
v___x_600_ = lean_float_div(v___x_598_, v___x_599_);
v___x_601_ = lean_float_of_nat(v___x_597_);
v___x_602_ = lean_float_div(v___x_601_, v___x_599_);
v___x_603_ = lean_box_float(v___x_600_);
v___x_604_ = lean_box_float(v___x_602_);
v___x_605_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_603_);
lean_ctor_set(v___x_605_, 1, v___x_604_);
v___x_606_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_606_, 0, v_a_596_);
lean_ctor_set(v___x_606_, 1, v___x_605_);
v___x_27692__overap_607_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_566_, v___x_567_, v_toMonadRef_568_, v___x_569_, lean_box(0), v___x_574_, v___f_575_, v_traceClass_570_, v_a_576_, v___x_589_, v_options_585_, v___x_592_, v___y_595_, v___f_577_, v___x_606_);
v___x_608_ = lean_apply_5(v___x_27692__overap_607_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, lean_box(0));
return v___x_608_;
}
v___jp_609_:
{
lean_object* v___x_613_; double v___x_614_; double v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_27675__overap_620_; lean_object* v___x_621_; 
v___x_613_ = lean_io_get_num_heartbeats();
v___x_614_ = lean_float_of_nat(v___y_611_);
v___x_615_ = lean_float_of_nat(v___x_613_);
v___x_616_ = lean_box_float(v___x_614_);
v___x_617_ = lean_box_float(v___x_615_);
v___x_618_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_618_, 0, v___x_616_);
lean_ctor_set(v___x_618_, 1, v___x_617_);
v___x_619_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_619_, 0, v_a_612_);
lean_ctor_set(v___x_619_, 1, v___x_618_);
v___x_27675__overap_620_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_566_, v___x_567_, v_toMonadRef_568_, v___x_569_, lean_box(0), v___x_574_, v___f_575_, v_traceClass_570_, v_a_576_, v___x_589_, v_options_585_, v___x_592_, v___y_610_, v___f_577_, v___x_619_);
v___x_621_ = lean_apply_5(v___x_27675__overap_620_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, lean_box(0));
return v___x_621_;
}
v___jp_622_:
{
lean_object* v___x_27694__overap_623_; lean_object* v___x_624_; 
lean_inc_ref(v___x_567_);
lean_inc_ref(v___x_566_);
v___x_27694__overap_623_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_566_, v___x_567_);
lean_inc(v___y_581_);
lean_inc_ref(v___y_580_);
lean_inc(v___y_579_);
lean_inc_ref(v___y_578_);
v___x_624_ = lean_apply_5(v___x_27694__overap_623_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, lean_box(0));
if (lean_obj_tag(v___x_624_) == 0)
{
lean_object* v_a_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; uint8_t v___x_629_; 
v_a_625_ = lean_ctor_get(v___x_624_, 0);
lean_inc(v_a_625_);
lean_dec_ref_known(v___x_624_, 1);
v___x_626_ = l_Lean_KVMap_instValueBool;
v___x_627_ = l_Lean_trace_profiler_useHeartbeats;
v___x_628_ = l_Lean_Option_get___redArg(v___x_626_, v_options_585_, v___x_627_);
v___x_629_ = lean_unbox(v___x_628_);
lean_dec(v___x_628_);
if (v___x_629_ == 0)
{
lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_630_ = lean_io_mono_nanos_now();
v___x_631_ = lp_aesop_Aesop_Goal_traceMetadata(v_val_572_, v___x_573_, v___y_578_, v___y_579_, v___y_580_, v___y_581_);
if (lean_obj_tag(v___x_631_) == 0)
{
lean_object* v_a_632_; lean_object* v___x_634_; uint8_t v_isShared_635_; uint8_t v_isSharedCheck_639_; 
v_a_632_ = lean_ctor_get(v___x_631_, 0);
v_isSharedCheck_639_ = !lean_is_exclusive(v___x_631_);
if (v_isSharedCheck_639_ == 0)
{
v___x_634_ = v___x_631_;
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
else
{
lean_inc(v_a_632_);
lean_dec(v___x_631_);
v___x_634_ = lean_box(0);
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
v_resetjp_633_:
{
lean_object* v___x_637_; 
if (v_isShared_635_ == 0)
{
lean_ctor_set_tag(v___x_634_, 1);
v___x_637_ = v___x_634_;
goto v_reusejp_636_;
}
else
{
lean_object* v_reuseFailAlloc_638_; 
v_reuseFailAlloc_638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_638_, 0, v_a_632_);
v___x_637_ = v_reuseFailAlloc_638_;
goto v_reusejp_636_;
}
v_reusejp_636_:
{
v___y_594_ = v___x_630_;
v___y_595_ = v_a_625_;
v_a_596_ = v___x_637_;
goto v___jp_593_;
}
}
}
else
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
v_a_640_ = lean_ctor_get(v___x_631_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_631_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_631_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_631_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_645_; 
if (v_isShared_643_ == 0)
{
lean_ctor_set_tag(v___x_642_, 0);
v___x_645_ = v___x_642_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v_a_640_);
v___x_645_ = v_reuseFailAlloc_646_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
v___y_594_ = v___x_630_;
v___y_595_ = v_a_625_;
v_a_596_ = v___x_645_;
goto v___jp_593_;
}
}
}
}
else
{
lean_object* v___x_648_; lean_object* v___x_649_; 
v___x_648_ = lean_io_get_num_heartbeats();
v___x_649_ = lp_aesop_Aesop_Goal_traceMetadata(v_val_572_, v___x_573_, v___y_578_, v___y_579_, v___y_580_, v___y_581_);
if (lean_obj_tag(v___x_649_) == 0)
{
lean_object* v_a_650_; lean_object* v___x_652_; uint8_t v_isShared_653_; uint8_t v_isSharedCheck_657_; 
v_a_650_ = lean_ctor_get(v___x_649_, 0);
v_isSharedCheck_657_ = !lean_is_exclusive(v___x_649_);
if (v_isSharedCheck_657_ == 0)
{
v___x_652_ = v___x_649_;
v_isShared_653_ = v_isSharedCheck_657_;
goto v_resetjp_651_;
}
else
{
lean_inc(v_a_650_);
lean_dec(v___x_649_);
v___x_652_ = lean_box(0);
v_isShared_653_ = v_isSharedCheck_657_;
goto v_resetjp_651_;
}
v_resetjp_651_:
{
lean_object* v___x_655_; 
if (v_isShared_653_ == 0)
{
lean_ctor_set_tag(v___x_652_, 1);
v___x_655_ = v___x_652_;
goto v_reusejp_654_;
}
else
{
lean_object* v_reuseFailAlloc_656_; 
v_reuseFailAlloc_656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_656_, 0, v_a_650_);
v___x_655_ = v_reuseFailAlloc_656_;
goto v_reusejp_654_;
}
v_reusejp_654_:
{
v___y_610_ = v_a_625_;
v___y_611_ = v___x_648_;
v_a_612_ = v___x_655_;
goto v___jp_609_;
}
}
}
else
{
lean_object* v_a_658_; lean_object* v___x_660_; uint8_t v_isShared_661_; uint8_t v_isSharedCheck_665_; 
v_a_658_ = lean_ctor_get(v___x_649_, 0);
v_isSharedCheck_665_ = !lean_is_exclusive(v___x_649_);
if (v_isSharedCheck_665_ == 0)
{
v___x_660_ = v___x_649_;
v_isShared_661_ = v_isSharedCheck_665_;
goto v_resetjp_659_;
}
else
{
lean_inc(v_a_658_);
lean_dec(v___x_649_);
v___x_660_ = lean_box(0);
v_isShared_661_ = v_isSharedCheck_665_;
goto v_resetjp_659_;
}
v_resetjp_659_:
{
lean_object* v___x_663_; 
if (v_isShared_661_ == 0)
{
lean_ctor_set_tag(v___x_660_, 0);
v___x_663_ = v___x_660_;
goto v_reusejp_662_;
}
else
{
lean_object* v_reuseFailAlloc_664_; 
v_reuseFailAlloc_664_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_664_, 0, v_a_658_);
v___x_663_ = v_reuseFailAlloc_664_;
goto v_reusejp_662_;
}
v_reusejp_662_:
{
v___y_610_ = v_a_625_;
v___y_611_ = v___x_648_;
v_a_612_ = v___x_663_;
goto v___jp_609_;
}
}
}
}
}
else
{
lean_object* v_a_666_; lean_object* v___x_668_; uint8_t v_isShared_669_; uint8_t v_isSharedCheck_673_; 
lean_dec(v___y_581_);
lean_dec_ref(v___y_580_);
lean_dec(v___y_579_);
lean_dec_ref(v___y_578_);
lean_dec_ref(v___f_577_);
lean_dec_ref(v___f_575_);
lean_dec_ref(v___x_574_);
lean_dec_ref(v___x_573_);
lean_dec(v_val_572_);
lean_dec(v_traceClass_570_);
lean_dec_ref(v___x_569_);
lean_dec_ref(v_toMonadRef_568_);
lean_dec_ref(v___x_567_);
lean_dec_ref(v___x_566_);
v_a_666_ = lean_ctor_get(v___x_624_, 0);
v_isSharedCheck_673_ = !lean_is_exclusive(v___x_624_);
if (v_isSharedCheck_673_ == 0)
{
v___x_668_ = v___x_624_;
v_isShared_669_ = v_isSharedCheck_673_;
goto v_resetjp_667_;
}
else
{
lean_inc(v_a_666_);
lean_dec(v___x_624_);
v___x_668_ = lean_box(0);
v_isShared_669_ = v_isSharedCheck_673_;
goto v_resetjp_667_;
}
v_resetjp_667_:
{
lean_object* v___x_671_; 
if (v_isShared_669_ == 0)
{
v___x_671_ = v___x_668_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v_a_666_);
v___x_671_ = v_reuseFailAlloc_672_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
return v___x_671_;
}
}
}
}
}
}
else
{
lean_dec(v___y_581_);
lean_dec_ref(v___y_580_);
lean_dec(v___y_579_);
lean_dec_ref(v___y_578_);
lean_dec_ref(v___f_577_);
lean_dec_ref(v___f_575_);
lean_dec_ref(v___x_574_);
lean_dec_ref(v___x_573_);
lean_dec(v_val_572_);
lean_dec(v_traceClass_570_);
lean_dec_ref(v___x_569_);
lean_dec_ref(v_toMonadRef_568_);
lean_dec_ref(v___x_567_);
lean_dec_ref(v___x_566_);
return v___x_584_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed(lean_object** _args){
lean_object* v___x_679_ = _args[0];
lean_object* v___x_680_ = _args[1];
lean_object* v_toMonadRef_681_ = _args[2];
lean_object* v___x_682_ = _args[3];
lean_object* v_traceClass_683_ = _args[4];
lean_object* v___x_684_ = _args[5];
lean_object* v_val_685_ = _args[6];
lean_object* v___x_686_ = _args[7];
lean_object* v___x_687_ = _args[8];
lean_object* v___f_688_ = _args[9];
lean_object* v_a_689_ = _args[10];
lean_object* v___f_690_ = _args[11];
lean_object* v___y_691_ = _args[12];
lean_object* v___y_692_ = _args[13];
lean_object* v___y_693_ = _args[14];
lean_object* v___y_694_ = _args[15];
lean_object* v___y_695_ = _args[16];
_start:
{
uint8_t v_a_28179__boxed_696_; lean_object* v_res_697_; 
v_a_28179__boxed_696_ = lean_unbox(v_a_689_);
v_res_697_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3(v___x_679_, v___x_680_, v_toMonadRef_681_, v___x_682_, v_traceClass_683_, v___x_684_, v_val_685_, v___x_686_, v___x_687_, v___f_688_, v_a_28179__boxed_696_, v___f_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_);
return v_res_697_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4(lean_object* v___x_698_, lean_object* v___x_699_, lean_object* v___x_700_, lean_object* v_toMonadRef_701_, lean_object* v___x_702_, lean_object* v___x_703_, lean_object* v___f_704_, uint8_t v_a_705_, lean_object* v___f_706_, lean_object* v___f_707_, lean_object* v_gref_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_){
_start:
{
lean_object* v___x_714_; lean_object* v_traceClass_715_; lean_object* v___x_716_; lean_object* v_elimGoal_717_; lean_object* v___x_718_; lean_object* v_preNormGoal_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___f_722_; lean_object* v___x_723_; 
v___x_714_ = lean_st_ref_get(v_gref_708_);
v_traceClass_715_ = lean_ctor_get(v___x_698_, 0);
v___x_716_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_717_ = lean_ctor_get(v___x_716_, 1);
lean_inc_ref(v_elimGoal_717_);
lean_inc_n(v___x_714_, 2);
v___x_718_ = lean_apply_1(v_elimGoal_717_, v___x_714_);
v_preNormGoal_719_ = lean_ctor_get(v___x_718_, 5);
lean_inc(v_preNormGoal_719_);
lean_dec_ref(v___x_718_);
v___x_720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_720_, 0, v_preNormGoal_719_);
v___x_721_ = lean_box(v_a_705_);
lean_inc_ref(v___x_698_);
lean_inc(v_traceClass_715_);
v___f_722_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___boxed), 17, 12);
lean_closure_set(v___f_722_, 0, v___x_699_);
lean_closure_set(v___f_722_, 1, v___x_700_);
lean_closure_set(v___f_722_, 2, v_toMonadRef_701_);
lean_closure_set(v___f_722_, 3, v___x_702_);
lean_closure_set(v___f_722_, 4, v_traceClass_715_);
lean_closure_set(v___f_722_, 5, v___x_720_);
lean_closure_set(v___f_722_, 6, v___x_714_);
lean_closure_set(v___f_722_, 7, v___x_698_);
lean_closure_set(v___f_722_, 8, v___x_703_);
lean_closure_set(v___f_722_, 9, v___f_704_);
lean_closure_set(v___f_722_, 10, v___x_721_);
lean_closure_set(v___f_722_, 11, v___f_706_);
v___x_723_ = lp_aesop_Aesop_Goal_withHeadlineTraceNode___redArg(v___x_714_, v___x_698_, v___f_722_, v_a_705_, v___f_707_, v___y_709_, v___y_710_, v___y_711_, v___y_712_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___boxed(lean_object* v___x_724_, lean_object* v___x_725_, lean_object* v___x_726_, lean_object* v_toMonadRef_727_, lean_object* v___x_728_, lean_object* v___x_729_, lean_object* v___f_730_, lean_object* v_a_731_, lean_object* v___f_732_, lean_object* v___f_733_, lean_object* v_gref_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_){
_start:
{
uint8_t v_a_28402__boxed_740_; lean_object* v_res_741_; 
v_a_28402__boxed_740_ = lean_unbox(v_a_731_);
v_res_741_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4(v___x_724_, v___x_725_, v___x_726_, v_toMonadRef_727_, v___x_728_, v___x_729_, v___f_730_, v_a_28402__boxed_740_, v___f_732_, v___f_733_, v_gref_734_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
lean_dec(v___y_738_);
lean_dec_ref(v___y_737_);
lean_dec(v___y_736_);
lean_dec_ref(v___y_735_);
lean_dec(v_gref_734_);
return v_res_741_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__0(void){
_start:
{
lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; 
v___x_742_ = l_Lean_Core_instMonadTraceCoreM;
v___x_743_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_744_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_743_, v___x_742_);
return v___x_744_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1(void){
_start:
{
lean_object* v___x_745_; lean_object* v___f_746_; lean_object* v___x_747_; 
v___x_745_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__0, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__0_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__0);
v___f_746_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___x_747_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_746_, v___x_745_);
return v___x_747_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__4(void){
_start:
{
lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; 
v___x_750_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_751_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_752_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3));
v___x_753_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_752_, v___x_751_, v___x_750_);
return v___x_753_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5(void){
_start:
{
lean_object* v___x_754_; lean_object* v___f_755_; lean_object* v___f_756_; lean_object* v___x_757_; 
v___x_754_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__4, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__4_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__4);
v___f_755_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___f_756_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__2));
v___x_757_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_756_, v___f_755_, v___x_754_);
return v___x_757_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__6(void){
_start:
{
lean_object* v___x_758_; 
v___x_758_ = l_instMonadExceptOfEIO(lean_box(0));
return v___x_758_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__7(void){
_start:
{
lean_object* v___x_759_; lean_object* v___x_760_; 
v___x_759_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__6, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__6_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__6);
v___x_760_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_759_);
return v___x_760_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__8(void){
_start:
{
lean_object* v___x_761_; lean_object* v___x_762_; 
v___x_761_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__7, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__7_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__7);
v___x_762_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_761_);
return v___x_762_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__9(void){
_start:
{
lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_763_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__8, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__8_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__8);
v___x_764_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_763_);
return v___x_764_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10(void){
_start:
{
lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_765_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__9, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__9_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__9);
v___x_766_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_765_);
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5(lean_object* v___x_787_, uint8_t v_a_788_, lean_object* v___f_789_, lean_object* v___f_790_, lean_object* v___x_791_, lean_object* v_a_792_, lean_object* v_x_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_){
_start:
{
lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v_toApplicative_807_; lean_object* v_toFunctor_808_; lean_object* v_toSeq_809_; lean_object* v_toSeqLeft_810_; lean_object* v_toSeqRight_811_; lean_object* v___f_812_; lean_object* v___f_813_; lean_object* v___f_814_; lean_object* v___f_815_; lean_object* v___x_816_; lean_object* v___f_817_; lean_object* v___f_818_; lean_object* v___f_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v_toApplicative_823_; lean_object* v___x_825_; uint8_t v_isShared_826_; uint8_t v_isSharedCheck_897_; 
v___x_804_ = lean_st_ref_get(v___y_796_);
lean_dec(v___x_804_);
v___x_805_ = lean_st_ref_get(v_a_792_);
v___x_806_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1);
v_toApplicative_807_ = lean_ctor_get(v___x_806_, 0);
v_toFunctor_808_ = lean_ctor_get(v_toApplicative_807_, 0);
v_toSeq_809_ = lean_ctor_get(v_toApplicative_807_, 2);
v_toSeqLeft_810_ = lean_ctor_get(v_toApplicative_807_, 3);
v_toSeqRight_811_ = lean_ctor_get(v_toApplicative_807_, 4);
v___f_812_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2));
v___f_813_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3));
lean_inc_ref_n(v_toFunctor_808_, 2);
v___f_814_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_814_, 0, v_toFunctor_808_);
v___f_815_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_815_, 0, v_toFunctor_808_);
v___x_816_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_816_, 0, v___f_814_);
lean_ctor_set(v___x_816_, 1, v___f_815_);
lean_inc(v_toSeqRight_811_);
v___f_817_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_817_, 0, v_toSeqRight_811_);
lean_inc(v_toSeqLeft_810_);
v___f_818_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_818_, 0, v_toSeqLeft_810_);
lean_inc(v_toSeq_809_);
v___f_819_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_819_, 0, v_toSeq_809_);
v___x_820_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_820_, 0, v___x_816_);
lean_ctor_set(v___x_820_, 1, v___f_812_);
lean_ctor_set(v___x_820_, 2, v___f_819_);
lean_ctor_set(v___x_820_, 3, v___f_818_);
lean_ctor_set(v___x_820_, 4, v___f_817_);
v___x_821_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_821_, 0, v___x_820_);
lean_ctor_set(v___x_821_, 1, v___f_813_);
v___x_822_ = l_StateRefT_x27_instMonad___redArg(v___x_821_);
v_toApplicative_823_ = lean_ctor_get(v___x_822_, 0);
v_isSharedCheck_897_ = !lean_is_exclusive(v___x_822_);
if (v_isSharedCheck_897_ == 0)
{
lean_object* v_unused_898_; 
v_unused_898_ = lean_ctor_get(v___x_822_, 1);
lean_dec(v_unused_898_);
v___x_825_ = v___x_822_;
v_isShared_826_ = v_isSharedCheck_897_;
goto v_resetjp_824_;
}
else
{
lean_inc(v_toApplicative_823_);
lean_dec(v___x_822_);
v___x_825_ = lean_box(0);
v_isShared_826_ = v_isSharedCheck_897_;
goto v_resetjp_824_;
}
v_resetjp_824_:
{
lean_object* v_toFunctor_827_; lean_object* v_toSeq_828_; lean_object* v_toSeqLeft_829_; lean_object* v_toSeqRight_830_; lean_object* v___x_832_; uint8_t v_isShared_833_; uint8_t v_isSharedCheck_895_; 
v_toFunctor_827_ = lean_ctor_get(v_toApplicative_823_, 0);
v_toSeq_828_ = lean_ctor_get(v_toApplicative_823_, 2);
v_toSeqLeft_829_ = lean_ctor_get(v_toApplicative_823_, 3);
v_toSeqRight_830_ = lean_ctor_get(v_toApplicative_823_, 4);
v_isSharedCheck_895_ = !lean_is_exclusive(v_toApplicative_823_);
if (v_isSharedCheck_895_ == 0)
{
lean_object* v_unused_896_; 
v_unused_896_ = lean_ctor_get(v_toApplicative_823_, 1);
lean_dec(v_unused_896_);
v___x_832_ = v_toApplicative_823_;
v_isShared_833_ = v_isSharedCheck_895_;
goto v_resetjp_831_;
}
else
{
lean_inc(v_toSeqRight_830_);
lean_inc(v_toSeqLeft_829_);
lean_inc(v_toSeq_828_);
lean_inc(v_toFunctor_827_);
lean_dec(v_toApplicative_823_);
v___x_832_ = lean_box(0);
v_isShared_833_ = v_isSharedCheck_895_;
goto v_resetjp_831_;
}
v_resetjp_831_:
{
lean_object* v___f_834_; lean_object* v___f_835_; lean_object* v___f_836_; lean_object* v___f_837_; lean_object* v___x_838_; lean_object* v___f_839_; lean_object* v___f_840_; lean_object* v___f_841_; lean_object* v___x_843_; 
v___f_834_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4));
v___f_835_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5));
lean_inc_ref(v_toFunctor_827_);
v___f_836_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_836_, 0, v_toFunctor_827_);
v___f_837_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_837_, 0, v_toFunctor_827_);
v___x_838_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_838_, 0, v___f_836_);
lean_ctor_set(v___x_838_, 1, v___f_837_);
v___f_839_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_839_, 0, v_toSeqRight_830_);
v___f_840_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_840_, 0, v_toSeqLeft_829_);
v___f_841_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_841_, 0, v_toSeq_828_);
if (v_isShared_833_ == 0)
{
lean_ctor_set(v___x_832_, 4, v___f_839_);
lean_ctor_set(v___x_832_, 3, v___f_840_);
lean_ctor_set(v___x_832_, 2, v___f_841_);
lean_ctor_set(v___x_832_, 1, v___f_834_);
lean_ctor_set(v___x_832_, 0, v___x_838_);
v___x_843_ = v___x_832_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v___x_838_);
lean_ctor_set(v_reuseFailAlloc_894_, 1, v___f_834_);
lean_ctor_set(v_reuseFailAlloc_894_, 2, v___f_841_);
lean_ctor_set(v_reuseFailAlloc_894_, 3, v___f_840_);
lean_ctor_set(v_reuseFailAlloc_894_, 4, v___f_839_);
v___x_843_ = v_reuseFailAlloc_894_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
lean_object* v___x_845_; 
if (v_isShared_826_ == 0)
{
lean_ctor_set(v___x_825_, 1, v___f_835_);
lean_ctor_set(v___x_825_, 0, v___x_843_);
v___x_845_ = v___x_825_;
goto v_reusejp_844_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v___x_843_);
lean_ctor_set(v_reuseFailAlloc_893_, 1, v___f_835_);
v___x_845_ = v_reuseFailAlloc_893_;
goto v_reusejp_844_;
}
v_reusejp_844_:
{
lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v_toMonadRef_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v_traceClass_851_; lean_object* v___x_852_; lean_object* v___f_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___f_856_; lean_object* v___x_857_; 
v___x_846_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1);
v___x_847_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5);
v_toMonadRef_848_ = lean_ctor_get(v___x_847_, 0);
v___x_849_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10);
v___x_850_ = lean_st_ref_get(v___y_796_);
lean_dec(v___x_850_);
v_traceClass_851_ = lean_ctor_get(v___x_787_, 0);
v___x_852_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_853_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__11));
v___x_854_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0));
v___x_855_ = lean_box(v_a_788_);
lean_inc_ref(v___f_789_);
lean_inc_ref(v_toMonadRef_848_);
lean_inc_ref(v___x_845_);
lean_inc(v_traceClass_851_);
lean_inc_ref_n(v___x_787_, 2);
lean_inc_n(v___x_805_, 2);
v___f_856_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___boxed), 17, 12);
lean_closure_set(v___f_856_, 0, v___x_805_);
lean_closure_set(v___f_856_, 1, v___x_787_);
lean_closure_set(v___f_856_, 2, v_traceClass_851_);
lean_closure_set(v___f_856_, 3, v___x_845_);
lean_closure_set(v___f_856_, 4, v___x_846_);
lean_closure_set(v___f_856_, 5, v_toMonadRef_848_);
lean_closure_set(v___f_856_, 6, v___x_852_);
lean_closure_set(v___f_856_, 7, v___x_849_);
lean_closure_set(v___f_856_, 8, v___f_853_);
lean_closure_set(v___f_856_, 9, v___x_855_);
lean_closure_set(v___f_856_, 10, v___x_854_);
lean_closure_set(v___f_856_, 11, v___f_789_);
lean_inc_ref(v___f_790_);
v___x_857_ = lp_aesop_Aesop_Rapp_withHeadlineTraceNode___redArg(v___x_805_, v___x_787_, v___f_856_, v_a_788_, v___f_790_, v___y_799_, v___y_800_, v___y_801_, v___y_802_);
if (lean_obj_tag(v___x_857_) == 0)
{
lean_object* v___x_858_; lean_object* v_elimRapp_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v_metaState_862_; lean_object* v___f_863_; lean_object* v___x_864_; lean_object* v___f_865_; lean_object* v___x_866_; lean_object* v___x_867_; 
lean_dec_ref_known(v___x_857_, 1);
v___x_858_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_859_ = lean_ctor_get(v___x_858_, 3);
v___x_860_ = lean_st_ref_get(v___y_796_);
lean_dec(v___x_860_);
lean_inc_ref(v_elimRapp_859_);
lean_inc(v___x_805_);
v___x_861_ = lean_apply_1(v_elimRapp_859_, v___x_805_);
v_metaState_862_ = lean_ctor_get(v___x_861_, 6);
lean_inc_ref(v_metaState_862_);
lean_dec_ref(v___x_861_);
v___f_863_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__20));
v___x_864_ = lean_box(v_a_788_);
lean_inc_ref(v_toMonadRef_848_);
lean_inc_ref(v___x_845_);
v___f_865_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__4___boxed), 16, 10);
lean_closure_set(v___f_865_, 0, v___x_787_);
lean_closure_set(v___f_865_, 1, v___x_845_);
lean_closure_set(v___f_865_, 2, v___x_846_);
lean_closure_set(v___f_865_, 3, v_toMonadRef_848_);
lean_closure_set(v___f_865_, 4, v___x_852_);
lean_closure_set(v___f_865_, 5, v___x_849_);
lean_closure_set(v___f_865_, 6, v___f_853_);
lean_closure_set(v___f_865_, 7, v___x_864_);
lean_closure_set(v___f_865_, 8, v___f_789_);
lean_closure_set(v___f_865_, 9, v___f_790_);
v___x_866_ = lp_aesop_Aesop_Rapp_forSubgoalsM___redArg(v___x_845_, v___f_863_, v___f_865_, v___x_805_);
v___x_867_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_metaState_862_, v___x_866_, v___y_799_, v___y_800_, v___y_801_, v___y_802_);
if (lean_obj_tag(v___x_867_) == 0)
{
lean_object* v___x_869_; uint8_t v_isShared_870_; uint8_t v_isSharedCheck_875_; 
v_isSharedCheck_875_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_875_ == 0)
{
lean_object* v_unused_876_; 
v_unused_876_ = lean_ctor_get(v___x_867_, 0);
lean_dec(v_unused_876_);
v___x_869_ = v___x_867_;
v_isShared_870_ = v_isSharedCheck_875_;
goto v_resetjp_868_;
}
else
{
lean_dec(v___x_867_);
v___x_869_ = lean_box(0);
v_isShared_870_ = v_isSharedCheck_875_;
goto v_resetjp_868_;
}
v_resetjp_868_:
{
lean_object* v___x_871_; lean_object* v___x_873_; 
v___x_871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_871_, 0, v___x_791_);
if (v_isShared_870_ == 0)
{
lean_ctor_set(v___x_869_, 0, v___x_871_);
v___x_873_ = v___x_869_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v___x_871_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
else
{
lean_object* v_a_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_884_; 
v_a_877_ = lean_ctor_get(v___x_867_, 0);
v_isSharedCheck_884_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_884_ == 0)
{
v___x_879_ = v___x_867_;
v_isShared_880_ = v_isSharedCheck_884_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_a_877_);
lean_dec(v___x_867_);
v___x_879_ = lean_box(0);
v_isShared_880_ = v_isSharedCheck_884_;
goto v_resetjp_878_;
}
v_resetjp_878_:
{
lean_object* v___x_882_; 
if (v_isShared_880_ == 0)
{
v___x_882_ = v___x_879_;
goto v_reusejp_881_;
}
else
{
lean_object* v_reuseFailAlloc_883_; 
v_reuseFailAlloc_883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_883_, 0, v_a_877_);
v___x_882_ = v_reuseFailAlloc_883_;
goto v_reusejp_881_;
}
v_reusejp_881_:
{
return v___x_882_;
}
}
}
}
else
{
lean_object* v_a_885_; lean_object* v___x_887_; uint8_t v_isShared_888_; uint8_t v_isSharedCheck_892_; 
lean_dec_ref(v___x_845_);
lean_dec(v___x_805_);
lean_dec_ref(v___f_790_);
lean_dec_ref(v___f_789_);
lean_dec_ref(v___x_787_);
v_a_885_ = lean_ctor_get(v___x_857_, 0);
v_isSharedCheck_892_ = !lean_is_exclusive(v___x_857_);
if (v_isSharedCheck_892_ == 0)
{
v___x_887_ = v___x_857_;
v_isShared_888_ = v_isSharedCheck_892_;
goto v_resetjp_886_;
}
else
{
lean_inc(v_a_885_);
lean_dec(v___x_857_);
v___x_887_ = lean_box(0);
v_isShared_888_ = v_isSharedCheck_892_;
goto v_resetjp_886_;
}
v_resetjp_886_:
{
lean_object* v___x_890_; 
if (v_isShared_888_ == 0)
{
v___x_890_ = v___x_887_;
goto v_reusejp_889_;
}
else
{
lean_object* v_reuseFailAlloc_891_; 
v_reuseFailAlloc_891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_891_, 0, v_a_885_);
v___x_890_ = v_reuseFailAlloc_891_;
goto v_reusejp_889_;
}
v_reusejp_889_:
{
return v___x_890_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___boxed(lean_object** _args){
lean_object* v___x_899_ = _args[0];
lean_object* v_a_900_ = _args[1];
lean_object* v___f_901_ = _args[2];
lean_object* v___f_902_ = _args[3];
lean_object* v___x_903_ = _args[4];
lean_object* v_a_904_ = _args[5];
lean_object* v_x_905_ = _args[6];
lean_object* v___y_906_ = _args[7];
lean_object* v___y_907_ = _args[8];
lean_object* v___y_908_ = _args[9];
lean_object* v___y_909_ = _args[10];
lean_object* v___y_910_ = _args[11];
lean_object* v___y_911_ = _args[12];
lean_object* v___y_912_ = _args[13];
lean_object* v___y_913_ = _args[14];
lean_object* v___y_914_ = _args[15];
lean_object* v___y_915_ = _args[16];
_start:
{
uint8_t v_a_28553__boxed_916_; lean_object* v_res_917_; 
v_a_28553__boxed_916_ = lean_unbox(v_a_900_);
v_res_917_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5(v___x_899_, v_a_28553__boxed_916_, v___f_901_, v___f_902_, v___x_903_, v_a_904_, v_x_905_, v___y_906_, v___y_907_, v___y_908_, v___y_909_, v___y_910_, v___y_911_, v___y_912_, v___y_913_, v___y_914_);
lean_dec(v___y_914_);
lean_dec_ref(v___y_913_);
lean_dec(v___y_912_);
lean_dec_ref(v___y_911_);
lean_dec(v___y_910_);
lean_dec(v___y_909_);
lean_dec(v___y_908_);
lean_dec_ref(v___y_907_);
lean_dec(v_a_904_);
return v_res_917_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0(void){
_start:
{
lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_918_ = lp_aesop_Aesop_BaseM_instMonadStats;
v___x_919_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_918_);
return v___x_919_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1(void){
_start:
{
lean_object* v___x_920_; lean_object* v___x_921_; 
v___x_920_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__0);
v___x_921_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_920_);
return v___x_921_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2(void){
_start:
{
lean_object* v___x_922_; lean_object* v___x_923_; 
v___x_922_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__1);
v___x_923_ = lp_aesop_Aesop_instMonadStatsReaderT___redArg(v___x_922_);
return v___x_923_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(lean_object* v_inst_926_, lean_object* v_newRapps_927_, lean_object* v_a_928_, lean_object* v_a_929_, lean_object* v_a_930_, lean_object* v_a_931_, lean_object* v_a_932_, lean_object* v_a_933_, lean_object* v_a_934_, lean_object* v_a_935_){
_start:
{
lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v_toMonadOptions_939_; lean_object* v___x_940_; lean_object* v___x_26577__overap_941_; lean_object* v___x_942_; 
v___x_937_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_926_);
v___x_938_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2);
v_toMonadOptions_939_ = lean_ctor_get(v___x_938_, 0);
v___x_940_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_toMonadOptions_939_);
lean_inc_ref(v___x_937_);
v___x_26577__overap_941_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_937_, v_toMonadOptions_939_, v___x_940_);
lean_inc(v_a_935_);
lean_inc_ref(v_a_934_);
lean_inc(v_a_933_);
lean_inc_ref(v_a_932_);
lean_inc(v_a_931_);
lean_inc(v_a_930_);
lean_inc(v_a_929_);
lean_inc_ref(v_a_928_);
v___x_942_ = lean_apply_9(v___x_26577__overap_941_, v_a_928_, v_a_929_, v_a_930_, v_a_931_, v_a_932_, v_a_933_, v_a_934_, v_a_935_, lean_box(0));
if (lean_obj_tag(v___x_942_) == 0)
{
lean_object* v_a_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_968_; 
v_a_943_ = lean_ctor_get(v___x_942_, 0);
v_isSharedCheck_968_ = !lean_is_exclusive(v___x_942_);
if (v_isSharedCheck_968_ == 0)
{
v___x_945_ = v___x_942_;
v_isShared_946_ = v_isSharedCheck_968_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_a_943_);
lean_dec(v___x_942_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_968_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
uint8_t v___x_947_; 
v___x_947_ = lean_unbox(v_a_943_);
if (v___x_947_ == 0)
{
lean_object* v___x_948_; lean_object* v___x_950_; 
lean_dec(v_a_943_);
lean_dec_ref(v___x_937_);
lean_dec_ref(v_newRapps_927_);
v___x_948_ = lean_box(0);
if (v_isShared_946_ == 0)
{
lean_ctor_set(v___x_945_, 0, v___x_948_);
v___x_950_ = v___x_945_;
goto v_reusejp_949_;
}
else
{
lean_object* v_reuseFailAlloc_951_; 
v_reuseFailAlloc_951_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_951_, 0, v___x_948_);
v___x_950_ = v_reuseFailAlloc_951_;
goto v_reusejp_949_;
}
v_reusejp_949_:
{
return v___x_950_;
}
}
else
{
lean_object* v___f_952_; lean_object* v___f_953_; lean_object* v___x_954_; lean_object* v___f_955_; size_t v_sz_956_; size_t v___x_957_; lean_object* v___x_27353__overap_958_; lean_object* v___x_959_; 
lean_del_object(v___x_945_);
v___f_952_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__3));
v___f_953_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__4));
v___x_954_ = lean_box(0);
v___f_955_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___boxed), 17, 5);
lean_closure_set(v___f_955_, 0, v___x_940_);
lean_closure_set(v___f_955_, 1, v_a_943_);
lean_closure_set(v___f_955_, 2, v___f_952_);
lean_closure_set(v___f_955_, 3, v___f_953_);
lean_closure_set(v___f_955_, 4, v___x_954_);
v_sz_956_ = lean_array_size(v_newRapps_927_);
v___x_957_ = ((size_t)0ULL);
v___x_27353__overap_958_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_937_, v_newRapps_927_, v___f_955_, v_sz_956_, v___x_957_, v___x_954_);
lean_inc(v_a_935_);
lean_inc_ref(v_a_934_);
lean_inc(v_a_933_);
lean_inc_ref(v_a_932_);
lean_inc(v_a_931_);
lean_inc(v_a_930_);
lean_inc(v_a_929_);
lean_inc_ref(v_a_928_);
v___x_959_ = lean_apply_9(v___x_27353__overap_958_, v_a_928_, v_a_929_, v_a_930_, v_a_931_, v_a_932_, v_a_933_, v_a_934_, v_a_935_, lean_box(0));
if (lean_obj_tag(v___x_959_) == 0)
{
lean_object* v___x_961_; uint8_t v_isShared_962_; uint8_t v_isSharedCheck_966_; 
v_isSharedCheck_966_ = !lean_is_exclusive(v___x_959_);
if (v_isSharedCheck_966_ == 0)
{
lean_object* v_unused_967_; 
v_unused_967_ = lean_ctor_get(v___x_959_, 0);
lean_dec(v_unused_967_);
v___x_961_ = v___x_959_;
v_isShared_962_ = v_isSharedCheck_966_;
goto v_resetjp_960_;
}
else
{
lean_dec(v___x_959_);
v___x_961_ = lean_box(0);
v_isShared_962_ = v_isSharedCheck_966_;
goto v_resetjp_960_;
}
v_resetjp_960_:
{
lean_object* v___x_964_; 
if (v_isShared_962_ == 0)
{
lean_ctor_set(v___x_961_, 0, v___x_954_);
v___x_964_ = v___x_961_;
goto v_reusejp_963_;
}
else
{
lean_object* v_reuseFailAlloc_965_; 
v_reuseFailAlloc_965_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_965_, 0, v___x_954_);
v___x_964_ = v_reuseFailAlloc_965_;
goto v_reusejp_963_;
}
v_reusejp_963_:
{
return v___x_964_;
}
}
}
else
{
return v___x_959_;
}
}
}
}
else
{
lean_object* v_a_969_; lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_976_; 
lean_dec_ref(v___x_937_);
lean_dec_ref(v_newRapps_927_);
v_a_969_ = lean_ctor_get(v___x_942_, 0);
v_isSharedCheck_976_ = !lean_is_exclusive(v___x_942_);
if (v_isSharedCheck_976_ == 0)
{
v___x_971_ = v___x_942_;
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
else
{
lean_inc(v_a_969_);
lean_dec(v___x_942_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_974_; 
if (v_isShared_972_ == 0)
{
v___x_974_ = v___x_971_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v_a_969_);
v___x_974_ = v_reuseFailAlloc_975_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
return v___x_974_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___boxed(lean_object* v_inst_977_, lean_object* v_newRapps_978_, lean_object* v_a_979_, lean_object* v_a_980_, lean_object* v_a_981_, lean_object* v_a_982_, lean_object* v_a_983_, lean_object* v_a_984_, lean_object* v_a_985_, lean_object* v_a_986_, lean_object* v_a_987_){
_start:
{
lean_object* v_res_988_; 
v_res_988_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(v_inst_977_, v_newRapps_978_, v_a_979_, v_a_980_, v_a_981_, v_a_982_, v_a_983_, v_a_984_, v_a_985_, v_a_986_);
lean_dec(v_a_986_);
lean_dec_ref(v_a_985_);
lean_dec(v_a_984_);
lean_dec_ref(v_a_983_);
lean_dec(v_a_982_);
lean_dec(v_a_981_);
lean_dec(v_a_980_);
lean_dec_ref(v_a_979_);
lean_dec_ref(v_inst_977_);
return v_res_988_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps(lean_object* v_Q_989_, lean_object* v_inst_990_, lean_object* v_newRapps_991_, lean_object* v_a_992_, lean_object* v_a_993_, lean_object* v_a_994_, lean_object* v_a_995_, lean_object* v_a_996_, lean_object* v_a_997_, lean_object* v_a_998_, lean_object* v_a_999_){
_start:
{
lean_object* v___x_1001_; 
v___x_1001_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(v_inst_990_, v_newRapps_991_, v_a_992_, v_a_993_, v_a_994_, v_a_995_, v_a_996_, v_a_997_, v_a_998_, v_a_999_);
return v___x_1001_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___boxed(lean_object* v_Q_1002_, lean_object* v_inst_1003_, lean_object* v_newRapps_1004_, lean_object* v_a_1005_, lean_object* v_a_1006_, lean_object* v_a_1007_, lean_object* v_a_1008_, lean_object* v_a_1009_, lean_object* v_a_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_, lean_object* v_a_1013_){
_start:
{
lean_object* v_res_1014_; 
v_res_1014_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps(v_Q_1002_, v_inst_1003_, v_newRapps_1004_, v_a_1005_, v_a_1006_, v_a_1007_, v_a_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_);
lean_dec(v_a_1012_);
lean_dec_ref(v_a_1011_);
lean_dec(v_a_1010_);
lean_dec_ref(v_a_1009_);
lean_dec(v_a_1008_);
lean_dec(v_a_1007_);
lean_dec(v_a_1006_);
lean_dec_ref(v_a_1005_);
lean_dec_ref(v_inst_1003_);
return v_res_1014_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__0(lean_object* v_a_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_){
_start:
{
lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; 
v___x_1025_ = lean_st_ref_get(v___y_1017_);
lean_dec(v___x_1025_);
v___x_1026_ = lean_st_ref_get(v_a_1015_);
v___x_1027_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1027_, 0, v___x_1026_);
return v___x_1027_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed(lean_object* v_a_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_){
_start:
{
lean_object* v_res_1038_; 
v_res_1038_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(v_a_1028_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_, v___y_1035_, v___y_1036_);
lean_dec(v___y_1036_);
lean_dec_ref(v___y_1035_);
lean_dec(v___y_1034_);
lean_dec_ref(v___y_1033_);
lean_dec(v___y_1032_);
lean_dec(v___y_1031_);
lean_dec(v___y_1030_);
lean_dec_ref(v___y_1029_);
lean_dec(v_a_1028_);
return v_res_1038_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1040_; lean_object* v___x_1041_; 
v___x_1040_ = ((lean_object*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__0));
v___x_1041_ = l_Lean_stringToMessageData(v___x_1040_);
return v___x_1041_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1(lean_object* v___x_1042_, lean_object* v___x_1043_, lean_object* v___x_1044_, lean_object* v___x_1045_, lean_object* v___f_1046_, lean_object* v_fst_1047_, lean_object* v___x_1048_, lean_object* v___x_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_){
_start:
{
lean_object* v___x_130778__overap_1055_; lean_object* v___x_1056_; 
lean_inc_ref(v___x_1044_);
lean_inc_ref(v___x_1042_);
v___x_130778__overap_1055_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1042_, v___x_1043_, v___x_1044_);
lean_inc(v___y_1053_);
lean_inc_ref(v___y_1052_);
lean_inc(v___y_1051_);
lean_inc_ref(v___y_1050_);
v___x_1056_ = lean_apply_5(v___x_130778__overap_1055_, v___y_1050_, v___y_1051_, v___y_1052_, v___y_1053_, lean_box(0));
if (lean_obj_tag(v___x_1056_) == 0)
{
lean_object* v_a_1057_; lean_object* v___x_1059_; uint8_t v_isShared_1060_; uint8_t v_isSharedCheck_1086_; 
v_a_1057_ = lean_ctor_get(v___x_1056_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v___x_1056_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1059_ = v___x_1056_;
v_isShared_1060_ = v_isSharedCheck_1086_;
goto v_resetjp_1058_;
}
else
{
lean_inc(v_a_1057_);
lean_dec(v___x_1056_);
v___x_1059_ = lean_box(0);
v_isShared_1060_ = v_isSharedCheck_1086_;
goto v_resetjp_1058_;
}
v_resetjp_1058_:
{
uint8_t v___x_1061_; 
v___x_1061_ = lean_unbox(v_a_1057_);
lean_dec(v_a_1057_);
if (v___x_1061_ == 0)
{
lean_object* v___x_1062_; lean_object* v___x_1064_; 
lean_dec(v___y_1053_);
lean_dec_ref(v___y_1052_);
lean_dec(v___y_1051_);
lean_dec_ref(v___y_1050_);
lean_dec_ref(v___x_1049_);
lean_dec_ref(v___x_1048_);
lean_dec(v_fst_1047_);
lean_dec(v___f_1046_);
lean_dec(v___x_1045_);
lean_dec_ref(v___x_1044_);
lean_dec_ref(v___x_1042_);
v___x_1062_ = lean_box(0);
if (v_isShared_1060_ == 0)
{
lean_ctor_set(v___x_1059_, 0, v___x_1062_);
v___x_1064_ = v___x_1059_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v___x_1062_);
v___x_1064_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
return v___x_1064_;
}
}
else
{
lean_object* v___f_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v_toMonadRef_1071_; lean_object* v_traceClass_1072_; lean_object* v___x_1074_; uint8_t v_isShared_1075_; uint8_t v_isSharedCheck_1084_; 
lean_del_object(v___x_1059_);
v___f_1066_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__2));
v___x_1067_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3));
v___x_1068_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_1069_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_1067_, v___x_1045_, v___x_1068_);
v___x_1070_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_1066_, v___f_1046_, v___x_1069_);
v_toMonadRef_1071_ = lean_ctor_get(v___x_1070_, 0);
lean_inc_ref(v_toMonadRef_1071_);
lean_dec_ref(v___x_1070_);
v_traceClass_1072_ = lean_ctor_get(v___x_1044_, 0);
v_isSharedCheck_1084_ = !lean_is_exclusive(v___x_1044_);
if (v_isSharedCheck_1084_ == 0)
{
lean_object* v_unused_1085_; 
v_unused_1085_ = lean_ctor_get(v___x_1044_, 1);
lean_dec(v_unused_1085_);
v___x_1074_ = v___x_1044_;
v_isShared_1075_ = v_isSharedCheck_1084_;
goto v_resetjp_1073_;
}
else
{
lean_inc(v_traceClass_1072_);
lean_dec(v___x_1044_);
v___x_1074_ = lean_box(0);
v_isShared_1075_ = v_isSharedCheck_1084_;
goto v_resetjp_1073_;
}
v_resetjp_1073_:
{
lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1080_; 
v___x_1076_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1, &lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__1___closed__1);
v___x_1077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1077_, 0, v_fst_1047_);
v___x_1078_ = l_Lean_indentD(v___x_1077_);
if (v_isShared_1075_ == 0)
{
lean_ctor_set_tag(v___x_1074_, 7);
lean_ctor_set(v___x_1074_, 1, v___x_1078_);
lean_ctor_set(v___x_1074_, 0, v___x_1076_);
v___x_1080_ = v___x_1074_;
goto v_reusejp_1079_;
}
else
{
lean_object* v_reuseFailAlloc_1083_; 
v_reuseFailAlloc_1083_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1083_, 0, v___x_1076_);
lean_ctor_set(v_reuseFailAlloc_1083_, 1, v___x_1078_);
v___x_1080_ = v_reuseFailAlloc_1083_;
goto v_reusejp_1079_;
}
v_reusejp_1079_:
{
lean_object* v___x_130793__overap_1081_; lean_object* v___x_1082_; 
v___x_130793__overap_1081_ = l_Lean_addTrace___redArg(v___x_1042_, v___x_1048_, v_toMonadRef_1071_, v___x_1049_, v_traceClass_1072_, v___x_1080_);
v___x_1082_ = lean_apply_5(v___x_130793__overap_1081_, v___y_1050_, v___y_1051_, v___y_1052_, v___y_1053_, lean_box(0));
return v___x_1082_;
}
}
}
}
}
else
{
lean_object* v_a_1087_; lean_object* v___x_1089_; uint8_t v_isShared_1090_; uint8_t v_isSharedCheck_1094_; 
lean_dec(v___y_1053_);
lean_dec_ref(v___y_1052_);
lean_dec(v___y_1051_);
lean_dec_ref(v___y_1050_);
lean_dec_ref(v___x_1049_);
lean_dec_ref(v___x_1048_);
lean_dec(v_fst_1047_);
lean_dec(v___f_1046_);
lean_dec(v___x_1045_);
lean_dec_ref(v___x_1044_);
lean_dec_ref(v___x_1042_);
v_a_1087_ = lean_ctor_get(v___x_1056_, 0);
v_isSharedCheck_1094_ = !lean_is_exclusive(v___x_1056_);
if (v_isSharedCheck_1094_ == 0)
{
v___x_1089_ = v___x_1056_;
v_isShared_1090_ = v_isSharedCheck_1094_;
goto v_resetjp_1088_;
}
else
{
lean_inc(v_a_1087_);
lean_dec(v___x_1056_);
v___x_1089_ = lean_box(0);
v_isShared_1090_ = v_isSharedCheck_1094_;
goto v_resetjp_1088_;
}
v_resetjp_1088_:
{
lean_object* v___x_1092_; 
if (v_isShared_1090_ == 0)
{
v___x_1092_ = v___x_1089_;
goto v_reusejp_1091_;
}
else
{
lean_object* v_reuseFailAlloc_1093_; 
v_reuseFailAlloc_1093_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1093_, 0, v_a_1087_);
v___x_1092_ = v_reuseFailAlloc_1093_;
goto v_reusejp_1091_;
}
v_reusejp_1091_:
{
return v___x_1092_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed(lean_object* v___x_1095_, lean_object* v___x_1096_, lean_object* v___x_1097_, lean_object* v___x_1098_, lean_object* v___f_1099_, lean_object* v_fst_1100_, lean_object* v___x_1101_, lean_object* v___x_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_){
_start:
{
lean_object* v_res_1108_; 
v_res_1108_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__1(v___x_1095_, v___x_1096_, v___x_1097_, v___x_1098_, v___f_1099_, v_fst_1100_, v___x_1101_, v___x_1102_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_);
return v_res_1108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__2(lean_object* v_snd_1109_, lean_object* v___f_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_){
_start:
{
lean_object* v___x_1120_; lean_object* v___x_1121_; 
v___x_1120_ = lean_st_ref_get(v___y_1112_);
lean_dec(v___x_1120_);
v___x_1121_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_snd_1109_, v___f_1110_, v___y_1115_, v___y_1116_, v___y_1117_, v___y_1118_);
return v___x_1121_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__2___boxed(lean_object* v_snd_1122_, lean_object* v___f_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_){
_start:
{
lean_object* v_res_1133_; 
v_res_1133_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__2(v_snd_1122_, v___f_1123_, v___y_1124_, v___y_1125_, v___y_1126_, v___y_1127_, v___y_1128_, v___y_1129_, v___y_1130_, v___y_1131_);
lean_dec(v___y_1131_);
lean_dec_ref(v___y_1130_);
lean_dec(v___y_1129_);
lean_dec_ref(v___y_1128_);
lean_dec(v___y_1127_);
lean_dec(v___y_1126_);
lean_dec(v___y_1125_);
lean_dec_ref(v___y_1124_);
return v_res_1133_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1(void){
_start:
{
lean_object* v___x_1135_; lean_object* v___x_1136_; 
v___x_1135_ = ((lean_object*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__0));
v___x_1136_ = l_Lean_stringToMessageData(v___x_1135_);
return v___x_1136_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3(void){
_start:
{
lean_object* v___x_1138_; lean_object* v___x_1139_; 
v___x_1138_ = ((lean_object*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__2));
v___x_1139_ = l_Lean_stringToMessageData(v___x_1138_);
return v___x_1139_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3(lean_object* v___f_1140_, lean_object* v_inst_1141_, lean_object* v_a_1142_, lean_object* v___x_1143_, lean_object* v_toMonadOptions_1144_, lean_object* v___x_1145_, lean_object* v___x_1146_, lean_object* v___x_1147_, lean_object* v___f_1148_, lean_object* v_____r_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_){
_start:
{
lean_object* v___y_1160_; lean_object* v_options_1181_; lean_object* v___x_1182_; 
v_options_1181_ = lean_ctor_get(v___y_1150_, 2);
lean_inc(v___y_1157_);
lean_inc_ref(v___y_1156_);
lean_inc(v___y_1155_);
lean_inc_ref(v___y_1154_);
lean_inc(v___y_1153_);
lean_inc(v___y_1152_);
lean_inc(v___y_1151_);
lean_inc_ref(v___y_1150_);
v___x_1182_ = lean_apply_9(v___f_1140_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_, lean_box(0));
if (lean_obj_tag(v___x_1182_) == 0)
{
lean_object* v_a_1183_; lean_object* v_toOptions_1262_; lean_object* v_maxRuleApplicationDepth_1263_; lean_object* v___x_1264_; uint8_t v___x_1265_; 
v_a_1183_ = lean_ctor_get(v___x_1182_, 0);
lean_inc(v_a_1183_);
lean_dec_ref_known(v___x_1182_, 1);
v_toOptions_1262_ = lean_ctor_get(v_options_1181_, 0);
v_maxRuleApplicationDepth_1263_ = lean_ctor_get(v_toOptions_1262_, 0);
v___x_1264_ = lean_unsigned_to_nat(0u);
v___x_1265_ = lean_nat_dec_eq(v_maxRuleApplicationDepth_1263_, v___x_1264_);
if (v___x_1265_ == 0)
{
lean_object* v___x_1266_; lean_object* v_elimGoal_1267_; lean_object* v___x_1268_; lean_object* v_depth_1269_; uint8_t v___x_1270_; 
v___x_1266_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1267_ = lean_ctor_get(v___x_1266_, 1);
lean_inc_ref(v_elimGoal_1267_);
v___x_1268_ = lean_apply_1(v_elimGoal_1267_, v_a_1183_);
v_depth_1269_ = lean_ctor_get(v___x_1268_, 4);
lean_inc(v_depth_1269_);
lean_dec_ref(v___x_1268_);
v___x_1270_ = lean_nat_dec_le(v_maxRuleApplicationDepth_1263_, v_depth_1269_);
lean_dec(v_depth_1269_);
if (v___x_1270_ == 0)
{
lean_dec(v___f_1148_);
lean_dec_ref(v___x_1147_);
lean_dec_ref(v___x_1146_);
lean_dec_ref(v___x_1145_);
lean_dec(v_toMonadOptions_1144_);
lean_dec_ref(v___x_1143_);
goto v___jp_1184_;
}
else
{
lean_object* v___x_130886__overap_1271_; lean_object* v___x_1272_; 
lean_dec_ref(v_inst_1141_);
lean_inc_ref(v___x_1145_);
lean_inc_ref(v___x_1143_);
v___x_130886__overap_1271_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1143_, v_toMonadOptions_1144_, v___x_1145_);
lean_inc(v___y_1157_);
lean_inc_ref(v___y_1156_);
lean_inc(v___y_1155_);
lean_inc_ref(v___y_1154_);
lean_inc(v___y_1153_);
lean_inc(v___y_1152_);
lean_inc(v___y_1151_);
lean_inc_ref(v___y_1150_);
v___x_1272_ = lean_apply_9(v___x_130886__overap_1271_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_, lean_box(0));
if (lean_obj_tag(v___x_1272_) == 0)
{
lean_object* v_a_1273_; uint8_t v___x_1274_; 
v_a_1273_ = lean_ctor_get(v___x_1272_, 0);
lean_inc(v_a_1273_);
lean_dec_ref_known(v___x_1272_, 1);
v___x_1274_ = lean_unbox(v_a_1273_);
lean_dec(v_a_1273_);
if (v___x_1274_ == 0)
{
lean_dec(v___f_1148_);
lean_dec_ref(v___x_1147_);
lean_dec_ref(v___x_1146_);
lean_dec_ref(v___x_1145_);
lean_dec_ref(v___x_1143_);
v___y_1160_ = v___y_1151_;
goto v___jp_1159_;
}
else
{
lean_object* v_traceClass_1275_; lean_object* v___x_1277_; uint8_t v_isShared_1278_; uint8_t v_isSharedCheck_1298_; 
v_traceClass_1275_ = lean_ctor_get(v___x_1145_, 0);
v_isSharedCheck_1298_ = !lean_is_exclusive(v___x_1145_);
if (v_isSharedCheck_1298_ == 0)
{
lean_object* v_unused_1299_; 
v_unused_1299_ = lean_ctor_get(v___x_1145_, 1);
lean_dec(v_unused_1299_);
v___x_1277_ = v___x_1145_;
v_isShared_1278_ = v_isSharedCheck_1298_;
goto v_resetjp_1276_;
}
else
{
lean_inc(v_traceClass_1275_);
lean_dec(v___x_1145_);
v___x_1277_ = lean_box(0);
v_isShared_1278_ = v_isSharedCheck_1298_;
goto v_resetjp_1276_;
}
v_resetjp_1276_:
{
lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1284_; 
v___x_1279_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1, &lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1);
lean_inc(v_maxRuleApplicationDepth_1263_);
v___x_1280_ = l_Nat_reprFast(v_maxRuleApplicationDepth_1263_);
v___x_1281_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1281_, 0, v___x_1280_);
v___x_1282_ = l_Lean_MessageData_ofFormat(v___x_1281_);
if (v_isShared_1278_ == 0)
{
lean_ctor_set_tag(v___x_1277_, 7);
lean_ctor_set(v___x_1277_, 1, v___x_1282_);
lean_ctor_set(v___x_1277_, 0, v___x_1279_);
v___x_1284_ = v___x_1277_;
goto v_reusejp_1283_;
}
else
{
lean_object* v_reuseFailAlloc_1297_; 
v_reuseFailAlloc_1297_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1297_, 0, v___x_1279_);
lean_ctor_set(v_reuseFailAlloc_1297_, 1, v___x_1282_);
v___x_1284_ = v_reuseFailAlloc_1297_;
goto v_reusejp_1283_;
}
v_reusejp_1283_:
{
lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_130915__overap_1287_; lean_object* v___x_1288_; 
v___x_1285_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3, &lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3);
v___x_1286_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1286_, 0, v___x_1284_);
lean_ctor_set(v___x_1286_, 1, v___x_1285_);
v___x_130915__overap_1287_ = l_Lean_addTrace___redArg(v___x_1143_, v___x_1146_, v___x_1147_, v___f_1148_, v_traceClass_1275_, v___x_1286_);
lean_inc(v___y_1157_);
lean_inc_ref(v___y_1156_);
lean_inc(v___y_1155_);
lean_inc_ref(v___y_1154_);
lean_inc(v___y_1153_);
lean_inc(v___y_1152_);
lean_inc(v___y_1151_);
lean_inc_ref(v___y_1150_);
v___x_1288_ = lean_apply_9(v___x_130915__overap_1287_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_, lean_box(0));
if (lean_obj_tag(v___x_1288_) == 0)
{
lean_dec_ref_known(v___x_1288_, 1);
v___y_1160_ = v___y_1151_;
goto v___jp_1159_;
}
else
{
lean_object* v_a_1289_; lean_object* v___x_1291_; uint8_t v_isShared_1292_; uint8_t v_isSharedCheck_1296_; 
lean_dec(v_a_1142_);
v_a_1289_ = lean_ctor_get(v___x_1288_, 0);
v_isSharedCheck_1296_ = !lean_is_exclusive(v___x_1288_);
if (v_isSharedCheck_1296_ == 0)
{
v___x_1291_ = v___x_1288_;
v_isShared_1292_ = v_isSharedCheck_1296_;
goto v_resetjp_1290_;
}
else
{
lean_inc(v_a_1289_);
lean_dec(v___x_1288_);
v___x_1291_ = lean_box(0);
v_isShared_1292_ = v_isSharedCheck_1296_;
goto v_resetjp_1290_;
}
v_resetjp_1290_:
{
lean_object* v___x_1294_; 
if (v_isShared_1292_ == 0)
{
v___x_1294_ = v___x_1291_;
goto v_reusejp_1293_;
}
else
{
lean_object* v_reuseFailAlloc_1295_; 
v_reuseFailAlloc_1295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1295_, 0, v_a_1289_);
v___x_1294_ = v_reuseFailAlloc_1295_;
goto v_reusejp_1293_;
}
v_reusejp_1293_:
{
return v___x_1294_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1300_; lean_object* v___x_1302_; uint8_t v_isShared_1303_; uint8_t v_isSharedCheck_1307_; 
lean_dec(v___f_1148_);
lean_dec_ref(v___x_1147_);
lean_dec_ref(v___x_1146_);
lean_dec_ref(v___x_1145_);
lean_dec_ref(v___x_1143_);
lean_dec(v_a_1142_);
v_a_1300_ = lean_ctor_get(v___x_1272_, 0);
v_isSharedCheck_1307_ = !lean_is_exclusive(v___x_1272_);
if (v_isSharedCheck_1307_ == 0)
{
v___x_1302_ = v___x_1272_;
v_isShared_1303_ = v_isSharedCheck_1307_;
goto v_resetjp_1301_;
}
else
{
lean_inc(v_a_1300_);
lean_dec(v___x_1272_);
v___x_1302_ = lean_box(0);
v_isShared_1303_ = v_isSharedCheck_1307_;
goto v_resetjp_1301_;
}
v_resetjp_1301_:
{
lean_object* v___x_1305_; 
if (v_isShared_1303_ == 0)
{
v___x_1305_ = v___x_1302_;
goto v_reusejp_1304_;
}
else
{
lean_object* v_reuseFailAlloc_1306_; 
v_reuseFailAlloc_1306_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1306_, 0, v_a_1300_);
v___x_1305_ = v_reuseFailAlloc_1306_;
goto v_reusejp_1304_;
}
v_reusejp_1304_:
{
return v___x_1305_;
}
}
}
}
}
else
{
lean_dec(v_a_1183_);
lean_dec(v___f_1148_);
lean_dec_ref(v___x_1147_);
lean_dec_ref(v___x_1146_);
lean_dec_ref(v___x_1145_);
lean_dec(v_toMonadOptions_1144_);
lean_dec_ref(v___x_1143_);
goto v___jp_1184_;
}
v___jp_1184_:
{
lean_object* v___x_1185_; 
lean_inc(v_a_1142_);
lean_inc_ref(v_inst_1141_);
v___x_1185_ = lp_aesop_Aesop_expandGoal___redArg(v_inst_1141_, v_a_1142_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_);
if (lean_obj_tag(v___x_1185_) == 0)
{
lean_object* v_a_1186_; lean_object* v___x_1187_; 
v_a_1186_ = lean_ctor_get(v___x_1185_, 0);
lean_inc(v_a_1186_);
lean_dec_ref_known(v___x_1185_, 1);
v___x_1187_ = lp_aesop_Aesop_getIteration___redArg(v___y_1151_);
if (lean_obj_tag(v___x_1187_) == 0)
{
lean_object* v_a_1188_; lean_object* v___x_1190_; uint8_t v_isShared_1191_; uint8_t v_isSharedCheck_1253_; 
v_a_1188_ = lean_ctor_get(v___x_1187_, 0);
v_isSharedCheck_1253_ = !lean_is_exclusive(v___x_1187_);
if (v_isSharedCheck_1253_ == 0)
{
v___x_1190_ = v___x_1187_;
v_isShared_1191_ = v_isSharedCheck_1253_;
goto v_resetjp_1189_;
}
else
{
lean_inc(v_a_1188_);
lean_dec(v___x_1187_);
v___x_1190_ = lean_box(0);
v_isShared_1191_ = v_isSharedCheck_1253_;
goto v_resetjp_1189_;
}
v_resetjp_1189_:
{
lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v_introGoal_1195_; lean_object* v_elimGoal_1196_; lean_object* v___x_1197_; lean_object* v_id_1198_; lean_object* v_parent_1199_; lean_object* v_children_1200_; lean_object* v_origin_1201_; lean_object* v_depth_1202_; uint8_t v_state_1203_; uint8_t v_isIrrelevant_1204_; uint8_t v_isForcedUnprovable_1205_; lean_object* v_preNormGoal_1206_; lean_object* v_normalizationState_1207_; lean_object* v_mvars_1208_; lean_object* v_forwardState_1209_; lean_object* v_forwardRuleMatches_1210_; double v_successProbability_1211_; lean_object* v_addedInIteration_1212_; uint8_t v_unsafeRulesSelected_1213_; lean_object* v_unsafeQueue_1214_; lean_object* v_failedRapps_1215_; lean_object* v___x_1217_; uint8_t v_isShared_1218_; uint8_t v_isSharedCheck_1251_; 
v___x_1192_ = lean_st_ref_get(v___y_1151_);
lean_dec(v___x_1192_);
v___x_1193_ = lean_st_ref_take(v_a_1142_);
v___x_1194_ = lp_aesop_Aesop_treeImpl;
v_introGoal_1195_ = lean_ctor_get(v___x_1194_, 0);
v_elimGoal_1196_ = lean_ctor_get(v___x_1194_, 1);
lean_inc_ref(v_elimGoal_1196_);
v___x_1197_ = lean_apply_1(v_elimGoal_1196_, v___x_1193_);
v_id_1198_ = lean_ctor_get(v___x_1197_, 0);
v_parent_1199_ = lean_ctor_get(v___x_1197_, 1);
v_children_1200_ = lean_ctor_get(v___x_1197_, 2);
v_origin_1201_ = lean_ctor_get(v___x_1197_, 3);
v_depth_1202_ = lean_ctor_get(v___x_1197_, 4);
v_state_1203_ = lean_ctor_get_uint8(v___x_1197_, sizeof(void*)*14 + 8);
v_isIrrelevant_1204_ = lean_ctor_get_uint8(v___x_1197_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1205_ = lean_ctor_get_uint8(v___x_1197_, sizeof(void*)*14 + 10);
v_preNormGoal_1206_ = lean_ctor_get(v___x_1197_, 5);
v_normalizationState_1207_ = lean_ctor_get(v___x_1197_, 6);
v_mvars_1208_ = lean_ctor_get(v___x_1197_, 7);
v_forwardState_1209_ = lean_ctor_get(v___x_1197_, 8);
v_forwardRuleMatches_1210_ = lean_ctor_get(v___x_1197_, 9);
v_successProbability_1211_ = lean_ctor_get_float(v___x_1197_, sizeof(void*)*14);
v_addedInIteration_1212_ = lean_ctor_get(v___x_1197_, 10);
v_unsafeRulesSelected_1213_ = lean_ctor_get_uint8(v___x_1197_, sizeof(void*)*14 + 11);
v_unsafeQueue_1214_ = lean_ctor_get(v___x_1197_, 12);
v_failedRapps_1215_ = lean_ctor_get(v___x_1197_, 13);
v_isSharedCheck_1251_ = !lean_is_exclusive(v___x_1197_);
if (v_isSharedCheck_1251_ == 0)
{
lean_object* v_unused_1252_; 
v_unused_1252_ = lean_ctor_get(v___x_1197_, 11);
lean_dec(v_unused_1252_);
v___x_1217_ = v___x_1197_;
v_isShared_1218_ = v_isSharedCheck_1251_;
goto v_resetjp_1216_;
}
else
{
lean_inc(v_failedRapps_1215_);
lean_inc(v_unsafeQueue_1214_);
lean_inc(v_addedInIteration_1212_);
lean_inc(v_forwardRuleMatches_1210_);
lean_inc(v_forwardState_1209_);
lean_inc(v_mvars_1208_);
lean_inc(v_normalizationState_1207_);
lean_inc(v_preNormGoal_1206_);
lean_inc(v_depth_1202_);
lean_inc(v_origin_1201_);
lean_inc(v_children_1200_);
lean_inc(v_parent_1199_);
lean_inc(v_id_1198_);
lean_dec(v___x_1197_);
v___x_1217_ = lean_box(0);
v_isShared_1218_ = v_isSharedCheck_1251_;
goto v_resetjp_1216_;
}
v_resetjp_1216_:
{
lean_object* v___x_1220_; 
if (v_isShared_1218_ == 0)
{
lean_ctor_set(v___x_1217_, 11, v_a_1188_);
v___x_1220_ = v___x_1217_;
goto v_reusejp_1219_;
}
else
{
lean_object* v_reuseFailAlloc_1250_; 
v_reuseFailAlloc_1250_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1250_, 0, v_id_1198_);
lean_ctor_set(v_reuseFailAlloc_1250_, 1, v_parent_1199_);
lean_ctor_set(v_reuseFailAlloc_1250_, 2, v_children_1200_);
lean_ctor_set(v_reuseFailAlloc_1250_, 3, v_origin_1201_);
lean_ctor_set(v_reuseFailAlloc_1250_, 4, v_depth_1202_);
lean_ctor_set(v_reuseFailAlloc_1250_, 5, v_preNormGoal_1206_);
lean_ctor_set(v_reuseFailAlloc_1250_, 6, v_normalizationState_1207_);
lean_ctor_set(v_reuseFailAlloc_1250_, 7, v_mvars_1208_);
lean_ctor_set(v_reuseFailAlloc_1250_, 8, v_forwardState_1209_);
lean_ctor_set(v_reuseFailAlloc_1250_, 9, v_forwardRuleMatches_1210_);
lean_ctor_set(v_reuseFailAlloc_1250_, 10, v_addedInIteration_1212_);
lean_ctor_set(v_reuseFailAlloc_1250_, 11, v_a_1188_);
lean_ctor_set(v_reuseFailAlloc_1250_, 12, v_unsafeQueue_1214_);
lean_ctor_set(v_reuseFailAlloc_1250_, 13, v_failedRapps_1215_);
lean_ctor_set_uint8(v_reuseFailAlloc_1250_, sizeof(void*)*14 + 8, v_state_1203_);
lean_ctor_set_uint8(v_reuseFailAlloc_1250_, sizeof(void*)*14 + 9, v_isIrrelevant_1204_);
lean_ctor_set_uint8(v_reuseFailAlloc_1250_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1205_);
lean_ctor_set_float(v_reuseFailAlloc_1250_, sizeof(void*)*14, v_successProbability_1211_);
lean_ctor_set_uint8(v_reuseFailAlloc_1250_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1213_);
v___x_1220_ = v_reuseFailAlloc_1250_;
goto v_reusejp_1219_;
}
v_reusejp_1219_:
{
lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; uint8_t v___x_1226_; 
lean_inc(v_introGoal_1195_);
v___x_1221_ = lean_apply_1(v_introGoal_1195_, v___x_1220_);
v___x_1222_ = lean_st_ref_set(v_a_1142_, v___x_1221_);
v___x_1223_ = lean_st_ref_get(v___y_1151_);
lean_dec(v___x_1223_);
v___x_1224_ = lean_st_ref_get(v_a_1142_);
v___x_1225_ = lean_st_ref_get(v___y_1151_);
lean_dec(v___x_1225_);
v___x_1226_ = lp_aesop_Aesop_Goal_isActive(v___x_1224_);
if (v___x_1226_ == 0)
{
lean_object* v___x_1228_; 
lean_dec(v_a_1142_);
lean_dec_ref(v_inst_1141_);
if (v_isShared_1191_ == 0)
{
lean_ctor_set(v___x_1190_, 0, v_a_1186_);
v___x_1228_ = v___x_1190_;
goto v_reusejp_1227_;
}
else
{
lean_object* v_reuseFailAlloc_1229_; 
v_reuseFailAlloc_1229_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1229_, 0, v_a_1186_);
v___x_1228_ = v_reuseFailAlloc_1229_;
goto v_reusejp_1227_;
}
v_reusejp_1227_:
{
return v___x_1228_;
}
}
else
{
lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; 
lean_del_object(v___x_1190_);
v___x_1230_ = lean_unsigned_to_nat(1u);
v___x_1231_ = lean_mk_empty_array_with_capacity(v___x_1230_);
v___x_1232_ = lean_array_push(v___x_1231_, v_a_1142_);
v___x_1233_ = lp_aesop_Aesop_enqueueGoals___redArg(v_inst_1141_, v___x_1232_, v___y_1151_);
if (lean_obj_tag(v___x_1233_) == 0)
{
lean_object* v___x_1235_; uint8_t v_isShared_1236_; uint8_t v_isSharedCheck_1240_; 
v_isSharedCheck_1240_ = !lean_is_exclusive(v___x_1233_);
if (v_isSharedCheck_1240_ == 0)
{
lean_object* v_unused_1241_; 
v_unused_1241_ = lean_ctor_get(v___x_1233_, 0);
lean_dec(v_unused_1241_);
v___x_1235_ = v___x_1233_;
v_isShared_1236_ = v_isSharedCheck_1240_;
goto v_resetjp_1234_;
}
else
{
lean_dec(v___x_1233_);
v___x_1235_ = lean_box(0);
v_isShared_1236_ = v_isSharedCheck_1240_;
goto v_resetjp_1234_;
}
v_resetjp_1234_:
{
lean_object* v___x_1238_; 
if (v_isShared_1236_ == 0)
{
lean_ctor_set(v___x_1235_, 0, v_a_1186_);
v___x_1238_ = v___x_1235_;
goto v_reusejp_1237_;
}
else
{
lean_object* v_reuseFailAlloc_1239_; 
v_reuseFailAlloc_1239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1239_, 0, v_a_1186_);
v___x_1238_ = v_reuseFailAlloc_1239_;
goto v_reusejp_1237_;
}
v_reusejp_1237_:
{
return v___x_1238_;
}
}
}
else
{
lean_object* v_a_1242_; lean_object* v___x_1244_; uint8_t v_isShared_1245_; uint8_t v_isSharedCheck_1249_; 
lean_dec(v_a_1186_);
v_a_1242_ = lean_ctor_get(v___x_1233_, 0);
v_isSharedCheck_1249_ = !lean_is_exclusive(v___x_1233_);
if (v_isSharedCheck_1249_ == 0)
{
v___x_1244_ = v___x_1233_;
v_isShared_1245_ = v_isSharedCheck_1249_;
goto v_resetjp_1243_;
}
else
{
lean_inc(v_a_1242_);
lean_dec(v___x_1233_);
v___x_1244_ = lean_box(0);
v_isShared_1245_ = v_isSharedCheck_1249_;
goto v_resetjp_1243_;
}
v_resetjp_1243_:
{
lean_object* v___x_1247_; 
if (v_isShared_1245_ == 0)
{
v___x_1247_ = v___x_1244_;
goto v_reusejp_1246_;
}
else
{
lean_object* v_reuseFailAlloc_1248_; 
v_reuseFailAlloc_1248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1248_, 0, v_a_1242_);
v___x_1247_ = v_reuseFailAlloc_1248_;
goto v_reusejp_1246_;
}
v_reusejp_1246_:
{
return v___x_1247_;
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
lean_object* v_a_1254_; lean_object* v___x_1256_; uint8_t v_isShared_1257_; uint8_t v_isSharedCheck_1261_; 
lean_dec(v_a_1186_);
lean_dec(v_a_1142_);
lean_dec_ref(v_inst_1141_);
v_a_1254_ = lean_ctor_get(v___x_1187_, 0);
v_isSharedCheck_1261_ = !lean_is_exclusive(v___x_1187_);
if (v_isSharedCheck_1261_ == 0)
{
v___x_1256_ = v___x_1187_;
v_isShared_1257_ = v_isSharedCheck_1261_;
goto v_resetjp_1255_;
}
else
{
lean_inc(v_a_1254_);
lean_dec(v___x_1187_);
v___x_1256_ = lean_box(0);
v_isShared_1257_ = v_isSharedCheck_1261_;
goto v_resetjp_1255_;
}
v_resetjp_1255_:
{
lean_object* v___x_1259_; 
if (v_isShared_1257_ == 0)
{
v___x_1259_ = v___x_1256_;
goto v_reusejp_1258_;
}
else
{
lean_object* v_reuseFailAlloc_1260_; 
v_reuseFailAlloc_1260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1260_, 0, v_a_1254_);
v___x_1259_ = v_reuseFailAlloc_1260_;
goto v_reusejp_1258_;
}
v_reusejp_1258_:
{
return v___x_1259_;
}
}
}
}
else
{
lean_dec(v_a_1142_);
lean_dec_ref(v_inst_1141_);
return v___x_1185_;
}
}
}
else
{
lean_object* v_a_1308_; lean_object* v___x_1310_; uint8_t v_isShared_1311_; uint8_t v_isSharedCheck_1315_; 
lean_dec(v___f_1148_);
lean_dec_ref(v___x_1147_);
lean_dec_ref(v___x_1146_);
lean_dec_ref(v___x_1145_);
lean_dec(v_toMonadOptions_1144_);
lean_dec_ref(v___x_1143_);
lean_dec(v_a_1142_);
lean_dec_ref(v_inst_1141_);
v_a_1308_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1315_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1315_ == 0)
{
v___x_1310_ = v___x_1182_;
v_isShared_1311_ = v_isSharedCheck_1315_;
goto v_resetjp_1309_;
}
else
{
lean_inc(v_a_1308_);
lean_dec(v___x_1182_);
v___x_1310_ = lean_box(0);
v_isShared_1311_ = v_isSharedCheck_1315_;
goto v_resetjp_1309_;
}
v_resetjp_1309_:
{
lean_object* v___x_1313_; 
if (v_isShared_1311_ == 0)
{
v___x_1313_ = v___x_1310_;
goto v_reusejp_1312_;
}
else
{
lean_object* v_reuseFailAlloc_1314_; 
v_reuseFailAlloc_1314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1314_, 0, v_a_1308_);
v___x_1313_ = v_reuseFailAlloc_1314_;
goto v_reusejp_1312_;
}
v_reusejp_1312_:
{
return v___x_1313_;
}
}
}
v___jp_1159_:
{
lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1161_ = lean_st_ref_get(v___y_1160_);
lean_dec(v___x_1161_);
v___x_1162_ = lp_aesop_Aesop_GoalRef_markForcedUnprovable(v_a_1142_);
lean_dec(v_a_1142_);
v___x_1163_ = lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(v___y_1160_);
if (lean_obj_tag(v___x_1163_) == 0)
{
lean_object* v___x_1165_; uint8_t v_isShared_1166_; uint8_t v_isSharedCheck_1171_; 
v_isSharedCheck_1171_ = !lean_is_exclusive(v___x_1163_);
if (v_isSharedCheck_1171_ == 0)
{
lean_object* v_unused_1172_; 
v_unused_1172_ = lean_ctor_get(v___x_1163_, 0);
lean_dec(v_unused_1172_);
v___x_1165_ = v___x_1163_;
v_isShared_1166_ = v_isSharedCheck_1171_;
goto v_resetjp_1164_;
}
else
{
lean_dec(v___x_1163_);
v___x_1165_ = lean_box(0);
v_isShared_1166_ = v_isSharedCheck_1171_;
goto v_resetjp_1164_;
}
v_resetjp_1164_:
{
lean_object* v___x_1167_; lean_object* v___x_1169_; 
v___x_1167_ = lean_box(2);
if (v_isShared_1166_ == 0)
{
lean_ctor_set(v___x_1165_, 0, v___x_1167_);
v___x_1169_ = v___x_1165_;
goto v_reusejp_1168_;
}
else
{
lean_object* v_reuseFailAlloc_1170_; 
v_reuseFailAlloc_1170_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1170_, 0, v___x_1167_);
v___x_1169_ = v_reuseFailAlloc_1170_;
goto v_reusejp_1168_;
}
v_reusejp_1168_:
{
return v___x_1169_;
}
}
}
else
{
lean_object* v_a_1173_; lean_object* v___x_1175_; uint8_t v_isShared_1176_; uint8_t v_isSharedCheck_1180_; 
v_a_1173_ = lean_ctor_get(v___x_1163_, 0);
v_isSharedCheck_1180_ = !lean_is_exclusive(v___x_1163_);
if (v_isSharedCheck_1180_ == 0)
{
v___x_1175_ = v___x_1163_;
v_isShared_1176_ = v_isSharedCheck_1180_;
goto v_resetjp_1174_;
}
else
{
lean_inc(v_a_1173_);
lean_dec(v___x_1163_);
v___x_1175_ = lean_box(0);
v_isShared_1176_ = v_isSharedCheck_1180_;
goto v_resetjp_1174_;
}
v_resetjp_1174_:
{
lean_object* v___x_1178_; 
if (v_isShared_1176_ == 0)
{
v___x_1178_ = v___x_1175_;
goto v_reusejp_1177_;
}
else
{
lean_object* v_reuseFailAlloc_1179_; 
v_reuseFailAlloc_1179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1179_, 0, v_a_1173_);
v___x_1178_ = v_reuseFailAlloc_1179_;
goto v_reusejp_1177_;
}
v_reusejp_1177_:
{
return v___x_1178_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__3___boxed(lean_object** _args){
lean_object* v___f_1316_ = _args[0];
lean_object* v_inst_1317_ = _args[1];
lean_object* v_a_1318_ = _args[2];
lean_object* v___x_1319_ = _args[3];
lean_object* v_toMonadOptions_1320_ = _args[4];
lean_object* v___x_1321_ = _args[5];
lean_object* v___x_1322_ = _args[6];
lean_object* v___x_1323_ = _args[7];
lean_object* v___f_1324_ = _args[8];
lean_object* v_____r_1325_ = _args[9];
lean_object* v___y_1326_ = _args[10];
lean_object* v___y_1327_ = _args[11];
lean_object* v___y_1328_ = _args[12];
lean_object* v___y_1329_ = _args[13];
lean_object* v___y_1330_ = _args[14];
lean_object* v___y_1331_ = _args[15];
lean_object* v___y_1332_ = _args[16];
lean_object* v___y_1333_ = _args[17];
lean_object* v___y_1334_ = _args[18];
_start:
{
lean_object* v_res_1335_; 
v_res_1335_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__3(v___f_1316_, v_inst_1317_, v_a_1318_, v___x_1319_, v_toMonadOptions_1320_, v___x_1321_, v___x_1322_, v___x_1323_, v___f_1324_, v_____r_1325_, v___y_1326_, v___y_1327_, v___y_1328_, v___y_1329_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_);
lean_dec(v___y_1333_);
lean_dec_ref(v___y_1332_);
lean_dec(v___y_1331_);
lean_dec_ref(v___y_1330_);
lean_dec(v___y_1329_);
lean_dec(v___y_1328_);
lean_dec(v___y_1327_);
lean_dec_ref(v___y_1326_);
return v_res_1335_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__4(lean_object* v_a_1336_, lean_object* v_____r_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_){
_start:
{
lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; 
v___x_1347_ = lean_st_ref_get(v___y_1339_);
lean_dec(v___x_1347_);
v___x_1348_ = lp_aesop_Aesop_GoalRef_markForcedUnprovable(v_a_1336_);
v___x_1349_ = lp_aesop_Aesop_setMaxRuleApplicationDepthReached___redArg(v___y_1339_);
if (lean_obj_tag(v___x_1349_) == 0)
{
lean_object* v___x_1351_; uint8_t v_isShared_1352_; uint8_t v_isSharedCheck_1357_; 
v_isSharedCheck_1357_ = !lean_is_exclusive(v___x_1349_);
if (v_isSharedCheck_1357_ == 0)
{
lean_object* v_unused_1358_; 
v_unused_1358_ = lean_ctor_get(v___x_1349_, 0);
lean_dec(v_unused_1358_);
v___x_1351_ = v___x_1349_;
v_isShared_1352_ = v_isSharedCheck_1357_;
goto v_resetjp_1350_;
}
else
{
lean_dec(v___x_1349_);
v___x_1351_ = lean_box(0);
v_isShared_1352_ = v_isSharedCheck_1357_;
goto v_resetjp_1350_;
}
v_resetjp_1350_:
{
lean_object* v___x_1353_; lean_object* v___x_1355_; 
v___x_1353_ = lean_box(2);
if (v_isShared_1352_ == 0)
{
lean_ctor_set(v___x_1351_, 0, v___x_1353_);
v___x_1355_ = v___x_1351_;
goto v_reusejp_1354_;
}
else
{
lean_object* v_reuseFailAlloc_1356_; 
v_reuseFailAlloc_1356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1356_, 0, v___x_1353_);
v___x_1355_ = v_reuseFailAlloc_1356_;
goto v_reusejp_1354_;
}
v_reusejp_1354_:
{
return v___x_1355_;
}
}
}
else
{
lean_object* v_a_1359_; lean_object* v___x_1361_; uint8_t v_isShared_1362_; uint8_t v_isSharedCheck_1366_; 
v_a_1359_ = lean_ctor_get(v___x_1349_, 0);
v_isSharedCheck_1366_ = !lean_is_exclusive(v___x_1349_);
if (v_isSharedCheck_1366_ == 0)
{
v___x_1361_ = v___x_1349_;
v_isShared_1362_ = v_isSharedCheck_1366_;
goto v_resetjp_1360_;
}
else
{
lean_inc(v_a_1359_);
lean_dec(v___x_1349_);
v___x_1361_ = lean_box(0);
v_isShared_1362_ = v_isSharedCheck_1366_;
goto v_resetjp_1360_;
}
v_resetjp_1360_:
{
lean_object* v___x_1364_; 
if (v_isShared_1362_ == 0)
{
v___x_1364_ = v___x_1361_;
goto v_reusejp_1363_;
}
else
{
lean_object* v_reuseFailAlloc_1365_; 
v_reuseFailAlloc_1365_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1365_, 0, v_a_1359_);
v___x_1364_ = v_reuseFailAlloc_1365_;
goto v_reusejp_1363_;
}
v_reusejp_1363_:
{
return v___x_1364_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___lam__4___boxed(lean_object* v_a_1367_, lean_object* v_____r_1368_, lean_object* v___y_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_){
_start:
{
lean_object* v_res_1378_; 
v_res_1378_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__4(v_a_1367_, v_____r_1368_, v___y_1369_, v___y_1370_, v___y_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
lean_dec(v___y_1376_);
lean_dec_ref(v___y_1375_);
lean_dec(v___y_1374_);
lean_dec_ref(v___y_1373_);
lean_dec(v___y_1372_);
lean_dec(v___y_1371_);
lean_dec(v___y_1370_);
lean_dec_ref(v___y_1369_);
lean_dec(v_a_1367_);
return v_res_1378_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__0(void){
_start:
{
lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; 
v___x_1379_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1);
v___x_1380_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_1381_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1380_, v___x_1379_);
return v___x_1381_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__1(void){
_start:
{
lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; 
v___x_1382_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__0, &lp_aesop_Aesop_expandNextGoal___redArg___closed__0_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__0);
v___x_1383_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_1384_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1383_, v___x_1382_);
return v___x_1384_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__2(void){
_start:
{
lean_object* v___x_1385_; lean_object* v___f_1386_; lean_object* v___x_1387_; 
v___x_1385_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__1, &lp_aesop_Aesop_expandNextGoal___redArg___closed__1_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__1);
v___f_1386_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___x_1387_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1386_, v___x_1385_);
return v___x_1387_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__3(void){
_start:
{
lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1388_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__10);
v___x_1389_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1388_);
return v___x_1389_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__4(void){
_start:
{
lean_object* v___x_1390_; lean_object* v___x_1391_; 
v___x_1390_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__3, &lp_aesop_Aesop_expandNextGoal___redArg___closed__3_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__3);
v___x_1391_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1390_);
return v___x_1391_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__5(void){
_start:
{
lean_object* v___x_1392_; lean_object* v___f_1393_; lean_object* v___x_1394_; 
v___x_1392_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__2, &lp_aesop_Aesop_expandNextGoal___redArg___closed__2_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__2);
v___f_1393_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__17));
v___x_1394_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1393_, v___x_1392_);
return v___x_1394_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__6(void){
_start:
{
lean_object* v___x_1395_; lean_object* v___x_1396_; 
v___x_1395_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__4, &lp_aesop_Aesop_expandNextGoal___redArg___closed__4_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__4);
v___x_1396_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1395_);
return v___x_1396_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__7(void){
_start:
{
lean_object* v___x_1397_; lean_object* v___x_1398_; 
v___x_1397_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__6, &lp_aesop_Aesop_expandNextGoal___redArg___closed__6_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__6);
v___x_1398_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_1397_);
return v___x_1398_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg(lean_object* v_inst_1399_, lean_object* v_a_1400_, lean_object* v_a_1401_, lean_object* v_a_1402_, lean_object* v_a_1403_, lean_object* v_a_1404_, lean_object* v_a_1405_, lean_object* v_a_1406_, lean_object* v_a_1407_){
_start:
{
lean_object* v___y_1410_; lean_object* v___f_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v_toApplicative_1434_; lean_object* v_toFunctor_1435_; lean_object* v_toSeq_1436_; lean_object* v_toSeqLeft_1437_; lean_object* v_toSeqRight_1438_; lean_object* v___f_1439_; lean_object* v___f_1440_; lean_object* v___f_1441_; lean_object* v___f_1442_; lean_object* v___x_1443_; lean_object* v___f_1444_; lean_object* v___f_1445_; lean_object* v___f_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v_toApplicative_1450_; lean_object* v___x_1452_; uint8_t v_isShared_1453_; uint8_t v_isSharedCheck_1827_; 
v___f_1430_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___x_1431_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_1432_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1);
v___x_1433_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1);
v_toApplicative_1434_ = lean_ctor_get(v___x_1433_, 0);
v_toFunctor_1435_ = lean_ctor_get(v_toApplicative_1434_, 0);
v_toSeq_1436_ = lean_ctor_get(v_toApplicative_1434_, 2);
v_toSeqLeft_1437_ = lean_ctor_get(v_toApplicative_1434_, 3);
v_toSeqRight_1438_ = lean_ctor_get(v_toApplicative_1434_, 4);
v___f_1439_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2));
v___f_1440_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3));
lean_inc_ref_n(v_toFunctor_1435_, 2);
v___f_1441_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1441_, 0, v_toFunctor_1435_);
v___f_1442_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1442_, 0, v_toFunctor_1435_);
v___x_1443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1443_, 0, v___f_1441_);
lean_ctor_set(v___x_1443_, 1, v___f_1442_);
lean_inc(v_toSeqRight_1438_);
v___f_1444_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1444_, 0, v_toSeqRight_1438_);
lean_inc(v_toSeqLeft_1437_);
v___f_1445_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1445_, 0, v_toSeqLeft_1437_);
lean_inc(v_toSeq_1436_);
v___f_1446_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1446_, 0, v_toSeq_1436_);
v___x_1447_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1447_, 0, v___x_1443_);
lean_ctor_set(v___x_1447_, 1, v___f_1439_);
lean_ctor_set(v___x_1447_, 2, v___f_1446_);
lean_ctor_set(v___x_1447_, 3, v___f_1445_);
lean_ctor_set(v___x_1447_, 4, v___f_1444_);
v___x_1448_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1448_, 0, v___x_1447_);
lean_ctor_set(v___x_1448_, 1, v___f_1440_);
v___x_1449_ = l_StateRefT_x27_instMonad___redArg(v___x_1448_);
v_toApplicative_1450_ = lean_ctor_get(v___x_1449_, 0);
v_isSharedCheck_1827_ = !lean_is_exclusive(v___x_1449_);
if (v_isSharedCheck_1827_ == 0)
{
lean_object* v_unused_1828_; 
v_unused_1828_ = lean_ctor_get(v___x_1449_, 1);
lean_dec(v_unused_1828_);
v___x_1452_ = v___x_1449_;
v_isShared_1453_ = v_isSharedCheck_1827_;
goto v_resetjp_1451_;
}
else
{
lean_inc(v_toApplicative_1450_);
lean_dec(v___x_1449_);
v___x_1452_ = lean_box(0);
v_isShared_1453_ = v_isSharedCheck_1827_;
goto v_resetjp_1451_;
}
v___jp_1409_:
{
if (lean_obj_tag(v___y_1410_) == 0)
{
lean_object* v_a_1411_; lean_object* v___x_1413_; uint8_t v_isShared_1414_; uint8_t v_isSharedCheck_1421_; 
v_a_1411_ = lean_ctor_get(v___y_1410_, 0);
v_isSharedCheck_1421_ = !lean_is_exclusive(v___y_1410_);
if (v_isSharedCheck_1421_ == 0)
{
v___x_1413_ = v___y_1410_;
v_isShared_1414_ = v_isSharedCheck_1421_;
goto v_resetjp_1412_;
}
else
{
lean_inc(v_a_1411_);
lean_dec(v___y_1410_);
v___x_1413_ = lean_box(0);
v_isShared_1414_ = v_isSharedCheck_1421_;
goto v_resetjp_1412_;
}
v_resetjp_1412_:
{
if (lean_obj_tag(v_a_1411_) == 2)
{
lean_object* v___x_1415_; lean_object* v___x_1417_; 
lean_dec_ref(v_inst_1399_);
v___x_1415_ = lean_box(0);
if (v_isShared_1414_ == 0)
{
lean_ctor_set(v___x_1413_, 0, v___x_1415_);
v___x_1417_ = v___x_1413_;
goto v_reusejp_1416_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v___x_1415_);
v___x_1417_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1416_;
}
v_reusejp_1416_:
{
return v___x_1417_;
}
}
else
{
lean_object* v_newRapps_1419_; lean_object* v___x_1420_; 
lean_del_object(v___x_1413_);
v_newRapps_1419_ = lean_ctor_get(v_a_1411_, 0);
lean_inc_ref(v_newRapps_1419_);
lean_dec(v_a_1411_);
v___x_1420_ = lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg(v_inst_1399_, v_newRapps_1419_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
lean_dec_ref(v_inst_1399_);
return v___x_1420_;
}
}
}
else
{
lean_object* v_a_1422_; lean_object* v___x_1424_; uint8_t v_isShared_1425_; uint8_t v_isSharedCheck_1429_; 
lean_dec_ref(v_inst_1399_);
v_a_1422_ = lean_ctor_get(v___y_1410_, 0);
v_isSharedCheck_1429_ = !lean_is_exclusive(v___y_1410_);
if (v_isSharedCheck_1429_ == 0)
{
v___x_1424_ = v___y_1410_;
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
else
{
lean_inc(v_a_1422_);
lean_dec(v___y_1410_);
v___x_1424_ = lean_box(0);
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
v_resetjp_1423_:
{
lean_object* v___x_1427_; 
if (v_isShared_1425_ == 0)
{
v___x_1427_ = v___x_1424_;
goto v_reusejp_1426_;
}
else
{
lean_object* v_reuseFailAlloc_1428_; 
v_reuseFailAlloc_1428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1428_, 0, v_a_1422_);
v___x_1427_ = v_reuseFailAlloc_1428_;
goto v_reusejp_1426_;
}
v_reusejp_1426_:
{
return v___x_1427_;
}
}
}
}
v_resetjp_1451_:
{
lean_object* v_toFunctor_1454_; lean_object* v_toSeq_1455_; lean_object* v_toSeqLeft_1456_; lean_object* v_toSeqRight_1457_; lean_object* v___x_1459_; uint8_t v_isShared_1460_; uint8_t v_isSharedCheck_1825_; 
v_toFunctor_1454_ = lean_ctor_get(v_toApplicative_1450_, 0);
v_toSeq_1455_ = lean_ctor_get(v_toApplicative_1450_, 2);
v_toSeqLeft_1456_ = lean_ctor_get(v_toApplicative_1450_, 3);
v_toSeqRight_1457_ = lean_ctor_get(v_toApplicative_1450_, 4);
v_isSharedCheck_1825_ = !lean_is_exclusive(v_toApplicative_1450_);
if (v_isSharedCheck_1825_ == 0)
{
lean_object* v_unused_1826_; 
v_unused_1826_ = lean_ctor_get(v_toApplicative_1450_, 1);
lean_dec(v_unused_1826_);
v___x_1459_ = v_toApplicative_1450_;
v_isShared_1460_ = v_isSharedCheck_1825_;
goto v_resetjp_1458_;
}
else
{
lean_inc(v_toSeqRight_1457_);
lean_inc(v_toSeqLeft_1456_);
lean_inc(v_toSeq_1455_);
lean_inc(v_toFunctor_1454_);
lean_dec(v_toApplicative_1450_);
v___x_1459_ = lean_box(0);
v_isShared_1460_ = v_isSharedCheck_1825_;
goto v_resetjp_1458_;
}
v_resetjp_1458_:
{
lean_object* v___f_1461_; lean_object* v___f_1462_; lean_object* v___f_1463_; lean_object* v___f_1464_; lean_object* v___x_1465_; lean_object* v___f_1466_; lean_object* v___f_1467_; lean_object* v___f_1468_; lean_object* v___x_1470_; 
v___f_1461_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4));
v___f_1462_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5));
lean_inc_ref(v_toFunctor_1454_);
v___f_1463_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1463_, 0, v_toFunctor_1454_);
v___f_1464_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1464_, 0, v_toFunctor_1454_);
v___x_1465_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1465_, 0, v___f_1463_);
lean_ctor_set(v___x_1465_, 1, v___f_1464_);
v___f_1466_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1466_, 0, v_toSeqRight_1457_);
v___f_1467_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1467_, 0, v_toSeqLeft_1456_);
v___f_1468_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1468_, 0, v_toSeq_1455_);
if (v_isShared_1460_ == 0)
{
lean_ctor_set(v___x_1459_, 4, v___f_1466_);
lean_ctor_set(v___x_1459_, 3, v___f_1467_);
lean_ctor_set(v___x_1459_, 2, v___f_1468_);
lean_ctor_set(v___x_1459_, 1, v___f_1461_);
lean_ctor_set(v___x_1459_, 0, v___x_1465_);
v___x_1470_ = v___x_1459_;
goto v_reusejp_1469_;
}
else
{
lean_object* v_reuseFailAlloc_1824_; 
v_reuseFailAlloc_1824_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1824_, 0, v___x_1465_);
lean_ctor_set(v_reuseFailAlloc_1824_, 1, v___f_1461_);
lean_ctor_set(v_reuseFailAlloc_1824_, 2, v___f_1468_);
lean_ctor_set(v_reuseFailAlloc_1824_, 3, v___f_1467_);
lean_ctor_set(v_reuseFailAlloc_1824_, 4, v___f_1466_);
v___x_1470_ = v_reuseFailAlloc_1824_;
goto v_reusejp_1469_;
}
v_reusejp_1469_:
{
lean_object* v___x_1472_; 
if (v_isShared_1453_ == 0)
{
lean_ctor_set(v___x_1452_, 1, v___f_1462_);
lean_ctor_set(v___x_1452_, 0, v___x_1470_);
v___x_1472_ = v___x_1452_;
goto v_reusejp_1471_;
}
else
{
lean_object* v_reuseFailAlloc_1823_; 
v_reuseFailAlloc_1823_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1823_, 0, v___x_1470_);
lean_ctor_set(v_reuseFailAlloc_1823_, 1, v___f_1462_);
v___x_1472_ = v_reuseFailAlloc_1823_;
goto v_reusejp_1471_;
}
v_reusejp_1471_:
{
lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v_toMonadOptions_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; 
v___x_1473_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_1399_);
v___x_1474_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__5, &lp_aesop_Aesop_expandNextGoal___redArg___closed__5_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__5);
v___x_1475_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_1399_);
v___x_1476_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2);
v_toMonadOptions_1477_ = lean_ctor_get(v___x_1476_, 0);
v___x_1478_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__7, &lp_aesop_Aesop_expandNextGoal___redArg___closed__7_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__7);
lean_inc_ref(v_inst_1399_);
v___x_1479_ = lp_aesop_Aesop_nextActiveGoal___redArg(v_inst_1399_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
if (lean_obj_tag(v___x_1479_) == 0)
{
lean_object* v_a_1480_; lean_object* v___f_1481_; lean_object* v___x_1482_; lean_object* v_a_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1814_; 
v_a_1480_ = lean_ctor_get(v___x_1479_, 0);
lean_inc_n(v_a_1480_, 2);
lean_dec_ref_known(v___x_1479_, 1);
v___f_1481_ = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__0___boxed), 10, 1);
lean_closure_set(v___f_1481_, 0, v_a_1480_);
v___x_1482_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(v_a_1480_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
v_a_1483_ = lean_ctor_get(v___x_1482_, 0);
v_isSharedCheck_1814_ = !lean_is_exclusive(v___x_1482_);
if (v_isSharedCheck_1814_ == 0)
{
v___x_1485_ = v___x_1482_;
v_isShared_1486_ = v_isSharedCheck_1814_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_a_1483_);
lean_dec(v___x_1482_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1814_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v___x_1487_; lean_object* v___x_1488_; 
v___x_1487_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1487_);
v___x_1488_ = lp_aesop_Aesop_getRootMetaState___redArg(v_a_1402_);
if (lean_obj_tag(v___x_1488_) == 0)
{
lean_object* v_a_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; 
v_a_1489_ = lean_ctor_get(v___x_1488_, 0);
lean_inc(v_a_1489_);
lean_dec_ref_known(v___x_1488_, 1);
v___x_1490_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1490_);
lean_inc(v_a_1483_);
v___x_1491_ = lp_aesop_Aesop_Goal_currentGoalAndMetaState___redArg(v_a_1483_, v_a_1489_);
lean_dec(v_a_1489_);
if (lean_obj_tag(v___x_1491_) == 0)
{
lean_object* v_a_1492_; lean_object* v_fst_1493_; lean_object* v_snd_1494_; lean_object* v___x_1496_; uint8_t v_isShared_1497_; uint8_t v_isSharedCheck_1797_; 
v_a_1492_ = lean_ctor_get(v___x_1491_, 0);
lean_inc(v_a_1492_);
lean_dec_ref_known(v___x_1491_, 1);
v_fst_1493_ = lean_ctor_get(v_a_1492_, 0);
v_snd_1494_ = lean_ctor_get(v_a_1492_, 1);
v_isSharedCheck_1797_ = !lean_is_exclusive(v_a_1492_);
if (v_isSharedCheck_1797_ == 0)
{
v___x_1496_ = v_a_1492_;
v_isShared_1497_ = v_isSharedCheck_1797_;
goto v_resetjp_1495_;
}
else
{
lean_inc(v_snd_1494_);
lean_inc(v_fst_1493_);
lean_dec(v_a_1492_);
v___x_1496_ = lean_box(0);
v_isShared_1497_ = v_isSharedCheck_1797_;
goto v_resetjp_1495_;
}
v_resetjp_1495_:
{
lean_object* v___x_1498_; lean_object* v_options_1499_; lean_object* v_introGoal_1500_; lean_object* v_elimGoal_1501_; lean_object* v_inheritedTraceOptions_1502_; uint8_t v_hasTrace_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___f_1506_; lean_object* v___x_1507_; lean_object* v___f_1508_; 
v___x_1498_ = lp_aesop_Aesop_treeImpl;
v_options_1499_ = lean_ctor_get(v_a_1406_, 2);
v_introGoal_1500_ = lean_ctor_get(v___x_1498_, 0);
v_elimGoal_1501_ = lean_ctor_get(v___x_1498_, 1);
v_inheritedTraceOptions_1502_ = lean_ctor_get(v_a_1406_, 13);
v_hasTrace_1503_ = lean_ctor_get_uint8(v_options_1499_, sizeof(void*)*1);
v___x_1504_ = l_Lean_Meta_instAddMessageContextMetaM;
v___x_1505_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__9));
v___f_1506_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
v___x_1507_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_fst_1493_);
v___f_1508_ = lean_alloc_closure((void*)(lp_aesop_Aesop_expandNextGoal___redArg___lam__1___boxed), 13, 8);
lean_closure_set(v___f_1508_, 0, v___x_1472_);
lean_closure_set(v___f_1508_, 1, v___x_1505_);
lean_closure_set(v___f_1508_, 2, v___x_1507_);
lean_closure_set(v___f_1508_, 3, v___x_1431_);
lean_closure_set(v___f_1508_, 4, v___f_1430_);
lean_closure_set(v___f_1508_, 5, v_fst_1493_);
lean_closure_set(v___f_1508_, 6, v___x_1432_);
lean_closure_set(v___f_1508_, 7, v___x_1504_);
if (v_hasTrace_1503_ == 0)
{
lean_object* v___x_1509_; 
lean_del_object(v___x_1496_);
lean_dec(v_fst_1493_);
lean_del_object(v___x_1485_);
lean_dec(v_a_1483_);
v___x_1509_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__2(v_snd_1494_, v___f_1508_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
if (lean_obj_tag(v___x_1509_) == 0)
{
lean_object* v_a_1510_; lean_object* v___x_1511_; 
v_a_1510_ = lean_ctor_get(v___x_1509_, 0);
lean_inc(v_a_1510_);
lean_dec_ref_known(v___x_1509_, 1);
lean_inc(v_toMonadOptions_1477_);
lean_inc_ref(v_inst_1399_);
v___x_1511_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__3(v___f_1481_, v_inst_1399_, v_a_1480_, v___x_1473_, v_toMonadOptions_1477_, v___x_1507_, v___x_1474_, v___x_1475_, v___f_1506_, v_a_1510_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
v___y_1410_ = v___x_1511_;
goto v___jp_1409_;
}
else
{
lean_dec_ref(v___f_1481_);
lean_dec(v_a_1480_);
lean_dec_ref(v___x_1475_);
lean_dec_ref(v___x_1473_);
lean_dec_ref(v_inst_1399_);
return v___x_1509_;
}
}
else
{
lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v_id_1514_; lean_object* v_traceClass_1515_; lean_object* v___f_1516_; double v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1522_; uint8_t v___x_1523_; lean_object* v___y_1525_; lean_object* v___y_1526_; lean_object* v_a_1527_; lean_object* v___y_1544_; lean_object* v___y_1545_; lean_object* v_a_1546_; lean_object* v___y_1551_; lean_object* v___y_1552_; lean_object* v_a_1553_; lean_object* v___y_1556_; lean_object* v___y_1557_; lean_object* v___y_1558_; lean_object* v___y_1562_; lean_object* v___y_1563_; lean_object* v___y_1610_; lean_object* v___y_1611_; lean_object* v___y_1612_; lean_object* v___y_1613_; lean_object* v___y_1643_; lean_object* v___y_1644_; lean_object* v_a_1645_; lean_object* v___y_1657_; lean_object* v___y_1658_; lean_object* v_a_1659_; lean_object* v___y_1662_; lean_object* v___y_1663_; lean_object* v_a_1664_; lean_object* v___y_1667_; lean_object* v___y_1668_; lean_object* v___y_1669_; lean_object* v___y_1673_; lean_object* v___y_1674_; 
v___x_1512_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1512_);
lean_inc_ref(v_elimGoal_1501_);
lean_inc(v_a_1483_);
v___x_1513_ = lean_apply_1(v_elimGoal_1501_, v_a_1483_);
v_id_1514_ = lean_ctor_get(v___x_1513_, 0);
lean_inc(v_id_1514_);
lean_dec_ref(v___x_1513_);
v_traceClass_1515_ = lean_ctor_get(v___x_1507_, 0);
v___f_1516_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__11));
v___x_1517_ = lp_aesop_Aesop_Goal_priority(v_a_1483_);
v___x_1518_ = lean_box_float(v___x_1517_);
lean_inc(v_snd_1494_);
lean_inc_ref(v_inst_1399_);
v___x_1519_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___boxed), 16, 6);
lean_closure_set(v___x_1519_, 0, lean_box(0));
lean_closure_set(v___x_1519_, 1, v_inst_1399_);
lean_closure_set(v___x_1519_, 2, v_id_1514_);
lean_closure_set(v___x_1519_, 3, v___x_1518_);
lean_closure_set(v___x_1519_, 4, v_fst_1493_);
lean_closure_set(v___x_1519_, 5, v_snd_1494_);
v___x_1520_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0));
v___x_1521_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__1));
lean_inc(v_traceClass_1515_);
v___x_1522_ = l_Lean_Name_append(v___x_1521_, v_traceClass_1515_);
v___x_1523_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1502_, v_options_1499_, v___x_1522_);
lean_dec(v___x_1522_);
if (v___x_1523_ == 0)
{
lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; uint8_t v___x_1793_; 
v___x_1790_ = l_Lean_KVMap_instValueBool;
v___x_1791_ = l_Lean_trace_profiler;
v___x_1792_ = l_Lean_Option_get___redArg(v___x_1790_, v_options_1499_, v___x_1791_);
v___x_1793_ = lean_unbox(v___x_1792_);
lean_dec(v___x_1792_);
if (v___x_1793_ == 0)
{
lean_object* v___x_1794_; 
lean_dec_ref(v___x_1519_);
lean_del_object(v___x_1496_);
lean_del_object(v___x_1485_);
v___x_1794_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__2(v_snd_1494_, v___f_1508_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
if (lean_obj_tag(v___x_1794_) == 0)
{
lean_object* v_a_1795_; lean_object* v___x_1796_; 
v_a_1795_ = lean_ctor_get(v___x_1794_, 0);
lean_inc(v_a_1795_);
lean_dec_ref_known(v___x_1794_, 1);
lean_inc(v_toMonadOptions_1477_);
lean_inc_ref(v_inst_1399_);
v___x_1796_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__3(v___f_1481_, v_inst_1399_, v_a_1480_, v___x_1473_, v_toMonadOptions_1477_, v___x_1507_, v___x_1474_, v___x_1475_, v___f_1506_, v_a_1795_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
v___y_1410_ = v___x_1796_;
goto v___jp_1409_;
}
else
{
lean_dec_ref(v___f_1481_);
lean_dec(v_a_1480_);
lean_dec_ref(v___x_1475_);
lean_dec_ref(v___x_1473_);
lean_dec_ref(v_inst_1399_);
return v___x_1794_;
}
}
else
{
lean_dec_ref(v___f_1481_);
goto v___jp_1720_;
}
}
else
{
lean_dec_ref(v___f_1481_);
goto v___jp_1720_;
}
v___jp_1524_:
{
lean_object* v___x_1528_; lean_object* v___x_1529_; double v___x_1530_; double v___x_1531_; double v___x_1532_; double v___x_1533_; double v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1538_; 
v___x_1528_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1528_);
v___x_1529_ = lean_io_mono_nanos_now();
v___x_1530_ = lean_float_of_nat(v___y_1526_);
v___x_1531_ = lean_float_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2);
v___x_1532_ = lean_float_div(v___x_1530_, v___x_1531_);
v___x_1533_ = lean_float_of_nat(v___x_1529_);
v___x_1534_ = lean_float_div(v___x_1533_, v___x_1531_);
v___x_1535_ = lean_box_float(v___x_1532_);
v___x_1536_ = lean_box_float(v___x_1534_);
if (v_isShared_1497_ == 0)
{
lean_ctor_set(v___x_1496_, 1, v___x_1536_);
lean_ctor_set(v___x_1496_, 0, v___x_1535_);
v___x_1538_ = v___x_1496_;
goto v_reusejp_1537_;
}
else
{
lean_object* v_reuseFailAlloc_1542_; 
v_reuseFailAlloc_1542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1542_, 0, v___x_1535_);
lean_ctor_set(v_reuseFailAlloc_1542_, 1, v___x_1536_);
v___x_1538_ = v_reuseFailAlloc_1542_;
goto v_reusejp_1537_;
}
v_reusejp_1537_:
{
lean_object* v___x_1539_; lean_object* v___x_130446__overap_1540_; lean_object* v___x_1541_; 
v___x_1539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1539_, 0, v_a_1527_);
lean_ctor_set(v___x_1539_, 1, v___x_1538_);
lean_inc(v_traceClass_1515_);
v___x_130446__overap_1540_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1473_, v___x_1474_, v___x_1475_, v___f_1506_, lean_box(0), v___x_1478_, v___f_1516_, v_traceClass_1515_, v_hasTrace_1503_, v___x_1520_, v_options_1499_, v___x_1523_, v___y_1525_, v___x_1519_, v___x_1539_);
lean_inc(v_a_1407_);
lean_inc_ref(v_a_1406_);
lean_inc(v_a_1405_);
lean_inc_ref(v_a_1404_);
lean_inc(v_a_1403_);
lean_inc(v_a_1402_);
lean_inc(v_a_1401_);
lean_inc_ref(v_a_1400_);
v___x_1541_ = lean_apply_9(v___x_130446__overap_1540_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, lean_box(0));
v___y_1410_ = v___x_1541_;
goto v___jp_1409_;
}
}
v___jp_1543_:
{
lean_object* v___x_1548_; 
if (v_isShared_1486_ == 0)
{
lean_ctor_set_tag(v___x_1485_, 1);
lean_ctor_set(v___x_1485_, 0, v_a_1546_);
v___x_1548_ = v___x_1485_;
goto v_reusejp_1547_;
}
else
{
lean_object* v_reuseFailAlloc_1549_; 
v_reuseFailAlloc_1549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1549_, 0, v_a_1546_);
v___x_1548_ = v_reuseFailAlloc_1549_;
goto v_reusejp_1547_;
}
v_reusejp_1547_:
{
v___y_1525_ = v___y_1544_;
v___y_1526_ = v___y_1545_;
v_a_1527_ = v___x_1548_;
goto v___jp_1524_;
}
}
v___jp_1550_:
{
lean_object* v___x_1554_; 
v___x_1554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1554_, 0, v_a_1553_);
v___y_1525_ = v___y_1551_;
v___y_1526_ = v___y_1552_;
v_a_1527_ = v___x_1554_;
goto v___jp_1524_;
}
v___jp_1555_:
{
if (lean_obj_tag(v___y_1558_) == 0)
{
lean_object* v_a_1559_; 
v_a_1559_ = lean_ctor_get(v___y_1558_, 0);
lean_inc(v_a_1559_);
lean_dec_ref_known(v___y_1558_, 1);
v___y_1544_ = v___y_1556_;
v___y_1545_ = v___y_1557_;
v_a_1546_ = v_a_1559_;
goto v___jp_1543_;
}
else
{
lean_object* v_a_1560_; 
lean_del_object(v___x_1485_);
v_a_1560_ = lean_ctor_get(v___y_1558_, 0);
lean_inc(v_a_1560_);
lean_dec_ref_known(v___y_1558_, 1);
v___y_1551_ = v___y_1556_;
v___y_1552_ = v___y_1557_;
v_a_1553_ = v_a_1560_;
goto v___jp_1550_;
}
}
v___jp_1561_:
{
lean_object* v___x_1564_; 
lean_inc(v_a_1480_);
lean_inc_ref(v_inst_1399_);
v___x_1564_ = lp_aesop_Aesop_expandGoal___redArg(v_inst_1399_, v_a_1480_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
if (lean_obj_tag(v___x_1564_) == 0)
{
lean_object* v_a_1565_; lean_object* v___x_1566_; 
v_a_1565_ = lean_ctor_get(v___x_1564_, 0);
lean_inc(v_a_1565_);
lean_dec_ref_known(v___x_1564_, 1);
v___x_1566_ = lp_aesop_Aesop_getIteration___redArg(v_a_1401_);
if (lean_obj_tag(v___x_1566_) == 0)
{
lean_object* v_a_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v_id_1571_; lean_object* v_parent_1572_; lean_object* v_children_1573_; lean_object* v_origin_1574_; lean_object* v_depth_1575_; uint8_t v_state_1576_; uint8_t v_isIrrelevant_1577_; uint8_t v_isForcedUnprovable_1578_; lean_object* v_preNormGoal_1579_; lean_object* v_normalizationState_1580_; lean_object* v_mvars_1581_; lean_object* v_forwardState_1582_; lean_object* v_forwardRuleMatches_1583_; double v_successProbability_1584_; lean_object* v_addedInIteration_1585_; uint8_t v_unsafeRulesSelected_1586_; lean_object* v_unsafeQueue_1587_; lean_object* v_failedRapps_1588_; lean_object* v___x_1590_; uint8_t v_isShared_1591_; uint8_t v_isSharedCheck_1606_; 
v_a_1567_ = lean_ctor_get(v___x_1566_, 0);
lean_inc(v_a_1567_);
lean_dec_ref_known(v___x_1566_, 1);
v___x_1568_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1568_);
v___x_1569_ = lean_st_ref_take(v_a_1480_);
lean_inc_ref(v_elimGoal_1501_);
v___x_1570_ = lean_apply_1(v_elimGoal_1501_, v___x_1569_);
v_id_1571_ = lean_ctor_get(v___x_1570_, 0);
v_parent_1572_ = lean_ctor_get(v___x_1570_, 1);
v_children_1573_ = lean_ctor_get(v___x_1570_, 2);
v_origin_1574_ = lean_ctor_get(v___x_1570_, 3);
v_depth_1575_ = lean_ctor_get(v___x_1570_, 4);
v_state_1576_ = lean_ctor_get_uint8(v___x_1570_, sizeof(void*)*14 + 8);
v_isIrrelevant_1577_ = lean_ctor_get_uint8(v___x_1570_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1578_ = lean_ctor_get_uint8(v___x_1570_, sizeof(void*)*14 + 10);
v_preNormGoal_1579_ = lean_ctor_get(v___x_1570_, 5);
v_normalizationState_1580_ = lean_ctor_get(v___x_1570_, 6);
v_mvars_1581_ = lean_ctor_get(v___x_1570_, 7);
v_forwardState_1582_ = lean_ctor_get(v___x_1570_, 8);
v_forwardRuleMatches_1583_ = lean_ctor_get(v___x_1570_, 9);
v_successProbability_1584_ = lean_ctor_get_float(v___x_1570_, sizeof(void*)*14);
v_addedInIteration_1585_ = lean_ctor_get(v___x_1570_, 10);
v_unsafeRulesSelected_1586_ = lean_ctor_get_uint8(v___x_1570_, sizeof(void*)*14 + 11);
v_unsafeQueue_1587_ = lean_ctor_get(v___x_1570_, 12);
v_failedRapps_1588_ = lean_ctor_get(v___x_1570_, 13);
v_isSharedCheck_1606_ = !lean_is_exclusive(v___x_1570_);
if (v_isSharedCheck_1606_ == 0)
{
lean_object* v_unused_1607_; 
v_unused_1607_ = lean_ctor_get(v___x_1570_, 11);
lean_dec(v_unused_1607_);
v___x_1590_ = v___x_1570_;
v_isShared_1591_ = v_isSharedCheck_1606_;
goto v_resetjp_1589_;
}
else
{
lean_inc(v_failedRapps_1588_);
lean_inc(v_unsafeQueue_1587_);
lean_inc(v_addedInIteration_1585_);
lean_inc(v_forwardRuleMatches_1583_);
lean_inc(v_forwardState_1582_);
lean_inc(v_mvars_1581_);
lean_inc(v_normalizationState_1580_);
lean_inc(v_preNormGoal_1579_);
lean_inc(v_depth_1575_);
lean_inc(v_origin_1574_);
lean_inc(v_children_1573_);
lean_inc(v_parent_1572_);
lean_inc(v_id_1571_);
lean_dec(v___x_1570_);
v___x_1590_ = lean_box(0);
v_isShared_1591_ = v_isSharedCheck_1606_;
goto v_resetjp_1589_;
}
v_resetjp_1589_:
{
lean_object* v___x_1593_; 
if (v_isShared_1591_ == 0)
{
lean_ctor_set(v___x_1590_, 11, v_a_1567_);
v___x_1593_ = v___x_1590_;
goto v_reusejp_1592_;
}
else
{
lean_object* v_reuseFailAlloc_1605_; 
v_reuseFailAlloc_1605_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1605_, 0, v_id_1571_);
lean_ctor_set(v_reuseFailAlloc_1605_, 1, v_parent_1572_);
lean_ctor_set(v_reuseFailAlloc_1605_, 2, v_children_1573_);
lean_ctor_set(v_reuseFailAlloc_1605_, 3, v_origin_1574_);
lean_ctor_set(v_reuseFailAlloc_1605_, 4, v_depth_1575_);
lean_ctor_set(v_reuseFailAlloc_1605_, 5, v_preNormGoal_1579_);
lean_ctor_set(v_reuseFailAlloc_1605_, 6, v_normalizationState_1580_);
lean_ctor_set(v_reuseFailAlloc_1605_, 7, v_mvars_1581_);
lean_ctor_set(v_reuseFailAlloc_1605_, 8, v_forwardState_1582_);
lean_ctor_set(v_reuseFailAlloc_1605_, 9, v_forwardRuleMatches_1583_);
lean_ctor_set(v_reuseFailAlloc_1605_, 10, v_addedInIteration_1585_);
lean_ctor_set(v_reuseFailAlloc_1605_, 11, v_a_1567_);
lean_ctor_set(v_reuseFailAlloc_1605_, 12, v_unsafeQueue_1587_);
lean_ctor_set(v_reuseFailAlloc_1605_, 13, v_failedRapps_1588_);
lean_ctor_set_uint8(v_reuseFailAlloc_1605_, sizeof(void*)*14 + 8, v_state_1576_);
lean_ctor_set_uint8(v_reuseFailAlloc_1605_, sizeof(void*)*14 + 9, v_isIrrelevant_1577_);
lean_ctor_set_uint8(v_reuseFailAlloc_1605_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1578_);
lean_ctor_set_float(v_reuseFailAlloc_1605_, sizeof(void*)*14, v_successProbability_1584_);
lean_ctor_set_uint8(v_reuseFailAlloc_1605_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1586_);
v___x_1593_ = v_reuseFailAlloc_1605_;
goto v_reusejp_1592_;
}
v_reusejp_1592_:
{
lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; uint8_t v___x_1599_; 
lean_inc(v_introGoal_1500_);
v___x_1594_ = lean_apply_1(v_introGoal_1500_, v___x_1593_);
v___x_1595_ = lean_st_ref_set(v_a_1480_, v___x_1594_);
v___x_1596_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1596_);
v___x_1597_ = lean_st_ref_get(v_a_1480_);
v___x_1598_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1598_);
v___x_1599_ = lp_aesop_Aesop_Goal_isActive(v___x_1597_);
if (v___x_1599_ == 0)
{
lean_dec(v_a_1480_);
v___y_1544_ = v___y_1562_;
v___y_1545_ = v___y_1563_;
v_a_1546_ = v_a_1565_;
goto v___jp_1543_;
}
else
{
lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; 
v___x_1600_ = lean_unsigned_to_nat(1u);
v___x_1601_ = lean_mk_empty_array_with_capacity(v___x_1600_);
v___x_1602_ = lean_array_push(v___x_1601_, v_a_1480_);
lean_inc_ref(v_inst_1399_);
v___x_1603_ = lp_aesop_Aesop_enqueueGoals___redArg(v_inst_1399_, v___x_1602_, v_a_1401_);
if (lean_obj_tag(v___x_1603_) == 0)
{
lean_dec_ref_known(v___x_1603_, 1);
v___y_1544_ = v___y_1562_;
v___y_1545_ = v___y_1563_;
v_a_1546_ = v_a_1565_;
goto v___jp_1543_;
}
else
{
lean_object* v_a_1604_; 
lean_dec(v_a_1565_);
lean_del_object(v___x_1485_);
v_a_1604_ = lean_ctor_get(v___x_1603_, 0);
lean_inc(v_a_1604_);
lean_dec_ref_known(v___x_1603_, 1);
v___y_1551_ = v___y_1562_;
v___y_1552_ = v___y_1563_;
v_a_1553_ = v_a_1604_;
goto v___jp_1550_;
}
}
}
}
}
else
{
lean_object* v_a_1608_; 
lean_dec(v_a_1565_);
lean_del_object(v___x_1485_);
lean_dec(v_a_1480_);
v_a_1608_ = lean_ctor_get(v___x_1566_, 0);
lean_inc(v_a_1608_);
lean_dec_ref_known(v___x_1566_, 1);
v___y_1551_ = v___y_1562_;
v___y_1552_ = v___y_1563_;
v_a_1553_ = v_a_1608_;
goto v___jp_1550_;
}
}
else
{
lean_dec(v_a_1480_);
v___y_1556_ = v___y_1562_;
v___y_1557_ = v___y_1563_;
v___y_1558_ = v___x_1564_;
goto v___jp_1555_;
}
}
v___jp_1609_:
{
lean_object* v___x_1614_; lean_object* v_depth_1615_; uint8_t v___x_1616_; 
lean_inc_ref(v_elimGoal_1501_);
v___x_1614_ = lean_apply_1(v_elimGoal_1501_, v___y_1611_);
v_depth_1615_ = lean_ctor_get(v___x_1614_, 4);
lean_inc(v_depth_1615_);
lean_dec_ref(v___x_1614_);
v___x_1616_ = lean_nat_dec_le(v___y_1610_, v_depth_1615_);
lean_dec(v_depth_1615_);
if (v___x_1616_ == 0)
{
v___y_1562_ = v___y_1612_;
v___y_1563_ = v___y_1613_;
goto v___jp_1561_;
}
else
{
lean_object* v___x_130539__overap_1617_; lean_object* v___x_1618_; 
lean_inc(v_toMonadOptions_1477_);
lean_inc_ref(v___x_1473_);
v___x_130539__overap_1617_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1473_, v_toMonadOptions_1477_, v___x_1507_);
lean_inc(v_a_1407_);
lean_inc_ref(v_a_1406_);
lean_inc(v_a_1405_);
lean_inc_ref(v_a_1404_);
lean_inc(v_a_1403_);
lean_inc(v_a_1402_);
lean_inc(v_a_1401_);
lean_inc_ref(v_a_1400_);
v___x_1618_ = lean_apply_9(v___x_130539__overap_1617_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, lean_box(0));
if (lean_obj_tag(v___x_1618_) == 0)
{
lean_object* v_a_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1640_; 
v_a_1619_ = lean_ctor_get(v___x_1618_, 0);
v_isSharedCheck_1640_ = !lean_is_exclusive(v___x_1618_);
if (v_isSharedCheck_1640_ == 0)
{
v___x_1621_ = v___x_1618_;
v_isShared_1622_ = v_isSharedCheck_1640_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_a_1619_);
lean_dec(v___x_1618_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1640_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
uint8_t v___x_1623_; 
v___x_1623_ = lean_unbox(v_a_1619_);
lean_dec(v_a_1619_);
if (v___x_1623_ == 0)
{
lean_object* v___x_1624_; lean_object* v___x_1625_; 
lean_del_object(v___x_1621_);
v___x_1624_ = lean_box(0);
v___x_1625_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__4(v_a_1480_, v___x_1624_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
lean_dec(v_a_1480_);
v___y_1556_ = v___y_1612_;
v___y_1557_ = v___y_1613_;
v___y_1558_ = v___x_1625_;
goto v___jp_1555_;
}
else
{
lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1629_; 
v___x_1626_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1, &lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1);
lean_inc(v___y_1610_);
v___x_1627_ = l_Nat_reprFast(v___y_1610_);
if (v_isShared_1622_ == 0)
{
lean_ctor_set_tag(v___x_1621_, 3);
lean_ctor_set(v___x_1621_, 0, v___x_1627_);
v___x_1629_ = v___x_1621_;
goto v_reusejp_1628_;
}
else
{
lean_object* v_reuseFailAlloc_1639_; 
v_reuseFailAlloc_1639_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1639_, 0, v___x_1627_);
v___x_1629_ = v_reuseFailAlloc_1639_;
goto v_reusejp_1628_;
}
v_reusejp_1628_:
{
lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_130553__overap_1634_; lean_object* v___x_1635_; 
v___x_1630_ = l_Lean_MessageData_ofFormat(v___x_1629_);
v___x_1631_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1631_, 0, v___x_1626_);
lean_ctor_set(v___x_1631_, 1, v___x_1630_);
v___x_1632_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3, &lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3);
v___x_1633_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1633_, 0, v___x_1631_);
lean_ctor_set(v___x_1633_, 1, v___x_1632_);
lean_inc(v_traceClass_1515_);
lean_inc_ref(v___x_1475_);
lean_inc_ref(v___x_1473_);
v___x_130553__overap_1634_ = l_Lean_addTrace___redArg(v___x_1473_, v___x_1474_, v___x_1475_, v___f_1506_, v_traceClass_1515_, v___x_1633_);
lean_inc(v_a_1407_);
lean_inc_ref(v_a_1406_);
lean_inc(v_a_1405_);
lean_inc_ref(v_a_1404_);
lean_inc(v_a_1403_);
lean_inc(v_a_1402_);
lean_inc(v_a_1401_);
lean_inc_ref(v_a_1400_);
v___x_1635_ = lean_apply_9(v___x_130553__overap_1634_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, lean_box(0));
if (lean_obj_tag(v___x_1635_) == 0)
{
lean_object* v_a_1636_; lean_object* v___x_1637_; 
v_a_1636_ = lean_ctor_get(v___x_1635_, 0);
lean_inc(v_a_1636_);
lean_dec_ref_known(v___x_1635_, 1);
v___x_1637_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__4(v_a_1480_, v_a_1636_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
lean_dec(v_a_1480_);
v___y_1556_ = v___y_1612_;
v___y_1557_ = v___y_1613_;
v___y_1558_ = v___x_1637_;
goto v___jp_1555_;
}
else
{
lean_object* v_a_1638_; 
lean_del_object(v___x_1485_);
lean_dec(v_a_1480_);
v_a_1638_ = lean_ctor_get(v___x_1635_, 0);
lean_inc(v_a_1638_);
lean_dec_ref_known(v___x_1635_, 1);
v___y_1551_ = v___y_1612_;
v___y_1552_ = v___y_1613_;
v_a_1553_ = v_a_1638_;
goto v___jp_1550_;
}
}
}
}
}
else
{
lean_object* v_a_1641_; 
lean_del_object(v___x_1485_);
lean_dec(v_a_1480_);
v_a_1641_ = lean_ctor_get(v___x_1618_, 0);
lean_inc(v_a_1641_);
lean_dec_ref_known(v___x_1618_, 1);
v___y_1551_ = v___y_1612_;
v___y_1552_ = v___y_1613_;
v_a_1553_ = v_a_1641_;
goto v___jp_1550_;
}
}
}
v___jp_1642_:
{
lean_object* v___x_1646_; lean_object* v___x_1647_; double v___x_1648_; double v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_130586__overap_1654_; lean_object* v___x_1655_; 
v___x_1646_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1646_);
v___x_1647_ = lean_io_get_num_heartbeats();
v___x_1648_ = lean_float_of_nat(v___y_1644_);
v___x_1649_ = lean_float_of_nat(v___x_1647_);
v___x_1650_ = lean_box_float(v___x_1648_);
v___x_1651_ = lean_box_float(v___x_1649_);
v___x_1652_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1652_, 0, v___x_1650_);
lean_ctor_set(v___x_1652_, 1, v___x_1651_);
v___x_1653_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1653_, 0, v_a_1645_);
lean_ctor_set(v___x_1653_, 1, v___x_1652_);
lean_inc(v_traceClass_1515_);
v___x_130586__overap_1654_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1473_, v___x_1474_, v___x_1475_, v___f_1506_, lean_box(0), v___x_1478_, v___f_1516_, v_traceClass_1515_, v_hasTrace_1503_, v___x_1520_, v_options_1499_, v___x_1523_, v___y_1643_, v___x_1519_, v___x_1653_);
lean_inc(v_a_1407_);
lean_inc_ref(v_a_1406_);
lean_inc(v_a_1405_);
lean_inc_ref(v_a_1404_);
lean_inc(v_a_1403_);
lean_inc(v_a_1402_);
lean_inc(v_a_1401_);
lean_inc_ref(v_a_1400_);
v___x_1655_ = lean_apply_9(v___x_130586__overap_1654_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, lean_box(0));
v___y_1410_ = v___x_1655_;
goto v___jp_1409_;
}
v___jp_1656_:
{
lean_object* v___x_1660_; 
v___x_1660_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1660_, 0, v_a_1659_);
v___y_1643_ = v___y_1657_;
v___y_1644_ = v___y_1658_;
v_a_1645_ = v___x_1660_;
goto v___jp_1642_;
}
v___jp_1661_:
{
lean_object* v___x_1665_; 
v___x_1665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1665_, 0, v_a_1664_);
v___y_1643_ = v___y_1662_;
v___y_1644_ = v___y_1663_;
v_a_1645_ = v___x_1665_;
goto v___jp_1642_;
}
v___jp_1666_:
{
if (lean_obj_tag(v___y_1669_) == 0)
{
lean_object* v_a_1670_; 
v_a_1670_ = lean_ctor_get(v___y_1669_, 0);
lean_inc(v_a_1670_);
lean_dec_ref_known(v___y_1669_, 1);
v___y_1662_ = v___y_1667_;
v___y_1663_ = v___y_1668_;
v_a_1664_ = v_a_1670_;
goto v___jp_1661_;
}
else
{
lean_object* v_a_1671_; 
v_a_1671_ = lean_ctor_get(v___y_1669_, 0);
lean_inc(v_a_1671_);
lean_dec_ref_known(v___y_1669_, 1);
v___y_1657_ = v___y_1667_;
v___y_1658_ = v___y_1668_;
v_a_1659_ = v_a_1671_;
goto v___jp_1656_;
}
}
v___jp_1672_:
{
lean_object* v___x_1675_; 
lean_inc(v_a_1480_);
lean_inc_ref(v_inst_1399_);
v___x_1675_ = lp_aesop_Aesop_expandGoal___redArg(v_inst_1399_, v_a_1480_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
if (lean_obj_tag(v___x_1675_) == 0)
{
lean_object* v_a_1676_; lean_object* v___x_1677_; 
v_a_1676_ = lean_ctor_get(v___x_1675_, 0);
lean_inc(v_a_1676_);
lean_dec_ref_known(v___x_1675_, 1);
v___x_1677_ = lp_aesop_Aesop_getIteration___redArg(v_a_1401_);
if (lean_obj_tag(v___x_1677_) == 0)
{
lean_object* v_a_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v_id_1682_; lean_object* v_parent_1683_; lean_object* v_children_1684_; lean_object* v_origin_1685_; lean_object* v_depth_1686_; uint8_t v_state_1687_; uint8_t v_isIrrelevant_1688_; uint8_t v_isForcedUnprovable_1689_; lean_object* v_preNormGoal_1690_; lean_object* v_normalizationState_1691_; lean_object* v_mvars_1692_; lean_object* v_forwardState_1693_; lean_object* v_forwardRuleMatches_1694_; double v_successProbability_1695_; lean_object* v_addedInIteration_1696_; uint8_t v_unsafeRulesSelected_1697_; lean_object* v_unsafeQueue_1698_; lean_object* v_failedRapps_1699_; lean_object* v___x_1701_; uint8_t v_isShared_1702_; uint8_t v_isSharedCheck_1717_; 
v_a_1678_ = lean_ctor_get(v___x_1677_, 0);
lean_inc(v_a_1678_);
lean_dec_ref_known(v___x_1677_, 1);
v___x_1679_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1679_);
v___x_1680_ = lean_st_ref_take(v_a_1480_);
lean_inc_ref(v_elimGoal_1501_);
v___x_1681_ = lean_apply_1(v_elimGoal_1501_, v___x_1680_);
v_id_1682_ = lean_ctor_get(v___x_1681_, 0);
v_parent_1683_ = lean_ctor_get(v___x_1681_, 1);
v_children_1684_ = lean_ctor_get(v___x_1681_, 2);
v_origin_1685_ = lean_ctor_get(v___x_1681_, 3);
v_depth_1686_ = lean_ctor_get(v___x_1681_, 4);
v_state_1687_ = lean_ctor_get_uint8(v___x_1681_, sizeof(void*)*14 + 8);
v_isIrrelevant_1688_ = lean_ctor_get_uint8(v___x_1681_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1689_ = lean_ctor_get_uint8(v___x_1681_, sizeof(void*)*14 + 10);
v_preNormGoal_1690_ = lean_ctor_get(v___x_1681_, 5);
v_normalizationState_1691_ = lean_ctor_get(v___x_1681_, 6);
v_mvars_1692_ = lean_ctor_get(v___x_1681_, 7);
v_forwardState_1693_ = lean_ctor_get(v___x_1681_, 8);
v_forwardRuleMatches_1694_ = lean_ctor_get(v___x_1681_, 9);
v_successProbability_1695_ = lean_ctor_get_float(v___x_1681_, sizeof(void*)*14);
v_addedInIteration_1696_ = lean_ctor_get(v___x_1681_, 10);
v_unsafeRulesSelected_1697_ = lean_ctor_get_uint8(v___x_1681_, sizeof(void*)*14 + 11);
v_unsafeQueue_1698_ = lean_ctor_get(v___x_1681_, 12);
v_failedRapps_1699_ = lean_ctor_get(v___x_1681_, 13);
v_isSharedCheck_1717_ = !lean_is_exclusive(v___x_1681_);
if (v_isSharedCheck_1717_ == 0)
{
lean_object* v_unused_1718_; 
v_unused_1718_ = lean_ctor_get(v___x_1681_, 11);
lean_dec(v_unused_1718_);
v___x_1701_ = v___x_1681_;
v_isShared_1702_ = v_isSharedCheck_1717_;
goto v_resetjp_1700_;
}
else
{
lean_inc(v_failedRapps_1699_);
lean_inc(v_unsafeQueue_1698_);
lean_inc(v_addedInIteration_1696_);
lean_inc(v_forwardRuleMatches_1694_);
lean_inc(v_forwardState_1693_);
lean_inc(v_mvars_1692_);
lean_inc(v_normalizationState_1691_);
lean_inc(v_preNormGoal_1690_);
lean_inc(v_depth_1686_);
lean_inc(v_origin_1685_);
lean_inc(v_children_1684_);
lean_inc(v_parent_1683_);
lean_inc(v_id_1682_);
lean_dec(v___x_1681_);
v___x_1701_ = lean_box(0);
v_isShared_1702_ = v_isSharedCheck_1717_;
goto v_resetjp_1700_;
}
v_resetjp_1700_:
{
lean_object* v___x_1704_; 
if (v_isShared_1702_ == 0)
{
lean_ctor_set(v___x_1701_, 11, v_a_1678_);
v___x_1704_ = v___x_1701_;
goto v_reusejp_1703_;
}
else
{
lean_object* v_reuseFailAlloc_1716_; 
v_reuseFailAlloc_1716_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1716_, 0, v_id_1682_);
lean_ctor_set(v_reuseFailAlloc_1716_, 1, v_parent_1683_);
lean_ctor_set(v_reuseFailAlloc_1716_, 2, v_children_1684_);
lean_ctor_set(v_reuseFailAlloc_1716_, 3, v_origin_1685_);
lean_ctor_set(v_reuseFailAlloc_1716_, 4, v_depth_1686_);
lean_ctor_set(v_reuseFailAlloc_1716_, 5, v_preNormGoal_1690_);
lean_ctor_set(v_reuseFailAlloc_1716_, 6, v_normalizationState_1691_);
lean_ctor_set(v_reuseFailAlloc_1716_, 7, v_mvars_1692_);
lean_ctor_set(v_reuseFailAlloc_1716_, 8, v_forwardState_1693_);
lean_ctor_set(v_reuseFailAlloc_1716_, 9, v_forwardRuleMatches_1694_);
lean_ctor_set(v_reuseFailAlloc_1716_, 10, v_addedInIteration_1696_);
lean_ctor_set(v_reuseFailAlloc_1716_, 11, v_a_1678_);
lean_ctor_set(v_reuseFailAlloc_1716_, 12, v_unsafeQueue_1698_);
lean_ctor_set(v_reuseFailAlloc_1716_, 13, v_failedRapps_1699_);
lean_ctor_set_uint8(v_reuseFailAlloc_1716_, sizeof(void*)*14 + 8, v_state_1687_);
lean_ctor_set_uint8(v_reuseFailAlloc_1716_, sizeof(void*)*14 + 9, v_isIrrelevant_1688_);
lean_ctor_set_uint8(v_reuseFailAlloc_1716_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1689_);
lean_ctor_set_float(v_reuseFailAlloc_1716_, sizeof(void*)*14, v_successProbability_1695_);
lean_ctor_set_uint8(v_reuseFailAlloc_1716_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1697_);
v___x_1704_ = v_reuseFailAlloc_1716_;
goto v_reusejp_1703_;
}
v_reusejp_1703_:
{
lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; uint8_t v___x_1710_; 
lean_inc(v_introGoal_1500_);
v___x_1705_ = lean_apply_1(v_introGoal_1500_, v___x_1704_);
v___x_1706_ = lean_st_ref_set(v_a_1480_, v___x_1705_);
v___x_1707_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1707_);
v___x_1708_ = lean_st_ref_get(v_a_1480_);
v___x_1709_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1709_);
v___x_1710_ = lp_aesop_Aesop_Goal_isActive(v___x_1708_);
if (v___x_1710_ == 0)
{
lean_dec(v_a_1480_);
v___y_1662_ = v___y_1673_;
v___y_1663_ = v___y_1674_;
v_a_1664_ = v_a_1676_;
goto v___jp_1661_;
}
else
{
lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; 
v___x_1711_ = lean_unsigned_to_nat(1u);
v___x_1712_ = lean_mk_empty_array_with_capacity(v___x_1711_);
v___x_1713_ = lean_array_push(v___x_1712_, v_a_1480_);
lean_inc_ref(v_inst_1399_);
v___x_1714_ = lp_aesop_Aesop_enqueueGoals___redArg(v_inst_1399_, v___x_1713_, v_a_1401_);
if (lean_obj_tag(v___x_1714_) == 0)
{
lean_dec_ref_known(v___x_1714_, 1);
v___y_1662_ = v___y_1673_;
v___y_1663_ = v___y_1674_;
v_a_1664_ = v_a_1676_;
goto v___jp_1661_;
}
else
{
lean_object* v_a_1715_; 
lean_dec(v_a_1676_);
v_a_1715_ = lean_ctor_get(v___x_1714_, 0);
lean_inc(v_a_1715_);
lean_dec_ref_known(v___x_1714_, 1);
v___y_1657_ = v___y_1673_;
v___y_1658_ = v___y_1674_;
v_a_1659_ = v_a_1715_;
goto v___jp_1656_;
}
}
}
}
}
else
{
lean_object* v_a_1719_; 
lean_dec(v_a_1676_);
lean_dec(v_a_1480_);
v_a_1719_ = lean_ctor_get(v___x_1677_, 0);
lean_inc(v_a_1719_);
lean_dec_ref_known(v___x_1677_, 1);
v___y_1657_ = v___y_1673_;
v___y_1658_ = v___y_1674_;
v_a_1659_ = v_a_1719_;
goto v___jp_1656_;
}
}
else
{
lean_dec(v_a_1480_);
v___y_1667_ = v___y_1673_;
v___y_1668_ = v___y_1674_;
v___y_1669_ = v___x_1675_;
goto v___jp_1666_;
}
}
v___jp_1720_:
{
lean_object* v___x_130417__overap_1721_; lean_object* v___x_1722_; 
lean_inc_ref(v___x_1473_);
v___x_130417__overap_1721_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_1473_, v___x_1474_);
lean_inc(v_a_1407_);
lean_inc_ref(v_a_1406_);
lean_inc(v_a_1405_);
lean_inc_ref(v_a_1404_);
lean_inc(v_a_1403_);
lean_inc(v_a_1402_);
lean_inc(v_a_1401_);
lean_inc_ref(v_a_1400_);
v___x_1722_ = lean_apply_9(v___x_130417__overap_1721_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, lean_box(0));
if (lean_obj_tag(v___x_1722_) == 0)
{
lean_object* v_a_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; uint8_t v___x_1727_; 
v_a_1723_ = lean_ctor_get(v___x_1722_, 0);
lean_inc(v_a_1723_);
lean_dec_ref_known(v___x_1722_, 1);
v___x_1724_ = l_Lean_KVMap_instValueBool;
v___x_1725_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1726_ = l_Lean_Option_get___redArg(v___x_1724_, v_options_1499_, v___x_1725_);
v___x_1727_ = lean_unbox(v___x_1726_);
if (v___x_1727_ == 0)
{
lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; 
v___x_1728_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1728_);
v___x_1729_ = lean_io_mono_nanos_now();
v___x_1730_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1730_);
v___x_1731_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_snd_1494_, v___f_1508_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
if (lean_obj_tag(v___x_1731_) == 0)
{
lean_object* v_options_1732_; lean_object* v___x_1733_; lean_object* v_toOptions_1734_; lean_object* v_a_1735_; lean_object* v_maxRuleApplicationDepth_1736_; lean_object* v___x_1737_; uint8_t v___x_1738_; 
lean_dec_ref_known(v___x_1731_, 1);
v_options_1732_ = lean_ctor_get(v_a_1400_, 2);
v___x_1733_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(v_a_1480_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
v_toOptions_1734_ = lean_ctor_get(v_options_1732_, 0);
v_a_1735_ = lean_ctor_get(v___x_1733_, 0);
lean_inc(v_a_1735_);
lean_dec_ref(v___x_1733_);
v_maxRuleApplicationDepth_1736_ = lean_ctor_get(v_toOptions_1734_, 0);
v___x_1737_ = lean_unsigned_to_nat(0u);
v___x_1738_ = lean_nat_dec_eq(v_maxRuleApplicationDepth_1736_, v___x_1737_);
if (v___x_1738_ == 0)
{
lean_dec(v___x_1726_);
v___y_1610_ = v_maxRuleApplicationDepth_1736_;
v___y_1611_ = v_a_1735_;
v___y_1612_ = v_a_1723_;
v___y_1613_ = v___x_1729_;
goto v___jp_1609_;
}
else
{
uint8_t v___x_1739_; 
v___x_1739_ = lean_unbox(v___x_1726_);
lean_dec(v___x_1726_);
if (v___x_1739_ == 0)
{
lean_dec(v_a_1735_);
v___y_1562_ = v_a_1723_;
v___y_1563_ = v___x_1729_;
goto v___jp_1561_;
}
else
{
v___y_1610_ = v_maxRuleApplicationDepth_1736_;
v___y_1611_ = v_a_1735_;
v___y_1612_ = v_a_1723_;
v___y_1613_ = v___x_1729_;
goto v___jp_1609_;
}
}
}
else
{
lean_object* v_a_1740_; 
lean_dec(v___x_1726_);
lean_del_object(v___x_1485_);
lean_dec(v_a_1480_);
v_a_1740_ = lean_ctor_get(v___x_1731_, 0);
lean_inc(v_a_1740_);
lean_dec_ref_known(v___x_1731_, 1);
v___y_1551_ = v_a_1723_;
v___y_1552_ = v___x_1729_;
v_a_1553_ = v_a_1740_;
goto v___jp_1550_;
}
}
else
{
lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; 
lean_del_object(v___x_1496_);
lean_del_object(v___x_1485_);
v___x_1741_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1741_);
v___x_1742_ = lean_io_get_num_heartbeats();
v___x_1743_ = lean_st_ref_get(v_a_1401_);
lean_dec(v___x_1743_);
v___x_1744_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_snd_1494_, v___f_1508_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
if (lean_obj_tag(v___x_1744_) == 0)
{
lean_object* v_options_1745_; lean_object* v___x_1746_; lean_object* v_toOptions_1747_; lean_object* v_a_1748_; lean_object* v_maxRuleApplicationDepth_1749_; lean_object* v___x_1750_; uint8_t v___x_1751_; 
lean_dec_ref_known(v___x_1744_, 1);
v_options_1745_ = lean_ctor_get(v_a_1400_, 2);
v___x_1746_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__0(v_a_1480_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
v_toOptions_1747_ = lean_ctor_get(v_options_1745_, 0);
v_a_1748_ = lean_ctor_get(v___x_1746_, 0);
lean_inc(v_a_1748_);
lean_dec_ref(v___x_1746_);
v_maxRuleApplicationDepth_1749_ = lean_ctor_get(v_toOptions_1747_, 0);
v___x_1750_ = lean_unsigned_to_nat(0u);
v___x_1751_ = lean_nat_dec_eq(v_maxRuleApplicationDepth_1749_, v___x_1750_);
if (v___x_1751_ == 0)
{
uint8_t v___x_1752_; 
v___x_1752_ = lean_unbox(v___x_1726_);
lean_dec(v___x_1726_);
if (v___x_1752_ == 0)
{
lean_dec(v_a_1748_);
v___y_1673_ = v_a_1723_;
v___y_1674_ = v___x_1742_;
goto v___jp_1672_;
}
else
{
lean_object* v___x_1753_; lean_object* v_depth_1754_; uint8_t v___x_1755_; 
lean_inc_ref(v_elimGoal_1501_);
v___x_1753_ = lean_apply_1(v_elimGoal_1501_, v_a_1748_);
v_depth_1754_ = lean_ctor_get(v___x_1753_, 4);
lean_inc(v_depth_1754_);
lean_dec_ref(v___x_1753_);
v___x_1755_ = lean_nat_dec_le(v_maxRuleApplicationDepth_1749_, v_depth_1754_);
lean_dec(v_depth_1754_);
if (v___x_1755_ == 0)
{
v___y_1673_ = v_a_1723_;
v___y_1674_ = v___x_1742_;
goto v___jp_1672_;
}
else
{
lean_object* v___x_130680__overap_1756_; lean_object* v___x_1757_; 
lean_inc(v_toMonadOptions_1477_);
lean_inc_ref(v___x_1473_);
v___x_130680__overap_1756_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_1473_, v_toMonadOptions_1477_, v___x_1507_);
lean_inc(v_a_1407_);
lean_inc_ref(v_a_1406_);
lean_inc(v_a_1405_);
lean_inc_ref(v_a_1404_);
lean_inc(v_a_1403_);
lean_inc(v_a_1402_);
lean_inc(v_a_1401_);
lean_inc_ref(v_a_1400_);
v___x_1757_ = lean_apply_9(v___x_130680__overap_1756_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, lean_box(0));
if (lean_obj_tag(v___x_1757_) == 0)
{
lean_object* v_a_1758_; lean_object* v___x_1760_; uint8_t v_isShared_1761_; uint8_t v_isSharedCheck_1779_; 
v_a_1758_ = lean_ctor_get(v___x_1757_, 0);
v_isSharedCheck_1779_ = !lean_is_exclusive(v___x_1757_);
if (v_isSharedCheck_1779_ == 0)
{
v___x_1760_ = v___x_1757_;
v_isShared_1761_ = v_isSharedCheck_1779_;
goto v_resetjp_1759_;
}
else
{
lean_inc(v_a_1758_);
lean_dec(v___x_1757_);
v___x_1760_ = lean_box(0);
v_isShared_1761_ = v_isSharedCheck_1779_;
goto v_resetjp_1759_;
}
v_resetjp_1759_:
{
uint8_t v___x_1762_; 
v___x_1762_ = lean_unbox(v_a_1758_);
lean_dec(v_a_1758_);
if (v___x_1762_ == 0)
{
lean_object* v___x_1763_; lean_object* v___x_1764_; 
lean_del_object(v___x_1760_);
v___x_1763_ = lean_box(0);
v___x_1764_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__4(v_a_1480_, v___x_1763_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
lean_dec(v_a_1480_);
v___y_1667_ = v_a_1723_;
v___y_1668_ = v___x_1742_;
v___y_1669_ = v___x_1764_;
goto v___jp_1666_;
}
else
{
lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1768_; 
v___x_1765_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1, &lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__1);
lean_inc(v_maxRuleApplicationDepth_1749_);
v___x_1766_ = l_Nat_reprFast(v_maxRuleApplicationDepth_1749_);
if (v_isShared_1761_ == 0)
{
lean_ctor_set_tag(v___x_1760_, 3);
lean_ctor_set(v___x_1760_, 0, v___x_1766_);
v___x_1768_ = v___x_1760_;
goto v_reusejp_1767_;
}
else
{
lean_object* v_reuseFailAlloc_1778_; 
v_reuseFailAlloc_1778_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1778_, 0, v___x_1766_);
v___x_1768_ = v_reuseFailAlloc_1778_;
goto v_reusejp_1767_;
}
v_reusejp_1767_:
{
lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_130694__overap_1773_; lean_object* v___x_1774_; 
v___x_1769_ = l_Lean_MessageData_ofFormat(v___x_1768_);
v___x_1770_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1770_, 0, v___x_1765_);
lean_ctor_set(v___x_1770_, 1, v___x_1769_);
v___x_1771_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3, &lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___lam__3___closed__3);
v___x_1772_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1772_, 0, v___x_1770_);
lean_ctor_set(v___x_1772_, 1, v___x_1771_);
lean_inc(v_traceClass_1515_);
lean_inc_ref(v___x_1475_);
lean_inc_ref(v___x_1473_);
v___x_130694__overap_1773_ = l_Lean_addTrace___redArg(v___x_1473_, v___x_1474_, v___x_1475_, v___f_1506_, v_traceClass_1515_, v___x_1772_);
lean_inc(v_a_1407_);
lean_inc_ref(v_a_1406_);
lean_inc(v_a_1405_);
lean_inc_ref(v_a_1404_);
lean_inc(v_a_1403_);
lean_inc(v_a_1402_);
lean_inc(v_a_1401_);
lean_inc_ref(v_a_1400_);
v___x_1774_ = lean_apply_9(v___x_130694__overap_1773_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_, lean_box(0));
if (lean_obj_tag(v___x_1774_) == 0)
{
lean_object* v_a_1775_; lean_object* v___x_1776_; 
v_a_1775_ = lean_ctor_get(v___x_1774_, 0);
lean_inc(v_a_1775_);
lean_dec_ref_known(v___x_1774_, 1);
v___x_1776_ = lp_aesop_Aesop_expandNextGoal___redArg___lam__4(v_a_1480_, v_a_1775_, v_a_1400_, v_a_1401_, v_a_1402_, v_a_1403_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
lean_dec(v_a_1480_);
v___y_1667_ = v_a_1723_;
v___y_1668_ = v___x_1742_;
v___y_1669_ = v___x_1776_;
goto v___jp_1666_;
}
else
{
lean_object* v_a_1777_; 
lean_dec(v_a_1480_);
v_a_1777_ = lean_ctor_get(v___x_1774_, 0);
lean_inc(v_a_1777_);
lean_dec_ref_known(v___x_1774_, 1);
v___y_1657_ = v_a_1723_;
v___y_1658_ = v___x_1742_;
v_a_1659_ = v_a_1777_;
goto v___jp_1656_;
}
}
}
}
}
else
{
lean_object* v_a_1780_; 
lean_dec(v_a_1480_);
v_a_1780_ = lean_ctor_get(v___x_1757_, 0);
lean_inc(v_a_1780_);
lean_dec_ref_known(v___x_1757_, 1);
v___y_1657_ = v_a_1723_;
v___y_1658_ = v___x_1742_;
v_a_1659_ = v_a_1780_;
goto v___jp_1656_;
}
}
}
}
else
{
lean_dec(v_a_1748_);
lean_dec(v___x_1726_);
v___y_1673_ = v_a_1723_;
v___y_1674_ = v___x_1742_;
goto v___jp_1672_;
}
}
else
{
lean_object* v_a_1781_; 
lean_dec(v___x_1726_);
lean_dec(v_a_1480_);
v_a_1781_ = lean_ctor_get(v___x_1744_, 0);
lean_inc(v_a_1781_);
lean_dec_ref_known(v___x_1744_, 1);
v___y_1657_ = v_a_1723_;
v___y_1658_ = v___x_1742_;
v_a_1659_ = v_a_1781_;
goto v___jp_1656_;
}
}
}
else
{
lean_object* v_a_1782_; lean_object* v___x_1784_; uint8_t v_isShared_1785_; uint8_t v_isSharedCheck_1789_; 
lean_dec_ref(v___x_1519_);
lean_dec_ref(v___f_1508_);
lean_del_object(v___x_1496_);
lean_dec(v_snd_1494_);
lean_del_object(v___x_1485_);
lean_dec(v_a_1480_);
lean_dec_ref(v___x_1475_);
lean_dec_ref(v___x_1473_);
lean_dec_ref(v_inst_1399_);
v_a_1782_ = lean_ctor_get(v___x_1722_, 0);
v_isSharedCheck_1789_ = !lean_is_exclusive(v___x_1722_);
if (v_isSharedCheck_1789_ == 0)
{
v___x_1784_ = v___x_1722_;
v_isShared_1785_ = v_isSharedCheck_1789_;
goto v_resetjp_1783_;
}
else
{
lean_inc(v_a_1782_);
lean_dec(v___x_1722_);
v___x_1784_ = lean_box(0);
v_isShared_1785_ = v_isSharedCheck_1789_;
goto v_resetjp_1783_;
}
v_resetjp_1783_:
{
lean_object* v___x_1787_; 
if (v_isShared_1785_ == 0)
{
v___x_1787_ = v___x_1784_;
goto v_reusejp_1786_;
}
else
{
lean_object* v_reuseFailAlloc_1788_; 
v_reuseFailAlloc_1788_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1788_, 0, v_a_1782_);
v___x_1787_ = v_reuseFailAlloc_1788_;
goto v_reusejp_1786_;
}
v_reusejp_1786_:
{
return v___x_1787_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1798_; lean_object* v___x_1800_; uint8_t v_isShared_1801_; uint8_t v_isSharedCheck_1805_; 
lean_del_object(v___x_1485_);
lean_dec(v_a_1483_);
lean_dec_ref(v___f_1481_);
lean_dec(v_a_1480_);
lean_dec_ref(v___x_1475_);
lean_dec_ref(v___x_1473_);
lean_dec_ref(v___x_1472_);
lean_dec_ref(v_inst_1399_);
v_a_1798_ = lean_ctor_get(v___x_1491_, 0);
v_isSharedCheck_1805_ = !lean_is_exclusive(v___x_1491_);
if (v_isSharedCheck_1805_ == 0)
{
v___x_1800_ = v___x_1491_;
v_isShared_1801_ = v_isSharedCheck_1805_;
goto v_resetjp_1799_;
}
else
{
lean_inc(v_a_1798_);
lean_dec(v___x_1491_);
v___x_1800_ = lean_box(0);
v_isShared_1801_ = v_isSharedCheck_1805_;
goto v_resetjp_1799_;
}
v_resetjp_1799_:
{
lean_object* v___x_1803_; 
if (v_isShared_1801_ == 0)
{
v___x_1803_ = v___x_1800_;
goto v_reusejp_1802_;
}
else
{
lean_object* v_reuseFailAlloc_1804_; 
v_reuseFailAlloc_1804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1804_, 0, v_a_1798_);
v___x_1803_ = v_reuseFailAlloc_1804_;
goto v_reusejp_1802_;
}
v_reusejp_1802_:
{
return v___x_1803_;
}
}
}
}
else
{
lean_object* v_a_1806_; lean_object* v___x_1808_; uint8_t v_isShared_1809_; uint8_t v_isSharedCheck_1813_; 
lean_del_object(v___x_1485_);
lean_dec(v_a_1483_);
lean_dec_ref(v___f_1481_);
lean_dec(v_a_1480_);
lean_dec_ref(v___x_1475_);
lean_dec_ref(v___x_1473_);
lean_dec_ref(v___x_1472_);
lean_dec_ref(v_inst_1399_);
v_a_1806_ = lean_ctor_get(v___x_1488_, 0);
v_isSharedCheck_1813_ = !lean_is_exclusive(v___x_1488_);
if (v_isSharedCheck_1813_ == 0)
{
v___x_1808_ = v___x_1488_;
v_isShared_1809_ = v_isSharedCheck_1813_;
goto v_resetjp_1807_;
}
else
{
lean_inc(v_a_1806_);
lean_dec(v___x_1488_);
v___x_1808_ = lean_box(0);
v_isShared_1809_ = v_isSharedCheck_1813_;
goto v_resetjp_1807_;
}
v_resetjp_1807_:
{
lean_object* v___x_1811_; 
if (v_isShared_1809_ == 0)
{
v___x_1811_ = v___x_1808_;
goto v_reusejp_1810_;
}
else
{
lean_object* v_reuseFailAlloc_1812_; 
v_reuseFailAlloc_1812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1812_, 0, v_a_1806_);
v___x_1811_ = v_reuseFailAlloc_1812_;
goto v_reusejp_1810_;
}
v_reusejp_1810_:
{
return v___x_1811_;
}
}
}
}
}
else
{
lean_object* v_a_1815_; lean_object* v___x_1817_; uint8_t v_isShared_1818_; uint8_t v_isSharedCheck_1822_; 
lean_dec_ref(v___x_1475_);
lean_dec_ref(v___x_1473_);
lean_dec_ref(v___x_1472_);
lean_dec_ref(v_inst_1399_);
v_a_1815_ = lean_ctor_get(v___x_1479_, 0);
v_isSharedCheck_1822_ = !lean_is_exclusive(v___x_1479_);
if (v_isSharedCheck_1822_ == 0)
{
v___x_1817_ = v___x_1479_;
v_isShared_1818_ = v_isSharedCheck_1822_;
goto v_resetjp_1816_;
}
else
{
lean_inc(v_a_1815_);
lean_dec(v___x_1479_);
v___x_1817_ = lean_box(0);
v_isShared_1818_ = v_isSharedCheck_1822_;
goto v_resetjp_1816_;
}
v_resetjp_1816_:
{
lean_object* v___x_1820_; 
if (v_isShared_1818_ == 0)
{
v___x_1820_ = v___x_1817_;
goto v_reusejp_1819_;
}
else
{
lean_object* v_reuseFailAlloc_1821_; 
v_reuseFailAlloc_1821_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1821_, 0, v_a_1815_);
v___x_1820_ = v_reuseFailAlloc_1821_;
goto v_reusejp_1819_;
}
v_reusejp_1819_:
{
return v___x_1820_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___redArg___boxed(lean_object* v_inst_1829_, lean_object* v_a_1830_, lean_object* v_a_1831_, lean_object* v_a_1832_, lean_object* v_a_1833_, lean_object* v_a_1834_, lean_object* v_a_1835_, lean_object* v_a_1836_, lean_object* v_a_1837_, lean_object* v_a_1838_){
_start:
{
lean_object* v_res_1839_; 
v_res_1839_ = lp_aesop_Aesop_expandNextGoal___redArg(v_inst_1829_, v_a_1830_, v_a_1831_, v_a_1832_, v_a_1833_, v_a_1834_, v_a_1835_, v_a_1836_, v_a_1837_);
lean_dec(v_a_1837_);
lean_dec_ref(v_a_1836_);
lean_dec(v_a_1835_);
lean_dec_ref(v_a_1834_);
lean_dec(v_a_1833_);
lean_dec(v_a_1832_);
lean_dec(v_a_1831_);
lean_dec_ref(v_a_1830_);
return v_res_1839_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal(lean_object* v_Q_1840_, lean_object* v_inst_1841_, lean_object* v_a_1842_, lean_object* v_a_1843_, lean_object* v_a_1844_, lean_object* v_a_1845_, lean_object* v_a_1846_, lean_object* v_a_1847_, lean_object* v_a_1848_, lean_object* v_a_1849_){
_start:
{
lean_object* v___x_1851_; 
v___x_1851_ = lp_aesop_Aesop_expandNextGoal___redArg(v_inst_1841_, v_a_1842_, v_a_1843_, v_a_1844_, v_a_1845_, v_a_1846_, v_a_1847_, v_a_1848_, v_a_1849_);
return v___x_1851_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandNextGoal___boxed(lean_object* v_Q_1852_, lean_object* v_inst_1853_, lean_object* v_a_1854_, lean_object* v_a_1855_, lean_object* v_a_1856_, lean_object* v_a_1857_, lean_object* v_a_1858_, lean_object* v_a_1859_, lean_object* v_a_1860_, lean_object* v_a_1861_, lean_object* v_a_1862_){
_start:
{
lean_object* v_res_1863_; 
v_res_1863_ = lp_aesop_Aesop_expandNextGoal(v_Q_1852_, v_inst_1853_, v_a_1854_, v_a_1855_, v_a_1856_, v_a_1857_, v_a_1858_, v_a_1859_, v_a_1860_, v_a_1861_);
lean_dec(v_a_1861_);
lean_dec_ref(v_a_1860_);
lean_dec(v_a_1859_);
lean_dec_ref(v_a_1858_);
lean_dec(v_a_1857_);
lean_dec(v_a_1856_);
lean_dec(v_a_1855_);
lean_dec_ref(v_a_1854_);
return v_res_1863_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__1(void){
_start:
{
lean_object* v___x_1865_; lean_object* v___x_1866_; 
v___x_1865_ = ((lean_object*)(lp_aesop_Aesop_checkGoalLimit___redArg___closed__0));
v___x_1866_ = l_Lean_stringToMessageData(v___x_1865_);
return v___x_1866_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__3(void){
_start:
{
lean_object* v___x_1868_; lean_object* v___x_1869_; 
v___x_1868_ = ((lean_object*)(lp_aesop_Aesop_checkGoalLimit___redArg___closed__2));
v___x_1869_ = l_Lean_stringToMessageData(v___x_1868_);
return v___x_1869_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___redArg(lean_object* v_a_1870_, lean_object* v_a_1871_, lean_object* v_a_1872_){
_start:
{
lean_object* v___x_1874_; 
v___x_1874_ = lp_aesop_Aesop_getTree___redArg(v_a_1871_, v_a_1872_);
if (lean_obj_tag(v___x_1874_) == 0)
{
lean_object* v_a_1875_; lean_object* v___x_1877_; uint8_t v_isShared_1878_; uint8_t v_isSharedCheck_1900_; 
v_a_1875_ = lean_ctor_get(v___x_1874_, 0);
v_isSharedCheck_1900_ = !lean_is_exclusive(v___x_1874_);
if (v_isSharedCheck_1900_ == 0)
{
v___x_1877_ = v___x_1874_;
v_isShared_1878_ = v_isSharedCheck_1900_;
goto v_resetjp_1876_;
}
else
{
lean_inc(v_a_1875_);
lean_dec(v___x_1874_);
v___x_1877_ = lean_box(0);
v_isShared_1878_ = v_isSharedCheck_1900_;
goto v_resetjp_1876_;
}
v_resetjp_1876_:
{
lean_object* v_options_1884_; lean_object* v_toOptions_1885_; lean_object* v_maxGoals_1886_; lean_object* v_numGoals_1887_; lean_object* v___x_1888_; uint8_t v___x_1889_; 
v_options_1884_ = lean_ctor_get(v_a_1870_, 2);
v_toOptions_1885_ = lean_ctor_get(v_options_1884_, 0);
v_maxGoals_1886_ = lean_ctor_get(v_toOptions_1885_, 2);
v_numGoals_1887_ = lean_ctor_get(v_a_1875_, 2);
lean_inc(v_numGoals_1887_);
lean_dec(v_a_1875_);
v___x_1888_ = lean_unsigned_to_nat(0u);
v___x_1889_ = lean_nat_dec_eq(v_maxGoals_1886_, v___x_1888_);
if (v___x_1889_ == 0)
{
uint8_t v___x_1890_; 
v___x_1890_ = lean_nat_dec_le(v_maxGoals_1886_, v_numGoals_1887_);
lean_dec(v_numGoals_1887_);
if (v___x_1890_ == 0)
{
goto v___jp_1879_;
}
else
{
lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; 
lean_del_object(v___x_1877_);
v___x_1891_ = lean_obj_once(&lp_aesop_Aesop_checkGoalLimit___redArg___closed__1, &lp_aesop_Aesop_checkGoalLimit___redArg___closed__1_once, _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__1);
lean_inc(v_maxGoals_1886_);
v___x_1892_ = l_Nat_reprFast(v_maxGoals_1886_);
v___x_1893_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1893_, 0, v___x_1892_);
v___x_1894_ = l_Lean_MessageData_ofFormat(v___x_1893_);
v___x_1895_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1895_, 0, v___x_1891_);
lean_ctor_set(v___x_1895_, 1, v___x_1894_);
v___x_1896_ = lean_obj_once(&lp_aesop_Aesop_checkGoalLimit___redArg___closed__3, &lp_aesop_Aesop_checkGoalLimit___redArg___closed__3_once, _init_lp_aesop_Aesop_checkGoalLimit___redArg___closed__3);
v___x_1897_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1897_, 0, v___x_1895_);
lean_ctor_set(v___x_1897_, 1, v___x_1896_);
v___x_1898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1898_, 0, v___x_1897_);
v___x_1899_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1899_, 0, v___x_1898_);
return v___x_1899_;
}
}
else
{
lean_dec(v_numGoals_1887_);
goto v___jp_1879_;
}
v___jp_1879_:
{
lean_object* v___x_1880_; lean_object* v___x_1882_; 
v___x_1880_ = lean_box(0);
if (v_isShared_1878_ == 0)
{
lean_ctor_set(v___x_1877_, 0, v___x_1880_);
v___x_1882_ = v___x_1877_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_1883_; 
v_reuseFailAlloc_1883_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1883_, 0, v___x_1880_);
v___x_1882_ = v_reuseFailAlloc_1883_;
goto v_reusejp_1881_;
}
v_reusejp_1881_:
{
return v___x_1882_;
}
}
}
}
else
{
lean_object* v_a_1901_; lean_object* v___x_1903_; uint8_t v_isShared_1904_; uint8_t v_isSharedCheck_1908_; 
v_a_1901_ = lean_ctor_get(v___x_1874_, 0);
v_isSharedCheck_1908_ = !lean_is_exclusive(v___x_1874_);
if (v_isSharedCheck_1908_ == 0)
{
v___x_1903_ = v___x_1874_;
v_isShared_1904_ = v_isSharedCheck_1908_;
goto v_resetjp_1902_;
}
else
{
lean_inc(v_a_1901_);
lean_dec(v___x_1874_);
v___x_1903_ = lean_box(0);
v_isShared_1904_ = v_isSharedCheck_1908_;
goto v_resetjp_1902_;
}
v_resetjp_1902_:
{
lean_object* v___x_1906_; 
if (v_isShared_1904_ == 0)
{
v___x_1906_ = v___x_1903_;
goto v_reusejp_1905_;
}
else
{
lean_object* v_reuseFailAlloc_1907_; 
v_reuseFailAlloc_1907_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1907_, 0, v_a_1901_);
v___x_1906_ = v_reuseFailAlloc_1907_;
goto v_reusejp_1905_;
}
v_reusejp_1905_:
{
return v___x_1906_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___redArg___boxed(lean_object* v_a_1909_, lean_object* v_a_1910_, lean_object* v_a_1911_, lean_object* v_a_1912_){
_start:
{
lean_object* v_res_1913_; 
v_res_1913_ = lp_aesop_Aesop_checkGoalLimit___redArg(v_a_1909_, v_a_1910_, v_a_1911_);
lean_dec(v_a_1911_);
lean_dec(v_a_1910_);
lean_dec_ref(v_a_1909_);
return v_res_1913_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit(lean_object* v_Q_1914_, lean_object* v_inst_1915_, lean_object* v_a_1916_, lean_object* v_a_1917_, lean_object* v_a_1918_, lean_object* v_a_1919_, lean_object* v_a_1920_, lean_object* v_a_1921_, lean_object* v_a_1922_, lean_object* v_a_1923_){
_start:
{
lean_object* v___x_1925_; 
v___x_1925_ = lp_aesop_Aesop_checkGoalLimit___redArg(v_a_1916_, v_a_1917_, v_a_1918_);
return v___x_1925_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkGoalLimit___boxed(lean_object* v_Q_1926_, lean_object* v_inst_1927_, lean_object* v_a_1928_, lean_object* v_a_1929_, lean_object* v_a_1930_, lean_object* v_a_1931_, lean_object* v_a_1932_, lean_object* v_a_1933_, lean_object* v_a_1934_, lean_object* v_a_1935_, lean_object* v_a_1936_){
_start:
{
lean_object* v_res_1937_; 
v_res_1937_ = lp_aesop_Aesop_checkGoalLimit(v_Q_1926_, v_inst_1927_, v_a_1928_, v_a_1929_, v_a_1930_, v_a_1931_, v_a_1932_, v_a_1933_, v_a_1934_, v_a_1935_);
lean_dec(v_a_1935_);
lean_dec_ref(v_a_1934_);
lean_dec(v_a_1933_);
lean_dec_ref(v_a_1932_);
lean_dec(v_a_1931_);
lean_dec(v_a_1930_);
lean_dec(v_a_1929_);
lean_dec_ref(v_a_1928_);
lean_dec_ref(v_inst_1927_);
return v_res_1937_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__1(void){
_start:
{
lean_object* v___x_1939_; lean_object* v___x_1940_; 
v___x_1939_ = ((lean_object*)(lp_aesop_Aesop_checkRappLimit___redArg___closed__0));
v___x_1940_ = l_Lean_stringToMessageData(v___x_1939_);
return v___x_1940_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__3(void){
_start:
{
lean_object* v___x_1942_; lean_object* v___x_1943_; 
v___x_1942_ = ((lean_object*)(lp_aesop_Aesop_checkRappLimit___redArg___closed__2));
v___x_1943_ = l_Lean_stringToMessageData(v___x_1942_);
return v___x_1943_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___redArg(lean_object* v_a_1944_, lean_object* v_a_1945_, lean_object* v_a_1946_){
_start:
{
lean_object* v___x_1948_; 
v___x_1948_ = lp_aesop_Aesop_getTree___redArg(v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1948_) == 0)
{
lean_object* v_a_1949_; lean_object* v___x_1951_; uint8_t v_isShared_1952_; uint8_t v_isSharedCheck_1974_; 
v_a_1949_ = lean_ctor_get(v___x_1948_, 0);
v_isSharedCheck_1974_ = !lean_is_exclusive(v___x_1948_);
if (v_isSharedCheck_1974_ == 0)
{
v___x_1951_ = v___x_1948_;
v_isShared_1952_ = v_isSharedCheck_1974_;
goto v_resetjp_1950_;
}
else
{
lean_inc(v_a_1949_);
lean_dec(v___x_1948_);
v___x_1951_ = lean_box(0);
v_isShared_1952_ = v_isSharedCheck_1974_;
goto v_resetjp_1950_;
}
v_resetjp_1950_:
{
lean_object* v_options_1958_; lean_object* v_toOptions_1959_; lean_object* v_maxRuleApplications_1960_; lean_object* v_numRapps_1961_; lean_object* v___x_1962_; uint8_t v___x_1963_; 
v_options_1958_ = lean_ctor_get(v_a_1944_, 2);
v_toOptions_1959_ = lean_ctor_get(v_options_1958_, 0);
v_maxRuleApplications_1960_ = lean_ctor_get(v_toOptions_1959_, 1);
v_numRapps_1961_ = lean_ctor_get(v_a_1949_, 3);
lean_inc(v_numRapps_1961_);
lean_dec(v_a_1949_);
v___x_1962_ = lean_unsigned_to_nat(0u);
v___x_1963_ = lean_nat_dec_eq(v_maxRuleApplications_1960_, v___x_1962_);
if (v___x_1963_ == 0)
{
uint8_t v___x_1964_; 
v___x_1964_ = lean_nat_dec_le(v_maxRuleApplications_1960_, v_numRapps_1961_);
lean_dec(v_numRapps_1961_);
if (v___x_1964_ == 0)
{
goto v___jp_1953_;
}
else
{
lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; 
lean_del_object(v___x_1951_);
v___x_1965_ = lean_obj_once(&lp_aesop_Aesop_checkRappLimit___redArg___closed__1, &lp_aesop_Aesop_checkRappLimit___redArg___closed__1_once, _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__1);
lean_inc(v_maxRuleApplications_1960_);
v___x_1966_ = l_Nat_reprFast(v_maxRuleApplications_1960_);
v___x_1967_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1967_, 0, v___x_1966_);
v___x_1968_ = l_Lean_MessageData_ofFormat(v___x_1967_);
v___x_1969_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1969_, 0, v___x_1965_);
lean_ctor_set(v___x_1969_, 1, v___x_1968_);
v___x_1970_ = lean_obj_once(&lp_aesop_Aesop_checkRappLimit___redArg___closed__3, &lp_aesop_Aesop_checkRappLimit___redArg___closed__3_once, _init_lp_aesop_Aesop_checkRappLimit___redArg___closed__3);
v___x_1971_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1971_, 0, v___x_1969_);
lean_ctor_set(v___x_1971_, 1, v___x_1970_);
v___x_1972_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1972_, 0, v___x_1971_);
v___x_1973_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1973_, 0, v___x_1972_);
return v___x_1973_;
}
}
else
{
lean_dec(v_numRapps_1961_);
goto v___jp_1953_;
}
v___jp_1953_:
{
lean_object* v___x_1954_; lean_object* v___x_1956_; 
v___x_1954_ = lean_box(0);
if (v_isShared_1952_ == 0)
{
lean_ctor_set(v___x_1951_, 0, v___x_1954_);
v___x_1956_ = v___x_1951_;
goto v_reusejp_1955_;
}
else
{
lean_object* v_reuseFailAlloc_1957_; 
v_reuseFailAlloc_1957_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1957_, 0, v___x_1954_);
v___x_1956_ = v_reuseFailAlloc_1957_;
goto v_reusejp_1955_;
}
v_reusejp_1955_:
{
return v___x_1956_;
}
}
}
}
else
{
lean_object* v_a_1975_; lean_object* v___x_1977_; uint8_t v_isShared_1978_; uint8_t v_isSharedCheck_1982_; 
v_a_1975_ = lean_ctor_get(v___x_1948_, 0);
v_isSharedCheck_1982_ = !lean_is_exclusive(v___x_1948_);
if (v_isSharedCheck_1982_ == 0)
{
v___x_1977_ = v___x_1948_;
v_isShared_1978_ = v_isSharedCheck_1982_;
goto v_resetjp_1976_;
}
else
{
lean_inc(v_a_1975_);
lean_dec(v___x_1948_);
v___x_1977_ = lean_box(0);
v_isShared_1978_ = v_isSharedCheck_1982_;
goto v_resetjp_1976_;
}
v_resetjp_1976_:
{
lean_object* v___x_1980_; 
if (v_isShared_1978_ == 0)
{
v___x_1980_ = v___x_1977_;
goto v_reusejp_1979_;
}
else
{
lean_object* v_reuseFailAlloc_1981_; 
v_reuseFailAlloc_1981_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1981_, 0, v_a_1975_);
v___x_1980_ = v_reuseFailAlloc_1981_;
goto v_reusejp_1979_;
}
v_reusejp_1979_:
{
return v___x_1980_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___redArg___boxed(lean_object* v_a_1983_, lean_object* v_a_1984_, lean_object* v_a_1985_, lean_object* v_a_1986_){
_start:
{
lean_object* v_res_1987_; 
v_res_1987_ = lp_aesop_Aesop_checkRappLimit___redArg(v_a_1983_, v_a_1984_, v_a_1985_);
lean_dec(v_a_1985_);
lean_dec(v_a_1984_);
lean_dec_ref(v_a_1983_);
return v_res_1987_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit(lean_object* v_Q_1988_, lean_object* v_inst_1989_, lean_object* v_a_1990_, lean_object* v_a_1991_, lean_object* v_a_1992_, lean_object* v_a_1993_, lean_object* v_a_1994_, lean_object* v_a_1995_, lean_object* v_a_1996_, lean_object* v_a_1997_){
_start:
{
lean_object* v___x_1999_; 
v___x_1999_ = lp_aesop_Aesop_checkRappLimit___redArg(v_a_1990_, v_a_1991_, v_a_1992_);
return v___x_1999_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRappLimit___boxed(lean_object* v_Q_2000_, lean_object* v_inst_2001_, lean_object* v_a_2002_, lean_object* v_a_2003_, lean_object* v_a_2004_, lean_object* v_a_2005_, lean_object* v_a_2006_, lean_object* v_a_2007_, lean_object* v_a_2008_, lean_object* v_a_2009_, lean_object* v_a_2010_){
_start:
{
lean_object* v_res_2011_; 
v_res_2011_ = lp_aesop_Aesop_checkRappLimit(v_Q_2000_, v_inst_2001_, v_a_2002_, v_a_2003_, v_a_2004_, v_a_2005_, v_a_2006_, v_a_2007_, v_a_2008_, v_a_2009_);
lean_dec(v_a_2009_);
lean_dec_ref(v_a_2008_);
lean_dec(v_a_2007_);
lean_dec_ref(v_a_2006_);
lean_dec(v_a_2005_);
lean_dec(v_a_2004_);
lean_dec(v_a_2003_);
lean_dec_ref(v_a_2002_);
lean_dec_ref(v_inst_2001_);
return v_res_2011_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1(void){
_start:
{
lean_object* v___x_2013_; lean_object* v___x_2014_; 
v___x_2013_ = ((lean_object*)(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__0));
v___x_2014_ = l_Lean_stringToMessageData(v___x_2013_);
return v___x_2014_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3(void){
_start:
{
lean_object* v___x_2016_; lean_object* v___x_2017_; 
v___x_2016_ = ((lean_object*)(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__2));
v___x_2017_ = l_Lean_stringToMessageData(v___x_2016_);
return v___x_2017_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5(void){
_start:
{
lean_object* v___x_2019_; lean_object* v___x_2020_; 
v___x_2019_ = ((lean_object*)(lp_aesop_Aesop_checkRootUnprovable___redArg___closed__4));
v___x_2020_ = l_Lean_stringToMessageData(v___x_2019_);
return v___x_2020_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg(lean_object* v_a_2021_, lean_object* v_a_2022_, lean_object* v_a_2023_){
_start:
{
lean_object* v___x_2025_; 
v___x_2025_ = lp_aesop_Aesop_getTree___redArg(v_a_2022_, v_a_2023_);
if (lean_obj_tag(v___x_2025_) == 0)
{
lean_object* v_a_2026_; lean_object* v___x_2028_; uint8_t v_isShared_2029_; uint8_t v_isSharedCheck_2074_; 
v_a_2026_ = lean_ctor_get(v___x_2025_, 0);
v_isSharedCheck_2074_ = !lean_is_exclusive(v___x_2025_);
if (v_isSharedCheck_2074_ == 0)
{
v___x_2028_ = v___x_2025_;
v_isShared_2029_ = v_isSharedCheck_2074_;
goto v_resetjp_2027_;
}
else
{
lean_inc(v_a_2026_);
lean_dec(v___x_2025_);
v___x_2028_ = lean_box(0);
v_isShared_2029_ = v_isSharedCheck_2074_;
goto v_resetjp_2027_;
}
v_resetjp_2027_:
{
lean_object* v___x_2030_; lean_object* v_root_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v_elimMVarCluster_2034_; lean_object* v___x_2035_; uint8_t v_state_2036_; uint8_t v___x_2037_; 
v___x_2030_ = lean_st_ref_get(v_a_2022_);
lean_dec(v___x_2030_);
v_root_2031_ = lean_ctor_get(v_a_2026_, 0);
lean_inc(v_root_2031_);
lean_dec(v_a_2026_);
v___x_2032_ = lean_st_ref_get(v_root_2031_);
lean_dec(v_root_2031_);
v___x_2033_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_2034_ = lean_ctor_get(v___x_2033_, 5);
lean_inc_ref(v_elimMVarCluster_2034_);
v___x_2035_ = lean_apply_1(v_elimMVarCluster_2034_, v___x_2032_);
v_state_2036_ = lean_ctor_get_uint8(v___x_2035_, sizeof(void*)*2 + 1);
lean_dec_ref(v___x_2035_);
v___x_2037_ = lp_aesop_Aesop_NodeState_isUnprovable(v_state_2036_);
if (v___x_2037_ == 0)
{
lean_object* v___x_2038_; lean_object* v___x_2040_; 
v___x_2038_ = lean_box(0);
if (v_isShared_2029_ == 0)
{
lean_ctor_set(v___x_2028_, 0, v___x_2038_);
v___x_2040_ = v___x_2028_;
goto v_reusejp_2039_;
}
else
{
lean_object* v_reuseFailAlloc_2041_; 
v_reuseFailAlloc_2041_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2041_, 0, v___x_2038_);
v___x_2040_ = v_reuseFailAlloc_2041_;
goto v_reusejp_2039_;
}
v_reusejp_2039_:
{
return v___x_2040_;
}
}
else
{
lean_object* v___x_2042_; 
lean_del_object(v___x_2028_);
v___x_2042_ = lp_aesop_Aesop_wasMaxRuleApplicationDepthReached___redArg(v_a_2022_);
if (lean_obj_tag(v___x_2042_) == 0)
{
lean_object* v_a_2043_; lean_object* v___x_2045_; uint8_t v_isShared_2046_; uint8_t v_isSharedCheck_2065_; 
v_a_2043_ = lean_ctor_get(v___x_2042_, 0);
v_isSharedCheck_2065_ = !lean_is_exclusive(v___x_2042_);
if (v_isSharedCheck_2065_ == 0)
{
v___x_2045_ = v___x_2042_;
v_isShared_2046_ = v_isSharedCheck_2065_;
goto v_resetjp_2044_;
}
else
{
lean_inc(v_a_2043_);
lean_dec(v___x_2042_);
v___x_2045_ = lean_box(0);
v_isShared_2046_ = v_isSharedCheck_2065_;
goto v_resetjp_2044_;
}
v_resetjp_2044_:
{
lean_object* v_msg_2048_; uint8_t v___x_2053_; 
v___x_2053_ = lean_unbox(v_a_2043_);
lean_dec(v_a_2043_);
if (v___x_2053_ == 0)
{
lean_object* v___x_2054_; 
v___x_2054_ = lean_obj_once(&lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1, &lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1_once, _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__1);
v_msg_2048_ = v___x_2054_;
goto v___jp_2047_;
}
else
{
lean_object* v_options_2055_; lean_object* v_toOptions_2056_; lean_object* v_maxRuleApplicationDepth_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; 
v_options_2055_ = lean_ctor_get(v_a_2021_, 2);
v_toOptions_2056_ = lean_ctor_get(v_options_2055_, 0);
v_maxRuleApplicationDepth_2057_ = lean_ctor_get(v_toOptions_2056_, 0);
v___x_2058_ = lean_obj_once(&lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3, &lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3_once, _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__3);
lean_inc(v_maxRuleApplicationDepth_2057_);
v___x_2059_ = l_Nat_reprFast(v_maxRuleApplicationDepth_2057_);
v___x_2060_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2060_, 0, v___x_2059_);
v___x_2061_ = l_Lean_MessageData_ofFormat(v___x_2060_);
v___x_2062_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2062_, 0, v___x_2058_);
lean_ctor_set(v___x_2062_, 1, v___x_2061_);
v___x_2063_ = lean_obj_once(&lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5, &lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5_once, _init_lp_aesop_Aesop_checkRootUnprovable___redArg___closed__5);
v___x_2064_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2064_, 0, v___x_2062_);
lean_ctor_set(v___x_2064_, 1, v___x_2063_);
v_msg_2048_ = v___x_2064_;
goto v___jp_2047_;
}
v___jp_2047_:
{
lean_object* v___x_2049_; lean_object* v___x_2051_; 
v___x_2049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2049_, 0, v_msg_2048_);
if (v_isShared_2046_ == 0)
{
lean_ctor_set(v___x_2045_, 0, v___x_2049_);
v___x_2051_ = v___x_2045_;
goto v_reusejp_2050_;
}
else
{
lean_object* v_reuseFailAlloc_2052_; 
v_reuseFailAlloc_2052_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2052_, 0, v___x_2049_);
v___x_2051_ = v_reuseFailAlloc_2052_;
goto v_reusejp_2050_;
}
v_reusejp_2050_:
{
return v___x_2051_;
}
}
}
}
else
{
lean_object* v_a_2066_; lean_object* v___x_2068_; uint8_t v_isShared_2069_; uint8_t v_isSharedCheck_2073_; 
v_a_2066_ = lean_ctor_get(v___x_2042_, 0);
v_isSharedCheck_2073_ = !lean_is_exclusive(v___x_2042_);
if (v_isSharedCheck_2073_ == 0)
{
v___x_2068_ = v___x_2042_;
v_isShared_2069_ = v_isSharedCheck_2073_;
goto v_resetjp_2067_;
}
else
{
lean_inc(v_a_2066_);
lean_dec(v___x_2042_);
v___x_2068_ = lean_box(0);
v_isShared_2069_ = v_isSharedCheck_2073_;
goto v_resetjp_2067_;
}
v_resetjp_2067_:
{
lean_object* v___x_2071_; 
if (v_isShared_2069_ == 0)
{
v___x_2071_ = v___x_2068_;
goto v_reusejp_2070_;
}
else
{
lean_object* v_reuseFailAlloc_2072_; 
v_reuseFailAlloc_2072_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2072_, 0, v_a_2066_);
v___x_2071_ = v_reuseFailAlloc_2072_;
goto v_reusejp_2070_;
}
v_reusejp_2070_:
{
return v___x_2071_;
}
}
}
}
}
}
else
{
lean_object* v_a_2075_; lean_object* v___x_2077_; uint8_t v_isShared_2078_; uint8_t v_isSharedCheck_2082_; 
v_a_2075_ = lean_ctor_get(v___x_2025_, 0);
v_isSharedCheck_2082_ = !lean_is_exclusive(v___x_2025_);
if (v_isSharedCheck_2082_ == 0)
{
v___x_2077_ = v___x_2025_;
v_isShared_2078_ = v_isSharedCheck_2082_;
goto v_resetjp_2076_;
}
else
{
lean_inc(v_a_2075_);
lean_dec(v___x_2025_);
v___x_2077_ = lean_box(0);
v_isShared_2078_ = v_isSharedCheck_2082_;
goto v_resetjp_2076_;
}
v_resetjp_2076_:
{
lean_object* v___x_2080_; 
if (v_isShared_2078_ == 0)
{
v___x_2080_ = v___x_2077_;
goto v_reusejp_2079_;
}
else
{
lean_object* v_reuseFailAlloc_2081_; 
v_reuseFailAlloc_2081_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2081_, 0, v_a_2075_);
v___x_2080_ = v_reuseFailAlloc_2081_;
goto v_reusejp_2079_;
}
v_reusejp_2079_:
{
return v___x_2080_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___redArg___boxed(lean_object* v_a_2083_, lean_object* v_a_2084_, lean_object* v_a_2085_, lean_object* v_a_2086_){
_start:
{
lean_object* v_res_2087_; 
v_res_2087_ = lp_aesop_Aesop_checkRootUnprovable___redArg(v_a_2083_, v_a_2084_, v_a_2085_);
lean_dec(v_a_2085_);
lean_dec(v_a_2084_);
lean_dec_ref(v_a_2083_);
return v_res_2087_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable(lean_object* v_Q_2088_, lean_object* v_inst_2089_, lean_object* v_a_2090_, lean_object* v_a_2091_, lean_object* v_a_2092_, lean_object* v_a_2093_, lean_object* v_a_2094_, lean_object* v_a_2095_, lean_object* v_a_2096_, lean_object* v_a_2097_){
_start:
{
lean_object* v___x_2099_; 
v___x_2099_ = lp_aesop_Aesop_checkRootUnprovable___redArg(v_a_2090_, v_a_2091_, v_a_2092_);
return v___x_2099_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkRootUnprovable___boxed(lean_object* v_Q_2100_, lean_object* v_inst_2101_, lean_object* v_a_2102_, lean_object* v_a_2103_, lean_object* v_a_2104_, lean_object* v_a_2105_, lean_object* v_a_2106_, lean_object* v_a_2107_, lean_object* v_a_2108_, lean_object* v_a_2109_, lean_object* v_a_2110_){
_start:
{
lean_object* v_res_2111_; 
v_res_2111_ = lp_aesop_Aesop_checkRootUnprovable(v_Q_2100_, v_inst_2101_, v_a_2102_, v_a_2103_, v_a_2104_, v_a_2105_, v_a_2106_, v_a_2107_, v_a_2108_, v_a_2109_);
lean_dec(v_a_2109_);
lean_dec_ref(v_a_2108_);
lean_dec(v_a_2107_);
lean_dec_ref(v_a_2106_);
lean_dec(v_a_2105_);
lean_dec(v_a_2104_);
lean_dec(v_a_2103_);
lean_dec_ref(v_a_2102_);
lean_dec_ref(v_inst_2101_);
return v_res_2111_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___redArg(lean_object* v_inst_2112_, lean_object* v_a_2113_, lean_object* v_a_2114_, lean_object* v_a_2115_, lean_object* v_a_2116_, lean_object* v_a_2117_, lean_object* v_a_2118_, lean_object* v_a_2119_, lean_object* v_a_2120_){
_start:
{
lean_object* v___x_2122_; lean_object* v_getMCtx_2123_; lean_object* v_modifyMCtx_2124_; lean_object* v___f_2125_; lean_object* v___x_2126_; lean_object* v___f_2127_; lean_object* v___x_2128_; lean_object* v___f_2129_; lean_object* v___x_2130_; lean_object* v___f_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___f_2134_; lean_object* v___f_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v_iteration_2139_; lean_object* v_ruleSet_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; 
v___x_2122_ = l_Lean_Meta_instMonadMCtxMetaM;
v_getMCtx_2123_ = lean_ctor_get(v___x_2122_, 0);
v_modifyMCtx_2124_ = lean_ctor_get(v___x_2122_, 1);
v___f_2125_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___x_2126_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
lean_inc(v_modifyMCtx_2124_);
v___f_2127_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2127_, 0, v_modifyMCtx_2124_);
lean_closure_set(v___f_2127_, 1, v___x_2126_);
lean_inc(v_getMCtx_2123_);
v___x_2128_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_2128_, 0, lean_box(0));
lean_closure_set(v___x_2128_, 1, lean_box(0));
lean_closure_set(v___x_2128_, 2, lean_box(0));
lean_closure_set(v___x_2128_, 3, lean_box(0));
lean_closure_set(v___x_2128_, 4, v_getMCtx_2123_);
v___f_2129_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2129_, 0, v___f_2127_);
lean_closure_set(v___f_2129_, 1, v___x_2126_);
v___x_2130_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_2130_, 0, lean_box(0));
lean_closure_set(v___x_2130_, 1, lean_box(0));
lean_closure_set(v___x_2130_, 2, lean_box(0));
lean_closure_set(v___x_2130_, 3, lean_box(0));
lean_closure_set(v___x_2130_, 4, v___x_2128_);
v___f_2131_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2131_, 0, v___f_2129_);
lean_closure_set(v___f_2131_, 1, v___f_2125_);
v___x_2132_ = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(v___x_2132_, 0, lean_box(0));
lean_closure_set(v___x_2132_, 1, v___x_2130_);
v___x_2133_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_2112_);
v___f_2134_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__17));
v___f_2135_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2135_, 0, v___f_2131_);
lean_closure_set(v___f_2135_, 1, v___f_2134_);
v___x_2136_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(v___x_2136_, 0, lean_box(0));
lean_closure_set(v___x_2136_, 1, v___x_2132_);
v___x_2137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2137_, 0, v___x_2136_);
lean_ctor_set(v___x_2137_, 1, v___f_2135_);
v___x_2138_ = lean_st_ref_get(v_a_2114_);
v_iteration_2139_ = lean_ctor_get(v___x_2138_, 0);
lean_inc(v_iteration_2139_);
lean_dec(v___x_2138_);
v_ruleSet_2140_ = lean_ctor_get(v_a_2113_, 0);
lean_inc_ref(v_ruleSet_2140_);
v___x_2141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2141_, 0, v_iteration_2139_);
lean_ctor_set(v___x_2141_, 1, v_ruleSet_2140_);
v___x_2142_ = lp_aesop_Aesop_getRootMVarId(v___x_2141_, v_a_2115_, v_a_2116_, v_a_2117_, v_a_2118_, v_a_2119_, v_a_2120_);
lean_dec_ref_known(v___x_2141_, 2);
if (lean_obj_tag(v___x_2142_) == 0)
{
lean_object* v_a_2143_; lean_object* v___x_361__overap_2144_; lean_object* v___x_2145_; 
v_a_2143_ = lean_ctor_get(v___x_2142_, 0);
lean_inc(v_a_2143_);
lean_dec_ref_known(v___x_2142_, 1);
v___x_361__overap_2144_ = l_Lean_getExprMVarAssignment_x3f___redArg(v___x_2133_, v___x_2137_, v_a_2143_);
lean_inc(v_a_2120_);
lean_inc_ref(v_a_2119_);
lean_inc(v_a_2118_);
lean_inc_ref(v_a_2117_);
lean_inc(v_a_2116_);
lean_inc(v_a_2115_);
lean_inc(v_a_2114_);
lean_inc_ref(v_a_2113_);
v___x_2145_ = lean_apply_9(v___x_361__overap_2144_, v_a_2113_, v_a_2114_, v_a_2115_, v_a_2116_, v_a_2117_, v_a_2118_, v_a_2119_, v_a_2120_, lean_box(0));
return v___x_2145_;
}
else
{
lean_object* v_a_2146_; lean_object* v___x_2148_; uint8_t v_isShared_2149_; uint8_t v_isSharedCheck_2153_; 
lean_dec_ref_known(v___x_2137_, 2);
lean_dec_ref(v___x_2133_);
v_a_2146_ = lean_ctor_get(v___x_2142_, 0);
v_isSharedCheck_2153_ = !lean_is_exclusive(v___x_2142_);
if (v_isSharedCheck_2153_ == 0)
{
v___x_2148_ = v___x_2142_;
v_isShared_2149_ = v_isSharedCheck_2153_;
goto v_resetjp_2147_;
}
else
{
lean_inc(v_a_2146_);
lean_dec(v___x_2142_);
v___x_2148_ = lean_box(0);
v_isShared_2149_ = v_isSharedCheck_2153_;
goto v_resetjp_2147_;
}
v_resetjp_2147_:
{
lean_object* v___x_2151_; 
if (v_isShared_2149_ == 0)
{
v___x_2151_ = v___x_2148_;
goto v_reusejp_2150_;
}
else
{
lean_object* v_reuseFailAlloc_2152_; 
v_reuseFailAlloc_2152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2152_, 0, v_a_2146_);
v___x_2151_ = v_reuseFailAlloc_2152_;
goto v_reusejp_2150_;
}
v_reusejp_2150_:
{
return v___x_2151_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___redArg___boxed(lean_object* v_inst_2154_, lean_object* v_a_2155_, lean_object* v_a_2156_, lean_object* v_a_2157_, lean_object* v_a_2158_, lean_object* v_a_2159_, lean_object* v_a_2160_, lean_object* v_a_2161_, lean_object* v_a_2162_, lean_object* v_a_2163_){
_start:
{
lean_object* v_res_2164_; 
v_res_2164_ = lp_aesop_Aesop_getProof_x3f___redArg(v_inst_2154_, v_a_2155_, v_a_2156_, v_a_2157_, v_a_2158_, v_a_2159_, v_a_2160_, v_a_2161_, v_a_2162_);
lean_dec(v_a_2162_);
lean_dec_ref(v_a_2161_);
lean_dec(v_a_2160_);
lean_dec_ref(v_a_2159_);
lean_dec(v_a_2158_);
lean_dec(v_a_2157_);
lean_dec(v_a_2156_);
lean_dec_ref(v_a_2155_);
lean_dec_ref(v_inst_2154_);
return v_res_2164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f(lean_object* v_Q_2165_, lean_object* v_inst_2166_, lean_object* v_a_2167_, lean_object* v_a_2168_, lean_object* v_a_2169_, lean_object* v_a_2170_, lean_object* v_a_2171_, lean_object* v_a_2172_, lean_object* v_a_2173_, lean_object* v_a_2174_){
_start:
{
lean_object* v___x_2176_; 
v___x_2176_ = lp_aesop_Aesop_getProof_x3f___redArg(v_inst_2166_, v_a_2167_, v_a_2168_, v_a_2169_, v_a_2170_, v_a_2171_, v_a_2172_, v_a_2173_, v_a_2174_);
return v___x_2176_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getProof_x3f___boxed(lean_object* v_Q_2177_, lean_object* v_inst_2178_, lean_object* v_a_2179_, lean_object* v_a_2180_, lean_object* v_a_2181_, lean_object* v_a_2182_, lean_object* v_a_2183_, lean_object* v_a_2184_, lean_object* v_a_2185_, lean_object* v_a_2186_, lean_object* v_a_2187_){
_start:
{
lean_object* v_res_2188_; 
v_res_2188_ = lp_aesop_Aesop_getProof_x3f(v_Q_2177_, v_inst_2178_, v_a_2179_, v_a_2180_, v_a_2181_, v_a_2182_, v_a_2183_, v_a_2184_, v_a_2185_, v_a_2186_);
lean_dec(v_a_2186_);
lean_dec_ref(v_a_2185_);
lean_dec(v_a_2184_);
lean_dec_ref(v_a_2183_);
lean_dec(v_a_2182_);
lean_dec(v_a_2181_);
lean_dec(v_a_2180_);
lean_dec_ref(v_a_2179_);
lean_dec_ref(v_inst_2178_);
return v_res_2188_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__0(lean_object* v___y_2189_, lean_object* v___y_2190_, lean_object* v___y_2191_, lean_object* v___y_2192_, lean_object* v___y_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_){
_start:
{
lean_object* v___x_2198_; lean_object* v_iteration_2199_; lean_object* v_ruleSet_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; 
v___x_2198_ = lean_st_ref_get(v___y_2190_);
v_iteration_2199_ = lean_ctor_get(v___x_2198_, 0);
lean_inc(v_iteration_2199_);
lean_dec(v___x_2198_);
v_ruleSet_2200_ = lean_ctor_get(v___y_2189_, 0);
lean_inc_ref(v_ruleSet_2200_);
v___x_2201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2201_, 0, v_iteration_2199_);
lean_ctor_set(v___x_2201_, 1, v_ruleSet_2200_);
v___x_2202_ = lp_aesop_Aesop_extractProof(v___x_2201_, v___y_2191_, v___y_2192_, v___y_2193_, v___y_2194_, v___y_2195_, v___y_2196_);
lean_dec_ref_known(v___x_2201_, 2);
return v___x_2202_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__0___boxed(lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_, lean_object* v___y_2206_, lean_object* v___y_2207_, lean_object* v___y_2208_, lean_object* v___y_2209_, lean_object* v___y_2210_, lean_object* v___y_2211_){
_start:
{
lean_object* v_res_2212_; 
v_res_2212_ = lp_aesop_Aesop_finalizeProof___redArg___lam__0(v___y_2203_, v___y_2204_, v___y_2205_, v___y_2206_, v___y_2207_, v___y_2208_, v___y_2209_, v___y_2210_);
lean_dec(v___y_2210_);
lean_dec_ref(v___y_2209_);
lean_dec(v___y_2208_);
lean_dec_ref(v___y_2207_);
lean_dec(v___y_2206_);
lean_dec(v___y_2205_);
lean_dec(v___y_2204_);
lean_dec_ref(v___y_2203_);
return v_res_2212_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__1(lean_object* v_x_2213_){
_start:
{
lean_inc(v_x_2213_);
return v_x_2213_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__1___boxed(lean_object* v_x_2214_){
_start:
{
lean_object* v_res_2215_; 
v_res_2215_ = lp_aesop_Aesop_finalizeProof___redArg___lam__1(v_x_2214_);
lean_dec(v_x_2214_);
return v_res_2215_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1(void){
_start:
{
lean_object* v___x_2217_; lean_object* v___x_2218_; 
v___x_2217_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__0));
v___x_2218_ = l_Lean_stringToMessageData(v___x_2217_);
return v___x_2218_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2(lean_object* v___x_2219_, lean_object* v___x_2220_, lean_object* v___f_2221_, lean_object* v___f_2222_, lean_object* v___x_2223_, lean_object* v_val_2224_, lean_object* v___x_2225_, lean_object* v___x_2226_, lean_object* v___f_2227_, uint8_t v_____do__lift_2228_, lean_object* v___y_2229_, lean_object* v___y_2230_, lean_object* v___y_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_, lean_object* v___y_2235_, lean_object* v___y_2236_){
_start:
{
if (v_____do__lift_2228_ == 0)
{
lean_object* v___x_2238_; lean_object* v___x_2239_; 
lean_dec(v___f_2227_);
lean_dec_ref(v___x_2226_);
lean_dec_ref(v___x_2225_);
lean_dec_ref(v_val_2224_);
lean_dec_ref(v___x_2223_);
lean_dec_ref(v___f_2222_);
lean_dec(v___f_2221_);
lean_dec(v___x_2220_);
lean_dec(v___x_2219_);
v___x_2238_ = lean_box(0);
v___x_2239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2239_, 0, v___x_2238_);
return v___x_2239_;
}
else
{
lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v_traceClass_2245_; lean_object* v___x_2247_; uint8_t v_isShared_2248_; uint8_t v_isSharedCheck_2256_; 
v___x_2240_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__1);
v___x_2241_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_2219_, v___x_2240_);
v___x_2242_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_2220_, v___x_2241_);
v___x_2243_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_2221_, v___x_2242_);
v___x_2244_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_2222_, v___x_2243_);
v_traceClass_2245_ = lean_ctor_get(v___x_2223_, 0);
v_isSharedCheck_2256_ = !lean_is_exclusive(v___x_2223_);
if (v_isSharedCheck_2256_ == 0)
{
lean_object* v_unused_2257_; 
v_unused_2257_ = lean_ctor_get(v___x_2223_, 1);
lean_dec(v_unused_2257_);
v___x_2247_ = v___x_2223_;
v_isShared_2248_ = v_isSharedCheck_2256_;
goto v_resetjp_2246_;
}
else
{
lean_inc(v_traceClass_2245_);
lean_dec(v___x_2223_);
v___x_2247_ = lean_box(0);
v_isShared_2248_ = v_isSharedCheck_2256_;
goto v_resetjp_2246_;
}
v_resetjp_2246_:
{
lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2252_; 
v___x_2249_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1, &lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1_once, _init_lp_aesop_Aesop_finalizeProof___redArg___lam__2___closed__1);
v___x_2250_ = l_Lean_indentExpr(v_val_2224_);
if (v_isShared_2248_ == 0)
{
lean_ctor_set_tag(v___x_2247_, 7);
lean_ctor_set(v___x_2247_, 1, v___x_2250_);
lean_ctor_set(v___x_2247_, 0, v___x_2249_);
v___x_2252_ = v___x_2247_;
goto v_reusejp_2251_;
}
else
{
lean_object* v_reuseFailAlloc_2255_; 
v_reuseFailAlloc_2255_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2255_, 0, v___x_2249_);
lean_ctor_set(v_reuseFailAlloc_2255_, 1, v___x_2250_);
v___x_2252_ = v_reuseFailAlloc_2255_;
goto v_reusejp_2251_;
}
v_reusejp_2251_:
{
lean_object* v___x_8889__overap_2253_; lean_object* v___x_2254_; 
v___x_8889__overap_2253_ = l_Lean_addTrace___redArg(v___x_2225_, v___x_2244_, v___x_2226_, v___f_2227_, v_traceClass_2245_, v___x_2252_);
lean_inc(v___y_2236_);
lean_inc_ref(v___y_2235_);
lean_inc(v___y_2234_);
lean_inc_ref(v___y_2233_);
lean_inc(v___y_2232_);
lean_inc(v___y_2231_);
lean_inc(v___y_2230_);
lean_inc_ref(v___y_2229_);
v___x_2254_ = lean_apply_9(v___x_8889__overap_2253_, v___y_2229_, v___y_2230_, v___y_2231_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_, v___y_2236_, lean_box(0));
return v___x_2254_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__2___boxed(lean_object** _args){
lean_object* v___x_2258_ = _args[0];
lean_object* v___x_2259_ = _args[1];
lean_object* v___f_2260_ = _args[2];
lean_object* v___f_2261_ = _args[3];
lean_object* v___x_2262_ = _args[4];
lean_object* v_val_2263_ = _args[5];
lean_object* v___x_2264_ = _args[6];
lean_object* v___x_2265_ = _args[7];
lean_object* v___f_2266_ = _args[8];
lean_object* v_____do__lift_2267_ = _args[9];
lean_object* v___y_2268_ = _args[10];
lean_object* v___y_2269_ = _args[11];
lean_object* v___y_2270_ = _args[12];
lean_object* v___y_2271_ = _args[13];
lean_object* v___y_2272_ = _args[14];
lean_object* v___y_2273_ = _args[15];
lean_object* v___y_2274_ = _args[16];
lean_object* v___y_2275_ = _args[17];
lean_object* v___y_2276_ = _args[18];
_start:
{
uint8_t v_____do__lift_9047__boxed_2277_; lean_object* v_res_2278_; 
v_____do__lift_9047__boxed_2277_ = lean_unbox(v_____do__lift_2267_);
v_res_2278_ = lp_aesop_Aesop_finalizeProof___redArg___lam__2(v___x_2258_, v___x_2259_, v___f_2260_, v___f_2261_, v___x_2262_, v_val_2263_, v___x_2264_, v___x_2265_, v___f_2266_, v_____do__lift_9047__boxed_2277_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_);
lean_dec(v___y_2275_);
lean_dec_ref(v___y_2274_);
lean_dec(v___y_2273_);
lean_dec_ref(v___y_2272_);
lean_dec(v___y_2271_);
lean_dec(v___y_2270_);
lean_dec(v___y_2269_);
lean_dec_ref(v___y_2268_);
return v_res_2278_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1(void){
_start:
{
lean_object* v___x_2280_; lean_object* v___x_2281_; 
v___x_2280_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__0));
v___x_2281_ = l_Lean_stringToMessageData(v___x_2280_);
return v___x_2281_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3(void){
_start:
{
lean_object* v___x_2283_; lean_object* v___x_2284_; 
v___x_2283_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__2));
v___x_2284_ = l_Lean_stringToMessageData(v___x_2283_);
return v___x_2284_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16(void){
_start:
{
lean_object* v___x_2306_; lean_object* v___x_2307_; 
v___x_2306_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__15));
v___x_2307_ = l_Lean_stringToMessageData(v___x_2306_);
return v___x_2307_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18(void){
_start:
{
lean_object* v___x_2309_; lean_object* v___x_2310_; 
v___x_2309_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__17));
v___x_2310_ = l_Lean_stringToMessageData(v___x_2309_);
return v___x_2310_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3(lean_object* v_inst_2311_, lean_object* v___x_2312_, lean_object* v___x_2313_, lean_object* v___x_2314_, lean_object* v___x_2315_, lean_object* v___f_2316_, lean_object* v___f_2317_, lean_object* v___x_2318_, lean_object* v___f_2319_, lean_object* v_toMonadOptions_2320_, lean_object* v___x_2321_, lean_object* v___f_2322_, lean_object* v___f_2323_, lean_object* v___x_2324_, lean_object* v_____r_2325_, lean_object* v___y_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_, lean_object* v___y_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_){
_start:
{
lean_object* v___x_2335_; 
v___x_2335_ = lp_aesop_Aesop_getProof_x3f___redArg(v_inst_2311_, v___y_2326_, v___y_2327_, v___y_2328_, v___y_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_);
if (lean_obj_tag(v___x_2335_) == 0)
{
lean_object* v_a_2336_; 
v_a_2336_ = lean_ctor_get(v___x_2335_, 0);
lean_inc(v_a_2336_);
lean_dec_ref_known(v___x_2335_, 1);
if (lean_obj_tag(v_a_2336_) == 1)
{
lean_object* v_val_2337_; lean_object* v___y_2339_; lean_object* v___y_2340_; lean_object* v___y_2341_; lean_object* v___y_2342_; lean_object* v___y_2343_; lean_object* v___y_2344_; lean_object* v___y_2345_; lean_object* v___y_2346_; lean_object* v___x_8917__overap_2353_; lean_object* v___x_2354_; 
v_val_2337_ = lean_ctor_get(v_a_2336_, 0);
lean_inc_n(v_val_2337_, 2);
lean_dec_ref_known(v_a_2336_, 1);
lean_inc_ref(v___x_2312_);
v___x_8917__overap_2353_ = l_Lean_instantiateMVars___redArg(v___x_2312_, v___x_2313_, v_val_2337_);
lean_inc(v___y_2333_);
lean_inc_ref(v___y_2332_);
lean_inc(v___y_2331_);
lean_inc_ref(v___y_2330_);
lean_inc(v___y_2329_);
lean_inc(v___y_2328_);
lean_inc(v___y_2327_);
lean_inc_ref(v___y_2326_);
v___x_2354_ = lean_apply_9(v___x_8917__overap_2353_, v___y_2326_, v___y_2327_, v___y_2328_, v___y_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, lean_box(0));
if (lean_obj_tag(v___x_2354_) == 0)
{
lean_object* v_a_2355_; uint8_t v___x_2356_; 
v_a_2355_ = lean_ctor_get(v___x_2354_, 0);
lean_inc(v_a_2355_);
lean_dec_ref_known(v___x_2354_, 1);
v___x_2356_ = l_Lean_Expr_hasExprMVar(v_a_2355_);
lean_dec(v_a_2355_);
if (v___x_2356_ == 0)
{
lean_dec_ref(v___x_2324_);
lean_dec_ref(v___f_2323_);
v___y_2339_ = v___y_2326_;
v___y_2340_ = v___y_2327_;
v___y_2341_ = v___y_2328_;
v___y_2342_ = v___y_2329_;
v___y_2343_ = v___y_2330_;
v___y_2344_ = v___y_2331_;
v___y_2345_ = v___y_2332_;
v___y_2346_ = v___y_2333_;
goto v___jp_2338_;
}
else
{
lean_object* v___x_2357_; lean_object* v___x_2358_; 
v___x_2357_ = lean_st_ref_get(v___y_2327_);
lean_dec(v___x_2357_);
lean_inc(v_val_2337_);
v___x_2358_ = l_Lean_Meta_getMVarsNoDelayed(v_val_2337_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_);
if (lean_obj_tag(v___x_2358_) == 0)
{
lean_object* v_a_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; size_t v_sz_2366_; size_t v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___x_8971__overap_2378_; lean_object* v___x_2379_; 
v_a_2359_ = lean_ctor_get(v___x_2358_, 0);
lean_inc(v_a_2359_);
lean_dec_ref_known(v___x_2358_, 1);
v___x_2360_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1, &lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1_once, _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__1);
lean_inc(v_val_2337_);
v___x_2361_ = l_Lean_MessageData_ofExpr(v_val_2337_);
v___x_2362_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2362_, 0, v___x_2360_);
lean_ctor_set(v___x_2362_, 1, v___x_2361_);
v___x_2363_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3, &lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3_once, _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__3);
v___x_2364_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2364_, 0, v___x_2362_);
lean_ctor_set(v___x_2364_, 1, v___x_2363_);
v___x_2365_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__13));
v_sz_2366_ = lean_array_size(v_a_2359_);
v___x_2367_ = ((size_t)0ULL);
v___x_2368_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_2365_, v___f_2323_, v_sz_2366_, v___x_2367_, v_a_2359_);
v___x_2369_ = lean_array_to_list(v___x_2368_);
v___x_2370_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__14));
v___x_2371_ = lean_box(0);
v___x_2372_ = l_List_mapTR_loop___redArg(v___x_2370_, v___x_2369_, v___x_2371_);
v___x_2373_ = l_Lean_MessageData_ofList(v___x_2372_);
v___x_2374_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2374_, 0, v___x_2364_);
lean_ctor_set(v___x_2374_, 1, v___x_2373_);
v___x_2375_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16, &lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16_once, _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__16);
v___x_2376_ = l_Lean_indentD(v___x_2374_);
v___x_2377_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2377_, 0, v___x_2375_);
lean_ctor_set(v___x_2377_, 1, v___x_2376_);
lean_inc_ref(v___x_2312_);
v___x_8971__overap_2378_ = l_Lean_throwError___redArg(v___x_2312_, v___x_2324_, v___x_2377_);
lean_inc(v___y_2333_);
lean_inc_ref(v___y_2332_);
lean_inc(v___y_2331_);
lean_inc_ref(v___y_2330_);
lean_inc(v___y_2329_);
lean_inc(v___y_2328_);
lean_inc(v___y_2327_);
lean_inc_ref(v___y_2326_);
v___x_2379_ = lean_apply_9(v___x_8971__overap_2378_, v___y_2326_, v___y_2327_, v___y_2328_, v___y_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, lean_box(0));
if (lean_obj_tag(v___x_2379_) == 0)
{
lean_dec_ref_known(v___x_2379_, 1);
v___y_2339_ = v___y_2326_;
v___y_2340_ = v___y_2327_;
v___y_2341_ = v___y_2328_;
v___y_2342_ = v___y_2329_;
v___y_2343_ = v___y_2330_;
v___y_2344_ = v___y_2331_;
v___y_2345_ = v___y_2332_;
v___y_2346_ = v___y_2333_;
goto v___jp_2338_;
}
else
{
lean_dec(v_val_2337_);
lean_dec(v___f_2322_);
lean_dec_ref(v___x_2321_);
lean_dec(v_toMonadOptions_2320_);
lean_dec(v___f_2319_);
lean_dec_ref(v___x_2318_);
lean_dec_ref(v___f_2317_);
lean_dec(v___f_2316_);
lean_dec(v___x_2315_);
lean_dec(v___x_2314_);
lean_dec_ref(v___x_2312_);
return v___x_2379_;
}
}
else
{
lean_object* v_a_2380_; lean_object* v___x_2382_; uint8_t v_isShared_2383_; uint8_t v_isSharedCheck_2387_; 
lean_dec(v_val_2337_);
lean_dec_ref(v___x_2324_);
lean_dec_ref(v___f_2323_);
lean_dec(v___f_2322_);
lean_dec_ref(v___x_2321_);
lean_dec(v_toMonadOptions_2320_);
lean_dec(v___f_2319_);
lean_dec_ref(v___x_2318_);
lean_dec_ref(v___f_2317_);
lean_dec(v___f_2316_);
lean_dec(v___x_2315_);
lean_dec(v___x_2314_);
lean_dec_ref(v___x_2312_);
v_a_2380_ = lean_ctor_get(v___x_2358_, 0);
v_isSharedCheck_2387_ = !lean_is_exclusive(v___x_2358_);
if (v_isSharedCheck_2387_ == 0)
{
v___x_2382_ = v___x_2358_;
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
else
{
lean_inc(v_a_2380_);
lean_dec(v___x_2358_);
v___x_2382_ = lean_box(0);
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
v_resetjp_2381_:
{
lean_object* v___x_2385_; 
if (v_isShared_2383_ == 0)
{
v___x_2385_ = v___x_2382_;
goto v_reusejp_2384_;
}
else
{
lean_object* v_reuseFailAlloc_2386_; 
v_reuseFailAlloc_2386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2386_, 0, v_a_2380_);
v___x_2385_ = v_reuseFailAlloc_2386_;
goto v_reusejp_2384_;
}
v_reusejp_2384_:
{
return v___x_2385_;
}
}
}
}
}
else
{
lean_object* v_a_2388_; lean_object* v___x_2390_; uint8_t v_isShared_2391_; uint8_t v_isSharedCheck_2395_; 
lean_dec(v_val_2337_);
lean_dec_ref(v___x_2324_);
lean_dec_ref(v___f_2323_);
lean_dec(v___f_2322_);
lean_dec_ref(v___x_2321_);
lean_dec(v_toMonadOptions_2320_);
lean_dec(v___f_2319_);
lean_dec_ref(v___x_2318_);
lean_dec_ref(v___f_2317_);
lean_dec(v___f_2316_);
lean_dec(v___x_2315_);
lean_dec(v___x_2314_);
lean_dec_ref(v___x_2312_);
v_a_2388_ = lean_ctor_get(v___x_2354_, 0);
v_isSharedCheck_2395_ = !lean_is_exclusive(v___x_2354_);
if (v_isSharedCheck_2395_ == 0)
{
v___x_2390_ = v___x_2354_;
v_isShared_2391_ = v_isSharedCheck_2395_;
goto v_resetjp_2389_;
}
else
{
lean_inc(v_a_2388_);
lean_dec(v___x_2354_);
v___x_2390_ = lean_box(0);
v_isShared_2391_ = v_isSharedCheck_2395_;
goto v_resetjp_2389_;
}
v_resetjp_2389_:
{
lean_object* v___x_2393_; 
if (v_isShared_2391_ == 0)
{
v___x_2393_ = v___x_2390_;
goto v_reusejp_2392_;
}
else
{
lean_object* v_reuseFailAlloc_2394_; 
v_reuseFailAlloc_2394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2394_, 0, v_a_2388_);
v___x_2393_ = v_reuseFailAlloc_2394_;
goto v_reusejp_2392_;
}
v_reusejp_2392_:
{
return v___x_2393_;
}
}
}
v___jp_2338_:
{
lean_object* v___x_2347_; lean_object* v___f_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_8934__overap_2351_; lean_object* v___x_2352_; 
v___x_2347_ = lp_aesop_Aesop_TraceOption_proof;
lean_inc_ref(v___x_2312_);
v___f_2348_ = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__2___boxed), 19, 9);
lean_closure_set(v___f_2348_, 0, v___x_2314_);
lean_closure_set(v___f_2348_, 1, v___x_2315_);
lean_closure_set(v___f_2348_, 2, v___f_2316_);
lean_closure_set(v___f_2348_, 3, v___f_2317_);
lean_closure_set(v___f_2348_, 4, v___x_2347_);
lean_closure_set(v___f_2348_, 5, v_val_2337_);
lean_closure_set(v___f_2348_, 6, v___x_2312_);
lean_closure_set(v___f_2348_, 7, v___x_2318_);
lean_closure_set(v___f_2348_, 8, v___f_2319_);
v___x_2349_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_2312_, v_toMonadOptions_2320_, v___x_2347_);
v___x_2350_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_2350_, 0, lean_box(0));
lean_closure_set(v___x_2350_, 1, lean_box(0));
lean_closure_set(v___x_2350_, 2, v___x_2321_);
lean_closure_set(v___x_2350_, 3, lean_box(0));
lean_closure_set(v___x_2350_, 4, lean_box(0));
lean_closure_set(v___x_2350_, 5, v___x_2349_);
lean_closure_set(v___x_2350_, 6, v___f_2348_);
v___x_8934__overap_2351_ = lp_aesop_Aesop_withPPAnalyze___redArg(v___f_2322_, v___x_2350_);
lean_inc(v___y_2346_);
lean_inc_ref(v___y_2345_);
lean_inc(v___y_2344_);
lean_inc_ref(v___y_2343_);
lean_inc(v___y_2342_);
lean_inc(v___y_2341_);
lean_inc(v___y_2340_);
lean_inc_ref(v___y_2339_);
v___x_2352_ = lean_apply_9(v___x_8934__overap_2351_, v___y_2339_, v___y_2340_, v___y_2341_, v___y_2342_, v___y_2343_, v___y_2344_, v___y_2345_, v___y_2346_, lean_box(0));
return v___x_2352_;
}
}
else
{
lean_object* v___x_2396_; lean_object* v___x_8984__overap_2397_; lean_object* v___x_2398_; 
lean_dec(v_a_2336_);
lean_dec_ref(v___f_2323_);
lean_dec(v___f_2322_);
lean_dec_ref(v___x_2321_);
lean_dec(v_toMonadOptions_2320_);
lean_dec(v___f_2319_);
lean_dec_ref(v___x_2318_);
lean_dec_ref(v___f_2317_);
lean_dec(v___f_2316_);
lean_dec(v___x_2315_);
lean_dec(v___x_2314_);
lean_dec_ref(v___x_2313_);
v___x_2396_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18, &lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18_once, _init_lp_aesop_Aesop_finalizeProof___redArg___lam__3___closed__18);
v___x_8984__overap_2397_ = l_Lean_throwError___redArg(v___x_2312_, v___x_2324_, v___x_2396_);
lean_inc(v___y_2333_);
lean_inc_ref(v___y_2332_);
lean_inc(v___y_2331_);
lean_inc_ref(v___y_2330_);
lean_inc(v___y_2329_);
lean_inc(v___y_2328_);
lean_inc(v___y_2327_);
lean_inc_ref(v___y_2326_);
v___x_2398_ = lean_apply_9(v___x_8984__overap_2397_, v___y_2326_, v___y_2327_, v___y_2328_, v___y_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, lean_box(0));
return v___x_2398_;
}
}
else
{
lean_object* v_a_2399_; lean_object* v___x_2401_; uint8_t v_isShared_2402_; uint8_t v_isSharedCheck_2406_; 
lean_dec_ref(v___x_2324_);
lean_dec_ref(v___f_2323_);
lean_dec(v___f_2322_);
lean_dec_ref(v___x_2321_);
lean_dec(v_toMonadOptions_2320_);
lean_dec(v___f_2319_);
lean_dec_ref(v___x_2318_);
lean_dec_ref(v___f_2317_);
lean_dec(v___f_2316_);
lean_dec(v___x_2315_);
lean_dec(v___x_2314_);
lean_dec_ref(v___x_2313_);
lean_dec_ref(v___x_2312_);
v_a_2399_ = lean_ctor_get(v___x_2335_, 0);
v_isSharedCheck_2406_ = !lean_is_exclusive(v___x_2335_);
if (v_isSharedCheck_2406_ == 0)
{
v___x_2401_ = v___x_2335_;
v_isShared_2402_ = v_isSharedCheck_2406_;
goto v_resetjp_2400_;
}
else
{
lean_inc(v_a_2399_);
lean_dec(v___x_2335_);
v___x_2401_ = lean_box(0);
v_isShared_2402_ = v_isSharedCheck_2406_;
goto v_resetjp_2400_;
}
v_resetjp_2400_:
{
lean_object* v___x_2404_; 
if (v_isShared_2402_ == 0)
{
v___x_2404_ = v___x_2401_;
goto v_reusejp_2403_;
}
else
{
lean_object* v_reuseFailAlloc_2405_; 
v_reuseFailAlloc_2405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2405_, 0, v_a_2399_);
v___x_2404_ = v_reuseFailAlloc_2405_;
goto v_reusejp_2403_;
}
v_reusejp_2403_:
{
return v___x_2404_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed(lean_object** _args){
lean_object* v_inst_2407_ = _args[0];
lean_object* v___x_2408_ = _args[1];
lean_object* v___x_2409_ = _args[2];
lean_object* v___x_2410_ = _args[3];
lean_object* v___x_2411_ = _args[4];
lean_object* v___f_2412_ = _args[5];
lean_object* v___f_2413_ = _args[6];
lean_object* v___x_2414_ = _args[7];
lean_object* v___f_2415_ = _args[8];
lean_object* v_toMonadOptions_2416_ = _args[9];
lean_object* v___x_2417_ = _args[10];
lean_object* v___f_2418_ = _args[11];
lean_object* v___f_2419_ = _args[12];
lean_object* v___x_2420_ = _args[13];
lean_object* v_____r_2421_ = _args[14];
lean_object* v___y_2422_ = _args[15];
lean_object* v___y_2423_ = _args[16];
lean_object* v___y_2424_ = _args[17];
lean_object* v___y_2425_ = _args[18];
lean_object* v___y_2426_ = _args[19];
lean_object* v___y_2427_ = _args[20];
lean_object* v___y_2428_ = _args[21];
lean_object* v___y_2429_ = _args[22];
lean_object* v___y_2430_ = _args[23];
_start:
{
lean_object* v_res_2431_; 
v_res_2431_ = lp_aesop_Aesop_finalizeProof___redArg___lam__3(v_inst_2407_, v___x_2408_, v___x_2409_, v___x_2410_, v___x_2411_, v___f_2412_, v___f_2413_, v___x_2414_, v___f_2415_, v_toMonadOptions_2416_, v___x_2417_, v___f_2418_, v___f_2419_, v___x_2420_, v_____r_2421_, v___y_2422_, v___y_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_, v___y_2428_, v___y_2429_);
lean_dec(v___y_2429_);
lean_dec_ref(v___y_2428_);
lean_dec(v___y_2427_);
lean_dec_ref(v___y_2426_);
lean_dec(v___y_2425_);
lean_dec(v___y_2424_);
lean_dec(v___y_2423_);
lean_dec_ref(v___y_2422_);
lean_dec_ref(v_inst_2407_);
return v_res_2431_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__0(void){
_start:
{
lean_object* v___x_2432_; 
v___x_2432_ = l_instMonadControlStateRefT_x27(lean_box(0), lean_box(0), lean_box(0));
return v___x_2432_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__1(void){
_start:
{
lean_object* v___x_2433_; 
v___x_2433_ = l_instMonadControlReaderT(lean_box(0), lean_box(0));
return v___x_2433_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__4(void){
_start:
{
lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___f_2438_; 
v___x_2436_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3));
v___x_2437_ = l_Lean_Core_instMonadWithOptionsCoreM;
v___f_2438_ = lean_alloc_closure((void*)(l_Lean_instMonadWithOptionsOfMonadFunctor___redArg___lam__1), 5, 2);
lean_closure_set(v___f_2438_, 0, v___x_2437_);
lean_closure_set(v___f_2438_, 1, v___x_2436_);
return v___f_2438_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__5(void){
_start:
{
lean_object* v___f_2439_; lean_object* v___f_2440_; lean_object* v___f_2441_; 
v___f_2439_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__2));
v___f_2440_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__4, &lp_aesop_Aesop_finalizeProof___redArg___closed__4_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__4);
v___f_2441_ = lean_alloc_closure((void*)(l_Lean_instMonadWithOptionsOfMonadFunctor___redArg___lam__1), 5, 2);
lean_closure_set(v___f_2441_, 0, v___f_2440_);
lean_closure_set(v___f_2441_, 1, v___f_2439_);
return v___f_2441_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__6(void){
_start:
{
lean_object* v___x_2442_; lean_object* v___f_2443_; lean_object* v___f_2444_; 
v___x_2442_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3));
v___f_2443_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__5, &lp_aesop_Aesop_finalizeProof___redArg___closed__5_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__5);
v___f_2444_ = lean_alloc_closure((void*)(l_Lean_instMonadWithOptionsOfMonadFunctor___redArg___lam__1), 5, 2);
lean_closure_set(v___f_2444_, 0, v___f_2443_);
lean_closure_set(v___f_2444_, 1, v___x_2442_);
return v___f_2444_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__7(void){
_start:
{
lean_object* v___x_2445_; lean_object* v___f_2446_; lean_object* v___f_2447_; 
v___x_2445_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3));
v___f_2446_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__6, &lp_aesop_Aesop_finalizeProof___redArg___closed__6_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__6);
v___f_2447_ = lean_alloc_closure((void*)(l_Lean_instMonadWithOptionsOfMonadFunctor___redArg___lam__1), 5, 2);
lean_closure_set(v___f_2447_, 0, v___f_2446_);
lean_closure_set(v___f_2447_, 1, v___x_2445_);
return v___f_2447_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__8(void){
_start:
{
lean_object* v___x_2448_; lean_object* v___f_2449_; lean_object* v___f_2450_; 
v___x_2448_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3));
v___f_2449_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__7, &lp_aesop_Aesop_finalizeProof___redArg___closed__7_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__7);
v___f_2450_ = lean_alloc_closure((void*)(l_Lean_instMonadWithOptionsOfMonadFunctor___redArg___lam__1), 5, 2);
lean_closure_set(v___f_2450_, 0, v___f_2449_);
lean_closure_set(v___f_2450_, 1, v___x_2448_);
return v___f_2450_;
}
}
static lean_object* _init_lp_aesop_Aesop_finalizeProof___redArg___closed__9(void){
_start:
{
lean_object* v___f_2451_; lean_object* v___f_2452_; lean_object* v___f_2453_; 
v___f_2451_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__2));
v___f_2452_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__8, &lp_aesop_Aesop_finalizeProof___redArg___closed__8_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__8);
v___f_2453_ = lean_alloc_closure((void*)(l_Lean_instMonadWithOptionsOfMonadFunctor___redArg___lam__1), 5, 2);
lean_closure_set(v___f_2453_, 0, v___f_2452_);
lean_closure_set(v___f_2453_, 1, v___f_2451_);
return v___f_2453_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg(lean_object* v_inst_2454_, lean_object* v_a_2455_, lean_object* v_a_2456_, lean_object* v_a_2457_, lean_object* v_a_2458_, lean_object* v_a_2459_, lean_object* v_a_2460_, lean_object* v_a_2461_, lean_object* v_a_2462_){
_start:
{
lean_object* v___x_2464_; lean_object* v___x_2465_; lean_object* v_toApplicative_2466_; lean_object* v_toFunctor_2467_; lean_object* v_toSeq_2468_; lean_object* v_toSeqLeft_2469_; lean_object* v_toSeqRight_2470_; lean_object* v___f_2471_; lean_object* v___f_2472_; lean_object* v___f_2473_; lean_object* v___f_2474_; lean_object* v___x_2475_; lean_object* v___f_2476_; lean_object* v___f_2477_; lean_object* v___f_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___f_2484_; lean_object* v___f_2485_; lean_object* v___x_2486_; lean_object* v___f_2487_; lean_object* v___f_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v_getMCtx_2491_; lean_object* v_modifyMCtx_2492_; lean_object* v___x_2493_; lean_object* v_toApplicative_2494_; lean_object* v_toFunctor_2495_; lean_object* v_toSeq_2496_; lean_object* v_toSeqLeft_2497_; lean_object* v_toSeqRight_2498_; lean_object* v___f_2499_; lean_object* v___f_2500_; lean_object* v___x_2501_; lean_object* v___f_2502_; lean_object* v___f_2503_; lean_object* v___f_2504_; lean_object* v___x_2505_; lean_object* v___x_2506_; lean_object* v___x_2507_; lean_object* v_toApplicative_2508_; lean_object* v___x_2510_; uint8_t v_isShared_2511_; uint8_t v_isSharedCheck_2585_; 
v___x_2464_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__0, &lp_aesop_Aesop_finalizeProof___redArg___closed__0_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__0);
v___x_2465_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1);
v_toApplicative_2466_ = lean_ctor_get(v___x_2465_, 0);
v_toFunctor_2467_ = lean_ctor_get(v_toApplicative_2466_, 0);
v_toSeq_2468_ = lean_ctor_get(v_toApplicative_2466_, 2);
v_toSeqLeft_2469_ = lean_ctor_get(v_toApplicative_2466_, 3);
v_toSeqRight_2470_ = lean_ctor_get(v_toApplicative_2466_, 4);
v___f_2471_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2));
v___f_2472_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3));
lean_inc_ref_n(v_toFunctor_2467_, 2);
v___f_2473_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2473_, 0, v_toFunctor_2467_);
v___f_2474_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2474_, 0, v_toFunctor_2467_);
v___x_2475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2475_, 0, v___f_2473_);
lean_ctor_set(v___x_2475_, 1, v___f_2474_);
lean_inc(v_toSeqRight_2470_);
v___f_2476_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2476_, 0, v_toSeqRight_2470_);
lean_inc(v_toSeqLeft_2469_);
v___f_2477_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2477_, 0, v_toSeqLeft_2469_);
lean_inc(v_toSeq_2468_);
v___f_2478_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2478_, 0, v_toSeq_2468_);
v___x_2479_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2479_, 0, v___x_2475_);
lean_ctor_set(v___x_2479_, 1, v___f_2471_);
lean_ctor_set(v___x_2479_, 2, v___f_2478_);
lean_ctor_set(v___x_2479_, 3, v___f_2477_);
lean_ctor_set(v___x_2479_, 4, v___f_2476_);
v___x_2480_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2480_, 0, v___x_2479_);
lean_ctor_set(v___x_2480_, 1, v___f_2472_);
v___x_2481_ = l_StateRefT_x27_instMonad___redArg(v___x_2480_);
v___x_2482_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_2482_, 0, lean_box(0));
lean_closure_set(v___x_2482_, 1, lean_box(0));
lean_closure_set(v___x_2482_, 2, v___x_2481_);
v___x_2483_ = l_instMonadControlTOfPure___redArg(v___x_2482_);
lean_inc_ref(v___x_2483_);
v___f_2484_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2484_, 0, v___x_2464_);
lean_closure_set(v___f_2484_, 1, v___x_2483_);
v___f_2485_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2485_, 0, v___x_2464_);
lean_closure_set(v___f_2485_, 1, v___x_2483_);
v___x_2486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2486_, 0, v___f_2484_);
lean_ctor_set(v___x_2486_, 1, v___f_2485_);
lean_inc_ref(v___x_2486_);
v___f_2487_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2487_, 0, v___x_2464_);
lean_closure_set(v___f_2487_, 1, v___x_2486_);
v___f_2488_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2488_, 0, v___x_2464_);
lean_closure_set(v___f_2488_, 1, v___x_2486_);
v___x_2489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2489_, 0, v___f_2487_);
lean_ctor_set(v___x_2489_, 1, v___f_2488_);
v___x_2490_ = l_Lean_Meta_instMonadMCtxMetaM;
v_getMCtx_2491_ = lean_ctor_get(v___x_2490_, 0);
v_modifyMCtx_2492_ = lean_ctor_get(v___x_2490_, 1);
v___x_2493_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_2454_);
v_toApplicative_2494_ = lean_ctor_get(v___x_2465_, 0);
v_toFunctor_2495_ = lean_ctor_get(v_toApplicative_2494_, 0);
v_toSeq_2496_ = lean_ctor_get(v_toApplicative_2494_, 2);
v_toSeqLeft_2497_ = lean_ctor_get(v_toApplicative_2494_, 3);
v_toSeqRight_2498_ = lean_ctor_get(v_toApplicative_2494_, 4);
lean_inc_ref_n(v_toFunctor_2495_, 2);
v___f_2499_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2499_, 0, v_toFunctor_2495_);
v___f_2500_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2500_, 0, v_toFunctor_2495_);
v___x_2501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2501_, 0, v___f_2499_);
lean_ctor_set(v___x_2501_, 1, v___f_2500_);
lean_inc(v_toSeqRight_2498_);
v___f_2502_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2502_, 0, v_toSeqRight_2498_);
lean_inc(v_toSeqLeft_2497_);
v___f_2503_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2503_, 0, v_toSeqLeft_2497_);
lean_inc(v_toSeq_2496_);
v___f_2504_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2504_, 0, v_toSeq_2496_);
v___x_2505_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2505_, 0, v___x_2501_);
lean_ctor_set(v___x_2505_, 1, v___f_2471_);
lean_ctor_set(v___x_2505_, 2, v___f_2504_);
lean_ctor_set(v___x_2505_, 3, v___f_2503_);
lean_ctor_set(v___x_2505_, 4, v___f_2502_);
v___x_2506_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2506_, 0, v___x_2505_);
lean_ctor_set(v___x_2506_, 1, v___f_2472_);
v___x_2507_ = l_StateRefT_x27_instMonad___redArg(v___x_2506_);
v_toApplicative_2508_ = lean_ctor_get(v___x_2507_, 0);
v_isSharedCheck_2585_ = !lean_is_exclusive(v___x_2507_);
if (v_isSharedCheck_2585_ == 0)
{
lean_object* v_unused_2586_; 
v_unused_2586_ = lean_ctor_get(v___x_2507_, 1);
lean_dec(v_unused_2586_);
v___x_2510_ = v___x_2507_;
v_isShared_2511_ = v_isSharedCheck_2585_;
goto v_resetjp_2509_;
}
else
{
lean_inc(v_toApplicative_2508_);
lean_dec(v___x_2507_);
v___x_2510_ = lean_box(0);
v_isShared_2511_ = v_isSharedCheck_2585_;
goto v_resetjp_2509_;
}
v_resetjp_2509_:
{
lean_object* v_toFunctor_2512_; lean_object* v_toSeq_2513_; lean_object* v_toSeqLeft_2514_; lean_object* v_toSeqRight_2515_; lean_object* v___x_2517_; uint8_t v_isShared_2518_; uint8_t v_isSharedCheck_2583_; 
v_toFunctor_2512_ = lean_ctor_get(v_toApplicative_2508_, 0);
v_toSeq_2513_ = lean_ctor_get(v_toApplicative_2508_, 2);
v_toSeqLeft_2514_ = lean_ctor_get(v_toApplicative_2508_, 3);
v_toSeqRight_2515_ = lean_ctor_get(v_toApplicative_2508_, 4);
v_isSharedCheck_2583_ = !lean_is_exclusive(v_toApplicative_2508_);
if (v_isSharedCheck_2583_ == 0)
{
lean_object* v_unused_2584_; 
v_unused_2584_ = lean_ctor_get(v_toApplicative_2508_, 1);
lean_dec(v_unused_2584_);
v___x_2517_ = v_toApplicative_2508_;
v_isShared_2518_ = v_isSharedCheck_2583_;
goto v_resetjp_2516_;
}
else
{
lean_inc(v_toSeqRight_2515_);
lean_inc(v_toSeqLeft_2514_);
lean_inc(v_toSeq_2513_);
lean_inc(v_toFunctor_2512_);
lean_dec(v_toApplicative_2508_);
v___x_2517_ = lean_box(0);
v_isShared_2518_ = v_isSharedCheck_2583_;
goto v_resetjp_2516_;
}
v_resetjp_2516_:
{
lean_object* v___f_2519_; lean_object* v___x_2520_; lean_object* v___f_2521_; lean_object* v___f_2522_; lean_object* v___f_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___f_2527_; lean_object* v___f_2528_; lean_object* v___f_2529_; lean_object* v___f_2530_; lean_object* v___x_2531_; lean_object* v___f_2532_; lean_object* v___f_2533_; lean_object* v___f_2534_; lean_object* v___x_2536_; 
v___f_2519_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___x_2520_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
lean_inc(v_modifyMCtx_2492_);
v___f_2521_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2521_, 0, v_modifyMCtx_2492_);
lean_closure_set(v___f_2521_, 1, v___x_2520_);
v___f_2522_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2522_, 0, v___f_2521_);
lean_closure_set(v___f_2522_, 1, v___x_2520_);
v___f_2523_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2523_, 0, v___f_2522_);
lean_closure_set(v___f_2523_, 1, v___f_2519_);
lean_inc(v_getMCtx_2491_);
v___x_2524_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_2524_, 0, lean_box(0));
lean_closure_set(v___x_2524_, 1, lean_box(0));
lean_closure_set(v___x_2524_, 2, lean_box(0));
lean_closure_set(v___x_2524_, 3, lean_box(0));
lean_closure_set(v___x_2524_, 4, v_getMCtx_2491_);
v___x_2525_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_2525_, 0, lean_box(0));
lean_closure_set(v___x_2525_, 1, lean_box(0));
lean_closure_set(v___x_2525_, 2, lean_box(0));
lean_closure_set(v___x_2525_, 3, lean_box(0));
lean_closure_set(v___x_2525_, 4, v___x_2524_);
v___x_2526_ = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(v___x_2526_, 0, lean_box(0));
lean_closure_set(v___x_2526_, 1, v___x_2525_);
v___f_2527_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__4));
v___f_2528_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__5));
lean_inc_ref(v_toFunctor_2512_);
v___f_2529_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2529_, 0, v_toFunctor_2512_);
v___f_2530_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2530_, 0, v_toFunctor_2512_);
v___x_2531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2531_, 0, v___f_2529_);
lean_ctor_set(v___x_2531_, 1, v___f_2530_);
v___f_2532_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2532_, 0, v_toSeqRight_2515_);
v___f_2533_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2533_, 0, v_toSeqLeft_2514_);
v___f_2534_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2534_, 0, v_toSeq_2513_);
if (v_isShared_2518_ == 0)
{
lean_ctor_set(v___x_2517_, 4, v___f_2532_);
lean_ctor_set(v___x_2517_, 3, v___f_2533_);
lean_ctor_set(v___x_2517_, 2, v___f_2534_);
lean_ctor_set(v___x_2517_, 1, v___f_2527_);
lean_ctor_set(v___x_2517_, 0, v___x_2531_);
v___x_2536_ = v___x_2517_;
goto v_reusejp_2535_;
}
else
{
lean_object* v_reuseFailAlloc_2582_; 
v_reuseFailAlloc_2582_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2582_, 0, v___x_2531_);
lean_ctor_set(v_reuseFailAlloc_2582_, 1, v___f_2527_);
lean_ctor_set(v_reuseFailAlloc_2582_, 2, v___f_2534_);
lean_ctor_set(v_reuseFailAlloc_2582_, 3, v___f_2533_);
lean_ctor_set(v_reuseFailAlloc_2582_, 4, v___f_2532_);
v___x_2536_ = v_reuseFailAlloc_2582_;
goto v_reusejp_2535_;
}
v_reusejp_2535_:
{
lean_object* v___x_2538_; 
if (v_isShared_2511_ == 0)
{
lean_ctor_set(v___x_2510_, 1, v___f_2528_);
lean_ctor_set(v___x_2510_, 0, v___x_2536_);
v___x_2538_ = v___x_2510_;
goto v_reusejp_2537_;
}
else
{
lean_object* v_reuseFailAlloc_2581_; 
v_reuseFailAlloc_2581_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2581_, 0, v___x_2536_);
lean_ctor_set(v_reuseFailAlloc_2581_, 1, v___f_2528_);
v___x_2538_ = v_reuseFailAlloc_2581_;
goto v_reusejp_2537_;
}
v_reusejp_2537_:
{
lean_object* v___x_2539_; lean_object* v___x_2540_; lean_object* v___x_2541_; lean_object* v___f_2542_; lean_object* v___x_2543_; lean_object* v___f_2544_; lean_object* v___f_2545_; lean_object* v___x_2546_; lean_object* v___f_2547_; lean_object* v___f_2548_; lean_object* v___x_2549_; lean_object* v___f_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; lean_object* v_toMonadOptions_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___f_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v_iteration_2561_; lean_object* v_ruleSet_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; 
v___x_2539_ = l_StateRefT_x27_instMonad___redArg(v___x_2538_);
v___x_2540_ = l_StateRefT_x27_instMonad___redArg(v___x_2539_);
v___x_2541_ = l_StateRefT_x27_instMonad___redArg(v___x_2540_);
v___f_2542_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__17));
v___x_2543_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__1, &lp_aesop_Aesop_finalizeProof___redArg___closed__1_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__1);
lean_inc_ref(v___x_2489_);
v___f_2544_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2544_, 0, v___x_2464_);
lean_closure_set(v___f_2544_, 1, v___x_2489_);
v___f_2545_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2545_, 0, v___x_2464_);
lean_closure_set(v___f_2545_, 1, v___x_2489_);
v___x_2546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2546_, 0, v___f_2544_);
lean_ctor_set(v___x_2546_, 1, v___f_2545_);
lean_inc_ref(v___x_2546_);
v___f_2547_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2547_, 0, v___x_2543_);
lean_closure_set(v___f_2547_, 1, v___x_2546_);
v___f_2548_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_2548_, 0, v___x_2543_);
lean_closure_set(v___f_2548_, 1, v___x_2546_);
v___x_2549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2549_, 0, v___f_2547_);
lean_ctor_set(v___x_2549_, 1, v___f_2548_);
v___f_2550_ = lean_alloc_closure((void*)(l_Lean_instMonadMCtxOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2550_, 0, v___f_2523_);
lean_closure_set(v___f_2550_, 1, v___f_2542_);
v___x_2551_ = lean_alloc_closure((void*)(lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed), 11, 2);
lean_closure_set(v___x_2551_, 0, lean_box(0));
lean_closure_set(v___x_2551_, 1, v___x_2526_);
v___x_2552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2552_, 0, v___x_2551_);
lean_ctor_set(v___x_2552_, 1, v___f_2550_);
v___x_2553_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2);
v_toMonadOptions_2554_ = lean_ctor_get(v___x_2553_, 0);
v___x_2555_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___x_2556_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_2454_);
v___f_2557_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
lean_inc_ref(v___x_2493_);
v___x_2558_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_2557_, v___x_2493_);
lean_inc_ref(v___x_2556_);
v___x_2559_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2559_, 0, v___x_2555_);
lean_ctor_set(v___x_2559_, 1, v___x_2556_);
lean_ctor_set(v___x_2559_, 2, v___x_2558_);
v___x_2560_ = lean_st_ref_get(v_a_2456_);
v_iteration_2561_ = lean_ctor_get(v___x_2560_, 0);
lean_inc(v_iteration_2561_);
lean_dec(v___x_2560_);
v_ruleSet_2562_ = lean_ctor_get(v_a_2455_, 0);
lean_inc_ref(v_ruleSet_2562_);
v___x_2563_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2563_, 0, v_iteration_2561_);
lean_ctor_set(v___x_2563_, 1, v_ruleSet_2562_);
v___x_2564_ = lp_aesop_Aesop_getRootMVarId(v___x_2563_, v_a_2457_, v_a_2458_, v_a_2459_, v_a_2460_, v_a_2461_, v_a_2462_);
lean_dec_ref_known(v___x_2563_, 2);
if (lean_obj_tag(v___x_2564_) == 0)
{
lean_object* v_a_2565_; lean_object* v___f_2566_; lean_object* v___f_2567_; lean_object* v___f_2568_; lean_object* v___f_2569_; lean_object* v___x_2570_; lean_object* v___x_8658__overap_2571_; lean_object* v___x_2572_; 
v_a_2565_ = lean_ctor_get(v___x_2564_, 0);
lean_inc(v_a_2565_);
lean_dec_ref_known(v___x_2564_, 1);
v___f_2566_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___closed__2));
v___f_2567_ = ((lean_object*)(lp_aesop_Aesop_finalizeProof___redArg___closed__3));
v___f_2568_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__9, &lp_aesop_Aesop_finalizeProof___redArg___closed__9_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__9);
lean_inc_ref(v___x_2541_);
lean_inc(v_toMonadOptions_2554_);
lean_inc_ref(v___x_2493_);
v___f_2569_ = lean_alloc_closure((void*)(lp_aesop_Aesop_finalizeProof___redArg___lam__3___boxed), 24, 14);
lean_closure_set(v___f_2569_, 0, v_inst_2454_);
lean_closure_set(v___f_2569_, 1, v___x_2493_);
lean_closure_set(v___f_2569_, 2, v___x_2552_);
lean_closure_set(v___f_2569_, 3, v___x_2520_);
lean_closure_set(v___f_2569_, 4, v___x_2520_);
lean_closure_set(v___f_2569_, 5, v___f_2519_);
lean_closure_set(v___f_2569_, 6, v___f_2542_);
lean_closure_set(v___f_2569_, 7, v___x_2556_);
lean_closure_set(v___f_2569_, 8, v___f_2557_);
lean_closure_set(v___f_2569_, 9, v_toMonadOptions_2554_);
lean_closure_set(v___f_2569_, 10, v___x_2541_);
lean_closure_set(v___f_2569_, 11, v___f_2568_);
lean_closure_set(v___f_2569_, 12, v___f_2567_);
lean_closure_set(v___f_2569_, 13, v___x_2559_);
v___x_2570_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_2570_, 0, lean_box(0));
lean_closure_set(v___x_2570_, 1, lean_box(0));
lean_closure_set(v___x_2570_, 2, v___x_2541_);
lean_closure_set(v___x_2570_, 3, lean_box(0));
lean_closure_set(v___x_2570_, 4, lean_box(0));
lean_closure_set(v___x_2570_, 5, v___f_2566_);
lean_closure_set(v___x_2570_, 6, v___f_2569_);
v___x_8658__overap_2571_ = l_Lean_MVarId_withContext___redArg(v___x_2549_, v___x_2493_, v_a_2565_, v___x_2570_);
lean_inc(v_a_2462_);
lean_inc_ref(v_a_2461_);
lean_inc(v_a_2460_);
lean_inc_ref(v_a_2459_);
lean_inc(v_a_2458_);
lean_inc(v_a_2457_);
lean_inc(v_a_2456_);
lean_inc_ref(v_a_2455_);
v___x_2572_ = lean_apply_9(v___x_8658__overap_2571_, v_a_2455_, v_a_2456_, v_a_2457_, v_a_2458_, v_a_2459_, v_a_2460_, v_a_2461_, v_a_2462_, lean_box(0));
return v___x_2572_;
}
else
{
lean_object* v_a_2573_; lean_object* v___x_2575_; uint8_t v_isShared_2576_; uint8_t v_isSharedCheck_2580_; 
lean_dec_ref_known(v___x_2559_, 3);
lean_dec_ref(v___x_2556_);
lean_dec_ref_known(v___x_2552_, 2);
lean_dec_ref_known(v___x_2549_, 2);
lean_dec_ref(v___x_2541_);
lean_dec_ref(v___x_2493_);
lean_dec_ref(v_inst_2454_);
v_a_2573_ = lean_ctor_get(v___x_2564_, 0);
v_isSharedCheck_2580_ = !lean_is_exclusive(v___x_2564_);
if (v_isSharedCheck_2580_ == 0)
{
v___x_2575_ = v___x_2564_;
v_isShared_2576_ = v_isSharedCheck_2580_;
goto v_resetjp_2574_;
}
else
{
lean_inc(v_a_2573_);
lean_dec(v___x_2564_);
v___x_2575_ = lean_box(0);
v_isShared_2576_ = v_isSharedCheck_2580_;
goto v_resetjp_2574_;
}
v_resetjp_2574_:
{
lean_object* v___x_2578_; 
if (v_isShared_2576_ == 0)
{
v___x_2578_ = v___x_2575_;
goto v_reusejp_2577_;
}
else
{
lean_object* v_reuseFailAlloc_2579_; 
v_reuseFailAlloc_2579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2579_, 0, v_a_2573_);
v___x_2578_ = v_reuseFailAlloc_2579_;
goto v_reusejp_2577_;
}
v_reusejp_2577_:
{
return v___x_2578_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___redArg___boxed(lean_object* v_inst_2587_, lean_object* v_a_2588_, lean_object* v_a_2589_, lean_object* v_a_2590_, lean_object* v_a_2591_, lean_object* v_a_2592_, lean_object* v_a_2593_, lean_object* v_a_2594_, lean_object* v_a_2595_, lean_object* v_a_2596_){
_start:
{
lean_object* v_res_2597_; 
v_res_2597_ = lp_aesop_Aesop_finalizeProof___redArg(v_inst_2587_, v_a_2588_, v_a_2589_, v_a_2590_, v_a_2591_, v_a_2592_, v_a_2593_, v_a_2594_, v_a_2595_);
lean_dec(v_a_2595_);
lean_dec_ref(v_a_2594_);
lean_dec(v_a_2593_);
lean_dec_ref(v_a_2592_);
lean_dec(v_a_2591_);
lean_dec(v_a_2590_);
lean_dec(v_a_2589_);
lean_dec_ref(v_a_2588_);
return v_res_2597_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof(lean_object* v_Q_2598_, lean_object* v_inst_2599_, lean_object* v_a_2600_, lean_object* v_a_2601_, lean_object* v_a_2602_, lean_object* v_a_2603_, lean_object* v_a_2604_, lean_object* v_a_2605_, lean_object* v_a_2606_, lean_object* v_a_2607_){
_start:
{
lean_object* v___x_2609_; 
v___x_2609_ = lp_aesop_Aesop_finalizeProof___redArg(v_inst_2599_, v_a_2600_, v_a_2601_, v_a_2602_, v_a_2603_, v_a_2604_, v_a_2605_, v_a_2606_, v_a_2607_);
return v___x_2609_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finalizeProof___boxed(lean_object* v_Q_2610_, lean_object* v_inst_2611_, lean_object* v_a_2612_, lean_object* v_a_2613_, lean_object* v_a_2614_, lean_object* v_a_2615_, lean_object* v_a_2616_, lean_object* v_a_2617_, lean_object* v_a_2618_, lean_object* v_a_2619_, lean_object* v_a_2620_){
_start:
{
lean_object* v_res_2621_; 
v_res_2621_ = lp_aesop_Aesop_finalizeProof(v_Q_2610_, v_inst_2611_, v_a_2612_, v_a_2613_, v_a_2614_, v_a_2615_, v_a_2616_, v_a_2617_, v_a_2618_, v_a_2619_);
lean_dec(v_a_2619_);
lean_dec_ref(v_a_2618_);
lean_dec(v_a_2617_);
lean_dec_ref(v_a_2616_);
lean_dec(v_a_2615_);
lean_dec(v_a_2614_);
lean_dec(v_a_2613_);
lean_dec_ref(v_a_2612_);
return v_res_2621_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2623_; lean_object* v___x_2624_; 
v___x_2623_ = ((lean_object*)(lp_aesop_Aesop_traceScript___redArg___lam__0___closed__0));
v___x_2624_ = l_Lean_stringToMessageData(v___x_2623_);
return v___x_2624_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0(lean_object* v_x_2625_, lean_object* v___y_2626_, lean_object* v___y_2627_, lean_object* v___y_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_, lean_object* v___y_2632_){
_start:
{
lean_object* v___x_2634_; lean_object* v___x_2635_; 
v___x_2634_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___lam__0___closed__1, &lp_aesop_Aesop_traceScript___redArg___lam__0___closed__1_once, _init_lp_aesop_Aesop_traceScript___redArg___lam__0___closed__1);
v___x_2635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2635_, 0, v___x_2634_);
return v___x_2635_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__0___boxed(lean_object* v_x_2636_, lean_object* v___y_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_){
_start:
{
lean_object* v_res_2645_; 
v_res_2645_ = lp_aesop_Aesop_traceScript___redArg___lam__0(v_x_2636_, v___y_2637_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_);
lean_dec(v___y_2643_);
lean_dec_ref(v___y_2642_);
lean_dec(v___y_2641_);
lean_dec_ref(v___y_2640_);
lean_dec(v___y_2639_);
lean_dec(v___y_2638_);
lean_dec_ref(v___y_2637_);
lean_dec_ref(v_x_2636_);
return v_res_2645_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2(void){
_start:
{
lean_object* v___x_2648_; lean_object* v___x_2649_; 
v___x_2648_ = ((lean_object*)(lp_aesop_Aesop_traceScript___redArg___lam__1___closed__1));
v___x_2649_ = l_Lean_stringToMessageData(v___x_2648_);
return v___x_2649_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1(lean_object* v___x_2650_, lean_object* v___x_2651_, lean_object* v___x_2652_, lean_object* v___x_2653_, lean_object* v___f_2654_, lean_object* v___f_2655_, lean_object* v___x_2656_, lean_object* v___f_2657_, lean_object* v_inst_2658_, lean_object* v_options_2659_, uint8_t v_completeProof_2660_, lean_object* v_____x_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_, lean_object* v___y_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_, lean_object* v___y_2669_){
_start:
{
lean_object* v_fst_2671_; lean_object* v_snd_2672_; lean_object* v___x_2674_; uint8_t v_isShared_2675_; uint8_t v_isSharedCheck_2795_; 
v_fst_2671_ = lean_ctor_get(v_____x_2661_, 0);
v_snd_2672_ = lean_ctor_get(v_____x_2661_, 1);
v_isSharedCheck_2795_ = !lean_is_exclusive(v_____x_2661_);
if (v_isSharedCheck_2795_ == 0)
{
v___x_2674_ = v_____x_2661_;
v_isShared_2675_ = v_isSharedCheck_2795_;
goto v_resetjp_2673_;
}
else
{
lean_inc(v_snd_2672_);
lean_inc(v_fst_2671_);
lean_dec(v_____x_2661_);
v___x_2674_ = lean_box(0);
v_isShared_2675_ = v_isSharedCheck_2795_;
goto v_resetjp_2673_;
}
v_resetjp_2673_:
{
lean_object* v___x_2676_; lean_object* v___x_2677_; 
v___x_2676_ = lean_st_ref_get(v___y_2663_);
lean_dec(v___x_2676_);
v___x_2677_ = lp_aesop_Aesop_Script_UScript_checkIfEnabled(v_fst_2671_, v___y_2666_, v___y_2667_, v___y_2668_, v___y_2669_);
if (lean_obj_tag(v___x_2677_) == 0)
{
lean_object* v___x_2678_; lean_object* v_iteration_2679_; lean_object* v_ruleSet_2680_; lean_object* v___x_2682_; 
lean_dec_ref_known(v___x_2677_, 1);
v___x_2678_ = lean_st_ref_get(v___y_2663_);
v_iteration_2679_ = lean_ctor_get(v___x_2678_, 0);
lean_inc(v_iteration_2679_);
lean_dec(v___x_2678_);
v_ruleSet_2680_ = lean_ctor_get(v___y_2662_, 0);
lean_inc_ref(v_ruleSet_2680_);
if (v_isShared_2675_ == 0)
{
lean_ctor_set(v___x_2674_, 1, v_ruleSet_2680_);
lean_ctor_set(v___x_2674_, 0, v_iteration_2679_);
v___x_2682_ = v___x_2674_;
goto v_reusejp_2681_;
}
else
{
lean_object* v_reuseFailAlloc_2794_; 
v_reuseFailAlloc_2794_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2794_, 0, v_iteration_2679_);
lean_ctor_set(v_reuseFailAlloc_2794_, 1, v_ruleSet_2680_);
v___x_2682_ = v_reuseFailAlloc_2794_;
goto v_reusejp_2681_;
}
v_reusejp_2681_:
{
lean_object* v___x_2683_; 
v___x_2683_ = lp_aesop_Aesop_getRootMVarId(v___x_2682_, v___y_2664_, v___y_2665_, v___y_2666_, v___y_2667_, v___y_2668_, v___y_2669_);
lean_dec_ref(v___x_2682_);
if (lean_obj_tag(v___x_2683_) == 0)
{
lean_object* v_a_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; 
v_a_2684_ = lean_ctor_get(v___x_2683_, 0);
lean_inc(v_a_2684_);
lean_dec_ref_known(v___x_2683_, 1);
v___x_2685_ = lean_st_ref_get(v___y_2663_);
lean_dec(v___x_2685_);
v___x_2686_ = lp_aesop_Aesop_getRootMetaState___redArg(v___y_2664_);
if (lean_obj_tag(v___x_2686_) == 0)
{
lean_object* v_a_2687_; lean_object* v_toMonadOptions_2688_; lean_object* v___x_2689_; lean_object* v___x_46986__overap_2690_; lean_object* v___x_2691_; 
v_a_2687_ = lean_ctor_get(v___x_2686_, 0);
lean_inc(v_a_2687_);
lean_dec_ref_known(v___x_2686_, 1);
v_toMonadOptions_2688_ = lean_ctor_get(v___x_2650_, 0);
v___x_2689_ = lp_aesop_Aesop_TraceOption_script;
lean_inc(v_toMonadOptions_2688_);
lean_inc_ref(v___x_2651_);
v___x_46986__overap_2690_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_2651_, v_toMonadOptions_2688_, v___x_2689_);
lean_inc(v___y_2669_);
lean_inc_ref(v___y_2668_);
lean_inc(v___y_2667_);
lean_inc_ref(v___y_2666_);
lean_inc(v___y_2665_);
lean_inc(v___y_2664_);
lean_inc(v___y_2663_);
lean_inc_ref(v___y_2662_);
v___x_2691_ = lean_apply_9(v___x_46986__overap_2690_, v___y_2662_, v___y_2663_, v___y_2664_, v___y_2665_, v___y_2666_, v___y_2667_, v___y_2668_, v___y_2669_, lean_box(0));
if (lean_obj_tag(v___x_2691_) == 0)
{
lean_object* v_a_2692_; lean_object* v___f_2693_; lean_object* v___f_2694_; lean_object* v___f_2695_; lean_object* v___f_2696_; lean_object* v___f_2697_; lean_object* v___y_2699_; lean_object* v___y_2700_; lean_object* v___y_2701_; lean_object* v___y_2702_; lean_object* v___y_2703_; lean_object* v___y_2704_; lean_object* v___y_2705_; lean_object* v___y_2706_; uint8_t v___x_2738_; 
v_a_2692_ = lean_ctor_get(v___x_2691_, 0);
lean_inc(v_a_2692_);
lean_dec_ref_known(v___x_2691_, 1);
v___f_2693_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__15));
lean_inc(v___x_2652_);
v___f_2694_ = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(v___f_2694_, 0, v___f_2693_);
lean_closure_set(v___f_2694_, 1, v___x_2652_);
lean_inc(v___x_2653_);
v___f_2695_ = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(v___f_2695_, 0, v___f_2694_);
lean_closure_set(v___f_2695_, 1, v___x_2653_);
lean_inc(v___f_2654_);
v___f_2696_ = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(v___f_2696_, 0, v___f_2695_);
lean_closure_set(v___f_2696_, 1, v___f_2654_);
lean_inc_ref(v___f_2655_);
v___f_2697_ = lean_alloc_closure((void*)(l_instMonadLiftTOfMonadLift___redArg___lam__0), 4, 2);
lean_closure_set(v___f_2697_, 0, v___f_2696_);
lean_closure_set(v___f_2697_, 1, v___f_2655_);
v___x_2738_ = lean_unbox(v_a_2692_);
lean_dec(v_a_2692_);
if (v___x_2738_ == 0)
{
v___y_2699_ = v___y_2662_;
v___y_2700_ = v___y_2663_;
v___y_2701_ = v___y_2664_;
v___y_2702_ = v___y_2665_;
v___y_2703_ = v___y_2666_;
v___y_2704_ = v___y_2667_;
v___y_2705_ = v___y_2668_;
v___y_2706_ = v___y_2669_;
goto v___jp_2698_;
}
else
{
lean_object* v___x_2739_; lean_object* v___x_2740_; 
v___x_2739_ = lean_st_ref_get(v___y_2663_);
lean_dec(v___x_2739_);
lean_inc(v_a_2684_);
lean_inc(v_a_2687_);
v___x_2740_ = lp_aesop_Aesop_Script_UScript_renderTacticSeq(v_fst_2671_, v_a_2687_, v_a_2684_, v___y_2666_, v___y_2667_, v___y_2668_, v___y_2669_);
if (lean_obj_tag(v___x_2740_) == 0)
{
lean_object* v_a_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; lean_object* v___x_2744_; lean_object* v___x_2745_; lean_object* v___x_2746_; lean_object* v___x_2747_; lean_object* v___x_2748_; lean_object* v___x_2749_; lean_object* v_traceClass_2750_; lean_object* v___x_2751_; lean_object* v___f_2752_; lean_object* v___f_2753_; lean_object* v___f_2754_; lean_object* v___f_2755_; lean_object* v___x_2756_; lean_object* v___x_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; lean_object* v___x_47073__overap_2760_; lean_object* v___x_2761_; 
v_a_2741_ = lean_ctor_get(v___x_2740_, 0);
lean_inc(v_a_2741_);
lean_dec_ref_known(v___x_2740_, 1);
v___x_2742_ = l_Lean_Core_instMonadTraceCoreM;
lean_inc(v___x_2656_);
v___x_2743_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_2656_, v___x_2742_);
lean_inc(v___f_2657_);
v___x_2744_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_2657_, v___x_2743_);
lean_inc_n(v___x_2652_, 2);
v___x_2745_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_2652_, v___x_2744_);
lean_inc_n(v___x_2653_, 2);
v___x_2746_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_2653_, v___x_2745_);
lean_inc_n(v___f_2654_, 2);
v___x_2747_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_2654_, v___x_2746_);
lean_inc_ref_n(v___f_2655_, 2);
v___x_2748_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_2655_, v___x_2747_);
v___x_2749_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_2658_);
v_traceClass_2750_ = lean_ctor_get(v___x_2689_, 0);
v___x_2751_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_2752_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2752_, 0, v___x_2751_);
lean_closure_set(v___f_2752_, 1, v___x_2652_);
v___f_2753_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2753_, 0, v___f_2752_);
lean_closure_set(v___f_2753_, 1, v___x_2653_);
v___f_2754_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2754_, 0, v___f_2753_);
lean_closure_set(v___f_2754_, 1, v___f_2654_);
v___f_2755_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2755_, 0, v___f_2754_);
lean_closure_set(v___f_2755_, 1, v___f_2655_);
v___x_2756_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2, &lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2_once, _init_lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2);
v___x_2757_ = l_Lean_MessageData_ofSyntax(v_a_2741_);
v___x_2758_ = l_Lean_indentD(v___x_2757_);
v___x_2759_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2759_, 0, v___x_2756_);
lean_ctor_set(v___x_2759_, 1, v___x_2758_);
lean_inc(v_traceClass_2750_);
lean_inc_ref(v___x_2651_);
v___x_47073__overap_2760_ = l_Lean_addTrace___redArg(v___x_2651_, v___x_2748_, v___x_2749_, v___f_2755_, v_traceClass_2750_, v___x_2759_);
lean_inc(v___y_2669_);
lean_inc_ref(v___y_2668_);
lean_inc(v___y_2667_);
lean_inc_ref(v___y_2666_);
lean_inc(v___y_2665_);
lean_inc(v___y_2664_);
lean_inc(v___y_2663_);
lean_inc_ref(v___y_2662_);
v___x_2761_ = lean_apply_9(v___x_47073__overap_2760_, v___y_2662_, v___y_2663_, v___y_2664_, v___y_2665_, v___y_2666_, v___y_2667_, v___y_2668_, v___y_2669_, lean_box(0));
if (lean_obj_tag(v___x_2761_) == 0)
{
lean_dec_ref_known(v___x_2761_, 1);
v___y_2699_ = v___y_2662_;
v___y_2700_ = v___y_2663_;
v___y_2701_ = v___y_2664_;
v___y_2702_ = v___y_2665_;
v___y_2703_ = v___y_2666_;
v___y_2704_ = v___y_2667_;
v___y_2705_ = v___y_2668_;
v___y_2706_ = v___y_2669_;
goto v___jp_2698_;
}
else
{
lean_dec_ref(v___f_2697_);
lean_dec(v_a_2687_);
lean_dec(v_a_2684_);
lean_dec(v_snd_2672_);
lean_dec(v_fst_2671_);
lean_dec_ref(v_options_2659_);
lean_dec(v___f_2657_);
lean_dec(v___x_2656_);
lean_dec_ref(v___f_2655_);
lean_dec(v___f_2654_);
lean_dec(v___x_2653_);
lean_dec(v___x_2652_);
lean_dec_ref(v___x_2651_);
lean_dec_ref(v___x_2650_);
return v___x_2761_;
}
}
else
{
lean_object* v_a_2762_; lean_object* v___x_2764_; uint8_t v_isShared_2765_; uint8_t v_isSharedCheck_2769_; 
lean_dec_ref(v___f_2697_);
lean_dec(v_a_2687_);
lean_dec(v_a_2684_);
lean_dec(v_snd_2672_);
lean_dec(v_fst_2671_);
lean_dec_ref(v_options_2659_);
lean_dec(v___f_2657_);
lean_dec(v___x_2656_);
lean_dec_ref(v___f_2655_);
lean_dec(v___f_2654_);
lean_dec(v___x_2653_);
lean_dec(v___x_2652_);
lean_dec_ref(v___x_2651_);
lean_dec_ref(v___x_2650_);
v_a_2762_ = lean_ctor_get(v___x_2740_, 0);
v_isSharedCheck_2769_ = !lean_is_exclusive(v___x_2740_);
if (v_isSharedCheck_2769_ == 0)
{
v___x_2764_ = v___x_2740_;
v_isShared_2765_ = v_isSharedCheck_2769_;
goto v_resetjp_2763_;
}
else
{
lean_inc(v_a_2762_);
lean_dec(v___x_2740_);
v___x_2764_ = lean_box(0);
v_isShared_2765_ = v_isSharedCheck_2769_;
goto v_resetjp_2763_;
}
v_resetjp_2763_:
{
lean_object* v___x_2767_; 
if (v_isShared_2765_ == 0)
{
v___x_2767_ = v___x_2764_;
goto v_reusejp_2766_;
}
else
{
lean_object* v_reuseFailAlloc_2768_; 
v_reuseFailAlloc_2768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2768_, 0, v_a_2762_);
v___x_2767_ = v_reuseFailAlloc_2768_;
goto v_reusejp_2766_;
}
v_reusejp_2766_:
{
return v___x_2767_;
}
}
}
}
v___jp_2698_:
{
lean_object* v___x_2707_; uint8_t v___x_2708_; lean_object* v___x_2709_; 
v___x_2707_ = lean_st_ref_get(v___y_2700_);
lean_dec(v___x_2707_);
v___x_2708_ = lean_unbox(v_snd_2672_);
lean_dec(v_snd_2672_);
lean_inc(v_a_2684_);
lean_inc(v_a_2687_);
lean_inc(v_fst_2671_);
v___x_2709_ = lp_aesop_Aesop_Script_UScript_optimize(v_fst_2671_, v___x_2708_, v_a_2687_, v_a_2684_, v___y_2703_, v___y_2704_, v___y_2705_, v___y_2706_);
if (lean_obj_tag(v___x_2709_) == 0)
{
lean_object* v_a_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; lean_object* v___x_2720_; lean_object* v___f_2721_; lean_object* v___f_2722_; lean_object* v___f_2723_; lean_object* v___f_2724_; lean_object* v___x_2725_; lean_object* v___x_2726_; lean_object* v___x_2727_; lean_object* v___x_47045__overap_2728_; lean_object* v___x_2729_; 
v_a_2710_ = lean_ctor_get(v___x_2709_, 0);
lean_inc(v_a_2710_);
lean_dec_ref_known(v___x_2709_, 1);
v___x_2711_ = l_Lean_Core_instMonadLogCoreM;
v___x_2712_ = l_Lean_instMonadLogOfMonadLift___redArg(v___x_2656_, v___x_2711_);
v___x_2713_ = l_Lean_instMonadLogOfMonadLift___redArg(v___f_2657_, v___x_2712_);
lean_inc(v___x_2652_);
v___x_2714_ = l_Lean_instMonadLogOfMonadLift___redArg(v___x_2652_, v___x_2713_);
lean_inc(v___x_2653_);
v___x_2715_ = l_Lean_instMonadLogOfMonadLift___redArg(v___x_2653_, v___x_2714_);
lean_inc(v___f_2654_);
v___x_2716_ = l_Lean_instMonadLogOfMonadLift___redArg(v___f_2654_, v___x_2715_);
lean_inc_ref(v___f_2655_);
v___x_2717_ = l_Lean_instMonadLogOfMonadLift___redArg(v___f_2655_, v___x_2716_);
v___x_2718_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_2658_);
v___x_2719_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___x_2720_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_2721_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2721_, 0, v___x_2720_);
lean_closure_set(v___f_2721_, 1, v___x_2652_);
v___f_2722_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2722_, 0, v___f_2721_);
lean_closure_set(v___f_2722_, 1, v___x_2653_);
v___f_2723_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2723_, 0, v___f_2722_);
lean_closure_set(v___f_2723_, 1, v___f_2654_);
v___f_2724_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2724_, 0, v___f_2723_);
lean_closure_set(v___f_2724_, 1, v___f_2655_);
lean_inc_ref(v___x_2651_);
lean_inc_ref(v___f_2724_);
v___x_2725_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_2724_, v___x_2651_);
lean_inc_ref(v___x_2718_);
v___x_2726_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2726_, 0, v___x_2719_);
lean_ctor_set(v___x_2726_, 1, v___x_2718_);
lean_ctor_set(v___x_2726_, 2, v___x_2725_);
v___x_2727_ = ((lean_object*)(lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0));
v___x_47045__overap_2728_ = lp_aesop_Aesop_checkAndTraceScript___redArg(v___x_2651_, v___x_2717_, v___x_2718_, v___x_2726_, v___f_2724_, v___x_2650_, v___f_2697_, v_fst_2671_, v_a_2710_, v_a_2687_, v_a_2684_, v_options_2659_, v_completeProof_2660_, v___x_2727_);
lean_inc(v___y_2706_);
lean_inc_ref(v___y_2705_);
lean_inc(v___y_2704_);
lean_inc_ref(v___y_2703_);
lean_inc(v___y_2702_);
lean_inc(v___y_2701_);
lean_inc(v___y_2700_);
lean_inc_ref(v___y_2699_);
v___x_2729_ = lean_apply_9(v___x_47045__overap_2728_, v___y_2699_, v___y_2700_, v___y_2701_, v___y_2702_, v___y_2703_, v___y_2704_, v___y_2705_, v___y_2706_, lean_box(0));
return v___x_2729_;
}
else
{
lean_object* v_a_2730_; lean_object* v___x_2732_; uint8_t v_isShared_2733_; uint8_t v_isSharedCheck_2737_; 
lean_dec_ref(v___f_2697_);
lean_dec(v_a_2687_);
lean_dec(v_a_2684_);
lean_dec(v_fst_2671_);
lean_dec_ref(v_options_2659_);
lean_dec(v___f_2657_);
lean_dec(v___x_2656_);
lean_dec_ref(v___f_2655_);
lean_dec(v___f_2654_);
lean_dec(v___x_2653_);
lean_dec(v___x_2652_);
lean_dec_ref(v___x_2651_);
lean_dec_ref(v___x_2650_);
v_a_2730_ = lean_ctor_get(v___x_2709_, 0);
v_isSharedCheck_2737_ = !lean_is_exclusive(v___x_2709_);
if (v_isSharedCheck_2737_ == 0)
{
v___x_2732_ = v___x_2709_;
v_isShared_2733_ = v_isSharedCheck_2737_;
goto v_resetjp_2731_;
}
else
{
lean_inc(v_a_2730_);
lean_dec(v___x_2709_);
v___x_2732_ = lean_box(0);
v_isShared_2733_ = v_isSharedCheck_2737_;
goto v_resetjp_2731_;
}
v_resetjp_2731_:
{
lean_object* v___x_2735_; 
if (v_isShared_2733_ == 0)
{
v___x_2735_ = v___x_2732_;
goto v_reusejp_2734_;
}
else
{
lean_object* v_reuseFailAlloc_2736_; 
v_reuseFailAlloc_2736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2736_, 0, v_a_2730_);
v___x_2735_ = v_reuseFailAlloc_2736_;
goto v_reusejp_2734_;
}
v_reusejp_2734_:
{
return v___x_2735_;
}
}
}
}
}
else
{
lean_object* v_a_2770_; lean_object* v___x_2772_; uint8_t v_isShared_2773_; uint8_t v_isSharedCheck_2777_; 
lean_dec(v_a_2687_);
lean_dec(v_a_2684_);
lean_dec(v_snd_2672_);
lean_dec(v_fst_2671_);
lean_dec_ref(v_options_2659_);
lean_dec(v___f_2657_);
lean_dec(v___x_2656_);
lean_dec_ref(v___f_2655_);
lean_dec(v___f_2654_);
lean_dec(v___x_2653_);
lean_dec(v___x_2652_);
lean_dec_ref(v___x_2651_);
lean_dec_ref(v___x_2650_);
v_a_2770_ = lean_ctor_get(v___x_2691_, 0);
v_isSharedCheck_2777_ = !lean_is_exclusive(v___x_2691_);
if (v_isSharedCheck_2777_ == 0)
{
v___x_2772_ = v___x_2691_;
v_isShared_2773_ = v_isSharedCheck_2777_;
goto v_resetjp_2771_;
}
else
{
lean_inc(v_a_2770_);
lean_dec(v___x_2691_);
v___x_2772_ = lean_box(0);
v_isShared_2773_ = v_isSharedCheck_2777_;
goto v_resetjp_2771_;
}
v_resetjp_2771_:
{
lean_object* v___x_2775_; 
if (v_isShared_2773_ == 0)
{
v___x_2775_ = v___x_2772_;
goto v_reusejp_2774_;
}
else
{
lean_object* v_reuseFailAlloc_2776_; 
v_reuseFailAlloc_2776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2776_, 0, v_a_2770_);
v___x_2775_ = v_reuseFailAlloc_2776_;
goto v_reusejp_2774_;
}
v_reusejp_2774_:
{
return v___x_2775_;
}
}
}
}
else
{
lean_object* v_a_2778_; lean_object* v___x_2780_; uint8_t v_isShared_2781_; uint8_t v_isSharedCheck_2785_; 
lean_dec(v_a_2684_);
lean_dec(v_snd_2672_);
lean_dec(v_fst_2671_);
lean_dec_ref(v_options_2659_);
lean_dec(v___f_2657_);
lean_dec(v___x_2656_);
lean_dec_ref(v___f_2655_);
lean_dec(v___f_2654_);
lean_dec(v___x_2653_);
lean_dec(v___x_2652_);
lean_dec_ref(v___x_2651_);
lean_dec_ref(v___x_2650_);
v_a_2778_ = lean_ctor_get(v___x_2686_, 0);
v_isSharedCheck_2785_ = !lean_is_exclusive(v___x_2686_);
if (v_isSharedCheck_2785_ == 0)
{
v___x_2780_ = v___x_2686_;
v_isShared_2781_ = v_isSharedCheck_2785_;
goto v_resetjp_2779_;
}
else
{
lean_inc(v_a_2778_);
lean_dec(v___x_2686_);
v___x_2780_ = lean_box(0);
v_isShared_2781_ = v_isSharedCheck_2785_;
goto v_resetjp_2779_;
}
v_resetjp_2779_:
{
lean_object* v___x_2783_; 
if (v_isShared_2781_ == 0)
{
v___x_2783_ = v___x_2780_;
goto v_reusejp_2782_;
}
else
{
lean_object* v_reuseFailAlloc_2784_; 
v_reuseFailAlloc_2784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2784_, 0, v_a_2778_);
v___x_2783_ = v_reuseFailAlloc_2784_;
goto v_reusejp_2782_;
}
v_reusejp_2782_:
{
return v___x_2783_;
}
}
}
}
else
{
lean_object* v_a_2786_; lean_object* v___x_2788_; uint8_t v_isShared_2789_; uint8_t v_isSharedCheck_2793_; 
lean_dec(v_snd_2672_);
lean_dec(v_fst_2671_);
lean_dec_ref(v_options_2659_);
lean_dec(v___f_2657_);
lean_dec(v___x_2656_);
lean_dec_ref(v___f_2655_);
lean_dec(v___f_2654_);
lean_dec(v___x_2653_);
lean_dec(v___x_2652_);
lean_dec_ref(v___x_2651_);
lean_dec_ref(v___x_2650_);
v_a_2786_ = lean_ctor_get(v___x_2683_, 0);
v_isSharedCheck_2793_ = !lean_is_exclusive(v___x_2683_);
if (v_isSharedCheck_2793_ == 0)
{
v___x_2788_ = v___x_2683_;
v_isShared_2789_ = v_isSharedCheck_2793_;
goto v_resetjp_2787_;
}
else
{
lean_inc(v_a_2786_);
lean_dec(v___x_2683_);
v___x_2788_ = lean_box(0);
v_isShared_2789_ = v_isSharedCheck_2793_;
goto v_resetjp_2787_;
}
v_resetjp_2787_:
{
lean_object* v___x_2791_; 
if (v_isShared_2789_ == 0)
{
v___x_2791_ = v___x_2788_;
goto v_reusejp_2790_;
}
else
{
lean_object* v_reuseFailAlloc_2792_; 
v_reuseFailAlloc_2792_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2792_, 0, v_a_2786_);
v___x_2791_ = v_reuseFailAlloc_2792_;
goto v_reusejp_2790_;
}
v_reusejp_2790_:
{
return v___x_2791_;
}
}
}
}
}
else
{
lean_del_object(v___x_2674_);
lean_dec(v_snd_2672_);
lean_dec(v_fst_2671_);
lean_dec_ref(v_options_2659_);
lean_dec(v___f_2657_);
lean_dec(v___x_2656_);
lean_dec_ref(v___f_2655_);
lean_dec(v___f_2654_);
lean_dec(v___x_2653_);
lean_dec(v___x_2652_);
lean_dec_ref(v___x_2651_);
lean_dec_ref(v___x_2650_);
return v___x_2677_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___lam__1___boxed(lean_object** _args){
lean_object* v___x_2796_ = _args[0];
lean_object* v___x_2797_ = _args[1];
lean_object* v___x_2798_ = _args[2];
lean_object* v___x_2799_ = _args[3];
lean_object* v___f_2800_ = _args[4];
lean_object* v___f_2801_ = _args[5];
lean_object* v___x_2802_ = _args[6];
lean_object* v___f_2803_ = _args[7];
lean_object* v_inst_2804_ = _args[8];
lean_object* v_options_2805_ = _args[9];
lean_object* v_completeProof_2806_ = _args[10];
lean_object* v_____x_2807_ = _args[11];
lean_object* v___y_2808_ = _args[12];
lean_object* v___y_2809_ = _args[13];
lean_object* v___y_2810_ = _args[14];
lean_object* v___y_2811_ = _args[15];
lean_object* v___y_2812_ = _args[16];
lean_object* v___y_2813_ = _args[17];
lean_object* v___y_2814_ = _args[18];
lean_object* v___y_2815_ = _args[19];
lean_object* v___y_2816_ = _args[20];
_start:
{
uint8_t v_completeProof_boxed_2817_; lean_object* v_res_2818_; 
v_completeProof_boxed_2817_ = lean_unbox(v_completeProof_2806_);
v_res_2818_ = lp_aesop_Aesop_traceScript___redArg___lam__1(v___x_2796_, v___x_2797_, v___x_2798_, v___x_2799_, v___f_2800_, v___f_2801_, v___x_2802_, v___f_2803_, v_inst_2804_, v_options_2805_, v_completeProof_boxed_2817_, v_____x_2807_, v___y_2808_, v___y_2809_, v___y_2810_, v___y_2811_, v___y_2812_, v___y_2813_, v___y_2814_, v___y_2815_);
lean_dec(v___y_2815_);
lean_dec_ref(v___y_2814_);
lean_dec(v___y_2813_);
lean_dec_ref(v___y_2812_);
lean_dec(v___y_2811_);
lean_dec(v___y_2810_);
lean_dec(v___y_2809_);
lean_dec_ref(v___y_2808_);
lean_dec_ref(v_inst_2804_);
return v_res_2818_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__1(void){
_start:
{
lean_object* v___x_2820_; lean_object* v___x_2821_; lean_object* v___x_2822_; lean_object* v___x_2823_; 
v___x_2820_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__5);
v___x_2821_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_2822_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3));
v___x_2823_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_2822_, v___x_2821_, v___x_2820_);
return v___x_2823_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__2(void){
_start:
{
lean_object* v___x_2824_; lean_object* v___x_2825_; lean_object* v___x_2826_; lean_object* v___x_2827_; 
v___x_2824_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__1, &lp_aesop_Aesop_traceScript___redArg___closed__1_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__1);
v___x_2825_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_2826_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__3));
v___x_2827_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_2826_, v___x_2825_, v___x_2824_);
return v___x_2827_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__3(void){
_start:
{
lean_object* v___x_2828_; lean_object* v___f_2829_; lean_object* v___f_2830_; lean_object* v___x_2831_; 
v___x_2828_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__2, &lp_aesop_Aesop_traceScript___redArg___closed__2_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__2);
v___f_2829_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___f_2830_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__2));
v___x_2831_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_2830_, v___f_2829_, v___x_2828_);
return v___x_2831_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__4(void){
_start:
{
lean_object* v___x_2832_; lean_object* v___x_2833_; 
v___x_2832_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__4, &lp_aesop_Aesop_expandNextGoal___redArg___closed__4_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__4);
v___x_2833_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_2832_);
return v___x_2833_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__5(void){
_start:
{
lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; 
v___x_2834_ = l_Lean_Core_instMonadLogCoreM;
v___x_2835_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_2836_ = l_Lean_instMonadLogOfMonadLift___redArg(v___x_2835_, v___x_2834_);
return v___x_2836_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__6(void){
_start:
{
lean_object* v___x_2837_; lean_object* v___f_2838_; lean_object* v___x_2839_; 
v___x_2837_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__5, &lp_aesop_Aesop_traceScript___redArg___closed__5_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__5);
v___f_2838_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___x_2839_ = l_Lean_instMonadLogOfMonadLift___redArg(v___f_2838_, v___x_2837_);
return v___x_2839_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__7(void){
_start:
{
lean_object* v___x_2840_; lean_object* v___x_2841_; lean_object* v___x_2842_; 
v___x_2840_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__6, &lp_aesop_Aesop_traceScript___redArg___closed__6_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__6);
v___x_2841_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_2842_ = l_Lean_instMonadLogOfMonadLift___redArg(v___x_2841_, v___x_2840_);
return v___x_2842_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__8(void){
_start:
{
lean_object* v___x_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; 
v___x_2843_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__7, &lp_aesop_Aesop_traceScript___redArg___closed__7_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__7);
v___x_2844_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___x_2845_ = l_Lean_instMonadLogOfMonadLift___redArg(v___x_2844_, v___x_2843_);
return v___x_2845_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__9(void){
_start:
{
lean_object* v___x_2846_; lean_object* v___f_2847_; lean_object* v___x_2848_; 
v___x_2846_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__8, &lp_aesop_Aesop_traceScript___redArg___closed__8_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__8);
v___f_2847_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___x_2848_ = l_Lean_instMonadLogOfMonadLift___redArg(v___f_2847_, v___x_2846_);
return v___x_2848_;
}
}
static lean_object* _init_lp_aesop_Aesop_traceScript___redArg___closed__10(void){
_start:
{
lean_object* v___x_2849_; lean_object* v___f_2850_; lean_object* v___x_2851_; 
v___x_2849_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__9, &lp_aesop_Aesop_traceScript___redArg___closed__9_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__9);
v___f_2850_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__17));
v___x_2851_ = l_Lean_instMonadLogOfMonadLift___redArg(v___f_2850_, v___x_2849_);
return v___x_2851_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg(lean_object* v_inst_2864_, uint8_t v_completeProof_2865_, lean_object* v_a_2866_, lean_object* v_a_2867_, lean_object* v_a_2868_, lean_object* v_a_2869_, lean_object* v_a_2870_, lean_object* v_a_2871_, lean_object* v_a_2872_, lean_object* v_a_2873_){
_start:
{
lean_object* v___y_2879_; lean_object* v_a_2880_; lean_object* v___y_2914_; lean_object* v___y_2915_; lean_object* v___y_2918_; lean_object* v___y_2919_; lean_object* v___y_2920_; lean_object* v___y_2932_; lean_object* v___y_2933_; lean_object* v_____do__lift_2934_; lean_object* v___y_2935_; lean_object* v___y_2936_; lean_object* v___y_2937_; lean_object* v___y_2938_; lean_object* v___y_2939_; lean_object* v___y_2940_; lean_object* v___y_2941_; lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v_toMonadOptions_2946_; lean_object* v___x_2947_; lean_object* v_options_2948_; lean_object* v_inheritedTraceOptions_2949_; lean_object* v___f_2950_; lean_object* v___y_2952_; lean_object* v___y_2953_; lean_object* v___y_2954_; lean_object* v___y_2955_; lean_object* v___y_2956_; lean_object* v___y_2957_; lean_object* v___y_2958_; uint8_t v___y_2959_; uint8_t v___y_2960_; lean_object* v___y_2961_; lean_object* v___y_2962_; lean_object* v___y_2963_; lean_object* v___y_2964_; lean_object* v___y_2965_; lean_object* v___y_2966_; lean_object* v___y_2967_; lean_object* v_a_2968_; lean_object* v___y_2982_; lean_object* v___y_2983_; lean_object* v___y_2984_; lean_object* v___y_2985_; lean_object* v___y_2986_; lean_object* v___y_2987_; lean_object* v___y_2988_; uint8_t v___y_2989_; uint8_t v___y_2990_; lean_object* v___y_2991_; lean_object* v___y_2992_; lean_object* v___y_2993_; lean_object* v___y_2994_; lean_object* v___y_2995_; lean_object* v___y_2996_; lean_object* v___y_2997_; lean_object* v_a_2998_; lean_object* v___y_3001_; lean_object* v___y_3002_; lean_object* v___y_3003_; lean_object* v___y_3004_; lean_object* v___y_3005_; lean_object* v___y_3006_; lean_object* v___y_3007_; uint8_t v___y_3008_; uint8_t v___y_3009_; lean_object* v___y_3010_; lean_object* v___y_3011_; lean_object* v___y_3012_; lean_object* v___y_3013_; lean_object* v___y_3014_; lean_object* v___y_3015_; lean_object* v___y_3016_; lean_object* v_a_3017_; lean_object* v___y_3028_; lean_object* v___y_3029_; lean_object* v___y_3030_; lean_object* v___y_3031_; lean_object* v___y_3032_; lean_object* v___y_3033_; lean_object* v___y_3034_; uint8_t v___y_3035_; uint8_t v___y_3036_; lean_object* v___y_3037_; lean_object* v___y_3038_; lean_object* v___y_3039_; lean_object* v___y_3040_; lean_object* v___y_3041_; lean_object* v___y_3042_; lean_object* v___y_3043_; lean_object* v_a_3044_; lean_object* v___y_3047_; lean_object* v___y_3048_; lean_object* v___y_3049_; lean_object* v___y_3050_; lean_object* v___y_3051_; uint8_t v___y_3052_; uint8_t v___y_3053_; lean_object* v___y_3054_; lean_object* v___y_3055_; lean_object* v___y_3056_; lean_object* v___y_3057_; lean_object* v___y_3058_; lean_object* v___y_3059_; lean_object* v___y_3060_; lean_object* v___f_3105_; lean_object* v___x_3106_; lean_object* v___f_3107_; uint8_t v___y_3172_; lean_object* v___y_3173_; lean_object* v___y_3174_; lean_object* v___y_3175_; lean_object* v___y_3176_; lean_object* v___y_3177_; lean_object* v___y_3178_; lean_object* v___y_3179_; lean_object* v___y_3180_; lean_object* v___y_3181_; lean_object* v___y_3182_; lean_object* v___y_3183_; lean_object* v___y_3184_; lean_object* v___y_3185_; lean_object* v___y_3207_; lean_object* v_____x_3208_; lean_object* v___y_3209_; lean_object* v___y_3210_; lean_object* v___y_3211_; lean_object* v___y_3212_; lean_object* v___y_3213_; lean_object* v___y_3214_; lean_object* v___y_3215_; lean_object* v___y_3216_; lean_object* v___y_3291_; lean_object* v___y_3292_; lean_object* v___y_3303_; lean_object* v_____do__lift_3304_; lean_object* v___y_3305_; lean_object* v___y_3306_; lean_object* v___y_3307_; lean_object* v___y_3308_; lean_object* v___y_3309_; lean_object* v___y_3310_; lean_object* v___y_3311_; lean_object* v___x_3314_; lean_object* v___x_3315_; uint8_t v___x_3316_; 
v___x_2944_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_2864_);
v___x_2945_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2);
v_toMonadOptions_2946_ = lean_ctor_get(v___x_2945_, 0);
v___x_2947_ = l_Lean_KVMap_instValueBool;
v_options_2948_ = lean_ctor_get(v_a_2872_, 2);
v_inheritedTraceOptions_2949_ = lean_ctor_get(v_a_2872_, 13);
v___f_2950_ = ((lean_object*)(lp_aesop_Aesop_traceScript___redArg___closed__0));
v___f_3105_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__0));
v___x_3106_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__1));
v___f_3107_ = ((lean_object*)(lp_aesop_Aesop_nextActiveGoal___redArg___closed__17));
v___x_3314_ = lp_aesop_Aesop_aesop_collectStats;
v___x_3315_ = l_Lean_Option_get___redArg(v___x_2947_, v_options_2948_, v___x_3314_);
v___x_3316_ = lean_unbox(v___x_3315_);
lean_dec(v___x_3315_);
if (v___x_3316_ == 0)
{
lean_object* v___y_3318_; lean_object* v___y_3319_; lean_object* v___y_3320_; uint8_t v___y_3321_; lean_object* v___y_3322_; lean_object* v___y_3323_; lean_object* v___y_3324_; lean_object* v___y_3325_; lean_object* v___y_3326_; uint8_t v___y_3327_; lean_object* v___y_3328_; lean_object* v___y_3329_; lean_object* v___y_3330_; lean_object* v___y_3331_; lean_object* v___y_3332_; lean_object* v_a_3333_; lean_object* v___y_3344_; lean_object* v___y_3345_; lean_object* v___y_3346_; uint8_t v___y_3347_; lean_object* v___y_3348_; lean_object* v___y_3349_; lean_object* v___y_3350_; lean_object* v___y_3351_; lean_object* v___y_3352_; uint8_t v___y_3353_; lean_object* v___y_3354_; lean_object* v___y_3355_; lean_object* v___y_3356_; lean_object* v___y_3357_; lean_object* v___y_3358_; lean_object* v_a_3359_; lean_object* v___y_3362_; lean_object* v___y_3363_; lean_object* v___y_3364_; uint8_t v___y_3365_; lean_object* v___y_3366_; lean_object* v___y_3367_; lean_object* v___y_3368_; lean_object* v___y_3369_; lean_object* v___y_3370_; uint8_t v___y_3371_; lean_object* v___y_3372_; lean_object* v___y_3373_; lean_object* v___y_3374_; lean_object* v___y_3375_; lean_object* v___y_3376_; lean_object* v_a_3377_; lean_object* v___y_3391_; lean_object* v___y_3392_; lean_object* v___y_3393_; uint8_t v___y_3394_; lean_object* v___y_3395_; lean_object* v___y_3396_; lean_object* v___y_3397_; lean_object* v___y_3398_; lean_object* v___y_3399_; uint8_t v___y_3400_; lean_object* v___y_3401_; lean_object* v___y_3402_; lean_object* v___y_3403_; lean_object* v___y_3404_; lean_object* v___y_3405_; lean_object* v_a_3406_; lean_object* v___y_3409_; lean_object* v___y_3410_; lean_object* v___y_3411_; uint8_t v___y_3412_; lean_object* v___y_3413_; lean_object* v___y_3414_; lean_object* v___y_3415_; lean_object* v___y_3416_; uint8_t v___y_3417_; lean_object* v___y_3418_; lean_object* v___y_3419_; lean_object* v___y_3420_; lean_object* v___y_3421_; uint8_t v_a_3467_; lean_object* v___y_3526_; lean_object* v___x_3538_; lean_object* v___x_45508__overap_3539_; lean_object* v___x_3540_; 
v___x_3538_ = lp_aesop_Aesop_TraceOption_stats;
lean_inc(v_toMonadOptions_2946_);
lean_inc_ref(v___x_2944_);
v___x_45508__overap_3539_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_2944_, v_toMonadOptions_2946_, v___x_3538_);
lean_inc(v_a_2873_);
lean_inc_ref(v_a_2872_);
lean_inc(v_a_2871_);
lean_inc_ref(v_a_2870_);
lean_inc(v_a_2869_);
lean_inc(v_a_2868_);
lean_inc(v_a_2867_);
lean_inc_ref(v_a_2866_);
v___x_3540_ = lean_apply_9(v___x_45508__overap_3539_, v_a_2866_, v_a_2867_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_, lean_box(0));
if (lean_obj_tag(v___x_3540_) == 0)
{
lean_object* v_a_3541_; uint8_t v___x_3542_; 
v_a_3541_ = lean_ctor_get(v___x_3540_, 0);
lean_inc(v_a_3541_);
v___x_3542_ = lean_unbox(v_a_3541_);
if (v___x_3542_ == 0)
{
lean_object* v___x_3543_; lean_object* v___x_3544_; lean_object* v___x_3545_; lean_object* v___x_3546_; uint8_t v___x_3547_; 
lean_dec_ref_known(v___x_3540_, 1);
v___x_3543_ = l_Lean_KVMap_instValueString;
v___x_3544_ = lp_aesop_Aesop_aesop_stats_file;
v___x_3545_ = l_Lean_Option_get___redArg(v___x_3543_, v_options_2948_, v___x_3544_);
v___x_3546_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0));
v___x_3547_ = lean_string_dec_eq(v___x_3545_, v___x_3546_);
lean_dec(v___x_3545_);
if (v___x_3547_ == 0)
{
lean_dec(v_a_3541_);
goto v___jp_3108_;
}
else
{
uint8_t v___x_3548_; 
v___x_3548_ = lean_unbox(v_a_3541_);
lean_dec(v_a_3541_);
v_a_3467_ = v___x_3548_;
goto v___jp_3466_;
}
}
else
{
lean_dec(v_a_3541_);
v___y_3526_ = v___x_3540_;
goto v___jp_3525_;
}
}
else
{
v___y_3526_ = v___x_3540_;
goto v___jp_3525_;
}
v___jp_3317_:
{
lean_object* v___x_3334_; double v___x_3335_; double v___x_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; lean_object* v___x_3339_; lean_object* v___x_3340_; lean_object* v___x_46250__overap_3341_; lean_object* v___x_3342_; 
v___x_3334_ = lean_io_get_num_heartbeats();
v___x_3335_ = lean_float_of_nat(v___y_3332_);
v___x_3336_ = lean_float_of_nat(v___x_3334_);
v___x_3337_ = lean_box_float(v___x_3335_);
v___x_3338_ = lean_box_float(v___x_3336_);
v___x_3339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3339_, 0, v___x_3337_);
lean_ctor_set(v___x_3339_, 1, v___x_3338_);
v___x_3340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3340_, 0, v_a_3333_);
lean_ctor_set(v___x_3340_, 1, v___x_3339_);
lean_inc_ref(v___y_3330_);
lean_inc_ref(v___y_3328_);
lean_inc_ref(v___y_3326_);
lean_inc(v___y_3325_);
lean_inc_ref(v___y_3331_);
lean_inc_ref(v___y_3329_);
lean_inc_ref(v___y_3322_);
v___x_46250__overap_3341_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___y_3322_, v___y_3329_, v___y_3331_, v___y_3325_, lean_box(0), v___y_3326_, v___y_3328_, v___y_3320_, v___y_3321_, v___y_3330_, v___y_3318_, v___y_3327_, v___y_3323_, v___f_2950_, v___x_3340_);
lean_inc(v_a_2873_);
lean_inc_ref(v_a_2872_);
lean_inc(v_a_2871_);
lean_inc_ref(v_a_2870_);
lean_inc(v_a_2869_);
lean_inc(v_a_2868_);
v___x_3342_ = lean_apply_8(v___x_46250__overap_3341_, v___y_3324_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_, lean_box(0));
v___y_3291_ = v___y_3319_;
v___y_3292_ = v___x_3342_;
goto v___jp_3290_;
}
v___jp_3343_:
{
lean_object* v___x_3360_; 
v___x_3360_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3360_, 0, v_a_3359_);
v___y_3318_ = v___y_3344_;
v___y_3319_ = v___y_3345_;
v___y_3320_ = v___y_3346_;
v___y_3321_ = v___y_3347_;
v___y_3322_ = v___y_3348_;
v___y_3323_ = v___y_3349_;
v___y_3324_ = v___y_3350_;
v___y_3325_ = v___y_3351_;
v___y_3326_ = v___y_3352_;
v___y_3327_ = v___y_3353_;
v___y_3328_ = v___y_3354_;
v___y_3329_ = v___y_3355_;
v___y_3330_ = v___y_3356_;
v___y_3331_ = v___y_3357_;
v___y_3332_ = v___y_3358_;
v_a_3333_ = v___x_3360_;
goto v___jp_3317_;
}
v___jp_3361_:
{
lean_object* v___x_3378_; double v___x_3379_; double v___x_3380_; double v___x_3381_; double v___x_3382_; double v___x_3383_; lean_object* v___x_3384_; lean_object* v___x_3385_; lean_object* v___x_3386_; lean_object* v___x_3387_; lean_object* v___x_46220__overap_3388_; lean_object* v___x_3389_; 
v___x_3378_ = lean_io_mono_nanos_now();
v___x_3379_ = lean_float_of_nat(v___y_3375_);
v___x_3380_ = lean_float_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2);
v___x_3381_ = lean_float_div(v___x_3379_, v___x_3380_);
v___x_3382_ = lean_float_of_nat(v___x_3378_);
v___x_3383_ = lean_float_div(v___x_3382_, v___x_3380_);
v___x_3384_ = lean_box_float(v___x_3381_);
v___x_3385_ = lean_box_float(v___x_3383_);
v___x_3386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3386_, 0, v___x_3384_);
lean_ctor_set(v___x_3386_, 1, v___x_3385_);
v___x_3387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3387_, 0, v_a_3377_);
lean_ctor_set(v___x_3387_, 1, v___x_3386_);
lean_inc_ref(v___y_3374_);
lean_inc_ref(v___y_3372_);
lean_inc_ref(v___y_3370_);
lean_inc(v___y_3369_);
lean_inc_ref(v___y_3376_);
lean_inc_ref(v___y_3373_);
lean_inc_ref(v___y_3366_);
v___x_46220__overap_3388_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___y_3366_, v___y_3373_, v___y_3376_, v___y_3369_, lean_box(0), v___y_3370_, v___y_3372_, v___y_3364_, v___y_3365_, v___y_3374_, v___y_3362_, v___y_3371_, v___y_3367_, v___f_2950_, v___x_3387_);
lean_inc(v_a_2873_);
lean_inc_ref(v_a_2872_);
lean_inc(v_a_2871_);
lean_inc_ref(v_a_2870_);
lean_inc(v_a_2869_);
lean_inc(v_a_2868_);
v___x_3389_ = lean_apply_8(v___x_46220__overap_3388_, v___y_3368_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_, lean_box(0));
v___y_3291_ = v___y_3363_;
v___y_3292_ = v___x_3389_;
goto v___jp_3290_;
}
v___jp_3390_:
{
lean_object* v___x_3407_; 
v___x_3407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3407_, 0, v_a_3406_);
v___y_3362_ = v___y_3391_;
v___y_3363_ = v___y_3392_;
v___y_3364_ = v___y_3393_;
v___y_3365_ = v___y_3394_;
v___y_3366_ = v___y_3395_;
v___y_3367_ = v___y_3396_;
v___y_3368_ = v___y_3397_;
v___y_3369_ = v___y_3398_;
v___y_3370_ = v___y_3399_;
v___y_3371_ = v___y_3400_;
v___y_3372_ = v___y_3401_;
v___y_3373_ = v___y_3402_;
v___y_3374_ = v___y_3403_;
v___y_3375_ = v___y_3405_;
v___y_3376_ = v___y_3404_;
v_a_3377_ = v___x_3407_;
goto v___jp_3361_;
}
v___jp_3408_:
{
lean_object* v___x_46197__overap_3422_; lean_object* v___x_3423_; 
lean_inc_ref(v___y_3419_);
lean_inc_ref(v___y_3413_);
v___x_46197__overap_3422_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___y_3413_, v___y_3419_);
lean_inc(v_a_2873_);
lean_inc_ref(v_a_2872_);
lean_inc(v_a_2871_);
lean_inc_ref(v_a_2870_);
lean_inc(v_a_2869_);
lean_inc(v_a_2868_);
lean_inc_ref(v___y_3414_);
v___x_3423_ = lean_apply_8(v___x_46197__overap_3422_, v___y_3414_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_, lean_box(0));
if (lean_obj_tag(v___x_3423_) == 0)
{
lean_object* v_a_3424_; lean_object* v___x_3425_; lean_object* v___x_3426_; uint8_t v___x_3427_; 
v_a_3424_ = lean_ctor_get(v___x_3423_, 0);
lean_inc(v_a_3424_);
lean_dec_ref_known(v___x_3423_, 1);
v___x_3425_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3426_ = l_Lean_Option_get___redArg(v___x_2947_, v___y_3409_, v___x_3425_);
v___x_3427_ = lean_unbox(v___x_3426_);
lean_dec(v___x_3426_);
if (v___x_3427_ == 0)
{
lean_object* v___x_3428_; lean_object* v___x_3429_; 
v___x_3428_ = lean_io_mono_nanos_now();
v___x_3429_ = lp_aesop_Aesop_getRootGoal(v___y_3414_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3429_) == 0)
{
lean_object* v_a_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; 
v_a_3430_ = lean_ctor_get(v___x_3429_, 0);
lean_inc(v_a_3430_);
lean_dec_ref_known(v___x_3429_, 1);
v___x_3431_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_3431_, 0, v_a_3430_);
v___x_3432_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_3431_, v___y_3414_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3432_) == 0)
{
lean_object* v_a_3433_; lean_object* v___x_3435_; uint8_t v_isShared_3436_; uint8_t v_isSharedCheck_3440_; 
v_a_3433_ = lean_ctor_get(v___x_3432_, 0);
v_isSharedCheck_3440_ = !lean_is_exclusive(v___x_3432_);
if (v_isSharedCheck_3440_ == 0)
{
v___x_3435_ = v___x_3432_;
v_isShared_3436_ = v_isSharedCheck_3440_;
goto v_resetjp_3434_;
}
else
{
lean_inc(v_a_3433_);
lean_dec(v___x_3432_);
v___x_3435_ = lean_box(0);
v_isShared_3436_ = v_isSharedCheck_3440_;
goto v_resetjp_3434_;
}
v_resetjp_3434_:
{
lean_object* v___x_3438_; 
if (v_isShared_3436_ == 0)
{
lean_ctor_set_tag(v___x_3435_, 1);
v___x_3438_ = v___x_3435_;
goto v_reusejp_3437_;
}
else
{
lean_object* v_reuseFailAlloc_3439_; 
v_reuseFailAlloc_3439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3439_, 0, v_a_3433_);
v___x_3438_ = v_reuseFailAlloc_3439_;
goto v_reusejp_3437_;
}
v_reusejp_3437_:
{
v___y_3362_ = v___y_3409_;
v___y_3363_ = v___y_3410_;
v___y_3364_ = v___y_3411_;
v___y_3365_ = v___y_3412_;
v___y_3366_ = v___y_3413_;
v___y_3367_ = v_a_3424_;
v___y_3368_ = v___y_3414_;
v___y_3369_ = v___y_3415_;
v___y_3370_ = v___y_3416_;
v___y_3371_ = v___y_3417_;
v___y_3372_ = v___y_3418_;
v___y_3373_ = v___y_3419_;
v___y_3374_ = v___y_3420_;
v___y_3375_ = v___x_3428_;
v___y_3376_ = v___y_3421_;
v_a_3377_ = v___x_3438_;
goto v___jp_3361_;
}
}
}
else
{
lean_object* v_a_3441_; 
v_a_3441_ = lean_ctor_get(v___x_3432_, 0);
lean_inc(v_a_3441_);
lean_dec_ref_known(v___x_3432_, 1);
v___y_3391_ = v___y_3409_;
v___y_3392_ = v___y_3410_;
v___y_3393_ = v___y_3411_;
v___y_3394_ = v___y_3412_;
v___y_3395_ = v___y_3413_;
v___y_3396_ = v_a_3424_;
v___y_3397_ = v___y_3414_;
v___y_3398_ = v___y_3415_;
v___y_3399_ = v___y_3416_;
v___y_3400_ = v___y_3417_;
v___y_3401_ = v___y_3418_;
v___y_3402_ = v___y_3419_;
v___y_3403_ = v___y_3420_;
v___y_3404_ = v___y_3421_;
v___y_3405_ = v___x_3428_;
v_a_3406_ = v_a_3441_;
goto v___jp_3390_;
}
}
else
{
lean_object* v_a_3442_; 
v_a_3442_ = lean_ctor_get(v___x_3429_, 0);
lean_inc(v_a_3442_);
lean_dec_ref_known(v___x_3429_, 1);
v___y_3391_ = v___y_3409_;
v___y_3392_ = v___y_3410_;
v___y_3393_ = v___y_3411_;
v___y_3394_ = v___y_3412_;
v___y_3395_ = v___y_3413_;
v___y_3396_ = v_a_3424_;
v___y_3397_ = v___y_3414_;
v___y_3398_ = v___y_3415_;
v___y_3399_ = v___y_3416_;
v___y_3400_ = v___y_3417_;
v___y_3401_ = v___y_3418_;
v___y_3402_ = v___y_3419_;
v___y_3403_ = v___y_3420_;
v___y_3404_ = v___y_3421_;
v___y_3405_ = v___x_3428_;
v_a_3406_ = v_a_3442_;
goto v___jp_3390_;
}
}
else
{
lean_object* v___x_3443_; lean_object* v___x_3444_; 
v___x_3443_ = lean_io_get_num_heartbeats();
v___x_3444_ = lp_aesop_Aesop_getRootGoal(v___y_3414_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3444_) == 0)
{
lean_object* v_a_3445_; lean_object* v___x_3446_; lean_object* v___x_3447_; 
v_a_3445_ = lean_ctor_get(v___x_3444_, 0);
lean_inc(v_a_3445_);
lean_dec_ref_known(v___x_3444_, 1);
v___x_3446_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_3446_, 0, v_a_3445_);
v___x_3447_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_3446_, v___y_3414_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3447_) == 0)
{
lean_object* v_a_3448_; lean_object* v___x_3450_; uint8_t v_isShared_3451_; uint8_t v_isSharedCheck_3455_; 
v_a_3448_ = lean_ctor_get(v___x_3447_, 0);
v_isSharedCheck_3455_ = !lean_is_exclusive(v___x_3447_);
if (v_isSharedCheck_3455_ == 0)
{
v___x_3450_ = v___x_3447_;
v_isShared_3451_ = v_isSharedCheck_3455_;
goto v_resetjp_3449_;
}
else
{
lean_inc(v_a_3448_);
lean_dec(v___x_3447_);
v___x_3450_ = lean_box(0);
v_isShared_3451_ = v_isSharedCheck_3455_;
goto v_resetjp_3449_;
}
v_resetjp_3449_:
{
lean_object* v___x_3453_; 
if (v_isShared_3451_ == 0)
{
lean_ctor_set_tag(v___x_3450_, 1);
v___x_3453_ = v___x_3450_;
goto v_reusejp_3452_;
}
else
{
lean_object* v_reuseFailAlloc_3454_; 
v_reuseFailAlloc_3454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3454_, 0, v_a_3448_);
v___x_3453_ = v_reuseFailAlloc_3454_;
goto v_reusejp_3452_;
}
v_reusejp_3452_:
{
v___y_3318_ = v___y_3409_;
v___y_3319_ = v___y_3410_;
v___y_3320_ = v___y_3411_;
v___y_3321_ = v___y_3412_;
v___y_3322_ = v___y_3413_;
v___y_3323_ = v_a_3424_;
v___y_3324_ = v___y_3414_;
v___y_3325_ = v___y_3415_;
v___y_3326_ = v___y_3416_;
v___y_3327_ = v___y_3417_;
v___y_3328_ = v___y_3418_;
v___y_3329_ = v___y_3419_;
v___y_3330_ = v___y_3420_;
v___y_3331_ = v___y_3421_;
v___y_3332_ = v___x_3443_;
v_a_3333_ = v___x_3453_;
goto v___jp_3317_;
}
}
}
else
{
lean_object* v_a_3456_; 
v_a_3456_ = lean_ctor_get(v___x_3447_, 0);
lean_inc(v_a_3456_);
lean_dec_ref_known(v___x_3447_, 1);
v___y_3344_ = v___y_3409_;
v___y_3345_ = v___y_3410_;
v___y_3346_ = v___y_3411_;
v___y_3347_ = v___y_3412_;
v___y_3348_ = v___y_3413_;
v___y_3349_ = v_a_3424_;
v___y_3350_ = v___y_3414_;
v___y_3351_ = v___y_3415_;
v___y_3352_ = v___y_3416_;
v___y_3353_ = v___y_3417_;
v___y_3354_ = v___y_3418_;
v___y_3355_ = v___y_3419_;
v___y_3356_ = v___y_3420_;
v___y_3357_ = v___y_3421_;
v___y_3358_ = v___x_3443_;
v_a_3359_ = v_a_3456_;
goto v___jp_3343_;
}
}
else
{
lean_object* v_a_3457_; 
v_a_3457_ = lean_ctor_get(v___x_3444_, 0);
lean_inc(v_a_3457_);
lean_dec_ref_known(v___x_3444_, 1);
v___y_3344_ = v___y_3409_;
v___y_3345_ = v___y_3410_;
v___y_3346_ = v___y_3411_;
v___y_3347_ = v___y_3412_;
v___y_3348_ = v___y_3413_;
v___y_3349_ = v_a_3424_;
v___y_3350_ = v___y_3414_;
v___y_3351_ = v___y_3415_;
v___y_3352_ = v___y_3416_;
v___y_3353_ = v___y_3417_;
v___y_3354_ = v___y_3418_;
v___y_3355_ = v___y_3419_;
v___y_3356_ = v___y_3420_;
v___y_3357_ = v___y_3421_;
v___y_3358_ = v___x_3443_;
v_a_3359_ = v_a_3457_;
goto v___jp_3343_;
}
}
}
else
{
lean_object* v_a_3458_; lean_object* v___x_3460_; uint8_t v_isShared_3461_; uint8_t v_isSharedCheck_3465_; 
lean_dec_ref(v___y_3414_);
lean_dec(v___y_3411_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3458_ = lean_ctor_get(v___x_3423_, 0);
v_isSharedCheck_3465_ = !lean_is_exclusive(v___x_3423_);
if (v_isSharedCheck_3465_ == 0)
{
v___x_3460_ = v___x_3423_;
v_isShared_3461_ = v_isSharedCheck_3465_;
goto v_resetjp_3459_;
}
else
{
lean_inc(v_a_3458_);
lean_dec(v___x_3423_);
v___x_3460_ = lean_box(0);
v_isShared_3461_ = v_isSharedCheck_3465_;
goto v_resetjp_3459_;
}
v_resetjp_3459_:
{
lean_object* v___x_3463_; 
if (v_isShared_3461_ == 0)
{
v___x_3463_ = v___x_3460_;
goto v_reusejp_3462_;
}
else
{
lean_object* v_reuseFailAlloc_3464_; 
v_reuseFailAlloc_3464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3464_, 0, v_a_3458_);
v___x_3463_ = v_reuseFailAlloc_3464_;
goto v_reusejp_3462_;
}
v_reusejp_3462_:
{
return v___x_3463_;
}
}
}
}
v___jp_3466_:
{
lean_object* v_options_3468_; uint8_t v_generateScript_3469_; 
v_options_3468_ = lean_ctor_get(v_a_2866_, 2);
v_generateScript_3469_ = lean_ctor_get_uint8(v_options_3468_, sizeof(void*)*2);
if (v_generateScript_3469_ == 0)
{
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
goto v___jp_2875_;
}
else
{
if (v_a_3467_ == 0)
{
if (v_completeProof_2865_ == 0)
{
lean_object* v_ruleSet_3470_; lean_object* v___x_3471_; lean_object* v_iteration_3472_; lean_object* v___x_3473_; lean_object* v___x_3474_; 
v_ruleSet_3470_ = lean_ctor_get(v_a_2866_, 0);
v___x_3471_ = lean_st_ref_get(v_a_2867_);
v_iteration_3472_ = lean_ctor_get(v___x_3471_, 0);
lean_inc(v_iteration_3472_);
lean_dec(v___x_3471_);
lean_inc_ref(v_ruleSet_3470_);
v___x_3473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3473_, 0, v_iteration_3472_);
lean_ctor_set(v___x_3473_, 1, v_ruleSet_3470_);
v___x_3474_ = lp_aesop_Aesop_extractSafePrefixScript(v___x_3473_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
lean_dec_ref_known(v___x_3473_, 2);
if (lean_obj_tag(v___x_3474_) == 0)
{
lean_object* v_a_3475_; 
v_a_3475_ = lean_ctor_get(v___x_3474_, 0);
lean_inc(v_a_3475_);
lean_dec_ref_known(v___x_3474_, 1);
v___y_3207_ = v_options_3468_;
v_____x_3208_ = v_a_3475_;
v___y_3209_ = v_a_2866_;
v___y_3210_ = v_a_2867_;
v___y_3211_ = v_a_2868_;
v___y_3212_ = v_a_2869_;
v___y_3213_ = v_a_2870_;
v___y_3214_ = v_a_2871_;
v___y_3215_ = v_a_2872_;
v___y_3216_ = v_a_2873_;
goto v___jp_3206_;
}
else
{
lean_object* v_a_3476_; lean_object* v___x_3478_; uint8_t v_isShared_3479_; uint8_t v_isSharedCheck_3483_; 
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3476_ = lean_ctor_get(v___x_3474_, 0);
v_isSharedCheck_3483_ = !lean_is_exclusive(v___x_3474_);
if (v_isSharedCheck_3483_ == 0)
{
v___x_3478_ = v___x_3474_;
v_isShared_3479_ = v_isSharedCheck_3483_;
goto v_resetjp_3477_;
}
else
{
lean_inc(v_a_3476_);
lean_dec(v___x_3474_);
v___x_3478_ = lean_box(0);
v_isShared_3479_ = v_isSharedCheck_3483_;
goto v_resetjp_3477_;
}
v_resetjp_3477_:
{
lean_object* v___x_3481_; 
if (v_isShared_3479_ == 0)
{
v___x_3481_ = v___x_3478_;
goto v_reusejp_3480_;
}
else
{
lean_object* v_reuseFailAlloc_3482_; 
v_reuseFailAlloc_3482_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3482_, 0, v_a_3476_);
v___x_3481_ = v_reuseFailAlloc_3482_;
goto v_reusejp_3480_;
}
v_reusejp_3480_:
{
return v___x_3481_;
}
}
}
}
else
{
lean_object* v_ruleSet_3484_; lean_object* v___x_3485_; lean_object* v_iteration_3486_; lean_object* v___x_3487_; lean_object* v___x_3488_; lean_object* v___x_3489_; lean_object* v_toMonadRef_3490_; lean_object* v___x_3491_; uint8_t v_hasTrace_3492_; lean_object* v___x_3493_; 
v_ruleSet_3484_ = lean_ctor_get(v_a_2866_, 0);
v___x_3485_ = lean_st_ref_get(v_a_2867_);
v_iteration_3486_ = lean_ctor_get(v___x_3485_, 0);
lean_inc(v_iteration_3486_);
lean_dec(v___x_3485_);
v___x_3487_ = lp_aesop_Aesop_TreeM_instMonad;
v___x_3488_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__2, &lp_aesop_Aesop_expandNextGoal___redArg___closed__2_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__2);
v___x_3489_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__3, &lp_aesop_Aesop_traceScript___redArg___closed__3_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__3);
v_toMonadRef_3490_ = lean_ctor_get(v___x_3489_, 0);
v___x_3491_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__4, &lp_aesop_Aesop_traceScript___redArg___closed__4_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__4);
v_hasTrace_3492_ = lean_ctor_get_uint8(v_options_2948_, sizeof(void*)*1);
lean_inc_ref(v_ruleSet_3484_);
v___x_3493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3493_, 0, v_iteration_3486_);
lean_ctor_set(v___x_3493_, 1, v_ruleSet_3484_);
if (v_hasTrace_3492_ == 0)
{
lean_object* v___x_3494_; 
v___x_3494_ = lp_aesop_Aesop_getRootGoal(v___x_3493_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3494_) == 0)
{
lean_object* v_a_3495_; 
v_a_3495_ = lean_ctor_get(v___x_3494_, 0);
lean_inc(v_a_3495_);
lean_dec_ref_known(v___x_3494_, 1);
v___y_3303_ = v_options_3468_;
v_____do__lift_3304_ = v_a_3495_;
v___y_3305_ = v___x_3493_;
v___y_3306_ = v_a_2868_;
v___y_3307_ = v_a_2869_;
v___y_3308_ = v_a_2870_;
v___y_3309_ = v_a_2871_;
v___y_3310_ = v_a_2872_;
v___y_3311_ = v_a_2873_;
goto v___jp_3302_;
}
else
{
lean_object* v_a_3496_; lean_object* v___x_3498_; uint8_t v_isShared_3499_; uint8_t v_isSharedCheck_3503_; 
lean_dec_ref_known(v___x_3493_, 2);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3496_ = lean_ctor_get(v___x_3494_, 0);
v_isSharedCheck_3503_ = !lean_is_exclusive(v___x_3494_);
if (v_isSharedCheck_3503_ == 0)
{
v___x_3498_ = v___x_3494_;
v_isShared_3499_ = v_isSharedCheck_3503_;
goto v_resetjp_3497_;
}
else
{
lean_inc(v_a_3496_);
lean_dec(v___x_3494_);
v___x_3498_ = lean_box(0);
v_isShared_3499_ = v_isSharedCheck_3503_;
goto v_resetjp_3497_;
}
v_resetjp_3497_:
{
lean_object* v___x_3501_; 
if (v_isShared_3499_ == 0)
{
v___x_3501_ = v___x_3498_;
goto v_reusejp_3500_;
}
else
{
lean_object* v_reuseFailAlloc_3502_; 
v_reuseFailAlloc_3502_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3502_, 0, v_a_3496_);
v___x_3501_ = v_reuseFailAlloc_3502_;
goto v_reusejp_3500_;
}
v_reusejp_3500_:
{
return v___x_3501_;
}
}
}
}
else
{
lean_object* v___x_3504_; lean_object* v_traceClass_3505_; lean_object* v___f_3506_; lean_object* v___f_3507_; lean_object* v___x_3508_; lean_object* v___x_3509_; lean_object* v___x_3510_; uint8_t v___x_3511_; 
v___x_3504_ = lp_aesop_Aesop_TraceOption_script;
v_traceClass_3505_ = lean_ctor_get(v___x_3504_, 0);
v___f_3506_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__16, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__16_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__16);
v___f_3507_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__11));
v___x_3508_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0));
v___x_3509_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__1));
lean_inc(v_traceClass_3505_);
v___x_3510_ = l_Lean_Name_append(v___x_3509_, v_traceClass_3505_);
v___x_3511_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2949_, v_options_2948_, v___x_3510_);
lean_dec(v___x_3510_);
if (v___x_3511_ == 0)
{
lean_object* v___x_3512_; lean_object* v___x_3513_; uint8_t v___x_3514_; 
v___x_3512_ = l_Lean_trace_profiler;
v___x_3513_ = l_Lean_Option_get___redArg(v___x_2947_, v_options_2948_, v___x_3512_);
v___x_3514_ = lean_unbox(v___x_3513_);
lean_dec(v___x_3513_);
if (v___x_3514_ == 0)
{
lean_object* v___x_3515_; 
v___x_3515_ = lp_aesop_Aesop_getRootGoal(v___x_3493_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3515_) == 0)
{
lean_object* v_a_3516_; 
v_a_3516_ = lean_ctor_get(v___x_3515_, 0);
lean_inc(v_a_3516_);
lean_dec_ref_known(v___x_3515_, 1);
v___y_3303_ = v_options_3468_;
v_____do__lift_3304_ = v_a_3516_;
v___y_3305_ = v___x_3493_;
v___y_3306_ = v_a_2868_;
v___y_3307_ = v_a_2869_;
v___y_3308_ = v_a_2870_;
v___y_3309_ = v_a_2871_;
v___y_3310_ = v_a_2872_;
v___y_3311_ = v_a_2873_;
goto v___jp_3302_;
}
else
{
lean_object* v_a_3517_; lean_object* v___x_3519_; uint8_t v_isShared_3520_; uint8_t v_isSharedCheck_3524_; 
lean_dec_ref_known(v___x_3493_, 2);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3517_ = lean_ctor_get(v___x_3515_, 0);
v_isSharedCheck_3524_ = !lean_is_exclusive(v___x_3515_);
if (v_isSharedCheck_3524_ == 0)
{
v___x_3519_ = v___x_3515_;
v_isShared_3520_ = v_isSharedCheck_3524_;
goto v_resetjp_3518_;
}
else
{
lean_inc(v_a_3517_);
lean_dec(v___x_3515_);
v___x_3519_ = lean_box(0);
v_isShared_3520_ = v_isSharedCheck_3524_;
goto v_resetjp_3518_;
}
v_resetjp_3518_:
{
lean_object* v___x_3522_; 
if (v_isShared_3520_ == 0)
{
v___x_3522_ = v___x_3519_;
goto v_reusejp_3521_;
}
else
{
lean_object* v_reuseFailAlloc_3523_; 
v_reuseFailAlloc_3523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3523_, 0, v_a_3517_);
v___x_3522_ = v_reuseFailAlloc_3523_;
goto v_reusejp_3521_;
}
v_reusejp_3521_:
{
return v___x_3522_;
}
}
}
}
else
{
lean_inc(v_traceClass_3505_);
v___y_3409_ = v_options_2948_;
v___y_3410_ = v_options_3468_;
v___y_3411_ = v_traceClass_3505_;
v___y_3412_ = v_hasTrace_3492_;
v___y_3413_ = v___x_3487_;
v___y_3414_ = v___x_3493_;
v___y_3415_ = v___f_3506_;
v___y_3416_ = v___x_3491_;
v___y_3417_ = v___x_3511_;
v___y_3418_ = v___f_3507_;
v___y_3419_ = v___x_3488_;
v___y_3420_ = v___x_3508_;
v___y_3421_ = v_toMonadRef_3490_;
goto v___jp_3408_;
}
}
else
{
lean_inc(v_traceClass_3505_);
v___y_3409_ = v_options_2948_;
v___y_3410_ = v_options_3468_;
v___y_3411_ = v_traceClass_3505_;
v___y_3412_ = v_hasTrace_3492_;
v___y_3413_ = v___x_3487_;
v___y_3414_ = v___x_3493_;
v___y_3415_ = v___f_3506_;
v___y_3416_ = v___x_3491_;
v___y_3417_ = v___x_3511_;
v___y_3418_ = v___f_3507_;
v___y_3419_ = v___x_3488_;
v___y_3420_ = v___x_3508_;
v___y_3421_ = v_toMonadRef_3490_;
goto v___jp_3408_;
}
}
}
}
else
{
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
goto v___jp_2875_;
}
}
}
v___jp_3525_:
{
if (lean_obj_tag(v___y_3526_) == 0)
{
lean_object* v_a_3527_; uint8_t v___x_3528_; 
v_a_3527_ = lean_ctor_get(v___y_3526_, 0);
lean_inc(v_a_3527_);
lean_dec_ref_known(v___y_3526_, 1);
v___x_3528_ = lean_unbox(v_a_3527_);
if (v___x_3528_ == 0)
{
uint8_t v___x_3529_; 
v___x_3529_ = lean_unbox(v_a_3527_);
lean_dec(v_a_3527_);
v_a_3467_ = v___x_3529_;
goto v___jp_3466_;
}
else
{
lean_dec(v_a_3527_);
goto v___jp_3108_;
}
}
else
{
lean_object* v_a_3530_; lean_object* v___x_3532_; uint8_t v_isShared_3533_; uint8_t v_isSharedCheck_3537_; 
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3530_ = lean_ctor_get(v___y_3526_, 0);
v_isSharedCheck_3537_ = !lean_is_exclusive(v___y_3526_);
if (v_isSharedCheck_3537_ == 0)
{
v___x_3532_ = v___y_3526_;
v_isShared_3533_ = v_isSharedCheck_3537_;
goto v_resetjp_3531_;
}
else
{
lean_inc(v_a_3530_);
lean_dec(v___y_3526_);
v___x_3532_ = lean_box(0);
v_isShared_3533_ = v_isSharedCheck_3537_;
goto v_resetjp_3531_;
}
v_resetjp_3531_:
{
lean_object* v___x_3535_; 
if (v_isShared_3533_ == 0)
{
v___x_3535_ = v___x_3532_;
goto v_reusejp_3534_;
}
else
{
lean_object* v_reuseFailAlloc_3536_; 
v_reuseFailAlloc_3536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3536_, 0, v_a_3530_);
v___x_3535_ = v_reuseFailAlloc_3536_;
goto v_reusejp_3534_;
}
v_reusejp_3534_:
{
return v___x_3535_;
}
}
}
}
}
else
{
goto v___jp_3108_;
}
v___jp_2875_:
{
lean_object* v___x_2876_; lean_object* v___x_2877_; 
v___x_2876_ = lean_box(0);
v___x_2877_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2877_, 0, v___x_2876_);
return v___x_2877_;
}
v___jp_2878_:
{
lean_object* v___x_2881_; lean_object* v___x_2882_; lean_object* v___x_2883_; lean_object* v_stats_2884_; lean_object* v_rulePatternCache_2885_; lean_object* v___x_2887_; uint8_t v_isShared_2888_; uint8_t v_isSharedCheck_2912_; 
v___x_2881_ = lean_st_ref_get(v_a_2867_);
lean_dec(v___x_2881_);
v___x_2882_ = lean_io_mono_nanos_now();
v___x_2883_ = lean_st_ref_take(v_a_2869_);
v_stats_2884_ = lean_ctor_get(v___x_2883_, 1);
v_rulePatternCache_2885_ = lean_ctor_get(v___x_2883_, 0);
v_isSharedCheck_2912_ = !lean_is_exclusive(v___x_2883_);
if (v_isSharedCheck_2912_ == 0)
{
v___x_2887_ = v___x_2883_;
v_isShared_2888_ = v_isSharedCheck_2912_;
goto v_resetjp_2886_;
}
else
{
lean_inc(v_stats_2884_);
lean_inc(v_rulePatternCache_2885_);
lean_dec(v___x_2883_);
v___x_2887_ = lean_box(0);
v_isShared_2888_ = v_isSharedCheck_2912_;
goto v_resetjp_2886_;
}
v_resetjp_2886_:
{
lean_object* v_total_2889_; lean_object* v_configParsing_2890_; lean_object* v_ruleSetConstruction_2891_; lean_object* v_search_2892_; lean_object* v_ruleSelection_2893_; lean_object* v_forwardState_2894_; lean_object* v_scriptGenerated_2895_; lean_object* v_ruleStats_2896_; lean_object* v_goalStats_2897_; lean_object* v___x_2899_; uint8_t v_isShared_2900_; uint8_t v_isSharedCheck_2910_; 
v_total_2889_ = lean_ctor_get(v_stats_2884_, 0);
v_configParsing_2890_ = lean_ctor_get(v_stats_2884_, 1);
v_ruleSetConstruction_2891_ = lean_ctor_get(v_stats_2884_, 2);
v_search_2892_ = lean_ctor_get(v_stats_2884_, 3);
v_ruleSelection_2893_ = lean_ctor_get(v_stats_2884_, 4);
v_forwardState_2894_ = lean_ctor_get(v_stats_2884_, 6);
v_scriptGenerated_2895_ = lean_ctor_get(v_stats_2884_, 7);
v_ruleStats_2896_ = lean_ctor_get(v_stats_2884_, 8);
v_goalStats_2897_ = lean_ctor_get(v_stats_2884_, 9);
v_isSharedCheck_2910_ = !lean_is_exclusive(v_stats_2884_);
if (v_isSharedCheck_2910_ == 0)
{
lean_object* v_unused_2911_; 
v_unused_2911_ = lean_ctor_get(v_stats_2884_, 5);
lean_dec(v_unused_2911_);
v___x_2899_ = v_stats_2884_;
v_isShared_2900_ = v_isSharedCheck_2910_;
goto v_resetjp_2898_;
}
else
{
lean_inc(v_goalStats_2897_);
lean_inc(v_ruleStats_2896_);
lean_inc(v_scriptGenerated_2895_);
lean_inc(v_forwardState_2894_);
lean_inc(v_ruleSelection_2893_);
lean_inc(v_search_2892_);
lean_inc(v_ruleSetConstruction_2891_);
lean_inc(v_configParsing_2890_);
lean_inc(v_total_2889_);
lean_dec(v_stats_2884_);
v___x_2899_ = lean_box(0);
v_isShared_2900_ = v_isSharedCheck_2910_;
goto v_resetjp_2898_;
}
v_resetjp_2898_:
{
lean_object* v___x_2901_; lean_object* v___x_2903_; 
v___x_2901_ = lean_nat_sub(v___x_2882_, v___y_2879_);
lean_dec(v___y_2879_);
lean_dec(v___x_2882_);
if (v_isShared_2900_ == 0)
{
lean_ctor_set(v___x_2899_, 5, v___x_2901_);
v___x_2903_ = v___x_2899_;
goto v_reusejp_2902_;
}
else
{
lean_object* v_reuseFailAlloc_2909_; 
v_reuseFailAlloc_2909_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2909_, 0, v_total_2889_);
lean_ctor_set(v_reuseFailAlloc_2909_, 1, v_configParsing_2890_);
lean_ctor_set(v_reuseFailAlloc_2909_, 2, v_ruleSetConstruction_2891_);
lean_ctor_set(v_reuseFailAlloc_2909_, 3, v_search_2892_);
lean_ctor_set(v_reuseFailAlloc_2909_, 4, v_ruleSelection_2893_);
lean_ctor_set(v_reuseFailAlloc_2909_, 5, v___x_2901_);
lean_ctor_set(v_reuseFailAlloc_2909_, 6, v_forwardState_2894_);
lean_ctor_set(v_reuseFailAlloc_2909_, 7, v_scriptGenerated_2895_);
lean_ctor_set(v_reuseFailAlloc_2909_, 8, v_ruleStats_2896_);
lean_ctor_set(v_reuseFailAlloc_2909_, 9, v_goalStats_2897_);
v___x_2903_ = v_reuseFailAlloc_2909_;
goto v_reusejp_2902_;
}
v_reusejp_2902_:
{
lean_object* v___x_2905_; 
if (v_isShared_2888_ == 0)
{
lean_ctor_set(v___x_2887_, 1, v___x_2903_);
v___x_2905_ = v___x_2887_;
goto v_reusejp_2904_;
}
else
{
lean_object* v_reuseFailAlloc_2908_; 
v_reuseFailAlloc_2908_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2908_, 0, v_rulePatternCache_2885_);
lean_ctor_set(v_reuseFailAlloc_2908_, 1, v___x_2903_);
v___x_2905_ = v_reuseFailAlloc_2908_;
goto v_reusejp_2904_;
}
v_reusejp_2904_:
{
lean_object* v___x_2906_; lean_object* v___x_2907_; 
v___x_2906_ = lean_st_ref_set(v_a_2869_, v___x_2905_);
v___x_2907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2907_, 0, v_a_2880_);
return v___x_2907_;
}
}
}
}
}
v___jp_2913_:
{
if (lean_obj_tag(v___y_2915_) == 0)
{
lean_object* v_a_2916_; 
v_a_2916_ = lean_ctor_get(v___y_2915_, 0);
lean_inc(v_a_2916_);
lean_dec_ref_known(v___y_2915_, 1);
v___y_2879_ = v___y_2914_;
v_a_2880_ = v_a_2916_;
goto v___jp_2878_;
}
else
{
lean_dec(v___y_2914_);
return v___y_2915_;
}
}
v___jp_2917_:
{
if (lean_obj_tag(v___y_2920_) == 0)
{
lean_object* v_a_2921_; lean_object* v___x_2922_; 
v_a_2921_ = lean_ctor_get(v___y_2920_, 0);
lean_inc(v_a_2921_);
lean_dec_ref_known(v___y_2920_, 1);
lean_inc(v_a_2873_);
lean_inc_ref(v_a_2872_);
lean_inc(v_a_2871_);
lean_inc_ref(v_a_2870_);
lean_inc(v_a_2869_);
lean_inc(v_a_2868_);
lean_inc(v_a_2867_);
lean_inc_ref(v_a_2866_);
v___x_2922_ = lean_apply_10(v___y_2919_, v_a_2921_, v_a_2866_, v_a_2867_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_, lean_box(0));
v___y_2914_ = v___y_2918_;
v___y_2915_ = v___x_2922_;
goto v___jp_2913_;
}
else
{
lean_object* v_a_2923_; lean_object* v___x_2925_; uint8_t v_isShared_2926_; uint8_t v_isSharedCheck_2930_; 
lean_dec_ref(v___y_2919_);
lean_dec(v___y_2918_);
v_a_2923_ = lean_ctor_get(v___y_2920_, 0);
v_isSharedCheck_2930_ = !lean_is_exclusive(v___y_2920_);
if (v_isSharedCheck_2930_ == 0)
{
v___x_2925_ = v___y_2920_;
v_isShared_2926_ = v_isSharedCheck_2930_;
goto v_resetjp_2924_;
}
else
{
lean_inc(v_a_2923_);
lean_dec(v___y_2920_);
v___x_2925_ = lean_box(0);
v_isShared_2926_ = v_isSharedCheck_2930_;
goto v_resetjp_2924_;
}
v_resetjp_2924_:
{
lean_object* v___x_2928_; 
if (v_isShared_2926_ == 0)
{
v___x_2928_ = v___x_2925_;
goto v_reusejp_2927_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v_a_2923_);
v___x_2928_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2927_;
}
v_reusejp_2927_:
{
return v___x_2928_;
}
}
}
}
v___jp_2931_:
{
lean_object* v___x_2942_; lean_object* v___x_2943_; 
v___x_2942_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_2942_, 0, v_____do__lift_2934_);
v___x_2943_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_2942_, v___y_2935_, v___y_2936_, v___y_2937_, v___y_2938_, v___y_2939_, v___y_2940_, v___y_2941_);
lean_dec_ref(v___y_2935_);
v___y_2918_ = v___y_2932_;
v___y_2919_ = v___y_2933_;
v___y_2920_ = v___x_2943_;
goto v___jp_2917_;
}
v___jp_2951_:
{
lean_object* v___x_2969_; double v___x_2970_; double v___x_2971_; double v___x_2972_; double v___x_2973_; double v___x_2974_; lean_object* v___x_2975_; lean_object* v___x_2976_; lean_object* v___x_2977_; lean_object* v___x_2978_; lean_object* v___x_46576__overap_2979_; lean_object* v___x_2980_; 
v___x_2969_ = lean_io_mono_nanos_now();
v___x_2970_ = lean_float_of_nat(v___y_2953_);
v___x_2971_ = lean_float_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__2);
v___x_2972_ = lean_float_div(v___x_2970_, v___x_2971_);
v___x_2973_ = lean_float_of_nat(v___x_2969_);
v___x_2974_ = lean_float_div(v___x_2973_, v___x_2971_);
v___x_2975_ = lean_box_float(v___x_2972_);
v___x_2976_ = lean_box_float(v___x_2974_);
v___x_2977_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2977_, 0, v___x_2975_);
lean_ctor_set(v___x_2977_, 1, v___x_2976_);
v___x_2978_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2978_, 0, v_a_2968_);
lean_ctor_set(v___x_2978_, 1, v___x_2977_);
lean_inc_ref(v___y_2954_);
lean_inc_ref(v___y_2966_);
lean_inc_ref(v___y_2956_);
lean_inc(v___y_2961_);
lean_inc_ref(v___y_2952_);
lean_inc_ref(v___y_2958_);
lean_inc_ref(v___y_2965_);
v___x_46576__overap_2979_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___y_2965_, v___y_2958_, v___y_2952_, v___y_2961_, lean_box(0), v___y_2956_, v___y_2966_, v___y_2962_, v___y_2960_, v___y_2954_, v___y_2967_, v___y_2959_, v___y_2957_, v___f_2950_, v___x_2978_);
lean_inc(v_a_2873_);
lean_inc_ref(v_a_2872_);
lean_inc(v_a_2871_);
lean_inc_ref(v_a_2870_);
lean_inc(v_a_2869_);
lean_inc(v_a_2868_);
v___x_2980_ = lean_apply_8(v___x_46576__overap_2979_, v___y_2955_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_, lean_box(0));
v___y_2918_ = v___y_2963_;
v___y_2919_ = v___y_2964_;
v___y_2920_ = v___x_2980_;
goto v___jp_2917_;
}
v___jp_2981_:
{
lean_object* v___x_2999_; 
v___x_2999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2999_, 0, v_a_2998_);
v___y_2952_ = v___y_2982_;
v___y_2953_ = v___y_2983_;
v___y_2954_ = v___y_2984_;
v___y_2955_ = v___y_2985_;
v___y_2956_ = v___y_2986_;
v___y_2957_ = v___y_2987_;
v___y_2958_ = v___y_2988_;
v___y_2959_ = v___y_2989_;
v___y_2960_ = v___y_2990_;
v___y_2961_ = v___y_2991_;
v___y_2962_ = v___y_2992_;
v___y_2963_ = v___y_2993_;
v___y_2964_ = v___y_2995_;
v___y_2965_ = v___y_2994_;
v___y_2966_ = v___y_2996_;
v___y_2967_ = v___y_2997_;
v_a_2968_ = v___x_2999_;
goto v___jp_2951_;
}
v___jp_3000_:
{
lean_object* v___x_3018_; double v___x_3019_; double v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; lean_object* v___x_3024_; lean_object* v___x_46606__overap_3025_; lean_object* v___x_3026_; 
v___x_3018_ = lean_io_get_num_heartbeats();
v___x_3019_ = lean_float_of_nat(v___y_3007_);
v___x_3020_ = lean_float_of_nat(v___x_3018_);
v___x_3021_ = lean_box_float(v___x_3019_);
v___x_3022_ = lean_box_float(v___x_3020_);
v___x_3023_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3023_, 0, v___x_3021_);
lean_ctor_set(v___x_3023_, 1, v___x_3022_);
v___x_3024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3024_, 0, v_a_3017_);
lean_ctor_set(v___x_3024_, 1, v___x_3023_);
lean_inc_ref(v___y_3002_);
lean_inc_ref(v___y_3015_);
lean_inc_ref(v___y_3004_);
lean_inc(v___y_3010_);
lean_inc_ref(v___y_3001_);
lean_inc_ref(v___y_3006_);
lean_inc_ref(v___y_3014_);
v___x_46606__overap_3025_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___y_3014_, v___y_3006_, v___y_3001_, v___y_3010_, lean_box(0), v___y_3004_, v___y_3015_, v___y_3011_, v___y_3009_, v___y_3002_, v___y_3016_, v___y_3008_, v___y_3005_, v___f_2950_, v___x_3024_);
lean_inc(v_a_2873_);
lean_inc_ref(v_a_2872_);
lean_inc(v_a_2871_);
lean_inc_ref(v_a_2870_);
lean_inc(v_a_2869_);
lean_inc(v_a_2868_);
v___x_3026_ = lean_apply_8(v___x_46606__overap_3025_, v___y_3003_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_, lean_box(0));
v___y_2918_ = v___y_3012_;
v___y_2919_ = v___y_3013_;
v___y_2920_ = v___x_3026_;
goto v___jp_2917_;
}
v___jp_3027_:
{
lean_object* v___x_3045_; 
v___x_3045_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3045_, 0, v_a_3044_);
v___y_3001_ = v___y_3028_;
v___y_3002_ = v___y_3029_;
v___y_3003_ = v___y_3030_;
v___y_3004_ = v___y_3031_;
v___y_3005_ = v___y_3032_;
v___y_3006_ = v___y_3033_;
v___y_3007_ = v___y_3034_;
v___y_3008_ = v___y_3035_;
v___y_3009_ = v___y_3036_;
v___y_3010_ = v___y_3037_;
v___y_3011_ = v___y_3038_;
v___y_3012_ = v___y_3039_;
v___y_3013_ = v___y_3041_;
v___y_3014_ = v___y_3040_;
v___y_3015_ = v___y_3042_;
v___y_3016_ = v___y_3043_;
v_a_3017_ = v___x_3045_;
goto v___jp_3000_;
}
v___jp_3046_:
{
lean_object* v___x_46553__overap_3061_; lean_object* v___x_3062_; 
lean_inc_ref(v___y_3051_);
lean_inc_ref(v___y_3058_);
v___x_46553__overap_3061_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___y_3058_, v___y_3051_);
lean_inc(v_a_2873_);
lean_inc_ref(v_a_2872_);
lean_inc(v_a_2871_);
lean_inc_ref(v_a_2870_);
lean_inc(v_a_2869_);
lean_inc(v_a_2868_);
lean_inc_ref(v___y_3049_);
v___x_3062_ = lean_apply_8(v___x_46553__overap_3061_, v___y_3049_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_, lean_box(0));
if (lean_obj_tag(v___x_3062_) == 0)
{
lean_object* v_a_3063_; lean_object* v___x_3064_; lean_object* v___x_3065_; uint8_t v___x_3066_; 
v_a_3063_ = lean_ctor_get(v___x_3062_, 0);
lean_inc(v_a_3063_);
lean_dec_ref_known(v___x_3062_, 1);
v___x_3064_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3065_ = l_Lean_Option_get___redArg(v___x_2947_, v___y_3060_, v___x_3064_);
v___x_3066_ = lean_unbox(v___x_3065_);
lean_dec(v___x_3065_);
if (v___x_3066_ == 0)
{
lean_object* v___x_3067_; lean_object* v___x_3068_; 
v___x_3067_ = lean_io_mono_nanos_now();
v___x_3068_ = lp_aesop_Aesop_getRootGoal(v___y_3049_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3068_) == 0)
{
lean_object* v_a_3069_; lean_object* v___x_3070_; lean_object* v___x_3071_; 
v_a_3069_ = lean_ctor_get(v___x_3068_, 0);
lean_inc(v_a_3069_);
lean_dec_ref_known(v___x_3068_, 1);
v___x_3070_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_3070_, 0, v_a_3069_);
v___x_3071_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_3070_, v___y_3049_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3071_) == 0)
{
lean_object* v_a_3072_; lean_object* v___x_3074_; uint8_t v_isShared_3075_; uint8_t v_isSharedCheck_3079_; 
v_a_3072_ = lean_ctor_get(v___x_3071_, 0);
v_isSharedCheck_3079_ = !lean_is_exclusive(v___x_3071_);
if (v_isSharedCheck_3079_ == 0)
{
v___x_3074_ = v___x_3071_;
v_isShared_3075_ = v_isSharedCheck_3079_;
goto v_resetjp_3073_;
}
else
{
lean_inc(v_a_3072_);
lean_dec(v___x_3071_);
v___x_3074_ = lean_box(0);
v_isShared_3075_ = v_isSharedCheck_3079_;
goto v_resetjp_3073_;
}
v_resetjp_3073_:
{
lean_object* v___x_3077_; 
if (v_isShared_3075_ == 0)
{
lean_ctor_set_tag(v___x_3074_, 1);
v___x_3077_ = v___x_3074_;
goto v_reusejp_3076_;
}
else
{
lean_object* v_reuseFailAlloc_3078_; 
v_reuseFailAlloc_3078_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3078_, 0, v_a_3072_);
v___x_3077_ = v_reuseFailAlloc_3078_;
goto v_reusejp_3076_;
}
v_reusejp_3076_:
{
v___y_2952_ = v___y_3047_;
v___y_2953_ = v___x_3067_;
v___y_2954_ = v___y_3048_;
v___y_2955_ = v___y_3049_;
v___y_2956_ = v___y_3050_;
v___y_2957_ = v_a_3063_;
v___y_2958_ = v___y_3051_;
v___y_2959_ = v___y_3052_;
v___y_2960_ = v___y_3053_;
v___y_2961_ = v___y_3054_;
v___y_2962_ = v___y_3055_;
v___y_2963_ = v___y_3056_;
v___y_2964_ = v___y_3057_;
v___y_2965_ = v___y_3058_;
v___y_2966_ = v___y_3059_;
v___y_2967_ = v___y_3060_;
v_a_2968_ = v___x_3077_;
goto v___jp_2951_;
}
}
}
else
{
lean_object* v_a_3080_; 
v_a_3080_ = lean_ctor_get(v___x_3071_, 0);
lean_inc(v_a_3080_);
lean_dec_ref_known(v___x_3071_, 1);
v___y_2982_ = v___y_3047_;
v___y_2983_ = v___x_3067_;
v___y_2984_ = v___y_3048_;
v___y_2985_ = v___y_3049_;
v___y_2986_ = v___y_3050_;
v___y_2987_ = v_a_3063_;
v___y_2988_ = v___y_3051_;
v___y_2989_ = v___y_3052_;
v___y_2990_ = v___y_3053_;
v___y_2991_ = v___y_3054_;
v___y_2992_ = v___y_3055_;
v___y_2993_ = v___y_3056_;
v___y_2994_ = v___y_3058_;
v___y_2995_ = v___y_3057_;
v___y_2996_ = v___y_3059_;
v___y_2997_ = v___y_3060_;
v_a_2998_ = v_a_3080_;
goto v___jp_2981_;
}
}
else
{
lean_object* v_a_3081_; 
v_a_3081_ = lean_ctor_get(v___x_3068_, 0);
lean_inc(v_a_3081_);
lean_dec_ref_known(v___x_3068_, 1);
v___y_2982_ = v___y_3047_;
v___y_2983_ = v___x_3067_;
v___y_2984_ = v___y_3048_;
v___y_2985_ = v___y_3049_;
v___y_2986_ = v___y_3050_;
v___y_2987_ = v_a_3063_;
v___y_2988_ = v___y_3051_;
v___y_2989_ = v___y_3052_;
v___y_2990_ = v___y_3053_;
v___y_2991_ = v___y_3054_;
v___y_2992_ = v___y_3055_;
v___y_2993_ = v___y_3056_;
v___y_2994_ = v___y_3058_;
v___y_2995_ = v___y_3057_;
v___y_2996_ = v___y_3059_;
v___y_2997_ = v___y_3060_;
v_a_2998_ = v_a_3081_;
goto v___jp_2981_;
}
}
else
{
lean_object* v___x_3082_; lean_object* v___x_3083_; 
v___x_3082_ = lean_io_get_num_heartbeats();
v___x_3083_ = lp_aesop_Aesop_getRootGoal(v___y_3049_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3083_) == 0)
{
lean_object* v_a_3084_; lean_object* v___x_3085_; lean_object* v___x_3086_; 
v_a_3084_ = lean_ctor_get(v___x_3083_, 0);
lean_inc(v_a_3084_);
lean_dec_ref_known(v___x_3083_, 1);
v___x_3085_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_3085_, 0, v_a_3084_);
v___x_3086_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_3085_, v___y_3049_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3086_) == 0)
{
lean_object* v_a_3087_; lean_object* v___x_3089_; uint8_t v_isShared_3090_; uint8_t v_isSharedCheck_3094_; 
v_a_3087_ = lean_ctor_get(v___x_3086_, 0);
v_isSharedCheck_3094_ = !lean_is_exclusive(v___x_3086_);
if (v_isSharedCheck_3094_ == 0)
{
v___x_3089_ = v___x_3086_;
v_isShared_3090_ = v_isSharedCheck_3094_;
goto v_resetjp_3088_;
}
else
{
lean_inc(v_a_3087_);
lean_dec(v___x_3086_);
v___x_3089_ = lean_box(0);
v_isShared_3090_ = v_isSharedCheck_3094_;
goto v_resetjp_3088_;
}
v_resetjp_3088_:
{
lean_object* v___x_3092_; 
if (v_isShared_3090_ == 0)
{
lean_ctor_set_tag(v___x_3089_, 1);
v___x_3092_ = v___x_3089_;
goto v_reusejp_3091_;
}
else
{
lean_object* v_reuseFailAlloc_3093_; 
v_reuseFailAlloc_3093_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3093_, 0, v_a_3087_);
v___x_3092_ = v_reuseFailAlloc_3093_;
goto v_reusejp_3091_;
}
v_reusejp_3091_:
{
v___y_3001_ = v___y_3047_;
v___y_3002_ = v___y_3048_;
v___y_3003_ = v___y_3049_;
v___y_3004_ = v___y_3050_;
v___y_3005_ = v_a_3063_;
v___y_3006_ = v___y_3051_;
v___y_3007_ = v___x_3082_;
v___y_3008_ = v___y_3052_;
v___y_3009_ = v___y_3053_;
v___y_3010_ = v___y_3054_;
v___y_3011_ = v___y_3055_;
v___y_3012_ = v___y_3056_;
v___y_3013_ = v___y_3057_;
v___y_3014_ = v___y_3058_;
v___y_3015_ = v___y_3059_;
v___y_3016_ = v___y_3060_;
v_a_3017_ = v___x_3092_;
goto v___jp_3000_;
}
}
}
else
{
lean_object* v_a_3095_; 
v_a_3095_ = lean_ctor_get(v___x_3086_, 0);
lean_inc(v_a_3095_);
lean_dec_ref_known(v___x_3086_, 1);
v___y_3028_ = v___y_3047_;
v___y_3029_ = v___y_3048_;
v___y_3030_ = v___y_3049_;
v___y_3031_ = v___y_3050_;
v___y_3032_ = v_a_3063_;
v___y_3033_ = v___y_3051_;
v___y_3034_ = v___x_3082_;
v___y_3035_ = v___y_3052_;
v___y_3036_ = v___y_3053_;
v___y_3037_ = v___y_3054_;
v___y_3038_ = v___y_3055_;
v___y_3039_ = v___y_3056_;
v___y_3040_ = v___y_3058_;
v___y_3041_ = v___y_3057_;
v___y_3042_ = v___y_3059_;
v___y_3043_ = v___y_3060_;
v_a_3044_ = v_a_3095_;
goto v___jp_3027_;
}
}
else
{
lean_object* v_a_3096_; 
v_a_3096_ = lean_ctor_get(v___x_3083_, 0);
lean_inc(v_a_3096_);
lean_dec_ref_known(v___x_3083_, 1);
v___y_3028_ = v___y_3047_;
v___y_3029_ = v___y_3048_;
v___y_3030_ = v___y_3049_;
v___y_3031_ = v___y_3050_;
v___y_3032_ = v_a_3063_;
v___y_3033_ = v___y_3051_;
v___y_3034_ = v___x_3082_;
v___y_3035_ = v___y_3052_;
v___y_3036_ = v___y_3053_;
v___y_3037_ = v___y_3054_;
v___y_3038_ = v___y_3055_;
v___y_3039_ = v___y_3056_;
v___y_3040_ = v___y_3058_;
v___y_3041_ = v___y_3057_;
v___y_3042_ = v___y_3059_;
v___y_3043_ = v___y_3060_;
v_a_3044_ = v_a_3096_;
goto v___jp_3027_;
}
}
}
else
{
lean_object* v_a_3097_; lean_object* v___x_3099_; uint8_t v_isShared_3100_; uint8_t v_isSharedCheck_3104_; 
lean_dec_ref(v___y_3057_);
lean_dec(v___y_3056_);
lean_dec(v___y_3055_);
lean_dec_ref(v___y_3049_);
v_a_3097_ = lean_ctor_get(v___x_3062_, 0);
v_isSharedCheck_3104_ = !lean_is_exclusive(v___x_3062_);
if (v_isSharedCheck_3104_ == 0)
{
v___x_3099_ = v___x_3062_;
v_isShared_3100_ = v_isSharedCheck_3104_;
goto v_resetjp_3098_;
}
else
{
lean_inc(v_a_3097_);
lean_dec(v___x_3062_);
v___x_3099_ = lean_box(0);
v_isShared_3100_ = v_isSharedCheck_3104_;
goto v_resetjp_3098_;
}
v_resetjp_3098_:
{
lean_object* v___x_3102_; 
if (v_isShared_3100_ == 0)
{
v___x_3102_ = v___x_3099_;
goto v_reusejp_3101_;
}
else
{
lean_object* v_reuseFailAlloc_3103_; 
v_reuseFailAlloc_3103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3103_, 0, v_a_3097_);
v___x_3102_ = v_reuseFailAlloc_3103_;
goto v_reusejp_3101_;
}
v_reusejp_3101_:
{
return v___x_3102_;
}
}
}
}
v___jp_3108_:
{
lean_object* v___x_3109_; lean_object* v___x_3110_; lean_object* v_options_3111_; uint8_t v_generateScript_3112_; 
v___x_3109_ = lean_st_ref_get(v_a_2867_);
lean_dec(v___x_3109_);
v___x_3110_ = lean_io_mono_nanos_now();
v_options_3111_ = lean_ctor_get(v_a_2866_, 2);
v_generateScript_3112_ = lean_ctor_get_uint8(v_options_3111_, sizeof(void*)*2);
if (v_generateScript_3112_ == 0)
{
lean_object* v___x_3113_; 
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v___x_3113_ = lean_box(0);
v___y_2879_ = v___x_3110_;
v_a_2880_ = v___x_3113_;
goto v___jp_2878_;
}
else
{
lean_object* v_ruleSet_3114_; lean_object* v___x_3115_; lean_object* v___f_3116_; 
v_ruleSet_3114_ = lean_ctor_get(v_a_2866_, 0);
v___x_3115_ = lean_box(v_completeProof_2865_);
lean_inc_ref(v_options_3111_);
lean_inc_ref(v_inst_2864_);
lean_inc_ref(v___x_2944_);
v___f_3116_ = lean_alloc_closure((void*)(lp_aesop_Aesop_traceScript___redArg___lam__1___boxed), 21, 11);
lean_closure_set(v___f_3116_, 0, v___x_2945_);
lean_closure_set(v___f_3116_, 1, v___x_2944_);
lean_closure_set(v___f_3116_, 2, v___x_3106_);
lean_closure_set(v___f_3116_, 3, v___x_3106_);
lean_closure_set(v___f_3116_, 4, v___f_3105_);
lean_closure_set(v___f_3116_, 5, v___f_3107_);
lean_closure_set(v___f_3116_, 6, v___x_3106_);
lean_closure_set(v___f_3116_, 7, v___f_3105_);
lean_closure_set(v___f_3116_, 8, v_inst_2864_);
lean_closure_set(v___f_3116_, 9, v_options_3111_);
lean_closure_set(v___f_3116_, 10, v___x_3115_);
if (v_completeProof_2865_ == 0)
{
lean_object* v___x_3117_; lean_object* v_iteration_3118_; lean_object* v___x_3119_; lean_object* v___x_3120_; 
lean_dec_ref(v___f_3116_);
v___x_3117_ = lean_st_ref_get(v_a_2867_);
v_iteration_3118_ = lean_ctor_get(v___x_3117_, 0);
lean_inc(v_iteration_3118_);
lean_dec(v___x_3117_);
lean_inc_ref(v_ruleSet_3114_);
v___x_3119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3119_, 0, v_iteration_3118_);
lean_ctor_set(v___x_3119_, 1, v_ruleSet_3114_);
v___x_3120_ = lp_aesop_Aesop_extractSafePrefixScript(v___x_3119_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
lean_dec_ref_known(v___x_3119_, 2);
if (lean_obj_tag(v___x_3120_) == 0)
{
lean_object* v_a_3121_; lean_object* v___x_3122_; 
v_a_3121_ = lean_ctor_get(v___x_3120_, 0);
lean_inc(v_a_3121_);
lean_dec_ref_known(v___x_3120_, 1);
lean_inc_ref(v_options_3111_);
v___x_3122_ = lp_aesop_Aesop_traceScript___redArg___lam__1(v___x_2945_, v___x_2944_, v___x_3106_, v___x_3106_, v___f_3105_, v___f_3107_, v___x_3106_, v___f_3105_, v_inst_2864_, v_options_3111_, v_completeProof_2865_, v_a_3121_, v_a_2866_, v_a_2867_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
lean_dec_ref(v_inst_2864_);
v___y_2914_ = v___x_3110_;
v___y_2915_ = v___x_3122_;
goto v___jp_2913_;
}
else
{
lean_object* v_a_3123_; lean_object* v___x_3125_; uint8_t v_isShared_3126_; uint8_t v_isSharedCheck_3130_; 
lean_dec(v___x_3110_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3123_ = lean_ctor_get(v___x_3120_, 0);
v_isSharedCheck_3130_ = !lean_is_exclusive(v___x_3120_);
if (v_isSharedCheck_3130_ == 0)
{
v___x_3125_ = v___x_3120_;
v_isShared_3126_ = v_isSharedCheck_3130_;
goto v_resetjp_3124_;
}
else
{
lean_inc(v_a_3123_);
lean_dec(v___x_3120_);
v___x_3125_ = lean_box(0);
v_isShared_3126_ = v_isSharedCheck_3130_;
goto v_resetjp_3124_;
}
v_resetjp_3124_:
{
lean_object* v___x_3128_; 
if (v_isShared_3126_ == 0)
{
v___x_3128_ = v___x_3125_;
goto v_reusejp_3127_;
}
else
{
lean_object* v_reuseFailAlloc_3129_; 
v_reuseFailAlloc_3129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3129_, 0, v_a_3123_);
v___x_3128_ = v_reuseFailAlloc_3129_;
goto v_reusejp_3127_;
}
v_reusejp_3127_:
{
return v___x_3128_;
}
}
}
}
else
{
lean_object* v___x_3131_; lean_object* v_iteration_3132_; lean_object* v___x_3133_; lean_object* v___x_3134_; lean_object* v___x_3135_; lean_object* v_toMonadRef_3136_; lean_object* v___x_3137_; uint8_t v_hasTrace_3138_; lean_object* v___x_3139_; 
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v___x_3131_ = lean_st_ref_get(v_a_2867_);
v_iteration_3132_ = lean_ctor_get(v___x_3131_, 0);
lean_inc(v_iteration_3132_);
lean_dec(v___x_3131_);
v___x_3133_ = lp_aesop_Aesop_TreeM_instMonad;
v___x_3134_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__2, &lp_aesop_Aesop_expandNextGoal___redArg___closed__2_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__2);
v___x_3135_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__3, &lp_aesop_Aesop_traceScript___redArg___closed__3_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__3);
v_toMonadRef_3136_ = lean_ctor_get(v___x_3135_, 0);
v___x_3137_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__4, &lp_aesop_Aesop_traceScript___redArg___closed__4_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__4);
v_hasTrace_3138_ = lean_ctor_get_uint8(v_options_2948_, sizeof(void*)*1);
lean_inc_ref(v_ruleSet_3114_);
v___x_3139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3139_, 0, v_iteration_3132_);
lean_ctor_set(v___x_3139_, 1, v_ruleSet_3114_);
if (v_hasTrace_3138_ == 0)
{
lean_object* v___x_3140_; 
v___x_3140_ = lp_aesop_Aesop_getRootGoal(v___x_3139_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3140_) == 0)
{
lean_object* v_a_3141_; 
v_a_3141_ = lean_ctor_get(v___x_3140_, 0);
lean_inc(v_a_3141_);
lean_dec_ref_known(v___x_3140_, 1);
v___y_2932_ = v___x_3110_;
v___y_2933_ = v___f_3116_;
v_____do__lift_2934_ = v_a_3141_;
v___y_2935_ = v___x_3139_;
v___y_2936_ = v_a_2868_;
v___y_2937_ = v_a_2869_;
v___y_2938_ = v_a_2870_;
v___y_2939_ = v_a_2871_;
v___y_2940_ = v_a_2872_;
v___y_2941_ = v_a_2873_;
goto v___jp_2931_;
}
else
{
lean_object* v_a_3142_; lean_object* v___x_3144_; uint8_t v_isShared_3145_; uint8_t v_isSharedCheck_3149_; 
lean_dec_ref_known(v___x_3139_, 2);
lean_dec_ref(v___f_3116_);
lean_dec(v___x_3110_);
v_a_3142_ = lean_ctor_get(v___x_3140_, 0);
v_isSharedCheck_3149_ = !lean_is_exclusive(v___x_3140_);
if (v_isSharedCheck_3149_ == 0)
{
v___x_3144_ = v___x_3140_;
v_isShared_3145_ = v_isSharedCheck_3149_;
goto v_resetjp_3143_;
}
else
{
lean_inc(v_a_3142_);
lean_dec(v___x_3140_);
v___x_3144_ = lean_box(0);
v_isShared_3145_ = v_isSharedCheck_3149_;
goto v_resetjp_3143_;
}
v_resetjp_3143_:
{
lean_object* v___x_3147_; 
if (v_isShared_3145_ == 0)
{
v___x_3147_ = v___x_3144_;
goto v_reusejp_3146_;
}
else
{
lean_object* v_reuseFailAlloc_3148_; 
v_reuseFailAlloc_3148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3148_, 0, v_a_3142_);
v___x_3147_ = v_reuseFailAlloc_3148_;
goto v_reusejp_3146_;
}
v_reusejp_3146_:
{
return v___x_3147_;
}
}
}
}
else
{
lean_object* v___x_3150_; lean_object* v_traceClass_3151_; lean_object* v___f_3152_; lean_object* v___f_3153_; lean_object* v___x_3154_; lean_object* v___x_3155_; lean_object* v___x_3156_; uint8_t v___x_3157_; 
v___x_3150_ = lp_aesop_Aesop_TraceOption_script;
v_traceClass_3151_ = lean_ctor_get(v___x_3150_, 0);
v___f_3152_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__16, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__16_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__16);
v___f_3153_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__5___closed__11));
v___x_3154_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0));
v___x_3155_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__2___closed__1));
lean_inc(v_traceClass_3151_);
v___x_3156_ = l_Lean_Name_append(v___x_3155_, v_traceClass_3151_);
v___x_3157_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2949_, v_options_2948_, v___x_3156_);
lean_dec(v___x_3156_);
if (v___x_3157_ == 0)
{
lean_object* v___x_3158_; lean_object* v___x_3159_; uint8_t v___x_3160_; 
v___x_3158_ = l_Lean_trace_profiler;
v___x_3159_ = l_Lean_Option_get___redArg(v___x_2947_, v_options_2948_, v___x_3158_);
v___x_3160_ = lean_unbox(v___x_3159_);
lean_dec(v___x_3159_);
if (v___x_3160_ == 0)
{
lean_object* v___x_3161_; 
v___x_3161_ = lp_aesop_Aesop_getRootGoal(v___x_3139_, v_a_2868_, v_a_2869_, v_a_2870_, v_a_2871_, v_a_2872_, v_a_2873_);
if (lean_obj_tag(v___x_3161_) == 0)
{
lean_object* v_a_3162_; 
v_a_3162_ = lean_ctor_get(v___x_3161_, 0);
lean_inc(v_a_3162_);
lean_dec_ref_known(v___x_3161_, 1);
v___y_2932_ = v___x_3110_;
v___y_2933_ = v___f_3116_;
v_____do__lift_2934_ = v_a_3162_;
v___y_2935_ = v___x_3139_;
v___y_2936_ = v_a_2868_;
v___y_2937_ = v_a_2869_;
v___y_2938_ = v_a_2870_;
v___y_2939_ = v_a_2871_;
v___y_2940_ = v_a_2872_;
v___y_2941_ = v_a_2873_;
goto v___jp_2931_;
}
else
{
lean_object* v_a_3163_; lean_object* v___x_3165_; uint8_t v_isShared_3166_; uint8_t v_isSharedCheck_3170_; 
lean_dec_ref_known(v___x_3139_, 2);
lean_dec_ref(v___f_3116_);
lean_dec(v___x_3110_);
v_a_3163_ = lean_ctor_get(v___x_3161_, 0);
v_isSharedCheck_3170_ = !lean_is_exclusive(v___x_3161_);
if (v_isSharedCheck_3170_ == 0)
{
v___x_3165_ = v___x_3161_;
v_isShared_3166_ = v_isSharedCheck_3170_;
goto v_resetjp_3164_;
}
else
{
lean_inc(v_a_3163_);
lean_dec(v___x_3161_);
v___x_3165_ = lean_box(0);
v_isShared_3166_ = v_isSharedCheck_3170_;
goto v_resetjp_3164_;
}
v_resetjp_3164_:
{
lean_object* v___x_3168_; 
if (v_isShared_3166_ == 0)
{
v___x_3168_ = v___x_3165_;
goto v_reusejp_3167_;
}
else
{
lean_object* v_reuseFailAlloc_3169_; 
v_reuseFailAlloc_3169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3169_, 0, v_a_3163_);
v___x_3168_ = v_reuseFailAlloc_3169_;
goto v_reusejp_3167_;
}
v_reusejp_3167_:
{
return v___x_3168_;
}
}
}
}
else
{
lean_inc(v_traceClass_3151_);
v___y_3047_ = v_toMonadRef_3136_;
v___y_3048_ = v___x_3154_;
v___y_3049_ = v___x_3139_;
v___y_3050_ = v___x_3137_;
v___y_3051_ = v___x_3134_;
v___y_3052_ = v___x_3157_;
v___y_3053_ = v_hasTrace_3138_;
v___y_3054_ = v___f_3152_;
v___y_3055_ = v_traceClass_3151_;
v___y_3056_ = v___x_3110_;
v___y_3057_ = v___f_3116_;
v___y_3058_ = v___x_3133_;
v___y_3059_ = v___f_3153_;
v___y_3060_ = v_options_2948_;
goto v___jp_3046_;
}
}
else
{
lean_inc(v_traceClass_3151_);
v___y_3047_ = v_toMonadRef_3136_;
v___y_3048_ = v___x_3154_;
v___y_3049_ = v___x_3139_;
v___y_3050_ = v___x_3137_;
v___y_3051_ = v___x_3134_;
v___y_3052_ = v___x_3157_;
v___y_3053_ = v_hasTrace_3138_;
v___y_3054_ = v___f_3152_;
v___y_3055_ = v_traceClass_3151_;
v___y_3056_ = v___x_3110_;
v___y_3057_ = v___f_3116_;
v___y_3058_ = v___x_3133_;
v___y_3059_ = v___f_3153_;
v___y_3060_ = v_options_2948_;
goto v___jp_3046_;
}
}
}
}
}
v___jp_3171_:
{
lean_object* v___x_3186_; lean_object* v___x_3187_; 
v___x_3186_ = lean_st_ref_get(v___y_3179_);
lean_dec(v___x_3186_);
lean_inc(v___y_3175_);
lean_inc_ref(v___y_3176_);
lean_inc_ref(v___y_3177_);
v___x_3187_ = lp_aesop_Aesop_Script_UScript_optimize(v___y_3177_, v___y_3172_, v___y_3176_, v___y_3175_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_);
if (lean_obj_tag(v___x_3187_) == 0)
{
lean_object* v_a_3188_; lean_object* v___x_3189_; lean_object* v___x_3190_; lean_object* v___x_3191_; lean_object* v___f_3192_; lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_46039__overap_3196_; lean_object* v___x_3197_; 
v_a_3188_ = lean_ctor_get(v___x_3187_, 0);
lean_inc(v_a_3188_);
lean_dec_ref_known(v___x_3187_, 1);
v___x_3189_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__10, &lp_aesop_Aesop_traceScript___redArg___closed__10_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__10);
v___x_3190_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_2864_);
lean_dec_ref(v_inst_2864_);
v___x_3191_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___f_3192_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
lean_inc_ref(v___x_2944_);
v___x_3193_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_3192_, v___x_2944_);
lean_inc_ref(v___x_3190_);
v___x_3194_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3194_, 0, v___x_3191_);
lean_ctor_set(v___x_3194_, 1, v___x_3190_);
lean_ctor_set(v___x_3194_, 2, v___x_3193_);
v___x_3195_ = ((lean_object*)(lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0));
lean_inc_ref(v___y_3173_);
lean_inc(v___y_3174_);
v___x_46039__overap_3196_ = lp_aesop_Aesop_checkAndTraceScript___redArg(v___x_2944_, v___x_3189_, v___x_3190_, v___x_3194_, v___f_3192_, v___x_2945_, v___y_3174_, v___y_3177_, v_a_3188_, v___y_3176_, v___y_3175_, v___y_3173_, v_completeProof_2865_, v___x_3195_);
lean_inc(v___y_3185_);
lean_inc_ref(v___y_3184_);
lean_inc(v___y_3183_);
lean_inc_ref(v___y_3182_);
lean_inc(v___y_3181_);
lean_inc(v___y_3180_);
lean_inc(v___y_3179_);
lean_inc_ref(v___y_3178_);
v___x_3197_ = lean_apply_9(v___x_46039__overap_3196_, v___y_3178_, v___y_3179_, v___y_3180_, v___y_3181_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_, lean_box(0));
return v___x_3197_;
}
else
{
lean_object* v_a_3198_; lean_object* v___x_3200_; uint8_t v_isShared_3201_; uint8_t v_isSharedCheck_3205_; 
lean_dec_ref(v___y_3177_);
lean_dec_ref(v___y_3176_);
lean_dec(v___y_3175_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3198_ = lean_ctor_get(v___x_3187_, 0);
v_isSharedCheck_3205_ = !lean_is_exclusive(v___x_3187_);
if (v_isSharedCheck_3205_ == 0)
{
v___x_3200_ = v___x_3187_;
v_isShared_3201_ = v_isSharedCheck_3205_;
goto v_resetjp_3199_;
}
else
{
lean_inc(v_a_3198_);
lean_dec(v___x_3187_);
v___x_3200_ = lean_box(0);
v_isShared_3201_ = v_isSharedCheck_3205_;
goto v_resetjp_3199_;
}
v_resetjp_3199_:
{
lean_object* v___x_3203_; 
if (v_isShared_3201_ == 0)
{
v___x_3203_ = v___x_3200_;
goto v_reusejp_3202_;
}
else
{
lean_object* v_reuseFailAlloc_3204_; 
v_reuseFailAlloc_3204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3204_, 0, v_a_3198_);
v___x_3203_ = v_reuseFailAlloc_3204_;
goto v_reusejp_3202_;
}
v_reusejp_3202_:
{
return v___x_3203_;
}
}
}
}
v___jp_3206_:
{
lean_object* v_fst_3217_; lean_object* v_snd_3218_; lean_object* v___x_3220_; uint8_t v_isShared_3221_; uint8_t v_isSharedCheck_3289_; 
v_fst_3217_ = lean_ctor_get(v_____x_3208_, 0);
v_snd_3218_ = lean_ctor_get(v_____x_3208_, 1);
v_isSharedCheck_3289_ = !lean_is_exclusive(v_____x_3208_);
if (v_isSharedCheck_3289_ == 0)
{
v___x_3220_ = v_____x_3208_;
v_isShared_3221_ = v_isSharedCheck_3289_;
goto v_resetjp_3219_;
}
else
{
lean_inc(v_snd_3218_);
lean_inc(v_fst_3217_);
lean_dec(v_____x_3208_);
v___x_3220_ = lean_box(0);
v_isShared_3221_ = v_isSharedCheck_3289_;
goto v_resetjp_3219_;
}
v_resetjp_3219_:
{
lean_object* v___x_3222_; lean_object* v___x_3223_; 
v___x_3222_ = lean_st_ref_get(v___y_3210_);
lean_dec(v___x_3222_);
v___x_3223_ = lp_aesop_Aesop_Script_UScript_checkIfEnabled(v_fst_3217_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_);
if (lean_obj_tag(v___x_3223_) == 0)
{
lean_object* v___x_3224_; lean_object* v_iteration_3225_; lean_object* v_ruleSet_3226_; lean_object* v___x_3228_; 
lean_dec_ref_known(v___x_3223_, 1);
v___x_3224_ = lean_st_ref_get(v___y_3210_);
v_iteration_3225_ = lean_ctor_get(v___x_3224_, 0);
lean_inc(v_iteration_3225_);
lean_dec(v___x_3224_);
v_ruleSet_3226_ = lean_ctor_get(v___y_3209_, 0);
lean_inc_ref(v_ruleSet_3226_);
if (v_isShared_3221_ == 0)
{
lean_ctor_set(v___x_3220_, 1, v_ruleSet_3226_);
lean_ctor_set(v___x_3220_, 0, v_iteration_3225_);
v___x_3228_ = v___x_3220_;
goto v_reusejp_3227_;
}
else
{
lean_object* v_reuseFailAlloc_3288_; 
v_reuseFailAlloc_3288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3288_, 0, v_iteration_3225_);
lean_ctor_set(v_reuseFailAlloc_3288_, 1, v_ruleSet_3226_);
v___x_3228_ = v_reuseFailAlloc_3288_;
goto v_reusejp_3227_;
}
v_reusejp_3227_:
{
lean_object* v___x_3229_; 
v___x_3229_ = lp_aesop_Aesop_getRootMVarId(v___x_3228_, v___y_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_);
lean_dec_ref(v___x_3228_);
if (lean_obj_tag(v___x_3229_) == 0)
{
lean_object* v_a_3230_; lean_object* v___x_3231_; lean_object* v___x_3232_; 
v_a_3230_ = lean_ctor_get(v___x_3229_, 0);
lean_inc(v_a_3230_);
lean_dec_ref_known(v___x_3229_, 1);
v___x_3231_ = lean_st_ref_get(v___y_3210_);
lean_dec(v___x_3231_);
v___x_3232_ = lp_aesop_Aesop_getRootMetaState___redArg(v___y_3211_);
if (lean_obj_tag(v___x_3232_) == 0)
{
lean_object* v_a_3233_; lean_object* v_toMonadOptions_3234_; lean_object* v___x_3235_; lean_object* v___x_45996__overap_3236_; lean_object* v___x_3237_; 
v_a_3233_ = lean_ctor_get(v___x_3232_, 0);
lean_inc(v_a_3233_);
lean_dec_ref_known(v___x_3232_, 1);
v_toMonadOptions_3234_ = lean_ctor_get(v___x_2945_, 0);
v___x_3235_ = lp_aesop_Aesop_TraceOption_script;
lean_inc(v_toMonadOptions_3234_);
lean_inc_ref(v___x_2944_);
v___x_45996__overap_3236_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_2944_, v_toMonadOptions_3234_, v___x_3235_);
lean_inc(v___y_3216_);
lean_inc_ref(v___y_3215_);
lean_inc(v___y_3214_);
lean_inc_ref(v___y_3213_);
lean_inc(v___y_3212_);
lean_inc(v___y_3211_);
lean_inc(v___y_3210_);
lean_inc_ref(v___y_3209_);
v___x_3237_ = lean_apply_9(v___x_45996__overap_3236_, v___y_3209_, v___y_3210_, v___y_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_, lean_box(0));
if (lean_obj_tag(v___x_3237_) == 0)
{
lean_object* v_a_3238_; lean_object* v___f_3239_; uint8_t v___x_3240_; 
v_a_3238_ = lean_ctor_get(v___x_3237_, 0);
lean_inc(v_a_3238_);
lean_dec_ref_known(v___x_3237_, 1);
v___f_3239_ = ((lean_object*)(lp_aesop_Aesop_traceScript___redArg___closed__14));
v___x_3240_ = lean_unbox(v_a_3238_);
lean_dec(v_a_3238_);
if (v___x_3240_ == 0)
{
uint8_t v___x_3241_; 
v___x_3241_ = lean_unbox(v_snd_3218_);
lean_dec(v_snd_3218_);
v___y_3172_ = v___x_3241_;
v___y_3173_ = v___y_3207_;
v___y_3174_ = v___f_3239_;
v___y_3175_ = v_a_3230_;
v___y_3176_ = v_a_3233_;
v___y_3177_ = v_fst_3217_;
v___y_3178_ = v___y_3209_;
v___y_3179_ = v___y_3210_;
v___y_3180_ = v___y_3211_;
v___y_3181_ = v___y_3212_;
v___y_3182_ = v___y_3213_;
v___y_3183_ = v___y_3214_;
v___y_3184_ = v___y_3215_;
v___y_3185_ = v___y_3216_;
goto v___jp_3171_;
}
else
{
lean_object* v___x_3242_; lean_object* v___x_3243_; 
v___x_3242_ = lean_st_ref_get(v___y_3210_);
lean_dec(v___x_3242_);
lean_inc(v_a_3230_);
lean_inc(v_a_3233_);
v___x_3243_ = lp_aesop_Aesop_Script_UScript_renderTacticSeq(v_fst_3217_, v_a_3233_, v_a_3230_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_);
if (lean_obj_tag(v___x_3243_) == 0)
{
lean_object* v_a_3244_; lean_object* v___x_3245_; lean_object* v___x_3246_; lean_object* v_traceClass_3247_; lean_object* v___f_3248_; lean_object* v___x_3249_; lean_object* v___x_3250_; lean_object* v___x_3251_; lean_object* v___x_3252_; lean_object* v___x_46071__overap_3253_; lean_object* v___x_3254_; 
v_a_3244_ = lean_ctor_get(v___x_3243_, 0);
lean_inc(v_a_3244_);
lean_dec_ref_known(v___x_3243_, 1);
v___x_3245_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__5, &lp_aesop_Aesop_expandNextGoal___redArg___closed__5_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__5);
v___x_3246_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_2864_);
v_traceClass_3247_ = lean_ctor_get(v___x_3235_, 0);
v___f_3248_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
v___x_3249_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2, &lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2_once, _init_lp_aesop_Aesop_traceScript___redArg___lam__1___closed__2);
v___x_3250_ = l_Lean_MessageData_ofSyntax(v_a_3244_);
v___x_3251_ = l_Lean_indentD(v___x_3250_);
v___x_3252_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3252_, 0, v___x_3249_);
lean_ctor_set(v___x_3252_, 1, v___x_3251_);
lean_inc(v_traceClass_3247_);
lean_inc_ref(v___x_2944_);
v___x_46071__overap_3253_ = l_Lean_addTrace___redArg(v___x_2944_, v___x_3245_, v___x_3246_, v___f_3248_, v_traceClass_3247_, v___x_3252_);
lean_inc(v___y_3216_);
lean_inc_ref(v___y_3215_);
lean_inc(v___y_3214_);
lean_inc_ref(v___y_3213_);
lean_inc(v___y_3212_);
lean_inc(v___y_3211_);
lean_inc(v___y_3210_);
lean_inc_ref(v___y_3209_);
v___x_3254_ = lean_apply_9(v___x_46071__overap_3253_, v___y_3209_, v___y_3210_, v___y_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_, lean_box(0));
if (lean_obj_tag(v___x_3254_) == 0)
{
uint8_t v___x_3255_; 
lean_dec_ref_known(v___x_3254_, 1);
v___x_3255_ = lean_unbox(v_snd_3218_);
lean_dec(v_snd_3218_);
v___y_3172_ = v___x_3255_;
v___y_3173_ = v___y_3207_;
v___y_3174_ = v___f_3239_;
v___y_3175_ = v_a_3230_;
v___y_3176_ = v_a_3233_;
v___y_3177_ = v_fst_3217_;
v___y_3178_ = v___y_3209_;
v___y_3179_ = v___y_3210_;
v___y_3180_ = v___y_3211_;
v___y_3181_ = v___y_3212_;
v___y_3182_ = v___y_3213_;
v___y_3183_ = v___y_3214_;
v___y_3184_ = v___y_3215_;
v___y_3185_ = v___y_3216_;
goto v___jp_3171_;
}
else
{
lean_dec(v_a_3233_);
lean_dec(v_a_3230_);
lean_dec(v_snd_3218_);
lean_dec(v_fst_3217_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
return v___x_3254_;
}
}
else
{
lean_object* v_a_3256_; lean_object* v___x_3258_; uint8_t v_isShared_3259_; uint8_t v_isSharedCheck_3263_; 
lean_dec(v_a_3233_);
lean_dec(v_a_3230_);
lean_dec(v_snd_3218_);
lean_dec(v_fst_3217_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3256_ = lean_ctor_get(v___x_3243_, 0);
v_isSharedCheck_3263_ = !lean_is_exclusive(v___x_3243_);
if (v_isSharedCheck_3263_ == 0)
{
v___x_3258_ = v___x_3243_;
v_isShared_3259_ = v_isSharedCheck_3263_;
goto v_resetjp_3257_;
}
else
{
lean_inc(v_a_3256_);
lean_dec(v___x_3243_);
v___x_3258_ = lean_box(0);
v_isShared_3259_ = v_isSharedCheck_3263_;
goto v_resetjp_3257_;
}
v_resetjp_3257_:
{
lean_object* v___x_3261_; 
if (v_isShared_3259_ == 0)
{
v___x_3261_ = v___x_3258_;
goto v_reusejp_3260_;
}
else
{
lean_object* v_reuseFailAlloc_3262_; 
v_reuseFailAlloc_3262_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3262_, 0, v_a_3256_);
v___x_3261_ = v_reuseFailAlloc_3262_;
goto v_reusejp_3260_;
}
v_reusejp_3260_:
{
return v___x_3261_;
}
}
}
}
}
else
{
lean_object* v_a_3264_; lean_object* v___x_3266_; uint8_t v_isShared_3267_; uint8_t v_isSharedCheck_3271_; 
lean_dec(v_a_3233_);
lean_dec(v_a_3230_);
lean_dec(v_snd_3218_);
lean_dec(v_fst_3217_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3264_ = lean_ctor_get(v___x_3237_, 0);
v_isSharedCheck_3271_ = !lean_is_exclusive(v___x_3237_);
if (v_isSharedCheck_3271_ == 0)
{
v___x_3266_ = v___x_3237_;
v_isShared_3267_ = v_isSharedCheck_3271_;
goto v_resetjp_3265_;
}
else
{
lean_inc(v_a_3264_);
lean_dec(v___x_3237_);
v___x_3266_ = lean_box(0);
v_isShared_3267_ = v_isSharedCheck_3271_;
goto v_resetjp_3265_;
}
v_resetjp_3265_:
{
lean_object* v___x_3269_; 
if (v_isShared_3267_ == 0)
{
v___x_3269_ = v___x_3266_;
goto v_reusejp_3268_;
}
else
{
lean_object* v_reuseFailAlloc_3270_; 
v_reuseFailAlloc_3270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3270_, 0, v_a_3264_);
v___x_3269_ = v_reuseFailAlloc_3270_;
goto v_reusejp_3268_;
}
v_reusejp_3268_:
{
return v___x_3269_;
}
}
}
}
else
{
lean_object* v_a_3272_; lean_object* v___x_3274_; uint8_t v_isShared_3275_; uint8_t v_isSharedCheck_3279_; 
lean_dec(v_a_3230_);
lean_dec(v_snd_3218_);
lean_dec(v_fst_3217_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3272_ = lean_ctor_get(v___x_3232_, 0);
v_isSharedCheck_3279_ = !lean_is_exclusive(v___x_3232_);
if (v_isSharedCheck_3279_ == 0)
{
v___x_3274_ = v___x_3232_;
v_isShared_3275_ = v_isSharedCheck_3279_;
goto v_resetjp_3273_;
}
else
{
lean_inc(v_a_3272_);
lean_dec(v___x_3232_);
v___x_3274_ = lean_box(0);
v_isShared_3275_ = v_isSharedCheck_3279_;
goto v_resetjp_3273_;
}
v_resetjp_3273_:
{
lean_object* v___x_3277_; 
if (v_isShared_3275_ == 0)
{
v___x_3277_ = v___x_3274_;
goto v_reusejp_3276_;
}
else
{
lean_object* v_reuseFailAlloc_3278_; 
v_reuseFailAlloc_3278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3278_, 0, v_a_3272_);
v___x_3277_ = v_reuseFailAlloc_3278_;
goto v_reusejp_3276_;
}
v_reusejp_3276_:
{
return v___x_3277_;
}
}
}
}
else
{
lean_object* v_a_3280_; lean_object* v___x_3282_; uint8_t v_isShared_3283_; uint8_t v_isSharedCheck_3287_; 
lean_dec(v_snd_3218_);
lean_dec(v_fst_3217_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3280_ = lean_ctor_get(v___x_3229_, 0);
v_isSharedCheck_3287_ = !lean_is_exclusive(v___x_3229_);
if (v_isSharedCheck_3287_ == 0)
{
v___x_3282_ = v___x_3229_;
v_isShared_3283_ = v_isSharedCheck_3287_;
goto v_resetjp_3281_;
}
else
{
lean_inc(v_a_3280_);
lean_dec(v___x_3229_);
v___x_3282_ = lean_box(0);
v_isShared_3283_ = v_isSharedCheck_3287_;
goto v_resetjp_3281_;
}
v_resetjp_3281_:
{
lean_object* v___x_3285_; 
if (v_isShared_3283_ == 0)
{
v___x_3285_ = v___x_3282_;
goto v_reusejp_3284_;
}
else
{
lean_object* v_reuseFailAlloc_3286_; 
v_reuseFailAlloc_3286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3286_, 0, v_a_3280_);
v___x_3285_ = v_reuseFailAlloc_3286_;
goto v_reusejp_3284_;
}
v_reusejp_3284_:
{
return v___x_3285_;
}
}
}
}
}
else
{
lean_del_object(v___x_3220_);
lean_dec(v_snd_3218_);
lean_dec(v_fst_3217_);
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
return v___x_3223_;
}
}
}
v___jp_3290_:
{
if (lean_obj_tag(v___y_3292_) == 0)
{
lean_object* v_a_3293_; 
v_a_3293_ = lean_ctor_get(v___y_3292_, 0);
lean_inc(v_a_3293_);
lean_dec_ref_known(v___y_3292_, 1);
v___y_3207_ = v___y_3291_;
v_____x_3208_ = v_a_3293_;
v___y_3209_ = v_a_2866_;
v___y_3210_ = v_a_2867_;
v___y_3211_ = v_a_2868_;
v___y_3212_ = v_a_2869_;
v___y_3213_ = v_a_2870_;
v___y_3214_ = v_a_2871_;
v___y_3215_ = v_a_2872_;
v___y_3216_ = v_a_2873_;
goto v___jp_3206_;
}
else
{
lean_object* v_a_3294_; lean_object* v___x_3296_; uint8_t v_isShared_3297_; uint8_t v_isSharedCheck_3301_; 
lean_dec_ref(v___x_2944_);
lean_dec_ref(v_inst_2864_);
v_a_3294_ = lean_ctor_get(v___y_3292_, 0);
v_isSharedCheck_3301_ = !lean_is_exclusive(v___y_3292_);
if (v_isSharedCheck_3301_ == 0)
{
v___x_3296_ = v___y_3292_;
v_isShared_3297_ = v_isSharedCheck_3301_;
goto v_resetjp_3295_;
}
else
{
lean_inc(v_a_3294_);
lean_dec(v___y_3292_);
v___x_3296_ = lean_box(0);
v_isShared_3297_ = v_isSharedCheck_3301_;
goto v_resetjp_3295_;
}
v_resetjp_3295_:
{
lean_object* v___x_3299_; 
if (v_isShared_3297_ == 0)
{
v___x_3299_ = v___x_3296_;
goto v_reusejp_3298_;
}
else
{
lean_object* v_reuseFailAlloc_3300_; 
v_reuseFailAlloc_3300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3300_, 0, v_a_3294_);
v___x_3299_ = v_reuseFailAlloc_3300_;
goto v_reusejp_3298_;
}
v_reusejp_3298_:
{
return v___x_3299_;
}
}
}
}
v___jp_3302_:
{
lean_object* v___x_3312_; lean_object* v___x_3313_; 
v___x_3312_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_3312_, 0, v_____do__lift_3304_);
v___x_3313_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_3312_, v___y_3305_, v___y_3306_, v___y_3307_, v___y_3308_, v___y_3309_, v___y_3310_, v___y_3311_);
lean_dec_ref(v___y_3305_);
v___y_3291_ = v___y_3303_;
v___y_3292_ = v___x_3313_;
goto v___jp_3290_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___redArg___boxed(lean_object* v_inst_3549_, lean_object* v_completeProof_3550_, lean_object* v_a_3551_, lean_object* v_a_3552_, lean_object* v_a_3553_, lean_object* v_a_3554_, lean_object* v_a_3555_, lean_object* v_a_3556_, lean_object* v_a_3557_, lean_object* v_a_3558_, lean_object* v_a_3559_){
_start:
{
uint8_t v_completeProof_boxed_3560_; lean_object* v_res_3561_; 
v_completeProof_boxed_3560_ = lean_unbox(v_completeProof_3550_);
v_res_3561_ = lp_aesop_Aesop_traceScript___redArg(v_inst_3549_, v_completeProof_boxed_3560_, v_a_3551_, v_a_3552_, v_a_3553_, v_a_3554_, v_a_3555_, v_a_3556_, v_a_3557_, v_a_3558_);
lean_dec(v_a_3558_);
lean_dec_ref(v_a_3557_);
lean_dec(v_a_3556_);
lean_dec_ref(v_a_3555_);
lean_dec(v_a_3554_);
lean_dec(v_a_3553_);
lean_dec(v_a_3552_);
lean_dec_ref(v_a_3551_);
return v_res_3561_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript(lean_object* v_Q_3562_, lean_object* v_inst_3563_, uint8_t v_completeProof_3564_, lean_object* v_a_3565_, lean_object* v_a_3566_, lean_object* v_a_3567_, lean_object* v_a_3568_, lean_object* v_a_3569_, lean_object* v_a_3570_, lean_object* v_a_3571_, lean_object* v_a_3572_){
_start:
{
lean_object* v___x_3574_; 
v___x_3574_ = lp_aesop_Aesop_traceScript___redArg(v_inst_3563_, v_completeProof_3564_, v_a_3565_, v_a_3566_, v_a_3567_, v_a_3568_, v_a_3569_, v_a_3570_, v_a_3571_, v_a_3572_);
return v___x_3574_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceScript___boxed(lean_object* v_Q_3575_, lean_object* v_inst_3576_, lean_object* v_completeProof_3577_, lean_object* v_a_3578_, lean_object* v_a_3579_, lean_object* v_a_3580_, lean_object* v_a_3581_, lean_object* v_a_3582_, lean_object* v_a_3583_, lean_object* v_a_3584_, lean_object* v_a_3585_, lean_object* v_a_3586_){
_start:
{
uint8_t v_completeProof_boxed_3587_; lean_object* v_res_3588_; 
v_completeProof_boxed_3587_ = lean_unbox(v_completeProof_3577_);
v_res_3588_ = lp_aesop_Aesop_traceScript(v_Q_3575_, v_inst_3576_, v_completeProof_boxed_3587_, v_a_3578_, v_a_3579_, v_a_3580_, v_a_3581_, v_a_3582_, v_a_3583_, v_a_3584_, v_a_3585_);
lean_dec(v_a_3585_);
lean_dec_ref(v_a_3584_);
lean_dec(v_a_3583_);
lean_dec_ref(v_a_3582_);
lean_dec(v_a_3581_);
lean_dec(v_a_3580_);
lean_dec(v_a_3579_);
lean_dec_ref(v_a_3578_);
return v_res_3588_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___redArg(lean_object* v_a_3589_, lean_object* v_a_3590_, lean_object* v_a_3591_, lean_object* v_a_3592_, lean_object* v_a_3593_, lean_object* v_a_3594_, lean_object* v_a_3595_, lean_object* v_a_3596_){
_start:
{
lean_object* v___x_3598_; lean_object* v_iteration_3599_; lean_object* v_ruleSet_3600_; lean_object* v___x_3601_; lean_object* v___x_3602_; 
v___x_3598_ = lean_st_ref_get(v_a_3590_);
v_iteration_3599_ = lean_ctor_get(v___x_3598_, 0);
lean_inc(v_iteration_3599_);
lean_dec(v___x_3598_);
v_ruleSet_3600_ = lean_ctor_get(v_a_3589_, 0);
lean_inc_ref(v_ruleSet_3600_);
v___x_3601_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3601_, 0, v_iteration_3599_);
lean_ctor_set(v___x_3601_, 1, v_ruleSet_3600_);
v___x_3602_ = lp_aesop_Aesop_getRootGoal(v___x_3601_, v_a_3591_, v_a_3592_, v_a_3593_, v_a_3594_, v_a_3595_, v_a_3596_);
lean_dec_ref_known(v___x_3601_, 2);
if (lean_obj_tag(v___x_3602_) == 0)
{
lean_object* v_a_3603_; lean_object* v___x_3604_; lean_object* v___x_3605_; lean_object* v___x_3606_; lean_object* v___x_3607_; lean_object* v___x_3608_; 
v_a_3603_ = lean_ctor_get(v___x_3602_, 0);
lean_inc(v_a_3603_);
lean_dec_ref_known(v___x_3602_, 1);
v___x_3604_ = lean_st_ref_get(v_a_3590_);
lean_dec(v___x_3604_);
v___x_3605_ = lean_st_ref_get(v_a_3603_);
lean_dec(v_a_3603_);
v___x_3606_ = lean_st_ref_get(v_a_3590_);
lean_dec(v___x_3606_);
v___x_3607_ = lp_aesop_Aesop_TraceOption_tree;
v___x_3608_ = lp_aesop_Aesop_Goal_traceTree(v___x_3605_, v___x_3607_, v_a_3593_, v_a_3594_, v_a_3595_, v_a_3596_);
return v___x_3608_;
}
else
{
lean_object* v_a_3609_; lean_object* v___x_3611_; uint8_t v_isShared_3612_; uint8_t v_isSharedCheck_3616_; 
v_a_3609_ = lean_ctor_get(v___x_3602_, 0);
v_isSharedCheck_3616_ = !lean_is_exclusive(v___x_3602_);
if (v_isSharedCheck_3616_ == 0)
{
v___x_3611_ = v___x_3602_;
v_isShared_3612_ = v_isSharedCheck_3616_;
goto v_resetjp_3610_;
}
else
{
lean_inc(v_a_3609_);
lean_dec(v___x_3602_);
v___x_3611_ = lean_box(0);
v_isShared_3612_ = v_isSharedCheck_3616_;
goto v_resetjp_3610_;
}
v_resetjp_3610_:
{
lean_object* v___x_3614_; 
if (v_isShared_3612_ == 0)
{
v___x_3614_ = v___x_3611_;
goto v_reusejp_3613_;
}
else
{
lean_object* v_reuseFailAlloc_3615_; 
v_reuseFailAlloc_3615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3615_, 0, v_a_3609_);
v___x_3614_ = v_reuseFailAlloc_3615_;
goto v_reusejp_3613_;
}
v_reusejp_3613_:
{
return v___x_3614_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___redArg___boxed(lean_object* v_a_3617_, lean_object* v_a_3618_, lean_object* v_a_3619_, lean_object* v_a_3620_, lean_object* v_a_3621_, lean_object* v_a_3622_, lean_object* v_a_3623_, lean_object* v_a_3624_, lean_object* v_a_3625_){
_start:
{
lean_object* v_res_3626_; 
v_res_3626_ = lp_aesop_Aesop_traceTree___redArg(v_a_3617_, v_a_3618_, v_a_3619_, v_a_3620_, v_a_3621_, v_a_3622_, v_a_3623_, v_a_3624_);
lean_dec(v_a_3624_);
lean_dec_ref(v_a_3623_);
lean_dec(v_a_3622_);
lean_dec_ref(v_a_3621_);
lean_dec(v_a_3620_);
lean_dec(v_a_3619_);
lean_dec(v_a_3618_);
lean_dec_ref(v_a_3617_);
return v_res_3626_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree(lean_object* v_Q_3627_, lean_object* v_inst_3628_, lean_object* v_a_3629_, lean_object* v_a_3630_, lean_object* v_a_3631_, lean_object* v_a_3632_, lean_object* v_a_3633_, lean_object* v_a_3634_, lean_object* v_a_3635_, lean_object* v_a_3636_){
_start:
{
lean_object* v___x_3638_; 
v___x_3638_ = lp_aesop_Aesop_traceTree___redArg(v_a_3629_, v_a_3630_, v_a_3631_, v_a_3632_, v_a_3633_, v_a_3634_, v_a_3635_, v_a_3636_);
return v___x_3638_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traceTree___boxed(lean_object* v_Q_3639_, lean_object* v_inst_3640_, lean_object* v_a_3641_, lean_object* v_a_3642_, lean_object* v_a_3643_, lean_object* v_a_3644_, lean_object* v_a_3645_, lean_object* v_a_3646_, lean_object* v_a_3647_, lean_object* v_a_3648_, lean_object* v_a_3649_){
_start:
{
lean_object* v_res_3650_; 
v_res_3650_ = lp_aesop_Aesop_traceTree(v_Q_3639_, v_inst_3640_, v_a_3641_, v_a_3642_, v_a_3643_, v_a_3644_, v_a_3645_, v_a_3646_, v_a_3647_, v_a_3648_);
lean_dec(v_a_3648_);
lean_dec_ref(v_a_3647_);
lean_dec(v_a_3646_);
lean_dec_ref(v_a_3645_);
lean_dec(v_a_3644_);
lean_dec(v_a_3643_);
lean_dec(v_a_3642_);
lean_dec_ref(v_a_3641_);
lean_dec_ref(v_inst_3640_);
return v_res_3650_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___redArg(lean_object* v_inst_3651_, lean_object* v_a_3652_, lean_object* v_a_3653_, lean_object* v_a_3654_, lean_object* v_a_3655_, lean_object* v_a_3656_, lean_object* v_a_3657_, lean_object* v_a_3658_, lean_object* v_a_3659_){
_start:
{
lean_object* v___x_3661_; lean_object* v___x_3662_; 
v___x_3661_ = lean_st_ref_get(v_a_3653_);
lean_dec(v___x_3661_);
v___x_3662_ = lp_aesop_Aesop_getRootMVarCluster___redArg(v_a_3654_);
if (lean_obj_tag(v___x_3662_) == 0)
{
lean_object* v_a_3663_; lean_object* v___x_3665_; uint8_t v_isShared_3666_; uint8_t v_isSharedCheck_3714_; 
v_a_3663_ = lean_ctor_get(v___x_3662_, 0);
v_isSharedCheck_3714_ = !lean_is_exclusive(v___x_3662_);
if (v_isSharedCheck_3714_ == 0)
{
v___x_3665_ = v___x_3662_;
v_isShared_3666_ = v_isSharedCheck_3714_;
goto v_resetjp_3664_;
}
else
{
lean_inc(v_a_3663_);
lean_dec(v___x_3662_);
v___x_3665_ = lean_box(0);
v_isShared_3666_ = v_isSharedCheck_3714_;
goto v_resetjp_3664_;
}
v_resetjp_3664_:
{
lean_object* v___x_3667_; lean_object* v___x_3668_; lean_object* v___x_3669_; lean_object* v_elimMVarCluster_3670_; lean_object* v___x_3671_; uint8_t v_state_3672_; uint8_t v___x_3673_; 
v___x_3667_ = lean_st_ref_get(v_a_3653_);
lean_dec(v___x_3667_);
v___x_3668_ = lean_st_ref_get(v_a_3663_);
lean_dec(v_a_3663_);
v___x_3669_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_3670_ = lean_ctor_get(v___x_3669_, 5);
lean_inc_ref(v_elimMVarCluster_3670_);
v___x_3671_ = lean_apply_1(v_elimMVarCluster_3670_, v___x_3668_);
v_state_3672_ = lean_ctor_get_uint8(v___x_3671_, sizeof(void*)*2 + 1);
lean_dec_ref(v___x_3671_);
v___x_3673_ = lp_aesop_Aesop_NodeState_isProven(v_state_3672_);
if (v___x_3673_ == 0)
{
lean_object* v___x_3674_; lean_object* v___x_3676_; 
lean_dec_ref(v_inst_3651_);
v___x_3674_ = lean_box(v___x_3673_);
if (v_isShared_3666_ == 0)
{
lean_ctor_set(v___x_3665_, 0, v___x_3674_);
v___x_3676_ = v___x_3665_;
goto v_reusejp_3675_;
}
else
{
lean_object* v_reuseFailAlloc_3677_; 
v_reuseFailAlloc_3677_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3677_, 0, v___x_3674_);
v___x_3676_ = v_reuseFailAlloc_3677_;
goto v_reusejp_3675_;
}
v_reusejp_3675_:
{
return v___x_3676_;
}
}
else
{
lean_object* v___x_3678_; 
lean_del_object(v___x_3665_);
lean_inc_ref(v_inst_3651_);
v___x_3678_ = lp_aesop_Aesop_finalizeProof___redArg(v_inst_3651_, v_a_3652_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_, v_a_3657_, v_a_3658_, v_a_3659_);
if (lean_obj_tag(v___x_3678_) == 0)
{
lean_object* v___x_3679_; 
lean_dec_ref_known(v___x_3678_, 1);
v___x_3679_ = lp_aesop_Aesop_traceScript___redArg(v_inst_3651_, v___x_3673_, v_a_3652_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_, v_a_3657_, v_a_3658_, v_a_3659_);
if (lean_obj_tag(v___x_3679_) == 0)
{
lean_object* v___x_3680_; 
lean_dec_ref_known(v___x_3679_, 1);
v___x_3680_ = lp_aesop_Aesop_traceTree___redArg(v_a_3652_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_, v_a_3657_, v_a_3658_, v_a_3659_);
if (lean_obj_tag(v___x_3680_) == 0)
{
lean_object* v___x_3682_; uint8_t v_isShared_3683_; uint8_t v_isSharedCheck_3688_; 
v_isSharedCheck_3688_ = !lean_is_exclusive(v___x_3680_);
if (v_isSharedCheck_3688_ == 0)
{
lean_object* v_unused_3689_; 
v_unused_3689_ = lean_ctor_get(v___x_3680_, 0);
lean_dec(v_unused_3689_);
v___x_3682_ = v___x_3680_;
v_isShared_3683_ = v_isSharedCheck_3688_;
goto v_resetjp_3681_;
}
else
{
lean_dec(v___x_3680_);
v___x_3682_ = lean_box(0);
v_isShared_3683_ = v_isSharedCheck_3688_;
goto v_resetjp_3681_;
}
v_resetjp_3681_:
{
lean_object* v___x_3684_; lean_object* v___x_3686_; 
v___x_3684_ = lean_box(v___x_3673_);
if (v_isShared_3683_ == 0)
{
lean_ctor_set(v___x_3682_, 0, v___x_3684_);
v___x_3686_ = v___x_3682_;
goto v_reusejp_3685_;
}
else
{
lean_object* v_reuseFailAlloc_3687_; 
v_reuseFailAlloc_3687_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3687_, 0, v___x_3684_);
v___x_3686_ = v_reuseFailAlloc_3687_;
goto v_reusejp_3685_;
}
v_reusejp_3685_:
{
return v___x_3686_;
}
}
}
else
{
lean_object* v_a_3690_; lean_object* v___x_3692_; uint8_t v_isShared_3693_; uint8_t v_isSharedCheck_3697_; 
v_a_3690_ = lean_ctor_get(v___x_3680_, 0);
v_isSharedCheck_3697_ = !lean_is_exclusive(v___x_3680_);
if (v_isSharedCheck_3697_ == 0)
{
v___x_3692_ = v___x_3680_;
v_isShared_3693_ = v_isSharedCheck_3697_;
goto v_resetjp_3691_;
}
else
{
lean_inc(v_a_3690_);
lean_dec(v___x_3680_);
v___x_3692_ = lean_box(0);
v_isShared_3693_ = v_isSharedCheck_3697_;
goto v_resetjp_3691_;
}
v_resetjp_3691_:
{
lean_object* v___x_3695_; 
if (v_isShared_3693_ == 0)
{
v___x_3695_ = v___x_3692_;
goto v_reusejp_3694_;
}
else
{
lean_object* v_reuseFailAlloc_3696_; 
v_reuseFailAlloc_3696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3696_, 0, v_a_3690_);
v___x_3695_ = v_reuseFailAlloc_3696_;
goto v_reusejp_3694_;
}
v_reusejp_3694_:
{
return v___x_3695_;
}
}
}
}
else
{
lean_object* v_a_3698_; lean_object* v___x_3700_; uint8_t v_isShared_3701_; uint8_t v_isSharedCheck_3705_; 
v_a_3698_ = lean_ctor_get(v___x_3679_, 0);
v_isSharedCheck_3705_ = !lean_is_exclusive(v___x_3679_);
if (v_isSharedCheck_3705_ == 0)
{
v___x_3700_ = v___x_3679_;
v_isShared_3701_ = v_isSharedCheck_3705_;
goto v_resetjp_3699_;
}
else
{
lean_inc(v_a_3698_);
lean_dec(v___x_3679_);
v___x_3700_ = lean_box(0);
v_isShared_3701_ = v_isSharedCheck_3705_;
goto v_resetjp_3699_;
}
v_resetjp_3699_:
{
lean_object* v___x_3703_; 
if (v_isShared_3701_ == 0)
{
v___x_3703_ = v___x_3700_;
goto v_reusejp_3702_;
}
else
{
lean_object* v_reuseFailAlloc_3704_; 
v_reuseFailAlloc_3704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3704_, 0, v_a_3698_);
v___x_3703_ = v_reuseFailAlloc_3704_;
goto v_reusejp_3702_;
}
v_reusejp_3702_:
{
return v___x_3703_;
}
}
}
}
else
{
lean_object* v_a_3706_; lean_object* v___x_3708_; uint8_t v_isShared_3709_; uint8_t v_isSharedCheck_3713_; 
lean_dec_ref(v_inst_3651_);
v_a_3706_ = lean_ctor_get(v___x_3678_, 0);
v_isSharedCheck_3713_ = !lean_is_exclusive(v___x_3678_);
if (v_isSharedCheck_3713_ == 0)
{
v___x_3708_ = v___x_3678_;
v_isShared_3709_ = v_isSharedCheck_3713_;
goto v_resetjp_3707_;
}
else
{
lean_inc(v_a_3706_);
lean_dec(v___x_3678_);
v___x_3708_ = lean_box(0);
v_isShared_3709_ = v_isSharedCheck_3713_;
goto v_resetjp_3707_;
}
v_resetjp_3707_:
{
lean_object* v___x_3711_; 
if (v_isShared_3709_ == 0)
{
v___x_3711_ = v___x_3708_;
goto v_reusejp_3710_;
}
else
{
lean_object* v_reuseFailAlloc_3712_; 
v_reuseFailAlloc_3712_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3712_, 0, v_a_3706_);
v___x_3711_ = v_reuseFailAlloc_3712_;
goto v_reusejp_3710_;
}
v_reusejp_3710_:
{
return v___x_3711_;
}
}
}
}
}
}
else
{
lean_object* v_a_3715_; lean_object* v___x_3717_; uint8_t v_isShared_3718_; uint8_t v_isSharedCheck_3722_; 
lean_dec_ref(v_inst_3651_);
v_a_3715_ = lean_ctor_get(v___x_3662_, 0);
v_isSharedCheck_3722_ = !lean_is_exclusive(v___x_3662_);
if (v_isSharedCheck_3722_ == 0)
{
v___x_3717_ = v___x_3662_;
v_isShared_3718_ = v_isSharedCheck_3722_;
goto v_resetjp_3716_;
}
else
{
lean_inc(v_a_3715_);
lean_dec(v___x_3662_);
v___x_3717_ = lean_box(0);
v_isShared_3718_ = v_isSharedCheck_3722_;
goto v_resetjp_3716_;
}
v_resetjp_3716_:
{
lean_object* v___x_3720_; 
if (v_isShared_3718_ == 0)
{
v___x_3720_ = v___x_3717_;
goto v_reusejp_3719_;
}
else
{
lean_object* v_reuseFailAlloc_3721_; 
v_reuseFailAlloc_3721_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3721_, 0, v_a_3715_);
v___x_3720_ = v_reuseFailAlloc_3721_;
goto v_reusejp_3719_;
}
v_reusejp_3719_:
{
return v___x_3720_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___redArg___boxed(lean_object* v_inst_3723_, lean_object* v_a_3724_, lean_object* v_a_3725_, lean_object* v_a_3726_, lean_object* v_a_3727_, lean_object* v_a_3728_, lean_object* v_a_3729_, lean_object* v_a_3730_, lean_object* v_a_3731_, lean_object* v_a_3732_){
_start:
{
lean_object* v_res_3733_; 
v_res_3733_ = lp_aesop_Aesop_finishIfProven___redArg(v_inst_3723_, v_a_3724_, v_a_3725_, v_a_3726_, v_a_3727_, v_a_3728_, v_a_3729_, v_a_3730_, v_a_3731_);
lean_dec(v_a_3731_);
lean_dec_ref(v_a_3730_);
lean_dec(v_a_3729_);
lean_dec_ref(v_a_3728_);
lean_dec(v_a_3727_);
lean_dec(v_a_3726_);
lean_dec(v_a_3725_);
lean_dec_ref(v_a_3724_);
return v_res_3733_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven(lean_object* v_Q_3734_, lean_object* v_inst_3735_, lean_object* v_a_3736_, lean_object* v_a_3737_, lean_object* v_a_3738_, lean_object* v_a_3739_, lean_object* v_a_3740_, lean_object* v_a_3741_, lean_object* v_a_3742_, lean_object* v_a_3743_){
_start:
{
lean_object* v___x_3745_; 
v___x_3745_ = lp_aesop_Aesop_finishIfProven___redArg(v_inst_3735_, v_a_3736_, v_a_3737_, v_a_3738_, v_a_3739_, v_a_3740_, v_a_3741_, v_a_3742_, v_a_3743_);
return v___x_3745_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_finishIfProven___boxed(lean_object* v_Q_3746_, lean_object* v_inst_3747_, lean_object* v_a_3748_, lean_object* v_a_3749_, lean_object* v_a_3750_, lean_object* v_a_3751_, lean_object* v_a_3752_, lean_object* v_a_3753_, lean_object* v_a_3754_, lean_object* v_a_3755_, lean_object* v_a_3756_){
_start:
{
lean_object* v_res_3757_; 
v_res_3757_ = lp_aesop_Aesop_finishIfProven(v_Q_3746_, v_inst_3747_, v_a_3748_, v_a_3749_, v_a_3750_, v_a_3751_, v_a_3752_, v_a_3753_, v_a_3754_, v_a_3755_);
lean_dec(v_a_3755_);
lean_dec_ref(v_a_3754_);
lean_dec(v_a_3753_);
lean_dec_ref(v_a_3752_);
lean_dec(v_a_3751_);
lean_dec(v_a_3750_);
lean_dec(v_a_3749_);
lean_dec_ref(v_a_3748_);
return v_res_3757_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(lean_object* v_val_3758_, lean_object* v_as_3759_, size_t v_i_3760_, size_t v_stop_3761_, lean_object* v_b_3762_, lean_object* v___y_3763_, lean_object* v___y_3764_, lean_object* v___y_3765_, lean_object* v___y_3766_, lean_object* v___y_3767_, lean_object* v___y_3768_, lean_object* v___y_3769_){
_start:
{
uint8_t v___x_3771_; 
v___x_3771_ = lean_usize_dec_eq(v_i_3760_, v_stop_3761_);
if (v___x_3771_ == 0)
{
lean_object* v___x_3772_; lean_object* v___x_3773_; lean_object* v___x_3774_; 
v___x_3772_ = lean_array_uget_borrowed(v_as_3759_, v_i_3760_);
lean_inc(v___x_3772_);
v___x_3773_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_3773_, 0, v___x_3772_);
v___x_3774_ = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(v_val_3758_, v___x_3773_, v___y_3763_, v___y_3764_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_, v___y_3769_);
lean_dec_ref_known(v___x_3773_, 1);
if (lean_obj_tag(v___x_3774_) == 0)
{
lean_object* v_a_3775_; size_t v___x_3776_; size_t v___x_3777_; 
v_a_3775_ = lean_ctor_get(v___x_3774_, 0);
lean_inc(v_a_3775_);
lean_dec_ref_known(v___x_3774_, 1);
v___x_3776_ = ((size_t)1ULL);
v___x_3777_ = lean_usize_add(v_i_3760_, v___x_3776_);
v_i_3760_ = v___x_3777_;
v_b_3762_ = v_a_3775_;
goto _start;
}
else
{
return v___x_3774_;
}
}
else
{
lean_object* v___x_3779_; 
v___x_3779_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3779_, 0, v_b_3762_);
return v___x_3779_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(lean_object* v_val_3780_, lean_object* v_as_3781_, size_t v_i_3782_, size_t v_stop_3783_, lean_object* v_b_3784_, lean_object* v___y_3785_, lean_object* v___y_3786_, lean_object* v___y_3787_, lean_object* v___y_3788_, lean_object* v___y_3789_, lean_object* v___y_3790_, lean_object* v___y_3791_){
_start:
{
uint8_t v___x_3793_; 
v___x_3793_ = lean_usize_dec_eq(v_i_3782_, v_stop_3783_);
if (v___x_3793_ == 0)
{
lean_object* v___x_3794_; lean_object* v___x_3795_; lean_object* v___x_3796_; 
v___x_3794_ = lean_array_uget_borrowed(v_as_3781_, v_i_3782_);
lean_inc(v___x_3794_);
v___x_3795_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3795_, 0, v___x_3794_);
v___x_3796_ = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(v_val_3780_, v___x_3795_, v___y_3785_, v___y_3786_, v___y_3787_, v___y_3788_, v___y_3789_, v___y_3790_, v___y_3791_);
lean_dec_ref_known(v___x_3795_, 1);
if (lean_obj_tag(v___x_3796_) == 0)
{
lean_object* v_a_3797_; size_t v___x_3798_; size_t v___x_3799_; 
v_a_3797_ = lean_ctor_get(v___x_3796_, 0);
lean_inc(v_a_3797_);
lean_dec_ref_known(v___x_3796_, 1);
v___x_3798_ = ((size_t)1ULL);
v___x_3799_ = lean_usize_add(v_i_3782_, v___x_3798_);
v_i_3782_ = v___x_3799_;
v_b_3784_ = v_a_3797_;
goto _start;
}
else
{
return v___x_3796_;
}
}
else
{
lean_object* v___x_3801_; 
v___x_3801_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3801_, 0, v_b_3784_);
return v___x_3801_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(lean_object* v_val_3802_, lean_object* v_x_3803_, lean_object* v___y_3804_, lean_object* v___y_3805_, lean_object* v___y_3806_, lean_object* v___y_3807_, lean_object* v___y_3808_, lean_object* v___y_3809_, lean_object* v___y_3810_){
_start:
{
switch(lean_obj_tag(v_x_3803_))
{
case 0:
{
lean_object* v_gref_3824_; lean_object* v___x_3842_; lean_object* v___x_3843_; lean_object* v_elimGoal_3844_; lean_object* v___x_3845_; lean_object* v_preNormGoal_3846_; lean_object* v_normalizationState_3847_; lean_object* v___x_3848_; 
v_gref_3824_ = lean_ctor_get(v_x_3803_, 0);
v___x_3842_ = lean_st_ref_get(v_gref_3824_);
v___x_3843_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_3844_ = lean_ctor_get(v___x_3843_, 1);
lean_inc_ref(v_elimGoal_3844_);
v___x_3845_ = lean_apply_1(v_elimGoal_3844_, v___x_3842_);
v_preNormGoal_3846_ = lean_ctor_get(v___x_3845_, 5);
lean_inc(v_preNormGoal_3846_);
v_normalizationState_3847_ = lean_ctor_get(v___x_3845_, 6);
lean_inc(v_normalizationState_3847_);
lean_dec_ref(v___x_3845_);
v___x_3848_ = lp_aesop_Aesop_NormalizationState_normalizedGoal_x3f(v_normalizationState_3847_);
lean_dec(v_normalizationState_3847_);
if (lean_obj_tag(v___x_3848_) == 1)
{
lean_object* v_val_3849_; lean_object* v___x_3851_; uint8_t v_isShared_3852_; uint8_t v_isSharedCheck_3861_; 
v_val_3849_ = lean_ctor_get(v___x_3848_, 0);
v_isSharedCheck_3861_ = !lean_is_exclusive(v___x_3848_);
if (v_isSharedCheck_3861_ == 0)
{
v___x_3851_ = v___x_3848_;
v_isShared_3852_ = v_isSharedCheck_3861_;
goto v_resetjp_3850_;
}
else
{
lean_inc(v_val_3849_);
lean_dec(v___x_3848_);
v___x_3851_ = lean_box(0);
v_isShared_3852_ = v_isSharedCheck_3861_;
goto v_resetjp_3850_;
}
v_resetjp_3850_:
{
uint8_t v___x_3853_; 
v___x_3853_ = l_Lean_instBEqMVarId_beq(v_val_3849_, v_preNormGoal_3846_);
lean_dec(v_preNormGoal_3846_);
lean_dec(v_val_3849_);
if (v___x_3853_ == 0)
{
uint8_t v___x_3854_; lean_object* v___x_3855_; lean_object* v___x_3856_; lean_object* v___x_3857_; lean_object* v___x_3859_; 
v___x_3854_ = 1;
v___x_3855_ = lean_box(v___x_3854_);
v___x_3856_ = lean_st_ref_set(v_val_3802_, v___x_3855_);
v___x_3857_ = lean_box(0);
if (v_isShared_3852_ == 0)
{
lean_ctor_set_tag(v___x_3851_, 0);
lean_ctor_set(v___x_3851_, 0, v___x_3857_);
v___x_3859_ = v___x_3851_;
goto v_reusejp_3858_;
}
else
{
lean_object* v_reuseFailAlloc_3860_; 
v_reuseFailAlloc_3860_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3860_, 0, v___x_3857_);
v___x_3859_ = v_reuseFailAlloc_3860_;
goto v_reusejp_3858_;
}
v_reusejp_3858_:
{
return v___x_3859_;
}
}
else
{
lean_del_object(v___x_3851_);
goto v___jp_3825_;
}
}
}
else
{
lean_dec(v___x_3848_);
lean_dec(v_preNormGoal_3846_);
goto v___jp_3825_;
}
v___jp_3825_:
{
lean_object* v___x_3826_; lean_object* v___x_3827_; lean_object* v_elimGoal_3828_; lean_object* v___x_3829_; lean_object* v_children_3830_; lean_object* v___x_3831_; lean_object* v___x_3832_; uint8_t v___x_3833_; 
v___x_3826_ = lean_st_ref_get(v_gref_3824_);
v___x_3827_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_3828_ = lean_ctor_get(v___x_3827_, 1);
lean_inc_ref(v_elimGoal_3828_);
v___x_3829_ = lean_apply_1(v_elimGoal_3828_, v___x_3826_);
v_children_3830_ = lean_ctor_get(v___x_3829_, 2);
lean_inc_ref(v_children_3830_);
lean_dec_ref(v___x_3829_);
v___x_3831_ = lean_unsigned_to_nat(0u);
v___x_3832_ = lean_array_get_size(v_children_3830_);
v___x_3833_ = lean_nat_dec_lt(v___x_3831_, v___x_3832_);
if (v___x_3833_ == 0)
{
lean_dec_ref(v_children_3830_);
goto v___jp_3818_;
}
else
{
lean_object* v___x_3834_; uint8_t v___x_3835_; 
v___x_3834_ = lean_box(0);
v___x_3835_ = lean_nat_dec_le(v___x_3832_, v___x_3832_);
if (v___x_3835_ == 0)
{
if (v___x_3833_ == 0)
{
lean_dec_ref(v_children_3830_);
goto v___jp_3818_;
}
else
{
size_t v___x_3836_; size_t v___x_3837_; lean_object* v___x_3838_; 
v___x_3836_ = ((size_t)0ULL);
v___x_3837_ = lean_usize_of_nat(v___x_3832_);
v___x_3838_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(v_val_3802_, v_children_3830_, v___x_3836_, v___x_3837_, v___x_3834_, v___y_3804_, v___y_3805_, v___y_3806_, v___y_3807_, v___y_3808_, v___y_3809_, v___y_3810_);
lean_dec_ref(v_children_3830_);
if (lean_obj_tag(v___x_3838_) == 0)
{
lean_dec_ref_known(v___x_3838_, 1);
goto v___jp_3818_;
}
else
{
return v___x_3838_;
}
}
}
else
{
size_t v___x_3839_; size_t v___x_3840_; lean_object* v___x_3841_; 
v___x_3839_ = ((size_t)0ULL);
v___x_3840_ = lean_usize_of_nat(v___x_3832_);
v___x_3841_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(v_val_3802_, v_children_3830_, v___x_3839_, v___x_3840_, v___x_3834_, v___y_3804_, v___y_3805_, v___y_3806_, v___y_3807_, v___y_3808_, v___y_3809_, v___y_3810_);
lean_dec_ref(v_children_3830_);
if (lean_obj_tag(v___x_3841_) == 0)
{
lean_dec_ref_known(v___x_3841_, 1);
goto v___jp_3818_;
}
else
{
return v___x_3841_;
}
}
}
}
}
case 1:
{
lean_object* v_rref_3862_; lean_object* v___x_3863_; lean_object* v___x_3864_; lean_object* v_elimRapp_3865_; lean_object* v___x_3866_; lean_object* v_appliedRule_3867_; uint8_t v___y_3874_; lean_object* v___x_3889_; lean_object* v_name_3890_; lean_object* v___x_3891_; uint8_t v_builder_3892_; uint64_t v_hash_3893_; lean_object* v_name_3894_; uint8_t v_builder_3895_; uint8_t v_phase_3896_; uint8_t v_scope_3897_; uint64_t v_hash_3898_; uint8_t v___y_3900_; uint8_t v___x_3907_; 
v_rref_3862_ = lean_ctor_get(v_x_3803_, 0);
v___x_3863_ = lean_st_ref_get(v_rref_3862_);
v___x_3864_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_3865_ = lean_ctor_get(v___x_3864_, 3);
lean_inc_ref(v_elimRapp_3865_);
v___x_3866_ = lean_apply_1(v_elimRapp_3865_, v___x_3863_);
v_appliedRule_3867_ = lean_ctor_get(v___x_3866_, 3);
lean_inc_ref(v_appliedRule_3867_);
lean_dec_ref(v___x_3866_);
v___x_3889_ = lp_aesop_Aesop_preprocessRule;
v_name_3890_ = lean_ctor_get(v___x_3889_, 0);
v___x_3891_ = lp_aesop_Aesop_RegularRule_name(v_appliedRule_3867_);
v_builder_3892_ = lean_ctor_get_uint8(v___x_3891_, sizeof(void*)*1 + 8);
v_hash_3893_ = lean_ctor_get_uint64(v___x_3891_, sizeof(void*)*1);
v_name_3894_ = lean_ctor_get(v_name_3890_, 0);
v_builder_3895_ = lean_ctor_get_uint8(v_name_3890_, sizeof(void*)*1 + 8);
v_phase_3896_ = lean_ctor_get_uint8(v_name_3890_, sizeof(void*)*1 + 9);
v_scope_3897_ = lean_ctor_get_uint8(v_name_3890_, sizeof(void*)*1 + 10);
v_hash_3898_ = lean_ctor_get_uint64(v_name_3890_, sizeof(void*)*1);
v___x_3907_ = lean_uint64_dec_eq(v_hash_3893_, v_hash_3898_);
if (v___x_3907_ == 0)
{
v___y_3900_ = v___x_3907_;
goto v___jp_3899_;
}
else
{
uint8_t v___x_3908_; 
v___x_3908_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_3892_, v_builder_3895_);
v___y_3900_ = v___x_3908_;
goto v___jp_3899_;
}
v___jp_3868_:
{
uint8_t v___x_3869_; 
v___x_3869_ = lp_aesop_Aesop_RegularRule_isUnsafe(v_appliedRule_3867_);
lean_dec_ref(v_appliedRule_3867_);
if (v___x_3869_ == 0)
{
uint8_t v___x_3870_; lean_object* v___x_3871_; lean_object* v___x_3872_; 
v___x_3870_ = 1;
v___x_3871_ = lean_box(v___x_3870_);
v___x_3872_ = lean_st_ref_set(v_val_3802_, v___x_3871_);
goto v___jp_3815_;
}
else
{
goto v___jp_3815_;
}
}
v___jp_3873_:
{
if (v___y_3874_ == 0)
{
goto v___jp_3868_;
}
else
{
lean_object* v___x_3875_; lean_object* v___x_3876_; lean_object* v_children_3877_; lean_object* v___x_3878_; lean_object* v___x_3879_; uint8_t v___x_3880_; 
lean_dec_ref(v_appliedRule_3867_);
v___x_3875_ = lean_st_ref_get(v_rref_3862_);
lean_inc_ref(v_elimRapp_3865_);
v___x_3876_ = lean_apply_1(v_elimRapp_3865_, v___x_3875_);
v_children_3877_ = lean_ctor_get(v___x_3876_, 2);
lean_inc_ref(v_children_3877_);
lean_dec_ref(v___x_3876_);
v___x_3878_ = lean_unsigned_to_nat(0u);
v___x_3879_ = lean_array_get_size(v_children_3877_);
v___x_3880_ = lean_nat_dec_lt(v___x_3878_, v___x_3879_);
if (v___x_3880_ == 0)
{
lean_dec_ref(v_children_3877_);
goto v___jp_3812_;
}
else
{
lean_object* v___x_3881_; uint8_t v___x_3882_; 
v___x_3881_ = lean_box(0);
v___x_3882_ = lean_nat_dec_le(v___x_3879_, v___x_3879_);
if (v___x_3882_ == 0)
{
if (v___x_3880_ == 0)
{
lean_dec_ref(v_children_3877_);
goto v___jp_3812_;
}
else
{
size_t v___x_3883_; size_t v___x_3884_; lean_object* v___x_3885_; 
v___x_3883_ = ((size_t)0ULL);
v___x_3884_ = lean_usize_of_nat(v___x_3879_);
v___x_3885_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(v_val_3802_, v_children_3877_, v___x_3883_, v___x_3884_, v___x_3881_, v___y_3804_, v___y_3805_, v___y_3806_, v___y_3807_, v___y_3808_, v___y_3809_, v___y_3810_);
lean_dec_ref(v_children_3877_);
if (lean_obj_tag(v___x_3885_) == 0)
{
lean_dec_ref_known(v___x_3885_, 1);
goto v___jp_3812_;
}
else
{
return v___x_3885_;
}
}
}
else
{
size_t v___x_3886_; size_t v___x_3887_; lean_object* v___x_3888_; 
v___x_3886_ = ((size_t)0ULL);
v___x_3887_ = lean_usize_of_nat(v___x_3879_);
v___x_3888_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(v_val_3802_, v_children_3877_, v___x_3886_, v___x_3887_, v___x_3881_, v___y_3804_, v___y_3805_, v___y_3806_, v___y_3807_, v___y_3808_, v___y_3809_, v___y_3810_);
lean_dec_ref(v_children_3877_);
if (lean_obj_tag(v___x_3888_) == 0)
{
lean_dec_ref_known(v___x_3888_, 1);
goto v___jp_3812_;
}
else
{
return v___x_3888_;
}
}
}
}
}
v___jp_3899_:
{
if (v___y_3900_ == 0)
{
lean_dec_ref(v___x_3891_);
goto v___jp_3868_;
}
else
{
lean_object* v_name_3901_; uint8_t v_phase_3902_; uint8_t v_scope_3903_; uint8_t v___x_3904_; 
v_name_3901_ = lean_ctor_get(v___x_3891_, 0);
lean_inc(v_name_3901_);
v_phase_3902_ = lean_ctor_get_uint8(v___x_3891_, sizeof(void*)*1 + 9);
v_scope_3903_ = lean_ctor_get_uint8(v___x_3891_, sizeof(void*)*1 + 10);
lean_dec_ref(v___x_3891_);
v___x_3904_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_3902_, v_phase_3896_);
if (v___x_3904_ == 0)
{
lean_dec(v_name_3901_);
v___y_3874_ = v___x_3904_;
goto v___jp_3873_;
}
else
{
uint8_t v___x_3905_; 
v___x_3905_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_3903_, v_scope_3897_);
if (v___x_3905_ == 0)
{
lean_dec(v_name_3901_);
v___y_3874_ = v___x_3905_;
goto v___jp_3873_;
}
else
{
uint8_t v___x_3906_; 
v___x_3906_ = lean_name_eq(v_name_3901_, v_name_3894_);
lean_dec(v_name_3901_);
v___y_3874_ = v___x_3906_;
goto v___jp_3873_;
}
}
}
}
}
default: 
{
lean_object* v_cref_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; lean_object* v_elimMVarCluster_3912_; lean_object* v___x_3913_; lean_object* v_goals_3914_; lean_object* v___x_3915_; lean_object* v___x_3916_; uint8_t v___x_3917_; 
v_cref_3909_ = lean_ctor_get(v_x_3803_, 0);
v___x_3910_ = lean_st_ref_get(v_cref_3909_);
v___x_3911_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_3912_ = lean_ctor_get(v___x_3911_, 5);
lean_inc_ref(v_elimMVarCluster_3912_);
v___x_3913_ = lean_apply_1(v_elimMVarCluster_3912_, v___x_3910_);
v_goals_3914_ = lean_ctor_get(v___x_3913_, 1);
lean_inc_ref(v_goals_3914_);
lean_dec_ref(v___x_3913_);
v___x_3915_ = lean_unsigned_to_nat(0u);
v___x_3916_ = lean_array_get_size(v_goals_3914_);
v___x_3917_ = lean_nat_dec_lt(v___x_3915_, v___x_3916_);
if (v___x_3917_ == 0)
{
lean_dec_ref(v_goals_3914_);
goto v___jp_3821_;
}
else
{
lean_object* v___x_3918_; uint8_t v___x_3919_; 
v___x_3918_ = lean_box(0);
v___x_3919_ = lean_nat_dec_le(v___x_3916_, v___x_3916_);
if (v___x_3919_ == 0)
{
if (v___x_3917_ == 0)
{
lean_dec_ref(v_goals_3914_);
goto v___jp_3821_;
}
else
{
size_t v___x_3920_; size_t v___x_3921_; lean_object* v___x_3922_; 
v___x_3920_ = ((size_t)0ULL);
v___x_3921_ = lean_usize_of_nat(v___x_3916_);
v___x_3922_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(v_val_3802_, v_goals_3914_, v___x_3920_, v___x_3921_, v___x_3918_, v___y_3804_, v___y_3805_, v___y_3806_, v___y_3807_, v___y_3808_, v___y_3809_, v___y_3810_);
lean_dec_ref(v_goals_3914_);
if (lean_obj_tag(v___x_3922_) == 0)
{
lean_dec_ref_known(v___x_3922_, 1);
goto v___jp_3821_;
}
else
{
return v___x_3922_;
}
}
}
else
{
size_t v___x_3923_; size_t v___x_3924_; lean_object* v___x_3925_; 
v___x_3923_ = ((size_t)0ULL);
v___x_3924_ = lean_usize_of_nat(v___x_3916_);
v___x_3925_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(v_val_3802_, v_goals_3914_, v___x_3923_, v___x_3924_, v___x_3918_, v___y_3804_, v___y_3805_, v___y_3806_, v___y_3807_, v___y_3808_, v___y_3809_, v___y_3810_);
lean_dec_ref(v_goals_3914_);
if (lean_obj_tag(v___x_3925_) == 0)
{
lean_dec_ref_known(v___x_3925_, 1);
goto v___jp_3821_;
}
else
{
return v___x_3925_;
}
}
}
}
}
v___jp_3812_:
{
lean_object* v___x_3813_; lean_object* v___x_3814_; 
v___x_3813_ = lean_box(0);
v___x_3814_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3814_, 0, v___x_3813_);
return v___x_3814_;
}
v___jp_3815_:
{
lean_object* v___x_3816_; lean_object* v___x_3817_; 
v___x_3816_ = lean_box(0);
v___x_3817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3817_, 0, v___x_3816_);
return v___x_3817_;
}
v___jp_3818_:
{
lean_object* v___x_3819_; lean_object* v___x_3820_; 
v___x_3819_ = lean_box(0);
v___x_3820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3820_, 0, v___x_3819_);
return v___x_3820_;
}
v___jp_3821_:
{
lean_object* v___x_3822_; lean_object* v___x_3823_; 
v___x_3822_ = lean_box(0);
v___x_3823_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3823_, 0, v___x_3822_);
return v___x_3823_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(lean_object* v_val_3926_, lean_object* v_as_3927_, size_t v_i_3928_, size_t v_stop_3929_, lean_object* v_b_3930_, lean_object* v___y_3931_, lean_object* v___y_3932_, lean_object* v___y_3933_, lean_object* v___y_3934_, lean_object* v___y_3935_, lean_object* v___y_3936_, lean_object* v___y_3937_){
_start:
{
uint8_t v___x_3939_; 
v___x_3939_ = lean_usize_dec_eq(v_i_3928_, v_stop_3929_);
if (v___x_3939_ == 0)
{
lean_object* v___x_3940_; lean_object* v___x_3941_; lean_object* v___x_3942_; 
v___x_3940_ = lean_array_uget_borrowed(v_as_3927_, v_i_3928_);
lean_inc(v___x_3940_);
v___x_3941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3941_, 0, v___x_3940_);
v___x_3942_ = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(v_val_3926_, v___x_3941_, v___y_3931_, v___y_3932_, v___y_3933_, v___y_3934_, v___y_3935_, v___y_3936_, v___y_3937_);
lean_dec_ref_known(v___x_3941_, 1);
if (lean_obj_tag(v___x_3942_) == 0)
{
lean_object* v_a_3943_; size_t v___x_3944_; size_t v___x_3945_; 
v_a_3943_ = lean_ctor_get(v___x_3942_, 0);
lean_inc(v_a_3943_);
lean_dec_ref_known(v___x_3942_, 1);
v___x_3944_ = ((size_t)1ULL);
v___x_3945_ = lean_usize_add(v_i_3928_, v___x_3944_);
v_i_3928_ = v___x_3945_;
v_b_3930_ = v_a_3943_;
goto _start;
}
else
{
return v___x_3942_;
}
}
else
{
lean_object* v___x_3947_; 
v___x_3947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3947_, 0, v_b_3930_);
return v___x_3947_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0___boxed(lean_object* v_val_3948_, lean_object* v_as_3949_, lean_object* v_i_3950_, lean_object* v_stop_3951_, lean_object* v_b_3952_, lean_object* v___y_3953_, lean_object* v___y_3954_, lean_object* v___y_3955_, lean_object* v___y_3956_, lean_object* v___y_3957_, lean_object* v___y_3958_, lean_object* v___y_3959_, lean_object* v___y_3960_){
_start:
{
size_t v_i_boxed_3961_; size_t v_stop_boxed_3962_; lean_object* v_res_3963_; 
v_i_boxed_3961_ = lean_unbox_usize(v_i_3950_);
lean_dec(v_i_3950_);
v_stop_boxed_3962_ = lean_unbox_usize(v_stop_3951_);
lean_dec(v_stop_3951_);
v_res_3963_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__0(v_val_3948_, v_as_3949_, v_i_boxed_3961_, v_stop_boxed_3962_, v_b_3952_, v___y_3953_, v___y_3954_, v___y_3955_, v___y_3956_, v___y_3957_, v___y_3958_, v___y_3959_);
lean_dec(v___y_3959_);
lean_dec_ref(v___y_3958_);
lean_dec(v___y_3957_);
lean_dec_ref(v___y_3956_);
lean_dec(v___y_3955_);
lean_dec(v___y_3954_);
lean_dec_ref(v___y_3953_);
lean_dec_ref(v_as_3949_);
lean_dec(v_val_3948_);
return v_res_3963_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1___boxed(lean_object* v_val_3964_, lean_object* v_as_3965_, lean_object* v_i_3966_, lean_object* v_stop_3967_, lean_object* v_b_3968_, lean_object* v___y_3969_, lean_object* v___y_3970_, lean_object* v___y_3971_, lean_object* v___y_3972_, lean_object* v___y_3973_, lean_object* v___y_3974_, lean_object* v___y_3975_, lean_object* v___y_3976_){
_start:
{
size_t v_i_boxed_3977_; size_t v_stop_boxed_3978_; lean_object* v_res_3979_; 
v_i_boxed_3977_ = lean_unbox_usize(v_i_3966_);
lean_dec(v_i_3966_);
v_stop_boxed_3978_ = lean_unbox_usize(v_stop_3967_);
lean_dec(v_stop_3967_);
v_res_3979_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__1(v_val_3964_, v_as_3965_, v_i_boxed_3977_, v_stop_boxed_3978_, v_b_3968_, v___y_3969_, v___y_3970_, v___y_3971_, v___y_3972_, v___y_3973_, v___y_3974_, v___y_3975_);
lean_dec(v___y_3975_);
lean_dec_ref(v___y_3974_);
lean_dec(v___y_3973_);
lean_dec_ref(v___y_3972_);
lean_dec(v___y_3971_);
lean_dec(v___y_3970_);
lean_dec_ref(v___y_3969_);
lean_dec_ref(v_as_3965_);
lean_dec(v_val_3964_);
return v_res_3979_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2___boxed(lean_object* v_val_3980_, lean_object* v_as_3981_, lean_object* v_i_3982_, lean_object* v_stop_3983_, lean_object* v_b_3984_, lean_object* v___y_3985_, lean_object* v___y_3986_, lean_object* v___y_3987_, lean_object* v___y_3988_, lean_object* v___y_3989_, lean_object* v___y_3990_, lean_object* v___y_3991_, lean_object* v___y_3992_){
_start:
{
size_t v_i_boxed_3993_; size_t v_stop_boxed_3994_; lean_object* v_res_3995_; 
v_i_boxed_3993_ = lean_unbox_usize(v_i_3982_);
lean_dec(v_i_3982_);
v_stop_boxed_3994_ = lean_unbox_usize(v_stop_3983_);
lean_dec(v_stop_3983_);
v_res_3995_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0_spec__2(v_val_3980_, v_as_3981_, v_i_boxed_3993_, v_stop_boxed_3994_, v_b_3984_, v___y_3985_, v___y_3986_, v___y_3987_, v___y_3988_, v___y_3989_, v___y_3990_, v___y_3991_);
lean_dec(v___y_3991_);
lean_dec_ref(v___y_3990_);
lean_dec(v___y_3989_);
lean_dec_ref(v___y_3988_);
lean_dec(v___y_3987_);
lean_dec(v___y_3986_);
lean_dec_ref(v___y_3985_);
lean_dec_ref(v_as_3981_);
lean_dec(v_val_3980_);
return v_res_3995_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0___boxed(lean_object* v_val_3996_, lean_object* v_x_3997_, lean_object* v___y_3998_, lean_object* v___y_3999_, lean_object* v___y_4000_, lean_object* v___y_4001_, lean_object* v___y_4002_, lean_object* v___y_4003_, lean_object* v___y_4004_, lean_object* v___y_4005_){
_start:
{
lean_object* v_res_4006_; 
v_res_4006_ = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(v_val_3996_, v_x_3997_, v___y_3998_, v___y_3999_, v___y_4000_, v___y_4001_, v___y_4002_, v___y_4003_, v___y_4004_);
lean_dec(v___y_4004_);
lean_dec_ref(v___y_4003_);
lean_dec(v___y_4002_);
lean_dec_ref(v___y_4001_);
lean_dec(v___y_4000_);
lean_dec(v___y_3999_);
lean_dec_ref(v___y_3998_);
lean_dec_ref(v_x_3997_);
lean_dec(v_val_3996_);
return v_res_4006_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeHasProgress(lean_object* v_a_4007_, lean_object* v_a_4008_, lean_object* v_a_4009_, lean_object* v_a_4010_, lean_object* v_a_4011_, lean_object* v_a_4012_, lean_object* v_a_4013_){
_start:
{
uint8_t v___x_4015_; lean_object* v___x_4016_; lean_object* v___x_4017_; lean_object* v___x_4018_; lean_object* v_root_4019_; lean_object* v___x_4020_; lean_object* v___x_4021_; 
v___x_4015_ = 0;
v___x_4016_ = lean_box(v___x_4015_);
v___x_4017_ = lean_st_mk_ref(v___x_4016_);
v___x_4018_ = lean_st_ref_get(v_a_4008_);
v_root_4019_ = lean_ctor_get(v___x_4018_, 0);
lean_inc(v_root_4019_);
lean_dec(v___x_4018_);
v___x_4020_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_4020_, 0, v_root_4019_);
v___x_4021_ = lp_aesop_Aesop_traverseDown___at___00Aesop_treeHasProgress_spec__0(v___x_4017_, v___x_4020_, v_a_4007_, v_a_4008_, v_a_4009_, v_a_4010_, v_a_4011_, v_a_4012_, v_a_4013_);
lean_dec_ref_known(v___x_4020_, 1);
if (lean_obj_tag(v___x_4021_) == 0)
{
lean_object* v___x_4023_; uint8_t v_isShared_4024_; uint8_t v_isSharedCheck_4029_; 
v_isSharedCheck_4029_ = !lean_is_exclusive(v___x_4021_);
if (v_isSharedCheck_4029_ == 0)
{
lean_object* v_unused_4030_; 
v_unused_4030_ = lean_ctor_get(v___x_4021_, 0);
lean_dec(v_unused_4030_);
v___x_4023_ = v___x_4021_;
v_isShared_4024_ = v_isSharedCheck_4029_;
goto v_resetjp_4022_;
}
else
{
lean_dec(v___x_4021_);
v___x_4023_ = lean_box(0);
v_isShared_4024_ = v_isSharedCheck_4029_;
goto v_resetjp_4022_;
}
v_resetjp_4022_:
{
lean_object* v___x_4025_; lean_object* v___x_4027_; 
v___x_4025_ = lean_st_ref_get(v___x_4017_);
lean_dec(v___x_4017_);
if (v_isShared_4024_ == 0)
{
lean_ctor_set(v___x_4023_, 0, v___x_4025_);
v___x_4027_ = v___x_4023_;
goto v_reusejp_4026_;
}
else
{
lean_object* v_reuseFailAlloc_4028_; 
v_reuseFailAlloc_4028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4028_, 0, v___x_4025_);
v___x_4027_ = v_reuseFailAlloc_4028_;
goto v_reusejp_4026_;
}
v_reusejp_4026_:
{
return v___x_4027_;
}
}
}
else
{
lean_object* v_a_4031_; lean_object* v___x_4033_; uint8_t v_isShared_4034_; uint8_t v_isSharedCheck_4038_; 
lean_dec(v___x_4017_);
v_a_4031_ = lean_ctor_get(v___x_4021_, 0);
v_isSharedCheck_4038_ = !lean_is_exclusive(v___x_4021_);
if (v_isSharedCheck_4038_ == 0)
{
v___x_4033_ = v___x_4021_;
v_isShared_4034_ = v_isSharedCheck_4038_;
goto v_resetjp_4032_;
}
else
{
lean_inc(v_a_4031_);
lean_dec(v___x_4021_);
v___x_4033_ = lean_box(0);
v_isShared_4034_ = v_isSharedCheck_4038_;
goto v_resetjp_4032_;
}
v_resetjp_4032_:
{
lean_object* v___x_4036_; 
if (v_isShared_4034_ == 0)
{
v___x_4036_ = v___x_4033_;
goto v_reusejp_4035_;
}
else
{
lean_object* v_reuseFailAlloc_4037_; 
v_reuseFailAlloc_4037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4037_, 0, v_a_4031_);
v___x_4036_ = v_reuseFailAlloc_4037_;
goto v_reusejp_4035_;
}
v_reusejp_4035_:
{
return v___x_4036_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_treeHasProgress___boxed(lean_object* v_a_4039_, lean_object* v_a_4040_, lean_object* v_a_4041_, lean_object* v_a_4042_, lean_object* v_a_4043_, lean_object* v_a_4044_, lean_object* v_a_4045_, lean_object* v_a_4046_){
_start:
{
lean_object* v_res_4047_; 
v_res_4047_ = lp_aesop_Aesop_treeHasProgress(v_a_4039_, v_a_4040_, v_a_4041_, v_a_4042_, v_a_4043_, v_a_4044_, v_a_4045_);
lean_dec(v_a_4045_);
lean_dec_ref(v_a_4044_);
lean_dec(v_a_4043_);
lean_dec_ref(v_a_4042_);
lean_dec(v_a_4041_);
lean_dec(v_a_4040_);
lean_dec_ref(v_a_4039_);
return v_res_4047_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg___lam__0(lean_object* v_a_4048_){
_start:
{
lean_object* v___x_4049_; 
v___x_4049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4049_, 0, v_a_4048_);
return v___x_4049_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__1(void){
_start:
{
lean_object* v___x_4051_; lean_object* v___x_4052_; 
v___x_4051_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__0));
v___x_4052_ = l_Lean_stringToMessageData(v___x_4051_);
return v___x_4052_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__3(void){
_start:
{
lean_object* v___x_4054_; lean_object* v___x_4055_; 
v___x_4054_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__2));
v___x_4055_ = l_Lean_stringToMessageData(v___x_4054_);
return v___x_4055_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__5(void){
_start:
{
lean_object* v___x_4057_; lean_object* v___x_4058_; 
v___x_4057_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__4));
v___x_4058_ = l_Lean_stringToMessageData(v___x_4057_);
return v___x_4058_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__9(void){
_start:
{
lean_object* v___x_4063_; lean_object* v___x_4064_; 
v___x_4063_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__8));
v___x_4064_ = l_Lean_MessageData_ofFormat(v___x_4063_);
return v___x_4064_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__11(void){
_start:
{
lean_object* v___x_4066_; lean_object* v___x_4067_; 
v___x_4066_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__10));
v___x_4067_ = l_Lean_stringToMessageData(v___x_4066_);
return v___x_4067_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__13(void){
_start:
{
lean_object* v___x_4069_; lean_object* v___x_4070_; 
v___x_4069_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__12));
v___x_4070_ = l_Lean_stringToMessageData(v___x_4069_);
return v___x_4070_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__15(void){
_start:
{
lean_object* v___x_4072_; lean_object* v___x_4073_; 
v___x_4072_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__14));
v___x_4073_ = l_Lean_stringToMessageData(v___x_4072_);
return v___x_4073_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__16(void){
_start:
{
lean_object* v___x_4074_; lean_object* v___x_4075_; 
v___x_4074_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___lam__3___closed__0));
v___x_4075_ = l_Lean_stringToMessageData(v___x_4074_);
return v___x_4075_;
}
}
static lean_object* _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__18(void){
_start:
{
lean_object* v___x_4077_; lean_object* v___x_4078_; 
v___x_4077_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__17));
v___x_4078_ = l_Lean_stringToMessageData(v___x_4077_);
return v___x_4078_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg(lean_object* v_inst_4079_, lean_object* v_mvarId_4080_, lean_object* v_remainingSafeGoals_4081_, uint8_t v_safePrefixExpansionSuccess_4082_, lean_object* v_msg_x3f_4083_, lean_object* v_a_4084_, lean_object* v_a_4085_, lean_object* v_a_4086_, lean_object* v_a_4087_, lean_object* v_a_4088_, lean_object* v_a_4089_, lean_object* v_a_4090_, lean_object* v_a_4091_){
_start:
{
lean_object* v___x_4093_; lean_object* v___x_4094_; lean_object* v___y_4096_; lean_object* v_options_4131_; lean_object* v___x_4132_; lean_object* v___x_4133_; uint8_t v___x_4134_; 
v___x_4093_ = l_Lean_KVMap_instValueBool;
v___x_4094_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_4079_);
v_options_4131_ = lean_ctor_get(v_a_4090_, 2);
v___x_4132_ = lp_aesop_Aesop_aesop_smallErrorMessages;
v___x_4133_ = l_Lean_Option_get___redArg(v___x_4093_, v_options_4131_, v___x_4132_);
v___x_4134_ = lean_unbox(v___x_4133_);
lean_dec(v___x_4133_);
if (v___x_4134_ == 0)
{
lean_object* v_options_4135_; lean_object* v_toOptions_4136_; lean_object* v_maxSafePrefixRuleApplications_4137_; lean_object* v___x_4138_; lean_object* v___x_4139_; uint8_t v___x_4140_; 
v_options_4135_ = lean_ctor_get(v_a_4084_, 2);
v_toOptions_4136_ = lean_ctor_get(v_options_4135_, 0);
v_maxSafePrefixRuleApplications_4137_ = lean_ctor_get(v_toOptions_4136_, 4);
v___x_4138_ = lean_array_get_size(v_remainingSafeGoals_4081_);
v___x_4139_ = lean_unsigned_to_nat(0u);
v___x_4140_ = lean_nat_dec_eq(v___x_4138_, v___x_4139_);
if (v___x_4140_ == 0)
{
lean_object* v___f_4141_; lean_object* v___x_4142_; lean_object* v___x_4143_; lean_object* v___x_4144_; lean_object* v___x_4145_; lean_object* v___x_4146_; lean_object* v___y_4148_; 
v___f_4141_ = ((lean_object*)(lp_aesop_Aesop_throwAesopEx___redArg___closed__6));
v___x_4142_ = lean_array_to_list(v_remainingSafeGoals_4081_);
v___x_4143_ = lean_box(0);
v___x_4144_ = l_List_mapTR_loop___redArg(v___f_4141_, v___x_4142_, v___x_4143_);
v___x_4145_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__9, &lp_aesop_Aesop_throwAesopEx___redArg___closed__9_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__9);
v___x_4146_ = l_Lean_MessageData_joinSep(v___x_4144_, v___x_4145_);
if (v_safePrefixExpansionSuccess_4082_ == 0)
{
lean_object* v___x_4153_; lean_object* v___x_4154_; lean_object* v___x_4155_; lean_object* v___x_4156_; lean_object* v___x_4157_; lean_object* v___x_4158_; lean_object* v___x_4159_; 
v___x_4153_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__13, &lp_aesop_Aesop_throwAesopEx___redArg___closed__13_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__13);
lean_inc(v_maxSafePrefixRuleApplications_4137_);
v___x_4154_ = l_Nat_reprFast(v_maxSafePrefixRuleApplications_4137_);
v___x_4155_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4155_, 0, v___x_4154_);
v___x_4156_ = l_Lean_MessageData_ofFormat(v___x_4155_);
v___x_4157_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4157_, 0, v___x_4153_);
lean_ctor_set(v___x_4157_, 1, v___x_4156_);
v___x_4158_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__15, &lp_aesop_Aesop_throwAesopEx___redArg___closed__15_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__15);
v___x_4159_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4159_, 0, v___x_4157_);
lean_ctor_set(v___x_4159_, 1, v___x_4158_);
v___y_4148_ = v___x_4159_;
goto v___jp_4147_;
}
else
{
lean_object* v___x_4160_; 
v___x_4160_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__16, &lp_aesop_Aesop_throwAesopEx___redArg___closed__16_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__16);
v___y_4148_ = v___x_4160_;
goto v___jp_4147_;
}
v___jp_4147_:
{
lean_object* v___x_4149_; lean_object* v___x_4150_; lean_object* v___x_4151_; lean_object* v___x_4152_; 
v___x_4149_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__11, &lp_aesop_Aesop_throwAesopEx___redArg___closed__11_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__11);
v___x_4150_ = l_Lean_indentD(v___x_4146_);
v___x_4151_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4151_, 0, v___x_4149_);
lean_ctor_set(v___x_4151_, 1, v___x_4150_);
v___x_4152_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4152_, 0, v___x_4151_);
lean_ctor_set(v___x_4152_, 1, v___y_4148_);
v___y_4096_ = v___x_4152_;
goto v___jp_4095_;
}
}
else
{
lean_object* v___x_4161_; 
lean_dec_ref(v_remainingSafeGoals_4081_);
v___x_4161_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__16, &lp_aesop_Aesop_throwAesopEx___redArg___closed__16_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__16);
v___y_4096_ = v___x_4161_;
goto v___jp_4095_;
}
}
else
{
lean_dec_ref(v_remainingSafeGoals_4081_);
lean_dec(v_mvarId_4080_);
if (lean_obj_tag(v_msg_x3f_4083_) == 0)
{
lean_object* v___x_4162_; lean_object* v___x_4163_; lean_object* v___f_4164_; lean_object* v___x_4165_; lean_object* v___x_4166_; lean_object* v___x_4167_; lean_object* v___x_5958__overap_4168_; lean_object* v___x_4169_; 
v___x_4162_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___x_4163_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_4079_);
v___f_4164_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
lean_inc_ref(v___x_4094_);
v___x_4165_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_4164_, v___x_4094_);
v___x_4166_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4166_, 0, v___x_4162_);
lean_ctor_set(v___x_4166_, 1, v___x_4163_);
lean_ctor_set(v___x_4166_, 2, v___x_4165_);
v___x_4167_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__18, &lp_aesop_Aesop_throwAesopEx___redArg___closed__18_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__18);
v___x_5958__overap_4168_ = l_Lean_throwError___redArg(v___x_4094_, v___x_4166_, v___x_4167_);
lean_inc(v_a_4091_);
lean_inc_ref(v_a_4090_);
lean_inc(v_a_4089_);
lean_inc_ref(v_a_4088_);
lean_inc(v_a_4087_);
lean_inc(v_a_4086_);
lean_inc(v_a_4085_);
lean_inc_ref(v_a_4084_);
v___x_4169_ = lean_apply_9(v___x_5958__overap_4168_, v_a_4084_, v_a_4085_, v_a_4086_, v_a_4087_, v_a_4088_, v_a_4089_, v_a_4090_, v_a_4091_, lean_box(0));
return v___x_4169_;
}
else
{
lean_object* v_val_4170_; lean_object* v___x_4171_; lean_object* v___x_4172_; lean_object* v___f_4173_; lean_object* v___x_4174_; lean_object* v___x_4175_; lean_object* v___x_4176_; lean_object* v___x_4177_; lean_object* v___x_5982__overap_4178_; lean_object* v___x_4179_; 
v_val_4170_ = lean_ctor_get(v_msg_x3f_4083_, 0);
lean_inc(v_val_4170_);
lean_dec_ref_known(v_msg_x3f_4083_, 1);
v___x_4171_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___x_4172_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_4079_);
v___f_4173_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
lean_inc_ref(v___x_4094_);
v___x_4174_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_4173_, v___x_4094_);
v___x_4175_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4175_, 0, v___x_4171_);
lean_ctor_set(v___x_4175_, 1, v___x_4172_);
lean_ctor_set(v___x_4175_, 2, v___x_4174_);
v___x_4176_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__3, &lp_aesop_Aesop_throwAesopEx___redArg___closed__3_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__3);
v___x_4177_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4177_, 0, v___x_4176_);
lean_ctor_set(v___x_4177_, 1, v_val_4170_);
v___x_5982__overap_4178_ = l_Lean_throwError___redArg(v___x_4094_, v___x_4175_, v___x_4177_);
lean_inc(v_a_4091_);
lean_inc_ref(v_a_4090_);
lean_inc(v_a_4089_);
lean_inc_ref(v_a_4088_);
lean_inc(v_a_4087_);
lean_inc(v_a_4086_);
lean_inc(v_a_4085_);
lean_inc_ref(v_a_4084_);
v___x_4179_ = lean_apply_9(v___x_5982__overap_4178_, v_a_4084_, v_a_4085_, v_a_4086_, v_a_4087_, v_a_4088_, v_a_4089_, v_a_4090_, v_a_4091_, lean_box(0));
return v___x_4179_;
}
}
v___jp_4095_:
{
if (lean_obj_tag(v_msg_x3f_4083_) == 0)
{
lean_object* v___x_4097_; lean_object* v___x_4098_; lean_object* v___f_4099_; lean_object* v___x_4100_; lean_object* v___x_4101_; lean_object* v___x_4102_; lean_object* v___x_4103_; lean_object* v___x_4104_; lean_object* v___x_4105_; lean_object* v___x_4106_; lean_object* v___x_5865__overap_4107_; lean_object* v___x_4108_; 
v___x_4097_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___x_4098_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_4079_);
v___f_4099_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
lean_inc_ref(v___x_4094_);
v___x_4100_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_4099_, v___x_4094_);
v___x_4101_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4101_, 0, v___x_4097_);
lean_ctor_set(v___x_4101_, 1, v___x_4098_);
lean_ctor_set(v___x_4101_, 2, v___x_4100_);
v___x_4102_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__1, &lp_aesop_Aesop_throwAesopEx___redArg___closed__1_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__1);
v___x_4103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4103_, 0, v_mvarId_4080_);
v___x_4104_ = l_Lean_indentD(v___x_4103_);
v___x_4105_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4105_, 0, v___x_4102_);
lean_ctor_set(v___x_4105_, 1, v___x_4104_);
v___x_4106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4106_, 0, v___x_4105_);
lean_ctor_set(v___x_4106_, 1, v___y_4096_);
v___x_5865__overap_4107_ = l_Lean_throwError___redArg(v___x_4094_, v___x_4101_, v___x_4106_);
lean_inc(v_a_4091_);
lean_inc_ref(v_a_4090_);
lean_inc(v_a_4089_);
lean_inc_ref(v_a_4088_);
lean_inc(v_a_4087_);
lean_inc(v_a_4086_);
lean_inc(v_a_4085_);
lean_inc_ref(v_a_4084_);
v___x_4108_ = lean_apply_9(v___x_5865__overap_4107_, v_a_4084_, v_a_4085_, v_a_4086_, v_a_4087_, v_a_4088_, v_a_4089_, v_a_4090_, v_a_4091_, lean_box(0));
return v___x_4108_;
}
else
{
lean_object* v_val_4109_; lean_object* v___x_4111_; uint8_t v_isShared_4112_; uint8_t v_isSharedCheck_4130_; 
v_val_4109_ = lean_ctor_get(v_msg_x3f_4083_, 0);
v_isSharedCheck_4130_ = !lean_is_exclusive(v_msg_x3f_4083_);
if (v_isSharedCheck_4130_ == 0)
{
v___x_4111_ = v_msg_x3f_4083_;
v_isShared_4112_ = v_isSharedCheck_4130_;
goto v_resetjp_4110_;
}
else
{
lean_inc(v_val_4109_);
lean_dec(v_msg_x3f_4083_);
v___x_4111_ = lean_box(0);
v_isShared_4112_ = v_isSharedCheck_4130_;
goto v_resetjp_4110_;
}
v_resetjp_4110_:
{
lean_object* v___x_4113_; lean_object* v___x_4114_; lean_object* v___f_4115_; lean_object* v___x_4116_; lean_object* v___x_4117_; lean_object* v___x_4118_; lean_object* v___x_4119_; lean_object* v___x_4120_; lean_object* v___x_4121_; lean_object* v___x_4123_; 
v___x_4113_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___x_4114_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_4079_);
v___f_4115_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
lean_inc_ref(v___x_4094_);
v___x_4116_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_4115_, v___x_4094_);
v___x_4117_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4117_, 0, v___x_4113_);
lean_ctor_set(v___x_4117_, 1, v___x_4114_);
lean_ctor_set(v___x_4117_, 2, v___x_4116_);
v___x_4118_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__3, &lp_aesop_Aesop_throwAesopEx___redArg___closed__3_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__3);
v___x_4119_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4119_, 0, v___x_4118_);
lean_ctor_set(v___x_4119_, 1, v_val_4109_);
v___x_4120_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__5, &lp_aesop_Aesop_throwAesopEx___redArg___closed__5_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__5);
v___x_4121_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4121_, 0, v___x_4119_);
lean_ctor_set(v___x_4121_, 1, v___x_4120_);
if (v_isShared_4112_ == 0)
{
lean_ctor_set(v___x_4111_, 0, v_mvarId_4080_);
v___x_4123_ = v___x_4111_;
goto v_reusejp_4122_;
}
else
{
lean_object* v_reuseFailAlloc_4129_; 
v_reuseFailAlloc_4129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4129_, 0, v_mvarId_4080_);
v___x_4123_ = v_reuseFailAlloc_4129_;
goto v_reusejp_4122_;
}
v_reusejp_4122_:
{
lean_object* v___x_4124_; lean_object* v___x_4125_; lean_object* v___x_4126_; lean_object* v___x_5896__overap_4127_; lean_object* v___x_4128_; 
v___x_4124_ = l_Lean_indentD(v___x_4123_);
v___x_4125_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4125_, 0, v___x_4121_);
lean_ctor_set(v___x_4125_, 1, v___x_4124_);
v___x_4126_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4126_, 0, v___x_4125_);
lean_ctor_set(v___x_4126_, 1, v___y_4096_);
v___x_5896__overap_4127_ = l_Lean_throwError___redArg(v___x_4094_, v___x_4117_, v___x_4126_);
lean_inc(v_a_4091_);
lean_inc_ref(v_a_4090_);
lean_inc(v_a_4089_);
lean_inc_ref(v_a_4088_);
lean_inc(v_a_4087_);
lean_inc(v_a_4086_);
lean_inc(v_a_4085_);
lean_inc_ref(v_a_4084_);
v___x_4128_ = lean_apply_9(v___x_5896__overap_4127_, v_a_4084_, v_a_4085_, v_a_4086_, v_a_4087_, v_a_4088_, v_a_4089_, v_a_4090_, v_a_4091_, lean_box(0));
return v___x_4128_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___redArg___boxed(lean_object* v_inst_4180_, lean_object* v_mvarId_4181_, lean_object* v_remainingSafeGoals_4182_, lean_object* v_safePrefixExpansionSuccess_4183_, lean_object* v_msg_x3f_4184_, lean_object* v_a_4185_, lean_object* v_a_4186_, lean_object* v_a_4187_, lean_object* v_a_4188_, lean_object* v_a_4189_, lean_object* v_a_4190_, lean_object* v_a_4191_, lean_object* v_a_4192_, lean_object* v_a_4193_){
_start:
{
uint8_t v_safePrefixExpansionSuccess_boxed_4194_; lean_object* v_res_4195_; 
v_safePrefixExpansionSuccess_boxed_4194_ = lean_unbox(v_safePrefixExpansionSuccess_4183_);
v_res_4195_ = lp_aesop_Aesop_throwAesopEx___redArg(v_inst_4180_, v_mvarId_4181_, v_remainingSafeGoals_4182_, v_safePrefixExpansionSuccess_boxed_4194_, v_msg_x3f_4184_, v_a_4185_, v_a_4186_, v_a_4187_, v_a_4188_, v_a_4189_, v_a_4190_, v_a_4191_, v_a_4192_);
lean_dec(v_a_4192_);
lean_dec_ref(v_a_4191_);
lean_dec(v_a_4190_);
lean_dec_ref(v_a_4189_);
lean_dec(v_a_4188_);
lean_dec(v_a_4187_);
lean_dec(v_a_4186_);
lean_dec_ref(v_a_4185_);
lean_dec_ref(v_inst_4180_);
return v_res_4195_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx(lean_object* v_Q_4196_, lean_object* v_inst_4197_, lean_object* v_00_u03b1_4198_, lean_object* v_mvarId_4199_, lean_object* v_remainingSafeGoals_4200_, uint8_t v_safePrefixExpansionSuccess_4201_, lean_object* v_msg_x3f_4202_, lean_object* v_a_4203_, lean_object* v_a_4204_, lean_object* v_a_4205_, lean_object* v_a_4206_, lean_object* v_a_4207_, lean_object* v_a_4208_, lean_object* v_a_4209_, lean_object* v_a_4210_){
_start:
{
lean_object* v___x_4212_; 
v___x_4212_ = lp_aesop_Aesop_throwAesopEx___redArg(v_inst_4197_, v_mvarId_4199_, v_remainingSafeGoals_4200_, v_safePrefixExpansionSuccess_4201_, v_msg_x3f_4202_, v_a_4203_, v_a_4204_, v_a_4205_, v_a_4206_, v_a_4207_, v_a_4208_, v_a_4209_, v_a_4210_);
return v___x_4212_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_throwAesopEx___boxed(lean_object* v_Q_4213_, lean_object* v_inst_4214_, lean_object* v_00_u03b1_4215_, lean_object* v_mvarId_4216_, lean_object* v_remainingSafeGoals_4217_, lean_object* v_safePrefixExpansionSuccess_4218_, lean_object* v_msg_x3f_4219_, lean_object* v_a_4220_, lean_object* v_a_4221_, lean_object* v_a_4222_, lean_object* v_a_4223_, lean_object* v_a_4224_, lean_object* v_a_4225_, lean_object* v_a_4226_, lean_object* v_a_4227_, lean_object* v_a_4228_){
_start:
{
uint8_t v_safePrefixExpansionSuccess_boxed_4229_; lean_object* v_res_4230_; 
v_safePrefixExpansionSuccess_boxed_4229_ = lean_unbox(v_safePrefixExpansionSuccess_4218_);
v_res_4230_ = lp_aesop_Aesop_throwAesopEx(v_Q_4213_, v_inst_4214_, v_00_u03b1_4215_, v_mvarId_4216_, v_remainingSafeGoals_4217_, v_safePrefixExpansionSuccess_boxed_4229_, v_msg_x3f_4219_, v_a_4220_, v_a_4221_, v_a_4222_, v_a_4223_, v_a_4224_, v_a_4225_, v_a_4226_, v_a_4227_);
lean_dec(v_a_4227_);
lean_dec_ref(v_a_4226_);
lean_dec(v_a_4225_);
lean_dec_ref(v_a_4224_);
lean_dec(v_a_4223_);
lean_dec(v_a_4222_);
lean_dec(v_a_4221_);
lean_dec_ref(v_a_4220_);
lean_dec_ref(v_inst_4214_);
return v_res_4230_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___lam__0(lean_object* v_x_4231_, lean_object* v___y_4232_, lean_object* v___y_4233_, lean_object* v___y_4234_, lean_object* v___y_4235_, lean_object* v___y_4236_, lean_object* v___y_4237_, lean_object* v___y_4238_, lean_object* v___y_4239_){
_start:
{
lean_object* v___x_4241_; lean_object* v___x_4242_; 
v___x_4241_ = lean_st_ref_get(v___y_4233_);
lean_dec(v___x_4241_);
v___x_4242_ = lp_aesop_Aesop_clearForwardImplDetailHyps(v_x_4231_, v___y_4236_, v___y_4237_, v___y_4238_, v___y_4239_);
return v___x_4242_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___lam__0___boxed(lean_object* v_x_4243_, lean_object* v___y_4244_, lean_object* v___y_4245_, lean_object* v___y_4246_, lean_object* v___y_4247_, lean_object* v___y_4248_, lean_object* v___y_4249_, lean_object* v___y_4250_, lean_object* v___y_4251_, lean_object* v___y_4252_){
_start:
{
lean_object* v_res_4253_; 
v_res_4253_ = lp_aesop_Aesop_handleNonfatalError___redArg___lam__0(v_x_4243_, v___y_4244_, v___y_4245_, v___y_4246_, v___y_4247_, v___y_4248_, v___y_4249_, v___y_4250_, v___y_4251_);
lean_dec(v___y_4251_);
lean_dec_ref(v___y_4250_);
lean_dec(v___y_4249_);
lean_dec_ref(v___y_4248_);
lean_dec(v___y_4247_);
lean_dec(v___y_4246_);
lean_dec(v___y_4245_);
lean_dec_ref(v___y_4244_);
return v_res_4253_;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__2(void){
_start:
{
lean_object* v___x_4256_; lean_object* v___x_4257_; 
v___x_4256_ = ((lean_object*)(lp_aesop_Aesop_handleNonfatalError___redArg___closed__1));
v___x_4257_ = l_Lean_stringToMessageData(v___x_4256_);
return v___x_4257_;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__4(void){
_start:
{
lean_object* v___x_4259_; lean_object* v___x_4260_; 
v___x_4259_ = ((lean_object*)(lp_aesop_Aesop_handleNonfatalError___redArg___closed__3));
v___x_4260_ = l_Lean_stringToMessageData(v___x_4259_);
return v___x_4260_;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__8(void){
_start:
{
lean_object* v___x_4266_; lean_object* v___x_4267_; 
v___x_4266_ = ((lean_object*)(lp_aesop_Aesop_handleNonfatalError___redArg___closed__7));
v___x_4267_ = l_Lean_MessageData_ofFormat(v___x_4266_);
return v___x_4267_;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__9(void){
_start:
{
lean_object* v___x_4268_; lean_object* v___x_4269_; 
v___x_4268_ = lean_obj_once(&lp_aesop_Aesop_handleNonfatalError___redArg___closed__8, &lp_aesop_Aesop_handleNonfatalError___redArg___closed__8_once, _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__8);
v___x_4269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4269_, 0, v___x_4268_);
return v___x_4269_;
}
}
static lean_object* _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__11(void){
_start:
{
lean_object* v___x_4271_; lean_object* v___x_4272_; 
v___x_4271_ = ((lean_object*)(lp_aesop_Aesop_handleNonfatalError___redArg___closed__10));
v___x_4272_ = l_Lean_stringToMessageData(v___x_4271_);
return v___x_4272_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg(lean_object* v_inst_4273_, lean_object* v_err_4274_, lean_object* v_a_4275_, lean_object* v_a_4276_, lean_object* v_a_4277_, lean_object* v_a_4278_, lean_object* v_a_4279_, lean_object* v_a_4280_, lean_object* v_a_4281_, lean_object* v_a_4282_){
_start:
{
lean_object* v___x_4284_; lean_object* v___x_4285_; lean_object* v___x_4286_; lean_object* v_toMonadOptions_4287_; lean_object* v___x_4288_; 
v___x_4284_ = l_Lean_KVMap_instValueBool;
v___x_4285_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_4273_);
v___x_4286_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_traceNewRapps___redArg___closed__2);
v_toMonadOptions_4287_ = lean_ctor_get(v___x_4286_, 0);
lean_inc_ref(v_inst_4273_);
v___x_4288_ = lp_aesop_Aesop_expandSafePrefix___redArg(v_inst_4273_, v_a_4275_, v_a_4276_, v_a_4277_, v_a_4278_, v_a_4279_, v_a_4280_, v_a_4281_, v_a_4282_);
if (lean_obj_tag(v___x_4288_) == 0)
{
lean_object* v_a_4289_; lean_object* v___x_4290_; lean_object* v_iteration_4291_; lean_object* v_ruleSet_4292_; lean_object* v___x_4293_; lean_object* v___x_4294_; 
v_a_4289_ = lean_ctor_get(v___x_4288_, 0);
lean_inc(v_a_4289_);
lean_dec_ref_known(v___x_4288_, 1);
v___x_4290_ = lean_st_ref_get(v_a_4276_);
v_iteration_4291_ = lean_ctor_get(v___x_4290_, 0);
lean_inc(v_iteration_4291_);
lean_dec(v___x_4290_);
v_ruleSet_4292_ = lean_ctor_get(v_a_4275_, 0);
lean_inc_ref(v_ruleSet_4292_);
v___x_4293_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4293_, 0, v_iteration_4291_);
lean_ctor_set(v___x_4293_, 1, v_ruleSet_4292_);
v___x_4294_ = lp_aesop_Aesop_extractSafePrefix(v___x_4293_, v_a_4277_, v_a_4278_, v_a_4279_, v_a_4280_, v_a_4281_, v_a_4282_);
lean_dec_ref_known(v___x_4293_, 2);
if (lean_obj_tag(v___x_4294_) == 0)
{
lean_object* v_a_4295_; lean_object* v___x_4297_; uint8_t v_isShared_4298_; uint8_t v_isSharedCheck_4584_; 
v_a_4295_ = lean_ctor_get(v___x_4294_, 0);
v_isSharedCheck_4584_ = !lean_is_exclusive(v___x_4294_);
if (v_isSharedCheck_4584_ == 0)
{
v___x_4297_ = v___x_4294_;
v_isShared_4298_ = v_isSharedCheck_4584_;
goto v_resetjp_4296_;
}
else
{
lean_inc(v_a_4295_);
lean_dec(v___x_4294_);
v___x_4297_ = lean_box(0);
v_isShared_4298_ = v_isSharedCheck_4584_;
goto v_resetjp_4296_;
}
v_resetjp_4296_:
{
lean_object* v___x_4299_; lean_object* v___x_42312__overap_4300_; lean_object* v___x_4301_; 
v___x_4299_ = lp_aesop_Aesop_TraceOption_proof;
lean_inc(v_toMonadOptions_4287_);
lean_inc_ref(v___x_4285_);
v___x_42312__overap_4300_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_4285_, v_toMonadOptions_4287_, v___x_4299_);
lean_inc(v_a_4282_);
lean_inc_ref(v_a_4281_);
lean_inc(v_a_4280_);
lean_inc_ref(v_a_4279_);
lean_inc(v_a_4278_);
lean_inc(v_a_4277_);
lean_inc(v_a_4276_);
lean_inc_ref(v_a_4275_);
v___x_4301_ = lean_apply_9(v___x_42312__overap_4300_, v_a_4275_, v_a_4276_, v_a_4277_, v_a_4278_, v_a_4279_, v_a_4280_, v_a_4281_, v_a_4282_, lean_box(0));
if (lean_obj_tag(v___x_4301_) == 0)
{
lean_object* v_a_4302_; lean_object* v___f_4303_; lean_object* v___y_4305_; lean_object* v___y_4306_; lean_object* v___y_4307_; lean_object* v___y_4308_; lean_object* v___y_4309_; lean_object* v___y_4310_; lean_object* v___y_4311_; lean_object* v___y_4312_; lean_object* v___y_4318_; lean_object* v___y_4319_; lean_object* v___y_4320_; lean_object* v___y_4321_; lean_object* v___y_4322_; lean_object* v___y_4323_; lean_object* v___y_4324_; lean_object* v___y_4325_; lean_object* v___y_4352_; lean_object* v___y_4353_; lean_object* v___y_4354_; lean_object* v___y_4355_; lean_object* v___y_4356_; lean_object* v___y_4357_; lean_object* v___y_4358_; lean_object* v___y_4359_; lean_object* v___y_4360_; lean_object* v___y_4382_; lean_object* v___y_4383_; lean_object* v___y_4384_; lean_object* v___y_4385_; lean_object* v___y_4386_; lean_object* v___y_4387_; lean_object* v___y_4388_; lean_object* v___y_4389_; lean_object* v___y_4390_; lean_object* v___y_4432_; lean_object* v___y_4433_; lean_object* v___y_4434_; lean_object* v___y_4435_; lean_object* v___y_4436_; lean_object* v___y_4437_; lean_object* v___y_4438_; lean_object* v___y_4439_; uint8_t v___x_4487_; 
v_a_4302_ = lean_ctor_get(v___x_4301_, 0);
lean_inc(v_a_4302_);
lean_dec_ref_known(v___x_4301_, 1);
v___f_4303_ = ((lean_object*)(lp_aesop_Aesop_handleNonfatalError___redArg___closed__0));
v___x_4487_ = lean_unbox(v_a_4302_);
lean_dec(v_a_4302_);
if (v___x_4487_ == 0)
{
v___y_4432_ = v_a_4275_;
v___y_4433_ = v_a_4276_;
v___y_4434_ = v_a_4277_;
v___y_4435_ = v_a_4278_;
v___y_4436_ = v_a_4279_;
v___y_4437_ = v_a_4280_;
v___y_4438_ = v_a_4281_;
v___y_4439_ = v_a_4282_;
goto v___jp_4431_;
}
else
{
lean_object* v___x_4488_; 
v___x_4488_ = lp_aesop_Aesop_getProof_x3f___redArg(v_inst_4273_, v_a_4275_, v_a_4276_, v_a_4277_, v_a_4278_, v_a_4279_, v_a_4280_, v_a_4281_, v_a_4282_);
if (lean_obj_tag(v___x_4488_) == 0)
{
lean_object* v_a_4489_; 
v_a_4489_ = lean_ctor_get(v___x_4488_, 0);
lean_inc(v_a_4489_);
lean_dec_ref_known(v___x_4488_, 1);
if (lean_obj_tag(v_a_4489_) == 0)
{
lean_object* v___x_4490_; lean_object* v___x_4491_; lean_object* v_traceClass_4492_; lean_object* v___f_4493_; lean_object* v___x_4494_; lean_object* v___x_42804__overap_4495_; lean_object* v___x_4496_; 
v___x_4490_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__5, &lp_aesop_Aesop_expandNextGoal___redArg___closed__5_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__5);
v___x_4491_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_4273_);
v_traceClass_4492_ = lean_ctor_get(v___x_4299_, 0);
v___f_4493_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
v___x_4494_ = lean_obj_once(&lp_aesop_Aesop_handleNonfatalError___redArg___closed__11, &lp_aesop_Aesop_handleNonfatalError___redArg___closed__11_once, _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__11);
lean_inc(v_traceClass_4492_);
lean_inc_ref(v___x_4285_);
v___x_42804__overap_4495_ = l_Lean_addTrace___redArg(v___x_4285_, v___x_4490_, v___x_4491_, v___f_4493_, v_traceClass_4492_, v___x_4494_);
lean_inc(v_a_4282_);
lean_inc_ref(v_a_4281_);
lean_inc(v_a_4280_);
lean_inc_ref(v_a_4279_);
lean_inc(v_a_4278_);
lean_inc(v_a_4277_);
lean_inc(v_a_4276_);
lean_inc_ref(v_a_4275_);
v___x_4496_ = lean_apply_9(v___x_42804__overap_4495_, v_a_4275_, v_a_4276_, v_a_4277_, v_a_4278_, v_a_4279_, v_a_4280_, v_a_4281_, v_a_4282_, lean_box(0));
if (lean_obj_tag(v___x_4496_) == 0)
{
lean_dec_ref_known(v___x_4496_, 1);
v___y_4432_ = v_a_4275_;
v___y_4433_ = v_a_4276_;
v___y_4434_ = v_a_4277_;
v___y_4435_ = v_a_4278_;
v___y_4436_ = v_a_4279_;
v___y_4437_ = v_a_4280_;
v___y_4438_ = v_a_4281_;
v___y_4439_ = v_a_4282_;
goto v___jp_4431_;
}
else
{
lean_object* v_a_4497_; lean_object* v___x_4499_; uint8_t v_isShared_4500_; uint8_t v_isSharedCheck_4504_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4497_ = lean_ctor_get(v___x_4496_, 0);
v_isSharedCheck_4504_ = !lean_is_exclusive(v___x_4496_);
if (v_isSharedCheck_4504_ == 0)
{
v___x_4499_ = v___x_4496_;
v_isShared_4500_ = v_isSharedCheck_4504_;
goto v_resetjp_4498_;
}
else
{
lean_inc(v_a_4497_);
lean_dec(v___x_4496_);
v___x_4499_ = lean_box(0);
v_isShared_4500_ = v_isSharedCheck_4504_;
goto v_resetjp_4498_;
}
v_resetjp_4498_:
{
lean_object* v___x_4502_; 
if (v_isShared_4500_ == 0)
{
v___x_4502_ = v___x_4499_;
goto v_reusejp_4501_;
}
else
{
lean_object* v_reuseFailAlloc_4503_; 
v_reuseFailAlloc_4503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4503_, 0, v_a_4497_);
v___x_4502_ = v_reuseFailAlloc_4503_;
goto v_reusejp_4501_;
}
v_reusejp_4501_:
{
return v___x_4502_;
}
}
}
}
else
{
lean_object* v_val_4505_; lean_object* v___x_4506_; lean_object* v_iteration_4507_; lean_object* v___x_4508_; lean_object* v___x_4509_; 
v_val_4505_ = lean_ctor_get(v_a_4489_, 0);
lean_inc(v_val_4505_);
lean_dec_ref_known(v_a_4489_, 1);
v___x_4506_ = lean_st_ref_get(v_a_4276_);
v_iteration_4507_ = lean_ctor_get(v___x_4506_, 0);
lean_inc(v_iteration_4507_);
lean_dec(v___x_4506_);
lean_inc_ref(v_ruleSet_4292_);
v___x_4508_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4508_, 0, v_iteration_4507_);
lean_ctor_set(v___x_4508_, 1, v_ruleSet_4292_);
v___x_4509_ = lp_aesop_Aesop_getRootMVarId(v___x_4508_, v_a_4277_, v_a_4278_, v_a_4279_, v_a_4280_, v_a_4281_, v_a_4282_);
lean_dec_ref_known(v___x_4508_, 2);
if (lean_obj_tag(v___x_4509_) == 0)
{
lean_object* v_a_4510_; lean_object* v___x_4511_; lean_object* v___x_4512_; lean_object* v___x_4513_; lean_object* v_toApplicative_4514_; lean_object* v_toFunctor_4515_; lean_object* v_toSeq_4516_; lean_object* v_toSeqLeft_4517_; lean_object* v_toSeqRight_4518_; lean_object* v___f_4519_; lean_object* v___f_4520_; lean_object* v___f_4521_; lean_object* v___f_4522_; lean_object* v___x_4523_; lean_object* v___f_4524_; lean_object* v___f_4525_; lean_object* v___f_4526_; lean_object* v___x_4527_; lean_object* v___x_4528_; lean_object* v___x_4529_; lean_object* v___x_4530_; lean_object* v___x_4531_; lean_object* v___f_4532_; lean_object* v___f_4533_; lean_object* v___x_4534_; lean_object* v___f_4535_; lean_object* v___f_4536_; lean_object* v___x_4537_; lean_object* v___f_4538_; lean_object* v___f_4539_; lean_object* v___x_4540_; lean_object* v___f_4541_; lean_object* v___f_4542_; lean_object* v___x_4543_; lean_object* v___x_4544_; lean_object* v___x_4545_; lean_object* v_traceClass_4546_; lean_object* v___f_4547_; lean_object* v___x_4548_; lean_object* v___x_4549_; lean_object* v___x_42852__overap_4550_; lean_object* v___x_4551_; 
v_a_4510_ = lean_ctor_get(v___x_4509_, 0);
lean_inc(v_a_4510_);
lean_dec_ref_known(v___x_4509_, 1);
v___x_4511_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__1, &lp_aesop_Aesop_finalizeProof___redArg___closed__1_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__1);
v___x_4512_ = lean_obj_once(&lp_aesop_Aesop_finalizeProof___redArg___closed__0, &lp_aesop_Aesop_finalizeProof___redArg___closed__0_once, _init_lp_aesop_Aesop_finalizeProof___redArg___closed__0);
v___x_4513_ = lean_obj_once(&lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1, &lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1_once, _init_lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__1);
v_toApplicative_4514_ = lean_ctor_get(v___x_4513_, 0);
v_toFunctor_4515_ = lean_ctor_get(v_toApplicative_4514_, 0);
v_toSeq_4516_ = lean_ctor_get(v_toApplicative_4514_, 2);
v_toSeqLeft_4517_ = lean_ctor_get(v_toApplicative_4514_, 3);
v_toSeqRight_4518_ = lean_ctor_get(v_toApplicative_4514_, 4);
v___f_4519_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__2));
v___f_4520_ = ((lean_object*)(lp_aesop___private_Aesop_Search_Main_0__Aesop_expandNextGoal_fmt___redArg___lam__0___closed__3));
lean_inc_ref_n(v_toFunctor_4515_, 2);
v___f_4521_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_4521_, 0, v_toFunctor_4515_);
v___f_4522_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4522_, 0, v_toFunctor_4515_);
v___x_4523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4523_, 0, v___f_4521_);
lean_ctor_set(v___x_4523_, 1, v___f_4522_);
lean_inc(v_toSeqRight_4518_);
v___f_4524_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_4524_, 0, v_toSeqRight_4518_);
lean_inc(v_toSeqLeft_4517_);
v___f_4525_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_4525_, 0, v_toSeqLeft_4517_);
lean_inc(v_toSeq_4516_);
v___f_4526_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4526_, 0, v_toSeq_4516_);
v___x_4527_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4527_, 0, v___x_4523_);
lean_ctor_set(v___x_4527_, 1, v___f_4519_);
lean_ctor_set(v___x_4527_, 2, v___f_4526_);
lean_ctor_set(v___x_4527_, 3, v___f_4525_);
lean_ctor_set(v___x_4527_, 4, v___f_4524_);
v___x_4528_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4528_, 0, v___x_4527_);
lean_ctor_set(v___x_4528_, 1, v___f_4520_);
v___x_4529_ = l_StateRefT_x27_instMonad___redArg(v___x_4528_);
v___x_4530_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_4530_, 0, lean_box(0));
lean_closure_set(v___x_4530_, 1, lean_box(0));
lean_closure_set(v___x_4530_, 2, v___x_4529_);
v___x_4531_ = l_instMonadControlTOfPure___redArg(v___x_4530_);
lean_inc_ref(v___x_4531_);
v___f_4532_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_4532_, 0, v___x_4512_);
lean_closure_set(v___f_4532_, 1, v___x_4531_);
v___f_4533_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_4533_, 0, v___x_4512_);
lean_closure_set(v___f_4533_, 1, v___x_4531_);
v___x_4534_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4534_, 0, v___f_4532_);
lean_ctor_set(v___x_4534_, 1, v___f_4533_);
lean_inc_ref(v___x_4534_);
v___f_4535_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_4535_, 0, v___x_4512_);
lean_closure_set(v___f_4535_, 1, v___x_4534_);
v___f_4536_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_4536_, 0, v___x_4512_);
lean_closure_set(v___f_4536_, 1, v___x_4534_);
v___x_4537_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4537_, 0, v___f_4535_);
lean_ctor_set(v___x_4537_, 1, v___f_4536_);
lean_inc_ref(v___x_4537_);
v___f_4538_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_4538_, 0, v___x_4512_);
lean_closure_set(v___f_4538_, 1, v___x_4537_);
v___f_4539_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_4539_, 0, v___x_4512_);
lean_closure_set(v___f_4539_, 1, v___x_4537_);
v___x_4540_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4540_, 0, v___f_4538_);
lean_ctor_set(v___x_4540_, 1, v___f_4539_);
lean_inc_ref(v___x_4540_);
v___f_4541_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_4541_, 0, v___x_4511_);
lean_closure_set(v___f_4541_, 1, v___x_4540_);
v___f_4542_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_4542_, 0, v___x_4511_);
lean_closure_set(v___f_4542_, 1, v___x_4540_);
v___x_4543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4543_, 0, v___f_4541_);
lean_ctor_set(v___x_4543_, 1, v___f_4542_);
v___x_4544_ = lean_obj_once(&lp_aesop_Aesop_expandNextGoal___redArg___closed__5, &lp_aesop_Aesop_expandNextGoal___redArg___closed__5_once, _init_lp_aesop_Aesop_expandNextGoal___redArg___closed__5);
v___x_4545_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_4273_);
v_traceClass_4546_ = lean_ctor_get(v___x_4299_, 0);
v___f_4547_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
v___x_4548_ = l_Lean_MessageData_ofExpr(v_val_4505_);
lean_inc(v_traceClass_4546_);
lean_inc_ref_n(v___x_4285_, 2);
v___x_4549_ = l_Lean_addTrace___redArg(v___x_4285_, v___x_4544_, v___x_4545_, v___f_4547_, v_traceClass_4546_, v___x_4548_);
v___x_42852__overap_4550_ = l_Lean_MVarId_withContext___redArg(v___x_4543_, v___x_4285_, v_a_4510_, v___x_4549_);
lean_inc(v_a_4282_);
lean_inc_ref(v_a_4281_);
lean_inc(v_a_4280_);
lean_inc_ref(v_a_4279_);
lean_inc(v_a_4278_);
lean_inc(v_a_4277_);
lean_inc(v_a_4276_);
lean_inc_ref(v_a_4275_);
v___x_4551_ = lean_apply_9(v___x_42852__overap_4550_, v_a_4275_, v_a_4276_, v_a_4277_, v_a_4278_, v_a_4279_, v_a_4280_, v_a_4281_, v_a_4282_, lean_box(0));
if (lean_obj_tag(v___x_4551_) == 0)
{
lean_dec_ref_known(v___x_4551_, 1);
v___y_4432_ = v_a_4275_;
v___y_4433_ = v_a_4276_;
v___y_4434_ = v_a_4277_;
v___y_4435_ = v_a_4278_;
v___y_4436_ = v_a_4279_;
v___y_4437_ = v_a_4280_;
v___y_4438_ = v_a_4281_;
v___y_4439_ = v_a_4282_;
goto v___jp_4431_;
}
else
{
lean_object* v_a_4552_; lean_object* v___x_4554_; uint8_t v_isShared_4555_; uint8_t v_isSharedCheck_4559_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4552_ = lean_ctor_get(v___x_4551_, 0);
v_isSharedCheck_4559_ = !lean_is_exclusive(v___x_4551_);
if (v_isSharedCheck_4559_ == 0)
{
v___x_4554_ = v___x_4551_;
v_isShared_4555_ = v_isSharedCheck_4559_;
goto v_resetjp_4553_;
}
else
{
lean_inc(v_a_4552_);
lean_dec(v___x_4551_);
v___x_4554_ = lean_box(0);
v_isShared_4555_ = v_isSharedCheck_4559_;
goto v_resetjp_4553_;
}
v_resetjp_4553_:
{
lean_object* v___x_4557_; 
if (v_isShared_4555_ == 0)
{
v___x_4557_ = v___x_4554_;
goto v_reusejp_4556_;
}
else
{
lean_object* v_reuseFailAlloc_4558_; 
v_reuseFailAlloc_4558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4558_, 0, v_a_4552_);
v___x_4557_ = v_reuseFailAlloc_4558_;
goto v_reusejp_4556_;
}
v_reusejp_4556_:
{
return v___x_4557_;
}
}
}
}
else
{
lean_object* v_a_4560_; lean_object* v___x_4562_; uint8_t v_isShared_4563_; uint8_t v_isSharedCheck_4567_; 
lean_dec(v_val_4505_);
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4560_ = lean_ctor_get(v___x_4509_, 0);
v_isSharedCheck_4567_ = !lean_is_exclusive(v___x_4509_);
if (v_isSharedCheck_4567_ == 0)
{
v___x_4562_ = v___x_4509_;
v_isShared_4563_ = v_isSharedCheck_4567_;
goto v_resetjp_4561_;
}
else
{
lean_inc(v_a_4560_);
lean_dec(v___x_4509_);
v___x_4562_ = lean_box(0);
v_isShared_4563_ = v_isSharedCheck_4567_;
goto v_resetjp_4561_;
}
v_resetjp_4561_:
{
lean_object* v___x_4565_; 
if (v_isShared_4563_ == 0)
{
v___x_4565_ = v___x_4562_;
goto v_reusejp_4564_;
}
else
{
lean_object* v_reuseFailAlloc_4566_; 
v_reuseFailAlloc_4566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4566_, 0, v_a_4560_);
v___x_4565_ = v_reuseFailAlloc_4566_;
goto v_reusejp_4564_;
}
v_reusejp_4564_:
{
return v___x_4565_;
}
}
}
}
}
else
{
lean_object* v_a_4568_; lean_object* v___x_4570_; uint8_t v_isShared_4571_; uint8_t v_isSharedCheck_4575_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4568_ = lean_ctor_get(v___x_4488_, 0);
v_isSharedCheck_4575_ = !lean_is_exclusive(v___x_4488_);
if (v_isSharedCheck_4575_ == 0)
{
v___x_4570_ = v___x_4488_;
v_isShared_4571_ = v_isSharedCheck_4575_;
goto v_resetjp_4569_;
}
else
{
lean_inc(v_a_4568_);
lean_dec(v___x_4488_);
v___x_4570_ = lean_box(0);
v_isShared_4571_ = v_isSharedCheck_4575_;
goto v_resetjp_4569_;
}
v_resetjp_4569_:
{
lean_object* v___x_4573_; 
if (v_isShared_4571_ == 0)
{
v___x_4573_ = v___x_4570_;
goto v_reusejp_4572_;
}
else
{
lean_object* v_reuseFailAlloc_4574_; 
v_reuseFailAlloc_4574_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4574_, 0, v_a_4568_);
v___x_4573_ = v_reuseFailAlloc_4574_;
goto v_reusejp_4572_;
}
v_reusejp_4572_:
{
return v___x_4573_;
}
}
}
}
v___jp_4304_:
{
size_t v_sz_4313_; size_t v___x_4314_; lean_object* v___x_43027__overap_4315_; lean_object* v___x_4316_; 
v_sz_4313_ = lean_array_size(v_a_4295_);
v___x_4314_ = ((size_t)0ULL);
v___x_43027__overap_4315_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_4285_, v___f_4303_, v_sz_4313_, v___x_4314_, v_a_4295_);
lean_inc(v___y_4312_);
lean_inc_ref(v___y_4311_);
lean_inc(v___y_4310_);
lean_inc_ref(v___y_4309_);
lean_inc(v___y_4308_);
lean_inc(v___y_4307_);
lean_inc(v___y_4306_);
lean_inc_ref(v___y_4305_);
v___x_4316_ = lean_apply_9(v___x_43027__overap_4315_, v___y_4305_, v___y_4306_, v___y_4307_, v___y_4308_, v___y_4309_, v___y_4310_, v___y_4311_, v___y_4312_, lean_box(0));
return v___x_4316_;
}
v___jp_4317_:
{
uint8_t v___x_4326_; 
v___x_4326_ = lean_unbox(v_a_4289_);
lean_dec(v_a_4289_);
if (v___x_4326_ == 0)
{
lean_object* v___x_4327_; lean_object* v_options_4328_; lean_object* v_toOptions_4329_; lean_object* v_maxSafePrefixRuleApplications_4330_; lean_object* v___f_4331_; lean_object* v___x_4332_; lean_object* v___x_4333_; lean_object* v___x_4335_; 
v___x_4327_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__10, &lp_aesop_Aesop_traceScript___redArg___closed__10_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__10);
v_options_4328_ = lean_ctor_get(v___y_4318_, 2);
v_toOptions_4329_ = lean_ctor_get(v_options_4328_, 0);
v_maxSafePrefixRuleApplications_4330_ = lean_ctor_get(v_toOptions_4329_, 4);
v___f_4331_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
v___x_4332_ = lean_obj_once(&lp_aesop_Aesop_handleNonfatalError___redArg___closed__2, &lp_aesop_Aesop_handleNonfatalError___redArg___closed__2_once, _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__2);
lean_inc(v_maxSafePrefixRuleApplications_4330_);
v___x_4333_ = l_Nat_reprFast(v_maxSafePrefixRuleApplications_4330_);
if (v_isShared_4298_ == 0)
{
lean_ctor_set_tag(v___x_4297_, 3);
lean_ctor_set(v___x_4297_, 0, v___x_4333_);
v___x_4335_ = v___x_4297_;
goto v_reusejp_4334_;
}
else
{
lean_object* v_reuseFailAlloc_4350_; 
v_reuseFailAlloc_4350_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4350_, 0, v___x_4333_);
v___x_4335_ = v_reuseFailAlloc_4350_;
goto v_reusejp_4334_;
}
v_reusejp_4334_:
{
lean_object* v___x_4336_; lean_object* v___x_4337_; lean_object* v___x_4338_; lean_object* v___x_4339_; lean_object* v___x_42984__overap_4340_; lean_object* v___x_4341_; 
v___x_4336_ = l_Lean_MessageData_ofFormat(v___x_4335_);
v___x_4337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4337_, 0, v___x_4332_);
lean_ctor_set(v___x_4337_, 1, v___x_4336_);
v___x_4338_ = lean_obj_once(&lp_aesop_Aesop_throwAesopEx___redArg___closed__15, &lp_aesop_Aesop_throwAesopEx___redArg___closed__15_once, _init_lp_aesop_Aesop_throwAesopEx___redArg___closed__15);
v___x_4339_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4339_, 0, v___x_4337_);
lean_ctor_set(v___x_4339_, 1, v___x_4338_);
lean_inc(v_toMonadOptions_4287_);
lean_inc_ref(v___x_4285_);
v___x_42984__overap_4340_ = l_Lean_logWarning___redArg(v___x_4285_, v___x_4327_, v___f_4331_, v_toMonadOptions_4287_, v___x_4339_);
lean_inc(v___y_4325_);
lean_inc_ref(v___y_4324_);
lean_inc(v___y_4323_);
lean_inc_ref(v___y_4322_);
lean_inc(v___y_4321_);
lean_inc(v___y_4320_);
lean_inc(v___y_4319_);
lean_inc_ref(v___y_4318_);
v___x_4341_ = lean_apply_9(v___x_42984__overap_4340_, v___y_4318_, v___y_4319_, v___y_4320_, v___y_4321_, v___y_4322_, v___y_4323_, v___y_4324_, v___y_4325_, lean_box(0));
if (lean_obj_tag(v___x_4341_) == 0)
{
lean_dec_ref_known(v___x_4341_, 1);
v___y_4305_ = v___y_4318_;
v___y_4306_ = v___y_4319_;
v___y_4307_ = v___y_4320_;
v___y_4308_ = v___y_4321_;
v___y_4309_ = v___y_4322_;
v___y_4310_ = v___y_4323_;
v___y_4311_ = v___y_4324_;
v___y_4312_ = v___y_4325_;
goto v___jp_4304_;
}
else
{
lean_object* v_a_4342_; lean_object* v___x_4344_; uint8_t v_isShared_4345_; uint8_t v_isSharedCheck_4349_; 
lean_dec(v_a_4295_);
lean_dec_ref(v___x_4285_);
v_a_4342_ = lean_ctor_get(v___x_4341_, 0);
v_isSharedCheck_4349_ = !lean_is_exclusive(v___x_4341_);
if (v_isSharedCheck_4349_ == 0)
{
v___x_4344_ = v___x_4341_;
v_isShared_4345_ = v_isSharedCheck_4349_;
goto v_resetjp_4343_;
}
else
{
lean_inc(v_a_4342_);
lean_dec(v___x_4341_);
v___x_4344_ = lean_box(0);
v_isShared_4345_ = v_isSharedCheck_4349_;
goto v_resetjp_4343_;
}
v_resetjp_4343_:
{
lean_object* v___x_4347_; 
if (v_isShared_4345_ == 0)
{
v___x_4347_ = v___x_4344_;
goto v_reusejp_4346_;
}
else
{
lean_object* v_reuseFailAlloc_4348_; 
v_reuseFailAlloc_4348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4348_, 0, v_a_4342_);
v___x_4347_ = v_reuseFailAlloc_4348_;
goto v_reusejp_4346_;
}
v_reusejp_4346_:
{
return v___x_4347_;
}
}
}
}
}
else
{
lean_del_object(v___x_4297_);
v___y_4305_ = v___y_4318_;
v___y_4306_ = v___y_4319_;
v___y_4307_ = v___y_4320_;
v___y_4308_ = v___y_4321_;
v___y_4309_ = v___y_4322_;
v___y_4310_ = v___y_4323_;
v___y_4311_ = v___y_4324_;
v___y_4312_ = v___y_4325_;
goto v___jp_4304_;
}
}
v___jp_4351_:
{
lean_object* v_toOptions_4361_; uint8_t v_warnOnNonterminal_4362_; 
v_toOptions_4361_ = lean_ctor_get(v___y_4352_, 0);
v_warnOnNonterminal_4362_ = lean_ctor_get_uint8(v_toOptions_4361_, sizeof(void*)*6 + 5);
if (v_warnOnNonterminal_4362_ == 0)
{
lean_dec_ref(v_err_4274_);
v___y_4318_ = v___y_4353_;
v___y_4319_ = v___y_4354_;
v___y_4320_ = v___y_4355_;
v___y_4321_ = v___y_4356_;
v___y_4322_ = v___y_4357_;
v___y_4323_ = v___y_4358_;
v___y_4324_ = v___y_4359_;
v___y_4325_ = v___y_4360_;
goto v___jp_4317_;
}
else
{
lean_object* v_options_4363_; lean_object* v___x_4364_; lean_object* v___x_4365_; uint8_t v___x_4366_; 
v_options_4363_ = lean_ctor_get(v___y_4359_, 2);
v___x_4364_ = lp_aesop_Aesop_aesop_warn_nonterminal;
v___x_4365_ = l_Lean_Option_get___redArg(v___x_4284_, v_options_4363_, v___x_4364_);
v___x_4366_ = lean_unbox(v___x_4365_);
lean_dec(v___x_4365_);
if (v___x_4366_ == 0)
{
lean_dec_ref(v_err_4274_);
v___y_4318_ = v___y_4353_;
v___y_4319_ = v___y_4354_;
v___y_4320_ = v___y_4355_;
v___y_4321_ = v___y_4356_;
v___y_4322_ = v___y_4357_;
v___y_4323_ = v___y_4358_;
v___y_4324_ = v___y_4359_;
v___y_4325_ = v___y_4360_;
goto v___jp_4317_;
}
else
{
lean_object* v___x_4367_; lean_object* v___f_4368_; lean_object* v___x_4369_; lean_object* v___x_4370_; lean_object* v___x_42951__overap_4371_; lean_object* v___x_4372_; 
v___x_4367_ = lean_obj_once(&lp_aesop_Aesop_traceScript___redArg___closed__10, &lp_aesop_Aesop_traceScript___redArg___closed__10_once, _init_lp_aesop_Aesop_traceScript___redArg___closed__10);
v___f_4368_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
v___x_4369_ = lean_obj_once(&lp_aesop_Aesop_handleNonfatalError___redArg___closed__4, &lp_aesop_Aesop_handleNonfatalError___redArg___closed__4_once, _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__4);
v___x_4370_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4370_, 0, v___x_4369_);
lean_ctor_set(v___x_4370_, 1, v_err_4274_);
lean_inc(v_toMonadOptions_4287_);
lean_inc_ref(v___x_4285_);
v___x_42951__overap_4371_ = l_Lean_logWarning___redArg(v___x_4285_, v___x_4367_, v___f_4368_, v_toMonadOptions_4287_, v___x_4370_);
lean_inc(v___y_4360_);
lean_inc_ref(v___y_4359_);
lean_inc(v___y_4358_);
lean_inc_ref(v___y_4357_);
lean_inc(v___y_4356_);
lean_inc(v___y_4355_);
lean_inc(v___y_4354_);
lean_inc_ref(v___y_4353_);
v___x_4372_ = lean_apply_9(v___x_42951__overap_4371_, v___y_4353_, v___y_4354_, v___y_4355_, v___y_4356_, v___y_4357_, v___y_4358_, v___y_4359_, v___y_4360_, lean_box(0));
if (lean_obj_tag(v___x_4372_) == 0)
{
lean_dec_ref_known(v___x_4372_, 1);
v___y_4318_ = v___y_4353_;
v___y_4319_ = v___y_4354_;
v___y_4320_ = v___y_4355_;
v___y_4321_ = v___y_4356_;
v___y_4322_ = v___y_4357_;
v___y_4323_ = v___y_4358_;
v___y_4324_ = v___y_4359_;
v___y_4325_ = v___y_4360_;
goto v___jp_4317_;
}
else
{
lean_object* v_a_4373_; lean_object* v___x_4375_; uint8_t v_isShared_4376_; uint8_t v_isSharedCheck_4380_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
v_a_4373_ = lean_ctor_get(v___x_4372_, 0);
v_isSharedCheck_4380_ = !lean_is_exclusive(v___x_4372_);
if (v_isSharedCheck_4380_ == 0)
{
v___x_4375_ = v___x_4372_;
v_isShared_4376_ = v_isSharedCheck_4380_;
goto v_resetjp_4374_;
}
else
{
lean_inc(v_a_4373_);
lean_dec(v___x_4372_);
v___x_4375_ = lean_box(0);
v_isShared_4376_ = v_isSharedCheck_4380_;
goto v_resetjp_4374_;
}
v_resetjp_4374_:
{
lean_object* v___x_4378_; 
if (v_isShared_4376_ == 0)
{
v___x_4378_ = v___x_4375_;
goto v_reusejp_4377_;
}
else
{
lean_object* v_reuseFailAlloc_4379_; 
v_reuseFailAlloc_4379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4379_, 0, v_a_4373_);
v___x_4378_ = v_reuseFailAlloc_4379_;
goto v_reusejp_4377_;
}
v_reusejp_4377_:
{
return v___x_4378_;
}
}
}
}
}
}
v___jp_4381_:
{
lean_object* v___x_4391_; lean_object* v_iteration_4392_; lean_object* v_ruleSet_4393_; lean_object* v___x_4394_; lean_object* v___x_4395_; 
v___x_4391_ = lean_st_ref_get(v___y_4384_);
v_iteration_4392_ = lean_ctor_get(v___x_4391_, 0);
lean_inc(v_iteration_4392_);
lean_dec(v___x_4391_);
v_ruleSet_4393_ = lean_ctor_get(v___y_4383_, 0);
lean_inc_ref(v_ruleSet_4393_);
v___x_4394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4394_, 0, v_iteration_4392_);
lean_ctor_set(v___x_4394_, 1, v_ruleSet_4393_);
v___x_4395_ = lp_aesop_Aesop_treeHasProgress(v___x_4394_, v___y_4385_, v___y_4386_, v___y_4387_, v___y_4388_, v___y_4389_, v___y_4390_);
lean_dec_ref_known(v___x_4394_, 2);
if (lean_obj_tag(v___x_4395_) == 0)
{
lean_object* v_a_4396_; uint8_t v___x_4397_; 
v_a_4396_ = lean_ctor_get(v___x_4395_, 0);
lean_inc(v_a_4396_);
lean_dec_ref_known(v___x_4395_, 1);
v___x_4397_ = lean_unbox(v_a_4396_);
lean_dec(v_a_4396_);
if (v___x_4397_ == 0)
{
lean_object* v___x_4398_; lean_object* v_iteration_4399_; lean_object* v___x_4400_; lean_object* v___x_4401_; 
v___x_4398_ = lean_st_ref_get(v___y_4384_);
v_iteration_4399_ = lean_ctor_get(v___x_4398_, 0);
lean_inc(v_iteration_4399_);
lean_dec(v___x_4398_);
lean_inc_ref(v_ruleSet_4393_);
v___x_4400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4400_, 0, v_iteration_4399_);
lean_ctor_set(v___x_4400_, 1, v_ruleSet_4393_);
v___x_4401_ = lp_aesop_Aesop_getRootMVarId(v___x_4400_, v___y_4385_, v___y_4386_, v___y_4387_, v___y_4388_, v___y_4389_, v___y_4390_);
lean_dec_ref_known(v___x_4400_, 2);
if (lean_obj_tag(v___x_4401_) == 0)
{
lean_object* v_a_4402_; lean_object* v___x_4403_; lean_object* v___x_4404_; uint8_t v___x_4405_; lean_object* v___x_4406_; 
v_a_4402_ = lean_ctor_get(v___x_4401_, 0);
lean_inc(v_a_4402_);
lean_dec_ref_known(v___x_4401_, 1);
v___x_4403_ = ((lean_object*)(lp_aesop_Aesop_handleNonfatalError___redArg___closed__5));
v___x_4404_ = lean_obj_once(&lp_aesop_Aesop_handleNonfatalError___redArg___closed__9, &lp_aesop_Aesop_handleNonfatalError___redArg___closed__9_once, _init_lp_aesop_Aesop_handleNonfatalError___redArg___closed__9);
v___x_4405_ = lean_unbox(v_a_4289_);
v___x_4406_ = lp_aesop_Aesop_throwAesopEx___redArg(v_inst_4273_, v_a_4402_, v___x_4403_, v___x_4405_, v___x_4404_, v___y_4383_, v___y_4384_, v___y_4385_, v___y_4386_, v___y_4387_, v___y_4388_, v___y_4389_, v___y_4390_);
lean_dec_ref(v_inst_4273_);
if (lean_obj_tag(v___x_4406_) == 0)
{
lean_dec_ref_known(v___x_4406_, 1);
v___y_4352_ = v___y_4382_;
v___y_4353_ = v___y_4383_;
v___y_4354_ = v___y_4384_;
v___y_4355_ = v___y_4385_;
v___y_4356_ = v___y_4386_;
v___y_4357_ = v___y_4387_;
v___y_4358_ = v___y_4388_;
v___y_4359_ = v___y_4389_;
v___y_4360_ = v___y_4390_;
goto v___jp_4351_;
}
else
{
lean_object* v_a_4407_; lean_object* v___x_4409_; uint8_t v_isShared_4410_; uint8_t v_isSharedCheck_4414_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
v_a_4407_ = lean_ctor_get(v___x_4406_, 0);
v_isSharedCheck_4414_ = !lean_is_exclusive(v___x_4406_);
if (v_isSharedCheck_4414_ == 0)
{
v___x_4409_ = v___x_4406_;
v_isShared_4410_ = v_isSharedCheck_4414_;
goto v_resetjp_4408_;
}
else
{
lean_inc(v_a_4407_);
lean_dec(v___x_4406_);
v___x_4409_ = lean_box(0);
v_isShared_4410_ = v_isSharedCheck_4414_;
goto v_resetjp_4408_;
}
v_resetjp_4408_:
{
lean_object* v___x_4412_; 
if (v_isShared_4410_ == 0)
{
v___x_4412_ = v___x_4409_;
goto v_reusejp_4411_;
}
else
{
lean_object* v_reuseFailAlloc_4413_; 
v_reuseFailAlloc_4413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4413_, 0, v_a_4407_);
v___x_4412_ = v_reuseFailAlloc_4413_;
goto v_reusejp_4411_;
}
v_reusejp_4411_:
{
return v___x_4412_;
}
}
}
}
else
{
lean_object* v_a_4415_; lean_object* v___x_4417_; uint8_t v_isShared_4418_; uint8_t v_isSharedCheck_4422_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4415_ = lean_ctor_get(v___x_4401_, 0);
v_isSharedCheck_4422_ = !lean_is_exclusive(v___x_4401_);
if (v_isSharedCheck_4422_ == 0)
{
v___x_4417_ = v___x_4401_;
v_isShared_4418_ = v_isSharedCheck_4422_;
goto v_resetjp_4416_;
}
else
{
lean_inc(v_a_4415_);
lean_dec(v___x_4401_);
v___x_4417_ = lean_box(0);
v_isShared_4418_ = v_isSharedCheck_4422_;
goto v_resetjp_4416_;
}
v_resetjp_4416_:
{
lean_object* v___x_4420_; 
if (v_isShared_4418_ == 0)
{
v___x_4420_ = v___x_4417_;
goto v_reusejp_4419_;
}
else
{
lean_object* v_reuseFailAlloc_4421_; 
v_reuseFailAlloc_4421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4421_, 0, v_a_4415_);
v___x_4420_ = v_reuseFailAlloc_4421_;
goto v_reusejp_4419_;
}
v_reusejp_4419_:
{
return v___x_4420_;
}
}
}
}
else
{
lean_dec_ref(v_inst_4273_);
v___y_4352_ = v___y_4382_;
v___y_4353_ = v___y_4383_;
v___y_4354_ = v___y_4384_;
v___y_4355_ = v___y_4385_;
v___y_4356_ = v___y_4386_;
v___y_4357_ = v___y_4387_;
v___y_4358_ = v___y_4388_;
v___y_4359_ = v___y_4389_;
v___y_4360_ = v___y_4390_;
goto v___jp_4351_;
}
}
else
{
lean_object* v_a_4423_; lean_object* v___x_4425_; uint8_t v_isShared_4426_; uint8_t v_isSharedCheck_4430_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4423_ = lean_ctor_get(v___x_4395_, 0);
v_isSharedCheck_4430_ = !lean_is_exclusive(v___x_4395_);
if (v_isSharedCheck_4430_ == 0)
{
v___x_4425_ = v___x_4395_;
v_isShared_4426_ = v_isSharedCheck_4430_;
goto v_resetjp_4424_;
}
else
{
lean_inc(v_a_4423_);
lean_dec(v___x_4395_);
v___x_4425_ = lean_box(0);
v_isShared_4426_ = v_isSharedCheck_4430_;
goto v_resetjp_4424_;
}
v_resetjp_4424_:
{
lean_object* v___x_4428_; 
if (v_isShared_4426_ == 0)
{
v___x_4428_ = v___x_4425_;
goto v_reusejp_4427_;
}
else
{
lean_object* v_reuseFailAlloc_4429_; 
v_reuseFailAlloc_4429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4429_, 0, v_a_4423_);
v___x_4428_ = v_reuseFailAlloc_4429_;
goto v_reusejp_4427_;
}
v_reusejp_4427_:
{
return v___x_4428_;
}
}
}
}
v___jp_4431_:
{
lean_object* v___x_4440_; 
v___x_4440_ = lp_aesop_Aesop_traceTree___redArg(v___y_4432_, v___y_4433_, v___y_4434_, v___y_4435_, v___y_4436_, v___y_4437_, v___y_4438_, v___y_4439_);
if (lean_obj_tag(v___x_4440_) == 0)
{
uint8_t v___x_4441_; lean_object* v___x_4442_; 
lean_dec_ref_known(v___x_4440_, 1);
v___x_4441_ = 0;
lean_inc_ref(v_inst_4273_);
v___x_4442_ = lp_aesop_Aesop_traceScript___redArg(v_inst_4273_, v___x_4441_, v___y_4432_, v___y_4433_, v___y_4434_, v___y_4435_, v___y_4436_, v___y_4437_, v___y_4438_, v___y_4439_);
if (lean_obj_tag(v___x_4442_) == 0)
{
lean_object* v_options_4443_; lean_object* v_toOptions_4444_; uint8_t v_terminal_4445_; 
lean_dec_ref_known(v___x_4442_, 1);
v_options_4443_ = lean_ctor_get(v___y_4432_, 2);
v_toOptions_4444_ = lean_ctor_get(v_options_4443_, 0);
v_terminal_4445_ = lean_ctor_get_uint8(v_toOptions_4444_, sizeof(void*)*6 + 4);
if (v_terminal_4445_ == 0)
{
v___y_4382_ = v_options_4443_;
v___y_4383_ = v___y_4432_;
v___y_4384_ = v___y_4433_;
v___y_4385_ = v___y_4434_;
v___y_4386_ = v___y_4435_;
v___y_4387_ = v___y_4436_;
v___y_4388_ = v___y_4437_;
v___y_4389_ = v___y_4438_;
v___y_4390_ = v___y_4439_;
goto v___jp_4381_;
}
else
{
lean_object* v_ruleSet_4446_; lean_object* v___x_4447_; lean_object* v_iteration_4448_; lean_object* v___x_4449_; lean_object* v___x_4450_; 
v_ruleSet_4446_ = lean_ctor_get(v___y_4432_, 0);
v___x_4447_ = lean_st_ref_get(v___y_4433_);
v_iteration_4448_ = lean_ctor_get(v___x_4447_, 0);
lean_inc(v_iteration_4448_);
lean_dec(v___x_4447_);
lean_inc_ref(v_ruleSet_4446_);
v___x_4449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4449_, 0, v_iteration_4448_);
lean_ctor_set(v___x_4449_, 1, v_ruleSet_4446_);
v___x_4450_ = lp_aesop_Aesop_getRootMVarId(v___x_4449_, v___y_4434_, v___y_4435_, v___y_4436_, v___y_4437_, v___y_4438_, v___y_4439_);
lean_dec_ref_known(v___x_4449_, 2);
if (lean_obj_tag(v___x_4450_) == 0)
{
lean_object* v_a_4451_; lean_object* v___x_4452_; uint8_t v___x_4453_; lean_object* v___x_4454_; 
v_a_4451_ = lean_ctor_get(v___x_4450_, 0);
lean_inc(v_a_4451_);
lean_dec_ref_known(v___x_4450_, 1);
lean_inc_ref(v_err_4274_);
v___x_4452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4452_, 0, v_err_4274_);
v___x_4453_ = lean_unbox(v_a_4289_);
lean_inc(v_a_4295_);
v___x_4454_ = lp_aesop_Aesop_throwAesopEx___redArg(v_inst_4273_, v_a_4451_, v_a_4295_, v___x_4453_, v___x_4452_, v___y_4432_, v___y_4433_, v___y_4434_, v___y_4435_, v___y_4436_, v___y_4437_, v___y_4438_, v___y_4439_);
if (lean_obj_tag(v___x_4454_) == 0)
{
lean_dec_ref_known(v___x_4454_, 1);
v___y_4382_ = v_options_4443_;
v___y_4383_ = v___y_4432_;
v___y_4384_ = v___y_4433_;
v___y_4385_ = v___y_4434_;
v___y_4386_ = v___y_4435_;
v___y_4387_ = v___y_4436_;
v___y_4388_ = v___y_4437_;
v___y_4389_ = v___y_4438_;
v___y_4390_ = v___y_4439_;
goto v___jp_4381_;
}
else
{
lean_object* v_a_4455_; lean_object* v___x_4457_; uint8_t v_isShared_4458_; uint8_t v_isSharedCheck_4462_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4455_ = lean_ctor_get(v___x_4454_, 0);
v_isSharedCheck_4462_ = !lean_is_exclusive(v___x_4454_);
if (v_isSharedCheck_4462_ == 0)
{
v___x_4457_ = v___x_4454_;
v_isShared_4458_ = v_isSharedCheck_4462_;
goto v_resetjp_4456_;
}
else
{
lean_inc(v_a_4455_);
lean_dec(v___x_4454_);
v___x_4457_ = lean_box(0);
v_isShared_4458_ = v_isSharedCheck_4462_;
goto v_resetjp_4456_;
}
v_resetjp_4456_:
{
lean_object* v___x_4460_; 
if (v_isShared_4458_ == 0)
{
v___x_4460_ = v___x_4457_;
goto v_reusejp_4459_;
}
else
{
lean_object* v_reuseFailAlloc_4461_; 
v_reuseFailAlloc_4461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4461_, 0, v_a_4455_);
v___x_4460_ = v_reuseFailAlloc_4461_;
goto v_reusejp_4459_;
}
v_reusejp_4459_:
{
return v___x_4460_;
}
}
}
}
else
{
lean_object* v_a_4463_; lean_object* v___x_4465_; uint8_t v_isShared_4466_; uint8_t v_isSharedCheck_4470_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4463_ = lean_ctor_get(v___x_4450_, 0);
v_isSharedCheck_4470_ = !lean_is_exclusive(v___x_4450_);
if (v_isSharedCheck_4470_ == 0)
{
v___x_4465_ = v___x_4450_;
v_isShared_4466_ = v_isSharedCheck_4470_;
goto v_resetjp_4464_;
}
else
{
lean_inc(v_a_4463_);
lean_dec(v___x_4450_);
v___x_4465_ = lean_box(0);
v_isShared_4466_ = v_isSharedCheck_4470_;
goto v_resetjp_4464_;
}
v_resetjp_4464_:
{
lean_object* v___x_4468_; 
if (v_isShared_4466_ == 0)
{
v___x_4468_ = v___x_4465_;
goto v_reusejp_4467_;
}
else
{
lean_object* v_reuseFailAlloc_4469_; 
v_reuseFailAlloc_4469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4469_, 0, v_a_4463_);
v___x_4468_ = v_reuseFailAlloc_4469_;
goto v_reusejp_4467_;
}
v_reusejp_4467_:
{
return v___x_4468_;
}
}
}
}
}
else
{
lean_object* v_a_4471_; lean_object* v___x_4473_; uint8_t v_isShared_4474_; uint8_t v_isSharedCheck_4478_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4471_ = lean_ctor_get(v___x_4442_, 0);
v_isSharedCheck_4478_ = !lean_is_exclusive(v___x_4442_);
if (v_isSharedCheck_4478_ == 0)
{
v___x_4473_ = v___x_4442_;
v_isShared_4474_ = v_isSharedCheck_4478_;
goto v_resetjp_4472_;
}
else
{
lean_inc(v_a_4471_);
lean_dec(v___x_4442_);
v___x_4473_ = lean_box(0);
v_isShared_4474_ = v_isSharedCheck_4478_;
goto v_resetjp_4472_;
}
v_resetjp_4472_:
{
lean_object* v___x_4476_; 
if (v_isShared_4474_ == 0)
{
v___x_4476_ = v___x_4473_;
goto v_reusejp_4475_;
}
else
{
lean_object* v_reuseFailAlloc_4477_; 
v_reuseFailAlloc_4477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4477_, 0, v_a_4471_);
v___x_4476_ = v_reuseFailAlloc_4477_;
goto v_reusejp_4475_;
}
v_reusejp_4475_:
{
return v___x_4476_;
}
}
}
}
else
{
lean_object* v_a_4479_; lean_object* v___x_4481_; uint8_t v_isShared_4482_; uint8_t v_isSharedCheck_4486_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4479_ = lean_ctor_get(v___x_4440_, 0);
v_isSharedCheck_4486_ = !lean_is_exclusive(v___x_4440_);
if (v_isSharedCheck_4486_ == 0)
{
v___x_4481_ = v___x_4440_;
v_isShared_4482_ = v_isSharedCheck_4486_;
goto v_resetjp_4480_;
}
else
{
lean_inc(v_a_4479_);
lean_dec(v___x_4440_);
v___x_4481_ = lean_box(0);
v_isShared_4482_ = v_isSharedCheck_4486_;
goto v_resetjp_4480_;
}
v_resetjp_4480_:
{
lean_object* v___x_4484_; 
if (v_isShared_4482_ == 0)
{
v___x_4484_ = v___x_4481_;
goto v_reusejp_4483_;
}
else
{
lean_object* v_reuseFailAlloc_4485_; 
v_reuseFailAlloc_4485_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4485_, 0, v_a_4479_);
v___x_4484_ = v_reuseFailAlloc_4485_;
goto v_reusejp_4483_;
}
v_reusejp_4483_:
{
return v___x_4484_;
}
}
}
}
}
else
{
lean_object* v_a_4576_; lean_object* v___x_4578_; uint8_t v_isShared_4579_; uint8_t v_isSharedCheck_4583_; 
lean_del_object(v___x_4297_);
lean_dec(v_a_4295_);
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4576_ = lean_ctor_get(v___x_4301_, 0);
v_isSharedCheck_4583_ = !lean_is_exclusive(v___x_4301_);
if (v_isSharedCheck_4583_ == 0)
{
v___x_4578_ = v___x_4301_;
v_isShared_4579_ = v_isSharedCheck_4583_;
goto v_resetjp_4577_;
}
else
{
lean_inc(v_a_4576_);
lean_dec(v___x_4301_);
v___x_4578_ = lean_box(0);
v_isShared_4579_ = v_isSharedCheck_4583_;
goto v_resetjp_4577_;
}
v_resetjp_4577_:
{
lean_object* v___x_4581_; 
if (v_isShared_4579_ == 0)
{
v___x_4581_ = v___x_4578_;
goto v_reusejp_4580_;
}
else
{
lean_object* v_reuseFailAlloc_4582_; 
v_reuseFailAlloc_4582_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4582_, 0, v_a_4576_);
v___x_4581_ = v_reuseFailAlloc_4582_;
goto v_reusejp_4580_;
}
v_reusejp_4580_:
{
return v___x_4581_;
}
}
}
}
}
else
{
lean_dec(v_a_4289_);
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
return v___x_4294_;
}
}
else
{
lean_object* v_a_4585_; lean_object* v___x_4587_; uint8_t v_isShared_4588_; uint8_t v_isSharedCheck_4592_; 
lean_dec_ref(v___x_4285_);
lean_dec_ref(v_err_4274_);
lean_dec_ref(v_inst_4273_);
v_a_4585_ = lean_ctor_get(v___x_4288_, 0);
v_isSharedCheck_4592_ = !lean_is_exclusive(v___x_4288_);
if (v_isSharedCheck_4592_ == 0)
{
v___x_4587_ = v___x_4288_;
v_isShared_4588_ = v_isSharedCheck_4592_;
goto v_resetjp_4586_;
}
else
{
lean_inc(v_a_4585_);
lean_dec(v___x_4288_);
v___x_4587_ = lean_box(0);
v_isShared_4588_ = v_isSharedCheck_4592_;
goto v_resetjp_4586_;
}
v_resetjp_4586_:
{
lean_object* v___x_4590_; 
if (v_isShared_4588_ == 0)
{
v___x_4590_ = v___x_4587_;
goto v_reusejp_4589_;
}
else
{
lean_object* v_reuseFailAlloc_4591_; 
v_reuseFailAlloc_4591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4591_, 0, v_a_4585_);
v___x_4590_ = v_reuseFailAlloc_4591_;
goto v_reusejp_4589_;
}
v_reusejp_4589_:
{
return v___x_4590_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___redArg___boxed(lean_object* v_inst_4593_, lean_object* v_err_4594_, lean_object* v_a_4595_, lean_object* v_a_4596_, lean_object* v_a_4597_, lean_object* v_a_4598_, lean_object* v_a_4599_, lean_object* v_a_4600_, lean_object* v_a_4601_, lean_object* v_a_4602_, lean_object* v_a_4603_){
_start:
{
lean_object* v_res_4604_; 
v_res_4604_ = lp_aesop_Aesop_handleNonfatalError___redArg(v_inst_4593_, v_err_4594_, v_a_4595_, v_a_4596_, v_a_4597_, v_a_4598_, v_a_4599_, v_a_4600_, v_a_4601_, v_a_4602_);
lean_dec(v_a_4602_);
lean_dec_ref(v_a_4601_);
lean_dec(v_a_4600_);
lean_dec_ref(v_a_4599_);
lean_dec(v_a_4598_);
lean_dec(v_a_4597_);
lean_dec(v_a_4596_);
lean_dec_ref(v_a_4595_);
return v_res_4604_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError(lean_object* v_Q_4605_, lean_object* v_inst_4606_, lean_object* v_err_4607_, lean_object* v_a_4608_, lean_object* v_a_4609_, lean_object* v_a_4610_, lean_object* v_a_4611_, lean_object* v_a_4612_, lean_object* v_a_4613_, lean_object* v_a_4614_, lean_object* v_a_4615_){
_start:
{
lean_object* v___x_4617_; 
v___x_4617_ = lp_aesop_Aesop_handleNonfatalError___redArg(v_inst_4606_, v_err_4607_, v_a_4608_, v_a_4609_, v_a_4610_, v_a_4611_, v_a_4612_, v_a_4613_, v_a_4614_, v_a_4615_);
return v___x_4617_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_handleNonfatalError___boxed(lean_object* v_Q_4618_, lean_object* v_inst_4619_, lean_object* v_err_4620_, lean_object* v_a_4621_, lean_object* v_a_4622_, lean_object* v_a_4623_, lean_object* v_a_4624_, lean_object* v_a_4625_, lean_object* v_a_4626_, lean_object* v_a_4627_, lean_object* v_a_4628_, lean_object* v_a_4629_){
_start:
{
lean_object* v_res_4630_; 
v_res_4630_ = lp_aesop_Aesop_handleNonfatalError(v_Q_4618_, v_inst_4619_, v_err_4620_, v_a_4621_, v_a_4622_, v_a_4623_, v_a_4624_, v_a_4625_, v_a_4626_, v_a_4627_, v_a_4628_);
lean_dec(v_a_4628_);
lean_dec_ref(v_a_4627_);
lean_dec(v_a_4626_);
lean_dec_ref(v_a_4625_);
lean_dec(v_a_4624_);
lean_dec(v_a_4623_);
lean_dec(v_a_4622_);
lean_dec_ref(v_a_4621_);
return v_res_4630_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___redArg(lean_object* v_inst_4631_, lean_object* v_a_4632_, lean_object* v_a_4633_, lean_object* v_a_4634_, lean_object* v_a_4635_, lean_object* v_a_4636_, lean_object* v_a_4637_, lean_object* v_a_4638_, lean_object* v_a_4639_){
_start:
{
lean_object* v___x_4641_; lean_object* v___x_4642_; lean_object* v___x_4643_; lean_object* v___f_4644_; lean_object* v___x_4645_; lean_object* v___x_4646_; lean_object* v_fileName_4647_; lean_object* v_fileMap_4648_; lean_object* v_options_4649_; lean_object* v_currRecDepth_4650_; lean_object* v_maxRecDepth_4651_; lean_object* v_ref_4652_; lean_object* v_currNamespace_4653_; lean_object* v_openDecls_4654_; lean_object* v_initHeartbeats_4655_; lean_object* v_maxHeartbeats_4656_; lean_object* v_quotContext_4657_; lean_object* v_currMacroScope_4658_; uint8_t v_diag_4659_; lean_object* v_cancelTk_x3f_4660_; uint8_t v_suppressElabErrors_4661_; lean_object* v_inheritedTraceOptions_4662_; lean_object* v___x_4663_; lean_object* v___x_4762_; uint8_t v___x_4763_; 
v___x_4641_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_4631_);
v___x_4642_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__23, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__23_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__23);
v___x_4643_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_4631_);
v___f_4644_ = lean_obj_once(&lp_aesop_Aesop_nextActiveGoal___redArg___closed__24, &lp_aesop_Aesop_nextActiveGoal___redArg___closed__24_once, _init_lp_aesop_Aesop_nextActiveGoal___redArg___closed__24);
v___x_4645_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_4644_, v___x_4641_);
v___x_4646_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4646_, 0, v___x_4642_);
lean_ctor_set(v___x_4646_, 1, v___x_4643_);
lean_ctor_set(v___x_4646_, 2, v___x_4645_);
v_fileName_4647_ = lean_ctor_get(v_a_4638_, 0);
v_fileMap_4648_ = lean_ctor_get(v_a_4638_, 1);
v_options_4649_ = lean_ctor_get(v_a_4638_, 2);
v_currRecDepth_4650_ = lean_ctor_get(v_a_4638_, 3);
v_maxRecDepth_4651_ = lean_ctor_get(v_a_4638_, 4);
v_ref_4652_ = lean_ctor_get(v_a_4638_, 5);
v_currNamespace_4653_ = lean_ctor_get(v_a_4638_, 6);
v_openDecls_4654_ = lean_ctor_get(v_a_4638_, 7);
v_initHeartbeats_4655_ = lean_ctor_get(v_a_4638_, 8);
v_maxHeartbeats_4656_ = lean_ctor_get(v_a_4638_, 9);
v_quotContext_4657_ = lean_ctor_get(v_a_4638_, 10);
v_currMacroScope_4658_ = lean_ctor_get(v_a_4638_, 11);
v_diag_4659_ = lean_ctor_get_uint8(v_a_4638_, sizeof(void*)*14);
v_cancelTk_x3f_4660_ = lean_ctor_get(v_a_4638_, 12);
v_suppressElabErrors_4661_ = lean_ctor_get_uint8(v_a_4638_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4662_ = lean_ctor_get(v_a_4638_, 13);
v___x_4663_ = ((lean_object*)(lp_aesop_Aesop_traceScript___redArg___lam__1___closed__0));
v___x_4762_ = lean_unsigned_to_nat(0u);
v___x_4763_ = lean_nat_dec_eq(v_maxRecDepth_4651_, v___x_4762_);
if (v___x_4763_ == 0)
{
uint8_t v___x_4764_; 
v___x_4764_ = lean_nat_dec_eq(v_currRecDepth_4650_, v_maxRecDepth_4651_);
if (v___x_4764_ == 0)
{
lean_inc_ref(v_inheritedTraceOptions_4662_);
lean_inc(v_cancelTk_x3f_4660_);
lean_inc(v_currMacroScope_4658_);
lean_inc(v_quotContext_4657_);
lean_inc(v_maxHeartbeats_4656_);
lean_inc(v_initHeartbeats_4655_);
lean_inc(v_openDecls_4654_);
lean_inc(v_currNamespace_4653_);
lean_inc(v_ref_4652_);
lean_inc(v_maxRecDepth_4651_);
lean_inc(v_currRecDepth_4650_);
lean_inc_ref(v_options_4649_);
lean_inc_ref(v_fileMap_4648_);
lean_inc_ref(v_fileName_4647_);
lean_dec_ref_known(v___x_4646_, 3);
lean_dec_ref(v_a_4638_);
goto v___jp_4664_;
}
else
{
lean_object* v___x_14596__overap_4765_; lean_object* v___x_4766_; 
lean_dec_ref(v_inst_4631_);
lean_inc(v_ref_4652_);
v___x_14596__overap_4765_ = l_Lean_throwMaxRecDepthAt___redArg(v___x_4646_, v_ref_4652_);
lean_inc(v_a_4639_);
lean_inc(v_a_4637_);
lean_inc_ref(v_a_4636_);
lean_inc(v_a_4635_);
lean_inc(v_a_4634_);
lean_inc(v_a_4633_);
lean_inc_ref(v_a_4632_);
v___x_4766_ = lean_apply_9(v___x_14596__overap_4765_, v_a_4632_, v_a_4633_, v_a_4634_, v_a_4635_, v_a_4636_, v_a_4637_, v_a_4638_, v_a_4639_, lean_box(0));
return v___x_4766_;
}
}
else
{
lean_inc_ref(v_inheritedTraceOptions_4662_);
lean_inc(v_cancelTk_x3f_4660_);
lean_inc(v_currMacroScope_4658_);
lean_inc(v_quotContext_4657_);
lean_inc(v_maxHeartbeats_4656_);
lean_inc(v_initHeartbeats_4655_);
lean_inc(v_openDecls_4654_);
lean_inc(v_currNamespace_4653_);
lean_inc(v_ref_4652_);
lean_inc(v_maxRecDepth_4651_);
lean_inc(v_currRecDepth_4650_);
lean_inc_ref(v_options_4649_);
lean_inc_ref(v_fileMap_4648_);
lean_inc_ref(v_fileName_4647_);
lean_dec_ref_known(v___x_4646_, 3);
lean_dec_ref(v_a_4638_);
goto v___jp_4664_;
}
v___jp_4664_:
{
lean_object* v___x_4665_; lean_object* v___x_4666_; lean_object* v___x_4667_; lean_object* v___x_4668_; lean_object* v___x_4669_; 
v___x_4665_ = lean_st_ref_get(v_a_4633_);
lean_dec(v___x_4665_);
v___x_4666_ = lean_unsigned_to_nat(1u);
v___x_4667_ = lean_nat_add(v_currRecDepth_4650_, v___x_4666_);
lean_dec(v_currRecDepth_4650_);
v___x_4668_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4668_, 0, v_fileName_4647_);
lean_ctor_set(v___x_4668_, 1, v_fileMap_4648_);
lean_ctor_set(v___x_4668_, 2, v_options_4649_);
lean_ctor_set(v___x_4668_, 3, v___x_4667_);
lean_ctor_set(v___x_4668_, 4, v_maxRecDepth_4651_);
lean_ctor_set(v___x_4668_, 5, v_ref_4652_);
lean_ctor_set(v___x_4668_, 6, v_currNamespace_4653_);
lean_ctor_set(v___x_4668_, 7, v_openDecls_4654_);
lean_ctor_set(v___x_4668_, 8, v_initHeartbeats_4655_);
lean_ctor_set(v___x_4668_, 9, v_maxHeartbeats_4656_);
lean_ctor_set(v___x_4668_, 10, v_quotContext_4657_);
lean_ctor_set(v___x_4668_, 11, v_currMacroScope_4658_);
lean_ctor_set(v___x_4668_, 12, v_cancelTk_x3f_4660_);
lean_ctor_set(v___x_4668_, 13, v_inheritedTraceOptions_4662_);
lean_ctor_set_uint8(v___x_4668_, sizeof(void*)*14, v_diag_4659_);
lean_ctor_set_uint8(v___x_4668_, sizeof(void*)*14 + 1, v_suppressElabErrors_4661_);
v___x_4669_ = l_Lean_Core_checkSystem(v___x_4663_, v___x_4668_, v_a_4639_);
if (lean_obj_tag(v___x_4669_) == 0)
{
lean_object* v___x_4670_; 
lean_dec_ref_known(v___x_4669_, 1);
v___x_4670_ = lp_aesop_Aesop_checkRootUnprovable___redArg(v_a_4632_, v_a_4633_, v_a_4634_);
if (lean_obj_tag(v___x_4670_) == 0)
{
lean_object* v_a_4671_; 
v_a_4671_ = lean_ctor_get(v___x_4670_, 0);
lean_inc(v_a_4671_);
lean_dec_ref_known(v___x_4670_, 1);
if (lean_obj_tag(v_a_4671_) == 1)
{
lean_object* v_val_4672_; lean_object* v___x_4673_; 
v_val_4672_ = lean_ctor_get(v_a_4671_, 0);
lean_inc(v_val_4672_);
lean_dec_ref_known(v_a_4671_, 1);
v___x_4673_ = lp_aesop_Aesop_handleNonfatalError___redArg(v_inst_4631_, v_val_4672_, v_a_4632_, v_a_4633_, v_a_4634_, v_a_4635_, v_a_4636_, v_a_4637_, v___x_4668_, v_a_4639_);
lean_dec_ref_known(v___x_4668_, 14);
return v___x_4673_;
}
else
{
lean_object* v___x_4674_; 
lean_dec(v_a_4671_);
lean_inc_ref(v_inst_4631_);
v___x_4674_ = lp_aesop_Aesop_finishIfProven___redArg(v_inst_4631_, v_a_4632_, v_a_4633_, v_a_4634_, v_a_4635_, v_a_4636_, v_a_4637_, v___x_4668_, v_a_4639_);
if (lean_obj_tag(v___x_4674_) == 0)
{
lean_object* v_a_4675_; lean_object* v___x_4677_; uint8_t v_isShared_4678_; uint8_t v_isSharedCheck_4737_; 
v_a_4675_ = lean_ctor_get(v___x_4674_, 0);
v_isSharedCheck_4737_ = !lean_is_exclusive(v___x_4674_);
if (v_isSharedCheck_4737_ == 0)
{
v___x_4677_ = v___x_4674_;
v_isShared_4678_ = v_isSharedCheck_4737_;
goto v_resetjp_4676_;
}
else
{
lean_inc(v_a_4675_);
lean_dec(v___x_4674_);
v___x_4677_ = lean_box(0);
v_isShared_4678_ = v_isSharedCheck_4737_;
goto v_resetjp_4676_;
}
v_resetjp_4676_:
{
uint8_t v___x_4679_; 
v___x_4679_ = lean_unbox(v_a_4675_);
lean_dec(v_a_4675_);
if (v___x_4679_ == 0)
{
lean_object* v___x_4680_; 
lean_del_object(v___x_4677_);
v___x_4680_ = lp_aesop_Aesop_checkGoalLimit___redArg(v_a_4632_, v_a_4633_, v_a_4634_);
if (lean_obj_tag(v___x_4680_) == 0)
{
lean_object* v_a_4681_; 
v_a_4681_ = lean_ctor_get(v___x_4680_, 0);
lean_inc(v_a_4681_);
lean_dec_ref_known(v___x_4680_, 1);
if (lean_obj_tag(v_a_4681_) == 1)
{
lean_object* v_val_4682_; lean_object* v___x_4683_; 
v_val_4682_ = lean_ctor_get(v_a_4681_, 0);
lean_inc(v_val_4682_);
lean_dec_ref_known(v_a_4681_, 1);
v___x_4683_ = lp_aesop_Aesop_handleNonfatalError___redArg(v_inst_4631_, v_val_4682_, v_a_4632_, v_a_4633_, v_a_4634_, v_a_4635_, v_a_4636_, v_a_4637_, v___x_4668_, v_a_4639_);
lean_dec_ref_known(v___x_4668_, 14);
return v___x_4683_;
}
else
{
lean_object* v___x_4684_; 
lean_dec(v_a_4681_);
v___x_4684_ = lp_aesop_Aesop_checkRappLimit___redArg(v_a_4632_, v_a_4633_, v_a_4634_);
if (lean_obj_tag(v___x_4684_) == 0)
{
lean_object* v_a_4685_; 
v_a_4685_ = lean_ctor_get(v___x_4684_, 0);
lean_inc(v_a_4685_);
lean_dec_ref_known(v___x_4684_, 1);
if (lean_obj_tag(v_a_4685_) == 1)
{
lean_object* v_val_4686_; lean_object* v___x_4687_; 
v_val_4686_ = lean_ctor_get(v_a_4685_, 0);
lean_inc(v_val_4686_);
lean_dec_ref_known(v_a_4685_, 1);
v___x_4687_ = lp_aesop_Aesop_handleNonfatalError___redArg(v_inst_4631_, v_val_4686_, v_a_4632_, v_a_4633_, v_a_4634_, v_a_4635_, v_a_4636_, v_a_4637_, v___x_4668_, v_a_4639_);
lean_dec_ref_known(v___x_4668_, 14);
return v___x_4687_;
}
else
{
lean_object* v___x_4688_; 
lean_dec(v_a_4685_);
lean_inc_ref(v_inst_4631_);
v___x_4688_ = lp_aesop_Aesop_expandNextGoal___redArg(v_inst_4631_, v_a_4632_, v_a_4633_, v_a_4634_, v_a_4635_, v_a_4636_, v_a_4637_, v___x_4668_, v_a_4639_);
if (lean_obj_tag(v___x_4688_) == 0)
{
lean_object* v___x_4689_; lean_object* v___x_4690_; 
lean_dec_ref_known(v___x_4688_, 1);
v___x_4689_ = lean_st_ref_get(v_a_4633_);
lean_dec(v___x_4689_);
v___x_4690_ = lp_aesop_Aesop_checkInvariantsIfEnabled___redArg(v_a_4634_, v_a_4636_, v_a_4637_, v___x_4668_, v_a_4639_);
if (lean_obj_tag(v___x_4690_) == 0)
{
lean_object* v___x_4691_; 
lean_dec_ref_known(v___x_4690_, 1);
v___x_4691_ = lp_aesop_Aesop_incrementIteration___redArg(v_a_4633_);
if (lean_obj_tag(v___x_4691_) == 0)
{
lean_dec_ref_known(v___x_4691_, 1);
v_a_4638_ = v___x_4668_;
goto _start;
}
else
{
lean_object* v_a_4693_; lean_object* v___x_4695_; uint8_t v_isShared_4696_; uint8_t v_isSharedCheck_4700_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v_a_4693_ = lean_ctor_get(v___x_4691_, 0);
v_isSharedCheck_4700_ = !lean_is_exclusive(v___x_4691_);
if (v_isSharedCheck_4700_ == 0)
{
v___x_4695_ = v___x_4691_;
v_isShared_4696_ = v_isSharedCheck_4700_;
goto v_resetjp_4694_;
}
else
{
lean_inc(v_a_4693_);
lean_dec(v___x_4691_);
v___x_4695_ = lean_box(0);
v_isShared_4696_ = v_isSharedCheck_4700_;
goto v_resetjp_4694_;
}
v_resetjp_4694_:
{
lean_object* v___x_4698_; 
if (v_isShared_4696_ == 0)
{
v___x_4698_ = v___x_4695_;
goto v_reusejp_4697_;
}
else
{
lean_object* v_reuseFailAlloc_4699_; 
v_reuseFailAlloc_4699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4699_, 0, v_a_4693_);
v___x_4698_ = v_reuseFailAlloc_4699_;
goto v_reusejp_4697_;
}
v_reusejp_4697_:
{
return v___x_4698_;
}
}
}
}
else
{
lean_object* v_a_4701_; lean_object* v___x_4703_; uint8_t v_isShared_4704_; uint8_t v_isSharedCheck_4708_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v_a_4701_ = lean_ctor_get(v___x_4690_, 0);
v_isSharedCheck_4708_ = !lean_is_exclusive(v___x_4690_);
if (v_isSharedCheck_4708_ == 0)
{
v___x_4703_ = v___x_4690_;
v_isShared_4704_ = v_isSharedCheck_4708_;
goto v_resetjp_4702_;
}
else
{
lean_inc(v_a_4701_);
lean_dec(v___x_4690_);
v___x_4703_ = lean_box(0);
v_isShared_4704_ = v_isSharedCheck_4708_;
goto v_resetjp_4702_;
}
v_resetjp_4702_:
{
lean_object* v___x_4706_; 
if (v_isShared_4704_ == 0)
{
v___x_4706_ = v___x_4703_;
goto v_reusejp_4705_;
}
else
{
lean_object* v_reuseFailAlloc_4707_; 
v_reuseFailAlloc_4707_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4707_, 0, v_a_4701_);
v___x_4706_ = v_reuseFailAlloc_4707_;
goto v_reusejp_4705_;
}
v_reusejp_4705_:
{
return v___x_4706_;
}
}
}
}
else
{
lean_object* v_a_4709_; lean_object* v___x_4711_; uint8_t v_isShared_4712_; uint8_t v_isSharedCheck_4716_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v_a_4709_ = lean_ctor_get(v___x_4688_, 0);
v_isSharedCheck_4716_ = !lean_is_exclusive(v___x_4688_);
if (v_isSharedCheck_4716_ == 0)
{
v___x_4711_ = v___x_4688_;
v_isShared_4712_ = v_isSharedCheck_4716_;
goto v_resetjp_4710_;
}
else
{
lean_inc(v_a_4709_);
lean_dec(v___x_4688_);
v___x_4711_ = lean_box(0);
v_isShared_4712_ = v_isSharedCheck_4716_;
goto v_resetjp_4710_;
}
v_resetjp_4710_:
{
lean_object* v___x_4714_; 
if (v_isShared_4712_ == 0)
{
v___x_4714_ = v___x_4711_;
goto v_reusejp_4713_;
}
else
{
lean_object* v_reuseFailAlloc_4715_; 
v_reuseFailAlloc_4715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4715_, 0, v_a_4709_);
v___x_4714_ = v_reuseFailAlloc_4715_;
goto v_reusejp_4713_;
}
v_reusejp_4713_:
{
return v___x_4714_;
}
}
}
}
}
else
{
lean_object* v_a_4717_; lean_object* v___x_4719_; uint8_t v_isShared_4720_; uint8_t v_isSharedCheck_4724_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v_a_4717_ = lean_ctor_get(v___x_4684_, 0);
v_isSharedCheck_4724_ = !lean_is_exclusive(v___x_4684_);
if (v_isSharedCheck_4724_ == 0)
{
v___x_4719_ = v___x_4684_;
v_isShared_4720_ = v_isSharedCheck_4724_;
goto v_resetjp_4718_;
}
else
{
lean_inc(v_a_4717_);
lean_dec(v___x_4684_);
v___x_4719_ = lean_box(0);
v_isShared_4720_ = v_isSharedCheck_4724_;
goto v_resetjp_4718_;
}
v_resetjp_4718_:
{
lean_object* v___x_4722_; 
if (v_isShared_4720_ == 0)
{
v___x_4722_ = v___x_4719_;
goto v_reusejp_4721_;
}
else
{
lean_object* v_reuseFailAlloc_4723_; 
v_reuseFailAlloc_4723_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4723_, 0, v_a_4717_);
v___x_4722_ = v_reuseFailAlloc_4723_;
goto v_reusejp_4721_;
}
v_reusejp_4721_:
{
return v___x_4722_;
}
}
}
}
}
else
{
lean_object* v_a_4725_; lean_object* v___x_4727_; uint8_t v_isShared_4728_; uint8_t v_isSharedCheck_4732_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v_a_4725_ = lean_ctor_get(v___x_4680_, 0);
v_isSharedCheck_4732_ = !lean_is_exclusive(v___x_4680_);
if (v_isSharedCheck_4732_ == 0)
{
v___x_4727_ = v___x_4680_;
v_isShared_4728_ = v_isSharedCheck_4732_;
goto v_resetjp_4726_;
}
else
{
lean_inc(v_a_4725_);
lean_dec(v___x_4680_);
v___x_4727_ = lean_box(0);
v_isShared_4728_ = v_isSharedCheck_4732_;
goto v_resetjp_4726_;
}
v_resetjp_4726_:
{
lean_object* v___x_4730_; 
if (v_isShared_4728_ == 0)
{
v___x_4730_ = v___x_4727_;
goto v_reusejp_4729_;
}
else
{
lean_object* v_reuseFailAlloc_4731_; 
v_reuseFailAlloc_4731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4731_, 0, v_a_4725_);
v___x_4730_ = v_reuseFailAlloc_4731_;
goto v_reusejp_4729_;
}
v_reusejp_4729_:
{
return v___x_4730_;
}
}
}
}
else
{
lean_object* v___x_4733_; lean_object* v___x_4735_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v___x_4733_ = ((lean_object*)(lp_aesop_Aesop_handleNonfatalError___redArg___closed__5));
if (v_isShared_4678_ == 0)
{
lean_ctor_set(v___x_4677_, 0, v___x_4733_);
v___x_4735_ = v___x_4677_;
goto v_reusejp_4734_;
}
else
{
lean_object* v_reuseFailAlloc_4736_; 
v_reuseFailAlloc_4736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4736_, 0, v___x_4733_);
v___x_4735_ = v_reuseFailAlloc_4736_;
goto v_reusejp_4734_;
}
v_reusejp_4734_:
{
return v___x_4735_;
}
}
}
}
else
{
lean_object* v_a_4738_; lean_object* v___x_4740_; uint8_t v_isShared_4741_; uint8_t v_isSharedCheck_4745_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v_a_4738_ = lean_ctor_get(v___x_4674_, 0);
v_isSharedCheck_4745_ = !lean_is_exclusive(v___x_4674_);
if (v_isSharedCheck_4745_ == 0)
{
v___x_4740_ = v___x_4674_;
v_isShared_4741_ = v_isSharedCheck_4745_;
goto v_resetjp_4739_;
}
else
{
lean_inc(v_a_4738_);
lean_dec(v___x_4674_);
v___x_4740_ = lean_box(0);
v_isShared_4741_ = v_isSharedCheck_4745_;
goto v_resetjp_4739_;
}
v_resetjp_4739_:
{
lean_object* v___x_4743_; 
if (v_isShared_4741_ == 0)
{
v___x_4743_ = v___x_4740_;
goto v_reusejp_4742_;
}
else
{
lean_object* v_reuseFailAlloc_4744_; 
v_reuseFailAlloc_4744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4744_, 0, v_a_4738_);
v___x_4743_ = v_reuseFailAlloc_4744_;
goto v_reusejp_4742_;
}
v_reusejp_4742_:
{
return v___x_4743_;
}
}
}
}
}
else
{
lean_object* v_a_4746_; lean_object* v___x_4748_; uint8_t v_isShared_4749_; uint8_t v_isSharedCheck_4753_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v_a_4746_ = lean_ctor_get(v___x_4670_, 0);
v_isSharedCheck_4753_ = !lean_is_exclusive(v___x_4670_);
if (v_isSharedCheck_4753_ == 0)
{
v___x_4748_ = v___x_4670_;
v_isShared_4749_ = v_isSharedCheck_4753_;
goto v_resetjp_4747_;
}
else
{
lean_inc(v_a_4746_);
lean_dec(v___x_4670_);
v___x_4748_ = lean_box(0);
v_isShared_4749_ = v_isSharedCheck_4753_;
goto v_resetjp_4747_;
}
v_resetjp_4747_:
{
lean_object* v___x_4751_; 
if (v_isShared_4749_ == 0)
{
v___x_4751_ = v___x_4748_;
goto v_reusejp_4750_;
}
else
{
lean_object* v_reuseFailAlloc_4752_; 
v_reuseFailAlloc_4752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4752_, 0, v_a_4746_);
v___x_4751_ = v_reuseFailAlloc_4752_;
goto v_reusejp_4750_;
}
v_reusejp_4750_:
{
return v___x_4751_;
}
}
}
}
else
{
lean_object* v_a_4754_; lean_object* v___x_4756_; uint8_t v_isShared_4757_; uint8_t v_isSharedCheck_4761_; 
lean_dec_ref_known(v___x_4668_, 14);
lean_dec_ref(v_inst_4631_);
v_a_4754_ = lean_ctor_get(v___x_4669_, 0);
v_isSharedCheck_4761_ = !lean_is_exclusive(v___x_4669_);
if (v_isSharedCheck_4761_ == 0)
{
v___x_4756_ = v___x_4669_;
v_isShared_4757_ = v_isSharedCheck_4761_;
goto v_resetjp_4755_;
}
else
{
lean_inc(v_a_4754_);
lean_dec(v___x_4669_);
v___x_4756_ = lean_box(0);
v_isShared_4757_ = v_isSharedCheck_4761_;
goto v_resetjp_4755_;
}
v_resetjp_4755_:
{
lean_object* v___x_4759_; 
if (v_isShared_4757_ == 0)
{
v___x_4759_ = v___x_4756_;
goto v_reusejp_4758_;
}
else
{
lean_object* v_reuseFailAlloc_4760_; 
v_reuseFailAlloc_4760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4760_, 0, v_a_4754_);
v___x_4759_ = v_reuseFailAlloc_4760_;
goto v_reusejp_4758_;
}
v_reusejp_4758_:
{
return v___x_4759_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___redArg___boxed(lean_object* v_inst_4767_, lean_object* v_a_4768_, lean_object* v_a_4769_, lean_object* v_a_4770_, lean_object* v_a_4771_, lean_object* v_a_4772_, lean_object* v_a_4773_, lean_object* v_a_4774_, lean_object* v_a_4775_, lean_object* v_a_4776_){
_start:
{
lean_object* v_res_4777_; 
v_res_4777_ = lp_aesop_Aesop_searchLoop___redArg(v_inst_4767_, v_a_4768_, v_a_4769_, v_a_4770_, v_a_4771_, v_a_4772_, v_a_4773_, v_a_4774_, v_a_4775_);
lean_dec(v_a_4775_);
lean_dec(v_a_4773_);
lean_dec_ref(v_a_4772_);
lean_dec(v_a_4771_);
lean_dec(v_a_4770_);
lean_dec(v_a_4769_);
lean_dec_ref(v_a_4768_);
return v_res_4777_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop(lean_object* v_Q_4778_, lean_object* v_inst_4779_, lean_object* v_a_4780_, lean_object* v_a_4781_, lean_object* v_a_4782_, lean_object* v_a_4783_, lean_object* v_a_4784_, lean_object* v_a_4785_, lean_object* v_a_4786_, lean_object* v_a_4787_){
_start:
{
lean_object* v___x_4789_; 
lean_inc_ref(v_a_4786_);
v___x_4789_ = lp_aesop_Aesop_searchLoop___redArg(v_inst_4779_, v_a_4780_, v_a_4781_, v_a_4782_, v_a_4783_, v_a_4784_, v_a_4785_, v_a_4786_, v_a_4787_);
return v___x_4789_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_searchLoop___boxed(lean_object* v_Q_4790_, lean_object* v_inst_4791_, lean_object* v_a_4792_, lean_object* v_a_4793_, lean_object* v_a_4794_, lean_object* v_a_4795_, lean_object* v_a_4796_, lean_object* v_a_4797_, lean_object* v_a_4798_, lean_object* v_a_4799_, lean_object* v_a_4800_){
_start:
{
lean_object* v_res_4801_; 
v_res_4801_ = lp_aesop_Aesop_searchLoop(v_Q_4790_, v_inst_4791_, v_a_4792_, v_a_4793_, v_a_4794_, v_a_4795_, v_a_4796_, v_a_4797_, v_a_4798_, v_a_4799_);
lean_dec(v_a_4799_);
lean_dec_ref(v_a_4798_);
lean_dec(v_a_4797_);
lean_dec_ref(v_a_4796_);
lean_dec(v_a_4795_);
lean_dec(v_a_4794_);
lean_dec(v_a_4793_);
lean_dec_ref(v_a_4792_);
return v_res_4801_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__0(lean_object* v___y_4802_, lean_object* v___y_4803_, lean_object* v___y_4804_, lean_object* v___y_4805_, lean_object* v___y_4806_, lean_object* v___y_4807_, lean_object* v___y_4808_, lean_object* v___y_4809_, lean_object* v_a_x3f_4810_){
_start:
{
lean_object* v___x_4812_; lean_object* v_iteration_4813_; lean_object* v_ruleSet_4814_; lean_object* v___x_4815_; lean_object* v___x_4816_; 
v___x_4812_ = lean_st_ref_get(v___y_4802_);
v_iteration_4813_ = lean_ctor_get(v___x_4812_, 0);
lean_inc(v_iteration_4813_);
lean_dec(v___x_4812_);
v_ruleSet_4814_ = lean_ctor_get(v___y_4803_, 0);
lean_inc_ref(v_ruleSet_4814_);
v___x_4815_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4815_, 0, v_iteration_4813_);
lean_ctor_set(v___x_4815_, 1, v_ruleSet_4814_);
v___x_4816_ = lp_aesop_Aesop_collectGoalStatsIfEnabled(v___x_4815_, v___y_4804_, v___y_4805_, v___y_4806_, v___y_4807_, v___y_4808_, v___y_4809_);
lean_dec_ref_known(v___x_4815_, 2);
if (lean_obj_tag(v___x_4816_) == 0)
{
lean_object* v___x_4817_; lean_object* v___x_4818_; 
lean_dec_ref_known(v___x_4816_, 1);
v___x_4817_ = lean_st_ref_get(v___y_4802_);
lean_dec(v___x_4817_);
v___x_4818_ = lp_aesop_Aesop_freeTree___redArg(v___y_4804_);
return v___x_4818_;
}
else
{
return v___x_4816_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__0___boxed(lean_object* v___y_4819_, lean_object* v___y_4820_, lean_object* v___y_4821_, lean_object* v___y_4822_, lean_object* v___y_4823_, lean_object* v___y_4824_, lean_object* v___y_4825_, lean_object* v___y_4826_, lean_object* v_a_x3f_4827_, lean_object* v___y_4828_){
_start:
{
lean_object* v_res_4829_; 
v_res_4829_ = lp_aesop_Aesop_search___lam__0(v___y_4819_, v___y_4820_, v___y_4821_, v___y_4822_, v___y_4823_, v___y_4824_, v___y_4825_, v___y_4826_, v_a_x3f_4827_);
lean_dec(v_a_x3f_4827_);
lean_dec(v___y_4826_);
lean_dec_ref(v___y_4825_);
lean_dec(v___y_4824_);
lean_dec_ref(v___y_4823_);
lean_dec(v___y_4822_);
lean_dec(v___y_4821_);
lean_dec_ref(v___y_4820_);
lean_dec(v___y_4819_);
return v_res_4829_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__1(lean_object* v_snd_4830_, lean_object* v___y_4831_, lean_object* v___y_4832_, lean_object* v___y_4833_, lean_object* v___y_4834_, lean_object* v___y_4835_, lean_object* v___y_4836_, lean_object* v___y_4837_, lean_object* v___y_4838_){
_start:
{
lean_object* v_r_4840_; 
lean_inc_ref(v___y_4837_);
v_r_4840_ = lp_aesop_Aesop_searchLoop___redArg(v_snd_4830_, v___y_4831_, v___y_4832_, v___y_4833_, v___y_4834_, v___y_4835_, v___y_4836_, v___y_4837_, v___y_4838_);
if (lean_obj_tag(v_r_4840_) == 0)
{
lean_object* v_a_4841_; lean_object* v___x_4843_; uint8_t v_isShared_4844_; uint8_t v_isSharedCheck_4865_; 
v_a_4841_ = lean_ctor_get(v_r_4840_, 0);
v_isSharedCheck_4865_ = !lean_is_exclusive(v_r_4840_);
if (v_isSharedCheck_4865_ == 0)
{
v___x_4843_ = v_r_4840_;
v_isShared_4844_ = v_isSharedCheck_4865_;
goto v_resetjp_4842_;
}
else
{
lean_inc(v_a_4841_);
lean_dec(v_r_4840_);
v___x_4843_ = lean_box(0);
v_isShared_4844_ = v_isSharedCheck_4865_;
goto v_resetjp_4842_;
}
v_resetjp_4842_:
{
lean_object* v___x_4846_; 
lean_inc(v_a_4841_);
if (v_isShared_4844_ == 0)
{
lean_ctor_set_tag(v___x_4843_, 1);
v___x_4846_ = v___x_4843_;
goto v_reusejp_4845_;
}
else
{
lean_object* v_reuseFailAlloc_4864_; 
v_reuseFailAlloc_4864_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4864_, 0, v_a_4841_);
v___x_4846_ = v_reuseFailAlloc_4864_;
goto v_reusejp_4845_;
}
v_reusejp_4845_:
{
lean_object* v___x_4847_; 
v___x_4847_ = lp_aesop_Aesop_search___lam__0(v___y_4832_, v___y_4831_, v___y_4833_, v___y_4834_, v___y_4835_, v___y_4836_, v___y_4837_, v___y_4838_, v___x_4846_);
lean_dec_ref(v___x_4846_);
if (lean_obj_tag(v___x_4847_) == 0)
{
lean_object* v___x_4849_; uint8_t v_isShared_4850_; uint8_t v_isSharedCheck_4854_; 
v_isSharedCheck_4854_ = !lean_is_exclusive(v___x_4847_);
if (v_isSharedCheck_4854_ == 0)
{
lean_object* v_unused_4855_; 
v_unused_4855_ = lean_ctor_get(v___x_4847_, 0);
lean_dec(v_unused_4855_);
v___x_4849_ = v___x_4847_;
v_isShared_4850_ = v_isSharedCheck_4854_;
goto v_resetjp_4848_;
}
else
{
lean_dec(v___x_4847_);
v___x_4849_ = lean_box(0);
v_isShared_4850_ = v_isSharedCheck_4854_;
goto v_resetjp_4848_;
}
v_resetjp_4848_:
{
lean_object* v___x_4852_; 
if (v_isShared_4850_ == 0)
{
lean_ctor_set(v___x_4849_, 0, v_a_4841_);
v___x_4852_ = v___x_4849_;
goto v_reusejp_4851_;
}
else
{
lean_object* v_reuseFailAlloc_4853_; 
v_reuseFailAlloc_4853_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4853_, 0, v_a_4841_);
v___x_4852_ = v_reuseFailAlloc_4853_;
goto v_reusejp_4851_;
}
v_reusejp_4851_:
{
return v___x_4852_;
}
}
}
else
{
lean_object* v_a_4856_; lean_object* v___x_4858_; uint8_t v_isShared_4859_; uint8_t v_isSharedCheck_4863_; 
lean_dec(v_a_4841_);
v_a_4856_ = lean_ctor_get(v___x_4847_, 0);
v_isSharedCheck_4863_ = !lean_is_exclusive(v___x_4847_);
if (v_isSharedCheck_4863_ == 0)
{
v___x_4858_ = v___x_4847_;
v_isShared_4859_ = v_isSharedCheck_4863_;
goto v_resetjp_4857_;
}
else
{
lean_inc(v_a_4856_);
lean_dec(v___x_4847_);
v___x_4858_ = lean_box(0);
v_isShared_4859_ = v_isSharedCheck_4863_;
goto v_resetjp_4857_;
}
v_resetjp_4857_:
{
lean_object* v___x_4861_; 
if (v_isShared_4859_ == 0)
{
v___x_4861_ = v___x_4858_;
goto v_reusejp_4860_;
}
else
{
lean_object* v_reuseFailAlloc_4862_; 
v_reuseFailAlloc_4862_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4862_, 0, v_a_4856_);
v___x_4861_ = v_reuseFailAlloc_4862_;
goto v_reusejp_4860_;
}
v_reusejp_4860_:
{
return v___x_4861_;
}
}
}
}
}
}
else
{
lean_object* v_a_4866_; lean_object* v___x_4867_; lean_object* v___x_4868_; 
v_a_4866_ = lean_ctor_get(v_r_4840_, 0);
lean_inc(v_a_4866_);
lean_dec_ref_known(v_r_4840_, 1);
v___x_4867_ = lean_box(0);
v___x_4868_ = lp_aesop_Aesop_search___lam__0(v___y_4832_, v___y_4831_, v___y_4833_, v___y_4834_, v___y_4835_, v___y_4836_, v___y_4837_, v___y_4838_, v___x_4867_);
if (lean_obj_tag(v___x_4868_) == 0)
{
lean_object* v___x_4870_; uint8_t v_isShared_4871_; uint8_t v_isSharedCheck_4875_; 
v_isSharedCheck_4875_ = !lean_is_exclusive(v___x_4868_);
if (v_isSharedCheck_4875_ == 0)
{
lean_object* v_unused_4876_; 
v_unused_4876_ = lean_ctor_get(v___x_4868_, 0);
lean_dec(v_unused_4876_);
v___x_4870_ = v___x_4868_;
v_isShared_4871_ = v_isSharedCheck_4875_;
goto v_resetjp_4869_;
}
else
{
lean_dec(v___x_4868_);
v___x_4870_ = lean_box(0);
v_isShared_4871_ = v_isSharedCheck_4875_;
goto v_resetjp_4869_;
}
v_resetjp_4869_:
{
lean_object* v___x_4873_; 
if (v_isShared_4871_ == 0)
{
lean_ctor_set_tag(v___x_4870_, 1);
lean_ctor_set(v___x_4870_, 0, v_a_4866_);
v___x_4873_ = v___x_4870_;
goto v_reusejp_4872_;
}
else
{
lean_object* v_reuseFailAlloc_4874_; 
v_reuseFailAlloc_4874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4874_, 0, v_a_4866_);
v___x_4873_ = v_reuseFailAlloc_4874_;
goto v_reusejp_4872_;
}
v_reusejp_4872_:
{
return v___x_4873_;
}
}
}
else
{
lean_object* v_a_4877_; lean_object* v___x_4879_; uint8_t v_isShared_4880_; uint8_t v_isSharedCheck_4884_; 
lean_dec(v_a_4866_);
v_a_4877_ = lean_ctor_get(v___x_4868_, 0);
v_isSharedCheck_4884_ = !lean_is_exclusive(v___x_4868_);
if (v_isSharedCheck_4884_ == 0)
{
v___x_4879_ = v___x_4868_;
v_isShared_4880_ = v_isSharedCheck_4884_;
goto v_resetjp_4878_;
}
else
{
lean_inc(v_a_4877_);
lean_dec(v___x_4868_);
v___x_4879_ = lean_box(0);
v_isShared_4880_ = v_isSharedCheck_4884_;
goto v_resetjp_4878_;
}
v_resetjp_4878_:
{
lean_object* v___x_4882_; 
if (v_isShared_4880_ == 0)
{
v___x_4882_ = v___x_4879_;
goto v_reusejp_4881_;
}
else
{
lean_object* v_reuseFailAlloc_4883_; 
v_reuseFailAlloc_4883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4883_, 0, v_a_4877_);
v___x_4882_ = v_reuseFailAlloc_4883_;
goto v_reusejp_4881_;
}
v_reusejp_4881_:
{
return v___x_4882_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__1___boxed(lean_object* v_snd_4885_, lean_object* v___y_4886_, lean_object* v___y_4887_, lean_object* v___y_4888_, lean_object* v___y_4889_, lean_object* v___y_4890_, lean_object* v___y_4891_, lean_object* v___y_4892_, lean_object* v___y_4893_, lean_object* v___y_4894_){
_start:
{
lean_object* v_res_4895_; 
v_res_4895_ = lp_aesop_Aesop_search___lam__1(v_snd_4885_, v___y_4886_, v___y_4887_, v___y_4888_, v___y_4889_, v___y_4890_, v___y_4891_, v___y_4892_, v___y_4893_);
lean_dec(v___y_4893_);
lean_dec_ref(v___y_4892_);
lean_dec(v___y_4891_);
lean_dec_ref(v___y_4890_);
lean_dec(v___y_4889_);
lean_dec(v___y_4888_);
lean_dec(v___y_4887_);
lean_dec_ref(v___y_4886_);
return v_res_4895_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__2(lean_object* v_snd_4896_, lean_object* v_ruleSet_4897_, lean_object* v_a_4898_, lean_object* v_simpConfig_4899_, lean_object* v_simpConfigSyntax_x3f_4900_, lean_object* v_goal_4901_, lean_object* v___f_4902_, lean_object* v___y_4903_, lean_object* v___y_4904_, lean_object* v___y_4905_, lean_object* v___y_4906_, lean_object* v___y_4907_){
_start:
{
lean_object* v___x_4909_; 
v___x_4909_ = lp_aesop_Aesop_SearchM_run___redArg(v_snd_4896_, v_ruleSet_4897_, v_a_4898_, v_simpConfig_4899_, v_simpConfigSyntax_x3f_4900_, v_goal_4901_, v___f_4902_, v___y_4903_, v___y_4904_, v___y_4905_, v___y_4906_, v___y_4907_);
return v___x_4909_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___lam__2___boxed(lean_object* v_snd_4910_, lean_object* v_ruleSet_4911_, lean_object* v_a_4912_, lean_object* v_simpConfig_4913_, lean_object* v_simpConfigSyntax_x3f_4914_, lean_object* v_goal_4915_, lean_object* v___f_4916_, lean_object* v___y_4917_, lean_object* v___y_4918_, lean_object* v___y_4919_, lean_object* v___y_4920_, lean_object* v___y_4921_, lean_object* v___y_4922_){
_start:
{
lean_object* v_res_4923_; 
v_res_4923_ = lp_aesop_Aesop_search___lam__2(v_snd_4910_, v_ruleSet_4911_, v_a_4912_, v_simpConfig_4913_, v_simpConfigSyntax_x3f_4914_, v_goal_4915_, v___f_4916_, v___y_4917_, v___y_4918_, v___y_4919_, v___y_4920_, v___y_4921_);
lean_dec(v___y_4921_);
lean_dec_ref(v___y_4920_);
lean_dec(v___y_4919_);
lean_dec_ref(v___y_4918_);
lean_dec(v___y_4917_);
return v_res_4923_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(lean_object* v_opt_4924_, lean_object* v___y_4925_){
_start:
{
lean_object* v_options_4927_; uint8_t v___x_4928_; lean_object* v___x_4929_; lean_object* v___x_4930_; 
v_options_4927_ = lean_ctor_get(v___y_4925_, 2);
v___x_4928_ = lp_aesop_Aesop_Check_get(v_options_4927_, v_opt_4924_);
v___x_4929_ = lean_box(v___x_4928_);
v___x_4930_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4930_, 0, v___x_4929_);
return v___x_4930_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg___boxed(lean_object* v_opt_4931_, lean_object* v___y_4932_, lean_object* v___y_4933_){
_start:
{
lean_object* v_res_4934_; 
v_res_4934_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(v_opt_4931_, v___y_4932_);
lean_dec_ref(v___y_4932_);
lean_dec_ref(v_opt_4931_);
return v_res_4934_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0(lean_object* v_opts_4935_, lean_object* v_opt_4936_){
_start:
{
lean_object* v_name_4937_; lean_object* v_defValue_4938_; lean_object* v_map_4939_; lean_object* v___x_4940_; 
v_name_4937_ = lean_ctor_get(v_opt_4936_, 0);
v_defValue_4938_ = lean_ctor_get(v_opt_4936_, 1);
v_map_4939_ = lean_ctor_get(v_opts_4935_, 0);
v___x_4940_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_4939_, v_name_4937_);
if (lean_obj_tag(v___x_4940_) == 0)
{
uint8_t v___x_4941_; 
v___x_4941_ = lean_unbox(v_defValue_4938_);
return v___x_4941_;
}
else
{
lean_object* v_val_4942_; 
v_val_4942_ = lean_ctor_get(v___x_4940_, 0);
lean_inc(v_val_4942_);
lean_dec_ref_known(v___x_4940_, 1);
if (lean_obj_tag(v_val_4942_) == 1)
{
uint8_t v_v_4943_; 
v_v_4943_ = lean_ctor_get_uint8(v_val_4942_, 0);
lean_dec_ref_known(v_val_4942_, 0);
return v_v_4943_;
}
else
{
uint8_t v___x_4944_; 
lean_dec(v_val_4942_);
v___x_4944_ = lean_unbox(v_defValue_4938_);
return v___x_4944_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0___boxed(lean_object* v_opts_4945_, lean_object* v_opt_4946_){
_start:
{
uint8_t v_res_4947_; lean_object* v_r_4948_; 
v_res_4947_ = lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0(v_opts_4945_, v_opt_4946_);
lean_dec_ref(v_opt_4946_);
lean_dec_ref(v_opts_4945_);
v_r_4948_ = lean_box(v_res_4947_);
return v_r_4948_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0(lean_object* v_opts_4949_, lean_object* v_forwardMaxDepth_x3f_4950_, lean_object* v___y_4951_, lean_object* v___y_4952_, lean_object* v___y_4953_, lean_object* v___y_4954_){
_start:
{
uint8_t v_a_4957_; lean_object* v___y_4961_; lean_object* v_options_4964_; lean_object* v___x_4965_; uint8_t v___x_4966_; 
v_options_4964_ = lean_ctor_get(v___y_4953_, 2);
v___x_4965_ = lp_aesop_Aesop_aesop_dev_generateScript;
v___x_4966_ = lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__0(v_options_4964_, v___x_4965_);
if (v___x_4966_ == 0)
{
uint8_t v_traceScript_4967_; 
v_traceScript_4967_ = lean_ctor_get_uint8(v_opts_4949_, sizeof(void*)*6 + 6);
if (v_traceScript_4967_ == 0)
{
lean_object* v___x_4968_; lean_object* v___x_4969_; lean_object* v_a_4970_; uint8_t v___x_4971_; 
v___x_4968_ = lp_aesop_Aesop_Check_script;
v___x_4969_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(v___x_4968_, v___y_4953_);
v_a_4970_ = lean_ctor_get(v___x_4969_, 0);
lean_inc(v_a_4970_);
v___x_4971_ = lean_unbox(v_a_4970_);
lean_dec(v_a_4970_);
if (v___x_4971_ == 0)
{
lean_object* v___x_4972_; lean_object* v___x_4973_; 
lean_dec_ref(v___x_4969_);
v___x_4972_ = lp_aesop_Aesop_Check_script_steps;
v___x_4973_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(v___x_4972_, v___y_4953_);
v___y_4961_ = v___x_4973_;
goto v___jp_4960_;
}
else
{
v___y_4961_ = v___x_4969_;
goto v___jp_4960_;
}
}
else
{
v_a_4957_ = v_traceScript_4967_;
goto v___jp_4956_;
}
}
else
{
v_a_4957_ = v___x_4966_;
goto v___jp_4956_;
}
v___jp_4956_:
{
lean_object* v___x_4958_; lean_object* v___x_4959_; 
v___x_4958_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_4958_, 0, v_opts_4949_);
lean_ctor_set(v___x_4958_, 1, v_forwardMaxDepth_x3f_4950_);
lean_ctor_set_uint8(v___x_4958_, sizeof(void*)*2, v_a_4957_);
v___x_4959_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4959_, 0, v___x_4958_);
return v___x_4959_;
}
v___jp_4960_:
{
lean_object* v_a_4962_; uint8_t v___x_4963_; 
v_a_4962_ = lean_ctor_get(v___y_4961_, 0);
lean_inc(v_a_4962_);
lean_dec_ref(v___y_4961_);
v___x_4963_ = lean_unbox(v_a_4962_);
lean_dec(v_a_4962_);
v_a_4957_ = v___x_4963_;
goto v___jp_4956_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0___boxed(lean_object* v_opts_4974_, lean_object* v_forwardMaxDepth_x3f_4975_, lean_object* v___y_4976_, lean_object* v___y_4977_, lean_object* v___y_4978_, lean_object* v___y_4979_, lean_object* v___y_4980_){
_start:
{
lean_object* v_res_4981_; 
v_res_4981_ = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0(v_opts_4974_, v_forwardMaxDepth_x3f_4975_, v___y_4976_, v___y_4977_, v___y_4978_, v___y_4979_);
lean_dec(v___y_4979_);
lean_dec_ref(v___y_4978_);
lean_dec(v___y_4977_);
lean_dec_ref(v___y_4976_);
return v_res_4981_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search(lean_object* v_goal_4984_, lean_object* v_ruleSet_x3f_4985_, lean_object* v_options_4986_, lean_object* v_simpConfig_4987_, lean_object* v_simpConfigSyntax_x3f_4988_, lean_object* v_stats_4989_, lean_object* v_a_4990_, lean_object* v_a_4991_, lean_object* v_a_4992_, lean_object* v_a_4993_){
_start:
{
lean_object* v___x_4995_; lean_object* v___x_4996_; 
v___x_4995_ = ((lean_object*)(lp_aesop_Aesop_search___closed__0));
lean_inc(v_goal_4984_);
v___x_4996_ = l_Lean_MVarId_checkNotAssigned(v_goal_4984_, v___x_4995_, v_a_4990_, v_a_4991_, v_a_4992_, v_a_4993_);
if (lean_obj_tag(v___x_4996_) == 0)
{
lean_object* v___x_4997_; lean_object* v___x_4998_; 
lean_dec_ref_known(v___x_4996_, 1);
v___x_4997_ = lean_box(0);
v___x_4998_ = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0(v_options_4986_, v___x_4997_, v_a_4990_, v_a_4991_, v_a_4992_, v_a_4993_);
if (lean_obj_tag(v___x_4998_) == 0)
{
lean_object* v_a_4999_; lean_object* v_ruleSet_5001_; lean_object* v___y_5002_; lean_object* v___y_5003_; lean_object* v___y_5004_; lean_object* v___y_5005_; 
v_a_4999_ = lean_ctor_get(v___x_4998_, 0);
lean_inc(v_a_4999_);
lean_dec_ref_known(v___x_4998_, 1);
if (lean_obj_tag(v_ruleSet_x3f_4985_) == 0)
{
lean_object* v___x_5039_; 
v___x_5039_ = lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets(v_a_4992_, v_a_4993_);
if (lean_obj_tag(v___x_5039_) == 0)
{
lean_object* v_a_5040_; lean_object* v___x_5041_; 
v_a_5040_ = lean_ctor_get(v___x_5039_, 0);
lean_inc(v_a_5040_);
lean_dec_ref_known(v___x_5039_, 1);
v___x_5041_ = lp_aesop_Aesop_mkLocalRuleSet(v_a_5040_, v_a_4999_, v_a_4992_, v_a_4993_);
lean_dec(v_a_5040_);
if (lean_obj_tag(v___x_5041_) == 0)
{
lean_object* v_a_5042_; 
v_a_5042_ = lean_ctor_get(v___x_5041_, 0);
lean_inc(v_a_5042_);
lean_dec_ref_known(v___x_5041_, 1);
v_ruleSet_5001_ = v_a_5042_;
v___y_5002_ = v_a_4990_;
v___y_5003_ = v_a_4991_;
v___y_5004_ = v_a_4992_;
v___y_5005_ = v_a_4993_;
goto v___jp_5000_;
}
else
{
lean_object* v_a_5043_; lean_object* v___x_5045_; uint8_t v_isShared_5046_; uint8_t v_isSharedCheck_5050_; 
lean_dec(v_a_4999_);
lean_dec_ref(v_stats_4989_);
lean_dec(v_simpConfigSyntax_x3f_4988_);
lean_dec_ref(v_simpConfig_4987_);
lean_dec(v_goal_4984_);
v_a_5043_ = lean_ctor_get(v___x_5041_, 0);
v_isSharedCheck_5050_ = !lean_is_exclusive(v___x_5041_);
if (v_isSharedCheck_5050_ == 0)
{
v___x_5045_ = v___x_5041_;
v_isShared_5046_ = v_isSharedCheck_5050_;
goto v_resetjp_5044_;
}
else
{
lean_inc(v_a_5043_);
lean_dec(v___x_5041_);
v___x_5045_ = lean_box(0);
v_isShared_5046_ = v_isSharedCheck_5050_;
goto v_resetjp_5044_;
}
v_resetjp_5044_:
{
lean_object* v___x_5048_; 
if (v_isShared_5046_ == 0)
{
v___x_5048_ = v___x_5045_;
goto v_reusejp_5047_;
}
else
{
lean_object* v_reuseFailAlloc_5049_; 
v_reuseFailAlloc_5049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5049_, 0, v_a_5043_);
v___x_5048_ = v_reuseFailAlloc_5049_;
goto v_reusejp_5047_;
}
v_reusejp_5047_:
{
return v___x_5048_;
}
}
}
}
else
{
lean_object* v_a_5051_; lean_object* v___x_5053_; uint8_t v_isShared_5054_; uint8_t v_isSharedCheck_5058_; 
lean_dec(v_a_4999_);
lean_dec_ref(v_stats_4989_);
lean_dec(v_simpConfigSyntax_x3f_4988_);
lean_dec_ref(v_simpConfig_4987_);
lean_dec(v_goal_4984_);
v_a_5051_ = lean_ctor_get(v___x_5039_, 0);
v_isSharedCheck_5058_ = !lean_is_exclusive(v___x_5039_);
if (v_isSharedCheck_5058_ == 0)
{
v___x_5053_ = v___x_5039_;
v_isShared_5054_ = v_isSharedCheck_5058_;
goto v_resetjp_5052_;
}
else
{
lean_inc(v_a_5051_);
lean_dec(v___x_5039_);
v___x_5053_ = lean_box(0);
v_isShared_5054_ = v_isSharedCheck_5058_;
goto v_resetjp_5052_;
}
v_resetjp_5052_:
{
lean_object* v___x_5056_; 
if (v_isShared_5054_ == 0)
{
v___x_5056_ = v___x_5053_;
goto v_reusejp_5055_;
}
else
{
lean_object* v_reuseFailAlloc_5057_; 
v_reuseFailAlloc_5057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5057_, 0, v_a_5051_);
v___x_5056_ = v_reuseFailAlloc_5057_;
goto v_reusejp_5055_;
}
v_reusejp_5055_:
{
return v___x_5056_;
}
}
}
}
else
{
lean_object* v_val_5059_; 
v_val_5059_ = lean_ctor_get(v_ruleSet_x3f_4985_, 0);
lean_inc(v_val_5059_);
lean_dec_ref_known(v_ruleSet_x3f_4985_, 1);
v_ruleSet_5001_ = v_val_5059_;
v___y_5002_ = v_a_4990_;
v___y_5003_ = v_a_4991_;
v___y_5004_ = v_a_4992_;
v___y_5005_ = v_a_4993_;
goto v___jp_5000_;
}
v___jp_5000_:
{
lean_object* v_toOptions_5006_; lean_object* v___x_5007_; lean_object* v_snd_5008_; lean_object* v___f_5009_; lean_object* v___f_5010_; lean_object* v___x_5011_; 
v_toOptions_5006_ = lean_ctor_get(v_a_4999_, 0);
v___x_5007_ = lp_aesop_Aesop_Options_queue(v_toOptions_5006_);
v_snd_5008_ = lean_ctor_get(v___x_5007_, 1);
lean_inc_n(v_snd_5008_, 2);
lean_dec_ref(v___x_5007_);
v___f_5009_ = lean_alloc_closure((void*)(lp_aesop_Aesop_search___lam__1___boxed), 10, 1);
lean_closure_set(v___f_5009_, 0, v_snd_5008_);
v___f_5010_ = lean_alloc_closure((void*)(lp_aesop_Aesop_search___lam__2___boxed), 13, 7);
lean_closure_set(v___f_5010_, 0, v_snd_5008_);
lean_closure_set(v___f_5010_, 1, v_ruleSet_5001_);
lean_closure_set(v___f_5010_, 2, v_a_4999_);
lean_closure_set(v___f_5010_, 3, v_simpConfig_4987_);
lean_closure_set(v___f_5010_, 4, v_simpConfigSyntax_x3f_4988_);
lean_closure_set(v___f_5010_, 5, v_goal_4984_);
lean_closure_set(v___f_5010_, 6, v___f_5009_);
v___x_5011_ = lp_aesop_Aesop_BaseM_run___redArg(v___f_5010_, v_stats_4989_, v___y_5002_, v___y_5003_, v___y_5004_, v___y_5005_);
if (lean_obj_tag(v___x_5011_) == 0)
{
lean_object* v_a_5012_; lean_object* v___x_5014_; uint8_t v_isShared_5015_; uint8_t v_isSharedCheck_5030_; 
v_a_5012_ = lean_ctor_get(v___x_5011_, 0);
v_isSharedCheck_5030_ = !lean_is_exclusive(v___x_5011_);
if (v_isSharedCheck_5030_ == 0)
{
v___x_5014_ = v___x_5011_;
v_isShared_5015_ = v_isSharedCheck_5030_;
goto v_resetjp_5013_;
}
else
{
lean_inc(v_a_5012_);
lean_dec(v___x_5011_);
v___x_5014_ = lean_box(0);
v_isShared_5015_ = v_isSharedCheck_5030_;
goto v_resetjp_5013_;
}
v_resetjp_5013_:
{
lean_object* v_fst_5016_; lean_object* v_snd_5017_; lean_object* v_fst_5018_; lean_object* v___x_5020_; uint8_t v_isShared_5021_; uint8_t v_isSharedCheck_5028_; 
v_fst_5016_ = lean_ctor_get(v_a_5012_, 0);
lean_inc(v_fst_5016_);
v_snd_5017_ = lean_ctor_get(v_a_5012_, 1);
lean_inc(v_snd_5017_);
lean_dec(v_a_5012_);
v_fst_5018_ = lean_ctor_get(v_fst_5016_, 0);
v_isSharedCheck_5028_ = !lean_is_exclusive(v_fst_5016_);
if (v_isSharedCheck_5028_ == 0)
{
lean_object* v_unused_5029_; 
v_unused_5029_ = lean_ctor_get(v_fst_5016_, 1);
lean_dec(v_unused_5029_);
v___x_5020_ = v_fst_5016_;
v_isShared_5021_ = v_isSharedCheck_5028_;
goto v_resetjp_5019_;
}
else
{
lean_inc(v_fst_5018_);
lean_dec(v_fst_5016_);
v___x_5020_ = lean_box(0);
v_isShared_5021_ = v_isSharedCheck_5028_;
goto v_resetjp_5019_;
}
v_resetjp_5019_:
{
lean_object* v___x_5023_; 
if (v_isShared_5021_ == 0)
{
lean_ctor_set(v___x_5020_, 1, v_snd_5017_);
v___x_5023_ = v___x_5020_;
goto v_reusejp_5022_;
}
else
{
lean_object* v_reuseFailAlloc_5027_; 
v_reuseFailAlloc_5027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5027_, 0, v_fst_5018_);
lean_ctor_set(v_reuseFailAlloc_5027_, 1, v_snd_5017_);
v___x_5023_ = v_reuseFailAlloc_5027_;
goto v_reusejp_5022_;
}
v_reusejp_5022_:
{
lean_object* v___x_5025_; 
if (v_isShared_5015_ == 0)
{
lean_ctor_set(v___x_5014_, 0, v___x_5023_);
v___x_5025_ = v___x_5014_;
goto v_reusejp_5024_;
}
else
{
lean_object* v_reuseFailAlloc_5026_; 
v_reuseFailAlloc_5026_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5026_, 0, v___x_5023_);
v___x_5025_ = v_reuseFailAlloc_5026_;
goto v_reusejp_5024_;
}
v_reusejp_5024_:
{
return v___x_5025_;
}
}
}
}
}
else
{
lean_object* v_a_5031_; lean_object* v___x_5033_; uint8_t v_isShared_5034_; uint8_t v_isSharedCheck_5038_; 
v_a_5031_ = lean_ctor_get(v___x_5011_, 0);
v_isSharedCheck_5038_ = !lean_is_exclusive(v___x_5011_);
if (v_isSharedCheck_5038_ == 0)
{
v___x_5033_ = v___x_5011_;
v_isShared_5034_ = v_isSharedCheck_5038_;
goto v_resetjp_5032_;
}
else
{
lean_inc(v_a_5031_);
lean_dec(v___x_5011_);
v___x_5033_ = lean_box(0);
v_isShared_5034_ = v_isSharedCheck_5038_;
goto v_resetjp_5032_;
}
v_resetjp_5032_:
{
lean_object* v___x_5036_; 
if (v_isShared_5034_ == 0)
{
v___x_5036_ = v___x_5033_;
goto v_reusejp_5035_;
}
else
{
lean_object* v_reuseFailAlloc_5037_; 
v_reuseFailAlloc_5037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5037_, 0, v_a_5031_);
v___x_5036_ = v_reuseFailAlloc_5037_;
goto v_reusejp_5035_;
}
v_reusejp_5035_:
{
return v___x_5036_;
}
}
}
}
}
else
{
lean_object* v_a_5060_; lean_object* v___x_5062_; uint8_t v_isShared_5063_; uint8_t v_isSharedCheck_5067_; 
lean_dec_ref(v_stats_4989_);
lean_dec(v_simpConfigSyntax_x3f_4988_);
lean_dec_ref(v_simpConfig_4987_);
lean_dec(v_ruleSet_x3f_4985_);
lean_dec(v_goal_4984_);
v_a_5060_ = lean_ctor_get(v___x_4998_, 0);
v_isSharedCheck_5067_ = !lean_is_exclusive(v___x_4998_);
if (v_isSharedCheck_5067_ == 0)
{
v___x_5062_ = v___x_4998_;
v_isShared_5063_ = v_isSharedCheck_5067_;
goto v_resetjp_5061_;
}
else
{
lean_inc(v_a_5060_);
lean_dec(v___x_4998_);
v___x_5062_ = lean_box(0);
v_isShared_5063_ = v_isSharedCheck_5067_;
goto v_resetjp_5061_;
}
v_resetjp_5061_:
{
lean_object* v___x_5065_; 
if (v_isShared_5063_ == 0)
{
v___x_5065_ = v___x_5062_;
goto v_reusejp_5064_;
}
else
{
lean_object* v_reuseFailAlloc_5066_; 
v_reuseFailAlloc_5066_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5066_, 0, v_a_5060_);
v___x_5065_ = v_reuseFailAlloc_5066_;
goto v_reusejp_5064_;
}
v_reusejp_5064_:
{
return v___x_5065_;
}
}
}
}
else
{
lean_object* v_a_5068_; lean_object* v___x_5070_; uint8_t v_isShared_5071_; uint8_t v_isSharedCheck_5075_; 
lean_dec_ref(v_stats_4989_);
lean_dec(v_simpConfigSyntax_x3f_4988_);
lean_dec_ref(v_simpConfig_4987_);
lean_dec_ref(v_options_4986_);
lean_dec(v_ruleSet_x3f_4985_);
lean_dec(v_goal_4984_);
v_a_5068_ = lean_ctor_get(v___x_4996_, 0);
v_isSharedCheck_5075_ = !lean_is_exclusive(v___x_4996_);
if (v_isSharedCheck_5075_ == 0)
{
v___x_5070_ = v___x_4996_;
v_isShared_5071_ = v_isSharedCheck_5075_;
goto v_resetjp_5069_;
}
else
{
lean_inc(v_a_5068_);
lean_dec(v___x_4996_);
v___x_5070_ = lean_box(0);
v_isShared_5071_ = v_isSharedCheck_5075_;
goto v_resetjp_5069_;
}
v_resetjp_5069_:
{
lean_object* v___x_5073_; 
if (v_isShared_5071_ == 0)
{
v___x_5073_ = v___x_5070_;
goto v_reusejp_5072_;
}
else
{
lean_object* v_reuseFailAlloc_5074_; 
v_reuseFailAlloc_5074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5074_, 0, v_a_5068_);
v___x_5073_ = v_reuseFailAlloc_5074_;
goto v_reusejp_5072_;
}
v_reusejp_5072_:
{
return v___x_5073_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_search___boxed(lean_object* v_goal_5076_, lean_object* v_ruleSet_x3f_5077_, lean_object* v_options_5078_, lean_object* v_simpConfig_5079_, lean_object* v_simpConfigSyntax_x3f_5080_, lean_object* v_stats_5081_, lean_object* v_a_5082_, lean_object* v_a_5083_, lean_object* v_a_5084_, lean_object* v_a_5085_, lean_object* v_a_5086_){
_start:
{
lean_object* v_res_5087_; 
v_res_5087_ = lp_aesop_Aesop_search(v_goal_5076_, v_ruleSet_x3f_5077_, v_options_5078_, v_simpConfig_5079_, v_simpConfigSyntax_x3f_5080_, v_stats_5081_, v_a_5082_, v_a_5083_, v_a_5084_, v_a_5085_);
lean_dec(v_a_5085_);
lean_dec_ref(v_a_5084_);
lean_dec(v_a_5083_);
lean_dec_ref(v_a_5082_);
return v_res_5087_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1(lean_object* v_opt_5088_, lean_object* v___y_5089_, lean_object* v___y_5090_, lean_object* v___y_5091_, lean_object* v___y_5092_){
_start:
{
lean_object* v___x_5094_; 
v___x_5094_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___redArg(v_opt_5088_, v___y_5091_);
return v___x_5094_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1___boxed(lean_object* v_opt_5095_, lean_object* v___y_5096_, lean_object* v___y_5097_, lean_object* v___y_5098_, lean_object* v___y_5099_, lean_object* v___y_5100_){
_start:
{
lean_object* v_res_5101_; 
v_res_5101_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_search_spec__0_spec__1(v_opt_5095_, v___y_5096_, v___y_5097_, v___y_5098_, v___y_5099_);
lean_dec(v___y_5099_);
lean_dec_ref(v___y_5098_);
lean_dec(v___y_5097_);
lean_dec_ref(v___y_5096_);
lean_dec_ref(v_opt_5095_);
return v_res_5101_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_Main(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_ExpandSafePrefix(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Check(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_ExtractProof(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_ExtractScript(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Tracing(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_Queue(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Free(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Stats(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_Main(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_ExpandSafePrefix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_ExtractProof(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_ExtractScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Queue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Stats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_Main(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Script_Main(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_ExpandSafePrefix(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Check(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_ExtractProof(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_ExtractScript(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Tracing(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Queue(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Free(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Stats(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Main(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_ExpandSafePrefix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_ExtractProof(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_ExtractScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Queue(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Stats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_Main(builtin);
}
#ifdef __cplusplus
}
#endif
