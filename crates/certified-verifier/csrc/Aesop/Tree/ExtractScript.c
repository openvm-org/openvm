// Lean compiler output
// Module: Aesop.Tree.ExtractScript
// Imports: public import Init public meta import Init public import Aesop.Script.UScript public import Aesop.Tree.TreeM
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lp_aesop_Aesop_RegularRule_name(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
size_t lean_array_size(lean_object*);
lean_object* lp_aesop_Aesop_Script_LazyStep_toStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_script;
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_of_nat(lean_object*);
double lean_float_div(double, double);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_instMonadExceptOfEIO(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptReaderT___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_NormalizationState_isProvenByNormalization(lean_object*);
lean_object* lp_aesop_Aesop_Goal_safeRapps(lean_object*);
lean_object* lp_aesop_Aesop_Script_Step_mkSorry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_MVarCluster_provenGoal_x3f(lean_object*);
lean_object* lp_aesop_Aesop_Goal_firstProvenRapp_x3f(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instExceptToTraceResult___lam__0___boxed(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_aesop_Aesop_getRootGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TreeM_instMonad;
lean_object* l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ExtractScriptM_run___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ExtractScriptM_run___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ExtractScriptM_run___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_ExtractScriptM_run___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_ExtractScriptM_run___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_ExtractScriptM_run___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_ExtractScriptM_run___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScriptM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScriptM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScriptM_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScriptM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__0 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__1;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__2 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__2_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__3 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__3_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__4 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__4_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__5 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__5_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__6 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__6_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__7 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__7_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__8 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__8_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__9 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__9_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__10 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__10_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__11 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__11_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__12 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__12_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "tactic script generation failed for rule "};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__13 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__13_value;
static lean_once_cell_t lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__14;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__15 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__15_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__16 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__16_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__17 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__17_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "<norm simp>"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__18 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__18_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "<norm unfold>"};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__19 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__19_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ExtractScript_lazyStepsToSteps_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ExtractScript_lazyStepsToSteps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "tactic script generation is not supported by rule "};
static const lean_object* lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__0 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_lazyStepsToSteps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordStep___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordStep___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordLazySteps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordLazySteps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_ExtractScript_visitGoal_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_ExtractScript_visitGoal_spec__5___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "expected goal "};
static const lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__1;
static const lean_string_object lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = " to be normalised"};
static const lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__2(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__6___boxed(lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0;
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__1 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__1_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___boxed(lean_object**);
static const lean_string_object lp_aesop_Aesop_ExtractScript_visitGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goal "};
static const lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___closed__0 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_visitGoal___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ExtractScript_visitGoal___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___closed__1;
static lean_once_cell_t lp_aesop_Aesop_ExtractScript_visitGoal___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_ExtractScript_visitGoal___closed__2;
static const lean_string_object lp_aesop_Aesop_ExtractScript_visitGoal___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___closed__3 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_visitGoal___closed__3_value;
static const lean_string_object lp_aesop_Aesop_ExtractScript_visitGoal___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___closed__4 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_visitGoal___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_ExtractScript_visitGoal___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_ExtractScript_visitGoal___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___closed__5 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_visitGoal___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ExtractScript_visitRapp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "rapp "};
static const lean_object* lp_aesop_Aesop_ExtractScript_visitRapp___closed__0 = (const lean_object*)&lp_aesop_Aesop_ExtractScript_visitRapp___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ExtractScript_visitRapp___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ExtractScript_visitRapp___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitRapp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitRapp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_MVarClusterRef_extractScriptCore_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_extractScriptCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_GoalRef_extractScriptCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = " does not have a proven rapp"};
static const lean_object* lp_aesop_Aesop_GoalRef_extractScriptCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_GoalRef_extractScriptCore___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_extractScriptCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_extractScriptCore___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_extractScriptCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "the mvar cluster with goals "};
static const lean_object* lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__1;
static const lean_string_object lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = " does not contain a proven goal"};
static const lean_object* lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__2 = (const lean_object*)&lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_extractScriptCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RappRef_extractScriptCore_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RappRef_extractScriptCore_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_extractScriptCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_extractScriptCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_extractScriptCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_extractScript___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Extract script"};
static const lean_object* lp_aesop_Aesop_extractScript___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_extractScript___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_extractScript___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractScript___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractScript___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_extractScript___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_extractScript___closed__0 = (const lean_object*)&lp_aesop_Aesop_extractScript___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_extractScript___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_extractScript___closed__1 = (const lean_object*)&lp_aesop_Aesop_extractScript___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__2;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__3;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__4;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__5;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__6;
static const lean_closure_object lp_aesop_Aesop_extractScript___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_extractScript___closed__7 = (const lean_object*)&lp_aesop_Aesop_extractScript___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_extractScript___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_extractScript___closed__8 = (const lean_object*)&lp_aesop_Aesop_extractScript___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__9;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__10;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__11;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__12;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__13;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__14;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__15;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__16;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__17;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__18;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__19;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__20;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__21;
static const lean_closure_object lp_aesop_Aesop_extractScript___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_extractScript___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_extractScript___closed__22 = (const lean_object*)&lp_aesop_Aesop_extractScript___closed__22_value;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__23;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__24;
static lean_once_cell_t lp_aesop_Aesop_extractScript___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractScript___closed__25;
static const lean_closure_object lp_aesop_Aesop_extractScript___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instExceptToTraceResult___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_extractScript___closed__26 = (const lean_object*)&lp_aesop_Aesop_extractScript___closed__26_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractScript(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractScript___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "aesop: internal error at extractSafePrefixScript: goal "};
static const lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__1;
static const lean_string_object lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = " is not normalised"};
static const lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__2 = (const lean_object*)&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__3;
static const lean_string_object lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "aesop: internal error: goal "};
static const lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__4 = (const lean_object*)&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__5;
static const lean_string_object lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " has "};
static const lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__6 = (const lean_object*)&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__7;
static const lean_string_object lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " safe rapps"};
static const lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__8 = (const lean_object*)&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_extractSafePrefixScriptCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_extractSafePrefixScriptCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_MVarClusterRef_extractSafePrefixScriptCore_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_MVarClusterRef_extractSafePrefixScriptCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_extractSafePrefixScriptCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_extractSafePrefixScriptCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Extract safe prefix script"};
static const lean_object* lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefixScript___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefixScript___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__3___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_extractSafePrefixScript___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_extractSafePrefixScript___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_extractSafePrefixScript___closed__0 = (const lean_object*)&lp_aesop_Aesop_extractSafePrefixScript___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefixScript(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefixScript___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScriptM_run___redArg(lean_object* v_x_6_, lean_object* v_a_7_, lean_object* v_a_8_, lean_object* v_a_9_, lean_object* v_a_10_, lean_object* v_a_11_, lean_object* v_a_12_, lean_object* v_a_13_){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_15_ = ((lean_object*)(lp_aesop_Aesop_ExtractScriptM_run___redArg___closed__1));
v___x_16_ = lean_st_mk_ref(v___x_15_);
lean_inc(v_a_13_);
lean_inc_ref(v_a_12_);
lean_inc(v_a_11_);
lean_inc_ref(v_a_10_);
lean_inc(v_a_9_);
lean_inc(v_a_8_);
lean_inc_ref(v_a_7_);
lean_inc(v___x_16_);
v___x_17_ = lean_apply_9(v_x_6_, v___x_16_, v_a_7_, v_a_8_, v_a_9_, v_a_10_, v_a_11_, v_a_12_, v_a_13_, lean_box(0));
if (lean_obj_tag(v___x_17_) == 0)
{
lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_29_; 
v_isSharedCheck_29_ = !lean_is_exclusive(v___x_17_);
if (v_isSharedCheck_29_ == 0)
{
lean_object* v_unused_30_; 
v_unused_30_ = lean_ctor_get(v___x_17_, 0);
lean_dec(v_unused_30_);
v___x_19_ = v___x_17_;
v_isShared_20_ = v_isSharedCheck_29_;
goto v_resetjp_18_;
}
else
{
lean_dec(v___x_17_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_29_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v___x_21_; lean_object* v_script_22_; uint8_t v_proofHasMVar_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_27_; 
v___x_21_ = lean_st_ref_get(v___x_16_);
lean_dec(v___x_16_);
v_script_22_ = lean_ctor_get(v___x_21_, 0);
lean_inc_ref(v_script_22_);
v_proofHasMVar_23_ = lean_ctor_get_uint8(v___x_21_, sizeof(void*)*1);
lean_dec(v___x_21_);
v___x_24_ = lean_box(v_proofHasMVar_23_);
v___x_25_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_25_, 0, v_script_22_);
lean_ctor_set(v___x_25_, 1, v___x_24_);
if (v_isShared_20_ == 0)
{
lean_ctor_set(v___x_19_, 0, v___x_25_);
v___x_27_ = v___x_19_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_28_; 
v_reuseFailAlloc_28_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_28_, 0, v___x_25_);
v___x_27_ = v_reuseFailAlloc_28_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
return v___x_27_;
}
}
}
else
{
lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_38_; 
lean_dec(v___x_16_);
v_a_31_ = lean_ctor_get(v___x_17_, 0);
v_isSharedCheck_38_ = !lean_is_exclusive(v___x_17_);
if (v_isSharedCheck_38_ == 0)
{
v___x_33_ = v___x_17_;
v_isShared_34_ = v_isSharedCheck_38_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_a_31_);
lean_dec(v___x_17_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_38_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_36_; 
if (v_isShared_34_ == 0)
{
v___x_36_ = v___x_33_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_37_; 
v_reuseFailAlloc_37_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_37_, 0, v_a_31_);
v___x_36_ = v_reuseFailAlloc_37_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
return v___x_36_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScriptM_run___redArg___boxed(lean_object* v_x_39_, lean_object* v_a_40_, lean_object* v_a_41_, lean_object* v_a_42_, lean_object* v_a_43_, lean_object* v_a_44_, lean_object* v_a_45_, lean_object* v_a_46_, lean_object* v_a_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v_x_39_, v_a_40_, v_a_41_, v_a_42_, v_a_43_, v_a_44_, v_a_45_, v_a_46_);
lean_dec(v_a_46_);
lean_dec_ref(v_a_45_);
lean_dec(v_a_44_);
lean_dec_ref(v_a_43_);
lean_dec(v_a_42_);
lean_dec(v_a_41_);
lean_dec_ref(v_a_40_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScriptM_run(lean_object* v_00_u03b1_49_, lean_object* v_x_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v_x_50_, v_a_51_, v_a_52_, v_a_53_, v_a_54_, v_a_55_, v_a_56_, v_a_57_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScriptM_run___boxed(lean_object* v_00_u03b1_60_, lean_object* v_x_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_aesop_Aesop_ExtractScriptM_run(v_00_u03b1_60_, v_x_61_, v_a_62_, v_a_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_, v_a_68_);
lean_dec(v_a_68_);
lean_dec_ref(v_a_67_);
lean_dec(v_a_66_);
lean_dec_ref(v_a_65_);
lean_dec(v_a_64_);
lean_dec(v_a_63_);
lean_dec_ref(v_a_62_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0(lean_object* v_msgData_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_){
_start:
{
lean_object* v___x_77_; lean_object* v_env_78_; lean_object* v___x_79_; lean_object* v_mctx_80_; lean_object* v_lctx_81_; lean_object* v_options_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_77_ = lean_st_ref_get(v___y_75_);
v_env_78_ = lean_ctor_get(v___x_77_, 0);
lean_inc_ref(v_env_78_);
lean_dec(v___x_77_);
v___x_79_ = lean_st_ref_get(v___y_73_);
v_mctx_80_ = lean_ctor_get(v___x_79_, 0);
lean_inc_ref(v_mctx_80_);
lean_dec(v___x_79_);
v_lctx_81_ = lean_ctor_get(v___y_72_, 2);
v_options_82_ = lean_ctor_get(v___y_74_, 2);
lean_inc_ref(v_options_82_);
lean_inc_ref(v_lctx_81_);
v___x_83_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_83_, 0, v_env_78_);
lean_ctor_set(v___x_83_, 1, v_mctx_80_);
lean_ctor_set(v___x_83_, 2, v_lctx_81_);
lean_ctor_set(v___x_83_, 3, v_options_82_);
v___x_84_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_msgData_71_);
v___x_85_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0___boxed(lean_object* v_msgData_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0(v_msgData_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
lean_dec(v___y_88_);
lean_dec_ref(v___y_87_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___redArg(lean_object* v_msg_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v_ref_99_; lean_object* v___x_100_; lean_object* v_a_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_109_; 
v_ref_99_ = lean_ctor_get(v___y_96_, 5);
v___x_100_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0(v_msg_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
v_a_101_ = lean_ctor_get(v___x_100_, 0);
v_isSharedCheck_109_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_109_ == 0)
{
v___x_103_ = v___x_100_;
v_isShared_104_ = v_isSharedCheck_109_;
goto v_resetjp_102_;
}
else
{
lean_inc(v_a_101_);
lean_dec(v___x_100_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_109_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v___x_105_; lean_object* v___x_107_; 
lean_inc(v_ref_99_);
v___x_105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_105_, 0, v_ref_99_);
lean_ctor_set(v___x_105_, 1, v_a_101_);
if (v_isShared_104_ == 0)
{
lean_ctor_set_tag(v___x_103_, 1);
lean_ctor_set(v___x_103_, 0, v___x_105_);
v___x_107_ = v___x_103_;
goto v_reusejp_106_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v___x_105_);
v___x_107_ = v_reuseFailAlloc_108_;
goto v_reusejp_106_;
}
v_reusejp_106_:
{
return v___x_107_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___redArg___boxed(lean_object* v_msg_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___redArg(v_msg_110_, v___y_111_, v___y_112_, v___y_113_, v___y_114_);
lean_dec(v___y_114_);
lean_dec_ref(v___y_113_);
lean_dec(v___y_112_);
lean_dec_ref(v___y_111_);
return v_res_116_;
}
}
static lean_object* _init_lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__1(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_118_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__0));
v___x_119_ = l_Lean_stringToMessageData(v___x_118_);
return v___x_119_;
}
}
static lean_object* _init_lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__14(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_132_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__13));
v___x_133_ = l_Lean_stringToMessageData(v___x_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep(lean_object* v_ruleName_139_, lean_object* v_lstep_140_, lean_object* v_a_141_, lean_object* v_a_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_aesop_Aesop_Script_LazyStep_toStep(v_lstep_140_, v_a_141_, v_a_142_, v_a_143_, v_a_144_);
if (lean_obj_tag(v___x_146_) == 0)
{
lean_dec(v_ruleName_139_);
return v___x_146_;
}
else
{
lean_object* v_a_147_; lean_object* v___y_149_; lean_object* v___y_150_; lean_object* v_name_161_; lean_object* v___y_162_; lean_object* v___y_163_; lean_object* v___y_164_; lean_object* v___y_165_; lean_object* v_name_172_; uint8_t v_scope_173_; lean_object* v___y_174_; lean_object* v___y_175_; lean_object* v___y_176_; lean_object* v___y_177_; lean_object* v_name_183_; uint8_t v_builder_184_; uint8_t v_scope_185_; lean_object* v___y_186_; lean_object* v___y_187_; uint8_t v___y_199_; uint8_t v___x_217_; 
v_a_147_ = lean_ctor_get(v___x_146_, 0);
lean_inc(v_a_147_);
v___x_217_ = l_Lean_Exception_isInterrupt(v_a_147_);
if (v___x_217_ == 0)
{
uint8_t v___x_218_; 
lean_inc(v_a_147_);
v___x_218_ = l_Lean_Exception_isRuntime(v_a_147_);
v___y_199_ = v___x_218_;
goto v___jp_198_;
}
else
{
v___y_199_ = v___x_217_;
goto v___jp_198_;
}
v___jp_148_:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_151_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_151_, 0, v___y_150_);
v___x_152_ = l_Lean_MessageData_ofFormat(v___x_151_);
lean_inc_ref(v___y_149_);
v___x_153_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_153_, 0, v___y_149_);
lean_ctor_set(v___x_153_, 1, v___x_152_);
v___x_154_ = lean_obj_once(&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__1, &lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__1_once, _init_lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__1);
v___x_155_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_153_);
lean_ctor_set(v___x_155_, 1, v___x_154_);
v___x_156_ = l_Lean_Exception_toMessageData(v_a_147_);
v___x_157_ = l_Lean_indentD(v___x_156_);
v___x_158_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_155_);
lean_ctor_set(v___x_158_, 1, v___x_157_);
v___x_159_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___redArg(v___x_158_, v_a_141_, v_a_142_, v_a_143_, v_a_144_);
return v___x_159_;
}
v___jp_160_:
{
lean_object* v___x_166_; lean_object* v___x_167_; uint8_t v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_166_ = lean_string_append(v___y_163_, v___y_165_);
v___x_167_ = lean_string_append(v___x_166_, v___y_162_);
v___x_168_ = 1;
v___x_169_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_161_, v___x_168_);
v___x_170_ = lean_string_append(v___x_167_, v___x_169_);
lean_dec_ref(v___x_169_);
v___y_149_ = v___y_164_;
v___y_150_ = v___x_170_;
goto v___jp_148_;
}
v___jp_171_:
{
lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_178_ = lean_string_append(v___y_175_, v___y_177_);
v___x_179_ = lean_string_append(v___x_178_, v___y_174_);
if (v_scope_173_ == 0)
{
lean_object* v___x_180_; 
v___x_180_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__2));
v_name_161_ = v_name_172_;
v___y_162_ = v___y_174_;
v___y_163_ = v___x_179_;
v___y_164_ = v___y_176_;
v___y_165_ = v___x_180_;
goto v___jp_160_;
}
else
{
lean_object* v___x_181_; 
v___x_181_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__3));
v_name_161_ = v_name_172_;
v___y_162_ = v___y_174_;
v___y_163_ = v___x_179_;
v___y_164_ = v___y_176_;
v___y_165_ = v___x_181_;
goto v___jp_160_;
}
}
v___jp_182_:
{
lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_188_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__4));
lean_inc_ref(v___y_187_);
v___x_189_ = lean_string_append(v___y_187_, v___x_188_);
switch(v_builder_184_)
{
case 0:
{
lean_object* v___x_190_; 
v___x_190_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__5));
v_name_172_ = v_name_183_;
v_scope_173_ = v_scope_185_;
v___y_174_ = v___x_188_;
v___y_175_ = v___x_189_;
v___y_176_ = v___y_186_;
v___y_177_ = v___x_190_;
goto v___jp_171_;
}
case 1:
{
lean_object* v___x_191_; 
v___x_191_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__6));
v_name_172_ = v_name_183_;
v_scope_173_ = v_scope_185_;
v___y_174_ = v___x_188_;
v___y_175_ = v___x_189_;
v___y_176_ = v___y_186_;
v___y_177_ = v___x_191_;
goto v___jp_171_;
}
case 2:
{
lean_object* v___x_192_; 
v___x_192_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__7));
v_name_172_ = v_name_183_;
v_scope_173_ = v_scope_185_;
v___y_174_ = v___x_188_;
v___y_175_ = v___x_189_;
v___y_176_ = v___y_186_;
v___y_177_ = v___x_192_;
goto v___jp_171_;
}
case 3:
{
lean_object* v___x_193_; 
v___x_193_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__8));
v_name_172_ = v_name_183_;
v_scope_173_ = v_scope_185_;
v___y_174_ = v___x_188_;
v___y_175_ = v___x_189_;
v___y_176_ = v___y_186_;
v___y_177_ = v___x_193_;
goto v___jp_171_;
}
case 4:
{
lean_object* v___x_194_; 
v___x_194_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__9));
v_name_172_ = v_name_183_;
v_scope_173_ = v_scope_185_;
v___y_174_ = v___x_188_;
v___y_175_ = v___x_189_;
v___y_176_ = v___y_186_;
v___y_177_ = v___x_194_;
goto v___jp_171_;
}
case 5:
{
lean_object* v___x_195_; 
v___x_195_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__10));
v_name_172_ = v_name_183_;
v_scope_173_ = v_scope_185_;
v___y_174_ = v___x_188_;
v___y_175_ = v___x_189_;
v___y_176_ = v___y_186_;
v___y_177_ = v___x_195_;
goto v___jp_171_;
}
case 6:
{
lean_object* v___x_196_; 
v___x_196_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__11));
v_name_172_ = v_name_183_;
v_scope_173_ = v_scope_185_;
v___y_174_ = v___x_188_;
v___y_175_ = v___x_189_;
v___y_176_ = v___y_186_;
v___y_177_ = v___x_196_;
goto v___jp_171_;
}
default: 
{
lean_object* v___x_197_; 
v___x_197_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__12));
v_name_172_ = v_name_183_;
v_scope_173_ = v_scope_185_;
v___y_174_ = v___x_188_;
v___y_175_ = v___x_189_;
v___y_176_ = v___y_186_;
v___y_177_ = v___x_197_;
goto v___jp_171_;
}
}
}
v___jp_198_:
{
if (v___y_199_ == 0)
{
lean_object* v___x_200_; 
lean_dec_ref_known(v___x_146_, 1);
v___x_200_ = lean_obj_once(&lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__14, &lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__14_once, _init_lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__14);
switch(lean_obj_tag(v_ruleName_139_))
{
case 0:
{
lean_object* v_n_201_; uint8_t v_phase_202_; 
v_n_201_ = lean_ctor_get(v_ruleName_139_, 0);
lean_inc_ref(v_n_201_);
lean_dec_ref_known(v_ruleName_139_, 1);
v_phase_202_ = lean_ctor_get_uint8(v_n_201_, sizeof(void*)*1 + 9);
switch(v_phase_202_)
{
case 0:
{
lean_object* v_name_203_; uint8_t v_builder_204_; uint8_t v_scope_205_; lean_object* v___x_206_; 
v_name_203_ = lean_ctor_get(v_n_201_, 0);
lean_inc(v_name_203_);
v_builder_204_ = lean_ctor_get_uint8(v_n_201_, sizeof(void*)*1 + 8);
v_scope_205_ = lean_ctor_get_uint8(v_n_201_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_201_);
v___x_206_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__15));
v_name_183_ = v_name_203_;
v_builder_184_ = v_builder_204_;
v_scope_185_ = v_scope_205_;
v___y_186_ = v___x_200_;
v___y_187_ = v___x_206_;
goto v___jp_182_;
}
case 1:
{
lean_object* v_name_207_; uint8_t v_builder_208_; uint8_t v_scope_209_; lean_object* v___x_210_; 
v_name_207_ = lean_ctor_get(v_n_201_, 0);
lean_inc(v_name_207_);
v_builder_208_ = lean_ctor_get_uint8(v_n_201_, sizeof(void*)*1 + 8);
v_scope_209_ = lean_ctor_get_uint8(v_n_201_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_201_);
v___x_210_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__16));
v_name_183_ = v_name_207_;
v_builder_184_ = v_builder_208_;
v_scope_185_ = v_scope_209_;
v___y_186_ = v___x_200_;
v___y_187_ = v___x_210_;
goto v___jp_182_;
}
default: 
{
lean_object* v_name_211_; uint8_t v_builder_212_; uint8_t v_scope_213_; lean_object* v___x_214_; 
v_name_211_ = lean_ctor_get(v_n_201_, 0);
lean_inc(v_name_211_);
v_builder_212_ = lean_ctor_get_uint8(v_n_201_, sizeof(void*)*1 + 8);
v_scope_213_ = lean_ctor_get_uint8(v_n_201_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_201_);
v___x_214_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__17));
v_name_183_ = v_name_211_;
v_builder_184_ = v_builder_212_;
v_scope_185_ = v_scope_213_;
v___y_186_ = v___x_200_;
v___y_187_ = v___x_214_;
goto v___jp_182_;
}
}
}
case 1:
{
lean_object* v___x_215_; 
v___x_215_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__18));
v___y_149_ = v___x_200_;
v___y_150_ = v___x_215_;
goto v___jp_148_;
}
default: 
{
lean_object* v___x_216_; 
v___x_216_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__19));
v___y_149_ = v___x_200_;
v___y_150_ = v___x_216_;
goto v___jp_148_;
}
}
}
else
{
lean_dec(v_a_147_);
lean_dec(v_ruleName_139_);
return v___x_146_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_lazyStepToStep___boxed(lean_object* v_ruleName_219_, lean_object* v_lstep_220_, lean_object* v_a_221_, lean_object* v_a_222_, lean_object* v_a_223_, lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_aesop_Aesop_ExtractScript_lazyStepToStep(v_ruleName_219_, v_lstep_220_, v_a_221_, v_a_222_, v_a_223_, v_a_224_);
lean_dec(v_a_224_);
lean_dec_ref(v_a_223_);
lean_dec(v_a_222_);
lean_dec_ref(v_a_221_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0(lean_object* v_00_u03b1_227_, lean_object* v_msg_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___redArg(v_msg_228_, v___y_229_, v___y_230_, v___y_231_, v___y_232_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___boxed(lean_object* v_00_u03b1_235_, lean_object* v_msg_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0(v_00_u03b1_235_, v_msg_236_, v___y_237_, v___y_238_, v___y_239_, v___y_240_);
lean_dec(v___y_240_);
lean_dec_ref(v___y_239_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ExtractScript_lazyStepsToSteps_spec__0(lean_object* v_ruleName_243_, size_t v_sz_244_, size_t v_i_245_, lean_object* v_bs_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_){
_start:
{
uint8_t v___x_252_; 
v___x_252_ = lean_usize_dec_lt(v_i_245_, v_sz_244_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; 
lean_dec(v_ruleName_243_);
v___x_253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_253_, 0, v_bs_246_);
return v___x_253_;
}
else
{
lean_object* v_v_254_; lean_object* v___x_255_; 
v_v_254_ = lean_array_uget_borrowed(v_bs_246_, v_i_245_);
lean_inc(v_v_254_);
lean_inc(v_ruleName_243_);
v___x_255_ = lp_aesop_Aesop_ExtractScript_lazyStepToStep(v_ruleName_243_, v_v_254_, v___y_247_, v___y_248_, v___y_249_, v___y_250_);
if (lean_obj_tag(v___x_255_) == 0)
{
lean_object* v_a_256_; lean_object* v___x_257_; lean_object* v_bs_x27_258_; size_t v___x_259_; size_t v___x_260_; lean_object* v___x_261_; 
v_a_256_ = lean_ctor_get(v___x_255_, 0);
lean_inc(v_a_256_);
lean_dec_ref_known(v___x_255_, 1);
v___x_257_ = lean_unsigned_to_nat(0u);
v_bs_x27_258_ = lean_array_uset(v_bs_246_, v_i_245_, v___x_257_);
v___x_259_ = ((size_t)1ULL);
v___x_260_ = lean_usize_add(v_i_245_, v___x_259_);
v___x_261_ = lean_array_uset(v_bs_x27_258_, v_i_245_, v_a_256_);
v_i_245_ = v___x_260_;
v_bs_246_ = v___x_261_;
goto _start;
}
else
{
lean_object* v_a_263_; lean_object* v___x_265_; uint8_t v_isShared_266_; uint8_t v_isSharedCheck_270_; 
lean_dec_ref(v_bs_246_);
lean_dec(v_ruleName_243_);
v_a_263_ = lean_ctor_get(v___x_255_, 0);
v_isSharedCheck_270_ = !lean_is_exclusive(v___x_255_);
if (v_isSharedCheck_270_ == 0)
{
v___x_265_ = v___x_255_;
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
else
{
lean_inc(v_a_263_);
lean_dec(v___x_255_);
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
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ExtractScript_lazyStepsToSteps_spec__0___boxed(lean_object* v_ruleName_271_, lean_object* v_sz_272_, lean_object* v_i_273_, lean_object* v_bs_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
size_t v_sz_boxed_280_; size_t v_i_boxed_281_; lean_object* v_res_282_; 
v_sz_boxed_280_ = lean_unbox_usize(v_sz_272_);
lean_dec(v_sz_272_);
v_i_boxed_281_ = lean_unbox_usize(v_i_273_);
lean_dec(v_i_273_);
v_res_282_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ExtractScript_lazyStepsToSteps_spec__0(v_ruleName_271_, v_sz_boxed_280_, v_i_boxed_281_, v_bs_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_);
lean_dec(v___y_278_);
lean_dec_ref(v___y_277_);
lean_dec(v___y_276_);
lean_dec_ref(v___y_275_);
return v_res_282_;
}
}
static lean_object* _init_lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__1(void){
_start:
{
lean_object* v___x_284_; lean_object* v___x_285_; 
v___x_284_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__0));
v___x_285_ = l_Lean_stringToMessageData(v___x_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_lazyStepsToSteps(lean_object* v_ruleName_286_, lean_object* v_x_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_, lean_object* v_a_291_){
_start:
{
if (lean_obj_tag(v_x_287_) == 0)
{
lean_object* v___x_293_; lean_object* v___y_295_; 
v___x_293_ = lean_obj_once(&lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__1, &lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__1_once, _init_lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___closed__1);
switch(lean_obj_tag(v_ruleName_286_))
{
case 0:
{
lean_object* v_n_300_; lean_object* v_name_301_; uint8_t v_builder_302_; uint8_t v_phase_303_; uint8_t v_scope_304_; lean_object* v___y_306_; lean_object* v___y_307_; lean_object* v___y_308_; lean_object* v___y_315_; lean_object* v___y_316_; lean_object* v___y_317_; lean_object* v___y_323_; 
v_n_300_ = lean_ctor_get(v_ruleName_286_, 0);
lean_inc_ref(v_n_300_);
lean_dec_ref_known(v_ruleName_286_, 1);
v_name_301_ = lean_ctor_get(v_n_300_, 0);
lean_inc(v_name_301_);
v_builder_302_ = lean_ctor_get_uint8(v_n_300_, sizeof(void*)*1 + 8);
v_phase_303_ = lean_ctor_get_uint8(v_n_300_, sizeof(void*)*1 + 9);
v_scope_304_ = lean_ctor_get_uint8(v_n_300_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_300_);
switch(v_phase_303_)
{
case 0:
{
lean_object* v___x_334_; 
v___x_334_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__15));
v___y_323_ = v___x_334_;
goto v___jp_322_;
}
case 1:
{
lean_object* v___x_335_; 
v___x_335_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__16));
v___y_323_ = v___x_335_;
goto v___jp_322_;
}
default: 
{
lean_object* v___x_336_; 
v___x_336_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__17));
v___y_323_ = v___x_336_;
goto v___jp_322_;
}
}
v___jp_305_:
{
lean_object* v___x_309_; lean_object* v___x_310_; uint8_t v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_309_ = lean_string_append(v___y_306_, v___y_308_);
v___x_310_ = lean_string_append(v___x_309_, v___y_307_);
v___x_311_ = 1;
v___x_312_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_301_, v___x_311_);
v___x_313_ = lean_string_append(v___x_310_, v___x_312_);
lean_dec_ref(v___x_312_);
v___y_295_ = v___x_313_;
goto v___jp_294_;
}
v___jp_314_:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = lean_string_append(v___y_316_, v___y_317_);
v___x_319_ = lean_string_append(v___x_318_, v___y_315_);
if (v_scope_304_ == 0)
{
lean_object* v___x_320_; 
v___x_320_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__2));
v___y_306_ = v___x_319_;
v___y_307_ = v___y_315_;
v___y_308_ = v___x_320_;
goto v___jp_305_;
}
else
{
lean_object* v___x_321_; 
v___x_321_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__3));
v___y_306_ = v___x_319_;
v___y_307_ = v___y_315_;
v___y_308_ = v___x_321_;
goto v___jp_305_;
}
}
v___jp_322_:
{
lean_object* v___x_324_; lean_object* v___x_325_; 
v___x_324_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__4));
lean_inc_ref(v___y_323_);
v___x_325_ = lean_string_append(v___y_323_, v___x_324_);
switch(v_builder_302_)
{
case 0:
{
lean_object* v___x_326_; 
v___x_326_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__5));
v___y_315_ = v___x_324_;
v___y_316_ = v___x_325_;
v___y_317_ = v___x_326_;
goto v___jp_314_;
}
case 1:
{
lean_object* v___x_327_; 
v___x_327_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__6));
v___y_315_ = v___x_324_;
v___y_316_ = v___x_325_;
v___y_317_ = v___x_327_;
goto v___jp_314_;
}
case 2:
{
lean_object* v___x_328_; 
v___x_328_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__7));
v___y_315_ = v___x_324_;
v___y_316_ = v___x_325_;
v___y_317_ = v___x_328_;
goto v___jp_314_;
}
case 3:
{
lean_object* v___x_329_; 
v___x_329_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__8));
v___y_315_ = v___x_324_;
v___y_316_ = v___x_325_;
v___y_317_ = v___x_329_;
goto v___jp_314_;
}
case 4:
{
lean_object* v___x_330_; 
v___x_330_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__9));
v___y_315_ = v___x_324_;
v___y_316_ = v___x_325_;
v___y_317_ = v___x_330_;
goto v___jp_314_;
}
case 5:
{
lean_object* v___x_331_; 
v___x_331_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__10));
v___y_315_ = v___x_324_;
v___y_316_ = v___x_325_;
v___y_317_ = v___x_331_;
goto v___jp_314_;
}
case 6:
{
lean_object* v___x_332_; 
v___x_332_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__11));
v___y_315_ = v___x_324_;
v___y_316_ = v___x_325_;
v___y_317_ = v___x_332_;
goto v___jp_314_;
}
default: 
{
lean_object* v___x_333_; 
v___x_333_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__12));
v___y_315_ = v___x_324_;
v___y_316_ = v___x_325_;
v___y_317_ = v___x_333_;
goto v___jp_314_;
}
}
}
}
case 1:
{
lean_object* v___x_337_; 
v___x_337_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__18));
v___y_295_ = v___x_337_;
goto v___jp_294_;
}
default: 
{
lean_object* v___x_338_; 
v___x_338_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_lazyStepToStep___closed__19));
v___y_295_ = v___x_338_;
goto v___jp_294_;
}
}
v___jp_294_:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_296_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_296_, 0, v___y_295_);
v___x_297_ = l_Lean_MessageData_ofFormat(v___x_296_);
v___x_298_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_298_, 0, v___x_293_);
lean_ctor_set(v___x_298_, 1, v___x_297_);
v___x_299_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0___redArg(v___x_298_, v_a_288_, v_a_289_, v_a_290_, v_a_291_);
return v___x_299_;
}
}
else
{
lean_object* v_val_339_; size_t v_sz_340_; size_t v___x_341_; lean_object* v___x_342_; 
v_val_339_ = lean_ctor_get(v_x_287_, 0);
lean_inc(v_val_339_);
lean_dec_ref_known(v_x_287_, 1);
v_sz_340_ = lean_array_size(v_val_339_);
v___x_341_ = ((size_t)0ULL);
v___x_342_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ExtractScript_lazyStepsToSteps_spec__0(v_ruleName_286_, v_sz_340_, v___x_341_, v_val_339_, v_a_288_, v_a_289_, v_a_290_, v_a_291_);
return v___x_342_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_lazyStepsToSteps___boxed(lean_object* v_ruleName_343_, lean_object* v_x_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_, lean_object* v_a_348_, lean_object* v_a_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_aesop_Aesop_ExtractScript_lazyStepsToSteps(v_ruleName_343_, v_x_344_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
lean_dec(v_a_348_);
lean_dec_ref(v_a_347_);
lean_dec(v_a_346_);
lean_dec_ref(v_a_345_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordStep___redArg(lean_object* v_step_351_, lean_object* v_a_352_){
_start:
{
lean_object* v___x_354_; lean_object* v_script_355_; uint8_t v_proofHasMVar_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_367_; 
v___x_354_ = lean_st_ref_take(v_a_352_);
v_script_355_ = lean_ctor_get(v___x_354_, 0);
v_proofHasMVar_356_ = lean_ctor_get_uint8(v___x_354_, sizeof(void*)*1);
v_isSharedCheck_367_ = !lean_is_exclusive(v___x_354_);
if (v_isSharedCheck_367_ == 0)
{
v___x_358_ = v___x_354_;
v_isShared_359_ = v_isSharedCheck_367_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_script_355_);
lean_dec(v___x_354_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_367_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_360_; lean_object* v___x_362_; 
v___x_360_ = lean_array_push(v_script_355_, v_step_351_);
if (v_isShared_359_ == 0)
{
lean_ctor_set(v___x_358_, 0, v___x_360_);
v___x_362_ = v___x_358_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v___x_360_);
lean_ctor_set_uint8(v_reuseFailAlloc_366_, sizeof(void*)*1, v_proofHasMVar_356_);
v___x_362_ = v_reuseFailAlloc_366_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_363_ = lean_st_ref_set(v_a_352_, v___x_362_);
v___x_364_ = lean_box(0);
v___x_365_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_365_, 0, v___x_364_);
return v___x_365_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordStep___redArg___boxed(lean_object* v_step_368_, lean_object* v_a_369_, lean_object* v_a_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_aesop_Aesop_ExtractScript_recordStep___redArg(v_step_368_, v_a_369_);
lean_dec(v_a_369_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordStep(lean_object* v_step_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_, lean_object* v_a_378_, lean_object* v_a_379_, lean_object* v_a_380_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lp_aesop_Aesop_ExtractScript_recordStep___redArg(v_step_372_, v_a_373_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordStep___boxed(lean_object* v_step_383_, lean_object* v_a_384_, lean_object* v_a_385_, lean_object* v_a_386_, lean_object* v_a_387_, lean_object* v_a_388_, lean_object* v_a_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_aesop_Aesop_ExtractScript_recordStep(v_step_383_, v_a_384_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, v_a_389_, v_a_390_, v_a_391_);
lean_dec(v_a_391_);
lean_dec_ref(v_a_390_);
lean_dec(v_a_389_);
lean_dec_ref(v_a_388_);
lean_dec(v_a_387_);
lean_dec(v_a_386_);
lean_dec_ref(v_a_385_);
lean_dec(v_a_384_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(lean_object* v_ruleName_394_, lean_object* v_steps_x3f_395_, lean_object* v_a_396_, lean_object* v_a_397_, lean_object* v_a_398_, lean_object* v_a_399_, lean_object* v_a_400_){
_start:
{
lean_object* v___x_402_; 
v___x_402_ = lp_aesop_Aesop_ExtractScript_lazyStepsToSteps(v_ruleName_394_, v_steps_x3f_395_, v_a_397_, v_a_398_, v_a_399_, v_a_400_);
if (lean_obj_tag(v___x_402_) == 0)
{
lean_object* v_a_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_423_; 
v_a_403_ = lean_ctor_get(v___x_402_, 0);
v_isSharedCheck_423_ = !lean_is_exclusive(v___x_402_);
if (v_isSharedCheck_423_ == 0)
{
v___x_405_ = v___x_402_;
v_isShared_406_ = v_isSharedCheck_423_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_a_403_);
lean_dec(v___x_402_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_423_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
lean_object* v___x_407_; lean_object* v_script_408_; uint8_t v_proofHasMVar_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_422_; 
v___x_407_ = lean_st_ref_take(v_a_396_);
v_script_408_ = lean_ctor_get(v___x_407_, 0);
v_proofHasMVar_409_ = lean_ctor_get_uint8(v___x_407_, sizeof(void*)*1);
v_isSharedCheck_422_ = !lean_is_exclusive(v___x_407_);
if (v_isSharedCheck_422_ == 0)
{
v___x_411_ = v___x_407_;
v_isShared_412_ = v_isSharedCheck_422_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_script_408_);
lean_dec(v___x_407_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_422_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
lean_object* v___x_413_; lean_object* v___x_415_; 
v___x_413_ = l_Array_append___redArg(v_script_408_, v_a_403_);
lean_dec(v_a_403_);
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 0, v___x_413_);
v___x_415_ = v___x_411_;
goto v_reusejp_414_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v___x_413_);
lean_ctor_set_uint8(v_reuseFailAlloc_421_, sizeof(void*)*1, v_proofHasMVar_409_);
v___x_415_ = v_reuseFailAlloc_421_;
goto v_reusejp_414_;
}
v_reusejp_414_:
{
lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_419_; 
v___x_416_ = lean_st_ref_set(v_a_396_, v___x_415_);
v___x_417_ = lean_box(0);
if (v_isShared_406_ == 0)
{
lean_ctor_set(v___x_405_, 0, v___x_417_);
v___x_419_ = v___x_405_;
goto v_reusejp_418_;
}
else
{
lean_object* v_reuseFailAlloc_420_; 
v_reuseFailAlloc_420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_420_, 0, v___x_417_);
v___x_419_ = v_reuseFailAlloc_420_;
goto v_reusejp_418_;
}
v_reusejp_418_:
{
return v___x_419_;
}
}
}
}
}
else
{
lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_431_; 
v_a_424_ = lean_ctor_get(v___x_402_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_402_);
if (v_isSharedCheck_431_ == 0)
{
v___x_426_ = v___x_402_;
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_402_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_429_; 
if (v_isShared_427_ == 0)
{
v___x_429_ = v___x_426_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v_a_424_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg___boxed(lean_object* v_ruleName_432_, lean_object* v_steps_x3f_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_){
_start:
{
lean_object* v_res_440_; 
v_res_440_ = lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(v_ruleName_432_, v_steps_x3f_433_, v_a_434_, v_a_435_, v_a_436_, v_a_437_, v_a_438_);
lean_dec(v_a_438_);
lean_dec_ref(v_a_437_);
lean_dec(v_a_436_);
lean_dec_ref(v_a_435_);
lean_dec(v_a_434_);
return v_res_440_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordLazySteps(lean_object* v_ruleName_441_, lean_object* v_steps_x3f_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(v_ruleName_441_, v_steps_x3f_442_, v_a_443_, v_a_447_, v_a_448_, v_a_449_, v_a_450_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_recordLazySteps___boxed(lean_object* v_ruleName_453_, lean_object* v_steps_x3f_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_, lean_object* v_a_459_, lean_object* v_a_460_, lean_object* v_a_461_, lean_object* v_a_462_, lean_object* v_a_463_){
_start:
{
lean_object* v_res_464_; 
v_res_464_ = lp_aesop_Aesop_ExtractScript_recordLazySteps(v_ruleName_453_, v_steps_x3f_454_, v_a_455_, v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_, v_a_461_, v_a_462_);
lean_dec(v_a_462_);
lean_dec_ref(v_a_461_);
lean_dec(v_a_460_);
lean_dec_ref(v_a_459_);
lean_dec(v_a_458_);
lean_dec(v_a_457_);
lean_dec_ref(v_a_456_);
lean_dec(v_a_455_);
return v_res_464_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_465_ = lean_unsigned_to_nat(32u);
v___x_466_ = lean_mk_empty_array_with_capacity(v___x_465_);
v___x_467_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_467_, 0, v___x_466_);
return v___x_467_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1(void){
_start:
{
size_t v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_468_ = ((size_t)5ULL);
v___x_469_ = lean_unsigned_to_nat(0u);
v___x_470_ = lean_unsigned_to_nat(32u);
v___x_471_ = lean_mk_empty_array_with_capacity(v___x_470_);
v___x_472_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__0);
v___x_473_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_473_, 0, v___x_472_);
lean_ctor_set(v___x_473_, 1, v___x_471_);
lean_ctor_set(v___x_473_, 2, v___x_469_);
lean_ctor_set(v___x_473_, 3, v___x_469_);
lean_ctor_set_usize(v___x_473_, 4, v___x_468_);
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg(lean_object* v___y_474_){
_start:
{
lean_object* v___x_476_; lean_object* v_traceState_477_; lean_object* v_traces_478_; lean_object* v___x_479_; lean_object* v_traceState_480_; lean_object* v_env_481_; lean_object* v_nextMacroScope_482_; lean_object* v_ngen_483_; lean_object* v_auxDeclNGen_484_; lean_object* v_cache_485_; lean_object* v_messages_486_; lean_object* v_infoState_487_; lean_object* v_snapshotTasks_488_; lean_object* v___x_490_; uint8_t v_isShared_491_; uint8_t v_isSharedCheck_507_; 
v___x_476_ = lean_st_ref_get(v___y_474_);
v_traceState_477_ = lean_ctor_get(v___x_476_, 4);
lean_inc_ref(v_traceState_477_);
lean_dec(v___x_476_);
v_traces_478_ = lean_ctor_get(v_traceState_477_, 0);
lean_inc_ref(v_traces_478_);
lean_dec_ref(v_traceState_477_);
v___x_479_ = lean_st_ref_take(v___y_474_);
v_traceState_480_ = lean_ctor_get(v___x_479_, 4);
v_env_481_ = lean_ctor_get(v___x_479_, 0);
v_nextMacroScope_482_ = lean_ctor_get(v___x_479_, 1);
v_ngen_483_ = lean_ctor_get(v___x_479_, 2);
v_auxDeclNGen_484_ = lean_ctor_get(v___x_479_, 3);
v_cache_485_ = lean_ctor_get(v___x_479_, 5);
v_messages_486_ = lean_ctor_get(v___x_479_, 6);
v_infoState_487_ = lean_ctor_get(v___x_479_, 7);
v_snapshotTasks_488_ = lean_ctor_get(v___x_479_, 8);
v_isSharedCheck_507_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_507_ == 0)
{
v___x_490_ = v___x_479_;
v_isShared_491_ = v_isSharedCheck_507_;
goto v_resetjp_489_;
}
else
{
lean_inc(v_snapshotTasks_488_);
lean_inc(v_infoState_487_);
lean_inc(v_messages_486_);
lean_inc(v_cache_485_);
lean_inc(v_traceState_480_);
lean_inc(v_auxDeclNGen_484_);
lean_inc(v_ngen_483_);
lean_inc(v_nextMacroScope_482_);
lean_inc(v_env_481_);
lean_dec(v___x_479_);
v___x_490_ = lean_box(0);
v_isShared_491_ = v_isSharedCheck_507_;
goto v_resetjp_489_;
}
v_resetjp_489_:
{
uint64_t v_tid_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_505_; 
v_tid_492_ = lean_ctor_get_uint64(v_traceState_480_, sizeof(void*)*1);
v_isSharedCheck_505_ = !lean_is_exclusive(v_traceState_480_);
if (v_isSharedCheck_505_ == 0)
{
lean_object* v_unused_506_; 
v_unused_506_ = lean_ctor_get(v_traceState_480_, 0);
lean_dec(v_unused_506_);
v___x_494_ = v_traceState_480_;
v_isShared_495_ = v_isSharedCheck_505_;
goto v_resetjp_493_;
}
else
{
lean_dec(v_traceState_480_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_505_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_496_; lean_object* v___x_498_; 
v___x_496_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1);
if (v_isShared_495_ == 0)
{
lean_ctor_set(v___x_494_, 0, v___x_496_);
v___x_498_ = v___x_494_;
goto v_reusejp_497_;
}
else
{
lean_object* v_reuseFailAlloc_504_; 
v_reuseFailAlloc_504_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_504_, 0, v___x_496_);
lean_ctor_set_uint64(v_reuseFailAlloc_504_, sizeof(void*)*1, v_tid_492_);
v___x_498_ = v_reuseFailAlloc_504_;
goto v_reusejp_497_;
}
v_reusejp_497_:
{
lean_object* v___x_500_; 
if (v_isShared_491_ == 0)
{
lean_ctor_set(v___x_490_, 4, v___x_498_);
v___x_500_ = v___x_490_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v_env_481_);
lean_ctor_set(v_reuseFailAlloc_503_, 1, v_nextMacroScope_482_);
lean_ctor_set(v_reuseFailAlloc_503_, 2, v_ngen_483_);
lean_ctor_set(v_reuseFailAlloc_503_, 3, v_auxDeclNGen_484_);
lean_ctor_set(v_reuseFailAlloc_503_, 4, v___x_498_);
lean_ctor_set(v_reuseFailAlloc_503_, 5, v_cache_485_);
lean_ctor_set(v_reuseFailAlloc_503_, 6, v_messages_486_);
lean_ctor_set(v_reuseFailAlloc_503_, 7, v_infoState_487_);
lean_ctor_set(v_reuseFailAlloc_503_, 8, v_snapshotTasks_488_);
v___x_500_ = v_reuseFailAlloc_503_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
lean_object* v___x_501_; lean_object* v___x_502_; 
v___x_501_ = lean_st_ref_set(v___y_474_, v___x_500_);
v___x_502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_502_, 0, v_traces_478_);
return v___x_502_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___boxed(lean_object* v___y_508_, lean_object* v___y_509_){
_start:
{
lean_object* v_res_510_; 
v_res_510_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg(v___y_508_);
lean_dec(v___y_508_);
return v_res_510_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2(lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_){
_start:
{
lean_object* v___x_520_; 
v___x_520_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg(v___y_518_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___boxed(lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_){
_start:
{
lean_object* v_res_530_; 
v_res_530_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2(v___y_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
lean_dec(v___y_528_);
lean_dec_ref(v___y_527_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
lean_dec(v___y_524_);
lean_dec(v___y_523_);
lean_dec_ref(v___y_522_);
lean_dec(v___y_521_);
return v_res_530_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(lean_object* v_opts_531_, lean_object* v_opt_532_){
_start:
{
lean_object* v_name_533_; lean_object* v_defValue_534_; lean_object* v_map_535_; lean_object* v___x_536_; 
v_name_533_ = lean_ctor_get(v_opt_532_, 0);
v_defValue_534_ = lean_ctor_get(v_opt_532_, 1);
v_map_535_ = lean_ctor_get(v_opts_531_, 0);
v___x_536_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_535_, v_name_533_);
if (lean_obj_tag(v___x_536_) == 0)
{
uint8_t v___x_537_; 
v___x_537_ = lean_unbox(v_defValue_534_);
return v___x_537_;
}
else
{
lean_object* v_val_538_; 
v_val_538_ = lean_ctor_get(v___x_536_, 0);
lean_inc(v_val_538_);
lean_dec_ref_known(v___x_536_, 1);
if (lean_obj_tag(v_val_538_) == 1)
{
uint8_t v_v_539_; 
v_v_539_ = lean_ctor_get_uint8(v_val_538_, 0);
lean_dec_ref_known(v_val_538_, 0);
return v_v_539_;
}
else
{
uint8_t v___x_540_; 
lean_dec(v_val_538_);
v___x_540_ = lean_unbox(v_defValue_534_);
return v___x_540_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3___boxed(lean_object* v_opts_541_, lean_object* v_opt_542_){
_start:
{
uint8_t v_res_543_; lean_object* v_r_544_; 
v_res_543_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_opts_541_, v_opt_542_);
lean_dec_ref(v_opt_542_);
lean_dec_ref(v_opts_541_);
v_r_544_ = lean_box(v_res_543_);
return v_r_544_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_ExtractScript_visitGoal_spec__5(lean_object* v_s_545_){
_start:
{
lean_object* v___x_546_; lean_object* v___x_547_; uint8_t v___x_548_; 
v___x_546_ = lean_array_get_size(v_s_545_);
v___x_547_ = lean_unsigned_to_nat(0u);
v___x_548_ = lean_nat_dec_eq(v___x_546_, v___x_547_);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_ExtractScript_visitGoal_spec__5___boxed(lean_object* v_s_549_){
_start:
{
uint8_t v_res_550_; lean_object* v_r_551_; 
v_res_550_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_ExtractScript_visitGoal_spec__5(v_s_549_);
lean_dec_ref(v_s_549_);
v_r_551_ = lean_box(v_res_550_);
return v_r_551_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(lean_object* v_msg_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_){
_start:
{
lean_object* v_ref_558_; lean_object* v___x_559_; lean_object* v_a_560_; lean_object* v___x_562_; uint8_t v_isShared_563_; uint8_t v_isSharedCheck_568_; 
v_ref_558_ = lean_ctor_get(v___y_555_, 5);
v___x_559_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0(v_msg_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_);
v_a_560_ = lean_ctor_get(v___x_559_, 0);
v_isSharedCheck_568_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_568_ == 0)
{
v___x_562_ = v___x_559_;
v_isShared_563_ = v_isSharedCheck_568_;
goto v_resetjp_561_;
}
else
{
lean_inc(v_a_560_);
lean_dec(v___x_559_);
v___x_562_ = lean_box(0);
v_isShared_563_ = v_isSharedCheck_568_;
goto v_resetjp_561_;
}
v_resetjp_561_:
{
lean_object* v___x_564_; lean_object* v___x_566_; 
lean_inc(v_ref_558_);
v___x_564_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_564_, 0, v_ref_558_);
lean_ctor_set(v___x_564_, 1, v_a_560_);
if (v_isShared_563_ == 0)
{
lean_ctor_set_tag(v___x_562_, 1);
lean_ctor_set(v___x_562_, 0, v___x_564_);
v___x_566_ = v___x_562_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_567_; 
v_reuseFailAlloc_567_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_567_, 0, v___x_564_);
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
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg___boxed(lean_object* v_msg_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_, lean_object* v___y_573_, lean_object* v___y_574_){
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(v_msg_569_, v___y_570_, v___y_571_, v___y_572_, v___y_573_);
lean_dec(v___y_573_);
lean_dec_ref(v___y_572_);
lean_dec(v___y_571_);
lean_dec_ref(v___y_570_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___redArg(lean_object* v_as_576_, size_t v_sz_577_, size_t v_i_578_, lean_object* v_b_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_){
_start:
{
uint8_t v___x_586_; 
v___x_586_ = lean_usize_dec_lt(v_i_578_, v_sz_577_);
if (v___x_586_ == 0)
{
lean_object* v___x_587_; 
v___x_587_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_587_, 0, v_b_579_);
return v___x_587_;
}
else
{
lean_object* v_a_588_; lean_object* v_fst_589_; lean_object* v_snd_590_; lean_object* v___x_591_; 
v_a_588_ = lean_array_uget_borrowed(v_as_576_, v_i_578_);
v_fst_589_ = lean_ctor_get(v_a_588_, 0);
v_snd_590_ = lean_ctor_get(v_a_588_, 1);
lean_inc(v_snd_590_);
lean_inc(v_fst_589_);
v___x_591_ = lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(v_fst_589_, v_snd_590_, v___y_580_, v___y_581_, v___y_582_, v___y_583_, v___y_584_);
if (lean_obj_tag(v___x_591_) == 0)
{
lean_object* v___x_592_; size_t v___x_593_; size_t v___x_594_; 
lean_dec_ref_known(v___x_591_, 1);
v___x_592_ = lean_box(0);
v___x_593_ = ((size_t)1ULL);
v___x_594_ = lean_usize_add(v_i_578_, v___x_593_);
v_i_578_ = v___x_594_;
v_b_579_ = v___x_592_;
goto _start;
}
else
{
return v___x_591_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___redArg___boxed(lean_object* v_as_596_, lean_object* v_sz_597_, lean_object* v_i_598_, lean_object* v_b_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_){
_start:
{
size_t v_sz_boxed_606_; size_t v_i_boxed_607_; lean_object* v_res_608_; 
v_sz_boxed_606_ = lean_unbox_usize(v_sz_597_);
lean_dec(v_sz_597_);
v_i_boxed_607_ = lean_unbox_usize(v_i_598_);
lean_dec(v_i_598_);
v_res_608_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___redArg(v_as_596_, v_sz_boxed_606_, v_i_boxed_607_, v_b_599_, v___y_600_, v___y_601_, v___y_602_, v___y_603_, v___y_604_);
lean_dec(v___y_604_);
lean_dec_ref(v___y_603_);
lean_dec(v___y_602_);
lean_dec_ref(v___y_601_);
lean_dec(v___y_600_);
lean_dec_ref(v_as_596_);
return v_res_608_;
}
}
static lean_object* _init_lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__1(void){
_start:
{
lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_610_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__0));
v___x_611_ = l_Lean_stringToMessageData(v___x_610_);
return v___x_611_;
}
}
static lean_object* _init_lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__3(void){
_start:
{
lean_object* v___x_613_; lean_object* v___x_614_; 
v___x_613_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__2));
v___x_614_ = l_Lean_stringToMessageData(v___x_613_);
return v___x_614_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__0(lean_object* v_normalizationState_615_, lean_object* v___x_616_, lean_object* v_____r_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_){
_start:
{
switch(lean_obj_tag(v_normalizationState_615_))
{
case 0:
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_627_ = lean_obj_once(&lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__1, &lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__1_once, _init_lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__1);
v___x_628_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_628_, 0, v___x_627_);
lean_ctor_set(v___x_628_, 1, v___x_616_);
v___x_629_ = lean_obj_once(&lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__3, &lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__3_once, _init_lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___closed__3);
v___x_630_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_630_, 0, v___x_628_);
lean_ctor_set(v___x_630_, 1, v___x_629_);
v___x_631_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(v___x_630_, v___y_622_, v___y_623_, v___y_624_, v___y_625_);
return v___x_631_;
}
case 1:
{
lean_object* v_script_632_; lean_object* v___x_633_; size_t v_sz_634_; size_t v___x_635_; lean_object* v___x_636_; 
lean_dec_ref(v___x_616_);
v_script_632_ = lean_ctor_get(v_normalizationState_615_, 2);
v___x_633_ = lean_box(0);
v_sz_634_ = lean_array_size(v_script_632_);
v___x_635_ = ((size_t)0ULL);
v___x_636_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___redArg(v_script_632_, v_sz_634_, v___x_635_, v___x_633_, v___y_618_, v___y_622_, v___y_623_, v___y_624_, v___y_625_);
if (lean_obj_tag(v___x_636_) == 0)
{
lean_object* v___x_638_; uint8_t v_isShared_639_; uint8_t v_isSharedCheck_643_; 
v_isSharedCheck_643_ = !lean_is_exclusive(v___x_636_);
if (v_isSharedCheck_643_ == 0)
{
lean_object* v_unused_644_; 
v_unused_644_ = lean_ctor_get(v___x_636_, 0);
lean_dec(v_unused_644_);
v___x_638_ = v___x_636_;
v_isShared_639_ = v_isSharedCheck_643_;
goto v_resetjp_637_;
}
else
{
lean_dec(v___x_636_);
v___x_638_ = lean_box(0);
v_isShared_639_ = v_isSharedCheck_643_;
goto v_resetjp_637_;
}
v_resetjp_637_:
{
lean_object* v___x_641_; 
if (v_isShared_639_ == 0)
{
lean_ctor_set(v___x_638_, 0, v___x_633_);
v___x_641_ = v___x_638_;
goto v_reusejp_640_;
}
else
{
lean_object* v_reuseFailAlloc_642_; 
v_reuseFailAlloc_642_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_642_, 0, v___x_633_);
v___x_641_ = v_reuseFailAlloc_642_;
goto v_reusejp_640_;
}
v_reusejp_640_:
{
return v___x_641_;
}
}
}
else
{
return v___x_636_;
}
}
default: 
{
lean_object* v_script_645_; lean_object* v___x_646_; size_t v_sz_647_; size_t v___x_648_; lean_object* v___x_649_; 
lean_dec_ref(v___x_616_);
v_script_645_ = lean_ctor_get(v_normalizationState_615_, 1);
v___x_646_ = lean_box(0);
v_sz_647_ = lean_array_size(v_script_645_);
v___x_648_ = ((size_t)0ULL);
v___x_649_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___redArg(v_script_645_, v_sz_647_, v___x_648_, v___x_646_, v___y_618_, v___y_622_, v___y_623_, v___y_624_, v___y_625_);
if (lean_obj_tag(v___x_649_) == 0)
{
lean_object* v___x_651_; uint8_t v_isShared_652_; uint8_t v_isSharedCheck_656_; 
v_isSharedCheck_656_ = !lean_is_exclusive(v___x_649_);
if (v_isSharedCheck_656_ == 0)
{
lean_object* v_unused_657_; 
v_unused_657_ = lean_ctor_get(v___x_649_, 0);
lean_dec(v_unused_657_);
v___x_651_ = v___x_649_;
v_isShared_652_ = v_isSharedCheck_656_;
goto v_resetjp_650_;
}
else
{
lean_dec(v___x_649_);
v___x_651_ = lean_box(0);
v_isShared_652_ = v_isSharedCheck_656_;
goto v_resetjp_650_;
}
v_resetjp_650_:
{
lean_object* v___x_654_; 
if (v_isShared_652_ == 0)
{
lean_ctor_set(v___x_651_, 0, v___x_646_);
v___x_654_ = v___x_651_;
goto v_reusejp_653_;
}
else
{
lean_object* v_reuseFailAlloc_655_; 
v_reuseFailAlloc_655_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_655_, 0, v___x_646_);
v___x_654_ = v_reuseFailAlloc_655_;
goto v_reusejp_653_;
}
v_reusejp_653_:
{
return v___x_654_;
}
}
}
else
{
return v___x_649_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___boxed(lean_object* v_normalizationState_658_, lean_object* v___x_659_, lean_object* v_____r_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_){
_start:
{
lean_object* v_res_670_; 
v_res_670_ = lp_aesop_Aesop_ExtractScript_visitGoal___lam__0(v_normalizationState_658_, v___x_659_, v_____r_660_, v___y_661_, v___y_662_, v___y_663_, v___y_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_);
lean_dec(v___y_668_);
lean_dec_ref(v___y_667_);
lean_dec(v___y_666_);
lean_dec_ref(v___y_665_);
lean_dec(v___y_664_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
lean_dec(v___y_661_);
lean_dec(v_normalizationState_658_);
return v_res_670_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__1(lean_object* v___x_671_, lean_object* v_x_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_682_, 0, v___x_671_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__1___boxed(lean_object* v___x_683_, lean_object* v_x_684_, lean_object* v___y_685_, lean_object* v___y_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_aesop_Aesop_ExtractScript_visitGoal___lam__1(v___x_683_, v_x_684_, v___y_685_, v___y_686_, v___y_687_, v___y_688_, v___y_689_, v___y_690_, v___y_691_, v___y_692_);
lean_dec(v___y_692_);
lean_dec_ref(v___y_691_);
lean_dec(v___y_690_);
lean_dec_ref(v___y_689_);
lean_dec(v___y_688_);
lean_dec(v___y_687_);
lean_dec_ref(v___y_686_);
lean_dec(v___y_685_);
lean_dec_ref(v_x_684_);
return v_res_694_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__2(uint8_t v___y_695_, lean_object* v___f_696_, uint8_t v___x_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_){
_start:
{
if (v___y_695_ == 0)
{
lean_object* v___x_707_; lean_object* v___x_708_; 
v___x_707_ = lean_box(0);
lean_inc(v___y_705_);
lean_inc_ref(v___y_704_);
lean_inc(v___y_703_);
lean_inc_ref(v___y_702_);
lean_inc(v___y_701_);
lean_inc(v___y_700_);
lean_inc_ref(v___y_699_);
lean_inc(v___y_698_);
v___x_708_ = lean_apply_10(v___f_696_, v___x_707_, v___y_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_, lean_box(0));
return v___x_708_;
}
else
{
lean_object* v___x_709_; lean_object* v_script_710_; lean_object* v___x_712_; uint8_t v_isShared_713_; uint8_t v_isSharedCheck_720_; 
v___x_709_ = lean_st_ref_take(v___y_698_);
v_script_710_ = lean_ctor_get(v___x_709_, 0);
v_isSharedCheck_720_ = !lean_is_exclusive(v___x_709_);
if (v_isSharedCheck_720_ == 0)
{
v___x_712_ = v___x_709_;
v_isShared_713_ = v_isSharedCheck_720_;
goto v_resetjp_711_;
}
else
{
lean_inc(v_script_710_);
lean_dec(v___x_709_);
v___x_712_ = lean_box(0);
v_isShared_713_ = v_isSharedCheck_720_;
goto v_resetjp_711_;
}
v_resetjp_711_:
{
lean_object* v___x_715_; 
if (v_isShared_713_ == 0)
{
v___x_715_ = v___x_712_;
goto v_reusejp_714_;
}
else
{
lean_object* v_reuseFailAlloc_719_; 
v_reuseFailAlloc_719_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_719_, 0, v_script_710_);
v___x_715_ = v_reuseFailAlloc_719_;
goto v_reusejp_714_;
}
v_reusejp_714_:
{
lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; 
lean_ctor_set_uint8(v___x_715_, sizeof(void*)*1, v___x_697_);
v___x_716_ = lean_st_ref_set(v___y_698_, v___x_715_);
v___x_717_ = lean_box(0);
lean_inc(v___y_705_);
lean_inc_ref(v___y_704_);
lean_inc(v___y_703_);
lean_inc_ref(v___y_702_);
lean_inc(v___y_701_);
lean_inc(v___y_700_);
lean_inc_ref(v___y_699_);
lean_inc(v___y_698_);
v___x_718_ = lean_apply_10(v___f_696_, v___x_717_, v___y_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_, lean_box(0));
return v___x_718_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___lam__2___boxed(lean_object* v___y_721_, lean_object* v___f_722_, lean_object* v___x_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_){
_start:
{
uint8_t v___y_36713__boxed_733_; uint8_t v___x_36715__boxed_734_; lean_object* v_res_735_; 
v___y_36713__boxed_733_ = lean_unbox(v___y_721_);
v___x_36715__boxed_734_ = lean_unbox(v___x_723_);
v_res_735_ = lp_aesop_Aesop_ExtractScript_visitGoal___lam__2(v___y_36713__boxed_733_, v___f_722_, v___x_36715__boxed_734_, v___y_724_, v___y_725_, v___y_726_, v___y_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
lean_dec(v___y_729_);
lean_dec_ref(v___y_728_);
lean_dec(v___y_727_);
lean_dec(v___y_726_);
lean_dec_ref(v___y_725_);
lean_dec(v___y_724_);
return v_res_735_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7(lean_object* v_opts_736_, lean_object* v_opt_737_){
_start:
{
lean_object* v_name_738_; lean_object* v_defValue_739_; lean_object* v_map_740_; lean_object* v___x_741_; 
v_name_738_ = lean_ctor_get(v_opt_737_, 0);
v_defValue_739_ = lean_ctor_get(v_opt_737_, 1);
v_map_740_ = lean_ctor_get(v_opts_736_, 0);
v___x_741_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_740_, v_name_738_);
if (lean_obj_tag(v___x_741_) == 0)
{
lean_inc(v_defValue_739_);
return v_defValue_739_;
}
else
{
lean_object* v_val_742_; 
v_val_742_ = lean_ctor_get(v___x_741_, 0);
lean_inc(v_val_742_);
lean_dec_ref_known(v___x_741_, 1);
if (lean_obj_tag(v_val_742_) == 3)
{
lean_object* v_v_743_; 
v_v_743_ = lean_ctor_get(v_val_742_, 0);
lean_inc(v_v_743_);
lean_dec_ref_known(v_val_742_, 1);
return v_v_743_;
}
else
{
lean_dec(v_val_742_);
lean_inc(v_defValue_739_);
return v_defValue_739_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7___boxed(lean_object* v_opts_744_, lean_object* v_opt_745_){
_start:
{
lean_object* v_res_746_; 
v_res_746_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7(v_opts_744_, v_opt_745_);
lean_dec_ref(v_opt_745_);
lean_dec_ref(v_opts_744_);
return v_res_746_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4_spec__6(size_t v_sz_747_, size_t v_i_748_, lean_object* v_bs_749_){
_start:
{
uint8_t v___x_750_; 
v___x_750_ = lean_usize_dec_lt(v_i_748_, v_sz_747_);
if (v___x_750_ == 0)
{
return v_bs_749_;
}
else
{
lean_object* v_v_751_; lean_object* v_msg_752_; lean_object* v___x_753_; lean_object* v_bs_x27_754_; size_t v___x_755_; size_t v___x_756_; lean_object* v___x_757_; 
v_v_751_ = lean_array_uget_borrowed(v_bs_749_, v_i_748_);
v_msg_752_ = lean_ctor_get(v_v_751_, 1);
lean_inc_ref(v_msg_752_);
v___x_753_ = lean_unsigned_to_nat(0u);
v_bs_x27_754_ = lean_array_uset(v_bs_749_, v_i_748_, v___x_753_);
v___x_755_ = ((size_t)1ULL);
v___x_756_ = lean_usize_add(v_i_748_, v___x_755_);
v___x_757_ = lean_array_uset(v_bs_x27_754_, v_i_748_, v_msg_752_);
v_i_748_ = v___x_756_;
v_bs_749_ = v___x_757_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4_spec__6___boxed(lean_object* v_sz_759_, lean_object* v_i_760_, lean_object* v_bs_761_){
_start:
{
size_t v_sz_boxed_762_; size_t v_i_boxed_763_; lean_object* v_res_764_; 
v_sz_boxed_762_ = lean_unbox_usize(v_sz_759_);
lean_dec(v_sz_759_);
v_i_boxed_763_ = lean_unbox_usize(v_i_760_);
lean_dec(v_i_760_);
v_res_764_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4_spec__6(v_sz_boxed_762_, v_i_boxed_763_, v_bs_761_);
return v_res_764_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___redArg(lean_object* v_oldTraces_765_, lean_object* v_data_766_, lean_object* v_ref_767_, lean_object* v_msg_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_){
_start:
{
lean_object* v_fileName_774_; lean_object* v_fileMap_775_; lean_object* v_options_776_; lean_object* v_currRecDepth_777_; lean_object* v_maxRecDepth_778_; lean_object* v_ref_779_; lean_object* v_currNamespace_780_; lean_object* v_openDecls_781_; lean_object* v_initHeartbeats_782_; lean_object* v_maxHeartbeats_783_; lean_object* v_quotContext_784_; lean_object* v_currMacroScope_785_; uint8_t v_diag_786_; lean_object* v_cancelTk_x3f_787_; uint8_t v_suppressElabErrors_788_; lean_object* v_inheritedTraceOptions_789_; lean_object* v___x_790_; lean_object* v_traceState_791_; lean_object* v_traces_792_; lean_object* v_ref_793_; lean_object* v___x_794_; lean_object* v___x_795_; size_t v_sz_796_; size_t v___x_797_; lean_object* v___x_798_; lean_object* v_msg_799_; lean_object* v___x_800_; lean_object* v_a_801_; lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_838_; 
v_fileName_774_ = lean_ctor_get(v___y_771_, 0);
v_fileMap_775_ = lean_ctor_get(v___y_771_, 1);
v_options_776_ = lean_ctor_get(v___y_771_, 2);
v_currRecDepth_777_ = lean_ctor_get(v___y_771_, 3);
v_maxRecDepth_778_ = lean_ctor_get(v___y_771_, 4);
v_ref_779_ = lean_ctor_get(v___y_771_, 5);
v_currNamespace_780_ = lean_ctor_get(v___y_771_, 6);
v_openDecls_781_ = lean_ctor_get(v___y_771_, 7);
v_initHeartbeats_782_ = lean_ctor_get(v___y_771_, 8);
v_maxHeartbeats_783_ = lean_ctor_get(v___y_771_, 9);
v_quotContext_784_ = lean_ctor_get(v___y_771_, 10);
v_currMacroScope_785_ = lean_ctor_get(v___y_771_, 11);
v_diag_786_ = lean_ctor_get_uint8(v___y_771_, sizeof(void*)*14);
v_cancelTk_x3f_787_ = lean_ctor_get(v___y_771_, 12);
v_suppressElabErrors_788_ = lean_ctor_get_uint8(v___y_771_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_789_ = lean_ctor_get(v___y_771_, 13);
v___x_790_ = lean_st_ref_get(v___y_772_);
v_traceState_791_ = lean_ctor_get(v___x_790_, 4);
lean_inc_ref(v_traceState_791_);
lean_dec(v___x_790_);
v_traces_792_ = lean_ctor_get(v_traceState_791_, 0);
lean_inc_ref(v_traces_792_);
lean_dec_ref(v_traceState_791_);
v_ref_793_ = l_Lean_replaceRef(v_ref_767_, v_ref_779_);
lean_inc_ref(v_inheritedTraceOptions_789_);
lean_inc(v_cancelTk_x3f_787_);
lean_inc(v_currMacroScope_785_);
lean_inc(v_quotContext_784_);
lean_inc(v_maxHeartbeats_783_);
lean_inc(v_initHeartbeats_782_);
lean_inc(v_openDecls_781_);
lean_inc(v_currNamespace_780_);
lean_inc(v_maxRecDepth_778_);
lean_inc(v_currRecDepth_777_);
lean_inc_ref(v_options_776_);
lean_inc_ref(v_fileMap_775_);
lean_inc_ref(v_fileName_774_);
v___x_794_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_794_, 0, v_fileName_774_);
lean_ctor_set(v___x_794_, 1, v_fileMap_775_);
lean_ctor_set(v___x_794_, 2, v_options_776_);
lean_ctor_set(v___x_794_, 3, v_currRecDepth_777_);
lean_ctor_set(v___x_794_, 4, v_maxRecDepth_778_);
lean_ctor_set(v___x_794_, 5, v_ref_793_);
lean_ctor_set(v___x_794_, 6, v_currNamespace_780_);
lean_ctor_set(v___x_794_, 7, v_openDecls_781_);
lean_ctor_set(v___x_794_, 8, v_initHeartbeats_782_);
lean_ctor_set(v___x_794_, 9, v_maxHeartbeats_783_);
lean_ctor_set(v___x_794_, 10, v_quotContext_784_);
lean_ctor_set(v___x_794_, 11, v_currMacroScope_785_);
lean_ctor_set(v___x_794_, 12, v_cancelTk_x3f_787_);
lean_ctor_set(v___x_794_, 13, v_inheritedTraceOptions_789_);
lean_ctor_set_uint8(v___x_794_, sizeof(void*)*14, v_diag_786_);
lean_ctor_set_uint8(v___x_794_, sizeof(void*)*14 + 1, v_suppressElabErrors_788_);
v___x_795_ = l_Lean_PersistentArray_toArray___redArg(v_traces_792_);
lean_dec_ref(v_traces_792_);
v_sz_796_ = lean_array_size(v___x_795_);
v___x_797_ = ((size_t)0ULL);
v___x_798_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4_spec__6(v_sz_796_, v___x_797_, v___x_795_);
v_msg_799_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_799_, 0, v_data_766_);
lean_ctor_set(v_msg_799_, 1, v_msg_768_);
lean_ctor_set(v_msg_799_, 2, v___x_798_);
v___x_800_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0(v_msg_799_, v___y_769_, v___y_770_, v___x_794_, v___y_772_);
lean_dec_ref_known(v___x_794_, 14);
v_a_801_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_838_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_838_ == 0)
{
v___x_803_ = v___x_800_;
v_isShared_804_ = v_isSharedCheck_838_;
goto v_resetjp_802_;
}
else
{
lean_inc(v_a_801_);
lean_dec(v___x_800_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_838_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v___x_805_; lean_object* v_traceState_806_; lean_object* v_env_807_; lean_object* v_nextMacroScope_808_; lean_object* v_ngen_809_; lean_object* v_auxDeclNGen_810_; lean_object* v_cache_811_; lean_object* v_messages_812_; lean_object* v_infoState_813_; lean_object* v_snapshotTasks_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_837_; 
v___x_805_ = lean_st_ref_take(v___y_772_);
v_traceState_806_ = lean_ctor_get(v___x_805_, 4);
v_env_807_ = lean_ctor_get(v___x_805_, 0);
v_nextMacroScope_808_ = lean_ctor_get(v___x_805_, 1);
v_ngen_809_ = lean_ctor_get(v___x_805_, 2);
v_auxDeclNGen_810_ = lean_ctor_get(v___x_805_, 3);
v_cache_811_ = lean_ctor_get(v___x_805_, 5);
v_messages_812_ = lean_ctor_get(v___x_805_, 6);
v_infoState_813_ = lean_ctor_get(v___x_805_, 7);
v_snapshotTasks_814_ = lean_ctor_get(v___x_805_, 8);
v_isSharedCheck_837_ = !lean_is_exclusive(v___x_805_);
if (v_isSharedCheck_837_ == 0)
{
v___x_816_ = v___x_805_;
v_isShared_817_ = v_isSharedCheck_837_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_snapshotTasks_814_);
lean_inc(v_infoState_813_);
lean_inc(v_messages_812_);
lean_inc(v_cache_811_);
lean_inc(v_traceState_806_);
lean_inc(v_auxDeclNGen_810_);
lean_inc(v_ngen_809_);
lean_inc(v_nextMacroScope_808_);
lean_inc(v_env_807_);
lean_dec(v___x_805_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_837_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
uint64_t v_tid_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_835_; 
v_tid_818_ = lean_ctor_get_uint64(v_traceState_806_, sizeof(void*)*1);
v_isSharedCheck_835_ = !lean_is_exclusive(v_traceState_806_);
if (v_isSharedCheck_835_ == 0)
{
lean_object* v_unused_836_; 
v_unused_836_ = lean_ctor_get(v_traceState_806_, 0);
lean_dec(v_unused_836_);
v___x_820_ = v_traceState_806_;
v_isShared_821_ = v_isSharedCheck_835_;
goto v_resetjp_819_;
}
else
{
lean_dec(v_traceState_806_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_835_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_825_; 
v___x_822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_822_, 0, v_ref_767_);
lean_ctor_set(v___x_822_, 1, v_a_801_);
v___x_823_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_765_, v___x_822_);
if (v_isShared_821_ == 0)
{
lean_ctor_set(v___x_820_, 0, v___x_823_);
v___x_825_ = v___x_820_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_823_);
lean_ctor_set_uint64(v_reuseFailAlloc_834_, sizeof(void*)*1, v_tid_818_);
v___x_825_ = v_reuseFailAlloc_834_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
lean_object* v___x_827_; 
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 4, v___x_825_);
v___x_827_ = v___x_816_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v_env_807_);
lean_ctor_set(v_reuseFailAlloc_833_, 1, v_nextMacroScope_808_);
lean_ctor_set(v_reuseFailAlloc_833_, 2, v_ngen_809_);
lean_ctor_set(v_reuseFailAlloc_833_, 3, v_auxDeclNGen_810_);
lean_ctor_set(v_reuseFailAlloc_833_, 4, v___x_825_);
lean_ctor_set(v_reuseFailAlloc_833_, 5, v_cache_811_);
lean_ctor_set(v_reuseFailAlloc_833_, 6, v_messages_812_);
lean_ctor_set(v_reuseFailAlloc_833_, 7, v_infoState_813_);
lean_ctor_set(v_reuseFailAlloc_833_, 8, v_snapshotTasks_814_);
v___x_827_ = v_reuseFailAlloc_833_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_831_; 
v___x_828_ = lean_st_ref_set(v___y_772_, v___x_827_);
v___x_829_ = lean_box(0);
if (v_isShared_804_ == 0)
{
lean_ctor_set(v___x_803_, 0, v___x_829_);
v___x_831_ = v___x_803_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_832_; 
v_reuseFailAlloc_832_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_832_, 0, v___x_829_);
v___x_831_ = v_reuseFailAlloc_832_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
return v___x_831_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___redArg___boxed(lean_object* v_oldTraces_839_, lean_object* v_data_840_, lean_object* v_ref_841_, lean_object* v_msg_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_){
_start:
{
lean_object* v_res_848_; 
v_res_848_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___redArg(v_oldTraces_839_, v_data_840_, v_ref_841_, v_msg_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
lean_dec(v___y_846_);
lean_dec_ref(v___y_845_);
lean_dec(v___y_844_);
lean_dec_ref(v___y_843_);
return v_res_848_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___redArg(lean_object* v_x_849_){
_start:
{
if (lean_obj_tag(v_x_849_) == 0)
{
lean_object* v_a_851_; lean_object* v___x_853_; uint8_t v_isShared_854_; uint8_t v_isSharedCheck_858_; 
v_a_851_ = lean_ctor_get(v_x_849_, 0);
v_isSharedCheck_858_ = !lean_is_exclusive(v_x_849_);
if (v_isSharedCheck_858_ == 0)
{
v___x_853_ = v_x_849_;
v_isShared_854_ = v_isSharedCheck_858_;
goto v_resetjp_852_;
}
else
{
lean_inc(v_a_851_);
lean_dec(v_x_849_);
v___x_853_ = lean_box(0);
v_isShared_854_ = v_isSharedCheck_858_;
goto v_resetjp_852_;
}
v_resetjp_852_:
{
lean_object* v___x_856_; 
if (v_isShared_854_ == 0)
{
lean_ctor_set_tag(v___x_853_, 1);
v___x_856_ = v___x_853_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_857_; 
v_reuseFailAlloc_857_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_857_, 0, v_a_851_);
v___x_856_ = v_reuseFailAlloc_857_;
goto v_reusejp_855_;
}
v_reusejp_855_:
{
return v___x_856_;
}
}
}
else
{
lean_object* v_a_859_; lean_object* v___x_861_; uint8_t v_isShared_862_; uint8_t v_isSharedCheck_866_; 
v_a_859_ = lean_ctor_get(v_x_849_, 0);
v_isSharedCheck_866_ = !lean_is_exclusive(v_x_849_);
if (v_isSharedCheck_866_ == 0)
{
v___x_861_ = v_x_849_;
v_isShared_862_ = v_isSharedCheck_866_;
goto v_resetjp_860_;
}
else
{
lean_inc(v_a_859_);
lean_dec(v_x_849_);
v___x_861_ = lean_box(0);
v_isShared_862_ = v_isSharedCheck_866_;
goto v_resetjp_860_;
}
v_resetjp_860_:
{
lean_object* v___x_864_; 
if (v_isShared_862_ == 0)
{
lean_ctor_set_tag(v___x_861_, 0);
v___x_864_ = v___x_861_;
goto v_reusejp_863_;
}
else
{
lean_object* v_reuseFailAlloc_865_; 
v_reuseFailAlloc_865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_865_, 0, v_a_859_);
v___x_864_ = v_reuseFailAlloc_865_;
goto v_reusejp_863_;
}
v_reusejp_863_:
{
return v___x_864_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___redArg___boxed(lean_object* v_x_867_, lean_object* v___y_868_){
_start:
{
lean_object* v_res_869_; 
v_res_869_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___redArg(v_x_867_);
return v_res_869_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__6(lean_object* v_e_870_){
_start:
{
if (lean_obj_tag(v_e_870_) == 0)
{
uint8_t v___x_871_; 
v___x_871_ = 2;
return v___x_871_;
}
else
{
uint8_t v___x_872_; 
v___x_872_ = 0;
return v___x_872_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__6___boxed(lean_object* v_e_873_){
_start:
{
uint8_t v_res_874_; lean_object* v_r_875_; 
v_res_874_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__6(v_e_873_);
lean_dec_ref(v_e_873_);
v_r_875_ = lean_box(v_res_874_);
return v_r_875_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0(void){
_start:
{
lean_object* v___x_876_; double v___x_877_; 
v___x_876_ = lean_unsigned_to_nat(0u);
v___x_877_ = lean_float_of_nat(v___x_876_);
return v___x_877_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2(void){
_start:
{
lean_object* v___x_879_; lean_object* v___x_880_; 
v___x_879_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__1));
v___x_880_ = l_Lean_stringToMessageData(v___x_879_);
return v___x_880_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3(void){
_start:
{
lean_object* v___x_881_; double v___x_882_; 
v___x_881_ = lean_unsigned_to_nat(1000u);
v___x_882_ = lean_float_of_nat(v___x_881_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4(lean_object* v_cls_883_, uint8_t v_collapsed_884_, lean_object* v_tag_885_, lean_object* v_opts_886_, uint8_t v_clsEnabled_887_, lean_object* v_oldTraces_888_, lean_object* v_msg_889_, lean_object* v_resStartStop_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_){
_start:
{
lean_object* v_fst_900_; lean_object* v_snd_901_; lean_object* v___y_903_; lean_object* v___y_904_; lean_object* v_data_905_; lean_object* v_fst_908_; lean_object* v_snd_909_; lean_object* v___x_910_; uint8_t v___x_911_; lean_object* v___y_913_; lean_object* v_a_914_; uint8_t v___y_929_; double v___y_960_; 
v_fst_900_ = lean_ctor_get(v_resStartStop_890_, 0);
lean_inc(v_fst_900_);
v_snd_901_ = lean_ctor_get(v_resStartStop_890_, 1);
lean_inc(v_snd_901_);
lean_dec_ref(v_resStartStop_890_);
v_fst_908_ = lean_ctor_get(v_snd_901_, 0);
lean_inc(v_fst_908_);
v_snd_909_ = lean_ctor_get(v_snd_901_, 1);
lean_inc(v_snd_909_);
lean_dec(v_snd_901_);
v___x_910_ = l_Lean_trace_profiler;
v___x_911_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_opts_886_, v___x_910_);
if (v___x_911_ == 0)
{
v___y_929_ = v___x_911_;
goto v___jp_928_;
}
else
{
lean_object* v___x_965_; uint8_t v___x_966_; 
v___x_965_ = l_Lean_trace_profiler_useHeartbeats;
v___x_966_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_opts_886_, v___x_965_);
if (v___x_966_ == 0)
{
lean_object* v___x_967_; lean_object* v___x_968_; double v___x_969_; double v___x_970_; double v___x_971_; 
v___x_967_ = l_Lean_trace_profiler_threshold;
v___x_968_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7(v_opts_886_, v___x_967_);
v___x_969_ = lean_float_of_nat(v___x_968_);
v___x_970_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3);
v___x_971_ = lean_float_div(v___x_969_, v___x_970_);
v___y_960_ = v___x_971_;
goto v___jp_959_;
}
else
{
lean_object* v___x_972_; lean_object* v___x_973_; double v___x_974_; 
v___x_972_ = l_Lean_trace_profiler_threshold;
v___x_973_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7(v_opts_886_, v___x_972_);
v___x_974_ = lean_float_of_nat(v___x_973_);
v___y_960_ = v___x_974_;
goto v___jp_959_;
}
}
v___jp_902_:
{
lean_object* v___x_906_; 
lean_inc(v___y_904_);
v___x_906_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___redArg(v_oldTraces_888_, v_data_905_, v___y_904_, v___y_903_, v___y_895_, v___y_896_, v___y_897_, v___y_898_);
if (lean_obj_tag(v___x_906_) == 0)
{
lean_object* v___x_907_; 
lean_dec_ref_known(v___x_906_, 1);
v___x_907_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___redArg(v_fst_900_);
return v___x_907_;
}
else
{
lean_dec(v_fst_900_);
return v___x_906_;
}
}
v___jp_912_:
{
uint8_t v_result_915_; lean_object* v___x_916_; lean_object* v___x_917_; double v___x_918_; lean_object* v_data_919_; 
v_result_915_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__6(v_fst_900_);
v___x_916_ = lean_box(v_result_915_);
v___x_917_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_917_, 0, v___x_916_);
v___x_918_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0);
lean_inc_ref(v_tag_885_);
lean_inc_ref(v___x_917_);
lean_inc(v_cls_883_);
v_data_919_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_919_, 0, v_cls_883_);
lean_ctor_set(v_data_919_, 1, v___x_917_);
lean_ctor_set(v_data_919_, 2, v_tag_885_);
lean_ctor_set_float(v_data_919_, sizeof(void*)*3, v___x_918_);
lean_ctor_set_float(v_data_919_, sizeof(void*)*3 + 8, v___x_918_);
lean_ctor_set_uint8(v_data_919_, sizeof(void*)*3 + 16, v_collapsed_884_);
if (v___x_911_ == 0)
{
lean_dec_ref_known(v___x_917_, 1);
lean_dec(v_snd_909_);
lean_dec(v_fst_908_);
lean_dec_ref(v_tag_885_);
lean_dec(v_cls_883_);
v___y_903_ = v_a_914_;
v___y_904_ = v___y_913_;
v_data_905_ = v_data_919_;
goto v___jp_902_;
}
else
{
lean_object* v_data_920_; double v___x_921_; double v___x_922_; 
lean_dec_ref_known(v_data_919_, 3);
v_data_920_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_920_, 0, v_cls_883_);
lean_ctor_set(v_data_920_, 1, v___x_917_);
lean_ctor_set(v_data_920_, 2, v_tag_885_);
v___x_921_ = lean_unbox_float(v_fst_908_);
lean_dec(v_fst_908_);
lean_ctor_set_float(v_data_920_, sizeof(void*)*3, v___x_921_);
v___x_922_ = lean_unbox_float(v_snd_909_);
lean_dec(v_snd_909_);
lean_ctor_set_float(v_data_920_, sizeof(void*)*3 + 8, v___x_922_);
lean_ctor_set_uint8(v_data_920_, sizeof(void*)*3 + 16, v_collapsed_884_);
v___y_903_ = v_a_914_;
v___y_904_ = v___y_913_;
v_data_905_ = v_data_920_;
goto v___jp_902_;
}
}
v___jp_923_:
{
lean_object* v_ref_924_; lean_object* v___x_925_; 
v_ref_924_ = lean_ctor_get(v___y_897_, 5);
lean_inc(v___y_898_);
lean_inc_ref(v___y_897_);
lean_inc(v___y_896_);
lean_inc_ref(v___y_895_);
lean_inc(v___y_894_);
lean_inc(v___y_893_);
lean_inc_ref(v___y_892_);
lean_inc(v___y_891_);
lean_inc(v_fst_900_);
v___x_925_ = lean_apply_10(v_msg_889_, v_fst_900_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_, v___y_898_, lean_box(0));
if (lean_obj_tag(v___x_925_) == 0)
{
lean_object* v_a_926_; 
v_a_926_ = lean_ctor_get(v___x_925_, 0);
lean_inc(v_a_926_);
lean_dec_ref_known(v___x_925_, 1);
v___y_913_ = v_ref_924_;
v_a_914_ = v_a_926_;
goto v___jp_912_;
}
else
{
lean_object* v___x_927_; 
lean_dec_ref_known(v___x_925_, 1);
v___x_927_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2);
v___y_913_ = v_ref_924_;
v_a_914_ = v___x_927_;
goto v___jp_912_;
}
}
v___jp_928_:
{
if (v_clsEnabled_887_ == 0)
{
if (v___y_929_ == 0)
{
lean_object* v___x_930_; lean_object* v_traceState_931_; lean_object* v_env_932_; lean_object* v_nextMacroScope_933_; lean_object* v_ngen_934_; lean_object* v_auxDeclNGen_935_; lean_object* v_cache_936_; lean_object* v_messages_937_; lean_object* v_infoState_938_; lean_object* v_snapshotTasks_939_; lean_object* v___x_941_; uint8_t v_isShared_942_; uint8_t v_isSharedCheck_958_; 
lean_dec(v_snd_909_);
lean_dec(v_fst_908_);
lean_dec_ref(v_msg_889_);
lean_dec_ref(v_tag_885_);
lean_dec(v_cls_883_);
v___x_930_ = lean_st_ref_take(v___y_898_);
v_traceState_931_ = lean_ctor_get(v___x_930_, 4);
v_env_932_ = lean_ctor_get(v___x_930_, 0);
v_nextMacroScope_933_ = lean_ctor_get(v___x_930_, 1);
v_ngen_934_ = lean_ctor_get(v___x_930_, 2);
v_auxDeclNGen_935_ = lean_ctor_get(v___x_930_, 3);
v_cache_936_ = lean_ctor_get(v___x_930_, 5);
v_messages_937_ = lean_ctor_get(v___x_930_, 6);
v_infoState_938_ = lean_ctor_get(v___x_930_, 7);
v_snapshotTasks_939_ = lean_ctor_get(v___x_930_, 8);
v_isSharedCheck_958_ = !lean_is_exclusive(v___x_930_);
if (v_isSharedCheck_958_ == 0)
{
v___x_941_ = v___x_930_;
v_isShared_942_ = v_isSharedCheck_958_;
goto v_resetjp_940_;
}
else
{
lean_inc(v_snapshotTasks_939_);
lean_inc(v_infoState_938_);
lean_inc(v_messages_937_);
lean_inc(v_cache_936_);
lean_inc(v_traceState_931_);
lean_inc(v_auxDeclNGen_935_);
lean_inc(v_ngen_934_);
lean_inc(v_nextMacroScope_933_);
lean_inc(v_env_932_);
lean_dec(v___x_930_);
v___x_941_ = lean_box(0);
v_isShared_942_ = v_isSharedCheck_958_;
goto v_resetjp_940_;
}
v_resetjp_940_:
{
uint64_t v_tid_943_; lean_object* v_traces_944_; lean_object* v___x_946_; uint8_t v_isShared_947_; uint8_t v_isSharedCheck_957_; 
v_tid_943_ = lean_ctor_get_uint64(v_traceState_931_, sizeof(void*)*1);
v_traces_944_ = lean_ctor_get(v_traceState_931_, 0);
v_isSharedCheck_957_ = !lean_is_exclusive(v_traceState_931_);
if (v_isSharedCheck_957_ == 0)
{
v___x_946_ = v_traceState_931_;
v_isShared_947_ = v_isSharedCheck_957_;
goto v_resetjp_945_;
}
else
{
lean_inc(v_traces_944_);
lean_dec(v_traceState_931_);
v___x_946_ = lean_box(0);
v_isShared_947_ = v_isSharedCheck_957_;
goto v_resetjp_945_;
}
v_resetjp_945_:
{
lean_object* v___x_948_; lean_object* v___x_950_; 
v___x_948_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_888_, v_traces_944_);
lean_dec_ref(v_traces_944_);
if (v_isShared_947_ == 0)
{
lean_ctor_set(v___x_946_, 0, v___x_948_);
v___x_950_ = v___x_946_;
goto v_reusejp_949_;
}
else
{
lean_object* v_reuseFailAlloc_956_; 
v_reuseFailAlloc_956_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_956_, 0, v___x_948_);
lean_ctor_set_uint64(v_reuseFailAlloc_956_, sizeof(void*)*1, v_tid_943_);
v___x_950_ = v_reuseFailAlloc_956_;
goto v_reusejp_949_;
}
v_reusejp_949_:
{
lean_object* v___x_952_; 
if (v_isShared_942_ == 0)
{
lean_ctor_set(v___x_941_, 4, v___x_950_);
v___x_952_ = v___x_941_;
goto v_reusejp_951_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v_env_932_);
lean_ctor_set(v_reuseFailAlloc_955_, 1, v_nextMacroScope_933_);
lean_ctor_set(v_reuseFailAlloc_955_, 2, v_ngen_934_);
lean_ctor_set(v_reuseFailAlloc_955_, 3, v_auxDeclNGen_935_);
lean_ctor_set(v_reuseFailAlloc_955_, 4, v___x_950_);
lean_ctor_set(v_reuseFailAlloc_955_, 5, v_cache_936_);
lean_ctor_set(v_reuseFailAlloc_955_, 6, v_messages_937_);
lean_ctor_set(v_reuseFailAlloc_955_, 7, v_infoState_938_);
lean_ctor_set(v_reuseFailAlloc_955_, 8, v_snapshotTasks_939_);
v___x_952_ = v_reuseFailAlloc_955_;
goto v_reusejp_951_;
}
v_reusejp_951_:
{
lean_object* v___x_953_; lean_object* v___x_954_; 
v___x_953_ = lean_st_ref_set(v___y_898_, v___x_952_);
v___x_954_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___redArg(v_fst_900_);
return v___x_954_;
}
}
}
}
}
else
{
goto v___jp_923_;
}
}
else
{
goto v___jp_923_;
}
}
v___jp_959_:
{
double v___x_961_; double v___x_962_; double v___x_963_; uint8_t v___x_964_; 
v___x_961_ = lean_unbox_float(v_snd_909_);
v___x_962_ = lean_unbox_float(v_fst_908_);
v___x_963_ = lean_float_sub(v___x_961_, v___x_962_);
v___x_964_ = lean_float_decLt(v___y_960_, v___x_963_);
v___y_929_ = v___x_964_;
goto v___jp_928_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___boxed(lean_object** _args){
lean_object* v_cls_975_ = _args[0];
lean_object* v_collapsed_976_ = _args[1];
lean_object* v_tag_977_ = _args[2];
lean_object* v_opts_978_ = _args[3];
lean_object* v_clsEnabled_979_ = _args[4];
lean_object* v_oldTraces_980_ = _args[5];
lean_object* v_msg_981_ = _args[6];
lean_object* v_resStartStop_982_ = _args[7];
lean_object* v___y_983_ = _args[8];
lean_object* v___y_984_ = _args[9];
lean_object* v___y_985_ = _args[10];
lean_object* v___y_986_ = _args[11];
lean_object* v___y_987_ = _args[12];
lean_object* v___y_988_ = _args[13];
lean_object* v___y_989_ = _args[14];
lean_object* v___y_990_ = _args[15];
lean_object* v___y_991_ = _args[16];
_start:
{
uint8_t v_collapsed_boxed_992_; uint8_t v_clsEnabled_boxed_993_; lean_object* v_res_994_; 
v_collapsed_boxed_992_ = lean_unbox(v_collapsed_976_);
v_clsEnabled_boxed_993_ = lean_unbox(v_clsEnabled_979_);
v_res_994_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4(v_cls_975_, v_collapsed_boxed_992_, v_tag_977_, v_opts_978_, v_clsEnabled_boxed_993_, v_oldTraces_980_, v_msg_981_, v_resStartStop_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_, v___y_990_);
lean_dec(v___y_990_);
lean_dec_ref(v___y_989_);
lean_dec(v___y_988_);
lean_dec_ref(v___y_987_);
lean_dec(v___y_986_);
lean_dec(v___y_985_);
lean_dec_ref(v___y_984_);
lean_dec(v___y_983_);
lean_dec_ref(v_opts_978_);
return v_res_994_;
}
}
static lean_object* _init_lp_aesop_Aesop_ExtractScript_visitGoal___closed__1(void){
_start:
{
lean_object* v___x_996_; lean_object* v___x_997_; 
v___x_996_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__0));
v___x_997_ = l_Lean_stringToMessageData(v___x_996_);
return v___x_997_;
}
}
static double _init_lp_aesop_Aesop_ExtractScript_visitGoal___closed__2(void){
_start:
{
lean_object* v___x_998_; double v___x_999_; 
v___x_998_ = lean_unsigned_to_nat(1000000000u);
v___x_999_ = lean_float_of_nat(v___x_998_);
return v___x_999_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal(lean_object* v_g_1004_, lean_object* v_a_1005_, lean_object* v_a_1006_, lean_object* v_a_1007_, lean_object* v_a_1008_, lean_object* v_a_1009_, lean_object* v_a_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_){
_start:
{
lean_object* v___x_1014_; lean_object* v_elimGoal_1015_; lean_object* v___x_1016_; lean_object* v_id_1017_; lean_object* v_normalizationState_1018_; lean_object* v_mvars_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___f_1025_; lean_object* v___x_1026_; lean_object* v___f_1027_; lean_object* v___y_1029_; uint8_t v___y_1030_; lean_object* v___y_1031_; lean_object* v___y_1032_; lean_object* v___y_1033_; uint8_t v___y_1034_; lean_object* v___y_1035_; lean_object* v_a_1036_; lean_object* v___y_1049_; uint8_t v___y_1050_; lean_object* v___y_1051_; lean_object* v___y_1052_; lean_object* v___y_1053_; uint8_t v___y_1054_; lean_object* v___y_1055_; lean_object* v_a_1056_; uint8_t v___y_1066_; lean_object* v___y_1067_; lean_object* v___y_1068_; uint8_t v___y_1069_; lean_object* v___y_1070_; lean_object* v___y_1071_; uint8_t v___y_1113_; uint8_t v___x_1130_; 
v___x_1014_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1015_ = lean_ctor_get(v___x_1014_, 1);
lean_inc_ref(v_elimGoal_1015_);
v___x_1016_ = lean_apply_1(v_elimGoal_1015_, v_g_1004_);
v_id_1017_ = lean_ctor_get(v___x_1016_, 0);
lean_inc(v_id_1017_);
v_normalizationState_1018_ = lean_ctor_get(v___x_1016_, 6);
lean_inc(v_normalizationState_1018_);
v_mvars_1019_ = lean_ctor_get(v___x_1016_, 7);
lean_inc_ref(v_mvars_1019_);
lean_dec_ref(v___x_1016_);
v___x_1020_ = lp_aesop_Aesop_TraceOption_script;
v___x_1021_ = lean_obj_once(&lp_aesop_Aesop_ExtractScript_visitGoal___closed__1, &lp_aesop_Aesop_ExtractScript_visitGoal___closed__1_once, _init_lp_aesop_Aesop_ExtractScript_visitGoal___closed__1);
v___x_1022_ = l_Nat_reprFast(v_id_1017_);
v___x_1023_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1023_, 0, v___x_1022_);
v___x_1024_ = l_Lean_MessageData_ofFormat(v___x_1023_);
lean_inc_ref(v___x_1024_);
v___f_1025_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ExtractScript_visitGoal___lam__0___boxed), 12, 2);
lean_closure_set(v___f_1025_, 0, v_normalizationState_1018_);
lean_closure_set(v___f_1025_, 1, v___x_1024_);
v___x_1026_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1026_, 0, v___x_1021_);
lean_ctor_set(v___x_1026_, 1, v___x_1024_);
v___f_1027_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ExtractScript_visitGoal___lam__1___boxed), 11, 1);
lean_closure_set(v___f_1027_, 0, v___x_1026_);
v___x_1130_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_ExtractScript_visitGoal_spec__5(v_mvars_1019_);
lean_dec_ref(v_mvars_1019_);
if (v___x_1130_ == 0)
{
uint8_t v___x_1131_; 
v___x_1131_ = 1;
v___y_1113_ = v___x_1131_;
goto v___jp_1112_;
}
else
{
uint8_t v___x_1132_; 
v___x_1132_ = 0;
v___y_1113_ = v___x_1132_;
goto v___jp_1112_;
}
v___jp_1028_:
{
lean_object* v___x_1037_; double v___x_1038_; double v___x_1039_; double v___x_1040_; double v___x_1041_; double v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; 
v___x_1037_ = lean_io_mono_nanos_now();
v___x_1038_ = lean_float_of_nat(v___y_1031_);
v___x_1039_ = lean_float_once(&lp_aesop_Aesop_ExtractScript_visitGoal___closed__2, &lp_aesop_Aesop_ExtractScript_visitGoal___closed__2_once, _init_lp_aesop_Aesop_ExtractScript_visitGoal___closed__2);
v___x_1040_ = lean_float_div(v___x_1038_, v___x_1039_);
v___x_1041_ = lean_float_of_nat(v___x_1037_);
v___x_1042_ = lean_float_div(v___x_1041_, v___x_1039_);
v___x_1043_ = lean_box_float(v___x_1040_);
v___x_1044_ = lean_box_float(v___x_1042_);
v___x_1045_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1045_, 0, v___x_1043_);
lean_ctor_set(v___x_1045_, 1, v___x_1044_);
v___x_1046_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1046_, 0, v_a_1036_);
lean_ctor_set(v___x_1046_, 1, v___x_1045_);
lean_inc_ref(v___y_1035_);
v___x_1047_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4(v___y_1032_, v___y_1034_, v___y_1035_, v___y_1033_, v___y_1030_, v___y_1029_, v___f_1027_, v___x_1046_, v_a_1005_, v_a_1006_, v_a_1007_, v_a_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_);
return v___x_1047_;
}
v___jp_1048_:
{
lean_object* v___x_1057_; double v___x_1058_; double v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; 
v___x_1057_ = lean_io_get_num_heartbeats();
v___x_1058_ = lean_float_of_nat(v___y_1052_);
v___x_1059_ = lean_float_of_nat(v___x_1057_);
v___x_1060_ = lean_box_float(v___x_1058_);
v___x_1061_ = lean_box_float(v___x_1059_);
v___x_1062_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1062_, 0, v___x_1060_);
lean_ctor_set(v___x_1062_, 1, v___x_1061_);
v___x_1063_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1063_, 0, v_a_1056_);
lean_ctor_set(v___x_1063_, 1, v___x_1062_);
lean_inc_ref(v___y_1055_);
v___x_1064_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4(v___y_1051_, v___y_1054_, v___y_1055_, v___y_1053_, v___y_1050_, v___y_1049_, v___f_1027_, v___x_1063_, v_a_1005_, v_a_1006_, v_a_1007_, v_a_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_);
return v___x_1064_;
}
v___jp_1065_:
{
lean_object* v___x_1072_; lean_object* v_a_1073_; lean_object* v___x_1074_; uint8_t v___x_1075_; 
v___x_1072_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg(v_a_1012_);
v_a_1073_ = lean_ctor_get(v___x_1072_, 0);
lean_inc(v_a_1073_);
lean_dec_ref(v___x_1072_);
v___x_1074_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1075_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v___y_1070_, v___x_1074_);
if (v___x_1075_ == 0)
{
lean_object* v___x_1076_; lean_object* v___x_1077_; 
v___x_1076_ = lean_io_mono_nanos_now();
lean_inc(v_a_1012_);
lean_inc_ref(v_a_1011_);
lean_inc(v_a_1010_);
lean_inc_ref(v_a_1009_);
lean_inc(v_a_1008_);
lean_inc(v_a_1007_);
lean_inc_ref(v_a_1006_);
lean_inc(v_a_1005_);
v___x_1077_ = lean_apply_9(v___y_1067_, v_a_1005_, v_a_1006_, v_a_1007_, v_a_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_, lean_box(0));
if (lean_obj_tag(v___x_1077_) == 0)
{
lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1085_; 
v_a_1078_ = lean_ctor_get(v___x_1077_, 0);
v_isSharedCheck_1085_ = !lean_is_exclusive(v___x_1077_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1080_ = v___x_1077_;
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_1077_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1083_; 
if (v_isShared_1081_ == 0)
{
lean_ctor_set_tag(v___x_1080_, 1);
v___x_1083_ = v___x_1080_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_a_1078_);
v___x_1083_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
v___y_1029_ = v_a_1073_;
v___y_1030_ = v___y_1066_;
v___y_1031_ = v___x_1076_;
v___y_1032_ = v___y_1068_;
v___y_1033_ = v___y_1070_;
v___y_1034_ = v___y_1069_;
v___y_1035_ = v___y_1071_;
v_a_1036_ = v___x_1083_;
goto v___jp_1028_;
}
}
}
else
{
lean_object* v_a_1086_; lean_object* v___x_1088_; uint8_t v_isShared_1089_; uint8_t v_isSharedCheck_1093_; 
v_a_1086_ = lean_ctor_get(v___x_1077_, 0);
v_isSharedCheck_1093_ = !lean_is_exclusive(v___x_1077_);
if (v_isSharedCheck_1093_ == 0)
{
v___x_1088_ = v___x_1077_;
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
else
{
lean_inc(v_a_1086_);
lean_dec(v___x_1077_);
v___x_1088_ = lean_box(0);
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
v_resetjp_1087_:
{
lean_object* v___x_1091_; 
if (v_isShared_1089_ == 0)
{
lean_ctor_set_tag(v___x_1088_, 0);
v___x_1091_ = v___x_1088_;
goto v_reusejp_1090_;
}
else
{
lean_object* v_reuseFailAlloc_1092_; 
v_reuseFailAlloc_1092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1092_, 0, v_a_1086_);
v___x_1091_ = v_reuseFailAlloc_1092_;
goto v_reusejp_1090_;
}
v_reusejp_1090_:
{
v___y_1029_ = v_a_1073_;
v___y_1030_ = v___y_1066_;
v___y_1031_ = v___x_1076_;
v___y_1032_ = v___y_1068_;
v___y_1033_ = v___y_1070_;
v___y_1034_ = v___y_1069_;
v___y_1035_ = v___y_1071_;
v_a_1036_ = v___x_1091_;
goto v___jp_1028_;
}
}
}
}
else
{
lean_object* v___x_1094_; lean_object* v___x_1095_; 
v___x_1094_ = lean_io_get_num_heartbeats();
lean_inc(v_a_1012_);
lean_inc_ref(v_a_1011_);
lean_inc(v_a_1010_);
lean_inc_ref(v_a_1009_);
lean_inc(v_a_1008_);
lean_inc(v_a_1007_);
lean_inc_ref(v_a_1006_);
lean_inc(v_a_1005_);
v___x_1095_ = lean_apply_9(v___y_1067_, v_a_1005_, v_a_1006_, v_a_1007_, v_a_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_, lean_box(0));
if (lean_obj_tag(v___x_1095_) == 0)
{
lean_object* v_a_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1103_; 
v_a_1096_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1103_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1103_ == 0)
{
v___x_1098_ = v___x_1095_;
v_isShared_1099_ = v_isSharedCheck_1103_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_a_1096_);
lean_dec(v___x_1095_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1103_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v___x_1101_; 
if (v_isShared_1099_ == 0)
{
lean_ctor_set_tag(v___x_1098_, 1);
v___x_1101_ = v___x_1098_;
goto v_reusejp_1100_;
}
else
{
lean_object* v_reuseFailAlloc_1102_; 
v_reuseFailAlloc_1102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1102_, 0, v_a_1096_);
v___x_1101_ = v_reuseFailAlloc_1102_;
goto v_reusejp_1100_;
}
v_reusejp_1100_:
{
v___y_1049_ = v_a_1073_;
v___y_1050_ = v___y_1066_;
v___y_1051_ = v___y_1068_;
v___y_1052_ = v___x_1094_;
v___y_1053_ = v___y_1070_;
v___y_1054_ = v___y_1069_;
v___y_1055_ = v___y_1071_;
v_a_1056_ = v___x_1101_;
goto v___jp_1048_;
}
}
}
else
{
lean_object* v_a_1104_; lean_object* v___x_1106_; uint8_t v_isShared_1107_; uint8_t v_isSharedCheck_1111_; 
v_a_1104_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1111_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1111_ == 0)
{
v___x_1106_ = v___x_1095_;
v_isShared_1107_ = v_isSharedCheck_1111_;
goto v_resetjp_1105_;
}
else
{
lean_inc(v_a_1104_);
lean_dec(v___x_1095_);
v___x_1106_ = lean_box(0);
v_isShared_1107_ = v_isSharedCheck_1111_;
goto v_resetjp_1105_;
}
v_resetjp_1105_:
{
lean_object* v___x_1109_; 
if (v_isShared_1107_ == 0)
{
lean_ctor_set_tag(v___x_1106_, 0);
v___x_1109_ = v___x_1106_;
goto v_reusejp_1108_;
}
else
{
lean_object* v_reuseFailAlloc_1110_; 
v_reuseFailAlloc_1110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1110_, 0, v_a_1104_);
v___x_1109_ = v_reuseFailAlloc_1110_;
goto v_reusejp_1108_;
}
v_reusejp_1108_:
{
v___y_1049_ = v_a_1073_;
v___y_1050_ = v___y_1066_;
v___y_1051_ = v___y_1068_;
v___y_1052_ = v___x_1094_;
v___y_1053_ = v___y_1070_;
v___y_1054_ = v___y_1069_;
v___y_1055_ = v___y_1071_;
v_a_1056_ = v___x_1109_;
goto v___jp_1048_;
}
}
}
}
}
v___jp_1112_:
{
lean_object* v_options_1114_; lean_object* v_inheritedTraceOptions_1115_; uint8_t v_hasTrace_1116_; uint8_t v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___y_1120_; 
v_options_1114_ = lean_ctor_get(v_a_1011_, 2);
v_inheritedTraceOptions_1115_ = lean_ctor_get(v_a_1011_, 13);
v_hasTrace_1116_ = lean_ctor_get_uint8(v_options_1114_, sizeof(void*)*1);
v___x_1117_ = 1;
v___x_1118_ = lean_box(v___y_1113_);
v___x_1119_ = lean_box(v___x_1117_);
lean_inc_ref(v___f_1025_);
v___y_1120_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ExtractScript_visitGoal___lam__2___boxed), 12, 3);
lean_closure_set(v___y_1120_, 0, v___x_1118_);
lean_closure_set(v___y_1120_, 1, v___f_1025_);
lean_closure_set(v___y_1120_, 2, v___x_1119_);
if (v_hasTrace_1116_ == 0)
{
lean_object* v___x_1121_; 
lean_dec_ref(v___y_1120_);
lean_dec_ref(v___f_1027_);
v___x_1121_ = lp_aesop_Aesop_ExtractScript_visitGoal___lam__2(v___y_1113_, v___f_1025_, v___x_1117_, v_a_1005_, v_a_1006_, v_a_1007_, v_a_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_);
return v___x_1121_;
}
else
{
lean_object* v_traceClass_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; uint8_t v___x_1126_; 
v_traceClass_1122_ = lean_ctor_get(v___x_1020_, 0);
v___x_1123_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__3));
v___x_1124_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__5));
lean_inc(v_traceClass_1122_);
v___x_1125_ = l_Lean_Name_append(v___x_1124_, v_traceClass_1122_);
v___x_1126_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1115_, v_options_1114_, v___x_1125_);
lean_dec(v___x_1125_);
if (v___x_1126_ == 0)
{
lean_object* v___x_1127_; uint8_t v___x_1128_; 
v___x_1127_ = l_Lean_trace_profiler;
v___x_1128_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_options_1114_, v___x_1127_);
if (v___x_1128_ == 0)
{
lean_object* v___x_1129_; 
lean_dec_ref(v___y_1120_);
lean_dec_ref(v___f_1027_);
v___x_1129_ = lp_aesop_Aesop_ExtractScript_visitGoal___lam__2(v___y_1113_, v___f_1025_, v___x_1117_, v_a_1005_, v_a_1006_, v_a_1007_, v_a_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_);
return v___x_1129_;
}
else
{
lean_dec_ref(v___f_1025_);
lean_inc(v_traceClass_1122_);
v___y_1066_ = v___x_1126_;
v___y_1067_ = v___y_1120_;
v___y_1068_ = v_traceClass_1122_;
v___y_1069_ = v___x_1117_;
v___y_1070_ = v_options_1114_;
v___y_1071_ = v___x_1123_;
goto v___jp_1065_;
}
}
else
{
lean_dec_ref(v___f_1025_);
lean_inc(v_traceClass_1122_);
v___y_1066_ = v___x_1126_;
v___y_1067_ = v___y_1120_;
v___y_1068_ = v_traceClass_1122_;
v___y_1069_ = v___x_1117_;
v___y_1070_ = v_options_1114_;
v___y_1071_ = v___x_1123_;
goto v___jp_1065_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitGoal___boxed(lean_object* v_g_1133_, lean_object* v_a_1134_, lean_object* v_a_1135_, lean_object* v_a_1136_, lean_object* v_a_1137_, lean_object* v_a_1138_, lean_object* v_a_1139_, lean_object* v_a_1140_, lean_object* v_a_1141_, lean_object* v_a_1142_){
_start:
{
lean_object* v_res_1143_; 
v_res_1143_ = lp_aesop_Aesop_ExtractScript_visitGoal(v_g_1133_, v_a_1134_, v_a_1135_, v_a_1136_, v_a_1137_, v_a_1138_, v_a_1139_, v_a_1140_, v_a_1141_);
lean_dec(v_a_1141_);
lean_dec_ref(v_a_1140_);
lean_dec(v_a_1139_);
lean_dec_ref(v_a_1138_);
lean_dec(v_a_1137_);
lean_dec(v_a_1136_);
lean_dec_ref(v_a_1135_);
lean_dec(v_a_1134_);
return v_res_1143_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0(lean_object* v_00_u03b1_1144_, lean_object* v_msg_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_){
_start:
{
lean_object* v___x_1155_; 
v___x_1155_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(v_msg_1145_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_);
return v___x_1155_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___boxed(lean_object* v_00_u03b1_1156_, lean_object* v_msg_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_){
_start:
{
lean_object* v_res_1167_; 
v_res_1167_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0(v_00_u03b1_1156_, v_msg_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_, v___y_1164_, v___y_1165_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
lean_dec(v___y_1161_);
lean_dec(v___y_1160_);
lean_dec_ref(v___y_1159_);
lean_dec(v___y_1158_);
return v_res_1167_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1(lean_object* v_as_1168_, size_t v_sz_1169_, size_t v_i_1170_, lean_object* v_b_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_){
_start:
{
lean_object* v___x_1181_; 
v___x_1181_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___redArg(v_as_1168_, v_sz_1169_, v_i_1170_, v_b_1171_, v___y_1172_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_);
return v___x_1181_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1___boxed(lean_object* v_as_1182_, lean_object* v_sz_1183_, lean_object* v_i_1184_, lean_object* v_b_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_){
_start:
{
size_t v_sz_boxed_1195_; size_t v_i_boxed_1196_; lean_object* v_res_1197_; 
v_sz_boxed_1195_ = lean_unbox_usize(v_sz_1183_);
lean_dec(v_sz_1183_);
v_i_boxed_1196_ = lean_unbox_usize(v_i_1184_);
lean_dec(v_i_1184_);
v_res_1197_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ExtractScript_visitGoal_spec__1(v_as_1182_, v_sz_boxed_1195_, v_i_boxed_1196_, v_b_1185_, v___y_1186_, v___y_1187_, v___y_1188_, v___y_1189_, v___y_1190_, v___y_1191_, v___y_1192_, v___y_1193_);
lean_dec(v___y_1193_);
lean_dec_ref(v___y_1192_);
lean_dec(v___y_1191_);
lean_dec_ref(v___y_1190_);
lean_dec(v___y_1189_);
lean_dec(v___y_1188_);
lean_dec_ref(v___y_1187_);
lean_dec(v___y_1186_);
lean_dec_ref(v_as_1182_);
return v_res_1197_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5(lean_object* v_00_u03b1_1198_, lean_object* v_x_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_){
_start:
{
lean_object* v___x_1209_; 
v___x_1209_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___redArg(v_x_1199_);
return v___x_1209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5___boxed(lean_object* v_00_u03b1_1210_, lean_object* v_x_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_){
_start:
{
lean_object* v_res_1221_; 
v_res_1221_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__5(v_00_u03b1_1210_, v_x_1211_, v___y_1212_, v___y_1213_, v___y_1214_, v___y_1215_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
lean_dec(v___y_1219_);
lean_dec_ref(v___y_1218_);
lean_dec(v___y_1217_);
lean_dec_ref(v___y_1216_);
lean_dec(v___y_1215_);
lean_dec(v___y_1214_);
lean_dec_ref(v___y_1213_);
lean_dec(v___y_1212_);
return v_res_1221_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4(lean_object* v_oldTraces_1222_, lean_object* v_data_1223_, lean_object* v_ref_1224_, lean_object* v_msg_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_){
_start:
{
lean_object* v___x_1235_; 
v___x_1235_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___redArg(v_oldTraces_1222_, v_data_1223_, v_ref_1224_, v_msg_1225_, v___y_1230_, v___y_1231_, v___y_1232_, v___y_1233_);
return v___x_1235_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4___boxed(lean_object* v_oldTraces_1236_, lean_object* v_data_1237_, lean_object* v_ref_1238_, lean_object* v_msg_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_){
_start:
{
lean_object* v_res_1249_; 
v_res_1249_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4(v_oldTraces_1236_, v_data_1237_, v_ref_1238_, v_msg_1239_, v___y_1240_, v___y_1241_, v___y_1242_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_);
lean_dec(v___y_1247_);
lean_dec_ref(v___y_1246_);
lean_dec(v___y_1245_);
lean_dec_ref(v___y_1244_);
lean_dec(v___y_1243_);
lean_dec(v___y_1242_);
lean_dec_ref(v___y_1241_);
lean_dec(v___y_1240_);
return v_res_1249_;
}
}
static lean_object* _init_lp_aesop_Aesop_ExtractScript_visitRapp___closed__1(void){
_start:
{
lean_object* v___x_1251_; lean_object* v___x_1252_; 
v___x_1251_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitRapp___closed__0));
v___x_1252_ = l_Lean_stringToMessageData(v___x_1251_);
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitRapp(lean_object* v_r_1253_, lean_object* v_a_1254_, lean_object* v_a_1255_, lean_object* v_a_1256_, lean_object* v_a_1257_, lean_object* v_a_1258_, lean_object* v_a_1259_, lean_object* v_a_1260_, lean_object* v_a_1261_){
_start:
{
lean_object* v___x_1263_; lean_object* v_elimRapp_1264_; lean_object* v___x_1265_; lean_object* v_options_1266_; lean_object* v_id_1267_; lean_object* v_appliedRule_1268_; lean_object* v_scriptSteps_x3f_1269_; lean_object* v_inheritedTraceOptions_1270_; uint8_t v_hasTrace_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; 
v___x_1263_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_1264_ = lean_ctor_get(v___x_1263_, 3);
lean_inc_ref(v_elimRapp_1264_);
v___x_1265_ = lean_apply_1(v_elimRapp_1264_, v_r_1253_);
v_options_1266_ = lean_ctor_get(v_a_1260_, 2);
v_id_1267_ = lean_ctor_get(v___x_1265_, 0);
lean_inc(v_id_1267_);
v_appliedRule_1268_ = lean_ctor_get(v___x_1265_, 3);
lean_inc_ref(v_appliedRule_1268_);
v_scriptSteps_x3f_1269_ = lean_ctor_get(v___x_1265_, 4);
lean_inc(v_scriptSteps_x3f_1269_);
lean_dec_ref(v___x_1265_);
v_inheritedTraceOptions_1270_ = lean_ctor_get(v_a_1260_, 13);
v_hasTrace_1271_ = lean_ctor_get_uint8(v_options_1266_, sizeof(void*)*1);
v___x_1272_ = lp_aesop_Aesop_RegularRule_name(v_appliedRule_1268_);
lean_dec_ref(v_appliedRule_1268_);
v___x_1273_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1273_, 0, v___x_1272_);
if (v_hasTrace_1271_ == 0)
{
lean_object* v___x_1274_; 
lean_dec(v_id_1267_);
v___x_1274_ = lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(v___x_1273_, v_scriptSteps_x3f_1269_, v_a_1254_, v_a_1258_, v_a_1259_, v_a_1260_, v_a_1261_);
return v___x_1274_;
}
else
{
lean_object* v___x_1275_; lean_object* v_traceClass_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___f_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; uint8_t v___x_1286_; lean_object* v___y_1288_; lean_object* v___y_1289_; lean_object* v_a_1290_; lean_object* v___y_1303_; lean_object* v___y_1304_; lean_object* v_a_1305_; 
v___x_1275_ = lp_aesop_Aesop_TraceOption_script;
v_traceClass_1276_ = lean_ctor_get(v___x_1275_, 0);
v___x_1277_ = lean_obj_once(&lp_aesop_Aesop_ExtractScript_visitRapp___closed__1, &lp_aesop_Aesop_ExtractScript_visitRapp___closed__1_once, _init_lp_aesop_Aesop_ExtractScript_visitRapp___closed__1);
v___x_1278_ = l_Nat_reprFast(v_id_1267_);
v___x_1279_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1279_, 0, v___x_1278_);
v___x_1280_ = l_Lean_MessageData_ofFormat(v___x_1279_);
v___x_1281_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1281_, 0, v___x_1277_);
lean_ctor_set(v___x_1281_, 1, v___x_1280_);
v___f_1282_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ExtractScript_visitGoal___lam__1___boxed), 11, 1);
lean_closure_set(v___f_1282_, 0, v___x_1281_);
v___x_1283_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__3));
v___x_1284_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__5));
lean_inc(v_traceClass_1276_);
v___x_1285_ = l_Lean_Name_append(v___x_1284_, v_traceClass_1276_);
v___x_1286_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1270_, v_options_1266_, v___x_1285_);
lean_dec(v___x_1285_);
if (v___x_1286_ == 0)
{
lean_object* v___x_1355_; uint8_t v___x_1356_; 
v___x_1355_ = l_Lean_trace_profiler;
v___x_1356_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_options_1266_, v___x_1355_);
if (v___x_1356_ == 0)
{
lean_object* v___x_1357_; 
lean_dec_ref(v___f_1282_);
v___x_1357_ = lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(v___x_1273_, v_scriptSteps_x3f_1269_, v_a_1254_, v_a_1258_, v_a_1259_, v_a_1260_, v_a_1261_);
return v___x_1357_;
}
else
{
goto v___jp_1314_;
}
}
else
{
goto v___jp_1314_;
}
v___jp_1287_:
{
lean_object* v___x_1291_; double v___x_1292_; double v___x_1293_; double v___x_1294_; double v___x_1295_; double v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; 
v___x_1291_ = lean_io_mono_nanos_now();
v___x_1292_ = lean_float_of_nat(v___y_1289_);
v___x_1293_ = lean_float_once(&lp_aesop_Aesop_ExtractScript_visitGoal___closed__2, &lp_aesop_Aesop_ExtractScript_visitGoal___closed__2_once, _init_lp_aesop_Aesop_ExtractScript_visitGoal___closed__2);
v___x_1294_ = lean_float_div(v___x_1292_, v___x_1293_);
v___x_1295_ = lean_float_of_nat(v___x_1291_);
v___x_1296_ = lean_float_div(v___x_1295_, v___x_1293_);
v___x_1297_ = lean_box_float(v___x_1294_);
v___x_1298_ = lean_box_float(v___x_1296_);
v___x_1299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1299_, 0, v___x_1297_);
lean_ctor_set(v___x_1299_, 1, v___x_1298_);
v___x_1300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1300_, 0, v_a_1290_);
lean_ctor_set(v___x_1300_, 1, v___x_1299_);
lean_inc(v_traceClass_1276_);
v___x_1301_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4(v_traceClass_1276_, v_hasTrace_1271_, v___x_1283_, v_options_1266_, v___x_1286_, v___y_1288_, v___f_1282_, v___x_1300_, v_a_1254_, v_a_1255_, v_a_1256_, v_a_1257_, v_a_1258_, v_a_1259_, v_a_1260_, v_a_1261_);
return v___x_1301_;
}
v___jp_1302_:
{
lean_object* v___x_1306_; double v___x_1307_; double v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; 
v___x_1306_ = lean_io_get_num_heartbeats();
v___x_1307_ = lean_float_of_nat(v___y_1304_);
v___x_1308_ = lean_float_of_nat(v___x_1306_);
v___x_1309_ = lean_box_float(v___x_1307_);
v___x_1310_ = lean_box_float(v___x_1308_);
v___x_1311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1311_, 0, v___x_1309_);
lean_ctor_set(v___x_1311_, 1, v___x_1310_);
v___x_1312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1312_, 0, v_a_1305_);
lean_ctor_set(v___x_1312_, 1, v___x_1311_);
lean_inc(v_traceClass_1276_);
v___x_1313_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4(v_traceClass_1276_, v_hasTrace_1271_, v___x_1283_, v_options_1266_, v___x_1286_, v___y_1303_, v___f_1282_, v___x_1312_, v_a_1254_, v_a_1255_, v_a_1256_, v_a_1257_, v_a_1258_, v_a_1259_, v_a_1260_, v_a_1261_);
return v___x_1313_;
}
v___jp_1314_:
{
lean_object* v___x_1315_; lean_object* v_a_1316_; lean_object* v___x_1317_; uint8_t v___x_1318_; 
v___x_1315_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg(v_a_1261_);
v_a_1316_ = lean_ctor_get(v___x_1315_, 0);
lean_inc(v_a_1316_);
lean_dec_ref(v___x_1315_);
v___x_1317_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1318_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_options_1266_, v___x_1317_);
if (v___x_1318_ == 0)
{
lean_object* v___x_1319_; lean_object* v___x_1320_; 
v___x_1319_ = lean_io_mono_nanos_now();
v___x_1320_ = lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(v___x_1273_, v_scriptSteps_x3f_1269_, v_a_1254_, v_a_1258_, v_a_1259_, v_a_1260_, v_a_1261_);
if (lean_obj_tag(v___x_1320_) == 0)
{
lean_object* v_a_1321_; lean_object* v___x_1323_; uint8_t v_isShared_1324_; uint8_t v_isSharedCheck_1328_; 
v_a_1321_ = lean_ctor_get(v___x_1320_, 0);
v_isSharedCheck_1328_ = !lean_is_exclusive(v___x_1320_);
if (v_isSharedCheck_1328_ == 0)
{
v___x_1323_ = v___x_1320_;
v_isShared_1324_ = v_isSharedCheck_1328_;
goto v_resetjp_1322_;
}
else
{
lean_inc(v_a_1321_);
lean_dec(v___x_1320_);
v___x_1323_ = lean_box(0);
v_isShared_1324_ = v_isSharedCheck_1328_;
goto v_resetjp_1322_;
}
v_resetjp_1322_:
{
lean_object* v___x_1326_; 
if (v_isShared_1324_ == 0)
{
lean_ctor_set_tag(v___x_1323_, 1);
v___x_1326_ = v___x_1323_;
goto v_reusejp_1325_;
}
else
{
lean_object* v_reuseFailAlloc_1327_; 
v_reuseFailAlloc_1327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1327_, 0, v_a_1321_);
v___x_1326_ = v_reuseFailAlloc_1327_;
goto v_reusejp_1325_;
}
v_reusejp_1325_:
{
v___y_1288_ = v_a_1316_;
v___y_1289_ = v___x_1319_;
v_a_1290_ = v___x_1326_;
goto v___jp_1287_;
}
}
}
else
{
lean_object* v_a_1329_; lean_object* v___x_1331_; uint8_t v_isShared_1332_; uint8_t v_isSharedCheck_1336_; 
v_a_1329_ = lean_ctor_get(v___x_1320_, 0);
v_isSharedCheck_1336_ = !lean_is_exclusive(v___x_1320_);
if (v_isSharedCheck_1336_ == 0)
{
v___x_1331_ = v___x_1320_;
v_isShared_1332_ = v_isSharedCheck_1336_;
goto v_resetjp_1330_;
}
else
{
lean_inc(v_a_1329_);
lean_dec(v___x_1320_);
v___x_1331_ = lean_box(0);
v_isShared_1332_ = v_isSharedCheck_1336_;
goto v_resetjp_1330_;
}
v_resetjp_1330_:
{
lean_object* v___x_1334_; 
if (v_isShared_1332_ == 0)
{
lean_ctor_set_tag(v___x_1331_, 0);
v___x_1334_ = v___x_1331_;
goto v_reusejp_1333_;
}
else
{
lean_object* v_reuseFailAlloc_1335_; 
v_reuseFailAlloc_1335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1335_, 0, v_a_1329_);
v___x_1334_ = v_reuseFailAlloc_1335_;
goto v_reusejp_1333_;
}
v_reusejp_1333_:
{
v___y_1288_ = v_a_1316_;
v___y_1289_ = v___x_1319_;
v_a_1290_ = v___x_1334_;
goto v___jp_1287_;
}
}
}
}
else
{
lean_object* v___x_1337_; lean_object* v___x_1338_; 
v___x_1337_ = lean_io_get_num_heartbeats();
v___x_1338_ = lp_aesop_Aesop_ExtractScript_recordLazySteps___redArg(v___x_1273_, v_scriptSteps_x3f_1269_, v_a_1254_, v_a_1258_, v_a_1259_, v_a_1260_, v_a_1261_);
if (lean_obj_tag(v___x_1338_) == 0)
{
lean_object* v_a_1339_; lean_object* v___x_1341_; uint8_t v_isShared_1342_; uint8_t v_isSharedCheck_1346_; 
v_a_1339_ = lean_ctor_get(v___x_1338_, 0);
v_isSharedCheck_1346_ = !lean_is_exclusive(v___x_1338_);
if (v_isSharedCheck_1346_ == 0)
{
v___x_1341_ = v___x_1338_;
v_isShared_1342_ = v_isSharedCheck_1346_;
goto v_resetjp_1340_;
}
else
{
lean_inc(v_a_1339_);
lean_dec(v___x_1338_);
v___x_1341_ = lean_box(0);
v_isShared_1342_ = v_isSharedCheck_1346_;
goto v_resetjp_1340_;
}
v_resetjp_1340_:
{
lean_object* v___x_1344_; 
if (v_isShared_1342_ == 0)
{
lean_ctor_set_tag(v___x_1341_, 1);
v___x_1344_ = v___x_1341_;
goto v_reusejp_1343_;
}
else
{
lean_object* v_reuseFailAlloc_1345_; 
v_reuseFailAlloc_1345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1345_, 0, v_a_1339_);
v___x_1344_ = v_reuseFailAlloc_1345_;
goto v_reusejp_1343_;
}
v_reusejp_1343_:
{
v___y_1303_ = v_a_1316_;
v___y_1304_ = v___x_1337_;
v_a_1305_ = v___x_1344_;
goto v___jp_1302_;
}
}
}
else
{
lean_object* v_a_1347_; lean_object* v___x_1349_; uint8_t v_isShared_1350_; uint8_t v_isSharedCheck_1354_; 
v_a_1347_ = lean_ctor_get(v___x_1338_, 0);
v_isSharedCheck_1354_ = !lean_is_exclusive(v___x_1338_);
if (v_isSharedCheck_1354_ == 0)
{
v___x_1349_ = v___x_1338_;
v_isShared_1350_ = v_isSharedCheck_1354_;
goto v_resetjp_1348_;
}
else
{
lean_inc(v_a_1347_);
lean_dec(v___x_1338_);
v___x_1349_ = lean_box(0);
v_isShared_1350_ = v_isSharedCheck_1354_;
goto v_resetjp_1348_;
}
v_resetjp_1348_:
{
lean_object* v___x_1352_; 
if (v_isShared_1350_ == 0)
{
lean_ctor_set_tag(v___x_1349_, 0);
v___x_1352_ = v___x_1349_;
goto v_reusejp_1351_;
}
else
{
lean_object* v_reuseFailAlloc_1353_; 
v_reuseFailAlloc_1353_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1353_, 0, v_a_1347_);
v___x_1352_ = v_reuseFailAlloc_1353_;
goto v_reusejp_1351_;
}
v_reusejp_1351_:
{
v___y_1303_ = v_a_1316_;
v___y_1304_ = v___x_1337_;
v_a_1305_ = v___x_1352_;
goto v___jp_1302_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ExtractScript_visitRapp___boxed(lean_object* v_r_1358_, lean_object* v_a_1359_, lean_object* v_a_1360_, lean_object* v_a_1361_, lean_object* v_a_1362_, lean_object* v_a_1363_, lean_object* v_a_1364_, lean_object* v_a_1365_, lean_object* v_a_1366_, lean_object* v_a_1367_){
_start:
{
lean_object* v_res_1368_; 
v_res_1368_ = lp_aesop_Aesop_ExtractScript_visitRapp(v_r_1358_, v_a_1359_, v_a_1360_, v_a_1361_, v_a_1362_, v_a_1363_, v_a_1364_, v_a_1365_, v_a_1366_);
lean_dec(v_a_1366_);
lean_dec_ref(v_a_1365_);
lean_dec(v_a_1364_);
lean_dec_ref(v_a_1363_);
lean_dec(v_a_1362_);
lean_dec(v_a_1361_);
lean_dec_ref(v_a_1360_);
lean_dec(v_a_1359_);
return v_res_1368_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___redArg(size_t v_sz_1369_, size_t v_i_1370_, lean_object* v_bs_1371_){
_start:
{
uint8_t v___x_1373_; 
v___x_1373_ = lean_usize_dec_lt(v_i_1370_, v_sz_1369_);
if (v___x_1373_ == 0)
{
lean_object* v___x_1374_; 
v___x_1374_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1374_, 0, v_bs_1371_);
return v___x_1374_;
}
else
{
lean_object* v_v_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v_bs_x27_1378_; size_t v___x_1379_; size_t v___x_1380_; lean_object* v___x_1381_; 
v_v_1375_ = lean_array_uget_borrowed(v_bs_1371_, v_i_1370_);
v___x_1376_ = lean_st_ref_get(v_v_1375_);
v___x_1377_ = lean_unsigned_to_nat(0u);
v_bs_x27_1378_ = lean_array_uset(v_bs_1371_, v_i_1370_, v___x_1377_);
v___x_1379_ = ((size_t)1ULL);
v___x_1380_ = lean_usize_add(v_i_1370_, v___x_1379_);
v___x_1381_ = lean_array_uset(v_bs_x27_1378_, v_i_1370_, v___x_1376_);
v_i_1370_ = v___x_1380_;
v_bs_1371_ = v___x_1381_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___redArg___boxed(lean_object* v_sz_1383_, lean_object* v_i_1384_, lean_object* v_bs_1385_, lean_object* v___y_1386_){
_start:
{
size_t v_sz_boxed_1387_; size_t v_i_boxed_1388_; lean_object* v_res_1389_; 
v_sz_boxed_1387_ = lean_unbox_usize(v_sz_1383_);
lean_dec(v_sz_1383_);
v_i_boxed_1388_ = lean_unbox_usize(v_i_1384_);
lean_dec(v_i_1384_);
v_res_1389_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___redArg(v_sz_boxed_1387_, v_i_boxed_1388_, v_bs_1385_);
return v_res_1389_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__1(size_t v_sz_1390_, size_t v_i_1391_, lean_object* v_bs_1392_){
_start:
{
uint8_t v___x_1393_; 
v___x_1393_ = lean_usize_dec_lt(v_i_1391_, v_sz_1390_);
if (v___x_1393_ == 0)
{
return v_bs_1392_;
}
else
{
lean_object* v___x_1394_; lean_object* v_elimGoal_1395_; lean_object* v_v_1396_; lean_object* v___x_1397_; lean_object* v_id_1398_; lean_object* v___x_1399_; lean_object* v_bs_x27_1400_; size_t v___x_1401_; size_t v___x_1402_; lean_object* v___x_1403_; 
v___x_1394_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1395_ = lean_ctor_get(v___x_1394_, 1);
v_v_1396_ = lean_array_uget_borrowed(v_bs_1392_, v_i_1391_);
lean_inc_ref(v_elimGoal_1395_);
lean_inc(v_v_1396_);
v___x_1397_ = lean_apply_1(v_elimGoal_1395_, v_v_1396_);
v_id_1398_ = lean_ctor_get(v___x_1397_, 0);
lean_inc(v_id_1398_);
lean_dec_ref(v___x_1397_);
v___x_1399_ = lean_unsigned_to_nat(0u);
v_bs_x27_1400_ = lean_array_uset(v_bs_1392_, v_i_1391_, v___x_1399_);
v___x_1401_ = ((size_t)1ULL);
v___x_1402_ = lean_usize_add(v_i_1391_, v___x_1401_);
v___x_1403_ = lean_array_uset(v_bs_x27_1400_, v_i_1391_, v_id_1398_);
v_i_1391_ = v___x_1402_;
v_bs_1392_ = v___x_1403_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__1___boxed(lean_object* v_sz_1405_, lean_object* v_i_1406_, lean_object* v_bs_1407_){
_start:
{
size_t v_sz_boxed_1408_; size_t v_i_boxed_1409_; lean_object* v_res_1410_; 
v_sz_boxed_1408_ = lean_unbox_usize(v_sz_1405_);
lean_dec(v_sz_1405_);
v_i_boxed_1409_ = lean_unbox_usize(v_i_1406_);
lean_dec(v_i_1406_);
v_res_1410_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__1(v_sz_boxed_1408_, v_i_boxed_1409_, v_bs_1407_);
return v_res_1410_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_MVarClusterRef_extractScriptCore_spec__2(lean_object* v_a_1411_, lean_object* v_a_1412_){
_start:
{
if (lean_obj_tag(v_a_1411_) == 0)
{
lean_object* v___x_1413_; 
v___x_1413_ = l_List_reverse___redArg(v_a_1412_);
return v___x_1413_;
}
else
{
lean_object* v_head_1414_; lean_object* v_tail_1415_; lean_object* v___x_1417_; uint8_t v_isShared_1418_; uint8_t v_isSharedCheck_1426_; 
v_head_1414_ = lean_ctor_get(v_a_1411_, 0);
v_tail_1415_ = lean_ctor_get(v_a_1411_, 1);
v_isSharedCheck_1426_ = !lean_is_exclusive(v_a_1411_);
if (v_isSharedCheck_1426_ == 0)
{
v___x_1417_ = v_a_1411_;
v_isShared_1418_ = v_isSharedCheck_1426_;
goto v_resetjp_1416_;
}
else
{
lean_inc(v_tail_1415_);
lean_inc(v_head_1414_);
lean_dec(v_a_1411_);
v___x_1417_ = lean_box(0);
v_isShared_1418_ = v_isSharedCheck_1426_;
goto v_resetjp_1416_;
}
v_resetjp_1416_:
{
lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1423_; 
v___x_1419_ = l_Nat_reprFast(v_head_1414_);
v___x_1420_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1420_, 0, v___x_1419_);
v___x_1421_ = l_Lean_MessageData_ofFormat(v___x_1420_);
if (v_isShared_1418_ == 0)
{
lean_ctor_set(v___x_1417_, 1, v_a_1412_);
lean_ctor_set(v___x_1417_, 0, v___x_1421_);
v___x_1423_ = v___x_1417_;
goto v_reusejp_1422_;
}
else
{
lean_object* v_reuseFailAlloc_1425_; 
v_reuseFailAlloc_1425_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1425_, 0, v___x_1421_);
lean_ctor_set(v_reuseFailAlloc_1425_, 1, v_a_1412_);
v___x_1423_ = v_reuseFailAlloc_1425_;
goto v_reusejp_1422_;
}
v_reusejp_1422_:
{
v_a_1411_ = v_tail_1415_;
v_a_1412_ = v___x_1423_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_extractScriptCore(lean_object* v_rref_1427_, lean_object* v_a_1428_, lean_object* v_a_1429_, lean_object* v_a_1430_, lean_object* v_a_1431_, lean_object* v_a_1432_, lean_object* v_a_1433_, lean_object* v_a_1434_, lean_object* v_a_1435_){
_start:
{
lean_object* v___x_1437_; lean_object* v___x_1438_; 
v___x_1437_ = lean_st_ref_get(v_rref_1427_);
lean_inc(v___x_1437_);
v___x_1438_ = lp_aesop_Aesop_ExtractScript_visitRapp(v___x_1437_, v_a_1428_, v_a_1429_, v_a_1430_, v_a_1431_, v_a_1432_, v_a_1433_, v_a_1434_, v_a_1435_);
if (lean_obj_tag(v___x_1438_) == 0)
{
lean_object* v___x_1440_; uint8_t v_isShared_1441_; uint8_t v_isSharedCheck_1463_; 
v_isSharedCheck_1463_ = !lean_is_exclusive(v___x_1438_);
if (v_isSharedCheck_1463_ == 0)
{
lean_object* v_unused_1464_; 
v_unused_1464_ = lean_ctor_get(v___x_1438_, 0);
lean_dec(v_unused_1464_);
v___x_1440_ = v___x_1438_;
v_isShared_1441_ = v_isSharedCheck_1463_;
goto v_resetjp_1439_;
}
else
{
lean_dec(v___x_1438_);
v___x_1440_ = lean_box(0);
v_isShared_1441_ = v_isSharedCheck_1463_;
goto v_resetjp_1439_;
}
v_resetjp_1439_:
{
lean_object* v___x_1442_; lean_object* v_elimRapp_1443_; lean_object* v___x_1444_; lean_object* v_children_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; uint8_t v___x_1449_; 
v___x_1442_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_1443_ = lean_ctor_get(v___x_1442_, 3);
lean_inc_ref(v_elimRapp_1443_);
v___x_1444_ = lean_apply_1(v_elimRapp_1443_, v___x_1437_);
v_children_1445_ = lean_ctor_get(v___x_1444_, 2);
lean_inc_ref(v_children_1445_);
lean_dec_ref(v___x_1444_);
v___x_1446_ = lean_unsigned_to_nat(0u);
v___x_1447_ = lean_array_get_size(v_children_1445_);
v___x_1448_ = lean_box(0);
v___x_1449_ = lean_nat_dec_lt(v___x_1446_, v___x_1447_);
if (v___x_1449_ == 0)
{
lean_object* v___x_1451_; 
lean_dec_ref(v_children_1445_);
if (v_isShared_1441_ == 0)
{
lean_ctor_set(v___x_1440_, 0, v___x_1448_);
v___x_1451_ = v___x_1440_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v___x_1448_);
v___x_1451_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
return v___x_1451_;
}
}
else
{
uint8_t v___x_1453_; 
v___x_1453_ = lean_nat_dec_le(v___x_1447_, v___x_1447_);
if (v___x_1453_ == 0)
{
if (v___x_1449_ == 0)
{
lean_object* v___x_1455_; 
lean_dec_ref(v_children_1445_);
if (v_isShared_1441_ == 0)
{
lean_ctor_set(v___x_1440_, 0, v___x_1448_);
v___x_1455_ = v___x_1440_;
goto v_reusejp_1454_;
}
else
{
lean_object* v_reuseFailAlloc_1456_; 
v_reuseFailAlloc_1456_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1456_, 0, v___x_1448_);
v___x_1455_ = v_reuseFailAlloc_1456_;
goto v_reusejp_1454_;
}
v_reusejp_1454_:
{
return v___x_1455_;
}
}
else
{
size_t v___x_1457_; size_t v___x_1458_; lean_object* v___x_1459_; 
lean_del_object(v___x_1440_);
v___x_1457_ = ((size_t)0ULL);
v___x_1458_ = lean_usize_of_nat(v___x_1447_);
v___x_1459_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RappRef_extractScriptCore_spec__5(v_children_1445_, v___x_1457_, v___x_1458_, v___x_1448_, v_a_1428_, v_a_1429_, v_a_1430_, v_a_1431_, v_a_1432_, v_a_1433_, v_a_1434_, v_a_1435_);
lean_dec_ref(v_children_1445_);
return v___x_1459_;
}
}
else
{
size_t v___x_1460_; size_t v___x_1461_; lean_object* v___x_1462_; 
lean_del_object(v___x_1440_);
v___x_1460_ = ((size_t)0ULL);
v___x_1461_ = lean_usize_of_nat(v___x_1447_);
v___x_1462_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RappRef_extractScriptCore_spec__5(v_children_1445_, v___x_1460_, v___x_1461_, v___x_1448_, v_a_1428_, v_a_1429_, v_a_1430_, v_a_1431_, v_a_1432_, v_a_1433_, v_a_1434_, v_a_1435_);
lean_dec_ref(v_children_1445_);
return v___x_1462_;
}
}
}
}
else
{
lean_dec(v___x_1437_);
return v___x_1438_;
}
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_extractScriptCore___closed__1(void){
_start:
{
lean_object* v___x_1466_; lean_object* v___x_1467_; 
v___x_1466_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_extractScriptCore___closed__0));
v___x_1467_ = l_Lean_stringToMessageData(v___x_1466_);
return v___x_1467_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_extractScriptCore(lean_object* v_gref_1468_, lean_object* v_a_1469_, lean_object* v_a_1470_, lean_object* v_a_1471_, lean_object* v_a_1472_, lean_object* v_a_1473_, lean_object* v_a_1474_, lean_object* v_a_1475_, lean_object* v_a_1476_){
_start:
{
lean_object* v___x_1478_; lean_object* v___x_1479_; 
v___x_1478_ = lean_st_ref_get(v_gref_1468_);
lean_inc(v___x_1478_);
v___x_1479_ = lp_aesop_Aesop_ExtractScript_visitGoal(v___x_1478_, v_a_1469_, v_a_1470_, v_a_1471_, v_a_1472_, v_a_1473_, v_a_1474_, v_a_1475_, v_a_1476_);
if (lean_obj_tag(v___x_1479_) == 0)
{
lean_object* v___x_1481_; uint8_t v_isShared_1482_; uint8_t v_isSharedCheck_1504_; 
v_isSharedCheck_1504_ = !lean_is_exclusive(v___x_1479_);
if (v_isSharedCheck_1504_ == 0)
{
lean_object* v_unused_1505_; 
v_unused_1505_ = lean_ctor_get(v___x_1479_, 0);
lean_dec(v_unused_1505_);
v___x_1481_ = v___x_1479_;
v_isShared_1482_ = v_isSharedCheck_1504_;
goto v_resetjp_1480_;
}
else
{
lean_dec(v___x_1479_);
v___x_1481_ = lean_box(0);
v_isShared_1482_ = v_isSharedCheck_1504_;
goto v_resetjp_1480_;
}
v_resetjp_1480_:
{
lean_object* v___x_1483_; lean_object* v_elimGoal_1484_; lean_object* v___x_1485_; lean_object* v_id_1486_; lean_object* v_normalizationState_1487_; uint8_t v___x_1488_; 
v___x_1483_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1484_ = lean_ctor_get(v___x_1483_, 1);
lean_inc_ref(v_elimGoal_1484_);
lean_inc(v___x_1478_);
v___x_1485_ = lean_apply_1(v_elimGoal_1484_, v___x_1478_);
v_id_1486_ = lean_ctor_get(v___x_1485_, 0);
lean_inc(v_id_1486_);
v_normalizationState_1487_ = lean_ctor_get(v___x_1485_, 6);
lean_inc(v_normalizationState_1487_);
lean_dec_ref(v___x_1485_);
v___x_1488_ = lp_aesop_Aesop_NormalizationState_isProvenByNormalization(v_normalizationState_1487_);
lean_dec(v_normalizationState_1487_);
if (v___x_1488_ == 0)
{
lean_object* v___x_1489_; 
lean_del_object(v___x_1481_);
v___x_1489_ = lp_aesop_Aesop_Goal_firstProvenRapp_x3f(v___x_1478_);
if (lean_obj_tag(v___x_1489_) == 1)
{
lean_object* v_val_1490_; lean_object* v___x_1491_; 
lean_dec(v_id_1486_);
v_val_1490_ = lean_ctor_get(v___x_1489_, 0);
lean_inc(v_val_1490_);
lean_dec_ref_known(v___x_1489_, 1);
v___x_1491_ = lp_aesop_Aesop_RappRef_extractScriptCore(v_val_1490_, v_a_1469_, v_a_1470_, v_a_1471_, v_a_1472_, v_a_1473_, v_a_1474_, v_a_1475_, v_a_1476_);
lean_dec(v_val_1490_);
return v___x_1491_;
}
else
{
lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; 
lean_dec(v___x_1489_);
v___x_1492_ = lean_obj_once(&lp_aesop_Aesop_ExtractScript_visitGoal___closed__1, &lp_aesop_Aesop_ExtractScript_visitGoal___closed__1_once, _init_lp_aesop_Aesop_ExtractScript_visitGoal___closed__1);
v___x_1493_ = l_Nat_reprFast(v_id_1486_);
v___x_1494_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1494_, 0, v___x_1493_);
v___x_1495_ = l_Lean_MessageData_ofFormat(v___x_1494_);
v___x_1496_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1496_, 0, v___x_1492_);
lean_ctor_set(v___x_1496_, 1, v___x_1495_);
v___x_1497_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_extractScriptCore___closed__1, &lp_aesop_Aesop_GoalRef_extractScriptCore___closed__1_once, _init_lp_aesop_Aesop_GoalRef_extractScriptCore___closed__1);
v___x_1498_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1496_);
lean_ctor_set(v___x_1498_, 1, v___x_1497_);
v___x_1499_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(v___x_1498_, v_a_1473_, v_a_1474_, v_a_1475_, v_a_1476_);
return v___x_1499_;
}
}
else
{
lean_object* v___x_1500_; lean_object* v___x_1502_; 
lean_dec(v_id_1486_);
lean_dec(v___x_1478_);
v___x_1500_ = lean_box(0);
if (v_isShared_1482_ == 0)
{
lean_ctor_set(v___x_1481_, 0, v___x_1500_);
v___x_1502_ = v___x_1481_;
goto v_reusejp_1501_;
}
else
{
lean_object* v_reuseFailAlloc_1503_; 
v_reuseFailAlloc_1503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1503_, 0, v___x_1500_);
v___x_1502_ = v_reuseFailAlloc_1503_;
goto v_reusejp_1501_;
}
v_reusejp_1501_:
{
return v___x_1502_;
}
}
}
}
else
{
lean_dec(v___x_1478_);
return v___x_1479_;
}
}
}
static lean_object* _init_lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__1(void){
_start:
{
lean_object* v___x_1507_; lean_object* v___x_1508_; 
v___x_1507_ = ((lean_object*)(lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__0));
v___x_1508_ = l_Lean_stringToMessageData(v___x_1507_);
return v___x_1508_;
}
}
static lean_object* _init_lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__3(void){
_start:
{
lean_object* v___x_1510_; lean_object* v___x_1511_; 
v___x_1510_ = ((lean_object*)(lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__2));
v___x_1511_ = l_Lean_stringToMessageData(v___x_1510_);
return v___x_1511_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_extractScriptCore(lean_object* v_cref_1512_, lean_object* v_a_1513_, lean_object* v_a_1514_, lean_object* v_a_1515_, lean_object* v_a_1516_, lean_object* v_a_1517_, lean_object* v_a_1518_, lean_object* v_a_1519_, lean_object* v_a_1520_){
_start:
{
lean_object* v___x_1522_; lean_object* v___x_1523_; 
v___x_1522_ = lean_st_ref_get(v_cref_1512_);
lean_inc(v___x_1522_);
v___x_1523_ = lp_aesop_Aesop_MVarCluster_provenGoal_x3f(v___x_1522_);
if (lean_obj_tag(v___x_1523_) == 1)
{
lean_object* v_val_1524_; lean_object* v___x_1525_; 
lean_dec(v___x_1522_);
v_val_1524_ = lean_ctor_get(v___x_1523_, 0);
lean_inc(v_val_1524_);
lean_dec_ref_known(v___x_1523_, 1);
v___x_1525_ = lp_aesop_Aesop_GoalRef_extractScriptCore(v_val_1524_, v_a_1513_, v_a_1514_, v_a_1515_, v_a_1516_, v_a_1517_, v_a_1518_, v_a_1519_, v_a_1520_);
lean_dec(v_val_1524_);
return v___x_1525_;
}
else
{
lean_object* v___x_1526_; lean_object* v_elimMVarCluster_1527_; lean_object* v___x_1528_; lean_object* v_goals_1529_; size_t v_sz_1530_; size_t v___x_1531_; lean_object* v___x_1532_; 
lean_dec(v___x_1523_);
v___x_1526_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_1527_ = lean_ctor_get(v___x_1526_, 5);
lean_inc_ref(v_elimMVarCluster_1527_);
v___x_1528_ = lean_apply_1(v_elimMVarCluster_1527_, v___x_1522_);
v_goals_1529_ = lean_ctor_get(v___x_1528_, 1);
lean_inc_ref(v_goals_1529_);
lean_dec_ref(v___x_1528_);
v_sz_1530_ = lean_array_size(v_goals_1529_);
v___x_1531_ = ((size_t)0ULL);
v___x_1532_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___redArg(v_sz_1530_, v___x_1531_, v_goals_1529_);
if (lean_obj_tag(v___x_1532_) == 0)
{
lean_object* v_a_1533_; lean_object* v___x_1534_; size_t v_sz_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; 
v_a_1533_ = lean_ctor_get(v___x_1532_, 0);
lean_inc(v_a_1533_);
lean_dec_ref_known(v___x_1532_, 1);
v___x_1534_ = lean_obj_once(&lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__1, &lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__1_once, _init_lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__1);
v_sz_1535_ = lean_array_size(v_a_1533_);
v___x_1536_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__1(v_sz_1535_, v___x_1531_, v_a_1533_);
v___x_1537_ = lean_array_to_list(v___x_1536_);
v___x_1538_ = lean_box(0);
v___x_1539_ = lp_aesop_List_mapTR_loop___at___00Aesop_MVarClusterRef_extractScriptCore_spec__2(v___x_1537_, v___x_1538_);
v___x_1540_ = l_Lean_MessageData_ofList(v___x_1539_);
v___x_1541_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1541_, 0, v___x_1534_);
lean_ctor_set(v___x_1541_, 1, v___x_1540_);
v___x_1542_ = lean_obj_once(&lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__3, &lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__3_once, _init_lp_aesop_Aesop_MVarClusterRef_extractScriptCore___closed__3);
v___x_1543_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1543_, 0, v___x_1541_);
lean_ctor_set(v___x_1543_, 1, v___x_1542_);
v___x_1544_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(v___x_1543_, v_a_1517_, v_a_1518_, v_a_1519_, v_a_1520_);
return v___x_1544_;
}
else
{
lean_object* v_a_1545_; lean_object* v___x_1547_; uint8_t v_isShared_1548_; uint8_t v_isSharedCheck_1552_; 
v_a_1545_ = lean_ctor_get(v___x_1532_, 0);
v_isSharedCheck_1552_ = !lean_is_exclusive(v___x_1532_);
if (v_isSharedCheck_1552_ == 0)
{
v___x_1547_ = v___x_1532_;
v_isShared_1548_ = v_isSharedCheck_1552_;
goto v_resetjp_1546_;
}
else
{
lean_inc(v_a_1545_);
lean_dec(v___x_1532_);
v___x_1547_ = lean_box(0);
v_isShared_1548_ = v_isSharedCheck_1552_;
goto v_resetjp_1546_;
}
v_resetjp_1546_:
{
lean_object* v___x_1550_; 
if (v_isShared_1548_ == 0)
{
v___x_1550_ = v___x_1547_;
goto v_reusejp_1549_;
}
else
{
lean_object* v_reuseFailAlloc_1551_; 
v_reuseFailAlloc_1551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1551_, 0, v_a_1545_);
v___x_1550_ = v_reuseFailAlloc_1551_;
goto v_reusejp_1549_;
}
v_reusejp_1549_:
{
return v___x_1550_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RappRef_extractScriptCore_spec__5(lean_object* v_as_1553_, size_t v_i_1554_, size_t v_stop_1555_, lean_object* v_b_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_){
_start:
{
uint8_t v___x_1566_; 
v___x_1566_ = lean_usize_dec_eq(v_i_1554_, v_stop_1555_);
if (v___x_1566_ == 0)
{
lean_object* v___x_1567_; lean_object* v___x_1568_; 
v___x_1567_ = lean_array_uget_borrowed(v_as_1553_, v_i_1554_);
v___x_1568_ = lp_aesop_Aesop_MVarClusterRef_extractScriptCore(v___x_1567_, v___y_1557_, v___y_1558_, v___y_1559_, v___y_1560_, v___y_1561_, v___y_1562_, v___y_1563_, v___y_1564_);
if (lean_obj_tag(v___x_1568_) == 0)
{
lean_object* v_a_1569_; size_t v___x_1570_; size_t v___x_1571_; 
v_a_1569_ = lean_ctor_get(v___x_1568_, 0);
lean_inc(v_a_1569_);
lean_dec_ref_known(v___x_1568_, 1);
v___x_1570_ = ((size_t)1ULL);
v___x_1571_ = lean_usize_add(v_i_1554_, v___x_1570_);
v_i_1554_ = v___x_1571_;
v_b_1556_ = v_a_1569_;
goto _start;
}
else
{
return v___x_1568_;
}
}
else
{
lean_object* v___x_1573_; 
v___x_1573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1573_, 0, v_b_1556_);
return v___x_1573_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RappRef_extractScriptCore_spec__5___boxed(lean_object* v_as_1574_, lean_object* v_i_1575_, lean_object* v_stop_1576_, lean_object* v_b_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_){
_start:
{
size_t v_i_boxed_1587_; size_t v_stop_boxed_1588_; lean_object* v_res_1589_; 
v_i_boxed_1587_ = lean_unbox_usize(v_i_1575_);
lean_dec(v_i_1575_);
v_stop_boxed_1588_ = lean_unbox_usize(v_stop_1576_);
lean_dec(v_stop_1576_);
v_res_1589_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RappRef_extractScriptCore_spec__5(v_as_1574_, v_i_boxed_1587_, v_stop_boxed_1588_, v_b_1577_, v___y_1578_, v___y_1579_, v___y_1580_, v___y_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_);
lean_dec(v___y_1585_);
lean_dec_ref(v___y_1584_);
lean_dec(v___y_1583_);
lean_dec_ref(v___y_1582_);
lean_dec(v___y_1581_);
lean_dec(v___y_1580_);
lean_dec_ref(v___y_1579_);
lean_dec(v___y_1578_);
lean_dec_ref(v_as_1574_);
return v_res_1589_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_extractScriptCore___boxed(lean_object* v_rref_1590_, lean_object* v_a_1591_, lean_object* v_a_1592_, lean_object* v_a_1593_, lean_object* v_a_1594_, lean_object* v_a_1595_, lean_object* v_a_1596_, lean_object* v_a_1597_, lean_object* v_a_1598_, lean_object* v_a_1599_){
_start:
{
lean_object* v_res_1600_; 
v_res_1600_ = lp_aesop_Aesop_RappRef_extractScriptCore(v_rref_1590_, v_a_1591_, v_a_1592_, v_a_1593_, v_a_1594_, v_a_1595_, v_a_1596_, v_a_1597_, v_a_1598_);
lean_dec(v_a_1598_);
lean_dec_ref(v_a_1597_);
lean_dec(v_a_1596_);
lean_dec_ref(v_a_1595_);
lean_dec(v_a_1594_);
lean_dec(v_a_1593_);
lean_dec_ref(v_a_1592_);
lean_dec(v_a_1591_);
lean_dec(v_rref_1590_);
return v_res_1600_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_extractScriptCore___boxed(lean_object* v_gref_1601_, lean_object* v_a_1602_, lean_object* v_a_1603_, lean_object* v_a_1604_, lean_object* v_a_1605_, lean_object* v_a_1606_, lean_object* v_a_1607_, lean_object* v_a_1608_, lean_object* v_a_1609_, lean_object* v_a_1610_){
_start:
{
lean_object* v_res_1611_; 
v_res_1611_ = lp_aesop_Aesop_GoalRef_extractScriptCore(v_gref_1601_, v_a_1602_, v_a_1603_, v_a_1604_, v_a_1605_, v_a_1606_, v_a_1607_, v_a_1608_, v_a_1609_);
lean_dec(v_a_1609_);
lean_dec_ref(v_a_1608_);
lean_dec(v_a_1607_);
lean_dec_ref(v_a_1606_);
lean_dec(v_a_1605_);
lean_dec(v_a_1604_);
lean_dec_ref(v_a_1603_);
lean_dec(v_a_1602_);
lean_dec(v_gref_1601_);
return v_res_1611_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_extractScriptCore___boxed(lean_object* v_cref_1612_, lean_object* v_a_1613_, lean_object* v_a_1614_, lean_object* v_a_1615_, lean_object* v_a_1616_, lean_object* v_a_1617_, lean_object* v_a_1618_, lean_object* v_a_1619_, lean_object* v_a_1620_, lean_object* v_a_1621_){
_start:
{
lean_object* v_res_1622_; 
v_res_1622_ = lp_aesop_Aesop_MVarClusterRef_extractScriptCore(v_cref_1612_, v_a_1613_, v_a_1614_, v_a_1615_, v_a_1616_, v_a_1617_, v_a_1618_, v_a_1619_, v_a_1620_);
lean_dec(v_a_1620_);
lean_dec_ref(v_a_1619_);
lean_dec(v_a_1618_);
lean_dec_ref(v_a_1617_);
lean_dec(v_a_1616_);
lean_dec(v_a_1615_);
lean_dec_ref(v_a_1614_);
lean_dec(v_a_1613_);
lean_dec(v_cref_1612_);
return v_res_1622_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0(size_t v_sz_1623_, size_t v_i_1624_, lean_object* v_bs_1625_, lean_object* v___y_1626_, lean_object* v___y_1627_, lean_object* v___y_1628_, lean_object* v___y_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_){
_start:
{
lean_object* v___x_1635_; 
v___x_1635_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___redArg(v_sz_1623_, v_i_1624_, v_bs_1625_);
return v___x_1635_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0___boxed(lean_object* v_sz_1636_, lean_object* v_i_1637_, lean_object* v_bs_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_){
_start:
{
size_t v_sz_boxed_1648_; size_t v_i_boxed_1649_; lean_object* v_res_1650_; 
v_sz_boxed_1648_ = lean_unbox_usize(v_sz_1636_);
lean_dec(v_sz_1636_);
v_i_boxed_1649_ = lean_unbox_usize(v_i_1637_);
lean_dec(v_i_1637_);
v_res_1650_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_MVarClusterRef_extractScriptCore_spec__0(v_sz_boxed_1648_, v_i_boxed_1649_, v_bs_1638_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_, v___y_1643_, v___y_1644_, v___y_1645_, v___y_1646_);
lean_dec(v___y_1646_);
lean_dec_ref(v___y_1645_);
lean_dec(v___y_1644_);
lean_dec_ref(v___y_1643_);
lean_dec(v___y_1642_);
lean_dec(v___y_1641_);
lean_dec_ref(v___y_1640_);
lean_dec(v___y_1639_);
return v_res_1650_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1652_; lean_object* v___x_1653_; 
v___x_1652_ = ((lean_object*)(lp_aesop_Aesop_extractScript___lam__0___closed__0));
v___x_1653_ = l_Lean_stringToMessageData(v___x_1652_);
return v___x_1653_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractScript___lam__0(lean_object* v_x_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_){
_start:
{
lean_object* v___x_1663_; lean_object* v___x_1664_; 
v___x_1663_ = lean_obj_once(&lp_aesop_Aesop_extractScript___lam__0___closed__1, &lp_aesop_Aesop_extractScript___lam__0___closed__1_once, _init_lp_aesop_Aesop_extractScript___lam__0___closed__1);
v___x_1664_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1664_, 0, v___x_1663_);
return v___x_1664_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractScript___lam__0___boxed(lean_object* v_x_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_){
_start:
{
lean_object* v_res_1674_; 
v_res_1674_ = lp_aesop_Aesop_extractScript___lam__0(v_x_1665_, v___y_1666_, v___y_1667_, v___y_1668_, v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_);
lean_dec(v___y_1672_);
lean_dec_ref(v___y_1671_);
lean_dec(v___y_1670_);
lean_dec_ref(v___y_1669_);
lean_dec(v___y_1668_);
lean_dec(v___y_1667_);
lean_dec_ref(v___y_1666_);
lean_dec_ref(v_x_1665_);
return v_res_1674_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__2(void){
_start:
{
lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; 
v___x_1677_ = l_Lean_Core_instMonadTraceCoreM;
v___x_1678_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__1));
v___x_1679_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1678_, v___x_1677_);
return v___x_1679_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__3(void){
_start:
{
lean_object* v___x_1680_; lean_object* v___f_1681_; lean_object* v___x_1682_; 
v___x_1680_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__2, &lp_aesop_Aesop_extractScript___closed__2_once, _init_lp_aesop_Aesop_extractScript___closed__2);
v___f_1681_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__0));
v___x_1682_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1681_, v___x_1680_);
return v___x_1682_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__4(void){
_start:
{
lean_object* v___x_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; 
v___x_1683_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__3, &lp_aesop_Aesop_extractScript___closed__3_once, _init_lp_aesop_Aesop_extractScript___closed__3);
v___x_1684_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__1));
v___x_1685_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1684_, v___x_1683_);
return v___x_1685_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__5(void){
_start:
{
lean_object* v___x_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; 
v___x_1686_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__4, &lp_aesop_Aesop_extractScript___closed__4_once, _init_lp_aesop_Aesop_extractScript___closed__4);
v___x_1687_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__1));
v___x_1688_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_1687_, v___x_1686_);
return v___x_1688_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__6(void){
_start:
{
lean_object* v___x_1689_; lean_object* v___f_1690_; lean_object* v___x_1691_; 
v___x_1689_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__5, &lp_aesop_Aesop_extractScript___closed__5_once, _init_lp_aesop_Aesop_extractScript___closed__5);
v___f_1690_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__0));
v___x_1691_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_1690_, v___x_1689_);
return v___x_1691_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__9(void){
_start:
{
lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; 
v___x_1694_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_1695_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__1));
v___x_1696_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__8));
v___x_1697_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_1696_, v___x_1695_, v___x_1694_);
return v___x_1697_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__10(void){
_start:
{
lean_object* v___x_1698_; lean_object* v___f_1699_; lean_object* v___f_1700_; lean_object* v___x_1701_; 
v___x_1698_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__9, &lp_aesop_Aesop_extractScript___closed__9_once, _init_lp_aesop_Aesop_extractScript___closed__9);
v___f_1699_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__0));
v___f_1700_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__7));
v___x_1701_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_1700_, v___f_1699_, v___x_1698_);
return v___x_1701_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__11(void){
_start:
{
lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; 
v___x_1702_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__10, &lp_aesop_Aesop_extractScript___closed__10_once, _init_lp_aesop_Aesop_extractScript___closed__10);
v___x_1703_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__1));
v___x_1704_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__8));
v___x_1705_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_1704_, v___x_1703_, v___x_1702_);
return v___x_1705_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__12(void){
_start:
{
lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; 
v___x_1706_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__11, &lp_aesop_Aesop_extractScript___closed__11_once, _init_lp_aesop_Aesop_extractScript___closed__11);
v___x_1707_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__1));
v___x_1708_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__8));
v___x_1709_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_1708_, v___x_1707_, v___x_1706_);
return v___x_1709_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__13(void){
_start:
{
lean_object* v___x_1710_; lean_object* v___f_1711_; lean_object* v___f_1712_; lean_object* v___x_1713_; 
v___x_1710_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__12, &lp_aesop_Aesop_extractScript___closed__12_once, _init_lp_aesop_Aesop_extractScript___closed__12);
v___f_1711_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__0));
v___f_1712_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__7));
v___x_1713_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_1712_, v___f_1711_, v___x_1710_);
return v___x_1713_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__14(void){
_start:
{
lean_object* v___x_1714_; 
v___x_1714_ = l_instMonadExceptOfEIO(lean_box(0));
return v___x_1714_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__15(void){
_start:
{
lean_object* v___x_1715_; lean_object* v___x_1716_; 
v___x_1715_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__14, &lp_aesop_Aesop_extractScript___closed__14_once, _init_lp_aesop_Aesop_extractScript___closed__14);
v___x_1716_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1715_);
return v___x_1716_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__16(void){
_start:
{
lean_object* v___x_1717_; lean_object* v___x_1718_; 
v___x_1717_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__15, &lp_aesop_Aesop_extractScript___closed__15_once, _init_lp_aesop_Aesop_extractScript___closed__15);
v___x_1718_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_1717_);
return v___x_1718_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__17(void){
_start:
{
lean_object* v___x_1719_; lean_object* v___x_1720_; 
v___x_1719_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__16, &lp_aesop_Aesop_extractScript___closed__16_once, _init_lp_aesop_Aesop_extractScript___closed__16);
v___x_1720_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1719_);
return v___x_1720_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__18(void){
_start:
{
lean_object* v___x_1721_; lean_object* v___x_1722_; 
v___x_1721_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__17, &lp_aesop_Aesop_extractScript___closed__17_once, _init_lp_aesop_Aesop_extractScript___closed__17);
v___x_1722_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_1721_);
return v___x_1722_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__19(void){
_start:
{
lean_object* v___x_1723_; lean_object* v___x_1724_; 
v___x_1723_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__18, &lp_aesop_Aesop_extractScript___closed__18_once, _init_lp_aesop_Aesop_extractScript___closed__18);
v___x_1724_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1723_);
return v___x_1724_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__20(void){
_start:
{
lean_object* v___x_1725_; lean_object* v___x_1726_; 
v___x_1725_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__19, &lp_aesop_Aesop_extractScript___closed__19_once, _init_lp_aesop_Aesop_extractScript___closed__19);
v___x_1726_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_1725_);
return v___x_1726_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__21(void){
_start:
{
lean_object* v___x_1727_; lean_object* v___x_1728_; 
v___x_1727_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__20, &lp_aesop_Aesop_extractScript___closed__20_once, _init_lp_aesop_Aesop_extractScript___closed__20);
v___x_1728_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_1727_);
return v___x_1728_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__23(void){
_start:
{
lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___f_1732_; 
v___x_1730_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__1));
v___x_1731_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_1732_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1732_, 0, v___x_1731_);
lean_closure_set(v___f_1732_, 1, v___x_1730_);
return v___f_1732_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__24(void){
_start:
{
lean_object* v___x_1733_; lean_object* v___f_1734_; lean_object* v___f_1735_; 
v___x_1733_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__1));
v___f_1734_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__23, &lp_aesop_Aesop_extractScript___closed__23_once, _init_lp_aesop_Aesop_extractScript___closed__23);
v___f_1735_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1735_, 0, v___f_1734_);
lean_closure_set(v___f_1735_, 1, v___x_1733_);
return v___f_1735_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractScript___closed__25(void){
_start:
{
lean_object* v___f_1736_; lean_object* v___f_1737_; lean_object* v___f_1738_; 
v___f_1736_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__0));
v___f_1737_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__24, &lp_aesop_Aesop_extractScript___closed__24_once, _init_lp_aesop_Aesop_extractScript___closed__24);
v___f_1738_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1738_, 0, v___f_1737_);
lean_closure_set(v___f_1738_, 1, v___f_1736_);
return v___f_1738_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractScript(lean_object* v_a_1740_, lean_object* v_a_1741_, lean_object* v_a_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_, lean_object* v_a_1745_, lean_object* v_a_1746_){
_start:
{
lean_object* v_____do__lift_1749_; lean_object* v___y_1750_; lean_object* v___y_1751_; lean_object* v___y_1752_; lean_object* v___y_1753_; lean_object* v___y_1754_; lean_object* v___y_1755_; lean_object* v___y_1756_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v_toMonadRef_1762_; lean_object* v___x_1763_; lean_object* v_options_1764_; uint8_t v_hasTrace_1765_; 
v___x_1759_ = lp_aesop_Aesop_TreeM_instMonad;
v___x_1760_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__6, &lp_aesop_Aesop_extractScript___closed__6_once, _init_lp_aesop_Aesop_extractScript___closed__6);
v___x_1761_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__13, &lp_aesop_Aesop_extractScript___closed__13_once, _init_lp_aesop_Aesop_extractScript___closed__13);
v_toMonadRef_1762_ = lean_ctor_get(v___x_1761_, 0);
v___x_1763_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__21, &lp_aesop_Aesop_extractScript___closed__21_once, _init_lp_aesop_Aesop_extractScript___closed__21);
v_options_1764_ = lean_ctor_get(v_a_1745_, 2);
v_hasTrace_1765_ = lean_ctor_get_uint8(v_options_1764_, sizeof(void*)*1);
if (v_hasTrace_1765_ == 0)
{
lean_object* v___x_1766_; 
v___x_1766_ = lp_aesop_Aesop_getRootGoal(v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
if (lean_obj_tag(v___x_1766_) == 0)
{
lean_object* v_a_1767_; 
v_a_1767_ = lean_ctor_get(v___x_1766_, 0);
lean_inc(v_a_1767_);
lean_dec_ref_known(v___x_1766_, 1);
v_____do__lift_1749_ = v_a_1767_;
v___y_1750_ = v_a_1740_;
v___y_1751_ = v_a_1741_;
v___y_1752_ = v_a_1742_;
v___y_1753_ = v_a_1743_;
v___y_1754_ = v_a_1744_;
v___y_1755_ = v_a_1745_;
v___y_1756_ = v_a_1746_;
goto v___jp_1748_;
}
else
{
lean_object* v_a_1768_; lean_object* v___x_1770_; uint8_t v_isShared_1771_; uint8_t v_isSharedCheck_1775_; 
v_a_1768_ = lean_ctor_get(v___x_1766_, 0);
v_isSharedCheck_1775_ = !lean_is_exclusive(v___x_1766_);
if (v_isSharedCheck_1775_ == 0)
{
v___x_1770_ = v___x_1766_;
v_isShared_1771_ = v_isSharedCheck_1775_;
goto v_resetjp_1769_;
}
else
{
lean_inc(v_a_1768_);
lean_dec(v___x_1766_);
v___x_1770_ = lean_box(0);
v_isShared_1771_ = v_isSharedCheck_1775_;
goto v_resetjp_1769_;
}
v_resetjp_1769_:
{
lean_object* v___x_1773_; 
if (v_isShared_1771_ == 0)
{
v___x_1773_ = v___x_1770_;
goto v_reusejp_1772_;
}
else
{
lean_object* v_reuseFailAlloc_1774_; 
v_reuseFailAlloc_1774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1774_, 0, v_a_1768_);
v___x_1773_ = v_reuseFailAlloc_1774_;
goto v_reusejp_1772_;
}
v_reusejp_1772_:
{
return v___x_1773_;
}
}
}
}
else
{
lean_object* v_inheritedTraceOptions_1776_; lean_object* v___x_1777_; lean_object* v_traceClass_1778_; lean_object* v___f_1779_; lean_object* v___f_1780_; lean_object* v___f_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; uint8_t v___x_1785_; lean_object* v___y_1787_; lean_object* v___y_1788_; lean_object* v_a_1789_; lean_object* v___y_1803_; lean_object* v___y_1804_; lean_object* v_a_1805_; lean_object* v___y_1808_; lean_object* v___y_1809_; lean_object* v_a_1810_; lean_object* v___y_1821_; lean_object* v___y_1822_; lean_object* v_a_1823_; 
v_inheritedTraceOptions_1776_ = lean_ctor_get(v_a_1745_, 13);
v___x_1777_ = lp_aesop_Aesop_TraceOption_script;
v_traceClass_1778_ = lean_ctor_get(v___x_1777_, 0);
v___f_1779_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__22));
v___f_1780_ = lean_obj_once(&lp_aesop_Aesop_extractScript___closed__25, &lp_aesop_Aesop_extractScript___closed__25_once, _init_lp_aesop_Aesop_extractScript___closed__25);
v___f_1781_ = ((lean_object*)(lp_aesop_Aesop_extractScript___closed__26));
v___x_1782_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__3));
v___x_1783_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__5));
lean_inc(v_traceClass_1778_);
v___x_1784_ = l_Lean_Name_append(v___x_1783_, v_traceClass_1778_);
v___x_1785_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1776_, v_options_1764_, v___x_1784_);
lean_dec(v___x_1784_);
if (v___x_1785_ == 0)
{
lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; uint8_t v___x_1874_; 
v___x_1871_ = l_Lean_KVMap_instValueBool;
v___x_1872_ = l_Lean_trace_profiler;
v___x_1873_ = l_Lean_Option_get___redArg(v___x_1871_, v_options_1764_, v___x_1872_);
v___x_1874_ = lean_unbox(v___x_1873_);
lean_dec(v___x_1873_);
if (v___x_1874_ == 0)
{
lean_object* v___x_1875_; 
v___x_1875_ = lp_aesop_Aesop_getRootGoal(v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
if (lean_obj_tag(v___x_1875_) == 0)
{
lean_object* v_a_1876_; 
v_a_1876_ = lean_ctor_get(v___x_1875_, 0);
lean_inc(v_a_1876_);
lean_dec_ref_known(v___x_1875_, 1);
v_____do__lift_1749_ = v_a_1876_;
v___y_1750_ = v_a_1740_;
v___y_1751_ = v_a_1741_;
v___y_1752_ = v_a_1742_;
v___y_1753_ = v_a_1743_;
v___y_1754_ = v_a_1744_;
v___y_1755_ = v_a_1745_;
v___y_1756_ = v_a_1746_;
goto v___jp_1748_;
}
else
{
lean_object* v_a_1877_; lean_object* v___x_1879_; uint8_t v_isShared_1880_; uint8_t v_isSharedCheck_1884_; 
v_a_1877_ = lean_ctor_get(v___x_1875_, 0);
v_isSharedCheck_1884_ = !lean_is_exclusive(v___x_1875_);
if (v_isSharedCheck_1884_ == 0)
{
v___x_1879_ = v___x_1875_;
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
else
{
lean_inc(v_a_1877_);
lean_dec(v___x_1875_);
v___x_1879_ = lean_box(0);
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
v_resetjp_1878_:
{
lean_object* v___x_1882_; 
if (v_isShared_1880_ == 0)
{
v___x_1882_ = v___x_1879_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_1883_; 
v_reuseFailAlloc_1883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1883_, 0, v_a_1877_);
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
goto v___jp_1825_;
}
}
else
{
goto v___jp_1825_;
}
v___jp_1786_:
{
lean_object* v___x_1790_; double v___x_1791_; double v___x_1792_; double v___x_1793_; double v___x_1794_; double v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_15813__overap_1800_; lean_object* v___x_1801_; 
v___x_1790_ = lean_io_mono_nanos_now();
v___x_1791_ = lean_float_of_nat(v___y_1788_);
v___x_1792_ = lean_float_once(&lp_aesop_Aesop_ExtractScript_visitGoal___closed__2, &lp_aesop_Aesop_ExtractScript_visitGoal___closed__2_once, _init_lp_aesop_Aesop_ExtractScript_visitGoal___closed__2);
v___x_1793_ = lean_float_div(v___x_1791_, v___x_1792_);
v___x_1794_ = lean_float_of_nat(v___x_1790_);
v___x_1795_ = lean_float_div(v___x_1794_, v___x_1792_);
v___x_1796_ = lean_box_float(v___x_1793_);
v___x_1797_ = lean_box_float(v___x_1795_);
v___x_1798_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1798_, 0, v___x_1796_);
lean_ctor_set(v___x_1798_, 1, v___x_1797_);
v___x_1799_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1799_, 0, v_a_1789_);
lean_ctor_set(v___x_1799_, 1, v___x_1798_);
lean_inc(v_traceClass_1778_);
lean_inc_ref(v_toMonadRef_1762_);
v___x_15813__overap_1800_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1759_, v___x_1760_, v_toMonadRef_1762_, v___f_1780_, lean_box(0), v___x_1763_, v___f_1781_, v_traceClass_1778_, v_hasTrace_1765_, v___x_1782_, v_options_1764_, v___x_1785_, v___y_1787_, v___f_1779_, v___x_1799_);
lean_inc(v_a_1746_);
lean_inc_ref(v_a_1745_);
lean_inc(v_a_1744_);
lean_inc_ref(v_a_1743_);
lean_inc(v_a_1742_);
lean_inc(v_a_1741_);
lean_inc_ref(v_a_1740_);
v___x_1801_ = lean_apply_8(v___x_15813__overap_1800_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_, lean_box(0));
return v___x_1801_;
}
v___jp_1802_:
{
lean_object* v___x_1806_; 
v___x_1806_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1806_, 0, v_a_1805_);
v___y_1787_ = v___y_1803_;
v___y_1788_ = v___y_1804_;
v_a_1789_ = v___x_1806_;
goto v___jp_1786_;
}
v___jp_1807_:
{
lean_object* v___x_1811_; double v___x_1812_; double v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_15843__overap_1818_; lean_object* v___x_1819_; 
v___x_1811_ = lean_io_get_num_heartbeats();
v___x_1812_ = lean_float_of_nat(v___y_1808_);
v___x_1813_ = lean_float_of_nat(v___x_1811_);
v___x_1814_ = lean_box_float(v___x_1812_);
v___x_1815_ = lean_box_float(v___x_1813_);
v___x_1816_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1816_, 0, v___x_1814_);
lean_ctor_set(v___x_1816_, 1, v___x_1815_);
v___x_1817_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1817_, 0, v_a_1810_);
lean_ctor_set(v___x_1817_, 1, v___x_1816_);
lean_inc(v_traceClass_1778_);
lean_inc_ref(v_toMonadRef_1762_);
v___x_15843__overap_1818_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_1759_, v___x_1760_, v_toMonadRef_1762_, v___f_1780_, lean_box(0), v___x_1763_, v___f_1781_, v_traceClass_1778_, v_hasTrace_1765_, v___x_1782_, v_options_1764_, v___x_1785_, v___y_1809_, v___f_1779_, v___x_1817_);
lean_inc(v_a_1746_);
lean_inc_ref(v_a_1745_);
lean_inc(v_a_1744_);
lean_inc_ref(v_a_1743_);
lean_inc(v_a_1742_);
lean_inc(v_a_1741_);
lean_inc_ref(v_a_1740_);
v___x_1819_ = lean_apply_8(v___x_15843__overap_1818_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_, lean_box(0));
return v___x_1819_;
}
v___jp_1820_:
{
lean_object* v___x_1824_; 
v___x_1824_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1824_, 0, v_a_1823_);
v___y_1808_ = v___y_1821_;
v___y_1809_ = v___y_1822_;
v_a_1810_ = v___x_1824_;
goto v___jp_1807_;
}
v___jp_1825_:
{
lean_object* v___x_15790__overap_1826_; lean_object* v___x_1827_; 
v___x_15790__overap_1826_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_1759_, v___x_1760_);
lean_inc(v_a_1746_);
lean_inc_ref(v_a_1745_);
lean_inc(v_a_1744_);
lean_inc_ref(v_a_1743_);
lean_inc(v_a_1742_);
lean_inc(v_a_1741_);
lean_inc_ref(v_a_1740_);
v___x_1827_ = lean_apply_8(v___x_15790__overap_1826_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_, lean_box(0));
if (lean_obj_tag(v___x_1827_) == 0)
{
lean_object* v_a_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; uint8_t v___x_1832_; 
v_a_1828_ = lean_ctor_get(v___x_1827_, 0);
lean_inc(v_a_1828_);
lean_dec_ref_known(v___x_1827_, 1);
v___x_1829_ = l_Lean_KVMap_instValueBool;
v___x_1830_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1831_ = l_Lean_Option_get___redArg(v___x_1829_, v_options_1764_, v___x_1830_);
v___x_1832_ = lean_unbox(v___x_1831_);
lean_dec(v___x_1831_);
if (v___x_1832_ == 0)
{
lean_object* v___x_1833_; lean_object* v___x_1834_; 
v___x_1833_ = lean_io_mono_nanos_now();
v___x_1834_ = lp_aesop_Aesop_getRootGoal(v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
if (lean_obj_tag(v___x_1834_) == 0)
{
lean_object* v_a_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; 
v_a_1835_ = lean_ctor_get(v___x_1834_, 0);
lean_inc(v_a_1835_);
lean_dec_ref_known(v___x_1834_, 1);
v___x_1836_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_1836_, 0, v_a_1835_);
v___x_1837_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_1836_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
if (lean_obj_tag(v___x_1837_) == 0)
{
lean_object* v_a_1838_; lean_object* v___x_1840_; uint8_t v_isShared_1841_; uint8_t v_isSharedCheck_1845_; 
v_a_1838_ = lean_ctor_get(v___x_1837_, 0);
v_isSharedCheck_1845_ = !lean_is_exclusive(v___x_1837_);
if (v_isSharedCheck_1845_ == 0)
{
v___x_1840_ = v___x_1837_;
v_isShared_1841_ = v_isSharedCheck_1845_;
goto v_resetjp_1839_;
}
else
{
lean_inc(v_a_1838_);
lean_dec(v___x_1837_);
v___x_1840_ = lean_box(0);
v_isShared_1841_ = v_isSharedCheck_1845_;
goto v_resetjp_1839_;
}
v_resetjp_1839_:
{
lean_object* v___x_1843_; 
if (v_isShared_1841_ == 0)
{
lean_ctor_set_tag(v___x_1840_, 1);
v___x_1843_ = v___x_1840_;
goto v_reusejp_1842_;
}
else
{
lean_object* v_reuseFailAlloc_1844_; 
v_reuseFailAlloc_1844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1844_, 0, v_a_1838_);
v___x_1843_ = v_reuseFailAlloc_1844_;
goto v_reusejp_1842_;
}
v_reusejp_1842_:
{
v___y_1787_ = v_a_1828_;
v___y_1788_ = v___x_1833_;
v_a_1789_ = v___x_1843_;
goto v___jp_1786_;
}
}
}
else
{
lean_object* v_a_1846_; 
v_a_1846_ = lean_ctor_get(v___x_1837_, 0);
lean_inc(v_a_1846_);
lean_dec_ref_known(v___x_1837_, 1);
v___y_1803_ = v_a_1828_;
v___y_1804_ = v___x_1833_;
v_a_1805_ = v_a_1846_;
goto v___jp_1802_;
}
}
else
{
lean_object* v_a_1847_; 
v_a_1847_ = lean_ctor_get(v___x_1834_, 0);
lean_inc(v_a_1847_);
lean_dec_ref_known(v___x_1834_, 1);
v___y_1803_ = v_a_1828_;
v___y_1804_ = v___x_1833_;
v_a_1805_ = v_a_1847_;
goto v___jp_1802_;
}
}
else
{
lean_object* v___x_1848_; lean_object* v___x_1849_; 
v___x_1848_ = lean_io_get_num_heartbeats();
v___x_1849_ = lp_aesop_Aesop_getRootGoal(v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
if (lean_obj_tag(v___x_1849_) == 0)
{
lean_object* v_a_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; 
v_a_1850_ = lean_ctor_get(v___x_1849_, 0);
lean_inc(v_a_1850_);
lean_dec_ref_known(v___x_1849_, 1);
v___x_1851_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_1851_, 0, v_a_1850_);
v___x_1852_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_1851_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_, v_a_1745_, v_a_1746_);
if (lean_obj_tag(v___x_1852_) == 0)
{
lean_object* v_a_1853_; lean_object* v___x_1855_; uint8_t v_isShared_1856_; uint8_t v_isSharedCheck_1860_; 
v_a_1853_ = lean_ctor_get(v___x_1852_, 0);
v_isSharedCheck_1860_ = !lean_is_exclusive(v___x_1852_);
if (v_isSharedCheck_1860_ == 0)
{
v___x_1855_ = v___x_1852_;
v_isShared_1856_ = v_isSharedCheck_1860_;
goto v_resetjp_1854_;
}
else
{
lean_inc(v_a_1853_);
lean_dec(v___x_1852_);
v___x_1855_ = lean_box(0);
v_isShared_1856_ = v_isSharedCheck_1860_;
goto v_resetjp_1854_;
}
v_resetjp_1854_:
{
lean_object* v___x_1858_; 
if (v_isShared_1856_ == 0)
{
lean_ctor_set_tag(v___x_1855_, 1);
v___x_1858_ = v___x_1855_;
goto v_reusejp_1857_;
}
else
{
lean_object* v_reuseFailAlloc_1859_; 
v_reuseFailAlloc_1859_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1859_, 0, v_a_1853_);
v___x_1858_ = v_reuseFailAlloc_1859_;
goto v_reusejp_1857_;
}
v_reusejp_1857_:
{
v___y_1808_ = v___x_1848_;
v___y_1809_ = v_a_1828_;
v_a_1810_ = v___x_1858_;
goto v___jp_1807_;
}
}
}
else
{
lean_object* v_a_1861_; 
v_a_1861_ = lean_ctor_get(v___x_1852_, 0);
lean_inc(v_a_1861_);
lean_dec_ref_known(v___x_1852_, 1);
v___y_1821_ = v___x_1848_;
v___y_1822_ = v_a_1828_;
v_a_1823_ = v_a_1861_;
goto v___jp_1820_;
}
}
else
{
lean_object* v_a_1862_; 
v_a_1862_ = lean_ctor_get(v___x_1849_, 0);
lean_inc(v_a_1862_);
lean_dec_ref_known(v___x_1849_, 1);
v___y_1821_ = v___x_1848_;
v___y_1822_ = v_a_1828_;
v_a_1823_ = v_a_1862_;
goto v___jp_1820_;
}
}
}
else
{
lean_object* v_a_1863_; lean_object* v___x_1865_; uint8_t v_isShared_1866_; uint8_t v_isSharedCheck_1870_; 
v_a_1863_ = lean_ctor_get(v___x_1827_, 0);
v_isSharedCheck_1870_ = !lean_is_exclusive(v___x_1827_);
if (v_isSharedCheck_1870_ == 0)
{
v___x_1865_ = v___x_1827_;
v_isShared_1866_ = v_isSharedCheck_1870_;
goto v_resetjp_1864_;
}
else
{
lean_inc(v_a_1863_);
lean_dec(v___x_1827_);
v___x_1865_ = lean_box(0);
v_isShared_1866_ = v_isSharedCheck_1870_;
goto v_resetjp_1864_;
}
v_resetjp_1864_:
{
lean_object* v___x_1868_; 
if (v_isShared_1866_ == 0)
{
v___x_1868_ = v___x_1865_;
goto v_reusejp_1867_;
}
else
{
lean_object* v_reuseFailAlloc_1869_; 
v_reuseFailAlloc_1869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1869_, 0, v_a_1863_);
v___x_1868_ = v_reuseFailAlloc_1869_;
goto v_reusejp_1867_;
}
v_reusejp_1867_:
{
return v___x_1868_;
}
}
}
}
}
v___jp_1748_:
{
lean_object* v___x_1757_; lean_object* v___x_1758_; 
v___x_1757_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractScriptCore___boxed), 10, 1);
lean_closure_set(v___x_1757_, 0, v_____do__lift_1749_);
v___x_1758_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_1757_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_, v___y_1755_, v___y_1756_);
return v___x_1758_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractScript___boxed(lean_object* v_a_1885_, lean_object* v_a_1886_, lean_object* v_a_1887_, lean_object* v_a_1888_, lean_object* v_a_1889_, lean_object* v_a_1890_, lean_object* v_a_1891_, lean_object* v_a_1892_){
_start:
{
lean_object* v_res_1893_; 
v_res_1893_ = lp_aesop_Aesop_extractScript(v_a_1885_, v_a_1886_, v_a_1887_, v_a_1888_, v_a_1889_, v_a_1890_, v_a_1891_);
lean_dec(v_a_1891_);
lean_dec_ref(v_a_1890_);
lean_dec(v_a_1889_);
lean_dec_ref(v_a_1888_);
lean_dec(v_a_1887_);
lean_dec(v_a_1886_);
lean_dec_ref(v_a_1885_);
return v_res_1893_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___redArg(lean_object* v_val_1894_, lean_object* v_as_1895_, size_t v_sz_1896_, size_t v_i_1897_, lean_object* v_b_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_){
_start:
{
uint8_t v___x_1905_; 
v___x_1905_ = lean_usize_dec_lt(v_i_1897_, v_sz_1896_);
if (v___x_1905_ == 0)
{
lean_object* v___x_1906_; 
lean_dec(v_val_1894_);
v___x_1906_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1906_, 0, v_b_1898_);
return v___x_1906_;
}
else
{
lean_object* v___x_1907_; lean_object* v_elimRapp_1908_; lean_object* v___x_1909_; lean_object* v_metaState_1910_; lean_object* v_a_1911_; lean_object* v___x_1912_; 
v___x_1907_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_1908_ = lean_ctor_get(v___x_1907_, 3);
lean_inc_ref(v_elimRapp_1908_);
lean_inc(v_val_1894_);
v___x_1909_ = lean_apply_1(v_elimRapp_1908_, v_val_1894_);
v_metaState_1910_ = lean_ctor_get(v___x_1909_, 6);
lean_inc_ref(v_metaState_1910_);
lean_dec_ref(v___x_1909_);
v_a_1911_ = lean_array_uget_borrowed(v_as_1895_, v_i_1897_);
lean_inc(v_a_1911_);
v___x_1912_ = lp_aesop_Aesop_Script_Step_mkSorry(v_a_1911_, v_metaState_1910_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_);
if (lean_obj_tag(v___x_1912_) == 0)
{
lean_object* v_a_1913_; lean_object* v___x_1914_; 
v_a_1913_ = lean_ctor_get(v___x_1912_, 0);
lean_inc(v_a_1913_);
lean_dec_ref_known(v___x_1912_, 1);
v___x_1914_ = lp_aesop_Aesop_ExtractScript_recordStep___redArg(v_a_1913_, v___y_1899_);
if (lean_obj_tag(v___x_1914_) == 0)
{
lean_object* v___x_1915_; size_t v___x_1916_; size_t v___x_1917_; 
lean_dec_ref_known(v___x_1914_, 1);
v___x_1915_ = lean_box(0);
v___x_1916_ = ((size_t)1ULL);
v___x_1917_ = lean_usize_add(v_i_1897_, v___x_1916_);
v_i_1897_ = v___x_1917_;
v_b_1898_ = v___x_1915_;
goto _start;
}
else
{
lean_dec(v_val_1894_);
return v___x_1914_;
}
}
else
{
lean_object* v_a_1919_; lean_object* v___x_1921_; uint8_t v_isShared_1922_; uint8_t v_isSharedCheck_1926_; 
lean_dec(v_val_1894_);
v_a_1919_ = lean_ctor_get(v___x_1912_, 0);
v_isSharedCheck_1926_ = !lean_is_exclusive(v___x_1912_);
if (v_isSharedCheck_1926_ == 0)
{
v___x_1921_ = v___x_1912_;
v_isShared_1922_ = v_isSharedCheck_1926_;
goto v_resetjp_1920_;
}
else
{
lean_inc(v_a_1919_);
lean_dec(v___x_1912_);
v___x_1921_ = lean_box(0);
v_isShared_1922_ = v_isSharedCheck_1926_;
goto v_resetjp_1920_;
}
v_resetjp_1920_:
{
lean_object* v___x_1924_; 
if (v_isShared_1922_ == 0)
{
v___x_1924_ = v___x_1921_;
goto v_reusejp_1923_;
}
else
{
lean_object* v_reuseFailAlloc_1925_; 
v_reuseFailAlloc_1925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1925_, 0, v_a_1919_);
v___x_1924_ = v_reuseFailAlloc_1925_;
goto v_reusejp_1923_;
}
v_reusejp_1923_:
{
return v___x_1924_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___redArg___boxed(lean_object* v_val_1927_, lean_object* v_as_1928_, lean_object* v_sz_1929_, lean_object* v_i_1930_, lean_object* v_b_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_){
_start:
{
size_t v_sz_boxed_1938_; size_t v_i_boxed_1939_; lean_object* v_res_1940_; 
v_sz_boxed_1938_ = lean_unbox_usize(v_sz_1929_);
lean_dec(v_sz_1929_);
v_i_boxed_1939_ = lean_unbox_usize(v_i_1930_);
lean_dec(v_i_1930_);
v_res_1940_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___redArg(v_val_1927_, v_as_1928_, v_sz_boxed_1938_, v_i_boxed_1939_, v_b_1931_, v___y_1932_, v___y_1933_, v___y_1934_, v___y_1935_, v___y_1936_);
lean_dec(v___y_1936_);
lean_dec_ref(v___y_1935_);
lean_dec(v___y_1934_);
lean_dec_ref(v___y_1933_);
lean_dec(v___y_1932_);
lean_dec_ref(v_as_1928_);
return v_res_1940_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__2(lean_object* v_f_1941_, lean_object* v_as_1942_, size_t v_i_1943_, size_t v_stop_1944_, lean_object* v_b_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_){
_start:
{
uint8_t v___x_1955_; 
v___x_1955_ = lean_usize_dec_eq(v_i_1943_, v_stop_1944_);
if (v___x_1955_ == 0)
{
lean_object* v___x_1956_; lean_object* v___x_1957_; 
v___x_1956_ = lean_array_uget_borrowed(v_as_1942_, v_i_1943_);
lean_inc_ref(v_f_1941_);
lean_inc(v___y_1953_);
lean_inc_ref(v___y_1952_);
lean_inc(v___y_1951_);
lean_inc_ref(v___y_1950_);
lean_inc(v___y_1949_);
lean_inc(v___y_1948_);
lean_inc_ref(v___y_1947_);
lean_inc(v___y_1946_);
lean_inc(v___x_1956_);
v___x_1957_ = lean_apply_10(v_f_1941_, v___x_1956_, v___y_1946_, v___y_1947_, v___y_1948_, v___y_1949_, v___y_1950_, v___y_1951_, v___y_1952_, v___y_1953_, lean_box(0));
if (lean_obj_tag(v___x_1957_) == 0)
{
lean_object* v_a_1958_; size_t v___x_1959_; size_t v___x_1960_; 
v_a_1958_ = lean_ctor_get(v___x_1957_, 0);
lean_inc(v_a_1958_);
lean_dec_ref_known(v___x_1957_, 1);
v___x_1959_ = ((size_t)1ULL);
v___x_1960_ = lean_usize_add(v_i_1943_, v___x_1959_);
v_i_1943_ = v___x_1960_;
v_b_1945_ = v_a_1958_;
goto _start;
}
else
{
lean_dec_ref(v_f_1941_);
return v___x_1957_;
}
}
else
{
lean_object* v___x_1962_; 
lean_dec_ref(v_f_1941_);
v___x_1962_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1962_, 0, v_b_1945_);
return v___x_1962_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__2___boxed(lean_object* v_f_1963_, lean_object* v_as_1964_, lean_object* v_i_1965_, lean_object* v_stop_1966_, lean_object* v_b_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_){
_start:
{
size_t v_i_boxed_1977_; size_t v_stop_boxed_1978_; lean_object* v_res_1979_; 
v_i_boxed_1977_ = lean_unbox_usize(v_i_1965_);
lean_dec(v_i_1965_);
v_stop_boxed_1978_ = lean_unbox_usize(v_stop_1966_);
lean_dec(v_stop_1966_);
v_res_1979_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__2(v_f_1963_, v_as_1964_, v_i_boxed_1977_, v_stop_boxed_1978_, v_b_1967_, v___y_1968_, v___y_1969_, v___y_1970_, v___y_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_);
lean_dec(v___y_1975_);
lean_dec_ref(v___y_1974_);
lean_dec(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec(v___y_1971_);
lean_dec(v___y_1970_);
lean_dec_ref(v___y_1969_);
lean_dec(v___y_1968_);
lean_dec_ref(v_as_1964_);
return v_res_1979_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__3(lean_object* v_f_1980_, lean_object* v_as_1981_, size_t v_i_1982_, size_t v_stop_1983_, lean_object* v_b_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_){
_start:
{
lean_object* v_a_1995_; lean_object* v___y_2000_; uint8_t v___x_2002_; 
v___x_2002_ = lean_usize_dec_eq(v_i_1982_, v_stop_1983_);
if (v___x_2002_ == 0)
{
lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v_elimMVarCluster_2006_; lean_object* v___x_2007_; lean_object* v_goals_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; uint8_t v___x_2012_; 
v___x_2003_ = lean_array_uget_borrowed(v_as_1981_, v_i_1982_);
v___x_2004_ = lean_st_ref_get(v___x_2003_);
v___x_2005_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_2006_ = lean_ctor_get(v___x_2005_, 5);
lean_inc_ref(v_elimMVarCluster_2006_);
v___x_2007_ = lean_apply_1(v_elimMVarCluster_2006_, v___x_2004_);
v_goals_2008_ = lean_ctor_get(v___x_2007_, 1);
lean_inc_ref(v_goals_2008_);
lean_dec_ref(v___x_2007_);
v___x_2009_ = lean_unsigned_to_nat(0u);
v___x_2010_ = lean_array_get_size(v_goals_2008_);
v___x_2011_ = lean_box(0);
v___x_2012_ = lean_nat_dec_lt(v___x_2009_, v___x_2010_);
if (v___x_2012_ == 0)
{
lean_dec_ref(v_goals_2008_);
v_a_1995_ = v___x_2011_;
goto v___jp_1994_;
}
else
{
uint8_t v___x_2013_; 
v___x_2013_ = lean_nat_dec_le(v___x_2010_, v___x_2010_);
if (v___x_2013_ == 0)
{
if (v___x_2012_ == 0)
{
lean_dec_ref(v_goals_2008_);
v_a_1995_ = v___x_2011_;
goto v___jp_1994_;
}
else
{
size_t v___x_2014_; size_t v___x_2015_; lean_object* v___x_2016_; 
v___x_2014_ = ((size_t)0ULL);
v___x_2015_ = lean_usize_of_nat(v___x_2010_);
lean_inc_ref(v_f_1980_);
v___x_2016_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__2(v_f_1980_, v_goals_2008_, v___x_2014_, v___x_2015_, v___x_2011_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_, v___y_1989_, v___y_1990_, v___y_1991_, v___y_1992_);
lean_dec_ref(v_goals_2008_);
v___y_2000_ = v___x_2016_;
goto v___jp_1999_;
}
}
else
{
size_t v___x_2017_; size_t v___x_2018_; lean_object* v___x_2019_; 
v___x_2017_ = ((size_t)0ULL);
v___x_2018_ = lean_usize_of_nat(v___x_2010_);
lean_inc_ref(v_f_1980_);
v___x_2019_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__2(v_f_1980_, v_goals_2008_, v___x_2017_, v___x_2018_, v___x_2011_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_, v___y_1989_, v___y_1990_, v___y_1991_, v___y_1992_);
lean_dec_ref(v_goals_2008_);
v___y_2000_ = v___x_2019_;
goto v___jp_1999_;
}
}
}
else
{
lean_object* v___x_2020_; 
lean_dec_ref(v_f_1980_);
v___x_2020_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2020_, 0, v_b_1984_);
return v___x_2020_;
}
v___jp_1994_:
{
size_t v___x_1996_; size_t v___x_1997_; 
v___x_1996_ = ((size_t)1ULL);
v___x_1997_ = lean_usize_add(v_i_1982_, v___x_1996_);
v_i_1982_ = v___x_1997_;
v_b_1984_ = v_a_1995_;
goto _start;
}
v___jp_1999_:
{
if (lean_obj_tag(v___y_2000_) == 0)
{
lean_object* v_a_2001_; 
v_a_2001_ = lean_ctor_get(v___y_2000_, 0);
lean_inc(v_a_2001_);
lean_dec_ref_known(v___y_2000_, 1);
v_a_1995_ = v_a_2001_;
goto v___jp_1994_;
}
else
{
lean_dec_ref(v_f_1980_);
return v___y_2000_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__3___boxed(lean_object* v_f_2021_, lean_object* v_as_2022_, lean_object* v_i_2023_, lean_object* v_stop_2024_, lean_object* v_b_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_, lean_object* v___y_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_, lean_object* v___y_2034_){
_start:
{
size_t v_i_boxed_2035_; size_t v_stop_boxed_2036_; lean_object* v_res_2037_; 
v_i_boxed_2035_ = lean_unbox_usize(v_i_2023_);
lean_dec(v_i_2023_);
v_stop_boxed_2036_ = lean_unbox_usize(v_stop_2024_);
lean_dec(v_stop_2024_);
v_res_2037_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__3(v_f_2021_, v_as_2022_, v_i_boxed_2035_, v_stop_boxed_2036_, v_b_2025_, v___y_2026_, v___y_2027_, v___y_2028_, v___y_2029_, v___y_2030_, v___y_2031_, v___y_2032_, v___y_2033_);
lean_dec(v___y_2033_);
lean_dec_ref(v___y_2032_);
lean_dec(v___y_2031_);
lean_dec_ref(v___y_2030_);
lean_dec(v___y_2029_);
lean_dec(v___y_2028_);
lean_dec_ref(v___y_2027_);
lean_dec(v___y_2026_);
lean_dec_ref(v_as_2022_);
return v_res_2037_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2(lean_object* v_f_2038_, lean_object* v_r_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_){
_start:
{
lean_object* v___x_2049_; lean_object* v_elimRapp_2050_; lean_object* v___x_2051_; lean_object* v_children_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; uint8_t v___x_2056_; 
v___x_2049_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_2050_ = lean_ctor_get(v___x_2049_, 3);
lean_inc_ref(v_elimRapp_2050_);
v___x_2051_ = lean_apply_1(v_elimRapp_2050_, v_r_2039_);
v_children_2052_ = lean_ctor_get(v___x_2051_, 2);
lean_inc_ref(v_children_2052_);
lean_dec_ref(v___x_2051_);
v___x_2053_ = lean_unsigned_to_nat(0u);
v___x_2054_ = lean_array_get_size(v_children_2052_);
v___x_2055_ = lean_box(0);
v___x_2056_ = lean_nat_dec_lt(v___x_2053_, v___x_2054_);
if (v___x_2056_ == 0)
{
lean_object* v___x_2057_; 
lean_dec_ref(v_children_2052_);
lean_dec_ref(v_f_2038_);
v___x_2057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2057_, 0, v___x_2055_);
return v___x_2057_;
}
else
{
uint8_t v___x_2058_; 
v___x_2058_ = lean_nat_dec_le(v___x_2054_, v___x_2054_);
if (v___x_2058_ == 0)
{
if (v___x_2056_ == 0)
{
lean_object* v___x_2059_; 
lean_dec_ref(v_children_2052_);
lean_dec_ref(v_f_2038_);
v___x_2059_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2059_, 0, v___x_2055_);
return v___x_2059_;
}
else
{
size_t v___x_2060_; size_t v___x_2061_; lean_object* v___x_2062_; 
v___x_2060_ = ((size_t)0ULL);
v___x_2061_ = lean_usize_of_nat(v___x_2054_);
v___x_2062_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__3(v_f_2038_, v_children_2052_, v___x_2060_, v___x_2061_, v___x_2055_, v___y_2040_, v___y_2041_, v___y_2042_, v___y_2043_, v___y_2044_, v___y_2045_, v___y_2046_, v___y_2047_);
lean_dec_ref(v_children_2052_);
return v___x_2062_;
}
}
else
{
size_t v___x_2063_; size_t v___x_2064_; lean_object* v___x_2065_; 
v___x_2063_ = ((size_t)0ULL);
v___x_2064_ = lean_usize_of_nat(v___x_2054_);
v___x_2065_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2_spec__3(v_f_2038_, v_children_2052_, v___x_2063_, v___x_2064_, v___x_2055_, v___y_2040_, v___y_2041_, v___y_2042_, v___y_2043_, v___y_2044_, v___y_2045_, v___y_2046_, v___y_2047_);
lean_dec_ref(v_children_2052_);
return v___x_2065_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2___boxed(lean_object* v_f_2066_, lean_object* v_r_2067_, lean_object* v___y_2068_, lean_object* v___y_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_){
_start:
{
lean_object* v_res_2077_; 
v_res_2077_ = lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2(v_f_2066_, v_r_2067_, v___y_2068_, v___y_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
lean_dec(v___y_2075_);
lean_dec_ref(v___y_2074_);
lean_dec(v___y_2073_);
lean_dec_ref(v___y_2072_);
lean_dec(v___y_2071_);
lean_dec(v___y_2070_);
lean_dec_ref(v___y_2069_);
lean_dec(v___y_2068_);
return v_res_2077_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__1(void){
_start:
{
lean_object* v___x_2079_; lean_object* v___x_2080_; 
v___x_2079_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__0));
v___x_2080_ = l_Lean_stringToMessageData(v___x_2079_);
return v___x_2080_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__3(void){
_start:
{
lean_object* v___x_2082_; lean_object* v___x_2083_; 
v___x_2082_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__2));
v___x_2083_ = l_Lean_stringToMessageData(v___x_2082_);
return v___x_2083_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__5(void){
_start:
{
lean_object* v___x_2085_; lean_object* v___x_2086_; 
v___x_2085_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__4));
v___x_2086_ = l_Lean_stringToMessageData(v___x_2085_);
return v___x_2086_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__7(void){
_start:
{
lean_object* v___x_2088_; lean_object* v___x_2089_; 
v___x_2088_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__6));
v___x_2089_ = l_Lean_stringToMessageData(v___x_2088_);
return v___x_2089_;
}
}
static lean_object* _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__9(void){
_start:
{
lean_object* v___x_2091_; lean_object* v___x_2092_; 
v___x_2091_ = ((lean_object*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__8));
v___x_2092_ = l_Lean_stringToMessageData(v___x_2091_);
return v___x_2092_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore(lean_object* v_gref_2093_, lean_object* v_a_2094_, lean_object* v_a_2095_, lean_object* v_a_2096_, lean_object* v_a_2097_, lean_object* v_a_2098_, lean_object* v_a_2099_, lean_object* v_a_2100_, lean_object* v_a_2101_){
_start:
{
lean_object* v___x_2103_; lean_object* v___x_2104_; 
v___x_2103_ = lean_st_ref_get(v_gref_2093_);
lean_inc(v___x_2103_);
v___x_2104_ = lp_aesop_Aesop_ExtractScript_visitGoal(v___x_2103_, v_a_2094_, v_a_2095_, v_a_2096_, v_a_2097_, v_a_2098_, v_a_2099_, v_a_2100_, v_a_2101_);
if (lean_obj_tag(v___x_2104_) == 0)
{
lean_object* v___x_2106_; uint8_t v_isShared_2107_; uint8_t v_isSharedCheck_2171_; 
v_isSharedCheck_2171_ = !lean_is_exclusive(v___x_2104_);
if (v_isSharedCheck_2171_ == 0)
{
lean_object* v_unused_2172_; 
v_unused_2172_ = lean_ctor_get(v___x_2104_, 0);
lean_dec(v_unused_2172_);
v___x_2106_ = v___x_2104_;
v_isShared_2107_ = v_isSharedCheck_2171_;
goto v_resetjp_2105_;
}
else
{
lean_dec(v___x_2104_);
v___x_2106_ = lean_box(0);
v_isShared_2107_ = v_isSharedCheck_2171_;
goto v_resetjp_2105_;
}
v_resetjp_2105_:
{
lean_object* v___x_2108_; lean_object* v_elimGoal_2109_; lean_object* v___x_2110_; lean_object* v_id_2111_; lean_object* v_normalizationState_2112_; uint8_t v___x_2113_; 
v___x_2108_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2109_ = lean_ctor_get(v___x_2108_, 1);
lean_inc_ref(v_elimGoal_2109_);
lean_inc(v___x_2103_);
v___x_2110_ = lean_apply_1(v_elimGoal_2109_, v___x_2103_);
v_id_2111_ = lean_ctor_get(v___x_2110_, 0);
lean_inc(v_id_2111_);
v_normalizationState_2112_ = lean_ctor_get(v___x_2110_, 6);
lean_inc(v_normalizationState_2112_);
lean_dec_ref(v___x_2110_);
v___x_2113_ = lp_aesop_Aesop_NormalizationState_isProvenByNormalization(v_normalizationState_2112_);
if (v___x_2113_ == 0)
{
lean_object* v___x_2114_; lean_object* v___y_2116_; lean_object* v___y_2117_; lean_object* v___y_2118_; lean_object* v___y_2119_; lean_object* v___y_2120_; lean_object* v___y_2121_; lean_object* v___y_2122_; lean_object* v___y_2123_; lean_object* v___x_2150_; lean_object* v___x_2151_; uint8_t v___x_2152_; 
lean_del_object(v___x_2106_);
v___x_2114_ = lp_aesop_Aesop_Goal_safeRapps(v___x_2103_);
v___x_2150_ = lean_unsigned_to_nat(1u);
v___x_2151_ = lean_array_get_size(v___x_2114_);
v___x_2152_ = lean_nat_dec_lt(v___x_2150_, v___x_2151_);
if (v___x_2152_ == 0)
{
v___y_2116_ = v_a_2094_;
v___y_2117_ = v_a_2095_;
v___y_2118_ = v_a_2096_;
v___y_2119_ = v_a_2097_;
v___y_2120_ = v_a_2098_;
v___y_2121_ = v_a_2099_;
v___y_2122_ = v_a_2100_;
v___y_2123_ = v_a_2101_;
goto v___jp_2115_;
}
else
{
lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; 
v___x_2153_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__5, &lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__5_once, _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__5);
lean_inc(v_id_2111_);
v___x_2154_ = l_Nat_reprFast(v_id_2111_);
v___x_2155_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2155_, 0, v___x_2154_);
v___x_2156_ = l_Lean_MessageData_ofFormat(v___x_2155_);
v___x_2157_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2157_, 0, v___x_2153_);
lean_ctor_set(v___x_2157_, 1, v___x_2156_);
v___x_2158_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__7, &lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__7_once, _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__7);
v___x_2159_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2159_, 0, v___x_2157_);
lean_ctor_set(v___x_2159_, 1, v___x_2158_);
v___x_2160_ = l_Nat_reprFast(v___x_2151_);
v___x_2161_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2161_, 0, v___x_2160_);
v___x_2162_ = l_Lean_MessageData_ofFormat(v___x_2161_);
v___x_2163_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2163_, 0, v___x_2159_);
lean_ctor_set(v___x_2163_, 1, v___x_2162_);
v___x_2164_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__9, &lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__9_once, _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__9);
v___x_2165_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2165_, 0, v___x_2163_);
lean_ctor_set(v___x_2165_, 1, v___x_2164_);
v___x_2166_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(v___x_2165_, v_a_2098_, v_a_2099_, v_a_2100_, v_a_2101_);
if (lean_obj_tag(v___x_2166_) == 0)
{
lean_dec_ref_known(v___x_2166_, 1);
v___y_2116_ = v_a_2094_;
v___y_2117_ = v_a_2095_;
v___y_2118_ = v_a_2096_;
v___y_2119_ = v_a_2097_;
v___y_2120_ = v_a_2098_;
v___y_2121_ = v_a_2099_;
v___y_2122_ = v_a_2100_;
v___y_2123_ = v_a_2101_;
goto v___jp_2115_;
}
else
{
lean_dec_ref(v___x_2114_);
lean_dec(v_normalizationState_2112_);
lean_dec(v_id_2111_);
return v___x_2166_;
}
}
v___jp_2115_:
{
lean_object* v___x_2124_; lean_object* v___x_2125_; uint8_t v___x_2126_; 
v___x_2124_ = lean_unsigned_to_nat(0u);
v___x_2125_ = lean_array_get_size(v___x_2114_);
v___x_2126_ = lean_nat_dec_lt(v___x_2124_, v___x_2125_);
if (v___x_2126_ == 0)
{
lean_dec_ref(v___x_2114_);
if (lean_obj_tag(v_normalizationState_2112_) == 1)
{
lean_object* v_postGoal_2127_; lean_object* v_postState_2128_; lean_object* v___x_2129_; 
lean_dec(v_id_2111_);
v_postGoal_2127_ = lean_ctor_get(v_normalizationState_2112_, 0);
lean_inc(v_postGoal_2127_);
v_postState_2128_ = lean_ctor_get(v_normalizationState_2112_, 1);
lean_inc_ref(v_postState_2128_);
lean_dec_ref_known(v_normalizationState_2112_, 3);
v___x_2129_ = lp_aesop_Aesop_Script_Step_mkSorry(v_postGoal_2127_, v_postState_2128_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_);
if (lean_obj_tag(v___x_2129_) == 0)
{
lean_object* v_a_2130_; lean_object* v___x_2131_; 
v_a_2130_ = lean_ctor_get(v___x_2129_, 0);
lean_inc(v_a_2130_);
lean_dec_ref_known(v___x_2129_, 1);
v___x_2131_ = lp_aesop_Aesop_ExtractScript_recordStep___redArg(v_a_2130_, v___y_2116_);
return v___x_2131_;
}
else
{
lean_object* v_a_2132_; lean_object* v___x_2134_; uint8_t v_isShared_2135_; uint8_t v_isSharedCheck_2139_; 
v_a_2132_ = lean_ctor_get(v___x_2129_, 0);
v_isSharedCheck_2139_ = !lean_is_exclusive(v___x_2129_);
if (v_isSharedCheck_2139_ == 0)
{
v___x_2134_ = v___x_2129_;
v_isShared_2135_ = v_isSharedCheck_2139_;
goto v_resetjp_2133_;
}
else
{
lean_inc(v_a_2132_);
lean_dec(v___x_2129_);
v___x_2134_ = lean_box(0);
v_isShared_2135_ = v_isSharedCheck_2139_;
goto v_resetjp_2133_;
}
v_resetjp_2133_:
{
lean_object* v___x_2137_; 
if (v_isShared_2135_ == 0)
{
v___x_2137_ = v___x_2134_;
goto v_reusejp_2136_;
}
else
{
lean_object* v_reuseFailAlloc_2138_; 
v_reuseFailAlloc_2138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2138_, 0, v_a_2132_);
v___x_2137_ = v_reuseFailAlloc_2138_;
goto v_reusejp_2136_;
}
v_reusejp_2136_:
{
return v___x_2137_;
}
}
}
}
else
{
lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; 
lean_dec(v_normalizationState_2112_);
v___x_2140_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__1, &lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__1_once, _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__1);
v___x_2141_ = l_Nat_reprFast(v_id_2111_);
v___x_2142_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2142_, 0, v___x_2141_);
v___x_2143_ = l_Lean_MessageData_ofFormat(v___x_2142_);
v___x_2144_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2144_, 0, v___x_2140_);
lean_ctor_set(v___x_2144_, 1, v___x_2143_);
v___x_2145_ = lean_obj_once(&lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__3, &lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__3_once, _init_lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___closed__3);
v___x_2146_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2146_, 0, v___x_2144_);
lean_ctor_set(v___x_2146_, 1, v___x_2145_);
v___x_2147_ = lp_aesop_Lean_throwError___at___00Aesop_ExtractScript_visitGoal_spec__0___redArg(v___x_2146_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_);
return v___x_2147_;
}
}
else
{
lean_object* v___x_2148_; lean_object* v___x_2149_; 
lean_dec(v_normalizationState_2112_);
lean_dec(v_id_2111_);
v___x_2148_ = lean_array_fget(v___x_2114_, v___x_2124_);
lean_dec_ref(v___x_2114_);
v___x_2149_ = lp_aesop_Aesop_RappRef_extractSafePrefixScriptCore(v___x_2148_, v___y_2116_, v___y_2117_, v___y_2118_, v___y_2119_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_);
lean_dec(v___x_2148_);
return v___x_2149_;
}
}
}
else
{
lean_object* v___x_2167_; lean_object* v___x_2169_; 
lean_dec(v_normalizationState_2112_);
lean_dec(v_id_2111_);
lean_dec(v___x_2103_);
v___x_2167_ = lean_box(0);
if (v_isShared_2107_ == 0)
{
lean_ctor_set(v___x_2106_, 0, v___x_2167_);
v___x_2169_ = v___x_2106_;
goto v_reusejp_2168_;
}
else
{
lean_object* v_reuseFailAlloc_2170_; 
v_reuseFailAlloc_2170_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2170_, 0, v___x_2167_);
v___x_2169_ = v_reuseFailAlloc_2170_;
goto v_reusejp_2168_;
}
v_reusejp_2168_:
{
return v___x_2169_;
}
}
}
}
else
{
lean_dec(v___x_2103_);
return v___x_2104_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___boxed(lean_object* v_gref_2173_, lean_object* v_a_2174_, lean_object* v_a_2175_, lean_object* v_a_2176_, lean_object* v_a_2177_, lean_object* v_a_2178_, lean_object* v_a_2179_, lean_object* v_a_2180_, lean_object* v_a_2181_, lean_object* v_a_2182_){
_start:
{
lean_object* v_res_2183_; 
v_res_2183_ = lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore(v_gref_2173_, v_a_2174_, v_a_2175_, v_a_2176_, v_a_2177_, v_a_2178_, v_a_2179_, v_a_2180_, v_a_2181_);
lean_dec(v_a_2181_);
lean_dec_ref(v_a_2180_);
lean_dec(v_a_2179_);
lean_dec_ref(v_a_2178_);
lean_dec(v_a_2177_);
lean_dec(v_a_2176_);
lean_dec_ref(v_a_2175_);
lean_dec(v_a_2174_);
lean_dec(v_gref_2173_);
return v_res_2183_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_extractSafePrefixScriptCore(lean_object* v_rref_2184_, lean_object* v_a_2185_, lean_object* v_a_2186_, lean_object* v_a_2187_, lean_object* v_a_2188_, lean_object* v_a_2189_, lean_object* v_a_2190_, lean_object* v_a_2191_, lean_object* v_a_2192_){
_start:
{
lean_object* v___x_2194_; lean_object* v___x_2195_; 
v___x_2194_ = lean_st_ref_get(v_rref_2184_);
lean_inc(v___x_2194_);
v___x_2195_ = lp_aesop_Aesop_ExtractScript_visitRapp(v___x_2194_, v_a_2185_, v_a_2186_, v_a_2187_, v_a_2188_, v_a_2189_, v_a_2190_, v_a_2191_, v_a_2192_);
if (lean_obj_tag(v___x_2195_) == 0)
{
lean_object* v___x_2196_; lean_object* v_elimRapp_2197_; lean_object* v___x_2198_; lean_object* v_introducedMVars_2199_; lean_object* v___x_2200_; size_t v_sz_2201_; size_t v___x_2202_; lean_object* v___x_2203_; 
lean_dec_ref_known(v___x_2195_, 1);
v___x_2196_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_2197_ = lean_ctor_get(v___x_2196_, 3);
lean_inc_ref(v_elimRapp_2197_);
lean_inc_n(v___x_2194_, 2);
v___x_2198_ = lean_apply_1(v_elimRapp_2197_, v___x_2194_);
v_introducedMVars_2199_ = lean_ctor_get(v___x_2198_, 7);
lean_inc_ref(v_introducedMVars_2199_);
lean_dec_ref(v___x_2198_);
v___x_2200_ = lean_box(0);
v_sz_2201_ = lean_array_size(v_introducedMVars_2199_);
v___x_2202_ = ((size_t)0ULL);
v___x_2203_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___redArg(v___x_2194_, v_introducedMVars_2199_, v_sz_2201_, v___x_2202_, v___x_2200_, v_a_2185_, v_a_2189_, v_a_2190_, v_a_2191_, v_a_2192_);
lean_dec_ref(v_introducedMVars_2199_);
if (lean_obj_tag(v___x_2203_) == 0)
{
lean_object* v___f_2204_; lean_object* v___x_2205_; 
lean_dec_ref_known(v___x_2203_, 1);
v___f_2204_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___boxed), 10, 0);
v___x_2205_ = lp_aesop_Aesop_Rapp_forSubgoalsM___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__2(v___f_2204_, v___x_2194_, v_a_2185_, v_a_2186_, v_a_2187_, v_a_2188_, v_a_2189_, v_a_2190_, v_a_2191_, v_a_2192_);
return v___x_2205_;
}
else
{
lean_dec(v___x_2194_);
return v___x_2203_;
}
}
else
{
lean_dec(v___x_2194_);
return v___x_2195_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_extractSafePrefixScriptCore___boxed(lean_object* v_rref_2206_, lean_object* v_a_2207_, lean_object* v_a_2208_, lean_object* v_a_2209_, lean_object* v_a_2210_, lean_object* v_a_2211_, lean_object* v_a_2212_, lean_object* v_a_2213_, lean_object* v_a_2214_, lean_object* v_a_2215_){
_start:
{
lean_object* v_res_2216_; 
v_res_2216_ = lp_aesop_Aesop_RappRef_extractSafePrefixScriptCore(v_rref_2206_, v_a_2207_, v_a_2208_, v_a_2209_, v_a_2210_, v_a_2211_, v_a_2212_, v_a_2213_, v_a_2214_);
lean_dec(v_a_2214_);
lean_dec_ref(v_a_2213_);
lean_dec(v_a_2212_);
lean_dec_ref(v_a_2211_);
lean_dec(v_a_2210_);
lean_dec(v_a_2209_);
lean_dec_ref(v_a_2208_);
lean_dec(v_a_2207_);
lean_dec(v_rref_2206_);
return v_res_2216_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1(lean_object* v_val_2217_, lean_object* v_as_2218_, size_t v_sz_2219_, size_t v_i_2220_, lean_object* v_b_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_, lean_object* v___y_2229_){
_start:
{
lean_object* v___x_2231_; 
v___x_2231_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___redArg(v_val_2217_, v_as_2218_, v_sz_2219_, v_i_2220_, v_b_2221_, v___y_2222_, v___y_2226_, v___y_2227_, v___y_2228_, v___y_2229_);
return v___x_2231_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1___boxed(lean_object* v_val_2232_, lean_object* v_as_2233_, lean_object* v_sz_2234_, lean_object* v_i_2235_, lean_object* v_b_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_, lean_object* v___y_2240_, lean_object* v___y_2241_, lean_object* v___y_2242_, lean_object* v___y_2243_, lean_object* v___y_2244_, lean_object* v___y_2245_){
_start:
{
size_t v_sz_boxed_2246_; size_t v_i_boxed_2247_; lean_object* v_res_2248_; 
v_sz_boxed_2246_ = lean_unbox_usize(v_sz_2234_);
lean_dec(v_sz_2234_);
v_i_boxed_2247_ = lean_unbox_usize(v_i_2235_);
lean_dec(v_i_2235_);
v_res_2248_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RappRef_extractSafePrefixScriptCore_spec__1(v_val_2232_, v_as_2233_, v_sz_boxed_2246_, v_i_boxed_2247_, v_b_2236_, v___y_2237_, v___y_2238_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_, v___y_2243_, v___y_2244_);
lean_dec(v___y_2244_);
lean_dec_ref(v___y_2243_);
lean_dec(v___y_2242_);
lean_dec_ref(v___y_2241_);
lean_dec(v___y_2240_);
lean_dec(v___y_2239_);
lean_dec_ref(v___y_2238_);
lean_dec(v___y_2237_);
lean_dec_ref(v_as_2233_);
return v_res_2248_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_MVarClusterRef_extractSafePrefixScriptCore_spec__0(lean_object* v_as_2249_, size_t v_i_2250_, size_t v_stop_2251_, lean_object* v_b_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_){
_start:
{
uint8_t v___x_2262_; 
v___x_2262_ = lean_usize_dec_eq(v_i_2250_, v_stop_2251_);
if (v___x_2262_ == 0)
{
lean_object* v___x_2263_; lean_object* v___x_2264_; 
v___x_2263_ = lean_array_uget_borrowed(v_as_2249_, v_i_2250_);
v___x_2264_ = lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore(v___x_2263_, v___y_2253_, v___y_2254_, v___y_2255_, v___y_2256_, v___y_2257_, v___y_2258_, v___y_2259_, v___y_2260_);
if (lean_obj_tag(v___x_2264_) == 0)
{
lean_object* v_a_2265_; size_t v___x_2266_; size_t v___x_2267_; 
v_a_2265_ = lean_ctor_get(v___x_2264_, 0);
lean_inc(v_a_2265_);
lean_dec_ref_known(v___x_2264_, 1);
v___x_2266_ = ((size_t)1ULL);
v___x_2267_ = lean_usize_add(v_i_2250_, v___x_2266_);
v_i_2250_ = v___x_2267_;
v_b_2252_ = v_a_2265_;
goto _start;
}
else
{
return v___x_2264_;
}
}
else
{
lean_object* v___x_2269_; 
v___x_2269_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2269_, 0, v_b_2252_);
return v___x_2269_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_MVarClusterRef_extractSafePrefixScriptCore_spec__0___boxed(lean_object* v_as_2270_, lean_object* v_i_2271_, lean_object* v_stop_2272_, lean_object* v_b_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_, lean_object* v___y_2281_, lean_object* v___y_2282_){
_start:
{
size_t v_i_boxed_2283_; size_t v_stop_boxed_2284_; lean_object* v_res_2285_; 
v_i_boxed_2283_ = lean_unbox_usize(v_i_2271_);
lean_dec(v_i_2271_);
v_stop_boxed_2284_ = lean_unbox_usize(v_stop_2272_);
lean_dec(v_stop_2272_);
v_res_2285_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_MVarClusterRef_extractSafePrefixScriptCore_spec__0(v_as_2270_, v_i_boxed_2283_, v_stop_boxed_2284_, v_b_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_, v___y_2280_, v___y_2281_);
lean_dec(v___y_2281_);
lean_dec_ref(v___y_2280_);
lean_dec(v___y_2279_);
lean_dec_ref(v___y_2278_);
lean_dec(v___y_2277_);
lean_dec(v___y_2276_);
lean_dec_ref(v___y_2275_);
lean_dec(v___y_2274_);
lean_dec_ref(v_as_2270_);
return v_res_2285_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_extractSafePrefixScriptCore(lean_object* v_mref_2286_, lean_object* v_a_2287_, lean_object* v_a_2288_, lean_object* v_a_2289_, lean_object* v_a_2290_, lean_object* v_a_2291_, lean_object* v_a_2292_, lean_object* v_a_2293_, lean_object* v_a_2294_){
_start:
{
lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v_elimMVarCluster_2298_; lean_object* v___x_2299_; lean_object* v_goals_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; uint8_t v___x_2304_; 
v___x_2296_ = lean_st_ref_get(v_mref_2286_);
v___x_2297_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_2298_ = lean_ctor_get(v___x_2297_, 5);
lean_inc_ref(v_elimMVarCluster_2298_);
v___x_2299_ = lean_apply_1(v_elimMVarCluster_2298_, v___x_2296_);
v_goals_2300_ = lean_ctor_get(v___x_2299_, 1);
lean_inc_ref(v_goals_2300_);
lean_dec_ref(v___x_2299_);
v___x_2301_ = lean_unsigned_to_nat(0u);
v___x_2302_ = lean_array_get_size(v_goals_2300_);
v___x_2303_ = lean_box(0);
v___x_2304_ = lean_nat_dec_lt(v___x_2301_, v___x_2302_);
if (v___x_2304_ == 0)
{
lean_object* v___x_2305_; 
lean_dec_ref(v_goals_2300_);
v___x_2305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2305_, 0, v___x_2303_);
return v___x_2305_;
}
else
{
uint8_t v___x_2306_; 
v___x_2306_ = lean_nat_dec_le(v___x_2302_, v___x_2302_);
if (v___x_2306_ == 0)
{
if (v___x_2304_ == 0)
{
lean_object* v___x_2307_; 
lean_dec_ref(v_goals_2300_);
v___x_2307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2307_, 0, v___x_2303_);
return v___x_2307_;
}
else
{
size_t v___x_2308_; size_t v___x_2309_; lean_object* v___x_2310_; 
v___x_2308_ = ((size_t)0ULL);
v___x_2309_ = lean_usize_of_nat(v___x_2302_);
v___x_2310_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_MVarClusterRef_extractSafePrefixScriptCore_spec__0(v_goals_2300_, v___x_2308_, v___x_2309_, v___x_2303_, v_a_2287_, v_a_2288_, v_a_2289_, v_a_2290_, v_a_2291_, v_a_2292_, v_a_2293_, v_a_2294_);
lean_dec_ref(v_goals_2300_);
return v___x_2310_;
}
}
else
{
size_t v___x_2311_; size_t v___x_2312_; lean_object* v___x_2313_; 
v___x_2311_ = ((size_t)0ULL);
v___x_2312_ = lean_usize_of_nat(v___x_2302_);
v___x_2313_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_MVarClusterRef_extractSafePrefixScriptCore_spec__0(v_goals_2300_, v___x_2311_, v___x_2312_, v___x_2303_, v_a_2287_, v_a_2288_, v_a_2289_, v_a_2290_, v_a_2291_, v_a_2292_, v_a_2293_, v_a_2294_);
lean_dec_ref(v_goals_2300_);
return v___x_2313_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_extractSafePrefixScriptCore___boxed(lean_object* v_mref_2314_, lean_object* v_a_2315_, lean_object* v_a_2316_, lean_object* v_a_2317_, lean_object* v_a_2318_, lean_object* v_a_2319_, lean_object* v_a_2320_, lean_object* v_a_2321_, lean_object* v_a_2322_, lean_object* v_a_2323_){
_start:
{
lean_object* v_res_2324_; 
v_res_2324_ = lp_aesop_Aesop_MVarClusterRef_extractSafePrefixScriptCore(v_mref_2314_, v_a_2315_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_, v_a_2320_, v_a_2321_, v_a_2322_);
lean_dec(v_a_2322_);
lean_dec_ref(v_a_2321_);
lean_dec(v_a_2320_);
lean_dec_ref(v_a_2319_);
lean_dec(v_a_2318_);
lean_dec(v_a_2317_);
lean_dec_ref(v_a_2316_);
lean_dec(v_a_2315_);
lean_dec(v_mref_2314_);
return v_res_2324_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___redArg(lean_object* v___y_2325_){
_start:
{
lean_object* v___x_2327_; lean_object* v_traceState_2328_; lean_object* v_traces_2329_; lean_object* v___x_2330_; lean_object* v_traceState_2331_; lean_object* v_env_2332_; lean_object* v_nextMacroScope_2333_; lean_object* v_ngen_2334_; lean_object* v_auxDeclNGen_2335_; lean_object* v_cache_2336_; lean_object* v_messages_2337_; lean_object* v_infoState_2338_; lean_object* v_snapshotTasks_2339_; lean_object* v___x_2341_; uint8_t v_isShared_2342_; uint8_t v_isSharedCheck_2360_; 
v___x_2327_ = lean_st_ref_get(v___y_2325_);
v_traceState_2328_ = lean_ctor_get(v___x_2327_, 4);
lean_inc_ref(v_traceState_2328_);
lean_dec(v___x_2327_);
v_traces_2329_ = lean_ctor_get(v_traceState_2328_, 0);
lean_inc_ref(v_traces_2329_);
lean_dec_ref(v_traceState_2328_);
v___x_2330_ = lean_st_ref_take(v___y_2325_);
v_traceState_2331_ = lean_ctor_get(v___x_2330_, 4);
v_env_2332_ = lean_ctor_get(v___x_2330_, 0);
v_nextMacroScope_2333_ = lean_ctor_get(v___x_2330_, 1);
v_ngen_2334_ = lean_ctor_get(v___x_2330_, 2);
v_auxDeclNGen_2335_ = lean_ctor_get(v___x_2330_, 3);
v_cache_2336_ = lean_ctor_get(v___x_2330_, 5);
v_messages_2337_ = lean_ctor_get(v___x_2330_, 6);
v_infoState_2338_ = lean_ctor_get(v___x_2330_, 7);
v_snapshotTasks_2339_ = lean_ctor_get(v___x_2330_, 8);
v_isSharedCheck_2360_ = !lean_is_exclusive(v___x_2330_);
if (v_isSharedCheck_2360_ == 0)
{
v___x_2341_ = v___x_2330_;
v_isShared_2342_ = v_isSharedCheck_2360_;
goto v_resetjp_2340_;
}
else
{
lean_inc(v_snapshotTasks_2339_);
lean_inc(v_infoState_2338_);
lean_inc(v_messages_2337_);
lean_inc(v_cache_2336_);
lean_inc(v_traceState_2331_);
lean_inc(v_auxDeclNGen_2335_);
lean_inc(v_ngen_2334_);
lean_inc(v_nextMacroScope_2333_);
lean_inc(v_env_2332_);
lean_dec(v___x_2330_);
v___x_2341_ = lean_box(0);
v_isShared_2342_ = v_isSharedCheck_2360_;
goto v_resetjp_2340_;
}
v_resetjp_2340_:
{
uint64_t v_tid_2343_; lean_object* v___x_2345_; uint8_t v_isShared_2346_; uint8_t v_isSharedCheck_2358_; 
v_tid_2343_ = lean_ctor_get_uint64(v_traceState_2331_, sizeof(void*)*1);
v_isSharedCheck_2358_ = !lean_is_exclusive(v_traceState_2331_);
if (v_isSharedCheck_2358_ == 0)
{
lean_object* v_unused_2359_; 
v_unused_2359_ = lean_ctor_get(v_traceState_2331_, 0);
lean_dec(v_unused_2359_);
v___x_2345_ = v_traceState_2331_;
v_isShared_2346_ = v_isSharedCheck_2358_;
goto v_resetjp_2344_;
}
else
{
lean_dec(v_traceState_2331_);
v___x_2345_ = lean_box(0);
v_isShared_2346_ = v_isSharedCheck_2358_;
goto v_resetjp_2344_;
}
v_resetjp_2344_:
{
lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2351_; 
v___x_2347_ = lean_unsigned_to_nat(32u);
v___x_2348_ = lean_mk_empty_array_with_capacity(v___x_2347_);
lean_dec_ref(v___x_2348_);
v___x_2349_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ExtractScript_visitGoal_spec__2___redArg___closed__1);
if (v_isShared_2346_ == 0)
{
lean_ctor_set(v___x_2345_, 0, v___x_2349_);
v___x_2351_ = v___x_2345_;
goto v_reusejp_2350_;
}
else
{
lean_object* v_reuseFailAlloc_2357_; 
v_reuseFailAlloc_2357_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2357_, 0, v___x_2349_);
lean_ctor_set_uint64(v_reuseFailAlloc_2357_, sizeof(void*)*1, v_tid_2343_);
v___x_2351_ = v_reuseFailAlloc_2357_;
goto v_reusejp_2350_;
}
v_reusejp_2350_:
{
lean_object* v___x_2353_; 
if (v_isShared_2342_ == 0)
{
lean_ctor_set(v___x_2341_, 4, v___x_2351_);
v___x_2353_ = v___x_2341_;
goto v_reusejp_2352_;
}
else
{
lean_object* v_reuseFailAlloc_2356_; 
v_reuseFailAlloc_2356_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2356_, 0, v_env_2332_);
lean_ctor_set(v_reuseFailAlloc_2356_, 1, v_nextMacroScope_2333_);
lean_ctor_set(v_reuseFailAlloc_2356_, 2, v_ngen_2334_);
lean_ctor_set(v_reuseFailAlloc_2356_, 3, v_auxDeclNGen_2335_);
lean_ctor_set(v_reuseFailAlloc_2356_, 4, v___x_2351_);
lean_ctor_set(v_reuseFailAlloc_2356_, 5, v_cache_2336_);
lean_ctor_set(v_reuseFailAlloc_2356_, 6, v_messages_2337_);
lean_ctor_set(v_reuseFailAlloc_2356_, 7, v_infoState_2338_);
lean_ctor_set(v_reuseFailAlloc_2356_, 8, v_snapshotTasks_2339_);
v___x_2353_ = v_reuseFailAlloc_2356_;
goto v_reusejp_2352_;
}
v_reusejp_2352_:
{
lean_object* v___x_2354_; lean_object* v___x_2355_; 
v___x_2354_ = lean_st_ref_set(v___y_2325_, v___x_2353_);
v___x_2355_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2355_, 0, v_traces_2329_);
return v___x_2355_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___redArg___boxed(lean_object* v___y_2361_, lean_object* v___y_2362_){
_start:
{
lean_object* v_res_2363_; 
v_res_2363_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___redArg(v___y_2361_);
lean_dec(v___y_2361_);
return v_res_2363_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0(lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_, lean_object* v___y_2370_){
_start:
{
lean_object* v___x_2372_; 
v___x_2372_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___redArg(v___y_2370_);
return v___x_2372_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___boxed(lean_object* v___y_2373_, lean_object* v___y_2374_, lean_object* v___y_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_){
_start:
{
lean_object* v_res_2381_; 
v_res_2381_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0(v___y_2373_, v___y_2374_, v___y_2375_, v___y_2376_, v___y_2377_, v___y_2378_, v___y_2379_);
lean_dec(v___y_2379_);
lean_dec_ref(v___y_2378_);
lean_dec(v___y_2377_);
lean_dec_ref(v___y_2376_);
lean_dec(v___y_2375_);
lean_dec(v___y_2374_);
lean_dec_ref(v___y_2373_);
return v_res_2381_;
}
}
static lean_object* _init_lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2383_; lean_object* v___x_2384_; 
v___x_2383_ = ((lean_object*)(lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__0));
v___x_2384_ = l_Lean_stringToMessageData(v___x_2383_);
return v___x_2384_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefixScript___lam__0(lean_object* v_x_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_, lean_object* v___y_2392_){
_start:
{
lean_object* v___x_2394_; lean_object* v___x_2395_; 
v___x_2394_ = lean_obj_once(&lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__1, &lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__1_once, _init_lp_aesop_Aesop_extractSafePrefixScript___lam__0___closed__1);
v___x_2395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2395_, 0, v___x_2394_);
return v___x_2395_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefixScript___lam__0___boxed(lean_object* v_x_2396_, lean_object* v___y_2397_, lean_object* v___y_2398_, lean_object* v___y_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_){
_start:
{
lean_object* v_res_2405_; 
v_res_2405_ = lp_aesop_Aesop_extractSafePrefixScript___lam__0(v_x_2396_, v___y_2397_, v___y_2398_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_, v___y_2403_);
lean_dec(v___y_2403_);
lean_dec_ref(v___y_2402_);
lean_dec(v___y_2401_);
lean_dec_ref(v___y_2400_);
lean_dec(v___y_2399_);
lean_dec(v___y_2398_);
lean_dec_ref(v___y_2397_);
lean_dec_ref(v_x_2396_);
return v_res_2405_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___redArg(lean_object* v_x_2406_){
_start:
{
if (lean_obj_tag(v_x_2406_) == 0)
{
lean_object* v_a_2408_; lean_object* v___x_2410_; uint8_t v_isShared_2411_; uint8_t v_isSharedCheck_2415_; 
v_a_2408_ = lean_ctor_get(v_x_2406_, 0);
v_isSharedCheck_2415_ = !lean_is_exclusive(v_x_2406_);
if (v_isSharedCheck_2415_ == 0)
{
v___x_2410_ = v_x_2406_;
v_isShared_2411_ = v_isSharedCheck_2415_;
goto v_resetjp_2409_;
}
else
{
lean_inc(v_a_2408_);
lean_dec(v_x_2406_);
v___x_2410_ = lean_box(0);
v_isShared_2411_ = v_isSharedCheck_2415_;
goto v_resetjp_2409_;
}
v_resetjp_2409_:
{
lean_object* v___x_2413_; 
if (v_isShared_2411_ == 0)
{
lean_ctor_set_tag(v___x_2410_, 1);
v___x_2413_ = v___x_2410_;
goto v_reusejp_2412_;
}
else
{
lean_object* v_reuseFailAlloc_2414_; 
v_reuseFailAlloc_2414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2414_, 0, v_a_2408_);
v___x_2413_ = v_reuseFailAlloc_2414_;
goto v_reusejp_2412_;
}
v_reusejp_2412_:
{
return v___x_2413_;
}
}
}
else
{
lean_object* v_a_2416_; lean_object* v___x_2418_; uint8_t v_isShared_2419_; uint8_t v_isSharedCheck_2423_; 
v_a_2416_ = lean_ctor_get(v_x_2406_, 0);
v_isSharedCheck_2423_ = !lean_is_exclusive(v_x_2406_);
if (v_isSharedCheck_2423_ == 0)
{
v___x_2418_ = v_x_2406_;
v_isShared_2419_ = v_isSharedCheck_2423_;
goto v_resetjp_2417_;
}
else
{
lean_inc(v_a_2416_);
lean_dec(v_x_2406_);
v___x_2418_ = lean_box(0);
v_isShared_2419_ = v_isSharedCheck_2423_;
goto v_resetjp_2417_;
}
v_resetjp_2417_:
{
lean_object* v___x_2421_; 
if (v_isShared_2419_ == 0)
{
lean_ctor_set_tag(v___x_2418_, 0);
v___x_2421_ = v___x_2418_;
goto v_reusejp_2420_;
}
else
{
lean_object* v_reuseFailAlloc_2422_; 
v_reuseFailAlloc_2422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2422_, 0, v_a_2416_);
v___x_2421_ = v_reuseFailAlloc_2422_;
goto v_reusejp_2420_;
}
v_reusejp_2420_:
{
return v___x_2421_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___redArg___boxed(lean_object* v_x_2424_, lean_object* v___y_2425_){
_start:
{
lean_object* v_res_2426_; 
v_res_2426_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___redArg(v_x_2424_);
return v_res_2426_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___redArg(lean_object* v_oldTraces_2427_, lean_object* v_data_2428_, lean_object* v_ref_2429_, lean_object* v_msg_2430_, lean_object* v___y_2431_, lean_object* v___y_2432_, lean_object* v___y_2433_, lean_object* v___y_2434_){
_start:
{
lean_object* v_fileName_2436_; lean_object* v_fileMap_2437_; lean_object* v_options_2438_; lean_object* v_currRecDepth_2439_; lean_object* v_maxRecDepth_2440_; lean_object* v_ref_2441_; lean_object* v_currNamespace_2442_; lean_object* v_openDecls_2443_; lean_object* v_initHeartbeats_2444_; lean_object* v_maxHeartbeats_2445_; lean_object* v_quotContext_2446_; lean_object* v_currMacroScope_2447_; uint8_t v_diag_2448_; lean_object* v_cancelTk_x3f_2449_; uint8_t v_suppressElabErrors_2450_; lean_object* v_inheritedTraceOptions_2451_; lean_object* v___x_2452_; lean_object* v_traceState_2453_; lean_object* v_traces_2454_; lean_object* v_ref_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; size_t v_sz_2458_; size_t v___x_2459_; lean_object* v___x_2460_; lean_object* v_msg_2461_; lean_object* v___x_2462_; lean_object* v_a_2463_; lean_object* v___x_2465_; uint8_t v_isShared_2466_; uint8_t v_isSharedCheck_2500_; 
v_fileName_2436_ = lean_ctor_get(v___y_2433_, 0);
v_fileMap_2437_ = lean_ctor_get(v___y_2433_, 1);
v_options_2438_ = lean_ctor_get(v___y_2433_, 2);
v_currRecDepth_2439_ = lean_ctor_get(v___y_2433_, 3);
v_maxRecDepth_2440_ = lean_ctor_get(v___y_2433_, 4);
v_ref_2441_ = lean_ctor_get(v___y_2433_, 5);
v_currNamespace_2442_ = lean_ctor_get(v___y_2433_, 6);
v_openDecls_2443_ = lean_ctor_get(v___y_2433_, 7);
v_initHeartbeats_2444_ = lean_ctor_get(v___y_2433_, 8);
v_maxHeartbeats_2445_ = lean_ctor_get(v___y_2433_, 9);
v_quotContext_2446_ = lean_ctor_get(v___y_2433_, 10);
v_currMacroScope_2447_ = lean_ctor_get(v___y_2433_, 11);
v_diag_2448_ = lean_ctor_get_uint8(v___y_2433_, sizeof(void*)*14);
v_cancelTk_x3f_2449_ = lean_ctor_get(v___y_2433_, 12);
v_suppressElabErrors_2450_ = lean_ctor_get_uint8(v___y_2433_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2451_ = lean_ctor_get(v___y_2433_, 13);
v___x_2452_ = lean_st_ref_get(v___y_2434_);
v_traceState_2453_ = lean_ctor_get(v___x_2452_, 4);
lean_inc_ref(v_traceState_2453_);
lean_dec(v___x_2452_);
v_traces_2454_ = lean_ctor_get(v_traceState_2453_, 0);
lean_inc_ref(v_traces_2454_);
lean_dec_ref(v_traceState_2453_);
v_ref_2455_ = l_Lean_replaceRef(v_ref_2429_, v_ref_2441_);
lean_inc_ref(v_inheritedTraceOptions_2451_);
lean_inc(v_cancelTk_x3f_2449_);
lean_inc(v_currMacroScope_2447_);
lean_inc(v_quotContext_2446_);
lean_inc(v_maxHeartbeats_2445_);
lean_inc(v_initHeartbeats_2444_);
lean_inc(v_openDecls_2443_);
lean_inc(v_currNamespace_2442_);
lean_inc(v_maxRecDepth_2440_);
lean_inc(v_currRecDepth_2439_);
lean_inc_ref(v_options_2438_);
lean_inc_ref(v_fileMap_2437_);
lean_inc_ref(v_fileName_2436_);
v___x_2456_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2456_, 0, v_fileName_2436_);
lean_ctor_set(v___x_2456_, 1, v_fileMap_2437_);
lean_ctor_set(v___x_2456_, 2, v_options_2438_);
lean_ctor_set(v___x_2456_, 3, v_currRecDepth_2439_);
lean_ctor_set(v___x_2456_, 4, v_maxRecDepth_2440_);
lean_ctor_set(v___x_2456_, 5, v_ref_2455_);
lean_ctor_set(v___x_2456_, 6, v_currNamespace_2442_);
lean_ctor_set(v___x_2456_, 7, v_openDecls_2443_);
lean_ctor_set(v___x_2456_, 8, v_initHeartbeats_2444_);
lean_ctor_set(v___x_2456_, 9, v_maxHeartbeats_2445_);
lean_ctor_set(v___x_2456_, 10, v_quotContext_2446_);
lean_ctor_set(v___x_2456_, 11, v_currMacroScope_2447_);
lean_ctor_set(v___x_2456_, 12, v_cancelTk_x3f_2449_);
lean_ctor_set(v___x_2456_, 13, v_inheritedTraceOptions_2451_);
lean_ctor_set_uint8(v___x_2456_, sizeof(void*)*14, v_diag_2448_);
lean_ctor_set_uint8(v___x_2456_, sizeof(void*)*14 + 1, v_suppressElabErrors_2450_);
v___x_2457_ = l_Lean_PersistentArray_toArray___redArg(v_traces_2454_);
lean_dec_ref(v_traces_2454_);
v_sz_2458_ = lean_array_size(v___x_2457_);
v___x_2459_ = ((size_t)0ULL);
v___x_2460_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__4_spec__6(v_sz_2458_, v___x_2459_, v___x_2457_);
v_msg_2461_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_2461_, 0, v_data_2428_);
lean_ctor_set(v_msg_2461_, 1, v_msg_2430_);
lean_ctor_set(v_msg_2461_, 2, v___x_2460_);
v___x_2462_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_ExtractScript_lazyStepToStep_spec__0_spec__0(v_msg_2461_, v___y_2431_, v___y_2432_, v___x_2456_, v___y_2434_);
lean_dec_ref_known(v___x_2456_, 14);
v_a_2463_ = lean_ctor_get(v___x_2462_, 0);
v_isSharedCheck_2500_ = !lean_is_exclusive(v___x_2462_);
if (v_isSharedCheck_2500_ == 0)
{
v___x_2465_ = v___x_2462_;
v_isShared_2466_ = v_isSharedCheck_2500_;
goto v_resetjp_2464_;
}
else
{
lean_inc(v_a_2463_);
lean_dec(v___x_2462_);
v___x_2465_ = lean_box(0);
v_isShared_2466_ = v_isSharedCheck_2500_;
goto v_resetjp_2464_;
}
v_resetjp_2464_:
{
lean_object* v___x_2467_; lean_object* v_traceState_2468_; lean_object* v_env_2469_; lean_object* v_nextMacroScope_2470_; lean_object* v_ngen_2471_; lean_object* v_auxDeclNGen_2472_; lean_object* v_cache_2473_; lean_object* v_messages_2474_; lean_object* v_infoState_2475_; lean_object* v_snapshotTasks_2476_; lean_object* v___x_2478_; uint8_t v_isShared_2479_; uint8_t v_isSharedCheck_2499_; 
v___x_2467_ = lean_st_ref_take(v___y_2434_);
v_traceState_2468_ = lean_ctor_get(v___x_2467_, 4);
v_env_2469_ = lean_ctor_get(v___x_2467_, 0);
v_nextMacroScope_2470_ = lean_ctor_get(v___x_2467_, 1);
v_ngen_2471_ = lean_ctor_get(v___x_2467_, 2);
v_auxDeclNGen_2472_ = lean_ctor_get(v___x_2467_, 3);
v_cache_2473_ = lean_ctor_get(v___x_2467_, 5);
v_messages_2474_ = lean_ctor_get(v___x_2467_, 6);
v_infoState_2475_ = lean_ctor_get(v___x_2467_, 7);
v_snapshotTasks_2476_ = lean_ctor_get(v___x_2467_, 8);
v_isSharedCheck_2499_ = !lean_is_exclusive(v___x_2467_);
if (v_isSharedCheck_2499_ == 0)
{
v___x_2478_ = v___x_2467_;
v_isShared_2479_ = v_isSharedCheck_2499_;
goto v_resetjp_2477_;
}
else
{
lean_inc(v_snapshotTasks_2476_);
lean_inc(v_infoState_2475_);
lean_inc(v_messages_2474_);
lean_inc(v_cache_2473_);
lean_inc(v_traceState_2468_);
lean_inc(v_auxDeclNGen_2472_);
lean_inc(v_ngen_2471_);
lean_inc(v_nextMacroScope_2470_);
lean_inc(v_env_2469_);
lean_dec(v___x_2467_);
v___x_2478_ = lean_box(0);
v_isShared_2479_ = v_isSharedCheck_2499_;
goto v_resetjp_2477_;
}
v_resetjp_2477_:
{
uint64_t v_tid_2480_; lean_object* v___x_2482_; uint8_t v_isShared_2483_; uint8_t v_isSharedCheck_2497_; 
v_tid_2480_ = lean_ctor_get_uint64(v_traceState_2468_, sizeof(void*)*1);
v_isSharedCheck_2497_ = !lean_is_exclusive(v_traceState_2468_);
if (v_isSharedCheck_2497_ == 0)
{
lean_object* v_unused_2498_; 
v_unused_2498_ = lean_ctor_get(v_traceState_2468_, 0);
lean_dec(v_unused_2498_);
v___x_2482_ = v_traceState_2468_;
v_isShared_2483_ = v_isSharedCheck_2497_;
goto v_resetjp_2481_;
}
else
{
lean_dec(v_traceState_2468_);
v___x_2482_ = lean_box(0);
v_isShared_2483_ = v_isSharedCheck_2497_;
goto v_resetjp_2481_;
}
v_resetjp_2481_:
{
lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2487_; 
v___x_2484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2484_, 0, v_ref_2429_);
lean_ctor_set(v___x_2484_, 1, v_a_2463_);
v___x_2485_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_2427_, v___x_2484_);
if (v_isShared_2483_ == 0)
{
lean_ctor_set(v___x_2482_, 0, v___x_2485_);
v___x_2487_ = v___x_2482_;
goto v_reusejp_2486_;
}
else
{
lean_object* v_reuseFailAlloc_2496_; 
v_reuseFailAlloc_2496_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2496_, 0, v___x_2485_);
lean_ctor_set_uint64(v_reuseFailAlloc_2496_, sizeof(void*)*1, v_tid_2480_);
v___x_2487_ = v_reuseFailAlloc_2496_;
goto v_reusejp_2486_;
}
v_reusejp_2486_:
{
lean_object* v___x_2489_; 
if (v_isShared_2479_ == 0)
{
lean_ctor_set(v___x_2478_, 4, v___x_2487_);
v___x_2489_ = v___x_2478_;
goto v_reusejp_2488_;
}
else
{
lean_object* v_reuseFailAlloc_2495_; 
v_reuseFailAlloc_2495_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2495_, 0, v_env_2469_);
lean_ctor_set(v_reuseFailAlloc_2495_, 1, v_nextMacroScope_2470_);
lean_ctor_set(v_reuseFailAlloc_2495_, 2, v_ngen_2471_);
lean_ctor_set(v_reuseFailAlloc_2495_, 3, v_auxDeclNGen_2472_);
lean_ctor_set(v_reuseFailAlloc_2495_, 4, v___x_2487_);
lean_ctor_set(v_reuseFailAlloc_2495_, 5, v_cache_2473_);
lean_ctor_set(v_reuseFailAlloc_2495_, 6, v_messages_2474_);
lean_ctor_set(v_reuseFailAlloc_2495_, 7, v_infoState_2475_);
lean_ctor_set(v_reuseFailAlloc_2495_, 8, v_snapshotTasks_2476_);
v___x_2489_ = v_reuseFailAlloc_2495_;
goto v_reusejp_2488_;
}
v_reusejp_2488_:
{
lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2493_; 
v___x_2490_ = lean_st_ref_set(v___y_2434_, v___x_2489_);
v___x_2491_ = lean_box(0);
if (v_isShared_2466_ == 0)
{
lean_ctor_set(v___x_2465_, 0, v___x_2491_);
v___x_2493_ = v___x_2465_;
goto v_reusejp_2492_;
}
else
{
lean_object* v_reuseFailAlloc_2494_; 
v_reuseFailAlloc_2494_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2494_, 0, v___x_2491_);
v___x_2493_ = v_reuseFailAlloc_2494_;
goto v_reusejp_2492_;
}
v_reusejp_2492_:
{
return v___x_2493_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___redArg___boxed(lean_object* v_oldTraces_2501_, lean_object* v_data_2502_, lean_object* v_ref_2503_, lean_object* v_msg_2504_, lean_object* v___y_2505_, lean_object* v___y_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_){
_start:
{
lean_object* v_res_2510_; 
v_res_2510_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___redArg(v_oldTraces_2501_, v_data_2502_, v_ref_2503_, v_msg_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_);
lean_dec(v___y_2508_);
lean_dec_ref(v___y_2507_);
lean_dec(v___y_2506_);
lean_dec_ref(v___y_2505_);
return v_res_2510_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__3(lean_object* v_e_2511_){
_start:
{
if (lean_obj_tag(v_e_2511_) == 0)
{
uint8_t v___x_2512_; 
v___x_2512_ = 2;
return v___x_2512_;
}
else
{
uint8_t v___x_2513_; 
v___x_2513_ = 0;
return v___x_2513_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__3___boxed(lean_object* v_e_2514_){
_start:
{
uint8_t v_res_2515_; lean_object* v_r_2516_; 
v_res_2515_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__3(v_e_2514_);
lean_dec_ref(v_e_2514_);
v_r_2516_ = lean_box(v_res_2515_);
return v_r_2516_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1(lean_object* v_cls_2517_, uint8_t v_collapsed_2518_, lean_object* v_tag_2519_, lean_object* v_opts_2520_, uint8_t v_clsEnabled_2521_, lean_object* v_oldTraces_2522_, lean_object* v_msg_2523_, lean_object* v_resStartStop_2524_, lean_object* v___y_2525_, lean_object* v___y_2526_, lean_object* v___y_2527_, lean_object* v___y_2528_, lean_object* v___y_2529_, lean_object* v___y_2530_, lean_object* v___y_2531_){
_start:
{
lean_object* v_fst_2533_; lean_object* v_snd_2534_; lean_object* v___y_2536_; lean_object* v___y_2537_; lean_object* v_data_2538_; lean_object* v_fst_2549_; lean_object* v_snd_2550_; lean_object* v___x_2551_; uint8_t v___x_2552_; lean_object* v___y_2554_; lean_object* v_a_2555_; uint8_t v___y_2570_; double v___y_2601_; 
v_fst_2533_ = lean_ctor_get(v_resStartStop_2524_, 0);
lean_inc(v_fst_2533_);
v_snd_2534_ = lean_ctor_get(v_resStartStop_2524_, 1);
lean_inc(v_snd_2534_);
lean_dec_ref(v_resStartStop_2524_);
v_fst_2549_ = lean_ctor_get(v_snd_2534_, 0);
lean_inc(v_fst_2549_);
v_snd_2550_ = lean_ctor_get(v_snd_2534_, 1);
lean_inc(v_snd_2550_);
lean_dec(v_snd_2534_);
v___x_2551_ = l_Lean_trace_profiler;
v___x_2552_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_opts_2520_, v___x_2551_);
if (v___x_2552_ == 0)
{
v___y_2570_ = v___x_2552_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2606_; uint8_t v___x_2607_; 
v___x_2606_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2607_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_opts_2520_, v___x_2606_);
if (v___x_2607_ == 0)
{
lean_object* v___x_2608_; lean_object* v___x_2609_; double v___x_2610_; double v___x_2611_; double v___x_2612_; 
v___x_2608_ = l_Lean_trace_profiler_threshold;
v___x_2609_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7(v_opts_2520_, v___x_2608_);
v___x_2610_ = lean_float_of_nat(v___x_2609_);
v___x_2611_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__3);
v___x_2612_ = lean_float_div(v___x_2610_, v___x_2611_);
v___y_2601_ = v___x_2612_;
goto v___jp_2600_;
}
else
{
lean_object* v___x_2613_; lean_object* v___x_2614_; double v___x_2615_; 
v___x_2613_ = l_Lean_trace_profiler_threshold;
v___x_2614_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4_spec__7(v_opts_2520_, v___x_2613_);
v___x_2615_ = lean_float_of_nat(v___x_2614_);
v___y_2601_ = v___x_2615_;
goto v___jp_2600_;
}
}
v___jp_2535_:
{
lean_object* v___x_2539_; 
lean_inc(v___y_2536_);
v___x_2539_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___redArg(v_oldTraces_2522_, v_data_2538_, v___y_2536_, v___y_2537_, v___y_2528_, v___y_2529_, v___y_2530_, v___y_2531_);
if (lean_obj_tag(v___x_2539_) == 0)
{
lean_object* v___x_2540_; 
lean_dec_ref_known(v___x_2539_, 1);
v___x_2540_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___redArg(v_fst_2533_);
return v___x_2540_;
}
else
{
lean_object* v_a_2541_; lean_object* v___x_2543_; uint8_t v_isShared_2544_; uint8_t v_isSharedCheck_2548_; 
lean_dec(v_fst_2533_);
v_a_2541_ = lean_ctor_get(v___x_2539_, 0);
v_isSharedCheck_2548_ = !lean_is_exclusive(v___x_2539_);
if (v_isSharedCheck_2548_ == 0)
{
v___x_2543_ = v___x_2539_;
v_isShared_2544_ = v_isSharedCheck_2548_;
goto v_resetjp_2542_;
}
else
{
lean_inc(v_a_2541_);
lean_dec(v___x_2539_);
v___x_2543_ = lean_box(0);
v_isShared_2544_ = v_isSharedCheck_2548_;
goto v_resetjp_2542_;
}
v_resetjp_2542_:
{
lean_object* v___x_2546_; 
if (v_isShared_2544_ == 0)
{
v___x_2546_ = v___x_2543_;
goto v_reusejp_2545_;
}
else
{
lean_object* v_reuseFailAlloc_2547_; 
v_reuseFailAlloc_2547_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2547_, 0, v_a_2541_);
v___x_2546_ = v_reuseFailAlloc_2547_;
goto v_reusejp_2545_;
}
v_reusejp_2545_:
{
return v___x_2546_;
}
}
}
}
v___jp_2553_:
{
uint8_t v_result_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; double v___x_2559_; lean_object* v_data_2560_; 
v_result_2556_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__3(v_fst_2533_);
v___x_2557_ = lean_box(v_result_2556_);
v___x_2558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2558_, 0, v___x_2557_);
v___x_2559_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__0);
lean_inc_ref(v_tag_2519_);
lean_inc_ref(v___x_2558_);
lean_inc(v_cls_2517_);
v_data_2560_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2560_, 0, v_cls_2517_);
lean_ctor_set(v_data_2560_, 1, v___x_2558_);
lean_ctor_set(v_data_2560_, 2, v_tag_2519_);
lean_ctor_set_float(v_data_2560_, sizeof(void*)*3, v___x_2559_);
lean_ctor_set_float(v_data_2560_, sizeof(void*)*3 + 8, v___x_2559_);
lean_ctor_set_uint8(v_data_2560_, sizeof(void*)*3 + 16, v_collapsed_2518_);
if (v___x_2552_ == 0)
{
lean_dec_ref_known(v___x_2558_, 1);
lean_dec(v_snd_2550_);
lean_dec(v_fst_2549_);
lean_dec_ref(v_tag_2519_);
lean_dec(v_cls_2517_);
v___y_2536_ = v___y_2554_;
v___y_2537_ = v_a_2555_;
v_data_2538_ = v_data_2560_;
goto v___jp_2535_;
}
else
{
lean_object* v_data_2561_; double v___x_2562_; double v___x_2563_; 
lean_dec_ref_known(v_data_2560_, 3);
v_data_2561_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2561_, 0, v_cls_2517_);
lean_ctor_set(v_data_2561_, 1, v___x_2558_);
lean_ctor_set(v_data_2561_, 2, v_tag_2519_);
v___x_2562_ = lean_unbox_float(v_fst_2549_);
lean_dec(v_fst_2549_);
lean_ctor_set_float(v_data_2561_, sizeof(void*)*3, v___x_2562_);
v___x_2563_ = lean_unbox_float(v_snd_2550_);
lean_dec(v_snd_2550_);
lean_ctor_set_float(v_data_2561_, sizeof(void*)*3 + 8, v___x_2563_);
lean_ctor_set_uint8(v_data_2561_, sizeof(void*)*3 + 16, v_collapsed_2518_);
v___y_2536_ = v___y_2554_;
v___y_2537_ = v_a_2555_;
v_data_2538_ = v_data_2561_;
goto v___jp_2535_;
}
}
v___jp_2564_:
{
lean_object* v_ref_2565_; lean_object* v___x_2566_; 
v_ref_2565_ = lean_ctor_get(v___y_2530_, 5);
lean_inc(v___y_2531_);
lean_inc_ref(v___y_2530_);
lean_inc(v___y_2529_);
lean_inc_ref(v___y_2528_);
lean_inc(v___y_2527_);
lean_inc(v___y_2526_);
lean_inc_ref(v___y_2525_);
lean_inc(v_fst_2533_);
v___x_2566_ = lean_apply_9(v_msg_2523_, v_fst_2533_, v___y_2525_, v___y_2526_, v___y_2527_, v___y_2528_, v___y_2529_, v___y_2530_, v___y_2531_, lean_box(0));
if (lean_obj_tag(v___x_2566_) == 0)
{
lean_object* v_a_2567_; 
v_a_2567_ = lean_ctor_get(v___x_2566_, 0);
lean_inc(v_a_2567_);
lean_dec_ref_known(v___x_2566_, 1);
v___y_2554_ = v_ref_2565_;
v_a_2555_ = v_a_2567_;
goto v___jp_2553_;
}
else
{
lean_object* v___x_2568_; 
lean_dec_ref_known(v___x_2566_, 1);
v___x_2568_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ExtractScript_visitGoal_spec__4___closed__2);
v___y_2554_ = v_ref_2565_;
v_a_2555_ = v___x_2568_;
goto v___jp_2553_;
}
}
v___jp_2569_:
{
if (v_clsEnabled_2521_ == 0)
{
if (v___y_2570_ == 0)
{
lean_object* v___x_2571_; lean_object* v_traceState_2572_; lean_object* v_env_2573_; lean_object* v_nextMacroScope_2574_; lean_object* v_ngen_2575_; lean_object* v_auxDeclNGen_2576_; lean_object* v_cache_2577_; lean_object* v_messages_2578_; lean_object* v_infoState_2579_; lean_object* v_snapshotTasks_2580_; lean_object* v___x_2582_; uint8_t v_isShared_2583_; uint8_t v_isSharedCheck_2599_; 
lean_dec(v_snd_2550_);
lean_dec(v_fst_2549_);
lean_dec_ref(v_msg_2523_);
lean_dec_ref(v_tag_2519_);
lean_dec(v_cls_2517_);
v___x_2571_ = lean_st_ref_take(v___y_2531_);
v_traceState_2572_ = lean_ctor_get(v___x_2571_, 4);
v_env_2573_ = lean_ctor_get(v___x_2571_, 0);
v_nextMacroScope_2574_ = lean_ctor_get(v___x_2571_, 1);
v_ngen_2575_ = lean_ctor_get(v___x_2571_, 2);
v_auxDeclNGen_2576_ = lean_ctor_get(v___x_2571_, 3);
v_cache_2577_ = lean_ctor_get(v___x_2571_, 5);
v_messages_2578_ = lean_ctor_get(v___x_2571_, 6);
v_infoState_2579_ = lean_ctor_get(v___x_2571_, 7);
v_snapshotTasks_2580_ = lean_ctor_get(v___x_2571_, 8);
v_isSharedCheck_2599_ = !lean_is_exclusive(v___x_2571_);
if (v_isSharedCheck_2599_ == 0)
{
v___x_2582_ = v___x_2571_;
v_isShared_2583_ = v_isSharedCheck_2599_;
goto v_resetjp_2581_;
}
else
{
lean_inc(v_snapshotTasks_2580_);
lean_inc(v_infoState_2579_);
lean_inc(v_messages_2578_);
lean_inc(v_cache_2577_);
lean_inc(v_traceState_2572_);
lean_inc(v_auxDeclNGen_2576_);
lean_inc(v_ngen_2575_);
lean_inc(v_nextMacroScope_2574_);
lean_inc(v_env_2573_);
lean_dec(v___x_2571_);
v___x_2582_ = lean_box(0);
v_isShared_2583_ = v_isSharedCheck_2599_;
goto v_resetjp_2581_;
}
v_resetjp_2581_:
{
uint64_t v_tid_2584_; lean_object* v_traces_2585_; lean_object* v___x_2587_; uint8_t v_isShared_2588_; uint8_t v_isSharedCheck_2598_; 
v_tid_2584_ = lean_ctor_get_uint64(v_traceState_2572_, sizeof(void*)*1);
v_traces_2585_ = lean_ctor_get(v_traceState_2572_, 0);
v_isSharedCheck_2598_ = !lean_is_exclusive(v_traceState_2572_);
if (v_isSharedCheck_2598_ == 0)
{
v___x_2587_ = v_traceState_2572_;
v_isShared_2588_ = v_isSharedCheck_2598_;
goto v_resetjp_2586_;
}
else
{
lean_inc(v_traces_2585_);
lean_dec(v_traceState_2572_);
v___x_2587_ = lean_box(0);
v_isShared_2588_ = v_isSharedCheck_2598_;
goto v_resetjp_2586_;
}
v_resetjp_2586_:
{
lean_object* v___x_2589_; lean_object* v___x_2591_; 
v___x_2589_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2522_, v_traces_2585_);
lean_dec_ref(v_traces_2585_);
if (v_isShared_2588_ == 0)
{
lean_ctor_set(v___x_2587_, 0, v___x_2589_);
v___x_2591_ = v___x_2587_;
goto v_reusejp_2590_;
}
else
{
lean_object* v_reuseFailAlloc_2597_; 
v_reuseFailAlloc_2597_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2597_, 0, v___x_2589_);
lean_ctor_set_uint64(v_reuseFailAlloc_2597_, sizeof(void*)*1, v_tid_2584_);
v___x_2591_ = v_reuseFailAlloc_2597_;
goto v_reusejp_2590_;
}
v_reusejp_2590_:
{
lean_object* v___x_2593_; 
if (v_isShared_2583_ == 0)
{
lean_ctor_set(v___x_2582_, 4, v___x_2591_);
v___x_2593_ = v___x_2582_;
goto v_reusejp_2592_;
}
else
{
lean_object* v_reuseFailAlloc_2596_; 
v_reuseFailAlloc_2596_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2596_, 0, v_env_2573_);
lean_ctor_set(v_reuseFailAlloc_2596_, 1, v_nextMacroScope_2574_);
lean_ctor_set(v_reuseFailAlloc_2596_, 2, v_ngen_2575_);
lean_ctor_set(v_reuseFailAlloc_2596_, 3, v_auxDeclNGen_2576_);
lean_ctor_set(v_reuseFailAlloc_2596_, 4, v___x_2591_);
lean_ctor_set(v_reuseFailAlloc_2596_, 5, v_cache_2577_);
lean_ctor_set(v_reuseFailAlloc_2596_, 6, v_messages_2578_);
lean_ctor_set(v_reuseFailAlloc_2596_, 7, v_infoState_2579_);
lean_ctor_set(v_reuseFailAlloc_2596_, 8, v_snapshotTasks_2580_);
v___x_2593_ = v_reuseFailAlloc_2596_;
goto v_reusejp_2592_;
}
v_reusejp_2592_:
{
lean_object* v___x_2594_; lean_object* v___x_2595_; 
v___x_2594_ = lean_st_ref_set(v___y_2531_, v___x_2593_);
v___x_2595_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___redArg(v_fst_2533_);
return v___x_2595_;
}
}
}
}
}
else
{
goto v___jp_2564_;
}
}
else
{
goto v___jp_2564_;
}
}
v___jp_2600_:
{
double v___x_2602_; double v___x_2603_; double v___x_2604_; uint8_t v___x_2605_; 
v___x_2602_ = lean_unbox_float(v_snd_2550_);
v___x_2603_ = lean_unbox_float(v_fst_2549_);
v___x_2604_ = lean_float_sub(v___x_2602_, v___x_2603_);
v___x_2605_ = lean_float_decLt(v___y_2601_, v___x_2604_);
v___y_2570_ = v___x_2605_;
goto v___jp_2569_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1___boxed(lean_object* v_cls_2616_, lean_object* v_collapsed_2617_, lean_object* v_tag_2618_, lean_object* v_opts_2619_, lean_object* v_clsEnabled_2620_, lean_object* v_oldTraces_2621_, lean_object* v_msg_2622_, lean_object* v_resStartStop_2623_, lean_object* v___y_2624_, lean_object* v___y_2625_, lean_object* v___y_2626_, lean_object* v___y_2627_, lean_object* v___y_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_){
_start:
{
uint8_t v_collapsed_boxed_2632_; uint8_t v_clsEnabled_boxed_2633_; lean_object* v_res_2634_; 
v_collapsed_boxed_2632_ = lean_unbox(v_collapsed_2617_);
v_clsEnabled_boxed_2633_ = lean_unbox(v_clsEnabled_2620_);
v_res_2634_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1(v_cls_2616_, v_collapsed_boxed_2632_, v_tag_2618_, v_opts_2619_, v_clsEnabled_boxed_2633_, v_oldTraces_2621_, v_msg_2622_, v_resStartStop_2623_, v___y_2624_, v___y_2625_, v___y_2626_, v___y_2627_, v___y_2628_, v___y_2629_, v___y_2630_);
lean_dec(v___y_2630_);
lean_dec_ref(v___y_2629_);
lean_dec(v___y_2628_);
lean_dec_ref(v___y_2627_);
lean_dec(v___y_2626_);
lean_dec(v___y_2625_);
lean_dec_ref(v___y_2624_);
lean_dec_ref(v_opts_2619_);
return v_res_2634_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefixScript(lean_object* v_a_2636_, lean_object* v_a_2637_, lean_object* v_a_2638_, lean_object* v_a_2639_, lean_object* v_a_2640_, lean_object* v_a_2641_, lean_object* v_a_2642_){
_start:
{
lean_object* v_____do__lift_2645_; lean_object* v___y_2646_; lean_object* v___y_2647_; lean_object* v___y_2648_; lean_object* v___y_2649_; lean_object* v___y_2650_; lean_object* v___y_2651_; lean_object* v___y_2652_; lean_object* v_options_2655_; uint8_t v_hasTrace_2656_; 
v_options_2655_ = lean_ctor_get(v_a_2641_, 2);
v_hasTrace_2656_ = lean_ctor_get_uint8(v_options_2655_, sizeof(void*)*1);
if (v_hasTrace_2656_ == 0)
{
lean_object* v___x_2657_; 
v___x_2657_ = lp_aesop_Aesop_getRootGoal(v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_);
if (lean_obj_tag(v___x_2657_) == 0)
{
lean_object* v_a_2658_; 
v_a_2658_ = lean_ctor_get(v___x_2657_, 0);
lean_inc(v_a_2658_);
lean_dec_ref_known(v___x_2657_, 1);
v_____do__lift_2645_ = v_a_2658_;
v___y_2646_ = v_a_2636_;
v___y_2647_ = v_a_2637_;
v___y_2648_ = v_a_2638_;
v___y_2649_ = v_a_2639_;
v___y_2650_ = v_a_2640_;
v___y_2651_ = v_a_2641_;
v___y_2652_ = v_a_2642_;
goto v___jp_2644_;
}
else
{
lean_object* v_a_2659_; lean_object* v___x_2661_; uint8_t v_isShared_2662_; uint8_t v_isSharedCheck_2666_; 
v_a_2659_ = lean_ctor_get(v___x_2657_, 0);
v_isSharedCheck_2666_ = !lean_is_exclusive(v___x_2657_);
if (v_isSharedCheck_2666_ == 0)
{
v___x_2661_ = v___x_2657_;
v_isShared_2662_ = v_isSharedCheck_2666_;
goto v_resetjp_2660_;
}
else
{
lean_inc(v_a_2659_);
lean_dec(v___x_2657_);
v___x_2661_ = lean_box(0);
v_isShared_2662_ = v_isSharedCheck_2666_;
goto v_resetjp_2660_;
}
v_resetjp_2660_:
{
lean_object* v___x_2664_; 
if (v_isShared_2662_ == 0)
{
v___x_2664_ = v___x_2661_;
goto v_reusejp_2663_;
}
else
{
lean_object* v_reuseFailAlloc_2665_; 
v_reuseFailAlloc_2665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2665_, 0, v_a_2659_);
v___x_2664_ = v_reuseFailAlloc_2665_;
goto v_reusejp_2663_;
}
v_reusejp_2663_:
{
return v___x_2664_;
}
}
}
}
else
{
lean_object* v_inheritedTraceOptions_2667_; lean_object* v___x_2668_; lean_object* v_traceClass_2669_; lean_object* v___f_2670_; lean_object* v___x_2671_; lean_object* v___x_2672_; lean_object* v___x_2673_; uint8_t v___x_2674_; lean_object* v___y_2676_; lean_object* v___y_2677_; lean_object* v_a_2678_; lean_object* v___y_2691_; lean_object* v___y_2692_; lean_object* v_a_2693_; lean_object* v___y_2696_; lean_object* v___y_2697_; lean_object* v_a_2698_; lean_object* v___y_2708_; lean_object* v___y_2709_; lean_object* v_a_2710_; 
v_inheritedTraceOptions_2667_ = lean_ctor_get(v_a_2641_, 13);
v___x_2668_ = lp_aesop_Aesop_TraceOption_script;
v_traceClass_2669_ = lean_ctor_get(v___x_2668_, 0);
v___f_2670_ = ((lean_object*)(lp_aesop_Aesop_extractSafePrefixScript___closed__0));
v___x_2671_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__3));
v___x_2672_ = ((lean_object*)(lp_aesop_Aesop_ExtractScript_visitGoal___closed__5));
lean_inc(v_traceClass_2669_);
v___x_2673_ = l_Lean_Name_append(v___x_2672_, v_traceClass_2669_);
v___x_2674_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2667_, v_options_2655_, v___x_2673_);
lean_dec(v___x_2673_);
if (v___x_2674_ == 0)
{
lean_object* v___x_2747_; uint8_t v___x_2748_; 
v___x_2747_ = l_Lean_trace_profiler;
v___x_2748_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_options_2655_, v___x_2747_);
if (v___x_2748_ == 0)
{
lean_object* v___x_2749_; 
v___x_2749_ = lp_aesop_Aesop_getRootGoal(v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_);
if (lean_obj_tag(v___x_2749_) == 0)
{
lean_object* v_a_2750_; 
v_a_2750_ = lean_ctor_get(v___x_2749_, 0);
lean_inc(v_a_2750_);
lean_dec_ref_known(v___x_2749_, 1);
v_____do__lift_2645_ = v_a_2750_;
v___y_2646_ = v_a_2636_;
v___y_2647_ = v_a_2637_;
v___y_2648_ = v_a_2638_;
v___y_2649_ = v_a_2639_;
v___y_2650_ = v_a_2640_;
v___y_2651_ = v_a_2641_;
v___y_2652_ = v_a_2642_;
goto v___jp_2644_;
}
else
{
lean_object* v_a_2751_; lean_object* v___x_2753_; uint8_t v_isShared_2754_; uint8_t v_isSharedCheck_2758_; 
v_a_2751_ = lean_ctor_get(v___x_2749_, 0);
v_isSharedCheck_2758_ = !lean_is_exclusive(v___x_2749_);
if (v_isSharedCheck_2758_ == 0)
{
v___x_2753_ = v___x_2749_;
v_isShared_2754_ = v_isSharedCheck_2758_;
goto v_resetjp_2752_;
}
else
{
lean_inc(v_a_2751_);
lean_dec(v___x_2749_);
v___x_2753_ = lean_box(0);
v_isShared_2754_ = v_isSharedCheck_2758_;
goto v_resetjp_2752_;
}
v_resetjp_2752_:
{
lean_object* v___x_2756_; 
if (v_isShared_2754_ == 0)
{
v___x_2756_ = v___x_2753_;
goto v_reusejp_2755_;
}
else
{
lean_object* v_reuseFailAlloc_2757_; 
v_reuseFailAlloc_2757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2757_, 0, v_a_2751_);
v___x_2756_ = v_reuseFailAlloc_2757_;
goto v_reusejp_2755_;
}
v_reusejp_2755_:
{
return v___x_2756_;
}
}
}
}
else
{
goto v___jp_2712_;
}
}
else
{
goto v___jp_2712_;
}
v___jp_2675_:
{
lean_object* v___x_2679_; double v___x_2680_; double v___x_2681_; double v___x_2682_; double v___x_2683_; double v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; 
v___x_2679_ = lean_io_mono_nanos_now();
v___x_2680_ = lean_float_of_nat(v___y_2676_);
v___x_2681_ = lean_float_once(&lp_aesop_Aesop_ExtractScript_visitGoal___closed__2, &lp_aesop_Aesop_ExtractScript_visitGoal___closed__2_once, _init_lp_aesop_Aesop_ExtractScript_visitGoal___closed__2);
v___x_2682_ = lean_float_div(v___x_2680_, v___x_2681_);
v___x_2683_ = lean_float_of_nat(v___x_2679_);
v___x_2684_ = lean_float_div(v___x_2683_, v___x_2681_);
v___x_2685_ = lean_box_float(v___x_2682_);
v___x_2686_ = lean_box_float(v___x_2684_);
v___x_2687_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2687_, 0, v___x_2685_);
lean_ctor_set(v___x_2687_, 1, v___x_2686_);
v___x_2688_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2688_, 0, v_a_2678_);
lean_ctor_set(v___x_2688_, 1, v___x_2687_);
lean_inc(v_traceClass_2669_);
v___x_2689_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1(v_traceClass_2669_, v_hasTrace_2656_, v___x_2671_, v_options_2655_, v___x_2674_, v___y_2677_, v___f_2670_, v___x_2688_, v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_);
return v___x_2689_;
}
v___jp_2690_:
{
lean_object* v___x_2694_; 
v___x_2694_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2694_, 0, v_a_2693_);
v___y_2676_ = v___y_2691_;
v___y_2677_ = v___y_2692_;
v_a_2678_ = v___x_2694_;
goto v___jp_2675_;
}
v___jp_2695_:
{
lean_object* v___x_2699_; double v___x_2700_; double v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; 
v___x_2699_ = lean_io_get_num_heartbeats();
v___x_2700_ = lean_float_of_nat(v___y_2696_);
v___x_2701_ = lean_float_of_nat(v___x_2699_);
v___x_2702_ = lean_box_float(v___x_2700_);
v___x_2703_ = lean_box_float(v___x_2701_);
v___x_2704_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2704_, 0, v___x_2702_);
lean_ctor_set(v___x_2704_, 1, v___x_2703_);
v___x_2705_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2705_, 0, v_a_2698_);
lean_ctor_set(v___x_2705_, 1, v___x_2704_);
lean_inc(v_traceClass_2669_);
v___x_2706_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1(v_traceClass_2669_, v_hasTrace_2656_, v___x_2671_, v_options_2655_, v___x_2674_, v___y_2697_, v___f_2670_, v___x_2705_, v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_);
return v___x_2706_;
}
v___jp_2707_:
{
lean_object* v___x_2711_; 
v___x_2711_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2711_, 0, v_a_2710_);
v___y_2696_ = v___y_2708_;
v___y_2697_ = v___y_2709_;
v_a_2698_ = v___x_2711_;
goto v___jp_2695_;
}
v___jp_2712_:
{
lean_object* v___x_2713_; lean_object* v_a_2714_; lean_object* v___x_2715_; uint8_t v___x_2716_; 
v___x_2713_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_extractSafePrefixScript_spec__0___redArg(v_a_2642_);
v_a_2714_ = lean_ctor_get(v___x_2713_, 0);
lean_inc(v_a_2714_);
lean_dec_ref(v___x_2713_);
v___x_2715_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2716_ = lp_aesop_Lean_Option_get___at___00Aesop_ExtractScript_visitGoal_spec__3(v_options_2655_, v___x_2715_);
if (v___x_2716_ == 0)
{
lean_object* v___x_2717_; lean_object* v___x_2718_; 
v___x_2717_ = lean_io_mono_nanos_now();
v___x_2718_ = lp_aesop_Aesop_getRootGoal(v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_);
if (lean_obj_tag(v___x_2718_) == 0)
{
lean_object* v_a_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; 
v_a_2719_ = lean_ctor_get(v___x_2718_, 0);
lean_inc(v_a_2719_);
lean_dec_ref_known(v___x_2718_, 1);
v___x_2720_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___boxed), 10, 1);
lean_closure_set(v___x_2720_, 0, v_a_2719_);
v___x_2721_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_2720_, v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_);
if (lean_obj_tag(v___x_2721_) == 0)
{
lean_object* v_a_2722_; lean_object* v___x_2724_; uint8_t v_isShared_2725_; uint8_t v_isSharedCheck_2729_; 
v_a_2722_ = lean_ctor_get(v___x_2721_, 0);
v_isSharedCheck_2729_ = !lean_is_exclusive(v___x_2721_);
if (v_isSharedCheck_2729_ == 0)
{
v___x_2724_ = v___x_2721_;
v_isShared_2725_ = v_isSharedCheck_2729_;
goto v_resetjp_2723_;
}
else
{
lean_inc(v_a_2722_);
lean_dec(v___x_2721_);
v___x_2724_ = lean_box(0);
v_isShared_2725_ = v_isSharedCheck_2729_;
goto v_resetjp_2723_;
}
v_resetjp_2723_:
{
lean_object* v___x_2727_; 
if (v_isShared_2725_ == 0)
{
lean_ctor_set_tag(v___x_2724_, 1);
v___x_2727_ = v___x_2724_;
goto v_reusejp_2726_;
}
else
{
lean_object* v_reuseFailAlloc_2728_; 
v_reuseFailAlloc_2728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2728_, 0, v_a_2722_);
v___x_2727_ = v_reuseFailAlloc_2728_;
goto v_reusejp_2726_;
}
v_reusejp_2726_:
{
v___y_2676_ = v___x_2717_;
v___y_2677_ = v_a_2714_;
v_a_2678_ = v___x_2727_;
goto v___jp_2675_;
}
}
}
else
{
lean_object* v_a_2730_; 
v_a_2730_ = lean_ctor_get(v___x_2721_, 0);
lean_inc(v_a_2730_);
lean_dec_ref_known(v___x_2721_, 1);
v___y_2691_ = v___x_2717_;
v___y_2692_ = v_a_2714_;
v_a_2693_ = v_a_2730_;
goto v___jp_2690_;
}
}
else
{
lean_object* v_a_2731_; 
v_a_2731_ = lean_ctor_get(v___x_2718_, 0);
lean_inc(v_a_2731_);
lean_dec_ref_known(v___x_2718_, 1);
v___y_2691_ = v___x_2717_;
v___y_2692_ = v_a_2714_;
v_a_2693_ = v_a_2731_;
goto v___jp_2690_;
}
}
else
{
lean_object* v___x_2732_; lean_object* v___x_2733_; 
v___x_2732_ = lean_io_get_num_heartbeats();
v___x_2733_ = lp_aesop_Aesop_getRootGoal(v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_);
if (lean_obj_tag(v___x_2733_) == 0)
{
lean_object* v_a_2734_; lean_object* v___x_2735_; lean_object* v___x_2736_; 
v_a_2734_ = lean_ctor_get(v___x_2733_, 0);
lean_inc(v_a_2734_);
lean_dec_ref_known(v___x_2733_, 1);
v___x_2735_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___boxed), 10, 1);
lean_closure_set(v___x_2735_, 0, v_a_2734_);
v___x_2736_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_2735_, v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_);
if (lean_obj_tag(v___x_2736_) == 0)
{
lean_object* v_a_2737_; lean_object* v___x_2739_; uint8_t v_isShared_2740_; uint8_t v_isSharedCheck_2744_; 
v_a_2737_ = lean_ctor_get(v___x_2736_, 0);
v_isSharedCheck_2744_ = !lean_is_exclusive(v___x_2736_);
if (v_isSharedCheck_2744_ == 0)
{
v___x_2739_ = v___x_2736_;
v_isShared_2740_ = v_isSharedCheck_2744_;
goto v_resetjp_2738_;
}
else
{
lean_inc(v_a_2737_);
lean_dec(v___x_2736_);
v___x_2739_ = lean_box(0);
v_isShared_2740_ = v_isSharedCheck_2744_;
goto v_resetjp_2738_;
}
v_resetjp_2738_:
{
lean_object* v___x_2742_; 
if (v_isShared_2740_ == 0)
{
lean_ctor_set_tag(v___x_2739_, 1);
v___x_2742_ = v___x_2739_;
goto v_reusejp_2741_;
}
else
{
lean_object* v_reuseFailAlloc_2743_; 
v_reuseFailAlloc_2743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2743_, 0, v_a_2737_);
v___x_2742_ = v_reuseFailAlloc_2743_;
goto v_reusejp_2741_;
}
v_reusejp_2741_:
{
v___y_2696_ = v___x_2732_;
v___y_2697_ = v_a_2714_;
v_a_2698_ = v___x_2742_;
goto v___jp_2695_;
}
}
}
else
{
lean_object* v_a_2745_; 
v_a_2745_ = lean_ctor_get(v___x_2736_, 0);
lean_inc(v_a_2745_);
lean_dec_ref_known(v___x_2736_, 1);
v___y_2708_ = v___x_2732_;
v___y_2709_ = v_a_2714_;
v_a_2710_ = v_a_2745_;
goto v___jp_2707_;
}
}
else
{
lean_object* v_a_2746_; 
v_a_2746_ = lean_ctor_get(v___x_2733_, 0);
lean_inc(v_a_2746_);
lean_dec_ref_known(v___x_2733_, 1);
v___y_2708_ = v___x_2732_;
v___y_2709_ = v_a_2714_;
v_a_2710_ = v_a_2746_;
goto v___jp_2707_;
}
}
}
}
v___jp_2644_:
{
lean_object* v___x_2653_; lean_object* v___x_2654_; 
v___x_2653_ = lean_alloc_closure((void*)(lp_aesop_Aesop_GoalRef_extractSafePrefixScriptCore___boxed), 10, 1);
lean_closure_set(v___x_2653_, 0, v_____do__lift_2645_);
v___x_2654_ = lp_aesop_Aesop_ExtractScriptM_run___redArg(v___x_2653_, v___y_2646_, v___y_2647_, v___y_2648_, v___y_2649_, v___y_2650_, v___y_2651_, v___y_2652_);
return v___x_2654_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefixScript___boxed(lean_object* v_a_2759_, lean_object* v_a_2760_, lean_object* v_a_2761_, lean_object* v_a_2762_, lean_object* v_a_2763_, lean_object* v_a_2764_, lean_object* v_a_2765_, lean_object* v_a_2766_){
_start:
{
lean_object* v_res_2767_; 
v_res_2767_ = lp_aesop_Aesop_extractSafePrefixScript(v_a_2759_, v_a_2760_, v_a_2761_, v_a_2762_, v_a_2763_, v_a_2764_, v_a_2765_);
lean_dec(v_a_2765_);
lean_dec_ref(v_a_2764_);
lean_dec(v_a_2763_);
lean_dec_ref(v_a_2762_);
lean_dec(v_a_2761_);
lean_dec(v_a_2760_);
lean_dec_ref(v_a_2759_);
return v_res_2767_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2(lean_object* v_00_u03b1_2768_, lean_object* v_x_2769_, lean_object* v___y_2770_, lean_object* v___y_2771_, lean_object* v___y_2772_, lean_object* v___y_2773_, lean_object* v___y_2774_, lean_object* v___y_2775_, lean_object* v___y_2776_){
_start:
{
lean_object* v___x_2778_; 
v___x_2778_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___redArg(v_x_2769_);
return v___x_2778_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2___boxed(lean_object* v_00_u03b1_2779_, lean_object* v_x_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_, lean_object* v___y_2785_, lean_object* v___y_2786_, lean_object* v___y_2787_, lean_object* v___y_2788_){
_start:
{
lean_object* v_res_2789_; 
v_res_2789_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__2(v_00_u03b1_2779_, v_x_2780_, v___y_2781_, v___y_2782_, v___y_2783_, v___y_2784_, v___y_2785_, v___y_2786_, v___y_2787_);
lean_dec(v___y_2787_);
lean_dec_ref(v___y_2786_);
lean_dec(v___y_2785_);
lean_dec_ref(v___y_2784_);
lean_dec(v___y_2783_);
lean_dec(v___y_2782_);
lean_dec_ref(v___y_2781_);
return v_res_2789_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1(lean_object* v_oldTraces_2790_, lean_object* v_data_2791_, lean_object* v_ref_2792_, lean_object* v_msg_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_){
_start:
{
lean_object* v___x_2802_; 
v___x_2802_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___redArg(v_oldTraces_2790_, v_data_2791_, v_ref_2792_, v_msg_2793_, v___y_2797_, v___y_2798_, v___y_2799_, v___y_2800_);
return v___x_2802_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1___boxed(lean_object* v_oldTraces_2803_, lean_object* v_data_2804_, lean_object* v_ref_2805_, lean_object* v_msg_2806_, lean_object* v___y_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_, lean_object* v___y_2810_, lean_object* v___y_2811_, lean_object* v___y_2812_, lean_object* v___y_2813_, lean_object* v___y_2814_){
_start:
{
lean_object* v_res_2815_; 
v_res_2815_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_extractSafePrefixScript_spec__1_spec__1(v_oldTraces_2803_, v_data_2804_, v_ref_2805_, v_msg_2806_, v___y_2807_, v___y_2808_, v___y_2809_, v___y_2810_, v___y_2811_, v___y_2812_, v___y_2813_);
lean_dec(v___y_2813_);
lean_dec_ref(v___y_2812_);
lean_dec(v___y_2811_);
lean_dec_ref(v___y_2810_);
lean_dec(v___y_2809_);
lean_dec(v___y_2808_);
lean_dec_ref(v___y_2807_);
return v_res_2815_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_UScript(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_ExtractScript(uint8_t builtin) {
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
res = runtime_initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_ExtractScript(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_ExtractScript(uint8_t builtin) {
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
res = initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_ExtractScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_ExtractScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_ExtractScript(builtin);
}
#ifdef __cplusplus
}
#endif
